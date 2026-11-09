#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Exceptions;
using Gym.Observations;
using Gym.Rendering;
using Gym.Spaces;
using Gym.Utils;
using NumSharp;
using Color = Gym.Rendering.Color;

namespace Gym.Environments.Envs.ToyText {
    /// <summary>
    ///     Taxi-v4: drive a taxi on a 5x5 grid with walls, pick a passenger up at one of four stands (R, G, Y, B) and drop them
    ///     at their destination: -1 per step, -10 for a wrong pickup or dropoff, 20 for delivering. Byte for byte what
    ///     Gymnasium's Taxi computes: the 500-state transition table <see cref="P"/> (dry or rainy), the fickle passenger,
    ///     the generator draws, observations, rewards, <c>{"prob", "action_mask"}</c> infos, <c>ansi</c> text and
    ///     <c>rgb_array</c> pixels.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Like Gymnasium's, the env checks no action against its space: <c>self.P[self.s][a]</c> is a dict lookup, so any
    ///     number equal to 0..5 is an action, anything else raises <c>KeyError</c>. Each step draws one float64 from the env's
    ///     generator; a fickle passenger draws once more at the reset (<c>random()</c>) and once when they change their mind
    ///     (<c>choice</c>). Observations are 0-d int64 arrays (DR-08); rewards are Gymnasium's -1, -10 and 20 as float64; the
    ///     info's <c>action_mask</c> is Gymnasium's int8 array.
    ///     </para>
    ///     <para>
    ///     Rendering is Gymnasium's pixel art (the background with walls and stands, the house, the destination marker, the
    ///     passenger and the taxi facing its last move); as in Gymnasium the taxi turns only when a render sees a movement as
    ///     the last action, and the sprite is chosen with Python's list indexing by that action. <c>ansi</c> colours the taxi,
    ///     passenger and destination with Gymnasium's escape codes. Human mode renders into the window from
    ///     <c>viewer_factory</c> or <c>Gymnasium.HumanViewerFactory</c> (DR-20) at 4 frames per second. Not thread-safe.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/taxi.py#L35">gymnasium/envs/toy_text/taxi.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class TaxiEnv : Env<NDArray, NDArray> {
        /// <summary>Gymnasium's <c>MAP</c>: the grid, its walls (<c>|</c>) and stands, as text.</summary>
        public static IReadOnlyList<string> Map { get; } = new[] {
            "+---------+",
            "|R: | : :G|",
            "| : | : : |",
            "| : : : : |",
            "| | : | : |",
            "|Y| : |B: |",
            "+---------+",
        };

        /// <summary>Gymnasium's <c>WINDOW_SIZE</c>: the frame, 536x456 pixels.</summary>
        private static readonly (int Width, int Height) WindowSize = (536, 456);

        /// <summary>Gymnasium's <c>CANVAS_COLOR</c>.</summary>
        private static readonly Color CanvasColor = new(250, 250, 246);

        /// <summary>Gymnasium's <c>BACKGROUND_POS</c>, <c>HOUSE_POS</c>, <c>GRID_ORIGIN</c> and sprite offsets.</summary>
        private static readonly (int X, int Y) BackgroundPos = (0, 80), HousePos = (403, 36), GridOrigin = (48, 128), PassengerOffset = (35, 19),
            TaxiHOffset = (15, 9), TaxiVOffset = (25, 0);

        /// <summary>The <c>ansi</c> caption of each action.</summary>
        private static readonly string[] ActionNames = { "South", "North", "East", "West", "Pickup", "Dropoff" };

        /// <summary>The human-mode window, opened on the first human render.</summary>
        private readonly HumanWindow _window;

        /// <summary>The map's characters, row by row.</summary>
        private readonly char[,] _cells;

        /// <summary>The transition table, by state then action.</summary>
        private readonly Dictionary<long, IReadOnlyDictionary<long, IReadOnlyList<ToyTextTransition>>> _p;

        /// <summary>The start distribution: uniform over the states with a waiting passenger not at their destination.</summary>
        private readonly double[] _initialStateDistrib;

        /// <summary>Python's <c>self.s</c>, meaningful only while <see cref="_hasState"/>.</summary>
        private long _s;

        /// <summary>Whether Python's <c>self.s</c> exists.</summary>
        private bool _hasState;

        /// <summary>
        ///     Python's <c>self.lastaction</c> for the renders: its dict key (<see langword="null"/> for <c>None</c>; the key
        ///     also decides <c>lastaction in [0, 1, 2, 3]</c>, an equality test like the lookup), and its list index or the
        ///     error Python's list indexing raises for it, deferred.
        /// </summary>
        private (long Key, Func<int> ListIndex)? _lastAction;

        /// <summary>
        ///     Python's <c>self.taxi_orientation</c>: 0 after a reset, else the last movement a render saw, as its value and a
        ///     deferred list index (a float action keeps the orientation equal to a movement, yet makes every later sprite
        ///     lookup raise until a reset or an integer movement replaces it, as in Python).
        /// </summary>
        private (long Value, Func<int> ListIndex) _taxiOrientation = (0, () => 0);

        /// <summary>The action as given to the untyped step, kept for the step it feeds (its Python type decides the list indexing).</summary>
        private object? _givenAction;

        /// <summary>The array the untyped step made of <see cref="_givenAction"/>; only compared by reference.</summary>
        [NDBorrowed]
        private NDArray? _convertedAction;

        /// <summary>Creates the env.</summary>
        /// <param name="renderMode">Gymnasium's <c>render_mode</c>: <see langword="null"/>, <c>"human"</c>, <c>"ansi"</c> or <c>"rgb_array"</c>.</param>
        /// <param name="isRainy">Gymnasium's <c>is_rainy</c>: a move goes as intended with <paramref name="rainyProbability"/>, else to either side.</param>
        /// <param name="ficklePassenger">Gymnasium's <c>fickle_passenger</c>: the passenger may change their destination after the first move with them aboard.</param>
        /// <param name="rainyProbability">Gymnasium's <c>rainy_probability</c>, 0.8 by default.</param>
        /// <param name="fickleProbability">Gymnasium's <c>fickle_probability</c>, 0.3 by default: the chance, drawn at each reset, that the passenger is fickle this episode.</param>
        /// <param name="viewerFactory">Gym.NET's <c>viewer_factory</c> (DR-20) for human mode; <see langword="null"/> uses <c>Gymnasium.HumanViewerFactory</c>.</param>
        public TaxiEnv(string? renderMode = null, bool isRainy = false, bool ficklePassenger = false, double rainyProbability = 0.8, double fickleProbability = 0.3,
            IEnvironmentViewerFactoryDelegate? viewerFactory = null) {
            Metadata = ClassMetadata;
            _cells = new char[Map.Count, Map[0].Length];
            for (int row = 0; row < Map.Count; row++)
                for (int column = 0; column < Map[row].Length; column++)
                    _cells[row, column] = Map[row][column];

            RainyProbability = rainyProbability;
            RainyLateralProbability = (1.0 - rainyProbability) / 2.0;
            FickleProbability = fickleProbability;

            const int numStates = 500, numRows = 5, numColumns = 5, numActions = 6;
            MaxRow = numRows - 1;
            MaxCol = numColumns - 1;
            var distribution = new double[numStates];
            _p = new Dictionary<long, IReadOnlyDictionary<long, IReadOnlyList<ToyTextTransition>>>(numStates);
            var lists = new Dictionary<long, List<ToyTextTransition>[]>(numStates);
            for (long state = 0; state < numStates; state++) {
                var byAction = new List<ToyTextTransition>[numActions];
                for (int a = 0; a < numActions; a++)
                    byAction[a] = new List<ToyTextTransition>();
                lists[state] = byAction;
            }

            for (int row = 0; row < numRows; row++) {
                for (int col = 0; col < numColumns; col++) {
                    for (int passIdx = 0; passIdx < Locs.Count + 1; passIdx++) { // +1 for being inside taxi
                        for (int destIdx = 0; destIdx < Locs.Count; destIdx++) {
                            long state = Encode(row, col, passIdx, destIdx);
                            if (passIdx < 4 && passIdx != destIdx)
                                distribution[state] += 1;
                            for (int action = 0; action < numActions; action++) {
                                if (isRainy)
                                    BuildRainyTransitions(lists, row, col, passIdx, destIdx, action);
                                else
                                    BuildDryTransitions(lists, row, col, passIdx, destIdx, action);
                            }
                        }
                    }
                }
            }

            foreach ((long state, List<ToyTextTransition>[] byAction) in lists) {
                var dict = new Dictionary<long, IReadOnlyList<ToyTextTransition>>(numActions);
                for (int a = 0; a < numActions; a++)
                    dict[a] = byAction[a];
                _p[state] = dict;
            }

            // self.initial_state_distrib /= self.initial_state_distrib.sum(): NumPy's pairwise sum, exact here (300 ones).
            double total = distribution.Sum();
            for (int state = 0; state < numStates; state++)
                distribution[state] /= total;
            _initialStateDistrib = distribution;

            ActionSpace = new Discrete(numActions);
            ObservationSpace = new Discrete(numStates);
            RenderMode = renderMode;
            FicklePassenger = ficklePassenger;
            _window = new HumanWindow("Taxi", viewerFactory);
        }

        /// <summary>
        ///     Gymnasium's class attribute <c>metadata</c>: <c>{"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 4}</c>.
        ///     Every env starts with a copy as its instance metadata.
        /// </summary>
        /// <remarks>A new dict on every read, so editing one env's metadata never reaches another (Python's is one shared dict).</remarks>
        public static new Metadata ClassMetadata => new() {
            [Metadata.RenderModesKey] = new List<string> { "human", "ansi", "rgb_array" },
            [Metadata.RenderFpsKey] = 4L,
        };

        /// <summary>Gymnasium's <c>locs</c>: the stands R, G, Y and B as <c>(row, col)</c>.</summary>
        public IReadOnlyList<(int Row, int Col)> Locs { get; } = new[] { (0, 0), (0, 4), (4, 0), (4, 3) };

        /// <summary>Gymnasium's <c>locs_colors</c>: red, green, yellow and blue, for the stands.</summary>
        public IReadOnlyList<(int R, int G, int B)> LocsColors { get; } = new[] { (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255) };

        /// <summary>Gymnasium's <c>desc</c>: the map as NumPy's <c>(7, 11)</c> <c>S1</c> array, a new array on every read that the caller owns.</summary>
        public NDArray Desc => ToyTextUtils.CharMap(Map);

        /// <summary>Gymnasium's <c>rainy_probability</c>, fixed at construction.</summary>
        public double RainyProbability { get; }

        /// <summary>Gymnasium's <c>_rainy_lateral_probability</c>: <c>(1 - rainy_probability) / 2</c>.</summary>
        [PyName("_rainy_lateral_probability")]
        public double RainyLateralProbability { get; }

        /// <summary>Gymnasium's <c>fickle_probability</c>: the chance, drawn at each reset of a fickle env, that this episode's passenger may change their mind.</summary>
        public double FickleProbability { get; set; }

        /// <summary>Gymnasium's <c>fickle_passenger</c>.</summary>
        public bool FicklePassenger { get; set; }

        /// <summary>Gymnasium's <c>fickle_step</c>: whether the passenger may still change their destination this episode.</summary>
        public bool FickleStep { get; set; }

        /// <summary>Gymnasium's <c>max_row</c>, 4.</summary>
        public int MaxRow { get; }

        /// <summary>Gymnasium's <c>max_col</c>, 4.</summary>
        public int MaxCol { get; }

        /// <summary>Gymnasium's <c>initial_state_distrib</c>, a new array on every read.</summary>
        public double[] InitialStateDistrib => (double[])_initialStateDistrib.Clone();

        /// <summary>Gymnasium's <c>P</c>: for each state and action, the possible outcomes <c>(probability, next state, reward, terminated)</c>.</summary>
        [PyName("P")]
        public IReadOnlyDictionary<long, IReadOnlyDictionary<long, IReadOnlyList<ToyTextTransition>>> P => _p;

        /// <summary>Gymnasium's <c>s</c>: the encoded state; <see langword="null"/> before the first reset. Assigning one injects a state.</summary>
        [PyName("s")]
        public long? S {
            get => _hasState ? _s : null;
            set {
                _hasState = value is not null;
                _s = value ?? 0;
            }
        }

        /// <summary>Gymnasium's <c>lastaction</c>: the action key of the last step, or <see langword="null"/> after a reset (Python's <c>None</c>).</summary>
        [PyName("lastaction")]
        public long? LastAction => _lastAction?.Key;

        /// <summary>
        ///     Gymnasium's <c>taxi_orientation</c>: the sprite the taxi is drawn with (0 south, 1 north, 2 east, 3 west), set to 0
        ///     by a reset and to the last movement by a render, not by the step.
        /// </summary>
        /// <remarks>
        ///     The value is the action's key: after a float movement such as <c>2.0</c> it reads 2, while every later image
        ///     render raises Python's <c>TypeError</c> at the sprite lookup, as Gymnasium's does with the float it stored.
        /// </remarks>
        [PyName("taxi_orientation")]
        public long TaxiOrientation => _taxiOrientation.Value;

        /// <summary>Gymnasium's <c>cell_size</c>: <c>(88, 56)</c> pixels.</summary>
        public (int Width, int Height) CellSize { get; } = (88, 56);

        /// <summary>Gymnasium's <c>window</c>: the frame every image render draws on, <see langword="null"/> until the first one.</summary>
        public Surface? Window { get; private set; }

        /// <summary>Gymnasium's <c>clock</c>, <see langword="null"/> until the first image render.</summary>
        public Clock? Clock { get; private set; }

        /// <summary>Gymnasium's <c>taxi_imgs</c>: the empty taxi facing south, north, east and west.</summary>
        public IReadOnlyList<Surface>? TaxiImgs { get; private set; }

        /// <summary>Gymnasium's <c>taxi_full_imgs</c>: the taxi with the passenger, by the same orientations.</summary>
        public IReadOnlyList<Surface>? TaxiFullImgs { get; private set; }

        /// <summary>Gymnasium's <c>passenger_img</c>.</summary>
        public Surface? PassengerImg { get; private set; }

        /// <summary>Gymnasium's <c>destination_img</c>.</summary>
        public Surface? DestinationImg { get; private set; }

        /// <summary>Gymnasium's <c>house_img</c>.</summary>
        public Surface? HouseImg { get; private set; }

        /// <summary>Gymnasium's <c>background_img</c>.</summary>
        public Surface? BackgroundImg { get; private set; }

        /// <summary>Gymnasium's <c>encode</c>: <c>((taxi_row * 5 + taxi_col) * 5 + pass_loc) * 4 + dest_idx</c>.</summary>
        /// <param name="taxiRow">The taxi's row.</param>
        /// <param name="taxiCol">The taxi's column.</param>
        /// <param name="passLoc">The passenger's stand (0-3), or 4 in the taxi.</param>
        /// <param name="destIdx">The destination stand.</param>
        /// <returns>The state.</returns>
        public long Encode(long taxiRow, long taxiCol, long passLoc, long destIdx) {
            // (5) 5, 5, 4
            long i = taxiRow;
            i *= 5;
            i += taxiCol;
            i *= 5;
            i += passLoc;
            i *= 4;
            i += destIdx;
            return i;
        }

        /// <summary>Gymnasium's <c>decode</c>: the state's <c>(taxi_row, taxi_col, pass_loc, dest_idx)</c>.</summary>
        /// <param name="i">The state.</param>
        /// <returns>The four parts.</returns>
        /// <exception cref="AssertionError">Thrown for a state beyond the grid (Gymnasium's bare <c>assert 0 &lt;= i &lt; 5</c> on the row).</exception>
        public (long TaxiRow, long TaxiCol, long PassLoc, long DestIdx) Decode(long i) {
            long destIdx = PyMath.Mod(i, 4);
            i = PyMath.FloorDiv(i, 4);
            long passLoc = PyMath.Mod(i, 5);
            i = PyMath.FloorDiv(i, 5);
            long taxiCol = PyMath.Mod(i, 5);
            i = PyMath.FloorDiv(i, 5);
            if (!(0 <= i && i < 5))
                throw new AssertionError();
            return (i, taxiCol, passLoc, destIdx);
        }

        /// <summary>Gymnasium's <c>action_mask</c>: which of the six actions change the state, as an int8 array of 0s and 1s.</summary>
        /// <param name="state">The state.</param>
        /// <returns>A new <c>(6,)</c> int8 array; the caller owns it.</returns>
        /// <exception cref="AssertionError">Thrown for a state beyond the grid.</exception>
        public NDArray ActionMask(long state) {
            var mask = new sbyte[6];
            (long taxiRow, long taxiCol, long passLoc, long destIdx) = Decode(state);
            if (taxiRow < 4)
                mask[0] = 1;
            if (taxiRow > 0)
                mask[1] = 1;
            if (taxiCol < 4 && _cells[taxiRow + 1, 2 * taxiCol + 2] == ':')
                mask[2] = 1;
            if (taxiCol > 0 && _cells[taxiRow + 1, 2 * taxiCol] == ':')
                mask[3] = 1;
            if (passLoc < 4 && (taxiRow, taxiCol) == Locs[(int)passLoc])
                mask[4] = 1;
            if (passLoc == 4 && ((taxiRow, taxiCol) == Locs[(int)destIdx] || Locs.Contains(((int)taxiRow, (int)taxiCol))))
                mask[5] = 1;
            return np.array(mask);
        }

        /// <summary>
        ///     Starts an episode: reseeds the generator when a seed is given, draws the start state (one draw), and for a fickle
        ///     env draws once more whether this episode's passenger may change their mind. Renders in human mode.
        /// </summary>
        /// <param name="seed">The seed of a fresh <c>Generator(PCG64(SeedSequence(seed)))</c>, or <see langword="null"/> to keep the generator.</param>
        /// <param name="options">Ignored, as in Gymnasium.</param>
        /// <returns>The state as a 0-d int64 array, and <c>{"prob": 1.0, "action_mask": mask}</c>.</returns>
        /// <exception cref="Error">Thrown for a negative seed, with Gymnasium's message.</exception>
        /// <exception cref="DependencyNotInstalled">Thrown in human mode by the first render when no viewer factory is set (DR-20).</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/taxi.py#L475">gymnasium/envs/toy_text/taxi.py</see>.</remarks>
        public override ResetResult<NDArray> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            SeedNpRandom(seed);
            _s = ToyTextUtils.CategoricalSample(_initialStateDistrib, NpRandom);
            _hasState = true;
            _lastAction = null;
            // self.fickle_passenger and self.np_random.random() < self.fickle_probability: the draw only for a fickle env.
            if (FicklePassenger) {
                using NDArray draw = NpRandom.random();
                FickleStep = draw.GetAtIndex<double>(0) < FickleProbability;
            } else {
                FickleStep = false;
            }

            _taxiOrientation = (0, () => 0);

            if (RenderMode == "human")
                Render();
            return new ResetResult<NDArray>(NDArray.Scalar<long>(_s), new Info { ["prob"] = 1.0, ["action_mask"] = ActionMask(_s) });
        }

        /// <summary>
        ///     Takes the action: looks up <c>P[s][a]</c> and draws one outcome (one float64); a fickle passenger who has just
        ///     ridden their first move changes their destination (one more draw). Renders in human mode.
        /// </summary>
        /// <param name="action">A number equal to 0 (south), 1 (north), 2 (east), 3 (west), 4 (pickup) or 5 (dropoff).</param>
        /// <returns>The new state, the reward, whether the passenger was delivered, <c>truncated=False</c>, and <c>{"prob": p, "action_mask": mask}</c>.</returns>
        /// <exception cref="AttributeError">Thrown before the first reset (<c>'TaxiEnv' object has no attribute 's'</c>), before the action is looked at.</exception>
        /// <exception cref="KeyError">Thrown for an action no key equals, or an injected state outside the table, with Python's message.</exception>
        /// <exception cref="TypeError">Thrown for an array action of one or more dimensions (<c>unhashable type: 'numpy.ndarray'</c>).</exception>
        /// <exception cref="DependencyNotInstalled">Thrown in human mode by the first render when no viewer factory is set (DR-20).</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/taxi.py#L437">gymnasium/envs/toy_text/taxi.py</see>.</remarks>
        public override StepResult<NDArray> Step(NDArray action) {
            if (!_hasState)
                throw new AttributeError("'TaxiEnv' object has no attribute 's'");
            if (!_p.TryGetValue(_s, out IReadOnlyDictionary<long, IReadOnlyList<ToyTextTransition>>? byAction))
                throw new KeyError(PyBuiltins.Repr(_s));
            long key = PyKeys.DictKey(action, 6);
            IReadOnlyList<ToyTextTransition> transitions = byAction[key];
            var probabilities = new double[transitions.Count];
            for (int t = 0; t < probabilities.Length; t++)
                probabilities[t] = transitions[t].Probability;
            ToyTextTransition chosen = transitions[(int)ToyTextUtils.CategoricalSample(probabilities, NpRandom)];
            object? given = ReferenceEquals(action, _convertedAction) ? _givenAction : action;
            _givenAction = null;
            _convertedAction = null;
            _lastAction = (key, DeferredListIndex(given));

            long s = chosen.NextState;
            (long shadowRow, long shadowCol, long shadowPassLoc, long shadowDestIdx) = Decode(_s);
            (long taxiRow, long taxiCol, long passLoc, _) = Decode(s);
            // If we are in the fickle step, the passenger has been in the vehicle for at least a step and this step the
            // position changed
            if (FicklePassenger && FickleStep && shadowPassLoc == 4 && (taxiRow != shadowRow || taxiCol != shadowCol)) {
                FickleStep = false;
                long[] possibleDestinations = Enumerable.Range(0, Locs.Count).Where(i => i != shadowDestIdx).Select(i => (long)i).ToArray();
                long destIdx;
                using (NDArray destinations = np.array(possibleDestinations))
                using (NDArray picked = NpRandom.choice(destinations))
                    destIdx = picked.GetAtIndex<long>(0);
                s = Encode(taxiRow, taxiCol, passLoc, destIdx);
            }

            _s = s;
            if (RenderMode == "human")
                Render();
            // truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`.
            return new StepResult<NDArray>(NDArray.Scalar<long>(s), chosen.Reward, chosen.Terminated, false,
                new Info { ["prob"] = chosen.Probability, ["action_mask"] = ActionMask(s) });
        }

        /// <summary>
        ///     Renders by the render mode: <c>ansi</c> returns the map as coloured text; any other mode draws Gymnasium's pixel
        ///     art (<c>rgb_array</c> returns the frame).
        /// </summary>
        /// <returns><c>ansi</c>: the text. <c>rgb_array</c>: the <c>(456, 536, 3)</c> uint8 frame, a new array the caller owns. <c>human</c>: nothing.</returns>
        /// <exception cref="AssertionError">
        ///     Thrown without a render mode when the env has no spec, and for a mode other than <c>human</c>, <c>ansi</c> and
        ///     <c>rgb_array</c> (<c>Something went wrong with pygame. This should never happen.</c>), as in Gymnasium.
        /// </exception>
        /// <exception cref="AttributeError">Thrown before the first reset (no <c>s</c>; for images after the background was drawn).</exception>
        /// <exception cref="TypeError">Thrown after an action Python's list indexing refuses (a float action), by the caption or the taxi sprite's lookup.</exception>
        /// <exception cref="KeyError">Thrown in human mode when <c>metadata["render_fps"]</c> was removed.</exception>
        /// <exception cref="DependencyNotInstalled">Thrown on the first human render when no viewer factory is set (DR-20).</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/taxi.py#L494">gymnasium/envs/toy_text/taxi.py</see>.</remarks>
        public override RenderResult Render() {
            if (RenderMode is null) {
                EnvSpec spec = Spec ?? throw new AssertionError();
                Logger.Warn("You are calling render method without specifying any render mode. You can specify the render_mode at initialization, "
                            + $"e.g. gym.make(\"{spec.Id}\", render_mode=\"rgb_array\")");
                return RenderResult.None;
            }

            return RenderMode == "ansi" ? new RenderResult.Ansi(RenderText()) : RenderGui(RenderMode);
        }

        /// <summary>Once a window exists, shuts the human-mode window down (pygame's <c>display.quit()</c>); the surface is kept.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/taxi.py#L626">gymnasium/envs/toy_text/taxi.py</see>.</remarks>
        public override void Close() {
            if (Window is not null)
                _window.Close();
        }

        /// <summary>
        ///     The untyped step's action conversion, with Python's order kept: the state's existence first, then the dict lookup
        ///     on the action as given (its Python type is also kept for the renders' list indexing).
        /// </summary>
        /// <param name="action">The action as given to the untyped step.</param>
        /// <returns>The action as an array: the one given, or a new one the step releases.</returns>
        /// <exception cref="AttributeError">Thrown before the first reset.</exception>
        /// <exception cref="KeyError">Thrown for an action no key equals.</exception>
        /// <exception cref="TypeError">Thrown for an array of one or more dimensions.</exception>
        protected override NDArray ConvertAction(object? action) {
            if (!_hasState)
                throw new AttributeError("'TaxiEnv' object has no attribute 's'");
            PyKeys.DictKey(action, 6);
            NDArray converted = base.ConvertAction(action);
            _givenAction = action;
            _convertedAction = converted;
            return converted;
        }

        /// <summary>Gymnasium's <c>_pickup</c>: the passenger boards when the taxi is at their stand, else -10.</summary>
        /// <param name="taxiRow">The taxi's row.</param>
        /// <param name="taxiCol">The taxi's column.</param>
        /// <param name="passIdx">The passenger's stand, or 4.</param>
        /// <param name="reward">The reward when the pickup succeeds.</param>
        /// <returns>The passenger's new place and the reward.</returns>
        private (int PassIdx, double Reward) Pickup(int taxiRow, int taxiCol, int passIdx, double reward) {
            if (passIdx < 4 && (taxiRow, taxiCol) == Locs[passIdx])
                return (4, reward);
            // passenger not at location
            return (passIdx, -10);
        }

        /// <summary>Gymnasium's <c>_dropoff</c>: delivery at the destination (20, terminal), a drop at another stand, or -10.</summary>
        /// <param name="taxiRow">The taxi's row.</param>
        /// <param name="taxiCol">The taxi's column.</param>
        /// <param name="passIdx">The passenger's place.</param>
        /// <param name="destIdx">The destination stand.</param>
        /// <param name="defaultReward">The reward of a drop at another stand.</param>
        /// <returns>The passenger's new place, the reward and whether the episode ends.</returns>
        private (int PassIdx, double Reward, bool Terminated) Dropoff(int taxiRow, int taxiCol, int passIdx, int destIdx, double defaultReward) {
            if ((taxiRow, taxiCol) == Locs[destIdx] && passIdx == 4)
                return (destIdx, 20, true);
            int stand = IndexOfLoc(taxiRow, taxiCol);
            if (stand >= 0 && passIdx == 4)
                return (stand, defaultReward, false);
            // dropoff at wrong location
            return (passIdx, -10, false);
        }

        /// <summary><c>self.locs.index((row, col))</c>, or -1 when the cell is not a stand.</summary>
        /// <param name="row">The row.</param>
        /// <param name="col">The column.</param>
        /// <returns>The stand's index, or -1.</returns>
        private int IndexOfLoc(int row, int col) {
            for (int i = 0; i < Locs.Count; i++)
                if (Locs[i] == (row, col))
                    return i;
            return -1;
        }

        /// <summary>Gymnasium's <c>_build_dry_transitions</c>: the one certain outcome of an action, walls checked against the map.</summary>
        /// <param name="lists">The table under construction.</param>
        /// <param name="row">The taxi's row.</param>
        /// <param name="col">The taxi's column.</param>
        /// <param name="passIdx">The passenger's place.</param>
        /// <param name="destIdx">The destination stand.</param>
        /// <param name="action">The action.</param>
        private void BuildDryTransitions(Dictionary<long, List<ToyTextTransition>[]> lists, int row, int col, int passIdx, int destIdx, int action) {
            long state = Encode(row, col, passIdx, destIdx);
            int newRow = row, newCol = col, newPassIdx = passIdx;
            double reward = -1; // default reward when there is no pickup/dropoff
            bool terminated = false;

            if (action == 0)
                newRow = Math.Min(row + 1, MaxRow);
            else if (action == 1)
                newRow = Math.Max(row - 1, 0);
            if (action == 2 && _cells[1 + row, 2 * col + 2] == ':')
                newCol = Math.Min(col + 1, MaxCol);
            else if (action == 3 && _cells[1 + row, 2 * col] == ':')
                newCol = Math.Max(col - 1, 0);
            else if (action == 4) // pickup
                (newPassIdx, reward) = Pickup(row, col, newPassIdx, reward);
            else if (action == 5) // dropoff
                (newPassIdx, reward, terminated) = Dropoff(row, col, newPassIdx, destIdx, reward);

            long newState = Encode(newRow, newCol, newPassIdx, destIdx);
            lists[state][action].Add(new ToyTextTransition(1.0, newState, reward, terminated));
        }

        /// <summary>Gymnasium's <c>_calc_new_position</c>: a lateral drift, clamped to the grid and blocked by a wall.</summary>
        /// <param name="row">The taxi's row.</param>
        /// <param name="col">The taxi's column.</param>
        /// <param name="movement">The drift.</param>
        /// <returns>The position after the drift.</returns>
        private (int Row, int Col) CalcNewPosition(int row, int col, (int Dr, int Dc) movement) {
            (int dr, int dc) = movement;
            int newRow = Math.Max(0, Math.Min(row + dr, MaxRow));
            int newCol = Math.Max(0, Math.Min(col + dc, MaxCol));
            if (dc == 1 && _cells[1 + newRow, 2 * newCol] != ':')
                return (row, col); // east wall blocks
            if (dc == -1 && _cells[1 + newRow, 2 * newCol + 2] != ':')
                return (row, col); // west wall blocks
            return (newRow, newCol);
        }

        /// <summary>
        ///     Gymnasium's <c>_build_rainy_transitions</c>: a possible move goes ahead with <c>rainy_probability</c> and drifts
        ///     to either side with the lateral probability; an impossible move stays put without drifting; pickup and dropoff
        ///     are certain.
        /// </summary>
        /// <param name="lists">The table under construction.</param>
        /// <param name="row">The taxi's row.</param>
        /// <param name="col">The taxi's column.</param>
        /// <param name="passIdx">The passenger's place.</param>
        /// <param name="destIdx">The destination stand.</param>
        /// <param name="action">The action.</param>
        private void BuildRainyTransitions(Dictionary<long, List<ToyTextTransition>[]> lists, int row, int col, int passIdx, int destIdx, int action) {
            long state = Encode(row, col, passIdx, destIdx);
            (int Row, int Col) leftPos = (row, col), rightPos = (row, col);
            int newRow = row, newCol = col, newPassIdx = passIdx;
            double reward = -1; // default reward when there is no pickup/dropoff
            bool terminated = false;

            // (forward, left, right) relative to each heading.
            (int, int)[][] moves = {
                new[] { (1, 0), (0, 1), (0, -1) }, // Down  (south): left=east,  right=west
                new[] { (-1, 0), (0, -1), (0, 1) }, // Up   (north): left=west,  right=east
                new[] { (0, 1), (-1, 0), (1, 0) }, // Right (east):  left=north, right=south
                new[] { (0, -1), (1, 0), (-1, 0) }, // Left  (west):  left=south, right=north
            };

            // Check if the primary move is possible. When it is not (wall or grid boundary), no lateral drift occurs either.
            if ((action == 0 && row < MaxRow) || (action == 1 && row > 0) || (action == 2 && _cells[1 + row, 2 * col + 2] == ':') || (action == 3 && _cells[1 + row, 2 * col] == ':')) {
                (int dr, int dc) = moves[action][0];
                newRow = Math.Max(0, Math.Min(row + dr, MaxRow));
                newCol = Math.Max(0, Math.Min(col + dc, MaxCol));
                leftPos = CalcNewPosition(row, col, moves[action][1]);
                rightPos = CalcNewPosition(row, col, moves[action][2]);
            } else if (action == 4) { // pickup
                (newPassIdx, reward) = Pickup(row, col, newPassIdx, reward);
            } else if (action == 5) { // dropoff
                (newPassIdx, reward, terminated) = Dropoff(row, col, newPassIdx, destIdx, reward);
            }

            long intendedState = Encode(newRow, newCol, newPassIdx, destIdx);
            if (action <= 3) {
                long leftState = Encode(leftPos.Row, leftPos.Col, newPassIdx, destIdx);
                long rightState = Encode(rightPos.Row, rightPos.Col, newPassIdx, destIdx);
                lists[state][action].Add(new ToyTextTransition(RainyProbability, intendedState, -1, terminated));
                lists[state][action].Add(new ToyTextTransition(RainyLateralProbability, leftState, -1, terminated));
                lists[state][action].Add(new ToyTextTransition(RainyLateralProbability, rightState, -1, terminated));
            } else {
                lists[state][action].Add(new ToyTextTransition(1.0, intendedState, reward, terminated));
            }
        }

        /// <summary>
        ///     What Python's list indexing will give for an action: its index, or the error it raises, deferred to the render
        ///     that does the indexing, as in Python.
        /// </summary>
        /// <param name="action">The action as given (only read now).</param>
        /// <returns>A function giving the index or throwing Python's error.</returns>
        /// <exception cref="TypeError">Thrown by the returned function, not by this method, for an action list indexing refuses.</exception>
        /// <exception cref="IndexError">Thrown by the returned function, not by this method, for an action outside the list.</exception>
        private static Func<int> DeferredListIndex(object? action) {
            try {
                // The caption list is the longest one indexed (six entries); the sprite lists are only indexed by movements.
                int index = PyKeys.ListIndex(action, ActionNames.Length);
                return () => index;
            } catch (TypeError error) {
                string message = error.Message;
                return () => throw new TypeError(message);
            } catch (IndexError error) {
                string message = error.Message;
                return () => throw new IndexError(message);
            }
        }

        /// <summary><c>get_surf_loc</c>: a grid cell's top-left pixel.</summary>
        /// <param name="row">The row.</param>
        /// <param name="col">The column.</param>
        /// <returns>The pixel.</returns>
        private (int X, int Y) GetSurfLoc(long row, long col) => ((int)(GridOrigin.X + col * CellSize.Width), (int)(GridOrigin.Y + row * CellSize.Height));

        /// <summary>Gymnasium's <c>_render_text</c>: the map with the taxi, passenger and destination coloured, then the last action's caption.</summary>
        /// <returns>The text.</returns>
        /// <exception cref="AttributeError">Thrown before the first reset.</exception>
        /// <exception cref="AssertionError">Thrown for an injected state beyond the grid.</exception>
        /// <exception cref="TypeError">Thrown when the last action cannot index the caption list.</exception>
        /// <exception cref="IndexError">Thrown when the last action is outside the caption list.</exception>
        private string RenderText() {
            var output = new string[_cells.GetLength(0)][];
            for (int r = 0; r < output.Length; r++) {
                output[r] = new string[_cells.GetLength(1)];
                for (int c = 0; c < output[r].Length; c++)
                    output[r][c] = _cells[r, c].ToString();
            }

            if (!_hasState)
                throw new AttributeError("'TaxiEnv' object has no attribute 's'");
            (long taxiRow, long taxiCol, long passIdx, long destIdx) = Decode(_s);

            static string Ul(string x) => x == " " ? "_" : x;

            if (passIdx < 4) {
                output[1 + taxiRow][2 * taxiCol + 1] = Logger.Colorize(output[1 + taxiRow][2 * taxiCol + 1], "yellow", highlight: true);
                (int pi, int pj) = Locs[(int)passIdx];
                output[1 + pi][2 * pj + 1] = Logger.Colorize(output[1 + pi][2 * pj + 1], "blue", bold: true);
            } else { // passenger in taxi
                output[1 + taxiRow][2 * taxiCol + 1] = Logger.Colorize(Ul(output[1 + taxiRow][2 * taxiCol + 1]), "green", highlight: true);
            }

            (int di, int dj) = Locs[(int)destIdx];
            output[1 + di][2 * dj + 1] = Logger.Colorize(output[1 + di][2 * dj + 1], "magenta");
            var outfile = new StringBuilder();
            outfile.Append(string.Join("\n", output.Select(row => string.Concat(row)))).Append('\n');
            if (_lastAction is { } last)
                outfile.Append("  (").Append(ActionNames[last.ListIndex()]).Append(")\n");
            else
                outfile.Append('\n');
            return outfile.ToString();
        }

        /// <summary>Gymnasium's <c>_render_gui</c>: canvas, background, house, destination marker, passenger and taxi.</summary>
        /// <param name="mode">The render mode.</param>
        /// <returns>The frame for <c>rgb_array</c>, nothing for <c>human</c>.</returns>
        /// <exception cref="AssertionError">Thrown for a mode other than <c>human</c> and <c>rgb_array</c>, as Gymnasium's assert does, and for a state beyond the grid.</exception>
        /// <exception cref="AttributeError">Thrown before the first reset, after the background was drawn.</exception>
        /// <exception cref="TypeError">Thrown when the taxi's orientation came from an action list indexing refuses.</exception>
        /// <exception cref="KeyError">Thrown in human mode when <c>metadata["render_fps"]</c> was removed.</exception>
        /// <exception cref="DependencyNotInstalled">Thrown on the first human render when no viewer factory is set.</exception>
        private RenderResult RenderGui(string mode) {
            if (Window is null) {
                // pygame.display.set_caption("Taxi")
                if (mode == "human") {
                    _window.Ensure(WindowSize.Width, WindowSize.Height);
                    Window = new Surface(WindowSize.Width, WindowSize.Height);
                } else if (mode == "rgb_array") {
                    Window = new Surface(WindowSize.Width, WindowSize.Height);
                }
            }

            Surface surface = Window ?? throw new AssertionError("Something went wrong with pygame. This should never happen.");
            Clock ??= new Clock();

            TaxiImgs ??= Orientations("");
            TaxiFullImgs ??= Orientations("_full");
            PassengerImg ??= Load("passenger");
            DestinationImg ??= Load("marker_destination");
            HouseImg ??= Load("house");
            BackgroundImg ??= Load("background");

            surface.Fill(CanvasColor);
            surface.Blit(BackgroundImg, BackgroundPos.X, BackgroundPos.Y);
            surface.Blit(HouseImg, HousePos.X, HousePos.Y);

            if (!_hasState)
                throw new AttributeError("'TaxiEnv' object has no attribute 's'");
            (long taxiRow, long taxiCol, long passIdx, long destIdx) = Decode(_s);
            (int destX, int destY) = GetSurfLoc(Locs[(int)destIdx].Row, Locs[(int)destIdx].Col);
            surface.Blit(DestinationImg, destX, destY);

            if (passIdx < 4) {
                (int x, int y) = GetSurfLoc(Locs[(int)passIdx].Row, Locs[(int)passIdx].Col);
                surface.Blit(PassengerImg, x + PassengerOffset.X, y + PassengerOffset.Y);
            }

            // if self.lastaction in [0, 1, 2, 3]: self.taxi_orientation = self.lastaction (equality: 2.0 and True count).
            if (_lastAction is { Key: >= 0 and <= 3 } last)
                _taxiOrientation = (last.Key, last.ListIndex);

            IReadOnlyList<Surface> sprites = passIdx == 4 ? TaxiFullImgs : TaxiImgs;
            // self.taxi_imgs[self.taxi_orientation]: Python's list indexing of the orientation.
            int orientation = _taxiOrientation.ListIndex();
            Surface taxiImg = sprites[orientation];
            (int dx, int dy) = orientation < 2 ? TaxiVOffset : TaxiHOffset;
            (int taxiX, int taxiY) = GetSurfLoc(taxiRow, taxiCol);
            surface.Blit(taxiImg, taxiX + dx, taxiY + dy);

            if (mode == "human") {
                // pygame.event.pump(); pygame.display.update(); self.clock.tick(self.metadata["render_fps"])
                _window.Show(surface);
                Clock.Tick(PyBuiltins.Float(Metadata[Metadata.RenderFpsKey]));
                return RenderResult.None;
            }

            return new RenderResult.Rgb(SurfArray.ToRgbArray(surface));
        }

        /// <summary>Gymnasium's <c>load</c>: an image of <c>taxi</c>, unscaled.</summary>
        /// <param name="name">The image's name without extension.</param>
        /// <returns>The shared, read-only surface (never drawn on).</returns>
        private static Surface Load(string name) => EnvAssets.Load($"toy_text/taxi/{name}.png");

        /// <summary>
        ///     Gymnasium's <c>orientations</c>: the vertical sprite flipped to face south, as is (north), the horizontal sprite
        ///     (east), and flipped to face west.
        /// </summary>
        /// <param name="name">The sprite's suffix, empty or <c>_full</c>.</param>
        /// <returns>The four sprites, by orientation.</returns>
        private static IReadOnlyList<Surface> Orientations(string name) {
            Surface vertical = Load($"taxi_v{name}"), horizontal = Load($"taxi_h{name}");
            return new[] { Transform.Flip(vertical, false, true), vertical, horizontal, Transform.Flip(horizontal, true, false) };
        }
    }
}
