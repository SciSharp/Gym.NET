#nullable enable
using System;
using System.Collections.Generic;
using System.IO;
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
    ///     FrozenLake-v1 and FrozenLake8x8-v1: walk from the start <c>S</c> to the goal <c>G</c> across a frozen lake without
    ///     falling into a hole <c>H</c>; on slippery ice a move goes perpendicular with probability <c>1 - success_rate</c>.
    ///     Byte for byte what Gymnasium's FrozenLake computes: the transition table <see cref="P"/>, the generator draws,
    ///     observations, rewards, <c>{"prob": p}</c> infos, <c>ansi</c> text and <c>rgb_array</c> pixels.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Like Gymnasium's, the env checks no action against its space: <c>self.P[self.s][a]</c> is a dict lookup, so any
    ///     number equal to 0, 1, 2 or 3 is an action (<c>True</c>, <c>2.0</c>, <c>np.float32(1.0)</c>) and anything else
    ///     raises <c>KeyError</c> with the action's <c>repr</c>. Each step draws one float64 from the env's generator
    ///     (<see cref="ToyTextUtils.CategoricalSample"/>), even on solid ice. Observations are 0-d int64 arrays (DR-08), the
    ///     Python <c>int</c> Gymnasium returns; rewards are <c>reward_schedule</c>'s values as float64.
    ///     </para>
    ///     <para>
    ///     The map is held as Gymnasium's <c>S1</c> character array (<see cref="Desc"/>, built by
    ///     <see cref="ToyTextUtils.CharMap"/>). Rendering is Gymnasium's pixel
    ///     art: tiles scaled to the cell, overlays scaled by <c>round</c> and blitted at truncated positions, a one-pixel
    ///     grid; <c>ansi</c> marks the agent in red with Gymnasium's escape codes. Human mode renders into the window from
    ///     <c>viewer_factory</c> or <c>Gymnasium.HumanViewerFactory</c> (DR-20) at 4 frames per second.
    ///     </para>
    ///     <para>
    ///     Accepted deviations: <c>reward_schedule</c> is a list of doubles, and <see langword="null"/> means its default
    ///     <c>(1, 0, 0)</c> (Python's <c>None</c> would fail in <c>min</c>); <see cref="LastAction"/> keeps the key the action
    ///     found, where Python keeps the action object (the <c>ansi</c> caption still applies Python's list-index rules to the
    ///     action given). Not thread-safe. Reference:
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/frozen_lake.py#L83">gymnasium/envs/toy_text/frozen_lake.py</see>.
    ///     </para>
    ///     <para>Its info reads typed (DR-91): <c>FrozenLakeInfo info = step.Info;</c> (<see cref="FrozenLakeInfo"/>), with the dict's own values still behind the indexer.</para>
    /// </remarks>
    [GymOptions]
    public partial class FrozenLakeEnv : Env<NDArray, NDArray> {
        /// <summary>Gymnasium's <c>LEFT</c>, action 0.</summary>
        public const long Left = 0;

        /// <summary>Gymnasium's <c>DOWN</c>, action 1.</summary>
        public const long Down = 1;

        /// <summary>Gymnasium's <c>RIGHT</c>, action 2.</summary>
        public const long Right = 2;

        /// <summary>Gymnasium's <c>UP</c>, action 3.</summary>
        public const long Up = 3;

        /// <summary>The <c>ansi</c> caption of each action, Gymnasium's list <c>['Left', 'Down', 'Right', 'Up']</c>.</summary>
        private static readonly string[] ActionNames = { "Left", "Down", "Right", "Up" };

        /// <summary>The human-mode window, opened on the first human render.</summary>
        private readonly HumanWindow _window;

        /// <summary>The map's characters, row by row, for the dynamics and the drawing.</summary>
        private readonly char[,] _cells;

        /// <summary>The transition table, by state then action.</summary>
        private readonly Dictionary<long, IReadOnlyDictionary<long, IReadOnlyList<ToyTextTransition>>> _p;

        /// <summary>Python's <c>self.s</c>, meaningful only while <see cref="_hasState"/>.</summary>
        private long _s;

        /// <summary>Whether Python's <c>self.s</c> exists: set by the first reset or an assignment of <see cref="S"/>.</summary>
        private bool _hasState;

        /// <summary>
        ///     The <c>ansi</c> caption's index into <see cref="ActionNames"/> for the last action, or the error Python's list
        ///     indexing raises for it at render time; <see langword="null"/> when there is no last action (<c>None</c>).
        /// </summary>
        private Func<int>? _lastActionIndex;

        /// <summary>
        ///     The action as given to the untyped step, kept from <see cref="ConvertAction"/> to the <see cref="Step"/> it
        ///     feeds, so the caption's list indexing sees the Python counterpart of what was given (<c>2.0</c> is a
        ///     <c>float</c>, not the float64 array it was converted to).
        /// </summary>
        private object? _givenAction;

        /// <summary>
        ///     The array <see cref="ConvertAction"/> made of <see cref="_givenAction"/>, which identifies the step it feeds. Only
        ///     compared by reference, never read: the untyped step owns and releases it.
        /// </summary>
        [NDBorrowed]
        private NDArray? _convertedAction;

        /// <summary>Creates the env; the map comes from <paramref name="desc"/>, else <paramref name="mapName"/>, else a random 8x8 map.</summary>
        /// <param name="renderMode">Gymnasium's <c>render_mode</c>: <see langword="null"/>, <c>"human"</c>, <c>"ansi"</c> or <c>"rgb_array"</c>.</param>
        /// <param name="desc">Gymnasium's <c>desc</c>: the map's rows (<c>S</c> start, <c>F</c> frozen, <c>H</c> hole, <c>G</c> goal), all of one length.</param>
        /// <param name="mapName">
        ///     Gymnasium's <c>map_name</c>: <c>"4x4"</c> or <c>"8x8"</c> (<see cref="Maps"/>), used when <paramref name="desc"/> is
        ///     <see langword="null"/>; with both <see langword="null"/>, <see cref="GenerateRandomMap"/> makes an 8x8 map from
        ///     fresh entropy (not reproducible).
        /// </param>
        /// <param name="isSlippery">Gymnasium's <c>is_slippery</c>: moves may slide perpendicular, each with probability <c>(1 - success_rate) / 2</c>.</param>
        /// <param name="successRate">Gymnasium's <c>success_rate</c>: the probability of moving as intended on slippery ice, 1/3 by default.</param>
        /// <param name="rewardSchedule">Gymnasium's <c>reward_schedule</c>: the rewards for reaching the goal, a hole and any other tile; <see langword="null"/> is <c>(1, 0, 0)</c>.</param>
        /// <param name="viewerFactory">Gym.NET's <c>viewer_factory</c> (DR-20) for human mode; <see langword="null"/> uses <c>Gymnasium.HumanViewerFactory</c>.</param>
        /// <exception cref="KeyError">Thrown for a <paramref name="mapName"/> <see cref="Maps"/> does not hold, with Python's message (<c>'5x5'</c>).</exception>
        /// <exception cref="ValueError">Thrown for rows of different lengths, with NumPy's message, or an empty <paramref name="rewardSchedule"/> (<c>min() arg is an empty sequence</c>).</exception>
        /// <exception cref="ZeroDivisionError">Thrown, as Python's <c>ZeroDivisionError</c>, for a map without rows or columns (the cell size divides by them).</exception>
        public FrozenLakeEnv(string? renderMode = null, IReadOnlyList<string>? desc = null, string? mapName = "4x4", bool isSlippery = true, double successRate = 1.0 / 3.0,
            IReadOnlyList<double>? rewardSchedule = null, EnvViewerFactory? viewerFactory = null) {
            Metadata = ClassMetadata;
            IReadOnlyList<string> rows;
            if (desc is null && mapName is null)
                rows = GenerateRandomMap();
            else if (desc is null)
                rows = Maps.TryGetValue(mapName!, out IReadOnlyList<string>? named) ? named : throw new KeyError(PyBuiltins.Repr(mapName));
            else
                rows = desc;

            using (NDArray map = ToyTextUtils.CharMap(rows))
                // Validates the rows (NumPy's ragged-shape error) before anything else is computed from them.
                _ = map.shape;
            Nrow = rows.Count;
            Ncol = rows.Count == 0 ? 0 : rows[0].Length;
            _cells = new char[Nrow, Ncol];
            for (int row = 0; row < Nrow; row++)
                for (int column = 0; column < Ncol; column++)
                    _cells[row, column] = rows[row][column];

            IReadOnlyList<double> schedule = rewardSchedule ?? new[] { 1.0, 0.0, 0.0 };
            if (schedule.Count == 0)
                throw new ValueError("min() arg is an empty sequence");
            // (min(reward_schedule), max(reward_schedule)): Python's builtins, which keep the first of equal values.
            RewardRange = (schedule.Aggregate((a, b) => b < a ? b : a), schedule.Aggregate((a, b) => b > a ? b : a));

            const int nA = 4;
            int nS = Nrow * Ncol;

            // np.array(desc == b"S").astype("float64").ravel() / its sum: uniform over the start tiles (NaN without one).
            var distribution = new double[nS];
            double starts = 0;
            for (int state = 0; state < nS; state++) {
                distribution[state] = _cells[state / Ncol, state % Ncol] == 'S' ? 1.0 : 0.0;
                starts += distribution[state];
            }

            for (int state = 0; state < nS; state++)
                distribution[state] /= starts;
            InitialStateDistrib = distribution;

            double failRate = (1.0 - successRate) / 2.0;
            _p = new Dictionary<long, IReadOnlyDictionary<long, IReadOnlyList<ToyTextTransition>>>(nS);
            for (int row = 0; row < Nrow; row++) {
                for (int column = 0; column < Ncol; column++) {
                    long s = ToState(row, column);
                    var byAction = new Dictionary<long, IReadOnlyList<ToyTextTransition>>(nA);
                    for (int a = 0; a < 4; a++) {
                        var li = new List<ToyTextTransition>(3);
                        char letter = _cells[row, column];
                        if (letter is 'G' or 'H') {
                            li.Add(new ToyTextTransition(1.0, s, 0, true));
                        } else if (isSlippery) {
                            foreach (int b in new[] { PyMath.Mod(a - 1, 4), a, PyMath.Mod(a + 1, 4) })
                                li.Add(Outcome(row, column, (int)b, b == a ? successRate : failRate, schedule));
                        } else {
                            li.Add(Outcome(row, column, a, 1.0, schedule));
                        }

                        byAction[a] = li;
                    }

                    _p[s] = byAction;
                }
            }

            ObservationSpace = new Discrete(nS);
            ActionSpace = new Discrete(nA);
            RenderMode = renderMode;

            // pygame utils
            WindowSize = (Math.Min(64 * Ncol, 512), Math.Min(64 * Nrow, 512));
            if (Ncol == 0 || Nrow == 0)
                throw new ZeroDivisionError("integer division or modulo by zero");
            CellSize = (WindowSize.Width / Ncol, WindowSize.Height / Nrow);
            _window = new HumanWindow("Frozen Lake", viewerFactory);
        }

        /// <summary>
        ///     Builds the env with the render mode given as the <see cref="Gym.Envs.RenderMode"/> enum (DR-84, DR-93), as
        ///     <see cref="FrozenLakeEnv(string, IReadOnlyList{string}, string, bool, double, IReadOnlyList{double}, EnvViewerFactory)"/>
        ///     builds it from the mode's string: <c>new FrozenLakeEnv(RenderMode.RgbArray)</c> builds what
        ///     <c>new FrozenLakeEnv(renderMode: "rgb_array")</c> builds, the member reaching the env as Gymnasium's string
        ///     (<see cref="RenderModes.Name"/>), which its <c>RenderMode</c> property then holds.
        /// </summary>
        /// <remarks>
        ///     The mode is required and comes first, so <c>new FrozenLakeEnv()</c> keeps meaning the other constructor, without a
        ///     mode; leave the mode out there for no rendering. Every other parameter means what it means there.
        /// </remarks>
        /// <param name="renderMode">
        ///     The render mode, which the env must list in its <c>metadata["render_modes"]</c> as for the string; it behaves
        ///     exactly as its Gymnasium string does in the other constructor.
        /// </param>
        /// <param name="desc">Gymnasium's <c>desc</c>: the map's rows (<c>S</c> start, <c>F</c> frozen, <c>H</c> hole, <c>G</c> goal), all of one length.</param>
        /// <param name="mapName">
        ///     Gymnasium's <c>map_name</c>: <c>"4x4"</c> or <c>"8x8"</c> (<see cref="Maps"/>), used when <paramref name="desc"/> is
        ///     <see langword="null"/>; with both <see langword="null"/>, <see cref="GenerateRandomMap"/> makes an 8x8 map from
        ///     fresh entropy (not reproducible).
        /// </param>
        /// <param name="isSlippery">Gymnasium's <c>is_slippery</c>: moves may slide perpendicular, each with probability <c>(1 - success_rate) / 2</c>.</param>
        /// <param name="successRate">Gymnasium's <c>success_rate</c>: the probability of moving as intended on slippery ice, 1/3 by default.</param>
        /// <param name="rewardSchedule">Gymnasium's <c>reward_schedule</c>: the rewards for reaching the goal, a hole and any other tile; <see langword="null"/> is <c>(1, 0, 0)</c>.</param>
        /// <param name="viewerFactory">Gym.NET's <c>viewer_factory</c> (DR-20) for human mode; <see langword="null"/> uses <c>Gymnasium.HumanViewerFactory</c>.</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a value <see cref="Gym.Envs.RenderMode"/> does not declare (a cast integer such as <c>(RenderMode)99</c>).</exception>
        /// <exception cref="KeyError">Thrown for a <paramref name="mapName"/> <see cref="Maps"/> does not hold, with Python's message (<c>'5x5'</c>).</exception>
        /// <exception cref="ValueError">Thrown for rows of different lengths, with NumPy's message, or an empty <paramref name="rewardSchedule"/> (<c>min() arg is an empty sequence</c>).</exception>
        /// <exception cref="ZeroDivisionError">Thrown, as Python's <c>ZeroDivisionError</c>, for a map without rows or columns (the cell size divides by them).</exception>
        public FrozenLakeEnv(RenderMode renderMode, IReadOnlyList<string>? desc = null, string? mapName = "4x4", bool isSlippery = true,
                             double successRate = 1.0 / 3.0, IReadOnlyList<double>? rewardSchedule = null,
                             EnvViewerFactory? viewerFactory = null)
            : this(RenderModes.Name(renderMode), desc, mapName, isSlippery, successRate, rewardSchedule, viewerFactory) { }

        /// <summary>Gymnasium's <c>MAPS</c>: the named maps, <c>"4x4"</c> and <c>"8x8"</c>, row by row.</summary>
        public static IReadOnlyDictionary<string, IReadOnlyList<string>> Maps { get; } = new Dictionary<string, IReadOnlyList<string>>(StringComparer.Ordinal) {
            ["4x4"] = new[] { "SFFF", "FHFH", "FFFH", "HFFG" },
            ["8x8"] = new[] { "SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF", "FFFHFFFF", "FHHFFFHF", "FHFFHFHF", "FFFHFFFG" },
        };

        /// <summary>
        ///     Gymnasium's class attribute <c>metadata</c>: <c>{"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 4}</c>.
        ///     Every env starts with a copy as its instance metadata.
        /// </summary>
        /// <remarks>A new dict on every read, so editing one env's metadata never reaches another (Python's is one shared dict).</remarks>
        public static new Metadata ClassMetadata => new() {
            [Metadata.RenderModesKey] = new List<string> { "human", "ansi", "rgb_array" },
            [Metadata.RenderFpsKey] = 4L,
        };

        /// <summary>Gymnasium's <c>nrow</c>: the map's rows.</summary>
        public int Nrow { get; }

        /// <summary>Gymnasium's <c>ncol</c>: the map's columns.</summary>
        public int Ncol { get; }

        /// <summary>Gymnasium's <c>desc</c>: the map as NumPy's <c>(nrow, ncol)</c> <c>S1</c> array, a new array on every read that the caller owns.</summary>
        public NDArray Desc {
            get {
                var rows = new string[Nrow];
                for (int row = 0; row < Nrow; row++) {
                    var line = new StringBuilder(Ncol);
                    for (int column = 0; column < Ncol; column++)
                        line.Append(_cells[row, column]);
                    rows[row] = line.ToString();
                }

                return ToyTextUtils.CharMap(rows);
            }
        }

        /// <summary>Gymnasium's <c>reward_range</c>: <c>(min(reward_schedule), max(reward_schedule))</c>.</summary>
        [PyName("reward_range")]
        public (double Min, double Max) RewardRange { get; }

        /// <summary>Gymnasium's <c>initial_state_distrib</c>: the start tiles' uniform distribution over the states, a new array on every read.</summary>
        public double[] InitialStateDistrib {
            get => (double[])_initialStateDistrib.Clone();
            private init => _initialStateDistrib = value;
        }

        /// <summary>The distribution <see cref="InitialStateDistrib"/> copies.</summary>
        private readonly double[] _initialStateDistrib = Array.Empty<double>();

        /// <summary>
        ///     Gymnasium's <c>P</c>: for each state and action, the possible outcomes <c>(probability, next state, reward,
        ///     terminated)</c>, in Gymnasium's order (for a slippery move: the move counter-clockwise, as intended, clockwise).
        /// </summary>
        [PyName("P")]
        public IReadOnlyDictionary<long, IReadOnlyDictionary<long, IReadOnlyList<ToyTextTransition>>> P => _p;

        /// <summary>
        ///     Gymnasium's <c>s</c>: the agent's state, <c>row * ncol + col</c>; <see langword="null"/> before the first reset.
        ///     Assigning one injects a state (one the map lacks makes the next step's lookup fail with <c>KeyError</c>).
        /// </summary>
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
        public long? LastAction { get; private set; }

        /// <summary>Gymnasium's <c>window_size</c>: <c>(min(64 * ncol, 512), min(64 * nrow, 512))</c> pixels.</summary>
        public (int Width, int Height) WindowSize { get; }

        /// <summary>Gymnasium's <c>cell_size</c>: <c>window_size // (ncol, nrow)</c> pixels.</summary>
        public (int Width, int Height) CellSize { get; }

        /// <summary>Gymnasium's <c>window_surface</c>: the frame every render draws on, <see langword="null"/> until the first <c>human</c> or <c>rgb_array</c> render.</summary>
        public Surface? WindowSurface { get; private set; }

        /// <summary>Gymnasium's <c>clock</c>: paces human rendering, <see langword="null"/> until the first render.</summary>
        public Clock? Clock { get; private set; }

        /// <summary>Gymnasium's <c>ice_img</c>, scaled to the cell; <see langword="null"/> until the first image render.</summary>
        public Surface? IceImg { get; private set; }

        /// <summary>Gymnasium's <c>ice_crack_imgs</c>, the two crack overlays scaled to the cell.</summary>
        public IReadOnlyList<Surface>? IceCrackImgs { get; private set; }

        /// <summary>Gymnasium's <c>hole_img</c>, scaled to the cell.</summary>
        public Surface? HoleImg { get; private set; }

        /// <summary>Gymnasium's <c>agent_img</c>, scaled from its 40x36 art by the cell's smaller side over 64.</summary>
        public Surface? AgentImg { get; private set; }

        /// <summary>Gymnasium's <c>goal_flag_img</c>, scaled from its 24x32 art.</summary>
        public Surface? GoalFlagImg { get; private set; }

        /// <summary>Gymnasium's <c>letter_s_img</c>, scaled from its 12x20 art.</summary>
        public Surface? LetterSImg { get; private set; }

        /// <summary>Gymnasium's <c>letter_g_img</c>, scaled from its 12x20 art.</summary>
        public Surface? LetterGImg { get; private set; }

        /// <summary>
        ///     Gymnasium's module function <c>generate_random_map</c>: draws <c>size x size</c> tiles, frozen with probability
        ///     <paramref name="p"/>, from a generator seeded with <paramref name="seed"/>, sets the start and goal corners, and
        ///     draws again until a path joins them.
        /// </summary>
        /// <param name="size">The map's side.</param>
        /// <param name="p">The probability of a frozen tile; above 1 counts as 1 (Python's <c>min(1, p)</c>, which also turns NaN into 1).</param>
        /// <param name="seed">The seed of <c>seeding.np_random</c>, or <see langword="null"/> for fresh entropy.</param>
        /// <returns>The map's rows.</returns>
        /// <exception cref="Error">Thrown for a negative seed, with Gymnasium's message.</exception>
        /// <exception cref="ValueError">Thrown when the probabilities are refused by the generator (a negative <paramref name="p"/>), with NumPy's message.</exception>
        /// <exception cref="IndexError">Thrown for <paramref name="size"/> 0, where Python's <c>board[0][0]</c> fails.</exception>
        /// <remarks>
        ///     Each attempt is one <c>np_random.choice(["F", "H"], (size, size), p=[p, 1 - p])</c>: the same draws as Gymnasium's,
        ///     so a seed gives Gymnasium's map. Reference:
        ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/frozen_lake.py#L56">gymnasium/envs/toy_text/frozen_lake.py</see>.
        /// </remarks>
        public static IReadOnlyList<string> GenerateRandomMap(int size = 8, double p = 0.8, BigInteger? seed = null) {
            (Generator npRandom, _) = Seeding.NpRandom(seed);
            var board = Array.Empty<string>();
            bool valid = false;
            while (!valid) {
                // p = min(1, p): Python's min keeps 1 unless p is smaller, so NaN becomes 1.
                p = p < 1 ? p : 1;
                double[] weights = { p, 1 - p };
                using NDArray probabilities = np.array(weights);
                using NDArray tiles = npRandom.choice(2, new Shape(size, size), true, probabilities);
                if (size == 0)
                    throw new IndexError("index 0 is out of bounds for axis 0 with size 0");
                var cells = new char[size, size];
                for (int row = 0; row < size; row++)
                    for (int column = 0; column < size; column++)
                        cells[row, column] = tiles.GetAtIndex<long>(row * size + column) == 0 ? 'F' : 'H';
                cells[0, 0] = 'S';
                cells[size - 1, size - 1] = 'G';
                board = new string[size];
                for (int row = 0; row < size; row++) {
                    var line = new StringBuilder(size);
                    for (int column = 0; column < size; column++)
                        line.Append(cells[row, column]);
                    board[row] = line.ToString();
                }

                valid = IsValid(board, size);
            }

            return board;
        }

        /// <summary>
        ///     Gymnasium's module function <c>is_valid</c>: whether a depth-first walk from the top-left tile over tiles that are
        ///     not holes reaches a <c>G</c> inside the <paramref name="maxSize"/> square.
        /// </summary>
        /// <param name="board">The rows of the map.</param>
        /// <param name="maxSize">The square's side the walk stays within.</param>
        /// <returns>Whether a goal is reachable.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="board"/> is <see langword="null"/>.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/frozen_lake.py#L36">gymnasium/envs/toy_text/frozen_lake.py</see>.</remarks>
        public static bool IsValid(IReadOnlyList<string> board, int maxSize) {
            if (board is null)
                throw new ArgumentNullException(nameof(board));
            var frontier = new Stack<(int Row, int Column)>();
            var discovered = new HashSet<(int Row, int Column)>();
            frontier.Push((0, 0));
            (int, int)[] directions = { (1, 0), (0, 1), (-1, 0), (0, -1) };
            while (frontier.Count > 0) {
                (int r, int c) = frontier.Pop();
                if (!discovered.Add((r, c)))
                    continue;
                foreach ((int x, int y) in directions) {
                    int rNew = r + x, cNew = c + y;
                    if (rNew < 0 || rNew >= maxSize || cNew < 0 || cNew >= maxSize)
                        continue;
                    if (board[rNew][cNew] == 'G')
                        return true;
                    if (board[rNew][cNew] != 'H')
                        frontier.Push((rNew, cNew));
                }
            }

            return false;
        }

        /// <summary>
        ///     Starts an episode: reseeds the generator when a seed is given, then draws the start state from
        ///     <see cref="InitialStateDistrib"/> (one draw) and clears the last action. Renders in human mode.
        /// </summary>
        /// <param name="seed">The seed of a fresh <c>Generator(PCG64(SeedSequence(seed)))</c>, or <see langword="null"/> to keep the generator.</param>
        /// <param name="options">Ignored, as in Gymnasium.</param>
        /// <returns>The state as a 0-d int64 array, and <c>{"prob": 1}</c> (a Python int).</returns>
        /// <exception cref="Error">Thrown for a negative seed, with Gymnasium's message.</exception>
        /// <exception cref="DependencyNotInstalled">Thrown in human mode by the first render when no viewer factory is set (DR-20).</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/frozen_lake.py#L336">gymnasium/envs/toy_text/frozen_lake.py</see>.</remarks>
        public override ResetResult<NDArray> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            SeedNpRandom(seed);
            _s = ToyTextUtils.CategoricalSample(_initialStateDistrib, NpRandom);
            _hasState = true;
            LastAction = null;
            _lastActionIndex = null;

            if (RenderMode == "human")
                Render();
            return new ResetResult<NDArray>(NDArray.Scalar<long>(_s), new Info { ["prob"] = 1L });
        }

        /// <summary>
        ///     Moves by the action: looks up <c>P[s][a]</c>, draws one outcome (one float64 from the generator) and moves there.
        ///     Renders in human mode.
        /// </summary>
        /// <param name="action">A number equal to 0 (left), 1 (down), 2 (right) or 3 (up): a 0-d array, or through the untyped step any .NET number.</param>
        /// <returns>The new state (a 0-d int64 array), the reward, whether it is a goal or a hole, <c>truncated=False</c>, and <c>{"prob": p}</c>.</returns>
        /// <exception cref="AttributeError">Thrown before the first reset (<c>'FrozenLakeEnv' object has no attribute 's'</c>), before the action is looked at.</exception>
        /// <exception cref="KeyError">Thrown for an action no key equals, or an injected state the map lacks, with Python's message (the value's <c>repr</c>).</exception>
        /// <exception cref="TypeError">Thrown for an array action of one or more dimensions (<c>unhashable type: 'numpy.ndarray'</c>).</exception>
        /// <exception cref="DependencyNotInstalled">Thrown in human mode by the first render when no viewer factory is set (DR-20).</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/frozen_lake.py#L324">gymnasium/envs/toy_text/frozen_lake.py</see>.</remarks>
        public override StepResult<NDArray> Step(NDArray action) {
            // transitions = self.P[self.s][a]: the state first, then the action.
            if (!_hasState)
                throw new AttributeError("'FrozenLakeEnv' object has no attribute 's'");
            if (!_p.TryGetValue(_s, out IReadOnlyDictionary<long, IReadOnlyList<ToyTextTransition>>? byAction))
                throw new KeyError(PyBuiltins.Repr(_s));
            long key = PyKeys.DictKey(action, 4);
            IReadOnlyList<ToyTextTransition> transitions = byAction[key];
            var probabilities = new double[transitions.Count];
            for (int t = 0; t < probabilities.Length; t++)
                probabilities[t] = transitions[t].Probability;
            ToyTextTransition chosen = transitions[(int)ToyTextUtils.CategoricalSample(probabilities, NpRandom)];
            _s = chosen.NextState;
            LastAction = key;
            // self.lastaction = a: the action as given, which through the untyped step is the value before its conversion.
            object? given = ReferenceEquals(action, _convertedAction) ? _givenAction : action;
            _givenAction = null;
            _convertedAction = null;
            _lastActionIndex = DeferredListIndex(given);

            if (RenderMode == "human")
                Render();
            // truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`.
            return new StepResult<NDArray>(NDArray.Scalar<long>(chosen.NextState), chosen.Reward, chosen.Terminated, false, new Info { ["prob"] = chosen.Probability });
        }

        /// <summary>
        ///     Renders by the render mode: <c>ansi</c> returns the map as text with the agent in red; <c>human</c> and
        ///     <c>rgb_array</c> draw Gymnasium's pixel art (and <c>rgb_array</c> returns the frame).
        /// </summary>
        /// <returns><c>ansi</c>: the text. <c>rgb_array</c>: the <c>(H, W, 3)</c> uint8 frame, a new array the caller owns. <c>human</c>: nothing.</returns>
        /// <exception cref="AssertionError">
        ///     Thrown without a render mode when the env has no spec, and for a render mode other than <c>human</c>, <c>ansi</c>
        ///     and <c>rgb_array</c> (<c>Something went wrong with pygame. This should never happen.</c>), as in Gymnasium.
        /// </exception>
        /// <exception cref="AttributeError">Thrown before the first reset (no <c>s</c>; after the map was drawn, for images).</exception>
        /// <exception cref="TypeError">Thrown by <c>ansi</c> after an action Python's list indexing refuses (a float action).</exception>
        /// <exception cref="IndexError">Thrown by <c>ansi</c> after an action outside the caption list, or for an injected state outside the map.</exception>
        /// <exception cref="KeyError">Thrown in human mode when <c>metadata["render_fps"]</c> was removed.</exception>
        /// <exception cref="DependencyNotInstalled">Thrown on the first human render when no viewer factory is set (DR-20).</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/frozen_lake.py#L350">gymnasium/envs/toy_text/frozen_lake.py</see>.</remarks>
        public override RenderResult Render() {
            if (RenderMode is null) {
                EnvSpec spec = Spec ?? throw new AssertionError();
                Logger.Warn("You are calling render method without specifying any render mode. You can specify the render_mode at initialization, "
                            + $"e.g. gym.make(\"{spec.Id}\", render_mode=\"rgb_array\")");
                return RenderResult.None;
            }

            if (RenderMode == "ansi")
                return new RenderResult.Ansi(RenderText());
            // self.render_mode in {"human", "rgb_array"}
            return RenderGui(RenderMode);
        }

        /// <summary>Once a window surface exists, shuts the human-mode window down (pygame's <c>display.quit()</c>); the surface is kept.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/frozen_lake.py#L502">gymnasium/envs/toy_text/frozen_lake.py</see>.</remarks>
        public override void Close() {
            if (WindowSurface is not null)
                _window.Close();
        }

        /// <summary>
        ///     The untyped step's action conversion, with Python's order kept: the state's existence first, then the dict lookup
        ///     on the action as given, so the errors name the Python counterpart of the value given (<c>KeyError: '1'</c> for a
        ///     string).
        /// </summary>
        /// <param name="action">The action as given to the untyped step.</param>
        /// <returns>The action as an array: the one given, or a new one the step releases.</returns>
        /// <exception cref="AttributeError">Thrown before the first reset.</exception>
        /// <exception cref="KeyError">Thrown for an action no key equals.</exception>
        /// <exception cref="TypeError">Thrown for an array of one or more dimensions.</exception>
        protected override NDArray ConvertAction(object? action) {
            if (!_hasState)
                throw new AttributeError("'FrozenLakeEnv' object has no attribute 's'");
            PyKeys.DictKey(action, 4);
            NDArray converted = base.ConvertAction(action);
            _givenAction = action;
            _convertedAction = converted;
            return converted;
        }

        /// <summary>The state of a cell: <c>row * ncol + col</c>.</summary>
        /// <param name="row">The row.</param>
        /// <param name="column">The column.</param>
        /// <returns>The state.</returns>
        private long ToState(int row, int column) => (long)row * Ncol + column;

        /// <summary>
        ///     Gymnasium's <c>update_probability_matrix</c> for one move: where it leads (clamped to the map), whether that tile
        ///     ends the episode (<c>G</c> or <c>H</c>), and its reward (<c>reward_schedule</c> by <c>G</c>, <c>H</c>, anything
        ///     else).
        /// </summary>
        /// <param name="row">The row moved from.</param>
        /// <param name="column">The column moved from.</param>
        /// <param name="action">The move.</param>
        /// <param name="probability">The outcome's probability.</param>
        /// <param name="schedule">The reward schedule.</param>
        /// <returns>The outcome.</returns>
        private ToyTextTransition Outcome(int row, int column, int action, double probability, IReadOnlyList<double> schedule) {
            // inc(row, col, a)
            switch (action) {
                case (int)Left:
                    column = Math.Max(column - 1, 0);
                    break;
                case (int)Down:
                    row = Math.Min(row + 1, Nrow - 1);
                    break;
                case (int)Right:
                    column = Math.Min(column + 1, Ncol - 1);
                    break;
                case (int)Up:
                    row = Math.Max(row - 1, 0);
                    break;
            }

            char letter = _cells[row, column];
            bool terminated = letter is 'G' or 'H';
            // reward_schedule[b"GHF".index(new_letter if new_letter in b"GHF" else b"F")]
            double reward = schedule[letter switch {
                'G' => 0,
                'H' => 1,
                _ => 2,
            }];
            return new ToyTextTransition(probability, ToState(row, column), reward, terminated);
        }

        /// <summary>
        ///     What Python's <c>['Left', 'Down', 'Right', 'Up'][self.lastaction]</c> will give for an action: its caption
        ///     index, or the error the list indexing raises, deferred to the <c>ansi</c> render as in Python.
        /// </summary>
        /// <param name="action">The action as the step received it (only read now; it may be released after the step).</param>
        /// <returns>A function giving the index or throwing Python's error.</returns>
        /// <exception cref="TypeError">Thrown by the returned function, not by this method, for an action Python's list indexing refuses.</exception>
        /// <exception cref="IndexError">Thrown by the returned function, not by this method, for an action outside the caption list.</exception>
        private static Func<int> DeferredListIndex(object? action) {
            try {
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

        /// <summary>Gymnasium's <c>_render_text</c>: the caption of the last action (or an empty line), then the map with the agent in red.</summary>
        /// <returns>The text.</returns>
        /// <exception cref="AttributeError">Thrown before the first reset.</exception>
        /// <exception cref="TypeError">Thrown when the last action cannot index the caption list.</exception>
        /// <exception cref="IndexError">Thrown when the last action is outside the caption list, or the state outside the map.</exception>
        private string RenderText() {
            if (!_hasState)
                throw new AttributeError("'FrozenLakeEnv' object has no attribute 's'");
            long row = PyMath.FloorDiv(_s, Ncol), column = PyMath.Mod(_s, Ncol);
            var desc = new string[Nrow][];
            for (int r = 0; r < Nrow; r++) {
                desc[r] = new string[Ncol];
                for (int c = 0; c < Ncol; c++)
                    desc[r][c] = _cells[r, c].ToString();
            }

            // desc[row][col]: a Python list, so a negative row (an injected negative state) counts from the end.
            if (row < 0)
                row += Nrow;
            if (row < 0 || row >= Nrow)
                throw new IndexError("list index out of range");
            desc[row][column] = Logger.Colorize(desc[row][column], "red", highlight: true);
            var outfile = new StringBuilder();
            if (_lastActionIndex is not null)
                outfile.Append("  (").Append(ActionNames[_lastActionIndex()]).Append(")\n");
            else
                outfile.Append('\n');
            outfile.Append(string.Join("\n", desc.Select(line => string.Concat(line)))).Append('\n');
            return outfile.ToString();
        }

        /// <summary>Gymnasium's <c>_render_gui</c>: the tiles, overlays and grid, then the agent unless it is in a hole.</summary>
        /// <param name="mode">The render mode.</param>
        /// <returns>The frame for <c>rgb_array</c>, nothing for <c>human</c>.</returns>
        /// <exception cref="AssertionError">Thrown for a mode other than <c>human</c> and <c>rgb_array</c>, as Gymnasium's assert does.</exception>
        /// <exception cref="AttributeError">Thrown before the first reset, after the map was drawn.</exception>
        /// <exception cref="IndexError">Thrown for an injected state beyond the map's rows, as Python's <c>desc[bot_row]</c> fails.</exception>
        /// <exception cref="KeyError">Thrown in human mode when <c>metadata["render_fps"]</c> was removed.</exception>
        /// <exception cref="DependencyNotInstalled">Thrown on the first human render when no viewer factory is set.</exception>
        private RenderResult RenderGui(string mode) {
            if (WindowSurface is null) {
                if (mode == "human") {
                    // pygame.display.set_caption("Frozen Lake"); self.window_surface = pygame.display.set_mode(self.window_size)
                    _window.Ensure(WindowSize.Width, WindowSize.Height);
                    WindowSurface = new Surface(WindowSize.Width, WindowSize.Height);
                } else if (mode == "rgb_array") {
                    WindowSurface = new Surface(WindowSize.Width, WindowSize.Height);
                }
            }

            Surface surface = WindowSurface ?? throw new AssertionError("Something went wrong with pygame. This should never happen.");
            Clock ??= new Clock();

            if (IceImg is null) {
                IceImg = LoadTile("tile_ice.png");
                HoleImg = LoadTile("tile_hole.png");
                IceCrackImgs = new[] { LoadTile("tile_ice_crack_a.png"), LoadTile("tile_ice_crack_b.png") };
                AgentImg = LoadOverlay("agent.png", 40, 36);
                GoalFlagImg = LoadOverlay("goal_flag.png", 24, 32);
                LetterSImg = LoadOverlay("letter_s.png", 12, 20);
                LetterGImg = LoadOverlay("letter_g.png", 12, 20);
            }

            for (int y = 0; y < Nrow; y++) {
                for (int x = 0; x < Ncol; x++) {
                    int posX = x * CellSize.Width, posY = y * CellSize.Height;
                    char tile = _cells[y, x];
                    if (tile == 'H') {
                        surface.Blit(HoleImg!, posX, posY);
                    } else {
                        surface.Blit(IceImg, posX, posY);
                        if (tile == 'F') {
                            // scatter subtle cracks on some frozen tiles; deterministic by position so the texture stays stable across frames
                            int crack = (x * 3 + y * 5) % 7;
                            if (crack < IceCrackImgs!.Count)
                                surface.Blit(IceCrackImgs[crack], posX, posY);
                        } else if (tile == 'S') {
                            BlitOverlay(surface, LetterSImg!, x, y, "corner");
                        } else if (tile == 'G') {
                            BlitOverlay(surface, GoalFlagImg!, x, y, "flag");
                            BlitOverlay(surface, LetterGImg!, x, y, "corner");
                        }
                    }

                    Draw.Rect(surface, new Color(180, 200, 230), posX, posY, CellSize.Width, CellSize.Height, 1);
                }
            }

            // paint the agent (single sprite, no facing direction); if it fell in a hole the hole tile already covers the cell
            if (!_hasState)
                throw new AttributeError("'FrozenLakeEnv' object has no attribute 's'");
            long botRow = PyMath.FloorDiv(_s, Ncol), botCol = PyMath.Mod(_s, Ncol);
            // desc[bot_row][bot_col] indexes a Python list (a negative row counts from the end); the overlay is still placed
            // at the unwrapped row, as Python passes bot_row on unchanged.
            long lookupRow = botRow < 0 ? botRow + Nrow : botRow;
            if (lookupRow < 0 || lookupRow >= Nrow)
                throw new IndexError("list index out of range");
            if (_cells[lookupRow, botCol] != 'H')
                BlitOverlay(surface, AgentImg!, (int)botCol, (int)botRow, "center");

            if (mode == "human") {
                // pygame.event.pump(); pygame.display.update(); self.clock.tick(self.metadata["render_fps"])
                _window.Show(surface);
                Clock.Tick(PyBuiltins.Float(Metadata[Metadata.RenderFpsKey]));
                return RenderResult.None;
            }

            return new RenderResult.Rgb(SurfArray.ToRgbArray(surface));
        }

        /// <summary>Gymnasium's <c>load_tile</c>: an image scaled to fill a cell.</summary>
        /// <param name="name">The image under <c>toy_text</c>.</param>
        /// <returns>A new surface.</returns>
        private Surface LoadTile(string name) => Transform.Scale(EnvAssets.Load("toy_text/" + name), CellSize.Width, CellSize.Height);

        /// <summary>
        ///     Gymnasium's <c>load_overlay</c>: an image scaled from its art size by the cell's smaller side over 64, each side
        ///     <c>max(1, round(...))</c> with Python's rounding (half to even).
        /// </summary>
        /// <param name="name">The image under <c>toy_text</c>.</param>
        /// <param name="nativeWidth">The art's width, as Gymnasium's call states it.</param>
        /// <param name="nativeHeight">The art's height.</param>
        /// <returns>A new surface.</returns>
        private Surface LoadOverlay(string name, int nativeWidth, int nativeHeight) {
            double scale = Math.Min(CellSize.Width, CellSize.Height) / 64.0;
            int width = (int)BigInteger.Max(1, PyMath.Round(nativeWidth * scale));
            int height = (int)BigInteger.Max(1, PyMath.Round(nativeHeight * scale));
            return Transform.Scale(EnvAssets.Load("toy_text/" + name), width, height);
        }

        /// <summary>Gymnasium's <c>blit_overlay</c>: an overlay at a tile's corner, planted on it as a flag, or centred, at truncated positions.</summary>
        /// <param name="surface">The frame.</param>
        /// <param name="img">The overlay.</param>
        /// <param name="column">The tile's column.</param>
        /// <param name="row">The tile's row.</param>
        /// <param name="anchor"><c>corner</c>, <c>flag</c>, or anything else for centred.</param>
        private void BlitOverlay(Surface surface, Surface img, int column, int row, string anchor) {
            int cx = column * CellSize.Width, cy = row * CellSize.Height;
            int w = img.Width, h = img.Height;
            if (anchor == "corner") {
                int margin = (int)PyMath.Round(CellSize.Width * 0.06);
                surface.Blit(img, cx + margin, cy + margin);
            } else if (anchor == "flag") {
                // planted, bottom-centered on the tile; pygame truncates the float x.
                double x = cx + (CellSize.Width - w) / 2.0;
                int y = cy + CellSize.Height - h - (int)PyMath.Round(CellSize.Height * 0.12);
                surface.Blit(img, RenderMath.CInt(x), y);
            } else {
                // centered on the tile
                double x = cx + (CellSize.Width - w) / 2.0, y = cy + (CellSize.Height - h) / 2.0;
                surface.Blit(img, RenderMath.CInt(x), RenderMath.CInt(y));
            }
        }
    }
}
