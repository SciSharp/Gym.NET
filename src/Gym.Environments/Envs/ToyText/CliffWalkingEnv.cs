#nullable enable
using System;
using System.Collections.Generic;
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

namespace Gym.Environments.Envs.ToyText {
    /// <summary>
    ///     CliffWalking-v1 and CliffWalkingSlippery-v1: a 4x12 grid walked from the bottom-left corner to the bottom-right one
    ///     along a cliff; stepping off it costs -100 and returns the agent to the start, every other step costs -1. Slippery
    ///     moves go perpendicular with probability 2/3. Byte for byte what Gymnasium's CliffWalking computes: the transition
    ///     table <see cref="P"/>, the generator draws, observations, rewards, <c>{"prob": p}</c> infos, <c>ansi</c> text and
    ///     <c>rgb_array</c> pixels.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Like Gymnasium's, the env checks no action against its space: <c>self.P[self.s][a]</c> is a dict lookup, so any
    ///     number equal to 0 (up), 1 (right), 2 (down) or 3 (left) is an action, anything else raises <c>KeyError</c>. Each
    ///     step draws one float64 from the env's generator, even on a certain move. The goal is not absorbing: a step from it
    ///     moves on, as in Gymnasium. Observations are 0-d int64 arrays (DR-08); rewards are Gymnasium's -1 and -100 as
    ///     float64.
    ///     </para>
    ///     <para>
    ///     Rendering is Gymnasium's pixel art at 64-pixel cells (its art scale is 1, so the images keep their sizes);
    ///     <c>ansi</c> marks the agent <c>x</c>, the goal <c>T</c>, the cliff <c>C</c>. Any render mode other than
    ///     <c>human</c> and <c>ansi</c> draws and returns the frame, as Gymnasium's <c>else</c> branches do. Human mode renders
    ///     into the window from <c>viewer_factory</c> or <c>Gymnasium.HumanViewerFactory</c> (DR-20) at 4 frames per second.
    ///     Not thread-safe.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/cliffwalking.py#L28">gymnasium/envs/toy_text/cliffwalking.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class CliffWalkingEnv : Env<NDArray, NDArray> {
        /// <summary>Gymnasium's <c>UP</c>, action 0.</summary>
        public const long Up = 0;

        /// <summary>Gymnasium's <c>RIGHT</c>, action 1.</summary>
        public const long Right = 1;

        /// <summary>Gymnasium's <c>DOWN</c>, action 2.</summary>
        public const long Down = 2;

        /// <summary>Gymnasium's <c>LEFT</c>, action 3.</summary>
        public const long Left = 3;

        /// <summary>Gymnasium's <c>ART_CELL_SIZE</c>: the art is authored against a 64x64 cell.</summary>
        private const int ArtCellSize = 64;

        /// <summary>Gymnasium's <c>POSITION_MAPPING</c>: each action's <c>(row, column)</c> step.</summary>
        private static readonly (int Row, int Column)[] PositionMapping = { (-1, 0), (0, 1), (1, 0), (0, -1) };

        /// <summary>Gymnasium's <c>LETTER_OFFSET</c>, <c>FLAG_OFFSET</c> and <c>AGENT_OFFSET</c>: the sprites' places within a 64-pixel cell.</summary>
        private static readonly (int X, int Y) LetterOffset = (8, 8), FlagOffset = (28, 16), AgentOffset = (20, 20);

        /// <summary>The human-mode window, opened on the first human render.</summary>
        private readonly HumanWindow _window;

        /// <summary>The transition table, by state then action.</summary>
        private readonly Dictionary<long, IReadOnlyDictionary<long, IReadOnlyList<ToyTextTransition>>> _p;

        /// <summary>Gymnasium's <c>_cliff</c>: which cells of the grid are the cliff (row 3, columns 1 to 10).</summary>
        private readonly bool[,] _cliff;

        /// <summary>Python's <c>self.s</c>, meaningful only while <see cref="_hasState"/>.</summary>
        private long _s;

        /// <summary>Whether Python's <c>self.s</c> exists: set by the first reset or an assignment of <see cref="S"/>.</summary>
        private bool _hasState;

        /// <summary>Creates the env.</summary>
        /// <param name="renderMode">Gymnasium's <c>render_mode</c>: <see langword="null"/>, <c>"human"</c>, <c>"rgb_array"</c> or <c>"ansi"</c>.</param>
        /// <param name="isSlippery">Gymnasium's <c>is_slippery</c>: each move goes as intended or to either side, with probability 1/3 each.</param>
        /// <param name="viewerFactory">Gym.NET's <c>viewer_factory</c> (DR-20) for human mode; <see langword="null"/> uses <c>Gymnasium.HumanViewerFactory</c>.</param>
        public CliffWalkingEnv(string? renderMode = null, bool isSlippery = false, IEnvironmentViewerFactoryDelegate? viewerFactory = null) {
            Metadata = ClassMetadata;
            IsSlippery = isSlippery;

            // Cliff Location
            _cliff = new bool[Rows, Columns];
            for (int column = 1; column < Columns - 1; column++)
                _cliff[3, column] = true;

            // Calculate transition probabilities and rewards
            _p = new Dictionary<long, IReadOnlyDictionary<long, IReadOnlyList<ToyTextTransition>>>(NS);
            for (long s = 0; s < NS; s++) {
                (int row, int column) = ((int)(s / Columns), (int)(s % Columns));
                var byAction = new Dictionary<long, IReadOnlyList<ToyTextTransition>>(NA);
                foreach (long move in new[] { Up, Right, Down, Left })
                    byAction[move] = CalculateTransitionProb(row, column, (int)move);
                _p[s] = byAction;
            }

            // Calculate initial state distribution: we always start in state (3, 0).
            var distribution = new double[NS];
            distribution[StartStateIndex] = 1.0;
            _initialStateDistrib = distribution;

            ObservationSpace = new Discrete(NS);
            ActionSpace = new Discrete(NA);
            RenderMode = renderMode;
            _window = new HumanWindow("CliffWalking", viewerFactory);
        }

        /// <summary>
        ///     Gymnasium's class attribute <c>metadata</c>: <c>{"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 4}</c>.
        ///     Every env starts with a copy as its instance metadata.
        /// </summary>
        /// <remarks>A new dict on every read, so editing one env's metadata never reaches another (Python's is one shared dict).</remarks>
        public static new Metadata ClassMetadata => new() {
            [Metadata.RenderModesKey] = new List<string> { "human", "rgb_array", "ansi" },
            [Metadata.RenderFpsKey] = 4L,
        };

        /// <summary>The grid's rows, <c>shape[0]</c>.</summary>
        private const int Rows = 4;

        /// <summary>The grid's columns, <c>shape[1]</c>.</summary>
        private const int Columns = 12;

        /// <summary>Gymnasium's <c>shape</c>: <c>(4, 12)</c>.</summary>
        public (int Rows, int Columns) Shape => (Rows, Columns);

        /// <summary>Gymnasium's <c>start_state_index</c>: 36, the cell <c>(3, 0)</c>.</summary>
        public long StartStateIndex => 3 * Columns;

        /// <summary>Gymnasium's <c>nS</c>: 48 states.</summary>
        [PyName("nS")]
        public int NS => Rows * Columns;

        /// <summary>Gymnasium's <c>nA</c>: 4 actions.</summary>
        [PyName("nA")]
        public int NA => 4;

        /// <summary>Gymnasium's <c>is_slippery</c>, fixed at construction.</summary>
        public bool IsSlippery { get; }

        /// <summary>Gymnasium's <c>initial_state_distrib</c>: 1 at <see cref="StartStateIndex"/>, a new array on every read.</summary>
        public double[] InitialStateDistrib => (double[])_initialStateDistrib.Clone();

        /// <summary>The distribution <see cref="InitialStateDistrib"/> copies.</summary>
        private readonly double[] _initialStateDistrib;

        /// <summary>
        ///     Gymnasium's <c>P</c>: for each state and action, the possible outcomes <c>(probability, next state, reward,
        ///     terminated)</c> in Gymnasium's order (a slippery move: counter-clockwise, as intended, clockwise).
        /// </summary>
        [PyName("P")]
        public IReadOnlyDictionary<long, IReadOnlyDictionary<long, IReadOnlyList<ToyTextTransition>>> P => _p;

        /// <summary>
        ///     Gymnasium's <c>s</c>: the agent's state, <c>row * 12 + col</c>; <see langword="null"/> before the first reset.
        ///     Assigning one injects a state (one outside the grid makes the next step's lookup fail with <c>KeyError</c>).
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

        /// <summary>Gymnasium's <c>cell_size</c>: <c>(64, 64)</c> pixels.</summary>
        public (int Width, int Height) CellSize { get; } = (64, 64);

        /// <summary>Gymnasium's <c>window_size</c>: <c>(768, 256)</c> pixels, twelve cells by four.</summary>
        public (int Width, int Height) WindowSize => (Columns * CellSize.Height, Rows * CellSize.Width);

        /// <summary>Gymnasium's <c>window_surface</c>: the frame every image render draws on, <see langword="null"/> until the first one.</summary>
        public Surface? WindowSurface { get; private set; }

        /// <summary>Gymnasium's <c>clock</c>: paces human rendering, <see langword="null"/> until the first image render.</summary>
        public Clock? Clock { get; private set; }

        /// <summary>Gymnasium's <c>ground_img</c>; <see langword="null"/> until the first image render.</summary>
        public Surface? GroundImg { get; private set; }

        /// <summary>Gymnasium's <c>cliff_imgs</c>: the cliff tiles by name (<c>left</c>, <c>mid_a</c>, <c>mid_b</c>, <c>right</c>).</summary>
        public IReadOnlyDictionary<string, Surface>? CliffImgs { get; private set; }

        /// <summary>Gymnasium's <c>start_img</c>, the letter S.</summary>
        public Surface? StartImg { get; private set; }

        /// <summary>Gymnasium's <c>goal_img</c>, the letter G.</summary>
        public Surface? GoalImg { get; private set; }

        /// <summary>Gymnasium's <c>goal_flag_img</c>.</summary>
        public Surface? GoalFlagImg { get; private set; }

        /// <summary>Gymnasium's <c>agent_img</c>.</summary>
        public Surface? AgentImg { get; private set; }

        /// <summary>
        ///     Starts an episode: reseeds the generator when a seed is given, then draws the start state (one draw, always state
        ///     36) and clears the last action. Renders in human mode.
        /// </summary>
        /// <param name="seed">The seed of a fresh <c>Generator(PCG64(SeedSequence(seed)))</c>, or <see langword="null"/> to keep the generator.</param>
        /// <param name="options">Ignored, as in Gymnasium.</param>
        /// <returns>The state as a 0-d int64 array, and <c>{"prob": 1}</c> (a Python int).</returns>
        /// <exception cref="Error">Thrown for a negative seed, with Gymnasium's message.</exception>
        /// <exception cref="DependencyNotInstalled">Thrown in human mode by the first render when no viewer factory is set (DR-20).</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/cliffwalking.py#L211">gymnasium/envs/toy_text/cliffwalking.py</see>.</remarks>
        public override ResetResult<NDArray> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            SeedNpRandom(seed);
            _s = ToyTextUtils.CategoricalSample(_initialStateDistrib, NpRandom);
            _hasState = true;
            LastAction = null;

            if (RenderMode == "human")
                Render();
            return new ResetResult<NDArray>(NDArray.Scalar<long>(_s), new Info { ["prob"] = 1L });
        }

        /// <summary>
        ///     Moves by the action: looks up <c>P[s][a]</c>, draws one outcome (one float64 from the generator) and moves there.
        ///     Renders in human mode.
        /// </summary>
        /// <param name="action">A number equal to 0 (up), 1 (right), 2 (down) or 3 (left): a 0-d array, or through the untyped step any .NET number.</param>
        /// <returns>The new state (a 0-d int64 array), the reward (-1, or -100 off the cliff), whether it is the goal, <c>truncated=False</c>, and <c>{"prob": p}</c>.</returns>
        /// <exception cref="AttributeError">Thrown before the first reset (<c>'CliffWalkingEnv' object has no attribute 's'</c>), before the action is looked at.</exception>
        /// <exception cref="KeyError">Thrown for an action no key equals, or an injected state outside the grid, with Python's message.</exception>
        /// <exception cref="TypeError">Thrown for an array action of one or more dimensions (<c>unhashable type: 'numpy.ndarray'</c>).</exception>
        /// <exception cref="DependencyNotInstalled">Thrown in human mode by the first render when no viewer factory is set (DR-20).</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/cliffwalking.py#L199">gymnasium/envs/toy_text/cliffwalking.py</see>.</remarks>
        public override StepResult<NDArray> Step(NDArray action) {
            // transitions = self.P[self.s][a]: the state first, then the action.
            if (!_hasState)
                throw new AttributeError("'CliffWalkingEnv' object has no attribute 's'");
            if (!_p.TryGetValue(_s, out IReadOnlyDictionary<long, IReadOnlyList<ToyTextTransition>>? byAction))
                throw new KeyError(PyBuiltins.Repr(_s));
            long key = PyKeys.DictKey(action, NA);
            IReadOnlyList<ToyTextTransition> transitions = byAction[key];
            var probabilities = new double[transitions.Count];
            for (int t = 0; t < probabilities.Length; t++)
                probabilities[t] = transitions[t].Probability;
            ToyTextTransition chosen = transitions[(int)ToyTextUtils.CategoricalSample(probabilities, NpRandom)];
            _s = chosen.NextState;
            LastAction = key;

            if (RenderMode == "human")
                Render();
            // truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`.
            return new StepResult<NDArray>(NDArray.Scalar<long>(chosen.NextState), chosen.Reward, chosen.Terminated, false, new Info { ["prob"] = chosen.Probability });
        }

        /// <summary>
        ///     Renders by the render mode: <c>ansi</c> returns the grid as text; any other mode draws Gymnasium's pixel art
        ///     (and returns the frame unless the mode is <c>human</c>).
        /// </summary>
        /// <returns><c>ansi</c>: the text. <c>human</c>: nothing. Any other mode: the <c>(256, 768, 3)</c> uint8 frame, a new array the caller owns.</returns>
        /// <exception cref="AssertionError">Thrown without a render mode when the env has no spec (<c>assert self.spec is not None</c>).</exception>
        /// <exception cref="AttributeError">Thrown before the first reset (no <c>s</c>; for images after the first cell's ground was drawn).</exception>
        /// <exception cref="KeyError">Thrown in human mode when <c>metadata["render_fps"]</c> was removed.</exception>
        /// <exception cref="DependencyNotInstalled">Thrown on the first human render when no viewer factory is set (DR-20).</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/cliffwalking.py#L220">gymnasium/envs/toy_text/cliffwalking.py</see>.</remarks>
        public override RenderResult Render() {
            if (RenderMode is null) {
                EnvSpec spec = Spec ?? throw new AssertionError();
                Logger.Warn("You are calling render method without specifying any render mode. You can specify the render_mode at initialization, "
                            + $"e.g. gym.make(\"{spec.Id}\", render_mode=\"rgb_array\")");
                return RenderResult.None;
            }

            return RenderMode == "ansi" ? new RenderResult.Ansi(RenderText()) : RenderGui(RenderMode);
        }

        /// <summary>Once a window surface exists, shuts the human-mode window down (pygame's <c>display.quit()</c>); the surface is kept.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/cliffwalking.py#L344">gymnasium/envs/toy_text/cliffwalking.py</see>.</remarks>
        public override void Close() {
            if (WindowSurface is not null)
                _window.Close();
        }

        /// <summary>
        ///     The untyped step's action conversion, with Python's order kept: the state's existence first, then the dict lookup
        ///     on the action as given, so the errors name the Python counterpart of the value given.
        /// </summary>
        /// <param name="action">The action as given to the untyped step.</param>
        /// <returns>The action as an array: the one given, or a new one the step releases.</returns>
        /// <exception cref="AttributeError">Thrown before the first reset.</exception>
        /// <exception cref="KeyError">Thrown for an action no key equals.</exception>
        /// <exception cref="TypeError">Thrown for an array of one or more dimensions.</exception>
        protected override NDArray ConvertAction(object? action) {
            if (!_hasState)
                throw new AttributeError("'CliffWalkingEnv' object has no attribute 's'");
            PyKeys.DictKey(action, NA);
            return base.ConvertAction(action);
        }

        /// <summary>
        ///     Gymnasium's <c>_calculate_transition_prob</c>: the outcomes of a move from a cell, each step clamped to the grid; a
        ///     step onto the cliff returns to the start with -100, any other costs -1 and ends the episode at the goal.
        /// </summary>
        /// <param name="row">The row moved from.</param>
        /// <param name="column">The column moved from.</param>
        /// <param name="move">The action.</param>
        /// <returns>One outcome, or three for a slippery move (counter-clockwise, as intended, clockwise), each of probability <c>1 / count</c>.</returns>
        private IReadOnlyList<ToyTextTransition> CalculateTransitionProb(int row, int column, int move) {
            int[] moves = IsSlippery ? new[] { (int)PyMath.Mod(move - 1, 4), move, (int)PyMath.Mod(move + 1, 4) } : new[] { move };
            var outcomes = new List<ToyTextTransition>(moves.Length);
            // 1 / len(deltas): a Python float.
            double probability = 1.0 / moves.Length;
            foreach (int act in moves) {
                (int dr, int dc) = PositionMapping[act];
                // _limit_coordinates: min then max on each coordinate.
                int newRow = Math.Max(Math.Min(row + dr, Rows - 1), 0);
                int newColumn = Math.Max(Math.Min(column + dc, Columns - 1), 0);
                long newState = (long)newRow * Columns + newColumn;
                if (_cliff[newRow, newColumn]) {
                    outcomes.Add(new ToyTextTransition(probability, StartStateIndex, -100, false));
                } else {
                    bool isTerminated = newRow == Rows - 1 && newColumn == Columns - 1;
                    outcomes.Add(new ToyTextTransition(probability, newState, -1, isTerminated));
                }
            }

            return outcomes;
        }

        /// <summary>Gymnasium's <c>_render_text</c>: the grid row by row, <c>x</c> for the agent, <c>T</c> the goal, <c>C</c> the cliff, <c>o</c> the rest.</summary>
        /// <returns>The text.</returns>
        /// <exception cref="AttributeError">Thrown before the first reset.</exception>
        private string RenderText() {
            var outfile = new StringBuilder();
            for (long s = 0; s < NS; s++) {
                (int row, int column) = ((int)(s / Columns), (int)(s % Columns));
                if (!_hasState)
                    throw new AttributeError("'CliffWalkingEnv' object has no attribute 's'");
                string output;
                if (_s == s)
                    output = " x ";
                // Print terminal state
                else if (row == 3 && column == 11)
                    output = " T ";
                else if (_cliff[row, column])
                    output = " C ";
                else
                    output = " o ";

                if (column == 0)
                    output = output.TrimStart();
                if (column == Columns - 1) {
                    output = output.TrimEnd();
                    output += "\n";
                }

                outfile.Append(output);
            }

            outfile.Append('\n');
            return outfile.ToString();
        }

        /// <summary>Gymnasium's <c>_render_gui</c>: ground, cliff tiles, the start and goal letters, the flag and the agent.</summary>
        /// <param name="mode">The render mode.</param>
        /// <returns>Nothing for <c>human</c>, the frame for any other mode.</returns>
        /// <exception cref="AttributeError">Thrown before the first reset, after the first cell's ground was drawn.</exception>
        /// <exception cref="KeyError">Thrown in human mode when <c>metadata["render_fps"]</c> was removed.</exception>
        /// <exception cref="DependencyNotInstalled">Thrown on the first human render when no viewer factory is set.</exception>
        private RenderResult RenderGui(string mode) {
            if (WindowSurface is null) {
                if (mode == "human")
                    // pygame.display.set_caption("CliffWalking"); pygame.display.set_mode(self.window_size)
                    _window.Ensure(WindowSize.Width, WindowSize.Height);
                WindowSurface = new Surface(WindowSize.Width, WindowSize.Height);
            }

            Clock ??= new Clock();
            double scale = (double)CellSize.Width / ArtCellSize;

            GroundImg ??= Load("tile_ground.png", scale);
            CliffImgs ??= new Dictionary<string, Surface>(StringComparer.Ordinal) {
                ["left"] = Load("tile_cliff_left.png", scale),
                ["mid_a"] = Load("tile_cliff_mid_a.png", scale),
                ["mid_b"] = Load("tile_cliff_mid_b.png", scale),
                ["right"] = Load("tile_cliff_right.png", scale),
            };
            StartImg ??= Load("letter_s.png", scale);
            GoalImg ??= Load("letter_g.png", scale);
            GoalFlagImg ??= Load("goal_flag.png", scale);
            AgentImg ??= Load("agent.png", scale);

            // np.flatnonzero(self._cliff.any(axis=0)): the cliff's first and last columns.
            const int firstCliffColumn = 1, lastCliffColumn = Columns - 2;
            (int letterDx, int letterDy) = (Rounded(LetterOffset.X * scale), Rounded(LetterOffset.Y * scale));
            (int flagDx, int flagDy) = (Rounded(FlagOffset.X * scale), Rounded(FlagOffset.Y * scale));
            (int agentDx, int agentDy) = (Rounded(AgentOffset.X * scale), Rounded(AgentOffset.Y * scale));

            Surface surface = WindowSurface;
            for (long s = 0; s < NS; s++) {
                (int row, int column) = ((int)(s / Columns), (int)(s % Columns));
                (int posX, int posY) = (column * CellSize.Width, row * CellSize.Height);
                surface.Blit(GroundImg, posX, posY);

                if (_cliff[row, column]) {
                    Surface tile = column == firstCliffColumn ? CliffImgs["left"]
                        : column == lastCliffColumn ? CliffImgs["right"]
                        : CliffImgs[column % 2 == 0 ? "mid_a" : "mid_b"];
                    surface.Blit(tile, posX, posY);
                }

                (int letterX, int letterY) = (posX + letterDx, posY + letterDy);
                if (s == StartStateIndex)
                    surface.Blit(StartImg, letterX, letterY);
                if (s == NS - 1) {
                    surface.Blit(GoalImg, letterX, letterY);
                    surface.Blit(GoalFlagImg, posX + flagDx, posY + flagDy);
                }

                if (!_hasState)
                    throw new AttributeError("'CliffWalkingEnv' object has no attribute 's'");
                if (s == _s)
                    surface.Blit(AgentImg, posX + agentDx, posY + agentDy);
            }

            if (mode == "human") {
                // pygame.event.pump(); pygame.display.update(); self.clock.tick(self.metadata["render_fps"])
                _window.Show(surface);
                Clock.Tick(PyBuiltins.Float(Metadata[Metadata.RenderFpsKey]));
                return RenderResult.None;
            }

            // else:  # rgb_array
            return new RenderResult.Rgb(SurfArray.ToRgbArray(surface));
        }

        /// <summary>Gymnasium's <c>load</c>: an image of <c>cliff_walking</c> scaled by the art scale, each side rounded as Python rounds.</summary>
        /// <param name="name">The image's file name.</param>
        /// <param name="scale">The cell size over the art's 64 pixels.</param>
        /// <returns>A new surface.</returns>
        private static Surface Load(string name, double scale) {
            Surface img = EnvAssets.Load("toy_text/cliff_walking/" + name);
            return Transform.Scale(img, Rounded(img.Width * scale), Rounded(img.Height * scale));
        }

        /// <summary>Python's <c>round(x)</c> (half to even) as an int.</summary>
        /// <param name="x">The value.</param>
        /// <returns>The rounded value.</returns>
        private static int Rounded(double x) => (int)PyMath.Round(x);
    }
}
