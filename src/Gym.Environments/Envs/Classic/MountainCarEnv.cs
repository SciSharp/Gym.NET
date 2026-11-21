#nullable enable
using System;
using System.Collections.Generic;
using System.Numerics;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Exceptions;
using Gym.Observations;
using Gym.Rendering;
using Gym.Spaces;
using Gym.Utils;
using NumSharp;
using Color = Gym.Rendering.Color;
using Vector2 = Gym.Rendering.Vector2;

namespace Gym.Environments.Envs.Classic {
    /// <summary>
    ///     MountainCar-v0 and MountainCarContinuous-v0 in one class (DR-32): a car in a valley that must build momentum to
    ///     reach the flag on the right hill. Discrete mode pushes left, not at all or right (actions 0, 1, 2) and pays -1 per
    ///     step; continuous mode applies a force in [-1, 1] and pays 100 at the flag minus 0.1 times the squared force. Byte
    ///     for byte what Gymnasium's two classes compute, errors included.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The continuous mode is NumPy-typed in a way that reaches the last bit: its state is float64 after a reset and
    ///     float32 after a step, the force is the action's own element or a Python float when clamped, and each operation's
    ///     precision follows NEP 50 (a float32 state with a Python float stays float32, so <c>position &gt;= 0.45</c> is
    ///     decided in float32). The state is kept as the scalars Python holds (<see cref="NpScalar"/>), so every step computes
    ///     in the precision Gymnasium's does. The discrete mode computes in float64 throughout, with the action's own integer
    ///     type: an unsigned action 0 wraps in <c>action - 1</c> (a <c>uint8</c> gives 255) and drives the car right, as in
    ///     Python.
    ///     </para>
    ///     <para>
    ///     It prints as Gymnasium's class of its mode (<c>&lt;Continuous_MountainCarEnv instance&gt;</c> in continuous mode),
    ///     and its errors name that class. Accepted deviations (DR-32): the constructor takes <c>continuous</c>, which
    ///     Gymnasium's classes do not, and <c>MountainCarContinuous-v0</c>'s <c>EnvSpec.kwargs</c> carries
    ///     <c>continuous=True</c>. Both modes' attributes exist in both modes (<see cref="Force"/> in continuous mode,
    ///     <see cref="Power"/> in discrete mode), unused by the other. Gymnasium's <c>low</c>/<c>high</c> (discrete) and
    ///     <c>low_state</c>/<c>high_state</c> (continuous) are the observation space's bounds and are read there.
    ///     <c>get_keys_to_action</c> exists in discrete mode only, as in Gymnasium (<see cref="GetKeysToAction"/>).
    ///     </para>
    ///     <para>
    ///     Rendering is Gymnasium's, pixel for pixel: the gfxdraw car, wheels and flag, and the hill, drawn with
    ///     <see cref="Draw.Aalines"/> (pygame-ce's own algorithm, DR-37). Human mode renders into the window from
    ///     <c>viewer_factory</c> or <c>Gymnasium.HumanViewerFactory</c> (DR-20), paced at 30 frames per second.
    ///     </para>
    ///     <para>
    ///     Not thread-safe. References: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/mountain_car.py#L16">gymnasium/envs/classic_control/mountain_car.py</see>,
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/continuous_mountain_car.py#L26">gymnasium/envs/classic_control/continuous_mountain_car.py</see>.
    ///     </para>
    ///     <para>Its info is always <c>{}</c>, as Gymnasium's, so it has no typed info record (DR-95).</para>
    /// </remarks>
    [GymOptions]
    public partial class MountainCarEnv : Env<NDArray, NDArray> {
        /// <summary>The human-mode window, opened on the first human render.</summary>
        private readonly HumanWindow _window;

        /// <summary>
        ///     Python's <c>self.state</c> as the scalars it holds; <see langword="null"/> when it is <c>None</c> (assigned) or
        ///     missing (before the first reset, told apart by <see cref="_hasState"/>).
        /// </summary>
        private NpScalar[]? _state;

        /// <summary>Whether Python's <c>self.state</c> attribute exists: set by the first reset or an assignment of <see cref="State"/>.</summary>
        private bool _hasState;

        /// <summary>Creates the env in either mode.</summary>
        /// <param name="renderMode">
        ///     Gymnasium's <c>render_mode</c>: <see langword="null"/>, <c>"human"</c> or <c>"rgb_array"</c>; not validated, as in
        ///     Gymnasium (<c>make</c> checks it against the metadata).
        /// </param>
        /// <param name="goalVelocity">Gymnasium's <c>goal_velocity</c>: the velocity the car must have at the flag (0 by default).</param>
        /// <param name="continuous">
        ///     Gym.NET's <c>continuous</c> (DR-32): MountainCarContinuous-v0's force-in-[-1, 1] dynamics, flag at 0.45 and
        ///     reward instead of MountainCar-v0's.
        /// </param>
        /// <param name="viewerFactory">Gym.NET's <c>viewer_factory</c> (DR-20) for human mode; <see langword="null"/> uses <c>Gymnasium.HumanViewerFactory</c>.</param>
        public MountainCarEnv(string? renderMode = null, double goalVelocity = 0, bool continuous = false, EnvViewerFactory? viewerFactory = null) {
            Metadata = ClassMetadata;
            Continuous = continuous;
            GoalPosition = continuous ? 0.45 : 0.5;
            GoalVelocity = goalVelocity;

            // np.array([min_position, -max_speed], dtype=np.float32) and its high counterpart: Python's float64 values rounded
            // to float32, as the cast rounds them.
            float[] lowValues = { (float)MinPosition, (float)-MaxSpeed }, highValues = { (float)MaxPosition, (float)MaxSpeed };
            using NDArray low = np.array(lowValues);
            using NDArray high = np.array(highValues);
            // Box copies its bounds, so the temporaries are released here.
            ObservationSpace = new Box(low, high, dtype: np.float32);
            ActionSpace = continuous ? new Box(MinAction, MaxAction, new Shape(1), np.float32) : new Discrete(3);

            RenderMode = renderMode;
            _window = new HumanWindow(continuous ? "mountaincarcontinuous-v0" : "mountaincar-v0", viewerFactory);
        }

        /// <summary>
        ///     Builds the env with the render mode given as the <see cref="Gym.Envs.RenderMode"/> enum (DR-84, DR-93), as
        ///     <see cref="MountainCarEnv(string, double, bool, EnvViewerFactory)"/>
        ///     builds it from the mode's string: <c>new MountainCarEnv(RenderMode.RgbArray)</c> builds what
        ///     <c>new MountainCarEnv(renderMode: "rgb_array")</c> builds, the member reaching the env as Gymnasium's string
        ///     (<see cref="RenderModes.Name"/>), which its <c>RenderMode</c> property then holds.
        /// </summary>
        /// <remarks>
        ///     The mode is required and comes first, so <c>new MountainCarEnv()</c> keeps meaning the other constructor, without a
        ///     mode; leave the mode out there for no rendering. Every other parameter means what it means there.
        /// </remarks>
        /// <param name="renderMode">
        ///     The render mode, which the env must list in its <c>metadata["render_modes"]</c> as for the string; it behaves
        ///     exactly as its Gymnasium string does in the other constructor.
        /// </param>
        /// <param name="goalVelocity">Gymnasium's <c>goal_velocity</c>: the velocity the car must have at the flag (0 by default).</param>
        /// <param name="continuous">
        ///     Gym.NET's <c>continuous</c> (DR-32): MountainCarContinuous-v0's force-in-[-1, 1] dynamics, flag at 0.45 and
        ///     reward instead of MountainCar-v0's.
        /// </param>
        /// <param name="viewerFactory">Gym.NET's <c>viewer_factory</c> (DR-20) for human mode; <see langword="null"/> uses <c>Gymnasium.HumanViewerFactory</c>.</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a value <see cref="Gym.Envs.RenderMode"/> does not declare (a cast integer such as <c>(RenderMode)99</c>).</exception>
        public MountainCarEnv(RenderMode renderMode, double goalVelocity = 0, bool continuous = false,
                              EnvViewerFactory? viewerFactory = null)
            : this(RenderModes.Name(renderMode), goalVelocity, continuous, viewerFactory) { }

        /// <summary>
        ///     Gymnasium's class attribute <c>metadata</c> (both classes have the same one): <c>{"render_modes": ["human",
        ///     "rgb_array"], "render_fps": 30}</c>. Every env starts with a copy as its instance metadata.
        /// </summary>
        /// <remarks>A new dict on every read, so editing one env's metadata never reaches another (Python's is one shared dict).</remarks>
        public static new Metadata ClassMetadata => new() {
            [Metadata.RenderModesKey] = new List<string> { "human", "rgb_array" },
            [Metadata.RenderFpsKey] = 30L,
        };

        /// <summary>Whether this is MountainCarContinuous's dynamics, fixed at construction.</summary>
        public bool Continuous { get; }

        /// <summary>Gymnasium's <c>min_position</c>, -1.2: the left wall. Read every step and render.</summary>
        public double MinPosition { get; set; } = -1.2;

        /// <summary>Gymnasium's <c>max_position</c>, 0.6: the right wall.</summary>
        public double MaxPosition { get; set; } = 0.6;

        /// <summary>Gymnasium's <c>max_speed</c>, 0.07: the velocity clamp.</summary>
        public double MaxSpeed { get; set; } = 0.07;

        /// <summary>Gymnasium's <c>goal_position</c>: the flag, 0.5 in discrete mode and 0.45 in continuous mode.</summary>
        public double GoalPosition { get; set; }

        /// <summary>
        ///     Gymnasium's <c>goal_velocity</c>: the car terminates at the flag only at this velocity or faster. Compared as a
        ///     Python number, so in continuous mode against a float32 velocity it is rounded to float32 first.
        /// </summary>
        public double GoalVelocity { get; set; }

        /// <summary>Gymnasium's <c>force</c> (discrete mode), 0.001: the push of actions 0 and 2.</summary>
        public double Force { get; set; } = 0.001;

        /// <summary>Gymnasium's <c>gravity</c> (discrete mode), 0.0025. Continuous mode uses the literal 0.0025 of its source instead.</summary>
        public double Gravity { get; set; } = 0.0025;

        /// <summary>Gymnasium's <c>min_action</c> (continuous mode), -1.0: the lower clamp of the force, read every step.</summary>
        public double MinAction { get; set; } = -1.0;

        /// <summary>Gymnasium's <c>max_action</c> (continuous mode), 1.0: the upper clamp of the force.</summary>
        public double MaxAction { get; set; } = 1.0;

        /// <summary>Gymnasium's <c>power</c> (continuous mode), 0.0015: the force's scale.</summary>
        public double Power { get; set; } = 0.0015;

        /// <summary>Gymnasium's <c>screen_width</c>, 600 pixels; read when a frame is rendered.</summary>
        public int ScreenWidth { get; set; } = 600;

        /// <summary>Gymnasium's <c>screen_height</c>, 400 pixels; read when a frame is rendered.</summary>
        public int ScreenHeight { get; set; } = 400;

        /// <summary>
        ///     Gymnasium's <c>screen</c>: the surface every frame is blitted onto, <see langword="null"/> until the first render.
        ///     <see cref="Close"/> keeps it, as Python's <c>close</c> does.
        /// </summary>
        public Surface? Screen { get; private set; }

        /// <summary>Gymnasium's <c>clock</c>: paces human rendering, <see langword="null"/> until the first render.</summary>
        public Clock? Clock { get; private set; }

        /// <summary>Gymnasium's <c>surf</c>: the last frame drawn before it was blitted onto <see cref="Screen"/>, or <see langword="null"/>.</summary>
        public Surface? Surf { get; private set; }

        /// <summary>Gymnasium's <c>isopen</c>: <see langword="true"/> until <see cref="Close"/> shuts a render window down.</summary>
        [PyName("isopen")]
        public bool IsOpen { get; set; } = true;

        /// <summary>
        ///     Gymnasium's <c>state</c>, <c>[position, velocity]</c>: a new array of the values on every read, or
        ///     <see langword="null"/> before the first reset (and after <see langword="null"/> was assigned). Assigning injects a
        ///     state, as <c>env.state = np.array([...])</c> does: float64 values, of any length, checked by the next step or
        ///     render with Python's errors.
        /// </summary>
        /// <remarks>
        ///     Python's attribute changes container between calls (a float64 array after a reset, a tuple after a discrete
        ///     step, a float32 array after a continuous step); the values read here are exactly its values, a float32 one
        ///     widened. Assigning <see langword="null"/> stands for Python's <c>None</c>: the next step or render raises the
        ///     <see cref="TypeError"/> Python raises for it.
        /// </remarks>
        public double[]? State {
            get {
                if (_state is null)
                    return null;
                var values = new double[_state.Length];
                for (int i = 0; i < values.Length; i++)
                    values[i] = _state[i].Value;
                return values;
            }
            set {
                _hasState = true;
                if (value is null) {
                    _state = null;
                    return;
                }

                _state = new NpScalar[value.Length];
                for (int i = 0; i < value.Length; i++)
                    _state[i] = NpScalar.Float64(value[i]);
            }
        }

        /// <summary>
        ///     Starts an episode: reseeds the generator when a seed is given, then draws the position from
        ///     <c>uniform(low, high)</c> (default [-0.6, -0.4], or the <c>"low"</c>/<c>"high"</c> options) with zero velocity,
        ///     both float64. Renders in human mode.
        /// </summary>
        /// <param name="seed">The seed of a fresh <c>Generator(PCG64(SeedSequence(seed)))</c>, or <see langword="null"/> to keep the generator.</param>
        /// <param name="options">
        ///     Gymnasium's reset options: <c>"low"</c> and <c>"high"</c>, anything Python's <c>float()</c> accepts; custom bounds
        ///     can put the car outside the observation space, as in Gymnasium.
        /// </param>
        /// <returns>The float32 observation <c>[position, 0]</c> and an empty info.</returns>
        /// <exception cref="Error">Thrown for a negative seed, with Gymnasium's message.</exception>
        /// <exception cref="ValueError">Thrown for an option <c>float()</c> rejects, or a lower bound above the upper.</exception>
        /// <exception cref="OverflowError">Thrown, as Python's <c>OverflowError</c>, for NaN or infinite bounds (from the generator).</exception>
        /// <exception cref="DependencyNotInstalled">Thrown in human mode by the first render when no viewer factory is set (DR-20).</exception>
        /// <remarks>
        ///     The seed is applied before the options are parsed, as <c>super().reset(seed=seed)</c> runs first: a reset that
        ///     fails on its options leaves the freshly seeded generator, undrawn. Reference:
        ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/mountain_car.py#L156">gymnasium/envs/classic_control/mountain_car.py</see>.
        /// </remarks>
        public override ResetResult<NDArray> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            SeedNpRandom(seed);
            // Note that if you use custom reset bounds, it may lead to out-of-bound state/observations.
            (double low, double high) = ClassicControlUtils.MaybeParseResetBounds(options, -0.6, -0.4);
            double position;
            using (NDArray drawn = NpRandom.uniform(low, high))
                position = drawn.GetAtIndex<double>(0);
            // np.array([uniform, 0]): a float64 array, whose elements are float64 scalars when read back.
            _state = new[] { NpScalar.Float64(position), NpScalar.Float64(0.0) };
            _hasState = true;

            if (RenderMode == "human")
                Render();
            return new ResetResult<NDArray>(Observation(), new Info());
        }

        /// <summary>
        ///     One step. Discrete: <c>velocity += (action - 1) * force + cos(3 * position) * -gravity</c>, clipped to
        ///     ±<see cref="MaxSpeed"/>, then the position, clipped to the walls; reward -1. Continuous: the force is
        ///     <c>action[0]</c> clamped to [-1, 1], <c>velocity += force * power - 0.0025 * cos(3 * position)</c> with the
        ///     clamps of Gymnasium's source, reward 100 at the flag minus <c>0.1 * action[0]²</c>. A car stopped against the
        ///     left wall loses its leftward velocity. Renders in human mode.
        /// </summary>
        /// <param name="action">
        ///     Discrete: an element of <c>Discrete(3)</c> (a 0-d integer array; the untyped step also takes a .NET integer).
        ///     Continuous: an array whose first element is the force; a float32 or float64 element keeps its precision through
        ///     the arithmetic, as NumPy's scalar does, and an integer or boolean one computes as float64. Read, never kept.
        /// </param>
        /// <returns>
        ///     The float32 observation, the reward, whether the car reached the flag at <see cref="GoalVelocity"/> or faster,
        ///     <c>truncated=False</c> and <c>{}</c>.
        /// </returns>
        /// <exception cref="AssertionError">Thrown in discrete mode for an action outside the action space, with Python's message, before anything else.</exception>
        /// <exception cref="AttributeError">Thrown before the first reset (<c>'MountainCarEnv' object has no attribute 'state'</c>, naming the mode's class).</exception>
        /// <exception cref="TypeError">Thrown when <see cref="State"/> was assigned <see langword="null"/> (Python's <c>None</c>).</exception>
        /// <exception cref="ValueError">Thrown in discrete mode when an assigned state does not hold exactly two values (Python's unpacking error).</exception>
        /// <exception cref="IndexError">
        ///     Thrown in continuous mode when an assigned state holds fewer than two values, or the action has no first element
        ///     (a 0-d or empty array), with NumPy's messages.
        /// </exception>
        /// <exception cref="OverflowError">
        ///     Thrown in continuous mode, as Python's <c>OverflowError</c>, when <c>math.pow(action[0], 2)</c> overflows
        ///     (<c>math range error</c>, a float64 force beyond about 1.3e154); the state is then unchanged, as in Python.
        /// </exception>
        /// <exception cref="NotSupportedException">Thrown in continuous mode for an action Gym.NET does not model: more than one dimension, or a float16 or complex dtype.</exception>
        /// <exception cref="DependencyNotInstalled">Thrown in human mode by the first render when no viewer factory is set (DR-20).</exception>
        /// <remarks>
        ///     The continuous observation is a copy of the new state, where Gymnasium returns its state array itself; the values
        ///     are the same. References: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/mountain_car.py#L132">mountain_car.py</see>,
        ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/continuous_mountain_car.py#L150">continuous_mountain_car.py</see>.
        /// </remarks>
        public override StepResult<NDArray> Step(NDArray action) {
            (bool terminated, double reward) = Continuous ? ContinuousStep(action) : DiscreteStep(action);
            if (RenderMode == "human")
                Render();
            // truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`.
            return new StepResult<NDArray>(Observation(), reward, terminated, false, new Info());
        }

        /// <summary>
        ///     Draws Gymnasium's frame with the same primitives in the same order: the hill (an anti-aliased polyline through 100
        ///     points), the car rotated to the slope with its two wheels, and the flag, in a y-up world flipped at the end.
        /// </summary>
        /// <returns><c>rgb_array</c>: the <c>(400, 600, 3)</c> uint8 frame, a new array the caller owns. <c>human</c> and any other mode: nothing.</returns>
        /// <exception cref="AssertionError">Thrown without a render mode when the env has no spec (<c>assert self.spec is not None</c>).</exception>
        /// <exception cref="AttributeError">Thrown before the first reset, after the screen was created, as Gymnasium's render reads the missing state there.</exception>
        /// <exception cref="TypeError">Thrown when <see cref="State"/> was assigned <see langword="null"/>.</exception>
        /// <exception cref="IndexError">Thrown when an assigned state is empty.</exception>
        /// <exception cref="KeyError">Thrown in human mode when <c>metadata["render_fps"]</c> was removed.</exception>
        /// <exception cref="ValueError">Thrown for a NaN coordinate (a NaN state), with pygame's message.</exception>
        /// <exception cref="OverflowError">Thrown, as pygame's <c>OverflowError</c>, for a coordinate pygame cannot draw.</exception>
        /// <exception cref="DependencyNotInstalled">Thrown on the first human render when no viewer factory is set (DR-20).</exception>
        /// <remarks>
        ///     The car's coordinates are computed in the state's precision: after a continuous step its position is float32, so
        ///     its height uses NumPy's float32 sine and its sums round to float32, as in Python. Reference:
        ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/mountain_car.py#L175">gymnasium/envs/classic_control/mountain_car.py</see>.
        /// </remarks>
        public override RenderResult Render() {
            if (RenderMode is null) {
                EnvSpec spec = Spec ?? throw new AssertionError();
                Logger.Warn("You are calling render method without specifying any render mode. You can specify the render_mode at initialization, "
                            + $"e.g. gym.make(\"{spec.Id}\", render_mode=\"rgb_array\")");
                return RenderResult.None;
            }

            // self.screen and the human window exist from the first call, before the state is looked at.
            if (Screen is null) {
                if (RenderMode == "human")
                    _window.Ensure(ScreenWidth, ScreenHeight);
                // mode == "rgb_array" draws off-screen; in human mode this surface stands for the display surface.
                Screen = new Surface(ScreenWidth, ScreenHeight);
            }

            Clock ??= new Clock();

            double worldWidth = MaxPosition - MinPosition;
            double scale = ScreenWidth / worldWidth;
            const int carwidth = 40, carheight = 20;

            var surf = new Surface(ScreenWidth, ScreenHeight);
            Surf = surf;
            surf.Fill(new Color(255, 255, 255));

            NpScalar pos = Indexed(0);

            // xs = np.linspace(min_position, max_position, 100): i * step + start, with the last point exactly the stop.
            const int count = 100;
            double step = (MaxPosition - MinPosition) / (count - 1);
            var xys = new (double X, double Y)[count];
            for (int i = 0; i < count; i++) {
                double x = i == count - 1 ? MaxPosition : i * step + MinPosition;
                // self._height(xs): NumPy's float64 sine, which is the C runtime's on the reference platform.
                double y = Math.Sin(3 * x) * 0.45 + 0.55;
                xys[i] = ((x - MinPosition) * scale, y * scale);
            }

            Draw.Aalines(surf, new Color(0, 0, 0), false, xys);

            const int clearance = 10;
            const double l = -carwidth / 2.0, r = carwidth / 2.0;
            const int t = carheight, b = 0;
            // math.cos(3 * pos): the product in pos's precision, then the double cosine.
            double angle = (NpScalar.PyInt(3) * pos).MathCos().Value;
            // (pos - min_position) * scale and self._height(pos) * scale, in pos's precision (float32 after a continuous step).
            NpScalar alongSlope = (pos - NpScalar.PyFloat(MinPosition)) * NpScalar.PyFloat(scale);
            NpScalar heightScaled = Height(pos) * NpScalar.PyFloat(scale);
            var coords = new List<(double X, double Y)>(4);
            foreach ((double X, double Y) c in new[] { (l, (double)b), (l, t), (r, t), (r, b) }) {
                Vector2 rotated = new Vector2(c.X, c.Y).RotateRad(angle);
                // c[0] + (...) and (c[1] + clearance) + (...): a Python float joined to a float32 scalar is a float32 sum.
                coords.Add(((NpScalar.PyFloat(rotated.X) + alongSlope).Value, (NpScalar.PyFloat(rotated.Y) + NpScalar.PyInt(clearance) + heightScaled).Value));
            }

            GfxDraw.AAPolygon(surf, coords, new Color(0, 0, 0));
            GfxDraw.FilledPolygon(surf, coords, new Color(0, 0, 0));

            foreach ((double X, double Y) c in new[] { (carwidth / 4.0, 0.0), (-carwidth / 4.0, 0.0) }) {
                Vector2 rotated = new Vector2(c.X, c.Y).RotateRad(angle);
                // int(...) of each coordinate first, left to right, then pygame's range checks, as Python evaluates the calls.
                long wheelX = RenderMath.Int((NpScalar.PyFloat(rotated.X) + alongSlope).Value);
                long wheelY = RenderMath.Int((NpScalar.PyFloat(rotated.Y) + NpScalar.PyInt(clearance) + heightScaled).Value);
                // int(carheight / 2.5) is 8.
                long wheelRadius = RenderMath.Int(carheight / 2.5);
                GfxDraw.AACircle(surf, wheelX, wheelY, wheelRadius, new Color(128, 128, 128));
                GfxDraw.FilledCircle(surf, wheelX, wheelY, wheelRadius, new Color(128, 128, 128));
            }

            long flagx = RenderMath.Int((GoalPosition - MinPosition) * scale);
            // self._height(self.goal_position): a Python float through np.sin, so a float64 scalar.
            long flagy1 = RenderMath.Int((Height(NpScalar.PyFloat(GoalPosition)) * NpScalar.PyFloat(scale)).Value);
            long flagy2 = flagy1 + 50;
            GfxDraw.VLine(surf, flagx, flagy1, flagy2, new Color(0, 0, 0));

            var flag = new List<(double X, double Y)> { (flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5) };
            GfxDraw.AAPolygon(surf, flag, new Color(204, 204, 0));
            GfxDraw.FilledPolygon(surf, flag, new Color(204, 204, 0));

            surf = Transform.Flip(surf, false, true);
            Surf = surf;
            Screen.Blit(surf, 0, 0);
            if (RenderMode == "human") {
                // pygame.event.pump() has no event queue to pump here; the viewer runs its own message loop.
                Clock.Tick(PyBuiltins.Float(Metadata[Metadata.RenderFpsKey]));
                // pygame.display.flip(): a window shut by Close() is reopened from the factory.
                _window.Show(Screen);
            } else if (RenderMode == "rgb_array") {
                // np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)): rows of RGB.
                return new RenderResult.Rgb(SurfArray.ToRgbArray(Screen));
            }

            return RenderResult.None;
        }

        /// <summary>
        ///     Once something was rendered, shuts the human-mode window down and marks the env closed (<see cref="IsOpen"/>).
        ///     Like Gymnasium's, it keeps <see cref="Screen"/>, so a later <c>rgb_array</c> render reuses it.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/mountain_car.py#L282">gymnasium/envs/classic_control/mountain_car.py</see>.</remarks>
        public override void Close() {
            if (Screen is not null) {
                // pygame.display.quit(); pygame.quit()
                _window.Close();
                IsOpen = false;
            }
        }

        /// <summary>
        ///     The discrete mode's keyboard mapping for <see cref="Gym.Utils.Play"/> (Gymnasium's
        ///     <c>get_keys_to_action()</c>): <c>{(): 1, (276,): 0, (275,): 2, (275, 276): 1}</c>, nothing pressed or both
        ///     arrows coasting, one arrow pushing that way. Each call builds a new mapping whose actions (0-d int64 arrays, DR-08)
        ///     the caller owns.
        /// </summary>
        /// <returns>The mapping, in Gymnasium's order.</returns>
        /// <exception cref="AttributeError">
        ///     Thrown in continuous mode, where Gymnasium's class has no such method:
        ///     <c>'Continuous_MountainCarEnv' object has no attribute 'get_keys_to_action'</c>.
        /// </exception>
        /// <remarks>
        ///     <para>
        ///     Two footguns are Gymnasium's own and kept: 275 and 276 are pygame 1's arrow codes, which no pygame 2 window (nor
        ///     a Gym.NET viewer) produces, so the arrows do nothing; and the empty combination <c>()</c> makes <c>play</c>
        ///     refuse the mapping (<c>Key combinations in keys_to_action tuples cannot be empty</c>), so <c>play</c> runs
        ///     MountainCar only with a mapping of the caller's. <c>PlayableGame</c> takes it (its relevant keys are 275 and 276).
        ///     </para>
        ///     <para>
        ///     The override of <c>Env</c>'s absent base (DR-45); <see cref="HasPythonAttribute"/> keeps <c>hasattr</c> false in
        ///     continuous mode, as on Gymnasium's continuous class.
        ///     </para>
        ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/mountain_car.py#L278">gymnasium/envs/classic_control/mountain_car.py</see>.</para>
        /// </remarks>
        public override IReadOnlyDictionary<KeyCombination, NDArray> GetKeysToAction() {
            if (Continuous)
                throw new AttributeError($"'{ClassName}' object has no attribute 'get_keys_to_action'");
            return new Dictionary<KeyCombination, NDArray> {
                [KeyCombination.None] = NDArray.Scalar(1L),
                [new KeyCombination(276)] = NDArray.Scalar(0L),
                [new KeyCombination(275)] = NDArray.Scalar(2L),
                [new KeyCombination(275, 276)] = NDArray.Scalar(1L),
            };
        }

        /// <summary>
        ///     Hides <c>get_keys_to_action</c> in continuous mode (DR-45): Gymnasium's <c>Continuous_MountainCarEnv</c> has no
        ///     such method, so <c>hasattr</c> and <c>has_wrapper_attr</c> answer false there.
        /// </summary>
        /// <param name="name">The attribute, in Python spelling.</param>
        /// <returns><see langword="false"/> for <c>get_keys_to_action</c> in continuous mode; <see langword="true"/> otherwise.</returns>
        protected override bool HasPythonAttribute(string name) => name != "get_keys_to_action" || !Continuous;

        /// <summary>The Python class this instance stands for: <c>Continuous_MountainCarEnv</c> in continuous mode (DR-32), else <c>MountainCarEnv</c>.</summary>
        protected override string PythonClassName => ClassName;

        /// <summary>
        ///     The env's description, as Gymnasium prints the class of this mode: <c>&lt;MountainCarEnv&lt;MountainCar-v0&gt;&gt;</c>,
        ///     <c>&lt;Continuous_MountainCarEnv instance&gt;</c>; wrapper descriptions embed it.
        /// </summary>
        /// <returns>The text.</returns>
        public override string ToString() {
            EnvSpec? spec = Spec;
            return spec is null ? $"<{ClassName} instance>" : $"<{ClassName}<{spec.Id}>>";
        }

        /// <summary>
        ///     The untyped step's action conversion, with the discrete env's own assert first, so its message shows exactly the
        ///     value given (<c>3 (&lt;class 'int'&gt;) invalid</c>). Continuous mode converts without a check, as Gymnasium's
        ///     continuous env asserts nothing.
        /// </summary>
        /// <param name="action">The action as given to the untyped step.</param>
        /// <returns>The action as an array: the one given, or a new one the step releases.</returns>
        /// <exception cref="AssertionError">Thrown in discrete mode for anything <c>Discrete(3).contains</c> rejects.</exception>
        /// <exception cref="TypeError">Thrown in continuous mode for a value that has no array form.</exception>
        protected override NDArray ConvertAction(object? action) {
            if (!Continuous)
                AssertValidAction(action);
            return base.ConvertAction(action);
        }

        /// <summary>Gymnasium's class name for this mode, which <c>repr</c> and errors use.</summary>
        private string ClassName => Continuous ? "Continuous_MountainCarEnv" : "MountainCarEnv";

        /// <summary>
        ///     <c>self._height(x)</c> for a scalar: <c>np.sin(3 * x) * 0.45 + 0.55</c> in the scalar's precision (NumPy's own
        ///     float32 sine for a float32 position).
        /// </summary>
        /// <param name="x">The position.</param>
        /// <returns>The hill's height there, a NumPy scalar.</returns>
        private static NpScalar Height(NpScalar x) => (NpScalar.PyInt(3) * x).NpSin() * NpScalar.PyFloat(0.45) + NpScalar.PyFloat(0.55);

        /// <summary>Gymnasium's discrete step, from its action assert to the new state.</summary>
        /// <param name="action">The action.</param>
        /// <returns>Whether the car reached the flag, and the reward (-1).</returns>
        /// <exception cref="AssertionError">Thrown for an action outside <c>Discrete(3)</c>.</exception>
        /// <exception cref="AttributeError">Thrown before the first reset.</exception>
        /// <exception cref="TypeError">Thrown for a <c>None</c> state.</exception>
        /// <exception cref="ValueError">Thrown for a state that does not hold exactly two values.</exception>
        private (bool Terminated, double Reward) DiscreteStep(NDArray action) {
            AssertValidAction(action);
            // position, velocity = self.state
            NpScalar[] state = RequireState();
            if (state.Length != 2)
                throw new ValueError(state.Length > 2 ? "too many values to unpack (expected 2)" : $"not enough values to unpack (expected 2, got {state.Length})");
            NpScalar position = state[0], velocity = state[1];

            NpScalar push = NpScalar.Float64(ActionMinusOne(action) * Force);
            NpScalar pull = (NpScalar.PyInt(3) * position).MathCos() * NpScalar.PyFloat(-Gravity);
            // velocity += A + B: the right side first, then the in-place add.
            velocity = velocity + (push + pull);
            velocity = NpScalar.Float64(PyMath.Clip(velocity.Value, -MaxSpeed, MaxSpeed));
            position = position + velocity;
            position = NpScalar.Float64(PyMath.Clip(position.Value, MinPosition, MaxPosition));
            if (position.PyEquals(NpScalar.PyFloat(MinPosition)) && velocity < NpScalar.PyInt(0))
                velocity = NpScalar.PyInt(0);

            bool terminated = position >= NpScalar.PyFloat(GoalPosition) && velocity >= NpScalar.PyFloat(GoalVelocity);
            // self.state = (position, velocity): a tuple of the two scalars, kept as they are.
            _state = new[] { position, velocity };
            return (terminated, -1.0);
        }

        /// <summary>Gymnasium's continuous step, operation for operation with NEP 50 scalars.</summary>
        /// <param name="action">The action.</param>
        /// <returns>Whether the car reached the flag, and the reward.</returns>
        /// <exception cref="AttributeError">Thrown before the first reset.</exception>
        /// <exception cref="TypeError">Thrown for a <c>None</c> state.</exception>
        /// <exception cref="IndexError">Thrown for a state of fewer than two values, or an action with no first element.</exception>
        /// <exception cref="OverflowError">Thrown when <c>math.pow(action[0], 2)</c> overflows.</exception>
        /// <exception cref="NotSupportedException">Thrown for an action of more than one dimension, or of a float16 or complex dtype.</exception>
        private (bool Terminated, double Reward) ContinuousStep(NDArray action) {
            // position = self.state[0]; velocity = self.state[1]; then action[0], in that order.
            NpScalar position = Indexed(0);
            NpScalar velocity = Indexed(1);
            NpScalar first = FirstElement(action);

            NpScalar force = NpScalar.PyMin(NpScalar.PyMax(first, NpScalar.PyFloat(MinAction)), NpScalar.PyFloat(MaxAction));
            velocity = velocity + (force * NpScalar.PyFloat(Power) - NpScalar.PyFloat(0.0025) * (NpScalar.PyInt(3) * position).MathCos());
            if (velocity > NpScalar.PyFloat(MaxSpeed))
                velocity = NpScalar.PyFloat(MaxSpeed);
            if (velocity < NpScalar.PyFloat(-MaxSpeed))
                velocity = NpScalar.PyFloat(-MaxSpeed);
            position = position + velocity;
            if (position > NpScalar.PyFloat(MaxPosition))
                position = NpScalar.PyFloat(MaxPosition);
            if (position < NpScalar.PyFloat(MinPosition))
                position = NpScalar.PyFloat(MinPosition);
            if (position.PyEquals(NpScalar.PyFloat(MinPosition)) && velocity < NpScalar.PyInt(0))
                velocity = NpScalar.PyInt(0);

            // Convert a possible numpy bool to a Python bool.
            bool terminated = position >= NpScalar.PyFloat(GoalPosition) && velocity >= NpScalar.PyFloat(GoalVelocity);

            // reward = 0 (or 100.0); reward -= math.pow(action[0], 2) * 0.1: an int minus a float is 0.0 - x, never -x.
            double squared = Math.Pow(first.Value, 2);
            // math.pow refuses an overflow from finite operands, before the state is assigned.
            if (double.IsInfinity(squared) && double.IsFinite(first.Value))
                throw new OverflowError("math range error");
            double reward = (terminated ? 100.0 : 0.0) - squared * 0.1;

            // self.state = np.array([position, velocity], dtype=np.float32)
            _state = new[] { NpScalar.Float32(position.ToFloat32()), NpScalar.Float32(velocity.ToFloat32()) };
            return (terminated, reward);
        }

        /// <summary>
        ///     <c>action - 1</c> in the action's integer type, as a double: an unsigned 0 wraps to the type's maximum, as NumPy's
        ///     unsigned arithmetic does with the weak Python <c>1</c> (<c>uint8</c>: 255).
        /// </summary>
        /// <param name="action">A <c>Discrete(3)</c> element.</param>
        /// <returns>The difference, exactly as NumPy's float64 conversion of it.</returns>
        private static double ActionMinusOne(NDArray action) {
            long value = Discrete.ToInt64(action);
            if (value != 0)
                return value - 1;
            return action.typecode switch {
                NPTypeCode.Byte => byte.MaxValue,
                NPTypeCode.UInt16 => ushort.MaxValue,
                NPTypeCode.UInt32 => uint.MaxValue,
                // (double)ulong.MaxValue rounds to 2^64, as NumPy's uint64 -> float64 cast does.
                NPTypeCode.UInt64 => ulong.MaxValue,
                _ => -1,
            };
        }

        /// <summary><c>action[0]</c> as NumPy indexes a one-dimensional array, with NumPy's errors.</summary>
        /// <param name="action">The action array.</param>
        /// <returns>The element as the scalar NumPy returns: float32, float64, or for an integer or boolean dtype its value as float64 (all it is used as).</returns>
        /// <exception cref="IndexError">Thrown for a 0-d or empty array, with NumPy's messages.</exception>
        /// <exception cref="NotSupportedException">Thrown for more than one dimension, or a float16 or complex dtype.</exception>
        private static NpScalar FirstElement(NDArray action) {
            // NumPy's own errors first, in the order indexing meets them, whatever the dtype.
            if (action.ndim == 0)
                throw new IndexError("too many indices for array: array is 0-dimensional, but 1 were indexed");
            if (action.shape[0] == 0)
                throw new IndexError("index 0 is out of bounds for axis 0 with size 0");
            // Python would carry a row (or a float16 scalar) through min/max into a state of another shape or precision;
            // refusing it is better than silently computing something else.
            if (action.ndim > 1)
                throw new NotSupportedException($"Gym.NET's continuous MountainCarEnv takes a one-dimensional action; a {action.ndim}-dimensional one would make Gymnasium's state an array of rows.");
            if (action.typecode is NPTypeCode.Half or NPTypeCode.Complex)
                throw new NotSupportedException($"Gym.NET's continuous MountainCarEnv does not model the arithmetic of a {action.typecode} action; pass float32 (the action space's dtype) or float64.");
            return NpScalar.Element(action, 0);
        }

        /// <summary><c>self.state[index]</c> for an index of 0 or 1, with Python's errors for a missing, <c>None</c> or short state.</summary>
        /// <param name="index">The index.</param>
        /// <returns>The scalar.</returns>
        /// <exception cref="AttributeError">Thrown before the first reset.</exception>
        /// <exception cref="TypeError">Thrown for a <c>None</c> state.</exception>
        /// <exception cref="IndexError">Thrown when the state is too short, with NumPy's message (an assigned state is an array).</exception>
        private NpScalar Indexed(int index) {
            if (!_hasState)
                throw new AttributeError($"'{ClassName}' object has no attribute 'state'");
            NpScalar[] state = _state ?? throw new TypeError("'NoneType' object is not subscriptable");
            if (index >= state.Length)
                throw new IndexError($"index {index} is out of bounds for axis 0 with size {state.Length}");
            return state[index];
        }

        /// <summary>The state for unpacking, with Python's errors for a missing or <c>None</c> one.</summary>
        /// <returns>The scalars; their count is the caller's to check.</returns>
        /// <exception cref="AttributeError">Thrown before the first reset.</exception>
        /// <exception cref="TypeError">Thrown for a <c>None</c> state (<c>cannot unpack non-iterable NoneType object</c>).</exception>
        private NpScalar[] RequireState() {
            if (!_hasState)
                throw new AttributeError($"'{ClassName}' object has no attribute 'state'");
            return _state ?? throw new TypeError("cannot unpack non-iterable NoneType object");
        }

        /// <summary>
        ///     <c>np.array(self.state, dtype=np.float32)</c>: a new float32 array of the state's values, rounded to nearest, as
        ///     NumPy's cast rounds.
        /// </summary>
        /// <returns>The observation; the caller owns it.</returns>
        private NDArray Observation() {
            var observation = new float[_state!.Length];
            for (int i = 0; i < observation.Length; i++)
                observation[i] = _state[i].ToFloat32();
            return np.array(observation);
        }

        /// <summary><c>assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"</c>.</summary>
        /// <param name="action">The action, as given (so the message shows its Python counterpart).</param>
        /// <exception cref="AssertionError">Thrown when the action space does not contain the action.</exception>
        private void AssertValidAction(object? action) {
            if (!ActionSpace.Contains(action!))
                throw new AssertionError($"{PyBuiltins.Repr(action)} ({PyBuiltins.TypeRepr(action)}) invalid");
        }
    }
}
