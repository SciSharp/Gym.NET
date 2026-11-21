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
    ///     Acrobot-v1: two links joined in a chain, hanging from a fixed joint, with a torque of -1, 0 or +1 (actions 0, 1, 2)
    ///     on the joint between them; the goal is to swing the free end above a line one link-length over the base. -1 per
    ///     step until then, 0 on the terminating step. Byte for byte what Gymnasium's Acrobot computes, errors included.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The dynamics are Gymnasium's: the "book" equations (<see cref="BookOrNips"/>) integrated over <see cref="Dt"/> by one
    ///     fourth-order Runge-Kutta step, in float64, every product and sum in the source's order and every <c>x ** 2</c>
    ///     the C runtime's <c>pow</c>, as NumPy computes them. The reset state is float32 (<c>uniform(...).astype(np.float32)</c>),
    ///     so the first observation's cosines and sines are NumPy's own float32 ones (through NumSharp, bit for bit) and the
    ///     first step starts from the float32-rounded values; after a step the state is float64.
    ///     </para>
    ///     <para>
    ///     The action is not checked against the action space, as in Gymnasium: it indexes <see cref="AvailTorque"/> the way
    ///     a Python list is indexed, so -1, -2 and -3 are valid (the torques of 2, 1 and 0), and other values raise Python's
    ///     <c>IndexError</c> or <c>TypeError</c>. The physical constants are Gymnasium's class attributes, settable per env
    ///     here (Python's are shared by the class); RLPy's unused <c>action_arrow</c>, <c>domain_fig</c> and
    ///     <c>actions_num</c> are not carried.
    ///     </para>
    ///     <para>
    ///     Rendering is Gymnasium's (the line, the links and joints, pixel for pixel). Human mode renders into the window from
    ///     <c>viewer_factory</c> or <c>Gymnasium.HumanViewerFactory</c> (DR-20) at 15 frames per second. A state angle that
    ///     is infinite never wraps, in Gymnasium's <c>wrap</c> as here, so such a step does not return.
    ///     </para>
    ///     <para>
    ///     Not thread-safe. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/acrobot.py#L26">gymnasium/envs/classic_control/acrobot.py</see>.
    ///     </para>
    /// </remarks>
    [GymOptions]
    public partial class AcrobotEnv : Env<NDArray, NDArray> {
        /// <summary>The message of the source's asserts on the state.</summary>
        private const string ResetFirst = "Call reset before using AcrobotEnv object.";

        /// <summary>The human-mode window, opened on the first human render.</summary>
        private readonly HumanWindow _window;

        /// <summary>
        ///     Python's <c>self.state</c> as the scalars it holds: float32 ones after a reset, float64 ones after a step or an
        ///     assignment; <see langword="null"/> for <c>None</c>, which it is until the first reset.
        /// </summary>
        private NpScalar[]? _state;

        /// <summary>Creates the env, stateless until the first reset.</summary>
        /// <param name="renderMode">
        ///     Gymnasium's <c>render_mode</c>: <see langword="null"/>, <c>"human"</c> or <c>"rgb_array"</c>; not validated, as in
        ///     Gymnasium (<c>make</c> checks it against the metadata).
        /// </param>
        /// <param name="viewerFactory">Gym.NET's <c>viewer_factory</c> (DR-20) for human mode; <see langword="null"/> uses <c>Gymnasium.HumanViewerFactory</c>.</param>
        public AcrobotEnv(string? renderMode = null, IEnvironmentViewerFactoryDelegate? viewerFactory = null) {
            Metadata = ClassMetadata;

            // np.array([1.0, 1.0, 1.0, 1.0, MAX_VEL_1, MAX_VEL_2], dtype=np.float32), and low = -high.
            float[] highValues = { 1.0f, 1.0f, 1.0f, 1.0f, (float)MaxVel1, (float)MaxVel2 };
            using NDArray high = np.array(highValues);
            using NDArray low = -high;
            // Box copies its bounds, so the temporaries are released here.
            ObservationSpace = new Box(low, high, dtype: np.float32);
            ActionSpace = new Discrete(3);

            RenderMode = renderMode;
            _window = new HumanWindow("acrobot-v1", viewerFactory);
        }

        /// <summary>
        ///     Gymnasium's class attribute <c>metadata</c>: <c>{"render_modes": ["human", "rgb_array"], "render_fps": 15}</c>.
        ///     Every env starts with a copy as its instance metadata.
        /// </summary>
        /// <remarks>A new dict on every read, so editing one env's metadata never reaches another (Python's is one shared dict).</remarks>
        public static new Metadata ClassMetadata => new() {
            [Metadata.RenderModesKey] = new List<string> { "human", "rgb_array" },
            [Metadata.RenderFpsKey] = 15L,
        };

        /// <summary>Gymnasium's <c>dt</c>, 0.2 s: the integration time step, one Runge-Kutta step per env step.</summary>
        public double Dt { get; set; } = 0.2;

        /// <summary>Gymnasium's <c>LINK_LENGTH_1</c>, 1.0 m. Also sizes the frame (the world spans both links plus 0.2).</summary>
        [PyName("LINK_LENGTH_1")]
        public double LinkLength1 { get; set; } = 1.0;

        /// <summary>Gymnasium's <c>LINK_LENGTH_2</c>, 1.0 m. Used only by rendering, as in Gymnasium's dynamics.</summary>
        [PyName("LINK_LENGTH_2")]
        public double LinkLength2 { get; set; } = 1.0;

        /// <summary>Gymnasium's <c>LINK_MASS_1</c>, 1.0 kg.</summary>
        [PyName("LINK_MASS_1")]
        public double LinkMass1 { get; set; } = 1.0;

        /// <summary>Gymnasium's <c>LINK_MASS_2</c>, 1.0 kg.</summary>
        [PyName("LINK_MASS_2")]
        public double LinkMass2 { get; set; } = 1.0;

        /// <summary>Gymnasium's <c>LINK_COM_POS_1</c>, 0.5 m: the first link's center of mass.</summary>
        [PyName("LINK_COM_POS_1")]
        public double LinkComPos1 { get; set; } = 0.5;

        /// <summary>Gymnasium's <c>LINK_COM_POS_2</c>, 0.5 m: the second link's center of mass.</summary>
        [PyName("LINK_COM_POS_2")]
        public double LinkComPos2 { get; set; } = 0.5;

        /// <summary>Gymnasium's <c>LINK_MOI</c>, 1.0: both links' moment of inertia.</summary>
        [PyName("LINK_MOI")]
        public double LinkMoi { get; set; } = 1.0;

        /// <summary>Gymnasium's <c>MAX_VEL_1</c>, 4π: the first joint's angular velocity bound, applied every step.</summary>
        [PyName("MAX_VEL_1")]
        public double MaxVel1 { get; set; } = 4 * Math.PI;

        /// <summary>Gymnasium's <c>MAX_VEL_2</c>, 9π: the second joint's angular velocity bound.</summary>
        [PyName("MAX_VEL_2")]
        public double MaxVel2 { get; set; } = 9 * Math.PI;

        /// <summary>
        ///     Gymnasium's <c>AVAIL_TORQUE</c>, <c>[-1.0, 0.0, +1]</c>: the torque of each action, indexed as a Python list (a
        ///     negative action counts from the end). Python's last entry is the int <c>1</c>, which joins the float64 state as
        ///     1.0, so a double holds it exactly.
        /// </summary>
        [PyName("AVAIL_TORQUE")]
        public double[] AvailTorque { get; set; } = { -1.0, 0.0, 1.0 };

        /// <summary>
        ///     Gymnasium's <c>torque_noise_max</c>, 0.0: when positive, every step adds <c>uniform(-max, max)</c> to the torque,
        ///     one draw from the env's generator.
        /// </summary>
        public double TorqueNoiseMax { get; set; }

        /// <summary>Gymnasium's <c>SCREEN_DIM</c>, 500: the frame's width and height in pixels.</summary>
        [PyName("SCREEN_DIM")]
        public int ScreenDim { get; set; } = 500;

        /// <summary>
        ///     Gymnasium's <c>book_or_nips</c>, <c>"book"</c>: which equation gives the second joint's acceleration. <c>"nips"</c>
        ///     uses the NIPS paper's, which leaves out a velocity term; any other string the book's (and the Java
        ///     implementation's).
        /// </summary>
        public string BookOrNips { get; set; } = "book";

        /// <summary>
        ///     Gymnasium's <c>screen</c>: the surface every frame is blitted onto, <see langword="null"/> until the first render.
        ///     <see cref="Close"/> keeps it, as Python's <c>close</c> does.
        /// </summary>
        public Surface? Screen { get; private set; }

        /// <summary>Gymnasium's <c>clock</c>: paces human rendering, <see langword="null"/> until the first render.</summary>
        public Clock? Clock { get; private set; }

        /// <summary>Gymnasium's <c>isopen</c>: <see langword="true"/> until <see cref="Close"/> shuts a render window down.</summary>
        [PyName("isopen")]
        public bool IsOpen { get; set; } = true;

        /// <summary>
        ///     Gymnasium's <c>state</c>, <c>[θ1, θ2, θ̇1, θ̇2]</c>: a new array of the values on every read, <see langword="null"/>
        ///     before the first reset. Assigning injects a state, as <c>env.state = np.array([...])</c> does: float64 values of
        ///     any length, checked by the next step with Python's errors; <see langword="null"/> is Python's <c>None</c>.
        /// </summary>
        /// <remarks>
        ///     Python's array is float32 after a reset and float64 after a step; the values read here are exactly its values, a
        ///     float32 one widened.
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
        ///     Starts an episode: reseeds the generator when a seed is given, then draws the four state variables from
        ///     <c>uniform(low, high, size=(4,))</c> (default ±0.1, or the <c>"low"</c>/<c>"high"</c> options) and rounds them to
        ///     float32. Renders in human mode.
        /// </summary>
        /// <param name="seed">The seed of a fresh <c>Generator(PCG64(SeedSequence(seed)))</c>, or <see langword="null"/> to keep the generator.</param>
        /// <param name="options">
        ///     Gymnasium's reset options: <c>"low"</c> and <c>"high"</c>, anything Python's <c>float()</c> accepts; custom bounds
        ///     can put the state outside the observation space, as in Gymnasium.
        /// </param>
        /// <returns>The float32 observation <c>[cos θ1, sin θ1, cos θ2, sin θ2, θ̇1, θ̇2]</c> and an empty info.</returns>
        /// <exception cref="Error">Thrown for a negative seed, with Gymnasium's message.</exception>
        /// <exception cref="ValueError">Thrown for an option <c>float()</c> rejects, or a lower bound above the upper.</exception>
        /// <exception cref="OverflowError">Thrown, as Python's <c>OverflowError</c>, for NaN or infinite bounds (from the generator).</exception>
        /// <exception cref="DependencyNotInstalled">Thrown in human mode by the first render when no viewer factory is set (DR-20).</exception>
        /// <remarks>
        ///     The seed is applied before the options are parsed, as <c>super().reset(seed=seed)</c> runs first. Reference:
        ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/acrobot.py#L199">gymnasium/envs/classic_control/acrobot.py</see>.
        /// </remarks>
        public override ResetResult<NDArray> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            SeedNpRandom(seed);
            // Note that if you use custom reset bounds, it may lead to out-of-bound state/observations.
            (double low, double high) = ClassicControlUtils.MaybeParseResetBounds(options, -0.1, 0.1);
            using (NDArray drawn = NpRandom.uniform(low, high, new Shape(4))) {
                // .astype(np.float32): each float64 draw rounded to nearest, as NumPy's cast rounds.
                var state = new NpScalar[4];
                for (int i = 0; i < state.Length; i++)
                    state[i] = NpScalar.Float32((float)drawn.GetAtIndex<double>(i));
                _state = state;
            }

            if (RenderMode == "human")
                Render();
            return new ResetResult<NDArray>(Observation(), new Info());
        }

        /// <summary>
        ///     Applies the action's torque to the middle joint for <see cref="Dt"/> seconds (one Runge-Kutta step), wraps both
        ///     angles into [-π, π] and bounds both velocities, then terminates when the free end is above the line. Renders in
        ///     human mode.
        /// </summary>
        /// <param name="action">
        ///     An index into <see cref="AvailTorque"/>: a 0-d integer array (what <c>Discrete(3).Sample()</c> returns), or through
        ///     the untyped step any .NET integer. Python's list indexing applies, so -3 to 2 are valid. Read, never kept.
        /// </param>
        /// <returns>The float32 observation, the reward (-1.0, or 0.0 on termination), <c>terminated</c>, <c>truncated=False</c> and <c>{}</c>.</returns>
        /// <exception cref="AssertionError">Thrown before the first reset (<c>Call reset before using AcrobotEnv object.</c>), whatever the action, as the source asserts first.</exception>
        /// <exception cref="TypeError">Thrown for an action Python cannot index a list with, with Python's message (<c>list indices must be integers or slices, not numpy.float64</c>).</exception>
        /// <exception cref="IndexError">
        ///     Thrown for an index outside the list (<c>list index out of range</c>), and for an assigned state of fewer than four
        ///     values, with NumPy's message.
        /// </exception>
        /// <exception cref="ValueError">Thrown for an assigned state of more than four values (NumPy's broadcast error).</exception>
        /// <exception cref="DependencyNotInstalled">Thrown in human mode by the first render when no viewer factory is set (DR-20).</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/acrobot.py#L216">gymnasium/envs/classic_control/acrobot.py</see>.</remarks>
        public override StepResult<NDArray> Step(NDArray action) {
            NpScalar[] s = _state ?? throw new AssertionError(ResetFirst);
            double torque = AvailTorque[TorqueIndex(action)];

            // Add noise to the force action
            if (TorqueNoiseMax > 0) {
                using NDArray noise = NpRandom.uniform(-TorqueNoiseMax, TorqueNoiseMax);
                torque += noise.GetAtIndex<double>(0);
            }

            // Now, augment the state with our force action so it can be passed to _dsdt: np.append(s, torque), a float64
            // array whatever the state's dtype, since the torque is a Python number.
            var sAugmented = new double[s.Length + 1];
            for (int i = 0; i < s.Length; i++)
                sAugmented[i] = s[i].Value;
            sAugmented[s.Length] = torque;

            // rk4(self._dsdt, s_augmented, [0, self.dt]): the final step's first four values (the torque's entry is dropped).
            double[] ns = Rk4(Dsdt, sAugmented, new[] { 0.0, Dt });

            ns[0] = Wrap(ns[0], -Math.PI, Math.PI);
            ns[1] = Wrap(ns[1], -Math.PI, Math.PI);
            ns[2] = Bound(ns[2], -MaxVel1, MaxVel1);
            ns[3] = Bound(ns[3], -MaxVel2, MaxVel2);
            var state = new NpScalar[4];
            for (int i = 0; i < state.Length; i++)
                state[i] = NpScalar.Float64(ns[i]);
            _state = state;
            bool terminated = Terminal(ns);
            double reward = !terminated ? -1.0 : 0.0;

            if (RenderMode == "human")
                Render();
            // truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`.
            return new StepResult<NDArray>(Observation(), reward, terminated, false, new Info());
        }

        /// <summary>
        ///     Draws Gymnasium's frame with the same primitives in the same order: the target line, then each link as a rotated
        ///     rectangle with a circle at its joint, in a y-up world flipped at the end.
        /// </summary>
        /// <returns>
        ///     <c>rgb_array</c>: the <c>(500, 500, 3)</c> uint8 frame, a new array the caller owns. <c>human</c>, any other mode,
        ///     and every mode before the first reset: nothing.
        /// </returns>
        /// <exception cref="AssertionError">Thrown without a render mode when the env has no spec (<c>assert self.spec is not None</c>).</exception>
        /// <exception cref="IndexError">Thrown for an assigned state of fewer than two values, with NumPy's message.</exception>
        /// <exception cref="KeyError">Thrown in human mode when <c>metadata["render_fps"]</c> was removed.</exception>
        /// <exception cref="ValueError">Thrown for a NaN coordinate (a NaN state), with pygame's or Python's message.</exception>
        /// <exception cref="OverflowError">Thrown, as pygame's <c>OverflowError</c>, for a coordinate pygame cannot draw.</exception>
        /// <exception cref="DependencyNotInstalled">Thrown on the first human render when no viewer factory is set (DR-20).</exception>
        /// <remarks>
        ///     After a reset the state is float32, so the first link's end and both angles are computed in float32 (NumPy's
        ///     float32 cosine and sine), as in Python; after a step, in float64. Reference:
        ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/acrobot.py#L295">gymnasium/envs/classic_control/acrobot.py</see>.
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
                    _window.Ensure(ScreenDim, ScreenDim);
                // mode in "rgb_array" draws off-screen; in human mode this surface stands for the display surface.
                Screen = new Surface(ScreenDim, ScreenDim);
            }

            Clock ??= new Clock();

            var surf = new Surface(ScreenDim, ScreenDim);
            surf.Fill(new Color(255, 255, 255));
            NpScalar[]? s = _state;

            double bound = LinkLength1 + LinkLength2 + 0.2; // 2.2 for default
            double scale = ScreenDim / (bound * 2);
            // self.SCREEN_DIM / 2: true division, a float.
            double offset = ScreenDim / 2.0;

            if (s is null)
                return RenderResult.None;

            NpScalar s0 = StateElement(s, 0);
            // p1 = [-LINK_LENGTH_1 * cos(s[0]) * scale, LINK_LENGTH_1 * sin(s[0]) * scale], in s[0]'s precision.
            NpScalar p1X = NpScalar.PyFloat(-LinkLength1) * s0.NpCos() * NpScalar.PyFloat(scale);
            NpScalar p1Y = NpScalar.PyFloat(LinkLength1) * s0.NpSin() * NpScalar.PyFloat(scale);

            // xys = np.array([[0, 0], p1])[:, ::-1]: a float64 array (the int row and float32 values promote to float64), each
            // row's coordinates swapped.
            (double X, double Y)[] xys = { (0.0, 0.0), (p1Y.Value, p1X.Value) };
            NpScalar halfPi = NpScalar.PyFloat(Math.PI / 2);
            NpScalar[] thetas = { s0 - halfPi, s0 + StateElement(s, 1) - halfPi };
            double[] linkLengths = { LinkLength1 * scale, LinkLength2 * scale };

            Draw.Line(surf, new Color(0, 0, 0), -2.2 * scale + offset, 1 * scale + offset, 2.2 * scale + offset, 1 * scale + offset);

            for (int link = 0; link < 2; link++) {
                double x = xys[link].X + offset;
                double y = xys[link].Y + offset;
                double l = 0, r = linkLengths[link], t = 0.1 * scale, b = -0.1 * scale;
                var transformedCoords = new List<(double X, double Y)>(4);
                foreach ((double X, double Y) coord in new[] { (l, b), (l, t), (r, t), (r, b) }) {
                    Vector2 rotated = new Vector2(coord.X, coord.Y).RotateRad(thetas[link].Value);
                    transformedCoords.Add((rotated.X + x, rotated.Y + y));
                }

                GfxDraw.AAPolygon(surf, transformedCoords, new Color(0, 204, 204));
                GfxDraw.FilledPolygon(surf, transformedCoords, new Color(0, 204, 204));

                // int(x), int(y), int(0.1 * scale), converted for each call before pygame checks their range.
                GfxDraw.AACircle(surf, RenderMath.Int(x), RenderMath.Int(y), RenderMath.Int(0.1 * scale), new Color(204, 204, 0));
                GfxDraw.FilledCircle(surf, RenderMath.Int(x), RenderMath.Int(y), RenderMath.Int(0.1 * scale), new Color(204, 204, 0));
            }

            surf = Transform.Flip(surf, false, true);
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
        ///     Like Gymnasium's, it keeps <see cref="Screen"/>, so a later render reuses it.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/acrobot.py#L380">gymnasium/envs/classic_control/acrobot.py</see>.</remarks>
        public override void Close() {
            if (Screen is not null) {
                // pygame.display.quit(); pygame.quit()
                _window.Close();
                IsOpen = false;
            }
        }

        /// <summary>
        ///     Gymnasium's module function <c>wrap</c>: moves <paramref name="x"/> into [<paramref name="m"/>, <paramref name="M"/>]
        ///     by whole periods, one subtraction or addition at a time (so the rounding of each step is Python's, not a
        ///     remainder's).
        /// </summary>
        /// <param name="x">The value.</param>
        /// <param name="m">The lower end.</param>
        /// <param name="M">The upper end.</param>
        /// <returns>The wrapped value; NaN stays NaN.</returns>
        /// <remarks>
        ///     One iteration per period, as in Python: a huge value takes as many iterations as periods it spans, and an infinite
        ///     one never returns. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/acrobot.py#L389">gymnasium/envs/classic_control/acrobot.py</see>.
        /// </remarks>
        public static double Wrap(double x, double m, double M) {
            double diff = M - m;
            while (x > M)
                x = x - diff;
            while (x < m)
                x = x + diff;
            return x;
        }

        /// <summary>
        ///     Gymnasium's module function <c>bound</c> with scalar limits: <c>min(max(x, m), M)</c> with Python's built-ins,
        ///     which keep <paramref name="x"/> when a comparison is false (so NaN stays NaN).
        /// </summary>
        /// <param name="x">The value.</param>
        /// <param name="m">The lower limit.</param>
        /// <param name="M">The upper limit.</param>
        /// <returns>The bounded value.</returns>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/acrobot.py#L410">gymnasium/envs/classic_control/acrobot.py</see>.</remarks>
        public static double Bound(double x, double m, double M) {
            // max(x, m) is m only when m > x; min(that, M) is M only when M < that.
            double raised = m > x ? m : x;
            return M < raised ? M : raised;
        }

        /// <summary>
        ///     The untyped step's action conversion, with the source's order kept: its state assert first, then the list index
        ///     on the action as given, so the errors name the Python counterpart of the value given (<c>not float</c> for a
        ///     <see langword="double"/>, <c>not numpy.float32</c> for a <see langword="float"/>).
        /// </summary>
        /// <param name="action">The action as given to the untyped step.</param>
        /// <returns>The action as a 0-d integer array: the one given, or a new one the step releases.</returns>
        /// <exception cref="AssertionError">Thrown before the first reset.</exception>
        /// <exception cref="TypeError">Thrown for a value Python cannot index a list with.</exception>
        /// <exception cref="IndexError">Thrown for an index outside the list, or an integer beyond 64 bits.</exception>
        protected override NDArray ConvertAction(object? action) {
            if (_state is null)
                throw new AssertionError(ResetFirst);
            TorqueIndex(action);
            return base.ConvertAction(action);
        }

        /// <summary>
        ///     <c>AVAIL_TORQUE[a]</c>'s index as a Python list computes it: an integer (a .NET integer, <see langword="bool"/>, or
        ///     a 0-d integer array), counted from the end when negative.
        /// </summary>
        /// <param name="action">The action as given.</param>
        /// <returns>The index into <see cref="AvailTorque"/>.</returns>
        /// <exception cref="TypeError">
        ///     Thrown for a non-integer: <c>list indices must be integers or slices, not float</c> (a NumPy scalar's type for a 0-d
        ///     array), or for an array of one or more dimensions <c>only integer scalar arrays can be converted to a scalar
        ///     index</c>.
        /// </exception>
        /// <exception cref="IndexError">Thrown for an integer beyond 64 bits (<c>cannot fit 'int' into an index-sized integer</c>) or outside the list (<c>list index out of range</c>).</exception>
        private int TorqueIndex(object? action) {
            BigInteger index;
            switch (action) {
                case NDArray { ndim: > 0 }:
                    // ndarray.__index__ refuses every array that is not a 0-d integer one.
                    throw new TypeError("only integer scalar arrays can be converted to a scalar index");
                case NDArray scalar when scalar.typecode is NPTypeCode.UInt64:
                    // np.uint64 may exceed a long, which the index-size check below reports.
                    index = scalar.GetAtIndex<ulong>(0);
                    break;
                case BigInteger big:
                    index = big;
                    break;
                case ulong unsigned:
                    index = unsigned;
                    break;
                default:
                    // Python ints (bool among them) and NumPy integer scalars have __index__; floats, NumPy's bool and the
                    // rest do not, and the list names their type.
                    if (!Discrete.TryToInt64(action, out long integer))
                        throw new TypeError($"list indices must be integers or slices, not {PyBuiltins.TypeName(action)}");
                    index = integer;
                    break;
            }

            if (index < long.MinValue || index > long.MaxValue)
                throw new IndexError($"cannot fit '{PyBuiltins.TypeName(action)}' into an index-sized integer");
            long position = (long)index;
            if (position < 0)
                position += AvailTorque.Length;
            if (position < 0 || position >= AvailTorque.Length)
                throw new IndexError("list index out of range");
            return (int)position;
        }

        /// <summary>
        ///     Gymnasium's module function <c>rk4</c>: integrates <c>dy/dt = derivs(y)</c> from <paramref name="y0"/> over the
        ///     sample times <paramref name="t"/> with the classic fourth-order Runge-Kutta step, element by element in float64 in
        ///     NumPy's order, and returns the first four values of the last sample (Acrobot's state, its torque entry dropped).
        /// </summary>
        /// <param name="derivs">The derivative of the system at a point; its result must be as long as <paramref name="y0"/>.</param>
        /// <param name="y0">The initial point; not modified.</param>
        /// <param name="t">The sample times, at least one; the steps are their differences.</param>
        /// <returns>The first four values (or fewer, for a shorter system) of the point at the last sample time; a new array.</returns>
        /// <exception cref="ArgumentNullException">Thrown when an argument is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown when <paramref name="t"/> is empty (NumPy's <c>yout[-1]</c> would raise on the empty result).</exception>
        /// <exception cref="ValueError">
        ///     Thrown when a derivative's length differs from the point's, with NumPy's broadcast message (<c>operands could not
        ///     be broadcast together with shapes (6,) (5,) </c>), at the first operation that meets them.
        /// </exception>
        /// <exception cref="NotSupportedException">Thrown for a one-value derivative of a longer system, which NumPy would broadcast and Gym.NET does not model.</exception>
        /// <remarks>
        ///     Each stage is two array operations, <c>y0 + h * k</c>, rounded separately, and the combination is
        ///     <c>y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)</c> left to right, so the result is NumPy's bit for bit. Reference:
        ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/acrobot.py#L429">gymnasium/envs/classic_control/acrobot.py</see>.
        /// </remarks>
        public static double[] Rk4(Func<double[], double[]> derivs, double[] y0, IReadOnlyList<double> t) {
            if (derivs is null)
                throw new ArgumentNullException(nameof(derivs));
            if (y0 is null)
                throw new ArgumentNullException(nameof(y0));
            if (t is null)
                throw new ArgumentNullException(nameof(t));
            if (t.Count == 0)
                throw new ArgumentException("rk4 needs at least one sample time.", nameof(t));

            // yout[0] = y0; then each row from the previous one.
            var y = (double[])y0.Clone();
            for (int i = 0; i < t.Count - 1; i++) {
                double dt = t[i + 1] - t[i];
                double dt2 = dt / 2.0;
                double[] k1 = Derivative(derivs, y);
                double[] k2 = Derivative(derivs, Combine(y, dt2, k1));
                double[] k3 = Derivative(derivs, Combine(y, dt2, k2));
                double[] k4 = Derivative(derivs, Combine(y, dt, k3));
                // yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
                double sixth = dt / 6.0;
                var next = new double[y.Length];
                for (int j = 0; j < next.Length; j++)
                    next[j] = y[j] + sixth * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]);
                y = next;
            }

            // We only care about the final timestep and we cleave off action value which will be zero.
            var first = new double[Math.Min(4, y.Length)];
            Array.Copy(y, first, first.Length);
            return first;
        }

        /// <summary><c>np.asarray(derivs(y))</c>, refusing a length NumPy could not, or would only by broadcasting, combine with <paramref name="y"/>.</summary>
        /// <param name="derivs">The derivative function.</param>
        /// <param name="y">The point.</param>
        /// <returns>The derivative, as long as <paramref name="y"/>.</returns>
        /// <exception cref="ValueError">Thrown for a length NumPy cannot broadcast against the point's.</exception>
        /// <exception cref="NotSupportedException">Thrown for a one-value derivative of a longer system (NumPy would broadcast it).</exception>
        private static double[] Derivative(Func<double[], double[]> derivs, double[] y) {
            double[] k = derivs(y);
            if (k.Length == y.Length)
                return k;
            // The next operation is y + h * k, which NumPy refuses for two lengths that are both above one.
            if (k.Length != 1 && y.Length != 1)
                throw new ValueError($"operands could not be broadcast together with shapes ({y.Length},) ({k.Length},) ");
            throw new NotSupportedException($"rk4 with a {k.Length}-value derivative of a {y.Length}-value system broadcasts in NumPy, which Gym.NET does not model.");
        }

        /// <summary><c>y0 + h * k</c> element by element, NumPy's two array operations rounded separately.</summary>
        /// <param name="y0">The base point.</param>
        /// <param name="h">The scale.</param>
        /// <param name="k">The derivative, as long as <paramref name="y0"/>.</param>
        /// <returns>A new array.</returns>
        private static double[] Combine(double[] y0, double h, double[] k) {
            var result = new double[y0.Length];
            for (int i = 0; i < result.Length; i++)
                result[i] = y0[i] + h * k[i];
            return result;
        }

        /// <summary>
        ///     Gymnasium's <c>_dsdt</c>: the time derivative of the augmented state, <c>(θ̇1, θ̇2, θ̈1, θ̈2, 0.0)</c>, with every
        ///     operation in the source's order (Python evaluates left to right) and every <c>x ** 2</c> the C runtime's <c>pow</c>.
        /// </summary>
        /// <param name="sAugmented">The state with the torque appended.</param>
        /// <returns>The derivative, five values.</returns>
        /// <exception cref="IndexError">Thrown when the state part holds fewer than four values, naming the first missing index, with NumPy's message.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/acrobot.py#L258">gymnasium/envs/classic_control/acrobot.py</see>.</remarks>
        private double[] Dsdt(double[] sAugmented) {
            double m1 = LinkMass1;
            double m2 = LinkMass2;
            double l1 = LinkLength1;
            double lc1 = LinkComPos1;
            double lc2 = LinkComPos2;
            double i1 = LinkMoi;
            double i2 = LinkMoi;
            const double g = 9.8;
            // a = s_augmented[-1]; s = s_augmented[:-1]; then s[0] .. s[3], the first missing one raising.
            int count = sAugmented.Length - 1;
            double a = sAugmented[count];
            if (count < 4)
                throw new IndexError($"index {count} is out of bounds for axis 0 with size {count}");
            double theta1 = sAugmented[0];
            double theta2 = sAugmented[1];
            double dtheta1 = sAugmented[2];
            double dtheta2 = sAugmented[3];
            double d1 = m1 * Math.Pow(lc1, 2) + m2 * (Math.Pow(l1, 2) + Math.Pow(lc2, 2) + 2 * l1 * lc2 * Math.Cos(theta2)) + i1 + i2;
            double d2 = m2 * (Math.Pow(lc2, 2) + l1 * lc2 * Math.Cos(theta2)) + i2;
            double phi2 = m2 * lc2 * g * Math.Cos(theta1 + theta2 - Math.PI / 2.0);
            double phi1 = -m2 * l1 * lc2 * Math.Pow(dtheta2, 2) * Math.Sin(theta2)
                          - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * Math.Sin(theta2)
                          + (m1 * lc1 + m2 * l1) * g * Math.Cos(theta1 - Math.PI / 2)
                          + phi2;
            double ddtheta2;
            if (BookOrNips == "nips") {
                // the following line is consistent with the description in the paper
                ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * Math.Pow(lc2, 2) + i2 - Math.Pow(d2, 2) / d1);
            } else {
                // the following line is consistent with the java implementation and the book
                ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * Math.Pow(dtheta1, 2) * Math.Sin(theta2) - phi2) / (m2 * Math.Pow(lc2, 2) + i2 - Math.Pow(d2, 2) / d1);
            }

            double ddtheta1 = -(d2 * ddtheta2 + phi1) / d1;
            return new[] { dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0 };
        }

        /// <summary>Gymnasium's <c>_terminal</c> on the new float64 state: <c>-cos(s[0]) - cos(s[1] + s[0]) &gt; 1.0</c>.</summary>
        /// <param name="s">The state after the step.</param>
        /// <returns>Whether the free end is above the line.</returns>
        private static bool Terminal(double[] s) => -Math.Cos(s[0]) - Math.Cos(s[1] + s[0]) > 1.0;

        /// <summary><c>s[index]</c> of the render's state, with NumPy's error for a short one.</summary>
        /// <param name="s">The state.</param>
        /// <param name="index">The index.</param>
        /// <returns>The scalar.</returns>
        /// <exception cref="IndexError">Thrown when the state is too short.</exception>
        private static NpScalar StateElement(NpScalar[] s, int index) {
            if (index >= s.Length)
                throw new IndexError($"index {index} is out of bounds for axis 0 with size {s.Length}");
            return s[index];
        }

        /// <summary>
        ///     <c>_get_ob()</c>: the new float32 array <c>[cos s[0], sin s[0], cos s[1], sin s[1], s[2], s[3]]</c>, the trigonometry
        ///     in the state's precision (NumPy's float32 functions after a reset).
        /// </summary>
        /// <returns>The observation; the caller owns it.</returns>
        private NDArray Observation() {
            NpScalar[] s = _state!;
            float[] observation = {
                s[0].NpCos().ToFloat32(), s[0].NpSin().ToFloat32(), s[1].NpCos().ToFloat32(), s[1].NpSin().ToFloat32(),
                s[2].ToFloat32(), s[3].ToFloat32(),
            };
            return np.array(observation);
        }
    }
}
