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
    ///     Pendulum-v1: a pendulum hanging from a pivot, swung up and balanced by a torque in [-2, 2]. The reward is the
    ///     negative cost <c>θ² + 0.1·θ̇² + 0.001·u²</c> (θ normalized to [-π, π)), so it is at most 0, reached upright and at
    ///     rest; the env never terminates. Byte for byte what Gymnasium's Pendulum computes, errors included.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The torque keeps the action's precision, as NumPy's scalar does: a float32 action (the action space's dtype) is
    ///     clipped, squared and scaled in float32 before it joins the float64 dynamics, a float64 one in float64, and an
    ///     integer or boolean one is clipped as float64 (NumPy's <c>clip</c> with Python-float bounds). The state is float64
    ///     <c>[θ, θ̇]</c> throughout; the observation is the float32 <c>[cos θ, sin θ, θ̇]</c>.
    ///     </para>
    ///     <para>
    ///     Rendering is Gymnasium's pixel for pixel: the rod, its joints, and the clockwise arrow image (embedded from
    ///     Gymnasium, DR-19) shrunk with <see cref="Transform.Smoothscale"/> to the last torque's size and mirrored by its
    ///     sign. Any render mode other than <c>"human"</c> returns the frame, as Gymnasium's <c>else</c> branch does. Human
    ///     mode renders into the window from <c>viewer_factory</c> or <c>Gymnasium.HumanViewerFactory</c> (DR-20) at 30 frames
    ///     per second.
    ///     </para>
    ///     <para>
    ///     Not thread-safe. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/pendulum.py#L16">gymnasium/envs/classic_control/pendulum.py</see>.
    ///     </para>
    /// </remarks>
    [GymOptions]
    public partial class PendulumEnv : Env<NDArray, NDArray> {
        /// <summary>Gymnasium's <c>DEFAULT_X</c>: the default bound of the initial angle, π.</summary>
        private const double DefaultX = Math.PI;

        /// <summary>Gymnasium's <c>DEFAULT_Y</c>: the default bound of the initial angular velocity, 1.0.</summary>
        private const double DefaultY = 1.0;

        /// <summary>The human-mode window, opened on the first human render.</summary>
        private readonly HumanWindow _window;

        /// <summary>Python's <c>self.state</c>; <see langword="null"/> when it is <c>None</c> (assigned) or missing (before the first reset, see <see cref="_hasState"/>).</summary>
        private double[]? _state;

        /// <summary>Whether Python's <c>self.state</c> attribute exists: set by the first reset or an assignment of <see cref="State"/>.</summary>
        private bool _hasState;

        /// <summary>Python's <c>self.last_u</c>, the last clipped torque as the NumPy scalar it is; <see langword="null"/> for <c>None</c> or missing.</summary>
        private NpScalar? _lastU;

        /// <summary>Whether Python's <c>self.last_u</c> attribute exists: set by the first reset, step or assignment of <see cref="LastU"/>.</summary>
        private bool _hasLastU;

        /// <summary>Creates the env.</summary>
        /// <param name="renderMode">
        ///     Gymnasium's <c>render_mode</c>: <see langword="null"/>, <c>"human"</c> or <c>"rgb_array"</c>; not validated, as in
        ///     Gymnasium (<c>make</c> checks it against the metadata).
        /// </param>
        /// <param name="g">Gymnasium's <c>g</c>: the gravitational acceleration, 10.0 by default (<see cref="Gravity"/>).</param>
        /// <param name="viewerFactory">Gym.NET's <c>viewer_factory</c> (DR-20) for human mode; <see langword="null"/> uses <c>Gymnasium.HumanViewerFactory</c>.</param>
        public PendulumEnv(string? renderMode = null, double g = 10.0, IEnvironmentViewerFactoryDelegate? viewerFactory = null) {
            Metadata = ClassMetadata;
            Gravity = g;

            // np.array([1.0, 1.0, self.max_speed], dtype=np.float32), and low = -high.
            float[] highValues = { 1.0f, 1.0f, (float)MaxSpeed };
            using NDArray high = np.array(highValues);
            using NDArray low = -high;
            // This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric or
            // normalised as max_torque == 2 by default (Gymnasium's own note: the defaults are kept for compatibility).
            ActionSpace = new Box(-MaxTorque, MaxTorque, new Shape(1), np.float32);
            // Box copies its bounds, so the temporaries are released here.
            ObservationSpace = new Box(low, high, dtype: np.float32);

            RenderMode = renderMode;
            _window = new HumanWindow("pendulum-v1", viewerFactory);
        }

        /// <summary>
        ///     Gymnasium's class attribute <c>metadata</c>: <c>{"render_modes": ["human", "rgb_array"], "render_fps": 30}</c>.
        ///     Every env starts with a copy as its instance metadata.
        /// </summary>
        /// <remarks>A new dict on every read, so editing one env's metadata never reaches another (Python's is one shared dict).</remarks>
        public static new Metadata ClassMetadata => new() {
            [Metadata.RenderModesKey] = new List<string> { "human", "rgb_array" },
            [Metadata.RenderFpsKey] = 30L,
        };

        /// <summary>Gymnasium's <c>max_speed</c>, 8: the clamp of the angular velocity, read every step.</summary>
        public double MaxSpeed { get; set; } = 8;

        /// <summary>Gymnasium's <c>max_torque</c>, 2.0: the clamp of the torque, read every step (the action space keeps its construction-time bounds).</summary>
        public double MaxTorque { get; set; } = 2.0;

        /// <summary>Gymnasium's <c>dt</c>, 0.05 s per step.</summary>
        public double Dt { get; set; } = 0.05;

        /// <summary>Gymnasium's <c>g</c>: the gravitational acceleration, from the constructor.</summary>
        [PyName("g")]
        public double Gravity { get; set; }

        /// <summary>Gymnasium's <c>m</c>, 1.0: the pendulum's mass.</summary>
        [PyName("m")]
        public double Mass { get; set; } = 1.0;

        /// <summary>Gymnasium's <c>l</c>, 1.0: the rod's length.</summary>
        [PyName("l")]
        public double Length { get; set; } = 1.0;

        /// <summary>Gymnasium's <c>screen_dim</c>, 500: the frame's width and height in pixels, read when a frame is rendered.</summary>
        public int ScreenDim { get; set; } = 500;

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
        ///     Gymnasium's <c>state</c>, the float64 <c>[θ, θ̇]</c>: a new array on every read, or <see langword="null"/> before
        ///     the first reset (and after <see langword="null"/> was assigned). Assigning injects a state, as
        ///     <c>env.state = np.array([...])</c> does, of any length: the next step or render checks it with Python's errors.
        /// </summary>
        /// <remarks>Assigning <see langword="null"/> stands for Python's <c>None</c>: the next step or render raises the <see cref="TypeError"/> Python raises for it.</remarks>
        public double[]? State {
            get => _state is null ? null : (double[])_state.Clone();
            set {
                _hasState = true;
                _state = value is null ? null : (double[])value.Clone();
            }
        }

        /// <summary>
        ///     Gymnasium's <c>last_u</c>: the last step's clipped torque, which the next render draws as the arrow;
        ///     <see langword="null"/> after a reset (no arrow) and before the first one. Assigning a value draws that torque's
        ///     arrow as a float64.
        /// </summary>
        /// <remarks>
        ///     Python's attribute is missing until the first reset or step, so a render of an env whose state was injected
        ///     before any reset raises <see cref="AttributeError"/>; reading it here gives <see langword="null"/> then.
        /// </remarks>
        [PyName("last_u")]
        public double? LastU {
            get => _lastU?.Value;
            set {
                _hasLastU = true;
                _lastU = value is null ? null : NpScalar.Float64(value.Value);
            }
        }

        /// <summary>
        ///     Starts an episode: reseeds the generator when a seed is given, then draws <c>[θ, θ̇]</c> from
        ///     <c>uniform(-high, high)</c> with <c>high = [π, 1.0]</c>, or <c>[x_init, y_init]</c> from the options, and clears
        ///     the arrow. Renders in human mode.
        /// </summary>
        /// <param name="seed">The seed of a fresh <c>Generator(PCG64(SeedSequence(seed)))</c>, or <see langword="null"/> to keep the generator.</param>
        /// <param name="options">
        ///     Gymnasium's reset options: <c>"x_init"</c> and <c>"y_init"</c>, anything Python's <c>float()</c> accepts; a missing
        ///     key keeps its default. Custom bounds can put the state outside the observation space, as in Gymnasium.
        /// </param>
        /// <returns>The float32 observation <c>[cos θ, sin θ, θ̇]</c> and an empty info.</returns>
        /// <exception cref="Error">Thrown for a negative seed, with Gymnasium's message.</exception>
        /// <exception cref="ValueError">
        ///     Thrown for an option <c>float()</c> rejects (<c>An option (None) could not be converted to a float.</c>), and by the
        ///     generator for a negative bound (<c>high - low &lt; 0</c>).
        /// </exception>
        /// <exception cref="OverflowException">Thrown, as Python's <c>OverflowError</c>, for a NaN or infinite bound (<c>Range exceeds valid bounds</c>).</exception>
        /// <exception cref="DependencyNotInstalled">Thrown in human mode by the first render when no viewer factory is set (DR-20).</exception>
        /// <remarks>
        ///     The seed is applied before the options are read, as <c>super().reset(seed=seed)</c> runs first; a failed reset
        ///     leaves the freshly seeded generator, undrawn, and the old state. Reference:
        ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/pendulum.py#L149">gymnasium/envs/classic_control/pendulum.py</see>.
        /// </remarks>
        public override ResetResult<NDArray> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            SeedNpRandom(seed);
            double x, y;
            if (options is null) {
                x = DefaultX;
                y = DefaultY;
            } else {
                // Note that if you use custom reset bounds, it may lead to out-of-bound state/observations. A present key
                // wins even when its value is None, which the cast then rejects.
                object? xInit = options.TryGetValue("x_init", out object? givenX) ? givenX : DefaultX;
                object? yInit = options.TryGetValue("y_init", out object? givenY) ? givenY : DefaultY;
                x = ClassicControlUtils.VerifyNumberAndCast(xInit);
                y = ClassicControlUtils.VerifyNumberAndCast(yInit);
            }

            double[] highValues = { x, y };
            using NDArray high = np.array(highValues);
            // We enforce symmetric limits.
            using NDArray low = -high;
            using (NDArray drawn = NpRandom.uniform(low, high))
                _state = drawn.ToArray<double>();
            _hasState = true;
            _lastU = null;
            _hasLastU = true;

            if (RenderMode == "human")
                Render();
            return new ResetResult<NDArray>(Observation(), new Info());
        }

        /// <summary>
        ///     Applies the clipped torque for <see cref="Dt"/> seconds: <c>θ̇ += (3g/(2l)·sin θ + 3/(m·l²)·u)·dt</c>, clipped to
        ///     ±<see cref="MaxSpeed"/>, then <c>θ += θ̇·dt</c>. Renders in human mode.
        /// </summary>
        /// <param name="action">
        ///     A one-dimensional array whose first element is the torque, clipped to ±<see cref="MaxTorque"/> in the array's
        ///     precision (float32 for the action space's samples). Read, never kept.
        /// </param>
        /// <returns>The float32 observation, the reward <c>-cost</c> (float64, never positive), <c>terminated=False</c>, <c>truncated=False</c> and <c>{}</c>.</returns>
        /// <exception cref="AttributeError">Thrown before the first reset (<c>'PendulumEnv' object has no attribute 'state'</c>).</exception>
        /// <exception cref="TypeError">Thrown when <see cref="State"/> was assigned <see langword="null"/> (<c>cannot unpack non-iterable NoneType object</c>).</exception>
        /// <exception cref="ValueError">Thrown when an assigned state does not hold exactly two values (Python's unpacking error).</exception>
        /// <exception cref="IndexError">
        ///     Thrown for a 0-d action (<c>invalid index to scalar variable.</c>, as <c>np.clip</c> returns a scalar for it) or an
        ///     empty one, with NumPy's messages.
        /// </exception>
        /// <exception cref="NotSupportedException">Thrown for an action Gym.NET does not model: more than one dimension, or a float16 or complex dtype.</exception>
        /// <exception cref="DivideByZeroException">Thrown, as Python's <c>ZeroDivisionError</c>, when <see cref="Length"/> or <see cref="Mass"/> was set to zero (<c>float division by zero</c>).</exception>
        /// <exception cref="DependencyNotInstalled">Thrown in human mode by the first render when no viewer factory is set (DR-20).</exception>
        /// <remarks>
        ///     The state is read before the action, as the source unpacks it first. Every <c>x ** 2</c> is the C runtime's
        ///     <c>pow</c> (<c>powf</c> for a float32 torque), as NumPy computes it, never <c>x * x</c>. Reference:
        ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/pendulum.py#L126">gymnasium/envs/classic_control/pendulum.py</see>.
        /// </remarks>
        public override StepResult<NDArray> Step(NDArray action) {
            // th, thdot = self.state  # th := theta
            double[] state = RequireState();
            if (state.Length != 2)
                throw new ValueError(state.Length > 2 ? "too many values to unpack (expected 2)" : $"not enough values to unpack (expected 2, got {state.Length})");
            double th = state[0], thdot = state[1];

            double g = Gravity, m = Mass, l = Length, dt = Dt;

            NpScalar u = ClippedTorque(action);
            // for rendering
            _lastU = u;
            _hasLastU = true;
            // angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2): the last term in u's precision, joined in float64.
            double costs = Math.Pow(AngleNormalize(th), 2) + 0.1 * Math.Pow(thdot, 2) + (NpScalar.PyFloat(0.001) * u.NpPow(2)).Value;

            // 3 * g / (2 * l) * np.sin(th) is float64; 3.0 / (m * l**2) * u is in u's precision; their sum is float64. Both
            // quotients are Python float divisions, evaluated left to right, so a zero length or mass raises as in Python.
            double gravityTerm = PythonDivide(3 * g, 2 * l) * Math.Sin(th);
            double torqueTerm = (NpScalar.PyFloat(PythonDivide(3.0, m * Math.Pow(l, 2))) * u).Value;
            double newthdot = thdot + (gravityTerm + torqueTerm) * dt;
            newthdot = PyMath.Clip(newthdot, -MaxSpeed, MaxSpeed);
            double newth = th + newthdot * dt;

            _state = new[] { newth, newthdot };

            if (RenderMode == "human")
                Render();
            // truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`. -costs is a
            // negation, so a zero cost is a -0.0 reward, as in Python.
            return new StepResult<NDArray>(Observation(), -costs, false, false, new Info());
        }

        /// <summary>
        ///     Draws Gymnasium's frame with the same primitives in the same order: the rod rotated by θ + π/2, its two round ends,
        ///     the torque arrow when there is one, and the axle, in a y-up world flipped at the end.
        /// </summary>
        /// <returns>
        ///     For any mode but <c>human</c>: the <c>(500, 500, 3)</c> uint8 frame, a new array the caller owns (Gymnasium's render
        ///     returns it from its <c>else</c> branch). <c>human</c>: nothing.
        /// </returns>
        /// <exception cref="AssertionError">Thrown without a render mode when the env has no spec (<c>assert self.spec is not None</c>).</exception>
        /// <exception cref="AttributeError">
        ///     Thrown before the first reset (no <c>state</c>), and for an injected state before any reset or step (no
        ///     <c>last_u</c>, after the rod was drawn), as in Gymnasium.
        /// </exception>
        /// <exception cref="TypeError">Thrown when <see cref="State"/> was assigned <see langword="null"/>.</exception>
        /// <exception cref="IndexError">Thrown when an assigned state is empty.</exception>
        /// <exception cref="KeyError">Thrown in human mode when <c>metadata["render_fps"]</c> was removed.</exception>
        /// <exception cref="ValueError">
        ///     Thrown for a NaN coordinate (a NaN state), with pygame's message, and for an arrow of no valid size (a NaN
        ///     <see cref="LastU"/>: <c>Cannot scale to negative size</c>).
        /// </exception>
        /// <exception cref="OverflowException">Thrown, as pygame's <c>OverflowError</c>, for a coordinate pygame cannot draw.</exception>
        /// <exception cref="NotSupportedException">Thrown when a torque beyond the image's size (a raised <see cref="MaxTorque"/>) would make smoothscale grow the arrow, which Gym.NET does not implement yet (P7).</exception>
        /// <exception cref="DependencyNotInstalled">Thrown on the first human render when no viewer factory is set (DR-20).</exception>
        /// <remarks>
        ///     The arrow's size is <c>float(scale * np.abs(last_u) / 2)</c>, computed in the torque's precision and truncated as
        ///     pygame's C code truncates it (<see cref="RenderMath.CInt"/>). Reference:
        ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/pendulum.py#L173">gymnasium/envs/classic_control/pendulum.py</see>.
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
            Surf = surf;
            surf.Fill(new Color(255, 255, 255));

            const double bound = 2.2;
            double scale = ScreenDim / (bound * 2);
            // self.screen_dim // 2: a Python int (the dimension is positive, or the surfaces above would have failed).
            int offset = ScreenDim / 2;

            double rodLength = 1 * scale;
            double rodWidth = 0.2 * scale;
            double l = 0, r = rodLength, t = rodWidth / 2, b = -rodWidth / 2;
            // self.state[0] + np.pi / 2, read as the first coordinate is rotated.
            double angle = FirstStateValue() + Math.PI / 2;
            var transformedCoords = new List<(double X, double Y)>(4);
            foreach ((double X, double Y) c in new[] { (l, b), (l, t), (r, t), (r, b) }) {
                Vector2 rotated = new Vector2(c.X, c.Y).RotateRad(angle);
                transformedCoords.Add((rotated.X + offset, rotated.Y + offset));
            }

            var rodColor = new Color(204, 77, 77);
            GfxDraw.AAPolygon(surf, transformedCoords, rodColor);
            GfxDraw.FilledPolygon(surf, transformedCoords, rodColor);

            GfxDraw.AACircle(surf, offset, offset, RenderMath.Int(rodWidth / 2), rodColor);
            GfxDraw.FilledCircle(surf, offset, offset, RenderMath.Int(rodWidth / 2), rodColor);

            Vector2 rodEnd = new Vector2(rodLength, 0).RotateRad(angle);
            // (int(rod_end[0] + offset), int(rod_end[1] + offset)), both converted before either circle is drawn.
            long endX = RenderMath.Int(rodEnd.X + offset), endY = RenderMath.Int(rodEnd.Y + offset);
            GfxDraw.AACircle(surf, endX, endY, RenderMath.Int(rodWidth / 2), rodColor);
            GfxDraw.FilledCircle(surf, endX, endY, RenderMath.Int(rodWidth / 2), rodColor);

            // pygame.image.load(fname) on every render in Python; the decoded image is shared here, and only read.
            Surface img = EnvAssets.Load("classic_control/clockwise.png");
            if (!_hasLastU)
                throw new AttributeError("'PendulumEnv' object has no attribute 'last_u'");
            if (_lastU is { } lastU) {
                // float(scale * np.abs(self.last_u) / 2), in the torque's precision, then pygame's C int.
                int side = RenderMath.CInt((NpScalar.PyFloat(scale) * lastU.Abs() / NpScalar.PyInt(2)).Value);
                Surface scaleImg = Transform.Smoothscale(img, side, side);
                bool isFlip = lastU > NpScalar.PyInt(0);
                scaleImg = Transform.Flip(scaleImg, isFlip, true);
                // offset - scale_img.get_rect().centerx: a rect at the origin has its center at (w // 2, h // 2).
                surf.Blit(scaleImg, offset - scaleImg.Width / 2, offset - scaleImg.Height / 2);
            }

            // drawing axle
            GfxDraw.AACircle(surf, offset, offset, RenderMath.Int(0.05 * scale), new Color(0, 0, 0));
            GfxDraw.FilledCircle(surf, offset, offset, RenderMath.Int(0.05 * scale), new Color(0, 0, 0));

            surf = Transform.Flip(surf, false, true);
            Surf = surf;
            Screen.Blit(surf, 0, 0);
            if (RenderMode == "human") {
                // pygame.event.pump() has no event queue to pump here; the viewer runs its own message loop.
                Clock.Tick(PyBuiltins.Float(Metadata[Metadata.RenderFpsKey]));
                // pygame.display.flip(): a window shut by Close() is reopened from the factory.
                _window.Show(Screen);
                return RenderResult.None;
            }

            // else:  # mode == "rgb_array": np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)).
            return new RenderResult.Rgb(SurfArray.ToRgbArray(Screen));
        }

        /// <summary>
        ///     Once something was rendered, shuts the human-mode window down and marks the env closed (<see cref="IsOpen"/>).
        ///     Like Gymnasium's, it keeps <see cref="Screen"/>, so a later render reuses it.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/pendulum.py#L272">gymnasium/envs/classic_control/pendulum.py</see>.</remarks>
        public override void Close() {
            if (Screen is not null) {
                // pygame.display.quit(); pygame.quit()
                _window.Close();
                IsOpen = false;
            }
        }

        /// <summary>
        ///     Gymnasium's module function <c>angle_normalize</c>: <c>((x + π) % 2π) - π</c>, the angle in [-π, π) with NumPy's
        ///     float remainder (the divisor's sign).
        /// </summary>
        /// <param name="x">The angle.</param>
        /// <returns>The normalized angle; NaN for a NaN or infinite one.</returns>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/pendulum.py#L281">gymnasium/envs/classic_control/pendulum.py</see>.</remarks>
        public static double AngleNormalize(double x) => PyMath.Mod(x + Math.PI, 2 * Math.PI) - Math.PI;

        /// <summary>
        ///     <c>np.clip(u, -self.max_torque, self.max_torque)[0]</c>: the first element clipped in the action's precision, with
        ///     NumPy's errors for an action that has no first element.
        /// </summary>
        /// <param name="action">The action.</param>
        /// <returns>The torque: a float32 scalar for a float32 action, a float64 one otherwise.</returns>
        /// <exception cref="IndexError">Thrown for a 0-d action (NumPy's clip returns a scalar, which <c>[0]</c> cannot index) or an empty one.</exception>
        /// <exception cref="NotSupportedException">Thrown for more than one dimension, or a float16 or complex dtype.</exception>
        private NpScalar ClippedTorque(NDArray action) {
            // NumPy's own errors first, in the order the expression meets them, whatever the dtype.
            if (action.ndim == 0)
                throw new IndexError("invalid index to scalar variable.");
            if (action.shape[0] == 0)
                throw new IndexError("index 0 is out of bounds for axis 0 with size 0");
            // Python would carry a row (or a float16 scalar) into a state of another shape or precision; refusing it is better
            // than silently computing something else.
            if (action.ndim > 1)
                throw new NotSupportedException($"Gym.NET's PendulumEnv takes a one-dimensional action; a {action.ndim}-dimensional one would make Gymnasium's state an array of rows.");
            if (action.typecode is NPTypeCode.Half or NPTypeCode.Complex)
                throw new NotSupportedException($"Gym.NET's PendulumEnv does not model the arithmetic of a {action.typecode} action; pass float32 (the action space's dtype) or float64.");

            return action.typecode switch {
                // The Python-float bounds are weak, so a float32 array clips against float32 bounds and stays float32.
                NPTypeCode.Single => NpScalar.Float32(PyMath.Clip(action.GetAtIndex<float>(0), (float)-MaxTorque, (float)MaxTorque)),
                NPTypeCode.Double => NpScalar.Float64(PyMath.Clip(action.GetAtIndex<double>(0), -MaxTorque, MaxTorque)),
                // An integer or boolean array with Python-float bounds clips as float64.
                _ => NpScalar.Float64(PyMath.Clip(NpScalar.Element(action, 0).Value, -MaxTorque, MaxTorque)),
            };
        }

        /// <summary>Python's <c>a / b</c> on floats: IEEE division, except that a zero divisor raises instead of giving an infinity.</summary>
        /// <param name="a">The dividend.</param>
        /// <param name="b">The divisor.</param>
        /// <returns>The quotient.</returns>
        /// <exception cref="DivideByZeroException">Thrown, as Python's <c>ZeroDivisionError</c>, for a zero divisor (<c>float division by zero</c>).</exception>
        private static double PythonDivide(double a, double b) {
            if (b == 0)
                throw new DivideByZeroException("float division by zero");
            return a / b;
        }

        /// <summary><c>self.state[0]</c>, with Python's errors for a missing, <c>None</c> or empty state.</summary>
        /// <returns>The angle.</returns>
        /// <exception cref="AttributeError">Thrown before the first reset.</exception>
        /// <exception cref="TypeError">Thrown for a <c>None</c> state.</exception>
        /// <exception cref="IndexError">Thrown for an empty state, with NumPy's message.</exception>
        private double FirstStateValue() {
            if (!_hasState)
                throw new AttributeError("'PendulumEnv' object has no attribute 'state'");
            double[] state = _state ?? throw new TypeError("'NoneType' object is not subscriptable");
            if (state.Length == 0)
                throw new IndexError("index 0 is out of bounds for axis 0 with size 0");
            return state[0];
        }

        /// <summary>The state for unpacking, with Python's errors for a missing or <c>None</c> one.</summary>
        /// <returns>The values; their count is the caller's to check.</returns>
        /// <exception cref="AttributeError">Thrown before the first reset.</exception>
        /// <exception cref="TypeError">Thrown for a <c>None</c> state (<c>cannot unpack non-iterable NoneType object</c>).</exception>
        private double[] RequireState() {
            if (!_hasState)
                throw new AttributeError("'PendulumEnv' object has no attribute 'state'");
            return _state ?? throw new TypeError("cannot unpack non-iterable NoneType object");
        }

        /// <summary><c>_get_obs()</c>: the new float32 array <c>[cos θ, sin θ, θ̇]</c>, the float64 values rounded to nearest.</summary>
        /// <returns>The observation; the caller owns it.</returns>
        private NDArray Observation() {
            double theta = _state![0], thetadot = _state[1];
            float[] observation = { (float)Math.Cos(theta), (float)Math.Sin(theta), (float)thetadot };
            return np.array(observation);
        }
    }
}
