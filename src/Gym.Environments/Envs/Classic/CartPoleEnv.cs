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
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Color = Gym.Rendering.Color;
using Vector2 = Gym.Rendering.Vector2;

namespace Gym.Environments.Envs.Classic {
    /// <summary>
    ///     CartPole-v1: a pole hinged on a cart that is pushed left (action 0) or right (action 1) with a fixed force, until
    ///     the pole leans more than 12° or the cart leaves ±2.4. Byte for byte what Gymnasium's CartPole computes: the same
    ///     float64 dynamics in the same operation order, float32 observations, rewards, termination, warnings, errors, reset
    ///     options, <c>rgb_array</c> pixels and human-mode pacing.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The physics parameters are Gymnasium's instance attributes, as settable properties (<see cref="Gravity"/>,
    ///     <see cref="Tau"/>, <see cref="KinematicsIntegrator"/>, ...): changing one after construction has the effect it has
    ///     in Python, and like Python the derived ones (<see cref="TotalMass"/>, <see cref="PolemassLength"/>) are computed
    ///     once in the constructor and not recomputed. <see cref="State"/> is Gymnasium's <c>self.state</c>, a float64
    ///     <c>(x, x_dot, theta, theta_dot)</c>; the observation is always a fresh float32 copy of it.
    ///     </para>
    ///     <para>
    ///     The render mode is fixed at construction, as in Gymnasium: <c>"rgb_array"</c> makes <see cref="Render"/> return
    ///     the frame; <c>"human"</c> renders into a viewer after every <see cref="Reset"/> and <see cref="Step"/>, paced at
    ///     <c>metadata["render_fps"]</c> (50) by a <see cref="Clock"/>, and returns nothing. The viewer comes from the
    ///     factory given to the constructor, <see cref="NullEnvViewer"/> (headless) when none is.
    ///     </para>
    ///     <para>
    ///     The unwrapped env never truncates: <c>TimeLimit</c> adds the 500-step limit of CartPole-v1, as <c>make</c> does.
    ///     Not thread-safe: step, reset and render an instance from one thread at a time.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/envs/classic_control/cartpole.py#L20">gymnasium/envs/classic_control/cartpole.py</see>.</para>
    /// </remarks>
    public class CartPoleEnv : Env<NDArray, long> {
        /// <summary>
        ///     Gymnasium's warning for a step after termination, issued once (then Python's filter suppresses repeats from the
        ///     same line).
        /// </summary>
        private const string SteppedAfterTerminationWarning = "You are calling 'step()' even though this environment has already returned terminated = True. You should always call 'reset()' once you receive 'terminated = True' -- any further steps are undefined behavior.";

        /// <summary>Creates the human-mode viewer on the first human render; <see langword="null"/> means <see cref="NullEnvViewer"/>.</summary>
        private IEnvironmentViewerFactoryDelegate? _viewerFactory;

        /// <summary>The human-mode viewer, Gymnasium's display window, created on the first human render.</summary>
        private IEnvViewer? _viewer;

        /// <summary>Guards the creation of <see cref="_viewer"/>, which a viewer factory may make slow.</summary>
        private readonly object _viewerGate = new();

        /// <summary>
        ///     Creates CartPole-v1 with Gymnasium's attributes, action space <c>Discrete(2)</c> and observation space
        ///     <c>Box(-high, high, float32)</c> with <c>high = [4.8, inf, 0.41887903, inf]</c>. The generator is created lazily
        ///     from OS entropy; seed it with <c>Reset(seed: s)</c>.
        /// </summary>
        /// <param name="suttonBartoReward">
        ///     Gymnasium's <c>sutton_barto_reward</c>: <see langword="true"/> rewards 0 per surviving step and -1 on and after
        ///     termination (Sutton and Barto's original), instead of 1 per step including the terminating one.
        /// </param>
        /// <param name="renderMode">
        ///     Gymnasium's <c>render_mode</c>: <see langword="null"/> (no rendering), <c>"human"</c> or <c>"rgb_array"</c>.
        ///     Not validated, as in Gymnasium (<c>make</c> checks it against <c>metadata["render_modes"]</c>): another mode draws
        ///     each frame off-screen and <see cref="Render"/> returns nothing, as Python's does.
        /// </param>
        /// <param name="viewerFactory">Creates the human-mode window on the first human render; <see langword="null"/> means <see cref="NullEnvViewer"/>.</param>
        public CartPoleEnv(bool suttonBartoReward = false, string? renderMode = null, IEnvironmentViewerFactoryDelegate? viewerFactory = null) {
            // Gymnasium's class attribute: {"render_modes": ["human", "rgb_array"], "render_fps": 50}, keys in that order.
            Metadata = new Metadata {
                [Metadata.RenderModesKey] = new List<string> { "human", "rgb_array" },
                [Metadata.RenderFpsKey] = 50L
            };

            SuttonBartoReward = suttonBartoReward;
            // The derived attributes are computed once from the defaults, as __init__ does.
            TotalMass = Masspole + Masscart;
            PolemassLength = Masspole * Length;

            // Angle limit set to 2 * theta_threshold_radians so a failing observation is still within bounds. The float64
            // values round to float32 as np.array(..., dtype=np.float32) rounds them; infinity stays infinity, so the
            // velocity dimensions are unbounded (B-12: float.MaxValue made them bounded).
            float[] highValues = { (float)(XThreshold * 2), float.PositiveInfinity, (float)(ThetaThresholdRadians * 2), float.PositiveInfinity };
            using NDArray high = np.array(highValues);
            using NDArray low = -high;
            ActionSpace = new Discrete(2);
            // Box copies (astype) both bounds, so the temporaries above can be released.
            ObservationSpace = new Box(low, high, np.float32);

            RenderMode = renderMode;
            _viewerFactory = viewerFactory;
        }

        /// <summary>Creates a <c>human</c>-mode env rendering into an existing viewer.</summary>
        /// <param name="viewer">The viewer, Gymnasium's display window.</param>
        /// <param name="suttonBartoReward">Gymnasium's <c>sutton_barto_reward</c> (see <see cref="CartPoleEnv(bool, string, IEnvironmentViewerFactoryDelegate)"/>).</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="viewer"/> is <see langword="null"/>.</exception>
        public CartPoleEnv(IEnvViewer viewer, bool suttonBartoReward = false) : this(suttonBartoReward, "human") {
            _viewer = viewer ?? throw new ArgumentNullException(nameof(viewer));
        }

        /// <summary>Gymnasium's <c>gravity</c>, 9.8 m/s².</summary>
        public double Gravity { get; set; } = 9.8;

        /// <summary>Gymnasium's <c>masscart</c>, 1.0 kg. Changing it does not update <see cref="TotalMass"/>, as in Python.</summary>
        public double Masscart { get; set; } = 1.0;

        /// <summary>Gymnasium's <c>masspole</c>, 0.1 kg. Changing it does not update <see cref="TotalMass"/> or <see cref="PolemassLength"/>.</summary>
        public double Masspole { get; set; } = 0.1;

        /// <summary>Gymnasium's <c>total_mass</c>, <c>masspole + masscart</c> as computed at construction (1.1).</summary>
        public double TotalMass { get; set; }

        /// <summary>Gymnasium's <c>length</c>, 0.5 m: half the pole's length.</summary>
        public double Length { get; set; } = 0.5;

        /// <summary>Gymnasium's <c>polemass_length</c>, <c>masspole * length</c> as computed at construction (0.05).</summary>
        public double PolemassLength { get; set; }

        /// <summary>Gymnasium's <c>force_mag</c>, 10.0 N: the push of either action.</summary>
        public double ForceMag { get; set; } = 10.0;

        /// <summary>Gymnasium's <c>tau</c>, 0.02 s between state updates. A float64: <c>0.02f</c> moves the state at step 1 (B-11).</summary>
        public double Tau { get; set; } = 0.02;

        /// <summary>
        ///     Gymnasium's <c>kinematics_integrator</c>: <c>"euler"</c> (the default) updates positions from the old velocities;
        ///     any other string selects semi-implicit Euler, which updates velocities first.
        /// </summary>
        public string KinematicsIntegrator { get; set; } = "euler";

        /// <summary>Gymnasium's <c>theta_threshold_radians</c>, <c>12 * 2 * pi / 360</c>: the angle (±12°) that ends an episode.</summary>
        public double ThetaThresholdRadians { get; set; } = 12 * 2 * Math.PI / 360;

        /// <summary>Gymnasium's <c>x_threshold</c>, 2.4: the cart position that ends an episode.</summary>
        public double XThreshold { get; set; } = 2.4;

        /// <summary>Gymnasium's <c>screen_width</c>, 600 pixels; read when a frame is rendered.</summary>
        public int ScreenWidth { get; set; } = 600;

        /// <summary>Gymnasium's <c>screen_height</c>, 400 pixels; read when a frame is rendered.</summary>
        public int ScreenHeight { get; set; } = 400;

        /// <summary>Gymnasium's <c>_sutton_barto_reward</c>, fixed at construction.</summary>
        [PyName("_sutton_barto_reward")]
        public bool SuttonBartoReward { get; }

        /// <summary>
        ///     Gymnasium's <c>screen</c>: the surface every frame is blitted onto (the display window's in human mode),
        ///     <see langword="null"/> until the first render. <see cref="Close"/> keeps it, as Python's <c>close</c> does.
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
        ///     Gymnasium's <c>state</c>: <c>[x, x_dot, theta, theta_dot]</c> in float64, <see langword="null"/> before the first
        ///     reset. Assigning it (state injection) makes the next step start from there; <see cref="Step"/> replaces it with a
        ///     new array, so a reference read earlier keeps its values.
        /// </summary>
        public double[]? State { get; set; }

        /// <summary>
        ///     Gymnasium's <c>steps_beyond_terminated</c>: <see langword="null"/> while the pole is up, then 0 on the
        ///     terminating step and one more for every step after it.
        /// </summary>
        public int? StepsBeyondTerminated { get; set; }

        /// <summary>
        ///     Starts an episode: reseeds <see cref="Env{TObs, TAct}.NpRandom"/> when a seed is given, then draws the four
        ///     state variables from <c>uniform(low, high, size=(4,))</c>, with <c>low</c>/<c>high</c> from the options (default
        ///     ±0.05), and clears <see cref="StepsBeyondTerminated"/>. Renders in human mode.
        /// </summary>
        /// <param name="seed">The seed of a fresh <c>Generator(PCG64(SeedSequence(seed)))</c>, or <see langword="null"/> to keep the current generator.</param>
        /// <param name="options">
        ///     Gymnasium's reset options: <c>"low"</c> and <c>"high"</c>, anything Python's <c>float()</c> accepts; custom bounds
        ///     can put the state outside the observation space, as in Gymnasium.
        /// </param>
        /// <returns>The float32 observation and an empty info, <c>(obs, {})</c>.</returns>
        /// <exception cref="Error">Thrown for a negative seed, with Gymnasium's message.</exception>
        /// <exception cref="ValueError">Thrown for an option <c>float()</c> rejects, or a lower bound above the upper (see <see cref="ClassicControlUtils.MaybeParseResetBounds"/>).</exception>
        /// <exception cref="OverflowException">
        ///     Thrown, as Python's <c>OverflowError</c>, when the bounds are NaN or infinite (<c>high - low range exceeds valid
        ///     bounds</c>, from the generator) or an integer bound is beyond the float range.
        /// </exception>
        /// <remarks>
        ///     The seed is applied before the options are parsed, as <c>super().reset(seed=seed)</c> runs first: a reset that
        ///     fails on its options still leaves the freshly seeded generator, undrawn. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/envs/classic_control/cartpole.py#L228">gymnasium/envs/classic_control/cartpole.py</see>.
        /// </remarks>
        public override ResetResult<NDArray> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            SeedNpRandom(seed);
            // Note that if you use custom reset bounds, it may lead to out-of-bound state/observations.
            (double low, double high) = ClassicControlUtils.MaybeParseResetBounds(options, -0.05, 0.05);
            using (NDArray drawn = NpRandom.uniform(low, high, new Shape(4)))
                State = drawn.ToArray<double>();
            StepsBeyondTerminated = null;

            if (RenderMode == "human")
                Render();
            return new ResetResult<NDArray>(Observation(), new Info());
        }

        /// <summary>
        ///     Pushes the cart left (0) or right (1) with <see cref="ForceMag"/> and integrates one step of <see cref="Tau"/>
        ///     seconds with Gymnasium's equations, operation for operation in float64. Renders in human mode.
        /// </summary>
        /// <param name="action">An element of <c>Discrete(2)</c>: 0 or 1.</param>
        /// <returns>
        ///     The float32 observation, the reward (1.0 per step and on termination, 0.0 after it; 0.0, -1.0 and -1.0 with
        ///     <see cref="SuttonBartoReward"/>), <c>terminated</c>, <c>truncated=False</c> and <c>{}</c>.
        /// </returns>
        /// <exception cref="AssertionError">
        ///     Thrown for an action outside the action space, with Python's message (<c>2 (&lt;class 'int'&gt;) invalid</c>),
        ///     and before the first reset (<c>Call reset before using step method.</c>), in that order.
        /// </exception>
        /// <exception cref="ValueError">Thrown when an assigned <see cref="State"/> does not hold exactly four values (Python's unpacking error).</exception>
        /// <remarks>
        ///     Stepping after termination warns once through <see cref="Logger.Warn"/>, as Gymnasium does, and keeps going.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/envs/classic_control/cartpole.py#L164">gymnasium/envs/classic_control/cartpole.py</see>.
        /// </remarks>
        public override StepResult<NDArray> Step(long action) {
            // The asserts of the source, in its order: the action first, then the reset.
            AssertValidAction(action);
            double[] state = State ?? throw new AssertionError("Call reset before using step method.");
            // x, x_dot, theta, theta_dot = self.state
            if (state.Length > 4)
                throw new ValueError("too many values to unpack (expected 4)");
            if (state.Length < 4)
                throw new ValueError($"not enough values to unpack (expected 4, got {state.Length})");
            double x = state[0], xDot = state[1], theta = state[2], thetaDot = state[3];

            double force = action == 1 ? ForceMag : -ForceMag;
            double costheta = Math.Cos(theta);
            double sintheta = Math.Sin(theta);

            // For the interested reader: https://coneural.org/florian/papers/05_cart_pole.pdf
            // np.square(v) is v * v, and the products associate left to right as in the source; each parenthesis matters
            // for the last bit (B-11).
            double temp = (force + PolemassLength * (thetaDot * thetaDot) * sintheta) / TotalMass;
            double thetaacc = (Gravity * sintheta - costheta * temp) / (Length * (4.0 / 3.0 - Masspole * (costheta * costheta) / TotalMass));
            double xacc = temp - PolemassLength * thetaacc * costheta / TotalMass;

            if (KinematicsIntegrator == "euler") {
                x = x + Tau * xDot;
                xDot = xDot + Tau * xacc;
                theta = theta + Tau * thetaDot;
                thetaDot = thetaDot + Tau * thetaacc;
            } else {
                // semi-implicit euler
                xDot = xDot + Tau * xacc;
                x = x + Tau * xDot;
                thetaDot = thetaDot + Tau * thetaacc;
                theta = theta + Tau * thetaDot;
            }

            State = new[] { x, xDot, theta, thetaDot };

            bool terminated = x < -XThreshold || x > XThreshold || theta < -ThetaThresholdRadians || theta > ThetaThresholdRadians;

            double reward;
            if (!terminated) {
                reward = SuttonBartoReward ? 0.0 : 1.0;
            } else if (StepsBeyondTerminated is null) {
                // Pole just fell!
                StepsBeyondTerminated = 0;
                reward = SuttonBartoReward ? -1.0 : 1.0;
            } else {
                if (StepsBeyondTerminated == 0)
                    Logger.Warn(SteppedAfterTerminationWarning);
                StepsBeyondTerminated += 1;
                reward = SuttonBartoReward ? -1.0 : 0.0;
            }

            if (RenderMode == "human")
                Render();

            // truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`.
            return new StepResult<NDArray>(Observation(), reward, terminated, false, new Info());
        }

        /// <summary>
        ///     Draws the frame Gymnasium's CartPole draws with pygame, with the same primitives in the same order (Gym.Rendering
        ///     reproduces SDL_gfx), so the <c>rgb_array</c> pixels are Gymnasium's exactly. The track is at <c>y = 100</c> in a
        ///     y-up world, flipped vertically at the end.
        /// </summary>
        /// <returns>
        ///     <c>rgb_array</c>: the <c>(400, 600, 3)</c> uint8 frame, a new array the caller owns. <c>human</c>: nothing, after
        ///     showing the frame in the viewer at <c>render_fps</c>. Any other mode, and every mode before the first reset:
        ///     nothing, as Gymnasium returns <c>None</c>.
        /// </returns>
        /// <exception cref="AssertionError">Thrown without a render mode when the env has no spec (<c>assert self.spec is not None</c>).</exception>
        /// <exception cref="KeyError">Thrown in human mode when <c>metadata["render_fps"]</c> was removed.</exception>
        /// <exception cref="ValueError">Thrown for a NaN coordinate (a NaN state), with pygame's message.</exception>
        /// <exception cref="OverflowException">
        ///     Thrown, as pygame's <c>OverflowError</c>, for a coordinate pygame cannot draw: beyond a C long in a polygon
        ///     (<c>x = 1e10</c>), or beyond Sint16 in the axle's circle (<c>x = 300</c>), or an infinite one.
        /// </exception>
        /// <remarks>
        ///     Without a render mode, Gymnasium asserts the env has a spec and warns once (<c>You are calling render method
        ///     without specifying any render mode. ...</c>), naming the spec's id; an env built with its constructor has no
        ///     spec, so it raises the bare <see cref="AssertionError"/>, as Python's does. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/envs/classic_control/cartpole.py#L249">gymnasium/envs/classic_control/cartpole.py</see>.
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
                    EnsureViewer();
                // mode == "rgb_array" draws off-screen; in human mode this surface stands for the display surface.
                Screen = new Surface(ScreenWidth, ScreenHeight);
            }

            Clock ??= new Clock();

            double worldWidth = XThreshold * 2;
            double scale = ScreenWidth / worldWidth;
            const double polewidth = 10.0;
            double polelen = scale * (2 * Length);
            const double cartwidth = 50.0;
            const double cartheight = 30.0;

            if (State is null)
                return RenderResult.None;

            double[] x = State;

            var surf = new Surface(ScreenWidth, ScreenHeight);
            Surf = surf;
            surf.Fill(new Color(255, 255, 255));

            double l = -cartwidth / 2, r = cartwidth / 2, t = cartheight / 2, b = -cartheight / 2;
            double axleoffset = cartheight / 4.0;
            double cartx = x[0] * scale + ScreenWidth / 2.0; // MIDDLE OF CART
            const int carty = 100; // TOP OF CART
            var cartCoords = new List<(double X, double Y)>(4);
            foreach ((double X, double Y) c in new[] { (l, b), (l, t), (r, t), (r, b) })
                cartCoords.Add((c.X + cartx, c.Y + carty));
            GfxDraw.AAPolygon(surf, cartCoords, new Color(0, 0, 0));
            GfxDraw.FilledPolygon(surf, cartCoords, new Color(0, 0, 0));

            l = -polewidth / 2;
            r = polewidth / 2;
            t = polelen - polewidth / 2;
            b = -polewidth / 2;

            var poleCoords = new List<(double X, double Y)>(4);
            foreach ((double X, double Y) coord in new[] { (l, b), (l, t), (r, t), (r, b) }) {
                Vector2 rotated = new Vector2(coord.X, coord.Y).RotateRad(-x[2]);
                // (coord[1] + carty) + axleoffset, left to right as in the source.
                poleCoords.Add((rotated.X + cartx, rotated.Y + carty + axleoffset));
            }

            GfxDraw.AAPolygon(surf, poleCoords, new Color(202, 152, 101));
            GfxDraw.FilledPolygon(surf, poleCoords, new Color(202, 152, 101));

            // int(...) of each argument first (left to right), then pygame's range checks, as Python evaluates the call.
            long axleX = PythonIntArgument(cartx), axleY = PythonIntArgument(carty + axleoffset), axleRadius = PythonIntArgument(polewidth / 2);
            GfxDraw.AACircle(surf, axleX, axleY, axleRadius, new Color(129, 132, 203));
            axleX = PythonIntArgument(cartx);
            axleY = PythonIntArgument(carty + axleoffset);
            axleRadius = PythonIntArgument(polewidth / 2);
            GfxDraw.FilledCircle(surf, axleX, axleY, axleRadius, new Color(129, 132, 203));

            GfxDraw.HLine(surf, 0, ScreenWidth, carty, new Color(0, 0, 0));

            surf = Transform.Flip(surf, false, true);
            Surf = surf;
            Screen.Blit(surf, 0, 0);
            if (RenderMode == "human") {
                // pygame.event.pump() has no event queue to pump here; the viewer runs its own message loop.
                Clock.Tick(PyBuiltins.Float(Metadata[Metadata.RenderFpsKey]));
                // pygame.display.flip(): the viewer copies the frame synchronously, so it is released right after. A window
                // closed by Close() is reopened from the factory (pygame would raise "display Surface quit" instead).
                EnsureViewer();
                using Image<Rgb24> frame = Image.LoadPixelData<Rgb24>(Screen.ToRgbBytes(), Screen.Width, Screen.Height);
                _viewer!.Render(frame);
            } else if (RenderMode == "rgb_array") {
                // np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)): rows of RGB.
                return new RenderResult.Rgb(SurfArray.ToRgbArray(Screen));
            }

            return RenderResult.None;
        }

        /// <summary>
        ///     Once something was rendered, shuts the human-mode window down and marks the env closed (<see cref="IsOpen"/>).
        ///     Like Gymnasium's, it keeps <see cref="Screen"/>, so a later <c>rgb_array</c> render reuses it; a later human
        ///     render opens a new window from the factory.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/envs/classic_control/cartpole.py#L346">gymnasium/envs/classic_control/cartpole.py</see>.</remarks>
        public override void Close() {
            if (Screen is not null) {
                // pygame.display.quit(); pygame.quit()
                lock (_viewerGate) {
                    if (_viewer != null) {
                        _viewer.CloseEnvironment();
                        _viewer = null;
                    }
                }

                IsOpen = false;
            }
        }

        /// <summary>
        ///     The untyped step's action conversion, with CartPole's own assert first: an action that is not an element of
        ///     <c>Discrete(2)</c> fails with Python's message for exactly the value given (<c>np.int64(5) (&lt;class
        ///     'numpy.int64'&gt;) invalid</c>, <c>'x' (&lt;class 'str'&gt;) invalid</c>).
        /// </summary>
        /// <param name="action">The action as given to <see cref="IEnv.Step"/>: any integer, a <see langword="bool"/>, an enum member, or a 0-d integer array.</param>
        /// <returns>The action as <see langword="long"/>.</returns>
        /// <exception cref="AssertionError">Thrown for anything <c>Discrete(2).contains</c> rejects.</exception>
        protected override long ConvertAction(object? action) {
            AssertValidAction(action);
            return Discrete.ToInt64(action!);
        }

        /// <summary><c>assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"</c>.</summary>
        /// <param name="action">The action, as given (so the message shows its Python counterpart).</param>
        /// <exception cref="AssertionError">Thrown when the action space does not contain the action.</exception>
        private void AssertValidAction(object? action) {
            if (!ActionSpace.Contains(action!))
                throw new AssertionError($"{PyBuiltins.Repr(action)} ({PyBuiltins.TypeRepr(action)}) invalid");
        }

        /// <summary><c>np.array(self.state, dtype=np.float32)</c>: a new float32 array, never an alias of <see cref="State"/> (B-13).</summary>
        /// <returns>The observation; the caller owns it.</returns>
        private NDArray Observation() {
            double[] state = State!;
            var observation = new float[state.Length];
            for (int i = 0; i < state.Length; i++)
                // Round-to-nearest-even narrowing, as NumPy's float64 -> float32 cast.
                observation[i] = (float)state[i];
            return np.array(observation);
        }

        /// <summary>
        ///     Python's <c>int(value)</c> as the circle's argument: the truncated integer, saturated to <see langword="long"/>,
        ///     which keeps pygame's verdict (any value beyond a C long is rejected with the same message).
        /// </summary>
        /// <param name="value">The float coordinate.</param>
        /// <returns>The integer for <see cref="GfxDraw"/>.</returns>
        /// <exception cref="ValueError">Thrown for NaN (<c>cannot convert float NaN to integer</c>).</exception>
        /// <exception cref="OverflowException">Thrown for an infinity (<c>cannot convert float infinity to integer</c>).</exception>
        private static long PythonIntArgument(double value) {
            BigInteger integer = PyBuiltins.Int(value);
            if (integer > long.MaxValue)
                return long.MaxValue;
            if (integer < long.MinValue)
                return long.MinValue;
            return (long)integer;
        }

        /// <summary>Creates the human-mode viewer from the factory once, under a lock (a factory may start a UI thread).</summary>
        private void EnsureViewer() {
            if (_viewer != null)
                return;
            lock (_viewerGate) {
                if (_viewer != null)
                    return;
                _viewerFactory ??= NullEnvViewer.Factory;
                _viewer = _viewerFactory(ScreenWidth, ScreenHeight, "cartpole-v1").GetAwaiter().GetResult();
            }
        }
    }
}
