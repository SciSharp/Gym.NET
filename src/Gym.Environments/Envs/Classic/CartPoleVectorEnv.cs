#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Exceptions;
using Gym.Rendering;
using Gym.Spaces;
using Gym.Utils;
using Gym.Vector;
using NumSharp;
using Color = Gym.Rendering.Color;
using Vector2 = Gym.Rendering.Vector2;

namespace Gym.Environments.Envs.Classic {
    /// <summary>
    ///     CartPole vectorized natively: <see cref="VectorEnv{TObs, TAct}.NumEnvs"/> carts stepped together as arrays, with
    ///     their own time limit and autoreset (<c>NEXT_STEP</c>), which <c>MakeVec("CartPole-v1", numEnvs: n)</c> builds
    ///     through the spec's vector entry point. Byte for byte what Gymnasium's <c>CartPoleVectorEnv</c> computes: the
    ///     dynamics of every cart, the generator draws (one <c>(4, num_envs)</c> draw at reset, one <c>(4, k)</c> draw per step
    ///     for the <c>k</c> carts that ended at the previous one), the observations and flags, and the frames.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Observations are a <c>(num_envs, 4)</c> float32 array; actions a <c>MultiDiscrete([2, ...])</c> sample, 0 pushing
    ///     left and 1 right. A cart whose episode ended (pole past 12°, cart past 2.4, or <c>max_episode_steps</c> steps) is
    ///     re-drawn at the next step, which returns its new state with a reward of 0 and no flags, as Gymnasium's does.
    ///     </para>
    ///     <para>
    ///     Rewards are a float64 array here (DR-21: Gym.NET's rewards are float64, single or batched) where Gymnasium's are
    ///     float32: the values are the same (1, 0, and -1 or -0 with <c>sutton_barto_reward</c>), the dtype is not. Like
    ///     Gymnasium's, it renders <c>rgb_array</c> frames only (a list, one per cart) and has no human window. Not
    ///     thread-safe.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/cartpole.py#L356">gymnasium/envs/classic_control/cartpole.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class CartPoleVectorEnv : VectorEnv<NDArray, NDArray> {
        /// <summary>The carts' state, <c>(4, num_envs)</c> float64 by row (x, x_dot, theta, theta_dot); <see langword="null"/> before the first reset.</summary>
        private double[,]? _state;

        /// <summary>Each cart's steps in its episode (Python's int32 <c>steps</c>).</summary>
        private int[] _steps;

        /// <summary>Which carts ended at the previous step and are re-drawn at the next (Python's <c>prev_done</c>).</summary>
        private bool[] _prevDone;

        /// <summary>Creates the carts.</summary>
        /// <param name="numEnvs">Gymnasium's <c>num_envs</c>: the number of carts.</param>
        /// <param name="maxEpisodeSteps">Gymnasium's <c>max_episode_steps</c>: the steps after which a cart's episode is truncated (<c>MakeVec</c> passes the spec's, 500 for CartPole-v1).</param>
        /// <param name="renderMode">Gymnasium's <c>render_mode</c>: <see langword="null"/>, or a mode under which <see cref="Render"/> draws every cart (<c>rgb_array</c>).</param>
        /// <param name="suttonBartoReward">Gymnasium's <c>sutton_barto_reward</c>: reward -1 at termination and 0 otherwise, instead of 1 per step.</param>
        public CartPoleVectorEnv(int numEnvs = 1, long maxEpisodeSteps = 500, string? renderMode = null, bool suttonBartoReward = false) {
            SuttonBartoReward = suttonBartoReward;
            NumEnvs = numEnvs;
            MaxEpisodeSteps = maxEpisodeSteps;
            RenderMode = renderMode;
            Metadata = ClassMetadata;

            _steps = new int[numEnvs];
            _prevDone = new bool[numEnvs];

            // Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds.
            float[] high = { (float)(XThreshold * 2), float.PositiveInfinity, (float)(ThetaThresholdRadians * 2), float.PositiveInfinity };
            SingleActionSpace = new Discrete(2);
            ActionSpace = (Space<NDArray>)VectorUtils.BatchSpace(SingleActionSpace, numEnvs);
            using (NDArray highArray = np.array(high))
            using (NDArray lowArray = -highArray)
                SingleObservationSpace = new Box(lowArray, highArray, dtype: np.float32);
            ObservationSpace = (Space<NDArray>)VectorUtils.BatchSpace(SingleObservationSpace, numEnvs);
        }

        /// <summary>
        ///     Gymnasium's class attribute <c>metadata</c>: <c>{"render_modes": ["rgb_array"], "render_fps": 50,
        ///     "autoreset_mode": AutoresetMode.NEXT_STEP}</c>. A new dict on every read.
        /// </summary>
        public static Metadata ClassMetadata => new() {
            [Metadata.RenderModesKey] = new List<string> { "rgb_array" },
            [Metadata.RenderFpsKey] = 50L,
            ["autoreset_mode"] = AutoresetMode.NextStep,
        };

        /// <summary>Gymnasium's <c>_sutton_barto_reward</c>.</summary>
        [PyName("_sutton_barto_reward")]
        public bool SuttonBartoReward { get; }

        /// <summary>Gymnasium's <c>max_episode_steps</c>: a cart's steps before truncation.</summary>
        public long MaxEpisodeSteps { get; set; }

        /// <summary>Gymnasium's <c>gravity</c>, 9.8.</summary>
        public double Gravity { get; set; } = 9.8;

        /// <summary>Gymnasium's <c>masscart</c>, 1.0.</summary>
        public double Masscart { get; set; } = 1.0;

        /// <summary>Gymnasium's <c>masspole</c>, 0.1.</summary>
        public double Masspole { get; set; } = 0.1;

        /// <summary>Gymnasium's <c>total_mass</c>: <c>masspole + masscart</c>, computed once at construction as in Gymnasium.</summary>
        public double TotalMass { get; set; } = 0.1 + 1.0;

        /// <summary>Gymnasium's <c>length</c>, 0.5: actually half the pole's length.</summary>
        public double Length { get; set; } = 0.5;

        /// <summary>Gymnasium's <c>polemass_length</c>: <c>masspole * length</c>, computed once at construction.</summary>
        public double PolemassLength { get; set; } = 0.1 * 0.5;

        /// <summary>Gymnasium's <c>force_mag</c>, 10.0.</summary>
        public double ForceMag { get; set; } = 10.0;

        /// <summary>Gymnasium's <c>tau</c>: seconds between state updates, 0.02.</summary>
        public double Tau { get; set; } = 0.02;

        /// <summary>Gymnasium's <c>kinematics_integrator</c>: <c>"euler"</c>, or anything else for semi-implicit Euler.</summary>
        public string KinematicsIntegrator { get; set; } = "euler";

        /// <summary>Gymnasium's <c>theta_threshold_radians</c>: <c>12 * 2 * math.pi / 360</c>, the angle that ends an episode.</summary>
        public double ThetaThresholdRadians { get; set; } = 12 * 2 * Math.PI / 360;

        /// <summary>Gymnasium's <c>x_threshold</c>, 2.4: the cart position that ends an episode.</summary>
        public double XThreshold { get; set; } = 2.4;

        /// <summary>Gymnasium's <c>low</c>: the lower bound of the state draws, -0.05 or the last reset's <c>options["low"]</c>.</summary>
        public double Low { get; private set; } = -0.05;

        /// <summary>Gymnasium's <c>high</c>: the upper bound of the state draws.</summary>
        public double High { get; private set; } = 0.05;

        /// <summary>Gymnasium's <c>screen_width</c>, 600.</summary>
        public int ScreenWidth { get; set; } = 600;

        /// <summary>Gymnasium's <c>screen_height</c>, 400.</summary>
        public int ScreenHeight { get; set; } = 400;

        /// <summary>Gymnasium's <c>screens</c>: one surface per cart, created by the first render.</summary>
        public IReadOnlyList<Surface>? Screens { get; private set; }

        /// <summary>Gymnasium's <c>surf</c>: the last cart drawn, before it is blitted onto its screen.</summary>
        public Surface? Surf { get; private set; }

        /// <summary>Gymnasium's <c>steps_beyond_terminated</c>: always <see langword="null"/> (Python's <c>None</c>) in the vector env.</summary>
        public long? StepsBeyondTerminated { get; private set; }

        /// <summary>
        ///     Gymnasium's <c>state</c>: a new <c>(4, num_envs)</c> float64 array of the carts' state (the caller owns it), or
        ///     <see langword="null"/> before the first reset. Assigning an array of that shape injects a state.
        /// </summary>
        /// <exception cref="ValueError">Thrown by the setter for an array of another shape.</exception>
        public NDArray? State {
            get => _state is null ? null : np.array(_state);
            set {
                if (value is null) {
                    _state = null;
                    return;
                }

                if (value.ndim != 2 || value.shape[0] != 4 || value.shape[1] != NumEnvs)
                    throw new ValueError($"CartPoleVectorEnv's state has shape (4, {NumEnvs}), not {PyBuiltins.Repr(value.shape.Select(d => (object?)d).ToArray())}");
                var state = new double[4, NumEnvs];
                using NDArray float64 = value.astype(np.float64);
                for (int row = 0; row < 4; row++)
                    for (int env = 0; env < NumEnvs; env++)
                        state[row, env] = float64.GetDouble(row, env);
                _state = state;
            }
        }

        /// <summary>Gymnasium's <c>steps</c>: a new int32 array of each cart's steps in its episode.</summary>
        public NDArray Steps => np.array(_steps);

        /// <summary>Gymnasium's <c>prev_done</c>: a new bool array of the carts re-drawn at the next step.</summary>
        public NDArray PrevDone => np.array(_prevDone);

        /// <summary>
        ///     Draws every cart's state (one <c>uniform(low, high, size=(4, num_envs))</c> draw) and clears the step counters.
        /// </summary>
        /// <param name="seed">The seed of a fresh generator, or <see langword="null"/> to keep the stream.</param>
        /// <param name="options">Optional <c>low</c>/<c>high</c> bounds of the draws (Gymnasium's <c>maybe_parse_reset_bounds</c>).</param>
        /// <returns>The <c>(num_envs, 4)</c> float32 observations (a new array the caller owns) and an empty info.</returns>
        /// <exception cref="Error">Thrown for a negative seed.</exception>
        /// <exception cref="ValueError">Thrown for bounds <c>float()</c> rejects or a lower bound above the upper.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/cartpole.py#L496">gymnasium/envs/classic_control/cartpole.py</see>.</remarks>
        public override VectorResetResult<NDArray> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            SeedNpRandom(seed);
            // Note that if you use custom reset bounds, it may lead to out-of-bound state/observations.
            (Low, High) = ClassicControlUtils.MaybeParseResetBounds(options, -0.05, 0.05);
            var state = new double[4, NumEnvs];
            using (NDArray drawn = NpRandom.uniform(Low, High, new Shape(4, NumEnvs))) {
                for (int row = 0; row < 4; row++)
                    for (int env = 0; env < NumEnvs; env++)
                        state[row, env] = drawn.GetDouble(row, env);
            }

            _state = state;
            StepsBeyondTerminated = null;
            _steps = new int[NumEnvs];
            _prevDone = new bool[NumEnvs];
            return new VectorResetResult<NDArray>(Observations(), new Info());
        }

        /// <summary>
        ///     Steps every cart with its push (0 left, 1 right), then re-draws the carts that ended at the previous step (one
        ///     <c>uniform(low, high, size=(4, k))</c> draw), zeroing their counters, rewards and flags.
        /// </summary>
        /// <param name="actions">A <c>(num_envs,)</c> integer array the <c>MultiDiscrete</c> action space contains.</param>
        /// <returns>
        ///     The <c>(num_envs, 4)</c> float32 observations, the float64 rewards, the terminations and truncations, and an
        ///     empty info; new arrays the caller owns.
        /// </returns>
        /// <exception cref="AssertionError">
        ///     Thrown for actions the space does not contain (<c>array([0, 2]) (&lt;class 'numpy.ndarray'&gt;) invalid</c>), or
        ///     before the first reset (<c>Call reset before using step method.</c>).
        /// </exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/cartpole.py#L424">gymnasium/envs/classic_control/cartpole.py</see>.</remarks>
        public override VectorStepResult<NDArray> Step(NDArray actions) {
            if (!ActionSpace.Contains(actions))
                throw new AssertionError($"{PyBuiltins.Repr(actions)} ({PyBuiltins.TypeRepr(actions)}) invalid");
            double[,] state = _state ?? throw new AssertionError("Call reset before using step method.");

            int n = NumEnvs;
            var terminated = new bool[n];
            var truncated = new bool[n];
            var reward = new double[n];
            for (int i = 0; i < n; i++) {
                double x = state[0, i], xDot = state[1, i], theta = state[2, i], thetaDot = state[3, i];
                // force = np.sign(action - 0.5) * self.force_mag
                double force = Math.Sign(PyBuiltins.Float(ActionAt(actions, i)) - 0.5) * ForceMag;
                double costheta = Math.Cos(theta);
                double sintheta = Math.Sin(theta);

                // For the interested reader: https://coneural.org/florian/papers/05_cart_pole.pdf
                double temp = (force + PolemassLength * (thetaDot * thetaDot) * sintheta) / TotalMass;
                double thetaacc = (Gravity * sintheta - costheta * temp) / (Length * (4.0 / 3.0 - Masspole * (costheta * costheta) / TotalMass));
                double xacc = temp - PolemassLength * thetaacc * costheta / TotalMass;

                if (KinematicsIntegrator == "euler") {
                    x = x + Tau * xDot;
                    xDot = xDot + Tau * xacc;
                    theta = theta + Tau * thetaDot;
                    thetaDot = thetaDot + Tau * thetaacc;
                } else { // semi-implicit euler
                    xDot = xDot + Tau * xacc;
                    x = x + Tau * xDot;
                    thetaDot = thetaDot + Tau * thetaacc;
                    theta = theta + Tau * thetaDot;
                }

                state[0, i] = x;
                state[1, i] = xDot;
                state[2, i] = theta;
                state[3, i] = thetaDot;
                terminated[i] = x < -XThreshold || x > XThreshold || theta < -ThetaThresholdRadians || theta > ThetaThresholdRadians;
                _steps[i] += 1;
                truncated[i] = _steps[i] >= MaxEpisodeSteps;
                // -np.array(terminated, dtype=np.float32): a cart that did not terminate gets -0.0, its sign kept.
                reward[i] = SuttonBartoReward ? -(terminated[i] ? 1.0f : 0.0f) : 1.0f;
            }

            // Reset all environments which terminated or were truncated in the last step.
            int[] done = Enumerable.Range(0, n).Where(i => _prevDone[i]).ToArray();
            using (NDArray drawn = NpRandom.uniform(Low, High, new Shape(4, done.Length))) {
                for (int row = 0; row < 4; row++)
                    for (int j = 0; j < done.Length; j++)
                        state[row, done[j]] = drawn.GetDouble(row, j);
            }

            foreach (int i in done) {
                _steps[i] = 0;
                reward[i] = 0.0f;
                terminated[i] = false;
                truncated[i] = false;
            }

            for (int i = 0; i < n; i++)
                _prevDone[i] = terminated[i] || truncated[i];
            return new VectorStepResult<NDArray>(Observations(), np.array(reward), np.array(terminated), np.array(truncated), new Info());
        }

        /// <summary>
        ///     Draws every cart (a list of <c>(400, 600, 3)</c> uint8 frames, one per cart); without a render mode, warns as
        ///     Gymnasium does and returns nothing.
        /// </summary>
        /// <returns>One <see cref="RenderResult.Rgb"/> per cart (new arrays the caller owns), or <see langword="null"/> without a render mode.</returns>
        /// <exception cref="AssertionError">Thrown without a render mode when the env has no spec (<c>assert self.spec is not None</c>).</exception>
        /// <exception cref="ValueError">Thrown before the first reset (<c>Cartpole's state is None, it probably hasn't be reset yet.</c>), and for a NaN coordinate with pygame's message.</exception>
        /// <exception cref="OverflowException">Thrown, as pygame's <c>OverflowError</c>, for a coordinate pygame cannot draw.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/cartpole.py#L516">gymnasium/envs/classic_control/cartpole.py</see>.</remarks>
        public override IReadOnlyList<RenderResult>? Render() {
            if (RenderMode is null) {
                EnvSpec spec = Spec ?? throw new AssertionError();
                Logger.Warn("You are calling render method without specifying any render mode. You can specify the render_mode at initialization, "
                            + $"e.g. gym.make_vec(\"{spec.Id}\", render_mode=\"rgb_array\")");
                return null;
            }

            Screens ??= Enumerable.Range(0, NumEnvs).Select(_ => new Surface(ScreenWidth, ScreenHeight)).ToArray();

            double worldWidth = XThreshold * 2;
            double scale = ScreenWidth / worldWidth;
            const double polewidth = 10.0;
            double polelen = scale * (2 * Length);
            const double cartwidth = 50.0;
            const double cartheight = 30.0;

            double[,] state = _state ?? throw new ValueError("Cartpole's state is None, it probably hasn't be reset yet.");
            for (int i = 0; i < NumEnvs; i++)
                DrawCart(state[0, i], state[2, i], Screens[i], scale, polewidth, polelen, cartwidth, cartheight);

            return Screens.Select(screen => (RenderResult)new RenderResult.Rgb(SurfArray.ToRgbArray(screen))).ToArray();
        }

        /// <summary>Gymnasium's <c>close</c>: <c>pygame.quit()</c> once screens exist. There is no window to close here.</summary>
        protected override void CloseExtras() { }

        /// <summary>One cart's frame onto its screen: the cart, the pole, the axle and the track, flipped to y-up, as the single CartPole draws.</summary>
        /// <param name="x">The cart's position.</param>
        /// <param name="theta">The pole's angle.</param>
        /// <param name="screen">The cart's screen.</param>
        /// <param name="scale">Pixels per world unit.</param>
        /// <param name="polewidth">The pole's width in pixels.</param>
        /// <param name="polelen">The pole's length in pixels.</param>
        /// <param name="cartwidth">The cart's width in pixels.</param>
        /// <param name="cartheight">The cart's height in pixels.</param>
        /// <exception cref="ValueError">Thrown for a NaN coordinate, with pygame's message.</exception>
        /// <exception cref="OverflowException">Thrown for a coordinate pygame cannot draw.</exception>
        private void DrawCart(double x, double theta, Surface screen, double scale, double polewidth, double polelen, double cartwidth, double cartheight) {
            var surf = new Surface(ScreenWidth, ScreenHeight);
            Surf = surf;
            surf.Fill(new Color(255, 255, 255));

            double l = -cartwidth / 2, r = cartwidth / 2, t = cartheight / 2, b = -cartheight / 2;
            double axleoffset = cartheight / 4.0;
            double cartx = x * scale + ScreenWidth / 2.0; // MIDDLE OF CART
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
                Vector2 rotated = new Vector2(coord.X, coord.Y).RotateRad(-theta);
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
            screen.Blit(surf, 0, 0);
        }

        /// <summary><c>self.state.T.astype(np.float32)</c>: a new <c>(num_envs, 4)</c> float32 array, each value rounded to nearest-even.</summary>
        /// <returns>The observations.</returns>
        private NDArray Observations() {
            double[,] state = _state!;
            var observations = new float[NumEnvs, 4];
            for (int env = 0; env < NumEnvs; env++)
                for (int row = 0; row < 4; row++)
                    observations[env, row] = (float)state[row, env];
            return np.array(observations);
        }

        /// <summary>One cart's action as its value, for <c>action - 0.5</c>.</summary>
        /// <param name="actions">The actions.</param>
        /// <param name="i">The cart.</param>
        /// <returns>The action's value, boxed.</returns>
        private static object ActionAt(NDArray actions, int i) => actions.GetAtIndex(i);

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
    }
}
