using System;
using System.Globalization;
using Gym.Tests.Parity.Frames;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Pair;
using NumSharp;

namespace Gym.Tests.Parity.Reference {
    /// <summary>
    ///     A test-local, float64 transcription of Gymnasium's CartPole: the harness's positive control. It must match live Gymnasium bit for bit, which proves the harness can
    ///     report parity at all; its deliberately broken variants must be reported at the exact call they break, which proves
    ///     it can report divergence (plan §11).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Not product code, and not a stand-in for Gym.NET's <c>CartPoleEnv</c>: it exists so the harness is tested against something whose
    ///     correctness is known independently of Gym.NET. It is the same transcription the port plan's A.7 (and the parity
    ///     audit) found bit-identical offline, now checked live.
    ///     </para>
    ///     <para>
    ///     Faithful where it matters for bits: float64 constants (Python floats), <see cref="Math.Cos"/>/<see cref="Math.Sin"/>
    ///     for <c>np.cos</c>/<c>np.sin</c> on float64 scalars (bit-identical on the reference platform, A.7), <c>x * x</c> for
    ///     <c>np.square</c>, the source's evaluation order, and <c>Generator(PCG64(SeedSequence(seed)))</c> with
    ///     <c>uniform(-0.05, 0.05, size=4)</c> for the reset, as <c>seeding.np_random</c> and <c>reset</c> do.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/envs/classic_control/cartpole.py#L20">gymnasium/envs/classic_control/cartpole.py</see>.</para>
    /// </remarks>
    public sealed class CartPoleReference : IDotNetEnv {
        // CartPoleEnv.__init__ (cartpole.py, lines 124-136 at the pinned commit): instance attributes, all Python floats.
        /// <summary><c>self.gravity</c>.</summary>
        private const double Gravity = 9.8;

        /// <summary><c>self.masscart</c>.</summary>
        private const double MassCart = 1.0;

        /// <summary><c>self.masspole</c>.</summary>
        private const double MassPole = 0.1;

        /// <summary><c>self.total_mass = self.masspole + self.masscart</c>; the compiler folds the same IEEE addition Python performs.</summary>
        private const double TotalMass = MassPole + MassCart;

        /// <summary><c>self.length</c>, half the pole's length.</summary>
        private const double Length = 0.5;

        /// <summary><c>self.polemass_length = self.masspole * self.length</c>.</summary>
        private const double PoleMassLength = MassPole * Length;

        /// <summary><c>self.force_mag</c>.</summary>
        private const double ForceMag = 10.0;

        /// <summary><c>self.theta_threshold_radians = 12 * 2 * math.pi / 360</c>: an int product, then float operations, as in C#.</summary>
        private const double ThetaThresholdRadians = 12 * 2 * Math.PI / 360;

        /// <summary><c>self.x_threshold</c>.</summary>
        private const double XThreshold = 2.4;

        /// <summary><c>self.tau</c>: 0.02, or a broken variant's value.</summary>
        private readonly double _tau;

        /// <summary>Whether the broken variant draws one extra <c>random()</c> after the reset.</summary>
        private readonly bool _extraDrawAfterReset;

        /// <summary><c>self.np_random</c>, created by the reset.</summary>
        private Generator? _npRandom;

        /// <summary><c>self.state</c>, float64; <see langword="null"/> before the first reset.</summary>
        private double[]? _state;

        /// <summary><c>self.steps_beyond_terminated</c>.</summary>
        private int? _stepsBeyondTerminated;

        /// <summary>Creates the transcription or one of its broken variants.</summary>
        /// <param name="tau">The integration step; Gymnasium's is 0.02.</param>
        /// <param name="extraDrawAfterReset">Whether to draw one extra value after the reset's <c>uniform</c>.</param>
        /// <param name="variant">What was broken, appended to <see cref="Name"/>; <see langword="null"/> for the faithful transcription.</param>
        private CartPoleReference(double tau, bool extraDrawAfterReset, string? variant) {
            _tau = tau;
            _extraDrawAfterReset = extraDrawAfterReset;
            Name = "CartPoleReference (float64 transcription of cartpole.py @ 9e04324)" + (variant is null ? "" : $" broken: {variant}");
        }

        /// <summary>The faithful transcription, which must match Gymnasium bit for bit.</summary>
        /// <returns>A new env.</returns>
        public static CartPoleReference Faithful() => new(0.02, false, null);

        /// <summary>
        ///     The float32-constant defect of today's Gym.NET CartPole (B-11), isolated: <c>tau = 0.02f</c>, which as a double is
        ///     0.019999999552965164. Every other value is faithful, so the state must diverge at step 1 and nowhere earlier.
        /// </summary>
        /// <returns>A new env.</returns>
        public static CartPoleReference WithFloat32Tau() => new(0.02f, false, "tau = 0.02f");

        /// <summary>
        ///     One extra <c>random()</c> draw after the reset: the observation is untouched, so only the generator-state check can
        ///     see it, and it must see it at the reset itself, before any value differs.
        /// </summary>
        /// <returns>A new env.</returns>
        public static CartPoleReference WithExtraDraw() => new(0.02, true, "one extra random() after the reset");

        /// <inheritdoc />
        public string Name { get; }

        /// <inheritdoc />
        public string[] Adaptations => Array.Empty<string>();

        /// <inheritdoc />
        /// <remarks>
        ///     Gymnasium's <c>reset</c>: <c>super().reset(seed=seed)</c> sets <c>np_random</c> to
        ///     <c>seeding.np_random(seed)[0]</c>; then <c>self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))</c>,
        ///     and the observation is its float32 copy with an empty info. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/envs/classic_control/cartpole.py#L228">gymnasium/envs/classic_control/cartpole.py (CartPoleEnv.reset)</see>.
        /// </remarks>
        public CallResult Reset(long seed) {
            _npRandom = new Generator(new PCG64(new SeedSequence(seed)));
            using (NDArray drawn = _npRandom.uniform(-0.05, 0.05, new Shape(4)))
                _state = new[] { drawn.GetDouble(0), drawn.GetDouble(1), drawn.GetDouble(2), drawn.GetDouble(3) };
            if (_extraDrawAfterReset) {
                using NDArray extra = _npRandom.random();
            }

            _stepsBeyondTerminated = null;
            return CallResult.Of(new Transition(Observation(), null, null, null, Transition.EmptyInfo));
        }

        /// <inheritdoc />
        /// <remarks>
        ///     Gymnasium's <c>CartPoleEnv.step</c>, one statement per Python statement, in the source's order.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/envs/classic_control/cartpole.py#L164">gymnasium/envs/classic_control/cartpole.py (CartPoleEnv.step)</see>.
        /// </remarks>
        public CallResult Step(object action) {
            long a = Convert.ToInt64(action, CultureInfo.InvariantCulture);
            if (_state is null)
                throw new InvalidOperationException("Call reset before using step method.");

            double x = _state[0], xDot = _state[1], theta = _state[2], thetaDot = _state[3];
            double force = a == 1 ? ForceMag : -ForceMag;
            double costheta = Math.Cos(theta);
            double sintheta = Math.Sin(theta);
            // np.square(v) is v * v; the products associate left to right, as in the source.
            double temp = (force + PoleMassLength * (thetaDot * thetaDot) * sintheta) / TotalMass;
            double thetaacc = (Gravity * sintheta - costheta * temp) / (Length * (4.0 / 3.0 - MassPole * (costheta * costheta) / TotalMass));
            double xacc = temp - PoleMassLength * thetaacc * costheta / TotalMass;

            // kinematics_integrator == "euler"
            x = x + _tau * xDot;
            xDot = xDot + _tau * xacc;
            theta = theta + _tau * thetaDot;
            thetaDot = thetaDot + _tau * thetaacc;
            _state = new[] { x, xDot, theta, thetaDot };

            bool terminated = x < -XThreshold || x > XThreshold || theta < -ThetaThresholdRadians || theta > ThetaThresholdRadians;
            double reward;
            if (!terminated) {
                reward = 1.0;
            } else if (_stepsBeyondTerminated is null) {
                // Pole just fell!
                _stepsBeyondTerminated = 0;
                reward = 1.0;
            } else {
                // Gymnasium also warns once here (logger.warn); warnings are not part of the compared transition.
                _stepsBeyondTerminated += 1;
                reward = 0.0;
            }

            return CallResult.Of(new Transition(Observation(), ValueFacts.FromDouble(reward), ValueFacts.FromBool(terminated), ValueFacts.FromBool(false), Transition.EmptyInfo));
        }

        /// <inheritdoc />
        public bool TryGetRngState(out ValueFacts? state, out string reason) {
            if (_npRandom is null) {
                state = null;
                reason = "the reference has not been reset";
                return false;
            }

            state = RngFacts.Of(_npRandom);
            reason = "";
            return true;
        }

        /// <inheritdoc />
        public void SetAttribute(string name, double[] values) {
            if (name != "state")
                throw new NotSupportedException($"The CartPole reference has no attribute '{name}'.");
            _state = (double[])values.Clone();
        }

        /// <inheritdoc />
        public ValueFacts? GetAttribute(string name) {
            if (name != "state" || _state is null)
                return null;
            using NDArray state = np.array(_state);
            return ValueFacts.Of(state);
        }

        /// <inheritdoc />
        public Frame Render() => throw new NotSupportedException("The CartPole reference transcribes the dynamics only.");

        /// <inheritdoc />
        public void SeedActionSpace(long seed) => throw new NotSupportedException("The CartPole reference has no action space.");

        /// <inheritdoc />
        public object SampleAction() => throw new NotSupportedException("The CartPole reference has no action space.");

        /// <summary>Nothing to release: the reference holds managed state only.</summary>
        public void Dispose() { }

        /// <summary><c>np.array(self.state, dtype=np.float32)</c>, described.</summary>
        /// <returns>The observation.</returns>
        private ValueFacts Observation() {
            using NDArray state = np.array(_state!);
            using NDArray observation = state.astype(np.float32);
            return ValueFacts.Of(observation);
        }
    }
}
