#nullable enable
using System;
using System.Collections.Generic;
using System.Numerics;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Observations;
using Gym.Utils;
using NumSharp;

namespace Gym.Wrappers {
    /// <summary>
    ///     Scales rewards so their exponential moving average has roughly unit variance: every reward is divided by the square
    ///     root of the running variance (plus <see cref="Epsilon"/>) of a discounted return, as Gymnasium's
    ///     <c>NormalizeReward</c> does, byte for byte.
    /// </summary>
    /// <typeparam name="TObs">The observation type, passed through.</typeparam>
    /// <typeparam name="TAct">The action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     The statistics live in this wrapper and survive resets: the discounted return is multiplied by
    ///     <c>gamma * (1 - terminated)</c> before each reward is added (truncation does not clear it), as Gymnasium keeps it for
    ///     backward compatibility. The variance starts at 1, so the first rewards are scaled by about 1. Freeze the statistics
    ///     for evaluation with <see cref="UpdateRunningMean"/>. Not thread-safe.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/stateful_reward.py#L20">gymnasium/wrappers/stateful_reward.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class NormalizeReward<TObs, TAct> : Wrapper<TObs, TAct>, IRecordConstructorArgs {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env whose rewards to scale.</param>
        /// <param name="gamma">The discount of the tracked return, in [0, 1]: a value above 1 would make the average diverge.</param>
        /// <param name="epsilon">Added to the variance under the square root, strictly positive, so the division never reaches zero.</param>
        /// <exception cref="InvalidBound">
        ///     Thrown for a <paramref name="gamma"/> outside [0, 1] (<c>`gamma` should be in the interval [0, 1]. Received 1.5</c>)
        ///     or a non-positive <paramref name="epsilon"/>, before the env is looked at, as in Gymnasium.
        /// </exception>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        public NormalizeReward(Env<TObs, TAct> env, double gamma = 0.99, double epsilon = 1e-8) : base(CheckArguments(env, gamma, epsilon)) {
            SavedKwargs = RecordConstructorArgs.Record(new Info { ["gamma"] = gamma, ["epsilon"] = epsilon });
            ReturnRms = new RunningMeanStd(shape: new Shape());
            DiscountedReward = np.array(new[] { 0.0 });
            Gamma = gamma;
            Epsilon = epsilon;
        }

        /// <summary>The recorded constructor arguments, <c>{'gamma': 0.99, 'epsilon': 1e-08}</c> (Python's <c>_saved_kwargs</c>).</summary>
        [PyName("_saved_kwargs")]
        public Info SavedKwargs { get; }

        /// <summary>The running statistics of the discounted return (<c>return_rms</c>); owned by this wrapper.</summary>
        public RunningMeanStd ReturnRms { get; }

        /// <summary>The tracked discounted return, a <c>(1,)</c> float64 array (<c>discounted_reward</c>), replaced at every step.</summary>
        public NDArray DiscountedReward { get; set; }

        /// <summary>The discount of the tracked return (<c>gamma</c>).</summary>
        public double Gamma { get; set; }

        /// <summary>The variance's offset under the square root (<c>epsilon</c>).</summary>
        public double Epsilon { get; set; }

        /// <summary>Whether steps update the statistics (<c>update_running_mean</c>); turn off to evaluate with frozen statistics.</summary>
        public bool UpdateRunningMean { get; set; } = true;

        /// <summary>Steps the env, updates the discounted return and its statistics, and scales the reward.</summary>
        /// <param name="action">The action, passed on.</param>
        /// <returns>The env's transition with the reward divided by <c>sqrt(var + epsilon)</c>.</returns>
        public override StepResult<TObs> Step(TAct action) {
            StepResult<TObs> result = base.Step(action);

            // Using the `discounted_reward` rather than `reward` makes no sense but for backward compatibility, it is being kept.
            double discounted = DiscountedReward.GetDouble(0) * Gamma * (1 - (result.Terminated ? 1 : 0)) + result.Reward;
            NDArray replaced = DiscountedReward;
            DiscountedReward = np.array(new[] { discounted });
            replaced.Dispose();
            if (UpdateRunningMean)
                ReturnRms.Update(DiscountedReward);

            // We don't (reward - self.return_rms.mean) see https://github.com/openai/baselines/issues/538
            double normalizedReward = result.Reward / Math.Sqrt(ReturnRms.Var.GetDouble() + Epsilon);
            return result with { Reward = normalizedReward };
        }

        /// <summary>Disposes the statistics and the tracked return with the stack.</summary>
        /// <param name="disposing"><see langword="true"/> from <see cref="Env{TObs, TAct}.Dispose()"/>.</param>
        protected override void Dispose(bool disposing) {
            base.Dispose(disposing);
            if (!disposing)
                return;
            ReturnRms.Dispose();
            DiscountedReward.Dispose();
        }

        /// <summary>Gymnasium's checks of the arguments, made before the wrapper is built.</summary>
        /// <param name="env">The env, passed through.</param>
        /// <param name="gamma">The discount.</param>
        /// <param name="epsilon">The offset.</param>
        /// <returns><paramref name="env"/>.</returns>
        /// <exception cref="InvalidBound">Thrown for a <paramref name="gamma"/> outside [0, 1] or a non-positive <paramref name="epsilon"/>.</exception>
        private static Env<TObs, TAct> CheckArguments(Env<TObs, TAct> env, double gamma, double epsilon) {
            // The exponential moving average multiplies the accumulator by `gamma` on every step, so a value above one makes
            // it diverge instead of converging, and a negative one alternates its sign.
            if (!(0 <= gamma && gamma <= 1))
                throw new InvalidBound($"`gamma` should be in the interval [0, 1]. Received {PyBuiltins.Str(gamma)}");
            // `epsilon` is added under the square root to keep the division away from zero, so a non-positive value defeats
            // its purpose and, once it exceeds the variance, silently turns every normalized reward into NaN.
            if (epsilon <= 0)
                throw new InvalidBound($"`epsilon` should be strictly positive. Received {PyBuiltins.Str(epsilon)}");
            return env;
        }
    }
}
