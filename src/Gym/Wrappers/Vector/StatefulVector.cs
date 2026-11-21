#nullable enable
using System;
using System.Collections.Generic;
using System.Numerics;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Spaces;
using Gym.Utils;
using Gym.Vector;
using NumSharp;

namespace Gym.Wrappers.Vector {
    /// <summary>
    ///     Normalizes the batched observations by running statistics, as Gymnasium's vector <c>NormalizeObservation</c>
    ///     does: every batch updates the mean and variance (one update of <c>num_envs</c> rows), and the observations come
    ///     back as <c>float32((obs - mean) / sqrt(var + epsilon))</c>.
    /// </summary>
    /// <typeparam name="TAct">The batched action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     The statistics are float32 (the new single space's dtype); a batch of a wider dtype promotes them, as NumPy does.
    ///     Partial resets are refused (the statistics would mix episodes), and the env must autoreset <c>NEXT_STEP</c>.
    ///     Not thread-safe.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/vector/stateful_observation.py#L29">gymnasium/wrappers/vector/stateful_observation.py</see>.</para>
    /// </remarks>
    [PyName("NormalizeObservation")]
    public class NormalizeObservationVector<TAct> : VectorObservationWrapper<NDArray, NDArray, TAct>, IRecordConstructorArgs {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The vector env, whose single observation space has a shape.</param>
        /// <param name="epsilon">Added to the variance under the square root, strictly positive.</param>
        /// <exception cref="InvalidBound">Thrown for a non-positive <paramref name="epsilon"/>, with Gymnasium's message.</exception>
        /// <exception cref="ValueError">Thrown when the env's autoreset mode is not <c>NEXT_STEP</c>, with Gymnasium's message.</exception>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        public NormalizeObservationVector(VectorEnv<NDArray, TAct> env, double epsilon = 1e-8) : base(CheckEpsilon(env, epsilon)) {
            SavedKwargs = RecordConstructorArgs.Record(new Info { ["epsilon"] = epsilon });
            if (!env.Metadata.TryGetValue("autoreset_mode", out object? mode)) {
                Logger.Warn($"{this} is missing `autoreset_mode` data. Assuming that the vector environment it follows the `NextStep` autoreset api or autoreset is disabled. Read https://farama.org/Vector-Autoreset-Mode for more details.");
            } else if (mode is not AutoresetMode.NextStep) {
                // PyBuiltins.Str of a member is its Python str, AutoresetMode.SAME_STEP, as Gymnasium's f-string shows it.
                throw new ValueError($"Expected env.metadata['autoreset_mode'] to be AutoresetMode.NEXT_STEP, got {PyBuiltins.Str(mode)}");
            }

            Shape shape = ObjectSpace.Unview(env.SingleObservationSpace)!.Shape ?? throw new ValueError("NormalizeObservation requires the observation space to have a shape.");
            var single = new Box(double.NegativeInfinity, double.PositiveInfinity, shape, np.float32);
            SingleObservationSpace = single;
            ObservationSpace = (Space<NDArray>)VectorUtils.BatchSpace(single, NumEnvs);
            // RunningMeanStd(shape=self.single_observation_space.shape, dtype=self.single_observation_space.dtype): the new
            // float32 space's, read after it replaced the env's.
            ObsRms = new RunningMeanStd(shape: shape, dtype: np.float32);
            Epsilon = epsilon;
        }

        /// <summary>The recorded constructor arguments, <c>{'epsilon': 1e-08}</c> (Python's <c>_saved_kwargs</c>).</summary>
        [PyName("_saved_kwargs")]
        public Info SavedKwargs { get; }

        /// <summary>The running statistics of the observations (<c>obs_rms</c>); owned by this wrapper.</summary>
        public RunningMeanStd ObsRms { get; }

        /// <summary>The variance's offset under the square root (<c>epsilon</c>).</summary>
        public double Epsilon { get; set; }

        /// <summary>Whether observations update the statistics (<c>update_running_mean</c>); turn off to evaluate with frozen statistics.</summary>
        public bool UpdateRunningMean { get; set; } = true;

        /// <summary>Resets the env; a partial reset (a <c>reset_mask</c> not all true) is refused.</summary>
        /// <param name="seed">The seed, passed on.</param>
        /// <param name="options">The options, passed on.</param>
        /// <returns>The normalized first observations and the env's infos.</returns>
        /// <exception cref="ValueError">Thrown for a partial <c>reset_mask</c>, with Gymnasium's message.</exception>
        public override VectorResetResult<NDArray> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            CheckFullReset(options);
            return base.Reset(seed, options);
        }

        /// <summary>Resets the env with one seed per sub-env; a partial reset is refused.</summary>
        /// <param name="seeds">The seeds, passed on.</param>
        /// <param name="options">The options, passed on.</param>
        /// <returns>The normalized first observations and the env's infos.</returns>
        /// <exception cref="ValueError">Thrown for a partial <c>reset_mask</c>, with Gymnasium's message.</exception>
        public override VectorResetResult<NDArray> Reset(IReadOnlyList<BigInteger?> seeds, IReadOnlyDictionary<string, object?>? options = null) {
            CheckFullReset(options);
            return base.Reset(seeds, options);
        }

        /// <summary>Updates the statistics with the batch, then returns <c>float32((obs - mean) / sqrt(var + epsilon))</c>.</summary>
        /// <param name="observations">The env's batched observations; left to the env's caller.</param>
        /// <returns>A new float32 batch.</returns>
        public override NDArray Observations(NDArray observations) {
            if (UpdateRunningMean)
                ObsRms.Update(observations);
            using NDArray difference = observations - ObsRms.Mean;
            using NDArray shifted = ObsRms.Var + Epsilon;
            using NDArray root = np.sqrt(shifted);
            using NDArray quotient = difference / root;
            return quotient.astype(np.float32);
        }

        /// <summary>Disposes the statistics with the stack.</summary>
        /// <param name="disposing"><see langword="true"/> from <see cref="IDisposable.Dispose"/>.</param>
        protected override void Dispose(bool disposing) {
            base.Dispose(disposing);
            if (disposing)
                ObsRms.Dispose();
        }

        /// <summary>Refuses a partial reset: the statistics normalize every sub-env alike.</summary>
        /// <param name="options">The reset options.</param>
        /// <exception cref="ValueError">Thrown for a <c>reset_mask</c> not all true.</exception>
        private static void CheckFullReset(IReadOnlyDictionary<string, object?>? options) {
            if (options is not null && options.TryGetValue("reset_mask", out object? mask) && mask is NDArray array && !np.all(array))
                throw new ValueError("NormalizeObservation does not support partial resets. The 'reset_mask' must contain all True values.");
        }

        /// <summary>Gymnasium's check of <paramref name="epsilon"/>, made before anything is built.</summary>
        /// <param name="env">The env, passed through.</param>
        /// <param name="epsilon">The offset.</param>
        /// <returns><paramref name="env"/>.</returns>
        /// <exception cref="InvalidBound">Thrown for a non-positive <paramref name="epsilon"/>.</exception>
        private static VectorEnv<NDArray, TAct> CheckEpsilon(VectorEnv<NDArray, TAct> env, double epsilon) {
            // `epsilon` sits under the square root to keep the division away from zero; a non-positive value defeats it.
            if (epsilon <= 0)
                throw new InvalidBound($"`epsilon` should be strictly positive. Received {PyBuiltins.Str(epsilon)}");
            return env;
        }
    }

    /// <summary>
    ///     Scales the batched rewards by the running standard deviation of each sub-env's discounted return, as Gymnasium's
    ///     vector <c>NormalizeReward</c> does: <c>reward / sqrt(var + epsilon)</c>, where the returns accumulate per sub-env
    ///     with discount <c>gamma</c>.
    /// </summary>
    /// <typeparam name="TObs">The batched observation type, passed through.</typeparam>
    /// <typeparam name="TAct">The batched action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     The accumulated returns are float32 (Gymnasium's array), so each update is computed as NumPy computes it (the
    ///     discounted return in float64 through the int <c>1 - terminated</c>, stored back as float32). With <c>NEXT_STEP</c>
    ///     the autoreset step after an episode's end is skipped; with <c>SAME_STEP</c> every step counts and an ended return
    ///     restarts at once. Not thread-safe.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/vector/stateful_reward.py#L21">gymnasium/wrappers/vector/stateful_reward.py</see>.</para>
    /// </remarks>
    [PyName("NormalizeReward")]
    public class NormalizeRewardVector<TObs, TAct> : VectorWrapper<TObs, TAct>, IRecordConstructorArgs {
        /// <summary>The env's autoreset mode (<c>_autoreset_mode</c>).</summary>
        private readonly AutoresetMode _autoresetMode;

        /// <summary>Which sub-envs ended at the last step, as float32 0 or 1 (<c>_prev_dones</c>).</summary>
        private NDArray _prevDones;

        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The vector env.</param>
        /// <param name="gamma">The discount, in [0, 1].</param>
        /// <param name="epsilon">Added to the variance under the square root, strictly positive.</param>
        /// <exception cref="InvalidBound">Thrown for a gamma outside [0, 1] or a non-positive epsilon, with Gymnasium's messages.</exception>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        public NormalizeRewardVector(VectorEnv<TObs, TAct> env, double gamma = 0.99, double epsilon = 1e-8) : base(CheckArguments(env, gamma, epsilon)) {
            SavedKwargs = RecordConstructorArgs.Record(new Info { ["gamma"] = gamma, ["epsilon"] = epsilon });
            ReturnRms = new RunningMeanStd(shape: new Shape());
            AccumulatedReward = np.zeros(new Shape(NumEnvs), np.float32);
            Gamma = gamma;
            Epsilon = epsilon;
            _prevDones = np.zeros(new Shape(NumEnvs), np.float32);
            _autoresetMode = env.Metadata.TryGetValue("autoreset_mode", out object? mode) && mode is AutoresetMode known ? known : AutoresetMode.NextStep;
        }

        /// <summary>The recorded constructor arguments (Python's <c>_saved_kwargs</c>).</summary>
        [PyName("_saved_kwargs")]
        public Info SavedKwargs { get; }

        /// <summary>The running statistics of the discounted returns (<c>return_rms</c>); owned by this wrapper.</summary>
        public RunningMeanStd ReturnRms { get; }

        /// <summary>Each sub-env's discounted return, float32 (<c>accumulated_reward</c>); owned by this wrapper.</summary>
        public NDArray AccumulatedReward { get; }

        /// <summary>The discount (<c>gamma</c>).</summary>
        public double Gamma { get; set; }

        /// <summary>The variance's offset under the square root (<c>epsilon</c>).</summary>
        public double Epsilon { get; set; }

        /// <summary>Whether returns update the statistics (<c>update_running_mean</c>).</summary>
        public bool UpdateRunningMean { get; set; } = true;

        /// <summary>Clears the returns and the done flags, then resets the env.</summary>
        /// <param name="seed">The seed, passed on.</param>
        /// <param name="options">The options, passed on.</param>
        /// <returns>The env's reset.</returns>
        public override VectorResetResult<TObs> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            ClearState();
            return base.Reset(seed, options);
        }

        /// <summary>Clears the returns and the done flags, then resets the env with one seed per sub-env.</summary>
        /// <param name="seeds">The seeds, passed on.</param>
        /// <param name="options">The options, passed on.</param>
        /// <returns>The env's reset.</returns>
        public override VectorResetResult<TObs> Reset(IReadOnlyList<BigInteger?> seeds, IReadOnlyDictionary<string, object?>? options = null) {
            ClearState();
            return base.Reset(seeds, options);
        }

        /// <summary>Steps the env, accumulates the returns of the active sub-envs, updates the statistics, and scales the rewards.</summary>
        /// <param name="actions">The actions, passed on.</param>
        /// <returns>The transition with the rewards divided by <c>sqrt(var + epsilon)</c> (a new float64 array).</returns>
        public override VectorStepResult<TObs> Step(TAct actions) {
            VectorStepResult<TObs> result = base.Step(actions);
            // SAME_STEP resets inside the terminating step, so every step counts; NEXT_STEP and DISABLED skip the step after a done.
            using NDArray previous = _prevDones.astype(np.@bool);
            using NDArray active = _autoresetMode == AutoresetMode.SameStep
                ? np.ones(new Shape(NumEnvs), np.@bool)
                : np.logical_not(previous);
            using (NDArray accumulated = AccumulatedReward[active])
            using (NDArray scaled = accumulated * Gamma)
            using (NDArray terminated = result.Terminations[active])
            using (NDArray terminatedInts = terminated.astype(np.int64))
            using (NDArray continuing = 1 - terminatedInts)
            using (NDArray discounted = scaled * continuing)
            using (NDArray rewards = result.Rewards[active])
            using (NDArray updated = discounted + rewards)
                AccumulatedReward[active] = updated;

            if (UpdateRunningMean && np.any(active)) {
                using NDArray batch = AccumulatedReward[active];
                ReturnRms.Update(batch);
            }

            using (NDArray ended = np.logical_or(result.Terminations, result.Truncations)) {
                _prevDones.Dispose();
                _prevDones = ended.astype(np.float32);
            }

            if (_autoresetMode == AutoresetMode.SameStep) {
                // The done sub-envs have already been reset: their returns restart at once.
                using NDArray done = _prevDones.astype(np.@bool);
                AccumulatedReward[done] = 0;
            }

            using NDArray variance = ReturnRms.Var + Epsilon;
            using NDArray root = np.sqrt(variance);
            NDArray normalized = result.Rewards / root;
            return new VectorStepResult<TObs>(result.Observations, normalized, result.Terminations, result.Truncations, result.Infos);
        }

        /// <summary>Releases the statistics and the per-sub-env arrays, then disposes as every vector wrapper does.</summary>
        /// <param name="disposing"><see langword="true"/> from <see cref="IDisposable.Dispose"/>.</param>
        protected override void Dispose(bool disposing) {
            base.Dispose(disposing);
            if (!disposing)
                return;
            ReturnRms.Dispose();
            AccumulatedReward.Dispose();
            _prevDones.Dispose();
        }

        /// <summary><c>accumulated_reward[:] = 0</c> and <c>_prev_dones[:] = 0</c>.</summary>
        private void ClearState() {
            AccumulatedReward[":"] = 0;
            _prevDones[":"] = 0;
        }

        /// <summary>Gymnasium's checks of the arguments, made before anything is built.</summary>
        /// <param name="env">The env, passed through.</param>
        /// <param name="gamma">The discount.</param>
        /// <param name="epsilon">The offset.</param>
        /// <returns><paramref name="env"/>.</returns>
        /// <exception cref="InvalidBound">Thrown for a gamma outside [0, 1] (NaN included) or a non-positive epsilon.</exception>
        private static VectorEnv<TObs, TAct> CheckArguments(VectorEnv<TObs, TAct> env, double gamma, double epsilon) {
            // The accumulator is multiplied by gamma every step, so a value outside [0, 1] diverges or alternates.
            if (!(0 <= gamma && gamma <= 1))
                throw new InvalidBound($"`gamma` should be in the interval [0, 1]. Received {PyBuiltins.Str(gamma)}");
            if (epsilon <= 0)
                throw new InvalidBound($"`epsilon` should be strictly positive. Received {PyBuiltins.Str(epsilon)}");
            return env;
        }
    }
}
