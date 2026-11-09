#nullable enable
using System;
using Gym.Envs;
using Gym.Observations;
using NumSharp;

namespace Gym.Vector {
    /// <summary>
    ///     One step of every sub-env at once, <c>(observations, rewards, terminations, truncations, infos)</c>, in Gymnasium's
    ///     vector layout. Deconstructs like a tuple: <c>var (obs, rewards, terminations, truncations, infos) = envs.Step(actions);</c>.
    /// </summary>
    /// <typeparam name="TObs">The batched observation type, an element of the vector env's batched observation space.</typeparam>
    /// <param name="Observations">The sub-envs' observations batched along the first axis (an array for array spaces); owned by the caller.</param>
    /// <param name="Rewards">A <c>(num_envs,)</c> float64 array; owned by the caller.</param>
    /// <param name="Terminations">A <c>(num_envs,)</c> bool array: which sub-envs reached a terminal state; owned by the caller.</param>
    /// <param name="Truncations">A <c>(num_envs,)</c> bool array: which sub-envs were cut short (a time limit); owned by the caller.</param>
    /// <param name="Infos">
    ///     The sub-envs' infos merged by key: each key holds a batch of the sub-envs' values (a <c>(num_envs, ...)</c> array, or
    ///     an <see langword="object"/>?[] for values that are not numbers or arrays), and its mask <c>_key</c> says which
    ///     sub-envs reported it.
    /// </param>
    /// <remarks>
    ///     Equality is exact: arrays by shape, dtype and elements; the infos by Python's <c>==</c>. Reference:
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/vector/vector_env.py#L174">gymnasium/vector/vector_env.py (VectorEnv.step)</see>.
    /// </remarks>
    public readonly record struct VectorStepResult<TObs>(TObs Observations, NDArray Rewards, NDArray Terminations, NDArray Truncations, Info Infos) : INDArrayCarrier {
        /// <summary>Hands every array of the result to an <see cref="NDScope"/> (D19), the infos' included, so a scoped method can return it.</summary>
        /// <param name="scope">The scope the arrays leave through.</param>
        void INDArrayCarrier.YieldTo(NDScope scope) => ResultEquality.Yield(scope, Observations, Rewards, Terminations, Truncations, Infos);

        /// <summary>Exact equality of every field (see the remarks), replacing the compiler's member-wise one, which would broadcast arrays.</summary>
        /// <param name="other">The other result.</param>
        /// <returns><see langword="true"/> when the results are equal.</returns>
        public bool Equals(VectorStepResult<TObs> other) {
            return ResultEquality.ObservationsEqual(Observations, other.Observations) && ResultEquality.ObservationsEqual(Rewards, other.Rewards)
                   && ResultEquality.ObservationsEqual(Terminations, other.Terminations) && ResultEquality.ObservationsEqual(Truncations, other.Truncations)
                   && ResultEquality.InfosEqual(Infos, other.Infos);
        }

        /// <summary>A hash consistent with <see cref="Equals(VectorStepResult{TObs})"/>: shapes and counts only, as arrays have no value hash.</summary>
        /// <returns>The hash.</returns>
        public override int GetHashCode() => HashCode.Combine(ResultEquality.ObservationHash(Observations), ResultEquality.ObservationHash(Rewards), Infos?.Count ?? 0);
    }

    /// <summary>
    ///     The first observations of every reset sub-env and their merged infos, <c>(observations, infos)</c>. Deconstructs
    ///     like a tuple: <c>var (obs, infos) = envs.Reset(seed: 42);</c>.
    /// </summary>
    /// <typeparam name="TObs">The batched observation type.</typeparam>
    /// <param name="Observations">The sub-envs' observations batched along the first axis; owned by the caller.</param>
    /// <param name="Infos">The sub-envs' reset infos merged by key, each with its <c>_key</c> mask, as a step's.</param>
    /// <remarks>
    ///     Equality is exact, as <see cref="VectorStepResult{TObs}"/>'s. Reference:
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/vector/vector_env.py#L144">gymnasium/vector/vector_env.py (VectorEnv.reset)</see>.
    /// </remarks>
    public readonly record struct VectorResetResult<TObs>(TObs Observations, Info Infos) : INDArrayCarrier {
        /// <summary>Hands the observations' and the infos' arrays to an <see cref="NDScope"/> (D19).</summary>
        /// <param name="scope">The scope the arrays leave through.</param>
        void INDArrayCarrier.YieldTo(NDScope scope) => ResultEquality.Yield(scope, Observations, Infos);

        /// <summary>Exact equality of the observations and Python equality of the infos.</summary>
        /// <param name="other">The other result.</param>
        /// <returns><see langword="true"/> when both are equal.</returns>
        public bool Equals(VectorResetResult<TObs> other) => ResultEquality.ObservationsEqual(Observations, other.Observations) && ResultEquality.InfosEqual(Infos, other.Infos);

        /// <summary>A hash consistent with <see cref="Equals(VectorResetResult{TObs})"/>.</summary>
        /// <returns>The hash.</returns>
        public override int GetHashCode() => HashCode.Combine(ResultEquality.ObservationHash(Observations), Infos?.Count ?? 0);
    }
}
