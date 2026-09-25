#nullable enable
using System;
using System.Collections.Generic;
using Gym.Envs;
using Gym.Utils;
using NumSharp;

namespace Gym.Observations {
    /// <summary>
    ///     One environment transition, <c>(observation, reward, terminated, truncated, info)</c>. Deconstructs like a tuple:
    ///     <c>var (obs, reward, terminated, truncated, info) = env.Step(action);</c>.
    /// </summary>
    /// <typeparam name="TObs">The observation type, an element of the env's observation space.</typeparam>
    /// <param name="Observation">The observation after the action; owned by the caller (envs return a fresh array).</param>
    /// <param name="Reward">
    ///     The reward as float64, the precision of Gymnasium's Python floats; a float32 reward would change every
    ///     accumulated return.
    /// </param>
    /// <param name="Terminated">
    ///     The MDP reached a terminal state: the value of what follows is zero, so do not bootstrap past it. Call
    ///     <c>Reset</c> before stepping again.
    /// </param>
    /// <param name="Truncated">
    ///     Something outside the MDP ended the episode, typically a <c>TimeLimit</c>: bootstrapping from the final observation
    ///     stays valid. Call <c>Reset</c> before stepping again.
    /// </param>
    /// <param name="Info">Diagnostics with Python dict semantics (insertion order is observable); never <see langword="null"/> from Gym's own envs.</param>
    /// <remarks>
    ///     Equality is exact and never broadcasts: observations that are arrays must have the same shape, the same dtype and
    ///     identical elements (a NaN element equals nothing, not even itself); any other observation compares as Python's
    ///     <c>==</c> does (<see cref="PyEquality"/>), and so does the info. The reward compares with
    ///     <see cref="double.Equals(double)"/>, which, unlike Python's <c>==</c>, treats NaN as equal to NaN so equality stays
    ///     reflexive. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/core.py#L76">gymnasium/core.py (Env.step)</see>.
    /// </remarks>
    public readonly record struct StepResult<TObs>(TObs Observation, double Reward, bool Terminated, bool Truncated, Info Info) {
        /// <summary>
        ///     Exact equality (see the type's remarks): observations, rewards, flags and infos all equal. Replaces the
        ///     compiler's member-wise equality, which would call <see cref="NDArray"/>'s broadcasting <c>Equals</c>.
        /// </summary>
        /// <param name="other">The other transition.</param>
        /// <returns><see langword="true"/> when the transitions are equal.</returns>
        public bool Equals(StepResult<TObs> other) {
            return ResultEquality.ObservationsEqual(Observation, other.Observation) && Reward.Equals(other.Reward) && Terminated == other.Terminated
                   && Truncated == other.Truncated && ResultEquality.InfosEqual(Info, other.Info);
        }

        /// <summary>
        ///     A hash consistent with <see cref="Equals(StepResult{TObs})"/>: an array observation contributes its shape only,
        ///     because <see cref="NDArray.GetHashCode"/> throws (arrays are mutable, hence unhashable in NumPy).
        /// </summary>
        /// <returns>The hash.</returns>
        public override int GetHashCode() => HashCode.Combine(ResultEquality.ObservationHash(Observation), Reward, Terminated, Truncated, Info?.Count ?? 0);
    }

    /// <summary>
    ///     The first observation of an episode and its diagnostics, <c>(observation, info)</c>. Deconstructs like a tuple:
    ///     <c>var (obs, info) = env.Reset(seed: 42);</c>.
    /// </summary>
    /// <typeparam name="TObs">The observation type, an element of the env's observation space.</typeparam>
    /// <param name="Observation">The initial observation; owned by the caller.</param>
    /// <param name="Info">Diagnostics with Python dict semantics, analogous to a step's; never <see langword="null"/> from Gym's own envs.</param>
    /// <remarks>
    ///     Equality follows <see cref="StepResult{TObs}"/>'s: exact for array observations, Python's <c>==</c> otherwise.
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/core.py#L116">gymnasium/core.py (Env.reset)</see>.
    /// </remarks>
    public readonly record struct ResetResult<TObs>(TObs Observation, Info Info) {
        /// <summary>Exact equality of the observations and Python equality of the infos (see the remarks).</summary>
        /// <param name="other">The other result.</param>
        /// <returns><see langword="true"/> when both are equal.</returns>
        public bool Equals(ResetResult<TObs> other) => ResultEquality.ObservationsEqual(Observation, other.Observation) && ResultEquality.InfosEqual(Info, other.Info);

        /// <summary>A hash consistent with <see cref="Equals(ResetResult{TObs})"/> (an array observation contributes its shape only).</summary>
        /// <returns>The hash.</returns>
        public override int GetHashCode() => HashCode.Combine(ResultEquality.ObservationHash(Observation), Info?.Count ?? 0);
    }

    /// <summary>
    ///     A transition in the pre-0.26 "done" step API, <c>(observation, reward, done, info)</c>, which
    ///     <see cref="StepApiCompatibility"/> converts to and from <see cref="StepResult{TObs}"/>. <c>done</c> merges
    ///     termination and truncation; <c>info["TimeLimit.truncated"]</c> tells them apart.
    /// </summary>
    /// <typeparam name="TObs">The observation type.</typeparam>
    /// <param name="Observation">The observation after the action.</param>
    /// <param name="Reward">The reward as float64.</param>
    /// <param name="Done">Whether the episode ended, for either reason.</param>
    /// <param name="Info">Diagnostics; <c>"TimeLimit.truncated"</c> is <see langword="true"/> when a time limit, not the MDP, ended it.</param>
    /// <remarks>
    ///     Equality follows <see cref="StepResult{TObs}"/>'s. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/utils/step_api_compatibility.py#L11">gymnasium/utils/step_api_compatibility.py (DoneStepType)</see>.
    /// </remarks>
    public readonly record struct DoneStepResult<TObs>(TObs Observation, double Reward, bool Done, Info Info) {
        /// <summary>Exact equality (see <see cref="StepResult{TObs}"/>'s remarks).</summary>
        /// <param name="other">The other transition.</param>
        /// <returns><see langword="true"/> when the transitions are equal.</returns>
        public bool Equals(DoneStepResult<TObs> other) {
            return ResultEquality.ObservationsEqual(Observation, other.Observation) && Reward.Equals(other.Reward) && Done == other.Done && ResultEquality.InfosEqual(Info, other.Info);
        }

        /// <summary>A hash consistent with <see cref="Equals(DoneStepResult{TObs})"/>.</summary>
        /// <returns>The hash.</returns>
        public override int GetHashCode() => HashCode.Combine(ResultEquality.ObservationHash(Observation), Reward, Done, Info?.Count ?? 0);
    }

    /// <summary>The equality rules the result types share, so the three stay consistent.</summary>
    internal static class ResultEquality {
        /// <summary>
        ///     Two observations are equal when both are arrays of the same shape, dtype and elements, or, for anything else,
        ///     when Python's <c>==</c> says so.
        /// </summary>
        /// <typeparam name="TObs">The observation type.</typeparam>
        /// <param name="left">The first observation.</param>
        /// <param name="right">The second observation.</param>
        /// <returns><see langword="true"/> when they are equal.</returns>
        internal static bool ObservationsEqual<TObs>(TObs left, TObs right) {
            object? a = left, b = right;
            if (ReferenceEquals(a, b))
                return true;
            if (a is NDArray x && b is NDArray y)
                // Shape and dtype first: np.array_equal alone would equal float32 and float64 observations holding the same values.
                return x.Shape == y.Shape && x.dtype == y.dtype && np.array_equal(x, y);
            return PyEquality.AreEqual(a, b);
        }

        /// <summary>Python equality of two infos; two <see langword="null"/>s (a default result) are equal.</summary>
        /// <param name="left">The first info.</param>
        /// <param name="right">The second info.</param>
        /// <returns><see langword="true"/> when they are equal.</returns>
        internal static bool InfosEqual(Info? left, Info? right) => left is null ? right is null : right is not null && PyEquality.AreEqual(left, right);

        /// <summary>
        ///     A hash of an observation consistent with <see cref="ObservationsEqual{TObs}"/> that never reads array data: the
        ///     shape of an array, the numeric value of a number (so <c>1</c> and <c>1.0</c>, equal in Python, hash alike), the
        ///     text of a string, and a constant for anything else.
        /// </summary>
        /// <typeparam name="TObs">The observation type.</typeparam>
        /// <param name="observation">The observation.</param>
        /// <returns>The hash.</returns>
        /// <remarks>
        ///     Containers (dicts, lists, tuples) compare by content under Python's rules, which their .NET hashes do not follow,
        ///     so they all share one hash: collisions cost a hash-based collection speed, never correctness.
        /// </remarks>
        internal static int ObservationHash<TObs>(TObs observation) {
            object? value = observation;
            switch (value) {
                case null:
                    return 0;
                case NDArray array:
                    return array.Shape.GetHashCode();
                case string text:
                    return StringComparer.Ordinal.GetHashCode(text);
                case bool flag:
                    return (flag ? 1.0 : 0.0).GetHashCode();
                case sbyte or byte or short or ushort or int or uint or long or ulong or float or double or decimal:
                    return Convert.ToDouble(value, System.Globalization.CultureInfo.InvariantCulture).GetHashCode();
                default:
                    return 1;
            }
        }
    }
}
