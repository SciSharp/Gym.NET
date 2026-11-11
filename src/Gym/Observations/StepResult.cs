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
    ///     reflexive. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/core.py#L79">gymnasium/core.py (Env.step)</see>.
    /// </remarks>
    public readonly record struct StepResult<TObs>(TObs Observation, double Reward, bool Terminated, bool Truncated, Info Info) : INDArrayCarrier, IDisposable {
        /// <summary>
        ///     Hands the observation's and the info's arrays to an <see cref="NDScope"/> (D19): an <c>[NDScoped]</c> method that
        ///     returns a transition keeps them alive while its temporaries are reclaimed.
        /// </summary>
        /// <param name="scope">The scope the arrays leave through.</param>
        void INDArrayCarrier.YieldTo(NDScope scope) => ResultEquality.Yield(scope, Observation, Info);

        /// <summary>
        ///     Releases the arrays the transition holds, the observation's and the info's, nested ones included, so
        ///     <c>using var step = env.Step(action);</c> frees them at the end of the scope: the arrays an env returns belong to
        ///     the caller (D19).
        /// </summary>
        /// <remarks>
        ///     Footgun: copies of this struct share the same arrays, so dispose once, after the last use of the observation
        ///     and the info; keep anything needed longer with <c>.copy()</c>. Disposing again does nothing more.
        /// </remarks>
        public void Dispose() => ResultEquality.Release(Observation, Info);

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
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/core.py#L124">gymnasium/core.py (Env.reset)</see>.
    /// </remarks>
    public readonly record struct ResetResult<TObs>(TObs Observation, Info Info) : INDArrayCarrier, IDisposable {
        /// <summary>Hands the observation's and the info's arrays to an <see cref="NDScope"/>, as <see cref="StepResult{TObs}"/> does.</summary>
        /// <param name="scope">The scope the arrays leave through.</param>
        void INDArrayCarrier.YieldTo(NDScope scope) => ResultEquality.Yield(scope, Observation, Info);

        /// <summary>
        ///     Releases the observation's and the info's arrays, as <see cref="StepResult{TObs}.Dispose"/> does:
        ///     <c>using var reset = env.Reset(seed: 0);</c>.
        /// </summary>
        /// <remarks>Footgun: copies share the arrays; dispose once, after their last use.</remarks>
        public void Dispose() => ResultEquality.Release(Observation, Info);

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
    ///     Equality follows <see cref="StepResult{TObs}"/>'s. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/step_api_compatibility.py#L11">gymnasium/utils/step_api_compatibility.py (DoneStepType)</see>.
    /// </remarks>
    public readonly record struct DoneStepResult<TObs>(TObs Observation, double Reward, bool Done, Info Info) : INDArrayCarrier, IDisposable {
        /// <summary>Hands the observation's and the info's arrays to an <see cref="NDScope"/>, as <see cref="StepResult{TObs}"/> does.</summary>
        /// <param name="scope">The scope the arrays leave through.</param>
        void INDArrayCarrier.YieldTo(NDScope scope) => ResultEquality.Yield(scope, Observation, Info);

        /// <summary>Releases the observation's and the info's arrays, as <see cref="StepResult{TObs}.Dispose"/> does.</summary>
        /// <remarks>Footgun: copies share the arrays; dispose once, after their last use.</remarks>
        public void Dispose() => ResultEquality.Release(Observation, Info);

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
        ///     Yields every array a result holds, the observation's and the info's, nested ones included (an info's
        ///     <c>final_obs</c>, a composite observation's parts), so none is reclaimed with the scope's temporaries.
        /// </summary>
        /// <param name="scope">The scope.</param>
        /// <param name="values">The observation and the info.</param>
        internal static void Yield(NDScope scope, params object?[] values) {
            foreach (object? value in values)
                YieldOne(scope, value, depth: 0);
        }

        /// <summary>Yields one value's arrays, walking containers to a fixed depth so a self-containing info cannot recurse forever.</summary>
        /// <param name="scope">The scope.</param>
        /// <param name="value">The value.</param>
        /// <param name="depth">How deep the walk is; containers below 16 levels are left alone.</param>
        private static void YieldOne(NDScope scope, object? value, int depth) {
            if (depth > 16)
                return;
            switch (value) {
                case NDArray array:
                    scope.Returns(array);
                    break;
                case INDArrayCarrier carrier:
                    carrier.YieldTo(scope);
                    break;
                case string:
                    break;
                case IEnumerable<KeyValuePair<string, object?>> pairs:
                    foreach (KeyValuePair<string, object?> pair in pairs)
                        YieldOne(scope, pair.Value, depth + 1);
                    break;
                case System.Collections.IDictionary dictionary:
                    foreach (System.Collections.DictionaryEntry entry in dictionary)
                        YieldOne(scope, entry.Value, depth + 1);
                    break;
                case System.Runtime.CompilerServices.ITuple tuple:
                    for (int i = 0; i < tuple.Length; i++)
                        YieldOne(scope, tuple[i], depth + 1);
                    break;
                case System.Collections.IEnumerable items:
                    foreach (object? item in items)
                        YieldOne(scope, item, depth + 1);
                    break;
            }
        }

        /// <summary>
        ///     Disposes every array a result holds, the ones <see cref="Yield"/> would hand to a scope: the observation's and the
        ///     info's, nested ones included (an info's <c>final_obs</c>, a composite observation's parts).
        /// </summary>
        /// <param name="values">The result's parts.</param>
        /// <remarks>An array reachable twice (an observation also kept in the info) is disposed twice, which NumSharp's idempotent <see cref="NDArray.Dispose"/> absorbs.</remarks>
        internal static void Release(params object?[] values) {
            foreach (object? value in values)
                ReleaseOne(value, depth: 0);
        }

        /// <summary>Disposes one value's arrays, walking containers to the same fixed depth as <see cref="YieldOne"/>.</summary>
        /// <param name="value">The value.</param>
        /// <param name="depth">How deep the walk is; containers below 16 levels are left alone.</param>
        private static void ReleaseOne(object? value, int depth) {
            if (depth > 16)
                return;
            switch (value) {
                case NDArray array:
                    array.Dispose();
                    break;
                case INDArrayCarrier and IDisposable carrier:
                    // A nested result or graph instance: its own Dispose knows its arrays.
                    carrier.Dispose();
                    break;
                case string:
                    break;
                case IEnumerable<KeyValuePair<string, object?>> pairs:
                    foreach (KeyValuePair<string, object?> pair in pairs)
                        ReleaseOne(pair.Value, depth + 1);
                    break;
                case System.Collections.IDictionary dictionary:
                    foreach (System.Collections.DictionaryEntry entry in dictionary)
                        ReleaseOne(entry.Value, depth + 1);
                    break;
                case System.Runtime.CompilerServices.ITuple tuple:
                    for (int i = 0; i < tuple.Length; i++)
                        ReleaseOne(tuple[i], depth + 1);
                    break;
                case System.Collections.IEnumerable items:
                    foreach (object? item in items)
                        ReleaseOne(item, depth + 1);
                    break;
            }
        }

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
