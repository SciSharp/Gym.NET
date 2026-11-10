#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Envs;
using Gym.Observations;
using NumSharp;

namespace Gym.Utils {
    /// <summary>
    ///     Converts a single env's transition between the pre-0.26 "done" step API, <c>(obs, reward, done, info)</c>, and
    ///     the terminated/truncated API, <c>(obs, reward, terminated, truncated, info)</c>. The two are told apart by
    ///     <c>info["TimeLimit.truncated"]</c>, which the old API used to mark a time-limit ending.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Python's <c>step_api_compatibility(step_returns, output_truncation_bool, is_vector_env)</c> accepts either form and
    ///     dispatches on the tuple's length: <see cref="Dispatch"/> does the same over an <see langword="object"/>?[] tuple,
    ///     single or vector (a list of infos, or a dict of arrays). For a typed single-env transition the argument's type
    ///     already says which form it is, so the typed conversions are called directly.
    ///     </para>
    ///     <para>
    ///     As in Gymnasium, every conversion edits the given info in place: the done conversion adds
    ///     <c>"TimeLimit.truncated"</c>, the terminated/truncated conversion removes it.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/step_api_compatibility.py">gymnasium/utils/step_api_compatibility.py</see>.</para>
    /// </remarks>
    public static class StepApiCompatibility {
        /// <summary>The info key the done API uses to mark a truncation (<c>"TimeLimit.truncated"</c>).</summary>
        public const string TimeLimitTruncatedKey = "TimeLimit.truncated";

        /// <summary>
        ///     Splits a done-API transition's <c>done</c> into <c>terminated = done and not truncated</c> and
        ///     <c>truncated = done and truncated</c>, where <c>truncated</c> is popped from <c>info["TimeLimit.truncated"]</c>
        ///     (absent means <see langword="false"/>), so a time-limit ending stays distinguishable from a terminal state.
        /// </summary>
        /// <typeparam name="TObs">The observation type.</typeparam>
        /// <param name="stepReturns">The done-API transition; its info loses the <c>"TimeLimit.truncated"</c> key.</param>
        /// <returns>The same transition in the terminated/truncated API.</returns>
        /// <exception cref="ArgumentNullException">Thrown when the transition's info is <see langword="null"/>.</exception>
        /// <remarks>
        ///     Python's <c>and</c> returns an operand (so <c>done</c> of <c>0</c> stays <c>0</c>); the flags here are
        ///     <see langword="bool"/>, the truth value of that operand. The popped value counts by Python truthiness.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/step_api_compatibility.py#L27">gymnasium/utils/step_api_compatibility.py</see>.
        /// </remarks>
        public static StepResult<TObs> ConvertToTerminatedTruncatedStepApi<TObs>(DoneStepResult<TObs> stepReturns) {
            (TObs observations, double rewards, bool dones, Info infos) = stepReturns;
            if (infos is null)
                throw new ArgumentNullException(nameof(stepReturns), "The transition's info is null; Gymnasium's step API always returns a dict.");
            bool truncated = IsTruthy(infos.Pop(TimeLimitTruncatedKey, false));
            return new StepResult<TObs>(observations, rewards, dones && !truncated, dones && truncated, infos);
        }

        /// <summary>
        ///     Merges a transition's flags into <c>done = terminated or truncated</c> and, when the episode ended, records
        ///     <c>info["TimeLimit.truncated"] = truncated and not terminated</c>, so legacy consumers can still tell a time limit
        ///     from a terminal state.
        /// </summary>
        /// <typeparam name="TObs">The observation type.</typeparam>
        /// <param name="stepReturns">The terminated/truncated transition; its info gains the key when the episode ended.</param>
        /// <returns>The same transition in the done API.</returns>
        /// <exception cref="ArgumentNullException">Thrown when the transition's info is <see langword="null"/>.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/step_api_compatibility.py#L82">gymnasium/utils/step_api_compatibility.py</see>.</remarks>
        public static DoneStepResult<TObs> ConvertToDoneStepApi<TObs>(StepResult<TObs> stepReturns) {
            (TObs observations, double rewards, bool terminated, bool truncated, Info infos) = stepReturns;
            if (infos is null)
                throw new ArgumentNullException(nameof(stepReturns), "The transition's info is null; Gymnasium's step API always returns a dict.");
            if (truncated || terminated)
                infos[TimeLimitTruncatedKey] = truncated && !terminated;
            return new DoneStepResult<TObs>(observations, rewards, terminated || truncated, infos);
        }

        /// <summary>
        ///     Python's <c>step_api_compatibility(step_returns, output_truncation_bool, is_vector_env)</c>: converts a step's
        ///     returns, whichever API they follow, to the terminated/truncated API (<paramref name="outputTruncationBool"/>) or to
        ///     the done API. The returns are a Python tuple, an <see langword="object"/>?[] of four or five items; the result is a
        ///     new array (a returns array already in the asked form is returned as it is, as Python returns its tuple).
        /// </summary>
        /// <param name="stepReturns"><c>(obs, rew, done, info)</c> or <c>(obs, rew, terminated, truncated, info)</c>; the info is edited in place.</param>
        /// <param name="outputTruncationBool">Whether the output has two flags (the new API, the default) or one.</param>
        /// <param name="isVectorEnv">Whether the returns are a vector env's: flag arrays, and infos as a list of dicts or one dict of arrays.</param>
        /// <returns>The returns in the asked API.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="stepReturns"/> is <see langword="null"/>.</exception>
        /// <exception cref="AssertionError">Thrown when the returns have neither four nor five items.</exception>
        /// <exception cref="TypeError">Thrown for vector returns whose infos are neither a list nor a dict, with Gymnasium's message.</exception>
        /// <exception cref="ValueError">Thrown when a list of infos and the flags differ in length (Python's <c>zip(strict=True)</c>).</exception>
        /// <remarks>
        ///     The class's name is taken, so the function is <c>Dispatch</c> in C# and <c>step_api_compatibility</c> by its
        ///     Python name. The typed overloads of the two conversions cover a single env's <see cref="StepResult{TObs}"/> and
        ///     <see cref="DoneStepResult{TObs}"/>.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/step_api_compatibility.py#L139">gymnasium/utils/step_api_compatibility.py</see>.
        /// </remarks>
        [PyName("step_api_compatibility")]
        public static object?[] Dispatch(object?[] stepReturns, bool outputTruncationBool = true, bool isVectorEnv = false) =>
            outputTruncationBool ? ConvertToTerminatedTruncatedStepApi(stepReturns, isVectorEnv) : ConvertToDoneStepApi(stepReturns, isVectorEnv);

        /// <summary>
        ///     Python's <c>convert_to_terminated_truncated_step_api(step_returns, is_vector_env)</c> over a returns tuple:
        ///     five items pass through; four are split by <c>info["TimeLimit.truncated"]</c>, popped from the info (or each
        ///     info of a list, or the dict of arrays).
        /// </summary>
        /// <param name="stepReturns">The returns tuple; the info loses the key.</param>
        /// <param name="isVectorEnv">Whether the returns are a vector env's.</param>
        /// <returns>The five returns.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="stepReturns"/> is <see langword="null"/>.</exception>
        /// <exception cref="AssertionError">Thrown when the returns have neither four nor five items.</exception>
        /// <exception cref="TypeError">Thrown for vector infos that are neither a list nor a dict.</exception>
        /// <remarks>
        ///     A single env's flags are Python's <c>and</c> of the operands, so a falsy <c>done</c> comes back as itself, and a
        ///     true one as <c>not truncated</c> and the popped value.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/step_api_compatibility.py#L27">gymnasium/utils/step_api_compatibility.py</see>.
        /// </remarks>
        public static object?[] ConvertToTerminatedTruncatedStepApi(object?[] stepReturns, bool isVectorEnv = false) {
            if (stepReturns is null)
                throw new ArgumentNullException(nameof(stepReturns));
            if (stepReturns.Length == 5)
                return stepReturns;
            if (stepReturns.Length != 4)
                throw new AssertionError("");
            (object? observations, object? rewards, object? dones, object? infos) = (stepReturns[0], stepReturns[1], stepReturns[2], stepReturns[3]);

            // Cases to handle - info single env /  info vector env (list) / info vector env (dict)
            if (!isVectorEnv) {
                object? truncated = ((Info)infos!).Pop(TimeLimitTruncatedKey, false);
                bool done = IsTruthy(dones);
                return new[] { observations, rewards, done ? !IsTruthy(truncated) : dones, done ? truncated : dones, infos };
            }

            switch (infos) {
                case System.Collections.IList list and not NDArray: {
                    bool[] popped = list.Cast<Info>().Select(info => IsTruthy(info.Pop(TimeLimitTruncatedKey, false))).ToArray();
                    using NDArray truncated = np.array(popped);
                    using NDArray notTruncated = np.logical_not(truncated);
                    using NDArray doneArray = AsArray(dones);
                    return new object?[] { observations, rewards, np.logical_and(doneArray, notTruncated), np.logical_and(doneArray, truncated), infos };
                }
                case Info dict: {
                    using NDArray doneArray = AsArray(dones);
                    int numEnvs = (int)doneArray.shape[0];
                    object? popped = dict.Pop(TimeLimitTruncatedKey, null);
                    NDArray truncated = popped as NDArray ?? (popped is null ? np.zeros(new Shape(numEnvs), NPTypeCode.Boolean) : np.asanyarray(popped));
                    try {
                        using NDArray notTruncated = np.logical_not(truncated);
                        return new object?[] { observations, rewards, np.logical_and(doneArray, notTruncated), np.logical_and(doneArray, truncated), infos };
                    } finally {
                        // The popped array left the info, and the default was made here: the result holds new arrays only.
                        truncated.Dispose();
                    }
                }
                default:
                    throw new TypeError($"Unexpected value of infos, as is_vector_envs=False, expects `info` to be a list or dict, actual type: {PyBuiltins.TypeRepr(infos)}");
            }
        }

        /// <summary>
        ///     Python's <c>convert_to_done_step_api(step_returns, is_vector_env)</c> over a returns tuple: four items pass
        ///     through; five merge their flags into <c>done</c>, recording <c>info["TimeLimit.truncated"]</c> for the endings.
        /// </summary>
        /// <param name="stepReturns">The returns tuple; the info gains the key where an episode ended.</param>
        /// <param name="isVectorEnv">Whether the returns are a vector env's.</param>
        /// <returns>The four returns.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="stepReturns"/> is <see langword="null"/>.</exception>
        /// <exception cref="AssertionError">Thrown when the returns have neither four nor five items.</exception>
        /// <exception cref="TypeError">Thrown for vector infos that are neither a list nor a dict.</exception>
        /// <exception cref="ValueError">Thrown when a list of infos and the flags differ in length.</exception>
        /// <remarks>
        ///     A single env's <c>done</c> is Python's <c>terminated or truncated</c> (the first truthy operand, else the second),
        ///     and the recorded value <c>truncated and not terminated</c>.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/step_api_compatibility.py#L82">gymnasium/utils/step_api_compatibility.py</see>.
        /// </remarks>
        public static object?[] ConvertToDoneStepApi(object?[] stepReturns, bool isVectorEnv = false) {
            if (stepReturns is null)
                throw new ArgumentNullException(nameof(stepReturns));
            if (stepReturns.Length == 4)
                return stepReturns;
            if (stepReturns.Length != 5)
                throw new AssertionError("");
            (object? observations, object? rewards, object? terminated, object? truncated, object? infos) =
                (stepReturns[0], stepReturns[1], stepReturns[2], stepReturns[3], stepReturns[4]);

            // Cases to handle - info single env /  info vector env (list) / info vector env (dict)
            if (!isVectorEnv) {
                bool isTruncated = IsTruthy(truncated), isTerminated = IsTruthy(terminated);
                if (isTruncated || isTerminated)
                    ((Info)infos!)[TimeLimitTruncatedKey] = isTruncated ? !isTerminated : truncated;
                return new[] { observations, rewards, isTerminated ? terminated : truncated, infos };
            }

            switch (infos) {
                case System.Collections.IList list and not NDArray: {
                    IReadOnlyList<object?> truncations = Items(truncated), terminations = Items(terminated);
                    // zip(infos, truncated, terminated, strict=True): the pairs before a mismatch are written first.
                    for (int i = 0;; i++) {
                        if (i == list.Count) {
                            if (i < truncations.Count)
                                throw new ValueError("zip() argument 2 is longer than argument 1");
                            if (i < terminations.Count)
                                throw new ValueError("zip() argument 3 is longer than arguments 1-2");
                            break;
                        }

                        if (i == truncations.Count)
                            throw new ValueError("zip() argument 2 is shorter than argument 1");
                        if (i == terminations.Count)
                            throw new ValueError("zip() argument 3 is shorter than arguments 1-2");
                        bool envTruncated = IsTruthy(truncations[i]), envTerminated = IsTruthy(terminations[i]);
                        if (envTruncated || envTerminated)
                            ((Info)list[i]!)[TimeLimitTruncatedKey] = envTruncated ? !envTerminated : truncations[i];
                    }

                    using NDArray terminatedArray = AsArray(terminated), truncatedArray = AsArray(truncated);
                    return new object?[] { observations, rewards, np.logical_or(terminatedArray, truncatedArray), infos };
                }
                case Info dict: {
                    using NDArray terminatedArray = AsArray(terminated), truncatedArray = AsArray(truncated);
                    if (np.any(truncatedArray) || np.any(terminatedArray)) {
                        using NDArray notTerminated = np.logical_not(terminatedArray);
                        dict[TimeLimitTruncatedKey] = np.logical_and(truncatedArray, notTerminated);
                    }

                    return new object?[] { observations, rewards, np.logical_or(terminatedArray, truncatedArray), infos };
                }
                default:
                    throw new TypeError($"Unexpected value of infos, as is_vector_envs=False, expects `info` to be a list or dict, actual type: {PyBuiltins.TypeRepr(infos)}");
            }
        }

        /// <summary>A flag batch as an array: a copy of an array (the caller keeps its own), or a new bool array of a list's truth values.</summary>
        /// <param name="value">The flags.</param>
        /// <returns>A new array the caller disposes.</returns>
        private static NDArray AsArray(object? value) => value is NDArray array ? array.copy() : np.array(Items(value).Select(IsTruthy).ToArray());

        /// <summary>A batch's items: an array's elements, or a list's or tuple's items.</summary>
        /// <param name="value">The batch.</param>
        /// <returns>The items (an array's as boxed values).</returns>
        /// <exception cref="TypeError">Thrown for a value that is not a batch.</exception>
        private static IReadOnlyList<object?> Items(object? value) => value switch {
            NDArray array => Enumerable.Range(0, (int)array.size).Select(i => array.GetAtIndex(i)).ToArray(),
            object?[] items => items,
            System.Collections.IList list => list.Cast<object?>().ToArray(),
            _ => throw new TypeError($"'{PyBuiltins.TypeName(value)}' object is not iterable"),
        };

        /// <summary>
        ///     Python's truth value of a popped <c>"TimeLimit.truncated"</c> value: <see langword="false"/> for <c>None</c>,
        ///     <c>False</c>, zero numbers, empty strings and empty collections; a one-element array's element; true for any
        ///     other object.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>Its truthiness.</returns>
        /// <exception cref="ValueError">Thrown for an empty array or one of more than one element, with NumPy 2.4's messages.</exception>
        private static bool IsTruthy(object? value) {
            switch (value) {
                case null:
                    return false;
                case bool flag:
                    return flag;
                case string text:
                    return text.Length > 0;
                case NDArray array:
                    if (array.size == 1)
                        return IsTruthy(array.GetAtIndex(0));
                    if (array.size == 0)
                        throw new ValueError("The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty.");
                    throw new ValueError("The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()");
                case System.Collections.ICollection collection:
                    return collection.Count > 0;
                case sbyte or byte or short or ushort or int or uint or long or ulong or float or double or decimal:
                    // bool(nan) is True in Python, and NaN != 0 here too.
                    return Convert.ToDouble(value, System.Globalization.CultureInfo.InvariantCulture) != 0.0;
                default:
                    return true;
            }
        }
    }
}
