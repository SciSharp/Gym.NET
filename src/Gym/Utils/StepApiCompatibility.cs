#nullable enable
using System;
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
    ///     Python's <c>step_api_compatibility(step_returns, output_truncation_bool)</c> accepts either form and dispatches on
    ///     the tuple's length; in C# the argument's type already says which form it is, so the two conversions are called
    ///     directly (a transition already in the requested form needs no call).
    ///     </para>
    ///     <para>
    ///     The <c>is_vector_env=True</c> forms (a list of infos, or a dict of arrays with <c>"_"</c> masks) come with the
    ///     vector API (P6). As in Gymnasium, both conversions edit the given info in place: the done conversion adds
    ///     <c>"TimeLimit.truncated"</c>, the terminated/truncated conversion removes it.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/utils/step_api_compatibility.py">gymnasium/utils/step_api_compatibility.py</see>.</para>
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
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/utils/step_api_compatibility.py#L27">gymnasium/utils/step_api_compatibility.py</see>.
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
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/utils/step_api_compatibility.py#L82">gymnasium/utils/step_api_compatibility.py</see>.</remarks>
        public static DoneStepResult<TObs> ConvertToDoneStepApi<TObs>(StepResult<TObs> stepReturns) {
            (TObs observations, double rewards, bool terminated, bool truncated, Info infos) = stepReturns;
            if (infos is null)
                throw new ArgumentNullException(nameof(stepReturns), "The transition's info is null; Gymnasium's step API always returns a dict.");
            if (truncated || terminated)
                infos[TimeLimitTruncatedKey] = truncated && !terminated;
            return new DoneStepResult<TObs>(observations, rewards, terminated || truncated, infos);
        }

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
