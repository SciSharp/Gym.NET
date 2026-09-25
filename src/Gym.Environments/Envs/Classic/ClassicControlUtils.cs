#nullable enable
using System;
using System.Collections.Generic;
using Gym.Utils;
using NumSharp;

namespace Gym.Environments.Envs.Classic {
    /// <summary>
    ///     The reset-option parsing every classic-control env shares, so <c>Reset(seed, options)</c> accepts and rejects
    ///     exactly what Gymnasium's <c>reset</c> does, with the same messages.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/envs/classic_control/utils.py">gymnasium/envs/classic_control/utils.py</see>.</remarks>
    public static class ClassicControlUtils {
        /// <summary>
        ///     Converts a reset option with Python's <c>float(x)</c>, turning its <c>ValueError</c> and <c>TypeError</c> into
        ///     one <see cref="ValueError"/> that names the option.
        /// </summary>
        /// <param name="x">The option's value: a number, a numeric string (<c>"-0.5"</c>, <c>"1_0"</c>), a 0-d array...</param>
        /// <returns>The value as a float.</returns>
        /// <exception cref="ValueError">
        ///     Thrown when Python's <c>float()</c> rejects the value: <c>An option (abc) could not be converted to a float.</c>
        ///     (<c>{x}</c> is Python's <c>str(x)</c>, see <see cref="PyBuiltins.Str"/>).
        /// </exception>
        /// <exception cref="OverflowException">
        ///     Thrown, as Python's <c>OverflowError</c>, for an integer beyond the float range: Gymnasium catches only
        ///     <c>ValueError</c> and <c>TypeError</c>, so this one propagates unchanged.
        /// </exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/envs/classic_control/utils.py#L8">gymnasium/envs/classic_control/utils.py</see>.</remarks>
        public static double VerifyNumberAndCast(object? x) {
            try {
                return PyBuiltins.Float(x);
            } catch (Exception e) when (e is ValueError or TypeError) {
                // raise ValueError(...) from e: the conversion error stays attached as the cause.
                throw new ValueError($"An option ({PyBuiltins.Str(x)}) could not be converted to a float.", e);
            }
        }

        /// <summary>
        ///     The bounds a reset draws its initial state from: <c>options["low"]</c> and <c>options["high"]</c> where given,
        ///     the defaults otherwise.
        /// </summary>
        /// <param name="options">The reset options (Python's <c>options</c> dict), or <see langword="null"/> for the defaults. Other keys are ignored.</param>
        /// <param name="defaultLow">The lower bound when <c>"low"</c> is absent.</param>
        /// <param name="defaultHigh">The upper bound when <c>"high"</c> is absent.</param>
        /// <returns>The bounds, both converted by <see cref="VerifyNumberAndCast"/>.</returns>
        /// <exception cref="ValueError">
        ///     Thrown for a bound <c>float()</c> rejects (a present key with a <see langword="null"/> value included, which is
        ///     Python's <c>None</c>), and when the lower bound exceeds the upper one:
        ///     <c>Lower bound (0.1) must be lower than higher bound (-0.1).</c>
        /// </exception>
        /// <exception cref="OverflowException">Thrown for an integer bound beyond the float range.</exception>
        /// <remarks>
        ///     Equal bounds are accepted, and NaN passes the comparison (<c>nan &gt; x</c> is false), exactly as in Python; a NaN or
        ///     infinite bound then fails in the generator's <c>uniform</c> instead (<c>high - low range exceeds valid bounds</c>).
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/envs/classic_control/utils.py#L17">gymnasium/envs/classic_control/utils.py</see>.
        /// </remarks>
        public static (double Low, double High) MaybeParseResetBounds(IReadOnlyDictionary<string, object?>? options, double defaultLow, double defaultHigh) {
            if (options is null)
                return (defaultLow, defaultHigh);

            // options.get("low") if "low" in options else default_low: a present key wins even when its value is None.
            object? low = options.TryGetValue("low", out object? givenLow) ? givenLow : defaultLow;
            object? high = options.TryGetValue("high", out object? givenHigh) ? givenHigh : defaultHigh;

            double lowValue = VerifyNumberAndCast(low);
            double highValue = VerifyNumberAndCast(high);
            if (lowValue > highValue)
                throw new ValueError($"Lower bound ({PyBuiltins.FloatRepr(lowValue)}) must be lower than higher bound ({PyBuiltins.FloatRepr(highValue)}).");
            return (lowValue, highValue);
        }
    }
}
