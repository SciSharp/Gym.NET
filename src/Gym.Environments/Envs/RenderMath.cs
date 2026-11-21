#nullable enable
using System;
using System.Numerics;
using Gym.Utils;
using NumSharp;

namespace Gym.Environments.Envs {
    /// <summary>The conversions Gymnasium's render code applies to coordinates before handing them to pygame.</summary>
    internal static class RenderMath {
        /// <summary>
        ///     Python's <c>int(value)</c> of a float coordinate (truncation toward zero), saturated to <see langword="long"/>,
        ///     which keeps pygame's verdict: any value beyond a C long is refused by the primitive with the same message.
        /// </summary>
        /// <param name="value">The coordinate.</param>
        /// <returns>The integer.</returns>
        /// <exception cref="ValueError">Thrown for NaN (<c>cannot convert float NaN to integer</c>).</exception>
        /// <exception cref="OverflowError">Thrown for an infinity (<c>cannot convert float infinity to integer</c>).</exception>
        public static long Int(double value) {
            BigInteger integer = PyBuiltins.Int(value);
            if (integer > long.MaxValue)
                return long.MaxValue;
            if (integer < long.MinValue)
                return long.MinValue;
            return (long)integer;
        }

        /// <summary>
        ///     pygame's reading of a float where it wants a C <c>int</c> (a <c>smoothscale</c> size): C's <c>(int)</c> cast as
        ///     compiled for x64, truncation toward zero, with NaN and every value outside the <see langword="int"/> range giving
        ///     <see cref="int.MinValue"/> (the processor's "integer indefinite"), which pygame then refuses as a negative size.
        /// </summary>
        /// <param name="value">The float.</param>
        /// <returns>The integer pygame works with.</returns>
        /// <remarks>
        ///     .NET's own <c>(int)</c> cast saturates out-of-range values since .NET 9 and maps NaN to 0, so it cannot stand in for
        ///     C's. Measured against pygame-ce 2.5.8 on Windows x64: <c>smoothscale(img, (nan, nan))</c>, <c>(inf, inf)</c> and
        ///     <c>(1e10, 1e10)</c> raise <c>Cannot scale to negative size</c>, <c>(0.9, 0.9)</c> and <c>(-0.5, -0.5)</c> give 0×0.
        /// </remarks>
        public static int CInt(double value) =>
            // NaN fails both comparisons, so it takes the indefinite value with the out-of-range ones.
            value > -2147483649.0 && value < 2147483648.0 ? (int)value : int.MinValue;
    }
}
