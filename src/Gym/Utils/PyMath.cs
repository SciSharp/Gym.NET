#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using NumSharp;

namespace Gym.Utils {
    /// <summary>
    ///     Python's and NumPy's scalar arithmetic where it differs from C#'s (port plan §3.2 and §5.3, workstream F3): floor
    ///     division and modulo that follow the divisor's sign, <c>min</c>/<c>max</c> whose NaN results depend on argument
    ///     order, <c>round</c> with half-to-even and exact decimal rounding, and NumPy's <c>clip</c> and <c>sign</c>. Gym's
    ///     env math calls these wherever Gymnasium's source uses the Python operator or function, so results match
    ///     Gymnasium's bit for bit; the C# operator there would drift on negative operands, NaNs and ties.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Python raises where C# would return a value: division or modulo by zero throws
    ///     <see cref="DivideByZeroException"/> with Python's <c>ZeroDivisionError</c> message, and an integer result Python
    ///     could hold but <see langword="long"/> cannot (only <c>long.MinValue // -1</c>) throws
    ///     <see cref="OverflowException"/>, like the checked arithmetic §3.2 asks for.
    ///     </para>
    ///     <para>
    ///     The float operations reproduce CPython 3.12's <c>floatobject.c</c> operation for operation on top of the reference
    ///     platform's C <c>fmod</c> (UCRT), including the sign of the NaN a non-finite dividend produces, so even the NaN bit
    ///     patterns match. The live parity suite checks every member against the embedded interpreter.
    ///     </para>
    ///     <para>
    ///     Reference: <see href="https://github.com/python/cpython/blob/v3.12.12/Objects/floatobject.c">Objects/floatobject.c</see> and <see href="https://github.com/python/cpython/blob/v3.12.12/Python/bltinmodule.c#L1745">Python/bltinmodule.c</see> (CPython
    ///     3.12), <see href="https://github.com/numpy/numpy/blob/v2.4.2/numpy/_core/src/umath/clip.cpp">numpy/_core/src/umath/clip.cpp</see> (NumPy 2.4.2).
    ///     </para>
    /// </remarks>
    public static class PyMath {
        /// <summary>
        ///     Python's <c>a // b</c> for ints: the quotient rounded toward negative infinity (<c>-7 // 2 == -4</c>), where C#'s
        ///     <c>/</c> truncates toward zero (<c>-3</c>).
        /// </summary>
        /// <param name="a">The dividend.</param>
        /// <param name="b">The divisor.</param>
        /// <returns>The floored quotient.</returns>
        /// <exception cref="DivideByZeroException">Thrown when <paramref name="b"/> is 0: <c>integer division or modulo by zero</c>.</exception>
        /// <exception cref="OverflowException">Thrown for <c>long.MinValue // -1</c>, whose result 2^63 Python holds and <see langword="long"/> cannot.</exception>
        public static long FloorDiv(long a, long b) {
            if (b == 0)
                throw new DivideByZeroException("integer division or modulo by zero");
            if (a == long.MinValue && b == -1)
                throw new OverflowException("Python's result 9223372036854775808 does not fit in a 64-bit integer.");
            long quotient = a / b;
            // C# truncated toward zero; a non-zero remainder whose sign differs from the divisor's means the true quotient
            // lies one below.
            if (a % b != 0 && (a % b < 0) != (b < 0))
                quotient -= 1;
            return quotient;
        }

        /// <summary>
        ///     Python's <c>a % b</c> for ints: the remainder takes the divisor's sign (<c>-7 % 2 == 1</c>), where C#'s <c>%</c>
        ///     takes the dividend's (<c>-1</c>).
        /// </summary>
        /// <param name="a">The dividend.</param>
        /// <param name="b">The divisor.</param>
        /// <returns>The remainder, in <c>[0, b)</c> for a positive divisor and <c>(b, 0]</c> for a negative one.</returns>
        /// <exception cref="DivideByZeroException">Thrown when <paramref name="b"/> is 0: <c>integer modulo by zero</c>.</exception>
        public static long Mod(long a, long b) {
            if (b == 0)
                throw new DivideByZeroException("integer modulo by zero");
            // Every integer is a multiple of -1; C#'s long.MinValue % -1 would throw an overflow from the hardware divide.
            if (b == -1)
                return 0;
            long remainder = a % b;
            if (remainder != 0 && (remainder < 0) != (b < 0))
                remainder += b;
            return remainder;
        }

        /// <summary>Python's <c>divmod(a, b)</c> for ints: <c>(a // b, a % b)</c> in one call.</summary>
        /// <param name="a">The dividend.</param>
        /// <param name="b">The divisor.</param>
        /// <returns>The floored quotient and the divisor-signed remainder.</returns>
        /// <exception cref="DivideByZeroException">Thrown when <paramref name="b"/> is 0: <c>integer division or modulo by zero</c>.</exception>
        /// <exception cref="OverflowException">Thrown for <c>divmod(long.MinValue, -1)</c> (see <see cref="FloorDiv(long, long)"/>).</exception>
        public static (long Quotient, long Remainder) DivMod(long a, long b) {
            if (b == 0)
                throw new DivideByZeroException("integer division or modulo by zero");
            return (FloorDiv(a, b), Mod(a, b));
        }

        /// <summary>
        ///     Python's <c>a // b</c> for floats, CPython's <c>float_floor_div</c>: the floored quotient computed from the exact
        ///     remainder and snapped to the nearest integer, which is not always <c>Math.Floor(a / b)</c> (<c>1 // 0.1</c> is
        ///     <c>9.0</c>, while <c>1 / 0.1</c> rounds to exactly 10).
        /// </summary>
        /// <param name="a">The dividend.</param>
        /// <param name="b">The divisor.</param>
        /// <returns>The floored quotient; a zero result carries the sign of <c>a / b</c>.</returns>
        /// <exception cref="DivideByZeroException">Thrown when <paramref name="b"/> is ±0: <c>float floor division by zero</c>.</exception>
        public static double FloorDiv(double a, double b) {
            if (b == 0.0)
                throw new DivideByZeroException("float floor division by zero");
            return FloatDivMod(a, b).Quotient;
        }

        /// <summary>
        ///     Python's <c>a % b</c> for floats, CPython's <c>float_rem</c>: C's <c>fmod</c> moved into the divisor's sign
        ///     (<c>-7.5 % 2 == 0.5</c>, <c>-5.0 % inf == inf</c>); a zero result is a zero of the divisor's sign.
        /// </summary>
        /// <param name="a">The dividend.</param>
        /// <param name="b">The divisor.</param>
        /// <returns>The remainder.</returns>
        /// <exception cref="DivideByZeroException">Thrown when <paramref name="b"/> is ±0: <c>float modulo</c> (Python 3.12's message).</exception>
        public static double Mod(double a, double b) {
            if (b == 0.0)
                throw new DivideByZeroException("float modulo");
            double mod = Fmod(a, b);
            // C's `if (mod)`: NaN is non-zero, so a NaN remainder takes this branch and stays NaN.
            if (mod != 0.0) {
                // Ensure the remainder has the same sign as the denominator.
                if ((b < 0) != (mod < 0))
                    mod += b;
            } else {
                // fmod's zero can carry either sign across platforms; Python makes it the divisor's.
                mod = Math.CopySign(0.0, b);
            }

            return mod;
        }

        /// <summary>Python's <c>divmod(a, b)</c> for floats, CPython's <c>float_divmod</c>.</summary>
        /// <param name="a">The dividend.</param>
        /// <param name="b">The divisor.</param>
        /// <returns><c>(a // b, a % b)</c> as <see cref="FloorDiv(double, double)"/> and <see cref="Mod(double, double)"/> compute them.</returns>
        /// <exception cref="DivideByZeroException">Thrown when <paramref name="b"/> is ±0: <c>float divmod()</c>.</exception>
        public static (double Quotient, double Remainder) DivMod(double a, double b) {
            if (b == 0.0)
                throw new DivideByZeroException("float divmod()");
            return FloatDivMod(a, b);
        }

        /// <summary>
        ///     Python's <c>min(a, b)</c>: <paramref name="b"/> only when it compares less than <paramref name="a"/>, so the
        ///     first argument wins ties and every comparison with NaN (<c>min(nan, 1)</c> is <c>nan</c>, <c>min(1, nan)</c> is
        ///     <c>1</c>). <see cref="Math.Min(double, double)"/> returns NaN for both.
        /// </summary>
        /// <typeparam name="T">A type with Python-like comparison operators (every .NET number).</typeparam>
        /// <param name="a">The first value.</param>
        /// <param name="b">The second value.</param>
        /// <returns>The minimum as Python picks it.</returns>
        public static T Min<T>(T a, T b) where T : IComparisonOperators<T, T, bool> => b < a ? b : a;

        /// <summary>
        ///     Python's <c>min(iterable)</c>: the first item no later item compares less than, scanning left to right with
        ///     <c>&lt;</c> as <see cref="Min{T}(T, T)"/> does.
        /// </summary>
        /// <typeparam name="T">A type with Python-like comparison operators.</typeparam>
        /// <param name="values">The items.</param>
        /// <returns>The minimum as Python picks it.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="values"/> is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">Thrown for an empty sequence: <c>min() iterable argument is empty</c>.</exception>
        public static T Min<T>(IEnumerable<T> values) where T : IComparisonOperators<T, T, bool> {
            if (values is null)
                throw new ArgumentNullException(nameof(values));
            using IEnumerator<T> items = values.GetEnumerator();
            if (!items.MoveNext())
                throw new ValueError("min() iterable argument is empty");
            T result = items.Current;
            while (items.MoveNext())
                result = Min(result, items.Current);
            return result;
        }

        /// <summary>
        ///     Python's <c>max(a, b)</c>: <paramref name="b"/> only when it compares greater than <paramref name="a"/>, so the
        ///     first argument wins ties and every comparison with NaN (<c>max(nan, 1)</c> is <c>nan</c>, <c>max(1, nan)</c> is
        ///     <c>1</c>; <c>max(0.0, -0.0)</c> is <c>0.0</c> and <c>max(-0.0, 0.0)</c> is <c>-0.0</c>).
        /// </summary>
        /// <typeparam name="T">A type with Python-like comparison operators (every .NET number).</typeparam>
        /// <param name="a">The first value.</param>
        /// <param name="b">The second value.</param>
        /// <returns>The maximum as Python picks it.</returns>
        public static T Max<T>(T a, T b) where T : IComparisonOperators<T, T, bool> => b > a ? b : a;

        /// <summary>Python's <c>max(iterable)</c>: the first item no later item compares greater than.</summary>
        /// <typeparam name="T">A type with Python-like comparison operators.</typeparam>
        /// <param name="values">The items.</param>
        /// <returns>The maximum as Python picks it.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="values"/> is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">Thrown for an empty sequence: <c>max() iterable argument is empty</c>.</exception>
        public static T Max<T>(IEnumerable<T> values) where T : IComparisonOperators<T, T, bool> {
            if (values is null)
                throw new ArgumentNullException(nameof(values));
            using IEnumerator<T> items = values.GetEnumerator();
            if (!items.MoveNext())
                throw new ValueError("max() iterable argument is empty");
            T result = items.Current;
            while (items.MoveNext())
                result = Max(result, items.Current);
            return result;
        }

        /// <summary>
        ///     NumPy's <c>np.clip(x, low, high)</c> on float64 scalars: <c>x</c> raised to <c>low</c>, then lowered to
        ///     <c>high</c>, where a tie keeps the value already held (<c>clip(-0.0, 0.0, 1)</c> is <c>-0.0</c>) and any NaN
        ///     operand gives NaN. With <c>low &gt; high</c> the result is <c>high</c>, as in NumPy.
        /// </summary>
        /// <param name="x">The value.</param>
        /// <param name="low">The lower bound.</param>
        /// <param name="high">The upper bound.</param>
        /// <returns>The clipped value.</returns>
        /// <remarks>
        ///     Matches NumPy 2.4.2 bit for bit, NaN payloads and signed zeros included, over every combination of
        ///     <c>{0, -0, 1, -1, 0.5, nan, inf, -inf}</c>; the textbook <c>min(max(x, low), high)</c> does not (it turns
        ///     <c>clip(0.0, -0.0, 1)</c> into <c>-0.0</c>). Reference: <see href="https://github.com/numpy/numpy/blob/v2.4.2/numpy/_core/src/umath/clip.cpp#L150">numpy/_core/src/umath/clip.cpp</see>.
        /// </remarks>
        public static double Clip(double x, double low, double high) {
            // NumPy's floating-point max/min keep a NaN first operand, else prefer it on ties.
            double raised = double.IsNaN(x) ? x : x >= low ? x : low;
            return double.IsNaN(raised) ? raised : raised <= high ? raised : high;
        }

        /// <summary>NumPy's <c>np.clip</c> on float32 scalars: <see cref="Clip(double, double, double)"/> computed in float32, as NumPy keeps the dtype.</summary>
        /// <param name="x">The value.</param>
        /// <param name="low">The lower bound.</param>
        /// <param name="high">The upper bound.</param>
        /// <returns>The clipped value.</returns>
        public static float Clip(float x, float low, float high) {
            float raised = float.IsNaN(x) ? x : x >= low ? x : low;
            return float.IsNaN(raised) ? raised : raised <= high ? raised : high;
        }

        /// <summary>
        ///     Python's <c>round(x)</c> with no digits: the nearest integer, ties to even (<c>round(0.5) == 0</c>,
        ///     <c>round(1.5) == 2</c>), as an unbounded integer.
        /// </summary>
        /// <param name="x">The value.</param>
        /// <returns>The rounded integer.</returns>
        /// <exception cref="ValueError">Thrown for NaN: <c>cannot convert float NaN to integer</c>.</exception>
        /// <exception cref="OverflowException">Thrown, as Python's <c>OverflowError</c>, for an infinity: <c>cannot convert float infinity to integer</c>.</exception>
        /// <remarks>
        ///     CPython rounds half away from zero with C's <c>round</c> and fixes exact halves to even; .NET's exact
        ///     <see cref="Math.Round(double, MidpointRounding)"/> with <see cref="MidpointRounding.ToEven"/> is the same function.
        /// </remarks>
        public static BigInteger Round(double x) => PyBuiltins.Int(Math.Round(x, MidpointRounding.ToEven));

        /// <summary>
        ///     Python's <c>round(x, ndigits)</c>: <paramref name="x"/> rounded to <paramref name="ndigits"/> decimal places
        ///     (tens, hundreds, ... when negative) by its exact binary value, ties to even, then converted back to the nearest
        ///     double. So <c>round(2.675, 2)</c> is <c>2.67</c>: the double closest to 2.675 lies below it.
        ///     <see cref="Math.Round(double, int, MidpointRounding)"/> scales by a power of ten first and can differ.
        /// </summary>
        /// <param name="x">The value.</param>
        /// <param name="ndigits">The number of decimal places; negative rounds to the left of the point.</param>
        /// <returns>The rounded value; NaN and the infinities are returned as they are, and a zero keeps <paramref name="x"/>'s sign.</returns>
        /// <exception cref="OverflowException">
        ///     Thrown, as Python's <c>OverflowError</c>, when the rounded value exceeds the double range:
        ///     <c>rounded value too large to represent</c> (<c>round(1.7976931348623157e308, -308)</c>).
        /// </exception>
        /// <remarks>
        ///     CPython 3.12's <c>float___round___impl</c> and <c>double_round</c> round to a decimal string with
        ///     <c>_Py_dg_dtoa</c> mode 3 and parse it back with <c>_Py_dg_strtod</c>; here the decimal rounding is exact
        ///     <see cref="BigInteger"/> arithmetic and the parse is .NET's correctly rounded
        ///     <see cref="double.Parse(string, NumberStyles, IFormatProvider)"/>, which reach the same double. Reference:
        ///     <see href="https://github.com/python/cpython/blob/v3.12.12/Objects/floatobject.c#L962">Objects/floatobject.c</see>.
        /// </remarks>
        public static double Round(double x, int ndigits) {
            // NaNs and infinities round to themselves.
            if (!double.IsFinite(x))
                return x;
            // CPython's NDIGITS_MAX and NDIGITS_MIN: beyond them x rounds to itself, or to a zero of its sign.
            const int ndigitsMax = (int)((53 - -1021) * 0.30103);
            const int ndigitsMin = -(int)((1024 + 1) * 0.30103);
            if (ndigits > ndigitsMax)
                return x;
            if (ndigits < ndigitsMin)
                return 0.0 * x;
            return DoubleRound(x, ndigits);
        }

        /// <summary>
        ///     NumPy's <c>np.sign</c> on a float64 scalar: 1.0 for a positive value, -1.0 for a negative one, 0.0 (positive,
        ///     also for <c>-0.0</c>) for a zero, and a NaN returned as it is. <see cref="Math.Sign(double)"/> throws for NaN
        ///     instead.
        /// </summary>
        /// <param name="x">The value.</param>
        /// <returns>The sign.</returns>
        public static double Sign(double x) => x > 0 ? 1.0 : x < 0 ? -1.0 : x == 0 ? 0.0 : x;

        /// <summary>NumPy's <c>np.sign</c> on a float32 scalar, in float32 (see <see cref="Sign(double)"/>).</summary>
        /// <param name="x">The value.</param>
        /// <returns>The sign.</returns>
        public static float Sign(float x) => x > 0 ? 1f : x < 0 ? -1f : x == 0 ? 0f : x;

        /// <summary>NumPy's <c>np.sign</c> on an integer: -1, 0 or 1.</summary>
        /// <param name="x">The value.</param>
        /// <returns>The sign.</returns>
        public static long Sign(long x) => Math.Sign(x);

        /// <summary>
        ///     <c>np.float32(x)</c>: the float64 value rounded to the nearest float32, ties to even (IEEE), exactly C#'s
        ///     <c>(float)x</c>. Spelled out so code that follows a Gymnasium source shows the cast that source makes, where the
        ///     §3.2 rules allow no other narrowing.
        /// </summary>
        /// <param name="x">The float64 value.</param>
        /// <returns>The float32 value.</returns>
        public static float Float32(double x) => (float)x;

        /// <summary>
        ///     CPython's <c>_float_div_mod</c>: the remainder from <c>fmod</c> moved into the divisor's sign, and the quotient
        ///     <c>(vx - mod) / wx</c> adjusted with it and snapped to the nearest integer.
        /// </summary>
        /// <param name="vx">The dividend.</param>
        /// <param name="wx">The divisor, not zero.</param>
        /// <returns>The floored quotient and the remainder.</returns>
        private static (double Quotient, double Remainder) FloatDivMod(double vx, double wx) {
            double mod = Fmod(vx, wx);
            // fmod is exact, so vx - mod is mathematically a multiple of wx; the division can still leave a tiny error.
            double div = (vx - mod) / wx;
            if (mod != 0.0) {
                // Ensure the remainder has the same sign as the denominator.
                if ((wx < 0) != (mod < 0)) {
                    mod += wx;
                    div -= 1.0;
                }
            } else {
                mod = Math.CopySign(0.0, wx);
            }

            double floordiv;
            if (div != 0.0) {
                // Snap the quotient to the nearest integral value.
                floordiv = Math.Floor(div);
                if (div - floordiv > 0.5)
                    floordiv += 1.0;
            } else {
                // A zero quotient takes the sign of the true quotient.
                floordiv = Math.CopySign(0.0, vx / wx);
            }

            return (floordiv, mod);
        }

        /// <summary>
        ///     C's <c>fmod(x, y)</c> as the reference platform's C runtime (UCRT) computes it: exact for a finite dividend (as
        ///     .NET's <c>%</c> is), and for a non-finite one the NaN UCRT returns, the dividend itself when it is NaN and a
        ///     quiet NaN carrying the infinity's sign otherwise. A NaN divisor is returned as it is.
        /// </summary>
        /// <param name="x">The dividend.</param>
        /// <param name="y">The divisor.</param>
        /// <returns>The truncated remainder.</returns>
        /// <remarks>
        ///     .NET 8's <c>%</c> returns its own sign-set NaN for a non-finite dividend (measured by the rasterizer's parity
        ///     cells), which would flip the sign bit of Python's results; <c>Gym.Rendering.Vector2</c> guards against the same
        ///     difference.
        /// </remarks>
        private static double Fmod(double x, double y) {
            if (double.IsNaN(x))
                return x;
            if (double.IsNaN(y))
                return y;
            if (double.IsInfinity(x))
                return BitConverter.Int64BitsToDouble(x > 0 ? 0x7FF8000000000000L : unchecked((long)0xFFF8000000000000UL));
            return x % y;
        }

        /// <summary>
        ///     CPython's <c>double_round</c> for a finite value: the exact value scaled by <c>10^ndigits</c>, rounded half to
        ///     even to an integer, and parsed back from <c>"&lt;sign&gt;&lt;digits&gt;e&lt;-ndigits&gt;"</c> with correct rounding.
        /// </summary>
        /// <param name="x">A finite value.</param>
        /// <param name="ndigits">The number of decimal places, within CPython's limits.</param>
        /// <returns>The rounded value.</returns>
        /// <exception cref="OverflowException">Thrown when the rounded value is beyond the double range.</exception>
        private static double DoubleRound(double x, int ndigits) {
            // |x| = mantissa * 2^exponent exactly; the sign is applied to the decimal string, as dtoa reports it separately.
            long bits = BitConverter.DoubleToInt64Bits(x);
            bool negative = bits < 0;
            int biased = (int)((bits >> 52) & 0x7FF);
            long fraction = bits & 0xFFFFFFFFFFFFFL;
            BigInteger mantissa = biased == 0 ? fraction : fraction | (1L << 52);
            int exponent = biased == 0 ? -1074 : biased - 1075;

            BigInteger numerator = mantissa, denominator = BigInteger.One;
            if (exponent >= 0)
                numerator <<= exponent;
            else
                denominator <<= -exponent;
            if (ndigits >= 0)
                numerator *= BigInteger.Pow(10, ndigits);
            else
                denominator *= BigInteger.Pow(10, -ndigits);

            // Round half to even on the exact quotient, as dtoa's mode 3 does.
            BigInteger quotient = BigInteger.DivRem(numerator, denominator, out BigInteger remainder);
            int half = (remainder * 2).CompareTo(denominator);
            if (half > 0 || (half == 0 && !quotient.IsEven))
                quotient += BigInteger.One;

            string text = (negative ? "-" : "") + quotient.ToString(CultureInfo.InvariantCulture) + "e" + (-ndigits).ToString(CultureInfo.InvariantCulture);
            double rounded = double.Parse(text, NumberStyles.Float, CultureInfo.InvariantCulture);
            // _Py_dg_strtod reports ERANGE with an infinity when the value overflows; underflow to zero is a valid result.
            if (double.IsInfinity(rounded))
                throw new OverflowException("rounded value too large to represent");
            return rounded;
        }
    }
}
