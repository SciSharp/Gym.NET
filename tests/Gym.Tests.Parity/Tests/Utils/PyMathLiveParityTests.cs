using System;
using System.Linq;
using System.Numerics;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Session;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Python.Runtime;

namespace Gym.Tests.Parity.Tests.Utils {
    /// <summary>
    ///     <see cref="PyMath"/> (F3) against the embedded CPython 3.12 and NumPy 2.4.2, bit for bit: <c>//</c>, <c>%</c> and
    ///     <c>divmod</c> on ints and floats, <c>min</c>/<c>max</c>, <c>round</c> with and without digits, <c>np.clip</c> and
    ///     <c>np.sign</c>, over corpora of edge values (signed zeros, subnormals, the largest double, infinities, NaN of
    ///     both signs), each result's IEEE bits or the error's type and message compared.
    /// </summary>
    /// <remarks>
    ///     Floats cross into Python as <c>PyFloat</c>, which keeps every bit (NaN payloads and signs included). One case is
    ///     left out on purpose: <c>long.MinValue // -1</c>, which Python answers with 2^63 and <see langword="long"/> cannot
    ///     hold (<see cref="PyMath"/> throws <see cref="OverflowException"/> there, as port plan §3.2 asks).
    /// </remarks>
    [TestClass]
    public sealed class PyMathLiveParityTests : ParityTestBase {
        /// <summary>The float corpus: zeros, units, fractions, extremes, infinities, and NaN of both signs.</summary>
        private static readonly double[] Floats = {
            0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 0.1, -7.5, 7.5, 3.0, 1e300, -1e-300, 5e-324, -5e-324, 1.7976931348623157e308, -1.7976931348623157e308,
            double.PositiveInfinity, double.NegativeInfinity, double.NaN, BitConverter.Int64BitsToDouble(unchecked((long)0xFFF8000000000000UL)),
        };

        /// <summary>The integer corpus, the edges of <see langword="long"/> included.</summary>
        private static readonly long[] Integers = { 0, 1, -1, 2, -2, 3, 7, -7, 10, -10, long.MaxValue, long.MinValue + 1, long.MinValue };

        /// <summary><c>a // b</c>, <c>a % b</c> and <c>divmod(a, b)</c> for every pair of the integer corpus.</summary>
        [TestMethod]
        public void IntegerDivision_MatchesPython() {
            foreach (long a in Integers)
            foreach (long b in Integers) {
                if (a == long.MinValue && b == -1)
                    continue;
                ByteContract.AssertSame($"{a} // {b}", Python2(a, b, Python.@operator.floordiv), Outcome.Capture(() => PyMath.FloorDiv(a, b)));
                ByteContract.AssertSame($"{a} % {b}", Python2(a, b, Python.@operator.mod), Outcome.Capture(() => PyMath.Mod(a, b)));
                ByteContract.AssertSame($"divmod({a}, {b})", Python2(a, b, Python.builtins.divmod), Outcome.Capture(() => PyMath.DivMod(a, b)));
            }
        }

        /// <summary><c>a // b</c>, <c>a % b</c> and <c>divmod(a, b)</c> for every pair of the float corpus, NaN bits included.</summary>
        [TestMethod]
        public void FloatDivision_MatchesPython() {
            foreach (double a in Floats)
            foreach (double b in Floats) {
                string pair = $"{Describe(a)}, {Describe(b)}";
                ByteContract.AssertSame($"floordiv({pair})", Python2(a, b, Python.@operator.floordiv), Outcome.Capture(() => PyMath.FloorDiv(a, b)));
                ByteContract.AssertSame($"mod({pair})", Python2(a, b, Python.@operator.mod), Outcome.Capture(() => PyMath.Mod(a, b)));
                ByteContract.AssertSame($"divmod({pair})", Python2(a, b, Python.builtins.divmod), Outcome.Capture(() => PyMath.DivMod(a, b)));
            }
        }

        /// <summary><c>min(a, b)</c> and <c>max(a, b)</c> for every pair: the first argument wins ties and NaN comparisons.</summary>
        [TestMethod]
        public void MinMax_MatchPython() {
            foreach (double a in Floats)
            foreach (double b in Floats) {
                string pair = $"{Describe(a)}, {Describe(b)}";
                ByteContract.AssertSame($"min({pair})", Python2(a, b, Python.builtins.min), Outcome.Capture(() => PyMath.Min(a, b)));
                ByteContract.AssertSame($"max({pair})", Python2(a, b, Python.builtins.max), Outcome.Capture(() => PyMath.Max(a, b)));
            }

            foreach (long a in Integers)
            foreach (long b in Integers) {
                ByteContract.AssertSame($"min({a}, {b})", Python2(a, b, Python.builtins.min), Outcome.Capture(() => PyMath.Min(a, b)));
                ByteContract.AssertSame($"max({a}, {b})", Python2(a, b, Python.builtins.max), Outcome.Capture(() => PyMath.Max(a, b)));
            }
        }

        /// <summary><c>round(x)</c> for the float corpus and exact halves: ties to even, Python's errors for NaN and infinity.</summary>
        [TestMethod]
        public void RoundToInteger_MatchesPython() {
            foreach (double x in Floats.Concat(new[] { 0.5, 1.5, 2.5, -0.5, -2.5, 0.49999999999999994, 4503599627370497.0, 1e16 + 2 })) {
                ByteContract.AssertSame($"round({Describe(x)})", Outcome.CapturePython(() => {
                    using PyObject value = PyConvert.ToPython(x);
                    return Python.builtins.round(value);
                }), Outcome.Capture(() => PyMath.Round(x)));
            }
        }

        /// <summary>
        ///     <c>round(x, ndigits)</c> over values that stress decimal rounding (stored just below a tie, exact ties, huge and
        ///     tiny magnitudes) and digit counts beyond both of CPython's limits, the overflow error included.
        /// </summary>
        [TestMethod]
        public void RoundToDigits_MatchesPython() {
            double[] values = Floats.Concat(new[] { 2.675, 0.125, 0.375, 1234.5678, -0.0001, 5.0, 15.0, 25.0, 1e308, 123456789.987654321, 0.285, 1.005, 2.5e-310 }).ToArray();
            int[] digits = { -400, -309, -308, -300, -10, -2, -1, 0, 1, 2, 3, 10, 17, 300, 323, 324, 400 };
            foreach (double x in values)
            foreach (int n in digits) {
                ByteContract.AssertSame($"round({Describe(x)}, {n})", Outcome.CapturePython(() => {
                    using PyObject value = PyConvert.ToPython(x);
                    using PyObject places = PyConvert.ToPython((long)n);
                    return Python.builtins.round(value, places);
                }), Outcome.Capture(() => PyMath.Round(x, n)));
            }
        }

        /// <summary><c>np.clip(x, low, high)</c> for every triple of an 8-value corpus (512 cases), and <c>np.sign(x)</c> for the corpus.</summary>
        [TestMethod]
        public void ClipAndSign_MatchNumpy() {
            double[] corpus = { 0.0, -0.0, 1.0, -1.0, 0.5, double.NaN, double.PositiveInfinity, double.NegativeInfinity };
            foreach (double x in corpus)
            foreach (double low in corpus)
            foreach (double high in corpus) {
                ByteContract.AssertSame($"np.clip({Describe(x)}, {Describe(low)}, {Describe(high)})", Outcome.CapturePython(() => {
                    using PyObject a = PyConvert.ToPython(x);
                    using PyObject aMin = PyConvert.ToPython(low);
                    using PyObject aMax = PyConvert.ToPython(high);
                    return Python.numpy.clip(a, aMin, aMax);
                }), Outcome.Capture(() => PyMath.Clip(x, low, high)));
            }

            foreach (double x in Floats) {
                ByteContract.AssertSame($"np.sign({Describe(x)})", Outcome.CapturePython(() => {
                    using PyObject value = PyConvert.ToPython(x);
                    return Python.numpy.sign(value);
                }), Outcome.Capture(() => PyMath.Sign(x)));
            }
        }

        /// <summary>Runs a two-argument Python function on two CLR values converted to Python, captured.</summary>
        /// <param name="a">The first argument.</param>
        /// <param name="b">The second argument.</param>
        /// <param name="function">The facade row.</param>
        /// <returns>The outcome.</returns>
        private static Outcome Python2(object a, object b, Func<PyObject, PyObject, PyObject> function) {
            return Outcome.CapturePython(() => {
                using PyObject left = PyConvert.ToPython(a);
                using PyObject right = PyConvert.ToPython(b);
                return function(left, right);
            });
        }

        /// <summary>A double as Python prints it, with the NaN's sign spelled out (Python prints both as <c>nan</c>).</summary>
        /// <param name="value">The value.</param>
        /// <returns>The text.</returns>
        private static string Describe(double value) {
            return double.IsNaN(value) && BitConverter.DoubleToInt64Bits(value) < 0 ? "-nan" : PyBuiltins.FloatRepr(value);
        }
    }
}
