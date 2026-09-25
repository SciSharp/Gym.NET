#nullable enable
using System;
using System.Numerics;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Utils {
    /// <summary>
    ///     Pins <see cref="PyMath"/> (F3) to values and messages taken from CPython 3.12.12 and NumPy 2.4.2 (the parity stack):
    ///     floor division and modulo with the divisor's sign, argument-order <c>min</c>/<c>max</c>, exact <c>round</c>, and
    ///     NumPy's <c>clip</c>/<c>sign</c>, signed zeros and NaN bit patterns included. The live parity suite compares a
    ///     larger corpus with the embedded interpreter.
    /// </summary>
    [TestClass]
    public class PyMathTests {
        /// <summary>Integer <c>//</c>, <c>%</c> and <c>divmod</c>, and their ZeroDivisionError messages.</summary>
        [TestMethod]
        public void IntegerDivision_FollowsTheDivisorsSign() {
            Assert.AreEqual(-4L, PyMath.FloorDiv(-7L, 2L));
            Assert.AreEqual(1L, PyMath.Mod(-7L, 2L));
            Assert.AreEqual(-1L, PyMath.Mod(7L, -2L));
            Assert.AreEqual((-4L, 1L), PyMath.DivMod(-7L, 2L));
            Assert.AreEqual(0L, PyMath.Mod(long.MinValue, -1L));
            Assert.AreEqual("integer division or modulo by zero", Assert.ThrowsExactly<DivideByZeroException>(() => PyMath.FloorDiv(1L, 0L)).Message);
            Assert.AreEqual("integer modulo by zero", Assert.ThrowsExactly<DivideByZeroException>(() => PyMath.Mod(1L, 0L)).Message);
            Assert.AreEqual("integer division or modulo by zero", Assert.ThrowsExactly<DivideByZeroException>(() => PyMath.DivMod(1L, 0L)).Message);
            Assert.ThrowsExactly<OverflowException>(() => PyMath.FloorDiv(long.MinValue, -1L));
        }

        /// <summary>Float <c>//</c> and <c>%</c> as CPython's <c>floatobject.c</c> computes them, signed zeros and infinities included.</summary>
        [TestMethod]
        public void FloatDivision_IsCPythons() {
            Assert.AreEqual(-4.0, PyMath.FloorDiv(-7.5, 2.0));
            Assert.AreEqual(0.5, PyMath.Mod(-7.5, 2.0));
            Assert.AreEqual(-0.5, PyMath.Mod(7.5, -2.0));
            AssertBits(-0.0, PyMath.Mod(0.0, -2.0));
            AssertBits(0.0, PyMath.Mod(-0.0, 2.0));
            Assert.AreEqual(1e300, PyMath.Mod(-1e-300, 1e300));
            AssertBits(0.0, PyMath.FloorDiv(5.0, double.PositiveInfinity));
            Assert.AreEqual(-1.0, PyMath.FloorDiv(-5.0, double.PositiveInfinity));
            Assert.AreEqual(5.0, PyMath.Mod(5.0, double.PositiveInfinity));
            Assert.AreEqual(double.PositiveInfinity, PyMath.Mod(-5.0, double.PositiveInfinity));
            Assert.AreEqual(9.0, PyMath.FloorDiv(1.0, 0.1), "1 // 0.1 is 9.0, although 1 / 0.1 rounds to 10.");
            AssertBits(BitConverter.Int64BitsToDouble(0x7FF8000000000000L), PyMath.Mod(double.PositiveInfinity, 2.0));
            AssertBits(BitConverter.Int64BitsToDouble(unchecked((long)0xFFF8000000000000UL)), PyMath.Mod(double.NegativeInfinity, 2.0));
            AssertBits(BitConverter.Int64BitsToDouble(unchecked((long)0xFFF8000000000000UL)), PyMath.FloorDiv(double.NegativeInfinity, 2.0));
            Assert.AreEqual("float floor division by zero", Assert.ThrowsExactly<DivideByZeroException>(() => PyMath.FloorDiv(1.0, 0.0)).Message);
            Assert.AreEqual("float modulo", Assert.ThrowsExactly<DivideByZeroException>(() => PyMath.Mod(1.0, -0.0)).Message);
            Assert.AreEqual("float divmod()", Assert.ThrowsExactly<DivideByZeroException>(() => PyMath.DivMod(1.0, 0.0)).Message);
        }

        /// <summary>Python's <c>min</c>/<c>max</c> keep the first argument on ties and on every comparison with NaN.</summary>
        [TestMethod]
        public void MinMax_DependOnArgumentOrder() {
            Assert.IsTrue(double.IsNaN(PyMath.Min(double.NaN, 1.0)));
            Assert.AreEqual(1.0, PyMath.Min(1.0, double.NaN));
            Assert.IsTrue(double.IsNaN(PyMath.Max(double.NaN, 1.0)));
            Assert.AreEqual(1.0, PyMath.Max(1.0, double.NaN));
            AssertBits(0.0, PyMath.Max(0.0, -0.0));
            AssertBits(0.0, PyMath.Min(0.0, -0.0));
            AssertBits(-0.0, PyMath.Max(-0.0, 0.0));
            Assert.AreEqual(-3L, PyMath.Min(new[] { 4L, -3L, 7L }));
            Assert.AreEqual(7L, PyMath.Max(new[] { 4L, -3L, 7L }));
            Assert.AreEqual("min() iterable argument is empty", Assert.ThrowsExactly<ValueError>(() => PyMath.Min(Array.Empty<double>())).Message);
            Assert.AreEqual("max() iterable argument is empty", Assert.ThrowsExactly<ValueError>(() => PyMath.Max(Array.Empty<double>())).Message);
        }

        /// <summary><c>round(x)</c> ties to even into an int; NaN and infinity raise Python's errors.</summary>
        [TestMethod]
        public void Round_ToInteger_TiesToEven() {
            Assert.AreEqual(BigInteger.Zero, PyMath.Round(0.5));
            Assert.AreEqual(new BigInteger(2), PyMath.Round(1.5));
            Assert.AreEqual(new BigInteger(2), PyMath.Round(2.5));
            Assert.AreEqual(BigInteger.Zero, PyMath.Round(-0.5));
            Assert.AreEqual(new BigInteger(1e300), PyMath.Round(1e300));
            Assert.AreEqual("cannot convert float NaN to integer", Assert.ThrowsExactly<ValueError>(() => PyMath.Round(double.NaN)).Message);
            Assert.AreEqual("cannot convert float infinity to integer", Assert.ThrowsExactly<OverflowException>(() => PyMath.Round(double.PositiveInfinity)).Message);
        }

        /// <summary><c>round(x, n)</c> rounds the exact binary value, ties to even, and keeps the sign of a zero.</summary>
        [TestMethod]
        public void Round_ToDigits_IsExact() {
            Assert.AreEqual(2.67, PyMath.Round(2.675, 2), "2.675 is stored just below 2.675.");
            Assert.AreEqual(0.12, PyMath.Round(0.125, 2), "0.125 is exact, so the tie goes to even.");
            Assert.AreEqual(0.38, PyMath.Round(0.375, 2));
            AssertBits(-0.0, PyMath.Round(-0.0001, 2));
            AssertBits(0.0, PyMath.Round(5.0, -1));
            Assert.AreEqual(20.0, PyMath.Round(15.0, -1));
            Assert.AreEqual(1e308, PyMath.Round(1e308, -308));
            Assert.AreEqual(1.5, PyMath.Round(1.5, 400), "Beyond NDIGITS_MAX a value rounds to itself.");
            AssertBits(-0.0, PyMath.Round(-1.5, -400));
            Assert.IsTrue(double.IsNaN(PyMath.Round(double.NaN, 2)));
            Assert.AreEqual(double.NegativeInfinity, PyMath.Round(double.NegativeInfinity, 2));
            Assert.AreEqual("rounded value too large to represent", Assert.ThrowsExactly<OverflowException>(() => PyMath.Round(1.7976931348623157e308, -308)).Message);
        }

        /// <summary>
        ///     <c>np.clip</c> keeps the value it holds on ties and propagates NaN from any operand; the textbook
        ///     <c>min(max(x, low), high)</c> gets the signed zeros wrong.
        /// </summary>
        [TestMethod]
        public void Clip_IsNumpys() {
            AssertBits(0.0, PyMath.Clip(0.0, -0.0, 1.0));
            AssertBits(-0.0, PyMath.Clip(-0.0, 0.0, 1.0));
            Assert.AreEqual(1.0, PyMath.Clip(2.0, 0.0, 1.0));
            AssertBits(0.0, PyMath.Clip(-1.0, 0.0, 1.0));
            AssertBits(0.0, PyMath.Clip(0.5, 1.0, 0.0), "low > high gives high.");
            Assert.IsTrue(double.IsNaN(PyMath.Clip(double.NaN, 0.0, 1.0)));
            Assert.IsTrue(double.IsNaN(PyMath.Clip(0.5, double.NaN, 1.0)));
            Assert.IsTrue(double.IsNaN(PyMath.Clip(0.5, 0.0, double.NaN)));
            Assert.AreEqual(1f, PyMath.Clip(3f, -1f, 1f));
        }

        /// <summary><c>np.sign</c>: a zero of either sign gives positive zero, NaN passes through with its bits.</summary>
        [TestMethod]
        public void Sign_IsNumpys() {
            AssertBits(0.0, PyMath.Sign(-0.0));
            Assert.AreEqual(-1.0, PyMath.Sign(-3.0));
            Assert.AreEqual(1.0, PyMath.Sign(double.PositiveInfinity));
            double negativeNan = BitConverter.Int64BitsToDouble(unchecked((long)0xFFF8000000000000UL));
            AssertBits(negativeNan, PyMath.Sign(negativeNan));
            Assert.AreEqual(-1L, PyMath.Sign(-9L));
            Assert.AreEqual(0f, PyMath.Sign(0f));
            Assert.AreEqual(0.1f, PyMath.Float32(0.1));
        }

        /// <summary>Asserts two doubles are bit-for-bit identical (signed zeros and NaN payloads included).</summary>
        /// <param name="expected">The expected value.</param>
        /// <param name="actual">The actual value.</param>
        /// <param name="message">What is compared.</param>
        private static void AssertBits(double expected, double actual, string message = "") {
            Assert.AreEqual(BitConverter.DoubleToInt64Bits(expected).ToString("X16"), BitConverter.DoubleToInt64Bits(actual).ToString("X16"), message);
        }
    }
}
