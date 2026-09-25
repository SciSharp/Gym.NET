using System;
using System.Numerics;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Utils {
    /// <summary>
    ///     <see cref="PyBuiltins"/> against Python 3.12's own answers (recorded from CPython 3.12.12 and NumPy 2.4.2): the
    ///     float formatting and parsing and the reprs that Gymnasium's messages are built from, so they hold without a Python
    ///     at hand. The live suite (<c>tests/Gym.Tests.Parity</c>) compares the messages themselves.
    /// </summary>
    [TestClass]
    public class PyBuiltinsTests {
        /// <summary><c>repr(float)</c>: shortest digits, positional between 1e-4 and 1e16, scientific with a 2-digit exponent outside.</summary>
        /// <param name="value">The float.</param>
        /// <param name="expected">Python's <c>repr</c>.</param>
        [TestMethod]
        [DataRow(0.1, "0.1")]
        [DataRow(1.0, "1.0")]
        [DataRow(-0.0, "-0.0")]
        [DataRow(123.0, "123.0")]
        [DataRow(0.0001, "0.0001")]
        [DataRow(1e-05, "1e-05")]
        [DataRow(1.5e-05, "1.5e-05")]
        [DataRow(1e15, "1000000000000000.0")]
        [DataRow(1e16, "1e+16")]
        [DataRow(1.5e300, "1.5e+300")]
        [DataRow(5e-324, "5e-324")]
        [DataRow(123456789012345680000.0, "1.2345678901234568e+20")]
        [DataRow(0.1 + 0.2, "0.30000000000000004")]
        [DataRow(double.PositiveInfinity, "inf")]
        [DataRow(double.NegativeInfinity, "-inf")]
        [DataRow(double.NaN, "nan")]
        public void FloatRepr_IsPythons(double value, string expected) {
            Assert.AreEqual(expected, PyBuiltins.FloatRepr(value));
        }

        /// <summary><c>float(str)</c> accepts what Python accepts: whitespace, underscores between digits, infinities and NaN in any case.</summary>
        /// <param name="text">The string.</param>
        /// <param name="expected">Python's value.</param>
        [TestMethod]
        [DataRow("1_000.5", 1000.5)]
        [DataRow(" -Infinity ", double.NegativeInfinity)]
        [DataRow("+inf", double.PositiveInfinity)]
        [DataRow("-0.5", -0.5)]
        [DataRow(".5", 0.5)]
        [DataRow("5.", 5.0)]
        [DataRow("1e-3", 0.001)]
        [DataRow("1E5", 100000.0)]
        [DataRow("1e500", double.PositiveInfinity)]
        [DataRow("\t7\n", 7.0)]
        public void Float_ParsesStringsLikePython(string text, double expected) {
            Assert.AreEqual(expected, PyBuiltins.Float(text));
        }

        /// <summary><c>float("nan")</c> is C's positive quiet NaN and <c>float("-nan")</c> its negation, bit for bit.</summary>
        [TestMethod]
        public void Float_OfNan_KeepsPythonsSign() {
            Assert.AreEqual(0x7FF8000000000000L, BitConverter.DoubleToInt64Bits(PyBuiltins.Float("NaN")));
            Assert.AreEqual(unchecked((long)0xFFF8000000000000UL), BitConverter.DoubleToInt64Bits(PyBuiltins.Float("-nan")));
        }

        /// <summary>Strings Python rejects raise <see cref="ValueError"/> with Python's message, the string in its repr.</summary>
        /// <param name="text">The string.</param>
        /// <param name="expected">Python's message.</param>
        [TestMethod]
        [DataRow("1__0", "could not convert string to float: '1__0'")]
        [DataRow("_1", "could not convert string to float: '_1'")]
        [DataRow("1_", "could not convert string to float: '1_'")]
        [DataRow("abc", "could not convert string to float: 'abc'")]
        [DataRow("1e", "could not convert string to float: '1e'")]
        [DataRow("", "could not convert string to float: ''")]
        [DataRow("0x10", "could not convert string to float: '0x10'")]
        [DataRow("it's", "could not convert string to float: \"it's\"")]
        public void Float_RejectsStringsLikePython(string text, string expected) {
            Assert.AreEqual(expected, Assert.ThrowsExactly<ValueError>(() => PyBuiltins.Float(text)).Message);
        }

        /// <summary>Numbers convert as Python converts them; <c>None</c>, arrays with dimensions and huge ints fail with Python's errors.</summary>
        [TestMethod]
        public void Float_ConvertsNumbers_AndRejectsTheRest() {
            Assert.AreEqual(1.0, PyBuiltins.Float(true));
            Assert.AreEqual(-3.0, PyBuiltins.Float(-3L));
            Assert.AreEqual(0.5, PyBuiltins.Float(0.5f));
            Assert.AreEqual(9007199254740992.0, PyBuiltins.Float(BigInteger.Pow(2, 53) + 1), "Round half to even, like Python's int -> float.");
            using (NDArray scalar = NDArray.Scalar<double>(0.25))
                Assert.AreEqual(0.25, PyBuiltins.Float(scalar));
            Assert.AreEqual("float() argument must be a string or a real number, not 'NoneType'", Assert.ThrowsExactly<TypeError>(() => PyBuiltins.Float(null)).Message);
            using (NDArray vector = np.array(new[] { 0.1 }))
                Assert.AreEqual("only 0-dimensional arrays can be converted to Python scalars", Assert.ThrowsExactly<TypeError>(() => PyBuiltins.Float(vector)).Message);
            Assert.AreEqual("int too large to convert to float", Assert.ThrowsExactly<OverflowException>(() => PyBuiltins.Float(BigInteger.Pow(10, 400))).Message);
        }

        /// <summary><c>int(float)</c> truncates toward zero and fails on NaN and infinities with Python's messages.</summary>
        [TestMethod]
        public void Int_TruncatesLikePython() {
            Assert.AreEqual(new BigInteger(-107), PyBuiltins.Int(-107.9));
            Assert.AreEqual(BigInteger.Parse("10000000000000000000000"), PyBuiltins.Int(1e22));
            Assert.AreEqual("cannot convert float NaN to integer", Assert.ThrowsExactly<ValueError>(() => PyBuiltins.Int(double.NaN)).Message);
            Assert.AreEqual("cannot convert float infinity to integer", Assert.ThrowsExactly<OverflowException>(() => PyBuiltins.Int(double.NegativeInfinity)).Message);
        }

        /// <summary><c>repr(x)</c> and <c>str(type(x))</c> of the Python counterparts of .NET values, as Gymnasium's assert messages print them.</summary>
        [TestMethod]
        public void Repr_AndTypeRepr_ArePythons() {
            Assert.AreEqual("2 (<class 'int'>)", Describe(2L));
            Assert.AreEqual("0.5 (<class 'float'>)", Describe(0.5));
            Assert.AreEqual("True (<class 'bool'>)", Describe(true));
            Assert.AreEqual("None (<class 'NoneType'>)", Describe(null));
            Assert.AreEqual("'1' (<class 'str'>)", Describe("1"));
            Assert.AreEqual("np.float32(1.0) (<class 'numpy.float32'>)", Describe(1.0f));
            Assert.AreEqual("1180591620717411303424 (<class 'int'>)", Describe(BigInteger.Pow(2, 70)));
            using (NDArray int64 = NDArray.Scalar<long>(5L))
                Assert.AreEqual("np.int64(5) (<class 'numpy.int64'>)", Describe(int64));
            using (NDArray float64 = NDArray.Scalar<double>(1e16))
                Assert.AreEqual("np.float64(1e+16) (<class 'numpy.float64'>)", Describe(float64));
            using (NDArray flag = NDArray.Scalar<bool>(true))
                Assert.AreEqual("np.True_ (<class 'numpy.bool'>)", Describe(flag));
            using (NDArray vector = np.array(new[] { 1L }))
                Assert.AreEqual("array([1]) (<class 'numpy.ndarray'>)", Describe(vector));
        }

        /// <summary><c>repr(str)</c> quotes and escapes like Python.</summary>
        [TestMethod]
        public void StrRepr_QuotesAndEscapesLikePython() {
            Assert.AreEqual("'abc'", PyBuiltins.StrRepr("abc"));
            Assert.AreEqual("\"it's\"", PyBuiltins.StrRepr("it's"));
            Assert.AreEqual("'say \"hi\" it\\'s'", PyBuiltins.StrRepr("say \"hi\" it's"));
            Assert.AreEqual("'a\\\\b\\n\\t\\x00'", PyBuiltins.StrRepr("a\\b\n\t\0"));
            Assert.AreEqual("'é\\x85'", PyBuiltins.StrRepr("é\u0085"));
        }

        /// <summary>An f-string's <c>{x}</c> is <c>str(x)</c>: strings bare, floats and <c>None</c> as <c>repr</c> prints them.</summary>
        [TestMethod]
        public void Str_IsPythons() {
            Assert.AreEqual("abc", PyBuiltins.Str("abc"));
            Assert.AreEqual("None", PyBuiltins.Str(null));
            Assert.AreEqual("1e-05", PyBuiltins.Str(1e-5));
            using NDArray vector = np.array(new[] { 0.1 });
            Assert.AreEqual("[0.1]", PyBuiltins.Str(vector));
        }

        /// <summary>The <c>{x!r} ({type(x)})</c> pair CartPole's assert prints.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The rendered pair.</returns>
        private static string Describe(object value) => $"{PyBuiltins.Repr(value)} ({PyBuiltins.TypeRepr(value)})";
    }
}
