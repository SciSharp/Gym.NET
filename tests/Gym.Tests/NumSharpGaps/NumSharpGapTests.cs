#nullable enable
using System;
using System.Linq;
using System.Reflection;
using NumSharp;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests.NumSharpGaps {
    /// <summary>
    ///     The NumSharp gaps Gym.NET depends on, each proven by a test that asserts what NumPy or Python does (decision DR-01,
    ///     docs/plans/DESIGN_REVIEW.md). Each test was <see cref="OpenBugsAttribute"/> while it failed; NumSharp 0.81.0 shipped
    ///     the fixes for N-1 to N-4, the tests passed unchanged, and the attributes came off, so each now guards its fix
    ///     (DR-76).
    /// </summary>
    /// <remarks>
    ///     The items are listed in docs/plans/COMPLETION_PLAN.md §7. A gap that only a live comparison can show (a value
    ///     against NumPy's) lives in the parity suite instead; these are the ones a plain assertion proves.
    /// </remarks>
    [TestClass]
    public class NumSharpGapTests {
        /// <summary>
        ///     N-1: Python's <c>str(AssertionError())</c> is <c>""</c>, so a bare <c>assert</c>'s message is empty. NumSharp
        ///     0.80.0's parameterless constructor used the type name instead; 0.81.0 has Python's text.
        /// </summary>
        [TestMethod]
        public void N1_ParameterlessAssertionError_HasPythonsEmptyMessage() {
            Assert.AreEqual("", new AssertionError().Message);
        }

        /// <summary>
        ///     N-2: NumPy's <c>'c'</c> code splits each string into its characters: <c>np.asarray(["SFFF", "FHFH", "FFFH",
        ///     "HFFG"], dtype="c")</c> is a <c>(4, 4)</c> <c>|S1</c> array whose bytes are the map's letters. NumSharp 0.80.0
        ///     kept one element per string, so <c>ToyTextUtils.CharMap</c> builds the toy_text maps from a two-dimensional
        ///     grid; 0.81.0 splits the strings itself, which this test guards.
        /// </summary>
        [TestMethod]
        public void N2_CharCode_SplitsStringsIntoCharacters() {
            string[] rows = { "SFFF", "FHFH", "FFFH", "HFFG" };
            using NDArray map = np.array(rows, np.dtype("c"));
            CollectionAssert.AreEqual(new long[] { 4, 4 }, map.shape, "NumPy's shape is (rows, characters).");
            Assert.AreEqual("|S1", map.dtype.str);
            CollectionAssert.AreEqual(System.Text.Encoding.ASCII.GetBytes(string.Concat(rows)), map.tobytes(), "One byte per character, row by row.");
        }

        /// <summary>
        ///     N-3: NumSharp owns every Python built-in exception Gym.NET throws (decision DR-02). NumSharp 0.80.0 lacked
        ///     <c>NotImplementedError</c> and <c>ImportError</c>, so <c>Gym.Exceptions</c> defined stop-gaps; 0.81.0 has every
        ///     Python exception, and the stop-gaps are gone.
        /// </summary>
        /// <param name="name">The Python exception name, which NumSharp spells as its class name in namespace NumSharp.</param>
        [TestMethod]
        [DataRow("NotImplementedError")]
        [DataRow("ImportError")]
        public void N3_NumSharp_DefinesThePythonException(string name) {
            Type? type = typeof(NDArray).Assembly.GetType("NumSharp." + name);
            Assert.IsNotNull(type, $"NumSharp has no {name}.");
            Assert.IsTrue(typeof(Exception).IsAssignableFrom(type), $"NumSharp.{name} is not an exception.");
        }

        /// <summary>
        ///     N-3: every NumSharp Python exception takes an inner exception, so a port can keep Python's
        ///     <c>raise ... from e</c> chain (found in P1: <c>get_wrapper_attr</c> loses it with <c>AttributeError</c>). These
        ///     four lacked the constructor in NumSharp 0.80.0 (measured from its source); 0.81.0 has it.
        /// </summary>
        /// <param name="name">The Python exception name, NumSharp's class name.</param>
        [TestMethod]
        [DataRow("AttributeError")]
        [DataRow("KeyError")]
        [DataRow("RuntimeError")]
        [DataRow("NameError")]
        public void N3_NumSharpException_TakesAnInnerException(string name) {
            AssertTakesAnInnerException(name);
        }

        /// <summary>
        ///     The NumSharp exceptions that already take an inner exception keep doing so, guarding the part of N-3 that
        ///     NumSharp 0.80.0 already provides.
        /// </summary>
        /// <param name="name">The Python exception name, NumSharp's class name.</param>
        [TestMethod]
        [DataRow("AssertionError")]
        [DataRow("ValueError")]
        [DataRow("TypeError")]
        [DataRow("IndexError")]
        public void NumSharpException_KeepsItsInnerExceptionConstructor(string name) {
            AssertTakesAnInnerException(name);
        }

        /// <summary>
        ///     N-4: NumPy's <c>np.linspace</c> takes array bounds and computes in their result dtype, so
        ///     <c>np.linspace(np.float32(0.1), np.float32(0.7), 11)</c> is float32 arithmetic, and four of its eleven samples
        ///     differ from the float64 samples cast to float32. NumSharp 0.80.0 had only scalar overloads, whose C#
        ///     <see cref="float"/> stands for a Python float (float64), so the discretizing wrappers computed NumPy's edges
        ///     through a workaround; 0.81.0 takes array ends and computes float32 ends in float32, and the wrappers call
        ///     <c>np.linspace</c> themselves.
        /// </summary>
        /// <exception cref="AssertFailedException">Thrown when NumSharp returns other bits than NumPy's.</exception>
        [TestMethod]
        public void N4_Linspace_TakesArrayBoundsAndKeepsTheirDtype() {
            MethodInfo? linspace = typeof(np).GetMethods(BindingFlags.Public | BindingFlags.Static)
                .FirstOrDefault(method => method.Name == "linspace" && method.GetParameters() is { Length: >= 3 } parameters &&
                                          parameters[0].ParameterType == typeof(NDArray) && parameters[1].ParameterType == typeof(NDArray));
            Assert.IsNotNull(linspace, "NumSharp has no np.linspace(NDArray start, NDArray stop, num, ...).");

            using NDArray start = NDArray.Scalar(0.1f), stop = NDArray.Scalar(0.7f);
            ParameterInfo[] declared = linspace!.GetParameters();
            // num converts to whichever integer width the overload takes; the remaining parameters keep their defaults.
            object?[] arguments = declared.Select((parameter, i) => i switch {
                0 => start,
                1 => stop,
                2 => Convert.ChangeType(11, parameter.ParameterType),
                _ => parameter.HasDefaultValue ? parameter.DefaultValue : Type.Missing,
            }).ToArray();
            using var samples = (NDArray)linspace.Invoke(null, arguments)!;

            Assert.AreEqual(NPTypeCode.Single, samples.typecode, "NumPy keeps the float32 bounds' dtype.");
            // np.linspace(np.float32(0.1), np.float32(0.7), 11).view(np.uint32), NumPy 2.4.2.
            uint[] numpyBits = { 1036831949, 1042536202, 1046562734, 1049582632, 1051595898, 1053609164, 1055622429, 1057300152, 1058306785, 1059313418, 1060320051 };
            CollectionAssert.AreEqual(numpyBits, samples.ToArray<float>().Select(BitConverter.SingleToUInt32Bits).ToArray(), "NumPy's float32 arithmetic, bit for bit.");
        }

        /// <summary>Asserts that NumSharp defines <paramref name="name"/> with a public <c>(string, Exception)</c> constructor.</summary>
        /// <param name="name">The Python exception name, NumSharp's class name in namespace NumSharp.</param>
        /// <exception cref="AssertFailedException">Thrown when the type or the constructor is missing.</exception>
        private static void AssertTakesAnInnerException(string name) {
            Type? type = typeof(NDArray).Assembly.GetType("NumSharp." + name);
            Assert.IsNotNull(type, $"NumSharp has no {name}.");
            Assert.IsNotNull(type!.GetConstructor(new[] { typeof(string), typeof(Exception) }),
                $"NumSharp.{name} has no public (string message, Exception innerException) constructor.");
        }
    }
}
