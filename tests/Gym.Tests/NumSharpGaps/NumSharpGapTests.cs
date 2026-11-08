#nullable enable
using System;
using NumSharp;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests.NumSharpGaps {
    /// <summary>
    ///     The NumSharp gaps Gym.NET depends on, each proven by a test that asserts what NumPy or Python does (decision DR-01,
    ///     docs/plans/DESIGN_REVIEW.md). Every test here is <see cref="OpenBugsAttribute"/>: it fails until NumSharp ships the
    ///     fix, then passes unchanged, and the attribute comes off so it guards the fix.
    /// </summary>
    /// <remarks>
    ///     The items are listed in docs/plans/COMPLETION_PLAN.md §7. A gap that only a live comparison can show (a value
    ///     against NumPy's) lives in the parity suite instead; these are the ones a plain assertion proves.
    /// </remarks>
    [TestClass]
    public class NumSharpGapTests {
        /// <summary>
        ///     N-1: Python's <c>str(AssertionError())</c> is <c>""</c>, so a bare <c>assert</c>'s message is empty. NumSharp
        ///     0.80.0's parameterless constructor uses the type name instead.
        /// </summary>
        [TestMethod]
        [OpenBugs("N-1")]
        public void N1_ParameterlessAssertionError_HasPythonsEmptyMessage() {
            Assert.AreEqual("", new AssertionError().Message);
        }

        /// <summary>
        ///     N-3: NumSharp owns every Python built-in exception Gym.NET throws (decision DR-02). NumSharp 0.80.0 lacks
        ///     <c>NotImplementedError</c> and <c>ImportError</c>, so <c>Gym.Exceptions</c> still defines them.
        /// </summary>
        /// <param name="name">The Python exception name, which NumSharp spells as its class name in namespace NumSharp.</param>
        [TestMethod]
        [OpenBugs("N-3")]
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
        ///     four lack the constructor in NumSharp 0.80.0 (measured from its source).
        /// </summary>
        /// <param name="name">The Python exception name, NumSharp's class name.</param>
        [TestMethod]
        [OpenBugs("N-3")]
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
