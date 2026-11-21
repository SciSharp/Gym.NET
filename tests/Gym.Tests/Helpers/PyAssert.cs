#nullable enable
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests.Helpers {
    /// <summary>
    ///     Assertions under Python's <c>==</c> (<see cref="PyEquality"/>), for the tests that compare dicts as Gymnasium's
    ///     tests do (<c>assert info == {"count": 0}</c>): since DR-92 an <c>Info</c>'s <see cref="object.Equals(object)"/> is
    ///     reference equality, which <see cref="Assert.AreEqual(object, object)"/> would use.
    /// </summary>
    internal static class PyAssert {
        /// <summary>Asserts that two values are equal under Python's <c>==</c>, printing both with Python's <c>repr</c> when they differ.</summary>
        /// <param name="expected">The expected value.</param>
        /// <param name="actual">The actual value.</param>
        /// <param name="message">A message for the failure, or <see langword="null"/>.</param>
        /// <exception cref="AssertFailedException">Thrown when the values differ.</exception>
        public static void AreEqual(object? expected, object? actual, string? message = null) {
            if (!PyEquality.AreEqual(expected, actual))
                throw new AssertFailedException($"PyAssert.AreEqual failed. Expected:<{PyBuiltins.Repr(expected)}>. Actual:<{PyBuiltins.Repr(actual)}>. {message}");
        }

        /// <summary>Asserts that two values differ under Python's <c>==</c>.</summary>
        /// <param name="notExpected">The value the actual one must differ from.</param>
        /// <param name="actual">The actual value.</param>
        /// <param name="message">A message for the failure, or <see langword="null"/>.</param>
        /// <exception cref="AssertFailedException">Thrown when the values are equal.</exception>
        public static void AreNotEqual(object? notExpected, object? actual, string? message = null) {
            if (PyEquality.AreEqual(notExpected, actual))
                throw new AssertFailedException($"PyAssert.AreNotEqual failed. Both:<{PyBuiltins.Repr(actual)}>. {message}");
        }
    }
}
