#nullable enable
using System;

namespace Gym.Exceptions {
    /// <summary>
    ///     Port of Python's built-in <c>AssertionError</c>: what a Gymnasium <c>assert</c> raises, such as CartPole's
    ///     <c>assert self.action_space.contains(action)</c>. Unlike <c>Debug.Assert</c>, it is thrown in Release builds too, so
    ///     an invalid action can never be silently treated as a valid one.
    /// </summary>
    /// <remarks>
    ///     NumSharp ports Python's built-ins that NumPy raises (<c>ValueError</c>, <c>TypeError</c>, ...) and Gym.NET reuses
    ///     them (port plan decision D17); NumSharp has no <c>AssertionError</c>, so it lives here. Messages are the Python
    ///     source's f-strings rendered with Python's formatting (see <see cref="Gym.Utils.PyBuiltins"/>), because callers and
    ///     ported tests match on them verbatim.
    /// </remarks>
    public class AssertionError : Exception {
        /// <summary>
        ///     An assertion failure without a message (a bare <c>assert condition</c>). Its <see cref="Exception.Message"/> is
        ///     empty, as Python's <c>str(AssertionError())</c> is.
        /// </summary>
        public AssertionError() : base("") { }

        /// <summary>An assertion failure with the message the Python <c>assert</c> statement carries.</summary>
        /// <param name="message">The message, rendered exactly as Python renders the assert's f-string.</param>
        public AssertionError(string message) : base(message) { }

        /// <summary>An assertion failure with a message and the exception that caused it.</summary>
        /// <param name="message">The message, rendered exactly as Python renders the assert's f-string.</param>
        /// <param name="innerException">The underlying cause.</param>
        public AssertionError(string message, Exception innerException) : base(message, innerException) { }
    }
}
