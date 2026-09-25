#nullable enable
using System;

namespace Gym.Exceptions {
    /// <summary>
    ///     A failed assertion, Python's <c>AssertionError</c>: what the envs throw where Gymnasium's asserts, such as CartPole's
    ///     check that an action belongs to its action space. Unlike <c>Debug.Assert</c>, it is thrown in Release builds too,
    ///     so an invalid action can never be silently treated as a valid one.
    /// </summary>
    /// <remarks>
    ///     NumSharp provides the Python built-ins that NumPy raises (<c>ValueError</c>, <c>TypeError</c>, ...) and Gym.NET
    ///     reuses them (port plan decision D17); NumSharp has no <c>AssertionError</c>, so it lives here. Messages are the
    ///     assert's f-string rendered with Python's formatting (see <see cref="Gym.Utils.PyBuiltins"/>), because callers and
    ///     tests match on them verbatim. Reference: <see href="https://docs.python.org/3.12/library/exceptions.html#AssertionError">AssertionError</see>.
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
