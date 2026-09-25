using System;

namespace Gym.Exceptions {
    /// <summary>
    ///     Port of <c>gymnasium.error.Error</c> (<c>gymnasium/error.py:4</c> @ 9e04324), the superclass of every error that
    ///     Gymnasium raises itself (as opposed to Python built-ins such as <c>ValueError</c>, which Gym.NET takes from
    ///     NumSharp, decision D17). Catch it to handle any Gymnasium-defined failure, for example an invalid seed.
    /// </summary>
    /// <remarks>
    ///     Only the superclass is ported so far; the subclasses of <c>error.py</c> arrive with P1. Gym.NET's older
    ///     exceptions in this namespace (<see cref="InvalidActionError"/>, <see cref="AlreadySteppingError"/>,
    ///     <see cref="NotSteppingError"/>) do not derive from it yet.
    /// </remarks>
    public class Error : Exception {
        /// <summary>An error without a message.</summary>
        public Error() { }

        /// <summary>An error with Gymnasium's message, which callers and ported tests may match verbatim.</summary>
        /// <param name="message">The message, ported verbatim from the Python source that raises it.</param>
        public Error(string message) : base(message) { }

        /// <summary>An error with a message and the exception that caused it.</summary>
        /// <param name="message">The message, ported verbatim from the Python source that raises it.</param>
        /// <param name="innerException">The underlying cause.</param>
        public Error(string message, Exception innerException) : base(message, innerException) { }
    }
}
