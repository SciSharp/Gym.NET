using System;

namespace Gym.Exceptions {
    /// <summary>
    ///     Port of <c>gymnasium.error.Error</c> (<c>gymnasium/error.py:4</c> @ 9e04324), the superclass of every error that
    ///     Gymnasium raises itself (as opposed to Python built-ins such as <c>ValueError</c>, which Gym.NET takes from
    ///     NumSharp, decision D17). Catch it to handle any Gymnasium-defined failure, for example an invalid seed.
    /// </summary>
    /// <remarks>
    ///     Every class of <c>error.py</c> is ported with its hierarchy (<see cref="UnregisteredEnv"/> and its three
    ///     subclasses, <see cref="ResetNeeded"/>, <see cref="InvalidAction"/>, ...); the ones Gymnasium does not derive from
    ///     <c>Error</c> (<see cref="AlreadyPendingCallError"/>, <see cref="DeprecatedWrapper"/>, ...) do not here either.
    ///     Gym.NET's pre-port exceptions (<see cref="InvalidActionError"/>, <see cref="AlreadySteppingError"/>,
    ///     <see cref="NotSteppingError"/>) remain for the legacy API.
    /// </remarks>
    public class Error : Exception {
        /// <summary>
        ///     An error without a message. Its <see cref="Exception.Message"/> is empty, as Python's <c>str(Error())</c> is,
        ///     rather than .NET's generic "Exception of type ... was thrown.", so messages compare verbatim with Gymnasium's.
        /// </summary>
        public Error() : base("") { }

        /// <summary>An error with Gymnasium's message, which callers and ported tests may match verbatim.</summary>
        /// <param name="message">The message, ported verbatim from the Python source that raises it.</param>
        public Error(string message) : base(message) { }

        /// <summary>An error with a message and the exception that caused it.</summary>
        /// <param name="message">The message, ported verbatim from the Python source that raises it.</param>
        /// <param name="innerException">The underlying cause.</param>
        public Error(string message, Exception innerException) : base(message, innerException) { }
    }
}
