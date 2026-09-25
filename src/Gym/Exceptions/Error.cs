using System;

namespace Gym.Exceptions {
    /// <summary>
    ///     The superclass of every error Gym.NET raises itself (as opposed to Python built-ins such as <c>ValueError</c>, which
    ///     Gym.NET takes from NumSharp, decision D17). Catch it to handle any library-defined failure, for example an invalid
    ///     seed.
    /// </summary>
    /// <remarks>
    ///     The error classes follow Gymnasium's hierarchy (<see cref="UnregisteredEnv"/> and its three subclasses,
    ///     <see cref="ResetNeeded"/>, <see cref="InvalidAction"/>, ...); the ones Gymnasium does not derive from <c>Error</c>
    ///     (<see cref="AlreadyPendingCallError"/>, <see cref="DeprecatedWrapper"/>, ...) do not here either. Gym.NET's older
    ///     exceptions (<see cref="InvalidActionError"/>, <see cref="AlreadySteppingError"/>, <see cref="NotSteppingError"/>)
    ///     remain for the legacy API. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/error.py#L4">gymnasium/error.py</see>.
    /// </remarks>
    public class Error : Exception {
        /// <summary>
        ///     An error without a message. Its <see cref="Exception.Message"/> is empty, as Python's <c>str(Error())</c> is,
        ///     rather than .NET's generic "Exception of type ... was thrown.", so messages compare verbatim with Gymnasium's.
        /// </summary>
        public Error() : base("") { }

        /// <summary>An error with Gymnasium's message, which callers and tests may match verbatim.</summary>
        /// <param name="message">The message; Gym's own raise sites pass Gymnasium's text verbatim.</param>
        public Error(string message) : base(message) { }

        /// <summary>An error with a message and the exception that caused it.</summary>
        /// <param name="message">The message; Gym's own raise sites pass Gymnasium's text verbatim.</param>
        /// <param name="innerException">The underlying cause.</param>
        public Error(string message, Exception innerException) : base(message, innerException) { }
    }
}
