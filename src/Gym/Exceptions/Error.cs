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
    ///     (<see cref="AlreadyPendingCallError"/>, <see cref="DeprecatedWrapper"/>, ...) do not here either. 0.2's own
    ///     exceptions are obsolete bases of their Gymnasium counterparts until 3.0 (DR-58), so 0.2's <c>catch</c> blocks still
    ///     compile: <see cref="InvalidActionError"/> under <see cref="InvalidAction"/> (the envs that assert their action raise
    ///     <see cref="NumSharp.AssertionError"/> instead), <see cref="AlreadySteppingError"/> under
    ///     <see cref="AlreadyPendingCallError"/>, <see cref="NotSteppingError"/> under <see cref="NoAsyncCallError"/>. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/error.py#L4">gymnasium/error.py</see>.
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
