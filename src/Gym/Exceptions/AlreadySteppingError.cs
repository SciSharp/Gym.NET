using System;

namespace Gym.Exceptions {
    /// <summary>
    ///     0.2's "already running an async step" error, kept so 0.2 code that catches it still compiles (DR-58). It is
    ///     <see cref="AlreadyPendingCallError"/>'s base class, so <c>catch (AlreadySteppingError)</c> catches
    ///     <c>AsyncVectorEnv</c>'s error for a call started while another is pending, the same situation.
    /// </summary>
    /// <remarks>
    ///     Obsolete, and removed in 3.0 with <c>Gym.Legacy</c>. 0.2 never threw it, so a <c>catch</c> of it that never ran
    ///     in 0.2 can now run. Like Gymnasium's <c>AlreadyPendingCallError</c>, it is not an <see cref="Error"/>; the extra
    ///     level changes no message and not the Python type the subclass stands for.
    /// </remarks>
    [Obsolete("0.2's AlreadySteppingError: catch Gym.Exceptions.AlreadyPendingCallError, which AsyncVectorEnv raises. Removed in 3.0 with Gym.Legacy.")]
    public class AlreadySteppingError : Exception {
        /// <summary>An error with 0.2's own message, <c>already running an async step</c>.</summary>
        public AlreadySteppingError() : base("already running an async step") { }

        /// <summary>An error with the given message; <see cref="AlreadyPendingCallError"/> passes Gymnasium's through it.</summary>
        /// <param name="message">The message.</param>
        public AlreadySteppingError(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public AlreadySteppingError(string message, Exception innerException) : base(message, innerException) { }
    }
}
