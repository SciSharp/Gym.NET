using System;

namespace Gym.Exceptions {
    /// <summary>
    ///     0.2's invalid-action error, kept so 0.2 code that catches or throws it still compiles (DR-58). It is
    ///     <see cref="InvalidAction"/>'s base class, so <c>catch (InvalidActionError)</c> catches what CarRacing and custom
    ///     envs raise for an action outside their space.
    /// </summary>
    /// <remarks>
    ///     Obsolete, and removed in 2.0 with <c>Gym.Legacy</c>. The compatibility is partial: the envs that assert their
    ///     action, as Gymnasium's do (LunarLander, CartPole, ...), raise <see cref="NumSharp.AssertionError"/>, which a
    ///     <c>catch (InvalidActionError)</c> does not see. It is an <see cref="Error"/>, so Gymnasium's hierarchy gains this
    ///     one level between <see cref="Error"/> and <see cref="InvalidAction"/>; messages, and the Python type
    ///     <see cref="InvalidAction"/> stands for, are unchanged.
    /// </remarks>
    [Obsolete("0.2's InvalidActionError: catch Gym.Exceptions.InvalidAction (CarRacing, custom envs) or NumSharp.AssertionError (the envs that assert their action, as Gymnasium's do). Removed in 2.0 with Gym.Legacy.")]
    public class InvalidActionError : Error {
        /// <summary>An error with 0.2's own message, <c>Action is outside of the configured action space.</c>, for 0.2 code that throws it.</summary>
        public InvalidActionError() : base("Action is outside of the configured action space.") { }

        /// <summary>An error with the given message; <see cref="InvalidAction"/> passes Gymnasium's through it.</summary>
        /// <param name="message">The message.</param>
        public InvalidActionError(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public InvalidActionError(string message, Exception innerException) : base(message, innerException) { }
    }
}
