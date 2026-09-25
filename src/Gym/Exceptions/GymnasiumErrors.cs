#nullable enable
using System;

namespace Gym.Exceptions {
    /// <summary>
    ///     Port of <c>error.UnregisteredEnv</c>: the registry has no env for the requested id. Catch it to handle any of its
    ///     three precise causes (<see cref="NamespaceNotFound"/>, <see cref="NameNotFound"/>, <see cref="VersionNotFound"/>).
    /// </summary>
    public class UnregisteredEnv : Error {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public UnregisteredEnv() : base("") { }

        /// <summary>An error with Gymnasium's message, which callers and ported tests match verbatim.</summary>
        /// <param name="message">The message, ported verbatim from the Python source that raises it.</param>
        public UnregisteredEnv(string message) : base(message) { }

        /// <summary>An error with a message and the exception that caused it.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public UnregisteredEnv(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>Port of <c>error.NamespaceNotFound</c>: the id's namespace (<c>ns/Name-v0</c>) is not registered.</summary>
    public class NamespaceNotFound : UnregisteredEnv {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public NamespaceNotFound() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message, ported verbatim.</param>
        public NamespaceNotFound(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public NamespaceNotFound(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>Port of <c>error.NameNotFound</c>: no env of the id's name exists (in its namespace).</summary>
    public class NameNotFound : UnregisteredEnv {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public NameNotFound() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message, ported verbatim.</param>
        public NameNotFound(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public NameNotFound(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>Port of <c>error.VersionNotFound</c>: the name exists, but not in the requested version.</summary>
    public class VersionNotFound : UnregisteredEnv {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public VersionNotFound() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message, ported verbatim.</param>
        public VersionNotFound(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public VersionNotFound(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>Port of <c>error.DeprecatedEnv</c>: the requested version is older than the latest one of the same name.</summary>
    public class DeprecatedEnv : Error {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public DeprecatedEnv() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message, ported verbatim.</param>
        public DeprecatedEnv(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public DeprecatedEnv(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>Port of <c>error.RegistrationError</c>: an invalid registration, such as an unversioned id next to versioned ones.</summary>
    public class RegistrationError : Error {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public RegistrationError() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message, ported verbatim.</param>
        public RegistrationError(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public RegistrationError(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>Port of <c>error.DependencyNotInstalled</c>: an optional dependency an env needs is missing.</summary>
    public class DependencyNotInstalled : Error {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public DependencyNotInstalled() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message, ported verbatim.</param>
        public DependencyNotInstalled(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public DependencyNotInstalled(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>Port of <c>error.UnsupportedMode</c>: a render mode the env does not support was requested.</summary>
    public class UnsupportedMode : Error {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public UnsupportedMode() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message, ported verbatim.</param>
        public UnsupportedMode(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public UnsupportedMode(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>Port of <c>error.InvalidMetadata</c>: an env's metadata is not valid (for example <c>render_fps</c> missing where needed).</summary>
    public class InvalidMetadata : Error {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public InvalidMetadata() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message, ported verbatim.</param>
        public InvalidMetadata(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public InvalidMetadata(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>
    ///     Port of <c>error.ResetNeeded</c>: <c>step</c> or <c>render</c> was called before the first <c>reset</c>, which
    ///     <see cref="Gym.Wrappers.OrderEnforcing{TObs, TAct}"/> enforces.
    /// </summary>
    public class ResetNeeded : Error {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public ResetNeeded() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message, ported verbatim.</param>
        public ResetNeeded(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public ResetNeeded(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>
    ///     Port of <c>error.InvalidAction</c>: an action outside the action space. Gymnasium's own envs mostly
    ///     <c>assert</c> instead (<see cref="AssertionError"/>); the legacy <see cref="InvalidActionError"/> is Gym.NET's
    ///     pre-port counterpart, kept for the legacy API.
    /// </summary>
    public class InvalidAction : Error {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public InvalidAction() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message, ported verbatim.</param>
        public InvalidAction(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public InvalidAction(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>Port of <c>error.MissingArgument</c>: a required constructor argument was not given.</summary>
    public class MissingArgument : Error {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public MissingArgument() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message, ported verbatim.</param>
        public MissingArgument(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public MissingArgument(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>Port of <c>error.InvalidProbability</c>: a probability outside [0, 1] (for example <c>StickyAction</c>'s).</summary>
    public class InvalidProbability : Error {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public InvalidProbability() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message, ported verbatim.</param>
        public InvalidProbability(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public InvalidProbability(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>Port of <c>error.InvalidBound</c>: clipping or rescaling with an invalid lower and/or upper bound.</summary>
    public class InvalidBound : Error {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public InvalidBound() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message, ported verbatim.</param>
        public InvalidBound(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public InvalidBound(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>
    ///     Port of <c>error.DeprecatedWrapper</c>, raised when importing an old version of a wrapper. Like the source it is an
    ///     <see cref="ImportError"/>, not a <see cref="Error"/>: <c>catch (Error)</c> does not catch it.
    /// </summary>
    public class DeprecatedWrapper : ImportError {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public DeprecatedWrapper() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message, ported verbatim.</param>
        public DeprecatedWrapper(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public DeprecatedWrapper(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>
    ///     Port of <c>error.AlreadyPendingCallError</c>: an asynchronous <c>reset</c> or <c>step</c> was started while the
    ///     previous one of the same kind is still pending. Not an <see cref="Error"/> in Gymnasium either; the legacy
    ///     <see cref="AlreadySteppingError"/> is its pre-port counterpart.
    /// </summary>
    public class AlreadyPendingCallError : Exception {
        /// <summary>The error with Gymnasium's message and the name of the call that is pending.</summary>
        /// <param name="message">The message, ported verbatim.</param>
        /// <param name="name">The pending call's name (<c>step</c>, <c>reset</c>, <c>call</c>), Python's <c>name</c> attribute.</param>
        public AlreadyPendingCallError(string message, string name) : base(message) {
            Name = name;
        }

        /// <summary>The name of the call that is still pending.</summary>
        public string Name { get; }
    }

    /// <summary>
    ///     Port of <c>error.NoAsyncCallError</c>: <c>reset_wait</c> or <c>step_wait</c> was called with no matching
    ///     asynchronous call running. The legacy <see cref="NotSteppingError"/> is its pre-port counterpart.
    /// </summary>
    public class NoAsyncCallError : Exception {
        /// <summary>The error with Gymnasium's message and the name of the call that was expected.</summary>
        /// <param name="message">The message, ported verbatim.</param>
        /// <param name="name">The expected call's name, Python's <c>name</c> attribute.</param>
        public NoAsyncCallError(string message, string name) : base(message) {
            Name = name;
        }

        /// <summary>The name of the call that was expected to be running.</summary>
        public string Name { get; }
    }

    /// <summary>Port of <c>error.ClosedEnvironmentError</c>: <c>reset</c> or <c>step</c> on an environment that was closed.</summary>
    public class ClosedEnvironmentError : Exception {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public ClosedEnvironmentError() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message, ported verbatim.</param>
        public ClosedEnvironmentError(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public ClosedEnvironmentError(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>
    ///     Port of <c>error.CustomSpaceError</c>: a custom space that <c>AsyncVectorEnv</c> with shared memory cannot place in
    ///     shared memory.
    /// </summary>
    public class CustomSpaceError : Exception {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public CustomSpaceError() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message, ported verbatim.</param>
        public CustomSpaceError(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public CustomSpaceError(string message, Exception innerException) : base(message, innerException) { }
    }
}
