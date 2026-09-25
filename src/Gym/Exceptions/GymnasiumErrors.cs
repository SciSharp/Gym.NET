#nullable enable
using System;

namespace Gym.Exceptions {
    /// <summary>
    ///     The registry has no env for the requested id. Catch it to handle any of its three precise causes
    ///     (<see cref="NamespaceNotFound"/>, <see cref="NameNotFound"/>, <see cref="VersionNotFound"/>).
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/error.py#L9">gymnasium/error.py</see>.</remarks>
    public class UnregisteredEnv : Error {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public UnregisteredEnv() : base("") { }

        /// <summary>An error with Gymnasium's message, which callers and tests match verbatim.</summary>
        /// <param name="message">The message; Gym's own raise sites pass Gymnasium's text verbatim.</param>
        public UnregisteredEnv(string message) : base(message) { }

        /// <summary>An error with a message and the exception that caused it.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public UnregisteredEnv(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>The id's namespace (<c>ns/Name-v0</c>) is not registered.</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/error.py#L13">gymnasium/error.py</see>.</remarks>
    public class NamespaceNotFound : UnregisteredEnv {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public NamespaceNotFound() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message; Gym's own raise sites pass Gymnasium's text verbatim.</param>
        public NamespaceNotFound(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public NamespaceNotFound(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>No env of the id's name exists (in its namespace).</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/error.py#L17">gymnasium/error.py</see>.</remarks>
    public class NameNotFound : UnregisteredEnv {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public NameNotFound() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message; Gym's own raise sites pass Gymnasium's text verbatim.</param>
        public NameNotFound(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public NameNotFound(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>The name exists, but not in the requested version.</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/error.py#L21">gymnasium/error.py</see>.</remarks>
    public class VersionNotFound : UnregisteredEnv {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public VersionNotFound() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message; Gym's own raise sites pass Gymnasium's text verbatim.</param>
        public VersionNotFound(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public VersionNotFound(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>The requested version is older than the latest one of the same name.</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/error.py#L25">gymnasium/error.py</see>.</remarks>
    public class DeprecatedEnv : Error {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public DeprecatedEnv() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message; Gym's own raise sites pass Gymnasium's text verbatim.</param>
        public DeprecatedEnv(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public DeprecatedEnv(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>An invalid registration, such as an unversioned id next to versioned ones.</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/error.py#L29">gymnasium/error.py</see>.</remarks>
    public class RegistrationError : Error {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public RegistrationError() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message; Gym's own raise sites pass Gymnasium's text verbatim.</param>
        public RegistrationError(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public RegistrationError(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>An optional dependency an env needs is missing.</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/error.py#L34">gymnasium/error.py</see>.</remarks>
    public class DependencyNotInstalled : Error {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public DependencyNotInstalled() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message; Gym's own raise sites pass Gymnasium's text verbatim.</param>
        public DependencyNotInstalled(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public DependencyNotInstalled(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>A render mode the env does not support was requested.</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/error.py#L38">gymnasium/error.py</see>.</remarks>
    public class UnsupportedMode : Error {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public UnsupportedMode() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message; Gym's own raise sites pass Gymnasium's text verbatim.</param>
        public UnsupportedMode(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public UnsupportedMode(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>An env's metadata is not valid (for example <c>render_fps</c> missing where needed).</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/error.py#L42">gymnasium/error.py</see>.</remarks>
    public class InvalidMetadata : Error {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public InvalidMetadata() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message; Gym's own raise sites pass Gymnasium's text verbatim.</param>
        public InvalidMetadata(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public InvalidMetadata(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>
    ///     <c>step</c> or <c>render</c> was called before the first <c>reset</c>, which
    ///     <see cref="Gym.Wrappers.OrderEnforcing{TObs, TAct}"/> enforces.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/error.py#L46">gymnasium/error.py</see>.</remarks>
    public class ResetNeeded : Error {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public ResetNeeded() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message; Gym's own raise sites pass Gymnasium's text verbatim.</param>
        public ResetNeeded(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public ResetNeeded(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>
    ///     An action outside the action space. The envs mostly assert instead (<see cref="AssertionError"/>), as Gymnasium's
    ///     do; the legacy <see cref="InvalidActionError"/> is Gym.NET's older counterpart, kept for the legacy API.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/error.py#L50">gymnasium/error.py</see>.</remarks>
    public class InvalidAction : Error {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public InvalidAction() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message; Gym's own raise sites pass Gymnasium's text verbatim.</param>
        public InvalidAction(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public InvalidAction(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>A required constructor argument was not given.</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/error.py#L54">gymnasium/error.py</see>.</remarks>
    public class MissingArgument : Error {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public MissingArgument() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message; Gym's own raise sites pass Gymnasium's text verbatim.</param>
        public MissingArgument(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public MissingArgument(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>A probability outside [0, 1] (for example <c>StickyAction</c>'s).</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/error.py#L58">gymnasium/error.py</see>.</remarks>
    public class InvalidProbability : Error {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public InvalidProbability() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message; Gym's own raise sites pass Gymnasium's text verbatim.</param>
        public InvalidProbability(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public InvalidProbability(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>Clipping or rescaling with an invalid lower and/or upper bound.</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/error.py#L62">gymnasium/error.py</see>.</remarks>
    public class InvalidBound : Error {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public InvalidBound() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message; Gym's own raise sites pass Gymnasium's text verbatim.</param>
        public InvalidBound(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public InvalidBound(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>
    ///     Raised when importing an old version of a wrapper. It is an <see cref="ImportError"/>, not an
    ///     <see cref="Error"/>, as in Gymnasium: <c>catch (Error)</c> does not catch it.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/error.py#L67">gymnasium/error.py</see>.</remarks>
    public class DeprecatedWrapper : ImportError {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public DeprecatedWrapper() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message; Gym's own raise sites pass Gymnasium's text verbatim.</param>
        public DeprecatedWrapper(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public DeprecatedWrapper(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>
    ///     An asynchronous <c>reset</c> or <c>step</c> was started while the previous one of the same kind is still pending.
    ///     Not an <see cref="Error"/>, as in Gymnasium; the legacy <see cref="AlreadySteppingError"/> is its older counterpart.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/error.py#L72">gymnasium/error.py</see>.</remarks>
    public class AlreadyPendingCallError : Exception {
        /// <summary>The error with Gymnasium's message and the name of the call that is pending.</summary>
        /// <param name="message">The message; Gym's own raise sites pass Gymnasium's text verbatim.</param>
        /// <param name="name">The pending call's name (<c>step</c>, <c>reset</c>, <c>call</c>), Python's <c>name</c> attribute.</param>
        public AlreadyPendingCallError(string message, string name) : base(message) {
            Name = name;
        }

        /// <summary>The name of the call that is still pending.</summary>
        public string Name { get; }
    }

    /// <summary>
    ///     <c>reset_wait</c> or <c>step_wait</c> was called with no matching asynchronous call running. The legacy
    ///     <see cref="NotSteppingError"/> is its older counterpart.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/error.py#L83">gymnasium/error.py</see>.</remarks>
    public class NoAsyncCallError : Exception {
        /// <summary>The error with Gymnasium's message and the name of the call that was expected.</summary>
        /// <param name="message">The message; Gym's own raise sites pass Gymnasium's text verbatim.</param>
        /// <param name="name">The expected call's name, Python's <c>name</c> attribute.</param>
        public NoAsyncCallError(string message, string name) : base(message) {
            Name = name;
        }

        /// <summary>The name of the call that was expected to be running.</summary>
        public string Name { get; }
    }

    /// <summary><c>reset</c> or <c>step</c> on an environment that was closed.</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/error.py#L94">gymnasium/error.py</see>.</remarks>
    public class ClosedEnvironmentError : Exception {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public ClosedEnvironmentError() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message; Gym's own raise sites pass Gymnasium's text verbatim.</param>
        public ClosedEnvironmentError(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public ClosedEnvironmentError(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>A custom space that <c>AsyncVectorEnv</c> with shared memory cannot place in shared memory.</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/error.py#L98">gymnasium/error.py</see>.</remarks>
    public class CustomSpaceError : Exception {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public CustomSpaceError() : base("") { }

        /// <summary>An error with Gymnasium's message.</summary>
        /// <param name="message">The message; Gym's own raise sites pass Gymnasium's text verbatim.</param>
        public CustomSpaceError(string message) : base(message) { }

        /// <summary>An error with a message and its cause.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public CustomSpaceError(string message, Exception innerException) : base(message, innerException) { }
    }
}
