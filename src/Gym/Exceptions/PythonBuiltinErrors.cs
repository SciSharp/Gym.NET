#nullable enable
using System;

namespace Gym.Exceptions {
    /// <summary>
    ///     A method a subclass is expected to provide was called on a class that does not: <c>Render</c> of an env that does
    ///     not render, for example. Python's <c>NotImplementedError</c>; it derives from
    ///     <see cref="NotImplementedException"/>, the closest .NET exception, so either catch works.
    /// </summary>
    /// <remarks>
    ///     NumSharp owns every Python built-in exception Gym.NET throws (decisions D17 and DR-02), but NumSharp 0.80.0 has no
    ///     <c>NotImplementedError</c> yet. This type is the stuck-case workaround DR-01 allows until NumSharp adds it (open
    ///     item N-3, proven by the <c>[OpenBugs("N-3")]</c> test in <c>tests/Gym.Tests/NumSharpGaps</c>); it is then deleted
    ///     and every raise site throws NumSharp's.
    ///     Reference: <see href="https://docs.python.org/3.12/library/exceptions.html#NotImplementedError">NotImplementedError</see>.
    /// </remarks>
    public class NotImplementedError : NotImplementedException {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public NotImplementedError() : base("") { }

        /// <summary>An error with a message; Gym raises it where Gymnasium does, with Gymnasium's text.</summary>
        /// <param name="message">The message; Gym's own raise sites pass Gymnasium's text verbatim.</param>
        public NotImplementedError(string message) : base(message) { }

        /// <summary>An error with a message and the exception that caused it.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public NotImplementedError(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>
    ///     Python's <c>ImportError</c>, the base of <see cref="DeprecatedWrapper"/>. There is no module system to import from
    ///     in .NET; the type exists so that the error hierarchy, and <c>catch</c> clauses written against it, keep their
    ///     shape.
    /// </summary>
    /// <remarks>
    ///     NumSharp owns every Python built-in exception Gym.NET throws (decisions D17 and DR-02), but NumSharp 0.80.0 has no
    ///     <c>ImportError</c> yet. This type is the stuck-case workaround DR-01 allows until NumSharp adds it (open item N-3,
    ///     proven by the <c>[OpenBugs("N-3")]</c> test in <c>tests/Gym.Tests/NumSharpGaps</c>); it is then deleted and every
    ///     raise site throws NumSharp's.
    ///     Reference: <see href="https://docs.python.org/3.12/library/exceptions.html#ImportError">ImportError</see>.
    /// </remarks>
    public class ImportError : Exception {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public ImportError() : base("") { }

        /// <summary>An error with a message; Gym raises it where Gymnasium does, with Gymnasium's text.</summary>
        /// <param name="message">The message; Gym's own raise sites pass Gymnasium's text verbatim.</param>
        public ImportError(string message) : base(message) { }

        /// <summary>An error with a message and the exception that caused it.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public ImportError(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>
    ///     Python's <c>ModuleNotFoundError</c>: what <c>make</c> and <c>load_env_creator</c> raise when an entry point or a
    ///     <c>"module:Env-v0"</c> id names a module that does not exist. To Gym.NET a module is a namespace (or the assembly
    ///     named after it), so the message names the first part of the dotted name no loaded assembly declares:
    ///     <c>No module named 'Gym.Nope'</c>.
    /// </summary>
    /// <remarks>
    ///     A subclass of <see cref="ImportError"/>, as in Python. The same stuck-case workaround as <see cref="ImportError"/>
    ///     (DR-01, open item N-3): it moves to NumSharp with it.
    ///     Reference: <see href="https://docs.python.org/3.12/library/exceptions.html#ModuleNotFoundError">ModuleNotFoundError</see>.
    /// </remarks>
    public class ModuleNotFoundError : ImportError {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public ModuleNotFoundError() : base("") { }

        /// <summary>An error with a message; Gym raises it where Gymnasium does, with Gymnasium's text.</summary>
        /// <param name="message">The message: <c>No module named 'x'</c>.</param>
        public ModuleNotFoundError(string message) : base(message) { }

        /// <summary>An error with a message and the exception that caused it.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public ModuleNotFoundError(string message, Exception innerException) : base(message, innerException) { }
    }
}
