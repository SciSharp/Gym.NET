#nullable enable
using System;

namespace Gym.Exceptions {
    /// <summary>
    ///     A method a subclass is expected to provide was called on a class that does not: <c>Render</c> of an env that does
    ///     not render, for example. Python's <c>NotImplementedError</c>; it derives from
    ///     <see cref="NotImplementedException"/>, the closest .NET exception, so either catch works.
    /// </summary>
    /// <remarks>
    ///     NumSharp, whose Python built-in exceptions Gym.NET reuses (decision D17), has no <c>NotImplementedError</c>.
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
    ///     NumSharp has no <c>ImportError</c> (decision D17 puts the built-ins it lacks here).
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
}
