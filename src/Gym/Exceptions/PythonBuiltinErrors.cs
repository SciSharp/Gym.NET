#nullable enable
using System;

namespace Gym.Exceptions {
    /// <summary>
    ///     Port of Python's built-in <c>NotImplementedError</c>, what Gymnasium's abstract-by-convention methods raise:
    ///     <c>Env.render</c> of an env that does not render, <c>ObservationWrapper.observation</c> not overridden. Derives from
    ///     <see cref="NotImplementedException"/>, the closest .NET exception, so either catch works.
    /// </summary>
    /// <remarks>NumSharp, whose ports of Python's built-ins Gym.NET reuses (decision D17), has no <c>NotImplementedError</c>.</remarks>
    public class NotImplementedError : NotImplementedException {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public NotImplementedError() : base("") { }

        /// <summary>An error with the message the Python source raises it with.</summary>
        /// <param name="message">The message, ported verbatim.</param>
        public NotImplementedError(string message) : base(message) { }

        /// <summary>An error with a message and the exception that caused it.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public NotImplementedError(string message, Exception innerException) : base(message, innerException) { }
    }

    /// <summary>
    ///     Port of Python's built-in <c>ImportError</c>, the base of Gymnasium's <see cref="DeprecatedWrapper"/>. There is no
    ///     module system to import from in .NET; the type exists so that the ported hierarchy, and <c>catch</c> clauses written
    ///     against it, keep Gymnasium's shape.
    /// </summary>
    /// <remarks>NumSharp has no <c>ImportError</c> (decision D17 puts the built-ins it lacks here).</remarks>
    public class ImportError : Exception {
        /// <summary>An error without a message; its <see cref="Exception.Message"/> is empty, as Python's <c>str(error)</c> is.</summary>
        public ImportError() : base("") { }

        /// <summary>An error with the message the Python source raises it with.</summary>
        /// <param name="message">The message, ported verbatim.</param>
        public ImportError(string message) : base(message) { }

        /// <summary>An error with a message and the exception that caused it.</summary>
        /// <param name="message">The message.</param>
        /// <param name="innerException">The underlying cause.</param>
        public ImportError(string message, Exception innerException) : base(message, innerException) { }
    }
}
