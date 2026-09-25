#nullable enable
using System;
using System.Collections.Generic;
using System.IO;

namespace Gym.Utils {
    /// <summary>
    ///     The Python warning classes <see cref="Logger"/> issues, as Gymnasium's logger does: <c>UserWarning</c> (the
    ///     default of <c>warnings.warn</c>) and <c>DeprecationWarning</c> (<see cref="Logger.Deprecation"/>).
    /// </summary>
    public enum WarningCategory {
        /// <summary>Python's <c>UserWarning</c>, shown once per call site by Python's default filter.</summary>
        UserWarning,

        /// <summary>Python's <c>DeprecationWarning</c>, shown once per message by the filter Gymnasium installs for its modules.</summary>
        DeprecationWarning,
    }

    /// <summary>One issued warning, as Python's <c>warnings.catch_warnings(record=True)</c> records it.</summary>
    /// <param name="Message">The text, <c>str(w.message)</c>: for <see cref="Logger"/> the colourised <c>WARN: ...</c> line.</param>
    /// <param name="Category">The warning class.</param>
    /// <param name="FileName">The source file of the call site the warning is attributed to (the C# caller, not a Python file).</param>
    /// <param name="LineNumber">The line of that call site.</param>
    public sealed record WarningMessage(string Message, WarningCategory Category, string FileName, int LineNumber) {
        /// <summary>
        ///     The line Python's <c>warnings.formatwarning</c> prints for the warning (without the source line Python appends
        ///     when it can read the file): <c>file:line: Category: message</c>.
        /// </summary>
        /// <returns>The formatted warning, without a trailing newline.</returns>
        public override string ToString() => $"{FileName}:{LineNumber}: {Category}: {Message}";
    }

    /// <summary>
    ///     The part of Python's <c>warnings</c> module that <see cref="Logger"/> depends on: its default filters, where shown
    ///     warnings go, and <c>catch_warnings(record=True)</c> for tests. Gymnasium never throws where it only warns, so
    ///     neither does Gym: a warning is text on the error stream (or a record), never an exception.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Filters, as Python applies them to Gymnasium's warnings: a <see cref="WarningCategory.UserWarning"/> is shown the
    ///     first time a given message is issued from a given call site and suppressed after that (Python's <c>default</c>
    ///     action, keyed on the text, the category and the location); a <see cref="WarningCategory.DeprecationWarning"/> is
    ///     shown the first time a given message is issued from anywhere (the <c>once</c> filter Gymnasium's logger installs
    ///     for its modules). So CartPole's "already returned terminated = True" warning appears once per process, not once
    ///     per episode, exactly as in Python. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/logger.py#L14">gymnasium/logger.py</see>.
    ///     </para>
    ///     <para>
    ///     Python forgets which warnings it has shown whenever the filters change (<c>_filters_mutated</c>), which leaving a
    ///     <c>catch_warnings</c> block does; <see cref="Catch"/> mirrors that by clearing the registry when a catcher is
    ///     disposed. All members are thread-safe.
    ///     </para>
    /// </remarks>
    public static class Warnings {
        /// <summary>Guards the registry, the catcher stack and the output.</summary>
        private static readonly object Gate = new();

        /// <summary>
        ///     What has been shown: <c>(category, message, file, line)</c> for user warnings, <c>(category, message)</c> with
        ///     an empty location for deprecations. Python's per-module <c>__warningregistry__</c> plus the <c>once</c> registry.
        /// </summary>
        private static readonly HashSet<(WarningCategory, string, string, int)> Shown = new();

        /// <summary>The active catchers, innermost last; the innermost one records everything issued while it is active.</summary>
        private static readonly List<WarningCatcher> Catchers = new();

        /// <summary>Where the warning text goes, <see langword="null"/> for <see cref="Console.Error"/>, Python's <c>sys.stderr</c>.</summary>
        private static TextWriter? _output;

        /// <summary>
        ///     Where shown warnings are written, one <see cref="WarningMessage.ToString"/> line each; <see langword="null"/>
        ///     restores <see cref="Console.Error"/>. Assign it to route warnings into a log or a test's buffer, as Python code
        ///     replaces <c>sys.stderr</c> or <c>warnings.showwarning</c>.
        /// </summary>
        public static TextWriter? Output {
            get {
                lock (Gate)
                    return _output;
            }
            set {
                lock (Gate)
                    _output = value;
            }
        }

        /// <summary>
        ///     Issues a warning the way <c>warnings.warn(message, category)</c> does: recorded by the innermost active
        ///     <see cref="Catch"/> block, otherwise written to <see cref="Output"/> unless the default filter has already shown
        ///     it.
        /// </summary>
        /// <param name="message">The warning text.</param>
        /// <param name="category">The warning class.</param>
        /// <param name="fileName">The call site's file, the location Python's <c>stacklevel</c> selects.</param>
        /// <param name="lineNumber">The call site's line.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="message"/> or <paramref name="fileName"/> is <see langword="null"/>.</exception>
        public static void Warn(string message, WarningCategory category, string fileName, int lineNumber) {
            if (message is null)
                throw new ArgumentNullException(nameof(message));
            if (fileName is null)
                throw new ArgumentNullException(nameof(fileName));

            var warning = new WarningMessage(message, category, fileName, lineNumber);
            lock (Gate) {
                if (Catchers.Count > 0) {
                    // catch_warnings(record=True) with simplefilter("always"): every warning is recorded, none is printed,
                    // and the registry is left alone (Python's "always" action does not register).
                    Catchers[^1].Add(warning);
                    return;
                }

                // DeprecationWarning: "once" per message anywhere; UserWarning: "default", once per message per call site.
                (WarningCategory, string, string, int) key = category == WarningCategory.DeprecationWarning
                    ? (category, message, "", 0)
                    : (category, message, fileName, lineNumber);
                if (!Shown.Add(key))
                    return;
                (_output ?? Console.Error).WriteLine(warning.ToString());
            }
        }

        /// <summary>
        ///     Starts recording every warning instead of showing it, as Python's
        ///     <c>with warnings.catch_warnings(record=True): warnings.simplefilter("always")</c> block does. Dispose the result
        ///     to stop; that also clears the shown-warnings registry, as leaving Python's block does.
        /// </summary>
        /// <returns>The catcher; its <see cref="WarningCatcher.Records"/> grow while it is the innermost active one.</returns>
        public static WarningCatcher Catch() {
            var catcher = new WarningCatcher();
            lock (Gate)
                Catchers.Add(catcher);
            return catcher;
        }

        /// <summary>Stops a catcher (any nesting order) and clears the registry. Called by <see cref="WarningCatcher.Dispose"/>.</summary>
        /// <param name="catcher">The catcher to remove.</param>
        internal static void Release(WarningCatcher catcher) {
            lock (Gate) {
                if (Catchers.Remove(catcher))
                    // Leaving catch_warnings restores the filters, and Python invalidates every registry when filters mutate.
                    Shown.Clear();
            }
        }

        /// <summary>
        ///     Forgets which warnings have been shown, so each is shown again once: the effect of any filter change in Python
        ///     (<c>warnings.resetwarnings()</c>, <c>filterwarnings(...)</c>). Tests use it to observe the first-time behaviour.
        /// </summary>
        public static void ResetRegistry() {
            lock (Gate)
                Shown.Clear();
        }
    }

    /// <summary>
    ///     A <see cref="Warnings.Catch"/> block: the warnings issued while it was the innermost active catcher, in order.
    ///     Dispose it (<c>using</c>) to end the block.
    /// </summary>
    public sealed class WarningCatcher : IDisposable {
        /// <summary>The recorded warnings; guarded by the instance itself.</summary>
        private readonly List<WarningMessage> _records = new();

        /// <summary>Only <see cref="Warnings.Catch"/> creates catchers, so every catcher is registered.</summary>
        internal WarningCatcher() { }

        /// <summary>A snapshot of the warnings recorded so far, oldest first (Python's <c>caught</c> list).</summary>
        public IReadOnlyList<WarningMessage> Records {
            get {
                lock (_records)
                    return _records.ToArray();
            }
        }

        /// <summary>Appends a warning; called by <see cref="Warnings.Warn"/> under its lock.</summary>
        /// <param name="warning">The warning.</param>
        internal void Add(WarningMessage warning) {
            lock (_records)
                _records.Add(warning);
        }

        /// <summary>Ends the block: later warnings are shown again, and the registry is cleared as Python's filters are restored.</summary>
        public void Dispose() => Warnings.Release(this);
    }
}
