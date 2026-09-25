#nullable enable
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Gym.Utils;

namespace Gym {
    /// <summary>
    ///     Where Gym's envs and wrappers report misuse and oddities without stopping: each message is issued through
    ///     <see cref="Warnings"/> as Python's <c>warnings.warn</c> would, with the <c>WARN: </c>/<c>ERROR: </c> prefix wrapped
    ///     in ANSI colour codes, so the text reads, and compares, exactly like Gymnasium's.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A warning never throws and is shown once per call site (Python's default filter), so an env that warns on every
    ///     misuse prints one line per process, not one per step. Tests observe warnings with <see cref="Warnings.Catch"/>.
    ///     </para>
    ///     <para>
    ///     Python's <c>msg % args</c> formatting is left to the caller: C# callers pass the finished message (interpolated),
    ///     which is the same text for Gymnasium's call sites, none of which contains a <c>%</c> directive without arguments.
    ///     The call site is the direct caller (<see cref="CallerFilePathAttribute"/>/<see cref="CallerLineNumberAttribute"/>),
    ///     which is what <c>stacklevel=2</c> selects for <c>warn</c> and <c>deprecation</c>; <c>error</c>'s
    ///     <c>stacklevel=3</c> would name the caller's caller, a difference that only changes which repeats are merged.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/logger.py">gymnasium/logger.py</see> and <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/utils/colorize.py">gymnasium/utils/colorize.py</see>.</para>
    /// </remarks>
    public static class Logger {
        /// <summary>The level of <see cref="Warn"/> and <see cref="Deprecation"/>, Python's <c>logger.WARN</c>.</summary>
        public const int WARN = 30;

        /// <summary>The level of <see cref="Error"/>, Python's <c>logger.ERROR</c>.</summary>
        public const int ERROR = 40;

        /// <summary>
        ///     ANSI colour numbers by name, <c>colorize.color2num</c>: gray 30 through crimson 38. Highlighting adds 10 (the
        ///     background variant).
        /// </summary>
        private static readonly IReadOnlyDictionary<string, int> ColorToNumber = new Dictionary<string, int>(StringComparer.Ordinal) {
            ["gray"] = 30,
            ["red"] = 31,
            ["green"] = 32,
            ["yellow"] = 33,
            ["blue"] = 34,
            ["magenta"] = 35,
            ["cyan"] = 36,
            ["white"] = 37,
            ["crimson"] = 38,
        };

        /// <summary>
        ///     The least severe level that is issued, Python's module global <c>logger.min_level</c> (30): raise it to
        ///     <see cref="ERROR"/> to silence warnings, or above to silence errors too. Process-wide, like the Python global.
        /// </summary>
        public static int MinLevel { get; set; } = WARN;

        /// <summary>
        ///     Issues <c>colorize("WARN: " + msg, "yellow")</c> as a warning of <paramref name="category"/>, attributed to the
        ///     calling line, when <see cref="MinLevel"/> allows it. The caller carries on: a warning never throws, and the
        ///     filters of <see cref="Warnings"/> drop repeats.
        /// </summary>
        /// <param name="msg">The finished message (Python's <c>msg % args</c> already applied).</param>
        /// <param name="category">The warning class; Python's <c>None</c> means <see cref="WarningCategory.UserWarning"/>.</param>
        /// <param name="callerFilePath">Filled in by the compiler: the call site's file.</param>
        /// <param name="callerLineNumber">Filled in by the compiler: the call site's line.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="msg"/> is <see langword="null"/>.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/logger.py#L17">gymnasium/logger.py</see>.</remarks>
        public static void Warn(string msg, WarningCategory category = WarningCategory.UserWarning,
            [CallerFilePath] string callerFilePath = "", [CallerLineNumber] int callerLineNumber = 0) {
            if (msg is null)
                throw new ArgumentNullException(nameof(msg));
            if (MinLevel <= WARN)
                Warnings.Warn(Colorize($"WARN: {msg}", "yellow"), category, callerFilePath, callerLineNumber);
        }

        /// <summary>
        ///     Warns that an API is going away: <see cref="Warn"/> with <see cref="WarningCategory.DeprecationWarning"/>, which
        ///     is shown once per message however many call sites issue it (the <c>once</c> filter Gymnasium installs).
        /// </summary>
        /// <param name="msg">The finished message.</param>
        /// <param name="callerFilePath">Filled in by the compiler: the call site's file.</param>
        /// <param name="callerLineNumber">Filled in by the compiler: the call site's line.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="msg"/> is <see langword="null"/>.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/logger.py#L39">gymnasium/logger.py</see>.</remarks>
        public static void Deprecation(string msg, [CallerFilePath] string callerFilePath = "", [CallerLineNumber] int callerLineNumber = 0) {
            Warn(msg, WarningCategory.DeprecationWarning, callerFilePath, callerLineNumber);
        }

        /// <summary>
        ///     Reports a serious problem without stopping the caller: issues <c>colorize("ERROR: " + msg, "red")</c> as a user
        ///     warning when <see cref="MinLevel"/> allows it. Despite the name it does not throw; the caller decides what to do.
        /// </summary>
        /// <param name="msg">The finished message.</param>
        /// <param name="callerFilePath">Filled in by the compiler: the call site's file.</param>
        /// <param name="callerLineNumber">Filled in by the compiler: the call site's line.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="msg"/> is <see langword="null"/>.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/logger.py#L44">gymnasium/logger.py</see>.</remarks>
        public static void Error(string msg, [CallerFilePath] string callerFilePath = "", [CallerLineNumber] int callerLineNumber = 0) {
            if (msg is null)
                throw new ArgumentNullException(nameof(msg));
            if (MinLevel <= ERROR)
                Warnings.Warn(Colorize($"ERROR: {msg}", "red"), WarningCategory.UserWarning, callerFilePath, callerLineNumber);
        }

        /// <summary>
        ///     Wraps <paramref name="text"/> in the ANSI escape sequence of <paramref name="color"/>,
        ///     <c>"\x1b[{attrs}m{text}\x1b[0m"</c>. The escape codes become part of the text, so terminals colour it and
        ///     captured warnings compare byte for byte with Gymnasium's; a sink that does not interpret ANSI shows them raw.
        /// </summary>
        /// <param name="text">The text to colour.</param>
        /// <param name="color">gray, red, green, yellow, blue, magenta, cyan, white or crimson.</param>
        /// <param name="bold">Adds the bold attribute (<c>1</c>).</param>
        /// <param name="highlight">Uses the background colour (the number plus 10).</param>
        /// <returns>The coloured text.</returns>
        /// <exception cref="System.Collections.Generic.KeyNotFoundException">Thrown for an unknown colour name, as Python's dict lookup raises <c>KeyError</c>.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/utils/colorize.py#L23">gymnasium/utils/colorize.py</see>.</remarks>
        public static string Colorize(string text, string color, bool bold = false, bool highlight = false) {
            int number = ColorToNumber[color];
            if (highlight)
                number += 10;
            // Python joins the attribute strings with ';': the colour number, then "1" when bold.
            string attributes = number.ToString(System.Globalization.CultureInfo.InvariantCulture) + (bold ? ";1" : "");
            return "\u001b[" + attributes + "m" + text + "\u001b[0m";
        }
    }
}
