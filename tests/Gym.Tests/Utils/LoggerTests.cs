using System.IO;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests.Utils {
    /// <summary>
    ///     <see cref="Logger"/> and <see cref="Warnings"/> behave as <c>gymnasium.logger</c> on Python's <c>warnings</c> module:
    ///     colourised prefixes, Python's default filters (once per call site for user warnings, once per message for
    ///     deprecations), <see cref="Logger.MinLevel"/>, and recording.
    /// </summary>
    /// <remarks>The shown-warnings registry and the output are process-wide, so each test resets them and restores the output.</remarks>
    [TestClass]
    [DoNotParallelize]
    public class LoggerTests {
        /// <summary>Where warnings went before the test.</summary>
        private TextWriter _output;

        /// <summary>Captures the warning output and forgets earlier warnings, so each test sees first-time behaviour.</summary>
        [TestInitialize]
        public void CaptureOutput() {
            _output = Warnings.Output;
            Warnings.Output = new StringWriter();
            Warnings.ResetRegistry();
            Logger.MinLevel = Logger.WARN;
        }

        /// <summary>Restores the output and the level.</summary>
        [TestCleanup]
        public void RestoreOutput() {
            Warnings.Output = _output;
            Logger.MinLevel = Logger.WARN;
            Warnings.ResetRegistry();
        }

        /// <summary><c>colorize(text, color, bold, highlight)</c> spells the ANSI sequence exactly as Python's does.</summary>
        [TestMethod]
        public void Colorize_SpellsTheAnsiSequence() {
            Assert.AreEqual("\u001b[33mWARN: x\u001b[0m", Logger.Colorize("WARN: x", "yellow"));
            Assert.AreEqual("\u001b[31;1mboom\u001b[0m", Logger.Colorize("boom", "red", bold: true));
            Assert.AreEqual("\u001b[42mok\u001b[0m", Logger.Colorize("ok", "green", highlight: true));
        }

        /// <summary>
        ///     A user warning is shown the first time a call site issues it and suppressed after that, as Python's
        ///     <c>default</c> filter does; another call site shows it again.
        /// </summary>
        [TestMethod]
        public void Warn_IsShownOncePerCallSite() {
            for (int i = 0; i < 3; i++)
                Logger.Warn("careful");
            Logger.Warn("careful");

            string[] lines = ShownLines();
            Assert.AreEqual(2, lines.Length, "One line for the loop's call site, one for the second call site.");
            StringAssert.EndsWith(lines[0], ": UserWarning: \u001b[33mWARN: careful\u001b[0m");
        }

        /// <summary>A deprecation is shown once per message, wherever it is issued from (Gymnasium's <c>once</c> filter).</summary>
        [TestMethod]
        public void Deprecation_IsShownOncePerMessage() {
            Logger.Deprecation("old");
            Logger.Deprecation("old");
            Logger.Deprecation("older");

            string[] lines = ShownLines();
            Assert.AreEqual(2, lines.Length);
            StringAssert.Contains(lines[0], ": DeprecationWarning: \u001b[33mWARN: old\u001b[0m");
        }

        /// <summary>Raising <see cref="Logger.MinLevel"/> above <see cref="Logger.WARN"/> silences warnings but not errors.</summary>
        [TestMethod]
        public void MinLevel_SilencesLowerLevels() {
            Logger.MinLevel = Logger.ERROR;
            Logger.Warn("quiet");
            Logger.Error("loud");

            string[] lines = ShownLines();
            Assert.AreEqual(1, lines.Length);
            StringAssert.EndsWith(lines[0], ": UserWarning: \u001b[31mERROR: loud\u001b[0m");
        }

        /// <summary>A catcher records every warning, repeats included, and shows none; leaving it lets warnings show again.</summary>
        [TestMethod]
        public void Catch_RecordsEverything_AndShowsNothing() {
            using (WarningCatcher catcher = Warnings.Catch()) {
                for (int i = 0; i < 3; i++)
                    Logger.Warn("recorded");
                Assert.AreEqual(3, catcher.Records.Count);
                Assert.AreEqual(WarningCategory.UserWarning, catcher.Records[0].Category);
                Assert.AreEqual("\u001b[33mWARN: recorded\u001b[0m", catcher.Records[0].Message);
            }

            Assert.AreEqual(0, ShownLines().Length, "Nothing is shown while a catcher records.");
            Logger.Warn("recorded");
            Assert.AreEqual(1, ShownLines().Length, "After the block the warning shows again, as Python's registry is reset.");
        }

        /// <summary>The lines written to the captured output.</summary>
        /// <returns>The non-empty lines.</returns>
        private static string[] ShownLines() {
            return Warnings.Output!.ToString()!.Split(new[] { '\r', '\n' }, System.StringSplitOptions.RemoveEmptyEntries);
        }
    }
}
