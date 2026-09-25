using System;
using System.IO;
using System.Linq;
using Gym.Tests.Parity.Session;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests.Parity.Frames {
    /// <summary>
    ///     Where a failed comparison leaves its evidence (frame PNGs, dumps): <c>GYMNET_PARITY_ARTIFACTS</c> when set,
    ///     otherwise the test's results directory, one subdirectory per test and label. Files are attached to the test result
    ///     too, so CI keeps them with the failure.
    /// </summary>
    internal static class ParityArtifacts {
        /// <summary>The environment variable that overrides the artifacts root (plan §4).</summary>
        public const string Variable = "GYMNET_PARITY_ARTIFACTS";

        /// <summary>Creates, and returns, the directory for one comparison of the running test.</summary>
        /// <param name="label">What was compared, for example <c>cartpole-state-1</c>; made file-name safe.</param>
        /// <returns>The directory's full path.</returns>
        public static string DirectoryFor(string label) {
            TestContext? context = ParityTestBase.CurrentContext;
            string? configured = Environment.GetEnvironmentVariable(Variable);
            string root = !string.IsNullOrWhiteSpace(configured)
                ? configured
                : context?.TestResultsDirectory ?? Path.Combine(AppContext.BaseDirectory, "ParityArtifacts");
            // The display name carries a data row's arguments, so the rows of one data-driven test do not overwrite each other.
            string test = context?.TestDisplayName ?? context?.TestName ?? "outside-a-test";
            string directory = Path.Combine(root, "parity", Safe(test), Safe(label));
            Directory.CreateDirectory(directory);
            return directory;
        }

        /// <summary>Attaches a file to the running test's result, when there is one.</summary>
        /// <param name="path">The file.</param>
        public static void Attach(string path) {
            ParityTestBase.CurrentContext?.AddResultFile(path);
        }

        /// <summary>Replaces the characters a file name cannot hold, and trims it to a length every file system accepts.</summary>
        /// <param name="name">The raw name.</param>
        /// <returns>A file-name-safe name.</returns>
        private static string Safe(string name) {
            char[] invalid = Path.GetInvalidFileNameChars();
            string safe = new(name.Select(c => invalid.Contains(c) || c == ' ' ? '_' : c).ToArray());
            return safe.Length > 80 ? safe.Substring(0, 80) : safe;
        }
    }
}
