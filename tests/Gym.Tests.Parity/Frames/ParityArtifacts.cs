using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.Versioning;
using Gym.Tests.Parity.Session;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests.Parity.Frames {
    /// <summary>
    ///     Where a comparison leaves its evidence (frame PNGs, dumps; a failed one's, and every MuJoCo frame cell's, DR-71):
    ///     <c>GYMNET_PARITY_ARTIFACTS</c> when set, in a folder per target framework (<c>net10.0</c>), otherwise the test's
    ///     results directory; below either, one subdirectory per test and label. Files are attached to the test result too,
    ///     so CI keeps them with the failure.
    /// </summary>
    internal static class ParityArtifacts {
        /// <summary>The environment variable that overrides the artifacts root (plan §4).</summary>
        public const string Variable = "GYMNET_PARITY_ARTIFACTS";

        /// <summary>
        ///     The target framework this test assembly was built for, as a folder name (<c>net8.0</c>, <c>net10.0</c>,
        ///     <c>net11.0</c>), read from its <see cref="TargetFrameworkAttribute"/>; <c>unknown-framework</c> without one.
        /// </summary>
        /// <remarks>
        ///     The assembly's own framework, not the runtime's version: a build for one framework may run on a newer runtime
        ///     (roll-forward), and its artifacts still belong to its build.
        /// </remarks>
        private static readonly string Framework = FrameworkFolder();

        /// <summary>Creates, and returns, the directory for one comparison of the running test.</summary>
        /// <param name="label">What was compared, for example <c>cartpole-state-1</c>; made file-name safe.</param>
        /// <returns>The directory's full path.</returns>
        /// <exception cref="IOException">The directory could not be created.</exception>
        /// <exception cref="UnauthorizedAccessException">The process may not create the directory.</exception>
        public static string DirectoryFor(string label) {
            TestContext? context = ParityTestBase.CurrentContext;
            string? configured = Environment.GetEnvironmentVariable(Variable);
            // A configured root is shared by every test host of the run, and `dotnet test` runs a multi-targeted project's
            // frameworks in parallel, each running the same cells: without a folder per framework they wrote the same files
            // at once and failed with an IOException (measured 2026-11-13: six MuJoCo frame cells, which write their PNGs on
            // every measurement). The default, each test host's own results directory, is per framework already.
            string root = !string.IsNullOrWhiteSpace(configured)
                ? Path.Combine(configured, Framework)
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

        /// <summary>Reads <see cref="Framework"/> from the assembly's <see cref="TargetFrameworkAttribute"/>.</summary>
        /// <returns>
        ///     <c>net</c> and the framework's version (<c>.NETCoreApp,Version=v10.0</c> gives <c>net10.0</c>), or
        ///     <c>unknown-framework</c> when the compiler wrote no attribute.
        /// </returns>
        /// <exception cref="ArgumentException">The attribute's name is not a framework name; the compiler always writes one.</exception>
        private static string FrameworkFolder() {
            string? name = typeof(ParityArtifacts).Assembly.GetCustomAttribute<TargetFrameworkAttribute>()?.FrameworkName;
            if (name is null)
                return "unknown-framework";
            Version version = new FrameworkName(name).Version;
            return $"net{version.Major}.{version.Minor}";
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
