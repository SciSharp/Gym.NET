using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Python.Runtime;

namespace Gym.Tests.Parity.Session {
    /// <summary>
    ///     The one embedded CPython of the test process, with Gymnasium imported from <c>refs/Gymnasium</c>.
    ///     <c>[AssemblyInitialize]</c> starts it and <c>[AssemblyCleanup]</c> stops it, because CPython and NumPy cannot be
    ///     initialized again after <c>Py_Finalize</c> in the same process.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Adapted from NumSharp's <c>PythonSession</c> (<c>refs/NumSharp/test/NumSharp.Tests.Interop</c>), extended per
    ///     docs/plans/GYM_PARITY_INTEROP_TESTING.md §3–§4: a parity venv, deterministic interpreter settings, and a stack
    ///     check against <c>tools/parity/manifest.json</c>. The engine itself is <see cref="PythonRuntime"/>, the pythonic
    ///     skill's backend; everything the session asks Python goes through the <see cref="Python"/> tree.
    ///     </para>
    ///     <para>
    ///     Failures here never throw out of <see cref="Start"/>: an exception from an assembly initializer would abort the run
    ///     with no usable message. The reason is recorded instead, and <see cref="EnsureReady"/> reports it in every test.
    ///     </para>
    /// </remarks>
    [TestClass]
    public sealed class GymPythonSession {
        /// <summary>
        ///     Whether a missing engine fails the run (the default) rather than reporting Inconclusive. A self-skipping suite
        ///     proves nothing and still shows green, so skipping must be asked for: set it to <c>0</c>, <c>false</c>,
        ///     <c>no</c> or <c>off</c> on a machine without the reference stack.
        /// </summary>
        public const string RequireEngineVariable = "GYMNET_PYTHONNET_REQUIRE_ENGINE";

        /// <summary>
        ///     When set (<c>1</c>), a stack that differs from the manifest no longer fails every test: the tests run, and
        ///     only the stack check itself reports Inconclusive. For exploring on another stack; such results are not evidence.
        /// </summary>
        public const string AllowStackDriftVariable = "GYMNET_PARITY_ALLOW_STACK_DRIFT";

        /// <summary>
        ///     Process state that must be fixed before the interpreter starts: a fixed string hash (set iteration order), no
        ///     bytecode written into the <c>refs/Gymnasium</c> submodule, headless pygame, and single-threaded BLAS so a
        ///     NumPy reduction's summation order cannot vary between runs.
        /// </summary>
        private static readonly (string Name, string Value)[] DeterministicEnvironment = {
            ("PYTHONHASHSEED", "0"),
            ("PYTHONDONTWRITEBYTECODE", "1"),
            ("PYGAME_HIDE_SUPPORT_PROMPT", "1"),
            ("SDL_VIDEODRIVER", "dummy"),
            ("SDL_AUDIODRIVER", "dummy"),
            ("OMP_NUM_THREADS", "1"),
            ("OPENBLAS_NUM_THREADS", "1"),
            ("MKL_NUM_THREADS", "1"),
        };

        /// <summary>
        ///     The bootstrap, run once in the fresh interpreter (the pythonic skill's snippet flavor, since it must read as the
        ///     Python it is). The interpreter's default <c>sys.path</c> is the base installation's; the probed one is what the
        ///     chosen executable (a venv's, typically) really uses, so it is adopted when known. <c>refs/Gymnasium</c> goes
        ///     first, and any other spelling of it is dropped, so that no installed <c>gymnasium</c> can shadow the submodule.
        /// </summary>
        private const string BootstrapStatements =
            "import os, sys\n" +
            "sys.dont_write_bytecode = True\n" +
            "if probed_path is not None:\n" +
            "    sys.path[:] = list(probed_path)\n" +
            "_root = os.path.normcase(os.path.abspath(gymnasium_root))\n" +
            "sys.path[:] = [gymnasium_root] + [p for p in sys.path if os.path.normcase(os.path.abspath(p)) != _root]\n" +
            "import numpy, gymnasium\n" +
            "imported = gymnasium.__file__\n";

        /// <summary>Whether the engine started and Gymnasium imported. Stack drift does not affect it.</summary>
        public static bool Available { get; private set; }

        /// <summary>Why <see cref="Available"/> is <see langword="false"/>; empty once the session is up.</summary>
        public static string Reason { get; private set; } = "the session has not started";

        /// <summary>The checkout the suite runs from, or <see langword="null"/> when it could not be found.</summary>
        public static RepositoryLayout? Repository { get; private set; }

        /// <summary>The pinned reference stack, or <see langword="null"/> when the manifest could not be read.</summary>
        public static StackManifest? Manifest { get; private set; }

        /// <summary>The embedded Python, or <see langword="null"/> when none was found.</summary>
        public static PythonInstallation? Installation { get; private set; }

        /// <summary>The comparison of the live stack with the manifest; set whenever <see cref="Available"/> is.</summary>
        public static StackReport? Stack { get; private set; }

        /// <summary>
        ///     The runtime identifier of this process (<c>win-x64</c>, <c>linux-x64</c>, <c>osx-arm64</c>), in the form
        ///     <see cref="StackManifest.ReferencePlatform"/> uses. The process architecture counts, not the OS's: it decides
        ///     which libm and which instructions compute the .NET side.
        /// </summary>
        public static string Platform {
            get {
                string os = OperatingSystem.IsWindows() ? "win" : OperatingSystem.IsMacOS() ? "osx" : "linux";
                return $"{os}-{RuntimeInformation.ProcessArchitecture.ToString().ToLowerInvariant()}";
            }
        }

        /// <summary>
        ///     Whether this process runs on the manifest's reference platform, where byte-exact results that depend on the
        ///     platform's libm are authoritative (port plan §14.9).
        /// </summary>
        public static bool OnReferencePlatform => Manifest != null && string.Equals(Platform, Manifest.ReferencePlatform, StringComparison.Ordinal);

        /// <summary>Whether a missing engine fails tests; read once from <see cref="RequireEngineVariable"/>.</summary>
        private static bool RequireEngine => ReadFlag(RequireEngineVariable, defaultValue: true);

        /// <summary>Whether stack drift is tolerated; read from <see cref="AllowStackDriftVariable"/>.</summary>
        private static bool AllowStackDrift => ReadFlag(AllowStackDriftVariable, defaultValue: false);

        /// <summary>
        ///     Starts the engine: finds the checkout, the manifest and a Python, fixes the process state, initializes the
        ///     runtime, imports Gymnasium from the submodule and checks the stack.
        /// </summary>
        /// <param name="context">The run's context (unused; MSTest requires the parameter).</param>
        [AssemblyInitialize]
        public static void Start(TestContext context) {
            try {
                Repository = RepositoryLayout.TryDiscover(AppContext.BaseDirectory, out string reason);
                if (Repository is null) {
                    Reason = reason;
                    return;
                }

                Manifest = StackManifest.Load(Repository.ManifestPath);
                Installation = PythonDiscovery.Discover(Repository, out reason);
                if (Installation is null) {
                    Reason = reason;
                    return;
                }

                // Through the C runtime, before the interpreter exists: CPython reads its settings once, at initialization.
                foreach (var (name, value) in DeterministicEnvironment)
                    NativeEnvironment.Set(name, value);

                PythonRuntime.Initialize(Installation.LibraryPath);
                RepositoryLayout repository = Repository;
                PythonInstallation installation = Installation;
                Dictionary<string, string> live = PythonRuntime.Run(() => {
                    Bootstrap(repository, installation);
                    return ReadLiveStack();
                });

                Stack = CheckStack(Repository, Manifest, Installation, live);
                Available = true;
                Reason = "";
            } catch (Exception e) {
                Available = false;
                Reason = $"the session failed to start: {e.GetType().Name}: {e.Message}";
            }
        }

        /// <summary>
        ///     Shuts the engine down at the end of the run. A failure is only traced: the test results are already final, and
        ///     an exception here would hide them behind a cleanup error.
        /// </summary>
        [AssemblyCleanup]
        public static void Stop() {
            if (!PythonRuntime.IsInitialized)
                return;

            try {
                PythonRuntime.Shutdown();
            } catch (Exception e) {
                Trace.WriteLine($"PythonRuntime.Shutdown reported: {e.GetType().Name}: {e.Message}");
            }
        }

        /// <summary>
        ///     Makes a test fail (or report Inconclusive, where configured) unless the session is up on the reference stack.
        ///     <see cref="ParityTestBase"/> calls it before every test.
        /// </summary>
        /// <exception cref="AssertFailedException">
        ///     Thrown when the engine is unavailable and <see cref="RequireEngineVariable"/> is on, or when the stack drifted and
        ///     <see cref="AllowStackDriftVariable"/> is off.
        /// </exception>
        /// <exception cref="AssertInconclusiveException">Thrown when the engine is unavailable and <see cref="RequireEngineVariable"/> is off.</exception>
        public static void EnsureReady() {
            if (!Available) {
                if (RequireEngine)
                    Assert.Fail(
                        $"The embedded Python did not start, so every parity test would have been skipped instead of proving anything. " +
                        $"Reason: {Reason}. On a machine without the reference stack, set {RequireEngineVariable}=0 to report Inconclusive " +
                        "instead; tools/parity/setup_venv.ps1 (or .sh) creates the stack.");
                Assert.Inconclusive($"The embedded Python is unavailable: {Reason}");
            }

            if (Stack!.Drift.Count > 0 && !AllowStackDrift)
                Assert.Fail(
                    "The embedded stack differs from tools/parity/manifest.json, and a live oracle on another stack proves nothing.\n" +
                    Stack.Format() +
                    $"Fix the stack (tools/parity/setup_venv.ps1), or set {AllowStackDriftVariable}=1 to run the tests on it anyway.");
        }

        /// <summary>
        ///     Runs <see cref="BootstrapStatements"/> in the fresh interpreter, under the GIL: stops bytecode from being
        ///     written into the submodule, adopts the probed interpreter's <c>sys.path</c> with <c>refs/Gymnasium</c> in front,
        ///     and imports NumPy and Gymnasium.
        /// </summary>
        /// <param name="repository">The checkout.</param>
        /// <param name="installation">The embedded Python, whose probed <c>sys.path</c> is adopted when known.</param>
        /// <exception cref="PythonException">Thrown when NumPy or Gymnasium fails to import.</exception>
        private static void Bootstrap(RepositoryLayout repository, PythonInstallation installation) {
            using var gymnasiumRoot = new PyString(repository.GymnasiumRoot);
            // None through PyConvert (a new reference), never PyObject.None: that is pythonnet's shared instance, and this
            // using would dispose it for the whole process.
            using PyObject probedPath = installation.SysPath is null ? PyConvert.ToPython(null) : ToPythonList(installation.SysPath);
            using PyObject imported = Python.evalStmts(BootstrapStatements, "imported", ("gymnasium_root", gymnasiumRoot), ("probed_path", probedPath));
        }

        /// <summary>Converts the probed <c>sys.path</c> to a Python list. Must run under the GIL.</summary>
        /// <param name="entries">The entries, in order.</param>
        /// <returns>A new list the caller owns and disposes.</returns>
        private static PyList ToPythonList(IReadOnlyList<string> entries) {
            var list = new PyList();
            foreach (string entry in entries) {
                using var item = new PyString(entry);
                list.Append(item);
            }

            return list;
        }

        /// <summary>
        ///     Reads the live versions the manifest pins, plus where <c>gymnasium</c> and <c>numpy</c> were imported from and
        ///     whether string hashing is deterministic. A package that fails to import is reported as <c>missing (...)</c>
        ///     rather than thrown, so the report lists every drift at once instead of stopping at the first. Must run under the
        ///     GIL.
        /// </summary>
        /// <returns>The live values by key, in the manifest's spelling.</returns>
        /// <exception cref="PythonException">Thrown when reading a core value (Gymnasium's, NumPy's, the interpreter's) fails.</exception>
        private static Dictionary<string, string> ReadLiveStack() {
            var live = new Dictionary<string, string>(StringComparer.Ordinal) {
                ["gymnasium.version"] = Python.gymnasium.__version__,
                ["gymnasium.file"] = Python.gymnasium.__file__,
                ["python.version"] = Python.platform.python_version_t(),
                ["python.hash_randomization"] = Python.sys.flags.hash_randomization.ToString(System.Globalization.CultureInfo.InvariantCulture),
                ["numpy.version"] = Python.numpy.__version__,
                ["numpy.file"] = Python.numpy.__file__,
            };

            // pygame and Box2D are optional to the interpreter but pinned by the manifest: their absence is a drift to
            // report with the rest, not a reason to abort the session.
            try {
                live["pygame.version"] = Python.pygame.version.ver;
                live["pygame.sdl"] = Python.pygame.version.SDL;
            } catch (PythonException error) {
                live["pygame.version"] = live["pygame.sdl"] = Missing(error);
            }

            try {
                live["box2d.version"] = Python.Box2D.__version__;
            } catch (PythonException error) {
                live["box2d.version"] = Missing(error);
            }

            return live;
        }

        /// <summary>How the stack report shows a package that failed to import.</summary>
        /// <param name="error">The import's error; its <c>Type</c> is the exception's own field and is not disposed.</param>
        /// <returns><c>missing (Type: message)</c>.</returns>
        private static string Missing(PythonException error) => $"missing ({ValueFacts.QualifiedName(error.Type)}: {error.Message})";

        /// <summary>
        ///     Compares the live stack with the manifest and adds the structural checks: Gymnasium comes from the submodule,
        ///     the submodule is unmodified, string hashing is deterministic. Informational rows (where Python came from, the
        ///     platform) always match; they make the report self-describing.
        /// </summary>
        /// <param name="repository">The checkout.</param>
        /// <param name="manifest">The pins.</param>
        /// <param name="installation">The embedded Python.</param>
        /// <param name="live">What <see cref="ReadLiveStack"/> reported.</param>
        /// <returns>Every check, pins first.</returns>
        private static StackReport CheckStack(RepositoryLayout repository, StackManifest manifest, PythonInstallation installation, IReadOnlyDictionary<string, string> live) {
            var items = new List<StackItem>();
            foreach (var (key, expected) in manifest.Stack) {
                string actual;
                if (key == "gymnasium.commit")
                    actual = repository.ReadGymnasiumHead(out string detail) ?? $"unreadable ({detail})";
                else
                    actual = live.TryGetValue(key, out string? value) ? value : "not read by the session";
                items.Add(new StackItem(key, expected, actual, StackManifest.Matches(expected, actual)));
            }

            // A pip-installed gymnasium earlier on sys.path would make every comparison meaningless, however well pinned.
            string file = live.TryGetValue("gymnasium.file", out string? f) ? f : "";
            string origin = Path.GetDirectoryName(file) ?? "";
            items.Add(new StackItem("gymnasium.origin", repository.GymnasiumPackage, origin, SamePath(origin, repository.GymnasiumPackage)));

            if (repository.TryListGymnasiumChanges(out string changes))
                items.Add(new StackItem("gymnasium.clean", "no local modifications", changes.Length == 0 ? "no local modifications" : $"modified: {changes}", changes.Length == 0));
            else
                items.Add(new StackItem("gymnasium.clean", "no local modifications", $"unverified: {changes}", Matches: true));

            string hash = live.TryGetValue("python.hash_randomization", out string? h) ? h : "not read";
            items.Add(new StackItem("python.hash_randomization", "0", hash, hash == "0"));

            items.Add(new StackItem("python.source", "", installation.Source, Matches: true));
            items.Add(new StackItem("python.library", "", installation.LibraryPath, Matches: true));
            // Which site-packages NumPy came from: with a venv it must be the venv's, not the base installation's.
            string numpy = live.TryGetValue("numpy.file", out string? n) ? Path.GetDirectoryName(n) ?? n : "not read";
            items.Add(new StackItem("numpy.origin", "", numpy, Matches: true));
            items.Add(new StackItem("platform", manifest.ReferencePlatform, Platform, Matches: true));
            return new StackReport(items);
        }

        /// <summary>Compares two paths the way the file system does: case-insensitively on Windows, exactly elsewhere.</summary>
        /// <param name="left">A path.</param>
        /// <param name="right">Another path.</param>
        /// <returns><see langword="true"/> when both resolve to the same full path.</returns>
        private static bool SamePath(string left, string right) {
            if (left.Length == 0 || right.Length == 0)
                return false;
            var comparison = OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
            return string.Equals(Path.GetFullPath(left).TrimEnd(Path.DirectorySeparatorChar), Path.GetFullPath(right).TrimEnd(Path.DirectorySeparatorChar), comparison);
        }

        /// <summary>Reads a boolean switch the way NumSharp reads its own: <c>0</c>, <c>false</c>, <c>no</c> and <c>off</c> are false.</summary>
        /// <param name="name">The environment variable.</param>
        /// <param name="defaultValue">The value when the variable is unset or empty.</param>
        /// <returns>The switch's value.</returns>
        private static bool ReadFlag(string name, bool defaultValue) {
            string? value = Environment.GetEnvironmentVariable(name);
            if (string.IsNullOrWhiteSpace(value))
                return defaultValue;
            return value.Trim().ToLowerInvariant() is not ("0" or "false" or "no" or "off");
        }
    }
}
