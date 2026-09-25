using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Python.Runtime;

namespace Gym.Tests.Parity.Session {
    /// <summary>
    ///     The one embedded CPython of the test process, with Gymnasium imported from <c>refs/Gymnasium</c> and the oracle
    ///     module (<c>Oracle/gym_oracle.py</c>) loaded. <c>[AssemblyInitialize]</c> starts it and <c>[AssemblyCleanup]</c>
    ///     stops it, because CPython and NumPy cannot be initialized again after <c>Py_Finalize</c> in the same process.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Port of NumSharp's <c>PythonSession</c> (<c>refs/NumSharp/test/NumSharp.Tests.Interop</c>), extended per
    ///     docs/plans/GYM_PARITY_INTEROP_TESTING.md §3–§4: a parity venv, deterministic interpreter settings, and a stack
    ///     check against <c>tools/parity/manifest.json</c>.
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

        /// <summary>The name the oracle module is registered under in <c>sys.modules</c>.</summary>
        public const string OracleModuleName = "gym_oracle";

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

        /// <summary>Whether <c>PythonEngine.Initialize</c> ran, so <see cref="Stop"/> must shut it down even if the bootstrap failed.</summary>
        private static bool _engineStarted;

        /// <summary>The oracle module, owned by the session and disposed under the GIL in <see cref="Stop"/>.</summary>
        private static PyObject? _oracle;

        /// <summary>Whether the engine started and Gymnasium and the oracle imported. Stack drift does not affect it.</summary>
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

        /// <summary>The loaded oracle module (<c>gym_oracle</c>). Use it under the GIL, and only after <see cref="EnsureReady"/>.</summary>
        /// <exception cref="InvalidOperationException">Thrown when the session is not available.</exception>
        internal static PyObject Oracle => _oracle ?? throw new InvalidOperationException($"The Python session is not available: {Reason}");

        /// <summary>Whether a missing engine fails tests; read once from <see cref="RequireEngineVariable"/>.</summary>
        private static bool RequireEngine => ReadFlag(RequireEngineVariable, defaultValue: true);

        /// <summary>Whether stack drift is tolerated; read from <see cref="AllowStackDriftVariable"/>.</summary>
        private static bool AllowStackDrift => ReadFlag(AllowStackDriftVariable, defaultValue: false);

        /// <summary>
        ///     Starts the engine: finds the checkout, the manifest and a Python, fixes the process state, initializes
        ///     pythonnet, imports Gymnasium from the submodule, loads the oracle and checks the stack.
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

                foreach (var (name, value) in DeterministicEnvironment)
                    NativeEnvironment.Set(name, value);

                Runtime.PythonDLL = Installation.LibraryPath;
                PythonEngine.Initialize();
                _engineStarted = true;
                // Initialize leaves this thread holding the GIL; release it, so that test threads can take it with Py.GIL().
                PythonEngine.BeginAllowThreads();

                Dictionary<string, string> live;
                using (Py.GIL()) {
                    Bootstrap(Repository, Installation);
                    live = ReadLiveStack();
                }

                Stack = CheckStack(Repository, Manifest, Installation, live);
                Available = true;
                Reason = "";
            } catch (Exception e) {
                Available = false;
                Reason = $"the session failed to start: {e.GetType().Name}: {e.Message}";
            }
        }

        /// <summary>
        ///     Shuts the engine down at the end of the run, after releasing the oracle module under the GIL. A failure is only
        ///     traced: the test results are already final, and an exception here would hide them behind a cleanup error.
        /// </summary>
        [AssemblyCleanup]
        public static void Stop() {
            if (!_engineStarted)
                return;

            try {
                using (Py.GIL()) {
                    _oracle?.Dispose();
                    _oracle = null;
                }

                // pythonnet 3.0.x stashes runtime state with BinaryFormatter, which .NET 8+ removed; NoopFormatter opts out.
                RuntimeData.FormatterType = typeof(NoopFormatter);
                PythonEngine.Shutdown();
            } catch (Exception e) {
                Trace.WriteLine($"PythonEngine.Shutdown reported: {e.GetType().Name}: {e.Message}");
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
        ///     Runs inside the fresh interpreter, under the GIL: stops bytecode from being written into the submodule, adopts
        ///     the probed interpreter's <c>sys.path</c> with <c>refs/Gymnasium</c> in front, imports Gymnasium, and loads the
        ///     oracle into <c>sys.modules</c>.
        /// </summary>
        /// <param name="repository">The checkout.</param>
        /// <param name="installation">The embedded Python, whose probed <c>sys.path</c> is adopted when known.</param>
        /// <exception cref="PythonException">Thrown when Gymnasium or the oracle fails to import.</exception>
        /// <exception cref="InvalidOperationException">Thrown when the oracle resource is missing from the test assembly.</exception>
        private static void Bootstrap(RepositoryLayout repository, PythonInstallation installation) {
            using (PyModule scope = Py.CreateScope()) {
                using (var gymnasiumRoot = new PyString(repository.GymnasiumRoot))
                    scope.Set("gymnasium_root", gymnasiumRoot);
                if (installation.SysPath is null) {
                    // Bound in Python rather than through PyObject.None: that is pythonnet's shared instance, and the using
                    // that disposes every other argument here would dispose it for the whole process.
                    scope.Exec("probed_path = None");
                } else {
                    using PyList probedPath = ToPythonList(installation.SysPath);
                    scope.Set("probed_path", probedPath);
                }
                // The interpreter's default sys.path is the base installation's; the probed one is what the chosen executable
                // (a venv's, typically) really uses. refs/Gymnasium goes first so that no installed gymnasium can shadow it.
                scope.Exec(
                    "import os, sys\n" +
                    "sys.dont_write_bytecode = True\n" +
                    "if probed_path is not None:\n" +
                    "    sys.path[:] = list(probed_path)\n" +
                    "_root = os.path.normcase(os.path.abspath(gymnasium_root))\n" +
                    "sys.path[:] = [gymnasium_root] + [p for p in sys.path if os.path.normcase(os.path.abspath(p)) != _root]\n" +
                    "import numpy, gymnasium\n");
            }

            string source = ReadOracleSource();
            _oracle = PyModule.FromString(OracleModuleName, source);
            // Registered so that test scopes can `import gym_oracle` and Python code can reach the same module object.
            using PyObject sys = Py.Import("sys");
            using PyObject modules = sys.GetAttr("modules");
            modules.SetItem(OracleModuleName, _oracle);
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

        /// <summary>Reads <c>Oracle/gym_oracle.py</c> from the assembly's resources.</summary>
        /// <returns>The module's source text.</returns>
        /// <exception cref="InvalidOperationException">Thrown when the resource is missing, i.e. the csproj no longer embeds it.</exception>
        private static string ReadOracleSource() {
            const string resource = "Gym.Tests.Parity.Oracle.gym_oracle.py";
            using Stream? stream = typeof(GymPythonSession).Assembly.GetManifestResourceStream(resource);
            if (stream is null)
                throw new InvalidOperationException($"The test assembly does not embed {resource}; see Gym.Tests.Parity.csproj.");
            using var reader = new StreamReader(stream);
            return reader.ReadToEnd();
        }

        /// <summary>Calls <c>gym_oracle.stack()</c>. Must run under the GIL.</summary>
        /// <returns>The live values by key.</returns>
        private static Dictionary<string, string> ReadLiveStack() {
            var live = new Dictionary<string, string>(StringComparer.Ordinal);
            using PyObject pairs = Oracle.InvokeMethod("stack");
            using var tuple = PyTuple.AsTuple(pairs);
            long count = tuple.Length();
            for (int i = 0; i < count; i++) {
                using PyObject pair = tuple[i];
                using PyObject key = pair[0];
                using PyObject value = pair[1];
                live[key.As<string>()] = value.As<string>();
            }

            return live;
        }

        /// <summary>
        ///     Compares the live stack with the manifest and adds the structural checks: Gymnasium comes from the submodule,
        ///     the submodule is unmodified, string hashing is deterministic. Informational rows (where Python came from, the
        ///     platform) always match; they make the report self-describing.
        /// </summary>
        /// <param name="repository">The checkout.</param>
        /// <param name="manifest">The pins.</param>
        /// <param name="installation">The embedded Python.</param>
        /// <param name="live">What <c>gym_oracle.stack()</c> reported.</param>
        /// <returns>Every check, pins first.</returns>
        private static StackReport CheckStack(RepositoryLayout repository, StackManifest manifest, PythonInstallation installation, IReadOnlyDictionary<string, string> live) {
            var items = new List<StackItem>();
            foreach (var (key, expected) in manifest.Stack) {
                string actual;
                if (key == "gymnasium.commit")
                    actual = repository.ReadGymnasiumHead(out string detail) ?? $"unreadable ({detail})";
                else
                    actual = live.TryGetValue(key, out string? value) ? value : "not reported by gym_oracle.stack()";
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

            string hash = live.TryGetValue("python.hash_randomization", out string? h) ? h : "not reported";
            items.Add(new StackItem("python.hash_randomization", "0", hash, hash == "0"));

            items.Add(new StackItem("python.source", "", installation.Source, Matches: true));
            items.Add(new StackItem("python.library", "", installation.LibraryPath, Matches: true));
            // Which site-packages NumPy came from: with a venv it must be the venv's, not the base installation's.
            string numpy = live.TryGetValue("numpy.file", out string? n) ? Path.GetDirectoryName(n) ?? n : "not reported";
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
