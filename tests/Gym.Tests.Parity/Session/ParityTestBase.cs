using Gym.Tests.Parity.Interop;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Python.Runtime;

namespace Gym.Tests.Parity.Session {
    /// <summary>
    ///     Base class of every engine-backed parity test: it checks that the session is up on the reference stack and fails
    ///     the test if it leaked a NumSharp ↔ numpy conversion.
    /// </summary>
    /// <remarks>
    ///     Port of NumSharp's <c>InteropTestBase</c>. GIL discipline carries over unchanged, spelled the pythonic skill's way:
    ///     every <see cref="PyObject"/> is created and disposed inside <see cref="PythonRuntime.Run(System.Action)"/> (pythonnet's
    ///     final decref needs the GIL), tests keep NumSharp arrays and plain CLR values outside it, and the harness types
    ///     (<see cref="Oracle.Outcome.CapturePython"/>, <see cref="PyRef"/>, <c>GymPair</c>) take the GIL themselves. Python is
    ///     reached through the <see cref="Python"/> tree only; there is no per-test scope to evaluate source in.
    /// </remarks>
    public abstract class ParityTestBase {
        /// <summary>The leak counters at the start of the test.</summary>
        private LeakGate.Baseline _leakBaseline;

        /// <summary>Whether <see cref="ParityInitialize"/> got past the session check, so the cleanup's leak gate applies.</summary>
        private bool _started;

        /// <summary>MSTest's context of the running test: output, result files and the outcome the cleanup reports on.</summary>
        public TestContext TestContext { get; set; } = null!;

        /// <summary>
        ///     The context of the test that is running now, for harness code that is not handed one (the artifacts writer), or
        ///     <see langword="null"/> between tests. MSTest's own <c>TestContext.Current</c> is experimental (MSTESTEXP) in 4.4.1;
        ///     a static slot is exact here because the assembly runs one test at a time (<c>[DoNotParallelize]</c>).
        /// </summary>
        internal static TestContext? CurrentContext { get; private set; }

        /// <summary>
        ///     Runs before each test: requires the session (see <see cref="GymPythonSession.EnsureReady"/>) and records the
        ///     leak baseline.
        /// </summary>
        /// <exception cref="AssertFailedException">Thrown when the session is not ready and the engine is required.</exception>
        /// <exception cref="AssertInconclusiveException">Thrown when the session is not ready and skipping was asked for.</exception>
        [TestInitialize]
        public void ParityInitialize() {
            CurrentContext = TestContext;
            GymPythonSession.EnsureReady();
            _leakBaseline = LeakGate.Capture();
            _started = true;
        }

        /// <summary>
        ///     Runs after each test: names the open defects of a failed <see cref="OpenBugsAttribute"/> test, and fails the test
        ///     unless the conversion counters returned to the baseline.
        /// </summary>
        /// <exception cref="AssertFailedException">Thrown when the test leaked an export or an import.</exception>
        [TestCleanup]
        public void ParityCleanup() {
            CurrentContext = null;
            if (!_started)
                return; // The session was not ready: the test already failed or was inconclusive in ParityInitialize.
            _started = false;

            if (TestContext.CurrentTestOutcome != UnitTestOutcome.Passed && OpenBugsAttribute.Find(GetType(), TestContext.TestName) is { } open)
                TestContext.WriteLine($"Known gap: [OpenBugs({string.Join(", ", open.Defects)})]. The expectation is Gymnasium's; see docs/plans/GYMNASIUM_PORT_PLAN.md, Appendix B.");

            if (!LeakGate.WaitForReturn(_leakBaseline, 12_000, out string counters))
                Assert.Fail($"The test leaked NumSharp <-> numpy conversions ({counters}). Dispose every PyObject and NDArray that crosses the bridge.");
        }

        /// <summary>
        ///     Reports Inconclusive unless a Python module imports, for cells that need an optional part of the stack. The
        ///     manifest's packages need no guard: missing ones already fail the run.
        /// </summary>
        /// <param name="module">The module to import, for example <c>mujoco</c>.</param>
        /// <exception cref="AssertInconclusiveException">Thrown when the import fails.</exception>
        protected static void SkipUnless(string module) {
            bool available = PythonRuntime.Run(() => {
                try {
                    // Cached by the runtime on success, so a later facade row reuses the same module object.
                    PythonRuntime.Import(module);
                    return true;
                } catch (PythonException) {
                    return false;
                }
            });

            if (!available)
                Assert.Inconclusive($"The Python module '{module}' is not installed.");
        }

        /// <summary>
        ///     Reports Inconclusive off the reference platform, for a byte-exact cell whose result depends on the platform's
        ///     libm (CartPole's <c>cos</c>/<c>sin</c>) or its native libraries. On the reference platform the cell stays strict;
        ///     elsewhere a difference is tracked, not a failure (port plan §14.9).
        /// </summary>
        /// <param name="cell">What depends on the platform, for the message.</param>
        /// <exception cref="AssertInconclusiveException">Thrown on any other platform.</exception>
        protected static void SkipOffReferencePlatform(string cell) {
            if (!GymPythonSession.OnReferencePlatform)
                Assert.Inconclusive($"{cell} is pinned to {GymPythonSession.Manifest?.ReferencePlatform}; this is {GymPythonSession.Platform}.");
        }
    }
}
