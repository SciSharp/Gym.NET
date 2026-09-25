using System;
using Gym.Tests.Parity.Oracle;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Python.Runtime;

namespace Gym.Tests.Parity.Session {
    /// <summary>
    ///     Base class of every engine-backed parity test: it checks that the session is up on the reference stack, gives the
    ///     test a Python scope with Gymnasium imported, and fails the test if it leaked a NumSharp ↔ numpy conversion.
    /// </summary>
    /// <remarks>
    ///     Port of NumSharp's <c>InteropTestBase</c>. GIL discipline carries over unchanged: every <see cref="PyObject"/> is
    ///     created and disposed under the GIL (pythonnet's final decref needs it), tests keep NumSharp arrays and plain CLR
    ///     values outside it, and the harness types (<see cref="PyGym"/>, <c>GymPair</c>) take the GIL themselves.
    /// </remarks>
    public abstract class ParityTestBase {
        /// <summary>The leak counters at the start of the test.</summary>
        private LeakGate.Baseline _leakBaseline;

        /// <summary>The test's Python scope, or <see langword="null"/> before initialization and after cleanup.</summary>
        private PyModule? _scope;

        /// <summary>MSTest's context of the running test: output, result files and the outcome the cleanup reports on.</summary>
        public TestContext TestContext { get; set; } = null!;

        /// <summary>
        ///     The context of the test that is running now, for harness code that is not handed one (the artifacts writer), or
        ///     <see langword="null"/> between tests. MSTest's own <c>TestContext.Current</c> is experimental (MSTESTEXP) in 4.4.1;
        ///     a static slot is exact here because the assembly runs one test at a time (<c>[DoNotParallelize]</c>).
        /// </summary>
        internal static TestContext? CurrentContext { get; private set; }

        /// <summary>
        ///     A scope with <c>np</c>, <c>gymnasium</c> (also as <c>gym</c>), <c>spaces</c>, <c>seeding</c> and
        ///     <c>gym_oracle</c> imported. Use it under the GIL (<see cref="Gil"/>); it is disposed after the test.
        /// </summary>
        /// <exception cref="InvalidOperationException">Thrown outside a test's lifetime.</exception>
        protected PyModule Scope => _scope ?? throw new InvalidOperationException("The Python scope exists only while a test runs.");

        /// <summary>
        ///     Runs before each test: requires the session (see <see cref="GymPythonSession.EnsureReady"/>), records the leak
        ///     baseline and creates the test's scope.
        /// </summary>
        [TestInitialize]
        public void ParityInitialize() {
            CurrentContext = TestContext;
            GymPythonSession.EnsureReady();
            _leakBaseline = LeakGate.Capture();
            using (Py.GIL()) {
                _scope = Py.CreateScope();
                _scope.Exec(
                    "import numpy as np\n" +
                    "import gymnasium\n" +
                    "import gymnasium as gym\n" +
                    "from gymnasium import spaces\n" +
                    "from gymnasium.utils import seeding\n" +
                    "import gym_oracle\n");
            }
        }

        /// <summary>
        ///     Runs after each test: disposes the scope, names the open defects of a failed <see cref="OpenBugsAttribute"/>
        ///     test, and fails the test unless the conversion counters returned to the baseline.
        /// </summary>
        /// <exception cref="AssertFailedException">Thrown when the test leaked an export or an import.</exception>
        [TestCleanup]
        public void ParityCleanup() {
            CurrentContext = null;
            if (_scope is null)
                return; // The session was not ready: the test already failed or was inconclusive in ParityInitialize.

            using (Py.GIL())
                _scope.Dispose();
            _scope = null;

            if (TestContext.CurrentTestOutcome != UnitTestOutcome.Passed && OpenBugsAttribute.Find(GetType(), TestContext.TestName) is { } open)
                TestContext.WriteLine($"Known gap: [OpenBugs({string.Join(", ", open.Defects)})]. The expectation is Gymnasium's; see docs/plans/GYMNASIUM_PORT_PLAN.md, Appendix B.");

            if (!LeakGate.WaitForReturn(_leakBaseline, 12_000, out string counters))
                Assert.Fail($"The test leaked NumSharp <-> numpy conversions ({counters}). Dispose every PyObject and NDArray that crosses the bridge.");
        }

        /// <summary>Takes the GIL for a block that touches <see cref="PyObject"/>s: <c>using (Gil()) { ... }</c>.</summary>
        /// <returns>The GIL state; disposing it releases the GIL.</returns>
        protected static IDisposable Gil() => Py.GIL();

        /// <summary>
        ///     Reports Inconclusive unless a Python module imports, for cells that need an optional part of the stack. The
        ///     manifest's packages need no guard: missing ones already fail the run.
        /// </summary>
        /// <param name="module">The module to import, for example <c>mujoco</c>.</param>
        /// <exception cref="AssertInconclusiveException">Thrown when the import fails.</exception>
        protected static void SkipUnless(string module) {
            bool available;
            using (Py.GIL()) {
                try {
                    using PyObject imported = Py.Import(module);
                    available = true;
                } catch (PythonException) {
                    available = false;
                }
            }

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

        /// <summary>Runs Python statements in <see cref="Scope"/>, taking the GIL.</summary>
        /// <param name="code">The statements.</param>
        /// <exception cref="PythonException">Thrown when the code raises.</exception>
        protected void Exec(string code) {
            using (Py.GIL())
                Scope.Exec(code);
        }

        /// <summary>Evaluates a Python expression in <see cref="Scope"/> and describes the result, taking the GIL.</summary>
        /// <param name="expression">The expression.</param>
        /// <returns>The result, described by <c>gym_oracle.describe</c>.</returns>
        /// <exception cref="PythonException">Thrown when the expression raises.</exception>
        protected ValueFacts Eval(string expression) {
            using (Py.GIL()) {
                using PyObject result = Scope.Eval(expression);
                return PyGym.Describe(result);
            }
        }
    }
}
