using System;
using System.IO;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Session;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace Gym.Tests.Parity.Tests.Harness {
    /// <summary>
    ///     The session's own guarantees: the oracle is the pinned Gymnasium on the pinned stack, the interpreter is
    ///     deterministic, and the two byte readers the whole suite relies on agree on a known array.
    /// </summary>
    [TestClass]
    public sealed class SessionTests : ParityTestBase {
        /// <summary>
        ///     The embedded stack matches <c>tools/parity/manifest.json</c> (Gymnasium commit, Python, NumPy, pygame-ce, SDL,
        ///     Box2D) and the structural checks hold. The full table is written to the test output either way, so every run
        ///     records the stack it proved parity on.
        /// </summary>
        [TestMethod]
        public void Stack_MatchesTheManifest() {
            StackReport stack = GymPythonSession.Stack!;
            TestContext.WriteLine(stack.Format());
            if (stack.Drift.Count > 0)
                Assert.Inconclusive($"The stack drifted and {GymPythonSession.AllowStackDriftVariable} is set; these results are not evidence:\n{stack.Format()}");
        }

        /// <summary>
        ///     <c>import gymnasium</c> resolves to <c>refs/Gymnasium</c>, not to an installed copy, and reports the pinned
        ///     version. An installed Gymnasium earlier on <c>sys.path</c> would make every comparison meaningless.
        /// </summary>
        [TestMethod]
        public void Gymnasium_IsImportedFromTheSubmodule() {
            ValueFacts file = Eval("gymnasium.__file__");
            string expected = Path.Combine(GymPythonSession.Repository!.GymnasiumPackage, "__init__.py");
            Assert.AreEqual(Path.GetFullPath(expected), Path.GetFullPath(file.Text), ignoreCase: OperatingSystem.IsWindows(), "gymnasium must be imported from refs/Gymnasium.");
            Assert.AreEqual(GymPythonSession.Manifest!.Stack["gymnasium.version"], Eval("gymnasium.__version__").Text);
        }

        /// <summary>
        ///     The interpreter runs with the deterministic settings the session applies through the C runtime before
        ///     <c>Py_Initialize</c>: no string-hash randomization, no bytecode written into the submodule, headless pygame.
        /// </summary>
        [TestMethod]
        public void Interpreter_RunsWithTheDeterministicSettings() {
            Exec("import os, sys");
            Assert.AreEqual(0, (int)Eval("sys.flags.hash_randomization").Integer, "PYTHONHASHSEED=0 must reach Py_Initialize.");
            Assert.IsTrue(Eval("sys.dont_write_bytecode").Boolean, "Bytecode must never be written into refs/Gymnasium.");
            Assert.AreEqual("dummy", Eval("os.environ.get('SDL_VIDEODRIVER')").Text, "pygame must stay headless.");
            Assert.AreEqual("1", Eval("os.environ.get('OPENBLAS_NUM_THREADS')").Text, "NumPy's BLAS must run single-threaded.");
        }

        /// <summary>
        ///     The two byte readers agree: an array exported zero-copy to NumPy (NumSharp's <c>ToNumpy</c>) and described by
        ///     numpy's <c>tobytes</c> equals the same array described by NumSharp's own <c>data.tobytes</c>, for a strided,
        ///     negative-step view. Every comparison in the suite rests on this; it also exercises the leak gate with a live export.
        /// </summary>
        [TestMethod]
        public void ByteReaders_AgreeOnAnExportedView() {
            using NDArray values = np.arange(24.0).reshape(4, 6);
            using NDArray view = values["::-1, 1::2"];
            ValueFacts dotnet = ValueFacts.Of(view);
            ValueFacts python;
            using (Gil()) {
                using PyObject exported = view.ToNumpy();
                python = PyGym.Describe(exported);
            }

            ByteContract.AssertSame("ToNumpy(values[::-1, 1::2])", python, dotnet);
            Assert.AreEqual("<f8", python.DType);
            CollectionAssert.AreEqual(new long[] { 4, 3 }, python.Shape);
        }
    }
}
