using System;
using System.IO;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Session;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace Gym.Tests.Parity.Tests.Harness {
    /// <summary>
    ///     The session's own guarantees: the oracle is the pinned Gymnasium on the pinned stack, the interpreter is
    ///     deterministic, and the byte readers the whole suite relies on agree with NumPy's own reading of an array.
    /// </summary>
    [TestClass]
    public sealed class SessionTests : ParityTestBase {
        /// <summary>
        ///     Every dtype family a Gymnasium value can carry: the import path of <see cref="ValueFacts.FromPython"/> must read
        ///     each with NumPy's own dtype string and bytes.
        /// </summary>
        private static readonly string[] ImportedDtypes = {
            "bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32", "float64",
        };

        /// <summary>
        ///     A strided, negative-step NumPy view of each dtype, built in Python so that NumSharp's import, not NumSharp's
        ///     export, is what the cell exercises. Snippet flavor: it is a small program over a bound dtype name. Negative
        ///     integers wrap into the unsigned dtypes (a defined integer cast, unlike a negative float's), and the floating
        ///     dtypes get fractional values so their mantissas are not all zero.
        /// </summary>
        private const string StridedViewStatements =
            "import numpy as np\n" +
            "values = np.arange(24) % 7 - 3\n" +
            "if np.dtype(dtype_name).kind == 'f':\n" +
            "    values = values * 1.25 + 0.1\n" +
            "view = values.astype(dtype_name).reshape(4, 6)[::-1, 1::2]\n";

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
            (string file, string version) = PythonRuntime.Run(() => (Python.gymnasium.__file__, Python.gymnasium.__version__));
            string expected = Path.Combine(GymPythonSession.Repository!.GymnasiumPackage, "__init__.py");
            Assert.AreEqual(Path.GetFullPath(expected), Path.GetFullPath(file), ignoreCase: OperatingSystem.IsWindows(), "gymnasium must be imported from refs/Gymnasium.");
            Assert.AreEqual(GymPythonSession.Manifest!.Stack["gymnasium.version"], version);
        }

        /// <summary>
        ///     The interpreter runs with the deterministic settings the session applies through the C runtime before
        ///     <c>Py_Initialize</c>: no string-hash randomization, no bytecode written into the submodule, headless pygame,
        ///     single-threaded BLAS.
        /// </summary>
        [TestMethod]
        public void Interpreter_RunsWithTheDeterministicSettings() {
            (long hash, bool noBytecode) = PythonRuntime.Run(() => (Python.sys.flags.hash_randomization, Python.sys.dont_write_bytecode));
            Assert.AreEqual(0L, hash, "PYTHONHASHSEED=0 must reach Py_Initialize.");
            Assert.IsTrue(noBytecode, "Bytecode must never be written into refs/Gymnasium.");
            Assert.AreEqual("dummy", Outcome.CapturePython(() => Python.os.environ.get("SDL_VIDEODRIVER")).Expect("os.environ.get('SDL_VIDEODRIVER')").Text, "pygame must stay headless.");
            Assert.AreEqual("1", Outcome.CapturePython(() => Python.os.environ.get("OPENBLAS_NUM_THREADS")).Expect("os.environ.get('OPENBLAS_NUM_THREADS')").Text, "NumPy's BLAS must run single-threaded.");
        }

        /// <summary>
        ///     The export direction: an array exported zero-copy to NumPy (NumSharp's <c>ToNumpy</c>) and read by NumPy itself
        ///     (<c>dtype.str</c>, <c>shape</c>, <c>tobytes()</c>) equals the same array described by NumSharp's own
        ///     <c>data.tobytes</c>, for a strided, negative-step view. Every .NET-to-Python action rests on this; it also
        ///     exercises the leak gate with a live export.
        /// </summary>
        [TestMethod]
        public void ByteReaders_AgreeOnAnExportedView() {
            using NDArray values = np.arange(24.0).reshape(4, 6);
            using NDArray view = values["::-1, 1::2"];
            ValueFacts dotnet = ValueFacts.Of(view);
            ValueFacts python = PythonRuntime.Run(() => {
                using PyObject exported = view.ToNumpy();
                return NumpyOwnReading(exported);
            });

            ByteContract.AssertSame("ToNumpy(values[::-1, 1::2])", python, dotnet);
            Assert.AreEqual("<f8", python.DType);
            CollectionAssert.AreEqual(new long[] { 4, 3 }, python.Shape);
        }

        /// <summary>
        ///     The import direction, which every Gymnasium value takes: for each dtype family, a strided, negative-step view
        ///     made in NumPy and described by <see cref="ValueFacts.FromPython"/> (NumSharp's <c>ToNDArray</c>, then NumSharp's
        ///     <c>data.tobytes</c>) equals NumPy's own reading of it. Without this cell the suite's Python reader would be
        ///     assumed, not verified.
        /// </summary>
        [TestMethod]
        public void ByteReaders_AgreeOnAnImportedView() {
            foreach (string dtype in ImportedDtypes) {
                (ValueFacts imported, ValueFacts numpy) = PythonRuntime.Run(() => {
                    using var name = new PyString(dtype);
                    using PyObject view = Python.evalStmts(StridedViewStatements, "view", ("dtype_name", name));
                    return (ValueFacts.FromPython(view), NumpyOwnReading(view));
                });

                ByteContract.AssertSame($"{dtype} view [::-1, 1::2]", numpy, imported);
                CollectionAssert.AreEqual(new long[] { 4, 3 }, numpy.Shape, $"{dtype}: the probe must be the strided 4 x 3 view.");
            }
        }

        /// <summary>
        ///     A NumPy array as NumPy itself reads it: <c>a.dtype.str</c>, <c>a.shape</c> and <c>a.tobytes()</c> (C order),
        ///     converted by pythonnet, never by NumSharp. Must run under the GIL.
        /// </summary>
        /// <param name="array">The array; borrowed.</param>
        /// <returns>The description.</returns>
        /// <exception cref="PythonException">Thrown when an attribute read fails.</exception>
        private static ValueFacts NumpyOwnReading(PyObject array) {
            using PyObject dtype = Python.numpy.ndarray.dtype(array);
            using PyObject shape = Python.numpy.ndarray.shape(array);
            // The dtype and the shape tuple go through FromPython's scalar rules (dtype.str, Python ints), which never touch
            // NumSharp's import; only the bytes needed a reader, and tobytes_t is pythonnet's.
            ValueFacts dtypeFacts = ValueFacts.FromPython(dtype);
            ValueFacts shapeFacts = ValueFacts.FromPython(shape);
            var dims = new long[shapeFacts.Items.Count];
            for (int i = 0; i < dims.Length; i++)
                dims[i] = (long)shapeFacts.Items[i].Integer;
            return ValueFacts.FromArray(dtypeFacts.Text, dims, Python.numpy.ndarray.tobytes_t(array), "numpy.ndarray");
        }
    }
}
