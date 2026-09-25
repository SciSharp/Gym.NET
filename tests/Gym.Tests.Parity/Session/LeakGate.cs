using System;
using System.Diagnostics;
using Gym.Tests.Parity.Interop;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace Gym.Tests.Parity.Session {
    /// <summary>
    ///     NumSharp's leak gate, as its interop suite runs it (<c>InteropTestBase</c>): the process-wide counters of live
    ///     NumSharp → numpy exports and numpy → NumSharp imports must return to their value at the start of every test. Every
    ///     parity test therefore doubles as a proof that the actions and states it handed to Gymnasium were released, neither
    ///     leaked nor freed early.
    /// </summary>
    /// <remarks>
    ///     Releases are asynchronous: an export is released by a Python finalizer when its last numpy view dies, and
    ///     pythonnet defers decrefs to its own finalizer queue. <see cref="Pump"/> drives all of those, so the gate waits for
    ///     the counters instead of reading them once.
    /// </remarks>
    internal static class LeakGate {
        /// <summary>The counters at the start of a test.</summary>
        /// <param name="Exports">Live NumSharp → numpy exports (<c>ToNumpy</c> views whose buffers Python still pins).</param>
        /// <param name="Imports">Live numpy → NumSharp imports (<c>AsNDArray</c> views over Python memory).</param>
        internal readonly record struct Baseline(int Exports, int Imports);

        /// <summary>Settles earlier tests' releases, then records the counters.</summary>
        /// <returns>The baseline the test must return to.</returns>
        internal static Baseline Capture() {
            Settle();
            return new Baseline(NDArrayPythonInterop.LiveExports, NDArrayPythonInterop.LiveImports);
        }

        /// <summary>Pumps the release machinery until both counters are back at or below the baseline, or the time runs out.</summary>
        /// <param name="baseline">The counters recorded by <see cref="Capture"/>.</param>
        /// <param name="timeoutMilliseconds">How long to keep pumping.</param>
        /// <param name="counters">The counters' movement, for the failure message.</param>
        /// <returns><see langword="true"/> when both counters returned in time.</returns>
        internal static bool WaitForReturn(Baseline baseline, int timeoutMilliseconds, out string counters) {
            var clock = Stopwatch.StartNew();
            bool returned;
            do {
                returned = NDArrayPythonInterop.LiveExports <= baseline.Exports && NDArrayPythonInterop.LiveImports <= baseline.Imports;
                if (returned || clock.ElapsedMilliseconds > timeoutMilliseconds)
                    break;
                Pump();
                System.Threading.Thread.Sleep(20);
            } while (true);

            counters = $"LiveExports {baseline.Exports} -> {NDArrayPythonInterop.LiveExports}, LiveImports {baseline.Imports} -> {NDArrayPythonInterop.LiveImports}";
            return returned;
        }

        /// <summary>
        ///     One full release turn: CLR collection and finalizers (NDArray finalizers drop their references and queue lease
        ///     releases), pythonnet's deferred decrefs, a Python <c>gc.collect()</c>, and one trivial conversion, which runs the
        ///     interop's inline drain of queued releases.
        /// </summary>
        internal static void Pump() {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            PythonRuntime.Run(() => {
                Finalizer.Instance.Collect();
                Python.gc.collect();
                using NDArray one = np.arange(1);
                using PyObject copy = one.ToNumpyCopy();
            });
        }

        /// <summary>Pumps until the counters stop moving, so a test starts from a quiet baseline rather than mid-release.</summary>
        private static void Settle() {
            int lastExports = -1, lastImports = -1;
            for (int turn = 0; turn < 40; turn++) {
                int exports = NDArrayPythonInterop.LiveExports, imports = NDArrayPythonInterop.LiveImports;
                if (exports == lastExports && imports == lastImports && (exports + imports == 0 || turn > 2))
                    return;
                lastExports = exports;
                lastImports = imports;
                Pump();
            }
        }
    }
}
