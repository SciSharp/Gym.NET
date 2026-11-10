#nullable enable
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Threading;
using Gym.Environments;
using Gym.Envs;
using Gym.Observations;
using Gym.Tests.Gymnasium;
using Gym.Utils;
using Gym.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Vector {
    /// <summary>
    ///     Gym.NET's own contract for <see cref="AsyncVectorEnv{TObs, TAct}"/> (DR-22), beyond Gymnasium's tests: every
    ///     execution mode gives <see cref="SyncVectorEnv{TObs, TAct}"/>'s results bit for bit, and each mode runs a sub-env's
    ///     calls where it promises (the pool, a pinned thread, a thread of its own, a dedicated thread for a window).
    /// </summary>
    [TestClass]
    public class AsyncVectorEnvTests {
        /// <summary>Every execution mode with every autoreset mode and both copy paths.</summary>
        public static IEnumerable<object[]> Cases =>
            from mode in Enum.GetValues<AsyncExecutionMode>()
            from autoreset in new[] { "NextStep", "SameStep", "Disabled" }
            from sharedMemory in new[] { true, false }
            select new object[] { mode, autoreset, sharedMemory };

        /// <summary>
        ///     Four CartPoles through 200 seeded steps across episode ends: the async env's every result equals the sync env's
        ///     exactly, whatever the execution mode, autoreset mode and copy path (<c>DISABLED</c> resetting the ended ones).
        /// </summary>
        /// <param name="mode">The execution mode.</param>
        /// <param name="autoreset">The autoreset mode's value.</param>
        /// <param name="sharedMemory">Whether the observations go through the shared buffer.</param>
        [TestMethod]
        [DynamicData(nameof(Cases))]
        public void AsyncEqualsSync_BitForBit(AsyncExecutionMode mode, string autoreset, bool sharedMemory) {
            using VectorEnv<NDArray, NDArray> sync = MakeVec<NDArray, NDArray>("CartPole-v1", numEnvs: 4, vectorizationMode: "sync",
                vectorKwargs: new Info { ["autoreset_mode"] = autoreset });
            using VectorEnv<NDArray, NDArray> asynchronous = MakeVec<NDArray, NDArray>("CartPole-v1", numEnvs: 4, vectorizationMode: "async",
                vectorKwargs: new Info { ["autoreset_mode"] = autoreset, ["shared_memory"] = sharedMemory, ["execution_mode"] = mode });

            AssertSame("reset", sync.Reset(42), asynchronous.Reset(42));
            sync.ActionSpace.Seed(42);
            for (int step = 0; step < 200; step++) {
                using NDArray actions = sync.ActionSpace.Sample();
                VectorStepResult<NDArray> expected = sync.Step(actions);
                VectorStepResult<NDArray> actual = asynchronous.Step(actions);
                Assert.IsTrue(EnvChecker.DataEquivalence(expected.Observations, actual.Observations, exact: true), $"observations at step {step}");
                Assert.IsTrue(EnvChecker.DataEquivalence(expected.Rewards, actual.Rewards, exact: true), $"rewards at step {step}");
                Assert.IsTrue(EnvChecker.DataEquivalence(expected.Terminations, actual.Terminations, exact: true), $"terminations at step {step}");
                Assert.IsTrue(EnvChecker.DataEquivalence(expected.Truncations, actual.Truncations, exact: true), $"truncations at step {step}");
                Assert.IsTrue(EnvChecker.DataEquivalence(expected.Infos, actual.Infos, exact: true), $"infos at step {step}");

                if (autoreset != "Disabled")
                    continue;
                using NDArray ended = np.logical_or(expected.Terminations, expected.Truncations);
                if (!np.any(ended))
                    continue;
                AssertSame($"masked reset after step {step}", sync.Reset(options: new Info { ["reset_mask"] = ended }), asynchronous.Reset(options: new Info { ["reset_mask"] = ended }));
            }

            asynchronous.Close();
        }

        /// <summary>
        ///     <see cref="AsyncExecutionMode.DedicatedThreads"/> pins sub-env <c>i</c> to thread <c>i % threadCount</c>, and
        ///     <see cref="AsyncExecutionMode.ThreadPerEnv"/> gives each sub-env a thread of its own, for every call.
        /// </summary>
        /// <param name="mode">The execution mode.</param>
        [TestMethod]
        [DataRow(AsyncExecutionMode.DedicatedThreads)]
        [DataRow(AsyncExecutionMode.ThreadPerEnv)]
        public void DedicatedModes_KeepEachSubEnvOnItsThread(AsyncExecutionMode mode) {
            var threads = new ConcurrentDictionary<int, ConcurrentBag<int>>();
            Func<IEnv>[] envFns = Enumerable.Range(0, 5).Select(index => (Func<IEnv>)(() => RecordingEnv(index, threads))).ToArray();
            using var envs = new AsyncVectorEnv(envFns, executionMode: mode, threadCount: mode == AsyncExecutionMode.DedicatedThreads ? 2 : null);
            envs.Reset(seed: 0);
            for (int step = 0; step < 10; step++)
                envs.Step(envs.ActionSpace.Sample());
            envs.Close();

            int[] threadOf = Enumerable.Range(0, 5).Select(index => {
                int[] seen = threads[index].Distinct().ToArray();
                Assert.AreEqual(1, seen.Length, $"sub-env {index} ran on {seen.Length} threads");
                return seen[0];
            }).ToArray();
            if (mode == AsyncExecutionMode.ThreadPerEnv) {
                Assert.AreEqual(5, threadOf.Distinct().Count(), "every sub-env has a thread of its own");
            } else {
                Assert.AreEqual(threadOf[0], threadOf[2], "sub-envs 0 and 2 share thread 0");
                Assert.AreEqual(threadOf[0], threadOf[4], "sub-envs 0 and 4 share thread 0");
                Assert.AreEqual(threadOf[1], threadOf[3], "sub-envs 1 and 3 share thread 1");
                Assert.AreNotEqual(threadOf[0], threadOf[1], "the two threads differ");
            }
        }

        /// <summary>
        ///     In <see cref="AsyncExecutionMode.ThreadPool"/> mode a sub-env rendering to a window gets a dedicated thread, so
        ///     every call of it (and the window it opens) stays on one thread, while the others run on the pool.
        /// </summary>
        [TestMethod]
        public void ThreadPoolMode_GivesAWindowItsOwnThread() {
            IEnvironmentViewerFactoryDelegate? previous = HumanViewerFactory;
            HumanViewerFactory = NullEnvViewer.Factory;
            try {
                var threads = new ConcurrentDictionary<int, ConcurrentBag<int>>();
                using var envs = new AsyncVectorEnv(Enumerable.Range(0, 3).Select(index => (Func<IEnv>)(() => RecordingEnv(index, threads, "human"))).ToArray());
                envs.Reset(seed: 0);
                for (int step = 0; step < 20; step++)
                    envs.Step(envs.ActionSpace.Sample());
                envs.Close();

                for (int index = 0; index < 3; index++)
                    Assert.AreEqual(1, threads[index].Distinct().Count(), $"the human sub-env {index} ran on more than one thread");
            } finally {
                HumanViewerFactory = previous;
            }
        }

        /// <summary>Disposing an open async env closes and disposes every sub-env before it returns, even with a call pending.</summary>
        [TestMethod]
        public void Dispose_ClosesTheSubEnvs_EvenWithACallPending() {
            var closed = new ConcurrentBag<int>();
            var envs = new AsyncVectorEnv(Enumerable.Range(0, 3).Select(index => (Func<IEnv>)(() => new ClosingEnv(index, closed))).ToArray());
            envs.Reset();
            envs.StepAsync(envs.ActionSpace.Sample());
            envs.Dispose();
            CollectionAssert.AreEquivalent(new[] { 0, 1, 2 }, closed.ToArray());
        }

        /// <summary>Asserts two resets are equal exactly.</summary>
        /// <param name="label">The comparison, for messages.</param>
        /// <param name="expected">The sync env's.</param>
        /// <param name="actual">The async env's.</param>
        /// <exception cref="AssertFailedException">Thrown when they differ.</exception>
        private static void AssertSame(string label, VectorResetResult<NDArray> expected, VectorResetResult<NDArray> actual) {
            Assert.IsTrue(EnvChecker.DataEquivalence(expected.Observations, actual.Observations, exact: true), $"{label}: observations");
            Assert.IsTrue(EnvChecker.DataEquivalence(expected.Infos, actual.Infos, exact: true), $"{label}: infos");
        }

        /// <summary>A test env that records, per sub-env, the managed thread of every reset and step.</summary>
        /// <param name="index">The sub-env.</param>
        /// <param name="threads">Where the threads are recorded.</param>
        /// <param name="renderMode">The render mode the env reports.</param>
        /// <returns>The env.</returns>
        private static GenericTestEnv RecordingEnv(int index, ConcurrentDictionary<int, ConcurrentBag<int>> threads, string? renderMode = null) {
            ConcurrentBag<int> seen = threads.GetOrAdd(index, _ => new ConcurrentBag<int>());
            return new GenericTestEnv(
                resetFunc: (self, seed, _) => {
                    seen.Add(Environment.CurrentManagedThreadId);
                    self.BaseReset(seed);
                    return new ResetResult<object?>(self.ObservationSpace.Sample(), new Info());
                },
                stepFunc: (self, _) => {
                    seen.Add(Environment.CurrentManagedThreadId);
                    return new StepResult<object?>(self.ObservationSpace.Sample(), 0.0, false, false, new Info());
                },
                renderFunc: _ => RenderResult.None,
                metadata: new Metadata { [Metadata.RenderModesKey] = new List<string> { "human" }, [Metadata.RenderFpsKey] = 1000 },
                renderMode: renderMode);
        }

        /// <summary>A test env that records its index when closed, and whose steps take a moment, so a pending call is still running when the vector env goes.</summary>
        private sealed class ClosingEnv : Env<object?, object?> {
            /// <summary>The sub-env's index.</summary>
            private readonly int _index;

            /// <summary>Where closes are recorded.</summary>
            private readonly ConcurrentBag<int> _closed;

            /// <summary>An env of the default spaces.</summary>
            /// <param name="index">The sub-env's index.</param>
            /// <param name="closed">Where closes are recorded.</param>
            public ClosingEnv(int index, ConcurrentBag<int> closed) {
                _index = index;
                _closed = closed;
                ((IEnv)this).ObservationSpace = new Gym.Spaces.Box(0, 1, new Shape(1));
                ((IEnv)this).ActionSpace = new Gym.Spaces.Box(0, 1, new Shape(1));
            }

            /// <summary>Resets.</summary>
            /// <param name="seed">The seed.</param>
            /// <param name="options">Ignored.</param>
            /// <returns>A sample and an empty info.</returns>
            public override ResetResult<object?> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
                SeedNpRandom(seed);
                return new ResetResult<object?>(((IEnv)this).ObservationSpace.Sample(), new Info());
            }

            /// <summary>Steps after a short sleep.</summary>
            /// <param name="action">Ignored.</param>
            /// <returns>A sample, a reward of 0, and no end.</returns>
            public override StepResult<object?> Step(object? action) {
                Thread.Sleep(100);
                return new StepResult<object?>(((IEnv)this).ObservationSpace.Sample(), 0.0, false, false, new Info());
            }

            /// <summary>Records the close once.</summary>
            public override void Close() {
                if (!_closed.Contains(_index))
                    _closed.Add(_index);
                base.Close();
            }
        }
    }
}
