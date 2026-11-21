#nullable enable
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
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
    ///     calls where it promises (the pool, a pinned thread, a thread of its own, a dedicated thread for a window); a custom
    ///     <see cref="AsyncVectorWorker"/> (DR-38) is in charge of its sub-env on a thread of its own, and wrapping the
    ///     default worker changes nothing.
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
        ///     A full reset right after an episode ended clears the pending autoreset in both copy paths (DR-42): the async
        ///     env's next steps equal the sync env's. Gymnasium's async env without shared memory keeps the pending autoreset
        ///     and resets the ended sub-env a second time with a reward of 0; Gym.NET deviates on purpose.
        /// </summary>
        /// <param name="sharedMemory">Whether the observations go through the shared buffer.</param>
        /// <exception cref="AssertFailedException">Thrown when no episode ends within 100 pushes, or a result differs from the sync env's.</exception>
        [TestMethod]
        [DataRow(true)]
        [DataRow(false)]
        public void FullResetAfterAnEpisodeEnd_ClearsThePendingAutoreset(bool sharedMemory) {
            using VectorEnv<NDArray, NDArray> sync = MakeVec<NDArray, NDArray>("CartPole-v1", numEnvs: 2, vectorizationMode: "sync");
            using VectorEnv<NDArray, NDArray> asynchronous = MakeVec<NDArray, NDArray>("CartPole-v1", numEnvs: 2, vectorizationMode: "async",
                vectorKwargs: new Info { ["shared_memory"] = sharedMemory });
            AssertSame("reset(seed=3)", sync.Reset(3), asynchronous.Reset(3));

            // Push right until a cart falls, so a NEXT_STEP autoreset is pending when the full reset comes.
            using NDArray push = np.array(new long[] { 1, 1 });
            bool ended = false;
            for (int step = 0; step < 100 && !ended; step++) {
                VectorStepResult<NDArray> expected = sync.Step(push);
                AssertSame($"push #{step}", expected, asynchronous.Step(push));
                using NDArray endedArray = np.logical_or(expected.Terminations, expected.Truncations);
                ended = np.any(endedArray);
            }

            Assert.IsTrue(ended, "no episode ended within 100 pushes");
            AssertSame("reset(seed=4) after the episode end", sync.Reset(4), asynchronous.Reset(4));
            using NDArray actions = np.array(new long[] { 0, 1 });
            for (int step = 0; step < 5; step++)
                AssertSame($"step #{step} after the reset", sync.Step(actions), asynchronous.Step(actions));
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
            EnvViewerFactory? previous = HumanViewerFactory;
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

        /// <summary>
        ///     A custom worker that is the default one (<see cref="AsyncVectorEnv.AsyncWorker"/> through the pipe, DR-38) gives
        ///     the sync env's every result, bit for bit, in both copy paths and autoreset modes: the pipe workers run the same
        ///     loop body as the built-in ones.
        /// </summary>
        /// <param name="autoreset">The autoreset mode's value.</param>
        /// <param name="sharedMemory">Whether the observations go through the shared buffer.</param>
        [TestMethod]
        [DataRow("NextStep", true)]
        [DataRow("NextStep", false)]
        [DataRow("SameStep", true)]
        [DataRow("SameStep", false)]
        public void CustomWorker_WrappingTheDefault_EqualsSync(string autoreset, bool sharedMemory) {
            using VectorEnv<NDArray, NDArray> sync = MakeVec<NDArray, NDArray>("CartPole-v1", numEnvs: 3, vectorizationMode: "sync",
                vectorKwargs: new Info { ["autoreset_mode"] = autoreset });
            using VectorEnv<NDArray, NDArray> asynchronous = MakeVec<NDArray, NDArray>("CartPole-v1", numEnvs: 3, vectorizationMode: "async",
                vectorKwargs: new Info { ["autoreset_mode"] = autoreset, ["shared_memory"] = sharedMemory, ["worker"] = (AsyncVectorWorker)AsyncVectorEnv.AsyncWorker });

            AssertSame("reset", sync.Reset(7), asynchronous.Reset(7));
            sync.ActionSpace.Seed(7);
            for (int step = 0; step < 150; step++) {
                using NDArray actions = sync.ActionSpace.Sample();
                AssertSame($"step {step}", sync.Step(actions), asynchronous.Step(actions));
            }

            asynchronous.Close();
        }

        /// <summary>
        ///     A custom worker decides how its sub-env runs (here, a sub-env of its own making that pays a reward of 2), on a
        ///     thread named as Gymnasium names its processes; with <c>daemon</c> off the threads are foreground threads,
        ///     <c>context</c> is kept, and closing waits for every worker to end.
        /// </summary>
        [TestMethod]
        public void CustomWorker_RunsItsSubEnv_OnAThreadOfItsOwn() {
            var threads = new ConcurrentDictionary<int, Thread>();

            void PayingWorker(int index, Func<IEnv> envFn, PipeConnection pipe, PipeConnection parentPipe, object? sharedMemory,
                ConcurrentQueue<AsyncWorkerError> errorQueue, AutoresetMode autoresetMode, SemaphoreSlim? semaphore) {
                threads[index] = Thread.CurrentThread;
                // The worker builds what it serves: the given function's env is replaced by one that pays 2 per step.
                AsyncVectorEnv.AsyncWorker(index, () => new GenericTestEnv(stepFunc: (self, _) =>
                    new StepResult<object?>(self.ObservationSpace.Sample(), 2.0, false, false, new Info())), pipe, parentPipe, sharedMemory, errorQueue, autoresetMode, semaphore);
            }

            var envs = new AsyncVectorEnv(Enumerable.Range(0, 3).Select(_ => (Func<IEnv>)(() => new GenericTestEnv())).ToArray(),
                context: "spawn", daemon: false, worker: PayingWorker);
            Assert.AreEqual("spawn", envs.Context);
            Assert.IsFalse(envs.Daemon);
            Assert.IsNotNull(envs.WorkerFunction);
            envs.Reset(seed: 0);
            VectorStepResult<object?> step = envs.Step(envs.ActionSpace.Sample());
            CollectionAssert.AreEqual(new[] { 2.0, 2.0, 2.0 }, ((NDArray)step.Rewards).ToArray<double>());

            Assert.AreEqual(3, threads.Count);
            foreach ((int index, Thread thread) in threads) {
                Assert.AreEqual($"Worker<AsyncVectorEnv>-{index}", thread.Name);
                Assert.IsFalse(thread.IsBackground, "daemon=False makes the worker threads foreground threads.");
            }

            envs.Close();
            Assert.IsTrue(threads.Values.All(thread => !thread.IsAlive), "Closing waits for every worker to end.");
            envs.Dispose();
        }

        /// <summary>
        ///     A custom worker that dies before answering (here, sub-env 1's worker throws once it has read its first command)
        ///     leaves the constructor's receive at the end of the stream, as Python's parent sees <c>EOFError</c>, with the
        ///     worker's exception as the inner one; the other workers are shut down. (A worker that dies before the first
        ///     command arrives makes that send fail instead, a broken pipe carrying the exception, as in Python; reading the
        ///     command first keeps this test on the one outcome.)
        /// </summary>
        [TestMethod]
        public void CustomWorker_ThatDies_EndsTheStream_CarryingItsException() {
            var threads = new ConcurrentBag<Thread>();

            void DyingWorker(int index, Func<IEnv> envFn, PipeConnection pipe, PipeConnection parentPipe, object? sharedMemory,
                ConcurrentQueue<AsyncWorkerError> errorQueue, AutoresetMode autoresetMode, SemaphoreSlim? semaphore) {
                threads.Add(Thread.CurrentThread);
                if (index == 1) {
                    pipe.Recv();
                    throw new InvalidOperationException("worker 1 failed to start");
                }

                AsyncVectorEnv.AsyncWorker(index, envFn, pipe, parentPipe, sharedMemory, errorQueue, autoresetMode, semaphore);
            }

            var error = Assert.ThrowsExactly<EndOfStreamException>(() =>
                new AsyncVectorEnv(Enumerable.Range(0, 3).Select(_ => (Func<IEnv>)(() => new GenericTestEnv())).ToArray(), worker: DyingWorker));
            Assert.IsInstanceOfType<InvalidOperationException>(error.InnerException);
            Assert.AreEqual("worker 1 failed to start", error.InnerException!.Message);
            Assert.IsTrue(threads.All(thread => !thread.IsAlive), "The failed constructor ended every worker.");
        }

        /// <summary>
        ///     A sub-env that raises under the default worker over the pipe follows Gymnasium's protocol: the step rethrows its
        ///     exception as it was thrown, and a later call to that sub-env finds its pipe dropped.
        /// </summary>
        [TestMethod]
        public void CustomWorker_SubEnvError_FollowsGymnasiumsProtocol() {
            Func<IEnv>[] envFns = Enumerable.Range(0, 2).Select(index => (Func<IEnv>)(() => new GenericTestEnv(stepFunc: (self, _) => index == 1
                ? throw new InvalidOperationException("sub-env 1 fell over")
                : new StepResult<object?>(self.ObservationSpace.Sample(), 0.0, false, false, new Info())))).ToArray();
            using var envs = new AsyncVectorEnv(envFns, worker: AsyncVectorEnv.AsyncWorker);
            envs.Reset(seed: 0);
            Assert.AreEqual("sub-env 1 fell over", Assert.ThrowsExactly<InvalidOperationException>(() => envs.Step(envs.ActionSpace.Sample())).Message);
            Assert.AreEqual("'NoneType' object has no attribute 'send'", Assert.ThrowsExactly<AttributeError>(() => envs.Step(envs.ActionSpace.Sample())).Message);
            envs.Close();
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

        /// <summary>Asserts two steps are equal exactly: observations, rewards, both flags and infos.</summary>
        /// <param name="label">The comparison, for messages.</param>
        /// <param name="expected">The sync env's.</param>
        /// <param name="actual">The async env's.</param>
        /// <exception cref="AssertFailedException">Thrown when they differ.</exception>
        private static void AssertSame(string label, VectorStepResult<NDArray> expected, VectorStepResult<NDArray> actual) {
            Assert.IsTrue(EnvChecker.DataEquivalence(expected.Observations, actual.Observations, exact: true), $"{label}: observations");
            Assert.IsTrue(EnvChecker.DataEquivalence(expected.Rewards, actual.Rewards, exact: true), $"{label}: rewards");
            Assert.IsTrue(EnvChecker.DataEquivalence(expected.Terminations, actual.Terminations, exact: true), $"{label}: terminations");
            Assert.IsTrue(EnvChecker.DataEquivalence(expected.Truncations, actual.Truncations, exact: true), $"{label}: truncations");
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
