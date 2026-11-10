#nullable enable
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text.RegularExpressions;
using System.Threading;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Observations;
using Gym.Spaces;
using Gym.Utils;
using Gym.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using SpaceTuple = Gym.Spaces.Tuple;

namespace Gym.Tests.Gymnasium.Vector {
    /// <summary>
    ///     Gymnasium's <c>AsyncVectorEnv</c> tests (<c>test_async_vector_env.py</c>), run against
    ///     <see cref="AsyncVectorEnv"/> in each of its three execution modes (DR-22), where Gymnasium runs processes.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Two deliberate changes, both from threads replacing processes: a test that builds every sub-env from one shared
    ///     space builds a space per sub-env (a process gets a copy; threads would sample one generator at once), and the
    ///     sub-env errors' traces are .NET stack traces, so the trace lines are matched on the exception's text alone. The two
    ///     custom-<c>worker</c> tests run over <see cref="AsyncVectorWorker"/> (DR-38), whose eight parameters make Gymnasium's
    ///     arity check a compile-time fact.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_async_vector_env.py">tests/vector/test_async_vector_env.py</see>.</para>
    /// </remarks>
    [TestClass]
    public class TestAsyncVectorEnv {
        /// <summary>Every execution mode.</summary>
        public static IEnumerable<object[]> Modes => Enum.GetValues<AsyncExecutionMode>().Select(mode => new object[] { mode });

        /// <summary>Every execution mode, with and without shared memory (Gymnasium's <c>shared_memory</c> parameter).</summary>
        public static IEnumerable<object[]> ModesAndSharedMemory =>
            from mode in Enum.GetValues<AsyncExecutionMode>() from sharedMemory in new[] { true, false } select new object[] { mode, sharedMemory };

        /// <summary>Test creating an async vector environment with or without shared memory.</summary>
        /// <param name="mode">The execution mode.</param>
        /// <param name="sharedMemory">Whether the observations go through the shared buffer.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_async_vector_env.py#L29">tests/vector/test_async_vector_env.py (test_create_async_vector_env)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(ModesAndSharedMemory))]
        public void TestCreateAsyncVectorEnv(AsyncExecutionMode mode, bool sharedMemory) {
            Func<IEnv>[] envFns = Enumerable.Range(0, 8).Select(i => TestingUtils.MakeEnv("CartPole-v1", i)).ToArray();
            using var env = new AsyncVectorEnv(envFns, sharedMemory: sharedMemory, executionMode: mode);
            Assert.AreEqual(8, env.NumEnvs);
            env.Close();
        }

        /// <summary>Tests that the vector env's metadata doesn't mutate the sub-environment's (class-level) metadata.</summary>
        /// <param name="mode">The execution mode.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_async_vector_env.py#L39">tests/vector/test_async_vector_env.py (test_metadata_async_vector_env)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Modes))]
        public void TestMetadataAsyncVectorEnv(AsyncExecutionMode mode) {
            using var envs1 = new AsyncVectorEnv(new[] { TestingUtils.MakeEnv("CartPole-v1", 0) }, autoresetMode: AutoresetMode.NextStep, executionMode: mode);
            using var envs2 = new AsyncVectorEnv(new[] { TestingUtils.MakeEnv("CartPole-v1", 1) }, autoresetMode: AutoresetMode.SameStep, executionMode: mode);

            Assert.AreSame(AutoresetMode.NextStep, envs1.Metadata["autoreset_mode"]);
            Assert.AreSame(AutoresetMode.SameStep, envs2.Metadata["autoreset_mode"]);

            using IEnv env = TestingUtils.MakeEnv("CartPole-v1", 0)();
            Assert.IsFalse(env.Metadata.ContainsKey("autoreset_mode"));

            env.Close();
            envs1.Close();
            envs2.Close();
        }

        /// <summary>Test the reset of async vector environment with or without shared memory.</summary>
        /// <param name="mode">The execution mode.</param>
        /// <param name="sharedMemory">Whether the observations go through the shared buffer.</param>
        /// <remarks>
        ///     Gymnasium's last check, <c>all([isinstance(info, dict) for info in infos])</c>, iterates the dict's keys (strings),
        ///     so it holds only for CartPole's empty reset infos; the port keeps that meaning.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_async_vector_env.py#L59">tests/vector/test_async_vector_env.py (test_reset_async_vector_env)</see>.
        /// </remarks>
        [TestMethod]
        [DynamicData(nameof(ModesAndSharedMemory))]
        public void TestResetAsyncVectorEnv(AsyncExecutionMode mode, bool sharedMemory) {
            Func<IEnv>[] envFns = Enumerable.Range(0, 8).Select(i => TestingUtils.MakeEnv("CartPole-v1", i)).ToArray();

            var env = new AsyncVectorEnv(envFns, sharedMemory: sharedMemory, executionMode: mode);
            (object? observations, Info infos) = env.Reset();
            env.Close();
            AssertBatch(env, observations);
            env.Dispose();

            env = new AsyncVectorEnv(envFns, sharedMemory: sharedMemory, executionMode: mode);
            try {
                (observations, infos) = env.Reset();
            } finally {
                env.Close();
            }

            AssertBatch(env, observations);
            Assert.IsInstanceOfType<Info>(infos);
            Assert.IsTrue(infos.Keys.All(key => (object)key is IReadOnlyDictionary<string, object?>));
            env.Dispose();

            static void AssertBatch(AsyncVectorEnv env, object? observations) {
                var space = (Box)ObjectSpace.Unview(env.ObservationSpace)!;
                Assert.IsInstanceOfType<NDArray>(observations);
                var batch = (NDArray)observations!;
                Assert.IsTrue(batch.dtype == space.DType!);
                CollectionAssert.AreEqual(new long[] { 8 }.Concat(ObjectSpace.Unview(env.SingleObservationSpace)!.Shape!.Value.dimensions).ToArray(), batch.shape);
                CollectionAssert.AreEqual(space.Shape.dimensions, batch.shape);
            }
        }

        /// <summary>Rendering three <c>rgb_array</c> CartPoles gives a frame each; without a render mode the vector env has none.</summary>
        /// <param name="mode">The execution mode.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_async_vector_env.py#L88">tests/vector/test_async_vector_env.py (test_render_async_vector)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Modes))]
        public void TestRenderAsyncVector(AsyncExecutionMode mode) {
            using (var envs = new AsyncVectorEnv(Enumerable.Range(0, 3).Select(i => TestingUtils.MakeEnv("CartPole-v1", i, new { render_mode = "rgb_array" })), executionMode: mode)) {
                Assert.AreEqual("rgb_array", envs.RenderMode);

                envs.Reset();
                IReadOnlyList<RenderResult>? renderedFrames = envs.Render();
                Assert.IsNotNull(renderedFrames);
                Assert.AreEqual(envs.NumEnvs, renderedFrames.Count);
                Assert.IsTrue(renderedFrames.All(frame => frame is RenderResult.Rgb));
                envs.Close();
            }

            using (var envs = new AsyncVectorEnv(Enumerable.Range(0, 3).Select(i => TestingUtils.MakeEnv("CartPole-v1", i)), executionMode: mode)) {
                Assert.IsNull(envs.RenderMode);
                envs.Close();
            }
        }

        /// <summary>Test the step async vector environment with and without shared memory, with batched and single-sampled actions.</summary>
        /// <param name="mode">The execution mode.</param>
        /// <param name="sharedMemory">Whether the observations go through the shared buffer.</param>
        /// <param name="useSingleActionSpace">Whether the actions are a list of single samples rather than a batched sample.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_async_vector_env.py#L108">tests/vector/test_async_vector_env.py (test_step_async_vector_env)</see>.</remarks>
        [TestMethod]
        [DataRow(AsyncExecutionMode.ThreadPool, true, true)]
        [DataRow(AsyncExecutionMode.ThreadPool, true, false)]
        [DataRow(AsyncExecutionMode.ThreadPool, false, true)]
        [DataRow(AsyncExecutionMode.ThreadPool, false, false)]
        [DataRow(AsyncExecutionMode.DedicatedThreads, true, true)]
        [DataRow(AsyncExecutionMode.DedicatedThreads, false, false)]
        [DataRow(AsyncExecutionMode.ThreadPerEnv, true, false)]
        [DataRow(AsyncExecutionMode.ThreadPerEnv, false, true)]
        public void TestStepAsyncVectorEnv(AsyncExecutionMode mode, bool sharedMemory, bool useSingleActionSpace) {
            Func<IEnv>[] envFns = Enumerable.Range(0, 8).Select(i => TestingUtils.MakeEnv("CartPole-v1", i)).ToArray();

            using var env = new AsyncVectorEnv(envFns, sharedMemory: sharedMemory, executionMode: mode);
            env.Reset();

            Assert.IsInstanceOfType<Discrete>(ObjectSpace.Unview(env.SingleActionSpace));
            Assert.IsInstanceOfType<MultiDiscrete>(ObjectSpace.Unview(env.ActionSpace));

            object? actions = useSingleActionSpace ? Enumerable.Range(0, 8).Select(_ => env.SingleActionSpace.Sample()).ToList() : env.ActionSpace.Sample();
            (object? observations, NDArray rewards, NDArray terminations, NDArray truncations, _) = env.Step(actions);

            env.Close();

            var space = (Box)ObjectSpace.Unview(env.ObservationSpace)!;
            Assert.IsInstanceOfType<NDArray>(observations);
            var batch = (NDArray)observations!;
            Assert.IsTrue(batch.dtype == space.DType!);
            CollectionAssert.AreEqual(new long[] { 8 }.Concat(ObjectSpace.Unview(env.SingleObservationSpace)!.Shape!.Value.dimensions).ToArray(), batch.shape);
            CollectionAssert.AreEqual(space.Shape.dimensions, batch.shape);

            Assert.IsTrue(rewards.typecode == NPTypeCode.Double);
            Assert.AreEqual(1, rewards.ndim);
            Assert.AreEqual(8L, rewards.size);

            Assert.IsTrue(terminations.typecode == NPTypeCode.Boolean);
            Assert.AreEqual(1, terminations.ndim);
            Assert.AreEqual(8L, terminations.size);

            Assert.IsTrue(truncations.typecode == NPTypeCode.Boolean);
            Assert.AreEqual(1, truncations.ndim);
            Assert.AreEqual(8L, truncations.size);
        }

        /// <summary>Test call with async vector environment: <c>render</c> of <c>rgb_array_list</c> CartPoles and their <c>gravity</c>.</summary>
        /// <param name="mode">The execution mode.</param>
        /// <param name="sharedMemory">Whether the observations go through the shared buffer.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_async_vector_env.py#L150">tests/vector/test_async_vector_env.py (test_call_async_vector_env)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(ModesAndSharedMemory))]
        public void TestCallAsyncVectorEnv(AsyncExecutionMode mode, bool sharedMemory) {
            Func<IEnv>[] envFns = Enumerable.Range(0, 4).Select(i => TestingUtils.MakeEnv("CartPole-v1", i, new { render_mode = "rgb_array_list" })).ToArray();

            using var env = new AsyncVectorEnv(envFns, sharedMemory: sharedMemory, executionMode: mode);
            env.Reset();
            IReadOnlyList<object?> images = env.Call("render");
            IReadOnlyList<object?> gravity = env.Call("gravity");

            env.Close();

            Assert.AreEqual(4, images.Count);
            for (int i = 0; i < 4; i++) {
                var frames = (RenderResult.Frames)images[i]!;
                Assert.AreEqual(1, frames.Items.Count);
                Assert.IsInstanceOfType<RenderResult.Rgb>(frames.Items[0]);
            }

            Assert.AreEqual(4, gravity.Count);
            for (int i = 0; i < 4; i++) {
                Assert.IsInstanceOfType<double>(gravity[i]);
                Assert.AreEqual(9.8, (double)gravity[i]!);
            }
        }

        /// <summary>Test <c>set_attr</c> for async vector environment with or without shared memory.</summary>
        /// <param name="mode">The execution mode.</param>
        /// <param name="sharedMemory">Whether the observations go through the shared buffer.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_async_vector_env.py#L177">tests/vector/test_async_vector_env.py (test_set_attr_async_vector_env)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(ModesAndSharedMemory))]
        public void TestSetAttrAsyncVectorEnv(AsyncExecutionMode mode, bool sharedMemory) {
            Func<IEnv>[] envFns = Enumerable.Range(0, 4).Select(i => TestingUtils.MakeEnv("CartPole-v1", i)).ToArray();

            using var env = new AsyncVectorEnv(envFns, sharedMemory: sharedMemory, executionMode: mode);
            env.SetAttr("gravity", new List<object?> { 9.81, 3.72, 8.87, 1.62 });
            IReadOnlyList<object?> gravity = env.GetAttr("gravity");
            CollectionAssert.AreEqual(new object?[] { 9.81, 3.72, 8.87, 1.62 }, gravity.ToArray());

            env.Close();
        }

        /// <summary>Observations returned with <c>copy=True</c> can be written to (Gymnasium marks the test as doing nothing more).</summary>
        /// <param name="mode">The execution mode.</param>
        /// <param name="sharedMemory">Whether the observations go through the shared buffer.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_async_vector_env.py#L190">tests/vector/test_async_vector_env.py (test_copy_async_vector_env)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(ModesAndSharedMemory))]
        public void TestCopyAsyncVectorEnv(AsyncExecutionMode mode, bool sharedMemory) {
            Func<IEnv>[] envFns = Enumerable.Range(0, 8).Select(i => TestingUtils.MakeEnv("CartPole-v1", i)).ToArray();

            // TODO, these tests do nothing, understand the purpose of the tests and fix them
            using var env = new AsyncVectorEnv(envFns, sharedMemory: sharedMemory, copy: true, executionMode: mode);
            (object? observations, _) = env.Reset();
            ((NDArray)observations!)[0] = 0;

            env.Close();
        }

        /// <summary>Observations returned with <c>copy=False</c> can be written to (Gymnasium marks the test as doing nothing more).</summary>
        /// <param name="mode">The execution mode.</param>
        /// <param name="sharedMemory">Whether the observations go through the shared buffer.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_async_vector_env.py#L203">tests/vector/test_async_vector_env.py (test_no_copy_async_vector_env)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(ModesAndSharedMemory))]
        public void TestNoCopyAsyncVectorEnv(AsyncExecutionMode mode, bool sharedMemory) {
            Func<IEnv>[] envFns = Enumerable.Range(0, 8).Select(i => TestingUtils.MakeEnv("CartPole-v1", i)).ToArray();

            // TODO, these tests do nothing, understand the purpose of the tests and fix them
            using var env = new AsyncVectorEnv(envFns, sharedMemory: sharedMemory, copy: false, executionMode: mode);
            (object? observations, _) = env.Reset();
            ((NDArray)observations!)[0] = 0;

            env.Close();
        }

        /// <summary>Test timeout error on reset with and without shared memory.</summary>
        /// <param name="mode">The execution mode.</param>
        /// <param name="sharedMemory">Whether the observations go through the shared buffer.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_async_vector_env.py#L216">tests/vector/test_async_vector_env.py (test_reset_timeout_async_vector_env)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(ModesAndSharedMemory))]
        public void TestResetTimeoutAsyncVectorEnv(AsyncExecutionMode mode, bool sharedMemory) {
            Func<IEnv>[] envFns = Enumerable.Range(0, 4).Select(i => TestingUtils.MakeSlowEnv(0.3, i)).ToArray();

            using var env = new AsyncVectorEnv(envFns, sharedMemory: sharedMemory, executionMode: mode);
            Assert.ThrowsExactly<TimeoutException>(() => {
                env.ResetAsync();
                env.ResetWait(timeout: 0.1);
            });

            env.Close(null, terminate: true);
        }

        /// <summary>Test timeout error on step with and without shared memory.</summary>
        /// <param name="mode">The execution mode.</param>
        /// <param name="sharedMemory">Whether the observations go through the shared buffer.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_async_vector_env.py#L229">tests/vector/test_async_vector_env.py (test_step_timeout_async_vector_env)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(ModesAndSharedMemory))]
        public void TestStepTimeoutAsyncVectorEnv(AsyncExecutionMode mode, bool sharedMemory) {
            Func<IEnv>[] envFns = Enumerable.Range(0, 4).Select(i => TestingUtils.MakeSlowEnv(0.0, i)).ToArray();

            using var env = new AsyncVectorEnv(envFns, sharedMemory: sharedMemory, executionMode: mode);
            using NDArray actions = np.array(new[] { 0.1, 0.1, 0.3, 0.1 });
            Assert.ThrowsExactly<TimeoutException>(() => {
                env.Reset();
                env.StepAsync(actions);
                env.StepWait(timeout: 0.1);
            });
            env.Close(null, terminate: true);
        }

        /// <summary>Test reset being called out of order with and without shared memory.</summary>
        /// <param name="mode">The execution mode.</param>
        /// <param name="sharedMemory">Whether the observations go through the shared buffer.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_async_vector_env.py#L242">tests/vector/test_async_vector_env.py (test_reset_out_of_order_async_vector_env)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(ModesAndSharedMemory))]
        public void TestResetOutOfOrderAsyncVectorEnv(AsyncExecutionMode mode, bool sharedMemory) {
            Func<IEnv>[] envFns = Enumerable.Range(0, 4).Select(i => TestingUtils.MakeEnv("CartPole-v1", i)).ToArray();

            using (var env = new AsyncVectorEnv(envFns, sharedMemory: sharedMemory, executionMode: mode)) {
                var error = Assert.ThrowsExactly<NoAsyncCallError>(() => env.ResetWait());
                Assert.AreEqual("Calling `reset_wait` without any prior call to `reset_async`.", error.Message);
                env.Close(null, terminate: true);
            }

            using (var env = new AsyncVectorEnv(envFns, sharedMemory: sharedMemory, executionMode: mode)) {
                var error = Assert.ThrowsExactly<AlreadyPendingCallError>(() => {
                    object? actions = env.ActionSpace.Sample();
                    env.Reset();
                    env.StepAsync(actions);
                    env.ResetAsync();
                });
                StringAssert.Contains(error.Message, "Calling `reset_async` while waiting for a pending call to `step` to complete");

                using WarningCatcher caught = Warnings.Catch();
                env.Close(null, terminate: true);
                Assert.IsTrue(caught.Records.Any(warning => warning.Category == WarningCategory.UserWarning
                                                            && warning.Message.Contains("Calling `close` while waiting for a pending call to `step` to complete.")));
            }
        }

        /// <summary>Test step out of order with and without shared memory.</summary>
        /// <param name="mode">The execution mode.</param>
        /// <param name="sharedMemory">Whether the observations go through the shared buffer.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_async_vector_env.py#L279">tests/vector/test_async_vector_env.py (test_step_out_of_order_async_vector_env)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(ModesAndSharedMemory))]
        public void TestStepOutOfOrderAsyncVectorEnv(AsyncExecutionMode mode, bool sharedMemory) {
            Func<IEnv>[] envFns = Enumerable.Range(0, 4).Select(i => TestingUtils.MakeEnv("CartPole-v1", i)).ToArray();

            using (var env = new AsyncVectorEnv(envFns, sharedMemory: sharedMemory, executionMode: mode)) {
                var error = Assert.ThrowsExactly<NoAsyncCallError>(() => {
                    env.ActionSpace.Sample();
                    env.Reset();
                    env.StepWait();
                });
                Assert.AreEqual("Calling `step_wait` without any prior call to `step_async`.", error.Message);
                env.Close(null, terminate: true);
            }

            using (var env = new AsyncVectorEnv(envFns, sharedMemory: sharedMemory, executionMode: mode)) {
                var error = Assert.ThrowsExactly<AlreadyPendingCallError>(() => {
                    object? actions = env.ActionSpace.Sample();
                    env.ResetAsync();
                    env.StepAsync(actions);
                });
                StringAssert.Contains(error.Message, "Calling `step_async` while waiting for a pending call to `reset` to complete");

                using WarningCatcher caught = Warnings.Catch();
                env.Close(null, terminate: true);
                Assert.IsTrue(caught.Records.Any(warning => warning.Category == WarningCategory.UserWarning
                                                            && warning.Message.Contains("Calling `close` while waiting for a pending call to `reset` to complete.")));
            }
        }

        /// <summary>Test the error if a function is called if environment is already closed.</summary>
        /// <param name="mode">The execution mode.</param>
        /// <param name="sharedMemory">Whether the observations go through the shared buffer.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_async_vector_env.py#L315">tests/vector/test_async_vector_env.py (test_already_closed_async_vector_env)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(ModesAndSharedMemory))]
        public void TestAlreadyClosedAsyncVectorEnv(AsyncExecutionMode mode, bool sharedMemory) {
            Func<IEnv>[] envFns = Enumerable.Range(0, 4).Select(i => TestingUtils.MakeEnv("CartPole-v1", i)).ToArray();
            using var env = new AsyncVectorEnv(envFns, sharedMemory: sharedMemory, executionMode: mode);
            env.Close();
            Assert.ThrowsExactly<ClosedEnvironmentError>(() => env.Reset());
        }

        /// <summary>Test check spaces for async vector environment with and without shared memory: a FrozenLake among CartPoles is refused.</summary>
        /// <param name="mode">The execution mode.</param>
        /// <param name="sharedMemory">Whether the observations go through the shared buffer.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_async_vector_env.py#L326">tests/vector/test_async_vector_env.py (test_check_spaces_async_vector_env)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(ModesAndSharedMemory))]
        public void TestCheckSpacesAsyncVectorEnv(AsyncExecutionMode mode, bool sharedMemory) {
            // CartPole-v1 - observation_space: Box(4,), action_space: Discrete(2)
            Func<IEnv>[] envFns = Enumerable.Range(0, 8).Select(i => TestingUtils.MakeEnv("CartPole-v1", i)).ToArray();
            // FrozenLake-v1 - Discrete(16), action_space: Discrete(4)
            envFns[1] = TestingUtils.MakeEnv("FrozenLake-v1", 1);
            Assert.ThrowsExactly<RuntimeError>(() => {
                var env = new AsyncVectorEnv(envFns, sharedMemory: sharedMemory, executionMode: mode);
                env.Close(null, terminate: true);
            });
        }

        /// <summary>Test custom spaces with async vector environment: string observations batched into a tuple.</summary>
        /// <param name="mode">The execution mode.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_async_vector_env.py#L337">tests/vector/test_async_vector_env.py (test_custom_space_async_vector_env)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Modes))]
        public void TestCustomSpaceAsyncVectorEnv(AsyncExecutionMode mode) {
            Func<IEnv>[] envFns = Enumerable.Range(0, 4).Select(i => TestingUtils.MakeCustomSpaceEnv(i)).ToArray();

            using var env = new AsyncVectorEnv(envFns, sharedMemory: false, executionMode: mode);
            (object? resetObservations, _) = env.Reset();

            Assert.IsInstanceOfType<TestingUtils.CustomSpace>(ObjectSpace.Unview(env.SingleActionSpace));
            Assert.IsInstanceOfType<SpaceTuple>(ObjectSpace.Unview(env.ActionSpace));

            object?[] actions = { "action-2", "action-3", "action-5", "action-7" };
            (object? stepObservations, _, _, _, _) = env.Step(actions);

            env.Close();

            Assert.IsInstanceOfType<TestingUtils.CustomSpace>(ObjectSpace.Unview(env.SingleObservationSpace));
            Assert.IsInstanceOfType<SpaceTuple>(ObjectSpace.Unview(env.ObservationSpace));

            Assert.IsInstanceOfType<object?[]>(resetObservations);
            CollectionAssert.AreEqual(new object?[] { "reset", "reset", "reset", "reset" }, (object?[])resetObservations!);

            Assert.IsInstanceOfType<object?[]>(stepObservations);
            CollectionAssert.AreEqual(new object?[] { "step(action-2)", "step(action-3)", "step(action-5)", "step(action-7)" }, (object?[])stepObservations!);
        }

        /// <summary>Test custom space with shared memory: refused, as it has no shared layout.</summary>
        /// <param name="mode">The execution mode.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_async_vector_env.py#L367">tests/vector/test_async_vector_env.py (test_custom_space_async_vector_env_shared_memory)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Modes))]
        public void TestCustomSpaceAsyncVectorEnvSharedMemory(AsyncExecutionMode mode) {
            Func<IEnv>[] envFns = Enumerable.Range(0, 4).Select(i => TestingUtils.MakeCustomSpaceEnv(i)).ToArray();
            Assert.ThrowsExactly<ValueError>(() => {
                var env = new AsyncVectorEnv(envFns, sharedMemory: true, executionMode: mode);
                env.Close(null, terminate: true);
            });
        }

        /// <summary>Test observation dtypes without an <c>array</c> typecode (float16) with shared memory.</summary>
        /// <param name="mode">The execution mode.</param>
        /// <remarks>
        ///     Each sub-env gets its own space where Gymnasium shares one (each process receives a copy of it).
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_async_vector_env.py#L375">tests/vector/test_async_vector_env.py (test_float16_async_vector_env_shared_memory)</see>.
        /// </remarks>
        [TestMethod]
        [DynamicData(nameof(Modes))]
        public void TestFloat16AsyncVectorEnvSharedMemory(AsyncExecutionMode mode) {
            static Box ObsSpace() => new(-1.0, 1.0, new Shape(3), np.float16);
            using var envs = new AsyncVectorEnv(Enumerable.Range(0, 2).Select(_ => (Func<IEnv>)(() => new GenericTestEnv(observationSpace: ObsSpace()))), sharedMemory: true,
                executionMode: mode);

            var observations = (NDArray)envs.Reset(seed: 0).Observations!;
            Assert.IsTrue(observations.typecode == NPTypeCode.Half);
            CollectionAssert.AreEqual(new long[] { 2, 3 }, observations.shape);

            observations = (NDArray)envs.Step(envs.ActionSpace.Sample()).Observations!;
            Assert.IsTrue(observations.typecode == NPTypeCode.Half);
            CollectionAssert.AreEqual(new long[] { 2, 3 }, observations.shape);

            envs.Close();
        }

        /// <summary>
        ///     A sub-env that raises in reset, then in step: the exception reaches the caller with the worker's index and
        ///     Gymnasium's three error lines per failure (the trace line is .NET's stack trace here).
        /// </summary>
        /// <param name="mode">The execution mode.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_async_vector_env.py#L407">tests/vector/test_async_vector_env.py (test_async_vector_subenv_error)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Modes))]
        public void TestAsyncVectorSubenvError(AsyncExecutionMode mode) {
            var envs = new AsyncVectorEnv(Enumerable.Range(0, 2).Select(_ => (Func<IEnv>)(() => new GenericTestEnv(resetFunc: RaiseErrorReset, stepFunc: RaiseErrorStep))),
                executionMode: mode);

            using (WarningCatcher caughtWarnings = Warnings.Catch()) {
                envs.Reset(new BigInteger?[] { 0, 0 });
                Assert.AreEqual(0, caughtWarnings.Records.Count);
            }

            IReadOnlyList<WarningMessage> warnings;
            using (WarningCatcher caughtWarnings = Warnings.Catch()) {
                var error = Assert.ThrowsExactly<ValueError>(() => envs.Reset(new BigInteger?[] { 1, 0 }));
                StringAssert.Contains(error.Message, "Error in reset");
                warnings = caughtWarnings.Records.ToList();
            }

            envs.Close();
            envs.Dispose();

            Assert.AreEqual(3, warnings.Count);
            StringAssert.Contains(warnings[0].Message, "Received the following error from Worker-0 - Shutting it down");
            StringAssert.Contains(warnings[1].Message, "Error in reset");
            Assert.AreEqual("\x1b[31mERROR: Raising the last exception back to the main process.\x1b[0m", warnings[2].Message);

            envs = new AsyncVectorEnv(Enumerable.Range(0, 3).Select(_ => (Func<IEnv>)(() => new GenericTestEnv(resetFunc: RaiseErrorReset, stepFunc: RaiseErrorStep))),
                executionMode: mode);
            using (WarningCatcher caughtWarnings = Warnings.Catch()) {
                var error = Assert.ThrowsExactly<ValueError>(() => envs.Step(new List<object?> { 0L, 1L, 2L }));
                StringAssert.Contains(error.Message, "Error in step");
                warnings = caughtWarnings.Records.ToList();
            }

            envs.Close();
            envs.Dispose();

            Assert.AreEqual(5, warnings.Count);
            // due to variance in the step time, the order of warnings is random
            StringAssert.Matches(warnings[0].Message, new Regex(@"\x1b\[31mERROR: Received the following error from Worker-[12] - Shutting it down\x1b\[0m"));
            StringAssert.Matches(warnings[1].Message, new Regex(@"\x1b\[31mERROR: (?s:.)*ValueError: Error in step with [12](?s:.)*\n\x1b\[0m"));
            StringAssert.Matches(warnings[2].Message, new Regex(@"\x1b\[31mERROR: Received the following error from Worker-[12] - Shutting it down\x1b\[0m"));
            StringAssert.Matches(warnings[3].Message, new Regex(@"\x1b\[31mERROR: (?s:.)*ValueError: Error in step with [12](?s:.)*\n\x1b\[0m"));
            Assert.AreEqual("\x1b[31mERROR: Raising the last exception back to the main process.\x1b[0m", warnings[4].Message);
        }

        /// <summary>Tests that <c>max_concurrency</c> limits the number of environments executing at any one time: two meet at a barrier, never more.</summary>
        /// <param name="mode">The execution mode.</param>
        /// <remarks>
        ///     In <see cref="AsyncExecutionMode.DedicatedThreads"/> mode the test asks for two threads, so two sub-envs can run
        ///     at once whatever the machine's processor count.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_async_vector_env.py#L503">tests/vector/test_async_vector_env.py (test_max_concurrency_async_vector_env)</see>.
        /// </remarks>
        [TestMethod]
        [DynamicData(nameof(Modes))]
        public void TestMaxConcurrencyAsyncVectorEnv(AsyncExecutionMode mode) {
            int activeCount = 0, maxActive = 0;
            var gate = new object();
            // Two environments must be executing simultaneously to get past the barrier, so the exact concurrency is
            // asserted rather than relying on the two happening to overlap.
            using var barrier = new Barrier(2);

            StepResult<object?> CountingStepFunc(GenericTestEnv self, object? action) {
                lock (gate) {
                    activeCount++;
                    if (activeCount > maxActive)
                        maxActive = activeCount;
                }

                barrier.SignalAndWait(TimeSpan.FromSeconds(30));
                lock (gate)
                    activeCount--;
                return new StepResult<object?>(self.ObservationSpace.Sample(), 0.0, false, false, new Info());
            }

            var envs = new AsyncVectorEnv(Enumerable.Range(0, 4).Select(_ => (Func<IEnv>)(() => new GenericTestEnv(stepFunc: CountingStepFunc))), maxConcurrency: 2,
                executionMode: mode, threadCount: mode == AsyncExecutionMode.DedicatedThreads ? 2 : null);
            try {
                envs.Reset();
                envs.Step(envs.ActionSpace.Sample());
            } finally {
                envs.Close(null, terminate: true);
                envs.Dispose();
            }

            Assert.AreEqual(2, maxActive);
        }

        /// <summary>Tests that <c>max_concurrency=1</c> fully serializes environment execution.</summary>
        /// <param name="mode">The execution mode.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_async_vector_env.py#L534">tests/vector/test_async_vector_env.py (test_max_concurrency_one_async_vector_env)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Modes))]
        public void TestMaxConcurrencyOneAsyncVectorEnv(AsyncExecutionMode mode) {
            int activeCount = 0, maxActive = 0;
            var gate = new object();

            StepResult<object?> CountingStepFunc(GenericTestEnv self, object? action) {
                lock (gate) {
                    activeCount++;
                    if (activeCount > maxActive)
                        maxActive = activeCount;
                }

                Thread.Sleep(50);
                lock (gate)
                    activeCount--;
                return new StepResult<object?>(self.ObservationSpace.Sample(), 0.0, false, false, new Info());
            }

            using var envs = new AsyncVectorEnv(Enumerable.Range(0, 4).Select(_ => (Func<IEnv>)(() => new GenericTestEnv(stepFunc: CountingStepFunc))), maxConcurrency: 1,
                executionMode: mode);
            envs.Reset();
            envs.Step(envs.ActionSpace.Sample());
            envs.Close();

            Assert.AreEqual(1, maxActive);
        }

        /// <summary>Tests that non-integer <c>max_concurrency</c> values are rejected rather than silently misused.</summary>
        /// <param name="kind">The value's kind: <c>float</c> (2.5), <c>str</c> ("2") or <c>whole</c> (2.0).</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_async_vector_env.py#L564">tests/vector/test_async_vector_env.py (test_non_integer_max_concurrency_async_vector_env)</see>.</remarks>
        [TestMethod]
        [DataRow("float")]
        [DataRow("str")]
        [DataRow("whole")]
        public void TestNonIntegerMaxConcurrencyAsyncVectorEnv(string kind) {
            object maxConcurrency = kind switch {
                "float" => 2.5,
                "str" => "2",
                _ => 2.0,
            };
            var error = Assert.ThrowsExactly<ValueError>(() => new AsyncVectorEnv(Enumerable.Range(0, 2).Select(_ => (Func<IEnv>)(() => new GenericTestEnv())),
                maxConcurrency: maxConcurrency));
            StringAssert.Contains(error.Message, "`max_concurrency` must be an integer or None, got");
        }

        /// <summary>Tests that invalid <c>max_concurrency</c> values raise a <see cref="ValueError"/>.</summary>
        /// <param name="maxConcurrency">The value.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_async_vector_env.py#L580">tests/vector/test_async_vector_env.py (test_invalid_max_concurrency_async_vector_env)</see>.</remarks>
        [TestMethod]
        [DataRow(0)]
        [DataRow(-1)]
        [DataRow(-10)]
        public void TestInvalidMaxConcurrencyAsyncVectorEnv(int maxConcurrency) {
            var error = Assert.ThrowsExactly<ValueError>(() => new AsyncVectorEnv(Enumerable.Range(0, 2).Select(_ => (Func<IEnv>)(() => new GenericTestEnv())),
                maxConcurrency: maxConcurrency));
            StringAssert.Contains(error.Message, "`max_concurrency` must be a positive or None, got");
        }

        /// <summary>A custom <c>worker</c> not accepting the semaphore argument is refused up front.</summary>
        /// <param name="maxConcurrency">The limit, or 0 for none.</param>
        /// <remarks>
        ///     Gymnasium checks a worker's arity when the env is built and raises <c>ValueError</c>; a worker here is an
        ///     <see cref="AsyncVectorWorker"/>, so a C# worker missing the semaphore does not compile (DR-38). The test pins the
        ///     delegate's eight parameters, and a seven-parameter delegate given by Python name is refused by the binding, with
        ///     its <see cref="TypeError"/> naming <c>worker</c>, before any worker starts.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_async_vector_env.py#L589">tests/vector/test_async_vector_env.py (test_incompatible_custom_worker)</see>.
        /// </remarks>
        [TestMethod]
        [DataRow(0)]
        [DataRow(2)]
        public void TestIncompatibleCustomWorker(int maxConcurrency) {
            static void CustomWorker(int index, Func<IEnv> envFn, PipeConnection pipe, PipeConnection parentPipe, object? sharedMemory,
                ConcurrentQueue<AsyncWorkerError> errorQueue, AutoresetMode autoresetMode) { }

            Assert.AreEqual(8, typeof(AsyncVectorWorker).GetMethod("Invoke")!.GetParameters().Length, "Gymnasium's eight worker arguments.");
            var error = Assert.ThrowsExactly<TypeError>(() => PyKwargs.Construct(typeof(AsyncVectorEnv), new Info {
                ["env_fns"] = Enumerable.Range(0, 2).Select(_ => (Func<IEnv>)(() => new GenericTestEnv())).ToList(),
                ["worker"] = (Action<int, Func<IEnv>, PipeConnection, PipeConnection, object?, ConcurrentQueue<AsyncWorkerError>, AutoresetMode>)CustomWorker,
                ["max_concurrency"] = maxConcurrency == 0 ? null : maxConcurrency,
            }));
            StringAssert.Contains(error.Message, "argument 'worker' must be");
        }

        /// <summary>A custom <c>worker</c> with the expected signature works with and without <c>max_concurrency</c>.</summary>
        /// <param name="maxConcurrency">The limit, or 0 for none.</param>
        /// <remarks>
        ///     The worker is Gymnasium's <c>custom_semaphore_worker</c>: the default worker, <see cref="AsyncVectorEnv.AsyncWorker"/>,
        ///     called with every argument it was given.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_async_vector_env.py#L617">tests/vector/test_async_vector_env.py (test_custom_worker_with_semaphore)</see>.
        /// </remarks>
        [TestMethod]
        [DataRow(0)]
        [DataRow(2)]
        public void TestCustomWorkerWithSemaphore(int maxConcurrency) {
            static void CustomSemaphoreWorker(int index, Func<IEnv> envFn, PipeConnection pipe, PipeConnection parentPipe, object? sharedMemory,
                ConcurrentQueue<AsyncWorkerError> errorQueue, AutoresetMode autoresetMode, SemaphoreSlim? semaphore) =>
                AsyncVectorEnv.AsyncWorker(index, envFn, pipe, parentPipe, sharedMemory, errorQueue, autoresetMode, semaphore);

            using var envs = (AsyncVectorEnv)PyKwargs.Construct(typeof(AsyncVectorEnv), new Info {
                ["env_fns"] = Enumerable.Range(0, 4).Select(_ => (Func<IEnv>)(() => new GenericTestEnv())).ToList(),
                ["worker"] = (AsyncVectorWorker)CustomSemaphoreWorker,
                ["max_concurrency"] = maxConcurrency == 0 ? null : maxConcurrency,
            });
            try {
                envs.Reset();
                envs.Step(envs.ActionSpace.Sample());
            } finally {
                envs.Close();
            }
        }

        /// <summary>Tests that <c>max_concurrency</c> greater than the number of environments works.</summary>
        /// <param name="mode">The execution mode.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_async_vector_env.py#L631">tests/vector/test_async_vector_env.py (test_max_concurrency_greater_than_num_envs)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Modes))]
        public void TestMaxConcurrencyGreaterThanNumEnvs(AsyncExecutionMode mode) {
            using var envs = new AsyncVectorEnv(Enumerable.Range(0, 2).Select(_ => (Func<IEnv>)(() => new GenericTestEnv())), maxConcurrency: 4, executionMode: mode);
            envs.Reset();
            envs.Step(envs.ActionSpace.Sample());
            envs.Close();
        }

        /// <summary>
        ///     A worker releases its <c>max_concurrency</c> permit before handing its result over: four sub-envs with large
        ///     observations and one permit still reset within the timeout.
        /// </summary>
        /// <param name="mode">The execution mode.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_async_vector_env.py#L642">tests/vector/test_async_vector_env.py (test_max_concurrency_releases_before_pipe_send)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Modes))]
        public void TestMaxConcurrencyReleasesBeforePipeSend(AsyncExecutionMode mode) {
            using var envs = new AsyncVectorEnv(Enumerable.Range(0, 4).Select(_ => (Func<IEnv>)(() => new GenericTestEnv(observationSpace: new Box(0, 255, new Shape(256, 256, 3), np.uint8)))),
                sharedMemory: false, maxConcurrency: 1, executionMode: mode);
            try {
                envs.ResetAsync();
                var obs = (NDArray)envs.ResetWait(timeout: 10).Observations!;
                CollectionAssert.AreEqual(new long[] { 4, 256, 256, 3 }, obs.shape);
            } finally {
                envs.Close(null, terminate: true);
            }
        }

        /// <summary><c>raise_error_reset</c>: seeds the env, and raises <c>Error in reset</c> for seed 1.</summary>
        /// <param name="self">The env.</param>
        /// <param name="seed">The seed.</param>
        /// <param name="options">Ignored.</param>
        /// <returns>A sample and an empty info.</returns>
        /// <exception cref="ValueError">Thrown for seed 1.</exception>
        private static ResetResult<object?> RaiseErrorReset(GenericTestEnv self, BigInteger? seed, IReadOnlyDictionary<string, object?>? options) {
            self.BaseReset(seed);
            if (seed == 1)
                throw new ValueError("Error in reset");
            return new ResetResult<object?>(self.ObservationSpace.Sample(), new Info());
        }

        /// <summary><c>raise_error_step</c>: raises <c>Error in step with {action}</c> for an action of 1 or more.</summary>
        /// <param name="self">The env.</param>
        /// <param name="action">The action, a number.</param>
        /// <returns>A sample, a reward of 0, and no end.</returns>
        /// <exception cref="ValueError">Thrown for an action of 1 or more.</exception>
        private static StepResult<object?> RaiseErrorStep(GenericTestEnv self, object? action) {
            if (PyBuiltins.Float(action) >= 1)
                throw new ValueError($"Error in step with {PyBuiltins.Str(action)}");
            return new StepResult<object?>(self.ObservationSpace.Sample(), 0.0, false, false, new Info());
        }
    }
}
