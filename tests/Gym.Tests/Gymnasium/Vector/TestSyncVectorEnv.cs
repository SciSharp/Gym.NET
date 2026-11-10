#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Spaces;
using Gym.Tests.Gymnasium.Envs;
using Gym.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using SpaceTuple = Gym.Spaces.Tuple;

namespace Gym.Tests.Gymnasium.Vector {
    /// <summary>Gymnasium's <c>SyncVectorEnv</c> tests (<c>test_sync_vector_env.py</c>), run against <see cref="SyncVectorEnv"/>.</summary>
    /// <remarks>
    ///     The seed test runs over BipedalWalker-v3, which arrives with Box2D (P8), so it is <c>[OpenBugs("P8")]</c>; the
    ///     determinism test runs over every env Gym.NET registers, as Gymnasium's runs over every one it can make.
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_sync_vector_env.py">tests/vector/test_sync_vector_env.py</see>.
    /// </remarks>
    [TestClass]
    public class TestSyncVectorEnv {
        /// <summary>The ids of every registered env Gym.NET can make (<c>all_testing_env_specs</c>).</summary>
        public static IEnumerable<object[]> AllTestingEnvIds => EnvTestingUtils.AllTestingEnvSpecs().Select(spec => new object[] { spec.Id });

        /// <summary>Tests creating the sync vector environment.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_sync_vector_env.py#L20">tests/vector/test_sync_vector_env.py (test_create_sync_vector_env)</see>.</remarks>
        [TestMethod]
        public void TestCreateSyncVectorEnv() {
            Func<IEnv>[] envFns = Enumerable.Range(0, 8).Select(i => TestingUtils.MakeEnv("FrozenLake-v1", i)).ToArray();
            using var env = new SyncVectorEnv(envFns);
            env.Close();

            Assert.AreEqual(8, env.NumEnvs);
        }

        /// <summary>Tests that the vector env's metadata doesn't mutate the sub-environment's (class-level) metadata.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_sync_vector_env.py#L29">tests/vector/test_sync_vector_env.py (test_metadata_sync_vector_env)</see>.</remarks>
        [TestMethod]
        public void TestMetadataSyncVectorEnv() {
            using var envs1 = new SyncVectorEnv(new[] { TestingUtils.MakeEnv("CartPole-v1", 0) }, autoresetMode: AutoresetMode.NextStep);
            using var envs2 = new SyncVectorEnv(new[] { TestingUtils.MakeEnv("CartPole-v1", 1) }, autoresetMode: AutoresetMode.SameStep);

            Assert.AreSame(AutoresetMode.NextStep, envs1.Metadata["autoreset_mode"]);
            Assert.AreSame(AutoresetMode.SameStep, envs2.Metadata["autoreset_mode"]);
            Assert.IsFalse(envs1.Envs[0].Metadata.ContainsKey("autoreset_mode"));

            envs1.Close();
            envs2.Close();
        }

        /// <summary>Tests sync vector <c>reset</c>: a batched <c>Box</c> observation of the space's dtype and shape.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_sync_vector_env.py#L46">tests/vector/test_sync_vector_env.py (test_reset_sync_vector_env)</see>.</remarks>
        [TestMethod]
        public void TestResetSyncVectorEnv() {
            Func<IEnv>[] envFns = Enumerable.Range(0, 8).Select(i => TestingUtils.MakeEnv("CartPole-v1", i)).ToArray();
            using var env = new SyncVectorEnv(envFns);
            (object? observations, _) = env.Reset();
            env.Close();

            var space = (Box)ObjectSpace.Unview(env.ObservationSpace)!;
            Assert.IsInstanceOfType<NDArray>(observations);
            var batch = (NDArray)observations!;
            Assert.IsTrue(batch.dtype == space.DType!);
            CollectionAssert.AreEqual(new long[] { 8 }.Concat(ObjectSpace.Unview(env.SingleObservationSpace)!.Shape!.Value.dimensions).ToArray(), batch.shape);
            CollectionAssert.AreEqual(space.Shape.dimensions, batch.shape);
            batch.Dispose();
        }

        /// <summary>Test sync vector <c>step</c>: FrozenLake's batched observations, float rewards and boolean flags, from batched or single-sampled actions.</summary>
        /// <param name="useSingleActionSpace">Whether the actions are a list of single samples rather than a batched sample.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_sync_vector_env.py#L62">tests/vector/test_sync_vector_env.py (test_step_sync_vector_env)</see>.</remarks>
        [TestMethod]
        [DataRow(true)]
        [DataRow(false)]
        public void TestStepSyncVectorEnv(bool useSingleActionSpace) {
            using var env = new SyncVectorEnv(Enumerable.Range(0, 8).Select(i => TestingUtils.MakeEnv("FrozenLake-v1", i)));
            env.Reset();

            Assert.IsInstanceOfType<Discrete>(ObjectSpace.Unview(env.SingleActionSpace));
            Assert.IsInstanceOfType<MultiDiscrete>(ObjectSpace.Unview(env.ActionSpace));

            object? actions = useSingleActionSpace ? Enumerable.Range(0, 8).Select(_ => env.SingleActionSpace.Sample()).ToList() : env.ActionSpace.Sample();
            (object? observations, NDArray rewards, NDArray terminations, NDArray truncations, _) = env.Step(actions);

            env.Close();

            var space = (MultiDiscrete)ObjectSpace.Unview(env.ObservationSpace)!;
            Assert.IsInstanceOfType<NDArray>(observations);
            var batch = (NDArray)observations!;
            Assert.IsTrue(batch.dtype == space.DType!);
            CollectionAssert.AreEqual(new long[] { 8 }.Concat(ObjectSpace.Unview(env.SingleObservationSpace)!.Shape!.Value.dimensions).ToArray(), batch.shape);
            CollectionAssert.AreEqual(space.Shape.dimensions, batch.shape);

            // isinstance(rewards[0], (float, np.floating)): float64 rewards (DR-21).
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

        /// <summary>Rendering three <c>rgb_array</c> CartPoles gives a frame each; without a render mode the vector env has none.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_sync_vector_env.py#L105">tests/vector/test_sync_vector_env.py (test_render_sync_vector)</see>.</remarks>
        [TestMethod]
        public void TestRenderSyncVector() {
            using (var envs = new SyncVectorEnv(Enumerable.Range(0, 3).Select(i => TestingUtils.MakeEnv("CartPole-v1", i, new { render_mode = "rgb_array" })))) {
                Assert.AreEqual("rgb_array", envs.RenderMode);

                envs.Reset();
                IReadOnlyList<RenderResult>? renderedFrames = envs.Render();
                Assert.IsNotNull(renderedFrames);
                Assert.AreEqual(envs.NumEnvs, renderedFrames.Count);
                Assert.IsTrue(renderedFrames.All(frame => frame is RenderResult.Rgb));
                foreach (RenderResult frame in renderedFrames)
                    frame.Dispose();
            }

            using (var envs = new SyncVectorEnv(Enumerable.Range(0, 3).Select(i => TestingUtils.MakeEnv("CartPole-v1", i))))
                Assert.IsNull(envs.RenderMode);
        }

        /// <summary>Test sync vector <c>call</c> on sub-environments: <c>render</c> of <c>rgb_array_list</c> CartPoles and their <c>gravity</c>.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_sync_vector_env.py#L121">tests/vector/test_sync_vector_env.py (test_call_sync_vector_env)</see>.</remarks>
        [TestMethod]
        public void TestCallSyncVectorEnv() {
            Func<IEnv>[] envFns = Enumerable.Range(0, 4).Select(i => TestingUtils.MakeEnv("CartPole-v1", i, new { render_mode = "rgb_array_list" })).ToArray();

            using var env = new SyncVectorEnv(envFns);
            env.Reset();
            IReadOnlyList<object?> images = env.Call("render");
            IReadOnlyList<object?> gravity = env.Call("gravity");

            env.Close();

            Assert.AreEqual(4, images.Count);
            for (int i = 0; i < 4; i++) {
                var frames = (RenderResult.Frames)images[i]!;
                Assert.AreEqual(1, frames.Items.Count);
                Assert.IsInstanceOfType<RenderResult.Rgb>(frames.Items[0]);
                frames.Dispose();
            }

            Assert.AreEqual(4, gravity.Count);
            for (int i = 0; i < 4; i++) {
                Assert.IsInstanceOfType<double>(gravity[i]);
                Assert.AreEqual(9.8, (double)gravity[i]!);
            }
        }

        /// <summary>Test sync vector <c>set_attr</c>: one value per sub-env, read back in order.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_sync_vector_env.py#L148">tests/vector/test_sync_vector_env.py (test_set_attr_sync_vector_env)</see>.</remarks>
        [TestMethod]
        public void TestSetAttrSyncVectorEnv() {
            Func<IEnv>[] envFns = Enumerable.Range(0, 4).Select(i => TestingUtils.MakeEnv("CartPole-v1", i)).ToArray();

            using var env = new SyncVectorEnv(envFns);
            env.SetAttr("gravity", new List<object?> { 9.81, 3.72, 8.87, 1.62 });
            IReadOnlyList<object?> gravity = env.GetAttr("gravity");
            CollectionAssert.AreEqual(new object?[] { 9.81, 3.72, 8.87, 1.62 }, gravity.ToArray());

            env.Close();
        }

        /// <summary>Tests the sync vector <c>check_spaces</c>: a FrozenLake among CartPoles is refused with Gymnasium's message.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_sync_vector_env.py#L160">tests/vector/test_sync_vector_env.py (test_check_spaces_sync_vector_env)</see>.</remarks>
        [TestMethod]
        public void TestCheckSpacesSyncVectorEnv() {
            // CartPole-v1 - observation_space: Box(4,), action_space: Discrete(2)
            Func<IEnv>[] envFns = Enumerable.Range(0, 8).Select(i => TestingUtils.MakeEnv("CartPole-v1", i)).ToArray();
            // FrozenLake-v1 - Discrete(16), action_space: Discrete(4)
            envFns[1] = TestingUtils.MakeEnv("FrozenLake-v1", 1);
            var error = Assert.ThrowsExactly<RuntimeError>(() => {
                var env = new SyncVectorEnv(envFns);
                env.Close();
            });
            StringAssert.Contains(error.Message, "SyncVectorEnv(..., observation_mode='same') however the sub-environments observation spaces are not equivalent.");
        }

        /// <summary>Test the use of custom spaces with sync vector environment: string observations batched into a tuple.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_sync_vector_env.py#L176">tests/vector/test_sync_vector_env.py (test_custom_space_sync_vector_env)</see>.</remarks>
        [TestMethod]
        public void TestCustomSpaceSyncVectorEnv() {
            Func<IEnv>[] envFns = Enumerable.Range(0, 4).Select(i => TestingUtils.MakeCustomSpaceEnv(i)).ToArray();

            using var env = new SyncVectorEnv(envFns);
            (object? resetObservations, _) = env.Reset();

            Assert.IsInstanceOfType<TestingUtils.CustomSpace>(ObjectSpace.Unview(env.SingleActionSpace));
            Assert.IsInstanceOfType<SpaceTuple>(ObjectSpace.Unview(env.ActionSpace));

            object?[] actions = { "action-2", "action-3", "action-5", "action-7" };
            (object? stepObservations, _, _, _, _) = env.Step(actions);

            env.Close();

            Assert.IsInstanceOfType<TestingUtils.CustomSpace>(ObjectSpace.Unview(env.SingleObservationSpace));
            Assert.IsInstanceOfType<SpaceTuple>(ObjectSpace.Unview(env.ObservationSpace));

            CollectionAssert.AreEqual(new object?[] { "reset", "reset", "reset", "reset" }, (object?[])resetObservations!);
            CollectionAssert.AreEqual(new object?[] { "step(action-2)", "step(action-3)", "step(action-5)", "step(action-7)" }, (object?[])stepObservations!);
        }

        /// <summary>Test seeding for sync vector environments: the vector env's action space samples as the single env's does.</summary>
        /// <remarks>
        ///     BipedalWalker-v3 arrives with Box2D (P8).
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_sync_vector_env.py#L205">tests/vector/test_sync_vector_env.py (test_sync_vector_env_seed)</see>.
        /// </remarks>
        [TestMethod]
        [OpenBugs("P8")]
        public void TestSyncVectorEnvSeed() {
            using IEnv env = TestingUtils.MakeEnv("BipedalWalker-v3", seed: 123)();
            using var syncVectorEnv = new SyncVectorEnv(new[] { TestingUtils.MakeEnv("BipedalWalker-v3", seed: 123) });

            TestingUtils.AssertRngEqual(env.ActionSpace.NpRandom, syncVectorEnv.ActionSpace.NpRandom);
            for (int i = 0; i < 100; i++) {
                using var envAction = (NDArray)env.ActionSpace.Sample()!;
                using var vectorAction = (NDArray)syncVectorEnv.ActionSpace.Sample()!;
                Assert.IsTrue(np.array_equal(envAction, vectorAction));
            }

            env.Close();
        }

        /// <summary>For every env, two sync vector envs of the same seeds sample the same actions (100 batches).</summary>
        /// <param name="envId">The env's id.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_sync_vector_env.py#L216">tests/vector/test_sync_vector_env.py (test_sync_vector_determinism)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(AllTestingEnvIds))]
        public void TestSyncVectorDeterminism(string envId) {
            const int seed = 123, n = 3;
            using var env1 = new SyncVectorEnv(Enumerable.Range(0, n).Select(_ => TestingUtils.MakeEnv(envId, seed)));
            using var env2 = new SyncVectorEnv(Enumerable.Range(0, n).Select(_ => TestingUtils.MakeEnv(envId, seed)));
            TestingUtils.AssertRngEqual(env1.ActionSpace.NpRandom, env2.ActionSpace.NpRandom);

            for (int i = 0; i < 100; i++) {
                object? env1Samples = env1.ActionSpace.Sample();
                object? env2Samples = env2.ActionSpace.Sample();
                Assert.IsTrue(Gym.Utils.PyEquality.AreEqual(env1Samples, env2Samples), $"{envId}: sample {i}");
                (env1Samples as IDisposable)?.Dispose();
                (env2Samples as IDisposable)?.Dispose();
            }

            env1.Close();
            env2.Close();
        }
    }
}
