#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Gym.Envs;
using Gym.Observations;
using Gym.Spaces;
using Gym.Tests.Gymnasium.Spaces;
using Gym.Utils;
using Gym.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Vector {
    /// <summary>
    ///     Gymnasium's vector env tests (<c>test_vector_env.py</c>), run against <see cref="SyncVectorEnv"/> and
    ///     <see cref="AsyncVectorEnv"/> with and without shared memory.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Gymnasium's <c>vector_env.np_random_seed</c> is a tuple of the sub-envs' seeds; Gym.NET's keeps the base class's
    ///     meaning (the vector env's own seed, a documented deviation), so the seeding tests read the sub-envs' seeds with
    ///     <c>get_attr("np_random_seed")</c>, which is what Gymnasium's property returns.
    ///     </para>
    ///     <para>
    ///     The space catalog test builds a space per sub-env and per use, as the async port does: Gymnasium's lambdas share
    ///     one space, which processes copy, threads would sample at once, and a Gym.NET env owns and disposes (D19).
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_vector_env.py">tests/vector/test_vector_env.py</see>.</para>
    /// </remarks>
    [TestClass]
    public class TestVectorEnv {
        /// <summary>Gymnasium's three vectorisers, by its ids.</summary>
        public static IEnumerable<object[]> Vectorisers => new[] { "Sync", "Async with shared memory", "Async without shared memory" }.Select(id => new object[] { id });

        /// <summary>Shared memory or not, with each autoreset mode <c>test_vector_env_equal</c> runs.</summary>
        public static IEnumerable<object[]> SharedMemoryAndModes =>
            from sharedMemory in new[] { true, false } from mode in new[] { "NextStep", "SameStep" } select new object[] { sharedMemory, mode };

        /// <summary>The vectorisers with every space of Gymnasium's testing catalog (<c>TESTING_SPACES</c>), by index.</summary>
        public static IEnumerable<object[]> VectorisersAndSpaces =>
            from id in Vectorisers.Select(row => (string)row[0]) from index in Enumerable.Range(0, TestingSpaces.All.Length) select new object[] { id, index };

        /// <summary>Async and sync vector envs over the same CartPoles agree for 100 steps of sampled actions: spaces, results and infos.</summary>
        /// <param name="sharedMemory">Whether the async env's observations go through the shared buffer.</param>
        /// <param name="autoresetMode">The autoreset mode's value.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_vector_env.py#L22">tests/vector/test_vector_env.py (test_vector_env_equal)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(SharedMemoryAndModes))]
        public void TestVectorEnvEqual(bool sharedMemory, string autoresetMode) {
            Func<IEnv>[] envFns = Enumerable.Range(0, 4).Select(i => TestingUtils.MakeEnv("CartPole-v1", i)).ToArray();
            const int numSteps = 100;

            using var asyncEnv = new AsyncVectorEnv(envFns, sharedMemory: sharedMemory, autoresetMode: autoresetMode);
            using var syncEnv = new SyncVectorEnv(envFns, autoresetMode: autoresetMode);

            Assert.AreEqual(asyncEnv.NumEnvs, syncEnv.NumEnvs);
            Assert.AreEqual(ObjectSpace.Unview(asyncEnv.ObservationSpace), ObjectSpace.Unview(syncEnv.ObservationSpace));
            Assert.AreEqual(ObjectSpace.Unview(asyncEnv.SingleObservationSpace), ObjectSpace.Unview(syncEnv.SingleObservationSpace));
            Assert.AreEqual(ObjectSpace.Unview(asyncEnv.ActionSpace), ObjectSpace.Unview(syncEnv.ActionSpace));
            Assert.AreEqual(ObjectSpace.Unview(asyncEnv.SingleActionSpace), ObjectSpace.Unview(syncEnv.SingleActionSpace));

            (object? asyncObservations, Info asyncInfos) = asyncEnv.Reset(seed: 0);
            (object? syncObservations, Info syncInfos) = syncEnv.Reset(seed: 0);
            Assert.IsTrue(np.array_equal((NDArray)asyncObservations!, (NDArray)syncObservations!));
            Assert.IsTrue(EnvChecker.DataEquivalence(asyncInfos, syncInfos));

            for (int step = 0; step < numSteps; step++) {
                object? actions = asyncEnv.ActionSpace.Sample();
                Assert.IsTrue(syncEnv.ActionSpace.Contains(actions));

                VectorStepResult<object?> asyncResult = asyncEnv.Step(actions);
                VectorStepResult<object?> syncResult = syncEnv.Step(actions);

                Assert.IsTrue(np.array_equal((NDArray)asyncResult.Observations!, (NDArray)syncResult.Observations!), $"step {step}: observations");
                Assert.IsTrue(np.array_equal(asyncResult.Rewards, syncResult.Rewards), $"step {step}: rewards");
                Assert.IsTrue(np.array_equal(asyncResult.Terminations, syncResult.Terminations), $"step {step}: terminations");
                Assert.IsTrue(np.array_equal(asyncResult.Truncations, syncResult.Truncations), $"step {step}: truncations");
                Assert.IsTrue(EnvChecker.DataEquivalence(asyncResult.Infos, syncResult.Infos), $"step {step}: infos");
            }

            asyncEnv.Close();
            syncEnv.Close();
        }

        /// <summary>
        ///     Every space of the catalog works as both spaces of a vector env (the dynamic-shape ones are skipped with shared
        ///     memory, which refuses them): the spaces are equal, the reset is a member, and a sampled step runs.
        /// </summary>
        /// <param name="vectoriser">The vectoriser.</param>
        /// <param name="index">The space's index in the catalog.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_vector_env.py#L80">tests/vector/test_vector_env.py (test_vector_obs_action_spaces)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(VectorisersAndSpaces))]
        public void TestVectorObsActionSpaces(string vectoriser, int index) {
            const int numEnvs = 3;
            IVectorEnv envs;
            try {
                // A space per use: an env owns and disposes both of its spaces (D19), where Gymnasium's lambdas share one.
                envs = Vectorise(vectoriser, Enumerable.Range(0, numEnvs).Select(_ => (Func<IEnv>)(() =>
                    new GenericTestEnv(actionSpace: TestingSpaces.All[index](), observationSpace: TestingSpaces.All[index](), stepFunc: DebugStepFunc))));
            } catch (TypeError error) {
                StringAssert.Contains(error.Message, "has a dynamic shape so its not possible to make a static shared memory.");
                Assert.Inconclusive("Skipping space with dynamic shape");
                return;
            }

            using (envs) {
                Assert.AreEqual(ObjectSpace.Unview(envs.ObservationSpace), ObjectSpace.Unview(envs.ActionSpace));

                (object? obs, _) = envs.Reset();
                Assert.IsTrue(envs.ObservationSpace.Contains(obs));
                envs.Step(envs.ActionSpace.Sample());

                envs.Close();
            }
        }

        /// <summary>
        ///     The final observation and info: a sub-env whose step ends the episode returns its last observation and info,
        ///     and the next step (<c>NEXT_STEP</c>) resets it, returning the reset's info.
        /// </summary>
        /// <param name="vectoriser">The vectoriser.</param>
        /// <remarks>
        ///     Gymnasium's last check reads <c>termination</c>, the flag of the step before (its own name for the last step's is
        ///     <c>terminated</c>); the port keeps that.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_vector_env.py#L124">tests/vector/test_vector_env.py (test_final_obs_info)</see>.
        /// </remarks>
        [TestMethod]
        [DynamicData(nameof(Vectorisers))]
        public void TestFinalObsInfo(string vectoriser) {
            Func<IEnv> thunk = () => new GenericTestEnv(
                actionSpace: new Discrete(4), observationSpace: new Discrete(4),
                resetFunc: (self, seed, options) => new ResetResult<object?>(0L, new Info { ["reset"] = true }),
                stepFunc: (self, action) => {
                    long value = Convert.ToInt64(action is NDArray scalar ? scalar.GetAtIndex(0) : action);
                    return new StepResult<object?>(value < 3 ? value : 0L, 0, value >= 3, false, new Info { ["action"] = value });
                });

            using IVectorEnv env = Vectorise(vectoriser, new[] { thunk });
            (object? obs, Info info) = env.Reset();
            Assert.IsTrue(PyEquality.AreEqual(obs, np.array(new[] { 0L })) && PyEquality.AreEqual(info, new Info { ["reset"] = np.array(new[] { true }), ["_reset"] = np.array(new[] { true }) }));

            VectorStepResult<object?> result = env.Step(Actions(1));
            Assert.IsTrue(PyEquality.AreEqual(result.Observations, np.array(new[] { 1L })) && PyEquality.AreEqual(result.Terminations, np.array(new[] { false }))
                          && PyEquality.AreEqual(result.Infos, new Info { ["action"] = np.array(new[] { 1L }), ["_action"] = np.array(new[] { true }) }));

            result = env.Step(Actions(2));
            Assert.IsTrue(PyEquality.AreEqual(result.Observations, np.array(new[] { 2L })) && PyEquality.AreEqual(result.Terminations, np.array(new[] { false }))
                          && PyEquality.AreEqual(result.Infos, new Info { ["action"] = np.array(new[] { 2L }), ["_action"] = np.array(new[] { true }) }));

            result = env.Step(Actions(3));
            NDArray termination = result.Terminations;
            Assert.IsTrue(PyEquality.AreEqual(result.Observations, np.array(new[] { 0L })) && PyEquality.AreEqual(result.Infos, new Info { ["action"] = 3L, ["_action"] = np.array(new[] { true }) }));

            result = env.Step(Actions(4));
            Assert.IsTrue(PyEquality.AreEqual(result.Observations, np.array(new[] { 0L })) && PyEquality.AreEqual(termination, np.array(new[] { true }))
                          && PyEquality.AreEqual(result.Infos["reset"], np.array(new[] { true })));

            env.Close();
        }

        /// <summary>A reset with seed 42 gives the sub-envs the seeds 42, 43, ...; a reset without one keeps them.</summary>
        /// <param name="vectoriser">The vectoriser.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_vector_env.py#L180">tests/vector/test_vector_env.py (test_random_seeding_basics)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Vectorisers))]
        public void TestRandomSeedingBasics(string vectoriser) {
            const int seed = 42;
            using IVectorEnv vectorEnv = Vectorise(vectoriser, ExampleEnvList());
            vectorEnv.Reset(seed: seed);
            CollectionAssert.AreEqual(Enumerable.Range(0, vectorEnv.NumEnvs).Select(i => (object)new BigInteger(seed + i)).ToArray(), SubEnvSeeds(vectorEnv));
            // resetting with seed=None means seed remains the same
            vectorEnv.Reset(seed: null);
            CollectionAssert.AreEqual(Enumerable.Range(0, vectorEnv.NumEnvs).Select(i => (object)new BigInteger(seed + i)).ToArray(), SubEnvSeeds(vectorEnv));
        }

        /// <summary>Before any reset the sub-envs report distinct seeds, the ones <c>make_env</c> reset them with (0, 1, 2, 3).</summary>
        /// <param name="vectoriser">The vectoriser.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_vector_env.py#L203">tests/vector/test_vector_env.py (test_random_seeds_set_at_retrieval)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Vectorisers))]
        public void TestRandomSeedsSetAtRetrieval(string vectoriser) {
            using IVectorEnv vectorEnv = Vectorise(vectoriser, ExampleEnvList());
            object[] seeds = SubEnvSeeds(vectorEnv);
            Assert.AreEqual(vectorEnv.NumEnvs, seeds.Distinct().Count());
            // default seed starts at zero. Adjust or remove this test if the default seed changes
            CollectionAssert.AreEqual(Enumerable.Range(0, vectorEnv.NumEnvs).Select(i => (object)new BigInteger(i)).ToArray(), seeds);
        }

        /// <summary>A masked reset resets only the masked sub-envs, with their seeds; the unmasked one keeps its stepped observation.</summary>
        /// <param name="vectoriser">The vectoriser.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_vector_env.py#L215">tests/vector/test_vector_env.py (test_partial_reset)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Vectorisers))]
        public void TestPartialReset(string vectoriser) {
            using IVectorEnv envs = Vectorise(vectoriser, Enumerable.Range(0, 3).Select(_ => (Func<IEnv>)(() => Make("CartPole-v1"))), AutoresetMode.Disabled);
            (object? resetObs, _) = envs.Reset(new BigInteger?[] { 0, 1, 2 });
            var reset = (NDArray)resetObs!;

            envs.ActionSpace.Seed(123);
            envs.Step(envs.ActionSpace.Sample());
            envs.Step(envs.ActionSpace.Sample());
            var stepObs = (NDArray)envs.Step(envs.ActionSpace.Sample()).Observations!;

            (object? maskedObs, _) = envs.Reset(new BigInteger?[] { 0, 1, 0 }, new Info { ["reset_mask"] = np.array(new[] { true, true, false }) });
            var masked = (NDArray)maskedObs!;
            Assert.IsTrue(np.array_equal(masked["0:2"], reset["0:2"]));
            Assert.IsTrue(np.array_equal(masked[2], stepObs[2]));

            envs.Close();
        }

        /// <summary>A reset mask of the wrong shape, with no true value, or of a non-boolean dtype is refused with Gymnasium's messages.</summary>
        /// <param name="vectoriser">The vectoriser.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_vector_env.py#L242">tests/vector/test_vector_env.py (test_partial_reset_failure)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Vectorisers))]
        public void TestPartialResetFailure(string vectoriser) {
            using IVectorEnv envs = Vectorise(vectoriser, Enumerable.Range(0, 3).Select(_ => (Func<IEnv>)(() => Make("CartPole-v1"))), AutoresetMode.Disabled);

            // Reset with all trues
            envs.Reset(options: new Info { ["reset_mask"] = np.array(new[] { true, true, true }) });

            // Reset with mask of an incorrect shape
            StringAssert.Contains(Assert.ThrowsExactly<ValueError>(() => envs.Reset(options: new Info { ["reset_mask"] = np.array(new[] { true }) })).Message,
                "`options['reset_mask']` must have shape `(3,)`, got (1,)");
            StringAssert.Contains(Assert.ThrowsExactly<ValueError>(() => envs.Reset(options: new Info { ["reset_mask"] = np.array(new[] { true, true, false, false }) })).Message,
                "`options['reset_mask']` must have shape `(3,)`, got (4,)");
            StringAssert.Contains(Assert.ThrowsExactly<ValueError>(() => envs.Reset(options: new Info { ["reset_mask"] = np.array(new[,] { { true, true, true } }) })).Message,
                "`options['reset_mask']` must have shape `(3,)`, got (1, 3)");
            StringAssert.Contains(Assert.ThrowsExactly<ValueError>(() => envs.Reset(options: new Info { ["reset_mask"] = np.array(new[] { false, false, false }) })).Message,
                "`options['reset_mask']` must contain a boolean array with at least one True value, got reset_mask=[False False False]");
            StringAssert.Contains(Assert.ThrowsExactly<TypeError>(() => envs.Reset(options: new Info { ["reset_mask"] = np.array(new[] { 1L, 1L, 0L }) })).Message,
                "`options['reset_mask']` must have `dtype=np.bool_`, got int64");
            StringAssert.Contains(Assert.ThrowsExactly<TypeError>(() => envs.Reset(options: new Info { ["reset_mask"] = np.array(new[] { 1.0, 1.0, 0.0 }) })).Message,
                "`options['reset_mask']` must have `dtype=np.bool_`, got float64");
        }

        /// <summary>
        ///     Stepping takes exactly one action per sub-env: the batch steps, one action too few or too many is refused
        ///     (<c>ValueError</c>), and a scalar cannot be iterated (<c>TypeError</c>).
        /// </summary>
        /// <param name="vectoriser">The vectoriser.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_vector_env.py#L310">tests/vector/test_vector_env.py (test_action_count_compatibility)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Vectorisers))]
        public void TestActionCountCompatibility(string vectoriser) {
            const int numEnvs = 4;
            using IVectorEnv envs = Vectorise(vectoriser, Enumerable.Range(0, numEnvs).Select(_ => (Func<IEnv>)(() => Make("CartPole-v1"))), AutoresetMode.Disabled);

            // Reset the environment
            envs.Reset();

            // Test correct number of actions (should work)
            var correctActions = (NDArray)envs.ActionSpace.Sample()!;
            Assert.AreEqual(numEnvs, (int)correctActions.shape[0]);

            // Test with actions that match the number of environments
            VectorStepResult<object?> result = envs.Step(correctActions);
            Assert.AreEqual(numEnvs, (int)((NDArray)result.Observations!).shape[0]);
            Assert.AreEqual(numEnvs, (int)result.Rewards.size);
            Assert.AreEqual(numEnvs, (int)result.Terminations.size);
            Assert.AreEqual(numEnvs, (int)result.Truncations.size);

            // Test with too few actions (should raise error)
            Assert.ThrowsExactly<ValueError>(() => envs.Step(correctActions[$":{numEnvs - 1}"]));

            // Test with too many actions (should raise error)
            Assert.ThrowsExactly<ValueError>(() => envs.Step(np.concatenate(new[] { correctActions, correctActions[":1"] })));

            // Test with scalar action (should raise error for vector env)
            Assert.ThrowsExactly<TypeError>(() => envs.Step(0));

            envs.Close();
        }

        /// <summary><c>debug_step_func</c>: asserts the action is in the space, returns a sampled observation, reward 0 and no flags.</summary>
        /// <param name="self">The env.</param>
        /// <param name="action">The action.</param>
        /// <returns>The step.</returns>
        /// <exception cref="AssertionError">Thrown when the action is not in the action space (a bare assertion).</exception>
        private static StepResult<object?> DebugStepFunc(GenericTestEnv self, object? action) {
            if (!self.ActionSpace.Contains(action))
                throw new AssertionError("");
            return new StepResult<object?>(self.ObservationSpace.Sample(), 0, false, false, new Info());
        }

        /// <summary>The <c>example_env_list</c> fixture: four CartPoles made by <c>make_env</c> with seeds 0 to 3.</summary>
        /// <returns>The functions.</returns>
        private static Func<IEnv>[] ExampleEnvList() => Enumerable.Range(0, 4).Select(i => TestingUtils.MakeEnv("CartPole-v1", i)).ToArray();

        /// <summary>Gymnasium's <c>vector_env.np_random_seed</c>: the sub-envs' seeds, through <c>get_attr</c>.</summary>
        /// <param name="env">The vector env.</param>
        /// <returns>The seeds.</returns>
        private static object[] SubEnvSeeds(IVectorEnv env) =>
            (env is SyncVectorEnv sync ? sync.GetAttr("np_random_seed") : ((AsyncVectorEnv)env).GetAttr("np_random_seed")).Select(seed => seed!).ToArray();

        /// <summary>A list of Python int actions.</summary>
        /// <param name="actions">The actions.</param>
        /// <returns>The list.</returns>
        private static List<object?> Actions(params long[] actions) => actions.Select(action => (object?)action).ToList();

        /// <summary>Builds the vectoriser of the id over the functions.</summary>
        /// <param name="id">Gymnasium's vectoriser id.</param>
        /// <param name="envFns">The sub-envs' functions.</param>
        /// <param name="autoresetMode">The autoreset mode, or <see langword="null"/> for the default.</param>
        /// <returns>The vector env; the caller disposes it.</returns>
        private static IVectorEnv Vectorise(string id, IEnumerable<Func<IEnv>> envFns, object? autoresetMode = null) => id switch {
            "Sync" => new SyncVectorEnv(envFns, autoresetMode: autoresetMode),
            "Async with shared memory" => new AsyncVectorEnv(envFns, sharedMemory: true, autoresetMode: autoresetMode),
            _ => new AsyncVectorEnv(envFns, sharedMemory: false, autoresetMode: autoresetMode),
        };
    }
}
