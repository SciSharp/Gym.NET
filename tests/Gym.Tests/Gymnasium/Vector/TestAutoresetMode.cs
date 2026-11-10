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
    ///     Gymnasium's autoreset tests (<c>test_autoreset_mode.py</c>), run against <see cref="SyncVectorEnv"/> and
    ///     <see cref="AsyncVectorEnv"/> with and without shared memory, and through <c>make_vec</c>.
    /// </summary>
    /// <remarks>
    ///     The counting env stores <c>count</c> and <c>max_count</c> as run-time attributes, as Gymnasium's functions do on the
    ///     instance; the reward is the action, a float64 as every Gym.NET reward. <c>None</c> in <c>final_obs</c> is
    ///     <see langword="null"/> in Gym.NET's object array (an <see langword="object"/>[]).
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_autoreset_mode.py">tests/vector/test_autoreset_mode.py</see>.
    /// </remarks>
    [TestClass]
    public class TestAutoresetMode {
        /// <summary>Gymnasium's three vectorisers, by its ids.</summary>
        public static IEnumerable<object[]> Vectorisers => new[] { "Sync", "Async(shared_memory=True)", "Async(shared_memory=False)" }.Select(id => new object[] { id });

        /// <summary>The vectorisers with every autoreset mode.</summary>
        public static IEnumerable<object[]> VectorisersAndModes =>
            from id in Vectorisers.Select(row => (string)row[0]) from mode in new[] { "NextStep", "Disabled", "SameStep" } select new object[] { id, mode };

        /// <summary><c>make_vec</c>'s two modes with every autoreset mode.</summary>
        public static IEnumerable<object[]> VectorizationModesAndAutoresetModes =>
            from vectorization in new[] { "sync", "async" } from mode in new[] { "NextStep", "Disabled", "SameStep" } select new object[] { vectorization, mode };

        /// <summary>Every space of Gymnasium's testing catalog (<c>TESTING_SPACES</c>), by index.</summary>
        public static IEnumerable<object[]> TestingSpaceIndexes => TestingSpaces.AllIndexes;

        /// <summary>
        ///     <c>NEXT_STEP</c>: an ended sub-env resets on the next step, which returns its reset observation with reward 0 and
        ///     no flags; the infos stay empty.
        /// </summary>
        /// <param name="vectoriser">The vectoriser.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_autoreset_mode.py#L42">tests/vector/test_autoreset_mode.py (test_autoreset_next_step)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Vectorisers))]
        public void TestAutoresetNextStep(string vectoriser) {
            using IVectorEnv envs = Vectorise(vectoriser, Enumerable.Range(0, 3).Select(_ => CountingEnv()), AutoresetMode.NextStep);
            Assert.AreSame(AutoresetMode.NextStep, envs.Metadata["autoreset_mode"]);
            SetAttr(envs, "max_count", new List<object?> { 2L, 3L, 3L });

            (object? obs, Info info) = envs.Reset();
            AssertLongs(obs, 0, 0, 0);
            Assert.AreEqual(0, info.Count);

            AssertStep(envs.Step(Actions(1, 2, 3)), new long[] { 1, 1, 1 }, new double[] { 1, 2, 3 }, new[] { false, false, false });
            AssertStep(envs.Step(Actions(1, 2, 3)), new long[] { 2, 2, 2 }, new double[] { 1, 2, 3 }, new[] { true, false, false });
            AssertStep(envs.Step(Actions(1, 2, 3)), new long[] { 0, 3, 3 }, new double[] { 0, 2, 3 }, new[] { false, true, true });
            AssertStep(envs.Step(Actions(1, 2, 3)), new long[] { 1, 0, 0 }, new double[] { 1, 0, 0 }, new[] { false, false, false });

            envs.Close();
        }

        /// <summary>
        ///     <c>SAME_STEP</c>: an ended sub-env resets within the step, returning its reset observation, with the ended
        ///     episode's last observation and info in <c>final_obs</c>/<c>final_info</c> and their masks.
        /// </summary>
        /// <param name="vectoriser">The vectoriser.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_autoreset_mode.py#L98">tests/vector/test_autoreset_mode.py (test_autoreset_within_step)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Vectorisers))]
        public void TestAutoresetWithinStep(string vectoriser) {
            using IVectorEnv envs = Vectorise(vectoriser, Enumerable.Range(0, 3).Select(_ => CountingEnv()), AutoresetMode.SameStep);
            Assert.AreSame(AutoresetMode.SameStep, envs.Metadata["autoreset_mode"]);
            SetAttr(envs, "max_count", new List<object?> { 2L, 3L, 3L });

            (object? obs, Info info) = envs.Reset();
            AssertLongs(obs, 0, 0, 0);
            Assert.AreEqual(0, info.Count);

            AssertStep(envs.Step(Actions(1, 2, 3)), new long[] { 1, 1, 1 }, new double[] { 1, 2, 3 }, new[] { false, false, false });
            AssertStep(envs.Step(Actions(1, 2, 3)), new long[] { 0, 2, 2 }, new double[] { 1, 2, 3 }, new[] { true, false, false }, FinalInfo(new object?[] { 2L, null, null }, true, false, false));
            AssertStep(envs.Step(Actions(1, 2, 3)), new long[] { 1, 0, 0 }, new double[] { 1, 2, 3 }, new[] { false, true, true }, FinalInfo(new object?[] { null, 3L, 3L }, false, true, true));
            AssertStep(envs.Step(Actions(1, 2, 3)), new long[] { 0, 1, 1 }, new double[] { 1, 2, 3 }, new[] { true, false, false }, FinalInfo(new object?[] { 2L, null, null }, true, false, false));

            envs.Close();
        }

        /// <summary><c>DISABLED</c>: nothing resets on its own; a reset with <c>reset_mask</c> resets the ended sub-envs only.</summary>
        /// <param name="vectoriser">The vectoriser.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_autoreset_mode.py#L180">tests/vector/test_autoreset_mode.py (test_autoreset_disabled)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Vectorisers))]
        public void TestAutoresetDisabled(string vectoriser) {
            using IVectorEnv envs = Vectorise(vectoriser, Enumerable.Range(0, 3).Select(_ => CountingEnv()), AutoresetMode.Disabled);
            Assert.AreSame(AutoresetMode.Disabled, envs.Metadata["autoreset_mode"]);
            SetAttr(envs, "max_count", new List<object?> { 2L, 3L, 3L });

            (object? obs, Info info) = envs.Reset();
            AssertLongs(obs, 0, 0, 0);
            Assert.AreEqual(0, info.Count);

            AssertStep(envs.Step(Actions(1, 2, 3)), new long[] { 1, 1, 1 }, new double[] { 1, 2, 3 }, new[] { false, false, false });
            NDArray terminations = AssertStep(envs.Step(Actions(1, 2, 3)), new long[] { 2, 2, 2 }, new double[] { 1, 2, 3 }, new[] { true, false, false });

            (obs, info) = envs.Reset(options: new Info { ["reset_mask"] = terminations });
            AssertLongs(obs, 0, 2, 2);
            Assert.AreEqual(0, info.Count);

            terminations = AssertStep(envs.Step(Actions(1, 2, 3)), new long[] { 1, 3, 3 }, new double[] { 1, 2, 3 }, new[] { false, true, true });

            (obs, info) = envs.Reset(options: new Info { ["reset_mask"] = terminations });
            AssertLongs(obs, 1, 0, 0);
            Assert.AreEqual(0, info.Count);

            AssertStep(envs.Step(Actions(1, 2, 3)), new long[] { 2, 1, 1 }, new double[] { 1, 2, 3 }, new[] { true, false, false });

            envs.Close();
        }

        /// <summary>The autoreset mode lands in the metadata, given as the member or as its value.</summary>
        /// <param name="vectoriser">The vectoriser.</param>
        /// <param name="autoresetMode">The mode's value.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_autoreset_mode.py#L245">tests/vector/test_autoreset_mode.py (test_autoreset_metadata)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(VectorisersAndModes))]
        public void TestAutoresetMetadata(string vectoriser, string autoresetMode) {
            AutoresetMode mode = AutoresetMode.Of(autoresetMode);
            using (IVectorEnv envs = Vectorise(vectoriser, new Func<IEnv>[] { () => new GenericTestEnv(), () => new GenericTestEnv() }, mode)) {
                Assert.AreSame(mode, envs.Metadata["autoreset_mode"]);
                envs.Close();
            }

            using (IVectorEnv envs = Vectorise(vectoriser, new Func<IEnv>[] { () => new GenericTestEnv(), () => new GenericTestEnv() }, mode.Value)) {
                Assert.AreSame(mode, envs.Metadata["autoreset_mode"]);
                envs.Close();
            }
        }

        /// <summary>
        ///     <c>make_vec</c> passes the autoreset mode through <c>vector_kwargs</c>, as the member or its value. (Gymnasium's
        ///     test assigns the metadata instead of comparing it; the port keeps the assignment.)
        /// </summary>
        /// <param name="vectorizationMode"><c>sync</c> or <c>async</c>.</param>
        /// <param name="autoresetMode">The mode's value.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_autoreset_mode.py#L266">tests/vector/test_autoreset_mode.py (test_make_vec_autoreset)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(VectorizationModesAndAutoresetModes))]
        public void TestMakeVecAutoreset(string vectorizationMode, string autoresetMode) {
            AutoresetMode mode = AutoresetMode.Of(autoresetMode);
            using (IVectorEnv envs = MakeVec("CartPole-v1", vectorizationMode: vectorizationMode, vectorKwargs: new Info { ["autoreset_mode"] = mode })) {
                envs.Metadata["autoreset_mode"] = mode;
                envs.Close();
            }

            using (IVectorEnv envs = MakeVec("CartPole-v1", vectorizationMode: vectorizationMode, vectorKwargs: new Info { ["autoreset_mode"] = mode.Value })) {
                envs.Metadata["autoreset_mode"] = mode;
                envs.Close();
            }
        }

        /// <summary>
        ///     <c>SAME_STEP</c>'s <c>final_obs</c> holds members of the single observation space, for every space of Gymnasium's
        ///     testing catalog (a space per sub-env, where Gymnasium's lambdas share one: a Gym.NET env owns its spaces, D19).
        /// </summary>
        /// <param name="index">The space's index in the catalog.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_autoreset_mode.py#L311">tests/vector/test_autoreset_mode.py (test_same_step_final_obs)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(TestingSpaceIndexes))]
        public void TestSameStepFinalObs(int index) {
            // A space per sub-env: an env owns and disposes its spaces (D19), where Gymnasium's lambdas share one.
            Func<IEnv> make = () => new GenericTestEnv(actionSpace: new Discrete(5), observationSpace: TestingSpaces.All[index](), resetFunc: CountResetObs, stepFunc: CountStepObs);
            using var envs = new SyncVectorEnv(Enumerable.Range(0, 3).Select(_ => make), autoresetMode: AutoresetMode.SameStep);
            Assert.AreSame(AutoresetMode.SameStep, envs.Metadata["autoreset_mode"]);
            envs.SetAttr("max_count", new List<object?> { 2L, 3L, 3L });

            envs.Reset();
            envs.Step(Actions(1, 2, 3));
            (_, _, _, _, Info info) = envs.Step(Actions(1, 2, 3));
            Assert.IsTrue(envs.SingleObservationSpace.Contains(((object?[])info["final_obs"]!)[0]), $"{envs.SingleObservationSpace}: final_obs[0]");
            (_, _, _, _, info) = envs.Step(Actions(1, 2, 3));
            Assert.IsTrue(envs.SingleObservationSpace.Contains(((object?[])info["final_obs"]!)[1]), $"{envs.SingleObservationSpace}: final_obs[1]");
            Assert.IsTrue(envs.SingleObservationSpace.Contains(((object?[])info["final_obs"]!)[2]), $"{envs.SingleObservationSpace}: final_obs[2]");
        }

        /// <summary>
        ///     <c>count_reset</c>: seeds the env as <c>Env.reset</c> does, sets <c>count</c> to the seed (0 without one), and
        ///     returns it.
        /// </summary>
        /// <param name="self">The env.</param>
        /// <param name="seed">The seed.</param>
        /// <param name="options">Unused.</param>
        /// <returns>The count and an empty info.</returns>
        private static ResetResult<object?> CountReset(GenericTestEnv self, BigInteger? seed, IReadOnlyDictionary<string, object?>? options) {
            self.BaseReset(seed);
            PyAttributes.SetAttr(self, "count", seed is { } given ? (long)given : 0L);
            return new ResetResult<object?>(PyAttributes.GetAttr(self, "count"), new Info());
        }

        /// <summary><c>count_step</c>: counts, rewards the action, and terminates at <c>max_count</c>.</summary>
        /// <param name="self">The env.</param>
        /// <param name="action">The action, an int.</param>
        /// <returns>The count, the action as the reward, the termination flag, and an empty info.</returns>
        private static StepResult<object?> CountStep(GenericTestEnv self, object? action) {
            long count = Convert.ToInt64(PyAttributes.GetAttr(self, "count")) + 1;
            PyAttributes.SetAttr(self, "count", count);
            bool terminated = count == Convert.ToInt64(PyAttributes.GetAttr(self, "max_count"));
            return new StepResult<object?>(count, PyBuiltins.Float(action), terminated, false, new Info());
        }

        /// <summary><c>count_reset_obs</c>: as <see cref="CountReset"/>, returning a sample of the observation space.</summary>
        /// <param name="self">The env.</param>
        /// <param name="seed">The seed.</param>
        /// <param name="options">Unused.</param>
        /// <returns>A sample and an empty info.</returns>
        private static ResetResult<object?> CountResetObs(GenericTestEnv self, BigInteger? seed, IReadOnlyDictionary<string, object?>? options) {
            self.BaseReset(seed);
            PyAttributes.SetAttr(self, "count", seed is { } given ? (long)given : 0L);
            return new ResetResult<object?>(self.ObservationSpace.Sample(), new Info());
        }

        /// <summary><c>count_step_obs</c>: as <see cref="CountStep"/>, returning a sample of the observation space.</summary>
        /// <param name="self">The env.</param>
        /// <param name="action">The action.</param>
        /// <returns>A sample, the action as the reward, the termination flag, and an empty info.</returns>
        private static StepResult<object?> CountStepObs(GenericTestEnv self, object? action) {
            long count = Convert.ToInt64(PyAttributes.GetAttr(self, "count")) + 1;
            PyAttributes.SetAttr(self, "count", count);
            bool terminated = count == Convert.ToInt64(PyAttributes.GetAttr(self, "max_count"));
            return new StepResult<object?>(self.ObservationSpace.Sample(), PyBuiltins.Float(action), terminated, false, new Info());
        }

        /// <summary>The counting env's function: <c>Discrete(5)</c> actions and observations, <see cref="CountReset"/> and <see cref="CountStep"/>.</summary>
        /// <returns>The function.</returns>
        private static Func<IEnv> CountingEnv() =>
            () => new GenericTestEnv(actionSpace: new Discrete(5), observationSpace: new Discrete(5), resetFunc: CountReset, stepFunc: CountStep);

        /// <summary>Builds the vectoriser of the id over the functions with the autoreset mode.</summary>
        /// <param name="id">Gymnasium's vectoriser id.</param>
        /// <param name="envFns">The sub-envs' functions.</param>
        /// <param name="autoresetMode">The mode (a member or its value).</param>
        /// <returns>The vector env; the caller disposes it.</returns>
        private static IVectorEnv Vectorise(string id, IEnumerable<Func<IEnv>> envFns, object autoresetMode) => id switch {
            "Sync" => new SyncVectorEnv(envFns, autoresetMode: autoresetMode),
            "Async(shared_memory=True)" => new AsyncVectorEnv(envFns, sharedMemory: true, autoresetMode: autoresetMode),
            _ => new AsyncVectorEnv(envFns, sharedMemory: false, autoresetMode: autoresetMode),
        };

        /// <summary><c>envs.set_attr(name, values)</c> on either vector env.</summary>
        /// <param name="envs">The vector env.</param>
        /// <param name="name">The attribute.</param>
        /// <param name="values">One value per sub-env.</param>
        private static void SetAttr(IVectorEnv envs, string name, object? values) {
            if (envs is SyncVectorEnv sync)
                sync.SetAttr(name, values);
            else
                ((AsyncVectorEnv)envs).SetAttr(name, values);
        }

        /// <summary>A list of Python int actions.</summary>
        /// <param name="actions">The actions.</param>
        /// <returns>The list.</returns>
        private static List<object?> Actions(params long[] actions) => actions.Select(action => (object?)action).ToList();

        /// <summary>The <c>SAME_STEP</c> info of Gymnasium's expectation: <c>final_obs</c>, an empty <c>final_info</c>, and both masks.</summary>
        /// <param name="finalObs">The final observations, <see langword="null"/> where a sub-env did not end.</param>
        /// <param name="mask">Which sub-envs ended.</param>
        /// <returns>The info.</returns>
        private static Info FinalInfo(object?[] finalObs, params bool[] mask) => new() {
            ["final_obs"] = finalObs, ["final_info"] = new Info(), ["_final_obs"] = np.array(mask), ["_final_info"] = np.array(mask),
        };

        /// <summary>Asserts a batch of ints.</summary>
        /// <param name="batch">The batch, an array.</param>
        /// <param name="expected">The values.</param>
        private static void AssertLongs(object? batch, params long[] expected) {
            var array = (NDArray)batch!;
            CollectionAssert.AreEqual(expected, array.ToArray<long>(), $"observations {array}");
        }

        /// <summary>Asserts a step's observations, rewards and flags (never truncated here), and its infos: empty, or the given ones.</summary>
        /// <param name="step">The step.</param>
        /// <param name="obs">The observations.</param>
        /// <param name="rewards">The rewards.</param>
        /// <param name="terminations">The terminations.</param>
        /// <param name="info">The expected infos, or <see langword="null"/> for none.</param>
        /// <returns>The terminations, for a <c>reset_mask</c>.</returns>
        private static NDArray AssertStep(VectorStepResult<object?> step, long[] obs, double[] rewards, bool[] terminations, Info? info = null) {
            AssertLongs(step.Observations, obs);
            CollectionAssert.AreEqual(rewards, step.Rewards.ToArray<double>(), $"rewards {step.Rewards}");
            CollectionAssert.AreEqual(terminations, step.Terminations.ToArray<bool>(), $"terminations {step.Terminations}");
            CollectionAssert.AreEqual(new bool[terminations.Length], step.Truncations.ToArray<bool>(), $"truncations {step.Truncations}");
            if (info is null)
                Assert.AreEqual(0, step.Infos.Count, $"infos {step.Infos}");
            else
                Assert.IsTrue(EnvChecker.DataEquivalence(step.Infos, info), $"infos {step.Infos} vs {info}");
            return step.Terminations;
        }
    }
}
