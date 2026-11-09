#nullable enable
using System.Collections.Generic;
using System.Linq;
using Gym.Envs;
using Gym.Spaces;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Tuple = Gym.Spaces.Tuple;

namespace Gym.Tests.Gymnasium.Wrappers {
    /// <summary>Gymnasium's <c>FilterObservation</c> tests (<c>test_filter_observation.py</c>), run against <see cref="FilterObservation{TObs, TAct}"/>.</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_filter_observation.py">tests/wrappers/test_filter_observation.py</see>.</remarks>
    [TestClass]
    public class TestFilterObservation {
        /// <summary>The right keys of a <c>Dict</c>, and the right items of a <c>Tuple</c>, are kept.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_filter_observation.py#L15">tests/wrappers/test_filter_observation.py (test_filter_observation_wrapper)</see>.</remarks>
        [TestMethod]
        public void TestFilterObservationWrapper() {
            var dictEnv = new GenericTestEnv(
                observationSpace: new Dict(new Dictionary<string, Space> { ["arm_1"] = new Box(0, 1), ["arm_2"] = new Box(2, 3), ["arm_3"] = new Box(-1, 1) }),
                resetFunc: WrapperUtils.RecordRandomObsReset, stepFunc: WrapperUtils.RecordRandomObsStep);
            using (var wrappedEnv = new FilterObservation<object?, object?>(dictEnv, ("arm_1", "arm_3"))) {
                var (obs, info) = wrappedEnv.Reset();
                CollectionAssert.AreEqual(new[] { "arm_1", "arm_3" }, ((Info)obs!).Keys.ToArray());
                CollectionAssert.AreEqual(new[] { "arm_1", "arm_2", "arm_3" }, ((Info)info["obs"]!).Keys.ToArray());
                WrapperUtils.CheckObs(dictEnv, wrappedEnv, obs, info["obs"]);

                (obs, _, _, _, info) = wrappedEnv.Step(null);
                CollectionAssert.AreEqual(new[] { "arm_1", "arm_3" }, ((Info)obs!).Keys.ToArray());
                CollectionAssert.AreEqual(new[] { "arm_1", "arm_2", "arm_3" }, ((Info)info["obs"]!).Keys.ToArray());
                WrapperUtils.CheckObs(dictEnv, wrappedEnv, obs, info["obs"]);
            }

            // Test tuple environments.
            var tupleEnv = new GenericTestEnv(
                observationSpace: new Tuple(new Box(0, 1), new Box(2, 3), new Box(-1, 1)),
                resetFunc: WrapperUtils.RecordRandomObsReset, stepFunc: WrapperUtils.RecordRandomObsStep);
            using (var wrappedEnv = new FilterObservation<object?, object?>(tupleEnv, System.ValueTuple.Create(2L))) {
                var (obs, info) = wrappedEnv.Reset();
                Assert.IsTrue(((object?[])obs!).Length == 1 && ((object?[])info["obs"]!).Length == 3);
                WrapperUtils.CheckObs(tupleEnv, wrappedEnv, obs, info["obs"]);

                (obs, _, _, _, info) = wrappedEnv.Step(null);
                Assert.IsTrue(((object?[])obs!).Length == 1 && ((object?[])info["obs"]!).Length == 3);
                WrapperUtils.CheckObs(tupleEnv, wrappedEnv, obs, info["obs"]);
            }
        }

        /// <summary>Arguments the wrapper refuses, each with its error type and message.</summary>
        /// <param name="filterCase">The case: <c>key</c> (a string of missing keys), <c>False</c>, <c>1</c> or <c>()</c>.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_filter_observation.py#L76">tests/wrappers/test_filter_observation.py (test_incorrect_arguments)</see>.</remarks>
        [TestMethod]
        [DataRow("key")]
        [DataRow("False")]
        [DataRow("1")]
        [DataRow("()")]
        public void TestIncorrectArguments(string filterCase) {
            var env = new GenericTestEnv(observationSpace: new Dict(new Dictionary<string, Space> { ["key_1"] = new Box(0, 1), ["key_2"] = new Box(2, 3) }));
            switch (filterCase) {
                case "key":
                    StringAssert.Contains(Assert.ThrowsExactly<ValueError>(() => new FilterObservation<object?, object?>(env, "key")).Message,
                        "All the `filter_keys` must be included in the observation space.");
                    break;
                case "False":
                    StringAssert.Contains(Assert.ThrowsExactly<TypeError>(() => new FilterObservation<object?, object?>(env, false)).Message,
                        "Expects `filter_keys` to be a Sequence, actual type: <class 'bool'>");
                    break;
                case "1":
                    StringAssert.Contains(Assert.ThrowsExactly<TypeError>(() => new FilterObservation<object?, object?>(env, 1L)).Message,
                        "Expects `filter_keys` to be a Sequence, actual type: <class 'int'>");
                    break;
                default:
                    StringAssert.Contains(Assert.ThrowsExactly<ValueError>(() => new FilterObservation<object?, object?>(env, System.Array.Empty<object?>())).Message,
                        "The observation space is empty due to filtering all of the keys");
                    break;
            }
        }
    }
}
