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
    /// <summary>Gymnasium's <c>FlattenObservation</c> tests (<c>test_flatten_observation.py</c>), run against <see cref="FlattenObservation{TObs, TAct}"/>.</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_flatten_observation.py">tests/wrappers/test_flatten_observation.py</see>.</remarks>
    [TestClass]
    public class TestFlattenObservation {
        /// <summary>The observations are flattened into the flattened space, and the action space is the env's.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_flatten_observation.py#L14">tests/wrappers/test_flatten_observation.py (test_flatten_observation_wrapper)</see>.</remarks>
        [TestMethod]
        public void TestFlattenObservationWrapper() {
            var env = new GenericTestEnv(
                observationSpace: new Dict(new Dictionary<string, Space> { ["arm"] = new Box(0, 1), ["head"] = new Box(2, 3) }),
                resetFunc: WrapperUtils.RecordRandomObsReset, stepFunc: WrapperUtils.RecordRandomObsStep);
            using var wrappedEnv = new FlattenObservation<object?, object?>(env);
            using (Space flattened = SpaceUtils.FlattenSpace(env.ObservationSpace))
                Assert.IsTrue(wrappedEnv.ObservationSpace.Equals(flattened));
            Assert.IsTrue(wrappedEnv.ActionSpace.Equals(env.ActionSpace));

            var (obs, info) = wrappedEnv.Reset();
            WrapperUtils.CheckObs(env, wrappedEnv, obs, info["obs"]);
            (obs, _, _, _, info) = wrappedEnv.Step(null);
            WrapperUtils.CheckObs(env, wrappedEnv, obs, info["obs"]);
        }

        /// <summary>The flattened observations of a <c>OneOf</c> space lie in the wrapper's space.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_flatten_observation.py#L33">tests/wrappers/test_flatten_observation.py (test_flatten_observation_wrapper_oneof)</see>.</remarks>
        [TestMethod]
        public void TestFlattenObservationWrapperOneof() {
            var env = new GenericTestEnv(
                observationSpace: new OneOf(new Box(-1, 1, new Shape(2)), new Box(-1, 1, new Shape(3))),
                resetFunc: WrapperUtils.RecordRandomObsReset, stepFunc: WrapperUtils.RecordRandomObsStep);
            env.ObservationSpace.Seed(WrapperUtils.Seed);
            using var wrappedEnv = new FlattenObservation<object?, object?>(env);
            using (Space flattened = SpaceUtils.FlattenSpace(env.ObservationSpace))
                Assert.IsTrue(wrappedEnv.ObservationSpace.Equals(flattened));

            var (obs, info) = wrappedEnv.Reset();
            WrapperUtils.CheckObs(env, wrappedEnv, obs, info["obs"]);
            for (int i = 0; i < 10; i++) {
                (obs, _, _, _, info) = wrappedEnv.Step(null);
                WrapperUtils.CheckObs(env, wrappedEnv, obs, info["obs"]);
            }
        }
    }
}
