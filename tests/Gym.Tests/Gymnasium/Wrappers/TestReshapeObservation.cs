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
    /// <summary>Gymnasium's <c>ReshapeObservation</c> tests (<c>test_reshape_observation.py</c>), run against <see cref="ReshapeObservation{TObs, TAct}"/>.</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_reshape_observation.py">tests/wrappers/test_reshape_observation.py</see>.</remarks>
    [TestClass]
    public class TestReshapeObservation {
        /// <summary>The observations take the new shape.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_reshape_observation.py#L13">tests/wrappers/test_reshape_observation.py (test_reshape_observation_wrapper)</see>.</remarks>
        [TestMethod]
        public void TestReshapeObservationWrapper() {
            var env = new GenericTestEnv(
                observationSpace: new Box(0, 1, new Shape(2, 3, 2)),
                resetFunc: WrapperUtils.RecordRandomObsReset, stepFunc: WrapperUtils.RecordRandomObsStep);
            using var wrappedEnv = new ReshapeObservation<object?, object?>(env, new[] { 6L, 2L });
            var (obs, info) = wrappedEnv.Reset();
            WrapperUtils.CheckObs(env, wrappedEnv, obs, info["obs"]);
            Assert.AreEqual(new Shape(6, 2), obs.Shape);
            (obs, _, _, _, info) = wrappedEnv.Step(null);
            WrapperUtils.CheckObs(env, wrappedEnv, obs, info["obs"]);
            Assert.AreEqual(new Shape(6, 2), obs.Shape);
        }
    }
}
