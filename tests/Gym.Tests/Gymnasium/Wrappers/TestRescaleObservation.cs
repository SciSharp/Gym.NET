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
    /// <summary>Gymnasium's <c>RescaleObservation</c> tests (<c>test_rescale_observation.py</c>), run against <see cref="RescaleObservation{TObs, TAct}"/>.</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_rescale_observation.py">tests/wrappers/test_rescale_observation.py</see>.</remarks>
    [TestClass]
    public class TestRescaleObservation {
        /// <summary>Observations are rescaled affinely, infinite components shifted only, onto the new bounds.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_rescale_observation.py#L11">tests/wrappers/test_rescale_observation.py (test_rescale_observation)</see>.</remarks>
        [TestMethod]
        public void TestRescaleObservationWrapper() {
            float inf = float.PositiveInfinity;
            var env = new GenericTestEnv(
                observationSpace: new Box(np.array(new[] { 0f, 1f, -inf, 5f, -inf }), np.array(new[] { 1f, 3f, inf, inf, 7f })),
                resetFunc: WrapperUtils.RecordObsReset, stepFunc: WrapperUtils.RecordActionAsObsStep);
            using var wrappedEnv = new RescaleObservation<object?, object?>(env,
                np.array(new[] { -5f, 0f, -inf, -1f, -inf }), np.array(new[] { 5f, 1f, inf, inf, 4f }));
            using (var expectedSpace = new Box(np.array(new[] { -5f, 0f, -inf, -1f, -inf }), np.array(new[] { 5f, 1f, inf, inf, 4f })))
                Assert.IsTrue(wrappedEnv.ObservationSpace.Equals(expectedSpace));

            var cases = new[] {
                (new[] { 0.5f, 2.0f, 7.0f, 5.0f, -20.0f }, new[] { 0.0f, 0.5f, 7.0f, -1.0f, -23.0f }),
                (new[] { 0.0f, 1.0f, -4.0f, 6.0f, 0.0f }, new[] { -5.0f, 0.0f, -4.0f, 0.0f, -3.0f }),
                (new[] { 1.0f, 3.0f, 0.0f, 7.0f, 7.0f }, new[] { 5.0f, 1.0f, 0.0f, 1.0f, 4.0f }),
            };
            foreach ((float[] sample, float[] expected) in cases) {
                using NDArray sampleObs = np.array(sample);
                using NDArray expectedObs = np.array(expected);
                Assert.IsTrue(env.ObservationSpace.Contains(sampleObs));
                Assert.IsTrue(wrappedEnv.ObservationSpace.Contains(expectedObs));

                var (obs, info) = wrappedEnv.Reset(options: new Dictionary<string, object?> { ["obs"] = sampleObs });
                Assert.IsTrue(np.array_equal(obs, expectedObs));
                WrapperUtils.CheckObs(env, wrappedEnv, obs, info["obs"], strict: false);

                (obs, _, _, _, info) = wrappedEnv.Step(sampleObs);
                Assert.IsTrue(np.array_equal(obs, expectedObs));
                WrapperUtils.CheckObs(env, wrappedEnv, obs, info["obs"], strict: false);
            }
        }
    }
}
