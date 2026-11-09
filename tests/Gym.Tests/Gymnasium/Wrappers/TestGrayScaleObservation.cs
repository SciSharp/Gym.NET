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
    /// <summary>Gymnasium's <c>GrayscaleObservation</c> tests (<c>test_gray_scale_observation.py</c>), run against <see cref="GrayscaleObservation{TObs, TAct}"/>.</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_gray_scale_observation.py">tests/wrappers/test_gray_scale_observation.py</see>.</remarks>
    [TestClass]
    public class TestGrayScaleObservation {
        /// <summary>The observation becomes a grayscale image, with or without the channel axis.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_gray_scale_observation.py#L15">tests/wrappers/test_gray_scale_observation.py (test_grayscale_observation_wrapper)</see>.</remarks>
        [TestMethod]
        public void TestGrayscaleObservationWrapper() {
            var env = new GenericTestEnv(
                observationSpace: new Box(0, 255, new Shape(25, 25, 3), np.uint8),
                resetFunc: WrapperUtils.RecordRandomObsReset, stepFunc: WrapperUtils.RecordRandomObsStep);
            var wrappedEnv = new GrayscaleObservation<object?, object?>(env);
            var (obs, info) = wrappedEnv.Reset();
            WrapperUtils.CheckObs(env, wrappedEnv, obs, info["obs"]);
            Assert.AreEqual(new Shape(25, 25), obs.Shape);
            (obs, _, _, _, info) = wrappedEnv.Step(null);
            WrapperUtils.CheckObs(env, wrappedEnv, obs, info["obs"]);

            // Keep_dim.
            var keptEnv = new GrayscaleObservation<object?, object?>(env, keepDim: true);
            (obs, info) = keptEnv.Reset();
            WrapperUtils.CheckObs(env, keptEnv, obs, info["obs"]);
            Assert.AreEqual(new Shape(25, 25, 1), obs.Shape);
            (obs, _, _, _, info) = keptEnv.Step(null);
            WrapperUtils.CheckObs(env, keptEnv, obs, info["obs"]);
            keptEnv.Dispose();
        }
    }
}
