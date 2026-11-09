#nullable enable
using Gym.Envs;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Wrappers {
    /// <summary>Gymnasium's <c>MaxAndSkipObservation</c> tests (<c>test_max_and_skip_observation.py</c>), run against <see cref="MaxAndSkipObservation{TObs, TAct}"/>.</summary>
    /// <remarks>
    ///     <c>test_skip_size_failures</c>' <c>TypeError</c> for a float skip has no counterpart: the skip is a <see cref="long"/>,
    ///     so the compiler refuses a float. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_max_and_skip_observation.py">tests/wrappers/test_max_and_skip_observation.py</see>.
    /// </remarks>
    [TestClass]
    public class TestMaxAndSkipObservation {
        /// <summary>Every observation, the reset's and each skipped step's maximum, is a member of CartPole's space, across episode ends.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_max_and_skip_observation.py#L11">tests/wrappers/test_max_and_skip_observation.py (test_max_and_skip_obs)</see>.</remarks>
        [TestMethod]
        public void TestMaxAndSkipObs() {
            const long skip = 4;
            using var env = new MaxAndSkipObservation<NDArray, NDArray>((Env<NDArray, NDArray>)Make("CartPole-v1"), skip: skip);

            NDArray obs = env.Reset().Observation;
            Assert.IsTrue(env.ObservationSpace.Contains(obs));

            for (int i = 0; i < 10; i++) {
                (obs, _, bool term, bool trunc, _) = env.Step(env.ActionSpace.Sample());
                Assert.IsTrue(env.ObservationSpace.Contains(obs));

                if (term || trunc) {
                    obs = env.Reset().Observation;
                    Assert.IsTrue(env.ObservationSpace.Contains(obs));
                }
            }
        }

        /// <summary>A skip below 2 is refused with Gymnasium's message.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_max_and_skip_observation.py#L29">tests/wrappers/test_max_and_skip_observation.py (test_skip_size_failures)</see>.</remarks>
        [TestMethod]
        public void TestSkipSizeFailures() {
            using var env = (Env<NDArray, NDArray>)Make("CartPole-v1");
            var error = Assert.ThrowsExactly<ValueError>(() => new MaxAndSkipObservation<NDArray, NDArray>(env, skip: 0));
            StringAssert.Contains(error.Message, "The skip value needs to be equal or greater than two, actual value: 0");
        }
    }
}
