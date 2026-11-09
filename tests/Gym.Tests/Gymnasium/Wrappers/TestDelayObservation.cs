#nullable enable
using System.Collections.Generic;
using System.Linq;
using Gym.Envs;
using Gym.Spaces;
using Gym.Tests.Gymnasium.Spaces;
using Gym.Utils;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Wrappers {
    /// <summary>Gymnasium's <c>DelayObservation</c> tests (<c>test_delay_observation.py</c>), run against <see cref="DelayObservation{TObs, TAct}"/>.</summary>
    /// <remarks>
    ///     <c>test_delay_failures</c>' <c>TypeError</c> for a float delay has no counterpart: the delay is a <see cref="long"/>,
    ///     so the compiler refuses a float. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_delay_observation.py">tests/wrappers/test_delay_observation.py</see>.
    /// </remarks>
    [TestClass]
    public class TestDelayObservation {
        /// <summary>Every space of Gymnasium's testing catalog, by position (<c>TESTING_OBS_ENVS</c>).</summary>
        public static IEnumerable<object[]> Spaces => TestingSpaces.AllIndexes;

        /// <summary>
        ///     Over an env of every testing space: the space's zero until the delay has passed, then the observations the
        ///     undelayed env returned, each a member of the space.
        /// </summary>
        /// <param name="index">The space's position in <see cref="TestingSpaces.All"/>.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_delay_observation.py#L15">tests/wrappers/test_delay_observation.py (test_env_obs)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Spaces))]
        public void TestEnvObs(int index) {
            const int delay = 3, extraSteps = 4;
            var env = new GenericTestEnv(observationSpace: TestingSpaces.All[index]());
            env.ActionSpace.Seed(WrapperUtils.Seed);
            var (obs, _) = env.Reset(seed: WrapperUtils.Seed);

            var undelayedObs = new List<object?> { obs };
            for (int i = 0; i < delay + extraSteps; i++) {
                (obs, _, _, _, _) = env.Step(env.ActionSpace.Sample());
                undelayedObs.Add(obs);
            }

            using var wrapped = new DelayObservation<object?, object?>(env, delay);
            object? exampleZeroObs = WrapperFunctions.CreateZeroArray(wrapped.ObservationSpace);
            wrapped.ActionSpace.Seed(WrapperUtils.Seed);
            (obs, _) = wrapped.Reset(seed: WrapperUtils.Seed);
            Assert.IsTrue(EnvChecker.DataEquivalence(obs, exampleZeroObs), $"{PyBuiltins.Repr(obs)} is not the zero {PyBuiltins.Repr(exampleZeroObs)}");
            Assert.IsTrue(wrapped.ObservationSpace.Contains(obs));

            var delayedObs = new List<object?> { obs };
            for (int i = 0; i < delay + extraSteps; i++) {
                (obs, _, _, _, _) = wrapped.Step(wrapped.ActionSpace.Sample());
                delayedObs.Add(obs);
                Assert.IsTrue(wrapped.ObservationSpace.Contains(obs));

                if (i < delay - 1)
                    Assert.IsTrue(EnvChecker.DataEquivalence(obs, exampleZeroObs), $"step {i}: {PyBuiltins.Repr(obs)} is not the zero");
            }

            // data_equivalence(delayed_obs[delay:], undelayed_obs[:-delay])
            Assert.IsTrue(EnvChecker.DataEquivalence(delayedObs.Skip(delay).ToList(), undelayedObs.Take(undelayedObs.Count - delay).ToList()));
        }

        /// <summary>CartPole delayed by 1 to 4 steps: zeros, then the reset's observation.</summary>
        /// <param name="delay">The delay.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_delay_observation.py#L45">tests/wrappers/test_delay_observation.py (test_delay_values)</see>.</remarks>
        [TestMethod]
        [DataRow(1L)]
        [DataRow(2L)]
        [DataRow(3L)]
        [DataRow(4L)]
        public void TestDelayValues(long delay) {
            using var plain = (Env<NDArray, NDArray>)Make("CartPole-v1");
            using NDArray firstObs = plain.Reset(seed: 123).Observation;

            using var env = new DelayObservation<NDArray, NDArray>((Env<NDArray, NDArray>)Make("CartPole-v1"), delay);
            using var zeroObs = (NDArray)WrapperFunctions.CreateZeroArray(env.ObservationSpace)!;
            NDArray obs = env.Reset(seed: 123).Observation;
            Assert.IsTrue(EnvChecker.DataEquivalence(obs, zeroObs));
            for (long i = 0; i < delay - 1; i++) {
                (obs, _, _, _, _) = env.Step(env.ActionSpace.Sample());
                Assert.IsTrue(EnvChecker.DataEquivalence(obs, zeroObs));
            }

            (obs, _, _, _, _) = env.Step(env.ActionSpace.Sample());
            Assert.IsTrue(EnvChecker.DataEquivalence(firstObs, obs));
        }

        /// <summary>A negative delay is refused with Gymnasium's message.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_delay_observation.py#L62">tests/wrappers/test_delay_observation.py (test_delay_failures)</see>.</remarks>
        [TestMethod]
        public void TestDelayFailures() {
            using var env = (Env<NDArray, NDArray>)Make("CartPole-v1");
            var error = Assert.ThrowsExactly<ValueError>(() => new DelayObservation<NDArray, NDArray>(env, delay: -1));
            StringAssert.Contains(error.Message, "The delay needs to be greater than zero, actual value: -1");
        }
    }
}
