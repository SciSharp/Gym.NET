#nullable enable
using System.Collections.Generic;
using System.Numerics;
using Gym.Envs;
using Gym.Observations;
using Gym.Spaces;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Utils {
    /// <summary>
    ///     Gymnasium's <c>check_environments_match</c> tests (<c>tests/utils/test_env_match.py</c>): identical envs pass,
    ///     different observation or action spaces fail, and <c>skip_obs</c> skips the observation space.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_env_match.py">tests/utils/test_env_match.py</see>.</remarks>
    [TestClass]
    public class TestEnvMatch {
        /// <summary>Two identical envs pass.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_env_match.py#L31">test_matching_environments</see>.</remarks>
        [TestMethod]
        public void TestMatchingEnvironments() {
            using GenericTestEnv envA = MakeEnv(BoxOf(1)), envB = MakeEnv(BoxOf(1));
            EnvMatch.CheckEnvironmentsMatch(envA, envB, numSteps: 5);
        }

        /// <summary>Different observation spaces fail, even when the observations are equal.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_env_match.py#L38">test_mismatched_observation_space</see>.</remarks>
        [TestMethod]
        public void TestMismatchedObservationSpace() {
            using GenericTestEnv envA = MakeEnv(BoxOf(1)), envB = MakeEnv(BoxOf(2));
            Assert.ThrowsExactly<AssertionError>(() => EnvMatch.CheckEnvironmentsMatch(envA, envB, numSteps: 0));
        }

        /// <summary><c>skip_obs</c> skips the observation space check.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_env_match.py#L46">test_mismatched_observation_space_skip_obs</see>.</remarks>
        [TestMethod]
        public void TestMismatchedObservationSpaceSkipObs() {
            using GenericTestEnv envA = MakeEnv(BoxOf(1)), envB = MakeEnv(BoxOf(2));
            EnvMatch.CheckEnvironmentsMatch(envA, envB, numSteps: 0, skipObs: true);
        }

        /// <summary>Different action spaces fail.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_env_match.py#L53">test_mismatched_action_space</see>.</remarks>
        [TestMethod]
        public void TestMismatchedActionSpace() {
            using var envA = new GenericTestEnv(actionSpace: new Discrete(2), resetFunc: FixedObsReset);
            using var envB = new GenericTestEnv(actionSpace: new Discrete(3), resetFunc: FixedObsReset);
            Assert.ThrowsExactly<AssertionError>(() => EnvMatch.CheckEnvironmentsMatch(envA, envB, numSteps: 0));
        }

        /// <summary>Gymnasium's <c>_make_env</c>: a <c>Discrete(2)</c>-action env with a fixed observation and the given observation space.</summary>
        /// <param name="observationSpace">The observation space.</param>
        /// <returns>The env.</returns>
        private static GenericTestEnv MakeEnv(Space observationSpace) =>
            new(actionSpace: new Discrete(2), observationSpace: observationSpace, resetFunc: FixedObsReset, stepFunc: FixedObsStep);

        /// <summary>A float32 box of 2 dimensions in <c>[0, high]</c>.</summary>
        /// <param name="high">The upper bound.</param>
        /// <returns>The box.</returns>
        private static Box BoxOf(double high) => new(0, high, new Shape(2L), dtype: np.float32);

        /// <summary>Gymnasium's <c>_fixed_obs_reset_func</c>: seeds as the base reset does, returns <c>[0.5, 0.5]</c>.</summary>
        /// <param name="self">The env.</param>
        /// <param name="seed">The seed.</param>
        /// <param name="options">Ignored.</param>
        /// <returns>The observation and an empty info.</returns>
        private static ResetResult<object?> FixedObsReset(GenericTestEnv self, BigInteger? seed, IReadOnlyDictionary<string, object?>? options) {
            self.BaseReset(seed);
            return new ResetResult<object?>(np.array(new[] { 0.5f, 0.5f }), new Info());
        }

        /// <summary>Gymnasium's <c>_fixed_obs_step_func</c>: <c>[0.5, 0.5]</c>, reward 0, no signals.</summary>
        /// <param name="self">The env.</param>
        /// <param name="action">Ignored.</param>
        /// <returns>The transition.</returns>
        private static StepResult<object?> FixedObsStep(GenericTestEnv self, object? action) => new(np.array(new[] { 0.5f, 0.5f }), 0.0, false, false, new Info());
    }
}
