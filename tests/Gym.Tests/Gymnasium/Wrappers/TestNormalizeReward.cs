#nullable enable
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Observations;
using Gym.Utils;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Wrappers {
    /// <summary>Gymnasium's <c>NormalizeReward</c> tests (<c>test_normalize_reward.py</c>), run against <see cref="NormalizeReward{TObs, TAct}"/>.</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_normalize_reward.py">tests/wrappers/test_normalize_reward.py</see>.</remarks>
    [TestClass]
    public class TestNormalizeReward {
        /// <summary>A step that samples the observation space and always rewards 1.0 (<c>constant_reward_step_func</c>).</summary>
        /// <param name="self">The env.</param>
        /// <param name="action">Ignored.</param>
        /// <returns><c>(sample, 1.0, False, False, {})</c>.</returns>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_normalize_reward.py#L13">tests/wrappers/test_normalize_reward.py (constant_reward_step_func)</see>.</remarks>
        private static StepResult<object?> ConstantRewardStepFunc(GenericTestEnv self, object? action) =>
            new(self.ObservationSpace.Sample(), 1.0, false, false, new Info());

        /// <summary>A reset that starts the reward list <c>[0, 1, 2, 3, 4]</c> and returns its first item (<c>reward_reset_func</c>).</summary>
        /// <param name="self">The env.</param>
        /// <param name="seed">Ignored.</param>
        /// <param name="options">Ignored.</param>
        /// <returns><c>(np.array([0]), {"reward": 0})</c>.</returns>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_normalize_reward.py#L57">tests/wrappers/test_normalize_reward.py (reward_reset_func)</see>.</remarks>
        private static ResetResult<object?> RewardResetFunc(GenericTestEnv self, BigInteger? seed, IReadOnlyDictionary<string, object?>? options) {
            var rewards = new List<long> { 0, 1, 2, 3, 4 };
            PyAttributes.SetAttr(self, "rewards", rewards);
            long reward = Pop(rewards);
            return new ResetResult<object?>(np.array(new[] { reward }), new Info { ["reward"] = reward });
        }

        /// <summary>A step that returns the next reward of the list, terminating when the list is empty (<c>reward_step_func</c>).</summary>
        /// <param name="self">The env.</param>
        /// <param name="action">Ignored.</param>
        /// <returns><c>(np.array([reward]), reward, len(rewards) == 0, False, {"reward": reward})</c>.</returns>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_normalize_reward.py#L63">tests/wrappers/test_normalize_reward.py (reward_step_func)</see>.</remarks>
        private static StepResult<object?> RewardStepFunc(GenericTestEnv self, object? action) {
            var rewards = (List<long>)PyAttributes.GetAttr(self, "rewards")!;
            long reward = Pop(rewards);
            return new StepResult<object?>(np.array(new[] { reward }), reward, rewards.Count == 0, false, new Info { ["reward"] = reward });
        }

        /// <summary><c>update_running_mean</c> freezes and resumes the return statistics.</summary>
        /// <remarks>
        ///     The statistics' arrays are replaced (and the old ones released) by each update, so the test keeps copies where
        ///     Python keeps the replaced arrays alive by reference.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_normalize_reward.py#L17">tests/wrappers/test_normalize_reward.py (test_running_mean_normalize_reward_wrapper)</see>.
        /// </remarks>
        [TestMethod]
        public void TestRunningMeanNormalizeRewardWrapper() {
            var env = new GenericTestEnv(stepFunc: ConstantRewardStepFunc);
            using var wrappedEnv = new NormalizeReward<object?, object?>(env);

            // Default value is True
            Assert.IsTrue(wrappedEnv.UpdateRunningMean);

            wrappedEnv.Reset();
            using NDArray rmsVarInit = wrappedEnv.ReturnRms.Var.copy();
            using NDArray rmsMeanInit = wrappedEnv.ReturnRms.Mean.copy();

            // Statistics are updated when env.step()
            wrappedEnv.Step(null);
            using NDArray rmsVarUpdated = wrappedEnv.ReturnRms.Var.copy();
            using NDArray rmsMeanUpdated = wrappedEnv.ReturnRms.Mean.copy();
            Assert.IsFalse(np.array_equal(rmsVarInit, rmsVarUpdated));
            Assert.IsFalse(np.array_equal(rmsMeanInit, rmsMeanUpdated));

            // Assure property is set
            wrappedEnv.UpdateRunningMean = false;
            Assert.IsFalse(wrappedEnv.UpdateRunningMean);

            // Statistics are frozen
            wrappedEnv.Step(null);
            Assert.IsTrue(np.array_equal(rmsVarUpdated, wrappedEnv.ReturnRms.Var));
            Assert.IsTrue(np.array_equal(rmsMeanUpdated, wrappedEnv.ReturnRms.Mean));
        }

        /// <summary>A step through the wrapper raises nothing and returns a scalar reward.</summary>
        /// <remarks>
        ///     <c>np.ndim(reward) == 0</c> holds by type here: every reward is a <see cref="double"/>.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_normalize_reward.py#L46">tests/wrappers/test_normalize_reward.py (test_normalize_reward_wrapper)</see>.
        /// </remarks>
        [TestMethod]
        public void TestNormalizeRewardWrapper() {
            var env = new GenericTestEnv(stepFunc: ConstantRewardStepFunc);
            using var wrappedEnv = new NormalizeReward<object?, object?>(env);
            wrappedEnv.Reset();
            double reward = wrappedEnv.Step(null).Reward;
            Assert.IsFalse(double.IsNaN(reward));
            env.Close();
        }

        /// <summary>The running mean of the discounted returns: <c>[1]</c> after one step, <c>[2 + γ, 1]</c> after two.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_normalize_reward.py#L68">tests/wrappers/test_normalize_reward.py (test_normalize_return)</see>.</remarks>
        [TestMethod]
        public void TestNormalizeReturn() {
            var genericEnv = new GenericTestEnv(resetFunc: RewardResetFunc, stepFunc: RewardStepFunc);
            using var env = new NormalizeReward<object?, object?>(genericEnv);
            env.Reset();

            env.Step(env.ActionSpace.Sample());
            AssertAlmostEqual(new[] { 1.0 }.Average(), (double)env.ReturnRms.Mean); // [first return]

            env.Step(env.ActionSpace.Sample());
            AssertAlmostEqual(new[] { 2 + 1 * env.Gamma, 1 }.Average(), (double)env.ReturnRms.Mean); // [second return, first return]
        }

        /// <summary>A <c>gamma</c> outside [0, 1] is refused: the accumulator would diverge, and every normalized reward become NaN.</summary>
        /// <param name="gamma">The discount.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_normalize_reward.py#L89">tests/wrappers/test_normalize_reward.py (test_gamma_outside_unit_interval_is_rejected)</see>.</remarks>
        [TestMethod]
        [DataRow(-1.0)]
        [DataRow(-0.01)]
        [DataRow(1.01)]
        [DataRow(2.0)]
        [DataRow(99.0)]
        public void TestGammaOutsideUnitIntervalIsRejected(double gamma) {
            var error = Assert.ThrowsExactly<InvalidBound>(() => new NormalizeReward<object?, object?>(new GenericTestEnv(), gamma: gamma));
            StringAssert.Contains(error.Message, "`gamma` should be in the interval");
        }

        /// <summary>Both ends of [0, 1] are accepted: 0 keeps only the immediate reward, 1 is undiscounted.</summary>
        /// <param name="gamma">The discount.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_normalize_reward.py#L100">tests/wrappers/test_normalize_reward.py (test_gamma_inside_unit_interval_is_accepted)</see>.</remarks>
        [TestMethod]
        [DataRow(0.0)]
        [DataRow(0.5)]
        [DataRow(0.99)]
        [DataRow(1.0)]
        public void TestGammaInsideUnitIntervalIsAccepted(double gamma) {
            using var env = new NormalizeReward<object?, object?>(new GenericTestEnv(), gamma: gamma);
            Assert.AreEqual(gamma, env.Gamma);
        }

        /// <summary>A non-positive <c>epsilon</c> is refused: it sits under a square root.</summary>
        /// <param name="epsilon">The epsilon.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_normalize_reward.py#L106">tests/wrappers/test_normalize_reward.py (test_non_positive_epsilon_is_rejected)</see>.</remarks>
        [TestMethod]
        [DataRow(0.0)]
        [DataRow(-1e-8)]
        [DataRow(-1.0)]
        public void TestNonPositiveEpsilonIsRejected(double epsilon) {
            var error = Assert.ThrowsExactly<InvalidBound>(() => new NormalizeReward<object?, object?>(new GenericTestEnv(), epsilon: epsilon));
            StringAssert.Contains(error.Message, "`epsilon` should be strictly positive");
        }

        /// <summary><c>list.pop(0)</c>.</summary>
        /// <param name="items">The list.</param>
        /// <returns>The first item, removed.</returns>
        private static long Pop(List<long> items) {
            long first = items[0];
            items.RemoveAt(0);
            return first;
        }

        /// <summary><c>np.testing.assert_almost_equal(actual, desired, decimal=4)</c>: <c>abs(desired - actual) &lt; 1.5 * 10**-4</c>.</summary>
        /// <param name="desired">The expected value.</param>
        /// <param name="actual">The value.</param>
        /// <exception cref="AssertFailedException">Thrown when they differ by more.</exception>
        private static void AssertAlmostEqual(double desired, double actual) =>
            Assert.IsTrue(System.Math.Abs(desired - actual) < 1.5e-4, $"{actual} is not {desired} to 4 decimals");
    }
}
