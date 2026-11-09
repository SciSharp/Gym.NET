#nullable enable
using Gym.Envs;
using Gym.Observations;
using Gym.Spaces;
using Gym.Utils;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Wrappers {
    /// <summary>Gymnasium's <c>RepeatAction</c> tests (<c>test_action_repeat.py</c>), run against <see cref="RepeatAction{TObs, TAct}"/>.</summary>
    /// <remarks>
    ///     <c>test_action_repeat_invalid_type</c>'s <c>TypeError</c> for float counts has no counterpart: the count is a
    ///     <see cref="long"/>, so the compiler refuses a float. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_action_repeat.py">tests/wrappers/test_action_repeat.py</see>.
    /// </remarks>
    [TestClass]
    public class TestActionRepeat {
        /// <summary>A step that returns an incrementing reward and the action as the observation (<c>_counting_step</c>).</summary>
        /// <param name="self">The env.</param>
        /// <param name="action">The action, which becomes the observation.</param>
        /// <returns><c>(action, float(step_count), False, False, {"step": step_count})</c>.</returns>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_action_repeat.py#L11">tests/wrappers/test_action_repeat.py (_counting_step)</see>.</remarks>
        private static StepResult<object?> CountingStep(GenericTestEnv self, object? action) {
            long stepCount = NextStepCount(self);
            return new StepResult<object?>(action, stepCount, false, false, new Info { ["step"] = stepCount });
        }

        /// <summary>A step that terminates after 3 steps (<c>_terminating_step</c>).</summary>
        /// <param name="self">The env.</param>
        /// <param name="action">The action, which becomes the observation.</param>
        /// <returns><c>(action, 1.0, step_count &gt;= 3, False, {"step": step_count})</c>.</returns>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_action_repeat.py#L18">tests/wrappers/test_action_repeat.py (_terminating_step)</see>.</remarks>
        private static StepResult<object?> TerminatingStep(GenericTestEnv self, object? action) {
            long stepCount = NextStepCount(self);
            return new StepResult<object?>(action, 1.0, stepCount >= 3, false, new Info { ["step"] = stepCount });
        }

        /// <summary>Exactly <c>num_repeats</c> inner steps: rewards 1 + 2 + 3, the last step's info, the action as the observation.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_action_repeat.py#L25">tests/wrappers/test_action_repeat.py (test_action_repeat_exact_steps)</see>.</remarks>
        [TestMethod]
        public void TestActionRepeatExactSteps() {
            using var env = new RepeatAction<object?, object?>(
                new GenericTestEnv(stepFunc: CountingStep, actionSpace: new Discrete(5), observationSpace: new Discrete(5)), numRepeats: 3);
            env.Reset();
            var (obs, reward, _, _, info) = env.Step(2L);
            // Rewards: 1.0 + 2.0 + 3.0 = 6.0 (proving exactly 3 inner steps)
            Assert.AreEqual(6.0, reward);
            Assert.AreEqual(3L, info["step"]); // info from last inner step
            Assert.AreEqual(2L, obs); // action passed through as observation
        }

        /// <summary>One wrapped step of CartPole sums the rewards of four manual steps and ends at the same observation.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_action_repeat.py#L43">tests/wrappers/test_action_repeat.py (test_action_repeat_reward_accumulation)</see>.</remarks>
        [TestMethod]
        public void TestActionRepeatRewardAccumulation() {
            using var env = (Env<NDArray, NDArray>)Make("CartPole-v1");
            using NDArray action = NDArray.Scalar(1L);

            // Without wrapper: step 4 times manually
            env.Reset(seed: 42);
            double totalRewardManual = 0.0;
            NDArray obsManual = null!;
            for (int i = 0; i < 4; i++) {
                (obsManual, double reward, bool term, bool trunc, _) = env.Step(action);
                totalRewardManual += reward;
                if (term || trunc)
                    break;
            }

            // With wrapper: single step should accumulate 4 rewards
            using var wrappedEnv = new RepeatAction<NDArray, NDArray>((Env<NDArray, NDArray>)Make("CartPole-v1"), numRepeats: 4);
            wrappedEnv.Reset(seed: 42);
            var (obsWrapped, rewardWrapped, _, _, _) = wrappedEnv.Step(action);

            Assert.AreEqual(totalRewardManual, rewardWrapped);
            Assert.IsTrue(np.array_equal(obsWrapped, obsManual));
        }

        /// <summary>The repetition stops at the step that terminates: three steps, reward 3, that step's info.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_action_repeat.py#L66">tests/wrappers/test_action_repeat.py (test_action_repeat_early_termination)</see>.</remarks>
        [TestMethod]
        public void TestActionRepeatEarlyTermination() {
            using var env = new RepeatAction<object?, object?>(
                new GenericTestEnv(stepFunc: TerminatingStep, actionSpace: new Discrete(3), observationSpace: new Discrete(3)), numRepeats: 10);
            env.Reset();
            var (_, reward, terminated, _, info) = env.Step(0L);
            // Environment terminates at step 3, so only 3 inner steps taken
            Assert.IsTrue(terminated);
            Assert.AreEqual(3.0, reward); // 1.0 * 3 steps
            Assert.AreEqual(3L, info["step"]);
        }

        /// <summary>The info returned is the last inner step's.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_action_repeat.py#L84">tests/wrappers/test_action_repeat.py (test_action_repeat_info_propagation)</see>.</remarks>
        [TestMethod]
        public void TestActionRepeatInfoPropagation() {
            using var env = new RepeatAction<object?, object?>(
                new GenericTestEnv(stepFunc: CountingStep, actionSpace: new Discrete(5), observationSpace: new Discrete(5)), numRepeats: 5);
            env.Reset();
            var (_, _, _, _, info) = env.Step(0L);
            Assert.AreEqual(5L, info["step"]); // info from the 5th (last) inner step
        }

        /// <summary>With one repeat, the wrapper steps exactly as the unwrapped env.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_action_repeat.py#L99">tests/wrappers/test_action_repeat.py (test_action_repeat_num_repeats_one)</see>.</remarks>
        [TestMethod]
        public void TestActionRepeatNumRepeatsOne() {
            using var env1 = (Env<NDArray, NDArray>)Make("CartPole-v1");
            using var wrappedEnv = new RepeatAction<NDArray, NDArray>((Env<NDArray, NDArray>)Make("CartPole-v1"), numRepeats: 1);

            env1.Reset(seed: 42);
            wrappedEnv.Reset(seed: 42);

            for (int i = 0; i < 10; i++) {
                using NDArray action = env1.ActionSpace.Sample();
                var (obs1, r1, term1, trunc1, _) = env1.Step(action);
                var (obs2, r2, term2, trunc2, _) = wrappedEnv.Step(action);
                Assert.AreEqual(r1, r2);
                Assert.AreEqual(term1, term2);
                Assert.AreEqual(trunc1, trunc2);
                Assert.IsTrue(np.array_equal(obs1, obs2));
                if (term1 || trunc1)
                    break;
            }
        }

        /// <summary>The wrapper keeps the env's spaces, and its observations are members of them.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_action_repeat.py#L120">tests/wrappers/test_action_repeat.py (test_action_repeat_observation_space)</see>.</remarks>
        [TestMethod]
        public void TestActionRepeatObservationSpace() {
            var env = (Env<NDArray, NDArray>)Make("CartPole-v1");
            using var wrappedEnv = new RepeatAction<NDArray, NDArray>(env, numRepeats: 4);

            Assert.AreEqual(env.ObservationSpace, wrappedEnv.ObservationSpace);
            Assert.AreEqual(env.ActionSpace, wrappedEnv.ActionSpace);

            NDArray obs = wrappedEnv.Reset(seed: 123).Observation;
            Assert.IsTrue(wrappedEnv.ObservationSpace.Contains(obs));

            for (int i = 0; i < 5; i++) {
                (obs, _, bool term, bool trunc, _) = wrappedEnv.Step(wrappedEnv.ActionSpace.Sample());
                Assert.IsTrue(wrappedEnv.ObservationSpace.Contains(obs));
                if (term || trunc)
                    obs = wrappedEnv.Reset().Observation;
            }
        }

        /// <summary>Counts below 1 are refused with Gymnasium's message.</summary>
        /// <param name="numRepeats">The count.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_action_repeat.py#L147">tests/wrappers/test_action_repeat.py (test_action_repeat_invalid_value)</see>.</remarks>
        [TestMethod]
        [DataRow(0L)]
        [DataRow(-1L)]
        [DataRow(-100L)]
        public void TestActionRepeatInvalidValue(long numRepeats) {
            using var env = (Env<NDArray, NDArray>)Make("CartPole-v1");
            var error = Assert.ThrowsExactly<ValueError>(() => new RepeatAction<NDArray, NDArray>(env, numRepeats));
            StringAssert.Contains(error.Message, "equal or greater than one");
        }

        /// <summary><c>self._step_count = getattr(self, "_step_count", 0) + 1</c>, kept on the env as Python keeps it.</summary>
        /// <param name="self">The env.</param>
        /// <returns>The incremented count.</returns>
        private static long NextStepCount(GenericTestEnv self) {
            // getattr with a default: the attribute exists only after the first step.
            long count = (PyAttributes.HasAttr(self, "_step_count") ? (long)PyAttributes.GetAttr(self, "_step_count")! : 0L) + 1;
            PyAttributes.SetAttr(self, "_step_count", count);
            return count;
        }
    }
}
