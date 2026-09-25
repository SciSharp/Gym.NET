#nullable enable
using System;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Observations;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Ported.Wrappers {
    /// <summary>
    ///     Gymnasium's <c>TimeLimit</c> tests, run against <see cref="TimeLimit{TObs, TAct}"/>.
    /// </summary>
    /// <remarks>
    ///     Two substitutions, until Gym.NET has the code they need: <c>gym.make("CartPole-v1")</c> is built by hand as
    ///     <c>make</c> would (P3), and <c>PendulumEnv</c>, used because it never terminates (P4), is a
    ///     <see cref="GenericTestEnv"/>, whose default step never terminates either. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/tests/wrappers/test_time_limit.py">tests/wrappers/test_time_limit.py</see>.
    /// </remarks>
    [TestClass]
    public class TestTimeLimit {
        /// <summary>The reset observation belongs to the space and the info is a dict.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/tests/wrappers/test_time_limit.py#L10">tests/wrappers/test_time_limit.py (test_time_limit_reset_info)</see>.</remarks>
        [TestMethod]
        public void TestTimeLimitResetInfo() {
            var env = new TimeLimit<NDArray, long>(MakeCartPole(), 100);
            var obSpace = env.ObservationSpace;
            var (obs, info) = env.Reset();
            Assert.IsTrue(obSpace.Contains(obs));
            Assert.IsNotNull(info);
        }

        /// <summary>An env that never terminates is truncated after exactly the limit.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/tests/wrappers/test_time_limit.py#L20">tests/wrappers/test_time_limit.py (test_time_limit_wrapper)</see>.</remarks>
        /// <param name="doubleWrap">Wrap twice with the same limit.</param>
        [TestMethod]
        [DataRow(false)]
        [DataRow(true)]
        public void TestTimeLimitWrapper(bool doubleWrap) {
            // The generic env does not terminate by default, so we are sure termination is only due to timeout.
            const int maxEpisodeLength = 20;
            Env<object?, object?> env = new TimeLimit<object?, object?>(new GenericTestEnv(), maxEpisodeLength);
            if (doubleWrap)
                env = new TimeLimit<object?, object?>(env, maxEpisodeLength);
            env.Reset();
            bool terminated = false, truncated = false;
            int nSteps = 0;
            while (!(terminated || truncated)) {
                nSteps += 1;
                (_, _, terminated, truncated, _) = env.Step(env.ActionSpace.Sample());
            }

            Assert.AreEqual(maxEpisodeLength, nSteps);
            Assert.IsTrue(truncated);
        }

        /// <summary>
        ///     Termination at the last timestep; truncation due to the timeout happens at the same step.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/tests/wrappers/test_time_limit.py#L41">tests/wrappers/test_time_limit.py (test_termination_on_last_step)</see>.</remarks>
        /// <param name="doubleWrap">Wrap twice with the same limit.</param>
        [TestMethod]
        [DataRow(false)]
        [DataRow(true)]
        public void TestTerminationOnLastStep(bool doubleWrap) {
            var generic = new GenericTestEnv(stepFunc: (self, _) => new StepResult<object?>(self.ObservationSpace.Sample(), 0.0, true, false, new Info()));

            const int maxEpisodeLength = 1;
            Env<object?, object?> env = new TimeLimit<object?, object?>(generic, maxEpisodeLength);
            if (doubleWrap)
                env = new TimeLimit<object?, object?>(env, maxEpisodeLength);
            env.Reset();
            (_, _, bool terminated, bool truncated, _) = env.Step(env.ActionSpace.Sample());
            Assert.IsTrue(terminated);
            Assert.IsTrue(truncated);
        }

        /// <summary>
        ///     The limit shows in the spec, and a limit that is not a positive integer is refused with Gymnasium's messages.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/tests/wrappers/test_time_limit.py#L62">tests/wrappers/test_time_limit.py (test_max_episode_steps)</see>.</remarks>
        [TestMethod]
        public void TestMaxEpisodeSteps() {
            var env = new TimeLimit<NDArray, long>(MakeCartPole(), 500);

            Assert.AreEqual(500L, env.Spec!.MaxEpisodeSteps);
            Assert.AreEqual(10L, new TimeLimit<NDArray, long>(env, maxEpisodeSteps: 10).Spec!.MaxEpisodeSteps);

            var valueError = Assert.ThrowsExactly<ValueError>(() => new TimeLimit<NDArray, long>(env, maxEpisodeSteps: -1));
            StringAssert.Contains(valueError.Message, "Expect the `max_episode_steps` to be positive, actually: -1");

            var typeError = Assert.ThrowsExactly<TypeError>(() => new TimeLimit<NDArray, long>(env, maxEpisodeSteps: null));
            StringAssert.Contains(typeError.Message, "Expect the `max_episode_steps` to be an integer");

            env.Close();
        }

        /// <summary>
        ///     <c>OrderEnforcing(CartPoleEnv)</c>, the unwrapped env carrying the spec <c>make</c> gives it (the wrappers'
        ///     settings cleared, so that each wrapper puts its own back): the part of
        ///     <c>gym.make("CartPole-v1", disable_env_checker=True)</c> beneath its <c>TimeLimit</c>.
        /// </summary>
        /// <returns>The order-enforced CartPole.</returns>
        internal static OrderEnforcing<NDArray, long> MakeCartPole() {
            var cartPole = new CartPoleEnv {
                Spec = new EnvSpec("CartPole-v1", "Gym.Environments.Envs.Classic:CartPoleEnv", rewardThreshold: 475.0, maxEpisodeSteps: null, orderEnforce: false,
                    disableEnvChecker: true)
            };
            return new OrderEnforcing<NDArray, long>(cartPole);
        }
    }
}
