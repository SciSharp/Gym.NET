#nullable enable
using System.Linq;
using Gym.Envs;
using Gym.Observations;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Wrappers {
    /// <summary>
    ///     Gymnasium's <c>RecordEpisodeStatistics</c> tests (<c>tests/wrappers/test_record_episode_statistics.py</c>): every
    ///     finished episode adds <c>{r, l, t}</c> to the info, and the queues keep the last <c>deque_size</c> episodes.
    /// </summary>
    /// <remarks>
    ///     Gymnasium runs it on CartPole-v1 and Pendulum-v1; the Pendulum cases are the <c>[OpenBugs("P4")]</c> method, which
    ///     passes once Pendulum registers. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_record_episode_statistics.py">tests/wrappers/test_record_episode_statistics.py</see>.
    /// </remarks>
    [TestClass]
    public class TestRecordEpisodeStatistics {
        /// <summary>Five episodes of CartPole with queues of 2 and of 5.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_record_episode_statistics.py#L9">test_record_episode_statistics</see>.</remarks>
        [TestMethod]
        public void TestRecordEpisodeStatisticsCartPole() {
            foreach (int dequeSize in new[] { 2, 5 })
                RunEpisodes("CartPole-v1", dequeSize);
        }

        /// <summary>The Pendulum cases (P4).</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_record_episode_statistics.py#L9">test_record_episode_statistics</see>.</remarks>
        [TestMethod]
        [OpenBugs("P4")]
        public void TestRecordEpisodeStatisticsPendulum() {
            foreach (int dequeSize in new[] { 2, 5 })
                RunEpisodes("Pendulum-v1", dequeSize);
        }

        /// <summary>Gymnasium's test body: five episodes of random actions, the statistics in the info of each episode's last step.</summary>
        /// <param name="envId">The env id.</param>
        /// <param name="dequeSize">The queue length.</param>
        private static void RunEpisodes(string envId, int dequeSize) {
            // The wrapper owns the made env: disposing it disposes the whole stack.
            IEnv made = Make(envId, disableEnvChecker: true);
            using IEnv env = (IEnv)Gym.Utils.PyKwargs.Construct(typeof(RecordEpisodeStatistics<,>), new Info { ["env"] = made, ["buffer_length"] = (long)dequeSize });
            dynamic statistics = env;

            for (int episode = 0; episode < 5; episode++) {
                env.Reset();
                Assert.AreEqual(0.0, (double)statistics.EpisodeReturns);
                Assert.AreEqual(0L, (long)statistics.EpisodeLengths);
                Assert.IsNotNull(env.Spec);
                for (long step = 0; step < env.Spec.MaxEpisodeSteps; step++) {
                    StepResult<object?> result = env.Step(env.ActionSpace.Sample());
                    if (result.Terminated || result.Truncated) {
                        Assert.IsTrue(result.Info.ContainsKey("episode"));
                        var episodeInfo = (Info)result.Info["episode"]!;
                        Assert.IsTrue(new[] { "r", "l", "t" }.All(episodeInfo.ContainsKey));
                        break;
                    }
                }
            }

            Assert.AreEqual(dequeSize, (int)statistics.ReturnQueue.Count);
            Assert.AreEqual(dequeSize, (int)statistics.LengthQueue.Count);
        }
    }
}
