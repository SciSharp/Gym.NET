#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Spaces;
using Gym.Utils;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Wrappers {
    /// <summary>Gymnasium's <c>ClipReward</c> tests (<c>test_clip_reward.py</c>), run against <see cref="ClipReward{TObs, TAct}"/>.</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_clip_reward.py">tests/wrappers/test_clip_reward.py</see>.</remarks>
    [TestClass]
    public class TestClipReward {
        /// <summary>The reward is clipped by the bounds given.</summary>
        /// <param name="lowerBound">The lower bound, or NaN for none.</param>
        /// <param name="upperBound">The upper bound, or NaN for none.</param>
        /// <param name="expectedReward">CartPole's first reward clipped.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_clip_reward.py#L16">tests/wrappers/test_clip_reward.py (test_clip_reward_wrapper)</see>.</remarks>
        [TestMethod]
        [DataRow(double.NaN, 0.5, 0.5)]
        [DataRow(0.0, double.NaN, 1.0)]
        [DataRow(0.0, 0.5, 0.5)]
        public void TestClipRewardWrapper(double lowerBound, double upperBound, double expectedReward) {
            using var env = new ClipReward<NDArray, NDArray>((Env<NDArray, NDArray>)Make(WrapperUtils.EnvId),
                double.IsNaN(lowerBound) ? null : lowerBound, double.IsNaN(upperBound) ? null : upperBound);
            using (NDArray first = env.Reset(seed: WrapperUtils.Seed).Observation) { }
            using NDArray action = NDArray.Scalar(WrapperUtils.DiscreteAction);
            var (obs, rew, _, _, _) = env.Step(action);
            obs.Dispose();
            Assert.AreEqual(expectedReward, rew);
        }

        /// <summary>No bounds, or an upper bound below the lower one, is refused with <c>InvalidBound</c>.</summary>
        /// <param name="boundsCase">The case: <c>none</c>, <c>numbers</c> (1, -1) or <c>arrays</c> ([1, 1], [0, 0]).</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_clip_reward.py#L33">tests/wrappers/test_clip_reward.py (test_clip_reward_incorrect_params)</see>.</remarks>
        [TestMethod]
        [DataRow("none")]
        [DataRow("numbers")]
        [DataRow("arrays")]
        public void TestClipRewardIncorrectParams(string boundsCase) {
            using var env = (Env<NDArray, NDArray>)Make(WrapperUtils.EnvId);
            using NDArray ones = np.array(new[] { 1L, 1L });
            using NDArray zeros = np.array(new[] { 0L, 0L });
            (object? lower, object? upper) = boundsCase switch {
                "none" => ((object?)null, (object?)null),
                "numbers" => ((object?)1L, (object?)(-1L)),
                _ => ((object?)ones, (object?)zeros),
            };
            Assert.ThrowsExactly<InvalidBound>(() => new ClipReward<NDArray, NDArray>(env, lower, upper));
        }
    }
}
