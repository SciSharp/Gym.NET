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
    /// <summary>Gymnasium's <c>TransformReward</c> tests (<c>test_lambda_reward.py</c>), run against <see cref="TransformReward{TObs, TAct}"/>.</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_lambda_reward.py">tests/wrappers/test_lambda_reward.py</see>.</remarks>
    [TestClass]
    public class TestLambdaReward {
        /// <summary>The reward is transformed: <c>2 * r + 1</c>.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_lambda_reward.py#L8">tests/wrappers/test_lambda_reward.py (test_lambda_reward)</see>.</remarks>
        [TestMethod]
        public void TestLambdaRewardWrapper() {
            var env = new GenericTestEnv(stepFunc: WrapperUtils.RecordActionAsRecordStep);
            using var wrappedEnv = new TransformReward<object?, object?>(env, r => 2 * r + 1);
            var (_, rew, _, _, _) = wrappedEnv.Step(0L);
            Assert.AreEqual(1.0, rew);
            (_, rew, _, _, _) = wrappedEnv.Step(1L);
            Assert.AreEqual(3.0, rew);
        }
    }
}
