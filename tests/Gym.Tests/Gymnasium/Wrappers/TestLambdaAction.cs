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
    /// <summary>Gymnasium's <c>TransformAction</c> tests (<c>test_lambda_action.py</c>), run against <see cref="TransformAction{TObs, TWAct, TAct}"/>.</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_lambda_action.py">tests/wrappers/test_lambda_action.py</see>.</remarks>
    [TestClass]
    public class TestLambdaAction {
        /// <summary>The env gets the transformed action.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_lambda_action.py#L9">tests/wrappers/test_lambda_action.py (test_lambda_action_wrapper)</see>.</remarks>
        [TestMethod]
        public void TestLambdaActionWrapper() {
            var env = new GenericTestEnv(stepFunc: WrapperUtils.RecordActionStep);
            using var wrappedEnv = new TransformAction<object?, NDArray, object?>(env, action => action - 2, new Box(2, 3));
            using NDArray sampledAction = wrappedEnv.ActionSpace.Sample();
            Assert.IsFalse(env.ActionSpace.Contains(sampledAction));
            var (_, _, _, _, info) = wrappedEnv.Step(sampledAction);
            var recorded = (NDArray)info["action"]!;
            Assert.IsTrue(env.ActionSpace.Contains(recorded));
            using NDArray expected = sampledAction - 2;
            Assert.IsTrue(np.array_equal(expected, recorded));
        }
    }
}
