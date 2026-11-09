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
    /// <summary>Gymnasium's <c>ClipAction</c> tests (<c>test_clip_action.py</c>), run against <see cref="ClipAction{TObs, TAct}"/>.</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_clip_action.py">tests/wrappers/test_clip_action.py</see>.</remarks>
    [TestClass]
    public class TestClipAction {
        /// <summary>The action is clipped into the env's action space.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_clip_action.py#L11">tests/wrappers/test_clip_action.py (test_clip_action_wrapper)</see>.</remarks>
        [TestMethod]
        public void TestClipActionWrapper() {
            var env = new GenericTestEnv(actionSpace: new Box(np.array(new[] { 0L, 0L, 3L }), np.array(new[] { 1L, 2L, 4L })), stepFunc: WrapperUtils.RecordActionStep);
            using var wrappedEnv = new ClipAction<object?, object?>(env);
            using NDArray sampledAction = np.array(new[] { -1f, 5f, 3.5f });
            Assert.IsFalse(env.ActionSpace.Contains(sampledAction));
            Assert.IsTrue(wrappedEnv.ActionSpace.Contains(sampledAction));
            var (_, _, _, _, info) = wrappedEnv.Step(sampledAction);
            var recorded = (NDArray)info["action"]!;
            Assert.IsTrue(env.ActionSpace.Contains(recorded));
            using NDArray expected = np.array(new[] { 0.0, 2.0, 3.5 });
            Assert.IsTrue(np.array_equal(recorded, expected));
        }
    }
}
