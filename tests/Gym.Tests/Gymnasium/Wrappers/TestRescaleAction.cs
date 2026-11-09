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
    /// <summary>Gymnasium's <c>RescaleAction</c> tests (<c>test_rescale_action.py</c>), run against <see cref="RescaleAction{TObs, TAct}"/>.</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_rescale_action.py">tests/wrappers/test_rescale_action.py</see>.</remarks>
    [TestClass]
    public class TestRescaleAction {
        /// <summary>Actions are rescaled from the wrapper's bounds onto the env's, infinite components shifted only.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_rescale_action.py#L13">tests/wrappers/test_rescale_action.py (test_rescale_action_wrapper)</see>.</remarks>
        [TestMethod]
        public void TestRescaleActionWrapper() {
            float inf = float.PositiveInfinity;
            var env = new GenericTestEnv(stepFunc: WrapperUtils.RecordActionStep,
                actionSpace: new Box(np.array(new[] { 0f, 1f, -inf, 5f, -inf }), np.array(new[] { 1f, 3f, inf, inf, 7f })));
            using var wrappedEnv = new RescaleAction<object?, object?>(env,
                np.array(new[] { -5f, 0f, -inf, -1f, -inf }), np.array(new[] { 5f, 1f, inf, inf, 4f }));
            using (var expectedSpace = new Box(np.array(new[] { -5f, 0f, -inf, -1f, -inf }), np.array(new[] { 5f, 1f, inf, inf, 4f })))
                Assert.IsTrue(wrappedEnv.ActionSpace.Equals(expectedSpace));

            var cases = new[] {
                (new[] { 0.0f, 0.5f, 7.0f, -1.0f, -23.0f }, new[] { 0.5f, 2.0f, 7.0f, 5.0f, -20.0f }),
                (new[] { -5.0f, 0.0f, -4.0f, 0.0f, -3.0f }, new[] { 0.0f, 1.0f, -4.0f, 6.0f, 0.0f }),
                (new[] { 5.0f, 1.0f, 0.0f, 1.0f, 4.0f }, new[] { 1.0f, 3.0f, 0.0f, 7.0f, 7.0f }),
            };
            foreach ((float[] sample, float[] expected) in cases) {
                using NDArray sampleAction = np.array(sample);
                using NDArray expectedAction = np.array(expected);
                Assert.IsTrue(wrappedEnv.ActionSpace.Contains(sampleAction));
                Assert.IsTrue(env.ActionSpace.Contains(expectedAction));
                var (_, _, _, _, info) = wrappedEnv.Step(sampleAction);
                Assert.IsTrue(np.array_equal((NDArray)info["action"]!, expectedAction));
            }
        }

        /// <summary>A min action equal to the max action is refused: the inverse map would divide by zero.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_rescale_action.py#L53">tests/wrappers/test_rescale_action.py (test_rescale_action_equal_bounds)</see>.</remarks>
        [TestMethod]
        public void TestRescaleActionEqualBounds() {
            var env = new GenericTestEnv(stepFunc: WrapperUtils.RecordActionStep, actionSpace: new Box(np.array(new[] { 0f, 1f }), np.array(new[] { 1f, 3f })));
            Assert.ThrowsExactly<InvalidBound>(() => new RescaleAction<object?, object?>(env, 0f, 0f));
            // Only one of the two components is degenerate.
            Assert.ThrowsExactly<InvalidBound>(() => new RescaleAction<object?, object?>(env, np.array(new[] { -1f, 2f }), np.array(new[] { 1f, 2f })));
        }
    }
}
