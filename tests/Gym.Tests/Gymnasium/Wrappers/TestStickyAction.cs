#nullable enable
using System;
using System.Collections.Generic;
using Gym.Exceptions;
using Gym.Spaces;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Wrappers {
    /// <summary>Gymnasium's <c>StickyAction</c> tests (<c>test_sticky_action.py</c>), run against <see cref="StickyAction{TObs, TAct}"/>.</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_sticky_action.py">tests/wrappers/test_sticky_action.py</see>.</remarks>
    [TestClass]
    public class TestStickyAction {
        /// <summary>
        ///     The actions the env executes, seeded with 11, for a probability of 0.25 and fixed or drawn run lengths: the same
        ///     sequences Gymnasium expects, since the draws come from the env's PCG64 generator.
        /// </summary>
        /// <param name="durationCase"><c>1</c>, <c>2</c> or <c>(1, 3)</c>.</param>
        /// <param name="expectedAction">The executed actions for the agent's actions 0 to 7.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_sticky_action.py#L20">tests/wrappers/test_sticky_action.py (test_sticky_action)</see>.</remarks>
        [TestMethod]
        [DataRow("1", new long[] { 0, 0, 2, 3, 3, 3, 6, 6 })]
        [DataRow("2", new long[] { 0, 0, 0, 3, 4, 4, 4, 4 })]
        [DataRow("(1, 3)", new long[] { 0, 0, 0, 0, 4, 4, 4, 4 })]
        public void TestStickyActionWrapper(string durationCase, long[] expectedAction) {
            const double repeatActionProbability = 0.25;
            object repeatActionDuration = durationCase switch {
                "1" => 1L,
                "2" => 2L,
                _ => (1L, 3L),
            };
            long[] actions = { 0, 1, 2, 3, 4, 5, 6, 7 };
            using var env = new StickyAction<object?, object?>(
                new GenericTestEnv(stepFunc: WrapperUtils.RecordActionAsObsStep, observationSpace: new Discrete(7)),
                repeatActionProbability, repeatActionDuration);
            env.Reset(seed: 11);

            Assert.AreEqual(actions.Length, expectedAction.Length);
            for (int i = 0; i < actions.Length; i++) {
                object? executedAction = env.Step(actions[i]).Observation;
                Assert.AreEqual(expectedAction[i], executedAction, $"step {i}");
            }
        }

        /// <summary>A probability outside [0, 1) is refused with <c>InvalidProbability</c>.</summary>
        /// <param name="repeatActionProbability">The probability.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_sticky_action.py#L40">tests/wrappers/test_sticky_action.py (test_sticky_action_raise_probability)</see>.</remarks>
        [TestMethod]
        [DataRow(-1.0)]
        [DataRow(1.0)]
        [DataRow(1.5)]
        public void TestStickyActionRaiseProbability(double repeatActionProbability) {
            Assert.ThrowsExactly<InvalidProbability>(() => new StickyAction<object?, object?>(new GenericTestEnv(), repeatActionProbability));
        }

        /// <summary>A duration below 1, reversed, or given as a list is refused with <c>ValueError</c> or <c>InvalidBound</c>.</summary>
        /// <param name="durationCase"><c>-4</c>, <c>0</c>, <c>(0, 0)</c>, <c>(4, 2)</c> or <c>[1, 2]</c>.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_sticky_action.py#L58">tests/wrappers/test_sticky_action.py (test_sticky_action_raise_duration)</see>.</remarks>
        [TestMethod]
        [DataRow("-4")]
        [DataRow("0")]
        [DataRow("(0, 0)")]
        [DataRow("(4, 2)")]
        [DataRow("[1, 2]")]
        public void TestStickyActionRaiseDuration(string durationCase) {
            object repeatActionDuration = durationCase switch {
                "-4" => -4L,
                "0" => 0L,
                "(0, 0)" => (0L, 0L),
                "(4, 2)" => (4L, 2L),
                _ => new List<object?> { 1L, 2L },
            };
            // pytest.raises((ValueError, InvalidBound)): either type passes.
            Exception? raised = null;
            try {
                using var env = new StickyAction<object?, object?>(new GenericTestEnv(), 0.5, repeatActionDuration);
            } catch (Exception e) {
                raised = e;
            }

            Assert.IsTrue(raised is ValueError or InvalidBound, $"expected ValueError or InvalidBound, got {raised?.GetType().Name ?? "nothing"}");
        }
    }
}
