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
    /// <summary>Gymnasium's <c>DiscretizeAction</c> tests (<c>test_discretize_action.py</c>), run against <see cref="DiscretizeAction{TObs, TAct}"/>.</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_discretize_action.py">tests/wrappers/test_discretize_action.py</see>.</remarks>
    [TestClass]
    public class TestDiscretizeAction {
        /// <summary>A <c>Box</c> action space is discretized uniformly: reverting the grid of bin corners gives every flat index once.</summary>
        /// <param name="dimensions">The box's length.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_discretize_action.py#L12">tests/wrappers/test_discretize_action.py (test_discretize_action_space_uniformity)</see>.</remarks>
        [TestMethod]
        [DataRow(1)]
        [DataRow(2)]
        [DataRow(3)]
        [DataRow(5)]
        public void TestDiscretizeActionSpaceUniformity(int dimensions) {
            var env = new GenericTestEnv(actionSpace: new Box(0, 99, new Shape(dimensions)));
            const int nBins = 7;
            using var wrapped = new DiscretizeAction<object?, object?>(env, (long)nBins);
            double[] points = Enumerable.Range(0, nBins).Select(i => i * (99.0 / (nBins - 1))).ToArray();
            points[nBins - 1] = 99.0;
            int count = (int)Math.Pow(nBins, dimensions);
            var reverted = new long[count];
            for (int combination = 0; combination < count; combination++) {
                var action = new double[dimensions];
                for (int d = 0, rest = combination; d < dimensions; d++, rest /= nBins)
                    action[d] = points[rest % nBins];
                using NDArray act = np.array(action);
                using NDArray index = wrapped.RevertAction(act);
                reverted[combination] = Convert.ToInt64(index.GetAtIndex(0));
            }

            Array.Sort(reverted);
            Assert.AreEqual((long)count, ((Discrete)wrapped.ActionSpace).N);
            CollectionAssert.AreEqual(Enumerable.Range(0, count).Select(i => (long)i).ToArray(), reverted);
        }

        /// <summary>A sampled discrete action maps to a continuous action of the env, which reverts to the same discrete action.</summary>
        /// <param name="dimensions">The box's length.</param>
        /// <param name="bins">The bins, comma-separated.</param>
        /// <param name="multidiscrete">Whether actions are MultiDiscrete.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_discretize_action.py#L35">tests/wrappers/test_discretize_action.py (test_revert_discretize_action_space)</see>.</remarks>
        [TestMethod]
        [DataRow(1, "3", false)]
        [DataRow(2, "3,4", false)]
        [DataRow(3, "3,4,5", false)]
        [DataRow(1, "3", true)]
        [DataRow(2, "3,4", true)]
        [DataRow(3, "3,4,5", true)]
        public void TestRevertDiscretizeActionSpace(int dimensions, string bins, bool multidiscrete) {
            var env = new GenericTestEnv(actionSpace: new Box(0, 99, new Shape(dimensions)));
            using var envDiscrete = new DiscretizeAction<object?, object?>(env, TestDiscretizeObservation.Bins(bins), multidiscrete);
            for (int i = 0; i < 1000; i++) {
                using NDArray actDiscrete = envDiscrete.ActionSpace.Sample();
                using var actContinuous = (NDArray)envDiscrete.Action(actDiscrete)!;
                Assert.IsTrue(env.ActionSpace.Contains(actContinuous));
                using NDArray reverted = envDiscrete.RevertAction(actContinuous);
                Assert.IsTrue(np.array_equal(reverted, actDiscrete));
            }
        }

        /// <summary>A box with an infinite bound is refused with <c>ValueError</c>.</summary>
        /// <param name="high">The first parametrized bound (passed as <c>low</c>, as Gymnasium's test does).</param>
        /// <param name="low">The second.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_discretize_action.py#L47">tests/wrappers/test_discretize_action.py (test_discretize_action_bounds)</see>.</remarks>
        [TestMethod]
        [DataRow(0.0, double.PositiveInfinity)]
        [DataRow(double.NegativeInfinity, double.PositiveInfinity)]
        [DataRow(double.NegativeInfinity, 0.0)]
        public void TestDiscretizeActionBounds(double high, double low) {
            Assert.ThrowsExactly<ValueError>(() => new DiscretizeAction<object?, object?>(new GenericTestEnv(actionSpace: new Box(low, high, new Shape(1))), 3L));
        }

        /// <summary>A <c>Discrete</c> action space is refused with <c>TypeError</c>.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_discretize_action.py#L53">tests/wrappers/test_discretize_action.py (test_discretize_action_dtype)</see>.</remarks>
        [TestMethod]
        public void TestDiscretizeActionDtype() {
            Assert.ThrowsExactly<TypeError>(() => new DiscretizeAction<object?, object?>(new GenericTestEnv(actionSpace: new Discrete(10)), 3L));
        }

        /// <summary>A <c>(2, 2)</c> box gives <c>Discrete(16)</c>, and every action keeps the box's shape and reverts to itself.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_discretize_action.py#L59">tests/wrappers/test_discretize_action.py (test_discretize_action_multidimensional_box)</see>.</remarks>
        [TestMethod]
        public void TestDiscretizeActionMultidimensionalBox() {
            var env = new GenericTestEnv(actionSpace: new Box(0, 1, new Shape(2, 2), np.float32));
            using var wrapped = new DiscretizeAction<object?, object?>(env, 2L);
            using (var expected = new Discrete(16))
                Assert.IsTrue(wrapped.ActionSpace.Equals(expected));
            for (long i = 0; i < ((Discrete)wrapped.ActionSpace).N; i++) {
                using NDArray discrete = NDArray.Scalar(i);
                using var act = (NDArray)wrapped.Action(discrete)!;
                Assert.AreEqual(new Shape(2, 2), act.Shape);
                Assert.IsTrue(env.ActionSpace.Contains(act));
                using NDArray reverted = wrapped.RevertAction(act);
                Assert.AreEqual(i, Convert.ToInt64(reverted.GetAtIndex(0)));
            }
        }

        /// <summary>A 0-D box discretizes without an index error.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_discretize_action.py#L76">tests/wrappers/test_discretize_action.py (test_discretize_action_scalar_box)</see>.</remarks>
        [TestMethod]
        public void TestDiscretizeActionScalarBox() {
            var env = new GenericTestEnv(actionSpace: new Box(0, 1, Shape.Scalar, np.float32));
            using var wrapped = new DiscretizeAction<object?, object?>(env, 2L);
            using (var expected = new Discrete(2))
                Assert.IsTrue(wrapped.ActionSpace.Equals(expected));
            using NDArray zero = NDArray.Scalar(0L);
            using var act = (NDArray)wrapped.Action(zero)!;
            Assert.IsTrue(env.ActionSpace.Contains(act));
            using NDArray reverted = wrapped.RevertAction(act);
            Assert.AreEqual(0L, Convert.ToInt64(reverted.GetAtIndex(0)));
        }

        /// <summary><c>multidiscrete=True</c> on a <c>(2, 2)</c> box gives a <c>MultiDiscrete</c> of length 4.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_discretize_action.py#L86">tests/wrappers/test_discretize_action.py (test_discretize_action_multidimensional_box_multidiscrete)</see>.</remarks>
        [TestMethod]
        public void TestDiscretizeActionMultidimensionalBoxMultidiscrete() {
            var env = new GenericTestEnv(actionSpace: new Box(0, 1, new Shape(2, 2), np.float32));
            using var wrapped = new DiscretizeAction<object?, object?>(env, 2L, multidiscrete: true);
            using (var expected = new MultiDiscrete(new[] { 2L, 2L, 2L, 2L }))
                Assert.IsTrue(wrapped.ActionSpace.Equals(expected));
            Assert.AreEqual(4L, ((MultiDiscrete)wrapped.ActionSpace).Nvec.size);
            for (int i = 0; i < 16; i++) {
                using NDArray actDiscrete = wrapped.ActionSpace.Sample();
                using var act = (NDArray)wrapped.Action(actDiscrete)!;
                Assert.AreEqual(new Shape(2, 2), act.Shape);
                Assert.IsTrue(env.ActionSpace.Contains(act));
                using NDArray reverted = wrapped.RevertAction(act);
                Assert.IsTrue(np.array_equal(reverted, actDiscrete));
            }
        }
    }
}
