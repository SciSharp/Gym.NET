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
    /// <summary>Gymnasium's <c>DiscretizeObservation</c> tests (<c>test_discretize_observation.py</c>), run against <see cref="DiscretizeObservation{TObs, TAct}"/>.</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_discretize_observation.py">tests/wrappers/test_discretize_observation.py</see>.</remarks>
    [TestClass]
    public class TestDiscretizeObservation {
        /// <summary>A <c>Box</c> is discretized uniformly: the grid of every bin's corner maps onto every flat index once.</summary>
        /// <param name="dimensions">The box's length.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_discretize_observation.py#L12">tests/wrappers/test_discretize_observation.py (test_discretize_observation_space_uniformity)</see>.</remarks>
        [TestMethod]
        [DataRow(1)]
        [DataRow(2)]
        [DataRow(3)]
        [DataRow(5)]
        public void TestDiscretizeObservationSpaceUniformity(int dimensions) {
            var env = new GenericTestEnv(observationSpace: new Box(0, 99, new Shape(dimensions)));
            const int nBins = 7;
            using var wrapped = new DiscretizeObservation<object?, object?>(env, (long)nBins);
            // np.meshgrid of np.linspace(0, 99, 7) per dimension: every combination of the 7 float64 points.
            double[] points = Enumerable.Range(0, nBins).Select(i => i * (99.0 / (nBins - 1))).ToArray();
            points[nBins - 1] = 99.0;
            int count = (int)Math.Pow(nBins, dimensions);
            var discretized = new long[count];
            for (int combination = 0; combination < count; combination++) {
                var observation = new double[dimensions];
                for (int d = 0, rest = combination; d < dimensions; d++, rest /= nBins)
                    observation[d] = points[rest % nBins];
                using NDArray obs = np.array(observation);
                using NDArray index = wrapped.Observation(obs);
                discretized[combination] = Convert.ToInt64(index.GetAtIndex(0));
            }

            Array.Sort(discretized);
            Assert.AreEqual((long)count, ((Discrete)wrapped.ObservationSpace).N);
            CollectionAssert.AreEqual(Enumerable.Range(0, count).Select(i => (long)i).ToArray(), discretized);
        }

        /// <summary>An observation lies within the bin edges its discretization reverts to.</summary>
        /// <param name="dimensions">The box's length.</param>
        /// <param name="bins">The bins, comma-separated.</param>
        /// <param name="multidiscrete">Whether observations are MultiDiscrete.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_discretize_observation.py#L35">tests/wrappers/test_discretize_observation.py (test_revert_discretize_observation_space)</see>.</remarks>
        [TestMethod]
        [DataRow(1, "3", false)]
        [DataRow(2, "3,4", false)]
        [DataRow(3, "3,4,5", false)]
        [DataRow(1, "3", true)]
        [DataRow(2, "3,4", true)]
        [DataRow(3, "3,4,5", true)]
        public void TestRevertDiscretizeObservationSpace(int dimensions, string bins, bool multidiscrete) {
            var env = new GenericTestEnv(observationSpace: new Box(0, 99, new Shape(dimensions)));
            using var envDiscrete = new DiscretizeObservation<object?, object?>(env, Bins(bins), multidiscrete);
            for (int i = 0; i < 1000; i++) {
                var (obs, _) = env.Reset(seed: i);
                var (obsDiscrete, _) = envDiscrete.Reset(seed: i);
                (NDArray low, NDArray high) = envDiscrete.RevertObservation(obsDiscrete);
                using (low)
                using (high)
                using (var original = (NDArray)obs!)
                using (obsDiscrete) {
                    using NDArray above = original >= low;
                    using NDArray below = original <= high;
                    Assert.IsTrue(np.all(above) && np.all(below));
                }
            }
        }

        /// <summary>A box with an infinite bound is refused with <c>ValueError</c>.</summary>
        /// <param name="high">The first parametrized bound (Gymnasium's names are swapped: it is passed as <c>high</c>).</param>
        /// <param name="low">The second.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_discretize_observation.py#L49">tests/wrappers/test_discretize_observation.py (test_discretize_observation_bounds)</see>.</remarks>
        [TestMethod]
        [DataRow(0.0, double.PositiveInfinity)]
        [DataRow(double.NegativeInfinity, double.PositiveInfinity)]
        [DataRow(double.NegativeInfinity, 0.0)]
        public void TestDiscretizeObservationBounds(double high, double low) {
            // Box(low, high, shape=(1,)) as Gymnasium's test calls it; either the Box or the wrapper refuses with ValueError.
            Assert.ThrowsExactly<ValueError>(() => new DiscretizeObservation<object?, object?>(new GenericTestEnv(observationSpace: new Box(low, high, new Shape(1))), 3L));
        }

        /// <summary>A <c>Discrete</c> space is refused with <c>TypeError</c>.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_discretize_observation.py#L57">tests/wrappers/test_discretize_observation.py (test_discretize_observation_dtype)</see>.</remarks>
        [TestMethod]
        public void TestDiscretizeObservationDtype() {
            Assert.ThrowsExactly<TypeError>(() => new DiscretizeObservation<object?, object?>(new GenericTestEnv(observationSpace: new Discrete(10)), 3L));
        }

        /// <summary>Any finite box is accepted, not only a 1-D one: a <c>(2, 2)</c> box gives <c>Discrete(16)</c>.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_discretize_observation.py#L63">tests/wrappers/test_discretize_observation.py (test_discretize_observation_multidimensional_box)</see>.</remarks>
        [TestMethod]
        public void TestDiscretizeObservationMultidimensionalBox() {
            var env = new GenericTestEnv(observationSpace: new Box(0, 1, new Shape(2, 2), np.float32));
            using var wrapped = new DiscretizeObservation<object?, object?>(env, 2L);
            using (var expected = new Discrete(16))
                Assert.IsTrue(wrapped.ObservationSpace.Equals(expected));
            var (obs, _) = wrapped.Reset();
            Assert.IsTrue(wrapped.ObservationSpace.Contains(obs));
        }

        /// <summary>The bins argument of a case: one number, or a Python tuple of them.</summary>
        /// <param name="bins">The bins, comma-separated.</param>
        /// <returns>A <see cref="long"/> or an <see langword="object"/>?[] of them.</returns>
        internal static object Bins(string bins) {
            long[] values = bins.Split(',').Select(long.Parse).ToArray();
            return values.Length == 1 ? values[0] : values.Select(v => (object?)v).ToArray();
        }
    }
}
