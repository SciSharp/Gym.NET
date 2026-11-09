#nullable enable
using System;
using System.Linq;
using Gym.Spaces;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Spaces {
    /// <summary>
    ///     Gymnasium's <c>MultiBinary</c> tests (<c>tests/spaces/test_multibinary.py</c>): masked samples (0, 1, or 2 for
    ///     either) and probability-weighted samples.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_multibinary.py">tests/spaces/test_multibinary.py</see>.</remarks>
    [TestClass]
    public class TestMultiBinary {
        /// <summary>A mask fixes the 0 and 1 entries and leaves the 2 entries free, for a vector and a matrix space.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_multibinary.py#L6">test_sample</see>.</remarks>
        [TestMethod]
        public void TestSample() {
            using (var space = new MultiBinary(4)) {
                using (NDArray mask = np.array(new sbyte[] { 0, 0, 1, 1 }))
                using (NDArray sample = space.Sample(mask))
                    CollectionAssert.AreEqual(new sbyte[] { 0, 0, 1, 1 }, sample.ToArray<sbyte>());

                using (NDArray mask = np.array(new sbyte[] { 0, 1, 2, 2 }))
                using (NDArray sample = space.Sample(mask)) {
                    sbyte[] values = sample.ToArray<sbyte>();
                    Assert.AreEqual(0, values[0]);
                    Assert.AreEqual(1, values[1]);
                    Assert.IsTrue(values[2] is 0 or 1);
                    Assert.IsTrue(values[3] is 0 or 1);
                }
            }

            using (var space = new MultiBinary(new long[] { 2, 3 }))
            using (NDArray mask = np.array(new sbyte[,] { { 0, 0, 0 }, { 1, 1, 1 } }))
            using (NDArray sample = space.Sample(mask))
                CollectionAssert.AreEqual(new sbyte[] { 0, 0, 0, 1, 1, 1 }, sample.ToArray<sbyte>());
        }

        /// <summary>10000 probability-weighted samples are members, and each entry's frequency is within 0.05 of its probability.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_multibinary.py#L23">test_sample_probabilities</see>.</remarks>
        [TestMethod]
        public void TestSampleProbabilities() {
            using var space = new MultiBinary(4);
            using NDArray probabilities = np.array(new[] { 0, 1, 0.5, 0.25 });
            var counts = new double[4];
            const int trials = 10000;
            for (int i = 0; i < trials; i++) {
                using NDArray sample = space.Sample(probability: probabilities);
                Assert.IsTrue(space.Contains(sample));
                sbyte[] values = sample.ToArray<sbyte>();
                for (int j = 0; j < 4; j++)
                    counts[j] += values[j];
            }

            double[] expected = { 0, 1, 0.5, 0.25 };
            for (int j = 0; j < 4; j++)
                Assert.AreEqual(expected[j], counts[j] / trials, 0.05, $"entry {j}");
        }
    }
}
