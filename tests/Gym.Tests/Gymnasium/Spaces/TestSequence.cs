#nullable enable
using System;
using System.Linq;
using Gym.Spaces;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Spaces {
    /// <summary>
    ///     Gymnasium's <c>Sequence</c> tests (<c>tests/spaces/test_sequence.py</c>): stacked samples, length masks (an int or an
    ///     array of lengths) with their errors, and masked and weighted feature samples.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_sequence.py">tests/spaces/test_sequence.py</see>.</remarks>
    [TestClass]
    public class TestSequence {
        /// <summary>A stacked sequence of boxes is a 2-D array, and of a discrete space an array rather than a tuple.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_sequence.py#L9">test_stacked_sequence</see>.</remarks>
        [TestMethod]
        public void TestStackedSequence() {
            using (var space = new Sequence(new Box(0, 1, new Shape(3L)), stack: true))
            using (var sample = (NDArray)space.Sample()!)
                Assert.AreEqual(2, sample.ndim);

            using (var space = new Sequence(new Discrete(3), stack: true)) {
                object? sample = space.Sample();
                try {
                    Assert.IsInstanceOfType<NDArray>(sample);
                } finally {
                    (sample as IDisposable)?.Dispose();
                }
            }
        }

        /// <summary>Integer and array length masks give lengths from them; malformed ones are refused with Gymnasium's messages.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_sequence.py#L23">test_sample</see>.</remarks>
        [TestMethod]
        public void TestSample() {
            using var space = new Sequence(new Box(0, 1));
            for (long length = 0; length < 4; length++) {
                var sample = (object?[])space.Sample(new object?[] { length, null })!;
                try {
                    Assert.IsTrue(space.Contains(sample));
                    Assert.AreEqual(length, sample.Length);
                } finally {
                    Release(sample);
                }
            }

            Assert.AreEqual("Expects the length mask of `mask` to be greater than or equal to zero, actual value: -1",
                Assert.ThrowsExactly<AssertionError>(() => space.Sample(new object?[] { -1L, null })).Message);

            using (NDArray five = np.array(new long[] { 5 })) {
                var sample = (object?[])space.Sample(new object?[] { five, null })!;
                try {
                    Assert.IsTrue(space.Contains(sample));
                    Assert.AreEqual(5, sample.Length);
                } finally {
                    Release(sample);
                }
            }

            using (NDArray lengths = np.array(new long[] { 3, 4, 5 })) {
                var sample = (object?[])space.Sample(new object?[] { lengths, null })!;
                try {
                    Assert.IsTrue(space.Contains(sample));
                    CollectionAssert.Contains(new[] { 3, 4, 5 }, sample.Length);
                } finally {
                    Release(sample);
                }
            }

            foreach (string kind in new[] { "mask", "probability" }) {
                Func<object?[], object?> sampleWith = pair => kind == "mask" ? space.Sample(pair) : space.Sample(probability: pair);
                using (NDArray grid = np.array(new long[,] { { 2, 2 }, { 2, 2 } }))
                    Assert.AreEqual($"Expects the shape of the length mask of `{kind}` to be 1-dimensional, actual shape: (2, 2)",
                        Assert.ThrowsExactly<AssertionError>(() => sampleWith(new object?[] { grid, null })).Message);
                using (NDArray negative = np.array(new long[] { 1, 2, -1 }))
                    Assert.AreEqual($"Expects all values in the length_mask of `{kind}` to be greater than or equal to zero, actual values: [ 1  2 -1]",
                        Assert.ThrowsExactly<AssertionError>(() => sampleWith(new object?[] { negative, null })).Message);
                Assert.AreEqual($"Expects the type of length_mask of `{kind}` to be an integer or a np.ndarray, actual type: <class 'str'>",
                    Assert.ThrowsExactly<TypeError>(() => sampleWith(new object?[] { "abc", null })).Message);
            }
        }

        /// <summary>A length array and a feature mask give that many samples, all of the allowed value.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_sequence.py#L103">test_sample_with_mask</see>.</remarks>
        [TestMethod]
        public void TestSampleWithMask() {
            using var space = new Sequence(new Discrete(2));
            using NDArray length = np.array(new long[] { 20 });
            using NDArray featureMask = np.array(new sbyte[] { 0, 1 });
            var sample = (object?[])space.Sample(new object?[] { length, featureMask })!;
            try {
                Assert.AreEqual(20, sample.Length);
                Assert.IsTrue(sample.All(value => Convert.ToInt64(((NDArray)value!).GetAtIndex(0)) == 1));
                Assert.IsTrue(sample.All(value => space.FeatureSpace.Contains(value)));
            } finally {
                Release(sample);
            }
        }

        /// <summary>A length array and a feature probability mask give that many samples, weighted as asked.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_sequence.py#L113">test_sample_with_probability</see>.</remarks>
        [TestMethod]
        public void TestSampleWithProbability() {
            using (var space = new Sequence(new Discrete(2)))
            using (NDArray length = np.array(new long[] { 20 }))
            using (NDArray onlyOne = np.array(new[] { 0.0, 1.0 })) {
                var sample = (object?[])space.Sample(probability: new object?[] { length, onlyOne })!;
                try {
                    Assert.AreEqual(20, sample.Length);
                    Assert.IsTrue(sample.All(value => Convert.ToInt64(((NDArray)value!).GetAtIndex(0)) == 1));
                } finally {
                    Release(sample);
                }
            }

            using (var space = new Sequence(new Discrete(3)))
            using (NDArray length = np.array(new long[] { 1000 }))
            using (NDArray weights = np.array(new[] { 0.0, 0.2, 0.8 })) {
                var sample = (object?[])space.Sample(probability: new object?[] { length, weights })!;
                try {
                    long[] values = sample.Select(value => Convert.ToInt64(((NDArray)value!).GetAtIndex(0))).ToArray();
                    Assert.IsTrue(values.All(v => v is 1 or 2));
                    Assert.AreEqual(0.2, values.Count(v => v == 1) / (double)values.Length, 0.05);
                    Assert.AreEqual(0.8, values.Count(v => v == 2) / (double)values.Length, 0.05);
                } finally {
                    Release(sample);
                }
            }
        }

        /// <summary>Disposes a tuple sample's arrays.</summary>
        /// <param name="sample">The sample.</param>
        private static void Release(object?[] sample) {
            foreach (object? item in sample)
                (item as IDisposable)?.Dispose();
        }
    }
}
