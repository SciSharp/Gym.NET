#nullable enable
using System;
using System.Collections.Generic;
using System.Numerics;
using Gym.Spaces;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Spaces {
    /// <summary>
    ///     Gymnasium's <c>Discrete</c> tests (<c>tests/spaces/test_discrete.py</c>): masks and probability masks with their
    ///     errors, the sample dtype, equality by dtype, membership by dtype (castable NumPy scalars only), no overflow at a
    ///     small dtype's top, out-of-range Python ints, and the dtypes refused.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <c>test_space_legacy_pickling</c> tests Python's pickle protocol, which Gym.NET has no equivalent of; the
    ///     <c>None</c> and <c>str</c> dtypes of <c>test_dtype_error</c> are Python-only (C#'s <see langword="null"/> is the
    ///     int64 default, and NumSharp's <c>np.str_</c> is refused like any non-integer dtype, as the remaining cases show).
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_discrete.py">tests/spaces/test_discrete.py</see>.</para>
    /// </remarks>
    [TestClass]
    public class TestDiscrete {
        /// <summary>Masks restrict the sample; an all-zero mask gives <c>start</c>.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_discrete.py#L31">test_sample_mask</see>.</remarks>
        [TestMethod]
        public void TestSampleMask() {
            using var space = new Discrete(4, start: 2);
            long sample = Value(space.Sample());
            Assert.IsTrue(sample >= 2 && sample < 6);
            Assert.AreEqual(3, Value(space.Sample(Int8(0, 1, 0, 0))));
            Assert.AreEqual(2, Value(space.Sample(Int8(0, 0, 0, 0))));
            CollectionAssert.Contains(new long[] { 3, 5 }, Value(space.Sample(Int8(0, 1, 0, 1))));
        }

        /// <summary>Probability masks weight the sample.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_discrete.py#L40">test_probability_mask</see>.</remarks>
        [TestMethod]
        public void TestProbabilityMask() {
            using var space = new Discrete(4, start: 2);
            Assert.AreEqual(3, Value(space.Sample(probability: Float64(0, 1, 0, 0))));
            CollectionAssert.Contains(new long[] { 3, 5 }, Value(space.Sample(probability: Float64(0, 0.5, 0, 0.5))));
            CollectionAssert.Contains(new long[] { 2, 3, 4, 5 }, Value(space.Sample(probability: Float64(0.1, 0.2, 0.3, 0.4))));
        }

        /// <summary>A mask and a probability mask together are refused.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_discrete.py#L58">test_sample_with_mask_and_probability</see>.</remarks>
        [TestMethod]
        public void TestSampleWithMaskAndProbability() {
            using var space = new Discrete(4, start: 2);
            using NDArray mask = Int8(0, 1, 0, 0);
            using NDArray probability = Float64(0, 1, 0, 0);
            var error = Assert.ThrowsExactly<ValueError>(() => space.Sample(mask, probability));
            StringAssert.Contains(error.Message, "Only one of `mask` or `probability` can be provided");
        }

        /// <summary>A probability mask that is not float64 is refused with Gymnasium's message.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_discrete.py#L72">test_invalid_probability_mask_dtype</see>.</remarks>
        [TestMethod]
        public void TestInvalidProbabilityMaskDtype() {
            using var space = new Discrete(4, start: 2);
            using NDArray probability = Int8(0, 1, 0, 0);
            var error = Assert.ThrowsExactly<AssertionError>(() => space.Sample(probability: probability));
            StringAssert.Contains(error.Message, "The expected dtype of the sample probability is np.float64, actual dtype: int8");
        }

        /// <summary>Probabilities outside [0, 1] or not summing to 1 are refused with Gymnasium's messages.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_discrete.py#L85">test_invalid_probability_mask_values</see>.</remarks>
        [TestMethod]
        public void TestInvalidProbabilityMaskValues() {
            using var space = new Discrete(4, start: 2);
            using (NDArray probability = Float64(-0.5, 1, 0.5, 0))
                StringAssert.Contains(Assert.ThrowsExactly<AssertionError>(() => space.Sample(probability: probability)).Message,
                    "All values of the sample probability should be between 0 and 1, actual values: [-0.5  1.   0.5  0. ]");
            using (NDArray probability = Float64(0.2, 0.3, 0.4, 0.2))
                StringAssert.Contains(Assert.ThrowsExactly<AssertionError>(() => space.Sample(probability: probability)).Message,
                    "The sum of the sample probability should be equal to 1, actual sum: 1.1");
            using (NDArray probability = Float64(0, 0, 0, 0))
                StringAssert.Contains(Assert.ThrowsExactly<AssertionError>(() => space.Sample(probability: probability)).Message,
                    "The sum of the sample probability should be equal to 1, actual sum: 0.0");
        }

        /// <summary>Samples, masked or weighted, are 0-d arrays of the space's dtype (NumPy scalars of it).</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_discrete.py#L118">test_dtype</see>.</remarks>
        [TestMethod]
        public void TestDtype() {
            foreach (DType dtype in new[] { np.int64, np.int32, np.uint8 }) {
                using var space = new Discrete(5, start: 2, dtype: dtype);
                using NDArray sample = space.Sample();
                using NDArray sampleMask = space.Sample(Int8(0, 1, 0, 0, 0));
                using NDArray sampleProbability = space.Sample(probability: Float64(0.2, 0.2, 0.2, 0.2, 0.2));
                Assert.AreEqual(dtype, sample.dtype);
                Assert.AreEqual(dtype, sampleMask.dtype);
                Assert.AreEqual(dtype, sampleProbability.dtype);
                Assert.AreEqual(0, sample.ndim);
            }
        }

        /// <summary>Spaces of different dtypes are not equal.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_discrete.py#L132">test_equality</see>.</remarks>
        [TestMethod]
        public void TestEquality() {
            using var space = new Discrete(3, dtype: np.int32);
            using var spaceDifDtype = new Discrete(3);
            Assert.AreNotEqual(space, spaceDifDtype);
        }

        /// <summary>Membership of <c>{-1, 0, 1}</c> by dtype: a NumPy scalar must cast safely to the space's dtype; a Python int need not.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_discrete.py#L139">test_contains</see>.</remarks>
        [TestMethod]
        public void TestContains() {
            var cases = new (Func<object> Element, DType DType, bool Expected, string Label)[] {
                (() => NDArray.Scalar(0), np.int32, true, "np.int32(0) in int32"),
                (() => NDArray.Scalar(2), np.int32, false, "np.int32(2) in int32"),
                (() => NDArray.Scalar(-1L), np.int32, false, "np.int64(-1) in int32"),
                (() => NDArray.Scalar((sbyte)0), np.int16, true, "np.array(0, int8) in int16"),
                (() => NDArray.Scalar((short)0), np.int8, false, "np.array(0, int16) in int8"),
                (() => NDArray.Scalar(0L), np.int32, false, "np.array(0) in int32"),
                (() => NDArray.Scalar(0L), np.int64, true, "np.array(0) in int64"),
                (() => 0L, np.int16, true, "0 in int16"),
                (() => 0L, np.int32, true, "0 in int32"),
                (() => 0L, np.int64, true, "0 in int64"),
                (() => NDArray.Scalar(0), np.int64, true, "np.int32(0) in int64"),
                (() => NDArray.Scalar(0L), np.int16, false, "np.int64(0) in int16"),
                (() => NDArray.Scalar(0), np.int16, false, "np.int32(0) in int16"),
                (() => NDArray.Scalar(0L), np.int8, false, "np.int64(0) in int8"),
                (() => NDArray.Scalar((sbyte)0), np.int16, true, "np.int8(0) in int16"),
            };
            foreach ((Func<object> element, DType dtype, bool expected, string label) in cases) {
                using var space = new Discrete(3, start: -1, dtype: dtype);
                object value = element();
                try {
                    Assert.AreEqual(expected, space.Contains(value), label);
                } finally {
                    (value as IDisposable)?.Dispose();
                }
            }
        }

        /// <summary>A space ending at its dtype's top contains its top value and its own samples (no overflow).</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_discrete.py#L163">test_contains_small_dtype_no_overflow</see>.</remarks>
        [TestMethod]
        public void TestContainsSmallDtypeNoOverflow() {
            foreach ((long n, long start, DType dtype, long element) in new[] { (3L, 125L, np.int8, 125L), (3L, 125L, np.int8, 127L), (6L, 250L, np.uint8, 250L), (6L, 250L, np.uint8, 255L) }) {
                using var space = new Discrete(n, start: start, dtype: dtype);
                Assert.IsTrue(space.Contains(element), $"{space} contains {element}");
                using (NDArray typed = NDArray.Scalar(element).astype(dtype))
                    Assert.IsTrue(space.Contains(typed), $"{space} contains {dtype}({element})");
                for (int i = 0; i < 50; i++) {
                    using NDArray sample = space.Sample();
                    Assert.IsTrue(space.Contains(sample), $"{space} contains its sample");
                }
            }
        }

        /// <summary>A Python int outside the space dtype's range is not a member, and asking does not throw.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_discrete.py#L186">test_contains_out_of_dtype_range_returns_false</see>.</remarks>
        [TestMethod]
        public void TestContainsOutOfDtypeRangeReturnsFalse() {
            var cases = new (long N, long Start, DType DType, object Element)[] {
                (5, 0, np.int64, BigInteger.Pow(10, 20)),
                (5, 0, np.int64, -BigInteger.Pow(10, 20)),
                (5, 0, np.int64, BigInteger.Pow(2, 63)),
                (5, 0, np.int64, -BigInteger.Pow(2, 63) - 1),
                (3, 125, np.int8, -129L),
                (3, 125, np.int8, 200L),
                (6, 250, np.uint8, -1L),
                (6, 250, np.uint8, 300L),
            };
            foreach ((long n, long start, DType dtype, object element) in cases) {
                using var space = new Discrete(n, start: start, dtype: dtype);
                Assert.IsFalse(space.Contains(element), $"{space} contains {element}");
            }
        }

        /// <summary>A dtype that is not an integer dtype is refused.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_discrete.py#L203">test_dtype_error</see>.</remarks>
        [TestMethod]
        public void TestDtypeError() {
            foreach (DType dtype in new[] { np.str_, np.float32, np.complex128 }) {
                var error = Assert.ThrowsExactly<TypeError>(() => new Discrete(4, dtype: dtype), $"{dtype}");
                StringAssert.Contains(error.Message, "Invalid Discrete dtype", $"{dtype}");
            }
        }

        /// <summary>A sample's value, disposing it.</summary>
        /// <param name="sample">The 0-d sample.</param>
        /// <returns>Its value.</returns>
        private static long Value(NDArray sample) {
            using (sample)
                return Convert.ToInt64(sample.GetAtIndex(0));
        }

        /// <summary>An int8 vector (a mask).</summary>
        /// <param name="values">The values.</param>
        /// <returns>A new array.</returns>
        private static NDArray Int8(params sbyte[] values) => np.array(values);

        /// <summary>A float64 vector (a probability mask).</summary>
        /// <param name="values">The values.</param>
        /// <returns>A new array.</returns>
        private static NDArray Float64(params double[] values) => np.array(values);
    }
}
