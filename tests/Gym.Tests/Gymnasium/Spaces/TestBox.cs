#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Gym.Spaces;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Spaces {
    /// <summary>
    ///     Gymnasium's <c>Box</c> tests (<c>tests/spaces/test_box.py</c>): dtype and shape validation, out-of-range and NaN
    ///     bounds, the bounds every dtype accepts, reaching a dtype's edge values, membership by dtype, infinite bounds,
    ///     equality, and the refused masks.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Python-only parametrizations have no C# counterpart and are left out: a <c>None</c> dtype (C#'s <see langword="null"/>
    ///     is the float32 default), dtypes that are not dtypes (<c>0</c>, <c>"unknown"</c>, an array: NumSharp's
    ///     <c>np.dtype</c> refuses them before a box is built), <c>np.complex64</c> and <c>object</c> (no NumSharp dtype), and
    ///     shapes of the wrong type (<c>1</c>, <c>(None,)</c>: <see cref="Shape"/> is typed). <c>test_legacy_state_pickling</c>
    ///     tests Python's pickle protocol, which Gym.NET has no equivalent of.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_box.py">tests/spaces/test_box.py</see>.</para>
    /// </remarks>
    [TestClass]
    public class TestBox {
        /// <summary>A dtype that is neither integer, floating nor bool is refused with Gymnasium's message.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_box.py#L46">test_dtype_errors</see>.</remarks>
        [TestMethod]
        public void TestDtypeErrors() {
            var complexError = Assert.ThrowsExactly<ValueError>(() => new Box(0, 1, dtype: np.complex128));
            Assert.AreEqual("Invalid Box dtype (complex128), must be an integer, floating, or bool dtype", complexError.Message);
            var stringError = Assert.ThrowsExactly<ValueError>(() => new Box(0, 1, dtype: np.str_));
            StringAssert.Contains(stringError.Message, "must be an integer, floating, or bool dtype");
        }

        /// <summary>The shape comes from the given shape, the array bounds, or <c>(1,)</c> for numbers (NumPy scalars included).</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_box.py#L69">test_shape_inference</see>.</remarks>
        [TestMethod]
        public void TestShapeInference() {
            var cases = new (Func<Box> Create, long[] Expected)[] {
                (() => Arrays(np.zeros(new Shape(2L)), np.ones(new Shape(2L)), np.float64), new long[] { 2 }),
                (() => Arrays(np.zeros(new Shape(2L, 1L)), np.ones(new Shape(2L, 1L)), np.float64), new long[] { 2, 1 }),
                (() => new Box(0, 1, new Shape(5L, 2L)), new long[] { 5, 2 }),
                (() => new Box(0, 1), new long[] { 1 }),
                (() => new Box(0.0, 1.0), new long[] { 1 }),
                (() => Scalars(NDArray.Scalar(0.0), NDArray.Scalar(1.0)), new long[] { 1 }),
                (() => new Box(0.0, 1), new long[] { 1 }),
                (() => Scalars(NDArray.Scalar(0L), NDArray.Scalar(1)), new long[] { 1 }),
                (() => HighArray(0, np.ones(new Shape(3L)), np.float64), new long[] { 3 }),
                (() => LowArray(np.zeros(new Shape(3L)), 1.0, np.float64), new long[] { 3 }),
            };
            foreach ((Func<Box> create, long[] expected) in cases) {
                using Box box = create();
                CollectionAssert.AreEqual(expected, box.Shape.dimensions, $"{box}");
                using NDArray sample = box.Sample();
                CollectionAssert.AreEqual(expected, sample.Shape.dimensions, $"{box}.sample()");
            }
        }

        /// <summary>Bounds whose shapes disagree with each other or with the given shape are refused with Gymnasium's messages.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_box.py#L136">test_shape_errors</see>.</remarks>
        [TestMethod]
        public void TestShapeErrors() {
            using NDArray zeros3 = np.zeros(new Shape(3L));
            using NDArray ones2 = np.ones(new Shape(2L));
            using NDArray zeros2 = np.zeros(new Shape(2L));
            Assert.AreEqual("Box low.shape and high.shape don't match, low.shape=(3,), high.shape=(2,)",
                Assert.ThrowsExactly<ValueError>(() => new Box(zeros3, ones2, dtype: np.float64)).Message);
            Assert.AreEqual("Box low.shape doesn't match provided shape, low.shape=(2,), shape=(3,)",
                Assert.ThrowsExactly<ValueError>(() => new Box(zeros2, ones2, new Shape(3L), np.float64)).Message);
            Assert.AreEqual("Box low.shape doesn't match provided shape, low.shape=(2,), shape=(3,)",
                Assert.ThrowsExactly<ValueError>(() => new Box(zeros2, 1, new Shape(3L), np.float64)).Message);
            Assert.AreEqual("Box high.shape doesn't match provided shape, high.shape=(2,), shape=(3,)",
                Assert.ThrowsExactly<ValueError>(() => new Box(0, ones2, new Shape(3L), np.float64)).Message);
        }

        /// <summary>A bound outside the dtype's range is refused.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_box.py#L186">test_out_of_bounds_error</see>.</remarks>
        [TestMethod]
        public void TestOutOfBoundsError() {
            var cases = new (Func<Box> Create, string Label)[] {
                (() => new Box(0, 65505.0, dtype: np.float16), "(0, 65505.0, float16)"),
                (() => new Box(-65505.0, 0, dtype: np.float16), "(-65505.0, 0, float16)"),
                (() => new Box(0, 32768, dtype: np.int16), "(0, 32768, int16)"),
                (() => new Box(-32769, 0, dtype: np.int16), "(-32769, 0, int16)"),
                (() => new Box(-1, 100, dtype: np.uint8), "(-1, 100, uint8)"),
                (() => new Box(0, 300, dtype: np.uint8), "(0, 300, uint8)"),
                (() => new Box(-1, 1, dtype: np.@bool), "(-1, 1, bool)"),
                (() => new Box(0, 2, dtype: np.@bool), "(0, 2, bool)"),
                (() => Arrays(np.array(new long[] { -1, 0 }), np.array(new long[] { 0, 100 }), np.uint8), "([-1, 0], [0, 100], uint8)"),
                (() => Arrays(Column(-1, 0), Column(0, 100), np.uint8), "([[-1], [0]], [[0], [100]], uint8)"),
                (() => Arrays(np.array(new long[] { 0, 0 }), np.array(new long[] { 0, 300 }), np.uint8), "([0, 0], [0, 300], uint8)"),
                (() => Arrays(Column(0, 0), Column(0, 300), np.uint8), "([[0], [0]], [[0], [300]], uint8)"),
            };
            foreach ((Func<Box> create, string label) in cases) {
                var error = Assert.ThrowsExactly<ValueError>(() => create().Dispose(), label);
                StringAssert.Contains(error.Message, "is out of bounds of the dtype range,", label);
            }
        }

        /// <summary>A NaN bound is refused, and so is an infinite bound of an unsigned or bool dtype.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_box.py#L233">test_invalid_low_high</see>.</remarks>
        [TestMethod]
        public void TestInvalidLowHigh() {
            var cases = new (Func<Box> Create, DType DType, string Label)[] {
                (() => new Box(double.NaN, 0, dtype: np.float32), np.float32, "(nan, 0)"),
                (() => new Box(0, double.NaN, dtype: np.float32), np.float32, "(0, nan)"),
                (() => Arrays(np.array(new[] { 0.0, double.NaN }), np.ones(new Shape(2L)), np.float32), np.float32, "([0, nan], ones(2))"),
                (() => new Box(double.NaN, 0, dtype: np.int32), np.int32, "(nan, 0)"),
                (() => new Box(0, double.NaN, dtype: np.int32), np.int32, "(0, nan)"),
                (() => Arrays(np.array(new[] { 0.0, double.NaN }), np.ones(new Shape(2L)), np.int32), np.int32, "([0, nan], ones(2))"),
                (() => new Box(double.NegativeInfinity, 1, dtype: np.uint8), np.uint8, "(-inf, 1)"),
                (() => LowArray(np.array(new[] { double.NegativeInfinity, 0 }), 1, np.uint8), np.uint8, "([-inf, 0], 1)"),
                (() => new Box(0, double.PositiveInfinity, dtype: np.uint8), np.uint8, "(0, inf)"),
                (() => HighArray(0, np.array(new[] { 1, double.PositiveInfinity }), np.uint8), np.uint8, "(0, [1, inf])"),
                (() => new Box(double.NegativeInfinity, 1, dtype: np.@bool), np.@bool, "(-inf, 1)"),
                (() => new Box(0, double.PositiveInfinity, dtype: np.@bool), np.@bool, "(0, inf)"),
            };
            foreach ((Func<Box> create, DType dtype, string label) in cases) {
                var error = Assert.ThrowsExactly<ValueError>(() => create().Dispose(), label);
                string expected = dtype == np.uint8 || dtype == np.@bool ? "Box unsigned int dtype don't support" : "value can be equal to `np.nan`,";
                StringAssert.Contains(error.Message, expected, label);
            }
        }

        /// <summary>
        ///     Bounds every dtype accepts build a box of that dtype whose samples are of that dtype and members; the only
        ///     warning allowed is the precision one.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_box.py#L263">test_valid_low_high</see>.</remarks>
        [TestMethod]
        public void TestValidLowHigh() {
            var cases = new (Func<Box> Create, DType DType)[] {
                // floats
                (() => new Box(0, 1, dtype: np.float64), np.float64),
                (() => new Box(0, 1, dtype: np.float32), np.float32),
                (() => new Box(0, 1, dtype: np.float16), np.float16),
                (() => Arrays(np.zeros(new Shape(2L)), np.ones(new Shape(2L)), np.float32), np.float32),
                (() => LowArray(np.zeros(new Shape(2L)), 1, np.float32), np.float32),
                (() => new Box(double.NegativeInfinity, 1, dtype: np.float32), np.float32),
                (() => LowArray(np.array(new[] { double.NegativeInfinity, 0 }), 1, np.float32), np.float32),
                (() => new Box(0, double.PositiveInfinity, dtype: np.float32), np.float32),
                (() => HighArray(0, np.array(new[] { double.PositiveInfinity, 1 }), np.float32), np.float32),
                (() => new Box(double.NegativeInfinity, double.PositiveInfinity, dtype: np.float32), np.float32),
                (() => Arrays(np.full(new Shape(2L), double.NegativeInfinity), np.full(new Shape(2L), double.PositiveInfinity), np.float32), np.float32),
                // signed ints
                (() => new Box(0, 1, dtype: np.int64), np.int64),
                (() => new Box(0, 1, dtype: np.int32), np.int32),
                (() => new Box(0, 1, dtype: np.int16), np.int16),
                (() => new Box(0, 1, dtype: np.int8), np.int8),
                (() => Arrays(np.zeros(new Shape(2L)), np.ones(new Shape(2L)), np.int32), np.int32),
                (() => LowArray(np.zeros(new Shape(2L)), 1, np.int32), np.int32),
                (() => new Box(double.NegativeInfinity, 1, dtype: np.int32), np.int32),
                (() => LowArray(np.array(new[] { double.NegativeInfinity, 0 }), 1, np.int32), np.int32),
                (() => new Box(0, double.PositiveInfinity, dtype: np.int32), np.int32),
                (() => HighArray(0, np.array(new[] { double.PositiveInfinity, 1 }), np.int32), np.int32),
                // unsigned ints
                (() => new Box(0, 1, dtype: np.uint64), np.uint64),
                (() => new Box(0, 1, dtype: np.uint32), np.uint32),
                (() => new Box(0, 1, dtype: np.uint16), np.uint16),
                (() => new Box(0, 1, dtype: np.uint8), np.uint8),
                // boolean
                (() => new Box(0, 1, dtype: np.@bool), np.@bool),
            };
            foreach ((Func<Box> create, DType dtype) in cases) {
                using WarningCatcher caught = Warnings.Catch();
                using Box space = create();
                Assert.AreEqual(dtype, space.DType, $"{space}");
                Assert.AreEqual(dtype, space.Low.dtype, $"{space}.low");
                Assert.AreEqual(dtype, space.High.dtype, $"{space}.high");
                space.Seed(0);
                using NDArray sample = space.Sample();
                Assert.AreEqual(dtype, sample.dtype, $"{space}.sample()");
                Assert.IsTrue(space.Contains(sample), $"{space} contains its sample");
                foreach (WarningMessage warning in caught.Records)
                    StringAssert.Contains(warning.Message, "precision lowered by casting to float32", $"{space}");
            }
        }

        /// <summary>A bounded integer box can sample every value up to its dtype's limits.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_box.py#L306">test_sample_dtype_edge_reachability</see>.</remarks>
        [TestMethod]
        public void TestSampleDtypeEdgeReachability() {
            foreach ((DType dtype, long min, long max) in new[] { (np.int8, (long)sbyte.MinValue, (long)sbyte.MaxValue), (np.int16, short.MinValue, short.MaxValue), (np.int32, int.MinValue, int.MaxValue), (np.uint8, 0L, 255L) }) {
                foreach ((long low, long high) in new[] { (max - 2, max), (min, min + 2) }) {
                    using var space = new Box(low, high, dtype: dtype, seed: 0);
                    var observed = new HashSet<long>();
                    for (int i = 0; i < 1000; i++) {
                        using NDArray sample = space.Sample();
                        for (long j = 0; j < sample.size; j++)
                            observed.Add(Convert.ToInt64(sample.GetAtIndex(j)));
                    }

                    CollectionAssert.AreEquivalent(Enumerable.Range(0, (int)(high - low + 1)).Select(i => low + i).ToArray(), observed.ToArray(), $"{dtype} [{low}, {high}]");
                }
            }
        }

        /// <summary>A float32 box contains float32 and float16 points but not float64 ones (they cannot be cast safely).</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_box.py#L327">test_contains_dtype</see>.</remarks>
        [TestMethod]
        public void TestContainsDtype() {
            using var space = new Box(0, 1, Shape.Scalar, np.float32);
            using NDArray single = NDArray.Scalar(0.5f);
            using NDArray half = NDArray.Scalar((Half)0.5);
            using NDArray dbl = NDArray.Scalar(0.5);
            Assert.IsTrue(space.Contains(single));
            Assert.IsTrue(space.Contains(half));
            Assert.IsFalse(space.Contains(dbl));
        }

        /// <summary>A list holding a value outside the dtype's range is not a member, and asking does not throw.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_box.py#L354">test_contains_out_of_dtype_range_returns_false</see>.</remarks>
        [TestMethod]
        public void TestContainsOutOfDtypeRangeReturnsFalse() {
            foreach ((DType dtype, object element) in new (DType, object)[] { (np.int8, 300L), (np.int8, -300L), (np.uint8, -1L), (np.uint8, 300L), (np.int64, BigInteger.Pow(10, 20)) }) {
                using var space = new Box(0, 10, new Shape(2L), dtype);
                Assert.IsFalse(space.Contains(new List<object?> { element, 0L }), $"{dtype}: [{element}, 0]");
            }
        }

        /// <summary>
        ///     Boxes with infinite bounds: integer boxes store their dtype's limits (bounded), float boxes stay unbounded; samples
        ///     are members and on the bounds' side of zero; an unknown manner is refused.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_box.py#L368">test_infinite_space</see>.</remarks>
        [TestMethod]
        public void TestInfiniteSpace() {
            var bounds = new (Func<DType, Box> Create, string Label)[] {
                (dtype => new Box(0, double.PositiveInfinity, new Shape(2L), dtype), "(0, inf, (2,))"),
                (dtype => new Box(double.NegativeInfinity, 0, new Shape(2L), dtype), "(-inf, 0, (2,))"),
                (dtype => new Box(double.NegativeInfinity, double.PositiveInfinity, new Shape(2L), dtype), "(-inf, inf, (2,))"),
                (dtype => new Box(0, double.PositiveInfinity, new Shape(2L, 3L), dtype), "(0, inf, (2, 3))"),
                (dtype => new Box(double.NegativeInfinity, 0, new Shape(2L, 3L), dtype), "(-inf, 0, (2, 3))"),
                (dtype => new Box(double.NegativeInfinity, double.PositiveInfinity, new Shape(2L, 3L), dtype), "(-inf, inf, (2, 3))"),
                (dtype => Arrays(np.array(new[] { double.NegativeInfinity, 0 }), np.array(new[] { 0.0, double.PositiveInfinity }), dtype), "([-inf, 0], [0, inf])"),
            };
            foreach (DType dtype in new[] { np.int32, np.int64, np.float32, np.float64 }) {
                foreach ((Func<DType, Box> create, string label) in bounds) {
                    if (dtype == np.int64 && label.StartsWith("([", StringComparison.Ordinal)) {
                        // Fresh float64 bounds holding +inf cannot become int64: Gymnasium stores int64's maximum as the float
                        // 2**63, which the cast wraps to the minimum, and raises this very error. Its own test passes the case
                        // only because the earlier int32 case wrote int32's limits into the shared parametrize arrays.
                        StringAssert.StartsWith(Assert.ThrowsExactly<ValueError>(() => create(dtype).Dispose(), label).Message,
                            "Box all low values must be less than or equal to high (some values break this)");
                        continue;
                    }

                    Box space;
                    using (WarningCatcher caught = Warnings.Catch()) {
                        space = create(dtype);
                        foreach (WarningMessage warning in caught.Records)
                            StringAssert.Contains(warning.Message, "precision lowered by casting to float32", label);
                    }

                    using (space) {
                        string context = $"{label} {dtype}";
                        using (NDArray lessThan = space.Low < space.High)
                            Assert.IsTrue(np.all(lessThan), context);

                        // A bound is unbounded exactly where it was infinite (non-zero here), integer boxes included. Gymnasium's
                        // own lines compare a Python bool with a NumPy bool by identity ("is not"), which always passes, and its
                        // "both" line reads high twice; this asserts the relation they intend, which Gymnasium's values satisfy.
                        using NDArray highNonZero = space.High != 0;
                        using NDArray lowNonZero = space.Low != 0;
                        bool anyHigh = np.any(highNonZero), anyLow = np.any(lowNonZero);
                        Assert.AreEqual(!anyHigh, space.IsBounded("above"), context);
                        Assert.AreEqual(!anyLow, space.IsBounded("below"), context);
                        Assert.AreEqual(!(anyHigh || anyLow), space.IsBounded("both"), context);

                        Assert.AreEqual(space.DType, space.High.dtype, context);
                        Assert.AreEqual(space.DType, space.Low.dtype, context);
                        var error = Assert.ThrowsExactly<ValueError>(() => space.IsBounded("test"), context);
                        StringAssert.Contains(error.Message, "manner is not in {'below', 'above', 'both'}, actual value:", context);

                        space.Seed(0);
                        using NDArray sample = space.Sample();
                        Assert.IsTrue(space.Contains(sample), context);
                        using NDArray sampleSign = np.sign(sample);
                        using NDArray highSign = np.sign(space.High);
                        using NDArray lowSign = np.sign(space.Low);
                        using NDArray belowHigh = sampleSign <= highSign;
                        using NDArray aboveLow = lowSign <= sampleSign;
                        Assert.IsTrue(np.all(belowHigh), context);
                        Assert.IsTrue(np.all(aboveLow), context);
                    }
                }
            }
        }

        /// <summary>Boxes are equal for the same shape, dtype and bounds, whatever form the bounds were given in.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_box.py#L417">test_equality</see>.</remarks>
        [TestMethod]
        public void TestEquality() {
            using var spaceA = new Box(0, 1, new Shape(2L), np.float32);
            using (var spaceB = new Box(0, 1, new Shape(2L), np.float32))
                Assert.AreEqual(spaceA, spaceB);
            using (NDArray ones = np.array(new long[] { 1, 1 }))
            using (var spaceB = new Box(0, ones, new Shape(2L), np.float32))
                Assert.AreEqual(spaceA, spaceB);
            using (var spaceB = new Box(-1, 1, new Shape(2L), np.float32))
                Assert.AreNotEqual(spaceA, spaceB);
            using (NDArray highs = np.array(new long[] { 1, 2 }))
            using (var spaceB = new Box(0, highs, new Shape(2L), np.float32))
                Assert.AreNotEqual(spaceA, spaceB);
            using (var spaceB = new Box(0, 1, new Shape(3L), np.float32))
                Assert.AreNotEqual(spaceA, spaceB);
            using (var spaceB = new Box(0, 1, new Shape(2L), np.int32))
                Assert.AreNotEqual(spaceA, spaceB);
        }

        /// <summary>A box takes no mask: Gymnasium's error.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_box.py#L462">test_sample_mask</see>.</remarks>
        [TestMethod]
        public void TestSampleMask() {
            using var space = new Box(0, 1);
            using NDArray mask = np.array(new sbyte[] { 0, 1, 0 });
            var error = Assert.ThrowsExactly<Gym.Exceptions.Error>(() => space.Sample(mask));
            StringAssert.StartsWith(error.Message, "Box.sample cannot be provided a mask, actual value: ");
        }

        /// <summary>A box takes no probability mask: Gymnasium's error.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_box.py#L472">test_sample_probability_mask</see>.</remarks>
        [TestMethod]
        public void TestSampleProbabilityMask() {
            using var space = new Box(0, 1);
            using NDArray probability = np.array(new[] { 0.0, 1.0, 0.0 });
            var error = Assert.ThrowsExactly<Gym.Exceptions.Error>(() => space.Sample(probability: probability));
            StringAssert.StartsWith(error.Message, "Box.sample cannot be provided a probability mask, actual value: ");
        }

        /// <summary>A box from two array bounds, which are disposed after (the box copies them).</summary>
        /// <param name="low">The lower bounds; disposed here.</param>
        /// <param name="high">The upper bounds; disposed here.</param>
        /// <param name="dtype">The dtype.</param>
        /// <returns>The box.</returns>
        private static Box Arrays(NDArray low, NDArray high, DType dtype) {
            using (low)
            using (high)
                return new Box(low, high, dtype: dtype);
        }

        /// <summary>A box from NumPy-scalar bounds (0-d arrays), which are disposed after.</summary>
        /// <param name="low">The lower bound; disposed here.</param>
        /// <param name="high">The upper bound; disposed here.</param>
        /// <returns>The float32 box.</returns>
        private static Box Scalars(NDArray low, NDArray high) {
            using (low)
            using (high)
                return new Box(low, high);
        }

        /// <summary>A box from a scalar lower bound and array upper bounds.</summary>
        /// <param name="low">The lower bound.</param>
        /// <param name="high">The upper bounds; disposed here.</param>
        /// <param name="dtype">The dtype.</param>
        /// <returns>The box.</returns>
        private static Box HighArray(double low, NDArray high, DType dtype) {
            using (high)
                return new Box(low, high, dtype: dtype);
        }

        /// <summary>A box from array lower bounds and a scalar upper bound.</summary>
        /// <param name="low">The lower bounds; disposed here.</param>
        /// <param name="high">The upper bound.</param>
        /// <param name="dtype">The dtype.</param>
        /// <returns>The box.</returns>
        private static Box LowArray(NDArray low, double high, DType dtype) {
            using (low)
                return new Box(low, high, dtype: dtype);
        }

        /// <summary>A two-row int64 column, <c>np.array([[a], [b]])</c>.</summary>
        /// <param name="a">The first row.</param>
        /// <param name="b">The second row.</param>
        /// <returns>A new (2, 1) array.</returns>
        private static NDArray Column(long a, long b) => np.array(new long[,] { { a }, { b } });
    }
}
