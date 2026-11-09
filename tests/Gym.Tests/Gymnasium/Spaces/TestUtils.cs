#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Envs;
using Gym.Spaces;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Tuple = Gym.Spaces.Tuple;

namespace Gym.Tests.Gymnasium.Spaces {
    /// <summary>
    ///     Gymnasium's <c>spaces.utils</c> tests (<c>tests/spaces/test_utils.py</c>): the flattened size of every space,
    ///     flattened spaces and samples, flat samples in the flat space, the flatten round trip, unflattening discrete
    ///     samples (dtypes and errors), dtype-and-shape equivalence, and <c>OneOf</c> indexes wider than the subspaces' dtype.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <c>test_all_space_pairs_for_is_space_dtype_shape_equiv</c> writes samples through
    ///     <c>gymnasium.vector.utils</c>' shared memory, which arrives with the vector envs (P6); it is ported there.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_utils.py">tests/spaces/test_utils.py</see>.</para>
    /// </remarks>
    [TestClass]
    public class TestUtils {
        /// <summary><c>TESTING_SPACES_EXPECTED_FLATDIMS</c>: the flattened size of each space of <see cref="TestingSpaces.All"/>, <see langword="null"/> where it has none.</summary>
        private static readonly long?[] ExpectedFlatdims = {
            // Discrete
            3, 3, 4,
            // Box
            1, 4, 2, 2, 2, 12, 3, 4,
            // Multi-discrete
            4, 10, 4, 10, 5, 5,
            // Multi-binary
            8, 6,
            // Text
            6, 6, 6,
            // Tuple
            9, 7, 10, 6, 5, null,
            // Dict
            7, 8, 17, null,
            // Graph
            null, null, null, null, null, null, null, null, null, null, null, null,
            // Sequence
            null, null, null, null, null,
            // OneOf
            4, 5, 4, 6, 5, 6, 4, 6, 5, 3,
        };

        /// <summary>Every space of the catalog, by position.</summary>
        public static IEnumerable<object[]> Spaces => TestingSpaces.AllIndexes;

        /// <summary>The flattened size is the expected one, or a <c>ValueError</c> for a space that cannot be flattened.</summary>
        /// <param name="index">The space's position in <see cref="TestingSpaces.All"/>.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_utils.py#L106">test_flatdim</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Spaces))]
        public void TestFlatdim(int index) {
            Assert.AreEqual(TestingSpaces.All.Length, ExpectedFlatdims.Length, "one expected size per space");
            using Space space = TestingSpaces.All[index]();
            if (space.IsNpFlattenable)
                Assert.AreEqual(ExpectedFlatdims[index], SpaceUtils.Flatdim(space), $"{space}");
            else
                Assert.ThrowsExactly<ValueError>(() => SpaceUtils.Flatdim(space), $"{space}");
        }

        /// <summary>A flattenable space flattens to a box of its flattened size; a graph to a graph of flat boxes; others keep their structure.</summary>
        /// <param name="index">The space's position in <see cref="TestingSpaces.All"/>.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_utils.py#L120">test_flatten_space</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Spaces))]
        public void TestFlattenSpace(int index) {
            using Space space = TestingSpaces.All[index]();
            using Space flatSpace = SpaceUtils.FlattenSpace(space);
            if (space.IsNpFlattenable) {
                Assert.IsInstanceOfType<Box>(flatSpace, $"{space}");
                CollectionAssert.AreEqual(new[] { SpaceUtils.Flatdim(space) }, flatSpace.Shape!.Value.dimensions, $"{space}");
            } else if (flatSpace is Graph flatGraph) {
                var graph = (Graph)space;
                CollectionAssert.AreEqual(new[] { SpaceUtils.Flatdim(graph.NodeSpace) }, flatGraph.NodeSpace.Shape!.Value.dimensions, $"{space}");
                if (flatGraph.EdgeSpace is not null)
                    CollectionAssert.AreEqual(new[] { SpaceUtils.Flatdim(graph.EdgeSpace!) }, flatGraph.EdgeSpace.Shape!.Value.dimensions, $"{space}");
            } else {
                Assert.IsTrue(space is Tuple or Dict or Sequence, $"{space}");
            }
        }

        /// <summary>A flattened sample of a flattenable space is an array of the flattened size and the flat space's dtype.</summary>
        /// <param name="index">The space's position in <see cref="TestingSpaces.All"/>.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_utils.py#L148">test_flatten</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Spaces))]
        public void TestFlatten(int index) {
            using Space space = TestingSpaces.All[index]();
            object? sample = space.Sample();
            object? flattened = SpaceUtils.Flatten(space, sample);
            try {
                if (space.IsNpFlattenable) {
                    var array = (NDArray)flattened!;
                    CollectionAssert.AreEqual(new[] { SpaceUtils.Flatdim(space) }, array.Shape.dimensions, $"{space}");
                    using Space flatSpace = SpaceUtils.FlattenSpace(space);
                    Assert.AreEqual(flatSpace.DType, array.dtype, $"{space}");
                } else {
                    Assert.IsTrue(space is Sequence || flattened is object?[] or Info or GraphInstance, $"{space}: {flattened?.GetType()}");
                }
            } finally {
                Release(sample);
                Release(flattened);
            }
        }

        /// <summary>Ten flattened samples are members of the flattened space, in its dtype when it is a box.</summary>
        /// <param name="index">The space's position in <see cref="TestingSpaces.All"/>.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_utils.py#L166">test_flat_space_contains_flat_points</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Spaces))]
        public void TestFlatSpaceContainsFlatPoints(int index) {
            using Space space = TestingSpaces.All[index]();
            using Space flatSpace = SpaceUtils.FlattenSpace(space);
            for (int i = 0; i < 10; i++) {
                object? sample = space.Sample();
                object? flat = SpaceUtils.Flatten(space, sample);
                try {
                    Assert.IsTrue(flatSpace.Contains(flat), $"{space}: flat sample {i}");
                    if (flatSpace is Box)
                        Assert.AreEqual(flatSpace.DType, ((NDArray)flat!).dtype, $"{space}");
                } finally {
                    Release(sample);
                    Release(flat);
                }
            }
        }

        /// <summary>Unflattening ten flattened samples gives back equivalent samples.</summary>
        /// <param name="index">The space's position in <see cref="TestingSpaces.All"/>.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_utils.py#L178">test_flatten_roundtripping</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Spaces))]
        public void TestFlattenRoundtripping(int index) {
            using Space space = TestingSpaces.All[index]();
            for (int i = 0; i < 10; i++) {
                object? sample = space.Sample();
                object? flat = SpaceUtils.Flatten(space, sample);
                object? roundtripped = SpaceUtils.Unflatten(space, flat);
                try {
                    Assert.IsTrue(EnvChecker.DataEquivalence(sample, roundtripped), $"{space}: {PyBuiltins.Repr(sample)} vs {PyBuiltins.Repr(roundtripped)}");
                } finally {
                    Release(sample);
                    Release(flat);
                    Release(roundtripped);
                }
            }
        }

        /// <summary>A discrete one-hot unflattens to <c>1</c> in the space's dtype.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_utils.py#L192">test_unflatten_discrete_with_dtype</see>.</remarks>
        [TestMethod]
        public void TestUnflattenDiscreteWithDtype() {
            foreach (DType dtype in new[] { np.int64, np.int16, np.int32 }) {
                using var space = new Discrete(3, dtype: dtype);
                using NDArray flattened = np.array(new long[] { 0, 1, 0 }).astype(dtype);
                using var unflattened = (NDArray)SpaceUtils.Unflatten(space, flattened)!;
                Assert.AreEqual(1L, Convert.ToInt64(unflattened.GetAtIndex(0)), $"{dtype}");
                Assert.AreEqual(space.DType, unflattened.dtype, $"{dtype}");
            }
        }

        /// <summary>A discrete vector with no one-hot entry cannot be unflattened.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_utils.py#L206">test_unflatten_discrete_error</see>.</remarks>
        [TestMethod]
        public void TestUnflattenDiscreteError() {
            using var space = new Discrete(1);
            using NDArray value = np.array(new long[] { 0 });
            Assert.ThrowsExactly<ValueError>(() => SpaceUtils.Unflatten(space, value));
        }

        /// <summary>A multi-discrete vector with no one-hot entry cannot be unflattened.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_utils.py#L212">test_unflatten_multidiscrete_error</see>.</remarks>
        [TestMethod]
        public void TestUnflattenMultidiscreteError() {
            using var space = new MultiDiscrete(new long[] { 1, 1 });
            using NDArray value = np.array(new long[] { 0, 0 });
            Assert.ThrowsExactly<ValueError>(() => SpaceUtils.Unflatten(space, value));
        }

        /// <summary>Every space is dtype-and-shape equivalent to itself.</summary>
        /// <param name="index">The space's position in <see cref="TestingSpaces.All"/>.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_utils.py#L218">test_is_space_dtype_shape_equiv</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Spaces))]
        public void TestIsSpaceDtypeShapeEquiv(int index) {
            using Space space = TestingSpaces.All[index]();
            Assert.IsTrue(SpaceUtils.IsSpaceDtypeShapeEquiv(space, space), $"{space}");
        }

        /// <summary>Tuples of different lengths are not equivalent, whichever comes first; equal ones are.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_utils.py#L223">test_tuple_dtype_shape_equivalence</see>.</remarks>
        [TestMethod]
        public void TestTupleDtypeShapeEquivalence() {
            using var shortTuple = new Tuple(new Discrete(2));
            using var longTuple = new Tuple(new Discrete(2), new Discrete(3));
            using var equal = new Tuple(new Discrete(2));
            Assert.IsFalse(SpaceUtils.IsSpaceDtypeShapeEquiv(shortTuple, longTuple));
            Assert.IsFalse(SpaceUtils.IsSpaceDtypeShapeEquiv(longTuple, shortTuple));
            Assert.IsTrue(SpaceUtils.IsSpaceDtypeShapeEquiv(shortTuple, equal));
        }

        /// <summary>
        ///     A <c>OneOf</c> with more subspaces than their dtype can index (3 bool boxes, 130 int8 multi-binaries, 300 uint8
        ///     boxes) flattens in a dtype wide enough for the index, the flat sample is a member, and the index comes back.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_utils.py#L279">test_oneof_index_dtype</see>.</remarks>
        [TestMethod]
        public void TestOneofIndexDtype() {
            var spaces = new Func<OneOf>[] {
                () => new OneOf(Enumerable.Range(0, 3).Select(_ => (Space)new Box(0, 1, new Shape(2L), np.@bool)).ToArray()),
                () => new OneOf(Enumerable.Range(0, 130).Select(_ => (Space)new MultiBinary(2)).ToArray()),
                () => new OneOf(Enumerable.Range(0, 300).Select(_ => (Space)new Box(0, 255, new Shape(2L), np.uint8)).ToArray()),
            };
            foreach (Func<OneOf> create in spaces) {
                using OneOf space = create();
                using Space flatSpace = SpaceUtils.FlattenSpace(space);
                space.Seed(123);
                foreach (int idx in new[] { 0, space.Count - 1 }) {
                    object? value = space[idx].Sample();
                    using var flatSample = (NDArray)SpaceUtils.Flatten(space, new object?[] { (long)idx, value })!;
                    var unflattened = (object?[])SpaceUtils.Unflatten(space, flatSample)!;
                    try {
                        Assert.AreEqual(flatSpace.DType, flatSample.dtype, $"{space.Count} subspaces, index {idx}");
                        Assert.IsTrue(flatSpace.Contains(flatSample), $"{space.Count} subspaces, index {idx}");
                        Assert.AreEqual((long)idx, Convert.ToInt64(((NDArray)unflattened[0]!).GetAtIndex(0)), $"{space.Count} subspaces");
                    } finally {
                        Release(value);
                        Release(unflattened);
                    }
                }
            }
        }

        /// <summary>Disposes the arrays inside a sample or a flattened value.</summary>
        /// <param name="value">The value.</param>
        private static void Release(object? value) {
            switch (value) {
                case IDisposable disposable:
                    disposable.Dispose();
                    break;
                case string:
                    break;
                case Info info:
                    foreach (KeyValuePair<string, object?> pair in info)
                        Release(pair.Value);
                    break;
                case object?[] items:
                    foreach (object? item in items)
                        Release(item);
                    break;
            }
        }
    }
}
