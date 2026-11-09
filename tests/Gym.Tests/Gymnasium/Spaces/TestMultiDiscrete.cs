#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Spaces;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Spaces {
    /// <summary>
    ///     Gymnasium's <c>MultiDiscrete</c> tests (<c>tests/spaces/test_multidiscrete.py</c>): indexing and slicing into
    ///     <c>Discrete</c> and <c>MultiDiscrete</c> subspaces (with starts and dtypes), subspaces drawing what their parent
    ///     draws, the length and its warning, flattening without overflow, membership with starts and at a dtype's edge,
    ///     equality, and sampling with and without masks or probabilities.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Python's slices are NumSharp's index strings: <c>space[0:2, :]</c> is <c>space["0:2, :"]</c>.
    ///     <c>test_space_legacy_pickling</c> tests Python's pickle protocol, which Gym.NET has no equivalent of.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_multidiscrete.py">tests/spaces/test_multidiscrete.py</see>.</para>
    /// </remarks>
    [TestClass]
    public class TestMultiDiscrete {
        /// <summary>Indexing and slicing a 1-D and a 2-D space give the matching subspaces, and <c>[:]</c> is a new equal space.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_multidiscrete.py#L10">test_multidiscrete_as_tuple</see>.</remarks>
        [TestMethod]
        public void TestMultidiscreteAsTuple() {
            CheckSlices(dtype: null, withStart: false);
        }

        /// <summary>The same with starts, which the subspaces keep.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_multidiscrete.py#L34">test_multidiscrete_start_as_tuple</see>.</remarks>
        [TestMethod]
        public void TestMultidiscreteStartAsTuple() {
            CheckSlices(dtype: null, withStart: true);
        }

        /// <summary>The same with dtypes (int8 for the vector, uint32 for the matrix), which the subspaces keep.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_multidiscrete.py#L62">test_multidiscrete_dtype_as_tuple</see>.</remarks>
        [TestMethod]
        public void TestMultidiscreteDtypeAsTuple() {
            CheckSlices(dtype: "typed", withStart: false);
        }

        /// <summary>A subspace starts from its parent's generator state, so two subspaces taken in a row draw the same sample.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_multidiscrete.py#L86">test_multidiscrete_subspace_reproducibility</see>.</remarks>
        [TestMethod]
        public void TestMultidiscreteSubspaceReproducibility() {
            using (var space = new MultiDiscrete(new long[] { 100, 200, 300 })) {
                space.Seed();
                foreach (string index in new[] { "0", "0:1", "0:2", ":" })
                    Assert.IsTrue(SubspaceSamplesEqual(space, index, index), index);
                Assert.IsTrue(SubspaceVersusSpace(space, ":"));
            }

            using (MultiDiscrete space = TestingSpaces.MultiDiscreteOf(new long[,] { { 300, 400, 500 }, { 600, 700, 800 } })) {
                space.Seed();
                foreach (string index in new[] { "0, 1", "0", "0:1", "0:2, :", ":, 0:1", "0:2, 0:2", ":", ":, :" })
                    Assert.IsTrue(SubspaceSamplesEqual(space, index, index), index);
                Assert.IsTrue(SubspaceVersusSpace(space, ":, :"));
            }
        }

        /// <summary>The same with starts.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_multidiscrete.py#L113">test_multidiscrete_start_subspace_reproducibility</see>.</remarks>
        [TestMethod]
        public void TestMultidiscreteStartSubspaceReproducibility() {
            using (var space = new MultiDiscrete(new long[] { 100, 200, 300 }, start: new long[] { -50, -100, -150 })) {
                space.Seed();
                foreach (string index in new[] { "0", "0:1", "0:2", ":" })
                    Assert.IsTrue(SubspaceSamplesEqual(space, index, index), index);
                Assert.IsTrue(SubspaceVersusSpace(space, ":"));
            }

            using (MultiDiscrete space = TestingSpaces.MultiDiscreteOf(new long[,] { { 300, 400, 500 }, { 600, 700, 800 } }, new long[,] { { -150, -200, -250 }, { -300, -350, -400 } })) {
                space.Seed();
                foreach (string index in new[] { "0, 1", "0", "0:1", "0:2, :", ":, 0:1", "0:2, 0:2", ":", ":, :" })
                    Assert.IsTrue(SubspaceSamplesEqual(space, index, index), index);
                Assert.IsTrue(SubspaceVersusSpace(space, ":, :"));
            }
        }

        /// <summary><c>len</c> is the first axis' length, with Gymnasium's warning for a multi-dimensional space.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_multidiscrete.py#L143">test_multidiscrete_length</see>.</remarks>
        [TestMethod]
        public void TestMultidiscreteLength() {
            using (var space = new MultiDiscrete(new long[] { 3, 2, 4 }))
                Assert.AreEqual(3, space.Count);
            using (var space = new MultiDiscrete(new long[] { 3, 2, 4 }, start: new long[] { 10, 10, 10 }))
                Assert.AreEqual(3, space.Count);
            foreach (MultiDiscrete space in new[] { TestingSpaces.MultiDiscreteOf(new long[,] { { 2, 3 }, { 3, 2 } }), TestingSpaces.MultiDiscreteOf(new long[,] { { 2, 3 }, { 3, 2 } }, new long[,] { { 10, 20 }, { 30, 40 } }) }) {
                using (space) {
                    using WarningCatcher caught = Warnings.Catch();
                    Assert.AreEqual(2, space.Count);
                    Assert.IsTrue(caught.Records.Any(w => w.Category == WarningCategory.UserWarning && w.Message.Contains("Getting the length of a multi-dimensional MultiDiscrete space.")));
                }
            }
        }

        /// <summary>An int8 space with counts of 101 flattens and unflattens without overflowing.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_multidiscrete.py#L165">test_multidiscrete_integer_overflow</see>.</remarks>
        [TestMethod]
        public void TestMultidiscreteIntegerOverflow() {
            using var space = new MultiDiscrete(new long[] { 101, 101, 101, 101 }, np.int8);
            using NDArray x = space.Sample();
            using var y = (NDArray)SpaceUtils.Flatten(space, x)!;
            using var z = (NDArray)SpaceUtils.Unflatten(space, y)!;
            Assert.AreEqual(4, z.size);
            Assert.IsTrue(np.array_equal(x, z));
        }

        /// <summary>Membership with starts: the lower and upper edges per entry.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_multidiscrete.py#L176">test_multidiscrete_start_contains</see>.</remarks>
        [TestMethod]
        public void TestMultidiscreteStartContains() {
            using var space = new MultiDiscrete(new long[] { 3, 4, 5 }, start: new long[] { 10, 20, 30 });
            Assert.IsTrue(space.Contains(new List<object?> { 10L, 20L, 30L }));
            Assert.IsFalse(space.Contains(new List<object?> { 9L, 20L, 30L }));
            Assert.IsTrue(space.Contains(new List<object?> { 12L, 23L, 34L }));
            Assert.IsFalse(space.Contains(new List<object?> { 13L, 23L, 34L }));
        }

        /// <summary>The upper-bound check does not overflow a small dtype or a negative start, and a space contains its own samples.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_multidiscrete.py#L187">test_contains_no_overflow</see>.</remarks>
        [TestMethod]
        public void TestContainsNoOverflow() {
            var cases = new (long Nvec, long Start, DType DType, long Element, bool Expected)[] {
                (120, -100, np.int8, 19, true),
                (120, -100, np.int8, 20, false),
                (120, -100, np.int8, 50, false),
                (120, -100, np.int8, 127, false),
                (3, 125, np.int8, 127, true),
                (6, 250, np.uint8, 255, true),
                (5, -10, np.int64, -6, true),
                (5, -10, np.int64, long.MaxValue, false),
            };
            foreach ((long nvec, long start, DType dtype, long element, bool expected) in cases) {
                using var space = new MultiDiscrete(new[] { nvec }, dtype, start: new[] { start });
                using (NDArray candidate = np.array(new[] { element }).astype(dtype))
                    Assert.AreEqual(expected, space.Contains(candidate), $"{space} contains [{element}]");
                for (int i = 0; i < 50; i++) {
                    using NDArray sample = space.Sample();
                    Assert.IsTrue(space.Contains(sample), $"{space} contains its sample");
                }
            }
        }

        /// <summary>Spaces are equal for the same counts and starts.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_multidiscrete.py#L210">test_multidiscrete_equality</see>.</remarks>
        [TestMethod]
        public void TestMultidiscreteEquality() {
            using var spaceA = new MultiDiscrete(new long[] { 2, 3, 4 }, start: new long[] { 0, 0, 1 });
            using (var spaceB = new MultiDiscrete(new long[] { 2, 3, 4 }, start: new long[] { 0, 0, 1 }))
                Assert.AreEqual(spaceA, spaceB);
            using (var spaceB = new MultiDiscrete(new long[] { 2, 4, 3 }, start: new long[] { 0, 0, 1 }))
                Assert.AreNotEqual(spaceA, spaceB);
            using (var spaceB = new MultiDiscrete(new long[] { 2, 3, 4 }, start: new long[] { 1, 0, 1 }))
                Assert.AreNotEqual(spaceA, spaceB);
            using (var spaceB = new MultiDiscrete(new long[] { 2, 3, 4 }, start: new long[] { 0, 1, 1 }))
                Assert.AreNotEqual(spaceA, spaceB);
            using (var spaceB = new MultiDiscrete(new long[] { 2, 3, 4, 2 }, start: new long[] { 1, 0, 0, 0 }))
                Assert.AreNotEqual(spaceA, spaceB);
        }

        /// <summary>An entry with a single value always samples it.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_multidiscrete.py#L250">test_multidiscrete_sample_edge_cases</see>.</remarks>
        [TestMethod]
        public void TestMultidiscreteSampleEdgeCases() {
            using var space = new MultiDiscrete(new long[] { 5, 1, 3 });
            for (int i = 0; i < 1000; i++) {
                using NDArray sample = space.Sample();
                Assert.AreEqual(0L, Convert.ToInt64(sample.GetAtIndex(1)));
            }
        }

        /// <summary>Unmasked samples fall within the counts.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_multidiscrete.py#L260">test_multidiscrete_sample</see>.</remarks>
        [TestMethod]
        public void TestMultidiscreteSample() {
            using var space = new MultiDiscrete(new long[] { 5, 2, 3 });
            for (int i = 0; i < 1000; i++) {
                using NDArray sample = space.Sample();
                long[] values = sample.ToArray<long>();
                Assert.IsTrue(values[0] < 5 && values[1] < 2 && values[2] < 3);
            }
        }

        /// <summary>Masked samples are members and never take a masked-out value.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_multidiscrete.py#L273">test_multidiscrete_sample_with_mask</see>.</remarks>
        [TestMethod]
        public void TestMultidiscreteSampleWithMask() {
            using var space = new MultiDiscrete(new long[] { 2, 3, 4 });
            sbyte[][] masks = { new sbyte[] { 1, 0 }, new sbyte[] { 1, 1, 0 }, new sbyte[] { 1, 0, 1, 0 } };
            NDArray[] maskArrays = masks.Select(m => np.array(m)).ToArray();
            try {
                object?[] mask = maskArrays.Cast<object?>().ToArray();
                for (int i = 0; i < 1000; i++) {
                    using NDArray sample = space.Sample(mask);
                    Assert.IsTrue(space.Contains(sample));
                    long[] values = sample.ToArray<long>();
                    for (int entry = 0; entry < 3; entry++)
                        Assert.AreEqual(1, masks[entry][values[entry]], $"entry {entry} took the masked-out {values[entry]}");
                }
            } finally {
                foreach (NDArray array in maskArrays)
                    array.Dispose();
            }
        }

        /// <summary>10000 weighted samples are members, and each entry's frequencies are within 0.05 of its probabilities.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_multidiscrete.py#L294">test_multidiscrete_sample_probabilities</see>.</remarks>
        [TestMethod]
        public void TestMultidiscreteSampleProbabilities() {
            using var space = new MultiDiscrete(new long[] { 3, 3 });
            double[][] probabilities = { new[] { 0.1, 0.7, 0.2 }, new[] { 0.3, 0.3, 0.4 } };
            NDArray[] arrays = probabilities.Select(p => np.array(p)).ToArray();
            try {
                object?[] probability = arrays.Cast<object?>().ToArray();
                var counts = new double[2, 3];
                const int trials = 10000;
                for (int i = 0; i < trials; i++) {
                    using NDArray sample = space.Sample(probability: probability);
                    Assert.IsTrue(space.Contains(sample));
                    long[] values = sample.ToArray<long>();
                    counts[0, values[0]]++;
                    counts[1, values[1]]++;
                }

                for (int entry = 0; entry < 2; entry++) {
                    for (int value = 0; value < 3; value++)
                        Assert.AreEqual(probabilities[entry][value], counts[entry, value] / trials, 0.05, $"entry {entry} value {value}");
                }
            } finally {
                foreach (NDArray array in arrays)
                    array.Dispose();
            }
        }

        /// <summary>
        ///     The slicing assertions shared by the three <c>as_tuple</c> tests: the 1-D space <c>[3, 4, 5]</c> and the 2-D
        ///     space <c>[[3, 4, 5], [6, 7, 8]]</c>, with or without starts, in the default dtype or int8 (1-D) and uint32 (2-D).
        /// </summary>
        /// <param name="dtype"><see langword="null"/> for the default dtype, anything else for the typed variant.</param>
        /// <param name="withStart">Whether the spaces have starts.</param>
        private static void CheckSlices(string? dtype, bool withStart) {
            DType? vectorType = dtype is null ? null : np.int8;
            DType? matrixType = dtype is null ? null : np.uint32;
            using (var space = new MultiDiscrete(new long[] { 3, 4, 5 }, vectorType, start: withStart ? new long[] { 10, 20, 30 } : null)) {
                CollectionAssert.AreEqual(new long[] { 3 }, space.Shape.dimensions);
                AssertSubspace(space, "0", new Discrete(3, start: withStart ? 10 : 0, dtype: vectorType));
                AssertSubspace(space, "0:1", new MultiDiscrete(new long[] { 3 }, vectorType, start: withStart ? new long[] { 10 } : null));
                AssertSubspace(space, "0:2", new MultiDiscrete(new long[] { 3, 4 }, vectorType, start: withStart ? new long[] { 10, 20 } : null));
                AssertWholeCopy(space, ":");
            }

            using (MultiDiscrete space = Matrix(new long[,] { { 3, 4, 5 }, { 6, 7, 8 } }, withStart ? new long[,] { { 10, 20, 30 }, { 40, 50, 60 } } : null, matrixType)) {
                CollectionAssert.AreEqual(new long[] { 2, 3 }, space.Shape.dimensions);
                AssertSubspace(space, "0, 1", new Discrete(4, start: withStart ? 20 : 0, dtype: matrixType));
                AssertSubspace(space, "0", new MultiDiscrete(new long[] { 3, 4, 5 }, matrixType, start: withStart ? new long[] { 10, 20, 30 } : null));
                AssertSubspace(space, "0:1", Matrix(new long[,] { { 3, 4, 5 } }, withStart ? new long[,] { { 10, 20, 30 } } : null, matrixType));
                AssertSubspace(space, "0:2, :", Matrix(new long[,] { { 3, 4, 5 }, { 6, 7, 8 } }, withStart ? new long[,] { { 10, 20, 30 }, { 40, 50, 60 } } : null, matrixType));
                AssertSubspace(space, ":, 0:1", Matrix(new long[,] { { 3 }, { 6 } }, withStart ? new long[,] { { 10 }, { 40 } } : null, matrixType));
                AssertSubspace(space, "0:2, 0:2", Matrix(new long[,] { { 3, 4 }, { 6, 7 } }, withStart ? new long[,] { { 10, 20 }, { 40, 50 } } : null, matrixType));
                AssertWholeCopy(space, ":");
                AssertWholeCopy(space, ":, :");
            }
        }

        /// <summary>Asserts that <c>space[index]</c> equals <paramref name="expected"/>, disposing both.</summary>
        /// <param name="space">The space.</param>
        /// <param name="index">The NumSharp index string.</param>
        /// <param name="expected">The expected subspace; disposed here.</param>
        private static void AssertSubspace(MultiDiscrete space, string index, Space expected) {
            using (expected) {
                using Space actual = space[index];
                Assert.AreEqual(expected, actual, $"space[{index}]");
            }
        }

        /// <summary>Asserts that <c>space[index]</c> equals the space but is a new object.</summary>
        /// <param name="space">The space.</param>
        /// <param name="index">An index selecting everything.</param>
        private static void AssertWholeCopy(MultiDiscrete space, string index) {
            using Space copy = space[index];
            Assert.AreEqual(space, copy, $"space[{index}] == space");
            Assert.AreNotSame(space, copy, $"space[{index}] is not space");
        }

        /// <summary>A 2-D space with an optional start and dtype.</summary>
        /// <param name="nvec">The counts.</param>
        /// <param name="start">The starts, or <see langword="null"/>.</param>
        /// <param name="dtype">The dtype, or <see langword="null"/>.</param>
        /// <returns>The space.</returns>
        private static MultiDiscrete Matrix(long[,] nvec, long[,]? start, DType? dtype) {
            using NDArray counts = np.array(nvec);
            if (start is null)
                return new MultiDiscrete(counts, dtype);
            using NDArray starts = np.array(start);
            return new MultiDiscrete(counts, dtype, start: starts);
        }

        /// <summary>Whether two subspaces taken in a row draw equivalent samples.</summary>
        /// <param name="space">The space.</param>
        /// <param name="first">The first subspace's index.</param>
        /// <param name="second">The second subspace's index.</param>
        /// <returns><see langword="true"/> when equivalent.</returns>
        private static bool SubspaceSamplesEqual(MultiDiscrete space, string first, string second) {
            using Space a = space[first];
            using Space b = space[second];
            using var sampleA = (NDArray)a.Sample()!;
            using var sampleB = (NDArray)b.Sample()!;
            return EnvChecker.DataEquivalence(sampleA, sampleB);
        }

        /// <summary>Whether a whole-space subspace draws what the space itself draws next.</summary>
        /// <param name="space">The space.</param>
        /// <param name="index">An index selecting everything.</param>
        /// <returns><see langword="true"/> when equivalent.</returns>
        private static bool SubspaceVersusSpace(MultiDiscrete space, string index) {
            using Space whole = space[index];
            using var fromSubspace = (NDArray)whole.Sample()!;
            using NDArray fromSpace = space.Sample();
            return EnvChecker.DataEquivalence(fromSubspace, fromSpace);
        }
    }
}
