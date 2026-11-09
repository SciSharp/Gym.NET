#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Spaces;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Tuple = Gym.Spaces.Tuple;

namespace Gym.Tests.Gymnasium.Spaces {
    /// <summary>
    ///     Gymnasium's <c>Tuple</c> tests (<c>tests/spaces/test_tuple.py</c>): the <c>Sequence</c> protocol (length, indexing,
    ///     iteration, <c>count</c>, <c>index</c>), seeding by every form and reseeding with what it returned, membership of
    ///     lists and arrays, the refused seed, and masked, weighted and refused samples.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <c>test_bad_space_calls</c> passes strings where spaces belong, which C#'s typed constructor rejects at compile time;
    ///     a <see langword="null"/> subspace is the runtime case left, and it raises the same <c>TypeError</c>.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_tuple.py">tests/spaces/test_tuple.py</see>.</para>
    /// </remarks>
    [TestClass]
    public class TestTuple {
        /// <summary>A tuple space behaves as a Python sequence of its subspaces.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_tuple.py#L9">test_sequence_inheritance</see>.</remarks>
        [TestMethod]
        public void TestSequenceInheritance() {
            Space[] spaces = { new Discrete(5), new Discrete(10), new Discrete(5) };
            using var tupleSpace = new Tuple(spaces.Select(s => s.DeepCopy()));
            try {
                Assert.AreEqual(spaces.Length, tupleSpace.Count);
                for (int i = 0; i < tupleSpace.Count; i++)
                    Assert.AreEqual(spaces[i], tupleSpace[i]);
                foreach (Space space in tupleSpace)
                    Assert.IsTrue(spaces.Contains(space));

                using (var five = new Discrete(5))
                using (var six = new Discrete(6))
                using (var binary = new MultiBinary(2)) {
                    Assert.AreEqual(2, tupleSpace.CountOf(five));
                    Assert.AreEqual(0, tupleSpace.CountOf(six));
                    Assert.AreEqual(0, tupleSpace.CountOf(binary));
                    Assert.AreEqual(0, tupleSpace.IndexOf(five));
                    Assert.AreEqual(2, tupleSpace.IndexOf(five, 1));
                }

                using (var ten = new Discrete(10))
                    Assert.ThrowsExactly<ValueError>(() => tupleSpace.IndexOf(ten, 0, 1));
                Assert.ThrowsExactly<IndexError>(() => tupleSpace[4]);
            } finally {
                foreach (Space space in spaces)
                    space.Dispose();
            }
        }

        /// <summary>
        ///     Seeding returns one seed per subspace (a list for the tuple), and seeding again with that list reproduces the
        ///     seeds and the sample, for <c>None</c>, an int, a pair and nested pairs or dicts.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_tuple.py#L39">test_seeds</see>.</remarks>
        [TestMethod]
        public void TestSeeds() {
            var cases = new (Func<Tuple> Create, object? Seed)[] {
                (() => new Tuple(new Discrete(5), new Discrete(4)), null),
                (() => new Tuple(new Discrete(5), new Discrete(4)), 123L),
                (() => new Tuple(new Discrete(5), new Discrete(4)), new object?[] { 123L, 456L }),
                (() => new Tuple(new Discrete(5), new Tuple(new Box(0.0, 1.0, new Shape(3L)), new Discrete(2))), new object?[] { 123L, new object?[] { 456L, 789L } }),
                (() => new Tuple(new Discrete(3), TestingSpaces.DictOf(("position", new Box(0.0, 1.0)), ("velocity", new Discrete(2)))),
                    new object?[] { 123L, new Dictionary<string, object?> { ["position"] = 456L, ["velocity"] = 789L } }),
            };
            foreach ((Func<Tuple> create, object? seed) in cases) {
                using Tuple space = create();
                object seeds1 = space.SeedWith(seed);
                Assert.IsInstanceOfType<IReadOnlyList<object>>(seeds1, $"{space}");
                Assert.AreEqual(space.Count, ((IReadOnlyList<object>)seeds1).Count, $"{space}");
                object?[] sample1 = space.Sample();

                object seeds2 = space.SeedWith(seeds1);
                object?[] sample2 = space.Sample();
                try {
                    Assert.IsTrue(EnvChecker.DataEquivalence(seeds1, seeds2), $"{space}: seeds");
                    Assert.IsTrue(EnvChecker.DataEquivalence(sample1, sample2), $"{space}: samples");
                } finally {
                    Release(sample1);
                    Release(sample2);
                }
            }
        }

        /// <summary>A <see langword="null"/> subspace is refused with a <c>TypeError</c>.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_tuple.py#L76">test_bad_space_calls</see>.</remarks>
        [TestMethod]
        public void TestBadSpaceCalls() {
            Assert.ThrowsExactly<TypeError>(() => new Tuple(new Space[] { null! }));
            using var box = new Box(0, 1);
            Assert.ThrowsExactly<TypeError>(() => new Tuple(new Space[] { box, null! }));
        }

        /// <summary>A tuple of arrays, and an array whose rows are the parts, are members.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_tuple.py#L88">test_contains_promotion</see>.</remarks>
        [TestMethod]
        public void TestContainsPromotion() {
            using (var space = new Tuple(new Box(0, 1), new Box(-1, 0, new Shape(2L))))
            using (NDArray first = np.array(new[] { 0.0f }))
            using (NDArray second = np.array(new[] { 0.0f, 0.0f }))
                Assert.IsTrue(space.Contains(new object?[] { first, second }));

            using (var space = new Tuple(new Box(0, 1), new Box(-1, 0, new Shape(1L))))
            using (NDArray rows = np.array(new float[,] { { 0.0f }, { 0.0f } }))
                Assert.IsTrue(space.Contains(rows));
        }

        /// <summary>A float seed is refused with Gymnasium's <c>TypeError</c>.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_tuple.py#L100">test_bad_seed</see>.</remarks>
        [TestMethod]
        public void TestBadSeed() {
            using var space = new Tuple(new Box(0, 1), new Box(0, 1));
            var error = Assert.ThrowsExactly<TypeError>(() => space.SeedWith(0.0));
            Assert.AreEqual("Expected seed type: list, tuple, int or None, actual type: <class 'float'>", error.Message);
        }

        /// <summary>Unmasked, masked and weighted samples.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_tuple.py#L108">test_oneof_sample</see>.</remarks>
        [TestMethod]
        public void TestOneofSample() {
            using var space = new Tuple(new Discrete(2), new Box(-1, 1, new Shape(2L)));
            object?[] sample = space.Sample();
            try {
                Assert.AreEqual(2, sample.Length);
                Assert.IsTrue(space[0].Contains(sample[0]));
                Assert.IsTrue(space[1].Contains(sample[1]));
            } finally {
                Release(sample);
            }

            using (NDArray maskArray = np.array(new sbyte[] { 1, 0 })) {
                object?[] masked = space.Sample(new object?[] { maskArray, null });
                try {
                    Assert.IsTrue(space[0].Contains(masked[0]));
                    Assert.IsTrue(space[1].Contains(masked[1]));
                    Assert.AreEqual(0L, Convert.ToInt64(((NDArray)masked[0]!).GetAtIndex(0)));
                } finally {
                    Release(masked);
                }
            }

            using (NDArray probabilityArray = np.array(new[] { 0.8, 0.2 })) {
                var counts = new double[2];
                for (int i = 0; i < 1000; i++) {
                    object?[] weighted = space.Sample(probability: new object?[] { probabilityArray, null });
                    counts[Convert.ToInt64(((NDArray)weighted[0]!).GetAtIndex(0))]++;
                    Release(weighted);
                }

                Assert.AreEqual(0.8, counts[0] / 1000, 0.05);
                Assert.AreEqual(0.2, counts[1] / 1000, 0.05);
            }
        }

        /// <summary>Both masks, a mask that is not a tuple, or one of the wrong length, are refused with Gymnasium's messages.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_tuple.py#L136">test_invalid_sample_inputs</see>.</remarks>
        [TestMethod]
        public void TestInvalidSampleInputs() {
            using var space = new Tuple(new Discrete(2), new Box(-1, 1, new Shape(2L)));
            StringAssert.Contains(Assert.ThrowsExactly<ValueError>(() => space.Sample(new object?[] { null, null }, new object?[] { 0.5, 0.5 })).Message,
                "Only one of `mask` or `probability` can be provided");
            StringAssert.Contains(Assert.ThrowsExactly<AssertionError>(() => space.Sample(new Dictionary<string, object?> { ["low"] = 0L, ["high"] = 1L })).Message,
                "Expected type of `mask` to be tuple, actual type: <class 'dict'>");
            StringAssert.Contains(Assert.ThrowsExactly<AssertionError>(() => space.Sample(new object?[] { null })).Message,
                "Expected length of `mask` to be 2, actual length: 1");
            StringAssert.Contains(Assert.ThrowsExactly<AssertionError>(() => space.Sample(probability: new object?[] { 0.5 })).Message,
                "Expected length of `probability` to be 2, actual length: 1");
            StringAssert.Contains(Assert.ThrowsExactly<AssertionError>(() => space.Sample(probability: new List<object?> { 0.5, 0.5 })).Message,
                "Expected type of `probability` to be tuple, actual type: <class 'list'>");
        }

        /// <summary>Disposes the arrays of a tuple sample.</summary>
        /// <param name="sample">The sample.</param>
        private static void Release(object?[] sample) {
            foreach (object? item in sample) {
                if (item is object?[] nested)
                    Release(nested);
                else if (item is Gym.Envs.Info info)
                    foreach (KeyValuePair<string, object?> pair in info)
                        (pair.Value as IDisposable)?.Dispose();
                else
                    (item as IDisposable)?.Dispose();
            }
        }
    }
}
