#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Spaces;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Spaces {
    /// <summary>
    ///     Gymnasium's <c>OneOf</c> tests (<c>tests/spaces/test_oneof.py</c>): the sequence protocol, seeding (one seed more
    ///     than subspaces), refused subspaces and seeds, membership by index type, and masked, weighted and refused samples.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <c>test_bad_oneof_calls</c> passes strings where spaces belong, which C#'s typed constructor rejects at compile time;
    ///     a <see langword="null"/> subspace is the runtime case left, and it raises the same <c>TypeError</c>.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_oneof.py">tests/spaces/test_oneof.py</see>.</para>
    /// </remarks>
    [TestClass]
    public class TestOneOf {
        /// <summary>Length, indexing and iteration over the subspaces.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_oneof.py#L7">test_oneof_inheritance</see>.</remarks>
        [TestMethod]
        public void TestOneofInheritance() {
            Space[] spaces = { new Discrete(5), new Box(-1, 1, new Shape(3L)), new MultiBinary(2) };
            try {
                using var oneofSpace = new OneOf(spaces.Select(s => s.DeepCopy()));
                Assert.AreEqual(spaces.Length, oneofSpace.Count);
                for (int i = 0; i < oneofSpace.Count; i++)
                    Assert.AreEqual(spaces[i], oneofSpace[i]);
                foreach (Space space in oneofSpace)
                    Assert.IsTrue(spaces.Contains(space));
            } finally {
                foreach (Space space in spaces)
                    space.Dispose();
            }
        }

        /// <summary>Seeding by <c>None</c>, an int or a full list returns one seed for the space and one per subspace.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_oneof.py#L22">test_oneof_seeds</see>.</remarks>
        [TestMethod]
        public void TestOneofSeeds() {
            foreach (object? seed in new object?[] { null, 123L, new object?[] { 123L, 456L, 789L } }) {
                using var oneofSpace = new OneOf(new Discrete(5), new Box(-1, 1, new Shape(3L)));
                var seeds = (IReadOnlyList<object>)oneofSpace.SeedWith(seed);
                Assert.AreEqual(3, seeds.Count);
            }
        }

        /// <summary>A <see langword="null"/> subspace is refused with a <c>TypeError</c>.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_oneof.py#L44">test_bad_oneof_calls</see>.</remarks>
        [TestMethod]
        public void TestBadOneofCalls() {
            Assert.ThrowsExactly<TypeError>(() => new OneOf(new Space[] { null! }));
            using var box = new Box(0, 1);
            Assert.ThrowsExactly<TypeError>(() => new OneOf(new Space[] { box, null! }));
        }

        /// <summary>Membership: an int or <c>np.int64</c> index is accepted, an <c>np.int32</c> one is not.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_oneof.py#L56">test_oneof_contains</see>.</remarks>
        [TestMethod]
        public void TestOneofContains() {
            using var space = new OneOf(new Box(0, 1), new Box(-1, 0, new Shape(2L)));
            using NDArray half = np.array(new[] { 0.5f });
            using NDArray minusHalves = np.array(new[] { -0.5f, -0.5f });
            using NDArray int64Index = NDArray.Scalar(0L);
            using NDArray int32Index = NDArray.Scalar(0);
            Assert.IsTrue(space.Contains(new object?[] { 0L, half }));
            Assert.IsTrue(space.Contains(new object?[] { 1L, minusHalves }));
            Assert.IsTrue(space.Contains(new object?[] { int64Index, half }));
            Assert.IsFalse(space.Contains(new object?[] { int32Index, half }));
        }

        /// <summary>A float seed is refused with Gymnasium's <c>TypeError</c>.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_oneof.py#L67">test_bad_oneof_seed</see>.</remarks>
        [TestMethod]
        public void TestBadOneofSeed() {
            using var space = new OneOf(new Box(0, 1), new Box(0, 1));
            Assert.AreEqual("Expected None, int, or tuple of ints, actual type: <class 'float'>", Assert.ThrowsExactly<TypeError>(() => space.SeedWith(0.0)).Message);
        }

        /// <summary>Unmasked, masked (the discrete subspace forced to 0) and weighted samples.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_oneof.py#L75">test_oneof_sample</see>.</remarks>
        [TestMethod]
        public void TestOneofSample() {
            using var space = new OneOf(new Discrete(2), new Box(-1, 1, new Shape(2L)));
            object?[] sample = space.Sample();
            try {
                long index = Convert.ToInt64(((NDArray)sample[0]!).GetAtIndex(0));
                Assert.IsTrue(index is 0 or 1);
                Assert.IsTrue(space[(int)index].Contains(sample[1]));
            } finally {
                Release(sample);
            }

            using (NDArray maskArray = np.array(new sbyte[] { 1, 0 })) {
                object?[] mask = { maskArray, null };
                long index;
                do {
                    object?[] masked = space.Sample(mask);
                    try {
                        index = Convert.ToInt64(((NDArray)masked[0]!).GetAtIndex(0));
                        Assert.IsTrue(index is 0 or 1);
                        if (index == 0)
                            Assert.AreEqual(0L, Convert.ToInt64(((NDArray)masked[1]!).GetAtIndex(0)));
                    } finally {
                        Release(masked);
                    }
                } while (index != 0);
            }

            using (NDArray weights = np.array(new[] { 0.8, 0.2 })) {
                object?[] weighted = space.Sample(probability: new object?[] { weights, null });
                try {
                    Assert.IsTrue(Convert.ToInt64(((NDArray)weighted[0]!).GetAtIndex(0)) is 0 or 1);
                } finally {
                    Release(weighted);
                }
            }
        }

        /// <summary>Both masks, a mask that is not a tuple, or one of the wrong length are refused with Gymnasium's messages.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_oneof.py#L102">test_invalid_sample_inputs</see>.</remarks>
        [TestMethod]
        public void TestInvalidSampleInputs() {
            using var space = new OneOf(new Discrete(2), new Box(-1, 1, new Shape(2L)));
            StringAssert.Contains(Assert.ThrowsExactly<ValueError>(() => space.Sample(new object?[] { null, null }, new object?[] { 0.5, 0.5 })).Message,
                "Only one of `mask` or `probability` can be provided");
            StringAssert.Contains(Assert.ThrowsExactly<AssertionError>(() => space.Sample(new Dictionary<string, object?> { ["low"] = 0L, ["high"] = 1L })).Message,
                "Expected type of `mask` is tuple");
            StringAssert.Contains(Assert.ThrowsExactly<AssertionError>(() => space.Sample(new object?[] { null })).Message, "Expected length of `mask` is 2");
            StringAssert.Contains(Assert.ThrowsExactly<AssertionError>(() => space.Sample(probability: new object?[] { 0.5 })).Message, "Expected length of `probability` is 2");
            StringAssert.Contains(Assert.ThrowsExactly<AssertionError>(() => space.Sample(probability: new List<object?> { 0.5, 0.5 })).Message, "Expected type of `probability` is tuple");
        }

        /// <summary>Disposes a pair sample's arrays.</summary>
        /// <param name="sample">The pair.</param>
        private static void Release(object?[] sample) {
            foreach (object? item in sample)
                (item as IDisposable)?.Dispose();
        }
    }
}
