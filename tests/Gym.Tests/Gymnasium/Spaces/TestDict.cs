#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Envs;
using Gym.Spaces;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Spaces {
    /// <summary>
    ///     Gymnasium's <c>Dict</c> tests (<c>tests/spaces/test_dict.py</c>): construction and key order (a mapping is sorted
    ///     unless <c>sort_keys=False</c>, pairs never are), seeding by dict, int and <c>None</c>, the refused seed, the mapping
    ///     protocol, and masked, weighted and refused samples.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The Python-only parts of <c>test_dict_init</c> have no C# counterpart: a space given where the mapping belongs,
    ///     keyword spaces that repeat a mapping key, a string where a space belongs, non-string keys, and assigning a non-space
    ///     value through the indexer (C#'s types reject each at compile time).
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_dict.py">tests/spaces/test_dict.py</see>.</para>
    /// </remarks>
    [TestClass]
    public class TestDict {
        /// <summary>
        ///     Equal dicts from a mapping, an ordered mapping and pairs; key order (sorted mapping, unsorted when asked, pairs as
        ///     given); no warnings.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_dict.py#L12">test_dict_init</see>.</remarks>
        [TestMethod]
        public void TestDictInit() {
            using (WarningCatcher caught = Warnings.Catch()) {
                using var a = new Dict(new Dictionary<string, Space> { ["a"] = new Discrete(2), ["b"] = new Box(0.0, 1.0) });
                using var c = TestingSpaces.DictOf(("a", new Discrete(2)), ("b", new Box(0.0, 1.0)));
                Assert.AreEqual(a, c);
                Assert.AreEqual(0, caught.Records.Count);
            }

            using (WarningCatcher caught = Warnings.Catch()) {
                using (var sorted = new Dict(new Dictionary<string, Space> { ["b"] = new Box(0.0, 1.0), ["a"] = new Discrete(2) }))
                    CollectionAssert.AreEqual(new[] { "a", "b" }, sorted.Keys.ToArray());
                using (var unsorted = new Dict(new Dictionary<string, Space> { ["b"] = new Box(0.0, 1.0), ["a"] = new Discrete(2) }, sortKeys: false))
                    CollectionAssert.AreEqual(new[] { "b", "a" }, unsorted.Keys.ToArray());
                using (Dict pairs = TestingSpaces.DictOf(("b", new Box(0.0, 1.0)), ("a", new Discrete(2))))
                    CollectionAssert.AreEqual(new[] { "b", "a" }, pairs.Keys.ToArray());
                Assert.AreEqual(0, caught.Records.Count);
            }
        }

        /// <summary>Seeding by a nested dict returns that dict, and each subspace then draws what a space seeded alike draws.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_dict.py#L80">test_dict_seeding</see>.</remarks>
        [TestMethod]
        public void TestDictSeeding() {
            using Dict dictSpace = DictSpace();
            var seedingValues = new Dictionary<string, object?> {
                ["a"] = 0L,
                ["b"] = new Dictionary<string, object?> { ["b_1"] = 1L, ["b_2"] = 2L },
                ["c"] = 3L,
            };
            object seededValues = dictSpace.SeedWith(seedingValues);
            Assert.IsTrue(EnvChecker.DataEquivalence(seededValues, seedingValues), PyBuiltins.Repr(seededValues));

            using var a = new Box(0, 1, new Shape(3L, 3L), seed: 0);
            using var b1 = new Box(-100, 100, new Shape(2L), seed: 1);
            using var b2 = new Box(-1, 1, new Shape(2L), seed: 2);
            using var c = new Discrete(5, seed: 3);
            AssertSamplesMatch(dictSpace, a, b1, b2, c);
        }

        /// <summary>Seeding by an int returns a dict of the subspaces' seeds, which reproduce the subspaces' streams.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_dict.py#L106">test_int_seeding</see>.</remarks>
        [TestMethod]
        public void TestIntSeeding() {
            using Dict dictSpace = DictSpace();
            var seeds = (IReadOnlyDictionary<string, object>)dictSpace.Seed(1);
            var bSeeds = (IReadOnlyDictionary<string, object>)seeds["b"];
            using var a = new Box(0, 1, new Shape(3L, 3L), seed: Seed(seeds["a"]));
            using var b1 = new Box(-100, 100, new Shape(2L), seed: Seed(bSeeds["b_1"]));
            using var b2 = new Box(-1, 1, new Shape(2L), seed: Seed(bSeeds["b_2"]));
            using var c = new Discrete(5, seed: Seed(seeds["c"]));
            AssertSamplesMatch(dictSpace, a, b1, b2, c);
        }

        /// <summary>Seeding by <c>None</c> returns a dict.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_dict.py#L128">test_none_seeding</see>.</remarks>
        [TestMethod]
        public void TestNoneSeeding() {
            using Dict dictSpace = DictSpace();
            Assert.IsInstanceOfType<IReadOnlyDictionary<string, object>>(dictSpace.SeedWith(null));
        }

        /// <summary>A string seed is refused with Gymnasium's <c>TypeError</c>.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_dict.py#L133">test_bad_seed</see>.</remarks>
        [TestMethod]
        public void TestBadSeed() {
            using Dict dictSpace = DictSpace();
            Assert.AreEqual("Expected seed type: dict, int or None, actual type: <class 'str'>", Assert.ThrowsExactly<TypeError>(() => dictSpace.SeedWith("a")).Message);
        }

        /// <summary>A dict space is a mapping: its length, reading and assigning a subspace by key.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_dict.py#L143">test_mapping</see>.</remarks>
        [TestMethod]
        public void TestMapping() {
            using Dict dictSpace = DictSpace();
            Assert.AreEqual(3, dictSpace.Count);
            Space a = dictSpace["a"];
            using var b = new Discrete(5);
            Assert.AreNotEqual(a, b);
            dictSpace["a"] = b;
            Assert.AreEqual(b, dictSpace["a"]);
            dictSpace["a"] = a;
        }

        /// <summary>Iterating gives the keys of the subspaces.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_dict.py#L162">test_iterator</see>.</remarks>
        [TestMethod]
        public void TestIterator() {
            using Dict dictSpace = DictSpace();
            foreach (KeyValuePair<string, Space> pair in dictSpace)
                Assert.IsTrue(dictSpace.ContainsKey(pair.Key));
            CollectionAssert.AreEquivalent(new[] { "a", "b", "c" }, dictSpace.Keys.ToArray());
        }

        /// <summary>The keys answer membership.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_dict.py#L170">test_keys_contains</see>.</remarks>
        [TestMethod]
        public void TestKeysContains() {
            using Dict space = TestingSpaces.DictOf(("a", new Box(0, 1)), ("b", new Box(1, 2)));
            foreach (string key in space.Keys)
                Assert.IsTrue(space.Keys.Contains(key));
            Assert.IsTrue(space.Keys.Contains("a"));
            Assert.IsFalse(space.Keys.Contains("c"));
        }

        /// <summary>A per-key mask restricts that subspace; <c>None</c> leaves the other free.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_dict.py#L181">test_sample_with_mask</see>.</remarks>
        [TestMethod]
        public void TestSampleWithMask() {
            using var space = new Dict(new Dictionary<string, Space> { ["a"] = new Discrete(5), ["b"] = new Box(0, 1, new Shape(2L)) });
            using NDArray maskA = np.array(new sbyte[] { 0, 1, 0, 0, 0 });
            var mask = new Dictionary<string, object?> { ["a"] = maskA, ["b"] = null };
            for (int i = 0; i < 10; i++) {
                Info sample = space.Sample(mask);
                try {
                    Assert.AreEqual(1L, Convert.ToInt64(((NDArray)sample["a"]!).GetAtIndex(0)));
                    Assert.IsTrue(space["b"].Contains(sample["b"]));
                } finally {
                    Release(sample);
                }
            }
        }

        /// <summary>A per-key probability mask weights that subspace.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_dict.py#L201">test_sample_with_probability</see>.</remarks>
        [TestMethod]
        public void TestSampleWithProbability() {
            using var space = new Dict(new Dictionary<string, Space> { ["a"] = new Discrete(3), ["b"] = new Box(0, 1, new Shape(2L)) });
            using NDArray probabilityA = np.array(new[] { 0.1, 0.7, 0.2 });
            var probability = new Dictionary<string, object?> { ["a"] = probabilityA, ["b"] = null };
            var counts = new double[3];
            for (int i = 0; i < 1000; i++) {
                Info sample = space.Sample(probability: probability);
                counts[Convert.ToInt64(((NDArray)sample["a"]!).GetAtIndex(0))]++;
                Release(sample);
            }

            // np.testing.assert_almost_equal(..., decimal=1): |actual - desired| < 1.5 * 10**-1.
            double[] expected = { 0.1, 0.7, 0.2 };
            for (int i = 0; i < 3; i++)
                Assert.IsTrue(Math.Abs(counts[i] / 1000 - expected[i]) < 1.5e-1, $"value {i}: {counts[i] / 1000}");
        }

        /// <summary>A per-key mask of the wrong shape or dtype is refused by the subspace, with Gymnasium's messages.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_dict.py#L224">test_sample_with_invalid_mask</see>.</remarks>
        [TestMethod]
        public void TestSampleWithInvalidMask() {
            using var space = new Dict(new Dictionary<string, Space> { ["a"] = new Discrete(5), ["b"] = new Box(0, 1, new Shape(2L)) });
            using (NDArray short3 = np.array(new sbyte[] { 1, 0, 0 }))
                StringAssert.Contains(Assert.ThrowsExactly<AssertionError>(() => space.Sample(new Dictionary<string, object?> { ["a"] = short3, ["b"] = null })).Message,
                    "The expected shape of the sample mask is (5,), actual shape: (3,)");
            using (NDArray floats = np.array(new[] { 1f, 0f, 0f, 1f, 1f }))
                StringAssert.Contains(Assert.ThrowsExactly<AssertionError>(() => space.Sample(new Dictionary<string, object?> { ["a"] = floats, ["b"] = null })).Message,
                    "The expected dtype of the sample mask is np.int8, actual dtype: float32");
        }

        /// <summary>A per-key probability mask of the wrong shape or dtype is refused, with Gymnasium's messages.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_dict.py#L256">test_sample_with_invalid_probability</see>.</remarks>
        [TestMethod]
        public void TestSampleWithInvalidProbability() {
            using var space = new Dict(new Dictionary<string, Space> { ["a"] = new Discrete(5), ["b"] = new Box(0, 1, new Shape(2L)) });
            using (NDArray short2 = np.array(new[] { 0.5, 0.5 }))
                StringAssert.Contains(Assert.ThrowsExactly<AssertionError>(() => space.Sample(probability: new Dictionary<string, object?> { ["a"] = short2, ["b"] = null })).Message,
                    "The expected shape of the sample probability is (5,), actual shape: (2,)");
            using (NDArray ints = np.array(new sbyte[] { 0, 0 }))
                StringAssert.Contains(Assert.ThrowsExactly<AssertionError>(() => space.Sample(probability: new Dictionary<string, object?> { ["a"] = ints, ["b"] = null })).Message,
                    "The expected dtype of the sample probability is np.float64, actual dtype: int8");
        }

        /// <summary>A mask and a probability mask together are refused.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_dict.py#L288">test_sample_with_mask_and_probability</see>.</remarks>
        [TestMethod]
        public void TestSampleWithMaskAndProbability() {
            using var space = new Dict(new Dictionary<string, Space> { ["a"] = new Discrete(3), ["b"] = new Box(0, 1, new Shape(2L)) });
            using NDArray mask = np.array(new sbyte[] { 1, 0, 1 });
            using NDArray probability = np.array(new[] { 0.5, 0.2, 0.3 });
            StringAssert.Contains(Assert.ThrowsExactly<ValueError>(() => space.Sample(new Dictionary<string, object?> { ["a"] = mask, ["b"] = null },
                new Dictionary<string, object?> { ["a"] = probability, ["b"] = null })).Message, "Only one of `mask` or `probability` can be provided");
        }

        /// <summary>Gymnasium's <c>DICT_SPACE</c>: a box, a dict of two boxes, and a discrete, built fresh per test.</summary>
        /// <returns>The space.</returns>
        private static Dict DictSpace() => new(new Dictionary<string, Space> {
            ["a"] = new Box(0, 1, new Shape(3L, 3L)),
            ["b"] = new Dict(new Dictionary<string, Space> { ["b_1"] = new Box(-100, 100, new Shape(2L)), ["b_2"] = new Box(-1, 1, new Shape(2L)) }),
            ["c"] = new Discrete(5),
        });

        /// <summary>Ten dict samples match the samples of the four separately seeded subspaces.</summary>
        /// <param name="dictSpace">The seeded dict space.</param>
        /// <param name="a">The space seeded like <c>a</c>.</param>
        /// <param name="b1">The space seeded like <c>b.b_1</c>.</param>
        /// <param name="b2">The space seeded like <c>b.b_2</c>.</param>
        /// <param name="c">The space seeded like <c>c</c>.</param>
        private static void AssertSamplesMatch(Dict dictSpace, Box a, Box b1, Box b2, Discrete c) {
            for (int i = 0; i < 10; i++) {
                Info sample = dictSpace.Sample();
                var b = (Info)sample["b"]!;
                try {
                    using NDArray expectedA = a.Sample();
                    using NDArray expectedB1 = b1.Sample();
                    using NDArray expectedB2 = b2.Sample();
                    using NDArray expectedC = c.Sample();
                    Assert.IsTrue(np.array_equal((NDArray)sample["a"]!, expectedA), $"sample {i}: a");
                    Assert.IsTrue(np.array_equal((NDArray)b["b_1"]!, expectedB1), $"sample {i}: b_1");
                    Assert.IsTrue(np.array_equal((NDArray)b["b_2"]!, expectedB2), $"sample {i}: b_2");
                    Assert.IsTrue(np.array_equal((NDArray)sample["c"]!, expectedC), $"sample {i}: c");
                } finally {
                    Release(sample);
                }
            }
        }

        /// <summary>A seed the dict returned, as the <c>long</c> a constructor takes.</summary>
        /// <param name="value">The seed (a <see cref="System.Numerics.BigInteger"/>).</param>
        /// <returns>The seed.</returns>
        private static long Seed(object value) => (long)(System.Numerics.BigInteger)value;

        /// <summary>Disposes the arrays of a dict sample, nested dicts included.</summary>
        /// <param name="sample">The sample.</param>
        private static void Release(Info sample) {
            foreach (KeyValuePair<string, object?> pair in sample) {
                if (pair.Value is Info nested)
                    Release(nested);
                else
                    (pair.Value as IDisposable)?.Dispose();
            }
        }
    }
}
