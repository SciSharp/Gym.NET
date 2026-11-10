#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Exceptions;
using Gym.Spaces;
using Gym.Tests.Gymnasium.Spaces;
using Gym.Utils;
using Gym.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Dict = Gym.Spaces.Dict;
using SpaceTuple = Gym.Spaces.Tuple;

namespace Gym.Tests.Gymnasium.Vector.Utils {
    /// <summary>
    ///     Gymnasium's vector space utility tests (<c>tests/vector/utils/test_space_utils.py</c>), run against
    ///     <see cref="VectorUtils"/>'s <c>batch_space</c>, <c>batch_differing_spaces</c>, <c>iterate</c>, <c>concatenate</c>
    ///     and <c>create_empty_array</c> over Gymnasium's testing space catalog.
    /// </summary>
    /// <remarks>
    ///     <c>test_non_space</c> passes the string <c>"space"</c> where Gymnasium's functions take any object; Gym.NET's take a
    ///     <see cref="Space"/>, so the one non-space C# can pass is <see langword="null"/>, whose refusal names
    ///     <c>NoneType</c>. <c>CustomSpace</c> is the catalog's own (<see cref="TestingSpaces"/>), so its class name in the
    ///     messages is the .NET one.
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/utils/test_space_utils.py">tests/vector/utils/test_space_utils.py</see>.
    /// </remarks>
    [TestClass]
    public class TestSpaceUtils {
        /// <summary>Every catalog space with <c>n</c> in [1, 4].</summary>
        public static IEnumerable<object[]> SpacesAndN14 => SpacesWith(1, 4);

        /// <summary>Every catalog space with <c>n</c> in [1, 3].</summary>
        public static IEnumerable<object[]> SpacesAndN13 => SpacesWith(1, 3);

        /// <summary>Every catalog space with <c>n</c> in [1, 2, 5] and each base seed.</summary>
        public static IEnumerable<object[]> SpacesN125AndSeeds =>
            from row in SpacesWith(1, 2, 5) from seed in new[] { 123, 456 } select new[] { row[0], row[1], seed };

        /// <summary>Every catalog space with <c>n</c> in [4, 5] and each base seed.</summary>
        public static IEnumerable<object[]> SpacesN45AndSeeds =>
            from row in SpacesWith(4, 5) from seed in new[] { 123, 456 } select new[] { row[0], row[1], seed };

        /// <summary>
        ///     The utilities together: a batched space samples members whose iterated items belong to the space; an empty
        ///     array iterates into <c>n</c> items; samples concatenated into it belong to the batched space and iterate back
        ///     into the same samples.
        /// </summary>
        /// <param name="index">The space's index in the catalog.</param>
        /// <param name="n">The batch size.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/utils/test_space_utils.py#L25">tests/vector/utils/test_space_utils.py (test_batch_space_concatenate_iterate_create_empty_array)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(SpacesAndN14))]
        public void TestBatchSpaceConcatenateIterateCreateEmptyArray(int index, int n) {
            Space space = TestingSpaces.All[index]();
            // Batch the space and create a sample
            Space batchedSpace = VectorUtils.BatchSpace(space, n);
            object? batchedSample = batchedSpace.Sample();
            Assert.IsTrue(batchedSpace.Contains(batchedSample), $"{batchedSpace}: own sample");

            // Check the batched samples are within the original space
            List<object?> unbatchedSamples = VectorUtils.Iterate(batchedSpace, batchedSample).ToList();
            Assert.AreEqual(n, unbatchedSamples.Count);
            Assert.IsTrue(unbatchedSamples.All(space.Contains), $"{space}: iterated samples");

            // Create an empty array and check that space is within the batch space
            object? array = VectorUtils.CreateEmptyArray(space, n);
            List<object?> unbatchedArray = VectorUtils.Iterate(batchedSpace, array).ToList();
            Assert.AreEqual(n, unbatchedArray.Count);

            // Generate samples from the original space and concatenate using array into a single object
            List<object?> spaceSamples = Enumerable.Range(0, n).Select(_ => space.Sample()).ToList();
            Assert.IsTrue(spaceSamples.All(space.Contains));
            object? concatenatedSamplesArray = VectorUtils.Concatenate(space, spaceSamples, array);
            Assert.IsTrue(batchedSpace.Contains(concatenatedSamplesArray), $"{batchedSpace}: concatenated samples");

            // Iterate over the samples and check that the concatenated samples == original samples
            unbatchedSamples = VectorUtils.Iterate(batchedSpace, concatenatedSamplesArray).ToList();
            Assert.AreEqual(n, unbatchedSamples.Count);
            for (int i = 0; i < n; i++)
                Assert.IsTrue(EnvChecker.DataEquivalence(unbatchedSamples[i], spaceSamples[i]), $"{space}: sample {i}");
        }

        /// <summary>
        ///     Batching is deterministic: a deep copy's generator starts equal but separate, the batches of both are seeded
        ///     alike and apart from their spaces, and sampling the original space does not move a batch's stream.
        /// </summary>
        /// <param name="index">The space's index in the catalog.</param>
        /// <param name="n">The batch size.</param>
        /// <param name="baseSeed">The seed.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/utils/test_space_utils.py#L69">tests/vector/utils/test_space_utils.py (test_batch_space_deterministic)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(SpacesN125AndSeeds))]
        public void TestBatchSpaceDeterministic(int index, int n, int baseSeed) {
            Space spaceA = TestingSpaces.All[index]();
            spaceA.Seed(baseSeed);
            Space spaceB = spaceA.DeepCopy();
            TestingUtils.AssertRngEqual(spaceA.NpRandom, spaceB.NpRandom);
            Assert.AreNotSame(spaceA.NpRandom, spaceB.NpRandom);

            // Batch the spaces and check that the np_random are not reference equal
            Space spaceABatched = VectorUtils.BatchSpace(spaceA, n);
            Space spaceBBatched = VectorUtils.BatchSpace(spaceB, n);
            TestingUtils.AssertRngEqual(spaceABatched.NpRandom, spaceBBatched.NpRandom);
            Assert.AreNotSame(spaceABatched.NpRandom, spaceBBatched.NpRandom);
            // Create that the batched space is not reference equal to the origin spaces
            Assert.AreNotSame(spaceA.NpRandom, spaceABatched.NpRandom);

            // Check that batched space a and b random number generator are not effected by the original space
            spaceA.Sample();
            AssertSameItems(spaceABatched, spaceABatched.Sample(), spaceBBatched, spaceBBatched.Sample());
        }

        /// <summary>A batched space's items differ from one another: the batch does not copy one generator for each item.</summary>
        /// <param name="index">The space's index in the catalog.</param>
        /// <param name="n">The batch size.</param>
        /// <param name="baseSeed">The seed.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/utils/test_space_utils.py#L106">tests/vector/utils/test_space_utils.py (test_batch_space_different_samples)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(SpacesN45AndSeeds))]
        public void TestBatchSpaceDifferentSamples(int index, int n, int baseSeed) {
            Space space = TestingSpaces.All[index]();
            space.Seed(baseSeed);

            Space batchedSpace = VectorUtils.BatchSpace(space, n);
            Assert.AreNotSame(space.NpRandom, batchedSpace.NpRandom);
            TestingUtils.AssertRngEqual(space.NpRandom, batchedSpace.NpRandom);

            object? batchedSample = batchedSpace.Sample();
            List<object?> unbatchedSamples = VectorUtils.Iterate(batchedSpace, batchedSample).ToList();
            Assert.AreEqual(n, unbatchedSamples.Count);
            Assert.IsTrue(unbatchedSamples.All(space.Contains));
            Assert.IsFalse(unbatchedSamples.All(element => EnvChecker.DataEquivalence(element, unbatchedSamples[0])), $"{space}: every item equal");
        }

        /// <summary>Each utility refuses a non-space with Gymnasium's message (C# can pass only <see langword="null"/>, Python's <c>None</c>).</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/utils/test_space_utils.py#L128">tests/vector/utils/test_space_utils.py (test_non_space)</see>.</remarks>
        [TestMethod]
        public void TestNonSpace() {
            var calls = new (string Name, Action Call)[] {
                ("batch_space", () => VectorUtils.BatchSpace(null!, 1)),
                ("concatenate", () => VectorUtils.Concatenate(null!, new object?[] { null }, null)),
                ("iterate", () => VectorUtils.Iterate(null!, null).ToList()),
                ("create_empty_array", () => VectorUtils.CreateEmptyArray(null!, 1)),
            };
            foreach ((string name, Action call) in calls) {
                var error = Assert.ThrowsExactly<TypeError>(call);
                StringAssert.StartsWith(error.Message, $"The space provided to `{name}` is not a gymnasium Space instance, type: <class 'NoneType'>, None");
            }
        }

        /// <summary>
        ///     A custom space batches into a tuple of itself, cannot be iterated (Gymnasium's <c>CustomSpaceError</c>),
        ///     concatenates into a tuple of its items, and has no empty array.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/utils/test_space_utils.py#L140">tests/vector/utils/test_space_utils.py (test_custom_space)</see>.</remarks>
        [TestMethod]
        public void TestCustomSpace() {
            var customSpace = new CustomSpace();

            Space batchedSpace = VectorUtils.BatchSpace(customSpace, n: 2);
            Assert.AreEqual(new SpaceTuple(customSpace, customSpace), batchedSpace);

            string type = PyBuiltins.TypeRepr(customSpace);
            var error = Assert.ThrowsExactly<CustomSpaceError>(() => VectorUtils.Iterate(customSpace, null).ToList());
            Assert.AreEqual($"Space of type `{type}` doesn't have an registered `iterate` function. Register `{type}` for `iterate` to support it.", error.Message);

            object? concatenatedItems = VectorUtils.Concatenate(customSpace, new object?[] { null, null }, @out: null);
            Assert.IsTrue(PyEquality.AreEqual(concatenatedItems, new object?[] { null, null }));

            Assert.IsNull(VectorUtils.CreateEmptyArray(customSpace));
        }

        /// <summary>Batching a <c>Dict</c> built without sorting keeps its key order, alone and with a copy.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/utils/test_space_utils.py#L161">tests/vector/utils/test_space_utils.py (test_batch_dict_preserves_key_order)</see>.</remarks>
        [TestMethod]
        public void TestBatchDictPreservesKeyOrder() {
            var space = new Dict(new[] { new KeyValuePair<string, Space>("b", new Discrete(2)), new KeyValuePair<string, Space>("a", new Discrete(3)) }, sortKeys: false);

            CollectionAssert.AreEqual(new[] { "b", "a" }, ((Dict)VectorUtils.BatchSpace(space, n: 2)).Keys.ToArray());
            CollectionAssert.AreEqual(new[] { "b", "a" }, ((Dict)VectorUtils.BatchDifferingSpaces(new Space[] { space, space.DeepCopy() })).Keys.ToArray());
        }

        /// <summary>Two boxes of different bounds batch into one box of both bounds, whose sample iterates into members of each.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/utils/test_space_utils.py#L171">tests/vector/utils/test_space_utils.py (test_varying_spaces)</see>.</remarks>
        [TestMethod]
        public void TestVaryingSpaces() {
            Space[] spaces = {
                new Box(0, 1, new Shape(2), np.float32),
                new Box(2, np.array(new[] { 3f, 5f })),
            };
            var expectedSpace = new Box(np.array(new long[,] { { 0, 0 }, { 2, 2 } }), np.array(new long[,] { { 1, 1 }, { 3, 5 } }));

            Space batchedSpace = VectorUtils.BatchDifferingSpaces(spaces);
            Assert.AreEqual(expectedSpace, batchedSpace);

            object? batchSamples = batchedSpace.Sample();
            List<object?> items = VectorUtils.Iterate(batchedSpace, batchSamples).ToList();
            Assert.AreEqual(spaces.Length, items.Count);
            for (int i = 0; i < spaces.Length; i++)
                Assert.IsTrue(spaces[i].Contains(items[i]), $"item {i}");
        }

        /// <summary><c>batch_space(space, n)</c> equals <c>batch_differing_spaces</c> of <c>n</c> copies.</summary>
        /// <param name="index">The space's index in the catalog.</param>
        /// <param name="n">The batch size.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/utils/test_space_utils.py#L195">tests/vector/utils/test_space_utils.py (test_batch_differing_space_vs_batch_space)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(SpacesAndN13))]
        public void TestBatchDifferingSpaceVsBatchSpace(int index, int n) {
            Space space = TestingSpaces.All[index]();
            Space batchedSpace = VectorUtils.BatchSpace(space, n);
            Space batchedSpaces = VectorUtils.BatchDifferingSpaces(Enumerable.Range(0, n).Select(_ => space.DeepCopy()).ToArray());

            Assert.AreEqual(batchedSpace, batchedSpaces, $"batched_space={batchedSpace}, batched_spaces={batchedSpaces}");
        }

        /// <summary><c>batch_differing_spaces</c> is deterministic, as <see cref="TestBatchSpaceDeterministic"/> checks for <c>batch_space</c>.</summary>
        /// <param name="index">The space's index in the catalog.</param>
        /// <param name="n">The batch size.</param>
        /// <param name="baseSeed">The seed.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/utils/test_space_utils.py#L205">tests/vector/utils/test_space_utils.py (test_batch_differing_spaces_deterministic)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(SpacesN125AndSeeds))]
        public void TestBatchDifferingSpacesDeterministic(int index, int n, int baseSeed) {
            Space spaceA = TestingSpaces.All[index]();
            spaceA.Seed(baseSeed);
            Space spaceB = spaceA.DeepCopy();
            TestingUtils.AssertRngEqual(spaceA.NpRandom, spaceB.NpRandom);
            Assert.AreNotSame(spaceA.NpRandom, spaceB.NpRandom);

            // Batch the spaces and check that the np_random are not reference equal
            Space spaceABatched = VectorUtils.BatchDifferingSpaces(Enumerable.Repeat(spaceA, n).ToArray());
            Space spaceBBatched = VectorUtils.BatchDifferingSpaces(Enumerable.Repeat(spaceB, n).ToArray());
            TestingUtils.AssertRngEqual(spaceABatched.NpRandom, spaceBBatched.NpRandom);
            Assert.AreNotSame(spaceABatched.NpRandom, spaceBBatched.NpRandom);
            // Create that the batched space is not reference equal to the origin spaces
            Assert.AreNotSame(spaceA.NpRandom, spaceABatched.NpRandom);

            // Check that batched space a and b random number generator are not effected by the original space
            spaceA.Sample();
            AssertSameItems(spaceABatched, spaceABatched.Sample(), spaceBBatched, spaceBBatched.Sample());
        }

        /// <summary>The dtype of a batch of differing <c>Discrete</c> spaces is the widest of theirs.</summary>
        /// <param name="case">Which of Gymnasium's four cases.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/utils/test_space_utils.py#L235">tests/vector/utils/test_space_utils.py (test_batch_differing_discrete_spaces_dtype)</see>.</remarks>
        [TestMethod]
        [DataRow(0)]
        [DataRow(1)]
        [DataRow(2)]
        [DataRow(3)]
        public void TestBatchDifferingDiscreteSpacesDtype(int @case) {
            (Space[] Spaces, NPTypeCode Expected) setup = @case switch {
                0 => (new Space[] { new Discrete(2), new Discrete(3) }, NPTypeCode.Int64),
                1 => (new Space[] { new Discrete(2, dtype: np.int16), new Discrete(3, dtype: np.int32) }, NPTypeCode.Int32),
                2 => (new Space[] { new Discrete(2, dtype: np.int32), new Discrete(3), new Discrete(1, dtype: np.int8) }, NPTypeCode.Int64),
                _ => (new Space[] { new Discrete(1, dtype: np.int8), new Discrete(5, dtype: np.int16) }, NPTypeCode.Int16),
            };
            var multiDiscrete = (MultiDiscrete)VectorUtils.BatchDifferingSpaces(setup.Spaces);
            Assert.AreEqual(setup.Expected, multiDiscrete.DType!.typecode);
        }

        /// <summary>The two batches' samples iterate into equal items, pair by pair (<c>zip(strict=True)</c> of <c>data_equivalence</c>).</summary>
        /// <param name="spaceA">The first batched space.</param>
        /// <param name="sampleA">Its sample.</param>
        /// <param name="spaceB">The second batched space.</param>
        /// <param name="sampleB">Its sample.</param>
        private static void AssertSameItems(Space spaceA, object? sampleA, Space spaceB, object? sampleB) {
            List<object?> itemsA = VectorUtils.Iterate(spaceA, sampleA).ToList();
            List<object?> itemsB = VectorUtils.Iterate(spaceB, sampleB).ToList();
            Assert.AreEqual(itemsA.Count, itemsB.Count);
            for (int i = 0; i < itemsA.Count; i++)
                Assert.IsTrue(EnvChecker.DataEquivalence(itemsA[i], itemsB[i]), $"{spaceA}: item {i}");
        }

        /// <summary>Every catalog index with each of the sizes.</summary>
        /// <param name="sizes">The batch sizes.</param>
        /// <returns>The rows <c>(index, n)</c>.</returns>
        private static IEnumerable<object[]> SpacesWith(params int[] sizes) =>
            from index in Enumerable.Range(0, TestingSpaces.All.Length) from n in sizes select new object[] { index, n };
    }
}
