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

namespace Gym.Tests.Gymnasium.Vector.Utils {
    /// <summary>Gymnasium's shared-memory tests (<c>tests/vector/utils/test_shared_memory.py</c>), run against <see cref="VectorUtils"/>.</summary>
    /// <remarks>
    ///     <para>
    ///     Gymnasium also parametrizes the round trip by <c>multiprocessing</c> start method (<c>fork</c>, <c>spawn</c>);
    ///     Gym.NET's buffer is an ordinary array shared by threads, so there is no start method to vary. A typed C# call
    ///     cannot pass a non-space (<c>test_non_space</c>): the port checks the <see langword="null"/> a caller could still pass.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/utils/test_shared_memory.py">tests/vector/utils/test_shared_memory.py</see>.</para>
    /// </remarks>
    [TestClass]
    public class TestSharedMemory {
        /// <summary>Every testing space, with 1 and 8 sub-envs.</summary>
        public static IEnumerable<object[]> SpacesAndCounts =>
            from index in Enumerable.Range(0, TestingSpaces.All.Length) from num in new[] { 1, 8 } select new object[] { index, num };

        /// <summary>
        ///     Test the shared memory functions, create, read and write for all testing spaces: the samples written read back as
        ///     a member of the batched space whose items are the samples; a dynamic shape is refused.
        /// </summary>
        /// <param name="spaceIndex">The space in <see cref="TestingSpaces.All"/>.</param>
        /// <param name="num">The number of sub-envs.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/utils/test_shared_memory.py#L24">tests/vector/utils/test_shared_memory.py (test_shared_memory_create_read_write)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(SpacesAndCounts))]
        public void TestSharedMemoryCreateReadWrite(int spaceIndex, int num) {
            using Space space = TestingSpaces.All[spaceIndex]();
            using Space batchedSpace = VectorUtils.BatchSpace(space, num);
            object sharedMemory;
            try {
                sharedMemory = VectorUtils.CreateSharedMemory(space, num);
            } catch (TypeError err) {
                StringAssert.Contains(err.Message, "has a dynamic shape so its not possible to make a static shared memory.");
                // Gymnasium skips the space here; the refusal it just checked is the whole case, so the port passes it
                // instead of reporting a skip for every dynamic space.
                return;
            }

            try {
                object?[] samples = Enumerable.Range(0, num).Select(_ => space.Sample()).ToArray();
                for (int i = 0; i < samples.Length; i++)
                    VectorUtils.WriteToSharedMemory(space, i, samples[i], sharedMemory);

                object? readSamples = VectorUtils.ReadFromSharedMemory(space, sharedMemory, num);
                Assert.IsTrue(batchedSpace.Contains(readSamples), $"{PyBuiltins.Repr(readSamples)} not in {batchedSpace}");
                List<object?> items = VectorUtils.Iterate(batchedSpace, readSamples).ToList();
                Assert.AreEqual(samples.Length, items.Count);
                for (int i = 0; i < samples.Length; i++)
                    Assert.IsTrue(EnvChecker.DataEquivalence(items[i], samples[i]), $"sample {i}: {PyBuiltins.Repr(items[i])} against {PyBuiltins.Repr(samples[i])}");
            } finally {
                VectorUtils.ReleaseSharedMemory(sharedMemory);
            }
        }

        /// <summary>Test dtypes without an <c>array</c> module typecode (float16) roundtrip through shared memory.</summary>
        /// <remarks>
        ///     NumPy's float128 does not exist in NumSharp, so float16 is the case.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/utils/test_shared_memory.py#L58">tests/vector/utils/test_shared_memory.py (test_shared_memory_no_array_typecode_dtypes)</see>.
        /// </remarks>
        [TestMethod]
        public void TestSharedMemoryNoArrayTypecodeDtypes() {
            using var space = new Box(-1.0, 1.0, new Shape(2, 3), np.float16);
            const int num = 8;

            object sharedMemory = VectorUtils.CreateSharedMemory(space, num);
            try {
                Generator rng = np.random.default_rng(123);
                var samples = new List<NDArray>();
                for (int i = 0; i < num; i++) {
                    using NDArray uniform = rng.uniform(-1.0, 1.0, new Shape(2, 3));
                    samples.Add(uniform.astype(np.float16));
                }

                for (int i = 0; i < samples.Count; i++)
                    VectorUtils.WriteToSharedMemory(space, i, samples[i], sharedMemory);

                using var readSamples = (NDArray)VectorUtils.ReadFromSharedMemory(space, sharedMemory, num)!;
                Assert.IsTrue(readSamples.typecode == NPTypeCode.Half);
                CollectionAssert.AreEqual(new long[] { num, 2, 3 }, readSamples.shape);
                for (int i = 0; i < num; i++) {
                    using NDArray readSample = readSamples[i];
                    Assert.IsTrue(EnvChecker.DataEquivalence(readSample, samples[i]), $"sample {i}");
                }

                foreach (NDArray sample in samples)
                    sample.Dispose();
            } finally {
                VectorUtils.ReleaseSharedMemory(sharedMemory);
            }
        }

        /// <summary>Test using custom spaces for shared memory functions: each refuses with Gymnasium's message naming the space's class.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/utils/test_shared_memory.py#L84">tests/vector/utils/test_shared_memory.py (test_custom_space)</see>.</remarks>
        [TestMethod]
        public void TestCustomSpace() {
            using var space = new TestingUtils.CustomSpace();
            string type = PyBuiltins.TypeRepr(space);

            var error = Assert.ThrowsExactly<CustomSpaceError>(() => VectorUtils.CreateSharedMemory(space));
            Assert.AreEqual($"Space of type `{type}` doesn't have an registered `create_shared_memory` function. Register `{type}` for `create_shared_memory` to support it.", error.Message);

            error = Assert.ThrowsExactly<CustomSpaceError>(() => VectorUtils.ReadFromSharedMemory(space, new object(), 1));
            Assert.AreEqual($"Space of type `{type}` doesn't have an registered `read_from_shared_memory` function. Register `{type}` for `read_from_shared_memory` to support it.", error.Message);

            error = Assert.ThrowsExactly<CustomSpaceError>(() => VectorUtils.WriteToSharedMemory(space, 1, null, new object()));
            Assert.AreEqual($"Space of type `{type}` doesn't have an registered `write_to_shared_memory` function. Register `{type}` for `write_to_shared_memory` to support it.", error.Message);
        }

        /// <summary>
        ///     Test the use of non-space types on the shared memory functions: the typed signature takes only spaces, and a
        ///     <see langword="null"/> is refused.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/utils/test_shared_memory.py#L111">tests/vector/utils/test_shared_memory.py (test_non_space)</see>.</remarks>
        [TestMethod]
        public void TestNonSpace() {
            Assert.ThrowsExactly<ArgumentNullException>(() => VectorUtils.CreateSharedMemory(null!));
            Assert.ThrowsExactly<ArgumentNullException>(() => VectorUtils.ReadFromSharedMemory(null!, new object(), 1));
            Assert.ThrowsExactly<ArgumentNullException>(() => VectorUtils.WriteToSharedMemory(null!, 1, null, new object()));
        }
    }
}
