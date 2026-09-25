using System;
using System.Collections.Generic;
using Gym.Environments.Envs.Aether;
using Gym.Spaces;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Spaces {
    /// <summary>
    ///     Pins <see cref="Discrete"/> after the move to NumSharp master and <see langword="long"/> elements: action
    ///     conversion (<see cref="Discrete.ToInt64"/>), membership with a start (B-02), masked sampling (B-04), and 0-d int64
    ///     samples (B-01, B-03's dtype).
    /// </summary>
    [TestClass]
    public class DiscreteTests {
        /// <summary>An enum backed by <see cref="byte"/>: <c>(int)(object)x</c> threw for it.</summary>
        private enum ByteAction : byte { Zero = 0, Three = 3 }

        /// <summary>An enum backed by <see cref="ulong"/>, with a member above <see cref="long.MaxValue"/>.</summary>
        private enum ULongAction : ulong { Two = 2, Huge = ulong.MaxValue }

        /// <summary>
        ///     Every integer width, <see cref="bool"/> (a Python int subclass) and enum members convert to the value they
        ///     denote, whichever type boxed them.
        /// </summary>
        [TestMethod]
        public void ToInt64_BoxedIntegersBoolsAndEnums_Convert()
        {
            var cases = new List<(object Action, long Expected)> {
                (3, 3L), (3L, 3L), ((short)3, 3L), ((sbyte)-3, -3L), ((byte)3, 3L), ((ushort)3, 3L), (3u, 3L), (3ul, 3L),
                (long.MinValue, long.MinValue), ((ulong)long.MaxValue, long.MaxValue),
                (true, 1L), (false, 0L),
                (ByteAction.Three, 3L), (ULongAction.Two, 2L), (LunarLanderDiscreteActions.FireMainEngine, 2L),
            };
            foreach (var (action, expected) in cases)
                Assert.AreEqual(expected, Discrete.ToInt64(action), $"{action} ({action.GetType().Name})");
        }

        /// <summary>
        ///     0-d arrays of any integer dtype convert, including the 0-d int64 array <see cref="Discrete.Sample"/> returns, which
        ///     is what an env receives when it steps with a sampled action.
        /// </summary>
        [TestMethod]
        public void ToInt64_ZeroDimensionalIntegerArrays_Convert()
        {
            Assert.AreEqual(3L, Discrete.ToInt64(NDArray.Scalar<int>(3)));
            Assert.AreEqual(-3L, Discrete.ToInt64(NDArray.Scalar<long>(-3L)));
            Assert.AreEqual(3L, Discrete.ToInt64(NDArray.Scalar<ulong>(3ul)));
            Assert.AreEqual(3L, Discrete.ToInt64(NDArray.Scalar<byte>(3)));
            Assert.AreEqual(3L, Discrete.ToInt64(new Discrete(4, start: 3).Sample(np.array(new sbyte[] { 1, 0, 0, 0 }))));
        }

        /// <summary>
        ///     Unsigned 64-bit values above <see cref="long.MaxValue"/> throw instead of wrapping: NumSharp's own
        ///     <c>(long)</c> conversion turns a 0-d uint64 <see cref="ulong.MaxValue"/> into -1.
        /// </summary>
        [TestMethod]
        public void ToInt64_UnsignedAboveInt64_ThrowsOverflow()
        {
            Assert.ThrowsExactly<OverflowException>(() => Discrete.ToInt64(ulong.MaxValue));
            Assert.ThrowsExactly<OverflowException>(() => Discrete.ToInt64(NDArray.Scalar<ulong>(ulong.MaxValue)));
            Assert.ThrowsExactly<OverflowException>(() => Discrete.ToInt64(ULongAction.Huge));
        }

        /// <summary>
        ///     Floats, strings, <see langword="null"/>, arrays that are not 0-d, and 0-d arrays of a non-integer dtype (including
        ///     boolean, which NumPy does not count as an integer dtype) are not discrete actions.
        /// </summary>
        [TestMethod]
        public void ToInt64_NonIntegers_ThrowArgumentException()
        {
            var rejected = new object[] {
                2.5, 2.5f, 2m, "3", null, np.array(new[] { 1L, 2L }), NDArray.Scalar<double>(3.0), NDArray.Scalar<bool>(true),
            };
            foreach (var action in rejected)
                Assert.ThrowsExactly<ArgumentException>(() => Discrete.ToInt64(action), $"{action ?? "null"}");
        }

        /// <summary>Defect B-02: membership honours <see cref="Discrete.Start"/>; <c>Discrete(3, start: 5)</c> holds 5, 6 and 7.</summary>
        [TestMethod]
        public void Contains_WithStart_HonoursTheOffset()
        {
            var space = new Discrete(3, start: 5);

            Assert.IsTrue(space.Contains(5L) && space.Contains(6L) && space.Contains(7L));
            Assert.IsFalse(space.Contains(4L), "4 is below start.");
            Assert.IsFalse(space.Contains(8L), "8 is start + n.");
            Assert.IsFalse(space.Contains(0L), "0 is not an element once start is 5.");
            Assert.IsTrue(space.Contains((object) 6), "A boxed int converts like any integer.");
            Assert.IsTrue(space.Contains(NDArray.Scalar<long>(7L)), "A 0-d int64 array is a NumPy scalar.");
        }

        /// <summary>
        ///     Membership returns <see langword="false"/> for anything that is not an integer, and for an unsigned value that
        ///     does not fit int64, as Gymnasium does; under NumSharp.Lite it threw <see cref="NotSupportedException"/>.
        /// </summary>
        [TestMethod]
        public void Contains_NonIntegersAndOverflow_AreFalse()
        {
            var space = new Discrete(4);

            Assert.IsFalse(space.Contains("a"));
            Assert.IsFalse(space.Contains(1.0));
            Assert.IsFalse(space.Contains((object) null));
            Assert.IsFalse(space.Contains(ulong.MaxValue));
            Assert.IsFalse(space.Contains(ULongAction.Huge));
            Assert.IsTrue(space.Contains(ByteAction.Three), "An enum not backed by int converts.");
            Assert.IsTrue(space.Contains(true), "A bool is the Python int 1.");
        }

        /// <summary>
        ///     The upper bound <c>Start + N</c> is exact even when it exceeds <see cref="long.MaxValue"/>: with wrapping int64
        ///     arithmetic, <c>1 + long.MaxValue</c> would become <see cref="long.MinValue"/> and reject every element.
        /// </summary>
        [TestMethod]
        public void Contains_UpperBoundBeyondInt64_DoesNotWrap()
        {
            var space = new Discrete(long.MaxValue, start: 1);

            Assert.IsTrue(space.Contains(long.MaxValue));
            Assert.IsTrue(space.Contains(1L));
            Assert.IsFalse(space.Contains(0L));
        }

        /// <summary>
        ///     Defects B-01 and B-03's dtype: <see cref="Discrete.Sample"/> works and returns a 0-d int64 array inside
        ///     <c>[Start, Start + N)</c>; the space's default dtype is int64.
        /// </summary>
        [TestMethod]
        public void Sample_ReturnsZeroDimensionalInt64WithinTheSpace()
        {
            var space = new Discrete(3, seed: 1, start: 5);

            Assert.AreEqual(np.int64, space.DType);
            for (int i = 0; i < 100; i++)
            {
                using var sample = space.Sample();
                Assert.AreEqual(0, sample.ndim);
                Assert.AreEqual(np.int64, sample.dtype);
                Assert.IsTrue(space.Contains(sample), $"{sample} is outside [5, 8).");
            }
        }

        /// <summary>
        ///     <see cref="Discrete.N"/> is <see langword="long"/>: a space with more than <see cref="int.MaxValue"/> elements
        ///     constructs and samples inside its range.
        /// </summary>
        [TestMethod]
        public void Sample_LargerThanInt32_StaysInRange()
        {
            var space = new Discrete(1L << 40, seed: 2);

            for (int i = 0; i < 20; i++)
            {
                using var sample = space.Sample();
                long value = (long) sample;
                Assert.IsTrue(value >= 0 && value < 1L << 40, $"{value} is outside [0, 2^40).");
            }
        }

        /// <summary>
        ///     Defect B-04: masked samples come from the allowed elements only, all of them are reachable, and a mask that
        ///     allows nothing returns <see cref="Discrete.Start"/>.
        /// </summary>
        [TestMethod]
        public void Sample_WithMask_DrawsOnlyAllowedElements()
        {
            var space = new Discrete(4, seed: 3, start: 10);
            var mask = np.array(new sbyte[] { 0, 0, 1, 1 });
            var seen = new HashSet<long>();

            for (int i = 0; i < 200; i++)
            {
                using var sample = space.Sample(mask);
                seen.Add((long) sample);
            }

            CollectionAssert.AreEquivalent(new[] { 12L, 13L }, new List<long>(seen), "Only the masked-in elements 12 and 13 may be drawn.");
            using var none = space.Sample(np.zeros(new Shape(4), np.int8));
            Assert.AreEqual(10L, (long) none, "A mask that allows nothing returns start.");
        }

        /// <summary>Seeding with the same seed replays the same samples, whether the seed is given at construction or later.</summary>
        [TestMethod]
        public void Seed_SameSeed_ReplaysTheSameSamples()
        {
            var constructed = new Discrete(1000, seed: 42);
            var reseeded = new Discrete(1000, seed: 7);
            reseeded.Seed(42);

            for (int i = 0; i < 20; i++)
            {
                using var a = constructed.Sample();
                using var b = reseeded.Sample();
                Assert.AreEqual((long) a, (long) b);
            }
        }
    }
}
