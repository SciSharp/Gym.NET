using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;
using Shape = NumSharp.Shape;
using Box = Gym.Spaces.Box;
using NumSharp;

namespace Gym.Tests.Spaces
{
    [TestClass]
    public class BoxTest
    {
        [TestMethod]
        public void TestBoxBoundedTest()
        {
            Box box = new Box(-5.0f, 5.0f, np.float32);
            Assert.IsTrue(box.IsBounded(Gym.Spaces.BoundedMannerEnum.Both), "Box should be bounded at both boundaries.");
            
            box = new Box(-np.inf, 5.0f, np.float32);
            Assert.IsFalse(box.IsBounded(Gym.Spaces.BoundedMannerEnum.Below), "Box should be unbound at the low bound.");
            Assert.IsTrue(box.IsBounded(Gym.Spaces.BoundedMannerEnum.Above), "Box should be bounded at the high bound.");
            Assert.IsFalse(box.IsBounded(Gym.Spaces.BoundedMannerEnum.Both), "Box should not be bounded.");

            box = new Box(5.0f, np.inf, np.float32);
            Assert.IsFalse(box.IsBounded(Gym.Spaces.BoundedMannerEnum.Above), "Box should be unbound at the high bound.");
            Assert.IsTrue(box.IsBounded(Gym.Spaces.BoundedMannerEnum.Below), "Box should be bounded at the low bound.");
            Assert.IsFalse(box.IsBounded(Gym.Spaces.BoundedMannerEnum.Both), "Box should not be bounded.");

            box = new Box(-np.inf, np.inf, np.float32);
            Assert.IsFalse(box.IsBounded(Gym.Spaces.BoundedMannerEnum.Above), "Box should be unbound at the high bound.");
            Assert.IsFalse(box.IsBounded(Gym.Spaces.BoundedMannerEnum.Below), "Box should be unbound at the low bound.");
            Assert.IsFalse(box.IsBounded(Gym.Spaces.BoundedMannerEnum.Both), "Box should not be bounded.");
        }

        /// <summary>
        ///     A 0-d box samples a 0-d value inside its bounds. The explicit <c>(float)</c> is required by NumSharp, whose
        ///     array-to-scalar conversions are explicit and accept 0-d arrays only (NumPy's <c>float(arr)</c>).
        /// </summary>
        [TestMethod]
        public void TestBoxBoundedSampling()
        {
            Box box = new Box(-5.0, 5.0);
            float sample = (float)box.Sample(null);
            Assert.IsTrue(sample >= -5.0 && sample <= 5.0, "Box sampling should be on the range [-5.0,5.0]");
        }

        /// <summary>
        ///     Defects B-05 and B-06: sampling and membership used to throw for every box that was not 0-d. A vector sample
        ///     keeps the box's shape and dtype, stays inside the bounds, and is contained by the box that drew it.
        /// </summary>
        [TestMethod]
        public void Sample_VectorBox_HasShapeAndDTypeAndIsContained()
        {
            var box = new Box(-1f, 1f, new Shape(3), np.float32, seed: 7);
            using var sample = box.Sample();

            Assert.AreEqual(new Shape(3), sample.Shape);
            Assert.AreEqual(np.float32, sample.dtype);
            Assert.IsTrue(np.all(sample >= -1f) && np.all(sample <= 1f), $"Sample {sample} left [-1, 1].");
            Assert.IsTrue(box.Contains(sample));
        }

        /// <summary>
        ///     Each element draws from the distribution its own bounds select (normal, shifted exponential up or down, uniform),
        ///     all in one sample: the masked assignments of the four categories must not overwrite one another.
        /// </summary>
        [TestMethod]
        public void Sample_MixedBounds_DrawsEachElementWithinItsOwnBounds()
        {
            var low = np.array(new[] { float.NegativeInfinity, 0f, float.NegativeInfinity, -1f });
            var high = np.array(new[] { float.PositiveInfinity, float.PositiveInfinity, 0f, 1f });
            var box = new Box(low, high, np.float32, seed: 11);

            for (int i = 0; i < 100; i++)
            {
                using var sample = box.Sample();
                Assert.IsTrue(float.IsFinite(sample.GetSingle(0)), "The unbounded element must be a finite normal draw.");
                Assert.IsTrue(sample.GetSingle(1) >= 0f, "The element bounded below by 0 must be >= 0.");
                Assert.IsTrue(sample.GetSingle(2) <= 0f, "The element bounded above by 0 must be <= 0.");
                Assert.IsTrue(sample.GetSingle(3) >= -1f && sample.GetSingle(3) <= 1f, "The bounded element must stay in [-1, 1].");
            }
        }

        /// <summary>
        ///     Membership rejects a point of another shape and a point outside the bounds, returning <see langword="false"/>
        ///     rather than throwing a broadcasting error.
        /// </summary>
        [TestMethod]
        public void Contains_WrongShapeOrOutOfBounds_IsFalse()
        {
            var box = new Box(-1f, 1f, new Shape(3), np.float32);

            Assert.IsFalse(box.Contains(np.zeros(new Shape(2), np.float32)), "A (2,) point cannot belong to a (3,) box.");
            Assert.IsFalse(box.Contains(np.full(new Shape(3), 2f, np.float32)), "2 lies above the upper bound 1.");
            Assert.IsFalse(box.Contains(np.full(new Shape(3), -2f, np.float32)), "-2 lies below the lower bound -1.");
            Assert.IsTrue(box.Contains(np.zeros(new Shape(3), np.float32)));
        }

        /// <summary>
        ///     Defect B-07: integer boxes could not be constructed under NumSharp.Lite (a mixed-dtype comparison failed). They
        ///     construct, report themselves bounded, and sample integers of their dtype within the bounds.
        /// </summary>
        [TestMethod]
        public void IntegerBox_ConstructsAndSamplesIntegersWithinBounds()
        {
            // Float literals: with int literals the two integer overloads are ambiguous (CS0121).
            var box = new Box(0f, 5f, new Shape(2000), np.int32, seed: 3);
            using var sample = box.Sample();

            Assert.IsTrue(box.IsBounded(Gym.Spaces.BoundedMannerEnum.Both));
            Assert.AreEqual(np.int32, sample.dtype);
            Assert.IsTrue(np.all(sample >= 0) && np.all(sample <= 5), "Integer samples must stay within [0, 5].");
        }

        /// <summary>
        ///     Every integer dtype is floored before the cast, as Gymnasium's <c>dtype.kind in ["i", "u", "b"]</c> requires.
        ///     A plain cast truncates toward zero and could never produce the lower bound -5 from a draw in (-5, -4); flooring
        ///     does, so the minimum of many samples reaches -5 only when the dtype was recognized as an integer.
        /// </summary>
        /// <param name="dtypeName">A NumPy dtype name; int8 is the type NumSharp's <c>np.@byte</c> denotes.</param>
        [TestMethod]
        [DataRow("int8")]
        [DataRow("int16")]
        [DataRow("int32")]
        [DataRow("int64")]
        public void Sample_SignedIntegerBox_FloorsNegativeDraws(string dtypeName)
        {
            DType dtype = np.dtype(dtypeName);
            var box = new Box(-5f, 5f, new Shape(1000), dtype, seed: 5);
            using var sample = box.Sample();

            Assert.AreEqual(dtype, sample.dtype);
            Assert.AreEqual(-5L, (long)np.amin(sample), "Flooring must map draws in (-5, -4) to -5.");
        }

        /// <summary>
        ///     Equality compares shape, dtype and bounds without broadcasting: boxes of different shapes are unequal instead of
        ///     throwing <see cref="IncorrectShapeException"/>, and a different dtype makes boxes unequal.
        /// </summary>
        [TestMethod]
        public void Equals_DifferentShapeOrDType_IsFalse()
        {
            var box2 = new Box(-1f, 1f, new Shape(2), np.float32);
            var box3 = new Box(-1f, 1f, new Shape(3), np.float32);
            var box3Double = new Box(-1f, 1f, new Shape(3), np.float64);

            Assert.IsFalse(box2 == box3, "Boxes of shapes (2,) and (3,) must be unequal.");
            Assert.IsFalse(box3.Equals(box3Double), "Boxes of different dtypes must be unequal.");
        }

        /// <summary>
        ///     Boxes with the same shape, dtype and bounds are equal and hash alike; hashing must not throw although
        ///     <see cref="NDArray.GetHashCode"/> does.
        /// </summary>
        [TestMethod]
        public void Equals_SameDefinition_IsTrueAndHashesMatch()
        {
            var first = new Box(-1f, 1f, new Shape(3), np.float32, seed: 1);
            var second = new Box(-1f, 1f, new Shape(3), np.float32, seed: 2);

            Assert.IsTrue(first == second, "The generator is not part of a box's identity.");
            Assert.AreEqual(first.GetHashCode(), second.GetHashCode());
        }

        /// <summary>
        ///     Bounds of different shapes are rejected with Gymnasium's <c>ValueError</c> message in every build
        ///     configuration; a <c>Debug.Assert</c> used to catch them in Debug builds only.
        /// </summary>
        [TestMethod]
        public void Constructor_MismatchedBoundShapes_ThrowsValueError()
        {
            var error = Assert.ThrowsExactly<ValueError>(() => new Box(np.zeros(new Shape(3)), np.ones(new Shape(2))));

            Assert.AreEqual("Box low.shape and high.shape don't match, low.shape=(3,), high.shape=(2,)", error.Message);
        }

        /// <summary>A box cannot be sampled under a mask, as in Gymnasium.</summary>
        [TestMethod]
        public void Sample_WithMask_Throws()
        {
            var box = new Box(-1f, 1f, new Shape(3), np.float32);

            Assert.ThrowsExactly<NotSupportedException>(() => box.Sample(np.ones(new Shape(3))));
        }
    }
}
