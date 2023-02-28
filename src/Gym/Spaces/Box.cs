using System;
using System.Diagnostics;
using System.Linq;
using System.Reflection.Metadata;
using System.Text.Json;
using NumSharp;

namespace Gym.Spaces {
    public enum BoundedMannerEnum
    {
        Both,
        Below,
        Above
    }
    public class Box : Space, IEquatable<Box> {
        protected NumPyRandom RandomState;
        public NDArray Low { get; }
        public NDArray High { get; }
        public NDArray BoundedLow { get; private set;  }
        public NDArray BoundedHigh { get; private set; }
        /// <summary>Initializes a new instance of the <see cref="T:System.Object" /> class.</summary>
        public Box(int low, int high, Shape shape, Type dType = null, int seed = -1) : this((float) low, (float) high, shape, dType, seed, null) { }
        public Box(int low, int high, Shape shape, Type dType = null, NumPyRandom random_state = null) : this((float)low, (float)high, shape, dType, -1, random_state) { }

        public Box(float low, float high, Shape shape, Type dType = null, int seed = -1, NumPyRandom random_state = null) : base(shape, (dType = dType ?? np.float32)) {
            if (Equals(shape, null)) throw new ArgumentNullException(nameof(shape));

            Low = (low + np.zeros(shape, dType)).astype(dType);
            High = (high + np.zeros(shape, dType)).astype(dType);
            RandomState = seed != -1 ? np.random.RandomState(seed) : random_state ?? np.random;
            CheckBounded();
        }

        /// <summary>Initializes a new instance of the <see cref="T:System.Object" /> class.</summary>
        public Box(NDArray low, NDArray high, Type dType = null, int seed = -1, NumPyRandom random_state = null) : base(null, (dType = dType ?? np.float32)) {
            if (Equals(low, null)) throw new ArgumentNullException(nameof(low));
            if (Equals(high, null)) throw new ArgumentNullException(nameof(high));
            Debug.Assert(low.shape.SequenceEqual(high.shape));
            Shape = low.shape;
            Low = low.astype(dType);
            High = high.astype(dType);
            RandomState = seed != -1 ? np.random.RandomState(seed) : random_state ?? np.random;
            CheckBounded();
        }

        private void CheckBounded() {
            NDArray neginf = np.full((float)-np.inf, Low.shape);
            BoundedLow = (Low > neginf);
            NDArray posinf = np.full((float)np.inf, Low.shape);
            BoundedHigh = (High < posinf);
        }

        public bool IsBounded(BoundedMannerEnum manner)
        {
            bool below = np.all(BoundedLow);
            bool above = np.all(BoundedHigh);
            switch (manner)
            {
                case BoundedMannerEnum.Both:
                    return below & above;
                case BoundedMannerEnum.Above:
                    return above;
                case BoundedMannerEnum.Below:
                    return below;
            }
            throw new ArgumentException("manner", "Unsupported BoundedMannerEnum value.");
        }

        public override NDArray Sample(NDArray mask = null) {
            if (!Equals(mask, null))
            {
                throw new NotSupportedException("Box.sample cannot be provided a mask.");
            }
            NDArray unbounded = ~BoundedLow & ~BoundedHigh;
            NDArray upp_bounded = ~BoundedLow & BoundedHigh;
            NDArray low_bounded = BoundedLow & ~BoundedHigh;
            NDArray bounded = BoundedLow & BoundedHigh;

            NDArray sample = np.empty(Shape);

            sample[unbounded] = RandomState.normal(0.5f, 1.0f, unbounded[unbounded].shape);
            sample[low_bounded] = RandomState.exponential(1.0f, low_bounded[low_bounded].shape) + Low[low_bounded];
            sample[upp_bounded] = RandomState.exponential(1.0f, upp_bounded[upp_bounded].shape) + High[upp_bounded];
            sample[bounded] = RandomState.uniform(Low[bounded], High[bounded], bounded[bounded].shape);
            if (DType == np.int32 || DType == np.uint32 || DType == np.@byte)
            {
                sample = np.floor(sample);
            }
            return sample.astype(DType); // RandomState.uniform(Low, High, DType);
        }

        public override bool Contains(object ndArray) {
            if (ndArray is NDArray x)
                return x.shape == Shape && x >= Low && x <= High;
            throw new NotSupportedException(ndArray?.ToString() ?? nameof(ndArray));
        }

        public override void Seed(int seed) {
            RandomState = np.random.RandomState(seed);
        }

        #region Object Overrides

        /// <summary>Returns a string that represents the current object.</summary>
        /// <returns>A string that represents the current object.</returns>
        public override string ToString() {
            return "Box" + Shape;
        }

        /// <summary>Indicates whether the current object is equal to another object of the same type.</summary>
        /// <param name="other">An object to compare with this object.</param>
        /// <returns>
        /// <see langword="true" /> if the current object is equal to the <paramref name="other" /> parameter; otherwise, <see langword="false" />.</returns>
        public bool Equals(Box other) {
            if (ReferenceEquals(null, other)) return false;
            if (ReferenceEquals(this, other)) return true;
            return Equals(Low, other.Low) && Equals(High, other.High);
        }

        /// <summary>Determines whether the specified object is equal to the current object.</summary>
        /// <param name="obj">The object to compare with the current object. </param>
        /// <returns>
        /// <see langword="true" /> if the specified object  is equal to the current object; otherwise, <see langword="false" />.</returns>
        public override bool Equals(object obj) {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return Equals((Box) obj);
        }

        /// <summary>Serves as the default hash function. </summary>
        /// <returns>A hash code for the current object.</returns>
        public override int GetHashCode() {
            unchecked {
                return ((Low != null ? Low.GetHashCode() : 0) * 397) ^ (High != null ? High.GetHashCode() : 0);
            }
        }

        /// <summary>Returns a value that indicates whether the values of two <see cref="T:Gym.Spaces.Box" /> objects are equal.</summary>
        /// <param name="left">The first value to compare.</param>
        /// <param name="right">The second value to compare.</param>
        /// <returns>true if the <paramref name="left" /> and <paramref name="right" /> parameters have the same value; otherwise, false.</returns>
        public static bool operator ==(Box left, Box right) {
            return Equals(left, right);
        }

        /// <summary>Returns a value that indicates whether two <see cref="T:Gym.Spaces.Box" /> objects have different values.</summary>
        /// <param name="left">The first value to compare.</param>
        /// <param name="right">The second value to compare.</param>
        /// <returns>true if <paramref name="left" /> and <paramref name="right" /> are not equal; otherwise, false.</returns>
        public static bool operator !=(Box left, Box right) {
            return !Equals(left, right);
        }

        #endregion
    }
}