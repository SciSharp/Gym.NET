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

        /// <summary>
        ///     Creates a box whose shape and per-element bounds come from <paramref name="low"/> and
        ///     <paramref name="high"/>; use this overload when bounds differ per dimension.
        /// </summary>
        /// <param name="low">Inclusive lower bounds; its shape becomes the space's shape. Cast to <paramref name="dType"/>.</param>
        /// <param name="high">Inclusive upper bounds with the same shape as <paramref name="low"/>. Cast to <paramref name="dType"/>.</param>
        /// <param name="dType">Element type of the space; <see langword="null"/> means float32.</param>
        /// <param name="seed">Seed for a private generator; -1 means use <paramref name="random_state"/> instead.</param>
        /// <param name="random_state">
        ///     Generator used when <paramref name="seed"/> is -1. <see langword="null"/> falls back to the process-wide
        ///     <c>np.random</c>, which other spaces and envs also draw from, so samples are then not reproducible in isolation.
        /// </param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="low"/> or <paramref name="high"/> is <see langword="null"/>.</exception>
        /// <remarks>
        ///     The shape check between <paramref name="low"/> and <paramref name="high"/> is a <c>Debug.Assert</c>, so
        ///     Release builds accept mismatched shapes silently.
        /// </remarks>
        public Box(NDArray low, NDArray high, Type dType = null, int seed = -1, NumPyRandom random_state = null) : base(default, (dType = dType ?? np.float32)) {
            // Shape became a struct in NumSharp 0.70, so the placeholder handed to the base is `default` rather than
            // null; the real shape is assigned from `low` below, after the null checks.
            if (Equals(low, null)) throw new ArgumentNullException(nameof(low));
            if (Equals(high, null)) throw new ArgumentNullException(nameof(high));
            Debug.Assert(low.shape.SequenceEqual(high.shape));
            Shape = low.shape;
            Low = low.astype(dType);
            High = high.astype(dType);
            RandomState = seed != -1 ? np.random.RandomState(seed) : random_state ?? np.random;
            CheckBounded();
        }

        /// <summary>
        ///     Recomputes <see cref="BoundedLow"/> and <see cref="BoundedHigh"/>: an element counts as bounded when its
        ///     limit is finite, which decides whether <see cref="Sample"/> draws from a uniform, exponential or normal
        ///     distribution for it.
        /// </summary>
        /// <remarks>Must run after <see cref="Low"/> and <see cref="High"/> are assigned; both constructors call it last.</remarks>
        private void CheckBounded() {
            // NumSharp 0.70 follows NumPy's np.full(shape, fill_value) argument order (NumSharp.Lite took the value
            // first). A float fill keeps the comparison arrays float32, as before.
            NDArray neginf = np.full(Low.shape, (float)-np.inf);
            BoundedLow = (Low > neginf);
            NDArray posinf = np.full(Low.shape, (float)np.inf);
            BoundedHigh = (High < posinf);
        }

        public bool IsBounded(BoundedMannerEnum manner)
        {
            bool below = All(BoundedLow);
            bool above = All(BoundedHigh);
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

        private static bool Any(NDArray a) => a.ndim == 0 ? a.GetBoolean(0) : np.any(a);
        private static bool All(NDArray a) => a.ndim == 0 ? a.GetBoolean(0) : np.all(a);

        /// <summary>
        ///     Draws one element of the box: a uniform value for dimensions bounded on both sides, a shifted
        ///     exponential for half-bounded ones, and a normal for unbounded ones, cast to <see cref="Space.DType"/>.
        /// </summary>
        /// <param name="mask">Must be <see langword="null"/>; boxes don't support masked sampling.</param>
        /// <returns>A sample shaped like the box, or a 0-d array for a scalar box.</returns>
        /// <exception cref="NotSupportedException">Thrown when <paramref name="mask"/> is not <see langword="null"/>.</exception>
        /// <remarks>
        ///     Unbounded dimensions draw <c>normal(0.5, 1)</c> rather than Gymnasium's standard normal (plan defect B-08),
        ///     and integer boxes never reach <see cref="High"/> (B-09); both are fixed by the Gymnasium re-port, not here.
        /// </remarks>
        public override NDArray Sample(NDArray mask = null) {
            if (!Equals(mask, null))
            {
                throw new NotSupportedException("Box.sample cannot be provided a mask.");
            }
            if (Low.ndim == 0) {
                bool isLowBounded = All(BoundedLow);
                bool isHighBounded = All(BoundedHigh);
                NDArray scalarSample;
                if (isLowBounded && isHighBounded) {
                    scalarSample = RandomState.uniform(Low.GetSingle(0), High.GetSingle(0));
                } else if (isLowBounded) {
                    scalarSample = RandomState.exponential(1.0f) + Low.GetSingle(0);
                } else if (isHighBounded) {
                    scalarSample = -RandomState.exponential(1.0f) + High.GetSingle(0);
                } else {
                    scalarSample = RandomState.normal(0.5f, 1.0f);
                }
                if (DType == np.int32 || DType == np.uint32 || DType == np.@byte) {
                    scalarSample = np.floor(scalarSample);
                }
                return scalarSample.astype(DType);
            }

            NDArray unbounded = ~BoundedLow & ~BoundedHigh;
            NDArray upp_bounded = ~BoundedLow & BoundedHigh;
            NDArray low_bounded = BoundedLow & ~BoundedHigh;
            NDArray bounded = BoundedLow & BoundedHigh;

            NDArray sample = np.empty(Shape);

            if (Any(unbounded))
            {
                sample[unbounded] = RandomState.normal(0.5f, 1.0f, unbounded[unbounded].shape);
            }
            if (Any(low_bounded))
            {
                sample[low_bounded] = RandomState.exponential(1.0f, low_bounded[low_bounded].shape) + Low[low_bounded];
            }
            if (Any(upp_bounded))
            {
                sample[upp_bounded] = -RandomState.exponential(1.0f, upp_bounded[upp_bounded].shape) + High[upp_bounded];
            }
            if (Any(bounded))
            {
                // NumSharp 0.70's array-bound uniform takes the output dtype, not a size: it draws one value per
                // element of the (already masked) bound arrays, and float64 keeps the random factor and the result in
                // double. It still subtracts (high - low) in the bounds' own dtype, unlike NumPy, which widens both
                // bounds to float64 first, so draws aren't bit-identical to NumPy for float32 boxes. The NumPy-exact
                // formula is planned in docs/plans/NUMSHARP_MIGRATION_PLAN.md §6.3 (trap 11).
                sample[bounded] = RandomState.uniform(Low[bounded], High[bounded], np.float64);
            }
            if (DType == np.int32 || DType == np.uint32 || DType == np.@byte)
            {
                sample = np.floor(sample);
            }
            return sample.astype(DType); // RandomState.uniform(Low, High, DType);
        }

        /// <summary>
        ///     Tests whether <paramref name="ndArray"/> has this box's shape and every element lies within
        ///     <c>[Low, High]</c>.
        /// </summary>
        /// <param name="ndArray">The candidate; must be an <see cref="NDArray"/>.</param>
        /// <returns><see langword="true"/> when the shape matches and all elements are within bounds; otherwise <see langword="false"/>.</returns>
        /// <exception cref="NotSupportedException">Thrown when <paramref name="ndArray"/> is not an <see cref="NDArray"/> (including <see langword="null"/>).</exception>
        /// <remarks>Unlike Gymnasium, the dtype of <paramref name="ndArray"/> isn't checked (plan §7).</remarks>
        public override bool Contains(object ndArray) {
            // np.all reduces the element-wise comparisons: NumSharp 0.70 returns NDArray<bool> from >= and <= with no
            // implicit bool conversion. NumSharp.Lite's implicit conversion read a single element, and threw for any
            // box that wasn't 0-d (plan defect B-06).
            if (ndArray is NDArray x)
                return x.shape == Shape && np.all(x >= Low) && np.all(x <= High);
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

        /// <summary>
        ///     Delegates to the hashes of <see cref="Low"/> and <see cref="High"/>, the members <see cref="Equals(Box)"/>
        ///     compares. Since NumSharp 0.70 that throws, because <see cref="NDArray"/> is unhashable, so a box can't
        ///     be a dictionary key or a set element.
        /// </summary>
        /// <returns>Never returns for a constructed box, because both constructors always assign the bounds.</returns>
        /// <exception cref="NotSupportedException">
        ///     Thrown by <see cref="NDArray.GetHashCode"/>: arrays are mutable and therefore unhashable, the same as NumPy's
        ///     <c>ndarray</c>. Gymnasium's <c>Box</c> is unhashable for the same reason (it defines <c>__eq__</c> without
        ///     <c>__hash__</c>). For a key, use something immutable, such as the bytes of the bounds.
        /// </exception>
        public override int GetHashCode() {
            unchecked {
                // `is not null` rather than `!= null`: since NumSharp 0.70, NDArray overloads != element-wise and
                // returns an NDArray<bool>, so a null check through the operator no longer yields a bool.
                return ((Low is not null ? Low.GetHashCode() : 0) * 397) ^ (High is not null ? High.GetHashCode() : 0);
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