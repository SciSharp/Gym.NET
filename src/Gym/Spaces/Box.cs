using System;
using System.Diagnostics;
using System.Linq;
using System.Reflection.Metadata;
using System.Text.Json;
using Gym.Exceptions;
using Gym.Utils;
using NumSharp;

namespace Gym.Spaces {
    public enum BoundedMannerEnum
    {
        Both,
        Below,
        Above
    }
    public class Box : Space, IEquatable<Box> {
        /// <summary>
        ///     The generator this box samples from: NumPy's <c>Generator(PCG64)</c>, Gymnasium's <c>np_random</c>. The box owns
        ///     it, unless one was passed in to be shared.
        /// </summary>
        /// <remarks>Named <c>RandomState</c> and typed as NumPy's legacy <c>RandomState</c> (MT19937) before Gym.NET adopted Gymnasium's seeding.</remarks>
        protected Generator NpRandom;
        public NDArray Low { get; }
        public NDArray High { get; }
        public NDArray BoundedLow { get; private set;  }
        public NDArray BoundedHigh { get; private set; }
        /// <summary>
        ///     A box of <paramref name="shape"/> whose every element lies in [<paramref name="low"/>, <paramref name="high"/>],
        ///     with integer bounds; forwards to <see cref="Box(float, float, Shape, DType, long, Generator)"/>.
        /// </summary>
        /// <param name="low">The lower bound of every element.</param>
        /// <param name="high">The upper bound of every element.</param>
        /// <param name="shape">The shape of every sample.</param>
        /// <param name="dType">The element type; <see langword="null"/> means float32.</param>
        /// <param name="seed">
        ///     A non-negative seed for this box's own generator, drawing what Gymnasium's <c>Box(..., seed=seed)</c> draws; or
        ///     -1 for a generator of its own seeded from OS entropy.
        /// </param>
        /// <exception cref="Error">Thrown when <paramref name="seed"/> is negative but not -1, with Gymnasium's message.</exception>
        /// <remarks>
        ///     Footgun: with <see langword="int"/> literals and no fifth argument (<c>new Box(0, 5, shape, np.int32)</c>) the
        ///     call is ambiguous with the <paramref name="seed"/>-less <see cref="Box(int, int, Shape, DType, Generator)"/>
        ///     overload (CS0121). Name the fifth argument (<c>seed: 3</c>), or pass <see langword="float"/> literals.
        /// </remarks>
        public Box(int low, int high, Shape shape, DType dType = null, long seed = -1) : this((float) low, (float) high, shape, dType, seed, null) { }

        /// <summary>
        ///     A box of <paramref name="shape"/> whose every element lies in [<paramref name="low"/>, <paramref name="high"/>],
        ///     with integer bounds, drawing from a generator the caller supplies (and may share with other spaces).
        /// </summary>
        /// <param name="low">The lower bound of every element.</param>
        /// <param name="high">The upper bound of every element.</param>
        /// <param name="shape">The shape of every sample.</param>
        /// <param name="dType">The element type; <see langword="null"/> means float32.</param>
        /// <param name="random_state">
        ///     The generator to draw from (it may be shared with other spaces), or <see langword="null"/> for a generator of the
        ///     box's own, seeded from OS entropy.
        /// </param>
        /// <remarks>
        ///     Footgun: see <see cref="Box(int, int, Shape, DType, long)"/>; the two integer overloads are ambiguous unless the
        ///     fifth argument is given by name.
        /// </remarks>
        public Box(int low, int high, Shape shape, DType dType = null, Generator random_state = null) : this((float)low, (float)high, shape, dType, -1, random_state) { }

        /// <summary>
        ///     A box of <paramref name="shape"/> whose every element lies in [<paramref name="low"/>, <paramref name="high"/>].
        ///     Infinite bounds make the matching side unbounded, which changes how <see cref="Sample"/> draws that side.
        /// </summary>
        /// <param name="low">The lower bound of every element, cast to <paramref name="dType"/>.</param>
        /// <param name="high">The upper bound of every element, cast to <paramref name="dType"/>.</param>
        /// <param name="shape">The shape of every sample.</param>
        /// <param name="dType">The element type; <see langword="null"/> means float32.</param>
        /// <param name="seed">
        ///     A non-negative seed for this box's own generator, drawing what Gymnasium's <c>Box(..., seed=seed)</c> draws; or
        ///     -1 to use <paramref name="random_state"/> instead.
        /// </param>
        /// <param name="random_state">
        ///     The generator to draw from when <paramref name="seed"/> is -1 (it may be shared with other spaces), or
        ///     <see langword="null"/> for a generator of the box's own, seeded from OS entropy.
        /// </param>
        /// <exception cref="Error">Thrown when <paramref name="seed"/> is negative but not -1, with Gymnasium's message.</exception>
        /// <remarks>
        ///     The bounds are built with <c>np.full(shape, value, dtype)</c>, which casts the scalar to the dtype exactly the
        ///     way NumPy does. It replaces <c>(low + np.zeros(shape, dType)).astype(dType)</c>, which relied on NumSharp.Lite's
        ///     promotion. There is no null check on <paramref name="shape"/>: <see cref="Shape"/> is a struct in NumSharp, so
        ///     the former <c>Equals(shape, null)</c> guard could never fire.
        /// </remarks>
        public Box(float low, float high, Shape shape, DType dType = null, long seed = -1, Generator random_state = null) : base(shape, (dType = dType ?? np.float32)) {
            Low = np.full(shape, low, dType);
            High = np.full(shape, high, dType);
            // A generator of the box's own unless one is shared on purpose: a shared generator couples two spaces' streams,
            // and generators are not synchronized, so two threads drawing from one corrupt its state.
            NpRandom = seed != -1 ? Seeding.NpRandom(seed).Rng : random_state ?? Seeding.NpRandom().Rng;
            CheckBounded();
        }

        /// <summary>
        ///     A box bounded element-wise by <paramref name="low"/> and <paramref name="high"/>, which must have the same shape.
        ///     Both are copied and cast to <paramref name="dType"/>, so later writes to the caller's arrays do not move the box.
        /// </summary>
        /// <param name="low">The lower bounds; its shape becomes the box's shape. A C# scalar converts to a 0-d box.</param>
        /// <param name="high">The upper bounds, with the shape of <paramref name="low"/>.</param>
        /// <param name="dType">The element type; <see langword="null"/> means float32.</param>
        /// <param name="seed">
        ///     A non-negative seed for this box's own generator, drawing what Gymnasium's <c>Box(..., seed=seed)</c> draws; or
        ///     -1 to use <paramref name="random_state"/> instead.
        /// </param>
        /// <param name="random_state">
        ///     The generator to draw from when <paramref name="seed"/> is -1 (it may be shared with other spaces), or
        ///     <see langword="null"/> for a generator of the box's own, seeded from OS entropy.
        /// </param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="low"/> or <paramref name="high"/> is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">Thrown when the two shapes differ, with Gymnasium's message.</exception>
        /// <exception cref="Error">Thrown when <paramref name="seed"/> is negative but not -1, with Gymnasium's message.</exception>
        public Box(NDArray low, NDArray high, DType dType = null, long seed = -1, Generator random_state = null) : base(RequireSameShape(low, high), (dType = dType ?? np.float32)) {
            Low = low.astype(dType);
            High = high.astype(dType);
            // A generator of the box's own unless one is shared on purpose: a shared generator couples two spaces' streams,
            // and generators are not synchronized, so two threads drawing from one corrupt its state.
            NpRandom = seed != -1 ? Seeding.NpRandom(seed).Rng : random_state ?? Seeding.NpRandom().Rng;
            CheckBounded();
        }

        /// <summary>
        ///     Validates the bounds of <see cref="Box(NDArray, NDArray, DType, long, Generator)"/> before the base constructor
        ///     runs, and returns the box's shape. It replaces a <c>Debug.Assert</c> that vanished in Release builds, where
        ///     mismatched bounds used to construct a box whose <see cref="Low"/> and <see cref="High"/> disagreed.
        /// </summary>
        /// <param name="low">The lower bounds.</param>
        /// <param name="high">The upper bounds.</param>
        /// <returns>
        ///     A fresh contiguous <see cref="Shape"/> over a copy of the bounds' dimensions, so a view passed as a bound does not
        ///     carry its strides or offset into <see cref="Space.Shape"/>.
        /// </returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="low"/> or <paramref name="high"/> is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">
        ///     Thrown when the shapes differ, with the message of <c>gymnasium/spaces/box.py:179</c> @ 9e04324 (shapes are
        ///     printed as Python tuples, so a 1-D shape reads <c>(3,)</c>).
        /// </exception>
        private static Shape RequireSameShape(NDArray low, NDArray high) {
            if (low is null) throw new ArgumentNullException(nameof(low));
            if (high is null) throw new ArgumentNullException(nameof(high));
            if (low.Shape != high.Shape)
                throw new ValueError($"Box low.shape and high.shape don't match, low.shape={PythonTuple(low.shape)}, high.shape={PythonTuple(high.shape)}");
            return new Shape((long[]) low.shape.Clone());
        }

        /// <summary>Formats dimensions the way Python prints a tuple, for messages ported verbatim from Gymnasium.</summary>
        /// <param name="dimensions">The dimensions to format.</param>
        /// <returns><c>()</c>, <c>(3,)</c> or <c>(2, 3)</c>: Python's trailing comma marks a one-element tuple.</returns>
        private static string PythonTuple(long[] dimensions) {
            return dimensions.Length == 1 ? $"({dimensions[0]},)" : "(" + string.Join(", ", dimensions) + ")";
        }

        /// <summary>
        ///     Classifies each element as bounded below and/or above, which decides the distribution <see cref="Sample"/> draws
        ///     it from. An element is bounded on a side when its bound is finite, as in Gymnasium (<c>-np.inf &lt; low</c>,
        ///     <c>high &lt; np.inf</c>); a NaN bound therefore counts as unbounded.
        /// </summary>
        /// <remarks>
        ///     Comparing against scalar infinities replaces NumSharp.Lite's full-size <c>np.full(±inf)</c> arrays. It also works
        ///     for integer boxes (defect B-07), because NumSharp promotes the comparison instead of failing on mixed dtypes.
        /// </remarks>
        private void CheckBounded() {
            BoundedLow = Low > double.NegativeInfinity;
            BoundedHigh = High < double.PositiveInfinity;
        }

        /// <summary>
        ///     Whether every element is bounded on the side(s) <paramref name="manner"/> names. A box is bounded only if all of
        ///     its elements are: a single infinite bound makes the whole box unbounded on that side.
        /// </summary>
        /// <param name="manner">Which side(s) to test.</param>
        /// <returns><see langword="true"/> when every element is bounded as <paramref name="manner"/> requires.</returns>
        /// <exception cref="ArgumentException">Thrown for a <paramref name="manner"/> that is not a defined value.</exception>
        /// <remarks>
        ///     <c>np.all</c> returns a <see cref="bool"/> for arrays of any rank, including 0-d, so NumSharp.Lite's 0-d special
        ///     case is gone.
        /// </remarks>
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

        /// <summary>
        ///     Draws one point of the box, choosing each element's distribution from its bounds: uniform on [low, high],
        ///     low + exponential when only bounded below, high − exponential when only bounded above, and a standard normal
        ///     when unbounded. Integer boxes draw up to <c>high</c> inclusive. The result is a fresh array of
        ///     <see cref="Space.DType"/> that the caller owns.
        /// </summary>
        /// <param name="mask">Must be <see langword="null"/>: a box cannot be sampled under a mask.</param>
        /// <returns>A new array of <see cref="Space.Shape"/> and <see cref="Space.DType"/>, 0-d for a 0-d box.</returns>
        /// <exception cref="NotSupportedException">Thrown when <paramref name="mask"/> is not <see langword="null"/>.</exception>
        /// <remarks>
        ///     <para>
        ///     Port of <c>gymnasium/spaces/box.py:404</c> @ 9e04324. It makes the same generator calls in the same order
        ///     (standard normal, exponential, exponential, uniform) with the same arguments, so a box seeded like Gymnasium's
        ///     samples the same bits. Boxes that are not floating-point shift <c>high</c> by one
        ///     (<c>high.astype(int64) + 1</c>), floor, clip integer dtypes to their range (int64 with Gymnasium's margin of 2
        ///     and its second clip to the bounds), then cast.
        ///     </para>
        ///     <para>
        ///     The pre-Gymnasium port drew <c>normal(0.5, 1)</c> for unbounded elements (B-08) and never reached <c>high</c> in
        ///     integer boxes (B-09); both follow Gymnasium now. Not ported yet (P2): the probability-mask argument, and
        ///     Gymnasium's <c>error.Error</c> for a mask instead of <see cref="NotSupportedException"/>.
        ///     </para>
        /// </remarks>
        public override NDArray Sample(NDArray mask = null) {
            // `mask != null` would be an element-wise comparison in NumSharp (an NDArray<bool>, not a bool).
            if (mask is not null)
            {
                throw new NotSupportedException("Box.sample cannot be provided a mask.");
            }

            // Gymnasium: high = self.high if self.dtype.kind == "f" else self.high.astype("int64") + 1, so that integer (and
            // boolean) samples reach high itself after the floor below. The shifted copy is owned here; High never is.
            NDArray shiftedHigh = null;
            try
            {
                NDArray high = High;
                if (DType.kind != 'f')
                {
                    using NDArray highInt64 = High.astype(np.int64);
                    shiftedHigh = highInt64 + 1;
                    high = shiftedHigh;
                }

                // Boolean selectors that classify the elements by interval type. One path serves every rank: NumSharp's masked
                // indexing handles 0-d arrays. Every intermediate is scoped with `using` because an undisposed NDArray is only
                // reclaimed by the finalizer.
                using NDArray notBoundedLow = ~BoundedLow;
                using NDArray notBoundedHigh = ~BoundedHigh;
                using NDArray unbounded = notBoundedLow & notBoundedHigh;
                using NDArray upp_bounded = notBoundedLow & BoundedHigh;
                using NDArray low_bounded = BoundedLow & notBoundedHigh;
                using NDArray bounded = BoundedLow & BoundedHigh;

                // float64 like Gymnasium's np.empty(self.shape); the cast to DType happens once, at the end.
                using NDArray sample = np.empty(Shape);

                // count_nonzero gives the size Gymnasium reads from mask[mask].shape without materializing the selection. A
                // category with no elements draws nothing, which leaves the generator exactly where NumPy leaves it for a
                // size-0 draw.
                long unboundedCount = np.count_nonzero(unbounded);
                if (unboundedCount > 0)
                {
                    // Gymnasium: np_random.normal(size=...), a STANDARD normal.
                    using NDArray drawn = NpRandom.normal(0.0, 1.0, unboundedCount);
                    sample[unbounded] = drawn;
                }
                long lowBoundedCount = np.count_nonzero(low_bounded);
                if (lowBoundedCount > 0)
                {
                    using NDArray exponential = NpRandom.exponential(1.0, lowBoundedCount);
                    using NDArray lows = Low[low_bounded];
                    using NDArray drawn = exponential + lows;
                    sample[low_bounded] = drawn;
                }
                long uppBoundedCount = np.count_nonzero(upp_bounded);
                if (uppBoundedCount > 0)
                {
                    // Gymnasium offsets from the shifted high here as well.
                    using NDArray exponential = NpRandom.exponential(1.0, uppBoundedCount);
                    using NDArray negated = -exponential;
                    using NDArray highs = high[upp_bounded];
                    using NDArray drawn = negated + highs;
                    sample[upp_bounded] = drawn;
                }
                long boundedCount = np.count_nonzero(bounded);
                if (boundedCount > 0)
                {
                    // Generator.uniform(low_array, high_array) casts both bounds to float64 and computes
                    // low + (high - low) * next_double() per element (numpy/random/_generator.pyx, uniform; random_uniform in
                    // numpy/random/src/distributions/distributions.c). NumSharp's Generator.uniform takes scalar bounds only,
                    // and random(n) draws the same n next_double() values, so the stream and the bits match NumPy.
                    using NDArray lows = Low[bounded];
                    using NDArray highs = high[bounded];
                    using NDArray low64 = lows.astype(np.float64);
                    using NDArray high64 = highs.astype(np.float64);
                    using NDArray range = high64 - low64;
                    using NDArray u = NpRandom.random(boundedCount);
                    using NDArray scaled = range * u;
                    using NDArray drawn = low64 + scaled;
                    sample[bounded] = drawn;
                }

                return CastSample(sample);
            }
            finally
            {
                shiftedHigh?.Dispose();
            }
        }

        /// <summary>
        ///     The last steps of Gymnasium's <c>Box.sample</c> (<c>box.py:459</c>): floor every dtype of kind i, u or b, clip
        ///     integer dtypes to their range so the cast cannot wrap, cast to <see cref="Space.DType"/>, and clip int64 once
        ///     more to the bounds.
        /// </summary>
        /// <param name="sample">The float64 sample; it is not modified.</param>
        /// <returns>A new array of <see cref="Space.DType"/>.</returns>
        /// <remarks>
        ///     Testing the kind, rather than a list of dtypes, also covers uint8: <c>np.@byte</c> is int8 in NumSharp. Boolean
        ///     boxes are floored but not clipped, because <c>np.issubdtype(bool, np.integer)</c> is false.
        /// </remarks>
        private NDArray CastSample(NDArray sample) {
            if (DType.kind is not ('i' or 'u' or 'b'))
                return sample.astype(DType);

            using NDArray floored = np.floor(sample);
            if (DType.kind == 'b')
                return floored.astype(DType);

            // np.iinfo bounds, converted to float64 as NumPy converts the Python ints Gymnasium passes to clip. For int64,
            // Gymnasium keeps a margin of 2, but even with it both limits round to ±2^63 in float64, which is why it clips
            // again after the cast.
            var info = np.iinfo(DType);
            double dtypeMin = DType == np.int64 ? info.min + 2 : info.min;
            double dtypeMax = DType == np.int64 ? info.max - 2 : DType.kind == 'u' ? info.maxUnsigned : info.max;
            using NDArray lowerLimit = NDArray.Scalar<double>(dtypeMin);
            using NDArray upperLimit = NDArray.Scalar<double>(dtypeMax);
            using NDArray clipped = np.clip(floored, lowerLimit, upperLimit);
            NDArray cast = clipped.astype(DType);
            if (DType != np.int64)
                return cast;

            // float64 values near the int64 limits can cast to out-of-bounds integers; Gymnasium clips them to the bounds.
            using (cast)
            {
                return np.clip(cast, Low, High);
            }
        }

        /// <summary>
        ///     Whether <paramref name="ndArray"/> is a point of this box: an <see cref="NDArray"/> of the box's shape whose
        ///     every element lies within its bounds. A shape mismatch returns <see langword="false"/> instead of throwing.
        /// </summary>
        /// <param name="ndArray">The candidate point.</param>
        /// <returns><see langword="true"/> when the shape matches and <c>Low &lt;= x &lt;= High</c> holds element-wise.</returns>
        /// <exception cref="NotSupportedException">Thrown when <paramref name="ndArray"/> is not an <see cref="NDArray"/>.</exception>
        /// <remarks>
        ///     Mirrors the shape and bound checks of <c>gymnasium/spaces/box.py:483</c> @ 9e04324, reduced with
        ///     <c>np.all</c> (defect B-06: the former <c>x &gt;= Low &amp;&amp; x &lt;= High</c> relied on NumSharp.Lite converting
        ///     arrays to <see cref="bool"/> and threw for every box that was not 0-d). Gymnasium's
        ///     <c>np.can_cast(x.dtype, self.dtype)</c> check is not ported yet (P2), so a float64 point can belong to a
        ///     float32 box.
        /// </remarks>
        public override bool Contains(object ndArray) {
            if (ndArray is NDArray x)
            {
                if (x.Shape != Shape)
                    return false;

                // Short-circuits like Gymnasium's `and`: the upper comparison is only computed when the lower one holds.
                using (NDArray aboveLow = x >= Low)
                {
                    if (!np.all(aboveLow))
                        return false;
                }
                using NDArray belowHigh = x <= High;
                return np.all(belowHigh);
            }
            throw new NotSupportedException(ndArray?.ToString() ?? nameof(ndArray));
        }

        /// <summary>
        ///     Port of <c>Space.seed(seed)</c>: gives this box a fresh generator, <c>Generator(PCG64(SeedSequence(seed)))</c>,
        ///     so it samples what Gymnasium's box samples after <c>box.seed(seed)</c>. A generator the box shared with other
        ///     spaces is left untouched.
        /// </summary>
        /// <param name="seed">Any non-negative seed.</param>
        /// <exception cref="Error">Thrown when <paramref name="seed"/> is negative, with Gymnasium's message.</exception>
        /// <remarks>Gymnasium's <c>seed</c> also accepts no seed and returns the one it used; that shape of the API arrives with P2.</remarks>
        public override void Seed(System.Numerics.BigInteger seed) {
            NpRandom = Seeding.NpRandom(seed).Rng;
        }

        #region Object Overrides

        /// <summary>Returns a string that represents the current object.</summary>
        /// <returns>A string that represents the current object.</returns>
        public override string ToString() {
            return "Box" + Shape;
        }

        /// <summary>
        ///     Structural equality: the same shape, the same dtype and element-identical bounds. Arrays are compared with
        ///     <see cref="np.array_equal(NDArray, NDArray)"/>, never with <see cref="NDArray.Equals(object)"/>, which broadcasts
        ///     (a 0-d array equals every array holding that value) and throws for shapes that cannot broadcast.
        /// </summary>
        /// <param name="other">The box to compare with; may be <see langword="null"/>.</param>
        /// <returns><see langword="true"/> when both boxes describe the same set of points with the same dtype.</returns>
        /// <remarks>
        ///     Gymnasium's <c>Box.__eq__</c> (<c>gymnasium/spaces/box.py:524</c> @ 9e04324) compares the bounds with
        ///     <c>np.allclose</c>; the exact comparison stays until P2 ports it. Because the comparison is exact, a box with a
        ///     NaN bound equals only itself.
        /// </remarks>
        public bool Equals(Box other) {
            if (ReferenceEquals(null, other)) return false;
            if (ReferenceEquals(this, other)) return true;
            return Shape == other.Shape && DType == other.DType
                && np.array_equal(Low, other.Low) && np.array_equal(High, other.High);
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
        ///     Hashes the shape and dtype only. The bounds cannot take part: <see cref="NDArray.GetHashCode"/> throws, because
        ///     NumPy arrays are mutable and therefore unhashable.
        /// </summary>
        /// <returns>
        ///     A hash consistent with <see cref="Equals(Box)"/>: equal boxes always share shape and dtype. Boxes that differ
        ///     only in their bounds collide, which costs dictionary performance, never correctness.
        /// </returns>
        public override int GetHashCode() => HashCode.Combine(Shape, DType);

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