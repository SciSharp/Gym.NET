#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using Gym.Exceptions;
using Gym.Utils;
using NumSharp;

namespace Gym.Spaces {
    /// <summary>Which side of a <see cref="Box"/> <see cref="Box.IsBounded(BoundedMannerEnum)"/> checks; Gymnasium spells it as a string.</summary>
    public enum BoundedMannerEnum {
        /// <summary>Bounded below and above (<c>"both"</c>).</summary>
        Both,

        /// <summary>Bounded below (<c>"below"</c>).</summary>
        Below,

        /// <summary>Bounded above (<c>"above"</c>).</summary>
        Above
    }

    /// <summary>
    ///     A possibly unbounded box in R^n: every element is an array of <see cref="Shape"/> and <see cref="DType"/> whose
    ///     entries lie within <see cref="Low"/> and <see cref="High"/> (each interval closed, or open to infinity). It describes
    ///     continuous observations and actions, such as CartPole's observation; an integer dtype makes it a discrete grid.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The box owns <see cref="Low"/>, <see cref="High"/>, <see cref="BoundedBelow"/> and <see cref="BoundedAbove"/> and
    ///     releases them in <see cref="Space.Dispose()"/>. They are exposed as they are, so writing into them changes the space,
    ///     as writing into Gymnasium's <c>space.low</c> does.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/spaces/box.py#L64">gymnasium/spaces/box.py</see>.</para>
    /// </remarks>
    public class Box : Space<NDArray>, IEquatable<Box> {
        /// <summary>The cached <c>repr</c> of <see cref="Low"/> (Gymnasium's <c>low_repr</c>), computed on first <see cref="ToString"/>.</summary>
        private string? _lowRepr;

        /// <summary>The cached <c>repr</c> of <see cref="High"/> (Gymnasium's <c>high_repr</c>).</summary>
        private string? _highRepr;

        /// <summary>A box whose every entry lies in [<paramref name="low"/>, <paramref name="high"/>] (scalar bounds).</summary>
        /// <param name="low">The lower bound of every entry; <c>double.NegativeInfinity</c> for none (a signed integer dtype stores its minimum).</param>
        /// <param name="high">The upper bound of every entry; <c>double.PositiveInfinity</c> for none.</param>
        /// <param name="shape">The shape; <see langword="null"/> is <c>(1,)</c>, as for Gymnasium's scalar bounds.</param>
        /// <param name="dtype">The dtype: integer, floating or bool; <see langword="null"/> is float32, Gymnasium's default.</param>
        /// <param name="seed">An integer seed, a generator to share, or <see langword="null"/> for a lazy entropy-seeded one.</param>
        /// <exception cref="NumSharp.ValueError">Thrown for a NaN bound, a bound outside the dtype, an infinite bound of an unsigned or bool dtype, <c>low &gt; high</c>, or a dtype that is not integer, floating or bool, with Gymnasium's messages.</exception>
        /// <exception cref="Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public Box(double low, double high, Shape? shape = null, DType? dtype = null, SpaceSeed? seed = null)
            : this(Bound.Of(low), Bound.Of(high), shape, dtype, seed) { }

        /// <summary>A box with per-entry bounds: the shape is the arrays' (they must agree), or <paramref name="shape"/> when given.</summary>
        /// <param name="low">The lower bounds; read, never kept (the box stores a copy in its dtype).</param>
        /// <param name="high">The upper bounds; read, never kept.</param>
        /// <param name="shape">The shape, or <see langword="null"/> to take the bounds'.</param>
        /// <param name="dtype">The dtype; <see langword="null"/> is float32.</param>
        /// <param name="seed">An integer seed, a generator to share, or <see langword="null"/>.</param>
        /// <exception cref="NumSharp.ValueError">Thrown for shapes that differ, a NaN bound, a bound outside the dtype, an infinite bound of an unsigned or bool dtype, or <c>low &gt; high</c>.</exception>
        /// <exception cref="Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public Box(NDArray low, NDArray high, Shape? shape = null, DType? dtype = null, SpaceSeed? seed = null)
            : this(Bound.Of(low), Bound.Of(high), shape, dtype, seed) { }

        /// <summary>
        ///     0.2's <c>new Box(low, high, np.float32)</c>, the dtype third, kept so 0.2 code compiles until 2.0 (DR-58): the same
        ///     box as <c>new Box(low, high, dtype: dtype)</c>, of shape <c>(1,)</c>, Gymnasium's for scalar bounds.
        /// </summary>
        /// <param name="low">The lower bound of every entry.</param>
        /// <param name="high">The upper bound of every entry.</param>
        /// <param name="dType">
        ///     The dtype. Named as 0.2 named it, so 1.0's <c>dtype:</c> calls never bind here: with the same name, a named
        ///     <c>dtype:</c> argument would prefer this overload (<c>DType</c> over <c>DType?</c>) or turn ambiguous.
        /// </param>
        /// <param name="seed">An integer seed, a generator to share, or <see langword="null"/>.</param>
        /// <exception cref="NumSharp.ValueError">Thrown as the shape-taking constructor throws, with Gymnasium's messages.</exception>
        /// <exception cref="Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        /// <remarks>Footgun: next to it, <c>new Box(low, high, null)</c> is ambiguous (a <see cref="Shape"/> or a dtype); name the argument.</remarks>
        [Obsolete("0.2's argument order: write new Box(low, high, dtype: dtype); the third positional argument is the shape. Removed in 2.0 with Gym.Legacy.")]
        public Box(double low, double high, DType dType, SpaceSeed? seed = null)
            : this(low, high, null, dType, seed) { }

        /// <summary>
        ///     0.2's <c>new Box(lowArray, highArray, np.float32)</c>, the dtype third, kept so 0.2 code compiles until 2.0
        ///     (DR-58): the same box as <c>new Box(low, high, dtype: dtype)</c>, of the bounds' shape.
        /// </summary>
        /// <param name="low">The lower bounds; read, never kept.</param>
        /// <param name="high">The upper bounds; read, never kept.</param>
        /// <param name="dType">
        ///     The dtype. Named as 0.2 named it, so 1.0's <c>dtype:</c> calls never bind here: with the same name, a named
        ///     <c>dtype:</c> argument would prefer this overload (<c>DType</c> over <c>DType?</c>) or turn ambiguous.
        /// </param>
        /// <param name="seed">An integer seed, a generator to share, or <see langword="null"/>.</param>
        /// <exception cref="NumSharp.ValueError">Thrown as the shape-taking constructor throws, with Gymnasium's messages.</exception>
        /// <exception cref="Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        /// <remarks>Footgun: next to it, <c>new Box(low, high, null)</c> is ambiguous (a <see cref="Shape"/> or a dtype); name the argument.</remarks>
        [Obsolete("0.2's argument order: write new Box(low, high, dtype: dtype); the third positional argument is the shape. Removed in 2.0 with Gym.Legacy.")]
        public Box(NDArray low, NDArray high, DType dType, SpaceSeed? seed = null)
            : this(low, high, null, dType, seed) { }

        /// <summary>A box with per-entry lower bounds and a scalar upper bound.</summary>
        /// <param name="low">The lower bounds; read, never kept.</param>
        /// <param name="high">The upper bound of every entry.</param>
        /// <param name="shape">The shape, or <see langword="null"/> to take <paramref name="low"/>'s.</param>
        /// <param name="dtype">The dtype; <see langword="null"/> is float32.</param>
        /// <param name="seed">An integer seed, a generator to share, or <see langword="null"/>.</param>
        /// <exception cref="NumSharp.ValueError">Thrown as for the other constructors.</exception>
        /// <exception cref="Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public Box(NDArray low, double high, Shape? shape = null, DType? dtype = null, SpaceSeed? seed = null)
            : this(Bound.Of(low), Bound.Of(high), shape, dtype, seed) { }

        /// <summary>A box with a scalar lower bound and per-entry upper bounds.</summary>
        /// <param name="low">The lower bound of every entry.</param>
        /// <param name="high">The upper bounds; read, never kept.</param>
        /// <param name="shape">The shape, or <see langword="null"/> to take <paramref name="high"/>'s.</param>
        /// <param name="dtype">The dtype; <see langword="null"/> is float32.</param>
        /// <param name="seed">An integer seed, a generator to share, or <see langword="null"/>.</param>
        /// <exception cref="NumSharp.ValueError">Thrown as for the other constructors.</exception>
        /// <exception cref="Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public Box(double low, NDArray high, Shape? shape = null, DType? dtype = null, SpaceSeed? seed = null)
            : this(Bound.Of(low), Bound.Of(high), shape, dtype, seed) { }

        /// <summary>The shared constructor: Gymnasium's <c>Box.__init__</c>, step by step.</summary>
        /// <param name="low">The lower bound, scalar or array.</param>
        /// <param name="high">The upper bound, scalar or array.</param>
        /// <param name="shape">The shape, or <see langword="null"/> to infer it.</param>
        /// <param name="dtype">The dtype, or <see langword="null"/> for float32.</param>
        /// <param name="seed">The seed.</param>
        /// <exception cref="NumSharp.ValueError">Thrown for every invalid argument, with Gymnasium's message.</exception>
        /// <exception cref="Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        private Box(Bound low, Bound high, Shape? shape, DType? dtype, SpaceSeed? seed)
            : base(ResolveShape(low, high, shape), ResolveDType(dtype), null) {
            DType elementType = DType;
            Shape boxShape = Shape;
            // Disallowed before casting: out of range, NaN, and infinities an unsigned or bool dtype cannot hold.
            (double dtypeMin, double dtypeMax) = DTypeRange(elementType);
            (NDArray lowArray, NDArray boundedBelow) = CastLow(low, dtypeMin, elementType, boxShape);
            (NDArray highArray, NDArray boundedAbove) = CastHigh(high, dtypeMax, elementType, boxShape);
            Low = lowArray;
            High = highArray;
            BoundedBelow = boundedBelow;
            BoundedAbove = boundedAbove;

            // Rechecked for the case where both a shape and array bounds were given.
            if (!SameShape(Low.Shape, boxShape))
                throw new ValueError($"Box low.shape doesn't match provided shape, low.shape={ShapeRepr(Low.Shape)}, shape={ShapeRepr(boxShape)}");
            if (!SameShape(High.Shape, boxShape))
                throw new ValueError($"Box high.shape doesn't match provided shape, high.shape={ShapeRepr(High.Shape)}, shape={ShapeRepr(boxShape)}");

            using (NDArray inverted = Low > High) {
                if (np.any(inverted))
                    throw new ValueError($"Box all low values must be less than or equal to high (some values break this), low={PyBuiltins.Str(Low)}, high={PyBuiltins.Str(High)}");
            }

            // Seeded last, as Gymnasium's super().__init__ runs after validation: a rejected box draws nothing.
            if (seed is { } given) {
                if (given.Generator != null)
                    NpRandom = given.Generator;
                else
                    Seed(given.Value);
            }
        }

        /// <summary>The lower bounds, an array of <see cref="Shape"/> and <see cref="DType"/> (Gymnasium's <c>low</c>); a signed integer box stores its dtype's minimum where the bound was <c>-inf</c>.</summary>
        public NDArray Low { get; }

        /// <summary>The upper bounds (Gymnasium's <c>high</c>).</summary>
        public NDArray High { get; }

        /// <summary>Where the box is bounded below: a bool array of <see cref="Shape"/> (Gymnasium's <c>bounded_below</c>, <c>-inf &lt; low</c>).</summary>
        public NDArray BoundedBelow { get; }

        /// <summary>Where the box is bounded above (Gymnasium's <c>bounded_above</c>, <c>high &lt; inf</c>).</summary>
        public NDArray BoundedAbove { get; }

        /// <summary>The element dtype; never <see langword="null"/> for a box.</summary>
        public new DType DType => base.DType!;

        /// <summary>The shape of every element; never <see langword="null"/> for a box.</summary>
        public new Shape Shape => base.Shape!.Value;

        /// <inheritdoc />
        public override bool IsNpFlattenable => true;

        /// <summary>
        ///     Whether the box is bounded (Gymnasium's <c>is_bounded</c>): on both sides, below or above, in every entry.
        /// </summary>
        /// <param name="manner"><c>"both"</c>, <c>"below"</c> or <c>"above"</c>.</param>
        /// <returns><see langword="true"/> when bounded in that manner.</returns>
        /// <exception cref="NumSharp.ValueError">Thrown for another manner: <c>manner is not in {'below', 'above', 'both'}, actual value: …</c>.</exception>
        public bool IsBounded(string manner = "both") {
            bool below = np.all(BoundedBelow);
            bool above = np.all(BoundedAbove);
            return manner switch {
                "both" => below && above,
                "below" => below,
                "above" => above,
                _ => throw new ValueError($"manner is not in {{'below', 'above', 'both'}}, actual value: {manner}"),
            };
        }

        /// <summary>The enum spelling of <see cref="IsBounded(string)"/>, kept from Gym.NET's earlier API.</summary>
        /// <param name="manner">The side to check.</param>
        /// <returns><see langword="true"/> when bounded in that manner.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a value outside the enum.</exception>
        public bool IsBounded(BoundedMannerEnum manner) => manner switch {
            BoundedMannerEnum.Both => IsBounded("both"),
            BoundedMannerEnum.Below => IsBounded("below"),
            BoundedMannerEnum.Above => IsBounded("above"),
            _ => throw new ArgumentOutOfRangeException(nameof(manner), manner, "Unsupported BoundedMannerEnum value."),
        };

        /// <summary>
        ///     Draws one element (Gymnasium's <c>sample</c>), each entry from the distribution its interval calls for:
        ///     uniform on <c>[a, b]</c>, a shifted exponential on <c>[a, inf)</c>, a shifted negative exponential on
        ///     <c>(-inf, b]</c>, a standard normal on <c>(-inf, inf)</c>. Integer boxes floor the draw (from <c>[a, b + 1)</c>)
        ///     and clip it to the dtype.
        /// </summary>
        /// <param name="mask">Must be <see langword="null"/>: a box takes no mask.</param>
        /// <param name="probability">Must be <see langword="null"/>.</param>
        /// <returns>A new array of <see cref="Shape"/> and <see cref="DType"/>, owned by the caller.</returns>
        /// <exception cref="Error">Thrown when a mask or a probability mask is given, with Gymnasium's message.</exception>
        /// <remarks>
        ///     The generator calls are Gymnasium's, in its order and sizes (one <c>normal</c>, two <c>exponential</c> and one
        ///     <c>uniform</c> call over the entries of each category in C order), and the arithmetic is its float64 arithmetic,
        ///     so a box seeded like Gymnasium's draws the same bits. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/spaces/box.py#L399">gymnasium/spaces/box.py</see>.
        /// </remarks>
        public override NDArray Sample(object? mask = null, object? probability = null) {
            if (mask is not null)
                throw new Error($"Box.sample cannot be provided a mask, actual value: {PyBuiltins.Str(mask)}");
            if (probability is not null)
                throw new Error($"Box.sample cannot be provided a probability mask, actual value: {PyBuiltins.Str(probability)}");

            int size = checked((int)Shape.size);
            bool isFloat = DType.kind == 'f';
            double[] low = ToDoubles(Low);
            // high = self.high if float else self.high.astype("int64") + 1, so floor() can reach high itself.
            double[] high = isFloat ? ToDoubles(High) : ToInt64s(High).Select(h => (double)(h + 1)).ToArray();
            bool[] below = BoundedBelow.ToArray<bool>(), above = BoundedAbove.ToArray<bool>();

            // The four categories, each as its entries' flat indices in C order: Gymnasium's sample[mask] = draws.
            var unbounded = new List<int>();
            var upperOnly = new List<int>();
            var lowerOnly = new List<int>();
            var bounded = new List<int>();
            for (int i = 0; i < size; i++) {
                if (!below[i] && !above[i])
                    unbounded.Add(i);
                else if (!below[i] && above[i])
                    upperOnly.Add(i);
                else if (below[i] && !above[i])
                    lowerOnly.Add(i);
                else
                    bounded.Add(i);
            }

            var sample = new double[size];
            // np.empty(self.shape) holds float64 garbage NumPy never reads: every entry belongs to exactly one category.
            using (NDArray normal = NpRandom.normal(size: new Shape(unbounded.Count)))
                Scatter(sample, unbounded, ToDoubles(normal), 1.0, null);
            using (NDArray exponential = NpRandom.exponential(size: new Shape(lowerOnly.Count)))
                Scatter(sample, lowerOnly, ToDoubles(exponential), 1.0, low);
            using (NDArray exponential = NpRandom.exponential(size: new Shape(upperOnly.Count)))
                Scatter(sample, upperOnly, ToDoubles(exponential), -1.0, high);
            using (NDArray lowB = np.array(bounded.Select(i => low[i]).ToArray()))
            using (NDArray highB = np.array(bounded.Select(i => high[i]).ToArray()))
            using (NDArray uniform = NpRandom.uniform(lowB, highB, new Shape(bounded.Count)))
                Scatter(sample, bounded, ToDoubles(uniform), 1.0, null);

            if (DType.kind is 'i' or 'u' or 'b') {
                for (int i = 0; i < size; i++)
                    sample[i] = Math.Floor(sample[i]);
            }

            // Clip values that would overflow the integer dtype; float64 cannot represent int64's limits, so int64 keeps a
            // margin of 2 to stay inside after the cast.
            if (DType.kind is 'i' or 'u') {
                (double dtypeMin, double dtypeMax) = DTypeRange(DType);
                if (DType.typecode == NPTypeCode.Int64) {
                    dtypeMin = (double)(long.MinValue + 2L);
                    dtypeMax = (double)(long.MaxValue - 2L);
                }

                for (int i = 0; i < size; i++)
                    sample[i] = Math.Min(Math.Max(sample[i], dtypeMin), dtypeMax);
            }

            NDArray result = Cast(sample, DType, Shape);
            if (DType.typecode == NPTypeCode.Int64) {
                // float64 is coarser than int64 near its limits, so the cast can land just outside the bounds: clip again.
                using (result) {
                    return np.clip(result, Low, High);
                }
            }

            return result;
        }

        /// <summary>
        ///     Whether <paramref name="x"/> is an element (Gymnasium's <c>contains</c>): an array whose dtype casts safely to the
        ///     box's, of the box's shape, inside the bounds. Anything else is converted to an array of the box's dtype first,
        ///     with Gymnasium's warning, and is not an element when it cannot be.
        /// </summary>
        /// <param name="x">The candidate: an <see cref="NDArray"/>, a .NET array or list, or a number.</param>
        /// <returns><see langword="true"/> for an element.</returns>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/spaces/box.py#L478">gymnasium/spaces/box.py</see>.</remarks>
        public override bool Contains(object? x) {
            NDArray? owned = null;
            NDArray candidate;
            if (x is NDArray array) {
                candidate = array;
            } else {
                Logger.Warn("Casting input x to numpy array.");
                try {
                    owned = AsArray(x, DType);
                } catch (Exception e) when (e is ValueError or TypeError or OverflowException or ArgumentException or InvalidCastException or NotSupportedException) {
                    // np.asarray's ValueError, TypeError and OverflowError make Gymnasium return False; .NET's conversion
                    // failures are the same refusals.
                    return false;
                }

                candidate = owned;
            }

            try {
                if (!np.can_cast(candidate.dtype, DType) || !SameShape(candidate.Shape, Shape))
                    return false;
                using NDArray atLeastLow = candidate >= Low;
                using NDArray atMostHigh = candidate <= High;
                return np.all(atLeastLow) && np.all(atMostHigh);
            } finally {
                owned?.Dispose();
            }
        }

        /// <summary>Converts samples to nested lists (Gymnasium's <c>to_jsonable</c>, <c>sample.tolist()</c>).</summary>
        /// <param name="samples">Samples of this box.</param>
        /// <returns>One nested list per sample.</returns>
        public override object ToJsonable(IEnumerable<NDArray> samples) => samples.Select(s => PythonList(s)).ToList();

        /// <summary>Converts nested lists back to arrays of the box's dtype (Gymnasium's <c>from_jsonable</c>, <c>np.asarray(sample, dtype)</c>).</summary>
        /// <param name="jsonable">Values <see cref="ToJsonable(IEnumerable{NDArray})"/> produced, or numbers.</param>
        /// <returns>New arrays, owned by the caller.</returns>
        /// <exception cref="ArgumentException">Thrown for a value that cannot become an array.</exception>
        public override IList<NDArray> FromJsonable(object jsonable) => JsonList(jsonable).Select(v => AsArray(v, DType)).ToList();

        /// <summary>
        ///     Gymnasium's <c>repr</c>: <c>Box(low, high, shape, dtype)</c>, each bound shortened to its value when it is the
        ///     same for every entry: <c>Box(-1.0, 2.0, (3, 4), float32)</c>, <c>Box([-1. -2.], [2. 4.], (2,), float32)</c>.
        /// </summary>
        /// <returns>The text.</returns>
        public override string ToString() {
            _lowRepr ??= ArrayShortRepr(Low);
            _highRepr ??= ArrayShortRepr(High);
            return $"Box({_lowRepr}, {_highRepr}, {ShapeRepr(Shape)}, {Discrete.DTypeName(DType)})";
        }

        /// <summary>Gymnasium's <c>==</c>: the same shape and dtype, and bounds equal within <c>np.allclose</c>.</summary>
        /// <param name="other">The other box.</param>
        /// <returns><see langword="true"/> when equal.</returns>
        public bool Equals(Box? other) {
            if (other is null)
                return false;
            if (ReferenceEquals(this, other))
                return true;
            return SameShape(Shape, other.Shape) && DType == other.DType && np.allclose(Low, other.Low) && np.allclose(High, other.High);
        }

        /// <inheritdoc />
        public override bool Equals(object? obj) => obj is Box other && Equals(other);

        /// <summary>A hash over the shape and dtype, which equal boxes share (their bounds only need to be close).</summary>
        /// <returns>The hash.</returns>
        public override int GetHashCode() => HashCode.Combine(ShapeRepr(Shape), Discrete.DTypeName(DType));

        /// <summary>Value equality, as <see cref="Equals(Box)"/>.</summary>
        /// <param name="left">A box.</param>
        /// <param name="right">A box.</param>
        /// <returns><see langword="true"/> when equal.</returns>
        public static bool operator ==(Box? left, Box? right) => left is null ? right is null : left.Equals(right);

        /// <summary>Value inequality.</summary>
        /// <param name="left">A box.</param>
        /// <param name="right">A box.</param>
        /// <returns><see langword="true"/> when not equal.</returns>
        public static bool operator !=(Box? left, Box? right) => !(left == right);

        /// <inheritdoc />
        protected override void Dispose(bool disposing) {
            if (disposing) {
                Low.Dispose();
                High.Dispose();
                BoundedBelow.Dispose();
                BoundedAbove.Dispose();
            }

            base.Dispose(disposing);
        }

        /// <summary>A bound as given: a number (Gymnasium's <c>is_float_integer</c>) or an array.</summary>
        /// <param name="Scalar">The number, when <paramref name="Array"/> is <see langword="null"/>.</param>
        /// <param name="Array">The array, or <see langword="null"/> for a number.</param>
        private readonly record struct Bound(double Scalar, [property: NDBorrowed] NDArray? Array) {
            /// <summary>A scalar bound.</summary>
            /// <param name="value">The value.</param>
            /// <returns>The bound.</returns>
            public static Bound Of(double value) => new(value, null);

            /// <summary>
            ///     An array bound, or a scalar one for a 0-d integer or float array: that is a NumPy scalar (DR-08), which
            ///     Gymnasium's <c>is_float_integer</c> takes as a number, so <c>Box(np.zeros(1)[0], np.ones(1)[0])</c> has shape
            ///     <c>(1,)</c>, not <c>()</c>. Pass the shape <c>()</c> explicitly for a 0-d box.
            /// </summary>
            /// <param name="value">The array.</param>
            /// <returns>The bound.</returns>
            /// <exception cref="ArgumentNullException">Thrown for <see langword="null"/>: Gymnasium's <c>None</c> bound is no bound shape can come from.</exception>
            public static Bound Of(NDArray value) {
                if (value is null)
                    throw new ArgumentNullException(nameof(value));
                if (value.ndim == 0 && value.dtype.kind is 'i' or 'u' or 'f')
                    return new(Convert.ToDouble(value.GetAtIndex(0), CultureInfo.InvariantCulture), null);
                return new(0, value);
            }
        }

        /// <summary>Gymnasium's shape inference: the given shape, else the arrays' (which must agree), else <c>(1,)</c> for scalars.</summary>
        /// <param name="low">The lower bound.</param>
        /// <param name="high">The upper bound.</param>
        /// <param name="shape">The given shape, or <see langword="null"/>.</param>
        /// <returns>The shape.</returns>
        /// <exception cref="NumSharp.ValueError">Thrown when two array bounds have different shapes.</exception>
        private static Shape ResolveShape(Bound low, Bound high, Shape? shape) {
            if (shape is { } given)
                return given;
            if (low.Array is not null && high.Array is not null) {
                if (!SameShape(low.Array.Shape, high.Array.Shape))
                    throw new ValueError($"Box low.shape and high.shape don't match, low.shape={ShapeRepr(low.Array.Shape)}, high.shape={ShapeRepr(high.Array.Shape)}");
                return Clean(low.Array.Shape);
            }

            if (low.Array is not null)
                return Clean(low.Array.Shape);
            if (high.Array is not null)
                return Clean(high.Array.Shape);
            return new Shape(1L);
        }

        /// <summary>The dtype a constructor was given, validated: <see langword="null"/> is float32; it must be integer, floating or bool.</summary>
        /// <param name="dtype">The dtype.</param>
        /// <returns>The dtype to use.</returns>
        /// <exception cref="NumSharp.ValueError">Thrown for any other kind.</exception>
        private static DType ResolveDType(DType? dtype) {
            DType resolved = dtype ?? np.float32;
            if (resolved.kind is not ('i' or 'u' or 'f' or 'b'))
                throw new ValueError($"Invalid Box dtype ({Discrete.DTypeName(resolved)}), must be an integer, floating, or bool dtype");
            return resolved;
        }

        /// <summary>A dtype's range as Gymnasium reads it: 0..1 for bool, <c>finfo</c> for floats, <c>iinfo</c> for integers.</summary>
        /// <param name="dtype">The dtype.</param>
        /// <returns>The minimum and maximum, as doubles (exact for every dtype but the 64-bit integers' extremes).</returns>
        private static (double Min, double Max) DTypeRange(DType dtype) {
            if (dtype.kind == 'b')
                return (0, 1);
            if (dtype.kind == 'f') {
                finfo info = np.finfo(dtype);
                return ((double)info.min, (double)info.max);
            }

            iinfo ints = np.iinfo(dtype);
            return (ints.min, dtype.typecode == NPTypeCode.UInt64 ? ulong.MaxValue : ints.max);
        }

        /// <summary>Gymnasium's <c>_cast_low</c>: the lower bound in the box's dtype, and where it is finite.</summary>
        /// <param name="low">The bound as given.</param>
        /// <param name="dtypeMin">The dtype's minimum.</param>
        /// <param name="dtype">The box's dtype.</param>
        /// <param name="shape">The box's shape.</param>
        /// <returns>The cast bound and <c>bounded_below</c>, both new arrays the box owns.</returns>
        /// <exception cref="NumSharp.ValueError">Thrown for NaN, an infinity the dtype cannot hold, or a value below the dtype.</exception>
        private static (NDArray Low, NDArray BoundedBelow) CastLow(Bound low, double dtypeMin, DType dtype, Shape shape) {
            if (low.Array is null) {
                double value = low.Scalar;
                NDArray boundedBelow = np.full(shape, double.NegativeInfinity < value, np.@bool);
                if (double.IsNaN(value))
                    throw new ValueError($"No low value can be equal to `np.nan`, low={PyBuiltins.FloatRepr(value)}");
                if (double.IsNegativeInfinity(value)) {
                    if (dtype.kind == 'i')
                        value = dtypeMin;
                    else if (dtype.kind is 'u' or 'b')
                        throw new ValueError($"Box unsigned int dtype don't support `-np.inf`, low={PyBuiltins.FloatRepr(value)}");
                } else if (value < dtypeMin) {
                    throw new ValueError($"Box low is out of bounds of the dtype range, low={PyBuiltins.FloatRepr(value)}, min dtype={NumberRepr(dtypeMin, dtype)}");
                }

                return (Full(shape, value, dtype), boundedBelow);
            }

            NDArray given = low.Array;
            if (given.dtype.kind is not ('f' or 'i' or 'u' or 'b'))
                throw new ValueError($"Box low must be a floating, integer, or bool dtype, actual dtype={Discrete.DTypeName(given.dtype)}");
            using (NDArray nan = np.isnan(given)) {
                if (np.any(nan))
                    throw new ValueError($"No low value can be equal to `np.nan`, low={PyBuiltins.Str(given)}");
            }

            NDArray bounded = given > double.NegativeInfinity;
            // A copy, because Gymnasium's low[neginf] = dtype_min writes into the caller's array and Gym.NET leaves it intact.
            using NDArray working = given.copy();
            using (NDArray neginf = np.isneginf(working)) {
                if (np.any(neginf)) {
                    if (dtype.kind == 'i')
                        working[neginf] = dtypeMin;
                    else if (dtype.kind is 'u' or 'b')
                        throw new ValueError($"Box unsigned int dtype don't support `-np.inf`, low={PyBuiltins.Str(given)}");
                } else if (working.dtype != dtype) {
                    using NDArray under = working < dtypeMin;
                    if (np.any(under))
                        throw new ValueError($"Box low is out of bounds of the dtype range, low={PyBuiltins.Str(given)}, min dtype={NumberRepr(dtypeMin, dtype)}");
                }
            }

            WarnIfPrecisionLowered("low", given.dtype, dtype);
            return (working.astype(dtype), bounded);
        }

        /// <summary>Gymnasium's <c>_cast_high</c>: the upper bound in the box's dtype, and where it is finite.</summary>
        /// <param name="high">The bound as given.</param>
        /// <param name="dtypeMax">The dtype's maximum.</param>
        /// <param name="dtype">The box's dtype.</param>
        /// <param name="shape">The box's shape.</param>
        /// <returns>The cast bound and <c>bounded_above</c>, both new arrays the box owns.</returns>
        /// <exception cref="NumSharp.ValueError">Thrown for NaN, an infinity the dtype cannot hold, or a value above the dtype.</exception>
        private static (NDArray High, NDArray BoundedAbove) CastHigh(Bound high, double dtypeMax, DType dtype, Shape shape) {
            if (high.Array is null) {
                double value = high.Scalar;
                NDArray boundedAbove = np.full(shape, value < double.PositiveInfinity, np.@bool);
                if (double.IsNaN(value))
                    throw new ValueError($"No high value can be equal to `np.nan`, high={PyBuiltins.FloatRepr(value)}");
                if (double.IsPositiveInfinity(value)) {
                    if (dtype.kind == 'i')
                        value = dtypeMax;
                    else if (dtype.kind is 'u' or 'b')
                        throw new ValueError($"Box unsigned int dtype don't support `np.inf`, high={PyBuiltins.FloatRepr(value)}");
                } else if (value > dtypeMax) {
                    throw new ValueError($"Box high is out of bounds of the dtype range, high={PyBuiltins.FloatRepr(value)}, max dtype={NumberRepr(dtypeMax, dtype)}");
                }

                return (Full(shape, value, dtype), boundedAbove);
            }

            NDArray given = high.Array;
            if (given.dtype.kind is not ('f' or 'i' or 'u' or 'b'))
                throw new ValueError($"Box high must be a floating or integer dtype, actual dtype={Discrete.DTypeName(given.dtype)}");
            using (NDArray nan = np.isnan(given)) {
                if (np.any(nan))
                    throw new ValueError($"No high value can be equal to `np.nan`, high={PyBuiltins.Str(given)}");
            }

            NDArray bounded = given < double.PositiveInfinity;
            using NDArray working = given.copy();
            using (NDArray posinf = np.isposinf(working)) {
                if (np.any(posinf)) {
                    if (dtype.kind == 'i')
                        working[posinf] = dtypeMax;
                    else if (dtype.kind is 'u' or 'b')
                        throw new ValueError($"Box unsigned int dtype don't support `np.inf`, high={PyBuiltins.Str(given)}");
                } else if (working.dtype != dtype) {
                    using NDArray over = working > dtypeMax;
                    if (np.any(over))
                        throw new ValueError($"Box high is out of bounds of the dtype range, high={PyBuiltins.Str(given)}, max dtype={NumberRepr(dtypeMax, dtype)}");
                }
            }

            WarnIfPrecisionLowered("high", given.dtype, dtype);
            return (working.astype(dtype), bounded);
        }

        /// <summary>Gymnasium's warning when a float bound array is cast to a less precise float dtype.</summary>
        /// <param name="which"><c>low</c> or <c>high</c>.</param>
        /// <param name="given">The bound's dtype.</param>
        /// <param name="dtype">The box's dtype.</param>
        private static void WarnIfPrecisionLowered(string which, DType given, DType dtype) {
            if (given.kind == 'f' && dtype.kind == 'f' && np.finfo(dtype).precision < np.finfo(given).precision)
                Logger.Warn($"Box {which}'s precision lowered by casting to {Discrete.DTypeName(dtype)}, current {which}.dtype={Discrete.DTypeName(given)}");
        }

        /// <summary>An array of <paramref name="shape"/> filled with <paramref name="value"/> in <paramref name="dtype"/> (<c>np.full</c>).</summary>
        /// <param name="shape">The shape.</param>
        /// <param name="value">The value, exact in the dtype after the range checks.</param>
        /// <param name="dtype">The dtype.</param>
        /// <returns>A new array.</returns>
        private static NDArray Full(Shape shape, double value, DType dtype) {
            // Integer dtypes get the value as an integer, so int64's extremes are not rounded through a double cast.
            if (dtype.kind is 'i' && value <= long.MinValue)
                return np.full(shape, long.MinValue, dtype);
            if (dtype.kind is 'i' && value >= long.MaxValue)
                return np.full(shape, long.MaxValue, dtype);
            if (dtype.kind == 'u' && value >= ulong.MaxValue)
                return np.full(shape, ulong.MaxValue, dtype);
            return np.full(shape, value, dtype);
        }

        /// <summary>A dtype's minimum or maximum as Gymnasium's f-string prints it: an int for integer dtypes, a float otherwise.</summary>
        /// <param name="value">The value.</param>
        /// <param name="dtype">The dtype.</param>
        /// <returns>The text.</returns>
        private static string NumberRepr(double value, DType dtype) {
            if (dtype.kind is 'f')
                return PyBuiltins.FloatRepr(value);
            if (dtype.typecode == NPTypeCode.UInt64 && value >= ulong.MaxValue)
                return ulong.MaxValue.ToString(CultureInfo.InvariantCulture);
            if (dtype.typecode == NPTypeCode.Int64 && value >= long.MaxValue)
                return long.MaxValue.ToString(CultureInfo.InvariantCulture);
            if (dtype.typecode == NPTypeCode.Int64 && value <= long.MinValue)
                return long.MinValue.ToString(CultureInfo.InvariantCulture);
            return ((long)value).ToString(CultureInfo.InvariantCulture);
        }

        /// <summary>
        ///     A short text for an array (Gymnasium's <c>array_short_repr</c>), what a box's <c>repr</c> prints for each bound:
        ///     the single value when every entry is the same (<c>-1.0</c>), else the whole array (<c>[-1. -2.]</c>).
        /// </summary>
        /// <param name="array">The array; an empty one prints as <c>[]</c>.</param>
        /// <returns>The text.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="array"/> is <see langword="null"/>.</exception>
        /// <remarks>
        ///     Reads every element twice (a min and a max), so it costs two passes over a large array. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/spaces/box.py#L40">gymnasium/spaces/box.py (array_short_repr)</see>.
        /// </remarks>
        public static string ArrayShortRepr(NDArray array) {
            if (array is null)
                throw new ArgumentNullException(nameof(array));
            if (array.size != 0) {
                using NDArray min = np.min(array);
                using NDArray max = np.max(array);
                using NDArray same = min == max;
                if (same.GetBoolean())
                    return PyBuiltins.Str(min);
            }

            return PyBuiltins.Str(array);
        }

        /// <summary>
        ///     Whether a value is a number Gymnasium takes as a scalar bound (its <c>is_float_integer</c>): a Python <c>int</c>
        ///     or <c>float</c> (any .NET integer, an enum, a <see cref="BigInteger"/>, <see langword="double"/>), a NumPy
        ///     integer or floating scalar (<see langword="float"/>, <see cref="Half"/>, a 0-d integer or float array), but never
        ///     a <see langword="bool"/>, which Python counts as an <c>int</c> and Gymnasium excludes by hand.
        /// </summary>
        /// <param name="var">The value.</param>
        /// <returns><see langword="true"/> for an integer or floating scalar that is not a boolean.</returns>
        /// <remarks>
        ///     <see langword="decimal"/> has no Python or NumPy counterpart Gymnasium accepts and is refused. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/spaces/box.py#L57">gymnasium/spaces/box.py (is_float_integer)</see>.
        /// </remarks>
        public static bool IsFloatInteger(object? var) => var switch {
            bool => false,
            NDArray { ndim: 0 } scalar => scalar.dtype.kind is 'i' or 'u' or 'f',
            NDArray => false,
            sbyte or byte or short or ushort or int or uint or long or ulong or BigInteger or Enum => true,
            double or float or Half => true,
            _ => false,
        };

        /// <summary>
        ///     Converts a candidate or JSON value to an array of <paramref name="dtype"/>, NumPy's <c>np.asarray(x, dtype)</c>:
        ///     an array is cast, and a list (any .NET array or list, nested for more dimensions), a number or a string is read
        ///     as the Python value it stands for, each item stored by the dtype as NumPy stores it. <see cref="Graph"/> reads
        ///     JSON edge links and <see cref="SpaceUtils"/> flattens through it too.
        /// </summary>
        /// <param name="value">An array, a .NET array or list, a number, or a string.</param>
        /// <param name="dtype">The dtype.</param>
        /// <returns>A new array, owned by the caller (a copy even when <paramref name="value"/> already is an array of that dtype).</returns>
        /// <exception cref="ArgumentException">Thrown for <see langword="null"/>, ragged nesting, or a value no array can be made from.</exception>
        /// <exception cref="OverflowException">Thrown for a Python int outside an integer dtype's range, NumPy's <c>OverflowError</c> (it never wraps, unlike a cast).</exception>
        /// <exception cref="NumSharp.ValueError">Thrown for a string that is not a number, or a NaN into an integer dtype.</exception>
        /// <exception cref="NumSharp.TypeError">Thrown for an item NumPy cannot store in the dtype.</exception>
        /// <remarks>
        ///     A list is packed item by item (NumPy's <c>PyArray_Pack</c>, which NumSharp's <c>np.asanyarray</c> implements):
        ///     <c>np.asarray([300, 0], dtype=np.int8)</c> raises where a cast would wrap 300 to 44, and int64 values beyond
        ///     2^53 stay exact.
        /// </remarks>
        internal static NDArray AsArray(object? value, DType dtype) {
            switch (value) {
                case null:
                    throw new ArgumentException("None cannot become a numeric array.", nameof(value));
                case NDArray array:
                    return array.astype(dtype);
                case Array or System.Collections.IList:
                    return np.asanyarray(NestedPythonList(value), dtype);
                default:
                    return np.asanyarray(value, dtype);
            }
        }

        /// <summary>
        ///     A .NET array or list as the Python list it stands for, nested lists included: a <see cref="List{T}"/> of the items,
        ///     which NumSharp reads as Python values (a typed array would otherwise be read as an ndarray and cast).
        ///     <see cref="MultiBinary"/> and <see cref="MultiDiscrete"/> read their list candidates and JSON through it too.
        /// </summary>
        /// <param name="value">The array or list.</param>
        /// <returns>The nested lists, or the value itself when it is a leaf.</returns>
        internal static object? NestedPythonList(object? value) {
            switch (value) {
                case string or NDArray:
                    return value;
                case Array { Rank: > 1 } grid: {
                    // A rectangular .NET array (int[,]) as rows of rows, as NumPy reads nested lists.
                    var flat = new List<object?>();
                    foreach (object? item in grid)
                        flat.Add(item);
                    return Rows(flat, Enumerable.Range(0, grid.Rank).Select(grid.GetLength).ToArray(), 0, 0);
                }
                case System.Collections.IList list: {
                    // Before IList would see it, a multidimensional array is folded above: its IList view is flat.
                    var items = new List<object?>(list.Count);
                    foreach (object? item in list)
                        items.Add(NestedPythonList(item));
                    return items;
                }
                default:
                    return value;
            }
        }

        /// <summary>Folds C-ordered items into nested lists of the given lengths.</summary>
        /// <param name="flat">The items in C order.</param>
        /// <param name="lengths">The length of each axis.</param>
        /// <param name="axis">The axis this list walks.</param>
        /// <param name="start">The first item's position in <paramref name="flat"/>.</param>
        /// <returns>The nested list.</returns>
        private static List<object?> Rows(List<object?> flat, int[] lengths, int axis, int start) {
            int stride = 1;
            for (int d = axis + 1; d < lengths.Length; d++)
                stride *= lengths[d];
            var rows = new List<object?>(lengths[axis]);
            for (int i = 0; i < lengths[axis]; i++)
                rows.Add(axis == lengths.Length - 1 ? flat[start + i] : Rows(flat, lengths, axis + 1, start + i * stride));
            return rows;
        }

        /// <summary>An array's entries as doubles in C order, exact for every dtype but 64-bit integers beyond 2^53.</summary>
        /// <param name="array">The array; read, not kept.</param>
        /// <returns>The values.</returns>
        private static double[] ToDoubles(NDArray array) {
            using NDArray asFloat = array.astype(np.float64);
            return asFloat.ToArray<double>();
        }

        /// <summary>An integer array's entries as <see langword="long"/>s in C order (<c>astype("int64")</c>).</summary>
        /// <param name="array">The array; read, not kept.</param>
        /// <returns>The values.</returns>
        private static long[] ToInt64s(NDArray array) {
            using NDArray asInt = array.astype(np.int64);
            return asInt.ToArray<long>();
        }

        /// <summary>Writes draws into their entries: <c>sample[i] = sign * draw + offset[i]</c> (NumPy's <c>-e + high</c> order).</summary>
        /// <param name="sample">The sample being built.</param>
        /// <param name="indices">The category's flat indices.</param>
        /// <param name="draws">One draw per index.</param>
        /// <param name="sign">1, or -1 for the negative exponential.</param>
        /// <param name="offset">The per-entry shift (the bound), or <see langword="null"/> for none.</param>
        private static void Scatter(double[] sample, List<int> indices, double[] draws, double sign, double[]? offset) {
            for (int k = 0; k < indices.Count; k++) {
                int i = indices[k];
                double value = sign < 0 ? -draws[k] : draws[k];
                sample[i] = offset is null ? value : value + offset[i];
            }
        }

        /// <summary>The float64 sample cast to the box's dtype (<c>sample.astype(dtype)</c>): float32 rounds to nearest, integers are already floored and clipped.</summary>
        /// <param name="sample">The values.</param>
        /// <param name="dtype">The dtype.</param>
        /// <param name="shape">The shape.</param>
        /// <returns>A new array.</returns>
        private static NDArray Cast(double[] sample, DType dtype, Shape shape) {
            using NDArray flat = np.array(sample);
            using NDArray shaped = flat.reshape(shape);
            return shaped.astype(dtype);
        }

        /// <summary>Whether two shapes have the same dimensions.</summary>
        /// <param name="left">A shape.</param>
        /// <param name="right">A shape.</param>
        /// <returns><see langword="true"/> when equal.</returns>
        private static bool SameShape(Shape left, Shape right) =>
            (left.dimensions ?? Array.Empty<long>()).SequenceEqual(right.dimensions ?? Array.Empty<long>());

        /// <summary>A shape without an array's view state (offset, strides), only its dimensions.</summary>
        /// <param name="shape">An array's shape.</param>
        /// <returns>A plain shape.</returns>
        private static Shape Clean(Shape shape) => new((long[])(shape.dimensions ?? Array.Empty<long>()).Clone());
    }
}
