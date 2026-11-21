#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using Gym.Utils;
using NumSharp;

namespace Gym.Spaces {
    /// <summary>
    ///     The integers <c>{Start, ..., Start + N - 1}</c>: a categorical action or observation, such as CartPole's two
    ///     pushes or FrozenLake's 16 cells. An element is a 0-d array of <see cref="Space.DType"/> (int64 by default), Gym.NET's
    ///     stand-in for NumPy's <c>np.int64</c> scalar (DR-08).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The shape is <c>()</c>, as in Gymnasium. Code that read the number of actions from <c>Shape.Size</c> must read
    ///     <see cref="N"/> instead (B-03).
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/spaces/discrete.py#L27">gymnasium/spaces/discrete.py</see>.</para>
    /// </remarks>
    public class Discrete : Space<NDArray>, IEquatable<Discrete> {
        /// <summary>The space <c>{start, ..., start + n - 1}</c> (Gymnasium's <c>Discrete(n, seed, start, dtype)</c>).</summary>
        /// <param name="n">The number of elements; must be positive.</param>
        /// <param name="seed">
        ///     An integer seed for the space's own generator (it then samples what Gymnasium's <c>Discrete(n, seed=seed)</c>
        ///     samples), a generator to share, or <see langword="null"/> for one drawn from OS entropy on first use.
        /// </param>
        /// <param name="start">The smallest element.</param>
        /// <param name="dtype">
        ///     The integer dtype of the elements and of <see cref="N"/> and <see cref="Start"/>; <see langword="null"/> is
        ///     int64, Gymnasium's default. Samples are drawn in this dtype, so another dtype draws other values.
        /// </param>
        /// <exception cref="NumSharp.ValueError">Thrown when <paramref name="n"/> is not positive: <c>n (counts) have to be positive, actual value: 0</c>.</exception>
        /// <exception cref="NumSharp.TypeError">Thrown when <paramref name="dtype"/> is not an integer dtype.</exception>
        /// <exception cref="OverflowError">Thrown when <paramref name="n"/> or <paramref name="start"/> does not fit <paramref name="dtype"/>, as NumPy's <c>OverflowError</c>.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public Discrete(long n, SpaceSeed? seed = null, long start = 0, DType? dtype = null)
            : base(new Shape(Array.Empty<long>()), ValidateDType(dtype), null) {
            if (n <= 0)
                throw new ValueError($"n (counts) have to be positive, actual value: {n.ToString(CultureInfo.InvariantCulture)}");
            DType elementType = DType!;
            // self.n = self.dtype.type(n): NumPy refuses a Python int outside the dtype instead of wrapping it.
            N = FitInDType(n, elementType);
            Start = FitInDType(start, elementType);
            // Seeded after validation, as Gymnasium's super().__init__ runs last: a rejected space draws nothing.
            if (seed is { } given) {
                if (given.Generator != null)
                    NpRandom = given.Generator;
                else
                    Seed(given.Value);
            }
        }

        /// <summary>The number of elements (Gymnasium's <c>n</c>, a scalar of the space's dtype).</summary>
        public long N { get; }

        /// <summary>The smallest element (Gymnasium's <c>start</c>, a scalar of the space's dtype).</summary>
        public long Start { get; }

        /// <summary>The element dtype; never <see langword="null"/> for a discrete space (int64 by default).</summary>
        public new DType DType => base.DType!;

        /// <summary>The shape of every element, <c>()</c>.</summary>
        public new Shape Shape => base.Shape!.Value;

        /// <inheritdoc />
        public override bool IsNpFlattenable => true;

        /// <summary>
        ///     Draws one element (Gymnasium's <c>sample</c>): uniformly, uniformly among the elements a mask allows, or by the
        ///     probabilities of a probability mask.
        /// </summary>
        /// <param name="mask">
        ///     <see langword="null"/>, or an int8 array of shape <c>(n,)</c> whose <c>1</c>s mark the allowed elements and
        ///     <c>0</c>s the others; when nothing is allowed, <see cref="Start"/> is returned.
        /// </param>
        /// <param name="probability">
        ///     <see langword="null"/>, or a float64 array of shape <c>(n,)</c> with values in <c>[0, 1]</c> summing to 1 (within
        ///     <c>np.isclose</c>).
        /// </param>
        /// <returns>A new 0-d array of the space's dtype, owned by the caller.</returns>
        /// <exception cref="NumSharp.ValueError">Thrown when both masks are given.</exception>
        /// <exception cref="NumSharp.AssertionError">
        ///     Thrown for a mask or probability of the wrong type, dtype, shape or values, with Gymnasium's assertion messages.
        /// </exception>
        /// <remarks>
        ///     The three branches make Gymnasium's generator calls, <c>choice(np.where(mask == 1)[0])</c>,
        ///     <c>choice(np.arange(n, dtype=dtype), p=probability)</c> and <c>integers(n, dtype=dtype)</c>, so a space seeded
        ///     like Gymnasium's draws the same values. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/spaces/discrete.py#L121">gymnasium/spaces/discrete.py</see>.
        /// </remarks>
        public override NDArray Sample(object? mask = null, object? probability = null) {
            if (mask is not null && probability is not null)
                throw new ValueError($"Only one of `mask` or `probability` can be provided, actual values: mask={PyBuiltins.Str(mask)}, probability={PyBuiltins.Str(probability)}");

            if (mask is not null) {
                NDArray m = mask as NDArray
                            ?? throw new AssertionError($"The expected type of the sample mask is np.ndarray, actual type: {PyBuiltins.TypeRepr(mask)}");
                if (m.dtype != np.int8)
                    throw new AssertionError($"The expected dtype of the sample mask is np.int8, actual dtype: {DTypeName(m.dtype)}");
                if (!IsVectorOfN(m))
                    throw new AssertionError($"The expected shape of the sample mask is ({N.ToString(CultureInfo.InvariantCulture)},), actual shape: {ShapeRepr(m.Shape)}");

                using NDArray valid = m == 1;
                using NDArray zero = m == 0;
                using NDArray either = np.logical_or(zero, valid);
                if (!np.all(either))
                    throw new AssertionError($"All values of the sample mask should be 0 or 1, actual values: {PyBuiltins.Str(m)}");

                if (!np.any(valid))
                    return Scalar(Start);
                // A 1-D mask: np.where returns exactly one index array, int64 like NumPy's intp.
                using NDArray candidates = np.nonzero(valid)[0];
                using NDArray chosen = NpRandom.choice(candidates);
                return Scalar(Start + ToInt64(chosen));
            }

            if (probability is not null) {
                NDArray p = probability as NDArray
                            ?? throw new AssertionError($"The expected type of the sample probability is np.ndarray, actual type: {PyBuiltins.TypeRepr(probability)}");
                if (p.dtype != np.float64)
                    throw new AssertionError($"The expected dtype of the sample probability is np.float64, actual dtype: {DTypeName(p.dtype)}");
                if (!IsVectorOfN(p))
                    throw new AssertionError($"The expected shape of the sample probability is ({N.ToString(CultureInfo.InvariantCulture)},), actual shape: {ShapeRepr(p.Shape)}");

                using NDArray atLeastZero = p >= 0;
                using NDArray atMostOne = p <= 1;
                using NDArray inRange = np.logical_and(atLeastZero, atMostOne);
                if (!np.all(inRange))
                    throw new AssertionError($"All values of the sample probability should be between 0 and 1, actual values: {PyBuiltins.Str(p)}");
                using NDArray total = np.sum(p);
                using NDArray close = np.isclose(total, 1.0);
                if (!close.GetBoolean())
                    throw new AssertionError($"The sum of the sample probability should be equal to 1, actual sum: {PyBuiltins.Str(total)}");

                using NDArray elements = np.arange(N, dtype: DType);
                using NDArray chosen = NpRandom.choice(elements, p: p);
                return Scalar(Start + ToInt64(chosen));
            }

            // Gymnasium: self.start + self.np_random.integers(self.n, dtype=self.dtype.type), i.e. [0, n) drawn in the dtype.
            using NDArray drawn = NpRandom.integers(N, dtype: DType);
            return Scalar(Start + ToInt64(drawn));
        }

        /// <summary>
        ///     Draws one element, typed (DR-86, DR-97): Gymnasium's <c>sample(mask, probability)</c> with both masks as arrays.
        ///     A call on a <see cref="Discrete"/> reference binds this overload unless an argument is typed
        ///     <see langword="object"/>; it draws exactly what <see cref="Sample(object, object)"/> draws.
        /// </summary>
        /// <param name="mask"><see langword="null"/>, or an int8 array of shape <c>(n,)</c> with 1 for the elements that may be drawn (all 0: the start).</param>
        /// <param name="probability"><see langword="null"/>, or a float64 array of shape <c>(n,)</c> with values in <c>[0, 1]</c> summing to 1.</param>
        /// <returns>A new 0-d array of the space's dtype, owned by the caller.</returns>
        /// <exception cref="NumSharp.ValueError">Thrown when both masks are given.</exception>
        /// <exception cref="NumSharp.AssertionError">Thrown for a mask or probability of the wrong dtype, shape or values, with Gymnasium's messages.</exception>
        public NDArray Sample(NDArray? mask = null, NDArray? probability = null) => Sample((object?)mask, (object?)probability);

        /// <summary>
        ///     Reseeds the space's generator (Gymnasium's <c>seed</c>), typed (DR-86): a simple space's seed is one integer, so
        ///     it comes back as a <see cref="BigInteger"/> where <see cref="Space.Seed(BigInteger?)"/> returns it boxed.
        /// </summary>
        /// <param name="seed">Any non-negative integer, or <see langword="null"/> for a fresh generator from OS entropy.</param>
        /// <returns>The seed used: <paramref name="seed"/> itself, or the 128-bit entropy of an unseeded call, which reproduces it.</returns>
        /// <exception cref="Gym.Exceptions.Error">Thrown when <paramref name="seed"/> is negative, with Gymnasium's message.</exception>
        /// <exception cref="InvalidCastException">Thrown by a subclass whose seeding returns something other than one integer.</exception>
        /// <remarks>Footgun: a call through a <see cref="Space"/> reference binds the untyped base method, which returns the same seed as an <see langword="object"/>.</remarks>
        public new BigInteger Seed(BigInteger? seed = null) => (BigInteger)base.Seed(seed);

        /// <summary>
        ///     Whether <paramref name="x"/> is an element (Gymnasium's <c>contains</c>): a .NET integer, <see cref="bool"/> or
        ///     enum member (a Python <c>int</c>: it must fit the dtype), or a 0-d integer array whose dtype casts safely to the
        ///     space's, with a value in <c>[Start, Start + N)</c>.
        /// </summary>
        /// <param name="x">The candidate.</param>
        /// <returns><see langword="true"/> for an element; anything else, a 0-d boolean or floating array included, is <see langword="false"/>.</returns>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/spaces/discrete.py#L202">gymnasium/spaces/discrete.py</see>.</remarks>
        public override bool Contains(object? x) {
            BigInteger value;
            DType valueType;
            if (TryPythonInt(x, out BigInteger pythonInt)) {
                // isinstance(x, int): it must fit the dtype, then it is that dtype's scalar.
                iinfo info = np.iinfo(DType);
                if (pythonInt < info.min || pythonInt > MaxOf(info, DType))
                    return false;
                value = pythonInt;
                valueType = DType;
            } else if (x is NDArray array && array.ndim == 0 && array.dtype.kind is 'i' or 'u') {
                value = array.dtype.kind == 'u' ? new BigInteger(Convert.ToUInt64(array.GetAtIndex(0), CultureInfo.InvariantCulture))
                                                : new BigInteger(Convert.ToInt64(array.GetAtIndex(0), CultureInfo.InvariantCulture));
                valueType = array.dtype;
            } else {
                return false;
            }

            bool inSpace = Start <= value && value < (BigInteger)Start + N;
            return inSpace && np.can_cast(valueType, DType);
        }

        /// <summary>Whether <paramref name="x"/> lies in <c>[Start, Start + N)</c>, as a Python <c>int</c> of the dtype's range.</summary>
        /// <param name="x">The candidate element.</param>
        /// <returns><see langword="true"/> when it is an element.</returns>
        public bool Contains(long x) => Contains((object)x);

        /// <summary>
        ///     Converts an action given to <c>Env.Step(object)</c> to the integer it denotes; envs call this rather than casting,
        ///     because <c>(int)action</c> unboxes exactly and throws for a <see langword="long"/> or a 0-d array.
        /// </summary>
        /// <param name="action">A boxed integer of any width, a <see cref="bool"/>, an enum member, or a 0-d integer array.</param>
        /// <returns>The integer.</returns>
        /// <exception cref="ArgumentException">Thrown when the action is <see langword="null"/>, not an integer, or an array that is not 0-d.</exception>
        /// <exception cref="OverflowException">Thrown when an unsigned 64-bit action exceeds <see cref="long.MaxValue"/>.</exception>
        public static long ToInt64(object action) {
            if (TryToInt64(action, out long value))
                return value;
            throw new ArgumentException(
                $"{action ?? "null"} ({action?.GetType().Name ?? "null"}) is not a discrete action: expected an integer, an enum or a 0-d integer NDArray.",
                nameof(action));
        }

        /// <summary>The non-throwing form of <see cref="ToInt64(object)"/>, for membership tests and action conversion.</summary>
        /// <param name="action">The candidate.</param>
        /// <param name="value">The integer when <see langword="true"/> is returned; otherwise 0.</param>
        /// <returns><see langword="true"/> when the candidate denotes an integer.</returns>
        /// <exception cref="OverflowException">Thrown when an unsigned 64-bit value exceeds <see cref="long.MaxValue"/>.</exception>
        /// <remarks>
        ///     Accepts what Gymnasium's envs accept as a discrete action: Python ints (a <see cref="bool"/> is one, as
        ///     Python's <c>bool</c> subclasses <c>int</c>) and 0-d arrays of an integer dtype. A 0-d boolean array is rejected,
        ///     because <c>np.bool_</c> is not an <c>np.integer</c>.
        /// </remarks>
        public static bool TryToInt64(object? action, out long value) {
            switch (action) {
                case long l: value = l; return true;
                case int i: value = i; return true;
                case short s: value = s; return true;
                case sbyte sb: value = sb; return true;
                case byte b: value = b; return true;
                case ushort us: value = us; return true;
                case uint ui: value = ui; return true;
                case ulong ul: value = checked((long)ul); return true;
                case BigInteger big: value = (long)big; return true;
                case bool flag: value = flag ? 1 : 0; return true;
                case Enum e: value = Convert.ToInt64(e, CultureInfo.InvariantCulture); return true;
                case NDArray nd when nd.ndim == 0 && nd.dtype.kind is 'i' or 'u':
                    // NumSharp's (long) is a C-style cast that wraps a uint64 above long.MaxValue (measured), so that
                    // dtype takes a checked conversion.
                    value = nd.typecode == NPTypeCode.UInt64 ? checked((long)Convert.ToUInt64(nd.GetAtIndex(0), CultureInfo.InvariantCulture))
                                                             : Convert.ToInt64(nd.GetAtIndex(0), CultureInfo.InvariantCulture);
                    return true;
                default:
                    value = 0;
                    return false;
            }
        }

        /// <summary>
        ///     Converts samples to Python ints (Gymnasium's <c>to_jsonable</c>: <c>[int(x) for x in sample_n]</c>), as
        ///     <see langword="long"/>s.
        /// </summary>
        /// <param name="samples">Samples of this space.</param>
        /// <returns>One integer per sample.</returns>
        /// <exception cref="ArgumentException">Thrown for a sample that is not a 0-d integer array.</exception>
        public override object ToJsonable(IEnumerable<NDArray> samples) => samples.Select(s => (object?)ToInt64(s)).ToList();

        /// <summary>Converts integers back to 0-d arrays of the space's dtype (Gymnasium's <c>from_jsonable</c>).</summary>
        /// <param name="jsonable">Integers, as <see cref="ToJsonable(IEnumerable{NDArray})"/> made them.</param>
        /// <returns>New 0-d arrays, owned by the caller.</returns>
        /// <exception cref="ArgumentException">Thrown for a value that is not an integer.</exception>
        public override IList<NDArray> FromJsonable(object jsonable) => JsonList(jsonable).Select(v => Scalar(ToInt64(v!))).ToList();

        /// <summary>
        ///     Gymnasium's <c>repr</c>: <c>Discrete(2)</c>, with <c>start=…</c> when it is not 0 and <c>dtype=…</c> when it is
        ///     not int64.
        /// </summary>
        /// <returns>The text.</returns>
        public override string ToString() {
            var info = new List<string> { N.ToString(CultureInfo.InvariantCulture) };
            if (Start != 0)
                info.Add($"start={Start.ToString(CultureInfo.InvariantCulture)}");
            if (DType != np.int64)
                info.Add($"dtype={DTypeName(DType)}");
            return $"Discrete({string.Join(", ", info)})";
        }

        /// <summary>Gymnasium's <c>==</c>: the same <see cref="N"/>, <see cref="Start"/> and dtype.</summary>
        /// <param name="other">The other space.</param>
        /// <returns><see langword="true"/> when equal.</returns>
        public bool Equals(Discrete? other) => other is not null && N == other.N && Start == other.Start && DType == other.DType;

        /// <inheritdoc />
        public override bool Equals(object? obj) => obj is Discrete other && Equals(other);

        /// <inheritdoc />
        public override int GetHashCode() => HashCode.Combine(N, Start, DTypeName(DType));

        /// <summary>The dtype a constructor was given, validated: <see langword="null"/> is int64, anything else must be an integer dtype.</summary>
        /// <param name="dtype">The dtype.</param>
        /// <returns>The dtype to use.</returns>
        /// <exception cref="NumSharp.TypeError">Thrown for a dtype that is not an integer dtype.</exception>
        private static DType ValidateDType(DType? dtype) {
            DType resolved = dtype ?? np.int64;
            if (resolved.kind is not ('i' or 'u'))
                throw new TypeError($"Invalid Discrete dtype ({DTypeName(resolved)}), must be an integer dtype");
            return resolved;
        }

        /// <summary>A value as a scalar of the dtype, refused when it does not fit, as NumPy's <c>dtype.type(x)</c> refuses a Python int.</summary>
        /// <param name="value">The value.</param>
        /// <param name="dtype">The integer dtype.</param>
        /// <returns>The value.</returns>
        /// <exception cref="OverflowError">Thrown when the value is outside the dtype: <c>Python integer 300 out of bounds for uint8</c>.</exception>
        private static long FitInDType(long value, DType dtype) {
            iinfo info = np.iinfo(dtype);
            if (value < info.min || value > MaxOf(info, dtype))
                throw new OverflowError($"Python integer {value.ToString(CultureInfo.InvariantCulture)} out of bounds for {DTypeName(dtype)}");
            return value;
        }

        /// <summary>A dtype's maximum as an exact integer, also for uint64, whose maximum exceeds <see cref="long"/>.</summary>
        /// <param name="info">The dtype's <c>iinfo</c>.</param>
        /// <param name="dtype">The dtype.</param>
        /// <returns>The maximum.</returns>
        private static BigInteger MaxOf(iinfo info, DType dtype) => dtype.typecode == NPTypeCode.UInt64 ? ulong.MaxValue : info.max;

        /// <summary>Whether a Python value is a Python <c>int</c> (a .NET integer, <see cref="bool"/> or enum member), and its value.</summary>
        /// <param name="x">The value.</param>
        /// <param name="value">The integer.</param>
        /// <returns><see langword="true"/> for a Python int.</returns>
        private static bool TryPythonInt(object? x, out BigInteger value) {
            switch (x) {
                case ulong ul:
                    value = ul;
                    return true;
                case BigInteger big:
                    value = big;
                    return true;
                case Enum e: {
                    // Through the underlying type, so a ulong-backed member above long.MaxValue stays exact (Python's
                    // IntEnum is an int of any size) instead of overflowing a conversion to long.
                    object underlying = Convert.ChangeType(e, Enum.GetUnderlyingType(e.GetType()), CultureInfo.InvariantCulture);
                    value = underlying is ulong large ? new BigInteger(large) : new BigInteger(Convert.ToInt64(underlying, CultureInfo.InvariantCulture));
                    return true;
                }
                case NDArray:
                case null:
                    value = default;
                    return false;
                default:
                    if (TryToInt64(x, out long l)) {
                        value = l;
                        return true;
                    }

                    value = default;
                    return false;
            }
        }

        /// <summary>Whether an array has shape <c>(n,)</c>.</summary>
        /// <param name="array">The array.</param>
        /// <returns><see langword="true"/> for a 1-D array of <see cref="N"/> elements.</returns>
        private bool IsVectorOfN(NDArray array) => array.ndim == 1 && array.shape[0] == N;

        /// <summary>A 0-d array of the space's dtype holding <paramref name="value"/>, NumPy's <c>dtype.type(value)</c> scalar.</summary>
        /// <param name="value">The value, which fits the dtype for every element of the space.</param>
        /// <returns>A new 0-d array, owned by the caller.</returns>
        private NDArray Scalar(long value) {
            using NDArray int64 = NDArray.Scalar<long>(value);
            return int64.astype(DType);
        }

        /// <summary>The integer a 0-d draw holds.</summary>
        /// <param name="draw">A 0-d integer array; the caller keeps ownership.</param>
        /// <returns>The value.</returns>
        /// <exception cref="InvalidOperationException">Thrown when the draw is not 0-d, which would surface a NumSharp regression to the 1-element shape of 0.70.</exception>
        private static long ToInt64(NDArray draw) {
            if (draw.ndim != 0)
                throw new InvalidOperationException($"Expected a 0-d drawn value, got shape {draw.Shape}.");
            return Convert.ToInt64(draw.GetAtIndex(0), CultureInfo.InvariantCulture);
        }

        /// <summary>A dtype's NumPy name, <c>int64</c>, <c>uint8</c>: what <c>str(dtype)</c> prints.</summary>
        /// <param name="dtype">The dtype.</param>
        /// <returns>The name.</returns>
        internal static string DTypeName(DType dtype) => dtype.name;
    }
}
