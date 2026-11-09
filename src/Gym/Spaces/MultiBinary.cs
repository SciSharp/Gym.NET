#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Gym.Utils;
using NumSharp;

namespace Gym.Spaces {
    /// <summary>
    ///     The binary arrays of a fixed shape: every element is an int8 array of 0s and 1s, such as a set of on/off switches.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/spaces/multi_binary.py#L15">gymnasium/spaces/multi_binary.py</see>.</remarks>
    public class MultiBinary : Space<NDArray>, IEquatable<MultiBinary> {
        /// <summary>Whether <see cref="N"/> was given as one integer (Gymnasium's <c>n</c> is then an <c>int</c>, else a tuple).</summary>
        private readonly bool _scalarN;

        /// <summary>The flat space of <paramref name="n"/> binary values (Gymnasium's <c>MultiBinary(n)</c>).</summary>
        /// <param name="n">The number of values; must be positive.</param>
        /// <param name="seed">An integer seed, a generator to share, or <see langword="null"/> for a lazy entropy-seeded one.</param>
        /// <exception cref="NumSharp.ValueError">Thrown when <paramref name="n"/> is not positive: <c>n (counts) have to be positive, got 0</c>.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public MultiBinary(long n, SpaceSeed? seed = null) : this(new[] { n }, scalar: true, seed) { }

        /// <summary>The space of binary arrays of shape <paramref name="n"/> (Gymnasium's <c>MultiBinary([3, 2])</c>).</summary>
        /// <param name="n">The shape; every dimension must be positive.</param>
        /// <param name="seed">An integer seed, a generator to share, or <see langword="null"/>.</param>
        /// <exception cref="NumSharp.ValueError">Thrown when a dimension is not positive.</exception>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="n"/> is <see langword="null"/>.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public MultiBinary(long[] n, SpaceSeed? seed = null) : this(n ?? throw new ArgumentNullException(nameof(n)), scalar: false, seed) { }

        /// <summary>The shared constructor.</summary>
        /// <param name="n">The dimensions.</param>
        /// <param name="scalar">Whether <paramref name="n"/> was one integer.</param>
        /// <param name="seed">The seed.</param>
        /// <exception cref="NumSharp.ValueError">Thrown when a dimension is not positive.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        private MultiBinary(long[] n, bool scalar, SpaceSeed? seed) : base(new Shape((long[])n.Clone()), np.int8, null) {
            _scalarN = scalar;
            N = (long[])n.Clone();
            if (N.Any(d => d <= 0))
                throw new ValueError($"n (counts) have to be positive, got {NRepr(N, scalar, asGiven: true)}");
            // Seeded after validation, as Gymnasium's super().__init__ runs last.
            if (seed is { } given) {
                if (given.Generator != null)
                    NpRandom = given.Generator;
                else
                    Seed(given.Value);
            }
        }

        /// <summary>The shape of the elements, Gymnasium's <c>n</c> (an <c>int</c> or a tuple; <see cref="ToString"/> tells which).</summary>
        public IReadOnlyList<long> N { get; }

        /// <summary>The element dtype, int8.</summary>
        public new DType DType => base.DType!;

        /// <summary>The element shape.</summary>
        public new Shape Shape => base.Shape!.Value;

        /// <inheritdoc />
        public override bool IsNpFlattenable => true;

        /// <summary>
        ///     Draws one element by a fair coin toss per value (Gymnasium's <c>sample</c>), or under a mask that fixes values
        ///     (<c>0</c> or <c>1</c>) or leaves them random (<c>2</c>), or with a per-value probability of 1.
        /// </summary>
        /// <param name="mask"><see langword="null"/>, or an int8 array of the space's shape holding 0, 1 or 2.</param>
        /// <param name="probability"><see langword="null"/>, or a float64 array of the space's shape with values in <c>[0, 1]</c>.</param>
        /// <returns>A new int8 array, owned by the caller.</returns>
        /// <exception cref="NumSharp.ValueError">Thrown when both masks are given.</exception>
        /// <exception cref="NumSharp.AssertionError">Thrown for a mask or probability of the wrong type, dtype, shape or values, with Gymnasium's messages.</exception>
        /// <remarks>
        ///     A mask draws a full array of coin tosses and keeps the ones the mask leaves random (<c>np.where(mask == 2,
        ///     integers(0, 2, size=n, dtype=int8), mask)</c>), so the generator moves by the whole shape either way, as in
        ///     Gymnasium. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/spaces/multi_binary.py#L73">gymnasium/spaces/multi_binary.py</see>.
        /// </remarks>
        public override NDArray Sample(object? mask = null, object? probability = null) {
            if (mask is not null && probability is not null)
                throw new ValueError($"Only one of `mask` or `probability` can be provided, actual values: mask={PyBuiltins.Str(mask)}, probability={PyBuiltins.Str(probability)}");

            if (mask is not null) {
                NDArray m = mask as NDArray ?? throw new AssertionError($"The expected type of the mask is np.ndarray, actual type: {PyBuiltins.TypeRepr(mask)}");
                if (m.dtype != np.int8)
                    throw new AssertionError($"The expected dtype of the mask is np.int8, actual dtype: {Discrete.DTypeName(m.dtype)}");
                if (!SameShape(m.Shape))
                    throw new AssertionError($"The expected shape of the mask is {ShapeRepr(Shape)}, actual shape: {ShapeRepr(m.Shape)}");
                using NDArray zero = m == 0;
                using NDArray one = m == 1;
                using NDArray two = m == 2;
                using NDArray zeroOrOne = np.logical_or(zero, one);
                using NDArray allowed = np.logical_or(zeroOrOne, two);
                if (!np.all(allowed))
                    throw new AssertionError($"All values of a mask should be 0, 1 or 2, actual values: {PyBuiltins.Str(m)}");
                using NDArray tosses = NpRandom.integers(0, 2, size: Shape, dtype: np.int8);
                using NDArray fixedValues = m.astype(np.int8);
                return np.where(two, tosses, fixedValues);
            }

            if (probability is not null) {
                NDArray p = probability as NDArray ?? throw new AssertionError($"The expected type of the probability is np.ndarray, actual type: {PyBuiltins.TypeRepr(probability)}");
                if (p.dtype != np.float64)
                    throw new AssertionError($"The expected dtype of the probability is np.float64, actual dtype: {Discrete.DTypeName(p.dtype)}");
                // Gymnasium's message prints the probability itself where it means its shape; kept as it is.
                if (!SameShape(p.Shape))
                    throw new AssertionError($"The expected shape of the probability is {ShapeRepr(Shape)}, actual shape: {PyBuiltins.Str(p)}");
                using NDArray atLeastZero = p >= 0;
                using NDArray atMostOne = p <= 1;
                using NDArray inRange = np.logical_and(atLeastZero, atMostOne);
                if (!np.all(inRange))
                    throw new AssertionError($"All values of the sample probability should be between 0 and 1, actual values: {PyBuiltins.Str(p)}");
                using NDArray uniform = NpRandom.random(Shape);
                using NDArray ones = uniform <= p;
                return ones.astype(np.int8);
            }

            return NpRandom.integers(0, 2, size: Shape, dtype: np.int8);
        }

        /// <summary>
        ///     Whether <paramref name="x"/> is an element (Gymnasium's <c>contains</c>): an array (a .NET array or list is
        ///     converted first) of the space's shape whose values are all 0 or 1. The dtype is not checked, as in Gymnasium.
        /// </summary>
        /// <param name="x">The candidate.</param>
        /// <returns><see langword="true"/> for an element.</returns>
        public override bool Contains(object? x) {
            NDArray? owned = null;
            try {
                NDArray? array = x switch {
                    NDArray nd => nd,
                    Array a => owned = np.array(a),
                    System.Collections.IList list => owned = SequenceToArray(list),
                    _ => null,
                };
                if (array is null || !SameShape(array.Shape))
                    return false;
                using NDArray zero = array == 0;
                using NDArray one = array == 1;
                using NDArray binary = np.logical_or(zero, one);
                return np.all(binary);
            } catch (Exception e) when (e is ArgumentException or InvalidCastException or NotSupportedException or ValueError or TypeError) {
                // A list np.array cannot read (ragged, non-numeric) is not a binary array.
                return false;
            } finally {
                owned?.Dispose();
            }
        }

        /// <summary>Converts samples to nested lists of ints (Gymnasium's <c>np.array(list(sample_n)).tolist()</c>).</summary>
        /// <param name="samples">Samples of this space.</param>
        /// <returns>One nested list per sample.</returns>
        public override object ToJsonable(IEnumerable<NDArray> samples) => samples.Select(s => PythonList(s)).ToList();

        /// <summary>Converts nested lists back to int8 arrays (Gymnasium's <c>np.asarray(sample, int8)</c>).</summary>
        /// <param name="jsonable">A list of nested lists.</param>
        /// <returns>New int8 arrays, owned by the caller.</returns>
        /// <exception cref="ArgumentException">Thrown for a value no int8 array can be made from.</exception>
        public override IList<NDArray> FromJsonable(object jsonable) => JsonList(jsonable).Select(v => ToInt8(v)).ToList();

        /// <summary>Gymnasium's <c>repr</c>: <c>MultiBinary(5)</c>, or <c>MultiBinary((3, 2))</c> for a shape.</summary>
        /// <returns>The text.</returns>
        public override string ToString() => $"MultiBinary({NRepr(N, _scalarN, asGiven: false)})";

        /// <summary>Gymnasium's <c>==</c>: the same <c>n</c>, an <c>int</c> differing from a one-element tuple.</summary>
        /// <param name="other">The other space.</param>
        /// <returns><see langword="true"/> when equal.</returns>
        public bool Equals(MultiBinary? other) => other is not null && _scalarN == other._scalarN && N.SequenceEqual(other.N);

        /// <inheritdoc />
        public override bool Equals(object? obj) => obj is MultiBinary other && Equals(other);

        /// <inheritdoc />
        public override int GetHashCode() => HashCode.Combine(_scalarN, NRepr(N, _scalarN, asGiven: false));

        /// <summary><c>n</c> as Python prints it: an int, or a tuple (or, for the error message, the list it was given as).</summary>
        /// <param name="n">The dimensions.</param>
        /// <param name="scalar">Whether it was one integer.</param>
        /// <param name="asGiven">Whether to print a sequence as the list a caller passed, rather than the tuple the space keeps.</param>
        /// <returns>The text.</returns>
        private static string NRepr(IReadOnlyList<long> n, bool scalar, bool asGiven) {
            if (scalar)
                return n[0].ToString(CultureInfo.InvariantCulture);
            string items = string.Join(", ", n.Select(d => d.ToString(CultureInfo.InvariantCulture)));
            if (asGiven)
                return "[" + items + "]";
            return n.Count == 1 ? "(" + items + ",)" : "(" + items + ")";
        }

        /// <summary>Whether an array's shape is the space's.</summary>
        /// <param name="shape">The array's shape.</param>
        /// <returns><see langword="true"/> when equal.</returns>
        private bool SameShape(Shape shape) => (shape.dimensions ?? Array.Empty<long>()).SequenceEqual(N);

        /// <summary>
        ///     A (possibly nested) .NET list as an array, Python's <c>np.array(list)</c>: NumPy's dtype discovery over the
        ///     Python values the items stand for (ints of any size, a <see cref="System.Numerics.BigInteger"/> included, floats,
        ///     booleans), through NumSharp's own coercion of a Python list.
        /// </summary>
        /// <param name="list">The list.</param>
        /// <returns>A new array.</returns>
        /// <exception cref="ArgumentException">Thrown for a list NumPy cannot read (ragged nesting, or an item no numeric array holds).</exception>
        /// <exception cref="NotSupportedException">Thrown for a list NumPy would make an object array of (a <c>None</c>, an int past 64 bits).</exception>
        internal static NDArray SequenceToArray(System.Collections.IList list) => np.asanyarray(Box.NestedPythonList(list));

        /// <summary>A JSON value as an int8 array.</summary>
        /// <param name="value">A nested list, an array or an NDArray.</param>
        /// <returns>A new int8 array.</returns>
        /// <exception cref="ArgumentException">Thrown for a value that is not a list or an array.</exception>
        private static NDArray ToInt8(object? value) {
            switch (value) {
                case NDArray nd:
                    return nd.astype(np.int8);
                case Array a:
                    using (NDArray made = np.array(a))
                        return made.astype(np.int8);
                case System.Collections.IList list:
                    using (NDArray made = SequenceToArray(list))
                        return made.astype(np.int8);
                default:
                    throw new ArgumentException($"{PyBuiltins.Repr(value)} is not a MultiBinary sample.", nameof(value));
            }
        }
    }
}
