#nullable enable
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using Gym.Utils;
using NumSharp;

namespace Gym.Spaces {
    /// <summary>
    ///     The direct sum of spaces: every element is an element of exactly one subspace, tagged with its index, as the pair
    ///     <c>(index, value)</c> (an <see langword="object"/>?[] whose index is a 0-d int64 array, NumPy's <c>np.int64</c>).
    ///     Use it for an action that is one of several kinds, each with its own parameters.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A sample first draws the subspace uniformly, then samples it; a mask or probability mask is a tuple with one entry
    ///     per subspace, of which only the drawn subspace's is used.
    ///     </para>
    ///     <para>
    ///     The space owns its subspaces: disposing it disposes them. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/spaces/oneof.py#L15">gymnasium/spaces/oneof.py</see>.
    ///     </para>
    /// </remarks>
    public class OneOf : Space<object?[]>, IReadOnlyList<Space>, IEquatable<OneOf> {
        /// <summary>The subspaces.</summary>
        private readonly Space[] _spaces;

        /// <summary>The sum of <paramref name="spaces"/> (Gymnasium's <c>OneOf(spaces, seed=seed)</c>).</summary>
        /// <param name="spaces">The subspaces, in order; at least one. The space owns them from now on.</param>
        /// <param name="seed">
        ///     An integer seed (it seeds the space's generator, which draws one <c>int32</c> seed per subspace, then is seeded
        ///     again with the same value), a generator to share, or <see langword="null"/>.
        /// </param>
        /// <exception cref="NumSharp.TypeError">Thrown when <paramref name="spaces"/> or a subspace is <see langword="null"/>, with Gymnasium's messages.</exception>
        /// <exception cref="NumSharp.ValueError">Thrown for no subspaces: <c>Empty `OneOf` spaces are not supported.</c></exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public OneOf(IEnumerable<Space> spaces, SpaceSeed? seed = null) : base(null, null, null) {
            _spaces = (spaces ?? throw new TypeError("None is not an iterable")).ToArray();
            if (_spaces.Length == 0)
                throw new ValueError("Empty `OneOf` spaces are not supported.");
            foreach (Space? space in _spaces) {
                if (space is null)
                    throw new TypeError("None does not inherit from `gymnasium.Space`. Actual Type: <class 'NoneType'>");
            }

            // Gymnasium's super().__init__(None, None, seed) runs self.seed(seed), which needs the subspaces: seeded here.
            if (seed is { } given) {
                if (given.Generator != null)
                    NpRandom = given.Generator;
                else
                    Seed(given.Value);
            }
        }

        /// <summary>The sum of the given subspaces.</summary>
        /// <param name="spaces">The subspaces, in order; at least one.</param>
        /// <exception cref="NumSharp.TypeError">Thrown for a <see langword="null"/> subspace.</exception>
        /// <exception cref="NumSharp.ValueError">Thrown for no subspaces.</exception>
        public OneOf(params Space[] spaces) : this((IEnumerable<Space>)spaces) { }

        /// <summary>The subspaces (Gymnasium's <c>spaces</c>).</summary>
        public IReadOnlyList<Space> Spaces => _spaces;

        /// <summary>The number of subspaces (Gymnasium's <c>len(space)</c>).</summary>
        public int Count => _spaces.Length;

        /// <summary>The subspace at <paramref name="index"/> (Gymnasium's <c>space[index]</c>).</summary>
        /// <param name="index">The index; negative counts from the end, as in Python.</param>
        /// <returns>The subspace.</returns>
        /// <exception cref="NumSharp.IndexError">Thrown for an index outside the space: <c>tuple index out of range</c>.</exception>
        public Space this[int index] {
            get {
                int resolved = index < 0 ? index + _spaces.Length : index;
                if (resolved < 0 || resolved >= _spaces.Length)
                    throw new IndexError("tuple index out of range");
                return _spaces[resolved];
            }
        }

        /// <summary>Whether every subspace flattens to a box.</summary>
        public override bool IsNpFlattenable => _spaces.All(s => s.IsNpFlattenable);

        /// <summary>
        ///     Draws one element (Gymnasium's <c>sample</c>): a subspace index, uniformly, then a sample of that subspace under
        ///     its entry of <paramref name="mask"/> or <paramref name="probability"/>.
        /// </summary>
        /// <param name="mask"><see langword="null"/>, or a tuple (an <see langword="object"/>?[] or an <see cref="ITuple"/>) with one mask (or <see langword="null"/>) per subspace.</param>
        /// <param name="probability"><see langword="null"/>, or a tuple with one probability mask (or <see langword="null"/>) per subspace.</param>
        /// <returns>A new pair <c>{index, value}</c>; the caller owns the index array and any arrays in the value.</returns>
        /// <exception cref="NumSharp.ValueError">Thrown when both masks are given.</exception>
        /// <exception cref="NumSharp.AssertionError">Thrown for a mask that is not a tuple of one entry per subspace, with Gymnasium's messages.</exception>
        /// <remarks>
        ///     The index is drawn before the arguments are checked, as in Gymnasium, so a refused call still moves the
        ///     generator.
        /// </remarks>
        public override object?[] Sample(object? mask = null, object? probability = null) {
            NDArray index = NpRandom.integers(0, (long)_spaces.Length, dtype: np.int64);
            try {
                int position = (int)Convert.ToInt64(index.GetAtIndex(0), CultureInfo.InvariantCulture);
                Space subspace = _spaces[position];
                object? value;
                if (mask is not null && probability is not null)
                    throw new ValueError($"Only one of `mask` or `probability` can be provided, actual values: mask={PyBuiltins.Str(mask)}, probability={PyBuiltins.Str(probability)}");
                if (mask is not null)
                    value = subspace.Sample(mask: TupleArgument(mask, "mask")[position]);
                else if (probability is not null)
                    value = subspace.Sample(probability: TupleArgument(probability, "probability")[position]);
                else
                    value = subspace.Sample();
                return new object?[] { index, value };
            } catch {
                // The index never reaches the caller on failure, so it is released here.
                index.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     Whether <paramref name="x"/> is an element (Gymnasium's <c>contains</c>): a pair whose index is a Python
        ///     <c>int</c> or an <c>np.int64</c> within the subspaces and whose value belongs to that subspace.
        /// </summary>
        /// <param name="x">The candidate: an <see langword="object"/>?[] or an <see cref="ITuple"/> of two items.</param>
        /// <returns><see langword="true"/> for an element.</returns>
        /// <remarks>
        ///     An index of another NumPy integer type (a 0-d int32 array) is refused, as Gymnasium's
        ///     <c>isinstance(x[0], (np.int64, int))</c> refuses <c>np.int32</c>; a <see langword="bool"/> is a Python <c>int</c>.
        /// </remarks>
        public override bool Contains(object? x) {
            object?[]? items = x switch {
                object?[] array => array,
                ITuple tuple => Tuple.TupleItems(tuple),
                _ => null,
            };
            if (items is null || items.Length != 2 || !TryIndex(items[0], out BigInteger index))
                return false;
            return index >= 0 && index < _spaces.Length && _spaces[(int)index].Contains(items[1]);
        }

        /// <summary>
        ///     Converts samples to Gymnasium's JSON form: one <c>[index, value]</c> list per sample, the value being the first
        ///     item of the subspace's <c>to_jsonable</c> of that one sample.
        /// </summary>
        /// <param name="samples">Samples of this space.</param>
        /// <returns>The list of pairs.</returns>
        /// <exception cref="NumSharp.KeyError">
        ///     Thrown for a <see cref="Dict"/> subspace, whose JSON form is a dict: Gymnasium indexes it with <c>[0]</c> and
        ///     raises <c>KeyError: 0</c>, and so does this.
        /// </exception>
        public override object ToJsonable(IEnumerable<object?[]> samples) =>
            samples.Select(sample => {
                long index = ToIndex(sample[0]);
                object json = this[(int)index].ToJsonable(new[] { sample[1] });
                return (object?)new List<object?> { index, FirstJsonItem(json) };
            }).ToList();

        /// <summary>Converts Gymnasium's <c>[index, value]</c> lists back to pairs (Gymnasium's <c>from_jsonable</c>).</summary>
        /// <param name="jsonable">The list of pairs.</param>
        /// <returns>The samples; each index is a new 0-d int64 array.</returns>
        /// <exception cref="ArgumentException">Thrown when the data is not a list of pairs.</exception>
        public override IList<object?[]> FromJsonable(object jsonable) =>
            JsonList(jsonable).Select(item => {
                IList<object?> pair = JsonList(item ?? throw new ArgumentException("Expected an [index, value] pair, got null.", nameof(jsonable)));
                long index = ToIndex(pair[0]);
                object? value = this[(int)index].FromJsonable(new List<object?> { pair[1] })[0];
                return new object?[] { NDArray.Scalar(index), value };
            }).ToList();

        /// <summary>Gymnasium's <c>repr</c>: <c>OneOf(Discrete(2), Box(-1.0, 1.0, (2,), float32))</c>.</summary>
        /// <returns>The text.</returns>
        public override string ToString() => "OneOf(" + string.Join(", ", _spaces.Select(s => s.ToString())) + ")";

        /// <summary>Gymnasium's <c>==</c>: the same subspaces, equal in order.</summary>
        /// <param name="other">The other space.</param>
        /// <returns><see langword="true"/> when equal.</returns>
        public bool Equals(OneOf? other) => other is not null && _spaces.Length == other._spaces.Length && _spaces.Zip(other._spaces).All(p => p.First.Equals(p.Second));

        /// <inheritdoc />
        public override bool Equals(object? obj) => obj is OneOf other && Equals(other);

        /// <inheritdoc />
        public override int GetHashCode() => HashCode.Combine(_spaces.Length, _spaces[0].GetHashCode());

        /// <inheritdoc />
        public IEnumerator<Space> GetEnumerator() => ((IEnumerable<Space>)_spaces).GetEnumerator();

        /// <inheritdoc />
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        /// <summary>
        ///     Gymnasium's <c>OneOf.seed</c> for an integer or <see langword="null"/>: <see langword="null"/> seeds the space
        ///     and every subspace from entropy; an integer seeds the space's generator, draws one seed below <c>int32</c>'s
        ///     maximum per subspace, and seeds the generator again with the same value, so integer and list seeding leave it
        ///     alike.
        /// </summary>
        /// <param name="seed">The seed, or <see langword="null"/>.</param>
        /// <returns>The space's seed followed by the subspaces', as a list.</returns>
        /// <exception cref="Gym.Exceptions.Error">Thrown for a negative seed.</exception>
        protected override object SeedCore(BigInteger? seed) {
            if (seed is null) {
                var entropySeeds = new List<object> { base.SeedCore(null) };
                entropySeeds.AddRange(_spaces.Select(space => space.Seed((BigInteger?)null)));
                return entropySeeds;
            }

            object superSeed = base.SeedCore(seed);
            using NDArray subseeds = NpRandom.integers(int.MaxValue, size: new Shape((long)_spaces.Length));
            base.SeedCore(seed);
            var seeds = new List<object> { superSeed };
            seeds.AddRange(_spaces.Select((space, i) => space.Seed(new BigInteger(Convert.ToInt64(subseeds.GetAtIndex(i), CultureInfo.InvariantCulture)))));
            return seeds;
        }

        /// <summary>Gymnasium's <c>OneOf.seed</c> for a list: the space's own seed, then one seed (of any form) per subspace.</summary>
        /// <param name="seeds">The seeds, one more than the subspaces.</param>
        /// <returns>The seeds used.</returns>
        /// <exception cref="NumSharp.ValueError">Thrown when the list's length is not the number of subspaces + 1.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when the space's own seed is negative or not a Python <c>int</c>.</exception>
        protected override object SeedListCore(IReadOnlyList<object?> seeds) {
            if (seeds.Count != _spaces.Length + 1)
                throw new ValueError($"Expects that the subspaces of seeds equals the number of subspaces + 1. Actual length of seeds: {seeds.Count}, length of subspaces: {_spaces.Length}");
            var used = new List<object> { SeedOwnGenerator(seeds[0]) };
            used.AddRange(_spaces.Select((space, i) => space.SeedWith(seeds[i + 1])));
            return used;
        }

        /// <summary>Gymnasium's <c>OneOf.seed</c> error for a seed that is not an integer, a list, a tuple or <see langword="null"/>.</summary>
        /// <param name="seed">The refused seed.</param>
        /// <returns><c>TypeError: Expected None, int, or tuple of ints, actual type: &lt;class 'dict'&gt;</c>.</returns>
        protected override Exception SeedTypeError(object? seed) =>
            new TypeError($"Expected None, int, or tuple of ints, actual type: {PyBuiltins.TypeRepr(seed)}");

        /// <inheritdoc />
        protected override void Dispose(bool disposing) {
            if (disposing) {
                foreach (Space space in _spaces)
                    space.Dispose();
            }

            base.Dispose(disposing);
        }

        /// <summary>A tuple argument's items, with Gymnasium's <c>OneOf</c> assertions (worded unlike <see cref="Tuple"/>'s).</summary>
        /// <param name="argument">The mask or probability.</param>
        /// <param name="name"><c>mask</c> or <c>probability</c>, for the message.</param>
        /// <returns>The items.</returns>
        /// <exception cref="NumSharp.AssertionError">Thrown when it is not a tuple of one item per subspace.</exception>
        private object?[] TupleArgument(object argument, string name) {
            object?[] items = argument switch {
                object?[] array => array,
                ITuple tuple => Tuple.TupleItems(tuple),
                _ => throw new AssertionError($"Expected type of `{name}` is tuple, actual type: {PyBuiltins.TypeRepr(argument)}"),
            };
            if (items.Length != _spaces.Length)
                throw new AssertionError($"Expected length of `{name}` is {_spaces.Length}, actual length: {items.Length}");
            return items;
        }

        /// <summary>
        ///     Whether a pair's first item is an index Gymnasium's <c>contains</c> accepts: a Python <c>int</c> (any .NET
        ///     integer, an enum, a <see langword="bool"/>, a <see cref="BigInteger"/>) or a 0-d int64 array (<c>np.int64</c>).
        /// </summary>
        /// <param name="value">The candidate index.</param>
        /// <param name="index">The index, when accepted.</param>
        /// <returns><see langword="true"/> when accepted.</returns>
        private static bool TryIndex(object? value, out BigInteger index) {
            switch (value) {
                case NDArray { ndim: 0 } scalar when scalar.typecode == NPTypeCode.Int64:
                    index = Convert.ToInt64(scalar.GetAtIndex(0), CultureInfo.InvariantCulture);
                    return true;
                case bool or BigInteger or ulong or long or int or short or sbyte or byte or ushort or uint or Enum:
                    index = PythonIntSeed(value)!.Value;
                    return true;
                default:
                    index = default;
                    return false;
            }
        }

        /// <summary>A sample's or JSON pair's index as Python's <c>int(i)</c> reads it.</summary>
        /// <param name="value">The index: a 0-d integer array or a .NET integer.</param>
        /// <returns>The index.</returns>
        /// <exception cref="ArgumentException">Thrown for a value that is not an integer.</exception>
        private static long ToIndex(object? value) =>
            Discrete.TryToInt64(value, out long index) ? index : throw new ArgumentException($"Expected an integer subspace index, got {PyBuiltins.TypeRepr(value)}.", nameof(value));

        /// <summary>The <c>[0]</c> of a subspace's JSON form, with Python's errors.</summary>
        /// <param name="json">The JSON data of one sample.</param>
        /// <returns>Its first item.</returns>
        /// <exception cref="NumSharp.KeyError">Thrown for a dict (a <see cref="Dict"/> subspace's form): Python's <c>KeyError: 0</c>.</exception>
        /// <exception cref="NumSharp.IndexError">Thrown for an empty list.</exception>
        private static object? FirstJsonItem(object json) {
            switch (json) {
                case IReadOnlyDictionary<string, object?>:
                case IDictionary:
                    throw new KeyError("0");
                case IList { Count: 0 }:
                    throw new IndexError("list index out of range");
                case IList list:
                    return list[0];
                default:
                    return JsonList(json)[0];
            }
        }
    }
}
