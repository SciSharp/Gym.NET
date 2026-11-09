#nullable enable
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using Gym.Utils;
using NumSharp;

namespace Gym.Spaces {
    /// <summary>
    ///     The cartesian product of spaces: every element is a tuple (an <see langword="object"/>?[], DR-13's generic form)
    ///     holding one element of each subspace, in order, such as Blackjack's <c>(player sum, dealer card, usable ace)</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Named as Gymnasium names it, so <c>using System;</c> makes <c>Tuple</c> ambiguous with <see cref="System.Tuple"/>
    ///     (CS0104): Gym.NET's files alias it, <c>using Tuple = Gym.Spaces.Tuple;</c> (port plan §3.8).
    ///     </para>
    ///     <para>
    ///     The space owns its subspaces: disposing it disposes them. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/spaces/tuple.py#L16">gymnasium/spaces/tuple.py</see>.
    ///     </para>
    /// </remarks>
    public class Tuple : Space<object?[]>, IReadOnlyList<Space>, IEquatable<Tuple> {
        /// <summary>The subspaces.</summary>
        private readonly Space[] _spaces;

        /// <summary>The product of <paramref name="spaces"/> (Gymnasium's <c>Tuple(spaces, seed=seed)</c>).</summary>
        /// <param name="spaces">The subspaces, in order; the tuple owns them from now on.</param>
        /// <param name="seed">
        ///     An integer seed (it seeds the tuple's own generator, which then draws one <c>int32</c> seed per subspace), a
        ///     generator to share, or <see langword="null"/>.
        /// </param>
        /// <exception cref="NumSharp.TypeError">Thrown for a <see langword="null"/> subspace, Gymnasium's "does not inherit from `gymnasium.Space`".</exception>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="spaces"/> is <see langword="null"/>.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public Tuple(IEnumerable<Space> spaces, SpaceSeed? seed = null) : base(null, null, null) {
            _spaces = (spaces ?? throw new ArgumentNullException(nameof(spaces))).ToArray();
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

        /// <summary>The product of the given subspaces.</summary>
        /// <param name="spaces">The subspaces, in order.</param>
        /// <exception cref="NumSharp.TypeError">Thrown for a <see langword="null"/> subspace.</exception>
        public Tuple(params Space[] spaces) : this((IEnumerable<Space>)spaces) { }

        /// <summary>The subspaces (Gymnasium's <c>spaces</c>).</summary>
        public IReadOnlyList<Space> Spaces => _spaces;

        /// <summary>The number of subspaces (Gymnasium's <c>len(space)</c>).</summary>
        public int Count => _spaces.Length;

        /// <summary>The subspace at <paramref name="index"/> (Gymnasium's <c>space[index]</c>).</summary>
        /// <param name="index">The index; negative counts from the end, as in Python.</param>
        /// <returns>The subspace.</returns>
        /// <exception cref="NumSharp.IndexError">Thrown for an index outside the tuple: <c>tuple index out of range</c>.</exception>
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
        ///     Draws one element (Gymnasium's <c>sample</c>): one sample from each subspace, in order, optionally under a tuple
        ///     of per-subspace masks or probability masks.
        /// </summary>
        /// <param name="mask"><see langword="null"/>, or a tuple (an <see langword="object"/>?[] or an <see cref="ITuple"/>) with one mask (or <see langword="null"/>) per subspace.</param>
        /// <param name="probability"><see langword="null"/>, or a tuple with one probability mask (or <see langword="null"/>) per subspace.</param>
        /// <returns>A new tuple of the subspaces' samples; arrays in it are owned by the caller.</returns>
        /// <exception cref="NumSharp.ValueError">Thrown when both masks are given.</exception>
        /// <exception cref="NumSharp.AssertionError">Thrown for a mask that is not a tuple of the right length, with Gymnasium's messages.</exception>
        public override object?[] Sample(object? mask = null, object? probability = null) {
            if (mask is not null && probability is not null)
                throw new ValueError($"Only one of `mask` or `probability` can be provided, actual values: mask={PyBuiltins.Str(mask)}, probability={PyBuiltins.Str(probability)}");
            if (mask is not null) {
                object?[] masks = TupleArgument(mask, "mask");
                return _spaces.Select((space, i) => space.Sample(mask: masks[i])).ToArray();
            }

            if (probability is not null) {
                object?[] probabilities = TupleArgument(probability, "probability");
                return _spaces.Select((space, i) => space.Sample(probability: probabilities[i])).ToArray();
            }

            return _spaces.Select(space => space.Sample()).ToArray();
        }

        /// <summary>
        ///     Whether <paramref name="x"/> is an element (Gymnasium's <c>contains</c>): a tuple (or a list, a .NET array, or
        ///     an array along its first axis, which Gymnasium promotes) with one member of each subspace, in order.
        /// </summary>
        /// <param name="x">The candidate.</param>
        /// <returns><see langword="true"/> for an element.</returns>
        public override bool Contains(object? x) {
            object?[]? items = x switch {
                object?[] array => array,
                ITuple tuple => TupleItems(tuple),
                NDArray nd when nd.ndim > 0 => Enumerable.Range(0, (int)nd.shape[0]).Select(i => (object?)nd[i.ToString(System.Globalization.CultureInfo.InvariantCulture)]).ToArray(),
                IList list => list.Cast<object?>().ToArray(),
                _ => null,
            };
            return items is not null && items.Length == _spaces.Length && _spaces.Select((space, i) => space.Contains(items[i])).All(contained => contained);
        }

        /// <summary>
        ///     Converts samples to Gymnasium's JSON form, a list with one entry per subspace, each the subspace's
        ///     <c>to_jsonable</c> of that position across the batch.
        /// </summary>
        /// <param name="samples">Samples of this space.</param>
        /// <returns>The per-subspace lists.</returns>
        public override object ToJsonable(IEnumerable<object?[]> samples) {
            object?[][] batch = samples.ToArray();
            return _spaces.Select((space, i) => space.ToJsonable(batch.Select(sample => sample[i]))).Cast<object?>().ToList();
        }

        /// <summary>Converts Gymnasium's per-subspace JSON lists back to tuples (Gymnasium's <c>from_jsonable</c>).</summary>
        /// <param name="jsonable">The list of per-subspace lists.</param>
        /// <returns>The samples.</returns>
        /// <exception cref="NumSharp.ValueError">Thrown when the per-subspace lists differ in length (Python's <c>zip(strict=True)</c>).</exception>
        public override IList<object?[]> FromJsonable(object jsonable) {
            IList<object?> perSpace = JsonList(jsonable);
            IList<object?>[] columns = _spaces.Select((space, i) => space.FromJsonable(perSpace[i]!)).ToArray();
            int count = columns.Length == 0 ? 0 : columns[0].Count;
            if (columns.Any(c => c.Count != count))
                throw new ValueError("zip() argument 2 is shorter than argument 1");
            return Enumerable.Range(0, count).Select(n => columns.Select(c => c[n]).ToArray()).ToList();
        }

        /// <summary>Gymnasium's <c>repr</c>: <c>Tuple(Discrete(2), Box(-1.0, 1.0, (2,), float32))</c>.</summary>
        /// <returns>The text.</returns>
        public override string ToString() => "Tuple(" + string.Join(", ", _spaces.Select(s => s.ToString())) + ")";

        /// <summary>Gymnasium's <c>==</c>: the same subspaces, equal in order.</summary>
        /// <param name="other">The other tuple.</param>
        /// <returns><see langword="true"/> when equal.</returns>
        public bool Equals(Tuple? other) => other is not null && _spaces.Length == other._spaces.Length && _spaces.Zip(other._spaces).All(p => p.First.Equals(p.Second));

        /// <inheritdoc />
        public override bool Equals(object? obj) => obj is Tuple other && Equals(other);

        /// <inheritdoc />
        public override int GetHashCode() => HashCode.Combine(_spaces.Length, _spaces.Length == 0 ? 0 : _spaces[0].GetHashCode());

        /// <inheritdoc />
        public IEnumerator<Space> GetEnumerator() => ((IEnumerable<Space>)_spaces).GetEnumerator();

        /// <inheritdoc />
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        /// <summary>
        ///     Gymnasium's <c>Tuple.seed</c> for an integer or <see langword="null"/>: <see langword="null"/> seeds every
        ///     subspace from entropy; an integer seeds the tuple's generator, which draws one seed below <c>int32</c>'s maximum
        ///     per subspace (<c>integers(iinfo(int32).max, size=len)</c>).
        /// </summary>
        /// <param name="seed">The seed, or <see langword="null"/>.</param>
        /// <returns>The subspaces' seeds, as a list.</returns>
        /// <exception cref="Gym.Exceptions.Error">Thrown for a negative seed.</exception>
        protected override object SeedCore(BigInteger? seed) {
            if (seed is null)
                return _spaces.Select(space => space.Seed((BigInteger?)null)).ToList();
            base.SeedCore(seed);
            using NDArray subseeds = NpRandom.integers(int.MaxValue, size: new Shape((long)_spaces.Length));
            return _spaces.Select((space, i) => space.Seed(new BigInteger(Convert.ToInt64(subseeds.GetAtIndex(i), System.Globalization.CultureInfo.InvariantCulture)))).ToList();
        }

        /// <summary>Gymnasium's <c>Tuple.seed</c> for a list: one seed (of any form) per subspace.</summary>
        /// <param name="seeds">The seeds.</param>
        /// <returns>The subspaces' seeds.</returns>
        /// <exception cref="NumSharp.ValueError">Thrown when the list's length is not the number of subspaces.</exception>
        protected override object SeedListCore(IReadOnlyList<object?> seeds) {
            if (seeds.Count != _spaces.Length)
                throw new ValueError($"Expects that the subspaces of seeds equals the number of subspaces. Actual length of seeds: {seeds.Count}, length of subspaces: {_spaces.Length}");
            return _spaces.Select((space, i) => space.SeedWith(seeds[i])).ToList();
        }

        /// <summary>Gymnasium's <c>Tuple.seed</c> error for a seed that is not an integer, a list, a tuple or <see langword="null"/>.</summary>
        /// <param name="seed">The refused seed.</param>
        /// <returns><c>TypeError: Expected seed type: list, tuple, int or None, actual type: &lt;class 'dict'&gt;</c>.</returns>
        protected override Exception SeedTypeError(object? seed) =>
            new TypeError($"Expected seed type: list, tuple, int or None, actual type: {PyBuiltins.TypeRepr(seed)}");

        /// <inheritdoc />
        protected override void Dispose(bool disposing) {
            if (disposing) {
                foreach (Space space in _spaces)
                    space.Dispose();
            }

            base.Dispose(disposing);
        }

        /// <summary>A tuple argument's items (an <see langword="object"/>?[] or an <see cref="ITuple"/>), with Gymnasium's assertions.</summary>
        /// <param name="argument">The mask or probability.</param>
        /// <param name="name"><c>mask</c> or <c>probability</c>, for the message.</param>
        /// <returns>The items.</returns>
        /// <exception cref="NumSharp.AssertionError">Thrown when it is not a tuple of one item per subspace.</exception>
        private object?[] TupleArgument(object argument, string name) {
            object?[] items = argument switch {
                object?[] array => array,
                ITuple tuple => TupleItems(tuple),
                _ => throw new AssertionError($"Expected type of `{name}` to be tuple, actual type: {PyBuiltins.TypeRepr(argument)}"),
            };
            if (items.Length != _spaces.Length)
                throw new AssertionError($"Expected length of `{name}` to be {_spaces.Length}, actual length: {items.Length}");
            return items;
        }

        /// <summary>A tuple's items.</summary>
        /// <param name="tuple">The tuple.</param>
        /// <returns>The items in order.</returns>
        internal static object?[] TupleItems(ITuple tuple) {
            var items = new object?[tuple.Length];
            for (int i = 0; i < items.Length; i++)
                items[i] = tuple[i];
            return items;
        }
    }
}
