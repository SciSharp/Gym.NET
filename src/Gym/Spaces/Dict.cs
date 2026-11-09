#nullable enable
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using Gym.Envs;
using Gym.Utils;
using NumSharp;

namespace Gym.Spaces {
    /// <summary>
    ///     A dictionary of spaces: every element is a string-keyed dict (Gym.NET's insertion-ordered Python dict,
    ///     <see cref="Info"/>, DR-13's generic form) holding one element of each subspace under its key, such as a robot's
    ///     <c>{"position": …, "velocity": …}</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Keys are sorted when the space is built from an unordered mapping and <c>sort_keys</c> is on (Gymnasium's legacy
    ///     default); an ordered mapping keeps its order. The order decides how samples flatten, so it is part of the space.
    ///     </para>
    ///     <para>
    ///     The space owns its subspaces: disposing it disposes them. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/spaces/dict.py#L17">gymnasium/spaces/dict.py</see>.
    ///     </para>
    /// </remarks>
    public class Dict : Space<Info>, IReadOnlyDictionary<string, Space>, IEquatable<Dict> {
        /// <summary>The subspaces, in the space's key order.</summary>
        private readonly Dictionary<string, Space> _spaces = new();

        /// <summary>The keys in the space's order (a <see cref="Dictionary{TKey, TValue}"/> does not promise its enumeration order).</summary>
        private readonly List<string> _keys = new();

        /// <summary>
        ///     A dict space from key/space pairs (Gymnasium's <c>Dict(spaces, seed=seed)</c>): an ordered sequence of pairs
        ///     keeps its order, as Gymnasium's sequence and <c>OrderedDict</c> inputs do.
        /// </summary>
        /// <param name="spaces">The pairs, in order; the space owns the subspaces from now on.</param>
        /// <param name="seed">An integer seed (the dict's generator then draws one <c>int32</c> seed per subspace), a generator to share, or <see langword="null"/>.</param>
        /// <param name="sortKeys">
        ///     Recorded as <see cref="SortKeys"/> but not applied: Gymnasium sorts only an unordered mapping. Batching
        ///     (<c>batch_space</c>) builds a mapping and applies it then, as Gymnasium does.
        /// </param>
        /// <exception cref="NumSharp.TypeError">Thrown for a <see langword="null"/> subspace, with Gymnasium's message.</exception>
        /// <exception cref="ArgumentException">Thrown for a repeated key.</exception>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="spaces"/> is <see langword="null"/>.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public Dict(IEnumerable<KeyValuePair<string, Space>> spaces, SpaceSeed? seed = null, bool sortKeys = true) : this(spaces, sortKeys, applySort: false, seed) { }

        /// <summary>
        ///     A dict space from a mapping (Gymnasium's <c>Dict(mapping, seed=seed, sort_keys=sort_keys)</c>): with
        ///     <paramref name="sortKeys"/> (Gymnasium's default) the keys are sorted ordinally, otherwise they keep the mapping's
        ///     enumeration order.
        /// </summary>
        /// <param name="spaces">The mapping; the space owns the subspaces from now on.</param>
        /// <param name="seed">An integer seed, a generator to share, or <see langword="null"/>.</param>
        /// <param name="sortKeys">Whether to sort the keys, Gymnasium's legacy behavior for unordered dicts.</param>
        /// <exception cref="NumSharp.TypeError">Thrown for a <see langword="null"/> subspace.</exception>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="spaces"/> is <see langword="null"/>.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public Dict(IReadOnlyDictionary<string, Space> spaces, SpaceSeed? seed = null, bool sortKeys = true)
            : this((IEnumerable<KeyValuePair<string, Space>>)(spaces ?? throw new ArgumentNullException(nameof(spaces))), sortKeys, applySort: sortKeys, seed) { }

        /// <summary>The shared constructor.</summary>
        /// <param name="spaces">The pairs.</param>
        /// <param name="sortKeys">Gymnasium's <c>sort_keys</c>, recorded.</param>
        /// <param name="applySort">Whether to sort the keys now (an unordered mapping with <paramref name="sortKeys"/>).</param>
        /// <param name="seed">The seed.</param>
        /// <exception cref="NumSharp.TypeError">Thrown for a <see langword="null"/> subspace.</exception>
        /// <exception cref="ArgumentException">Thrown for a repeated key.</exception>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="spaces"/> is <see langword="null"/>.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        private Dict(IEnumerable<KeyValuePair<string, Space>> spaces, bool sortKeys, bool applySort, SpaceSeed? seed) : base(null, null, null) {
            SortKeys = sortKeys;
            IEnumerable<KeyValuePair<string, Space>> pairs = spaces ?? throw new ArgumentNullException(nameof(spaces));
            if (applySort)
                pairs = pairs.OrderBy(p => p.Key, StringComparer.Ordinal);
            foreach (KeyValuePair<string, Space> pair in pairs) {
                if (pair.Value is null)
                    throw new TypeError($"Dict space element is not an instance of Space: key='{pair.Key}', space=None");
                _spaces.Add(pair.Key, pair.Value);
                _keys.Add(pair.Key);
            }

            if (seed is { } given) {
                if (given.Generator != null)
                    NpRandom = given.Generator;
                else
                    Seed(given.Value);
            }
        }

        /// <summary>Gymnasium's <c>sort_keys</c> as given (applied only to an unordered mapping); batching passes it on.</summary>
        public bool SortKeys { get; }

        /// <summary>The subspaces by key, in the space's order (Gymnasium's <c>spaces</c>).</summary>
        public IReadOnlyDictionary<string, Space> Spaces => this;

        /// <summary>The keys, in the space's order (Gymnasium's <c>keys()</c>).</summary>
        public IEnumerable<string> Keys => _keys;

        /// <summary>The subspaces, in the space's order.</summary>
        public IEnumerable<Space> Values => _keys.Select(k => _spaces[k]);

        /// <summary>The number of subspaces (Gymnasium's <c>len(space)</c>).</summary>
        public int Count => _keys.Count;

        /// <summary>
        ///     The subspace under <paramref name="key"/> (Gymnasium's <c>space[key]</c>); assigning one adds or replaces it
        ///     (Gymnasium's <c>__setitem__</c>), a new key going last.
        /// </summary>
        /// <param name="key">The key.</param>
        /// <returns>The subspace.</returns>
        /// <exception cref="NumSharp.KeyError">Thrown by the getter for an unknown key, with Python's message.</exception>
        /// <exception cref="NumSharp.TypeError">Thrown by the setter for a <see langword="null"/> space.</exception>
        public Space this[string key] {
            get => _spaces.TryGetValue(key, out Space? space) ? space : throw new KeyError(PyBuiltins.Repr(key));
            set {
                if (value is null)
                    throw new TypeError($"Trying to set {key} to Dict space with value that is not a gymnasium space, actual type: <class 'NoneType'>");
                if (!_spaces.ContainsKey(key))
                    _keys.Add(key);
                _spaces[key] = value;
            }
        }

        /// <summary>Whether every subspace flattens to a box.</summary>
        public override bool IsNpFlattenable => Values.All(s => s.IsNpFlattenable);

        /// <summary>
        ///     Draws one element (Gymnasium's <c>sample</c>): a dict with one sample per subspace in the space's key order,
        ///     optionally under a dict of per-key masks or probability masks.
        /// </summary>
        /// <param name="mask"><see langword="null"/>, or a dict with the space's keys mapping to each subspace's mask.</param>
        /// <param name="probability"><see langword="null"/>, or a dict with the space's keys mapping to each subspace's probability mask.</param>
        /// <returns>A new dict; arrays in it are owned by the caller.</returns>
        /// <exception cref="NumSharp.ValueError">Thrown when both masks are given.</exception>
        /// <exception cref="NumSharp.AssertionError">Thrown for a mask that is not a dict with the space's keys, with Gymnasium's messages.</exception>
        public override Info Sample(object? mask = null, object? probability = null) {
            if (mask is not null && probability is not null)
                throw new ValueError($"Only one of `mask` or `probability` can be provided, actual values: mask={PyBuiltins.Str(mask)}, probability={PyBuiltins.Str(probability)}");
            var sample = new Info();
            if (mask is not null) {
                IReadOnlyDictionary<string, object?> masks = DictArgument(mask, "mask");
                foreach (string key in _keys)
                    sample[key] = _spaces[key].Sample(mask: masks[key]);
            } else if (probability is not null) {
                IReadOnlyDictionary<string, object?> probabilities = DictArgument(probability, "probability");
                foreach (string key in _keys)
                    sample[key] = _spaces[key].Sample(probability: probabilities[key]);
            } else {
                foreach (string key in _keys)
                    sample[key] = _spaces[key].Sample();
            }

            return sample;
        }

        /// <summary>
        ///     Whether <paramref name="x"/> is an element (Gymnasium's <c>contains</c>): a dict with exactly the space's keys (in
        ///     any order) whose every value is a member of its subspace.
        /// </summary>
        /// <param name="x">The candidate: an <see cref="Info"/> or any string-keyed dictionary.</param>
        /// <returns><see langword="true"/> for an element.</returns>
        public override bool Contains(object? x) {
            IReadOnlyDictionary<string, object?>? candidate = AsDict(x);
            if (candidate is null || candidate.Count != _keys.Count || !_keys.All(candidate.ContainsKey))
                return false;
            return _keys.All(key => _spaces[key].Contains(candidate[key]));
        }

        /// <summary>Converts samples to Gymnasium's JSON form: a dict with each key's <c>to_jsonable</c> across the batch.</summary>
        /// <param name="samples">Samples of this space.</param>
        /// <returns>The dict of per-key JSON data.</returns>
        public override object ToJsonable(IEnumerable<Info> samples) {
            Info[] batch = samples.ToArray();
            var json = new Dictionary<string, object?>();
            foreach (string key in _keys)
                json[key] = _spaces[key].ToJsonable(batch.Select(sample => sample[key]));
            return json;
        }

        /// <summary>Converts Gymnasium's dict of per-key JSON data back to dicts (Gymnasium's <c>from_jsonable</c>).</summary>
        /// <param name="jsonable">The dict of per-key data.</param>
        /// <returns>The samples.</returns>
        /// <exception cref="ArgumentException">Thrown when the data is not a string-keyed dictionary.</exception>
        public override IList<Info> FromJsonable(object jsonable) {
            IReadOnlyDictionary<string, object?> perKey = AsDict(jsonable) ?? throw new ArgumentException("Expected a dict of per-key JSON data.", nameof(jsonable));
            Dictionary<string, IList<object?>> columns = _keys.ToDictionary(key => key, key => _spaces[key].FromJsonable(perKey[key]!));
            int count = columns.Count == 0 ? 0 : columns.Values.First().Count;
            var samples = new List<Info>(count);
            for (int n = 0; n < count; n++) {
                var sample = new Info();
                foreach (string key in _keys)
                    sample[key] = columns[key][n];
                samples.Add(sample);
            }

            return samples;
        }

        /// <summary>Gymnasium's <c>repr</c>: <c>Dict('position': Box(…), 'velocity': Box(…))</c>, keys as Python string reprs.</summary>
        /// <returns>The text.</returns>
        public override string ToString() => "Dict(" + string.Join(", ", _keys.Select(k => $"{PyBuiltins.Repr(k)}: {_spaces[k]}")) + ")";

        /// <summary>Gymnasium's <c>==</c>: the same keys in the same order with equal subspaces (an ordered comparison).</summary>
        /// <param name="other">The other space.</param>
        /// <returns><see langword="true"/> when equal.</returns>
        public bool Equals(Dict? other) => other is not null && _keys.SequenceEqual(other._keys) && _keys.All(k => _spaces[k].Equals(other._spaces[k]));

        /// <inheritdoc />
        public override bool Equals(object? obj) => obj is Dict other && Equals(other);

        /// <inheritdoc />
        public override int GetHashCode() => HashCode.Combine(_keys.Count, _keys.Count == 0 ? 0 : StringComparer.Ordinal.GetHashCode(_keys[0]));

        /// <inheritdoc />
        public bool ContainsKey(string key) => _spaces.ContainsKey(key);

        /// <inheritdoc />
        public bool TryGetValue(string key, out Space value) => _spaces.TryGetValue(key, out value!);

        /// <inheritdoc />
        public IEnumerator<KeyValuePair<string, Space>> GetEnumerator() => _keys.Select(k => new KeyValuePair<string, Space>(k, _spaces[k])).GetEnumerator();

        /// <inheritdoc />
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        /// <summary>
        ///     Gymnasium's <c>Dict.seed</c> for an integer or <see langword="null"/>: <see langword="null"/> seeds every
        ///     subspace from entropy; an integer seeds the dict's generator, which draws one <c>int32</c> seed per subspace.
        /// </summary>
        /// <param name="seed">The seed, or <see langword="null"/>.</param>
        /// <returns>The subspaces' seeds, by key.</returns>
        /// <exception cref="Gym.Exceptions.Error">Thrown for a negative seed.</exception>
        protected override object SeedCore(BigInteger? seed) {
            var seeds = new Dictionary<string, object>();
            if (seed is null) {
                foreach (string key in _keys)
                    seeds[key] = _spaces[key].Seed((BigInteger?)null);
                return seeds;
            }

            base.SeedCore(seed);
            using NDArray subseeds = NpRandom.integers(int.MaxValue, size: new Shape((long)_keys.Count));
            for (int i = 0; i < _keys.Count; i++)
                seeds[_keys[i]] = _spaces[_keys[i]].Seed(new BigInteger(Convert.ToInt64(subseeds.GetAtIndex(i), CultureInfo.InvariantCulture)));
            return seeds;
        }

        /// <summary>Gymnasium's <c>Dict.seed</c> for a dict: one seed (of any form) per key; the keys must be the space's.</summary>
        /// <param name="seeds">The seeds.</param>
        /// <returns>The subspaces' seeds, by key.</returns>
        /// <exception cref="NumSharp.ValueError">Thrown when the keys are not the space's.</exception>
        protected override object SeedDictCore(IReadOnlyDictionary<string, object?> seeds) {
            if (seeds.Count != _keys.Count || !_keys.All(seeds.ContainsKey))
                throw new ValueError($"The seed keys: dict_keys({PyBuiltins.Repr(seeds.Keys.ToList())}) are not identical to space keys: dict_keys({PyBuiltins.Repr(_keys)})");
            var used = new Dictionary<string, object>();
            foreach (string key in seeds.Keys)
                used[key] = _spaces[key].SeedWith(seeds[key]);
            return used;
        }

        /// <summary>Gymnasium's <c>Dict.seed</c> error for a seed that is not an integer, a dict or <see langword="null"/> (a list included).</summary>
        /// <param name="seed">The refused seed.</param>
        /// <returns><c>TypeError: Expected seed type: dict, int or None, actual type: &lt;class 'list'&gt;</c>.</returns>
        protected override Exception SeedTypeError(object? seed) =>
            new TypeError($"Expected seed type: dict, int or None, actual type: {PyBuiltins.TypeRepr(seed)}");

        /// <inheritdoc />
        protected override void Dispose(bool disposing) {
            if (disposing) {
                foreach (Space space in _spaces.Values)
                    space.Dispose();
            }

            base.Dispose(disposing);
        }

        /// <summary>A value as a string-keyed dictionary, or <see langword="null"/> when it is none.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The dictionary view.</returns>
        internal static IReadOnlyDictionary<string, object?>? AsDict(object? value) => value switch {
            IReadOnlyDictionary<string, object?> dictionary => dictionary,
            IDictionary dictionary when dictionary.Keys.Cast<object>().All(k => k is string) =>
                dictionary.Keys.Cast<string>().ToDictionary(k => k, k => dictionary[k]),
            _ => null,
        };

        /// <summary>A dict argument (a mask or probability) with Gymnasium's assertions.</summary>
        /// <param name="argument">The argument.</param>
        /// <param name="name"><c>mask</c> or <c>probability</c>.</param>
        /// <returns>The dictionary.</returns>
        /// <exception cref="NumSharp.AssertionError">Thrown when it is not a dict with the space's keys.</exception>
        private IReadOnlyDictionary<string, object?> DictArgument(object argument, string name) {
            string what = name == "mask" ? "mask" : "probability mask";
            IReadOnlyDictionary<string, object?> dictionary = AsDict(argument)
                ?? throw new AssertionError($"Expected sample {what} to be a dict, actual type: {PyBuiltins.TypeRepr(argument)}");
            if (dictionary.Count != _keys.Count || !_keys.All(dictionary.ContainsKey))
                throw new AssertionError($"Expected sample {what} keys to be same as space keys, mask keys: dict_keys({PyBuiltins.Repr(dictionary.Keys.ToList())}), space keys: dict_keys({PyBuiltins.Repr(_keys)})");
            return dictionary;
        }
    }
}
