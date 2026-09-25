#nullable enable
using System;
using System.Collections;
using System.Collections.Generic;
using Gym.Utils;
using NumSharp;

namespace Gym.Envs {
    /// <summary>
    ///     Gymnasium's <c>info</c> dict (and the base of <see cref="Metadata"/>): a <c>string → object</c> map that behaves like
    ///     a Python <c>dict</c>. Iteration follows insertion order, re-assigning a key keeps its place, and equality ignores
    ///     order and compares values as Python's <c>==</c> does (<see cref="PyEquality"/>). Keys keep Gymnasium's spelling
    ///     byte for byte (<c>"TimeLimit.truncated"</c>, <c>"episode"</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A missing key raises <see cref="KeyError"/> with Python's message (<c>'x'</c>), not
    ///     <see cref="KeyNotFoundException"/>, because ported tests catch Python's error. <see cref="Get"/> and
    ///     <see cref="Pop(string, object)"/> are the defaulting forms, as on a Python dict.
    ///     </para>
    ///     <para>
    ///     Mutable and not synchronized, like a Python dict. It hashes by its count only, so it stays usable as a key while
    ///     equal instances hash alike; Python refuses to hash a dict at all (<c>unhashable type: 'dict'</c>).
    ///     </para>
    /// </remarks>
    public class Info : IDictionary<string, object?>, IReadOnlyDictionary<string, object?>, IEquatable<Info> {
        /// <summary>The entries in insertion order; removal shifts the ones after it, which keeps the order.</summary>
        private readonly List<KeyValuePair<string, object?>> _entries;

        /// <summary>The position of every key in <see cref="_entries"/>, so a lookup is O(1).</summary>
        private readonly Dictionary<string, int> _positions;

        /// <summary>An empty info, Python's <c>{}</c>.</summary>
        public Info() {
            _entries = new List<KeyValuePair<string, object?>>();
            _positions = new Dictionary<string, int>(StringComparer.Ordinal);
        }

        /// <summary>
        ///     An info holding <paramref name="entries"/> in their order, Python's <c>dict(entries)</c>: a repeated key keeps
        ///     its first position and its last value.
        /// </summary>
        /// <param name="entries">The entries, for example another dictionary.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="entries"/> or one of its keys is <see langword="null"/>.</exception>
        public Info(IEnumerable<KeyValuePair<string, object?>> entries) : this() {
            if (entries is null)
                throw new ArgumentNullException(nameof(entries));
            foreach (KeyValuePair<string, object?> entry in entries)
                this[entry.Key] = entry.Value;
        }

        /// <summary>The number of keys.</summary>
        public int Count => _entries.Count;

        /// <summary>Always <see langword="false"/>: an info can be edited, like a Python dict.</summary>
        public bool IsReadOnly => false;

        /// <summary>The keys, in insertion order (a snapshot; editing the info afterwards does not change it).</summary>
        public ICollection<string> Keys {
            get {
                var keys = new List<string>(_entries.Count);
                foreach (KeyValuePair<string, object?> entry in _entries)
                    keys.Add(entry.Key);
                return keys;
            }
        }

        /// <summary>The values, in insertion order of their keys (a snapshot).</summary>
        public ICollection<object?> Values {
            get {
                var values = new List<object?>(_entries.Count);
                foreach (KeyValuePair<string, object?> entry in _entries)
                    values.Add(entry.Value);
                return values;
            }
        }

        /// <inheritdoc cref="Keys"/>
        IEnumerable<string> IReadOnlyDictionary<string, object?>.Keys => Keys;

        /// <inheritdoc cref="Values"/>
        IEnumerable<object?> IReadOnlyDictionary<string, object?>.Values => Values;

        /// <summary>
        ///     Python's <c>info[key]</c>: reading a missing key raises <see cref="KeyError"/>; assigning adds the key at the end,
        ///     or replaces the value of an existing key in place.
        /// </summary>
        /// <param name="key">The key.</param>
        /// <returns>The value.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="key"/> is <see langword="null"/>.</exception>
        /// <exception cref="KeyError">Thrown when reading a key the info does not hold; the message is the key's repr, <c>'x'</c>.</exception>
        public object? this[string key] {
            get {
                if (key is null)
                    throw new ArgumentNullException(nameof(key));
                if (!_positions.TryGetValue(key, out int position))
                    throw new KeyError(PyBuiltins.Repr(key));
                return _entries[position].Value;
            }
            set {
                if (key is null)
                    throw new ArgumentNullException(nameof(key));
                if (_positions.TryGetValue(key, out int position)) {
                    _entries[position] = new KeyValuePair<string, object?>(key, value);
                } else {
                    _positions.Add(key, _entries.Count);
                    _entries.Add(new KeyValuePair<string, object?>(key, value));
                }
            }
        }

        /// <summary>
        ///     Adds a new key, which is what a collection initializer (<c>new Info { { "x", 1 } }</c>) calls. Unlike assigning
        ///     through the indexer, an existing key is an error, as for every .NET dictionary.
        /// </summary>
        /// <param name="key">The key.</param>
        /// <param name="value">The value.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="key"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown when the info already holds <paramref name="key"/>.</exception>
        public void Add(string key, object? value) {
            if (key is null)
                throw new ArgumentNullException(nameof(key));
            if (_positions.ContainsKey(key))
                throw new ArgumentException($"An item with the same key has already been added. Key: {key}", nameof(key));
            this[key] = value;
        }

        /// <summary>Python's <c>key in info</c>.</summary>
        /// <param name="key">The key.</param>
        /// <returns><see langword="true"/> when the info holds <paramref name="key"/>.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="key"/> is <see langword="null"/>.</exception>
        public bool ContainsKey(string key) {
            if (key is null)
                throw new ArgumentNullException(nameof(key));
            return _positions.ContainsKey(key);
        }

        /// <summary>Reads a key without raising when it is missing.</summary>
        /// <param name="key">The key.</param>
        /// <param name="value">The value, or <see langword="null"/> when the key is missing.</param>
        /// <returns><see langword="true"/> when the info holds <paramref name="key"/>.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="key"/> is <see langword="null"/>.</exception>
        public bool TryGetValue(string key, out object? value) {
            if (key is null)
                throw new ArgumentNullException(nameof(key));
            if (_positions.TryGetValue(key, out int position)) {
                value = _entries[position].Value;
                return true;
            }

            value = null;
            return false;
        }

        /// <summary>Python's <c>info.get(key, default)</c>: the value, or <paramref name="default"/> when the key is missing.</summary>
        /// <param name="key">The key.</param>
        /// <param name="default">What to return for a missing key (Python's default is <c>None</c>).</param>
        /// <returns>The value or the default.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="key"/> is <see langword="null"/>.</exception>
        public object? Get(string key, object? @default = null) => TryGetValue(key, out object? value) ? value : @default;

        /// <summary>
        ///     Python's <c>del info[key]</c> without the error: removes the key and closes the gap, so the other keys keep
        ///     their relative order.
        /// </summary>
        /// <param name="key">The key.</param>
        /// <returns><see langword="true"/> when the key was present.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="key"/> is <see langword="null"/>.</exception>
        public bool Remove(string key) {
            if (key is null)
                throw new ArgumentNullException(nameof(key));
            if (!_positions.TryGetValue(key, out int position))
                return false;
            _entries.RemoveAt(position);
            _positions.Remove(key);
            // Every later entry moved one place to the left.
            for (int i = position; i < _entries.Count; i++)
                _positions[_entries[i].Key] = i;
            return true;
        }

        /// <summary>Python's <c>info.pop(key)</c>: removes the key and returns its value.</summary>
        /// <param name="key">The key.</param>
        /// <returns>The removed value.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="key"/> is <see langword="null"/>.</exception>
        /// <exception cref="KeyError">Thrown when the key is missing; the message is the key's repr.</exception>
        public object? Pop(string key) {
            if (!TryGetValue(key, out object? value))
                throw new KeyError(PyBuiltins.Repr(key));
            Remove(key);
            return value;
        }

        /// <summary>
        ///     Python's <c>info.pop(key, default)</c>: removes the key and returns its value, or returns
        ///     <paramref name="default"/> when it is missing. <c>step_api_compatibility</c> reads <c>"TimeLimit.truncated"</c>
        ///     this way.
        /// </summary>
        /// <param name="key">The key.</param>
        /// <param name="default">What to return for a missing key.</param>
        /// <returns>The removed value, or the default.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="key"/> is <see langword="null"/>.</exception>
        public object? Pop(string key, object? @default) {
            if (!TryGetValue(key, out object? value))
                return @default;
            Remove(key);
            return value;
        }

        /// <summary>
        ///     Python's <c>info.update(other)</c>: assigns every entry of <paramref name="other"/> in its order, so new keys go
        ///     to the end and existing ones keep their place.
        /// </summary>
        /// <param name="other">The entries to merge in.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> or one of its keys is <see langword="null"/>.</exception>
        public void Update(IEnumerable<KeyValuePair<string, object?>> other) {
            if (other is null)
                throw new ArgumentNullException(nameof(other));
            // A snapshot first, so updating an info with itself does not enumerate a list it is editing.
            foreach (KeyValuePair<string, object?> entry in new List<KeyValuePair<string, object?>>(other))
                this[entry.Key] = entry.Value;
        }

        /// <summary>
        ///     Python's <c>info.copy()</c>: a new info of the same type with the same entries in the same order; the values
        ///     themselves are shared, as a shallow copy shares them.
        /// </summary>
        /// <returns>The copy.</returns>
        public Info Copy() {
            Info copy = CreateEmpty();
            foreach (KeyValuePair<string, object?> entry in _entries)
                copy[entry.Key] = entry.Value;
            return copy;
        }

        /// <summary>
        ///     <c>copy.deepcopy(info)</c>: a new info of the same type whose values are deep copies (<see cref="PyCopy"/>), so
        ///     editing the copy, even inside a nested dict or array, never shows through the original.
        /// </summary>
        /// <returns>The copy.</returns>
        public Info DeepCopy() {
            Info copy = CreateEmpty();
            foreach (KeyValuePair<string, object?> entry in _entries)
                copy[entry.Key] = PyCopy.DeepCopy(entry.Value);
            return copy;
        }

        /// <summary>Python's <c>info.clear()</c>: removes every key.</summary>
        public void Clear() {
            _entries.Clear();
            _positions.Clear();
        }

        /// <summary>Enumerates the entries in insertion order.</summary>
        /// <returns>The enumerator; editing the info while enumerating throws, as a Python dict raises <c>RuntimeError</c>.</returns>
        public IEnumerator<KeyValuePair<string, object?>> GetEnumerator() => _entries.GetEnumerator();

        /// <inheritdoc cref="GetEnumerator"/>
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        /// <summary>Adds an entry (see <see cref="Add(string, object)"/>).</summary>
        /// <param name="item">The entry.</param>
        /// <exception cref="ArgumentException">Thrown when the key is already present.</exception>
        void ICollection<KeyValuePair<string, object?>>.Add(KeyValuePair<string, object?> item) => Add(item.Key, item.Value);

        /// <summary>Whether the info holds the key with a value equal to the entry's, under Python's equality.</summary>
        /// <param name="item">The entry.</param>
        /// <returns><see langword="true"/> when the entry is present.</returns>
        bool ICollection<KeyValuePair<string, object?>>.Contains(KeyValuePair<string, object?> item) {
            return item.Key != null && TryGetValue(item.Key, out object? value) && PyEquality.AreEqual(value, item.Value);
        }

        /// <summary>Copies the entries, in order, into <paramref name="array"/> from <paramref name="arrayIndex"/>.</summary>
        /// <param name="array">The destination.</param>
        /// <param name="arrayIndex">The first index to write.</param>
        void ICollection<KeyValuePair<string, object?>>.CopyTo(KeyValuePair<string, object?>[] array, int arrayIndex) => _entries.CopyTo(array, arrayIndex);

        /// <summary>Removes the entry when the key holds a value equal to the entry's.</summary>
        /// <param name="item">The entry.</param>
        /// <returns><see langword="true"/> when it was removed.</returns>
        bool ICollection<KeyValuePair<string, object?>>.Remove(KeyValuePair<string, object?> item) {
            return ((ICollection<KeyValuePair<string, object?>>)this).Contains(item) && Remove(item.Key);
        }

        /// <summary>
        ///     Python's <c>==</c> between dicts: the same keys, each with an equal value (<see cref="PyEquality.AreEqual"/>),
        ///     regardless of insertion order and of the concrete type (an <see cref="Info"/> equals a <see cref="Metadata"/>
        ///     with the same entries, as two dicts compare).
        /// </summary>
        /// <param name="other">The other info; may be <see langword="null"/>.</param>
        /// <returns><see langword="true"/> when the two are equal.</returns>
        public bool Equals(Info? other) => other is not null && PyEquality.AreEqual(this, other);

        /// <summary>Python's <c>==</c> with any dict-like value (an <see cref="Info"/>, or a <see cref="Dictionary{TKey, TValue}"/>).</summary>
        /// <param name="obj">The other value.</param>
        /// <returns><see langword="true"/> when the two are equal dicts.</returns>
        public override bool Equals(object? obj) => obj is not null && PyEquality.AreEqual(this, obj);

        /// <summary>The number of keys: equal infos always have equal counts, so this is consistent with <see cref="Equals(Info)"/>.</summary>
        /// <returns>The hash.</returns>
        public override int GetHashCode() => _entries.Count;

        /// <summary>Python's <c>repr(info)</c>: <c>{'render_modes': ['human', 'rgb_array'], 'render_fps': 50}</c>.</summary>
        /// <returns>The text.</returns>
        public override string ToString() => PyBuiltins.Repr(this);

        /// <summary>Python's <c>==</c> between two infos (see <see cref="Equals(Info)"/>); two <see langword="null"/>s are equal.</summary>
        /// <param name="left">The first info.</param>
        /// <param name="right">The second info.</param>
        /// <returns><see langword="true"/> when they are equal.</returns>
        public static bool operator ==(Info? left, Info? right) => left is null ? right is null : left.Equals(right);

        /// <summary>Python's <c>!=</c> between two infos.</summary>
        /// <param name="left">The first info.</param>
        /// <param name="right">The second info.</param>
        /// <returns><see langword="true"/> when they differ.</returns>
        public static bool operator !=(Info? left, Info? right) => !(left == right);

        /// <summary>
        ///     Creates an empty instance of this info's own type for <see cref="Copy"/> and <see cref="DeepCopy"/>, so copying a
        ///     <see cref="Metadata"/> yields a <see cref="Metadata"/>. Subclasses with other constructors override it.
        /// </summary>
        /// <returns>An empty info of the same type.</returns>
        protected virtual Info CreateEmpty() => new Info();
    }
}
