#nullable enable
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Gym.Utils;
using NumSharp;

namespace Gym.Envs.Registration {
    /// <summary>
    ///     The registry's dict, env id to <see cref="EnvSpec"/> (Python's <c>gymnasium.registry</c>, a plain <c>dict</c>):
    ///     iteration follows registration order, which <c>pprint_registry</c> and the "Did you mean" suggestions read, and
    ///     re-registering an id replaces its spec in place. Tests and plugins edit it directly (<c>del registry[id]</c> is
    ///     <see cref="Remove(string)"/>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A missing id raises <see cref="KeyError"/> with Python's message (<c>'x'</c>), not
    ///     <see cref="KeyNotFoundException"/>; <see cref="Get"/> and <see cref="Pop(string, EnvSpec)"/> are the defaulting
    ///     forms, as on a Python dict.
    ///     </para>
    ///     <para>
    ///     Unlike a Python dict it is synchronized: env assemblies register from module initializers, which may run on any
    ///     thread while another thread makes envs. Enumerating walks a snapshot, so editing the registry during a loop over it
    ///     is safe (Python raises <c>RuntimeError: dictionary changed size during iteration</c> instead).
    ///     </para>
    /// </remarks>
    public sealed class EnvSpecRegistry : IDictionary<string, EnvSpec>, IReadOnlyDictionary<string, EnvSpec> {
        /// <summary>Guards every read and write.</summary>
        private readonly object _gate = new();

        /// <summary>The entries in insertion order; removal shifts the ones after it, which keeps the order.</summary>
        private readonly List<KeyValuePair<string, EnvSpec>> _entries = new();

        /// <summary>The position of every id in <see cref="_entries"/>.</summary>
        private readonly Dictionary<string, int> _positions = new(StringComparer.Ordinal);

        /// <summary>An empty registry, Python's <c>{}</c>.</summary>
        public EnvSpecRegistry() { }

        /// <summary>A registry holding <paramref name="entries"/> in their order (a repeated id keeps its first position and last spec).</summary>
        /// <param name="entries">The entries, for example another registry.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="entries"/>, an id or a spec is <see langword="null"/>.</exception>
        public EnvSpecRegistry(IEnumerable<KeyValuePair<string, EnvSpec>> entries) {
            if (entries is null)
                throw new ArgumentNullException(nameof(entries));
            foreach (KeyValuePair<string, EnvSpec> entry in entries)
                this[entry.Key] = entry.Value;
        }

        /// <summary>The number of registered ids.</summary>
        public int Count {
            get {
                lock (_gate)
                    return _entries.Count;
            }
        }

        /// <summary>Always <see langword="false"/>: the registry can be edited, like a Python dict.</summary>
        public bool IsReadOnly => false;

        /// <summary>The ids in registration order (a snapshot).</summary>
        public ICollection<string> Keys {
            get {
                lock (_gate)
                    return _entries.Select(entry => entry.Key).ToList();
            }
        }

        /// <summary>The specs in registration order (a snapshot).</summary>
        public ICollection<EnvSpec> Values {
            get {
                lock (_gate)
                    return _entries.Select(entry => entry.Value).ToList();
            }
        }

        /// <inheritdoc cref="Keys"/>
        IEnumerable<string> IReadOnlyDictionary<string, EnvSpec>.Keys => Keys;

        /// <inheritdoc cref="Values"/>
        IEnumerable<EnvSpec> IReadOnlyDictionary<string, EnvSpec>.Values => Values;

        /// <summary>
        ///     Python's <c>registry[id]</c>: reading a missing id raises <see cref="KeyError"/>; assigning adds the id at the
        ///     end, or replaces the spec of a registered id in place. Assigning bypasses <c>register</c>'s checks, as in Python.
        /// </summary>
        /// <param name="key">The env id.</param>
        /// <returns>The spec.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="key"/> or the assigned spec is <see langword="null"/>.</exception>
        /// <exception cref="KeyError">Thrown when reading an id that is not registered; the message is the id's repr, <c>'x'</c>.</exception>
        public EnvSpec this[string key] {
            get {
                if (key is null)
                    throw new ArgumentNullException(nameof(key));
                lock (_gate) {
                    if (!_positions.TryGetValue(key, out int position))
                        throw new KeyError(PyBuiltins.Repr(key));
                    return _entries[position].Value;
                }
            }
            set {
                if (key is null)
                    throw new ArgumentNullException(nameof(key));
                if (value is null)
                    throw new ArgumentNullException(nameof(value));
                lock (_gate) {
                    if (_positions.TryGetValue(key, out int position)) {
                        _entries[position] = new KeyValuePair<string, EnvSpec>(key, value);
                    } else {
                        _positions.Add(key, _entries.Count);
                        _entries.Add(new KeyValuePair<string, EnvSpec>(key, value));
                    }
                }
            }
        }

        /// <summary>Adds a new id, as a collection initializer does; an id already registered is an error, as for every .NET dictionary.</summary>
        /// <param name="key">The env id.</param>
        /// <param name="value">The spec.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="key"/> or <paramref name="value"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown when <paramref name="key"/> is already registered.</exception>
        public void Add(string key, EnvSpec value) {
            if (key is null)
                throw new ArgumentNullException(nameof(key));
            lock (_gate) {
                if (_positions.ContainsKey(key))
                    throw new ArgumentException($"An item with the same key has already been added. Key: {key}", nameof(key));
                this[key] = value;
            }
        }

        /// <summary>Python's <c>id in registry</c>.</summary>
        /// <param name="key">The env id.</param>
        /// <returns><see langword="true"/> when it is registered.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="key"/> is <see langword="null"/>.</exception>
        public bool ContainsKey(string key) {
            if (key is null)
                throw new ArgumentNullException(nameof(key));
            lock (_gate)
                return _positions.ContainsKey(key);
        }

        /// <summary>Reads an id's spec without raising when it is missing.</summary>
        /// <param name="key">The env id.</param>
        /// <param name="value">The spec, or <see langword="null"/> when the id is not registered.</param>
        /// <returns><see langword="true"/> when the id is registered.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="key"/> is <see langword="null"/>.</exception>
        public bool TryGetValue(string key, [System.Diagnostics.CodeAnalysis.MaybeNullWhen(false)] out EnvSpec value) {
            if (key is null)
                throw new ArgumentNullException(nameof(key));
            lock (_gate) {
                if (_positions.TryGetValue(key, out int position)) {
                    value = _entries[position].Value;
                    return true;
                }
            }

            value = null;
            return false;
        }

        /// <summary>Python's <c>registry.get(id)</c>: the spec, or <see langword="null"/> when the id is not registered.</summary>
        /// <param name="key">The env id.</param>
        /// <returns>The spec, or <see langword="null"/>.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="key"/> is <see langword="null"/>.</exception>
        public EnvSpec? Get(string key) => TryGetValue(key, out EnvSpec? value) ? value : null;

        /// <summary>Python's <c>del registry[id]</c> without its error: unregisters an id.</summary>
        /// <param name="key">The env id.</param>
        /// <returns><see langword="true"/> when it was registered.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="key"/> is <see langword="null"/>.</exception>
        public bool Remove(string key) {
            if (key is null)
                throw new ArgumentNullException(nameof(key));
            lock (_gate) {
                if (!_positions.TryGetValue(key, out int position))
                    return false;
                _entries.RemoveAt(position);
                _positions.Remove(key);
                // Every entry after the removed one moved down by one.
                for (int i = position; i < _entries.Count; i++)
                    _positions[_entries[i].Key] = i;
                return true;
            }
        }

        /// <summary>Python's <c>registry.pop(id)</c>: unregisters an id and returns its spec.</summary>
        /// <param name="key">The env id.</param>
        /// <returns>The spec it had.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="key"/> is <see langword="null"/>.</exception>
        /// <exception cref="KeyError">Thrown when the id is not registered.</exception>
        public EnvSpec Pop(string key) {
            lock (_gate) {
                EnvSpec value = this[key];
                Remove(key);
                return value;
            }
        }

        /// <summary>Python's <c>registry.pop(id, default)</c>: unregisters an id if it is registered.</summary>
        /// <param name="key">The env id.</param>
        /// <param name="default">What to return when it is not registered.</param>
        /// <returns>The spec it had, or <paramref name="default"/>.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="key"/> is <see langword="null"/>.</exception>
        public EnvSpec? Pop(string key, EnvSpec? @default) {
            lock (_gate) {
                if (!TryGetValue(key, out EnvSpec? value))
                    return @default;
                Remove(key);
                return value;
            }
        }

        /// <summary>Python's <c>registry.copy()</c>: a new registry with the same entries (the specs are shared, not copied).</summary>
        /// <returns>The copy.</returns>
        public EnvSpecRegistry Copy() {
            lock (_gate)
                return new EnvSpecRegistry(_entries);
        }

        /// <summary>Unregisters every id.</summary>
        public void Clear() {
            lock (_gate) {
                _entries.Clear();
                _positions.Clear();
            }
        }

        /// <inheritdoc cref="Add(string, EnvSpec)"/>
        /// <param name="item">The entry.</param>
        void ICollection<KeyValuePair<string, EnvSpec>>.Add(KeyValuePair<string, EnvSpec> item) => Add(item.Key, item.Value);

        /// <summary>Whether the registry holds this id with this very spec.</summary>
        /// <param name="item">The entry.</param>
        /// <returns><see langword="true"/> when it does.</returns>
        bool ICollection<KeyValuePair<string, EnvSpec>>.Contains(KeyValuePair<string, EnvSpec> item) =>
            TryGetValue(item.Key, out EnvSpec? value) && Equals(value, item.Value);

        /// <summary>Copies the entries, in order, into an array.</summary>
        /// <param name="array">The destination.</param>
        /// <param name="arrayIndex">Where to start in it.</param>
        void ICollection<KeyValuePair<string, EnvSpec>>.CopyTo(KeyValuePair<string, EnvSpec>[] array, int arrayIndex) {
            lock (_gate)
                _entries.CopyTo(array, arrayIndex);
        }

        /// <summary>Removes the id when it holds this very spec.</summary>
        /// <param name="item">The entry.</param>
        /// <returns><see langword="true"/> when it was removed.</returns>
        bool ICollection<KeyValuePair<string, EnvSpec>>.Remove(KeyValuePair<string, EnvSpec> item) {
            lock (_gate)
                return ((ICollection<KeyValuePair<string, EnvSpec>>)this).Contains(item) && Remove(item.Key);
        }

        /// <summary>Enumerates a snapshot of the entries in registration order.</summary>
        /// <returns>The enumerator.</returns>
        public IEnumerator<KeyValuePair<string, EnvSpec>> GetEnumerator() {
            List<KeyValuePair<string, EnvSpec>> snapshot;
            lock (_gate)
                snapshot = new List<KeyValuePair<string, EnvSpec>>(_entries);
            return snapshot.GetEnumerator();
        }

        /// <inheritdoc cref="GetEnumerator"/>
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        /// <summary>Python's <c>repr(registry)</c>: <c>{'CartPole-v1': EnvSpec(...), ...}</c>.</summary>
        /// <returns>The text.</returns>
        public override string ToString() {
            lock (_gate)
                return "{" + string.Join(", ", _entries.Select(entry => $"{PyBuiltins.Repr(entry.Key)}: {entry.Value}")) + "}";
        }
    }
}
