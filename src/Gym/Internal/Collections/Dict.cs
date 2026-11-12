// Gym.Internal is 0.2's code, [Obsolete] since 1.0 and removed in 2.0 with Gym.Legacy (DR-56): it keeps no nullable
// context (DR-54), rather than 89 annotations for code on its way out.
#nullable disable
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Dynamic;
using System.Linq;
using Gym.Dynamic;
using Gym.Merging;
using JetBrains.Annotations;

namespace Gym.Collections {
    /// <summary>
    ///     Alias to <see cref="Dictionary{TKey, TValue}"/> of <see cref="string"/> to <see cref="object"/> with ability to
    ///     construct inline, as dynamic, merging and cloning: 0.2's info and metadata type, which <c>Gym.Legacy</c> still
    ///     returns (<c>Step.Information</c>, <c>ILegacyEnv.Metadata</c>, <c>VecEnv.Metadata</c>).
    /// </summary>
    /// <remarks>
    ///     Obsolete (DR-56) and removed in 2.0 with <c>Gym.Legacy</c>. 1.0's envs use <see cref="Gym.Envs.Info"/> for infos
    ///     (Python's dict: insertion order kept, <c>Get(key, default)</c>, <c>Update</c>, <c>Copy</c>/<c>DeepCopy</c>,
    ///     <c>Pop</c>) and <see cref="Gym.Envs.Metadata"/> for metadata. Unlike <see cref="Gym.Envs.Info"/>, enumeration
    ///     order is <see cref="Dictionary{TKey, TValue}"/>'s, which removals can reorder. It shares its name with the Dict
    ///     space (<c>Gym.Spaces.Dict</c>): a file importing both namespaces must alias one (CS0104).
    /// </remarks>
    [Serializable]
    [Obsolete("Dict is the legacy API's info and metadata type: use Gym.Envs.Info for infos and Gym.Envs.Metadata for metadata. Removed in 2.0 with Gym.Legacy.")]
    public class Dict : Dictionary<string, object>, IMergable<IDictionary>, IMergable<IList>, IMergable<IMergable> {
        /// <summary>
        ///     Returns a new empty Dict with 0 capacity (not fixed, meaning you can add items to it).
        /// </summary>
        /// <remarks>A new dictionary on every read, not a shared instance, so mutating it is safe.</remarks>
        public static Dict Empty => new Dict(0);

        #region Consturctors

        /// <summary>An empty dictionary with the default (ordinal) key comparer.</summary>
        public Dict() { }

        /// <summary>An empty dictionary with room for <paramref name="capacity"/> entries before it grows.</summary>
        /// <param name="capacity">The initial capacity. Footgun: <c>new Dict(5)</c> is this constructor, not one inline value.</param>
        /// <exception cref="ArgumentOutOfRangeException">When <paramref name="capacity"/> is negative.</exception>
        public Dict(int capacity) : base(capacity) { }

        /// <summary>An empty dictionary comparing keys with <paramref name="comparer"/>.</summary>
        /// <param name="comparer">The key comparer (for example <see cref="StringComparer.OrdinalIgnoreCase"/>); <see langword="null"/> means the default.</param>
        public Dict(IEqualityComparer<string> comparer) : base(comparer) { }

        /// <summary>An empty dictionary with an initial capacity and a key comparer.</summary>
        /// <param name="capacity">The initial capacity.</param>
        /// <param name="comparer">The key comparer; <see langword="null"/> means the default.</param>
        /// <exception cref="ArgumentOutOfRangeException">When <paramref name="capacity"/> is negative.</exception>
        public Dict(int capacity, IEqualityComparer<string> comparer) : base(capacity, comparer) { }

        /// <summary>A shallow copy of <paramref name="dictionary"/>'s entries (the values are shared, not copied).</summary>
        /// <param name="dictionary">The entries to copy.</param>
        /// <exception cref="ArgumentNullException">When <paramref name="dictionary"/> is <see langword="null"/>.</exception>
        public Dict([NotNull] IDictionary<string, object> dictionary) : base(dictionary) { }

        /// <summary>A dictionary of the given pairs, in order (values shared, not copied).</summary>
        /// <param name="keyValuePairs">The pairs.</param>
        /// <exception cref="ArgumentException">When two pairs share a key.</exception>
        /// <exception cref="NullReferenceException">When <paramref name="keyValuePairs"/> is <see langword="null"/>.</exception>
        public Dict([NotNull] IEnumerable<KeyValuePair<string, object>> keyValuePairs) {
            foreach (var pair in keyValuePairs) {
                Add(pair.Key, pair.Value);
            }
        }

        /// <summary>A shallow copy of <paramref name="dictionary"/>'s entries, comparing keys with <paramref name="comparer"/>.</summary>
        /// <param name="dictionary">The entries to copy.</param>
        /// <param name="comparer">The key comparer; <see langword="null"/> means the default.</param>
        /// <exception cref="ArgumentNullException">When <paramref name="dictionary"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">When two keys are equal under <paramref name="comparer"/>.</exception>
        public Dict([NotNull] IDictionary<string, object> dictionary, IEqualityComparer<string> comparer) : base(dictionary, comparer) { }

        /// <summary>
        ///     A dictionary of inline key-value pairs: <c>new Dict("render.modes", modes, "video.frames_per_second", 50)</c>.
        ///     A later pair with the same key overwrites the earlier.
        /// </summary>
        /// <param name="inlines">Keys and values, alternating; each key must be a <see cref="string"/>.</param>
        /// <exception cref="DictInlineInitializationException">When the count is odd, or a key is <see langword="null"/> or not a <see cref="string"/>.</exception>
        public Dict(params object[] inlines) {
            AddInline(inlines);
        }

        #endregion

        /// <summary>
        ///     Returns this wrapped in private class, allowing to class AddRange, Set, Update in a dynamic manner like python:
        ///     named arguments become keys, and a dynamic member read finds the dictionary's keys.
        /// </summary>
        /// <returns>
        ///     A dynamic view over this dictionary (writes go to it), convertible back to <see cref="Dict"/>. Footguns:
        ///     <c>AddRange</c> takes named arguments only, since a positional one shifts the names onto the wrong values;
        ///     and the view resolves reflected members on its own type, not the dictionary's, so read the dictionary's own
        ///     properties (<c>Count</c>, <c>Keys</c>) on the <see cref="Dict"/> itself.
        /// </returns>
        /// <example>Dict d = new Dict().AsDynamic().AddRange(x: 1, y: 2);</example>
        public dynamic AsDynamic() {
            return new DynamicAccessor(this);
        }

        /// <summary>
        ///     Appends or sets given value to given key. returns self, not a copy.
        /// </summary>
        /// <param name="key">The key.</param>
        /// <param name="val">The value, stored as given.</param>
        /// <returns>This dictionary, for chaining.</returns>
        /// <exception cref="ArgumentNullException">When <paramref name="key"/> is <see langword="null"/>.</exception>
        public Dict Append(string key, object val) {
            this[key] = val;
            return this;
        }

        /// <summary>
        ///     Appends or sets given value to given key. returns self, not a copy.
        /// </summary>
        /// <param name="dict">The entries to set; despite the name, existing keys are overwritten (as <see cref="Update(IDictionary)"/>).</param>
        /// <returns>This dictionary, for chaining.</returns>
        /// <exception cref="NullReferenceException">When an entry's key is <see langword="null"/>.</exception>
        public Dict Append(IDictionary dict) {
            Update(dict);
            return this;
        }

        /// <summary>Sets every entry of <paramref name="dict"/>, overwriting existing keys: the same as <see cref="Append(IDictionary)"/>.</summary>
        /// <param name="dict">The entries to set; <see langword="null"/> sets nothing.</param>
        /// <returns>This dictionary, for chaining.</returns>
        /// <exception cref="NullReferenceException">When an entry's key is <see langword="null"/>.</exception>
        public Dict Override(IDictionary dict) {
            Update(dict);
            return this;
        }

        /// <summary>
        ///     Simple this[key] and cast.
        /// </summary>
        /// <typeparam name="T">The type to cast to; a boxed value casts only to its exact type (an <see cref="int"/> stored is not a <see cref="long"/>).</typeparam>
        /// <param name="key">The key.</param>
        /// <returns>The value cast to <typeparamref name="T"/>. Footgun: a missing key returns <c>default(T)</c> silently.</returns>
        /// <exception cref="InvalidCastException">When the object in dictionary corresponding to key <paramref name="key"/> is not <typeparamref name="T"/></exception>
        /// <exception cref="ArgumentNullException">When <paramref name="key"/> is <see langword="null"/>.</exception>
        public T Get<T>(string key) {
            if (this.TryGetValue(key, out var val)) {
                return (T) val;
            }

            return default;
        }

        /// <summary>
        ///     Simple this[key] and cast or return default.
        /// </summary>
        /// <typeparam name="T">The type to cast to.</typeparam>
        /// <param name="key">The key.</param>
        /// <param name="default">Returned when the key is missing.</param>
        /// <returns>The value cast to <typeparamref name="T"/>, or <paramref name="default"/>.</returns>
        /// <exception cref="InvalidCastException">When the object in dictionary corresponding to key <paramref name="key"/> is not <typeparamref name="T"/></exception>
        /// <exception cref="ArgumentNullException">When <paramref name="key"/> is <see langword="null"/>.</exception>
        public T Get<T>(string key, T @default) {
            if (this.TryGetValue(key, out var val)) {
                return (T) val;
            }

            return @default;
        }

        /// <summary>
        ///     Simple this[key] and cast or return default.
        /// </summary>
        /// <typeparam name="T">The type to cast to.</typeparam>
        /// <param name="key">The key.</param>
        /// <param name="default">Called for the value to return when the key is missing; not called otherwise.</param>
        /// <returns>The value cast to <typeparamref name="T"/>, or what <paramref name="default"/> returns.</returns>
        /// <exception cref="InvalidCastException">When the object in dictionary corresponding to key <paramref name="key"/> is not <typeparamref name="T"/></exception>
        /// <exception cref="ArgumentNullException">When <paramref name="key"/> is <see langword="null"/>.</exception>
        /// <exception cref="NullReferenceException">When the key is missing and <paramref name="default"/> is <see langword="null"/>.</exception>
        public T Get<T>(string key, Func<T> @default) {
            if (this.TryGetValue(key, out var val)) {
                return (T) val;
            }

            return @default();
        }

        /// <summary>
        ///     Updates/adds all values in this dictionary, based on <paramref name="dict"/>.
        /// </summary>
        /// <param name="dict">The entries to set, overwriting existing keys; a key that is not a <see cref="string"/> is stored under its <c>ToString()</c>. <see langword="null"/> sets nothing.</param>
        /// <exception cref="NullReferenceException">When an entry's key is <see langword="null"/>; the entries before it are already set.</exception>
        public void Update(IDictionary dict) {
            if (dict == null)
                return;
            foreach (DictionaryEntry d in dict) {
                if (d.Key == null)
                    throw new NullReferenceException();
                this[d.Key is string s ? s : d.Key.ToString()] = d.Value;
            }
        }

        /// <summary>
        ///     Updates/adds all values in this dictionary, based on <paramref name="dict"/>.
        /// </summary>
        /// <param name="dict">The entries to set, overwriting existing keys; <see langword="null"/> sets nothing.</param>
        /// <exception cref="NullReferenceException">When an entry's key is <see langword="null"/> (a <see cref="Dict"/> cannot hold one).</exception>
        public void Update(Dict dict) {
            if (dict == null)
                return;
            foreach (var d in dict) {
                if (d.Key == null)
                    throw new NullReferenceException();
                this[d.Key] = d.Value;
            }
        }

        /// <summary>
        ///     Creates a copy of current Dict.
        /// </summary>
        /// <returns>A new copy of this dict, doesn't clone the contents, just references to them.</returns>
        public Dict Clone() {
            return new Dict(this, this.Comparer);
        }

        /// <summary>
        ///     Adds or overwrites inline props, for example ("A", 1) will will do dictionary["A"] = 1;<br></br>But ("A", 1, "B") will throw because no value is assigned to "B" (No even arguments passed).
        /// </summary>
        /// <param name="inlines">Keys and values, alternating; each key must be exactly a <see cref="string"/>.</param>
        /// <exception cref="DictInlineInitializationException">When the count is odd (checked first, so nothing is set), or a key is <see langword="null"/> or not a <see cref="string"/> (the pairs before it are set).</exception>
        /// <exception cref="NullReferenceException">When <paramref name="inlines"/> itself is <see langword="null"/>.</exception>
        public void AddInline(params object[] inlines) {
            if (inlines.Length % 2 != 0)
                throw new DictInlineInitializationException("The amount of inline params passed must be even, got odd length.");

            for (int i = 0; i < inlines.Length; i += 2) {
                if (inlines[i] == null)
                    throw new DictInlineInitializationException("Name can't be null");
                if (inlines[i].GetType() != typeof(string))
                    throw new DictInlineInitializationException("Name must be string.");
                this[(string) inlines[i]] = inlines[i + 1];
            }
        }

        /// <summary>
        ///     Similar to <see cref="List{T}.RemoveAll"/>.
        /// </summary>
        /// <param name="comparer">The comparer that returns true when to remove and false to keep.</param>
        /// <returns>The amount of matching predictions and successful removals</returns>
        /// <exception cref="NullReferenceException">When <paramref name="comparer"/> is <see langword="null"/> and the dictionary is not empty.</exception>
        public int RemoveWhere(Predicate<KeyValuePair<string, object>> comparer) {
            var count = 0;
            foreach (var kv in this.Where(kv => comparer(kv)).ToList()) {
                //copy to allow removal
                if (Remove(kv.Key))
                    count++;
            }

            return count;
        }

        /// <summary>
        ///     Updates properties that return (true, ...).
        /// </summary>
        /// <param name="comparer">The comparer that returns (bool Set, object Value).</param>
        /// <returns>The amount of matching predictions and successful updates</returns>
        /// <exception cref="ArgumentNullException">When <paramref name="comparer"/> is <see langword="null"/>.</exception>
        public int SetWhere([NotNull] Func<KeyValuePair<string, object>, (bool Set, object Value)> comparer) {
            if (comparer == null) throw new ArgumentNullException(nameof(comparer));
            var count = 0;
            foreach (var kv in this.Select(kv => (Left: kv, Right: comparer(kv))).Where(kv => kv.Right.Set).ToList()) {
                this[kv.Left.Key] = kv.Right.Value;
                count++;
            }

            return count;
        }

        /// <summary>
        ///     Updates properties that return (true, ...).
        /// </summary>
        /// <param name="settingsComparer">The comparer that returns if this key should have it's value replaced.</param>
        /// <param name="valueFactory">Called for each selected entry, after all entries were tested, for its new value.</param>
        /// <returns>The amount of matching predictions and successful updates</returns>
        /// <exception cref="ArgumentNullException">When <paramref name="settingsComparer"/> or <paramref name="valueFactory"/> is <see langword="null"/>.</exception>
        public int SetWhere([NotNull] Func<KeyValuePair<string, object>, bool> settingsComparer, [NotNull] Func<KeyValuePair<string, object>, object> valueFactory) {
            if (settingsComparer == null) throw new ArgumentNullException(nameof(settingsComparer));
            if (valueFactory == null) throw new ArgumentNullException(nameof(valueFactory));
            var count = 0;
            foreach (var kv in this.Select(kv => (Left: kv, Right: settingsComparer(kv))).Where(kv => kv.Right).ToList()) {
                this[kv.Left.Key] = valueFactory(kv.Left);
                count++;
            }

            return count;
        }

        /// <summary>
        ///     <see cref="AsDynamic"/>'s view: an <see cref="Expando"/> whose dictionary is the <see cref="Dict"/> itself, with
        ///     <c>AddRange</c>/<c>Set</c>/<c>Update</c> taking named arguments as entries and <c>Clone</c> copying.
        /// </summary>
        [DebuggerStepThrough]
        private class DynamicAccessor : Expando, ICloneable {
            /// <summary>A view over <paramref name="dict"/>: dynamic reads and writes go to it.</summary>
            /// <param name="dict">The dictionary to expose.</param>
            public DynamicAccessor(Dict dict) : base() {
                this.Properties = dict;
                this.Instance = dict;
            }

            /// <summary>
            ///     Handles <c>AddRange</c>, <c>Set</c> and <c>Update</c> (each named argument becomes an entry; the call returns
            ///     the view, for chaining) and <c>Clone</c> (a view over a shallow copy); other calls go to <see cref="Expando"/>.
            /// </summary>
            /// <param name="binder">The dynamic call, with its argument names.</param>
            /// <param name="args">The arguments.</param>
            /// <param name="result">The view, a cloned view, or the reflected method's result.</param>
            /// <returns>Whether the call was handled. Footgun: the argument-count check guards all four names, so a <c>Clone()</c> without arguments skips this case and falls through to <see cref="Expando"/>'s reflection.</returns>
            public override bool TryInvokeMember(InvokeMemberBinder binder, object[] args, out object result) {
                if (args != null && args.Length > 0) {
                    if (binder.Name == "AddRange" || binder.Name == "Set" || binder.Name == "Update") {
                        var namedargs = args.Zip(binder.CallInfo.ArgumentNames, (val, key) => (key: key, val: val));
                        foreach (var kv in namedargs)
                            Properties[kv.key] = kv.val;
                        result = this;
                        return true;
                    } else if (binder.Name == "Clone") {
                        result = (DynamicAccessor) Clone();
                        return true;
                    }
                }

                return base.TryInvokeMember(binder, args, out result);
            }

            /// <summary>The <see cref="Dict"/> a view shows, so <c>Dict d = dict.AsDynamic().AddRange(x: 1);</c> compiles.</summary>
            /// <param name="acc">The view.</param>
            /// <returns>The dictionary behind it (the same instance, not a copy).</returns>
            /// <exception cref="NullReferenceException">When <paramref name="acc"/> is <see langword="null"/>.</exception>
            public static implicit operator Dict(DynamicAccessor acc) {
                return (Dict) acc.Instance;
            }

            /// <summary>A view over a shallow copy of the dictionary.</summary>
            /// <returns>The new view.</returns>
            public object Clone() {
                return new DynamicAccessor(((Dict) Instance).Clone());
            }
        }

        /// <summary>
        ///     Deep-merges <paramref name="right"/> into this dictionary with <see cref="Merger"/>: nested dictionaries merged,
        ///     lists appended, new values copied in, and <paramref name="preference"/> settling two plain values.
        /// </summary>
        /// <param name="right">The dictionary to merge in.</param>
        /// <param name="preference">The side that wins when two values cannot be merged.</param>
        /// <exception cref="DictMergingException">When one key's merge fails; the keys before it stay merged.</exception>
        public void Merge(IDictionary right, MergePrefer preference) {
            Merger.Merge(this, right, preference);
        }

        /// <inheritdoc />
        void IMergable<IList>.Merge(IList right, MergePrefer preference) {
            Merger.Merge(this, right, preference);
        }

        /// <inheritdoc />
        void IMergable<IMergable>.Merge(IMergable right, MergePrefer preference) {
            Merger.Merge(this, right, preference);
        }

        /// <summary>
        ///     Deep-merges <paramref name="right"/> into this dictionary: a dictionary, another <see cref="IMergable"/> or a
        ///     sequence of string-keyed pairs. Footgun: any other value merges nothing, silently.
        /// </summary>
        /// <param name="right">The value to merge in.</param>
        /// <param name="preference">The side that wins when two values cannot be merged.</param>
        /// <exception cref="DictMergingException">When one key's merge fails; the keys before it stay merged.</exception>
        public void Merge(object right, MergePrefer preference) {
            Merger.Merge(this, right, preference);
        }

        /// <inheritdoc />
        IEnumerable<KeyValuePair<string, object>> IMergable.Contents => this;

        /// <inheritdoc />
        void IMergable.Set(string key, object value) {
            this[key] = value;
        }

        /// <inheritdoc />
        object IMergable.Get(string key) => this[key];

        /// <inheritdoc />
        IEnumerable<string> IMergable.Keys => base.Keys;

        /// <inheritdoc />
        bool IMergable.IsExpandable => true;
    }
}
