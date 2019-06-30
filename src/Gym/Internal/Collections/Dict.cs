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
    ///     Alias to <see cref=" Dictionary{string,object}"/> with ability to construct inline, as dynamic, merging and cloning.
    /// </summary>
    [Serializable]
    public class Dict : Dictionary<string, object>, IMergable<IDictionary>, IMergable<IList>, IMergable<IMergable> {
        /// <summary>
        ///     Returns a new empty Dict with 0 capacity (not fixed, meaning you can add items to it).
        /// </summary>
        public static Dict Empty => new Dict(0);

        #region Consturctors

        public Dict() { }

        public Dict(int capacity) : base(capacity) { }

        public Dict(IEqualityComparer<string> comparer) : base(comparer) { }

        public Dict(int capacity, IEqualityComparer<string> comparer) : base(capacity, comparer) { }

        public Dict([NotNull] IDictionary<string, object> dictionary) : base(dictionary) { }

        public Dict([NotNull] IEnumerable<KeyValuePair<string, object>> keyValuePairs) {
            foreach (var pair in keyValuePairs) {
                Add(pair.Key, pair.Value);
            }
        }

        public Dict([NotNull] IDictionary<string, object> dictionary, IEqualityComparer<string> comparer) : base(dictionary, comparer) { }

        public Dict(params object[] inlines) {
            AddInline(inlines);
        }

        #endregion

        /// <summary>
        ///     Returns this wrapped in private class, allowing to class AddRange, Set, Update in a dynamic manner like python:
        /// 
        /// </summary>
        /// <returns></returns>
        /// <example>Dict d = new Dict().AsDynamic().AddRange(x: 1, y: 2);</example>
        public dynamic AsDynamic() {
            return new DynamicAccessor(this);
        }

        /// <summary>
        ///     Appends or sets given value to given key. returns self, not a copy.
        /// </summary>
        public Dict Append(string key, object val) {
            this[key] = val;
            return this;
        }

        /// <summary>
        ///     Appends or sets given value to given key. returns self, not a copy.
        /// </summary>
        public Dict Append(IDictionary dict) {
            Update(dict);
            return this;
        }

        public Dict Override(IDictionary dict) {
            Update(dict);
            return this;
        }

        /// <summary>
        ///     Simple this[key] and cast.
        /// </summary>
        /// <exception cref="InvalidCastException">When the object in dictionary corresponding to key <paramref name="key"/> is not <typeparamref name="T"/></exception>
        public T Get<T>(string key) {
            if (this.TryGetValue(key, out var val)) {
                return (T) val;
            }

            return default;
        }

        /// <summary>
        ///     Simple this[key] and cast or return default.
        /// </summary>
        /// <exception cref="InvalidCastException">When the object in dictionary corresponding to key <paramref name="key"/> is not <typeparamref name="T"/></exception>
        public T Get<T>(string key, T @default) {
            if (this.TryGetValue(key, out var val)) {
                return (T) val;
            }

            return @default;
        }

        /// <summary>
        ///     Simple this[key] and cast or return default.
        /// </summary>
        /// <exception cref="InvalidCastException">When the object in dictionary corresponding to key <paramref name="key"/> is not <typeparamref name="T"/></exception>
        public T Get<T>(string key, Func<T> @default) {
            if (this.TryGetValue(key, out var val)) {
                return (T) val;
            }

            return @default();
        }

        /// <summary>
        ///     Updates/adds all values in this dictionary, based on <paramref name="dict"/>.
        /// </summary>
        /// <param name="dict"></param>
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
        /// <param name="dict"></param>
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
        /// <param name="inlines"></param>
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
        /// <param name="valueFactory">The</param>
        /// <returns>The amount of matching predictions and successful updates</returns>
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

        [DebuggerStepThrough]
        private class DynamicAccessor : Expando, ICloneable {
            /// <inheritdoc />
            public DynamicAccessor(Dict dict) : base() {
                this.Properties = dict;
                this.Instance = dict;
            }

            /// <inheritdoc />
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

            public static implicit operator Dict(DynamicAccessor acc) {
                return (Dict) acc.Instance;
            }

            /// <inheritdoc />
            public object Clone() {
                return new DynamicAccessor(((Dict) Instance).Clone());
            }
        }

        /// <inheritdoc />
        public void Merge(IDictionary right, MergePrefer preference) {
            Merger.Merge(this, right, preference);
        }

        /// <inheritdoc />
        void IMergable<IList>.Merge(IList right, MergePrefer preference) {
            Merger.Merge(this, right, preference);
        }

        void IMergable<IMergable>.Merge(IMergable right, MergePrefer preference) {
            Merger.Merge(this, right, preference);
        }

        /// <inheritdoc />
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