// Gym.Internal is 0.2's code, [Obsolete] since 1.0 and removed in 2.0 with Gym.Legacy (DR-56): it keeps no nullable
// context (DR-54), rather than 89 annotations for code on its way out.
#nullable disable
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Gym.Collections;
using Gym.Reflection;

namespace Gym.Merging {
    /// <summary>
    ///     The legacy deep merge behind <see cref="Dict"/>'s <c>Merge</c>: nested dictionaries are merged key by key, lists
    ///     are appended to, values only the right side has are copied in (lists and dictionaries cloned), and two plain
    ///     values on the same key are settled by <see cref="MergePrefer"/>. The left side is changed in place.
    /// </summary>
    /// <remarks>
    ///     Obsolete with <see cref="Dict"/> (DR-56) and removed in 2.0; <see cref="Gym.Envs.Info.Update"/> replaces it, one
    ///     level deep with the new value winning, as Python's <c>dict.update</c>. Footguns: lists are concatenated whatever
    ///     the preference; a fixed-size or read-only left dictionary is returned unmerged, silently; merging a dictionary
    ///     into a non-expandable <see cref="IMergable"/> overwrites its existing keys whatever the preference; a right side
    ///     of any other type merges nothing; and a failing key leaves the keys before it merged.
    /// </remarks>
    [Obsolete("The legacy Dict's deep merge: use Gym.Envs.Info.Update (Python's dict.update: the new value wins, one level deep). Removed in 2.0 with Gym.Legacy.")]
    public static class Merger {
        /// <summary>
        ///     Merges <paramref name="right"/> with <paramref name="left"/> object, appending all lists and merging dictionaries.
        /// </summary>
        /// <param name="left">The object that will accept all merges.</param>
        /// <param name="right">The dictionary to take data from.</param>
        /// <param name="preference">See the enum values docs.</param>
        /// <exception cref="DictMergingException">Thrown when one key's merge fails; the inner exception is the cause, and the keys before it stay merged.</exception>
        public static void Merge(IMergable left, IMergable right, MergePrefer preference) {
            merging_core(left, right, preference);
        }

        /// <summary>
        ///     Merges <paramref name="right"/> with current dictionary, appending all lists and merging dictionaries.
        /// </summary>
        /// <param name="left">The object that will accept all merges.</param>
        /// <param name="right">The dictionary to take data from: an <see cref="IDictionary"/>, an <see cref="IMergable"/> or a sequence of string-keyed pairs; anything else merges nothing.</param>
        /// <param name="preference">See the enum values docs.</param>
        /// <exception cref="DictMergingException">Thrown when one key's merge fails; the inner exception is the cause, and the keys before it stay merged.</exception>
        public static void Merge(IMergable left, object right, MergePrefer preference) {
            merging_core(left, right, preference);
        }


        /// <summary>
        ///     Merges <paramref name="right"/> with <paramref name="left"/> dictionary, appending all lists and merging dictionaries.
        /// </summary>
        /// <param name="left">The object that will accept all merges.</param>
        /// <param name="right">The dictionary to take data from.</param>
        /// <param name="prefer">See the enum values docs.</param>
        /// <exception cref="DictMergingException">Thrown when one key's merge fails; the inner exception is the cause, and the keys before it stay merged.</exception>
        /// <remarks>A <see langword="null"/> or empty <paramref name="right"/> merges nothing; so does a fixed-size or read-only <paramref name="left"/>, silently.</remarks>
        public static void Merge(IDictionary left, IDictionary right, MergePrefer prefer = MergePrefer.Old) {
            if (right == null || right.Count == 0)
                return;
            merging_core(left, right, prefer);
        }

        /// <summary>
        ///     Merges any two values and returns the result: the merged dictionary or list (the left one, changed in place)
        ///     when both sides are dictionaries or lists, the side <paramref name="preference"/> picks otherwise.
        /// </summary>
        /// <param name="left">The value that receives the merge.</param>
        /// <param name="right">The value merged into it.</param>
        /// <param name="preference">The side that wins when the values cannot be merged.</param>
        /// <returns>The merged value: <paramref name="left"/> changed, a replacement list, or the preferred side.</returns>
        /// <exception cref="DictMergingException">Thrown when a dictionary key's merge fails.</exception>
        /// <exception cref="NotSupportedException">Thrown when a list reports itself growable but refuses <c>Add</c>.</exception>
        private static object merging_core(object left, object right, MergePrefer preference) {
            //handle nulls
            if (left == null && right != null)
                return right;

            if (right == null)
                return left;

            if (ReferenceEquals(left, right))
                return left;

            if (left is IMergable leftmerger) {
                return merging_core_imerger(leftmerger, right, preference);
            }

            //handle dict
            if (left is IDictionary maindict && right is IDictionary otherdict) {
                //handle empty, fixed or readonly
                if (otherdict.Count == 0 || maindict.IsFixedSize || maindict.IsReadOnly)
                    return maindict;
                //get keys to process
                var mainkeys = maindict.Keys.Cast<object>().ToList();
                var otherkeys = otherdict.Keys.Cast<object>().ToList();
                var sharedkeys = mainkeys.Intersect(otherkeys); //keys both share
                var missingkeys = otherkeys.Except(mainkeys); //keys only otherdict has

                object current_key = null; //for exception handling
                try {
                    //handle merging of shared keys
                    foreach (var key in sharedkeys) {
                        current_key = key;
                        maindict[key] = merging_core(maindict[key], otherdict[key], preference); //try to merge both values before returning.
                    }

                    //handle settings of missing keys in sharedkeys
                    foreach (var key in missingkeys) {
                        current_key = key;
                        var val = otherdict[key];
                        maindict[key] = TryClone(val);
                    }
                } catch (Exception e) {
                    throw new DictMergingException($"Merging failed during merge of key '{current_key}', see inner exception.", e);
                }

                return maindict;
            }

            //handle list
            if (left is IList mainlist && right is IList otherlist) {
                //handle left expandable
                if (mainlist.IsFixedSize || mainlist.IsReadOnly)
                    if (mainlist is ExpandableList e)
                        mainlist = e.Expand();

                //handle fixed or readonly
                if (mainlist.IsFixedSize || mainlist.IsReadOnly) {
                    if (mainlist is ExpandableList e)
                        mainlist = e.Expand();
                    //if other is also fixed, swap.
                    if (otherlist.IsFixedSize || otherlist.IsReadOnly)
                        return preference == MergePrefer.Old ? left : right;
                    return mainlist;
                }

                //append all data from other to main.
                foreach (var l in otherlist) {
                    mainlist.Add(l);
                }

                return mainlist;
            }

            return preference == MergePrefer.Old ? left : right;
        }

        /// <summary>
        ///     A copy of a value the right side alone has, so the merged dictionary does not share it: arrays, lists and
        ///     dictionaries are copied one level deep, an <see cref="ExpandableList"/> becomes a real list, anything else is
        ///     returned as it is.
        /// </summary>
        /// <param name="obj">The value to copy.</param>
        /// <returns>The copy, or <paramref name="obj"/> itself when it is not a collection.</returns>
        /// <exception cref="Exception">Thrown for an <see cref="IList"/> that is neither an array, an <see cref="ExpandableList"/> nor a <see cref="List{T}"/>.</exception>
        /// <exception cref="InvalidOperationException">Thrown for a non-generic <see cref="IList"/> such as <see cref="ArrayList"/> (<c>GetGenericTypeDefinition</c> on a non-generic type).</exception>
        /// <exception cref="DefaultConstructorNotFoundException">Thrown for a dictionary type without a parameterless constructor.</exception>
        private static object TryClone(object obj) {
            if (obj == null)
                return null;
            var type = obj.GetType();
            if (obj is IList list) {
                if (type.IsArray) {
                    var ret = Array.CreateInstance(type.GetElementType(), list.Count);
                    Array.Copy((Array) list, 0, ret, 0, list.Count);
                    return ret;
                }

                if (list is ExpandableList e)
                    return e.Expand();

                if (type.GetGenericTypeDefinition() == typeof(List<>)) {
                    var ret = (IList) DefaultConstructor.Activator(type)();
                    foreach (var o in list) {
                        ret.Add(o);
                    }

                    return ret;
                }

                throw new Exception($"Unable to clone {type.FullName} from IList");
            }

            if (obj is IDictionary dict) {
                var ret = (IDictionary) DefaultConstructor.Activator(type)();
                foreach (DictionaryEntry entry in dict) {
                    ret.Add(entry.Key, entry.Value);
                }

                return ret;
            }

            return obj;
        }

        /// <summary>
        ///     Merges <paramref name="right"/> into an <see cref="IMergable"/> through its key-by-key interface: shared keys
        ///     are merged with <see cref="merging_core"/>, missing keys copied in when the target is expandable.
        /// </summary>
        /// <param name="left">The mergeable that receives the merge, changed in place.</param>
        /// <param name="right">A dictionary, another mergeable, or a sequence of string-keyed pairs (wrapped in a <see cref="Dict"/> first).</param>
        /// <param name="preference">The side that wins when two values cannot be merged.</param>
        /// <returns><paramref name="left"/>, or the preferred side when <paramref name="right"/> is of no mergeable type.</returns>
        /// <exception cref="DictMergingException">Thrown when one key's merge fails, including a non-string key (an <see cref="InvalidOperationException"/> inside).</exception>
        private static object merging_core_imerger(IMergable left, object right, MergePrefer preference) {
            //handle nulls
            if (left == null && right != null)
                return right;

            if (right == null)
                return left;

            if (ReferenceEquals(left, right))
                return left;

            //handle kv pairs
            if (right is IEnumerable<KeyValuePair<string, object>> rightenumerable && !(right is IDictionary)) {
                right = new Dict(rightenumerable);
            }

            //handle dict
            if (right is IDictionary rightdict) {
                //handle empty, fixed or readonly
                if (rightdict.Count == 0)
                    return left;

                if (!left.IsExpandable)
                    foreach (var leftkey in left.Keys) {
                        if (rightdict.Contains(leftkey))
                            left.Set(leftkey, rightdict[leftkey]);
                    }
                else {
                    //get keys to process
                    var mainkeys = left.Keys.Cast<object>().ToList();
                    var otherkeys = rightdict.Keys.Cast<object>().ToList();
                    var sharedkeys = mainkeys.Intersect(otherkeys); //keys both share
                    var missingkeys = otherkeys.Except(mainkeys); //keys only otherdict has

                    object current_key = null; //for exception handling
                    try {
                        //handle merging of shared keys
                        foreach (var key in sharedkeys) {
                            current_key = key;
                            var leftkey = (key as string) ?? key.ToString();
                            try {
                                left.Set(leftkey, merging_core(left.Get(leftkey), rightdict[key], preference)); //try to merge both values before returning.
                            } catch (KeyNotFoundException) { } //has no setter.
                        }

                        //handle settings of missing keys in sharedkeys
                        foreach (var key in missingkeys) {
                            current_key = key;
                            try {
                                left.Set(
                                    (key as string) ?? throw new InvalidOperationException("A key for IMergable must be a string."),
                                    _trycopy(rightdict[key])
                                ); //try to merge both values before returning.
                            } catch (KeyNotFoundException) { } //has no setter.
                        }
                    } catch (Exception e) {
                        throw new DictMergingException($"Merging failed during merge of key '{current_key}', see inner exception.", e);
                    }
                }

                return left;
            }

            //handle dict
            if (right is IMergable rightmergable) {
                //handle empty, fixed or readonly
                if (rightmergable.Count == 0)
                    return left;
                var rightkeys = rightmergable.Keys.ToArray();
                if (!left.IsExpandable) {
                    if (preference == MergePrefer.New) {
                        foreach (var leftkey in left.Keys) {
                            if (rightkeys.Contains(leftkey))
                                left.Set(leftkey, rightmergable.Get(leftkey));
                        }
                    }
                } else {
                    //get keys to process
                    var mainkeys = left.Keys.Cast<object>().ToList();
                    var otherkeys = rightkeys.Cast<object>().ToList();
                    var sharedkeys = mainkeys.Intersect(otherkeys); //keys both share
                    var missingkeys = otherkeys.Except(mainkeys); //keys only otherdict has

                    string current_key = null; //for exception handling
                    try {
                        //handle merging of shared keys
                        foreach (var key in sharedkeys) {
                            var leftkey = current_key = (string) key;
                            try {
                                left.Set(leftkey, merging_core(left.Get(leftkey), rightmergable.Get(leftkey), preference)); //try to merge both values before returning.
                            } catch (KeyNotFoundException) { } //has no setter.
                        }

                        //handle settings of missing keys in sharedkeys
                        foreach (var o_key in missingkeys) {
                            var key = current_key = (string) o_key;
                            try {
                                left.Set(key, _trycopy(rightmergable.Get(current_key))); //try to merge both values before returning.
                            } catch (KeyNotFoundException) { } //has no setter.
                        }
                    } catch (Exception e) {
                        throw new DictMergingException($"Merging failed during merge of key '{current_key}', see inner exception.", e);
                    }
                }

                return left;
            }

            return preference == MergePrefer.Old ? left : right;
        }

        /// <summary>
        ///     A deep copy of a value the right side alone has: an <see cref="IExpandableList"/> expanded, lists and
        ///     dictionaries copied element by element (recursively), anything else returned as it is.
        /// </summary>
        /// <param name="o">The value to copy.</param>
        /// <returns>The copy, or <paramref name="o"/> itself when it is not a collection.</returns>
        /// <exception cref="MissingMethodException">Thrown for a list or dictionary type without a parameterless constructor.</exception>
        private static object _trycopy(object o) {
            switch (o) {
                case IExpandableList e:
                    return e.Expand();
                case IList l:
                    return _copy(l);
                case IDictionary d:
                    return _copy(d);
            }

            return o;
        }

        /// <summary>A copy of a list (an array stays an array), each element copied with <see cref="_trycopy"/>.</summary>
        /// <param name="src">The list to copy.</param>
        /// <returns>The copy, or <see langword="null"/> for a <see langword="null"/> list.</returns>
        /// <exception cref="MissingMethodException">Thrown when the list type has no parameterless constructor.</exception>
        private static IList _copy(IList src) {
            if (src == null)
                return null;
            var type = src.GetType();
            IList ret;

            if (type.IsArray) {
                ret = Array.CreateInstance(type.GetElementType(), src.Count);
                for (int i = 0; i < ret.Count; i++) {
                    ret[i] = _trycopy(src[i]);
                }
            } else {
                ret = (IList) Activator.CreateInstance(type);
                foreach (var o in src)
                    ret.Add(_trycopy(o));
            }

            return ret;
        }

        /// <summary>A copy of a dictionary of the same type, each value copied with <see cref="_trycopy"/> (keys shared).</summary>
        /// <param name="src">The dictionary to copy.</param>
        /// <returns>The copy, or <see langword="null"/> for a <see langword="null"/> dictionary.</returns>
        /// <exception cref="MissingMethodException">Thrown when the dictionary type has no parameterless constructor.</exception>
        private static IDictionary _copy(IDictionary src) {
            if (src == null)
                return null;
            var ret = (IDictionary) Activator.CreateInstance(src.GetType());
            foreach (DictionaryEntry o in src)
                ret.Add(o.Key, _trycopy(o.Value));

            return ret;
        }
    }
}
