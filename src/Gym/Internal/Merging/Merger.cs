using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Gym.Collections;
using Gym.Reflection;

namespace Gym.Merging {
    /// <summary>
    ///     A utitlity to 
    /// </summary>
    public static class Merger {
        /// <summary>
        ///     Merges <paramref name="right"/> with <see cref="left"/> object, appending all lists and merging dictionaries.
        /// </summary>
        /// <param name="left">The object that will accept all merges.</param>
        /// <param name="right">The dictionary to take data from.</param>
        /// <param name="preference">See the enum values docs.</param>
        public static void Merge(IMergable left, IMergable right, MergePrefer preference) {
            merging_core(left, right, preference);
        }

        /// <summary>
        ///     Merges <paramref name="right"/> with current dictionary, appending all lists and merging dictionaries.
        /// </summary>
        /// <param name="left">The object that will accept all merges.</param>
        /// <param name="right">The dictionary to take data from.</param>
        /// <param name="preference">See the enum values docs.</param>
        public static void Merge(IMergable left, object right, MergePrefer preference) {
            merging_core(left, right, preference);
        }


        /// <summary>
        ///     Merges <paramref name="right"/> with <see cref="left"/> dictionary, appending all lists and merging dictionaries.
        /// </summary>
        /// <param name="left">The object that will accept all merges.</param>
        /// <param name="right">The dictionary to take data from.</param>
        /// <param name="prefer">See the enum values docs.</param>
        public static void Merge(IDictionary left, IDictionary right, MergePrefer prefer = MergePrefer.Old) {
            if (right == null || right.Count == 0)
                return;
            merging_core(left, right, prefer);
        }

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