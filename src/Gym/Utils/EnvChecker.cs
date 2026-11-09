#nullable enable
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using Gym.Spaces;
using NumSharp;

namespace Gym.Utils {
    /// <summary>
    ///     Gymnasium's <c>gymnasium.utils.env_checker</c>: the checks an env must pass to follow the API. Today it holds
    ///     <see cref="DataEquivalence"/>, which tests and checkers use to compare observations, actions and infos; the
    ///     env checks themselves arrive with the registry (P3).
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/env_checker.py">gymnasium/utils/env_checker.py</see>.</remarks>
    public static class EnvChecker {
        /// <summary>
        ///     Whether two values are equivalent (Gymnasium's <c>data_equivalence</c>): of the same Python type, dicts with the
        ///     same keys and equivalent values, tuples and lists of equivalent items, arrays of the same shape and dtype with
        ///     close (or, when <paramref name="exact"/>, equal) entries, and anything else equal by Python's <c>==</c>.
        /// </summary>
        /// <param name="data1">The first value.</param>
        /// <param name="data2">The second value.</param>
        /// <param name="exact">Whether arrays must be equal entry for entry rather than within <c>np.allclose(rtol=1e-5, atol=1e-5)</c>.</param>
        /// <returns><see langword="true"/> when equivalent.</returns>
        /// <remarks>
        ///     <para>
        ///     The type test is on the values' Python counterparts (<see cref="PyBuiltins"/>): every .NET integer is an
        ///     <c>int</c>, an <see langword="object"/>?[] or <see cref="ITuple"/> a <c>tuple</c>, a typed array or list a
        ///     <c>list</c>, a 0-d array the NumPy scalar it stands for (DR-08), so <c>1</c> and <c>1L</c> are equivalent but a
        ///     tuple never matches a list, as in Gymnasium.
        ///     </para>
        ///     <para>
        ///     A 0-d array compares with <c>==</c>, exactly, as Gymnasium compares NumPy scalars (only arrays of one or more
        ///     dimensions go through <c>allclose</c>); a NaN therefore never equals itself.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/env_checker.py#L36">gymnasium/utils/env_checker.py (data_equivalence)</see>.
        ///     </para>
        /// </remarks>
        public static bool DataEquivalence(object? data1, object? data2, bool exact = false) {
            if (PyBuiltins.TypeName(data1) != PyBuiltins.TypeName(data2))
                return false;
            if (Entries(data1) is { } left) {
                Dictionary<string, object?> right = Entries(data2)!;
                // dict_keys compare as sets: the same keys in any order.
                return left.Count == right.Count && left.Keys.All(right.ContainsKey) && left.All(pair => DataEquivalence(pair.Value, right[pair.Key], exact));
            }

            if (Items(data1) is { } leftItems) {
                IReadOnlyList<object?> rightItems = Items(data2)!;
                return leftItems.Count == rightItems.Count && leftItems.Zip(rightItems).All(pair => DataEquivalence(pair.First, pair.Second, exact));
            }

            if (data1 is NDArray first && first.ndim > 0) {
                var second = (NDArray)data2!;
                if (!first.Shape.dimensions.SequenceEqual(second.Shape.dimensions) || first.dtype != second.dtype)
                    return false;
                if (exact) {
                    using NDArray equal = first == second;
                    return np.all(equal);
                }

                return np.allclose(first, second, rtol: 1e-5, atol: 1e-5);
            }

            // NumPy scalars (0-d arrays), numbers, strings, None: Python's ==, where NaN equals nothing.
            if (data1 is NDArray scalar1 && data2 is NDArray scalar2) {
                using NDArray equal = scalar1 == scalar2;
                return equal.GetBoolean();
            }

            return PyEquality.AreEqual(data1, data2) && !(data1 is double d && double.IsNaN(d)) && !(data1 is float f && float.IsNaN(f));
        }

        /// <summary>A dict's entries by key, for <see cref="DataEquivalence"/>; <see langword="null"/> for anything that is not a dict.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The entries, or <see langword="null"/>.</returns>
        private static Dictionary<string, object?>? Entries(object? value) {
            switch (value) {
                case IEnumerable<KeyValuePair<string, object?>> pairs:
                    return pairs.ToDictionary(pair => pair.Key, pair => pair.Value);
                case IDictionary dictionary: {
                    var entries = new Dictionary<string, object?>();
                    foreach (DictionaryEntry entry in dictionary)
                        entries[Convert.ToString(entry.Key, System.Globalization.CultureInfo.InvariantCulture)!] = entry.Value;
                    return entries;
                }
                default:
                    return null;
            }
        }

        /// <summary>A tuple's or list's items, for <see cref="DataEquivalence"/>; <see langword="null"/> for anything else.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The items, or <see langword="null"/>.</returns>
        private static IReadOnlyList<object?>? Items(object? value) {
            switch (value) {
                case string:
                case NDArray:
                    return null;
                case ITuple tuple: {
                    // GraphInstance too: a NamedTuple is a tuple to isinstance(data_1, (tuple, list)).
                    var items = new object?[tuple.Length];
                    for (int i = 0; i < items.Length; i++)
                        items[i] = tuple[i];
                    return items;
                }
                case IList list:
                    return list.Cast<object?>().ToArray();
                default:
                    return null;
            }
        }
    }
}
