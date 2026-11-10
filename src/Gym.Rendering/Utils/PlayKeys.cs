using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.ExceptionServices;
using NumSharp;

namespace Gym.Utils {
    /// <summary>
    ///     The Python semantics <see cref="Play"/> and <see cref="PlayableGame"/> apply to <c>keys_to_action</c> and to key
    ///     events: what counts as an <c>int</c>, a <c>str</c>, a <c>tuple</c> and a <c>dict</c>, <c>ord</c>, <c>sorted</c>,
    ///     and set and dict membership with Python's hashing (<c>97 == 97.0 == np.int64(97)</c>).
    /// </summary>
    internal static class PlayKeys {
        /// <summary>Python's <c>isinstance(value, int)</c>: <see langword="bool"/>, every .NET integer width, and <see cref="BigInteger"/>; not a NumPy scalar (a 0-d array), as <c>np.int64</c> is no <c>int</c>.</summary>
        /// <param name="value">The value.</param>
        /// <returns><see langword="true"/> for a Python int.</returns>
        public static bool IsInt(object? value) => value is bool or sbyte or byte or short or ushort or int or uint or long or ulong or BigInteger;

        /// <summary>Python's <c>isinstance(value, str)</c>.</summary>
        /// <param name="value">The value.</param>
        /// <returns><see langword="true"/> for a string.</returns>
        public static bool IsStr(object? value) => value is string;

        /// <summary>
        ///     The items of a Python tuple: an <see cref="ITuple"/> (a value tuple such as <c>(275, 276)</c>,
        ///     <c>ValueTuple.Create(97)</c>, the empty <c>new ValueTuple()</c>), or Gym.NET's generic tuple, an exact
        ///     <see langword="object"/>[] (DR-13).
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>The items, or <see langword="null"/> when the value is no tuple.</returns>
        public static IReadOnlyList<object?>? TupleItems(object? value) {
            switch (value) {
                case ITuple tuple: {
                    var items = new object?[tuple.Length];
                    for (int i = 0; i < items.Length; i++)
                        items[i] = tuple[i];
                    return items;
                }
                case object?[] array when array.GetType() == typeof(object[]):
                    return array;
                default:
                    return null;
            }
        }

        /// <summary>
        ///     Python's <c>list(value)</c> of a key: a tuple's items, a string's characters (one-character strings), any other
        ///     sequence's items.
        /// </summary>
        /// <param name="value">The key.</param>
        /// <returns>The items.</returns>
        /// <exception cref="TypeError">Thrown for anything not iterable, with Python's message: <c>'int' object is not iterable</c>.</exception>
        public static IReadOnlyList<object?> Iterate(object? value) {
            if (TupleItems(value) is { } items)
                return items;
            switch (value) {
                case string text:
                    return CodePoints(text);
                case IEnumerable sequence and not NDArray:
                    return sequence.Cast<object?>().ToList();
                case NDArray { ndim: > 0 } array:
                    // Iterating an array yields its rows (or its elements as NumPy scalars, 0-d arrays here).
                    return Enumerable.Range(0, (int)array.shape[0]).Select(i => (object?)array[i]).ToList();
                default:
                    throw new TypeError($"'{PyBuiltins.TypeName(value)}' object is not iterable");
            }
        }

        /// <summary>A string's Python characters: one string per code point, a surrogate pair kept together.</summary>
        /// <param name="text">The string.</param>
        /// <returns>The characters.</returns>
        public static IReadOnlyList<object?> CodePoints(string text) {
            var characters = new List<object?>(text.Length);
            for (int i = 0; i < text.Length; i++) {
                bool pair = char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]);
                characters.Add(text.Substring(i, pair ? 2 : 1));
                if (pair)
                    i++;
            }

            return characters;
        }

        /// <summary>Python's <c>ord(character)</c>: the code point of a one-character string.</summary>
        /// <param name="character">The string.</param>
        /// <returns>The code point.</returns>
        /// <exception cref="TypeError">
        ///     Thrown for a string of another length (in code points), with Python's message: <c>ord() expected a character, but
        ///     string of length 2 found</c>.
        /// </exception>
        public static int Ord(string character) {
            IReadOnlyList<object?> characters = CodePoints(character);
            if (characters.Count != 1)
                throw new TypeError($"ord() expected a character, but string of length {characters.Count} found");
            return char.ConvertToUtf32((string)characters[0]!, 0);
        }

        /// <summary>
        ///     Python's <c>isinstance(value, dict)</c> and <c>value.items()</c>: the entries of any .NET dictionary (a non-generic
        ///     <see cref="IDictionary"/>, or any enumeration of <see cref="KeyValuePair{TKey, TValue}"/>, as Gym.NET's
        ///     <c>Info</c> and every read-only dictionary are), in enumeration order.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>The entries, or <see langword="null"/> when the value is no dictionary.</returns>
        public static List<KeyValuePair<object?, object?>>? DictEntries(object? value) {
            switch (value) {
                case null or string or NDArray:
                    return null;
                case IDictionary dictionary: {
                    var entries = new List<KeyValuePair<object?, object?>>();
                    foreach (DictionaryEntry entry in dictionary)
                        entries.Add(new KeyValuePair<object?, object?>(entry.Key, entry.Value));
                    return entries;
                }
            }

            // A generic dictionary that is not an IDictionary (a read-only map): read its KeyValuePair<K, V> items by reflection.
            Type? pairType = value.GetType().GetInterfaces()
                                  .Where(i => i.IsGenericType && i.GetGenericTypeDefinition() == typeof(IEnumerable<>))
                                  .Select(i => i.GetGenericArguments()[0])
                                  .FirstOrDefault(t => t.IsGenericType && t.GetGenericTypeDefinition() == typeof(KeyValuePair<,>));
            if (pairType is null)
                return null;
            PropertyInfo key = pairType.GetProperty("Key")!, item = pairType.GetProperty("Value")!;
            var pairs = new List<KeyValuePair<object?, object?>>();
            foreach (object? pair in (IEnumerable)value)
                pairs.Add(new KeyValuePair<object?, object?>(key.GetValue(pair), item.GetValue(pair)));
            return pairs;
        }

        /// <summary>
        ///     Calls an attribute's value with no arguments, as Python's <c>getattr(...)()</c> does:
        ///     <c>get_wrapper_attr("get_keys_to_action")()</c>.
        /// </summary>
        /// <param name="value">The attribute's value, a delegate.</param>
        /// <returns>What the call returned.</returns>
        /// <exception cref="TypeError">
        ///     Thrown when the value is not callable (<c>'int' object is not callable</c>) or needs arguments; whatever the call
        ///     itself raises propagates unwrapped.
        /// </exception>
        public static object? Call(object? value) {
            if (value is not Delegate function)
                throw new TypeError($"'{PyBuiltins.TypeName(value)}' object is not callable");
            if (function.Method.GetParameters().Length != 0)
                throw new TypeError($"{function.Method.Name}() missing {function.Method.GetParameters().Length} required positional argument(s)");
            try {
                return function.DynamicInvoke();
            } catch (TargetInvocationException e) when (e.InnerException is not null) {
                // Python sees the function's own exception, not reflection's wrapper.
                ExceptionDispatchInfo.Capture(e.InnerException).Throw();
                throw;
            }
        }

        /// <summary>
        ///     Python's <c>sorted(values)</c> for key codes: numbers (ints, bools, floats, NumPy scalars) by value and strings by
        ///     code point, stably, so equal values keep their order (<c>sorted([True, 1])</c> is <c>[True, 1]</c>).
        /// </summary>
        /// <param name="values">The values.</param>
        /// <returns>A new sorted list.</returns>
        /// <exception cref="TypeError">
        ///     Thrown when numbers and non-numbers meet, or other kinds that Python cannot order, with Python's message:
        ///     <c>'&lt;' not supported between instances of 'str' and 'int'</c>.
        /// </exception>
        public static List<object?> Sorted(IEnumerable<object?> values) {
            List<object?> items = values.ToList();
            if (items.All(item => Number(item) is not null))
                return items.OrderBy(item => Number(item)!.Value, NumberComparer.Instance).ToList();
            if (items.All(item => item is string))
                return items.OrderBy(item => (string)item!, StringComparer.Ordinal).ToList();

            // Mixed kinds: Python's sort fails at the first pair it cannot order, comparing the later item with the earlier
            // one ("'<' not supported between instances of 'str' and 'int'" for [97, 'a']). Which pair it meets first
            // depends on the sort's merges; the first item and the first of another kind stand for it. Kinds that cannot
            // be key codes at all (lists, tuples) are named against themselves.
            int firstOther = items.FindIndex(item => Kind(item) != Kind(items[0]));
            object? later = firstOther < 0 ? items[0] : items[firstOther];
            throw new TypeError($"'<' not supported between instances of '{PyBuiltins.TypeName(later)}' and '{PyBuiltins.TypeName(items[0])}'");
        }

        /// <summary>The ordering family of a value for <see cref="Sorted"/>: numbers together, strings together, anything else by its Python type.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The family's name.</returns>
        private static string Kind(object? value) => Number(value) is not null ? "number" : value is string ? "str" : PyBuiltins.TypeName(value);

        /// <summary>
        ///     A Python number's value for comparing and hashing: an integral value as its exact integer, any other as its
        ///     float. <see langword="null"/> for anything that is no number.
        /// </summary>
        /// <param name="value">A bool, a .NET integer or float, a <see cref="BigInteger"/>, or a 0-d numeric array (a NumPy scalar).</param>
        /// <returns>The value, or <see langword="null"/>.</returns>
        public static NumberValue? Number(object? value) {
            switch (value) {
                case bool flag:
                    return new NumberValue(flag ? BigInteger.One : BigInteger.Zero, null);
                case sbyte or byte or short or ushort or int or uint or long:
                    return new NumberValue(Convert.ToInt64(value, System.Globalization.CultureInfo.InvariantCulture), null);
                case ulong unsigned:
                    return new NumberValue(unsigned, null);
                case BigInteger integer:
                    return new NumberValue(integer, null);
                case double d:
                    return Float(d);
                case float f:
                    return Float(f);
                case Half h:
                    return Float((double)h);
                case NDArray { ndim: 0 } scalar:
                    // A NumPy scalar: NumPy's numbers compare and hash with Python's, by value.
                    switch (scalar.typecode) {
                        case NPTypeCode.Boolean:
                            return new NumberValue(scalar.GetAtIndex<bool>(0) ? BigInteger.One : BigInteger.Zero, null);
                        case NPTypeCode.SByte or NPTypeCode.Byte or NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64:
                            return new NumberValue(Convert.ToInt64(scalar.GetAtIndex(0), System.Globalization.CultureInfo.InvariantCulture), null);
                        case NPTypeCode.UInt64:
                            return new NumberValue(scalar.GetAtIndex<ulong>(0), null);
                        case NPTypeCode.Half:
                            return Float((double)scalar.GetAtIndex<Half>(0));
                        case NPTypeCode.Single:
                            return Float(scalar.GetAtIndex<float>(0));
                        case NPTypeCode.Double:
                            return Float(scalar.GetAtIndex<double>(0));
                        default:
                            // Decimal, char, string and complex scalars are no key codes; they compare by their own equality.
                            return null;
                    }
                default:
                    return null;
            }
        }

        /// <summary>A float's <see cref="NumberValue"/>: integral when it holds a whole finite number, so <c>97.0</c> hashes as <c>97</c>.</summary>
        /// <param name="value">The float.</param>
        /// <returns>The value.</returns>
        private static NumberValue Float(double value) {
            return double.IsFinite(value) && Math.Floor(value) == value ? new NumberValue(new BigInteger(value), null) : new NumberValue(null, value);
        }

        /// <summary>A Python number: exactly one of an integral value and a non-integral (or non-finite) float.</summary>
        /// <param name="Integer">The value when it is a whole number.</param>
        /// <param name="Fraction">The value when it is not (a fraction, an infinity, NaN).</param>
        internal readonly record struct NumberValue(BigInteger? Integer, double? Fraction) {
            /// <summary>The value as a float, for comparisons between an integer and a fraction.</summary>
            public double AsDouble => Integer is { } integer ? (double)integer : Fraction!.Value;
        }

        /// <summary>Orders <see cref="NumberValue"/>s as Python orders numbers: integers exactly, anything involving a float as floats.</summary>
        private sealed class NumberComparer : IComparer<NumberValue> {
            /// <summary>The shared instance; stateless.</summary>
            public static readonly NumberComparer Instance = new();

            /// <summary>Compares two numbers; NaN compares as neither less nor greater, as Python's <c>&lt;</c> answers false both ways.</summary>
            /// <param name="x">The first number.</param>
            /// <param name="y">The second number.</param>
            /// <returns>Negative, zero or positive.</returns>
            public int Compare(NumberValue x, NumberValue y) {
                if (x.Integer is { } a && y.Integer is { } b)
                    return a.CompareTo(b);
                double left = x.AsDouble, right = y.AsDouble;
                return left < right ? -1 : right < left ? 1 : 0;
            }
        }
    }

    /// <summary>
    ///     Python's hashing and equality for the keys of <c>keys_to_action</c> and the key sets of <see cref="PlayableGame"/>:
    ///     numbers equal by value across types (<c>97 == 97.0 == True + 96</c>), strings by content, tuples item by item
    ///     (a value tuple equals Gym.NET's <see langword="object"/>[] tuple), anything else by its own equality.
    /// </summary>
    internal sealed class PyKeyComparer : IEqualityComparer<object?> {
        /// <summary>The shared instance; stateless.</summary>
        public static readonly PyKeyComparer Instance = new();

        /// <summary>Python's <c>==</c> (<see cref="PyEquality.AreEqual"/>).</summary>
        /// <param name="x">The first value.</param>
        /// <param name="y">The second value.</param>
        /// <returns><see langword="true"/> when Python would find them equal.</returns>
        public new bool Equals(object? x, object? y) => PyEquality.AreEqual(x, y);

        /// <summary>A hash consistent with <see cref="Equals(object?, object?)"/>: equal numbers of any type hash alike.</summary>
        /// <param name="obj">The value.</param>
        /// <returns>The hash.</returns>
        /// <exception cref="TypeError">Thrown for a value Python cannot hash: <c>unhashable type: 'list'</c> (also <c>dict</c>, <c>set</c>, <c>numpy.ndarray</c>).</exception>
        public int GetHashCode(object? obj) {
            switch (obj) {
                case null:
                    return 0;
                case string text:
                    return StringComparer.Ordinal.GetHashCode(text);
                case NDArray { ndim: > 0 }:
                    throw new TypeError("unhashable type: 'numpy.ndarray'");
            }

            if (PlayKeys.Number(obj) is { } number)
                return number.Integer is { } integer ? integer.GetHashCode() : number.Fraction!.Value.GetHashCode();
            if (PlayKeys.TupleItems(obj) is { } items) {
                var hash = new HashCode();
                foreach (object? item in items)
                    hash.Add(GetHashCode(item));
                return hash.ToHashCode();
            }

            if (obj is IDictionary or IList or IEnumerable<KeyValuePair<string, object?>> || PyBuiltins.TypeName(obj) == "set")
                throw new TypeError($"unhashable type: '{PyBuiltins.TypeName(obj)}'");
            return obj.GetHashCode();
        }
    }
}
