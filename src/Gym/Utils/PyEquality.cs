#nullable enable
using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp;

namespace Gym.Utils {
    /// <summary>
    ///     Python's <c>==</c> between the Python counterparts of .NET values (see <see cref="PyBuiltins"/> for the mapping),
    ///     which is how Gymnasium compares <c>info</c> dicts, metadata, spec kwargs and the values inside them: numbers of any
    ///     width compare by value (<c>1 == 1.0 == True</c>), dicts ignore insertion order, lists and tuples compare element by
    ///     element and never equal each other.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Arrays are the one deliberate deviation. In Python, <c>==</c> between arrays is element-wise and a dict comparison
    ///     then asks for the truth of the result, which raises <c>ValueError</c> for arrays of more than one element. Here two
    ///     arrays are equal when <see cref="np.array_equal(NDArray, NDArray)"/> says so (same shape, equal elements, any
    ///     dtypes; NaN never equals NaN), so an <see cref="Gym.Envs.Info"/> holding arrays can still be compared. A 0-d array
    ///     is the NumPy scalar it stands for and compares as a number; a one-element array compared with a number compares
    ///     its element, as Python's truth test of a one-element array does.
    ///     </para>
    ///     <para>
    ///     Identity short-circuits, as in Python's container comparisons (<c>PyObject_RichCompareBool</c>): the same object
    ///     is equal to itself even when it is, or contains, a NaN.
    ///     </para>
    /// </remarks>
    public static class PyEquality {
        /// <summary>Python's <c>left == right</c> for the values' Python counterparts, with the array rule of the class remarks.</summary>
        /// <param name="left">The first value; <see langword="null"/> is <c>None</c>.</param>
        /// <param name="right">The second value; <see langword="null"/> is <c>None</c>.</param>
        /// <returns><see langword="true"/> when Python would consider the values equal.</returns>
        /// <remarks>
        ///     A container that holds itself recurses without end; Python detects that and raises <c>RecursionError</c>,
        ///     which no Gymnasium comparison relies on.
        /// </remarks>
        public static bool AreEqual(object? left, object? right) {
            if (ReferenceEquals(left, right))
                return true;
            if (left is null || right is null)
                return false;

            // Numbers first: bool, every integer width, BigInteger, the floats, and 0-d numeric arrays (NumPy scalars).
            if (TryNumber(left, out Number a) && TryNumber(right, out Number b))
                return NumbersEqual(a, b);

            switch (left) {
                case string text:
                    return right is string other && string.Equals(text, other, StringComparison.Ordinal);
                case NDArray array:
                    return ArrayEquals(array, right);
            }

            if (right is NDArray rightArray)
                return ArrayEquals(rightArray, left);

            if (TryEntries(left, out List<KeyValuePair<object?, object?>>? leftEntries)) {
                return TryEntries(right, out List<KeyValuePair<object?, object?>>? rightEntries) && DictsEqual(leftEntries!, rightEntries!);
            }

            if (left is ITuple leftTuple) {
                if (right is not ITuple rightTuple || leftTuple.Length != rightTuple.Length)
                    return false;
                for (int i = 0; i < leftTuple.Length; i++) {
                    if (!AreEqual(leftTuple[i], rightTuple[i]))
                        return false;
                }

                return true;
            }

            if (left is IList leftList) {
                // A tuple never equals a list in Python, and a string is not a list of characters here.
                if (right is not IList rightList || right is ITuple || leftList.Count != rightList.Count)
                    return false;
                for (int i = 0; i < leftList.Count; i++) {
                    if (!AreEqual(leftList[i], rightList[i]))
                        return false;
                }

                return true;
            }

            return left.Equals(right);
        }

        /// <summary>
        ///     Compares an array with any value: another array through <see cref="np.array_equal(NDArray, NDArray)"/>, a
        ///     number through the array's only element (the truth of Python's element-wise result for one element).
        /// </summary>
        /// <param name="array">The array, not 0-d (0-d arrays are numbers).</param>
        /// <param name="other">The other value, not <see langword="null"/>.</param>
        /// <returns>Whether the two are equal under the class remarks' array rule.</returns>
        private static bool ArrayEquals(NDArray array, object other) {
            if (other is NDArray otherArray) {
                // np.array_equal already requires equal shapes; comparing first keeps it from broadcasting a 0-d array.
                if (!ShapesEqual(array.shape, otherArray.shape))
                    return false;
                return np.array_equal(array, otherArray);
            }

            // GetAtIndex reads the only element of any one-element view without allocating another array.
            if (array.size == 1 && TryNumber(other, out Number number))
                return TryNumber(array.GetAtIndex(0), out Number single) && NumbersEqual(single, number);

            return false;
        }

        /// <summary>Whether two shapes are identical dimension for dimension.</summary>
        /// <param name="left">The first shape.</param>
        /// <param name="right">The second shape.</param>
        /// <returns><see langword="true"/> when both have the same rank and extents.</returns>
        private static bool ShapesEqual(long[] left, long[] right) {
            if (left.Length != right.Length)
                return false;
            for (int i = 0; i < left.Length; i++) {
                if (left[i] != right[i])
                    return false;
            }

            return true;
        }

        /// <summary>
        ///     Python's dict equality: the same number of keys, and every key of one found in the other (keys compare with
        ///     <see cref="AreEqual"/>, so <c>1</c> and <c>1.0</c> are the same key, as their hashes are in Python) with an equal
        ///     value. Insertion order does not matter.
        /// </summary>
        /// <param name="left">The first dict's entries.</param>
        /// <param name="right">The second dict's entries.</param>
        /// <returns><see langword="true"/> when the dicts are equal.</returns>
        private static bool DictsEqual(List<KeyValuePair<object?, object?>> left, List<KeyValuePair<object?, object?>> right) {
            if (left.Count != right.Count)
                return false;

            // String keys (every Gymnasium info and metadata key) are looked up by ordinal equality in one pass; any other
            // key falls back to a scan, which is quadratic but only for exotic keys.
            Dictionary<string, object?>? byString = null;
            foreach (KeyValuePair<object?, object?> entry in right) {
                if (entry.Key is string key) {
                    byString ??= new Dictionary<string, object?>(StringComparer.Ordinal);
                    byString[key] = entry.Value;
                }
            }

            foreach (KeyValuePair<object?, object?> entry in left) {
                if (entry.Key is string key) {
                    if (byString is null || !byString.TryGetValue(key, out object? value) || !AreEqual(entry.Value, value))
                        return false;
                    continue;
                }

                bool found = false;
                foreach (KeyValuePair<object?, object?> candidate in right) {
                    if (candidate.Key is not string && AreEqual(entry.Key, candidate.Key)) {
                        if (!AreEqual(entry.Value, candidate.Value))
                            return false;
                        found = true;
                        break;
                    }
                }

                if (!found)
                    return false;
            }

            return true;
        }

        /// <summary>
        ///     The entries of a value that stands for a Python <c>dict</c>: a non-generic <see cref="IDictionary"/> (every
        ///     <see cref="Dictionary{TKey, TValue}"/>) or a sequence of string-keyed pairs (<see cref="Gym.Envs.Info"/>).
        /// </summary>
        /// <param name="value">The value.</param>
        /// <param name="entries">The entries in iteration order, when <see langword="true"/> is returned.</param>
        /// <returns><see langword="true"/> when the value is dict-like.</returns>
        private static bool TryEntries(object value, out List<KeyValuePair<object?, object?>>? entries) {
            switch (value) {
                case IDictionary dictionary:
                    entries = new List<KeyValuePair<object?, object?>>(dictionary.Count);
                    foreach (DictionaryEntry entry in dictionary)
                        entries.Add(new KeyValuePair<object?, object?>(entry.Key, entry.Value));
                    return true;
                case IEnumerable<KeyValuePair<string, object?>> pairs:
                    entries = new List<KeyValuePair<object?, object?>>();
                    foreach (KeyValuePair<string, object?> pair in pairs)
                        entries.Add(new KeyValuePair<object?, object?>(pair.Key, pair.Value));
                    return true;
                default:
                    entries = null;
                    return false;
            }
        }

        /// <summary>A number reduced to what Python's comparison needs: an exact integer or a double.</summary>
        /// <param name="IsInteger">Whether the number is an integer (Python <c>int</c>/<c>bool</c>, a NumPy integer scalar).</param>
        /// <param name="Integer">The value when <paramref name="IsInteger"/> is <see langword="true"/>.</param>
        /// <param name="Real">The value when <paramref name="IsInteger"/> is <see langword="false"/>.</param>
        private readonly record struct Number(bool IsInteger, BigInteger Integer, double Real);

        /// <summary>
        ///     Reads a value as a Python number: <see langword="bool"/> (0 or 1), every .NET integer, <see cref="BigInteger"/>,
        ///     an enum member (its value, as Gym.NET treats enums as ints), the floats (a <see langword="float"/> widened
        ///     exactly, as NumPy compares a float32 scalar with a Python float), <see langword="decimal"/>, and a 0-d array of a
        ///     boolean, integer or floating dtype (the NumPy scalar it stands for).
        /// </summary>
        /// <param name="value">The value.</param>
        /// <param name="number">The number, when <see langword="true"/> is returned.</param>
        /// <returns><see langword="true"/> when the value is a number.</returns>
        private static bool TryNumber(object value, out Number number) {
            switch (value) {
                case bool flag:
                    number = new Number(true, flag ? BigInteger.One : BigInteger.Zero, 0);
                    return true;
                case sbyte or byte or short or ushort or int or uint or long:
                    number = new Number(true, Convert.ToInt64(value, System.Globalization.CultureInfo.InvariantCulture), 0);
                    return true;
                case ulong u:
                    number = new Number(true, u, 0);
                    return true;
                case BigInteger big:
                    number = new Number(true, big, 0);
                    return true;
                case Enum e:
                    return TryNumber(Convert.ChangeType(e, Enum.GetUnderlyingType(e.GetType()), System.Globalization.CultureInfo.InvariantCulture), out number);
                case double d:
                    number = new Number(false, BigInteger.Zero, d);
                    return true;
                case float f:
                    number = new Number(false, BigInteger.Zero, f);
                    return true;
                case Half h:
                    number = new Number(false, BigInteger.Zero, (double)h);
                    return true;
                case decimal m:
                    // A decimal is exact; an integral one compares as an integer, anything else through its nearest double.
                    if (decimal.Truncate(m) == m) {
                        number = new Number(true, new BigInteger(m), 0);
                        return true;
                    }

                    number = new Number(false, BigInteger.Zero, (double)m);
                    return true;
                case NDArray array when array.ndim == 0 && array.dtype.kind is 'b' or 'i' or 'u' or 'f':
                    return TryNumber(array.GetAtIndex(0), out number);
                default:
                    number = default;
                    return false;
            }
        }

        /// <summary>
        ///     Python's exact numeric comparison: integers by value, floats by IEEE equality (so <c>-0.0 == 0.0</c> and NaN is
        ///     never equal), and an integer with a float only when the float is finite, integral and the same value (Python
        ///     never rounds the integer: <c>2**53 + 1 != 2.0**53</c>).
        /// </summary>
        /// <param name="a">The first number.</param>
        /// <param name="b">The second number.</param>
        /// <returns><see langword="true"/> when the numbers are equal.</returns>
        private static bool NumbersEqual(Number a, Number b) {
            if (a.IsInteger && b.IsInteger)
                return a.Integer == b.Integer;
            if (!a.IsInteger && !b.IsInteger)
                return a.Real == b.Real;
            double real = a.IsInteger ? b.Real : a.Real;
            BigInteger integer = a.IsInteger ? a.Integer : b.Integer;
            if (double.IsNaN(real) || double.IsInfinity(real) || Math.Floor(real) != real)
                return false;
            return new BigInteger(real) == integer;
        }
    }
}
