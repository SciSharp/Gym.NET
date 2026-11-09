#nullable enable
using System;
using System.Numerics;
using Gym.Spaces;
using Gym.Utils;
using NumSharp;

namespace Gym.Environments.Envs.ToyText {
    /// <summary>
    ///     The two Python lookups toy_text actions go through: a dict keyed <c>0 .. n - 1</c> (<c>self.P[self.s][a]</c>), and a
    ///     list indexed by the last action (the <c>ansi</c> caption, <c>['Left', 'Down', 'Right', 'Up'][self.lastaction]</c>),
    ///     each with Python's acceptance rules and error texts.
    /// </summary>
    /// <remarks>
    ///     A dict finds a key by hash and equality, so every number equal to an integer key finds it: <c>True</c>,
    ///     <c>1.0</c>, <c>np.float32(1.0)</c>, <c>np.int8(1)</c>; anything else is <c>KeyError(repr(a))</c>, and a NumPy array
    ///     of one or more dimensions is unhashable. A 0-d NumSharp array is the NumPy scalar it stands for (hashable).
    /// </remarks>
    internal static class PyKeys {
        /// <summary>
        ///     <c>d[value]</c> for a dict whose keys are the Python ints <c>0 .. count - 1</c>: the key the value finds, with
        ///     Python's errors.
        /// </summary>
        /// <param name="value">The action as given.</param>
        /// <param name="count">How many keys the dict has.</param>
        /// <returns>The key found.</returns>
        /// <exception cref="TypeError">Thrown for an array of one or more dimensions (<c>unhashable type: 'numpy.ndarray'</c>).</exception>
        /// <exception cref="KeyError">Thrown when no key equals the value, with Python's message, the value's <c>repr</c>.</exception>
        public static long DictKey(object? value, int count) {
            if (value is NDArray { ndim: > 0 })
                throw new TypeError("unhashable type: 'numpy.ndarray'");
            // Integers too wide for a long: compared as the unbounded integers they are (none equals a small key).
            BigInteger? wide = value switch {
                BigInteger big => big,
                ulong unsigned => unsigned,
                NDArray { typecode: NPTypeCode.UInt64 } scalar => scalar.GetAtIndex<ulong>(0),
                _ => null,
            };
            if (wide is { } integer) {
                if (integer >= 0 && integer < count)
                    return (long)integer;
                throw new KeyError(PyBuiltins.Repr(value));
            }

            // Equality with an int key: integers (bool among them) by value, floats when they hold a whole number.
            double? number = NumberOf(value);
            if (number is { } whole && Math.Floor(whole) == whole && whole >= 0 && whole < count)
                return (long)whole;
            throw new KeyError(PyBuiltins.Repr(value));
        }

        /// <summary>
        ///     <c>items[value]</c> for a Python list of <paramref name="length"/> items: the position the value selects (a
        ///     negative one counts from the end), with Python's errors.
        /// </summary>
        /// <param name="value">The index as given.</param>
        /// <param name="length">The list's length.</param>
        /// <returns>The position, in <c>[0, length)</c>.</returns>
        /// <exception cref="TypeError">Thrown for a non-integer (<c>list indices must be integers or slices, not float</c>), or an array of one or more dimensions.</exception>
        /// <exception cref="IndexError">Thrown for an integer outside the list, or beyond 64 bits.</exception>
        public static int ListIndex(object? value, int length) {
            BigInteger index;
            switch (value) {
                case NDArray { ndim: > 0 }:
                    throw new TypeError("only integer scalar arrays can be converted to a scalar index");
                case NDArray scalar when scalar.typecode is NPTypeCode.UInt64:
                    index = scalar.GetAtIndex<ulong>(0);
                    break;
                case BigInteger big:
                    index = big;
                    break;
                case ulong unsigned:
                    index = unsigned;
                    break;
                default:
                    // Python ints (bool among them) and NumPy integer scalars have __index__; floats, NumPy's bool and the
                    // rest do not, and the list names their type.
                    if (!Discrete.TryToInt64(value, out long integer))
                        throw new TypeError($"list indices must be integers or slices, not {PyBuiltins.TypeName(value)}");
                    index = integer;
                    break;
            }

            if (index < long.MinValue || index > long.MaxValue)
                throw new IndexError($"cannot fit '{PyBuiltins.TypeName(value)}' into an index-sized integer");
            long position = (long)index;
            if (position < 0)
                position += length;
            if (position < 0 || position >= length)
                throw new IndexError("list index out of range");
            return (int)position;
        }

        /// <summary>The value of a number Python would compare with an int key, or <see langword="null"/> for anything else.</summary>
        /// <param name="value">The candidate.</param>
        /// <returns>The number as a double, exact for the integers a small dict's keys can equal.</returns>
        private static double? NumberOf(object? value) {
            switch (value) {
                case bool flag:
                    return flag ? 1 : 0;
                case double d:
                    return d;
                case float f:
                    return f;
                case Half h:
                    return (double)h;
                case NDArray array:
                    // A 0-d array is its NumPy scalar: np.bool_ hashes and compares like 0/1, floats by value.
                    return array.typecode switch {
                        NPTypeCode.Boolean => array.GetAtIndex<bool>(0) ? 1 : 0,
                        NPTypeCode.Single => array.GetAtIndex<float>(0),
                        NPTypeCode.Double => array.GetAtIndex<double>(0),
                        NPTypeCode.Half => (double)array.GetAtIndex<Half>(0),
                        _ => Discrete.TryToInt64(array, out long integer) ? integer : null,
                    };
                default:
                    return Discrete.TryToInt64(value, out long whole) ? whole : null;
            }
        }
    }
}
