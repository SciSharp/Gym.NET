using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace Gym.Tests.Parity.Oracle {
    /// <summary>
    ///     Plain CLR value to Python value marshaling for the <see cref="Python"/> tree's arguments: the suite's counterpart
    ///     of OptunaSharp's <c>PyConvert</c>. The pythonic skill keeps this generic, run-time-typed conversion out of the tree,
    ///     which only holds fixed calls.
    /// </summary>
    /// <remarks>
    ///     Every member runs under the GIL and returns a new reference the caller disposes. Arrays cross through NumSharp's
    ///     interop (<c>ToNumpyCopy</c>, an independent NumPy copy with the same dtype and bytes), never element by element;
    ///     tuples are built only for small fixed structures such as shapes and colours.
    /// </remarks>
    internal static class PyConvert {
        /// <summary>
        ///     Converts a CLR value to the Python value a Gymnasium user would pass: <see langword="null"/> is <c>None</c>,
        ///     booleans, integers (any width, <see cref="BigInteger"/> included), floating-point numbers and strings map to
        ///     their Python types, an <see cref="NDArray"/> becomes a NumPy copy, and containers convert recursively: a
        ///     string-keyed dictionary (Gym.NET's <c>Info</c> included) to a <c>dict</c> in its order, an
        ///     <see cref="ITuple"/> to a <c>tuple</c>, any other list or array to a <c>list</c>.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>A new owned reference.</returns>
        /// <exception cref="ArgumentException">Thrown for a type with no defined conversion, so a test never passes something by accident.</exception>
        internal static PyObject ToPython(object? value) {
            switch (value) {
                case null:
                    // A new reference to None. PyObject.None is pythonnet's shared instance: the caller's using would
                    // dispose it for the whole process.
                    return PythonEngine.Eval("None");
                case bool flag:
                    return flag.ToPython();
                case sbyte or byte or short or ushort or int or uint or long:
                    return new PyInt(Convert.ToInt64(value, CultureInfo.InvariantCulture));
                case ulong unsigned:
                    return new PyInt(unsigned);
                case BigInteger big:
                    return new PyInt(big);
                case double d:
                    return new PyFloat(d);
                case float f:
                    return new PyFloat(f);
                case string text:
                    return new PyString(text);
                case NDArray array:
                    return array.ToNumpyCopy();
                case IEnumerable<KeyValuePair<string, object?>> entries:
                    return ToDict(entries);
                case ITuple tuple: {
                    var items = new List<object?>(tuple.Length);
                    for (int i = 0; i < tuple.Length; i++)
                        items.Add(tuple[i]);
                    return ToTupleOf(items);
                }
                case IList list:
                    return ToList(list.Cast<object?>());
                default:
                    throw new ArgumentException($"No Python conversion is defined for {value.GetType().FullName}.", nameof(value));
            }
        }

        /// <summary>Builds keyword arguments (or an options dictionary) from CLR values converted by <see cref="ToPython"/>.</summary>
        /// <param name="values">The entries in Python spelling, or <see langword="null"/> for none.</param>
        /// <returns>A new dictionary the caller disposes.</returns>
        /// <exception cref="ArgumentException">Thrown for a value with no defined conversion.</exception>
        internal static PyDict ToKwargs(IReadOnlyDictionary<string, object?>? values) {
            var dictionary = new PyDict();
            if (values is null)
                return dictionary;
            try {
                foreach ((string name, object? value) in values) {
                    using PyObject converted = ToPython(value);
                    dictionary[name] = converted;
                }
            } catch {
                dictionary.Dispose();
                throw;
            }

            return dictionary;
        }

        /// <summary>
        ///     A Python list of values converted by <see cref="ToPython"/>: <c>[0.0, 0.5, -1.0]</c>, for the few calls whose
        ///     point is that the argument is a list rather than an array (<c>Box.contains</c> converts a list itself). Arrays
        ///     still cross as arrays; this is not a way to move array data.
        /// </summary>
        /// <param name="items">The items, in order.</param>
        /// <returns>A new owned list.</returns>
        /// <exception cref="ArgumentException">Thrown for an item with no defined conversion.</exception>
        internal static PyObject ToList(IEnumerable<object?> items) {
            var list = new PyList();
            try {
                foreach (object? item in items) {
                    using PyObject converted = ToPython(item);
                    // Append takes its own reference, so the converted item is released by the using either way.
                    list.Append(converted);
                }
            } catch {
                list.Dispose();
                throw;
            }

            return list;
        }

        /// <summary>A Python <c>dict</c> of converted values, keys in the entries' order (Python dicts keep insertion order).</summary>
        /// <param name="entries">The entries.</param>
        /// <returns>A new owned reference.</returns>
        /// <exception cref="ArgumentException">Thrown for a value with no defined conversion.</exception>
        internal static PyObject ToDict(IEnumerable<KeyValuePair<string, object?>> entries) {
            var dictionary = new PyDict();
            try {
                foreach ((string key, object? value) in entries) {
                    using PyObject converted = ToPython(value);
                    // The dict takes its own reference to the value.
                    dictionary[key] = converted;
                }
            } catch {
                dictionary.Dispose();
                throw;
            }

            return dictionary;
        }

        /// <summary>A Python <c>tuple</c> of converted items, for tuples the suite builds from arbitrary values (a step's 5-tuple).</summary>
        /// <param name="items">The items.</param>
        /// <returns>A new owned reference.</returns>
        /// <exception cref="ArgumentException">Thrown for an item with no defined conversion.</exception>
        internal static PyObject ToTupleOf(IReadOnlyList<object?> items) {
            var converted = new PyObject[items.Count];
            try {
                for (int i = 0; i < converted.Length; i++)
                    converted[i] = ToPython(items[i]);
                return new PyTuple(converted);
            } finally {
                // PyTuple takes its own references to the items.
                foreach (PyObject? item in converted)
                    item?.Dispose();
            }
        }

        /// <summary>A tuple of Python ints: a shape (<c>(400, 600)</c>), a colour (<c>(202, 152, 101)</c>).</summary>
        /// <param name="values">The items.</param>
        /// <returns>A new owned tuple.</returns>
        internal static PyObject ToTuple(IReadOnlyList<long> values) {
            var items = new PyObject[values.Count];
            try {
                for (int i = 0; i < items.Length; i++)
                    items[i] = new PyInt(values[i]);
                return new PyTuple(items);
            } finally {
                // PyTuple takes its own references to the items.
                foreach (PyObject? item in items)
                    item?.Dispose();
            }
        }
    }
}
