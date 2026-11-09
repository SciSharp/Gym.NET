using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using Gym.Envs;
using Gym.Tests.Parity.Interop;
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

        /// <summary>
        ///     Converts a Gym.NET value to Python the way <see cref="ToPython"/> does, except that arrays cross as zero-copy
        ///     NumPy views (NumSharp's <c>ToNumpy</c>) instead of copies. This is what the reverse bridge hands Gymnasium: two
        ///     results that share memory in Gym.NET share it in Python too, which is exactly what Gymnasium's data-reuse checks
        ///     (<c>check_returned_data_not_reused</c>, <c>PassiveEnvChecker</c>) look for, so a copy here would hide the defect
        ///     they exist to catch. An <see langword="object"/>?[] crosses as a <c>tuple</c> (Gym.NET's tuple, DR-13), any other
        ///     list as a <c>list</c>, a string-keyed dictionary (<c>Info</c>) as a <c>dict</c> in its order, recursively; other
        ///     values convert as <see cref="ToPython"/> converts them.
        /// </summary>
        /// <param name="value">
        ///     The value. Its arrays stay owned by the caller, which may dispose them as soon as this returns: each view holds its
        ///     own reference to the memory until Python lets go of the last view over it.
        /// </param>
        /// <returns>A new owned reference. Must be called under the GIL.</returns>
        /// <exception cref="ArgumentException">Thrown for a value with no defined conversion.</exception>
        /// <exception cref="NotSupportedException">Thrown for a <c>StringDType</c> array, which keeps its text outside any buffer a view could share.</exception>
        internal static PyObject ToPythonShared(object? value) {
            switch (value) {
                case NDArray array:
                    return array.ToNumpy();
                case object?[] tuple: {
                    var items = new PyObject[tuple.Length];
                    try {
                        for (int i = 0; i < items.Length; i++)
                            items[i] = ToPythonShared(tuple[i]);
                        return new PyTuple(items);
                    } finally {
                        // PyTuple takes its own references to the items.
                        foreach (PyObject? item in items)
                            item?.Dispose();
                    }
                }
                case IEnumerable<KeyValuePair<string, object?>> entries: {
                    var dictionary = new PyDict();
                    try {
                        foreach ((string key, object? item) in entries) {
                            using PyObject converted = ToPythonShared(item);
                            // The dict takes its own reference to the value.
                            dictionary[key] = converted;
                        }
                    } catch {
                        dictionary.Dispose();
                        throw;
                    }

                    return dictionary;
                }
                case IList list: {
                    var pythonList = new PyList();
                    try {
                        foreach (object? item in list) {
                            using PyObject converted = ToPythonShared(item);
                            pythonList.Append(converted);
                        }
                    } catch {
                        pythonList.Dispose();
                        throw;
                    }

                    return pythonList;
                }
                default:
                    // Scalars, strings, None and value tuples hold no Gym.NET memory, so the copying conversion is the same thing.
                    return ToPython(value);
            }
        }

        /// <summary>
        ///     Converts a Python value to the CLR value a Gym.NET user would pass for it: what the reverse bridge does to every
        ///     argument Gymnasium hands a Gym.NET env (an action, reset options, constructor kwargs, a spec's fields).
        /// </summary>
        /// <param name="value">The value; stays owned by the caller (<see langword="null"/> reads as <c>None</c>).</param>
        /// <returns>
        ///     <see langword="null"/> for <c>None</c>; a <see langword="bool"/>, a <see langword="long"/> (a
        ///     <see cref="BigInteger"/> past 64 bits), a <see langword="double"/> or a <see langword="string"/> for Python's
        ///     scalars; an <see cref="NDArray"/> copied from a NumPy array, and a 0-d one for a NumPy scalar (NumSharp has no
        ///     scalar types, so a 0-d array is how Gym.NET holds <c>np.int64(1)</c>, DR-08); an <see cref="Info"/> for a
        ///     <c>dict</c>, an <see langword="object"/>?[] for a <c>tuple</c> (Gym.NET's tuple, DR-13), a
        ///     <see cref="List{T}"/> for a <c>list</c>, recursively. The caller owns the arrays in it
        ///     (<see cref="Outcome.Release"/> releases them).
        /// </returns>
        /// <exception cref="TypeError">
        ///     Thrown for a value Gym.NET has no counterpart for (an arbitrary object, a <c>dict</c> key that is not a
        ///     <c>str</c>), naming its type; the bridge raises it in Python as Python's <c>TypeError</c>.
        /// </exception>
        /// <exception cref="PythonException">Thrown when inspecting the value raises.</exception>
        /// <remarks>
        ///     Must be called under the GIL. The order of the checks matters: <c>bool</c> before <c>int</c> (Python's
        ///     <c>bool</c> subclasses <c>int</c>), and NumPy's scalars before Python's (<c>np.float64</c> subclasses
        ///     <c>float</c> and must stay a float64 array). Arrays cross as copies (<c>ToNDArray</c>), so nothing the env
        ///     keeps can alias Python memory a later Python call mutates.
        /// </remarks>
        internal static object? FromPython(PyObject? value) {
            if (value is null || value.IsNone())
                return null;

            // The cached modules: never disposed here.
            PyObject builtins = PythonRuntime.Builtins;
            PyObject numpy = PythonRuntime.Import("numpy");
            if (IsInstance(value, builtins, "bool"))
                return value.IsTrue();
            if (IsInstance(value, numpy, "ndarray"))
                return value.ToNDArray();
            if (IsInstance(value, numpy, "generic")) {
                using PyObject array = Python.numpy.asarray(value);
                return array.ToNDArray();
            }

            if (IsInstance(value, builtins, "int")) {
                using var integer = new PyInt(value);
                BigInteger whole = integer.ToBigInteger();
                return whole >= long.MinValue && whole <= long.MaxValue ? (long)whole : whole;
            }

            if (IsInstance(value, builtins, "float"))
                return value.As<double>();
            if (IsInstance(value, builtins, "str")) {
                // str() first: pythonnet's As<string> accepts the exact str type only, not a subclass.
                using PyObject text = Python.builtins.str(value);
                return text.As<string>();
            }

            if (IsInstance(value, builtins, "dict"))
                return InfoFromPython(value);
            if (IsInstance(value, builtins, "tuple"))
                return ItemsFromPython(value).ToArray();
            if (IsInstance(value, builtins, "list"))
                return ItemsFromPython(value);
            throw new TypeError($"Gym.NET has no counterpart for a Python {ValueFacts.PythonTypeName(value)} ({value.Repr()})");
        }

        /// <summary>A <c>dict</c> as an <see cref="Info"/>, values converted by <see cref="FromPython"/>, keys in the dict's order.</summary>
        /// <param name="value">The dict; stays owned by the caller.</param>
        /// <returns>The info; the caller owns the arrays in it.</returns>
        /// <exception cref="TypeError">Thrown for a key that is not a <c>str</c>, or a value <see cref="FromPython"/> refuses.</exception>
        private static Info InfoFromPython(PyObject value) {
            var info = new Info();
            try {
                using PyObject items = value.InvokeMethod("items");
                using var iterator = PyIter.GetIter(items);
                while (iterator.MoveNext()) {
                    using PyObject item = iterator.Current;
                    using PyObject key = item[0];
                    using PyObject entry = item[1];
                    if (!IsInstance(key, PythonRuntime.Builtins, "str"))
                        throw new TypeError($"Gym.NET's dicts have str keys, not {ValueFacts.PythonTypeName(key)} ({key.Repr()})");
                    info[key.As<string>()] = FromPython(entry);
                }
            } catch {
                // What was converted before the failure belongs to nobody else.
                Outcome.Release(info);
                throw;
            }

            return info;
        }

        /// <summary>A <c>tuple</c> or <c>list</c>'s items converted by <see cref="FromPython"/>, in order.</summary>
        /// <param name="value">The sequence; stays owned by the caller.</param>
        /// <returns>The items; the caller owns the arrays among them.</returns>
        /// <exception cref="TypeError">Thrown for an item <see cref="FromPython"/> refuses.</exception>
        private static List<object?> ItemsFromPython(PyObject value) {
            var items = new List<object?>();
            try {
                using var iterator = PyIter.GetIter(value);
                while (iterator.MoveNext()) {
                    using PyObject item = iterator.Current;
                    items.Add(FromPython(item));
                }
            } catch {
                Outcome.Release(items);
                throw;
            }

            return items;
        }

        /// <summary><c>isinstance(value, getattr(module, name))</c>, disposing the type object. Must run under the GIL.</summary>
        /// <param name="value">The value.</param>
        /// <param name="module">The module that defines the type (never disposed here).</param>
        /// <param name="name">The type's name in it.</param>
        /// <returns><see langword="true"/> when the value is an instance.</returns>
        private static bool IsInstance(PyObject value, PyObject module, string name) {
            using PyObject type = module.GetAttr(name);
            return value.IsInstance(type);
        }

        /// <summary>
        ///     Whether a value is a container of numbers whose float64 form is exactly its values: a <c>tuple</c> or <c>list</c>
        ///     (MountainCar's state after a discrete step is a tuple of NumPy scalars and Python numbers), or an array of a
        ///     floating dtype (float32 widens to float64 exactly). Must run under the GIL.
        /// </summary>
        /// <param name="value">The value; stays owned by the caller.</param>
        /// <returns><see langword="true"/> for such a container; <see langword="false"/> for <c>None</c>, a scalar, or an array of another kind.</returns>
        /// <exception cref="PythonException">Thrown when reading an array's dtype raises.</exception>
        internal static bool IsFloatContainer(PyObject value) {
            // The cached modules: never disposed here.
            PyObject builtins = PythonRuntime.Builtins;
            PyObject numpy = PythonRuntime.Import("numpy");
            if (IsInstance(value, builtins, "tuple") || IsInstance(value, builtins, "list"))
                return true;
            if (!IsInstance(value, numpy, "ndarray"))
                return false;
            // Type reflection on a run-time value, like FromPython's: the array's dtype kind, 'f' for floating.
            using PyObject dtype = value.GetAttr("dtype");
            using PyObject kind = dtype.GetAttr("kind");
            return kind.As<string>() == "f";
        }
    }
}
