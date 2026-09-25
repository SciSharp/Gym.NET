using System;
using System.Buffers.Binary;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace Gym.Tests.Parity.Oracle {
    /// <summary>What kind of value a <see cref="ValueFacts"/> describes.</summary>
    public enum FactKind {
        /// <summary>Python <c>None</c>, .NET <see langword="null"/>.</summary>
        None,

        /// <summary>A boolean: Python <c>bool</c> and <c>numpy.bool</c>, .NET <see cref="bool"/>.</summary>
        Bool,

        /// <summary>An integer: Python <c>int</c> (no dtype, unbounded), a numpy integer scalar or a .NET integer (with a dtype).</summary>
        Int,

        /// <summary>A floating-point scalar, kept as its raw bytes: Python <c>float</c>, a numpy floating scalar, .NET <see cref="float"/>/<see cref="double"/>.</summary>
        Float,

        /// <summary>An n-dimensional array: <c>numpy.ndarray</c> or <see cref="NDArray"/>, as dtype, shape and C-order bytes.</summary>
        Array,

        /// <summary>A string.</summary>
        Str,

        /// <summary>A mapping, entries in iteration order (Python dicts keep insertion order, and Gymnasium's infos rely on it).</summary>
        Dict,

        /// <summary>A tuple or list.</summary>
        Seq,

        /// <summary>Anything else, kept as its type name and <c>repr</c>/<c>ToString</c>.</summary>
        Other,
    }

    /// <summary>
    ///     A canonical, side-neutral description of one value, produced from Python by <see cref="FromPython"/> and from .NET by
    ///     <see cref="Of"/>. Comparing two descriptions (<see cref="ByteContract"/>) compares the bits each side produced:
    ///     arrays by dtype, shape and C-order bytes, floats by their IEEE-754 bytes, integers by value.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The two sides reach the bytes by different paths: a Python array crosses through NumSharp's interop import
    ///     (<c>ToNDArray</c>) first, a path Gym.NET's own arrays never take, so a fault in the crossing shows up as a difference
    ///     rather than hiding one. <c>SessionTests.ByteReaders_AgreeOnAnExportedView</c> and
    ///     <c>ByteReaders_AgreeOnAnImportedView</c> pin the crossing itself against NumPy's own <c>tobytes</c>.
    ///     </para>
    ///     <para>
    ///     Instances are immutable and hold no Python or NumSharp objects, so they can be kept after the GIL is released and
    ///     after the arrays they describe are disposed.
    ///     </para>
    /// </remarks>
    public sealed class ValueFacts {
        /// <summary>Creates a description of the given kind; the factories fill the members that kind uses.</summary>
        /// <param name="kind">The kind of value.</param>
        /// <param name="typeName">The value's type as its own side names it; diagnostics only, never compared.</param>
        private ValueFacts(FactKind kind, string typeName) {
            Kind = kind;
            TypeName = typeName;
        }

        /// <summary>The kind of value, which decides which of the other members are meaningful.</summary>
        public FactKind Kind { get; }

        /// <summary>
        ///     The value's type as its side names it (<c>numpy.float32</c>, <c>float</c>, <c>NDArray</c>, <c>long</c>). Shown in
        ///     reports; never compared, because the two sides spell equivalent types differently.
        /// </summary>
        public string TypeName { get; }

        /// <summary>
        ///     The NumPy array-interface type string (<c>&lt;f4</c>, <c>|b1</c>, <c>&lt;i8</c>) of an <see cref="FactKind.Array"/>,
        ///     a <see cref="FactKind.Float"/>, or a fixed-width <see cref="FactKind.Int"/>; empty for a Python <c>int</c>, which
        ///     has no width.
        /// </summary>
        public string DType { get; private init; } = "";

        /// <summary>The value of an <see cref="FactKind.Int"/>.</summary>
        public BigInteger Integer { get; private init; }

        /// <summary>The value of a <see cref="FactKind.Bool"/>.</summary>
        public bool Boolean { get; private init; }

        /// <summary>The raw bytes of a <see cref="FactKind.Float"/> (one element) or an <see cref="FactKind.Array"/> (C order), in <see cref="DType"/>'s byte order.</summary>
        public byte[] Bytes { get; private init; } = System.Array.Empty<byte>();

        /// <summary>The shape of an <see cref="FactKind.Array"/>; empty for a 0-d array.</summary>
        public long[] Shape { get; private init; } = System.Array.Empty<long>();

        /// <summary>The value of a <see cref="FactKind.Str"/>, or the <c>repr</c> of an <see cref="FactKind.Other"/>.</summary>
        public string Text { get; private init; } = "";

        /// <summary>The entries of a <see cref="FactKind.Dict"/>, in iteration order.</summary>
        public IReadOnlyList<KeyValuePair<ValueFacts, ValueFacts>> Entries { get; private init; } = System.Array.Empty<KeyValuePair<ValueFacts, ValueFacts>>();

        /// <summary>The items of a <see cref="FactKind.Seq"/>.</summary>
        public IReadOnlyList<ValueFacts> Items { get; private init; } = System.Array.Empty<ValueFacts>();

        /// <summary>Describes <c>None</c>/<see langword="null"/>.</summary>
        /// <param name="typeName">How the side names it.</param>
        /// <returns>The description.</returns>
        public static ValueFacts NoneValue(string typeName = "NoneType") => new(FactKind.None, typeName);

        /// <summary>Describes a boolean.</summary>
        /// <param name="value">The value.</param>
        /// <param name="typeName">How the side names its type.</param>
        /// <returns>The description.</returns>
        public static ValueFacts FromBool(bool value, string typeName = "bool") => new(FactKind.Bool, typeName) { Boolean = value };

        /// <summary>Describes an integer.</summary>
        /// <param name="value">The value.</param>
        /// <param name="dtype">Its dtype string, or empty for an unbounded Python <c>int</c>.</param>
        /// <param name="typeName">How the side names its type.</param>
        /// <returns>The description.</returns>
        public static ValueFacts FromInteger(BigInteger value, string dtype, string typeName) => new(FactKind.Int, typeName) { Integer = value, DType = dtype };

        /// <summary>Describes a floating-point scalar by its raw bytes.</summary>
        /// <param name="dtype">Its dtype string, which fixes width and byte order (<c>&lt;f8</c> for a Python <c>float</c>).</param>
        /// <param name="bytes">The value's bytes in that order; not copied, so the caller must not reuse the array.</param>
        /// <param name="typeName">How the side names its type.</param>
        /// <returns>The description.</returns>
        public static ValueFacts FromFloat(string dtype, byte[] bytes, string typeName) => new(FactKind.Float, typeName) { DType = dtype, Bytes = bytes };

        /// <summary>Describes a <see langword="double"/> as a Python <c>float</c> would be described.</summary>
        /// <param name="value">The value.</param>
        /// <param name="typeName">How the side names its type.</param>
        /// <returns>The description, dtype <c>&lt;f8</c>.</returns>
        public static ValueFacts FromDouble(double value, string typeName = "double") {
            var bytes = new byte[8];
            BinaryPrimitives.WriteDoubleLittleEndian(bytes, value);
            return FromFloat("<f8", bytes, typeName);
        }

        /// <summary>Describes an array.</summary>
        /// <param name="dtype">Its dtype string.</param>
        /// <param name="shape">Its shape; not copied.</param>
        /// <param name="bytes">Its elements in C order; not copied.</param>
        /// <param name="typeName">How the side names its type.</param>
        /// <returns>The description.</returns>
        public static ValueFacts FromArray(string dtype, long[] shape, byte[] bytes, string typeName) => new(FactKind.Array, typeName) { DType = dtype, Shape = shape, Bytes = bytes };

        /// <summary>Describes a string.</summary>
        /// <param name="value">The value.</param>
        /// <param name="typeName">How the side names its type.</param>
        /// <returns>The description.</returns>
        public static ValueFacts FromText(string value, string typeName = "str") => new(FactKind.Str, typeName) { Text = value };

        /// <summary>Describes a mapping.</summary>
        /// <param name="entries">Its entries in iteration order.</param>
        /// <param name="typeName">How the side names its type.</param>
        /// <returns>The description.</returns>
        public static ValueFacts FromDict(IEnumerable<KeyValuePair<ValueFacts, ValueFacts>> entries, string typeName = "dict") => new(FactKind.Dict, typeName) { Entries = entries.ToList() };

        /// <summary>Describes a tuple or list.</summary>
        /// <param name="items">Its items in order.</param>
        /// <param name="typeName">How the side names its type (<c>tuple</c>, <c>list</c>).</param>
        /// <returns>The description.</returns>
        public static ValueFacts FromSequence(IEnumerable<ValueFacts> items, string typeName) => new(FactKind.Seq, typeName) { Items = items.ToList() };

        /// <summary>Describes a value no other kind covers.</summary>
        /// <param name="typeName">Its type.</param>
        /// <param name="repr">Its <c>repr</c> or <c>ToString</c>.</param>
        /// <returns>The description.</returns>
        public static ValueFacts FromOther(string typeName, string repr) => new(FactKind.Other, typeName) { Text = repr };

        /// <summary>
        ///     Describes a Python value: <c>None</c>, booleans (<c>bool</c>, <c>numpy.bool</c>), integers (Python <c>int</c>
        ///     without a dtype, NumPy integer scalars with theirs), floats (Python <c>float</c> as <c>&lt;f8</c>, NumPy floating
        ///     scalars at their width), arrays, dtypes (as their array-interface string), strings, dictionaries in iteration
        ///     order, tuples and lists. Must run under the GIL; <paramref name="value"/> stays owned by the caller.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>The description.</returns>
        /// <exception cref="PythonException">Thrown when a Python call made to inspect the value raises.</exception>
        /// <remarks>
        ///     <para>
        ///     This is the suite's generic Python-to-CLR marshaling, which the pythonic skill keeps out of the <c>Python.*</c>
        ///     tree: which attribute it reads depends on the value's type at run time.
        ///     </para>
        ///     <para>
        ///     Arrays cross through NumSharp's interop (<c>ToNDArray</c>: one bulk, stride-honouring copy into a C-contiguous
        ///     <see cref="NDArray"/>), then are read with NumSharp's own <c>data.tobytes</c>. The .NET side never goes through that
        ///     import, so the two sides still share no code path: a faulty import would show as a difference, never hide one.
        ///     </para>
        /// </remarks>
        public static ValueFacts FromPython(PyObject value) {
            string typeName = PythonTypeName(value);
            if (value.IsNone())
                return NoneValue(typeName);

            // The cached module: never disposed here.
            PyObject numpy = Interop.PythonRuntime.Import("numpy");
            // bool first: Python's bool subclasses int, and numpy.bool is not a numpy.integer.
            if (IsInstance(value, PythonRuntimeBuiltin("bool")) || IsInstance(value, numpy, "bool_"))
                return FromBool(value.IsTrue(), typeName);
            if (IsInstance(value, numpy, "ndarray")) {
                using NDArray array = value.ToNDArray();
                return FromArray(array.dtype.str, (long[])array.shape.Clone(), array.data.tobytes("C"), typeName);
            }

            // A dtype is described by NumPy's own array-interface string (dtype.str: '<f4', '|u1'), the text NumSharp's
            // DType.str spells, so a space's dtype compares with Gym.NET's without either side printing a repr.
            if (IsInstance(value, numpy, "dtype")) {
                using PyObject text = value.GetAttr("str");
                return FromText(text.As<string>(), typeName);
            }

            if (IsInstance(value, numpy, "integer")) {
                using PyObject asInt = Python.builtins.@int(value);
                using var integer = new PyInt(asInt);
                return FromInteger(integer.ToBigInteger(), DTypeStr(value), typeName);
            }

            // numpy.floating before float: numpy.float64 subclasses Python's float and must keep its NumPy dtype. The bytes
            // are NumPy's own (the scalar as a one-element array, crossed like any array) rather than float(x) narrowed
            // back: that round trip would quiet a signalling NaN, and a NaN payload is part of the bits compared.
            if (IsInstance(value, numpy, "floating")) {
                using PyObject wrapped = Python.numpy.atleast_1d(value);
                using NDArray element = wrapped.ToNDArray();
                return FromFloat(element.dtype.str, element.data.tobytes("C"), typeName);
            }
            if (IsInstance(value, PythonRuntimeBuiltin("int"))) {
                using var integer = new PyInt(value);
                return FromInteger(integer.ToBigInteger(), "", typeName);
            }

            if (IsInstance(value, PythonRuntimeBuiltin("float")))
                return FromDouble(value.As<double>(), typeName);
            if (IsInstance(value, PythonRuntimeBuiltin("str")))
                return FromText(value.As<string>(), typeName);
            if (IsInstance(value, PythonRuntimeBuiltin("dict"))) {
                var pairs = new List<KeyValuePair<ValueFacts, ValueFacts>>();
                using PyObject items = value.InvokeMethod("items");
                using var iterator = PyIter.GetIter(items);
                while (iterator.MoveNext()) {
                    using PyObject item = iterator.Current;
                    using PyObject key = item[0];
                    using PyObject entry = item[1];
                    pairs.Add(new KeyValuePair<ValueFacts, ValueFacts>(FromPython(key), FromPython(entry)));
                }

                return FromDict(pairs, typeName);
            }

            if (IsInstance(value, PythonRuntimeBuiltin("tuple")) || IsInstance(value, PythonRuntimeBuiltin("list"))) {
                var items = new List<ValueFacts>();
                using var iterator = PyIter.GetIter(value);
                while (iterator.MoveNext()) {
                    using PyObject item = iterator.Current;
                    items.Add(FromPython(item));
                }

                return FromSequence(items, typeName);
            }

            return FromOther(typeName, value.Repr() ?? "");
        }

        /// <summary>
        ///     A Python value's type as reports print it: <c>float</c>, <c>numpy.float32</c>, <c>numpy.ndarray</c>. Builtins go
        ///     unqualified; everything else keeps its module. Must run under the GIL.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>The type's name.</returns>
        internal static string PythonTypeName(PyObject value) {
            using PyType type = value.GetPythonType();
            return QualifiedName(type);
        }

        /// <summary>A Python type's name as <see cref="PythonTypeName"/> prints it. Must run under the GIL.</summary>
        /// <param name="type">The type object.</param>
        /// <returns>The name.</returns>
        internal static string QualifiedName(PyObject type) {
            using PyObject module = type.GetAttr("__module__");
            using PyObject name = type.GetAttr("__qualname__");
            string moduleName = module.As<string>();
            return moduleName == "builtins" ? name.As<string>() : $"{moduleName}.{name.As<string>()}";
        }

        /// <summary><c>isinstance(value, getattr(module, name))</c>, disposing the type object. Must run under the GIL.</summary>
        /// <param name="value">The value.</param>
        /// <param name="module">The module that defines the type.</param>
        /// <param name="name">The type's name in it.</param>
        /// <returns><see langword="true"/> when the value is an instance.</returns>
        private static bool IsInstance(PyObject value, PyObject module, string name) {
            using PyObject type = module.GetAttr(name);
            return value.IsInstance(type);
        }

        /// <summary><c>isinstance(value, type)</c>, disposing <paramref name="type"/>. Must run under the GIL.</summary>
        /// <param name="value">The value.</param>
        /// <param name="type">The type object; disposed here.</param>
        /// <returns><see langword="true"/> when the value is an instance.</returns>
        private static bool IsInstance(PyObject value, PyObject type) {
            using (type)
                return value.IsInstance(type);
        }

        /// <summary>A built-in type (<c>int</c>, <c>dict</c>, ...) as a new reference. Must run under the GIL.</summary>
        /// <param name="name">The type's name.</param>
        /// <returns>A new owned reference.</returns>
        private static PyObject PythonRuntimeBuiltin(string name) => Interop.PythonRuntime.Builtins.GetAttr(name);

        /// <summary><c>value.dtype.str</c> of a NumPy scalar. Must run under the GIL.</summary>
        /// <param name="value">The scalar.</param>
        /// <returns>The array-interface type string.</returns>
        private static string DTypeStr(PyObject value) {
            using PyObject dtype = value.GetAttr("dtype");
            using PyObject text = dtype.GetAttr("str");
            return text.As<string>();
        }

        /// <summary>
        ///     Describes a .NET value the way <see cref="FromPython"/> describes its Python counterpart: an
        ///     <see cref="NDArray"/> as an array, primitives with the dtype of their width, strings, dictionaries in
        ///     enumeration order, tuples and other sequences item by item.
        /// </summary>
        /// <param name="value">The value; it is only read, never disposed.</param>
        /// <returns>The description.</returns>
        /// <remarks>
        ///     A <see cref="BigInteger"/> maps to a Python <c>int</c> (no dtype), because both are unbounded. An enum member maps
        ///     to its underlying integer, as a Python <c>IntEnum</c> would compare.
        /// </remarks>
        public static ValueFacts Of(object? value) {
            switch (value) {
                case null:
                    return NoneValue("null");
                case NDArray array:
                    // Python-side bytes come from numpy's tobytes; these come from NumSharp's own ndarray.data.tobytes, never
                    // from the interop's export, so a byte match cross-checks two implementations.
                    return FromArray(array.dtype.str, (long[])array.shape.Clone(), array.data.tobytes("C"), "NDArray");
                case bool flag:
                    return FromBool(flag, "bool");
                case sbyte n:
                    return FromInteger(n, "|i1", "sbyte");
                case byte n:
                    return FromInteger(n, "|u1", "byte");
                case short n:
                    return FromInteger(n, "<i2", "short");
                case ushort n:
                    return FromInteger(n, "<u2", "ushort");
                case int n:
                    return FromInteger(n, "<i4", "int");
                case uint n:
                    return FromInteger(n, "<u4", "uint");
                case long n:
                    return FromInteger(n, "<i8", "long");
                case ulong n:
                    return FromInteger(n, "<u8", "ulong");
                case BigInteger n:
                    return FromInteger(n, "", "BigInteger");
                case Half h: {
                    var bytes = new byte[2];
                    BinaryPrimitives.WriteHalfLittleEndian(bytes, h);
                    return FromFloat("<f2", bytes, "Half");
                }
                case float f: {
                    var bytes = new byte[4];
                    BinaryPrimitives.WriteSingleLittleEndian(bytes, f);
                    return FromFloat("<f4", bytes, "float");
                }
                case double d:
                    return FromDouble(d);
                case string s:
                    return FromText(s, "string");
                case DType dtype:
                    // As FromPython describes numpy.dtype: by the array-interface string, '<f4', '|b1'.
                    return FromText(dtype.str, "DType");
                case Enum e:
                    return Of(Convert.ChangeType(e, Enum.GetUnderlyingType(e.GetType()), CultureInfo.InvariantCulture));
                case IDictionary dictionary: {
                    var pairs = new List<KeyValuePair<ValueFacts, ValueFacts>>();
                    foreach (DictionaryEntry entry in dictionary)
                        pairs.Add(new KeyValuePair<ValueFacts, ValueFacts>(Of(entry.Key), Of(entry.Value)));
                    return FromDict(pairs, dictionary.GetType().Name);
                }
                case ITuple tuple: {
                    var items = new List<ValueFacts>(tuple.Length);
                    for (int i = 0; i < tuple.Length; i++)
                        items.Add(Of(tuple[i]));
                    return FromSequence(items, "tuple");
                }
                case IEnumerable sequence:
                    return FromSequence(sequence.Cast<object?>().Select(Of), value.GetType().Name);
                default:
                    return FromOther(value.GetType().FullName ?? value.GetType().Name, value.ToString() ?? "");
            }
        }

        /// <summary>
        ///     The scalar a 0-d array holds, described as the numpy scalar of the same dtype. NumSharp has no scalar types: where
        ///     NumPy returns <c>np.int64(3)</c>, NumSharp returns a 0-d int64 array, and <see cref="ByteContract"/> treats the two
        ///     as the same value.
        /// </summary>
        /// <returns>The scalar, or this description when it is not a 0-d array of a boolean, integer or floating dtype.</returns>
        public ValueFacts AsScalar() {
            if (Kind != FactKind.Array || Shape.Length != 0)
                return this;
            var dtype = DTypeInfo.Parse(DType);
            return dtype.Kind switch {
                'b' => FromBool(Bytes[0] != 0, TypeName),
                'i' or 'u' => FromInteger(dtype.ReadInteger(Bytes, 0), DType, TypeName),
                'f' => FromFloat(DType, Bytes, TypeName),
                _ => this,
            };
        }

        /// <summary>A key that is equal for equal descriptions, used to match dictionary keys between the sides.</summary>
        /// <returns>A string that identifies the value, without its type name.</returns>
        public string Canonical() {
            return Kind switch {
                FactKind.Array => $"array<{DType}>{FormatShape(Shape)}:{Convert.ToHexString(Bytes)}",
                FactKind.Float => $"float<{DType}>:{Convert.ToHexString(Bytes)}",
                _ => Render(),
            };
        }

        /// <summary>Renders the value for a report: Python spelling where one exists, arrays abbreviated to their first elements.</summary>
        /// <returns>A one-line rendering.</returns>
        public string Render() {
            switch (Kind) {
                case FactKind.None:
                    return "None";
                case FactKind.Bool:
                    return Boolean ? "True" : "False";
                case FactKind.Int:
                    return DType.Length == 0 ? Integer.ToString(CultureInfo.InvariantCulture) : $"{Integer.ToString(CultureInfo.InvariantCulture)} ({DType})";
                case FactKind.Float: {
                    var dtype = DTypeInfo.Parse(DType);
                    return $"{FormatNumber(dtype.ReadDouble(Bytes, 0))} ({DType} 0x{dtype.Hex(Bytes, 0)})";
                }
                case FactKind.Array: {
                    var dtype = DTypeInfo.Parse(DType);
                    int count = dtype.Size == 0 ? 0 : Bytes.Length / dtype.Size;
                    var shown = Enumerable.Range(0, Math.Min(count, 6)).Select(i => dtype.Format(Bytes, i));
                    return $"{DType} array {FormatShape(Shape)} [{string.Join(", ", shown)}{(count > 6 ? ", ..." : "")}]";
                }
                case FactKind.Str:
                    return $"'{Text}'";
                case FactKind.Dict:
                    return "{" + string.Join(", ", Entries.Select(e => $"{e.Key.Render()}: {e.Value.Render()}")) + "}";
                case FactKind.Seq:
                    if (TypeName == "list")
                        return "[" + string.Join(", ", Items.Select(i => i.Render())) + "]";
                    // Python's trailing comma marks a one-element tuple: (4,), not (4).
                    return "(" + string.Join(", ", Items.Select(i => i.Render())) + (Items.Count == 1 ? ",)" : ")");
                default:
                    return $"<{TypeName}> {Text}";
            }
        }

        /// <inheritdoc />
        public override string ToString() => Render();

        /// <summary>Formats a shape as a Python tuple: <c>()</c>, <c>(4,)</c>, <c>(400, 600, 3)</c>.</summary>
        /// <param name="shape">The dimensions.</param>
        /// <returns>The tuple's text.</returns>
        public static string FormatShape(IReadOnlyList<long> shape) {
            return shape.Count == 1 ? $"({shape[0]},)" : "(" + string.Join(", ", shape) + ")";
        }

        /// <summary>Formats a double with the shortest text that round-trips, so two different values never print alike.</summary>
        /// <param name="value">The value.</param>
        /// <returns>Its round-trip text.</returns>
        internal static string FormatNumber(double value) => value.ToString("R", CultureInfo.InvariantCulture);
    }

    /// <summary>
    ///     A parsed NumPy dtype string (<c>&lt;f8</c>, <c>|u1</c>): byte order, kind and item size, with the element decoders
    ///     the reports and the ulp arithmetic need.
    /// </summary>
    /// <param name="Order">The byte order character: <c>&lt;</c> little, <c>&gt;</c> big, <c>|</c> not applicable, <c>=</c> native.</param>
    /// <param name="Kind">The kind character: <c>b</c>, <c>i</c>, <c>u</c>, <c>f</c>, <c>c</c>, ...</param>
    /// <param name="Size">The item size in bytes.</param>
    internal readonly record struct DTypeInfo(char Order, char Kind, int Size) {
        /// <summary>Parses a dtype string of the form byte order, kind, size.</summary>
        /// <param name="dtype">The string, for example <c>&lt;f4</c>.</param>
        /// <returns>The parsed parts; kind <c>?</c> and size 0 for a string that does not have that form.</returns>
        public static DTypeInfo Parse(string dtype) {
            if (dtype.Length < 3 || !int.TryParse(dtype.AsSpan(2), NumberStyles.None, CultureInfo.InvariantCulture, out int size))
                return new DTypeInfo('|', '?', 0);
            return new DTypeInfo(dtype[0], dtype[1], size);
        }

        /// <summary>Whether multi-byte elements are stored big-endian.</summary>
        private bool BigEndian => Order == '>' || (Order == '=' && !BitConverter.IsLittleEndian);

        /// <summary>The bytes of element <paramref name="index"/>, reordered to little-endian.</summary>
        /// <param name="bytes">The array's bytes.</param>
        /// <param name="index">The element's flat index.</param>
        /// <returns>A copy of the element's bytes in little-endian order.</returns>
        private byte[] Element(byte[] bytes, int index) {
            byte[] element = bytes.AsSpan(index * Size, Size).ToArray();
            if (BigEndian)
                System.Array.Reverse(element);
            return element;
        }

        /// <summary>Decodes a floating element (<c>f2</c>, <c>f4</c>, <c>f8</c>) as a double; floats widen exactly.</summary>
        /// <param name="bytes">The array's bytes.</param>
        /// <param name="index">The element's flat index.</param>
        /// <returns>The value, or NaN for an element that is not floating-point.</returns>
        public double ReadDouble(byte[] bytes, int index) {
            byte[] element = Element(bytes, index);
            return (Kind, Size) switch {
                ('f', 2) => (double)BinaryPrimitives.ReadHalfLittleEndian(element),
                ('f', 4) => BinaryPrimitives.ReadSingleLittleEndian(element),
                ('f', 8) => BinaryPrimitives.ReadDoubleLittleEndian(element),
                ('i' or 'u' or 'b', _) => (double)ReadInteger(bytes, index),
                _ => double.NaN,
            };
        }

        /// <summary>Decodes an integer or boolean element.</summary>
        /// <param name="bytes">The array's bytes.</param>
        /// <param name="index">The element's flat index.</param>
        /// <returns>The value; zero for any other kind.</returns>
        public BigInteger ReadInteger(byte[] bytes, int index) {
            byte[] element = Element(bytes, index);
            return Kind switch {
                'i' => new BigInteger(element, isUnsigned: false, isBigEndian: false),
                'u' or 'b' => new BigInteger(element, isUnsigned: true, isBigEndian: false),
                _ => BigInteger.Zero,
            };
        }

        /// <summary>The element's bits as hexadecimal, most significant byte first, the way IEEE-754 patterns are usually written.</summary>
        /// <param name="bytes">The array's bytes.</param>
        /// <param name="index">The element's flat index.</param>
        /// <returns>For example <c>3F947AE147AE147B</c> for the double 0.02.</returns>
        public string Hex(byte[] bytes, int index) {
            byte[] element = Element(bytes, index);
            System.Array.Reverse(element);
            return Convert.ToHexString(element);
        }

        /// <summary>Renders one element for a report.</summary>
        /// <param name="bytes">The array's bytes.</param>
        /// <param name="index">The element's flat index.</param>
        /// <returns>The value in round-trip form, or the raw hex for kinds without a decoder.</returns>
        public string Format(byte[] bytes, int index) {
            return Kind switch {
                'f' => ValueFacts.FormatNumber(ReadDouble(bytes, index)),
                'b' => ReadInteger(bytes, index).IsZero ? "False" : "True",
                'i' or 'u' => ReadInteger(bytes, index).ToString(CultureInfo.InvariantCulture),
                _ => "0x" + Hex(bytes, index),
            };
        }

        /// <summary>
        ///     The distance in units in the last place between two elements of this floating dtype: how many representable
        ///     values lie between them. Zero means bit-identical (or <c>+0</c> against <c>-0</c>, which differ in bits but are
        ///     adjacent).
        /// </summary>
        /// <param name="left">The first array's bytes.</param>
        /// <param name="right">The second array's bytes.</param>
        /// <param name="index">The element's flat index in both.</param>
        /// <returns>The distance, or <see langword="null"/> when the dtype is not a 32- or 64-bit float or either value is NaN.</returns>
        public BigInteger? Ulps(byte[] left, byte[] right, int index) {
            double a = ReadDouble(left, index), b = ReadDouble(right, index);
            if (double.IsNaN(a) || double.IsNaN(b))
                return null;
            return Size switch {
                8 => BigInteger.Abs((BigInteger)Ordered(BitConverter.DoubleToInt64Bits(a)) - Ordered(BitConverter.DoubleToInt64Bits(b))),
                4 => BigInteger.Abs((BigInteger)Ordered(BitConverter.SingleToInt32Bits((float)a)) - Ordered(BitConverter.SingleToInt32Bits((float)b))),
                _ => null,
            };
        }

        /// <summary>
        ///     Maps IEEE-754 bits to an integer that increases with the value, so that subtracting two of them counts the
        ///     representable values between (negative floats are stored as sign-magnitude, so their bits run backwards).
        /// </summary>
        /// <param name="bits">The bits of a double.</param>
        /// <returns>The ordered integer.</returns>
        internal static long Ordered(long bits) => bits < 0 ? long.MinValue - bits : bits;

        /// <summary>The 32-bit counterpart of <see cref="Ordered(long)"/>.</summary>
        /// <param name="bits">The bits of a float.</param>
        /// <returns>The ordered integer.</returns>
        internal static long Ordered(int bits) => bits < 0 ? (long)int.MinValue - bits : bits;
    }
}
