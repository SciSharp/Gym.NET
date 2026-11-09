#nullable enable
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using Gym.Envs;
using NumSharp;

namespace Gym.Utils {
    /// <summary>
    ///     Python's <c>json.dumps</c> and <c>json.loads</c> with their default arguments, over the .NET counterparts of
    ///     Python's values (<see cref="Info"/> for dicts, lists and tuples, .NET integers for <c>int</c>, <see langword="double"/>
    ///     for <c>float</c>). <c>EnvSpec.to_json</c> and <c>from_json</c> go through it, so a spec's JSON is Gymnasium's text
    ///     byte for byte and reads back to equal values.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The output is <c>json.dumps</c>'s: <c>", "</c> and <c>": "</c> separators, every non-ASCII character escaped
    ///     (<c>ensure_ascii</c>), floats by Python's <c>repr</c>, and <c>NaN</c>/<c>Infinity</c> written bare
    ///     (<c>allow_nan</c>). What Python cannot serialize fails with its <see cref="TypeError"/>: a NumPy scalar or array, a
    ///     <see langword="float"/> (NumPy's float32 to Gym.NET), any object that is not a dict, list, string, number, bool or
    ///     <c>None</c>.
    ///     </para>
    ///     <para>
    ///     Reading follows <c>json.loads</c>: objects become <see cref="Info"/> (a repeated key keeps its last value), arrays
    ///     <see cref="List{T}"/> of <see langword="object"/>, integers <see langword="long"/> (<see cref="BigInteger"/> beyond
    ///     it), other numbers <see langword="double"/>; <c>NaN</c>, <c>Infinity</c> and <c>-Infinity</c> are accepted. A
    ///     syntax error is a <see cref="ValueError"/> with the decoder's message and position (Python's
    ///     <c>JSONDecodeError</c> is a <c>ValueError</c>).
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/python/cpython/blob/v3.12.12/Lib/json/encoder.py">Lib/json/encoder.py</see>, <see href="https://github.com/python/cpython/blob/v3.12.12/Lib/json/decoder.py">Lib/json/decoder.py</see>.</para>
    /// </remarks>
    internal static class PyJson {
        /// <summary>Python's <c>json.dumps(value)</c>.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The JSON text, ASCII only.</returns>
        /// <exception cref="TypeError">Thrown for a value Python cannot serialize: <c>Object of type float32 is not JSON serializable</c>.</exception>
        public static string Dumps(object? value) {
            var text = new StringBuilder();
            Encode(value, text);
            return text.ToString();
        }

        /// <summary>Python's <c>json.loads(text)</c>.</summary>
        /// <param name="text">The JSON text.</param>
        /// <returns>The value.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="text"/> is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">Thrown for malformed JSON, with the decoder's message: <c>Expecting value: line 1 column 1 (char 0)</c>.</exception>
        public static object? Loads(string text) {
            if (text is null)
                throw new ArgumentNullException(nameof(text));
            var reader = new Reader(text);
            reader.SkipWhitespace();
            object? value = reader.ReadValue();
            reader.SkipWhitespace();
            if (!reader.AtEnd)
                throw reader.Error("Extra data", reader.Position);
            return value;
        }

        /// <summary>Writes one value as the encoder does.</summary>
        /// <param name="value">The value.</param>
        /// <param name="text">The output.</param>
        /// <exception cref="TypeError">Thrown for a value that is not serializable.</exception>
        private static void Encode(object? value, StringBuilder text) {
            switch (value) {
                case null:
                    text.Append("null");
                    return;
                case string s:
                    EncodeString(s, text);
                    return;
                case bool flag:
                    // bool before int: Python's True is an int, but JSON writes it as true.
                    text.Append(flag ? "true" : "false");
                    return;
                case double d:
                    text.Append(double.IsNaN(d) ? "NaN" : double.IsPositiveInfinity(d) ? "Infinity" : double.IsNegativeInfinity(d) ? "-Infinity" : PyBuiltins.Repr(d));
                    return;
                case sbyte or byte or short or ushort or int or uint or long or ulong or BigInteger:
                    text.Append(((IFormattable)value).ToString(null, CultureInfo.InvariantCulture));
                    return;
                case Enum member:
                    // An enum member is an int to Gym.NET (as an IntEnum is to Python), written as its value.
                    text.Append(PyBuiltins.Repr(member));
                    return;
            }

            if (Dict(value) is { } entries) {
                text.Append('{');
                bool first = true;
                foreach (KeyValuePair<string, object?> entry in entries) {
                    if (!first)
                        text.Append(", ");
                    first = false;
                    EncodeString(entry.Key, text);
                    text.Append(": ");
                    Encode(entry.Value, text);
                }

                text.Append('}');
                return;
            }

            if (value is IList or ITuple && value is not NDArray) {
                text.Append('[');
                bool first = true;
                foreach (object? item in value is ITuple tuple ? TupleItems(tuple) : (IEnumerable)value) {
                    if (!first)
                        text.Append(", ");
                    first = false;
                    Encode(item, text);
                }

                text.Append(']');
                return;
            }

            // o.__class__.__name__: the bare class name, numpy.float32's "float32".
            string typeName = PyBuiltins.TypeName(value);
            throw new TypeError($"Object of type {typeName.Substring(typeName.LastIndexOf('.') + 1)} is not JSON serializable");
        }

        /// <summary>
        ///     A string as <c>ensure_ascii</c> writes it: quotes and backslashes escaped, the short escapes for <c>\n</c>,
        ///     <c>\r</c>, <c>\t</c>, <c>\b</c>, <c>\f</c>, and everything else outside printable ASCII as lower-case
        ///     <c>\uXXXX</c> (a character beyond the BMP as its surrogate pair).
        /// </summary>
        /// <param name="value">The string.</param>
        /// <param name="text">The output.</param>
        private static void EncodeString(string value, StringBuilder text) {
            text.Append('"');
            foreach (char c in value) {
                switch (c) {
                    case '"':
                        text.Append("\\\"");
                        break;
                    case '\\':
                        text.Append("\\\\");
                        break;
                    case '\n':
                        text.Append("\\n");
                        break;
                    case '\r':
                        text.Append("\\r");
                        break;
                    case '\t':
                        text.Append("\\t");
                        break;
                    case '\b':
                        text.Append("\\b");
                        break;
                    case '\f':
                        text.Append("\\f");
                        break;
                    default:
                        // The encoder's ESCAPE_ASCII class [^\ -~]: anything that is not printable ASCII.
                        if (c < ' ' || c > '~')
                            text.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                        else
                            text.Append(c);
                        break;
                }
            }

            text.Append('"');
        }

        /// <summary>A dict's entries in order, or <see langword="null"/> for a value that is not a dict.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The entries.</returns>
        /// <exception cref="TypeError">Thrown for a dict key that is not a string (JSON keys are strings; Gym.NET's dicts have string keys).</exception>
        private static IEnumerable<KeyValuePair<string, object?>>? Dict(object value) {
            switch (value) {
                case IEnumerable<KeyValuePair<string, object?>> pairs:
                    return pairs;
                case IDictionary dictionary: {
                    var entries = new List<KeyValuePair<string, object?>>();
                    foreach (DictionaryEntry entry in dictionary)
                        entries.Add(new KeyValuePair<string, object?>(entry.Key as string ?? throw new TypeError($"keys must be str, int, float, bool or None, not {PyBuiltins.TypeName(entry.Key)}"), entry.Value));
                    return entries;
                }
                default:
                    return null;
            }
        }

        /// <summary>A tuple's items.</summary>
        /// <param name="tuple">The tuple.</param>
        /// <returns>The items, in order.</returns>
        private static IEnumerable<object?> TupleItems(ITuple tuple) {
            for (int i = 0; i < tuple.Length; i++)
                yield return tuple[i];
        }

        /// <summary>The decoder: a recursive-descent reader over the text with Python's error messages.</summary>
        private sealed class Reader {
            /// <summary>The text.</summary>
            private readonly string _text;

            /// <summary>Reads <paramref name="text"/> from its start.</summary>
            /// <param name="text">The JSON text.</param>
            public Reader(string text) => _text = text;

            /// <summary>The index of the next character.</summary>
            public int Position { get; private set; }

            /// <summary>Whether the whole text was read.</summary>
            public bool AtEnd => Position >= _text.Length;

            /// <summary>Skips JSON whitespace: space, tab, newline, carriage return.</summary>
            public void SkipWhitespace() {
                while (Position < _text.Length && _text[Position] is ' ' or '\t' or '\n' or '\r')
                    Position++;
            }

            /// <summary>Reads one value at the current position.</summary>
            /// <returns>The value.</returns>
            /// <exception cref="ValueError">Thrown for malformed JSON.</exception>
            public object? ReadValue() {
                if (AtEnd)
                    throw Error("Expecting value", Position);
                char c = _text[Position];
                switch (c) {
                    case '{':
                        return ReadObject();
                    case '[':
                        return ReadArray();
                    case '"':
                        Position++;
                        return ReadString(Position - 1);
                }

                if (Match("null"))
                    return null;
                if (Match("true"))
                    return true;
                if (Match("false"))
                    return false;
                if (Match("NaN"))
                    return double.NaN;
                if (Match("Infinity"))
                    return double.PositiveInfinity;
                if (Match("-Infinity"))
                    return double.NegativeInfinity;
                return ReadNumber() ?? throw Error("Expecting value", Position);
            }

            /// <summary>Reads an object into an <see cref="Info"/>.</summary>
            /// <returns>The dict.</returns>
            /// <exception cref="ValueError">Thrown for a malformed object.</exception>
            private Info ReadObject() {
                var result = new Info();
                Position++;
                SkipWhitespace();
                if (!AtEnd && _text[Position] == '}') {
                    Position++;
                    return result;
                }

                while (true) {
                    if (AtEnd || _text[Position] != '"')
                        throw Error("Expecting property name enclosed in double quotes", Position);
                    Position++;
                    string key = ReadString(Position - 1);
                    SkipWhitespace();
                    if (AtEnd || _text[Position] != ':')
                        throw Error("Expecting ':' delimiter", Position);
                    Position++;
                    SkipWhitespace();
                    result[key] = ReadValue();
                    SkipWhitespace();
                    if (!AtEnd && _text[Position] == '}') {
                        Position++;
                        return result;
                    }

                    if (AtEnd || _text[Position] != ',')
                        throw Error("Expecting ',' delimiter", Position);
                    Position++;
                    SkipWhitespace();
                }
            }

            /// <summary>Reads an array into a list.</summary>
            /// <returns>The list.</returns>
            /// <exception cref="ValueError">Thrown for a malformed array.</exception>
            private List<object?> ReadArray() {
                var result = new List<object?>();
                Position++;
                SkipWhitespace();
                if (!AtEnd && _text[Position] == ']') {
                    Position++;
                    return result;
                }

                while (true) {
                    result.Add(ReadValue());
                    SkipWhitespace();
                    if (!AtEnd && _text[Position] == ']') {
                        Position++;
                        return result;
                    }

                    if (AtEnd || _text[Position] != ',')
                        throw Error("Expecting ',' delimiter", Position);
                    Position++;
                    SkipWhitespace();
                }
            }

            /// <summary>Reads a string's body after its opening quote, decoding escapes (a <c>\uXXXX</c> surrogate pair is one character).</summary>
            /// <param name="start">The index of the opening quote, for the unterminated-string message.</param>
            /// <returns>The string.</returns>
            /// <exception cref="ValueError">Thrown for an unterminated string, a raw control character, or a bad escape.</exception>
            private string ReadString(int start) {
                var text = new StringBuilder();
                while (true) {
                    if (AtEnd)
                        throw Error("Unterminated string starting at", start);
                    char c = _text[Position];
                    if (c == '"') {
                        Position++;
                        return text.ToString();
                    }

                    if (c < ' ')
                        throw Error("Invalid control character at", Position);
                    if (c != '\\') {
                        text.Append(c);
                        Position++;
                        continue;
                    }

                    if (Position + 1 >= _text.Length)
                        throw Error("Unterminated string starting at", start);
                    char escape = _text[Position + 1];
                    switch (escape) {
                        case '"':
                        case '\\':
                        case '/':
                            text.Append(escape);
                            break;
                        case 'b':
                            text.Append('\b');
                            break;
                        case 'f':
                            text.Append('\f');
                            break;
                        case 'n':
                            text.Append('\n');
                            break;
                        case 'r':
                            text.Append('\r');
                            break;
                        case 't':
                            text.Append('\t');
                            break;
                        case 'u':
                            text.Append(ReadUnicodeEscape(Position));
                            // ReadUnicodeEscape moved past the escape (and a following low surrogate).
                            continue;
                        default:
                            throw Error("Invalid \\escape", Position);
                    }

                    Position += 2;
                }
            }

            /// <summary>Reads <c>\uXXXX</c> at <paramref name="at"/>, joining a high surrogate with a following <c>\uXXXX</c> low one.</summary>
            /// <param name="at">The index of the backslash.</param>
            /// <returns>The character or surrogate pair.</returns>
            /// <exception cref="ValueError">Thrown when four hex digits do not follow.</exception>
            private string ReadUnicodeEscape(int at) {
                int code = HexAt(at + 2) ?? throw Error("Invalid \\uXXXX escape", at + 1);
                Position = at + 6;
                if (code is >= 0xD800 and <= 0xDBFF && Position + 5 < _text.Length && _text[Position] == '\\' && _text[Position + 1] == 'u'
                    && HexAt(Position + 2) is int low and >= 0xDC00 and <= 0xDFFF) {
                    Position += 6;
                    return new string(new[] { (char)code, (char)low });
                }

                return ((char)code).ToString();
            }

            /// <summary>Four hex digits at <paramref name="at"/>, or <see langword="null"/>.</summary>
            /// <param name="at">The index of the first digit.</param>
            /// <returns>The value.</returns>
            private int? HexAt(int at) {
                if (at + 4 > _text.Length)
                    return null;
                return int.TryParse(_text.AsSpan(at, 4), NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture, out int value) ? value : null;
            }

            /// <summary>
            ///     Reads a number by JSON's grammar (<c>-?(0|[1-9]\d*)(\.\d+)?([eE][-+]?\d+)?</c>): without a fraction or exponent
            ///     a Python int, otherwise a float.
            /// </summary>
            /// <returns>The number, or <see langword="null"/> when none starts here.</returns>
            private object? ReadNumber() {
                int start = Position, i = Position;
                if (i < _text.Length && _text[i] == '-')
                    i++;
                if (i >= _text.Length || !char.IsAsciiDigit(_text[i]))
                    return null;
                if (_text[i] == '0')
                    i++;
                else
                    while (i < _text.Length && char.IsAsciiDigit(_text[i]))
                        i++;

                bool isFloat = false;
                if (i + 1 < _text.Length && _text[i] == '.' && char.IsAsciiDigit(_text[i + 1])) {
                    isFloat = true;
                    i++;
                    while (i < _text.Length && char.IsAsciiDigit(_text[i]))
                        i++;
                }

                if (i < _text.Length && _text[i] is 'e' or 'E') {
                    int exponent = i + 1;
                    if (exponent < _text.Length && _text[exponent] is '+' or '-')
                        exponent++;
                    if (exponent < _text.Length && char.IsAsciiDigit(_text[exponent])) {
                        isFloat = true;
                        i = exponent;
                        while (i < _text.Length && char.IsAsciiDigit(_text[i]))
                            i++;
                    }
                }

                string literal = _text.Substring(start, i - start);
                Position = i;
                if (isFloat)
                    return double.Parse(literal, NumberStyles.Float, CultureInfo.InvariantCulture);
                BigInteger integer = BigInteger.Parse(literal, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture);
                return integer >= long.MinValue && integer <= long.MaxValue ? (long)integer : integer;
            }

            /// <summary>Consumes <paramref name="literal"/> when the text continues with it.</summary>
            /// <param name="literal">The literal.</param>
            /// <returns><see langword="true"/> when it was there.</returns>
            private bool Match(string literal) {
                if (string.CompareOrdinal(_text, Position, literal, 0, literal.Length) != 0)
                    return false;
                Position += literal.Length;
                return true;
            }

            /// <summary>The decoder's error: <c>{msg}: line {l} column {c} (char {pos})</c>.</summary>
            /// <param name="message">The message.</param>
            /// <param name="position">The index it refers to.</param>
            /// <returns>The error to throw.</returns>
            public ValueError Error(string message, int position) {
                int line = 1, lastNewline = -1;
                for (int i = 0; i < position && i < _text.Length; i++) {
                    if (_text[i] == '\n') {
                        line++;
                        lastNewline = i;
                    }
                }

                int column = position - lastNewline;
                return new ValueError($"{message}: line {line.ToString(CultureInfo.InvariantCulture)} column {column.ToString(CultureInfo.InvariantCulture)} (char {position.ToString(CultureInfo.InvariantCulture)})");
            }
        }
    }
}
