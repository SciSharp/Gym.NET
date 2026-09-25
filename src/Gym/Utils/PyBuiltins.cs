#nullable enable
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using NumSharp;

namespace Gym.Utils {
    /// <summary>
    ///     Python's built-in conversions and representations, as the Gymnasium source relies on them: <c>float(x)</c> for
    ///     option parsing, <c>int(x)</c> of a float for pixel coordinates, and <c>repr</c>/<c>str</c>/<c>type</c> for the
    ///     messages of asserts and errors, which callers and ported tests match verbatim.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     .NET values stand for the Python values a Gymnasium user would pass: any .NET integer is a Python <c>int</c>,
    ///     <see langword="double"/> is <c>float</c>, <see langword="bool"/> is <c>bool</c>, <see langword="string"/> is
    ///     <c>str</c>, <see langword="null"/> is <c>None</c>. <see langword="float"/> and <see cref="Half"/> are NumPy's
    ///     <c>float32</c>/<c>float16</c> scalars (Python has no such built-in). A 0-d <see cref="NDArray"/> is the NumPy scalar
    ///     of its dtype (<c>np.int64(3)</c>), because NumSharp has no scalar types and Gym.NET returns 0-d arrays where
    ///     Gymnasium returns NumPy scalars (<c>Discrete.sample()</c>); any other <see cref="NDArray"/> is an <c>ndarray</c>.
    ///     A dictionary (<see cref="IDictionary"/>, or string-keyed pairs such as <see cref="Gym.Envs.Info"/>) is a
    ///     <c>dict</c>, a <see cref="ITuple"/> a <c>tuple</c>, and a .NET array or other <see cref="IList"/> a <c>list</c>.
    ///     Other .NET objects have no Python counterpart: they print through <see cref="object.ToString"/> and their .NET type
    ///     name.
    ///     </para>
    ///     <para>
    ///     Numbers print as Python prints them (<see cref="FloatRepr"/>): the shortest digits that round-trip, positional
    ///     between 1e-4 and 1e16 and scientific (<c>1e-05</c>, <c>1e+16</c>) outside, never with .NET's <c>E+16</c>.
    ///     </para>
    /// </remarks>
    public static class PyBuiltins {
        /// <summary>
        ///     Python's <c>float(x)</c>: a number converted exactly or correctly rounded, a string parsed by Python's rules
        ///     (whitespace, <c>_</c> between digits, <c>inf</c>/<c>infinity</c>/<c>nan</c> in any case, overflow to infinity), a
        ///     0-d array's element.
        /// </summary>
        /// <param name="value">The value, standing for its Python counterpart (see the class remarks).</param>
        /// <returns>The float.</returns>
        /// <exception cref="ValueError">Thrown for a string that is not a float literal: <c>could not convert string to float: 'abc'</c>.</exception>
        /// <exception cref="TypeError">
        ///     Thrown for <see langword="null"/> and other non-numbers (<c>float() argument must be a string or a real number,
        ///     not 'NoneType'</c>), and for an array with dimensions (<c>only 0-dimensional arrays can be converted to Python
        ///     scalars</c>, NumPy 2.4's message).
        /// </exception>
        /// <exception cref="OverflowException">Thrown, as Python's <c>OverflowError</c>, for an integer beyond the float range: <c>int too large to convert to float</c>.</exception>
        public static double Float(object? value) {
            switch (value) {
                case null:
                    throw new TypeError("float() argument must be a string or a real number, not 'NoneType'");
                case bool flag:
                    return flag ? 1.0 : 0.0;
                case double d:
                    return d;
                case float f:
                    return f;
                case Half h:
                    return (double)h;
                case sbyte or byte or short or ushort or int or uint or long:
                    // Exact up to 2^53, then round-half-even like Python's int -> float.
                    return Convert.ToInt64(value, CultureInfo.InvariantCulture);
                case ulong u:
                    return u;
                case BigInteger big:
                    return BigIntegerToFloat(big);
                case decimal m:
                    // float(Decimal) is correctly rounded from the exact decimal value, as double.Parse is.
                    return double.Parse(m.ToString(CultureInfo.InvariantCulture), NumberStyles.Float, CultureInfo.InvariantCulture);
                case string text:
                    return ParseFloat(text);
                case NDArray array:
                    return ArrayToFloat(array);
                default:
                    throw new TypeError($"float() argument must be a string or a real number, not '{TypeName(value)}'");
            }
        }

        /// <summary>
        ///     Python's <c>int(x)</c> of a float: truncation toward zero into an unbounded integer. The pixel coordinates of
        ///     Gymnasium's renderers go through it, so a NaN or infinite coordinate fails here exactly as in Python.
        /// </summary>
        /// <param name="value">The float.</param>
        /// <returns>The integer part.</returns>
        /// <exception cref="ValueError">Thrown for NaN: <c>cannot convert float NaN to integer</c>.</exception>
        /// <exception cref="OverflowException">Thrown, as Python's <c>OverflowError</c>, for an infinity: <c>cannot convert float infinity to integer</c>.</exception>
        public static BigInteger Int(double value) {
            if (double.IsNaN(value))
                throw new ValueError("cannot convert float NaN to integer");
            if (double.IsInfinity(value))
                throw new OverflowException("cannot convert float infinity to integer");
            // Every finite double with a zero fraction is an integer the BigInteger constructor represents exactly.
            return new BigInteger(Math.Truncate(value));
        }

        /// <summary>Python's <c>repr(x)</c> of the value's Python counterpart (see the class remarks).</summary>
        /// <param name="value">The value.</param>
        /// <returns><c>2</c>, <c>0.5</c>, <c>True</c>, <c>None</c>, <c>'1'</c>, <c>np.int64(5)</c>, <c>array([1])</c>, ...</returns>
        public static string Repr(object? value) {
            switch (value) {
                case null:
                    return "None";
                case bool flag:
                    return flag ? "True" : "False";
                case double d:
                    return FloatRepr(d);
                case float f:
                    return $"np.float32({ShortestRepr(f.ToString("R", CultureInfo.InvariantCulture))})";
                case Half h:
                    return $"np.float16({ShortestRepr(h.ToString(CultureInfo.InvariantCulture))})";
                case string text:
                    return StrRepr(text);
                case NDArray array:
                    return array.ndim == 0 ? NumpyScalarRepr(array) : np.array_repr(array);
                default:
                    return IntegerText(value) ?? ContainerRepr(value) ?? value.ToString() ?? "";
            }
        }

        /// <summary>
        ///     Python's <c>repr</c> of a container's counterpart: a dictionary (<see cref="IDictionary"/>, or any sequence of
        ///     string-keyed pairs such as <see cref="Gym.Envs.Info"/>) as a <c>dict</c>, <c>{'render_fps': 50}</c>; a
        ///     <see cref="ITuple"/> as a <c>tuple</c>, <c>(1, 2)</c> or <c>(1,)</c>; an <see cref="IList"/> (a .NET array or
        ///     list) as a <c>list</c>, <c>['human', 'rgb_array']</c>. Items print through <see cref="Repr"/>, recursively.
        /// </summary>
        /// <param name="value">A value that is not a string, a number or an array (those are handled before).</param>
        /// <returns>The text, or <see langword="null"/> when the value is not one of these containers.</returns>
        /// <remarks>
        ///     Other enumerables (sets, lazy sequences) are left to <see cref="object.ToString"/>: enumerating an arbitrary
        ///     sequence to print it could be unbounded or have side effects. A container that holds itself recurses without
        ///     end; Python prints <c>{...}</c> there, which no Gymnasium message relies on.
        /// </remarks>
        private static string? ContainerRepr(object value) {
            switch (value) {
                case IDictionary dictionary: {
                    var text = new StringBuilder("{");
                    bool first = true;
                    foreach (DictionaryEntry entry in dictionary) {
                        if (!first)
                            text.Append(", ");
                        first = false;
                        text.Append(Repr(entry.Key)).Append(": ").Append(Repr(entry.Value));
                    }

                    return text.Append('}').ToString();
                }
                case IEnumerable<KeyValuePair<string, object?>> pairs: {
                    // Info and other generic string-keyed maps that are not a non-generic IDictionary.
                    var text = new StringBuilder("{");
                    bool first = true;
                    foreach (KeyValuePair<string, object?> pair in pairs) {
                        if (!first)
                            text.Append(", ");
                        first = false;
                        text.Append(StrRepr(pair.Key)).Append(": ").Append(Repr(pair.Value));
                    }

                    return text.Append('}').ToString();
                }
                case ITuple tuple: {
                    var text = new StringBuilder("(");
                    for (int i = 0; i < tuple.Length; i++) {
                        if (i > 0)
                            text.Append(", ");
                        text.Append(Repr(tuple[i]));
                    }

                    // A one-element tuple keeps its trailing comma, as Python prints (1,).
                    if (tuple.Length == 1)
                        text.Append(',');
                    return text.Append(')').ToString();
                }
                case IList list: {
                    var text = new StringBuilder("[");
                    for (int i = 0; i < list.Count; i++) {
                        if (i > 0)
                            text.Append(", ");
                        text.Append(Repr(list[i]));
                    }

                    return text.Append(']').ToString();
                }
                default:
                    return null;
            }
        }

        /// <summary>
        ///     Python's <c>str(x)</c> of the value's Python counterpart, what an f-string's <c>{x}</c> prints: a string itself,
        ///     numbers and <c>None</c> as <see cref="Repr"/> prints them, a NumPy scalar without its <c>np.float64(...)</c>
        ///     wrapper, an array as <c>np.array_str</c>.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>The text.</returns>
        public static string Str(object? value) {
            switch (value) {
                case string text:
                    return text;
                case float f:
                    return ShortestRepr(f.ToString("R", CultureInfo.InvariantCulture));
                case Half h:
                    return ShortestRepr(h.ToString(CultureInfo.InvariantCulture));
                case NDArray array when array.ndim == 0:
                    return NumpyScalarStr(array);
                case NDArray array:
                    return np.array_str(array);
                default:
                    return Repr(value);
            }
        }

        /// <summary>Python's <c>str(type(x))</c>: <c>&lt;class 'int'&gt;</c>, <c>&lt;class 'numpy.int64'&gt;</c>, <c>&lt;class 'numpy.ndarray'&gt;</c>.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The type's text.</returns>
        public static string TypeRepr(object? value) => $"<class '{TypeName(value)}'>";

        /// <summary>
        ///     Python's <c>repr(float)</c> (also its <c>str</c>): the shortest digits that round-trip, positional for
        ///     1e-4 &lt;= |x| &lt; 1e16 with at least one fractional digit (<c>1.0</c>, <c>0.0001</c>), scientific otherwise with a
        ///     signed exponent of at least two digits (<c>1e-05</c>, <c>1e+16</c>, <c>1.5e+300</c>); <c>inf</c>, <c>-inf</c>,
        ///     <c>nan</c>; <c>-0.0</c>.
        /// </summary>
        /// <param name="value">The float.</param>
        /// <returns>Python's text.</returns>
        public static string FloatRepr(double value) {
            if (double.IsNaN(value))
                return "nan";
            if (double.IsInfinity(value))
                return value > 0 ? "inf" : "-inf";
            // .NET Core 3.0+ "R" is the shortest round-trip string, the same digits David Gay's algorithm gives Python.
            return ShortestRepr(value.ToString("R", CultureInfo.InvariantCulture));
        }

        /// <summary>
        ///     Python's <c>repr(str)</c>: single quotes unless the text holds a single quote and no double quote, backslash
        ///     escapes for the quote, <c>\\</c>, <c>\t</c>, <c>\n</c>, <c>\r</c>, other control characters as <c>\xhh</c>, and
        ///     non-printable non-ASCII characters as <c>\xhh</c>, <c>\uhhhh</c> or <c>\Uhhhhhhhh</c>.
        /// </summary>
        /// <param name="value">The text.</param>
        /// <returns>The quoted text.</returns>
        /// <remarks>Printability follows .NET's Unicode tables, which can trail Python's by a Unicode version for new characters.</remarks>
        public static string StrRepr(string value) {
            char quote = value.Contains('\'') && !value.Contains('"') ? '"' : '\'';
            var text = new StringBuilder(value.Length + 2);
            text.Append(quote);
            for (int i = 0; i < value.Length; i++) {
                char c = value[i];
                int codePoint = c;
                // A valid surrogate pair is one code point; a lone surrogate is printed as the code unit Python stores.
                if (char.IsHighSurrogate(c) && i + 1 < value.Length && char.IsLowSurrogate(value[i + 1])) {
                    codePoint = char.ConvertToUtf32(c, value[i + 1]);
                    i++;
                }

                if (codePoint == quote || codePoint == '\\')
                    text.Append('\\').Append((char)codePoint);
                else if (codePoint == '\t')
                    text.Append("\\t");
                else if (codePoint == '\n')
                    text.Append("\\n");
                else if (codePoint == '\r')
                    text.Append("\\r");
                else if (codePoint < ' ' || codePoint == 0x7F)
                    text.Append("\\x").Append(codePoint.ToString("x2", CultureInfo.InvariantCulture));
                else if (codePoint < 0x7F)
                    text.Append((char)codePoint);
                else if (IsPrintable(codePoint))
                    text.Append(char.ConvertFromUtf32(codePoint));
                else if (codePoint <= 0xFF)
                    text.Append("\\x").Append(codePoint.ToString("x2", CultureInfo.InvariantCulture));
                else if (codePoint <= 0xFFFF)
                    text.Append("\\u").Append(codePoint.ToString("x4", CultureInfo.InvariantCulture));
                else
                    text.Append("\\U").Append(codePoint.ToString("x8", CultureInfo.InvariantCulture));
            }

            return text.Append(quote).ToString();
        }

        /// <summary>
        ///     Reformats a .NET shortest round-trip string (<c>"1E-05"</c>, <c>"123.25"</c>, <c>"-0"</c>) with Python's float
        ///     rules, which NumPy's scalar reprs share: exponent form when the decimal point would sit more than 16 digits
        ///     right of the first digit or 4 or more places left of it.
        /// </summary>
        /// <param name="roundTrip">The shortest round-trip text of a finite value.</param>
        /// <returns>Python's text.</returns>
        private static string ShortestRepr(string roundTrip) {
            if (roundTrip is "NaN")
                return "nan";
            if (roundTrip is "Infinity" or "∞")
                return "inf";
            if (roundTrip is "-Infinity" or "-∞")
                return "-inf";

            bool negative = roundTrip.StartsWith('-');
            string unsigned = negative ? roundTrip.Substring(1) : roundTrip;
            int exponentAt = unsigned.IndexOfAny(new[] { 'E', 'e' });
            string mantissa = exponentAt >= 0 ? unsigned.Substring(0, exponentAt) : unsigned;
            int exponent = exponentAt >= 0 ? int.Parse(unsigned.Substring(exponentAt + 1), NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture) : 0;
            int point = mantissa.IndexOf('.');
            string integral = point >= 0 ? mantissa.Substring(0, point) : mantissa;
            string all = integral + (point >= 0 ? mantissa.Substring(point + 1) : "");
            string digits = all.TrimStart('0');
            // decpt: where the decimal point sits relative to the first significant digit (Python's _Py_dg_dtoa output).
            int decpt = integral.Length + exponent - (all.Length - digits.Length);
            digits = digits.TrimEnd('0');
            string sign = negative ? "-" : "";
            if (digits.Length == 0)
                return sign + "0.0";

            if (decpt <= -4 || decpt > 16) {
                int shown = decpt - 1;
                string head = digits.Length > 1 ? $"{digits[0]}.{digits.Substring(1)}" : digits;
                return $"{sign}{head}e{(shown < 0 ? '-' : '+')}{Math.Abs(shown).ToString("00", CultureInfo.InvariantCulture)}";
            }

            if (decpt <= 0)
                return $"{sign}0.{new string('0', -decpt)}{digits}";
            if (decpt >= digits.Length)
                return $"{sign}{digits}{new string('0', decpt - digits.Length)}.0";
            return $"{sign}{digits.Substring(0, decpt)}.{digits.Substring(decpt)}";
        }

        /// <summary>The Python type name of a value's counterpart: <c>int</c>, <c>float</c>, <c>numpy.float32</c>, <c>numpy.ndarray</c>.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The qualified name Python prints inside <c>&lt;class '...'&gt;</c>.</returns>
        private static string TypeName(object? value) {
            switch (value) {
                case null:
                    return "NoneType";
                case bool:
                    return "bool";
                case double:
                    return "float";
                case float:
                    return "numpy.float32";
                case Half:
                    return "numpy.float16";
                case string:
                    return "str";
                case NDArray array:
                    return array.ndim == 0 && NumpyScalarName(array) is { } name ? $"numpy.{name}" : "numpy.ndarray";
                case IDictionary or IEnumerable<KeyValuePair<string, object?>>:
                    // The containers ContainerRepr prints as Python's built-ins are named like them.
                    return "dict";
                case ITuple:
                    return "tuple";
                case IList:
                    return "list";
                default:
                    return IntegerText(value) != null ? "int" : value.GetType().FullName ?? value.GetType().Name;
            }
        }

        /// <summary>The decimal text of a .NET integer or enum (an <c>int</c> in Python), or <see langword="null"/> for anything else.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The digits, or <see langword="null"/>.</returns>
        private static string? IntegerText(object value) {
            switch (value) {
                case sbyte or byte or short or ushort or int or uint or long or ulong or BigInteger:
                    return ((IFormattable)value).ToString(null, CultureInfo.InvariantCulture);
                case Enum e:
                    // An enum member stands for its value, as Gym.NET's Discrete accepts it.
                    return ((IFormattable)Convert.ChangeType(e, Enum.GetUnderlyingType(e.GetType()), CultureInfo.InvariantCulture)).ToString(null, CultureInfo.InvariantCulture);
                default:
                    return null;
            }
        }

        /// <summary>
        ///     The NumPy scalar type name of a 0-d array's dtype (<c>int64</c>, <c>float32</c>, <c>bool</c>), or
        ///     <see langword="null"/> for NumSharp-only dtypes that have no NumPy scalar.
        /// </summary>
        /// <param name="array">The 0-d array.</param>
        /// <returns>The name, or <see langword="null"/>.</returns>
        private static string? NumpyScalarName(NDArray array) {
            char kind = array.dtype.kind;
            return kind is 'b' or 'i' or 'u' or 'f' ? array.dtype.name : null;
        }

        /// <summary>NumPy 2's scalar repr: <c>np.int64(5)</c>, <c>np.float64(0.5)</c>, <c>np.float32(0.1)</c>, <c>np.True_</c>.</summary>
        /// <param name="array">The 0-d array standing for the scalar.</param>
        /// <returns>The repr.</returns>
        private static string NumpyScalarRepr(NDArray array) {
            string? name = NumpyScalarName(array);
            if (name is null)
                return np.array_repr(array);
            if (array.dtype.kind == 'b')
                return (bool)array.GetAtIndex(0) ? "np.True_" : "np.False_";
            return $"np.{name}({NumpyScalarStr(array)})";
        }

        /// <summary>NumPy's scalar str: the value alone (<c>5</c>, <c>0.5</c>, <c>True</c>), floats with Python's rules at their own width.</summary>
        /// <param name="array">The 0-d array standing for the scalar.</param>
        /// <returns>The text.</returns>
        private static string NumpyScalarStr(NDArray array) {
            object element = array.GetAtIndex(0);
            switch (element) {
                case bool flag:
                    return flag ? "True" : "False";
                case double d:
                    return FloatRepr(d);
                case float f:
                    return ShortestRepr(f.ToString("R", CultureInfo.InvariantCulture));
                case Half h:
                    return ShortestRepr(h.ToString(CultureInfo.InvariantCulture));
                default:
                    return IntegerText(element) ?? np.array_str(array);
            }
        }

        /// <summary>
        ///     Python's <c>float(int)</c> for an unbounded integer: correctly rounded, and <c>OverflowError</c> when the rounded
        ///     value is beyond the float range.
        /// </summary>
        /// <param name="value">The integer.</param>
        /// <returns>The float.</returns>
        /// <exception cref="OverflowException">Thrown for an integer that rounds beyond <see cref="double.MaxValue"/>.</exception>
        private static double BigIntegerToFloat(BigInteger value) {
            // The exact decimal digits, correctly rounded by double.Parse (IEEE-compliant since .NET Core 3.0); an overflow
            // comes back as an infinity instead of an exception.
            double result = double.Parse(value.ToString(CultureInfo.InvariantCulture), NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture);
            if (double.IsInfinity(result))
                throw new OverflowException("int too large to convert to float");
            return result;
        }

        /// <summary><c>float(ndarray)</c> under NumPy 2.4: only a 0-d array converts, to its element as a float.</summary>
        /// <param name="array">The array.</param>
        /// <returns>The float.</returns>
        /// <exception cref="TypeError">Thrown for an array with dimensions, or a dtype that is not a real number.</exception>
        private static double ArrayToFloat(NDArray array) {
            if (array.ndim != 0)
                throw new TypeError("only 0-dimensional arrays can be converted to Python scalars");
            object element = array.GetAtIndex(0);
            return element switch {
                bool or sbyte or byte or short or ushort or int or uint or long or ulong or float or double or Half => Float(element),
                _ => throw new TypeError($"float() argument must be a string or a real number, not '{TypeName(array)}'"),
            };
        }

        /// <summary>
        ///     Python's <c>float(str)</c>: Unicode whitespace and decimal digits are first mapped to ASCII, surrounding
        ///     whitespace is stripped, <c>_</c> is allowed only between digits, and the rest must be a decimal literal or
        ///     <c>inf</c>/<c>infinity</c>/<c>nan</c> (any case, optional sign). Overflow gives an infinity, as in Python.
        /// </summary>
        /// <param name="text">The string.</param>
        /// <returns>The float.</returns>
        /// <exception cref="ValueError">Thrown when the string is not a float literal.</exception>
        private static double ParseFloat(string text) {
            string? ascii = ToAsciiDecimal(text);
            if (ascii != null) {
                string stripped = ascii.Trim(' ', '\t', '\n', '\r', '\v', '\f');
                string? cleaned = RemoveUnderscores(stripped);
                if (cleaned != null && TryParseLiteral(cleaned, out double parsed))
                    return parsed;
            }

            throw new ValueError($"could not convert string to float: {StrRepr(text)}");
        }

        /// <summary>
        ///     <c>_PyUnicode_TransformDecimalAndSpaceToASCII</c>: Unicode whitespace becomes a space and Unicode decimal digits
        ///     their ASCII digit; any other non-ASCII character makes the string invalid.
        /// </summary>
        /// <param name="text">The string.</param>
        /// <returns>The ASCII string, or <see langword="null"/> when a character cannot be mapped.</returns>
        private static string? ToAsciiDecimal(string text) {
            var ascii = new StringBuilder(text.Length);
            for (int i = 0; i < text.Length; i++) {
                char c = text[i];
                if (c < 127) {
                    ascii.Append(c);
                    continue;
                }

                if (char.IsWhiteSpace(c)) {
                    ascii.Append(' ');
                    continue;
                }

                int digit = CharUnicodeInfo.GetDecimalDigitValue(text, i);
                if (digit < 0)
                    return null;
                ascii.Append((char)('0' + digit));
                // A decimal digit outside the BMP is a surrogate pair: both code units are consumed.
                if (char.IsHighSurrogate(c) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
                    i++;
            }

            return ascii.ToString();
        }

        /// <summary>
        ///     <c>_Py_string_to_number_with_underscores</c>: removes <c>_</c> characters that each sit between two digits, and
        ///     rejects any other placement (leading, trailing, doubled, next to <c>.</c> or <c>e</c>).
        /// </summary>
        /// <param name="text">The stripped ASCII string.</param>
        /// <returns>The string without underscores, or <see langword="null"/> when an underscore is misplaced.</returns>
        private static string? RemoveUnderscores(string text) {
            if (!text.Contains('_'))
                return text;
            var result = new StringBuilder(text.Length);
            char previous = '\0';
            foreach (char c in text) {
                if (c == '_') {
                    if (!char.IsAsciiDigit(previous))
                        return null;
                } else {
                    if (previous == '_' && !char.IsAsciiDigit(c))
                        return null;
                    result.Append(c);
                }

                previous = c;
            }

            return previous == '_' ? null : result.ToString();
        }

        /// <summary>
        ///     Parses what <c>PyOS_string_to_double</c> accepts in full: an optional sign, then <c>inf</c>, <c>infinity</c> or
        ///     <c>nan</c> in any case, or digits with at most one <c>.</c> (at least one digit) and an optional exponent with
        ///     digits.
        /// </summary>
        /// <param name="text">The literal, already stripped and without underscores.</param>
        /// <param name="value">The parsed value.</param>
        /// <returns><see langword="true"/> when the whole string is a literal.</returns>
        private static bool TryParseLiteral(string text, out double value) {
            value = 0;
            int i = 0;
            bool negative = false;
            if (i < text.Length && (text[i] == '+' || text[i] == '-')) {
                negative = text[i] == '-';
                i++;
            }

            string rest = text.Substring(i);
            if (rest.Equals("inf", StringComparison.OrdinalIgnoreCase) || rest.Equals("infinity", StringComparison.OrdinalIgnoreCase)) {
                value = negative ? double.NegativeInfinity : double.PositiveInfinity;
                return true;
            }

            if (rest.Equals("nan", StringComparison.OrdinalIgnoreCase)) {
                // Python's Py_NAN is C's positive quiet NaN (0x7FF8...), negated for "-nan". .NET's double.NaN is the x86
                // default NaN with the sign bit set, so the bits are spelled out to keep the sign Python gives.
                value = BitConverter.Int64BitsToDouble(negative ? unchecked((long)0xFFF8000000000000UL) : 0x7FF8000000000000L);
                return true;
            }

            int digits = 0, points = 0, j = 0;
            while (j < rest.Length && (char.IsAsciiDigit(rest[j]) || rest[j] == '.')) {
                if (rest[j] == '.')
                    points++;
                else
                    digits++;
                j++;
            }

            if (digits == 0 || points > 1)
                return false;
            if (j < rest.Length) {
                if (rest[j] != 'e' && rest[j] != 'E')
                    return false;
                j++;
                if (j < rest.Length && (rest[j] == '+' || rest[j] == '-'))
                    j++;
                int exponentDigits = 0;
                while (j < rest.Length && char.IsAsciiDigit(rest[j])) {
                    exponentDigits++;
                    j++;
                }

                if (exponentDigits == 0 || j != rest.Length)
                    return false;
            }

            // Correctly rounded like Python's dtoa; .NET returns an infinity or a signed zero on overflow or underflow.
            value = double.Parse(text, NumberStyles.AllowLeadingSign | NumberStyles.AllowDecimalPoint | NumberStyles.AllowExponent, CultureInfo.InvariantCulture);
            return true;
        }

        /// <summary>
        ///     Python's <c>str.isprintable</c> for one non-ASCII code point: every character except those in the categories
        ///     Other (Cc, Cf, Cs, Co, Cn) and Separator (Zl, Zp, Zs).
        /// </summary>
        /// <param name="codePoint">The code point.</param>
        /// <returns><see langword="true"/> when Python prints it as itself.</returns>
        private static bool IsPrintable(int codePoint) {
            if (codePoint >= 0xD800 && codePoint <= 0xDFFF)
                return false;
            UnicodeCategory category = CharUnicodeInfo.GetUnicodeCategory(codePoint);
            return category switch {
                UnicodeCategory.Control or UnicodeCategory.Format or UnicodeCategory.Surrogate or UnicodeCategory.PrivateUse
                    or UnicodeCategory.OtherNotAssigned or UnicodeCategory.LineSeparator or UnicodeCategory.ParagraphSeparator
                    or UnicodeCategory.SpaceSeparator => false,
                _ => true,
            };
        }
    }
}
