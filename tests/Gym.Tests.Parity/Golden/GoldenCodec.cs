using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.Json;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Pair;

namespace Gym.Tests.Parity.Golden {
    /// <summary>
    ///     Writes and reads the values a golden holds, losslessly: a <see cref="ValueFacts"/> round-trips to an equal
    ///     description (same kind, dtype, shape, bytes, integer, text, entry order), so a replay compares exactly what the live
    ///     cell compared.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A value is one JSON object: <c>{"kind": "array", "type": "numpy.ndarray", "dtype": "&lt;f4", "shape": [4],
    ///     "hex": "…"}</c>. Raw bytes are uppercase hexadecimal rather than decimal text, so a float's NaN payload and the sign
    ///     of zero survive, and integers are decimal strings because a Python <c>int</c> has no width (a PCG64 state is 128
    ///     bits). <c>type</c> is the side's own type name, kept for reports; like <see cref="ValueFacts.TypeName"/> it is never
    ///     compared.
    ///     </para>
    ///     <para>
    ///     Every writer emits properties in one fixed order, so recording the same trajectory twice produces the same text,
    ///     which is what lets <see cref="GoldenStore"/> leave an unchanged golden untouched.
    ///     </para>
    /// </remarks>
    internal static class GoldenCodec {
        /// <summary>Writes a value as one JSON object.</summary>
        /// <param name="writer">The writer, positioned where a value may start.</param>
        /// <param name="facts">The value.</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a <see cref="FactKind"/> this codec does not know, which only a new kind added to <see cref="ValueFacts"/> without a codec case can produce.</exception>
        public static void Write(Utf8JsonWriter writer, ValueFacts facts) {
            writer.WriteStartObject();
            writer.WriteString("kind", KindName(facts.Kind));
            writer.WriteString("type", facts.TypeName);
            switch (facts.Kind) {
                case FactKind.None:
                    break;
                case FactKind.Bool:
                    writer.WriteBoolean("value", facts.Boolean);
                    break;
                case FactKind.Int:
                    writer.WriteString("dtype", facts.DType);
                    writer.WriteString("value", facts.Integer.ToString(CultureInfo.InvariantCulture));
                    break;
                case FactKind.Float:
                    writer.WriteString("dtype", facts.DType);
                    writer.WriteString("hex", Convert.ToHexString(facts.Bytes));
                    break;
                case FactKind.Array:
                    writer.WriteString("dtype", facts.DType);
                    writer.WriteStartArray("shape");
                    foreach (long dimension in facts.Shape)
                        writer.WriteNumberValue(dimension);
                    writer.WriteEndArray();
                    writer.WriteString("hex", Convert.ToHexString(facts.Bytes));
                    break;
                case FactKind.Str:
                    writer.WriteString("value", facts.Text);
                    break;
                case FactKind.Dict:
                    // Pairs rather than a JSON object: keys are values of any kind (Python dicts take ints and tuples as
                    // keys), and the iteration order is part of what is compared.
                    writer.WriteStartArray("entries");
                    foreach (KeyValuePair<ValueFacts, ValueFacts> entry in facts.Entries) {
                        writer.WriteStartArray();
                        Write(writer, entry.Key);
                        Write(writer, entry.Value);
                        writer.WriteEndArray();
                    }

                    writer.WriteEndArray();
                    break;
                case FactKind.Seq:
                    writer.WriteStartArray("items");
                    foreach (ValueFacts item in facts.Items)
                        Write(writer, item);
                    writer.WriteEndArray();
                    break;
                case FactKind.Other:
                    writer.WriteString("repr", facts.Text);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(facts), facts.Kind, "A value kind the golden codec does not know.");
            }

            writer.WriteEndObject();
        }

        /// <summary>Writes a value, or JSON <c>null</c> for <see langword="null"/>.</summary>
        /// <param name="writer">The writer.</param>
        /// <param name="name">The property name.</param>
        /// <param name="facts">The value, or <see langword="null"/> when the call did not compare this field.</param>
        public static void WriteOptional(Utf8JsonWriter writer, string name, ValueFacts? facts) {
            writer.WritePropertyName(name);
            if (facts is null)
                writer.WriteNullValue();
            else
                Write(writer, facts);
        }

        /// <summary>Reads a value written by <see cref="Write"/>.</summary>
        /// <param name="element">The JSON object.</param>
        /// <returns>An equal description.</returns>
        /// <exception cref="InvalidDataException">Thrown when the object is not a value this codec wrote: an unknown kind, a missing member, malformed hex or integer text.</exception>
        public static ValueFacts Read(JsonElement element) {
            if (element.ValueKind != JsonValueKind.Object)
                throw new InvalidDataException($"A golden value must be a JSON object, not {element.ValueKind}.");
            string kind = RequiredString(element, "kind");
            string type = RequiredString(element, "type");
            switch (kind) {
                case "none":
                    return ValueFacts.NoneValue(type);
                case "bool":
                    return ValueFacts.FromBool(Required(element, "value").GetBoolean(), type);
                case "int": {
                    string text = RequiredString(element, "value");
                    if (!BigInteger.TryParse(text, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out BigInteger value))
                        throw new InvalidDataException($"A golden integer must be decimal text, not '{text}'.");
                    return ValueFacts.FromInteger(value, RequiredString(element, "dtype"), type);
                }
                case "float":
                    return ValueFacts.FromFloat(RequiredString(element, "dtype"), Hex(element), type);
                case "array": {
                    long[] shape = Required(element, "shape").EnumerateArray().Select(d => d.GetInt64()).ToArray();
                    return ValueFacts.FromArray(RequiredString(element, "dtype"), shape, Hex(element), type);
                }
                case "str":
                    return ValueFacts.FromText(RequiredString(element, "value"), type);
                case "dict": {
                    var entries = new List<KeyValuePair<ValueFacts, ValueFacts>>();
                    foreach (JsonElement pair in Required(element, "entries").EnumerateArray()) {
                        if (pair.GetArrayLength() != 2)
                            throw new InvalidDataException("A golden dict entry must be a [key, value] pair.");
                        entries.Add(new KeyValuePair<ValueFacts, ValueFacts>(Read(pair[0]), Read(pair[1])));
                    }

                    return ValueFacts.FromDict(entries, type);
                }
                case "seq":
                    return ValueFacts.FromSequence(Required(element, "items").EnumerateArray().Select(Read).ToList(), type);
                case "other":
                    return ValueFacts.FromOther(type, RequiredString(element, "repr"));
                default:
                    throw new InvalidDataException($"Unknown golden value kind '{kind}'.");
            }
        }

        /// <summary>Reads an optional value: JSON <c>null</c> or a missing member is <see langword="null"/>.</summary>
        /// <param name="parent">The object holding the member.</param>
        /// <param name="name">The member's name.</param>
        /// <returns>The value, or <see langword="null"/>.</returns>
        /// <exception cref="InvalidDataException">Thrown when the member holds something that is not a value.</exception>
        public static ValueFacts? ReadOptional(JsonElement parent, string name) {
            if (!parent.TryGetProperty(name, out JsonElement element) || element.ValueKind == JsonValueKind.Null)
                return null;
            return Read(element);
        }

        /// <summary>
        ///     Writes one call's result: <c>"result": {"observation": …, "reward": …, "terminated": …, "truncated": …,
        ///     "info": …}</c> (a reset has no reward or flags), or <c>"raised": {"type": …, "message": …}</c>.
        /// </summary>
        /// <param name="writer">The writer, inside the call's object.</param>
        /// <param name="result">Gymnasium's result, which matched Gym.NET's when it was recorded.</param>
        public static void WriteResult(Utf8JsonWriter writer, CallResult result) {
            if (result.Raised is { } raised) {
                writer.WriteStartObject("raised");
                // The Python type as ValueFacts.PythonTypeName prints it: ByteContract maps it to the .NET type it must equal.
                writer.WriteString("type", raised.Type);
                writer.WriteString("message", raised.Message);
                writer.WriteEndObject();
                return;
            }

            Transition transition = result.Transition!;
            writer.WriteStartObject("result");
            WriteOptional(writer, "observation", transition.Observation);
            // A reset's transition has no reward and no flags: they are left out rather than written as null, so the
            // reader can tell a reset from a step that returned None.
            if (transition.Reward != null) {
                WriteOptional(writer, "reward", transition.Reward);
                WriteOptional(writer, "terminated", transition.Terminated);
                WriteOptional(writer, "truncated", transition.Truncated);
            }

            WriteOptional(writer, "info", transition.Info);
            writer.WriteEndObject();
        }

        /// <summary>Reads a result written by <see cref="WriteResult"/>.</summary>
        /// <param name="call">The call's object, holding <c>result</c> or <c>raised</c>.</param>
        /// <returns>The result.</returns>
        /// <exception cref="InvalidDataException">Thrown when neither member is present, or a value inside is malformed.</exception>
        public static CallResult ReadResult(JsonElement call) {
            if (call.TryGetProperty("raised", out JsonElement raised))
                return CallResult.Threw(new RaisedError(RequiredString(raised, "type"), RequiredString(raised, "message")));
            if (!call.TryGetProperty("result", out JsonElement result))
                throw new InvalidDataException("A golden call holds neither 'result' nor 'raised'.");

            ValueFacts observation = Read(Required(result, "observation"));
            ValueFacts info = Read(Required(result, "info"));
            if (!result.TryGetProperty("reward", out _))
                return CallResult.Of(new Transition(observation, null, null, null, info));
            return CallResult.Of(new Transition(observation, Read(Required(result, "reward")), Read(Required(result, "terminated")),
                Read(Required(result, "truncated")), info));
        }

        /// <summary>The kind's name in the file.</summary>
        /// <param name="kind">The kind.</param>
        /// <returns>Its lowercase name.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a kind added to <see cref="FactKind"/> without a name here.</exception>
        private static string KindName(FactKind kind) => kind switch {
            FactKind.None => "none",
            FactKind.Bool => "bool",
            FactKind.Int => "int",
            FactKind.Float => "float",
            FactKind.Array => "array",
            FactKind.Str => "str",
            FactKind.Dict => "dict",
            FactKind.Seq => "seq",
            FactKind.Other => "other",
            _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, "A value kind the golden codec does not know."),
        };

        /// <summary>A member that must be present.</summary>
        /// <param name="element">The object.</param>
        /// <param name="name">The member.</param>
        /// <returns>The member's value.</returns>
        /// <exception cref="InvalidDataException">Thrown when it is missing.</exception>
        internal static JsonElement Required(JsonElement element, string name) {
            if (!element.TryGetProperty(name, out JsonElement value))
                throw new InvalidDataException($"A golden object lacks '{name}'.");
            return value;
        }

        /// <summary>A string member that must be present.</summary>
        /// <param name="element">The object.</param>
        /// <param name="name">The member.</param>
        /// <returns>The string.</returns>
        /// <exception cref="InvalidDataException">Thrown when it is missing or not a string.</exception>
        internal static string RequiredString(JsonElement element, string name) {
            JsonElement value = Required(element, name);
            if (value.ValueKind != JsonValueKind.String)
                throw new InvalidDataException($"A golden object's '{name}' must be a string, not {value.ValueKind}.");
            return value.GetString()!;
        }

        /// <summary>The bytes of a <c>hex</c> member.</summary>
        /// <param name="element">The object.</param>
        /// <returns>The decoded bytes.</returns>
        /// <exception cref="InvalidDataException">Thrown when the text is not hexadecimal of even length.</exception>
        private static byte[] Hex(JsonElement element) {
            string text = RequiredString(element, "hex");
            try {
                return Convert.FromHexString(text);
            } catch (FormatException exception) {
                throw new InvalidDataException($"A golden value's hex is malformed: {exception.Message}", exception);
            }
        }
    }
}
