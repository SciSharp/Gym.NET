using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using Gym.Tests.Parity.Pair;

namespace Gym.Tests.Parity.Oracle {
    /// <summary>How a field is compared.</summary>
    public enum FieldPolicy {
        /// <summary>
        ///     Bit for bit: same kind of value, same dtype, same shape, same bytes; dictionaries with the same keys in the same
        ///     order. The only equivalences are the ones NumSharp's model forces (see <see cref="ByteContract"/>).
        /// </summary>
        Exact,

        /// <summary>
        ///     The float64 value: both sides' numbers are widened to double and compared by bits, whatever their scalar type.
        ///     For rewards, which Gymnasium returns as a Python <c>float</c>, <c>np.float64</c> or <c>np.float32</c> and Gym.NET
        ///     as a .NET number (plan §5): a float32 reward widens exactly, so a reward computed in float32 still shows up as a
        ///     bit difference with its ulp distance.
        /// </summary>
        Float64Value,
    }

    /// <summary>
    ///     Compares two <see cref="ValueFacts"/> and reports every difference with both values, their bits and, for floats,
    ///     their distance in ulps. Byte-exact is the only default; there is no tolerance (plan §12).
    /// </summary>
    /// <remarks>
    ///     <para>Equivalences under <see cref="FieldPolicy.Exact"/>, each forced by how NumSharp models NumPy:</para>
    ///     <list type="bullet">
    ///         <item>
    ///             A NumPy scalar (<c>np.int64(3)</c>) equals a 0-d NumSharp array of the same dtype and bytes: NumSharp has no
    ///             scalar types and returns 0-d arrays where NumPy returns scalars. The reverse (a Python 0-d array against a
    ///             .NET scalar) stays a difference, because NumSharp can represent a 0-d array.
    ///         </item>
    ///         <item>
    ///             A Python <c>int</c> equals a .NET integer of any width holding the same value: Python ints have no width,
    ///             so only the value exists to compare. Two sized integers must also agree on their dtype.
    ///         </item>
    ///     </list>
    ///     <para>Type names are never compared; kinds, dtypes, shapes and bits are.</para>
    /// </remarks>
    public static class ByteContract {
        /// <summary>How many differing elements an array report lists before summarizing.</summary>
        private const int ElementsListed = 3;

        /// <summary>Compares two descriptions of the same field.</summary>
        /// <param name="field">The top-level field name the divergences carry (<c>observation</c>, <c>info</c>, ...).</param>
        /// <param name="python">Gymnasium's value, or <see langword="null"/> when the field is absent on that side.</param>
        /// <param name="dotnet">Gym.NET's value, or <see langword="null"/> when absent.</param>
        /// <param name="policy">How to compare.</param>
        /// <returns>Every difference found, in a stable order; empty when the values match.</returns>
        public static IReadOnlyList<FieldDivergence> Compare(string field, ValueFacts? python, ValueFacts? dotnet, FieldPolicy policy = FieldPolicy.Exact) {
            var found = new List<FieldDivergence>();
            if (python is null && dotnet is null)
                return found;
            if (python is null || dotnet is null) {
                found.Add(new FieldDivergence(field, field, python is null ? $"absent on the python side; dotnet has {dotnet!.Render()}" : $"python has {python.Render()}; absent on the dotnet side"));
                return found;
            }

            if (policy == FieldPolicy.Float64Value)
                CompareFloat64(field, field, python, dotnet, found);
            else
                CompareExact(field, field, python, dotnet, found);
            return found;
        }

        /// <summary>
        ///     Compares two call outcomes: both returned (their values are compared), both raised (their types, mapped through
        ///     <see cref="ExceptionMap"/>, and their messages are compared), or one of each (a divergence by itself).
        /// </summary>
        /// <param name="field">The field name the divergences carry.</param>
        /// <param name="python">Gymnasium's outcome.</param>
        /// <param name="dotnet">Gym.NET's outcome.</param>
        /// <param name="policy">How returned values are compared.</param>
        /// <returns>Every difference found; empty when the outcomes match.</returns>
        public static IReadOnlyList<FieldDivergence> Compare(string field, Outcome python, Outcome dotnet, FieldPolicy policy = FieldPolicy.Exact) {
            if (python.Raised is null && dotnet.Raised is null)
                return Compare(field, python.Value, dotnet.Value, policy);

            var found = new List<FieldDivergence>();
            if (python.Raised is null || dotnet.Raised is null) {
                found.Add(new FieldDivergence("raised", field, $"python {python} vs dotnet {dotnet}"));
                return found;
            }

            string? expectedType = ExceptionMap.DotNetTypeFor(python.Raised.Type);
            if (expectedType is null)
                found.Add(new FieldDivergence("raised", field + ".type", $"python raised {python.Raised.Type}, which the port plan's section 3.4 maps to no .NET type; dotnet raised {dotnet.Raised.Type}"));
            else if (!string.Equals(expectedType, dotnet.Raised.Type, StringComparison.Ordinal))
                found.Add(new FieldDivergence("raised", field + ".type", $"python raised {python.Raised.Type} (expects {expectedType}) vs dotnet {dotnet.Raised.Type}"));
            if (!string.Equals(python.Raised.Message, dotnet.Raised.Message, StringComparison.Ordinal))
                found.Add(new FieldDivergence("raised", field + ".message", $"python '{python.Raised.Message}' vs dotnet '{dotnet.Raised.Message}'"));
            return found;
        }

        /// <summary>Throws a <see cref="ParityException"/> unless the two values match.</summary>
        /// <param name="what">What was compared, the report's call line (<c>Discrete(4, seed=0).sample() #3</c>).</param>
        /// <param name="python">Gymnasium's value.</param>
        /// <param name="dotnet">Gym.NET's value.</param>
        /// <param name="policy">How to compare.</param>
        /// <param name="field">The field name the divergences carry.</param>
        /// <exception cref="ParityException">Thrown when the values differ.</exception>
        public static void AssertSame(string what, ValueFacts? python, ValueFacts? dotnet, FieldPolicy policy = FieldPolicy.Exact, string field = "value") {
            Throw(what, Compare(field, python, dotnet, policy));
        }

        /// <summary>Throws a <see cref="ParityException"/> unless the two outcomes match, exceptions included.</summary>
        /// <param name="what">What was compared, the report's call line.</param>
        /// <param name="python">Gymnasium's outcome.</param>
        /// <param name="dotnet">Gym.NET's outcome.</param>
        /// <param name="policy">How returned values are compared.</param>
        /// <param name="field">The field name the divergences carry.</param>
        /// <exception cref="ParityException">Thrown when the outcomes differ.</exception>
        public static void AssertSame(string what, Outcome python, Outcome dotnet, FieldPolicy policy = FieldPolicy.Exact, string field = "value") {
            Throw(what, Compare(field, python, dotnet, policy));
        }

        /// <summary>Throws a single-call divergence report when <paramref name="found"/> is not empty.</summary>
        /// <param name="what">The report's call line.</param>
        /// <param name="found">The differences.</param>
        /// <exception cref="ParityException">Thrown when there is any difference.</exception>
        private static void Throw(string what, IReadOnlyList<FieldDivergence> found) {
            if (found.Count > 0)
                throw new ParityException(new Divergence { Case = "direct comparison", Call = what, Fields = found });
        }

        /// <summary>The <see cref="FieldPolicy.Exact"/> comparison, recursing into dictionaries and sequences.</summary>
        /// <param name="field">The top-level field.</param>
        /// <param name="path">Where in the field.</param>
        /// <param name="python">Gymnasium's value.</param>
        /// <param name="dotnet">Gym.NET's value.</param>
        /// <param name="found">Receives the differences.</param>
        private static void CompareExact(string field, string path, ValueFacts python, ValueFacts dotnet, List<FieldDivergence> found) {
            // NumSharp returns a 0-d array where NumPy returns a scalar (np.int64(3)); compare that array as the scalar it is.
            if (python.Kind is FactKind.Bool or FactKind.Int or FactKind.Float && dotnet.Kind == FactKind.Array && dotnet.Shape.Length == 0)
                dotnet = dotnet.AsScalar();

            if (python.Kind != dotnet.Kind) {
                found.Add(new FieldDivergence(field, path, $"python {python.Render()} ({python.TypeName}) vs dotnet {dotnet.Render()} ({dotnet.TypeName}): different kinds of value"));
                return;
            }

            switch (python.Kind) {
                case FactKind.None:
                    return;
                case FactKind.Bool:
                    if (python.Boolean != dotnet.Boolean)
                        found.Add(new FieldDivergence(field, path, $"python {python.Render()} vs dotnet {dotnet.Render()}"));
                    return;
                case FactKind.Int:
                    // A Python int has no dtype, so only sized integers on both sides must agree on it.
                    if (python.DType.Length > 0 && dotnet.DType.Length > 0 && python.DType != dotnet.DType)
                        found.Add(new FieldDivergence(field, path, $"dtype {python.DType} vs {dotnet.DType} (python {python.Render()}, dotnet {dotnet.Render()})"));
                    else if (python.Integer != dotnet.Integer)
                        found.Add(new FieldDivergence(field, path, $"python {python.Render()} vs dotnet {dotnet.Render()}"));
                    return;
                case FactKind.Float:
                    if (python.DType != dotnet.DType)
                        found.Add(new FieldDivergence(field, path, $"dtype {python.DType} vs {dotnet.DType} (python {python.Render()}, dotnet {dotnet.Render()})"));
                    else if (!python.Bytes.AsSpan().SequenceEqual(dotnet.Bytes))
                        found.Add(new FieldDivergence(field, path, $"python {python.Render()} vs dotnet {dotnet.Render()}{UlpSuffix(DTypeInfo.Parse(python.DType), python.Bytes, dotnet.Bytes, 0)}"));
                    return;
                case FactKind.Array:
                    CompareArrays(field, path, python, dotnet, found);
                    return;
                case FactKind.Str:
                    if (!string.Equals(python.Text, dotnet.Text, StringComparison.Ordinal))
                        found.Add(new FieldDivergence(field, path, $"python {python.Render()} vs dotnet {dotnet.Render()}"));
                    return;
                case FactKind.Dict:
                    CompareDicts(field, path, python, dotnet, found);
                    return;
                case FactKind.Seq:
                    if (python.Items.Count != dotnet.Items.Count) {
                        found.Add(new FieldDivergence(field, path, $"length {python.Items.Count} vs {dotnet.Items.Count}: python {python.Render()} vs dotnet {dotnet.Render()}"));
                        return;
                    }

                    for (int i = 0; i < python.Items.Count; i++)
                        CompareExact(field, $"{path}[{i}]", python.Items[i], dotnet.Items[i], found);
                    return;
                default:
                    if (python.TypeName != dotnet.TypeName || python.Text != dotnet.Text)
                        found.Add(new FieldDivergence(field, path, $"python {python.Render()} vs dotnet {dotnet.Render()}"));
                    return;
            }
        }

        /// <summary>
        ///     Compares two arrays by dtype, then shape, then bytes. A byte difference is reported per element: how many
        ///     differ, where the first ones are, both values with their bits, the ulp distance, and the largest absolute
        ///     difference. A dtype difference between float arrays also says whether casting Gym.NET's values to Gymnasium's
        ///     dtype would make them identical, which tells a missing cast from wrong values.
        /// </summary>
        /// <param name="field">The top-level field.</param>
        /// <param name="path">Where in the field.</param>
        /// <param name="python">Gymnasium's array.</param>
        /// <param name="dotnet">Gym.NET's array.</param>
        /// <param name="found">Receives the differences.</param>
        private static void CompareArrays(string field, string path, ValueFacts python, ValueFacts dotnet, List<FieldDivergence> found) {
            if (!python.Shape.SequenceEqual(dotnet.Shape)) {
                found.Add(new FieldDivergence(field, path + ".shape", $"python {ValueFacts.FormatShape(python.Shape)} vs dotnet {ValueFacts.FormatShape(dotnet.Shape)} (python {python.Render()}, dotnet {dotnet.Render()})"));
                return;
            }

            DTypeInfo pythonType = DTypeInfo.Parse(python.DType), dotnetType = DTypeInfo.Parse(dotnet.DType);
            if (python.DType != dotnet.DType) {
                found.Add(new FieldDivergence(field, path + ".dtype", $"python {python.DType} vs dotnet {dotnet.DType}{CastNote(python, pythonType, dotnet, dotnetType)} (python {python.Render()}, dotnet {dotnet.Render()})"));
                return;
            }

            if (python.Bytes.AsSpan().SequenceEqual(dotnet.Bytes))
                return;

            int size = Math.Max(pythonType.Size, 1);
            int count = python.Bytes.Length / size;
            var differing = new List<int>();
            for (int i = 0; i < count; i++) {
                if (!python.Bytes.AsSpan(i * size, size).SequenceEqual(dotnet.Bytes.AsSpan(i * size, size)))
                    differing.Add(i);
            }

            var parts = new List<string> { $"{differing.Count} of {count} elements differ" };
            foreach (int i in differing.Take(ElementsListed)) {
                parts.Add($"{Index(i, python.Shape)}: python {pythonType.Format(python.Bytes, i)} (0x{pythonType.Hex(python.Bytes, i)}) vs dotnet {dotnetType.Format(dotnet.Bytes, i)} (0x{dotnetType.Hex(dotnet.Bytes, i)}){UlpSuffix(pythonType, python.Bytes, dotnet.Bytes, i)}");
            }

            if (pythonType.Kind == 'f') {
                double largest = differing.Max(i => Math.Abs(pythonType.ReadDouble(python.Bytes, i) - dotnetType.ReadDouble(dotnet.Bytes, i)));
                parts.Add($"max |python - dotnet| = {ValueFacts.FormatNumber(largest)}");
            }

            found.Add(new FieldDivergence(field, path, string.Join("; ", parts)));
        }

        /// <summary>
        ///     For two float arrays of different widths, whether rounding the wider side to the narrower dtype reproduces the
        ///     narrower side's bytes. "yes" means the values are right and only the cast is missing (CartPole's float64
        ///     observation, B-13); "no" means the values differ as well.
        /// </summary>
        /// <param name="python">Gymnasium's array.</param>
        /// <param name="pythonType">Its parsed dtype.</param>
        /// <param name="dotnet">Gym.NET's array.</param>
        /// <param name="dotnetType">Its parsed dtype.</param>
        /// <returns>A note starting with "; ", or empty when the note does not apply.</returns>
        private static string CastNote(ValueFacts python, DTypeInfo pythonType, ValueFacts dotnet, DTypeInfo dotnetType) {
            // Only the float32/float64 pair is judged: it is the one Gymnasium's observations and Gym.NET's legacy code produce,
            // and rounding to float32 is then exactly NumPy's astype(float32) on the wider side and exact on the narrower one.
            bool floatPair = pythonType.Kind == 'f' && dotnetType.Kind == 'f' && pythonType.Size + dotnetType.Size == 12;
            if (!floatPair)
                return "";
            int count = python.Bytes.Length / pythonType.Size;
            bool narrowPython = pythonType.Size < dotnetType.Size;
            for (int i = 0; i < count; i++) {
                double p = pythonType.ReadDouble(python.Bytes, i), d = dotnetType.ReadDouble(dotnet.Bytes, i);
                bool equal = BitConverter.SingleToInt32Bits((float)p) == BitConverter.SingleToInt32Bits((float)d);
                if (!equal)
                    return $"; casting {(narrowPython ? "dotnet" : "python")} to {(narrowPython ? python.DType : dotnet.DType)} would still differ at {Index(i, python.Shape)}";
            }

            return $"; casting {(narrowPython ? "dotnet" : "python")} to {(narrowPython ? python.DType : dotnet.DType)} would make them identical";
        }

        /// <summary>
        ///     Compares two dictionaries: the same keys in the same order (Gymnasium's infos are insertion-ordered, and vector
        ///     envs rely on it), then each value.
        /// </summary>
        /// <param name="field">The top-level field.</param>
        /// <param name="path">Where in the field.</param>
        /// <param name="python">Gymnasium's dictionary.</param>
        /// <param name="dotnet">Gym.NET's dictionary.</param>
        /// <param name="found">Receives the differences.</param>
        private static void CompareDicts(string field, string path, ValueFacts python, ValueFacts dotnet, List<FieldDivergence> found) {
            List<string> pythonKeys = python.Entries.Select(e => e.Key.Canonical()).ToList();
            List<string> dotnetKeys = dotnet.Entries.Select(e => e.Key.Canonical()).ToList();
            if (!pythonKeys.SequenceEqual(dotnetKeys)) {
                var missing = pythonKeys.Except(dotnetKeys).ToList();
                var extra = dotnetKeys.Except(pythonKeys).ToList();
                string detail = missing.Count == 0 && extra.Count == 0
                    ? $"same keys in a different order: python [{string.Join(", ", pythonKeys)}] vs dotnet [{string.Join(", ", dotnetKeys)}]"
                    : $"keys differ: missing on dotnet [{string.Join(", ", missing)}], extra on dotnet [{string.Join(", ", extra)}]";
                found.Add(new FieldDivergence(field, path + ".keys", detail));
            }

            foreach (KeyValuePair<ValueFacts, ValueFacts> entry in python.Entries) {
                string key = entry.Key.Canonical();
                KeyValuePair<ValueFacts, ValueFacts> match = dotnet.Entries.FirstOrDefault(e => e.Key.Canonical() == key);
                if (match.Key != null)
                    CompareExact(field, $"{path}[{key}]", entry.Value, match.Value, found);
            }
        }

        /// <summary>The <see cref="FieldPolicy.Float64Value"/> comparison: both sides widened to double, compared by bits.</summary>
        /// <param name="field">The top-level field.</param>
        /// <param name="path">Where in the field.</param>
        /// <param name="python">Gymnasium's number.</param>
        /// <param name="dotnet">Gym.NET's number.</param>
        /// <param name="found">Receives the differences.</param>
        private static void CompareFloat64(string field, string path, ValueFacts python, ValueFacts dotnet, List<FieldDivergence> found) {
            double? p = AsDouble(python), d = AsDouble(dotnet);
            if (p is null || d is null) {
                found.Add(new FieldDivergence(field, path, $"python {python.Render()} ({python.TypeName}) vs dotnet {dotnet.Render()} ({dotnet.TypeName}): not both numbers"));
                return;
            }

            long pythonBits = BitConverter.DoubleToInt64Bits(p.Value), dotnetBits = BitConverter.DoubleToInt64Bits(d.Value);
            if (pythonBits == dotnetBits)
                return;
            BigInteger ulps = BigInteger.Abs((BigInteger)DTypeInfo.Ordered(pythonBits) - DTypeInfo.Ordered(dotnetBits));
            found.Add(new FieldDivergence(field, path,
                $"python {ValueFacts.FormatNumber(p.Value)} (0x{pythonBits:X16}, {python.TypeName} {python.DType}) vs dotnet {ValueFacts.FormatNumber(d.Value)} (0x{dotnetBits:X16}, {dotnet.TypeName} {dotnet.DType}); {ulps.ToString(CultureInfo.InvariantCulture)} ulp apart in float64"));
        }

        /// <summary>A number's value as a double: floats widen exactly, integers and booleans convert, 0-d arrays unwrap.</summary>
        /// <param name="value">The description.</param>
        /// <returns>The value, or <see langword="null"/> when it is not a number.</returns>
        private static double? AsDouble(ValueFacts value) {
            value = value.AsScalar();
            return value.Kind switch {
                FactKind.Float => DTypeInfo.Parse(value.DType).ReadDouble(value.Bytes, 0),
                FactKind.Int => (double)value.Integer,
                FactKind.Bool => value.Boolean ? 1.0 : 0.0,
                _ => null,
            };
        }

        /// <summary>The ulp distance of element <paramref name="index"/>, formatted for a report.</summary>
        /// <param name="dtype">The shared dtype.</param>
        /// <param name="python">Gymnasium's bytes.</param>
        /// <param name="dotnet">Gym.NET's bytes.</param>
        /// <param name="index">The element's flat index.</param>
        /// <returns><c>, N ulp</c>, or empty when the distance is undefined (not a float, or NaN).</returns>
        private static string UlpSuffix(DTypeInfo dtype, byte[] python, byte[] dotnet, int index) {
            BigInteger? ulps = dtype.Kind == 'f' ? dtype.Ulps(python, dotnet, index) : null;
            return ulps is { } distance ? $", {distance.ToString(CultureInfo.InvariantCulture)} ulp" : "";
        }

        /// <summary>Unravels a flat C-order index into NumPy's index notation: <c>[2]</c>, <c>[1, 0, 3]</c>.</summary>
        /// <param name="flat">The flat index.</param>
        /// <param name="shape">The array's shape.</param>
        /// <returns>The multi-index; <c>[]</c> for a 0-d array.</returns>
        private static string Index(int flat, IReadOnlyList<long> shape) {
            var index = new long[shape.Count];
            long rest = flat;
            for (int axis = shape.Count - 1; axis >= 0; axis--) {
                long extent = Math.Max(shape[axis], 1);
                index[axis] = rest % extent;
                rest /= extent;
            }

            return "[" + string.Join(", ", index) + "]";
        }
    }
}
