#nullable enable
using System;
using NumSharp;

namespace Gym.Environments.Envs {
    /// <summary>The kinds of number NumPy 2's scalar arithmetic tells apart (NEP 50), weakest first.</summary>
    internal enum NpKind {
        /// <summary>A Python <c>int</c>: weak, it takes the other operand's type.</summary>
        PyInt,

        /// <summary>A Python <c>float</c> (a C double): weak against NumPy floats, so <c>np.float32(x) * 0.5</c> computes in float32.</summary>
        PyFloat,

        /// <summary>A NumPy <c>float32</c>: strong, but narrower than <see cref="Float64"/>.</summary>
        Float32,

        /// <summary>A NumPy <c>float64</c>: strong.</summary>
        Float64,
    }

    /// <summary>
    ///     A Python or NumPy scalar as NumPy 2 computes with it (NEP 50), for env code whose precision changes with the
    ///     operands' types at run time. Gymnasium's MountainCarContinuous is the case in point: its state is float64 after a
    ///     reset and float32 after a step, a clamp turns its velocity into a Python float, and every operation's precision
    ///     follows: <c>np.float32 * 0.0015</c> is a float32 product, <c>np.float64 + np.float32</c> a float64 sum, and a
    ///     comparison of a float32 with a Python float is made in float32 (<c>np.float32(0.07) == 0.07</c> is <c>True</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The result kind of a binary operation is the stronger of the two (<see cref="NpKind"/>'s order); a float32 result
    ///     rounds both operands to float32 first and computes in float32, everything else computes in double. That is the
    ///     whole of NEP 50 for these four kinds, verified against NumPy 2.4.2 operation by operation; integer kinds other
    ///     than Python's <c>int</c> are not modelled.
    ///     </para>
    ///     <para>
    ///     <see cref="Value"/> holds the exact number (a float32 value widened, which is exact). Value semantics: a struct,
    ///     copied freely; assignment in Python rebinds the name to another object, which is what reassigning one does here.
    ///     </para>
    /// </remarks>
    internal readonly struct NpScalar {
        /// <summary>A scalar of a kind; a float32 value is rounded to float32 on the way in.</summary>
        /// <param name="value">The value.</param>
        /// <param name="kind">The kind.</param>
        private NpScalar(double value, NpKind kind) {
            Value = kind == NpKind.Float32 ? (float)value : value;
            Kind = kind;
        }

        /// <summary>The number, exactly (a float32 value is stored widened).</summary>
        public double Value { get; }

        /// <summary>Which Python or NumPy type the number has.</summary>
        public NpKind Kind { get; }

        /// <summary>A Python <c>int</c>.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The scalar.</returns>
        public static NpScalar PyInt(long value) => new(value, NpKind.PyInt);

        /// <summary>A Python <c>float</c>.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The scalar.</returns>
        public static NpScalar PyFloat(double value) => new(value, NpKind.PyFloat);

        /// <summary>A NumPy <c>float32</c>.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The scalar.</returns>
        public static NpScalar Float32(float value) => new(value, NpKind.Float32);

        /// <summary>A NumPy <c>float64</c>.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The scalar.</returns>
        public static NpScalar Float64(double value) => new(value, NpKind.Float64);

        /// <summary>
        ///     An element of an array as NumPy's indexing returns it: a <c>float32</c> or <c>float64</c> scalar for those
        ///     dtypes, and for any other dtype a <c>float64</c> holding the element's value (the env code using this only
        ///     reads float actions; an integer array's element would be an integer scalar in NumPy).
        /// </summary>
        /// <param name="array">The array; only read.</param>
        /// <param name="index">The flat index.</param>
        /// <returns>The scalar.</returns>
        public static NpScalar Element(NDArray array, long index) => array.typecode switch {
            NPTypeCode.Single => Float32(array.GetAtIndex<float>(index)),
            NPTypeCode.Double => Float64(array.GetAtIndex<double>(index)),
            _ => WidenedElement(array, index),
        };

        /// <summary>An element of an array of another dtype, as a float64 holding its value.</summary>
        /// <param name="array">The array; only read.</param>
        /// <param name="index">The flat index.</param>
        /// <returns>The scalar.</returns>
        private static NpScalar WidenedElement(NDArray array, long index) {
            using NDArray widened = array.astype(np.float64);
            return Float64(widened.GetAtIndex<double>(index));
        }

        /// <summary><c>a + b</c> under NEP 50.</summary>
        /// <param name="a">The left operand.</param>
        /// <param name="b">The right operand.</param>
        /// <returns>The sum, of the stronger kind.</returns>
        public static NpScalar operator +(NpScalar a, NpScalar b) => Combine(a, b, (x, y) => x + y, (x, y) => x + y);

        /// <summary><c>a - b</c> under NEP 50.</summary>
        /// <param name="a">The left operand.</param>
        /// <param name="b">The right operand.</param>
        /// <returns>The difference, of the stronger kind.</returns>
        public static NpScalar operator -(NpScalar a, NpScalar b) => Combine(a, b, (x, y) => x - y, (x, y) => x - y);

        /// <summary><c>a * b</c> under NEP 50.</summary>
        /// <param name="a">The left operand.</param>
        /// <param name="b">The right operand.</param>
        /// <returns>The product, of the stronger kind.</returns>
        public static NpScalar operator *(NpScalar a, NpScalar b) => Combine(a, b, (x, y) => x * y, (x, y) => x * y);

        /// <summary><c>a / b</c> under NEP 50 (true division: two Python ints give a Python float).</summary>
        /// <param name="a">The left operand.</param>
        /// <param name="b">The right operand.</param>
        /// <returns>The quotient.</returns>
        /// <exception cref="ZeroDivisionError">Thrown for a Python-number zero divisor, as Python raises <c>ZeroDivisionError</c> (NumPy scalars divide to inf or NaN instead).</exception>
        public static NpScalar operator /(NpScalar a, NpScalar b) {
            if (a.Kind <= NpKind.PyFloat && b.Kind <= NpKind.PyFloat) {
                if (b.Value == 0)
                    throw new ZeroDivisionError("float division by zero");
                return PyFloat(a.Value / b.Value);
            }

            return Combine(a, b, (x, y) => x / y, (x, y) => x / y);
        }

        /// <summary><c>-a</c>, keeping the kind.</summary>
        /// <param name="a">The operand.</param>
        /// <returns>The negation.</returns>
        public static NpScalar operator -(NpScalar a) => new(-a.Value, a.Kind);

        /// <summary><c>a &gt; b</c>, compared in the precision NEP 50 promotes to (float32 when the stronger kind is float32).</summary>
        /// <param name="a">The left operand.</param>
        /// <param name="b">The right operand.</param>
        /// <returns>The comparison.</returns>
        public static bool operator >(NpScalar a, NpScalar b) => Compare(a, b) > 0;

        /// <summary><c>a &lt; b</c>, compared in the promoted precision.</summary>
        /// <param name="a">The left operand.</param>
        /// <param name="b">The right operand.</param>
        /// <returns>The comparison.</returns>
        public static bool operator <(NpScalar a, NpScalar b) => Compare(a, b) < 0;

        /// <summary><c>a &gt;= b</c>, compared in the promoted precision (false when either is NaN).</summary>
        /// <param name="a">The left operand.</param>
        /// <param name="b">The right operand.</param>
        /// <returns>The comparison.</returns>
        public static bool operator >=(NpScalar a, NpScalar b) {
            (double x, double y) = Promoted(a, b);
            return x >= y;
        }

        /// <summary><c>a &lt;= b</c>, compared in the promoted precision (false when either is NaN).</summary>
        /// <param name="a">The left operand.</param>
        /// <param name="b">The right operand.</param>
        /// <returns>The comparison.</returns>
        public static bool operator <=(NpScalar a, NpScalar b) {
            (double x, double y) = Promoted(a, b);
            return x <= y;
        }

        /// <summary><c>a == b</c> in the promoted precision (Python's <c>==</c> on these numbers; false when either is NaN).</summary>
        /// <param name="other">The other operand.</param>
        /// <returns>The comparison.</returns>
        public bool PyEquals(NpScalar other) {
            (double x, double y) = Promoted(this, other);
            return x == y;
        }

        /// <summary><c>math.cos(x)</c>: the value as a C double, then the C runtime's cosine; a Python float.</summary>
        /// <returns>The cosine.</returns>
        public NpScalar MathCos() => PyFloat(Math.Cos(Value));

        /// <summary><c>math.pow(x, y)</c>: both as C doubles, then the C runtime's <c>pow</c> (not <c>x * x</c> for <c>y = 2</c>); a Python float.</summary>
        /// <param name="exponent">The exponent.</param>
        /// <returns>The power.</returns>
        public NpScalar MathPow(double exponent) => PyFloat(Math.Pow(Value, exponent));

        /// <summary>
        ///     <c>np.sin(x)</c> on this scalar: for a float32 NumPy's own float32 sine (NumSharp ports it bit for bit; the C
        ///     runtime's <c>sinf</c> differs in about one argument in ten), for anything else the double sine, as a float64.
        /// </summary>
        /// <returns>The sine, of kind float32 or float64.</returns>
        public NpScalar NpSin() {
            if (Kind == NpKind.Float32) {
                using NDArray zero = NDArray.Scalar((float)Value);
                using NDArray sine = np.sin(zero);
                return Float32(sine.GetAtIndex<float>(0));
            }

            return Float64(Math.Sin(Value));
        }

        /// <summary>
        ///     <c>np.cos(x)</c> on this scalar: for a float32 NumPy's own float32 cosine (through NumSharp, which matches it bit
        ///     for bit where the C runtime's <c>cosf</c> does not), for anything else the double cosine, as a float64.
        /// </summary>
        /// <returns>The cosine, of kind float32 or float64.</returns>
        public NpScalar NpCos() {
            if (Kind == NpKind.Float32) {
                using NDArray zero = NDArray.Scalar((float)Value);
                using NDArray cosine = np.cos(zero);
                return Float32(cosine.GetAtIndex<float>(0));
            }

            return Float64(Math.Cos(Value));
        }

        /// <summary>
        ///     <c>x ** exponent</c> on a NumPy scalar with a Python <c>int</c> exponent: a float32 computes <c>powf</c> in float32
        ///     (the exponent is weak), anything else <c>pow</c> in double. Never <c>x * x</c>: NumPy calls the C runtime's power
        ///     function, and so does this.
        /// </summary>
        /// <param name="exponent">The exponent.</param>
        /// <returns>The power, of this scalar's kind (a Python <c>float</c> base stays one: Python's own <c>**</c> is also <c>pow</c>).</returns>
        /// <exception cref="InvalidOperationException">Thrown for a Python <c>int</c> base, whose <c>**</c> is exact integer arithmetic and not modelled.</exception>
        public NpScalar NpPow(int exponent) {
            if (Kind == NpKind.PyInt)
                throw new InvalidOperationException("A Python int's ** is exact integer arithmetic, which NpScalar does not model.");
            if (Kind == NpKind.Float32)
                return Float32(MathF.Pow((float)Value, exponent));
            return new NpScalar(Math.Pow(Value, exponent), Kind);
        }

        /// <summary>
        ///     <c>np.abs(x)</c>: the magnitude, keeping a NumPy scalar's kind; a Python <c>float</c> gives a float64, as NumPy
        ///     converts it (a Python <c>int</c> would give an int64, which is not modelled: it gives a float64 too).
        /// </summary>
        /// <returns>The absolute value.</returns>
        public NpScalar Abs() => new(Math.Abs(Value), Kind <= NpKind.PyFloat ? NpKind.Float64 : Kind);

        /// <summary>The value rounded to float32, as <c>np.array(..., dtype=np.float32)</c> stores it.</summary>
        /// <returns>The float32 value.</returns>
        public float ToFloat32() => (float)Value;

        /// <summary>Python's <c>repr</c> kind and value, for debugging.</summary>
        /// <returns>The text.</returns>
        public override string ToString() => $"{Kind}({Value:R})";

        /// <summary>Python's <c>max(a, b)</c>: <paramref name="b"/> when <c>b &gt; a</c>, else <paramref name="a"/>, keeping the winner's kind.</summary>
        /// <param name="a">The first candidate.</param>
        /// <param name="b">The second candidate.</param>
        /// <returns>The winner, as the object it is.</returns>
        public static NpScalar PyMax(NpScalar a, NpScalar b) => b > a ? b : a;

        /// <summary>Python's <c>min(a, b)</c>: <paramref name="b"/> when <c>b &lt; a</c>, else <paramref name="a"/>, keeping the winner's kind.</summary>
        /// <param name="a">The first candidate.</param>
        /// <param name="b">The second candidate.</param>
        /// <returns>The winner, as the object it is.</returns>
        public static NpScalar PyMin(NpScalar a, NpScalar b) => b < a ? b : a;

        /// <summary>Applies a binary operation in the precision of the stronger kind, which the result has.</summary>
        /// <param name="a">The left operand.</param>
        /// <param name="b">The right operand.</param>
        /// <param name="single">The float32 operation.</param>
        /// <param name="wide">The double operation.</param>
        /// <returns>The result.</returns>
        private static NpScalar Combine(NpScalar a, NpScalar b, Func<float, float, float> single, Func<double, double, double> wide) {
            NpKind kind = a.Kind > b.Kind ? a.Kind : b.Kind;
            if (kind == NpKind.Float32)
                // The weak operand is cast to float32 first, then the operation rounds in float32.
                return Float32(single((float)a.Value, (float)b.Value));
            if (kind == NpKind.PyInt)
                return PyInt((long)wide(a.Value, b.Value));
            return new NpScalar(wide(a.Value, b.Value), kind);
        }

        /// <summary>Both operands in the precision NEP 50 compares them in.</summary>
        /// <param name="a">The left operand.</param>
        /// <param name="b">The right operand.</param>
        /// <returns>The two values, rounded to float32 when the stronger kind is float32.</returns>
        private static (double X, double Y) Promoted(NpScalar a, NpScalar b) {
            NpKind kind = a.Kind > b.Kind ? a.Kind : b.Kind;
            return kind == NpKind.Float32 ? ((float)a.Value, (float)b.Value) : (a.Value, b.Value);
        }

        /// <summary>A three-way comparison in the promoted precision; NaN compares as neither greater nor less.</summary>
        /// <param name="a">The left operand.</param>
        /// <param name="b">The right operand.</param>
        /// <returns>1, -1, or 0 (equal or unordered).</returns>
        private static int Compare(NpScalar a, NpScalar b) {
            (double x, double y) = Promoted(a, b);
            return x > y ? 1 : x < y ? -1 : 0;
        }
    }
}
