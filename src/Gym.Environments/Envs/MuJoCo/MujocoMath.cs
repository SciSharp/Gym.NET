using System;
using Gym.MuJoCo;
using NumSharp;

namespace Gym.Environments.Envs.MuJoCo {
    /// <summary>
    ///     NumPy's arithmetic as Gymnasium's MuJoCo envs run it, where it decides bits: <c>np.sum</c>'s pairwise reduction,
    ///     <c>np.linalg.norm</c> of 2 or 3 values, the control cost's <c>np.sum(np.square(action))</c> in the action's own
    ///     dtype, and the NumPy scalar types an info value carries.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Every rule here was measured against NumPy 2.4.2 on win-x64, bit for bit over random trials, and the trace
    ///         checker (<c>tools/parity/mujoco/halfcheetah_poc</c>) holds the rewards built from them to Gymnasium's over 49
    ///         traces: <c>np.sum</c> is the identity 0 plus NumPy's pairwise sum (a plain loop below 8 entries, 8
    ///         accumulators combined as a tree up to 128, halves beyond); a float32 array sums in float32; <c>np.linalg.norm</c>
    ///         of 2 or 3 float64 values is the square root of their squares added left to right.
    ///     </para>
    ///     <para>
    ///         Footgun: these reproduce NumPy only for the shapes Gymnasium's envs pass (one-dimensional or C-contiguous
    ///         arrays reduced whole); a strided or partial reduction pairs its terms differently in NumPy.
    ///     </para>
    /// </remarks>
    internal static class MujocoMath {
        /// <summary>NumPy's <c>PW_BLOCKSIZE</c>: the longest run its pairwise sum adds with 8 accumulators before halving.</summary>
        private const int PairwiseBlock = 128;

        /// <summary>
        ///     NumPy's float64 pairwise sum of <paramref name="values"/> (<c>DOUBLE_pairwise_sum</c>), without the reduction's
        ///     identity: callers add it to 0 as <see cref="Sum(ReadOnlySpan{double})"/> does.
        /// </summary>
        /// <param name="values">The values, in memory order.</param>
        /// <returns>The sum, paired as NumPy pairs it.</returns>
        internal static double PairwiseSum(ReadOnlySpan<double> values) {
            int n = values.Length;
            if (n < 8) {
                double small = 0.0;
                for (int i = 0; i < n; i++)
                    small += values[i];
                return small;
            }
            if (n <= PairwiseBlock) {
                double r0 = values[0], r1 = values[1], r2 = values[2], r3 = values[3];
                double r4 = values[4], r5 = values[5], r6 = values[6], r7 = values[7];
                int i = 8;
                for (; i < n - n % 8; i += 8) {
                    r0 += values[i];
                    r1 += values[i + 1];
                    r2 += values[i + 2];
                    r3 += values[i + 3];
                    r4 += values[i + 4];
                    r5 += values[i + 5];
                    r6 += values[i + 6];
                    r7 += values[i + 7];
                }
                // NumPy combines the accumulators as a fixed tree before adding the tail one by one
                double res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
                for (; i < n; i++)
                    res += values[i];
                return res;
            }
            // halves, the first rounded down to a multiple of the unroll factor, as NumPy splits them
            int n2 = n / 2;
            n2 -= n2 % 8;
            return PairwiseSum(values[..n2]) + PairwiseSum(values[n2..]);
        }

        /// <summary>
        ///     NumPy's float32 pairwise sum (<c>FLOAT_pairwise_sum</c>): the float64 algorithm with float32 accumulators, so
        ///     every partial sum rounds to float32.
        /// </summary>
        /// <param name="values">The values, in memory order.</param>
        /// <returns>The float32 sum, without the reduction's identity.</returns>
        internal static float PairwiseSum(ReadOnlySpan<float> values) {
            int n = values.Length;
            if (n < 8) {
                float small = 0f;
                for (int i = 0; i < n; i++)
                    small += values[i];
                return small;
            }
            if (n <= PairwiseBlock) {
                float r0 = values[0], r1 = values[1], r2 = values[2], r3 = values[3];
                float r4 = values[4], r5 = values[5], r6 = values[6], r7 = values[7];
                int i = 8;
                for (; i < n - n % 8; i += 8) {
                    r0 += values[i];
                    r1 += values[i + 1];
                    r2 += values[i + 2];
                    r3 += values[i + 3];
                    r4 += values[i + 4];
                    r5 += values[i + 5];
                    r6 += values[i + 6];
                    r7 += values[i + 7];
                }
                float res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
                for (; i < n; i++)
                    res += values[i];
                return res;
            }
            int n2 = n / 2;
            n2 -= n2 % 8;
            return PairwiseSum(values[..n2]) + PairwiseSum(values[n2..]);
        }

        /// <summary>
        ///     <c>np.sum</c> of a float64 array: the reduction's identity 0 plus the pairwise sum, so a sum of negative zeros
        ///     is +0, as NumPy's is.
        /// </summary>
        /// <param name="values">The values.</param>
        /// <returns>The sum.</returns>
        internal static double Sum(ReadOnlySpan<double> values) => 0.0 + PairwiseSum(values);

        /// <summary>
        ///     <c>np.sum(np.square(values))</c> of a float64 array (and <c>np.square(values).sum()</c>, the same reduction):
        ///     each square rounded, then <see cref="Sum(ReadOnlySpan{double})"/>.
        /// </summary>
        /// <param name="values">The values, for example the external forces <c>cfrc_ext</c> row by row.</param>
        /// <returns>The sum of squares.</returns>
        internal static double SumOfSquares(ReadOnlySpan<double> values) {
            // a contact force array is at most a few hundred values; a heap buffer keeps big models off the stack
            double[] squares = new double[values.Length];
            for (int i = 0; i < values.Length; i++)
                squares[i] = values[i] * values[i];
            return Sum(squares);
        }

        /// <summary>
        ///     <c>np.sum(np.square(np.clip(values, low, high)))</c>: Ant's contact cost, each value clipped as <c>np.clip</c>
        ///     clips (a NaN stays NaN) before it is squared.
        /// </summary>
        /// <param name="values">The values.</param>
        /// <param name="low">The lower bound.</param>
        /// <param name="high">The upper bound.</param>
        /// <returns>The sum of the clipped squares.</returns>
        internal static double SumOfClippedSquares(ReadOnlySpan<double> values, double low, double high) {
            double[] squares = new double[values.Length];
            for (int i = 0; i < values.Length; i++) {
                double clipped = Gym.Utils.PyMath.Clip(values[i], low, high);
                squares[i] = clipped * clipped;
            }
            return Sum(squares);
        }

        /// <summary><c>np.linalg.norm</c> of two float64 values: <c>sqrt(x*x + y*y)</c>, as NumPy's <c>sqrt(x.dot(x))</c> rounds it.</summary>
        /// <param name="x">The first value.</param>
        /// <param name="y">The second value.</param>
        /// <returns>The Euclidean norm.</returns>
        internal static double Norm(double x, double y) => Math.Sqrt(x * x + y * y);

        /// <summary><c>np.linalg.norm</c> of three float64 values: <c>sqrt((x*x + y*y) + z*z)</c>.</summary>
        /// <param name="x">The first value.</param>
        /// <param name="y">The second value.</param>
        /// <param name="z">The third value.</param>
        /// <returns>The Euclidean norm.</returns>
        internal static double Norm(double x, double y, double z) => Math.Sqrt(x * x + y * y + z * z);

        /// <summary>
        ///     <c>np.sum(np.square(action))</c> in the action's own dtype, as Gymnasium's control costs compute it: float32
        ///     actions (the action space's dtype) square and sum in float32, float64 actions in float64, integer actions in
        ///     their own width with NumPy's wraparound and sum as int64 (uint64 when unsigned), booleans count.
        /// </summary>
        /// <param name="action">The action as the step received it.</param>
        /// <returns>The NumPy scalar the sum is.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="action"/> is null.</exception>
        /// <exception cref="NotSupportedException">
        ///     A float16, complex, decimal, char or string action: NumPy keeps such costs in that dtype, which Gym.NET's
        ///     rewards (float64, port plan §5) and info scalars do not model.
        /// </exception>
        internal static SquaredSum SumOfSquares(NDArray action) {
            ArgumentNullException.ThrowIfNull(action);
            long n = action.size;
            switch (action.typecode) {
                case NPTypeCode.Single: {
                    float[] squares = new float[n];
                    using (NDArray flat = action.flatten())
                        for (long i = 0; i < n; i++) {
                            float value = flat.GetSingle(i);
                            squares[i] = value * value;
                        }
                    // the reduction's identity, 0, in float32: a sum of negative zeros is +0
                    return SquaredSum.OfFloat32(0f + PairwiseSum(squares));
                }
                case NPTypeCode.Double: {
                    double[] squares = new double[n];
                    using (NDArray flat = action.flatten())
                        for (long i = 0; i < n; i++) {
                            double value = flat.GetDouble(i);
                            squares[i] = value * value;
                        }
                    return SquaredSum.OfFloat64(Sum(squares));
                }
                case NPTypeCode.Boolean: {
                    // np.square turns booleans into int8 0 and 1, whose int64 sum counts the true elements
                    long count = 0;
                    using (NDArray flat = action.flatten())
                        for (long i = 0; i < n; i++)
                            if (flat.GetBoolean(i))
                                count++;
                    return SquaredSum.OfInt64(count);
                }
                case NPTypeCode.SByte or NPTypeCode.Int16 or NPTypeCode.Int32 or NPTypeCode.Int64:
                    return SquaredSum.OfInt64(SignedSquares(action));
                case NPTypeCode.Byte or NPTypeCode.UInt16 or NPTypeCode.UInt32 or NPTypeCode.UInt64:
                    return SquaredSum.OfUInt64(UnsignedSquares(action));
                default:
                    throw new NotSupportedException(
                        $"A MuJoCo env's control cost of a {action.dtype} action is not modelled: NumPy keeps it in that dtype. "
                        + "Pass float32 actions (the action space's dtype) or float64 ones.");
            }
        }

        /// <summary>
        ///     The squares of a signed integer action, each wrapped to the action's own width as NumPy's <c>np.square</c>
        ///     wraps it, summed as int64 with wraparound.
        /// </summary>
        /// <param name="action">An int8, int16, int32 or int64 action.</param>
        /// <returns>The int64 sum.</returns>
        private static long SignedSquares(NDArray action) {
            long sum = 0;
            using NDArray flat = action.flatten();
            for (long i = 0; i < flat.size; i++) {
                long square = action.typecode switch {
                    NPTypeCode.SByte => unchecked((sbyte)(flat.GetSByte(i) * flat.GetSByte(i))),
                    NPTypeCode.Int16 => unchecked((short)(flat.GetInt16(i) * flat.GetInt16(i))),
                    NPTypeCode.Int32 => unchecked(flat.GetInt32(i) * flat.GetInt32(i)),
                    _ => unchecked(flat.GetInt64(i) * flat.GetInt64(i)),
                };
                sum = unchecked(sum + square);
            }
            return sum;
        }

        /// <summary>
        ///     The squares of an unsigned integer action, each wrapped to the action's own width, summed as uint64 with
        ///     wraparound.
        /// </summary>
        /// <param name="action">A uint8, uint16, uint32 or uint64 action.</param>
        /// <returns>The uint64 sum.</returns>
        private static ulong UnsignedSquares(NDArray action) {
            ulong sum = 0;
            using NDArray flat = action.flatten();
            for (long i = 0; i < flat.size; i++) {
                ulong square = action.typecode switch {
                    NPTypeCode.Byte => unchecked((byte)(flat.GetByte(i) * flat.GetByte(i))),
                    NPTypeCode.UInt16 => unchecked((ushort)(flat.GetUInt16(i) * flat.GetUInt16(i))),
                    NPTypeCode.UInt32 => unchecked(flat.GetUInt32(i) * flat.GetUInt32(i)),
                    _ => unchecked(flat.GetUInt64(i) * flat.GetUInt64(i)),
                };
                sum = unchecked(sum + square);
            }
            return sum;
        }

        /// <summary>
        ///     An info value as Gymnasium's dict holds it (DR-08): a NumPy float32 or float64 scalar as a 0-d array of that
        ///     dtype, a Python <c>float</c> as a <see langword="double"/>, a Python <c>int</c> as a <see langword="long"/>.
        /// </summary>
        /// <param name="value">The scalar and its kind.</param>
        /// <returns>The value to store; a 0-d array belongs to the info that holds it.</returns>
        internal static object InfoValue(NpScalar value) => value.Kind switch {
            // each arm cast to object: NDArray converts implicitly from double and long, so without the casts the switch's
            // natural type would be NDArray and the Python numbers would turn into 0-d arrays
            NpKind.Float32 => (object)NDArray.Scalar((float)value.Value),
            NpKind.Float64 => (object)NDArray.Scalar(value.Value),
            NpKind.PyFloat => (object)value.Value,
            _ => (object)(long)value.Value,
        };

        /// <summary>An <c>np.float64</c> info value, such as <c>data.qpos[0]</c>: a 0-d float64 array (DR-08).</summary>
        /// <param name="value">The value.</param>
        /// <returns>The 0-d array, which belongs to the info that holds it.</returns>
        internal static NDArray Float64(double value) => NDArray.Scalar(value);

        /// <summary>A copy of an engine array as a new float64 array, as <c>data.ten_length.copy()</c> makes one.</summary>
        /// <param name="values">The engine's values.</param>
        /// <returns>The new array, which the caller owns.</returns>
        internal static NDArray Copy(double[] values) => np.array((double[])values.Clone());

        /// <summary>
        ///     The center of mass in the world's x and y (Gymnasium's <c>mass_center</c>):
        ///     <c>einsum("b,bj->j", body_mass, xipos)</c>, each column summed in body order from 0, over
        ///     <c>body_mass.sum()</c>.
        /// </summary>
        /// <param name="model">The model, for the body masses.</param>
        /// <param name="data">The simulation, whose last forward pass set the bodies' inertial frames.</param>
        /// <returns>The x and y of the center of mass.</returns>
        internal static (double X, double Y) MassCenter(MjModel model, MjData data) {
            double denominator = Sum(model.body_mass);
            double x = 0, y = 0;
            for (int b = 0; b < model.nbody; b++) {
                x += model.body_mass[b] * data.xipos[3 * b];
                y += model.body_mass[b] * data.xipos[3 * b + 1];
            }
            return (x / denominator, y / denominator);
        }
    }

    /// <summary>
    ///     The NumPy scalar <c>np.sum(np.square(action))</c> gives, kept in its own type until a weight meets it: float32,
    ///     float64, int64 or uint64 (<see cref="MujocoMath.SumOfSquares(NDArray)"/>).
    /// </summary>
    /// <remarks>
    ///     Footgun: Gymnasium's weights are Python numbers, and an integer weight keeps an integer action's cost an integer
    ///     there. Gym.NET's weights are doubles, which act as Python floats; the one integer default that changes a cost's
    ///     type, Reacher-v5's <c>reward_control_weight=1</c>, is modelled by that env itself. Float actions are unaffected.
    /// </remarks>
    internal readonly struct SquaredSum {
        /// <summary>The sum of a float32 or float64 action.</summary>
        private readonly double real;

        /// <summary>The sum of a signed integer or boolean action.</summary>
        private readonly long signed;

        /// <summary>The sum of an unsigned integer action.</summary>
        private readonly ulong unsigned;

        /// <summary>Which of the three holds the value, and the NumPy type it has.</summary>
        private readonly NPTypeCode type;

        /// <summary>Holds one sum.</summary>
        /// <param name="type">The NumPy type: float32, float64, int64 or uint64.</param>
        /// <param name="real">The float sum.</param>
        /// <param name="signed">The int64 sum.</param>
        /// <param name="unsigned">The uint64 sum.</param>
        private SquaredSum(NPTypeCode type, double real, long signed, ulong unsigned) {
            this.type = type;
            this.real = real;
            this.signed = signed;
            this.unsigned = unsigned;
        }

        /// <summary>A float32 sum.</summary>
        /// <param name="value">The sum.</param>
        /// <returns>The scalar.</returns>
        internal static SquaredSum OfFloat32(float value) => new(NPTypeCode.Single, value, 0, 0);

        /// <summary>A float64 sum.</summary>
        /// <param name="value">The sum.</param>
        /// <returns>The scalar.</returns>
        internal static SquaredSum OfFloat64(double value) => new(NPTypeCode.Double, value, 0, 0);

        /// <summary>An int64 sum.</summary>
        /// <param name="value">The sum.</param>
        /// <returns>The scalar.</returns>
        internal static SquaredSum OfInt64(long value) => new(NPTypeCode.Int64, 0, value, 0);

        /// <summary>A uint64 sum.</summary>
        /// <param name="value">The sum.</param>
        /// <returns>The scalar.</returns>
        internal static SquaredSum OfUInt64(ulong value) => new(NPTypeCode.UInt64, 0, 0, value);

        /// <summary>Whether the sum is an integer one (<c>np.int64</c> or <c>np.uint64</c>): the action was an integer or boolean array.</summary>
        internal bool IsInteger => type is NPTypeCode.Int64 or NPTypeCode.UInt64;

        /// <summary>
        ///     <c>-sum</c>, NumPy's negation in the sum's own type: integers wrap (an unsigned sum becomes <c>2**64 - sum</c>),
        ///     floats change sign. Reacher and Pusher negate before weighting.
        /// </summary>
        /// <returns>The negated sum.</returns>
        internal SquaredSum Negate() => type switch {
            NPTypeCode.Int64 => OfInt64(unchecked(-signed)),
            NPTypeCode.UInt64 => OfUInt64(unchecked(0UL - unsigned)),
            NPTypeCode.Single => OfFloat32(-(float)real),
            _ => OfFloat64(-real),
        };

        /// <summary>
        ///     <c>weight * sum</c> with a Python <c>float</c> weight, under NEP 50: a float32 sum stays float32 (the weight
        ///     rounded to float32 first), anything else becomes float64 (an integer sum converted, then multiplied).
        /// </summary>
        /// <param name="weight">The weight.</param>
        /// <returns>The product as the NumPy scalar it is.</returns>
        internal NpScalar Times(double weight) => type switch {
            NPTypeCode.Single => NpScalar.PyFloat(weight) * NpScalar.Float32((float)real),
            NPTypeCode.Double => NpScalar.PyFloat(weight) * NpScalar.Float64(real),
            NPTypeCode.Int64 => NpScalar.Float64((double)signed * weight),
            _ => NpScalar.Float64((double)unsigned * weight),
        };

        /// <summary>
        ///     The sum as an operand of float arithmetic (Reacher-v4's unweighted <c>reward_dist + reward_ctrl</c>): a float32 or
        ///     float64 scalar as it is, an integer one as the float64 NumPy converts it to when it meets a float64.
        /// </summary>
        /// <returns>The scalar.</returns>
        internal NpScalar AsOperand() => type switch {
            NPTypeCode.Single => NpScalar.Float32((float)real),
            NPTypeCode.Double => NpScalar.Float64(real),
            NPTypeCode.Int64 => NpScalar.Float64(signed),
            _ => NpScalar.Float64(unsigned),
        };

        /// <summary>
        ///     The sum as an info value (DR-08): a 0-d array of its NumPy type, <c>np.float32</c>, <c>np.float64</c>,
        ///     <c>np.int64</c> or <c>np.uint64</c>.
        /// </summary>
        /// <returns>The 0-d array, which belongs to the info that holds it.</returns>
        internal NDArray InfoValue() => type switch {
            NPTypeCode.Single => NDArray.Scalar((float)real),
            NPTypeCode.Double => NDArray.Scalar(real),
            NPTypeCode.Int64 => NDArray.Scalar(signed),
            _ => NDArray.Scalar(unsigned),
        };
    }
}
