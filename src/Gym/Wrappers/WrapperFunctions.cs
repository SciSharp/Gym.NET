#nullable enable
using System;
using System.Linq;
using Gym.Utils;
using NumSharp;

namespace Gym.Wrappers {
    /// <summary>
    ///     A running mean and variance (<c>RunningMeanStd</c>), updated batch by batch with Chan et al.'s parallel algorithm:
    ///     what <see cref="NormalizeObservation{TObs, TAct}"/> and <see cref="NormalizeReward{TObs, TAct}"/> normalize by.
    ///     Byte for byte Gymnasium's, NumPy 2's dtype rules included: the statistics keep their dtype (float32 for an
    ///     observation normalizer) until a batch of a wider dtype promotes them, and the count stays a float64.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The arrays are this object's: an update replaces them with new ones and disposes the old, so a reference taken
    ///     from <see cref="Mean"/> or <see cref="Var"/> is valid only until the next update. Not thread-safe.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/utils.py#L33">gymnasium/wrappers/utils.py</see>.</para>
    /// </remarks>
    public sealed class RunningMeanStd : IDisposable {
        /// <summary>Starts with mean 0, variance 1 and a count of <paramref name="epsilon"/>, which keeps the first update's division away from zero.</summary>
        /// <param name="epsilon">The initial count, 1e-4 by default.</param>
        /// <param name="shape">The statistics' shape, <c>()</c> by default (a scalar, as for rewards).</param>
        /// <param name="dtype">The statistics' dtype, float64 by default.</param>
        public RunningMeanStd(double epsilon = 1e-4, Shape? shape = null, DType? dtype = null) {
            Shape statsShape = shape ?? new Shape();
            DType statsDtype = dtype ?? np.float64;
            Mean = np.zeros(statsShape, statsDtype);
            Var = np.ones(statsShape, statsDtype);
            Count = epsilon;
        }

        /// <summary>The running mean (<c>mean</c>); owned by this object and replaced by every update.</summary>
        public NDArray Mean { get; set; }

        /// <summary>The running variance (<c>var</c>); owned by this object and replaced by every update.</summary>
        public NDArray Var { get; set; }

        /// <summary>The number of samples seen, plus the initial epsilon (<c>count</c>), a Python float.</summary>
        public double Count { get; set; }

        /// <summary>
        ///     Folds a batch in (<c>update(x)</c>): its mean and variance along the first axis, and its length as the count.
        /// </summary>
        /// <param name="x">The batch, <c>(k, *shape)</c>; left to the caller.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="x"/> is <see langword="null"/>.</exception>
        public void Update(NDArray x) {
            if (x is null)
                throw new ArgumentNullException(nameof(x));
            using NDArray batchMean = np.mean(x, axis: 0);
            using NDArray batchVar = np.var(x, axis: 0);
            UpdateFromMoments(batchMean, batchVar, x.shape[0]);
        }

        /// <summary>Folds a batch's moments in (<c>update_from_moments</c>).</summary>
        /// <param name="batchMean">The batch's mean; left to the caller.</param>
        /// <param name="batchVar">The batch's variance; left to the caller.</param>
        /// <param name="batchCount">The batch's length.</param>
        public void UpdateFromMoments(NDArray batchMean, NDArray batchVar, long batchCount) {
            (NDArray mean, NDArray var, double count) = WrapperFunctions.UpdateMeanVarCountFromMoments(Mean, Var, Count, batchMean, batchVar, batchCount);
            // The new moments are kept until the next update, past any NumSharp scope the caller opened around this step (DR-96).
            ScopeSafety.Keep(mean);
            ScopeSafety.Keep(var);
            Mean.Dispose();
            Var.Dispose();
            Mean = mean;
            Var = var;
            Count = count;
        }

        /// <summary>Releases the statistics' arrays.</summary>
        public void Dispose() {
            Mean.Dispose();
            Var.Dispose();
        }
    }

    /// <summary>Gymnasium's <c>gymnasium.wrappers.utils</c> functions that this phase's wrappers use.</summary>
    public static partial class WrapperFunctions {
        /// <summary>
        ///     Gymnasium's <c>update_mean_var_count_from_moments</c>: merges running moments with a batch's, element by element,
        ///     in the dtype NumPy 2 computes them in (the statistics' dtype promoted with the batch's; the Python counts cast
        ///     to it as weak scalars), with the source's order of operations.
        /// </summary>
        /// <param name="mean">The running mean.</param>
        /// <param name="var">The running variance.</param>
        /// <param name="count">The running count.</param>
        /// <param name="batchMean">The batch's mean, of the running mean's shape.</param>
        /// <param name="batchVar">The batch's variance, of the same shape.</param>
        /// <param name="batchCount">The batch's length.</param>
        /// <returns>New mean and variance arrays the caller owns, and the new count.</returns>
        /// <exception cref="ArgumentNullException">Thrown when an array is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">Thrown when the shapes differ.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/utils.py#L57">gymnasium/wrappers/utils.py</see>.</remarks>
        public static (NDArray Mean, NDArray Var, double Count) UpdateMeanVarCountFromMoments(NDArray mean, NDArray var, double count, NDArray batchMean, NDArray batchVar, long batchCount) {
            if (mean is null)
                throw new ArgumentNullException(nameof(mean));
            if (var is null)
                throw new ArgumentNullException(nameof(var));
            if (batchMean is null)
                throw new ArgumentNullException(nameof(batchMean));
            if (batchVar is null)
                throw new ArgumentNullException(nameof(batchVar));
            if (!mean.shape.SequenceEqual(batchMean.shape) || !var.shape.SequenceEqual(batchVar.shape) || !mean.shape.SequenceEqual(var.shape))
                throw new ValueError("operands could not be broadcast together with shapes " + ShapeText(mean) + " " + ShapeText(batchMean));

            // delta = batch_mean - mean: two arrays, so their dtypes promote.
            DType deltaType = np.promote_types(batchMean.dtype, mean.dtype);
            // new_mean = mean + delta * batch_count / tot_count keeps delta's dtype (the counts are weak Python numbers).
            DType meanType = np.promote_types(mean.dtype, deltaType);
            // M2 = var * count + batch_var * batch_count + square(delta) * ...: var, batch_var and delta promote together.
            DType varType = np.promote_types(np.promote_types(var.dtype, batchVar.dtype), deltaType);
            double totCount = count + batchCount;

            Shape shape = mean.Shape;
            double[] m = Values(mean), v = Values(var), bm = Values(batchMean), bv = Values(batchVar);
            var newMean = new double[m.Length];
            var newVar = new double[m.Length];
            bool deltaSingle = IsSingle(deltaType), meanSingle = IsSingle(meanType), varSingle = IsSingle(varType);
            bool varTermSingle = IsSingle(var.dtype), batchVarTermSingle = IsSingle(batchVar.dtype);
            bool sumSingle = IsSingle(np.promote_types(var.dtype, batchVar.dtype));
            for (int i = 0; i < m.Length; i++) {
                // Every operation is rounded to its NumPy dtype before the next: a float32 operation computed in double and
                // rounded once is the correctly rounded float32 result (53 >= 2 * 24 + 2 bits), and a weak Python count
                // meeting a float32 operand is cast to float32 first, as NEP 50 does.
                double delta = Round(bm[i] - m[i], deltaSingle);
                double scaled = Round(Round(delta * Weak(batchCount, deltaSingle), deltaSingle) / Weak(totCount, deltaSingle), deltaSingle);
                newMean[i] = Round(m[i] + scaled, meanSingle);

                double mA = Round(v[i] * Weak(count, varTermSingle), varTermSingle);
                double mB = Round(bv[i] * Weak(batchCount, batchVarTermSingle), batchVarTermSingle);
                // np.square(delta) * count * batch_count / tot_count, left to right in delta's dtype.
                double square = Round(delta * delta, deltaSingle);
                square = Round(square * Weak(count, deltaSingle), deltaSingle);
                square = Round(square * Weak(batchCount, deltaSingle), deltaSingle);
                square = Round(square / Weak(totCount, deltaSingle), deltaSingle);
                double m2 = Round(Round(mA + mB, sumSingle) + square, varSingle);
                newVar[i] = Round(m2 / Weak(totCount, varSingle), varSingle);
            }

            return (Array(newMean, shape, meanType), Array(newVar, shape, varType), totCount);
        }

        /// <summary>An array's values widened to double, in C order (exact for every float and integer dtype up to 2^53).</summary>
        /// <param name="array">The array.</param>
        /// <returns>The values.</returns>
        private static double[] Values(NDArray array) {
            using NDArray widened = array.astype(np.float64);
            return widened.ToArray<double>();
        }

        /// <summary>A double rounded to float32 when the operation's dtype is float32, else as it is.</summary>
        /// <param name="value">The value.</param>
        /// <param name="single">Whether the dtype is float32.</param>
        /// <returns>The value in that precision.</returns>
        private static double Round(double value, bool single) => single ? (float)value : value;

        /// <summary>A weak Python number as an operation of that precision sees it: cast to float32 for a float32 operation (NEP 50).</summary>
        /// <param name="value">The Python number.</param>
        /// <param name="single">Whether the operation is float32.</param>
        /// <returns>The operand.</returns>
        private static double Weak(double value, bool single) => single ? (float)value : value;

        /// <summary>Whether a dtype is float32 (any other dtype here is float64, as the statistics' promotions only reach those).</summary>
        /// <param name="dtype">The dtype.</param>
        /// <returns><see langword="true"/> for float32.</returns>
        private static bool IsSingle(DType dtype) => dtype.typecode == NPTypeCode.Single;

        /// <summary>A new array of the values, reshaped and cast to the dtype (exact: each value was computed in it).</summary>
        /// <param name="values">The values.</param>
        /// <param name="shape">The shape.</param>
        /// <param name="dtype">The dtype.</param>
        /// <returns>The array; the caller owns it.</returns>
        private static NDArray Array(double[] values, Shape shape, DType dtype) {
            using NDArray flat = np.array(values);
            using NDArray shaped = flat.reshape(shape);
            return shaped.astype(dtype);
        }

        /// <summary>An array's shape as NumPy prints it in a broadcast error: <c>(3,)</c>.</summary>
        /// <param name="array">The array.</param>
        /// <returns>The text.</returns>
        private static string ShapeText(NDArray array) => array.ndim == 1 ? $"({array.shape[0]},)" : "(" + string.Join(",", array.shape) + ")";
    }
}
