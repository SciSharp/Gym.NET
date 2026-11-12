using System;

namespace Gym.MuJoCo {
    /// <summary>
    ///     The vector, matrix and factorization kernels the HalfCheetah core needs, each written to round exactly as MuJoCo
    ///     4.0.0's win-x64 build does: AVX lanes without FMA, so a dot product sums four lanes and reduces them as
    ///     <c>(l0 + l2) + (l1 + l3)</c> before a grouped tail, and every element-wise kernel is plain IEEE arithmetic.
    /// </summary>
    /// <remarks>
    ///     The operation order is the whole contract: reordering a sum, fusing a multiply-add or skipping a zero test that
    ///     MuJoCo performs changes the last bit and, a few hundred steps later, the trajectory. The zero tests matter for the
    ///     sign of zero (MuJoCo skips rows whose scale is zero, so their entries stay +0). Spans are indexed without bounds
    ///     beyond what the caller passes; a too-short span throws <see cref="IndexOutOfRangeException"/>.
    ///     Reference: MuJoCo 4.0.0's <see href="https://github.com/google-deepmind/mujoco/blob/4.0.0/src/engine/engine_util_blas.c">engine_util_blas.c</see>,
    ///     <see href="https://github.com/google-deepmind/mujoco/blob/4.0.0/src/engine/engine_util_blas_avx.h">engine_util_blas_avx.h</see>,
    ///     <see href="https://github.com/google-deepmind/mujoco/blob/4.0.0/src/engine/engine_util_sparse_avx.h">engine_util_sparse_avx.h</see>,
    ///     <see href="https://github.com/google-deepmind/mujoco/blob/4.0.0/src/engine/engine_util_solve.c">engine_util_solve.c</see> and the
    ///     factorization in <see href="https://github.com/google-deepmind/mujoco/blob/4.0.0/src/engine/engine_core_smooth.c">engine_core_smooth.c</see>.
    /// </remarks>
    internal static class Kernels {
        /// <summary>MuJoCo's smallest denominator, <c>mjMINVAL</c>: the clamp of every division and pivot.</summary>
        public const double MinVal = 1e-15;

        /// <summary>
        ///     The dot product of the first <paramref name="n"/> entries, summed in AVX order: four lane accumulators seeded
        ///     with the first four products, reduced as <c>(l0 + l2) + (l1 + l3)</c>, then a tail of up to three products added
        ///     as one grouped sum. Below four entries the lanes are skipped and the tail is added to +0.
        /// </summary>
        /// <param name="a">The first vector.</param>
        /// <param name="b">The second vector.</param>
        /// <param name="n">How many entries to use.</param>
        /// <returns>The dot product, bit-identical to <c>mju_dot</c>.</returns>
        public static double Dot(ReadOnlySpan<double> a, ReadOnlySpan<double> b, int n) {
            double res = 0;
            int i = 0;
            if (n - 4 >= 0) {
                // The AVX kernel seeds the lanes with the first products (not 0 + product), which keeps a -0 product's sign.
                double l0 = a[0] * b[0], l1 = a[1] * b[1], l2 = a[2] * b[2], l3 = a[3] * b[3];
                i = 4;
                while (i <= n - 4) {
                    l0 += a[i] * b[i];
                    l1 += a[i + 1] * b[i + 1];
                    l2 += a[i + 2] * b[i + 2];
                    l3 += a[i + 3] * b[i + 3];
                    i += 4;
                }
                res = (l0 + l2) + (l1 + l3);
            }
            // The tail is one grouped expression in MuJoCo: res + ((p0 + p1) + p2), not three separate additions.
            int tail = n - i;
            if (tail == 3)
                res += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2];
            else if (tail == 2)
                res += a[i] * b[i] + a[i + 1] * b[i + 1];
            else if (tail == 1)
                res += a[i] * b[i];
            return res;
        }

        /// <summary>
        ///     The dot product of a sparse row with a dense vector, in MuJoCo's AVX order: lanes over the first multiple of four
        ///     entries, then the remaining products added one at a time (unlike <see cref="Dot"/>'s grouped tail).
        /// </summary>
        /// <param name="row">The row's nonzero values.</param>
        /// <param name="x">The dense vector.</param>
        /// <param name="nnz">The number of nonzeros.</param>
        /// <param name="ind">The column of each nonzero.</param>
        /// <returns>The dot product, bit-identical to <c>mju_dotSparse</c>.</returns>
        public static double DotSparse(ReadOnlySpan<double> row, ReadOnlySpan<double> x, int nnz, ReadOnlySpan<int> ind) {
            int i = 0;
            double res = 0;
            if (nnz - 4 >= 0) {
                double l0 = row[0] * x[ind[0]], l1 = row[1] * x[ind[1]], l2 = row[2] * x[ind[2]], l3 = row[3] * x[ind[3]];
                i = 4;
                while (i <= nnz - 4) {
                    l0 += row[i] * x[ind[i]];
                    l1 += row[i + 1] * x[ind[i + 1]];
                    l2 += row[i + 2] * x[ind[i + 2]];
                    l3 += row[i + 3] * x[ind[i + 3]];
                    i += 4;
                }
                res = (l0 + l2) + (l1 + l3);
            }
            for (; i < nnz; i++)
                res += row[i] * x[ind[i]];
            return res;
        }

        /// <summary>The Euclidean norm, <c>sqrt(Dot(v, v))</c>, as <c>mju_norm</c> computes it.</summary>
        /// <param name="v">The vector.</param>
        /// <param name="n">How many entries to use.</param>
        /// <returns>The norm.</returns>
        public static double Norm(ReadOnlySpan<double> v, int n) => Math.Sqrt(Dot(v, v, n));

        /// <summary><c>res[i] += vec[i]*scl</c>: the scaled accumulation every matrix product here is built from.</summary>
        /// <param name="res">The accumulator, updated in place.</param>
        /// <param name="vec">The vector to add.</param>
        /// <param name="scl">The scale.</param>
        /// <param name="n">How many entries.</param>
        public static void AddToScl(Span<double> res, ReadOnlySpan<double> vec, double scl, int n) {
            for (int i = 0; i < n; i++)
                res[i] += vec[i] * scl;
        }

        /// <summary><c>res[i] = a[i] + b[i]*scl</c>.</summary>
        /// <param name="res">The output (may not alias <paramref name="b"/> at a different offset).</param>
        /// <param name="a">The base vector.</param>
        /// <param name="b">The scaled vector.</param>
        /// <param name="scl">The scale.</param>
        /// <param name="n">How many entries.</param>
        public static void AddScl(Span<double> res, ReadOnlySpan<double> a, ReadOnlySpan<double> b, double scl, int n) {
            for (int i = 0; i < n; i++)
                res[i] = a[i] + b[i] * scl;
        }

        /// <summary><c>res[i] = vec[i]*scl</c>; in place when the spans are the same.</summary>
        /// <param name="res">The output.</param>
        /// <param name="vec">The input.</param>
        /// <param name="scl">The scale.</param>
        /// <param name="n">How many entries.</param>
        public static void Scl(Span<double> res, ReadOnlySpan<double> vec, double scl, int n) {
            for (int i = 0; i < n; i++)
                res[i] = vec[i] * scl;
        }

        /// <summary><c>res[i] = a[i] + b[i]</c>.</summary>
        /// <param name="res">The output.</param>
        /// <param name="a">The first addend.</param>
        /// <param name="b">The second addend.</param>
        /// <param name="n">How many entries.</param>
        public static void Add(Span<double> res, ReadOnlySpan<double> a, ReadOnlySpan<double> b, int n) {
            for (int i = 0; i < n; i++)
                res[i] = a[i] + b[i];
        }

        /// <summary><c>res[i] = a[i] - b[i]</c>.</summary>
        /// <param name="res">The output.</param>
        /// <param name="a">The minuend.</param>
        /// <param name="b">The subtrahend.</param>
        /// <param name="n">How many entries.</param>
        public static void Sub(Span<double> res, ReadOnlySpan<double> a, ReadOnlySpan<double> b, int n) {
            for (int i = 0; i < n; i++)
                res[i] = a[i] - b[i];
        }

        /// <summary><c>res[i] += vec[i]</c>.</summary>
        /// <param name="res">The accumulator.</param>
        /// <param name="vec">The addend.</param>
        /// <param name="n">How many entries.</param>
        public static void AddTo(Span<double> res, ReadOnlySpan<double> vec, int n) {
            for (int i = 0; i < n; i++)
                res[i] += vec[i];
        }

        /// <summary><c>res[i] -= vec[i]</c>.</summary>
        /// <param name="res">The accumulator.</param>
        /// <param name="vec">The subtrahend.</param>
        /// <param name="n">How many entries.</param>
        public static void SubFrom(Span<double> res, ReadOnlySpan<double> vec, int n) {
            for (int i = 0; i < n; i++)
                res[i] -= vec[i];
        }

        /// <summary><c>res = mat*vec</c> for a dense row-major <paramref name="nr"/>-by-<paramref name="nc"/> matrix, one
        ///     <see cref="Dot"/> per row.</summary>
        /// <param name="res">The output, <paramref name="nr"/> entries.</param>
        /// <param name="mat">The matrix.</param>
        /// <param name="vec">The vector, <paramref name="nc"/> entries.</param>
        /// <param name="nr">Rows.</param>
        /// <param name="nc">Columns.</param>
        public static void MulMatVec(Span<double> res, ReadOnlySpan<double> mat, ReadOnlySpan<double> vec, int nr, int nc) {
            for (int r = 0; r < nr; r++)
                res[r] = Dot(mat.Slice(r * nc), vec, nc);
        }

        /// <summary>
        ///     <c>res = mat'*vec</c>: zero <paramref name="res"/>, then add each row scaled by its entry of
        ///     <paramref name="vec"/>, skipping rows whose entry is zero (so an all-satisfied constraint set leaves +0).
        /// </summary>
        /// <param name="res">The output, <paramref name="nc"/> entries.</param>
        /// <param name="mat">The matrix.</param>
        /// <param name="vec">The vector, <paramref name="nr"/> entries.</param>
        /// <param name="nr">Rows.</param>
        /// <param name="nc">Columns.</param>
        public static void MulMatTVec(Span<double> res, ReadOnlySpan<double> mat, ReadOnlySpan<double> vec, int nr, int nc) {
            res.Slice(0, nc).Clear();
            for (int r = 0; r < nr; r++) {
                double tmp = vec[r];
                if (tmp != 0)
                    AddToScl(res, mat.Slice(r * nc), tmp, nc);
            }
        }

        /// <summary>
        ///     <c>res = mat1*mat2</c>, exploiting zeros of <paramref name="mat1"/>: each nonzero <c>mat1[i,k]</c> adds row
        ///     <c>k</c> of <paramref name="mat2"/> into row <c>i</c>, in increasing <c>k</c>.
        /// </summary>
        /// <param name="res">The output, <paramref name="r1"/>-by-<paramref name="c2"/>.</param>
        /// <param name="mat1">The left matrix, <paramref name="r1"/>-by-<paramref name="c1"/>.</param>
        /// <param name="mat2">The right matrix, <paramref name="c1"/>-by-<paramref name="c2"/>.</param>
        /// <param name="r1">Rows of the left matrix.</param>
        /// <param name="c1">Columns of the left matrix.</param>
        /// <param name="c2">Columns of the right matrix.</param>
        public static void MulMatMat(Span<double> res, ReadOnlySpan<double> mat1, ReadOnlySpan<double> mat2, int r1, int c1, int c2) {
            res.Slice(0, r1 * c2).Clear();
            for (int i = 0; i < r1; i++)
                for (int k = 0; k < c1; k++) {
                    double tmp = mat1[i * c1 + k];
                    if (tmp != 0)
                        AddToScl(res.Slice(i * c2), mat2.Slice(k * c2), tmp, c2);
                }
        }

        /// <summary>
        ///     The lower triangle of <c>mat'*diag(d)*mat</c> (<c>mju_sqrMatTD_impl</c> without the upper copy): rows with a zero
        ///     weight are skipped, and each nonzero <c>mat[j,i]</c> adds <c>mat[j,0..i]*(mat[j,i]*d[j])</c> into row <c>i</c>.
        /// </summary>
        /// <param name="res">The <paramref name="nc"/>-by-<paramref name="nc"/> output; only its lower triangle is meaningful.</param>
        /// <param name="mat">The <paramref name="nr"/>-by-<paramref name="nc"/> matrix.</param>
        /// <param name="diag">The row weights.</param>
        /// <param name="nr">Rows.</param>
        /// <param name="nc">Columns.</param>
        public static void SqrMatTDLower(Span<double> res, ReadOnlySpan<double> mat, ReadOnlySpan<double> diag, int nr, int nc) {
            res.Slice(0, nc * nc).Clear();
            for (int j = 0; j < nr; j++) {
                if (diag[j] == 0)
                    continue;
                for (int i = 0; i < nc; i++) {
                    double tmp = mat[j * nc + i];
                    if (tmp != 0)
                        AddToScl(res.Slice(i * nc), mat.Slice(j * nc), tmp * diag[j], i + 1);
                }
            }
        }

        /// <summary>Adds the stored lower triangle of a sparse symmetric matrix to a dense one (lower part only).</summary>
        /// <param name="res">The dense <paramref name="n"/>-by-<paramref name="n"/> accumulator.</param>
        /// <param name="mat">The sparse values.</param>
        /// <param name="n">The dimension.</param>
        /// <param name="rownnz">Nonzeros per row.</param>
        /// <param name="rowadr">Row start addresses.</param>
        /// <param name="colind">Column indices.</param>
        public static void AddToSymSparseLower(Span<double> res, ReadOnlySpan<double> mat, int n, int[] rownnz, int[] rowadr, int[] colind) {
            for (int i = 0; i < n; i++) {
                int end = rowadr[i] + rownnz[i];
                for (int adr = rowadr[i]; adr < end; adr++)
                    res[i * n + colind[adr]] += mat[adr];
            }
        }

        /// <summary>
        ///     <c>res = M*vec</c> for MuJoCo's sparse lower-triangular storage of a symmetric matrix: the diagonal product
        ///     first, then the off-diagonal entries from the last to the first, each feeding its row and its column.
        /// </summary>
        /// <param name="res">The output, <paramref name="n"/> entries.</param>
        /// <param name="mat">The sparse values (diagonal last in each row).</param>
        /// <param name="vec">The vector.</param>
        /// <param name="n">The dimension.</param>
        /// <param name="rownnz">Nonzeros per row.</param>
        /// <param name="rowadr">Row start addresses.</param>
        /// <param name="colind">Column indices.</param>
        public static void MulSymVecSparse(Span<double> res, ReadOnlySpan<double> mat, ReadOnlySpan<double> vec, int n,
                                           int[] rownnz, int[] rowadr, int[] colind) {
            res.Slice(0, n).Clear();
            for (int i = 0; i < n; i++) {
                int adr = rowadr[i];
                int diag = rownnz[i] - 1;
                res[i] = mat[adr + diag] * vec[i];
                for (int k = diag - 1; k >= 0; k--) {
                    int j = colind[adr + k];
                    double val = mat[adr + k];
                    res[i] += val * vec[j];
                    res[j] += val * vec[i];
                }
            }
        }

        /// <summary>
        ///     In-place dense Cholesky <c>mat = L*L'</c> (lower triangle), clamping pivots below <paramref name="mindiag"/> and
        ///     clearing their column, as <c>mju_cholFactor</c> does.
        /// </summary>
        /// <param name="mat">The <paramref name="n"/>-by-<paramref name="n"/> matrix; its lower triangle becomes L.</param>
        /// <param name="n">The dimension.</param>
        /// <param name="mindiag">The pivot floor.</param>
        /// <returns>The rank: <paramref name="n"/> minus the number of clamped pivots.</returns>
        public static int CholFactor(Span<double> mat, int n, double mindiag) {
            int rank = n;
            for (int j = 0; j < n; j++) {
                double tmp = mat[j * (n + 1)];
                if (j != 0)
                    tmp -= Dot(mat.Slice(j * n), mat.Slice(j * n), j);
                bool deficient = tmp < mindiag;
                if (deficient) {
                    tmp = mindiag;
                    rank--;
                }
                mat[j * (n + 1)] = Math.Sqrt(tmp);
                if (deficient) {
                    for (int i = j + 1; i < n; i++)
                        mat[i * n + j] = 0;
                } else {
                    tmp = 1 / mat[j * (n + 1)];
                    for (int i = j + 1; i < n; i++)
                        mat[i * n + j] = (mat[i * n + j] - Dot(mat.Slice(i * n), mat.Slice(j * n), j)) * tmp;
                }
            }
            return rank;
        }

        /// <summary>Solves <c>L*L'*res = vec</c> with the factor of <see cref="CholFactor"/>.</summary>
        /// <param name="res">The solution (may be the same storage as <paramref name="vec"/>).</param>
        /// <param name="mat">The factor.</param>
        /// <param name="vec">The right-hand side.</param>
        /// <param name="n">The dimension.</param>
        public static void CholSolve(Span<double> res, ReadOnlySpan<double> mat, ReadOnlySpan<double> vec, int n) {
            vec.Slice(0, n).CopyTo(res);
            for (int i = 0; i < n; i++) {
                if (i != 0)
                    res[i] -= Dot(mat.Slice(i * n), res, i);
                res[i] /= mat[i * (n + 1)];
            }
            // MuJoCo 4 substitutes backward by columns: divide, then push row i of L into the entries before it
            // (mju_addToScl with -res[i]), so each entry takes its terms in descending order; a zero is skipped
            for (int i = n - 1; i >= 0; i--) {
                res[i] /= mat[i * (n + 1)];
                if (i != 0 && res[i] != 0) {
                    double scl = -res[i];
                    for (int k = 0; k < i; k++)
                        res[k] += mat[i * n + k] * scl;
                }
            }
        }

        /// <summary>
        ///     The rank-one update <c>L*L' ± x*x'</c> of a dense Cholesky factor, consuming <paramref name="x"/>; a pivot that
        ///     falls below <see cref="MinVal"/> is clamped and lowers the returned rank (the caller then refactorizes).
        /// </summary>
        /// <param name="mat">The factor, updated in place.</param>
        /// <param name="x">The update vector, overwritten.</param>
        /// <param name="n">The dimension.</param>
        /// <param name="plus"><see langword="true"/> to add <c>x*x'</c>, <see langword="false"/> to subtract it.</param>
        /// <returns>The rank after the update.</returns>
        public static int CholUpdate(Span<double> mat, Span<double> x, int n, bool plus) {
            int rank = n;
            for (int k = 0; k < n; k++) {
                if (x[k] == 0)
                    continue;
                double lkk = mat[k * (n + 1)];
                double tmp = lkk * lkk + (plus ? x[k] * x[k] : -x[k] * x[k]);
                if (tmp < MinVal) {
                    tmp = MinVal;
                    rank--;
                }
                double r = Math.Sqrt(tmp);
                double c = r / lkk;
                double cinv = 1 / c;
                double s = x[k] / lkk;
                mat[k * (n + 1)] = r;
                if (plus) {
                    for (int i = k + 1; i < n; i++)
                        mat[i * n + k] = (mat[i * n + k] + s * x[i]) * cinv;
                } else {
                    for (int i = k + 1; i < n; i++)
                        mat[i * n + k] = (mat[i * n + k] - s * x[i]) * cinv;
                }
                for (int i = k + 1; i < n; i++)
                    x[i] = c * x[i] - s * mat[i * n + k];
            }
            return rank;
        }

        /// <summary>
        ///     The sparse <c>L'*D*L</c> factorization of an inertia-like matrix in MuJoCo's storage (<c>mj_factorI</c>): rows
        ///     from the last to the first, pivots clamped to <see cref="MinVal"/>, each row's ancestors updated before the row is
        ///     scaled. Relies on every row's sparsity being its dof ancestor chain, which a kinematic tree guarantees.
        /// </summary>
        /// <param name="mat">The matrix, overwritten by L (unit diagonal implied) and D on the diagonal.</param>
        /// <param name="diaginv">Receives <c>1/D</c> per row.</param>
        /// <param name="nv">The dimension.</param>
        /// <param name="rownnz">Nonzeros per row.</param>
        /// <param name="rowadr">Row start addresses.</param>
        /// <param name="colind">Column indices.</param>
        /// <returns>The first row whose pivot was clamped, or -1.</returns>
        public static int FactorI(Span<double> mat, Span<double> diaginv, int nv, int[] rownnz, int[] rowadr, int[] colind) {
            int clamped = -1;
            for (int k = nv - 1; k >= 0; k--) {
                int start = rowadr[k];
                int diag = rownnz[k] - 1;
                int end = start + diag;
                if (mat[end] < MinVal) {
                    mat[end] = MinVal;
                    if (clamped < 0)
                        clamped = k;
                }
                double invD = 1 / mat[end];
                diaginv[k] = invD;
                for (int adr = end - 1; adr >= start; adr--) {
                    int i = colind[adr];
                    AddToScl(mat.Slice(rowadr[i]), mat.Slice(start), -mat[adr] * invD, rownnz[i]);
                }
                Scl(mat.Slice(start), mat.Slice(start), invD, diag);
            }
            return clamped;
        }

        /// <summary>
        ///     Solves <c>L'*D*L*x = x</c> in place with <see cref="FactorI"/>'s factor (<c>mj_solveLD</c> for one vector):
        ///     <c>L^-T</c> from the last row, then <c>D^-1</c>, then <c>L^-1</c> with sparse dot products.
        /// </summary>
        /// <param name="x">The right-hand side, overwritten by the solution.</param>
        /// <param name="qLD">The factor.</param>
        /// <param name="qLDiagInv">The inverse diagonal.</param>
        /// <param name="nv">The dimension.</param>
        /// <param name="rownnz">Nonzeros per row.</param>
        /// <param name="rowadr">Row start addresses.</param>
        /// <param name="colind">Column indices.</param>
        public static void SolveLD(Span<double> x, ReadOnlySpan<double> qLD, ReadOnlySpan<double> qLDiagInv, int nv,
                                   int[] rownnz, int[] rowadr, int[] colind) {
            for (int i = nv - 1; i >= 0; i--) {
                if (rownnz[i] == 1)
                    continue;
                double xi = x[i];
                if (xi != 0) {
                    int start = rowadr[i];
                    int end = start + rownnz[i] - 1;
                    for (int adr = start; adr < end; adr++)
                        x[colind[adr]] -= qLD[adr] * xi;
                }
            }
            for (int i = 0; i < nv; i++)
                x[i] *= qLDiagInv[i];
            for (int i = 0; i < nv; i++) {
                if (rownnz[i] == 1)
                    continue;
                int d = rownnz[i] - 1;
                if (d > 0) {
                    int adr = rowadr[i];
                    x[i] -= DotSparse(qLD.Slice(adr), x, d, colind.AsSpan(adr));
                }
            }
        }

        /// <summary><c>mju_max</c>: <paramref name="a"/> when <c>a &gt;= b</c>, else <paramref name="b"/> (NaN in
        ///     <paramref name="a"/> or <paramref name="b"/> picks <paramref name="b"/>).</summary>
        /// <param name="a">The first value.</param>
        /// <param name="b">The second value.</param>
        /// <returns>The larger value, MuJoCo's way.</returns>
        public static double Max(double a, double b) => a >= b ? a : b;

        /// <summary><c>mju_min</c>: <paramref name="a"/> when <c>a &lt;= b</c>, else <paramref name="b"/>.</summary>
        /// <param name="a">The first value.</param>
        /// <param name="b">The second value.</param>
        /// <returns>The smaller value, MuJoCo's way.</returns>
        public static double Min(double a, double b) => a <= b ? a : b;

        /// <summary><c>res = mat'</c> for a dense row-major <paramref name="nr"/>-by-<paramref name="nc"/> matrix.</summary>
        /// <param name="res">The <paramref name="nc"/>-by-<paramref name="nr"/> output (must not alias <paramref name="mat"/>).</param>
        /// <param name="mat">The matrix.</param>
        /// <param name="nr">Rows.</param>
        /// <param name="nc">Columns.</param>
        public static void Transpose(Span<double> res, ReadOnlySpan<double> mat, int nr, int nc) {
            for (int i = 0; i < nr; i++)
                for (int j = 0; j < nc; j++)
                    res[j * nr + i] = mat[i * nc + j];
        }

        /// <summary>
        ///     <c>mat'*mat</c> without weights, symmetric (<c>mju_sqrMatTD</c> with <c>diag = NULL</c>): MuJoCo's unweighted
        ///     branch loops columns outside and rows inside, the reverse of the weighted branch of <see cref="SqrMatTDLower"/>,
        ///     and each nonzero <c>mat[j,i]</c> adds <c>mat[j,0..i]*mat[j,i]</c> into row <c>i</c>; the upper triangle is
        ///     then copied from the lower.
        /// </summary>
        /// <param name="res">The <paramref name="nc"/>-by-<paramref name="nc"/> output.</param>
        /// <param name="mat">The <paramref name="nr"/>-by-<paramref name="nc"/> matrix.</param>
        /// <param name="nr">Rows.</param>
        /// <param name="nc">Columns.</param>
        public static void SqrMatT(Span<double> res, ReadOnlySpan<double> mat, int nr, int nc) {
            res.Slice(0, nc * nc).Clear();
            for (int i = 0; i < nc; i++)
                for (int j = 0; j < nr; j++) {
                    double tmp = mat[j * nc + i];
                    if (tmp != 0)
                        AddToScl(res.Slice(i * nc), mat.Slice(j * nc), tmp, i + 1);
                }
            for (int i = 0; i < nc; i++)
                for (int j = i + 1; j < nc; j++)
                    res[i * nc + j] = res[j * nc + i];
        }
    }

    /// <summary>
    ///     The PCG32 generator MuJoCo's PGS solver shuffles its sweep order with (<c>pcg32_next</c> in
    ///     <c>engine_solver.c</c>): a 64-bit LCG state, output by an xorshift and a data-dependent rotation.
    /// </summary>
    /// <remarks>
    ///     MuJoCo restarts it at state 0, increment 1, and draws one value before every solve, so the shuffles repeat from
    ///     solve to solve; <see cref="Restart"/> does the same. A mutable struct: pass it by reference.
    /// </remarks>
    internal struct Pcg32 {
        private ulong state;
        private ulong inc;

        /// <summary>Restarts the sequence as <c>solPGS</c> does: state 0, increment 1, one value drawn and discarded.</summary>
        public void Restart() {
            state = 0;
            inc = 1;
            Next();
        }

        /// <summary>The next 32-bit value.</summary>
        /// <returns>The value.</returns>
        public uint Next() {
            ulong old = state;
            state = unchecked(old * 6364136223846793005UL + (inc | 1));
            uint xorshifted = (uint)(((old >> 18) ^ old) >> 27);
            int rot = (int)(old >> 59);
            // C's (-rot) & 31 on an unsigned rotation count is (32 - rot) & 31
            return (xorshifted >> rot) | (xorshifted << ((32 - rot) & 31));
        }

        /// <summary>Fisher-Yates in place, from the last element down (<c>shuffle_int</c>).</summary>
        /// <param name="array">The values to shuffle.</param>
        /// <param name="n">How many.</param>
        public void Shuffle(Span<int> array, int n) {
            for (int i = n - 1; i > 0; i--) {
                int j = (int)(Next() % (uint)(i + 1));
                (array[i], array[j]) = (array[j], array[i]);
            }
        }
    }

    /// <summary>
    ///     The 3-vector, quaternion and spatial-algebra helpers of the kinematics, each with MuJoCo's term order
    ///     (<c>engine_inline.h</c> and <c>engine_util_spatial.c</c>).
    /// </summary>
    /// <remarks>
    ///     Outputs must not alias inputs unless a member says it may, as in MuJoCo's <c>restrict</c> helpers.
    ///     Reference: <see href="https://github.com/google-deepmind/mujoco/blob/4.0.0/src/engine/engine_inline.h">engine_inline.h</see>
    ///     and <see href="https://github.com/google-deepmind/mujoco/blob/4.0.0/src/engine/engine_util_spatial.c">engine_util_spatial.c</see> at MuJoCo 4.0.0.
    /// </remarks>
    internal static class Spatial {
        /// <summary><c>res = mat*vec</c> for a 3x3 row-major matrix.</summary>
        /// <param name="res">The output.</param>
        /// <param name="mat">The matrix.</param>
        /// <param name="vec">The vector.</param>
        public static void MulMatVec3(Span<double> res, ReadOnlySpan<double> mat, ReadOnlySpan<double> vec) {
            res[0] = mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2];
            res[1] = mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2];
            res[2] = mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2];
        }

        /// <summary>
        ///     The product of two row-major 3x3 matrices (<c>mju_mulMatMat3</c>), each entry summed left to right.
        /// </summary>
        /// <param name="res">The output; must not alias either input.</param>
        /// <param name="mat1">The left matrix.</param>
        /// <param name="mat2">The right matrix.</param>
        public static void MulMatMat3(Span<double> res, ReadOnlySpan<double> mat1, ReadOnlySpan<double> mat2) {
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    res[3 * i + j] = mat1[3 * i] * mat2[j] + mat1[3 * i + 1] * mat2[3 + j] + mat1[3 * i + 2] * mat2[6 + j];
        }

        /// <summary>
        ///     The transposed product <c>mat'*vec</c> of a row-major 3x3 matrix, each component summed left to right
        ///     (<c>mji_mulMatTVec3</c>): a world vector expressed in the matrix's frame.
        /// </summary>
        /// <param name="res">The output; must not alias <paramref name="vec"/>.</param>
        /// <param name="mat">The matrix.</param>
        /// <param name="vec">The vector.</param>
        public static void MulMatTVec3(Span<double> res, ReadOnlySpan<double> mat, ReadOnlySpan<double> vec) {
            res[0] = mat[0] * vec[0] + mat[3] * vec[1] + mat[6] * vec[2];
            res[1] = mat[1] * vec[0] + mat[4] * vec[1] + mat[7] * vec[2];
            res[2] = mat[2] * vec[0] + mat[5] * vec[1] + mat[8] * vec[2];
        }

        /// <summary>The cross product <c>a x b</c>.</summary>
        /// <param name="res">The output.</param>
        /// <param name="a">The left vector.</param>
        /// <param name="b">The right vector.</param>
        public static void Cross(Span<double> res, ReadOnlySpan<double> a, ReadOnlySpan<double> b) {
            res[0] = a[1] * b[2] - a[2] * b[1];
            res[1] = a[2] * b[0] - a[0] * b[2];
            res[2] = a[0] * b[1] - a[1] * b[0];
        }

        /// <summary>
        ///     The time derivative of a rotation vector (<c>mju_derivExpmap</c>, MuJoCo 4): <c>vel + 1/2 r x vel + c r x (r x
        ///     vel)</c> with Bortz's coefficient <c>c</c> of <c>|r|²</c>, after reducing <c>r</c>'s angle to (-π, π].
        /// </summary>
        /// <param name="res">Receives the rate; may alias <paramref name="vel"/> (the inputs are read first).</param>
        /// <param name="expmap">The rotation vector.</param>
        /// <param name="vel">The angular velocity in the rotated frame.</param>
        /// <remarks>
        ///     Reference: MuJoCo 4.0.0's
        ///     <see href="https://github.com/google-deepmind/mujoco/blob/4.0.0/src/engine/engine_util_spatial.c">engine_util_spatial.c</see>.
        ///     <c>mju_floor</c> and <c>mju_tan</c> are C's <c>floor</c> and <c>tan</c>, which .NET's <c>Math</c> calls on win-x64.
        /// </remarks>
        public static void DerivExpmap(Span<double> res, ReadOnlySpan<double> expmap, ReadOnlySpan<double> vel) {
            Span<double> r = stackalloc double[3] { expmap[0], expmap[1], expmap[2] };
            double xx = Dot3(r, r);
            // the coefficient is singular at multiples of 2π: take the equivalent angle in (-π, π]
            if (xx > Math.PI * Math.PI) {
                double x = Math.Sqrt(xx);
                double scale = (x - 2 * Math.PI * Math.Floor(x / (2 * Math.PI) + 0.5)) / x;
                for (int k = 0; k < 3; k++)
                    r[k] *= scale;
                xx = Dot3(r, r);
            }
            double c = BortzCoef(xx);
            // vel is read whole before res is written: res may alias it
            Span<double> v = stackalloc double[3] { vel[0], vel[1], vel[2] };
            Span<double> rxv = stackalloc double[3], rxrxv = stackalloc double[3];
            Cross(rxv, r, v);
            Cross(rxrxv, r, rxv);
            for (int k = 0; k < 3; k++)
                res[k] = v[k] + 0.5 * rxv[k] + c * rxrxv[k];
        }

        /// <summary>The minimax coefficients of <see cref="BortzCoef"/> below |x| = 1.75 (MuJoCo's double-precision set).</summary>
        private static readonly double[] BortzK = [
            0.083333333333333384, 0.0013888888888859755, 3.3068783094433201e-05,
            8.2671949071520734e-07, 2.0876900642039142e-08, 5.2828494303844588e-10,
            1.3455485951057888e-11, 3.1615515500778219e-13, 1.230313397310169e-14,
        ];

        /// <summary>
        ///     Bortz's coefficient <c>(1 - (x/2) cot(x/2)) / x²</c> of <c>x² = xx</c> (<c>bortzCoef</c>): a minimax polynomial in
        ///     <c>xx</c> below |x| = 1.75, where the closed form cancels, and the closed form above.
        /// </summary>
        /// <param name="xx">The squared rotation angle, at most π².</param>
        /// <returns>The coefficient.</returns>
        private static double BortzCoef(double xx) {
            if (xx < 1.75 * 1.75) {
                // Horner form, highest coefficient first, as MuJoCo writes it
                double res = BortzK[8];
                for (int i = 7; i >= 0; i--)
                    res = BortzK[i] + xx * res;
                return res;
            }
            double x = Math.Sqrt(xx);
            return (1 - 0.5 * x / Math.Tan(0.5 * x)) / xx;
        }

        /// <summary>The 3-vector dot product, left to right.</summary>
        /// <param name="a">The first vector.</param>
        /// <param name="b">The second vector.</param>
        /// <returns><c>a0*b0 + a1*b1 + a2*b2</c>.</returns>
        public static double Dot3(ReadOnlySpan<double> a, ReadOnlySpan<double> b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

        /// <summary>Normalizes a 3-vector in place (to (1,0,0) when shorter than <see cref="Kernels.MinVal"/>).</summary>
        /// <param name="v">The vector.</param>
        /// <returns>Its length before normalization.</returns>
        public static double Normalize3(Span<double> v) {
            double norm = Math.Sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            if (norm < Kernels.MinVal) {
                v[0] = 1;
                v[1] = 0;
                v[2] = 0;
            } else {
                double inv = 1 / norm;
                v[0] *= inv;
                v[1] *= inv;
                v[2] *= inv;
            }
            return norm;
        }

        /// <summary>
        ///     Normalizes a quaternion in place, leaving it untouched when its length is within <see cref="Kernels.MinVal"/> of 1
        ///     (MuJoCo's shortcut: an already-unit quaternion keeps its exact bits).
        /// </summary>
        /// <param name="q">The quaternion.</param>
        /// <returns>Its length before normalization.</returns>
        public static double Normalize4(Span<double> q) {
            // MuJoCo 4 sums the squares in pairs (mju_normalize4_avx and its scalar twin), not left to right
            double norm = Math.Sqrt((q[0] * q[0] + q[1] * q[1]) + (q[2] * q[2] + q[3] * q[3]));
            if (norm < Kernels.MinVal) {
                q[0] = 1;
                q[1] = 0;
                q[2] = 0;
                q[3] = 0;
            } else if (Math.Abs(norm - 1) > Kernels.MinVal) {
                double inv = 1 / norm;
                q[0] *= inv;
                q[1] *= inv;
                q[2] *= inv;
                q[3] *= inv;
            }
            return norm;
        }

        /// <summary>The quaternion product <c>qa*qb</c>; <paramref name="res"/> may alias either input.</summary>
        /// <param name="res">The output.</param>
        /// <param name="qa">The left quaternion.</param>
        /// <param name="qb">The right quaternion.</param>
        public static void MulQuat(Span<double> res, ReadOnlySpan<double> qa, ReadOnlySpan<double> qb) {
            // Computed into locals first: the hinge update multiplies a body's quaternion into itself. MuJoCo 4's AVX
            // kernel (mju_mulQuat_avx) adds the four products in two pairs, lane by lane; its scalar twin is these sums.
            double r0 = (qa[0] * qb[0] - qa[1] * qb[1]) + (-qa[2] * qb[2] - qa[3] * qb[3]);
            double r1 = (qa[0] * qb[1] + qa[1] * qb[0]) + (qa[2] * qb[3] - qa[3] * qb[2]);
            double r2 = (qa[0] * qb[2] - qa[1] * qb[3]) + (qa[2] * qb[0] + qa[3] * qb[1]);
            double r3 = (qa[0] * qb[3] + qa[1] * qb[2]) + (-qa[2] * qb[1] + qa[3] * qb[0]);
            res[0] = r0;
            res[1] = r1;
            res[2] = r2;
            res[3] = r3;
        }

        /// <summary>Rotates a vector by a unit quaternion; the identity quaternion copies the vector exactly.</summary>
        /// <param name="res">The output.</param>
        /// <param name="vec">The vector.</param>
        /// <param name="q">The quaternion.</param>
        public static void RotVecQuat(Span<double> res, ReadOnlySpan<double> vec, ReadOnlySpan<double> q) {
            if (q[0] == 1 && q[1] == 0 && q[2] == 0 && q[3] == 0) {
                res[0] = vec[0];
                res[1] = vec[1];
                res[2] = vec[2];
                return;
            }
            double t0 = q[0] * vec[0] + q[2] * vec[2] - q[3] * vec[1];
            double t1 = q[0] * vec[1] + q[3] * vec[0] - q[1] * vec[2];
            double t2 = q[0] * vec[2] + q[1] * vec[1] - q[2] * vec[0];
            res[0] = vec[0] + 2 * (q[2] * t2 - q[3] * t1);
            res[1] = vec[1] + 2 * (q[3] * t0 - q[1] * t2);
            res[2] = vec[2] + 2 * (q[1] * t1 - q[2] * t0);
        }

        /// <summary>The quaternion of a rotation by <paramref name="angle"/> about a unit axis; angle 0 gives the exact identity.</summary>
        /// <param name="res">The output.</param>
        /// <param name="axis">The unit axis.</param>
        /// <param name="angle">The angle in radians.</param>
        public static void AxisAngle2Quat(Span<double> res, ReadOnlySpan<double> axis, double angle) {
            if (angle == 0) {
                res[0] = 1;
                res[1] = 0;
                res[2] = 0;
                res[3] = 0;
                return;
            }
            // Math.Sin/Cos call the same UCRT routines mujoco.dll imports, which is what makes the angles bit-identical.
            double s = Math.Sin(angle * 0.5);
            res[0] = Math.Cos(angle * 0.5);
            res[1] = axis[0] * s;
            res[2] = axis[1] * s;
            res[3] = axis[2] * s;
        }

        /// <summary>The rotation matrix of a unit quaternion (the identity quaternion gives the exact identity matrix).</summary>
        /// <param name="res">The 3x3 row-major output.</param>
        /// <param name="q">The quaternion.</param>
        public static void Quat2Mat(Span<double> res, ReadOnlySpan<double> q) {
            if (q[0] == 1 && q[1] == 0 && q[2] == 0 && q[3] == 0) {
                res.Slice(0, 9).Clear();
                res[0] = res[4] = res[8] = 1;
                return;
            }
            double q00 = q[0] * q[0], q01 = q[0] * q[1], q02 = q[0] * q[2], q03 = q[0] * q[3];
            double q11 = q[1] * q[1], q12 = q[1] * q[2], q13 = q[1] * q[3];
            double q22 = q[2] * q[2], q23 = q[2] * q[3], q33 = q[3] * q[3];
            // MuJoCo 4 (mju_quat2Mat_avx) forms the diagonal from paired differences and sums; the off-diagonals are as before
            res[0] = (q00 - q22) + (q11 - q33);
            res[4] = (q00 + q22) - (q11 + q33);
            res[8] = (q00 - q22) - (q11 - q33);
            res[1] = 2 * (q12 - q03);
            res[2] = 2 * (q13 + q02);
            res[3] = 2 * (q12 + q03);
            res[5] = 2 * (q23 - q01);
            res[6] = 2 * (q13 - q02);
            res[7] = 2 * (q23 + q01);
        }

        /// <summary>
        ///     The 10-number spatial inertia of a body about a point offset by <paramref name="dif"/> from its center of mass,
        ///     expressed in the world frame (<c>mju_inertCom</c>).
        /// </summary>
        /// <param name="res">The output: 6 rotational terms, 3 mass-moment terms, the mass.</param>
        /// <param name="inert">The principal inertia.</param>
        /// <param name="mat">The inertial frame's rotation.</param>
        /// <param name="dif">The offset of the center of mass from the reference point.</param>
        /// <param name="mass">The mass.</param>
        public static void InertCom(Span<double> res, ReadOnlySpan<double> inert, ReadOnlySpan<double> mat, ReadOnlySpan<double> dif, double mass) {
            Span<double> t = stackalloc double[9] {
                mat[0] * inert[0], mat[3] * inert[0], mat[6] * inert[0],
                mat[1] * inert[1], mat[4] * inert[1], mat[7] * inert[1],
                mat[2] * inert[2], mat[5] * inert[2], mat[8] * inert[2]
            };
            res[0] = mat[0] * t[0] + mat[1] * t[3] + mat[2] * t[6];
            res[1] = mat[3] * t[1] + mat[4] * t[4] + mat[5] * t[7];
            res[2] = mat[6] * t[2] + mat[7] * t[5] + mat[8] * t[8];
            res[3] = mat[0] * t[1] + mat[1] * t[4] + mat[2] * t[7];
            res[4] = mat[0] * t[2] + mat[1] * t[5] + mat[2] * t[8];
            res[5] = mat[3] * t[2] + mat[4] * t[5] + mat[5] * t[8];
            res[0] += mass * (dif[1] * dif[1] + dif[2] * dif[2]);
            res[1] += mass * (dif[0] * dif[0] + dif[2] * dif[2]);
            res[2] += mass * (dif[0] * dif[0] + dif[1] * dif[1]);
            res[3] -= mass * dif[0] * dif[1];
            res[4] -= mass * dif[0] * dif[2];
            res[5] -= mass * dif[1] * dif[2];
            res[6] = mass * dif[0];
            res[7] = mass * dif[1];
            res[8] = mass * dif[2];
            res[9] = mass;
        }

        /// <summary>The product of a 10-number spatial inertia and a 6-vector motion (<c>mju_mulInertVec</c>).</summary>
        /// <param name="res">The 6-vector force output.</param>
        /// <param name="i">The inertia.</param>
        /// <param name="v">The motion.</param>
        public static void MulInertVec(Span<double> res, ReadOnlySpan<double> i, ReadOnlySpan<double> v) {
            res[0] = i[0] * v[0] + i[3] * v[1] + i[4] * v[2] - i[8] * v[4] + i[7] * v[5];
            res[1] = i[3] * v[0] + i[1] * v[1] + i[5] * v[2] + i[8] * v[3] - i[6] * v[5];
            res[2] = i[4] * v[0] + i[5] * v[1] + i[2] * v[2] - i[7] * v[3] + i[6] * v[4];
            res[3] = i[8] * v[1] - i[7] * v[2] + i[9] * v[3];
            res[4] = i[6] * v[2] - i[8] * v[0] + i[9] * v[4];
            res[5] = i[7] * v[0] - i[6] * v[1] + i[9] * v[5];
        }

        /// <summary>The 6-vector dot product in <see cref="Kernels.Dot"/>'s AVX order (<c>mji_dot6</c>).</summary>
        /// <param name="a">The first 6-vector.</param>
        /// <param name="b">The second 6-vector.</param>
        /// <returns><c>((a0b0 + a2b2) + (a1b1 + a3b3)) + (a4b4 + a5b5)</c>.</returns>
        public static double Dot6(ReadOnlySpan<double> a, ReadOnlySpan<double> b) =>
            ((a[0] * b[0] + a[2] * b[2]) + (a[1] * b[1] + a[3] * b[3])) + (a[4] * b[4] + a[5] * b[5]);

        /// <summary>The motion cross product <c>vel x v</c> (<c>mji_crossMotion</c>).</summary>
        /// <param name="res">The output.</param>
        /// <param name="vel">The spatial velocity.</param>
        /// <param name="v">The motion vector.</param>
        public static void CrossMotion(Span<double> res, ReadOnlySpan<double> vel, ReadOnlySpan<double> v) {
            res[0] = -vel[2] * v[1] + vel[1] * v[2];
            res[1] = vel[2] * v[0] - vel[0] * v[2];
            res[2] = -vel[1] * v[0] + vel[0] * v[1];
            res[3] = -vel[2] * v[4] + vel[1] * v[5];
            res[4] = vel[2] * v[3] - vel[0] * v[5];
            res[5] = -vel[1] * v[3] + vel[0] * v[4];
            res[3] += -vel[5] * v[1] + vel[4] * v[2];
            res[4] += vel[5] * v[0] - vel[3] * v[2];
            res[5] += -vel[4] * v[0] + vel[3] * v[1];
        }

        /// <summary>The force cross product <c>vel x* f</c> (<c>mji_crossForce</c>).</summary>
        /// <param name="res">The output.</param>
        /// <param name="vel">The spatial velocity.</param>
        /// <param name="f">The force vector.</param>
        public static void CrossForce(Span<double> res, ReadOnlySpan<double> vel, ReadOnlySpan<double> f) {
            res[0] = -vel[2] * f[1] + vel[1] * f[2];
            res[1] = vel[2] * f[0] - vel[0] * f[2];
            res[2] = -vel[1] * f[0] + vel[0] * f[1];
            res[3] = -vel[2] * f[4] + vel[1] * f[5];
            res[4] = vel[2] * f[3] - vel[0] * f[5];
            res[5] = -vel[1] * f[3] + vel[0] * f[4];
            res[0] += -vel[5] * f[4] + vel[4] * f[5];
            res[1] += vel[5] * f[3] - vel[3] * f[5];
            res[2] += -vel[4] * f[3] + vel[3] * f[4];
        }
    }
}
