using System;

namespace MujocoPoc;

/// <summary>
///     The vector, matrix and factorization kernels the HalfCheetah core needs, each written to round exactly as MuJoCo
///     3.15.0's win-x64 build does: AVX lanes without FMA, so a dot product sums four lanes and reduces them as
///     <c>(l0 + l2) + (l1 + l3)</c> before a grouped tail, and every element-wise kernel is plain IEEE arithmetic.
/// </summary>
/// <remarks>
///     The operation order is the whole contract: reordering a sum, fusing a multiply-add or skipping a zero test that
///     MuJoCo performs changes the last bit and, a few hundred steps later, the trajectory. The zero tests matter for the
///     sign of zero (MuJoCo skips rows whose scale is zero, so their entries stay +0). Spans are indexed without bounds
///     beyond what the caller passes; a too-short span throws <see cref="IndexOutOfRangeException"/>.
///     Reference: MuJoCo 3.15.0's <see href="https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_util_blas.c">engine_util_blas.c</see>,
///     <see href="https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_util_blas_avx.h">engine_util_blas_avx.h</see>,
///     <see href="https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_util_sparse_avx.h">engine_util_sparse_avx.h</see>,
///     <see href="https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_util_solve.c">engine_util_solve.c</see> and the
///     factorization in <see href="https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_core_smooth.c">engine_core_smooth.c</see>.
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
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i + 1; j < n; j++)
                res[i] -= mat[j * n + i] * res[j];
            res[i] /= mat[i * (n + 1)];
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
///     Reference: <see href="https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_inline.h">engine_inline.h</see>
///     and <see href="https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_util_spatial.c">engine_util_spatial.c</see> at MuJoCo 3.15.0.
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
        double norm = Math.Sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
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
        // Computed into locals first: the hinge update multiplies a body's quaternion into itself.
        double r0 = qa[0] * qb[0] - qa[1] * qb[1] - qa[2] * qb[2] - qa[3] * qb[3];
        double r1 = qa[0] * qb[1] + qa[1] * qb[0] + qa[2] * qb[3] - qa[3] * qb[2];
        double r2 = qa[0] * qb[2] - qa[1] * qb[3] + qa[2] * qb[0] + qa[3] * qb[1];
        double r3 = qa[0] * qb[3] + qa[1] * qb[2] - qa[2] * qb[1] + qa[3] * qb[0];
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
        res[0] = q00 + q11 - q22 - q33;
        res[4] = q00 - q11 + q22 - q33;
        res[8] = q00 - q11 - q22 + q33;
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

/// <summary>Which support function a <see cref="CcdObject"/> answers with.</summary>
internal enum CcdSupport {
    /// <summary>The geom's own shape (sphere, capsule or cylinder).</summary>
    Shape,

    /// <summary>A capsule's segment (<c>mjc_lineSupport</c>), while MuJoCo collides it as a line.</summary>
    Line,

    /// <summary>A sphere's centre (<c>mjc_pointSupport</c>), while MuJoCo collides it as a point.</summary>
    Point,
}

/// <summary>
///     A geom as MuJoCo's convex collider sees it (<c>mjCCDObj</c>): its type, size, frame, the pair's margin and the
///     support function in use. The collider rotates the frame (MultiCCD) and swaps the support (sphere and capsule
///     shrinking) in place, so the caller restores both.
/// </summary>
internal sealed class CcdObject {
    /// <summary>The geom type (2 sphere, 3 capsule, 5 cylinder).</summary>
    public int GeomType;
    /// <summary>The geom size (sphere and capsule: radius, half-length; cylinder: radius, half-height).</summary>
    public readonly double[] Size = new double[3];
    /// <summary>The geom position.</summary>
    public readonly double[] Pos = new double[3];
    /// <summary>The geom rotation matrix.</summary>
    public readonly double[] Mat = new double[9];
    /// <summary>The pair's margin; half of it inflates each support point.</summary>
    public double Margin;
    /// <summary>The support function in use.</summary>
    public CcdSupport Support;

    /// <summary>Sets the object from a geom (<c>mjc_initCCDObj</c>).</summary>
    /// <param name="type">The geom type.</param>
    /// <param name="size">The geom size.</param>
    /// <param name="pos">The geom position.</param>
    /// <param name="mat">The geom rotation matrix.</param>
    /// <param name="margin">The pair's margin.</param>
    public void Init(int type, ReadOnlySpan<double> size, ReadOnlySpan<double> pos, ReadOnlySpan<double> mat, double margin) {
        GeomType = type;
        size.Slice(0, 3).CopyTo(Size);
        pos.Slice(0, 3).CopyTo(Pos);
        mat.Slice(0, 9).CopyTo(Mat);
        Margin = margin;
        Support = CcdSupport.Shape;
    }

    /// <summary>
    ///     The support point in a unit direction: the shape's farthest point along it (<c>mjc_*Support</c>), each in
    ///     MuJoCo's arithmetic.
    /// </summary>
    /// <param name="res">Receives the point.</param>
    /// <param name="dir">The direction (normalized by the caller).</param>
    /// <exception cref="NotSupportedException">A geom type the collider does not reproduce.</exception>
    public void SupportPoint(Span<double> res, ReadOnlySpan<double> dir) {
        if (Support == CcdSupport.Point) {
            res[0] = Pos[0];
            res[1] = Pos[1];
            res[2] = Pos[2];
            return;
        }
        if (Support == CcdSupport.Line) {
            double dot = Mat[2] * dir[0] + Mat[5] * dir[1] + Mat[8] * dir[2];
            double scl = dot >= 0 ? Size[1] : -Size[1];
            res[0] = Mat[2] * scl + Pos[0];
            res[1] = Mat[5] * scl + Pos[1];
            res[2] = Mat[8] * scl + Pos[2];
            return;
        }
        Span<double> local = stackalloc double[3], supp = stackalloc double[3];
        switch (GeomType) {
            case 2:
                res[0] = Size[0] * dir[0] + Pos[0];
                res[1] = Size[0] * dir[1] + Pos[1];
                res[2] = Size[0] * dir[2] + Pos[2];
                return;
            case 3:
                Spatial.MulMatTVec3(local, Mat, dir);
                supp[0] = local[0] * Size[0];
                supp[1] = local[1] * Size[0];
                supp[2] = local[2] * Size[0];
                supp[2] += local[2] >= 0 ? Size[1] : -Size[1];
                break;
            case 5:
                Spatial.MulMatTVec3(local, Mat, dir);
                double n2 = local[0] * local[0] + local[1] * local[1];
                double s = n2 >= Ccd.MinVal2 ? Size[0] / Math.Sqrt(n2) : 0;
                supp[0] = s * local[0];
                supp[1] = s * local[1];
                supp[2] = local[2] >= 0 ? Size[1] : -Size[1];
                break;
            default:
                throw new NotSupportedException($"the convex collider: geom type {GeomType}");
        }
        // localToGlobal: the rotation first, then the position
        Spatial.MulMatVec3(res, Mat, supp);
        res[0] += Pos[0];
        res[1] += Pos[1];
        res[2] += Pos[2];
    }
}

/// <summary>
///     The state of one convex collision query (<c>mjCCDStatus</c>): the witness points and distances it found, its
///     configuration, and GJK's simplex, whose vertices are rows of 9 values (the Minkowski difference point, then the
///     points on the two geoms).
/// </summary>
internal sealed class CcdStatus {
    /// <summary>Whether GJK proved the geoms apart.</summary>
    public bool Separated;
    /// <summary>The distance at each witness pair (negative: penetration).</summary>
    public readonly double[] Dist = new double[Ccd.MaxConPair];
    /// <summary>The witness points on geom 1 (3 per pair).</summary>
    public readonly double[] X1 = new double[3 * Ccd.MaxConPair];
    /// <summary>The witness points on geom 2.</summary>
    public readonly double[] X2 = new double[3 * Ccd.MaxConPair];
    /// <summary>The number of witness pairs.</summary>
    public int Nx;
    /// <summary>The iteration budget of GJK and EPA.</summary>
    public int MaxIterations;
    /// <summary>The tolerance of GJK and EPA.</summary>
    public double Tolerance;
    /// <summary>The number of contacts wanted.</summary>
    public int MaxContacts;
    /// <summary>The distance beyond which GJK may stop proving separation.</summary>
    public double DistCutoff;
    /// <summary>GJK's iterations.</summary>
    public int GjkIterations;
    /// <summary>EPA's iterations.</summary>
    public int EpaIterations;
    /// <summary>EPA's outcome (<c>mjEPAStatus</c>; -1 when EPA did not run).</summary>
    public int EpaStatus;
    /// <summary>GJK's simplex, 4 vertices of 9 values.</summary>
    public readonly double[] Simplex = new double[4 * Ccd.VertexSize];
    /// <summary>The simplex's vertex count.</summary>
    public int Nsimplex;
}

/// <summary>
///     MuJoCo 3.15.0's native convex collision (<c>mjc_ccd</c> of <c>engine_collision_gjk.c</c>): GJK with the
///     signed-volume subalgorithm for distances and intersection, and EPA on an expanding polytope for the penetration
///     depth, written to MuJoCo's arithmetic order. Pusher reaches it for the gripper's capsules against the object's
///     cylinder.
/// </summary>
/// <remarks>
///     Implemented as far as the bundled models reach: spheres and capsules shrink to points and lines (shallow contacts
///     are inflated back), deep contacts run GJK's intersection test and EPA from GJK's segment, triangle or
///     tetrahedron. The multi-contact polygon clipping and the discrete-geom paths (meshes, boxes) throw
///     <see cref="NotSupportedException"/> by name instead of guessing.
/// </remarks>
internal static class Ccd {
    /// <summary>Doubles per simplex or polytope vertex: the difference point, the point on geom 1, the point on geom 2.</summary>
    public const int VertexSize = 9;
    /// <summary>MuJoCo's <c>mjMAXCONPAIR</c>, the witness capacity.</summary>
    public const int MaxConPair = 50;
    /// <summary><c>mjMINVAL*mjMINVAL</c>.</summary>
    public const double MinVal2 = Kernels.MinVal * Kernels.MinVal;
    /// <summary><c>mjMAXVAL*mjMAXVAL</c>.</summary>
    private const double MaxVal2 = 1e10 * 1e10;
    /// <summary><c>mjMAX_LIMIT</c> (<c>DBL_MAX</c>): "no distance".</summary>
    public const double MaxLimit = double.MaxValue;
    /// <summary>EPA status: the hexahedron from a segment is not convex (<c>mjEPA_P2_NONCONVEX</c>).</summary>
    private const int P2Nonconvex = 2;
    /// <summary>EPA status: the triangle's normal is degenerate (<c>mjEPA_P3_BAD_NORMAL</c>).</summary>
    private const int P3BadNormal = 4;
    /// <summary>EPA status: the support point along the normal lies in the triangle (<c>mjEPA_P3_INVALID_V4</c>).</summary>
    private const int P3InvalidV4 = 5;
    /// <summary>EPA status: the support point against the normal lies in the triangle (<c>mjEPA_P3_INVALID_V5</c>).</summary>
    private const int P3InvalidV5 = 6;
    /// <summary>EPA status: the hexahedron misses the origin (<c>mjEPA_P3_MISSING_ORIGIN</c>).</summary>
    private const int P3MissingOrigin = 7;
    /// <summary>EPA status: the origin lies on a face of the hexahedron (<c>mjEPA_P3_ORIGIN_ON_FACE</c>).</summary>
    private const int P3OriginOnFace = 8;
    /// <summary>EPA status when the origin is not in the initial tetrahedron (<c>mjEPA_P4_MISSING_ORIGIN</c>).</summary>
    private const int P4MissingOrigin = 9;

    /// <summary>
    ///     One convex query (<c>mjc_ccd</c>) with no distance cutoff: GJK from the geom centres, shrinking a sphere or
    ///     capsule to its point or segment first, then for a deep contact GJK on the full shapes and EPA.
    /// </summary>
    /// <param name="status">Receives the witness points, distances and statistics.</param>
    /// <param name="poly">The polytope workspace, sized for <paramref name="maxIterations"/>.</param>
    /// <param name="obj1">The first geom (restored on return).</param>
    /// <param name="obj2">The second geom (restored on return).</param>
    /// <param name="maxIterations">GJK's and EPA's budget (<c>opt.ccd_iterations</c>).</param>
    /// <param name="tolerance">Their tolerance (<c>opt.ccd_tolerance</c>).</param>
    /// <param name="maxContacts">The contacts wanted (1 here).</param>
    /// <returns>The smallest witness distance: negative for penetration, <see cref="MaxLimit"/> when apart.</returns>
    /// <exception cref="NotSupportedException">A path the collider does not reproduce (multi-contact polygon clipping).</exception>
    public static double Run(CcdStatus status, CcdPolytope poly, CcdObject obj1, CcdObject obj2, int maxIterations, double tolerance,
                             int maxContacts) {
        const double configCutoff = 0;
        obj1.Pos.CopyTo(status.X1, 0);
        obj2.Pos.CopyTo(status.X2, 0);
        status.GjkIterations = 0;
        status.EpaIterations = 0;
        status.EpaStatus = -1;
        status.Tolerance = tolerance;
        status.MaxIterations = maxIterations;
        status.MaxContacts = maxContacts;
        status.DistCutoff = configCutoff;
        if (obj1.GeomType == 2 || obj2.GeomType == 2 || obj1.GeomType == 3 || obj2.GeomType == 3) {
            CcdSupport support1 = obj1.Support, support2 = obj2.Support;
            double fullMargin1 = 0, fullMargin2 = 0, margin1 = obj1.Margin, margin2 = obj2.Margin;
            if (obj1.GeomType == 2 || obj1.GeomType == 3) {
                fullMargin1 = obj1.Size[0] + 0.5 * margin1;
                obj1.Support = obj1.GeomType == 2 ? CcdSupport.Point : CcdSupport.Line;
                obj1.Margin = 0;
            }
            if (obj2.GeomType == 2 || obj2.GeomType == 3) {
                fullMargin2 = obj2.Size[0] + 0.5 * margin2;
                obj2.Support = obj2.GeomType == 2 ? CcdSupport.Point : CcdSupport.Line;
                obj2.Margin = 0;
            }
            status.DistCutoff += fullMargin1 + fullMargin2;
            Gjk(status, obj1, obj2);
            status.DistCutoff = configCutoff;
            obj1.Margin = margin1;
            obj2.Margin = margin2;
            obj1.Support = support1;
            obj2.Support = support2;
            // shallow penetration: inflate the point or segment back to the shape
            if (status.Dist[0] > status.Tolerance) {
                Inflate(status, fullMargin1, fullMargin2);
                if (status.Dist[0] > status.DistCutoff) {
                    status.Dist[0] = MaxLimit;
                } else if (maxContacts > 1 && status.Dist[0] < 0 && status.Nsimplex > 0) {
                    throw new NotSupportedException("the convex collider: multi-contact polygon clipping");
                }
                return status.Dist[0];
            }
            if (maxContacts == 0) {
                status.Nx = 0;
                status.Dist[0] = 0;
                return 0;
            }
            // deep penetration: GJK again on the full shapes, from the centres
            status.GjkIterations = 0;
            obj1.Pos.CopyTo(status.X1, 0);
            obj2.Pos.CopyTo(status.X2, 0);
        }
        Gjk(status, obj1, obj2);
        if (maxContacts == 0)
            return status.Dist[0];
        if (status.Dist[0] <= tolerance && status.Nsimplex > 1 && !status.Separated) {
            status.Dist[0] = 0;
            poly.Reset(maxIterations);
            int ret = status.Nsimplex switch {
                2 => Polytope2(poly, status, obj1, obj2),
                3 => Polytope3(poly, status, obj1, obj2),
                _ => Polytope4(poly, status, obj1, obj2),
            };
            status.EpaStatus = ret;
            if (ret == 0) {
                int face = Epa(status, poly, obj1, obj2);
                if (maxContacts > 1 && face >= 0)
                    throw new NotSupportedException("the convex collider: multi-contact polygon clipping");
            }
        }
        double minDist = status.Dist[0];
        for (int i = 1; i < status.Nx; i++)
            if (status.Dist[i] < minDist)
                minDist = status.Dist[i];
        return minDist;
    }

    /// <summary>
    ///     Moves the witness points outward by the margins a shrunk sphere or capsule left out (<c>inflate</c>), along the
    ///     line between them, and lowers the distance by both.
    /// </summary>
    /// <param name="status">The status, whose first witness pair and distance change.</param>
    /// <param name="margin1">Geom 1's radius plus half the margin, or 0.</param>
    /// <param name="margin2">Geom 2's, or 0.</param>
    private static void Inflate(CcdStatus status, double margin1, double margin2) {
        Span<double> n = stackalloc double[3];
        for (int k = 0; k < 3; k++)
            n[k] = status.X2[k] - status.X1[k];
        Spatial.Normalize3(n);
        if (margin1 != 0)
            for (int k = 0; k < 3; k++)
                status.X1[k] += margin1 * n[k];
        if (margin2 != 0)
            for (int k = 0; k < 3; k++)
                status.X2[k] -= margin2 * n[k];
        status.Dist[0] -= margin1 + margin2;
    }

    // ------------------------------------------------------------------------------------------ vector helpers

    /// <summary><c>dot3</c> of two spans.</summary>
    /// <param name="a">The first vector (its first three entries).</param>
    /// <param name="b">The second vector (its first three entries).</param>
    /// <returns>The sum of the three products, left to right as MuJoCo's <c>dot3</c> adds them.</returns>
    private static double Dot3(ReadOnlySpan<double> a, ReadOnlySpan<double> b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

    /// <summary>
    ///     <c>det3</c>: the determinant of the matrix with columns <paramref name="v1"/>, <paramref name="v2"/>,
    ///     <paramref name="v3"/>, as <c>v1 · (v2 x v3)</c> written out.
    /// </summary>
    /// <param name="v1">The first column.</param>
    /// <param name="v2">The second column.</param>
    /// <param name="v3">The third column.</param>
    /// <returns>The determinant, signed: positive when the three columns are right-handed.</returns>
    private static double Det3(ReadOnlySpan<double> v1, ReadOnlySpan<double> v2, ReadOnlySpan<double> v3) =>
        v1[0] * (v2[1] * v3[2] - v2[2] * v3[1])
        + v1[1] * (v2[2] * v3[0] - v2[0] * v3[2])
        + v1[2] * (v2[0] * v3[1] - v2[1] * v3[0]);

    /// <summary>
    ///     <c>lincomb</c>: <c>sum coef[i]*v_i</c> over the first <paramref name="n"/> 3-vectors, left to right; zeros for
    ///     none.
    /// </summary>
    /// <param name="res">Receives the combination.</param>
    /// <param name="coef">The coefficients.</param>
    /// <param name="n">How many vectors.</param>
    /// <param name="src">The array holding the vectors.</param>
    /// <param name="o1">Offset of the first vector.</param>
    /// <param name="o2">Offset of the second.</param>
    /// <param name="o3">Offset of the third.</param>
    /// <param name="o4">Offset of the fourth.</param>
    private static void Lincomb(Span<double> res, ReadOnlySpan<double> coef, int n, double[] src, int o1, int o2, int o3, int o4) {
        for (int k = 0; k < 3; k++) {
            res[k] = n switch {
                0 => 0,
                1 => coef[0] * src[o1 + k],
                2 => coef[0] * src[o1 + k] + coef[1] * src[o2 + k],
                3 => coef[0] * src[o1 + k] + coef[1] * src[o2 + k] + coef[2] * src[o3 + k],
                _ => coef[0] * src[o1 + k] + coef[1] * src[o2 + k] + coef[2] * src[o3 + k] + coef[3] * src[o4 + k],
            };
        }
    }

    /// <summary><c>sameSign2</c>: 1 if both are positive, -1 if both negative, else 0.</summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    /// <returns>1, -1 or 0; a zero in either value gives 0, which the subalgorithms read as disagreement.</returns>
    private static int SameSign2(double a, double b) => a > 0 && b > 0 ? 1 : a < 0 && b < 0 ? -1 : 0;

    // ------------------------------------------------------------------------------------------ support points

    /// <summary>
    ///     The Minkowski-difference vertex in a direction (<c>support</c>): each geom's support point (geom 2's in the
    ///     opposite direction), each moved out by half its margin, and their difference.
    /// </summary>
    /// <param name="v">The vertex storage.</param>
    /// <param name="o">The vertex's offset (its 9 values).</param>
    /// <param name="obj1">Geom 1.</param>
    /// <param name="obj2">Geom 2.</param>
    /// <param name="dir">The direction.</param>
    /// <param name="dirNeg">Its opposite.</param>
    private static void Support(double[] v, int o, CcdObject obj1, CcdObject obj2, ReadOnlySpan<double> dir, ReadOnlySpan<double> dirNeg) {
        Span<double> p1 = v.AsSpan(o + 3, 3), p2 = v.AsSpan(o + 6, 3);
        obj1.SupportPoint(p1, dir);
        if (obj1.Margin > 0) {
            double margin = 0.5 * obj1.Margin;
            p1[0] += dir[0] * margin;
            p1[1] += dir[1] * margin;
            p1[2] += dir[2] * margin;
        }
        obj2.SupportPoint(p2, dirNeg);
        if (obj2.Margin > 0) {
            double margin = 0.5 * obj2.Margin;
            p2[0] += dirNeg[0] * margin;
            p2[1] += dirNeg[1] * margin;
            p2[2] += dirNeg[2] * margin;
        }
        v[o] = p1[0] - p2[0];
        v[o + 1] = p1[1] - p2[1];
        v[o + 2] = p1[2] - p2[2];
    }

    /// <summary>GJK's support vertex toward the origin from <paramref name="xk"/> (<c>gjkSupport</c>).</summary>
    /// <param name="v">The vertex array that receives the support vertex (the simplex).</param>
    /// <param name="o">The vertex's offset in <paramref name="v"/>; the vertex takes <c>VertexSize</c> entries.</param>
    /// <param name="obj1">Geom 1, supported along <c>-xk</c>.</param>
    /// <param name="obj2">Geom 2, supported along <c>xk</c>.</param>
    /// <param name="xk">The current closest point of the Minkowski difference to the origin.</param>
    /// <param name="xNorm">The norm of <paramref name="xk"/>, nonzero (GJK stops before a zero one).</param>
    private static void GjkSupport(double[] v, int o, CcdObject obj1, CcdObject obj2, ReadOnlySpan<double> xk, double xNorm) {
        Span<double> dir = stackalloc double[3], dirNeg = stackalloc double[3];
        double s = 1 / xNorm;
        for (int k = 0; k < 3; k++)
            dirNeg[k] = s * xk[k];
        for (int k = 0; k < 3; k++)
            dir[k] = -1 * dirNeg[k];
        Support(v, o, obj1, obj2, dir, dirNeg);
    }

    /// <summary>
    ///     EPA's support vertex along a face's normal (<c>epaSupport</c>), appended to the polytope; the x axis when the
    ///     normal is degenerate.
    /// </summary>
    /// <param name="pt">The polytope, which gains the vertex.</param>
    /// <param name="obj1">Geom 1, supported along the normal.</param>
    /// <param name="obj2">Geom 2, supported against it.</param>
    /// <param name="d">The face's normal, not normalized.</param>
    /// <param name="dnorm">The normal's norm; at or below <c>mjMINVAL</c> the x axis is used instead.</param>
    /// <returns>The new vertex's index.</returns>
    private static int EpaSupport(CcdPolytope pt, CcdObject obj1, CcdObject obj2, ReadOnlySpan<double> d, double dnorm) {
        Span<double> dir = stackalloc double[3] { 1, 0, 0 }, dirNeg = stackalloc double[3] { -1, 0, 0 };
        if (dnorm > Kernels.MinVal) {
            dir[0] = d[0] / dnorm;
            dir[1] = d[1] / dnorm;
            dir[2] = d[2] / dnorm;
            for (int k = 0; k < 3; k++)
                dirNeg[k] = -1 * dir[k];
        }
        int n = pt.Nverts++;
        Support(pt.Verts, n * VertexSize, obj1, obj2, dir, dirNeg);
        return n;
    }

    // ------------------------------------------------------------------------------------------ GJK

    /// <summary>
    ///     GJK (<c>gjk</c>): from the witness guess in the status, support points toward the origin and the signed-volume
    ///     subalgorithm shrink the simplex to the part closest to the origin, until the Frank-Wolfe gap or the norm stops
    ///     changing, a tetrahedron encloses the origin, or the geoms are proved apart. Without a distance cutoff, a full
    ///     tetrahedron hands over to <see cref="GjkIntersect"/>.
    /// </summary>
    /// <param name="status">The status: witness guess in, distance, witnesses and simplex out.</param>
    /// <param name="obj1">Geom 1.</param>
    /// <param name="obj2">Geom 2.</param>
    private static void Gjk(CcdStatus status, CcdObject obj1, CcdObject obj2) {
        bool getDist = status.DistCutoff > 0;
        bool backupGjk = !getDist;
        double[] simplex = status.Simplex;
        int n = 0, k = 0, kmax = status.MaxIterations;
        Span<double> xk = stackalloc double[3], lambda = stackalloc double[4], diff = stackalloc double[3];
        double tol2 = status.Tolerance * status.Tolerance;
        status.Separated = false;
        // capsule, cylinder and sphere are not discrete geoms: tolerances stay positive
        double epsilon = 0.5 * tol2;
        double minNorm = status.Tolerance;
        for (int i = 0; i < 3; i++)
            xk[i] = status.X1[i] - status.X2[i];
        double xNorm = Math.Sqrt(Dot3(xk, xk)), xNormPrev = 0;
        for (; k < kmax; k++) {
            if (xNorm < minNorm || Math.Abs(xNormPrev - xNorm) < Kernels.MinVal)
                break;
            GjkSupport(simplex, n * VertexSize, obj1, obj2, xk, xNorm);
            int sk = n * VertexSize;
            for (int i = 0; i < 3; i++)
                diff[i] = xk[i] - simplex[sk + i];
            if (Dot3(xk, diff) < epsilon)
                break;
            double lower = xk[0] * simplex[sk] + xk[1] * simplex[sk + 1] + xk[2] * simplex[sk + 2];
            if (!getDist) {
                if (lower > 0) {
                    SetApart(status, k);
                    return;
                }
            } else if (status.DistCutoff < MaxLimit) {
                if (lower > 0 && lower >= status.DistCutoff * xNorm) {
                    SetApart(status, k);
                    return;
                }
            }
            if (n == 3 && backupGjk) {
                status.GjkIterations = k;
                int ret = GjkIntersect(status, obj1, obj2);
                if (ret != -1) {
                    status.Nx = 0;
                    status.Separated = ret == 0;
                    status.Dist[0] = ret > 0 ? 0 : MaxLimit;
                    return;
                }
                k = status.GjkIterations;
                backupGjk = false;
            }
            Subdistance(lambda, n + 1, simplex);
            n = 0;
            for (int i = 0; i < 4; i++) {
                if (lambda[i] == 0)
                    continue;
                if (n != i)
                    Array.Copy(simplex, i * VertexSize, simplex, n * VertexSize, VertexSize);
                lambda[n++] = lambda[i];
            }
            if (n < 1) {
                status.GjkIterations = k;
                status.Nsimplex = 0;
                status.Nx = 0;
                status.Dist[0] = MaxLimit;
                status.Separated = true;
                return;
            }
            Lincomb(xk, lambda, n, simplex, 0, VertexSize, 2 * VertexSize, 3 * VertexSize);
            xNormPrev = xNorm;
            xNorm = Math.Sqrt(Dot3(xk, xk));
            if (n == 4)
                break;
        }
        if (n > 0) {
            Lincomb(status.X1, lambda, n, simplex, 3, VertexSize + 3, 2 * VertexSize + 3, 3 * VertexSize + 3);
            Lincomb(status.X2, lambda, n, simplex, 6, VertexSize + 6, 2 * VertexSize + 6, 3 * VertexSize + 6);
        }
        // a final separation check with one more support point
        double[] tmp = new double[VertexSize];
        GjkSupport(tmp, 0, obj1, obj2, xk, xNorm);
        if (xk[0] * tmp[0] + xk[1] * tmp[1] + xk[2] * tmp[2] > 0)
            status.Separated = true;
        status.Nx = 1;
        status.GjkIterations = k;
        status.Nsimplex = n;
        status.Dist[0] = n == 4 && !status.Separated ? 0 : xNorm;
    }

    /// <summary>Records GJK's proof that the geoms are apart.</summary>
    /// <param name="status">
    ///     The status: marked separated, its simplex and witnesses cleared, its distance set to the maximum.
    /// </param>
    /// <param name="k">The GJK iterations spent, reported in the status.</param>
    private static void SetApart(CcdStatus status, int k) {
        status.Separated = true;
        status.GjkIterations = k;
        status.Nsimplex = 0;
        status.Nx = 0;
        status.Dist[0] = MaxLimit;
    }

    /// <summary>
    ///     The signed distance of the origin to a face's plane and the face's unit normal (<c>signedDistance</c>);
    ///     <see cref="MaxLimit"/> when the normal cannot be recovered.
    /// </summary>
    /// <param name="normal">Receives the face's unit normal (unnormalized when the method returns the maximum).</param>
    /// <param name="s">The vertex array (the simplex).</param>
    /// <param name="v1">The offset of the face's first vertex in <paramref name="s"/>.</param>
    /// <param name="v2">The offset of its second vertex.</param>
    /// <param name="v3">The offset of its third vertex.</param>
    /// <returns>
    ///     The plane's signed offset along the unit normal (<c>n·v1</c>), positive when the origin lies on the side the
    ///     normal points away from; <see cref="MaxLimit"/> for a degenerate face.
    /// </returns>
    private static double SignedDistance(Span<double> normal, double[] s, int v1, int v2, int v3) {
        Span<double> diff1 = stackalloc double[3], diff2 = stackalloc double[3];
        for (int k = 0; k < 3; k++) {
            diff1[k] = s[v3 + k] - s[v1 + k];
            diff2[k] = s[v2 + k] - s[v1 + k];
        }
        Spatial.Cross(normal, diff1, diff2);
        double norm2 = Dot3(normal, normal);
        if (norm2 > MinVal2 && norm2 < MaxVal2) {
            double scl = 1 / Math.Sqrt(norm2);
            for (int k = 0; k < 3; k++)
                normal[k] = scl * normal[k];
            return normal[0] * s[v1] + normal[1] * s[v1 + 1] + normal[2] * s[v1 + 2];
        }
        return MaxLimit;
    }

    /// <summary>
    ///     GJK's intersection test (<c>gjkIntersect</c>): the tetrahedron's faces nearest the origin are replaced by support
    ///     points beyond them until it encloses the origin (contact) or a support point fails to pass it (no contact).
    /// </summary>
    /// <param name="status">The status: GJK's tetrahedron in, the enclosing tetrahedron and the iterations out.</param>
    /// <param name="obj1">Geom 1.</param>
    /// <param name="obj2">Geom 2.</param>
    /// <returns>1 contact (the simplex is the enclosing tetrahedron), 0 none, -1 inconclusive.</returns>
    private static int GjkIntersect(CcdStatus status, CcdObject obj1, CcdObject obj2) {
        double[] simplex = (double[])status.Simplex.Clone();
        Span<int> s = stackalloc int[4] { 0, 1, 2, 3 };
        Span<double> dist = stackalloc double[4], normals = stackalloc double[12], dirNeg = stackalloc double[3];
        int k = status.GjkIterations, kmax = status.MaxIterations;
        for (; k < kmax; k++) {
            dist[0] = SignedDistance(normals.Slice(0, 3), simplex, s[2] * VertexSize, s[1] * VertexSize, s[3] * VertexSize);
            dist[1] = SignedDistance(normals.Slice(3, 3), simplex, s[0] * VertexSize, s[2] * VertexSize, s[3] * VertexSize);
            dist[2] = SignedDistance(normals.Slice(6, 3), simplex, s[1] * VertexSize, s[0] * VertexSize, s[3] * VertexSize);
            dist[3] = SignedDistance(normals.Slice(9, 3), simplex, s[0] * VertexSize, s[1] * VertexSize, s[2] * VertexSize);
            if (dist[3] == 0 || dist[2] == 0 || dist[1] == 0 || dist[0] == 0) {
                status.GjkIterations = k;
                return -1;
            }
            int i = dist[0] < dist[1] ? 0 : 1;
            int j = dist[2] < dist[3] ? 2 : 3;
            int index = dist[i] < dist[j] ? i : j;
            if (dist[index] > 0) {
                status.Nsimplex = 4;
                for (int q = 0; q < 4; q++)
                    Array.Copy(simplex, s[q] * VertexSize, status.Simplex, q * VertexSize, VertexSize);
                status.GjkIterations = k;
                return 1;
            }
            // replace the face's opposite vertex with a support point beyond the face
            ReadOnlySpan<double> nrm = normals.Slice(3 * index, 3);
            dirNeg[0] = -nrm[0];
            dirNeg[1] = -nrm[1];
            dirNeg[2] = -nrm[2];
            Support(simplex, s[index] * VertexSize, obj1, obj2, nrm, dirNeg);
            int o = s[index] * VertexSize;
            if (nrm[0] * simplex[o] + nrm[1] * simplex[o + 1] + nrm[2] * simplex[o + 2] < 0) {
                status.Nsimplex = 0;
                status.GjkIterations = k;
                return 0;
            }
            // swap two vertices to keep the orientation
            i = (index + 1) & 3;
            j = (index + 2) & 3;
            (s[i], s[j]) = (s[j], s[i]);
        }
        status.GjkIterations = k;
        return -1;
    }

    // ------------------------------------------------------------------------------------------ signed-volume subalgorithm

    /// <summary>
    ///     The barycentric coordinates of the simplex point closest to the origin (<c>subdistance</c>, Montanari et al.):
    ///     S3D for a tetrahedron, S2D for a triangle, S1D for a segment, the vertex itself for one.
    /// </summary>
    /// <param name="lambda">Receives the four barycentric coordinates; zero for the vertices dropped.</param>
    /// <param name="n">The simplex's vertex count, 1 to 4.</param>
    /// <param name="simplex">The simplex's vertices, <c>VertexSize</c> entries each.</param>
    private static void Subdistance(Span<double> lambda, int n, double[] simplex) {
        lambda.Clear();
        ReadOnlySpan<double> s1 = simplex.AsSpan(0, 3), s2 = simplex.AsSpan(VertexSize, 3);
        ReadOnlySpan<double> s3 = simplex.AsSpan(2 * VertexSize, 3), s4 = simplex.AsSpan(3 * VertexSize, 3);
        switch (n) {
            case 4:
                S3D(lambda, s1, s2, s3, s4);
                break;
            case 3:
                S2D(lambda, s1, s2, s3);
                break;
            case 2:
                S1D(lambda, s1, s2);
                break;
            default:
                lambda[0] = 1;
                break;
        }
    }

    /// <summary>
    ///     The closest point of a tetrahedron (<c>S3D</c>): the origin's own coordinates when all four cofactors share the
    ///     determinant's sign, else the best of the faces whose cofactor disagrees, by squared distance.
    /// </summary>
    /// <param name="lambda">Receives the four coordinates.</param>
    /// <param name="s1">The first vertex.</param>
    /// <param name="s2">The second vertex.</param>
    /// <param name="s3">The third vertex.</param>
    /// <param name="s4">The fourth vertex.</param>
    private static void S3D(Span<double> lambda, ReadOnlySpan<double> s1, ReadOnlySpan<double> s2, ReadOnlySpan<double> s3, ReadOnlySpan<double> s4) {
        double c41 = -Det3(s2, s3, s4), c42 = Det3(s1, s3, s4), c43 = -Det3(s1, s2, s4), c44 = Det3(s1, s2, s3);
        double mDet = c41 + c42 + c43 + c44;
        int comp1 = SameSign2(mDet, c41), comp2 = SameSign2(mDet, c42), comp3 = SameSign2(mDet, c43), comp4 = SameSign2(mDet, c44);
        if (comp1 != 0 && comp2 != 0 && comp3 != 0 && comp4 != 0) {
            lambda[0] = c41 / mDet;
            lambda[1] = c42 / mDet;
            lambda[2] = c43 / mDet;
            lambda[3] = c44 / mDet;
            return;
        }
        // each face whose cofactor disagrees, in MuJoCo's order: the first unconditionally, the others when closer
        double dmin = MaxLimit;
        Span<double> l2 = stackalloc double[3], x = stackalloc double[3];
        if (comp1 == 0)
            S3DFace(lambda, 0, s2, s3, s4, l2, x, ref dmin, true);
        if (comp2 == 0)
            S3DFace(lambda, 1, s1, s3, s4, l2, x, ref dmin, false);
        if (comp3 == 0)
            S3DFace(lambda, 2, s1, s2, s4, l2, x, ref dmin, false);
        if (comp4 == 0)
            S3DFace(lambda, 3, s1, s2, s3, l2, x, ref dmin, false);
    }

    /// <summary>
    ///     One face of <see cref="S3D"/>'s fallback: the face's closest point by <see cref="S2D"/>, taken when it is the
    ///     first face tried or closer than the best so far (MuJoCo's <c>d &lt; dmin</c>, so a NaN never wins).
    /// </summary>
    /// <param name="lambda">The tetrahedron's coordinates, written when the face is taken (0 at the skipped vertex).</param>
    /// <param name="skip">The vertex the face leaves out.</param>
    /// <param name="a">The face's first vertex.</param>
    /// <param name="b">Its second.</param>
    /// <param name="c">Its third.</param>
    /// <param name="l2">Scratch for the face's coordinates.</param>
    /// <param name="x">Scratch for the face's closest point.</param>
    /// <param name="best">The best squared distance so far, updated when the face is taken.</param>
    /// <param name="unconditional">Whether the face is taken regardless (MuJoCo's first branch).</param>
    private static void S3DFace(Span<double> lambda, int skip, ReadOnlySpan<double> a, ReadOnlySpan<double> b, ReadOnlySpan<double> c,
                                Span<double> l2, Span<double> x, ref double best, bool unconditional) {
        S2D(l2, a, b, c);
        for (int k = 0; k < 3; k++)
            x[k] = l2[0] * a[k] + l2[1] * b[k] + l2[2] * c[k];
        double d = Dot3(x, x);
        if (unconditional || d < best) {
            int idx = 0;
            for (int q = 0; q < 4; q++)
                lambda[q] = q == skip ? 0 : l2[idx++];
            best = d;
        }
    }

    /// <summary>
    ///     The closest point of a triangle (<c>S2D</c>): the origin projected on its plane, its 2-D coordinates on the
    ///     projection that keeps the most area, else the best of the edges whose cofactor disagrees; a degenerate plane
    ///     falls back to the first edge.
    /// </summary>
    /// <param name="lambda">Receives the three coordinates.</param>
    /// <param name="s1">The first vertex.</param>
    /// <param name="s2">The second vertex.</param>
    /// <param name="s3">The third vertex.</param>
    private static void S2D(Span<double> lambda, ReadOnlySpan<double> s1, ReadOnlySpan<double> s2, ReadOnlySpan<double> s3) {
        Span<double> po = stackalloc double[3];
        if (ProjectOriginPlane(po, s1, s2, s3) != 0) {
            S1D(lambda, s1, s2);
            lambda[2] = 0;
            return;
        }
        double m14 = s2[1] * s3[2] - s2[2] * s3[1] - s1[1] * s3[2] + s1[2] * s3[1] + s1[1] * s2[2] - s1[2] * s2[1];
        double m24 = s2[0] * s3[2] - s2[2] * s3[0] - s1[0] * s3[2] + s1[2] * s3[0] + s1[0] * s2[2] - s1[2] * s2[0];
        double m34 = s2[0] * s3[1] - s2[1] * s3[0] - s1[0] * s3[1] + s1[1] * s3[0] + s1[0] * s2[1] - s1[1] * s2[0];
        double mu1 = Math.Abs(m14), mu2 = Math.Abs(m24), mu3 = Math.Abs(m34);
        int x, y;
        double mMax;
        if (mu1 >= mu2 && mu1 >= mu3) {
            mMax = m14;
            x = 1;
            y = 2;
        } else if (mu2 >= mu3) {
            mMax = m24;
            x = 0;
            y = 2;
        } else {
            mMax = m34;
            x = 0;
            y = 1;
        }
        double c31 = po[x] * s2[y] + po[y] * s3[x] + s2[x] * s3[y] - po[x] * s3[y] - po[y] * s2[x] - s3[x] * s2[y];
        double c32 = po[x] * s3[y] + po[y] * s1[x] + s3[x] * s1[y] - po[x] * s1[y] - po[y] * s3[x] - s1[x] * s3[y];
        double c33 = po[x] * s1[y] + po[y] * s2[x] + s1[x] * s2[y] - po[x] * s2[y] - po[y] * s1[x] - s2[x] * s1[y];
        int comp1 = SameSign2(mMax, c31), comp2 = SameSign2(mMax, c32), comp3 = SameSign2(mMax, c33);
        if (comp1 != 0 && comp2 != 0 && comp3 != 0) {
            lambda[0] = c31 / mMax;
            lambda[1] = c32 / mMax;
            lambda[2] = c33 / mMax;
            return;
        }
        // each edge whose cofactor disagrees, in MuJoCo's order: the first unconditionally, the others when closer
        double dmin = MaxLimit;
        Span<double> l1 = stackalloc double[2], p = stackalloc double[3];
        if (comp1 == 0) {
            S1D(l1, s2, s3);
            for (int k = 0; k < 3; k++)
                p[k] = l1[0] * s2[k] + l1[1] * s3[k];
            lambda[0] = 0;
            lambda[1] = l1[0];
            lambda[2] = l1[1];
            dmin = Dot3(p, p);
        }
        if (comp2 == 0) {
            S1D(l1, s1, s3);
            for (int k = 0; k < 3; k++)
                p[k] = l1[0] * s1[k] + l1[1] * s3[k];
            double d = Dot3(p, p);
            if (d < dmin) {
                lambda[0] = l1[0];
                lambda[1] = 0;
                lambda[2] = l1[1];
                dmin = d;
            }
        }
        if (comp3 == 0) {
            S1D(l1, s1, s2);
            for (int k = 0; k < 3; k++)
                p[k] = l1[0] * s1[k] + l1[1] * s2[k];
            double d = Dot3(p, p);
            if (d < dmin) {
                lambda[0] = l1[0];
                lambda[1] = l1[1];
                lambda[2] = 0;
            }
        }
    }

    /// <summary>
    ///     The closest point of a segment (<c>S1D</c>): the origin projected on its line, its coordinate on the axis of the
    ///     longest shadow, or the second vertex when the projection falls outside.
    /// </summary>
    /// <param name="lambda">Receives the two coordinates.</param>
    /// <param name="s1">The first vertex.</param>
    /// <param name="s2">The second vertex.</param>
    private static void S1D(Span<double> lambda, ReadOnlySpan<double> s1, ReadOnlySpan<double> s2) {
        Span<double> po = stackalloc double[3];
        ProjectOriginLine(po, s1, s2);
        double mu = s1[0] - s2[0], muMax = mu;
        int index = 0;
        mu = s1[1] - s2[1];
        if (Math.Abs(mu) >= Math.Abs(muMax)) {
            muMax = mu;
            index = 1;
        }
        mu = s1[2] - s2[2];
        if (Math.Abs(mu) >= Math.Abs(muMax)) {
            muMax = mu;
            index = 2;
        }
        double c1 = po[index] - s2[index], c2 = s1[index] - po[index];
        bool same = SameSign2(muMax, c1) != 0 && SameSign2(muMax, c2) != 0;
        lambda[0] = same ? c1 / muMax : 0;
        lambda[1] = same ? c2 / muMax : 1;
    }

    /// <summary>
    ///     The origin projected on a triangle's plane (<c>projectOriginPlane</c>), trying the normals at the second, first
    ///     and third vertex in turn until one is usable.
    /// </summary>
    /// <param name="res">Receives the projection; untouched for a degenerate triangle.</param>
    /// <param name="v1">The first vertex.</param>
    /// <param name="v2">The second vertex.</param>
    /// <param name="v3">The third vertex.</param>
    /// <returns>1 for a degenerate triangle (zero normal), else 0.</returns>
    private static int ProjectOriginPlane(Span<double> res, ReadOnlySpan<double> v1, ReadOnlySpan<double> v2, ReadOnlySpan<double> v3) {
        Span<double> d21 = stackalloc double[3], d31 = stackalloc double[3], d32 = stackalloc double[3], n = stackalloc double[3];
        for (int k = 0; k < 3; k++) {
            d21[k] = v2[k] - v1[k];
            d31[k] = v3[k] - v1[k];
            d32[k] = v3[k] - v2[k];
        }
        Spatial.Cross(n, d32, d21);
        double nv = Dot3(n, v2), nn = Dot3(n, n);
        if (nn == 0)
            return 1;
        if (nv != 0 && nn > Kernels.MinVal) {
            Scale(res, n, nv / nn);
            return 0;
        }
        Spatial.Cross(n, d21, d31);
        nv = Dot3(n, v1);
        nn = Dot3(n, n);
        if (nn == 0)
            return 1;
        if (nv != 0 && nn > Kernels.MinVal) {
            Scale(res, n, nv / nn);
            return 0;
        }
        Spatial.Cross(n, d31, d32);
        nv = Dot3(n, v3);
        nn = Dot3(n, n);
        Scale(res, n, nv / nn);
        return 0;
    }

    /// <summary><c>scl3</c>: <c>s*v</c>.</summary>
    /// <param name="res">Receives the scaled vector; may alias <paramref name="v"/>.</param>
    /// <param name="v">The vector.</param>
    /// <param name="s">The factor.</param>
    private static void Scale(Span<double> res, ReadOnlySpan<double> v, double s) {
        res[0] = s * v[0];
        res[1] = s * v[1];
        res[2] = s * v[2];
    }

    /// <summary>The origin projected on a line (<c>projectOriginLine</c>): <c>v2 - (v2·d / d·d)*d</c> with <c>d = v2 - v1</c>.</summary>
    /// <param name="res">Receives the projection.</param>
    /// <param name="v1">The line's first point.</param>
    /// <param name="v2">Its second point.</param>
    private static void ProjectOriginLine(Span<double> res, ReadOnlySpan<double> v1, ReadOnlySpan<double> v2) {
        Span<double> diff = stackalloc double[3];
        for (int k = 0; k < 3; k++)
            diff[k] = v2[k] - v1[k];
        double scl = -(Dot3(v2, diff) / Dot3(diff, diff));
        for (int k = 0; k < 3; k++)
            res[k] = v2[k] + scl * diff[k];
    }

    // ------------------------------------------------------------------------------------------ EPA

    /// <summary>
    ///     The initial polytope from GJK's segment (<c>polytope2</c>): three support points around the segment, 120
    ///     degrees apart about it, make a hexahedron of six faces; a face through the origin hands the problem to
    ///     <see cref="Polytope3"/>, and a non-convex hexahedron stops EPA.
    /// </summary>
    /// <param name="pt">The polytope, empty on entry, built here.</param>
    /// <param name="status">The status holding GJK's segment.</param>
    /// <param name="obj1">Geom 1.</param>
    /// <param name="obj2">Geom 2.</param>
    /// <returns>0 to run EPA, or the EPA status that stops it.</returns>
    private static int Polytope2(CcdPolytope pt, CcdStatus status, CcdObject obj1, CcdObject obj2) {
        double[] sx = status.Simplex;
        for (int k = 0; k < 3; k++)
            pt.Center[k] = sx[k] + sx[VertexSize + k];
        for (int k = 0; k < 3; k++)
            pt.Center[k] = 0.5 * pt.Center[k];
        Span<double> diff = stackalloc double[3];
        for (int k = 0; k < 3; k++)
            diff[k] = sx[VertexSize + k] - sx[k];
        // the coordinate axis most perpendicular to the segment (the strictly smallest component, first on ties)
        double value = MaxLimit;
        int index = 0;
        for (int i = 0; i < 3; i++) {
            if (Math.Abs(diff[i]) < value) {
                value = Math.Abs(diff[i]);
                index = i;
            }
        }
        Span<double> e = stackalloc double[3], d1 = stackalloc double[3], d2 = stackalloc double[3], d3 = stackalloc double[3];
        e[index] = 1;
        Spatial.Cross(d1, e, diff);
        Span<double> r = stackalloc double[9];
        RotMat(r, diff);
        Spatial.MulMatVec3(d2, r, d1);
        Spatial.MulMatVec3(d3, r, d2);
        int v1i = pt.InsertVertex(sx, 0), v2i = pt.InsertVertex(sx, VertexSize);
        int v3i = EpaSupport(pt, obj1, obj2, d1, Math.Sqrt(Dot3(d1, d1)));
        int v4i = EpaSupport(pt, obj1, obj2, d2, Math.Sqrt(Dot3(d2, d2)));
        int v5i = EpaSupport(pt, obj1, obj2, d3, Math.Sqrt(Dot3(d3, d3)));
        ReadOnlySpan<int> faces = [v1i, v3i, v4i, 1, 3, 2, v1i, v5i, v3i, 2, 4, 0, v1i, v4i, v5i, 0, 5, 1,
                                   v2i, v4i, v3i, 5, 0, 4, v2i, v3i, v5i, 3, 1, 5, v2i, v5i, v4i, 4, 2, 3];
        for (int f = 0; f < 6; f++) {
            if (AttachFace(pt, faces[6 * f], faces[6 * f + 1], faces[6 * f + 2], faces[6 * f + 3], faces[6 * f + 4], faces[6 * f + 5]) < MinVal2) {
                ReplaceSimplex3(pt, status, faces[6 * f], faces[6 * f + 1], faces[6 * f + 2]);
                return Polytope3(pt, status, obj1, obj2);
            }
        }
        // the hexahedron must be convex: the segment's line crosses the triangle of the three new points
        double[] v = pt.Verts;
        if (RayTriangle(sx.AsSpan(0, 3), sx.AsSpan(VertexSize, 3), v.AsSpan(v3i * VertexSize, 3), v.AsSpan(v4i * VertexSize, 3),
                        v.AsSpan(v5i * VertexSize, 3)) == 0)
            return P2Nonconvex;
        for (int i = 0; i < 6; i++) {
            pt.Map[i] = i;
            pt.Faces[i].Index = i;
        }
        pt.Nmap = 6;
        return 0;
    }

    /// <summary>
    ///     A 120-degree rotation about an axis (<c>rotmat</c>), from MuJoCo's rounded constants <c>sin = 0.86602540378</c>
    ///     and <c>cos = -0.5</c>.
    /// </summary>
    /// <param name="r">Receives the row-major matrix.</param>
    /// <param name="axis">The axis (any length).</param>
    private static void RotMat(Span<double> r, ReadOnlySpan<double> axis) {
        double n = Math.Sqrt(Dot3(axis, axis));
        double u1 = axis[0] / n, u2 = axis[1] / n, u3 = axis[2] / n;
        const double sin = 0.86602540378, cos = -0.5;
        r[0] = cos + u1 * u1 * (1 - cos);
        r[1] = u1 * u2 * (1 - cos) - u3 * sin;
        r[2] = u1 * u3 * (1 - cos) + u2 * sin;
        r[3] = u2 * u1 * (1 - cos) + u3 * sin;
        r[4] = cos + u2 * u2 * (1 - cos);
        r[5] = u2 * u3 * (1 - cos) - u1 * sin;
        r[6] = u1 * u3 * (1 - cos) - u2 * sin;
        r[7] = u2 * u3 * (1 - cos) + u1 * sin;
        r[8] = cos + u3 * u3 * (1 - cos);
    }

    /// <summary>
    ///     Whether the ray from <c>v1</c> through <c>v2</c> crosses the triangle <c>v3v4v5</c> (<c>rayTriangle</c>), by the
    ///     signs of three volumes.
    /// </summary>
    /// <param name="v1">The ray's origin.</param>
    /// <param name="v2">A second point on the ray.</param>
    /// <param name="v3">The triangle's first vertex.</param>
    /// <param name="v4">Its second vertex.</param>
    /// <param name="v5">Its third vertex.</param>
    /// <returns>1 or -1 when it does (all volumes of one sign, zero allowed), 0 when not.</returns>
    private static int RayTriangle(ReadOnlySpan<double> v1, ReadOnlySpan<double> v2, ReadOnlySpan<double> v3, ReadOnlySpan<double> v4,
                                   ReadOnlySpan<double> v5) {
        Span<double> d12 = stackalloc double[3], d13 = stackalloc double[3], d14 = stackalloc double[3], d15 = stackalloc double[3];
        for (int k = 0; k < 3; k++) {
            d12[k] = v2[k] - v1[k];
            d13[k] = v3[k] - v1[k];
            d14[k] = v4[k] - v1[k];
            d15[k] = v5[k] - v1[k];
        }
        double vol1 = Det3(d13, d14, d12), vol2 = Det3(d14, d15, d12), vol3 = Det3(d15, d13, d12);
        if (vol1 >= 0 && vol2 >= 0 && vol3 >= 0)
            return 1;
        if (vol1 <= 0 && vol2 <= 0 && vol3 <= 0)
            return -1;
        return 0;
    }

    /// <summary>
    ///     Replaces GJK's simplex with a face of the polytope and empties the polytope (<c>replaceSimplex3</c>), before
    ///     <see cref="Polytope3"/> starts over from that triangle.
    /// </summary>
    /// <param name="pt">The polytope, emptied.</param>
    /// <param name="status">The status whose simplex becomes the face.</param>
    /// <param name="v1">The face's first vertex index in the polytope.</param>
    /// <param name="v2">Its second vertex index.</param>
    /// <param name="v3">Its third vertex index.</param>
    private static void ReplaceSimplex3(CcdPolytope pt, CcdStatus status, int v1, int v2, int v3) {
        status.Nsimplex = 3;
        Array.Copy(pt.Verts, v1 * VertexSize, status.Simplex, 0, VertexSize);
        Array.Copy(pt.Verts, v2 * VertexSize, status.Simplex, VertexSize, VertexSize);
        Array.Copy(pt.Verts, v3 * VertexSize, status.Simplex, 2 * VertexSize, VertexSize);
        pt.Nfaces = 0;
        pt.Nverts = 0;
        pt.Nmap = 0;
    }

    /// <summary>
    ///     Whether a point lies in a triangle (<c>triPointIntersect</c>): non-negative affine coordinates, and the point
    ///     they rebuild within <c>mjMINVAL</c> of it.
    /// </summary>
    /// <param name="v1">The triangle's first vertex.</param>
    /// <param name="v2">Its second vertex.</param>
    /// <param name="v3">Its third vertex.</param>
    /// <param name="p">The point.</param>
    /// <returns><see langword="true"/> when the point is in the triangle, edges included.</returns>
    private static bool TriPointIntersect(ReadOnlySpan<double> v1, ReadOnlySpan<double> v2, ReadOnlySpan<double> v3, ReadOnlySpan<double> p) {
        Span<double> lambda = stackalloc double[3], diff = stackalloc double[3];
        TriAffineCoord(lambda, v1, v2, v3, p);
        if (lambda[0] < 0 || lambda[1] < 0 || lambda[2] < 0)
            return false;
        for (int k = 0; k < 3; k++)
            diff[k] = v1[k] * lambda[0] + v2[k] * lambda[1] + v3[k] * lambda[2] - p[k];
        return Math.Sqrt(Dot3(diff, diff)) < Kernels.MinVal;
    }

    /// <summary>
    ///     The initial polytope from GJK's triangle (<c>polytope3</c>): the support points against and along its normal
    ///     (in that order) close a hexahedron of six faces around it; a degenerate normal, a support point inside the
    ///     triangle, a hexahedron missing the origin or a face through it stops EPA.
    /// </summary>
    /// <param name="pt">The polytope, empty on entry, built here.</param>
    /// <param name="status">The status holding GJK's triangle.</param>
    /// <param name="obj1">Geom 1.</param>
    /// <param name="obj2">Geom 2.</param>
    /// <returns>0 to run EPA, or the EPA status that stops it.</returns>
    private static int Polytope3(CcdPolytope pt, CcdStatus status, CcdObject obj1, CcdObject obj2) {
        double[] sx = status.Simplex;
        for (int k = 0; k < 3; k++)
            pt.Center[k] = sx[k] + sx[VertexSize + k];
        for (int k = 0; k < 3; k++)
            pt.Center[k] = pt.Center[k] + sx[2 * VertexSize + k];
        for (int k = 0; k < 3; k++)
            pt.Center[k] = 1.0 / 3.0 * pt.Center[k];
        Span<double> diff1 = stackalloc double[3], diff2 = stackalloc double[3], n = stackalloc double[3], nNeg = stackalloc double[3];
        for (int k = 0; k < 3; k++) {
            diff1[k] = sx[VertexSize + k] - sx[k];
            diff2[k] = sx[2 * VertexSize + k] - sx[k];
        }
        Spatial.Cross(n, diff1, diff2);
        double nNorm = Math.Sqrt(Dot3(n, n));
        if (nNorm < Kernels.MinVal)
            return P3BadNormal;
        for (int k = 0; k < 3; k++)
            nNeg[k] = -1 * n[k];
        // the triangle first (the polytope is empty here, so its vertices are 0, 1 and 2), then against, then along n
        int v1i = pt.InsertVertex(sx, 0), v2i = pt.InsertVertex(sx, VertexSize), v3i = pt.InsertVertex(sx, 2 * VertexSize);
        int v5i = EpaSupport(pt, obj1, obj2, nNeg, nNorm);
        int v4i = EpaSupport(pt, obj1, obj2, n, nNorm);
        double[] v = pt.Verts;
        ReadOnlySpan<double> s1 = sx.AsSpan(0, 3), s2 = sx.AsSpan(VertexSize, 3), s3 = sx.AsSpan(2 * VertexSize, 3);
        if (TriPointIntersect(s1, s2, s3, v.AsSpan(v4i * VertexSize, 3)))
            return P3InvalidV4;
        if (TriPointIntersect(s1, s2, s3, v.AsSpan(v5i * VertexSize, 3)))
            return P3InvalidV5;
        // EPA's caller set the distance to 0 just before, so MuJoCo's check cannot fire; kept as written
        if (status.Dist[0] > 10 * Kernels.MinVal
            && !TestTetra(v, v1i * VertexSize, v2i * VertexSize, v3i * VertexSize, v4i * VertexSize)
            && !TestTetra(v, v1i * VertexSize, v2i * VertexSize, v3i * VertexSize, v5i * VertexSize))
            return P3MissingOrigin;
        ReadOnlySpan<int> faces = [v4i, v1i, v2i, 1, 3, 2, v4i, v3i, v1i, 2, 4, 0, v4i, v2i, v3i, 0, 5, 1,
                                   v5i, v2i, v1i, 5, 0, 4, v5i, v1i, v3i, 3, 1, 5, v5i, v3i, v2i, 4, 2, 3];
        for (int f = 0; f < 6; f++)
            if (AttachFace(pt, faces[6 * f], faces[6 * f + 1], faces[6 * f + 2], faces[6 * f + 3], faces[6 * f + 4], faces[6 * f + 5]) < MinVal2)
                return P3OriginOnFace;
        for (int i = 0; i < 6; i++) {
            pt.Map[i] = i;
            pt.Faces[i].Index = i;
        }
        pt.Nmap = 6;
        return 0;
    }

    /// <summary>
    ///     The initial polytope from GJK's tetrahedron (<c>polytope4</c>): its four faces, checked for the origin on a face
    ///     (then <see cref="Polytope3"/> starts over from that face) and numerically inside.
    /// </summary>
    /// <param name="pt">The polytope, empty on entry, built here.</param>
    /// <param name="status">The status holding GJK's tetrahedron.</param>
    /// <param name="obj1">Geom 1.</param>
    /// <param name="obj2">Geom 2.</param>
    /// <returns>0 to run EPA, or the EPA status that stops it.</returns>
    private static int Polytope4(CcdPolytope pt, CcdStatus status, CcdObject obj1, CcdObject obj2) {
        int v1 = pt.InsertVertex(status.Simplex, 0), v2 = pt.InsertVertex(status.Simplex, VertexSize);
        int v3 = pt.InsertVertex(status.Simplex, 2 * VertexSize), v4 = pt.InsertVertex(status.Simplex, 3 * VertexSize);
        double[] verts = pt.Verts;
        for (int k = 0; k < 3; k++) {
            pt.Center[k] = verts[v1 * VertexSize + k] + verts[v2 * VertexSize + k];
            pt.Center[k] = pt.Center[k] + verts[v3 * VertexSize + k];
            pt.Center[k] = pt.Center[k] + verts[v4 * VertexSize + k];
        }
        for (int k = 0; k < 3; k++)
            pt.Center[k] = 0.25 * pt.Center[k];
        ReadOnlySpan<int> faces = [v1, v2, v3, 1, 3, 2, v1, v4, v2, 2, 3, 0, v1, v3, v4, 0, 3, 1, v4, v3, v2, 2, 0, 1];
        for (int f = 0; f < 4; f++) {
            if (AttachFace(pt, faces[6 * f], faces[6 * f + 1], faces[6 * f + 2], faces[6 * f + 3], faces[6 * f + 4], faces[6 * f + 5]) < MinVal2) {
                ReplaceSimplex3(pt, status, faces[6 * f], faces[6 * f + 1], faces[6 * f + 2]);
                return Polytope3(pt, status, obj1, obj2);
            }
        }
        if (!TestTetra(verts, v1 * VertexSize, v2 * VertexSize, v3 * VertexSize, v4 * VertexSize))
            return P4MissingOrigin;
        for (int i = 0; i < 4; i++) {
            pt.Map[i] = i;
            pt.Faces[i].Index = i;
        }
        pt.Nmap = 4;
        return 0;
    }

    /// <summary>Whether the origin and <c>p3</c> lie on the same side of the plane through the first three (<c>sameSide</c>).</summary>
    /// <param name="v">The vertex array.</param>
    /// <param name="p0">The offset of the plane's first point.</param>
    /// <param name="p1">Of its second point.</param>
    /// <param name="p2">Of its third point.</param>
    /// <param name="p3">The offset of the point compared with the origin.</param>
    /// <returns>
    ///     <see langword="true"/> when both lie strictly on one side; a point on the plane gives <see langword="false"/>.
    /// </returns>
    private static bool SameSide(double[] v, int p0, int p1, int p2, int p3) {
        Span<double> d1 = stackalloc double[3], d2 = stackalloc double[3], d3 = stackalloc double[3], d4 = stackalloc double[3], n = stackalloc double[3];
        for (int k = 0; k < 3; k++) {
            d1[k] = v[p1 + k] - v[p0 + k];
            d2[k] = v[p2 + k] - v[p0 + k];
        }
        Spatial.Cross(n, d1, d2);
        for (int k = 0; k < 3; k++)
            d3[k] = v[p3 + k] - v[p0 + k];
        double dot1 = Dot3(n, d3);
        for (int k = 0; k < 3; k++)
            d4[k] = -1 * v[p0 + k];
        double dot2 = Dot3(n, d4);
        return (dot1 > 0 && dot2 > 0) || (dot1 < 0 && dot2 < 0);
    }

    /// <summary>Whether the origin lies inside the tetrahedron (<c>testTetra</c>).</summary>
    /// <param name="v">The vertex array.</param>
    /// <param name="p0">The offset of the first vertex.</param>
    /// <param name="p1">Of the second.</param>
    /// <param name="p2">Of the third.</param>
    /// <param name="p3">Of the fourth.</param>
    /// <returns><see langword="true"/> when the origin is strictly inside; on a face counts as outside.</returns>
    private static bool TestTetra(double[] v, int p0, int p1, int p2, int p3) =>
        SameSide(v, p0, p1, p2, p3) && SameSide(v, p1, p2, p3, p0) && SameSide(v, p2, p3, p0, p1) && SameSide(v, p3, p0, p1, p2);

    /// <summary>
    ///     Appends a face (<c>attachFace</c>): its vertices packed 10 bits each, its neighbours, the origin's projection on
    ///     its plane turned outward from the polytope's centre, and the projection's squared norm.
    /// </summary>
    /// <param name="pt">The polytope, which gains the face.</param>
    /// <param name="v1">The face's first vertex index.</param>
    /// <param name="v2">Its second vertex index.</param>
    /// <param name="v3">Its third vertex index.</param>
    /// <param name="adj1">The neighbouring face across edge 0, from the first vertex to the second.</param>
    /// <param name="adj2">Across edge 1, from the second vertex to the third.</param>
    /// <param name="adj3">Across edge 2, from the third vertex back to the first.</param>
    /// <returns>The squared distance of the face's plane to the origin; 0 for a degenerate face.</returns>
    private static double AttachFace(CcdPolytope pt, int v1, int v2, int v3, int adj1, int adj2, int adj3) {
        ref CcdFace face = ref pt.Faces[pt.Nfaces++];
        face.Verts = v1 + (v2 << 10) + (v3 << 20);
        face.Adj0 = adj1;
        face.Adj1 = adj2;
        face.Adj2 = adj3;
        double[] verts = pt.Verts;
        Span<double> projection = stackalloc double[3];
        if (ProjectOriginPlane(projection, verts.AsSpan(v3 * VertexSize, 3), verts.AsSpan(v2 * VertexSize, 3), verts.AsSpan(v1 * VertexSize, 3)) != 0)
            return 0;
        Span<double> outward = stackalloc double[3];
        for (int k = 0; k < 3; k++)
            outward[k] = verts[v1 * VertexSize + k] - pt.Center[k];
        if (Dot3(projection, outward) < 0)
            for (int k = 0; k < 3; k++)
                projection[k] = -1 * projection[k];
        face.V0 = projection[0];
        face.V1 = projection[1];
        face.V2 = projection[2];
        face.Dist2 = Dot3(projection, projection);
        face.Index = -1;
        return face.Dist2;
    }

    /// <summary>
    ///     The expanding polytope (<c>epa</c>): the face nearest the origin is pushed out by a support point along its
    ///     normal, the faces it can see are replaced by a fan around the new vertex, until the upper and lower depth
    ///     bounds meet within the tolerance.
    /// </summary>
    /// <param name="status">
    ///     The status: the tolerance and iteration bound in, the distance, witnesses and iterations out.
    /// </param>
    /// <param name="pt">The initial polytope, expanded in place.</param>
    /// <param name="obj1">Geom 1.</param>
    /// <param name="obj2">Geom 2.</param>
    /// <returns>The face whose plane gives the penetration (witnesses written to the status), or -1.</returns>
    private static int Epa(CcdStatus status, CcdPolytope pt, CcdObject obj1, CcdObject obj2) {
        double upper = MaxLimit, upper2 = MaxLimit, lower2;
        int face = -1, pface;
        double tolerance = status.Tolerance;
        int k, kmax = status.MaxIterations < 1000 ? status.MaxIterations : 1000;
        Span<double> fv = stackalloc double[3];
        for (k = 0; k < kmax; k++) {
            pface = face;
            lower2 = MaxLimit;
            for (int i = 0; i < pt.Nmap; i++) {
                if (pt.Faces[pt.Map[i]].Dist2 < lower2) {
                    face = pt.Map[i];
                    lower2 = pt.Faces[face].Dist2;
                }
            }
            if (lower2 > upper2 || face < 0) {
                face = pface;
                break;
            }
            if (lower2 <= 0)
                break;
            double lower = Math.Sqrt(lower2);
            fv[0] = pt.Faces[face].V0;
            fv[1] = pt.Faces[face].V1;
            fv[2] = pt.Faces[face].V2;
            int wi = EpaSupport(pt, obj1, obj2, fv, lower);
            int w = wi * VertexSize;
            double upperK = (fv[0] * pt.Verts[w] + fv[1] * pt.Verts[w + 1] + fv[2] * pt.Verts[w + 2]) / lower;
            if (upperK < upper) {
                upper = upperK;
                upper2 = upper * upper;
            }
            if (upper - lower < tolerance) {
                if (k == 0 && upper < lower - 1e-10)
                    face = -1;
                break;
            }
            pt.HorizonW = w;
            Horizon(pt, face);
            if (pt.Nedges < 3) {
                face = -1;
                break;
            }
            int nfaces = pt.Nfaces, nedges = pt.Nedges;
            if (nedges > pt.MaxFaces - pt.Nfaces)
                break;
            int hznIndex = pt.HorizonIndices[0], hznEdge = pt.HorizonEdges[0];
            int v1 = Vert(pt.Faces[hznIndex].Verts, hznEdge), v2 = Vert(pt.Faces[hznIndex].Verts, (hznEdge + 1) % 3);
            SetAdj(ref pt.Faces[hznIndex], hznEdge, nfaces);
            double dist2 = AttachFace(pt, wi, v2, v1, nfaces + nedges - 1, hznIndex, nfaces + 1);
            if (dist2 == 0) {
                face = -1;
                break;
            }
            if (dist2 >= lower2 && dist2 <= upper2) {
                int i = pt.Nmap++;
                pt.Map[i] = pt.Nfaces - 1;
                pt.Faces[pt.Nfaces - 1].Index = i;
            }
            for (int i = 1; i < nedges; i++) {
                int cur = nfaces + i, next = nfaces + (i + 1) % nedges;
                hznIndex = pt.HorizonIndices[i];
                hznEdge = pt.HorizonEdges[i];
                v1 = Vert(pt.Faces[hznIndex].Verts, hznEdge);
                v2 = Vert(pt.Faces[hznIndex].Verts, (hznEdge + 1) % 3);
                SetAdj(ref pt.Faces[hznIndex], hznEdge, cur);
                dist2 = AttachFace(pt, wi, v2, v1, cur - 1, hznIndex, next);
                if (dist2 == 0) {
                    face = -1;
                    break;
                }
                if (dist2 >= lower2 && dist2 <= upper2) {
                    int idx = pt.Nmap++;
                    pt.Map[idx] = pt.Nfaces - 1;
                    pt.Faces[pt.Nfaces - 1].Index = idx;
                }
            }
            pt.Nedges = 0;
            if (pt.Nmap == 0 || face < 0)
                break;
        }
        status.EpaIterations = k;
        if (face >= 0) {
            status.Dist[0] = EpaWitness(pt, face, status.X1, status.X2);
            status.Nx = 1;
        } else {
            status.Nx = 0;
            status.Dist[0] = 0;
        }
        return face;
    }

    /// <summary>A face's vertex at slot 0, 1 or 2 of its packed indices (<c>EPA_VERT_EXPAND</c>).</summary>
    /// <param name="verts">The face's packed vertex indices.</param>
    /// <param name="slot">The slot, 0 to 2.</param>
    /// <returns>The vertex index.</returns>
    private static int Vert(int verts, int slot) => slot switch {
        0 => verts & 0x3FF,
        1 => (verts >> 10) & 0x3FF,
        _ => (verts >> 20) & 0x3FF,
    };

    /// <summary>A face's neighbour across edge 0, 1 or 2.</summary>
    /// <param name="f">The face.</param>
    /// <param name="edge">The edge, 0 to 2.</param>
    /// <returns>The neighbouring face's index.</returns>
    private static int Adj(in CcdFace f, int edge) => edge == 0 ? f.Adj0 : edge == 1 ? f.Adj1 : f.Adj2;

    /// <summary>Sets a face's neighbour across an edge.</summary>
    /// <param name="f">The face, changed in place.</param>
    /// <param name="edge">The edge, 0 to 2.</param>
    /// <param name="value">The neighbouring face's index.</param>
    private static void SetAdj(ref CcdFace f, int edge, int value) {
        if (edge == 0)
            f.Adj0 = value;
        else if (edge == 1)
            f.Adj1 = value;
        else
            f.Adj2 = value;
    }

    /// <summary>Removes a face from the candidate map and marks it deleted (<c>deleteFace</c>).</summary>
    /// <param name="pt">The polytope.</param>
    /// <param name="face">The face's index.</param>
    private static void DeleteFace(CcdPolytope pt, int face) {
        int index = pt.Faces[face].Index;
        if (index >= 0) {
            pt.Map[index] = pt.Map[--pt.Nmap];
            pt.Faces[pt.Map[index]].Index = index;
        }
        pt.Faces[face].Index = -2;
    }

    /// <summary>The edge of a face that starts at a vertex (<c>getEdge</c>).</summary>
    /// <param name="face">The face.</param>
    /// <param name="vertex">The vertex index the edge starts at; a vertex not on the face gives edge 2.</param>
    /// <returns>The edge, 0 to 2.</returns>
    private static int GetEdge(in CcdFace face, int vertex) =>
        Vert(face.Verts, 0) == vertex ? 0 : Vert(face.Verts, 1) == vertex ? 1 : 2;

    /// <summary>Appends an edge to the horizon (<c>addEdge</c>).</summary>
    /// <param name="pt">The polytope holding the horizon.</param>
    /// <param name="index">The face that stays: invisible from the new vertex, it borders the deleted region.</param>
    /// <param name="edge">Its edge on the horizon.</param>
    private static void AddEdge(CcdPolytope pt, int index, int edge) {
        pt.HorizonEdges[pt.Nedges] = edge;
        pt.HorizonIndices[pt.Nedges++] = index;
    }

    /// <summary>
    ///     The horizon's recursive step (<c>horizonRec</c>): a face visible from the new vertex is deleted and its other two
    ///     neighbours searched; an invisible one ends the search on the horizon.
    /// </summary>
    /// <param name="pt">The polytope; its new vertex is <c>HorizonW</c>.</param>
    /// <param name="face">The face to test.</param>
    /// <param name="e">The edge of <paramref name="face"/> the search came across.</param>
    /// <returns>Whether the face was visible.</returns>
    private static bool HorizonRec(CcdPolytope pt, int face, int e) {
        ref CcdFace f = ref pt.Faces[face];
        int w = pt.HorizonW;
        if (f.V0 * pt.Verts[w] + f.V1 * pt.Verts[w + 1] + f.V2 * pt.Verts[w + 2] - f.Dist2 > Kernels.MinVal) {
            int verts = f.Verts;
            DeleteFace(pt, face);
            for (int k = 1; k < 3; k++) {
                int i = (e + k) % 3;
                int adj = Adj(pt.Faces[face], i);
                if (pt.Faces[adj].Index > -2) {
                    int adjEdge = GetEdge(pt.Faces[adj], Vert(verts, (i + 1) % 3));
                    if (!HorizonRec(pt, adj, adjEdge))
                        AddEdge(pt, adj, adjEdge);
                }
            }
            return true;
        }
        return false;
    }

    /// <summary>
    ///     The horizon seen from the new vertex, starting at the face it was found from (<c>horizon</c>): that face is
    ///     deleted, and its three neighbours are searched in edge order.
    /// </summary>
    /// <param name="pt">The polytope; its new vertex is <c>HorizonW</c>, its horizon is rebuilt.</param>
    /// <param name="face">The face the new vertex was found from.</param>
    private static void Horizon(CcdPolytope pt, int face) {
        DeleteFace(pt, face);
        int verts = pt.Faces[face].Verts;
        int adj = pt.Faces[face].Adj0;
        int adjEdge = GetEdge(pt.Faces[adj], Vert(verts, 1));
        if (!HorizonRec(pt, adj, adjEdge))
            AddEdge(pt, adj, adjEdge);
        adj = pt.Faces[face].Adj1;
        adjEdge = GetEdge(pt.Faces[adj], Vert(verts, 2));
        if (pt.Faces[adj].Index > -2 && !HorizonRec(pt, adj, adjEdge))
            AddEdge(pt, adj, adjEdge);
        adj = pt.Faces[face].Adj2;
        adjEdge = GetEdge(pt.Faces[adj], Vert(verts, 0));
        if (pt.Faces[adj].Index > -2 && !HorizonRec(pt, adj, adjEdge))
            AddEdge(pt, adj, adjEdge);
    }

    /// <summary>
    ///     The witness points of the final face (<c>epaWitness</c>): the origin's projection in the face's affine
    ///     coordinates, applied to the geoms' points.
    /// </summary>
    /// <param name="pt">The polytope.</param>
    /// <param name="face">The final face.</param>
    /// <param name="x1">Receives geom 1's witness point.</param>
    /// <param name="x2">Receives geom 2's witness point.</param>
    /// <returns>Minus the face's distance to the origin (the penetration).</returns>
    private static double EpaWitness(CcdPolytope pt, int face, double[] x1, double[] x2) {
        ref CcdFace f = ref pt.Faces[face];
        int a = Vert(f.Verts, 0) * VertexSize, b = Vert(f.Verts, 1) * VertexSize, c = Vert(f.Verts, 2) * VertexSize;
        Span<double> lambda = stackalloc double[3], p = stackalloc double[3] { f.V0, f.V1, f.V2 };
        TriAffineCoord(lambda, pt.Verts.AsSpan(a, 3), pt.Verts.AsSpan(b, 3), pt.Verts.AsSpan(c, 3), p);
        Lincomb(x1, lambda, 3, pt.Verts, a + 3, b + 3, c + 3, 0);
        Lincomb(x2, lambda, 3, pt.Verts, a + 6, b + 6, c + 6, 0);
        return -Math.Sqrt(f.Dist2);
    }

    /// <summary>
    ///     The affine coordinates of a point in a triangle's plane (<c>triAffineCoord</c>): signed areas on the coordinate
    ///     plane that keeps the most of the triangle, over the triangle's.
    /// </summary>
    /// <param name="lambda">Receives the three coordinates.</param>
    /// <param name="v1">The triangle's first vertex.</param>
    /// <param name="v2">Its second vertex.</param>
    /// <param name="v3">Its third vertex.</param>
    /// <param name="p">The point, in the triangle's plane.</param>
    private static void TriAffineCoord(Span<double> lambda, ReadOnlySpan<double> v1, ReadOnlySpan<double> v2, ReadOnlySpan<double> v3, ReadOnlySpan<double> p) {
        double m14 = v2[1] * v3[2] - v2[2] * v3[1] - v1[1] * v3[2] + v1[2] * v3[1] + v1[1] * v2[2] - v1[2] * v2[1];
        double m24 = v2[0] * v3[2] - v2[2] * v3[0] - v1[0] * v3[2] + v1[2] * v3[0] + v1[0] * v2[2] - v1[2] * v2[0];
        double m34 = v2[0] * v3[1] - v2[1] * v3[0] - v1[0] * v3[1] + v1[1] * v3[0] + v1[0] * v2[1] - v1[1] * v2[0];
        double mu1 = Math.Abs(m14), mu2 = Math.Abs(m24), mu3 = Math.Abs(m34);
        int x, y;
        double mMax;
        if (mu1 >= mu2 && mu1 >= mu3) {
            mMax = m14;
            x = 1;
            y = 2;
        } else if (mu2 >= mu3) {
            mMax = m24;
            x = 0;
            y = 2;
        } else {
            mMax = m34;
            x = 0;
            y = 1;
        }
        double c31 = p[x] * v2[y] + p[y] * v3[x] + v2[x] * v3[y] - p[x] * v3[y] - p[y] * v2[x] - v3[x] * v2[y];
        double c32 = p[x] * v3[y] + p[y] * v1[x] + v3[x] * v1[y] - p[x] * v1[y] - p[y] * v3[x] - v1[x] * v3[y];
        double c33 = p[x] * v1[y] + p[y] * v2[x] + v1[x] * v2[y] - p[x] * v2[y] - p[y] * v1[x] - v2[x] * v1[y];
        lambda[0] = c31 / mMax;
        lambda[1] = c32 / mMax;
        lambda[2] = c33 / mMax;
    }
}

/// <summary>An EPA face (<c>Face</c>): packed vertices, neighbours per edge, the origin's projection, its map slot.</summary>
internal struct CcdFace {
    /// <summary>The three vertex indices, 10 bits each.</summary>
    public int Verts;
    /// <summary>The neighbour across edge (v1, v2).</summary>
    public int Adj0;
    /// <summary>The neighbour across edge (v2, v3).</summary>
    public int Adj1;
    /// <summary>The neighbour across edge (v3, v1).</summary>
    public int Adj2;
    /// <summary>The origin's projection on the face's plane, x.</summary>
    public double V0;
    /// <summary>The projection, y.</summary>
    public double V1;
    /// <summary>The projection, z.</summary>
    public double V2;
    /// <summary>The projection's squared norm.</summary>
    public double Dist2;
    /// <summary>The face's slot in the candidate map; -1 not in it, -2 deleted.</summary>
    public int Index;
}

/// <summary>
///     The expanding polytope (<c>Polytope</c>): vertices as rows of 9 values, faces, the map of candidate faces and the
///     horizon of the current expansion, sized as MuJoCo sizes its buffer for an iteration budget.
/// </summary>
internal sealed class CcdPolytope {
    /// <summary>Vertex rows (difference point, geom 1 point, geom 2 point).</summary>
    public double[] Verts = [];
    /// <summary>The vertex count.</summary>
    public int Nverts;
    /// <summary>The faces.</summary>
    public CcdFace[] Faces = [];
    /// <summary>The face count.</summary>
    public int Nfaces;
    /// <summary>The face capacity (<c>6*iterations</c>).</summary>
    public int MaxFaces;
    /// <summary>The initial tetrahedron's centre, against which faces are turned outward.</summary>
    public readonly double[] Center = new double[3];
    /// <summary>The candidate faces (indices into <see cref="Faces"/>).</summary>
    public int[] Map = [];
    /// <summary>The candidate count.</summary>
    public int Nmap;
    /// <summary>The horizon's faces.</summary>
    public readonly int[] HorizonIndices = new int[24];
    /// <summary>The horizon's edges, one per face.</summary>
    public readonly int[] HorizonEdges = new int[24];
    /// <summary>The horizon's edge count.</summary>
    public int Nedges;
    /// <summary>The offset of the vertex the horizon is seen from.</summary>
    public int HorizonW;

    /// <summary>Empties the polytope and sizes it for an iteration budget (vertices <c>5+N</c>, faces <c>6N</c>).</summary>
    /// <param name="iterations">The budget.</param>
    public void Reset(int iterations) {
        if (Faces.Length < 6 * iterations) {
            Verts = new double[(5 + iterations) * Ccd.VertexSize];
            Faces = new CcdFace[6 * iterations];
            Map = new int[6 * iterations];
        }
        MaxFaces = 6 * iterations;
        Nverts = Nfaces = Nmap = Nedges = 0;
    }

    /// <summary>Copies a vertex in (<c>insertVertex</c>).</summary>
    /// <param name="src">The array holding the vertex.</param>
    /// <param name="offset">Its offset.</param>
    /// <returns>Its index in the polytope.</returns>
    public int InsertVertex(double[] src, int offset) {
        int n = Nverts++;
        Array.Copy(src, offset, Verts, n * Ccd.VertexSize, Ccd.VertexSize);
        return n;
    }
}
