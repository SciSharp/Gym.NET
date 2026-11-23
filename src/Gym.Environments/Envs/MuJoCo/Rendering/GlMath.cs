using System;

namespace Gym.Environments.Envs.MuJoCo.Rendering {
    /// <summary>
    ///     Column-major 4×4 float matrices as OpenGL composes them (<c>glTranslatef</c>, <c>glScalef</c>,
    ///     <c>glFrustum</c>, <c>glOrtho</c>, <c>glMultMatrixf</c>), and MuJoCo's render helpers built on them
    ///     (<c>mjr_lookAt</c>, <c>mjr_reflect</c>, <c>mjr_orthoVec</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Float32 throughout, as the driver and MuJoCo's <c>float</c> helpers compute, with each one's own order of
    ///     operations, measured against the reference driver (NVIDIA 616.56) by reading its matrix stack back with
    ///     <c>glGetFloatv</c> after random calls (2026-11-21, DR-98): <c>glMultMatrixf</c> and <c>glTranslatef</c> sum each
    ///     element's four products pairwise, <c>(a₀b₀ + a₁b₁) + (a₂b₂ + a₃b₃)</c> (<see cref="MultiplyDriver"/>, every element
    ///     of 400 trials; the sequential sum got 72%, any fused order about half); <c>glScalef</c> is plain products;
    ///     <c>glFrustum</c> divides a numerator kept in double by a denominator rounded to float32 (<see cref="Frustum"/>,
    ///     every element of 250 matrices); <c>glOrtho</c> computes each element in double and rounds it once
    ///     (<see cref="Ortho"/>, every element of 300). MuJoCo's own <c>mjr_mulMat44</c> sums sequentially from zero
    ///     (<see cref="Multiply"/>), so the shadow map's texture matrix, which MuJoCo multiplies itself, keeps that order.
    ///     </para>
    ///     <para>
    ///     Footgun: the two products are not interchangeable; each must be the one the code it stands for uses, or a
    ///     matrix differs in its last bits and a vertex can land on the other side of a pixel's sample.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/google-deepmind/mujoco/blob/4.0.0/src/render/classic/render_util.c">render_util.c</see>.</para>
    /// </remarks>
    internal static class GlMath {
        /// <summary>The identity.</summary>
        /// <returns>A new identity matrix.</returns>
        internal static float[] Identity() => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

        /// <summary>
        ///     The product <c>a·b</c> as the reference driver's matrix stack computes it (<c>glMultMatrixf</c>,
        ///     <c>glTranslatef</c>, the frustum's and ortho's products): each element's four products summed pairwise,
        ///     <c>(a₀b₀ + a₁b₁) + (a₂b₂ + a₃b₃)</c>, every operation rounded to float32.
        /// </summary>
        /// <param name="a">The current matrix.</param>
        /// <param name="b">The matrix it is multiplied by.</param>
        /// <returns>A new matrix.</returns>
        /// <remarks>
        ///     The order was measured on the reference driver (see the class remarks); a translation's column then comes
        ///     out as <c>(m₀x + m₄y) + (m₈z + m₁₂)</c>, as its own <c>glTranslatef</c> gives it.
        /// </remarks>
        internal static float[] MultiplyDriver(float[] a, float[] b) {
            var res = new float[16];
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    float p0 = a[r] * b[4 * c], p1 = a[r + 4] * b[1 + 4 * c], p2 = a[r + 8] * b[2 + 4 * c], p3 = a[r + 12] * b[3 + 4 * c];
                    res[r + 4 * c] = (p0 + p1) + (p2 + p3);
                }
            }

            return res;
        }

        /// <summary>
        ///     The product <c>a·b</c> in <c>mjr_mulMat44</c>'s order: each element summed sequentially from zero, every
        ///     operation rounded to float32. For the matrices MuJoCo multiplies itself (the shadow map's texture matrix);
        ///     the driver's own stack is <see cref="MultiplyDriver"/>.
        /// </summary>
        /// <param name="a">The left matrix.</param>
        /// <param name="b">The right matrix.</param>
        /// <returns>A new matrix.</returns>
        internal static float[] Multiply(float[] a, float[] b) {
            var res = new float[16];
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) {
                    float s = 0;
                    for (int i = 0; i < 4; i++)
                        s += a[r + 4 * i] * b[i + 4 * c];
                    res[r + 4 * c] = s;
                }
            }

            return res;
        }

        /// <summary>Transforms a homogeneous point (<c>mjr_multiply4</c>).</summary>
        /// <param name="m">The matrix.</param>
        /// <param name="x">x.</param>
        /// <param name="y">y.</param>
        /// <param name="z">z.</param>
        /// <param name="w">w.</param>
        /// <param name="ox">The result's x.</param>
        /// <param name="oy">The result's y.</param>
        /// <param name="oz">The result's z.</param>
        /// <param name="ow">The result's w.</param>
        internal static void Transform(float[] m, float x, float y, float z, float w, out float ox, out float oy, out float oz, out float ow) {
            ox = m[0] * x + m[4] * y + m[8] * z + m[12] * w;
            oy = m[1] * x + m[5] * y + m[9] * z + m[13] * w;
            oz = m[2] * x + m[6] * y + m[10] * z + m[14] * w;
            ow = m[3] * x + m[7] * y + m[11] * z + m[15] * w;
        }

        /// <summary>
        ///     Transforms a homogeneous point as the reference GPU's vertex stage does: each component a fused
        ///     multiply-add chain in x, y, z, w order from the first product, <c>fma(m₁₂, w, fma(m₈, z, fma(m₄, y, m₀·x)))</c>,
        ///     one rounding per step. For the coordinates the GPU computes itself (clip, eye and texture coordinates); the
        ///     fixed-function position is this over the projection times the modelview from <see cref="MultiplyDriver"/>.
        /// </summary>
        /// <param name="m">The matrix, column-major.</param>
        /// <param name="x">x.</param>
        /// <param name="y">y.</param>
        /// <param name="z">z.</param>
        /// <param name="w">w.</param>
        /// <param name="ox">The result's x.</param>
        /// <param name="oy">The result's y.</param>
        /// <param name="oz">The result's z.</param>
        /// <param name="ow">The result's w.</param>
        /// <remarks>
        ///     Measured on 2026-11-21 (DR-98) by capturing <c>ftransform()</c>, which GLSL defines as the fixed-function
        ///     position, through transform feedback: every component of 6,400 random vertices under 200 random matrix pairs;
        ///     transforming by the modelview and then the projection, or summing without fusion, got 44% to 84%.
        /// </remarks>
        internal static void TransformGpu(float[] m, float x, float y, float z, float w, out float ox, out float oy, out float oz, out float ow) {
            ox = MathF.FusedMultiplyAdd(m[12], w, MathF.FusedMultiplyAdd(m[8], z, MathF.FusedMultiplyAdd(m[4], y, m[0] * x)));
            oy = MathF.FusedMultiplyAdd(m[13], w, MathF.FusedMultiplyAdd(m[9], z, MathF.FusedMultiplyAdd(m[5], y, m[1] * x)));
            oz = MathF.FusedMultiplyAdd(m[14], w, MathF.FusedMultiplyAdd(m[10], z, MathF.FusedMultiplyAdd(m[6], y, m[2] * x)));
            ow = MathF.FusedMultiplyAdd(m[15], w, MathF.FusedMultiplyAdd(m[11], z, MathF.FusedMultiplyAdd(m[7], y, m[3] * x)));
        }

        /// <summary>A plane (a row vector) times a matrix: how OpenGL stores eye planes and clip planes (<c>p·M⁻¹</c>).</summary>
        /// <param name="planes">The source planes.</param>
        /// <param name="offset">The plane's offset in <paramref name="planes"/>.</param>
        /// <param name="m">The matrix.</param>
        /// <param name="result">The destination.</param>
        /// <param name="resultOffset">The result's offset.</param>
        internal static void PlaneTimes(float[] planes, int offset, float[] m, float[] result, int resultOffset) {
            for (int c = 0; c < 4; c++) {
                result[resultOffset + c] = planes[offset] * m[4 * c] + planes[offset + 1] * m[4 * c + 1] +
                                           planes[offset + 2] * m[4 * c + 2] + planes[offset + 3] * m[4 * c + 3];
            }
        }

        /// <summary>The inverse of a general 4×4 matrix (cofactors), in double, stored as float.</summary>
        /// <param name="m">The matrix.</param>
        /// <returns>A new matrix; the identity for a singular one.</returns>
        internal static float[] Invert(float[] m) {
            var a = new double[16];
            for (int i = 0; i < 16; i++)
                a[i] = m[i];
            var inv = new double[16];
            inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] + a[9] * a[7] * a[14] + a[13] * a[6] * a[11] - a[13] * a[7] * a[10];
            inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] - a[8] * a[7] * a[14] - a[12] * a[6] * a[11] + a[12] * a[7] * a[10];
            inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] + a[8] * a[7] * a[13] + a[12] * a[5] * a[11] - a[12] * a[7] * a[9];
            inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] - a[8] * a[6] * a[13] - a[12] * a[5] * a[10] + a[12] * a[6] * a[9];
            inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] - a[9] * a[3] * a[14] - a[13] * a[2] * a[11] + a[13] * a[3] * a[10];
            inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] + a[8] * a[3] * a[14] + a[12] * a[2] * a[11] - a[12] * a[3] * a[10];
            inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] - a[8] * a[3] * a[13] - a[12] * a[1] * a[11] + a[12] * a[3] * a[9];
            inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] + a[8] * a[2] * a[13] + a[12] * a[1] * a[10] - a[12] * a[2] * a[9];
            inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] + a[5] * a[3] * a[14] + a[13] * a[2] * a[7] - a[13] * a[3] * a[6];
            inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] - a[4] * a[3] * a[14] - a[12] * a[2] * a[7] + a[12] * a[3] * a[6];
            inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] + a[4] * a[3] * a[13] + a[12] * a[1] * a[7] - a[12] * a[3] * a[5];
            inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] - a[4] * a[2] * a[13] - a[12] * a[1] * a[6] + a[12] * a[2] * a[5];
            inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] - a[5] * a[3] * a[10] - a[9] * a[2] * a[7] + a[9] * a[3] * a[6];
            inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] + a[4] * a[3] * a[10] + a[8] * a[2] * a[7] - a[8] * a[3] * a[6];
            inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] - a[4] * a[3] * a[9] - a[8] * a[1] * a[7] + a[8] * a[3] * a[5];
            inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] + a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - a[8] * a[2] * a[5];
            double det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
            if (det == 0)
                return Identity();
            var res = new float[16];
            for (int i = 0; i < 16; i++)
                res[i] = (float)(inv[i] / det);
            return res;
        }

        /// <summary>
        ///     The normal matrix: the inverse transpose of the modelview's upper 3×3, column-major 3×3, which transforms
        ///     normals so they stay perpendicular under non-uniform scaling (the builtin meshes are scaled per geom).
        /// </summary>
        /// <param name="mv">The modelview.</param>
        /// <returns>A new 3×3 matrix, column-major.</returns>
        internal static float[] NormalMatrix(float[] mv) {
            double a = mv[0], b = mv[4], c = mv[8], d = mv[1], e = mv[5], f = mv[9], g = mv[2], h = mv[6], i = mv[10];
            double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
            if (det == 0)
                det = 1;
            // the inverse's transpose: element (r, c) is the cofactor (r, c) over the determinant
            return [
                (float)((e * i - f * h) / det), (float)(-(b * i - c * h) / det), (float)((b * f - c * e) / det),
                (float)(-(d * i - f * g) / det), (float)((a * i - c * g) / det), (float)(-(a * f - c * d) / det),
                (float)((d * h - e * g) / det), (float)(-(a * h - b * g) / det), (float)((a * e - b * d) / det),
            ];
        }

        /// <summary>A translation (<c>glTranslatef</c>).</summary>
        /// <param name="x">x.</param>
        /// <param name="y">y.</param>
        /// <param name="z">z.</param>
        /// <returns>A new matrix.</returns>
        internal static float[] Translate(float x, float y, float z) => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1];

        /// <summary>A scaling (<c>glScalef</c>).</summary>
        /// <param name="x">x.</param>
        /// <param name="y">y.</param>
        /// <param name="z">z.</param>
        /// <returns>A new matrix.</returns>
        internal static float[] Scale(float x, float y, float z) => [x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1];

        /// <summary>
        ///     A rotation by <paramref name="degrees"/> about an axis (<c>glRotatef</c>, OpenGL 2.1 §2.11.2), built in double
        ///     from the normalized axis; a zero axis gives the identity.
        /// </summary>
        /// <param name="degrees">The angle, counter-clockwise looking down the axis.</param>
        /// <param name="x">The axis's x.</param>
        /// <param name="y">The axis's y.</param>
        /// <param name="z">The axis's z.</param>
        /// <returns>A new matrix.</returns>
        internal static float[] Rotate(float degrees, float x, float y, float z) {
            double len = Math.Sqrt((double)x * x + (double)y * y + (double)z * z);
            if (len == 0)
                return Identity();
            double ax = x / len, ay = y / len, az = z / len;
            double rad = degrees * Math.PI / 180.0, c = Math.Cos(rad), s = Math.Sin(rad), t = 1 - c;
            return [
                (float)(ax * ax * t + c), (float)(ay * ax * t + az * s), (float)(az * ax * t - ay * s), 0,
                (float)(ax * ay * t - az * s), (float)(ay * ay * t + c), (float)(az * ay * t + ax * s), 0,
                (float)(ax * az * t + ay * s), (float)(ay * az * t - ax * s), (float)(az * az * t + c), 0,
                0, 0, 0, 1,
            ];
        }

        /// <summary>
        ///     A perspective frustum (<c>glFrustum</c>) as the reference driver builds it: each element a numerator kept in
        ///     double over a denominator (<c>r - l</c>, <c>t - b</c>, <c>f - n</c>) rounded to float32, the quotient rounded
        ///     once to float32 (every element of 250 matrices with double arguments, 2026-11-21; dividing in double alone got
        ///     about 83%).
        /// </summary>
        /// <param name="l">Left.</param>
        /// <param name="r">Right.</param>
        /// <param name="b">Bottom.</param>
        /// <param name="t">Top.</param>
        /// <param name="n">Near.</param>
        /// <param name="f">Far.</param>
        /// <returns>A new matrix, to be multiplied into the stack with <see cref="MultiplyDriver"/>.</returns>
        internal static float[] Frustum(double l, double r, double b, double t, double n, double f) {
            // the denominators as the driver keeps them: rounded to float32, then used in double
            double rl = (float)(r - l), tb = (float)(t - b), fn = (float)(f - n);
            return [
                (float)(2 * n / rl), 0, 0, 0,
                0, (float)(2 * n / tb), 0, 0,
                (float)((r + l) / rl), (float)((t + b) / tb), (float)(-(f + n) / fn), -1,
                0, 0, (float)(-2 * f * n / fn), 0,
            ];
        }

        /// <summary>
        ///     An orthographic projection (<c>glOrtho</c>) as the reference driver builds it: each element computed in double
        ///     from the double arguments and rounded once to float32 (every element of 300 matrices, 2026-11-21).
        /// </summary>
        /// <param name="l">Left.</param>
        /// <param name="r">Right.</param>
        /// <param name="b">Bottom.</param>
        /// <param name="t">Top.</param>
        /// <param name="n">Near.</param>
        /// <param name="f">Far.</param>
        /// <returns>A new matrix.</returns>
        internal static float[] Ortho(double l, double r, double b, double t, double n, double f) => [
            (float)(2 / (r - l)), 0, 0, 0,
            0, (float)(2 / (t - b)), 0, 0,
            0, 0, (float)(-2 / (f - n)), 0,
            (float)(-(r + l) / (r - l)), (float)(-(t + b) / (t - b)), (float)(-(f + n) / (f - n)), 1,
        ];

        /// <summary>
        ///     MuJoCo's reverse-Z prefix with clip control (<c>glTranslatef(0, 0, 0.5)</c>, <c>glScalef(1, 1, -0.5)</c>):
        ///     near maps to depth 1, far to 0.
        /// </summary>
        /// <returns>A new matrix.</returns>
        internal static float[] ReverseZ() => MultiplyDriver(Translate(0, 0, 0.5f), Scale(1, 1, -0.5f));

        /// <summary>Normalizes a 3-vector as <c>mjr_normalizeVec</c> does (a tiny one becomes +z).</summary>
        /// <param name="v">The vector, normalized in place.</param>
        internal static void Normalize(Span<float> v) {
            float len = MathF.Sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            if (len < 1e-10f) {
                v[0] = 0;
                v[1] = 0;
                v[2] = 1;
            } else {
                float scl = 1.0f / len;
                v[0] *= scl;
                v[1] *= scl;
                v[2] *= scl;
            }
        }

        /// <summary>The cross product <c>b×c</c> (<c>mjr_crossVec</c>).</summary>
        /// <param name="a">The result.</param>
        /// <param name="b">b.</param>
        /// <param name="c">c.</param>
        internal static void Cross(Span<float> a, ReadOnlySpan<float> b, ReadOnlySpan<float> c) {
            a[0] = b[1] * c[2] - b[2] * c[1];
            a[1] = b[2] * c[0] - b[0] * c[2];
            a[2] = b[0] * c[1] - b[1] * c[0];
        }

        /// <summary><c>mjr_lookAt</c>: the view of an eye looking along <paramref name="forward"/> with <paramref name="up"/>.</summary>
        /// <param name="eye">The eye.</param>
        /// <param name="forward">The forward direction.</param>
        /// <param name="up">The up direction.</param>
        /// <returns>A new matrix.</returns>
        internal static float[] LookAt(ReadOnlySpan<float> eye, ReadOnlySpan<float> forward, ReadOnlySpan<float> up) {
            Span<float> f = stackalloc float[3], s = stackalloc float[3], u = stackalloc float[3];
            forward[..3].CopyTo(f);
            Normalize(f);
            Cross(s, f, up);
            Normalize(s);
            Cross(u, s, f);
            Normalize(u);
            return [
                s[0], u[0], -f[0], 0,
                s[1], u[1], -f[1], 0,
                s[2], u[2], -f[2], 0,
                -(s[0] * eye[0] + s[1] * eye[1] + s[2] * eye[2]), -(u[0] * eye[0] + u[1] * eye[1] + u[2] * eye[2]),
                f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2], 1,
            ];
        }

        /// <summary><c>mjr_reflect</c>: the reflection about the plane through <paramref name="pos"/> with normal = the frame's z axis.</summary>
        /// <param name="pos">A point of the plane.</param>
        /// <param name="mat">The plane's frame, row-major 3×3.</param>
        /// <returns>A new matrix.</returns>
        internal static float[] Reflect(ReadOnlySpan<float> pos, ReadOnlySpan<float> mat) {
            float v0 = mat[2], v1 = mat[5], v2 = mat[8];
            Span<float> vvt = [v0 * v0, v0 * v1, v0 * v2, v1 * v0, v1 * v1, v1 * v2, v2 * v0, v2 * v1, v2 * v2];
            return [
                1 - 2 * vvt[0], -2 * vvt[1], -2 * vvt[2], 0,
                -2 * vvt[3], 1 - 2 * vvt[4], -2 * vvt[5], 0,
                -2 * vvt[6], -2 * vvt[7], 1 - 2 * vvt[8], 0,
                2 * (vvt[0] * pos[0] + vvt[1] * pos[1] + vvt[2] * pos[2]),
                2 * (vvt[3] * pos[0] + vvt[4] * pos[1] + vvt[5] * pos[2]),
                2 * (vvt[6] * pos[0] + vvt[7] * pos[1] + vvt[8] * pos[2]), 1,
            ];
        }

        /// <summary><c>mjr_orthoVec</c>: a unit vector orthogonal to <paramref name="v"/> (an up direction for a light's view).</summary>
        /// <param name="res">The result.</param>
        /// <param name="v">The vector.</param>
        internal static void OrthoVec(Span<float> res, ReadOnlySpan<float> v) {
            Span<float> other = [-1, 0, 0];
            Cross(res, v, other);
            if (res[0] * res[0] + res[1] * res[1] + res[2] * res[2] > 0.01) {
                Normalize(res);
                return;
            }

            other[0] = 0;
            other[1] = 1;
            Cross(res, v, other);
            Normalize(res);
        }
    }
}
