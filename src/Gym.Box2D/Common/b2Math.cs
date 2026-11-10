/*
* Copyright (c) 2006-2009 Erin Catto http://www.box2d.org
*
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
*/
// Altered source version (Gym.NET, 2026-11-10): Box2D/Common/b2Math.h and b2Math.cpp of pybox2d 2.3.10's bundled
// Box2D 2.3.0, translated to C#. Every operation keeps the C++ float32 order; the free functions live in the static
// class b2Math, which the engine imports with 'using static'; sinf/cosf/atan2f/sqrtf/floorf are MathF's, which call the
// same C runtime functions on Windows.
using System;
using System.Runtime.CompilerServices;
using static Gym.Box2D.b2Settings;

namespace Gym.Box2D {
    /// <summary>
    ///     Box2D's 2-D column vector, float32, the type every position, velocity, force and vertex travels in, and the value
    ///     pybox2d hands Python as <c>b2Vec2</c>. Arithmetic on it is float32, as in pybox2d, whose vector operators run in
    ///     C++: <c>v * 2.5</c> rounds 2.5 to float32 first and multiplies in float32.
    /// </summary>
    /// <remarks>
    ///     Footgun: reading <see cref="x"/> and <see cref="y"/> into Python-style double arithmetic is exact, but multiplying
    ///     two components in C# (<c>v.x * v.y</c>) stays float32, where Gymnasium's Python computes a float64 product. Write
    ///     <c>(double)v.x * v.y</c> wherever the Python operands are plain floats.
    ///     <para>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Common/b2Math.h#L58">Box2D/Common/b2Math.h</see>.</para>
    /// </remarks>
    public partial struct b2Vec2 {
        /// <summary>The x coordinate.</summary>
        public float x;

        /// <summary>The y coordinate.</summary>
        public float y;

        /// <summary>A vector of the given coordinates.</summary>
        /// <param name="x">The x coordinate.</param>
        /// <param name="y">The y coordinate.</param>
        public b2Vec2(float x, float y) {
            this.x = x;
            this.y = y;
        }

        /// <summary>Sets this vector to all zeros.</summary>
        public void SetZero() {
            x = 0.0f;
            y = 0.0f;
        }

        /// <summary>Sets this vector to some specified coordinates.</summary>
        /// <param name="x_">The x coordinate.</param>
        /// <param name="y_">The y coordinate.</param>
        public void Set(float x_, float y_) {
            x = x_;
            y = y_;
        }

        /// <summary>The vector's length, <c>sqrtf(x * x + y * y)</c>.</summary>
        /// <returns>The length.</returns>
        public readonly float Length() => MathF.Sqrt(x * x + y * y);

        /// <summary>The squared length, cheaper than <see cref="Length"/> when only comparisons are needed.</summary>
        /// <returns><c>x * x + y * y</c>.</returns>
        public readonly float LengthSquared() => x * x + y * y;

        /// <summary>Converts this vector into a unit vector in place; a vector shorter than FLT_EPSILON is left as it is.</summary>
        /// <returns>The length before normalizing, or 0 for a vector too short to normalize.</returns>
        public float Normalize() {
            float length = Length();
            if (length < b2_epsilon)
                return 0.0f;
            float invLength = 1.0f / length;
            x *= invLength;
            y *= invLength;
            return length;
        }

        /// <summary>Whether both coordinates are finite (not NaN or infinite).</summary>
        /// <returns>True for a finite vector.</returns>
        public readonly bool IsValid() => b2Math.b2IsValid(x) && b2Math.b2IsValid(y);

        /// <summary>The skew vector, <c>(-y, x)</c>, such that <c>dot(skew_vec, other) == cross(vec, other)</c>.</summary>
        /// <returns>The skew vector.</returns>
        public readonly b2Vec2 Skew() => new b2Vec2(-y, x);

        /// <summary>Negates both coordinates.</summary>
        /// <param name="v">The vector.</param>
        /// <returns><c>(-x, -y)</c>.</returns>
        public static b2Vec2 operator -(b2Vec2 v) => new b2Vec2(-v.x, -v.y);

        /// <summary>Adds two vectors coordinate by coordinate in float32.</summary>
        /// <param name="a">The first vector.</param>
        /// <param name="b">The second vector.</param>
        /// <returns><c>(a.x + b.x, a.y + b.y)</c>.</returns>
        public static b2Vec2 operator +(b2Vec2 a, b2Vec2 b) => new b2Vec2(a.x + b.x, a.y + b.y);

        /// <summary>Subtracts two vectors coordinate by coordinate in float32.</summary>
        /// <param name="a">The first vector.</param>
        /// <param name="b">The second vector.</param>
        /// <returns><c>(a.x - b.x, a.y - b.y)</c>.</returns>
        public static b2Vec2 operator -(b2Vec2 a, b2Vec2 b) => new b2Vec2(a.x - b.x, a.y - b.y);

        /// <summary>Scales a vector in float32.</summary>
        /// <param name="s">The scale.</param>
        /// <param name="a">The vector.</param>
        /// <returns><c>(s * a.x, s * a.y)</c>.</returns>
        public static b2Vec2 operator *(float s, b2Vec2 a) => new b2Vec2(s * a.x, s * a.y);

        /// <summary>Scales a vector in float32 (pybox2d's <c>__mul__</c>, and C++'s <c>*=</c>).</summary>
        /// <param name="a">The vector.</param>
        /// <param name="s">The scale.</param>
        /// <returns><c>(a.x * s, a.y * s)</c>, the same bits as <c>s * a</c>.</returns>
        public static b2Vec2 operator *(b2Vec2 a, float s) => new b2Vec2(a.x * s, a.y * s);

        /// <summary>Divides a vector by a scalar in float32 (pybox2d's <c>__truediv__</c>).</summary>
        /// <param name="a">The vector.</param>
        /// <param name="s">The divisor.</param>
        /// <returns><c>(a.x / s, a.y / s)</c>.</returns>
        public static b2Vec2 operator /(b2Vec2 a, float s) => new b2Vec2(a.x / s, a.y / s);

        /// <summary>Exact coordinate equality, as C++'s <c>operator ==</c> and pybox2d's <c>__eq__</c> (NaN never equals).</summary>
        /// <param name="a">The first vector.</param>
        /// <param name="b">The second vector.</param>
        /// <returns>True when both coordinates compare equal.</returns>
        public static bool operator ==(b2Vec2 a, b2Vec2 b) => a.x == b.x && a.y == b.y;

        /// <summary>The negation of <see cref="op_Equality"/>.</summary>
        /// <param name="a">The first vector.</param>
        /// <param name="b">The second vector.</param>
        /// <returns>True when a coordinate differs (or either is NaN).</returns>
        public static bool operator !=(b2Vec2 a, b2Vec2 b) => !(a == b);

        /// <summary>Equality with another boxed vector, by <see cref="op_Equality"/>.</summary>
        /// <param name="obj">The other object.</param>
        /// <returns>True for an equal <see cref="b2Vec2"/>.</returns>
        public override readonly bool Equals(object? obj) => obj is b2Vec2 other && this == other;

        /// <summary>A hash of both coordinates' bits.</summary>
        /// <returns>The hash code.</returns>
        public override readonly int GetHashCode() => HashCode.Combine(x, y);
    }

    /// <summary>A 3-D column vector, float32, used by the joint solvers' 3x3 systems.</summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Common/b2Math.h#L145">Box2D/Common/b2Math.h</see>.</remarks>
    internal struct b2Vec3 {
        /// <summary>The x coordinate.</summary>
        public float x;

        /// <summary>The y coordinate.</summary>
        public float y;

        /// <summary>The z coordinate.</summary>
        public float z;

        /// <summary>A vector of the given coordinates.</summary>
        /// <param name="x">The x coordinate.</param>
        /// <param name="y">The y coordinate.</param>
        /// <param name="z">The z coordinate.</param>
        public b2Vec3(float x, float y, float z) {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        /// <summary>Sets this vector to all zeros.</summary>
        public void SetZero() {
            x = 0.0f;
            y = 0.0f;
            z = 0.0f;
        }

        /// <summary>Sets this vector to some specified coordinates.</summary>
        /// <param name="x_">The x coordinate.</param>
        /// <param name="y_">The y coordinate.</param>
        /// <param name="z_">The z coordinate.</param>
        public void Set(float x_, float y_, float z_) {
            x = x_;
            y = y_;
            z = z_;
        }

        /// <summary>Negates every coordinate.</summary>
        /// <param name="v">The vector.</param>
        /// <returns><c>(-x, -y, -z)</c>.</returns>
        public static b2Vec3 operator -(b2Vec3 v) => new b2Vec3(-v.x, -v.y, -v.z);

        /// <summary>Adds two vectors coordinate by coordinate.</summary>
        /// <param name="a">The first vector.</param>
        /// <param name="b">The second vector.</param>
        /// <returns>The sum.</returns>
        public static b2Vec3 operator +(b2Vec3 a, b2Vec3 b) => new b2Vec3(a.x + b.x, a.y + b.y, a.z + b.z);

        /// <summary>Subtracts two vectors coordinate by coordinate.</summary>
        /// <param name="a">The first vector.</param>
        /// <param name="b">The second vector.</param>
        /// <returns>The difference.</returns>
        public static b2Vec3 operator -(b2Vec3 a, b2Vec3 b) => new b2Vec3(a.x - b.x, a.y - b.y, a.z - b.z);

        /// <summary>Scales a vector.</summary>
        /// <param name="s">The scale.</param>
        /// <param name="a">The vector.</param>
        /// <returns><c>(s * a.x, s * a.y, s * a.z)</c>.</returns>
        public static b2Vec3 operator *(float s, b2Vec3 a) => new b2Vec3(s * a.x, s * a.y, s * a.z);

        /// <summary>Scales a vector, as C++'s <c>*=</c> does in place.</summary>
        /// <param name="a">The vector.</param>
        /// <param name="s">The scale.</param>
        /// <returns><c>(a.x * s, a.y * s, a.z * s)</c>.</returns>
        public static b2Vec3 operator *(b2Vec3 a, float s) => new b2Vec3(a.x * s, a.y * s, a.z * s);
    }

    /// <summary>A 2x2 matrix stored in column-major order (<c>ex</c>, <c>ey</c> are the columns).</summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Common/b2Math.h#L190">Box2D/Common/b2Math.h</see>.</remarks>
    internal struct b2Mat22 {
        /// <summary>The first column.</summary>
        public b2Vec2 ex;

        /// <summary>The second column.</summary>
        public b2Vec2 ey;

        /// <summary>A matrix of the given columns.</summary>
        /// <param name="c1">The first column.</param>
        /// <param name="c2">The second column.</param>
        public b2Mat22(b2Vec2 c1, b2Vec2 c2) {
            ex = c1;
            ey = c2;
        }

        /// <summary>A matrix of the given scalars, in row-major reading order.</summary>
        /// <param name="a11">Row 1, column 1.</param>
        /// <param name="a12">Row 1, column 2.</param>
        /// <param name="a21">Row 2, column 1.</param>
        /// <param name="a22">Row 2, column 2.</param>
        public b2Mat22(float a11, float a12, float a21, float a22) {
            ex.x = a11;
            ex.y = a21;
            ey.x = a12;
            ey.y = a22;
        }

        /// <summary>Sets the columns.</summary>
        /// <param name="c1">The first column.</param>
        /// <param name="c2">The second column.</param>
        public void Set(b2Vec2 c1, b2Vec2 c2) {
            ex = c1;
            ey = c2;
        }

        /// <summary>Sets this to the identity matrix.</summary>
        public void SetIdentity() {
            ex.x = 1.0f;
            ey.x = 0.0f;
            ex.y = 0.0f;
            ey.y = 1.0f;
        }

        /// <summary>Sets this matrix to all zeros.</summary>
        public void SetZero() {
            ex.x = 0.0f;
            ey.x = 0.0f;
            ex.y = 0.0f;
            ey.y = 0.0f;
        }

        /// <summary>The inverse, with a zero determinant left as 0 (so a singular matrix inverts to zeros).</summary>
        /// <returns>The inverse.</returns>
        public readonly b2Mat22 GetInverse() {
            float a = ex.x, b = ey.x, c = ex.y, d = ey.y;
            b2Mat22 B;
            float det = a * d - b * c;
            if (det != 0.0f)
                det = 1.0f / det;
            B.ex.x = det * d;
            B.ey.x = -det * b;
            B.ex.y = -det * c;
            B.ey.y = det * a;
            return B;
        }

        /// <summary>Solves <c>A * x = b</c> without forming the inverse; a singular matrix yields zeros.</summary>
        /// <param name="b">The right-hand side.</param>
        /// <returns>The solution x.</returns>
        public readonly b2Vec2 Solve(b2Vec2 b) {
            float a11 = ex.x, a12 = ey.x, a21 = ex.y, a22 = ey.y;
            float det = a11 * a22 - a12 * a21;
            if (det != 0.0f)
                det = 1.0f / det;
            b2Vec2 x;
            x.x = det * (a22 * b.x - a12 * b.y);
            x.y = det * (a11 * b.y - a21 * b.x);
            return x;
        }
    }

    /// <summary>A 3x3 matrix stored in column-major order, for the joints' coupled constraints.</summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Common/b2Math.cpp">Box2D/Common/b2Math.cpp</see>.</remarks>
    internal struct b2Mat33 {
        /// <summary>The first column.</summary>
        public b2Vec3 ex;

        /// <summary>The second column.</summary>
        public b2Vec3 ey;

        /// <summary>The third column.</summary>
        public b2Vec3 ez;

        /// <summary>A matrix of the given columns.</summary>
        /// <param name="c1">The first column.</param>
        /// <param name="c2">The second column.</param>
        /// <param name="c3">The third column.</param>
        public b2Mat33(b2Vec3 c1, b2Vec3 c2, b2Vec3 c3) {
            ex = c1;
            ey = c2;
            ez = c3;
        }

        /// <summary>Sets this matrix to all zeros.</summary>
        public void SetZero() {
            ex.SetZero();
            ey.SetZero();
            ez.SetZero();
        }

        /// <summary>Solves <c>A * x = b</c> by Cramer's rule; a singular matrix yields zeros.</summary>
        /// <param name="b">The right-hand side.</param>
        /// <returns>The solution x.</returns>
        public readonly b2Vec3 Solve33(b2Vec3 b) {
            float det = b2Math.b2Dot(ex, b2Math.b2Cross(ey, ez));
            if (det != 0.0f)
                det = 1.0f / det;
            b2Vec3 x;
            x.x = det * b2Math.b2Dot(b, b2Math.b2Cross(ey, ez));
            x.y = det * b2Math.b2Dot(ex, b2Math.b2Cross(b, ez));
            x.z = det * b2Math.b2Dot(ex, b2Math.b2Cross(ey, b));
            return x;
        }

        /// <summary>Solves the upper-left 2x2 block, <c>A22 * x = b</c>; a singular block yields zeros.</summary>
        /// <param name="b">The right-hand side.</param>
        /// <returns>The solution x.</returns>
        public readonly b2Vec2 Solve22(b2Vec2 b) {
            float a11 = ex.x, a12 = ey.x, a21 = ex.y, a22 = ey.y;
            float det = a11 * a22 - a12 * a21;
            if (det != 0.0f)
                det = 1.0f / det;
            b2Vec2 x;
            x.x = det * (a22 * b.x - a12 * b.y);
            x.y = det * (a11 * b.y - a21 * b.x);
            return x;
        }

        /// <summary>The inverse of the upper-left 2x2 block, embedded in a 3x3 matrix with zeros elsewhere.</summary>
        /// <param name="M">Receives the inverse.</param>
        public readonly void GetInverse22(ref b2Mat33 M) {
            float a = ex.x, b = ey.x, c = ex.y, d = ey.y;
            float det = a * d - b * c;
            if (det != 0.0f)
                det = 1.0f / det;
            M.ex.x = det * d;
            M.ey.x = -det * b;
            M.ex.z = 0.0f;
            M.ex.y = -det * c;
            M.ey.y = det * a;
            M.ey.z = 0.0f;
            M.ez.x = 0.0f;
            M.ez.y = 0.0f;
            M.ez.z = 0.0f;
        }

        /// <summary>The symmetric inverse of this (assumed symmetric) matrix; the zero matrix if singular.</summary>
        /// <param name="M">Receives the inverse.</param>
        public readonly void GetSymInverse33(ref b2Mat33 M) {
            float det = b2Math.b2Dot(ex, b2Math.b2Cross(ey, ez));
            if (det != 0.0f)
                det = 1.0f / det;
            float a11 = ex.x, a12 = ey.x, a13 = ez.x;
            float a22 = ey.y, a23 = ez.y;
            float a33 = ez.z;
            M.ex.x = det * (a22 * a33 - a23 * a23);
            M.ex.y = det * (a13 * a23 - a12 * a33);
            M.ex.z = det * (a12 * a23 - a13 * a22);
            M.ey.x = M.ex.y;
            M.ey.y = det * (a11 * a33 - a13 * a13);
            M.ey.z = det * (a13 * a12 - a11 * a23);
            M.ez.x = M.ex.z;
            M.ez.y = M.ey.z;
            M.ez.z = det * (a11 * a22 - a12 * a12);
        }
    }

    /// <summary>A rotation as its sine and cosine, float32, computed once by <c>sinf</c>/<c>cosf</c> when set from an angle.</summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Common/b2Math.h#L307">Box2D/Common/b2Math.h</see>.</remarks>
    public partial struct b2Rot {
        /// <summary>The sine of the angle.</summary>
        public float s;

        /// <summary>The cosine of the angle.</summary>
        public float c;

        /// <summary>The rotation by an angle in radians.</summary>
        /// <param name="angle">The angle.</param>
        public b2Rot(float angle) {
            s = MathF.Sin(angle);
            c = MathF.Cos(angle);
        }

        /// <summary>Sets the rotation from an angle in radians.</summary>
        /// <param name="angle">The angle.</param>
        public void Set(float angle) {
            s = MathF.Sin(angle);
            c = MathF.Cos(angle);
        }

        /// <summary>Sets this to the identity rotation.</summary>
        public void SetIdentity() {
            s = 0.0f;
            c = 1.0f;
        }

        /// <summary>The angle in radians, <c>atan2f(s, c)</c>, in (-π, π].</summary>
        /// <returns>The angle.</returns>
        public readonly float GetAngle() => MathF.Atan2(s, c);

        /// <summary>The rotated x-axis, <c>(c, s)</c>.</summary>
        /// <returns>The axis.</returns>
        public readonly b2Vec2 GetXAxis() => new b2Vec2(c, s);

        /// <summary>The rotated y-axis, <c>(-s, c)</c>.</summary>
        /// <returns>The axis.</returns>
        public readonly b2Vec2 GetYAxis() => new b2Vec2(-s, c);
    }

    /// <summary>A rigid transform, a translation <see cref="p"/> and a rotation <see cref="q"/>: a body's frame in the world.</summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Common/b2Math.h#L353">Box2D/Common/b2Math.h</see>.</remarks>
    public partial struct b2Transform {
        /// <summary>The translation.</summary>
        public b2Vec2 p;

        /// <summary>The rotation.</summary>
        public b2Rot q;

        /// <summary>A transform of the given translation and rotation.</summary>
        /// <param name="position">The translation.</param>
        /// <param name="rotation">The rotation.</param>
        public b2Transform(b2Vec2 position, b2Rot rotation) {
            p = position;
            q = rotation;
        }

        /// <summary>Sets this to the identity transform.</summary>
        public void SetIdentity() {
            p.SetZero();
            q.SetIdentity();
        }

        /// <summary>Sets the translation and the rotation's angle.</summary>
        /// <param name="position">The translation.</param>
        /// <param name="angle">The angle in radians.</param>
        public void Set(b2Vec2 position, float angle) {
            p = position;
            q.Set(angle);
        }
    }

    /// <summary>
    ///     The motion of a body over a time step, for continuous collision: its center's start and end positions and angles.
    ///     The center may differ from the body origin, so the transform derived at a fraction of the step shifts back by the
    ///     local center.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Common/b2Math.h#L385">Box2D/Common/b2Math.h</see>.</remarks>
    internal struct b2Sweep {
        /// <summary>The local center of mass position.</summary>
        public b2Vec2 localCenter;

        /// <summary>The center's world position at the start of the sweep.</summary>
        public b2Vec2 c0;

        /// <summary>The center's world position at the end of the sweep.</summary>
        public b2Vec2 c;

        /// <summary>The world angle at the start of the sweep.</summary>
        public float a0;

        /// <summary>The world angle at the end of the sweep.</summary>
        public float a;

        /// <summary>The fraction of the current time step in [0, 1) that <see cref="c0"/> and <see cref="a0"/> are at.</summary>
        public float alpha0;

        /// <summary>The interpolated transform at a fraction <paramref name="beta"/> of the sweep, shifted to the body origin.</summary>
        /// <param name="xf">Receives the transform.</param>
        /// <param name="beta">The fraction in [0, 1].</param>
        public readonly void GetTransform(out b2Transform xf, float beta) {
            xf.p = (1.0f - beta) * c0 + beta * c;
            float angle = (1.0f - beta) * a0 + beta * a;
            xf.q = default;
            xf.q.Set(angle);
            // Shift to origin
            xf.p -= b2Math.b2Mul(xf.q, localCenter);
        }

        /// <summary>Advances the sweep's start forward to a new fraction of the step, interpolating position and angle.</summary>
        /// <param name="alpha">The new initial fraction.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown when the sweep already starts at the end of the step.</exception>
        public void Advance(float alpha) {
            b2Assert(alpha0 < 1.0f);
            float beta = (alpha - alpha0) / (1.0f - alpha0);
            c0 = (1.0f - beta) * c0 + beta * c;
            a0 = (1.0f - beta) * a0 + beta * a;
            alpha0 = alpha;
        }

        /// <summary>Normalizes the angles into [0, 2π) by whole turns, keeping their difference.</summary>
        public void Normalize() {
            float twoPi = 2.0f * b2_pi;
            float d = twoPi * MathF.Floor(a0 / twoPi);
            a0 -= d;
            a -= d;
        }
    }

    /// <summary>
    ///     Box2D's free math functions (dot and cross products, matrix and transform products, min/max/abs/clamp templates),
    ///     float32 with the C++ order of operations; the engine imports them with <c>using static</c>.
    /// </summary>
    /// <remarks>
    ///     Footgun: <c>b2Min</c>/<c>b2Max</c>/<c>b2Abs</c> are the C++ templates (<c>a &lt; b ? a : b</c>), not
    ///     <see cref="MathF.Min(float, float)"/>: they differ for NaN and signed zeros, and the engine relies on the templates.
    ///     <para>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Common/b2Math.h#L420">Box2D/Common/b2Math.h</see>.</para>
    /// </remarks>
    internal static class b2Math {
        /// <summary>The zero vector.</summary>
        public static readonly b2Vec2 b2Vec2_zero = new b2Vec2(0.0f, 0.0f);

        /// <summary>Whether a float is finite: not NaN and strictly between the infinities.</summary>
        /// <param name="x">The value.</param>
        /// <returns>True for a finite value.</returns>
        public static bool b2IsValid(float x) {
            // C++ tests NaN as 'x != x', which C# flags as a self-comparison; IsNaN is the same test.
            if (float.IsNaN(x))
                // NaN.
                return false;
            float infinity = float.PositiveInfinity;
            return -infinity < x && x < infinity;
        }

        /// <summary>The dot product of two vectors, <c>a.x * b.x + a.y * b.y</c>.</summary>
        /// <param name="a">The first vector.</param>
        /// <param name="b">The second vector.</param>
        /// <returns>The dot product.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float b2Dot(in b2Vec2 a, in b2Vec2 b) => a.x * b.x + a.y * b.y;

        /// <summary>The 2-D cross product of two vectors, a scalar: <c>a.x * b.y - a.y * b.x</c>.</summary>
        /// <param name="a">The first vector.</param>
        /// <param name="b">The second vector.</param>
        /// <returns>The cross product.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float b2Cross(in b2Vec2 a, in b2Vec2 b) => a.x * b.y - a.y * b.x;

        /// <summary>The cross product of a vector and a scalar, a vector: <c>(s * a.y, -s * a.x)</c>.</summary>
        /// <param name="a">The vector.</param>
        /// <param name="s">The scalar.</param>
        /// <returns>The cross product.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static b2Vec2 b2Cross(in b2Vec2 a, float s) => new b2Vec2(s * a.y, -s * a.x);

        /// <summary>The cross product of a scalar and a vector, a vector: <c>(-s * a.y, s * a.x)</c>.</summary>
        /// <param name="s">The scalar.</param>
        /// <param name="a">The vector.</param>
        /// <returns>The cross product.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static b2Vec2 b2Cross(float s, in b2Vec2 a) => new b2Vec2(-s * a.y, s * a.x);

        /// <summary>Multiplies a matrix by a vector.</summary>
        /// <param name="A">The matrix.</param>
        /// <param name="v">The vector.</param>
        /// <returns><c>A * v</c>.</returns>
        public static b2Vec2 b2Mul(in b2Mat22 A, in b2Vec2 v) => new b2Vec2(A.ex.x * v.x + A.ey.x * v.y, A.ex.y * v.x + A.ey.y * v.y);

        /// <summary>Multiplies the transpose of a matrix by a vector.</summary>
        /// <param name="A">The matrix.</param>
        /// <param name="v">The vector.</param>
        /// <returns><c>A^T * v</c>.</returns>
        public static b2Vec2 b2MulT(in b2Mat22 A, in b2Vec2 v) => new b2Vec2(b2Dot(v, A.ex), b2Dot(v, A.ey));

        /// <summary>The distance between two points.</summary>
        /// <param name="a">The first point.</param>
        /// <param name="b">The second point.</param>
        /// <returns>The distance.</returns>
        public static float b2Distance(in b2Vec2 a, in b2Vec2 b) {
            b2Vec2 c = a - b;
            return c.Length();
        }

        /// <summary>The squared distance between two points.</summary>
        /// <param name="a">The first point.</param>
        /// <param name="b">The second point.</param>
        /// <returns>The squared distance.</returns>
        public static float b2DistanceSquared(in b2Vec2 a, in b2Vec2 b) {
            b2Vec2 c = a - b;
            return b2Dot(c, c);
        }

        /// <summary>The dot product of two 3-D vectors.</summary>
        /// <param name="a">The first vector.</param>
        /// <param name="b">The second vector.</param>
        /// <returns>The dot product.</returns>
        public static float b2Dot(in b2Vec3 a, in b2Vec3 b) => a.x * b.x + a.y * b.y + a.z * b.z;

        /// <summary>The cross product of two 3-D vectors.</summary>
        /// <param name="a">The first vector.</param>
        /// <param name="b">The second vector.</param>
        /// <returns>The cross product.</returns>
        public static b2Vec3 b2Cross(in b2Vec3 a, in b2Vec3 b) => new b2Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);

        /// <summary>Adds two matrices column by column.</summary>
        /// <param name="A">The first matrix.</param>
        /// <param name="B">The second matrix.</param>
        /// <returns><c>A + B</c>.</returns>
        public static b2Mat22 b2Add(in b2Mat22 A, in b2Mat22 B) => new b2Mat22(A.ex + B.ex, A.ey + B.ey);

        /// <summary>Multiplies two matrices.</summary>
        /// <param name="A">The first matrix.</param>
        /// <param name="B">The second matrix.</param>
        /// <returns><c>A * B</c>.</returns>
        public static b2Mat22 b2Mul(in b2Mat22 A, in b2Mat22 B) => new b2Mat22(b2Mul(A, B.ex), b2Mul(A, B.ey));

        /// <summary>Multiplies the transpose of a matrix by another matrix.</summary>
        /// <param name="A">The first matrix.</param>
        /// <param name="B">The second matrix.</param>
        /// <returns><c>A^T * B</c>.</returns>
        public static b2Mat22 b2MulT(in b2Mat22 A, in b2Mat22 B) {
            b2Vec2 c1 = new b2Vec2(b2Dot(A.ex, B.ex), b2Dot(A.ey, B.ex));
            b2Vec2 c2 = new b2Vec2(b2Dot(A.ex, B.ey), b2Dot(A.ey, B.ey));
            return new b2Mat22(c1, c2);
        }

        /// <summary>Multiplies a 3x3 matrix by a 3-D vector, as <c>v.x * ex + v.y * ey + v.z * ez</c>.</summary>
        /// <param name="A">The matrix.</param>
        /// <param name="v">The vector.</param>
        /// <returns><c>A * v</c>.</returns>
        public static b2Vec3 b2Mul(in b2Mat33 A, in b2Vec3 v) => v.x * A.ex + v.y * A.ey + v.z * A.ez;

        /// <summary>Multiplies the upper-left 2x2 block of a 3x3 matrix by a 2-D vector.</summary>
        /// <param name="A">The matrix.</param>
        /// <param name="v">The vector.</param>
        /// <returns><c>A22 * v</c>.</returns>
        public static b2Vec2 b2Mul22(in b2Mat33 A, in b2Vec2 v) => new b2Vec2(A.ex.x * v.x + A.ey.x * v.y, A.ex.y * v.x + A.ey.y * v.y);

        /// <summary>Composes two rotations, <c>q * r</c>.</summary>
        /// <param name="q">The first rotation.</param>
        /// <param name="r">The second rotation.</param>
        /// <returns>The composed rotation.</returns>
        public static b2Rot b2Mul(in b2Rot q, in b2Rot r) {
            // [qc -qs] * [rc -rs] = [qc*rc-qs*rs -qc*rs-qs*rc]
            // [qs  qc]   [rs  rc]   [qs*rc+qc*rs -qs*rs+qc*rc]
            b2Rot qr;
            qr.s = q.s * r.c + q.c * r.s;
            qr.c = q.c * r.c - q.s * r.s;
            return qr;
        }

        /// <summary>Composes the transpose of a rotation with another, <c>q^T * r</c>.</summary>
        /// <param name="q">The first rotation.</param>
        /// <param name="r">The second rotation.</param>
        /// <returns>The composed rotation.</returns>
        public static b2Rot b2MulT(in b2Rot q, in b2Rot r) {
            // [ qc qs] * [rc -rs] = [qc*rc+qs*rs -qc*rs+qs*rc]
            // [-qs qc]   [rs  rc]   [-qs*rc+qc*rs qs*rs+qc*rc]
            b2Rot qr;
            qr.s = q.c * r.s - q.s * r.c;
            qr.c = q.c * r.c + q.s * r.s;
            return qr;
        }

        /// <summary>Rotates a vector.</summary>
        /// <param name="q">The rotation.</param>
        /// <param name="v">The vector.</param>
        /// <returns><c>q * v</c>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static b2Vec2 b2Mul(in b2Rot q, in b2Vec2 v) => new b2Vec2(q.c * v.x - q.s * v.y, q.s * v.x + q.c * v.y);

        /// <summary>Inverse-rotates a vector.</summary>
        /// <param name="q">The rotation.</param>
        /// <param name="v">The vector.</param>
        /// <returns><c>q^T * v</c>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static b2Vec2 b2MulT(in b2Rot q, in b2Vec2 v) => new b2Vec2(q.c * v.x + q.s * v.y, -q.s * v.x + q.c * v.y);

        /// <summary>Transforms a point from local to world coordinates: <c>(q.c * v.x - q.s * v.y) + p.x</c>, and so on.</summary>
        /// <param name="T">The transform.</param>
        /// <param name="v">The local point.</param>
        /// <returns>The world point.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static b2Vec2 b2Mul(in b2Transform T, in b2Vec2 v) {
            float x = (T.q.c * v.x - T.q.s * v.y) + T.p.x;
            float y = (T.q.s * v.x + T.q.c * v.y) + T.p.y;
            return new b2Vec2(x, y);
        }

        /// <summary>Transforms a point from world to local coordinates.</summary>
        /// <param name="T">The transform.</param>
        /// <param name="v">The world point.</param>
        /// <returns>The local point.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static b2Vec2 b2MulT(in b2Transform T, in b2Vec2 v) {
            float px = v.x - T.p.x;
            float py = v.y - T.p.y;
            float x = (T.q.c * px + T.q.s * py);
            float y = (-T.q.s * px + T.q.c * py);
            return new b2Vec2(x, y);
        }

        /// <summary>Composes two transforms: <c>A * B</c> maps B's local frame through A.</summary>
        /// <param name="A">The outer transform.</param>
        /// <param name="B">The inner transform.</param>
        /// <returns>The composed transform.</returns>
        public static b2Transform b2Mul(in b2Transform A, in b2Transform B) {
            // v2 = A.q.Rot(B.q.Rot(v1) + B.p) + A.p = (A.q * B.q).Rot(v1) + A.q.Rot(B.p) + A.p
            b2Transform C;
            C.q = b2Mul(A.q, B.q);
            C.p = b2Mul(A.q, B.p) + A.p;
            return C;
        }

        /// <summary>The relative transform <c>A^-1 * B</c>: B's frame expressed in A's.</summary>
        /// <param name="A">The reference transform.</param>
        /// <param name="B">The other transform.</param>
        /// <returns>The relative transform.</returns>
        public static b2Transform b2MulT(in b2Transform A, in b2Transform B) {
            // v2 = A.q' * (B.q * v1 + B.p - A.p) = A.q' * B.q * v1 + A.q' * (B.p - A.p)
            b2Transform C;
            C.q = b2MulT(A.q, B.q);
            C.p = b2MulT(A.q, B.p - A.p);
            return C;
        }

        /// <summary>The C++ template <c>a &gt; 0 ? a : -a</c> (so NaN comes back negated).</summary>
        /// <param name="a">The value.</param>
        /// <returns>The absolute value.</returns>
        public static float b2Abs(float a) => a > 0.0f ? a : -a;

        /// <summary>The C++ template <c>a &gt; 0 ? a : -a</c> for integers.</summary>
        /// <param name="a">The value.</param>
        /// <returns>The absolute value.</returns>
        public static int b2Abs(int a) => a > 0 ? a : -a;

        /// <summary>The absolute value of each coordinate.</summary>
        /// <param name="a">The vector.</param>
        /// <returns>The absolute vector.</returns>
        public static b2Vec2 b2Abs(in b2Vec2 a) => new b2Vec2(b2Abs(a.x), b2Abs(a.y));

        /// <summary>The absolute value of each matrix entry.</summary>
        /// <param name="A">The matrix.</param>
        /// <returns>The absolute matrix.</returns>
        public static b2Mat22 b2Abs(in b2Mat22 A) => new b2Mat22(b2Abs(A.ex), b2Abs(A.ey));

        /// <summary>The C++ template <c>a &lt; b ? a : b</c>.</summary>
        /// <param name="a">The first value.</param>
        /// <param name="b">The second value.</param>
        /// <returns>The smaller value, or <paramref name="b"/> when they compare unordered or equal.</returns>
        public static float b2Min(float a, float b) => a < b ? a : b;

        /// <summary>The C++ template <c>a &lt; b ? a : b</c> for integers.</summary>
        /// <param name="a">The first value.</param>
        /// <param name="b">The second value.</param>
        /// <returns>The smaller value.</returns>
        public static int b2Min(int a, int b) => a < b ? a : b;

        /// <summary>The coordinate-wise <see cref="b2Min(float, float)"/>.</summary>
        /// <param name="a">The first vector.</param>
        /// <param name="b">The second vector.</param>
        /// <returns>The coordinate-wise minimum.</returns>
        public static b2Vec2 b2Min(in b2Vec2 a, in b2Vec2 b) => new b2Vec2(b2Min(a.x, b.x), b2Min(a.y, b.y));

        /// <summary>The C++ template <c>a &gt; b ? a : b</c>.</summary>
        /// <param name="a">The first value.</param>
        /// <param name="b">The second value.</param>
        /// <returns>The larger value, or <paramref name="b"/> when they compare unordered or equal.</returns>
        public static float b2Max(float a, float b) => a > b ? a : b;

        /// <summary>The C++ template <c>a &gt; b ? a : b</c> for integers.</summary>
        /// <param name="a">The first value.</param>
        /// <param name="b">The second value.</param>
        /// <returns>The larger value.</returns>
        public static int b2Max(int a, int b) => a > b ? a : b;

        /// <summary>The coordinate-wise <see cref="b2Max(float, float)"/>.</summary>
        /// <param name="a">The first vector.</param>
        /// <param name="b">The second vector.</param>
        /// <returns>The coordinate-wise maximum.</returns>
        public static b2Vec2 b2Max(in b2Vec2 a, in b2Vec2 b) => new b2Vec2(b2Max(a.x, b.x), b2Max(a.y, b.y));

        /// <summary>The C++ template <c>b2Max(low, b2Min(a, high))</c>.</summary>
        /// <param name="a">The value.</param>
        /// <param name="low">The lower bound.</param>
        /// <param name="high">The upper bound.</param>
        /// <returns>The clamped value.</returns>
        public static float b2Clamp(float a, float low, float high) => b2Max(low, b2Min(a, high));

        /// <summary>The C++ template <c>b2Max(low, b2Min(a, high))</c> for integers.</summary>
        /// <param name="a">The value.</param>
        /// <param name="low">The lower bound.</param>
        /// <param name="high">The upper bound.</param>
        /// <returns>The clamped value.</returns>
        public static int b2Clamp(int a, int low, int high) => b2Max(low, b2Min(a, high));

        /// <summary>The coordinate-wise clamp.</summary>
        /// <param name="a">The vector.</param>
        /// <param name="low">The lower bounds.</param>
        /// <param name="high">The upper bounds.</param>
        /// <returns>The clamped vector.</returns>
        public static b2Vec2 b2Clamp(in b2Vec2 a, in b2Vec2 low, in b2Vec2 high) => b2Max(low, b2Min(a, high));

        /// <summary>Swaps two values.</summary>
        /// <typeparam name="T">The type.</typeparam>
        /// <param name="a">The first value.</param>
        /// <param name="b">The second value.</param>
        public static void b2Swap<T>(ref T a, ref T b) => (a, b) = (b, a);

        /// <summary>The next power of two above <paramref name="x"/> (bit smearing), as Box2D uses it to size pools.</summary>
        /// <param name="x">The value.</param>
        /// <returns>The power of two.</returns>
        public static uint b2NextPowerOfTwo(uint x) {
            x |= x >> 1;
            x |= x >> 2;
            x |= x >> 4;
            x |= x >> 8;
            x |= x >> 16;
            return x + 1;
        }

        /// <summary>Whether a value is a power of two.</summary>
        /// <param name="x">The value.</param>
        /// <returns>True for a positive power of two.</returns>
        public static bool b2IsPowerOfTwo(uint x) => x > 0 && (x & (x - 1)) == 0;
    }

    /// <summary>
    ///     Box2D's <c>b2GrowableStack&lt;int32, N&gt;</c>: a stack of tree node ids that starts with room for
    ///     <c>N</c> and doubles when full; the dynamic tree walks with it. Only the order of pushes and pops matters.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Common/b2GrowableStack.h">Box2D/Common/b2GrowableStack.h</see>.</remarks>
    internal struct b2GrowableStack {
        /// <summary>The storage.</summary>
        private int[] m_stack;

        /// <summary>The number of elements on the stack.</summary>
        private int m_count;

        /// <summary>An empty stack with room for <paramref name="capacity"/> elements before it grows.</summary>
        /// <param name="capacity">The initial capacity (Box2D uses 256).</param>
        public b2GrowableStack(int capacity) {
            m_stack = new int[capacity];
            m_count = 0;
        }

        /// <summary>Pushes an element, doubling the storage when full.</summary>
        /// <param name="element">The element.</param>
        public void Push(int element) {
            if (m_count == m_stack.Length)
                Array.Resize(ref m_stack, m_stack.Length * 2);
            m_stack[m_count] = element;
            ++m_count;
        }

        /// <summary>Pops the top element.</summary>
        /// <returns>The element.</returns>
        /// <exception cref="NumSharp.AssertionError">Thrown when the stack is empty.</exception>
        public int Pop() {
            b2Assert(m_count > 0);
            --m_count;
            return m_stack[m_count];
        }

        /// <summary>The number of elements on the stack.</summary>
        /// <returns>The count.</returns>
        public readonly int GetCount() => m_count;
    }
}
