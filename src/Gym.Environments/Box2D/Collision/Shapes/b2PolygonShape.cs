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
// Altered source version (Gym.NET, 2026-11-10): Box2D/Collision/Shapes/b2PolygonShape.h and b2PolygonShape.cpp of
// pybox2d 2.3.10's bundled Box2D 2.3.0, translated to C#. The vertex and normal arrays are managed arrays of
// b2_maxPolygonVertices, deep-copied by Clone as C++'s '*clone = *this' copies the inline arrays.
using System;
using static Gym.Box2D.b2Math;
using static Gym.Box2D.b2Settings;

namespace Gym.Box2D {
    /// <summary>
    ///     A solid convex polygon of 3 to 16 vertices in counter-clockwise order, with the polygon skin radius. Setting
    ///     vertices builds their convex hull (gift wrapping), so collinear and interior points are dropped and the order may
    ///     change.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Collision/Shapes/b2PolygonShape.cpp">Box2D/Collision/Shapes/b2PolygonShape.cpp</see>.</remarks>
    public sealed partial class b2PolygonShape : b2Shape {
        /// <summary>The centroid in the body's frame.</summary>
        internal b2Vec2 m_centroid;

        /// <summary>The hull's vertices; the first <see cref="m_count"/> are valid.</summary>
        internal b2Vec2[] m_vertices = new b2Vec2[b2_maxPolygonVertices];

        /// <summary>The outward unit normal of each edge (edge i runs from vertex i to i + 1).</summary>
        internal b2Vec2[] m_normals = new b2Vec2[b2_maxPolygonVertices];

        /// <summary>The number of vertices.</summary>
        internal int m_count;

        /// <summary>An empty polygon with the polygon skin radius; set its vertices before use.</summary>
        public b2PolygonShape() {
            m_type = b2ShapeType.e_polygon;
            m_radius = b2_polygonRadius;
            m_count = 0;
            m_centroid.SetZero();
        }

        /// <inheritdoc/>
        internal override b2Shape Clone() {
            var clone = (b2PolygonShape)MemberwiseClone();
            // C++ copies the inline arrays with the object; a managed clone would share them otherwise.
            clone.m_vertices = (b2Vec2[])m_vertices.Clone();
            clone.m_normals = (b2Vec2[])m_normals.Clone();
            return clone;
        }

        /// <summary>Makes this an axis-aligned box centered on the origin.</summary>
        /// <param name="hx">The half-width.</param>
        /// <param name="hy">The half-height.</param>
        internal void SetAsBox(float hx, float hy) {
            m_count = 4;
            m_vertices[0].Set(-hx, -hy);
            m_vertices[1].Set(hx, -hy);
            m_vertices[2].Set(hx, hy);
            m_vertices[3].Set(-hx, hy);
            m_normals[0].Set(0.0f, -1.0f);
            m_normals[1].Set(1.0f, 0.0f);
            m_normals[2].Set(0.0f, 1.0f);
            m_normals[3].Set(-1.0f, 0.0f);
            m_centroid.SetZero();
        }

        /// <summary>Makes this an oriented box.</summary>
        /// <param name="hx">The half-width.</param>
        /// <param name="hy">The half-height.</param>
        /// <param name="center">The box's center in local coordinates.</param>
        /// <param name="angle">The box's rotation in local coordinates.</param>
        internal void SetAsBox(float hx, float hy, in b2Vec2 center, float angle) {
            m_count = 4;
            m_vertices[0].Set(-hx, -hy);
            m_vertices[1].Set(hx, -hy);
            m_vertices[2].Set(hx, hy);
            m_vertices[3].Set(-hx, hy);
            m_normals[0].Set(0.0f, -1.0f);
            m_normals[1].Set(1.0f, 0.0f);
            m_normals[2].Set(0.0f, 1.0f);
            m_normals[3].Set(-1.0f, 0.0f);
            m_centroid = center;

            b2Transform xf;
            xf.p = center;
            xf.q = default;
            xf.q.Set(angle);

            // Transform vertices and normals.
            for (int i = 0; i < m_count; ++i) {
                m_vertices[i] = b2Mul(xf, m_vertices[i]);
                m_normals[i] = b2Mul(xf.q, m_normals[i]);
            }
        }

        /// <inheritdoc/>
        internal override int GetChildCount() => 1;

        /// <summary>The area-weighted centroid of a polygon, from triangles fanned around the origin.</summary>
        /// <param name="vs">The vertices.</param>
        /// <param name="count">The vertex count.</param>
        /// <returns>The centroid.</returns>
        /// <exception cref="NumSharp.AssertionError">Thrown for fewer than 3 vertices or a polygon of (nearly) no area.</exception>
        private static b2Vec2 ComputeCentroid(b2Vec2[] vs, int count) {
            b2Assert(count >= 3);

            b2Vec2 c = default;
            c.Set(0.0f, 0.0f);
            float area = 0.0f;

            // pRef is the reference point for forming triangles.
            // It's location doesn't change the result (except for rounding error).
            b2Vec2 pRef = new b2Vec2(0.0f, 0.0f);

            const float inv3 = 1.0f / 3.0f;

            for (int i = 0; i < count; ++i) {
                // Triangle vertices.
                b2Vec2 p1 = pRef;
                b2Vec2 p2 = vs[i];
                b2Vec2 p3 = i + 1 < count ? vs[i + 1] : vs[0];

                b2Vec2 e1 = p2 - p1;
                b2Vec2 e2 = p3 - p1;

                float D = b2Cross(e1, e2);

                float triangleArea = 0.5f * D;
                area += triangleArea;

                // Area weighted centroid
                c += triangleArea * inv3 * (p1 + p2 + p3);
            }

            // Centroid
            b2Assert(area > b2_epsilon);
            c *= 1.0f / area;
            return c;
        }

        /// <summary>
        ///     Sets the polygon to the convex hull of the points (gift wrapping from the right-most, lowest point), computing
        ///     the normals and centroid.
        /// </summary>
        /// <param name="vertices">The points.</param>
        /// <param name="count">How many points to read, 3 to 16.</param>
        /// <exception cref="NumSharp.AssertionError">
        ///     Thrown for a count outside 3..16, for a hull edge of (nearly) zero length, or for a hull of (nearly) no area.
        /// </exception>
        internal void Set(ReadOnlySpan<b2Vec2> vertices, int count) {
            b2Assert(3 <= count && count <= b2_maxPolygonVertices);
            if (count < 3) {
                SetAsBox(1.0f, 1.0f);
                return;
            }

            int n = b2Min(count, b2_maxPolygonVertices);

            // Copy vertices into local buffer
            Span<b2Vec2> ps = stackalloc b2Vec2[b2_maxPolygonVertices];
            for (int i = 0; i < n; ++i)
                ps[i] = vertices[i];

            // Create the convex hull using the Gift wrapping algorithm
            // http://en.wikipedia.org/wiki/Gift_wrapping_algorithm

            // Find the right most point on the hull
            int i0 = 0;
            float x0 = ps[0].x;
            for (int i = 1; i < count; ++i) {
                float x = ps[i].x;
                if (x > x0 || (x == x0 && ps[i].y < ps[i0].y)) {
                    i0 = i;
                    x0 = x;
                }
            }

            Span<int> hull = stackalloc int[b2_maxPolygonVertices];
            int m = 0;
            int ih = i0;

            for (; ; ) {
                hull[m] = ih;

                int ie = 0;
                for (int j = 1; j < n; ++j) {
                    if (ie == ih) {
                        ie = j;
                        continue;
                    }

                    b2Vec2 r = ps[ie] - ps[hull[m]];
                    b2Vec2 v = ps[j] - ps[hull[m]];
                    float c = b2Cross(r, v);
                    if (c < 0.0f)
                        ie = j;

                    // Collinearity check
                    if (c == 0.0f && v.LengthSquared() > r.LengthSquared())
                        ie = j;
                }

                ++m;
                ih = ie;

                if (ie == i0)
                    break;
            }

            m_count = m;

            // Copy vertices.
            for (int i = 0; i < m; ++i)
                m_vertices[i] = ps[hull[i]];

            // Compute normals. Ensure the edges have non-zero length.
            for (int i = 0; i < m; ++i) {
                int i1 = i;
                int i2 = i + 1 < m ? i + 1 : 0;
                b2Vec2 edge = m_vertices[i2] - m_vertices[i1];
                b2Assert(edge.LengthSquared() > b2_epsilon * b2_epsilon);
                m_normals[i] = b2Cross(edge, 1.0f);
                m_normals[i].Normalize();
            }

            // Compute the polygon centroid.
            m_centroid = ComputeCentroid(m_vertices, m);
        }

        /// <inheritdoc/>
        internal override bool TestPoint(in b2Transform xf, in b2Vec2 p) {
            b2Vec2 pLocal = b2MulT(xf.q, p - xf.p);

            for (int i = 0; i < m_count; ++i) {
                float dot = b2Dot(m_normals[i], pLocal - m_vertices[i]);
                if (dot > 0.0f)
                    return false;
            }

            return true;
        }

        /// <summary>Casts a ray against the polygon by clipping it against each edge's half-plane.</summary>
        /// <param name="output">Receives the hit; untouched on a miss.</param>
        /// <param name="input">The ray.</param>
        /// <param name="xf">The shape's transform.</param>
        /// <param name="childIndex">Unused.</param>
        /// <returns>True on a hit; a ray starting inside misses.</returns>
        /// <exception cref="NumSharp.AssertionError">Thrown if the entry fraction falls outside [0, maxFraction] (rounding).</exception>
        internal override bool RayCast(ref b2RayCastOutput output, in b2RayCastInput input, in b2Transform xf, int childIndex) {
            // Put the ray into the polygon's frame of reference.
            b2Vec2 p1 = b2MulT(xf.q, input.p1 - xf.p);
            b2Vec2 p2 = b2MulT(xf.q, input.p2 - xf.p);
            b2Vec2 d = p2 - p1;

            float lower = 0.0f, upper = input.maxFraction;

            int index = -1;

            for (int i = 0; i < m_count; ++i) {
                // p = p1 + a * d
                // dot(normal, p - v) = 0
                // dot(normal, p1 - v) + a * dot(normal, d) = 0
                float numerator = b2Dot(m_normals[i], m_vertices[i] - p1);
                float denominator = b2Dot(m_normals[i], d);

                if (denominator == 0.0f) {
                    if (numerator < 0.0f)
                        return false;
                } else {
                    // Note: we want this predicate without division:
                    // lower < numerator / denominator, where denominator < 0
                    // Since denominator < 0, we have to flip the inequality:
                    // lower < numerator / denominator <==> denominator * lower > numerator.
                    if (denominator < 0.0f && numerator < lower * denominator) {
                        // Increase lower.
                        // The segment enters this half-space.
                        lower = numerator / denominator;
                        index = i;
                    } else if (denominator > 0.0f && numerator < upper * denominator) {
                        // Decrease upper.
                        // The segment exits this half-space.
                        upper = numerator / denominator;
                    }
                }

                // The use of epsilon here causes the assert on lower to trip
                // in some cases. Apparently the use of epsilon was to make edge
                // shapes work, but now those are handled separately.
                //if (upper < lower - b2_epsilon)
                if (upper < lower)
                    return false;
            }

            b2Assert(0.0f <= lower && lower <= input.maxFraction);

            if (index >= 0) {
                output.fraction = lower;
                output.normal = b2Mul(xf.q, m_normals[index]);
                return true;
            }

            return false;
        }

        /// <inheritdoc/>
        internal override void ComputeAABB(out b2AABB aabb, in b2Transform xf, int childIndex) {
            b2Vec2 lower = b2Mul(xf, m_vertices[0]);
            b2Vec2 upper = lower;

            for (int i = 1; i < m_count; ++i) {
                b2Vec2 v = b2Mul(xf, m_vertices[i]);
                lower = b2Min(lower, v);
                upper = b2Max(upper, v);
            }

            b2Vec2 r = new b2Vec2(m_radius, m_radius);
            aabb.lowerBound = lower - r;
            aabb.upperBound = upper + r;
        }

        /// <summary>
        ///     The polygon's mass, centroid and inertia at a density, summed over triangles fanned around the vertex mean, then
        ///     shifted to the body origin.
        /// </summary>
        /// <param name="massData">Receives the mass properties.</param>
        /// <param name="density">The density.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown for fewer than 3 vertices or (nearly) no area.</exception>
        internal override void ComputeMass(out b2MassData massData, float density) {
            b2Assert(m_count >= 3);

            b2Vec2 center = default;
            center.Set(0.0f, 0.0f);
            float area = 0.0f;
            float I = 0.0f;

            // s is the reference point for forming triangles.
            // It's location doesn't change the result (except for rounding error).
            b2Vec2 s = new b2Vec2(0.0f, 0.0f);

            // This code would put the reference point inside the polygon.
            for (int i = 0; i < m_count; ++i)
                s += m_vertices[i];
            s *= 1.0f / m_count;

            const float k_inv3 = 1.0f / 3.0f;

            for (int i = 0; i < m_count; ++i) {
                // Triangle vertices.
                b2Vec2 e1 = m_vertices[i] - s;
                b2Vec2 e2 = i + 1 < m_count ? m_vertices[i + 1] - s : m_vertices[0] - s;

                float D = b2Cross(e1, e2);

                float triangleArea = 0.5f * D;
                area += triangleArea;

                // Area weighted centroid
                center += triangleArea * k_inv3 * (e1 + e2);

                float ex1 = e1.x, ey1 = e1.y;
                float ex2 = e2.x, ey2 = e2.y;

                float intx2 = ex1 * ex1 + ex2 * ex1 + ex2 * ex2;
                float inty2 = ey1 * ey1 + ey2 * ey1 + ey2 * ey2;

                I += (0.25f * k_inv3 * D) * (intx2 + inty2);
            }

            // Total mass
            massData.mass = density * area;

            // Center of mass
            b2Assert(area > b2_epsilon);
            center *= 1.0f / area;
            massData.center = center + s;

            // Inertia tensor relative to the local origin (point s).
            massData.I = density * I;

            // Shift to center of mass then to original body origin.
            massData.I += massData.mass * (b2Dot(massData.center, massData.center) - b2Dot(center, center));
        }

        /// <summary>Whether the polygon is convex (every vertex on the inner side of every edge); a debugging aid.</summary>
        /// <returns>True for a convex polygon.</returns>
        internal bool Validate() {
            for (int i = 0; i < m_count; ++i) {
                int i1 = i;
                int i2 = i < m_count - 1 ? i1 + 1 : 0;
                b2Vec2 p = m_vertices[i1];
                b2Vec2 e = m_vertices[i2] - p;

                for (int j = 0; j < m_count; ++j) {
                    if (j == i1 || j == i2)
                        continue;

                    b2Vec2 v = m_vertices[j] - p;
                    float c = b2Cross(e, v);
                    if (c < 0.0f)
                        return false;
                }
            }

            return true;
        }
    }
}
