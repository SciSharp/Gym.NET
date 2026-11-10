/*
* Copyright (c) 2007-2009 Erin Catto http://www.box2d.org
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
// Altered source version (Gym.NET, 2026-11-10): Box2D/Collision/b2Distance.h and b2Distance.cpp of pybox2d 2.3.10's
// bundled Box2D 2.3.0, translated to C#. The proxy copies a circle's center or an edge's two vertices into its inline
// buffer (C++ points into the shape, which is equivalent as shapes don't change during a query) and references a
// polygon's vertex array; the simplex's three vertices are an inline array indexed as C++'s '&m_v1 + i'; the
// statistics counters (b2_gjkCalls, b2_gjkIters, b2_gjkMaxIters) are left out, as nothing reads them and they would be
// shared mutable state across the threads of vector envs. b2Distance is b2DistanceGJK.b2Distance, since a C# type named
// b2Distance would hide b2Math.b2Distance.
using System.Runtime.CompilerServices;
using static Gym.Box2D.b2Math;
using static Gym.Box2D.b2Settings;

namespace Gym.Box2D {
    /// <summary>
    ///     A convex shape as the GJK algorithm sees it: its vertices and radius (a circle is one vertex with a radius, an
    ///     edge two vertices with the skin radius).
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Collision/b2Distance.h#L28">Box2D/Collision/b2Distance.h</see>.</remarks>
    internal struct b2DistanceProxy {
        /// <summary>The vertices of a circle or an edge, copied in.</summary>
        public b2Vec2Pair m_buffer;

        /// <summary>A polygon's vertex array, or null when the vertices are in <see cref="m_buffer"/>.</summary>
        public b2Vec2[]? m_array;

        /// <summary>The number of vertices.</summary>
        public int m_count;

        /// <summary>The radius.</summary>
        public float m_radius;

        /// <summary>Initializes the proxy from a shape's child.</summary>
        /// <param name="shape">The shape.</param>
        /// <param name="index">The child index (only chains use it).</param>
        /// <exception cref="NumSharp.AssertionError">Thrown for a chain shape, which this port does not include.</exception>
        public void Set(b2Shape shape, int index) {
            switch (shape.GetShapeType()) {
                case b2ShapeType.e_circle: {
                    var circle = (b2CircleShape)shape;
                    m_buffer[0] = circle.m_p;
                    m_array = null;
                    m_count = 1;
                    m_radius = circle.m_radius;
                }
                    break;

                case b2ShapeType.e_polygon: {
                    var polygon = (b2PolygonShape)shape;
                    m_array = polygon.m_vertices;
                    m_count = polygon.m_count;
                    m_radius = polygon.m_radius;
                }
                    break;

                case b2ShapeType.e_edge: {
                    var edge = (b2EdgeShape)shape;
                    // C++ points at m_vertex1, whose neighbour in memory is m_vertex2.
                    m_buffer[0] = edge.m_vertex1;
                    m_buffer[1] = edge.m_vertex2;
                    m_array = null;
                    m_count = 2;
                    m_radius = edge.m_radius;
                }
                    break;

                default:
                    b2Assert(false);
                    break;
            }
        }

        /// <summary>The index of the vertex furthest along a direction (the first one on ties).</summary>
        /// <param name="d">The direction.</param>
        /// <returns>The vertex index.</returns>
        public readonly int GetSupport(in b2Vec2 d) {
            int bestIndex = 0;
            float bestValue = b2Dot(Vertex(0), d);
            for (int i = 1; i < m_count; ++i) {
                float value = b2Dot(Vertex(i), d);
                if (value > bestValue) {
                    bestIndex = i;
                    bestValue = value;
                }
            }

            return bestIndex;
        }

        /// <summary>The vertex furthest along a direction.</summary>
        /// <param name="d">The direction.</param>
        /// <returns>The vertex.</returns>
        public readonly b2Vec2 GetSupportVertex(in b2Vec2 d) => Vertex(GetSupport(d));

        /// <summary>The vertex count.</summary>
        /// <returns>The count.</returns>
        public readonly int GetVertexCount() => m_count;

        /// <summary>A vertex by index, checked.</summary>
        /// <param name="index">The index.</param>
        /// <returns>The vertex.</returns>
        /// <exception cref="NumSharp.AssertionError">Thrown for an index outside the vertex count.</exception>
        public readonly b2Vec2 GetVertex(int index) {
            b2Assert(0 <= index && index < m_count);
            return Vertex(index);
        }

        /// <summary>A vertex by index, unchecked, from whichever storage holds them.</summary>
        /// <param name="index">The index.</param>
        /// <returns>The vertex.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private readonly b2Vec2 Vertex(int index) => m_array != null ? m_array[index] : m_buffer[index];
    }

    /// <summary>Three bytes, inline (C++ <c>uint8[3]</c>).</summary>
    [InlineArray(3)]
    internal struct b2Byte3 {
        /// <summary>The first element; the runtime lays the others after it.</summary>
        private byte _element0;
    }

    /// <summary>
    ///     The simplex of a previous GJK call, to warm-start the next one between the same proxies (time of impact calls it
    ///     repeatedly). Set <see cref="count"/> to 0 on the first call.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Collision/b2Distance.h#L62">Box2D/Collision/b2Distance.h</see>.</remarks>
    internal struct b2SimplexCache {
        /// <summary>The simplex's length or area, to detect a stale cache.</summary>
        public float metric;

        /// <summary>The number of cached vertices.</summary>
        public ushort count;

        /// <summary>The vertices on shape A.</summary>
        public b2Byte3 indexA;

        /// <summary>The vertices on shape B.</summary>
        public b2Byte3 indexB;
    }

    /// <summary>The input of a distance query: two proxies and their transforms.</summary>
    internal struct b2DistanceInput {
        /// <summary>The first proxy.</summary>
        public b2DistanceProxy proxyA;

        /// <summary>The second proxy.</summary>
        public b2DistanceProxy proxyB;

        /// <summary>The first proxy's transform.</summary>
        public b2Transform transformA;

        /// <summary>The second proxy's transform.</summary>
        public b2Transform transformB;

        /// <summary>Whether to account for the proxies' radii.</summary>
        public bool useRadii;
    }

    /// <summary>The output of a distance query: the closest points and their distance.</summary>
    internal struct b2DistanceOutput {
        /// <summary>The closest point on shape A.</summary>
        public b2Vec2 pointA;

        /// <summary>The closest point on shape B.</summary>
        public b2Vec2 pointB;

        /// <summary>The distance between them.</summary>
        public float distance;

        /// <summary>The number of GJK iterations used.</summary>
        public int iterations;
    }

    /// <summary>A vertex of the GJK simplex: support points on both shapes, their difference and its barycentric weight.</summary>
    internal struct b2SimplexVertex {
        /// <summary>The support point in proxy A.</summary>
        public b2Vec2 wA;

        /// <summary>The support point in proxy B.</summary>
        public b2Vec2 wB;

        /// <summary><c>wB - wA</c>.</summary>
        public b2Vec2 w;

        /// <summary>The barycentric coordinate for the closest point.</summary>
        public float a;

        /// <summary>The index of <see cref="wA"/>.</summary>
        public int indexA;

        /// <summary>The index of <see cref="wB"/>.</summary>
        public int indexB;
    }

    /// <summary>The three simplex vertices, inline, as C++'s consecutive <c>m_v1, m_v2, m_v3</c>.</summary>
    [InlineArray(3)]
    internal struct b2SimplexVertex3 {
        /// <summary>The first element (C++ <c>m_v1</c>); the runtime lays the others after it.</summary>
        private b2SimplexVertex _element0;
    }

    /// <summary>The GJK simplex: one to three vertices of the Minkowski difference and the solver for the closest point.</summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Collision/b2Distance.cpp#L80">Box2D/Collision/b2Distance.cpp</see>.</remarks>
    internal struct b2Simplex {
        /// <summary>The vertices; <c>m_v[0]</c> is C++'s <c>m_v1</c>.</summary>
        public b2SimplexVertex3 m_v;

        /// <summary>The number of vertices in use.</summary>
        public int m_count;

        /// <summary>Loads the simplex from a cache, flushing it when its metric changed substantially, or starts from vertex 0.</summary>
        /// <param name="cache">The cache.</param>
        /// <param name="proxyA">Proxy A.</param>
        /// <param name="transformA">Proxy A's transform.</param>
        /// <param name="proxyB">Proxy B.</param>
        /// <param name="transformB">Proxy B's transform.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown for a cache of more than 3 vertices or indices out of range.</exception>
        public void ReadCache(in b2SimplexCache cache, in b2DistanceProxy proxyA, in b2Transform transformA, in b2DistanceProxy proxyB, in b2Transform transformB) {
            b2Assert(cache.count <= 3, "cache->count <= 3");

            // Copy data from cache.
            m_count = cache.count;
            for (int i = 0; i < m_count; ++i) {
                ref b2SimplexVertex v = ref m_v[i];
                v.indexA = cache.indexA[i];
                v.indexB = cache.indexB[i];
                b2Vec2 wALocal = proxyA.GetVertex(v.indexA);
                b2Vec2 wBLocal = proxyB.GetVertex(v.indexB);
                v.wA = b2Mul(transformA, wALocal);
                v.wB = b2Mul(transformB, wBLocal);
                v.w = v.wB - v.wA;
                v.a = 0.0f;
            }

            // Compute the new simplex metric, if it is substantially different than
            // old metric then flush the simplex.
            if (m_count > 1) {
                float metric1 = cache.metric;
                float metric2 = GetMetric();
                if (metric2 < 0.5f * metric1 || 2.0f * metric1 < metric2 || metric2 < b2_epsilon)
                    // Reset the simplex.
                    m_count = 0;
            }

            // If the cache is empty or invalid ...
            if (m_count == 0) {
                ref b2SimplexVertex v = ref m_v[0];
                v.indexA = 0;
                v.indexB = 0;
                b2Vec2 wALocal = proxyA.GetVertex(0);
                b2Vec2 wBLocal = proxyB.GetVertex(0);
                v.wA = b2Mul(transformA, wALocal);
                v.wB = b2Mul(transformB, wBLocal);
                v.w = v.wB - v.wA;
                v.a = 1.0f;
                m_count = 1;
            }
        }

        /// <summary>Stores the simplex in a cache for the next call.</summary>
        /// <param name="cache">The cache.</param>
        public readonly void WriteCache(ref b2SimplexCache cache) {
            cache.metric = GetMetric();
            cache.count = (ushort)m_count;
            for (int i = 0; i < m_count; ++i) {
                cache.indexA[i] = (byte)m_v[i].indexA;
                cache.indexB[i] = (byte)m_v[i].indexB;
            }
        }

        /// <summary>The direction towards the origin from the simplex.</summary>
        /// <returns>The search direction.</returns>
        /// <exception cref="NumSharp.AssertionError">Thrown for a simplex of other than 1 or 2 vertices.</exception>
        public readonly b2Vec2 GetSearchDirection() {
            switch (m_count) {
                case 1:
                    return -m_v[0].w;

                case 2: {
                    b2Vec2 e12 = m_v[1].w - m_v[0].w;
                    float sgn = b2Cross(e12, -m_v[0].w);
                    if (sgn > 0.0f)
                        // Origin is left of e12.
                        return b2Cross(1.0f, e12);
                    else
                        // Origin is right of e12.
                        return b2Cross(e12, 1.0f);
                }

                default:
                    b2Assert(false);
                    return b2Vec2_zero;
            }
        }

        /// <summary>The simplex's closest point to the origin.</summary>
        /// <returns>The point (zero for a triangle).</returns>
        /// <exception cref="NumSharp.AssertionError">Thrown for an empty simplex.</exception>
        public readonly b2Vec2 GetClosestPoint() {
            switch (m_count) {
                case 0:
                    b2Assert(false);
                    return b2Vec2_zero;

                case 1:
                    return m_v[0].w;

                case 2:
                    return m_v[0].a * m_v[0].w + m_v[1].a * m_v[1].w;

                case 3:
                    return b2Vec2_zero;

                default:
                    b2Assert(false);
                    return b2Vec2_zero;
            }
        }

        /// <summary>The witness points on both shapes, by barycentric weights.</summary>
        /// <param name="pA">Receives the point on shape A.</param>
        /// <param name="pB">Receives the point on shape B.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown for an empty simplex.</exception>
        public readonly void GetWitnessPoints(ref b2Vec2 pA, ref b2Vec2 pB) {
            switch (m_count) {
                case 0:
                    b2Assert(false);
                    break;

                case 1:
                    pA = m_v[0].wA;
                    pB = m_v[0].wB;
                    break;

                case 2:
                    pA = m_v[0].a * m_v[0].wA + m_v[1].a * m_v[1].wA;
                    pB = m_v[0].a * m_v[0].wB + m_v[1].a * m_v[1].wB;
                    break;

                case 3:
                    pA = m_v[0].a * m_v[0].wA + m_v[1].a * m_v[1].wA + m_v[2].a * m_v[2].wA;
                    pB = pA;
                    break;

                default:
                    b2Assert(false);
                    break;
            }
        }

        /// <summary>The simplex's size: 0 for a point, the length of a segment, the signed area of a triangle.</summary>
        /// <returns>The metric.</returns>
        /// <exception cref="NumSharp.AssertionError">Thrown for an empty simplex.</exception>
        public readonly float GetMetric() {
            switch (m_count) {
                case 0:
                    b2Assert(false);
                    return 0.0f;

                case 1:
                    return 0.0f;

                case 2:
                    return b2Distance(m_v[0].w, m_v[1].w);

                case 3:
                    return b2Cross(m_v[1].w - m_v[0].w, m_v[2].w - m_v[0].w);

                default:
                    b2Assert(false);
                    return 0.0f;
            }
        }

        /// <summary>
        ///     Solves a segment by barycentric coordinates: <c>a1 = d12_1 / d12</c>, <c>a2 = d12_2 / d12</c>, dropping to a
        ///     vertex when the origin lies beyond an end.
        /// </summary>
        public void Solve2() {
            b2Vec2 w1 = m_v[0].w;
            b2Vec2 w2 = m_v[1].w;
            b2Vec2 e12 = w2 - w1;

            // w1 region
            float d12_2 = -b2Dot(w1, e12);
            if (d12_2 <= 0.0f) {
                // a2 <= 0, so we clamp it to 0
                m_v[0].a = 1.0f;
                m_count = 1;
                return;
            }

            // w2 region
            float d12_1 = b2Dot(w2, e12);
            if (d12_1 <= 0.0f) {
                // a1 <= 0, so we clamp it to 0
                m_v[1].a = 1.0f;
                m_count = 1;
                m_v[0] = m_v[1];
                return;
            }

            // Must be in e12 region.
            float inv_d12 = 1.0f / (d12_1 + d12_2);
            m_v[0].a = d12_1 * inv_d12;
            m_v[1].a = d12_2 * inv_d12;
            m_count = 2;
        }

        /// <summary>Solves a triangle: the origin's Voronoi region picks a vertex, an edge, or the interior.</summary>
        public void Solve3() {
            b2Vec2 w1 = m_v[0].w;
            b2Vec2 w2 = m_v[1].w;
            b2Vec2 w3 = m_v[2].w;

            // Edge12
            // [1      1     ][a1] = [1]
            // [w1.e12 w2.e12][a2] = [0]
            // a3 = 0
            b2Vec2 e12 = w2 - w1;
            float w1e12 = b2Dot(w1, e12);
            float w2e12 = b2Dot(w2, e12);
            float d12_1 = w2e12;
            float d12_2 = -w1e12;

            // Edge13
            // [1      1     ][a1] = [1]
            // [w1.e13 w3.e13][a3] = [0]
            // a2 = 0
            b2Vec2 e13 = w3 - w1;
            float w1e13 = b2Dot(w1, e13);
            float w3e13 = b2Dot(w3, e13);
            float d13_1 = w3e13;
            float d13_2 = -w1e13;

            // Edge23
            // [1      1     ][a2] = [1]
            // [w2.e23 w3.e23][a3] = [0]
            // a1 = 0
            b2Vec2 e23 = w3 - w2;
            float w2e23 = b2Dot(w2, e23);
            float w3e23 = b2Dot(w3, e23);
            float d23_1 = w3e23;
            float d23_2 = -w2e23;

            // Triangle123
            float n123 = b2Cross(e12, e13);

            float d123_1 = n123 * b2Cross(w2, w3);
            float d123_2 = n123 * b2Cross(w3, w1);
            float d123_3 = n123 * b2Cross(w1, w2);

            // w1 region
            if (d12_2 <= 0.0f && d13_2 <= 0.0f) {
                m_v[0].a = 1.0f;
                m_count = 1;
                return;
            }

            // e12
            if (d12_1 > 0.0f && d12_2 > 0.0f && d123_3 <= 0.0f) {
                float inv_d12 = 1.0f / (d12_1 + d12_2);
                m_v[0].a = d12_1 * inv_d12;
                m_v[1].a = d12_2 * inv_d12;
                m_count = 2;
                return;
            }

            // e13
            if (d13_1 > 0.0f && d13_2 > 0.0f && d123_2 <= 0.0f) {
                float inv_d13 = 1.0f / (d13_1 + d13_2);
                m_v[0].a = d13_1 * inv_d13;
                m_v[2].a = d13_2 * inv_d13;
                m_count = 2;
                m_v[1] = m_v[2];
                return;
            }

            // w2 region
            if (d12_1 <= 0.0f && d23_2 <= 0.0f) {
                m_v[1].a = 1.0f;
                m_count = 1;
                m_v[0] = m_v[1];
                return;
            }

            // w3 region
            if (d13_1 <= 0.0f && d23_1 <= 0.0f) {
                m_v[2].a = 1.0f;
                m_count = 1;
                m_v[0] = m_v[2];
                return;
            }

            // e23
            if (d23_1 > 0.0f && d23_2 > 0.0f && d123_1 <= 0.0f) {
                float inv_d23 = 1.0f / (d23_1 + d23_2);
                m_v[1].a = d23_1 * inv_d23;
                m_v[2].a = d23_2 * inv_d23;
                m_count = 2;
                m_v[0] = m_v[2];
                return;
            }

            // Must be in triangle123
            float inv_d123 = 1.0f / (d123_1 + d123_2 + d123_3);
            m_v[0].a = d123_1 * inv_d123;
            m_v[1].a = d123_2 * inv_d123;
            m_v[2].a = d123_3 * inv_d123;
            m_count = 3;
        }
    }

    /// <summary>
    ///     GJK distance (Voronoi regions after Christer Ericson, barycentric coordinates): the closest points of two convex
    ///     proxies, warm-started by a <see cref="b2SimplexCache"/>.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Collision/b2Distance.cpp#L451">Box2D/Collision/b2Distance.cpp</see>.</remarks>
    internal static class b2DistanceGJK {
        /// <summary>
        ///     Computes the closest points between two shapes, at most 20 support-point iterations, stopping at a repeated
        ///     support point or a degenerate search direction; with radii, the points move to the surfaces, or meet halfway
        ///     when the shapes overlap.
        /// </summary>
        /// <param name="output">Receives the closest points and distance.</param>
        /// <param name="cache">The warm-start cache, updated for the next call; set its count to 0 the first time.</param>
        /// <param name="input">The proxies and transforms.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown for a corrupt cache (more than 3 vertices or indices out of range).</exception>
        public static void b2Distance(out b2DistanceOutput output, ref b2SimplexCache cache, in b2DistanceInput input) {
            output = default;

            ref readonly b2DistanceProxy proxyA = ref input.proxyA;
            ref readonly b2DistanceProxy proxyB = ref input.proxyB;

            b2Transform transformA = input.transformA;
            b2Transform transformB = input.transformB;

            // Initialize the simplex.
            b2Simplex simplex = default;
            simplex.ReadCache(cache, proxyA, transformA, proxyB, transformB);

            const int k_maxIters = 20;

            // These store the vertices of the last simplex so that we
            // can check for duplicates and prevent cycling.
            b2Int3 saveA = default, saveB = default;
            int saveCount = 0;

            float distanceSqr1 = b2_maxFloat;
            float distanceSqr2 = distanceSqr1;

            // Main iteration loop.
            int iter = 0;
            while (iter < k_maxIters) {
                // Copy simplex so we can identify duplicates.
                saveCount = simplex.m_count;
                for (int i = 0; i < saveCount; ++i) {
                    saveA[i] = simplex.m_v[i].indexA;
                    saveB[i] = simplex.m_v[i].indexB;
                }

                switch (simplex.m_count) {
                    case 1:
                        break;

                    case 2:
                        simplex.Solve2();
                        break;

                    case 3:
                        simplex.Solve3();
                        break;

                    default:
                        b2Assert(false);
                        break;
                }

                // If we have 3 points, then the origin is in the corresponding triangle.
                if (simplex.m_count == 3)
                    break;

                // Compute closest point.
                b2Vec2 p = simplex.GetClosestPoint();
                distanceSqr2 = p.LengthSquared();

                // Ensure progress (the C++ check is disabled: it only records the distance)
                distanceSqr1 = distanceSqr2;

                // Get search direction.
                b2Vec2 d = simplex.GetSearchDirection();

                // Ensure the search direction is numerically fit.
                if (d.LengthSquared() < b2_epsilon * b2_epsilon)
                    // The origin is probably contained by a line segment
                    // or triangle. Thus the shapes are overlapped.

                    // We can't return zero here even though there may be overlap.
                    // In case the simplex is a point, segment, or triangle it is difficult
                    // to determine if the origin is contained in the CSO or very close to it.
                    break;

                // Compute a tentative new simplex vertex using support points.
                ref b2SimplexVertex vertex = ref simplex.m_v[simplex.m_count];
                vertex.indexA = proxyA.GetSupport(b2MulT(transformA.q, -d));
                vertex.wA = b2Mul(transformA, proxyA.GetVertex(vertex.indexA));
                vertex.indexB = proxyB.GetSupport(b2MulT(transformB.q, d));
                vertex.wB = b2Mul(transformB, proxyB.GetVertex(vertex.indexB));
                vertex.w = vertex.wB - vertex.wA;

                // Iteration count is equated to the number of support point calls.
                ++iter;

                // Check for duplicate support points. This is the main termination criteria.
                bool duplicate = false;
                for (int i = 0; i < saveCount; ++i) {
                    if (vertex.indexA == saveA[i] && vertex.indexB == saveB[i]) {
                        duplicate = true;
                        break;
                    }
                }

                // If we found a duplicate support point we must exit to avoid cycling.
                if (duplicate)
                    break;

                // New vertex is ok and needed.
                ++simplex.m_count;
            }

            // Prepare output.
            simplex.GetWitnessPoints(ref output.pointA, ref output.pointB);
            output.distance = b2Math.b2Distance(output.pointA, output.pointB);
            output.iterations = iter;

            // Cache the simplex.
            simplex.WriteCache(ref cache);

            // Apply radii if requested.
            if (input.useRadii) {
                float rA = proxyA.m_radius;
                float rB = proxyB.m_radius;

                if (output.distance > rA + rB && output.distance > b2_epsilon) {
                    // Shapes are still no overlapped.
                    // Move the witness points to the outer surface.
                    output.distance -= rA + rB;
                    b2Vec2 normal = output.pointB - output.pointA;
                    normal.Normalize();
                    output.pointA += rA * normal;
                    output.pointB -= rB * normal;
                } else {
                    // Shapes are overlapped when radii are considered.
                    // Move the witness points to the middle.
                    b2Vec2 p = 0.5f * (output.pointA + output.pointB);
                    output.pointA = p;
                    output.pointB = p;
                    output.distance = 0.0f;
                }
            }
        }
    }

    /// <summary>Three ints, inline (C++ <c>int32[3]</c>).</summary>
    [InlineArray(3)]
    internal struct b2Int3 {
        /// <summary>The first element; the runtime lays the others after it.</summary>
        private int _element0;
    }
}
