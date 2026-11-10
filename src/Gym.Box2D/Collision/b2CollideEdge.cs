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
// Altered source version (Gym.NET, 2026-11-10): Box2D/Collision/b2CollideEdge.cpp of pybox2d 2.3.10's bundled Box2D
// 2.3.0, translated to C#: edge-circle and edge-polygon manifolds with edge adjacency (ghost vertices). The temporary
// polygon's arrays are inline, as in C++, so a collision allocates nothing.
using System.Runtime.CompilerServices;
using static Gym.Box2D.b2Math;
using static Gym.Box2D.b2Settings;

namespace Gym.Box2D {
    /// <summary>Sixteen vectors, inline (C++ <c>b2Vec2[b2_maxPolygonVertices]</c>).</summary>
    [InlineArray(b2_maxPolygonVertices)]
    internal struct b2Vec2x16 {
        /// <summary>The first element; the runtime lays the others after it.</summary>
        private b2Vec2 _element0;
    }

    /// <summary>The kind of separating axis the edge-polygon collider found (C++ <c>b2EPAxis::Type</c>).</summary>
    internal enum b2EPAxisType {
        /// <summary>No valid axis.</summary>
        e_unknown,

        /// <summary>The edge's normal.</summary>
        e_edgeA,

        /// <summary>A polygon normal.</summary>
        e_edgeB,
    }

    /// <summary>The best separating axis of an edge-polygon pair.</summary>
    internal struct b2EPAxis {
        /// <summary>The axis kind.</summary>
        public b2EPAxisType type;

        /// <summary>The axis's edge index.</summary>
        public int index;

        /// <summary>The separation along it.</summary>
        public float separation;
    }

    /// <summary>Polygon B expressed in the edge's frame.</summary>
    internal struct b2TempPolygon {
        /// <summary>The vertices.</summary>
        public b2Vec2x16 vertices;

        /// <summary>The normals.</summary>
        public b2Vec2x16 normals;

        /// <summary>The vertex count.</summary>
        public int count;
    }

    /// <summary>The reference face used for clipping.</summary>
    internal struct b2ReferenceFace {
        /// <summary>The face's first vertex index.</summary>
        public int i1;

        /// <summary>The face's second vertex index.</summary>
        public int i2;

        /// <summary>The face's first vertex.</summary>
        public b2Vec2 v1;

        /// <summary>The face's second vertex.</summary>
        public b2Vec2 v2;

        /// <summary>The face normal.</summary>
        public b2Vec2 normal;

        /// <summary>The first side plane's normal.</summary>
        public b2Vec2 sideNormal1;

        /// <summary>The first side plane's offset.</summary>
        public float sideOffset1;

        /// <summary>The second side plane's normal.</summary>
        public b2Vec2 sideNormal2;

        /// <summary>The second side plane's offset.</summary>
        public float sideOffset2;
    }

    /// <summary>How an edge vertex is joined to its neighbour (C++ <c>b2EPCollider::VertexType</c>).</summary>
    internal enum b2EPVertexType {
        /// <summary>No neighbour.</summary>
        e_isolated,

        /// <summary>A concave joint.</summary>
        e_concave,

        /// <summary>A convex joint.</summary>
        e_convex,
    }

    /// <summary>
    ///     Collides an edge and a polygon taking edge adjacency into account: classify the edge's vertices, pick the front
    ///     or back face, limit the collision normal by the neighbouring edges, search the axes, then clip.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Collision/b2CollideEdge.cpp#L175">Box2D/Collision/b2CollideEdge.cpp</see>.</remarks>
    internal struct b2EPCollider {
        /// <summary>Polygon B in the edge's frame.</summary>
        public b2TempPolygon m_polygonB;

        /// <summary>B's frame relative to A's.</summary>
        public b2Transform m_xf;

        /// <summary>B's centroid in A's frame.</summary>
        public b2Vec2 m_centroidB;

        /// <summary>The ghost vertex before the edge.</summary>
        public b2Vec2 m_v0;

        /// <summary>The edge's first vertex.</summary>
        public b2Vec2 m_v1;

        /// <summary>The edge's second vertex.</summary>
        public b2Vec2 m_v2;

        /// <summary>The ghost vertex after the edge.</summary>
        public b2Vec2 m_v3;

        /// <summary>The previous edge's normal.</summary>
        public b2Vec2 m_normal0;

        /// <summary>The edge's normal.</summary>
        public b2Vec2 m_normal1;

        /// <summary>The next edge's normal.</summary>
        public b2Vec2 m_normal2;

        /// <summary>The collision normal (front or back of the edge).</summary>
        public b2Vec2 m_normal;

        /// <summary>The first vertex's joint type (unused by the algorithm, kept from C++).</summary>
        public b2EPVertexType m_type1;

        /// <summary>The second vertex's joint type (unused by the algorithm, kept from C++).</summary>
        public b2EPVertexType m_type2;

        /// <summary>The lower limit of admissible normals.</summary>
        public b2Vec2 m_lowerLimit;

        /// <summary>The upper limit of admissible normals.</summary>
        public b2Vec2 m_upperLimit;

        /// <summary>The combined skin radius, <c>2.0f * b2_polygonRadius</c>.</summary>
        public float m_radius;

        /// <summary>Whether the polygon is in front of the edge.</summary>
        public bool m_front;

        /// <summary>Computes the manifold of the edge and the polygon.</summary>
        /// <param name="manifold">Receives the manifold.</param>
        /// <param name="edgeA">The edge.</param>
        /// <param name="xfA">The edge's transform.</param>
        /// <param name="polygonB">The polygon.</param>
        /// <param name="xfB">The polygon's transform.</param>
        public void Collide(ref b2Manifold manifold, b2EdgeShape edgeA, in b2Transform xfA, b2PolygonShape polygonB, in b2Transform xfB) {
            m_xf = b2MulT(xfA, xfB);

            m_centroidB = b2Mul(m_xf, polygonB.m_centroid);

            m_v0 = edgeA.m_vertex0;
            m_v1 = edgeA.m_vertex1;
            m_v2 = edgeA.m_vertex2;
            m_v3 = edgeA.m_vertex3;

            bool hasVertex0 = edgeA.m_hasVertex0;
            bool hasVertex3 = edgeA.m_hasVertex3;

            b2Vec2 edge1 = m_v2 - m_v1;
            edge1.Normalize();
            m_normal1.Set(edge1.y, -edge1.x);
            float offset1 = b2Dot(m_normal1, m_centroidB - m_v1);
            float offset0 = 0.0f, offset2 = 0.0f;
            bool convex1 = false, convex2 = false;

            // Is there a preceding edge?
            if (hasVertex0) {
                b2Vec2 edge0 = m_v1 - m_v0;
                edge0.Normalize();
                m_normal0.Set(edge0.y, -edge0.x);
                convex1 = b2Cross(edge0, edge1) >= 0.0f;
                offset0 = b2Dot(m_normal0, m_centroidB - m_v0);
            }

            // Is there a following edge?
            if (hasVertex3) {
                b2Vec2 edge2 = m_v3 - m_v2;
                edge2.Normalize();
                m_normal2.Set(edge2.y, -edge2.x);
                convex2 = b2Cross(edge1, edge2) > 0.0f;
                offset2 = b2Dot(m_normal2, m_centroidB - m_v2);
            }

            // Determine front or back collision. Determine collision normal limits.
            if (hasVertex0 && hasVertex3) {
                if (convex1 && convex2) {
                    m_front = offset0 >= 0.0f || offset1 >= 0.0f || offset2 >= 0.0f;
                    if (m_front) {
                        m_normal = m_normal1;
                        m_lowerLimit = m_normal0;
                        m_upperLimit = m_normal2;
                    } else {
                        m_normal = -m_normal1;
                        m_lowerLimit = -m_normal1;
                        m_upperLimit = -m_normal1;
                    }
                } else if (convex1) {
                    m_front = offset0 >= 0.0f || (offset1 >= 0.0f && offset2 >= 0.0f);
                    if (m_front) {
                        m_normal = m_normal1;
                        m_lowerLimit = m_normal0;
                        m_upperLimit = m_normal1;
                    } else {
                        m_normal = -m_normal1;
                        m_lowerLimit = -m_normal2;
                        m_upperLimit = -m_normal1;
                    }
                } else if (convex2) {
                    m_front = offset2 >= 0.0f || (offset0 >= 0.0f && offset1 >= 0.0f);
                    if (m_front) {
                        m_normal = m_normal1;
                        m_lowerLimit = m_normal1;
                        m_upperLimit = m_normal2;
                    } else {
                        m_normal = -m_normal1;
                        m_lowerLimit = -m_normal1;
                        m_upperLimit = -m_normal0;
                    }
                } else {
                    m_front = offset0 >= 0.0f && offset1 >= 0.0f && offset2 >= 0.0f;
                    if (m_front) {
                        m_normal = m_normal1;
                        m_lowerLimit = m_normal1;
                        m_upperLimit = m_normal1;
                    } else {
                        m_normal = -m_normal1;
                        m_lowerLimit = -m_normal2;
                        m_upperLimit = -m_normal0;
                    }
                }
            } else if (hasVertex0) {
                if (convex1) {
                    m_front = offset0 >= 0.0f || offset1 >= 0.0f;
                    if (m_front) {
                        m_normal = m_normal1;
                        m_lowerLimit = m_normal0;
                        m_upperLimit = -m_normal1;
                    } else {
                        m_normal = -m_normal1;
                        m_lowerLimit = m_normal1;
                        m_upperLimit = -m_normal1;
                    }
                } else {
                    m_front = offset0 >= 0.0f && offset1 >= 0.0f;
                    if (m_front) {
                        m_normal = m_normal1;
                        m_lowerLimit = m_normal1;
                        m_upperLimit = -m_normal1;
                    } else {
                        m_normal = -m_normal1;
                        m_lowerLimit = m_normal1;
                        m_upperLimit = -m_normal0;
                    }
                }
            } else if (hasVertex3) {
                if (convex2) {
                    m_front = offset1 >= 0.0f || offset2 >= 0.0f;
                    if (m_front) {
                        m_normal = m_normal1;
                        m_lowerLimit = -m_normal1;
                        m_upperLimit = m_normal2;
                    } else {
                        m_normal = -m_normal1;
                        m_lowerLimit = -m_normal1;
                        m_upperLimit = m_normal1;
                    }
                } else {
                    m_front = offset1 >= 0.0f && offset2 >= 0.0f;
                    if (m_front) {
                        m_normal = m_normal1;
                        m_lowerLimit = -m_normal1;
                        m_upperLimit = m_normal1;
                    } else {
                        m_normal = -m_normal1;
                        m_lowerLimit = -m_normal2;
                        m_upperLimit = m_normal1;
                    }
                }
            } else {
                m_front = offset1 >= 0.0f;
                if (m_front) {
                    m_normal = m_normal1;
                    m_lowerLimit = -m_normal1;
                    m_upperLimit = -m_normal1;
                } else {
                    m_normal = -m_normal1;
                    m_lowerLimit = m_normal1;
                    m_upperLimit = m_normal1;
                }
            }

            // Get polygonB in frameA
            m_polygonB.count = polygonB.m_count;
            for (int i = 0; i < polygonB.m_count; ++i) {
                m_polygonB.vertices[i] = b2Mul(m_xf, polygonB.m_vertices[i]);
                m_polygonB.normals[i] = b2Mul(m_xf.q, polygonB.m_normals[i]);
            }

            m_radius = 2.0f * b2_polygonRadius;

            manifold.pointCount = 0;

            b2EPAxis edgeAxis = ComputeEdgeSeparation();

            // If no valid normal can be found than this edge should not collide.
            if (edgeAxis.type == b2EPAxisType.e_unknown)
                return;

            if (edgeAxis.separation > m_radius)
                return;

            b2EPAxis polygonAxis = ComputePolygonSeparation();
            if (polygonAxis.type != b2EPAxisType.e_unknown && polygonAxis.separation > m_radius)
                return;

            // Use hysteresis for jitter reduction.
            const float k_relativeTol = 0.98f;
            const float k_absoluteTol = 0.001f;

            b2EPAxis primaryAxis;
            if (polygonAxis.type == b2EPAxisType.e_unknown)
                primaryAxis = edgeAxis;
            else if (polygonAxis.separation > k_relativeTol * edgeAxis.separation + k_absoluteTol)
                primaryAxis = polygonAxis;
            else
                primaryAxis = edgeAxis;

            b2ClipVertex2 ie = default;
            b2ReferenceFace rf;
            if (primaryAxis.type == b2EPAxisType.e_edgeA) {
                manifold.type = b2ManifoldType.e_faceA;

                // Search for the polygon normal that is most anti-parallel to the edge normal.
                int bestIndex = 0;
                float bestValue = b2Dot(m_normal, m_polygonB.normals[0]);
                for (int i = 1; i < m_polygonB.count; ++i) {
                    float value = b2Dot(m_normal, m_polygonB.normals[i]);
                    if (value < bestValue) {
                        bestValue = value;
                        bestIndex = i;
                    }
                }

                int i1 = bestIndex;
                int i2 = i1 + 1 < m_polygonB.count ? i1 + 1 : 0;

                ie[0].v = m_polygonB.vertices[i1];
                ie[0].id.cf.indexA = 0;
                ie[0].id.cf.indexB = (byte)i1;
                ie[0].id.cf.typeA = b2ContactFeature.e_face;
                ie[0].id.cf.typeB = b2ContactFeature.e_vertex;

                ie[1].v = m_polygonB.vertices[i2];
                ie[1].id.cf.indexA = 0;
                ie[1].id.cf.indexB = (byte)i2;
                ie[1].id.cf.typeA = b2ContactFeature.e_face;
                ie[1].id.cf.typeB = b2ContactFeature.e_vertex;

                if (m_front) {
                    rf.i1 = 0;
                    rf.i2 = 1;
                    rf.v1 = m_v1;
                    rf.v2 = m_v2;
                    rf.normal = m_normal1;
                } else {
                    rf.i1 = 1;
                    rf.i2 = 0;
                    rf.v1 = m_v2;
                    rf.v2 = m_v1;
                    rf.normal = -m_normal1;
                }
            } else {
                manifold.type = b2ManifoldType.e_faceB;

                ie[0].v = m_v1;
                ie[0].id.cf.indexA = 0;
                ie[0].id.cf.indexB = (byte)primaryAxis.index;
                ie[0].id.cf.typeA = b2ContactFeature.e_vertex;
                ie[0].id.cf.typeB = b2ContactFeature.e_face;

                ie[1].v = m_v2;
                ie[1].id.cf.indexA = 0;
                ie[1].id.cf.indexB = (byte)primaryAxis.index;
                ie[1].id.cf.typeA = b2ContactFeature.e_vertex;
                ie[1].id.cf.typeB = b2ContactFeature.e_face;

                rf.i1 = primaryAxis.index;
                rf.i2 = rf.i1 + 1 < m_polygonB.count ? rf.i1 + 1 : 0;
                rf.v1 = m_polygonB.vertices[rf.i1];
                rf.v2 = m_polygonB.vertices[rf.i2];
                rf.normal = m_polygonB.normals[rf.i1];
            }

            rf.sideNormal1 = new b2Vec2(rf.normal.y, -rf.normal.x);
            rf.sideNormal2 = -rf.sideNormal1;
            rf.sideOffset1 = b2Dot(rf.sideNormal1, rf.v1);
            rf.sideOffset2 = b2Dot(rf.sideNormal2, rf.v2);

            // Clip incident edge against extruded edge1 side edges.
            b2ClipVertex2 clipPoints1 = default;
            b2ClipVertex2 clipPoints2 = default;
            int np;

            // Clip to box side 1
            np = b2Collision.b2ClipSegmentToLine(ref clipPoints1, ie, rf.sideNormal1, rf.sideOffset1, rf.i1);

            if (np < b2_maxManifoldPoints)
                return;

            // Clip to negative box side 1
            np = b2Collision.b2ClipSegmentToLine(ref clipPoints2, clipPoints1, rf.sideNormal2, rf.sideOffset2, rf.i2);

            if (np < b2_maxManifoldPoints)
                return;

            // Now clipPoints2 contains the clipped points.
            if (primaryAxis.type == b2EPAxisType.e_edgeA) {
                manifold.localNormal = rf.normal;
                manifold.localPoint = rf.v1;
            } else {
                manifold.localNormal = polygonB.m_normals[rf.i1];
                manifold.localPoint = polygonB.m_vertices[rf.i1];
            }

            int pointCount = 0;
            for (int i = 0; i < b2_maxManifoldPoints; ++i) {
                float separation;

                separation = b2Dot(rf.normal, clipPoints2[i].v - rf.v1);

                if (separation <= m_radius) {
                    ref b2ManifoldPoint cp = ref manifold.points[pointCount];

                    if (primaryAxis.type == b2EPAxisType.e_edgeA) {
                        cp.localPoint = b2MulT(m_xf, clipPoints2[i].v);
                        cp.id = clipPoints2[i].id;
                    } else {
                        cp.localPoint = clipPoints2[i].v;
                        cp.id.cf.typeA = clipPoints2[i].id.cf.typeB;
                        cp.id.cf.typeB = clipPoints2[i].id.cf.typeA;
                        cp.id.cf.indexA = clipPoints2[i].id.cf.indexB;
                        cp.id.cf.indexB = clipPoints2[i].id.cf.indexA;
                    }

                    ++pointCount;
                }
            }

            manifold.pointCount = pointCount;
        }

        /// <summary>The separation of the polygon's vertices along the edge normal.</summary>
        /// <returns>The edge axis.</returns>
        public readonly b2EPAxis ComputeEdgeSeparation() {
            b2EPAxis axis;
            axis.type = b2EPAxisType.e_edgeA;
            axis.index = m_front ? 0 : 1;
            axis.separation = float.MaxValue;

            for (int i = 0; i < m_polygonB.count; ++i) {
                float s = b2Dot(m_normal, m_polygonB.vertices[i] - m_v1);
                if (s < axis.separation)
                    axis.separation = s;
            }

            return axis;
        }

        /// <summary>The best polygon normal within the adjacency limits, or a separating one found early.</summary>
        /// <returns>The polygon axis, or an unknown axis when none is admissible.</returns>
        public readonly b2EPAxis ComputePolygonSeparation() {
            b2EPAxis axis;
            axis.type = b2EPAxisType.e_unknown;
            axis.index = -1;
            axis.separation = -float.MaxValue;

            b2Vec2 perp = new b2Vec2(-m_normal.y, m_normal.x);

            for (int i = 0; i < m_polygonB.count; ++i) {
                b2Vec2 n = -m_polygonB.normals[i];

                float s1 = b2Dot(n, m_polygonB.vertices[i] - m_v1);
                float s2 = b2Dot(n, m_polygonB.vertices[i] - m_v2);
                float s = b2Min(s1, s2);

                if (s > m_radius) {
                    // No collision
                    axis.type = b2EPAxisType.e_edgeB;
                    axis.index = i;
                    axis.separation = s;
                    return axis;
                }

                // Adjacency
                if (b2Dot(n, perp) >= 0.0f) {
                    if (b2Dot(n - m_upperLimit, m_normal) < -b2_angularSlop)
                        continue;
                } else {
                    if (b2Dot(n - m_lowerLimit, m_normal) < -b2_angularSlop)
                        continue;
                }

                if (s > axis.separation) {
                    axis.type = b2EPAxisType.e_edgeB;
                    axis.index = i;
                    axis.separation = s;
                }
            }

            return axis;
        }
    }

    internal static partial class b2Collision {
        /// <summary>
        ///     The manifold of an edge and a circle, accounting for edge connectivity: a circle in the region of a vertex
        ///     shared with a neighbouring edge is left to that edge.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Collision/b2CollideEdge.cpp#L27">Box2D/Collision/b2CollideEdge.cpp</see>.</remarks>
        /// <param name="manifold">Receives the manifold.</param>
        /// <param name="edgeA">The edge.</param>
        /// <param name="xfA">The edge's transform.</param>
        /// <param name="circleB">The circle.</param>
        /// <param name="xfB">The circle's transform.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown for a zero-length edge reaching the face region.</exception>
        public static void b2CollideEdgeAndCircle(ref b2Manifold manifold, b2EdgeShape edgeA, in b2Transform xfA, b2CircleShape circleB, in b2Transform xfB) {
            manifold.pointCount = 0;

            // Compute circle in frame of edge
            b2Vec2 Q = b2MulT(xfA, b2Mul(xfB, circleB.m_p));

            b2Vec2 A = edgeA.m_vertex1, B = edgeA.m_vertex2;
            b2Vec2 e = B - A;

            // Barycentric coordinates
            float u = b2Dot(e, B - Q);
            float v = b2Dot(e, Q - A);

            float radius = edgeA.m_radius + circleB.m_radius;

            b2ContactFeature cf;
            cf.indexB = 0;
            cf.typeB = b2ContactFeature.e_vertex;

            // Region A
            if (v <= 0.0f) {
                b2Vec2 P = A;
                b2Vec2 d = Q - P;
                float dd = b2Dot(d, d);
                if (dd > radius * radius)
                    return;

                // Is there an edge connected to A?
                if (edgeA.m_hasVertex0) {
                    b2Vec2 A1 = edgeA.m_vertex0;
                    b2Vec2 B1 = A;
                    b2Vec2 e1 = B1 - A1;
                    float u1 = b2Dot(e1, B1 - Q);

                    // Is the circle in Region AB of the previous edge?
                    if (u1 > 0.0f)
                        return;
                }

                cf.indexA = 0;
                cf.typeA = b2ContactFeature.e_vertex;
                manifold.pointCount = 1;
                manifold.type = b2ManifoldType.e_circles;
                manifold.localNormal.SetZero();
                manifold.localPoint = P;
                manifold.points[0].id.key = 0;
                manifold.points[0].id.cf = cf;
                manifold.points[0].localPoint = circleB.m_p;
                return;
            }

            // Region B
            if (u <= 0.0f) {
                b2Vec2 P = B;
                b2Vec2 d = Q - P;
                float dd = b2Dot(d, d);
                if (dd > radius * radius)
                    return;

                // Is there an edge connected to B?
                if (edgeA.m_hasVertex3) {
                    b2Vec2 B2 = edgeA.m_vertex3;
                    b2Vec2 A2 = B;
                    b2Vec2 e2 = B2 - A2;
                    float v2 = b2Dot(e2, Q - A2);

                    // Is the circle in Region AB of the next edge?
                    if (v2 > 0.0f)
                        return;
                }

                cf.indexA = 1;
                cf.typeA = b2ContactFeature.e_vertex;
                manifold.pointCount = 1;
                manifold.type = b2ManifoldType.e_circles;
                manifold.localNormal.SetZero();
                manifold.localPoint = P;
                manifold.points[0].id.key = 0;
                manifold.points[0].id.cf = cf;
                manifold.points[0].localPoint = circleB.m_p;
                return;
            }

            // Region AB
            float den = b2Dot(e, e);
            b2Assert(den > 0.0f);
            b2Vec2 Pab = (1.0f / den) * (u * A + v * B);
            b2Vec2 dab = Q - Pab;
            float ddab = b2Dot(dab, dab);
            if (ddab > radius * radius)
                return;

            b2Vec2 n = new b2Vec2(-e.y, e.x);
            if (b2Dot(n, Q - A) < 0.0f)
                n.Set(-n.x, -n.y);
            n.Normalize();

            cf.indexA = 0;
            cf.typeA = b2ContactFeature.e_face;
            manifold.pointCount = 1;
            manifold.type = b2ManifoldType.e_faceA;
            manifold.localNormal = n;
            manifold.localPoint = A;
            manifold.points[0].id.key = 0;
            manifold.points[0].id.cf = cf;
            manifold.points[0].localPoint = circleB.m_p;
        }

        /// <summary>The manifold of an edge and a polygon (see <see cref="b2EPCollider"/>).</summary>
        /// <param name="manifold">Receives the manifold.</param>
        /// <param name="edgeA">The edge.</param>
        /// <param name="xfA">The edge's transform.</param>
        /// <param name="polygonB">The polygon.</param>
        /// <param name="xfB">The polygon's transform.</param>
        public static void b2CollideEdgeAndPolygon(ref b2Manifold manifold, b2EdgeShape edgeA, in b2Transform xfA, b2PolygonShape polygonB, in b2Transform xfB) {
            b2EPCollider collider = default;
            collider.Collide(ref manifold, edgeA, xfA, polygonB, xfB);
        }
    }
}
