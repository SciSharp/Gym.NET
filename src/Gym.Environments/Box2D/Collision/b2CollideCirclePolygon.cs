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
// Altered source version (Gym.NET, 2026-11-10): Box2D/Collision/b2CollideCircle.cpp and b2CollidePolygon.cpp of
// pybox2d 2.3.10's bundled Box2D 2.3.0, translated to C#: the manifolds of circle-circle, polygon-circle and
// polygon-polygon pairs, with the C++ order of operations and tie-breaking kept.
using static Gym.Box2D.b2Math;
using static Gym.Box2D.b2Settings;

namespace Gym.Box2D {
    internal static partial class b2Collision {
        /// <summary>The manifold of two circles: one point when they overlap, none otherwise.</summary>
        /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Collision/b2CollideCircle.cpp#L23">Box2D/Collision/b2CollideCircle.cpp</see>.</remarks>
        /// <param name="manifold">Receives the manifold.</param>
        /// <param name="circleA">Circle A.</param>
        /// <param name="xfA">Circle A's transform.</param>
        /// <param name="circleB">Circle B.</param>
        /// <param name="xfB">Circle B's transform.</param>
        public static void b2CollideCircles(ref b2Manifold manifold, b2CircleShape circleA, in b2Transform xfA, b2CircleShape circleB, in b2Transform xfB) {
            manifold.pointCount = 0;

            b2Vec2 pA = b2Mul(xfA, circleA.m_p);
            b2Vec2 pB = b2Mul(xfB, circleB.m_p);

            b2Vec2 d = pB - pA;
            float distSqr = b2Dot(d, d);
            float rA = circleA.m_radius, rB = circleB.m_radius;
            float radius = rA + rB;
            if (distSqr > radius * radius)
                return;

            manifold.type = b2ManifoldType.e_circles;
            manifold.localPoint = circleA.m_p;
            manifold.localNormal.SetZero();
            manifold.pointCount = 1;

            manifold.points[0].localPoint = circleB.m_p;
            manifold.points[0].id.key = 0;
        }

        /// <summary>The manifold of a polygon and a circle: the closest face, or the nearest vertex region.</summary>
        /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Collision/b2CollideCircle.cpp#L50">Box2D/Collision/b2CollideCircle.cpp</see>.</remarks>
        /// <param name="manifold">Receives the manifold.</param>
        /// <param name="polygonA">The polygon.</param>
        /// <param name="xfA">The polygon's transform.</param>
        /// <param name="circleB">The circle.</param>
        /// <param name="xfB">The circle's transform.</param>
        public static void b2CollidePolygonAndCircle(ref b2Manifold manifold, b2PolygonShape polygonA, in b2Transform xfA, b2CircleShape circleB, in b2Transform xfB) {
            manifold.pointCount = 0;

            // Compute circle position in the frame of the polygon.
            b2Vec2 c = b2Mul(xfB, circleB.m_p);
            b2Vec2 cLocal = b2MulT(xfA, c);

            // Find the min separating edge.
            int normalIndex = 0;
            float separation = -b2_maxFloat;
            float radius = polygonA.m_radius + circleB.m_radius;
            int vertexCount = polygonA.m_count;
            b2Vec2[] vertices = polygonA.m_vertices;
            b2Vec2[] normals = polygonA.m_normals;

            for (int i = 0; i < vertexCount; ++i) {
                float s = b2Dot(normals[i], cLocal - vertices[i]);

                if (s > radius)
                    // Early out.
                    return;

                if (s > separation) {
                    separation = s;
                    normalIndex = i;
                }
            }

            // Vertices that subtend the incident face.
            int vertIndex1 = normalIndex;
            int vertIndex2 = vertIndex1 + 1 < vertexCount ? vertIndex1 + 1 : 0;
            b2Vec2 v1 = vertices[vertIndex1];
            b2Vec2 v2 = vertices[vertIndex2];

            // If the center is inside the polygon ...
            if (separation < b2_epsilon) {
                manifold.pointCount = 1;
                manifold.type = b2ManifoldType.e_faceA;
                manifold.localNormal = normals[normalIndex];
                manifold.localPoint = 0.5f * (v1 + v2);
                manifold.points[0].localPoint = circleB.m_p;
                manifold.points[0].id.key = 0;
                return;
            }

            // Compute barycentric coordinates
            float u1 = b2Dot(cLocal - v1, v2 - v1);
            float u2 = b2Dot(cLocal - v2, v1 - v2);
            if (u1 <= 0.0f) {
                if (b2DistanceSquared(cLocal, v1) > radius * radius)
                    return;

                manifold.pointCount = 1;
                manifold.type = b2ManifoldType.e_faceA;
                manifold.localNormal = cLocal - v1;
                manifold.localNormal.Normalize();
                manifold.localPoint = v1;
                manifold.points[0].localPoint = circleB.m_p;
                manifold.points[0].id.key = 0;
            } else if (u2 <= 0.0f) {
                if (b2DistanceSquared(cLocal, v2) > radius * radius)
                    return;

                manifold.pointCount = 1;
                manifold.type = b2ManifoldType.e_faceA;
                manifold.localNormal = cLocal - v2;
                manifold.localNormal.Normalize();
                manifold.localPoint = v2;
                manifold.points[0].localPoint = circleB.m_p;
                manifold.points[0].id.key = 0;
            } else {
                b2Vec2 faceCenter = 0.5f * (v1 + v2);
                float faceSeparation = b2Dot(cLocal - faceCenter, normals[vertIndex1]);
                if (faceSeparation > radius)
                    return;

                manifold.pointCount = 1;
                manifold.type = b2ManifoldType.e_faceA;
                manifold.localNormal = normals[vertIndex1];
                manifold.localPoint = faceCenter;
                manifold.points[0].localPoint = circleB.m_p;
                manifold.points[0].id.key = 0;
            }
        }

        /// <summary>The separation of two polygons along one edge normal of the first.</summary>
        /// <param name="poly1">The polygon whose edge normal is used.</param>
        /// <param name="xf1">Its transform.</param>
        /// <param name="edge1">The edge index.</param>
        /// <param name="poly2">The other polygon.</param>
        /// <param name="xf2">Its transform.</param>
        /// <returns>The separation (negative when overlapping).</returns>
        /// <exception cref="NumSharp.AssertionError">Thrown for an edge index out of range.</exception>
        private static float b2EdgeSeparation(b2PolygonShape poly1, in b2Transform xf1, int edge1, b2PolygonShape poly2, in b2Transform xf2) {
            b2Vec2[] vertices1 = poly1.m_vertices;
            b2Vec2[] normals1 = poly1.m_normals;

            int count2 = poly2.m_count;
            b2Vec2[] vertices2 = poly2.m_vertices;

            b2Assert(0 <= edge1 && edge1 < poly1.m_count, "0 <= edge1 && edge1 < poly1->m_count");

            // Convert normal from poly1's frame into poly2's frame.
            b2Vec2 normal1World = b2Mul(xf1.q, normals1[edge1]);
            b2Vec2 normal1 = b2MulT(xf2.q, normal1World);

            // Find support vertex on poly2 for -normal.
            int index = 0;
            float minDot = b2_maxFloat;

            for (int i = 0; i < count2; ++i) {
                float dot = b2Dot(vertices2[i], normal1);
                if (dot < minDot) {
                    minDot = dot;
                    index = i;
                }
            }

            b2Vec2 v1 = b2Mul(xf1, vertices1[edge1]);
            b2Vec2 v2 = b2Mul(xf2, vertices2[index]);
            float separation = b2Dot(v2 - v1, normal1World);
            return separation;
        }

        /// <summary>The largest separation of two polygons over the first one's edge normals, by a local search from the normal facing the second's centroid.</summary>
        /// <param name="edgeIndex">Receives the edge of maximum separation.</param>
        /// <param name="poly1">The polygon whose edges are searched.</param>
        /// <param name="xf1">Its transform.</param>
        /// <param name="poly2">The other polygon.</param>
        /// <param name="xf2">Its transform.</param>
        /// <returns>The maximum separation.</returns>
        private static float b2FindMaxSeparation(out int edgeIndex, b2PolygonShape poly1, in b2Transform xf1, b2PolygonShape poly2, in b2Transform xf2) {
            int count1 = poly1.m_count;
            b2Vec2[] normals1 = poly1.m_normals;

            // Vector pointing from the centroid of poly1 to the centroid of poly2.
            b2Vec2 d = b2Mul(xf2, poly2.m_centroid) - b2Mul(xf1, poly1.m_centroid);
            b2Vec2 dLocal1 = b2MulT(xf1.q, d);

            // Find edge normal on poly1 that has the largest projection onto d.
            int edge = 0;
            float maxDot = -b2_maxFloat;
            for (int i = 0; i < count1; ++i) {
                float dot = b2Dot(normals1[i], dLocal1);
                if (dot > maxDot) {
                    maxDot = dot;
                    edge = i;
                }
            }

            // Get the separation for the edge normal.
            float s = b2EdgeSeparation(poly1, xf1, edge, poly2, xf2);

            // Check the separation for the previous edge normal.
            int prevEdge = edge - 1 >= 0 ? edge - 1 : count1 - 1;
            float sPrev = b2EdgeSeparation(poly1, xf1, prevEdge, poly2, xf2);

            // Check the separation for the next edge normal.
            int nextEdge = edge + 1 < count1 ? edge + 1 : 0;
            float sNext = b2EdgeSeparation(poly1, xf1, nextEdge, poly2, xf2);

            // Find the best edge and the search direction.
            int bestEdge;
            float bestSeparation;
            int increment;
            if (sPrev > s && sPrev > sNext) {
                increment = -1;
                bestEdge = prevEdge;
                bestSeparation = sPrev;
            } else if (sNext > s) {
                increment = 1;
                bestEdge = nextEdge;
                bestSeparation = sNext;
            } else {
                edgeIndex = edge;
                return s;
            }

            // Perform a local search for the best edge normal.
            for (; ; ) {
                if (increment == -1)
                    edge = bestEdge - 1 >= 0 ? bestEdge - 1 : count1 - 1;
                else
                    edge = bestEdge + 1 < count1 ? bestEdge + 1 : 0;

                s = b2EdgeSeparation(poly1, xf1, edge, poly2, xf2);

                if (s > bestSeparation) {
                    bestEdge = edge;
                    bestSeparation = s;
                } else {
                    break;
                }
            }

            edgeIndex = bestEdge;
            return bestSeparation;
        }

        /// <summary>The incident edge of polygon 2 against reference edge <paramref name="edge1"/> of polygon 1: the edge whose normal is most anti-parallel.</summary>
        /// <param name="c">Receives the edge's two clip vertices in world coordinates.</param>
        /// <param name="poly1">The reference polygon.</param>
        /// <param name="xf1">Its transform.</param>
        /// <param name="edge1">The reference edge.</param>
        /// <param name="poly2">The incident polygon.</param>
        /// <param name="xf2">Its transform.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown for a reference edge out of range.</exception>
        private static void b2FindIncidentEdge(ref b2ClipVertex2 c, b2PolygonShape poly1, in b2Transform xf1, int edge1, b2PolygonShape poly2, in b2Transform xf2) {
            b2Vec2[] normals1 = poly1.m_normals;

            int count2 = poly2.m_count;
            b2Vec2[] vertices2 = poly2.m_vertices;
            b2Vec2[] normals2 = poly2.m_normals;

            b2Assert(0 <= edge1 && edge1 < poly1.m_count, "0 <= edge1 && edge1 < poly1->m_count");

            // Get the normal of the reference edge in poly2's frame.
            b2Vec2 normal1 = b2MulT(xf2.q, b2Mul(xf1.q, normals1[edge1]));

            // Find the incident edge on poly2.
            int index = 0;
            float minDot = b2_maxFloat;
            for (int i = 0; i < count2; ++i) {
                float dot = b2Dot(normal1, normals2[i]);
                if (dot < minDot) {
                    minDot = dot;
                    index = i;
                }
            }

            // Build the clip vertices for the incident edge.
            int i1 = index;
            int i2 = i1 + 1 < count2 ? i1 + 1 : 0;

            c[0].v = b2Mul(xf2, vertices2[i1]);
            c[0].id.cf.indexA = (byte)edge1;
            c[0].id.cf.indexB = (byte)i1;
            c[0].id.cf.typeA = b2ContactFeature.e_face;
            c[0].id.cf.typeB = b2ContactFeature.e_vertex;

            c[1].v = b2Mul(xf2, vertices2[i2]);
            c[1].id.cf.indexA = (byte)edge1;
            c[1].id.cf.indexB = (byte)i2;
            c[1].id.cf.typeA = b2ContactFeature.e_face;
            c[1].id.cf.typeB = b2ContactFeature.e_vertex;
        }

        /// <summary>
        ///     The manifold of two polygons: the reference face is the one of larger separation (B's only when clearly
        ///     larger, within 0.98 relative and 0.001 absolute), the incident edge is clipped against its side planes.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Collision/b2CollidePolygon.cpp#L192">Box2D/Collision/b2CollidePolygon.cpp</see>.</remarks>
        /// <param name="manifold">Receives the manifold.</param>
        /// <param name="polyA">Polygon A.</param>
        /// <param name="xfA">Polygon A's transform.</param>
        /// <param name="polyB">Polygon B.</param>
        /// <param name="xfB">Polygon B's transform.</param>
        public static void b2CollidePolygons(ref b2Manifold manifold, b2PolygonShape polyA, in b2Transform xfA, b2PolygonShape polyB, in b2Transform xfB) {
            manifold.pointCount = 0;
            float totalRadius = polyA.m_radius + polyB.m_radius;

            float separationA = b2FindMaxSeparation(out int edgeA, polyA, xfA, polyB, xfB);
            if (separationA > totalRadius)
                return;

            float separationB = b2FindMaxSeparation(out int edgeB, polyB, xfB, polyA, xfA);
            if (separationB > totalRadius)
                return;

            b2PolygonShape poly1; // reference polygon
            b2PolygonShape poly2; // incident polygon
            b2Transform xf1, xf2;
            int edge1; // reference edge
            byte flip;
            const float k_relativeTol = 0.98f;
            const float k_absoluteTol = 0.001f;

            if (separationB > k_relativeTol * separationA + k_absoluteTol) {
                poly1 = polyB;
                poly2 = polyA;
                xf1 = xfB;
                xf2 = xfA;
                edge1 = edgeB;
                manifold.type = b2ManifoldType.e_faceB;
                flip = 1;
            } else {
                poly1 = polyA;
                poly2 = polyB;
                xf1 = xfA;
                xf2 = xfB;
                edge1 = edgeA;
                manifold.type = b2ManifoldType.e_faceA;
                flip = 0;
            }

            b2ClipVertex2 incidentEdge = default;
            b2FindIncidentEdge(ref incidentEdge, poly1, xf1, edge1, poly2, xf2);

            int count1 = poly1.m_count;
            b2Vec2[] vertices1 = poly1.m_vertices;

            int iv1 = edge1;
            int iv2 = edge1 + 1 < count1 ? edge1 + 1 : 0;

            b2Vec2 v11 = vertices1[iv1];
            b2Vec2 v12 = vertices1[iv2];

            b2Vec2 localTangent = v12 - v11;
            localTangent.Normalize();

            b2Vec2 localNormal = b2Cross(localTangent, 1.0f);
            b2Vec2 planePoint = 0.5f * (v11 + v12);

            b2Vec2 tangent = b2Mul(xf1.q, localTangent);
            b2Vec2 normal = b2Cross(tangent, 1.0f);

            v11 = b2Mul(xf1, v11);
            v12 = b2Mul(xf1, v12);

            // Face offset.
            float frontOffset = b2Dot(normal, v11);

            // Side offsets, extended by polytope skin thickness.
            float sideOffset1 = -b2Dot(tangent, v11) + totalRadius;
            float sideOffset2 = b2Dot(tangent, v12) + totalRadius;

            // Clip incident edge against extruded edge1 side edges.
            b2ClipVertex2 clipPoints1 = default;
            b2ClipVertex2 clipPoints2 = default;
            int np;

            // Clip to box side 1
            np = b2ClipSegmentToLine(ref clipPoints1, incidentEdge, -tangent, sideOffset1, iv1);

            if (np < 2)
                return;

            // Clip to negative box side 1
            np = b2ClipSegmentToLine(ref clipPoints2, clipPoints1, tangent, sideOffset2, iv2);

            if (np < 2)
                return;

            // Now clipPoints2 contains the clipped points.
            manifold.localNormal = localNormal;
            manifold.localPoint = planePoint;

            int pointCount = 0;
            for (int i = 0; i < b2_maxManifoldPoints; ++i) {
                float separation = b2Dot(normal, clipPoints2[i].v) - frontOffset;

                if (separation <= totalRadius) {
                    ref b2ManifoldPoint cp = ref manifold.points[pointCount];
                    cp.localPoint = b2MulT(xf2, clipPoints2[i].v);
                    cp.id = clipPoints2[i].id;
                    if (flip != 0) {
                        // Swap features
                        b2ContactFeature cf = cp.id.cf;
                        cp.id.cf.indexA = cf.indexB;
                        cp.id.cf.indexB = cf.indexA;
                        cp.id.cf.typeA = cf.typeB;
                        cp.id.cf.typeB = cf.typeA;
                    }

                    ++pointCount;
                }
            }

            manifold.pointCount = pointCount;
        }
    }
}
