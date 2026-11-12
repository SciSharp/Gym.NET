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
// Altered source version (Gym.NET, 2026-11-10): Box2D/Collision/b2Collision.h and b2Collision.cpp of pybox2d 2.3.10's
// bundled Box2D 2.3.0, translated to C#. The contact id stays a union (explicit layout) so its key compares exactly as
// the C++ one; the manifold's point arrays are inline arrays; b2Manifold::Type is the namespace-level b2ManifoldType.
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using static Gym.Box2D.b2Math;
using static Gym.Box2D.b2Settings;

namespace Gym.Box2D {
    /// <summary>The features that intersect to form a contact point; four bytes that alias <see cref="b2ContactID.key"/>.</summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Collision/b2Collision.h#L37">Box2D/Collision/b2Collision.h</see>.</remarks>
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    internal struct b2ContactFeature {
        /// <summary>A vertex feature.</summary>
        public const byte e_vertex = 0;

        /// <summary>A face feature.</summary>
        public const byte e_face = 1;

        /// <summary>The feature index on shape A.</summary>
        public byte indexA;

        /// <summary>The feature index on shape B.</summary>
        public byte indexB;

        /// <summary>The feature type on shape A (<see cref="e_vertex"/> or <see cref="e_face"/>).</summary>
        public byte typeA;

        /// <summary>The feature type on shape B.</summary>
        public byte typeB;
    }

    /// <summary>
    ///     A contact point's id, the union of its <see cref="b2ContactFeature"/> bytes and a 32-bit key; warm starting
    ///     matches points of consecutive manifolds by key.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Collision/b2Collision.h#L53">Box2D/Collision/b2Collision.h</see>.</remarks>
    [StructLayout(LayoutKind.Explicit)]
    internal struct b2ContactID {
        /// <summary>The features, overlaying <see cref="key"/> byte for byte (indexA in the lowest byte).</summary>
        [FieldOffset(0)]
        public b2ContactFeature cf;

        /// <summary>The features as one key, to compare ids quickly.</summary>
        [FieldOffset(0)]
        public uint key;
    }

    /// <summary>A manifold point: a contact point with its warm-start impulses and its feature id.</summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Collision/b2Collision.h#L69">Box2D/Collision/b2Collision.h</see>.</remarks>
    internal struct b2ManifoldPoint {
        /// <summary>The point, whose meaning depends on the manifold type.</summary>
        public b2Vec2 localPoint;

        /// <summary>The non-penetration impulse.</summary>
        public float normalImpulse;

        /// <summary>The friction impulse.</summary>
        public float tangentImpulse;

        /// <summary>Uniquely identifies the point between two shapes.</summary>
        public b2ContactID id;
    }

    /// <summary>The two manifold points of a <see cref="b2Manifold"/>, inline as in C++.</summary>
    [InlineArray(b2_maxManifoldPoints)]
    internal struct b2ManifoldPointArray {
        /// <summary>The first element; the runtime lays the others after it.</summary>
        private b2ManifoldPoint _element0;
    }

    /// <summary>The two world points of a <see cref="b2WorldManifold"/>, inline as in C++.</summary>
    [InlineArray(b2_maxManifoldPoints)]
    internal struct b2Vec2Pair {
        /// <summary>The first element; the runtime lays the second after it.</summary>
        private b2Vec2 _element0;
    }

    /// <summary>What a manifold's local point and normal mean (C++ <c>b2Manifold::Type</c>).</summary>
    internal enum b2ManifoldType {
        /// <summary>Circles: the local point is circle A's center.</summary>
        e_circles,

        /// <summary>Face A: the normal and point are on shape A's face.</summary>
        e_faceA,

        /// <summary>Face B: the normal and point are on shape B's face.</summary>
        e_faceB,
    }

    /// <summary>
    ///     The contact points of two touching convex shapes, in local coordinates so they survive small motions; a face
    ///     manifold has up to two points, a circles manifold one.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Collision/b2Collision.h#L92">Box2D/Collision/b2Collision.h</see>.</remarks>
    internal struct b2Manifold {
        /// <summary>The points of contact.</summary>
        public b2ManifoldPointArray points;

        /// <summary>The local normal (unused for circles).</summary>
        public b2Vec2 localNormal;

        /// <summary>The local point, whose meaning depends on <see cref="type"/>.</summary>
        public b2Vec2 localPoint;

        /// <summary>The manifold type.</summary>
        public b2ManifoldType type;

        /// <summary>The number of manifold points.</summary>
        public int pointCount;
    }

    /// <summary>A manifold's normal and points in world coordinates, for the solver and contact listeners.</summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Collision/b2Collision.h#L111">Box2D/Collision/b2Collision.h</see>.</remarks>
    internal struct b2WorldManifold {
        /// <summary>The world vector pointing from A to B.</summary>
        public b2Vec2 normal;

        /// <summary>The world contact points (the points of intersection).</summary>
        public b2Vec2Pair points;

        /// <summary>
        ///     Evaluates the manifold with the given transforms and shape radii; a manifold without points leaves this
        ///     untouched, as in C++.
        /// </summary>
        /// <param name="manifold">The manifold.</param>
        /// <param name="xfA">Shape A's transform.</param>
        /// <param name="radiusA">Shape A's radius.</param>
        /// <param name="xfB">Shape B's transform.</param>
        /// <param name="radiusB">Shape B's radius.</param>
        public void Initialize(in b2Manifold manifold, in b2Transform xfA, float radiusA, in b2Transform xfB, float radiusB) {
            if (manifold.pointCount == 0)
                return;

            switch (manifold.type) {
                case b2ManifoldType.e_circles: {
                    normal.Set(1.0f, 0.0f);
                    b2Vec2 pointA = b2Mul(xfA, manifold.localPoint);
                    b2Vec2 pointB = b2Mul(xfB, manifold.points[0].localPoint);
                    if (b2DistanceSquared(pointA, pointB) > b2_epsilon * b2_epsilon) {
                        normal = pointB - pointA;
                        normal.Normalize();
                    }

                    b2Vec2 cA = pointA + radiusA * normal;
                    b2Vec2 cB = pointB - radiusB * normal;
                    points[0] = 0.5f * (cA + cB);
                }
                    break;

                case b2ManifoldType.e_faceA: {
                    normal = b2Mul(xfA.q, manifold.localNormal);
                    b2Vec2 planePoint = b2Mul(xfA, manifold.localPoint);
                    for (int i = 0; i < manifold.pointCount; ++i) {
                        b2Vec2 clipPoint = b2Mul(xfB, manifold.points[i].localPoint);
                        b2Vec2 cA = clipPoint + (radiusA - b2Dot(clipPoint - planePoint, normal)) * normal;
                        b2Vec2 cB = clipPoint - radiusB * normal;
                        points[i] = 0.5f * (cA + cB);
                    }
                }
                    break;

                case b2ManifoldType.e_faceB: {
                    normal = b2Mul(xfB.q, manifold.localNormal);
                    b2Vec2 planePoint = b2Mul(xfB, manifold.localPoint);
                    for (int i = 0; i < manifold.pointCount; ++i) {
                        b2Vec2 clipPoint = b2Mul(xfA, manifold.points[i].localPoint);
                        b2Vec2 cB = clipPoint + (radiusB - b2Dot(clipPoint - planePoint, normal)) * normal;
                        b2Vec2 cA = clipPoint - radiusA * normal;
                        points[i] = 0.5f * (cA + cB);
                    }

                    // Ensure normal points from A to B.
                    normal = -normal;
                }
                    break;
            }
        }
    }

    /// <summary>The state of a contact point between two updates (C++ <c>b2PointState</c>).</summary>
    internal enum b2PointState {
        /// <summary>The point does not exist.</summary>
        b2_nullState,

        /// <summary>The point was added in the update.</summary>
        b2_addState,

        /// <summary>The point persisted across the update.</summary>
        b2_persistState,

        /// <summary>The point was removed in the update.</summary>
        b2_removeState,
    }

    /// <summary>A vertex used to compute contact points by clipping, with its feature id.</summary>
    internal struct b2ClipVertex {
        /// <summary>The vertex.</summary>
        public b2Vec2 v;

        /// <summary>The feature id it carries.</summary>
        public b2ContactID id;
    }

    /// <summary>Two clip vertices, inline as C++'s <c>b2ClipVertex[2]</c>.</summary>
    [InlineArray(2)]
    internal struct b2ClipVertex2 {
        /// <summary>The first element; the runtime lays the second after it.</summary>
        private b2ClipVertex _element0;
    }

    /// <summary>A ray-cast input: the ray from <see cref="p1"/> through <see cref="p2"/>, up to <c>p1 + maxFraction * (p2 - p1)</c>.</summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Collision/b2Collision.h#L150">Box2D/Collision/b2Collision.h</see>.</remarks>
    internal struct b2RayCastInput {
        /// <summary>The ray's start.</summary>
        public b2Vec2 p1;

        /// <summary>The point the ray passes through at fraction 1.</summary>
        public b2Vec2 p2;

        /// <summary>How far along the ray to look, as a fraction of <c>p2 - p1</c>.</summary>
        public float maxFraction;
    }

    /// <summary>A ray-cast hit: the surface normal and the fraction along the ray at the hit.</summary>
    internal struct b2RayCastOutput {
        /// <summary>The surface normal at the hit.</summary>
        public b2Vec2 normal;

        /// <summary>The hit's fraction along the ray.</summary>
        public float fraction;
    }

    /// <summary>An axis-aligned bounding box.</summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Collision/b2Collision.h#L165">Box2D/Collision/b2Collision.h</see>.</remarks>
    internal struct b2AABB {
        /// <summary>The lower vertex.</summary>
        public b2Vec2 lowerBound;

        /// <summary>The upper vertex.</summary>
        public b2Vec2 upperBound;

        /// <summary>Whether the bounds are sorted and finite.</summary>
        /// <returns>True for a valid box.</returns>
        public readonly bool IsValid() {
            b2Vec2 d = upperBound - lowerBound;
            bool valid = d.x >= 0.0f && d.y >= 0.0f;
            valid = valid && lowerBound.IsValid() && upperBound.IsValid();
            return valid;
        }

        /// <summary>The center.</summary>
        /// <returns><c>0.5f * (lowerBound + upperBound)</c>.</returns>
        public readonly b2Vec2 GetCenter() => 0.5f * (lowerBound + upperBound);

        /// <summary>The half-widths.</summary>
        /// <returns><c>0.5f * (upperBound - lowerBound)</c>.</returns>
        public readonly b2Vec2 GetExtents() => 0.5f * (upperBound - lowerBound);

        /// <summary>The perimeter, the dynamic tree's cost metric.</summary>
        /// <returns><c>2.0f * (wx + wy)</c>.</returns>
        public readonly float GetPerimeter() {
            float wx = upperBound.x - lowerBound.x;
            float wy = upperBound.y - lowerBound.y;
            return 2.0f * (wx + wy);
        }

        /// <summary>Grows this box to contain another.</summary>
        /// <param name="aabb">The other box.</param>
        public void Combine(in b2AABB aabb) {
            lowerBound = b2Min(lowerBound, aabb.lowerBound);
            upperBound = b2Max(upperBound, aabb.upperBound);
        }

        /// <summary>Sets this box to the union of two boxes.</summary>
        /// <param name="aabb1">The first box.</param>
        /// <param name="aabb2">The second box.</param>
        public void Combine(in b2AABB aabb1, in b2AABB aabb2) {
            lowerBound = b2Min(aabb1.lowerBound, aabb2.lowerBound);
            upperBound = b2Max(aabb1.upperBound, aabb2.upperBound);
        }

        /// <summary>Whether this box contains another.</summary>
        /// <param name="aabb">The other box.</param>
        /// <returns>True when <paramref name="aabb"/> lies inside, bounds included.</returns>
        public readonly bool Contains(in b2AABB aabb) {
            bool result = true;
            result = result && lowerBound.x <= aabb.lowerBound.x;
            result = result && lowerBound.y <= aabb.lowerBound.y;
            result = result && aabb.upperBound.x <= upperBound.x;
            result = result && aabb.upperBound.y <= upperBound.y;
            return result;
        }

        /// <summary>Casts a ray against the box (Real-Time Collision Detection, p. 179).</summary>
        /// <param name="output">Receives the hit; untouched on a miss.</param>
        /// <param name="input">The ray.</param>
        /// <returns>True on a hit within the ray's fraction; a ray starting inside misses.</returns>
        public readonly bool RayCast(ref b2RayCastOutput output, in b2RayCastInput input) {
            float tmin = -b2_maxFloat;
            float tmax = b2_maxFloat;

            b2Vec2 p = input.p1;
            b2Vec2 d = input.p2 - input.p1;
            b2Vec2 absD = b2Abs(d);

            b2Vec2 normal = default;

            for (int i = 0; i < 2; ++i) {
                if (absD[i] < b2_epsilon) {
                    // Parallel.
                    if (p[i] < lowerBound[i] || upperBound[i] < p[i])
                        return false;
                } else {
                    float inv_d = 1.0f / d[i];
                    float t1 = (lowerBound[i] - p[i]) * inv_d;
                    float t2 = (upperBound[i] - p[i]) * inv_d;

                    // Sign of the normal vector.
                    float s = -1.0f;

                    if (t1 > t2) {
                        b2Swap(ref t1, ref t2);
                        s = 1.0f;
                    }

                    // Push the min up
                    if (t1 > tmin) {
                        normal.SetZero();
                        normal[i] = s;
                        tmin = t1;
                    }

                    // Pull the max down
                    tmax = b2Min(tmax, t2);

                    if (tmin > tmax)
                        return false;
                }
            }

            // Does the ray start inside the box?
            // Does the ray intersect beyond the max fraction?
            if (tmin < 0.0f || input.maxFraction < tmin)
                return false;

            // Intersection.
            output.fraction = tmin;
            output.normal = normal;
            return true;
        }
    }

    /// <summary>
    ///     Box2D's collision free functions: point-state comparison, Sutherland-Hodgman clipping, overlap tests, and (in the
    ///     other partial files) the convex-pair manifold generators. Imported with <c>using static</c>.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Collision/b2Collision.cpp">Box2D/Collision/b2Collision.cpp</see>.</remarks>
    internal static partial class b2Collision {
        /// <summary>The feature index meaning "no feature" (UCHAR_MAX).</summary>
        public const byte b2_nullFeature = byte.MaxValue;

        /// <summary>Compares two manifolds' points by id to tell which were removed, persisted or added.</summary>
        /// <param name="state1">Receives the states of <paramref name="manifold1"/>'s points (removed or persisted).</param>
        /// <param name="state2">Receives the states of <paramref name="manifold2"/>'s points (added or persisted).</param>
        /// <param name="manifold1">The old manifold.</param>
        /// <param name="manifold2">The new manifold.</param>
        public static void b2GetPointStates(b2PointState[] state1, b2PointState[] state2, in b2Manifold manifold1, in b2Manifold manifold2) {
            for (int i = 0; i < b2_maxManifoldPoints; ++i) {
                state1[i] = b2PointState.b2_nullState;
                state2[i] = b2PointState.b2_nullState;
            }

            // Detect persists and removes.
            for (int i = 0; i < manifold1.pointCount; ++i) {
                b2ContactID id = manifold1.points[i].id;
                state1[i] = b2PointState.b2_removeState;
                for (int j = 0; j < manifold2.pointCount; ++j) {
                    if (manifold2.points[j].id.key == id.key) {
                        state1[i] = b2PointState.b2_persistState;
                        break;
                    }
                }
            }

            // Detect persists and adds.
            for (int i = 0; i < manifold2.pointCount; ++i) {
                b2ContactID id = manifold2.points[i].id;
                state2[i] = b2PointState.b2_addState;
                for (int j = 0; j < manifold1.pointCount; ++j) {
                    if (manifold1.points[j].id.key == id.key) {
                        state2[i] = b2PointState.b2_persistState;
                        break;
                    }
                }
            }
        }

        /// <summary>Clips a segment against a half-plane (Sutherland-Hodgman), keeping the points behind it.</summary>
        /// <param name="vOut">Receives up to two clipped vertices.</param>
        /// <param name="vIn">The segment's two vertices.</param>
        /// <param name="normal">The clipping plane's normal.</param>
        /// <param name="offset">The plane's offset along the normal.</param>
        /// <param name="vertexIndexA">The feature index given to an intersection point (truncated to a byte).</param>
        /// <returns>The number of output vertices, 0 to 2.</returns>
        public static int b2ClipSegmentToLine(ref b2ClipVertex2 vOut, in b2ClipVertex2 vIn, in b2Vec2 normal, float offset, int vertexIndexA) {
            // Start with no output points
            int numOut = 0;

            // Calculate the distance of end points to the line
            float distance0 = b2Dot(normal, vIn[0].v) - offset;
            float distance1 = b2Dot(normal, vIn[1].v) - offset;

            // If the points are behind the plane
            if (distance0 <= 0.0f)
                vOut[numOut++] = vIn[0];
            if (distance1 <= 0.0f)
                vOut[numOut++] = vIn[1];

            // If the points are on different sides of the plane
            if (distance0 * distance1 < 0.0f) {
                // Find intersection point of edge and plane
                float interp = distance0 / (distance0 - distance1);
                vOut[numOut].v = vIn[0].v + interp * (vIn[1].v - vIn[0].v);

                // VertexA is hitting edgeB.
                vOut[numOut].id.cf.indexA = (byte)vertexIndexA;
                vOut[numOut].id.cf.indexB = vIn[0].id.cf.indexB;
                vOut[numOut].id.cf.typeA = b2ContactFeature.e_vertex;
                vOut[numOut].id.cf.typeB = b2ContactFeature.e_face;
                ++numOut;
            }

            return numOut;
        }

        /// <summary>Whether two shape children overlap, by GJK distance with radii (closer than <c>10.0f * b2_epsilon</c>).</summary>
        /// <param name="shapeA">The first shape.</param>
        /// <param name="indexA">The first shape's child index.</param>
        /// <param name="shapeB">The second shape.</param>
        /// <param name="indexB">The second shape's child index.</param>
        /// <param name="xfA">The first shape's transform.</param>
        /// <param name="xfB">The second shape's transform.</param>
        /// <returns>True when they touch or overlap.</returns>
        public static bool b2TestOverlap(b2Shape shapeA, int indexA, b2Shape shapeB, int indexB, in b2Transform xfA, in b2Transform xfB) {
            b2DistanceInput input = default;
            input.proxyA.Set(shapeA, indexA);
            input.proxyB.Set(shapeB, indexB);
            input.transformA = xfA;
            input.transformB = xfB;
            input.useRadii = true;

            b2SimplexCache cache = default;
            cache.count = 0;

            b2DistanceOutput output;

            b2DistanceGJK.b2Distance(out output, ref cache, input);

            return output.distance < 10.0f * b2_epsilon;
        }

        /// <summary>Whether two boxes overlap (touching counts).</summary>
        /// <param name="a">The first box.</param>
        /// <param name="b">The second box.</param>
        /// <returns>True when they overlap.</returns>
        public static bool b2TestOverlap(in b2AABB a, in b2AABB b) {
            b2Vec2 d1, d2;
            d1 = b.lowerBound - a.upperBound;
            d2 = a.lowerBound - b.upperBound;

            if (d1.x > 0.0f || d1.y > 0.0f)
                return false;

            if (d2.x > 0.0f || d2.y > 0.0f)
                return false;

            return true;
        }
    }
}
