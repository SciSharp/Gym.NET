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
// Altered source version (Gym.NET, 2026-11-10): Box2D/Collision/Shapes/b2Shape.h, b2CircleShape.h/.cpp and
// b2EdgeShape.h/.cpp of pybox2d 2.3.10's bundled Box2D 2.3.0, translated to C#. b2Shape::Type is the namespace-level
// b2ShapeType and GetType() is GetShapeType(), since C# objects already have a GetType(); Clone allocates with the GC.
using System;
using static Gym.Box2D.b2Math;
using static Gym.Box2D.b2Settings;

namespace Gym.Box2D {
    /// <summary>The kind of a <see cref="b2Shape"/> (C++ <c>b2Shape::Type</c>); the contact factory dispatches on the pair.</summary>
    public enum b2ShapeType {
        /// <summary>A circle (<see cref="b2CircleShape"/>).</summary>
        e_circle = 0,

        /// <summary>A line segment (<see cref="b2EdgeShape"/>).</summary>
        e_edge = 1,

        /// <summary>A convex polygon (<see cref="b2PolygonShape"/>).</summary>
        e_polygon = 2,

        /// <summary>A chain of edges; not part of this port, as no Gymnasium env uses one.</summary>
        e_chain = 3,

        /// <summary>The number of shape kinds.</summary>
        e_typeCount = 4,
    }

    /// <summary>The mass properties a shape contributes at a density: mass, centroid, and rotational inertia about the shape origin.</summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Collision/Shapes/b2Shape.h#L26">Box2D/Collision/Shapes/b2Shape.h</see>.</remarks>
    internal struct b2MassData {
        /// <summary>The mass in kilograms.</summary>
        public float mass;

        /// <summary>The position of the shape's centroid relative to the shape's origin.</summary>
        public b2Vec2 center;

        /// <summary>The rotational inertia about the shape's local origin.</summary>
        public float I;
    }

    /// <summary>
    ///     A collision shape, the geometry a <see cref="b2Fixture"/> carries. Shapes hold no body: a fixture clones the shape
    ///     it is created from, so changing a shape object afterwards changes nothing in the world.
    /// </summary>
    /// <remarks>
    ///     Box2D's shape kinds outside this port (chains) are left out because no Gymnasium environment uses them; the
    ///     internal abstract members keep users from deriving their own shapes, which the contact factory could not collide.
    ///     <para>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Collision/Shapes/b2Shape.h#L41">Box2D/Collision/Shapes/b2Shape.h</see>.</para>
    /// </remarks>
    public abstract partial class b2Shape {
        /// <summary>The shape kind.</summary>
        internal b2ShapeType m_type;

        /// <summary>The radius: a circle's, or the polygon/edge skin (<c>b2_polygonRadius</c>).</summary>
        internal float m_radius;

        /// <summary>A copy of this shape, as the fixture that adopts it keeps its own (C++ <c>Clone</c>).</summary>
        /// <returns>The copy.</returns>
        internal abstract b2Shape Clone();

        /// <summary>The shape kind (C++ <c>GetType</c>).</summary>
        /// <returns>The kind.</returns>
        internal b2ShapeType GetShapeType() => m_type;

        /// <summary>The number of child primitives (1 for every shape in this port).</summary>
        /// <returns>The count.</returns>
        internal abstract int GetChildCount();

        /// <summary>Whether a world point lies inside this shape; edges contain nothing.</summary>
        /// <param name="xf">The shape's transform.</param>
        /// <param name="p">The point in world coordinates.</param>
        /// <returns>True when the point is inside.</returns>
        internal abstract bool TestPoint(in b2Transform xf, in b2Vec2 p);

        /// <summary>Casts a ray against a child shape.</summary>
        /// <param name="output">Receives the hit; untouched on a miss.</param>
        /// <param name="input">The ray.</param>
        /// <param name="transform">The shape's transform.</param>
        /// <param name="childIndex">The child index.</param>
        /// <returns>True on a hit.</returns>
        internal abstract bool RayCast(ref b2RayCastOutput output, in b2RayCastInput input, in b2Transform transform, int childIndex);

        /// <summary>The bounding box of a child shape under a transform.</summary>
        /// <param name="aabb">Receives the box.</param>
        /// <param name="xf">The transform.</param>
        /// <param name="childIndex">The child index.</param>
        internal abstract void ComputeAABB(out b2AABB aabb, in b2Transform xf, int childIndex);

        /// <summary>The mass properties at a density (kg/m²).</summary>
        /// <param name="massData">Receives the mass properties.</param>
        /// <param name="density">The density.</param>
        internal abstract void ComputeMass(out b2MassData massData, float density);
    }

    /// <summary>A solid circle of radius <c>m_radius</c> around the local point <c>m_p</c>.</summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Collision/Shapes/b2CircleShape.cpp">Box2D/Collision/Shapes/b2CircleShape.cpp</see>.</remarks>
    public sealed partial class b2CircleShape : b2Shape {
        /// <summary>The center in the body's frame.</summary>
        internal b2Vec2 m_p;

        /// <summary>A circle of radius 0 at the origin.</summary>
        public b2CircleShape() {
            m_type = b2ShapeType.e_circle;
            m_radius = 0.0f;
            m_p.SetZero();
        }

        /// <inheritdoc/>
        internal override b2Shape Clone() => (b2CircleShape)MemberwiseClone();

        /// <inheritdoc/>
        internal override int GetChildCount() => 1;

        /// <summary>The support vertex index in a direction: the circle's only vertex.</summary>
        /// <param name="d">The direction (unused).</param>
        /// <returns>0.</returns>
        internal int GetSupport(in b2Vec2 d) => 0;

        /// <summary>The support vertex in a direction: the center.</summary>
        /// <param name="d">The direction (unused).</param>
        /// <returns>The center.</returns>
        internal b2Vec2 GetSupportVertex(in b2Vec2 d) => m_p;

        /// <summary>The vertex count used by the distance algorithm: 1.</summary>
        /// <returns>1.</returns>
        internal int GetVertexCount() => 1;

        /// <summary>A vertex by index; the circle's only one is its center.</summary>
        /// <param name="index">0.</param>
        /// <returns>The center.</returns>
        /// <exception cref="NumSharp.AssertionError">Thrown for an index other than 0.</exception>
        internal b2Vec2 GetVertex(int index) {
            b2Assert(index == 0);
            return m_p;
        }

        /// <inheritdoc/>
        internal override bool TestPoint(in b2Transform transform, in b2Vec2 p) {
            b2Vec2 center = transform.p + b2Mul(transform.q, m_p);
            b2Vec2 d = p - center;
            return b2Dot(d, d) <= m_radius * m_radius;
        }

        /// <summary>Casts a ray against the circle (van den Bergen, Collision Detection in Interactive 3D Environments, §3.1.2).</summary>
        /// <param name="output">Receives the hit; untouched on a miss.</param>
        /// <param name="input">The ray.</param>
        /// <param name="transform">The shape's transform.</param>
        /// <param name="childIndex">Unused.</param>
        /// <returns>True on a hit within the ray's fraction.</returns>
        internal override bool RayCast(ref b2RayCastOutput output, in b2RayCastInput input, in b2Transform transform, int childIndex) {
            b2Vec2 position = transform.p + b2Mul(transform.q, m_p);
            b2Vec2 s = input.p1 - position;
            float b = b2Dot(s, s) - m_radius * m_radius;

            // Solve quadratic equation.
            b2Vec2 r = input.p2 - input.p1;
            float c = b2Dot(s, r);
            float rr = b2Dot(r, r);
            float sigma = c * c - rr * b;

            // Check for negative discriminant and short segment.
            if (sigma < 0.0f || rr < b2_epsilon)
                return false;

            // Find the point of intersection of the line with the circle.
            float a = -(c + MathF.Sqrt(sigma));

            // Is the intersection point on the segment?
            if (0.0f <= a && a <= input.maxFraction * rr) {
                a /= rr;
                output.fraction = a;
                output.normal = s + a * r;
                output.normal.Normalize();
                return true;
            }

            return false;
        }

        /// <inheritdoc/>
        internal override void ComputeAABB(out b2AABB aabb, in b2Transform transform, int childIndex) {
            b2Vec2 p = transform.p + b2Mul(transform.q, m_p);
            aabb.lowerBound = new b2Vec2(p.x - m_radius, p.y - m_radius);
            aabb.upperBound = new b2Vec2(p.x + m_radius, p.y + m_radius);
        }

        /// <inheritdoc/>
        internal override void ComputeMass(out b2MassData massData, float density) {
            massData.mass = density * b2_pi * m_radius * m_radius;
            massData.center = m_p;

            // inertia about the local origin
            massData.I = massData.mass * (0.5f * m_radius * m_radius + b2Dot(m_p, m_p));
        }
    }

    /// <summary>
    ///     A line segment from <c>m_vertex1</c> to <c>m_vertex2</c>, with optional ghost neighbours (<c>m_vertex0</c>,
    ///     <c>m_vertex3</c>) for smooth collision along connected edges. Edges have no mass and contain no point.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Collision/Shapes/b2EdgeShape.cpp">Box2D/Collision/Shapes/b2EdgeShape.cpp</see>.</remarks>
    public sealed partial class b2EdgeShape : b2Shape {
        /// <summary>The first vertex.</summary>
        internal b2Vec2 m_vertex1;

        /// <summary>The second vertex.</summary>
        internal b2Vec2 m_vertex2;

        /// <summary>The optional ghost vertex before <see cref="m_vertex1"/>.</summary>
        internal b2Vec2 m_vertex0;

        /// <summary>The optional ghost vertex after <see cref="m_vertex2"/>.</summary>
        internal b2Vec2 m_vertex3;

        /// <summary>Whether <see cref="m_vertex0"/> is set.</summary>
        internal bool m_hasVertex0;

        /// <summary>Whether <see cref="m_vertex3"/> is set.</summary>
        internal bool m_hasVertex3;

        /// <summary>A degenerate edge at the origin with the polygon skin radius and no ghost vertices.</summary>
        public b2EdgeShape() {
            m_type = b2ShapeType.e_edge;
            m_radius = b2_polygonRadius;
            m_vertex0.x = 0.0f;
            m_vertex0.y = 0.0f;
            m_vertex3.x = 0.0f;
            m_vertex3.y = 0.0f;
            m_hasVertex0 = false;
            m_hasVertex3 = false;
        }

        /// <summary>Sets the segment's vertices and drops the ghost vertices.</summary>
        /// <param name="v1">The first vertex.</param>
        /// <param name="v2">The second vertex.</param>
        internal void Set(in b2Vec2 v1, in b2Vec2 v2) {
            m_vertex1 = v1;
            m_vertex2 = v2;
            m_hasVertex0 = false;
            m_hasVertex3 = false;
        }

        /// <inheritdoc/>
        internal override b2Shape Clone() => (b2EdgeShape)MemberwiseClone();

        /// <inheritdoc/>
        internal override int GetChildCount() => 1;

        /// <inheritdoc/>
        internal override bool TestPoint(in b2Transform xf, in b2Vec2 p) => false;

        /// <summary>Casts a ray against the segment, in the edge's frame.</summary>
        /// <param name="output">Receives the hit; untouched on a miss.</param>
        /// <param name="input">The ray.</param>
        /// <param name="xf">The shape's transform.</param>
        /// <param name="childIndex">Unused.</param>
        /// <returns>True on a hit within the ray's fraction and the segment's span.</returns>
        internal override bool RayCast(ref b2RayCastOutput output, in b2RayCastInput input, in b2Transform xf, int childIndex) {
            // Put the ray into the edge's frame of reference.
            b2Vec2 p1 = b2MulT(xf.q, input.p1 - xf.p);
            b2Vec2 p2 = b2MulT(xf.q, input.p2 - xf.p);
            b2Vec2 d = p2 - p1;

            b2Vec2 v1 = m_vertex1;
            b2Vec2 v2 = m_vertex2;
            b2Vec2 e = v2 - v1;
            b2Vec2 normal = new b2Vec2(e.y, -e.x);
            normal.Normalize();

            // q = p1 + t * d
            // dot(normal, q - v1) = 0
            // dot(normal, p1 - v1) + t * dot(normal, d) = 0
            float numerator = b2Dot(normal, v1 - p1);
            float denominator = b2Dot(normal, d);

            if (denominator == 0.0f)
                return false;

            float t = numerator / denominator;
            if (t < 0.0f || input.maxFraction < t)
                return false;

            b2Vec2 q = p1 + t * d;

            // q = v1 + s * r
            // s = dot(q - v1, r) / dot(r, r)
            b2Vec2 r = v2 - v1;
            float rr = b2Dot(r, r);
            if (rr == 0.0f)
                return false;

            float s = b2Dot(q - v1, r) / rr;
            if (s < 0.0f || 1.0f < s)
                return false;

            output.fraction = t;
            if (numerator > 0.0f)
                output.normal = -normal;
            else
                output.normal = normal;
            return true;
        }

        /// <inheritdoc/>
        internal override void ComputeAABB(out b2AABB aabb, in b2Transform xf, int childIndex) {
            b2Vec2 v1 = b2Mul(xf, m_vertex1);
            b2Vec2 v2 = b2Mul(xf, m_vertex2);

            b2Vec2 lower = b2Min(v1, v2);
            b2Vec2 upper = b2Max(v1, v2);

            b2Vec2 r = new b2Vec2(m_radius, m_radius);
            aabb.lowerBound = lower - r;
            aabb.upperBound = upper + r;
        }

        /// <inheritdoc/>
        internal override void ComputeMass(out b2MassData massData, float density) {
            massData.mass = 0.0f;
            massData.center = 0.5f * (m_vertex1 + m_vertex2);
            massData.I = 0.0f;
        }
    }
}
