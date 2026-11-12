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
// Altered source version (Gym.NET, 2026-11-10): Box2D/Dynamics/Contacts/b2Contact.h, b2Contact.cpp and the circle,
// polygon, polygon-circle, edge-circle and edge-polygon contacts of pybox2d 2.3.10's bundled Box2D 2.3.0, translated to
// C#. The registration table is built once by the static constructor (C++ fills it lazily on the first contact, which is
// the same table); chain contacts are left out with chain shapes; contact edges are objects, so the body lists link them
// by reference as C++ links the embedded edges by address.
using System;
using static Gym.Box2D.b2Math;
using static Gym.Box2D.b2Settings;

namespace Gym.Box2D {
    /// <summary>
    ///     A contact edge: connects a body to a contact in the body's contact list (pybox2d's <c>b2ContactEdge</c>, what
    ///     <c>body.contacts</c> holds). Each contact has two, one per body.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Dynamics/Contacts/b2Contact.h#L67">Box2D/Dynamics/Contacts/b2Contact.h</see>.</remarks>
    public sealed partial class b2ContactEdge {
        /// <summary>The other body of the contact.</summary>
        internal b2Body? other;

        /// <summary>The contact.</summary>
        internal b2Contact? contact;

        /// <summary>The previous edge of the body's list.</summary>
        internal b2ContactEdge? prev;

        /// <summary>The next edge of the body's list.</summary>
        internal b2ContactEdge? next;
    }

    /// <summary>
    ///     The contact between two fixtures whose fat boxes overlap: its manifold, touching state, friction and restitution.
    ///     The world creates and destroys contacts; a contact exists while the boxes overlap, touching or not.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Dynamics/Contacts/b2Contact.cpp">Box2D/Dynamics/Contacts/b2Contact.cpp</see>.</remarks>
    public abstract partial class b2Contact {
        /// <summary>Used when crawling the contact graph to form islands.</summary>
        internal const uint e_islandFlag = 0x0001;

        /// <summary>Set when the shapes are touching.</summary>
        internal const uint e_touchingFlag = 0x0002;

        /// <summary>This contact can be disabled (by the user).</summary>
        internal const uint e_enabledFlag = 0x0004;

        /// <summary>This contact needs filtering because a fixture filter was changed.</summary>
        internal const uint e_filterFlag = 0x0008;

        /// <summary>This bullet contact had a time-of-impact event.</summary>
        internal const uint e_bulletHitFlag = 0x0010;

        /// <summary>This contact has a valid time of impact in <see cref="m_toi"/>.</summary>
        internal const uint e_toiFlag = 0x0020;

        /// <summary>The factory per pair of shape kinds, built once.</summary>
        private static readonly b2ContactRegister[,] s_registers = InitializeRegisters();

        /// <summary>The flags.</summary>
        internal uint m_flags;

        /// <summary>The previous contact of the world's list.</summary>
        internal b2Contact? m_prev;

        /// <summary>The next contact of the world's list.</summary>
        internal b2Contact? m_next;

        /// <summary>The edge in body A's contact list.</summary>
        internal readonly b2ContactEdge m_nodeA = new b2ContactEdge();

        /// <summary>The edge in body B's contact list.</summary>
        internal readonly b2ContactEdge m_nodeB = new b2ContactEdge();

        /// <summary>Fixture A.</summary>
        internal b2Fixture? m_fixtureA;

        /// <summary>Fixture B.</summary>
        internal b2Fixture? m_fixtureB;

        /// <summary>Fixture A's child index.</summary>
        internal int m_indexA;

        /// <summary>Fixture B's child index.</summary>
        internal int m_indexB;

        /// <summary>The manifold.</summary>
        internal b2Manifold m_manifold;

        /// <summary>The number of time-of-impact sub-steps this step.</summary>
        internal int m_toiCount;

        /// <summary>The cached time of impact.</summary>
        internal float m_toi;

        /// <summary>The mixed friction.</summary>
        internal float m_friction;

        /// <summary>The mixed restitution.</summary>
        internal float m_restitution;

        /// <summary>The conveyor-belt tangent speed.</summary>
        internal float m_tangentSpeed;

        /// <summary>A contact between two fixture children, enabled, with mixed friction and restitution.</summary>
        /// <param name="fA">Fixture A.</param>
        /// <param name="indexA">Fixture A's child index.</param>
        /// <param name="fB">Fixture B.</param>
        /// <param name="indexB">Fixture B's child index.</param>
        protected b2Contact(b2Fixture fA, int indexA, b2Fixture fB, int indexB) {
            m_flags = e_enabledFlag;

            m_fixtureA = fA;
            m_fixtureB = fB;

            m_indexA = indexA;
            m_indexB = indexB;

            m_manifold.pointCount = 0;

            m_prev = null;
            m_next = null;

            m_nodeA.contact = null;
            m_nodeA.prev = null;
            m_nodeA.next = null;
            m_nodeA.other = null;

            m_nodeB.contact = null;
            m_nodeB.prev = null;
            m_nodeB.next = null;
            m_nodeB.other = null;

            m_toiCount = 0;

            m_friction = b2MixFriction(m_fixtureA.m_friction, m_fixtureB.m_friction);
            m_restitution = b2MixRestitution(m_fixtureA.m_restitution, m_fixtureB.m_restitution);

            m_tangentSpeed = 0.0f;
        }

        /// <summary>Friction mixing: the geometric mean, <c>sqrtf(friction1 * friction2)</c>.</summary>
        /// <param name="friction1">The first friction.</param>
        /// <param name="friction2">The second friction.</param>
        /// <returns>The mixed friction.</returns>
        internal static float b2MixFriction(float friction1, float friction2) => MathF.Sqrt(friction1 * friction2);

        /// <summary>Restitution mixing: the larger one, so a bouncy object bounces off anything.</summary>
        /// <param name="restitution1">The first restitution.</param>
        /// <param name="restitution2">The second restitution.</param>
        /// <returns>The mixed restitution.</returns>
        internal static float b2MixRestitution(float restitution1, float restitution2) => restitution1 > restitution2 ? restitution1 : restitution2;

        /// <summary>The manifold, in place.</summary>
        /// <returns>A reference to the manifold.</returns>
        internal ref b2Manifold GetManifold() => ref m_manifold;

        /// <summary>The world manifold under the bodies' current transforms.</summary>
        /// <param name="worldManifold">Receives the world manifold.</param>
        internal void GetWorldManifold(ref b2WorldManifold worldManifold) {
            b2Body bodyA = m_fixtureA!.GetBody();
            b2Body bodyB = m_fixtureB!.GetBody();
            b2Shape shapeA = m_fixtureA.GetShape();
            b2Shape shapeB = m_fixtureB.GetShape();

            worldManifold.Initialize(m_manifold, bodyA.GetTransform(), shapeA.m_radius, bodyB.GetTransform(), shapeB.m_radius);
        }

        /// <summary>Enables or disables the contact for the current step (it is re-enabled at each update).</summary>
        /// <param name="flag">True to enable.</param>
        internal void SetEnabled(bool flag) {
            if (flag)
                m_flags |= e_enabledFlag;
            else
                m_flags &= ~e_enabledFlag;
        }

        /// <summary>Whether the contact is enabled.</summary>
        /// <returns>True when enabled.</returns>
        internal bool IsEnabled() => (m_flags & e_enabledFlag) == e_enabledFlag;

        /// <summary>Whether the shapes are touching.</summary>
        /// <returns>True when touching.</returns>
        internal bool IsTouching() => (m_flags & e_touchingFlag) == e_touchingFlag;

        /// <summary>The next contact of the world's list.</summary>
        /// <returns>The next contact, or null.</returns>
        internal b2Contact? GetNext() => m_next;

        /// <summary>Fixture A.</summary>
        /// <returns>The fixture.</returns>
        internal b2Fixture GetFixtureA() => m_fixtureA!;

        /// <summary>Fixture A's child index.</summary>
        /// <returns>The child index.</returns>
        internal int GetChildIndexA() => m_indexA;

        /// <summary>Fixture B.</summary>
        /// <returns>The fixture.</returns>
        internal b2Fixture GetFixtureB() => m_fixtureB!;

        /// <summary>Fixture B's child index.</summary>
        /// <returns>The child index.</returns>
        internal int GetChildIndexB() => m_indexB;

        /// <summary>Flags the contact to be re-filtered at the next step.</summary>
        internal void FlagForFiltering() => m_flags |= e_filterFlag;

        /// <summary>Resets the friction to the fixtures' mix.</summary>
        internal void ResetFriction() => m_friction = b2MixFriction(m_fixtureA!.m_friction, m_fixtureB!.m_friction);

        /// <summary>Resets the restitution to the fixtures' mix.</summary>
        internal void ResetRestitution() => m_restitution = b2MixRestitution(m_fixtureA!.m_restitution, m_fixtureB!.m_restitution);

        /// <summary>Computes the manifold of the two shapes under the given transforms.</summary>
        /// <param name="manifold">Receives the manifold.</param>
        /// <param name="xfA">Body A's transform.</param>
        /// <param name="xfB">Body B's transform.</param>
        internal abstract void Evaluate(ref b2Manifold manifold, in b2Transform xfA, in b2Transform xfB);

        /// <summary>Builds the factory table for the pairs of shape kinds this port collides.</summary>
        /// <returns>The table.</returns>
        private static b2ContactRegister[,] InitializeRegisters() {
            var registers = new b2ContactRegister[(int)b2ShapeType.e_typeCount, (int)b2ShapeType.e_typeCount];
            AddType(registers, (fA, iA, fB, iB) => new b2CircleContact(fA, fB), b2ShapeType.e_circle, b2ShapeType.e_circle);
            AddType(registers, (fA, iA, fB, iB) => new b2PolygonAndCircleContact(fA, fB), b2ShapeType.e_polygon, b2ShapeType.e_circle);
            AddType(registers, (fA, iA, fB, iB) => new b2PolygonContact(fA, fB), b2ShapeType.e_polygon, b2ShapeType.e_polygon);
            AddType(registers, (fA, iA, fB, iB) => new b2EdgeAndCircleContact(fA, fB), b2ShapeType.e_edge, b2ShapeType.e_circle);
            AddType(registers, (fA, iA, fB, iB) => new b2EdgeAndPolygonContact(fA, fB), b2ShapeType.e_edge, b2ShapeType.e_polygon);
            return registers;
        }

        /// <summary>Registers a factory for a pair of shape kinds and, swapped, for the reverse pair.</summary>
        /// <param name="registers">The table.</param>
        /// <param name="createFcn">The factory.</param>
        /// <param name="type1">The primary pair's first kind.</param>
        /// <param name="type2">The primary pair's second kind.</param>
        private static void AddType(b2ContactRegister[,] registers, Func<b2Fixture, int, b2Fixture, int, b2Contact> createFcn, b2ShapeType type1, b2ShapeType type2) {
            b2Assert(0 <= type1 && type1 < b2ShapeType.e_typeCount, "0 <= type1 && type1 < b2Shape::e_typeCount");
            b2Assert(0 <= type2 && type2 < b2ShapeType.e_typeCount, "0 <= type2 && type2 < b2Shape::e_typeCount");

            registers[(int)type1, (int)type2] = new b2ContactRegister(createFcn, true);

            if (type1 != type2)
                registers[(int)type2, (int)type1] = new b2ContactRegister(createFcn, false);
        }

        /// <summary>
        ///     Creates the contact for two fixture children through the factory table, swapping them when the pair is
        ///     registered the other way round; null for pairs that never collide (edge-edge).
        /// </summary>
        /// <param name="fixtureA">The first fixture.</param>
        /// <param name="indexA">Its child index.</param>
        /// <param name="fixtureB">The second fixture.</param>
        /// <param name="indexB">Its child index.</param>
        /// <returns>The contact, or null.</returns>
        internal static b2Contact? Create(b2Fixture fixtureA, int indexA, b2Fixture fixtureB, int indexB) {
            b2ShapeType type1 = fixtureA.GetShapeType();
            b2ShapeType type2 = fixtureB.GetShapeType();

            b2Assert(0 <= type1 && type1 < b2ShapeType.e_typeCount, "0 <= type1 && type1 < b2Shape::e_typeCount");
            b2Assert(0 <= type2 && type2 < b2ShapeType.e_typeCount, "0 <= type2 && type2 < b2Shape::e_typeCount");

            b2ContactRegister register = s_registers[(int)type1, (int)type2];
            if (register.createFcn != null) {
                if (register.primary)
                    return register.createFcn(fixtureA, indexA, fixtureB, indexB);
                else
                    return register.createFcn(fixtureB, indexB, fixtureA, indexA);
            }

            return null;
        }

        /// <summary>Destroys a contact: a touching, non-sensor contact wakes its bodies on the way out.</summary>
        /// <param name="contact">The contact.</param>
        internal static void Destroy(b2Contact contact) {
            b2Fixture fixtureA = contact.m_fixtureA!;
            b2Fixture fixtureB = contact.m_fixtureB!;

            if (contact.m_manifold.pointCount > 0 && fixtureA.IsSensor() == false && fixtureB.IsSensor() == false) {
                fixtureA.GetBody().SetAwake(true);
                fixtureB.GetBody().SetAwake(true);
            }
        }

        /// <summary>
        ///     Updates the manifold and touching state (sensors only test overlap), carries the warm-start impulses over by
        ///     contact id, wakes the bodies when touching changes, and calls the listener's Begin/End/PreSolve.
        /// </summary>
        /// <param name="listener">The contact listener, or null.</param>
        internal void Update(b2ContactListener? listener) {
            b2Manifold oldManifold = m_manifold;

            // Re-enable this contact.
            m_flags |= e_enabledFlag;

            bool touching = false;
            bool wasTouching = (m_flags & e_touchingFlag) == e_touchingFlag;

            bool sensorA = m_fixtureA!.IsSensor();
            bool sensorB = m_fixtureB!.IsSensor();
            bool sensor = sensorA || sensorB;

            b2Body bodyA = m_fixtureA.GetBody();
            b2Body bodyB = m_fixtureB.GetBody();
            ref readonly b2Transform xfA = ref bodyA.GetTransform();
            ref readonly b2Transform xfB = ref bodyB.GetTransform();

            // Is this contact a sensor?
            if (sensor) {
                b2Shape shapeA = m_fixtureA.GetShape();
                b2Shape shapeB = m_fixtureB.GetShape();
                touching = b2Collision.b2TestOverlap(shapeA, m_indexA, shapeB, m_indexB, xfA, xfB);

                // Sensors don't generate manifolds.
                m_manifold.pointCount = 0;
            } else {
                Evaluate(ref m_manifold, xfA, xfB);
                touching = m_manifold.pointCount > 0;

                // Match old contact ids to new contact ids and copy the
                // stored impulses to warm start the solver.
                for (int i = 0; i < m_manifold.pointCount; ++i) {
                    ref b2ManifoldPoint mp2 = ref m_manifold.points[i];
                    mp2.normalImpulse = 0.0f;
                    mp2.tangentImpulse = 0.0f;
                    b2ContactID id2 = mp2.id;

                    for (int j = 0; j < oldManifold.pointCount; ++j) {
                        ref b2ManifoldPoint mp1 = ref oldManifold.points[j];

                        if (mp1.id.key == id2.key) {
                            mp2.normalImpulse = mp1.normalImpulse;
                            mp2.tangentImpulse = mp1.tangentImpulse;
                            break;
                        }
                    }
                }

                if (touching != wasTouching) {
                    bodyA.SetAwake(true);
                    bodyB.SetAwake(true);
                }
            }

            if (touching)
                m_flags |= e_touchingFlag;
            else
                m_flags &= ~e_touchingFlag;

            if (wasTouching == false && touching == true && listener != null)
                listener.BeginContact(this);

            if (wasTouching == true && touching == false && listener != null)
                listener.EndContact(this);

            if (sensor == false && touching && listener != null)
                listener.PreSolve(this, oldManifold);
        }
    }

    /// <summary>A factory entry of the contact table: the constructor and whether the pair is registered in this order.</summary>
    /// <param name="createFcn">The factory.</param>
    /// <param name="primary">False when the fixtures must be swapped first.</param>
    internal readonly record struct b2ContactRegister(Func<b2Fixture, int, b2Fixture, int, b2Contact>? createFcn, bool primary);

    /// <summary>The contact of two circles.</summary>
    internal sealed class b2CircleContact : b2Contact {
        /// <summary>A circle-circle contact.</summary>
        /// <param name="fixtureA">The first circle.</param>
        /// <param name="fixtureB">The second circle.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown when a fixture is not a circle.</exception>
        public b2CircleContact(b2Fixture fixtureA, b2Fixture fixtureB) : base(fixtureA, 0, fixtureB, 0) {
            b2Assert(m_fixtureA!.GetShapeType() == b2ShapeType.e_circle, "m_fixtureA->GetType() == b2Shape::e_circle");
            b2Assert(m_fixtureB!.GetShapeType() == b2ShapeType.e_circle, "m_fixtureB->GetType() == b2Shape::e_circle");
        }

        /// <inheritdoc/>
        internal override void Evaluate(ref b2Manifold manifold, in b2Transform xfA, in b2Transform xfB) =>
            b2Collision.b2CollideCircles(ref manifold, (b2CircleShape)m_fixtureA!.GetShape(), xfA, (b2CircleShape)m_fixtureB!.GetShape(), xfB);
    }

    /// <summary>The contact of two polygons.</summary>
    internal sealed class b2PolygonContact : b2Contact {
        /// <summary>A polygon-polygon contact.</summary>
        /// <param name="fixtureA">The first polygon.</param>
        /// <param name="fixtureB">The second polygon.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown when a fixture is not a polygon.</exception>
        public b2PolygonContact(b2Fixture fixtureA, b2Fixture fixtureB) : base(fixtureA, 0, fixtureB, 0) {
            b2Assert(m_fixtureA!.GetShapeType() == b2ShapeType.e_polygon, "m_fixtureA->GetType() == b2Shape::e_polygon");
            b2Assert(m_fixtureB!.GetShapeType() == b2ShapeType.e_polygon, "m_fixtureB->GetType() == b2Shape::e_polygon");
        }

        /// <inheritdoc/>
        internal override void Evaluate(ref b2Manifold manifold, in b2Transform xfA, in b2Transform xfB) =>
            b2Collision.b2CollidePolygons(ref manifold, (b2PolygonShape)m_fixtureA!.GetShape(), xfA, (b2PolygonShape)m_fixtureB!.GetShape(), xfB);
    }

    /// <summary>The contact of a polygon (A) and a circle (B).</summary>
    internal sealed class b2PolygonAndCircleContact : b2Contact {
        /// <summary>A polygon-circle contact.</summary>
        /// <param name="fixtureA">The polygon.</param>
        /// <param name="fixtureB">The circle.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown for the wrong shape kinds.</exception>
        public b2PolygonAndCircleContact(b2Fixture fixtureA, b2Fixture fixtureB) : base(fixtureA, 0, fixtureB, 0) {
            b2Assert(m_fixtureA!.GetShapeType() == b2ShapeType.e_polygon, "m_fixtureA->GetType() == b2Shape::e_polygon");
            b2Assert(m_fixtureB!.GetShapeType() == b2ShapeType.e_circle, "m_fixtureB->GetType() == b2Shape::e_circle");
        }

        /// <inheritdoc/>
        internal override void Evaluate(ref b2Manifold manifold, in b2Transform xfA, in b2Transform xfB) =>
            b2Collision.b2CollidePolygonAndCircle(ref manifold, (b2PolygonShape)m_fixtureA!.GetShape(), xfA, (b2CircleShape)m_fixtureB!.GetShape(), xfB);
    }

    /// <summary>The contact of an edge (A) and a circle (B).</summary>
    internal sealed class b2EdgeAndCircleContact : b2Contact {
        /// <summary>An edge-circle contact.</summary>
        /// <param name="fixtureA">The edge.</param>
        /// <param name="fixtureB">The circle.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown for the wrong shape kinds.</exception>
        public b2EdgeAndCircleContact(b2Fixture fixtureA, b2Fixture fixtureB) : base(fixtureA, 0, fixtureB, 0) {
            b2Assert(m_fixtureA!.GetShapeType() == b2ShapeType.e_edge, "m_fixtureA->GetType() == b2Shape::e_edge");
            b2Assert(m_fixtureB!.GetShapeType() == b2ShapeType.e_circle, "m_fixtureB->GetType() == b2Shape::e_circle");
        }

        /// <inheritdoc/>
        internal override void Evaluate(ref b2Manifold manifold, in b2Transform xfA, in b2Transform xfB) =>
            b2Collision.b2CollideEdgeAndCircle(ref manifold, (b2EdgeShape)m_fixtureA!.GetShape(), xfA, (b2CircleShape)m_fixtureB!.GetShape(), xfB);
    }

    /// <summary>The contact of an edge (A) and a polygon (B).</summary>
    internal sealed class b2EdgeAndPolygonContact : b2Contact {
        /// <summary>An edge-polygon contact.</summary>
        /// <param name="fixtureA">The edge.</param>
        /// <param name="fixtureB">The polygon.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown for the wrong shape kinds.</exception>
        public b2EdgeAndPolygonContact(b2Fixture fixtureA, b2Fixture fixtureB) : base(fixtureA, 0, fixtureB, 0) {
            b2Assert(m_fixtureA!.GetShapeType() == b2ShapeType.e_edge, "m_fixtureA->GetType() == b2Shape::e_edge");
            b2Assert(m_fixtureB!.GetShapeType() == b2ShapeType.e_polygon, "m_fixtureB->GetType() == b2Shape::e_polygon");
        }

        /// <inheritdoc/>
        internal override void Evaluate(ref b2Manifold manifold, in b2Transform xfA, in b2Transform xfB) =>
            b2Collision.b2CollideEdgeAndPolygon(ref manifold, (b2EdgeShape)m_fixtureA!.GetShape(), xfA, (b2PolygonShape)m_fixtureB!.GetShape(), xfB);
    }
}
