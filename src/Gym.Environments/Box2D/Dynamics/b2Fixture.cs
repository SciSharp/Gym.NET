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
// Altered source version (Gym.NET, 2026-11-10): Box2D/Dynamics/b2Fixture.h and b2Fixture.cpp of pybox2d 2.3.10's
// bundled Box2D 2.3.0, translated to C#. A fixture proxy is a class (the broad-phase stores it as user data); the shape is
// cloned with the GC; GetType is GetShapeType; Dump is left out.
using static Gym.Box2D.b2Math;
using static Gym.Box2D.b2Settings;

namespace Gym.Box2D {
    /// <summary>
    ///     Collision filtering data: a fixture collides with another when each one's mask accepts the other's category,
    ///     unless they share a non-zero group (positive: always collide, negative: never).
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Dynamics/b2Fixture.h#L36">Box2D/Dynamics/b2Fixture.h</see>.</remarks>
    public struct b2Filter {
        /// <summary>The collision category bits; normally one bit set.</summary>
        public ushort categoryBits;

        /// <summary>The categories this fixture accepts collisions with.</summary>
        public ushort maskBits;

        /// <summary>The collision group (0: none).</summary>
        public short groupIndex;

        /// <summary>Box2D's default filter: category 0x0001, mask 0xFFFF, no group.</summary>
        public b2Filter() {
            categoryBits = 0x0001;
            maskBits = 0xFFFF;
            groupIndex = 0;
        }
    }

    /// <summary>
    ///     A fixture definition (pybox2d's <c>fixtureDef</c>): the shape and material a <see cref="b2Body"/> creates a
    ///     fixture from. The shape is cloned into the fixture, so one definition can create many fixtures.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Dynamics/b2Fixture.h#L58">Box2D/Dynamics/b2Fixture.h</see>.</remarks>
    public partial class b2FixtureDef {
        /// <summary>The shape, which must be set; it is cloned.</summary>
        internal b2Shape? shape;

        /// <summary>The fixture's user data.</summary>
        internal object? userData;

        /// <summary>The friction coefficient, usually in [0, 1] (default 0.2).</summary>
        internal float friction = 0.2f;

        /// <summary>The restitution (elasticity), usually in [0, 1] (default 0).</summary>
        internal float restitution = 0.0f;

        /// <summary>The density in kg/m² (default 0, a massless fixture).</summary>
        internal float density = 0.0f;

        /// <summary>Whether the fixture is a sensor: it detects contacts but produces no response.</summary>
        internal bool isSensor = false;

        /// <summary>The contact filtering data.</summary>
        internal b2Filter filter = new b2Filter();
    }

    /// <summary>A fixture's broad-phase proxy: one per child shape, stored in the tree as the proxy's user data.</summary>
    internal sealed class b2FixtureProxy {
        /// <summary>The child's box at the last synchronization.</summary>
        public b2AABB aabb;

        /// <summary>The fixture.</summary>
        public b2Fixture? fixture;

        /// <summary>The child index.</summary>
        public int childIndex;

        /// <summary>The broad-phase proxy id.</summary>
        public int proxyId;
    }

    /// <summary>
    ///     A shape attached to a body, with its density, friction, restitution and filter. Created by
    ///     <see cref="b2Body"/>, never directly; destroying its body destroys it.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Dynamics/b2Fixture.cpp">Box2D/Dynamics/b2Fixture.cpp</see>.</remarks>
    public partial class b2Fixture {
        /// <summary>The density.</summary>
        internal float m_density;

        /// <summary>The next fixture of the body's list.</summary>
        internal b2Fixture? m_next;

        /// <summary>The body.</summary>
        internal b2Body? m_body;

        /// <summary>The fixture's own clone of the shape.</summary>
        internal b2Shape? m_shape;

        /// <summary>The friction.</summary>
        internal float m_friction;

        /// <summary>The restitution.</summary>
        internal float m_restitution;

        /// <summary>One proxy per child shape.</summary>
        internal b2FixtureProxy[] m_proxies = System.Array.Empty<b2FixtureProxy>();

        /// <summary>The number of proxies in the broad-phase (0 while the body is inactive).</summary>
        internal int m_proxyCount;

        /// <summary>The filter.</summary>
        internal b2Filter m_filter;

        /// <summary>Whether this is a sensor.</summary>
        internal bool m_isSensor;

        /// <summary>The user data.</summary>
        internal object? m_userData;

        /// <summary>An unattached fixture; <see cref="Create"/> attaches it.</summary>
        internal b2Fixture() {
            m_userData = null;
            m_body = null;
            m_next = null;
            m_proxyCount = 0;
            m_shape = null;
            m_density = 0.0f;
        }

        /// <summary>The shape kind (C++ <c>GetType</c>).</summary>
        /// <returns>The kind.</returns>
        internal b2ShapeType GetShapeType() => m_shape!.GetShapeType();

        /// <summary>The fixture's shape.</summary>
        /// <returns>The shape.</returns>
        internal b2Shape GetShape() => m_shape!;

        /// <summary>Whether this is a sensor.</summary>
        /// <returns>True for a sensor.</returns>
        internal bool IsSensor() => m_isSensor;

        /// <summary>The filter.</summary>
        /// <returns>The filter.</returns>
        internal b2Filter GetFilterData() => m_filter;

        /// <summary>The body.</summary>
        /// <returns>The body.</returns>
        internal b2Body GetBody() => m_body!;

        /// <summary>The next fixture of the body.</summary>
        /// <returns>The next fixture, or null.</returns>
        internal b2Fixture? GetNext() => m_next;

        /// <summary>Sets the density; mass changes take effect at the body's next <c>ResetMassData</c>.</summary>
        /// <param name="density">The density.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown for a negative or non-finite density.</exception>
        internal void SetDensity(float density) {
            b2Assert(b2IsValid(density) && density >= 0.0f);
            m_density = density;
        }

        /// <summary>Whether a world point lies inside the fixture (pybox2d's <c>TestPoint</c>); never true for an edge.</summary>
        /// <param name="p">The point.</param>
        /// <returns>True when inside.</returns>
        public bool TestPoint(b2Vec2 p) => m_shape!.TestPoint(m_body!.GetTransform(), p);

        /// <summary>Casts a ray against a child shape.</summary>
        /// <param name="output">Receives the hit.</param>
        /// <param name="input">The ray.</param>
        /// <param name="childIndex">The child index.</param>
        /// <returns>True on a hit.</returns>
        internal bool RayCast(ref b2RayCastOutput output, in b2RayCastInput input, int childIndex) => m_shape!.RayCast(ref output, input, m_body!.GetTransform(), childIndex);

        /// <summary>The fixture's mass properties at its density.</summary>
        /// <param name="massData">Receives them.</param>
        internal void GetMassData(out b2MassData massData) => m_shape!.ComputeMass(out massData, m_density);

        /// <summary>A child's box at the last synchronization.</summary>
        /// <param name="childIndex">The child index.</param>
        /// <returns>The box.</returns>
        /// <exception cref="NumSharp.AssertionError">Thrown for a child index out of range.</exception>
        internal b2AABB GetAABB(int childIndex) {
            b2Assert(0 <= childIndex && childIndex < m_proxyCount);
            return m_proxies[childIndex].aabb;
        }

        /// <summary>Attaches the fixture to a body from a definition, cloning the shape and reserving one proxy per child.</summary>
        /// <param name="body">The body.</param>
        /// <param name="def">The definition.</param>
        internal void Create(b2Body body, b2FixtureDef def) {
            m_userData = def.userData;
            m_friction = def.friction;
            m_restitution = def.restitution;

            m_body = body;
            m_next = null;

            m_filter = def.filter;

            m_isSensor = def.isSensor;

            m_shape = def.shape!.Clone();

            // Reserve proxy space
            int childCount = m_shape.GetChildCount();
            m_proxies = new b2FixtureProxy[childCount];
            for (int i = 0; i < childCount; ++i) {
                m_proxies[i] = new b2FixtureProxy {
                    fixture = null,
                    proxyId = b2BroadPhase.e_nullProxy,
                };
            }

            m_proxyCount = 0;

            m_density = def.density;
        }

        /// <summary>Releases the shape and proxies; the proxies must already be out of the broad-phase.</summary>
        /// <exception cref="NumSharp.AssertionError">Thrown while proxies remain in the broad-phase.</exception>
        internal void Destroy() {
            // The proxies must be destroyed before calling this.
            b2Assert(m_proxyCount == 0);

            // Free the proxy array.
            m_proxies = System.Array.Empty<b2FixtureProxy>();

            m_shape = null;
        }

        /// <summary>Creates the fixture's proxies in the broad-phase at a transform.</summary>
        /// <param name="broadPhase">The broad-phase.</param>
        /// <param name="xf">The body's transform.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown when the proxies already exist.</exception>
        internal void CreateProxies(b2BroadPhase broadPhase, in b2Transform xf) {
            b2Assert(m_proxyCount == 0);

            // Create proxies in the broad-phase.
            m_proxyCount = m_shape!.GetChildCount();

            for (int i = 0; i < m_proxyCount; ++i) {
                b2FixtureProxy proxy = m_proxies[i];
                m_shape.ComputeAABB(out proxy.aabb, xf, i);
                proxy.proxyId = broadPhase.CreateProxy(proxy.aabb, proxy);
                proxy.fixture = this;
                proxy.childIndex = i;
            }
        }

        /// <summary>Removes the fixture's proxies from the broad-phase.</summary>
        /// <param name="broadPhase">The broad-phase.</param>
        internal void DestroyProxies(b2BroadPhase broadPhase) {
            // Destroy proxies in the broad-phase.
            for (int i = 0; i < m_proxyCount; ++i) {
                b2FixtureProxy proxy = m_proxies[i];
                broadPhase.DestroyProxy(proxy.proxyId);
                proxy.proxyId = b2BroadPhase.e_nullProxy;
            }

            m_proxyCount = 0;
        }

        /// <summary>Moves the proxies to cover the swept shape between two transforms.</summary>
        /// <param name="broadPhase">The broad-phase.</param>
        /// <param name="transform1">The transform at the start of the step.</param>
        /// <param name="transform2">The transform at the end.</param>
        internal void Synchronize(b2BroadPhase broadPhase, in b2Transform transform1, in b2Transform transform2) {
            if (m_proxyCount == 0)
                return;

            for (int i = 0; i < m_proxyCount; ++i) {
                b2FixtureProxy proxy = m_proxies[i];

                // Compute an AABB that covers the swept shape (may miss some rotation effect).
                m_shape!.ComputeAABB(out b2AABB aabb1, transform1, proxy.childIndex);
                m_shape.ComputeAABB(out b2AABB aabb2, transform2, proxy.childIndex);

                proxy.aabb.Combine(aabb1, aabb2);

                b2Vec2 displacement = transform2.p - transform1.p;

                broadPhase.MoveProxy(proxy.proxyId, proxy.aabb, displacement);
            }
        }

        /// <summary>Sets the filter and flags the fixture's contacts for re-filtering at the next step.</summary>
        /// <param name="filter">The filter.</param>
        internal void SetFilterData(in b2Filter filter) {
            m_filter = filter;

            Refilter();
        }

        /// <summary>Flags the fixture's contacts for filtering and touches its proxies so new pairs may form.</summary>
        internal void Refilter() {
            if (m_body == null)
                return;

            // Flag associated contacts for filtering.
            b2ContactEdge? edge = m_body.GetContactList();
            while (edge != null) {
                b2Contact contact = edge.contact!;
                b2Fixture fixtureA = contact.GetFixtureA();
                b2Fixture fixtureB = contact.GetFixtureB();
                if (fixtureA == this || fixtureB == this)
                    contact.FlagForFiltering();

                edge = edge.next;
            }

            b2World? world = m_body.GetWorld();

            if (world == null)
                return;

            // Touch each proxy so that new pairs may be created
            b2BroadPhase broadPhase = world.m_contactManager.m_broadPhase;
            for (int i = 0; i < m_proxyCount; ++i)
                broadPhase.TouchProxy(m_proxies[i].proxyId);
        }

        /// <summary>Turns the fixture into a sensor or back, waking the body when it changes.</summary>
        /// <param name="sensor">Whether it is a sensor.</param>
        internal void SetSensor(bool sensor) {
            if (sensor != m_isSensor) {
                m_body!.SetAwake(true);
                m_isSensor = sensor;
            }
        }
    }
}
