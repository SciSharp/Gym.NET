/*
* Copyright (c) 2006-2011 Erin Catto http://www.box2d.org
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
// Altered source version (Gym.NET, 2026-11-10): Box2D/Dynamics/b2Body.h and b2Body.cpp of pybox2d 2.3.10's bundled
// Box2D 2.3.0, translated to C#. The C++ API lives in the internal members below (GetType is GetBodyType, as C# objects
// already have a GetType); the pybox2d-shaped public surface is in Facade/b2Body.Facade.cs; Dump is left out.
using static Gym.Box2D.b2Math;
using static Gym.Box2D.b2Settings;

namespace Gym.Box2D {
    /// <summary>The body type: static (zero mass, zero velocity, may be moved manually), kinematic, or dynamic.</summary>
    public enum b2BodyType {
        /// <summary>Zero mass and velocity; moved only manually.</summary>
        b2_staticBody = 0,

        /// <summary>Zero mass, velocity set by the user, moved by the solver.</summary>
        b2_kinematicBody,

        /// <summary>Positive mass, velocity determined by forces, moved by the solver.</summary>
        b2_dynamicBody,
    }

    /// <summary>A body definition: everything a body is created from (pybox2d's <c>b2BodyDef</c>). Reusable.</summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Dynamics/b2Body.h#L46">Box2D/Dynamics/b2Body.h</see>.</remarks>
    public partial class b2BodyDef {
        /// <summary>The body type (default static).</summary>
        internal b2BodyType type = b2BodyType.b2_staticBody;

        /// <summary>The world position of the body origin.</summary>
        internal b2Vec2 position = new b2Vec2(0.0f, 0.0f);

        /// <summary>The world angle in radians.</summary>
        internal float angle = 0.0f;

        /// <summary>The linear velocity of the body origin.</summary>
        internal b2Vec2 linearVelocity = new b2Vec2(0.0f, 0.0f);

        /// <summary>The angular velocity.</summary>
        internal float angularVelocity = 0.0f;

        /// <summary>The linear damping.</summary>
        internal float linearDamping = 0.0f;

        /// <summary>The angular damping.</summary>
        internal float angularDamping = 0.0f;

        /// <summary>Whether the body may fall asleep.</summary>
        internal bool allowSleep = true;

        /// <summary>Whether the body starts awake.</summary>
        internal bool awake = true;

        /// <summary>Whether the body may not rotate.</summary>
        internal bool fixedRotation = false;

        /// <summary>Whether the body is a fast "bullet" with continuous collision against dynamic bodies too.</summary>
        internal bool bullet = false;

        /// <summary>Whether the body starts active (simulated and colliding).</summary>
        internal bool active = true;

        /// <summary>The body's user data.</summary>
        internal object? userData;

        /// <summary>The scale of the world's gravity on this body.</summary>
        internal float gravityScale = 1.0f;
    }

    /// <summary>
    ///     A rigid body. Created by <see cref="b2World"/>, never directly; its fixtures, contacts and joints die with it.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Dynamics/b2Body.cpp">Box2D/Dynamics/b2Body.cpp</see>.</remarks>
    public partial class b2Body {
        /// <summary>Set while the body is in the island being built.</summary>
        internal const ushort e_islandFlag = 0x0001;

        /// <summary>Set while the body is awake.</summary>
        internal const ushort e_awakeFlag = 0x0002;

        /// <summary>Set when the body may fall asleep.</summary>
        internal const ushort e_autoSleepFlag = 0x0004;

        /// <summary>Set for a bullet.</summary>
        internal const ushort e_bulletFlag = 0x0008;

        /// <summary>Set when the body may not rotate.</summary>
        internal const ushort e_fixedRotationFlag = 0x0010;

        /// <summary>Set while the body is active.</summary>
        internal const ushort e_activeFlag = 0x0020;

        /// <summary>Set during time of impact (unused by 2.3's solver, kept from C++).</summary>
        internal const ushort e_toiFlag = 0x0040;

        /// <summary>The body type.</summary>
        internal b2BodyType m_type;

        /// <summary>The flags.</summary>
        internal ushort m_flags;

        /// <summary>The body's index in the island being solved.</summary>
        internal int m_islandIndex;

        /// <summary>The body origin transform.</summary>
        internal b2Transform m_xf;

        /// <summary>The swept motion for continuous collision.</summary>
        internal b2Sweep m_sweep;

        /// <summary>The linear velocity of the center of mass.</summary>
        internal b2Vec2 m_linearVelocity;

        /// <summary>The angular velocity.</summary>
        internal float m_angularVelocity;

        /// <summary>The force accumulated for the next step.</summary>
        internal b2Vec2 m_force;

        /// <summary>The torque accumulated for the next step.</summary>
        internal float m_torque;

        /// <summary>The world.</summary>
        internal b2World m_world;

        /// <summary>The previous body of the world's list.</summary>
        internal b2Body? m_prev;

        /// <summary>The next body of the world's list.</summary>
        internal b2Body? m_next;

        /// <summary>The fixture list, newest first.</summary>
        internal b2Fixture? m_fixtureList;

        /// <summary>The number of fixtures.</summary>
        internal int m_fixtureCount;

        /// <summary>The joint edges, newest first.</summary>
        internal b2JointEdge? m_jointList;

        /// <summary>The contact edges, newest first.</summary>
        internal b2ContactEdge? m_contactList;

        /// <summary>The mass.</summary>
        internal float m_mass;

        /// <summary>The inverse mass.</summary>
        internal float m_invMass;

        /// <summary>The rotational inertia about the center of mass.</summary>
        internal float m_I;

        /// <summary>The inverse inertia.</summary>
        internal float m_invI;

        /// <summary>The linear damping.</summary>
        internal float m_linearDamping;

        /// <summary>The angular damping.</summary>
        internal float m_angularDamping;

        /// <summary>The gravity scale.</summary>
        internal float m_gravityScale;

        /// <summary>How long the body has been still enough to sleep.</summary>
        internal float m_sleepTime;

        /// <summary>The user data.</summary>
        internal object? m_userData;

        /// <summary>A body from a definition, in a world (called by <see cref="b2World"/>).</summary>
        /// <param name="bd">The definition.</param>
        /// <param name="world">The world.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown for a non-finite position, velocity or angle, or negative damping.</exception>
        internal b2Body(b2BodyDef bd, b2World world) {
            b2Assert(bd.position.IsValid(), "bd->position.IsValid()");
            b2Assert(bd.linearVelocity.IsValid(), "bd->linearVelocity.IsValid()");
            b2Assert(b2IsValid(bd.angle), "b2IsValid(bd->angle)");
            b2Assert(b2IsValid(bd.angularVelocity), "b2IsValid(bd->angularVelocity)");
            b2Assert(b2IsValid(bd.angularDamping) && bd.angularDamping >= 0.0f, "b2IsValid(bd->angularDamping) && bd->angularDamping >= 0.0f");
            b2Assert(b2IsValid(bd.linearDamping) && bd.linearDamping >= 0.0f, "b2IsValid(bd->linearDamping) && bd->linearDamping >= 0.0f");

            m_flags = 0;

            if (bd.bullet)
                m_flags |= e_bulletFlag;
            if (bd.fixedRotation)
                m_flags |= e_fixedRotationFlag;
            if (bd.allowSleep)
                m_flags |= e_autoSleepFlag;
            if (bd.awake)
                m_flags |= e_awakeFlag;
            if (bd.active)
                m_flags |= e_activeFlag;

            m_world = world;

            m_xf.p = bd.position;
            m_xf.q.Set(bd.angle);

            m_sweep.localCenter.SetZero();
            m_sweep.c0 = m_xf.p;
            m_sweep.c = m_xf.p;
            m_sweep.a0 = bd.angle;
            m_sweep.a = bd.angle;
            m_sweep.alpha0 = 0.0f;

            m_jointList = null;
            m_contactList = null;
            m_prev = null;
            m_next = null;

            m_linearVelocity = bd.linearVelocity;
            m_angularVelocity = bd.angularVelocity;

            m_linearDamping = bd.linearDamping;
            m_angularDamping = bd.angularDamping;
            m_gravityScale = bd.gravityScale;

            m_force.SetZero();
            m_torque = 0.0f;

            m_sleepTime = 0.0f;

            m_type = bd.type;

            if (m_type == b2BodyType.b2_dynamicBody) {
                m_mass = 1.0f;
                m_invMass = 1.0f;
            } else {
                m_mass = 0.0f;
                m_invMass = 0.0f;
            }

            m_I = 0.0f;
            m_invI = 0.0f;

            m_userData = bd.userData;

            m_fixtureList = null;
            m_fixtureCount = 0;
        }

        /// <summary>The body type (C++ <c>GetType</c>).</summary>
        /// <returns>The type.</returns>
        internal b2BodyType GetBodyType() => m_type;

        /// <summary>The body origin transform.</summary>
        /// <returns>The transform.</returns>
        internal ref readonly b2Transform GetTransform() => ref m_xf;

        /// <summary>The body origin's world position.</summary>
        /// <returns>The position.</returns>
        internal b2Vec2 GetPosition() => m_xf.p;

        /// <summary>The body's angle in radians (the sweep's angle, not normalized).</summary>
        /// <returns>The angle.</returns>
        internal float GetAngle() => m_sweep.a;

        /// <summary>The center of mass in world coordinates.</summary>
        /// <returns>The center.</returns>
        internal b2Vec2 GetWorldCenter() => m_sweep.c;

        /// <summary>The center of mass in local coordinates.</summary>
        /// <returns>The local center.</returns>
        internal b2Vec2 GetLocalCenter() => m_sweep.localCenter;

        /// <summary>Sets the center of mass's linear velocity; a static body ignores it, a non-zero one wakes the body.</summary>
        /// <param name="v">The velocity.</param>
        internal void SetLinearVelocity(in b2Vec2 v) {
            if (m_type == b2BodyType.b2_staticBody)
                return;

            if (b2Dot(v, v) > 0.0f)
                SetAwake(true);

            m_linearVelocity = v;
        }

        /// <summary>The center of mass's linear velocity.</summary>
        /// <returns>The velocity.</returns>
        internal b2Vec2 GetLinearVelocity() => m_linearVelocity;

        /// <summary>Sets the angular velocity; a static body ignores it, a non-zero one wakes the body.</summary>
        /// <param name="w">The angular velocity.</param>
        internal void SetAngularVelocity(float w) {
            if (m_type == b2BodyType.b2_staticBody)
                return;

            if (w * w > 0.0f)
                SetAwake(true);

            m_angularVelocity = w;
        }

        /// <summary>The angular velocity.</summary>
        /// <returns>The angular velocity.</returns>
        internal float GetAngularVelocity() => m_angularVelocity;

        /// <summary>The mass.</summary>
        /// <returns>The mass.</returns>
        internal float GetMass() => m_mass;

        /// <summary>The rotational inertia about the body origin.</summary>
        /// <returns>The inertia.</returns>
        internal float GetInertia() => m_I + m_mass * b2Dot(m_sweep.localCenter, m_sweep.localCenter);

        /// <summary>The mass data about the body origin.</summary>
        /// <param name="data">Receives the mass data.</param>
        internal void GetMassData(out b2MassData data) {
            data.mass = m_mass;
            data.I = m_I + m_mass * b2Dot(m_sweep.localCenter, m_sweep.localCenter);
            data.center = m_sweep.localCenter;
        }

        /// <summary>A local point in world coordinates.</summary>
        /// <param name="localPoint">The local point.</param>
        /// <returns>The world point.</returns>
        public b2Vec2 GetWorldPoint(in b2Vec2 localPoint) => b2Mul(m_xf, localPoint);

        /// <summary>A local vector in world coordinates.</summary>
        /// <param name="localVector">The local vector.</param>
        /// <returns>The world vector.</returns>
        public b2Vec2 GetWorldVector(in b2Vec2 localVector) => b2Mul(m_xf.q, localVector);

        /// <summary>A world point in local coordinates.</summary>
        /// <param name="worldPoint">The world point.</param>
        /// <returns>The local point.</returns>
        public b2Vec2 GetLocalPoint(in b2Vec2 worldPoint) => b2MulT(m_xf, worldPoint);

        /// <summary>A world vector in local coordinates.</summary>
        /// <param name="worldVector">The world vector.</param>
        /// <returns>The local vector.</returns>
        public b2Vec2 GetLocalVector(in b2Vec2 worldVector) => b2MulT(m_xf.q, worldVector);

        /// <summary>The velocity of a world point attached to the body.</summary>
        /// <param name="worldPoint">The world point.</param>
        /// <returns>The velocity.</returns>
        public b2Vec2 GetLinearVelocityFromWorldPoint(in b2Vec2 worldPoint) => m_linearVelocity + b2Cross(m_angularVelocity, worldPoint - m_sweep.c);

        /// <summary>The velocity of a local point attached to the body.</summary>
        /// <param name="localPoint">The local point.</param>
        /// <returns>The velocity.</returns>
        public b2Vec2 GetLinearVelocityFromLocalPoint(in b2Vec2 localPoint) => GetLinearVelocityFromWorldPoint(GetWorldPoint(localPoint));

        /// <summary>Sets whether the body is a bullet.</summary>
        /// <param name="flag">True for a bullet.</param>
        internal void SetBullet(bool flag) {
            if (flag)
                m_flags |= e_bulletFlag;
            else
                m_flags &= unchecked((ushort)~e_bulletFlag);
        }

        /// <summary>Whether the body is a bullet.</summary>
        /// <returns>True for a bullet.</returns>
        internal bool IsBullet() => (m_flags & e_bulletFlag) == e_bulletFlag;

        /// <summary>
        ///     Wakes the body (resetting its sleep time) or puts it to sleep, which zeroes its velocities, force and torque.
        /// </summary>
        /// <param name="flag">True to wake, false to sleep.</param>
        internal void SetAwake(bool flag) {
            if (flag) {
                if ((m_flags & e_awakeFlag) == 0) {
                    m_flags |= e_awakeFlag;
                    m_sleepTime = 0.0f;
                }
            } else {
                m_flags &= unchecked((ushort)~e_awakeFlag);
                m_sleepTime = 0.0f;
                m_linearVelocity.SetZero();
                m_angularVelocity = 0.0f;
                m_force.SetZero();
                m_torque = 0.0f;
            }
        }

        /// <summary>Whether the body is awake.</summary>
        /// <returns>True when awake.</returns>
        internal bool IsAwake() => (m_flags & e_awakeFlag) == e_awakeFlag;

        /// <summary>Whether the body is active.</summary>
        /// <returns>True when active.</returns>
        internal bool IsActive() => (m_flags & e_activeFlag) == e_activeFlag;

        /// <summary>Whether the body has fixed rotation.</summary>
        /// <returns>True when it may not rotate.</returns>
        internal bool IsFixedRotation() => (m_flags & e_fixedRotationFlag) == e_fixedRotationFlag;

        /// <summary>Allows or forbids sleeping; forbidding wakes the body.</summary>
        /// <param name="flag">True to allow sleeping.</param>
        internal void SetSleepingAllowed(bool flag) {
            if (flag) {
                m_flags |= e_autoSleepFlag;
            } else {
                m_flags &= unchecked((ushort)~e_autoSleepFlag);
                SetAwake(true);
            }
        }

        /// <summary>Whether the body may sleep.</summary>
        /// <returns>True when sleeping is allowed.</returns>
        internal bool IsSleepingAllowed() => (m_flags & e_autoSleepFlag) == e_autoSleepFlag;

        /// <summary>The fixture list, newest first.</summary>
        /// <returns>The first fixture, or null.</returns>
        internal b2Fixture? GetFixtureList() => m_fixtureList;

        /// <summary>The joint edges, newest first.</summary>
        /// <returns>The first edge, or null.</returns>
        internal b2JointEdge? GetJointList() => m_jointList;

        /// <summary>The contact edges, newest first.</summary>
        /// <returns>The first edge, or null.</returns>
        internal b2ContactEdge? GetContactList() => m_contactList;

        /// <summary>The next body of the world's list.</summary>
        /// <returns>The next body, or null.</returns>
        internal b2Body? GetNext() => m_next;

        /// <summary>The world.</summary>
        /// <returns>The world.</returns>
        internal b2World GetWorld() => m_world;

        /// <summary>
        ///     Adds a force at a world point (wakes the body when <paramref name="wake"/>); a sleeping body that is not woken
        ///     ignores it, and only dynamic bodies take forces.
        /// </summary>
        /// <param name="force">The force in newtons.</param>
        /// <param name="point">The world point of application.</param>
        /// <param name="wake">Whether to wake the body.</param>
        public void ApplyForce(in b2Vec2 force, in b2Vec2 point, bool wake) {
            if (m_type != b2BodyType.b2_dynamicBody)
                return;

            if (wake && (m_flags & e_awakeFlag) == 0)
                SetAwake(true);

            // Don't accumulate a force if the body is sleeping.
            if ((m_flags & e_awakeFlag) != 0) {
                m_force += force;
                m_torque += b2Cross(point - m_sweep.c, force);
            }
        }

        /// <summary>Adds a force at the center of mass.</summary>
        /// <param name="force">The force in newtons.</param>
        /// <param name="wake">Whether to wake the body.</param>
        public void ApplyForceToCenter(in b2Vec2 force, bool wake) {
            if (m_type != b2BodyType.b2_dynamicBody)
                return;

            if (wake && (m_flags & e_awakeFlag) == 0)
                SetAwake(true);

            // Don't accumulate a force if the body is sleeping
            if ((m_flags & e_awakeFlag) != 0)
                m_force += force;
        }

        /// <summary>Adds a torque about the z-axis.</summary>
        /// <param name="torque">The torque in N·m.</param>
        /// <param name="wake">Whether to wake the body.</param>
        public void ApplyTorque(float torque, bool wake) {
            if (m_type != b2BodyType.b2_dynamicBody)
                return;

            if (wake && (m_flags & e_awakeFlag) == 0)
                SetAwake(true);

            // Don't accumulate a force if the body is sleeping
            if ((m_flags & e_awakeFlag) != 0)
                m_torque += torque;
        }

        /// <summary>Applies an impulse at a world point: the velocities change at once.</summary>
        /// <param name="impulse">The impulse in N·s.</param>
        /// <param name="point">The world point of application.</param>
        /// <param name="wake">Whether to wake the body.</param>
        public void ApplyLinearImpulse(in b2Vec2 impulse, in b2Vec2 point, bool wake) {
            if (m_type != b2BodyType.b2_dynamicBody)
                return;

            if (wake && (m_flags & e_awakeFlag) == 0)
                SetAwake(true);

            // Don't accumulate velocity if the body is sleeping
            if ((m_flags & e_awakeFlag) != 0) {
                m_linearVelocity += m_invMass * impulse;
                m_angularVelocity += m_invI * b2Cross(point - m_sweep.c, impulse);
            }
        }

        /// <summary>Applies an angular impulse.</summary>
        /// <param name="impulse">The angular impulse in kg·m²/s.</param>
        /// <param name="wake">Whether to wake the body.</param>
        public void ApplyAngularImpulse(float impulse, bool wake) {
            if (m_type != b2BodyType.b2_dynamicBody)
                return;

            if (wake && (m_flags & e_awakeFlag) == 0)
                SetAwake(true);

            // Don't accumulate velocity if the body is sleeping
            if ((m_flags & e_awakeFlag) != 0)
                m_angularVelocity += m_invI * impulse;
        }

        /// <summary>Recomputes the transform from the sweep's end.</summary>
        internal void SynchronizeTransform() {
            m_xf.q.Set(m_sweep.a);
            m_xf.p = m_sweep.c - b2Mul(m_xf.q, m_sweep.localCenter);
        }

        /// <summary>Advances the body to a fraction of the step (time of impact), leaving it at the sweep's new start.</summary>
        /// <param name="alpha">The fraction.</param>
        internal void Advance(float alpha) {
            // Advance to the new safe time. This doesn't sync the broad-phase.
            m_sweep.Advance(alpha);
            m_sweep.c = m_sweep.c0;
            m_sweep.a = m_sweep.a0;
            m_xf.q.Set(m_sweep.a);
            m_xf.p = m_sweep.c - b2Mul(m_xf.q, m_sweep.localCenter);
        }

        /// <summary>
        ///     Changes the body type: mass is recomputed, a static body is stopped, the attached contacts are destroyed and the
        ///     proxies touched so contacts are recreated as appropriate.
        /// </summary>
        /// <param name="type">The new type.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown while the world is stepping.</exception>
        internal void SetType(b2BodyType type) {
            b2Assert(m_world.IsLocked() == false, "m_world->IsLocked() == false");
            if (m_world.IsLocked() == true)
                return;

            if (m_type == type)
                return;

            m_type = type;

            ResetMassData();

            if (m_type == b2BodyType.b2_staticBody) {
                m_linearVelocity.SetZero();
                m_angularVelocity = 0.0f;
                m_sweep.a0 = m_sweep.a;
                m_sweep.c0 = m_sweep.c;
                SynchronizeFixtures();
            }

            SetAwake(true);

            m_force.SetZero();
            m_torque = 0.0f;

            // Delete the attached contacts.
            b2ContactEdge? ce = m_contactList;
            while (ce != null) {
                b2ContactEdge ce0 = ce;
                ce = ce.next;
                m_world.m_contactManager.Destroy(ce0.contact!);
            }

            m_contactList = null;

            // Touch the proxies so that new contacts will be created (when appropriate)
            b2BroadPhase broadPhase = m_world.m_contactManager.m_broadPhase;
            for (b2Fixture? f = m_fixtureList; f != null; f = f.m_next) {
                int proxyCount = f.m_proxyCount;
                for (int i = 0; i < proxyCount; ++i)
                    broadPhase.TouchProxy(f.m_proxies[i].proxyId);
            }
        }

        /// <summary>
        ///     Creates a fixture from a definition and attaches it (newest first); a dense fixture updates the mass, and the
        ///     world finds its contacts at the next step.
        /// </summary>
        /// <param name="def">The definition; its shape is cloned.</param>
        /// <returns>The fixture.</returns>
        /// <exception cref="NumSharp.AssertionError">Thrown while the world is stepping.</exception>
        public b2Fixture CreateFixture(b2FixtureDef def) {
            b2Assert(m_world.IsLocked() == false, "m_world->IsLocked() == false");

            var fixture = new b2Fixture();
            fixture.Create(this, def);

            if ((m_flags & e_activeFlag) != 0) {
                b2BroadPhase broadPhase = m_world.m_contactManager.m_broadPhase;
                fixture.CreateProxies(broadPhase, m_xf);
            }

            fixture.m_next = m_fixtureList;
            m_fixtureList = fixture;
            ++m_fixtureCount;

            fixture.m_body = this;

            // Adjust mass properties if needed.
            if (fixture.m_density > 0.0f)
                ResetMassData();

            // Let the world know we have a new fixture. This will cause new contacts
            // to be created at the beginning of the next time step.
            m_world.m_flags |= b2World.e_newFixture;

            return fixture;
        }

        /// <summary>Creates a fixture from a shape and density, with the definition's other defaults.</summary>
        /// <param name="shape">The shape; it is cloned.</param>
        /// <param name="density">The density.</param>
        /// <returns>The fixture.</returns>
        /// <exception cref="NumSharp.AssertionError">Thrown while the world is stepping.</exception>
        internal b2Fixture CreateFixture(b2Shape shape, float density) {
            var def = new b2FixtureDef {
                shape = shape,
                density = density,
            };
            return CreateFixture(def);
        }

        /// <summary>Destroys a fixture and its contacts, then recomputes the mass.</summary>
        /// <param name="fixture">The fixture.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown while the world is stepping, or for a fixture of another body.</exception>
        public void DestroyFixture(b2Fixture fixture) {
            b2Assert(m_world.IsLocked() == false, "m_world->IsLocked() == false");
            if (m_world.IsLocked() == true)
                return;

            b2Assert(fixture.m_body == this, "fixture->m_body == this");

            // Remove the fixture from this body's singly linked list.
            b2Assert(m_fixtureCount > 0);
            bool found = false;
            if (m_fixtureList == fixture) {
                m_fixtureList = fixture.m_next;
                found = true;
            } else {
                for (b2Fixture? node = m_fixtureList; node != null; node = node.m_next) {
                    if (node.m_next == fixture) {
                        node.m_next = fixture.m_next;
                        found = true;
                        break;
                    }
                }
            }

            // You tried to remove a shape that is not attached to this body.
            b2Assert(found);

            // Destroy any contacts associated with the fixture.
            b2ContactEdge? edge = m_contactList;
            while (edge != null) {
                b2Contact c = edge.contact!;
                edge = edge.next;

                b2Fixture fixtureA = c.GetFixtureA();
                b2Fixture fixtureB = c.GetFixtureB();

                if (fixture == fixtureA || fixture == fixtureB)
                    // This destroys the contact and removes it from
                    // this body's contact list.
                    m_world.m_contactManager.Destroy(c);
            }

            if ((m_flags & e_activeFlag) != 0) {
                b2BroadPhase broadPhase = m_world.m_contactManager.m_broadPhase;
                fixture.DestroyProxies(broadPhase);
            }

            fixture.Destroy();
            fixture.m_body = null;
            fixture.m_next = null;

            --m_fixtureCount;

            // Reset the mass data.
            ResetMassData();
        }

        /// <summary>
        ///     Recomputes the mass, center and inertia from the fixtures' densities; static and kinematic bodies get zero
        ///     mass, a dynamic body without dense fixtures a mass of 1, and the center's velocity follows the moved center.
        /// </summary>
        /// <exception cref="NumSharp.AssertionError">Thrown when a dynamic body's centered inertia is not positive.</exception>
        public void ResetMassData() {
            // Compute mass data from shapes. Each shape has its own density.
            m_mass = 0.0f;
            m_invMass = 0.0f;
            m_I = 0.0f;
            m_invI = 0.0f;
            m_sweep.localCenter.SetZero();

            // Static and kinematic bodies have zero mass.
            if (m_type == b2BodyType.b2_staticBody || m_type == b2BodyType.b2_kinematicBody) {
                m_sweep.c0 = m_xf.p;
                m_sweep.c = m_xf.p;
                m_sweep.a0 = m_sweep.a;
                return;
            }

            b2Assert(m_type == b2BodyType.b2_dynamicBody, "m_type == b2_dynamicBody");

            // Accumulate mass over all fixtures.
            b2Vec2 localCenter = b2Vec2_zero;
            for (b2Fixture? f = m_fixtureList; f != null; f = f.m_next) {
                if (f.m_density == 0.0f)
                    continue;

                f.GetMassData(out b2MassData massData);
                m_mass += massData.mass;
                localCenter += massData.mass * massData.center;
                m_I += massData.I;
            }

            // Compute center of mass.
            if (m_mass > 0.0f) {
                m_invMass = 1.0f / m_mass;
                localCenter *= m_invMass;
            } else {
                // Force all dynamic bodies to have a positive mass.
                m_mass = 1.0f;
                m_invMass = 1.0f;
            }

            if (m_I > 0.0f && (m_flags & e_fixedRotationFlag) == 0) {
                // Center the inertia about the center of mass.
                m_I -= m_mass * b2Dot(localCenter, localCenter);
                b2Assert(m_I > 0.0f);
                m_invI = 1.0f / m_I;
            } else {
                m_I = 0.0f;
                m_invI = 0.0f;
            }

            // Move center of mass.
            b2Vec2 oldCenter = m_sweep.c;
            m_sweep.localCenter = localCenter;
            m_sweep.c0 = m_sweep.c = b2Mul(m_xf, m_sweep.localCenter);

            // Update center of mass velocity.
            m_linearVelocity += b2Cross(m_angularVelocity, m_sweep.c - oldCenter);
        }

        /// <summary>Overrides the mass properties of a dynamic body.</summary>
        /// <param name="massData">The mass properties.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown while the world is stepping, or for a non-positive centered inertia.</exception>
        internal void SetMassData(in b2MassData massData) {
            b2Assert(m_world.IsLocked() == false, "m_world->IsLocked() == false");
            if (m_world.IsLocked() == true)
                return;

            if (m_type != b2BodyType.b2_dynamicBody)
                return;

            m_invMass = 0.0f;
            m_I = 0.0f;
            m_invI = 0.0f;

            m_mass = massData.mass;
            if (m_mass <= 0.0f)
                m_mass = 1.0f;

            m_invMass = 1.0f / m_mass;

            if (massData.I > 0.0f && (m_flags & e_fixedRotationFlag) == 0) {
                m_I = massData.I - m_mass * b2Dot(massData.center, massData.center);
                b2Assert(m_I > 0.0f);
                m_invI = 1.0f / m_I;
            }

            // Move center of mass.
            b2Vec2 oldCenter = m_sweep.c;
            m_sweep.localCenter = massData.center;
            m_sweep.c0 = m_sweep.c = b2Mul(m_xf, m_sweep.localCenter);

            // Update center of mass velocity.
            m_linearVelocity += b2Cross(m_angularVelocity, m_sweep.c - oldCenter);
        }

        /// <summary>Whether this body may collide with another: at least one must be dynamic, and no joint between them may forbid it.</summary>
        /// <param name="other">The other body.</param>
        /// <returns>True when they may collide.</returns>
        internal bool ShouldCollide(b2Body other) {
            // At least one body should be dynamic.
            if (m_type != b2BodyType.b2_dynamicBody && other.m_type != b2BodyType.b2_dynamicBody)
                return false;

            // Does a joint prevent collision?
            for (b2JointEdge? jn = m_jointList; jn != null; jn = jn.next) {
                if (jn.other == other) {
                    if (jn.joint!.m_collideConnected == false)
                        return false;
                }
            }

            return true;
        }

        /// <summary>
        ///     Moves the body origin to a position and angle at once, re-synchronizing the proxies and finding new contacts
        ///     immediately; the velocities are kept.
        /// </summary>
        /// <param name="position">The world position of the body origin.</param>
        /// <param name="angle">The angle in radians.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown while the world is stepping.</exception>
        public void SetTransform(in b2Vec2 position, float angle) {
            b2Assert(m_world.IsLocked() == false, "m_world->IsLocked() == false");
            if (m_world.IsLocked() == true)
                return;

            m_xf.q.Set(angle);
            m_xf.p = position;

            m_sweep.c = b2Mul(m_xf, m_sweep.localCenter);
            m_sweep.a = angle;

            m_sweep.c0 = m_sweep.c;
            m_sweep.a0 = angle;

            b2BroadPhase broadPhase = m_world.m_contactManager.m_broadPhase;
            for (b2Fixture? f = m_fixtureList; f != null; f = f.m_next)
                f.Synchronize(broadPhase, m_xf, m_xf);

            m_world.m_contactManager.FindNewContacts();
        }

        /// <summary>Moves the proxies to cover the swept shape from the sweep's start to the current transform.</summary>
        internal void SynchronizeFixtures() {
            b2Transform xf1 = default;
            xf1.q.Set(m_sweep.a0);
            xf1.p = m_sweep.c0 - b2Mul(xf1.q, m_sweep.localCenter);

            b2BroadPhase broadPhase = m_world.m_contactManager.m_broadPhase;
            for (b2Fixture? f = m_fixtureList; f != null; f = f.m_next)
                f.Synchronize(broadPhase, xf1, m_xf);
        }

        /// <summary>Activates the body (creating its proxies) or deactivates it (destroying its proxies and contacts).</summary>
        /// <param name="flag">True to activate.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown while the world is stepping.</exception>
        internal void SetActive(bool flag) {
            b2Assert(m_world.IsLocked() == false, "m_world->IsLocked() == false");

            if (flag == IsActive())
                return;

            if (flag) {
                m_flags |= e_activeFlag;

                // Create all proxies.
                b2BroadPhase broadPhase = m_world.m_contactManager.m_broadPhase;
                for (b2Fixture? f = m_fixtureList; f != null; f = f.m_next)
                    f.CreateProxies(broadPhase, m_xf);

                // Contacts are created the next time step.
            } else {
                m_flags &= unchecked((ushort)~e_activeFlag);

                // Destroy all proxies.
                b2BroadPhase broadPhase = m_world.m_contactManager.m_broadPhase;
                for (b2Fixture? f = m_fixtureList; f != null; f = f.m_next)
                    f.DestroyProxies(broadPhase);

                // Destroy the attached contacts.
                b2ContactEdge? ce = m_contactList;
                while (ce != null) {
                    b2ContactEdge ce0 = ce;
                    ce = ce.next;
                    m_world.m_contactManager.Destroy(ce0.contact!);
                }

                m_contactList = null;
            }
        }

        /// <summary>Sets fixed rotation, zeroing the angular velocity and recomputing the mass.</summary>
        /// <param name="flag">True to fix the rotation.</param>
        internal void SetFixedRotation(bool flag) {
            bool status = (m_flags & e_fixedRotationFlag) == e_fixedRotationFlag;
            if (status == flag)
                return;

            if (flag)
                m_flags |= e_fixedRotationFlag;
            else
                m_flags &= unchecked((ushort)~e_fixedRotationFlag);

            m_angularVelocity = 0.0f;

            ResetMassData();
        }
    }
}
