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
// Altered source version (Gym.NET, 2026-11-10): Box2D/Dynamics/b2World.h and b2World.cpp of pybox2d 2.3.10's bundled
// Box2D 2.3.0, translated to C#. The C++ API lives in the internal members below; the pybox2d-shaped public surface is in
// Facade/b2World.Facade.cs. The block and stack allocators, the profile timers, debug drawing and Dump are left out.
using static Gym.Box2D.b2Math;
using static Gym.Box2D.b2Settings;

namespace Gym.Box2D {
    /// <summary>
    ///     The physics world: bodies, joints and contacts, stepped with <see cref="Step(float, int, int)"/>. Not
    ///     thread-safe: one world belongs to one thread at a time (each env instance owns its own world).
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Dynamics/b2World.cpp">Box2D/Dynamics/b2World.cpp</see>.</remarks>
    public partial class b2World {
        /// <summary>Set when a fixture was added since the last step, so new contacts are looked for first.</summary>
        internal const int e_newFixture = 0x0001;

        /// <summary>Set while the world is stepping; creating or destroying anything then is refused.</summary>
        internal const int e_locked = 0x0002;

        /// <summary>Set when forces are cleared after each step (the default).</summary>
        internal const int e_clearForces = 0x0004;

        /// <summary>The flags.</summary>
        internal int m_flags;

        /// <summary>The contact manager.</summary>
        internal readonly b2ContactManager m_contactManager = new b2ContactManager();

        /// <summary>The bodies, newest first.</summary>
        internal b2Body? m_bodyList;

        /// <summary>The joints, newest first.</summary>
        internal b2Joint? m_jointList;

        /// <summary>The number of bodies.</summary>
        internal int m_bodyCount;

        /// <summary>The number of joints.</summary>
        internal int m_jointCount;

        /// <summary>The gravity.</summary>
        internal b2Vec2 m_gravity;

        /// <summary>Whether bodies may fall asleep.</summary>
        internal bool m_allowSleep;

        /// <summary>The destruction listener.</summary>
        internal b2DestructionListener? m_destructionListener;

        /// <summary>The inverse of the previous step's dt, for the warm-start ratio.</summary>
        internal float m_inv_dt0;

        /// <summary>Whether to warm-start the solvers.</summary>
        internal bool m_warmStarting;

        /// <summary>Whether to run continuous collision (time of impact).</summary>
        internal bool m_continuousPhysics;

        /// <summary>Whether to stop after each time-of-impact sub-step (debugging).</summary>
        internal bool m_subStepping;

        /// <summary>Whether the last step completed (false between sub-steps).</summary>
        internal bool m_stepComplete;

        /// <summary>The step profile (always zero in this port).</summary>
        internal b2Profile m_profile;

        /// <summary>A world with the given gravity: warm starting, continuous physics and sleeping on, forces cleared after each step.</summary>
        /// <param name="gravity">The gravity vector.</param>
        internal b2World(in b2Vec2 gravity) {
            m_destructionListener = null;

            m_bodyList = null;
            m_jointList = null;

            m_bodyCount = 0;
            m_jointCount = 0;

            m_warmStarting = true;
            m_continuousPhysics = true;
            m_subStepping = false;

            m_stepComplete = true;

            m_allowSleep = true;
            m_gravity = gravity;

            m_flags = e_clearForces;

            m_inv_dt0 = 0.0f;
        }

        /// <summary>Whether the world is stepping.</summary>
        /// <returns>True while locked.</returns>
        internal bool IsLocked() => (m_flags & e_locked) == e_locked;

        /// <summary>Sets the destruction listener.</summary>
        /// <param name="listener">The listener, or null.</param>
        internal void SetDestructionListener(b2DestructionListener? listener) => m_destructionListener = listener;

        /// <summary>Sets the contact filter (null disables user filtering).</summary>
        /// <param name="filter">The filter, or null.</param>
        internal void SetContactFilter(b2ContactFilter? filter) => m_contactManager.m_contactFilter = filter;

        /// <summary>Sets the contact listener.</summary>
        /// <param name="listener">The listener, or null.</param>
        internal void SetContactListener(b2ContactListener? listener) => m_contactManager.m_contactListener = listener;

        /// <summary>Creates a body from a definition, first in the world's body list (C++ <c>CreateBody</c>; the public <see cref="CreateBody"/> adds pybox2d's fixtures).</summary>
        /// <param name="def">The definition.</param>
        /// <returns>The body.</returns>
        /// <exception cref="NumSharp.AssertionError">Thrown while the world is stepping or for an invalid definition.</exception>
        internal b2Body CreateBodyCore(b2BodyDef def) {
            b2Assert(IsLocked() == false);

            var b = new b2Body(def, this);

            // Add to world doubly linked list.
            b.m_prev = null;
            b.m_next = m_bodyList;
            if (m_bodyList != null)
                m_bodyList.m_prev = b;
            m_bodyList = b;
            ++m_bodyCount;

            return b;
        }

        /// <summary>
        ///     Destroys a body with its joints (said goodbye to), contacts and fixtures (said goodbye to, proxies destroyed).
        /// </summary>
        /// <param name="b">The body.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown while the world is stepping or when it has no bodies.</exception>
        public void DestroyBody(b2Body b) {
            b2Assert(m_bodyCount > 0);
            b2Assert(IsLocked() == false);
            if (IsLocked())
                return;

            // Delete the attached joints.
            b2JointEdge? je = b.m_jointList;
            while (je != null) {
                b2JointEdge je0 = je;
                je = je.next;

                m_destructionListener?.SayGoodbye(je0.joint!);

                DestroyJoint(je0.joint!);

                b.m_jointList = je;
            }

            b.m_jointList = null;

            // Delete the attached contacts.
            b2ContactEdge? ce = b.m_contactList;
            while (ce != null) {
                b2ContactEdge ce0 = ce;
                ce = ce.next;
                m_contactManager.Destroy(ce0.contact!);
            }

            b.m_contactList = null;

            // Delete the attached fixtures. This destroys broad-phase proxies.
            b2Fixture? f = b.m_fixtureList;
            while (f != null) {
                b2Fixture f0 = f;
                f = f.m_next;

                m_destructionListener?.SayGoodbye(f0);

                f0.DestroyProxies(m_contactManager.m_broadPhase);
                f0.Destroy();

                b.m_fixtureList = f;
                b.m_fixtureCount -= 1;
            }

            b.m_fixtureList = null;
            b.m_fixtureCount = 0;

            // Remove world body list.
            if (b.m_prev != null)
                b.m_prev.m_next = b.m_next;

            if (b.m_next != null)
                b.m_next.m_prev = b.m_prev;

            if (b == m_bodyList)
                m_bodyList = b.m_next;

            --m_bodyCount;
        }

        /// <summary>
        ///     Creates a joint, first in the world's and both bodies' lists; when its bodies may not collide, their contacts
        ///     are flagged for filtering. Creating a joint doesn't wake the bodies.
        /// </summary>
        /// <param name="def">The definition.</param>
        /// <returns>The joint.</returns>
        /// <exception cref="NumSharp.AssertionError">Thrown while the world is stepping, or for an unsupported joint type or one body twice.</exception>
        internal b2Joint CreateJointCore(b2JointDef def) {
            b2Assert(IsLocked() == false);

            b2Joint j = b2Joint.Create(def);

            // Connect to the world list.
            j.m_prev = null;
            j.m_next = m_jointList;
            if (m_jointList != null)
                m_jointList.m_prev = j;
            m_jointList = j;
            ++m_jointCount;

            // Connect to the bodies' doubly linked lists.
            j.m_edgeA.joint = j;
            j.m_edgeA.other = j.m_bodyB;
            j.m_edgeA.prev = null;
            j.m_edgeA.next = j.m_bodyA.m_jointList;
            if (j.m_bodyA.m_jointList != null)
                j.m_bodyA.m_jointList.prev = j.m_edgeA;
            j.m_bodyA.m_jointList = j.m_edgeA;

            j.m_edgeB.joint = j;
            j.m_edgeB.other = j.m_bodyA;
            j.m_edgeB.prev = null;
            j.m_edgeB.next = j.m_bodyB.m_jointList;
            if (j.m_bodyB.m_jointList != null)
                j.m_bodyB.m_jointList.prev = j.m_edgeB;
            j.m_bodyB.m_jointList = j.m_edgeB;

            b2Body bodyA = def.bodyA!;
            b2Body bodyB = def.bodyB!;

            // If the joint prevents collisions, then flag any contacts for filtering.
            if (def.collideConnected == false) {
                b2ContactEdge? edge = bodyB.GetContactList();
                while (edge != null) {
                    if (edge.other == bodyA)
                        // Flag the contact for filtering at the next time step (where either
                        // body is awake).
                        edge.contact!.FlagForFiltering();

                    edge = edge.next;
                }
            }

            // Note: creating a joint doesn't wake the bodies.

            return j;
        }

        /// <summary>Destroys a joint, waking its bodies; their contacts are re-filtered when the joint forbade collision.</summary>
        /// <param name="j">The joint.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown while the world is stepping.</exception>
        public void DestroyJoint(b2Joint j) {
            b2Assert(IsLocked() == false);
            if (IsLocked())
                return;

            bool collideConnected = j.m_collideConnected;

            // Remove from the doubly linked list.
            if (j.m_prev != null)
                j.m_prev.m_next = j.m_next;

            if (j.m_next != null)
                j.m_next.m_prev = j.m_prev;

            if (j == m_jointList)
                m_jointList = j.m_next;

            // Disconnect from island graph.
            b2Body bodyA = j.m_bodyA;
            b2Body bodyB = j.m_bodyB;

            // Wake up connected bodies.
            bodyA.SetAwake(true);
            bodyB.SetAwake(true);

            // Remove from body 1.
            if (j.m_edgeA.prev != null)
                j.m_edgeA.prev.next = j.m_edgeA.next;

            if (j.m_edgeA.next != null)
                j.m_edgeA.next.prev = j.m_edgeA.prev;

            if (j.m_edgeA == bodyA.m_jointList)
                bodyA.m_jointList = j.m_edgeA.next;

            j.m_edgeA.prev = null;
            j.m_edgeA.next = null;

            // Remove from body 2
            if (j.m_edgeB.prev != null)
                j.m_edgeB.prev.next = j.m_edgeB.next;

            if (j.m_edgeB.next != null)
                j.m_edgeB.next.prev = j.m_edgeB.prev;

            if (j.m_edgeB == bodyB.m_jointList)
                bodyB.m_jointList = j.m_edgeB.next;

            j.m_edgeB.prev = null;
            j.m_edgeB.next = null;

            b2Assert(m_jointCount > 0);
            --m_jointCount;

            // If the joint prevents collisions, then flag any contacts for filtering.
            if (collideConnected == false) {
                b2ContactEdge? edge = bodyB.GetContactList();
                while (edge != null) {
                    if (edge.other == bodyA)
                        // Flag the contact for filtering at the next time step (where either
                        // body is awake).
                        edge.contact!.FlagForFiltering();

                    edge = edge.next;
                }
            }
        }

        /// <summary>
        ///     Allows or forbids sleeping (pybox2d's <c>SetAllowSleeping</c>, the only way it turns sleeping off); forbidding
        ///     wakes every body.
        /// </summary>
        /// <param name="flag">True to allow sleeping.</param>
        public void SetAllowSleeping(bool flag) {
            if (flag == m_allowSleep)
                return;

            m_allowSleep = flag;
            if (m_allowSleep == false) {
                for (b2Body? b = m_bodyList; b != null; b = b.m_next)
                    b.SetAwake(true);
            }
        }

        /// <summary>Whether bodies may fall asleep (pybox2d's <c>GetAllowSleeping</c>); true unless <see cref="SetAllowSleeping"/> turned it off.</summary>
        /// <returns>The flag.</returns>
        public bool GetAllowSleeping() => m_allowSleep;

        /// <summary>
        ///     Finds the islands of awake bodies (depth-first over touching contacts and joints, not across static bodies),
        ///     solves each, then synchronizes the moved bodies' proxies and finds new contacts.
        /// </summary>
        /// <param name="step">The step.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown on an inconsistent island (an inactive body in it, a stack overflow).</exception>
        internal void Solve(in b2TimeStep step) {
            // Size the island for the worst case.
            var island = new b2Island(m_bodyCount, m_contactManager.m_contactCount, m_jointCount, m_contactManager.m_contactListener);

            // Clear all the island flags.
            for (b2Body? b = m_bodyList; b != null; b = b.m_next)
                b.m_flags &= unchecked((ushort)~b2Body.e_islandFlag);
            for (b2Contact? c = m_contactManager.m_contactList; c != null; c = c.m_next)
                c.m_flags &= ~b2Contact.e_islandFlag;
            for (b2Joint? j = m_jointList; j != null; j = j.m_next)
                j.m_islandFlag = false;

            // Build and simulate all awake islands.
            int stackSize = m_bodyCount;
            var stack = new b2Body[stackSize];
            for (b2Body? seed = m_bodyList; seed != null; seed = seed.m_next) {
                if ((seed.m_flags & b2Body.e_islandFlag) != 0)
                    continue;

                if (seed.IsAwake() == false || seed.IsActive() == false)
                    continue;

                // The seed can be dynamic or kinematic.
                if (seed.GetBodyType() == b2BodyType.b2_staticBody)
                    continue;

                // Reset island and stack.
                island.Clear();
                int stackCount = 0;
                stack[stackCount++] = seed;
                seed.m_flags |= b2Body.e_islandFlag;

                // Perform a depth first search (DFS) on the constraint graph.
                while (stackCount > 0) {
                    // Grab the next body off the stack and add it to the island.
                    b2Body b = stack[--stackCount];
                    b2Assert(b.IsActive() == true, "b->IsActive() == true");
                    island.Add(b);

                    // Make sure the body is awake.
                    b.SetAwake(true);

                    // To keep islands as small as possible, we don't
                    // propagate islands across static bodies.
                    if (b.GetBodyType() == b2BodyType.b2_staticBody)
                        continue;

                    // Search all contacts connected to this body.
                    for (b2ContactEdge? ce = b.m_contactList; ce != null; ce = ce.next) {
                        b2Contact contact = ce.contact!;

                        // Has this contact already been added to an island?
                        if ((contact.m_flags & b2Contact.e_islandFlag) != 0)
                            continue;

                        // Is this contact solid and touching?
                        if (contact.IsEnabled() == false || contact.IsTouching() == false)
                            continue;

                        // Skip sensors.
                        bool sensorA = contact.m_fixtureA!.m_isSensor;
                        bool sensorB = contact.m_fixtureB!.m_isSensor;
                        if (sensorA || sensorB)
                            continue;

                        island.Add(contact);
                        contact.m_flags |= b2Contact.e_islandFlag;

                        b2Body other = ce.other!;

                        // Was the other body already added to this island?
                        if ((other.m_flags & b2Body.e_islandFlag) != 0)
                            continue;

                        b2Assert(stackCount < stackSize);
                        stack[stackCount++] = other;
                        other.m_flags |= b2Body.e_islandFlag;
                    }

                    // Search all joints connect to this body.
                    for (b2JointEdge? je = b.m_jointList; je != null; je = je.next) {
                        if (je.joint!.m_islandFlag == true)
                            continue;

                        b2Body other = je.other!;

                        // Don't simulate joints connected to inactive bodies.
                        if (other.IsActive() == false)
                            continue;

                        island.Add(je.joint);
                        je.joint.m_islandFlag = true;

                        if ((other.m_flags & b2Body.e_islandFlag) != 0)
                            continue;

                        b2Assert(stackCount < stackSize);
                        stack[stackCount++] = other;
                        other.m_flags |= b2Body.e_islandFlag;
                    }
                }

                island.Solve(step, m_gravity, m_allowSleep);

                // Post solve cleanup.
                for (int i = 0; i < island.m_bodyCount; ++i) {
                    // Allow static bodies to participate in other islands.
                    b2Body b = island.m_bodies[i];
                    if (b.GetBodyType() == b2BodyType.b2_staticBody)
                        b.m_flags &= unchecked((ushort)~b2Body.e_islandFlag);
                }
            }

            {
                // Synchronize fixtures, check for out of range bodies.
                for (b2Body? b = m_bodyList; b != null; b = b.GetNext()) {
                    // If a body was not in an island then it did not move.
                    if ((b.m_flags & b2Body.e_islandFlag) == 0)
                        continue;

                    if (b.GetBodyType() == b2BodyType.b2_staticBody)
                        continue;

                    // Update fixtures (for broad-phase).
                    b.SynchronizeFixtures();
                }

                // Look for new contacts.
                m_contactManager.FindNewContacts();
            }
        }

        /// <summary>
        ///     Continuous collision: repeatedly finds the earliest time of impact among contacts with a bullet, static or
        ///     kinematic body, advances both bodies there and solves a sub-step island.
        /// </summary>
        /// <param name="step">The step.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown on an inconsistent sweep (a body already at the end of the step).</exception>
        internal void SolveTOI(in b2TimeStep step) {
            var island = new b2Island(2 * b2_maxTOIContacts, b2_maxTOIContacts, 0, m_contactManager.m_contactListener);

            if (m_stepComplete) {
                for (b2Body? b = m_bodyList; b != null; b = b.m_next) {
                    b.m_flags &= unchecked((ushort)~b2Body.e_islandFlag);
                    b.m_sweep.alpha0 = 0.0f;
                }

                for (b2Contact? c = m_contactManager.m_contactList; c != null; c = c.m_next) {
                    // Invalidate TOI
                    c.m_flags &= ~(b2Contact.e_toiFlag | b2Contact.e_islandFlag);
                    c.m_toiCount = 0;
                    c.m_toi = 1.0f;
                }
            }

            // Find TOI events and solve them.
            for (; ; ) {
                // Find the first TOI.
                b2Contact? minContact = null;
                float minAlpha = 1.0f;

                for (b2Contact? c = m_contactManager.m_contactList; c != null; c = c.m_next) {
                    // Is this contact disabled?
                    if (c.IsEnabled() == false)
                        continue;

                    // Prevent excessive sub-stepping.
                    if (c.m_toiCount > b2_maxSubSteps)
                        continue;

                    float alpha = 1.0f;
                    if ((c.m_flags & b2Contact.e_toiFlag) != 0) {
                        // This contact has a valid cached TOI.
                        alpha = c.m_toi;
                    } else {
                        b2Fixture fA = c.GetFixtureA();
                        b2Fixture fB = c.GetFixtureB();

                        // Is there a sensor?
                        if (fA.IsSensor() || fB.IsSensor())
                            continue;

                        b2Body bA = fA.GetBody();
                        b2Body bB = fB.GetBody();

                        b2BodyType typeA = bA.m_type;
                        b2BodyType typeB = bB.m_type;
                        b2Assert(typeA == b2BodyType.b2_dynamicBody || typeB == b2BodyType.b2_dynamicBody, "typeA == b2_dynamicBody || typeB == b2_dynamicBody");

                        bool activeA = bA.IsAwake() && typeA != b2BodyType.b2_staticBody;
                        bool activeB = bB.IsAwake() && typeB != b2BodyType.b2_staticBody;

                        // Is at least one body active (awake and dynamic or kinematic)?
                        if (activeA == false && activeB == false)
                            continue;

                        bool collideA = bA.IsBullet() || typeA != b2BodyType.b2_dynamicBody;
                        bool collideB = bB.IsBullet() || typeB != b2BodyType.b2_dynamicBody;

                        // Are these two non-bullet dynamic bodies?
                        if (collideA == false && collideB == false)
                            continue;

                        // Compute the TOI for this contact.
                        // Put the sweeps onto the same time interval.
                        float alpha0 = bA.m_sweep.alpha0;

                        if (bA.m_sweep.alpha0 < bB.m_sweep.alpha0) {
                            alpha0 = bB.m_sweep.alpha0;
                            bA.m_sweep.Advance(alpha0);
                        } else if (bB.m_sweep.alpha0 < bA.m_sweep.alpha0) {
                            alpha0 = bA.m_sweep.alpha0;
                            bB.m_sweep.Advance(alpha0);
                        }

                        b2Assert(alpha0 < 1.0f);

                        int indexA = c.GetChildIndexA();
                        int indexB = c.GetChildIndexB();

                        // Compute the time of impact in interval [0, minTOI]
                        b2TOIInput input = default;
                        input.proxyA.Set(fA.GetShape(), indexA);
                        input.proxyB.Set(fB.GetShape(), indexB);
                        input.sweepA = bA.m_sweep;
                        input.sweepB = bB.m_sweep;
                        input.tMax = 1.0f;

                        b2TimeOfImpactFn.b2TimeOfImpact(out b2TOIOutput output, input);

                        // Beta is the fraction of the remaining portion of the .
                        float beta = output.t;
                        if (output.state == b2TOIOutputState.e_touching)
                            alpha = b2Min(alpha0 + (1.0f - alpha0) * beta, 1.0f);
                        else
                            alpha = 1.0f;

                        c.m_toi = alpha;
                        c.m_flags |= b2Contact.e_toiFlag;
                    }

                    if (alpha < minAlpha) {
                        // This is the minimum TOI found so far.
                        minContact = c;
                        minAlpha = alpha;
                    }
                }

                if (minContact == null || 1.0f - 10.0f * b2_epsilon < minAlpha) {
                    // No more TOI events. Done!
                    m_stepComplete = true;
                    break;
                }

                {
                    // Advance the bodies to the TOI.
                    b2Fixture fA = minContact.GetFixtureA();
                    b2Fixture fB = minContact.GetFixtureB();
                    b2Body bA = fA.GetBody();
                    b2Body bB = fB.GetBody();

                    b2Sweep backup1 = bA.m_sweep;
                    b2Sweep backup2 = bB.m_sweep;

                    bA.Advance(minAlpha);
                    bB.Advance(minAlpha);

                    // The TOI contact likely has some new contact points.
                    minContact.Update(m_contactManager.m_contactListener);
                    minContact.m_flags &= ~b2Contact.e_toiFlag;
                    ++minContact.m_toiCount;

                    // Is the contact solid?
                    if (minContact.IsEnabled() == false || minContact.IsTouching() == false) {
                        // Restore the sweeps.
                        minContact.SetEnabled(false);
                        bA.m_sweep = backup1;
                        bB.m_sweep = backup2;
                        bA.SynchronizeTransform();
                        bB.SynchronizeTransform();
                        continue;
                    }

                    bA.SetAwake(true);
                    bB.SetAwake(true);

                    // Build the island
                    island.Clear();
                    island.Add(bA);
                    island.Add(bB);
                    island.Add(minContact);

                    bA.m_flags |= b2Body.e_islandFlag;
                    bB.m_flags |= b2Body.e_islandFlag;
                    minContact.m_flags |= b2Contact.e_islandFlag;

                    // Get contacts on bodyA and bodyB.
                    b2Body[] bodies = { bA, bB };
                    for (int i = 0; i < 2; ++i) {
                        b2Body body = bodies[i];
                        if (body.m_type == b2BodyType.b2_dynamicBody) {
                            for (b2ContactEdge? ce = body.m_contactList; ce != null; ce = ce.next) {
                                if (island.m_bodyCount == island.m_bodyCapacity)
                                    break;

                                if (island.m_contactCount == island.m_contactCapacity)
                                    break;

                                b2Contact contact = ce.contact!;

                                // Has this contact already been added to the island?
                                if ((contact.m_flags & b2Contact.e_islandFlag) != 0)
                                    continue;

                                // Only add static, kinematic, or bullet bodies.
                                b2Body other = ce.other!;
                                if (other.m_type == b2BodyType.b2_dynamicBody && body.IsBullet() == false && other.IsBullet() == false)
                                    continue;

                                // Skip sensors.
                                bool sensorA = contact.m_fixtureA!.m_isSensor;
                                bool sensorB = contact.m_fixtureB!.m_isSensor;
                                if (sensorA || sensorB)
                                    continue;

                                // Tentatively advance the body to the TOI.
                                b2Sweep backup = other.m_sweep;
                                if ((other.m_flags & b2Body.e_islandFlag) == 0)
                                    other.Advance(minAlpha);

                                // Update the contact points
                                contact.Update(m_contactManager.m_contactListener);

                                // Was the contact disabled by the user?
                                if (contact.IsEnabled() == false) {
                                    other.m_sweep = backup;
                                    other.SynchronizeTransform();
                                    continue;
                                }

                                // Are there contact points?
                                if (contact.IsTouching() == false) {
                                    other.m_sweep = backup;
                                    other.SynchronizeTransform();
                                    continue;
                                }

                                // Add the contact to the island
                                contact.m_flags |= b2Contact.e_islandFlag;
                                island.Add(contact);

                                // Has the other body already been added to the island?
                                if ((other.m_flags & b2Body.e_islandFlag) != 0)
                                    continue;

                                // Add the other body to the island.
                                other.m_flags |= b2Body.e_islandFlag;

                                if (other.m_type != b2BodyType.b2_staticBody)
                                    other.SetAwake(true);

                                island.Add(other);
                            }
                        }
                    }

                    b2TimeStep subStep;
                    subStep.dt = (1.0f - minAlpha) * step.dt;
                    subStep.inv_dt = 1.0f / subStep.dt;
                    subStep.dtRatio = 1.0f;
                    subStep.positionIterations = 20;
                    subStep.velocityIterations = step.velocityIterations;
                    subStep.warmStarting = false;
                    island.SolveTOI(subStep, bA.m_islandIndex, bB.m_islandIndex);

                    // Reset island flags and synchronize broad-phase proxies.
                    for (int i = 0; i < island.m_bodyCount; ++i) {
                        b2Body body = island.m_bodies[i];
                        body.m_flags &= unchecked((ushort)~b2Body.e_islandFlag);

                        if (body.m_type != b2BodyType.b2_dynamicBody)
                            continue;

                        body.SynchronizeFixtures();

                        // Invalidate all contact TOIs on this displaced body.
                        for (b2ContactEdge? ce = body.m_contactList; ce != null; ce = ce.next)
                            ce.contact!.m_flags &= ~(b2Contact.e_toiFlag | b2Contact.e_islandFlag);
                    }

                    // Commit fixture proxy movements to the broad-phase so that new contacts are created.
                    // Also, some contacts can be destroyed.
                    m_contactManager.FindNewContacts();

                    if (m_subStepping) {
                        m_stepComplete = false;
                        break;
                    }
                }
            }
        }

        /// <summary>
        ///     Steps the world: finds the contacts of new fixtures, updates contacts (calling the listener), solves the islands,
        ///     runs continuous collision, and clears the forces (unless auto-clear is off).
        /// </summary>
        /// <param name="dt">The time step in seconds (float32: pybox2d rounds Python's <c>1.0 / FPS</c> to float32).</param>
        /// <param name="velocityIterations">The velocity iterations.</param>
        /// <param name="positionIterations">The position iterations.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown on an internal inconsistency; the world stays locked afterwards, as in pybox2d.</exception>
        internal void Step(float dt, int velocityIterations, int positionIterations) {
            // If new fixtures were added, we need to find the new contacts.
            if ((m_flags & e_newFixture) != 0) {
                m_contactManager.FindNewContacts();
                m_flags &= ~e_newFixture;
            }

            m_flags |= e_locked;

            b2TimeStep step;
            step.dt = dt;
            step.velocityIterations = velocityIterations;
            step.positionIterations = positionIterations;
            if (dt > 0.0f)
                step.inv_dt = 1.0f / dt;
            else
                step.inv_dt = 0.0f;

            step.dtRatio = m_inv_dt0 * dt;

            step.warmStarting = m_warmStarting;

            // Update contacts. This is where some contacts are destroyed.
            m_contactManager.Collide();

            // Integrate velocities, solve velocity constraints, and integrate positions.
            if (m_stepComplete && step.dt > 0.0f)
                Solve(step);

            // Handle TOI events.
            if (m_continuousPhysics && step.dt > 0.0f)
                SolveTOI(step);

            if (step.dt > 0.0f)
                m_inv_dt0 = step.inv_dt;

            if ((m_flags & e_clearForces) != 0)
                ClearForces();

            m_flags &= ~e_locked;
        }

        /// <summary>Zeroes every body's accumulated force and torque (needed only when <see cref="AutoClearForces"/> is off).</summary>
        public void ClearForces() {
            for (b2Body? body = m_bodyList; body != null; body = body.GetNext()) {
                body.m_force.SetZero();
                body.m_torque = 0.0f;
            }
        }

        /// <summary>Reports every fixture whose fat box overlaps a box.</summary>
        /// <param name="callback">The callback; returning false stops.</param>
        /// <param name="aabb">The box.</param>
        internal void QueryAABB(b2QueryCallback callback, in b2AABB aabb) {
            b2WorldQueryWrapper wrapper = new b2WorldQueryWrapper(m_contactManager.m_broadPhase, callback);
            m_contactManager.m_broadPhase.Query(wrapper, aabb);
        }

        /// <summary>
        ///     Casts a ray from <paramref name="point1"/> to <paramref name="point2"/>, telling the callback about each fixture
        ///     hit (in tree order); the callback's return value clips or stops the ray.
        /// </summary>
        /// <param name="callback">The callback.</param>
        /// <param name="point1">The ray's start.</param>
        /// <param name="point2">The ray's end.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown for a ray of zero length: <c>r.LengthSquared() &gt; 0.0f</c>.</exception>
        public void RayCast(b2RayCastCallback callback, b2Vec2 point1, b2Vec2 point2) {
            b2WorldRayCastWrapper wrapper = new b2WorldRayCastWrapper(m_contactManager.m_broadPhase, callback);
            b2RayCastInput input;
            input.maxFraction = 1.0f;
            input.p1 = point1;
            input.p2 = point2;
            m_contactManager.m_broadPhase.RayCast(wrapper, input);
        }

        /// <summary>The number of broad-phase proxies.</summary>
        /// <returns>The count.</returns>
        internal int GetProxyCount() => m_contactManager.m_broadPhase.GetProxyCount();

        /// <summary>Shifts the world origin (useful for large worlds): every position moves by <c>-newOrigin</c>.</summary>
        /// <param name="newOrigin">The new origin in the old frame.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown while the world is stepping.</exception>
        internal void ShiftOrigin(in b2Vec2 newOrigin) {
            b2Assert((m_flags & e_locked) == 0);
            if ((m_flags & e_locked) == e_locked)
                return;

            for (b2Body? b = m_bodyList; b != null; b = b.m_next) {
                b.m_xf.p -= newOrigin;
                b.m_sweep.c0 -= newOrigin;
                b.m_sweep.c -= newOrigin;
            }

            for (b2Joint? j = m_jointList; j != null; j = j.m_next)
                j.ShiftOrigin(newOrigin);

            m_contactManager.m_broadPhase.ShiftOrigin(newOrigin);
        }

        /// <summary>Forwards tree query hits to the user's query callback as fixtures.</summary>
        private readonly struct b2WorldQueryWrapper : Ib2TreeQueryCallback {
            /// <summary>The broad-phase.</summary>
            private readonly b2BroadPhase broadPhase;

            /// <summary>The user's callback.</summary>
            private readonly b2QueryCallback callback;

            /// <summary>A wrapper over a broad-phase and a callback.</summary>
            /// <param name="broadPhase">The broad-phase.</param>
            /// <param name="callback">The callback.</param>
            public b2WorldQueryWrapper(b2BroadPhase broadPhase, b2QueryCallback callback) {
                this.broadPhase = broadPhase;
                this.callback = callback;
            }

            /// <summary>Reports the proxy's fixture.</summary>
            /// <param name="proxyId">The proxy.</param>
            /// <returns>The callback's answer.</returns>
            public bool QueryCallback(int proxyId) {
                var proxy = (b2FixtureProxy)broadPhase.GetUserData(proxyId)!;
                return callback.ReportFixture(proxy.fixture!);
            }
        }

        /// <summary>Ray-casts each candidate fixture and forwards hits to the user's ray-cast callback.</summary>
        private readonly struct b2WorldRayCastWrapper : Ib2TreeRayCastCallback {
            /// <summary>The broad-phase.</summary>
            private readonly b2BroadPhase broadPhase;

            /// <summary>The user's callback.</summary>
            private readonly b2RayCastCallback callback;

            /// <summary>A wrapper over a broad-phase and a callback.</summary>
            /// <param name="broadPhase">The broad-phase.</param>
            /// <param name="callback">The callback.</param>
            public b2WorldRayCastWrapper(b2BroadPhase broadPhase, b2RayCastCallback callback) {
                this.broadPhase = broadPhase;
                this.callback = callback;
            }

            /// <summary>Ray-casts the proxy's fixture; a hit reports the point <c>(1 - fraction) * p1 + fraction * p2</c>.</summary>
            /// <param name="input">The ray.</param>
            /// <param name="proxyId">The proxy.</param>
            /// <returns>The callback's fraction, or the ray's current max fraction on a miss.</returns>
            public float RayCastCallback(in b2RayCastInput input, int proxyId) {
                object? userData = broadPhase.GetUserData(proxyId);
                var proxy = (b2FixtureProxy)userData!;
                b2Fixture fixture = proxy.fixture!;
                int index = proxy.childIndex;
                b2RayCastOutput output = default;
                bool hit = fixture.RayCast(ref output, input, index);

                if (hit) {
                    float fraction = output.fraction;
                    b2Vec2 point = (1.0f - fraction) * input.p1 + fraction * input.p2;
                    return callback.ReportFixture(fixture, point, output.normal, fraction);
                }

                return input.maxFraction;
            }
        }
    }
}
