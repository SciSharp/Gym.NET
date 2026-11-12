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
// Altered source version (Gym.NET, 2026-11-10): Box2D/Dynamics/b2Island.h and b2Island.cpp of pybox2d 2.3.10's bundled
// Box2D 2.3.0, translated to C#. The arrays are managed (C++ takes them from the stack allocator); the profile timers are
// left out.
using static Gym.Box2D.b2Math;
using static Gym.Box2D.b2Settings;

namespace Gym.Box2D {
    /// <summary>
    ///     An island: bodies connected by touching contacts and joints, solved together. Integration is semi-implicit Euler
    ///     with damping, then velocity iterations (joints before contacts), position integration with the translation and
    ///     rotation caps, position iterations, and sleeping.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Dynamics/b2Island.cpp">Box2D/Dynamics/b2Island.cpp</see>.</remarks>
    internal sealed class b2Island {
        /// <summary>The listener told about post-solve impulses.</summary>
        public b2ContactListener? m_listener;

        /// <summary>The bodies.</summary>
        public b2Body[] m_bodies;

        /// <summary>The contacts.</summary>
        public b2Contact[] m_contacts;

        /// <summary>The joints.</summary>
        public b2Joint[] m_joints;

        /// <summary>The solver positions, one per body.</summary>
        public b2Position[] m_positions;

        /// <summary>The solver velocities, one per body.</summary>
        public b2Velocity[] m_velocities;

        /// <summary>The number of bodies.</summary>
        public int m_bodyCount;

        /// <summary>The number of joints.</summary>
        public int m_jointCount;

        /// <summary>The number of contacts.</summary>
        public int m_contactCount;

        /// <summary>The body capacity.</summary>
        public int m_bodyCapacity;

        /// <summary>The contact capacity.</summary>
        public int m_contactCapacity;

        /// <summary>The joint capacity.</summary>
        public int m_jointCapacity;

        /// <summary>An empty island sized for the worst case.</summary>
        /// <param name="bodyCapacity">The maximum number of bodies.</param>
        /// <param name="contactCapacity">The maximum number of contacts.</param>
        /// <param name="jointCapacity">The maximum number of joints.</param>
        /// <param name="listener">The contact listener, or null.</param>
        public b2Island(int bodyCapacity, int contactCapacity, int jointCapacity, b2ContactListener? listener) {
            m_bodyCapacity = bodyCapacity;
            m_contactCapacity = contactCapacity;
            m_jointCapacity = jointCapacity;
            m_bodyCount = 0;
            m_contactCount = 0;
            m_jointCount = 0;

            m_listener = listener;

            m_bodies = new b2Body[bodyCapacity];
            m_contacts = new b2Contact[contactCapacity];
            m_joints = new b2Joint[jointCapacity];

            m_velocities = new b2Velocity[m_bodyCapacity];
            m_positions = new b2Position[m_bodyCapacity];
        }

        /// <summary>Empties the island for the next one.</summary>
        public void Clear() {
            m_bodyCount = 0;
            m_contactCount = 0;
            m_jointCount = 0;
        }

        /// <summary>Adds a body, giving it its island index.</summary>
        /// <param name="body">The body.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown when the island is full.</exception>
        public void Add(b2Body body) {
            b2Assert(m_bodyCount < m_bodyCapacity);
            body.m_islandIndex = m_bodyCount;
            m_bodies[m_bodyCount] = body;
            ++m_bodyCount;
        }

        /// <summary>Adds a contact.</summary>
        /// <param name="contact">The contact.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown when the island is full.</exception>
        public void Add(b2Contact contact) {
            b2Assert(m_contactCount < m_contactCapacity);
            m_contacts[m_contactCount++] = contact;
        }

        /// <summary>Adds a joint.</summary>
        /// <param name="joint">The joint.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown when the island is full.</exception>
        public void Add(b2Joint joint) {
            b2Assert(m_jointCount < m_jointCapacity);
            m_joints[m_jointCount++] = joint;
        }

        /// <summary>Solves the island for a step: integrate, solve velocities and positions, copy back, report, sleep.</summary>
        /// <param name="step">The step.</param>
        /// <param name="gravity">The gravity.</param>
        /// <param name="allowSleep">Whether bodies may fall asleep.</param>
        public void Solve(in b2TimeStep step, in b2Vec2 gravity, bool allowSleep) {
            float h = step.dt;

            // Integrate velocities and apply damping. Initialize the body state.
            for (int i = 0; i < m_bodyCount; ++i) {
                b2Body b = m_bodies[i];

                b2Vec2 c = b.m_sweep.c;
                float a = b.m_sweep.a;
                b2Vec2 v = b.m_linearVelocity;
                float w = b.m_angularVelocity;

                // Store positions for continuous collision.
                b.m_sweep.c0 = b.m_sweep.c;
                b.m_sweep.a0 = b.m_sweep.a;

                if (b.m_type == b2BodyType.b2_dynamicBody) {
                    // Integrate velocities.
                    v += h * (b.m_gravityScale * gravity + b.m_invMass * b.m_force);
                    w += h * b.m_invI * b.m_torque;

                    // Apply damping.
                    // ODE: dv/dt + c * v = 0
                    // Solution: v(t) = v0 * exp(-c * t)
                    // Time step: v(t + dt) = v0 * exp(-c * (t + dt)) = v0 * exp(-c * t) * exp(-c * dt) = v * exp(-c * dt)
                    // v2 = exp(-c * dt) * v1
                    // Taylor expansion:
                    // v2 = (1.0f - c * dt) * v1
                    v *= b2Clamp(1.0f - h * b.m_linearDamping, 0.0f, 1.0f);
                    w *= b2Clamp(1.0f - h * b.m_angularDamping, 0.0f, 1.0f);
                }

                m_positions[i].c = c;
                m_positions[i].a = a;
                m_velocities[i].v = v;
                m_velocities[i].w = w;
            }

            // Solver data
            b2SolverData solverData;
            solverData.step = step;
            solverData.positions = m_positions;
            solverData.velocities = m_velocities;

            // Initialize velocity constraints.
            b2ContactSolverDef contactSolverDef;
            contactSolverDef.step = step;
            contactSolverDef.contacts = m_contacts;
            contactSolverDef.count = m_contactCount;
            contactSolverDef.positions = m_positions;
            contactSolverDef.velocities = m_velocities;

            var contactSolver = new b2ContactSolver(contactSolverDef);
            contactSolver.InitializeVelocityConstraints();

            if (step.warmStarting)
                contactSolver.WarmStart();

            for (int i = 0; i < m_jointCount; ++i)
                m_joints[i].InitVelocityConstraints(solverData);

            // Solve velocity constraints
            for (int i = 0; i < step.velocityIterations; ++i) {
                for (int j = 0; j < m_jointCount; ++j)
                    m_joints[j].SolveVelocityConstraints(solverData);

                contactSolver.SolveVelocityConstraints();
            }

            // Store impulses for warm starting
            contactSolver.StoreImpulses();

            // Integrate positions
            for (int i = 0; i < m_bodyCount; ++i) {
                b2Vec2 c = m_positions[i].c;
                float a = m_positions[i].a;
                b2Vec2 v = m_velocities[i].v;
                float w = m_velocities[i].w;

                // Check for large velocities
                b2Vec2 translation = h * v;
                if (b2Dot(translation, translation) > b2_maxTranslationSquared) {
                    float ratio = b2_maxTranslation / translation.Length();
                    v *= ratio;
                }

                float rotation = h * w;
                if (rotation * rotation > b2_maxRotationSquared) {
                    float ratio = b2_maxRotation / b2Abs(rotation);
                    w *= ratio;
                }

                // Integrate
                c += h * v;
                a += h * w;

                m_positions[i].c = c;
                m_positions[i].a = a;
                m_velocities[i].v = v;
                m_velocities[i].w = w;
            }

            // Solve position constraints
            bool positionSolved = false;
            for (int i = 0; i < step.positionIterations; ++i) {
                bool contactsOkay = contactSolver.SolvePositionConstraints();

                bool jointsOkay = true;
                for (int j = 0; j < m_jointCount; ++j) {
                    bool jointOkay = m_joints[j].SolvePositionConstraints(solverData);
                    jointsOkay = jointsOkay && jointOkay;
                }

                if (contactsOkay && jointsOkay) {
                    // Exit early if the position errors are small.
                    positionSolved = true;
                    break;
                }
            }

            // Copy state buffers back to the bodies
            for (int i = 0; i < m_bodyCount; ++i) {
                b2Body body = m_bodies[i];
                body.m_sweep.c = m_positions[i].c;
                body.m_sweep.a = m_positions[i].a;
                body.m_linearVelocity = m_velocities[i].v;
                body.m_angularVelocity = m_velocities[i].w;
                body.SynchronizeTransform();
            }

            Report(contactSolver.m_velocityConstraints);

            if (allowSleep) {
                float minSleepTime = b2_maxFloat;

                const float linTolSqr = b2_linearSleepTolerance * b2_linearSleepTolerance;
                const float angTolSqr = b2_angularSleepTolerance * b2_angularSleepTolerance;

                for (int i = 0; i < m_bodyCount; ++i) {
                    b2Body b = m_bodies[i];
                    if (b.GetBodyType() == b2BodyType.b2_staticBody)
                        continue;

                    if ((b.m_flags & b2Body.e_autoSleepFlag) == 0 ||
                        b.m_angularVelocity * b.m_angularVelocity > angTolSqr ||
                        b2Dot(b.m_linearVelocity, b.m_linearVelocity) > linTolSqr) {
                        b.m_sleepTime = 0.0f;
                        minSleepTime = 0.0f;
                    } else {
                        b.m_sleepTime += h;
                        minSleepTime = b2Min(minSleepTime, b.m_sleepTime);
                    }
                }

                if (minSleepTime >= b2_timeToSleep && positionSolved) {
                    for (int i = 0; i < m_bodyCount; ++i) {
                        b2Body b = m_bodies[i];
                        b.SetAwake(false);
                    }
                }
            }
        }

        /// <summary>
        ///     Solves a time-of-impact sub-step: position iterations moving only the two TOI bodies, a leap to the new safe
        ///     state, velocity iterations without warm starting, then integration of the rest of the step.
        /// </summary>
        /// <param name="subStep">The sub-step.</param>
        /// <param name="toiIndexA">The first TOI body's island index.</param>
        /// <param name="toiIndexB">The second TOI body's island index.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown for TOI indices outside the island.</exception>
        public void SolveTOI(in b2TimeStep subStep, int toiIndexA, int toiIndexB) {
            b2Assert(toiIndexA < m_bodyCount);
            b2Assert(toiIndexB < m_bodyCount);

            // Initialize the body state.
            for (int i = 0; i < m_bodyCount; ++i) {
                b2Body b = m_bodies[i];
                m_positions[i].c = b.m_sweep.c;
                m_positions[i].a = b.m_sweep.a;
                m_velocities[i].v = b.m_linearVelocity;
                m_velocities[i].w = b.m_angularVelocity;
            }

            b2ContactSolverDef contactSolverDef;
            contactSolverDef.contacts = m_contacts;
            contactSolverDef.count = m_contactCount;
            contactSolverDef.step = subStep;
            contactSolverDef.positions = m_positions;
            contactSolverDef.velocities = m_velocities;
            var contactSolver = new b2ContactSolver(contactSolverDef);

            // Solve position constraints.
            for (int i = 0; i < subStep.positionIterations; ++i) {
                bool contactsOkay = contactSolver.SolveTOIPositionConstraints(toiIndexA, toiIndexB);
                if (contactsOkay)
                    break;
            }

            // Leap of faith to new safe state.
            m_bodies[toiIndexA].m_sweep.c0 = m_positions[toiIndexA].c;
            m_bodies[toiIndexA].m_sweep.a0 = m_positions[toiIndexA].a;
            m_bodies[toiIndexB].m_sweep.c0 = m_positions[toiIndexB].c;
            m_bodies[toiIndexB].m_sweep.a0 = m_positions[toiIndexB].a;

            // No warm starting is needed for TOI events because warm
            // starting impulses were applied in the discrete solver.
            contactSolver.InitializeVelocityConstraints();

            // Solve velocity constraints.
            for (int i = 0; i < subStep.velocityIterations; ++i)
                contactSolver.SolveVelocityConstraints();

            // Don't store the TOI contact forces for warm starting
            // because they can be quite large.

            float h = subStep.dt;

            // Integrate positions
            for (int i = 0; i < m_bodyCount; ++i) {
                b2Vec2 c = m_positions[i].c;
                float a = m_positions[i].a;
                b2Vec2 v = m_velocities[i].v;
                float w = m_velocities[i].w;

                // Check for large velocities
                b2Vec2 translation = h * v;
                if (b2Dot(translation, translation) > b2_maxTranslationSquared) {
                    float ratio = b2_maxTranslation / translation.Length();
                    v *= ratio;
                }

                float rotation = h * w;
                if (rotation * rotation > b2_maxRotationSquared) {
                    float ratio = b2_maxRotation / b2Abs(rotation);
                    w *= ratio;
                }

                // Integrate
                c += h * v;
                a += h * w;

                m_positions[i].c = c;
                m_positions[i].a = a;
                m_velocities[i].v = v;
                m_velocities[i].w = w;

                // Sync bodies
                b2Body body = m_bodies[i];
                body.m_sweep.c = c;
                body.m_sweep.a = a;
                body.m_linearVelocity = v;
                body.m_angularVelocity = w;
                body.SynchronizeTransform();
            }

            Report(contactSolver.m_velocityConstraints);
        }

        /// <summary>Tells the listener each contact's solved impulses (PostSolve).</summary>
        /// <param name="constraints">The solver's velocity constraints, in contact order.</param>
        public void Report(b2ContactVelocityConstraint[] constraints) {
            if (m_listener == null)
                return;

            for (int i = 0; i < m_contactCount; ++i) {
                b2Contact c = m_contacts[i];

                ref b2ContactVelocityConstraint vc = ref constraints[i];

                b2ContactImpulse impulse = default;
                impulse.count = vc.pointCount;
                for (int j = 0; j < vc.pointCount; ++j) {
                    impulse.normalImpulses[j] = vc.points[j].normalImpulse;
                    impulse.tangentImpulses[j] = vc.points[j].tangentImpulse;
                }

                m_listener.PostSolve(c, impulse);
            }
        }
    }
}
