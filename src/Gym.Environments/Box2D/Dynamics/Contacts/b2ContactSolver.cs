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
// Altered source version (Gym.NET, 2026-11-10): Box2D/Dynamics/Contacts/b2ContactSolver.h and b2ContactSolver.cpp of
// pybox2d 2.3.10's bundled Box2D 2.3.0, translated to C#. The constraint arrays are managed arrays (C++ takes them from
// the stack allocator); the B2_DEBUG_SOLVER postconditions are left out, as the C++ build compiles them out.
using System.Runtime.CompilerServices;
using static Gym.Box2D.b2Math;
using static Gym.Box2D.b2Settings;

namespace Gym.Box2D {
    /// <summary>A velocity constraint point: anchors relative to the centers, accumulated impulses, effective masses and bias.</summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Dynamics/Contacts/b2ContactSolver.h">Box2D/Dynamics/Contacts/b2ContactSolver.h</see>.</remarks>
    internal struct b2VelocityConstraintPoint {
        /// <summary>The anchor relative to body A's center.</summary>
        public b2Vec2 rA;

        /// <summary>The anchor relative to body B's center.</summary>
        public b2Vec2 rB;

        /// <summary>The accumulated normal impulse.</summary>
        public float normalImpulse;

        /// <summary>The accumulated tangent impulse.</summary>
        public float tangentImpulse;

        /// <summary>The effective normal mass.</summary>
        public float normalMass;

        /// <summary>The effective tangent mass.</summary>
        public float tangentMass;

        /// <summary>The restitution velocity bias.</summary>
        public float velocityBias;
    }

    /// <summary>The two velocity constraint points, inline.</summary>
    [InlineArray(b2_maxManifoldPoints)]
    internal struct b2VelocityConstraintPointArray {
        /// <summary>The first element; the runtime lays the second after it.</summary>
        private b2VelocityConstraintPoint _element0;
    }

    /// <summary>A contact's velocity constraint.</summary>
    internal struct b2ContactVelocityConstraint {
        /// <summary>The constraint points.</summary>
        public b2VelocityConstraintPointArray points;

        /// <summary>The world normal.</summary>
        public b2Vec2 normal;

        /// <summary>The inverse of <see cref="K"/>, for the block solver.</summary>
        public b2Mat22 normalMass;

        /// <summary>The block solver's 2x2 effective-mass matrix.</summary>
        public b2Mat22 K;

        /// <summary>Body A's island index.</summary>
        public int indexA;

        /// <summary>Body B's island index.</summary>
        public int indexB;

        /// <summary>Body A's inverse mass.</summary>
        public float invMassA;

        /// <summary>Body B's inverse mass.</summary>
        public float invMassB;

        /// <summary>Body A's inverse inertia.</summary>
        public float invIA;

        /// <summary>Body B's inverse inertia.</summary>
        public float invIB;

        /// <summary>The mixed friction.</summary>
        public float friction;

        /// <summary>The mixed restitution.</summary>
        public float restitution;

        /// <summary>The conveyor-belt tangent speed.</summary>
        public float tangentSpeed;

        /// <summary>The number of points (2 may drop to 1 when the block is ill-conditioned).</summary>
        public int pointCount;

        /// <summary>The contact's index in the solver's contact array.</summary>
        public int contactIndex;
    }

    /// <summary>A contact's position constraint, in local coordinates so it can be re-evaluated as the bodies move.</summary>
    internal struct b2ContactPositionConstraint {
        /// <summary>The manifold's local points.</summary>
        public b2Vec2Pair localPoints;

        /// <summary>The manifold's local normal.</summary>
        public b2Vec2 localNormal;

        /// <summary>The manifold's local point.</summary>
        public b2Vec2 localPoint;

        /// <summary>Body A's island index.</summary>
        public int indexA;

        /// <summary>Body B's island index.</summary>
        public int indexB;

        /// <summary>Body A's inverse mass.</summary>
        public float invMassA;

        /// <summary>Body B's inverse mass.</summary>
        public float invMassB;

        /// <summary>Body A's local center of mass.</summary>
        public b2Vec2 localCenterA;

        /// <summary>Body B's local center of mass.</summary>
        public b2Vec2 localCenterB;

        /// <summary>Body A's inverse inertia.</summary>
        public float invIA;

        /// <summary>Body B's inverse inertia.</summary>
        public float invIB;

        /// <summary>The manifold type.</summary>
        public b2ManifoldType type;

        /// <summary>Shape A's radius.</summary>
        public float radiusA;

        /// <summary>Shape B's radius.</summary>
        public float radiusB;

        /// <summary>The number of points.</summary>
        public int pointCount;
    }

    /// <summary>What a contact solver is built from.</summary>
    internal struct b2ContactSolverDef {
        /// <summary>The step.</summary>
        public b2TimeStep step;

        /// <summary>The island's contacts.</summary>
        public b2Contact[] contacts;

        /// <summary>How many of <see cref="contacts"/> to solve.</summary>
        public int count;

        /// <summary>The island's positions.</summary>
        public b2Position[] positions;

        /// <summary>The island's velocities.</summary>
        public b2Velocity[] velocities;
    }

    /// <summary>A contact point's evaluation by the position solver: normal, point and separation.</summary>
    internal struct b2PositionSolverManifold {
        /// <summary>The world normal from A to B.</summary>
        public b2Vec2 normal;

        /// <summary>The world point.</summary>
        public b2Vec2 point;

        /// <summary>The separation, negative when penetrating.</summary>
        public float separation;

        /// <summary>Evaluates point <paramref name="index"/> of a position constraint under the given transforms.</summary>
        /// <param name="pc">The constraint.</param>
        /// <param name="xfA">Body A's transform.</param>
        /// <param name="xfB">Body B's transform.</param>
        /// <param name="index">The point.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown for a constraint without points.</exception>
        public void Initialize(in b2ContactPositionConstraint pc, in b2Transform xfA, in b2Transform xfB, int index) {
            b2Assert(pc.pointCount > 0, "pc->pointCount > 0");

            switch (pc.type) {
                case b2ManifoldType.e_circles: {
                    b2Vec2 pointA = b2Mul(xfA, pc.localPoint);
                    b2Vec2 pointB = b2Mul(xfB, pc.localPoints[0]);
                    normal = pointB - pointA;
                    normal.Normalize();
                    point = 0.5f * (pointA + pointB);
                    separation = b2Dot(pointB - pointA, normal) - pc.radiusA - pc.radiusB;
                }
                    break;

                case b2ManifoldType.e_faceA: {
                    normal = b2Mul(xfA.q, pc.localNormal);
                    b2Vec2 planePoint = b2Mul(xfA, pc.localPoint);

                    b2Vec2 clipPoint = b2Mul(xfB, pc.localPoints[index]);
                    separation = b2Dot(clipPoint - planePoint, normal) - pc.radiusA - pc.radiusB;
                    point = clipPoint;
                }
                    break;

                case b2ManifoldType.e_faceB: {
                    normal = b2Mul(xfB.q, pc.localNormal);
                    b2Vec2 planePoint = b2Mul(xfB, pc.localPoint);

                    b2Vec2 clipPoint = b2Mul(xfA, pc.localPoints[index]);
                    separation = b2Dot(clipPoint - planePoint, normal) - pc.radiusA - pc.radiusB;
                    point = clipPoint;

                    // Ensure normal points from A to B
                    normal = -normal;
                }
                    break;
            }
        }
    }

    /// <summary>
    ///     The contact solver of an island: sequential impulses with friction first, a 2x2 block solver for two-point
    ///     manifolds (total enumeration, after Dirk Gregorius), and sequential position correction.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Dynamics/Contacts/b2ContactSolver.cpp">Box2D/Dynamics/Contacts/b2ContactSolver.cpp</see>.</remarks>
    internal sealed class b2ContactSolver {
        /// <summary>The step.</summary>
        public b2TimeStep m_step;

        /// <summary>The island's positions.</summary>
        public b2Position[] m_positions;

        /// <summary>The island's velocities.</summary>
        public b2Velocity[] m_velocities;

        /// <summary>The position constraints.</summary>
        public b2ContactPositionConstraint[] m_positionConstraints;

        /// <summary>The velocity constraints.</summary>
        public b2ContactVelocityConstraint[] m_velocityConstraints;

        /// <summary>The contacts.</summary>
        public b2Contact[] m_contacts;

        /// <summary>The number of contacts.</summary>
        public int m_count;

        /// <summary>Builds the position-independent parts of the constraints, warm-start impulses scaled by the step ratio.</summary>
        /// <param name="def">The contacts, positions, velocities and step.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown for a contact without manifold points.</exception>
        public b2ContactSolver(in b2ContactSolverDef def) {
            m_step = def.step;
            m_count = def.count;
            m_positionConstraints = new b2ContactPositionConstraint[m_count];
            m_velocityConstraints = new b2ContactVelocityConstraint[m_count];
            m_positions = def.positions;
            m_velocities = def.velocities;
            m_contacts = def.contacts;

            // Initialize position independent portions of the constraints.
            for (int i = 0; i < m_count; ++i) {
                b2Contact contact = m_contacts[i];

                b2Fixture fixtureA = contact.m_fixtureA!;
                b2Fixture fixtureB = contact.m_fixtureB!;
                b2Shape shapeA = fixtureA.GetShape();
                b2Shape shapeB = fixtureB.GetShape();
                float radiusA = shapeA.m_radius;
                float radiusB = shapeB.m_radius;
                b2Body bodyA = fixtureA.GetBody();
                b2Body bodyB = fixtureB.GetBody();
                ref b2Manifold manifold = ref contact.GetManifold();

                int pointCount = manifold.pointCount;
                b2Assert(pointCount > 0);

                ref b2ContactVelocityConstraint vc = ref m_velocityConstraints[i];
                vc.friction = contact.m_friction;
                vc.restitution = contact.m_restitution;
                vc.tangentSpeed = contact.m_tangentSpeed;
                vc.indexA = bodyA.m_islandIndex;
                vc.indexB = bodyB.m_islandIndex;
                vc.invMassA = bodyA.m_invMass;
                vc.invMassB = bodyB.m_invMass;
                vc.invIA = bodyA.m_invI;
                vc.invIB = bodyB.m_invI;
                vc.contactIndex = i;
                vc.pointCount = pointCount;
                vc.K.SetZero();
                vc.normalMass.SetZero();

                ref b2ContactPositionConstraint pc = ref m_positionConstraints[i];
                pc.indexA = bodyA.m_islandIndex;
                pc.indexB = bodyB.m_islandIndex;
                pc.invMassA = bodyA.m_invMass;
                pc.invMassB = bodyB.m_invMass;
                pc.localCenterA = bodyA.m_sweep.localCenter;
                pc.localCenterB = bodyB.m_sweep.localCenter;
                pc.invIA = bodyA.m_invI;
                pc.invIB = bodyB.m_invI;
                pc.localNormal = manifold.localNormal;
                pc.localPoint = manifold.localPoint;
                pc.pointCount = pointCount;
                pc.radiusA = radiusA;
                pc.radiusB = radiusB;
                pc.type = manifold.type;

                for (int j = 0; j < pointCount; ++j) {
                    ref b2ManifoldPoint cp = ref manifold.points[j];
                    ref b2VelocityConstraintPoint vcp = ref vc.points[j];

                    if (m_step.warmStarting) {
                        vcp.normalImpulse = m_step.dtRatio * cp.normalImpulse;
                        vcp.tangentImpulse = m_step.dtRatio * cp.tangentImpulse;
                    } else {
                        vcp.normalImpulse = 0.0f;
                        vcp.tangentImpulse = 0.0f;
                    }

                    vcp.rA.SetZero();
                    vcp.rB.SetZero();
                    vcp.normalMass = 0.0f;
                    vcp.tangentMass = 0.0f;
                    vcp.velocityBias = 0.0f;

                    pc.localPoints[j] = cp.localPoint;
                }
            }
        }

        /// <summary>Initializes the position-dependent parts of the velocity constraints (anchors, effective masses, restitution bias, the block).</summary>
        /// <exception cref="NumSharp.AssertionError">Thrown for a contact without manifold points.</exception>
        public void InitializeVelocityConstraints() {
            for (int i = 0; i < m_count; ++i) {
                ref b2ContactVelocityConstraint vc = ref m_velocityConstraints[i];
                ref b2ContactPositionConstraint pc = ref m_positionConstraints[i];

                float radiusA = pc.radiusA;
                float radiusB = pc.radiusB;
                ref b2Manifold manifold = ref m_contacts[vc.contactIndex].GetManifold();

                int indexA = vc.indexA;
                int indexB = vc.indexB;

                float mA = vc.invMassA;
                float mB = vc.invMassB;
                float iA = vc.invIA;
                float iB = vc.invIB;
                b2Vec2 localCenterA = pc.localCenterA;
                b2Vec2 localCenterB = pc.localCenterB;

                b2Vec2 cA = m_positions[indexA].c;
                float aA = m_positions[indexA].a;
                b2Vec2 vA = m_velocities[indexA].v;
                float wA = m_velocities[indexA].w;

                b2Vec2 cB = m_positions[indexB].c;
                float aB = m_positions[indexB].a;
                b2Vec2 vB = m_velocities[indexB].v;
                float wB = m_velocities[indexB].w;

                b2Assert(manifold.pointCount > 0, "manifold->pointCount > 0");

                b2Transform xfA = default, xfB = default;
                xfA.q.Set(aA);
                xfB.q.Set(aB);
                xfA.p = cA - b2Mul(xfA.q, localCenterA);
                xfB.p = cB - b2Mul(xfB.q, localCenterB);

                b2WorldManifold worldManifold = default;
                worldManifold.Initialize(manifold, xfA, radiusA, xfB, radiusB);

                vc.normal = worldManifold.normal;

                int pointCount = vc.pointCount;
                for (int j = 0; j < pointCount; ++j) {
                    ref b2VelocityConstraintPoint vcp = ref vc.points[j];

                    vcp.rA = worldManifold.points[j] - cA;
                    vcp.rB = worldManifold.points[j] - cB;

                    float rnA = b2Cross(vcp.rA, vc.normal);
                    float rnB = b2Cross(vcp.rB, vc.normal);

                    float kNormal = mA + mB + iA * rnA * rnA + iB * rnB * rnB;

                    vcp.normalMass = kNormal > 0.0f ? 1.0f / kNormal : 0.0f;

                    b2Vec2 tangent = b2Cross(vc.normal, 1.0f);

                    float rtA = b2Cross(vcp.rA, tangent);
                    float rtB = b2Cross(vcp.rB, tangent);

                    float kTangent = mA + mB + iA * rtA * rtA + iB * rtB * rtB;

                    vcp.tangentMass = kTangent > 0.0f ? 1.0f / kTangent : 0.0f;

                    // Setup a velocity bias for restitution.
                    vcp.velocityBias = 0.0f;
                    float vRel = b2Dot(vc.normal, vB + b2Cross(wB, vcp.rB) - vA - b2Cross(wA, vcp.rA));
                    if (vRel < -b2_velocityThreshold)
                        vcp.velocityBias = -vc.restitution * vRel;
                }

                // If we have two points, then prepare the block solver.
                if (vc.pointCount == 2) {
                    ref b2VelocityConstraintPoint vcp1 = ref vc.points[0];
                    ref b2VelocityConstraintPoint vcp2 = ref vc.points[1];

                    float rn1A = b2Cross(vcp1.rA, vc.normal);
                    float rn1B = b2Cross(vcp1.rB, vc.normal);
                    float rn2A = b2Cross(vcp2.rA, vc.normal);
                    float rn2B = b2Cross(vcp2.rB, vc.normal);

                    float k11 = mA + mB + iA * rn1A * rn1A + iB * rn1B * rn1B;
                    float k22 = mA + mB + iA * rn2A * rn2A + iB * rn2B * rn2B;
                    float k12 = mA + mB + iA * rn1A * rn2A + iB * rn1B * rn2B;

                    // Ensure a reasonable condition number.
                    const float k_maxConditionNumber = 1000.0f;
                    if (k11 * k11 < k_maxConditionNumber * (k11 * k22 - k12 * k12)) {
                        // K is safe to invert.
                        vc.K.ex.Set(k11, k12);
                        vc.K.ey.Set(k12, k22);
                        vc.normalMass = vc.K.GetInverse();
                    } else {
                        // The constraints are redundant, just use one.
                        // TODO_ERIN use deepest?
                        vc.pointCount = 1;
                    }
                }
            }
        }

        /// <summary>Applies the warm-start impulses to the velocities.</summary>
        public void WarmStart() {
            // Warm start.
            for (int i = 0; i < m_count; ++i) {
                ref b2ContactVelocityConstraint vc = ref m_velocityConstraints[i];

                int indexA = vc.indexA;
                int indexB = vc.indexB;
                float mA = vc.invMassA;
                float iA = vc.invIA;
                float mB = vc.invMassB;
                float iB = vc.invIB;
                int pointCount = vc.pointCount;

                b2Vec2 vA = m_velocities[indexA].v;
                float wA = m_velocities[indexA].w;
                b2Vec2 vB = m_velocities[indexB].v;
                float wB = m_velocities[indexB].w;

                b2Vec2 normal = vc.normal;
                b2Vec2 tangent = b2Cross(normal, 1.0f);

                for (int j = 0; j < pointCount; ++j) {
                    ref b2VelocityConstraintPoint vcp = ref vc.points[j];
                    b2Vec2 P = vcp.normalImpulse * normal + vcp.tangentImpulse * tangent;
                    wA -= iA * b2Cross(vcp.rA, P);
                    vA -= mA * P;
                    wB += iB * b2Cross(vcp.rB, P);
                    vB += mB * P;
                }

                m_velocities[indexA].v = vA;
                m_velocities[indexA].w = wA;
                m_velocities[indexB].v = vB;
                m_velocities[indexB].w = wB;
            }
        }

        /// <summary>One velocity iteration: friction per point, then the normal constraints (one point, or the 2x2 block).</summary>
        /// <exception cref="NumSharp.AssertionError">Thrown for a constraint of other than 1 or 2 points, or a negative accumulated impulse.</exception>
        public void SolveVelocityConstraints() {
            for (int i = 0; i < m_count; ++i) {
                ref b2ContactVelocityConstraint vc = ref m_velocityConstraints[i];

                int indexA = vc.indexA;
                int indexB = vc.indexB;
                float mA = vc.invMassA;
                float iA = vc.invIA;
                float mB = vc.invMassB;
                float iB = vc.invIB;
                int pointCount = vc.pointCount;

                b2Vec2 vA = m_velocities[indexA].v;
                float wA = m_velocities[indexA].w;
                b2Vec2 vB = m_velocities[indexB].v;
                float wB = m_velocities[indexB].w;

                b2Vec2 normal = vc.normal;
                b2Vec2 tangent = b2Cross(normal, 1.0f);
                float friction = vc.friction;

                b2Assert(pointCount == 1 || pointCount == 2);

                // Solve tangent constraints first because non-penetration is more important
                // than friction.
                for (int j = 0; j < pointCount; ++j) {
                    ref b2VelocityConstraintPoint vcp = ref vc.points[j];

                    // Relative velocity at contact
                    b2Vec2 dv = vB + b2Cross(wB, vcp.rB) - vA - b2Cross(wA, vcp.rA);

                    // Compute tangent force
                    float vt = b2Dot(dv, tangent) - vc.tangentSpeed;
                    float lambda = vcp.tangentMass * (-vt);

                    // b2Clamp the accumulated force
                    float maxFriction = friction * vcp.normalImpulse;
                    float newImpulse = b2Clamp(vcp.tangentImpulse + lambda, -maxFriction, maxFriction);
                    lambda = newImpulse - vcp.tangentImpulse;
                    vcp.tangentImpulse = newImpulse;

                    // Apply contact impulse
                    b2Vec2 P = lambda * tangent;

                    vA -= mA * P;
                    wA -= iA * b2Cross(vcp.rA, P);

                    vB += mB * P;
                    wB += iB * b2Cross(vcp.rB, P);
                }

                // Solve normal constraints
                if (vc.pointCount == 1) {
                    ref b2VelocityConstraintPoint vcp = ref vc.points[0];

                    // Relative velocity at contact
                    b2Vec2 dv = vB + b2Cross(wB, vcp.rB) - vA - b2Cross(wA, vcp.rA);

                    // Compute normal impulse
                    float vn = b2Dot(dv, normal);
                    float lambda = -vcp.normalMass * (vn - vcp.velocityBias);

                    // b2Clamp the accumulated impulse
                    float newImpulse = b2Max(vcp.normalImpulse + lambda, 0.0f);
                    lambda = newImpulse - vcp.normalImpulse;
                    vcp.normalImpulse = newImpulse;

                    // Apply contact impulse
                    b2Vec2 P = lambda * normal;
                    vA -= mA * P;
                    wA -= iA * b2Cross(vcp.rA, P);

                    vB += mB * P;
                    wB += iB * b2Cross(vcp.rB, P);
                } else {
                    // Block solver developed in collaboration with Dirk Gregorius (back in 01/07 on Box2D_Lite).
                    // Build the mini LCP for this contact patch
                    //
                    // vn = A * x + b, vn >= 0, , vn >= 0, x >= 0 and vn_i * x_i = 0 with i = 1..2
                    //
                    // A = J * W * JT and J = ( -n, -r1 x n, n, r2 x n )
                    // b = vn0 - velocityBias
                    //
                    // The system is solved using the "Total enumeration method" (s. Murty). The complementary constraint vn_i * x_i
                    // implies that we must have in any solution either vn_i = 0 or x_i = 0. So for the 2D contact problem the cases
                    // vn1 = 0 and vn2 = 0, x1 = 0 and x2 = 0, x1 = 0 and vn2 = 0, x2 = 0 and vn1 = 0 need to be tested. The first valid
                    // solution that satisfies the problem is chosen.
                    //
                    // In order to account of the accumulated impulse 'a' (because of the iterative nature of the solver which only requires
                    // that the accumulated impulse is clamped and not the incremental impulse) we change the impulse variable (x_i).
                    //
                    // Substitute:
                    //
                    // x = a + d
                    //
                    // a := old total impulse
                    // x := new total impulse
                    // d := incremental impulse
                    //
                    // For the current iteration we extend the formula for the incremental impulse
                    // to compute the new total impulse:
                    //
                    // vn = A * d + b
                    //    = A * (x - a) + b
                    //    = A * x + b - A * a
                    //    = A * x + b'
                    // b' = b - A * a;

                    ref b2VelocityConstraintPoint cp1 = ref vc.points[0];
                    ref b2VelocityConstraintPoint cp2 = ref vc.points[1];

                    b2Vec2 a = new b2Vec2(cp1.normalImpulse, cp2.normalImpulse);
                    b2Assert(a.x >= 0.0f && a.y >= 0.0f);

                    // Relative velocity at contact
                    b2Vec2 dv1 = vB + b2Cross(wB, cp1.rB) - vA - b2Cross(wA, cp1.rA);
                    b2Vec2 dv2 = vB + b2Cross(wB, cp2.rB) - vA - b2Cross(wA, cp2.rA);

                    // Compute normal velocity
                    float vn1 = b2Dot(dv1, normal);
                    float vn2 = b2Dot(dv2, normal);

                    b2Vec2 b;
                    b.x = vn1 - cp1.velocityBias;
                    b.y = vn2 - cp2.velocityBias;

                    // Compute b'
                    b -= b2Mul(vc.K, a);

                    for (; ; ) {
                        //
                        // Case 1: vn = 0
                        //
                        // 0 = A * x + b'
                        //
                        // Solve for x:
                        //
                        // x = - inv(A) * b'
                        //
                        b2Vec2 x = -b2Mul(vc.normalMass, b);

                        if (x.x >= 0.0f && x.y >= 0.0f) {
                            // Get the incremental impulse
                            b2Vec2 d = x - a;

                            // Apply incremental impulse
                            b2Vec2 P1 = d.x * normal;
                            b2Vec2 P2 = d.y * normal;
                            vA -= mA * (P1 + P2);
                            wA -= iA * (b2Cross(cp1.rA, P1) + b2Cross(cp2.rA, P2));

                            vB += mB * (P1 + P2);
                            wB += iB * (b2Cross(cp1.rB, P1) + b2Cross(cp2.rB, P2));

                            // Accumulate
                            cp1.normalImpulse = x.x;
                            cp2.normalImpulse = x.y;
                            break;
                        }

                        //
                        // Case 2: vn1 = 0 and x2 = 0
                        //
                        //   0 = a11 * x1 + a12 * 0 + b1'
                        // vn2 = a21 * x1 + a22 * 0 + b2'
                        //
                        x.x = -cp1.normalMass * b.x;
                        x.y = 0.0f;
                        vn1 = 0.0f;
                        vn2 = vc.K.ex.y * x.x + b.y;

                        if (x.x >= 0.0f && vn2 >= 0.0f) {
                            // Get the incremental impulse
                            b2Vec2 d = x - a;

                            // Apply incremental impulse
                            b2Vec2 P1 = d.x * normal;
                            b2Vec2 P2 = d.y * normal;
                            vA -= mA * (P1 + P2);
                            wA -= iA * (b2Cross(cp1.rA, P1) + b2Cross(cp2.rA, P2));

                            vB += mB * (P1 + P2);
                            wB += iB * (b2Cross(cp1.rB, P1) + b2Cross(cp2.rB, P2));

                            // Accumulate
                            cp1.normalImpulse = x.x;
                            cp2.normalImpulse = x.y;
                            break;
                        }

                        //
                        // Case 3: vn2 = 0 and x1 = 0
                        //
                        // vn1 = a11 * 0 + a12 * x2 + b1'
                        //   0 = a21 * 0 + a22 * x2 + b2'
                        //
                        x.x = 0.0f;
                        x.y = -cp2.normalMass * b.y;
                        vn1 = vc.K.ey.x * x.y + b.x;
                        vn2 = 0.0f;

                        if (x.y >= 0.0f && vn1 >= 0.0f) {
                            // Resubstitute for the incremental impulse
                            b2Vec2 d = x - a;

                            // Apply incremental impulse
                            b2Vec2 P1 = d.x * normal;
                            b2Vec2 P2 = d.y * normal;
                            vA -= mA * (P1 + P2);
                            wA -= iA * (b2Cross(cp1.rA, P1) + b2Cross(cp2.rA, P2));

                            vB += mB * (P1 + P2);
                            wB += iB * (b2Cross(cp1.rB, P1) + b2Cross(cp2.rB, P2));

                            // Accumulate
                            cp1.normalImpulse = x.x;
                            cp2.normalImpulse = x.y;
                            break;
                        }

                        //
                        // Case 4: x1 = 0 and x2 = 0
                        //
                        // vn1 = b1
                        // vn2 = b2;
                        x.x = 0.0f;
                        x.y = 0.0f;
                        vn1 = b.x;
                        vn2 = b.y;

                        if (vn1 >= 0.0f && vn2 >= 0.0f) {
                            // Resubstitute for the incremental impulse
                            b2Vec2 d = x - a;

                            // Apply incremental impulse
                            b2Vec2 P1 = d.x * normal;
                            b2Vec2 P2 = d.y * normal;
                            vA -= mA * (P1 + P2);
                            wA -= iA * (b2Cross(cp1.rA, P1) + b2Cross(cp2.rA, P2));

                            vB += mB * (P1 + P2);
                            wB += iB * (b2Cross(cp1.rB, P1) + b2Cross(cp2.rB, P2));

                            // Accumulate
                            cp1.normalImpulse = x.x;
                            cp2.normalImpulse = x.y;

                            break;
                        }

                        // No solution, give up. This is hit sometimes, but it doesn't seem to matter.
                        break;
                    }
                }

                m_velocities[indexA].v = vA;
                m_velocities[indexA].w = wA;
                m_velocities[indexB].v = vB;
                m_velocities[indexB].w = wB;
            }
        }

        /// <summary>Copies the accumulated impulses back into the manifolds, for the next step's warm start.</summary>
        public void StoreImpulses() {
            for (int i = 0; i < m_count; ++i) {
                ref b2ContactVelocityConstraint vc = ref m_velocityConstraints[i];
                ref b2Manifold manifold = ref m_contacts[vc.contactIndex].GetManifold();

                for (int j = 0; j < vc.pointCount; ++j) {
                    manifold.points[j].normalImpulse = vc.points[j].normalImpulse;
                    manifold.points[j].tangentImpulse = vc.points[j].tangentImpulse;
                }
            }
        }

        /// <summary>One sequential position iteration (Baumgarte 0.2, slop allowed, corrections capped).</summary>
        /// <returns>True when the worst separation is within <c>-3 * b2_linearSlop</c>.</returns>
        public bool SolvePositionConstraints() {
            float minSeparation = 0.0f;

            for (int i = 0; i < m_count; ++i) {
                ref b2ContactPositionConstraint pc = ref m_positionConstraints[i];

                int indexA = pc.indexA;
                int indexB = pc.indexB;
                b2Vec2 localCenterA = pc.localCenterA;
                float mA = pc.invMassA;
                float iA = pc.invIA;
                b2Vec2 localCenterB = pc.localCenterB;
                float mB = pc.invMassB;
                float iB = pc.invIB;
                int pointCount = pc.pointCount;

                b2Vec2 cA = m_positions[indexA].c;
                float aA = m_positions[indexA].a;

                b2Vec2 cB = m_positions[indexB].c;
                float aB = m_positions[indexB].a;

                // Solve normal constraints
                for (int j = 0; j < pointCount; ++j) {
                    b2Transform xfA = default, xfB = default;
                    xfA.q.Set(aA);
                    xfB.q.Set(aB);
                    xfA.p = cA - b2Mul(xfA.q, localCenterA);
                    xfB.p = cB - b2Mul(xfB.q, localCenterB);

                    b2PositionSolverManifold psm = default;
                    psm.Initialize(pc, xfA, xfB, j);
                    b2Vec2 normal = psm.normal;

                    b2Vec2 point = psm.point;
                    float separation = psm.separation;

                    b2Vec2 rA = point - cA;
                    b2Vec2 rB = point - cB;

                    // Track max constraint error.
                    minSeparation = b2Min(minSeparation, separation);

                    // Prevent large corrections and allow slop.
                    float C = b2Clamp(b2_baumgarte * (separation + b2_linearSlop), -b2_maxLinearCorrection, 0.0f);

                    // Compute the effective mass.
                    float rnA = b2Cross(rA, normal);
                    float rnB = b2Cross(rB, normal);
                    float K = mA + mB + iA * rnA * rnA + iB * rnB * rnB;

                    // Compute normal impulse
                    float impulse = K > 0.0f ? -C / K : 0.0f;

                    b2Vec2 P = impulse * normal;

                    cA -= mA * P;
                    aA -= iA * b2Cross(rA, P);

                    cB += mB * P;
                    aB += iB * b2Cross(rB, P);
                }

                m_positions[indexA].c = cA;
                m_positions[indexA].a = aA;

                m_positions[indexB].c = cB;
                m_positions[indexB].a = aB;
            }

            // We can't expect minSpeparation >= -b2_linearSlop because we don't
            // push the separation above -b2_linearSlop.
            return minSeparation >= -3.0f * b2_linearSlop;
        }

        /// <summary>
        ///     One position iteration of a time-of-impact sub-step: only the two TOI bodies move (Baumgarte 0.75), the others
        ///     act as static.
        /// </summary>
        /// <param name="toiIndexA">The first TOI body's island index.</param>
        /// <param name="toiIndexB">The second TOI body's island index.</param>
        /// <returns>True when the worst separation is within <c>-1.5 * b2_linearSlop</c>.</returns>
        public bool SolveTOIPositionConstraints(int toiIndexA, int toiIndexB) {
            float minSeparation = 0.0f;

            for (int i = 0; i < m_count; ++i) {
                ref b2ContactPositionConstraint pc = ref m_positionConstraints[i];

                int indexA = pc.indexA;
                int indexB = pc.indexB;
                b2Vec2 localCenterA = pc.localCenterA;
                b2Vec2 localCenterB = pc.localCenterB;
                int pointCount = pc.pointCount;

                float mA = 0.0f;
                float iA = 0.0f;
                if (indexA == toiIndexA || indexA == toiIndexB) {
                    mA = pc.invMassA;
                    iA = pc.invIA;
                }

                float mB = 0.0f;
                float iB = 0.0f;
                if (indexB == toiIndexA || indexB == toiIndexB) {
                    mB = pc.invMassB;
                    iB = pc.invIB;
                }

                b2Vec2 cA = m_positions[indexA].c;
                float aA = m_positions[indexA].a;

                b2Vec2 cB = m_positions[indexB].c;
                float aB = m_positions[indexB].a;

                // Solve normal constraints
                for (int j = 0; j < pointCount; ++j) {
                    b2Transform xfA = default, xfB = default;
                    xfA.q.Set(aA);
                    xfB.q.Set(aB);
                    xfA.p = cA - b2Mul(xfA.q, localCenterA);
                    xfB.p = cB - b2Mul(xfB.q, localCenterB);

                    b2PositionSolverManifold psm = default;
                    psm.Initialize(pc, xfA, xfB, j);
                    b2Vec2 normal = psm.normal;

                    b2Vec2 point = psm.point;
                    float separation = psm.separation;

                    b2Vec2 rA = point - cA;
                    b2Vec2 rB = point - cB;

                    // Track max constraint error.
                    minSeparation = b2Min(minSeparation, separation);

                    // Prevent large corrections and allow slop.
                    float C = b2Clamp(b2_toiBaugarte * (separation + b2_linearSlop), -b2_maxLinearCorrection, 0.0f);

                    // Compute the effective mass.
                    float rnA = b2Cross(rA, normal);
                    float rnB = b2Cross(rB, normal);
                    float K = mA + mB + iA * rnA * rnA + iB * rnB * rnB;

                    // Compute normal impulse
                    float impulse = K > 0.0f ? -C / K : 0.0f;

                    b2Vec2 P = impulse * normal;

                    cA -= mA * P;
                    aA -= iA * b2Cross(rA, P);

                    cB += mB * P;
                    aB += iB * b2Cross(rB, P);
                }

                m_positions[indexA].c = cA;
                m_positions[indexA].a = aA;

                m_positions[indexB].c = cB;
                m_positions[indexB].a = aB;
            }

            // We can't expect minSpeparation >= -b2_linearSlop because we don't
            // push the separation above -b2_linearSlop.
            return minSeparation >= -1.5f * b2_linearSlop;
        }
    }
}
