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
// Altered source version (Gym.NET, 2026-11-10): Box2D/Dynamics/Joints/b2RevoluteJoint.h and b2RevoluteJoint.cpp of
// pybox2d 2.3.10's bundled Box2D 2.3.0, translated to C#; Dump is left out.
using static Gym.Box2D.b2Math;
using static Gym.Box2D.b2Settings;

namespace Gym.Box2D {
    /// <summary>
    ///     A revolute joint definition (pybox2d's <c>revoluteJointDef</c>): two bodies share an anchor point, with an
    ///     optional angle limit and an optional motor.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Dynamics/Joints/b2RevoluteJoint.h#L37">Box2D/Dynamics/Joints/b2RevoluteJoint.h</see>.</remarks>
    public sealed partial class b2RevoluteJointDef : b2JointDef {
        /// <summary>The anchor relative to body A's origin.</summary>
        internal b2Vec2 localAnchorA = new b2Vec2(0.0f, 0.0f);

        /// <summary>The anchor relative to body B's origin.</summary>
        internal b2Vec2 localAnchorB = new b2Vec2(0.0f, 0.0f);

        /// <summary>Body B's angle minus body A's in the reference state (radians).</summary>
        internal float referenceAngle = 0.0f;

        /// <summary>Whether the angle limit is enforced.</summary>
        internal bool enableLimit = false;

        /// <summary>The lower angle limit (radians).</summary>
        internal float lowerAngle = 0.0f;

        /// <summary>The upper angle limit (radians).</summary>
        internal float upperAngle = 0.0f;

        /// <summary>Whether the motor is enabled.</summary>
        internal bool enableMotor = false;

        /// <summary>The motor's target speed (rad/s).</summary>
        internal float motorSpeed = 0.0f;

        /// <summary>The motor's maximum torque (N·m).</summary>
        internal float maxMotorTorque = 0.0f;

        /// <summary>A revolute joint definition with no limit and no motor.</summary>
        public b2RevoluteJointDef() {
            type = b2JointType.e_revoluteJoint;
        }

        /// <summary>Sets the bodies, the anchors from a world anchor point, and the reference angle from the current angles.</summary>
        /// <param name="bA">Body A.</param>
        /// <param name="bB">Body B.</param>
        /// <param name="anchor">The world anchor point.</param>
        internal void Initialize(b2Body bA, b2Body bB, in b2Vec2 anchor) {
            bodyA = bA;
            bodyB = bB;
            localAnchorA = bodyA.GetLocalPoint(anchor);
            localAnchorB = bodyB.GetLocalPoint(anchor);
            referenceAngle = bodyB.GetAngle() - bodyA.GetAngle();
        }
    }

    /// <summary>
    ///     A revolute joint: a point-to-point constraint at the anchor, with an optional angle limit and an optional motor
    ///     that drives the relative angular velocity up to a maximum torque. BipedalWalker's and LunarLander's legs are
    ///     revolute joints.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Dynamics/Joints/b2RevoluteJoint.cpp">Box2D/Dynamics/Joints/b2RevoluteJoint.cpp</see>.</remarks>
    public sealed partial class b2RevoluteJoint : b2Joint {
        /// <summary>The anchor relative to body A's origin.</summary>
        internal b2Vec2 m_localAnchorA;

        /// <summary>The anchor relative to body B's origin.</summary>
        internal b2Vec2 m_localAnchorB;

        /// <summary>The accumulated point (x, y) and limit (z) impulses.</summary>
        internal b2Vec3 m_impulse;

        /// <summary>The accumulated motor impulse.</summary>
        internal float m_motorImpulse;

        /// <summary>Whether the motor is enabled.</summary>
        internal bool m_enableMotor;

        /// <summary>The motor's maximum torque.</summary>
        internal float m_maxMotorTorque;

        /// <summary>The motor's target speed.</summary>
        internal float m_motorSpeed;

        /// <summary>Whether the limit is enabled.</summary>
        internal bool m_enableLimit;

        /// <summary>The reference angle.</summary>
        internal float m_referenceAngle;

        /// <summary>The lower angle limit.</summary>
        internal float m_lowerAngle;

        /// <summary>The upper angle limit.</summary>
        internal float m_upperAngle;

        /// <summary>Body A's island index (solver temporary).</summary>
        internal int m_indexA;

        /// <summary>Body B's island index (solver temporary).</summary>
        internal int m_indexB;

        /// <summary>The anchor relative to body A's center (solver temporary).</summary>
        internal b2Vec2 m_rA;

        /// <summary>The anchor relative to body B's center (solver temporary).</summary>
        internal b2Vec2 m_rB;

        /// <summary>Body A's local center (solver temporary).</summary>
        internal b2Vec2 m_localCenterA;

        /// <summary>Body B's local center (solver temporary).</summary>
        internal b2Vec2 m_localCenterB;

        /// <summary>Body A's inverse mass (solver temporary).</summary>
        internal float m_invMassA;

        /// <summary>Body B's inverse mass (solver temporary).</summary>
        internal float m_invMassB;

        /// <summary>Body A's inverse inertia (solver temporary).</summary>
        internal float m_invIA;

        /// <summary>Body B's inverse inertia (solver temporary).</summary>
        internal float m_invIB;

        /// <summary>The effective mass for the point-to-point constraint.</summary>
        internal b2Mat33 m_mass;

        /// <summary>The effective mass for the motor/limit angular constraint.</summary>
        internal float m_motorMass;

        /// <summary>The limit's state.</summary>
        internal b2LimitState m_limitState;

        /// <summary>A revolute joint from its definition.</summary>
        /// <param name="def">The definition.</param>
        internal b2RevoluteJoint(b2RevoluteJointDef def) : base(def) {
            m_localAnchorA = def.localAnchorA;
            m_localAnchorB = def.localAnchorB;
            m_referenceAngle = def.referenceAngle;

            m_impulse.SetZero();
            m_motorImpulse = 0.0f;

            m_lowerAngle = def.lowerAngle;
            m_upperAngle = def.upperAngle;
            m_maxMotorTorque = def.maxMotorTorque;
            m_motorSpeed = def.motorSpeed;
            m_enableLimit = def.enableLimit;
            m_enableMotor = def.enableMotor;
            m_limitState = b2LimitState.e_inactiveLimit;
        }

        /// <summary>
        ///     Prepares the step: anchors, the 3x3 effective mass, the motor mass, the limit state, and the warm start
        ///     scaled by the step ratio.
        /// </summary>
        /// <param name="data">The solver data.</param>
        internal override void InitVelocityConstraints(in b2SolverData data) {
            m_indexA = m_bodyA.m_islandIndex;
            m_indexB = m_bodyB.m_islandIndex;
            m_localCenterA = m_bodyA.m_sweep.localCenter;
            m_localCenterB = m_bodyB.m_sweep.localCenter;
            m_invMassA = m_bodyA.m_invMass;
            m_invMassB = m_bodyB.m_invMass;
            m_invIA = m_bodyA.m_invI;
            m_invIB = m_bodyB.m_invI;

            float aA = data.positions[m_indexA].a;
            b2Vec2 vA = data.velocities[m_indexA].v;
            float wA = data.velocities[m_indexA].w;

            float aB = data.positions[m_indexB].a;
            b2Vec2 vB = data.velocities[m_indexB].v;
            float wB = data.velocities[m_indexB].w;

            b2Rot qA = new b2Rot(aA), qB = new b2Rot(aB);

            m_rA = b2Mul(qA, m_localAnchorA - m_localCenterA);
            m_rB = b2Mul(qB, m_localAnchorB - m_localCenterB);

            // J = [-I -r1_skew I r2_skew]
            //     [ 0       -1 0       1]
            // r_skew = [-ry; rx]

            // Matlab
            // K = [ mA+r1y^2*iA+mB+r2y^2*iB,  -r1y*iA*r1x-r2y*iB*r2x,          -r1y*iA-r2y*iB]
            //     [  -r1y*iA*r1x-r2y*iB*r2x, mA+r1x^2*iA+mB+r2x^2*iB,           r1x*iA+r2x*iB]
            //     [          -r1y*iA-r2y*iB,           r1x*iA+r2x*iB,                   iA+iB]

            float mA = m_invMassA, mB = m_invMassB;
            float iA = m_invIA, iB = m_invIB;

            bool fixedRotation = (iA + iB == 0.0f);

            m_mass.ex.x = mA + mB + m_rA.y * m_rA.y * iA + m_rB.y * m_rB.y * iB;
            m_mass.ey.x = -m_rA.y * m_rA.x * iA - m_rB.y * m_rB.x * iB;
            m_mass.ez.x = -m_rA.y * iA - m_rB.y * iB;
            m_mass.ex.y = m_mass.ey.x;
            m_mass.ey.y = mA + mB + m_rA.x * m_rA.x * iA + m_rB.x * m_rB.x * iB;
            m_mass.ez.y = m_rA.x * iA + m_rB.x * iB;
            m_mass.ex.z = m_mass.ez.x;
            m_mass.ey.z = m_mass.ez.y;
            m_mass.ez.z = iA + iB;

            m_motorMass = iA + iB;
            if (m_motorMass > 0.0f)
                m_motorMass = 1.0f / m_motorMass;

            if (m_enableMotor == false || fixedRotation)
                m_motorImpulse = 0.0f;

            if (m_enableLimit && fixedRotation == false) {
                float jointAngle = aB - aA - m_referenceAngle;
                if (b2Abs(m_upperAngle - m_lowerAngle) < 2.0f * b2_angularSlop) {
                    m_limitState = b2LimitState.e_equalLimits;
                } else if (jointAngle <= m_lowerAngle) {
                    if (m_limitState != b2LimitState.e_atLowerLimit)
                        m_impulse.z = 0.0f;
                    m_limitState = b2LimitState.e_atLowerLimit;
                } else if (jointAngle >= m_upperAngle) {
                    if (m_limitState != b2LimitState.e_atUpperLimit)
                        m_impulse.z = 0.0f;
                    m_limitState = b2LimitState.e_atUpperLimit;
                } else {
                    m_limitState = b2LimitState.e_inactiveLimit;
                    m_impulse.z = 0.0f;
                }
            } else {
                m_limitState = b2LimitState.e_inactiveLimit;
            }

            if (data.step.warmStarting) {
                // Scale impulses to support a variable time step.
                m_impulse *= data.step.dtRatio;
                m_motorImpulse *= data.step.dtRatio;

                b2Vec2 P = new b2Vec2(m_impulse.x, m_impulse.y);

                vA -= mA * P;
                wA -= iA * (b2Cross(m_rA, P) + m_motorImpulse + m_impulse.z);

                vB += mB * P;
                wB += iB * (b2Cross(m_rB, P) + m_motorImpulse + m_impulse.z);
            } else {
                m_impulse.SetZero();
                m_motorImpulse = 0.0f;
            }

            data.velocities[m_indexA].v = vA;
            data.velocities[m_indexA].w = wA;
            data.velocities[m_indexB].v = vB;
            data.velocities[m_indexB].w = wB;
        }

        /// <summary>
        ///     One velocity iteration: the motor (clamped to the maximum torque over the step), then either the limit with the
        ///     point constraint (3x3, falling back to 2x2 when the limit impulse would change sign) or the point constraint alone.
        /// </summary>
        /// <param name="data">The solver data.</param>
        internal override void SolveVelocityConstraints(in b2SolverData data) {
            b2Vec2 vA = data.velocities[m_indexA].v;
            float wA = data.velocities[m_indexA].w;
            b2Vec2 vB = data.velocities[m_indexB].v;
            float wB = data.velocities[m_indexB].w;

            float mA = m_invMassA, mB = m_invMassB;
            float iA = m_invIA, iB = m_invIB;

            bool fixedRotation = (iA + iB == 0.0f);

            // Solve motor constraint.
            if (m_enableMotor && m_limitState != b2LimitState.e_equalLimits && fixedRotation == false) {
                float Cdot = wB - wA - m_motorSpeed;
                float impulse = -m_motorMass * Cdot;
                float oldImpulse = m_motorImpulse;
                float maxImpulse = data.step.dt * m_maxMotorTorque;
                m_motorImpulse = b2Clamp(m_motorImpulse + impulse, -maxImpulse, maxImpulse);
                impulse = m_motorImpulse - oldImpulse;

                wA -= iA * impulse;
                wB += iB * impulse;
            }

            // Solve limit constraint.
            if (m_enableLimit && m_limitState != b2LimitState.e_inactiveLimit && fixedRotation == false) {
                b2Vec2 Cdot1 = vB + b2Cross(wB, m_rB) - vA - b2Cross(wA, m_rA);
                float Cdot2 = wB - wA;
                b2Vec3 Cdot = new b2Vec3(Cdot1.x, Cdot1.y, Cdot2);

                b2Vec3 impulse = -m_mass.Solve33(Cdot);

                if (m_limitState == b2LimitState.e_equalLimits) {
                    m_impulse += impulse;
                } else if (m_limitState == b2LimitState.e_atLowerLimit) {
                    float newImpulse = m_impulse.z + impulse.z;
                    if (newImpulse < 0.0f) {
                        b2Vec2 rhs = -Cdot1 + m_impulse.z * new b2Vec2(m_mass.ez.x, m_mass.ez.y);
                        b2Vec2 reduced = m_mass.Solve22(rhs);
                        impulse.x = reduced.x;
                        impulse.y = reduced.y;
                        impulse.z = -m_impulse.z;
                        m_impulse.x += reduced.x;
                        m_impulse.y += reduced.y;
                        m_impulse.z = 0.0f;
                    } else {
                        m_impulse += impulse;
                    }
                } else if (m_limitState == b2LimitState.e_atUpperLimit) {
                    float newImpulse = m_impulse.z + impulse.z;
                    if (newImpulse > 0.0f) {
                        b2Vec2 rhs = -Cdot1 + m_impulse.z * new b2Vec2(m_mass.ez.x, m_mass.ez.y);
                        b2Vec2 reduced = m_mass.Solve22(rhs);
                        impulse.x = reduced.x;
                        impulse.y = reduced.y;
                        impulse.z = -m_impulse.z;
                        m_impulse.x += reduced.x;
                        m_impulse.y += reduced.y;
                        m_impulse.z = 0.0f;
                    } else {
                        m_impulse += impulse;
                    }
                }

                b2Vec2 P = new b2Vec2(impulse.x, impulse.y);

                vA -= mA * P;
                wA -= iA * (b2Cross(m_rA, P) + impulse.z);

                vB += mB * P;
                wB += iB * (b2Cross(m_rB, P) + impulse.z);
            } else {
                // Solve point-to-point constraint
                b2Vec2 Cdot = vB + b2Cross(wB, m_rB) - vA - b2Cross(wA, m_rA);
                b2Vec2 impulse = m_mass.Solve22(-Cdot);

                m_impulse.x += impulse.x;
                m_impulse.y += impulse.y;

                vA -= mA * impulse;
                wA -= iA * b2Cross(m_rA, impulse);

                vB += mB * impulse;
                wB += iB * b2Cross(m_rB, impulse);
            }

            data.velocities[m_indexA].v = vA;
            data.velocities[m_indexA].w = wA;
            data.velocities[m_indexB].v = vB;
            data.velocities[m_indexB].w = wB;
        }

        /// <summary>One position iteration: the angular limit (with slop and capped corrections), then the point constraint.</summary>
        /// <param name="data">The solver data.</param>
        /// <returns>True when the position and angle errors are within the slops.</returns>
        internal override bool SolvePositionConstraints(in b2SolverData data) {
            b2Vec2 cA = data.positions[m_indexA].c;
            float aA = data.positions[m_indexA].a;
            b2Vec2 cB = data.positions[m_indexB].c;
            float aB = data.positions[m_indexB].a;

            b2Rot qA = new b2Rot(aA), qB = new b2Rot(aB);

            float angularError = 0.0f;
            float positionError = 0.0f;

            bool fixedRotation = (m_invIA + m_invIB == 0.0f);

            // Solve angular limit constraint.
            if (m_enableLimit && m_limitState != b2LimitState.e_inactiveLimit && fixedRotation == false) {
                float angle = aB - aA - m_referenceAngle;
                float limitImpulse = 0.0f;

                if (m_limitState == b2LimitState.e_equalLimits) {
                    // Prevent large angular corrections
                    float C = b2Clamp(angle - m_lowerAngle, -b2_maxAngularCorrection, b2_maxAngularCorrection);
                    limitImpulse = -m_motorMass * C;
                    angularError = b2Abs(C);
                } else if (m_limitState == b2LimitState.e_atLowerLimit) {
                    float C = angle - m_lowerAngle;
                    angularError = -C;

                    // Prevent large angular corrections and allow some slop.
                    C = b2Clamp(C + b2_angularSlop, -b2_maxAngularCorrection, 0.0f);
                    limitImpulse = -m_motorMass * C;
                } else if (m_limitState == b2LimitState.e_atUpperLimit) {
                    float C = angle - m_upperAngle;
                    angularError = C;

                    // Prevent large angular corrections and allow some slop.
                    C = b2Clamp(C - b2_angularSlop, 0.0f, b2_maxAngularCorrection);
                    limitImpulse = -m_motorMass * C;
                }

                aA -= m_invIA * limitImpulse;
                aB += m_invIB * limitImpulse;
            }

            // Solve point-to-point constraint.
            {
                qA.Set(aA);
                qB.Set(aB);
                b2Vec2 rA = b2Mul(qA, m_localAnchorA - m_localCenterA);
                b2Vec2 rB = b2Mul(qB, m_localAnchorB - m_localCenterB);

                b2Vec2 C = cB + rB - cA - rA;
                positionError = C.Length();

                float mA = m_invMassA, mB = m_invMassB;
                float iA = m_invIA, iB = m_invIB;

                b2Mat22 K;
                K.ex.x = mA + mB + iA * rA.y * rA.y + iB * rB.y * rB.y;
                K.ex.y = -iA * rA.x * rA.y - iB * rB.x * rB.y;
                K.ey.x = K.ex.y;
                K.ey.y = mA + mB + iA * rA.x * rA.x + iB * rB.x * rB.x;

                b2Vec2 impulse = -K.Solve(C);

                cA -= mA * impulse;
                aA -= iA * b2Cross(rA, impulse);

                cB += mB * impulse;
                aB += iB * b2Cross(rB, impulse);
            }

            data.positions[m_indexA].c = cA;
            data.positions[m_indexA].a = aA;
            data.positions[m_indexB].c = cB;
            data.positions[m_indexB].a = aB;

            return positionError <= b2_linearSlop && angularError <= b2_angularSlop;
        }

        /// <inheritdoc/>
        internal override b2Vec2 GetAnchorA() => m_bodyA.GetWorldPoint(m_localAnchorA);

        /// <inheritdoc/>
        internal override b2Vec2 GetAnchorB() => m_bodyB.GetWorldPoint(m_localAnchorB);

        /// <inheritdoc/>
        internal override b2Vec2 GetReactionForce(float inv_dt) {
            b2Vec2 P = new b2Vec2(m_impulse.x, m_impulse.y);
            return inv_dt * P;
        }

        /// <inheritdoc/>
        internal override float GetReactionTorque(float inv_dt) => inv_dt * m_impulse.z;

        /// <summary>The current joint angle: body B's angle minus body A's minus the reference angle.</summary>
        /// <returns>The angle in radians.</returns>
        internal float GetJointAngle() => m_bodyB.m_sweep.a - m_bodyA.m_sweep.a - m_referenceAngle;

        /// <summary>The current joint speed: body B's angular velocity minus body A's.</summary>
        /// <returns>The speed in rad/s.</returns>
        internal float GetJointSpeed() => m_bodyB.m_angularVelocity - m_bodyA.m_angularVelocity;

        /// <summary>Whether the motor is enabled.</summary>
        /// <returns>True when enabled.</returns>
        internal bool IsMotorEnabled() => m_enableMotor;

        /// <summary>Enables or disables the motor, waking both bodies.</summary>
        /// <param name="flag">True to enable.</param>
        internal void EnableMotor(bool flag) {
            m_bodyA.SetAwake(true);
            m_bodyB.SetAwake(true);
            m_enableMotor = flag;
        }

        /// <summary>The motor torque over a step.</summary>
        /// <param name="inv_dt">The inverse time step.</param>
        /// <returns>The torque in N·m.</returns>
        internal float GetMotorTorque(float inv_dt) => inv_dt * m_motorImpulse;

        /// <summary>Sets the motor speed, waking both bodies.</summary>
        /// <param name="speed">The speed in rad/s.</param>
        internal void SetMotorSpeed(float speed) {
            m_bodyA.SetAwake(true);
            m_bodyB.SetAwake(true);
            m_motorSpeed = speed;
        }

        /// <summary>The motor speed.</summary>
        /// <returns>The speed in rad/s.</returns>
        internal float GetMotorSpeed() => m_motorSpeed;

        /// <summary>Sets the motor's maximum torque, waking both bodies.</summary>
        /// <param name="torque">The torque in N·m.</param>
        internal void SetMaxMotorTorque(float torque) {
            m_bodyA.SetAwake(true);
            m_bodyB.SetAwake(true);
            m_maxMotorTorque = torque;
        }

        /// <summary>The motor's maximum torque.</summary>
        /// <returns>The torque in N·m.</returns>
        internal float GetMaxMotorTorque() => m_maxMotorTorque;

        /// <summary>Whether the limit is enabled.</summary>
        /// <returns>True when enabled.</returns>
        internal bool IsLimitEnabled() => m_enableLimit;

        /// <summary>Enables or disables the limit, waking both bodies and dropping the limit impulse when it changes.</summary>
        /// <param name="flag">True to enable.</param>
        internal void EnableLimit(bool flag) {
            if (flag != m_enableLimit) {
                m_bodyA.SetAwake(true);
                m_bodyB.SetAwake(true);
                m_enableLimit = flag;
                m_impulse.z = 0.0f;
            }
        }

        /// <summary>The lower limit.</summary>
        /// <returns>The angle in radians.</returns>
        internal float GetLowerLimit() => m_lowerAngle;

        /// <summary>The upper limit.</summary>
        /// <returns>The angle in radians.</returns>
        internal float GetUpperLimit() => m_upperAngle;

        /// <summary>Sets the limits, waking both bodies and dropping the limit impulse when they change.</summary>
        /// <param name="lower">The lower angle.</param>
        /// <param name="upper">The upper angle.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown when the lower limit exceeds the upper one.</exception>
        internal void SetLimits(float lower, float upper) {
            b2Assert(lower <= upper);

            if (lower != m_lowerAngle || upper != m_upperAngle) {
                m_bodyA.SetAwake(true);
                m_bodyB.SetAwake(true);
                m_impulse.z = 0.0f;
                m_lowerAngle = lower;
                m_upperAngle = upper;
            }
        }
    }
}
