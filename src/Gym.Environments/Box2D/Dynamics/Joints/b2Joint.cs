/*
* Copyright (c) 2006-2007 Erin Catto http://www.box2d.org
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
// Altered source version (Gym.NET, 2026-11-10): Box2D/Dynamics/Joints/b2Joint.h and b2Joint.cpp of pybox2d 2.3.10's
// bundled Box2D 2.3.0, translated to C#. The factory builds the revolute joint, the one joint Gymnasium's envs use; the
// other joint types of Box2D 2.3 are not part of this port and are refused like an unknown type. Joint edges are objects
// linked by reference.
using static Gym.Box2D.b2Settings;

namespace Gym.Box2D {
    /// <summary>The joint types of Box2D 2.3 (C++ <c>b2JointType</c>); this port builds <see cref="e_revoluteJoint"/>.</summary>
    public enum b2JointType {
        /// <summary>No type.</summary>
        e_unknownJoint,

        /// <summary>A revolute joint (<see cref="b2RevoluteJoint"/>).</summary>
        e_revoluteJoint,

        /// <summary>A prismatic joint (not ported).</summary>
        e_prismaticJoint,

        /// <summary>A distance joint (not ported).</summary>
        e_distanceJoint,

        /// <summary>A pulley joint (not ported).</summary>
        e_pulleyJoint,

        /// <summary>A mouse joint (not ported).</summary>
        e_mouseJoint,

        /// <summary>A gear joint (not ported).</summary>
        e_gearJoint,

        /// <summary>A wheel joint (not ported).</summary>
        e_wheelJoint,

        /// <summary>A weld joint (not ported).</summary>
        e_weldJoint,

        /// <summary>A friction joint (not ported).</summary>
        e_frictionJoint,

        /// <summary>A rope joint (not ported).</summary>
        e_ropeJoint,

        /// <summary>A motor joint (not ported).</summary>
        e_motorJoint,
    }

    /// <summary>A joint limit's state.</summary>
    internal enum b2LimitState {
        /// <summary>The limit is not active.</summary>
        e_inactiveLimit,

        /// <summary>At the lower limit.</summary>
        e_atLowerLimit,

        /// <summary>At the upper limit.</summary>
        e_atUpperLimit,

        /// <summary>The limits are equal: the joint is locked.</summary>
        e_equalLimits,
    }

    /// <summary>A joint edge: connects a body to a joint in the body's joint list. Each joint has two, one per body.</summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Dynamics/Joints/b2Joint.h#L68">Box2D/Dynamics/Joints/b2Joint.h</see>.</remarks>
    public sealed partial class b2JointEdge {
        /// <summary>The other body of the joint.</summary>
        internal b2Body? other;

        /// <summary>The joint.</summary>
        internal b2Joint? joint;

        /// <summary>The previous edge of the body's list.</summary>
        internal b2JointEdge? prev;

        /// <summary>The next edge of the body's list.</summary>
        internal b2JointEdge? next;
    }

    /// <summary>The definition shared by every joint: the type, the two bodies, and whether they collide with each other.</summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Dynamics/Joints/b2Joint.h#L77">Box2D/Dynamics/Joints/b2Joint.h</see>.</remarks>
    public abstract partial class b2JointDef {
        /// <summary>The joint type, set by the concrete definition.</summary>
        internal b2JointType type = b2JointType.e_unknownJoint;

        /// <summary>The joint's user data.</summary>
        internal object? userData;

        /// <summary>The first body.</summary>
        internal b2Body? bodyA;

        /// <summary>The second body.</summary>
        internal b2Body? bodyB;

        /// <summary>Whether the two bodies may collide (default false).</summary>
        internal bool collideConnected = false;
    }

    /// <summary>
    ///     A constraint between two bodies, solved with the contacts in each island. Created by <see cref="b2World"/>;
    ///     destroying either body destroys it.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Dynamics/Joints/b2Joint.cpp">Box2D/Dynamics/Joints/b2Joint.cpp</see>.</remarks>
    public abstract partial class b2Joint {
        /// <summary>The type.</summary>
        internal b2JointType m_type;

        /// <summary>The previous joint of the world's list.</summary>
        internal b2Joint? m_prev;

        /// <summary>The next joint of the world's list.</summary>
        internal b2Joint? m_next;

        /// <summary>The edge in body A's joint list.</summary>
        internal readonly b2JointEdge m_edgeA = new b2JointEdge();

        /// <summary>The edge in body B's joint list.</summary>
        internal readonly b2JointEdge m_edgeB = new b2JointEdge();

        /// <summary>Body A.</summary>
        internal b2Body m_bodyA;

        /// <summary>Body B.</summary>
        internal b2Body m_bodyB;

        /// <summary>Kept from C++ (unused by 2.3's solver).</summary>
        internal int m_index;

        /// <summary>Set while the joint is in the island being built.</summary>
        internal bool m_islandFlag;

        /// <summary>Whether the bodies may collide.</summary>
        internal bool m_collideConnected;

        /// <summary>The user data.</summary>
        internal object? m_userData;

        /// <summary>A joint from a definition.</summary>
        /// <param name="def">The definition.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown when both bodies are the same.</exception>
        protected b2Joint(b2JointDef def) {
            b2Assert(def.bodyA != def.bodyB, "def->bodyA != def->bodyB");

            m_type = def.type;
            m_prev = null;
            m_next = null;
            m_bodyA = def.bodyA!;
            m_bodyB = def.bodyB!;
            m_index = 0;
            m_collideConnected = def.collideConnected;
            m_islandFlag = false;
            m_userData = def.userData;

            m_edgeA.joint = null;
            m_edgeA.other = null;
            m_edgeA.prev = null;
            m_edgeA.next = null;

            m_edgeB.joint = null;
            m_edgeB.other = null;
            m_edgeB.prev = null;
            m_edgeB.next = null;
        }

        /// <summary>Builds the joint a definition describes.</summary>
        /// <param name="def">The definition.</param>
        /// <returns>The joint.</returns>
        /// <exception cref="NumSharp.AssertionError">Thrown for a joint type this port does not include.</exception>
        internal static b2Joint Create(b2JointDef def) {
            switch (def.type) {
                case b2JointType.e_revoluteJoint:
                    return new b2RevoluteJoint((b2RevoluteJointDef)def);

                default:
                    b2Assert(false);
                    return null!;
            }
        }

        /// <summary>The joint type.</summary>
        /// <returns>The type.</returns>
        internal b2JointType GetJointType() => m_type;

        /// <summary>Body A.</summary>
        /// <returns>The body.</returns>
        internal b2Body GetBodyA() => m_bodyA;

        /// <summary>Body B.</summary>
        /// <returns>The body.</returns>
        internal b2Body GetBodyB() => m_bodyB;

        /// <summary>The next joint of the world's list.</summary>
        /// <returns>The next joint, or null.</returns>
        internal b2Joint? GetNext() => m_next;

        /// <summary>Whether both bodies are active.</summary>
        /// <returns>True when the joint is simulated.</returns>
        internal bool IsActive() => m_bodyA.IsActive() && m_bodyB.IsActive();

        /// <summary>The anchor on body A in world coordinates.</summary>
        /// <returns>The anchor.</returns>
        internal abstract b2Vec2 GetAnchorA();

        /// <summary>The anchor on body B in world coordinates.</summary>
        /// <returns>The anchor.</returns>
        internal abstract b2Vec2 GetAnchorB();

        /// <summary>The reaction force on body B at the anchor.</summary>
        /// <param name="inv_dt">The inverse time step.</param>
        /// <returns>The force in newtons.</returns>
        internal abstract b2Vec2 GetReactionForce(float inv_dt);

        /// <summary>The reaction torque on body B.</summary>
        /// <param name="inv_dt">The inverse time step.</param>
        /// <returns>The torque in N·m.</returns>
        internal abstract float GetReactionTorque(float inv_dt);

        /// <summary>Shifts the joint's world-space data by a new origin (nothing for most joints).</summary>
        /// <param name="newOrigin">The new origin in the old frame.</param>
        internal virtual void ShiftOrigin(in b2Vec2 newOrigin) { }

        /// <summary>Initializes the velocity constraints for a step and applies the warm start.</summary>
        /// <param name="data">The solver data.</param>
        internal abstract void InitVelocityConstraints(in b2SolverData data);

        /// <summary>One velocity iteration.</summary>
        /// <param name="data">The solver data.</param>
        internal abstract void SolveVelocityConstraints(in b2SolverData data);

        /// <summary>One position iteration.</summary>
        /// <param name="data">The solver data.</param>
        /// <returns>True when the position error is within tolerance.</returns>
        internal abstract bool SolvePositionConstraints(in b2SolverData data);
    }
}
