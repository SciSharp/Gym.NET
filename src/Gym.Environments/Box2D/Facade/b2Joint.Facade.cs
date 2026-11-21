/*
* pybox2d -- http://pybox2d.googlecode.com
*
* Copyright (c) 2010 Ken Lauer / sirkne at gmail dot com
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
// Altered source version (Gym.NET, 2026-11-10): the joint, revolute joint and revolute joint definition extensions of
// pybox2d 2.3.10's Box2D/Box2D_joints.i and Box2D_kwargs.i, plus the contact and edge properties of Box2D_contact.i,
// translated to C#: the definition's keyword initializer (bodies first, then the rest, then the reference angle from the
// bodies' current angles unless given) and the joints' read-write properties (setting a motor speed wakes the bodies).
using System;
using NumSharp;

namespace Gym.Box2D {
    public abstract partial class b2JointDef {
        /// <summary>The first body (pybox2d's <c>bodyA</c>).</summary>
        public b2Body? BodyA {
            get => bodyA;
            set => bodyA = value;
        }

        /// <summary>The second body (pybox2d's <c>bodyB</c>).</summary>
        public b2Body? BodyB {
            get => bodyB;
            set => bodyB = value;
        }

        /// <summary>Whether the two bodies may collide (pybox2d's <c>collideConnected</c>).</summary>
        public bool CollideConnected {
            get => collideConnected;
            set => collideConnected = value;
        }

        /// <summary>The joint's user data.</summary>
        public object? UserData {
            get => userData;
            set => userData = value;
        }
    }

    public sealed partial class b2RevoluteJointDef {
        /// <summary>
        ///     A revolute joint definition from pybox2d's keyword arguments (<c>revoluteJointDef(bodyA=..., bodyB=..., ...)</c>):
        ///     the bodies are set first, then the rest, and when both bodies are given without a reference angle, the
        ///     reference angle becomes <c>bodyB.angle - bodyA.angle</c> at this moment, as pybox2d computes it (a Python
        ///     float difference stored as float32). A refused argument raises pybox2d's
        ///     <c>Failed on kwargs for b2RevoluteJointDef.anchor: …</c> (its keyword named).
        /// </summary>
        /// <param name="bodyA">Body A.</param>
        /// <param name="bodyB">Body B.</param>
        /// <param name="localAnchorA">The anchor relative to body A's origin (origin when null).</param>
        /// <param name="localAnchorB">The anchor relative to body B's origin (origin when null).</param>
        /// <param name="referenceAngle">The reference angle, or null to take it from the bodies.</param>
        /// <param name="enableLimit">Whether the angle limit is enforced.</param>
        /// <param name="lowerAngle">The lower angle limit.</param>
        /// <param name="upperAngle">The upper angle limit.</param>
        /// <param name="enableMotor">Whether the motor is enabled.</param>
        /// <param name="motorSpeed">The motor's target speed.</param>
        /// <param name="maxMotorTorque">The motor's maximum torque.</param>
        /// <param name="collideConnected">Whether the bodies may collide.</param>
        /// <param name="anchor">A world anchor point; sets both local anchors from the bodies' current frames (pybox2d's <c>anchor</c>).</param>
        /// <exception cref="ValueError">Thrown when <paramref name="anchor"/> is given without both bodies.</exception>
        /// <exception cref="OverflowError">Thrown for a scalar beyond the float32 range.</exception>
        public b2RevoluteJointDef(b2Body? bodyA = null, b2Body? bodyB = null, b2Vec2? localAnchorA = null, b2Vec2? localAnchorB = null,
                                  double? referenceAngle = null, bool enableLimit = false, double lowerAngle = 0.0, double upperAngle = 0.0,
                                  bool enableMotor = false, double motorSpeed = 0.0, double maxMotorTorque = 0.0, bool collideConnected = false,
                                  b2Vec2? anchor = null) : this() {
            // pybox2d's _init_jointdef_kwargs sets the bodies before anything else.
            this.bodyA = bodyA;
            this.bodyB = bodyB;
            if (localAnchorA.HasValue)
                this.localAnchorA = localAnchorA.Value;
            if (localAnchorB.HasValue)
                this.localAnchorB = localAnchorB.Value;
            // The float32 conversions and the anchor can fail; pybox2d's _init_kwargs prefixes their errors with the class
            // and keyword.
            if (referenceAngle.HasValue)
                Box2DConvert.Kwarg("b2RevoluteJointDef", "referenceAngle", () => this.referenceAngle = Box2DConvert.ToFloat32(referenceAngle.Value, "b2RevoluteJointDef_referenceAngle_set", 2));
            this.enableLimit = enableLimit;
            Box2DConvert.Kwarg("b2RevoluteJointDef", "lowerAngle", () => this.lowerAngle = Box2DConvert.ToFloat32(lowerAngle, "b2RevoluteJointDef_lowerAngle_set", 2));
            Box2DConvert.Kwarg("b2RevoluteJointDef", "upperAngle", () => this.upperAngle = Box2DConvert.ToFloat32(upperAngle, "b2RevoluteJointDef_upperAngle_set", 2));
            this.enableMotor = enableMotor;
            Box2DConvert.Kwarg("b2RevoluteJointDef", "motorSpeed", () => this.motorSpeed = Box2DConvert.ToFloat32(motorSpeed, "b2RevoluteJointDef_motorSpeed_set", 2));
            Box2DConvert.Kwarg("b2RevoluteJointDef", "maxMotorTorque", () => this.maxMotorTorque = Box2DConvert.ToFloat32(maxMotorTorque, "b2RevoluteJointDef_maxMotorTorque_set", 2));
            this.collideConnected = collideConnected;
            if (anchor.HasValue)
                Box2DConvert.Kwarg("b2RevoluteJointDef", "anchor", () => Anchor = anchor.Value);

            // pybox2d's __init__: the reference angle from the bodies, unless given.
            if (this.bodyA != null && this.bodyB != null && !referenceAngle.HasValue)
                this.referenceAngle = (float)((double)this.bodyB.m_sweep.a - this.bodyA.m_sweep.a);
        }

        /// <summary>The anchor relative to body A's origin.</summary>
        public b2Vec2 LocalAnchorA {
            get => localAnchorA;
            set => localAnchorA = value;
        }

        /// <summary>The anchor relative to body B's origin.</summary>
        public b2Vec2 LocalAnchorB {
            get => localAnchorB;
            set => localAnchorB = value;
        }

        /// <summary>The reference angle (stored as float32).</summary>
        public double ReferenceAngle {
            get => referenceAngle;
            set => referenceAngle = Box2DConvert.ToFloat32(value, "b2RevoluteJointDef_referenceAngle_set", 2);
        }

        /// <summary>Whether the angle limit is enforced.</summary>
        public bool EnableLimit {
            get => enableLimit;
            set => enableLimit = value;
        }

        /// <summary>The lower angle limit (stored as float32).</summary>
        public double LowerAngle {
            get => lowerAngle;
            set => lowerAngle = Box2DConvert.ToFloat32(value, "b2RevoluteJointDef_lowerAngle_set", 2);
        }

        /// <summary>The upper angle limit (stored as float32).</summary>
        public double UpperAngle {
            get => upperAngle;
            set => upperAngle = Box2DConvert.ToFloat32(value, "b2RevoluteJointDef_upperAngle_set", 2);
        }

        /// <summary>Whether the motor is enabled.</summary>
        public bool EnableMotor {
            get => enableMotor;
            set => enableMotor = value;
        }

        /// <summary>The motor's target speed (stored as float32).</summary>
        public double MotorSpeed {
            get => motorSpeed;
            set => motorSpeed = Box2DConvert.ToFloat32(value, "b2RevoluteJointDef_motorSpeed_set", 2);
        }

        /// <summary>The motor's maximum torque (stored as float32).</summary>
        public double MaxMotorTorque {
            get => maxMotorTorque;
            set => maxMotorTorque = Box2DConvert.ToFloat32(value, "b2RevoluteJointDef_maxMotorTorque_set", 2);
        }

        /// <summary>
        ///     The world anchor (pybox2d's <c>anchor</c>): reading maps local anchor A (or B) to the world; writing sets both
        ///     local anchors from the bodies' current frames.
        /// </summary>
        /// <exception cref="ValueError">Thrown when the needed bodies are unset, with pybox2d's messages.</exception>
        public b2Vec2 Anchor {
            get {
                if (bodyA != null)
                    return bodyA.GetWorldPoint(localAnchorA);
                if (bodyB != null)
                    return bodyB.GetWorldPoint(localAnchorB);
                throw new ValueError("Neither body was set; unable to get world point.");
            }
            set {
                if (bodyA == null)
                    throw new ValueError("bodyA not set.");
                if (bodyB == null)
                    throw new ValueError("bodyB not set.");
                localAnchorA = bodyA.GetLocalPoint(value);
                localAnchorB = bodyB.GetLocalPoint(value);
            }
        }
    }

    public abstract partial class b2Joint {
        /// <summary>Body A (pybox2d's <c>bodyA</c>).</summary>
        public b2Body BodyA => m_bodyA;

        /// <summary>Body B (pybox2d's <c>bodyB</c>).</summary>
        public b2Body BodyB => m_bodyB;

        /// <summary>The joint type (pybox2d's <c>type</c>).</summary>
        public b2JointType Type => m_type;

        /// <summary>Whether both bodies are active (pybox2d's <c>active</c>).</summary>
        public bool Active => IsActive();

        /// <summary>The anchor on body A in world coordinates (pybox2d's <c>anchorA</c>).</summary>
        public b2Vec2 AnchorA => GetAnchorA();

        /// <summary>The anchor on body B in world coordinates (pybox2d's <c>anchorB</c>).</summary>
        public b2Vec2 AnchorB => GetAnchorB();

        /// <summary>Whether the bodies may collide (pybox2d's <c>collideConnected</c>).</summary>
        public bool CollideConnected => m_collideConnected;

        /// <summary>The user data.</summary>
        public object? UserData {
            get => m_userData;
            set => m_userData = value;
        }
    }

    public sealed partial class b2RevoluteJoint {
        /// <summary>The motor's target speed (pybox2d's <c>motorSpeed</c>); setting it wakes both bodies, as BipedalWalker's every step does.</summary>
        public float MotorSpeed {
            get => m_motorSpeed;
            set => SetMotorSpeed(value);
        }

        /// <summary>
        ///     The motor's maximum torque (pybox2d's write-only <c>maxMotorTorque</c>; readable here); setting it wakes both
        ///     bodies.
        /// </summary>
        public float MaxMotorTorque {
            get => m_maxMotorTorque;
            set => SetMaxMotorTorque(value);
        }

        /// <summary>Whether the motor is enabled (pybox2d's <c>motorEnabled</c>); changing it wakes both bodies.</summary>
        public bool MotorEnabled {
            get => m_enableMotor;
            set => EnableMotor(value);
        }

        /// <summary>Whether the limit is enabled (pybox2d's <c>limitEnabled</c>).</summary>
        public bool LimitEnabled {
            get => m_enableLimit;
            set => EnableLimit(value);
        }

        /// <summary>The lower limit (pybox2d's <c>lowerLimit</c>).</summary>
        /// <exception cref="AssertionError">Thrown by the setter above the upper limit.</exception>
        public float LowerLimit {
            get => m_lowerAngle;
            set => SetLimits(value, m_upperAngle);
        }

        /// <summary>The upper limit (pybox2d's <c>upperLimit</c>).</summary>
        /// <exception cref="AssertionError">Thrown by the setter below the lower limit.</exception>
        public float UpperLimit {
            get => m_upperAngle;
            set => SetLimits(m_lowerAngle, value);
        }

        /// <summary>The joint angle: body B's angle minus body A's minus the reference angle (pybox2d's <c>angle</c>).</summary>
        public float Angle => GetJointAngle();

        /// <summary>The joint speed: body B's angular velocity minus body A's (pybox2d's <c>speed</c>).</summary>
        public float Speed => GetJointSpeed();

        /// <summary>The motor torque over a step of inverse length <paramref name="inv_dt"/>.</summary>
        /// <param name="inv_dt">The inverse time step.</param>
        /// <returns>The torque in N·m.</returns>
        public float GetMotorTorque(double inv_dt) => GetMotorTorque(Box2DConvert.ToFloat32(inv_dt, "b2RevoluteJoint_GetMotorTorque", 2));
    }

    public abstract partial class b2Contact {
        /// <summary>Fixture A (pybox2d's <c>fixtureA</c>).</summary>
        public b2Fixture FixtureA => m_fixtureA!;

        /// <summary>Fixture B (pybox2d's <c>fixtureB</c>).</summary>
        public b2Fixture FixtureB => m_fixtureB!;

        /// <summary>Fixture A's child index (pybox2d's <c>childIndexA</c>).</summary>
        public int ChildIndexA => m_indexA;

        /// <summary>Fixture B's child index (pybox2d's <c>childIndexB</c>).</summary>
        public int ChildIndexB => m_indexB;

        /// <summary>Whether the shapes are touching (pybox2d's <c>touching</c>).</summary>
        public bool Touching => IsTouching();

        /// <summary>Whether the contact is enabled for this step (pybox2d's <c>enabled</c>); updates re-enable it.</summary>
        public bool Enabled {
            get => IsEnabled();
            set => SetEnabled(value);
        }

        /// <summary>The mixed friction (pybox2d's <c>friction</c>).</summary>
        public float Friction {
            get => m_friction;
            set => m_friction = value;
        }

        /// <summary>The mixed restitution (pybox2d's <c>restitution</c>).</summary>
        public float Restitution {
            get => m_restitution;
            set => m_restitution = value;
        }

        /// <summary>The conveyor-belt tangent speed (pybox2d's <c>tangentSpeed</c>).</summary>
        public float TangentSpeed {
            get => m_tangentSpeed;
            set => m_tangentSpeed = value;
        }
    }

    public sealed partial class b2ContactEdge {
        /// <summary>The other body of the contact (pybox2d's <c>other</c>).</summary>
        public b2Body Other => other!;

        /// <summary>The contact (pybox2d's <c>contact</c>).</summary>
        public b2Contact Contact => contact!;
    }

    public sealed partial class b2JointEdge {
        /// <summary>The other body of the joint (pybox2d's <c>other</c>).</summary>
        public b2Body Other => other!;

        /// <summary>The joint (pybox2d's <c>joint</c>).</summary>
        public b2Joint Joint => joint!;
    }
}
