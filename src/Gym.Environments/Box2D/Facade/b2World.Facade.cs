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
// Altered source version (Gym.NET, 2026-11-10): the b2World extensions of pybox2d 2.3.10's Box2D/Box2D_world.i,
// translated to C#: the constructor's gravity and its recorded, inert doSleep, CreateBody with fixtures and shapes created in order, the
// typed CreateDynamicBody/CreateStaticBody/CreateKinematicBody, CreateJoint's checks, and the listener and list
// properties. Python's keyword arguments are C# named parameters; Python floats are rounded to float32 as SWIG does.
using System.Collections.Generic;
using NumSharp;

namespace Gym.Box2D {
    public partial class b2World {
        /// <summary>The contact listener the world holds a reference to (pybox2d's <c>contactListener</c> property).</summary>
        private b2ContactListener? _contactListener;

        /// <summary>The contact filter the world holds a reference to (pybox2d's <c>contactFilter</c> property).</summary>
        private b2ContactFilter? _contactFilter;

        /// <summary>
        ///     A world, as pybox2d's <c>b2World(gravity=(0, -10), doSleep=True)</c>: Box2D's world with the given gravity;
        ///     <paramref name="doSleep"/> is only recorded, in <see cref="DoSleep"/>.
        /// </summary>
        /// <remarks>
        ///     Footgun, kept from pybox2d 2.3.10 for parity: <paramref name="doSleep"/> never reaches the engine. pybox2d's
        ///     constructor runs <c>self.allowSleeping = doSleep</c>, and its world has no <c>allowSleeping</c> property (only
        ///     <c>GetAllowSleeping</c>/<c>SetAllowSleeping</c>), so the value lands in a plain attribute and bodies fall
        ///     asleep either way (measured against live pybox2d). Call <see cref="SetAllowSleeping"/> to turn sleeping off.
        /// </remarks>
        /// <param name="gravity">The gravity; <see langword="null"/> for pybox2d's default <c>(0, -10)</c>.</param>
        /// <param name="doSleep">Recorded in <see cref="DoSleep"/>; it does not stop bodies sleeping.</param>
        public b2World(b2Vec2? gravity = null, bool doSleep = true) : this(gravity ?? new b2Vec2(0.0f, -10.0f)) {
            DoSleep = doSleep;
        }

        /// <summary>
        ///     The constructor's <c>doSleep</c>, as pybox2d keeps it in its plain <c>allowSleeping</c> attribute: a record no
        ///     engine code reads (see <see cref="GetAllowSleeping"/> for the engine's flag).
        /// </summary>
        public bool DoSleep { get; }

        /// <summary>The gravity (pybox2d's <c>gravity</c>); changing it does not wake sleeping bodies.</summary>
        public b2Vec2 Gravity {
            get => m_gravity;
            set => m_gravity = value;
        }

        /// <summary>
        ///     The contact listener (pybox2d's <c>contactListener</c>). Gymnasium's envs set a <c>ContactDetector</c> here; its
        ///     callbacks run inside <see cref="Step(double, int, int)"/>.
        /// </summary>
        public b2ContactListener? ContactListener {
            get => _contactListener;
            set {
                _contactListener = value;
                SetContactListener(value);
            }
        }

        /// <summary>The contact filter (pybox2d's <c>contactFilter</c>); <see langword="null"/> reads back until one is set, while Box2D's default filter keeps filtering.</summary>
        public b2ContactFilter? ContactFilter {
            get => _contactFilter;
            set {
                _contactFilter = value;
                SetContactFilter(value);
            }
        }

        /// <summary>Whether the solvers warm-start from the last step's impulses (pybox2d's <c>warmStarting</c>).</summary>
        public bool WarmStarting {
            get => m_warmStarting;
            set => m_warmStarting = value;
        }

        /// <summary>Whether continuous collision runs (pybox2d's <c>continuousPhysics</c>).</summary>
        public bool ContinuousPhysics {
            get => m_continuousPhysics;
            set => m_continuousPhysics = value;
        }

        /// <summary>Whether each step stops after one time-of-impact sub-step (pybox2d's <c>subStepping</c>; for debugging).</summary>
        public bool SubStepping {
            get => m_subStepping;
            set => m_subStepping = value;
        }

        /// <summary>Whether forces are cleared after each step (pybox2d's <c>autoClearForces</c>, on by default).</summary>
        public bool AutoClearForces {
            get => (m_flags & e_clearForces) == e_clearForces;
            set {
                if (value)
                    m_flags |= e_clearForces;
                else
                    m_flags &= ~e_clearForces;
            }
        }

        /// <summary>Whether the world is stepping (pybox2d's <c>locked</c>): creating or destroying anything then fails.</summary>
        public bool Locked => IsLocked();

        /// <summary>The number of bodies.</summary>
        public int BodyCount => m_bodyCount;

        /// <summary>The number of joints.</summary>
        public int JointCount => m_jointCount;

        /// <summary>The number of contacts (touching or not).</summary>
        public int ContactCount => m_contactManager.m_contactCount;

        /// <summary>The number of broad-phase proxies.</summary>
        public int ProxyCount => GetProxyCount();

        /// <summary>Every body, newest first (pybox2d's <c>bodies</c>); a new list on each read.</summary>
        public IReadOnlyList<b2Body> Bodies {
            get {
                var list = new List<b2Body>();
                for (b2Body? b = m_bodyList; b != null; b = b.m_next)
                    list.Add(b);
                return list;
            }
        }

        /// <summary>Every joint, newest first (pybox2d's <c>joints</c>); a new list on each read.</summary>
        public IReadOnlyList<b2Joint> Joints {
            get {
                var list = new List<b2Joint>();
                for (b2Joint? j = m_jointList; j != null; j = j.m_next)
                    list.Add(j);
                return list;
            }
        }

        /// <summary>Every contact, newest first (pybox2d's <c>contacts</c>); a new list on each read.</summary>
        public IReadOnlyList<b2Contact> Contacts {
            get {
                var list = new List<b2Contact>();
                for (b2Contact? c = m_contactManager.m_contactList; c != null; c = c.m_next)
                    list.Add(c);
                return list;
            }
        }

        /// <summary>
        ///     Creates a body from a definition, then its fixtures in order (pybox2d's <c>CreateBody(defn)</c>): each of
        ///     <see cref="b2BodyDef.Fixtures"/>, then each of <see cref="b2BodyDef.Shapes"/> with <see cref="b2BodyDef.ShapeFixture"/>.
        /// </summary>
        /// <param name="def">The definition.</param>
        /// <returns>The body.</returns>
        /// <exception cref="AssertionError">Thrown while the world is stepping, or for an invalid definition or shape.</exception>
        public b2Body CreateBody(b2BodyDef def) {
            b2Body body = CreateBodyCore(def);
            if (def.Fixtures.Count > 0) {
                foreach (b2FixtureDef fixture in def.Fixtures)
                    body.CreateFixture(fixture);
            }

            if (def.Shapes.Count > 0)
                body.CreateFixturesFromShapes(def.Shapes, def.ShapeFixture);
            return body;
        }

        /// <summary>
        ///     Creates a dynamic body (pybox2d's <c>CreateDynamicBody(**kwargs)</c>): a <see cref="b2BodyDef"/> of these
        ///     arguments, then its fixtures in order.
        /// </summary>
        /// <param name="position">The body origin's world position.</param>
        /// <param name="angle">The angle in radians.</param>
        /// <param name="fixtures">The fixtures to create, in order.</param>
        /// <param name="shapes">Shapes to create fixtures of with <paramref name="shapeFixture"/>'s material, after <paramref name="fixtures"/>.</param>
        /// <param name="shapeFixture">The material for <paramref name="shapes"/> (Box2D's defaults when null).</param>
        /// <param name="linearVelocity">The initial linear velocity.</param>
        /// <param name="angularVelocity">The initial angular velocity.</param>
        /// <param name="linearDamping">The linear damping.</param>
        /// <param name="angularDamping">The angular damping.</param>
        /// <param name="allowSleep">Whether the body may sleep.</param>
        /// <param name="awake">Whether it starts awake.</param>
        /// <param name="fixedRotation">Whether it may not rotate.</param>
        /// <param name="bullet">Whether it is a bullet.</param>
        /// <param name="active">Whether it starts active.</param>
        /// <param name="gravityScale">The gravity scale.</param>
        /// <param name="userData">The user data.</param>
        /// <returns>The body.</returns>
        /// <exception cref="AssertionError">Thrown while the world is stepping, or for an invalid definition or shape.</exception>
        /// <exception cref="System.OverflowException">Thrown for a scalar beyond the float32 range.</exception>
        /// <exception cref="TypeError">Thrown for a vector coordinate beyond the float32 range.</exception>
        public b2Body CreateDynamicBody(b2Vec2? position = null, double angle = 0.0, IEnumerable<b2FixtureDef>? fixtures = null,
                                        IEnumerable<b2Shape>? shapes = null, b2FixtureDef? shapeFixture = null, b2Vec2? linearVelocity = null,
                                        double angularVelocity = 0.0, double linearDamping = 0.0, double angularDamping = 0.0,
                                        bool allowSleep = true, bool awake = true, bool fixedRotation = false, bool bullet = false,
                                        bool active = true, double gravityScale = 1.0, object? userData = null) =>
            CreateBody(new b2BodyDef(b2BodyType.b2_dynamicBody, position, angle, fixtures, shapes, shapeFixture, linearVelocity, angularVelocity,
                                     linearDamping, angularDamping, allowSleep, awake, fixedRotation, bullet, active, gravityScale, userData));

        /// <summary>Creates a dynamic body with one fixture (pybox2d accepts a single fixture definition for <c>fixtures</c>).</summary>
        /// <param name="fixtures">The fixture.</param>
        /// <param name="position">The body origin's world position.</param>
        /// <param name="angle">The angle in radians.</param>
        /// <param name="userData">The user data.</param>
        /// <returns>The body.</returns>
        /// <exception cref="AssertionError">Thrown while the world is stepping, or for an invalid shape.</exception>
        /// <exception cref="System.OverflowException">Thrown for a scalar beyond the float32 range.</exception>
        public b2Body CreateDynamicBody(b2FixtureDef fixtures, b2Vec2? position = null, double angle = 0.0, object? userData = null) =>
            CreateDynamicBody(position, angle, new[] { fixtures }, userData: userData);

        /// <summary>Creates a static body (pybox2d's <c>CreateStaticBody(**kwargs)</c>).</summary>
        /// <param name="position">The body origin's world position.</param>
        /// <param name="angle">The angle in radians.</param>
        /// <param name="fixtures">The fixtures to create, in order.</param>
        /// <param name="shapes">Shapes to create fixtures of with <paramref name="shapeFixture"/>'s material, after <paramref name="fixtures"/>.</param>
        /// <param name="shapeFixture">The material for <paramref name="shapes"/> (Box2D's defaults when null).</param>
        /// <param name="userData">The user data.</param>
        /// <returns>The body.</returns>
        /// <exception cref="AssertionError">Thrown while the world is stepping, or for an invalid shape.</exception>
        /// <exception cref="System.OverflowException">Thrown for a scalar beyond the float32 range.</exception>
        public b2Body CreateStaticBody(b2Vec2? position = null, double angle = 0.0, IEnumerable<b2FixtureDef>? fixtures = null,
                                       IEnumerable<b2Shape>? shapes = null, b2FixtureDef? shapeFixture = null, object? userData = null) =>
            CreateBody(new b2BodyDef(b2BodyType.b2_staticBody, position, angle, fixtures, shapes, shapeFixture, userData: userData));

        /// <summary>Creates a static body with one fixture.</summary>
        /// <param name="fixtures">The fixture.</param>
        /// <param name="position">The body origin's world position.</param>
        /// <param name="angle">The angle in radians.</param>
        /// <param name="userData">The user data.</param>
        /// <returns>The body.</returns>
        /// <exception cref="AssertionError">Thrown while the world is stepping, or for an invalid shape.</exception>
        /// <exception cref="System.OverflowException">Thrown for a scalar beyond the float32 range.</exception>
        public b2Body CreateStaticBody(b2FixtureDef fixtures, b2Vec2? position = null, double angle = 0.0, object? userData = null) =>
            CreateStaticBody(position, angle, new[] { fixtures }, userData: userData);

        /// <summary>Creates a static body with one shape and Box2D's default material (pybox2d's <c>CreateStaticBody(shapes=shape)</c>).</summary>
        /// <param name="shapes">The shape.</param>
        /// <param name="position">The body origin's world position.</param>
        /// <param name="angle">The angle in radians.</param>
        /// <returns>The body.</returns>
        /// <exception cref="AssertionError">Thrown while the world is stepping, or for an invalid shape.</exception>
        /// <exception cref="System.OverflowException">Thrown for a scalar beyond the float32 range.</exception>
        public b2Body CreateStaticBody(b2Shape shapes, b2Vec2? position = null, double angle = 0.0) =>
            CreateStaticBody(position, angle, shapes: new[] { shapes });

        /// <summary>Creates a kinematic body (pybox2d's <c>CreateKinematicBody(**kwargs)</c>).</summary>
        /// <param name="position">The body origin's world position.</param>
        /// <param name="angle">The angle in radians.</param>
        /// <param name="fixtures">The fixtures to create, in order.</param>
        /// <param name="linearVelocity">The linear velocity.</param>
        /// <param name="angularVelocity">The angular velocity.</param>
        /// <param name="userData">The user data.</param>
        /// <returns>The body.</returns>
        /// <exception cref="AssertionError">Thrown while the world is stepping, or for an invalid shape.</exception>
        /// <exception cref="System.OverflowException">Thrown for a scalar beyond the float32 range.</exception>
        public b2Body CreateKinematicBody(b2Vec2? position = null, double angle = 0.0, IEnumerable<b2FixtureDef>? fixtures = null,
                                          b2Vec2? linearVelocity = null, double angularVelocity = 0.0, object? userData = null) =>
            CreateBody(new b2BodyDef(b2BodyType.b2_kinematicBody, position, angle, fixtures, linearVelocity: linearVelocity,
                                     angularVelocity: angularVelocity, userData: userData));

        /// <summary>
        ///     Creates a joint (pybox2d's <c>CreateJoint(defn)</c>), refusing a definition without both bodies first, as
        ///     pybox2d does before Box2D sees it.
        /// </summary>
        /// <param name="def">The definition.</param>
        /// <returns>The joint.</returns>
        /// <exception cref="ValueError">Thrown when <c>bodyA</c> or <c>bodyB</c> is unset: <c>bodyA and bodyB must be set</c>.</exception>
        /// <exception cref="AssertionError">Thrown while the world is stepping, for one body twice, or for a joint type this port lacks.</exception>
        public b2Joint CreateJoint(b2JointDef def) {
            if (def.bodyA is null || def.bodyB is null)
                throw new ValueError("bodyA and bodyB must be set");
            return CreateJointCore(def);
        }

        /// <summary>
        ///     Steps the world (pybox2d's <c>Step(timeStep, velocityIterations, positionIterations)</c>): the time step is
        ///     rounded to float32 first, as SWIG converts it, so <c>1.0 / 50</c> becomes <c>0.02f</c>.
        /// </summary>
        /// <param name="timeStep">The time step in seconds.</param>
        /// <param name="velocityIterations">The velocity iterations (Gymnasium's envs use <c>6 * 30</c>).</param>
        /// <param name="positionIterations">The position iterations (Gymnasium's envs use <c>2 * 30</c>).</param>
        /// <exception cref="System.OverflowException">Thrown for a time step beyond the float32 range.</exception>
        /// <exception cref="AssertionError">Thrown on an internal inconsistency, leaving the world locked as pybox2d does.</exception>
        public void Step(double timeStep, int velocityIterations, int positionIterations) =>
            Step(Box2DConvert.ToFloat32(timeStep, "b2World_Step", 2), velocityIterations, positionIterations);

    }
}
