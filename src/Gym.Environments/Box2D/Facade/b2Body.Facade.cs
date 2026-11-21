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
// Altered source version (Gym.NET, 2026-11-10): the b2BodyDef and b2Body extensions of pybox2d 2.3.10's
// Box2D/Box2D_bodyfixture.i, translated to C#: the definition's fixtures/shapes/shapeFixture, the body's read-write
// properties (position and angle go through SetTransform, as in pybox2d), the fixture lists, and the shape-fixture
// shortcuts (CreatePolygonFixture, CreateEdgeFixture, CreateCircleFixture, CreateFixturesFromShapes).
using System.Collections.Generic;
using NumSharp;

namespace Gym.Box2D {
    public partial class b2BodyDef {
        /// <summary>A body definition with Box2D's defaults: a static body at the origin.</summary>
        public b2BodyDef() { }

        /// <summary>
        ///     A body definition from pybox2d's keyword arguments; scalars round to float32 as SWIG converts them, and a
        ///     refused one raises pybox2d's <c>Failed on kwargs for b2BodyDef.angle: …</c> (its keyword named).
        /// </summary>
        /// <param name="type">The body type.</param>
        /// <param name="position">The body origin's world position (origin when null).</param>
        /// <param name="angle">The angle in radians.</param>
        /// <param name="fixtures">The fixtures <see cref="b2World.CreateBody"/> creates, in order.</param>
        /// <param name="shapes">Shapes to create fixtures of with <paramref name="shapeFixture"/>'s material.</param>
        /// <param name="shapeFixture">The material for <paramref name="shapes"/>.</param>
        /// <param name="linearVelocity">The initial linear velocity (zero when null).</param>
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
        /// <exception cref="OverflowError">Thrown for a scalar beyond the float32 range.</exception>
        public b2BodyDef(b2BodyType type = b2BodyType.b2_staticBody, b2Vec2? position = null, double angle = 0.0, IEnumerable<b2FixtureDef>? fixtures = null,
                         IEnumerable<b2Shape>? shapes = null, b2FixtureDef? shapeFixture = null, b2Vec2? linearVelocity = null,
                         double angularVelocity = 0.0, double linearDamping = 0.0, double angularDamping = 0.0, bool allowSleep = true,
                         bool awake = true, bool fixedRotation = false, bool bullet = false, bool active = true, double gravityScale = 1.0,
                         object? userData = null) {
            this.type = type;
            if (position.HasValue)
                this.position = position.Value;
            // The float32 conversions can fail; pybox2d's _init_kwargs prefixes their errors with the class and keyword.
            Box2DConvert.Kwarg("b2BodyDef", "angle", () => this.angle = Box2DConvert.ToFloat32(angle, "b2BodyDef_angle_set", 2));
            if (fixtures != null)
                Fixtures.AddRange(fixtures);
            if (shapes != null)
                Shapes.AddRange(shapes);
            ShapeFixture = shapeFixture;
            if (linearVelocity.HasValue)
                this.linearVelocity = linearVelocity.Value;
            Box2DConvert.Kwarg("b2BodyDef", "angularVelocity", () => this.angularVelocity = Box2DConvert.ToFloat32(angularVelocity, "b2BodyDef_angularVelocity_set", 2));
            Box2DConvert.Kwarg("b2BodyDef", "linearDamping", () => this.linearDamping = Box2DConvert.ToFloat32(linearDamping, "b2BodyDef_linearDamping_set", 2));
            Box2DConvert.Kwarg("b2BodyDef", "angularDamping", () => this.angularDamping = Box2DConvert.ToFloat32(angularDamping, "b2BodyDef_angularDamping_set", 2));
            this.allowSleep = allowSleep;
            this.awake = awake;
            this.fixedRotation = fixedRotation;
            this.bullet = bullet;
            this.active = active;
            Box2DConvert.Kwarg("b2BodyDef", "gravityScale", () => this.gravityScale = Box2DConvert.ToFloat32(gravityScale, "b2BodyDef_gravityScale_set", 2));
            this.userData = userData;
        }

        /// <summary>The body type.</summary>
        public b2BodyType Type {
            get => type;
            set => type = value;
        }

        /// <summary>The body origin's world position.</summary>
        public b2Vec2 Position {
            get => position;
            set => position = value;
        }

        /// <summary>The angle in radians (stored as float32).</summary>
        public double Angle {
            get => angle;
            set => angle = Box2DConvert.ToFloat32(value, "b2BodyDef_angle_set", 2);
        }

        /// <summary>The initial linear velocity.</summary>
        public b2Vec2 LinearVelocity {
            get => linearVelocity;
            set => linearVelocity = value;
        }

        /// <summary>The initial angular velocity (stored as float32).</summary>
        public double AngularVelocity {
            get => angularVelocity;
            set => angularVelocity = Box2DConvert.ToFloat32(value, "b2BodyDef_angularVelocity_set", 2);
        }

        /// <summary>The user data.</summary>
        public object? UserData {
            get => userData;
            set => userData = value;
        }

        /// <summary>The fixtures <see cref="b2World.CreateBody"/> creates on the body, in order (pybox2d's <c>fixtures</c>).</summary>
        public List<b2FixtureDef> Fixtures { get; } = new List<b2FixtureDef>();

        /// <summary>Shapes <see cref="b2World.CreateBody"/> creates fixtures of, after <see cref="Fixtures"/> (pybox2d's <c>shapes</c>).</summary>
        public List<b2Shape> Shapes { get; } = new List<b2Shape>();

        /// <summary>The material for <see cref="Shapes"/> (pybox2d's <c>shapeFixture</c>; Box2D's defaults when null).</summary>
        public b2FixtureDef? ShapeFixture { get; set; }
    }

    public partial class b2Body {
        /// <summary>
        ///     The body origin's world position (pybox2d's <c>position</c>). Setting it calls <c>SetTransform(pos, angle)</c>,
        ///     which re-synchronizes the proxies and finds new contacts at once.
        /// </summary>
        /// <exception cref="AssertionError">Thrown by the setter while the world is stepping.</exception>
        public b2Vec2 Position {
            get => m_xf.p;
            set => SetTransform(value, m_sweep.a);
        }

        /// <summary>The angle in radians, not normalized (pybox2d's <c>angle</c>); setting it calls <c>SetTransform(position, angle)</c>.</summary>
        /// <exception cref="AssertionError">Thrown by the setter while the world is stepping.</exception>
        public float Angle {
            get => m_sweep.a;
            set => SetTransform(m_xf.p, value);
        }

        /// <summary>The body origin transform (pybox2d's <c>transform</c>); <c>trans * vertex</c> places a fixture vertex in the world.</summary>
        public b2Transform Transform => m_xf;

        /// <summary>The center of mass's linear velocity (pybox2d's <c>linearVelocity</c>); setting a non-zero one wakes the body.</summary>
        public b2Vec2 LinearVelocity {
            get => m_linearVelocity;
            set => SetLinearVelocity(value);
        }

        /// <summary>The angular velocity (pybox2d's <c>angularVelocity</c>); setting a non-zero one wakes the body.</summary>
        public float AngularVelocity {
            get => m_angularVelocity;
            set => SetAngularVelocity(value);
        }

        /// <summary>Whether the body is awake (pybox2d's <c>awake</c>); putting it to sleep zeroes its velocities and forces.</summary>
        public bool Awake {
            get => IsAwake();
            set => SetAwake(value);
        }

        /// <summary>Whether the body may sleep (pybox2d's <c>sleepingAllowed</c>).</summary>
        public bool SleepingAllowed {
            get => IsSleepingAllowed();
            set => SetSleepingAllowed(value);
        }

        /// <summary>Whether the body is a bullet (pybox2d's <c>bullet</c>).</summary>
        public bool Bullet {
            get => IsBullet();
            set => SetBullet(value);
        }

        /// <summary>Whether the body may not rotate (pybox2d's <c>fixedRotation</c>); changing it recomputes the mass.</summary>
        public bool FixedRotation {
            get => IsFixedRotation();
            set => SetFixedRotation(value);
        }

        /// <summary>Whether the body is active (pybox2d's <c>active</c>).</summary>
        /// <exception cref="AssertionError">Thrown by the setter while the world is stepping.</exception>
        public bool Active {
            get => IsActive();
            set => SetActive(value);
        }

        /// <summary>The body type (pybox2d's <c>type</c>); changing it destroys the body's contacts.</summary>
        /// <exception cref="AssertionError">Thrown by the setter while the world is stepping.</exception>
        public b2BodyType Type {
            get => m_type;
            set => SetType(value);
        }

        /// <summary>The linear damping (pybox2d's <c>linearDamping</c>).</summary>
        public float LinearDamping {
            get => m_linearDamping;
            set => m_linearDamping = value;
        }

        /// <summary>The angular damping (pybox2d's <c>angularDamping</c>).</summary>
        public float AngularDamping {
            get => m_angularDamping;
            set => m_angularDamping = value;
        }

        /// <summary>The gravity scale (pybox2d's <c>gravityScale</c>).</summary>
        public float GravityScale {
            get => m_gravityScale;
            set => m_gravityScale = value;
        }

        /// <summary>The center of mass in world coordinates (pybox2d's <c>worldCenter</c>).</summary>
        public b2Vec2 WorldCenter => m_sweep.c;

        /// <summary>The center of mass in local coordinates (pybox2d's <c>localCenter</c>).</summary>
        public b2Vec2 LocalCenter => m_sweep.localCenter;

        /// <summary>The mass (pybox2d's <c>mass</c>).</summary>
        public float Mass => m_mass;

        /// <summary>The rotational inertia about the body origin (pybox2d's <c>inertia</c>).</summary>
        public float Inertia => GetInertia();

        /// <summary>The user data (pybox2d's <c>userData</c>): any object, such as the env's own record of the body.</summary>
        public object? UserData {
            get => m_userData;
            set => m_userData = value;
        }

        /// <summary>The world (pybox2d's <c>world</c>).</summary>
        public b2World World => m_world;

        /// <summary>The fixtures, newest first (pybox2d's <c>fixtures</c>); a new list on each read.</summary>
        public IReadOnlyList<b2Fixture> Fixtures {
            get {
                var list = new List<b2Fixture>();
                for (b2Fixture? f = m_fixtureList; f != null; f = f.m_next)
                    list.Add(f);
                return list;
            }
        }

        /// <summary>The joint edges, newest first (pybox2d's <c>joints</c>); a new list on each read.</summary>
        public IReadOnlyList<b2JointEdge> Joints {
            get {
                var list = new List<b2JointEdge>();
                for (b2JointEdge? e = m_jointList; e != null; e = e.next)
                    list.Add(e);
                return list;
            }
        }

        /// <summary>The contact edges, newest first (pybox2d's <c>contacts</c>); a new list on each read.</summary>
        public IReadOnlyList<b2ContactEdge> Contacts {
            get {
                var list = new List<b2ContactEdge>();
                for (b2ContactEdge? e = m_contactList; e != null; e = e.next)
                    list.Add(e);
                return list;
            }
        }

        /// <summary>Adds a torque given as a Python float, rounded to float32 as SWIG converts it.</summary>
        /// <param name="torque">The torque in N·m.</param>
        /// <param name="wake">Whether to wake the body.</param>
        /// <exception cref="OverflowError">Thrown for a torque beyond the float32 range.</exception>
        public void ApplyTorque(double torque, bool wake) => ApplyTorque(Box2DConvert.ToFloat32(torque, "b2Body_ApplyTorque", 2), wake);

        /// <summary>Applies an angular impulse given as a Python float, rounded to float32 as SWIG converts it.</summary>
        /// <param name="impulse">The angular impulse.</param>
        /// <param name="wake">Whether to wake the body.</param>
        /// <exception cref="OverflowError">Thrown for an impulse beyond the float32 range.</exception>
        public void ApplyAngularImpulse(double impulse, bool wake) => ApplyAngularImpulse(Box2DConvert.ToFloat32(impulse, "b2Body_ApplyAngularImpulse", 2), wake);

        /// <summary>
        ///     Creates a fixture from a shape and material (pybox2d's <c>CreateFixture(shape=..., ...)</c>): a
        ///     <see cref="b2FixtureDef"/> of these arguments.
        /// </summary>
        /// <param name="shape">The shape; it is cloned.</param>
        /// <param name="density">The density.</param>
        /// <param name="friction">The friction.</param>
        /// <param name="restitution">The restitution.</param>
        /// <param name="isSensor">Whether it is a sensor.</param>
        /// <param name="categoryBits">The collision category bits.</param>
        /// <param name="maskBits">The collision mask bits.</param>
        /// <param name="groupIndex">The collision group.</param>
        /// <param name="userData">The user data.</param>
        /// <returns>The fixture.</returns>
        /// <exception cref="AssertionError">Thrown while the world is stepping.</exception>
        public b2Fixture CreateFixture(b2Shape shape, double density = 0.0, double friction = 0.2, double restitution = 0.0, bool isSensor = false,
                                       ushort categoryBits = 0x0001, ushort maskBits = 0xFFFF, short groupIndex = 0, object? userData = null) =>
            CreateFixture(new b2FixtureDef(shape, density, friction, restitution, isSensor, categoryBits, maskBits, groupIndex, userData));

        /// <summary>
        ///     Creates an edge fixture without a definition (pybox2d's <c>CreateEdgeFixture(vertices=..., ...)</c>); a refused
        ///     argument raises pybox2d's <c>Failed on kwargs, class="b2EdgeShape" key="vertices": …</c>.
        /// </summary>
        /// <param name="vertices">Two to four vertices, as <see cref="b2EdgeShape.Vertices"/> takes them.</param>
        /// <param name="density">The density.</param>
        /// <param name="friction">The friction.</param>
        /// <param name="restitution">The restitution.</param>
        /// <param name="isSensor">Whether it is a sensor.</param>
        /// <param name="categoryBits">The collision category bits.</param>
        /// <param name="maskBits">The collision mask bits.</param>
        /// <param name="groupIndex">The collision group.</param>
        /// <returns>The fixture.</returns>
        /// <exception cref="ValueError">Thrown for other than 2 to 4 vertices.</exception>
        /// <exception cref="AssertionError">Thrown while the world is stepping.</exception>
        /// <exception cref="OverflowError">Thrown for a material scalar beyond the float32 range.</exception>
        public b2Fixture CreateEdgeFixture(IReadOnlyList<b2Vec2?> vertices, double density = 0.0, double friction = 0.2, double restitution = 0.0,
                                           bool isSensor = false, ushort categoryBits = 0x0001, ushort maskBits = 0xFFFF, short groupIndex = 0) {
            var shape = new b2EdgeShape();
            var fixture = new b2FixtureDef { shape = shape };
            Box2DConvert.ShapeFixtureKwarg("b2EdgeShape", "vertices", () => shape.Vertices = vertices);
            SetShapeFixtureMaterial(fixture, density, friction, restitution, isSensor, categoryBits, maskBits, groupIndex);
            return CreateFixture(fixture);
        }

        /// <summary>
        ///     Creates a polygon fixture without a definition (pybox2d's <c>CreatePolygonFixture(vertices=... or box=..., ...)</c>);
        ///     a refused argument raises pybox2d's <c>Failed on kwargs, class="b2PolygonShape" key="box": …</c>.
        /// </summary>
        /// <remarks>
        ///     Footgun, as in pybox2d: with neither <paramref name="vertices"/> nor <paramref name="box"/> the fixture gets an
        ///     empty polygon, accepted at density 0 and refused by Box2D's mass assertion (<c>m_count &gt;= 3</c>) above it.
        /// </remarks>
        /// <param name="vertices">The vertices, or null.</param>
        /// <param name="box">The half-extents of an axis-aligned box, applied after <paramref name="vertices"/>.</param>
        /// <param name="density">The density.</param>
        /// <param name="friction">The friction.</param>
        /// <param name="restitution">The restitution.</param>
        /// <param name="isSensor">Whether it is a sensor.</param>
        /// <param name="categoryBits">The collision category bits.</param>
        /// <param name="maskBits">The collision mask bits.</param>
        /// <param name="groupIndex">The collision group.</param>
        /// <returns>The fixture.</returns>
        /// <exception cref="ValueError">Thrown for a vertex count outside 2..16.</exception>
        /// <exception cref="AssertionError">Thrown while the world is stepping, or for a degenerate or (with a density) empty polygon.</exception>
        /// <exception cref="OverflowError">Thrown for a half-extent or material scalar beyond the float32 range.</exception>
        public b2Fixture CreatePolygonFixture(IReadOnlyList<b2Vec2>? vertices = null, (double hx, double hy)? box = null, double density = 0.0,
                                              double friction = 0.2, double restitution = 0.0, bool isSensor = false, ushort categoryBits = 0x0001,
                                              ushort maskBits = 0xFFFF, short groupIndex = 0) {
            var shape = new b2PolygonShape();
            var fixture = new b2FixtureDef { shape = shape };
            if (vertices != null)
                Box2DConvert.ShapeFixtureKwarg("b2PolygonShape", "vertices", () => shape.Vertices = vertices);
            if (box.HasValue)
                Box2DConvert.ShapeFixtureKwarg("b2PolygonShape", "box", () => shape.Box = box.Value);
            SetShapeFixtureMaterial(fixture, density, friction, restitution, isSensor, categoryBits, maskBits, groupIndex);
            return CreateFixture(fixture);
        }

        /// <summary>
        ///     Creates a circle fixture without a definition (pybox2d's <c>CreateCircleFixture(radius=..., pos=..., ...)</c>); a
        ///     refused argument raises pybox2d's <c>Failed on kwargs, class="b2CircleShape" key="radius": …</c>.
        /// </summary>
        /// <param name="radius">The radius.</param>
        /// <param name="pos">The center in the body's frame (origin when null).</param>
        /// <param name="density">The density.</param>
        /// <param name="friction">The friction.</param>
        /// <param name="restitution">The restitution.</param>
        /// <param name="isSensor">Whether it is a sensor.</param>
        /// <param name="categoryBits">The collision category bits.</param>
        /// <param name="maskBits">The collision mask bits.</param>
        /// <param name="groupIndex">The collision group.</param>
        /// <returns>The fixture.</returns>
        /// <exception cref="AssertionError">Thrown while the world is stepping.</exception>
        /// <exception cref="OverflowError">Thrown for the radius or a material scalar beyond the float32 range.</exception>
        public b2Fixture CreateCircleFixture(double radius, b2Vec2? pos = null, double density = 0.0, double friction = 0.2, double restitution = 0.0,
                                             bool isSensor = false, ushort categoryBits = 0x0001, ushort maskBits = 0xFFFF, short groupIndex = 0) {
            var shape = new b2CircleShape();
            var fixture = new b2FixtureDef { shape = shape };
            Box2DConvert.ShapeFixtureKwarg("b2CircleShape", "radius", () => shape.m_radius = Box2DConvert.ToFloat32(radius, "b2Shape_radius_set", 2));
            if (pos.HasValue)
                shape.m_p = pos.Value;
            SetShapeFixtureMaterial(fixture, density, friction, restitution, isSensor, categoryBits, maskBits, groupIndex);
            return CreateFixture(fixture);
        }

        /// <summary>
        ///     Sets the material keywords of a fixture shortcut on its definition, as pybox2d's <c>__CreateShapeFixture</c>
        ///     sets every keyword the shape lacks: refused scalars raise its <c>class="b2FixtureDef"</c> format.
        /// </summary>
        /// <param name="fixture">The shortcut's definition, already holding its shape.</param>
        /// <param name="density">The density.</param>
        /// <param name="friction">The friction.</param>
        /// <param name="restitution">The restitution.</param>
        /// <param name="isSensor">Whether it is a sensor.</param>
        /// <param name="categoryBits">The collision category bits.</param>
        /// <param name="maskBits">The collision mask bits.</param>
        /// <param name="groupIndex">The collision group.</param>
        /// <exception cref="OverflowError">Thrown, prefixed, for a scalar beyond the float32 range.</exception>
        private static void SetShapeFixtureMaterial(b2FixtureDef fixture, double density, double friction, double restitution, bool isSensor,
                                                    ushort categoryBits, ushort maskBits, short groupIndex) {
            Box2DConvert.ShapeFixtureKwarg("b2FixtureDef", "density", () => fixture.density = Box2DConvert.ToFloat32(density, "b2FixtureDef_density_set", 2));
            Box2DConvert.ShapeFixtureKwarg("b2FixtureDef", "friction", () => fixture.friction = Box2DConvert.ToFloat32(friction, "b2FixtureDef_friction_set", 2));
            Box2DConvert.ShapeFixtureKwarg("b2FixtureDef", "restitution", () => fixture.restitution = Box2DConvert.ToFloat32(restitution, "b2FixtureDef_restitution_set", 2));
            fixture.isSensor = isSensor;
            fixture.filter.categoryBits = categoryBits;
            fixture.filter.maskBits = maskBits;
            fixture.filter.groupIndex = groupIndex;
        }

        /// <summary>
        ///     Creates one fixture per shape with a shared material (pybox2d's <c>CreateFixturesFromShapes</c>): the material's
        ///     shape is replaced for each and restored afterwards, even on failure.
        /// </summary>
        /// <param name="shapes">The shapes.</param>
        /// <param name="shapeFixture">The material (Box2D's defaults when null).</param>
        /// <returns>The fixtures, in order.</returns>
        /// <exception cref="AssertionError">Thrown while the world is stepping.</exception>
        public IReadOnlyList<b2Fixture> CreateFixturesFromShapes(IEnumerable<b2Shape> shapes, b2FixtureDef? shapeFixture = null) {
            shapeFixture ??= new b2FixtureDef();
            b2Shape? oldShape = shapeFixture.shape;
            var created = new List<b2Fixture>();
            try {
                foreach (b2Shape shape in shapes) {
                    shapeFixture.shape = shape;
                    created.Add(CreateFixture(shapeFixture));
                }
            } finally {
                // pybox2d restores the definition's own shape whatever happened.
                shapeFixture.shape = oldShape;
            }

            return created;
        }
    }
}
