using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using Gym.Box2D;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Session;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Python.Runtime;

namespace Gym.Tests.Parity.Tests.Box2D {
    /// <summary>
    ///     The Box2D engine (P8.1, D5, DR-26) against live pybox2d 2.3.10, float32 bit for bit: scenes described once as
    ///     data, built and stepped by both engines through their pybox2d-shaped surfaces, every body's position, angle,
    ///     velocities and awake flag compared after every step; the contact listener's begin and end events in order; ray
    ///     casts with every callback answer; Box2D's assertion texts and pybox2d's conversion errors.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A scene is a JSON list of commands (bodies with fixtures, revolute joints, initial forces, impulses and torques,
    ///     per-step motor schedules) read by a Python snippet that drives pybox2d and by <see cref="SceneRunner"/>, which
    ///     drives Gym.Box2D; both report the same flat float32 trace, so a divergence names its step, body and field.
    ///     Hand-written scenes cover each subsystem (polygon, edge and circle manifolds, the block solver, sleeping, revolute
    ///     motors and limits, bullets through time of impact, sensors, filters, kinematic bodies); seeded random scenes mix
    ///     them.
    ///     </para>
    ///     <para>
    ///     The scenes also settle the platform question of the completion plan's §8: Box2D's rotations call the C runtime's
    ///     <c>sinf</c>/<c>cosf</c>, and .NET's <see cref="MathF"/> must give the same bits. That holds on the reference
    ///     platform (win-x64); elsewhere the cells are Inconclusive, as CartPole's are.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Dynamics/b2World.cpp">Box2D/Dynamics/b2World.cpp</see>.</para>
    /// </remarks>
    [TestClass]
    public sealed class Box2DEngineLiveParityTests : ParityTestBase {
        /// <summary>The fields recorded per body per step: x, y, angle, vx, vy, angular velocity, awake (1 or 0).</summary>
        private const int Fields = 7;

        /// <summary>The field names of a body's row, for divergence reports.</summary>
        private static readonly string[] FieldNames = { "position.x", "position.y", "angle", "linearVelocity.x", "linearVelocity.y", "angularVelocity", "awake" };

        /// <summary>The field names of a ray hit, for divergence reports.</summary>
        private static readonly string[] HitNames = { "ray", "userData", "point.x", "point.y", "normal.x", "normal.y", "fraction" };

        /// <summary>
        ///     The pybox2d side of a scene: builds the world from <c>scene_json</c> and steps it, leaving the float32 trace in
        ///     <c>trace</c> and the contact events in <c>events</c> (step, 1 begin or 2 end, userData A, userData B).
        /// </summary>
        private const string ScenePython =
            "import json\n" +
            "import numpy as np\n" +
            "import Box2D\n" +
            "from Box2D.b2 import fixtureDef, polygonShape, circleShape, edgeShape, revoluteJointDef, contactListener\n" +
            "desc = json.loads(scene_json)\n" +
            "events = []\n" +
            "step = [0]\n" +
            "def _recorder_init(self):\n" +
            "    contactListener.__init__(self)\n" +
            "def _begin(self, c):\n" +
            "    events.append((step[0], 1, c.fixtureA.body.userData, c.fixtureB.body.userData))\n" +
            "def _end(self, c):\n" +
            "    events.append((step[0], 2, c.fixtureA.body.userData, c.fixtureB.body.userData))\n" +
            "Recorder = type('Recorder', (contactListener,), {'__init__': _recorder_init, 'BeginContact': _begin, 'EndContact': _end})\n" +
            "world = Box2D.b2World(gravity=tuple(desc['gravity']), doSleep=desc['doSleep'])\n" +
            "world.SetAllowSleeping(desc['allowSleep'])\n" +
            "recorder = Recorder()\n" +
            "world.contactListener = recorder\n" +
            "def make_fixture(f):\n" +
            "    s = f['shape']\n" +
            "    if s == 'polygon':\n" +
            "        shape = polygonShape(vertices=[tuple(v) for v in f['vertices']])\n" +
            "    elif s == 'box':\n" +
            "        shape = polygonShape(box=(f['hx'], f['hy']))\n" +
            "    elif s == 'circle':\n" +
            "        shape = circleShape(radius=f['r'], pos=tuple(f['pos']))\n" +
            "    else:\n" +
            "        shape = edgeShape(vertices=[tuple(v) for v in f['v']])\n" +
            "    return fixtureDef(shape=shape, density=f['density'], friction=f['friction'], restitution=f['restitution'],\n" +
            "                      categoryBits=f['category'], maskBits=f['mask'], groupIndex=f['group'], isSensor=f['sensor'])\n" +
            "bodies = []\n" +
            "joints = []\n" +
            "types = {'static': Box2D.b2_staticBody, 'kinematic': Box2D.b2_kinematicBody, 'dynamic': Box2D.b2_dynamicBody}\n" +
            "for op in desc['ops']:\n" +
            "    kind = op['op']\n" +
            "    if kind == 'body':\n" +
            "        b = world.CreateBody(type=types[op['type']], position=tuple(op['pos']), angle=op['angle'], linearVelocity=tuple(op['lv']),\n" +
            "                             angularVelocity=op['av'], bullet=op['bullet'], fixedRotation=op['fixedRotation'],\n" +
            "                             fixtures=[make_fixture(f) for f in op['fixtures']])\n" +
            "        b.userData = len(bodies)\n" +
            "        bodies.append(b)\n" +
            "    elif kind == 'joint':\n" +
            "        kw = dict(bodyA=bodies[op['a']], bodyB=bodies[op['b']], localAnchorA=tuple(op['anchorA']), localAnchorB=tuple(op['anchorB']),\n" +
            "                  enableMotor=op['motor'], motorSpeed=op['speed'], maxMotorTorque=op['torque'], enableLimit=op['limit'],\n" +
            "                  lowerAngle=op['lower'], upperAngle=op['upper'], collideConnected=op['collide'])\n" +
            "        if op['ref'] is not None:\n" +
            "            kw['referenceAngle'] = op['ref']\n" +
            "        joints.append(world.CreateJoint(revoluteJointDef(**kw)))\n" +
            "    elif kind == 'force':\n" +
            "        bodies[op['body']].ApplyForceToCenter(tuple(op['f']), True)\n" +
            "    elif kind == 'impulse':\n" +
            "        bodies[op['body']].ApplyLinearImpulse(tuple(op['imp']), tuple(op['point']), True)\n" +
            "    elif kind == 'torque':\n" +
            "        bodies[op['body']].ApplyTorque(op['t'], True)\n" +
            "rows = []\n" +
            "schedule = desc['schedule']\n" +
            "for i in range(desc['steps']):\n" +
            "    step[0] = i\n" +
            "    for m in schedule:\n" +
            "        j = joints[m['joint']]\n" +
            "        j.motorSpeed = m['speeds'][i % len(m['speeds'])]\n" +
            "        j.maxMotorTorque = m['torques'][i % len(m['torques'])]\n" +
            "    world.Step(desc['dt'], desc['vel'], desc['pos'])\n" +
            "    for b in bodies:\n" +
            "        p = b.position\n" +
            "        v = b.linearVelocity\n" +
            "        rows.append((p.x, p.y, b.angle, v.x, v.y, b.angularVelocity, 1.0 if b.awake else 0.0))\n" +
            "trace = np.array(rows, dtype=np.float32).reshape(-1)\n";

        /// <summary>The scene cells' result: the trace and the flattened events, as a tuple of two arrays.</summary>
        private const string SceneResultPython =
            ScenePython +
            "result = (trace, np.array(events, dtype=np.int64).reshape(-1))\n";

        /// <summary>
        ///     The pybox2d side of the ray-cast cell: builds and steps the scene, then casts each ray with a callback that
        ///     answers per its mode; the result's first array lists (ray, userData, point x, point y, normal x, normal y,
        ///     fraction) in callback order, its second the contact events of the steps.
        /// </summary>
        private const string RayCastPython =
            ScenePython +
            "from Box2D.b2 import rayCastCallback\n" +
            "hits = []\n" +
            "def _ray_init(self, ray, mode):\n" +
            "    rayCastCallback.__init__(self)\n" +
            "    self.ray = ray\n" +
            "    self.mode = mode\n" +
            "def _report(self, fixture, point, normal, fraction):\n" +
            "    hits.append((self.ray, fixture.body.userData, point[0], point[1], normal[0], normal[1], fraction))\n" +
            "    if self.mode == 'closest':\n" +
            "        return fraction\n" +
            "    if self.mode == 'all':\n" +
            "        return 1.0\n" +
            "    if self.mode == 'first':\n" +
            "        return 0.0\n" +
            "    return -1.0 if (fixture.filterData.categoryBits & 1) == 0 else fraction\n" +
            "Ray = type('Ray', (rayCastCallback,), {'__init__': _ray_init, 'ReportFixture': _report})\n" +
            "for k, r in enumerate(desc['rays']):\n" +
            "    world.RayCast(Ray(k, r['mode']), tuple(r['p1']), tuple(r['p2']))\n" +
            "result = (np.array(hits, dtype=np.float32).reshape(-1), np.array(events, dtype=np.int64).reshape(-1))\n";

        /// <summary>LunarLander's hull polygon in pixels (divided by its <c>SCALE</c> of 30 for metres).</summary>
        private static readonly (double x, double y)[] LanderPoly = { (-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17) };

        /// <summary>LunarLander's hull falling onto an edge ground and settling (density 5, a 300 N push), 400 steps.</summary>
        [TestMethod]
        public void LanderOnEdge_MatchesPybox2d() {
            var scene = Scene.Lunar(400);
            scene.Ground(-1, 0, 21, 0);
            (double x, double y)[] hull = LanderPoly.Select(p => (p.x / 30.0, p.y / 30.0)).ToArray();
            int lander = scene.Body("dynamic", 10, 400 / 30.0, fixtures: Fixture.Polygon(hull, density: 5.0, friction: 0.1, category: 0x0010, mask: 0x001));
            scene.Force(lander, 300, 0);
            AssertSceneMatches(scene, "lander on an edge");
        }

        /// <summary>
        ///     A pyramid of 15 boxes on an edge: polygon-polygon manifolds, the 2x2 block solver, warm starting, and sleeping
        ///     (required to happen, so the sleep path is compared and not merely possible).
        /// </summary>
        [TestMethod]
        public void BoxPyramid_MatchesPybox2d() {
            Scene scene = Pyramid();
            float[] trace = AssertSceneMatches(scene, "box pyramid");
            Assert.IsTrue(SleepingAtEnd(scene, trace) > 0, "box pyramid: no body fell asleep, so the sleep path went untested");
        }

        /// <summary>
        ///     The pyramid in a world made with <c>doSleep=False</c>: bodies still fall asleep, in pybox2d (whose constructor
        ///     stores the flag in a plain attribute the engine never reads) and in Gym.NET, which keeps that behavior.
        /// </summary>
        [TestMethod]
        public void DoSleepFalse_StillSleeps_AsInPybox2d() {
            Scene scene = Pyramid();
            scene.DoSleep = false;
            float[] trace = AssertSceneMatches(scene, "box pyramid, doSleep=False");
            Assert.IsTrue(SleepingAtEnd(scene, trace) > 0, "doSleep=False: no body fell asleep, so pybox2d's inert flag went untested");
        }

        /// <summary>A chain of revolute joints with motors and one limit, a ball dropped onto it: joint solver, motors, limits, circle-polygon.</summary>
        [TestMethod]
        public void RevoluteChain_MatchesPybox2d() {
            var scene = Scene.Lunar(400);
            int ground = scene.Ground(-20, 0, 20, 0);
            int prev = ground;
            for (int i = 0; i < 6; i++) {
                int link = scene.Body("dynamic", 0.5 + i, 10, fixtures: Fixture.Box(0.5, 0.125, density: 20.0, friction: 0.2));
                scene.Joint(prev, link, i == 0 ? (0, 10) : (0.5, 0), (-0.5, 0), motor: i % 2 == 0, speed: 1.5, torque: 50.0, limit: i == 3, lower: -0.5, upper: 0.7);
                prev = link;
            }

            scene.Body("dynamic", 3.2, 14, fixtures: Fixture.Circle(0.6, density: 2.0, restitution: 0.5));
            AssertSceneMatches(scene, "revolute chain");
        }

        /// <summary>
        ///     BipedalWalker-like legs: a hull and four legs on motorized, limited joints whose speeds and torques change every
        ///     step (setting them wakes the bodies, as the env's actions do), walking on polygon terrain.
        /// </summary>
        [TestMethod]
        public void MotorSchedule_MatchesPybox2d() {
            var scene = Scene.Lunar(300);
            scene.Ground(-30, 0, 30, 0);
            for (int t = 0; t < 6; t++)
                scene.Body("static", -10 + t * 4.0, 0, fixtures: Fixture.Polygon(new[] { (0.0, 0.0), (3.0, 0.0), (3.0, 0.4 + 0.1 * t), (0.0, 0.2 * t) }, friction: 2.5));
            int hull = scene.Body("dynamic", 0, 3.0, fixtures: Fixture.Polygon(new[] { (-0.9, 0.3), (0.2, 0.3), (1.1, 0.0), (1.1, -0.4), (-0.9, -0.4) }, density: 5.0, friction: 0.1, category: 0x0020, mask: 0x001));
            var joints = new List<int>();
            for (int side = -1; side <= 1; side += 2) {
                int leg = scene.Body("dynamic", 0, 2.0, angle: side * 0.05, fixtures: Fixture.Box(0.13, 0.57, density: 1.0, category: 0x0020, mask: 0x001));
                joints.Add(scene.Joint(hull, leg, (0, -0.3), (0, 0.57), motor: true, speed: side, torque: 80.0, limit: true, lower: -0.8, upper: 1.1));
                int lower = scene.Body("dynamic", 0, 0.9, angle: side * 0.05, fixtures: Fixture.Box(0.1, 0.57, density: 1.0, category: 0x0020, mask: 0x001));
                joints.Add(scene.Joint(leg, lower, (0, -0.57), (0, 0.57), motor: true, speed: 1, torque: 80.0, limit: true, lower: -1.6, upper: -0.1));
            }

            scene.Force(hull, 3.7, 0);
            double[] speeds = { 4.0, -4.0, 6.0, -6.0, 0.0, 4.0, -6.0 };
            double[] torques = { 80.0, 40.0, 0.0, 80.0, 20.0 };
            for (int j = 0; j < joints.Count; j++)
                scene.Schedule(j, speeds.Skip(j).Concat(speeds.Take(j)).ToArray(), torques.Skip(j % torques.Length).Concat(torques.Take(j % torques.Length)).ToArray());
            AssertSceneMatches(scene, "motor schedule");
        }

        /// <summary>Circles against circles, polygons and edges, with restitution: the circle manifolds and the restitution bias.</summary>
        [TestMethod]
        public void BouncingCircles_MatchesPybox2d() {
            var scene = Scene.Lunar(400);
            scene.Ground(-10, 0, 10, 0);
            scene.Body("static", -10, 5, angle: 0.3, fixtures: Fixture.Edge((-1, -5), (1, 5)));
            scene.Body("static", 3, 2, angle: -0.4, fixtures: Fixture.Box(2.0, 0.2, friction: 0.4));
            for (int i = 0; i < 8; i++)
                scene.Body("dynamic", -6 + i * 1.6, 6 + (i % 3), lv: (i - 4, 0), av: 0.5 * i,
                           fixtures: Fixture.Circle(0.3 + 0.07 * i, pos: (0.02 * i, 0), density: 1.0 + i, friction: 0.3, restitution: 0.1 * i));
            AssertSceneMatches(scene, "bouncing circles");
        }

        /// <summary>
        ///     Bullets fired at thin static walls and edges, and a fast body against static ground: the continuous-collision
        ///     path (time of impact, the sub-step island, the TOI position solver) and the solver's translation clamp.
        /// </summary>
        [TestMethod]
        public void BulletsThroughTimeOfImpact_MatchPybox2d() {
            var scene = Scene.Lunar(200);
            scene.Ground(-50, 0, 50, 0);
            scene.Body("static", 8, 5, fixtures: Fixture.Box(0.05, 5.0));
            scene.Body("static", -8, 5, fixtures: Fixture.Edge((0, -5), (0, 5)));
            for (int i = 0; i < 5; i++) {
                scene.Body("dynamic", -4 + i * 2, 3 + i, angle: 0.2 * i, lv: (i % 2 == 0 ? 180.0 + 13 * i : -170.0 - 11 * i, 2.0 * i), av: 3.0 * i, bullet: i != 3,
                           fixtures: i % 2 == 0 ? Fixture.Circle(0.1, density: 3.0) : Fixture.Box(0.12, 0.08, density: 2.0));
            }

            scene.Body("dynamic", 0, 20, lv: (0, -300), fixtures: Fixture.Box(0.3, 0.3, density: 1.0));
            AssertSceneMatches(scene, "bullets");
        }

        /// <summary>
        ///     Sensors, collision filters and groups, fixed rotation, a kinematic body, and bodies of two fixtures: contacts that
        ///     are filtered out, sensors that touch without response, and their events in order.
        /// </summary>
        [TestMethod]
        public void SensorsFiltersAndKinematics_MatchPybox2d() {
            var scene = Scene.Lunar(300);
            scene.Ground(-15, 0, 15, 0);
            scene.Body("static", 0, 1.5, fixtures: Fixture.Box(3.0, 1.5, sensor: true));
            scene.Body("kinematic", -6, 4, lv: (1.5, 0), av: 0.7, fixtures: Fixture.Box(1.5, 0.2));
            for (int i = 0; i < 6; i++) {
                short group = (short)(i < 2 ? -1 : i < 4 ? 1 : 0);
                scene.Body("dynamic", -4 + i * 1.7, 6 + i * 0.8, fixedRotation: i == 5,
                           fixtures: new[] {
                               Fixture.Box(0.4, 0.4, density: 1.0, category: (ushort)(1 << (i % 3)), mask: (ushort)(i == 1 ? 0xFFFD : 0xFFFF), group: group),
                               Fixture.Circle(0.25, pos: (0.4, 0.4), density: 0.5, group: group),
                           });
            }

            AssertSceneMatches(scene, "sensors, filters and kinematics");
        }

        /// <summary>
        ///     A different time step and iteration counts (1/60 s, 8 velocity and 3 position iterations), with sleeping turned
        ///     off through <c>SetAllowSleeping(False)</c> (required to keep every body awake).
        /// </summary>
        [TestMethod]
        public void OtherStepSettings_MatchPybox2d() {
            var scene = new Scene { Steps = 360, Dt = 1.0 / 60, Vel = 8, Pos = 3, AllowSleep = false };
            scene.Ground(-20, 0, 20, 0);
            for (int i = 0; i < 6; i++)
                scene.Body("dynamic", -3 + i * 1.3, 2 + i * 1.5, angle: 0.3 * i, fixtures: i % 2 == 0 ? Fixture.Box(0.5, 0.25, density: 1.0) : Fixture.Circle(0.4, density: 1.0));
            float[] trace = AssertSceneMatches(scene, "1/60 s, 8/3 iterations, no sleep");
            Assert.AreEqual(0, SleepingAtEnd(scene, trace), "SetAllowSleeping(False): a body fell asleep");
        }

        /// <summary>
        ///     Seeded random scenes: 8 to 24 bodies of random shapes (convex polygons, boxes, circles, edges), types, angles and
        ///     velocities in a walled arena, random revolute joints with motors and limits, random impulses and torques, 250
        ///     steps each, 12 seeds.
        /// </summary>
        [TestMethod]
        public void RandomScenes_MatchPybox2d() {
            for (int seed = 1; seed <= 12; seed++)
                AssertSceneMatches(RandomScene(seed), $"random scene {seed}");
        }

        /// <summary>
        ///     Ray casts after 60 steps of a random scene: each ray with every callback answer (clip to the closest, report all,
        ///     stop at the first, filter by category), every reported fixture, point, normal and fraction in callback order,
        ///     and the steps' contact events.
        /// </summary>
        [TestMethod]
        public void RayCasts_MatchPybox2d() {
            SkipOffReferencePlatform("Box2D's sinf/cosf");
            Scene scene = RandomScene(101);
            scene.Steps = 60;
            var rng = new Random(5);
            string[] modes = { "closest", "all", "first", "filter" };
            for (int i = 0; i < 80; i++) {
                (double, double) p1 = (rng.NextDouble() * 30 - 15, rng.NextDouble() * 15 + 0.5);
                double angle = rng.NextDouble() * 2 * Math.PI;
                double length = 2 + rng.NextDouble() * 25;
                (double, double) p2 = (p1.Item1 + Math.Cos(angle) * length, p1.Item2 + Math.Sin(angle) * length);
                scene.Rays.Add(new Ray(modes[i % modes.Length], p1, p2));
            }

            (float[] expectedHits, long[] expectedEvents) = PythonRun(RayCastPython, scene);
            var runner = new SceneRunner(scene);
            float[] actualHits = runner.RunRays();
            Assert.IsTrue(expectedHits.Length > 0, "ray casts: pybox2d reported no hit, so the cell compares nothing");
            AssertFloatsEqual("ray casts", expectedHits, actualHits, HitNames.Length, HitNames);
            AssertEventsEqual("ray casts' scene", expectedEvents, runner.FlatEvents());
        }

        /// <summary>
        ///     The statements every error case's pybox2d side starts with: a world and two dynamic bodies without fixtures,
        ///     as <see cref="CaptureDotNet"/> builds them for Gym.Box2D.
        /// </summary>
        private const string ErrorPrelude =
            "import Box2D\n" +
            "from Box2D.b2 import circleShape, contactListener, edgeShape, fixtureDef, polygonShape, rayCastCallback, revoluteJointDef\n" +
            "world = Box2D.b2World()\n" +
            "body = world.CreateDynamicBody(position=(0, 0))\n" +
            "other = world.CreateDynamicBody(position=(1, 0))\n";

        /// <summary>
        ///     Box2D's assertions reach the caller as Python's AssertionError carrying C++'s stringified condition, and
        ///     pybox2d's argument checks and float32 conversions raise its errors, the keyword constructors'
        ///     <c>Failed on kwargs</c> prefixes included (both of pybox2d's formats), on both sides: 33 cases over shapes,
        ///     fixture and body definitions, bodies, joints, the fixture shortcuts, forces, the step, a ray of zero length,
        ///     and a body created while the world is locked.
        /// </summary>
        [TestMethod]
        public void ErrorTexts_MatchPybox2d() {
            var cases = new (string What, string PythonType, string Python, Action<b2World, b2Body, b2Body> DotNet)[] {
                ("polygon of two vertices", "AssertionError", "polygonShape(vertices=[(0, 0), (1, 0)])",
                    (w, b, o) => new b2PolygonShape(new b2Vec2[] { (0, 0), (1, 0) })),
                ("collinear polygon", "AssertionError", "polygonShape(vertices=[(0, 0), (1, 0), (2, 0)])",
                    (w, b, o) => new b2PolygonShape(new b2Vec2[] { (0, 0), (1, 0), (2, 0) })),
                ("polygon of one vertex", "ValueError", "polygonShape(vertices=[(0, 0)])",
                    (w, b, o) => new b2PolygonShape(new b2Vec2[] { (0, 0) })),
                ("polygon of 17 vertices", "ValueError", "polygonShape(vertices=[(i, i * i) for i in range(17)])",
                    (w, b, o) => new b2PolygonShape(Enumerable.Range(0, 17).Select(i => new b2Vec2(i, i * i)).ToArray())),
                ("box half-extent beyond float32", "OverflowError", "polygonShape(box=(1e39, 1))",
                    (w, b, o) => new b2PolygonShape(null, (1e39, 1))),
                ("oriented box angle beyond float32", "OverflowError", "polygonShape().SetAsBox(1, 1, (0, 0), 1e39)",
                    (w, b, o) => new b2PolygonShape().SetAsBox(1, 1, (0, 0), 1e39)),
                ("circle radius beyond float32", "OverflowError", "circleShape(radius=1e39)",
                    (w, b, o) => new b2CircleShape(1e39)),
                ("edge of one vertex", "ValueError", "edgeShape(vertices=[(0, 0)])",
                    (w, b, o) => new b2EdgeShape(new b2Vec2?[] { (0, 0) })),
                ("edge of five vertices", "ValueError", "edgeShape(vertices=[(0, 0)] * 5)",
                    (w, b, o) => new b2EdgeShape(Enumerable.Repeat((b2Vec2?)new b2Vec2(0, 0), 5).ToArray())),
                ("density beyond float32", "OverflowError", "fixtureDef(density=1e39)",
                    (w, b, o) => new b2FixtureDef(null, density: 1e39)),
                ("friction below the float32 range", "OverflowError", "fixtureDef(friction=-1e39)",
                    (w, b, o) => new b2FixtureDef(null, friction: -1e39)),
                ("body definition angle beyond float32", "OverflowError", "Box2D.b2BodyDef(angle=1e39)",
                    (w, b, o) => new b2BodyDef(angle: 1e39)),
                ("body angle beyond float32", "OverflowError", "world.CreateDynamicBody(angle=1e39)",
                    (w, b, o) => w.CreateDynamicBody(angle: 1e39)),
                ("gravity scale beyond float32", "OverflowError", "world.CreateDynamicBody(gravityScale=1e39)",
                    (w, b, o) => w.CreateDynamicBody(gravityScale: 1e39)),
                ("NaN position", "AssertionError", "world.CreateDynamicBody(position=(float('nan'), 0))",
                    (w, b, o) => w.CreateDynamicBody(position: (double.NaN, 0))),
                ("NaN angle", "AssertionError", "world.CreateDynamicBody(angle=float('nan'))",
                    (w, b, o) => w.CreateDynamicBody(angle: double.NaN)),
                ("negative damping", "AssertionError", "world.CreateDynamicBody(linearDamping=-1.0)",
                    (w, b, o) => w.CreateDynamicBody(linearDamping: -1.0)),
                ("joint limit beyond float32", "OverflowError", "revoluteJointDef(bodyA=body, bodyB=other, lowerAngle=1e39)",
                    (w, b, o) => new b2RevoluteJointDef(bodyA: b, bodyB: o, lowerAngle: 1e39)),
                ("anchor without bodies", "ValueError", "revoluteJointDef(anchor=(0, 0))",
                    (w, b, o) => new b2RevoluteJointDef(anchor: (0, 0))),
                ("anchor without body B", "ValueError", "revoluteJointDef(bodyA=body, anchor=(0, 0))",
                    (w, b, o) => new b2RevoluteJointDef(bodyA: b, anchor: (0, 0))),
                ("joint without bodies", "ValueError", "world.CreateJoint(revoluteJointDef())",
                    (w, b, o) => w.CreateJoint(new b2RevoluteJointDef())),
                ("joint between a body and itself", "AssertionError", "world.CreateJoint(revoluteJointDef(bodyA=body, bodyB=body))",
                    (w, b, o) => w.CreateJoint(new b2RevoluteJointDef(bodyA: b, bodyB: b))),
                ("polygon fixture box beyond float32", "OverflowError", "body.CreatePolygonFixture(box=(1e39, 1), density=1)",
                    (w, b, o) => b.CreatePolygonFixture(box: (1e39, 1), density: 1)),
                ("polygon fixture density beyond float32", "OverflowError", "body.CreatePolygonFixture(box=(1, 1), density=1e39)",
                    (w, b, o) => b.CreatePolygonFixture(box: (1, 1), density: 1e39)),
                ("empty polygon fixture with a density", "AssertionError", "body.CreatePolygonFixture(density=1)",
                    (w, b, o) => b.CreatePolygonFixture(density: 1)),
                ("edge fixture of one vertex", "ValueError", "body.CreateEdgeFixture(vertices=[(0, 0)])",
                    (w, b, o) => b.CreateEdgeFixture(new b2Vec2?[] { (0, 0) })),
                ("circle fixture radius beyond float32", "OverflowError", "body.CreateCircleFixture(radius=1e39)",
                    (w, b, o) => b.CreateCircleFixture(1e39)),
                ("torque beyond float32", "OverflowError", "body.ApplyTorque(1e39, True)",
                    (w, b, o) => b.ApplyTorque(1e39, true)),
                ("force coordinate beyond float32", "TypeError", "body.ApplyForceToCenter((1e39, 0), True)",
                    (w, b, o) => b.ApplyForceToCenter((1e39, 0), true)),
                ("gravity coordinate beyond float32", "TypeError", "Box2D.b2World(gravity=(0, 1e39))",
                    (w, b, o) => new b2World((0, 1e39))),
                ("time step beyond float32", "OverflowError", "world.Step(1e39, 6, 2)",
                    (w, b, o) => w.Step(1e39, 6, 2)),
                ("ray of zero length", "AssertionError",
                    "world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (1, 0)]))\n" +
                    "R = type('R', (rayCastCallback,), {'ReportFixture': lambda self, f, p, n, fr: fr})\n" +
                    "world.RayCast(R(), (1, 1), (1, 1))",
                    (w, b, o) => {
                        w.CreateStaticBody(new b2EdgeShape(new b2Vec2?[] { (0, 0), (1, 0) }));
                        w.RayCast(new ClosestRay(), (1, 1), (1, 1));
                    }),
                ("a body created from a contact callback", "AssertionError",
                    "def _init(self):\n" +
                    "    contactListener.__init__(self)\n" +
                    "def _begin(self, c):\n" +
                    "    world.CreateDynamicBody(position=(0, 5))\n" +
                    "L = type('L', (contactListener,), {'__init__': _init, 'BeginContact': _begin})\n" +
                    "listener = L()\n" +
                    "world.contactListener = listener\n" +
                    "world.CreateStaticBody(shapes=edgeShape(vertices=[(-5, 0), (5, 0)]))\n" +
                    "world.CreateDynamicBody(position=(0, 0.4), fixtures=fixtureDef(shape=polygonShape(box=(0.5, 0.5)), density=1.0))\n" +
                    "for i in range(5):\n" +
                    "    world.Step(1.0 / 50, 6, 2)",
                    (w, b, o) => {
                        w.ContactListener = new CreatingListener(w);
                        w.CreateStaticBody(new b2EdgeShape(new b2Vec2?[] { (-5, 0), (5, 0) }));
                        w.CreateDynamicBody(new b2FixtureDef(new b2PolygonShape(null, (0.5, 0.5)), density: 1.0), position: (0, 0.4));
                        for (int i = 0; i < 5; i++)
                            w.Step(1.0 / 50, 6, 2);
                    }),
            };
            foreach ((string what, string pythonType, string python, Action<b2World, b2Body, b2Body> dotnet) in cases)
                AssertSameError(what, pythonType, python, dotnet);
        }

        /// <summary>
        ///     The deviation C#'s argument evaluation forces, pinned on both sides so neither drifts unnoticed: a vector
        ///     coordinate beyond float32 in a keyword constructor. pybox2d converts it inside the constructor and prefixes
        ///     the error with the class and keyword (a polygon vertex even fails as SWIG's OverflowError); Gym.Box2D converts
        ///     the tuple where the caller builds the <see cref="b2Vec2"/>, before the constructor runs, so the conversion's
        ///     own TypeError arrives unprefixed.
        /// </summary>
        [TestMethod]
        public void VectorOverflowInKeywords_DeviatesAsDocumented() {
            AssertPinnedDeviation("body position", "world.CreateDynamicBody(position=(0, 1e39))",
                new RaisedError("TypeError", "Failed on kwargs for b2BodyDef.position: Converting from sequence to b2Vec2, expected int/float arguments index 1"),
                (w, b, o) => w.CreateDynamicBody(position: (0, 1e39)),
                new RaisedError("NumSharp.TypeError", "Converting from sequence to b2Vec2, expected int/float arguments index 1"));
            AssertPinnedDeviation("polygon vertex", "polygonShape(vertices=[(0, 0), (1, 0), (1e39, 1)])",
                new RaisedError("OverflowError", "Failed on kwargs for b2PolygonShape.vertices: in method 'b2PolygonShape_set_vertex', argument 3 of type 'float32'"),
                (w, b, o) => new b2PolygonShape(new b2Vec2[] { (0, 0), (1, 0), (1e39, 1) }),
                new RaisedError("NumSharp.TypeError", "Converting from sequence to b2Vec2, expected int/float arguments index 0"));
        }

        /// <summary>
        ///     Runs one error case on both sides and asserts pybox2d raised the expected Python type (the reference must fail
        ///     as expected first: two equal non-failures would pass any comparison), then that Gym.NET raised the mapped type
        ///     with the same message.
        /// </summary>
        /// <param name="what">The case, for the messages.</param>
        /// <param name="pythonType">The Python exception type pybox2d must raise.</param>
        /// <param name="statements">The pybox2d statements after <see cref="ErrorPrelude"/>; their last one must raise.</param>
        /// <param name="dotnet">The Gym.Box2D calls over the same world and bodies; their last one must throw.</param>
        private static void AssertSameError(string what, string pythonType, string statements, Action<b2World, b2Body, b2Body> dotnet) {
            Outcome python = CapturePybox2d(statements);
            Assert.IsNotNull(python.Raised, $"{what}: pybox2d raised nothing");
            Assert.AreEqual(pythonType, python.Raised!.Type, $"{what}: pybox2d's exception type");
            ByteContract.AssertSame(what, python, CaptureDotNet(dotnet));
        }

        /// <summary>
        ///     Asserts a documented deviation exactly: pybox2d raises its error and Gym.Box2D raises its own, each given in
        ///     full, so a change on either side fails the cell instead of passing as parity or hiding as a new deviation.
        /// </summary>
        /// <param name="what">The case, for the messages.</param>
        /// <param name="statements">The pybox2d statements after <see cref="ErrorPrelude"/>.</param>
        /// <param name="python">pybox2d's error: its Python type and message.</param>
        /// <param name="dotnet">The Gym.Box2D calls over the same world and bodies.</param>
        /// <param name="net">Gym.Box2D's error: its .NET type's full name and message.</param>
        private static void AssertPinnedDeviation(string what, string statements, RaisedError python, Action<b2World, b2Body, b2Body> dotnet, RaisedError net) {
            Assert.AreEqual(python, CapturePybox2d(statements).Raised, $"{what}: pybox2d's error");
            Assert.AreEqual(net, CaptureDotNet(dotnet).Raised, $"{what}: Gym.Box2D's error");
        }

        /// <summary>Runs error-case statements in pybox2d after <see cref="ErrorPrelude"/>, describing their outcome.</summary>
        /// <param name="statements">The statements.</param>
        /// <returns>The outcome: <c>None</c>, or the exception's type and message.</returns>
        private static Outcome CapturePybox2d(string statements) =>
            Outcome.CapturePython(() => Python.evalStmts(ErrorPrelude + statements + "\nresult = None\n", "result"));

        /// <summary>
        ///     Runs error-case calls on Gym.Box2D over the prelude's world (a world and two fixtureless dynamic bodies at
        ///     (0, 0) and (1, 0)), describing their outcome.
        /// </summary>
        /// <param name="calls">The calls.</param>
        /// <returns>The outcome: <c>None</c>, or the exception's type and message.</returns>
        private static Outcome CaptureDotNet(Action<b2World, b2Body, b2Body> calls) =>
            Outcome.Capture(() => {
                var world = new b2World();
                b2Body body = world.CreateDynamicBody(position: (0, 0));
                b2Body other = world.CreateDynamicBody(position: (1, 0));
                calls(world, body, other);
                return null;
            });

        /// <summary>Runs a scene on both engines and asserts the traces and the contact events are equal.</summary>
        /// <param name="scene">The scene.</param>
        /// <param name="what">The scene's name, for the messages.</param>
        /// <returns>The compared trace, for checks on what the scene exercised.</returns>
        private static float[] AssertSceneMatches(Scene scene, string what) {
            SkipOffReferencePlatform("Box2D's sinf/cosf");
            (float[] expectedTrace, long[] expectedEvents) = PythonRun(SceneResultPython, scene);
            var runner = new SceneRunner(scene);
            float[] actualTrace = runner.Run();
            Assert.AreEqual(scene.Steps * scene.BodyCount * Fields, expectedTrace.Length, $"{what}: pybox2d's trace length");
            Assert.IsTrue(expectedEvents.Length > 0, $"{what}: pybox2d reported no contact, so the events compare nothing");
            AssertFloatsEqual(what, expectedTrace, actualTrace, scene.BodyCount * Fields, FieldNames, scene.BodyCount);
            AssertEventsEqual(what, expectedEvents, runner.FlatEvents());
            return expectedTrace;
        }

        /// <summary>
        ///     Compares two flat float32 traces bit for bit, reporting the first divergence by row (step or hit), body and
        ///     field.
        /// </summary>
        /// <param name="what">The scene's name.</param>
        /// <param name="expected">pybox2d's trace.</param>
        /// <param name="actual">Gym.Box2D's trace.</param>
        /// <param name="rowLength">The number of floats per row.</param>
        /// <param name="fieldNames">The field names of one body's part of a row.</param>
        /// <param name="bodies">The number of bodies per row (1 when rows are not per body).</param>
        private static void AssertFloatsEqual(string what, float[] expected, float[] actual, int rowLength, string[] fieldNames, int bodies = 1) {
            int fieldsPerBody = rowLength / bodies;
            int common = Math.Min(expected.Length, actual.Length);
            for (int i = 0; i < common; i++) {
                if (BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]))
                    continue;
                int within = i % rowLength;
                Assert.Fail($"{what}: first divergence at row {i / rowLength}, body {within / fieldsPerBody}, {fieldNames[within % fieldsPerBody]}: " +
                            $"pybox2d {Describe(expected[i])} vs Gym.Box2D {Describe(actual[i])}");
            }

            Assert.AreEqual(expected.Length, actual.Length, $"{what}: trace length (equal up to the shorter one)");
        }

        /// <summary>Compares the flattened contact events, reporting the first differing event with both tuples.</summary>
        /// <param name="what">The scene's name.</param>
        /// <param name="expected">pybox2d's events, four values each.</param>
        /// <param name="actual">Gym.Box2D's events, four values each.</param>
        private static void AssertEventsEqual(string what, long[] expected, long[] actual) {
            int common = Math.Min(expected.Length, actual.Length) / 4;
            for (int e = 0; e < common; e++) {
                ArraySegment<long> left = new(expected, e * 4, 4), right = new(actual, e * 4, 4);
                if (!left.SequenceEqual(right))
                    Assert.Fail($"{what}: contact event {e} (step, 1 begin / 2 end, userData A, userData B): pybox2d ({string.Join(", ", left)}) vs Gym.Box2D ({string.Join(", ", right)})");
            }

            Assert.AreEqual(expected.Length / 4, actual.Length / 4, $"{what}: contact event count (equal up to the shorter list)");
        }

        /// <summary>A float32 as its round-trip text and its bits.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The description.</returns>
        private static string Describe(float value) =>
            $"{value.ToString("R", CultureInfo.InvariantCulture)} (0x{BitConverter.SingleToUInt32Bits(value):x8})";

        /// <summary>
        ///     Runs a scene's snippet in pybox2d and reads its result, a tuple of a float32 array and an int64 array, through
        ///     the suite's own value reader.
        /// </summary>
        /// <param name="statements">The snippet; it must leave the tuple in <c>result</c>.</param>
        /// <param name="scene">The scene, bound as <c>scene_json</c>.</param>
        /// <returns>The floats and the events.</returns>
        /// <exception cref="PythonException">Thrown when pybox2d raises.</exception>
        private static (float[] Floats, long[] Events) PythonRun(string statements, Scene scene) {
            ValueFacts? facts = null;
            string json = scene.ToJson();
            PythonRuntime.Run(() => {
                using PyObject sceneJson = PyConvert.ToPython(json);
                using PyObject result = Python.evalStmts(statements, "result", ("scene_json", sceneJson));
                facts = ValueFacts.FromPython(result);
            });
            Assert.AreEqual(2, facts!.Items.Count, "the snippet's result is a pair");
            Assert.AreEqual("<f4", facts.Items[0].DType, "the floats' dtype");
            Assert.AreEqual("<i8", facts.Items[1].DType, "the events' dtype");
            return (MemoryMarshal.Cast<byte, float>(facts.Items[0].Bytes).ToArray(), MemoryMarshal.Cast<byte, long>(facts.Items[1].Bytes).ToArray());
        }

        /// <summary>A pyramid of 15 boxes on an edge ground, 500 steps: it settles and falls asleep within them.</summary>
        /// <returns>The scene.</returns>
        private static Scene Pyramid() {
            var scene = Scene.Lunar(500);
            scene.Ground(-20, 0, 20, 0);
            for (int row = 0; row < 5; row++)
                for (int col = 0; col < 5 - row; col++)
                    scene.Body("dynamic", -2.0 + col * 1.05 + row * 0.52, 0.5 + row * 1.0, angle: 0.01 * (row + col), fixtures: Fixture.Box(0.5, 0.5, density: 1.0, friction: 0.6));
            return scene;
        }

        /// <summary>The number of bodies asleep after a scene's last step, read from its trace's awake flags.</summary>
        /// <param name="scene">The scene.</param>
        /// <param name="trace">Its trace.</param>
        /// <returns>The count (static bodies included, since an island going to sleep clears theirs too).</returns>
        private static int SleepingAtEnd(Scene scene, float[] trace) {
            int lastRow = trace.Length - scene.BodyCount * Fields;
            return Enumerable.Range(0, scene.BodyCount).Count(b => trace[lastRow + b * Fields + 6] == 0.0f);
        }

        /// <summary>A seeded random scene: an arena, random bodies and joints, random impulses and torques.</summary>
        /// <param name="seed">The seed.</param>
        /// <returns>The scene.</returns>
        private static Scene RandomScene(int seed) {
            var rng = new Random(seed);
            var scene = Scene.Lunar(250);
            scene.Ground(-16, 0, 16, 0);
            scene.Body("static", -16, 0, fixtures: Fixture.Edge((0, 0), (0, 20)));
            scene.Body("static", 16, 0, fixtures: Fixture.Edge((0, 0), (0, 20)));
            int count = 8 + rng.Next(17);
            var dynamics = new List<int>();
            for (int i = 0; i < count; i++) {
                double x = rng.NextDouble() * 28 - 14, y = 1 + rng.NextDouble() * 14;
                string type = rng.Next(10) < 8 ? "dynamic" : rng.Next(2) == 0 ? "static" : "kinematic";
                Fixture fixture = rng.Next(4) switch {
                    0 => Fixture.Box(0.2 + rng.NextDouble(), 0.2 + rng.NextDouble(), density: 0.5 + rng.NextDouble() * 3, friction: rng.NextDouble(), restitution: rng.NextDouble() * 0.6),
                    1 => Fixture.Circle(0.15 + rng.NextDouble() * 0.8, pos: (rng.NextDouble() * 0.2, rng.NextDouble() * 0.2), density: 0.5 + rng.NextDouble() * 3, friction: rng.NextDouble(), restitution: rng.NextDouble() * 0.6),
                    2 => Fixture.Polygon(RandomConvex(rng), density: 0.5 + rng.NextDouble() * 3, friction: rng.NextDouble(), restitution: rng.NextDouble() * 0.4),
                    _ => type == "dynamic" ? Fixture.Box(0.3, 0.6, density: 1.0) : Fixture.Edge((-1 - rng.NextDouble(), rng.NextDouble() - 0.5), (1 + rng.NextDouble(), rng.NextDouble() - 0.5)),
                };
                int body = scene.Body(type, x, y, angle: rng.NextDouble() * 6.3 - 3.15, lv: type == "static" ? (0, 0) : (rng.NextDouble() * 6 - 3, rng.NextDouble() * 6 - 3),
                                      av: type == "static" ? 0 : rng.NextDouble() * 4 - 2, bullet: rng.Next(6) == 0, fixtures: fixture);
                if (type == "dynamic")
                    dynamics.Add(body);
            }

            for (int j = 0; j + 1 < dynamics.Count && j < 6; j += 2) {
                bool limit = rng.Next(2) == 0;
                double lower = -rng.NextDouble(), upper = rng.NextDouble();
                scene.Joint(dynamics[j], dynamics[j + 1], (rng.NextDouble() - 0.5, rng.NextDouble() - 0.5), (rng.NextDouble() - 0.5, rng.NextDouble() - 0.5),
                            motor: rng.Next(2) == 0, speed: rng.NextDouble() * 4 - 2, torque: rng.NextDouble() * 60, limit: limit, lower: lower, upper: upper,
                            reference: rng.Next(3) == 0 ? rng.NextDouble() - 0.5 : null, collide: rng.Next(4) == 0);
            }

            foreach (int b in dynamics.Where((_, k) => k % 3 == 0))
                scene.Impulse(b, (rng.NextDouble() * 20 - 10, rng.NextDouble() * 20), (rng.NextDouble() - 0.5, rng.NextDouble() - 0.5));
            foreach (int b in dynamics.Where((_, k) => k % 4 == 1))
                scene.Torque(b, rng.NextDouble() * 40 - 20);
            return scene;
        }

        /// <summary>A random convex polygon of 3 to 8 vertices, from points on a jittered ellipse.</summary>
        /// <param name="rng">The generator.</param>
        /// <returns>The vertices.</returns>
        private static (double x, double y)[] RandomConvex(Random rng) {
            int n = 3 + rng.Next(6);
            double radius = 0.3 + rng.NextDouble() * 0.9;
            var points = new (double, double)[n];
            for (int i = 0; i < n; i++) {
                double a = (i + rng.NextDouble() * 0.6) * 2 * Math.PI / n;
                points[i] = (Math.Cos(a) * radius, Math.Sin(a) * radius * (0.6 + rng.NextDouble() * 0.4));
            }

            return points;
        }

        /// <summary>A ray-cast callback that clips to each hit, for the error cell.</summary>
        private sealed class ClosestRay : b2RayCastCallback {
            /// <summary>Clips the ray to the hit.</summary>
            /// <param name="fixture">The fixture hit.</param>
            /// <param name="point">The point of intersection.</param>
            /// <param name="normal">The normal there.</param>
            /// <param name="fraction">The fraction along the ray.</param>
            /// <returns><paramref name="fraction"/>.</returns>
            public override float ReportFixture(b2Fixture fixture, b2Vec2 point, b2Vec2 normal, float fraction) => fraction;
        }

        /// <summary>A contact listener that creates a body when a contact begins, which a locked world refuses.</summary>
        private sealed class CreatingListener : b2ContactListener {
            /// <summary>The world to create the body in.</summary>
            private readonly b2World _world;

            /// <summary>A listener over a world.</summary>
            /// <param name="world">The world.</param>
            public CreatingListener(b2World world) => _world = world;

            /// <summary>Creates a body during the step, as the Python listener does.</summary>
            /// <param name="contact">The contact.</param>
            /// <exception cref="NumSharp.AssertionError">Always: the world is locked while it steps.</exception>
            public override void BeginContact(b2Contact contact) => _world.CreateDynamicBody(position: (0, 5));
        }

        /// <summary>A fixture of a scene: shape and material, as JSON for both interpreters.</summary>
        internal sealed class Fixture {
            /// <summary>The fixture's JSON object.</summary>
            public Dictionary<string, object> Json { get; } = new Dictionary<string, object>();

            /// <summary>A fixture of a shape with a material.</summary>
            /// <param name="shape">The shape's JSON fields.</param>
            /// <param name="density">The density.</param>
            /// <param name="friction">The friction.</param>
            /// <param name="restitution">The restitution.</param>
            /// <param name="category">The category bits.</param>
            /// <param name="mask">The mask bits.</param>
            /// <param name="group">The group.</param>
            /// <param name="sensor">Whether it is a sensor.</param>
            private Fixture(Dictionary<string, object> shape, double density, double friction, double restitution, ushort category, ushort mask, short group, bool sensor) {
                foreach (KeyValuePair<string, object> pair in shape)
                    Json[pair.Key] = pair.Value;
                Json["density"] = density;
                Json["friction"] = friction;
                Json["restitution"] = restitution;
                Json["category"] = (int)category;
                Json["mask"] = (int)mask;
                Json["group"] = (int)group;
                Json["sensor"] = sensor;
            }

            /// <summary>A polygon fixture (pybox2d's <c>polygonShape(vertices=...)</c>, Box2D's convex hull of the points).</summary>
            /// <param name="vertices">The vertices.</param>
            /// <param name="density">The density.</param>
            /// <param name="friction">The friction.</param>
            /// <param name="restitution">The restitution.</param>
            /// <param name="category">The category bits.</param>
            /// <param name="mask">The mask bits.</param>
            /// <param name="group">The group.</param>
            /// <param name="sensor">Whether it is a sensor.</param>
            /// <returns>The fixture.</returns>
            public static Fixture Polygon((double x, double y)[] vertices, double density = 0, double friction = 0.2, double restitution = 0, ushort category = 1, ushort mask = 0xFFFF, short group = 0, bool sensor = false) {
                var shape = new Dictionary<string, object> { ["shape"] = "polygon", ["vertices"] = vertices.Select(v => new[] { v.x, v.y }).ToArray() };
                return new Fixture(shape, density, friction, restitution, category, mask, group, sensor);
            }

            /// <summary>A box fixture (pybox2d's <c>polygonShape(box=(hx, hy))</c>).</summary>
            /// <param name="hx">The half-width.</param>
            /// <param name="hy">The half-height.</param>
            /// <param name="density">The density.</param>
            /// <param name="friction">The friction.</param>
            /// <param name="restitution">The restitution.</param>
            /// <param name="category">The category bits.</param>
            /// <param name="mask">The mask bits.</param>
            /// <param name="group">The group.</param>
            /// <param name="sensor">Whether it is a sensor.</param>
            /// <returns>The fixture.</returns>
            public static Fixture Box(double hx, double hy, double density = 0, double friction = 0.2, double restitution = 0, ushort category = 1, ushort mask = 0xFFFF, short group = 0, bool sensor = false) {
                var shape = new Dictionary<string, object> { ["shape"] = "box", ["hx"] = hx, ["hy"] = hy };
                return new Fixture(shape, density, friction, restitution, category, mask, group, sensor);
            }

            /// <summary>A circle fixture.</summary>
            /// <param name="r">The radius.</param>
            /// <param name="pos">The local center.</param>
            /// <param name="density">The density.</param>
            /// <param name="friction">The friction.</param>
            /// <param name="restitution">The restitution.</param>
            /// <param name="category">The category bits.</param>
            /// <param name="mask">The mask bits.</param>
            /// <param name="group">The group.</param>
            /// <param name="sensor">Whether it is a sensor.</param>
            /// <returns>The fixture.</returns>
            public static Fixture Circle(double r, (double x, double y) pos = default, double density = 0, double friction = 0.2, double restitution = 0, ushort category = 1, ushort mask = 0xFFFF, short group = 0, bool sensor = false) {
                var shape = new Dictionary<string, object> { ["shape"] = "circle", ["r"] = r, ["pos"] = new[] { pos.x, pos.y } };
                return new Fixture(shape, density, friction, restitution, category, mask, group, sensor);
            }

            /// <summary>An edge fixture of Box2D's default material (friction 0.2 unless given).</summary>
            /// <param name="v1">The first vertex.</param>
            /// <param name="v2">The second vertex.</param>
            /// <param name="friction">The friction.</param>
            /// <returns>The fixture.</returns>
            public static Fixture Edge((double x, double y) v1, (double x, double y) v2, double friction = 0.2) {
                var shape = new Dictionary<string, object> { ["shape"] = "edge", ["v"] = new[] { new[] { v1.x, v1.y }, new[] { v2.x, v2.y } } };
                return new Fixture(shape, 0, friction, 0, 1, 0xFFFF, 0, false);
            }
        }

        /// <summary>A ray of the ray-cast cell.</summary>
        /// <param name="Mode">The callback answer: closest, all, first or filter.</param>
        /// <param name="P1">The start.</param>
        /// <param name="P2">The end.</param>
        internal sealed record Ray(string Mode, (double x, double y) P1, (double x, double y) P2);

        /// <summary>
        ///     A scene: settings and a list of commands, serialized to JSON for both interpreters (doubles round-trip exactly,
        ///     so both read the same values).
        /// </summary>
        internal sealed class Scene {
            /// <summary>The number of joints created so far.</summary>
            private int _joints;

            /// <summary>The commands, in order.</summary>
            public List<Dictionary<string, object?>> Ops { get; } = new List<Dictionary<string, object?>>();

            /// <summary>The per-step motor schedules.</summary>
            public List<Dictionary<string, object>> MotorSchedule { get; } = new List<Dictionary<string, object>>();

            /// <summary>The rays of the ray-cast cell, cast after the steps.</summary>
            public List<Ray> Rays { get; } = new List<Ray>();

            /// <summary>The number of steps.</summary>
            public int Steps { get; set; }

            /// <summary>The time step.</summary>
            public double Dt { get; set; } = 1.0 / 50;

            /// <summary>The velocity iterations.</summary>
            public int Vel { get; set; } = 6 * 30;

            /// <summary>The position iterations.</summary>
            public int Pos { get; set; } = 2 * 30;

            /// <summary>The world constructor's <c>doSleep</c>, which pybox2d records without effect.</summary>
            public bool DoSleep { get; set; } = true;

            /// <summary>Whether bodies may sleep, set through <c>SetAllowSleeping</c> after the world is made.</summary>
            public bool AllowSleep { get; set; } = true;

            /// <summary>The gravity.</summary>
            public (double x, double y) Gravity { get; set; } = (0, -10);

            /// <summary>The number of bodies created so far; each step's trace row holds this many bodies.</summary>
            public int BodyCount { get; private set; }

            /// <summary>A scene with Gymnasium's Box2D settings (1/50 s, 180 velocity and 60 position iterations).</summary>
            /// <param name="steps">The number of steps.</param>
            /// <returns>The scene.</returns>
            public static Scene Lunar(int steps) => new Scene { Steps = steps };

            /// <summary>Adds a static body at the origin with one edge.</summary>
            /// <param name="x1">The first vertex's x.</param>
            /// <param name="y1">The first vertex's y.</param>
            /// <param name="x2">The second vertex's x.</param>
            /// <param name="y2">The second vertex's y.</param>
            /// <returns>The body's index.</returns>
            public int Ground(double x1, double y1, double x2, double y2) => Body("static", 0, 0, fixtures: Fixture.Edge((x1, y1), (x2, y2)));

            /// <summary>Adds a body with fixtures; its index is also its <c>userData</c> on both sides.</summary>
            /// <param name="type">static, kinematic or dynamic.</param>
            /// <param name="x">The position's x.</param>
            /// <param name="y">The position's y.</param>
            /// <param name="angle">The angle.</param>
            /// <param name="lv">The linear velocity.</param>
            /// <param name="av">The angular velocity.</param>
            /// <param name="bullet">Whether it is a bullet.</param>
            /// <param name="fixedRotation">Whether it may not rotate.</param>
            /// <param name="fixtures">The fixtures, created in order.</param>
            /// <returns>The body's index.</returns>
            public int Body(string type, double x, double y, double angle = 0, (double x, double y) lv = default, double av = 0, bool bullet = false, bool fixedRotation = false, params Fixture[] fixtures) {
                Ops.Add(new Dictionary<string, object?> {
                    ["op"] = "body", ["type"] = type, ["pos"] = new[] { x, y }, ["angle"] = angle, ["lv"] = new[] { lv.x, lv.y }, ["av"] = av,
                    ["bullet"] = bullet, ["fixedRotation"] = fixedRotation, ["fixtures"] = fixtures.Select(f => f.Json).ToArray(),
                });
                return BodyCount++;
            }

            /// <summary>Adds a revolute joint.</summary>
            /// <param name="a">Body A's index.</param>
            /// <param name="b">Body B's index.</param>
            /// <param name="anchorA">The local anchor on A.</param>
            /// <param name="anchorB">The local anchor on B.</param>
            /// <param name="motor">Whether the motor is enabled.</param>
            /// <param name="speed">The motor speed.</param>
            /// <param name="torque">The maximum motor torque.</param>
            /// <param name="limit">Whether the limit is enabled.</param>
            /// <param name="lower">The lower angle.</param>
            /// <param name="upper">The upper angle.</param>
            /// <param name="reference">The reference angle, or null to take it from the bodies (pybox2d's default).</param>
            /// <param name="collide">Whether the bodies collide.</param>
            /// <returns>The joint's index.</returns>
            public int Joint(int a, int b, (double x, double y) anchorA, (double x, double y) anchorB, bool motor = false, double speed = 0, double torque = 0,
                             bool limit = false, double lower = 0, double upper = 0, double? reference = null, bool collide = false) {
                Ops.Add(new Dictionary<string, object?> {
                    ["op"] = "joint", ["a"] = a, ["b"] = b, ["anchorA"] = new[] { anchorA.x, anchorA.y }, ["anchorB"] = new[] { anchorB.x, anchorB.y },
                    ["motor"] = motor, ["speed"] = speed, ["torque"] = torque, ["limit"] = limit, ["lower"] = lower, ["upper"] = upper, ["ref"] = reference,
                    ["collide"] = collide,
                });
                return _joints++;
            }

            /// <summary>Applies a force to a body's center before the first step (waking it).</summary>
            /// <param name="body">The body.</param>
            /// <param name="fx">The force's x.</param>
            /// <param name="fy">The force's y.</param>
            public void Force(int body, double fx, double fy) => Ops.Add(new Dictionary<string, object?> { ["op"] = "force", ["body"] = body, ["f"] = new[] { fx, fy } });

            /// <summary>Applies an impulse at a world point before the first step (waking the body).</summary>
            /// <param name="body">The body.</param>
            /// <param name="impulse">The impulse.</param>
            /// <param name="point">The world point.</param>
            public void Impulse(int body, (double x, double y) impulse, (double x, double y) point) =>
                Ops.Add(new Dictionary<string, object?> { ["op"] = "impulse", ["body"] = body, ["imp"] = new[] { impulse.x, impulse.y }, ["point"] = new[] { point.x, point.y } });

            /// <summary>Applies a torque before the first step (waking the body).</summary>
            /// <param name="body">The body.</param>
            /// <param name="torque">The torque.</param>
            public void Torque(int body, double torque) => Ops.Add(new Dictionary<string, object?> { ["op"] = "torque", ["body"] = body, ["t"] = torque });

            /// <summary>Sets a joint's motor speed and maximum torque before every step, cycling through the given values.</summary>
            /// <param name="joint">The joint.</param>
            /// <param name="speeds">The speeds.</param>
            /// <param name="torques">The torques.</param>
            public void Schedule(int joint, double[] speeds, double[] torques) =>
                MotorSchedule.Add(new Dictionary<string, object> { ["joint"] = joint, ["speeds"] = speeds, ["torques"] = torques });

            /// <summary>The scene as JSON.</summary>
            /// <returns>The JSON text.</returns>
            public string ToJson() {
                var root = new Dictionary<string, object?> {
                    ["gravity"] = new[] { Gravity.x, Gravity.y }, ["doSleep"] = DoSleep, ["allowSleep"] = AllowSleep, ["dt"] = Dt, ["vel"] = Vel, ["pos"] = Pos,
                    ["steps"] = Steps,
                    ["ops"] = Ops, ["schedule"] = MotorSchedule,
                    ["rays"] = Rays.Select(r => new Dictionary<string, object> { ["mode"] = r.Mode, ["p1"] = new[] { r.P1.x, r.P1.y }, ["p2"] = new[] { r.P2.x, r.P2.y } }).ToArray(),
                };
                return JsonSerializer.Serialize(root);
            }
        }

        /// <summary>Builds and runs a scene on Gym.Box2D through its pybox2d-shaped surface, as the Python snippet does on pybox2d.</summary>
        internal sealed class SceneRunner {
            /// <summary>The scene's JSON, parsed back so both sides read the same doubles.</summary>
            private readonly JsonElement _desc;

            /// <summary>The world.</summary>
            private readonly b2World _world;

            /// <summary>The bodies, in creation order.</summary>
            private readonly List<b2Body> _bodies = new List<b2Body>();

            /// <summary>The joints, in creation order.</summary>
            private readonly List<b2RevoluteJoint> _joints = new List<b2RevoluteJoint>();

            /// <summary>The contact events: step, 1 begin / 2 end, userData A, userData B.</summary>
            private readonly List<long> _events = new List<long>();

            /// <summary>The step being run, for the contact events.</summary>
            private int _step;

            /// <summary>Builds the scene's world: the listener first, then every command in order.</summary>
            /// <param name="scene">The scene.</param>
            public SceneRunner(Scene scene) {
                using JsonDocument document = JsonDocument.Parse(scene.ToJson());
                // Clone: the element must outlive the document, which owns pooled buffers.
                _desc = document.RootElement.Clone();
                _world = new b2World(V(_desc.GetProperty("gravity")), _desc.GetProperty("doSleep").GetBoolean());
                _world.SetAllowSleeping(_desc.GetProperty("allowSleep").GetBoolean());
                _world.ContactListener = new Recorder(this);
                foreach (JsonElement op in _desc.GetProperty("ops").EnumerateArray())
                    Apply(op);
            }

            /// <summary>The contact events so far, flattened four values per event.</summary>
            /// <returns>A copy of the events.</returns>
            public long[] FlatEvents() => _events.ToArray();

            /// <summary>Steps the scene and records every body after every step.</summary>
            /// <returns>The float32 trace.</returns>
            public float[] Run() {
                var trace = new List<float>();
                int steps = _desc.GetProperty("steps").GetInt32();
                for (int i = 0; i < steps; i++) {
                    StepOnce(i);
                    foreach (b2Body b in _bodies) {
                        b2Vec2 p = b.Position;
                        b2Vec2 v = b.LinearVelocity;
                        trace.AddRange(new[] { p.x, p.y, b.Angle, v.x, v.y, b.AngularVelocity, b.Awake ? 1.0f : 0.0f });
                    }
                }

                return trace.ToArray();
            }

            /// <summary>Steps the scene, then casts its rays and records every callback.</summary>
            /// <returns>The hits, seven floats each.</returns>
            public float[] RunRays() {
                int steps = _desc.GetProperty("steps").GetInt32();
                for (int i = 0; i < steps; i++)
                    StepOnce(i);
                var hits = new List<float>();
                int k = 0;
                foreach (JsonElement ray in _desc.GetProperty("rays").EnumerateArray())
                    _world.RayCast(new RayRecorder(k++, ray.GetProperty("mode").GetString()!, hits), V(ray.GetProperty("p1")), V(ray.GetProperty("p2")));
                return hits.ToArray();
            }

            /// <summary>Applies the motor schedule, then steps once.</summary>
            /// <param name="i">The step.</param>
            private void StepOnce(int i) {
                _step = i;
                foreach (JsonElement m in _desc.GetProperty("schedule").EnumerateArray()) {
                    b2RevoluteJoint joint = _joints[m.GetProperty("joint").GetInt32()];
                    JsonElement speeds = m.GetProperty("speeds"), torques = m.GetProperty("torques");
                    // pybox2d's property setters convert the Python floats to float32, as the casts do here.
                    joint.MotorSpeed = (float)speeds[i % speeds.GetArrayLength()].GetDouble();
                    joint.MaxMotorTorque = (float)torques[i % torques.GetArrayLength()].GetDouble();
                }

                _world.Step(_desc.GetProperty("dt").GetDouble(), _desc.GetProperty("vel").GetInt32(), _desc.GetProperty("pos").GetInt32());
            }

            /// <summary>Applies one command.</summary>
            /// <param name="op">The command.</param>
            /// <exception cref="InvalidOperationException">Thrown for a command the runner does not know.</exception>
            private void Apply(JsonElement op) {
                switch (op.GetProperty("op").GetString()) {
                    case "body": {
                        b2BodyType type = op.GetProperty("type").GetString() switch {
                            "static" => b2BodyType.b2_staticBody,
                            "kinematic" => b2BodyType.b2_kinematicBody,
                            _ => b2BodyType.b2_dynamicBody,
                        };
                        b2FixtureDef[] fixtures = op.GetProperty("fixtures").EnumerateArray().Select(MakeFixture).ToArray();
                        var def = new b2BodyDef(type, V(op.GetProperty("pos")), op.GetProperty("angle").GetDouble(), fixtures, linearVelocity: V(op.GetProperty("lv")),
                                                angularVelocity: op.GetProperty("av").GetDouble(), bullet: op.GetProperty("bullet").GetBoolean(),
                                                fixedRotation: op.GetProperty("fixedRotation").GetBoolean());
                        b2Body body = _world.CreateBody(def);
                        body.UserData = _bodies.Count;
                        _bodies.Add(body);
                        break;
                    }

                    case "joint": {
                        JsonElement reference = op.GetProperty("ref");
                        var def = new b2RevoluteJointDef(bodyA: _bodies[op.GetProperty("a").GetInt32()], bodyB: _bodies[op.GetProperty("b").GetInt32()],
                                                         localAnchorA: V(op.GetProperty("anchorA")), localAnchorB: V(op.GetProperty("anchorB")),
                                                         referenceAngle: reference.ValueKind == JsonValueKind.Null ? null : reference.GetDouble(),
                                                         enableLimit: op.GetProperty("limit").GetBoolean(), lowerAngle: op.GetProperty("lower").GetDouble(),
                                                         upperAngle: op.GetProperty("upper").GetDouble(), enableMotor: op.GetProperty("motor").GetBoolean(),
                                                         motorSpeed: op.GetProperty("speed").GetDouble(), maxMotorTorque: op.GetProperty("torque").GetDouble(),
                                                         collideConnected: op.GetProperty("collide").GetBoolean());
                        _joints.Add((b2RevoluteJoint)_world.CreateJoint(def));
                        break;
                    }

                    case "force":
                        _bodies[op.GetProperty("body").GetInt32()].ApplyForceToCenter(V(op.GetProperty("f")), true);
                        break;

                    case "impulse":
                        _bodies[op.GetProperty("body").GetInt32()].ApplyLinearImpulse(V(op.GetProperty("imp")), V(op.GetProperty("point")), true);
                        break;

                    case "torque":
                        _bodies[op.GetProperty("body").GetInt32()].ApplyTorque(op.GetProperty("t").GetDouble(), true);
                        break;

                    default:
                        throw new InvalidOperationException($"unknown scene command {op.GetProperty("op").GetString()}");
                }
            }

            /// <summary>Builds a fixture definition from its JSON.</summary>
            /// <param name="f">The fixture's JSON.</param>
            /// <returns>The definition.</returns>
            private static b2FixtureDef MakeFixture(JsonElement f) {
                b2Shape shape = f.GetProperty("shape").GetString() switch {
                    "polygon" => new b2PolygonShape(f.GetProperty("vertices").EnumerateArray().Select(V).ToArray()),
                    "box" => new b2PolygonShape(null, (f.GetProperty("hx").GetDouble(), f.GetProperty("hy").GetDouble())),
                    "circle" => new b2CircleShape(f.GetProperty("r").GetDouble(), V(f.GetProperty("pos"))),
                    _ => new b2EdgeShape(f.GetProperty("v").EnumerateArray().Select(v => (b2Vec2?)V(v)).ToArray()),
                };
                return new b2FixtureDef(shape, f.GetProperty("density").GetDouble(), f.GetProperty("friction").GetDouble(), f.GetProperty("restitution").GetDouble(),
                                        f.GetProperty("sensor").GetBoolean(), (ushort)f.GetProperty("category").GetInt32(), (ushort)f.GetProperty("mask").GetInt32(),
                                        (short)f.GetProperty("group").GetInt32());
            }

            /// <summary>A JSON pair as a vector, through pybox2d's tuple conversion (each coordinate rounded to float32).</summary>
            /// <param name="pair">The pair.</param>
            /// <returns>The vector.</returns>
            private static b2Vec2 V(JsonElement pair) => (pair[0].GetDouble(), pair[1].GetDouble());

            /// <summary>Records begin and end events with the bodies' scene indices.</summary>
            private sealed class Recorder : b2ContactListener {
                /// <summary>The runner whose events and step are recorded.</summary>
                private readonly SceneRunner _runner;

                /// <summary>A recorder for a runner.</summary>
                /// <param name="runner">The runner.</param>
                public Recorder(SceneRunner runner) => _runner = runner;

                /// <summary>Records a begin event.</summary>
                /// <param name="contact">The contact.</param>
                public override void BeginContact(b2Contact contact) => Record(1, contact);

                /// <summary>Records an end event.</summary>
                /// <param name="contact">The contact.</param>
                public override void EndContact(b2Contact contact) => Record(2, contact);

                /// <summary>Appends one event.</summary>
                /// <param name="kind">1 begin, 2 end.</param>
                /// <param name="contact">The contact.</param>
                private void Record(int kind, b2Contact contact) =>
                    _runner._events.AddRange(new long[] { _runner._step, kind, (int)contact.FixtureA.Body.UserData!, (int)contact.FixtureB.Body.UserData! });
            }

            /// <summary>Records every reported hit and answers per its mode, as the Python callback does.</summary>
            private sealed class RayRecorder : b2RayCastCallback {
                /// <summary>The ray's index.</summary>
                private readonly int _ray;

                /// <summary>The answer mode.</summary>
                private readonly string _mode;

                /// <summary>The hits of every ray.</summary>
                private readonly List<float> _hits;

                /// <summary>A recorder for one ray.</summary>
                /// <param name="ray">The ray's index.</param>
                /// <param name="mode">The answer mode.</param>
                /// <param name="hits">The hit list to append to.</param>
                public RayRecorder(int ray, string mode, List<float> hits) {
                    _ray = ray;
                    _mode = mode;
                    _hits = hits;
                }

                /// <summary>Records the hit and answers per the mode.</summary>
                /// <param name="fixture">The fixture hit.</param>
                /// <param name="point">The point of intersection.</param>
                /// <param name="normal">The normal there.</param>
                /// <param name="fraction">The fraction along the ray.</param>
                /// <returns>The fraction (clip), 1 (all), 0 (stop), or -1 for a fixture outside category 1 (filter).</returns>
                public override float ReportFixture(b2Fixture fixture, b2Vec2 point, b2Vec2 normal, float fraction) {
                    _hits.AddRange(new[] { _ray, (float)(int)fixture.Body.UserData!, point.x, point.y, normal.x, normal.y, fraction });
                    return _mode switch {
                        "closest" => fraction,
                        "all" => 1.0f,
                        "first" => 0.0f,
                        _ => (fixture.FilterData.categoryBits & 1) == 0 ? -1.0f : fraction,
                    };
                }
            }
        }
    }
}
