#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.ExceptionServices;
using System.Threading;
using Gym.Box2D;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Box2D {
    /// <summary>
    ///     What the live parity cells against pybox2d cannot show about the Box2D engine (its scenes are compared bit for bit
    ///     in Gym.Tests.Parity's <c>Box2DEngineLiveParityTests</c>): worlds share no mutable state, so worlds stepped on
    ///     parallel threads, as a vector env's sub-envs are, give the bits one thread gives; the world's sleeping switches
    ///     behave as pybox2d's; and the keyword errors keep the original exception, as Python's chaining does.
    /// </summary>
    [TestClass]
    public class Box2DEngineTests {
        /// <summary>
        ///     Eight copies of a mixed scene (boxes, circles, polygons, bullets, motorized joints) stepped at once on eight
        ///     threads produce, float32 bit for bit, the trace one thread produces: no static scratch state, counters or
        ///     caches are shared between worlds.
        /// </summary>
        [TestMethod]
        public void Worlds_SteppedOnParallelThreads_MatchOneThread() {
            uint[] reference = RunMixedScene();
            const int threads = 8;
            var traces = new uint[threads][];
            var failures = new Exception?[threads];
            using var start = new Barrier(threads);
            Thread[] workers = Enumerable.Range(0, threads).Select(i => new Thread(() => {
                // An exception escaping a thread would end the test host; it is carried back to the test thread instead.
                try {
                    start.SignalAndWait();
                    traces[i] = RunMixedScene();
                } catch (Exception ex) {
                    failures[i] = ex;
                }
            })).ToArray();
            foreach (Thread worker in workers)
                worker.Start();
            foreach (Thread worker in workers)
                worker.Join();

            foreach (Exception? failure in failures)
                if (failure != null)
                    ExceptionDispatchInfo.Capture(failure).Throw();
            for (int i = 0; i < threads; i++)
                CollectionAssert.AreEqual(reference, traces[i], $"thread {i} diverged from the one-thread trace");
        }

        /// <summary>
        ///     <c>doSleep=False</c> is recorded and inert, as in pybox2d 2.3.10 (its constructor sets a plain attribute):
        ///     a resting box still falls asleep. <see cref="b2World.SetAllowSleeping"/> is the switch that works: turning it
        ///     off wakes the box at once and keeps it awake.
        /// </summary>
        [TestMethod]
        public void DoSleep_IsInert_AndSetAllowSleepingRules() {
            var world = new b2World(doSleep: false);
            Assert.IsFalse(world.DoSleep, "the constructor's value is recorded");
            Assert.IsTrue(world.GetAllowSleeping(), "but the engine's flag is untouched");
            world.CreateStaticBody(new b2EdgeShape(new b2Vec2?[] { (-5, 0), (5, 0) }));
            b2Body box = world.CreateDynamicBody(new b2FixtureDef(new b2PolygonShape(null, (0.5, 0.5)), density: 1.0), position: (0, 0.5));
            Step(world, 300);
            Assert.IsFalse(box.Awake, "doSleep=False must not stop the box sleeping, as in pybox2d");

            world.SetAllowSleeping(false);
            Assert.IsFalse(world.GetAllowSleeping());
            Assert.IsTrue(box.Awake, "forbidding sleep wakes every body");
            Step(world, 300);
            Assert.IsTrue(box.Awake, "with sleeping forbidden the box stays awake");
        }

        /// <summary>
        ///     A refused keyword argument raises the same exception type with pybox2d's prefix (<c>Failed on kwargs for
        ///     Class.key: </c> in the constructors, <c>Failed on kwargs, class="…" key="…": </c> in the fixture shortcuts) and
        ///     keeps the original as its inner exception, as Python's implicit chaining keeps <c>__context__</c>.
        /// </summary>
        [TestMethod]
        public void KeywordErrors_NameClassAndKeyword_AndChainTheOriginal() {
            var overflow = Assert.ThrowsExactly<OverflowError>(() => new b2FixtureDef(null, density: 1e39));
            Assert.AreEqual("Failed on kwargs for b2FixtureDef.density: in method 'b2FixtureDef_density_set', argument 2 of type 'float32'", overflow.Message);
            Assert.IsInstanceOfType<OverflowError>(overflow.InnerException);
            Assert.AreEqual("in method 'b2FixtureDef_density_set', argument 2 of type 'float32'", overflow.InnerException!.Message);

            var assertion = Assert.ThrowsExactly<AssertionError>(() => new b2PolygonShape(new b2Vec2[] { (0, 0), (1, 0) }));
            Assert.AreEqual("Failed on kwargs for b2PolygonShape.vertices: 3 <= count && count <= b2_maxPolygonVertices", assertion.Message);

            var world = new b2World();
            b2Body body = world.CreateDynamicBody(position: (0, 0));
            var value = Assert.ThrowsExactly<ValueError>(() => body.CreateEdgeFixture(new b2Vec2?[] { (0, 0) }));
            Assert.AreEqual("Failed on kwargs, class=\"b2EdgeShape\" key=\"vertices\": Expected from 2 to 4 vertices.", value.Message);
            Assert.AreEqual(0, body.Fixtures.Count, "a refused shortcut creates no fixture");
        }

        /// <summary>
        ///     Builds and steps a mixed scene from a fixed seed: an arena, 40 bodies of every shape (some bullets), 6
        ///     motorized and limited revolute joints, initial impulses; 200 steps at Gymnasium's settings.
        /// </summary>
        /// <returns>Every dynamic body's position, angle and velocities after every step, as float32 bits.</returns>
        private static uint[] RunMixedScene() {
            var rng = new Random(3);
            var world = new b2World();
            world.CreateStaticBody(new b2EdgeShape(new b2Vec2?[] { (-16, 0), (16, 0) }));
            world.CreateStaticBody(new b2EdgeShape(new b2Vec2?[] { (-16, 0), (-16, 20) }));
            world.CreateStaticBody(new b2EdgeShape(new b2Vec2?[] { (16, 0), (16, 20) }));
            var bodies = new List<b2Body>();
            for (int i = 0; i < 40; i++) {
                b2Shape shape = (i % 3) switch {
                    0 => new b2PolygonShape(null, (0.2 + rng.NextDouble() * 0.6, 0.2 + rng.NextDouble() * 0.6)),
                    1 => new b2CircleShape(0.2 + rng.NextDouble() * 0.5),
                    _ => new b2PolygonShape(new b2Vec2[] { (-0.5, -0.3), (0.6, -0.4), (0.4 * rng.NextDouble(), 0.7) }),
                };
                bodies.Add(world.CreateDynamicBody(fixtures: new[] { new b2FixtureDef(shape, density: 0.5 + rng.NextDouble() * 2, friction: rng.NextDouble(), restitution: rng.NextDouble() * 0.5) },
                                                   position: (rng.NextDouble() * 28 - 14, 1 + rng.NextDouble() * 16), angle: rng.NextDouble() * 6 - 3,
                                                   linearVelocity: (rng.NextDouble() * 8 - 4, rng.NextDouble() * 8 - 4), bullet: i % 7 == 0));
            }

            for (int j = 0; j < 12; j += 2)
                world.CreateJoint(new b2RevoluteJointDef(bodyA: bodies[j], bodyB: bodies[j + 1], localAnchorA: (0.3, 0), localAnchorB: (-0.3, 0),
                                                         enableMotor: true, motorSpeed: rng.NextDouble() * 4 - 2, maxMotorTorque: 40,
                                                         enableLimit: j % 4 == 0, lowerAngle: -0.6, upperAngle: 0.8));
            foreach (b2Body body in bodies.Where((_, k) => k % 5 == 0))
                body.ApplyLinearImpulse((rng.NextDouble() * 10 - 5, rng.NextDouble() * 10), body.WorldCenter, true);

            var trace = new List<uint>();
            for (int step = 0; step < 200; step++) {
                world.Step(1.0 / 50, 6 * 30, 2 * 30);
                foreach (b2Body body in bodies) {
                    b2Vec2 p = body.Position, v = body.LinearVelocity;
                    foreach (float value in new[] { p.x, p.y, body.Angle, v.x, v.y, body.AngularVelocity })
                        trace.Add(BitConverter.SingleToUInt32Bits(value));
                }
            }

            return trace.ToArray();
        }

        /// <summary>Steps a world at Gymnasium's settings (1/50 s, 180 velocity and 60 position iterations).</summary>
        /// <param name="world">The world.</param>
        /// <param name="steps">The number of steps.</param>
        private static void Step(b2World world, int steps) {
            for (int i = 0; i < steps; i++)
                world.Step(1.0 / 50, 6 * 30, 2 * 30);
        }
    }
}
