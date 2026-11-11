#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Box2D;
using Gym.Rendering;
using Color = Gym.Rendering.Color;

namespace Gym.Environments.Envs.Box2D {
    /// <summary>
    ///     A skid mark (Gymnasium's <c>car_dynamics.Particle</c>): the polyline a skidding wheel leaves, black on the road and
    ///     mud-coloured on grass, extended point by point while the wheel keeps skidding on the same ground (up to 30 points).
    /// </summary>
    /// <remarks>
    ///     A mutable record of plain values, as Python's slotted dataclass is: <see cref="Car"/> appends to <see cref="Poly"/>
    ///     while the skid lasts and keeps at most 30 marks. Reference:
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/car_dynamics.py#L57">gymnasium/envs/box2d/car_dynamics.py</see>.
    /// </remarks>
    public sealed class Particle {
        /// <summary>A mark of a colour and starting polyline.</summary>
        /// <param name="color">The colour: <see cref="Car.WheelColor"/> on the road, <see cref="Car.MudColor"/> on grass.</param>
        /// <param name="poly">The world points, in metres; kept, not copied (Python's list is the same object).</param>
        /// <param name="grass">Whether the mark is on grass; a wheel only extends a mark on the same ground.</param>
        /// <param name="ttl">Python's <c>ttl</c>, 1 by default and never read by Gymnasium.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="poly"/> is <see langword="null"/>.</exception>
        public Particle(Color color, List<(double X, double Y)> poly, bool grass, int ttl = 1) {
            Color = color;
            Poly = poly ?? throw new ArgumentNullException(nameof(poly));
            Grass = grass;
            Ttl = ttl;
        }

        /// <summary>Python's <c>color</c>: the mark's colour.</summary>
        public Color Color { get; set; }

        /// <summary>Python's <c>poly</c>: the polyline's world points in metres, which a skidding wheel appends to.</summary>
        public List<(double X, double Y)> Poly { get; set; }

        /// <summary>Python's <c>grass</c>: whether the mark lies on grass.</summary>
        public bool Grass { get; set; }

        /// <summary>Python's <c>ttl</c>: 1, unused by Gymnasium's code.</summary>
        public int Ttl { get; set; }
    }

    /// <summary>
    ///     A wheel of <see cref="Car"/>: its Box2D body with the attributes Gymnasium hangs on it (<c>wheel_rad</c>, <c>gas</c>,
    ///     <c>brake</c>, <c>steer</c>, <c>phase</c>, <c>omega</c>, the skid state, <c>joint</c>, <c>tiles</c>). The body's user
    ///     data is this object, as Python's <c>w.userData = w</c> makes the contact listener find the wheel.
    /// </summary>
    public sealed class CarWheel {
        /// <summary>Creates the wheel's attributes over its body and joint.</summary>
        /// <param name="body">The wheel's body.</param>
        /// <param name="wheelRad">The radius in metres.</param>
        internal CarWheel(b2Body body, double wheelRad) {
            Body = body;
            WheelRad = wheelRad;
            body.UserData = this;
        }

        /// <summary>The wheel's Box2D body.</summary>
        public b2Body Body { get; }

        /// <summary>Python's <c>wheel_rad</c>: the radius in metres (0.54).</summary>
        public double WheelRad { get; }

        /// <summary>Python's <c>color</c>: <see cref="Car.WheelColor"/>, as fractions of 255 the draw scales (black).</summary>
        public (double R, double G, double B) Color { get; set; } = (0, 0, 0);

        /// <summary>Python's <c>gas</c>: the throttle, which <see cref="Car.Gas"/> raises by at most 0.1 per call (rear wheels only).</summary>
        public double Gas { get; set; }

        /// <summary>Python's <c>brake</c>: the brake; 0.9 and above locks the wheel.</summary>
        public double Brake { get; set; }

        /// <summary>Python's <c>steer</c>: the steering target the joint's motor turns toward (front wheels only).</summary>
        public double Steer { get; set; }

        /// <summary>Python's <c>phase</c>: the wheel's rotation angle, which places its white stripe.</summary>
        public double Phase { get; set; }

        /// <summary>Python's <c>omega</c>: the wheel's angular velocity.</summary>
        public double Omega { get; set; }

        /// <summary>
        ///     Whether a skid has started (Python's <c>skid_start</c> is not <c>None</c>). Python keeps the body's position
        ///     there, which pybox2d returns as a live view, so the point it later reads is the wheel's position then, not
        ///     when the skid started; this flag stands for it.
        /// </summary>
        public bool SkidStarted { get; set; }

        /// <summary>Python's <c>skid_particle</c>: the mark this wheel extends, or <see langword="null"/>.</summary>
        public Particle? SkidParticle { get; set; }

        /// <summary>Python's <c>joint</c>: the revolute joint to the hull, whose motor steers the wheel.</summary>
        public b2RevoluteJoint Joint { get; internal set; } = null!;

        /// <summary>Python's <c>tiles</c>: the road tiles the wheel touches, kept by the contact listener.</summary>
        public HashSet<CarRacingEnv.RoadTile> Tiles { get; } = new();
    }

    /// <summary>
    ///     Gymnasium's <c>car_dynamics.Car</c>: a top-down car of a four-polygon hull and four wheels on limited revolute
    ///     joints, rear-wheel drive. Each <see cref="Step"/> turns the front wheels toward their steering target, spins the
    ///     wheels from gas and brakes, and applies each wheel's friction force (more on road tiles than on grass), leaving
    ///     skid marks when a wheel slides.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The arithmetic is Gymnasium's: float64 Python arithmetic over the bodies' float32 state, forces rounded to float32
    ///     where pybox2d converts them. <see cref="Draw"/> renders onto a pygame-compatible <see cref="Surface"/> through
    ///     <see cref="Vector2.RotateRad"/> and <see cref="Gym.Rendering.Draw"/>, as Gymnasium's draws through pygame.
    ///     </para>
    ///     <para>
    ///     Not thread-safe; it changes its world's bodies. Reference:
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/car_dynamics.py#L65">gymnasium/envs/box2d/car_dynamics.py</see>.
    ///     </para>
    /// </remarks>
    public sealed class Car {
        /// <summary>Gymnasium's <c>SIZE</c>: metres per car-geometry unit.</summary>
        public const double Size = 0.02;

        /// <summary>Gymnasium's <c>ENGINE_POWER</c>: the drive's power factor.</summary>
        public const double EnginePower = 100000000 * Size * Size;

        /// <summary>Gymnasium's <c>WHEEL_MOMENT_OF_INERTIA</c>.</summary>
        public const double WheelMomentOfInertia = 4000 * Size * Size;

        /// <summary>Gymnasium's <c>FRICTION_LIMIT</c>: a wheel's friction force cap on road (0.6 of it on grass).</summary>
        public const double FrictionLimit = 1000000 * Size * Size;

        /// <summary>Gymnasium's <c>WHEEL_R</c>: a wheel's half-length in car units.</summary>
        public const int WheelR = 27;

        /// <summary>Gymnasium's <c>WHEEL_W</c>: a wheel's half-width in car units.</summary>
        public const int WheelW = 14;

        /// <summary>Gymnasium's <c>WHEELPOS</c>: the wheels' anchors in car units: front left, front right, rear left, rear right.</summary>
        public static IReadOnlyList<(int X, int Y)> WheelPos { get; } = new[] { (-55, +80), (+55, +80), (-55, -82), (+55, -82) };

        /// <summary>Gymnasium's <c>HULL_POLY1</c>: the front spoiler.</summary>
        public static IReadOnlyList<(int X, int Y)> HullPoly1 { get; } = new[] { (-60, +130), (+60, +130), (+60, +110), (-60, +110) };

        /// <summary>Gymnasium's <c>HULL_POLY2</c>: the nose.</summary>
        public static IReadOnlyList<(int X, int Y)> HullPoly2 { get; } = new[] { (-15, +120), (+15, +120), (+20, +20), (-20, 20) };

        /// <summary>Gymnasium's <c>HULL_POLY3</c>: the body.</summary>
        public static IReadOnlyList<(int X, int Y)> HullPoly3 { get; } =
            new[] { (+25, +20), (+50, -10), (+50, -40), (+20, -90), (-20, -90), (-50, -40), (-50, -10), (-25, +20) };

        /// <summary>Gymnasium's <c>HULL_POLY4</c>: the rear spoiler.</summary>
        public static IReadOnlyList<(int X, int Y)> HullPoly4 { get; } = new[] { (-50, -120), (+50, -120), (+50, -90), (-50, -90) };

        /// <summary>Gymnasium's <c>WHEEL_COLOR</c>: the wheels' and the road marks' colour.</summary>
        public static readonly Color WheelColor = new(0, 0, 0);

        /// <summary>Gymnasium's <c>WHEEL_WHITE</c>: the wheels' stripe.</summary>
        public static readonly Color WheelWhite = new(77, 77, 77);

        /// <summary>Gymnasium's <c>MUD_COLOR</c>: the marks left on grass.</summary>
        public static readonly Color MudColor = new(102, 102, 0);

        /// <summary>The marks, oldest first; a 31st drops the oldest (Python's <c>deque(maxlen=30)</c>).</summary>
        private readonly LinkedList<Particle> _particles = new();

        /// <summary>The wheels, as created.</summary>
        private List<CarWheel> _wheels = new();

        /// <summary>
        ///     Builds the car in a world (Gymnasium's <c>Car(world, init_angle, init_x, init_y)</c>): the hull's four polygons
        ///     (density 1), then each wheel (density 0.1, colliding only with category 1) on a motorized revolute joint limited
        ///     to ±0.4 rad.
        /// </summary>
        /// <param name="world">The world, which keeps the bodies.</param>
        /// <param name="initAngle">The hull's and wheels' angle in radians.</param>
        /// <param name="initX">The hull's x in metres.</param>
        /// <param name="initY">The hull's y in metres.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="world"/> is <see langword="null"/>.</exception>
        /// <exception cref="NumSharp.AssertionError">Thrown when the world is stepping.</exception>
        public Car(b2World world, double initAngle, double initX, double initY) {
            World = world ?? throw new ArgumentNullException(nameof(world));
            static b2FixtureDef HullFixture(IReadOnlyList<(int X, int Y)> poly) {
                var vertices = new b2Vec2[poly.Count];
                for (int i = 0; i < vertices.Length; i++)
                    vertices[i] = (poly[i].X * Size, poly[i].Y * Size);
                return new b2FixtureDef(new b2PolygonShape(vertices), density: 1.0);
            }

            Hull = World.CreateDynamicBody(position: (initX, initY), angle: initAngle,
                                           fixtures: new[] { HullFixture(HullPoly1), HullFixture(HullPoly2), HullFixture(HullPoly3), HullFixture(HullPoly4) });
            HullColor = (0.8, 0.0, 0.0);
            FuelSpent = 0.0;
            (int X, int Y)[] wheelPoly = { (-WheelW, +WheelR), (+WheelW, +WheelR), (+WheelW, -WheelR), (-WheelW, -WheelR) };
            foreach ((int wx, int wy) in WheelPos) {
                const double frontK = 1.0;
                var vertices = new b2Vec2[wheelPoly.Length];
                for (int i = 0; i < vertices.Length; i++)
                    vertices[i] = (wheelPoly[i].X * frontK * Size, wheelPoly[i].Y * frontK * Size);
                b2Body body = World.CreateDynamicBody(new b2FixtureDef(new b2PolygonShape(vertices), density: 0.1, categoryBits: 0x0020, maskBits: 0x001,
                                                                       restitution: 0.0),
                                                      position: (initX + wx * Size, initY + wy * Size), angle: initAngle);
                var wheel = new CarWheel(body, frontK * WheelR * Size);
                var rjd = new b2RevoluteJointDef(bodyA: Hull, bodyB: body, localAnchorA: (wx * Size, wy * Size), localAnchorB: (0, 0), enableMotor: true,
                                                 enableLimit: true, maxMotorTorque: 180 * 900 * Size * Size, motorSpeed: 0, lowerAngle: -0.4, upperAngle: +0.4);
                wheel.Joint = (b2RevoluteJoint)World.CreateJoint(rjd);
                _wheels.Add(wheel);
            }
        }

        /// <summary>Python's <c>world</c>: the world the car's bodies live in.</summary>
        public b2World World { get; }

        /// <summary>Python's <c>hull</c>: the car's body, <see langword="null"/> after <see cref="Destroy"/>.</summary>
        public b2Body? Hull { get; private set; }

        /// <summary>Python's <c>hull.color</c>: the hull's colour as fractions of 255 (dark red).</summary>
        public (double R, double G, double B) HullColor { get; set; }

        /// <summary>Python's <c>wheels</c>: front left, front right, rear left, rear right; empty after <see cref="Destroy"/>.</summary>
        public IReadOnlyList<CarWheel> Wheels => _wheels;

        /// <summary>Python's <c>fuel_spent</c>: the engine work since CarRacing last cleared it (every step).</summary>
        public double FuelSpent { get; set; }

        /// <summary>Python's <c>drawlist</c>: the wheels' bodies, then the hull's, in drawing order.</summary>
        public IReadOnlyList<b2Body> Drawlist => _wheels.Select(w => w.Body).Append(Hull!).ToList();

        /// <summary>Python's <c>particles</c>: the skid marks, oldest first, at most 30; a new list on each read.</summary>
        public IReadOnlyList<Particle> Particles => _particles.ToList();

        /// <summary>
        ///     Gymnasium's <c>gas(gas)</c>: rear-wheel drive. The throttle is clipped to [0, 1]; each rear wheel's gas rises
        ///     toward it by at most 0.1 per call and drops at once (NaN propagates, as NumPy's clip keeps it).
        /// </summary>
        /// <param name="gas">The throttle.</param>
        public void Gas(double gas) {
            gas = double.IsNaN(gas) ? gas : Math.Min(Math.Max(gas, 0.0), 1.0);
            foreach (CarWheel w in _wheels.Skip(2).Take(2)) {
                double diff = gas - w.Gas;
                if (diff > 0.1)
                    diff = 0.1; // gradually increase, but stop immediately
                w.Gas += diff;
            }
        }

        /// <summary>Gymnasium's <c>brake(b)</c>: every wheel's brake, unclipped; 0.9 and above locks the wheels.</summary>
        /// <param name="b">The brake, normally in [0, 1].</param>
        public void Brake(double b) {
            foreach (CarWheel w in _wheels)
                w.Brake = b;
        }

        /// <summary>Gymnasium's <c>steer(s)</c>: the front wheels' steering target; the joints' motors turn toward it over time.</summary>
        /// <param name="s">The target, -1 (left) to 1 (right) in Gymnasium's use.</param>
        public void Steer(double s) {
            _wheels[0].Steer = s;
            _wheels[1].Steer = s;
        }

        /// <summary>
        ///     Gymnasium's <c>step(dt)</c>: for each wheel, its steering motor's speed, the friction limit of the ground under
        ///     it, the spin from gas and brake, the slip force capped at the friction limit (a skid mark when it exceeds twice
        ///     the limit), the spin the friction takes back, and the force applied to the wheel's body. The world is not stepped.
        /// </summary>
        /// <param name="dt">The time step in seconds (1/50 in CarRacing).</param>
        public void Step(double dt) {
            foreach (CarWheel w in _wheels) {
                // Steer each wheel
                double toTarget = w.Steer - w.Joint.Angle;
                double dir = NpSign(toTarget);
                double val = Math.Abs(toTarget);
                // dir * min(50.0 * val, 3.0): Python's min keeps its first argument unless the second is smaller (NaN stays).
                double limited = 3.0 < 50.0 * val ? 3.0 : 50.0 * val;
                w.Joint.MotorSpeed = (float)(dir * limited);

                // Position => friction_limit
                bool grass = true;
                double frictionLimit = FrictionLimit * 0.6; // Grass friction if no tile
                foreach (CarRacingEnv.RoadTile tile in w.Tiles) {
                    double onTile = FrictionLimit * tile.RoadFriction;
                    frictionLimit = onTile > frictionLimit ? onTile : frictionLimit;
                    grass = false;
                }

                // Force
                b2Vec2 forw = w.Body.GetWorldVector((0, 1));
                b2Vec2 side = w.Body.GetWorldVector((1, 0));
                b2Vec2 v = w.Body.LinearVelocity;
                double vf = (double)forw.x * v.x + (double)forw.y * v.y; // forward speed
                double vs = (double)side.x * v.x + (double)side.y * v.y; // side speed

                // WHEEL_MOMENT_OF_INERTIA*np.square(w.omega)/2 = E -- energy
                // WHEEL_MOMENT_OF_INERTIA*w.omega * domega/dt = dE/dt = W -- power
                // domega = dt*W/WHEEL_MOMENT_OF_INERTIA/w.omega

                // add small coef not to divide by zero
                w.Omega += dt * EnginePower * w.Gas / WheelMomentOfInertia / (Math.Abs(w.Omega) + 5.0);
                FuelSpent += dt * EnginePower * w.Gas;

                if (w.Brake >= 0.9) {
                    w.Omega = 0;
                } else if (w.Brake > 0) {
                    const double brakeForce = 15; // radians per second
                    double brakeDir = -NpSign(w.Omega);
                    double brakeVal = brakeForce * w.Brake;
                    if (Math.Abs(brakeVal) > Math.Abs(w.Omega))
                        brakeVal = Math.Abs(w.Omega); // low speed => same as = 0
                    w.Omega += brakeDir * brakeVal;
                }

                w.Phase += w.Omega * dt;

                double vr = w.Omega * w.WheelRad; // rotating wheel speed
                double fForce = -vf + vr; // force direction is direction of speed difference
                double pForce = -vs;

                // Physically correct is to always apply friction_limit until speed is equal.
                // But dt is finite, that will lead to oscillations if difference is already near zero.

                // Random coefficient to cut oscillations in few steps (have no effect on friction_limit)
                fForce *= 205000 * Size * Size;
                pForce *= 205000 * Size * Size;
                double force = Math.Sqrt(fForce * fForce + pForce * pForce);

                // Skid trace
                if (Math.Abs(force) > 2.0 * frictionLimit) {
                    b2Vec2 position = w.Body.Position;
                    if (w.SkidParticle is { } particle && particle.Grass == grass && particle.Poly.Count < 30) {
                        particle.Poly.Add((position.x, position.y));
                    } else if (!w.SkidStarted) {
                        w.SkidStarted = true;
                    } else {
                        // Python's skid_start is the body's live position, so both points are the wheel's position now.
                        w.SkidParticle = CreateParticle((position.x, position.y), (position.x, position.y), grass);
                        w.SkidStarted = false;
                    }
                } else {
                    w.SkidStarted = false;
                    w.SkidParticle = null;
                }

                if (Math.Abs(force) > frictionLimit) {
                    fForce /= force;
                    pForce /= force;
                    force = frictionLimit; // Correct physics here
                    fForce *= force;
                    pForce *= force;
                }

                w.Omega -= dt * fForce * w.WheelRad / WheelMomentOfInertia;

                w.Body.ApplyForceToCenter((pForce * side.x + fForce * forw.x, pForce * side.y + fForce * forw.y), true);
            }
        }

        /// <summary>
        ///     Gymnasium's <c>draw(surface, zoom, translation, angle, draw_particles)</c>: the skid marks as 2-pixel polylines
        ///     (when <paramref name="drawParticles"/>), then each wheel and the hull fixture by fixture as filled polygons, and on
        ///     each wheel its white stripe where the wheel's phase shows it, all rotated by <paramref name="angle"/>, scaled by
        ///     <paramref name="zoom"/> and moved by <paramref name="translation"/> (in a y-up frame flipped later).
        /// </summary>
        /// <param name="surface">The surface to draw on.</param>
        /// <param name="zoom">Pixels per metre.</param>
        /// <param name="translation">The screen offset added after rotating and scaling.</param>
        /// <param name="angle">The rotation in radians (the negated hull angle, so the car points up).</param>
        /// <param name="drawParticles">Whether to draw the skid marks (CarRacing leaves them out of the observation).</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="surface"/> is <see langword="null"/>.</exception>
        /// <exception cref="NotSupportedException">Thrown for a surface with per-pixel alpha (not modelled).</exception>
        public void Draw(Surface surface, double zoom, (double X, double Y) translation, double angle, bool drawParticles = true) {
            if (surface is null)
                throw new ArgumentNullException(nameof(surface));
            (double X, double Y) ToScreen((double X, double Y) c) {
                Vector2 rotated = new Vector2(c.X, c.Y).RotateRad(angle);
                return (rotated.X * zoom + translation.X, rotated.Y * zoom + translation.Y);
            }

            if (drawParticles) {
                foreach (Particle p in _particles) {
                    var poly = new List<(double X, double Y)>(p.Poly.Count);
                    foreach ((double X, double Y) c in p.Poly)
                        poly.Add(ToScreen(c));
                    Gym.Rendering.Draw.Lines(surface, p.Color, false, poly, 2);
                }
            }

            var drawlist = new List<(b2Body Body, (double R, double G, double B) Color, CarWheel? Wheel)>();
            foreach (CarWheel wheel in _wheels)
                drawlist.Add((wheel.Body, wheel.Color, wheel));
            drawlist.Add((Hull!, HullColor, null));
            foreach ((b2Body body, (double R, double G, double B) colorFractions, CarWheel? wheel) in drawlist) {
                foreach (b2Fixture f in body.Fixtures) {
                    b2Transform trans = f.Body.Transform;
                    var path = new List<(double X, double Y)>();
                    foreach (b2Vec2 v in ((b2PolygonShape)f.Shape).Vertices) {
                        b2Vec2 world = trans * v;
                        path.Add(ToScreen((world.x, world.y)));
                    }

                    // [int(c * 255) for c in obj.color]: Python's int() truncates.
                    var color = new Color((byte)(int)(colorFractions.R * 255), (byte)(int)(colorFractions.G * 255), (byte)(int)(colorFractions.B * 255));
                    Gym.Rendering.Draw.Polygon(surface, color, path);

                    if (wheel is null)
                        continue;
                    double a1 = wheel.Phase;
                    double a2 = wheel.Phase + 1.2; // radians
                    double s1 = Math.Sin(a1);
                    double s2 = Math.Sin(a2);
                    double c1 = Math.Cos(a1);
                    double c2 = Math.Cos(a2);
                    if (s1 > 0 && s2 > 0)
                        continue;
                    if (s1 > 0)
                        c1 = NpSign(c1);
                    if (s2 > 0)
                        c2 = NpSign(c2);
                    (double X, double Y)[] whitePoly = {
                        (-WheelW * Size, +WheelR * c1 * Size), (+WheelW * Size, +WheelR * c1 * Size),
                        (+WheelW * Size, +WheelR * c2 * Size), (-WheelW * Size, +WheelR * c2 * Size),
                    };
                    var stripe = new List<(double X, double Y)>(whitePoly.Length);
                    foreach ((double X, double Y) local in whitePoly) {
                        // trans * v of a Python tuple: pybox2d converts it to a float32 vector first.
                        b2Vec2 world = trans * (b2Vec2)local;
                        stripe.Add(ToScreen((world.x, world.y)));
                    }

                    Gym.Rendering.Draw.Polygon(surface, WheelWhite, stripe);
                }
            }
        }

        /// <summary>Gymnasium's <c>destroy()</c>: destroys the hull, then each wheel (their joints go with the hull).</summary>
        /// <exception cref="NumSharp.AssertionError">Thrown when the world is stepping.</exception>
        public void Destroy() {
            World.DestroyBody(Hull!);
            Hull = null;
            foreach (CarWheel w in _wheels)
                World.DestroyBody(w.Body);
            _wheels = new List<CarWheel>();
        }

        /// <summary>Gymnasium's <c>_create_particle</c>: a new mark of two points, black on road, mud on grass, appended to the 30 kept.</summary>
        /// <param name="point1">The first point.</param>
        /// <param name="point2">The second point.</param>
        /// <param name="grass">Whether the wheel is on grass.</param>
        /// <returns>The mark, which the wheel extends while it skids.</returns>
        private Particle CreateParticle((double X, double Y) point1, (double X, double Y) point2, bool grass) {
            var p = new Particle(!grass ? WheelColor : MudColor, new List<(double X, double Y)> { point1, point2 }, grass);
            _particles.AddLast(p);
            // deque(maxlen=30): appending to a full deque drops the oldest.
            if (_particles.Count > 30)
                _particles.RemoveFirst();
            return p;
        }

        /// <summary><c>np.sign</c> of a float64: 1, -1, 0 for either zero, NaN for NaN.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The sign.</returns>
        private static double NpSign(double value) => value > 0 ? 1.0 : value < 0 ? -1.0 : value == 0 ? 0.0 : double.NaN;
    }
}
