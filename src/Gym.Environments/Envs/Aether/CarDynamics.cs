using Gym.Collections;
using Gym.Envs;
using Gym.Observations;
using Gym.Spaces;
using NumSharp;
using SixLabors.ImageSharp;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using tainicom.Aether.Physics2D.Common;
using tainicom.Aether.Physics2D.Collision.Shapes;
using tainicom.Aether.Physics2D.Dynamics;
using tainicom.Aether.Physics2D.Dynamics.Contacts;
using tainicom.Aether.Physics2D.Dynamics.Joints;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Drawing;
using SixLabors.ImageSharp.Drawing.Processing;
using System.Linq;
using Gym.Threading;
using SixLabors.ImageSharp.ColorSpaces;
using Gym.Exceptions;
using System.Diagnostics;

namespace Gym.Environments.Envs.Aether
{
    #region Road Tiles
    /// <summary>
    /// Not used in the python Gym environment, added here for completeness.
    /// </summary>
    public class RoadTile
    {
        public float Friction { get; set; } = 0f;
        public bool RoadVisited { get; set; } = false;
        public Rgba32 Color { get; set; } = new Rgba32(102,102,102); // 0.4,0.4,0.4
        public int Index { get; set; } = 0;
        public Fixture PhysicsFixture { get; set; }
    }
    #endregion
    /// <summary>
    /// Top-down car dynamics simulation.
    /// Some ideas are taken from this great tutorial http://www.iforce2d.net/b2dtut/top-down-car by Chris Campbell.
    /// This simulation is a bit more detailed, with wheels rotation.
    /// Created by Oleg Klimov, converted to C# by Jacob Anderson
    /// </summary>
    public class CarDynamics
    {

        #region CONSTANTS
        private const float SIZE = 0.02f;
        private const float ENGINE_POWER = 100000000 * SIZE * SIZE;
        private const float WHEEL_MOMENT_OF_INERTIA = 4000 * SIZE * SIZE;
        private const float FRICTION_LIMIT = (
            1000000 * SIZE * SIZE
        );  // friction ~= mass ~= size^2 (calculated implicitly using density)
        private const float WHEEL_R = 27;
        private const float WHEEL_W = 14;
        #endregion

        #region Polygon Presets
        private static readonly Vertices WHEELPOS = new Vertices(new Vector2[] {
            new Vector2(-55f,80f), new Vector2(55f,80f), new Vector2(-55f,-82f), new Vector2(55f,-82f)
        });
        private static readonly Vertices HULL_POLY1 = new Vertices(new Vector2[] {
            new Vector2(-60f, +130f), new Vector2(+60f, +130f), new Vector2(+60f, +110f), new Vector2(-60f, +110f)
        });
        private static readonly Vertices HULL_POLY2 = new Vertices(new Vector2[] {
            new Vector2(-15f, +120f), new Vector2(+15f, +120f), new Vector2(+20f, +20f), new Vector2(-20f, 20f)
        });
        private static readonly Vertices HULL_POLY3 = new Vertices(new Vector2[] {
            new Vector2(+25f, +20f),
            new Vector2(+50f, -10f),
            new Vector2(+50f, -40f),
            new Vector2(+20f, -90f),
            new Vector2(-20f, -90f),
            new Vector2(-50f, -40f),
            new Vector2(-50f, -10f),
            new Vector2(-25f, +20f)
        });
        private static readonly Vertices HULL_POLY4 = new Vertices(new Vector2[] {
            new Vector2(-50f, -120f), new Vector2(+50f, -120f), new Vector2(+50f, -90f), new Vector2(-50f, -90f)
        });
        #endregion

        #region Visualization Customization
        public Rgba32 WheelColor { get; set; } = new Rgba32(0, 0, 0);
        public Rgba32 WheelWhite { get; set; } = new Rgba32(77, 77, 77);
        public Rgba32 MudColor { get; set; } = new Rgba32(102, 102, 0);
        public Rgba32 BodyColor { get; set; } = new Rgba32(204, 0, 0);
        #endregion

        private World World { get; set; }
        public Body Hull { get; private set; }
        public List<Wheel> Wheels { get; private set; }
        public float FuelSpent { get; set; } = 0f;

        #region Telemetry
        public float SteerAngle { get; private set; }
        public float Speed { get; private set; }
        #endregion

        #region Particles
        internal class CarParticle
        {
            internal Rgba32 RenderColor { get; set; }
            internal List<Vector2> Poly { get; set; }
            internal bool IsGrass { get; set; } = false;

            internal CarParticle(Vector2 point1, Vector2 point2, bool grass)
            {
                Poly = new List<Vector2>();
                Poly.Add(point1);
                Poly.Add(point2);
                IsGrass = grass;
            }
        }
        private List<CarParticle> Particles { get; set; } = new List<CarParticle>();
        #endregion

        #region Wheels
        public class Wheel
        {
            internal Body Unit { get; set; }
            internal float WheelRad { get; set; }
            internal float Gas { get; set; }
            internal float Brake { get; set; }
            internal float Steer { get; set; }
            /// <summary>
            /// Wheel angle
            /// </summary>
            internal float Phase { get; set; }
            /// <summary>
            /// Angular velocity
            /// </summary>
            internal float Omega { get; set; }
            internal Rgba32 Color { get; set; }
            internal Vector2? SkidStart { get; set; }
            internal CarParticle SkidParticle { get; set; }
            internal RevoluteJoint Joint { get; set; }
            public List<RoadTile> Tiles { get; set; }

            internal Wheel()
            {
            }
            internal Wheel CreateBody(Body car, Vector2 pos)
            {
                float front_k = (pos.Y > 0f ? 1f : 1f); // ??
                Vertices wheel_poly = new Vertices(new Vector2[] {
                    new Vector2(-WHEEL_W*front_k*SIZE,WHEEL_R*front_k*SIZE),
                    new Vector2(WHEEL_W*front_k*SIZE,WHEEL_R*front_k*SIZE),
                    new Vector2(WHEEL_W*front_k*SIZE,-WHEEL_R*front_k*SIZE),
                    new Vector2(-WHEEL_W*front_k*SIZE,-WHEEL_R*front_k*SIZE)
                });
                Vector2 p = car.Position + pos * SIZE;
                Unit = car.World.CreateBody(position: p, rotation: car.Rotation, bodyType: BodyType.Dynamic);
                Unit.Tag = this;
                Fixture f = Unit.CreateFixture(new PolygonShape(new Vertices(wheel_poly), 0.1f));
                f.CollisionCategories = Category.All;
                f.CollidesWith = Category.Cat1;
                f.Restitution = 0f;
                f.Tag = this;
                WheelRad = front_k * WHEEL_R * SIZE;
                Gas = 0f;
                Brake = 0f;
                Steer = 0f;
                Phase = 0f;
                Omega = 0f;
                SkidStart = null;
                SkidParticle = null;
                // The wheel revolution physics
                RevoluteJoint rjd = new RevoluteJoint(car, Unit, new Vector2(pos.X * SIZE, pos.Y * SIZE), new Vector2(0f, 0f));
                rjd.MotorEnabled = true;
                rjd.LimitEnabled = true;
                rjd.MaxMotorTorque = 180f * 900f * SIZE * SIZE;
                rjd.MotorSpeed = 0f;
                rjd.LowerLimit = -0.4f;
                rjd.UpperLimit = 0.4f;
                Joint = rjd;
                car.World.Add(rjd);
                Tiles = new List<RoadTile>();
                return (this);
            }
        }
        #endregion


        public CarDynamics(World world, float init_angle, float init_x, float init_y)
        {
            World = world;
            Wheels = new List<Wheel>();
            Hull = world.CreateBody(position: new Vector2(init_x, init_y), rotation: init_angle, bodyType: BodyType.Dynamic);
            Fixture f = Hull.CreateFixture(new PolygonShape(ScaleIt(HULL_POLY1), 1f));
            f = Hull.CreateFixture(new PolygonShape(ScaleIt(HULL_POLY2), 1f));
            f = Hull.CreateFixture(new PolygonShape(ScaleIt(HULL_POLY3), 1f));
            f = Hull.CreateFixture(new PolygonShape(ScaleIt(HULL_POLY4), 1f));
            foreach (Vector2 v in WHEELPOS)
            {
                Wheel w = new Wheel();
                w.CreateBody(Hull, v);
                Wheels.Add(w);
                w.Color = WheelColor;
            }
        }

        private Vertices ScaleIt(Vertices vx, float scale = SIZE)
        {
            Vertices n = new Vertices();
            foreach (Vector2 v in vx)
            {
                n.Add(new Vector2(v.X * scale, v.Y * scale));
            }
            return (n);
        }

        #region Car Interface
        /// <summary>
        /// Apply gas to rear 2 wheels
        /// </summary>
        /// <param name="gas">How much gas gets applied. Gets clipped between 0 and 1</param>
        /// <returns></returns>
        public float Gas(float gas)
        {
            gas = np.clip(gas, 0f, 1f);
            for(int i=2; i < Wheels.Count; i++) 
            {
                // Rear-wheel drive, positive traction
                Wheel w = Wheels[i];
                float diff = gas - w.Gas;
                if (diff > 0.1f)
                {
                    diff = 0.1f; // gradually increase, but stop immediately
                }
                w.Gas += diff;
            }
            return (gas);
        }
        /// <summary>
        /// Apply braking across all 4 wheels
        /// </summary>
        /// <param name="b">Degree to which the brakes are applied. More than 0.9 blocks the wheels to zero rotation, between 0 and 1.0</param>
        public void Brake(float b)
        {
            foreach (Wheel w in Wheels)
            {
                w.Brake = b;
            }
        }
        /// <summary>
        /// Assumes 2-wheel control, sign is direction, turning the front wheels of the car left or right.
        /// </summary>
        /// <param name="s">target position, it takes time to rotate steering wheel from side-to-side, range is -1 to 1</param>
        public void Steer(float s)
        {
            Wheels[0].Steer = s;
            Wheels[1].Steer = s;
        }

        /// <summary>
        /// Computes the updae on the given wheel and returns the fuel spent by the wheel
        /// </summary>
        /// <param name="dt">The time step</param>
        /// <param name="w">The wheel being simulated</param>
        /// <returns>The amount of fuel spent in the step</returns>
        private float Step(float dt, Wheel w)
        {
            float steer_angle = w.Steer - w.Joint.JointAngle;
            float dir = (steer_angle < 0f ? -1f : (steer_angle > 0f ? 1f : 0f));
            float val = Math.Abs(steer_angle);
            w.Joint.MotorSpeed = dir * Math.Min(50f * val, 3f);

            // Position -> friction_limit
            bool grass = true;
            float friction_limit = FRICTION_LIMIT * 0.6f; // Grass friction if no tile
            foreach (RoadTile tile in w.Tiles)
            {
                friction_limit = Math.Max(friction_limit, FRICTION_LIMIT * tile.Friction);
                grass = false;
            }

            // Force
            Vector2 forw = w.Unit.GetWorldVector(new Vector2(0f, 1f));
            Vector2 side = w.Unit.GetWorldVector(new Vector2(1f, 0f));
            Vector2 v = w.Unit.LinearVelocity;
            float vf = forw.X * v.X + forw.Y * v.Y;
            float vs = side.X * v.X + side.Y * v.Y;

            // # WHEEL_MOMENT_OF_INERTIA*np.square(w.omega)/2 = E -- energy
            // # WHEEL_MOMENT_OF_INERTIA*w.omega * domega/dt = dE/dt = W -- power
            // # domega = dt*W/WHEEL_MOMENT_OF_INERTIA/w.omega

            // # add small coef not to divide by zero
            w.Omega += dt * ENGINE_POWER * w.Gas / WHEEL_MOMENT_OF_INERTIA / (Math.Abs(w.Omega) + 5f);
            float fuel_spent = dt * ENGINE_POWER * w.Gas;
            if (w.Brake >= 0.9f)
            {
                w.Omega = 0f;
            }
            else if(w.Brake > 0f)
            {
                float brake_force = 15f; // Radians per second
                dir = (w.Omega < 0f ? 1f : -1f); // Oppposite sign
                val = brake_force * w.Brake;
                if (Math.Abs(val) > Math.Abs(w.Omega))
                {
                    val = Math.Abs(w.Omega); // low speed => same as = 0
                }
                w.Omega += dir * val;
            }
            w.Phase += w.Omega * dt;

            float vr = w.Omega * w.WheelRad; // rotating wheel speed
            float f_force = -vf + vr;// force direction is direction of speed difference
            float p_force = -vs;

            // Physically correct is to always apply friction_limit until speed is equal.
            // But dt is finite, that will lead to oscillations if difference is already near zero.

            // Random coefficient to cut oscillations in few steps (have no effect on friction_limit)
            f_force *= 205000f * SIZE * SIZE;
            p_force *= 205000f * SIZE * SIZE;
            float force = np.sqrt(f_force * f_force + p_force * p_force);

            // Skid trace
            if (Math.Abs(force) > 2f * friction_limit)
            {
                if (w.SkidParticle != null && w.SkidParticle.IsGrass && w.SkidParticle.Poly.Count < 30)
                {
                    w.SkidParticle.Poly.Add(w.Unit.Position);
                }
                else if (!w.SkidStart.HasValue)
                {
                    w.SkidStart = w.Unit.Position;
                }
                else
                {
                    w.SkidParticle = new CarParticle(w.SkidStart.Value, w.Unit.Position, grass);
                    Particles.Add(w.SkidParticle);
                    if (Particles.Count > 30)
                    {
                        Particles.RemoveAt(0);
                    }
                    w.SkidStart = null;
                }
            }
            else
            {
                w.SkidStart = null;
                w.SkidParticle = null;
            }
            if (Math.Abs(force) > friction_limit)
            {
                f_force /= force;
                p_force /= force;
                force = friction_limit; // Correct physics here
                f_force *= force;
                p_force *= force;
            }

            w.Omega -= dt * f_force * w.WheelRad / WHEEL_MOMENT_OF_INERTIA;

            w.Unit.ApplyForce(new Vector2(p_force * side.X + f_force * forw.X, p_force * side.Y + f_force * forw.Y));

            return (fuel_spent);

        }

        public CarDynamics Step(float dt)
        {
            foreach (Wheel w in Wheels)
            {
                FuelSpent += Step(dt, w);
            }
            SteerAngle = Hull.Rotation;
            Speed = Hull.LinearVelocity.Length();
            return (this);
        }

        public static Vector2 RotateVec(Vector2 v, float angle) {
            float cos = (float)Math.Cos(angle);
            float sin = (float)Math.Sin(angle);
            return new Vector2(v.X * cos - v.Y * sin, v.X * sin + v.Y * cos); 
        }
        public void Draw(Image<Rgba32> img, float zoom, Vector2 translation, float angle, bool draw_particles = true)
        {
            if (draw_particles)
            {
                foreach (CarParticle p in Particles)
                {
                    PointF[] poly = new PointF[p.Poly.Count];
                    
                    for (int i = 0; i < p.Poly.Count; i++)
                    {
                        Vector2 px = RotateVec(p.Poly[i], angle) * zoom + translation;
                        poly[i] = new PointF(px.X, px.Y);
                    }
                    img.Mutate(i => i.DrawLines(p.RenderColor, 2f, poly));
                }
            }
            // Draw the Hull
            foreach (Fixture f in Hull.FixtureList)
            {
                Transform trans = f.Body.GetTransform();
                Vertices polyVerts = ((PolygonShape)f.Shape).Vertices;
                PointF[] path = new PointF[polyVerts.Count];
                for (int i = 0; i < polyVerts.Count; i++)
                {
                    Vector2 px = RotateVec(Transform.Multiply(polyVerts[i], ref trans), angle) * zoom + translation;
                    path[i] = new PointF(px.X, px.Y);
                }
                img.Mutate(i => i.FillPolygon(BodyColor, path));
            }

            foreach (Wheel w in Wheels)
            {
                Transform trans = w.Unit.GetTransform();
                PointF[] path = null;
                foreach (Fixture f in w.Unit.FixtureList)
                {
                    Vertices polyVerts = ((PolygonShape)f.Shape).Vertices;
                    path = new PointF[polyVerts.Count];
                    for (int i = 0; i < polyVerts.Count; i++)
                    {
                        Vector2 px = RotateVec(Transform.Multiply(polyVerts[i], ref trans), angle) * zoom + translation;
                        path[i] = new PointF(px.X, px.Y);
                    }
                    img.Mutate(i => i.FillPolygon(w.Color, path));
                }
                float a1 = w.Phase;
                float a2 = a1 + 1.2f; // radians
                float s1 = (float)Math.Sin(a1);
                float s2 = (float)Math.Sin(a2);
                float c1 = (float)Math.Cos(a1);
                float c2 = (float)Math.Cos(a2);
                if (s1 > 0f && s2 > 0f)
                {
                    continue;
                }
                if (s1 > 0f)
                {
                    c1 = (c1 < 0f ? -1f : (c1 > 0f ? 1f : 0f));
                }
                if (s2 > 0f)
                {
                    c2 = (c2 < 0f ? -1f : (c2 > 0f ? 1f : 0f));
                }
                Vector2[] wp = new Vector2[4] {
                    new Vector2(-WHEEL_W*SIZE,WHEEL_R*c1*SIZE),
                    new Vector2( WHEEL_W*SIZE,WHEEL_R*c1*SIZE),
                    new Vector2( WHEEL_W*SIZE,WHEEL_R*c2*SIZE),
                    new Vector2(-WHEEL_W*SIZE,WHEEL_R*c2*SIZE)
                };
                path = new PointF[wp.Length];
                for (int i = 0; i < wp.Length; i++)
                {
                    Vector2 vec = RotateVec(Transform.Multiply(wp[i], ref trans), angle) * zoom + translation;
                    path[i] = new PointF(vec.X, vec.Y);
                }
                img.Mutate(i => i.FillPolygon(WheelWhite, path));
            }
        }

        #endregion
    }
}
