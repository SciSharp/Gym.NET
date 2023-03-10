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
    /// <summary>
    /// Helpful enum for the discrete action space.
    /// </summary>
    public enum LunarLanderDiscreteActions : int
    {
        Nothing = 0,
        FireLeftThruster = 1,
        FireMainEngine = 2,
        FireRightThruster = 3
    }

    /// <summary>
    //    ## Description
    //    This environment is a classic rocket trajectory optimization problem.
    //    According to Pontryagin's maximum principle, it is optimal to fire the
    //    engine at full throttle or turn it off. This is the reason why this
    //    environment has discrete actions: engine on or off.

    //    There are two environment versions: discrete or continuous.
    //    The landing pad is always at coordinates (0,0). The coordinates are the
    //    first two numbers in the state vector.
    //    Landing outside of the landing pad is possible.Fuel is infinite, so an agent
    //    can learn to fly and then land on its first attempt.

    //    To see a heuristic landing, run:
    //    ```
    //    python gymnasium/envs/box2d/lunar_lander.py
    //    ```
    //    <!-- To play yourself, run: -->
    //    <!-- python examples/agents/keyboard_agent.py LunarLander-v2 -->

    //    ## Action Space
    //    There are four discrete actions available:
    //    - 0: do nothing
    //    - 1: fire left orientation engine
    //    - 2: fire main engine
    //    - 3: fire right orientation engine

    //    ## Observation Space
    //    The state is an 8-dimensional vector: the coordinates of the lander in `x` & `y`, its linear
    //    velocities in `x` & `y`, its angle, its angular velocity, and two booleans
    //    that represent whether each leg is in contact with the ground or not.

    //    ## Rewards
    //    After every step a reward is granted.The total reward of an episode is the
    //    sum of the rewards for all the steps within that episode.

    //    For each step, the reward:
    //    - is increased/decreased the closer/further the lander is to the landing pad.
    //    - is increased/decreased the slower/faster the lander is moving.
    //    - is decreased the more the lander is tilted (angle not horizontal).
    //    - is increased by 10 points for each leg that is in contact with the ground.
    //    - is decreased by 0.03 points each frame a side engine is firing.
    //    - is decreased by 0.3 points each frame the main engine is firing.

    //    The episode receive an additional reward of -100 or +100 points for crashing or landing safely respectively.

    //    An episode is considered a solution if it scores at least 200 points.

    //## Starting State
    //    The lander starts at the top center of the viewport with a random initial
    //    force applied to its center of mass.

    //## Episode Termination
    //    The episode finishes if:
    //    1) the lander crashes(the lander body gets in contact with the moon);
    //    2) the lander gets outside of the viewport(`x` coordinate is greater than 1);
    //    3) the lander is not awake.From the[Box2D docs](https://box2d.org/documentation/md__d_1__git_hub_box2d_docs_dynamics.html#autotoc_md61),
    //        a body which is not awake is a body which doesn't move and doesn't
    //        collide with any other body:
    //    > When Box2D determines that a body (or group of bodies) has come to rest,
    //    > the body enters a sleep state which has very little CPU overhead.If a
    //    > body is awake and collides with a sleeping body, then the sleeping body
    //    > wakes up. Bodies will also wake up if a joint or contact attached to
    //    > them is destroyed.

    //## Arguments
    //    To use to the _continuous_ environment, you need to specify the
    //    `continuous= True` argument like below:
    //    ```python
    //    import gymnasium as gym
    //    env = gym.make(
    //        "LunarLander-v2",
    //        continuous: bool = False,
    //        gravity: float = -10.0,
    //        enable_wind: bool = False,
    //        wind_power: float = 15.0,
    //        turbulence_power: float = 1.5,
    //    )
    //    ```
    //    If `continuous= True` is passed, continuous actions (corresponding to the throttle of the engines) will be used and the
    //    action space will be `Box(-1, +1, (2,), dtype=np.float32)`.
    //    The first coordinate of an action determines the throttle of the main engine, while the second
    //    coordinate specifies the throttle of the lateral boosters.
    //    Given an action `np.array([main, lateral])`, the main engine will be turned off completely if
    //    `main< 0` and the throttle scales affinely from 50% to 100% for `0 <= main <= 1` (in particular, the
    //    main engine doesn't work  with less than 50% power).
    //    Similarly, if `-0.5 < lateral< 0.5`, the lateral boosters will not fire at all.If `lateral< -0.5`, the left
    //    booster will fire, and if `lateral> 0.5`, the right booster will fire.Again, the throttle scales affinely
    //    from 50% to 100% between -1 and -0.5 (and 0.5 and 1, respectively).

    //    `gravity` dictates the gravitational constant, this is bounded to be within 0 and -12.

    //    If `enable_wind= True` is passed, there will be wind effects applied to the lander.
    //    The wind is generated using the function `tanh(sin(2 k (t+C)) + sin(pi k (t+C)))`.
    //    `k` is set to 0.01.
    //    `C` is sampled randomly between -9999 and 9999.

    //    `wind_power` dictates the maximum magnitude of linear wind applied to the craft.The recommended value for `wind_power` is between 0.0 and 20.0.
    //    `turbulence_power` dictates the maximum magnitude of rotational wind applied to the craft.The recommended value for `turbulence_power` is between 0.0 and 2.0.

    //    ## Version History
    //    - v2: Count energy spent and in v0.24, added turbulence with wind power and turbulence_power parameters
    //    - v1: Legs contact with ground added in state vector; contact with ground
    //        give +10 reward points, and -10 if then lose contact; reward
    //        renormalized to 200; harder initial random push.
    //    - v0: Initial version

    //    <!-- ## References -->

    //    ## Credits
    //    Created by Oleg Klimov
    /// </summary>
    public class LunarLanderEnv : Env
    {
        private IEnvironmentViewerFactoryDelegate _viewerFactory;
        private IEnvViewer _viewer;
        // static fields
        public int FPS = 50;
        public const float SCALE = 30f; //  affects how fast-paced the game is, forces should be adjusted as well
        public const float MAIN_ENGINE_POWER = 13f;
        public const float SIDE_ENGINE_POWER = 0.6f;
        public const float INITIAL_RANDOM = 1000f; // Set 1500 to make game harder
        public float SIDE_ENGINE_HEIGHT = 14f;
        public float SIDE_ENGINE_AWAY = 12f;
        // Screen
        public int VIEWPORT_W = 600;
        public int VIEWPORT_H = 400;


        public bool GameOver { get; set; } = false;

        #region Lunar Lander Body
        private LunarLanderBody _Lander = null;
        internal class LunarLanderComponent {
            public Body Unit { get; set; } = null;
            public bool Contact { get; set; } = false;
            public Rgba32 Color1 { get; set; }
            public Rgba32 Color2 { get; set; }
            public bool Is(Body b1, Body b2)
            {
                return (Unit == b1 || Unit == b2);
            }
        }
        internal class LunarLanderBody {
            internal const int LEFT_LEG = 0;
            internal const int RIGHT_LEG = 1;
            internal float LEG_W = 2f;
            internal float LEG_H = 8f;
            internal float LEG_AWAY = 20f;
            internal float LEG_DOWN = 18f;
            private float LEG_SPRING_TORQUE = 40f;
            private Vector2[] LANDER_POLY = { new Vector2(-14f / SCALE, 17f / SCALE), new Vector2(-17f / SCALE, 0f / SCALE), new Vector2(-17f / SCALE, -10f / SCALE), new Vector2(17f / SCALE, -10f / SCALE), new Vector2(17f / SCALE, 0f / SCALE), new Vector2(14f / SCALE, 17f / SCALE) };
            internal LunarLanderComponent Fuselage { get; set; } = null;
            internal LunarLanderComponent[] Legs { get; }
            internal Rgba32 Color1 { get; set; }
            internal Rgba32 Color2 { get; set; }

            internal LunarLanderBody(int nLegs, World world)
            {
                Legs = new LunarLanderComponent[nLegs];
                Fuselage = CreateFuselage(world);
                Legs[0] = CreateLeg(LEFT_LEG, Fuselage, world);
                Legs[1] = CreateLeg(RIGHT_LEG, Fuselage, world);
                Color1 = new Rgba32(128, 102, 230);
                Color2 = new Rgba32(77, 77, 128);
            }

            internal Body[] Bodies
            {
                get
                {
                    return new Body[] { Fuselage.Unit, Legs[0].Unit, Legs[1].Unit };
                }
            }
            internal LunarLanderComponent[] Components
            {
                get
                {
                    return new LunarLanderComponent[] { Fuselage, Legs[0], Legs[1] };
                }
            }
            internal bool TouchedGround
            {
                get
                {
                    return (Legs[0].Contact || Legs[1].Contact);
                }
            }

            private LunarLanderComponent CreateFuselage(World world)
            {
                LunarLanderComponent c = new LunarLanderComponent();
                c.Unit = world.CreateBody();
                c.Unit.Rotation = 0f;
                c.Unit.BodyType = BodyType.Dynamic;
                Vertices v = new Vertices(LANDER_POLY.Length);
                foreach (Vector2 v2 in LANDER_POLY)
                {
                    v.Add(v2);
                }
                Fixture f = c.Unit.CreateFixture(new PolygonShape(v, 5f));
                f.Friction = 0.1f;
                f.CollisionCategories = Category.Cat16; // 0x0010
                f.CollidesWith = Category.Cat1;
                f.Restitution = 0f;
                c.Color1 = new Rgba32(128, 102, 230);
                c.Color2 = new Rgba32(77, 77, 128);
                return (c);
            }
            /// <summary>
            /// Creates the body and fixture leg component of the lander. The iLeg parameter should be 0 or 1 for the left and right
            /// legs.
            /// </summary>
            /// <param name="iLeg"></param>
            /// <param name="fuselage"></param>
            /// <param name="world"></param>
            /// <returns></returns>
            private LunarLanderComponent CreateLeg(int iLeg, LunarLanderComponent fuselage, World world)
            {
                LunarLanderComponent c = new LunarLanderComponent();
                c.Unit = world.CreateBody();
                c.Unit.Rotation = iLeg == 0 ? -0.05f : 0.05f;
                c.Unit.BodyType = BodyType.Dynamic;
                c.Unit.Position = new Vector2((iLeg == 0 ? -1f : 1f) * LEG_AWAY / SCALE,0f);
                Vertices v = new Vertices(4);
                v.Add(new Vector2(0f, 0f));
                v.Add(new Vector2(LEG_W / SCALE, 0f));
                v.Add(new Vector2(LEG_W / SCALE, LEG_H / SCALE));
                v.Add(new Vector2(0f, LEG_H / SCALE));
                Fixture f = c.Unit.CreateFixture(new PolygonShape(v, 1f));
                f.Restitution = 0f;
                f.CollisionCategories = Category.Cat20;
                f.CollidesWith = Category.Cat1;
                RevoluteJoint rj = new RevoluteJoint(fuselage.Unit, c.Unit, new Vector2(0f, 0f), new Vector2((iLeg == 0 ? -1f : 1f) * LEG_AWAY / SCALE, LEG_DOWN / SCALE));
                rj.MotorEnabled = true;
                rj.LimitEnabled = true;
                rj.MaxMotorTorque = LEG_SPRING_TORQUE;
                rj.MotorSpeed = (iLeg == 0 ? -1f : 1f) * 0.3f;
                rj.UpperLimit = (iLeg == 0 ? 1f : -1f) * 0.9f + (iLeg == 0 ? 0f : 0.5f);
                rj.LowerLimit = (iLeg == 0 ? 1f : -1f) * 0.9f + (iLeg == 0 ? -0.5f : 0f);
                c.Color1 = new Rgba32(128, 102, 230);
                c.Color2 = new Rgba32(77, 77, 128);
                world.Add(rj);
                return (c);
            }

            internal bool HasContact
            {
                get
                {
                    if (Fuselage.Contact)
                    {
                        return (true);
                    }
                    for (int i = 0; i < Legs.Length; i++)
                    {
                        if (Legs[i].Contact)
                        {
                            return (true);
                        }
                    }
                    return (false);
                }
            }
        }
        #endregion

        internal class ContactDetector
        {
            private LunarLanderEnv _env { get; set; }

            public ContactDetector(LunarLanderEnv env)
            {
                _env = env;
            }

            public bool BeginContact(Contact contact)
            {
                _env._Lander.Fuselage.Contact = false;
                if (_env._Lander.Fuselage.Is(contact.FixtureA.Body,contact.FixtureB.Body))
                {
                    _env._Lander.Fuselage.Contact = true;
                }
                Body[] b = new Body[] { contact.FixtureA.Body, contact.FixtureB.Body };
                for (int i = 0; i < _env._Lander.Legs.Length; i++)
                {
                    _env._Lander.Legs[i].Contact = false;
                    if (_env._Lander.Legs[i].Is(contact.FixtureA.Body, contact.FixtureB.Body))
                    {
                        _env._Lander.Legs[i].Contact = true;
                    }
                }
                return (false);
            }

            public void EndContact(Contact contact)
            {
                Debug.WriteLine("END CONTACT!");
                Body[] b = new Body[] { contact.FixtureA.Body, contact.FixtureB.Body };
                for (int i = 0; i < _env._Lander.Legs.Length; i++)
                {
                    if (_env._Lander.Legs[i].Is(contact.FixtureA.Body, contact.FixtureB.Body))
                    {
                        _env._Lander.Legs[i].Contact = false;
                    }
                }
            }

        }

        public byte[] MainExhaustRGB { get; set; } = new byte[] { 255, 0, 0 };
        public byte[] ThrusterRGB { get; set; } = new byte[] { 255, 255, 0 };
        public bool ContinuousMode { get; private set; } = false;
        public float Gravity { get; private set; } = -10f;
        public bool UseWind { get; private set; } = false;
        public float WindPower { get; private set; } = 15f;
        public float TurbulencePower { get; private set; } = 1.5f;
        private NumPyRandom RandomState { get; set; }

        /// <summary>
        /// True when the filled circles are drawn for the side and main thrusters
        /// </summary>
        public bool RenderEngineParticles { get; set; } = true;
        public bool Verbose { get; set; } = false;

        private int _wind_idx;
        private int _torque_idx;
        private float _InitialY;
        private World _World;
        private Body _Moon;
        private List<LunarLanderParticle> _Particles = new List<LunarLanderParticle>();
        private float _Helipad_X1 = 0f;
        private float _Helipad_X2 = 0f;
        private float _Helipad_Y = 0f;
        private Vector2[] _Sky; // For display
        private ContactDetector _Contacts;
        private float _PrevShaping;

        public LunarLanderEnv(IEnvViewer viewer) : this((IEnvironmentViewerFactoryDelegate)null)
        {
            _viewer = viewer ?? throw new ArgumentNullException(nameof(viewer));
        }

        public LunarLanderEnv(IEnvironmentViewerFactoryDelegate viewerFactory, bool continuous = false, float gravity = -10f, bool enable_wind = false, float wind_power = 15f, float turbulence_power = 1.5f, NumPyRandom random_state=null)
        {
            RandomState = random_state;
            if (RandomState == null)
            {
                RandomState = np.random;
            }
            _viewerFactory = viewerFactory;
            ContinuousMode = continuous;
            Gravity = gravity;
            UseWind = enable_wind;
            WindPower = wind_power;
            TurbulencePower = turbulence_power;
            _Contacts = new ContactDetector(this);

            if (gravity < -12f || gravity > 0f)
            {
                throw (new ArgumentException("Gravity must be between -12 and 0", "gravity"));
            }
            if (wind_power < 0f || wind_power > 20f)
            {
                throw (new WarningException("wind_power value is recommended to be between 0.0 and 20.0"));
            }
            if (turbulence_power < 0f || turbulence_power > 2f)
            {
                throw (new WarningException("turbulence_power value is recommended to be between 0.0 and 2.0"));
            }

            _wind_idx = RandomState.randint(-9999, 9999);
            _torque_idx = RandomState.randint(-9999, 9999);

            NDArray low = np.array(new float[] { -1.5f, -1.5f, -5f, -5f, (float)-Math.PI, -5f, 0f, 0f });
            NDArray high = np.array(new float[] { 1.5f, 1.5f, 5f, 5f, (float)Math.PI, 5f, 1f, 1f });
            ObservationSpace = new Box(low, high);
            if (ContinuousMode)
            {
                ActionSpace = new Box(-1f, 1f, random_state : random_state);
            }
            else
            {
                ActionSpace = new Discrete(4, random_state: random_state);
            }

        }

        #region Particles
        class LunarLanderParticle
        {
            internal Body Unit { get; set; }
            internal int TTL { get; set; }
            internal Rgba32 RenderColor { get; set; }

            public static LunarLanderParticle Create(float x, float y, float mass, int ttl, World w)
            {
                Body b = w.CreateBody(new Vector2(x, y), 0f, BodyType.Dynamic);
                Fixture f = b.CreateFixture(new CircleShape(2f / SCALE, mass));
                f.Friction = 0.1f;
                f.CollisionCategories = Category.None;
                f.CollidesWith = Category.None;
                f.Restitution = 0.3f;
                return (new LunarLanderParticle(b, ttl));
            }
            public LunarLanderParticle(Body b, int ttl)
            {
                Unit = b;
                TTL = ttl;
            }
            public int Decay(int amt = 15)
            {
                TTL -= amt;
                return (TTL);
            }
        }
        private LunarLanderParticle CreateParticle(float mass, float x, float y, int ttl)
        {
            LunarLanderParticle p = LunarLanderParticle.Create(x, y, mass, ttl,_World);
            _Particles.Add(p);
            return (p);
        }

        private void CleanParticles(bool all)
        {
            if (_Particles == null || _Particles.Count == 0)
                return;
            for(int i=_Particles.Count-1; i >= 0; i--) 
            {
                LunarLanderParticle b = _Particles[i];
                if (b.TTL < 0 || all)
                {
                    _World.Remove(b.Unit);
                    _Particles.RemoveAt(i);
                }
            }
        }
        #endregion

        private void Destroy()
        {
            if (_Moon == null)
            {
                return;
            }
            CleanParticles(true);
            _Moon = null;
            _World = null;
            _Lander = null;
        }

        public override NDArray Reset()
        {
            Destroy();
            _World = new World(new Vector2(0f, Gravity));
            _World.ContactManager.BeginContact = new BeginContactDelegate(_Contacts.BeginContact);
            _World.ContactManager.EndContact = new EndContactDelegate(_Contacts.EndContact);
            _Lander = new LunarLanderBody(2,_World);
            _Lander.Fuselage.Unit.ApplyForce(new Vector2(RandomState.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), RandomState.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)));
            GameOver = false;
            _PrevShaping = float.MinValue;
            float w = VIEWPORT_W / SCALE;
            float h = VIEWPORT_H / SCALE;

            // Terrain Setup
            int CHUNKS = 11; // pieces of the moon
            float[] height = new float[CHUNKS + 1];
            for (int i = 0; i < height.Length; i++)
            {
                height[i] = RandomState.uniform(0f, h / 2f);
            }
            float[] chunk_x = new float[CHUNKS];
            for (int i = 0; i < CHUNKS; i++)
            {
                chunk_x[i] = w / (CHUNKS - 1) * i;
            }
            int mid_chunk = CHUNKS / 2;
            _Helipad_X1 = chunk_x[mid_chunk - 1]; // Helipad in the middle of the moon
            _Helipad_X2 = chunk_x[mid_chunk + 1];
            _Helipad_Y = h / 4f;
            height[mid_chunk - 2] = _Helipad_Y;
            height[mid_chunk - 1] = _Helipad_Y;
            height[mid_chunk] = _Helipad_Y;
            height[mid_chunk + 1] = _Helipad_Y;
            height[mid_chunk + 2] = _Helipad_Y;
            float[] smooth_y = new float[CHUNKS];
            for (int i = 0; i < CHUNKS; i++)
            {
                float h1 = 0f;
                if (i > 0)
                {
                    h1 = height[i - 1];
                }
                smooth_y[i] = 0.33f * (h1 + height[i] + height[i + 1]);
                if (smooth_y[i] > h)
                {
                    smooth_y[i] = h / 4f;
                }
            }
            // Make the moon surface for landing.
            _Moon = _World.CreateBody();
            _Moon.BodyType = BodyType.Static;
            _Moon.Position = new Vector2(0f, 0f);
            Fixture mf = _Moon.CreateFixture(new EdgeShape(new Vector2(0f, 0f), new Vector2(w, 0f)));

            // Define the sky
            _Sky = new Vector2[(CHUNKS - 1)*4];
            for (int i = 0,j=0; i < CHUNKS - 1; i++)
            {
                Vector2 p1 = new Vector2(chunk_x[i], smooth_y[i]);
                Vector2 p2 = new Vector2(chunk_x[i + 1], smooth_y[i + 1]);
                Fixture f = _Moon.CreateFixture(new EdgeShape(p1, p2));
                f.Friction = 0.1f;
                f.CollidesWith = Category.All;
                f.CollisionCategories = Category.All;
                _Sky[j++] = p1;
                _Sky[j++] = p2;
                _Sky[j++] = new Vector2(p2.X, 0f);
                _Sky[j++] = new Vector2(p1.X, 0f);
            }
            //
            // Place the lander starting position
            //
            _InitialY = VIEWPORT_H / SCALE;
            Vector2 dv = new Vector2(VIEWPORT_W / SCALE / 2, _InitialY);
            _Lander.Fuselage.Unit.Position += dv;
            _Lander.Legs[0].Unit.Position += dv;
            _Lander.Legs[1].Unit.Position += dv;
            // Do the zero step
            if (ContinuousMode)
            {
                return Step(np.array(new float[] { 0f, 0f })).Observation;
            }
            return Step(0).Observation;
        }

        public override Step Step(object action)
        {
            int i_action = -1;
            NDArray c_action = null;
            LunarLanderDiscreteActions l_action = LunarLanderDiscreteActions.Nothing;
            if (ContinuousMode)
            {
                c_action = (NDArray)action;
            }
            else
            {
                i_action = (int)action;
                l_action = (LunarLanderDiscreteActions)i_action;
            }
            if (UseWind && !_Lander.TouchedGround)
            {
                float wind_mag = (float)(Math.Tanh(Math.Sin(0.02 * _wind_idx) + Math.Sin(Math.PI * 0.01 * _wind_idx)))*WindPower;
                _wind_idx++;
                _Lander.Fuselage.Unit.ApplyForce(new Vector2(wind_mag, 0f));
                float torque_mag = (float)(Math.Tanh(Math.Sin(0.02 * _torque_idx) + Math.Sin(Math.PI * 0.01 * _torque_idx))) * TurbulencePower;
                _torque_idx++;
                _Lander.Fuselage.Unit.ApplyTorque(torque_mag);
            }
            if (ContinuousMode)
            {
                // Not implemented
                c_action = np.clip(c_action, -1f, 1f).astype(NPTypeCode.Float);
            }
            else
            {
                if (!ActionSpace.Contains(i_action))
                {
                    throw(new InvalidActionError(string.Format("Action {0} is invalid.", i_action)));
                }
            }
            Vector2 tip = new Vector2((float)Math.Sin(_Lander.Fuselage.Unit.Rotation), (float)Math.Cos(_Lander.Fuselage.Unit.Rotation));
            Vector2 side = new Vector2(-tip.Y, tip.X);
            float disp_x = RandomState.uniform(-1f, 1f) / SCALE;
            float disp_y = RandomState.uniform(-1f, 1f) / SCALE;

            bool fire_main = false;
            bool fire_thruster = false;
            if (ContinuousMode)
            {
                if (c_action[0] > 0f)
                    fire_main = true;
                if (np.abs(c_action[1]) > 0.5f)
                    fire_thruster = true;
                if (Verbose)
                {
                    System.Diagnostics.Debug.WriteLine("CONTINUOUS: fire_main {0}, fire_thruster {1}", fire_main, fire_thruster);
                }
            }
            else
            {
                fire_main = (l_action == LunarLanderDiscreteActions.FireMainEngine);
                fire_thruster = (l_action == LunarLanderDiscreteActions.FireLeftThruster || l_action == LunarLanderDiscreteActions.FireRightThruster);
            }
            #region Main Engine Physics
            float m_power = 0f;
            if (fire_main)
            {
                // Main Engine
                //
                if (ContinuousMode)
                {
                    m_power = (np.clip(c_action[0], 0f, 1f) + 1f) *0.5f;
                    if (Verbose)
                    {
                        System.Diagnostics.Debug.WriteLine("MAIN: m_power {0}", m_power);
                    }
                    if (m_power < 0.5f || m_power > 1f) {
                        // TODO: Throw an exception or assert here.
                        throw (new Exception("Main power is out of range"));
                    }
                }
                else
                {
                    m_power = 1f; // ALL or NONE for the main engine.
                }
                // 4 is move a bit downwards, +-2 for randomness
                float ox = tip.X * (4f / SCALE + 2f * disp_x) + side.X * disp_y;
                float oy = -tip.Y * (4F / SCALE + 2f * disp_x) - side.Y * disp_y;
                Vector2 impulse_pos = _Lander.Fuselage.Unit.Position + new Vector2(ox, oy);
                if (RenderEngineParticles)
                {
                    // Create the exhaust particles
                    LunarLanderParticle p = CreateParticle(3.5f, impulse_pos.X, impulse_pos.Y, (int)(m_power*100f));
                    p.RenderColor = new Rgba32(MainExhaustRGB[0], MainExhaustRGB[1], MainExhaustRGB[2]);
                    p.Unit.ApplyLinearImpulse(new Vector2(ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power),impulse_pos);
                }
                Vector2 impulse = new Vector2(-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power);
                if (Verbose)
                {
                    System.Diagnostics.Debug.WriteLine("MAIN: @{2}: applying {0} to {1}", impulse, impulse_pos, _Lander.Fuselage.Unit.Position);
                }
                _Lander.Fuselage.Unit.ApplyLinearImpulse(impulse,impulse_pos);
            }
            #endregion

            #region Orientation Engine Physics
            float s_power = 0f;
            if (fire_thruster)
            {
                // Orientation Engines
                //
                float direction = 0f;
                if (ContinuousMode)
                {
                    direction = c_action[1] < 0f ? -1f : 1f;
                    s_power = np.clip(np.abs(c_action[1]), 0.5f, 1.0f);
                    if (Verbose)
                    {
                        System.Diagnostics.Debug.WriteLine("SIDE {1}: s_power {0}", s_power, direction);
                    }
                    if (s_power < 0.5f || s_power > 1f)
                    {
                        // TODO: Assert power is out of bounds
                        throw (new Exception("Thruster power is out of range"));
                    }
                }
                else
                {
                    direction = (float)i_action - 2f;
                    s_power = 1f;
                }
                float ox = tip.X * disp_x + side.X * (3f * disp_y + direction * SIDE_ENGINE_AWAY / SCALE);
                float oy = -tip.Y * disp_x - side.Y * (3f * disp_y + direction * SIDE_ENGINE_AWAY / SCALE);
                Vector2 impulse_pos = _Lander.Fuselage.Unit.Position + new Vector2(ox - tip.X * 17f / SCALE, oy + tip.Y * SIDE_ENGINE_HEIGHT / SCALE);
                if (RenderEngineParticles)
                {
                    LunarLanderParticle p = CreateParticle(.7f, impulse_pos.X, impulse_pos.Y, (int)(s_power * 100f));
                    p.RenderColor = new Rgba32(ThrusterRGB[0], ThrusterRGB[1], ThrusterRGB[2]);
                    p.Unit.ApplyLinearImpulse(new Vector2(ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power), impulse_pos);
                }
                Vector2 impulse = new Vector2(-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power);
                if (Verbose)
                {
                    System.Diagnostics.Debug.WriteLine("SIDE {2}: applying {0} to {1}", impulse, impulse_pos, direction);
                }
                _Lander.Fuselage.Unit.ApplyLinearImpulse(impulse, impulse_pos);
            }
            #endregion

            //
            // Run the physics
            //
            float dt = 1f / FPS;
            SolverIterations si = new SolverIterations();
            si.PositionIterations = 2 * 30;
            si.VelocityIterations = 6 * 30;
            _World.Step(dt, ref si);
            if (_Lander.Fuselage.Contact)
            {
                GameOver = true;
            }
            //
            // Update our observation state
            //
            Vector2 pos = _Lander.Fuselage.Unit.Position;
            pos.X = (pos.X - VIEWPORT_W / SCALE / 2f) / (VIEWPORT_W / SCALE / 2f);
            pos.Y = (pos.Y - (_Helipad_Y + _Lander.LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2f);
            Vector2 vel = _Lander.Fuselage.Unit.LinearVelocity;
            vel.X *= (VIEWPORT_W / SCALE / 2F) / FPS;
            vel.Y *= (VIEWPORT_H / SCALE / 2F) / FPS;
            Step step = new Step();
            step.Information = new Dict();
            step.Information["pos"] = pos;
            step.Information["velocity"] = vel;
            step.Information["angle"] = _Lander.Fuselage.Unit.Rotation;
            step.Information["omega"] = 20f * _Lander.Fuselage.Unit.AngularVelocity / FPS;
            step.Information["LeftContact"] = _Lander.Legs[0].Contact;
            step.Information["RightContact"] = _Lander.Legs[1].Contact;
            step.Observation = new float[] { pos.X, pos.Y, vel.X, vel.Y, _Lander.Fuselage.Unit.Rotation, (float)step.Information["omega"], _Lander.Legs[0].Contact ? 1f : 0f, _Lander.Legs[1].Contact ? 1f : 0f };
            float reward = 0f;
            float shaping = -100f * np.sqrt(pos.X * pos.X + pos.Y * pos.Y);
            shaping += -100f * np.sqrt(vel.X * vel.X + vel.Y * vel.Y);
            shaping += -100f * Math.Abs((float)step.Information["angle"]);
            shaping += 10f * (_Lander.Legs[0].Contact ? 1f : 0f);
            shaping += 10f * (_Lander.Legs[1].Contact ? 1f : 0f);
            if (_PrevShaping != float.MinValue)
            {
                reward = shaping - _PrevShaping;
            }
            _PrevShaping = shaping;
            reward -= m_power * 0.3f;
            reward -= s_power * 0.03f;
            step.Done = false;
            if (GameOver || pos.X > 1f)
            {
                step.Done = true;
                reward = -100;
            }
            if (!_Lander.Fuselage.Unit.Awake)
            {
                step.Done = true;
                reward = 100;
            }
            step.Reward = reward;
            return (step);
        }

        public override Image Render(string mode = "human")
        {
            if (_viewer == null) {
                lock (this) {
                    //to prevent double initalization.
                    if (_viewer == null) {
                        if (_viewerFactory == null)
                            _viewerFactory = NullEnvViewer.Factory;
                        _viewer = _viewerFactory(VIEWPORT_W, VIEWPORT_H, "lunarlander-v2").GetAwaiter().GetResult();
                    }
                }
            }

            // Define the buffer image for drawing
            var img = new Image<Rgba32>(VIEWPORT_W, VIEWPORT_H);
            img.Mutate(i => i.BackgroundColor(new Rgba32(0, 0, 0))); // Space is black
            // Draw particles
            if (_Particles != null && _Particles.Count > 0) 
            {
                foreach (LunarLanderParticle p in _Particles)
                {
                    float ttl = (float)p.Decay() / 100f;
                    if (ttl <= 0f)
                    {
                        continue;
                    }
                    int r = (int)(Math.Clamp(Math.Max(0.2f, 0.15f + ttl) * 255.0, 0f, 255f));
                    int g = (int)(Math.Clamp(Math.Max(0.2f, 0.5f + ttl) * 255.0, 0f, 255f));
                    int b = (int)(Math.Clamp(Math.Max(0.2f, 0.5f + ttl) * 255.0, 0f, 255f));
                    Color c1 = new Rgba32(r, g, b);
                    if (p.RenderColor != null)
                    {
                        r = (int)(Math.Clamp(Math.Max(0.2f, 0.15f + ttl) * (float)p.RenderColor.R, 0f, 255f));
                        g = (int)(Math.Clamp(Math.Max(0.2f, 0.5f + ttl) * (float)p.RenderColor.G, 0f, 255f));
                        b = (int)(Math.Clamp(Math.Max(0.2f, 0.5f + ttl) * (float)p.RenderColor.B, 0f, 255f));
                        c1 = new Rgba32(r, g, b);
                    }
                    img.Mutate(i => i.Fill(c1, new EllipsePolygon(new PointF(p.Unit.Position.X * SCALE, VIEWPORT_H - p.Unit.Position.Y * SCALE), p.Unit.FixtureList[0].Shape.Radius * SCALE * ttl)));
                    // Original LunarLander.py has two circles drawn that are the same color.
                }
                CleanParticles(false);
            }
            // Draw the lander
            LunarLanderComponent[] drawlist = _Lander.Components;
            foreach (LunarLanderComponent c in drawlist)
            {
                Body obj = c.Unit;
                Transform t = obj.GetTransform();
                foreach (Fixture f in obj.FixtureList)
                {
                    if (f.Shape.ShapeType == ShapeType.Circle)
                    {
                    }
                    else
                    {
                        PolygonShape poly = f.Shape as PolygonShape;
                        LinearLineSegment[] segments = new LinearLineSegment[poly.Vertices.Count - 1];
                        List<PointF> points = new List<PointF>();
                        foreach (Vector2 v in poly.Vertices)
                        {
                            Vector2 xv = Transform.Multiply(v, ref t) * SCALE;
                            PointF pv = new PointF(xv.X, VIEWPORT_H - xv.Y);
                            points.Add(pv);
                        }
                        img.Mutate(i => i.DrawPolygon(c.Color1, 1f, points.ToArray()));
                    }
                }
            }
            // Draw the moon terrain
            PointF[] sky_poly = new PointF[4];
            for (int i = 0; i < _Sky.Length; i++)
            {
                PointF pv = new PointF(_Sky[i].X*SCALE, VIEWPORT_H - _Sky[i].Y*SCALE);
                sky_poly[i % 4] = pv;
                if ((i + 1) % 4 == 0)
                {
                    // Draw it!
                    img.Mutate(i => i.FillPolygon(new Rgba32(255, 255, 255), sky_poly));
                }
            }
            
            Transform moonT = _Moon.GetTransform();
            foreach (Fixture f in _Moon.FixtureList)
            {
                EdgeShape poly = f.Shape as EdgeShape;
                List<PointF> points = new List<PointF>();
                foreach (Vector2 v in new Vector2[] { poly.Vertex1, poly.Vertex2 })
                {
                    Vector2 xv = Transform.Multiply(v, ref moonT) * SCALE;
                    PointF pv = new PointF(xv.X, VIEWPORT_H - xv.Y);
                    points.Add(pv);
                }
                img.Mutate(i => i.DrawPolygon(new Rgba32(255,0,0), 1f, points.ToArray()));
            }

            // Draw the helipad
            foreach (float x in new float[] { _Helipad_X1, _Helipad_X2 }) {
                float x1 = x * SCALE;
                float y1 = VIEWPORT_H - _Helipad_Y*SCALE;
                float flag_y1 = y1;
                float flag_y2 = flag_y1 - 50f;
                PointF flag1 = new PointF(x1, flag_y1);
                PointF flag2 = new PointF(x1, flag_y2);
                // Pole
                img.Mutate(i => i.DrawLines(new Rgba32(255, 255, 255), 1, new PointF[] { flag1, flag2 }));
                // Chevron
                PointF p1 = new PointF(x1, flag_y2);
                PointF p2 = new PointF(x1, flag_y2 + 10f);
                PointF p3 = new PointF(x1 + 25f, flag_y2 + 5f);
                img.Mutate(i => i.DrawPolygon(new Rgba32(204, 204, 0), 1f, new PointF[] { p1, p2, p3}));
            }

            _viewer.Render(img);
            return img;
        }

        public override void CloseEnvironment()
        {
            if (_viewer != null) {
                _viewer.CloseEnvironment();
                _viewer = null;
            }
        }

        public override void Seed(int seed)
        {
            RandomState.seed(seed);
        }
    }
}
