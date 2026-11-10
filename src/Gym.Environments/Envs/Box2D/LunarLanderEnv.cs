#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using Gym.Box2D;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Exceptions;
using Gym.Observations;
using Gym.Rendering;
using Gym.Spaces;
using Gym.Utils;
using NumSharp;
using Color = Gym.Rendering.Color;

namespace Gym.Environments.Envs.Box2D {
    /// <summary>
    ///     LunarLander-v3 and LunarLanderContinuous-v3 in one class, as in Gymnasium: a lander dropped from the top of the
    ///     screen with a random push, which must touch down on the pad between the flags. Discrete actions fire nothing, the
    ///     left engine, the main engine or the right engine (0 to 3); continuous actions throttle the main engine and the
    ///     side engines (<c>Box(-1, 1, (2,), float32)</c>). Step for step and frame for frame what Gymnasium's
    ///     <c>LunarLander</c> computes on Box2D 2.3, errors included.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The physics runs on <see cref="b2World"/>, Gym.NET's managed Box2D 2.3.0 (D5), which matches pybox2d 2.3.10 bit for
    ///     bit, through the same pybox2d-shaped calls Gymnasium makes in the same order: a new world per reset, the terrain's
    ///     edges, the hull with its random push, the legs on motorized, limited revolute joints, and every step's engine
    ///     impulses at dispersed points. The observation is computed in float64 from the float32 bodies, then stored as
    ///     float32; the reward is the change of a shaping term minus the fuel spent, -100 when the hull touches the ground or
    ///     the lander leaves the screen sideways, +100 when the lander comes to rest (falls asleep).
    ///     </para>
    ///     <para>
    ///     Gymnasium's quirks are kept: <c>reset</c> runs one no-op <see cref="Step"/> (drawing two dispersion values and
    ///     stepping the world) and returns its observation; the side engines push from a point 17 rather than
    ///     <see cref="SideEngineHeight"/> pixels to the side, so their torque depends on the lander's angle; particles are
    ///     bodies in the world, created only when a render mode is set, which never touch the lander; the flags are redrawn
    ///     once per drawn fixture, which deepens their anti-aliased edges as particles accumulate.
    ///     </para>
    ///     <para>
    ///     Python's attributes are properties under their names (<c>helipad_y</c> is <see cref="HelipadY"/>); the ones
    ///     Gymnasium hangs on Box2D bodies (<c>legs[i].ground_contact</c>, <c>particles[i].ttl</c>, the colours) live beside
    ///     them here (<see cref="LegGroundContact"/>, <see cref="ParticleTtl"/>). The class prints as Gymnasium's
    ///     <c>LunarLander</c> (DR-32), and its contact listener is a private nested <c>ContactDetector</c> (DR-28).
    ///     </para>
    ///     <para>
    ///     Not thread-safe; each env owns its world, so envs on different threads do not interfere. Reference:
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/lunar_lander.py#L79">gymnasium/envs/box2d/lunar_lander.py</see>.
    ///     </para>
    /// </remarks>
    [GymOptions]
    [PyName("LunarLander")]
    public partial class LunarLanderEnv : Env<NDArray, NDArray> {
        /// <summary>Gymnasium's <c>FPS</c>: steps per simulated second (the world steps 1/50 s) and the human frame rate.</summary>
        public const int Fps = 50;

        /// <summary>Gymnasium's <c>SCALE</c>: pixels per metre; also divides every geometry constant below, which are in pixels.</summary>
        public const double Scale = 30.0;

        /// <summary>Gymnasium's <c>MAIN_ENGINE_POWER</c>: the main engine's impulse factor.</summary>
        public const double MainEnginePower = 13.0;

        /// <summary>Gymnasium's <c>SIDE_ENGINE_POWER</c>: the side engines' impulse factor.</summary>
        public const double SideEnginePower = 0.6;

        /// <summary>Gymnasium's <c>INITIAL_RANDOM</c>: each component of the reset's push is uniform in ±1000 N.</summary>
        public const double InitialRandom = 1000.0;

        /// <summary>Gymnasium's <c>LEG_AWAY</c>: the legs' horizontal offset from the hull's centre, in pixels.</summary>
        public const int LegAway = 20;

        /// <summary>Gymnasium's <c>LEG_DOWN</c>: the legs' joint anchor below the hull's centre, in pixels (also the observation's y offset).</summary>
        public const int LegDown = 18;

        /// <summary>Gymnasium's <c>LEG_W</c>: a leg's half-width, in pixels.</summary>
        public const int LegW = 2;

        /// <summary>Gymnasium's <c>LEG_H</c>: a leg's half-height, in pixels.</summary>
        public const int LegH = 8;

        /// <summary>Gymnasium's <c>LEG_SPRING_TORQUE</c>: the legs' motor torque, holding them against their limits.</summary>
        public const int LegSpringTorque = 40;

        /// <summary>Gymnasium's <c>SIDE_ENGINE_HEIGHT</c>: the side engines' height above the hull's centre, in pixels.</summary>
        public const int SideEngineHeight = 14;

        /// <summary>Gymnasium's <c>SIDE_ENGINE_AWAY</c>: the side engines' horizontal offset, in pixels.</summary>
        public const int SideEngineAway = 12;

        /// <summary>Gymnasium's <c>MAIN_ENGINE_Y_LOCATION</c>: the main engine below the hull's centre, in pixels.</summary>
        public const int MainEngineYLocation = 4;

        /// <summary>Gymnasium's <c>VIEWPORT_W</c>: the frame's width in pixels (20 m).</summary>
        public const int ViewportW = 600;

        /// <summary>Gymnasium's <c>VIEWPORT_H</c>: the frame's height in pixels (13.3 m).</summary>
        public const int ViewportH = 400;

        /// <summary>Gymnasium's <c>LANDER_POLY</c>: the hull's vertices in pixels, counter-clockwise from the top left.</summary>
        public static IReadOnlyList<(int X, int Y)> LanderPoly { get; } = new[] { (-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17) };

        /// <summary>The hull and legs' fill colour (Gymnasium's <c>color1</c>).</summary>
        private static readonly Color BodyColor1 = new(128, 102, 230);

        /// <summary>The hull and legs' outline colour (Gymnasium's <c>color2</c>).</summary>
        private static readonly Color BodyColor2 = new(77, 77, 128);

        /// <summary>The human-mode window, opened on the first human render.</summary>
        private readonly HumanWindow _window;

        /// <summary>The particles in creation order (Gymnasium's <c>particles</c>), each with the attributes Python hangs on its body.</summary>
        private readonly List<Particle> _particles = new();

        /// <summary>The two legs (Gymnasium's <c>legs</c>), left (i = -1) then right; empty before the first reset.</summary>
        private b2Body[] _legs = Array.Empty<b2Body>();

        /// <summary>Each leg's <c>ground_contact</c>, set and cleared by the contact listener.</summary>
        private readonly bool[] _legGroundContact = new bool[2];

        /// <summary>Each leg's <c>joint</c>: its revolute joint to the hull.</summary>
        private b2RevoluteJoint?[] _legJoints = new b2RevoluteJoint?[2];

        /// <summary>The sky's polygons above the terrain (Gymnasium's <c>sky_polys</c>), in metres; <see langword="null"/> before the first reset.</summary>
        private (double X, double Y)[][]? _skyPolys;

        /// <summary>Gymnasium's <c>wind_idx</c>, <see langword="null"/> while the attribute does not exist (no reset with wind yet).</summary>
        private long? _windIdx;

        /// <summary>Gymnasium's <c>torque_idx</c>, <see langword="null"/> while the attribute does not exist.</summary>
        private long? _torqueIdx;

        /// <summary>Creates the env, in either mode.</summary>
        /// <param name="renderMode">
        ///     Gymnasium's <c>render_mode</c>: <see langword="null"/>, <c>"human"</c> or <c>"rgb_array"</c>; not validated, as in
        ///     Gymnasium (<c>make</c> checks it). Any mode but <see langword="null"/> also makes the engines emit particles.
        /// </param>
        /// <param name="continuous">Gymnasium's <c>continuous</c>: two throttles (LunarLanderContinuous-v3) instead of four discrete actions.</param>
        /// <param name="gravity">Gymnasium's <c>gravity</c>, in (-12, 0) m/s².</param>
        /// <param name="enableWind">Gymnasium's <c>enable_wind</c>: a non-periodic sideways force and torque while no leg touches the ground.</param>
        /// <param name="windPower">Gymnasium's <c>wind_power</c>: the wind force's magnitude (0 to 20 recommended).</param>
        /// <param name="turbulencePower">Gymnasium's <c>turbulence_power</c>: the wind torque's magnitude (0 to 2 recommended).</param>
        /// <param name="viewerFactory">Gym.NET's <c>viewer_factory</c> (DR-20) for human mode; <see langword="null"/> uses <c>Gymnasium.HumanViewerFactory</c>.</param>
        /// <exception cref="AssertionError">Thrown for a gravity outside (-12, 0), with Gymnasium's message.</exception>
        /// <remarks>
        ///     A wind or turbulence power outside the recommended range warns through <see cref="Logger.Warn"/>, as Gymnasium
        ///     does, and is kept. Footgun: the gravity in the message prints as a float (<c>-13.0</c>) where Python prints an
        ///     int passed as such (<c>-13</c>), since the parameter is a <see langword="double"/>.
        /// </remarks>
        public LunarLanderEnv(string? renderMode = null, bool continuous = false, double gravity = -10.0, bool enableWind = false, double windPower = 15.0,
                              double turbulencePower = 1.5, IEnvironmentViewerFactoryDelegate? viewerFactory = null) {
            Metadata = ClassMetadata;
            if (!(-12.0 < gravity && gravity < 0.0))
                throw new AssertionError($"gravity (current value: {PyBuiltins.FloatRepr(gravity)}) must be between -12 and 0");
            Gravity = gravity;

            if (0.0 > windPower || windPower > 20.0)
                Logger.Warn($"wind_power value is recommended to be between 0.0 and 20.0, (current value: {PyBuiltins.FloatRepr(windPower)})");
            WindPower = windPower;

            if (0.0 > turbulencePower || turbulencePower > 2.0)
                Logger.Warn($"turbulence_power value is recommended to be between 0.0 and 2.0, (current value: {PyBuiltins.FloatRepr(turbulencePower)})");
            TurbulencePower = turbulencePower;

            EnableWind = enableWind;
            World = new b2World((0, gravity));
            Continuous = continuous;

            // np.array([...]).astype(np.float32): Python's float64 bounds rounded to float32 (2π included, -0.0 kept).
            float[] lowValues = { -2.5f, -2.5f, -10.0f, -10.0f, (float)(-2 * Math.PI), -10.0f, -0.0f, -0.0f };
            float[] highValues = { 2.5f, 2.5f, 10.0f, 10.0f, (float)(2 * Math.PI), 10.0f, 1.0f, 1.0f };
            using NDArray low = np.array(lowValues);
            using NDArray high = np.array(highValues);
            // Box copies its bounds, so the temporaries are released here.
            ObservationSpace = new Box(low, high);
            ActionSpace = continuous ? new Box(-1, +1, new Shape(2), np.float32) : new Discrete(4);

            RenderMode = renderMode;
            _window = new HumanWindow(continuous ? "lunarlandercontinuous-v3" : "lunarlander-v3", viewerFactory);
        }

        /// <summary>
        ///     Gymnasium's class attribute <c>metadata</c>: <c>{"render_modes": ["human", "rgb_array"], "render_fps": 50}</c>.
        ///     Every env starts with a copy as its instance metadata.
        /// </summary>
        /// <remarks>A new dict on every read, so editing one env's metadata never reaches another (Python's is one shared dict).</remarks>
        public static new Metadata ClassMetadata => new() {
            [Metadata.RenderModesKey] = new List<string> { "human", "rgb_array" },
            [Metadata.RenderFpsKey] = (long)Fps,
        };

        /// <summary>Gymnasium's <c>gravity</c>: the world's downward acceleration, read again by every reset's new world.</summary>
        public double Gravity { get; set; }

        /// <summary>Gymnasium's <c>wind_power</c>: the wind force's magnitude, read every windy step.</summary>
        public double WindPower { get; set; }

        /// <summary>Gymnasium's <c>turbulence_power</c>: the wind torque's magnitude, read every windy step.</summary>
        public double TurbulencePower { get; set; }

        /// <summary>Gymnasium's <c>enable_wind</c>: whether resets draw wind indices and steps apply wind; read at both.</summary>
        public bool EnableWind { get; set; }

        /// <summary>Gymnasium's <c>continuous</c>, fixed at construction: which action space and which action handling.</summary>
        public bool Continuous { get; }

        /// <summary>
        ///     Gymnasium's <c>world</c>: the Box2D world, a new one per reset (Gymnasium's workaround for issue 728); before the
        ///     first reset, the constructor's empty one.
        /// </summary>
        public b2World World { get; private set; }

        /// <summary>Gymnasium's <c>moon</c>: the static terrain body, <see langword="null"/> before the first reset.</summary>
        public b2Body? Moon { get; private set; }

        /// <summary>Gymnasium's <c>lander</c>: the hull, <see langword="null"/> before the first reset; stepping requires it.</summary>
        public b2Body? Lander { get; private set; }

        /// <summary>Gymnasium's <c>legs</c>: the left and right leg bodies, empty before the first reset.</summary>
        public IReadOnlyList<b2Body> Legs => _legs;

        /// <summary>Gymnasium's <c>legs[i].ground_contact</c>: whether each leg touches the ground, as the contact listener last saw it.</summary>
        public IReadOnlyList<bool> LegGroundContact => _legGroundContact;

        /// <summary>Gymnasium's <c>legs[i].joint</c>: each leg's revolute joint to the hull, or <see langword="null"/> before the first reset.</summary>
        public IReadOnlyList<b2RevoluteJoint?> LegJoints => _legJoints;

        /// <summary>Gymnasium's <c>particles</c>: the exhaust particle bodies, oldest first; a new list on each read.</summary>
        public IReadOnlyList<b2Body> Particles => _particles.Select(p => p.Body).ToList();

        /// <summary>Gymnasium's <c>particles[i].ttl</c>: each particle's remaining life, which every render shortens by 0.15.</summary>
        public IReadOnlyList<double> ParticleTtl => _particles.Select(p => p.Ttl).ToList();

        /// <summary>Gymnasium's <c>drawlist</c>: the hull then the legs, drawn after the particles; empty before the first reset.</summary>
        public IReadOnlyList<b2Body> Drawlist => Lander is null ? Array.Empty<b2Body>() : new[] { Lander }.Concat(_legs).ToList();

        /// <summary>Gymnasium's <c>game_over</c>: set by the contact listener when the hull touches anything; ends the episode at the next step's check.</summary>
        public bool GameOver { get; set; }

        /// <summary>Gymnasium's <c>prev_shaping</c>: the last step's shaping term, <see langword="null"/> right after a reset began.</summary>
        public double? PrevShaping { get; set; }

        /// <summary>Gymnasium's <c>prev_reward</c>: always <see langword="null"/> (Python sets it to <c>None</c> and never reads it).</summary>
        public double? PrevReward => null;

        /// <summary>Gymnasium's <c>helipad_x1</c>: the pad's left flag, in metres.</summary>
        public double HelipadX1 { get; private set; }

        /// <summary>Gymnasium's <c>helipad_x2</c>: the pad's right flag, in metres.</summary>
        public double HelipadX2 { get; private set; }

        /// <summary>Gymnasium's <c>helipad_y</c>: the pad's height, in metres (H / 4).</summary>
        public double HelipadY { get; private set; }

        /// <summary>Gymnasium's <c>sky_polys</c>: the polygon above each terrain segment, in metres, as Python's float64 tuples.</summary>
        /// <exception cref="AttributeError">Thrown before the first reset, when the attribute does not exist yet.</exception>
        public IReadOnlyList<IReadOnlyList<(double X, double Y)>> SkyPolys =>
            _skyPolys ?? throw new AttributeError($"'{PythonClassName}' object has no attribute 'sky_polys'");

        /// <summary>Gymnasium's <c>wind_idx</c>: the wind function's position, advanced by every windy step.</summary>
        /// <exception cref="AttributeError">Thrown until a reset with wind enabled has drawn it, as Python's attribute does not exist before.</exception>
        public long WindIdx {
            get => _windIdx ?? throw new AttributeError($"'{PythonClassName}' object has no attribute 'wind_idx'");
            set => _windIdx = value;
        }

        /// <summary>Gymnasium's <c>torque_idx</c>: the turbulence function's position, advanced by every windy step.</summary>
        /// <exception cref="AttributeError">Thrown until a reset with wind enabled has drawn it.</exception>
        public long TorqueIdx {
            get => _torqueIdx ?? throw new AttributeError($"'{PythonClassName}' object has no attribute 'torque_idx'");
            set => _torqueIdx = value;
        }

        /// <summary>Gymnasium's <c>screen</c>: the human window's surface, created by the first human render; <see langword="null"/> otherwise.</summary>
        public Surface? Screen { get; private set; }

        /// <summary>Gymnasium's <c>clock</c>: paces human rendering, created by the first render.</summary>
        public Clock? Clock { get; private set; }

        /// <summary>Gymnasium's <c>surf</c>: the last frame drawn, flipped; <see langword="null"/> until the first render.</summary>
        public Surface? Surf { get; private set; }

        /// <summary>Gymnasium's <c>isopen</c>: <see langword="true"/> until <see cref="Close"/> shuts a human window down.</summary>
        [PyName("isopen")]
        public bool IsOpen { get; set; } = true;

        /// <summary>
        ///     Starts an episode, as Gymnasium's does in its order: reseeds the generator when a seed is given, destroys the old
        ///     bodies, makes a new world with the contact listener, draws the terrain (12 heights), creates the moon, the hull
        ///     and its random push (2 draws), the wind indices when wind is enabled (2 draws), and the legs, renders in human
        ///     mode, then runs a no-op <see cref="Step"/> (2 more draws) and returns its observation.
        /// </summary>
        /// <param name="seed">The seed of a fresh <c>Generator(PCG64(SeedSequence(seed)))</c>, or <see langword="null"/> to keep the generator.</param>
        /// <param name="options">Ignored, as in Gymnasium.</param>
        /// <returns>The no-op step's float32 observation and an empty info.</returns>
        /// <exception cref="Error">Thrown for a negative seed, with Gymnasium's message.</exception>
        /// <exception cref="DependencyNotInstalled">Thrown in human mode by the first render when no viewer factory is set (DR-20).</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/lunar_lander.py#L321">gymnasium/envs/box2d/lunar_lander.py</see>.</remarks>
        public override ResetResult<NDArray> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            SeedNpRandom(seed);
            Destroy();

            // Bug's workaround for Gymnasium issue 728: a totally new world for every reset.
            World = new b2World((0, Gravity));
            World.ContactListener = new ContactDetector(this);
            GameOver = false;
            PrevShaping = null;

            const double w = ViewportW / Scale;
            const double h = ViewportH / Scale;

            // Create Terrain
            const int chunks = 11;
            double[] height;
            using (NDArray drawn = NpRandom.uniform(0, h / 2, new Shape(chunks + 1)))
                height = drawn.ToArray<double>();
            var chunkX = new double[chunks];
            for (int i = 0; i < chunks; i++)
                chunkX[i] = w / (chunks - 1) * i;
            HelipadX1 = chunkX[chunks / 2 - 1];
            HelipadX2 = chunkX[chunks / 2 + 1];
            HelipadY = h / 4;
            for (int i = chunks / 2 - 2; i <= chunks / 2 + 2; i++)
                height[i] = HelipadY;
            var smoothY = new double[chunks];
            for (int i = 0; i < chunks; i++) {
                // height[i - 1] at i = 0 is Python's height[-1], the last of the 12.
                double previous = height[i == 0 ? height.Length - 1 : i - 1];
                smoothY[i] = 0.33 * (previous + height[i + 0] + height[i + 1]);
            }

            Moon = World.CreateStaticBody(new b2EdgeShape(new b2Vec2?[] { (0, 0), (w, 0) }));
            _skyPolys = new (double X, double Y)[chunks - 1][];
            for (int i = 0; i < chunks - 1; i++) {
                (double X, double Y) p1 = (chunkX[i], smoothY[i]);
                (double X, double Y) p2 = (chunkX[i + 1], smoothY[i + 1]);
                Moon.CreateEdgeFixture(new b2Vec2?[] { p1, p2 }, density: 0, friction: 0.1);
                _skyPolys[i] = new[] { p1, p2, (p2.X, h), (p1.X, h) };
            }

            // Create Lander body
            const double initialY = ViewportH / Scale;
            const double initialX = ViewportW / Scale / 2;
            var hull = new b2Vec2[LanderPoly.Count];
            for (int i = 0; i < hull.Length; i++)
                hull[i] = (LanderPoly[i].X / Scale, LanderPoly[i].Y / Scale);
            Lander = World.CreateDynamicBody(new b2FixtureDef(new b2PolygonShape(hull), density: 5.0, friction: 0.1, categoryBits: 0x0010,
                                                              maskBits: 0x001 /* collide only with ground */, restitution: 0.0),
                                             position: (initialX, initialY), angle: 0.0);

            // Apply the initial random impulse to the lander: two draws, x first.
            double pushX = Uniform(-InitialRandom, InitialRandom);
            double pushY = Uniform(-InitialRandom, InitialRandom);
            Lander.ApplyForceToCenter((pushX, pushY), true);

            if (EnableWind) {
                // Initialize wind pattern based on index
                _windIdx = Integer(-9999, 9999);
                _torqueIdx = Integer(-9999, 9999);
            }

            // Create Lander Legs
            _legs = new b2Body[2];
            _legJoints = new b2RevoluteJoint?[2];
            int k = 0;
            foreach (int i in new[] { -1, +1 }) {
                b2Body leg = World.CreateDynamicBody(new b2FixtureDef(new b2PolygonShape(null, (LegW / Scale, LegH / Scale)), density: 1.0, restitution: 0.0,
                                                                      categoryBits: 0x0020, maskBits: 0x001),
                                                     position: (initialX - i * LegAway / Scale, initialY), angle: i * 0.05);
                _legGroundContact[k] = false;
                var rjd = new b2RevoluteJointDef(bodyA: Lander, bodyB: leg, localAnchorA: (0, 0), localAnchorB: (i * LegAway / Scale, LegDown / Scale),
                                                 enableMotor: true, enableLimit: true, maxMotorTorque: LegSpringTorque,
                                                 motorSpeed: +0.3 * i /* low enough not to jump back into the sky */);
                if (i == -1) {
                    // The most esoteric numbers here, angled legs have freedom to travel within
                    rjd.LowerAngle = +0.9 - 0.5;
                    rjd.UpperAngle = +0.9;
                } else {
                    rjd.LowerAngle = -0.9;
                    rjd.UpperAngle = -0.9 + 0.5;
                }

                _legJoints[k] = (b2RevoluteJoint)World.CreateJoint(rjd);
                _legs[k] = leg;
                k++;
            }

            if (RenderMode == "human")
                Render();
            NDArray noop = Continuous ? np.array(new long[] { 0, 0 }) : NDArray.Scalar(0L);
            StepResult<NDArray> first;
            using (noop)
                first = Step(noop);
            return new ResetResult<NDArray>(first.Observation, new Info());
        }

        /// <summary>
        ///     One step, in Gymnasium's order: the wind and turbulence when enabled and no leg touches the ground, the action's
        ///     check (discrete) or clipping (continuous), two dispersion draws, the main engine's impulse, the side engine's,
        ///     one 1/50 s world step (180 velocity and 60 position iterations), then the observation, the shaped reward and the
        ///     termination checks. Renders in human mode.
        /// </summary>
        /// <param name="action">
        ///     Discrete: an element of <c>Discrete(4)</c> (a 0-d integer array; the untyped step also takes a .NET integer).
        ///     Continuous: an array of two throttles, clipped to [-1, 1] as float64; the main engine fires above 0 (50% to 100%),
        ///     the left or right engine beyond ±0.5. Read, never kept.
        /// </param>
        /// <returns>
        ///     The float32 observation (position, velocity, angle and angular velocity in Gymnasium's units, and each leg's
        ///     contact), the reward, whether the episode ended (crash, out of the screen, or at rest), <c>truncated=False</c> and
        ///     <c>{}</c>.
        /// </returns>
        /// <exception cref="AssertionError">
        ///     Thrown before the first reset (Python's bare <c>assert</c>, whose empty message NumSharp's <c>AssertionError</c>
        ///     does not give yet, N-1), and in discrete mode for an action outside the space, with Gymnasium's message
        ///     (<c>5 (&lt;class 'int'&gt;) invalid </c>, trailing space included).
        /// </exception>
        /// <exception cref="IndexError">
        ///     Thrown in continuous mode for an action without a first or second element, with NumPy's message; the wind, the
        ///     dispersion draws and (for a one-element action) the main engine have happened by then, as in Gymnasium.
        /// </exception>
        /// <exception cref="NotSupportedException">Thrown in continuous mode for an action of more than one dimension (NumPy's truth-value error is not modelled).</exception>
        /// <exception cref="DependencyNotInstalled">Thrown in human mode by the first render when no viewer factory is set (DR-20).</exception>
        /// <remarks>
        ///     The reward is a float64 here always, where Gymnasium's is an <c>np.float64</c>, and the int -100 or +100 on
        ///     termination (equal values). Reference:
        ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/lunar_lander.py#L467">gymnasium/envs/box2d/lunar_lander.py</see>.
        /// </remarks>
        public override StepResult<NDArray> Step(NDArray action) {
            // assert self.lander is not None (twice in the source; the first, bare one fails first).
            b2Body lander = Lander ?? throw new AssertionError();

            // Update wind and apply to the lander
            if (EnableWind && !(_legGroundContact[0] || _legGroundContact[1])) {
                // the function used for wind is tanh(sin(2 k x) + sin(pi k x)), which is proven to never be periodic, k = 0.01
                long windIdx = WindIdx;
                double windMag = Math.Tanh(Math.Sin(0.02 * windIdx) + Math.Sin(Math.PI * 0.01 * windIdx)) * WindPower;
                _windIdx = windIdx + 1;
                lander.ApplyForceToCenter((windMag, 0.0), true);

                // the function used for torque is tanh(sin(2 k x) + sin(pi k x)), which is proven to never be periodic, k = 0.01
                long torqueIdx = TorqueIdx;
                double torqueMag = Math.Tanh(Math.Sin(0.02 * torqueIdx) + Math.Sin(Math.PI * 0.01 * torqueIdx)) * TurbulencePower;
                _torqueIdx = torqueIdx + 1;
                lander.ApplyTorque(torqueMag, true);
            }

            ThrottleAction? throttles = null;
            long discrete = 0;
            if (Continuous)
                throttles = new ThrottleAction(action);
            else {
                AssertValidAction(action);
                discrete = Discrete.ToInt64(action);
            }

            // Apply Engine Impulses

            // Tip is the (X and Y) components of the rotation of the lander.
            double angle = lander.Angle;
            (double X, double Y) tip = (Math.Sin(angle), Math.Cos(angle));

            // Side is the (-Y and X) components of the rotation of the lander.
            (double X, double Y) side = (-tip.Y, tip.X);

            // Generate two random numbers between -1/SCALE and 1/SCALE.
            double dispersion0 = Uniform(-1.0, +1.0) / Scale;
            double dispersion1 = Uniform(-1.0, +1.0) / Scale;

            double mPower = 0.0;
            if (throttles is { } c ? c.Main > 0.0 : discrete == 2) {
                // Main engine
                if (throttles is { } m) {
                    mPower = (Math.Clamp(m.Main, 0.0, 1.0) + 1.0) * 0.5; // 0.5..1.0
                    if (!(mPower >= 0.5 && mPower <= 1.0))
                        throw new AssertionError();
                } else {
                    mPower = 1.0;
                }

                // 4 is move a bit downwards, +-2 for randomness
                // The components of the impulse to be applied by the main engine.
                double ox = tip.X * (MainEngineYLocation / Scale + 2 * dispersion0) + side.X * dispersion1;
                double oy = -tip.Y * (MainEngineYLocation / Scale + 2 * dispersion0) - side.Y * dispersion1;

                b2Vec2 position = lander.Position;
                (double X, double Y) impulsePos = (position.x + ox, position.y + oy);
                if (RenderMode is not null) {
                    // particles are just a decoration, with no impact on the physics, so don't add them when not rendering
                    b2Body p = CreateParticle(3.5 /* 3.5 is here to make particle speed adequate */, impulsePos.X, impulsePos.Y, mPower);
                    p.ApplyLinearImpulse((ox * MainEnginePower * mPower, oy * MainEnginePower * mPower), impulsePos, true);
                }

                lander.ApplyLinearImpulse((-ox * MainEnginePower * mPower, -oy * MainEnginePower * mPower), impulsePos, true);
            }

            double sPower = 0.0;
            if (throttles is { } s ? Math.Abs(s.Side) > 0.5 : discrete is 1 or 3) {
                // Orientation/Side engines
                double direction;
                if (throttles is { } t) {
                    direction = Math.Sign(t.Side);
                    sPower = Math.Clamp(Math.Abs(t.Side), 0.5, 1.0);
                    if (!(sPower >= 0.5 && sPower <= 1.0))
                        throw new AssertionError();
                } else {
                    // action = 1 is left, action = 3 is right
                    direction = discrete - 2;
                    sPower = 1.0;
                }

                // The components of the impulse to be applied by the side engines.
                double ox = tip.X * dispersion0 + side.X * (3 * dispersion1 + direction * SideEngineAway / Scale);
                double oy = -tip.Y * dispersion0 - side.Y * (3 * dispersion1 + direction * SideEngineAway / Scale);

                // The constant 17 is a constant, that is presumably meant to be SIDE_ENGINE_HEIGHT. However,
                // SIDE_ENGINE_HEIGHT is defined as 14. This causes the position of the thrust on the body of the lander to
                // change, depending on the orientation of the lander. This in turn results in an orientation dependent torque
                // being applied to the lander.
                b2Vec2 position = lander.Position;
                (double X, double Y) impulsePos = (position.x + ox - tip.X * 17 / Scale, position.y + oy + tip.Y * SideEngineHeight / Scale);
                if (RenderMode is not null) {
                    // particles are just a decoration, with no impact on the physics, so don't add them when not rendering
                    b2Body p = CreateParticle(0.7, impulsePos.X, impulsePos.Y, sPower);
                    p.ApplyLinearImpulse((ox * SideEnginePower * sPower, oy * SideEnginePower * sPower), impulsePos, true);
                }

                lander.ApplyLinearImpulse((-ox * SideEnginePower * sPower, -oy * SideEnginePower * sPower), impulsePos, true);
            }

            World.Step(1.0 / Fps, 6 * 30, 2 * 30);

            b2Vec2 pos = lander.Position;
            b2Vec2 vel = lander.LinearVelocity;

            double[] state = {
                (pos.x - ViewportW / Scale / 2) / (ViewportW / Scale / 2),
                (pos.y - (HelipadY + LegDown / Scale)) / (ViewportH / Scale / 2),
                vel.x * (ViewportW / Scale / 2) / Fps,
                vel.y * (ViewportH / Scale / 2) / Fps,
                lander.Angle,
                20.0 * lander.AngularVelocity / Fps,
                _legGroundContact[0] ? 1.0 : 0.0,
                _legGroundContact[1] ? 1.0 : 0.0,
            };

            double reward = 0;
            // And ten points for legs contact, the idea is if you lose contact again after landing, you get negative reward
            double shaping = -100 * Math.Sqrt(state[0] * state[0] + state[1] * state[1])
                             - 100 * Math.Sqrt(state[2] * state[2] + state[3] * state[3])
                             - 100 * Math.Abs(state[4])
                             + 10 * state[6]
                             + 10 * state[7];
            if (PrevShaping is { } previous)
                reward = shaping - previous;
            PrevShaping = shaping;

            reward -= mPower * 0.30; // less fuel spent is better, about -30 for heuristic landing
            reward -= sPower * 0.03;

            bool terminated = false;
            if (GameOver || Math.Abs(state[0]) >= 1.0) {
                terminated = true;
                reward = -100;
            }

            if (!lander.Awake) {
                terminated = true;
                reward = +100;
            }

            if (RenderMode == "human")
                Render();
            // truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
            float[] observation = new float[state.Length];
            for (int i = 0; i < state.Length; i++)
                observation[i] = (float)state[i];
            return new StepResult<NDArray>(np.array(observation), reward, terminated, false, new Info());
        }

        /// <summary>
        ///     Draws Gymnasium's frame with the same primitives in the same order: white, the black sky polygons above the
        ///     terrain, the particles (circles fading by life) and the hull and legs (filled, anti-aliased and outlined), each
        ///     followed by the two flags, in a y-up world flipped at the end. Shortens every particle's life by 0.15 first and
        ///     destroys the expired ones at the front, as Gymnasium's does, so rendering changes the world.
        /// </summary>
        /// <returns><c>rgb_array</c>: the <c>(400, 600, 3)</c> uint8 frame, a new array the caller owns. <c>human</c> and any other mode: nothing.</returns>
        /// <exception cref="AssertionError">Thrown without a render mode when the env has no spec (<c>assert self.spec is not None</c>).</exception>
        /// <exception cref="AttributeError">Thrown before the first reset (<c>'LunarLander' object has no attribute 'sky_polys'</c>), after the human window opened.</exception>
        /// <exception cref="KeyError">Thrown in human mode when <c>metadata["render_fps"]</c> was removed.</exception>
        /// <exception cref="DependencyNotInstalled">Thrown on the first human render when no viewer factory is set (DR-20).</exception>
        /// <remarks>
        ///     Gymnasium's discarded <c>pygame.transform.scale(self.surf, (SCALE, SCALE))</c> draws nothing and is left out.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/lunar_lander.py#L668">gymnasium/envs/box2d/lunar_lander.py</see>.
        /// </remarks>
        public override RenderResult Render() {
            if (RenderMode is null) {
                EnvSpec spec = Spec ?? throw new AssertionError();
                Logger.Warn("You are calling render method without specifying any render mode. You can specify the render_mode at initialization, "
                            + $"e.g. gym.make(\"{spec.Id}\", render_mode=\"rgb_array\")");
                return RenderResult.None;
            }

            if (Screen is null && RenderMode == "human") {
                // pygame.display.init(); self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
                _window.Ensure(ViewportW, ViewportH);
                Screen = new Surface(ViewportW, ViewportH);
            }

            Clock ??= new Clock();

            var surf = new Surface(ViewportW, ViewportH);
            Surf = surf;
            // pygame.draw.rect(self.surf, (255, 255, 255), self.surf.get_rect()): the whole opaque surface.
            surf.Fill(new Color(255, 255, 255));

            foreach (Particle particle in _particles) {
                particle.Ttl -= 0.15;
                // int(max(0.2, ...) * 255): Python's int() truncates.
                particle.Color1 = new Color((byte)(int)(Math.Max(0.2, 0.15 + particle.Ttl) * 255), (byte)(int)(Math.Max(0.2, 0.5 * particle.Ttl) * 255),
                                            (byte)(int)(Math.Max(0.2, 0.5 * particle.Ttl) * 255));
                particle.Color2 = particle.Color1;
            }

            CleanParticles(false);

            (double X, double Y)[][] skyPolys = _skyPolys ?? throw new AttributeError($"'{PythonClassName}' object has no attribute 'sky_polys'");
            foreach ((double X, double Y)[] p in skyPolys) {
                var scaledPoly = new List<(double X, double Y)>(p.Length);
                foreach ((double X, double Y) coord in p)
                    scaledPoly.Add((coord.X * Scale, coord.Y * Scale));
                Draw.Polygon(surf, new Color(0, 0, 0), scaledPoly);
                GfxDraw.AAPolygon(surf, scaledPoly, new Color(0, 0, 0));
            }

            var drawn = new List<(b2Body Body, Color Color1, Color Color2)>();
            foreach (Particle particle in _particles)
                drawn.Add((particle.Body, particle.Color1, particle.Color2));
            foreach (b2Body body in Drawlist)
                drawn.Add((body, BodyColor1, BodyColor2));

            foreach ((b2Body body, Color color1, Color color2) in drawn) {
                foreach (b2Fixture f in body.Fixtures) {
                    b2Transform trans = f.Body.Transform;
                    if (f.Shape is b2CircleShape circle) {
                        // trans * f.shape.pos * SCALE: Box2D's float32 transform, then a float32 product with SCALE.
                        b2Vec2 center = trans * circle.Pos * Scale;
                        // f.shape.radius * SCALE: the float32 radius widened, times SCALE in float64.
                        double radius = circle.Radius * Scale;
                        Draw.Circle(surf, color1, center.x, center.y, radius);
                        Draw.Circle(surf, color2, center.x, center.y, radius);
                    } else {
                        var path = new List<(double X, double Y)>();
                        foreach (b2Vec2 v in ((b2PolygonShape)f.Shape).Vertices) {
                            b2Vec2 point = trans * v * Scale;
                            path.Add((point.x, point.y));
                        }

                        Draw.Polygon(surf, color1, path);
                        GfxDraw.AAPolygon(surf, path, color1);
                        Draw.Aalines(surf, color2, true, path);
                    }

                    // Gymnasium draws the flags inside the fixture loop: once per drawn fixture.
                    foreach (double helipadX in new[] { HelipadX1, HelipadX2 }) {
                        double x = helipadX * Scale;
                        double flagy1 = HelipadY * Scale;
                        double flagy2 = flagy1 + 50;
                        Draw.Line(surf, new Color(255, 255, 255), x, flagy1, x, flagy2);
                        var flag = new List<(double X, double Y)> { (x, flagy2), (x, flagy2 - 10), (x + 25, flagy2 - 5) };
                        Draw.Polygon(surf, new Color(204, 204, 0), flag);
                        GfxDraw.AAPolygon(surf, flag, new Color(204, 204, 0));
                    }
                }
            }

            surf = Transform.Flip(surf, false, true);
            Surf = surf;

            if (RenderMode == "human") {
                Surface screen = Screen!;
                screen.Blit(surf, 0, 0);
                // pygame.event.pump() has no event queue to pump here; the viewer runs its own message loop.
                Clock.Tick(PyBuiltins.Float(Metadata[Metadata.RenderFpsKey]));
                // pygame.display.flip(): a window shut by Close() is reopened from the factory.
                _window.Show(screen);
            } else if (RenderMode == "rgb_array") {
                // np.transpose(np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)): rows of RGB.
                return new RenderResult.Rgb(SurfArray.ToRgbArray(surf));
            }

            return RenderResult.None;
        }

        /// <summary>Shuts the human window down once one was opened, and marks the env closed (<see cref="IsOpen"/>), as Gymnasium's does.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/lunar_lander.py#L796">gymnasium/envs/box2d/lunar_lander.py</see>.</remarks>
        public override void Close() {
            if (Screen is not null) {
                // pygame.display.quit(); pygame.quit()
                _window.Close();
                IsOpen = false;
            }
        }

        /// <summary>
        ///     Gymnasium's <c>heuristic(env, s)</c>: the hand-written controller that lands the lander, from an observation.
        ///     It steers toward an angle pointing at the pad and a hover height proportional to the horizontal offset, and after
        ///     touchdown only slows the fall.
        /// </summary>
        /// <param name="env">The env, wrapped or not: its unwrapped env's <c>continuous</c> attribute picks the action form.</param>
        /// <param name="s">The observation: 8 values (float32 as the env returns them; other dtypes are read as their values).</param>
        /// <returns>
        ///     Discrete: a 0-d int64 action, 0 to 3 (Python returns an <c>int</c>). Continuous: two throttles clipped to [-1, 1],
        ///     a float32 array, or float64 after touchdown (where Python's array mixes a float32 with the int 0).
        /// </returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="env"/> or <paramref name="s"/> is <see langword="null"/>.</exception>
        /// <exception cref="AttributeError">Thrown when the unwrapped env has no <c>continuous</c> attribute.</exception>
        /// <remarks>
        ///     The arithmetic follows NumPy's on a float32 observation (NEP 50): every product and difference with Python's
        ///     floats stays float32, and the comparisons with 0.4 and 0.05 are float32 comparisons, so the action is the one
        ///     Gymnasium's heuristic returns for the same observation. Reference:
        ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/lunar_lander.py#L805">gymnasium/envs/box2d/lunar_lander.py</see>.
        /// </remarks>
        public static NDArray Heuristic(IEnv env, NDArray s) {
            if (env is null)
                throw new ArgumentNullException(nameof(env));
            if (s is null)
                throw new ArgumentNullException(nameof(s));
            float S(int i) {
                using NDArray element = s[i];
                return Convert.ToSingle(element.GetValue(), CultureInfo.InvariantCulture);
            }

            float s0 = S(0), s1 = S(1), s2 = S(2), s3 = S(3), s4 = S(4), s5 = S(5), s6 = S(6), s7 = S(7);
            float angleTarg = s0 * 0.5f + s2 * 1.0f; // angle should point towards center
            if (angleTarg > 0.4f)
                angleTarg = 0.4f; // more than 0.4 radians (22 degrees) is bad
            if (angleTarg < -0.4f)
                angleTarg = -0.4f;
            float hoverTarg = 0.55f * Math.Abs(s0); // target y should be proportional to horizontal offset

            float angleTodo = (angleTarg - s4) * 0.5f - s5 * 1.0f;
            float hoverTodo = (hoverTarg - s1) * 0.5f - s3 * 0.5f;

            bool contact = s6 != 0 || s7 != 0;
            if (contact) {
                // legs have contact: override to reduce fall speed, that's all we need after contact
                angleTodo = 0;
                hoverTodo = -s3 * 0.5f;
            }

            object? flag = PyAttributes.GetAttr(env.Unwrapped, "continuous");
            bool continuous = flag is bool b ? b : Convert.ToDouble(flag, CultureInfo.InvariantCulture) != 0;
            if (continuous) {
                float main = Math.Clamp(hoverTodo * 20 - 1, -1f, 1f);
                if (contact) {
                    // np.array([float32, int 0]) promotes to float64: the int is not a weak scalar inside an array literal.
                    return np.array(new[] { (double)main, 0.0 });
                }

                float lateral = Math.Clamp(-angleTodo * 20, -1f, 1f);
                return np.array(new[] { main, lateral });
            }

            long a = 0;
            if (hoverTodo > Math.Abs(angleTodo) && hoverTodo > 0.05f)
                a = 2;
            else if (angleTodo < -0.05f)
                a = 3;
            else if (angleTodo > +0.05f)
                a = 1;
            return NDArray.Scalar(a);
        }

        /// <summary>
        ///     Gymnasium's <c>demo_heuristic_lander(env, seed, render)</c>: plays one episode with <see cref="Heuristic"/>,
        ///     printing the observation and the running reward every 20 steps and at the end, and returns the total reward.
        /// </summary>
        /// <param name="env">The env (Gymnasium's demo makes <c>LunarLander-v3</c> with <c>render_mode="rgb_array"</c>).</param>
        /// <param name="seed">The reset seed, or <see langword="null"/>.</param>
        /// <param name="render">Whether to render every step and close the env at the end.</param>
        /// <returns>The episode's total reward.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        /// <remarks>
        ///     Prints to <see cref="Console.Out"/> in Python's format (<c>observations: +0.01 -0.02 ...</c>,
        ///     <c>step 20 total_reward -12.34</c>). Reference:
        ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/lunar_lander.py#L852">gymnasium/envs/box2d/lunar_lander.py</see>.
        /// </remarks>
        public static double DemoHeuristicLander(IEnv env, BigInteger? seed = null, bool render = false) {
            if (env is null)
                throw new ArgumentNullException(nameof(env));
            double totalReward = 0;
            int steps = 0;
            NDArray s = (NDArray)env.Reset(seed).Observation!;
            while (true) {
                NDArray a = Heuristic(env, s);
                StepResult<object?> result = env.Step(a);
                s = (NDArray)result.Observation!;
                totalReward += result.Reward;

                if (render)
                    env.Render();

                if (steps % 20 == 0 || result.Terminated || result.Truncated) {
                    float[] values = s.ToArray<float>();
                    Console.WriteLine("observations: " + string.Join(" ", values.Select(x => Signed2((double)x))));
                    Console.WriteLine($"step {steps} total_reward {Signed2(totalReward)}");
                }

                steps += 1;
                if (result.Terminated || result.Truncated)
                    break;
            }

            if (render)
                env.Close();
            return totalReward;
        }

        /// <summary>The Python class this instance stands for: Gymnasium's <c>LunarLander</c> (DR-32).</summary>
        protected override string PythonClassName => "LunarLander";

        /// <summary>
        ///     Python's <c>f"{x:+0.2f}"</c>: the sign always (from the sign bit, so <c>-0.0</c> prints <c>-0.00</c>), two decimals
        ///     rounded half to even on the exact binary value, as Python's formatting rounds (<c>0.125</c> is <c>+0.12</c>).
        /// </summary>
        /// <param name="x">The value.</param>
        /// <returns>The text; <c>+nan</c>, <c>+inf</c> and <c>-inf</c> for the non-finite values.</returns>
        private static string Signed2(double x) {
            if (double.IsNaN(x))
                return "+nan";
            if (double.IsInfinity(x))
                return x > 0 ? "+inf" : "-inf";
            // PyMath.Round(x, 2) is Python's round(): correctly rounded, half to even, so its two-decimal text is exact.
            string digits = Math.Abs(PyMath.Round(x, 2)).ToString("0.00", CultureInfo.InvariantCulture);
            return (double.IsNegative(x) ? "-" : "+") + digits;
        }

        /// <summary>
        ///     The untyped step's action conversion, with LunarLander's own assert first in discrete mode: an action outside
        ///     <c>Discrete(4)</c> fails with Python's message for exactly the value given.
        /// </summary>
        /// <param name="action">The action as given to <see cref="IEnv.Step"/>.</param>
        /// <returns>The action as an array: the one given, or a new one for a plain .NET value.</returns>
        /// <exception cref="AssertionError">Thrown in discrete mode for anything <c>Discrete(4).contains</c> rejects.</exception>
        protected override NDArray ConvertAction(object? action) {
            if (!Continuous)
                AssertValidAction(action);
            return base.ConvertAction(action);
        }

        /// <summary><c>assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid "</c> (the trailing space is Gymnasium's).</summary>
        /// <param name="action">The action, as given (so the message shows its Python counterpart).</param>
        /// <exception cref="AssertionError">Thrown when the action space does not contain the action.</exception>
        private void AssertValidAction(object? action) {
            if (!ActionSpace.Contains(action!))
                throw new AssertionError($"{PyBuiltins.Repr(action)} ({PyBuiltins.TypeRepr(action)}) invalid ");
        }

        /// <summary>
        ///     Gymnasium's <c>_destroy</c>: once a reset made bodies, detaches the listener, destroys every particle, the moon,
        ///     the hull and both legs (their joints go with them). The world itself is replaced by the reset that calls this.
        /// </summary>
        private void Destroy() {
            if (Moon is null)
                return;
            World.ContactListener = null;
            CleanParticles(true);
            World.DestroyBody(Moon);
            Moon = null;
            World.DestroyBody(Lander!);
            Lander = null;
            World.DestroyBody(_legs[0]);
            World.DestroyBody(_legs[1]);
        }

        /// <summary>
        ///     Gymnasium's <c>_create_particle</c>: a small dynamic circle (radius 2 px) at a point, colliding only with the
        ///     ground, with a life of <paramref name="ttl"/>; then the expired particles at the front are destroyed.
        /// </summary>
        /// <param name="mass">The circle's density.</param>
        /// <param name="x">The world x.</param>
        /// <param name="y">The world y.</param>
        /// <param name="ttl">The life, the engine's power (0.5 to 1).</param>
        /// <returns>The particle's body.</returns>
        private b2Body CreateParticle(double mass, double x, double y, double ttl) {
            b2Body p = World.CreateDynamicBody(new b2FixtureDef(new b2CircleShape(2 / Scale, (0, 0)), density: mass, friction: 0.1, categoryBits: 0x0100,
                                                                maskBits: 0x001 /* collide only with ground */, restitution: 0.3),
                                               position: (x, y), angle: 0.0);
            _particles.Add(new Particle(p, ttl));
            CleanParticles(false);
            return p;
        }

        /// <summary>Gymnasium's <c>_clean_particles</c>: destroys particles from the front while all are to go or the front one's life is below zero.</summary>
        /// <param name="all">Whether to destroy every particle.</param>
        private void CleanParticles(bool all) {
            while (_particles.Count > 0 && (all || _particles[0].Ttl < 0)) {
                World.DestroyBody(_particles[0].Body);
                _particles.RemoveAt(0);
            }
        }

        /// <summary>One <c>np_random.uniform(low, high)</c> draw, a float64.</summary>
        /// <param name="low">The lower bound.</param>
        /// <param name="high">The upper bound.</param>
        /// <returns>The value.</returns>
        private double Uniform(double low, double high) {
            using NDArray drawn = NpRandom.uniform(low, high);
            return drawn.GetAtIndex<double>(0);
        }

        /// <summary>One <c>np_random.integers(low, high)</c> draw, an int64 in [low, high).</summary>
        /// <param name="low">The lower bound.</param>
        /// <param name="high">The exclusive upper bound.</param>
        /// <returns>The value.</returns>
        private long Integer(long low, long high) {
            using NDArray drawn = NpRandom.integers(low, high);
            return drawn.GetAtIndex<long>(0);
        }

        /// <summary>
        ///     A continuous action read as Gymnasium reads it: <c>np.clip(action, -1, +1).astype(np.float64)</c>, then
        ///     <c>action[0]</c> and <c>action[1]</c> on demand, with NumPy's errors for missing elements.
        /// </summary>
        private readonly struct ThrottleAction {
            /// <summary>The clipped elements as float64.</summary>
            private readonly double[] _values;

            /// <summary>Reads and clips an action.</summary>
            /// <param name="action">The action array.</param>
            /// <exception cref="NotSupportedException">Thrown for an array of more than one dimension.</exception>
            public ThrottleAction(NDArray action) {
                if (action.ndim > 1)
                    throw new NotSupportedException($"A continuous LunarLander action of {action.ndim} dimensions is not modelled (NumPy's truth-value error).");
                using NDArray widened = action.astype(np.float64);
                double[] values = action.ndim == 0 ? new[] { widened.GetAtIndex<double>(0) } : widened.ToArray<double>();
                for (int i = 0; i < values.Length; i++) {
                    // np.clip keeps NaN; float32 values clip exactly in float64.
                    if (!double.IsNaN(values[i]))
                        values[i] = Math.Clamp(values[i], -1.0, 1.0);
                }

                _values = values;
                Dimensions = action.ndim;
            }

            /// <summary>The action's number of dimensions (0 for a scalar array).</summary>
            private int Dimensions { get; }

            /// <summary><c>action[0]</c>: the main throttle.</summary>
            /// <exception cref="IndexError">Thrown for a 0-d or empty action, with NumPy's message.</exception>
            public double Main => Element(0);

            /// <summary><c>action[1]</c>: the side throttle.</summary>
            /// <exception cref="IndexError">Thrown for an action of fewer than two elements, with NumPy's message.</exception>
            public double Side => Element(1);

            /// <summary>One element, as NumPy indexes the clipped action.</summary>
            /// <param name="index">The index.</param>
            /// <returns>The clipped value.</returns>
            /// <exception cref="IndexError">Thrown for a 0-d action or an index past the end, with NumPy's messages.</exception>
            private double Element(int index) {
                // np.clip of a 0-d array returns a NumPy scalar, whose indexing fails with the scalar's own message.
                if (Dimensions == 0)
                    throw new IndexError("invalid index to scalar variable.");
                if (index >= _values.Length)
                    throw new IndexError($"index {index} is out of bounds for axis 0 with size {_values.Length}");
                return _values[index];
            }
        }

        /// <summary>An exhaust particle: its body and the attributes Gymnasium hangs on it (<c>ttl</c>, <c>color1</c>, <c>color2</c>).</summary>
        private sealed class Particle {
            /// <summary>A particle with its life; colours are set by the first render.</summary>
            /// <param name="body">The body.</param>
            /// <param name="ttl">The life.</param>
            public Particle(b2Body body, double ttl) {
                Body = body;
                Ttl = ttl;
            }

            /// <summary>The circle body.</summary>
            public b2Body Body { get; }

            /// <summary>The remaining life; every render subtracts 0.15, and below zero the particle is destroyed.</summary>
            public double Ttl { get; set; }

            /// <summary>The fill colour, from the life at the last render.</summary>
            public Color Color1 { get; set; }

            /// <summary>The second colour, equal to <see cref="Color1"/>.</summary>
            public Color Color2 { get; set; }
        }

        /// <summary>
        ///     Gymnasium's <c>ContactDetector</c>, private to the env (DR-28): the hull touching anything ends the game; a leg's
        ///     contact sets its <c>ground_contact</c> and the contact's end clears it.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/lunar_lander.py#L58">gymnasium/envs/box2d/lunar_lander.py</see>.</remarks>
        private sealed class ContactDetector : b2ContactListener {
            /// <summary>The env whose flags the contacts set; borrowed, the env owns its arrays and disposes them.</summary>
            [NDBorrowed]
            private readonly LunarLanderEnv _env;

            /// <summary>A detector for an env.</summary>
            /// <param name="env">The env.</param>
            public ContactDetector(LunarLanderEnv env) => _env = env;

            /// <summary>Sets <c>game_over</c> for the hull and <c>ground_contact</c> for a leg.</summary>
            /// <param name="contact">The contact.</param>
            public override void BeginContact(b2Contact contact) {
                b2Body a = contact.FixtureA.Body, b = contact.FixtureB.Body;
                if (_env.Lander == a || _env.Lander == b)
                    _env.GameOver = true;
                for (int i = 0; i < 2; i++) {
                    if (_env._legs[i] == a || _env._legs[i] == b)
                        _env._legGroundContact[i] = true;
                }
            }

            /// <summary>Clears <c>ground_contact</c> for a leg.</summary>
            /// <param name="contact">The contact.</param>
            public override void EndContact(b2Contact contact) {
                b2Body a = contact.FixtureA.Body, b = contact.FixtureB.Body;
                for (int i = 0; i < 2; i++) {
                    if (_env._legs[i] == a || _env._legs[i] == b)
                        _env._legGroundContact[i] = false;
                }
            }
        }
    }

    /// <summary>
    ///     Gymnasium's <c>LunarLanderContinuous</c>: a class that only refuses to be built, pointing at
    ///     <c>make("LunarLander-v3", continuous=True)</c> (DR-32).
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/lunar_lander.py#L869">gymnasium/envs/box2d/lunar_lander.py</see>.</remarks>
    [PyName("LunarLanderContinuous")]
    public sealed class LunarLanderContinuousEnv {
        /// <summary>Always refuses, with Gymnasium's message.</summary>
        /// <exception cref="Error">Always: the continuous mode is a <c>continuous=True</c> kwarg of LunarLander.</exception>
        public LunarLanderContinuousEnv() =>
            throw new Error("Error initializing LunarLanderContinuous Environment.\n"
                            + "Currently, we do not support initializing this mode of environment by calling the class directly.\n"
                            + "To use this environment, instead create it by specifying the continuous keyword in gym.make, i.e.\n"
                            + "gym.make(\"LunarLander-v3\", continuous=True)");
    }
}
