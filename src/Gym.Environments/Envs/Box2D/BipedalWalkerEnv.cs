#nullable enable
using System;
using System.Collections.Generic;
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
    ///     BipedalWalker-v3 and BipedalWalkerHardcore-v3 in one class, as in Gymnasium: a four-joint walker (a hull, two legs of
    ///     two segments each) that must cross 200 terrain steps to the right. Actions are the four motors' commands in
    ///     <c>Box(-1, 1, (4,), float32)</c> (hip, knee, hip, knee): the sign picks the motor's direction at full speed, the
    ///     magnitude its torque. The observation is 24 float32 values: the hull's angle and velocities, each joint's angle and
    ///     speed, each lower leg's ground contact, and ten lidar distances. Step for step and frame for frame what Gymnasium's
    ///     <c>BipedalWalker</c> computes on Box2D 2.3, errors included.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The physics runs on <see cref="b2World"/>, Gym.NET's managed Box2D 2.3.0 (D5), through the same pybox2d-shaped calls
    ///     Gymnasium makes in the same order. Unlike LunarLander, the env keeps one world for its whole life, as Gymnasium
    ///     does: each reset destroys the last episode's bodies in Gymnasium's order and creates the new ones, so the broad
    ///     phase reuses its freed nodes exactly as pybox2d's does, which decides the order contacts are created and solved.
    ///     </para>
    ///     <para>
    ///     Footgun, Gymnasium's own: <c>reset(seed)</c> does not alone decide an episode. The freed tree nodes order the new
    ///     contacts, and with them the contact listener's begin and end events, which set the lower legs' ground contact
    ///     (and, through the solver's order, the bodies' last bits). An env that ran other episodes first can drift from a
    ///     fresh one after the same reset: Gymnasium's BipedalWalker-v3, used once by <c>check_environments_match</c> and then
    ///     compared with a fresh one, differs at step 96 of the seed-0 loop. Gym.NET matches Gymnasium step for step when both
    ///     envs have the same past; to reproduce an episode exactly, start from a new env.
    ///     </para>
    ///     <para>
    ///     The terrain is grass with gentle random slopes; the hardcore variant adds stumps, stairs and pits, each drawing its
    ///     sizes from the generator. The reward is the hull's progress (130 per metre over <see cref="Scale"/>) minus 5 per
    ///     radian of hull tilt, less 0.028 per unit of torque spent; -100 and the end when the hull touches the ground or the
    ///     walker leaves the left edge; the episode also ends past the far end's grass. The reward's precision is the
    ///     action's, as NumPy 2 computes it (NEP 50): float32 actions (the action space's samples) give a float32 reward,
    ///     float64 actions (the heuristic's) a float64 one; Gym.NET returns either as the double it equals.
    ///     </para>
    ///     <para>
    ///     Gymnasium's quirks are kept: <c>reset</c> runs one step with the int64 action <c>[0, 0, 0, 0]</c> (no torque at all)
    ///     and returns its observation; the lidar ignores the walker's own fixtures by category and keeps the closest hit;
    ///     <c>human</c> mode blits the scrolling frame at the truncated scroll offset onto a window it never clears, so while
    ///     the walker stands left of its start the window's left columns keep the previous frames' pixels, as pygame's do.
    ///     </para>
    ///     <para>
    ///     Python's attributes are properties under their names (<c>terrain_x</c> is <see cref="TerrainX"/>); the ones
    ///     Gymnasium hangs on Box2D bodies (<c>color1</c>, <c>color2</c>, the lower legs' <c>ground_contact</c>) live beside
    ///     them (<see cref="TerrainColors"/>, <see cref="LegGroundContact"/>). The class prints as Gymnasium's
    ///     <c>BipedalWalker</c> (DR-32); its contact listener is a private nested <c>ContactDetector</c> (DR-28) and its lidar
    ///     callbacks a private nested class, as Python's is local to <c>reset</c>.
    ///     </para>
    ///     <para>
    ///     Not thread-safe; each env owns its world, so envs on different threads do not interfere. Reference:
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/bipedal_walker.py#L103">gymnasium/envs/box2d/bipedal_walker.py</see>.
    ///     </para>
    /// </remarks>
    [GymOptions]
    [PyName("BipedalWalker")]
    public partial class BipedalWalkerEnv : Env<NDArray, NDArray> {
        /// <summary>Gymnasium's <c>FPS</c>: steps per simulated second (the world steps 1/50 s) and the human frame rate.</summary>
        public const int Fps = 50;

        /// <summary>Gymnasium's <c>SCALE</c>: pixels per metre; also divides every geometry constant below that is given in pixels.</summary>
        public const double Scale = 30.0;

        /// <summary>Gymnasium's <c>MOTORS_TORQUE</c>: each motor's torque at a full command, and the energy cost's factor.</summary>
        public const int MotorsTorque = 80;

        /// <summary>Gymnasium's <c>SPEED_HIP</c>: the hip motors' speed (rad/s) and the observation's hip speed scale.</summary>
        public const int SpeedHip = 4;

        /// <summary>Gymnasium's <c>SPEED_KNEE</c>: the knee motors' speed (rad/s) and the observation's knee speed scale.</summary>
        public const int SpeedKnee = 6;

        /// <summary>Gymnasium's <c>LIDAR_RANGE</c>: each lidar ray's length in metres (160 px).</summary>
        public const double LidarRange = 160 / Scale;

        /// <summary>Gymnasium's <c>INITIAL_RANDOM</c>: the reset's sideways push on the hull is uniform in ±5 N.</summary>
        public const int InitialRandom = 5;

        /// <summary>Gymnasium's <c>LEG_DOWN</c>: the hips' anchor below the hull's centre, in metres (-8 px).</summary>
        public const double LegDown = -8 / Scale;

        /// <summary>Gymnasium's <c>LEG_W</c>: an upper leg's width in metres (8 px); the lower legs are 0.8 of it.</summary>
        public const double LegW = 8 / Scale;

        /// <summary>Gymnasium's <c>LEG_H</c>: each leg segment's length in metres (34 px).</summary>
        public const double LegH = 34 / Scale;

        /// <summary>Gymnasium's <c>VIEWPORT_W</c>: the frame's width in pixels (20 m).</summary>
        public const int ViewportW = 600;

        /// <summary>Gymnasium's <c>VIEWPORT_H</c>: the frame's height in pixels (13.3 m).</summary>
        public const int ViewportH = 400;

        /// <summary>Gymnasium's <c>TERRAIN_STEP</c>: one terrain step's width in metres (14 px).</summary>
        public const double TerrainStep = 14 / Scale;

        /// <summary>Gymnasium's <c>TERRAIN_LENGTH</c>: the terrain's number of steps.</summary>
        public const int TerrainLength = 200;

        /// <summary>Gymnasium's <c>TERRAIN_HEIGHT</c>: the grass's resting height in metres (a quarter of the frame).</summary>
        public const double TerrainHeight = ViewportH / Scale / 4;

        /// <summary>Gymnasium's <c>TERRAIN_GRASS</c>: grass stretches last from half of this to this many steps.</summary>
        public const int TerrainGrass = 10;

        /// <summary>Gymnasium's <c>TERRAIN_STARTPAD</c>: the flat steps the walker starts on.</summary>
        public const int TerrainStartpad = 20;

        /// <summary>Gymnasium's <c>FRICTION</c>: the terrain's friction.</summary>
        public const double Friction = 2.5;

        /// <summary>Gymnasium's <c>HULL_POLY</c>: the hull's vertices in pixels.</summary>
        public static IReadOnlyList<(int X, int Y)> HullPoly { get; } = new[] { (-30, +9), (+6, +9), (+34, +1), (+34, -8), (-30, -8) };

        /// <summary>The hull's fill colour (Gymnasium's <c>hull.color1</c>).</summary>
        private static readonly Color HullColor1 = new(127, 51, 229);

        /// <summary>The hull's outline colour (Gymnasium's <c>hull.color2</c>).</summary>
        private static readonly Color HullColor2 = new(76, 76, 127);

        /// <summary>The hardcore obstacles' fill colour (stumps, stairs, pit walls).</summary>
        private static readonly Color ObstacleColor1 = new(255, 255, 255);

        /// <summary>The hardcore obstacles' outline colour.</summary>
        private static readonly Color ObstacleColor2 = new(153, 153, 153);

        /// <summary>The ground's fill under the terrain line.</summary>
        private static readonly Color GroundColor = new(102, 153, 76);

        /// <summary>The human-mode window, opened on the first human render.</summary>
        private readonly HumanWindow _window;

        /// <summary>Gymnasium's module-level <c>HULL_FD</c>: the hull's polygon, density 5, colliding only with the ground.</summary>
        private readonly b2FixtureDef _hullFd;

        /// <summary>Gymnasium's module-level <c>LEG_FD</c>: an upper leg's box, density 1.</summary>
        private readonly b2FixtureDef _legFd;

        /// <summary>Gymnasium's module-level <c>LOWER_FD</c>: a lower leg's narrower box, density 1.</summary>
        private readonly b2FixtureDef _lowerFd;

        /// <summary>Gymnasium's <c>terrain</c> with the colours Python hangs on each body, in Python's (reversed) order.</summary>
        private readonly List<(b2Body Body, Color Color1, Color Color2)> _terrain = new();

        /// <summary>Gymnasium's <c>terrain_x</c>: each step's x in metres.</summary>
        private readonly List<double> _terrainX = new();

        /// <summary>Gymnasium's <c>terrain_y</c>: each step's height in metres.</summary>
        private readonly List<double> _terrainY = new();

        /// <summary>Gymnasium's <c>terrain_poly</c>: the ground polygon under each terrain segment, in metres, with its colour.</summary>
        private readonly List<((double X, double Y)[] Poly, Color Color)> _terrainPoly = new();

        /// <summary>Gymnasium's <c>cloud_poly</c>: each cloud's polygon in metres with its leftmost and rightmost x.</summary>
        private readonly List<((double X, double Y)[] Poly, double X1, double X2)> _cloudPoly = new();

        /// <summary>The drawlist with each body's colours: the terrain, the four leg segments, then the hull.</summary>
        private readonly List<(b2Body Body, Color Color1, Color Color2)> _drawlist = new();

        /// <summary>Gymnasium's <c>legs</c>: upper and lower segment of the left leg (i = -1), then of the right; empty before the first reset.</summary>
        private b2Body[] _legs = Array.Empty<b2Body>();

        /// <summary>Gymnasium's <c>joints</c>: hip and knee of the left leg, then of the right; empty before the first reset.</summary>
        private b2RevoluteJoint[] _joints = Array.Empty<b2RevoluteJoint>();

        /// <summary>The <c>ground_contact</c> of each leg segment; only the lower ones (1 and 3) have the attribute in Python.</summary>
        private readonly bool[] _groundContact = new bool[4];

        /// <summary>Gymnasium's <c>lidar</c>: the ten ray-cast callbacks, a new set per reset.</summary>
        private LidarCallback[] _lidar = Array.Empty<LidarCallback>();

        /// <summary>Gymnasium's <c>scroll</c>, <see langword="null"/> while the attribute does not exist (before the first reset).</summary>
        private double? _scroll;

        /// <summary>Creates the env, in either mode.</summary>
        /// <param name="renderMode">
        ///     Gymnasium's <c>render_mode</c>: <see langword="null"/>, <c>"human"</c> or <c>"rgb_array"</c>; not validated, as in
        ///     Gymnasium (<c>make</c> checks it).
        /// </param>
        /// <param name="hardcore">Gymnasium's <c>hardcore</c>: stumps, stairs and pits in the terrain (BipedalWalkerHardcore-v3); read by every reset.</param>
        /// <param name="viewerFactory">Gym.NET's <c>viewer_factory</c> (DR-20) for human mode; <see langword="null"/> uses <c>Gymnasium.HumanViewerFactory</c>.</param>
        public BipedalWalkerEnv(string? renderMode = null, bool hardcore = false, IEnvironmentViewerFactoryDelegate? viewerFactory = null) {
            Metadata = ClassMetadata;
            World = new b2World();
            Hardcore = hardcore;

            // The terrain's two reusable definitions: every obstacle and every segment sets their shape's vertices, and each
            // fixture clones the shape as it is then.
            FdPolygon = new b2FixtureDef(new b2PolygonShape(new b2Vec2[] { (0, 0), (1, 0), (1, -1), (0, -1) }), friction: Friction);
            FdEdge = new b2FixtureDef(new b2EdgeShape(new b2Vec2?[] { (0, 0), (1, 1) }), friction: Friction, categoryBits: 0x0001);

            var hull = new b2Vec2[HullPoly.Count];
            for (int i = 0; i < hull.Length; i++)
                hull[i] = (HullPoly[i].X / Scale, HullPoly[i].Y / Scale);
            _hullFd = new b2FixtureDef(new b2PolygonShape(hull), density: 5.0, friction: 0.1, categoryBits: 0x0020,
                                       maskBits: 0x001 /* collide only with ground */, restitution: 0.0 /* 0.99 bouncy */);
            _legFd = new b2FixtureDef(new b2PolygonShape(null, (LegW / 2, LegH / 2)), density: 1.0, restitution: 0.0, categoryBits: 0x0020, maskBits: 0x001);
            _lowerFd = new b2FixtureDef(new b2PolygonShape(null, (0.8 * LegW / 2, LegH / 2)), density: 1.0, restitution: 0.0, categoryBits: 0x0020, maskBits: 0x001);

            // We use 5.0 to represent the joints moving at maximum 5 x the rated speed due to impulses from ground contact
            // etc. np.array([...]).astype(np.float32): Python's float64 bounds rounded to float32 (π included, -0.0 kept).
            const float pi = (float)Math.PI;
            float[] lowValues = {
                -pi, -5.0f, -5.0f, -5.0f, -pi, -5.0f, -pi, -5.0f, -0.0f, -pi, -5.0f, -pi, -5.0f, -0.0f,
                -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
            };
            float[] highValues = {
                pi, 5.0f, 5.0f, 5.0f, pi, 5.0f, pi, 5.0f, 5.0f, pi, 5.0f, pi, 5.0f, 5.0f,
                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
            };
            using NDArray actionLow = np.array(new[] { -1f, -1f, -1f, -1f });
            using NDArray actionHigh = np.array(new[] { 1f, 1f, 1f, 1f });
            using NDArray low = np.array(lowValues);
            using NDArray high = np.array(highValues);
            // Box copies its bounds, so the temporaries are released here.
            ActionSpace = new Box(actionLow, actionHigh);
            ObservationSpace = new Box(low, high);

            RenderMode = renderMode;
            _window = new HumanWindow(hardcore ? "bipedalwalkerhardcore-v3" : "bipedalwalker-v3", viewerFactory);
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

        /// <summary>
        ///     Gymnasium's <c>world</c>: the Box2D world (gravity (0, -10)), one for the env's whole life; resets destroy and
        ///     recreate the bodies in it.
        /// </summary>
        public b2World World { get; }

        /// <summary>Gymnasium's <c>hardcore</c>: whether the next reset's terrain has stumps, stairs and pits.</summary>
        public bool Hardcore { get; set; }

        /// <summary>
        ///     Gymnasium's <c>fd_polygon</c>: the obstacles' fixture definition (friction 2.5), whose polygon each obstacle
        ///     overwrites before its body is made; it holds the last obstacle's vertices.
        /// </summary>
        public b2FixtureDef FdPolygon { get; }

        /// <summary>Gymnasium's <c>fd_edge</c>: the terrain segments' fixture definition (friction 2.5, category 1), holding the last segment's vertices.</summary>
        public b2FixtureDef FdEdge { get; }

        /// <summary>
        ///     Gymnasium's <c>terrain</c>: the static terrain bodies in Python's order, the segments' edges last to first, then
        ///     the obstacles last to first (the list is built forward and reversed). Empty before the first reset.
        /// </summary>
        public IReadOnlyList<b2Body> Terrain => _terrain.Select(t => t.Body).ToList();

        /// <summary>
        ///     Gymnasium's <c>terrain[i].color1</c> and <c>color2</c>, index for index with <see cref="Terrain"/>: green
        ///     segments (alternating shades), white obstacles outlined grey.
        /// </summary>
        public IReadOnlyList<(Color Color1, Color Color2)> TerrainColors => _terrain.Select(t => (t.Color1, t.Color2)).ToList();

        /// <summary>Gymnasium's <c>terrain_x</c>: the 200 steps' x in metres; empty before the first reset.</summary>
        public IReadOnlyList<double> TerrainX => _terrainX.ToList();

        /// <summary>Gymnasium's <c>terrain_y</c>: the 200 steps' heights in metres; empty before the first reset.</summary>
        public IReadOnlyList<double> TerrainY => _terrainY.ToList();

        /// <summary>
        ///     Gymnasium's <c>terrain_poly</c>: for each of the 199 segments, the ground polygon down to y = 0 (as Python's
        ///     float64 tuples) and its colour; empty before the first reset.
        /// </summary>
        public IReadOnlyList<(IReadOnlyList<(double X, double Y)> Poly, Color Color)> TerrainPoly =>
            _terrainPoly.Select(p => ((IReadOnlyList<(double X, double Y)>)p.Poly.ToArray(), p.Color)).ToList();

        /// <summary>
        ///     Gymnasium's <c>cloud_poly</c>: the ten clouds' pentagons in metres with their leftmost and rightmost x; empty
        ///     before the first reset.
        /// </summary>
        public IReadOnlyList<(IReadOnlyList<(double X, double Y)> Poly, double X1, double X2)> CloudPoly =>
            _cloudPoly.Select(c => ((IReadOnlyList<(double X, double Y)>)c.Poly.ToArray(), c.X1, c.X2)).ToList();

        /// <summary>Gymnasium's <c>hull</c>: the walker's body, <see langword="null"/> before the first reset; stepping requires it.</summary>
        public b2Body? Hull { get; private set; }

        /// <summary>Gymnasium's <c>legs</c>: the left leg's upper and lower segments, then the right's; empty before the first reset.</summary>
        public IReadOnlyList<b2Body> Legs => _legs;

        /// <summary>Gymnasium's <c>joints</c>: the left hip, the left knee, the right hip, the right knee, as the actions drive them.</summary>
        public IReadOnlyList<b2RevoluteJoint> Joints => _joints;

        /// <summary>
        ///     Gymnasium's <c>legs[i].ground_contact</c>, index for index with <see cref="Legs"/>: whether each lower segment
        ///     touches the ground, as the contact listener last saw it, and <see langword="null"/> for the upper segments, which
        ///     have no such attribute in Python.
        /// </summary>
        public IReadOnlyList<bool?> LegGroundContact {
            get {
                var contacts = new bool?[_legs.Length];
                for (int i = 0; i < contacts.Length; i++)
                    contacts[i] = i % 2 == 1 ? _groundContact[i] : null;
                return contacts;
            }
        }

        /// <summary>Gymnasium's <c>drawlist</c>: the terrain, the four leg segments, then the hull; empty before the first reset.</summary>
        public IReadOnlyList<b2Body> Drawlist => _drawlist.Select(d => d.Body).ToList();

        /// <summary>Gymnasium's <c>game_over</c>: set by the contact listener when the hull touches the ground; ends the episode at the step's check.</summary>
        public bool GameOver { get; set; }

        /// <summary>Gymnasium's <c>prev_shaping</c>: the last step's shaping term, <see langword="null"/> right after a reset began.</summary>
        public double? PrevShaping { get; set; }

        /// <summary>
        ///     Gymnasium's <c>scroll</c>: the view's left edge in metres (the hull's x minus 4), which places the clouds, picks
        ///     the drawn terrain and offsets the human blit.
        /// </summary>
        /// <exception cref="AttributeError">Thrown before the first reset, as Python's attribute does not exist yet.</exception>
        public double Scroll {
            get => _scroll ?? throw new AttributeError($"'{PythonClassName}' object has no attribute 'scroll'");
            set => _scroll = value;
        }

        /// <summary>Gymnasium's <c>lidar_render</c>: which lidar ray the next frames draw (one per frame, sweeping out and back, then 80 frames without).</summary>
        public int LidarRender { get; set; }

        /// <summary>Gymnasium's <c>screen</c>: the human window's surface, created by the first human render and never cleared; <see langword="null"/> otherwise.</summary>
        public Surface? Screen { get; private set; }

        /// <summary>Gymnasium's <c>clock</c>: paces human rendering, created by the first render.</summary>
        public Clock? Clock { get; private set; }

        /// <summary>
        ///     Gymnasium's <c>surf</c>: the last frame drawn, flipped, as wide as the scrolled world so far
        ///     (<c>600 + 30 * scroll</c> pixels); <see langword="null"/> until the first render.
        /// </summary>
        public Surface? Surf { get; private set; }

        /// <summary>Gymnasium's <c>isopen</c>: <see langword="true"/> until <see cref="Close"/> shuts a human window down.</summary>
        [PyName("isopen")]
        public bool IsOpen { get; set; } = true;

        /// <summary>
        ///     Starts an episode, as Gymnasium's does in its order: reseeds the generator when a seed is given, destroys the last
        ///     episode's bodies, installs a new contact listener, draws the terrain (grass slopes, and the hardcore obstacles'
        ///     kinds and sizes) and the ten clouds, creates the hull with its random push and the legs on their motorized,
        ///     limited joints, renders in human mode, then runs a step with the int64 action <c>[0, 0, 0, 0]</c> and returns
        ///     its observation.
        /// </summary>
        /// <param name="seed">The seed of a fresh <c>Generator(PCG64(SeedSequence(seed)))</c>, or <see langword="null"/> to keep the generator.</param>
        /// <param name="options">Ignored, as in Gymnasium.</param>
        /// <returns>The zero-action step's float32 observation and an empty info.</returns>
        /// <exception cref="Error">Thrown for a negative seed, with Gymnasium's message.</exception>
        /// <exception cref="DependencyNotInstalled">Thrown in human mode by the first render when no viewer factory is set (DR-20).</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/bipedal_walker.py#L431">gymnasium/envs/box2d/bipedal_walker.py</see>.</remarks>
        public override ResetResult<NDArray> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            SeedNpRandom(seed);
            Destroy();
            World.ContactListener = new ContactDetector(this);
            GameOver = false;
            PrevShaping = null;
            _scroll = 0.0;
            LidarRender = 0;

            GenerateTerrain(Hardcore);
            GenerateClouds();

            const double initX = TerrainStep * TerrainStartpad / 2;
            const double initY = TerrainHeight + 2 * LegH;
            b2Body hull = World.CreateDynamicBody(_hullFd, position: (initX, initY));
            Hull = hull;
            hull.ApplyForceToCenter((Uniform(-InitialRandom, InitialRandom), 0.0), true);

            _legs = new b2Body[4];
            _joints = new b2RevoluteJoint[4];
            Array.Clear(_groundContact);
            var legColors = new (Color Color1, Color Color2)[4];
            int k = 0;
            foreach (int i in new[] { -1, +1 }) {
                var color1 = new Color((byte)(153 - i * 25), (byte)(76 - i * 25), (byte)(127 - i * 25));
                var color2 = new Color((byte)(102 - i * 25), (byte)(51 - i * 25), (byte)(76 - i * 25));

                b2Body leg = World.CreateDynamicBody(_legFd, position: (initX, initY - LegH / 2 - LegDown), angle: i * 0.05);
                var rjd = new b2RevoluteJointDef(bodyA: hull, bodyB: leg, localAnchorA: (0, LegDown), localAnchorB: (0, LegH / 2), enableMotor: true,
                                                 enableLimit: true, maxMotorTorque: MotorsTorque, motorSpeed: i, lowerAngle: -0.8, upperAngle: 1.1);
                _legs[k] = leg;
                legColors[k] = (color1, color2);
                _joints[k] = (b2RevoluteJoint)World.CreateJoint(rjd);
                k++;

                b2Body lower = World.CreateDynamicBody(_lowerFd, position: (initX, initY - LegH * 3 / 2 - LegDown), angle: i * 0.05);
                rjd = new b2RevoluteJointDef(bodyA: leg, bodyB: lower, localAnchorA: (0, -LegH / 2), localAnchorB: (0, LegH / 2), enableMotor: true,
                                             enableLimit: true, maxMotorTorque: MotorsTorque, motorSpeed: 1, lowerAngle: -1.6, upperAngle: -0.1);
                _groundContact[k] = false;
                _legs[k] = lower;
                legColors[k] = (color1, color2);
                _joints[k] = (b2RevoluteJoint)World.CreateJoint(rjd);
                k++;
            }

            _drawlist.Clear();
            _drawlist.AddRange(_terrain);
            for (int i = 0; i < _legs.Length; i++)
                _drawlist.Add((_legs[i], legColors[i].Color1, legColors[i].Color2));
            _drawlist.Add((hull, HullColor1, HullColor2));

            _lidar = new LidarCallback[10];
            for (int i = 0; i < _lidar.Length; i++)
                _lidar[i] = new LidarCallback();
            if (RenderMode == "human")
                Render();
            StepResult<NDArray> first;
            // np.array([0, 0, 0, 0]): an int64 action, so the reset's step costs a float64 zero of energy.
            using (NDArray zero = np.array(new long[] { 0, 0, 0, 0 }))
                first = Step(zero);
            return new ResetResult<NDArray>(first.Observation, new Info());
        }

        /// <summary>
        ///     One step, in Gymnasium's order: each joint's motor speed (the command's sign times the joint's rated speed) and
        ///     maximum torque (its clipped magnitude times 80), one 1/50 s world step (180 velocity and 60 position iterations),
        ///     the ten lidar ray casts from the hull, then the observation, the scroll, the shaped reward less the energy spent,
        ///     and the termination checks. Renders in human mode.
        /// </summary>
        /// <param name="action">
        ///     The four motor commands, normally a float32 array of <c>Box(-1, 1, (4,), float32)</c>; any 1-d numeric array is
        ///     read as NumPy indexes it (beyond ±1 the torque is clipped, NaN propagates into the physics). Elements past the
        ///     fourth add to the energy cost only. Read, never kept.
        /// </param>
        /// <returns>
        ///     The float32 observation, the reward (a float32 value for float32 actions, float64 otherwise, -100 on a fall),
        ///     whether the episode ended (a fall, the left edge, or past the far end), <c>truncated=False</c> and <c>{}</c>.
        /// </returns>
        /// <exception cref="AssertionError">
        ///     Thrown before the first reset (Python's bare <c>assert</c>, with its empty message, which NumSharp's
        ///     <c>AssertionError</c> gives since 0.81.0).
        /// </exception>
        /// <exception cref="IndexError">
        ///     Thrown for a 0-d action or one of fewer than four elements, with NumPy's message; the joints before the missing
        ///     element have their new motor settings by then, as in Gymnasium.
        /// </exception>
        /// <exception cref="TypeError">
        ///     Thrown for an action of more than one dimension (<c>float()</c> of a row) or of booleans (<c>np.sign</c> has no
        ///     boolean loop), with NumPy's messages, before anything changes.
        /// </exception>
        /// <exception cref="NotSupportedException">Thrown for a float16 action, whose half-precision energy cost is not modelled.</exception>
        /// <exception cref="DependencyNotInstalled">Thrown in human mode by the first render when no viewer factory is set (DR-20).</exception>
        /// <remarks>
        ///     The reward is returned as a double always: the float32, float64 or int value Gymnasium's step returns, widened
        ///     exactly. Reference:
        ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/bipedal_walker.py#L523">gymnasium/envs/box2d/bipedal_walker.py</see>.
        /// </remarks>
        public override StepResult<NDArray> Step(NDArray action) {
            b2Body hull = Hull ?? throw new AssertionError();
            var motors = new MotorAction(action);

            // self.hull.ApplyForceToCenter((0, 20), True) -- Uncomment this to receive a bit of stability help
            // control_speed = False: the speed is the command's sign at the rated speed, the torque the command's magnitude.
            for (int j = 0; j < 4; j++) {
                int ratedSpeed = j % 2 == 0 ? SpeedHip : SpeedKnee;
                // float(SPEED * np.sign(action[j])): ±speed, 0 or NaN, exact in any precision; SWIG stores it as float32.
                _joints[j].MotorSpeed = (float)(ratedSpeed * NpSign(motors[j].Value));
                // float(MOTORS_TORQUE * np.clip(np.abs(action[j]), 0, 1)): a float32 command's product is exact in double, so
                // one rounding to float32 gives NumPy's float32 product; a float64 command rounds twice, as Python does.
                _joints[j].MaxMotorTorque = (float)(MotorsTorque * motors.ClippedMagnitude(j).Value);
            }

            World.Step(1.0 / Fps, 6 * 30, 2 * 30);

            b2Vec2 pos = hull.Position;
            b2Vec2 vel = hull.LinearVelocity;

            for (int i = 0; i < _lidar.Length; i++) {
                LidarCallback lidar = _lidar[i];
                lidar.Fraction = 1.0;
                lidar.P1 = (pos.x, pos.y);
                // The end point is Python's float64 tuple; the ray cast receives it rounded to float32, as SWIG converts it.
                lidar.P2 = ((double)pos.x + Math.Sin(1.5 * i / 10.0) * LidarRange, (double)pos.y - Math.Cos(1.5 * i / 10.0) * LidarRange);
                World.RayCast(lidar, pos, (lidar.P2.Value.X, lidar.P2.Value.Y));
            }

            var state = new double[24];
            state[0] = hull.Angle; // Normal angles up to 0.5 here, but sure more is possible.
            state[1] = 2.0 * hull.AngularVelocity / Fps;
            state[2] = 0.3 * vel.x * (ViewportW / Scale) / Fps; // Normalized to get -1..1 range
            state[3] = 0.3 * vel.y * (ViewportH / Scale) / Fps;
            // This will give 1.1 on high up, but it's still OK (and there should be spikes on hitting the ground, that's normal too)
            state[4] = _joints[0].Angle;
            state[5] = (double)_joints[0].Speed / SpeedHip;
            state[6] = _joints[1].Angle + 1.0;
            state[7] = (double)_joints[1].Speed / SpeedKnee;
            state[8] = _groundContact[1] ? 1.0 : 0.0;
            state[9] = _joints[2].Angle;
            state[10] = (double)_joints[2].Speed / SpeedHip;
            state[11] = _joints[3].Angle + 1.0;
            state[12] = (double)_joints[3].Speed / SpeedKnee;
            state[13] = _groundContact[3] ? 1.0 : 0.0;
            for (int i = 0; i < _lidar.Length; i++)
                state[14 + i] = _lidar[i].Fraction;

            _scroll = (double)pos.x - ViewportW / Scale / 5;

            // moving forward is a way to receive reward (normalized to get 300 on completion)
            double shaping = 130 * (double)pos.x / Scale;
            // keep head straight, other than that and falling, any behavior is unpunished
            shaping -= 5.0 * Math.Abs(state[0]);

            NpScalar reward = NpScalar.PyInt(0);
            if (PrevShaping is { } previous)
                reward = NpScalar.PyFloat(shaping - previous);
            PrevShaping = shaping;

            for (int i = 0; i < motors.Count; i++) {
                // normalized to about -50.0 using heuristic, more optimal agent should spend less. The term takes the
                // action's precision (NEP 50), and so does the reward once it subtracts a float32 term.
                reward -= NpScalar.PyFloat(0.00035 * MotorsTorque) * motors.ClippedMagnitude(i);
            }

            bool terminated = false;
            if (GameOver || pos.x < 0) {
                reward = NpScalar.PyInt(-100);
                terminated = true;
            }

            if (pos.x > (TerrainLength - TerrainGrass) * TerrainStep)
                terminated = true;

            if (RenderMode == "human")
                Render();
            // truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
            float[] observation = new float[state.Length];
            for (int i = 0; i < state.Length; i++)
                observation[i] = (float)state[i];
            return new StepResult<NDArray>(np.array(observation), reward.Value, terminated, false, new Info());
        }

        /// <summary>
        ///     Draws Gymnasium's frame with the same primitives in the same order on a surface as wide as the scrolled world:
        ///     the sky, the visible clouds and ground polygons, one lidar ray (sweeping one per frame), every terrain body, leg
        ///     segment and the hull (filled, anti-aliased and outlined; segments as anti-aliased lines), and the start flag,
        ///     in a y-up world flipped at the end.
        /// </summary>
        /// <returns>
        ///     <c>rgb_array</c>: the <c>(400, 600, 3)</c> uint8 frame, the surface's right-most 600 columns, a new array the caller
        ///     owns. <c>human</c> and any other mode: nothing.
        /// </returns>
        /// <exception cref="AssertionError">Thrown without a render mode when the env has no spec (<c>assert self.spec is not None</c>).</exception>
        /// <exception cref="AttributeError">Thrown before the first reset (<c>'BipedalWalker' object has no attribute 'scroll'</c>), after the human window opened.</exception>
        /// <exception cref="KeyError">Thrown in human mode when <c>metadata["render_fps"]</c> was removed.</exception>
        /// <exception cref="DependencyNotInstalled">Thrown on the first human render when no viewer factory is set (DR-20).</exception>
        /// <remarks>
        ///     Each frame advances <see cref="LidarRender"/>, so rendering changes what the next frame shows, as in Gymnasium.
        ///     Gymnasium's discarded <c>pygame.transform.scale(self.surf, (SCALE, SCALE))</c> draws nothing and is left out.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/bipedal_walker.py#L615">gymnasium/envs/box2d/bipedal_walker.py</see>.
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

            double scroll = Scroll;
            // pygame.Surface((VIEWPORT_W + max(0.0, self.scroll) * SCALE, VIEWPORT_H)): pygame truncates the float width, and
            // Python's max keeps 0.0 against a NaN scroll, where Math.Max would return NaN.
            var surf = new Surface((int)(ViewportW + (scroll > 0.0 ? scroll : 0.0) * Scale), ViewportH);
            Surf = surf;

            Draw.Polygon(surf, new Color(215, 215, 255), new List<(double X, double Y)> {
                (scroll * Scale, 0), (scroll * Scale + ViewportW, 0), (scroll * Scale + ViewportW, ViewportH), (scroll * Scale, ViewportH),
            });

            foreach (((double X, double Y)[] poly, double x1, double x2) in _cloudPoly) {
                if (x2 < scroll / 2)
                    continue;
                if (x1 > scroll / 2 + ViewportW / Scale)
                    continue;
                var points = new List<(double X, double Y)>(poly.Length);
                foreach ((double X, double Y) p in poly)
                    points.Add((p.X * Scale + scroll * Scale / 2, p.Y * Scale));
                Draw.Polygon(surf, new Color(255, 255, 255), points);
                GfxDraw.AAPolygon(surf, points, new Color(255, 255, 255));
            }

            foreach (((double X, double Y)[] poly, Color color) in _terrainPoly) {
                if (poly[1].X < scroll)
                    continue;
                if (poly[0].X > scroll + ViewportW / Scale)
                    continue;
                var scaledPoly = new List<(double X, double Y)>(poly.Length);
                foreach ((double X, double Y) coord in poly)
                    scaledPoly.Add((coord.X * Scale, coord.Y * Scale));
                Draw.Polygon(surf, color, scaledPoly);
                GfxDraw.AAPolygon(surf, scaledPoly, color);
            }

            LidarRender = (LidarRender + 1) % 100;
            int i = LidarRender;
            if (i < 2 * _lidar.Length) {
                // self.lidar[len(self.lidar) - i - 1]: a negative Python index for the sweep back, 9 down to 0.
                LidarCallback singleLidar = i < _lidar.Length ? _lidar[i] : _lidar[_lidar.Length + (_lidar.Length - i - 1)];
                // hasattr(single_lidar, "p1") and hasattr(single_lidar, "p2"): false until a step cast the rays.
                if (singleLidar.P1 is { } p1 && singleLidar.P2 is { } p2)
                    Draw.Line(surf, new Color(255, 0, 0), p1.X * Scale, p1.Y * Scale, p2.X * Scale, p2.Y * Scale);
            }

            foreach ((b2Body body, Color color1, Color color2) in _drawlist) {
                foreach (b2Fixture f in body.Fixtures) {
                    b2Transform trans = f.Body.Transform;
                    if (f.Shape is b2CircleShape circle) {
                        // trans * f.shape.pos * SCALE: Box2D's float32 transform, then a float32 product with SCALE.
                        b2Vec2 center = trans * circle.Pos * Scale;
                        // f.shape.radius * SCALE: the float32 radius widened, times SCALE in float64.
                        double radius = circle.Radius * Scale;
                        Draw.Circle(surf, color1, center.x, center.y, radius);
                        Draw.Circle(surf, color2, center.x, center.y, radius);
                        continue;
                    }

                    // pybox2d's vertices: the polygon's hull order, or an edge's two vertices (no ghosts here).
                    IReadOnlyList<b2Vec2> vertices = f.Shape is b2EdgeShape edge ? new[] { edge.Vertex1, edge.Vertex2 } : ((b2PolygonShape)f.Shape).Vertices;
                    var path = new List<(double X, double Y)>(vertices.Count + 1);
                    foreach (b2Vec2 v in vertices) {
                        b2Vec2 point = trans * v * Scale;
                        path.Add((point.x, point.y));
                    }

                    if (path.Count > 2) {
                        Draw.Polygon(surf, color1, path);
                        GfxDraw.AAPolygon(surf, path, color1);
                        path.Add(path[0]);
                        // polygon(width=1) of the closed path: its outline, the repeated first point closing on itself.
                        Draw.Polygon(surf, color2, path, width: 1);
                        GfxDraw.AAPolygon(surf, path, color2);
                    } else {
                        Draw.Aaline(surf, color1, path[0].X, path[0].Y, path[1].X, path[1].Y);
                    }
                }
            }

            const double flagy1 = TerrainHeight * Scale;
            const double flagy2 = flagy1 + 50;
            const double x = TerrainStep * 3 * Scale;
            Draw.Aaline(surf, new Color(0, 0, 0), x, flagy1, x, flagy2);
            var flag = new List<(double X, double Y)> { (x, flagy2), (x, flagy2 - 10), (x + 25, flagy2 - 5) };
            Draw.Polygon(surf, new Color(230, 51, 0), flag);
            Draw.Lines(surf, new Color(0, 0, 0), false, new List<(double X, double Y)>(flag) { flag[0] }, 1);

            surf = Transform.Flip(surf, false, true);
            Surf = surf;

            if (RenderMode == "human") {
                Surface screen = Screen!;
                // self.screen.blit(self.surf, (-self.scroll * SCALE, 0)): pygame truncates the float position toward zero,
                // and the window is never cleared, so columns the surface leaves uncovered keep earlier frames.
                screen.Blit(surf, (int)(-scroll * Scale), 0);
                // pygame.event.pump() has no event queue to pump here; the viewer runs its own message loop.
                Clock.Tick(PyBuiltins.Float(Metadata[Metadata.RenderFpsKey]));
                // pygame.display.flip(): a window shut by Close() is reopened from the factory.
                _window.Show(screen);
            } else if (RenderMode == "rgb_array") {
                // np.transpose(np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2))[:, -VIEWPORT_W:]: the right-most
                // 600 columns, copied onto a frame-sized surface first.
                var frame = new Surface(ViewportW, ViewportH);
                frame.Blit(surf, ViewportW - surf.Width, 0);
                return new RenderResult.Rgb(SurfArray.ToRgbArray(frame));
            }

            return RenderResult.None;
        }

        /// <summary>Shuts the human window down once one was opened, and marks the env closed (<see cref="IsOpen"/>), as Gymnasium's does.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/bipedal_walker.py#L764">gymnasium/envs/box2d/bipedal_walker.py</see>.</remarks>
        public override void Close() {
            if (Screen is not null) {
                // pygame.display.quit(); pygame.quit()
                _window.Close();
                IsOpen = false;
            }
        }

        /// <summary>The Python class this instance stands for: Gymnasium's <c>BipedalWalker</c> (DR-32).</summary>
        protected override string PythonClassName => "BipedalWalker";

        /// <summary>
        ///     Gymnasium's <c>_destroy</c>: once a reset made bodies, detaches the listener (so no contact ends are reported),
        ///     then destroys the terrain bodies in list order, the hull and the four leg segments (their joints go with them).
        ///     The order is Gymnasium's, since the broad phase hands freed nodes to the next episode's fixtures last-freed first.
        /// </summary>
        private void Destroy() {
            if (_terrain.Count == 0)
                return;
            World.ContactListener = null;
            foreach ((b2Body body, _, _) in _terrain)
                World.DestroyBody(body);
            _terrain.Clear();
            World.DestroyBody(Hull!);
            Hull = null;
            foreach (b2Body leg in _legs)
                World.DestroyBody(leg);
            _legs = Array.Empty<b2Body>();
            _joints = Array.Empty<b2RevoluteJoint>();
        }

        /// <summary>
        ///     Gymnasium's <c>_generate_terrain</c>: walks the 200 steps through its states (grass, and in hardcore mode
        ///     stumps, stairs and pits after each grass stretch), drawing from the generator as Gymnasium does, creates each
        ///     obstacle's static body from <see cref="FdPolygon"/>, then each segment's edge body from <see cref="FdEdge"/> and its
        ///     ground polygon, and finally reverses the terrain list.
        /// </summary>
        /// <param name="hardcore">Whether stumps, stairs and pits follow the grass stretches.</param>
        private void GenerateTerrain(bool hardcore) {
            const long grass = 0, stump = 1, stairs = 2, pit = 3, statesCount = 4;
            long state = grass;
            double velocity = 0.0;
            double y = TerrainHeight;
            long counter = TerrainStartpad;
            bool oneshot = false;
            _terrain.Clear();
            _terrainX.Clear();
            _terrainY.Clear();

            long stairSteps = 0, stairWidth = 0, stairHeight = 0;
            double originalY = 0;
            for (int i = 0; i < TerrainLength; i++) {
                double x = i * TerrainStep;
                _terrainX.Add(x);

                if (state == grass && !oneshot) {
                    velocity = 0.8 * velocity + 0.01 * NpSign(TerrainHeight - y);
                    if (i > TerrainStartpad)
                        velocity += Uniform(-1, 1) / Scale; // 1
                    y += velocity;
                } else if (state == pit && oneshot) {
                    counter = Integer(3, 5);
                    (double X, double Y)[] poly = { (x, y), (x + TerrainStep, y), (x + TerrainStep, y - 4 * TerrainStep), (x, y - 4 * TerrainStep) };
                    AddObstacle(poly);
                    var shifted = new (double X, double Y)[poly.Length];
                    for (int p = 0; p < poly.Length; p++)
                        shifted[p] = (poly[p].X + TerrainStep * counter, poly[p].Y);
                    AddObstacle(shifted);
                    counter += 2;
                    originalY = y;
                } else if (state == pit && !oneshot) {
                    y = originalY;
                    if (counter > 1)
                        y -= 4 * TerrainStep;
                } else if (state == stump && oneshot) {
                    counter = Integer(1, 3);
                    AddObstacle(new[] {
                        (x, y), (x + counter * TerrainStep, y), (x + counter * TerrainStep, y + counter * TerrainStep), (x, y + counter * TerrainStep),
                    });
                } else if (state == stairs && oneshot) {
                    stairHeight = Random() > 0.5 ? +1 : -1;
                    // integers(4, 5) is always 4 and draws nothing: a one-value range consumes no bits in NumPy.
                    stairWidth = Integer(4, 5);
                    stairSteps = Integer(3, 5);
                    originalY = y;
                    for (long s = 0; s < stairSteps; s++) {
                        AddObstacle(new[] {
                            (x + (s * stairWidth) * TerrainStep, y + (s * stairHeight) * TerrainStep),
                            (x + ((1 + s) * stairWidth) * TerrainStep, y + (s * stairHeight) * TerrainStep),
                            (x + ((1 + s) * stairWidth) * TerrainStep, y + (-1 + s * stairHeight) * TerrainStep),
                            (x + (s * stairWidth) * TerrainStep, y + (-1 + s * stairHeight) * TerrainStep),
                        });
                    }

                    counter = stairSteps * stairWidth;
                } else if (state == stairs && !oneshot) {
                    long s = stairSteps * stairWidth - counter - stairHeight;
                    // s / stair_width: NumPy's true division of two int64s, a float64.
                    double n = (double)s / stairWidth;
                    y = originalY + (n * stairHeight) * TerrainStep;
                }

                oneshot = false;
                _terrainY.Add(y);
                counter -= 1;
                if (counter == 0) {
                    counter = Integer(TerrainGrass / 2, TerrainGrass);
                    if (state == grass && hardcore) {
                        state = Integer(1, statesCount);
                        oneshot = true;
                    } else {
                        state = grass;
                        oneshot = true;
                    }
                }
            }

            _terrainPoly.Clear();
            for (int i = 0; i < TerrainLength - 1; i++) {
                (double X, double Y) p0 = (_terrainX[i], _terrainY[i]);
                (double X, double Y) p1 = (_terrainX[i + 1], _terrainY[i + 1]);
                ((b2EdgeShape)FdEdge.Shape!).Vertices = new b2Vec2?[] { p0, p1 };
                b2Body t = World.CreateStaticBody(FdEdge);
                var color = new Color(76, (byte)(i % 2 == 0 ? 255 : 204), 76);
                _terrain.Add((t, color, color));
                _terrainPoly.Add((new[] { p0, p1, (p1.X, 0), (p0.X, 0) }, GroundColor));
            }

            _terrain.Reverse();
        }

        /// <summary>
        ///     One hardcore obstacle: sets <see cref="FdPolygon"/>'s vertices (rounded to float32, then Box2D's hull) and creates
        ///     a static body from it, white outlined grey.
        /// </summary>
        /// <param name="poly">The four corners in metres, as Python's float64 tuples.</param>
        private void AddObstacle(IReadOnlyList<(double X, double Y)> poly) {
            var vertices = new b2Vec2[poly.Count];
            for (int i = 0; i < vertices.Length; i++)
                vertices[i] = (poly[i].X, poly[i].Y);
            ((b2PolygonShape)FdPolygon.Shape!).Vertices = vertices;
            b2Body t = World.CreateStaticBody(FdPolygon);
            _terrain.Add((t, ObstacleColor1, ObstacleColor2));
        }

        /// <summary>
        ///     Gymnasium's <c>_generate_clouds</c>: ten pentagons at three quarters of the frame's height, each from 11 draws (its
        ///     x, then each vertex's x and y jitter in turn), with their extreme x for the render's visibility test.
        /// </summary>
        private void GenerateClouds() {
            // Sorry for the clouds, couldn't resist
            _cloudPoly.Clear();
            for (int c = 0; c < TerrainLength / 20; c++) {
                double x = Uniform(0, TerrainLength) * TerrainStep;
                const double y = ViewportH / Scale * 3 / 4;
                var poly = new (double X, double Y)[5];
                for (int a = 0; a < poly.Length; a++) {
                    double px = x + 15 * TerrainStep * Math.Sin(3.14 * 2 * a / 5) + Uniform(0, 5 * TerrainStep);
                    double py = y + 5 * TerrainStep * Math.Cos(3.14 * 2 * a / 5) + Uniform(0, 5 * TerrainStep);
                    poly[a] = (px, py);
                }

                // min(p[0] for p in poly) and max(...): the values are finite, so any extreme is Python's.
                double x1 = poly.Min(p => p.X);
                double x2 = poly.Max(p => p.X);
                _cloudPoly.Add((poly, x1, x2));
            }
        }

        /// <summary><c>np.sign</c> of a float64: 1, -1, 0 for either zero (NumPy's sign of -0.0 is +0.0), NaN for NaN.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The sign.</returns>
        private static double NpSign(double value) => value > 0 ? 1.0 : value < 0 ? -1.0 : value == 0 ? 0.0 : double.NaN;

        /// <summary>One <c>np_random.uniform(low, high)</c> draw, a float64.</summary>
        /// <param name="low">The lower bound.</param>
        /// <param name="high">The upper bound.</param>
        /// <returns>The value.</returns>
        private double Uniform(double low, double high) {
            using NDArray drawn = NpRandom.uniform(low, high);
            return drawn.GetAtIndex<double>(0);
        }

        /// <summary>One <c>np_random.integers(low, high)</c> draw, an int64 in [low, high); a one-value range draws nothing.</summary>
        /// <param name="low">The lower bound.</param>
        /// <param name="high">The exclusive upper bound.</param>
        /// <returns>The value.</returns>
        private long Integer(long low, long high) {
            using NDArray drawn = NpRandom.integers(low, high);
            return drawn.GetAtIndex<long>(0);
        }

        /// <summary>One <c>np_random.random()</c> draw, a float64 in [0, 1).</summary>
        /// <returns>The value.</returns>
        private double Random() {
            using NDArray drawn = NpRandom.random();
            return drawn.GetAtIndex<double>(0);
        }

        /// <summary>
        ///     An action read as Gymnasium's step reads it: <c>action[0]</c> to <c>action[3]</c> on demand, with NumPy's errors
        ///     for a 0-d array or a missing element, and every element of the first axis for the energy cost
        ///     (<c>for a in action</c>), each as the NumPy scalar its dtype makes.
        /// </summary>
        private readonly struct MotorAction {
            /// <summary>The action, borrowed from the caller for the step's duration; the caller owns and disposes it.</summary>
            [NDBorrowed]
            private readonly NDArray _action;

            /// <summary>
            ///     Checks an action of more than one dimension before anything is applied, where Gymnasium's first statement
            ///     fails: <c>action[0]</c> is then a row, which <c>np.sign</c> refuses for booleans and <c>float()</c> refuses
            ///     for anything else.
            /// </summary>
            /// <param name="action">The action.</param>
            /// <exception cref="IndexError">Thrown for an array of more than one dimension without rows.</exception>
            /// <exception cref="TypeError">Thrown for an array of more than one dimension with rows, with NumPy 2.4's messages.</exception>
            public MotorAction(NDArray action) {
                if (action.ndim > 1) {
                    if (action.shape[0] == 0)
                        throw new IndexError("index 0 is out of bounds for axis 0 with size 0");
                    if (action.typecode == NPTypeCode.Boolean)
                        throw new TypeError(NoBooleanSign);
                    throw new TypeError("only 0-dimensional arrays can be converted to Python scalars");
                }

                _action = action;
            }

            /// <summary>NumPy's error for <c>np.sign</c> of booleans, which has no boolean loop.</summary>
            private const string NoBooleanSign = "ufunc 'sign' did not contain a loop with signature matching types <class 'numpy.dtypes.BoolDType'> -> None";

            /// <summary>The number of elements <c>for a in action</c> visits (0 for a 0-d array, which fails before it).</summary>
            public int Count => _action.ndim == 0 ? 0 : (int)_action.shape[0];

            /// <summary>
            ///     <c>action[index]</c> as a NumPy scalar of the action's precision, checked as Gymnasium's first use of it
            ///     fails: NumPy's indexing errors, then <c>np.sign</c>'s refusal of a boolean.
            /// </summary>
            /// <param name="index">The index.</param>
            /// <returns>The element.</returns>
            /// <exception cref="IndexError">Thrown for a 0-d action or an index past the end, with NumPy's messages.</exception>
            /// <exception cref="TypeError">Thrown for a boolean action, with NumPy's message.</exception>
            /// <exception cref="NotSupportedException">Thrown for a float16 action, whose half-precision arithmetic is not modelled.</exception>
            public NpScalar this[int index] {
                get {
                    if (_action.ndim == 0)
                        throw new IndexError("too many indices for array: array is 0-dimensional, but 1 were indexed");
                    if (index >= _action.shape[0])
                        throw new IndexError($"index {index} is out of bounds for axis 0 with size {_action.shape[0]}");
                    if (_action.typecode == NPTypeCode.Boolean)
                        throw new TypeError(NoBooleanSign);
                    if (_action.typecode == NPTypeCode.Half)
                        throw new NotSupportedException("A float16 BipedalWalker action is not modelled: NumPy computes its energy cost in half precision.");
                    return NpScalar.Element(_action, index);
                }
            }

            /// <summary>
            ///     <c>np.clip(np.abs(action[index]), 0, 1)</c>: the magnitude clipped to 1 in the element's precision, NaN kept.
            ///     A signed integer's minimum stays negative under <c>np.abs</c> (it wraps), so it clips to 0, as in NumPy.
            /// </summary>
            /// <param name="index">The index.</param>
            /// <returns>The clipped magnitude.</returns>
            /// <exception cref="IndexError">Thrown for a 0-d action or an index past the end.</exception>
            /// <exception cref="TypeError">Thrown for a boolean action.</exception>
            /// <exception cref="NotSupportedException">Thrown for a float16 action.</exception>
            public NpScalar ClippedMagnitude(int index) {
                NpScalar element = this[index];
                if (IsSignedIntegerMinimum(index))
                    return NpScalar.Float64(0.0);
                NpScalar magnitude = element.Abs();
                double clipped = double.IsNaN(magnitude.Value) ? magnitude.Value : Math.Min(Math.Max(magnitude.Value, 0.0), 1.0);
                return magnitude.Kind == NpKind.Float32 ? NpScalar.Float32((float)clipped) : NpScalar.Float64(clipped);
            }

            /// <summary>
            ///     Whether an element is its signed integer dtype's minimum, which <c>np.abs</c> leaves negative; read in the
            ///     dtype itself, since int64's minimum and its neighbour widen to the same double.
            /// </summary>
            /// <param name="index">The element's index, already checked.</param>
            /// <returns><see langword="true"/> for int8 -128, int16 -32768, int32 -2^31 or int64 -2^63.</returns>
            private bool IsSignedIntegerMinimum(int index) => _action.typecode switch {
                NPTypeCode.SByte => _action.GetAtIndex<sbyte>(index) == sbyte.MinValue,
                NPTypeCode.Int16 => _action.GetAtIndex<short>(index) == short.MinValue,
                NPTypeCode.Int32 => _action.GetAtIndex<int>(index) == int.MinValue,
                NPTypeCode.Int64 => _action.GetAtIndex<long>(index) == long.MinValue,
                _ => false,
            };
        }

        /// <summary>
        ///     One lidar ray (Gymnasium's <c>LidarCallback</c>, local to its <c>reset</c>): ignores fixtures outside category 1
        ///     (the walker's own), and clips the ray at each hit, so the last hit kept is the closest.
        /// </summary>
        private sealed class LidarCallback : b2RayCastCallback {
            /// <summary>Python's <c>fraction</c>: 1.0 before the cast, else the closest hit's float32 fraction widened.</summary>
            public double Fraction { get; set; } = 1.0;

            /// <summary>Python's <c>p1</c>: the ray's start (the hull's float32 position), <see langword="null"/> until a step sets it.</summary>
            public (double X, double Y)? P1 { get; set; }

            /// <summary>Python's <c>p2</c>: the ray's float64 end, or the closest hit's float32 point; <see langword="null"/> until a step sets it.</summary>
            public (double X, double Y)? P2 { get; set; }

            /// <summary>Takes a hit on the ground: records its point and fraction and clips the ray there.</summary>
            /// <param name="fixture">The fixture hit.</param>
            /// <param name="point">The hit point.</param>
            /// <param name="normal">The surface normal (unused).</param>
            /// <param name="fraction">The fraction along the ray.</param>
            /// <returns>-1 to ignore a walker fixture, else <paramref name="fraction"/>.</returns>
            public override float ReportFixture(b2Fixture fixture, b2Vec2 point, b2Vec2 normal, float fraction) {
                if ((fixture.FilterData.categoryBits & 1) == 0)
                    return -1;
                // pybox2d passes the point as a tuple of the float32 coordinates.
                P2 = (point.x, point.y);
                Fraction = fraction;
                return fraction;
            }
        }

        /// <summary>
        ///     Gymnasium's <c>ContactDetector</c>, private to the env (DR-28): the hull touching the ground ends the game; a lower
        ///     leg's contact sets its <c>ground_contact</c> and the contact's end clears it.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/bipedal_walker.py#L82">gymnasium/envs/box2d/bipedal_walker.py</see>.</remarks>
        private sealed class ContactDetector : b2ContactListener {
            /// <summary>The env whose flags the contacts set; borrowed, the env owns its arrays and disposes them.</summary>
            [NDBorrowed]
            private readonly BipedalWalkerEnv _env;

            /// <summary>A detector for an env.</summary>
            /// <param name="env">The env.</param>
            public ContactDetector(BipedalWalkerEnv env) => _env = env;

            /// <summary>Sets <c>game_over</c> for the hull and <c>ground_contact</c> for a lower leg.</summary>
            /// <param name="contact">The contact.</param>
            public override void BeginContact(b2Contact contact) {
                b2Body a = contact.FixtureA.Body, b = contact.FixtureB.Body;
                if (_env.Hull == a || _env.Hull == b)
                    _env.GameOver = true;
                foreach (int i in new[] { 1, 3 }) {
                    if (_env._legs[i] == a || _env._legs[i] == b)
                        _env._groundContact[i] = true;
                }
            }

            /// <summary>Clears <c>ground_contact</c> for a lower leg.</summary>
            /// <param name="contact">The contact.</param>
            public override void EndContact(b2Contact contact) {
                b2Body a = contact.FixtureA.Body, b = contact.FixtureB.Body;
                foreach (int i in new[] { 1, 3 }) {
                    if (_env._legs[i] == a || _env._legs[i] == b)
                        _env._groundContact[i] = false;
                }
            }
        }
    }

    /// <summary>
    ///     Gymnasium's <c>BipedalWalkerHardcore</c>: a class that only refuses to be built, pointing at
    ///     <c>make("BipedalWalker-v3", hardcore=True)</c> (DR-32).
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/bipedal_walker.py#L773">gymnasium/envs/box2d/bipedal_walker.py</see>.</remarks>
    [PyName("BipedalWalkerHardcore")]
    public sealed class BipedalWalkerHardcoreEnv {
        /// <summary>Always refuses, with Gymnasium's message.</summary>
        /// <exception cref="Error">Always: the hardcore mode is a <c>hardcore=True</c> kwarg of BipedalWalker.</exception>
        public BipedalWalkerHardcoreEnv() =>
            throw new Error("Error initializing BipedalWalkerHardcore Environment.\n"
                            + "Currently, we do not support initializing this mode of environment by calling the class directly.\n"
                            + "To use this environment, instead create it by specifying the hardcore keyword in gym.make, i.e.\n"
                            + "gym.make(\"BipedalWalker-v3\", hardcore=True)");
    }
}
