#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using BigInteger = System.Numerics.BigInteger;
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
    ///     CarRacing-v3: a top-down race on a random closed track, learned from pixels. The observation is a 96×96 RGB image
    ///     of the car and the track around it (with the speed, ABS, steering and gyro indicators and the score at the bottom);
    ///     the reward is -0.1 per frame plus 1000/N for each of the track's N tiles first touched, and -100 when the car
    ///     leaves the playfield. Continuous actions are <c>Box([-1, 0, 0], [1, 1, 1], float32)</c> (steer, gas, brake);
    ///     discrete ones <c>Discrete(5)</c> (nothing, right, left, gas, brake). Step for step and frame for frame what
    ///     Gymnasium's <c>CarRacing</c> computes on Box2D 2.3.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The physics runs on <see cref="b2World"/>, Gym.NET's managed Box2D 2.3.0 (D5), with no gravity and Gymnasium's
    ///     <see cref="Car"/>; one world serves the env's whole life, as in Gymnasium, so (as for BipedalWalker) an episode
    ///     after other episodes can differ from a fresh env's after the same <c>reset(seed)</c>. The track is 12 random
    ///     checkpoints joined by a turning path of 21/6 m steps, regenerated until it closes; each step of it is a sensor tile
    ///     the <see cref="FrictionDetector"/> watches, recolouring touched tiles and paying for first visits.
    ///     </para>
    ///     <para>
    ///     Every step renders the observation, as Gymnasium does: a 1000×800 frame of the road and grass (anti-aliased and
    ///     filled polygons), the car, the indicators and the score, shrunk with pygame's <c>smoothscale</c>. The score is
    ///     pygame's default font at size 42 laid out as SDL_ttf lays it out, from glyph data recorded from pygame and embedded
    ///     in this package (DR-27, DR-48), for every text <c>f"{reward:04.0f}"</c> can show (NaN and the infinities included).
    ///     Python's attributes are properties under
    ///     their names (<c>tile_visited_count</c> is <see cref="TileVisitedCount"/>); the tiles' attributes live in
    ///     <see cref="RoadTile"/>. The class prints as Gymnasium's <c>CarRacing</c> (DR-32).
    ///     </para>
    ///     <para>
    ///     Not thread-safe; each env owns its world. Reference:
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/car_racing.py#L106">gymnasium/envs/box2d/car_racing.py</see>.
    ///     </para>
    /// </remarks>
    [GymOptions]
    [PyName("CarRacing")]
    public partial class CarRacingEnv : Env<NDArray, NDArray> {
        /// <summary>Gymnasium's <c>STATE_W</c>: the observation's width in pixels.</summary>
        public const int StateW = 96;

        /// <summary>Gymnasium's <c>STATE_H</c>: the observation's height in pixels.</summary>
        public const int StateH = 96;

        /// <summary>Gymnasium's <c>VIDEO_W</c>: the <c>rgb_array</c> frame's width.</summary>
        public const int VideoW = 600;

        /// <summary>Gymnasium's <c>VIDEO_H</c>: the <c>rgb_array</c> frame's height.</summary>
        public const int VideoH = 400;

        /// <summary>Gymnasium's <c>WINDOW_W</c>: the frame every mode draws first, and the human window, in pixels.</summary>
        public const int WindowW = 1000;

        /// <summary>Gymnasium's <c>WINDOW_H</c>.</summary>
        public const int WindowH = 800;

        /// <summary>Gymnasium's <c>SCALE</c>: the track's scale.</summary>
        public const double Scale = 6.0;

        /// <summary>Gymnasium's <c>TRACK_RAD</c>: the morphed circle's radius the checkpoints scatter around, in metres.</summary>
        public const double TrackRad = 900 / Scale;

        /// <summary>Gymnasium's <c>PLAYFIELD</c>: the game-over boundary, in metres from the origin on either axis.</summary>
        public const double Playfield = 2000 / Scale;

        /// <summary>Gymnasium's <c>FPS</c>: steps per simulated second and the human frame rate.</summary>
        public const int Fps = 50;

        /// <summary>Gymnasium's <c>ZOOM</c>: the camera's zoom after its first second.</summary>
        public const double Zoom = 2.7;

        /// <summary>Gymnasium's <c>ZOOM_FOLLOW</c>: kept for its name; Gymnasium's renderer always follows.</summary>
        public const bool ZoomFollow = true;

        /// <summary>Gymnasium's <c>TRACK_DETAIL_STEP</c>: the length of one track step, in metres.</summary>
        public const double TrackDetailStep = 21 / Scale;

        /// <summary>Gymnasium's <c>TRACK_TURN_RATE</c>: the most the path turns per step, in radians.</summary>
        public const double TrackTurnRate = 0.31;

        /// <summary>Gymnasium's <c>TRACK_WIDTH</c>: the road's half-width, in metres.</summary>
        public const double TrackWidth = 40 / Scale;

        /// <summary>Gymnasium's <c>BORDER</c>: the red-white kerb's width on hard turns, in metres.</summary>
        public const double Border = 8 / Scale;

        /// <summary>Gymnasium's <c>BORDER_MIN_COUNT</c>: the consecutive sharp steps that make a kerb.</summary>
        public const int BorderMinCount = 4;

        /// <summary>Gymnasium's <c>GRASS_DIM</c>: a grass patch's side, in metres.</summary>
        public const double GrassDim = Playfield / 20.0;

        /// <summary>
        ///     Gymnasium's <c>MAX_SHAPE_DIM</c>: the largest object's diagonal on screen, the margin of the off-screen test that
        ///     skips drawing far polygons.
        /// </summary>
        public static readonly double MaxShapeDim = Math.Max(Math.Max(GrassDim, TrackWidth), TrackDetailStep) * Math.Sqrt(2) * Zoom * Scale;

        /// <summary>The human-mode window, opened on the first human render.</summary>
        private readonly HumanWindow _window;

        /// <summary>Gymnasium's <c>road_color</c> (and whether it is NumPy's int64 default rather than a random float64 draw).</summary>
        private (double[] Values, bool IsInt) _roadColor;

        /// <summary>Gymnasium's <c>bg_color</c>.</summary>
        private (double[] Values, bool IsInt) _bgColor;

        /// <summary>Gymnasium's <c>grass_color</c>.</summary>
        private (double[] Values, bool IsInt) _grassColor;

        /// <summary>Gymnasium's <c>road</c>: the tiles, <see langword="null"/> before the first reset.</summary>
        private List<RoadTile>? _road;

        /// <summary>Gymnasium's <c>road_poly</c>: each tile's and kerb's polygon with its colour (a tile's colour array is the tile's own).</summary>
        private List<((double X, double Y)[] Poly, double[] Color)> _roadPoly = new();

        /// <summary>Gymnasium's <c>track</c>: the closed path's steps, <see langword="null"/> before the first reset.</summary>
        private List<(double Alpha, double Beta, double X, double Y)>? _track;

        /// <summary>Gymnasium's <c>t</c>, <see langword="null"/> while the attribute does not exist (before the first reset).</summary>
        private double? _t;

        /// <summary>Gymnasium's <c>state</c>: the last observation, kept as the env's own copy; <see langword="null"/> before the first step.</summary>
        private NDArray? _state;

        /// <summary>Creates the env.</summary>
        /// <param name="renderMode">Gymnasium's <c>render_mode</c>: <see langword="null"/>, <c>"human"</c>, <c>"rgb_array"</c> or <c>"state_pixels"</c>.</param>
        /// <param name="verbose">Gymnasium's <c>verbose</c>: print the track generation's progress.</param>
        /// <param name="lapCompletePercent">Gymnasium's <c>lap_complete_percent</c>: the share of tiles visited that, back at the start tile, completes the lap.</param>
        /// <param name="domainRandomize">Gymnasium's <c>domain_randomize</c>: random road, background and grass colours, drawn now and at resets.</param>
        /// <param name="continuous">Gymnasium's <c>continuous</c>: three continuous controls instead of five discrete actions.</param>
        /// <param name="viewerFactory">Gym.NET's <c>viewer_factory</c> (DR-20) for human mode; <see langword="null"/> uses <c>Gymnasium.HumanViewerFactory</c>.</param>
        /// <remarks>
        ///     With <paramref name="domainRandomize"/> the constructor draws the colours from the env's generator, which no seed
        ///     has reached yet (it is entropy-seeded), as Gymnasium's does; a reset draws them again unless told not to.
        /// </remarks>
        public CarRacingEnv(string? renderMode = null, bool verbose = false, double lapCompletePercent = 0.95, bool domainRandomize = false,
                            bool continuous = true, IEnvironmentViewerFactoryDelegate? viewerFactory = null) {
            Metadata = ClassMetadata;
            Continuous = continuous;
            DomainRandomize = domainRandomize;
            LapCompletePercent = lapCompletePercent;
            InitColors();

            ContactListenerKeepref = new FrictionDetector(this, LapCompletePercent);
            World = new b2World((0, 0));
            World.ContactListener = ContactListenerKeepref;
            IsOpen = true;
            Reward = 0.0;
            PrevReward = 0.0;
            Verbose = verbose;
            NewLap = false;
            FdTile = new b2FixtureDef(new b2PolygonShape(new b2Vec2[] { (0, 0), (1, 0), (1, -1), (0, -1) }));

            // This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
            //   or normalised however this is not possible here so ignore
            if (Continuous) {
                using NDArray low = np.array(new[] { -1f, 0f, 0f });
                using NDArray high = np.array(new[] { +1f, +1f, +1f });
                ActionSpace = new Box(low, high); // steer, gas, brake
            } else {
                ActionSpace = new Discrete(5); // do nothing, right, left, gas, brake
            }

            ObservationSpace = new Box(0, 255, new Shape(StateH, StateW, 3), np.uint8);
            RenderMode = renderMode;
            _window = new HumanWindow("carracing-v3", viewerFactory);
        }

        /// <summary>
        ///     Gymnasium's class attribute <c>metadata</c>:
        ///     <c>{"render_modes": ["human", "rgb_array", "state_pixels"], "render_fps": 50}</c>.
        /// </summary>
        /// <remarks>A new dict on every read, so editing one env's metadata never reaches another.</remarks>
        public static new Metadata ClassMetadata => new() {
            [Metadata.RenderModesKey] = new List<string> { "human", "rgb_array", "state_pixels" },
            [Metadata.RenderFpsKey] = (long)Fps,
        };

        /// <summary>Gymnasium's <c>continuous</c>, fixed at construction.</summary>
        public bool Continuous { get; }

        /// <summary>Gymnasium's <c>domain_randomize</c>: whether resets draw new colours.</summary>
        public bool DomainRandomize { get; set; }

        /// <summary>Gymnasium's <c>lap_complete_percent</c>, which the next reset's contact listener takes.</summary>
        public double LapCompletePercent { get; set; }

        /// <summary>Gymnasium's <c>contactListener_keepref</c>: the constructor's contact listener, replaced in the world by each reset's.</summary>
        public FrictionDetector ContactListenerKeepref { get; }

        /// <summary>Gymnasium's <c>world</c>: the Box2D world without gravity, one for the env's life.</summary>
        public b2World World { get; }

        /// <summary>Gymnasium's <c>verbose</c>.</summary>
        public bool Verbose { get; set; }

        /// <summary>Gymnasium's <c>reward</c>: the episode's running score (the text the frames show), which the contact listener raises.</summary>
        public double Reward { get; set; }

        /// <summary>Gymnasium's <c>prev_reward</c>: the score at the last step, whose difference is the step's reward.</summary>
        public double PrevReward { get; set; }

        /// <summary>Gymnasium's <c>new_lap</c>: set by the contact listener when the start tile is reached with enough of the track visited.</summary>
        public bool NewLap { get; set; }

        /// <summary>Gymnasium's <c>tile_visited_count</c>: tiles visited this episode.</summary>
        public long TileVisitedCount { get; set; }

        /// <summary>Gymnasium's <c>fd_tile</c>: the tiles' fixture definition, whose polygon each tile overwrites before its body is made.</summary>
        public b2FixtureDef FdTile { get; }

        /// <summary>Gymnasium's <c>road</c>: the track's tiles in order; <see langword="null"/> before the first reset.</summary>
        public IReadOnlyList<RoadTile>? Road => _road;

        /// <summary>Gymnasium's <c>car</c>: the car, <see langword="null"/> before the first reset; stepping requires it.</summary>
        public Car? Car { get; private set; }

        /// <summary>
        ///     Gymnasium's <c>track</c>: the closed path's steps <c>(alpha, beta, x, y)</c>, the first being the start.
        /// </summary>
        /// <exception cref="AttributeError">Thrown before the first reset, as Python's attribute does not exist yet.</exception>
        public IReadOnlyList<(double Alpha, double Beta, double X, double Y)> Track =>
            _track ?? throw new AttributeError($"'{PythonClassName}' object has no attribute 'track'");

        /// <summary>Gymnasium's <c>start_alpha</c>: the angle the track's closing is cut at, set by the first track generation.</summary>
        public double StartAlpha { get; private set; }

        /// <summary>
        ///     Gymnasium's <c>road_poly</c>: each tile's polygon (in metres) with its colour, followed by a kerb's where the tile
        ///     has one. A tile's colour is the tile's own array, which the contact listener overwrites, so touched tiles lose
        ///     their shade.
        /// </summary>
        public IReadOnlyList<(IReadOnlyList<(double X, double Y)> Poly, IReadOnlyList<double> Color)> RoadPoly =>
            _roadPoly.Select(p => ((IReadOnlyList<(double X, double Y)>)p.Poly.ToArray(), (IReadOnlyList<double>)p.Color.ToArray())).ToList();

        /// <summary>Gymnasium's <c>road_color</c>: an int64 array by default, float64 when randomized; a new array on each read.</summary>
        public NDArray RoadColor => ColorArray(_roadColor);

        /// <summary>The road colour's channels, as the contact listener copies them into a touched tile (no array allocated).</summary>
        internal double[] RoadColorValues => _roadColor.Values;

        /// <summary>Gymnasium's <c>bg_color</c>: the background's colour, as <see cref="RoadColor"/>.</summary>
        public NDArray BgColor => ColorArray(_bgColor);

        /// <summary>Gymnasium's <c>grass_color</c>: the grass patches' colour, as <see cref="RoadColor"/>.</summary>
        public NDArray GrassColor => ColorArray(_grassColor);

        /// <summary>Gymnasium's <c>t</c>: the episode's simulated time in seconds, which drives the first second's zoom.</summary>
        /// <exception cref="AttributeError">Thrown before the first reset.</exception>
        public double T {
            get => _t ?? throw new AttributeError($"'{PythonClassName}' object has no attribute 't'");
            set => _t = value;
        }

        /// <summary>Gymnasium's <c>state</c>: the last observation, a new copy on each read.</summary>
        /// <exception cref="AttributeError">Thrown before the first step, as Python's attribute does not exist yet.</exception>
        public NDArray State => _state?.copy() ?? throw new AttributeError($"'{PythonClassName}' object has no attribute 'state'");

        /// <summary>Gymnasium's <c>screen</c>: the human window's surface, created by the first human render.</summary>
        public Surface? Screen { get; private set; }

        /// <summary>Gymnasium's <c>clock</c>: paces human rendering, created by the first render.</summary>
        public Clock? Clock { get; private set; }

        /// <summary>Gymnasium's <c>surf</c>: the last 1000×800 frame drawn (by any mode, the observation's included).</summary>
        public Surface? Surf { get; private set; }

        /// <summary>Gymnasium's <c>isopen</c>: <see langword="true"/> until <see cref="Close"/> shuts a human window down.</summary>
        [PyName("isopen")]
        public bool IsOpen { get; set; }

        /// <summary>Gymnasium's <c>invisible_state_window</c>: always <see langword="null"/> (Python's <c>None</c>, never used).</summary>
        public object? InvisibleStateWindow => null;

        /// <summary>Gymnasium's <c>invisible_video_window</c>: always <see langword="null"/>.</summary>
        public object? InvisibleVideoWindow => null;

        /// <summary>
        ///     Starts an episode, as Gymnasium's does in its order: reseeds the generator when a seed is given, destroys the last
        ///     episode's tiles and car (the old listener sees their contacts end), installs a new <see cref="FrictionDetector"/>,
        ///     redraws the colours when randomizing, generates tracks until one closes, builds the car on the first step, renders
        ///     in human mode, and returns the observation of a step without an action.
        /// </summary>
        /// <param name="seed">The seed of a fresh <c>Generator(PCG64(SeedSequence(seed)))</c>, or <see langword="null"/> to keep the generator.</param>
        /// <param name="options">
        ///     Gymnasium's options: with <c>domain_randomize</c>, <c>{"randomize": False}</c> keeps the current colours (any value
        ///     is read by Python's truth value); without it they are ignored.
        /// </param>
        /// <returns>The first observation and an empty info.</returns>
        /// <exception cref="Error">Thrown for a negative seed, with Gymnasium's message.</exception>
        /// <exception cref="DependencyNotInstalled">Thrown in human mode by the first render when no viewer factory is set (DR-20).</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/car_racing.py#L497">gymnasium/envs/box2d/car_racing.py</see>.</remarks>
        public override ResetResult<NDArray> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            SeedNpRandom(seed);
            Destroy();
            World.ContactListener = new FrictionDetector(this, LapCompletePercent);
            Reward = 0.0;
            PrevReward = 0.0;
            TileVisitedCount = 0;
            _t = 0.0;
            NewLap = false;
            _roadPoly = new List<((double X, double Y)[] Poly, double[] Color)>();

            if (DomainRandomize) {
                bool randomize = true;
                if (options is not null && options.TryGetValue("randomize", out object? value))
                    randomize = PyTruth(value);
                ReinitColors(randomize);
            }

            while (true) {
                bool success = CreateTrack();
                if (success)
                    break;
                if (Verbose)
                    Console.WriteLine("retry to generate track (normal if there are not manyinstances of this message)");
            }

            // Car(self.world, *self.track[0][1:4]): the first step's beta, x and y.
            (_, double beta, double x, double y) = _track![0];
            Car = new Car(World, beta, x, y);

            if (RenderMode == "human")
                Render();
            StepResult<NDArray> first = StepCore(null);
            return new ResetResult<NDArray>(first.Observation, new Info());
        }

        /// <summary>
        ///     One step, in Gymnasium's order: the action's controls (continuous: steer by <c>-action[0]</c>, gas, brake; discrete:
        ///     checked against <c>Discrete(5)</c>, then fixed controls), the car's wheel physics, one 1/50 s world step (180 and 60
        ///     iterations, the contact listener paying for new tiles), the observation's frame, then -0.1 on the score, the
        ///     step's reward, and the lap's or the playfield's end. Renders in human mode.
        /// </summary>
        /// <param name="action">
        ///     Continuous: three controls (read as float64, unclipped except the gas); discrete: an element of <c>Discrete(5)</c>
        ///     (a 0-d integer array; the untyped step also takes a .NET integer). Read, never kept.
        /// </param>
        /// <returns>
        ///     The 96×96 RGB observation, the reward (the score's change, or -100 off the playfield), whether the episode ended,
        ///     <c>truncated=False</c>, and the info (<c>lap_finished</c> when it ended).
        /// </returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="action"/> is <see langword="null"/> (only the reset's own step goes without one).</exception>
        /// <exception cref="AssertionError">Thrown before the first reset (Python's bare <c>assert</c>; N-1 for its message).</exception>
        /// <exception cref="InvalidAction">Thrown in discrete mode for an action outside <c>Discrete(5)</c>, with Gymnasium's message.</exception>
        /// <exception cref="IndexError">Thrown in continuous mode for a 0-d action or one of fewer than three elements, with NumPy's message, after the controls before the missing one were applied.</exception>
        /// <exception cref="NotSupportedException">Thrown in continuous mode for an action of more than one dimension (NumPy's array arithmetic on the wheels is not modelled).</exception>
        /// <exception cref="DependencyNotInstalled">Thrown in human mode by the first render when no viewer factory is set (DR-20).</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/car_racing.py#L539">gymnasium/envs/box2d/car_racing.py</see>.</remarks>
        public override StepResult<NDArray> Step(NDArray action) => StepCore(action ?? throw new ArgumentNullException(nameof(action)));

        /// <summary>
        ///     Renders the current mode's frame (Gymnasium's <c>render()</c>): nothing and a warning without a render mode; the
        ///     frame of <see cref="Env{TObs, TAct}.RenderMode"/> otherwise.
        /// </summary>
        /// <returns><c>rgb_array</c>: the <c>(400, 600, 3)</c> frame; <c>state_pixels</c>: the <c>(96, 96, 3)</c> observation's frame; otherwise nothing.</returns>
        /// <exception cref="AssertionError">Thrown without a render mode when the env has no spec, or for a mode the metadata does not list.</exception>
        /// <exception cref="DependencyNotInstalled">Thrown on the first human render when no viewer factory is set (DR-20).</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/car_racing.py#L588">gymnasium/envs/box2d/car_racing.py</see>.</remarks>
        public override RenderResult Render() {
            if (RenderMode is null) {
                EnvSpec spec = Spec ?? throw new AssertionError();
                Logger.Warn("You are calling render method without specifying any render mode. You can specify the render_mode at initialization, "
                            + $"e.g. gym.make(\"{spec.Id}\", render_mode=\"rgb_array\")");
                return RenderResult.None;
            }

            NDArray? frame = RenderFrame(RenderMode);
            return frame is null ? RenderResult.None : new RenderResult.Rgb(frame);
        }

        /// <summary>Shuts the human window down once one was opened, and marks the env closed (<see cref="IsOpen"/>).</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/car_racing.py#L791">gymnasium/envs/box2d/car_racing.py</see>.</remarks>
        public override void Close() {
            if (Screen is not null) {
                // pygame.display.quit(); self.isopen = False; pygame.quit()
                _window.Close();
                IsOpen = false;
            }
        }

        /// <summary>
        ///     Gymnasium's demo keyboard handler (<c>register_input</c>): reads the pending window events and sets the demo's
        ///     controls from the arrow keys (left/right steer -1/+1, up gas 1, down brake 0.8, cleared on release), and
        ///     requests a restart on Return and quitting on Escape or the window's close button.
        /// </summary>
        /// <param name="a">The demo's action, three float64 controls, written in place.</param>
        /// <param name="quit">Set when the demo should end; never cleared here.</param>
        /// <param name="restart">Set when the episode should restart; never cleared here.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="a"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown when <paramref name="a"/> is not a float64 array of at least three elements.</exception>
        /// <remarks>
        ///     Python's function writes the module's globals <c>a</c>, <c>quit</c> and <c>restart</c>; here they are the caller's.
        ///     Events come from <see cref="EventQueue"/>, the queue the interactive viewers fill (DR-41). Reference:
        ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/car_racing.py#L801">gymnasium/envs/box2d/car_racing.py</see>.
        /// </remarks>
        public static void RegisterInput(NDArray a, ref bool quit, ref bool restart) {
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            // SetDouble writes raw float64 elements, so any other dtype or a short array is refused before an event is read.
            if (a.typecode != NPTypeCode.Double || a.size < 3)
                throw new ArgumentException("The demo's action is a float64 array of three controls (steer, gas, brake).", nameof(a));
            foreach (Event e in EventQueue.Get()) {
                long key = e.Dict.TryGetValue("key", out object? k) && k is not null ? Convert.ToInt64(k, CultureInfo.InvariantCulture) : -1;
                if (e.Type == EventType.KeyDown) {
                    if (key == KeyCode.Left)
                        a.SetDouble(-1.0, 0L);
                    if (key == KeyCode.Right)
                        a.SetDouble(+1.0, 0L);
                    if (key == KeyCode.Up)
                        a.SetDouble(+1.0, 1L);
                    if (key == KeyCode.Down)
                        a.SetDouble(+0.8, 2L); // set 1.0 for wheels to block to zero rotation
                    if (key == KeyCode.Return)
                        restart = true;
                    if (key == KeyCode.Escape)
                        quit = true;
                }

                if (e.Type == EventType.KeyUp) {
                    if (key == KeyCode.Left)
                        a.SetDouble(0, 0L);
                    if (key == KeyCode.Right)
                        a.SetDouble(0, 0L);
                    if (key == KeyCode.Up)
                        a.SetDouble(0, 1L);
                    if (key == KeyCode.Down)
                        a.SetDouble(0, 2L);
                }

                if (e.Type == EventType.Quit)
                    quit = true;
            }
        }

        /// <summary>The Python class this instance stands for: Gymnasium's <c>CarRacing</c> (DR-32).</summary>
        protected override string PythonClassName => "CarRacing";

        /// <summary>Releases the kept observation along with the env's spaces.</summary>
        /// <param name="disposing">Whether managed resources are released.</param>
        protected override void Dispose(bool disposing) {
            if (disposing) {
                _state?.Dispose();
                _state = null;
            }

            base.Dispose(disposing);
        }

        /// <summary>
        ///     The step's body, with Gymnasium's <c>action is None</c> case for the reset's first step: no controls, no score
        ///     change, reward 0.
        /// </summary>
        /// <param name="action">The action, or <see langword="null"/> for the reset's step.</param>
        /// <returns>The step's result.</returns>
        /// <exception cref="AssertionError">Thrown before the first reset.</exception>
        /// <exception cref="InvalidAction">Thrown for a discrete action outside the space.</exception>
        /// <exception cref="IndexError">Thrown for a continuous action without enough elements.</exception>
        /// <exception cref="NotSupportedException">Thrown for a continuous action of more than one dimension.</exception>
        private StepResult<NDArray> StepCore(NDArray? action) {
            Car car = Car ?? throw new AssertionError();
            if (action is not null) {
                if (Continuous) {
                    if (action.ndim > 1)
                        throw new NotSupportedException($"A continuous CarRacing action of {action.ndim} dimensions is not modelled (NumPy's array arithmetic on the wheels).");
                    // action = action.astype(np.float64), then action[0], [1], [2] as the controls need them.
                    using NDArray controls = action.astype(np.float64);
                    car.Steer(-Control(controls, 0));
                    car.Gas(Control(controls, 1));
                    car.Brake(Control(controls, 2));
                } else {
                    if (!ActionSpace.Contains(action))
                        throw new InvalidAction($"you passed the invalid action `{PyBuiltins.Str(action)}`. The supported action_space is `{ActionSpace}`");
                    long chosen = Discrete.ToInt64(action);
                    // -0.6 * (action == 1) + 0.6 * (action == 2): Python's bools are 0 and 1, so action 0 steers by -0.0 + 0.0.
                    car.Steer(-0.6 * (chosen == 1 ? 1 : 0) + 0.6 * (chosen == 2 ? 1 : 0));
                    car.Gas(0.2 * (chosen == 3 ? 1 : 0));
                    car.Brake(0.8 * (chosen == 4 ? 1 : 0));
                }
            }

            car.Step(1.0 / Fps);
            World.Step(1.0 / Fps, 6 * 30, 2 * 30);
            _t = T + 1.0 / Fps;

            NDArray state = RenderFrame("state_pixels")!;
            _state?.Dispose();
            _state = state;

            double stepReward = 0;
            bool terminated = false;
            var info = new Info();
            if (action is not null) { // First step without action, called from reset()
                Reward -= 0.1;
                // We actually don't want to count fuel spent, we want car to be faster.
                // self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
                car.FuelSpent = 0.0;
                stepReward = Reward - PrevReward;
                PrevReward = Reward;
                if (TileVisitedCount == _track!.Count || NewLap) {
                    // Termination due to finishing lap
                    terminated = true;
                    info["lap_finished"] = true;
                }

                b2Vec2 position = car.Hull!.Position;
                if (Math.Abs((double)position.x) > Playfield || Math.Abs((double)position.y) > Playfield) {
                    terminated = true;
                    info["lap_finished"] = false;
                    stepReward = -100;
                }
            }

            if (RenderMode == "human")
                Render();
            return new StepResult<NDArray>(state.copy(), stepReward, terminated, false, info);
        }

        /// <summary><c>action[index]</c> of a 1-d float64 action, with NumPy's errors for a 0-d array or a missing element.</summary>
        /// <param name="controls">The action as float64.</param>
        /// <param name="index">The index.</param>
        /// <returns>The control.</returns>
        /// <exception cref="IndexError">Thrown for a 0-d action or an index past the end.</exception>
        private static double Control(NDArray controls, int index) {
            if (controls.ndim == 0)
                throw new IndexError("too many indices for array: array is 0-dimensional, but 1 were indexed");
            if (index >= controls.shape[0])
                throw new IndexError($"index {index} is out of bounds for axis 0 with size {controls.shape[0]}");
            return controls.GetAtIndex<double>(index);
        }

        /// <summary>
        ///     Gymnasium's <c>_render(mode)</c>: draws the 1000×800 frame (road, car, the flip, indicators, score) and returns
        ///     the mode's image; in human mode shows it instead. Before the first reset it opens the window and the clock and
        ///     returns nothing, as Python's early <c>return</c> does.
        /// </summary>
        /// <param name="mode">The mode, one the metadata lists.</param>
        /// <returns><c>rgb_array</c>: the frame shrunk to 600×400; <c>state_pixels</c>: shrunk to 96×96; otherwise <see langword="null"/>.</returns>
        /// <exception cref="AssertionError">Thrown for a mode the metadata does not list.</exception>
        /// <exception cref="DependencyNotInstalled">Thrown on the first human render when no viewer factory is set (DR-20).</exception>
        private NDArray? RenderFrame(string mode) {
            if (!(Metadata[Metadata.RenderModesKey] is IEnumerable<string> modes && modes.Contains(mode)))
                throw new AssertionError();
            // pygame.font.init(): nothing to initialise here.
            if (Screen is null && mode == "human") {
                // pygame.display.init(); self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
                _window.Ensure(WindowW, WindowH);
                Screen = new Surface(WindowW, WindowH);
            }

            Clock ??= new Clock();
            if (_t is not { } t)
                return null; // reset() not called yet

            var surf = new Surface(WindowW, WindowH);
            Surf = surf;
            Car car = Car!;

            // computing transformations
            double angle = -(double)car.Hull!.Angle;
            // Animating first second zoom.
            double zoom = 0.1 * Scale * (0 > 1 - t ? 0 : 1 - t) + Zoom * Scale * (1 < t ? 1 : t);
            b2Vec2 position = car.Hull.Position;
            double scrollX = -(double)position.x * zoom;
            double scrollY = -(double)position.y * zoom;
            Vector2 rotated = new Vector2(scrollX, scrollY).RotateRad(angle);
            (double X, double Y) trans = (WindowW / 2 + rotated.X, WindowH / 4 + rotated.Y);

            RenderRoad(surf, zoom, trans, angle);
            car.Draw(surf, zoom, trans, angle, mode != "state_pixels_list" && mode != "state_pixels");

            surf = Transform.Flip(surf, false, true);
            Surf = surf;

            // showing stats
            RenderIndicators(surf, WindowW, WindowH);
            RenderScore(surf);

            if (mode == "human") {
                // pygame.event.pump() has no queue to pump here; the viewer runs its own message loop.
                Clock.Tick(PyBuiltins.Float(Metadata[Metadata.RenderFpsKey]));
                Surface screen = Screen!;
                screen.Fill(new Color(0, 0, 0));
                screen.Blit(surf, 0, 0);
                _window.Show(screen);
                return null;
            }

            if (mode == "rgb_array")
                return CreateImageArray(surf, VideoW, VideoH);
            if (mode == "state_pixels")
                return CreateImageArray(surf, StateW, StateH);
            return null;
        }

        /// <summary>
        ///     Gymnasium's <c>_render_road</c>: the playfield's background (never culled), the 20×20 grass patches, and the road
        ///     and kerb polygons, each culled when no vertex lies within <see cref="MaxShapeDim"/> of the frame.
        /// </summary>
        /// <param name="surf">The frame.</param>
        /// <param name="zoom">Pixels per metre.</param>
        /// <param name="translation">The screen offset.</param>
        /// <param name="angle">The rotation.</param>
        private void RenderRoad(Surface surf, double zoom, (double X, double Y) translation, double angle) {
            const double bounds = Playfield;
            (double X, double Y)[] field = { (bounds, bounds), (bounds, -bounds), (-bounds, -bounds), (-bounds, bounds) };

            // draw background
            DrawColoredPolygon(surf, field, _bgColor.Values, zoom, translation, angle, clip: false);

            // draw grass patches
            var grass = new List<(double X, double Y)[]>();
            for (int x = -20; x < 20; x += 2) {
                for (int y = -20; y < 20; y += 2) {
                    grass.Add(new[] {
                        (GrassDim * x + GrassDim, GrassDim * y + 0), (GrassDim * x + 0, GrassDim * y + 0),
                        (GrassDim * x + 0, GrassDim * y + GrassDim), (GrassDim * x + GrassDim, GrassDim * y + GrassDim),
                    });
                }
            }

            foreach ((double X, double Y)[] poly in grass)
                DrawColoredPolygon(surf, poly, _grassColor.Values, zoom, translation, angle);

            // draw road
            foreach (((double X, double Y)[] poly, double[] color) in _roadPoly) {
                // color = [int(c) for c in color]: Python's int() truncates.
                double[] truncated = color.Select(c => Math.Truncate(c)).ToArray();
                DrawColoredPolygon(surf, poly, truncated, zoom, translation, angle);
            }
        }

        /// <summary>
        ///     Gymnasium's <c>_render_indicators</c>: the black bar along the bottom, then the speed, the four wheels' spin, the
        ///     front wheel's steering and the hull's rotation as bars, each drawn only when its value exceeds 1e-4 in size.
        /// </summary>
        /// <param name="surf">The frame, already flipped.</param>
        /// <param name="w">The frame's width.</param>
        /// <param name="h">The frame's height.</param>
        private void RenderIndicators(Surface surf, int w, int h) {
            double s = w / 40.0;
            double hh = h / 40.0;
            var color = new Color(0, 0, 0);
            Draw.Polygon(surf, color, new List<(double X, double Y)> { (w, h), (w, h - 5 * hh), (0, h - 5 * hh), (0, h) });

            List<(double X, double Y)> VerticalInd(int place, double val) => new() {
                (place * s, h - (hh + hh * val)), ((place + 1) * s, h - (hh + hh * val)), ((place + 1) * s, h - hh), ((place + 0) * s, h - hh),
            };

            List<(double X, double Y)> HorizInd(int place, double val) => new() {
                ((place + 0) * s, h - 4 * hh), ((place + val) * s, h - 4 * hh), ((place + val) * s, h - 2 * hh), ((place + 0) * s, h - 2 * hh),
            };

            Car car = Car!;
            b2Vec2 velocity = car.Hull!.LinearVelocity;
            double trueSpeed = Math.Sqrt((double)velocity.x * velocity.x + (double)velocity.y * velocity.y);

            // simple wrapper to render if the indicator value is above a threshold
            void RenderIfMin(double value, List<(double X, double Y)> points, Color c) {
                if (Math.Abs(value) > 1e-4)
                    Draw.Polygon(surf, c, points);
            }

            RenderIfMin(trueSpeed, VerticalInd(5, 0.02 * trueSpeed), new Color(255, 255, 255));
            // ABS sensors
            RenderIfMin(car.Wheels[0].Omega, VerticalInd(7, 0.01 * car.Wheels[0].Omega), new Color(0, 0, 255));
            RenderIfMin(car.Wheels[1].Omega, VerticalInd(8, 0.01 * car.Wheels[1].Omega), new Color(0, 0, 255));
            RenderIfMin(car.Wheels[2].Omega, VerticalInd(9, 0.01 * car.Wheels[2].Omega), new Color(51, 0, 255));
            RenderIfMin(car.Wheels[3].Omega, VerticalInd(10, 0.01 * car.Wheels[3].Omega), new Color(51, 0, 255));

            RenderIfMin(car.Wheels[0].Joint.Angle, HorizInd(20, -10.0 * car.Wheels[0].Joint.Angle), new Color(0, 255, 0));
            RenderIfMin(car.Hull.AngularVelocity, HorizInd(30, -0.8 * car.Hull.AngularVelocity), new Color(255, 0, 0));
        }

        /// <summary>
        ///     The score, as Gymnasium draws it: <c>f"{self.reward:04.0f}"</c> in pygame's default font at size 42 (the embedded
        ///     <see cref="ScoreFont"/>), white on black, centred on (60, 750) of the flipped frame. A long score's box is wider
        ///     than the frame (309 characters for <see cref="double.MaxValue"/>) and is clipped at both edges, as pygame's blit
        ///     clips it.
        /// </summary>
        /// <param name="surf">The frame, already flipped.</param>
        /// <exception cref="InvalidOperationException">Thrown when the assembly lacks the embedded glyph data (a build that dropped the asset).</exception>
        private void RenderScore(Surface surf) {
            (byte[] indices, int width, int height) = ScoreFont.Embedded.RenderShaded(FormatScore(Reward));
            // The shaded surface's palette runs from the background (black) to the foreground (white), so index i is (i, i, i).
            byte[] bytes = new byte[width * height * 3];
            for (int x = 0, j = 0; x < width; x++) {
                for (int y = 0; y < height; y++, j += 3) {
                    byte level = indices[y * width + x];
                    bytes[j] = level;
                    bytes[j + 1] = level;
                    bytes[j + 2] = level;
                }
            }

            using NDArray flat = np.array(bytes);
            using NDArray source = flat.reshape(width, height, 3);
            Surface text = SurfArray.MakeSurface(source);
            // text_rect.center = (60, WINDOW_H - WINDOW_H * 2.5 / 40.0): pygame truncates the float, then centres with C's
            // integer halving of the size.
            int centerY = (int)(WindowH - WindowH * 2.5 / 40.0);
            surf.Blit(text, 60 - width / 2, centerY - height / 2);
        }

        /// <summary>
        ///     Python's <c>f"{x:04.0f}"</c>: rounded half to even on the exact value, every digit of the exact integer the double
        ///     holds, the sign kept for a negative zero, zeros padded after the sign to four characters; <c>0nan</c> for a NaN of
        ///     either sign, <c>0inf</c> and <c>-inf</c> for the infinities.
        /// </summary>
        /// <param name="x">The score.</param>
        /// <returns>The text, for example <c>-000</c>, <c>0012</c>, <c>1000</c>, <c>0nan</c> or <c>99999999999999991611392</c> (1e23).</returns>
        internal static string FormatScore(double x) {
            // Python writes "nan" whatever the NaN's sign and pads it, and "inf", with zeros as it pads digits under the '0' flag.
            if (double.IsNaN(x))
                return "0nan";
            if (double.IsInfinity(x))
                return x > 0 ? "0inf" : "-inf";
            double rounded = Math.Round(x, MidpointRounding.ToEven);
            // BigInteger gives every digit of the integer the double holds, as Python prints it; .NET's own numeric formats stop
            // at 15 significant digits for "0" (9007199254740992 would print as 9007199254740990).
            string digits = new BigInteger(Math.Abs(rounded)).ToString(CultureInfo.InvariantCulture);
            string sign = double.IsNegative(rounded) ? "-" : "";
            return sign + digits.PadLeft(4 - sign.Length, '0');
        }

        /// <summary>
        ///     Gymnasium's <c>_draw_colored_polygon</c>: rotates, scales and translates the polygon, and unless culled draws its
        ///     anti-aliased outline then fills it (SDL_gfx's <c>aapolygon</c> and <c>filled_polygon</c>) in the colour, truncated
        ///     to integers as pygame converts a float colour.
        /// </summary>
        /// <param name="surf">The frame.</param>
        /// <param name="poly">The polygon in metres.</param>
        /// <param name="color">The colour's three channels.</param>
        /// <param name="zoom">Pixels per metre.</param>
        /// <param name="translation">The screen offset.</param>
        /// <param name="angle">The rotation.</param>
        /// <param name="clip">Whether to skip a polygon none of whose vertices is near the frame.</param>
        private static void DrawColoredPolygon(Surface surf, IReadOnlyList<(double X, double Y)> poly, IReadOnlyList<double> color, double zoom,
                                               (double X, double Y) translation, double angle, bool clip = true) {
            var screen = new List<(double X, double Y)>(poly.Count);
            foreach ((double X, double Y) c in poly) {
                Vector2 rotated = new Vector2(c.X, c.Y).RotateRad(angle);
                screen.Add((rotated.X * zoom + translation.X, rotated.Y * zoom + translation.Y));
            }

            // This checks if the polygon is out of bounds of the screen, and we skip drawing if so.
            if (!clip || screen.Any(coord => -MaxShapeDim <= coord.X && coord.X <= WindowW + MaxShapeDim && -MaxShapeDim <= coord.Y && coord.Y <= WindowH + MaxShapeDim)) {
                var c = new Color((byte)(int)color[0], (byte)(int)color[1], (byte)(int)color[2]);
                GfxDraw.AAPolygon(surf, screen, c);
                GfxDraw.FilledPolygon(surf, screen, c);
            }
        }

        /// <summary>Gymnasium's <c>_create_image_array</c>: the frame shrunk with pygame's <c>smoothscale</c>, as <c>(height, width, 3)</c> RGB.</summary>
        /// <param name="screen">The frame.</param>
        /// <param name="width">The image's width.</param>
        /// <param name="height">The image's height.</param>
        /// <returns>A new array the caller owns.</returns>
        private static NDArray CreateImageArray(Surface screen, int width, int height) {
            Surface scaled = Transform.Smoothscale(screen, width, height);
            return SurfArray.ToRgbArray(scaled);
        }

        /// <summary>
        ///     Gymnasium's <c>_destroy</c>: once a reset made a road, destroys its tiles in order (the current listener sees their
        ///     contacts with the wheels end), then the car.
        /// </summary>
        /// <exception cref="AssertionError">Thrown when a road exists without a car (<c>assert self.car is not None</c>), which no reset leaves.</exception>
        private void Destroy() {
            if (_road is null || _road.Count == 0)
                return;
            foreach (RoadTile t in _road)
                World.DestroyBody(t.Body);
            _road = new List<RoadTile>();
            Car car = Car ?? throw new AssertionError();
            car.Destroy();
        }

        /// <summary>Gymnasium's <c>_init_colors</c>: random colours (3 + 3 draws and a channel) with domain randomization, the defaults otherwise.</summary>
        private void InitColors() {
            if (DomainRandomize) {
                DrawColors();
            } else {
                // default colours
                _roadColor = (new[] { 102.0, 102.0, 102.0 }, true);
                _bgColor = (new[] { 102.0, 204.0, 102.0 }, true);
                _grassColor = (new[] { 102.0, 230.0, 102.0 }, true);
            }
        }

        /// <summary>Gymnasium's <c>_reinit_colors(randomize)</c>: new random colours when asked.</summary>
        /// <param name="randomize">Whether to draw new colours.</param>
        /// <exception cref="AssertionError">Thrown without domain randomization, with Gymnasium's message.</exception>
        private void ReinitColors(bool randomize) {
            if (!DomainRandomize)
                throw new AssertionError("domain_randomize must be True to use this function.");
            if (randomize)
                DrawColors();
        }

        /// <summary>
        ///     The random colours: the road's three channels, the background's three (uniform in [0, 210)), the grass a copy of the
        ///     background with one channel (an <c>integers(3)</c> draw) 20 brighter.
        /// </summary>
        private void DrawColors() {
            // domain randomize the bg and grass colour
            _roadColor = (UniformArray(0, 210, 3), false);
            _bgColor = (UniformArray(0, 210, 3), false);
            double[] grass = (double[])_bgColor.Values.Clone();
            long idx = Integer(0, 3);
            grass[idx] += 20;
            _grassColor = (grass, false);
        }

        /// <summary>
        ///     Gymnasium's <c>_create_track</c>: 12 checkpoints (two draws each), the path that turns toward each in turn for up to
        ///     2,500 steps, the second full lap cut out between two passes of the start angle, a check that its ends meet, the
        ///     kerbs on sustained sharp turns, and the tiles (static sensors) with their polygons.
        /// </summary>
        /// <returns>Whether a closed track was built; on <see langword="false"/> nothing was created and the caller retries.</returns>
        /// <exception cref="AssertionError">Thrown when the closing cut was not found (<c>assert i1 != -1</c>), which the search's own exit rules out.</exception>
        private bool CreateTrack() {
            const int checkpointCount = 12;

            // Create checkpoints
            var checkpoints = new List<(double Alpha, double X, double Y)>();
            for (int c = 0; c < checkpointCount; c++) {
                double noise = Uniform(0, 2 * Math.PI * 1 / checkpointCount);
                double alphaC = 2 * Math.PI * c / checkpointCount + noise;
                double rad = Uniform(TrackRad / 3, TrackRad);

                if (c == 0) {
                    alphaC = 0;
                    rad = 1.5 * TrackRad;
                }

                if (c == checkpointCount - 1) {
                    alphaC = 2 * Math.PI * c / checkpointCount;
                    StartAlpha = 2 * Math.PI * (-0.5) / checkpointCount;
                    rad = 1.5 * TrackRad;
                }

                checkpoints.Add((alphaC, rad * Math.Cos(alphaC), rad * Math.Sin(alphaC)));
            }

            _road = new List<RoadTile>();

            // Go from one checkpoint to another to create track
            double x = 1.5 * TrackRad, y = 0, beta = 0;
            int destI = 0;
            int laps = 0;
            var track = new List<(double Alpha, double Beta, double X, double Y)>();
            int noFreeze = 2500;
            bool visitedOtherSide = false;
            while (true) {
                double alpha = Math.Atan2(y, x);
                if (visitedOtherSide && alpha > 0) {
                    laps += 1;
                    visitedOtherSide = false;
                }

                if (alpha < 0) {
                    visitedOtherSide = true;
                    alpha += 2 * Math.PI;
                }

                double destX, destY;
                while (true) { // Find destination from checkpoints
                    bool failed = true;
                    while (true) {
                        (double destAlpha, double dx, double dy) = checkpoints[destI % checkpoints.Count];
                        destX = dx;
                        destY = dy;
                        if (alpha <= destAlpha) {
                            failed = false;
                            break;
                        }

                        destI += 1;
                        if (destI % checkpoints.Count == 0)
                            break;
                    }

                    if (!failed)
                        break;
                    alpha -= 2 * Math.PI;
                }

                double r1x = Math.Cos(beta);
                double r1y = Math.Sin(beta);
                double p1x = -r1y;
                double p1y = r1x;
                double destDx = destX - x; // vector towards destination
                double destDy = destY - y;
                // destination vector projected on rad:
                double proj = r1x * destDx + r1y * destDy;
                while (beta - alpha > 1.5 * Math.PI)
                    beta -= 2 * Math.PI;
                while (beta - alpha < -1.5 * Math.PI)
                    beta += 2 * Math.PI;
                double prevBeta = beta;
                proj *= Scale;
                if (proj > 0.3)
                    beta -= PyMin(TrackTurnRate, Math.Abs(0.001 * proj));
                if (proj < -0.3)
                    beta += PyMin(TrackTurnRate, Math.Abs(0.001 * proj));
                x += p1x * TrackDetailStep;
                y += p1y * TrackDetailStep;
                track.Add((alpha, prevBeta * 0.5 + beta * 0.5, x, y));
                if (laps > 4)
                    break;
                noFreeze -= 1;
                if (noFreeze == 0)
                    break;
            }

            // Find closed loop range i1..i2, first loop should be ignored, second is OK
            int i1 = -1, i2 = -1;
            int i = track.Count;
            while (true) {
                i -= 1;
                if (i == 0)
                    return false; // Failed
                bool passThroughStart = track[i].Alpha > StartAlpha && track[i - 1].Alpha <= StartAlpha;
                if (passThroughStart && i2 == -1) {
                    i2 = i;
                } else if (passThroughStart && i1 == -1) {
                    i1 = i;
                    break;
                }
            }

            if (Verbose)
                Console.WriteLine($"Track generation: {i1}..{i2} -> {i2 - i1}-tiles track");
            if (i1 == -1 || i2 == -1)
                throw new AssertionError();

            track = track.GetRange(i1, Math.Max(0, i2 - 1 - i1));

            double firstBeta = track[0].Beta;
            double firstPerpX = Math.Cos(firstBeta);
            double firstPerpY = Math.Sin(firstBeta);
            // Length of perpendicular jump to put together head and tail
            double gx = firstPerpX * (track[0].X - track[^1].X);
            double gy = firstPerpY * (track[0].Y - track[^1].Y);
            double wellGluedTogether = Math.Sqrt(gx * gx + gy * gy);
            if (wellGluedTogether > TrackDetailStep)
                return false;

            // Red-white border on hard turns
            int n = track.Count;
            int Wrap(int index) => index < 0 ? index + n : index; // Python's negative indices
            var border = new bool[n];
            for (i = 0; i < n; i++) {
                bool good = true;
                double oneside = 0;
                for (int neg = 0; neg < BorderMinCount; neg++) {
                    double beta1 = track[Wrap(i - neg - 0)].Beta;
                    double beta2 = track[Wrap(i - neg - 1)].Beta;
                    good &= Math.Abs(beta1 - beta2) > TrackTurnRate * 0.2;
                    oneside += NpSign(beta1 - beta2);
                }

                good &= Math.Abs(oneside) == BorderMinCount;
                border[i] = good;
            }

            for (i = 0; i < n; i++) {
                for (int neg = 0; neg < BorderMinCount; neg++)
                    border[Wrap(i - neg)] |= border[i];
            }

            // Create tiles
            var polygon = (b2PolygonShape)FdTile.Shape!;
            for (i = 0; i < n; i++) {
                (double _, double beta1, double x1, double y1) = track[i];
                (double _, double beta2, double x2, double y2) = track[Wrap(i - 1)];
                (double X, double Y) road1L = (x1 - TrackWidth * Math.Cos(beta1), y1 - TrackWidth * Math.Sin(beta1));
                (double X, double Y) road1R = (x1 + TrackWidth * Math.Cos(beta1), y1 + TrackWidth * Math.Sin(beta1));
                (double X, double Y) road2L = (x2 - TrackWidth * Math.Cos(beta2), y2 - TrackWidth * Math.Sin(beta2));
                (double X, double Y) road2R = (x2 + TrackWidth * Math.Cos(beta2), y2 + TrackWidth * Math.Sin(beta2));
                polygon.Vertices = new b2Vec2[] { road1L, road1R, road2R, road2L };
                b2Body body = World.CreateStaticBody(FdTile);
                double c = 0.01 * (i % 3) * 255;
                var tile = new RoadTile(body, _roadColor.Values.Select(v => v + c).ToArray(), roadFriction: 1.0, idx: i);
                body.Fixtures[0].Sensor = true;
                _roadPoly.Add((new[] { road1L, road1R, road2R, road2L }, tile.Color));
                _road.Add(tile);
                if (border[i]) {
                    double side = NpSign(beta2 - beta1);
                    (double X, double Y) b1L = (x1 + side * TrackWidth * Math.Cos(beta1), y1 + side * TrackWidth * Math.Sin(beta1));
                    (double X, double Y) b1R = (x1 + side * (TrackWidth + Border) * Math.Cos(beta1), y1 + side * (TrackWidth + Border) * Math.Sin(beta1));
                    (double X, double Y) b2L = (x2 + side * TrackWidth * Math.Cos(beta2), y2 + side * TrackWidth * Math.Sin(beta2));
                    (double X, double Y) b2R = (x2 + side * (TrackWidth + Border) * Math.Cos(beta2), y2 + side * (TrackWidth + Border) * Math.Sin(beta2));
                    _roadPoly.Add((new[] { b1L, b1R, b2R, b2L }, i % 2 == 0 ? new[] { 255.0, 255.0, 255.0 } : new[] { 255.0, 0.0, 0.0 }));
                }
            }

            _track = track;
            return true;
        }

        /// <summary>A colour as Gymnasium holds it: a new 3-element array, int64 for the defaults and float64 for random ones.</summary>
        /// <param name="color">The colour and its kind.</param>
        /// <returns>The array, the caller's to dispose.</returns>
        private static NDArray ColorArray((double[] Values, bool IsInt) color) =>
            color.IsInt ? np.array(color.Values.Select(v => (long)v).ToArray()) : np.array((double[])color.Values.Clone());

        /// <summary>
        ///     Python's truth value of an option: <c>None</c>, <c>False</c>, zero and empty text or collections are false, anything
        ///     else (an object, a non-empty collection) is true.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>What Python's <c>if value:</c> decides.</returns>
        private static bool PyTruth(object? value) => value switch {
            null => false,
            bool b => b,
            string text => text.Length > 0,
            NDArray array => array.size == 1 ? Convert.ToDouble(array.GetAtIndex(0), CultureInfo.InvariantCulture) != 0 : array.size > 0,
            sbyte or byte or short or ushort or int or uint or long or ulong or float or double or decimal => Convert.ToDouble(value, CultureInfo.InvariantCulture) != 0,
            System.Collections.ICollection collection => collection.Count > 0,
            _ => true,
        };

        /// <summary>Python's <c>min(a, b)</c> of two floats: <paramref name="b"/> only when it is smaller.</summary>
        /// <param name="a">The first value.</param>
        /// <param name="b">The second value.</param>
        /// <returns>The minimum as Python picks it.</returns>
        private static double PyMin(double a, double b) => b < a ? b : a;

        /// <summary><c>np.sign</c> of a float64: 1, -1, 0 for either zero, NaN for NaN.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The sign.</returns>
        private static double NpSign(double value) => value > 0 ? 1.0 : value < 0 ? -1.0 : value == 0 ? 0.0 : double.NaN;

        /// <summary>One <c>np_random.uniform(low, high)</c> draw.</summary>
        /// <param name="low">The lower bound.</param>
        /// <param name="high">The upper bound.</param>
        /// <returns>The value.</returns>
        private double Uniform(double low, double high) {
            using NDArray drawn = NpRandom.uniform(low, high);
            return drawn.GetAtIndex<double>(0);
        }

        /// <summary>One <c>np_random.uniform(low, high, size=count)</c> draw of several values.</summary>
        /// <param name="low">The lower bound.</param>
        /// <param name="high">The upper bound.</param>
        /// <param name="count">How many.</param>
        /// <returns>The values.</returns>
        private double[] UniformArray(double low, double high, int count) {
            using NDArray drawn = NpRandom.uniform(low, high, new Shape(count));
            return drawn.ToArray<double>();
        }

        /// <summary>One <c>np_random.integers(low, high)</c> draw.</summary>
        /// <param name="low">The lower bound.</param>
        /// <param name="high">The exclusive upper bound.</param>
        /// <returns>The value.</returns>
        private long Integer(long low, long high) {
            using NDArray drawn = NpRandom.integers(low, high);
            return drawn.GetAtIndex<long>(0);
        }

        /// <summary>
        ///     A track tile: its static sensor body and the attributes Gymnasium hangs on it (<c>color</c>, <c>road_visited</c>,
        ///     <c>road_friction</c>, <c>idx</c>). The body's user data is this object, as Python's <c>t.userData = t</c> lets the
        ///     <see cref="FrictionDetector"/> recognize a tile.
        /// </summary>
        public sealed class RoadTile {
            /// <summary>A tile over its body.</summary>
            /// <param name="body">The tile's static body.</param>
            /// <param name="color">The colour array, shared with <c>road_poly</c>.</param>
            /// <param name="roadFriction">The friction factor (1.0).</param>
            /// <param name="idx">The tile's position on the track.</param>
            internal RoadTile(b2Body body, double[] color, double roadFriction, long idx) {
                Body = body;
                Color = color;
                RoadFriction = roadFriction;
                Idx = idx;
                body.UserData = this;
            }

            /// <summary>The tile's static body.</summary>
            public b2Body Body { get; }

            /// <summary>
            ///     Python's <c>color</c>: the road colour plus the tile's shade, a float64 array the env's <c>road_poly</c> shares;
            ///     the contact listener overwrites it in place with the road colour when anything touches the tile.
            /// </summary>
            public double[] Color { get; }

            /// <summary>Python's <c>road_visited</c>: whether a wheel has touched the tile this episode.</summary>
            public bool RoadVisited { get; set; }

            /// <summary>Python's <c>road_friction</c>: the friction factor a wheel on the tile gets (1.0).</summary>
            public double RoadFriction { get; set; }

            /// <summary>Python's <c>idx</c>: the tile's position on the track (0 is the start).</summary>
            public long Idx { get; set; }
        }
    }

    /// <summary>
    ///     Gymnasium's <c>FrictionDetector</c>: CarRacing's contact listener. Any contact with a tile recolours the tile with the
    ///     road colour; a wheel's contact adds the tile to the wheel's tiles (removed when it ends), and a tile's first visit
    ///     pays 1000/N to the score and may complete the lap.
    /// </summary>
    /// <remarks>
    ///     Public, as Gymnasium's is a module-level class whose name no other env shares (DR-28 keeps only the duplicated
    ///     <c>ContactDetector</c> private). Reference:
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/car_racing.py#L58">gymnasium/envs/box2d/car_racing.py</see>.
    /// </remarks>
    public sealed class FrictionDetector : b2ContactListener {
        /// <summary>The env whose score and tiles the contacts change; borrowed, the env owns its arrays.</summary>
        [NDBorrowed]
        private readonly CarRacingEnv _env;

        /// <summary>A detector for an env.</summary>
        /// <param name="env">The env.</param>
        /// <param name="lapCompletePercent">The share of the track's tiles that, visited when the start tile is reached, completes the lap.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        public FrictionDetector(CarRacingEnv env, double lapCompletePercent) {
            _env = env ?? throw new ArgumentNullException(nameof(env));
            LapCompletePercent = lapCompletePercent;
        }

        /// <summary>Python's <c>env</c>.</summary>
        public CarRacingEnv Env => _env;

        /// <summary>Python's <c>lap_complete_percent</c>.</summary>
        public double LapCompletePercent { get; }

        /// <summary>A contact begins: <c>_contact(contact, True)</c>.</summary>
        /// <param name="contact">The contact.</param>
        public override void BeginContact(b2Contact contact) => OnContact(contact, true);

        /// <summary>A contact ends: <c>_contact(contact, False)</c>.</summary>
        /// <param name="contact">The contact.</param>
        /// <exception cref="KeyError">Thrown when a wheel loses a tile it did not have, as Python's <c>set.remove</c> raises.</exception>
        public override void EndContact(b2Contact contact) => OnContact(contact, false);

        /// <summary>Gymnasium's <c>_contact</c>: finds the tile and the other body's wheel, recolours the tile, and keeps the wheel's tiles and the score.</summary>
        /// <param name="contact">The contact.</param>
        /// <param name="begin">Whether it begins.</param>
        /// <exception cref="KeyError">Thrown when an ending contact's tile is not in the wheel's set.</exception>
        private void OnContact(b2Contact contact, bool begin) {
            CarRacingEnv.RoadTile? tile = null;
            object? obj = null;
            object? u1 = contact.FixtureA.Body.UserData;
            object? u2 = contact.FixtureB.Body.UserData;
            if (u1 is CarRacingEnv.RoadTile t1) {
                tile = t1;
                obj = u2;
            }

            if (u2 is CarRacingEnv.RoadTile t2) {
                tile = t2;
                obj = u1;
            }

            if (tile is null)
                return;

            // inherit tile color from env: tile.color[:] = self.env.road_color, in place, so road_poly sees it too.
            Array.Copy(_env.RoadColorValues, tile.Color, tile.Color.Length);

            if (obj is not CarWheel wheel)
                return;
            if (begin) {
                wheel.Tiles.Add(tile);
                if (!tile.RoadVisited) {
                    tile.RoadVisited = true;
                    _env.Reward += 1000.0 / _env.Track.Count;
                    _env.TileVisitedCount += 1;

                    // Lap is considered completed if enough % of the track was covered
                    if (tile.Idx == 0 && (double)_env.TileVisitedCount / _env.Track.Count > LapCompletePercent)
                        _env.NewLap = true;
                }
            } else {
                if (!wheel.Tiles.Remove(tile))
                    throw new KeyError(PyBuiltins.Repr(tile));
            }
        }
    }
}
