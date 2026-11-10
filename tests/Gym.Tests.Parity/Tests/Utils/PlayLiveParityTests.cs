using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Environments;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Observations;
using Gym.Rendering;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Session;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace Gym.Tests.Parity.Tests.Utils {
    /// <summary>
    ///     <c>gymnasium.utils.play</c> against Gym.NET's <see cref="Play"/>, <see cref="PlayableGame"/> and
    ///     <see cref="PlayPlot"/>, live: pygame's dummy display against Gym.NET's <see cref="Display"/> over
    ///     <see cref="NullEnvViewer"/>, events posted to both queues as Gymnasium's own tests post them.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <c>display_arr</c> is compared pixel for pixel on surfaces of every relation to the frame (equal, larger by odd
    ///     and even margins, smaller); <c>play</c> callback by callback, each with the display as it stood (the previous
    ///     frame); <c>PlayableGame</c> event by event; <c>PlayPlot</c> call by call through a matplotlib stand-in, as the
    ///     rendering suite records moviepy's calls, so the cells need no matplotlib and draw nothing.
    ///     </para>
    ///     <para>
    ///     Gymnasium's <c>play</c> ends in <c>pygame.quit()</c>, and so do these cells; the pygame cells initialize the display
    ///     themselves. The display, the queue and the viewer factory are process-wide on the .NET side too, which the suite's
    ///     one-test-at-a-time runs make safe.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/play.py">gymnasium/utils/play.py</see>.</para>
    /// </remarks>
    [TestClass]
    public sealed class PlayLiveParityTests : ParityTestBase {
        /// <summary>The viewer factory before the cell, restored by <see cref="Cleanup"/>.</summary>
        private IEnvironmentViewerFactoryDelegate? _previousFactory;

        /// <summary>Headless windows on the .NET side, an empty queue, and pygame's display initialized on the Python side.</summary>
        [TestInitialize]
        public void Initialize() {
            _previousFactory = Gymnasium.HumanViewerFactory;
            Gymnasium.HumanViewerFactory = NullEnvViewer.Factory;
            EventQueue.Clear();
            PythonRuntime.Run(Python.pygame.display.init);
        }

        /// <summary>Quits both displays (pygame's <c>pygame.quit()</c>, as Gymnasium's tests do after each test) and restores the factory.</summary>
        [TestCleanup]
        public void Cleanup() {
            Display.Quit();
            Gymnasium.HumanViewerFactory = _previousFactory;
            PythonRuntime.Run(() => {
                using PyObject _ = Python.evalStmts("import pygame\npygame.quit()", "None");
            });
        }

        /// <summary>
        ///     <c>display_arr</c> on 12 frame, size and screen combinations: RGB and 2-D (pygame's 8-bit palette) frames,
        ///     transposed, not transposed and <c>None</c>, scaled up, down and to nothing, centred with odd and even margins and
        ///     cropped when larger than the screen (offsets truncated toward zero), over a screen first filled with colour.
        /// </summary>
        [TestMethod]
        public void DisplayArr_MatchesPygame() {
            var cases = new (int H, int W, int Channels, bool? Transpose, (int W, int H) Video, (int W, int H) Screen)[] {
                (4, 6, 3, true, (6, 4), (6, 4)),
                (4, 6, 3, true, (12, 8), (12, 8)),
                (40, 60, 3, true, (27, 18), (27, 18)),
                (4, 6, 3, true, (6, 4), (9, 7)),
                (4, 6, 3, true, (6, 4), (10, 8)),
                (4, 6, 3, true, (12, 8), (7, 5)),
                (4, 6, 3, true, (12, 8), (6, 3)),
                (4, 6, 3, false, (4, 6), (4, 6)),
                (4, 6, 3, null, (4, 6), (5, 9)),
                (5, 7, 0, true, (14, 10), (15, 12)),
                (5, 7, 0, false, (5, 7), (5, 7)),
                (3, 5, 3, true, (0, 0), (5, 3)),
            };
            var random = new Random(20261110);
            foreach (var c in cases) {
                using NDArray frame = RandomFrame(random, c.H, c.W, c.Channels);
                string label = $"display_arr(({c.H}, {c.W}{(c.Channels == 0 ? "" : ", 3")}), video={c.Video}, transpose={c.Transpose?.ToString() ?? "None"}) on {c.Screen}";
                Outcome python = Outcome.CapturePython(() => {
                    using PyObject screen = Python.pygame.Surface(c.Screen.W, c.Screen.H);
                    Python.pygame.surface.Surface.fill(screen, (12, 34, 56));
                    using PyObject arr = frame.ToNumpyCopy();
                    Python.gymnasium.utils.play.display_arr(screen, arr, c.Video, c.Transpose);
                    return Python.pygame.surfarray.array3d(screen);
                });
                Outcome dotnet = Outcome.Capture(() => {
                    var screen = new Surface(c.Screen.W, c.Screen.H);
                    screen.Fill(new Color(12, 34, 56));
                    Play.DisplayArr(screen, frame, c.Video, c.Transpose);
                    return SurfArray.Array3d(screen);
                });
                ByteContract.AssertSame(label, python, dotnet);
            }
        }

        /// <summary>
        ///     <c>display_arr</c>'s refusals: a float frame (the bare dtype assertion), four channels (pygame's shape error), a
        ///     one-dimensional frame transposed (NumPy's <c>swapaxes</c> error), and a negative size (pygame's scale error).
        /// </summary>
        [TestMethod]
        public void DisplayArr_Refusals_MatchPygame() {
            var cases = new (string Label, Func<NDArray> Frame, bool Transpose, (int W, int H) Video)[] {
                ("float64 frame", () => np.ones(new Shape(2, 3, 3)), true, (3, 2)),
                ("four channels", () => np.zeros(new Shape(2, 3, 4), np.uint8), true, (3, 2)),
                ("one dimension, transposed", () => np.zeros(new Shape(5), np.uint8), true, (3, 2)),
                ("negative size", () => np.zeros(new Shape(2, 3, 3), np.uint8), true, (-1, 2)),
            };
            foreach (var c in cases) {
                using NDArray frame = c.Frame();
                Outcome python = Outcome.CapturePython(() => {
                    using PyObject screen = Python.pygame.Surface(4, 4);
                    using PyObject arr = frame.ToNumpyCopy();
                    Python.gymnasium.utils.play.display_arr(screen, arr, c.Video, c.Transpose);
                    return Python.pygame.surfarray.array3d(screen);
                });
                Outcome dotnet = Outcome.Capture(() => {
                    var screen = new Surface(4, 4);
                    Play.DisplayArr(screen, frame, c.Video, c.Transpose);
                    return SurfArray.Array3d(screen);
                });
                ByteContract.AssertSame($"display_arr refuses: {c.Label}", python, dotnet);
            }
        }

        /// <summary>
        ///     <c>PlayableGame</c> over CartPole-v1, event by event: its relevant keys, sizes (zoomed by 1, 2.2, 0.5 and 1.1, all
        ///     truncated) and screen; then presses and releases of relevant and irrelevant keys, a repeated press, a release
        ///     never pressed, resizes (rounded half to even, keeping the aspect, never drifting), Escape (ignored while mapped),
        ///     and <c>QUIT</c>, with the pressed keys, <c>running</c> and the video size after each.
        /// </summary>
        /// <param name="scenario">Which mapping, zoom and event script.</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a scenario the cell does not define.</exception>
        [TestMethod]
        [DataRow("keys")]
        [DataRow("zoom")]
        [DataRow("shrink")]
        [DataRow("escape-mapped")]
        [DataRow("resize")]
        public void PlayableGame_MatchesGymnasium(string scenario) {
            (object?[] Pairs, double? Zoom, (int Type, object? Attributes)[] Events) setup = scenario switch {
                "keys" => (Pairs(((97, 100), 2), (ValueTuple.Create(97), 0), (ValueTuple.Create(100), 1)), null, new[] {
                    Key(EventType.KeyDown, 97), Key(EventType.KeyDown, 1), Key(EventType.KeyDown, 100), Key(EventType.KeyUp, 97),
                    Key(EventType.KeyDown, 100), Key(EventType.KeyUp, 100), Key(EventType.KeyUp, 100), Key(EventType.KeyUp, 5),
                    Key(EventType.KeyDown, KeyCode.Escape),
                }),
                "zoom" => (Pairs((ValueTuple.Create(KeyCode.Left), 0), (ValueTuple.Create(KeyCode.Right), 1)), 2.2, new[] {
                    Key(EventType.KeyDown, KeyCode.Left), Key(EventType.KeyDown, KeyCode.Right), Key(EventType.KeyUp, KeyCode.Left), (EventType.Quit, (object?)null),
                }),
                "shrink" => (Pairs((ValueTuple.Create(97), 0)), 0.5, new[] { Key(EventType.KeyDown, 97), (EventType.WindowResized, new { x = 301, y = 99 }) }),
                "escape-mapped" => (Pairs((ValueTuple.Create(KeyCode.Escape), 0), (ValueTuple.Create(97), 1)), 1.1, new[] {
                    Key(EventType.KeyDown, KeyCode.Escape), Key(EventType.KeyUp, KeyCode.Escape), Key(EventType.KeyDown, 13),
                }),
                "resize" => (Pairs((ValueTuple.Create(97), 0)), null, new[] {
                    (EventType.WindowResized, (object?)new { x = 25, y = 30 }), (EventType.WindowResized, new { x = 750, y = 500 }),
                    (EventType.WindowResized, new { x = 1201, y = 1200 }), (EventType.WindowResized, new { x = 15, y = 10 }),
                    (EventType.WindowResized, new { x = 3, y = 1 }), (EventType.WindowResized, new { x = 600, y = 400 }),
                    (EventType.WindowResized, new { x = 300.5, y = 1000 }), (EventType.VideoResize, new { w = 10, h = 10 }),
                }),
                _ => throw new ArgumentOutOfRangeException(nameof(scenario)),
            };

            const string statements = """
                import gymnasium as gym
                import pygame
                from gymnasium.utils.play import PlayableGame
                env = gym.make("CartPole-v1", render_mode="rgb_array")
                env.reset(seed=1)
                game = PlayableGame(env, dict(pairs), zoom)
                log = [(sorted(game.relevant_keys), game.original_video_size, game.video_size, game.screen.get_size(), sorted(game.pressed_keys), game.running)]
                for event_type, attributes in events:
                    game.process_event(pygame.event.Event(event_type, attributes or {}))
                    log.append((sorted(game.pressed_keys), game.running, game.video_size, game.screen.get_size()))
                env.close()
                result = tuple(log)
                """;
            Outcome python = Outcome.CapturePython(() => {
                using PyObject pairs = PyConvert.ToPython(setup.Pairs.ToList());
                using PyObject zoom = PyConvert.ToPython(setup.Zoom);
                using PyObject events = PyConvert.ToPython(setup.Events.Select(e => (object?)(e.Type, AttributesOf(e.Attributes))).ToList());
                return Python.evalStmts(statements, "result", ("pairs", pairs), ("zoom", zoom), ("events", events));
            });

            Outcome dotnet = Outcome.Capture(() => {
                using IEnv env = Gymnasium.Make("CartPole-v1", new { render_mode = "rgb_array" });
                Outcome.Release(env.Reset(1).Observation);
                var game = new PlayableGame(env, ToDictionary(setup.Pairs), setup.Zoom);
                var log = new List<object?> {
                    new object?[] { Sorted(game.RelevantKeys), game.OriginalVideoSize, game.VideoSize, (game.Screen.Width, game.Screen.Height), Sorted(game.PressedKeys), game.Running },
                };
                foreach ((int type, object? attributes) in setup.Events) {
                    game.ProcessEvent(new Event(type, attributes));
                    log.Add(new object?[] { Sorted(game.PressedKeys), game.Running, game.VideoSize, (game.Screen.Width, game.Screen.Height) });
                }

                return log.ToArray();
            });
            ByteContract.AssertSame($"PlayableGame over CartPole-v1 ({scenario})", python, dotnet);
        }

        /// <summary>
        ///     <c>PlayableGame</c>'s refusals and lookups over Gymnasium's envs: no render mode, zoom 0, -1.5 and NaN, no mapping
        ///     with and without a spec (Python's bare <c>assert</c>), an int key (not iterable), an attribute mapping that is not
        ///     a dict, a mapping from an attribute set at run time, and MountainCar-v0's own mapping (its relevant keys 275,
        ///     276).
        /// </summary>
        [TestMethod]
        public void PlayableGame_Refusals_MatchGymnasium() {
            var cases = new (string Label, string Env, string Setup, object?[]? Pairs, double? Zoom)[] {
                ("no render mode", "make('CartPole-v1')", "", Pairs((ValueTuple.Create(97), 0)), null),
                ("zoom 0", "make('CartPole-v1', render_mode='rgb_array')", "", Pairs((ValueTuple.Create(97), 0)), 0.0),
                ("zoom -1.5", "make('CartPole-v1', render_mode='rgb_array')", "", Pairs((ValueTuple.Create(97), 0)), -1.5),
                ("zoom NaN", "make('CartPole-v1', render_mode='rgb_array')", "", Pairs((ValueTuple.Create(97), 0)), double.NaN),
                ("no mapping", "make('CartPole-v1', render_mode='rgb_array')", "", null, null),
                ("no mapping, no spec", "bare", "", null, null),
                ("int key", "make('CartPole-v1', render_mode='rgb_array')", "", Pairs((97, 0)), null),
                ("attribute not a dict", "make('CartPole-v1', render_mode='rgb_array')", "list", null, null),
                ("attribute mapping", "make('CartPole-v1', render_mode='rgb_array')", "dict", null, null),
                ("MountainCar-v0's mapping", "make('MountainCar-v0', render_mode='rgb_array')", "", null, null),
            };
            foreach (var c in cases) {
                string statements = $$"""
                    import gymnasium as gym
                    from gymnasium.envs.classic_control import CartPoleEnv
                    from gymnasium.utils.play import PlayableGame
                    env = CartPoleEnv(render_mode="rgb_array") if {{PyBool(c.Env == "bare")}} else gym.{{(c.Env == "bare" ? "make('CartPole-v1')" : c.Env)}}
                    env.reset(seed=1)
                    if setup == "list":
                        env.get_keys_to_action = lambda: [((97,), 0)]
                    elif setup == "dict":
                        env.get_keys_to_action = lambda: {(97, 100): 1, (32,): 0}
                    try:
                        game = PlayableGame(env, None if pairs is None else dict(pairs), zoom)
                        result = (sorted(game.relevant_keys), game.video_size)
                    finally:
                        env.close()
                    """;
                Outcome python = Outcome.CapturePython(() => {
                    using PyObject setup = PyConvert.ToPython(c.Setup);
                    using PyObject pairs = PyConvert.ToPython(c.Pairs?.ToList());
                    using PyObject zoom = PyConvert.ToPython(c.Zoom);
                    return Python.evalStmts(statements, "result", ("setup", setup), ("pairs", pairs), ("zoom", zoom));
                });

                Outcome dotnet = Outcome.Capture(() => {
                    using IEnv env = c.Env switch {
                        "bare" => new CartPoleEnv(renderMode: "rgb_array"),
                        "make('CartPole-v1')" => Gymnasium.Make("CartPole-v1"),
                        "make('MountainCar-v0', render_mode='rgb_array')" => Gymnasium.Make("MountainCar-v0", new { render_mode = "rgb_array" }),
                        _ => Gymnasium.Make("CartPole-v1", new { render_mode = "rgb_array" }),
                    };
                    Outcome.Release(env.Reset(1).Observation);
                    if (c.Setup == "list")
                        PyAttributes.SetAttr(env, "get_keys_to_action", (Func<object>)(() => new List<object?> { (ValueTuple.Create(97), 0) }));
                    else if (c.Setup == "dict")
                        PyAttributes.SetAttr(env, "get_keys_to_action", (Func<object>)(() => new Dictionary<object, object?> { [(97, 100)] = 1, [ValueTuple.Create(32)] = 0 }));
                    var game = new PlayableGame(env, c.Pairs is null ? null : ToDictionary(c.Pairs), c.Zoom);
                    return new object?[] { Sorted(game.RelevantKeys), game.VideoSize };
                });
                ByteContract.AssertSame($"PlayableGame: {c.Label}", python, dotnet);
                Display.Quit();
                PythonRuntime.Run(() => {
                    using PyObject _ = Python.evalStmts("import pygame\npygame.quit()\npygame.display.init()", "None");
                });
            }
        }

        /// <summary>
        ///     <c>play</c> over CartPole-v1, callback by callback, with the same events posted from the callbacks on both sides:
        ///     every argument (the reset's <c>(obs, info)</c> as the first <c>obs_t</c> of each episode), the action the held
        ///     keys chose, and the display as it stood (the previous frame, scaled and centred). Scenarios: each key form of
        ///     Gymnasium's test (strings, ints, tuples) and the mapping from a wrapper's <c>get_keys_to_action</c>; episodes that
        ///     end and restart with the same seed on the no-op; a held combination; <c>wait_on_player</c>; <c>rgb_array_list</c>;
        ///     an explicit no-op; no transpose; and resizes posted mid-game.
        /// </summary>
        /// <param name="scenario">Which mapping and script.</param>
        [TestMethod]
        [DataRow("str")]
        [DataRow("int")]
        [DataRow("tuple")]
        [DataRow("wrapper")]
        [DataRow("episodes")]
        [DataRow("combination")]
        [DataRow("wait")]
        [DataRow("list-render")]
        [DataRow("noop-transpose")]
        [DataRow("resize")]
        public void Play_MatchesGymnasium(string scenario) => AssertPlay(scenario, viaRunAsync: false);

        /// <summary>
        ///     The scenarios of <see cref="Play_MatchesGymnasium"/> through <see cref="Play.RunAsync"/> (DR-44), awaited: headless,
        ///     so the loop runs on the thread pool with awaited pacing, and every callback, display and refusal must still be
        ///     Gymnasium's <c>play</c>'s.
        /// </summary>
        /// <param name="scenario">Which mapping and script.</param>
        [TestMethod]
        [DataRow("str")]
        [DataRow("int")]
        [DataRow("tuple")]
        [DataRow("wrapper")]
        [DataRow("episodes")]
        [DataRow("combination")]
        [DataRow("wait")]
        [DataRow("list-render")]
        [DataRow("noop-transpose")]
        [DataRow("resize")]
        public void Play_RunAsync_MatchesGymnasium(string scenario) => AssertPlay(scenario, viaRunAsync: true);

        /// <summary>
        ///     Runs one <c>play</c> scenario on both sides and compares the outcomes: the callbacks' arguments with the display
        ///     at each, or the error.
        /// </summary>
        /// <param name="scenario">Which mapping and script.</param>
        /// <param name="viaRunAsync">Play through <see cref="Play.RunAsync"/>, awaited, instead of <see cref="Play.Run"/>.</param>
        /// <exception cref="Gym.Tests.Parity.Pair.ParityException">Thrown when the outcomes differ.</exception>
        private static void AssertPlay(string scenario, bool viaRunAsync) {
            const int a = 97, d = 100;
            object?[] tuples = Pairs((ValueTuple.Create(a), 0), (ValueTuple.Create(d), 1));
            var setup = new PlayScenario { Pairs = tuples, Script = AlternateKeys(a, d) };
            switch (scenario) {
                case "str":
                    setup.Pairs = Pairs(("a", 0), ("d", 1));
                    break;
                case "int":
                    setup.Pairs = Pairs((a, 0), (d, 1));
                    break;
                case "wrapper":
                    setup.Wrapper = true;
                    break;
                case "episodes":
                    // The no-op pushes left until the pole falls, three episodes in a row, each reset with the same seed.
                    setup.Script = Enumerable.Repeat(Array.Empty<(int, object?)>(), 30).ToArray();
                    setup.Fps = 1000;
                    break;
                case "combination":
                    setup.Pairs = Pairs(("ad", 1), ((a, d), 0), (ValueTuple.Create(a), 1), (ValueTuple.Create(d), 0));
                    setup.Script = new[] {
                        new[] { Key(EventType.KeyDown, a) }, new[] { Key(EventType.KeyDown, d) }, Array.Empty<(int, object?)>(),
                        new[] { Key(EventType.KeyUp, a) }, new[] { Key(EventType.KeyUp, d) },
                    };
                    break;
                case "wait":
                    // Nothing steps until a key is held: the first press is posted before play starts, and each step swaps keys.
                    setup.WaitOnPlayer = true;
                    setup.Before = new[] { Key(EventType.KeyDown, a) };
                    setup.Script = new[] {
                        new[] { Key(EventType.KeyUp, a), Key(EventType.KeyDown, d) }, new[] { Key(EventType.KeyUp, d), Key(EventType.KeyDown, a) },
                        new[] { Key(EventType.KeyDown, d) },
                    };
                    break;
                case "list-render":
                    setup.RenderMode = "rgb_array_list";
                    setup.Zoom = 0.3;
                    break;
                case "noop-transpose":
                    setup.Noop = 1;
                    setup.Transpose = false;
                    setup.Zoom = 0.15;
                    setup.Script = new[] { Array.Empty<(int, object?)>(), new[] { Key(EventType.KeyDown, a) }, Array.Empty<(int, object?)>(), new[] { Key(EventType.KeyUp, a) } };
                    break;
                case "resize":
                    setup.Script = new[] {
                        new[] { (EventType.WindowResized, (object?)new { x = 70, y = 20 }) }, new[] { (EventType.WindowResized, (object?)new { x = 33, y = 90 }) },
                        new[] { Key(EventType.KeyDown, a), (EventType.WindowResized, new { x = 1, y = 1 }) },
                    };
                    break;
            }

            ByteContract.AssertSame($"play over CartPole-v1 ({scenario}{(viaRunAsync ? ", RunAsync" : "")})", PlayPython(setup), PlayDotNet(setup, viaRunAsync));
        }

        /// <summary>
        ///     <c>play</c>'s refusals over Gymnasium's envs, each before any frame is shown: no mapping (with and without a spec),
        ///     an attribute mapping of <c>None</c> or a list, an empty key tuple (MountainCar-v0's own mapping has one), a tuple
        ///     holding a float, a float key, an action outside the space, a two-character string inside a tuple (<c>ord</c>),
        ///     no render mode, a zero zoom, a float <c>render_fps</c>, a negative seed, and MountainCarContinuous-v0, which has no
        ///     mapping; through <see cref="Play.Run"/>, and through <see cref="Play.RunAsync"/>, whose task carries the same error.
        /// </summary>
        /// <param name="viaRunAsync">Play through <see cref="Play.RunAsync"/> and wait for its task, instead of <see cref="Play.Run"/>.</param>
        [TestMethod]
        [DataRow(false)]
        [DataRow(true)]
        public void Play_Refusals_MatchGymnasium(bool viaRunAsync) {
            var cases = new (string Label, string Env, string Setup, object?[]? Pairs, double? Zoom, long Seed)[] {
                ("no mapping", "make('CartPole-v1', render_mode='rgb_array')", "", null, null, 1),
                ("no mapping, no spec", "bare", "", null, null, 1),
                ("attribute None", "make('CartPole-v1', render_mode='rgb_array')", "none", null, null, 1),
                ("attribute list", "make('CartPole-v1', render_mode='rgb_array')", "list", null, null, 1),
                ("empty tuple", "make('CartPole-v1', render_mode='rgb_array')", "", Pairs((new ValueTuple(), 0)), null, 1),
                ("float in tuple", "make('CartPole-v1', render_mode='rgb_array')", "", Pairs(((97, 1.5), 0)), null, 1),
                ("float key", "make('CartPole-v1', render_mode='rgb_array')", "", Pairs((1.5, 0)), null, 1),
                ("bad action", "make('CartPole-v1', render_mode='rgb_array')", "", Pairs(("a", 0), ("d", 5)), null, 1),
                ("long string in tuple", "make('CartPole-v1', render_mode='rgb_array')", "", Pairs((ValueTuple.Create("ab"), 0)), null, 1),
                ("no render mode", "make('CartPole-v1')", "", Pairs(("a", 0)), null, 1),
                ("zoom 0", "make('CartPole-v1', render_mode='rgb_array')", "", Pairs(("a", 0)), 0.0, 1),
                ("float render_fps", "make('CartPole-v1', render_mode='rgb_array')", "fps", Pairs(("a", 0)), null, 1),
                ("negative seed", "make('CartPole-v1', render_mode='rgb_array')", "", Pairs(("a", 0)), null, -1),
                ("MountainCar-v0's own mapping", "make('MountainCar-v0', render_mode='rgb_array')", "", null, null, 1),
                ("MountainCarContinuous-v0", "make('MountainCarContinuous-v0', render_mode='rgb_array')", "", null, null, 1),
            };
            foreach (var c in cases) {
                string statements = $$"""
                    import gymnasium as gym
                    from gymnasium.envs.classic_control import CartPoleEnv
                    from gymnasium.utils.play import play
                    env = CartPoleEnv(render_mode="rgb_array") if {{PyBool(c.Env == "bare")}} else gym.{{(c.Env == "bare" ? "make('CartPole-v1')" : c.Env)}}
                    if setup == "none":
                        env.get_keys_to_action = lambda: None
                    elif setup == "list":
                        env.get_keys_to_action = lambda: [((97,), 0)]
                    elif setup == "fps":
                        env.metadata = {**env.metadata, "render_fps": 50.0}
                    try:
                        play(env, zoom=zoom, keys_to_action=None if pairs is None else dict(pairs), seed=seed)
                    finally:
                        env.close()
                    result = None
                    """;
                Outcome python = Outcome.CapturePython(() => {
                    using PyObject setup = PyConvert.ToPython(c.Setup);
                    using PyObject pairs = PyConvert.ToPython(c.Pairs?.ToList());
                    using PyObject zoom = PyConvert.ToPython(c.Zoom);
                    using PyObject seed = PyConvert.ToPython(c.Seed);
                    return Python.evalStmts(statements, "result", ("setup", setup), ("pairs", pairs), ("zoom", zoom), ("seed", seed));
                });

                Outcome dotnet = Outcome.Capture(() => {
                    using IEnv env = c.Env switch {
                        "bare" => new CartPoleEnv(renderMode: "rgb_array"),
                        "make('CartPole-v1')" => Gymnasium.Make("CartPole-v1"),
                        "make('MountainCar-v0', render_mode='rgb_array')" => Gymnasium.Make("MountainCar-v0", new { render_mode = "rgb_array" }),
                        "make('MountainCarContinuous-v0', render_mode='rgb_array')" => Gymnasium.Make("MountainCarContinuous-v0", new { render_mode = "rgb_array" }),
                        _ => Gymnasium.Make("CartPole-v1", new { render_mode = "rgb_array" }),
                    };
                    switch (c.Setup) {
                        case "none":
                            PyAttributes.SetAttr(env, "get_keys_to_action", (Func<object?>)(() => null));
                            break;
                        case "list":
                            PyAttributes.SetAttr(env, "get_keys_to_action", (Func<object>)(() => new List<object?> { (ValueTuple.Create(97), 0) }));
                            break;
                        case "fps":
                            env.Metadata = new Metadata(env.Metadata) { [Metadata.RenderFpsKey] = 50.0 };
                            break;
                    }

                    if (viaRunAsync)
                        Play.RunAsync(env, zoom: c.Zoom, keysToAction: c.Pairs is null ? null : ToDictionary(c.Pairs), seed: c.Seed).GetAwaiter().GetResult();
                    else
                        Play.Run(env, zoom: c.Zoom, keysToAction: c.Pairs is null ? null : ToDictionary(c.Pairs), seed: c.Seed);
                    return null;
                });
                ByteContract.AssertSame($"play refuses: {c.Label}", python, dotnet);
                Display.Quit();
                PythonRuntime.Run(() => {
                    using PyObject _ = Python.evalStmts("import pygame\npygame.quit()\npygame.display.init()", "None");
                });
            }
        }

        /// <summary>MountainCar-v0's <c>get_keys_to_action()</c>, Gymnasium's dict of tuples in its order, through <c>make</c>'s stack; MountainCarContinuous-v0 has none.</summary>
        [TestMethod]
        public void MountainCar_GetKeysToAction_MatchesGymnasium() {
            Outcome python = Outcome.CapturePython(() => Python.evalStmts("""
                import gymnasium as gym
                env = gym.make("MountainCar-v0")
                result = (env.has_wrapper_attr("get_keys_to_action"), list(env.get_wrapper_attr("get_keys_to_action")().items()), gym.make("MountainCarContinuous-v0").has_wrapper_attr("get_keys_to_action"))
                """, "result"));
            Outcome dotnet = Outcome.Capture(() => {
                using IEnv discrete = Gymnasium.Make("MountainCar-v0");
                using IEnv continuous = Gymnasium.Make("MountainCarContinuous-v0");
                // The mapping as its (key, action) items in order: the harness matches dict keys by their rendering, where a
                // Python int has no width and a .NET int does, so items compare value by value instead (the order too). The
                // keys are key combinations (tuples) and the actions 0-d int64 arrays (DR-45), released with the result.
                var mapping = (IReadOnlyDictionary<KeyCombination, NDArray>)PlayKeysCall(discrete.GetWrapperAttr("get_keys_to_action"))!;
                return new object?[] { discrete.HasWrapperAttr("get_keys_to_action"), mapping.Select(item => (object?)(item.Key, item.Value)).ToList(), continuous.HasWrapperAttr("get_keys_to_action") };
            });
            ByteContract.AssertSame("MountainCar's get_keys_to_action", python, dotnet);
        }

        /// <summary>
        ///     <c>PlayPlot</c> call by call through a matplotlib stand-in (<c>subplots</c>, <c>set_title</c>, <c>scatter</c>,
        ///     <c>remove</c>, <c>set_xlim</c>, <c>pause</c>), with its data and <c>t</c> afterwards: one and three plots over
        ///     horizons 3, 2 and 0; points from rewards and from transitions; Python's <c>zip(strict=True)</c> errors after the
        ///     points they paired; no plots (matplotlib's error); a negative horizon (the deque's error after the figure).
        /// </summary>
        /// <param name="scenario">Which plots, horizon and metrics.</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a scenario the cell does not define.</exception>
        [TestMethod]
        [DataRow("one")]
        [DataRow("three")]
        [DataRow("horizon0")]
        [DataRow("short")]
        [DataRow("long")]
        [DataRow("none")]
        [DataRow("negative")]
        public void PlayPlot_MatchesGymnasium(string scenario) {
            (string[] Names, int Horizon, double[] Rewards) setup = scenario switch {
                "one" => (new[] { "reward" }, 3, new[] { 1.0, -2.5, 3.0, 0.5, 4.0 }),
                "three" => (new[] { "a", "b", "c" }, 2, new[] { 0.25, 1.0, -1.0, 2.0 }),
                "horizon0" => (new[] { "a" }, 0, new[] { 1.0, 2.0 }),
                "short" => (new[] { "a", "b" }, 4, new[] { 1.0 }),
                "long" => (new[] { "a" }, 4, new[] { 1.0 }),
                "none" => (Array.Empty<string>(), 4, Array.Empty<double>()),
                "negative" => (new[] { "a" }, -1, Array.Empty<double>()),
                _ => throw new ArgumentOutOfRangeException(nameof(scenario)),
            };

            const string statements = """
                import numpy as np
                from gymnasium.utils import play as play_module
                log = []
                def keep_index(self, i):
                    self.i = i
                def remove(self):
                    log.append(("remove", self.i))
                Artist = type("Artist", (), {"__init__": keep_index, "remove": remove})
                def set_title(self, name):
                    log.append(("set_title", self.i, name))
                def scatter(self, x, y, c):
                    log.append(("scatter", self.i, [float(v) for v in x], [float(v) for v in y], c))
                    return Artist(self.i)
                def set_xlim(self, left, right):
                    log.append(("set_xlim", self.i, float(left), float(right)))
                Axes = type("Axes", (), {"__init__": keep_index, "set_title": set_title, "scatter": scatter, "set_xlim": set_xlim})
                def subplots(self, rows):
                    log.append(("subplots", rows))
                    if rows < 1:
                        raise ValueError(f"Number of rows must be a positive integer, not {rows}")
                    axes = [Axes(i) for i in range(rows)]
                    return "figure", (axes[0] if rows == 1 else np.array(axes, dtype=object))
                def pause(self, interval):
                    log.append(("pause", interval))
                StandInPlt = type("StandInPlt", (), {"subplots": subplots, "pause": pause})
                def metrics(obs_t, obs_tp1, action, rew, terminated, truncated, info):
                    if kind == "short":
                        return [rew]
                    if kind == "long":
                        return [rew, rew]
                    if kind == "three":
                        return [rew, -rew, rew * rew]
                    return [rew]
                original = play_module.plt
                play_module.plt = StandInPlt()
                error = None
                plot = None
                try:
                    plot = play_module.PlayPlot(metrics, horizon, names)
                    for step, rew in enumerate(rewards):
                        plot.callback(None, None, step, rew, False, False, {})
                except Exception as e:
                    error = (type(e).__name__, str(e))
                finally:
                    play_module.plt = original
                result = (tuple(log), None if plot is None else [list(series) for series in plot.data], None if plot is None else plot.t, error)
                """;
            Outcome python = Outcome.CapturePython(() => {
                using PyObject kind = PyConvert.ToPython(scenario);
                using PyObject names = PyConvert.ToPython(setup.Names.ToList());
                using PyObject horizon = PyConvert.ToPython(setup.Horizon);
                using PyObject rewards = PyConvert.ToPython(setup.Rewards.ToList());
                return Python.evalStmts(statements, "result", ("kind", kind), ("names", names), ("horizon", horizon), ("rewards", rewards));
            });

            Outcome dotnet = Outcome.Capture(() => {
                var log = new List<object?>();
                IEnumerable<double> Metrics(object? obsT, object? obsTp1, object? action, double rew, bool terminated, bool truncated, Info info) => scenario switch {
                    "short" => new[] { rew },
                    "long" => new[] { rew, rew },
                    "three" => new[] { rew, -rew, rew * rew },
                    _ => new[] { rew },
                };

                PlayPlot? plot = null;
                object? error = null;
                try {
                    plot = new PlayPlot(Metrics, setup.Horizon, setup.Names, rows => new StandInFigure(log, rows));
                    for (int step = 0; step < setup.Rewards.Length; step++)
                        plot.Callback(null, null, step, setup.Rewards[step], false, false, new Info());
                } catch (ValueError e) {
                    error = ("ValueError", e.Message);
                }

                return new object?[] {
                    log.ToArray(), plot?.Data.Select(series => (object?)series.ToList()).ToList(), plot?.T, error,
                };
            });
            ByteContract.AssertSame($"PlayPlot ({scenario})", python, dotnet);
        }

        /// <summary>One <c>play</c> run's settings, shared by both sides.</summary>
        private sealed class PlayScenario {
            /// <summary>The mapping's <c>(key, action)</c> pairs.</summary>
            public object?[] Pairs { get; set; } = Array.Empty<object?>();

            /// <summary>The events each callback posts, by call; the call after the last posts <c>QUIT</c>.</summary>
            public (int Type, object? Attributes)[][] Script { get; set; } = Array.Empty<(int, object?)[]>();

            /// <summary>Events posted before <c>play</c> starts (pygame initialized first on the Python side).</summary>
            public (int Type, object? Attributes)[] Before { get; set; } = Array.Empty<(int, object?)>();

            /// <summary>Whether the mapping comes from a wrapper's <c>get_keys_to_action</c> instead of the argument.</summary>
            public bool Wrapper { get; set; }

            /// <summary>The render mode.</summary>
            public string RenderMode { get; set; } = "rgb_array";

            /// <summary>The zoom; small, so the recorded displays stay small.</summary>
            public double? Zoom { get; set; } = 0.2;

            /// <summary>The frame rate; <see langword="null"/> reads the metadata's 50.</summary>
            public double? Fps { get; set; }

            /// <summary>The no-op, or <see langword="null"/> for the default 0.</summary>
            public object? Noop { get; set; }

            /// <summary>Whether frames are transposed.</summary>
            public bool? Transpose { get; set; } = true;

            /// <summary>Whether play waits for a held key.</summary>
            public bool WaitOnPlayer { get; set; }
        }

        /// <summary>Gymnasium's <c>play</c> under the scenario: every callback's arguments and the display at that moment.</summary>
        /// <param name="setup">The scenario.</param>
        /// <returns>The described records, or the error.</returns>
        private static Outcome PlayPython(PlayScenario setup) {
            const string statements = """
                import gymnasium as gym
                import pygame
                from gymnasium.utils.play import play
                def wrapper_init(self, env, keys_to_action):
                    gym.Wrapper.__init__(self, env)
                    self.keys_to_action = keys_to_action
                def get_keys_to_action(self):
                    return self.keys_to_action
                KeysToActionWrapper = type("KeysToActionWrapper", (gym.Wrapper,), {"__init__": wrapper_init, "get_keys_to_action": get_keys_to_action})
                env = gym.make("CartPole-v1", render_mode=render_mode, disable_env_checker=True)
                keys = dict(pairs)
                if wrapper:
                    env = KeysToActionWrapper(env, keys)
                records = []
                calls = [0]
                def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
                    records.append((obs_t, obs_tp1, action, rew, terminated, truncated, info, pygame.surfarray.array3d(pygame.display.get_surface())))
                    i = calls[0]
                    calls[0] += 1
                    for event_type, attributes in (script[i] if i < len(script) else [(pygame.QUIT, None)]):
                        pygame.event.post(pygame.event.Event(event_type, attributes or {}))
                pygame.init()
                for event_type, attributes in before:
                    pygame.event.post(pygame.event.Event(event_type, attributes or {}))
                play(env, transpose=transpose, fps=fps, zoom=zoom, callback=callback, keys_to_action=None if wrapper else keys, seed=7,
                     noop=0 if noop is None else noop, wait_on_player=wait)
                env.close()
                result = tuple(records)
                """;
            return Outcome.CapturePython(() => {
                using PyObject renderMode = PyConvert.ToPython(setup.RenderMode);
                using PyObject pairs = PyConvert.ToPython(setup.Pairs.ToList());
                using PyObject wrapper = PyConvert.ToPython(setup.Wrapper);
                using PyObject script = PyConvert.ToPython(setup.Script.Select(events => (object?)events.Select(e => (object?)(e.Type, AttributesOf(e.Attributes))).ToList()).ToList());
                using PyObject before = PyConvert.ToPython(setup.Before.Select(e => (object?)(e.Type, AttributesOf(e.Attributes))).ToList());
                using PyObject transpose = PyConvert.ToPython(setup.Transpose);
                using PyObject fps = PyConvert.ToPython(setup.Fps);
                using PyObject zoom = PyConvert.ToPython(setup.Zoom);
                using PyObject noop = PyConvert.ToPython(setup.Noop);
                using PyObject wait = PyConvert.ToPython(setup.WaitOnPlayer);
                return Python.evalStmts(statements, "result", ("render_mode", renderMode), ("pairs", pairs), ("wrapper", wrapper), ("script", script),
                    ("before", before), ("transpose", transpose), ("fps", fps), ("zoom", zoom), ("noop", noop), ("wait", wait));
            });
        }

        /// <summary>Gym.NET's <see cref="Play.Run"/> under the scenario: every callback's arguments, described at the call, and the display then.</summary>
        /// <param name="setup">The scenario.</param>
        /// <param name="viaRunAsync">Play through <see cref="Play.RunAsync"/> and wait for its task, instead of <see cref="Play.Run"/>.</param>
        /// <returns>The described records, or the error.</returns>
        private static Outcome PlayDotNet(PlayScenario setup, bool viaRunAsync = false) {
            var records = new List<ValueFacts>();
            int calls = 0;

            // Describes the borrowed arguments at once (play releases them afterwards), posting the scripted events.
            void Callback(object? obsT, object? obsTp1, object? action, double rew, bool terminated, bool truncated, Info info) {
                object? previous = obsT is ResetResult<object?> reset ? new object?[] { reset.Observation, reset.Info } : obsT;
                using NDArray screen = SurfArray.Array3d(Display.GetSurface()!);
                records.Add(ValueFacts.FromSequence(new[] {
                    ValueFacts.Of(previous), ValueFacts.Of(obsTp1), ValueFacts.Of(action), ValueFacts.Of(rew), ValueFacts.Of(terminated), ValueFacts.Of(truncated),
                    ValueFacts.Of(info), ValueFacts.Of(screen),
                }, "tuple"));
                int i = calls++;
                foreach ((int type, object? attributes) in i < setup.Script.Length ? setup.Script[i] : new[] { (EventType.Quit, (object?)null) })
                    EventQueue.Post(new Event(type, attributes));
            }

            try {
                using var made = (Env<NDArray, NDArray>)Gymnasium.Make("CartPole-v1", new { render_mode = setup.RenderMode }, disableEnvChecker: true);
                Dictionary<object, object?> keys = ToDictionary(setup.Pairs);
                IEnv env = setup.Wrapper ? new KeysToActionWrapper(made, keys) : made;
                foreach ((int type, object? attributes) in setup.Before)
                    EventQueue.Post(new Event(type, attributes));
                if (viaRunAsync) {
                    // GetResult rethrows the task's failure as thrown, so a refusal is compared as Run's would be.
                    Play.RunAsync(env, transpose: setup.Transpose, fps: setup.Fps, zoom: setup.Zoom, callback: Callback, keysToAction: setup.Wrapper ? null : keys,
                        seed: 7, noop: setup.Noop, waitOnPlayer: setup.WaitOnPlayer).GetAwaiter().GetResult();
                } else {
                    Play.Run(env, transpose: setup.Transpose, fps: setup.Fps, zoom: setup.Zoom, callback: Callback, keysToAction: setup.Wrapper ? null : keys, seed: 7,
                        noop: setup.Noop, waitOnPlayer: setup.WaitOnPlayer);
                }
            } catch (Exception e) {
                return Outcome.Capture(() => throw e);
            }

            return Outcome.Returned(ValueFacts.FromSequence(records, "tuple"));
        }

        /// <summary>
        ///     Gymnasium's test wrapper: <c>get_keys_to_action</c> returns a given mapping. The method is typed (DR-45), key
        ///     combinations to the env's actions, so the wrapper holds a tuple key as its codes and an int action as a 0-d int64
        ///     array, which the harness compares with Gymnasium's Python int by value.
        /// </summary>
        private sealed class KeysToActionWrapper : Wrapper<NDArray, NDArray> {
            /// <summary>The mapping as key combinations and the actions' values, in the given order.</summary>
            private readonly List<(KeyCombination Keys, long Action)> _mapping;

            /// <summary>Wraps <paramref name="env"/>.</summary>
            /// <param name="env">The env.</param>
            /// <param name="keysToAction">The mapping: tuple keys of int codes, int actions.</param>
            /// <exception cref="InvalidCastException">Thrown for a key that is no tuple of ints, or an action that is no int.</exception>
            public KeysToActionWrapper(Env<NDArray, NDArray> env, Dictionary<object, object?> keysToAction) : base(env) =>
                _mapping = keysToAction.Select(entry => (Combination((System.Runtime.CompilerServices.ITuple)entry.Key), Convert.ToInt64(entry.Value))).ToList();

            /// <summary>The wrapper's <c>get_keys_to_action</c>: the mapping, with new action arrays on every call (play owns and releases them).</summary>
            /// <returns>The mapping.</returns>
            public override IReadOnlyDictionary<KeyCombination, NDArray> GetKeysToAction() =>
                _mapping.ToDictionary(entry => entry.Keys, entry => NDArray.Scalar(entry.Action));

            /// <summary>A tuple key's codes.</summary>
            /// <param name="key">The tuple.</param>
            /// <returns>The combination.</returns>
            /// <exception cref="InvalidCastException">Thrown for an item that is no int.</exception>
            private static KeyCombination Combination(System.Runtime.CompilerServices.ITuple key) =>
                new(Enumerable.Range(0, key.Length).Select(i => (int)key[i]!).ToArray());
        }

        /// <summary>The matplotlib stand-in's .NET twin: a figure logging <see cref="PlayPlot"/>'s calls as the Python stand-in logs them.</summary>
        private sealed class StandInFigure : IPlotFigure {
            /// <summary>The shared log.</summary>
            private readonly List<object?> _log;

            /// <summary><c>subplots(rows)</c>: logs the call, refuses no rows with matplotlib's message, makes the axes.</summary>
            /// <param name="log">The log.</param>
            /// <param name="rows">The number of axes.</param>
            /// <exception cref="ValueError">Thrown for fewer than one row.</exception>
            public StandInFigure(List<object?> log, int rows) {
                _log = log;
                _log.Add(("subplots", rows));
                if (rows < 1)
                    throw new ValueError($"Number of rows must be a positive integer, not {rows}");
                Axes = Enumerable.Range(0, rows).Select(i => (IPlotAxes)new StandInAxes(log, i)).ToList();
            }

            /// <inheritdoc />
            public IReadOnlyList<IPlotAxes> Axes { get; }

            /// <summary>Logs the pause.</summary>
            /// <param name="interval">The interval.</param>
            public void Pause(double interval) => _log.Add(("pause", interval));
        }

        /// <summary>A logging axes.</summary>
        /// <param name="Log">The shared log.</param>
        /// <param name="Index">The axes' index.</param>
        private sealed record StandInAxes(List<object?> Log, int Index) : IPlotAxes {
            /// <summary>Logs the title.</summary>
            /// <param name="title">The title.</param>
            public void SetTitle(string title) => Log.Add(("set_title", Index, title));

            /// <summary>Logs the scatter.</summary>
            /// <param name="x">The x values.</param>
            /// <param name="y">The y values.</param>
            /// <param name="color">The colour.</param>
            /// <returns>An artist that logs its removal.</returns>
            public IPlotArtist Scatter(IReadOnlyList<double> x, IReadOnlyList<double> y, string color) {
                Log.Add(("scatter", Index, x.ToList(), y.ToList(), color));
                return new StandInArtist(Log, Index);
            }

            /// <summary>Logs the limits.</summary>
            /// <param name="left">The left limit.</param>
            /// <param name="right">The right limit.</param>
            public void SetXLim(double left, double right) => Log.Add(("set_xlim", Index, left, right));
        }

        /// <summary>A logging scatter.</summary>
        /// <param name="Log">The shared log.</param>
        /// <param name="Index">The axes' index.</param>
        private sealed record StandInArtist(List<object?> Log, int Index) : IPlotArtist {
            /// <summary>Logs the removal.</summary>
            public void Remove() => Log.Add(("remove", Index));
        }

        /// <summary>A key event of the script.</summary>
        /// <param name="type">The event type.</param>
        /// <param name="key">The key code.</param>
        /// <returns>The type and its attributes.</returns>
        private static (int Type, object? Attributes) Key(int type, int key) => (type, new { key });

        /// <summary>The default script: press and release <paramref name="a"/>, then <paramref name="d"/>, then <paramref name="a"/> twice, as Gymnasium's test does.</summary>
        /// <param name="a">The first key.</param>
        /// <param name="d">The second key.</param>
        /// <returns>The events each callback posts.</returns>
        private static (int Type, object? Attributes)[][] AlternateKeys(int a, int d) => new[] {
            new[] { Key(EventType.KeyDown, a) }, new[] { Key(EventType.KeyUp, a), Key(EventType.KeyDown, d) },
            new[] { Key(EventType.KeyUp, d), Key(EventType.KeyDown, a) }, new[] { Key(EventType.KeyUp, a), Key(EventType.KeyDown, a) },
            new[] { Key(EventType.KeyUp, a), Key(EventType.KeyDown, d) }, new[] { Key(EventType.KeyUp, d) },
        };

        /// <summary>Mapping pairs as a list of <c>(key, action)</c> tuples, which cross to Python as a list of tuples for <c>dict(pairs)</c>.</summary>
        /// <param name="pairs">The pairs.</param>
        /// <returns>The pairs as objects.</returns>
        private static object?[] Pairs(params (object Key, object? Action)[] pairs) => pairs.Select(p => (object?)p).ToArray();

        /// <summary>The .NET mapping of <see cref="Pairs"/>' tuples, in the same order.</summary>
        /// <param name="pairs">The pairs.</param>
        /// <returns>A new dictionary.</returns>
        private static Dictionary<object, object?> ToDictionary(object?[] pairs) {
            var dictionary = new Dictionary<object, object?>();
            foreach ((object key, object? action) in pairs.Cast<(object, object?)>())
                dictionary[key] = action;
            return dictionary;
        }

        /// <summary>An anonymous attribute object as a string-keyed dict for Python (<see langword="null"/> stays <c>None</c>).</summary>
        /// <param name="attributes">The attributes.</param>
        /// <returns>The dict, or <see langword="null"/>.</returns>
        private static Info? AttributesOf(object? attributes) => attributes is null ? null : PyKwargs.From(attributes);

        /// <summary>A key set sorted as Python's <c>sorted</c> sorts ints.</summary>
        /// <param name="keys">The keys, ints.</param>
        /// <returns>A new list.</returns>
        private static List<object?> Sorted(ISet<object?> keys) => keys.OrderBy(k => Convert.ToInt64(k, System.Globalization.CultureInfo.InvariantCulture)).ToList();

        /// <summary>Python's literal for a boolean, for the snippets.</summary>
        /// <param name="value">The value.</param>
        /// <returns><c>True</c> or <c>False</c>.</returns>
        private static string PyBool(bool value) => value ? "True" : "False";

        /// <summary>Calls an attribute that holds a method, as Python's <c>getattr(...)()</c>.</summary>
        /// <param name="attribute">A delegate.</param>
        /// <returns>What it returned.</returns>
        private static object? PlayKeysCall(object? attribute) => ((Delegate)attribute!).DynamicInvoke();

        /// <summary>A random frame: <c>(height, width, 3)</c> RGB, or <c>(height, width)</c> palette indices for no channels.</summary>
        /// <param name="random">The generator.</param>
        /// <param name="height">The rows.</param>
        /// <param name="width">The columns.</param>
        /// <param name="channels">3, or 0 for a 2-D frame.</param>
        /// <returns>A new uint8 array the caller disposes.</returns>
        private static NDArray RandomFrame(Random random, int height, int width, int channels) {
            var bytes = new byte[height * width * Math.Max(channels, 1)];
            random.NextBytes(bytes);
            using NDArray flat = np.array(bytes);
            return channels == 0 ? flat.reshape(height, width) : flat.reshape(height, width, channels);
        }
    }
}
