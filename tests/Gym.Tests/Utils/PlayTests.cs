#nullable enable
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Gym.Environments;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Rendering;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using SixLabors.ImageSharp;
using RColor = Gym.Rendering.Color;

namespace Gym.Tests.Utils {
    /// <summary>
    ///     Gym.NET's own parts of <c>play</c>: pygame's event queue and display (<see cref="EventQueue"/>,
    ///     <see cref="Display"/>) with a scripted interactive window, <c>make_surface</c>, <c>display_arr</c>'s
    ///     letterboxing, a whole game driven by a window's keys, <see cref="PlayPlot"/> over a recording figure, and
    ///     MountainCar's key mapping.
    /// </summary>
    /// <remarks>
    ///     The display, the queue and the viewer factory are process-wide, so the class never runs in parallel; every test
    ///     quits the display afterwards. Parity with pygame and Gymnasium is proven in the live suite
    ///     (<c>PlayLiveParityTests</c>); these tests pin the .NET-only behaviour (the window's input, ownership).
    /// </remarks>
    [TestClass]
    [DoNotParallelize]
    public class PlayTests {
        /// <summary>The viewer factory before the test, restored by <see cref="Cleanup"/>.</summary>
        private IEnvironmentViewerFactoryDelegate? _previousFactory;

        /// <summary>Points the process-wide viewer factory at <see cref="NullEnvViewer"/> and empties the queue.</summary>
        [TestInitialize]
        public void Initialize() {
            _previousFactory = HumanViewerFactory;
            HumanViewerFactory = NullEnvViewer.Factory;
            EventQueue.Clear();
        }

        /// <summary>Quits the display and restores the factory.</summary>
        [TestCleanup]
        public void Cleanup() {
            Display.Quit();
            HumanViewerFactory = _previousFactory;
        }

        /// <summary>Posted events come back in order, once; the count and <see cref="EventQueue.Clear"/> agree.</summary>
        [TestMethod]
        public void EventQueue_PostAndGet_InOrderOnce() {
            var first = new Event(EventType.KeyDown, new { key = 97 });
            var second = new Event(EventType.Quit);
            EventQueue.Post(first);
            EventQueue.Post(second);
            Assert.AreEqual(2, EventQueue.Count);
            CollectionAssert.AreEqual(new[] { first, second }, EventQueue.Get().ToArray());
            Assert.AreEqual(0, EventQueue.Get().Count);

            EventQueue.Post(first);
            EventQueue.Clear();
            Assert.AreEqual(0, EventQueue.Count);
            Assert.ThrowsExactly<ArgumentNullException>(() => EventQueue.Post(null!));
        }

        /// <summary>An event prints, compares and reports a missing attribute as pygame's does, and refuses a <c>type</c> attribute.</summary>
        [TestMethod]
        public void Event_ReprEqualityAndAttributes_ArePygames() {
            var down = new Event(EventType.KeyDown, new { key = 97 });
            Assert.AreEqual("<Event(768-KeyDown {'key': 97})>", down.ToString());
            Assert.AreEqual("<Event(256-Quit {})>", new Event(EventType.Quit).ToString());
            Assert.AreEqual("<Event(32777-WindowResized {'x': 25, 'y': 30})>", new Event(EventType.WindowResized, new { x = 25, y = 30 }).ToString());
            Assert.AreEqual("<Event(32866-UserEvent {})>", new Event(EventType.UserEvent + 1).ToString());
            Assert.AreEqual("<Event(12345-Unknown {})>", new Event(12345).ToString());

            Assert.AreEqual(down, new Event(EventType.KeyDown, new Info { ["key"] = 97L }));
            Assert.AreNotEqual(down, new Event(EventType.KeyDown, new { key = 98 }));
            Assert.AreNotEqual(down, new Event(EventType.KeyUp, new { key = 97 }));
            Assert.IsFalse(new Event(EventType.NoEvent).IsTruthy);
            Assert.IsTrue(down.IsTruthy);

            Assert.AreEqual(97, PyAttributes.GetAttr(down, "key"));
            Assert.AreEqual(EventType.KeyDown, PyAttributes.GetAttr(down, "type"));
            PyAttributes.SetAttr(down, "key", 99);
            Assert.AreEqual("<Event(768-KeyDown {'key': 99})>", down.ToString());
            Assert.AreEqual("'pygame.event.Event' object has no attribute 'x'", Assert.ThrowsExactly<AttributeError>(() => PyAttributes.GetAttr(down, "x")).Message);
            Assert.AreEqual("redundant type field in event dict", Assert.ThrowsExactly<ValueError>(() => new Event(EventType.KeyDown, new Info { ["type"] = 5 })).Message);
        }

        /// <summary>
        ///     <see cref="Display.SetMode"/> opens a black surface of the size through the factory and starts the window's
        ///     input; <see cref="Display.Flip"/> hands the window the pixels; <see cref="Display.Quit"/> closes it and drops the
        ///     pending events.
        /// </summary>
        [TestMethod]
        public void Display_SetModeFlipQuit_DriveTheWindow() {
            Assert.IsNull(Display.GetSurface());
            Assert.AreEqual("Display mode not set", Assert.ThrowsExactly<SdlError>(Display.Flip).Message);

            var window = new ScriptedViewer();
            Surface surface = Display.SetMode(4, 3, resizable: true, window.Factory, "title");
            Assert.AreSame(surface, Display.GetSurface());
            Assert.IsTrue(Display.GetInit());
            Assert.AreEqual((4, 3, "title"), (window.Width, window.Height, window.Title));
            Assert.IsTrue(window.InputStarted);
            Assert.AreEqual(new RColor(0, 0, 0), surface.GetAt(3, 2));

            surface.Fill(new RColor(10, 20, 30));
            Display.Flip();
            Assert.AreEqual(1, window.Frames.Count);
            Assert.AreEqual((4, 3), window.Frames[0].Size);
            Assert.IsTrue(window.Frames[0].Pixels.All(p => p == (10, 20, 30)));

            EventQueue.Post(new Event(EventType.Quit));
            Display.Quit();
            Assert.IsTrue(window.Closed);
            Assert.IsNull(Display.GetSurface());
            Assert.IsFalse(Display.GetInit());
            Assert.AreEqual(0, EventQueue.Count);
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => Display.SetMode(0, 3));
        }

        /// <summary>
        ///     A window's keys and close button reach the queue as pygame's events when it is pumped; its resize replaces the
        ///     surface first (resizable displays only), where a posted <c>WINDOWRESIZED</c> changes nothing, as in pygame.
        /// </summary>
        [TestMethod]
        public void Display_PumpsTheWindowsInput_AsPygameEvents() {
            var window = new ScriptedViewer();
            Surface before = Display.SetMode(10, 10, resizable: true, window.Factory);

            EventQueue.Post(new Event(EventType.WindowResized, new { x = 30, y = 40 }));
            window.Inputs.Enqueue(ViewerInput.KeyDown(KeyCode.Left));
            window.Inputs.Enqueue(ViewerInput.KeyUp(KeyCode.Left));
            window.Inputs.Enqueue(ViewerInput.Resized(25, 15));
            window.Inputs.Enqueue(ViewerInput.Close());
            IReadOnlyList<Event> events = EventQueue.Get();

            // The posted event comes first: the window's input joins the queue when it is pumped, at Get.
            CollectionAssert.AreEqual(new[] {
                new Event(EventType.WindowResized, new { x = 30, y = 40 }), new Event(EventType.KeyDown, new { key = KeyCode.Left }),
                new Event(EventType.KeyUp, new { key = KeyCode.Left }), new Event(EventType.WindowResized, new { x = 25, y = 15 }),
                new Event(EventType.WindowClose), new Event(EventType.Quit),
            }, events.ToArray());
            Surface after = Display.GetSurface()!;
            Assert.AreNotSame(before, after);
            Assert.AreEqual((25, 15), (after.Width, after.Height));

            // A display that is not resizable keeps its surface and posts no resize.
            Display.SetMode(10, 10, resizable: false, window.Factory);
            window.Inputs.Enqueue(ViewerInput.Resized(25, 15));
            Assert.AreEqual(0, EventQueue.Get().Count);
            Assert.AreEqual(10, Display.GetSurface()!.Width);
        }

        /// <summary>
        ///     A whole game from a window's keys: the held key's action plays, Escape ends it, the display is quit, and
        ///     the arrays play handed out were released (only the callback's copies survive).
        /// </summary>
        [TestMethod]
        public void Run_PlaysTheWindowsKeys_AndReleasesWhatItDrops() {
            var window = new ScriptedViewer();
            var actions = new List<object?>();
            var observations = new List<NDArray>();
            using IEnv env = Make("CartPole-v1", new { render_mode = "rgb_array" });

            void Callback(object? obsT, object? obsTp1, object? action, double rew, bool terminated, bool truncated, Info info) {
                actions.Add(action);
                observations.Add(((NDArray)obsTp1!).copy());
                // The window reports what a player would type: hold 'd' for two steps, then release it and press Escape.
                switch (actions.Count) {
                    case 1:
                        window.Inputs.Enqueue(ViewerInput.KeyDown(KeyCode.D));
                        break;
                    case 3:
                        window.Inputs.Enqueue(ViewerInput.KeyUp(KeyCode.D));
                        window.Inputs.Enqueue(ViewerInput.KeyDown(KeyCode.Escape));
                        break;
                }
            }

            Play.Run(env, fps: 1000, callback: Callback, keysToAction: new Dictionary<object, object?> { ["a"] = 0, ["d"] = 1 }, seed: 3, viewerFactory: window.Factory);

            CollectionAssert.AreEqual(new object?[] { 0, 1, 1 }, actions);
            Assert.IsTrue(window.Closed);
            Assert.IsFalse(Display.GetInit());
            Assert.IsTrue(window.Frames.Count >= 3);
            Assert.AreEqual((600, 400), window.Frames[0].Size);
            foreach (NDArray observation in observations)
                observation.Dispose();
        }

        /// <summary><c>make_surface</c> reads <c>[x][y]</c> RGB, maps 2-D values through pygame's 8-bit palette, and refuses other shapes and dtypes.</summary>
        [TestMethod]
        public void MakeSurface_ReadsPygamesAxisOrderAndPalette() {
            using NDArray rgb = np.array(Enumerable.Range(0, 2 * 3 * 3).Select(i => (byte)i).ToArray()).reshape(2, 3, 3);
            Surface surface = SurfArray.MakeSurface(rgb);
            Assert.AreEqual((2, 3), (surface.Width, surface.Height));
            Assert.AreEqual(new RColor(15, 16, 17), surface.GetAt(1, 2));

            using NDArray indices = np.array(new byte[,] { { 0, 100 }, { 200, 255 } });
            Surface palette = SurfArray.MakeSurface(indices);
            Assert.AreEqual(new RColor(0, 0, 0), palette.GetAt(0, 0));
            Assert.AreEqual(new RColor(255, 36, 0), palette.GetAt(0, 1));
            Assert.AreEqual(new RColor(170, 73, 0), palette.GetAt(1, 0));
            Assert.AreEqual(new RColor(255, 255, 255), palette.GetAt(1, 1));

            using NDArray fourChannels = np.zeros(new Shape(2, 2, 4), np.uint8);
            Assert.AreEqual("must be a valid 2d or 3d array\n", Assert.ThrowsExactly<ValueError>(() => SurfArray.MakeSurface(fourChannels)).Message);
            using NDArray oneDimension = np.zeros(new Shape(3), np.uint8);
            Assert.ThrowsExactly<ValueError>(() => SurfArray.MakeSurface(oneDimension));
            using NDArray floats = np.zeros(new Shape(2, 2, 3));
            Assert.ThrowsExactly<NotSupportedException>(() => SurfArray.MakeSurface(floats));
        }

        /// <summary>
        ///     <c>display_arr</c> scales the frame and centres it on black, cropping a frame larger than the screen by half its
        ///     excess truncated toward zero; a non-uint8 frame fails the bare assertion.
        /// </summary>
        [TestMethod]
        public void DisplayArr_LetterboxesAndCrops() {
            // A 2 x 1 white frame (height 1, width 2) shown 4 x 2 on a 7 x 4 screen: offsets (7 - 4) / 2 = 1 and (4 - 2) / 2 = 1.
            using NDArray frame = np.full(new Shape(1, 2, 3), (byte)255, np.uint8);
            var screen = new Surface(7, 4);
            screen.Fill(new RColor(9, 9, 9));
            Play.DisplayArr(screen, frame, (4, 2), transpose: true);
            for (int x = 0; x < 7; x++) {
                for (int y = 0; y < 4; y++) {
                    bool inside = x >= 1 && x < 5 && y >= 1 && y < 3;
                    Assert.AreEqual(inside ? new RColor(255, 255, 255) : new RColor(0, 0, 0), screen.GetAt(x, y), $"({x}, {y})");
                }
            }

            // Larger than the screen: (3 - 6) / 2 truncates to -1, so the frame starts one column left of the screen.
            using NDArray ramp = np.array(Enumerable.Range(0, 6).SelectMany(i => new[] { (byte)i, (byte)i, (byte)i }).ToArray()).reshape(1, 6, 3);
            var small = new Surface(3, 1);
            Play.DisplayArr(small, ramp, (6, 1), transpose: true);
            Assert.AreEqual(new RColor(1, 1, 1), small.GetAt(0, 0));
            Assert.AreEqual(new RColor(3, 3, 3), small.GetAt(2, 0));

            using NDArray floats = np.ones(new Shape(1, 2, 3));
            Assert.ThrowsExactly<AssertionError>(() => Play.DisplayArr(screen, floats, (4, 2), transpose: true));
        }

        /// <summary>
        ///     <see cref="PlayPlot"/> over a recording figure: titles, the data each callback appends, the scatters it removes
        ///     and redraws over the horizon, the x limits, the pause; Python's <c>zip(strict=True)</c> errors after the points it
        ///     could pair.
        /// </summary>
        [TestMethod]
        public void PlayPlot_DrawsTheHorizon_AndChecksThePointCount() {
            var figure = new RecordingFigure();
            int calls = 0;
            var plot = new PlayPlot((_, _, _, rew, _, _, _) => new[] { rew, ++calls * 10.0 }, 2, new[] { "reward", "count" }, rows => figure.Make(rows));
            CollectionAssert.AreEqual(new[] { "set_title(0, reward)", "set_title(1, count)" }, figure.Log);

            for (int step = 1; step <= 3; step++)
                plot.Callback(null, null, 0, step, false, false, new Info());

            Assert.AreEqual(3, plot.T);
            CollectionAssert.AreEqual(new[] { 2.0, 3.0 }, plot.Data[0].ToArray());
            CollectionAssert.AreEqual(new[] { 20.0, 30.0 }, plot.Data[1].ToArray());
            CollectionAssert.AreEqual(new[] {
                "set_title(0, reward)", "set_title(1, count)",
                "scatter(0, [0], [1], blue)", "set_xlim(0, 0, 1)", "scatter(1, [0], [10], blue)", "set_xlim(1, 0, 1)", "pause(1E-06)",
                "remove(0)", "scatter(0, [0,1], [1,2], blue)", "set_xlim(0, 0, 2)", "remove(1)", "scatter(1, [0,1], [10,20], blue)", "set_xlim(1, 0, 2)", "pause(1E-06)",
                "remove(0)", "scatter(0, [1,2], [2,3], blue)", "set_xlim(0, 1, 3)", "remove(1)", "scatter(1, [1,2], [20,30], blue)", "set_xlim(1, 1, 3)", "pause(1E-06)",
            }, figure.Log);

            // One point for two plots: the first plot gets it, then zip refuses; t does not move.
            var shortPlot = new PlayPlot((_, _, _, rew, _, _, _) => new[] { rew }, 5, new[] { "a", "b" }, rows => new RecordingFigure().Make(rows));
            Assert.AreEqual("zip() argument 2 is longer than argument 1", Assert.ThrowsExactly<ValueError>(() => shortPlot.Callback(null, null, 0, 7, false, false, new Info())).Message);
            Assert.AreEqual(1, shortPlot.Data[0].Count);
            Assert.AreEqual(0, shortPlot.Data[1].Count);
            Assert.AreEqual(0, shortPlot.T);

            var longPlot = new PlayPlot((_, _, _, rew, _, _, _) => new[] { rew, rew }, 5, new[] { "a" }, rows => new RecordingFigure().Make(rows));
            Assert.AreEqual("zip() argument 2 is shorter than argument 1", Assert.ThrowsExactly<ValueError>(() => longPlot.Callback(null, null, 0, 7, false, false, new Info())).Message);
            Assert.AreEqual(1, longPlot.Data[0].Count);

            Assert.AreEqual("maxlen must be non-negative", Assert.ThrowsExactly<ValueError>(() => new PlayPlot((_, _, _, _, _, _, _) => Array.Empty<double>(), -1, new[] { "a" }, rows => new RecordingFigure().Make(rows))).Message);
        }

        /// <summary>
        ///     The default figure needs a viewer factory (Gymnasium needs matplotlib), refuses zero plots with matplotlib's
        ///     message, and shows a 640 by 480 figure at each pause.
        /// </summary>
        [TestMethod]
        public void PlotFigure_ShowsInAViewer_AndRefusesZeroRows() {
            var window = new ScriptedViewer();
            var plot = new PlayPlot((_, _, _, rew, _, _, _) => new[] { rew }, 10, new[] { "reward" }, PlotFigure.Subplots(window.Factory));
            plot.Callback(null, null, 0, 1.5, false, false, new Info());
            plot.Callback(null, null, 0, -0.5, false, false, new Info());
            Assert.AreEqual(2, window.Frames.Count);
            Assert.AreEqual((PlotFigure.Width, PlotFigure.Height), window.Frames[1].Size);
            Assert.AreEqual("reward", window.Title);
            // Blue markers were drawn somewhere inside the axes.
            Assert.IsTrue(window.Frames[1].Pixels.Any(p => p == (0, 0, 255)));

            Assert.AreEqual("Number of rows must be a positive integer, not 0", Assert.ThrowsExactly<ValueError>(() => new PlotFigure(0, window.Factory)).Message);
            HumanViewerFactory = null;
            Assert.ThrowsExactly<DependencyNotInstalled>(() => new PlayPlot((_, _, _, rew, _, _, _) => new[] { rew }, 10, new[] { "reward" }));
        }

        /// <summary>
        ///     MountainCar-v0 maps keys as Gymnasium's (pygame 1's arrow codes, with the empty combination that makes
        ///     <c>play</c> refuse it); MountainCarContinuous-v0 has no mapping at all, so <c>play</c> reports it missing.
        /// </summary>
        [TestMethod]
        public void MountainCar_KeysToAction_IsDiscreteOnly() {
            using var discrete = new MountainCarEnv(renderMode: "rgb_array");
            Dictionary<object, object?> mapping = discrete.GetKeysToAction();
            Assert.AreEqual("{(): 1, (276,): 0, (275,): 2, (275, 276): 1}", PyBuiltins.Repr(mapping));

            using IEnv made = Make("MountainCar-v0", new { render_mode = "rgb_array" });
            Assert.IsTrue(made.HasWrapperAttr("get_keys_to_action"));
            Assert.AreEqual("Key combinations in keys_to_action tuples cannot be empty", Assert.ThrowsExactly<ValueError>(() => Play.Run(made)).Message);
            var game = new PlayableGame(made);
            Assert.AreEqual(2, game.RelevantKeys.Count);
            Assert.IsTrue(game.RelevantKeys.Contains(275) && game.RelevantKeys.Contains(276));
            Display.Quit();

            using IEnv continuous = Make("MountainCarContinuous-v0", new { render_mode = "rgb_array" });
            Assert.IsFalse(continuous.HasWrapperAttr("get_keys_to_action"));
            Assert.AreEqual("'Continuous_MountainCarEnv' object has no attribute 'get_keys_to_action'",
                Assert.ThrowsExactly<AttributeError>(() => ((MountainCarEnv)continuous.Unwrapped).GetKeysToAction()).Message);
            Assert.AreEqual("MountainCarContinuous-v0 does not have explicit key to action mapping, please specify one manually",
                Assert.ThrowsExactly<MissingKeysToAction>(() => Play.Run(continuous)).Message);
        }

        /// <summary>
        ///     A viewer whose input a test scripts: it records the frames it is shown (size and pixels), its title, whether
        ///     input started and whether it was closed, and hands out the inputs a test queues.
        /// </summary>
        private sealed class ScriptedViewer : IInteractiveEnvViewer {
            /// <summary>The inputs <see cref="TakeInput"/> hands out next, queued by the test.</summary>
            public ConcurrentQueue<ViewerInput> Inputs { get; } = new();

            /// <summary>Every frame shown: its size and its pixels in row order.</summary>
            public List<((int Width, int Height) Size, (byte R, byte G, byte B)[] Pixels)> Frames { get; } = new();

            /// <summary>The width the factory was asked for.</summary>
            public int Width { get; private set; }

            /// <summary>The height the factory was asked for.</summary>
            public int Height { get; private set; }

            /// <summary>The title the factory was given.</summary>
            public string? Title { get; private set; }

            /// <summary>Whether <see cref="BeginInput"/> ran.</summary>
            public bool InputStarted { get; private set; }

            /// <summary>Whether <see cref="CloseEnvironment"/> ran.</summary>
            public bool Closed { get; private set; }

            /// <summary>A factory that hands out this viewer, recording the size and title.</summary>
            /// <param name="width">The width.</param>
            /// <param name="height">The height.</param>
            /// <param name="title">The title.</param>
            /// <returns>This viewer, completed.</returns>
            public Task<IEnvViewer> Factory(int width, int height, string? title = null) {
                (Width, Height, Title, Closed) = (width, height, title, false);
                return Task.FromResult<IEnvViewer>(this);
            }

            /// <summary>Records the frame's size and pixels.</summary>
            /// <param name="img">The frame.</param>
            public void Render(SixLabors.ImageSharp.Image img) {
                using var rgb = img.CloneAs<SixLabors.ImageSharp.PixelFormats.Rgb24>();
                var pixels = new (byte, byte, byte)[rgb.Width * rgb.Height];
                for (int y = 0; y < rgb.Height; y++) {
                    for (int x = 0; x < rgb.Width; x++) {
                        var pixel = rgb[x, y];
                        pixels[y * rgb.Width + x] = (pixel.R, pixel.G, pixel.B);
                    }
                }

                Frames.Add(((rgb.Width, rgb.Height), pixels));
            }

            /// <summary>Marks the viewer closed.</summary>
            public void CloseEnvironment() => Closed = true;

            /// <summary>Marks input started.</summary>
            public void BeginInput() => InputStarted = true;

            /// <summary>Hands out the queued inputs.</summary>
            /// <returns>The inputs, oldest first.</returns>
            public IReadOnlyList<ViewerInput> TakeInput() {
                var taken = new List<ViewerInput>();
                while (Inputs.TryDequeue(out ViewerInput input))
                    taken.Add(input);
                return taken;
            }

            /// <summary>Does nothing: nothing is held.</summary>
            public void Dispose() { }
        }

        /// <summary>A figure that records <see cref="PlayPlot"/>'s calls as text, as the parity suite's stand-in for matplotlib does.</summary>
        private sealed class RecordingFigure : IPlotFigure {
            /// <summary>The calls, in order.</summary>
            public List<string> Log { get; } = new();

            /// <summary>The axes.</summary>
            private readonly List<IPlotAxes> _axes = new();

            /// <inheritdoc />
            public IReadOnlyList<IPlotAxes> Axes => _axes;

            /// <summary>Makes the figure's axes (<c>subplots(rows)</c>).</summary>
            /// <param name="rows">The number of axes.</param>
            /// <returns>This figure.</returns>
            public IPlotFigure Make(int rows) {
                for (int i = 0; i < rows; i++)
                    _axes.Add(new RecordingAxes(this, i));
                return this;
            }

            /// <summary>Records the pause.</summary>
            /// <param name="interval">The interval.</param>
            public void Pause(double interval) => Log.Add($"pause({interval.ToString(System.Globalization.CultureInfo.InvariantCulture)})");

            /// <summary>Formats numbers for the log: invariant, comma-separated.</summary>
            /// <param name="values">The numbers.</param>
            /// <returns>The text.</returns>
            private static string List(IEnumerable<double> values) => "[" + string.Join(",", values.Select(v => v.ToString(System.Globalization.CultureInfo.InvariantCulture))) + "]";

            /// <summary>A recording axes.</summary>
            /// <param name="Figure">The figure that logs.</param>
            /// <param name="Index">The axes' index.</param>
            private sealed record RecordingAxes(RecordingFigure Figure, int Index) : IPlotAxes {
                /// <summary>Records the title.</summary>
                /// <param name="title">The title.</param>
                public void SetTitle(string title) => Figure.Log.Add($"set_title({Index}, {title})");

                /// <summary>Records the scatter.</summary>
                /// <param name="x">The x values.</param>
                /// <param name="y">The y values.</param>
                /// <param name="color">The colour.</param>
                /// <returns>An artist that records its removal.</returns>
                public IPlotArtist Scatter(IReadOnlyList<double> x, IReadOnlyList<double> y, string color) {
                    Figure.Log.Add($"scatter({Index}, {List(x)}, {List(y)}, {color})");
                    return new Artist(Figure, Index);
                }

                /// <summary>Records the limits.</summary>
                /// <param name="left">The left limit.</param>
                /// <param name="right">The right limit.</param>
                public void SetXLim(double left, double right) => Figure.Log.Add($"set_xlim({Index}, {left}, {right})");
            }

            /// <summary>A recording scatter.</summary>
            /// <param name="Figure">The figure that logs.</param>
            /// <param name="Index">The axes' index.</param>
            private sealed record Artist(RecordingFigure Figure, int Index) : IPlotArtist {
                /// <summary>Records the removal.</summary>
                public void Remove() => Figure.Log.Add($"remove({Index})");
            }
        }
    }
}
