using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Gym.Rendering;
using Gym.Tests.Parity.Frames;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Pair;
using Gym.Tests.Parity.Session;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Python.Runtime;

namespace Gym.Tests.Parity.Tests.Rendering {
    /// <summary>
    ///     <c>Gym.Rendering</c> against pygame-ce 2.5.8 itself, primitive by primitive: the SDL_gfx port
    ///     (<c>gfxdraw.aapolygon</c>, <c>filled_polygon</c>, <c>aacircle</c>, <c>filled_circle</c>, <c>hline</c>, <c>vline</c>),
    ///     <c>transform.flip</c>, <c>Vector2.rotate_rad</c>, and pygame's argument conversion errors. Every frame must be
    ///     identical to the last pixel, on seeded random shapes that also exercise clipping, translucent colours and Sint16
    ///     wrap-around, so a CartPole frame that matches is not a coincidence of its few shapes.
    /// </summary>
    [TestClass]
    public sealed class GfxDrawLiveParityTests : ParityTestBase {
        /// <summary>How many random shapes each primitive is drawn with.</summary>
        private const int CasesPerPrimitive = 150;

        /// <summary>The small surface random shapes are drawn on: small, so most shapes cross an edge and get clipped.</summary>
        private const int Width = 64;

        /// <summary>The small surface's height.</summary>
        private const int Height = 48;

        /// <summary>Starts pygame's video system once per test, as Gymnasium's renderers do (gfxdraw needs it).</summary>
        [TestInitialize]
        public void InitializePygame() => PythonRuntime.Run(Python.pygame.display.init);

        /// <summary>Anti-aliased polygon outlines, 3 to 6 vertices anywhere around the surface, any colour and alpha.</summary>
        [TestMethod]
        public void AAPolygon_MatchesPygame() {
            var rng = new Random(20260925);
            AssertCases("aapolygon", Enumerable.Range(0, CasesPerPrimitive).Select(_ => PolygonCase(rng, aa: true)));
        }

        /// <summary>Filled polygons, convex or not, 3 to 6 vertices, any colour and alpha.</summary>
        [TestMethod]
        public void FilledPolygon_MatchesPygame() {
            var rng = new Random(20260926);
            AssertCases("filled_polygon", Enumerable.Range(0, CasesPerPrimitive).Select(_ => PolygonCase(rng, aa: false)));
        }

        /// <summary>
        ///     Polygons whose vertices sit near the Sint16 limits, where pygame's truncation wraps them to the other side and
        ///     the scanline fill's crossings are more than 32767 pixels apart (the qsort comparator overflows).
        /// </summary>
        [TestMethod]
        public void Polygons_WrappingAroundSint16_MatchPygame() {
            var rng = new Random(7);
            var cases = new List<DrawCase>();
            for (int i = 0; i < 40; i++) {
                // One vertex just inside a Sint16 limit and two just beyond it: pygame truncates the latter to the other
                // end of the range, so the polygon spans the whole Sint16 line, the visible surface included. This is
                // CartPole's cart when its x puts the cart's edges on both sides of 32767.
                bool positive = rng.Next(2) == 0;
                double limit = positive ? 32767.0 : -32768.0, outward = positive ? 1 : -1;
                var points = new[] {
                    (limit - outward * (1 + rng.NextDouble() * 60), rng.NextDouble() * (Height + 20) - 10),
                    (limit + outward * (1 + rng.NextDouble() * 60), rng.NextDouble() * (Height + 20) - 10),
                    (limit + outward * (1 + rng.NextDouble() * 60), rng.NextDouble() * (Height + 20) - 10),
                };
                cases.Add(Polygon(points, RandomColor(rng), aa: i % 2 == 0));
            }

            AssertCases("wrapped polygons", cases);
        }

        /// <summary>Anti-aliased circle outlines of radius 0 to 40, centred anywhere around the surface.</summary>
        [TestMethod]
        public void AACircle_MatchesPygame() {
            var rng = new Random(11);
            AssertCases("aacircle", Enumerable.Range(0, CasesPerPrimitive).Select(_ => CircleCase(rng, aa: true)));
        }

        /// <summary>Filled circles of radius 0 to 40, centred anywhere around the surface.</summary>
        [TestMethod]
        public void FilledCircle_MatchesPygame() {
            var rng = new Random(12);
            AssertCases("filled_circle", Enumerable.Range(0, CasesPerPrimitive).Select(_ => CircleCase(rng, aa: false)));
        }

        /// <summary>Horizontal and vertical lines with ends anywhere around the surface, in either order.</summary>
        [TestMethod]
        public void Lines_MatchPygame() {
            var rng = new Random(13);
            var cases = new List<DrawCase>();
            for (int i = 0; i < CasesPerPrimitive; i++) {
                long a = rng.Next(-20, Width + 20), b = rng.Next(-20, Width + 20), c = rng.Next(-20, Height + 20);
                (int R, int G, int B, int A) color = RandomColor(rng);
                if (i % 2 == 0) {
                    cases.Add(new DrawCase($"hline({a}, {b}, {c}, {color})",
                        surface => Python.pygame.gfxdraw.hline(surface, a, b, c, color),
                        surface => GfxDraw.HLine(surface, a, b, c, ToColor(color))));
                } else {
                    cases.Add(new DrawCase($"vline({c}, {a}, {b}, {color})",
                        surface => Python.pygame.gfxdraw.vline(surface, c, a, b, color),
                        surface => GfxDraw.VLine(surface, c, a, b, ToColor(color))));
                }
            }

            AssertCases("hline/vline", cases);
        }

        /// <summary>Ten random primitives composited on one surface: blending over earlier drawing, in order.</summary>
        [TestMethod]
        public void Compositions_MatchPygame() {
            var rng = new Random(14);
            var cases = new List<DrawCase>();
            for (int i = 0; i < 30; i++) {
                var parts = new List<DrawCase>();
                for (int j = 0; j < 10; j++) {
                    parts.Add((j % 4) switch {
                        0 => PolygonCase(rng, aa: true),
                        1 => PolygonCase(rng, aa: false),
                        2 => CircleCase(rng, aa: true),
                        _ => CircleCase(rng, aa: false),
                    });
                }

                cases.Add(new DrawCase($"composition {i}: {string.Join("; ", parts.Select(p => p.Description))}",
                    surface => parts.ForEach(p => p.Python(surface)),
                    surface => parts.ForEach(p => p.DotNet(surface))));
            }

            AssertCases("compositions", cases);
        }

        /// <summary>
        ///     CartPole's own shapes at 600 × 400: the cart, a tilted pole, the axle and the track, the calls
        ///     <c>cartpole.py</c> makes, before its flip.
        /// </summary>
        [TestMethod]
        public void CartPoleShapes_MatchPygame() {
            var cart = new[] { (275.0, 85.0), (275.0, 115.0), (325.0, 115.0), (325.0, 85.0) };
            var pole = new[] { (296.5, 101.2), (310.3, 219.9), (320.2, 218.7), (306.5, 100.1) };
            var shapes = new DrawCase("cart, pole, axle and track",
                surface => {
                    Python.pygame.gfxdraw.aapolygon(surface, ToMatrix(cart), (0, 0, 0, 255));
                    Python.pygame.gfxdraw.filled_polygon(surface, ToMatrix(cart), (0, 0, 0, 255));
                    Python.pygame.gfxdraw.aapolygon(surface, ToMatrix(pole), (202, 152, 101, 255));
                    Python.pygame.gfxdraw.filled_polygon(surface, ToMatrix(pole), (202, 152, 101, 255));
                    Python.pygame.gfxdraw.aacircle(surface, 300, 107, 5, (129, 132, 203, 255));
                    Python.pygame.gfxdraw.filled_circle(surface, 300, 107, 5, (129, 132, 203, 255));
                    Python.pygame.gfxdraw.hline(surface, 0, 600, 100, (0, 0, 0, 255));
                },
                surface => {
                    GfxDraw.AAPolygon(surface, cart, new Color(0, 0, 0));
                    GfxDraw.FilledPolygon(surface, cart, new Color(0, 0, 0));
                    GfxDraw.AAPolygon(surface, pole, new Color(202, 152, 101));
                    GfxDraw.FilledPolygon(surface, pole, new Color(202, 152, 101));
                    GfxDraw.AACircle(surface, 300, 107, 5, new Color(129, 132, 203));
                    GfxDraw.FilledCircle(surface, 300, 107, 5, new Color(129, 132, 203));
                    GfxDraw.HLine(surface, 0, 600, 100, new Color(0, 0, 0));
                });
            AssertCases("CartPole shapes", new[] { shapes }, 600, 400, (255, 255, 255));
        }

        /// <summary><c>transform.flip</c> in all four combinations, on a surface with an asymmetric drawing.</summary>
        [TestMethod]
        public void Flip_MatchesPygame() {
            var rng = new Random(15);
            DrawCase drawing = PolygonCase(rng, aa: false);
            foreach ((bool x, bool y) in new[] { (false, false), (true, false), (false, true), (true, true) }) {
                Frame python = PythonRuntime.Run(() => {
                    using PyObject surface = Python.pygame.Surface(Width, Height);
                    Python.pygame.surface.Surface.fill(surface, (10, 20, 30));
                    drawing.Python(surface);
                    using PyObject flipped = Python.pygame.transform.flip(surface, x, y);
                    return PythonFrame(flipped);
                });
                var dotnetSurface = new Surface(Width, Height);
                dotnetSurface.Fill(new Color(10, 20, 30));
                drawing.DotNet(dotnetSurface);
                Surface dotnetFlipped = Transform.Flip(dotnetSurface, x, y);
                FrameDiff.AssertSame($"flip(flip_x={x}, flip_y={y}) of {drawing.Description}", python, new Frame(Height, Width, dotnetFlipped.ToRgbBytes()), "pygame.transform.flip");
            }
        }

        /// <summary>
        ///     <c>Vector2(x, y).rotate_rad(angle)</c>, bit for bit: random angles, angles within and just outside 1e-6 of each
        ///     quarter turn (pygame snaps the former), negative and huge angles, signed zeros, NaN and infinities.
        /// </summary>
        [TestMethod]
        public void Vector2_RotateRad_MatchesPygame() {
            var rng = new Random(16);
            var angles = new List<double> { 0.0, -0.0, double.NaN, double.PositiveInfinity, double.NegativeInfinity, 1e300, -1e300, 2 * Math.PI, -2 * Math.PI, 7.5e-7, -7.5e-7, 1.5e-6, -1.5e-6 };
            // NaNs with payloads and both signs, a signalling one included: the C runtime's fmod hands them through, so the
            // rotated coordinates carry them (the .NET 8 runtime's own % would not).
            angles.AddRange(new[] { 0x7FF8000000000000L, unchecked((long)0xFFF8000000000123UL), 0x7FF8000000000123L, 0x7FF0000000000123L }.Select(BitConverter.Int64BitsToDouble));
            for (int k = -8; k <= 8; k++) {
                double quarter = k * Math.PI / 2;
                angles.AddRange(new[] { quarter, quarter + 5e-7, quarter - 5e-7, quarter + 9.99e-7, quarter - 9.99e-7, quarter + 1.01e-6, quarter - 1.01e-6, Math.BitIncrement(quarter), Math.BitDecrement(quarter) });
            }

            for (int i = 0; i < 200; i++)
                angles.Add((rng.NextDouble() - 0.5) * 20);

            var vectors = new List<(double, double)> { (-5.0, -5.0), (-5.0, 120.0), (5.0, 120.0), (5.0, -5.0), (0.0, 0.0), (-0.0, 1.0), (1e-300, -1e300) };
            for (int i = 0; i < 5; i++)
                vectors.Add(((rng.NextDouble() - 0.5) * 200, (rng.NextDouble() - 0.5) * 200));

            foreach ((double x, double y) in vectors) {
                foreach (double angle in angles) {
                    string what = $"Vector2({R(x)}, {R(y)}).rotate_rad({R(angle)})";
                    (Outcome px, Outcome py) = PythonRuntime.Run(() => {
                        using PyObject vector = Python.pygame.Vector2(x, y);
                        using PyObject rotated = Python.pygame.math.Vector2.rotate_rad(vector, angle);
                        return (Outcome.CapturePython(() => Python.pygame.math.Vector2.x(rotated)), Outcome.CapturePython(() => Python.pygame.math.Vector2.y(rotated)));
                    });
                    Vector2 dotnet = new Vector2(x, y).RotateRad(angle);
                    ByteContract.AssertSame($"{what}.x", px, Outcome.Returned(ValueFacts.FromDouble(dotnet.X)));
                    ByteContract.AssertSame($"{what}.y", py, Outcome.Returned(ValueFacts.FromDouble(dotnet.Y)));
                }
            }
        }

        /// <summary>
        ///     pygame's argument errors: circle and line arguments beyond Sint16 or a C long, polygon coordinates that are
        ///     NaN, infinite or beyond a C long, and too few points, each with pygame's exception type (mapped) and message.
        /// </summary>
        [TestMethod]
        public void ArgumentErrors_MatchPygame() {
            (int, int, int, int) color = (1, 2, 3, 255);
            var cases = new List<(string What, Action<PyObject> Python, Action<Surface> DotNet)> {
                ("aacircle(40000, 5, 5)", s => Python.pygame.gfxdraw.aacircle(s, 40000, 5, 5, color), s => GfxDraw.AACircle(s, 40000, 5, 5, new Color(1, 2, 3))),
                ("aacircle(5, -40000, 5)", s => Python.pygame.gfxdraw.aacircle(s, 5, -40000, 5, color), s => GfxDraw.AACircle(s, 5, -40000, 5, new Color(1, 2, 3))),
                ("aacircle(2**40, 5, 5)", s => Python.pygame.gfxdraw.aacircle(s, 1L << 40, 5, 5, color), s => GfxDraw.AACircle(s, 1L << 40, 5, 5, new Color(1, 2, 3))),
                ("filled_circle(5, 5, 32768)", s => Python.pygame.gfxdraw.filled_circle(s, 5, 5, 32768, color), s => GfxDraw.FilledCircle(s, 5, 5, 32768, new Color(1, 2, 3))),
                ("hline(0, 70000, 5)", s => Python.pygame.gfxdraw.hline(s, 0, 70000, 5, color), s => GfxDraw.HLine(s, 0, 70000, 5, new Color(1, 2, 3))),
                ("vline(-2**33, 0, 5)", s => Python.pygame.gfxdraw.vline(s, -(1L << 33), 0, 5, color), s => GfxDraw.VLine(s, -(1L << 33), 0, 5, new Color(1, 2, 3))),
                ("aapolygon with a NaN", s => Python.pygame.gfxdraw.aapolygon(s, ToMatrix(new[] { (double.NaN, 0.0), (1.0, 1.0), (2.0, 3.0) }), color),
                    s => GfxDraw.AAPolygon(s, new[] { (double.NaN, 0.0), (1.0, 1.0), (2.0, 3.0) }, new Color(1, 2, 3))),
                ("filled_polygon with an infinity", s => Python.pygame.gfxdraw.filled_polygon(s, ToMatrix(new[] { (0.0, 0.0), (1.0, double.NegativeInfinity), (2.0, 3.0) }), color),
                    s => GfxDraw.FilledPolygon(s, new[] { (0.0, 0.0), (1.0, double.NegativeInfinity), (2.0, 3.0) }, new Color(1, 2, 3))),
                ("aapolygon with 2**40", s => Python.pygame.gfxdraw.aapolygon(s, ToMatrix(new[] { (0.0, 0.0), (Math.Pow(2, 40), 1.0), (2.0, 3.0) }), color),
                    s => GfxDraw.AAPolygon(s, new[] { (0.0, 0.0), (Math.Pow(2, 40), 1.0), (2.0, 3.0) }, new Color(1, 2, 3))),
                ("filled_polygon with 2 points", s => Python.pygame.gfxdraw.filled_polygon(s, ToMatrix(new[] { (0.0, 0.0), (1.0, 1.0) }), color),
                    s => GfxDraw.FilledPolygon(s, new[] { (0.0, 0.0), (1.0, 1.0) }, new Color(1, 2, 3))),
            };

            foreach ((string what, Action<PyObject> python, Action<Surface> dotnet) in cases) {
                Outcome pythonOutcome = Outcome.CapturePython(() => {
                    using PyObject surface = Python.pygame.Surface(Width, Height);
                    python(surface);
                    return PyConvert.ToPython(null);
                });
                Outcome dotnetOutcome = Outcome.Capture(() => {
                    dotnet(new Surface(Width, Height));
                    return null;
                });
                ByteContract.AssertSame($"gfxdraw.{what}", pythonOutcome, dotnetOutcome);
            }
        }

        /// <summary>One drawing, spelled for each side, and how reports name it.</summary>
        /// <param name="Description">The call in pygame's spelling.</param>
        /// <param name="Python">Draws on a pygame surface; runs under the GIL.</param>
        /// <param name="DotNet">Draws on a Gym.Rendering surface.</param>
        private sealed record DrawCase(string Description, Action<PyObject> Python, Action<Surface> DotNet);

        /// <summary>
        ///     Draws each case on a fresh surface on both sides, over a background of its own, and requires identical frames
        ///     (artifacts are written for the first that differs).
        /// </summary>
        /// <param name="family">The primitive family, for reports.</param>
        /// <param name="cases">The drawings.</param>
        /// <param name="width">The surface width.</param>
        /// <param name="height">The surface height.</param>
        /// <param name="background">The fill colour, or <see langword="null"/> for one derived from the case index.</param>
        /// <exception cref="ParityException">Thrown at the first case whose frames differ.</exception>
        /// <exception cref="AssertFailedException">
        ///     Thrown when fewer than half of the cases changed a pixel on pygame's side: identical untouched backgrounds
        ///     would make the comparison vacuous (pygame drawing nothing, for example without its video system).
        /// </exception>
        private static void AssertCases(string family, IEnumerable<DrawCase> cases, int width = Width, int height = Height, (int R, int G, int B)? background = null) {
            int index = 0, drawn = 0;
            foreach (DrawCase drawing in cases) {
                // A varied background makes translucent blends and anti-aliased weights visible in every channel.
                (int R, int G, int B) fill = background ?? ((index * 37) % 256, (index * 91 + 50) % 256, (index * 53 + 200) % 256);
                Frame python = PythonRuntime.Run(() => {
                    using PyObject surface = Python.pygame.Surface(width, height);
                    Python.pygame.surface.Surface.fill(surface, fill);
                    drawing.Python(surface);
                    return PythonFrame(surface);
                });
                var dotnetSurface = new Surface(width, height);
                dotnetSurface.Fill(new Color((byte)fill.R, (byte)fill.G, (byte)fill.B));
                drawing.DotNet(dotnetSurface);
                FrameDiff.AssertSame($"{family} #{index}: {drawing.Description} on {fill}", python, new Frame(height, width, dotnetSurface.ToRgbBytes()), "pygame.gfxdraw primitive");
                if (ChangesBackground(python, fill))
                    drawn++;
                index++;
            }

            Assert.IsTrue(drawn * 2 >= index, $"{family}: only {drawn} of {index} cases drew anything on pygame's surface, so the comparison proves little.");
        }

        /// <summary>Whether any pixel of a frame differs from the fill colour, i.e. whether the drawing left a mark.</summary>
        /// <param name="frame">The frame.</param>
        /// <param name="fill">The background it was drawn on.</param>
        /// <returns><see langword="true"/> when at least one pixel changed.</returns>
        private static bool ChangesBackground(Frame frame, (int R, int G, int B) fill) {
            for (int i = 0; i < frame.Rgb.Length; i += 3) {
                if (frame.Rgb[i] != fill.R || frame.Rgb[i + 1] != fill.G || frame.Rgb[i + 2] != fill.B)
                    return true;
            }

            return false;
        }

        /// <summary>A random polygon of 3 to 6 vertices within 20 pixels of the small surface, with a random colour.</summary>
        /// <param name="rng">The case generator.</param>
        /// <param name="aa">Whether it is an <c>aapolygon</c> (otherwise a <c>filled_polygon</c>).</param>
        /// <returns>The case.</returns>
        private static DrawCase PolygonCase(Random rng, bool aa) {
            int count = rng.Next(3, 7);
            var points = new (double X, double Y)[count];
            for (int i = 0; i < count; i++)
                points[i] = (Coordinate(rng, Width), Coordinate(rng, Height));
            return Polygon(points, RandomColor(rng), aa);
        }

        /// <summary>A polygon drawing on both sides.</summary>
        /// <param name="points">The vertices.</param>
        /// <param name="color">The colour.</param>
        /// <param name="aa">Whether it is an <c>aapolygon</c>.</param>
        /// <returns>The case.</returns>
        private static DrawCase Polygon((double X, double Y)[] points, (int R, int G, int B, int A) color, bool aa) {
            string text = $"{(aa ? "aapolygon" : "filled_polygon")}([{string.Join(", ", points.Select(p => $"({R(p.X)}, {R(p.Y)})"))}], {color})";
            return aa
                ? new DrawCase(text, surface => Python.pygame.gfxdraw.aapolygon(surface, ToMatrix(points), color), surface => GfxDraw.AAPolygon(surface, points, ToColor(color)))
                : new DrawCase(text, surface => Python.pygame.gfxdraw.filled_polygon(surface, ToMatrix(points), color), surface => GfxDraw.FilledPolygon(surface, points, ToColor(color)));
        }

        /// <summary>A random circle, centre within 20 pixels of the small surface, radius 0 to 40, with a random colour.</summary>
        /// <param name="rng">The case generator.</param>
        /// <param name="aa">Whether it is an <c>aacircle</c> (otherwise a <c>filled_circle</c>).</param>
        /// <returns>The case.</returns>
        private static DrawCase CircleCase(Random rng, bool aa) {
            long x = rng.Next(-20, Width + 20), y = rng.Next(-20, Height + 20), r = rng.Next(0, 41);
            (int R, int G, int B, int A) color = RandomColor(rng);
            return aa
                ? new DrawCase($"aacircle({x}, {y}, {r}, {color})", surface => Python.pygame.gfxdraw.aacircle(surface, x, y, r, color), surface => GfxDraw.AACircle(surface, x, y, r, ToColor(color)))
                : new DrawCase($"filled_circle({x}, {y}, {r}, {color})", surface => Python.pygame.gfxdraw.filled_circle(surface, x, y, r, color), surface => GfxDraw.FilledCircle(surface, x, y, r, ToColor(color)));
        }

        /// <summary>
        ///     A coordinate within 20 pixels of <c>[0, extent]</c>: usually fractional (pygame truncates toward zero, so
        ///     negative fractions matter), sometimes a whole number or an exact half.
        /// </summary>
        /// <param name="rng">The case generator.</param>
        /// <param name="extent">The surface's size along the axis.</param>
        /// <returns>The coordinate.</returns>
        private static double Coordinate(Random rng, int extent) {
            double value = rng.NextDouble() * (extent + 40) - 20;
            return rng.Next(6) switch {
                0 => Math.Round(value),
                1 => Math.Floor(value) + 0.5,
                _ => value,
            };
        }

        /// <summary>A random colour: opaque half of the time (as Gymnasium draws), otherwise any alpha.</summary>
        /// <param name="rng">The case generator.</param>
        /// <returns>The colour as <c>(r, g, b, a)</c>.</returns>
        private static (int R, int G, int B, int A) RandomColor(Random rng) {
            return (rng.Next(256), rng.Next(256), rng.Next(256), rng.Next(2) == 0 ? 255 : rng.Next(256));
        }

        /// <summary>The colour as <see cref="Color"/>.</summary>
        /// <param name="color">The colour as a tuple.</param>
        /// <returns>The colour.</returns>
        private static Color ToColor((int R, int G, int B, int A) color) => new((byte)color.R, (byte)color.G, (byte)color.B, (byte)color.A);

        /// <summary>The vertices as the <c>(n, 2)</c> matrix the facade crosses to NumPy.</summary>
        /// <param name="points">The vertices.</param>
        /// <returns>The matrix.</returns>
        private static double[,] ToMatrix(IReadOnlyList<(double X, double Y)> points) {
            var matrix = new double[points.Count, 2];
            for (int i = 0; i < points.Count; i++) {
                matrix[i, 0] = points[i].X;
                matrix[i, 1] = points[i].Y;
            }

            return matrix;
        }

        /// <summary>
        ///     A pygame surface as a <see cref="Frame"/>: <c>surfarray.array3d</c> (<c>[x][y][c]</c>) reordered to rows. Must
        ///     run under the GIL.
        /// </summary>
        /// <param name="surface">The pygame surface; borrowed.</param>
        /// <returns>The frame.</returns>
        private static Frame PythonFrame(PyObject surface) {
            using PyObject columns = Python.pygame.surfarray.array3d(surface);
            ValueFacts facts = ValueFacts.FromPython(columns);
            int width = checked((int)facts.Shape[0]), height = checked((int)facts.Shape[1]);
            var rgb = new byte[width * height * 3];
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    int from = (x * height + y) * 3, to = (y * width + x) * 3;
                    rgb[to] = facts.Bytes[from];
                    rgb[to + 1] = facts.Bytes[from + 1];
                    rgb[to + 2] = facts.Bytes[from + 2];
                }
            }

            return new Frame(height, width, rgb);
        }

        /// <summary>Formats a double the way reports print numbers (shortest round-trip text).</summary>
        /// <param name="value">The value.</param>
        /// <returns>The text.</returns>
        private static string R(double value) => value.ToString("R", CultureInfo.InvariantCulture);
    }
}
