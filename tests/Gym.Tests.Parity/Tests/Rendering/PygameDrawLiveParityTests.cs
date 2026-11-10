using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Gym.Rendering;
using Gym.Tests.Parity.Frames;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Pair;
using Gym.Tests.Parity.Session;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace Gym.Tests.Parity.Tests.Rendering {
    /// <summary>
    ///     The rasterizer of P7 against pygame-ce 2.5.8 itself: <c>draw.line</c>, <c>draw.rect</c>, <c>image.load</c> of every
    ///     asset the envs load, <c>image.frombytes</c>, blitting per-pixel alpha onto a frame, <c>transform.scale</c>,
    ///     shrinking <c>transform.smoothscale</c>, <c>transform.flip</c> with alpha, and the anti-aliased lines. Gym.NET's
    ///     implementations are its own (decision D4), derived from pygame's observable behaviour; these random corpora are
    ///     what proves them, every pixel and every alpha byte.
    /// </summary>
    /// <remarks>
    ///     <c>draw.aaline</c> and <c>draw.aalines</c> are the exception to D4 (decision DR-37): D4's black-box model of them did
    ///     not converge, so they are translated from pygame-ce's <c>src_c/draw.c</c> (the class remarks of <see cref="Draw"/>),
    ///     and their cells here hold the translation to pygame's pixels, its quirks included.
    /// </remarks>
    [TestClass]
    public sealed class PygameDrawLiveParityTests : ParityTestBase {
        /// <summary>The small surface random shapes are drawn on: small, so most shapes cross an edge and get clipped.</summary>
        private const int Width = 64;

        /// <summary>The small surface's height.</summary>
        private const int Height = 48;

        /// <summary>Starts pygame's video system once per test, as Gymnasium's renderers do.</summary>
        [TestInitialize]
        public void InitializePygame() => PythonRuntime.Run(Python.pygame.display.init);

        /// <summary>
        ///     <c>draw.line</c> at width 1: 400 lines with integer endpoints around the surface (some to ±5,000 away), 200 with
        ///     float endpoints (truncated), in both directions, any colour and alpha.
        /// </summary>
        [TestMethod]
        public void Line_MatchesPygame() {
            var rng = new Random(20261109);
            var cases = new List<DrawCase>();
            for (int i = 0; i < 400; i++) {
                int reach = i % 10 == 0 ? 5000 : 20;
                long x0 = rng.Next(-reach, Width + reach), y0 = rng.Next(-reach, Height + reach);
                long x1 = i % 7 == 0 ? x0 + rng.Next(-3, 4) : rng.Next(-reach, Width + reach);
                long y1 = i % 7 == 0 ? y0 + rng.Next(-3, 4) : rng.Next(-reach, Height + reach);
                (int R, int G, int B, int A) color = RandomColor(rng);
                cases.Add(new DrawCase($"line({color}, ({x0}, {y0}), ({x1}, {y1}))",
                    surface => Python.pygame.draw.line(surface, color, (x0, y0), (x1, y1)),
                    surface => Draw.Line(surface, ToColor(color), x0, y0, x1, y1)));
            }

            for (int i = 0; i < 200; i++) {
                double x0 = rng.NextDouble() * (Width + 40) - 20, y0 = rng.NextDouble() * (Height + 40) - 20;
                double x1 = rng.NextDouble() * (Width + 40) - 20, y1 = rng.NextDouble() * (Height + 40) - 20;
                (int R, int G, int B, int A) color = RandomColor(rng);
                cases.Add(new DrawCase($"line({color}, ({x0:R}, {y0:R}), ({x1:R}, {y1:R}))",
                    surface => Python.pygame.draw.line(surface, color, (x0, y0), (x1, y1)),
                    surface => Draw.Line(surface, ToColor(color), x0, y0, x1, y1)));
            }

            AssertCases("draw.line", cases);
        }

        /// <summary>
        ///     <c>draw.rect</c> filled (width 0), outlined (width 1) and not drawn (width -1): 600 rectangles of every size from
        ///     empty to wider than the surface, around and across its edges, so the clipped-sliver rule is exercised.
        /// </summary>
        [TestMethod]
        public void Rect_MatchesPygame() {
            var rng = new Random(20261110);
            var cases = new List<DrawCase>();
            for (int i = 0; i < 600; i++) {
                long x = rng.Next(-30, Width + 30), y = rng.Next(-30, Height + 30);
                long w = i % 5 == 0 ? rng.Next(-1, 5) : rng.Next(-3, 70), h = i % 5 == 0 ? rng.Next(-1, 5) : rng.Next(-3, 70);
                int lineWidth = (i % 3) switch { 0 => 0, 1 => 1, _ => -1 };
                (int R, int G, int B, int A) color = RandomColor(rng);
                cases.Add(new DrawCase($"rect({color}, ({x}, {y}, {w}, {h}), {lineWidth})",
                    surface => Python.pygame.draw.rect(surface, color, (x, y, w, h), lineWidth),
                    surface => Draw.Rect(surface, ToColor(color), x, y, w, h, lineWidth)));
            }

            // A third of the cases are width -1, which never draws, and many others lie off the surface or have no area; the
            // ones that do draw are about a quarter of the corpus (137 of 600), which still leaves every rule exercised.
            AssertCases("draw.rect", cases, minimumDrawnShare: 0.2);
        }

        /// <summary>
        ///     <c>image.load</c> of every PNG Gymnasium's envs load (Pendulum's arrow, the toy_text sprites, tiles and labels):
        ///     every pixel's RGB and alpha, and whether the surface has per-pixel alpha.
        /// </summary>
        [TestMethod]
        public void ImageLoad_EveryEnvAsset_MatchesPygame() {
            string envs = GymnasiumEnvs;
            string[] files = Directory.GetFiles(Path.Combine(envs, "classic_control"), "*.png", SearchOption.AllDirectories)
                .Concat(Directory.GetFiles(Path.Combine(envs, "toy_text"), "*.png", SearchOption.AllDirectories))
                .OrderBy(path => path, StringComparer.Ordinal)
                .ToArray();
            Assert.IsTrue(files.Length >= 90, $"Found only {files.Length} PNG assets under {envs}.");
            foreach (string file in files) {
                string label = $"image.load('{Path.GetRelativePath(envs, file)}')";
                (byte[] rgba, bool alpha, int width, int height) = PythonRuntime.Run(() => {
                    using PyObject surface = Python.pygame.image.load(file);
                    return PythonRgba(surface);
                });
                Surface dotnet = ImageIO.Load(file);
                Assert.AreEqual((width, height, alpha), (dotnet.Width, dotnet.Height, dotnet.HasAlpha), $"{label}: size and per-pixel alpha");
                AssertSameRgba(label, rgba, dotnet, width, height);
            }
        }

        /// <summary><c>image.frombytes</c> with <c>RGBA</c> and <c>RGB</c> bytes, and its errors' types and messages.</summary>
        [TestMethod]
        public void ImageFromBytes_MatchesPygame() {
            var rng = new Random(5);
            foreach ((int w, int h, string format) in new[] { (7, 5, "RGBA"), (3, 9, "RGB"), (1, 1, "RGBA") }) {
                int perPixel = format == "RGBA" ? 4 : 3;
                byte[] bytes = RandomBytes(rng, w * h * perPixel);
                (byte[] rgba, bool alpha, int width, int height) = PythonRuntime.Run(() => {
                    using PyObject surface = FromBytesPython(bytes, w, h, format);
                    return PythonRgba(surface);
                });
                Surface dotnet = ImageIO.FromBytes(bytes, w, h, format);
                Assert.AreEqual((width, height, alpha), (dotnet.Width, dotnet.Height, dotnet.HasAlpha), $"frombytes({format}, {w}x{h})");
                AssertSameRgba($"frombytes({format}, {w}x{h})", rgba, dotnet, w, h);
            }

            foreach ((int length, int w, int h, string format) in new[] { (15, 2, 2, "RGBA"), (16, -1, 2, "RGBA"), (16, 2, 2, "XYZ") }) {
                byte[] bytes = new byte[length];
                ByteContract.AssertSame($"frombytes({length} bytes, ({w}, {h}), '{format}')",
                    Outcome.CapturePython(() => FromBytesPython(bytes, w, h, format)),
                    Outcome.Capture(() => ImageIO.FromBytes(bytes, w, h, format).Width));
            }
        }

        /// <summary>
        ///     <c>surface.blit</c> of a surface with per-pixel alpha (as image.load gives) onto a frame: 150 random sources of
        ///     every alpha (fully transparent, opaque and in between) at positions on, across and off the frame's edges.
        /// </summary>
        [TestMethod]
        public void Blit_PerPixelAlphaOntoFrame_MatchesPygame() {
            var rng = new Random(20261111);
            for (int i = 0; i < 150; i++) {
                int sw = rng.Next(1, 40), sh = rng.Next(1, 40);
                byte[] bytes = RandomRgba(rng, sw, sh);
                int x = rng.Next(-sw - 5, Width + 5), y = rng.Next(-sh - 5, Height + 5);
                (int R, int G, int B) fill = (rng.Next(256), rng.Next(256), rng.Next(256));
                Frame python = PythonRuntime.Run(() => {
                    using PyObject frame = Python.pygame.Surface(Width, Height);
                    Python.pygame.surface.Surface.fill(frame, fill);
                    using PyObject source = FromBytesPython(bytes, sw, sh, "RGBA");
                    Python.pygame.surface.Surface.blit(frame, source, x, y);
                    return PythonFrame(frame);
                });
                var dotnet = new Surface(Width, Height);
                dotnet.Fill(new Color((byte)fill.R, (byte)fill.G, (byte)fill.B));
                dotnet.Blit(ImageIO.FromBytes(bytes, sw, sh, "RGBA"), x, y);
                FrameDiff.AssertSame($"blit #{i}: {sw}x{sh} RGBA at ({x}, {y}) onto {fill}", python, new Frame(Height, Width, dotnet.ToRgbBytes()), "alpha blit");
            }
        }

        /// <summary>
        ///     <c>transform.scale</c> of random RGBA images, shrinking and growing each axis (sizes 1 to 120), and of the
        ///     toy_text tiles and sprites at the sizes FrozenLake and CliffWalking ask for: every pixel and alpha byte.
        /// </summary>
        [TestMethod]
        public void Scale_MatchesPygame() {
            var rng = new Random(20261112);
            var sizes = new List<(int SW, int SH, int DW, int DH)> { (64, 64, 128, 128), (64, 64, 64, 64), (40, 36, 80, 72), (24, 32, 48, 64), (12, 20, 24, 40), (1, 1, 7, 3), (9, 4, 1, 1) };
            for (int i = 0; i < 150; i++)
                sizes.Add((rng.Next(1, 121), rng.Next(1, 121), rng.Next(0, 121), rng.Next(0, 121)));
            foreach ((int sw, int sh, int dw, int dh) in sizes) {
                byte[] bytes = RandomRgba(rng, sw, sh);
                (byte[] rgba, bool _, int width, int height) = PythonRuntime.Run(() => {
                    using PyObject source = FromBytesPython(bytes, sw, sh, "RGBA");
                    using PyObject scaled = Python.pygame.transform.scale(source, dw, dh);
                    return PythonRgba(scaled);
                });
                Surface dotnet = Transform.Scale(ImageIO.FromBytes(bytes, sw, sh, "RGBA"), dw, dh);
                Assert.AreEqual((width, height), (dotnet.Width, dotnet.Height), $"scale {sw}x{sh} -> {dw}x{dh}: size");
                AssertSameRgba($"scale {sw}x{sh} -> {dw}x{dh}", rgba, dotnet, width, height);
            }

            // The negative size error, Python's ValueError with pygame's message.
            ByteContract.AssertSame("scale to (-1, 3)", Outcome.CapturePython(() => {
                using PyObject surface = Python.pygame.Surface(4, 4);
                return Python.pygame.transform.scale(surface, -1, 3);
            }), Outcome.Capture(() => Transform.Scale(new Surface(4, 4), -1, 3).Width));
        }

        /// <summary>
        ///     <c>transform.smoothscale</c>, shrinking: 150 random RGBA images shrunk along one axis or both, and Pendulum's
        ///     arrow at the sizes its torque gives (every 9th pixel size from 0 to 227): every pixel and alpha byte. Growing,
        ///     which no env in this release needs, is refused rather than approximated.
        /// </summary>
        [TestMethod]
        public void Smoothscale_Shrinking_MatchesPygame() {
            var rng = new Random(20261113);
            var cases = new List<(byte[]? Bytes, string? File, int SW, int SH, int DW, int DH)>();
            for (int i = 0; i < 150; i++) {
                int sw = rng.Next(1, 160), sh = rng.Next(1, 160);
                int dw = i % 3 == 2 ? sw : rng.Next(0, sw + 1), dh = i % 3 == 1 ? sh : rng.Next(0, sh + 1);
                cases.Add((RandomRgba(rng, sw, sh), null, sw, sh, dw, dh));
            }

            string arrow = Path.Combine(GymnasiumEnvs, "classic_control", "assets", "clockwise.png");
            for (int size = 0; size <= 227; size += 9)
                cases.Add((null, arrow, 329, 312, size, size));

            foreach ((byte[]? bytes, string? file, int sw, int sh, int dw, int dh) in cases) {
                string label = $"smoothscale {(file is null ? "random" : "clockwise.png")} {sw}x{sh} -> {dw}x{dh}";
                (byte[] rgba, bool _, int width, int height) = PythonRuntime.Run(() => {
                    using PyObject source = file is null ? FromBytesPython(bytes!, sw, sh, "RGBA") : Python.pygame.image.load(file);
                    using PyObject scaled = Python.pygame.transform.smoothscale(source, dw, dh);
                    return PythonRgba(scaled);
                });
                Surface source = file is null ? ImageIO.FromBytes(bytes!, sw, sh, "RGBA") : ImageIO.Load(file);
                Surface dotnet = Transform.Smoothscale(source, dw, dh);
                Assert.AreEqual((width, height), (dotnet.Width, dotnet.Height), $"{label}: size");
                AssertSameRgba(label, rgba, dotnet, width, height);
            }

            Assert.ThrowsExactly<NotSupportedException>(() => Transform.Smoothscale(new Surface(4, 4, hasAlpha: true), 5, 4));
        }

        /// <summary><c>transform.flip</c> of random RGBA images, all four ways: every pixel and alpha byte, per-pixel alpha kept.</summary>
        [TestMethod]
        public void Flip_PerPixelAlpha_MatchesPygame() {
            var rng = new Random(20261114);
            for (int i = 0; i < 20; i++) {
                int w = rng.Next(1, 30), h = rng.Next(1, 30);
                byte[] bytes = RandomRgba(rng, w, h);
                foreach ((bool fx, bool fy) in new[] { (false, false), (true, false), (false, true), (true, true) }) {
                    (byte[] rgba, bool alpha, int width, int height) = PythonRuntime.Run(() => {
                        using PyObject source = FromBytesPython(bytes, w, h, "RGBA");
                        using PyObject flipped = Python.pygame.transform.flip(source, fx, fy);
                        return PythonRgba(flipped);
                    });
                    Surface dotnet = Transform.Flip(ImageIO.FromBytes(bytes, w, h, "RGBA"), fx, fy);
                    Assert.AreEqual(alpha, dotnet.HasAlpha, "flip keeps per-pixel alpha");
                    AssertSameRgba($"flip {w}x{h} ({fx}, {fy})", rgba, dotnet, width, height);
                }
            }
        }

        /// <summary>
        ///     <c>draw.aaline</c> on 300 random segments, any colour, over varied backgrounds, most of them clipped by the
        ///     surface's edges.
        /// </summary>
        [TestMethod]
        public void Aaline_MatchesPygame() {
            var rng = new Random(20261115);
            var cases = new List<DrawCase>();
            for (int i = 0; i < 300; i++) {
                double x0 = rng.NextDouble() * (Width + 40) - 20, y0 = rng.NextDouble() * (Height + 40) - 20;
                double x1 = rng.NextDouble() * (Width + 40) - 20, y1 = rng.NextDouble() * (Height + 40) - 20;
                (int R, int G, int B, int A) color = RandomColor(rng);
                cases.Add(new DrawCase($"aaline({color}, ({x0:R}, {y0:R}), ({x1:R}, {y1:R}))",
                    surface => Python.pygame.draw.aaline(surface, color, (x0, y0), (x1, y1)),
                    surface => Draw.Aaline(surface, ToColor(color), x0, y0, x1, y1)));
            }

            AssertCases("draw.aaline", cases);
        }

        /// <summary>
        ///     <c>draw.aalines</c> on 150 random polylines of 2 to 6 points, open and closed: the endpoints pygame leaves out
        ///     where a segment ends off a whole pixel, the extra pixel where steepness changes, and the 23 two-point polylines,
        ///     whose first test reads the float past pygame's buffer. Where that float decides (a steep segment ending off a
        ///     whole pixel), the interpreter may hold anything there, and pygame's drawing changes from run to run (fresh
        ///     interpreters included), so either of its two outcomes passes and the report counts the cases that took the one
        ///     Gym.NET does not draw (0, Gym.NET's value for that float, is <c>Draw.Aalines</c>' convention).
        /// </summary>
        [TestMethod]
        public void Aalines_MatchesPygame() {
            var cases = new List<DrawCase>();
            foreach (((int R, int G, int B, int A) color, bool closed, (double X, double Y)[] points) in AalinesCorpus())
                cases.Add(AalinesCase(color, closed, points));

            int alternatives = AssertCases("draw.aalines", cases);
            TestContext.WriteLine($"draw.aalines: {cases.Count} polylines identical, {alternatives} of them as pygame's other outcome of its read past the buffer.");
        }

        /// <summary>
        ///     Segments and polylines whose start lies in (-1, 0) on their major axis, both ways round: pygame truncates the
        ///     start to pixel 0, so the first endpoint's coverage exceeds 1 and the blend leaves 0..255, and pygame's cast keeps
        ///     the low byte of the truncated int (200 segments, 100 open polylines of 2 to 4 points, random colours over the
        ///     varied backgrounds).
        /// </summary>
        [TestMethod]
        public void Aaline_StartLeftOfPixelZero_WrapsAsPygame() {
            var rng = new Random(20261110);
            var segments = new List<DrawCase>();
            for (int i = 0; i < 200; i++) {
                // The major axis is x for even cases and y for odd ones; the start is in (-1, 0] on it, and the minor extent
                // stays below the major one so the segment keeps that axis.
                bool steep = i % 2 == 1;
                double start = -rng.NextDouble(), end = rng.NextDouble() * 50 + 5;
                double from = rng.NextDouble() * 40 + 10, to = from + (rng.NextDouble() * 2 - 1) * (end - start) * 0.9;
                (double x0, double y0, double x1, double y1) = steep ? (from, start, to, end) : (start, from, end, to);
                if (i % 4 >= 2)
                    (x0, y0, x1, y1) = (x1, y1, x0, y0);
                (int R, int G, int B, int A) color = RandomColor(rng);
                segments.Add(new DrawCase($"aaline({color}, ({x0:R}, {y0:R}), ({x1:R}, {y1:R}))",
                    surface => Python.pygame.draw.aaline(surface, color, (x0, y0), (x1, y1)),
                    surface => Draw.Aaline(surface, ToColor(color), x0, y0, x1, y1)));
            }

            AssertCases("draw.aaline", segments);

            var polylines = new List<DrawCase>();
            for (int i = 0; i < 100; i++) {
                var points = new List<(double X, double Y)> { (-rng.NextDouble(), rng.NextDouble() * 40 + 10) };
                int count = rng.Next(2, 5);
                while (points.Count < count)
                    points.Add((rng.NextDouble() * (Width - 4) + 2, rng.NextDouble() * (Height - 4) + 2));
                (int R, int G, int B, int A) color = RandomColor(rng);
                polylines.Add(AalinesCase(color, false, points.ToArray()));
            }

            AssertCases("draw.aalines", polylines);
        }

        /// <summary>
        ///     Segments shorter than pygame's 0.0001 in both directions, and of no length at all: one pixel, the start rounded
        ///     half up (<c>floor(v + 0.5)</c> in double), set to the colour at full strength, nothing when it is off the surface
        ///     (60 random starts across and around the surface, half of them on half-pixel coordinates).
        /// </summary>
        [TestMethod]
        public void Aaline_ZeroLength_SetsTheRoundedPixelAsPygame() {
            var rng = new Random(20261111);
            var cases = new List<DrawCase>();
            for (int i = 0; i < 60; i++) {
                double x0 = rng.NextDouble() * (Width + 4) - 2, y0 = rng.NextDouble() * (Height + 4) - 2;
                if (i % 2 == 1)
                    (x0, y0) = (Math.Floor(x0) + 0.5, Math.Floor(y0) + 0.5);
                double x1 = x0 + (i % 3 == 0 ? 0.0 : (rng.NextDouble() * 2 - 1) * 0.00009);
                double y1 = y0 + (i % 3 == 0 ? 0.0 : (rng.NextDouble() * 2 - 1) * 0.00009);
                (int R, int G, int B, int A) color = RandomColor(rng);
                cases.Add(new DrawCase($"aaline({color}, ({x0:R}, {y0:R}), ({x1:R}, {y1:R}))",
                    surface => Python.pygame.draw.aaline(surface, color, (x0, y0), (x1, y1)),
                    surface => Draw.Aaline(surface, ToColor(color), x0, y0, x1, y1)));
            }

            AssertCases("draw.aaline", cases, minimumDrawnShare: 0.8);
        }

        /// <summary>
        ///     <c>draw.polygon</c> filled (width 0), the box2d envs' terrain, hulls and flags: 500 polygons of 4 to 8 vertices
        ///     (convex, star-shaped and self-intersecting, thin slivers, rows with horizontal edges inside, far off the surface,
        ///     float vertices truncated toward zero) and 400 triangles, which pygame fills on a path of their own (flat ones and
        ///     ones with tied rows included), any colour and alpha.
        /// </summary>
        [TestMethod]
        public void Polygon_Filled_MatchesPygame() {
            var rng = new Random(20261112);
            var cases = new List<DrawCase>();
            for (int i = 0; i < 900; i++) {
                (double X, double Y)[] points = RandomPolygon(rng, i < 500 ? rng.Next(4, 9) : 3, i);
                (int R, int G, int B, int A) color = RandomColor(rng);
                cases.Add(new DrawCase($"polygon({color}, [{string.Join(", ", points.Select(p => $"({p.X:R}, {p.Y:R})"))}])",
                    surface => Python.pygame.draw.polygon(surface, color, points, 0),
                    surface => Draw.Polygon(surface, ToColor(color), points)));
            }

            AssertCases("draw.polygon", cases);
        }

        /// <summary>
        ///     <c>draw.polygon</c> with a width (BipedalWalker's outlines) and <c>draw.lines</c> (BipedalWalker's flag): 200
        ///     outlines at width 1 (pygame's closed polyline), 200 open and closed polylines of 2 to 7 points, closed two-point
        ///     ones (one segment only), widths 0 and -1 (nothing), and endpoints millions of pixels away, which Gym.NET steps
        ///     only across the surface from the error term's closed form while pygame walks every pixel.
        /// </summary>
        /// <remarks>
        ///     NaN, infinite and out-of-range endpoints (pygame's <c>INT_MIN</c>) are left out on purpose: pygame walks every one
        ///     of a line's two billion steps to the surface, minutes per line (measured), so they cannot run in a gate. The
        ///     conversion they rely on is the circles' (NaN centres in <see cref="Circle_Filled_MatchesPygame"/>), and the
        ///     stepping from far away is these million-pixel endpoints'.
        /// </remarks>
        [TestMethod]
        public void PolygonOutlineAndLines_MatchPygame() {
            var rng = new Random(20261113);
            var cases = new List<DrawCase>();
            for (int i = 0; i < 200; i++) {
                (double X, double Y)[] points = RandomPolygon(rng, rng.Next(3, 8), i);
                (int R, int G, int B, int A) color = RandomColor(rng);
                long width = i % 20 == 0 ? -1 : 1;
                cases.Add(new DrawCase($"polygon({color}, {points.Length} points, width={width})",
                    surface => Python.pygame.draw.polygon(surface, color, points, width),
                    surface => Draw.Polygon(surface, ToColor(color), points, (int)width)));
            }

            for (int i = 0; i < 200; i++) {
                int count = i % 10 == 0 ? 2 : rng.Next(2, 8);
                var points = new (double X, double Y)[count];
                for (int k = 0; k < count; k++)
                    points[k] = (rng.NextDouble() * (Width + 40) - 20, rng.NextDouble() * (Height + 40) - 20);
                if (i % 25 == 3)
                    points[rng.Next(count)] = (i % 2 == 0 ? -2.5e6 - rng.NextDouble() : 3.1e6 + rng.NextDouble(), rng.NextDouble() * Height);
                if (i % 25 == 7)
                    points[rng.Next(count)] = (rng.NextDouble() * Width, -1.7e6 - rng.NextDouble());
                bool closed = i % 3 == 0;
                long width = i % 40 == 5 ? 0 : 1;
                (int R, int G, int B, int A) color = RandomColor(rng);
                cases.Add(new DrawCase($"lines({color}, closed={closed}, [{string.Join(", ", points.Select(p => $"({p.X:R}, {p.Y:R})"))}], width={width})",
                    surface => Python.pygame.draw.lines(surface, color, closed, points, width),
                    surface => Draw.Lines(surface, ToColor(color), closed, points, (int)width)));
            }

            for (int i = 0; i < 40; i++) {
                // Both endpoints far off on one axis each: the closed form starts the walk deep into a long line.
                double x0 = i % 2 == 0 ? -2.2e6 - rng.NextDouble() : rng.NextDouble() * Width, y0 = rng.NextDouble() * Height;
                double x1 = rng.NextDouble() * Width, y1 = i % 4 < 2 ? 2.4e6 + rng.NextDouble() : -2.9e6 - rng.NextDouble();
                (int R, int G, int B, int A) color = RandomColor(rng);
                cases.Add(new DrawCase($"line({color}, ({x0:R}, {y0:R}), ({x1:R}, {y1:R}))",
                    surface => Python.pygame.draw.line(surface, color, (x0, y0), (x1, y1)),
                    surface => Draw.Line(surface, ToColor(color), x0, y0, x1, y1)));
            }

            AssertCases("draw.polygon/lines", cases);
        }

        /// <summary>
        ///     <c>draw.circle</c> filled (LunarLander's particles): every radius from 0 to 60 around a fractional centre, and 300
        ///     random circles across and around the surface's edges with fractional radii (truncated) and NaN centres or radii
        ///     (pygame's <c>INT_MIN</c>: nothing), plus widths -1 (nothing).
        /// </summary>
        [TestMethod]
        public void Circle_Filled_MatchesPygame() {
            var rng = new Random(20261114);
            var cases = new List<DrawCase>();
            for (int r = 0; r <= 60; r++) {
                (double X, double Y) center = (31.7, 23.2);
                double radius = r + (r % 3 == 0 ? 0.0 : rng.NextDouble());
                (int R, int G, int B, int A) color = RandomColor(rng);
                cases.Add(new DrawCase($"circle({color}, {center}, {radius:R})",
                    surface => Python.pygame.draw.circle(surface, color, center, radius, 0),
                    surface => Draw.Circle(surface, ToColor(color), center.X, center.Y, radius)));
            }

            for (int i = 0; i < 300; i++) {
                // Centres around and just beyond the edges, so most discs show while many are clipped (a few lie wholly off).
                (double X, double Y) center = (rng.NextDouble() * (Width + 40) - 20, rng.NextDouble() * (Height + 40) - 20);
                double radius = i % 3 == 0 ? rng.NextDouble() * 4 : rng.NextDouble() * 40;
                if (i % 50 == 1)
                    center = (double.NaN, center.Y);
                if (i % 50 == 2)
                    radius = double.NaN;
                long width = i % 30 == 4 ? -1 : 0;
                (int R, int G, int B, int A) color = RandomColor(rng);
                cases.Add(new DrawCase($"circle({color}, ({center.X:R}, {center.Y:R}), {radius:R}, width={width})",
                    surface => Python.pygame.draw.circle(surface, color, center, radius, width),
                    surface => Draw.Circle(surface, ToColor(color), center.X, center.Y, radius, (int)width)));
            }

            AssertCases("draw.circle", cases);
        }

        /// <summary>
        ///     The new primitives' refusals, with pygame's types and messages: a filled polygon of fewer than 3 points, an
        ///     outline or a polyline of fewer than 2 (checked before the width).
        /// </summary>
        [TestMethod]
        public void PolygonAndLines_TooFewPoints_RaiseAsPygame() {
            (string What, Func<PyObject> Python, Func<object?> DotNet)[] cases = {
                ("polygon of 2 points", () => DrawOnFreshSurface(s => Python.pygame.draw.polygon(s, (1, 2, 3, 255), new[] { (0.0, 0.0), (5.0, 5.0) }, 0)),
                    () => DrawDotNet(s => Draw.Polygon(s, new Color(1, 2, 3), new[] { (0.0, 0.0), (5.0, 5.0) }))),
                ("polygon outline of 1 point", () => DrawOnFreshSurface(s => Python.pygame.draw.polygon(s, (1, 2, 3, 255), new[] { (0.0, 0.0) }, -1)),
                    () => DrawDotNet(s => Draw.Polygon(s, new Color(1, 2, 3), new[] { (0.0, 0.0) }, -1))),
                ("lines of 1 point", () => DrawOnFreshSurface(s => Python.pygame.draw.lines(s, (1, 2, 3, 255), false, new[] { (0.0, 0.0) }, 0)),
                    () => DrawDotNet(s => Draw.Lines(s, new Color(1, 2, 3), false, new[] { (0.0, 0.0) }, 0))),
            };
            foreach ((string what, Func<PyObject> python, Func<object?> dotnet) in cases)
                ByteContract.AssertSame(what, Outcome.CapturePython(python), Outcome.Capture(dotnet));
        }

        /// <summary>Draws on a fresh pygame surface and returns <c>None</c>, so an outcome compares the call's exception alone.</summary>
        /// <param name="draw">The drawing; runs under the GIL.</param>
        /// <returns>A new owned <c>None</c>.</returns>
        /// <exception cref="PythonException">Thrown when drawing raises.</exception>
        private static PyObject DrawOnFreshSurface(Action<PyObject> draw) {
            using PyObject surface = Python.pygame.Surface(Width, Height);
            draw(surface);
            return PyConvert.ToPython(null);
        }

        /// <summary>Draws on a fresh Gym.Rendering surface and returns <see langword="null"/>, the counterpart of <see cref="DrawOnFreshSurface"/>.</summary>
        /// <param name="draw">The drawing.</param>
        /// <returns><see langword="null"/>.</returns>
        private static object? DrawDotNet(Action<Surface> draw) {
            draw(new Surface(Width, Height));
            return null;
        }

        /// <summary>
        ///     A random polygon for the fill and outline corpora, varied by the case's index: integer vertices, big ones
        ///     reaching far off the surface, thin slivers, rows shared by several vertices (horizontal edges), star shapes, a
        ///     flat one, or plain random floats around the surface.
        /// </summary>
        /// <param name="rng">The generator.</param>
        /// <param name="count">The number of vertices.</param>
        /// <param name="index">The case's index, which picks the kind.</param>
        /// <returns>The vertices.</returns>
        private static (double X, double Y)[] RandomPolygon(Random rng, int count, int index) {
            var points = new (double X, double Y)[count];
            double Uniform(double low, double high) => low + rng.NextDouble() * (high - low);
            switch (index % 7) {
                case 0:
                    for (int k = 0; k < count; k++)
                        points[k] = (rng.Next(-10, Width + 10), rng.Next(-10, Height + 10));
                    break;
                case 1:
                    for (int k = 0; k < count; k++)
                        points[k] = (Uniform(-80, Width + 80), Uniform(-80, Height + 80));
                    break;
                case 2: {
                    double y = Uniform(-5, Height + 5);
                    for (int k = 0; k < count; k++)
                        points[k] = (Uniform(-10, Width + 10), y + Uniform(-1.5, 1.5));
                    break;
                }

                case 3: {
                    int[] rows = { rng.Next(-5, Height + 5), rng.Next(-5, Height + 5), rng.Next(-5, Height + 5) };
                    for (int k = 0; k < count; k++)
                        points[k] = (Uniform(-10, Width + 10), rows[rng.Next(rows.Length)] + new[] { 0.0, 0.25, 0.75 }[rng.Next(3)]);
                    break;
                }

                case 4: {
                    double cx = Uniform(-10, Width + 10), cy = Uniform(-10, Height + 10), radius = Uniform(1, 40);
                    double[] angles = Enumerable.Range(0, count).Select(_ => Uniform(0, 2 * Math.PI)).OrderBy(a => a).ToArray();
                    for (int k = 0; k < count; k++)
                        points[k] = (cx + Math.Cos(angles[k]) * radius * Uniform(0.3, 1), cy + Math.Sin(angles[k]) * radius * Uniform(0.3, 1));
                    break;
                }

                case 5: {
                    double y = Uniform(-2, Height + 2);
                    for (int k = 0; k < count; k++)
                        points[k] = (Uniform(-10, Width + 10), y + Uniform(0, 0.9));
                    break;
                }

                default:
                    for (int k = 0; k < count; k++)
                        points[k] = (Uniform(-10, Width + 10), Uniform(-10, Height + 10));
                    break;
            }

            return points;
        }

        /// <summary>The Gymnasium envs folder of the pinned submodule, where the PNG assets live.</summary>
        /// <exception cref="AssertInconclusiveException">Thrown when the session found no checkout (no submodule to read assets from).</exception>
        private static string GymnasiumEnvs => GymPythonSession.Repository?.GymnasiumPackage is { } package
            ? Path.Combine(package, "envs")
            : throw new AssertInconclusiveException("No repository layout: refs/Gymnasium is needed for the env assets.");

        /// <summary>One drawing, spelled for each side, and how reports name it.</summary>
        /// <param name="Description">The call in pygame's spelling.</param>
        /// <param name="Python">Draws on a pygame surface; runs under the GIL.</param>
        /// <param name="DotNet">Draws on a Gym.Rendering surface.</param>
        /// <param name="Alternatives">
        ///     Gym.NET's drawings of pygame's other outcomes where pygame's own is undefined (a read of memory it does not own),
        ///     any of which pygame's frame may equal; <see langword="null"/> for a defined drawing, which must be
        ///     <paramref name="DotNet"/>'s.
        /// </param>
        private sealed record DrawCase(string Description, Action<PyObject> Python, Action<Surface> DotNet, IReadOnlyList<Action<Surface>>? Alternatives = null);

        /// <summary>
        ///     Draws each case on a fresh frame on both sides, over a background of its own (<see cref="Background"/>), and
        ///     requires identical frames (artifacts are written for the first that differs). A case with alternatives also
        ///     passes when pygame's frame is one of them.
        /// </summary>
        /// <param name="family">The primitive family, for reports.</param>
        /// <param name="cases">The drawings.</param>
        /// <param name="minimumDrawnShare">The share of cases that must change a pixel on pygame's side, so the comparison is not vacuous.</param>
        /// <returns>How many cases passed as one of their alternatives rather than as Gym.NET's own drawing.</returns>
        /// <exception cref="ParityException">Thrown at the first case whose frames differ (from Gym.NET's own drawing, when no alternative matched either).</exception>
        /// <exception cref="AssertFailedException">Thrown when too few cases drew anything on pygame's side.</exception>
        private static int AssertCases(string family, IReadOnlyList<DrawCase> cases, double minimumDrawnShare = 0.5) {
            int drawn = 0, alternatives = 0;
            for (int index = 0; index < cases.Count; index++) {
                DrawCase drawing = cases[index];
                (int R, int G, int B) fill = Background(index);
                Frame python = PythonRuntime.Run(() => {
                    using PyObject surface = Python.pygame.Surface(Width, Height);
                    Python.pygame.surface.Surface.fill(surface, fill);
                    drawing.Python(surface);
                    return PythonFrame(surface);
                });
                Frame dotnet = Drawn(drawing.DotNet, fill);
                // An undefined drawing passes as any outcome pygame can produce; a failure still reports Gym.NET's own.
                if (drawing.Alternatives is { } others && !python.Rgb.AsSpan().SequenceEqual(dotnet.Rgb)
                    && others.Any(other => python.Rgb.AsSpan().SequenceEqual(Drawn(other, fill).Rgb)))
                    alternatives++;
                else
                    FrameDiff.AssertSame($"{family} #{index}: {drawing.Description} on {fill}", python, dotnet, $"pygame.{family}");
                if (ChangesBackground(python, fill))
                    drawn++;
            }

            Assert.IsTrue(drawn >= minimumDrawnShare * cases.Count, $"{family}: only {drawn} of {cases.Count} cases drew anything on pygame's surface, so the comparison proves little.");
            return alternatives;
        }

        /// <summary>The background a case is drawn over: varied, so anti-aliased weights show in every channel.</summary>
        /// <param name="index">The case's position in its corpus.</param>
        /// <returns>The fill colour.</returns>
        private static (int R, int G, int B) Background(int index) => ((index * 37) % 256, (index * 91 + 50) % 256, (index * 53 + 200) % 256);

        /// <summary>Gym.NET's frame of one drawing over a background.</summary>
        /// <param name="draw">The drawing.</param>
        /// <param name="fill">The background.</param>
        /// <returns>The frame.</returns>
        private static Frame Drawn(Action<Surface> draw, (int R, int G, int B) fill) {
            var surface = new Surface(Width, Height);
            surface.Fill(new Color((byte)fill.R, (byte)fill.G, (byte)fill.B));
            draw(surface);
            return new Frame(Height, Width, surface.ToRgbBytes());
        }

        /// <summary>
        ///     The polylines of <see cref="Aalines_MatchesPygame"/>: 150 of 2 to 6 points around the surface (seed 20261116),
        ///     open and closed in turn, in random colours.
        /// </summary>
        /// <returns>Each polyline's colour, closure and points, in corpus order.</returns>
        private static IEnumerable<((int R, int G, int B, int A) Color, bool Closed, (double X, double Y)[] Points)> AalinesCorpus() {
            var rng = new Random(20261116);
            for (int i = 0; i < 150; i++) {
                int count = rng.Next(2, 7);
                (double X, double Y)[] points = Enumerable.Range(0, count).Select(_ => (rng.NextDouble() * (Width + 20) - 10, rng.NextDouble() * (Height + 20) - 10)).ToArray();
                bool closed = i % 2 == 1;
                (int R, int G, int B, int A) color = RandomColor(rng);
                yield return (color, closed, points);
            }
        }

        /// <summary>
        ///     One <c>draw.aalines</c> case; a two-point polyline also carries pygame's two outcomes of its read past the buffer
        ///     (its first test false, and true), drawn through <c>Draw</c>'s private seam <c>DrawAalines</c> by reflection.
        /// </summary>
        /// <param name="color">The colour.</param>
        /// <param name="closed">Whether the polyline is closed.</param>
        /// <param name="points">The points.</param>
        /// <returns>The case.</returns>
        /// <exception cref="AssertFailedException">Thrown when the seam is missing (renamed or retyped in <c>Draw.Aaline.cs</c>).</exception>
        private static DrawCase AalinesCase((int R, int G, int B, int A) color, bool closed, (double X, double Y)[] points) {
            string description = $"aalines({color}, closed={(closed ? "True" : "False")}, [{string.Join(", ", points.Select(p => $"({p.X:R}, {p.Y:R})"))}])";
            List<Action<Surface>>? alternatives = null;
            if (points.Length == 2) {
                MethodInfo seam = typeof(Draw).GetMethod("DrawAalines", BindingFlags.NonPublic | BindingFlags.Static)
                                  ?? throw new AssertFailedException("Draw.DrawAalines, the seam for the float pygame reads past its buffer, is missing.");
                // NaN makes pygame's first test false and an infinity makes it true: the two outcomes its memory can give.
                alternatives = new[] { float.NaN, float.PositiveInfinity }
                    .Select(pastTheBuffer => (Action<Surface>)(surface => seam.Invoke(null, new object[] { surface, ToColor(color), closed, points, pastTheBuffer })))
                    .ToList();
            }

            return new DrawCase(description,
                surface => Python.pygame.draw.aalines(surface, color, closed, points),
                surface => Draw.Aalines(surface, ToColor(color), closed, points),
                alternatives);
        }

        /// <summary>Requires a pygame surface's RGBA bytes to equal a Gym.Rendering surface's, naming the first differing pixel.</summary>
        /// <param name="label">What was compared.</param>
        /// <param name="python">pygame's bytes, <c>(height, width, 4)</c>.</param>
        /// <param name="dotnet">Gym.NET's surface.</param>
        /// <param name="width">The width both have.</param>
        /// <param name="height">The height both have.</param>
        /// <exception cref="AssertFailedException">Thrown at the first differing byte.</exception>
        private static void AssertSameRgba(string label, byte[] python, Surface dotnet, int width, int height) {
            byte[] ours = dotnet.ToRgbaBytes();
            Assert.AreEqual(python.Length, ours.Length, $"{label}: byte count");
            for (int i = 0; i < python.Length; i++) {
                if (python[i] != ours[i]) {
                    int pixel = i / 4;
                    Assert.Fail($"{label}: pixel ({pixel % width}, {pixel / width}) channel {"RGBA"[i % 4]}: pygame {python[i]}, Gym.NET {ours[i]} (surface {width}x{height}).");
                }
            }
        }

        /// <summary>
        ///     A pygame surface's pixels as <c>(height, width, 4)</c> RGBA bytes, its size, and whether it has per-pixel alpha
        ///     (<c>array3d</c> and <c>array_alpha</c>, reordered to rows; 255 alpha without per-pixel alpha). Must run under the GIL.
        /// </summary>
        /// <param name="surface">The surface; borrowed.</param>
        /// <returns>The bytes, whether the surface has per-pixel alpha, its width and height.</returns>
        private static (byte[] Rgba, bool Alpha, int Width, int Height) PythonRgba(PyObject surface) {
            using PyObject columns = Python.pygame.surfarray.array3d(surface);
            using PyObject alphaPlane = Python.pygame.surfarray.array_alpha(surface);
            using PyObject flags = Python.pygame.surface.Surface.get_flags(surface);
            ValueFacts rgb = ValueFacts.FromPython(columns), alpha = ValueFacts.FromPython(alphaPlane);
            int width = checked((int)rgb.Shape[0]), height = checked((int)rgb.Shape[1]);
            var rgba = new byte[width * height * 4];
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    int to = (y * width + x) * 4;
                    rgba[to] = rgb.Bytes[(x * height + y) * 3];
                    rgba[to + 1] = rgb.Bytes[(x * height + y) * 3 + 1];
                    rgba[to + 2] = rgb.Bytes[(x * height + y) * 3 + 2];
                    rgba[to + 3] = alpha.Bytes[x * height + y];
                }
            }

            // pygame.SRCALPHA is 0x00010000.
            bool hasAlpha = (flags.As<long>() & 0x10000) != 0;
            return (rgba, hasAlpha, width, height);
        }

        /// <summary>A pygame surface as a <see cref="Frame"/>: <c>surfarray.array3d</c> reordered to rows. Must run under the GIL.</summary>
        /// <param name="surface">The pygame surface; borrowed.</param>
        /// <returns>The frame.</returns>
        private static Frame PythonFrame(PyObject surface) {
            (byte[] rgba, bool _, int width, int height) = PythonRgba(surface);
            var rgb = new byte[width * height * 3];
            for (int i = 0, j = 0; i < rgb.Length; i += 3, j += 4) {
                rgb[i] = rgba[j];
                rgb[i + 1] = rgba[j + 1];
                rgb[i + 2] = rgba[j + 2];
            }

            return new Frame(height, width, rgb);
        }

        /// <summary><c>pygame.image.frombytes(bytes, (w, h), format)</c> with the bytes crossed through NumPy. Must run under the GIL.</summary>
        /// <param name="bytes">The pixels.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="format">The format.</param>
        /// <returns>A new owned reference to the surface.</returns>
        /// <exception cref="PythonException">Thrown with pygame's error for bytes that do not fit.</exception>
        private static PyObject FromBytesPython(byte[] bytes, int width, int height, string format) {
            using NDArray array = np.array(bytes);
            using PyObject numpyArray = array.ToNumpyCopy();
            using PyObject pythonBytes = Python.numpy.ndarray.tobytes(numpyArray);
            return Python.pygame.image.frombytes(pythonBytes, width, height, format);
        }

        /// <summary>Random RGBA pixels: a quarter fully transparent, a quarter opaque, the rest of any alpha.</summary>
        /// <param name="rng">The case generator.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <returns>The bytes, row by row.</returns>
        private static byte[] RandomRgba(Random rng, int width, int height) {
            byte[] bytes = RandomBytes(rng, width * height * 4);
            for (int i = 3; i < bytes.Length; i += 4) {
                bytes[i] = rng.Next(4) switch {
                    0 => 0,
                    1 => 255,
                    _ => bytes[i],
                };
            }

            return bytes;
        }

        /// <summary>Random bytes.</summary>
        /// <param name="rng">The case generator.</param>
        /// <param name="count">How many.</param>
        /// <returns>The bytes.</returns>
        private static byte[] RandomBytes(Random rng, int count) {
            var bytes = new byte[count];
            rng.NextBytes(bytes);
            return bytes;
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

        /// <summary>A random colour: opaque half of the time, otherwise any alpha (which the draw functions ignore on a frame).</summary>
        /// <param name="rng">The case generator.</param>
        /// <returns>The colour as <c>(r, g, b, a)</c>.</returns>
        private static (int R, int G, int B, int A) RandomColor(Random rng) =>
            (rng.Next(256), rng.Next(256), rng.Next(256), rng.Next(2) == 0 ? 255 : rng.Next(256));

        /// <summary>The colour as <see cref="Color"/>.</summary>
        /// <param name="color">The colour as a tuple.</param>
        /// <returns>The colour.</returns>
        private static Color ToColor((int R, int G, int B, int A) color) => new((byte)color.R, (byte)color.G, (byte)color.B, (byte)color.A);
    }
}
