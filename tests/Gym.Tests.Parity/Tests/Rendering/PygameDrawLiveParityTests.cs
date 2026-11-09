using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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
    ///     <c>draw.aaline</c> and <c>draw.aalines</c> have not converged (the class remarks of <see cref="Draw"/>): their cells
    ///     are open, reported to the maintainer as D4 requires.
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
        ///     <c>draw.aaline</c> on 300 random segments, any colour, over varied backgrounds. Open: the model leaves about one
        ///     drawn pixel in ten thousand off by one (D4, reported to the maintainer).
        /// </summary>
        [TestMethod]
        [OpenBugs("P7")]
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
        ///     <c>draw.aalines</c> on 150 random polylines of 2 to 6 points, open and closed. Open: pygame's rule where segments
        ///     join is not settled (D4, reported to the maintainer).
        /// </summary>
        [TestMethod]
        [OpenBugs("P7")]
        public void Aalines_MatchesPygame() {
            var rng = new Random(20261116);
            var cases = new List<DrawCase>();
            for (int i = 0; i < 150; i++) {
                int count = rng.Next(2, 7);
                var points = Enumerable.Range(0, count).Select(_ => (rng.NextDouble() * (Width + 20) - 10, rng.NextDouble() * (Height + 20) - 10)).ToArray();
                bool closed = i % 2 == 1;
                (int R, int G, int B, int A) color = RandomColor(rng);
                cases.Add(new DrawCase($"aalines({color}, closed={closed}, [{string.Join(", ", points.Select(p => $"({p.Item1:R}, {p.Item2:R})"))}])",
                    surface => Python.pygame.draw.aalines(surface, color, closed, points),
                    surface => Draw.Aalines(surface, ToColor(color), closed, points)));
            }

            AssertCases("draw.aalines", cases);
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
        private sealed record DrawCase(string Description, Action<PyObject> Python, Action<Surface> DotNet);

        /// <summary>
        ///     Draws each case on a fresh frame on both sides, over a background of its own, and requires identical frames
        ///     (artifacts are written for the first that differs).
        /// </summary>
        /// <param name="family">The primitive family, for reports.</param>
        /// <param name="cases">The drawings.</param>
        /// <param name="minimumDrawnShare">The share of cases that must change a pixel on pygame's side, so the comparison is not vacuous.</param>
        /// <exception cref="ParityException">Thrown at the first case whose frames differ.</exception>
        /// <exception cref="AssertFailedException">Thrown when too few cases drew anything on pygame's side.</exception>
        private static void AssertCases(string family, IReadOnlyList<DrawCase> cases, double minimumDrawnShare = 0.5) {
            int drawn = 0;
            for (int index = 0; index < cases.Count; index++) {
                DrawCase drawing = cases[index];
                // A varied background makes anti-aliased weights visible in every channel.
                (int R, int G, int B) fill = ((index * 37) % 256, (index * 91 + 50) % 256, (index * 53 + 200) % 256);
                Frame python = PythonRuntime.Run(() => {
                    using PyObject surface = Python.pygame.Surface(Width, Height);
                    Python.pygame.surface.Surface.fill(surface, fill);
                    drawing.Python(surface);
                    return PythonFrame(surface);
                });
                var dotnet = new Surface(Width, Height);
                dotnet.Fill(new Color((byte)fill.R, (byte)fill.G, (byte)fill.B));
                drawing.DotNet(dotnet);
                FrameDiff.AssertSame($"{family} #{index}: {drawing.Description} on {fill}", python, new Frame(Height, Width, dotnet.ToRgbBytes()), $"pygame.{family}");
                if (ChangesBackground(python, fill))
                    drawn++;
            }

            Assert.IsTrue(drawn >= minimumDrawnShare * cases.Count, $"{family}: only {drawn} of {cases.Count} cases drew anything on pygame's surface, so the comparison proves little.");
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
