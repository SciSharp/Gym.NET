#nullable enable
using System;
using System.IO;
using Gym.Rendering;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Rendering {
    /// <summary>
    ///     The P7 additions to <c>Gym.Rendering</c>, checked without pygame: per-pixel-alpha surfaces, the alpha blit, nearest
    ///     and smooth scaling, <c>draw.line</c> and <c>draw.rect</c>, the overflow and two-point quirks of the anti-aliased
    ///     lines (DR-37), <c>image.frombytes</c> and <c>image.load</c>, and what is refused. Where a value is pinned it is
    ///     pygame-ce 2.5.8's own output, observed while the behaviour was derived; the random corpora against live pygame are
    ///     the parity suite's (<c>PygameDrawLiveParityTests</c>).
    /// </summary>
    [TestClass]
    public class PygameImageAndDrawTests {
        /// <summary>A surface with per-pixel alpha starts transparent, and <c>fill</c> and <c>get_at</c> keep the alpha.</summary>
        [TestMethod]
        public void AlphaSurface_StartsTransparent_AndKeepsAlpha() {
            var surface = new Surface(2, 2, hasAlpha: true);
            Assert.IsTrue(surface.HasAlpha);
            Assert.AreEqual(new Color(0, 0, 0, 0), surface.GetAt(1, 1));
            surface.Fill(new Color(10, 20, 30, 40));
            Assert.AreEqual(new Color(10, 20, 30, 40), surface.GetAt(0, 1));
            CollectionAssert.AreEqual(new byte[] { 10, 20, 30, 40 }, surface.ToRgbaBytes()[..4]);
            Assert.IsFalse(new Surface(1, 1).HasAlpha);
        }

        /// <summary>
        ///     Blitting per-pixel alpha onto a frame computes <c>d + (((s - d) * a + s) &gt;&gt; 8)</c>, flooring: the three
        ///     triples that told it from the textbook formulas, as pygame computed them; alpha 0 keeps, alpha 255 copies.
        /// </summary>
        [TestMethod]
        public void Blit_PerPixelAlpha_BlendsAsPygame() {
            foreach ((byte s, byte a, byte d, byte expected) in new (byte, byte, byte, byte)[] { (0, 1, 1, 0), (0, 2, 128, 127), (1, 1, 2, 2), (77, 0, 9, 9), (77, 255, 9, 77) }) {
                var frame = new Surface(1, 1);
                frame.Fill(new Color(d, d, d));
                frame.Blit(ImageIO.FromBytes(new[] { s, s, s, a }, 1, 1, "RGBA"), 0, 0);
                Assert.AreEqual(new Color(expected, expected, expected), frame.GetAt(0, 0), $"s={s} a={a} d={d}");
            }
        }

        /// <summary>
        ///     What is not modelled is refused, never guessed: blitting onto, or drawing on, a surface with per-pixel alpha,
        ///     outlines wider than 1, and growing with <c>smoothscale</c>.
        /// </summary>
        [TestMethod]
        public void UnmodelledCases_AreRefused() {
            var alpha = new Surface(4, 4, hasAlpha: true);
            Assert.ThrowsExactly<NotSupportedException>(() => alpha.Blit(new Surface(1, 1), 0, 0));
            Assert.ThrowsExactly<NotSupportedException>(() => Draw.Line(alpha, new Color(1, 2, 3), 0L, 0L, 3L, 3L));
            Assert.ThrowsExactly<NotSupportedException>(() => GfxDraw.HLine(alpha, 0, 3, 1, new Color(1, 2, 3)));
            Assert.ThrowsExactly<NotSupportedException>(() => Draw.Rect(new Surface(4, 4), new Color(1, 2, 3), 0, 0, 2, 2, 2));
            Assert.ThrowsExactly<NotSupportedException>(() => Transform.Smoothscale(alpha, 5, 4));
        }

        /// <summary>
        ///     <c>transform.scale</c>: doubling repeats every pixel twice, alpha included; the same size copies; a zero size is
        ///     empty; a negative size raises pygame's <see cref="ValueError"/>.
        /// </summary>
        [TestMethod]
        public void Scale_RepeatsPixels_AndChecksTheSize() {
            Surface source = ImageIO.FromBytes(new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 }, 2, 1, "RGBA");
            Surface doubled = Transform.Scale(source, 4, 2);
            Assert.IsTrue(doubled.HasAlpha);
            Assert.AreEqual(new Color(1, 2, 3, 4), doubled.GetAt(1, 1));
            Assert.AreEqual(new Color(5, 6, 7, 8), doubled.GetAt(2, 0));
            CollectionAssert.AreEqual(source.ToRgbaBytes(), Transform.Scale(source, 2, 1).ToRgbaBytes());
            Assert.AreEqual(0, Transform.Scale(source, 0, 3).Width);
            Assert.AreEqual("Cannot scale to negative size", Assert.ThrowsExactly<ValueError>(() => Transform.Scale(source, -1, 1)).Message);
        }

        /// <summary>
        ///     <c>transform.smoothscale</c> shrinking, pinned to pygame's own outputs: a constant 255 row of 3 shrinks to 254s
        ///     (the 14-bit weights truncate), <c>[10, 20]</c> to 15, <c>[255, 0, 0]</c> to 84, and a 255 impulse at the start of
        ///     a 4-pixel row shrunk to 3 gives <c>[191, 0, 0]</c>, at the end <c>[0, 0, 190]</c>.
        /// </summary>
        [TestMethod]
        public void Smoothscale_Shrinks_AsPygame() {
            Assert.AreEqual(254, Red(Transform.Smoothscale(Row(255, 255, 255), 2, 1), 1));
            Assert.AreEqual(15, Red(Transform.Smoothscale(Row(10, 20), 1, 1), 0));
            Assert.AreEqual(84, Red(Transform.Smoothscale(Row(255, 0, 0), 1, 1), 0));
            Surface start = Transform.Smoothscale(Row(255, 0, 0, 0), 3, 1), end = Transform.Smoothscale(Row(0, 0, 0, 255), 3, 1);
            CollectionAssert.AreEqual(new[] { 191, 0, 0 }, new[] { Red(start, 0), Red(start, 1), Red(start, 2) });
            CollectionAssert.AreEqual(new[] { 0, 0, 190 }, new[] { Red(end, 0), Red(end, 1), Red(end, 2) });
            Assert.AreEqual("Cannot scale to negative size", Assert.ThrowsExactly<ValueError>(() => Transform.Smoothscale(Row(1), -1, 1)).Message);
        }

        /// <summary>
        ///     <c>draw.line</c> breaks ties toward its start, as pygame's does: <c>(0, 0)-(4, 2)</c> lights <c>(1, 0)</c> while
        ///     <c>(4, 2)-(0, 0)</c> lights <c>(1, 1)</c>; float endpoints are truncated.
        /// </summary>
        [TestMethod]
        public void Line_BreaksTiesTowardItsStart() {
            var forward = new Surface(5, 3);
            Draw.Line(forward, new Color(255, 255, 255), 0L, 0L, 4L, 2L);
            Assert.AreEqual(new Color(255, 255, 255), forward.GetAt(1, 0));
            Assert.AreEqual(new Color(0, 0, 0), forward.GetAt(1, 1));

            var backward = new Surface(5, 3);
            Draw.Line(backward, new Color(255, 255, 255), 4L, 2L, 0L, 0L);
            Assert.AreEqual(new Color(255, 255, 255), backward.GetAt(1, 1));

            var truncated = new Surface(5, 3);
            Draw.Line(truncated, new Color(255, 255, 255), 0.9, 0.9, 4.9, 2.9);
            CollectionAssert.AreEqual(forward.ToRgbBytes(), truncated.ToRgbBytes());
        }

        /// <summary>
        ///     <c>draw.rect</c> at width 1 outlines, and fills a clipped sliver at most 2 pixels across, as pygame does:
        ///     <c>(11, -1, 23, 3)</c> on a 64 × 48 frame lights rows 0 and 1 fully (46 pixels), <c>(10, -1, 5, 4)</c> 9 pixels.
        /// </summary>
        [TestMethod]
        public void Rect_OutlinesAndFillsClippedSlivers() {
            Assert.AreEqual(46, Lit(64, 48, surface => Draw.Rect(surface, new Color(255, 255, 255), 11, -1, 23, 3, 1)));
            Assert.AreEqual(9, Lit(64, 48, surface => Draw.Rect(surface, new Color(255, 255, 255), 10, -1, 5, 4, 1)));
            Assert.AreEqual(20, Lit(64, 48, surface => Draw.Rect(surface, new Color(255, 255, 255), 2, 2, 5, 4, 0)));
            Assert.AreEqual(0, Lit(64, 48, surface => Draw.Rect(surface, new Color(255, 255, 255), 2, 2, 5, 4, -1)));
        }

        /// <summary>
        ///     <c>draw.aaline</c> from <c>(-0.5, 10.75)</c> to <c>(40, 10.75)</c>: the start truncates to column 0, so the first
        ///     endpoint's coverage there is <c>0.75 × 1.5 = 1.125</c> and the blend leaves 0..255; pygame keeps the low byte of
        ///     the truncated int, 225 for black on white and 30 for white on black, where a saturating cast would give 0 and 255.
        ///     Every value is pygame-ce 2.5.8's own pixel. Pinned on both target frameworks because a direct float-to-byte cast
        ///     saturates from .NET 9 on.
        /// </summary>
        [TestMethod]
        public void Aaline_CoverageAboveOne_KeepsTheLowByteAsPygame() {
            var blackOnWhite = new Surface(64, 64);
            blackOnWhite.Fill(new Color(255, 255, 255));
            Draw.Aaline(blackOnWhite, new Color(0, 0, 0), -0.5, 10.75, 40.0, 10.75);
            AssertGrey(blackOnWhite, (0, 11, 225), (0, 10, 159), (1, 11, 63), (1, 10, 191));

            var whiteOnBlack = new Surface(64, 64);
            Draw.Aaline(whiteOnBlack, new Color(255, 255, 255), -0.5, 10.75, 40.0, 10.75);
            AssertGrey(whiteOnBlack, (0, 11, 30), (0, 10, 95), (1, 11, 191), (1, 10, 63));
        }

        /// <summary>
        ///     <c>draw.aalines</c> through the same two points blends the segment twice, pygame's two-point quirk: as the first
        ///     segment with its start endpoint and as the last with its end endpoint, each leaving the other endpoint out because
        ///     the end point is not whole. Column 0 becomes 56 and 119, column 1 15 and 143: pygame-ce 2.5.8's own pixels.
        /// </summary>
        [TestMethod]
        public void Aalines_TwoPoints_BlendTheSegmentTwiceAsPygame() {
            var surface = new Surface(64, 64);
            surface.Fill(new Color(255, 255, 255));
            Draw.Aalines(surface, new Color(0, 0, 0), false, new[] { (-0.5, 10.75), (40.0, 10.75) });
            AssertGrey(surface, (0, 11, 56), (0, 10, 119), (1, 11, 15), (1, 10, 143));
        }

        /// <summary><c>image.frombytes</c>: row-major pixels, RGBA with alpha and RGB without, and pygame's error messages.</summary>
        [TestMethod]
        public void FromBytes_LaysOutPixels_AndChecksItsArguments() {
            Surface rgb = ImageIO.FromBytes(new byte[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }, 2, 2, "RGB");
            Assert.IsFalse(rgb.HasAlpha);
            Assert.AreEqual(new Color(3, 4, 5), rgb.GetAt(1, 0));
            Assert.AreEqual("Bytes length does not equal format and resolution size", Assert.ThrowsExactly<ValueError>(() => ImageIO.FromBytes(new byte[15], 2, 2, "RGBA")).Message);
            Assert.AreEqual("Resolution must be nonzero positive values", Assert.ThrowsExactly<ValueError>(() => ImageIO.FromBytes(new byte[16], -1, 2, "RGBA")).Message);
            Assert.AreEqual("Unrecognized type of format", Assert.ThrowsExactly<ValueError>(() => ImageIO.FromBytes(new byte[16], 2, 2, "XYZ")).Message);
        }

        /// <summary><c>image.load</c> of Pendulum's arrow (329 × 312, with alpha) from the Gymnasium submodule; a missing file raises.</summary>
        [TestMethod]
        public void Load_ReadsAnRgbaPng() {
            string? arrow = FindAsset(Path.Combine("refs", "Gymnasium", "gymnasium", "envs", "classic_control", "assets", "clockwise.png"));
            if (arrow is null)
                Assert.Inconclusive("refs/Gymnasium is not checked out.");
            Surface image = ImageIO.Load(arrow);
            Assert.AreEqual((329, 312, true), (image.Width, image.Height, image.HasAlpha));
            Assert.ThrowsExactly<FileNotFoundException>(() => ImageIO.Load(arrow + ".missing"));
        }

        /// <summary>A one-row RGBA surface whose red channel holds the values (green and blue zero, alpha opaque).</summary>
        /// <param name="reds">The red values, left to right.</param>
        /// <returns>The surface.</returns>
        private static Surface Row(params byte[] reds) {
            var bytes = new byte[reds.Length * 4];
            for (int i = 0; i < reds.Length; i++) {
                bytes[i * 4] = reds[i];
                bytes[i * 4 + 3] = 255;
            }

            return ImageIO.FromBytes(bytes, reds.Length, 1, "RGBA");
        }

        /// <summary>The red channel of a pixel of a one-row surface.</summary>
        /// <param name="surface">The surface.</param>
        /// <param name="x">The column.</param>
        /// <returns>The red value.</returns>
        private static int Red(Surface surface, int x) => surface.GetAt(x, 0).R;

        /// <summary>How many pixels a drawing lights on a black frame.</summary>
        /// <param name="width">The frame's width.</param>
        /// <param name="height">The frame's height.</param>
        /// <param name="draw">The drawing.</param>
        /// <returns>The count of non-black pixels.</returns>
        private static int Lit(int width, int height, Action<Surface> draw) {
            var surface = new Surface(width, height);
            draw(surface);
            int lit = 0;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    if (surface.GetAt(x, y) != new Color(0, 0, 0))
                        lit++;
                }
            }

            return lit;
        }

        /// <summary>Requires grey pixels: each <c>(x, y)</c> must hold <c>(grey, grey, grey)</c>, the value pygame left there.</summary>
        /// <param name="surface">The surface drawn on.</param>
        /// <param name="pixels">The pixels and the grey each must hold.</param>
        /// <exception cref="AssertFailedException">Thrown at the first pixel that holds another colour.</exception>
        private static void AssertGrey(Surface surface, params (int X, int Y, byte Grey)[] pixels) {
            foreach ((int x, int y, byte grey) in pixels)
                Assert.AreEqual(new Color(grey, grey, grey), surface.GetAt(x, y), $"pixel ({x}, {y})");
        }

        /// <summary>Finds a file by its path relative to the repository root, walking up from the test's directory.</summary>
        /// <param name="relative">The path from the root.</param>
        /// <returns>The full path, or <see langword="null"/> when no ancestor has it.</returns>
        private static string? FindAsset(string relative) {
            for (DirectoryInfo? directory = new DirectoryInfo(AppContext.BaseDirectory); directory != null; directory = directory.Parent) {
                string candidate = Path.Combine(directory.FullName, relative);
                if (File.Exists(candidate))
                    return candidate;
            }

            return null;
        }
    }
}
