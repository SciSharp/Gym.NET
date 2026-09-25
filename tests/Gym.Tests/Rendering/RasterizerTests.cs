using System;
using Gym.Rendering;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Rendering {
    /// <summary>
    ///     The contract of <c>Gym.Rendering</c>'s pygame-compatible surface and primitives, checked without pygame: pixel
    ///     addressing, fills, blits, flips, <c>surfarray</c> layouts, SDL_gfx's inclusive lines and blending, the quarter-turn
    ///     snapping of <see cref="Vector2.RotateRad"/>, and pygame's argument errors. Pixel-for-pixel equality with pygame
    ///     itself is the live suite's job (<c>tests/Gym.Tests.Parity</c>, <c>GfxDrawLiveParityTests</c>).
    /// </summary>
    [TestClass]
    public class RasterizerTests {
        /// <summary>A new surface is black; <c>fill</c> sets every pixel's RGB and ignores the alpha (the format has none).</summary>
        [TestMethod]
        public void Surface_StartsBlack_AndFillsIgnoringAlpha() {
            var surface = new Surface(3, 2);
            Assert.AreEqual(new Color(0, 0, 0), surface.GetAt(2, 1));
            surface.Fill(new Color(10, 20, 30, 40));
            Assert.AreEqual(new Color(10, 20, 30), surface.GetAt(0, 0));
            Assert.ThrowsExactly<IndexOutOfRangeException>(() => surface.GetAt(3, 0));
        }

        /// <summary><c>hline</c> and <c>vline</c> include both ends, in either order, clipped to the surface.</summary>
        [TestMethod]
        public void Lines_IncludeBothEnds_AndClip() {
            var surface = new Surface(10, 5);
            GfxDraw.HLine(surface, 7, 2, 1, new Color(255, 0, 0));
            for (int x = 0; x < 10; x++)
                Assert.AreEqual(x >= 2 && x <= 7 ? new Color(255, 0, 0) : new Color(0, 0, 0), surface.GetAt(x, 1), $"x = {x}");

            GfxDraw.VLine(surface, 9, -100, 100, new Color(0, 255, 0));
            for (int y = 0; y < 5; y++)
                Assert.AreEqual(new Color(0, 255, 0), surface.GetAt(9, y));
        }

        /// <summary>
        ///     A translucent colour blends each channel as SDL_gfx does, <c>d + floor((s - d) * a / 256)</c>: from white towards
        ///     black at alpha 128, 255 + floor(-255 * 128 / 256) = 127.
        /// </summary>
        [TestMethod]
        public void TranslucentColors_BlendWithFlooredWeights() {
            var surface = new Surface(4, 1);
            surface.Fill(new Color(255, 255, 255));
            GfxDraw.HLine(surface, 0, 3, 0, new Color(0, 100, 255, 128));
            Assert.AreEqual(new Color(127, 177, 255), surface.GetAt(1, 0));
        }

        /// <summary><c>filled_circle</c> of radius 0 is one pixel; a negative radius is SDL_gfx's -1, pygame's <c>pygame.error</c>.</summary>
        [TestMethod]
        public void FilledCircle_OfRadiusZero_IsOnePixel_AndNegativeRadiiFail() {
            var surface = new Surface(5, 5);
            GfxDraw.FilledCircle(surface, 2, 2, 0, new Color(1, 2, 3));
            Assert.AreEqual(new Color(1, 2, 3), surface.GetAt(2, 2));
            Assert.AreEqual(new Color(0, 0, 0), surface.GetAt(2, 1));
            Assert.ThrowsExactly<SdlError>(() => GfxDraw.FilledCircle(surface, 2, 2, -1, new Color(1, 2, 3)));
            Assert.ThrowsExactly<SdlError>(() => GfxDraw.AACircle(surface, 2, 2, -1, new Color(1, 2, 3)));
        }

        /// <summary>pygame's argument conversion: too few points, NaN, infinities, and values beyond a C long or Sint16, with pygame's messages.</summary>
        [TestMethod]
        public void Arguments_FailWithPygamesErrors() {
            var surface = new Surface(5, 5);
            var black = new Color(0, 0, 0);
            Assert.AreEqual("points must contain more than 2 points", Assert.ThrowsExactly<ValueError>(() => GfxDraw.AAPolygon(surface, new[] { (0.0, 0.0), (1.0, 1.0) }, black)).Message);
            Assert.AreEqual("cannot convert float NaN to integer", Assert.ThrowsExactly<ValueError>(() => GfxDraw.FilledPolygon(surface, new[] { (double.NaN, 0.0), (1.0, 1.0), (2.0, 3.0) }, black)).Message);
            Assert.AreEqual("cannot convert float infinity to integer", Assert.ThrowsExactly<OverflowException>(() => GfxDraw.AAPolygon(surface, new[] { (0.0, 0.0), (1.0, double.PositiveInfinity), (2.0, 3.0) }, black)).Message);
            Assert.AreEqual("Python int too large to convert to C long", Assert.ThrowsExactly<OverflowException>(() => GfxDraw.AAPolygon(surface, new[] { (0.0, 0.0), (1e10, 1.0), (2.0, 3.0) }, black)).Message);
            Assert.AreEqual("signed short integer is greater than maximum", Assert.ThrowsExactly<OverflowException>(() => GfxDraw.AACircle(surface, 40000, 0, 5, black)).Message);
            Assert.AreEqual("signed short integer is less than minimum", Assert.ThrowsExactly<OverflowException>(() => GfxDraw.HLine(surface, 0, -40000, 0, black)).Message);
            Assert.AreEqual("Python int too large to convert to C long", Assert.ThrowsExactly<OverflowException>(() => GfxDraw.VLine(surface, 1L << 40, 0, 1, black)).Message);
        }

        /// <summary>A polygon vertex beyond Sint16 wraps around rather than failing, as pygame truncates it unchecked.</summary>
        [TestMethod]
        public void PolygonVertices_WrapAroundSint16() {
            var wrapped = new Surface(8, 8);
            var plain = new Surface(8, 8);
            // 65536 + 2 truncates to 2, so both triangles are the same one.
            GfxDraw.FilledPolygon(wrapped, new[] { (65538.0, 0.0), (7.0, 0.0), (65538.0, 7.0) }, new Color(9, 9, 9));
            GfxDraw.FilledPolygon(plain, new[] { (2.0, 0.0), (7.0, 0.0), (2.0, 7.0) }, new Color(9, 9, 9));
            CollectionAssert.AreEqual(plain.ToRgbBytes(), wrapped.ToRgbBytes());
        }

        /// <summary><c>transform.flip</c> mirrors rows and/or columns into a new surface and leaves the source alone.</summary>
        [TestMethod]
        public void Flip_MirrorsIntoANewSurface() {
            var surface = new Surface(2, 2);
            GfxDraw.HLine(surface, 0, 0, 0, new Color(1, 1, 1));
            Surface vertical = Transform.Flip(surface, false, true);
            Surface both = Transform.Flip(surface, true, true);
            Assert.AreEqual(new Color(1, 1, 1), vertical.GetAt(0, 1));
            Assert.AreEqual(new Color(1, 1, 1), both.GetAt(1, 1));
            Assert.AreEqual(new Color(1, 1, 1), surface.GetAt(0, 0), "The source is unchanged.");
        }

        /// <summary><c>blit</c> copies an opaque surface, clipped to the destination.</summary>
        [TestMethod]
        public void Blit_CopiesClipped() {
            var source = new Surface(2, 2);
            source.Fill(new Color(5, 6, 7));
            var target = new Surface(3, 3);
            target.Blit(source, 2, -1);
            Assert.AreEqual(new Color(5, 6, 7), target.GetAt(2, 0));
            Assert.AreEqual(new Color(0, 0, 0), target.GetAt(2, 1));
            Assert.AreEqual(new Color(0, 0, 0), target.GetAt(1, 0));
        }

        /// <summary><c>surfarray.array3d</c> is <c>(width, height, 3)</c> in <c>[x][y]</c> order; the rgb_array frame is <c>(height, width, 3)</c>.</summary>
        [TestMethod]
        public void SurfArray_LaysPixelsOutLikePygame() {
            var surface = new Surface(3, 2);
            GfxDraw.HLine(surface, 2, 2, 0, new Color(10, 20, 30));
            using NDArray columns = SurfArray.Array3d(surface);
            using NDArray frame = SurfArray.ToRgbArray(surface);
            CollectionAssert.AreEqual(new long[] { 3, 2, 3 }, columns.shape);
            CollectionAssert.AreEqual(new long[] { 2, 3, 3 }, frame.shape);
            Assert.AreEqual((byte)20, columns.GetByte(2, 0, 1));
            Assert.AreEqual((byte)20, frame.GetByte(0, 2, 1));
        }

        /// <summary>
        ///     <see cref="Vector2.RotateRad"/> snaps angles within 1e-6 of a quarter turn to the exact quarter turn, and rotates
        ///     by sine and cosine otherwise.
        /// </summary>
        [TestMethod]
        public void RotateRad_SnapsNearQuarterTurns() {
            var v = new Vector2(3.0, 4.0);
            Assert.AreEqual(new Vector2(-4.0, 3.0), v.RotateRad(Math.PI / 2));
            Assert.AreEqual(new Vector2(-3.0, -4.0), v.RotateRad(-Math.PI));
            Assert.AreEqual(new Vector2(4.0, -3.0), v.RotateRad(-Math.PI / 2));
            Assert.AreEqual(v, v.RotateRad(5e-7), "Within 1e-6 of 0: no rotation at all.");
            Assert.AreEqual(v, v.RotateRad(-5e-7), "Within 1e-6 of 2π: no rotation at all.");
            Vector2 turned = v.RotateRad(2e-6);
            Assert.AreEqual(Math.Cos(2e-6) * 3.0 - Math.Sin(2e-6) * 4.0, turned.X);
            Assert.IsTrue(double.IsNaN(v.RotateRad(double.PositiveInfinity).X));
        }
    }
}
