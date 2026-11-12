#nullable enable
using System;
using System.Linq;
using Gym.Environments.Envs.MuJoCo.Rendering;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests.Envs.MuJoCo {
    /// <summary>
    ///     MuJoCo's text overlay as Gym.NET draws it (<see cref="MujocoOverlay"/>, DR-70): the embedded <c>normal150</c> font,
    ///     <c>draw_overlay</c>'s box sizes and grid positions, the half-black blend and the glyphs' levels.
    /// </summary>
    /// <remarks>
    ///     The expectations follow <c>render_gl2.c</c>'s arithmetic; the parity suite draws the same menus with MuJoCo's own
    ///     <c>mjr_overlay</c> and requires Gym.NET's pixels byte for byte (<c>MujocoWindowLiveParityTests</c>).
    /// </remarks>
    [TestClass]
    public class MujocoOverlayTests {
        /// <summary>The frame's level everywhere before an overlay is drawn.</summary>
        private const byte Grey = 200;

        /// <summary>
        ///     The embedded font is MuJoCo's <c>normal150</c>: 28 pixels high, and the widths its source records for the first
        ///     printable characters (space and <c>!</c> 6, <c>"</c> 9); a box is the widest line plus 13 pixels wide and 33 pixels
        ///     a line plus 38 high, a text ending in a newline counting one empty line more.
        /// </summary>
        [TestMethod]
        public void Font_IsMujocosNormal150() {
            Assert.AreEqual((5 + 6 + 8, 5 + 28), MujocoOverlay.Measure(" "));
            Assert.AreEqual((5 + 6 + 8, 5 + 28), MujocoOverlay.Measure("!"));
            Assert.AreEqual((5 + 9 + 8, 5 + 28), MujocoOverlay.Measure("\""));
            Assert.AreEqual((5 + 0 + 8, 5 + 28 + 33), MujocoOverlay.Measure("\n"));
            Assert.AreEqual((5 + 12 + 8, 5 + 28 + 2 * 33), MujocoOverlay.Measure("!!\n\"\n"));
            Assert.AreEqual(MujocoOverlay.Measure(""), MujocoOverlay.Measure("é\u0001"), "bytes without a glyph take no room");
        }

        /// <summary>
        ///     Only the first 500 bytes count (<c>mjMAXOVERLAY</c>): a 600-character line measures as 500 of them.
        /// </summary>
        [TestMethod]
        public void Text_IsCutAt500Bytes() {
            Assert.AreEqual(MujocoOverlay.Measure(new string('!', 500)), MujocoOverlay.Measure(new string('!', 600)));
            Assert.AreEqual(5 + 500 * 6 + 8, MujocoOverlay.Measure(new string('!', 600)).Width);
        }

        /// <summary>
        ///     One white column at the top left: its box's pixels are the frame blended half with black (level 200 becomes
        ///     (200·128 + 127) / 255 = 100), the glyph's set bits white, 3 pixels in from the box's left and with the first line's
        ///     top at the box's top less 5; the box sits 5 pixels from the left and 6 from the top; every sample of a pixel
        ///     takes the same values.
        /// </summary>
        [TestMethod]
        public void OneColumn_BlendsTheBoxAndDrawsTheGlyphsWhite() {
            const int w = 60, h = 90;
            GlTarget target = Uniform(w, h, samples: 4);
            MujocoOverlay.Draw(target, MujocoOverlay.TopLeft, 0, 0, w, h, "!\n", "");
            (int bw, int bh) = MujocoOverlay.Measure("!\n");
            int x0 = 5, y0 = h - 1 - 5 - bh;
            int text = 0;
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    byte level = Level(target, x, y, out bool samplesAgree);
                    Assert.IsTrue(samplesAgree, $"({x}, {y}): every sample alike");
                    bool inBox = x >= x0 && x < x0 + bw && y >= y0 && y < y0 + bh;
                    if (!inBox) {
                        Assert.AreEqual(Grey, level, $"({x}, {y}) outside the box");
                        continue;
                    }

                    if (level == 255) {
                        text++;
                        // the glyph's cell: 6 columns from x0 + 3, the first line's 28 rows below the box's top less 5
                        Assert.IsTrue(x >= x0 + 3 && x < x0 + 3 + 6 && y >= y0 + bh - 28 && y < y0 + bh, $"({x}, {y}): a glyph pixel outside its cell");
                    } else {
                        Assert.AreEqual((byte)100, level, $"({x}, {y}) inside the box");
                    }
                }
            }

            Assert.IsTrue(text > 10, $"the exclamation mark's pixels, got {text}");
        }

        /// <summary>
        ///     Two columns at the top left: the labels grey (level 178, what the GPU stores 0.7 as) and the values white, the
        ///     second box starting at the first's width less 2 plus 5, so the two boxes share two columns that are blended
        ///     twice (100 becomes 50).
        /// </summary>
        [TestMethod]
        public void TwoColumns_GreyLabelsWhiteValuesAndTwoSharedColumns() {
            const int w = 80, h = 60;
            GlTarget target = Uniform(w, h, samples: 1);
            MujocoOverlay.Draw(target, MujocoOverlay.TopLeft, 0, 0, w, h, "!", "!");
            (int bw, int bh) = MujocoOverlay.Measure("!");
            int y0 = h - 1 - 5 - bh, second = (bw - 2) + 5;
            Assert.AreEqual((byte)100, Level(target, 5, y0, out _), "the first box's corner");
            Assert.AreEqual((byte)50, Level(target, second, y0, out _), "a column of both boxes");
            Assert.AreEqual((byte)50, Level(target, second + 1, y0, out _));
            Assert.AreEqual((byte)100, Level(target, second + 2, y0, out _), "the second box alone");
            byte[] firstGlyph = Column(target, 5 + 3, 5 + 3 + 6, y0, y0 + bh);
            byte[] secondGlyph = Column(target, second + 3, second + 3 + 6, y0, y0 + bh);
            CollectionAssert.Contains(firstGlyph, (byte)178, "the label is grey");
            CollectionAssert.DoesNotContain(firstGlyph, (byte)255);
            CollectionAssert.Contains(secondGlyph, (byte)255, "the value is white");
        }

        /// <summary>
        ///     The grid positions place the box as <c>draw_overlay</c>'s viewports do: the bottom left 5 pixels from the left and
        ///     the bottom; the top right 5 from the right edge less one; the right-hand positions put the values first, so the
        ///     labels end up outermost; an empty viewport draws nothing; and a position outside <c>mjtGridPos</c> is refused.
        /// </summary>
        [TestMethod]
        public void GridPositions_PlaceTheBoxesAsDrawOverlay() {
            const int w = 70, h = 70;
            (int bw, int bh) = MujocoOverlay.Measure("!");
            GlTarget bottomLeft = Uniform(w, h, 1);
            MujocoOverlay.Draw(bottomLeft, MujocoOverlay.BottomLeft, 0, 0, w, h, "!", "");
            Assert.AreEqual(Grey, Level(bottomLeft, 4, 5, out _));
            Assert.AreEqual((byte)100, Level(bottomLeft, 5, 5, out _));
            Assert.AreEqual(Grey, Level(bottomLeft, 5, 4, out _));
            Assert.AreEqual((byte)100, Level(bottomLeft, 5 + bw - 1, 5 + bh - 1, out _));
            Assert.AreEqual(Grey, Level(bottomLeft, 5 + bw, 5 + bh, out _));

            GlTarget topRight = Uniform(w, h, 1);
            MujocoOverlay.Draw(topRight, MujocoOverlay.TopRight, 0, 0, w, h, "!", "");
            // the box's top row is 6 below the frame's top: draw_overlay's viewport starts at height - 1 - 5 - H
            int x0 = w - 1 - 5 - bw;
            Assert.AreEqual((byte)100, Level(topRight, x0, h - 7, out _));
            Assert.AreEqual(Grey, Level(topRight, x0 - 1, h - 7, out _));
            Assert.AreEqual(Grey, Level(topRight, x0 + bw, h - 7, out _));
            Assert.AreEqual(Grey, Level(topRight, x0, h - 6, out _));

            GlTarget untouched = Uniform(w, h, 1);
            MujocoOverlay.Draw(untouched, MujocoOverlay.TopLeft, 0, 0, 0, h, "!", "!");
            Assert.IsTrue(untouched.Color!.Where((_, i) => i % 4 != 3).All(b => b == Grey), "an empty viewport draws nothing");
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => MujocoOverlay.Draw(untouched, 8, 0, 0, w, h, "!", ""));
        }

        /// <summary>A target of one level everywhere, alpha 255.</summary>
        /// <param name="w">The width.</param>
        /// <param name="h">The height.</param>
        /// <param name="samples">Samples per pixel.</param>
        /// <returns>The target.</returns>
        private static GlTarget Uniform(int w, int h, int samples) {
            var target = new GlTarget(w, h, samples, color: true, stencil: false);
            for (int i = 0; i < target.Color!.Length; i++)
                target.Color[i] = i % 4 == 3 ? (byte)255 : Grey;
            return target;
        }

        /// <summary>A pixel's red level (the overlay is grey, so the three channels agree), from its first sample.</summary>
        /// <param name="target">The target.</param>
        /// <param name="x">The column.</param>
        /// <param name="y">The row from the bottom.</param>
        /// <param name="samplesAgree">Whether every sample and channel holds that level.</param>
        /// <returns>The level.</returns>
        private static byte Level(GlTarget target, int x, int y, out bool samplesAgree) {
            int ns = target.Samples, o = 4 * ((y * target.Width + x) * ns);
            byte level = target.Color![o];
            samplesAgree = true;
            for (int s = 0; s < ns; s++) {
                for (int c = 0; c < 3; c++)
                    samplesAgree &= target.Color[o + 4 * s + c] == level;
            }

            return level;
        }

        /// <summary>The levels of a rectangle of pixels.</summary>
        /// <param name="target">The target.</param>
        /// <param name="x0">The first column.</param>
        /// <param name="x1">The column past the last.</param>
        /// <param name="y0">The first row.</param>
        /// <param name="y1">The row past the last.</param>
        /// <returns>The levels, row by row.</returns>
        private static byte[] Column(GlTarget target, int x0, int x1, int y0, int y1) =>
            Enumerable.Range(y0, y1 - y0).SelectMany(y => Enumerable.Range(x0, x1 - x0).Select(x => Level(target, x, y, out _))).ToArray();
    }
}
