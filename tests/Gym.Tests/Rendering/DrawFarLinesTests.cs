#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Rendering;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests.Rendering {
    /// <summary>
    ///     The lines of <see cref="Draw.Line(Surface, Color, double, double, double, double)"/>, <see cref="Draw.Lines"/> and
    ///     polygon outlines whose endpoints lie far away: pygame converts a NaN, infinite or beyond-32-bit coordinate to
    ///     <c>INT_MIN</c> and walks the line's two billion steps to the surface (minutes per line, so the parity suite cannot
    ///     run them live), where Gym.NET starts at the first visible step from the error term's closed form. The pixels pinned
    ///     here are pygame-ce 2.5.8's own, recorded once; the random corpora with million-pixel endpoints are the parity
    ///     suite's (<c>PygameDrawLiveParityTests</c>).
    /// </summary>
    [TestClass]
    public class DrawFarLinesTests {
        /// <summary>The surface's width, the parity corpora's.</summary>
        private const int Width = 64;

        /// <summary>The surface's height.</summary>
        private const int Height = 48;

        /// <summary>
        ///     Lines to a NaN, an infinite and a 5e9 coordinate, a polyline and a polygon outline through a NaN vertex: exactly
        ///     the pixels pygame lit (recorded from <c>draw.line</c>, <c>draw.lines</c> and <c>draw.polygon(width=1)</c>).
        /// </summary>
        [TestMethod]
        public void FarEndpoints_LightPygamesRecordedPixels() {
            AssertLit("line((nan, 3), (10, 10))", s => Draw.Line(s, White, double.NaN, 3, 10, 10), Row(10, 0, 10));
            AssertLit("line((3, 3), (inf, 10))", s => Draw.Line(s, White, 3, 3, double.PositiveInfinity, 10), Row(3, 0, 3));
            AssertLit("line((3, 3), (5e9, 10))", s => Draw.Line(s, White, 3, 3, 5e9, 10), Row(3, 0, 3));
            AssertLit("line((20, -inf), (30, 40))", s => Draw.Line(s, White, 20, double.NegativeInfinity, 30, 40),
                Enumerable.Range(0, 41).Select(y => (30, y)));
            (double X, double Y)[] throughNaN = { (3, 3), (double.NaN, 10), (20, 20) };
            AssertLit("lines(open, [(3, 3), (nan, 10), (20, 20)])", s => Draw.Lines(s, White, false, throughNaN), Row(3, 0, 3).Concat(Row(20, 0, 20)));
            AssertLit("polygon([(3, 3), (nan, 10), (20, 20)], width=1)", s => Draw.Polygon(s, White, throughNaN, 1),
                Row(3, 0, 3).Concat(Row(20, 0, 20)).Concat(Enumerable.Range(4, 16).Select(v => (v, v))));
        }

        /// <summary>
        ///     The clipped walk (<see cref="Draw.Line(Surface, Color, double, double, double, double)"/>, which starts at the
        ///     first visible step) lights exactly the pixels of the full walk (<see cref="Draw.Line(Surface, Color, long, long, long, long)"/>,
        ///     which steps from the start) on 3,000 random lines, both directions, with endpoints up to 60,000 pixels away.
        /// </summary>
        [TestMethod]
        public void ClippedWalk_EqualsTheFullWalk() {
            var rng = new Random(20261111);
            for (int i = 0; i < 3000; i++) {
                int reach = i % 3 == 0 ? 60000 : i % 3 == 1 ? 600 : 30;
                long x0 = rng.Next(-reach, Width + reach), y0 = rng.Next(-reach, Height + reach);
                long x1 = rng.Next(-reach, Width + reach), y1 = rng.Next(-reach, Height + reach);
                var full = new Surface(Width, Height);
                Draw.Line(full, White, x0, y0, x1, y1);
                var clipped = new Surface(Width, Height);
                Draw.Line(clipped, White, (double)x0, y0, x1, y1);
                CollectionAssert.AreEqual(full.ToRgbBytes(), clipped.ToRgbBytes(), $"line(({x0}, {y0}), ({x1}, {y1}))");
            }
        }

        /// <summary>The colour every case draws in, over the surface's black.</summary>
        private static Color White => new(255, 255, 255);

        /// <summary>The pixels of one row from one column to another, both included.</summary>
        /// <param name="y">The row.</param>
        /// <param name="from">The first column.</param>
        /// <param name="to">The last column.</param>
        /// <returns>The pixels as (column, row).</returns>
        private static IEnumerable<(int X, int Y)> Row(int y, int from, int to) => Enumerable.Range(from, to - from + 1).Select(x => (x, y));

        /// <summary>Draws on a fresh black surface and requires exactly the given pixels to be lit.</summary>
        /// <param name="what">The call, for the message.</param>
        /// <param name="draw">The drawing.</param>
        /// <param name="expected">The lit pixels as (column, row).</param>
        private static void AssertLit(string what, Action<Surface> draw, IEnumerable<(int X, int Y)> expected) {
            var surface = new Surface(Width, Height);
            draw(surface);
            var lit = new List<(int X, int Y)>();
            for (int y = 0; y < Height; y++) {
                for (int x = 0; x < Width; x++) {
                    if (surface.GetAt(x, y) != new Color(0, 0, 0))
                        lit.Add((x, y));
                }
            }

            CollectionAssert.AreEquivalent(expected.ToList(), lit, what);
        }
    }
}
