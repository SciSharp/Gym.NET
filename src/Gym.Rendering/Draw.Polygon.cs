using System;
using System.Collections.Generic;
using NumSharp;

namespace Gym.Rendering {
    public static partial class Draw {
        /// <summary>
        ///     <c>pygame.draw.polygon(surface, color, points, width)</c>: filled when <paramref name="width"/> is 0, otherwise
        ///     the closed outline <see cref="Lines"/> draws (nothing for a negative width), clipped to the surface.
        /// </summary>
        /// <param name="surface">The surface, without per-pixel alpha.</param>
        /// <param name="color">The colour; its alpha is ignored.</param>
        /// <param name="points">The vertices, as floats (each converted as pygame converts it, <see cref="PyInt"/>).</param>
        /// <param name="width">0 fills; a positive width outlines (1 and 2 are modelled, see <see cref="Lines"/>); a negative width draws nothing.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="surface"/> or <paramref name="points"/> is <see langword="null"/>.</exception>
        /// <exception cref="NotSupportedException">Thrown for a surface with per-pixel alpha, or an outline wider than 1 (not modelled).</exception>
        /// <exception cref="ValueError">
        ///     Thrown, with pygame's messages, for fewer than 3 points when filling (<c>points argument must contain more than 2
        ///     points</c>), or fewer than 2 when outlining or not drawing (<see cref="Lines"/>'s check runs first there).
        /// </exception>
        /// <remarks>
        ///     <para>
        ///     Gym.NET's own implementation of pygame-ce 2.5.8's observable behaviour (decision D4: derived from live pygame's
        ///     output, none of its source read), exact on 7,500 random polygons of 4 to 8 vertices and 20,000 random triangles,
        ///     clipped, flat, self-intersecting and far off the surface included, and on 3,000 more with coordinates to ±3e6.
        ///     </para>
        ///     <para>
        ///     Every vertex is first truncated toward zero, so a polygon's fractions only matter through that. Polygons of 4 or
        ///     more vertices fill by scanline: on each row between the lowest and highest vertex rows, each non-horizontal edge
        ///     whose rows include it (its lower end excluded, except on the last row) crosses at its upper end's x plus
        ///     <c>(row - upper y) * dx / dy</c>, the product in wrapping 32-bit integers and the division in float32. The
        ///     crossings are rounded alternately down and up in the order the edges are met (edge <c>i - 1 → i</c> for <c>i</c>
        ///     from 0), so which side rounds which way depends on the starting vertex, then sorted and filled pairwise;
        ///     horizontal edges strictly between the top and bottom rows are drawn on their own, and a polygon one row high is
        ///     the span between its outermost vertices.
        ///     </para>
        ///     <para>
        ///     Three-point polygons take another path: the vertices sorted by row (stable), each row from the top vertex's to
        ///     the bottom vertex's filled between the long edge and the short edge above or below the middle vertex, each
        ///     end at its edge's start plus the truncated float32 product of the row offset and the edge's float32 slope
        ///     (<c>dx / (dy + 1e-17)</c> in double first, so a flat edge's slope stays finite). Float32 slopes put some ends one
        ///     pixel short of the exact intersection (a slope of 60/58 times 58 lands below 60). A flat triangle whose first two
        ///     sorted points coincide spans to its third.
        ///     </para>
        ///     <para>
        ///     Footgun, as in pygame: the rounding makes adjacent polygons that share an edge overlap or leave a gap on some
        ///     rows, and the same polygon listed from another vertex can fill differently by a pixel.
        ///     </para>
        /// </remarks>
        public static void Polygon(Surface surface, Color color, IReadOnlyList<(double X, double Y)> points, int width = 0) {
            RequireOpaque(surface);
            if (points is null)
                throw new ArgumentNullException(nameof(points));
            if (width != 0) {
                // pygame outlines (and refuses or skips) through draw.lines with the polygon closed.
                Lines(surface, color, closed: true, points, width);
                return;
            }

            if (points.Count < 3)
                throw new ValueError("points argument must contain more than 2 points");
            int count = points.Count;
            var xs = new int[count];
            var ys = new int[count];
            for (int i = 0; i < count; i++) {
                xs[i] = PyInt(points[i].X);
                ys[i] = PyInt(points[i].Y);
            }

            uint mapped = color.Mapped;
            if (count == 3)
                FillTriangle(surface, mapped, xs, ys);
            else
                FillPolygon(surface, mapped, xs, ys);
        }

        /// <summary>
        ///     <c>pygame.draw.circle(surface, color, center, radius)</c>, filled (width 0): the pixels of a circle of the
        ///     truncated radius centred on the corner <c>(cx - 0.5, cy - 0.5)</c> of the truncated centre's pixel, clipped.
        /// </summary>
        /// <param name="surface">The surface, without per-pixel alpha.</param>
        /// <param name="color">The colour; its alpha is ignored.</param>
        /// <param name="centerX">The centre's x (converted as pygame converts it, <see cref="PyInt"/>).</param>
        /// <param name="centerY">The centre's y, likewise.</param>
        /// <param name="radius">The radius, likewise; below 1 nothing is drawn.</param>
        /// <param name="width">0 fills; a negative width draws nothing; a ring (positive width) is not modelled.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="surface"/> is <see langword="null"/>.</exception>
        /// <exception cref="NotSupportedException">Thrown for a surface with per-pixel alpha, or a positive width (no Gymnasium env draws a ring).</exception>
        /// <remarks>
        ///     Gym.NET's own implementation of pygame-ce 2.5.8's observable behaviour (D4), exact on every radius from 0 to 300
        ///     and on 3,000 random circles across and beyond the surface's edges. The disc is a midpoint walk over one octant
        ///     (x from 0 and y from the radius, a decision term starting at <c>1 - r</c>); each step's <c>(x, y)</c> fills four
        ///     spans, rows <c>cy + y - 1</c> and <c>cy - y</c> over columns <c>cx - x .. cx + x - 1</c>, and rows <c>cy + x - 1</c>
        ///     and <c>cy - x</c> over columns <c>cx - y .. cx + y - 1</c>, so a radius-r disc is 2r pixels across. Fractions of
        ///     the centre and radius are dropped (2.9 draws a radius-2 disc), and a NaN or huge value becomes <c>int.MinValue</c>,
        ///     which draws nothing visible, as in pygame.
        /// </remarks>
        public static void Circle(Surface surface, Color color, double centerX, double centerY, double radius, int width = 0) {
            RequireOpaque(surface);
            if (width < 0)
                return;
            if (width > 0)
                throw new NotSupportedException($"draw.circle with width {width} (a ring) is not modelled: only filled circles (0) are.");
            int cx = PyInt(centerX), cy = PyInt(centerY), r = PyInt(radius);
            uint mapped = color.Mapped;
            unchecked {
                // The midpoint walk in 32-bit integers, wrapping as pygame's C ints do for absurd radii.
                int f = 1 - r, ddx = 0, ddy = -2 * r, x = 0, y = r;
                while (x < y) {
                    if (f >= 0) {
                        y--;
                        ddy += 2;
                        f += ddy;
                    }

                    x++;
                    ddx += 2;
                    f += ddx + 1;
                    HLine(surface, mapped, (long)cx - x, (long)cy + y - 1, (long)cx + x - 1);
                    HLine(surface, mapped, (long)cx - x, (long)cy - y, (long)cx + x - 1);
                    HLine(surface, mapped, (long)cx - y, (long)cy + x - 1, (long)cx + y - 1);
                    HLine(surface, mapped, (long)cx - y, (long)cy - x, (long)cx + y - 1);
                }
            }
        }

        /// <summary>
        ///     <c>pygame.draw.lines(surface, color, closed, points, width)</c> at width 1 or 2: <c>draw.line</c> between each pair
        ///     of consecutive points, and from the last point back to the first when <paramref name="closed"/> and there are
        ///     more than two points; at width 2 each segment also gets a copy one pixel over (CarRacing's skid marks).
        /// </summary>
        /// <param name="surface">The surface, without per-pixel alpha.</param>
        /// <param name="color">The colour; its alpha is ignored.</param>
        /// <param name="closed">Whether the last point joins the first (ignored for two points, as in pygame).</param>
        /// <param name="points">The points, as floats (converted as pygame converts them, <see cref="PyInt"/>).</param>
        /// <param name="width">1 or 2 draws; below 1 draws nothing; above 2 is not modelled yet.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="surface"/> or <paramref name="points"/> is <see langword="null"/>.</exception>
        /// <exception cref="NotSupportedException">Thrown for a surface with per-pixel alpha, or a width above 2.</exception>
        /// <exception cref="ValueError">Thrown for fewer than 2 points, with pygame's message, whatever the width.</exception>
        /// <remarks>
        ///     Gym.NET's own implementation of pygame-ce 2.5.8's observable behaviour (D4): pygame's polyline at width 1 is the
        ///     union of its segments' <c>draw.line</c>s (exact on 600 random open polylines and 600 closed polygon outlines), and
        ///     a closed polyline of two points draws its one segment only (a line and its reverse can differ by a pixel, so
        ///     drawing the return segment would show). Each segment is the line <see cref="Line(Surface, Color, long, long, long, long)"/>
        ///     draws, stepped only across the surface, so a far-off endpoint (a NaN becomes <c>int.MinValue</c>) costs nothing.
        ///     At width 2 a segment is its width-1 line plus the same line moved one pixel right when it is steep
        ///     (<c>|dx| &lt;= |dy|</c> of the truncated ends) or one pixel down otherwise: exact on 3,000 random segments, 600
        ///     open polylines of float points, 600 closed ones, 600 closed two-point ones and 600 width-2 polygon outlines.
        /// </remarks>
        public static void Lines(Surface surface, Color color, bool closed, IReadOnlyList<(double X, double Y)> points, int width = 1) {
            RequireOpaque(surface);
            if (points is null)
                throw new ArgumentNullException(nameof(points));
            if (points.Count < 2)
                throw new ValueError("points argument must contain 2 or more points");
            if (width < 1)
                return;
            if (width > 2)
                throw new NotSupportedException($"draw.lines with width {width} is not modelled: only widths 1 and 2 are.");
            uint mapped = color.Mapped;
            int count = points.Count;
            for (int i = 1; i < count; i++)
                Segment(surface, mapped, PyInt(points[i - 1].X), PyInt(points[i - 1].Y), PyInt(points[i].X), PyInt(points[i].Y), width);
            if (closed && count > 2)
                Segment(surface, mapped, PyInt(points[count - 1].X), PyInt(points[count - 1].Y), PyInt(points[0].X), PyInt(points[0].Y), width);
        }

        /// <summary>
        ///     One polyline segment at width 1 or 2: the line, and at width 2 the line again one pixel right (steep) or down
        ///     (shallow), as measured on pygame (see <see cref="Lines"/>).
        /// </summary>
        /// <param name="surface">The surface.</param>
        /// <param name="mapped">The pixel word.</param>
        /// <param name="x0">The start column.</param>
        /// <param name="y0">The start row.</param>
        /// <param name="x1">The end column.</param>
        /// <param name="y1">The end row.</param>
        /// <param name="width">1 or 2.</param>
        private static void Segment(Surface surface, uint mapped, int x0, int y0, int x1, int y1, int width) {
            ClippedLine(surface, mapped, x0, y0, x1, y1);
            if (width != 2)
                return;
            // The differences in long arithmetic: int.MinValue ends (NaN) would overflow C's int subtraction here.
            bool steep = Math.Abs((long)x0 - x1) <= Math.Abs((long)y0 - y1);
            if (steep)
                ClippedLine(surface, mapped, unchecked(x0 + 1), y0, unchecked(x1 + 1), y1);
            else
                ClippedLine(surface, mapped, x0, unchecked(y0 + 1), x1, unchecked(y1 + 1));
        }

        /// <summary>
        ///     A Python float as pygame's C code turns it into an <c>int</c>: truncated toward zero, and NaN, the infinities and
        ///     anything outside the 32-bit range become <c>int.MinValue</c> (x86's <c>cvttsd2si</c>, which C's cast compiles to).
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>The integer.</returns>
        /// <remarks>
        ///     Footgun: .NET 9 and later saturate an out-of-range cast (<c>int.MaxValue</c> for 1e10) where .NET 8 on x64 gives
        ///     <c>int.MinValue</c>, so the conversion is spelled out rather than left to <c>(int)</c>.
        /// </remarks>
        internal static int PyInt(double value) =>
            double.IsNaN(value) || value >= 2147483648.0 || value < -2147483648.0 ? int.MinValue : (int)value;

        /// <summary>A float32 as C's cast to <c>int</c> turns it (x86's <c>cvttss2si</c>), as <see cref="PyInt"/> does for doubles.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The integer: truncated toward zero, or <c>int.MinValue</c> for NaN and out-of-range values.</returns>
        private static int CInt(float value) =>
            float.IsNaN(value) || value >= 2147483648f || value < -2147483648f ? int.MinValue : (int)value;

        /// <summary>
        ///     The scanline fill of polygons with 4 or more vertices (see <see cref="Polygon"/>'s remarks), over the
        ///     surface's rows only: rows outside it draw nothing, so skipping them changes no pixel.
        /// </summary>
        /// <param name="surface">The surface.</param>
        /// <param name="mapped">The pixel word.</param>
        /// <param name="xs">The truncated x of each vertex.</param>
        /// <param name="ys">The truncated y of each vertex.</param>
        private static void FillPolygon(Surface surface, uint mapped, int[] xs, int[] ys) {
            int count = xs.Length;
            int miny = ys[0], maxy = ys[0];
            for (int i = 1; i < count; i++) {
                miny = Math.Min(miny, ys[i]);
                maxy = Math.Max(maxy, ys[i]);
            }

            if (miny == maxy) {
                // One row high: the span between the outermost vertices.
                int minx = xs[0], maxx = xs[0];
                for (int i = 1; i < count; i++) {
                    minx = Math.Min(minx, xs[i]);
                    maxx = Math.Max(maxx, xs[i]);
                }

                HLine(surface, mapped, minx, miny, maxx);
                return;
            }

            var crossings = new int[count];
            int first = Math.Max(miny, 0), last = Math.Min(maxy, surface.Height - 1);
            for (int y = first; y <= last; y++) {
                int found = 0;
                for (int i = 0; i < count; i++) {
                    int previous = i == 0 ? count - 1 : i - 1;
                    int y1 = ys[previous], y2 = ys[i], x1, x2;
                    if (y1 < y2) {
                        x1 = xs[previous];
                        x2 = xs[i];
                    } else if (y1 > y2) {
                        y2 = ys[previous];
                        y1 = ys[i];
                        x2 = xs[previous];
                        x1 = xs[i];
                    } else {
                        // Horizontal edges are drawn by the pass after the scan.
                        continue;
                    }

                    if ((y >= y1 && y < y2) || (y == maxy && y2 == maxy)) {
                        unchecked {
                            // The product wraps as C's int does and divides in float32; the crossings round down and up in turn.
                            float offset = (float)((y - y1) * (x2 - x1)) / (float)(y2 - y1);
                            offset = found % 2 == 0 ? MathF.Floor(offset) : MathF.Ceiling(offset);
                            crossings[found++] = CInt(offset) + x1;
                        }
                    }
                }

                Array.Sort(crossings, 0, found);
                for (int k = 0; k + 1 < found; k += 2)
                    HLine(surface, mapped, crossings[k], y, crossings[k + 1]);
            }

            // Horizontal edges strictly between the top and bottom rows, which the scan's crossings never cover.
            for (int i = 0; i < count; i++) {
                int previous = i == 0 ? count - 1 : i - 1;
                int y = ys[i];
                if (miny < y && ys[previous] == y && y < maxy)
                    HLine(surface, mapped, xs[i], y, xs[previous]);
            }
        }

        /// <summary>
        ///     The three-point fill (see <see cref="Polygon"/>'s remarks): sorted vertices, float32 slopes, truncated ends,
        ///     over the surface's rows only.
        /// </summary>
        /// <param name="surface">The surface.</param>
        /// <param name="mapped">The pixel word.</param>
        /// <param name="xs">The truncated x of the three vertices.</param>
        /// <param name="ys">The truncated y of the three vertices.</param>
        private static void FillTriangle(Surface surface, uint mapped, int[] xs, int[] ys) {
            int x0 = xs[0], y0 = ys[0], x1 = xs[1], y1 = ys[1], x2 = xs[2], y2 = ys[2];
            // A stable sort by row: compare-and-swap with strict comparisons keeps tied vertices in their given order.
            if (y1 < y0)
                (x0, y0, x1, y1) = (x1, y1, x0, y0);
            if (y2 < y1) {
                (x1, y1, x2, y2) = (x2, y2, x1, y1);
                if (y1 < y0)
                    (x0, y0, x1, y1) = (x1, y1, x0, y0);
            }

            // A flat triangle whose first two points coincide spans to the third one.
            if (y0 == y1 && y1 == y2 && x0 == x1 && x1 != x2)
                (x1, x2) = (x2, x1);
            unchecked {
                float d1 = Slope(x2 - x0, y2 - y0), d2 = Slope(x1 - x0, y1 - y0), d3 = Slope(x2 - x1, y2 - y1);
                int first = Math.Max(y0, 0), last = Math.Min(y2, surface.Height - 1);
                for (int y = first; y <= last; y++) {
                    int a = x0 + CInt((y - y0) * d1);
                    int b = y < y1 ? x0 + CInt((y - y0) * d2) : x1 + CInt((y - y1) * d3);
                    HLine(surface, mapped, a, y, b);
                }
            }

            // The slope in double with a tiny epsilon (a flat edge stays finite), kept as float32.
            static float Slope(int dx, int dy) => (float)(dx / (dy + 1e-17));
        }

        /// <summary>A horizontal span between two columns, either order, both ends included, clipped to the surface.</summary>
        /// <param name="surface">The surface.</param>
        /// <param name="mapped">The pixel word.</param>
        /// <param name="xa">One end's column.</param>
        /// <param name="y">The row.</param>
        /// <param name="xb">The other end's column.</param>
        private static void HLine(Surface surface, uint mapped, long xa, long y, long xb) {
            if (y < 0 || y >= surface.Height)
                return;
            long left = Math.Max(Math.Min(xa, xb), 0), right = Math.Min(Math.Max(xa, xb), surface.Width - 1);
            if (left <= right)
                Array.Fill(surface.Pixels, mapped, (int)(y * surface.Width + left), (int)(right - left + 1));
        }

        /// <summary>
        ///     The line <see cref="Line(Surface, Color, long, long, long, long)"/> draws, pixel for pixel, but stepped only
        ///     across the surface: the error term at the first visible step comes from its closed form, so endpoints billions
        ///     of pixels away cost no more than the surface's width or height.
        /// </summary>
        /// <param name="surface">The surface.</param>
        /// <param name="mapped">The pixel word.</param>
        /// <param name="x0">The start column.</param>
        /// <param name="y0">The start row.</param>
        /// <param name="x1">The end column.</param>
        /// <param name="y1">The end row.</param>
        /// <remarks>
        ///     <see cref="Line(Surface, Color, long, long, long, long)"/> steps the longer axis from the start with an error term
        ///     starting at half its extent <c>D</c> (rounded down); each step subtracts the shorter extent <c>d</c> and, when the
        ///     term drops below zero, steps the other axis and adds <c>D</c>. The term stays in <c>[0, D)</c>, so after
        ///     <c>i</c> steps the other axis has moved <c>max(0, ceil((i * d - D / 2) / D))</c> times, which gives the state at
        ///     any step directly (128-bit, since <c>i * d</c> can exceed 64 bits).
        /// </remarks>
        private static void ClippedLine(Surface surface, uint mapped, long x0, long y0, long x1, long y1) {
            long dx = Math.Abs(x1 - x0), dy = Math.Abs(y1 - y0);
            long sx = x1 >= x0 ? 1 : -1, sy = y1 >= y0 ? 1 : -1;
            if (dx >= dy)
                StepClipped(surface, mapped, x0, y0, sx, sy, dx, dy, majorIsX: true);
            else
                StepClipped(surface, mapped, y0, x0, sy, sx, dy, dx, majorIsX: false);
        }

        /// <summary>Steps a line along its major axis over the visible steps only (see <see cref="ClippedLine"/>).</summary>
        /// <param name="surface">The surface.</param>
        /// <param name="mapped">The pixel word.</param>
        /// <param name="major0">The start on the major axis.</param>
        /// <param name="minor0">The start on the minor axis.</param>
        /// <param name="majorStep">The major axis's direction (+1 or -1).</param>
        /// <param name="minorStep">The minor axis's direction (+1 or -1).</param>
        /// <param name="majorExtent">The major extent <c>D</c> (at least the minor one).</param>
        /// <param name="minorExtent">The minor extent <c>d</c>.</param>
        /// <param name="majorIsX">Whether the major axis is x.</param>
        private static void StepClipped(Surface surface, uint mapped, long major0, long minor0, long majorStep, long minorStep, long majorExtent, long minorExtent, bool majorIsX) {
            long majorLimit = majorIsX ? surface.Width : surface.Height;
            // The steps whose major coordinate lies on the surface.
            long firstStep = majorStep > 0 ? Math.Max(0, -major0) : Math.Max(0, major0 - (majorLimit - 1));
            long lastStep = majorStep > 0 ? Math.Min(majorExtent, majorLimit - 1 - major0) : Math.Min(majorExtent, major0);
            if (firstStep > lastStep)
                return;
            long error = majorExtent / 2;
            long minorMoves = 0;
            if (firstStep > 0 && majorExtent > 0) {
                Int128 behind = (Int128)firstStep * minorExtent - error;
                if (behind > 0)
                    minorMoves = (long)((behind + majorExtent - 1) / majorExtent);
                error = (long)(error - (Int128)firstStep * minorExtent + (Int128)minorMoves * majorExtent);
            }

            long major = major0 + firstStep * majorStep, minor = minor0 + minorMoves * minorStep;
            for (long step = firstStep; step <= lastStep; step++) {
                if (majorIsX)
                    Put(surface, major, minor, mapped);
                else
                    Put(surface, minor, major, mapped);
                error -= minorExtent;
                if (error < 0) {
                    minor += minorStep;
                    error += majorExtent;
                }

                major += majorStep;
            }
        }
    }
}
