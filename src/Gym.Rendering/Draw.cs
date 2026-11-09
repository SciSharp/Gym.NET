using System;
using System.Collections.Generic;
using NumSharp;

namespace Gym.Rendering {
    /// <summary>
    ///     The <c>pygame.draw</c> functions Gymnasium's envs call: <c>line</c> (Acrobot), <see cref="Rect"/>
    ///     (FrozenLake) and <see cref="Aalines"/> (MountainCar's hill), on surfaces without per-pixel alpha.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     These are Gym.NET's own implementations of pygame's behaviour (decision D4: no pygame code is used), each derived
    ///     from live pygame-ce 2.5.8 and proven against it on random corpora in the parity suite. <c>line</c> and
    ///     <see cref="Rect"/> are pixel-exact (thousands of random lines and rectangles, clipping included).
    ///     <see cref="Aaline"/> and <see cref="Aalines"/> are not yet: the single-segment model reproduces all but about one
    ///     drawn pixel in ten thousand (each off by one), and the rule pygame applies where segments join is not settled, so
    ///     MountainCar's frames differ near the hill's vertices. That gap is open in the parity suite and was reported to the
    ///     maintainer, as D4 requires for a primitive that does not converge.
    ///     </para>
    ///     <para>
    ///     The colour is written as the surface maps it: its alpha is dropped on a surface without per-pixel alpha (no
    ///     blending, as pygame's <c>draw</c> functions on such a surface), except where anti-aliasing weights a pixel.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/pygame-community/pygame-ce/blob/2.5.8/src_c/draw.c">src_c/draw.c</see> (pygame-ce 2.5.8; behaviour only).</para>
    /// </remarks>
    public static class Draw {
        /// <summary>
        ///     <c>pygame.draw.line(surface, color, start_pos, end_pos)</c> with integer endpoints and width 1: every pixel of the
        ///     line from <c>(x0, y0)</c> to <c>(x1, y1)</c>, both ends included, clipped to the surface.
        /// </summary>
        /// <param name="surface">The surface, without per-pixel alpha.</param>
        /// <param name="color">The colour; its alpha is ignored.</param>
        /// <param name="x0">The start column.</param>
        /// <param name="y0">The start row.</param>
        /// <param name="x1">The end column.</param>
        /// <param name="y1">The end row.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="surface"/> is <see langword="null"/>.</exception>
        /// <exception cref="NotSupportedException">Thrown for a surface with per-pixel alpha (not modelled).</exception>
        /// <remarks>
        ///     The line steps along its longer axis from the start; an error term starting at half the longer extent steps the
        ///     other axis when it drops below zero. Ties therefore fall toward the start, so a line and its reverse can differ
        ///     by a pixel, as pygame's do (<c>(0, 0)-(4, 2)</c> lights <c>(1, 0)</c>, <c>(4, 2)-(0, 0)</c> lights <c>(1, 1)</c>).
        ///     Off-surface pixels are dropped without moving the others (exact on 5,000 random lines, endpoints to ±5,000).
        /// </remarks>
        public static void Line(Surface surface, Color color, long x0, long y0, long x1, long y1) {
            RequireOpaque(surface);
            uint mapped = color.Mapped;
            long dx = Math.Abs(x1 - x0), dy = Math.Abs(y1 - y0);
            long sx = x1 >= x0 ? 1 : -1, sy = y1 >= y0 ? 1 : -1;
            long x = x0, y = y0;
            if (dx >= dy) {
                long error = dx / 2;
                for (long i = 0; i <= dx; i++) {
                    Put(surface, x, y, mapped);
                    error -= dy;
                    if (error < 0) {
                        y += sy;
                        error += dx;
                    }

                    x += sx;
                }
            } else {
                long error = dy / 2;
                for (long i = 0; i <= dy; i++) {
                    Put(surface, x, y, mapped);
                    error -= dx;
                    if (error < 0) {
                        x += sx;
                        error += dy;
                    }

                    y += sy;
                }
            }
        }

        /// <summary>
        ///     <c>pygame.draw.line(surface, color, start_pos, end_pos)</c> with float endpoints (as Acrobot passes them): each
        ///     coordinate is truncated toward zero, as pygame does, then drawn as <see cref="Line(Surface, Color, long, long, long, long)"/>.
        /// </summary>
        /// <param name="surface">The surface, without per-pixel alpha.</param>
        /// <param name="color">The colour; its alpha is ignored.</param>
        /// <param name="x0">The start column.</param>
        /// <param name="y0">The start row.</param>
        /// <param name="x1">The end column.</param>
        /// <param name="y1">The end row.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="surface"/> is <see langword="null"/>.</exception>
        /// <exception cref="NotSupportedException">Thrown for a surface with per-pixel alpha (not modelled).</exception>
        /// <exception cref="ValueError">Thrown for a coordinate that is NaN or infinite, which has no integer to truncate to.</exception>
        public static void Line(Surface surface, Color color, double x0, double y0, double x1, double y1) {
            Line(surface, color, Truncate(x0), Truncate(y0), Truncate(x1), Truncate(y1));
        }

        /// <summary>
        ///     <c>pygame.draw.rect(surface, color, (x, y, width, height), line_width)</c>: filled when
        ///     <paramref name="lineWidth"/> is 0, a one-pixel outline when it is 1, nothing when it is negative.
        /// </summary>
        /// <param name="surface">The surface, without per-pixel alpha.</param>
        /// <param name="color">The colour; its alpha is ignored.</param>
        /// <param name="x">The left column.</param>
        /// <param name="y">The top row.</param>
        /// <param name="width">The width; nothing is drawn unless it is positive.</param>
        /// <param name="height">The height; nothing is drawn unless it is positive.</param>
        /// <param name="lineWidth">0 fills, 1 outlines, a negative value draws nothing.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="surface"/> is <see langword="null"/>.</exception>
        /// <exception cref="NotSupportedException">Thrown for a surface with per-pixel alpha, or a line width above 1 (not modelled).</exception>
        /// <remarks>
        ///     An outline clipped by the surface edge keeps its edges where they are (an edge off the surface is simply not
        ///     drawn), except that a rectangle whose visible part is at most 2 pixels wide or tall is filled, visible part only:
        ///     pygame's outline of such a sliver covers it completely. Measured exact on 8,000 random rectangles.
        /// </remarks>
        public static void Rect(Surface surface, Color color, long x, long y, long width, long height, int lineWidth = 0) {
            RequireOpaque(surface);
            if (lineWidth > 1)
                throw new NotSupportedException($"draw.rect with line width {lineWidth} is not modelled: only 0 (filled) and 1 (outline) are.");
            if (lineWidth < 0 || width <= 0 || height <= 0)
                return;
            long left = Math.Max(x, 0), top = Math.Max(y, 0), right = Math.Min(x + width, surface.Width), bottom = Math.Min(y + height, surface.Height);
            if (left >= right || top >= bottom)
                return;
            uint mapped = color.Mapped;
            if (lineWidth == 0 || right - left <= 2 || bottom - top <= 2) {
                for (long row = top; row < bottom; row++)
                    Array.Fill(surface.Pixels, mapped, (int)(row * surface.Width + left), (int)(right - left));
                return;
            }

            // The original rectangle's four edges, each clipped to the surface.
            for (long column = left; column < right; column++) {
                Put(surface, column, y, mapped);
                Put(surface, column, y + height - 1, mapped);
            }

            for (long row = top; row < bottom; row++) {
                Put(surface, x, row, mapped);
                Put(surface, x + width - 1, row, mapped);
            }
        }

        /// <summary>
        ///     <c>pygame.draw.aaline(surface, color, start_pos, end_pos)</c>: an anti-aliased line between float endpoints,
        ///     each pixel blended with the colour by the line's coverage of it. Not yet pixel-exact with pygame (see the class
        ///     remarks).
        /// </summary>
        /// <param name="surface">The surface, without per-pixel alpha.</param>
        /// <param name="color">The colour; its alpha is ignored (coverage alone weights the blend).</param>
        /// <param name="x0">The start column; pixel centres are at whole coordinates.</param>
        /// <param name="y0">The start row.</param>
        /// <param name="x1">The end column.</param>
        /// <param name="y1">The end row.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="surface"/> is <see langword="null"/>.</exception>
        /// <exception cref="NotSupportedException">Thrown for a surface with per-pixel alpha (not modelled).</exception>
        /// <exception cref="ValueError">Thrown for a coordinate that is NaN or infinite.</exception>
        /// <remarks>
        ///     The model, measured against pygame-ce 2.5.8: coordinates in float32; the line steps along its longer axis in
        ///     increasing order; each column splits its coverage between the two pixels around the line, <c>1 - f</c> and
        ///     <c>f</c> for the fractional part <c>f</c> of the other coordinate, which is accumulated by repeated float32
        ///     addition of the slope; the first column <c>floor(x0)</c> is weighted <c>1 - frac(x0)</c> and the last
        ///     <c>floor(x1) + 1</c> by <c>frac(x1)</c>; a pixel becomes <c>trunc(b * colour + (1 - b) * old)</c>. A zero-length
        ///     line sets its rounded pixel.
        /// </remarks>
        public static void Aaline(Surface surface, Color color, double x0, double y0, double x1, double y1) {
            RequireOpaque(surface);
            AalineSegment(surface, color, x0, y0, x1, y1, skipStartColumn: false, skipEndColumn: false);
        }

        /// <summary>
        ///     <c>pygame.draw.aalines(surface, color, closed, points)</c>: anti-aliased segments through the points, closed
        ///     back to the first when <paramref name="closed"/>. Not yet pixel-exact with pygame near the vertices (see the
        ///     class remarks).
        /// </summary>
        /// <param name="surface">The surface, without per-pixel alpha.</param>
        /// <param name="color">The colour; its alpha is ignored.</param>
        /// <param name="closed">Whether a segment joins the last point back to the first.</param>
        /// <param name="points">At least 2 points, as <c>(x, y)</c> floats.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="surface"/> or <paramref name="points"/> is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">Thrown for fewer than 2 points, with pygame's message, or a NaN or infinite coordinate.</exception>
        /// <exception cref="NotSupportedException">Thrown for a surface with per-pixel alpha (not modelled).</exception>
        /// <remarks>
        ///     Each segment is drawn as <see cref="Aaline"/> draws it, except that a segment ending at a joint leaves out its
        ///     column at its larger major coordinate, so a joint is not blended twice: the closest rule found, not pygame's
        ///     exact one.
        /// </remarks>
        public static void Aalines(Surface surface, Color color, bool closed, IReadOnlyList<(double X, double Y)> points) {
            RequireOpaque(surface);
            if (points is null)
                throw new ArgumentNullException(nameof(points));
            if (points.Count < 2)
                throw new ValueError("points argument must contain 2 or more points");
            int segments = closed ? points.Count : points.Count - 1;
            for (int i = 0; i < segments; i++) {
                (double ax, double ay) = points[i];
                (double bx, double by) = points[(i + 1) % points.Count];
                bool joint = i < points.Count - 2 || closed;
                AalineSegment(surface, color, ax, ay, bx, by, skipStartColumn: false, skipEndColumn: joint);
            }
        }

        /// <summary>
        ///     One anti-aliased segment (the model in <see cref="Aaline"/>'s remarks), optionally without its column at the
        ///     smaller or the larger major coordinate (the columns are ordered after the segment is put in increasing order).
        /// </summary>
        /// <param name="surface">The surface, without per-pixel alpha.</param>
        /// <param name="color">The colour.</param>
        /// <param name="x0">The start column.</param>
        /// <param name="y0">The start row.</param>
        /// <param name="x1">The end column.</param>
        /// <param name="y1">The end row.</param>
        /// <param name="skipStartColumn">Leave out the column at the smaller major coordinate.</param>
        /// <param name="skipEndColumn">Leave out the column at the larger major coordinate.</param>
        /// <exception cref="ValueError">Thrown for a coordinate that is NaN or infinite.</exception>
        private static void AalineSegment(Surface surface, Color color, double x0, double y0, double x1, double y1, bool skipStartColumn, bool skipEndColumn) {
            RequireFinite(x0);
            RequireFinite(y0);
            RequireFinite(x1);
            RequireFinite(y1);
            float fx0 = (float)x0, fy0 = (float)y0, fx1 = (float)x1, fy1 = (float)y1;
            float dx = fx1 - fx0, dy = fy1 - fy0;
            if (MathF.Abs(dx) < 0.0001f && MathF.Abs(dy) < 0.0001f) {
                // A segment of no length is its rounded point, at full strength.
                Put(surface, (long)MathF.Floor(fx0 + 0.5f), (long)MathF.Floor(fy0 + 0.5f), color.Mapped);
                return;
            }

            bool steep = MathF.Abs(dx) < MathF.Abs(dy);
            if (steep) {
                (fx0, fy0) = (fy0, fx0);
                (fx1, fy1) = (fy1, fx1);
                (dx, dy) = (dy, dx);
            }

            if (fx0 > fx1) {
                (fx0, fx1) = (fx1, fx0);
                (fy0, fy1) = (fy1, fy0);
                dx = -dx;
                dy = -dy;
            }

            float slope = dy / dx;
            long first = (long)MathF.Floor(fx0), last = (long)MathF.Floor(fx1) + 1;
            float startY = fy0 + slope * ((float)first - fx0);
            if (!skipStartColumn)
                Column(surface, color, steep, first, startY, 1f - (fx0 - first));
            // The interior y is accumulated by float32 addition, step by step: its rounding is part of what is matched.
            float y = startY + slope;
            for (long column = first + 1; column < last; column++) {
                Column(surface, color, steep, column, y, 1f);
                y += slope;
            }

            if (!skipEndColumn)
                Column(surface, color, steep, last, fy1 + slope * ((float)last - fx1), fx1 - (last - 1));
        }

        /// <summary>One column of an anti-aliased segment: the two pixels around <paramref name="y"/>, weighted by <paramref name="weight"/>.</summary>
        /// <param name="surface">The surface.</param>
        /// <param name="color">The colour.</param>
        /// <param name="steep">Whether the major axis is y (coordinates are swapped back when plotting).</param>
        /// <param name="column">The major coordinate.</param>
        /// <param name="y">The minor coordinate of the line at that column.</param>
        /// <param name="weight">The column's coverage along the major axis (1 inside the segment, less at its ends).</param>
        private static void Column(Surface surface, Color color, bool steep, long column, float y, float weight) {
            float floor = MathF.Floor(y);
            float fraction = y - floor;
            long row = (long)floor;
            Blend(surface, color, steep ? row : column, steep ? column : row, weight * (1f - fraction));
            Blend(surface, color, steep ? row + 1 : column, steep ? column : row + 1, weight * fraction);
        }

        /// <summary>Blends one pixel with the colour by coverage <paramref name="brightness"/>: <c>trunc(b * c + (1 - b) * old)</c> per channel, in float32.</summary>
        /// <param name="surface">The surface.</param>
        /// <param name="color">The colour.</param>
        /// <param name="x">The column; off-surface pixels are skipped.</param>
        /// <param name="y">The row.</param>
        /// <param name="brightness">The coverage; nothing is drawn unless it is positive.</param>
        private static void Blend(Surface surface, Color color, long x, long y, float brightness) {
            if (brightness <= 0f || x < 0 || x >= surface.Width || y < 0 || y >= surface.Height)
                return;
            int index = (int)(y * surface.Width + x);
            uint old = surface.Pixels[index];
            float keep = 1f - brightness;
            uint r = (uint)(brightness * color.R + keep * ((old >> 16) & 0xFF));
            uint g = (uint)(brightness * color.G + keep * ((old >> 8) & 0xFF));
            uint b = (uint)(brightness * color.B + keep * (old & 0xFF));
            surface.Pixels[index] = (r << 16) | (g << 8) | b;
        }

        /// <summary>Writes one pixel when it lies on the surface.</summary>
        /// <param name="surface">The surface.</param>
        /// <param name="x">The column.</param>
        /// <param name="y">The row.</param>
        /// <param name="mapped">The pixel word.</param>
        private static void Put(Surface surface, long x, long y, uint mapped) {
            if (x >= 0 && x < surface.Width && y >= 0 && y < surface.Height)
                surface.Pixels[y * surface.Width + x] = mapped;
        }

        /// <summary>Refuses a missing surface or one with per-pixel alpha, which these primitives do not model.</summary>
        /// <param name="surface">The surface.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="surface"/> is <see langword="null"/>.</exception>
        /// <exception cref="NotSupportedException">Thrown for a surface with per-pixel alpha.</exception>
        private static void RequireOpaque(Surface surface) {
            if (surface is null)
                throw new ArgumentNullException(nameof(surface));
            if (surface.HasAlpha)
                throw new NotSupportedException("Drawing on a surface with per-pixel alpha is not modelled: no Gymnasium env does it.");
        }

        /// <summary>Python's <c>int()</c> of a float coordinate: truncation toward zero.</summary>
        /// <param name="value">The coordinate.</param>
        /// <returns>The integer.</returns>
        /// <exception cref="ValueError">Thrown for NaN or an infinity.</exception>
        private static long Truncate(double value) {
            RequireFinite(value);
            return (long)Math.Truncate(value);
        }

        /// <summary>Refuses a coordinate with no position on a surface.</summary>
        /// <param name="value">The coordinate.</param>
        /// <exception cref="ValueError">Thrown for NaN or an infinity.</exception>
        private static void RequireFinite(double value) {
            if (!double.IsFinite(value))
                throw new ValueError($"invalid coordinate {value}");
        }
    }
}
