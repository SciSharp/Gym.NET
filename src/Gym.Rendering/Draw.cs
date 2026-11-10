using System;
using NumSharp;

namespace Gym.Rendering {
    /// <summary>
    ///     The <c>pygame.draw</c> functions Gymnasium's envs call: <c>line</c> (Acrobot, BipedalWalker's lidar), <see cref="Rect"/>
    ///     (FrozenLake), <see cref="Aalines"/> and <see cref="Aaline"/> (MountainCar's hill, LunarLander, BipedalWalker), and
    ///     for the box2d envs <see cref="Polygon"/>, <see cref="Circle"/> and <see cref="Lines"/>, on surfaces without per-pixel
    ///     alpha.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <c>line</c>, <see cref="Rect"/>, <see cref="Polygon"/>, <see cref="Circle"/> and <see cref="Lines"/> are Gym.NET's
    ///     own implementations of pygame's behaviour (decision D4: derived from live pygame-ce 2.5.8's output, none of its
    ///     source read), pixel-exact on thousands of random shapes each, clipping included. Float coordinates convert as
    ///     pygame's C code converts them (<see cref="PyInt"/>).
    ///     </para>
    ///     <para>
    ///     <see cref="Aaline"/> and <see cref="Aalines"/> are the exception (decision DR-37): D4's black-box model of them did
    ///     not converge where segments join, so they are translated from pygame-ce 2.5.8's <c>src_c/draw.c</c>, statement for
    ///     statement in float32, and MountainCar's frames match pygame's pixel for pixel. That translation lives in its own
    ///     file, <c>Draw.Aaline.cs</c>, under its source's license (LGPL-2.0-or-later, see THIRD-PARTY-NOTICES.md), which makes
    ///     the Gym.NET.Rendering package <c>Apache-2.0 AND LGPL-2.0-or-later</c>.
    ///     </para>
    ///     <para>
    ///     The colour is written as the surface maps it: its alpha is dropped on a surface without per-pixel alpha (no
    ///     blending, as pygame's <c>draw</c> functions on such a surface), except where anti-aliasing weights a pixel.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/pygame-community/pygame-ce/blob/2.5.8/src_c/draw.c">src_c/draw.c</see> (pygame-ce 2.5.8).</para>
    /// </remarks>
    public static partial class Draw {
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
        ///     <c>pygame.draw.line(surface, color, start_pos, end_pos)</c> with float endpoints (as Acrobot and the box2d envs
        ///     pass them): each coordinate converted as pygame's C code converts it (<see cref="PyInt"/>: truncated toward zero),
        ///     then drawn as <see cref="Line(Surface, Color, long, long, long, long)"/> draws it.
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
        ///     A NaN, infinite or out-of-range coordinate becomes <c>int.MinValue</c>, as in pygame, which then draws the visible
        ///     part of a line to a point two billion pixels away (measured: <c>line((nan, 3), (10, 10))</c> lights row 10 from
        ///     column 0 to 10). Only the surface's steps are walked, so such a line costs no more than any other.
        /// </remarks>
        public static void Line(Surface surface, Color color, double x0, double y0, double x1, double y1) {
            RequireOpaque(surface);
            ClippedLine(surface, color.Mapped, PyInt(x0), PyInt(y0), PyInt(x1), PyInt(y1));
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

        /// <summary>Refuses a coordinate with no position on a surface.</summary>
        /// <param name="value">The coordinate.</param>
        /// <exception cref="ValueError">Thrown for NaN or an infinity.</exception>
        private static void RequireFinite(double value) {
            if (!double.IsFinite(value))
                throw new ValueError($"invalid coordinate {value}");
        }
    }
}
