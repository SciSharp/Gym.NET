/*
    The anti-aliased line drawing in this file is translated to C# from pygame-ce 2.5.8's src_c/draw.c (the functions
    aaline, aalines, draw_aaline, get_antialiased_color and set_and_check_rect), under decision DR-37 of
    docs/plans/DESIGN_REVIEW.md. That source carries this notice:

    pygame-ce - Python Game Library
    Copyright (C) 2000-2001  Pete Shinners

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Library General Public License for more details.

    You should have received a copy of the GNU Library General Public
    License along with this library; if not, write to the Free
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    This file is therefore distributed under the GNU Library General Public License, version 2 or (at your option) any
    later version (SPDX: LGPL-2.0-or-later), while the rest of Gym.NET.Rendering is Apache-2.0; see THIRD-PARTY-NOTICES.md.

    Changes from the original, made on 2026-11-10 by the Gym.NET authors: translated from C to C#; reduced to width 1 on
    32-bit surfaces without per-pixel alpha, whose clip rectangle is the whole surface; the bounding box of the drawn
    area (drawn_area, pygame's return value) and the deprecated blend argument left out; NaN and infinite coordinates
    refused instead of converted.
*/
using System;
using System.Collections.Generic;
using NumSharp;

namespace Gym.Rendering {
    public static partial class Draw {
        /// <summary>
        ///     <c>pygame.draw.aaline(surface, color, start_pos, end_pos)</c>: an anti-aliased line between float endpoints, each
        ///     pixel blended with the colour by the line's coverage of it, pixel for pixel as pygame-ce 2.5.8 draws it.
        /// </summary>
        /// <param name="surface">The surface, without per-pixel alpha.</param>
        /// <param name="color">The colour; its alpha is dropped, as the surface maps it (coverage alone weights the blend).</param>
        /// <param name="x0">The start column; pixel centres are at whole coordinates.</param>
        /// <param name="y0">The start row.</param>
        /// <param name="x1">The end column.</param>
        /// <param name="y1">The end row.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="surface"/> is <see langword="null"/>.</exception>
        /// <exception cref="NotSupportedException">Thrown for a surface with per-pixel alpha (not modelled).</exception>
        /// <exception cref="ValueError">Thrown for a coordinate that is NaN or infinite, which pygame's C code would cast to an integer with undefined results.</exception>
        /// <remarks>
        ///     Translated from pygame-ce 2.5.8's <c>draw_aaline</c> (DR-37, LGPL-2.0-or-later, see this file's header): the
        ///     coordinates are float32, as pygame converts them, and every step is a float32 operation in pygame's order, which
        ///     C# reproduces bit for bit (x64 single precision, no fused multiply-adds). Width 1 only, which is all Gymnasium
        ///     draws. Reference: <see href="https://github.com/pygame-community/pygame-ce/blob/2.5.8/src_c/draw.c#L132">src_c/draw.c</see>.
        /// </remarks>
        public static void Aaline(Surface surface, Color color, double x0, double y0, double x1, double y1) {
            RequireOpaque(surface);
            RequireFinite(x0);
            RequireFinite(y0);
            RequireFinite(x1);
            RequireFinite(y1);
            DrawAaline(surface, color, (float)x0, (float)y0, (float)x1, (float)y1, disableFirstEndpoint: false, disableSecondEndpoint: false, extraPixel: false);
        }

        /// <summary>
        ///     <c>pygame.draw.aalines(surface, color, closed, points)</c>: anti-aliased segments through the points, closed back
        ///     to the first when <paramref name="closed"/>, pixel for pixel as pygame-ce 2.5.8 draws them, its joint rules and
        ///     quirks included.
        /// </summary>
        /// <param name="surface">The surface, without per-pixel alpha.</param>
        /// <param name="color">The colour; its alpha is dropped, as the surface maps it.</param>
        /// <param name="closed">Whether a segment joins the last point back to the first.</param>
        /// <param name="points">At least 2 points, as <c>(x, y)</c> floats.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="surface"/> or <paramref name="points"/> is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">Thrown for fewer than 2 points, with pygame's message, or a NaN or infinite coordinate.</exception>
        /// <exception cref="NotSupportedException">Thrown for a surface with per-pixel alpha (not modelled).</exception>
        /// <remarks>
        ///     <para>
        ///     Translated from pygame-ce 2.5.8's <c>aalines</c> (DR-37). A segment leaves out its endpoint columns where its end
        ///     point is not a whole pixel, unless the segment before and this one differ in steepness, which draws an extra
        ///     pixel there; an open polyline keeps its two outer endpoints. pygame's quirks are kept as they are: the first
        ///     segment's steepness test compares the third point with the first point's y (<c>pts[1]</c>, not <c>pts[3]</c>) and
        ///     uses <c>&gt;</c> where the others use <c>!=</c>; the last segment's first endpoint follows its end point's
        ///     wholeness; and with two points the one segment is drawn twice (as the first and as the last segment).
        ///     </para>
        ///     <para>
        ///     Footgun, with two points only: pygame keeps the x's and then the y's in one buffer, and its first test reads
        ///     <c>xlist[2]</c> and <c>ylist[2]</c> whatever the length, so with two points it compares the first point's y with
        ///     the float just past its buffer, memory it does not own. For a steep segment whose end point is not a whole pixel,
        ///     that float decides whether the end column is blended once more, so pygame's own drawing of such a polyline changes
        ///     from run to run. Gym.NET lays its buffer out the same way with one float more, 0: the null link pymalloc writes
        ///     into the free block after one it carves, which is what a fresh interpreter usually finds there (6 short fresh
        ///     runs out of 6 drew as Gym.NET does; a longer one differed at its 12th polyline, and a long-running interpreter
        ///     differed on up to 4 of 11 such polylines per run). The parity suite accepts either of pygame's two outcomes for
        ///     them and holds every other polyline to its pixels.
        ///     </para>
        ///     <para>Reference: <see href="https://github.com/pygame-community/pygame-ce/blob/2.5.8/src_c/draw.c#L303">src_c/draw.c</see>.</para>
        /// </remarks>
        public static void Aalines(Surface surface, Color color, bool closed, IReadOnlyList<(double X, double Y)> points) =>
            DrawAalines(surface, color, closed, points, pastTheBuffer: 0f);

        /// <summary>
        ///     <see cref="Aalines"/> with the float pygame reads past its buffer when given two points: the seam through which the
        ///     parity suite (by reflection) draws pygame's other outcome for them. Everything else calls <see cref="Aalines"/>,
        ///     which passes 0, a fresh interpreter's value.
        /// </summary>
        /// <param name="surface">The surface, without per-pixel alpha.</param>
        /// <param name="color">The colour; its alpha is dropped, as the surface maps it.</param>
        /// <param name="closed">Whether a segment joins the last point back to the first.</param>
        /// <param name="points">At least 2 points, as <c>(x, y)</c> floats.</param>
        /// <param name="pastTheBuffer">The float after the y's, read only with two points: NaN makes the first test false, an infinity true.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="surface"/> or <paramref name="points"/> is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">Thrown for fewer than 2 points, with pygame's message, or a NaN or infinite coordinate.</exception>
        /// <exception cref="NotSupportedException">Thrown for a surface with per-pixel alpha (not modelled).</exception>
        private static void DrawAalines(Surface surface, Color color, bool closed, IReadOnlyList<(double X, double Y)> points, float pastTheBuffer) {
            RequireOpaque(surface);
            if (points is null)
                throw new ArgumentNullException(nameof(points));
            if (points.Count < 2)
                throw new ValueError("points argument must contain 2 or more points");
            int length = points.Count;
            // pygame's layout: the x's, then the y's, in one buffer; the float after them stands for the memory pygame reads
            // past its buffer when given two points.
            var buffer = new float[2 * length + 1];
            for (int i = 0; i < length; i++) {
                RequireFinite(points[i].X);
                RequireFinite(points[i].Y);
                buffer[i] = (float)points[i].X;
                buffer[length + i] = (float)points[i].Y;
            }

            buffer[2 * length] = pastTheBuffer;
            ReadOnlySpan<float> xs = buffer.AsSpan(0, length), ys = buffer.AsSpan(length, length);

            // The first segment: an open polyline keeps its first endpoint.
            float p0 = xs[0], p1 = ys[0], p2 = xs[1], p3 = ys[1];
            float q0 = p0, q1 = p1, q2 = p2, q3 = p3;
            bool steepPrevious = MathF.Abs(q2 - q0) < MathF.Abs(q3 - q1);
            // pygame's test reads xlist[2] and ylist[2] against pts[2] and pts[1] (the first point's y, not pts[3]); with two
            // points they are the first point's y and the float past the y's (see the remarks of Aalines).
            bool steepCurrent = MathF.Abs(buffer[2] - p2) < MathF.Abs(buffer[length + 2] - p1);
            bool extraPixel = steepPrevious && !steepCurrent;
            bool disableEndpoints = !IsWhole(p2) || !IsWhole(p3);
            if (closed)
                DrawAaline(surface, color, p0, p1, p2, p3, disableEndpoints, disableEndpoints, extraPixel);
            else
                DrawAaline(surface, color, p0, p1, p2, p3, false, disableEndpoints, extraPixel);

            for (int loop = 2; loop < length - 1; loop++) {
                p0 = xs[loop - 1];
                p1 = ys[loop - 1];
                p2 = xs[loop];
                p3 = ys[loop];
                // When one segment is steep and the next is not, the joint gets an extra pixel.
                steepPrevious = MathF.Abs(q2 - q0) < MathF.Abs(q3 - q1);
                steepCurrent = MathF.Abs(p2 - p0) < MathF.Abs(p3 - p1);
                extraPixel = steepPrevious != steepCurrent;
                disableEndpoints = !IsWhole(p2) || !IsWhole(p3);
                (q0, q1, q2, q3) = (p0, p1, p2, p3);
                DrawAaline(surface, color, p0, p1, p2, p3, disableEndpoints, disableEndpoints, extraPixel);
            }

            // The last segment: an open polyline keeps its second endpoint, and its first follows the end point's wholeness.
            p0 = xs[length - 2];
            p1 = ys[length - 2];
            p2 = xs[length - 1];
            p3 = ys[length - 1];
            steepPrevious = MathF.Abs(q2 - q0) < MathF.Abs(q3 - q1);
            steepCurrent = MathF.Abs(p2 - p0) < MathF.Abs(p3 - p1);
            extraPixel = steepPrevious != steepCurrent;
            disableEndpoints = !IsWhole(p2) || !IsWhole(p3);
            (q0, q1, q2, q3) = (p0, p1, p2, p3);
            if (closed)
                DrawAaline(surface, color, p0, p1, p2, p3, disableEndpoints, disableEndpoints, extraPixel);
            else
                DrawAaline(surface, color, p0, p1, p2, p3, disableEndpoints, false, extraPixel);

            if (closed && length > 2) {
                p0 = xs[length - 1];
                p1 = ys[length - 1];
                p2 = xs[0];
                p3 = ys[0];
                steepPrevious = MathF.Abs(q2 - q0) < MathF.Abs(q3 - q1);
                steepCurrent = MathF.Abs(p2 - p0) < MathF.Abs(p3 - p1);
                extraPixel = steepPrevious != steepCurrent;
                disableEndpoints = !IsWhole(p2) || !IsWhole(p3);
                DrawAaline(surface, color, p0, p1, p2, p3, disableEndpoints, disableEndpoints, extraPixel);
            }
        }

        /// <summary>
        ///     pygame-ce's <c>draw_aaline</c>: one anti-aliased segment, clipped one pixel wider than the surface, its two
        ///     endpoint columns drawn unless disabled (an extra pixel forces them back), then its interior column by column.
        /// </summary>
        /// <param name="surface">The surface.</param>
        /// <param name="color">The colour.</param>
        /// <param name="fromX">The start column, float32.</param>
        /// <param name="fromY">The start row.</param>
        /// <param name="toX">The end column.</param>
        /// <param name="toY">The end row.</param>
        /// <param name="disableFirstEndpoint">Leave out the first endpoint's column (a joint of <c>aalines</c>).</param>
        /// <param name="disableSecondEndpoint">Leave out the second endpoint's column.</param>
        /// <param name="extraPixel">Draw both endpoint columns even when disabled (a joint between a steep segment and a flat one).</param>
        /// <remarks>
        ///     Every variable is float32 or int, and every expression keeps C's types and order, so the results are pygame's bit
        ///     for bit: <c>int - float</c> is a float subtraction, <c>(int)f</c> truncates, and the thresholds and floors of the
        ///     single-point case are taken in double, as C promotes them.
        /// </remarks>
        private static void DrawAaline(Surface surface, Color color, float fromX, float fromY, float toX, float toY, bool disableFirstEndpoint, bool disableSecondEndpoint,
            bool extraPixel) {
            float dx = toX - fromX;
            float dy = toY - fromY;

            // Single point: a line of length 0 is one pixel at full brightness. C's fabs and floor run in double here.
            if (Math.Abs((double)dx) < 0.0001 && Math.Abs((double)dy) < 0.0001) {
                int pointX = (int)Math.Floor(fromX + 0.5), pointY = (int)Math.Floor(fromY + 0.5);
                Plot(surface, color, pointX, pointY, 1f);
                return;
            }

            // To draw correctly the pixels at the border of the clipping area when the line crosses it, the line is clipped one
            // pixel wider in all four directions (the clip rectangle is the whole surface).
            float clipLeft = 0f - 1.0f;
            float clipRight = clipLeft + surface.Width + 1.0f;
            float clipTop = 0f - 1.0f;
            float clipBottom = clipTop + surface.Height + 1.0f;

            bool steep = MathF.Abs(dx) < MathF.Abs(dy);
            if (steep) {
                (fromX, fromY) = (fromY, fromX);
                (toX, toY) = (toY, toX);
                (dx, dy) = (dy, dx);
                (clipLeft, clipTop) = (clipTop, clipLeft);
                (clipRight, clipBottom) = (clipBottom, clipRight);
            }

            if (dx < 0) {
                (fromX, toX) = (toX, fromX);
                (fromY, toY) = (toY, fromY);
                dx = -dx;
                dy = -dy;
            }

            // The line is completely to the side of the surface.
            if (toX <= clipLeft || fromX >= clipRight)
                return;

            // No division by zero: a zero dx was either a single point (returned) or swapped with a non-zero dy.
            float gradient = dy / dx;

            // No pixels off the surface: the ends move onto the widened clip.
            if (fromX < clipLeft) {
                fromY += gradient * (clipLeft - fromX);
                fromX = clipLeft;
            }

            if (toX > clipRight) {
                toY += gradient * (clipRight - toX);
                toX = clipRight;
            }

            if (gradient > 0.0f) {
                // from is the topmost endpoint.
                if (toY <= clipTop || fromY >= clipBottom)
                    return;
                if (fromY < clipTop) {
                    fromX += (clipTop - fromY) / gradient;
                    fromY = clipTop;
                }

                if (toY > clipBottom) {
                    toX += (clipBottom - toY) / gradient;
                    toY = clipBottom;
                }
            } else {
                // to is the topmost endpoint.
                if (fromY <= clipTop || toY >= clipBottom)
                    return;
                if (toY < clipTop) {
                    toX += (clipTop - toY) / gradient;
                    toY = clipTop;
                }

                if (fromY > clipBottom) {
                    fromX += (clipBottom - fromY) / gradient;
                    fromY = clipBottom;
                }
            }

            // Moved one pixel down so y is never negative and (int)y rounds down; the pixels are drawn one higher.
            fromY += 1.0f;
            toY += 1.0f;

            int x, y, xPixelStart = 0;
            float intersectY = 0f, yEndpoint, xGap, brightness;
            // The first endpoint, unless disabled; an extra pixel draws it all the same.
            if (!disableFirstEndpoint || extraPixel) {
                xPixelStart = (int)fromX;
                yEndpoint = intersectY = fromY + gradient * (xPixelStart - fromX);
                if (toX > clipLeft + 1.0f) {
                    xGap = 1 + xPixelStart - fromX;
                    brightness = yEndpoint - (int)yEndpoint;
                    if (steep) {
                        x = (int)yEndpoint;
                        y = xPixelStart;
                    } else {
                        x = xPixelStart;
                        y = (int)yEndpoint;
                    }

                    if ((int)yEndpoint < yEndpoint)
                        Plot(surface, color, x, y, brightness * xGap);
                    if (steep)
                        x--;
                    else
                        y--;
                    brightness = 1 - brightness;
                    Plot(surface, color, x, y, brightness * xGap);
                    intersectY += gradient;
                    xPixelStart++;
                }
            }

            // The main loop skips the first endpoint when it is disabled.
            if (disableFirstEndpoint) {
                xPixelStart = (int)Math.Ceiling(fromX);
                intersectY = fromY + gradient * (xPixelStart - fromX);
            }

            // The second endpoint, unless disabled; an extra pixel draws it all the same.
            int xPixelEnd = (int)Math.Ceiling(toX);
            if (!disableSecondEndpoint || extraPixel) {
                if (fromX < clipRight - 1.0f) {
                    yEndpoint = toY + gradient * (xPixelEnd - toX);
                    xGap = 1 - xPixelEnd + toX;
                    brightness = yEndpoint - (int)yEndpoint;
                    if (steep) {
                        x = (int)yEndpoint;
                        y = xPixelEnd;
                    } else {
                        x = xPixelEnd;
                        y = (int)yEndpoint;
                    }

                    if ((int)yEndpoint < yEndpoint)
                        Plot(surface, color, x, y, brightness * xGap);
                    if (steep)
                        x--;
                    else
                        y--;
                    brightness = 1 - brightness;
                    Plot(surface, color, x, y, brightness * xGap);
                }
            }

            // The main line drawing loop, the minor coordinate accumulated by float32 addition as pygame does.
            for (x = xPixelStart; x < xPixelEnd; x++) {
                y = (int)intersectY;
                brightness = 1 - intersectY + y;
                if (steep) {
                    Plot(surface, color, y - 1, x, brightness);
                    if (y < intersectY) {
                        brightness = 1 - brightness;
                        Plot(surface, color, y, x, brightness);
                    }
                } else {
                    Plot(surface, color, x, y - 1, brightness);
                    if (y < intersectY) {
                        brightness = 1 - brightness;
                        Plot(surface, color, x, y, brightness);
                    }
                }

                intersectY += gradient;
            }
        }

        /// <summary>
        ///     pygame-ce's <c>get_antialiased_color</c> and <c>set_and_check_rect</c>: blends one on-surface pixel with the colour
        ///     by <paramref name="brightness"/>, <c>(Uint8)(b * colour + (1 - b) * background)</c> per channel in float32.
        /// </summary>
        /// <param name="surface">The surface.</param>
        /// <param name="color">The colour.</param>
        /// <param name="x">The column; a pixel off the surface is left alone.</param>
        /// <param name="y">The row.</param>
        /// <param name="brightness">The coverage, 0 to 1; 0 writes the background back unchanged.</param>
        private static void Plot(Surface surface, Color color, int x, int y, float brightness) {
            if (x < 0 || x >= surface.Width || y < 0 || y >= surface.Height)
                return;
            int index = y * surface.Width + x;
            uint background = surface.Pixels[index];
            // The alpha channel is blended too in pygame, then dropped by the map onto a surface without an alpha mask.
            // A first endpoint in (-1, 0) truncates to column 0, so its coverage can exceed 1 and the blend leave 0..255;
            // pygame's (Uint8) cast then keeps the low byte of the truncated int (cvttss2si), which (byte)(int) does on
            // every runtime, where a direct float-to-byte cast saturates from .NET 9 on.
            uint r = (byte)(int)(brightness * color.R + (1 - brightness) * ((background >> 16) & 0xFF));
            uint g = (byte)(int)(brightness * color.G + (1 - brightness) * ((background >> 8) & 0xFF));
            uint b = (byte)(int)(brightness * color.B + (1 - brightness) * (background & 0xFF));
            surface.Pixels[index] = (r << 16) | (g << 8) | b;
        }

        /// <summary>C's <c>roundf(v) == v</c>: whether a float is a whole number (an endpoint on a pixel centre).</summary>
        /// <param name="value">The coordinate.</param>
        /// <returns><see langword="true"/> for a whole number.</returns>
        private static bool IsWhole(float value) => MathF.Round(value, MidpointRounding.AwayFromZero) == value;
    }
}
