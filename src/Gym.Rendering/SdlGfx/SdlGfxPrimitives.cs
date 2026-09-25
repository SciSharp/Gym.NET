/*
 * This file is a C# port of parts of SDL_gfxPrimitives.c from SDL_gfx 2.0.26, whose notice follows unchanged:
 *
 * SDL_gfxPrimitives.c: graphics primitives for SDL surfaces
 *
 * Copyright (C) 2001-2012  Andreas Schiffler
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 * claim that you wrote the original software. If you use this software
 * in a product, an acknowledgment in the product documentation would be
 * appreciated but is not required.
 *
 * 2. Altered source versions must be plainly marked as such, and must not be
 * misrepresented as being the original software.
 *
 * 3. This notice may not be removed or altered from any source
 * distribution.
 *
 * Andreas Schiffler -- aschiffler at ferzkopp dot net
 *
 * ALTERED SOURCE VERSION (Gym.NET, 2026): translated to C#; reduced to the functions that pygame.gfxdraw's aapolygon,
 * filled_polygon, aacircle, filled_circle, hline and vline reach; specialized to 32-bit surfaces without an alpha mask
 * (the XRGB8888 surfaces Gymnasium renders on); surface locking dropped (there is nothing to lock); and aligned with the
 * copy of this file that pygame-ce 2.5.8 bundles, which differs from 2.0.26 in aaellipseColor (no "od" overdraw term)
 * and in _aalineColor (the error adjustment is shifted as an unsigned value). It is not the original software.
 */

using System;
using System.Collections.Generic;

namespace Gym.Rendering.SdlGfx {
    /// <summary>
    ///     The SDL_gfx 2.0.26 primitives behind <c>pygame.gfxdraw</c>, reproduced instruction for instruction so every pixel
    ///     matches what pygame-ce 2.5.8 draws: the same Sint16 coordinates and wrap-arounds, the same single-precision
    ///     clipping arithmetic, the same fixed-point anti-aliasing weights, the same blend rounding.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Colours arrive as SDL_gfx takes them, <c>0xRRGGBBAA</c>. An alpha of 255 overwrites; anything lower blends each
    ///     channel as <c>d + ((s - d) * alpha >> 8)</c>, which is SDL_gfx's unsigned 32-bit arithmetic worked out (the
    ///     wrapped products and masks come to a floored division by 256 either way). Anti-aliased edges scale the colour's
    ///     alpha by a weight first, <c>(alpha * weight) >> 8</c>, so even an opaque colour blends at the edges.
    ///     </para>
    ///     <para>
    ///     Coordinates are C <c>Sint16</c> (<see langword="short"/>): sums like <c>x + rad</c> wrap exactly as the C code's
    ///     assignments to Sint16 do, which is observable for shapes near ±32768. Integer arithmetic is unchecked, as C's is.
    ///     Return values keep SDL_gfx's convention: 0 for success, -1 for a rejected argument (a negative radius), which
    ///     <see cref="GfxDraw"/> turns into pygame's error.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/pygame-community/pygame-ce/blob/2.5.8/src_c/SDL_gfx/SDL_gfxPrimitives.c">src_c/SDL_gfx/SDL_gfxPrimitives.c</see> (as vendored by pygame-ce 2.5.8).</para>
    /// </remarks>
    internal static class SdlGfxPrimitives {
        /// <summary>Anti-aliasing bits: the weight of an anti-aliased pixel is the top 8 bits of the error accumulator.</summary>
        private const int AAbits = 8;

        /// <summary>Anti-aliasing levels, <c>2^AAbits</c>.</summary>
        private const int AAlevels = 256;

        /// <summary>Cohen–Sutherland outcode: left of the clip rectangle.</summary>
        private const int ClipLeftEdge = 0x1;

        /// <summary>Cohen–Sutherland outcode: right of the clip rectangle.</summary>
        private const int ClipRightEdge = 0x2;

        /// <summary>Cohen–Sutherland outcode: below the clip rectangle (greater y).</summary>
        private const int ClipBottomEdge = 0x4;

        /// <summary>Cohen–Sutherland outcode: above the clip rectangle (smaller y).</summary>
        private const int ClipTopEdge = 0x8;

        /// <summary>
        ///     A guard against the clipping loop cycling forever, which the C loop has no protection from. Clipping moves one
        ///     endpoint onto an edge per turn, so a terminating clip takes a handful of turns; only a case that would hang
        ///     pygame reaches this.
        /// </summary>
        private const int MaxClipTurns = 1024;

        /// <summary><c>pixelColor</c>: one pixel, blended by the colour's alpha. Locking is a no-op here, so it is <see cref="PixelColorNolock"/>.</summary>
        /// <param name="dst">The surface.</param>
        /// <param name="x">The column.</param>
        /// <param name="y">The row.</param>
        /// <param name="color">The colour, <c>0xRRGGBBAA</c>.</param>
        /// <returns>0.</returns>
        internal static int PixelColor(Surface dst, short x, short y, uint color) => PixelColorNolock(dst, x, y, color);

        /// <summary><c>pixelColorNolock</c>: maps the colour into the surface format and puts it with the colour's alpha.</summary>
        /// <param name="dst">The surface.</param>
        /// <param name="x">The column.</param>
        /// <param name="y">The row.</param>
        /// <param name="color">The colour, <c>0xRRGGBBAA</c>.</param>
        /// <returns>0.</returns>
        internal static int PixelColorNolock(Surface dst, short x, short y, uint color) {
            byte alpha = (byte)(color & 0x000000ff);
            // SDL_MapRGBA for XRGB8888: the alpha has no bits to go to.
            uint mcolor = (((color & 0xff000000) >> 24) << 16) | (((color & 0x00ff0000) >> 16) << 8) | ((color & 0x0000ff00) >> 8);
            return PutPixelAlpha(dst, x, y, mcolor, alpha);
        }

        /// <summary><c>pixelColorWeightNolock</c>: scales the colour's alpha by <paramref name="weight"/>/256 first, the anti-aliasing step.</summary>
        /// <param name="dst">The surface.</param>
        /// <param name="x">The column.</param>
        /// <param name="y">The row.</param>
        /// <param name="color">The colour, <c>0xRRGGBBAA</c>.</param>
        /// <param name="weight">The coverage, 0–255.</param>
        /// <returns>0.</returns>
        internal static int PixelColorWeightNolock(Surface dst, short x, short y, uint color, uint weight) {
            uint a = color & 0x000000ff;
            a = (a * weight) >> 8;
            return PixelColorNolock(dst, x, y, (color & 0xffffff00) | a);
        }

        /// <summary>
        ///     <c>_putPixelAlpha</c>, 32-bit case: inside the clip rectangle, an alpha of 255 stores the mapped colour and any
        ///     other alpha blends each channel.
        /// </summary>
        /// <param name="dst">The surface.</param>
        /// <param name="x">The column.</param>
        /// <param name="y">The row.</param>
        /// <param name="color">The mapped colour, <c>0x00RRGGBB</c>.</param>
        /// <param name="alpha">The blend weight.</param>
        /// <returns>0 (also when the pixel is clipped away).</returns>
        private static int PutPixelAlpha(Surface dst, short x, short y, uint color, byte alpha) {
            if (x >= 0 && x <= dst.Width - 1 && y >= 0 && y <= dst.Height - 1) {
                uint[] pixels = dst.Pixels;
                int index = y * dst.Width + x;
                if (alpha == 255) {
                    pixels[index] = color;
                } else {
                    unchecked {
                        uint dc = pixels[index];
                        uint dR = (dc & 0x00ff0000) >> 16;
                        uint dG = (dc & 0x0000ff00) >> 8;
                        uint dB = dc & 0x000000ff;
                        // Uint32 arithmetic as in C: (s - d) wraps when s < d, and the shift and mask bring it back to
                        // d + floor((s - d) * alpha / 256).
                        uint r = ((dR + ((((color & 0x00ff0000) >> 16) - dR) * alpha >> 8)) << 16) & 0x00ff0000;
                        uint g = ((dG + ((((color & 0x0000ff00) >> 8) - dG) * alpha >> 8)) << 8) & 0x0000ff00;
                        uint b = (dB + (((color & 0x000000ff) - dB) * alpha >> 8)) & 0x000000ff;
                        // No alpha mask: the pixel keeps only the three channels.
                        pixels[index] = r | g | b;
                    }
                }
            }

            return 0;
        }

        /// <summary>
        ///     <c>hlineColor</c>: the pixels from <paramref name="x1"/> to <paramref name="x2"/> inclusive on row
        ///     <paramref name="y"/>, clipped; opaque colours are stored, others blended.
        /// </summary>
        /// <param name="dst">The surface.</param>
        /// <param name="x1">One end.</param>
        /// <param name="x2">The other end (the order does not matter).</param>
        /// <param name="y">The row.</param>
        /// <param name="color">The colour, <c>0xRRGGBBAA</c>.</param>
        /// <returns>0.</returns>
        internal static int HlineColor(Surface dst, short x1, short x2, short y, uint color) {
            if (dst.Width == 0 || dst.Height == 0)
                return 0;
            if (x1 > x2)
                (x1, x2) = (x2, x1);

            short left = 0;
            if (x2 < left)
                return 0;
            short right = (short)(dst.Width - 1);
            if (x1 > right)
                return 0;
            short top = 0, bottom = (short)(dst.Height - 1);
            if (y < top || y > bottom)
                return 0;

            if (x1 < left)
                x1 = left;
            if (x2 > right)
                x2 = right;
            int dx = x2 - x1;

            if ((color & 255) == 255) {
                uint mapped = MapRgb(color);
                Array.Fill(dst.Pixels, mapped, y * dst.Width + x1, dx + 1);
                return 0;
            }

            return FilledRectAlpha(dst, x1, y, (short)(x1 + dx), y, color);
        }

        /// <summary>
        ///     <c>vlineColor</c>: the pixels from <paramref name="y1"/> to <paramref name="y2"/> inclusive in column
        ///     <paramref name="x"/>, clipped; opaque colours are stored, others blended.
        /// </summary>
        /// <param name="dst">The surface.</param>
        /// <param name="x">The column.</param>
        /// <param name="y1">One end.</param>
        /// <param name="y2">The other end (the order does not matter).</param>
        /// <param name="color">The colour, <c>0xRRGGBBAA</c>.</param>
        /// <returns>0.</returns>
        internal static int VlineColor(Surface dst, short x, short y1, short y2, uint color) {
            if (dst.Width == 0 || dst.Height == 0)
                return 0;
            if (y1 > y2)
                (y1, y2) = (y2, y1);

            short left = 0, right = (short)(dst.Width - 1);
            if (x < left || x > right)
                return 0;
            short top = 0;
            if (y2 < top)
                return 0;
            short bottom = (short)(dst.Height - 1);
            if (y1 > bottom)
                return 0;

            if (y1 < top)
                y1 = top;
            if (y2 > bottom)
                y2 = bottom;
            short h = (short)(y2 - y1);

            if ((color & 255) == 255) {
                uint mapped = MapRgb(color);
                uint[] pixels = dst.Pixels;
                for (int row = y1, index = y1 * dst.Width + x; row <= y1 + h; row++, index += dst.Width)
                    pixels[index] = mapped;
                return 0;
            }

            return FilledRectAlpha(dst, x, y1, x, (short)(y1 + h), color);
        }

        /// <summary>
        ///     <c>filledRectAlpha</c> with <c>_filledRectAlpha</c>'s 32-bit case: blends every pixel of the (already clipped)
        ///     rectangle with the colour's alpha. Reached only for non-opaque lines.
        /// </summary>
        /// <param name="dst">The surface.</param>
        /// <param name="x1">The left column.</param>
        /// <param name="y1">The top row.</param>
        /// <param name="x2">The right column.</param>
        /// <param name="y2">The bottom row.</param>
        /// <param name="color">The colour, <c>0xRRGGBBAA</c>.</param>
        /// <returns>0.</returns>
        private static int FilledRectAlpha(Surface dst, short x1, short y1, short x2, short y2, uint color) {
            uint alpha = color & 0x000000ff;
            uint sR = (color & 0xff000000) >> 24;
            uint sG = (color & 0x00ff0000) >> 16;
            uint sB = (color & 0x0000ff00) >> 8;
            uint[] pixels = dst.Pixels;
            unchecked {
                for (int y = y1; y <= y2; y++) {
                    int row = y * dst.Width;
                    for (int x = x1; x <= x2; x++) {
                        uint dc = pixels[row + x];
                        uint dR = (dc & 0x00ff0000) >> 16;
                        uint dG = (dc & 0x0000ff00) >> 8;
                        uint dB = dc & 0x000000ff;
                        // Uint32 wrap-around, as in _putPixelAlpha.
                        uint r = ((dR + ((sR - dR) * alpha >> 8)) << 16) & 0x00ff0000;
                        uint g = ((dG + ((sG - dG) * alpha >> 8)) << 8) & 0x0000ff00;
                        uint b = (dB + ((sB - dB) * alpha >> 8)) & 0x000000ff;
                        pixels[row + x] = r | g | b;
                    }
                }
            }

            return 0;
        }

        /// <summary><c>_clipEncode</c>: the Cohen–Sutherland outcode of a point against the clip rectangle.</summary>
        /// <param name="x">The column.</param>
        /// <param name="y">The row.</param>
        /// <param name="left">The clip rectangle's left column.</param>
        /// <param name="top">Its top row.</param>
        /// <param name="right">Its right column.</param>
        /// <param name="bottom">Its bottom row.</param>
        /// <returns>The outcode; 0 inside.</returns>
        private static int ClipEncode(short x, short y, short left, short top, short right, short bottom) {
            int code = 0;
            if (x < left)
                code |= ClipLeftEdge;
            else if (x > right)
                code |= ClipRightEdge;
            if (y < top)
                code |= ClipTopEdge;
            else if (y > bottom)
                code |= ClipBottomEdge;
            return code;
        }

        /// <summary>
        ///     <c>_clipLine</c>: Cohen–Sutherland clipping of a segment to the surface, with the slope in single precision and
        ///     every move truncated to Sint16, which decides where clipped anti-aliased edges start.
        /// </summary>
        /// <param name="dst">The surface.</param>
        /// <param name="x1">The first end's column; moved onto the clip rectangle.</param>
        /// <param name="y1">The first end's row.</param>
        /// <param name="x2">The second end's column.</param>
        /// <param name="y2">The second end's row.</param>
        /// <returns>Whether any of the segment is visible.</returns>
        /// <exception cref="InvalidOperationException">Thrown if clipping does not terminate, where the C loop would hang.</exception>
        private static bool ClipLine(Surface dst, ref short x1, ref short y1, ref short x2, ref short y2) {
            short left = 0, right = (short)(dst.Width - 1), top = 0, bottom = (short)(dst.Height - 1);
            for (int turn = 0; turn < MaxClipTurns; turn++) {
                int code1 = ClipEncode(x1, y1, left, top, right, bottom);
                int code2 = ClipEncode(x2, y2, left, top, right, bottom);
                if ((code1 | code2) == 0)
                    return true;
                if ((code1 & code2) != 0)
                    return false;

                if (code1 == 0) {
                    // Always move the first end: swap so that it is the outside one.
                    (x1, x2) = (x2, x1);
                    (y1, y2) = (y2, y1);
                    (code1, code2) = (code2, code1);
                }

                // float m and int * float products, as C computes them: single precision, then truncated to Sint16.
                float m = x2 != x1 ? (float)(y2 - y1) / (float)(x2 - x1) : 1.0f;
                unchecked {
                    if ((code1 & ClipLeftEdge) != 0) {
                        y1 = (short)(y1 + (short)(int)(float)((float)(left - x1) * m));
                        x1 = left;
                    } else if ((code1 & ClipRightEdge) != 0) {
                        y1 = (short)(y1 + (short)(int)(float)((float)(right - x1) * m));
                        x1 = right;
                    } else if ((code1 & ClipBottomEdge) != 0) {
                        if (x2 != x1)
                            x1 = (short)(x1 + (short)(int)(float)((float)(bottom - y1) / m));
                        y1 = bottom;
                    } else if ((code1 & ClipTopEdge) != 0) {
                        if (x2 != x1)
                            x1 = (short)(x1 + (short)(int)(float)((float)(top - y1) / m));
                        y1 = top;
                    }
                }
            }

            throw new InvalidOperationException("SDL_gfx line clipping did not terminate; pygame would hang on this segment.");
        }

        /// <summary>
        ///     <c>_aalineColor</c> with <c>draw_endpoint = 0</c> (aapolygon's edges): Wu's anti-aliased line with 16.16 fixed-point
        ///     error accumulation. Its quirks are kept: the first pixel is the clipped first end even when the line is walked
        ///     from the other end, the last pixel is never drawn, and axis-aligned edges are drawn as full lines, both ends
        ///     included.
        /// </summary>
        /// <param name="dst">The surface.</param>
        /// <param name="x1">The first end's column.</param>
        /// <param name="y1">The first end's row.</param>
        /// <param name="x2">The second end's column.</param>
        /// <param name="y2">The second end's row.</param>
        /// <param name="color">The colour, <c>0xRRGGBBAA</c>.</param>
        /// <returns>0.</returns>
        private static int AalineColorNoEndpoint(Surface dst, short x1, short y1, short x2, short y2, uint color) {
            if (dst.Width == 0 || dst.Height == 0)
                return 0;
            if (!ClipLine(dst, ref x1, ref y1, ref x2, ref y2))
                return 0;

            int xx0 = x1, yy0 = y1, xx1 = x2, yy1 = y2;
            if (yy0 > yy1) {
                (yy0, yy1) = (yy1, yy0);
                (xx0, xx1) = (xx1, xx0);
            }

            int dx = xx1 - xx0;
            int dy = yy1 - yy0;

            if (dx == 0) {
                // Vertical: both ends drawn despite draw_endpoint = 0 (the C code calls vlineColor either way).
                return dy > 0 ? VlineColor(dst, x1, (short)yy0, (short)(yy0 + dy), color) : PixelColor(dst, x1, y1, color);
            }

            if (dy == 0) {
                // Horizontal: likewise a full hline (dx != 0 here, so the C code's single-pixel branch is unreachable).
                return HlineColor(dst, (short)xx0, (short)(xx0 + dx), y1, color);
            }

            int xdir;
            if (dx >= 0) {
                xdir = 1;
            } else {
                xdir = -1;
                dx = -dx;
            }

            uint erracc = 0;
            const int intshift = 32 - AAbits;
            int result = 0;

            // The initial pixel at the clipped first end, whichever end the walk below starts from.
            result |= PixelColorNolock(dst, x1, y1, color);

            unchecked {
                if (dy > dx) {
                    // y-major: 16.16 fraction of a pixel that x advances per row. pygame shifts it as Uint32, so dx == dy
                    // would give 0 (never reached here: dy > dx).
                    uint erradj = (uint)((dx << 16) / dy) << 16;
                    int x0pxdir = xx0 + xdir;
                    while (--dy != 0) {
                        uint erracctmp = erracc;
                        erracc += erradj;
                        if (erracc <= erracctmp) {
                            // The accumulator rolled over: x advances.
                            xx0 = x0pxdir;
                            x0pxdir += xdir;
                        }

                        yy0++;
                        // The top AAbits bits weight this pixel's partner; this pixel gets the complement.
                        uint wgt = (erracc >> intshift) & 255;
                        result |= PixelColorWeightNolock(dst, (short)xx0, (short)yy0, color, 255 - wgt);
                        result |= PixelColorWeightNolock(dst, (short)x0pxdir, (short)yy0, color, wgt);
                    }
                } else {
                    // x-major (dx == dy included): 16.16 fraction of a pixel that y advances per column; for dx == dy the
                    // Uint32 shift wraps to 0, and the rollover test (<=) then advances y on every column.
                    uint erradj = (uint)((dy << 16) / dx) << 16;
                    int y0p1 = yy0 + 1;
                    while (--dx != 0) {
                        uint erracctmp = erracc;
                        erracc += erradj;
                        if (erracc <= erracctmp) {
                            yy0 = y0p1;
                            y0p1++;
                        }

                        xx0 += xdir;
                        uint wgt = (erracc >> intshift) & 255;
                        result |= PixelColorWeightNolock(dst, (short)xx0, (short)yy0, color, 255 - wgt);
                        result |= PixelColorWeightNolock(dst, (short)xx0, (short)y0p1, color, wgt);
                    }
                }
            }

            return result;
        }

        /// <summary>
        ///     <c>aapolygonColor</c>: the anti-aliased outline, edge by edge in vertex order and then the closing edge from the
        ///     last vertex to the first, each without its end pixel.
        /// </summary>
        /// <param name="dst">The surface.</param>
        /// <param name="vx">The vertices' columns.</param>
        /// <param name="vy">The vertices' rows.</param>
        /// <param name="color">The colour, <c>0xRRGGBBAA</c>.</param>
        /// <returns>0, or -1 for fewer than 3 vertices.</returns>
        internal static int AapolygonColor(Surface dst, short[] vx, short[] vy, uint color) {
            if (dst.Width == 0 || dst.Height == 0)
                return 0;
            int n = vx.Length;
            if (n < 3)
                return -1;

            int result = 0;
            for (int i = 1; i < n; i++)
                result |= AalineColorNoEndpoint(dst, vx[i - 1], vy[i - 1], vx[i], vy[i], color);
            result |= AalineColorNoEndpoint(dst, vx[n - 1], vy[n - 1], vx[0], vy[0], color);
            return result;
        }

        /// <summary>
        ///     <c>filledPolygonColorMT</c>: scanline fill. Each row from the lowest to the highest vertex collects the
        ///     crossings of the edges that span it (half-open, except that edges ending on the last row count there), in 16.16
        ///     fixed point; sorts them; and fills between pairs, rounding the left end up and the right end down.
        /// </summary>
        /// <param name="dst">The surface.</param>
        /// <param name="vx">The vertices' columns.</param>
        /// <param name="vy">The vertices' rows.</param>
        /// <param name="color">The colour, <c>0xRRGGBBAA</c>.</param>
        /// <returns>0, or -1 for fewer than 3 vertices.</returns>
        internal static int FilledPolygonColor(Surface dst, short[] vx, short[] vy, uint color) {
            if (dst.Width == 0 || dst.Height == 0)
                return 0;
            int n = vx.Length;
            if (n < 3)
                return -1;

            // The C code keeps one growable global buffer of n ints; a fresh one is equivalent (every read follows a write).
            var polyInts = new int[n];
            int miny = vy[0], maxy = vy[0];
            for (int i = 1; i < n; i++) {
                if (vy[i] < miny)
                    miny = vy[i];
                else if (vy[i] > maxy)
                    maxy = vy[i];
            }

            int result = 0;
            unchecked {
                for (int y = miny; y <= maxy; y++) {
                    int ints = 0;
                    for (int i = 0; i < n; i++) {
                        int ind1, ind2;
                        if (i == 0) {
                            ind1 = n - 1;
                            ind2 = 0;
                        } else {
                            ind1 = i - 1;
                            ind2 = i;
                        }

                        int y1 = vy[ind1], y2 = vy[ind2], x1, x2;
                        if (y1 < y2) {
                            x1 = vx[ind1];
                            x2 = vx[ind2];
                        } else if (y1 > y2) {
                            y2 = vy[ind1];
                            y1 = vy[ind2];
                            x2 = vx[ind1];
                            x1 = vx[ind2];
                        } else {
                            continue;
                        }

                        if ((y >= y1 && y < y2) || (y == maxy && y > y1 && y <= y2))
                            // int arithmetic that wraps on overflow, as MSVC's does for extreme coordinates.
                            polyInts[ints++] = ((65536 * (y - y1)) / (y2 - y1)) * (x2 - x1) + 65536 * x1;
                    }

                    SortCrossings(polyInts, ints);

                    for (int i = 0; i < ints; i += 2) {
                        int xa = polyInts[i] + 1;
                        xa = (xa >> 16) + ((xa & 32768) >> 15);
                        int xb = polyInts[i + 1] - 1;
                        xb = (xb >> 16) + ((xb & 32768) >> 15);
                        result |= HlineColor(dst, (short)xa, (short)xb, (short)y, color);
                    }
                }
            }

            return result;
        }

        /// <summary>
        ///     The <c>qsort(ints, count, sizeof(int), _gfxPrimitivesCompareInt)</c> of the scanline fill, as the MSVC runtime
        ///     pygame-ce's Windows builds link performs it: up to 8 elements, a selection sort that moves the maximum to the
        ///     end, comparing with <c>a - b</c> in wrapping 32-bit arithmetic.
        /// </summary>
        /// <param name="values">The crossings; sorted in place.</param>
        /// <param name="count">How many of them are in use.</param>
        /// <remarks>
        ///     The wrapping comparator only misorders crossings more than 32767 pixels apart, which wrapped coordinates can
        ///     produce; reproducing the runtime's algorithm makes even those rows identical. Rows with more than 8 crossings
        ///     (only self-intersecting polygons have them) use a plain sort, identical whenever the comparator is consistent.
        /// </remarks>
        private static void SortCrossings(int[] values, int count) {
            if (count < 2)
                return;
            if (count > 8) {
                Array.Sort(values, 0, count, Comparer<int>.Create(static (a, b) => unchecked(a - b)));
                return;
            }

            for (int hi = count - 1; hi > 0; hi--) {
                int max = 0;
                for (int p = 1; p <= hi; p++) {
                    if (unchecked(values[p] - values[max]) > 0)
                        max = p;
                }

                if (max != hi)
                    (values[max], values[hi]) = (values[hi], values[max]);
            }
        }

        /// <summary>
        ///     <c>aaellipseColor</c> as pygame-ce 2.5.8 builds it (no <c>od</c> overdraw): the anti-aliased outline, walked
        ///     in two passes (x steps, then y steps) from the top, mirrored into four quadrants, each pixel paired with its
        ///     neighbour and weighted by the single-precision error ratio.
        /// </summary>
        /// <param name="dst">The surface.</param>
        /// <param name="x">The centre's column.</param>
        /// <param name="y">The centre's row.</param>
        /// <param name="rx">The horizontal radius.</param>
        /// <param name="ry">The vertical radius.</param>
        /// <param name="color">The colour, <c>0xRRGGBBAA</c>.</param>
        /// <returns>0, or -1 for a negative radius.</returns>
        internal static int AaellipseColor(Surface dst, short x, short y, short rx, short ry, uint color) {
            if (dst.Width == 0 || dst.Height == 0)
                return 0;
            if (rx < 0 || ry < 0)
                return -1;

            unchecked {
                if (rx == 0)
                    return VlineColor(dst, x, (short)(y - ry), (short)(y + ry), color);
                if (ry == 0)
                    return HlineColor(dst, (short)(x - rx), (short)(x + rx), y, color);

                // The bounding box, in Sint16 (so it wraps near the ends of the range, as in C).
                short x2 = (short)(x + rx);
                short left = 0;
                if (x2 < left)
                    return 0;
                short x1 = (short)(x - rx);
                short right = (short)(dst.Width - 1);
                if (x1 > right)
                    return 0;
                short y2 = (short)(y + ry);
                short top = 0;
                if (y2 < top)
                    return 0;
                short y1 = (short)(y - ry);
                short bottom = (short)(dst.Height - 1);
                if (y1 > bottom)
                    return 0;

                int a2 = rx * rx;
                int b2 = ry * ry;
                int ds = 2 * a2;
                int dt = 2 * b2;
                short xc2 = (short)(2 * x);
                short yc2 = (short)(2 * y);

                double sab = Math.Sqrt(a2 + b2);
                int dxt = (short)LRint(a2 / sab);

                int t = 0;
                int s = -2 * a2 * ry;
                int d = 0;

                short xp = x;
                short yp = (short)(y - ry);

                int result = 0;

                // The four "end points", opaque (full colour alpha, no weight).
                result |= PixelColorNolock(dst, xp, yp, color);
                result |= PixelColorNolock(dst, (short)(xc2 - xp), yp, color);
                result |= PixelColorNolock(dst, xp, (short)(yc2 - yp), color);
                result |= PixelColorNolock(dst, (short)(xc2 - xp), (short)(yc2 - yp), color);

                short xs, ys, xx, yy;
                for (int i = 1; i <= dxt; i++) {
                    xp--;
                    d += t - b2;

                    if (d >= 0) {
                        ys = (short)(yp - 1);
                    } else if (d - s - a2 > 0) {
                        if (2 * d - s - a2 >= 0) {
                            ys = (short)(yp + 1);
                        } else {
                            ys = yp;
                            yp++;
                            d -= s + a2;
                            s += ds;
                        }
                    } else {
                        yp++;
                        ys = (short)(yp + 1);
                        d -= s + a2;
                        s += ds;
                    }

                    t -= dt;

                    byte weight = Weight(d, s);
                    byte iweight = (byte)(255 - weight);

                    // Upper half.
                    xx = (short)(xc2 - xp);
                    result |= PixelColorWeightNolock(dst, xp, yp, color, iweight);
                    result |= PixelColorWeightNolock(dst, xx, yp, color, iweight);
                    result |= PixelColorWeightNolock(dst, xp, ys, color, weight);
                    result |= PixelColorWeightNolock(dst, xx, ys, color, weight);

                    // Lower half.
                    yy = (short)(yc2 - yp);
                    result |= PixelColorWeightNolock(dst, xp, yy, color, iweight);
                    result |= PixelColorWeightNolock(dst, xx, yy, color, iweight);
                    yy = (short)(yc2 - ys);
                    result |= PixelColorWeightNolock(dst, xp, yy, color, weight);
                    result |= PixelColorWeightNolock(dst, xx, yy, color, weight);
                }

                short dyt = (short)LRint(b2 / sab);
                for (int i = 1; i <= dyt; i++) {
                    yp++;
                    d -= s + a2;

                    if (d <= 0) {
                        xs = (short)(xp + 1);
                    } else if (d + t - b2 < 0) {
                        if (2 * d + t - b2 <= 0) {
                            xs = (short)(xp - 1);
                        } else {
                            xs = xp;
                            xp--;
                            d += t - b2;
                            t -= dt;
                        }
                    } else {
                        xp--;
                        xs = (short)(xp - 1);
                        d += t - b2;
                        t -= dt;
                    }

                    s += ds;

                    byte weight = Weight(d, t);
                    byte iweight = (byte)(255 - weight);

                    // Left half.
                    xx = (short)(xc2 - xp);
                    yy = (short)(yc2 - yp);
                    result |= PixelColorWeightNolock(dst, xp, yp, color, iweight);
                    result |= PixelColorWeightNolock(dst, xx, yp, color, iweight);
                    result |= PixelColorWeightNolock(dst, xp, yy, color, iweight);
                    result |= PixelColorWeightNolock(dst, xx, yy, color, iweight);

                    // Right half.
                    xx = (short)(xc2 - xs);
                    result |= PixelColorWeightNolock(dst, xs, yp, color, weight);
                    result |= PixelColorWeightNolock(dst, xx, yp, color, weight);
                    result |= PixelColorWeightNolock(dst, xs, yy, color, weight);
                    result |= PixelColorWeightNolock(dst, xx, yy, color, weight);
                }

                return result;
            }
        }

        /// <summary>
        ///     The anti-aliasing weight of the ellipse walk: <c>cp = (float)abs(d) / (float)abs(den)</c> capped at 1 (1 when the
        ///     denominator is 0), then <c>(Uint8)(cp * 255)</c> in single precision, truncated.
        /// </summary>
        /// <param name="d">The decision variable.</param>
        /// <param name="denominator">The step term it is compared with (<c>s</c> in the first pass, <c>t</c> in the second).</param>
        /// <returns>The weight of the paired pixel.</returns>
        private static byte Weight(int d, int denominator) {
            float cp;
            if (denominator != 0) {
                // C's abs() on int: int.MinValue stays negative (Math.Abs would throw), which only huge radii reach.
                cp = (float)AbsWrapping(d) / (float)AbsWrapping(denominator);
                if (cp > 1.0)
                    cp = 1.0f;
            } else {
                cp = 1.0f;
            }

            return unchecked((byte)(int)(float)(cp * 255));
        }

        /// <summary>C's <c>abs(int)</c> as compiled: the two's-complement negation, so <see cref="int.MinValue"/> maps to itself.</summary>
        /// <param name="value">The value.</param>
        /// <returns>Its magnitude, or <see cref="int.MinValue"/>.</returns>
        private static int AbsWrapping(int value) => value < 0 ? unchecked(-value) : value;

        /// <summary>C's <c>lrint</c> in the default rounding mode: to the nearest integer, ties to even.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The rounded value.</returns>
        private static long LRint(double value) => (long)Math.Round(value, MidpointRounding.ToEven);

        /// <summary>
        ///     <c>filledCircleColor</c>: the midpoint circle, filled with horizontal lines, each row drawn once (tracked by
        ///     <c>ocx</c>/<c>ocy</c>) except where the octants meet, so translucent colours overlap exactly where SDL_gfx's do.
        /// </summary>
        /// <param name="dst">The surface.</param>
        /// <param name="x">The centre's column.</param>
        /// <param name="y">The centre's row.</param>
        /// <param name="rad">The radius.</param>
        /// <param name="color">The colour, <c>0xRRGGBBAA</c>.</param>
        /// <returns>0, or -1 for a negative radius.</returns>
        internal static int FilledCircleColor(Surface dst, short x, short y, short rad, uint color) {
            if (dst.Width == 0 || dst.Height == 0)
                return 0;
            if (rad < 0)
                return -1;
            if (rad == 0)
                return PixelColor(dst, x, y, color);

            unchecked {
                short cx = 0;
                short cy = rad;
                short ocx = -1;
                short ocy = -1;
                short df = (short)(1 - rad);
                short dE = 3;
                short dSe = (short)(-2 * rad + 5);

                short x2 = (short)(x + rad);
                short left = 0;
                if (x2 < left)
                    return 0;
                short x1 = (short)(x - rad);
                short right = (short)(dst.Width - 1);
                if (x1 > right)
                    return 0;
                short y2 = (short)(y + rad);
                short top = 0;
                if (y2 < top)
                    return 0;
                short y1 = (short)(y - rad);
                short bottom = (short)(dst.Height - 1);
                if (y1 > bottom)
                    return 0;

                int result = 0;
                do {
                    short xpcx = (short)(x + cx);
                    short xmcx = (short)(x - cx);
                    short xpcy = (short)(x + cy);
                    short xmcy = (short)(x - cy);
                    if (ocy != cy) {
                        if (cy > 0) {
                            short ypcy = (short)(y + cy);
                            short ymcy = (short)(y - cy);
                            result |= HlineColor(dst, xmcx, xpcx, ypcy, color);
                            result |= HlineColor(dst, xmcx, xpcx, ymcy, color);
                        } else {
                            result |= HlineColor(dst, xmcx, xpcx, y, color);
                        }

                        ocy = cy;
                    }

                    if (ocx != cx) {
                        if (cx != cy) {
                            if (cx > 0) {
                                short ypcx = (short)(y + cx);
                                short ymcx = (short)(y - cx);
                                result |= HlineColor(dst, xmcy, xpcy, ymcx, color);
                                result |= HlineColor(dst, xmcy, xpcy, ypcx, color);
                            } else {
                                result |= HlineColor(dst, xmcy, xpcy, y, color);
                            }
                        }

                        ocx = cx;
                    }

                    if (df < 0) {
                        df = (short)(df + dE);
                        dE = (short)(dE + 2);
                        dSe = (short)(dSe + 2);
                    } else {
                        df = (short)(df + dSe);
                        dE = (short)(dE + 2);
                        dSe = (short)(dSe + 4);
                        cy--;
                    }

                    cx++;
                } while (cx <= cy);

                return result;
            }
        }

        /// <summary><c>SDL_MapRGBA</c> of a packed colour for XRGB8888: <c>0x00RRGGBB</c>, the alpha dropped.</summary>
        /// <param name="color">The colour, <c>0xRRGGBBAA</c>.</param>
        /// <returns>The pixel value.</returns>
        private static uint MapRgb(uint color) => color >> 8;
    }
}
