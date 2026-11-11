using System;
using System.Buffers;
using NumSharp;

namespace Gym.Rendering {
    /// <summary>
    ///     The part of <c>pygame.transform</c> Gymnasium's envs use: <see cref="Flip"/>, which turns a y-up drawing into a
    ///     frame and mirrors sprites, and <see cref="Scale"/>/<see cref="Smoothscale"/>, which size the toy_text tiles and
    ///     Pendulum's torque arrow.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <see cref="Scale"/> and <see cref="Smoothscale"/> are Gym.NET's own implementations of pygame's behaviour (decision
    ///     D4: no pygame code is used), derived from and proven against live pygame-ce 2.5.8 pixel for pixel, alpha included.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/pygame-community/pygame-ce/blob/2.5.8/src_c/transform.c#L770">src_c/transform.c</see> (pygame-ce 2.5.8).</para>
    /// </remarks>
    public static class Transform {
        /// <summary>One source pixel in the 2.14 fixed point of <see cref="Smoothscale"/>.</summary>
        private const int SmoothOne = 0x4000;

        /// <summary>
        ///     <c>pygame.transform.flip(surface, flip_x, flip_y)</c>: a new surface, mirrored left-right and/or top-bottom. The
        ///     source is unchanged. Gymnasium's classic envs draw in a y-up world and flip vertically before returning a frame;
        ///     Taxi and Pendulum mirror their sprites. Per-pixel alpha is kept.
        /// </summary>
        /// <param name="surface">The surface to mirror.</param>
        /// <param name="flipX">Mirror the columns (left becomes right).</param>
        /// <param name="flipY">Mirror the rows (top becomes bottom).</param>
        /// <returns>The mirrored copy.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="surface"/> is <see langword="null"/>.</exception>
        public static Surface Flip(Surface surface, bool flipX, bool flipY) {
            if (surface is null)
                throw new ArgumentNullException(nameof(surface));
            var flipped = new Surface(surface.Width, surface.Height, surface.HasAlpha);
            uint[] source = surface.Pixels, target = flipped.Pixels;
            int width = surface.Width, height = surface.Height;
            for (int y = 0; y < height; y++) {
                int from = (flipY ? height - 1 - y : y) * width;
                int to = y * width;
                if (!flipX) {
                    Array.Copy(source, from, target, to, width);
                    continue;
                }

                for (int x = 0; x < width; x++)
                    target[to + x] = source[from + width - 1 - x];
            }

            return flipped;
        }

        /// <summary>
        ///     <c>pygame.transform.scale(surface, (width, height))</c>: a new surface of the given size, each pixel a copy of
        ///     one source pixel (nearest neighbour, no blending), per-pixel alpha kept. Gymnasium's toy_text envs size their
        ///     tiles and sprites with it.
        /// </summary>
        /// <param name="surface">The surface to scale.</param>
        /// <param name="width">The new width; 0 gives an empty surface. pygame truncates a float size (<c>int()</c>), as callers must.</param>
        /// <param name="height">The new height; 0 gives an empty surface.</param>
        /// <returns>The scaled copy.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="surface"/> is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">Thrown for a negative size, with pygame's message (<c>Cannot scale to negative size</c>).</exception>
        /// <remarks>
        ///     Destination column <c>x</c> takes source column <c>(x * step + step / 2) &gt;&gt; 16</c> with
        ///     <c>step = (sourceWidth &lt;&lt; 16) / width</c> in integer arithmetic, and rows likewise: sampling at the middle of each
        ///     16.16 fixed-point step, which is what pygame-ce 2.5.8 picks (measured on 2,011 size pairs, shrinking and growing).
        /// </remarks>
        public static Surface Scale(Surface surface, int width, int height) {
            if (surface is null)
                throw new ArgumentNullException(nameof(surface));
            if (width < 0 || height < 0)
                throw new ValueError("Cannot scale to negative size");
            var scaled = new Surface(width, height, surface.HasAlpha);
            if (width == 0 || height == 0 || surface.Width == 0 || surface.Height == 0)
                return scaled;

            int[] columns = NearestIndices(surface.Width, width);
            int[] rows = NearestIndices(surface.Height, height);
            uint[] source = surface.Pixels, target = scaled.Pixels;
            for (int y = 0; y < height; y++) {
                int from = rows[y] * surface.Width, to = y * width;
                for (int x = 0; x < width; x++)
                    target[to + x] = source[from + columns[x]];
            }

            return scaled;
        }

        /// <summary>
        ///     <c>pygame.transform.smoothscale(surface, (width, height))</c> for a shrinking size: a new surface whose pixels
        ///     average the source pixels they cover (area averaging), each channel of each pixel separately, per-pixel alpha
        ///     included. Pendulum shrinks its torque arrow with it every frame.
        /// </summary>
        /// <param name="surface">The surface to scale.</param>
        /// <param name="width">The new width, at most the source's; 0 gives an empty surface. pygame truncates a float size (<c>int()</c>).</param>
        /// <param name="height">The new height, at most the source's; 0 gives an empty surface.</param>
        /// <returns>The scaled copy.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="surface"/> is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">Thrown for a negative size, with pygame's message (<c>Cannot scale to negative size</c>).</exception>
        /// <exception cref="NotSupportedException">
        ///     Thrown when either dimension grows: pygame expands with a different (interpolating) filter, which no env in this
        ///     release needs, so it is deferred rather than approximated.
        /// </exception>
        /// <remarks>
        ///     <para>
        ///     The filter, as measured against pygame-ce 2.5.8 (exact on 800 random rows and 300 random images, shrinking one or
        ///     both axes): the horizontal pass runs first (every row narrowed), then the vertical one, each pass writing 8-bit
        ///     channels. Along an axis of <c>n</c>
        ///     pixels shrunk to <c>d</c>, a destination pixel spans <c>span = 0x4000 * n / d</c> source pixels in 2.14 fixed
        ///     point; a whole source pixel adds its value, the pixel straddling a boundary adds <c>(v * w) &gt;&gt; 14</c> to each
        ///     side for its 14-bit weight <c>w</c>, and a finished pixel is <c>(sum * (2^30 / span)) &gt;&gt; 16</c>. The 14-bit
        ///     weights and the truncations are what make the result differ from an exact average (a constant 255 row shrinks
        ///     to 254), and they are kept on purpose.
        ///     </para>
        /// </remarks>
        public static Surface Smoothscale(Surface surface, int width, int height) {
            if (surface is null)
                throw new ArgumentNullException(nameof(surface));
            if (width < 0 || height < 0)
                throw new ValueError("Cannot scale to negative size");
            if (width > surface.Width || height > surface.Height)
                throw new NotSupportedException(
                    $"smoothscale from {surface.Width}x{surface.Height} to {width}x{height} expands, which is not implemented yet: only shrinking (or keeping) each dimension is.");
            var scaled = new Surface(width, height, surface.HasAlpha);
            if (width == 0 || height == 0)
                return scaled;

            // Horizontal pass into a temporary of the new width and the old height, then the vertical pass. The temporary is
            // rented: an env that renders every step (CarRacing shrinks 1000x800 to 96x96) would otherwise allocate it each
            // time, often on the large object heap. The horizontal pass writes every element before the vertical pass reads
            // any, so a rented array's old contents never show through.
            uint[] horizontal = ArrayPool<uint>.Shared.Rent(checked(width * surface.Height));
            try {
                for (int y = 0; y < surface.Height; y++)
                    ShrinkLine(surface.Pixels, y * surface.Width, 1, surface.Width, horizontal, y * width, 1, width);
                for (int x = 0; x < width; x++)
                    ShrinkLine(horizontal, x, width, surface.Height, scaled.Pixels, x, width, height);
            } finally {
                ArrayPool<uint>.Shared.Return(horizontal);
            }

            return scaled;
        }

        /// <summary>
        ///     For each destination index, the source index <see cref="Scale"/> copies: the middle of each 16.16 step.
        /// </summary>
        /// <param name="source">The source size, positive.</param>
        /// <param name="destination">The destination size, positive.</param>
        /// <returns>The source index of each destination index.</returns>
        private static int[] NearestIndices(int source, int destination) {
            long step = ((long)source << 16) / destination;
            var indices = new int[destination];
            for (long i = 0; i < destination; i++)
                indices[i] = (int)((i * step + (step >> 1)) >> 16);
            return indices;
        }

        /// <summary>
        ///     Shrinks one line of pixels (a row or a column) with <see cref="Smoothscale"/>'s 2.14 fixed-point area filter,
        ///     all four channels of each word independently. When the length is unchanged the line is copied as it is.
        /// </summary>
        /// <param name="source">The source words.</param>
        /// <param name="sourceStart">The index of the line's first pixel.</param>
        /// <param name="sourceStride">The distance between consecutive pixels of the line (1 for a row, the width for a column).</param>
        /// <param name="sourceLength">The line's length, at least <paramref name="destinationLength"/>.</param>
        /// <param name="destination">The destination words.</param>
        /// <param name="destinationStart">The index of the destination line's first pixel.</param>
        /// <param name="destinationStride">The distance between consecutive destination pixels.</param>
        /// <param name="destinationLength">The destination line's length, positive.</param>
        private static void ShrinkLine(uint[] source, long sourceStart, long sourceStride, int sourceLength, uint[] destination, long destinationStart,
            long destinationStride, int destinationLength) {
            if (sourceLength == destinationLength) {
                for (long i = 0; i < sourceLength; i++)
                    destination[destinationStart + i * destinationStride] = source[sourceStart + i * sourceStride];
                return;
            }

            long span = (long)SmoothOne * sourceLength / destinationLength;
            long reciprocal = (1L << 30) / span;
            // Four channel accumulators, one per byte of the word, kept apart because each channel is filtered on its own.
            // Locals rather than a span indexed per channel: this loop runs for every pixel of every frame (800,000 of them
            // for CarRacing's 1000x800), and the unrolled form computes the same integers three times faster (R2 of
            // docs/plans/CSHARP_CODE_REVIEW.md).
            long s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            long left = span;
            int written = 0;
            for (long i = 0; i < sourceLength; i++) {
                uint pixel = source[sourceStart + i * sourceStride];
                long v0 = pixel & 0xFF, v1 = (pixel >> 8) & 0xFF, v2 = (pixel >> 16) & 0xFF, v3 = pixel >> 24;
                if (left > SmoothOne) {
                    // The whole source pixel lies inside the current destination pixel.
                    s0 += v0;
                    s1 += v1;
                    s2 += v2;
                    s3 += v3;
                    left -= SmoothOne;
                    continue;
                }

                // The pixel straddles the boundary: `left` of it finishes this destination pixel, the rest starts the next.
                // No mask needed: a sum is at most 255 * span / 0x4000 and the reciprocal at most 2^30 / span, so each
                // channel's result is at most 255 and cannot spill into the next channel's byte.
                long fraction = SmoothOne - left;
                uint finished = (uint)(((s0 + ((v0 * left) >> 14)) * reciprocal) >> 16)
                                | (uint)(((s1 + ((v1 * left) >> 14)) * reciprocal) >> 16) << 8
                                | (uint)(((s2 + ((v2 * left) >> 14)) * reciprocal) >> 16) << 16
                                | (uint)(((s3 + ((v3 * left) >> 14)) * reciprocal) >> 16) << 24;
                s0 = (v0 * fraction) >> 14;
                s1 = (v1 * fraction) >> 14;
                s2 = (v2 * fraction) >> 14;
                s3 = (v3 * fraction) >> 14;

                if (written < destinationLength)
                    destination[destinationStart + written * destinationStride] = finished;
                written++;
                left = span - fraction;
            }

            // A line whose last boundary falls on its end leaves the final destination pixel in the accumulators.
            while (written < destinationLength) {
                destination[destinationStart + written * destinationStride] = (uint)((s0 * reciprocal) >> 16)
                                                                             | (uint)((s1 * reciprocal) >> 16) << 8
                                                                             | (uint)((s2 * reciprocal) >> 16) << 16
                                                                             | (uint)((s3 * reciprocal) >> 16) << 24;
                written++;
            }
        }
    }
}
