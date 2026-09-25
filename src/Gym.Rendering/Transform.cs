using System;

namespace Gym.Rendering {
    /// <summary>Port of the part of <c>pygame.transform</c> Gymnasium's envs use.</summary>
    public static class Transform {
        /// <summary>
        ///     <c>pygame.transform.flip(surface, flip_x, flip_y)</c>: a new surface, mirrored left-right and/or top-bottom. The
        ///     source is unchanged. Gymnasium's classic envs draw in a y-up world and flip vertically before returning a frame.
        /// </summary>
        /// <param name="surface">The surface to mirror.</param>
        /// <param name="flipX">Mirror the columns (left becomes right).</param>
        /// <param name="flipY">Mirror the rows (top becomes bottom).</param>
        /// <returns>The mirrored copy.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="surface"/> is <see langword="null"/>.</exception>
        public static Surface Flip(Surface surface, bool flipX, bool flipY) {
            if (surface is null)
                throw new ArgumentNullException(nameof(surface));
            var flipped = new Surface(surface.Width, surface.Height);
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
    }
}
