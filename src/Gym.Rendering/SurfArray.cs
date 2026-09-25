using System;
using NumSharp;

namespace Gym.Rendering {
    /// <summary>Port of the part of <c>pygame.surfarray</c> Gymnasium's envs use to hand out <c>rgb_array</c> frames.</summary>
    public static class SurfArray {
        /// <summary>
        ///     <c>pygame.surfarray.array3d(surface)</c>, equivalently <c>np.array(pygame.surfarray.pixels3d(surface))</c>: a
        ///     <c>(width, height, 3)</c> uint8 copy of the pixels in pygame's <c>[x][y]</c> order.
        /// </summary>
        /// <param name="surface">The surface.</param>
        /// <returns>A new array the caller owns (dispose it).</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="surface"/> is <see langword="null"/>.</exception>
        /// <remarks>
        ///     Gymnasium's envs transpose it to <c>(height, width, 3)</c>; <see cref="Surface.ToRgbBytes"/> and
        ///     <see cref="ToRgbArray"/> give that frame directly.
        /// </remarks>
        public static NDArray Array3d(Surface surface) {
            if (surface is null)
                throw new ArgumentNullException(nameof(surface));
            int width = surface.Width, height = surface.Height;
            uint[] pixels = surface.Pixels;
            var bytes = new byte[pixels.Length * 3];
            // [x][y][c] order: x is the slowest axis.
            for (int x = 0, j = 0; x < width; x++) {
                for (int y = 0; y < height; y++, j += 3) {
                    uint pixel = pixels[y * width + x];
                    bytes[j] = (byte)(pixel >> 16);
                    bytes[j + 1] = (byte)(pixel >> 8);
                    bytes[j + 2] = (byte)pixel;
                }
            }

            // The fresh buffer becomes the array's storage: nothing else holds it, so no copy is needed.
            return new NDArray(bytes, new Shape(width, height, 3));
        }

        /// <summary>
        ///     The frame Gymnasium's <c>render()</c> returns for <c>render_mode="rgb_array"</c>,
        ///     <c>np.transpose(np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2))</c>: <c>(height, width, 3)</c>
        ///     uint8, C-contiguous as NumPy's is.
        /// </summary>
        /// <param name="surface">The surface.</param>
        /// <returns>A new array the caller owns (dispose it).</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="surface"/> is <see langword="null"/>.</exception>
        public static NDArray ToRgbArray(Surface surface) {
            if (surface is null)
                throw new ArgumentNullException(nameof(surface));
            return new NDArray(surface.ToRgbBytes(), new Shape(surface.Height, surface.Width, 3));
        }
    }
}
