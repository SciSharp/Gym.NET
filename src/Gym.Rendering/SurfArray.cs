using System;
using NumSharp;

namespace Gym.Rendering {
    /// <summary>
    ///     The part of <c>pygame.surfarray</c> Gymnasium's envs use to hand out <c>rgb_array</c> frames: a surface's pixels as
    ///     a uint8 array in pygame's axis order, or already transposed into the frame an env returns.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/pygame-community/pygame-ce/blob/2.5.8/src_py/surfarray.py#L187">src_py/surfarray.py</see> (pygame-ce 2.5.8).</remarks>
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
        ///     <c>pygame.surfarray.make_surface(array)</c>: a new surface holding an array's pixels in pygame's <c>[x][y]</c>
        ///     order, so a <c>(width, height, 3)</c> array gives a <c>width</c> by <c>height</c> surface. <c>play</c> shows each
        ///     frame through it, after swapping a frame's first two axes.
        /// </summary>
        /// <param name="array">
        ///     A uint8 array: <c>(width, height, 3)</c> RGB, or <c>(width, height)</c>, whose values pygame reads as indices
        ///     into its 8-bit palette (not as gray levels; see the remarks). Views (a transposed frame) are read as they are.
        /// </param>
        /// <returns>A new opaque surface the caller owns.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="array"/> is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">
        ///     Thrown for any other shape, with pygame's message, which ends in a newline: <c>must be a valid 2d or 3d array</c>
        ///     followed by <c>\n</c>.
        /// </exception>
        /// <exception cref="NotSupportedException">Thrown for a dtype other than uint8: pygame converts those by rules no Gymnasium code relies on, which are not modelled.</exception>
        /// <remarks>
        ///     <para>
        ///     pygame makes an 8-bit palettized surface of a 2-D array, with a palette of period 128 measured on pygame-ce
        ///     2.5.8: index <c>i</c> is <c>(((i &gt;&gt; 5) &amp; 3) * 85, round(((i &gt;&gt; 2) &amp; 7) * 255 / 7), (i &amp; 3) * 85)</c>, so
        ///     100 is <c>(255, 36, 0)</c>. This surface holds those colours directly; scaling it and blitting it give the same
        ///     pixels as pygame's palettized one.
        ///     </para>
        ///     <para>Reference: <see href="https://github.com/pygame-community/pygame-ce/blob/2.5.8/src_c/pixelcopy.c">src_c/pixelcopy.c</see> (pygame-ce 2.5.8).</para>
        /// </remarks>
        public static Surface MakeSurface(NDArray array) {
            if (array is null)
                throw new ArgumentNullException(nameof(array));
            bool rgb = array.ndim == 3 && array.shape[2] == 3;
            if (!rgb && array.ndim != 2)
                // pygame's message ends in a newline (measured on pygame-ce 2.5.8).
                throw new ValueError("must be a valid 2d or 3d array\n");
            if (array.typecode != NPTypeCode.Byte)
                throw new NotSupportedException($"make_surface of a {array.dtype.name} array is not modelled: pygame's conversion of non-uint8 arrays is not one any Gymnasium code relies on.");

            int width = (int)array.shape[0], height = (int)array.shape[1];
            var surface = new Surface(width, height);
            // ToArray reads a view (a swapped frame) in logical C order: [x][y][c], x the slowest axis.
            byte[] bytes = array.ToArray<byte>();
            uint[] pixels = surface.Pixels;
            for (int x = 0, j = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    if (rgb) {
                        pixels[y * width + x] = ((uint)bytes[j] << 16) | ((uint)bytes[j + 1] << 8) | bytes[j + 2];
                        j += 3;
                    } else {
                        pixels[y * width + x] = PaletteColor[bytes[j]];
                        j++;
                    }
                }
            }

            return surface;
        }

        /// <summary>pygame's 8-bit palette for <see cref="MakeSurface"/> of a 2-D array, as <c>0x00RRGGBB</c> words (see its remarks).</summary>
        private static readonly uint[] PaletteColor = BuildPalette();

        /// <summary>Builds <see cref="PaletteColor"/>: red from bits 5-6, green from bits 2-4 scaled to 255 and rounded, blue from bits 0-1.</summary>
        /// <returns>The 256 colours.</returns>
        private static uint[] BuildPalette() {
            var palette = new uint[256];
            for (int i = 0; i < 256; i++) {
                uint red = (uint)(((i >> 5) & 3) * 85);
                // round(g * 255 / 7) never lands on a half (255 / 7 has no factor of 2 in its denominator), so any rounding agrees.
                uint green = (uint)Math.Round(((i >> 2) & 7) * 255 / 7.0);
                uint blue = (uint)((i & 3) * 85);
                palette[i] = (red << 16) | (green << 8) | blue;
            }

            return palette;
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
