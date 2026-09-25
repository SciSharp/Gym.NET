using System;

namespace Gym.Rendering {
    /// <summary>
    ///     Port of <c>pygame.Surface((width, height))</c> as Gymnasium's envs create it: a 32-bit surface without per-pixel
    ///     alpha (SDL's XRGB8888, masks <c>0xff0000</c>, <c>0xff00</c>, <c>0xff</c>, alpha mask 0), black when created, with
    ///     pygame's <c>(x, y)</c> addressing and its whole area as the clip rectangle.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Pixels are kept as SDL keeps them, one <c>0x00RRGGBB</c> word each, row by row, so the drawing primitives read and
    ///     write exactly the words SDL_gfx does. There is no alpha channel: an RGBA colour's alpha only weights a blend with
    ///     the pixel underneath, it is never stored. Surfaces with per-pixel alpha (<c>pygame.SRCALPHA</c>) are a different
    ///     format with different blending, and are not modelled.
    ///     </para>
    ///     <para>Not thread-safe: draw on a surface from one thread at a time, as with pygame.</para>
    /// </remarks>
    public sealed class Surface {
        /// <summary>The pixels, <c>0x00RRGGBB</c>, row-major: pixel <c>(x, y)</c> is at <c>y * Width + x</c>.</summary>
        private readonly uint[] _pixels;

        /// <summary>Creates a black surface, <c>pygame.Surface((width, height))</c>.</summary>
        /// <param name="width">Columns; 0 is allowed, as in pygame (nothing can be drawn on it).</param>
        /// <param name="height">Rows; 0 is allowed.</param>
        /// <exception cref="ArgumentOutOfRangeException">
        ///     Thrown for a negative size (pygame raises <c>pygame.error: Invalid resolution for Surface</c>) or one whose
        ///     pixel count does not fit an array.
        /// </exception>
        public Surface(int width, int height) {
            if (width < 0)
                throw new ArgumentOutOfRangeException(nameof(width), width, "Invalid resolution for Surface");
            if (height < 0)
                throw new ArgumentOutOfRangeException(nameof(height), height, "Invalid resolution for Surface");
            long count = (long)width * height;
            if (count > Array.MaxLength)
                throw new ArgumentOutOfRangeException(nameof(width), $"A {width}x{height} surface does not fit in memory.");
            Width = width;
            Height = height;
            _pixels = new uint[count];
        }

        /// <summary>The number of columns, <c>surface.get_width()</c>.</summary>
        public int Width { get; }

        /// <summary>The number of rows, <c>surface.get_height()</c>.</summary>
        public int Height { get; }

        /// <summary>
        ///     The raw pixel words, for the primitives and array views of this assembly. Writing through it bypasses no
        ///     bookkeeping, but callers must keep the <c>0x00RRGGBB</c> form (the top byte stays zero, as SDL writes it).
        /// </summary>
        internal uint[] Pixels => _pixels;

        /// <summary>
        ///     <c>surface.fill(color)</c>: every pixel becomes the colour's RGB. Alpha is ignored, as <c>SDL_FillRect</c> maps
        ///     the colour into a format that has no alpha.
        /// </summary>
        /// <param name="color">The colour.</param>
        public void Fill(Color color) => Array.Fill(_pixels, color.Mapped);

        /// <summary><c>surface.get_at((x, y))</c>: the pixel's colour, opaque (the format has no alpha to report).</summary>
        /// <param name="x">The column.</param>
        /// <param name="y">The row.</param>
        /// <returns>The colour, alpha 255.</returns>
        /// <exception cref="IndexOutOfRangeException">Thrown outside the surface, as pygame raises <c>IndexError: pixel index out of range</c>.</exception>
        public Color GetAt(int x, int y) {
            if (x < 0 || x >= Width || y < 0 || y >= Height)
                throw new IndexOutOfRangeException("pixel index out of range");
            uint pixel = _pixels[y * Width + x];
            return new Color((byte)(pixel >> 16), (byte)(pixel >> 8), (byte)pixel);
        }

        /// <summary>
        ///     <c>surface.blit(source, (x, y))</c> for an opaque source (no colour key, no surface alpha, the only kind this
        ///     assembly creates): the source's pixels are copied, clipped to this surface.
        /// </summary>
        /// <param name="source">The surface to copy from.</param>
        /// <param name="x">Where the source's left column lands.</param>
        /// <param name="y">Where the source's top row lands.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="source"/> is <see langword="null"/>.</exception>
        public void Blit(Surface source, int x, int y) {
            if (source is null)
                throw new ArgumentNullException(nameof(source));
            // Clip the destination rectangle to this surface, then copy row by row; long arithmetic keeps far-off
            // positions from overflowing into a false overlap.
            long left = Math.Max(0L, x), top = Math.Max(0L, y);
            long right = Math.Min((long)Width, (long)x + source.Width), bottom = Math.Min((long)Height, (long)y + source.Height);
            if (left >= right || top >= bottom)
                return;
            int columns = (int)(right - left);
            for (long row = top; row < bottom; row++) {
                int from = (int)((row - y) * source.Width + (left - x));
                Array.Copy(source._pixels, from, _pixels, (int)(row * Width + left), columns);
            }
        }

        /// <summary>
        ///     The pixels as <c>(height, width, 3)</c> RGB bytes, row-major: the byte layout of Gymnasium's <c>rgb_array</c>
        ///     frames, <c>np.transpose(np.array(surfarray.pixels3d(surface)), axes=(1, 0, 2))</c>.
        /// </summary>
        /// <returns>A new array of <c>Height * Width * 3</c> bytes.</returns>
        public byte[] ToRgbBytes() {
            var rgb = new byte[_pixels.Length * 3];
            for (int i = 0, j = 0; i < _pixels.Length; i++, j += 3) {
                uint pixel = _pixels[i];
                rgb[j] = (byte)(pixel >> 16);
                rgb[j + 1] = (byte)(pixel >> 8);
                rgb[j + 2] = (byte)pixel;
            }

            return rgb;
        }
    }
}
