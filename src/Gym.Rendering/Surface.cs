using System;

namespace Gym.Rendering {
    /// <summary>
    ///     A drawing surface as Gymnasium's envs create one with <c>pygame.Surface((width, height))</c>: 32 bits per pixel
    ///     without per-pixel alpha (SDL's XRGB8888, masks <c>0xff0000</c>, <c>0xff00</c>, <c>0xff</c>, alpha mask 0), black
    ///     when created, with pygame's <c>(x, y)</c> addressing and its whole area as the clip rectangle. A surface with
    ///     per-pixel alpha (<see cref="HasAlpha"/>) is what <see cref="ImageIO.Load"/> gives for a PNG with an alpha channel,
    ///     and what <c>transform.scale</c>/<c>smoothscale</c>/<c>flip</c> give for one.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Pixels are kept as SDL keeps them, one <c>0x00RRGGBB</c> word each, row by row, so the drawing primitives read and
    ///     write exactly the words SDL_gfx does. Without per-pixel alpha there is no alpha channel: an RGBA colour's alpha only
    ///     weights a blend with the pixel underneath, it is never stored.
    ///     </para>
    ///     <para>
    ///     With per-pixel alpha each word is <c>0xAARRGGBB</c>. pygame stores such a surface in its own channel order
    ///     (a loaded PNG is ABGR8888); only the values matter here, since every operation works channel by channel. What is
    ///     modelled for them is what Gymnasium's envs do with images: loading, <see cref="Transform"/>'s scaling and flipping,
    ///     and blitting onto a surface without alpha. Drawing on them, blitting onto them, and blitting an opaque surface onto
    ///     them are refused (<see cref="NotSupportedException"/>) rather than guessed.
    ///     </para>
    ///     <para>Not thread-safe: draw on a surface from one thread at a time, as with pygame.</para>
    ///     <para>Reference: <see href="https://github.com/pygame-community/pygame-ce/blob/2.5.8/src_c/surface.c#L471">src_c/surface.c</see> (pygame-ce 2.5.8).</para>
    /// </remarks>
    public sealed class Surface {
        /// <summary>
        ///     The pixels, <c>0x00RRGGBB</c> (or <c>0xAARRGGBB</c> with <see cref="HasAlpha"/>), row-major: pixel <c>(x, y)</c>
        ///     is at <c>y * Width + x</c>.
        /// </summary>
        private readonly uint[] _pixels;

        /// <summary>Creates a black surface, <c>pygame.Surface((width, height))</c>.</summary>
        /// <param name="width">Columns; 0 is allowed, as in pygame (nothing can be drawn on it).</param>
        /// <param name="height">Rows; 0 is allowed.</param>
        /// <exception cref="ArgumentOutOfRangeException">
        ///     Thrown for a negative size (pygame raises <c>pygame.error: Invalid resolution for Surface</c>) or one whose
        ///     pixel count does not fit an array.
        /// </exception>
        public Surface(int width, int height) : this(width, height, hasAlpha: false) { }

        /// <summary>
        ///     Creates a surface, <c>pygame.Surface((width, height))</c>, or with <paramref name="hasAlpha"/>
        ///     <c>pygame.Surface((width, height), pygame.SRCALPHA)</c>, which starts fully transparent (every pixel
        ///     <c>(0, 0, 0, 0)</c>, as SDL zero-fills a new surface).
        /// </summary>
        /// <param name="width">Columns; 0 is allowed, as in pygame (nothing can be drawn on it).</param>
        /// <param name="height">Rows; 0 is allowed.</param>
        /// <param name="hasAlpha">Whether each pixel carries its own alpha (<c>pygame.SRCALPHA</c>).</param>
        /// <exception cref="ArgumentOutOfRangeException">
        ///     Thrown for a negative size (pygame raises <c>pygame.error: Invalid resolution for Surface</c>) or one whose
        ///     pixel count does not fit an array.
        /// </exception>
        public Surface(int width, int height, bool hasAlpha) {
            if (width < 0)
                throw new ArgumentOutOfRangeException(nameof(width), width, "Invalid resolution for Surface");
            if (height < 0)
                throw new ArgumentOutOfRangeException(nameof(height), height, "Invalid resolution for Surface");
            long count = (long)width * height;
            if (count > Array.MaxLength)
                throw new ArgumentOutOfRangeException(nameof(width), $"A {width}x{height} surface does not fit in memory.");
            Width = width;
            Height = height;
            HasAlpha = hasAlpha;
            _pixels = new uint[count];
        }

        /// <summary>The number of columns, <c>surface.get_width()</c>.</summary>
        public int Width { get; }

        /// <summary>The number of rows, <c>surface.get_height()</c>.</summary>
        public int Height { get; }

        /// <summary>
        ///     Whether each pixel carries its own alpha (<c>surface.get_flags() &amp; pygame.SRCALPHA</c>): blitting the surface
        ///     then blends each pixel by its alpha instead of copying it.
        /// </summary>
        public bool HasAlpha { get; }

        /// <summary>
        ///     The raw pixel words, for the primitives and array views of this assembly. Writing through it bypasses no
        ///     bookkeeping, but callers must keep the word form: <c>0x00RRGGBB</c> (the top byte stays zero, as SDL writes it)
        ///     without per-pixel alpha, <c>0xAARRGGBB</c> with it.
        /// </summary>
        internal uint[] Pixels => _pixels;

        /// <summary>
        ///     <c>surface.fill(color)</c>: every pixel becomes the colour's RGB, and with per-pixel alpha its alpha too. Without
        ///     it, alpha is ignored, as <c>SDL_FillRect</c> maps the colour into a format that has no alpha.
        /// </summary>
        /// <param name="color">The colour.</param>
        public void Fill(Color color) => Array.Fill(_pixels, HasAlpha ? color.Mapped | ((uint)color.A << 24) : color.Mapped);

        /// <summary>
        ///     <c>surface.get_at((x, y))</c>: the pixel's colour, opaque when the surface has no per-pixel alpha (the format
        ///     has no alpha to report), else with the pixel's own alpha.
        /// </summary>
        /// <param name="x">The column.</param>
        /// <param name="y">The row.</param>
        /// <returns>The colour.</returns>
        /// <exception cref="IndexOutOfRangeException">Thrown outside the surface, as pygame raises <c>IndexError: pixel index out of range</c>.</exception>
        public Color GetAt(int x, int y) {
            if (x < 0 || x >= Width || y < 0 || y >= Height)
                throw new IndexOutOfRangeException("pixel index out of range");
            uint pixel = _pixels[y * Width + x];
            return new Color((byte)(pixel >> 16), (byte)(pixel >> 8), (byte)pixel, HasAlpha ? (byte)(pixel >> 24) : (byte)255);
        }

        /// <summary>
        ///     <c>surface.blit(source, (x, y))</c> onto this surface (which has no per-pixel alpha), clipped to it. An opaque
        ///     source (no colour key, no surface alpha, the only kind this assembly creates) is copied; a source with per-pixel
        ///     alpha (a loaded PNG) is blended, each channel as <c>d + (((s - d) * a + s) &gt;&gt; 8)</c> with an arithmetic
        ///     shift, which is what pygame-ce 2.5.8 computes for this case (measured on all 2^24 source, alpha and destination
        ///     values, every channel): alpha 0 keeps the destination and alpha 255 copies the source.
        /// </summary>
        /// <param name="source">The surface to copy or blend from.</param>
        /// <param name="x">Where the source's left column lands.</param>
        /// <param name="y">Where the source's top row lands.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="source"/> is <see langword="null"/>.</exception>
        /// <exception cref="NotSupportedException">
        ///     Thrown when this surface has per-pixel alpha: pygame blends onto such a destination with other rules (its
        ///     alpha is composited too), which no Gymnasium env relies on, so they are not modelled.
        /// </exception>
        public void Blit(Surface source, int x, int y) {
            if (source is null)
                throw new ArgumentNullException(nameof(source));
            if (HasAlpha)
                throw new NotSupportedException("Blitting onto a surface with per-pixel alpha is not modelled: no Gymnasium env does it.");
            // Clip the destination rectangle to this surface, then copy row by row; long arithmetic keeps far-off
            // positions from overflowing into a false overlap.
            long left = Math.Max(0L, x), top = Math.Max(0L, y);
            long right = Math.Min((long)Width, (long)x + source.Width), bottom = Math.Min((long)Height, (long)y + source.Height);
            if (left >= right || top >= bottom)
                return;
            int columns = (int)(right - left);
            for (long row = top; row < bottom; row++) {
                int from = (int)((row - y) * source.Width + (left - x));
                int to = (int)(row * Width + left);
                if (!source.HasAlpha) {
                    Array.Copy(source._pixels, from, _pixels, to, columns);
                    continue;
                }

                for (int i = 0; i < columns; i++)
                    _pixels[to + i] = BlendOver(_pixels[to + i], source._pixels[from + i]);
            }
        }

        /// <summary>
        ///     The pixels as <c>(height, width, 3)</c> RGB bytes, row-major: the byte layout of Gymnasium's <c>rgb_array</c>
        ///     frames, <c>np.transpose(np.array(surfarray.pixels3d(surface)), axes=(1, 0, 2))</c>. A per-pixel alpha is left out,
        ///     as <c>pixels3d</c> leaves it out.
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

        /// <summary>
        ///     The pixels as <c>(height, width, 4)</c> RGBA bytes, row-major: what <c>pixels3d</c> and <c>pixels_alpha</c>
        ///     together hold for a surface with per-pixel alpha, so an image's every channel can be compared. Without per-pixel
        ///     alpha the alpha byte is 255, as <c>get_at</c> reports it.
        /// </summary>
        /// <returns>A new array of <c>Height * Width * 4</c> bytes.</returns>
        public byte[] ToRgbaBytes() {
            var rgba = new byte[_pixels.Length * 4];
            for (int i = 0, j = 0; i < _pixels.Length; i++, j += 4) {
                uint pixel = _pixels[i];
                rgba[j] = (byte)(pixel >> 16);
                rgba[j + 1] = (byte)(pixel >> 8);
                rgba[j + 2] = (byte)pixel;
                rgba[j + 3] = HasAlpha ? (byte)(pixel >> 24) : (byte)255;
            }

            return rgba;
        }

        /// <summary>
        ///     One destination pixel (no alpha) after blending one source pixel (<c>0xAARRGGBB</c>) over it, channel by
        ///     channel: <c>d + (((s - d) * a + s) &gt;&gt; 8)</c>. The shift is arithmetic (it floors), so a darker source pulls the
        ///     channel down by one more than a truncating division would.
        /// </summary>
        /// <param name="destination">The destination word, <c>0x00RRGGBB</c>.</param>
        /// <param name="source">The source word, <c>0xAARRGGBB</c>.</param>
        /// <returns>The blended word, <c>0x00RRGGBB</c>.</returns>
        private static uint BlendOver(uint destination, uint source) {
            int alpha = (int)(source >> 24);
            uint result = 0;
            for (int shift = 0; shift <= 16; shift += 8) {
                int s = (int)((source >> shift) & 0xFF), d = (int)((destination >> shift) & 0xFF);
                // (s - d) * a + s can be negative; C# >> on int is arithmetic, as the measured behaviour requires.
                int blended = d + ((((s - d) * alpha) + s) >> 8);
                result |= (uint)blended << shift;
            }

            return result;
        }
    }
}
