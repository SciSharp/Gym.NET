using System;
using System.IO;
// For GetPngMetadata (an extension method). This class is ImageIO, not Image, so it never clashes with ImageSharp's Image.
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;

namespace Gym.Rendering {
    /// <summary>
    ///     <c>pygame.image</c>, named <c>ImageIO</c> so it does not clash with ImageSharp's <c>Image</c> in code that uses
    ///     both. The part Gymnasium's envs use: <see cref="Load"/>, for the PNG sprites and tiles of the
    ///     toy_text envs and Pendulum's torque arrow, and <see cref="FromBytes"/>, which builds the same kind of surface from
    ///     raw pixels.
    /// </summary>
    /// <remarks>
    ///     PNG is lossless, so the pixels are the file's own; the decoder is ImageSharp (decision D18 keeps it), and the parity
    ///     suite proves its pixels equal pygame-ce 2.5.8's (SDL_image) for every asset the envs load. Reference:
    ///     <see href="https://github.com/pygame-community/pygame-ce/blob/2.5.8/src_c/imageext.c">src_c/imageext.c</see> (pygame-ce 2.5.8).
    /// </remarks>
    public static class ImageIO {
        /// <summary>
        ///     <c>pygame.image.load(path)</c> for an 8-bit PNG: a new surface with the image's pixels, with per-pixel alpha when
        ///     the PNG has an alpha channel (as every Gymnasium asset does), without it otherwise. A grey PNG loads as grey RGB.
        /// </summary>
        /// <param name="path">The file.</param>
        /// <returns>The surface; the caller owns it.</returns>
        /// <exception cref="ArgumentException">Thrown when <paramref name="path"/> is <see langword="null"/> or empty.</exception>
        /// <exception cref="FileNotFoundException">Thrown when the file does not exist (pygame raises <c>FileNotFoundError</c>).</exception>
        /// <exception cref="NotSupportedException">
        ///     Thrown for a file that is not a PNG, a palette PNG, or a bit depth other than 8: pygame loads those too, but
        ///     through conversions (palette surfaces, 16-bit stripping) no env needs, so they are refused rather than guessed.
        /// </exception>
        /// <exception cref="SixLabors.ImageSharp.InvalidImageContentException">Thrown for a corrupt file.</exception>
        public static Surface Load(string path) {
            if (string.IsNullOrEmpty(path))
                throw new ArgumentException("A file path is required.", nameof(path));
            if (!File.Exists(path))
                throw new FileNotFoundException($"No file '{path}' found.", path);

            using SixLabors.ImageSharp.Image<Rgba32> image = SixLabors.ImageSharp.Image.Load<Rgba32>(path, out IImageFormat format);
            if (!string.Equals(format.Name, "PNG", StringComparison.OrdinalIgnoreCase))
                throw new NotSupportedException($"Only PNG images are loaded; '{path}' is {format.Name}.");
            PngMetadata png = image.Metadata.GetPngMetadata();
            if (png.BitDepth is not PngBitDepth.Bit8)
                throw new NotSupportedException($"Only 8-bit PNGs are loaded; '{path}' has bit depth {png.BitDepth}.");
            bool hasAlpha = png.ColorType switch {
                PngColorType.RgbWithAlpha or PngColorType.GrayscaleWithAlpha => true,
                PngColorType.Rgb or PngColorType.Grayscale => false,
                _ => throw new NotSupportedException($"Only RGB and grey PNGs are loaded; '{path}' is {png.ColorType}."),
            };

            var surface = new Surface(image.Width, image.Height, hasAlpha);
            var rgba = new Rgba32[image.Width * image.Height];
            image.CopyPixelDataTo(rgba);
            uint[] pixels = surface.Pixels;
            for (int i = 0; i < rgba.Length; i++) {
                Rgba32 p = rgba[i];
                // The surface's word form: 0xAARRGGBB with per-pixel alpha, 0x00RRGGBB (top byte zero) without.
                pixels[i] = ((uint)p.R << 16) | ((uint)p.G << 8) | p.B | (hasAlpha ? (uint)p.A << 24 : 0u);
            }

            return surface;
        }

        /// <summary>
        ///     <c>pygame.image.frombytes(bytes, (width, height), format)</c> for <c>"RGBA"</c> and <c>"RGB"</c>: a new surface
        ///     from raw pixel bytes, row by row from the top, with per-pixel alpha for <c>"RGBA"</c> (the format a loaded PNG
        ///     has) and without it for <c>"RGB"</c>. The bytes are copied.
        /// </summary>
        /// <param name="bytes">The pixels, 4 (RGBA) or 3 (RGB) bytes each.</param>
        /// <param name="width">The width, positive.</param>
        /// <param name="height">The height, positive.</param>
        /// <param name="format"><c>"RGBA"</c> or <c>"RGB"</c>.</param>
        /// <returns>The surface; the caller owns it.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="bytes"/> or <paramref name="format"/> is <see langword="null"/>.</exception>
        /// <exception cref="NumSharp.ValueError">
        ///     Thrown with pygame's messages for a size that is not positive (<c>Resolution must be nonzero positive values</c>),
        ///     a byte count that does not match (<c>Bytes length does not equal format and resolution size</c>), or a format
        ///     pygame does not know (<c>Unrecognized type of format</c>).
        /// </exception>
        /// <exception cref="NotSupportedException">Thrown for a format pygame knows but this method does not model (<c>"ARGB"</c>, <c>"P"</c>, ...).</exception>
        public static Surface FromBytes(byte[] bytes, int width, int height, string format) {
            if (bytes is null)
                throw new ArgumentNullException(nameof(bytes));
            if (format is null)
                throw new ArgumentNullException(nameof(format));
            if (width <= 0 || height <= 0)
                throw new NumSharp.ValueError("Resolution must be nonzero positive values");
            int perPixel = format switch {
                "RGBA" => 4,
                "RGB" => 3,
                "P" or "RGBX" or "ARGB" or "BGRA" or "RGBA_PREMULT" or "ARGB_PREMULT" =>
                    throw new NotSupportedException($"pygame.image.frombytes format '{format}' is not modelled: only RGBA and RGB are."),
                _ => throw new NumSharp.ValueError("Unrecognized type of format"),
            };
            if ((long)bytes.Length != (long)width * height * perPixel)
                throw new NumSharp.ValueError("Bytes length does not equal format and resolution size");

            var surface = new Surface(width, height, hasAlpha: perPixel == 4);
            uint[] pixels = surface.Pixels;
            for (int i = 0, j = 0; i < pixels.Length; i++, j += perPixel) {
                uint word = ((uint)bytes[j] << 16) | ((uint)bytes[j + 1] << 8) | bytes[j + 2];
                pixels[i] = perPixel == 4 ? word | ((uint)bytes[j + 3] << 24) : word;
            }

            return surface;
        }
    }
}
