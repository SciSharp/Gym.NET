#nullable enable
using System;
using NumSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Gym.Rendering {
    /// <summary>
    ///     The frame check and conversion both viewer packages share (this file is compiled into each, as
    ///     <c>ScottPlotAxes.cs</c> is): the viewer contract's <c>(H, W, 3)</c> frame (DR-85) as an ImageSharp image, which the
    ///     viewers then turn into their toolkit's bitmap. ImageSharp is the viewers' own business, not the contract's.
    /// </summary>
    internal static class ViewerFrames {
        /// <summary>
        ///     Checks a frame as every viewer does: present, three axes, three channels, and not empty. The dtype is not
        ///     checked here: <see cref="ToImage"/> casts another dtype to uint8.
        /// </summary>
        /// <param name="frame">The frame.</param>
        /// <param name="parameterName">The caller's parameter name, for the exception.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="frame"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown for a frame that is not <c>(H, W, 3)</c> with both sides positive.</exception>
        public static void Check(NDArray frame, string parameterName = "frame") {
            if (frame is null)
                throw new ArgumentNullException(parameterName);
            if (frame.ndim != 3 || frame.shape[2] != 3 || frame.shape[0] <= 0 || frame.shape[1] <= 0)
                throw new ArgumentException($"A viewer shows (H, W, 3) frames, got shape ({string.Join(", ", frame.shape)}).", parameterName);
        }

        /// <summary>
        ///     A frame as a new RGB image the caller disposes: the frame's bytes in C order (another dtype is cast to uint8,
        ///     as pygame reads a frame), so the image is the caller's to keep however the frame is laid out or reused later.
        /// </summary>
        /// <param name="frame">The frame, <c>(H, W, 3)</c>; it is read, not changed or kept.</param>
        /// <returns>A new <c>Image&lt;Rgb24&gt;</c> of the frame's size.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="frame"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown for a frame that is not <c>(H, W, 3)</c>.</exception>
        public static Image<Rgb24> ToImage(NDArray frame) {
            Check(frame);
            byte[] bytes;
            if (frame.typecode == NPTypeCode.Byte) {
                // ToArray copies in C order whatever the frame's strides (a transposed view included).
                bytes = frame.ToArray<byte>();
            } else {
                using NDArray cast = frame.astype(np.uint8);
                bytes = cast.ToArray<byte>();
            }

            return Image.LoadPixelData<Rgb24>(bytes, (int)frame.shape[1], (int)frame.shape[0]);
        }

        /// <summary>
        ///     A frame as opaque 32-bit BGRA pixels, rows top first with no padding: the layout both toolkits take without a
        ///     conversion of their own (Avalonia's <c>Bgra8888</c>, GDI+'s <c>Format32bppRgb</c>), so the window shows the
        ///     frame's very bytes (DR-99). The alpha byte is 255 everywhere.
        /// </summary>
        /// <param name="frame">The frame, <c>(H, W, 3)</c>; another dtype is cast to uint8, as <see cref="ToImage"/> does. Read, never kept.</param>
        /// <param name="width">The frame's width in pixels.</param>
        /// <param name="height">The frame's height in pixels.</param>
        /// <returns>A new array of <c>4 × width × height</c> bytes the caller owns.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="frame"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown for a frame that is not <c>(H, W, 3)</c>.</exception>
        /// <remarks>
        ///     Footgun: the result is the frame's copy at the time of the call; a viewer that keeps it to redraw later (a move
        ///     to a monitor of another scale) redraws that copy, never the caller's array, which the caller may reuse.
        /// </remarks>
        public static byte[] ToBgra(NDArray frame, out int width, out int height) {
            Check(frame);
            byte[] rgb;
            if (frame.typecode == NPTypeCode.Byte) {
                // ToArray copies in C order whatever the frame's strides (a transposed view included).
                rgb = frame.ToArray<byte>();
            } else {
                using NDArray cast = frame.astype(np.uint8);
                rgb = cast.ToArray<byte>();
            }

            width = (int)frame.shape[1];
            height = (int)frame.shape[0];
            var bgra = new byte[4 * width * height];
            for (int i = 0, o = 0; i < rgb.Length; i += 3, o += 4) {
                bgra[o] = rgb[i + 2];
                bgra[o + 1] = rgb[i + 1];
                bgra[o + 2] = rgb[i];
                bgra[o + 3] = 255;
            }

            return bgra;
        }
    }
}
