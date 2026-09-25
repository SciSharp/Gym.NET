using System;
using Gym.Tests.Parity.Oracle;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Gym.Tests.Parity.Frames {
    /// <summary>
    ///     An RGB frame in Gymnasium's <c>rgb_array</c> layout: <c>(height, width, 3)</c> uint8, row-major, no alpha. Both
    ///     sides' renders are brought into this form before they are compared.
    /// </summary>
    public sealed class Frame {
        /// <summary>Wraps pixel data that is already in <c>rgb_array</c> layout.</summary>
        /// <param name="height">Rows.</param>
        /// <param name="width">Columns.</param>
        /// <param name="rgb">Height × width × 3 bytes, row-major; not copied, so the caller must not reuse the array.</param>
        /// <exception cref="ArgumentException">Thrown when the byte count does not match the dimensions.</exception>
        public Frame(int height, int width, byte[] rgb) {
            if (rgb.Length != (long)height * width * 3)
                throw new ArgumentException($"A {height}x{width} RGB frame has {(long)height * width * 3} bytes, not {rgb.Length}.", nameof(rgb));
            Height = height;
            Width = width;
            Rgb = rgb;
        }

        /// <summary>Rows.</summary>
        public int Height { get; }

        /// <summary>Columns.</summary>
        public int Width { get; }

        /// <summary>The pixels, three bytes (R, G, B) each, row-major.</summary>
        public byte[] Rgb { get; }

        /// <summary>Reads a Gymnasium <c>rgb_array</c> render.</summary>
        /// <param name="facts">The described render: a uint8 array of shape <c>(H, W, 3)</c>.</param>
        /// <returns>The frame.</returns>
        /// <exception cref="ArgumentException">Thrown when the value is not a uint8 <c>(H, W, 3)</c> array, for example <c>None</c> from an env without a render mode.</exception>
        public static Frame FromFacts(ValueFacts facts) {
            if (facts.Kind != FactKind.Array || facts.DType != "|u1" || facts.Shape.Length != 3 || facts.Shape[2] != 3)
                throw new ArgumentException($"An rgb_array frame is a uint8 (H, W, 3) array; got {facts.Render()}.", nameof(facts));
            return new Frame(checked((int)facts.Shape[0]), checked((int)facts.Shape[1]), facts.Bytes);
        }

        /// <summary>Reads a Gym.NET render (an ImageSharp image of any pixel format), dropping alpha as <c>rgb_array</c> has none.</summary>
        /// <param name="image">The image; it is only read, and stays owned by the caller.</param>
        /// <returns>The frame.</returns>
        public static Frame FromImage(Image image) {
            using Image<Rgb24> rgb = image.CloneAs<Rgb24>();
            var bytes = new byte[rgb.Width * rgb.Height * 3];
            rgb.CopyPixelDataTo(bytes);
            return new Frame(rgb.Height, rgb.Width, bytes);
        }

        /// <summary>Writes the frame as a PNG, for the artifacts of a failed comparison.</summary>
        /// <param name="path">The file to create or overwrite.</param>
        public void SavePng(string path) {
            using Image<Rgb24> image = Image.LoadPixelData<Rgb24>(Rgb, Width, Height);
            image.SaveAsPng(path);
        }
    }
}
