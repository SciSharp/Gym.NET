using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Gym.Tests.Parity.Pair;

namespace Gym.Tests.Parity.Frames {
    /// <summary>
    ///     The difference between two frames, with the statistics plan §6 asks for: they tell anti-aliasing noise (a few
    ///     pixels, small deltas, along edges) from geometry or colour errors (large deltas in bands).
    /// </summary>
    /// <param name="PythonSize">Gymnasium's frame size as <c>(H, W)</c>.</param>
    /// <param name="DotNetSize">Gym.NET's frame size as <c>(H, W)</c>.</param>
    /// <param name="DifferingPixels">Pixels with any channel differing; 0 when the sizes differ, since pixels cannot be paired.</param>
    /// <param name="MaxChannelDelta">The largest absolute difference of any channel of any pixel (0–255).</param>
    /// <param name="Bounds">The smallest rectangle holding every differing pixel, inclusive, as (left, top, right, bottom); <see langword="null"/> when none differ.</param>
    /// <param name="Histogram">Differing pixels per bucket of their largest channel delta; the buckets are <see cref="FrameDiff.Buckets"/>.</param>
    public sealed record FrameComparison(
        (int Height, int Width) PythonSize,
        (int Height, int Width) DotNetSize,
        long DifferingPixels,
        int MaxChannelDelta,
        (int Left, int Top, int Right, int Bottom)? Bounds,
        IReadOnlyList<long> Histogram) {
        /// <summary>Whether the sizes agree.</summary>
        public bool SameSize => PythonSize == DotNetSize;

        /// <summary>Whether the frames are byte-identical, the only passing result.</summary>
        public bool Identical => SameSize && DifferingPixels == 0;

        /// <summary>The share of pixels that differ.</summary>
        public double Fraction => SameSize && PythonSize.Height * PythonSize.Width > 0 ? (double)DifferingPixels / (PythonSize.Height * PythonSize.Width) : 1.0;

        /// <summary>Renders the statistics on one line, for a divergence report.</summary>
        /// <returns>The size mismatch, or the counts, share, bounds, max delta and histogram.</returns>
        public string Describe() {
            if (!SameSize)
                return $"size python {PythonSize.Height}x{PythonSize.Width} vs dotnet {DotNetSize.Height}x{DotNetSize.Width}";
            if (DifferingPixels == 0)
                return "identical";
            var bounds = Bounds!.Value;
            string histogram = string.Join(", ", FrameDiff.Buckets.Select((b, i) => (b, count: Histogram[i])).Where(x => x.count > 0)
                .Select(x => $"{(x.b.From == x.b.To ? x.b.From.ToString(CultureInfo.InvariantCulture) : $"{x.b.From}-{x.b.To}")}: {x.count}"));
            // Four decimals of a percent, so a single pixel of a 600x400 frame (0.0004%) does not print as 0.00%.
            string share = (Fraction * 100).ToString("0.####", CultureInfo.InvariantCulture) + "%";
            return $"{DifferingPixels} of {PythonSize.Height * PythonSize.Width} pixels differ ({share}), " +
                   $"within x {bounds.Left}..{bounds.Right}, y {bounds.Top}..{bounds.Bottom}; max channel delta {MaxChannelDelta}; by delta {{{histogram}}}";
        }
    }

    /// <summary>
    ///     Pixel parity (plan §6): compares two <c>rgb_array</c> frames exactly and, when they differ, writes
    ///     <c>python.png</c>, <c>dotnet.png</c> and an amplified <c>diff.png</c> next to the test result.
    /// </summary>
    public static class FrameDiff {
        /// <summary>
        ///     The histogram's buckets of per-pixel delta. 1 and 2 are anti-aliasing territory; 129–255 means different
        ///     colours or geometry.
        /// </summary>
        public static readonly IReadOnlyList<(int From, int To)> Buckets = new[] { (1, 1), (2, 2), (3, 8), (9, 32), (33, 128), (129, 255) };

        /// <summary>Compares two frames pixel by pixel.</summary>
        /// <param name="python">Gymnasium's frame.</param>
        /// <param name="dotnet">Gym.NET's frame.</param>
        /// <returns>The statistics.</returns>
        public static FrameComparison Compare(Frame python, Frame dotnet) {
            var histogram = new long[Buckets.Count];
            if (python.Height != dotnet.Height || python.Width != dotnet.Width)
                return new FrameComparison((python.Height, python.Width), (dotnet.Height, dotnet.Width), 0, 0, null, histogram);

            long differing = 0;
            int maxDelta = 0, left = int.MaxValue, top = int.MaxValue, right = -1, bottom = -1;
            byte[] p = python.Rgb, d = dotnet.Rgb;
            for (int y = 0; y < python.Height; y++) {
                for (int x = 0; x < python.Width; x++) {
                    int delta = PixelDelta(p, d, (y * python.Width + x) * 3);
                    if (delta == 0)
                        continue;
                    differing++;
                    maxDelta = Math.Max(maxDelta, delta);
                    left = Math.Min(left, x);
                    right = Math.Max(right, x);
                    top = Math.Min(top, y);
                    bottom = Math.Max(bottom, y);
                    histogram[Bucket(delta)]++;
                }
            }

            (int, int, int, int)? bounds = differing == 0 ? null : (left, top, right, bottom);
            return new FrameComparison((python.Height, python.Width), (dotnet.Height, dotnet.Width), differing, maxDelta, bounds, histogram);
        }

        /// <summary>
        ///     Throws a <see cref="ParityException"/> with field <c>frame</c> unless the frames are identical, after writing the
        ///     artifacts.
        /// </summary>
        /// <param name="what">What was rendered, the report's call line (<c>render() at state [0.5, 0, 0.15, 0]</c>).</param>
        /// <param name="python">Gymnasium's frame.</param>
        /// <param name="dotnet">Gym.NET's frame.</param>
        /// <param name="case">The report's case line.</param>
        /// <exception cref="ParityException">Thrown when the frames differ.</exception>
        public static void AssertSame(string what, Frame python, Frame dotnet, string @case = "frame comparison") {
            FrameComparison comparison = Compare(python, dotnet);
            if (comparison.Identical)
                return;
            IReadOnlyList<string> artifacts = WriteArtifacts(python, dotnet, comparison, what);
            throw new ParityException(new Divergence {
                Case = @case,
                Call = what,
                Fields = new[] { new FieldDivergence("frame", "frame", comparison.Describe()) },
                Artifacts = artifacts,
            });
        }

        /// <summary>
        ///     Writes both frames and an amplified difference as PNGs, plus the statistics as text, and attaches them to the
        ///     test result. In <c>diff.png</c> matching pixels are a dimmed grey copy of Gymnasium's frame and differing
        ///     pixels are red, bright enough that a delta of 1 is visible.
        /// </summary>
        /// <param name="python">Gymnasium's frame.</param>
        /// <param name="dotnet">Gym.NET's frame.</param>
        /// <param name="comparison">Their comparison.</param>
        /// <param name="label">What was rendered; names the artifacts' directory.</param>
        /// <returns>The files written.</returns>
        public static IReadOnlyList<string> WriteArtifacts(Frame python, Frame dotnet, FrameComparison comparison, string label) {
            string directory = ParityArtifacts.DirectoryFor(label);
            var files = new List<string> { Path.Combine(directory, "python.png"), Path.Combine(directory, "dotnet.png") };
            python.SavePng(files[0]);
            dotnet.SavePng(files[1]);
            if (comparison.SameSize) {
                files.Add(Path.Combine(directory, "diff.png"));
                Amplified(python, dotnet).SavePng(files[2]);
            }

            files.Add(Path.Combine(directory, "stats.txt"));
            File.WriteAllText(files[^1], comparison.Describe() + Environment.NewLine);
            foreach (string file in files)
                ParityArtifacts.Attach(file);
            return files;
        }

        /// <summary>Builds the amplified difference image described in <see cref="WriteArtifacts"/>.</summary>
        /// <param name="python">Gymnasium's frame.</param>
        /// <param name="dotnet">Gym.NET's frame, of the same size.</param>
        /// <returns>The difference image.</returns>
        private static Frame Amplified(Frame python, Frame dotnet) {
            var rgb = new byte[python.Rgb.Length];
            for (int offset = 0; offset < rgb.Length; offset += 3) {
                int delta = PixelDelta(python.Rgb, dotnet.Rgb, offset);
                if (delta == 0) {
                    byte grey = (byte)((python.Rgb[offset] + python.Rgb[offset + 1] + python.Rgb[offset + 2]) / 12);
                    rgb[offset] = rgb[offset + 1] = rgb[offset + 2] = grey;
                } else {
                    rgb[offset] = (byte)(96 + delta * 159 / 255);
                }
            }

            return new Frame(python.Height, python.Width, rgb);
        }

        /// <summary>The largest absolute channel difference of the pixel at <paramref name="offset"/>.</summary>
        /// <param name="python">Gymnasium's bytes.</param>
        /// <param name="dotnet">Gym.NET's bytes.</param>
        /// <param name="offset">The pixel's first byte.</param>
        /// <returns>0 to 255.</returns>
        private static int PixelDelta(byte[] python, byte[] dotnet, int offset) {
            return Math.Max(Math.Abs(python[offset] - dotnet[offset]), Math.Max(Math.Abs(python[offset + 1] - dotnet[offset + 1]), Math.Abs(python[offset + 2] - dotnet[offset + 2])));
        }

        /// <summary>The histogram bucket of a non-zero delta.</summary>
        /// <param name="delta">1 to 255.</param>
        /// <returns>The index into <see cref="Buckets"/>.</returns>
        private static int Bucket(int delta) {
            for (int i = 0; i < Buckets.Count; i++) {
                if (delta <= Buckets[i].To)
                    return i;
            }

            return Buckets.Count - 1;
        }
    }
}
