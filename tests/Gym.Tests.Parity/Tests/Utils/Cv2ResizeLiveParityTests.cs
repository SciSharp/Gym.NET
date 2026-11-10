using System;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Session;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace Gym.Tests.Parity.Tests.Utils {
    /// <summary>
    ///     <see cref="Cv2.Resize"/> against live OpenCV's <c>cv2.resize</c> (decisions DR-23 and DR-46), bit for bit: random
    ///     uint8 and float32 images of 1 to 5 channels (one channel as <c>(H, W)</c> and as <c>(H, W, 1)</c>), through each
    ///     kernel OpenCV picks by the scales.
    /// </summary>
    /// <remarks>
    ///     The kernels and what each case exercises: the integer-factor area average (the 2×2 kernel's round-half-up for 1,
    ///     3 and 4 channels, the float product <c>sum * (1.f / area)</c> otherwise), the general area decimation (float32
    ///     weights per cell), the bilinear emulation <c>INTER_AREA</c> takes when an axis grows (alone or with the other
    ///     shrinking), and plain <c>INTER_LINEAR</c>, which becomes the 2×2 area average at exactly half size. A one-channel
    ///     result is compared with its shape, so <c>cv2</c>'s dropped channel axis is pinned too.
    /// </remarks>
    [TestClass]
    public sealed class Cv2ResizeLiveParityTests : ParityTestBase {
        /// <summary>Random cases of one kernel family, each image and target size compared with <c>cv2.resize</c>.</summary>
        /// <param name="family">
        ///     <c>area-fast</c> (integer factors on both axes), <c>area</c> (shrinking by fractional factors), <c>area-up</c>
        ///     (growing on at least one axis), <c>linear</c> (any size) or <c>linear-half</c> (exactly half size).
        /// </param>
        /// <param name="seed">The case generator's seed.</param>
        [TestMethod]
        [DataRow("area-fast", 1)]
        [DataRow("area", 2)]
        [DataRow("area-up", 3)]
        [DataRow("linear", 4)]
        [DataRow("linear-half", 5)]
        public void Resize_MatchesOpenCv(string family, int seed) {
            var rng = new Random(20261110 + seed);
            for (int i = 0; i < 250; i++) {
                // OpenCV's general area kernel takes at most 4 channels (Resize_FiveChannelArea_IsRefusedLikeOpenCv).
                int channels = rng.Next(1, family == "area" ? 5 : 6);
                bool channelAxis = channels > 1 || rng.Next(2) == 0;
                (int srcWidth, int srcHeight, int width, int height) = Sizes(rng, family);
                int interpolation = family.StartsWith("area", StringComparison.Ordinal) ? Cv2.InterArea : Cv2.InterLinear;

                var bytes = new byte[srcWidth * srcHeight * channels];
                rng.NextBytes(bytes);
                using NDArray image = new NDArray(bytes, channelAxis ? new Shape(srcHeight, srcWidth, channels) : new Shape(srcHeight, srcWidth));
                string label = $"cv2.resize({image.dtype.name}{FormatShape(image)}, ({width}, {height}), interpolation={interpolation}) #{i}";
                Outcome python = Outcome.CapturePython(() => {
                    using PyObject source = image.ToNumpyCopy();
                    return Python.cv2.resize(source, width, height, interpolation);
                });
                ByteContract.AssertSame(label, python, Outcome.Capture(() => Cv2.Resize(image, (width, height), interpolation)));
            }
        }

        /// <summary>
        ///     The float32 kernels (DR-46) through the same families as <see cref="Resize_MatchesOpenCv"/>: random float32 images
        ///     of 1 to 5 channels, values from -300 to 300, every float compared bit for bit. <c>INTER_LINEAR</c> is compared with
        ///     OpenCV's own kernel, IPP switched off for the cell (the wheels hand float32 bilinear to Intel IPP by default,
        ///     whose results differ: <see cref="Resize_Float32_Linear_MatchesDefaultOpenCv"/>); <c>INTER_AREA</c> does not depend
        ///     on IPP (measured).
        /// </summary>
        /// <param name="family">The kernel family, as in <see cref="Resize_MatchesOpenCv"/>.</param>
        /// <param name="seed">The case generator's seed.</param>
        [TestMethod]
        [DataRow("area-fast", 11)]
        [DataRow("area", 12)]
        [DataRow("area-up", 13)]
        [DataRow("linear", 14)]
        [DataRow("linear-half", 15)]
        public void Resize_Float32_MatchesOpenCv(string family, int seed) {
            bool ippBefore = PythonRuntime.Run(() => Python.cv2.ipp.useIPP());
            PythonRuntime.Run(() => Python.cv2.ipp.setUseIPP(false));
            try {
                AssertFloat32Cases(family, seed, 250);
            } finally {
                PythonRuntime.Run(() => Python.cv2.ipp.setUseIPP(ippBefore));
            }
        }

        /// <summary>
        ///     Float32 <c>INTER_LINEAR</c> against the wheels' default, IPP on. Open: Intel IPP's bilinear is not published, and
        ///     no float32 formula tried reproduces it (the best matched two thirds of its outputs); Gym.NET gives OpenCV's own
        ///     kernel, what <c>cv2</c> computes without IPP. The gap is up to about 4.4e-4 (a few dozen units in the last place
        ///     at 255), on most cases; DR-46's measurement, reported to the maintainer.
        /// </summary>
        [TestMethod]
        [OpenBugs("DR-46-IPP")]
        public void Resize_Float32_Linear_MatchesDefaultOpenCv() {
            bool ippBefore = PythonRuntime.Run(() => Python.cv2.ipp.useIPP());
            PythonRuntime.Run(() => Python.cv2.ipp.setUseIPP(true));
            try {
                AssertFloat32Cases("linear", 16, 100);
            } finally {
                PythonRuntime.Run(() => Python.cv2.ipp.setUseIPP(ippBefore));
            }
        }

        /// <summary>Gymnasium's real inputs: CartPole's 600 × 400 frame and Atari-sized frames shrunk to 84 × 84 and 64 × 64, by area.</summary>
        /// <param name="srcWidth">The frame's width.</param>
        /// <param name="srcHeight">The frame's height.</param>
        /// <param name="width">The target width.</param>
        /// <param name="height">The target height.</param>
        [TestMethod]
        [DataRow(600, 400, 84, 84)]
        [DataRow(600, 400, 64, 64)]
        [DataRow(160, 210, 84, 84)]
        [DataRow(96, 96, 32, 32)]
        [DataRow(96, 96, 48, 48)]
        public void Resize_FramesMatchOpenCv(int srcWidth, int srcHeight, int width, int height) {
            var rng = new Random(srcWidth * 1000 + width);
            var bytes = new byte[srcWidth * srcHeight * 3];
            rng.NextBytes(bytes);
            using NDArray image = new NDArray(bytes, new Shape(srcHeight, srcWidth, 3));
            Outcome python = Outcome.CapturePython(() => {
                using PyObject source = image.ToNumpyCopy();
                return Python.cv2.resize(source, width, height, Python.cv2.INTER_AREA);
            });
            ByteContract.AssertSame($"cv2.resize(frame {srcHeight}x{srcWidth}x3, ({width}, {height}), INTER_AREA)", python,
                Outcome.Capture(() => Cv2.Resize(image, (width, height), Cv2.InterArea)));
        }

        /// <summary>
        ///     Five channels through the general area kernel: OpenCV refuses them with an assertion (<c>cv2.error</c>, which
        ///     maps to no .NET type), and Gym.NET refuses them with <see cref="NotSupportedException"/> and the same text.
        /// </summary>
        [TestMethod]
        public void Resize_FiveChannelArea_IsRefusedLikeOpenCv() {
            using NDArray image = np.zeros(new Shape(20L, 5L, 5L), np.uint8);
            Outcome python = Outcome.CapturePython(() => {
                using PyObject source = image.ToNumpyCopy();
                return Python.cv2.resize(source, 3, 9, Python.cv2.INTER_AREA);
            });
            Assert.IsNotNull(python.Raised, "cv2 resized five channels by area");
            StringAssert.Contains(python.Raised!.Message, "(-215:Assertion failed) func != 0 && cn <= 4 in function 'cv::hal::resize'");
            var refusal = Assert.ThrowsExactly<NotSupportedException>(() => Cv2.Resize(image, (3, 9), Cv2.InterArea));
            StringAssert.Contains(python.Raised.Message, refusal.Message);
        }

        /// <summary>
        ///     Random float32 cases of one kernel family, each image and target size compared with <c>cv2.resize</c> as it is
        ///     configured now (the caller sets IPP).
        /// </summary>
        /// <param name="family">The kernel family.</param>
        /// <param name="seed">The case generator's seed.</param>
        /// <param name="count">How many cases.</param>
        /// <exception cref="Gym.Tests.Parity.Pair.ParityException">Thrown at the first case that differs.</exception>
        private static void AssertFloat32Cases(string family, int seed, int count) {
            var rng = new Random(20261110 + seed);
            for (int i = 0; i < count; i++) {
                int channels = rng.Next(1, family == "area" ? 5 : 6);
                bool channelAxis = channels > 1 || rng.Next(2) == 0;
                (int srcWidth, int srcHeight, int width, int height) = Sizes(rng, family);
                int interpolation = family.StartsWith("area", StringComparison.Ordinal) ? Cv2.InterArea : Cv2.InterLinear;

                var floats = new float[srcWidth * srcHeight * channels];
                for (int j = 0; j < floats.Length; j++)
                    floats[j] = (float)(rng.NextDouble() * 600 - 300);
                using NDArray image = new NDArray(floats, channelAxis ? new Shape(srcHeight, srcWidth, channels) : new Shape(srcHeight, srcWidth));
                string label = $"cv2.resize({image.dtype.name}{FormatShape(image)}, ({width}, {height}), interpolation={interpolation}) #{i}";
                Outcome python = Outcome.CapturePython(() => {
                    using PyObject source = image.ToNumpyCopy();
                    return Python.cv2.resize(source, width, height, interpolation);
                });
                ByteContract.AssertSame(label, python, Outcome.Capture(() => Cv2.Resize(image, (width, height), interpolation)));
            }
        }

        /// <summary>The source and target sizes of one random case of a family.</summary>
        /// <param name="rng">The case generator.</param>
        /// <param name="family">The kernel family.</param>
        /// <returns>The source width and height and the target width and height.</returns>
        private static (int SrcWidth, int SrcHeight, int Width, int Height) Sizes(Random rng, string family) {
            switch (family) {
                case "area-fast": {
                    // Integer factors on both axes, 2×2 half the time (its own kernel).
                    bool twoByTwo = rng.Next(2) == 0;
                    int fx = twoByTwo ? 2 : rng.Next(1, 6), fy = twoByTwo ? 2 : rng.Next(1, 6);
                    if (fx == 1 && fy == 1)
                        fx = 3;
                    int width = rng.Next(1, 40), height = rng.Next(1, 40);
                    return (width * fx, height * fy, width, height);
                }
                case "area": {
                    int srcWidth = rng.Next(2, 160), srcHeight = rng.Next(2, 160);
                    return (srcWidth, srcHeight, rng.Next(1, srcWidth + 1), rng.Next(1, srcHeight + 1));
                }
                case "area-up": {
                    int srcWidth = rng.Next(1, 60), srcHeight = rng.Next(1, 60);
                    // One axis grows; the other grows or shrinks.
                    int width = rng.Next(2) == 0 ? rng.Next(srcWidth + 1, srcWidth * 4 + 2) : rng.Next(1, srcWidth * 3 + 2);
                    int height = width <= srcWidth ? rng.Next(srcHeight + 1, srcHeight * 4 + 2) : rng.Next(1, srcHeight * 3 + 2);
                    return (srcWidth, srcHeight, width, height);
                }
                case "linear": {
                    int srcWidth = rng.Next(1, 120), srcHeight = rng.Next(1, 120);
                    return (srcWidth, srcHeight, rng.Next(1, 160), rng.Next(1, 160));
                }
                default: {
                    int width = rng.Next(1, 60), height = rng.Next(1, 60);
                    return (width * 2, height * 2, width, height);
                }
            }
        }

        /// <summary>An array's shape as Python prints it, for the case label.</summary>
        /// <param name="array">The array.</param>
        /// <returns><c>(h, w)</c> or <c>(h, w, c)</c>.</returns>
        private static string FormatShape(NDArray array) => "(" + string.Join(", ", array.shape) + ")";
    }
}
