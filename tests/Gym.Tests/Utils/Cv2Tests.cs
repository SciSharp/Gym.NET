#nullable enable
using System;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Utils {
    /// <summary>
    ///     <see cref="Cv2.Resize"/>'s float32 kernels (DR-46), checked without Python: the area-fast kernel's two orders of
    ///     summation, which only floats can tell apart. Every pinned value is <c>cv2.resize</c>'s own (OpenCV 4.13.0, IPP
    ///     off); the random corpora against live OpenCV are the parity suite's (<c>Cv2ResizeLiveParityTests</c>).
    /// </summary>
    /// <remarks>
    ///     The blocks hold <c>1e8, 1</c> over <c>-1e8, 1</c>: summed pairwise, <c>(1e8 + 1) + (-1e8 + 1)</c> is 0 in float32
    ///     (each 1 is lost against 1e8); summed left to right, <c>((1e8 + 1) - 1e8) + 1</c> is 1, a mean of 0.25.
    /// </remarks>
    [TestClass]
    public class Cv2Tests {
        /// <summary>
        ///     One channel at half size: the vector kernel sums the first whole group of four outputs pairwise (0), the fifth
        ///     output is the generic loop's tail, summed left to right (0.25). <c>INTER_LINEAR</c> at exactly half size takes
        ///     the same kernel.
        /// </summary>
        [TestMethod]
        public void Float32AreaFast_OneChannel_GroupsOfFourSumPairwise_TheTailInOrder() {
            var pixels = new float[20];
            foreach (int column in new[] { 0, 8 }) {
                pixels[column] = 1e8f;
                pixels[column + 1] = 1f;
                pixels[10 + column] = -1e8f;
                pixels[10 + column + 1] = 1f;
            }

            using var image = new NDArray(pixels, new Shape(2, 10));
            foreach (int interpolation in new[] { Cv2.InterArea, Cv2.InterLinear }) {
                using NDArray resized = Cv2.Resize(image, (5, 1), interpolation);
                CollectionAssert.AreEqual(new[] { 0f, 0f, 0f, 0f, 0.25f }, resized.ToArray<float>(), $"interpolation {interpolation}");
                Assert.AreEqual(NPTypeCode.Single, resized.typecode);
            }
        }

        /// <summary>
        ///     Four channels go through the 128-bit vector kernel, each channel's block summed by rows first (0); three channels
        ///     take the generic loop, left to right (0.25). The wheels' 256-bit variant would pair the columns first (0.5).
        /// </summary>
        [TestMethod]
        public void Float32AreaFast_FourChannelsPairwise_ThreeInOrder() {
            float[] Block(int channels) {
                var block = new float[2 * 2 * channels];
                for (int c = 0; c < channels; c++) {
                    block[c] = 1e8f;
                    block[channels + c] = 1f;
                    block[2 * channels + c] = -1e8f;
                    block[3 * channels + c] = 1f;
                }

                return block;
            }

            using var four = new NDArray(Block(4), new Shape(2, 2, 4));
            using NDArray fourResized = Cv2.Resize(four, (1, 1), Cv2.InterArea);
            CollectionAssert.AreEqual(new[] { 0f, 0f, 0f, 0f }, fourResized.ToArray<float>());

            using var three = new NDArray(Block(3), new Shape(2, 2, 3));
            using NDArray threeResized = Cv2.Resize(three, (1, 1), Cv2.InterArea);
            CollectionAssert.AreEqual(new[] { 0.25f, 0.25f, 0.25f }, threeResized.ToArray<float>());
        }

        /// <summary>A dtype other than uint8 and float32 is refused, naming both.</summary>
        [TestMethod]
        public void OtherDtypes_AreRefused() {
            using NDArray doubles = np.zeros(new Shape(2L, 2L), np.float64);
            StringAssert.Contains(Assert.ThrowsExactly<NotSupportedException>(() => Cv2.Resize(doubles, (1, 1), Cv2.InterArea)).Message,
                "uint8 and float32 images only");
        }
    }
}
