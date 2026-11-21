#nullable enable
using System;
using Gym.Envs;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Envs {
    /// <summary>Pins <see cref="RenderResult"/>: its single <c>None</c>, and structural equality that never broadcasts.</summary>
    [TestClass]
    public class RenderResultTests {
        /// <summary><see cref="RenderResult.None"/> is one value and only it is <c>None</c>.</summary>
        [TestMethod]
        public void None_IsSingle() {
            Assert.AreSame(RenderResult.None, RenderResult.None);
            Assert.IsTrue(RenderResult.None.IsNone);
            Assert.AreEqual("None", RenderResult.None.ToString());
            using var frame = np.zeros(new Shape(2, 2, 3), np.uint8);
            Assert.IsFalse(new RenderResult.Rgb(frame).IsNone);
        }

        /// <summary>Equal cases compare their contents exactly: shape, dtype and elements; different cases are never equal.</summary>
        [TestMethod]
        public void Equality_IsStructural() {
            using var a = np.zeros(new Shape(2, 2, 3), np.uint8);
            using var b = np.zeros(new Shape(2, 2, 3), np.uint8);
            using var wider = np.zeros(new Shape(2, 2, 3), np.int32);
            using var scalar = NDArray.Scalar<byte>(0);

            Assert.AreEqual<RenderResult>(new RenderResult.Rgb(a), new RenderResult.Rgb(b));
            Assert.AreNotEqual<RenderResult>(new RenderResult.Rgb(a), new RenderResult.Rgb(wider), "The dtype takes part.");
            Assert.AreNotEqual<RenderResult>(new RenderResult.Rgb(a), new RenderResult.Rgb(scalar), "A 0-d array does not broadcast.");
            Assert.AreNotEqual<RenderResult>(new RenderResult.Rgb(a), new RenderResult.Depth(a));
            Assert.AreEqual<RenderResult>(new RenderResult.Ansi("x"), new RenderResult.Ansi("x"));
            Assert.AreEqual<RenderResult>(new RenderResult.Frames(new[] { new RenderResult.Ansi("x") }), new RenderResult.Frames(new[] { new RenderResult.Ansi("x") }));
            Assert.AreNotEqual<RenderResult>(new RenderResult.Frames(new RenderResult[0]), RenderResult.None);
            Assert.AreEqual("Rgb((2, 2, 3), uint8)", new RenderResult.Rgb(a).ToString());
            Assert.ThrowsExactly<ArgumentNullException>(() => new RenderResult.Ansi(null!));
        }

        /// <summary>
        ///     A single frame converts explicitly (DR-84): <c>(NDArray)</c> is the result's own <see cref="RenderResult.Rgb"/> or
        ///     <see cref="RenderResult.Depth"/> frame, not a copy; every other case refuses with <see cref="InvalidCastException"/>.
        /// </summary>
        [TestMethod]
        public void ToNDArray_IsTheFrameItself() {
            NDArray frame = np.zeros(new Shape(2, 2, 3), np.uint8);
            using RenderResult rgb = new RenderResult.Rgb(frame);
            Assert.AreSame(frame, (NDArray)rgb, "The result's own frame: disposing the result disposes it.");

            NDArray depthFrame = np.zeros(new Shape(2, 2), np.float32);
            using RenderResult depth = new RenderResult.Depth(depthFrame);
            Assert.AreSame(depthFrame, (NDArray)depth);

            Assert.ThrowsExactly<InvalidCastException>(() => (NDArray)RenderResult.None, "human mode returns no frame.");
            Assert.ThrowsExactly<InvalidCastException>(() => (NDArray)new RenderResult.Ansi("x"));
            using RenderResult pair = new RenderResult.RgbDepth(np.zeros(new Shape(1, 1, 3), np.uint8), np.zeros(new Shape(1, 1), np.float32));
            Assert.ThrowsExactly<InvalidCastException>(() => (NDArray)pair, "Two frames are no single frame.");
            using RenderResult frames = new RenderResult.Frames(new RenderResult[0]);
            Assert.ThrowsExactly<InvalidCastException>(() => (NDArray)frames);
            Assert.ThrowsExactly<ArgumentNullException>(() => (NDArray)(RenderResult)null!);
        }

        /// <summary>
        ///     An <see cref="RenderResult.Rgb"/> frame converts to a new ImageSharp image of its size and pixels (DR-84), which the
        ///     caller owns; a depth frame and the cases without a single RGB frame refuse.
        /// </summary>
        [TestMethod]
        public void ToImage_CopiesTheRgbFrame() {
            // A 2x3 frame (height 2, width 3) whose pixel (row 1, column 2) is (10, 20, 30).
            using NDArray frame = np.zeros(new Shape(2, 3, 3), np.uint8);
            frame[1, 2] = np.array(new byte[] { 10, 20, 30 });
            using RenderResult rgb = new RenderResult.Rgb(frame);
            using (var image = (SixLabors.ImageSharp.Image)rgb) {
                Assert.AreEqual(3, image.Width);
                Assert.AreEqual(2, image.Height);
                var pixels = (SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgb24>)image;
                Assert.AreEqual(new SixLabors.ImageSharp.PixelFormats.Rgb24(10, 20, 30), pixels[2, 1], "ImageSharp indexes (x, y).");
            }

            Assert.AreEqual((byte)10, (byte)frame[1, 2, 0], "The result keeps its frame after the image is disposed.");
            using RenderResult depth = new RenderResult.Depth(np.zeros(new Shape(2, 2), np.float32));
            Assert.ThrowsExactly<InvalidCastException>(() => (SixLabors.ImageSharp.Image)depth, "A depth frame has no colors.");
            Assert.ThrowsExactly<InvalidCastException>(() => (SixLabors.ImageSharp.Image)RenderResult.None);
        }
    }
}
