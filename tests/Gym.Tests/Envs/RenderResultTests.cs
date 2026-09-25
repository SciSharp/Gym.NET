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
    }
}
