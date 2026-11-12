#nullable enable
using System;
using System.Collections.Generic;
using Gym.Environments;
using Gym.Environments.Envs;
using Gym.Environments.Envs.MuJoCo;
using Gym.Environments.Envs.MuJoCo.Rendering;
using Gym.Envs;
using Gym.Exceptions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Envs.MuJoCo {
    /// <summary>
    ///     Gym.NET's MuJoCo renderer (DR-62, DR-69): its frames are deterministic and well formed, the visual options and the
    ///     default camera configuration take effect, and the GPU behaviours it reproduces (measured on the reference GPU,
    ///     see <see cref="GlContext"/>) hold: the 12-bit colour store, the level-of-detail table and the mipmap chain.
    /// </summary>
    /// <remarks>
    ///     How close the frames come to Gymnasium's GPU frames is the parity suite's to measure (DR-71); these tests pin the
    ///     renderer's own contract and the measured constants, so a change to either shows here first.
    /// </remarks>
    [TestClass]
    public class MujocoRendererTests {
        /// <summary>
        ///     The same state draws the same bytes: twice from one env, and from a second env reset with the same seed, in
        ///     all three array modes.
        /// </summary>
        /// <param name="mode">The render mode.</param>
        /// <exception cref="AssertFailedException">A render returned a result of another kind than the mode's.</exception>
        [TestMethod]
        [DataRow("rgb_array")]
        [DataRow("depth_array")]
        [DataRow("rgbd_tuple")]
        public void Frames_AreDeterministic(string mode) {
            using IEnv a = Make("Hopper-v5", new { render_mode = mode });
            using IEnv b = Make("Hopper-v5", new { render_mode = mode });
            using (a.Reset(seed: 3)) { }
            using (b.Reset(seed: 3)) { }
            NDArray first = Copy(a), again = Copy(a), other = Copy(b);
            Assert.IsTrue(np.array_equal(first, again), "one env, twice");
            Assert.IsTrue(np.array_equal(first, other), "two envs, one seed");

            // a frame copied out of its result: disposing the result releases its arrays (DR-51)
            static NDArray Copy(IEnv env) {
                using RenderResult r = env.Render();
                return r switch {
                    RenderResult.Rgb rgb => rgb.Frame.copy(),
                    RenderResult.Depth depth => depth.Frame.copy(),
                    RenderResult.RgbDepth both => np.concatenate(new[] { both.RgbFrame.astype(np.float32).flatten(), both.DepthFrame.flatten() }),
                    _ => throw new AssertFailedException($"unexpected result {r}"),
                };
            }
        }

        /// <summary>
        ///     The depth frame is MuJoCo's read-back depth: float32, 0 at the near plane and 1 at the far one, so the sky behind
        ///     InvertedPendulum's rail reads exactly 1 and the bodies less.
        /// </summary>
        [TestMethod]
        public void Depth_IsZeroAtTheNearPlaneAndOneAtTheFar() {
            using IEnv env = Make("InvertedPendulum-v5", new { render_mode = "depth_array" });
            using (env.Reset(seed: 0)) { }
            using var depth = (RenderResult.Depth)env.Render();
            Assert.AreEqual(NPTypeCode.Single, depth.Frame.typecode);
            CollectionAssert.AreEqual(new long[] { 480, 480 }, depth.Frame.shape);
            float[] d = depth.Frame.ToArray<float>();
            float min = float.MaxValue, max = float.MinValue;
            foreach (float v in d) {
                min = Math.Min(min, v);
                max = Math.Max(max, v);
            }

            Assert.AreEqual(1f, max, "the sky is at the far plane");
            Assert.IsTrue(min > 0.98f && min < 1f, $"the bodies are near the far plane's end of the reverse-Z range, got {min}");
        }

        /// <summary>
        ///     <c>visual_options</c> switch the viewer's MuJoCo flags: turning textures off (<c>mjVIS_TEXTURE</c>, 1) changes the
        ///     checkered floor, and a flag that is not an <c>mjtVisFlag</c> fails at the first frame with NumPy's message.
        /// </summary>
        [TestMethod]
        public void VisualOptions_SwitchTheViewersFlags() {
            using IEnv plain = Make("HalfCheetah-v5", new { render_mode = "rgb_array" });
            using IEnv untextured = Make("HalfCheetah-v5", new { render_mode = "rgb_array", visual_options = new Dictionary<int, bool> { [1] = false } });
            using (plain.Reset(seed: 0)) { }
            using (untextured.Reset(seed: 0)) { }
            using RenderResult p = plain.Render(), u = untextured.Render();
            Assert.IsFalse(np.array_equal(((RenderResult.Rgb)p).Frame, ((RenderResult.Rgb)u).Frame));

            using IEnv bad = Make("HalfCheetah-v5", new { render_mode = "rgb_array", visual_options = new Dictionary<int, bool> { [31] = true } });
            using (bad.Reset(seed: 0)) { }
            var e = Assert.ThrowsExactly<IndexError>(() => bad.Render());
            Assert.AreEqual("index 31 is out of bounds for axis 0 with size 31", e.Message);
        }

        /// <summary>
        ///     <c>default_camera_config</c> sets the free camera when the mode's first frame is drawn: a farther camera draws
        ///     another frame, and a look-at point that is not three numbers is refused.
        /// </summary>
        [TestMethod]
        public void DefaultCameraConfig_SetsTheFreeCamera() {
            using IEnv near = Make("Swimmer-v5", new { render_mode = "rgb_array" });
            using IEnv far = Make("Swimmer-v5", new { render_mode = "rgb_array", default_camera_config = new Dictionary<string, object?> { ["distance"] = 9.0 } });
            using (near.Reset(seed: 0)) { }
            using (far.Reset(seed: 0)) { }
            using RenderResult n = near.Render(), f = far.Render();
            Assert.IsFalse(np.array_equal(((RenderResult.Rgb)n).Frame, ((RenderResult.Rgb)f).Frame));

            using IEnv bad = Make("Swimmer-v5", new { render_mode = "rgb_array", default_camera_config = new Dictionary<string, object?> { ["lookat"] = "origin" } });
            using (bad.Reset(seed: 0)) { }
            Assert.ThrowsExactly<InvalidCastException>(() => bad.Render());
        }

        /// <summary>
        ///     <c>human</c> mode's window (DR-70) is not written yet: drawing one refuses with <see cref="NotImplementedError"/>
        ///     rather than showing nothing.
        /// </summary>
        [TestMethod]
        public void Human_RefusesUntilTheWindowIsWritten() {
            using IEnv env = Make("Reacher-v5", new { render_mode = "human" });
            var e = Assert.ThrowsExactly<NotImplementedError>(() => env.Unwrapped.Render());
            StringAssert.Contains(e.Message, "DR-70");
        }

        /// <summary>
        ///     A colour reaches the 8-bit buffer as the reference GPU stores it: truncated to 12 bits, then rounded with an
        ///     exact half going down. So ½ stores 127 (a plain rounding gives 128), level 1 starts at 9/4096 rather than at
        ///     0.5/255, and the ends and NaN clamp.
        /// </summary>
        [TestMethod]
        public void ToUnorm8_IsTheReferenceGpusStore() {
            Assert.AreEqual((byte)127, GlContext.ToUnorm8(0.5f), "the tie goes down");
            Assert.AreEqual((byte)0, GlContext.ToUnorm8(8.999f / 4096));
            Assert.AreEqual((byte)1, GlContext.ToUnorm8(9f / 4096));
            // measured threshold of level 251: 4024/4096 (a 4095 scale would put it at 4024/4095)
            Assert.AreEqual((byte)250, GlContext.ToUnorm8(4023.99f / 4096));
            Assert.AreEqual((byte)251, GlContext.ToUnorm8(4024f / 4096));
            Assert.AreEqual((byte)255, GlContext.ToUnorm8(1f));
            Assert.AreEqual((byte)255, GlContext.ToUnorm8(2f));
            Assert.AreEqual((byte)0, GlContext.ToUnorm8(-1f));
            Assert.AreEqual((byte)0, GlContext.ToUnorm8(float.NaN));
            // every level is reached and kept by its own value: v / 255 stores v
            for (int v = 0; v <= 255; v++)
                Assert.AreEqual((byte)v, GlContext.ToUnorm8(v / 255f), $"level {v}");
        }

        /// <summary>
        ///     The level of detail in 1/256 steps: the exponent of the derivative length plus the measured table of its
        ///     mantissa truncated to 8 bits; zero, NaN and lengths under one texel per pixel magnify.
        /// </summary>
        [TestMethod]
        public void LodSteps_AreTheMeasuredTable() {
            Assert.AreEqual(0, GlTexture.LodSteps(1f));
            Assert.AreEqual(256, GlTexture.LodSteps(2f));
            Assert.AreEqual(3 * 256, GlTexture.LodSteps(8f));
            // entry 1 of the table is 2, not round(256 log2(1 + 1/256)) = 1
            Assert.AreEqual(2, GlTexture.LodSteps(1 + 1f / 256));
            // the mantissa is truncated to 8 bits: just under 1 + 2/256 still reads entry 1
            Assert.AreEqual(2, GlTexture.LodSteps(1 + 1.99f / 256));
            Assert.AreEqual(4, GlTexture.LodSteps(1 + 2f / 256));
            // entry 61 is one lower than the rounded logarithm (79, not 80)
            Assert.AreEqual(256 + 79, GlTexture.LodSteps(2 * (1 + 61f / 256)));
            Assert.AreEqual(255, GlTexture.LodSteps(1.999f));
            Assert.AreEqual(-256, GlTexture.LodSteps(0.5f));
            Assert.IsTrue(GlTexture.LodSteps(0f) <= 0);
            Assert.IsTrue(GlTexture.LodSteps(float.NaN) <= 0);
            Assert.IsTrue(GlTexture.LodSteps(float.Epsilon) <= 0);
            Assert.AreEqual(int.MaxValue, GlTexture.LodSteps(float.PositiveInfinity));
        }

        /// <summary>
        ///     The mipmap chain of Gymnasium's 100×100 floor checker against the one the reference GPU's
        ///     <c>glGenerateMipmap</c> built, read back level by level: levels 1, 2 and 6 agree in every texel; levels 3, 4
        ///     and 5 differ by one in 10, 4 and 2 texels, where the GPU's 25 → 12 filter departs from the area box (it
        ///     stores 179 and 180 for texels whose exact area averages are equal) and the departure carries down.
        /// </summary>
        [TestMethod]
        public void MipmapChain_IsTheReferenceGpus() {
            using IEnv env = Make("Humanoid-v5");
            var model = ((MujocoEnv)env.Unwrapped).Model;
            const int tex = 2;
            Assert.AreEqual(0, model.tex_type[tex], "the floor's 2D checker");
            byte[] rgb = EnvAssets.Load("mujoco/" + model.tex_png[tex]).ToRgbBytes();
            var texture = new GlTexture(model.tex_type[tex], model.tex_width[tex], model.tex_height[tex], rgb);

            int[,] l3 = {
                { 0, 0, 0, 0, 0, 24, 180, 204, 204, 204, 204, 204 }, { 0, 0, 0, 0, 0, 24, 180, 204, 204, 204, 204, 204 },
                { 0, 0, 0, 0, 0, 24, 180, 204, 204, 204, 204, 204 }, { 0, 0, 0, 0, 0, 24, 180, 204, 204, 204, 204, 204 },
                { 0, 0, 0, 0, 0, 24, 180, 204, 204, 204, 204, 204 }, { 24, 24, 24, 24, 24, 43, 161, 179, 179, 179, 179, 179 },
                { 180, 180, 180, 180, 180, 161, 43, 24, 24, 24, 24, 24 }, { 204, 204, 204, 204, 204, 179, 24, 0, 0, 0, 0, 0 },
                { 204, 204, 204, 204, 204, 179, 24, 0, 0, 0, 0, 0 }, { 204, 204, 204, 204, 204, 179, 24, 0, 0, 0, 0, 0 },
                { 204, 204, 204, 204, 204, 179, 24, 0, 0, 0, 0, 0 }, { 204, 204, 204, 204, 204, 179, 24, 0, 0, 0, 0, 0 },
            };
            int[,] l4 = {
                { 0, 0, 12, 192, 204, 204 }, { 0, 0, 12, 192, 204, 204 }, { 12, 12, 23, 181, 191, 191 },
                { 192, 192, 181, 23, 12, 12 }, { 204, 204, 191, 12, 0, 0 }, { 204, 204, 191, 12, 0, 0 },
            };
            int[,] l5 = { { 0, 102, 204 }, { 102, 102, 101 }, { 204, 101, 0 } };
            int[,] l6 = { { 102 } };
            foreach ((int level, int[,] gpu, int departures) in new[] { (3, l3, 10), (4, l4, 4), (5, l5, 2), (6, l6, 0) }) {
                int n = gpu.GetLength(0), differ = 0;
                for (int j = 0; j < n; j++) {
                    for (int i = 0; i < n; i++) {
                        // a level's texel exactly: its centre at that whole level, so no weight falls on a neighbour
                        texture.Sample2D((i + 0.5f) / n, (j + 0.5f) / n, level * 256, out float r, out float g, out float b);
                        int ours = (int)MathF.Round(g * 255);
                        Assert.AreEqual(ours, (int)MathF.Round(r * 255), "a grey texture");
                        Assert.IsTrue(Math.Abs(gpu[j, i] - ours) <= 1, $"level {level} texel ({i}, {j}): GPU {gpu[j, i]}, Gym.NET {ours}");
                        differ += gpu[j, i] != ours ? 1 : 0;
                    }
                }

                Assert.AreEqual(departures, differ, $"level {level}'s departures from the GPU's");
            }
        }
    }
}
