#nullable enable
using System;
using System.IO;
using System.Linq;
using Gym.Environments.Envs.MuJoCo;
using Gym.Envs;
using Gym.Exceptions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Envs.MuJoCo {
    /// <summary>
    ///     Gymnasium's MuJoCo rendering tests (<c>tests/envs/mujoco/test_mujoco_rendering.py</c>): frames of any size, the
    ///     render modes every MuJoCo env declares, <c>max_geom</c>, <c>camera_id</c> and <c>rgbd_tuple</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Gymnasium's fixture model is <c>walker2d_v5_uneven_feet.xml</c>, a file of the test folder that Gym.NET cannot
    ///         compile (DR-63): the fixture cases draw Walker2d-v5's own model instead, which differs only in its feet.
    ///         Gymnasium's <c>OffScreenViewer</c> and its markers are not ported (DR-69): the dimension case draws through
    ///         <see cref="MujocoRenderer"/>, which inlines the offscreen viewer, and <see cref="TestAddMarkers"/> pins that
    ///         no marker API exists. The <c>human</c> cases check the renderer's settings only until the window (DR-70) is
    ///         written; drawing one throws <see cref="NotImplementedError"/>, which <c>MujocoRendererTests</c> pins.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_rendering.py">tests/envs/mujoco/test_mujoco_rendering.py</see>.</para>
    /// </remarks>
    [TestClass]
    public class TestMujocoRendering {
        /// <summary>Gymnasium's <c>DEFAULT_SIZE</c>: the frame side the fixture renderer uses.</summary>
        private const int DefaultSize = 480;

        /// <summary>
        ///     A renderer of any size returns frames of that size: <c>(height, width, 3)</c> bytes.
        /// </summary>
        /// <param name="width">The frame's width.</param>
        /// <param name="height">The frame's height.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_rendering.py#L39">test_offscreen_viewer_custom_dimensions</see>.</remarks>
        [TestMethod]
        [DataRow(10, 10)]
        [DataRow(10, 100)]
        [DataRow(10, 200)]
        [DataRow(10, 480)]
        [DataRow(100, 10)]
        [DataRow(100, 100)]
        [DataRow(100, 200)]
        [DataRow(100, 480)]
        [DataRow(200, 10)]
        [DataRow(200, 100)]
        [DataRow(200, 200)]
        [DataRow(200, 480)]
        [DataRow(480, 10)]
        [DataRow(480, 100)]
        [DataRow(480, 200)]
        [DataRow(480, 480)]
        public void TestOffscreenViewerCustomDimensions(int width, int height) {
            using IEnv env = Make("Walker2d-v5");
            var mujocoEnv = (MujocoEnv)env.Unwrapped;

            // initialize viewer
            var viewer = new MujocoRenderer(mujocoEnv.Model, mujocoEnv.Data, width: width, height: height);

            // check that the render method returns an image of the correct shape
            using RenderResult img = viewer.Render("rgb_array");
            NDArray frame = ((RenderResult.Rgb)img).Frame;
            CollectionAssert.AreEqual(new long[] { height, width, 3 }, frame.shape);
            Assert.AreEqual(NPTypeCode.Byte, frame.typecode);

            // close viewer after usage
            viewer.Close();
        }

        /// <summary>Every MuJoCo env declares the four render modes, in Gymnasium's order.</summary>
        /// <param name="envId">The env's name.</param>
        /// <param name="version">Its version.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_rendering.py#L62">test_mujoco_metadata_render_modes</see>.</remarks>
        [TestMethod]
        [DataRow("Ant", "v4")]
        [DataRow("Ant", "v5")]
        [DataRow("HalfCheetah", "v4")]
        [DataRow("HalfCheetah", "v5")]
        [DataRow("Hopper", "v4")]
        [DataRow("Hopper", "v5")]
        [DataRow("Humanoid", "v4")]
        [DataRow("Humanoid", "v5")]
        [DataRow("HumanoidStandup", "v4")]
        [DataRow("HumanoidStandup", "v5")]
        [DataRow("InvertedDoublePendulum", "v4")]
        [DataRow("InvertedDoublePendulum", "v5")]
        [DataRow("InvertedPendulum", "v4")]
        [DataRow("InvertedPendulum", "v5")]
        [DataRow("Pusher", "v4")]
        [DataRow("Pusher", "v5")]
        [DataRow("Reacher", "v4")]
        [DataRow("Reacher", "v5")]
        [DataRow("Swimmer", "v4")]
        [DataRow("Swimmer", "v5")]
        [DataRow("Walker2d", "v4")]
        [DataRow("Walker2d", "v5")]
        public void TestMujocoMetadataRenderModes(string envId, string version) {
            // Pusher-v4 needs mujoco<3, which Gymnasium skips and Gym.NET refuses alike
            if (envId == "Pusher" && version == "v4")
                return;
            using IEnv env = Make($"{envId}-{version}");
            CollectionAssert.AreEqual(new[] { "human", "rgb_array", "depth_array", "rgbd_tuple" },
                                      ((System.Collections.Generic.IEnumerable<string>)env.Metadata[Metadata.RenderModesKey]!).ToArray());
            env.Close();
        }

        /// <summary>
        ///     The renderer keeps <c>max_geom</c>, and a mode's frames hold at most that many geoms: a cap below the
        ///     model's count drops geoms (the frame changes) and prints MuJoCo's warning once.
        /// </summary>
        /// <param name="renderMode">The mode.</param>
        /// <param name="maxGeom">The cap.</param>
        /// <exception cref="AssertFailedException">A render returned a result of another kind than the mode's.</exception>
        /// <remarks>
        ///     Gymnasium reads the viewer's <c>scn.maxgeom</c>; Gym.NET's scene is internal (DR-69), so the cap is checked by
        ///     its effect. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_rendering.py#L89">test_max_geom_attribute</see>.
        /// </remarks>
        [TestMethod]
        [DataRow("human", 10)]
        [DataRow("human", 100)]
        [DataRow("human", 1000)]
        [DataRow("human", 10000)]
        [DataRow("rgb_array", 10)]
        [DataRow("rgb_array", 100)]
        [DataRow("rgb_array", 1000)]
        [DataRow("rgb_array", 10000)]
        [DataRow("depth_array", 10)]
        [DataRow("depth_array", 100)]
        [DataRow("depth_array", 1000)]
        [DataRow("depth_array", 10000)]
        [DataRow("rgbd_tuple", 10)]
        [DataRow("rgbd_tuple", 100)]
        [DataRow("rgbd_tuple", 1000)]
        [DataRow("rgbd_tuple", 10000)]
        public void TestMaxGeomAttribute(string renderMode, int maxGeom) {
            using IEnv env = Make("Walker2d-v5");
            // a reset places the bodies (Gymnasium's fixture leaves them all at the origin, which no frame here compares)
            using (env.Reset(seed: 0)) { }
            var mujocoEnv = (MujocoEnv)env.Unwrapped;

            // initialize renderer
            var renderer = new MujocoRenderer(mujocoEnv.Model, mujocoEnv.Data, width: DefaultSize, height: DefaultSize, maxGeom: maxGeom);

            // assert max_geom attribute
            Assert.AreEqual(maxGeom, renderer.MaxGeom);
            if (renderMode == "human")
                return;

            // the cap's effect: Walker2d-v5 draws 8 geoms, so every cap here keeps them all; 3 drops some
            // copied out: disposing the result releases its arrays (DR-51)
            NDArray Frame(MujocoRenderer r) {
                using RenderResult result = r.Render(renderMode);
                return result switch {
                    RenderResult.Rgb rgb => rgb.Frame.copy(),
                    RenderResult.Depth depth => depth.Frame.copy(),
                    RenderResult.RgbDepth both => both.RgbFrame.copy(),
                    _ => throw new AssertFailedException($"unexpected result {result}"),
                };
            }

            var all = new MujocoRenderer(mujocoEnv.Model, mujocoEnv.Data, width: DefaultSize, height: DefaultSize);
            Assert.IsTrue(np.array_equal(Frame(all), Frame(renderer)), $"max_geom={maxGeom} keeps Walker2d-v5's geoms");
            var capped = new MujocoRenderer(mujocoEnv.Model, mujocoEnv.Data, width: DefaultSize, height: DefaultSize, maxGeom: 3);
            TextWriter stdout = Console.Out;
            var printed = new StringWriter();
            Console.SetOut(printed);
            try {
                Assert.IsFalse(np.array_equal(Frame(all), Frame(capped)), "max_geom=3 drops geoms");
                Frame(capped);
            } finally {
                Console.SetOut(stdout);
            }

            Assert.AreEqual($"WARNING: Pre-allocated visual geom buffer is full. Increase maxgeom above 3.{Environment.NewLine}{Environment.NewLine}",
                            printed.ToString(), "printed once per scene, as MuJoCo's default handler does");

            // close viewer after usage
            renderer.Close();
        }

        /// <summary>Gymnasium's viewers take markers (<c>add_marker</c>); Gym.NET's renderer has none (DR-69).</summary>
        /// <remarks>
        ///     Markers are a debugging aid of Gymnasium's viewer classes, which DR-69 leaves out; adding them later is not
        ///     breaking, and this pin then becomes Gymnasium's test. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_rendering.py#L113">test_add_markers</see>.
        /// </remarks>
        [TestMethod]
        public void TestAddMarkers() {
            Assert.IsNull(typeof(MujocoRenderer).GetMethod("AddMarker"), "DR-69: no markers");
            Assert.IsNull(typeof(MujocoRenderer).GetMethod("add_marker"), "DR-69: no markers");
        }

        /// <summary>
        ///     <c>camera_id</c> is kept as given, and two envs on the same camera draw the same frames while the free camera
        ///     draws others.
        /// </summary>
        /// <param name="renderMode">The mode.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_rendering.py#L136">test_camera_id</see>.</remarks>
        [TestMethod]
        [DataRow("human")]
        [DataRow("rgb_array")]
        [DataRow("depth_array")]
        [DataRow("rgbd_tuple")]
        public void TestCameraId(string renderMode) {
            using IEnv a = Make("Ant-v5", new { camera_id = 0, render_mode = renderMode });
            using IEnv b = Make("Ant-v5", new { camera_id = 0, render_mode = renderMode });
            using IEnv c = Make("Ant-v5", new { camera_id = -1, render_mode = renderMode });
            var envA = (MujocoEnv)a.Unwrapped;
            var envB = (MujocoEnv)b.Unwrapped;
            var envC = (MujocoEnv)c.Unwrapped;

            Assert.AreEqual(envA.MujocoRenderer.CameraId, envB.MujocoRenderer.CameraId);
            Assert.AreNotEqual(envA.MujocoRenderer.CameraId, envC.MujocoRenderer.CameraId);

            if (renderMode == "rgbd_tuple") {
                using var rgbdA = (RenderResult.RgbDepth)envA.Render();
                using var rgbdB = (RenderResult.RgbDepth)envB.Render();
                using var rgbdC = (RenderResult.RgbDepth)envC.Render();
                Assert.IsTrue(np.array_equal(rgbdA.RgbFrame, rgbdB.RgbFrame));
                Assert.IsTrue(np.array_equal(rgbdA.DepthFrame, rgbdB.DepthFrame));
                Assert.IsFalse(np.array_equal(rgbdA.RgbFrame, rgbdC.RgbFrame));
                Assert.IsFalse(np.array_equal(rgbdA.DepthFrame, rgbdC.DepthFrame));
            } else if (renderMode != "human") {
                using RenderResult frameA = envA.Render(), frameB = envB.Render(), frameC = envC.Render();
                Assert.IsTrue(np.array_equal(Array(frameA), Array(frameB)));
                Assert.IsFalse(np.array_equal(Array(frameA), Array(frameC)));
            }

            static NDArray Array(RenderResult r) => r is RenderResult.Rgb rgb ? rgb.Frame : ((RenderResult.Depth)r).Frame;
        }

        /// <summary><c>rgbd_tuple</c> is <c>rgb_array</c>'s frame and <c>depth_array</c>'s depth together: uint8 (h, w, 3) and float32 (h, w).</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_rendering.py#L161">test_rgbd_tuple</see>.</remarks>
        [TestMethod]
        public void TestRgbdTuple() {
            using IEnv a = Make("Ant-v5", new { render_mode = "rgbd_tuple" });
            using IEnv b = Make("Ant-v5", new { render_mode = "rgb_array" });
            using IEnv c = Make("Ant-v5", new { render_mode = "depth_array" });

            using var rgbdA = (RenderResult.RgbDepth)a.Unwrapped.Render();
            using var rgbB = (RenderResult.Rgb)b.Unwrapped.Render();
            using var depthC = (RenderResult.Depth)c.Unwrapped.Render();

            Assert.AreEqual(NPTypeCode.Byte, rgbdA.RgbFrame.typecode);
            Assert.AreEqual(NPTypeCode.Single, rgbdA.DepthFrame.typecode);
            Assert.AreEqual(3, rgbdA.RgbFrame.ndim);
            Assert.AreEqual(2, rgbdA.DepthFrame.ndim);

            Assert.IsTrue(np.array_equal(rgbdA.RgbFrame, rgbB.Frame));
            Assert.IsTrue(np.array_equal(rgbdA.DepthFrame, depthC.Frame));
        }
    }
}
