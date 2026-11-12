#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Environments;
using Gym.Environments.Envs.MuJoCo;
using Gym.Envs;
using Gym.Observations;
using Gym.Spaces;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Envs.MuJoCo {
    /// <summary>
    ///     Gymnasium's third-party MuJoCo env tests (<c>tests/envs/mujoco/test_mujoco_custom_env.py</c>): a
    ///     <see cref="MujocoEnv"/> subclass of the test's own, <see cref="PointEnv"/> on the bundled <c>point.xml</c>, under
    ///     five frame skips through <c>check_env</c>, its reset info, the model files it may load, and the offscreen buffer.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Two cases load XML files of the test folder, which Gym.NET refuses: it embeds Gymnasium's 14 compiled models and
    ///         has no MJCF compiler (DR-63). <see cref="TestXmlFile"/> pins that refusal, and
    ///         <see cref="TestOffscreenFramebufferNotShrunk"/> checks the same growth rule on <c>point.xml</c>'s own buffer.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_custom_env.py">tests/envs/mujoco/test_mujoco_custom_env.py</see>.</para>
    /// </remarks>
    [TestClass]
    public class TestMujocoCustomEnv {
        /// <summary><c>check_env</c>'s warnings a MuJoCo env may raise (Gymnasium's <c>CHECK_ENV_IGNORE_WARNINGS</c>).</summary>
        private static readonly string[] CheckEnvIgnoreWarnings = new[] {
            "A Box observation space minimum value is -infinity. This is probably too low.",
            "A Box observation space maximum value is infinity. This is probably too high.",
            "For Box action spaces, we recommend using a symmetric and normalized space (range=[-1, 1] or [0, 1]). See https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html for more information.",
        }.Select(message => $"\u001b[33mWARN: {message}\u001b[0m").ToArray();

        /// <summary>A third-party env built on <see cref="MujocoEnv"/> passes <c>check_env</c> under frame skips 1 to 5.</summary>
        /// <param name="frameSkip">The frame skip.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_custom_env.py#L99">test_frame_skip</see>.</remarks>
        [TestMethod]
        [DataRow(1)]
        [DataRow(2)]
        [DataRow(3)]
        [DataRow(4)]
        [DataRow(5)]
        public void TestFrameSkip(int frameSkip) {
            using var env = new PointEnv(frameSkip: frameSkip);

            // Test if env adheres to Gym API
            using WarningCatcher w = Warnings.Catch();
            EnvChecker.CheckEnv(env.Unwrapped, skipRenderCheck: true);
            env.Close();
            foreach (WarningMessage warning in w.Records) {
                if (!CheckEnvIgnoreWarnings.Contains(warning.Message))
                    Assert.Fail($"Unexpected warning: {warning.Message}");
            }
        }

        /// <summary>
        ///     A model file of the test folder, by relative or absolute path: Gymnasium compiles it; Gym.NET refuses it with the
        ///     bundled names (DR-63), and takes a bundled model of the same structure (Walker2d-v5's) instead.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_custom_env.py#L113">test_xml_file</see>.</remarks>
        [TestMethod]
        public void TestXmlFile() {
            const string relativePath = "./tests/envs/mujoco/assets/walker2d_v5_uneven_feet.xml";
            StringAssert.Contains(Assert.ThrowsExactly<NotSupportedException>(() => new PointEnv(xmlFile: relativePath)).Message,
                $"'{relativePath}' is not one of");
            string fullPath = Environment.CurrentDirectory + "/tests/envs/mujoco/assets/walker2d_v5_uneven_feet.xml";
            Assert.ThrowsExactly<NotSupportedException>(() => new PointEnv(xmlFile: fullPath));

            using var env = new PointEnv(xmlFile: "walker2d_v5.xml");
            Assert.AreEqual(9, env.Model.Nq);
        }

        /// <summary>The env's own reset info reaches the caller.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_custom_env.py#L128">test_reset_info</see>.</remarks>
        [TestMethod]
        public void TestResetInfo() {
            using var env = new PointEnv();
            using ResetResult<NDArray> reset = env.Reset();
            Assert.AreEqual(true, reset.Info["works"]);
        }

        /// <summary>
        ///     The offscreen buffer only grows: a model's own size survives a smaller render window, and a larger window
        ///     enlarges it (Gymnasium's regression test of issue 1607, on <c>point.xml</c>'s buffer instead of a written XML).
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_custom_env.py#L154">test_offscreen_framebuffer_not_shrunk</see>.</remarks>
        [TestMethod]
        public void TestOffscreenFramebufferNotShrunk() {
            int ownWidth, ownHeight;
            using (MjModelProbe probe = new())
                (ownWidth, ownHeight) = (probe.Width, probe.Height);

            // a smaller render window keeps the model's own buffer
            using (var env = new PointEnv(width: ownWidth - 1, height: ownHeight - 1)) {
                Assert.AreEqual(ownWidth, env.Model.Vis.Global.Offwidth);
                Assert.AreEqual(ownHeight, env.Model.Vis.Global.Offheight);
            }

            // When the requested render window is larger than the XML declaration, the larger window size wins so that
            // on-screen rendering still fits the buffer.
            using (var env = new PointEnv(width: 2560, height: 1440)) {
                Assert.AreEqual(2560, env.Model.Vis.Global.Offwidth);
                Assert.AreEqual(1440, env.Model.Vis.Global.Offheight);
            }
        }

        /// <summary>point.xml's own offscreen buffer, read from a freshly loaded model.</summary>
        private sealed class MjModelProbe : IDisposable {
            /// <summary>The loaded model, released by <see cref="Dispose"/>.</summary>
            private readonly Gym.MuJoCo.MjModel model = Gym.MuJoCo.MjModel.FromXmlPath("point.xml");

            /// <summary>The model's offscreen width.</summary>
            public int Width => model.Vis.Global.Offwidth;

            /// <summary>The model's offscreen height.</summary>
            public int Height => model.Vis.Global.Offheight;

            /// <summary>Releases the model.</summary>
            public void Dispose() => model.Dispose();
        }

        /// <summary>
        ///     Gymnasium's test <c>PointEnv</c>: a third-party env on <see cref="MujocoEnv"/> over <c>point.xml</c>, rewarded by
        ///     its x displacement, reset to the reference state, reporting <c>{"works": True}</c> at reset.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_custom_env.py#L17">PointEnv</see>.</remarks>
        public sealed class PointEnv : MujocoEnv {
            /// <summary>Loads the model and sets the spaces and the frame rate the time step gives.</summary>
            /// <param name="xmlFile">The model, a bundled file name.</param>
            /// <param name="frameSkip">Physics steps per env step.</param>
            /// <param name="width">The render window's width.</param>
            /// <param name="height">The render window's height.</param>
            public PointEnv(string xmlFile = "point.xml", int frameSkip = 1, int width = DefaultSize, int height = DefaultSize)
                : base(xmlFile, frameSkip, null, ClassMetadata, width: width, height: height,
                       defaultCameraConfig: new Dictionary<string, object?>()) {
                Metadata = new Metadata {
                    [Metadata.RenderModesKey] = new List<string> { "human", "rgb_array", "depth_array", "rgbd_tuple" },
                    [Metadata.RenderFpsKey] = (long)Math.Round(1.0 / Dt, MidpointRounding.ToEven),
                };
                ObservationSpace = new Box(double.NegativeInfinity, double.PositiveInfinity, new Shape(Model.Nq + Model.Nv), np.float64);
            }

            /// <summary>The class's metadata: the four render modes, the frame rate set per instance.</summary>
            public static new Metadata ClassMetadata => new() {
                [Metadata.RenderModesKey] = new List<string> { "human", "rgb_array", "depth_array", "rgbd_tuple" },
            };

            /// <summary>Steps the simulation and rewards the root's x displacement.</summary>
            /// <param name="action">The controls.</param>
            /// <returns>The observation, the displacement, never terminated or truncated, an empty info.</returns>
            public override StepResult<NDArray> Step(NDArray action) {
                double xPositionBefore = Data.qpos[0];
                DoSimulation(action, FrameSkip);
                double xPositionAfter = Data.qpos[0];
                return new StepResult<NDArray>(GetObs(), xPositionAfter - xPositionBefore, false, false, new Info());
            }

            /// <summary>The reference state, without noise.</summary>
            /// <returns>The first observation.</returns>
            protected override NDArray ResetModel() {
                SetState(InitQpos, InitQvel);
                return GetObs();
            }

            /// <summary>The reset info the test looks for.</summary>
            /// <returns><c>{"works": True}</c>.</returns>
            protected override Info GetResetInfo() => new() { ["works"] = true };

            /// <summary>The positions, then the velocities.</summary>
            /// <returns>A new float64 array.</returns>
            private NDArray GetObs() => np.array(Data.qpos.Concat(Data.qvel).ToArray());
        }
    }
}
