#nullable enable
using Gym.Environments;
using Gym.Wrappers.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Wrappers.Vector {
    /// <summary>Gymnasium's vector <c>HumanRendering</c> tests (<c>test_human_rendering.py</c>), run against <see cref="HumanRenderingVector{TObs, TAct}"/>.</summary>
    /// <remarks>
    ///     The window is <see cref="NullEnvViewer"/>'s (DR-20), so the tests open nothing on screen; the frames are paced at
    ///     CartPole's 50 frames per second, as in Gymnasium. The <c>Ant-v4</c> cases wait for the MuJoCo renderer (P9,
    ///     DR-62: the env steps, its frames are not drawn yet) and are <c>[OpenBugs("P9")]</c>.
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_human_rendering.py">tests/wrappers/vector/test_human_rendering.py</see>.
    /// </remarks>
    [TestClass]
    public class TestHumanRendering {
        /// <summary>CartPole in 1, 3 and 9 sub-envs, under each screen size: the wrapper reads <c>human</c> and shows a reset and 25 steps.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_human_rendering.py#L11">tests/wrappers/vector/test_human_rendering.py (test_num_envs_screen_size)</see>.</remarks>
        [TestMethod]
        public void TestNumEnvsScreenSize() => RunScreenSizes("CartPole-v1");

        /// <summary>The <c>Ant-v4</c> cases of <see cref="TestNumEnvsScreenSize"/> (P9).</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_human_rendering.py#L11">tests/wrappers/vector/test_human_rendering.py (test_num_envs_screen_size)</see>.</remarks>
        [TestMethod]
        [OpenBugs("P9")]
        public void TestNumEnvsScreenSize_Ant() => RunScreenSizes("Ant-v4");

        /// <summary>
        ///     Over <c>rgb_array_list</c> sub-envs the wrapper reads <c>human</c> and plays 25 steps; wrapping it again (a
        ///     <c>human</c> env) is refused with Gymnasium's message.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_human_rendering.py#L26">tests/wrappers/vector/test_human_rendering.py (test_render_modes)</see>.</remarks>
        [TestMethod]
        public void TestRenderModes() {
            IEnvironmentViewerFactoryDelegate? previous = HumanViewerFactory;
            HumanViewerFactory = NullEnvViewer.Factory;
            try {
                // make_vec's default for CartPole-v1 is its vector entry point, CartPoleVectorEnv, whose batches are arrays.
                using var envs = new HumanRenderingVector<NDArray, NDArray>(MakeVec<NDArray, NDArray>("CartPole-v1", numEnvs: 3, kwargs: new { render_mode = "rgb_array_list" }));
                Assert.AreEqual("human", envs.RenderMode);

                envs.Reset();
                for (int i = 0; i < 25; i++)
                    envs.Step(envs.ActionSpace.Sample());
                envs.Close();

                // HumanRenderer on human renderer should not work
                var error = Assert.ThrowsExactly<ValueError>(() => new HumanRenderingVector<NDArray, NDArray>(envs));
                StringAssert.Contains(error.Message, "Expected env.render_mode to be one of ['rgb_array', 'rgb_array_list', 'depth_array', 'depth_array_list'] but got 'human'");
            } finally {
                HumanViewerFactory = previous;
            }
        }

        /// <summary>Runs the env in 1, 3 and 9 sub-envs under each screen size (none, 400 × 300, 300 × 600, 600 × 600) through a reset and 25 sampled steps.</summary>
        /// <param name="envId">The env id.</param>
        /// <exception cref="AssertFailedException">Thrown when the wrapper does not read <c>human</c>.</exception>
        private static void RunScreenSizes(string envId) {
            IEnvironmentViewerFactoryDelegate? previous = HumanViewerFactory;
            HumanViewerFactory = NullEnvViewer.Factory;
            try {
                foreach (int numEnvs in new[] { 1, 3, 9 }) {
                    foreach ((int Width, int Height)? screenSize in new (int, int)?[] { null, (400, 300), (300, 600), (600, 600) }) {
                        using var envs = new HumanRenderingVector<NDArray, NDArray>(MakeVec<NDArray, NDArray>(envId, numEnvs: numEnvs, kwargs: new { render_mode = "rgb_array" }), screenSize);
                        Assert.AreEqual("human", envs.RenderMode, $"{numEnvs} envs, screen {screenSize}");

                        envs.Reset();
                        for (int i = 0; i < 25; i++)
                            envs.Step(envs.ActionSpace.Sample());
                        envs.Close();
                    }
                }
            } finally {
                HumanViewerFactory = previous;
            }
        }
    }
}
