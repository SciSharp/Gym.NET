#nullable enable
using Gym.Environments;
using Gym.Envs;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Wrappers {
    /// <summary>Gymnasium's <c>HumanRendering</c> tests (<c>test_human_rendering.py</c>), run against <see cref="HumanRendering{TObs, TAct}"/>.</summary>
    /// <remarks>
    ///     The window is <see cref="NullEnvViewer"/>'s (DR-20), so the tests open nothing on screen; the frames are paced at
    ///     CartPole's 50 frames per second, as in Gymnasium.
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_human_rendering.py">tests/wrappers/test_human_rendering.py</see>.
    /// </remarks>
    [TestClass]
    public class TestHumanRendering {
        /// <summary>
        ///     Over <c>rgb_array</c> and <c>rgb_array_list</c> the wrapper reads <c>human</c> and plays 75 steps; an env already
        ///     rendering <c>human</c> is refused with Gymnasium's message.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_human_rendering.py#L11">tests/wrappers/test_human_rendering.py (test_human_rendering)</see>.</remarks>
        [TestMethod]
        public void TestHumanRenderingWrapper() {
            IEnvironmentViewerFactoryDelegate? previous = HumanViewerFactory;
            HumanViewerFactory = NullEnvViewer.Factory;
            try {
                foreach (string mode in new[] { "rgb_array", "rgb_array_list" }) {
                    var env = new HumanRendering<NDArray, NDArray>((Env<NDArray, NDArray>)Make("CartPole-v1", new { render_mode = mode }, disableEnvChecker: true));
                    Assert.AreEqual("human", env.RenderMode);
                    env.Reset();

                    for (int i = 0; i < 75; i++) {
                        var (_, _, terminated, truncated, _) = env.Step(env.ActionSpace.Sample());
                        if (terminated || truncated)
                            env.Reset();
                    }

                    env.Close();
                }

                using var human = (Env<NDArray, NDArray>)Make("CartPole-v1", new { render_mode = "human" });
                var error = Assert.ThrowsExactly<ValueError>(() => new HumanRendering<NDArray, NDArray>(human));
                StringAssert.Contains(error.Message, "Expected env.render_mode to be one of ['rgb_array', 'rgb_array_list', 'depth_array', 'depth_array_list'] but got 'human'");
            } finally {
                HumanViewerFactory = previous;
            }
        }
    }
}
