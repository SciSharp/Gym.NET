#nullable enable
using Gym.Environments;
using Gym.Envs;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Wrappers {
    /// <summary>Gymnasium's <c>AddWhiteNoise</c> and <c>ObstructView</c> tests (<c>test_white_noise_rendering.py</c>), run against <see cref="AddWhiteNoise{TObs, TAct}"/> and <see cref="ObstructView{TObs, TAct}"/>.</summary>
    /// <remarks>
    ///     The window is <see cref="NullEnvViewer"/>'s (DR-20), so nothing opens on screen.
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_white_noise_rendering.py">tests/wrappers/test_white_noise_rendering.py</see>.
    /// </remarks>
    [TestClass]
    public class TestWhiteNoiseRendering {
        /// <summary>Both noise wrappers under <c>HumanRendering</c> read <c>human</c> and play 75 steps over CartPole's frames.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_white_noise_rendering.py#L7">tests/wrappers/test_white_noise_rendering.py (test_white_noise_rendering)</see>.</remarks>
        [TestMethod]
        public void TestWhiteNoiseRenderingWrappers() {
            IEnvironmentViewerFactoryDelegate? previous = HumanViewerFactory;
            HumanViewerFactory = NullEnvViewer.Factory;
            try {
                foreach (string mode in new[] { "rgb_array" }) {
                    var noisy = new AddWhiteNoise<NDArray, NDArray>(
                        (Env<NDArray, NDArray>)Make("CartPole-v1", new { render_mode = mode }, disableEnvChecker: true), probabilityOfNoisePerPixel: 0.5);
                    Play(new HumanRendering<NDArray, NDArray>(noisy));

                    var obstructed = new ObstructView<NDArray, NDArray>(
                        (Env<NDArray, NDArray>)Make("CartPole-v1", new { render_mode = mode }, disableEnvChecker: true), obstructedPixelsRatio: 0.5, obstructionWidth: 100);
                    Play(new HumanRendering<NDArray, NDArray>(obstructed));
                }
            } finally {
                HumanViewerFactory = previous;
            }
        }

        /// <summary>Plays 75 random steps, resetting at episode ends, and closes the env.</summary>
        /// <param name="env">The env, rendering <c>human</c>.</param>
        /// <exception cref="AssertFailedException">Thrown when the render mode is not <c>human</c>.</exception>
        private static void Play(HumanRendering<NDArray, NDArray> env) {
            Assert.AreEqual("human", env.RenderMode);
            env.Reset();

            for (int i = 0; i < 75; i++) {
                var (_, _, terminated, truncated, _) = env.Step(env.ActionSpace.Sample());
                if (terminated || truncated)
                    env.Reset();
            }

            env.Close();
        }
    }
}
