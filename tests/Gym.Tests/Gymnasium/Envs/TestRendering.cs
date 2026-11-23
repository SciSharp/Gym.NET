#nullable enable
using System.Collections.Generic;
using System.Linq;
using Gym.Envs;
using Gym.Envs.Registration;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Envs {
    /// <summary>
    ///     Gymnasium's rendering tests (<c>tests/envs/test_rendering.py</c>): every env that can be made renders in every
    ///     render mode its metadata lists but <c>human</c>, after the reset and after a sampled step, and each frame has the
    ///     shape and type of its mode.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <c>test_pygame_selective_init</c> has no counterpart: it checks that an <c>rgb_array</c> frame initialises pygame's
    ///     display module and not its mixer, and Gym.NET draws frames without SDL, so there is neither subsystem to
    ///     initialise (an env's <c>human</c> window is a viewer, never <c>Gym.Rendering.Display</c>).
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/test_rendering.py">tests/envs/test_rendering.py</see>.</para>
    /// </remarks>
    [TestClass]
    public class TestRendering {
        /// <summary>The ids of every registered env Gym.NET can make (<c>render_mode_env_specs</c>, all of <c>all_testing_env_specs</c>).</summary>
        public static IEnumerable<object[]> RenderModeEnvIds => EnvTestingUtils.AllTestingEnvSpecs().Select(spec => new object[] { spec.Id });

        /// <summary>
        ///     Checks that a rendered frame is what its mode promises (<c>check_rendered</c>): a list of such frames for the
        ///     <c>*_list</c> modes, an <c>(H, W, 3)</c> array of values within 0 to 255 for <c>rgb_array</c> and
        ///     <c>state_pixels</c>, non-empty text for <c>ansi</c>, a two-dimensional array for <c>depth_array</c>, and two
        ///     arrays for <c>rgbd_tuple</c>.
        /// </summary>
        /// <param name="renderedFrame">What <c>render()</c> returned.</param>
        /// <param name="mode">The render mode it was rendered in.</param>
        /// <exception cref="AssertFailedException">Thrown when the frame is not of its mode's case, shape or range.</exception>
        /// <remarks>
        ///     <para>
        ///     Gymnasium warns and passes for a mode it has no case for; so does this check (as an inconclusive-free pass with
        ///     the same warning through <see cref="Logger"/>), since an env may list a mode of its own.
        ///     </para>
        ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/test_rendering.py#L40">tests/envs/test_rendering.py (check_rendered)</see>.</para>
        /// </remarks>
        public static void CheckRendered(RenderResult renderedFrame, string mode) {
            switch (mode) {
                case "rgb_array_list":
                case "state_pixels_list": {
                    // Python's isinstance(rendered_frame, list), then every frame as rgb_array
                    var frames = AssertCase<RenderResult.Frames>(renderedFrame, mode);
                    foreach (RenderResult frame in frames.Items)
                        CheckRendered(frame, "rgb_array");
                    break;
                }
                case "rgb_array":
                case "state_pixels": {
                    NDArray frame = AssertCase<RenderResult.Rgb>(renderedFrame, mode).Frame;
                    Assert.AreEqual(3, frame.ndim, $"{mode}: an (H, W, 3) frame, got shape ({string.Join(", ", frame.shape)})");
                    Assert.AreEqual(3L, frame.shape[2], $"{mode}: three channels, got shape ({string.Join(", ", frame.shape)})");
                    // np.all(rendered_frame >= 0) and np.all(rendered_frame <= 255): read as doubles, so a frame of any dtype
                    // is held to the byte range the way NumPy's comparisons hold it
                    using (NDArray wide = frame.astype(np.float64)) {
                        double[] values = wide.ToArray<double>();
                        Assert.IsTrue(values.All(v => v >= 0 && v <= 255), $"{mode}: every value within 0 to 255");
                    }

                    break;
                }
                case "ansi": {
                    string text = AssertCase<RenderResult.Ansi>(renderedFrame, mode).Text;
                    Assert.IsTrue(text.Length > 0, "ansi: non-empty text");
                    break;
                }
                case "depth_array_list": {
                    var frames = AssertCase<RenderResult.Frames>(renderedFrame, mode);
                    foreach (RenderResult frame in frames.Items)
                        CheckRendered(frame, "depth_array");
                    break;
                }
                case "depth_array": {
                    NDArray frame = AssertCase<RenderResult.Depth>(renderedFrame, mode).Frame;
                    Assert.AreEqual(2, frame.ndim, $"depth_array: an (H, W) frame, got shape ({string.Join(", ", frame.shape)})");
                    break;
                }
                case "rgbd_tuple":
                    // rendered_rgb, rendered_depth = rendered_frame: both arrays, which the pair's constructor guarantees
                    AssertCase<RenderResult.RgbDepth>(renderedFrame, mode);
                    break;
                default:
                    Gym.Logger.Warn($"Unknown render mode: {mode}, cannot check that the rendered data is correct. Add case to `check_rendered`");
                    break;
            }
        }

        /// <summary>Asserts a render result's case, Python's <c>isinstance</c> on what <c>render()</c> returned.</summary>
        /// <typeparam name="T">The case the mode returns.</typeparam>
        /// <param name="rendered">The result.</param>
        /// <param name="mode">The mode, for the message.</param>
        /// <returns>The result as its case.</returns>
        /// <exception cref="AssertFailedException">Thrown when the result is another case.</exception>
        private static T AssertCase<T>(RenderResult rendered, string mode) where T : RenderResult {
            Assert.IsInstanceOfType<T>(rendered, $"{mode}: render() returned {rendered}");
            return (T)rendered;
        }

        /// <summary>
        ///     Every render mode an env lists but <c>human</c> renders after the reset and after a sampled step
        ///     (<c>test_render_modes</c>); every env lists <c>rgb_array</c>.
        /// </summary>
        /// <param name="envId">The env's id; its minimal spec (<c>all_testing_env_specs</c>) is made, as Gymnasium's <c>spec.make()</c>.</param>
        /// <remarks>
        ///     <para>
        ///     Gymnasium skips the env when OpenGL cannot load (<c>gladLoadGL error</c>, MuJoCo on a machine without a GPU);
        ///     Gym.NET draws MuJoCo's frames itself, so every MuJoCo id renders here whatever the machine.
        ///     </para>
        ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/test_rendering.py#L84">tests/envs/test_rendering.py (test_render_modes)</see>.</para>
        /// </remarks>
        [TestMethod]
        [DynamicData(nameof(RenderModeEnvIds))]
        public void TestRenderModes(string envId) {
            EnvSpec spec = EnvTestingUtils.AllTestingEnvSpecs().Single(s => s.Id == envId);
            using IEnv env = spec.Make();

            // env.metadata["render_modes"]: every built-in env lists its modes, so a missing list fails here as Python's KeyError would
            Assert.IsNotNull(env.Metadata.RenderModes, $"{envId} lists no render modes");
            List<string> renderModes = env.Metadata.RenderModes.ToList();
            CollectionAssert.Contains(renderModes, "rgb_array");

            foreach (string mode in renderModes) {
                if (mode == "human")
                    continue;
                // try/finally: Gymnasium's new_env.close() runs whether or not a check fails
                IEnv newEnv = spec.Make(new Dictionary<string, object?> { ["render_mode"] = mode });
                try {
                    newEnv.Reset().Dispose();
                    using (RenderResult rendered = newEnv.Render())
                        CheckRendered(rendered, mode);

                    object? action = newEnv.ActionSpace.Sample();
                    newEnv.Step(action).Dispose();
                    (action as NDArray)?.Dispose();
                    using (RenderResult rendered = newEnv.Render())
                        CheckRendered(rendered, mode);
                } finally {
                    newEnv.Close();
                    newEnv.Dispose();
                }
            }

            env.Close();
        }
    }
}
