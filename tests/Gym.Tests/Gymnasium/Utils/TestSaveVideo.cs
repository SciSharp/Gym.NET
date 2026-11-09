#nullable enable
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Gym.Envs;
using Gym.Utils;
using Gym.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Utils {
    /// <summary>Gymnasium's <c>save_video</c> tests (<c>test_save_video.py</c>), run against <see cref="Video"/>.</summary>
    /// <remarks>
    ///     Gymnasium counts the <c>.mp4</c> files moviepy wrote; the default <see cref="PngVideoWriter"/> writes each video as
    ///     a folder named like that file without its extension, so the tests count those folders. Each test writes into its
    ///     own temporary folder where Gymnasium's write into the working directory.
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_save_video.py">tests/utils/test_save_video.py</see>.
    /// </remarks>
    [TestClass]
    public class TestSaveVideo {
        /// <summary>The default trigger saves the episodes <see cref="Video.CappedCubicVideoSchedule"/> picks.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_save_video.py#L10">tests/utils/test_save_video.py (test_record_video_using_default_trigger)</see>.</remarks>
        [TestMethod]
        public void TestRecordVideoUsingDefaultTrigger() {
            string videoFolder = TemporaryFolder();
            try {
                var env = (Env<NDArray, NDArray>)Make("CartPole-v1", new { render_mode = "rgb_array_list" }, disableEnvChecker: true);
                env.Reset();
                long stepStartingIndex = 0, episodeIndex = 0;
                for (int stepIndex = 0; stepIndex < 199; stepIndex++) {
                    var (_, _, terminated, truncated, _) = env.Step(env.ActionSpace.Sample());
                    if (terminated || truncated) {
                        using (RenderResult frames = env.Render())
                            Video.SaveVideo(Frames(frames), videoFolder, fps: FpsOf(env.Metadata), stepStartingIndex: stepStartingIndex, episodeIndex: episodeIndex);
                        stepStartingIndex = stepIndex + 1;
                        episodeIndex += 1;
                        env.Reset();
                    }
                }

                env.Close();
                Assert.IsTrue(Directory.Exists(videoFolder));
                long expected = Enumerable.Range(0, (int)episodeIndex).Count(i => Video.CappedCubicVideoSchedule(i));
                Assert.AreEqual(expected, Directory.EnumerateDirectories(videoFolder).Count());
            } finally {
                Delete(videoFolder);
            }
        }

        /// <summary>A step trigger every 100 steps over 199 steps saves two clips.</summary>
        /// <remarks>
        ///     Gymnasium's <c>env._max_episode_steps = 20</c> sets an attribute on the outer <c>RenderCollection</c>, which
        ///     nothing reads; the port sets it the same way.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_save_video.py#L49">tests/utils/test_save_video.py (test_record_video_step_trigger)</see>.
        /// </remarks>
        [TestMethod]
        public void TestRecordVideoStepTrigger() {
            string videoFolder = TemporaryFolder();
            try {
                var env = (Env<NDArray, NDArray>)Make("CartPole-v1", new { render_mode = "rgb_array_list" });
                PyAttributes.SetAttr(env, "_max_episode_steps", 20L);
                env.Reset();
                long stepStartingIndex = 0, episodeIndex = 0;
                for (int stepIndex = 0; stepIndex < 199; stepIndex++) {
                    var (_, _, terminated, truncated, _) = env.Step(env.ActionSpace.Sample());
                    if (terminated || truncated) {
                        using (RenderResult frames = env.Render())
                            Video.SaveVideo(Frames(frames), videoFolder, fps: FpsOf(env.Metadata), stepTrigger: ModuloStepTrigger(100),
                                stepStartingIndex: stepStartingIndex, episodeIndex: episodeIndex);
                        stepStartingIndex = stepIndex + 1;
                        episodeIndex += 1;
                        env.Reset();
                    }
                }

                env.Close();
                Assert.IsTrue(Directory.Exists(videoFolder));
                Assert.AreEqual(2, Directory.EnumerateDirectories(videoFolder).Count());
            } finally {
                Delete(videoFolder);
            }
        }

        /// <summary>Clips of the first sub-env of a sync vector env, saved at each end of any sub-env's episode.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_save_video.py#L79">tests/utils/test_save_video.py (test_record_video_within_vector)</see>.</remarks>
        [TestMethod]
        public void TestRecordVideoWithinVector() {
            Func<long, bool> stepTrigger = ModuloStepTrigger(100);
            const int nSteps = 199;
            int expectedVideo = 2;
            string videoFolder = TemporaryFolder();
            try {
                var envs = (SyncVectorEnv)MakeVec("CartPole-v1", numEnvs: 2, vectorizationMode: "sync", kwargs: new { render_mode = "rgb_array_list" });
                envs.Reset();
                var episodeFrames = new List<NDArray>();
                long stepStartingIndex = 0, episodeIndex = 0;
                for (int stepIndex = 0; stepIndex < nSteps; stepIndex++) {
                    var (_, _, terminated, truncated, _) = envs.Step(envs.ActionSpace.Sample());
                    var rendered = (RenderResult)envs.Call("render")[0]!;
                    episodeFrames.AddRange(Frames(rendered));

                    using NDArray ended = np.logical_or(terminated, truncated);
                    if (np.any(ended)) {
                        Video.SaveVideo(episodeFrames, videoFolder, fps: FpsOf(envs.Metadata), stepTrigger: stepTrigger,
                            stepStartingIndex: stepStartingIndex, episodeIndex: episodeIndex);
                        foreach (NDArray frame in episodeFrames)
                            frame.Dispose();
                        episodeFrames = new List<NDArray>();
                        stepStartingIndex = stepIndex + 1;
                        episodeIndex += 1;

                        // TODO: fix this test (see https://github.com/openai/gym/issues/3054)
                        if (stepTrigger(stepIndex))
                            expectedVideo -= 1;
                    }
                }

                envs.Close();
                Assert.IsTrue(Directory.Exists(videoFolder));
                Assert.AreEqual(expectedVideo, Directory.EnumerateDirectories(videoFolder).Count());
            } finally {
                Delete(videoFolder);
            }
        }

        /// <summary>Gym.NET's own: the schedule's picks below 1000 are the cubes, then every 1000th; a negative episode is Python's TypeError.</summary>
        [TestMethod]
        public void TestCappedCubicVideoSchedule() {
            long[] picked = Enumerable.Range(0, 3001).Where(i => Video.CappedCubicVideoSchedule(i)).Select(i => (long)i).ToArray();
            CollectionAssert.AreEqual(new long[] { 0, 1, 8, 27, 64, 125, 216, 343, 512, 729, 1000, 2000, 3000 }, picked);
            Assert.ThrowsExactly<TypeError>(() => Video.CappedCubicVideoSchedule(-1));
        }

        /// <summary><c>modulo_step_trigger(mod)</c>: a trigger true at every multiple of <paramref name="mod"/>.</summary>
        /// <param name="mod">The period.</param>
        /// <returns>The trigger.</returns>
        private static Func<long, bool> ModuloStepTrigger(long mod) => stepIndex => stepIndex % mod == 0;

        /// <summary>The frames of an <c>rgb_array_list</c> render.</summary>
        /// <param name="rendered">The render, a <see cref="RenderResult.Frames"/>.</param>
        /// <returns>The frame arrays, still owned by <paramref name="rendered"/>.</returns>
        private static List<NDArray> Frames(RenderResult rendered) =>
            ((RenderResult.Frames)rendered).Items.Select(item => ((RenderResult.Rgb)item).Frame).ToList();

        /// <summary><c>metadata["render_fps"]</c> as the frame rate.</summary>
        /// <param name="metadata">The metadata.</param>
        /// <returns>The frame rate.</returns>
        private static double FpsOf(Metadata metadata) => PyBuiltins.Float(metadata[Metadata.RenderFpsKey]);

        /// <summary>A new folder path under the temporary directory, not created.</summary>
        /// <returns>The path.</returns>
        private static string TemporaryFolder() => Path.Combine(Path.GetTempPath(), "gymnet-videos-" + Guid.NewGuid().ToString("N"));

        /// <summary><c>shutil.rmtree(folder)</c>, when it exists.</summary>
        /// <param name="folder">The folder.</param>
        private static void Delete(string folder) {
            if (Directory.Exists(folder))
                Directory.Delete(folder, recursive: true);
        }
    }
}
