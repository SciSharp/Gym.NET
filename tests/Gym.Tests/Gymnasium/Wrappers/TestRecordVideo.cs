#nullable enable
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Gym.Envs;
using Gym.Utils;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Wrappers {
    /// <summary>Gymnasium's <c>RecordVideo</c> tests (<c>test_record_video.py</c>), run against <see cref="RecordVideo{TObs, TAct}"/>.</summary>
    /// <remarks>
    ///     Gymnasium counts the <c>.mp4</c> files moviepy wrote; the default <see cref="PngVideoWriter"/> writes each video as
    ///     a folder named like that file without its extension, so the tests count those folders, and one test runs
    ///     <see cref="FfmpegVideoWriter"/> for real mp4 files when <c>ffmpeg</c> is installed. Each test writes into its own
    ///     temporary folder where Gymnasium's write into the working directory.
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_record_video.py">tests/wrappers/test_record_video.py</see>.
    /// </remarks>
    [TestClass]
    public class TestRecordVideo {
        /// <summary>The episode and step triggers name their videos with the prefix: steps 0 and 25, episodes 1 and 4.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_record_video.py#L13">tests/wrappers/test_record_video.py (test_video_folder_and_filenames)</see>.</remarks>
        [TestMethod]
        public void TestVideoFolderAndFilenames() {
            string videoFolder = TemporaryFolder();
            try {
                var env = new RecordVideo<NDArray, NDArray>(MakeRgb(), videoFolder, namePrefix: "video-prefix",
                    episodeTrigger: x => x is 1 or 4, stepTrigger: x => x is 0 or 25);
                env.Reset(seed: 123);
                env.ActionSpace.Seed(123);
                for (int i = 0; i < 100; i++) {
                    var (_, _, terminated, truncated, _) = env.Step(env.ActionSpace.Sample());
                    if (terminated || truncated)
                        env.Reset();
                }

                env.Close();

                Assert.IsTrue(Directory.Exists(videoFolder));
                CollectionAssert.AreEquivalent(new[] { "video-prefix-step-0", "video-prefix-step-25", "video-prefix-episode-1", "video-prefix-episode-4" }, Videos(videoFolder));
            } finally {
                Delete(videoFolder);
            }
        }

        /// <summary>The default trigger (and an explicit one) records the episodes it picks among those played.</summary>
        /// <param name="explicitTrigger">Whether <c>x in [0, 3, 5, 10, 12]</c> is given instead of the default.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_record_video.py#L46">tests/wrappers/test_record_video.py (test_episodic_trigger)</see>.</remarks>
        [TestMethod]
        [DataRow(false)]
        [DataRow(true)]
        public void TestEpisodicTrigger(bool explicitTrigger) {
            string videoFolder = TemporaryFolder();
            try {
                Func<long, bool>? trigger = explicitTrigger ? x => x is 0 or 3 or 5 or 10 or 12 : null;
                var env = new RecordVideo<NDArray, NDArray>(MakeRgb(), videoFolder, episodeTrigger: trigger);
                env.Reset();
                int episodeCount = 0;
                for (int i = 0; i < 199; i++) {
                    var (_, _, terminated, truncated, _) = env.Step(env.ActionSpace.Sample());
                    if (terminated || truncated) {
                        env.Reset();
                        episodeCount++;
                    }
                }

                env.Close();

                Assert.IsTrue(Directory.Exists(videoFolder));
                Assert.IsNotNull(env.EpisodeTrigger);
                Assert.AreEqual(Enumerable.Range(0, episodeCount + 1).Count(i => env.EpisodeTrigger!(i)), Videos(videoFolder).Count);
            } finally {
                Delete(videoFolder);
            }
        }

        /// <summary>A step trigger every 100 steps over 199 steps records two videos.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_record_video.py#L70">tests/wrappers/test_record_video.py (test_step_trigger)</see>.</remarks>
        [TestMethod]
        public void TestStepTrigger() {
            string videoFolder = TemporaryFolder();
            try {
                var env = new RecordVideo<NDArray, NDArray>(MakeRgb(), videoFolder, stepTrigger: x => x % 100 == 0);
                env.Reset();
                for (int i = 0; i < 199; i++) {
                    var (_, _, terminated, truncated, _) = env.Step(env.ActionSpace.Sample());
                    if (terminated || truncated)
                        env.Reset();
                }

                env.Close();
                Assert.IsTrue(Directory.Exists(videoFolder));
                Assert.AreEqual(2, Videos(videoFolder).Count);
            } finally {
                Delete(videoFolder);
            }
        }

        /// <summary>Step 100 and episodes 0 and 3 record three videos (the seed's episode ends make episode 3 and step 100 distinct).</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_record_video.py#L87">tests/wrappers/test_record_video.py (test_both_episodic_and_step_trigger)</see>.</remarks>
        [TestMethod]
        public void TestBothEpisodicAndStepTrigger() {
            string videoFolder = TemporaryFolder();
            try {
                var env = new RecordVideo<NDArray, NDArray>(MakeRgb(), videoFolder, stepTrigger: x => x == 100, episodeTrigger: x => x is 0 or 3);
                // episode reset time steps: 0, 18, 44, 55, 80, 103, 117, 143, 173, 191
                // steps recorded: 0-18, 55-80, 100-103
                env.Reset(seed: 123);
                env.ActionSpace.Seed(123);
                for (int i = 0; i < 199; i++) {
                    var (_, _, terminated, truncated, _) = env.Step(env.ActionSpace.Sample());
                    if (terminated || truncated)
                        env.Reset();
                }

                env.Close();
                Assert.IsTrue(Directory.Exists(videoFolder));
                Assert.AreEqual(3, Videos(videoFolder).Count);
            } finally {
                Delete(videoFolder);
            }
        }

        /// <summary>A fixed video length stops the recording once the frames pass it, and only one video is written.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_record_video.py#L114">tests/wrappers/test_record_video.py (test_video_length)</see>.</remarks>
        [TestMethod]
        public void TestVideoLength() {
            const int videoLength = 10;
            string videoFolder = TemporaryFolder();
            try {
                var env = new RecordVideo<NDArray, NDArray>(MakeRgb(), videoFolder, stepTrigger: x => x == 0, videoLength: videoLength);
                env.Reset(seed: 123);
                env.ActionSpace.Seed(123);
                for (int i = 0; i < videoLength; i++) {
                    var (_, _, term, trunc, _) = env.Step(env.ActionSpace.Sample());
                    if (term || trunc)
                        break;
                }

                // check that the environment is still recording then take a step to take the number of steps > video length
                Assert.IsTrue(env.Recording);
                env.Step(env.ActionSpace.Sample());
                Assert.IsFalse(env.Recording);
                env.Close();

                // check that only one video is recorded
                Assert.IsTrue(Directory.Exists(videoFolder));
                Assert.AreEqual(1, Videos(videoFolder).Count);
            } finally {
                Delete(videoFolder);
            }
        }

        /// <summary><c>render()</c> through <c>RecordVideo</c> and <c>RenderCollection</c>, in both orders: every frame since the reset, then none.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_record_video.py#L141">tests/wrappers/test_record_video.py (test_with_rgb_array_list)</see>.</remarks>
        [TestMethod]
        public void TestWithRgbArrayList() {
            const int nSteps = 10;
            string videoFolder = TemporaryFolder();
            try {
                IEnv env = new RecordVideo<NDArray, NDArray>(new RenderCollection<NDArray, NDArray>(MakeRgb()), videoFolder);
                AssertCollectedFrames(env, nSteps);
                env.Close();
                Delete(videoFolder);

                // Test in reverse order
                env = new RenderCollection<NDArray, NDArray>(new RecordVideo<NDArray, NDArray>(MakeRgb(), videoFolder));
                AssertCollectedFrames(env, nSteps);
                env.Close();
            } finally {
                Delete(videoFolder);
            }
        }

        /// <summary>An env that renders no image is refused with one sentence telling what to do.</summary>
        /// <remarks>
        ///     Gymnasium also calls <c>__del__</c> on a half-built wrapper; a C# constructor that throws leaves no object to
        ///     finalize there.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_record_video.py#L188">tests/wrappers/test_record_video.py (test_incompatible_render_mode)</see>.
        /// </remarks>
        [TestMethod]
        public void TestIncompatibleRenderMode() {
            string videoFolder = TemporaryFolder();
            try {
                using var env = (Env<NDArray, NDArray>)Make("CartPole-v1"); // render_mode is None
                var error = Assert.ThrowsExactly<ValueError>(() => new RecordVideo<NDArray, NDArray>(env, videoFolder));
                StringAssert.Contains(error.Message, "such as rgb_array");
            } finally {
                Delete(videoFolder);
            }
        }

        /// <summary>
        ///     Gym.NET's own: <see cref="FfmpegVideoWriter"/> writes the triggered videos as mp4 files of Gymnasium's names, when
        ///     <c>ffmpeg</c> is installed (Inconclusive otherwise).
        /// </summary>
        /// <remarks>
        ///     The writer starts ffmpeg when a recording stops, and here that happens inside <c>Close</c> (the stack trace of
        ///     the failure on GitHub's Windows runner, which has no ffmpeg, shows <c>Close</c> calling <c>StopRecording</c>).
        ///     So the Inconclusive check wraps the close as well as the steps; it wrapped only the steps, and a machine without
        ///     ffmpeg failed the test with <see cref="Gym.Exceptions.DependencyNotInstalled"/> from the <c>finally</c>.
        /// </remarks>
        /// <exception cref="AssertInconclusiveException">Thrown when ffmpeg is not installed or not on <c>PATH</c>.</exception>
        /// <exception cref="AssertFailedException">Thrown when ffmpeg ran but wrote no non-empty <c>rl-video-episode-0.mp4</c>.</exception>
        [TestMethod]
        public void TestFfmpegWriterWritesMp4Files() {
            string videoFolder = TemporaryFolder();
            try {
                var env = new RecordVideo<NDArray, NDArray>(MakeRgb(), videoFolder, videoWriter: new FfmpegVideoWriter(), episodeTrigger: x => x == 0, videoLength: 5);
                try {
                    try {
                        env.Reset(seed: 123);
                        for (int i = 0; i < 6; i++)
                            env.Step(env.ActionSpace.Sample());
                    } finally {
                        // Close stops the open recording, which is when the writer starts ffmpeg. It runs after a failed step
                        // too, so the env is released; a missing ffmpeg it reports (again, after a step's) reaches the catch below.
                        env.Close();
                    }
                } catch (Gym.Exceptions.DependencyNotInstalled e) {
                    Assert.Inconclusive(e.Message);
                }

                string video = Path.Combine(videoFolder, "rl-video-episode-0.mp4");
                Assert.IsTrue(File.Exists(video), "ffmpeg wrote no rl-video-episode-0.mp4");
                Assert.IsTrue(new FileInfo(video).Length > 0);
            } finally {
                Delete(videoFolder);
            }
        }

        /// <summary>Steps through an env collecting frames and checks what its render returns: every frame since the reset as arrays, then nothing.</summary>
        /// <param name="env">The env, a <c>RenderCollection</c> and a <c>RecordVideo</c> in either order.</param>
        /// <param name="nSteps">The steps.</param>
        /// <exception cref="AssertFailedException">Thrown when a render differs.</exception>
        private static void AssertCollectedFrames(IEnv env, int nSteps) {
            env.Reset(seed: 123);
            env.ActionSpace.Seed(123);
            for (int i = 0; i < nSteps; i++)
                env.Step(env.ActionSpace.Sample());

            using (RenderResult renderOut = env.Render()) {
                Assert.IsInstanceOfType<RenderResult.Frames>(renderOut);
                var frames = (RenderResult.Frames)renderOut;
                Assert.AreEqual(nSteps + 1, frames.Items.Count);
                Assert.IsTrue(frames.Items.All(item => item is RenderResult.Rgb { Frame.ndim: 3 }));
            }

            using (RenderResult renderOut = env.Render()) {
                Assert.IsInstanceOfType<RenderResult.Frames>(renderOut);
                Assert.AreEqual(0, ((RenderResult.Frames)renderOut).Items.Count);
            }
        }

        /// <summary>Gymnasium's <c>gym.make("CartPole-v1", render_mode="rgb_array")</c>.</summary>
        /// <returns>The env, which the wrapper over it owns.</returns>
        private static Env<NDArray, NDArray> MakeRgb() => (Env<NDArray, NDArray>)Make("CartPole-v1", new { render_mode = "rgb_array" });

        /// <summary>The videos in a folder: the folders the PNG writer made, or the mp4 files without extension.</summary>
        /// <param name="videoFolder">The folder.</param>
        /// <returns>The video names.</returns>
        private static List<string> Videos(string videoFolder) =>
            Directory.EnumerateDirectories(videoFolder).Select(Path.GetFileName)
                .Concat(Directory.EnumerateFiles(videoFolder, "*.mp4").Select(Path.GetFileNameWithoutExtension))
                .Select(name => name!).ToList();

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
