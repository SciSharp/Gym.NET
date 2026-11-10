#nullable enable
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Gym.Envs;
using Gym.Utils;
using Gym.Vector;
using Gym.Wrappers.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Wrappers.Vector {
    /// <summary>Gymnasium's vector <c>RecordVideo</c> tests (<c>test_record_video.py</c>), run against <see cref="RecordVideoVector{TObs, TAct}"/>.</summary>
    /// <remarks>
    ///     Gymnasium counts the <c>.mp4</c> files moviepy wrote; the default <see cref="PngVideoWriter"/> writes each video as
    ///     a folder named like that file without its extension (DR-25), so the tests count those folders. Each test writes
    ///     into its own temporary folder where Gymnasium's write into the working directory, and each runs under every
    ///     autoreset mode, <c>DISABLED</c> resetting the ended sub-envs through <c>reset_mask</c>.
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_record_video.py">tests/wrappers/vector/test_record_video.py</see>.
    /// </remarks>
    [TestClass]
    public class TestRecordVideo {
        /// <summary>The episode and step triggers name their videos with the prefix: steps 0 and 30, episodes 1 and 4.</summary>
        /// <param name="autoresetMode">The autoreset mode's value.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_record_video.py#L14">tests/wrappers/vector/test_record_video.py (test_video_folder_and_filenames)</see>.</remarks>
        [TestMethod]
        [DataRow("NextStep")]
        [DataRow("SameStep")]
        [DataRow("Disabled")]
        public void TestVideoFolderAndFilenames(string autoresetMode) {
            string videoFolder = TemporaryFolder();
            try {
                using var envs = new RecordVideoVector<NDArray, NDArray>(MakeRgb(10, autoresetMode), videoFolder, namePrefix: "video-prefix",
                    episodeTrigger: x => x is 1 or 4, stepTrigger: x => x is 0 or 30, videoAspectRatio: (1, 1));

                envs.Reset(seed: 123);
                envs.ActionSpace.Seed(123);
                for (int i = 0; i < 150; i++)
                    StepAndReset(envs, autoresetMode);

                envs.Close();

                Assert.IsTrue(Directory.Exists(videoFolder));
                CollectionAssert.AreEquivalent(new[] { "video-prefix-step-0", "video-prefix-step-30", "video-prefix-episode-1", "video-prefix-episode-4" }, Videos(videoFolder));
            } finally {
                Delete(videoFolder);
            }
        }

        /// <summary>The default trigger (and an explicit one) records the first sub-env's episodes it picks among those played.</summary>
        /// <param name="autoresetMode">The autoreset mode's value.</param>
        /// <param name="explicitTrigger">Whether <c>x in [0, 3, 5, 10, 12]</c> is given instead of the default.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_record_video.py#L58">tests/wrappers/vector/test_record_video.py (test_episodic_trigger)</see>.</remarks>
        [TestMethod]
        [DataRow("NextStep", false)]
        [DataRow("SameStep", false)]
        [DataRow("Disabled", false)]
        [DataRow("NextStep", true)]
        [DataRow("SameStep", true)]
        [DataRow("Disabled", true)]
        public void TestEpisodicTrigger(string autoresetMode, bool explicitTrigger) {
            string videoFolder = TemporaryFolder();
            try {
                Func<long, bool>? trigger = explicitTrigger ? x => x is 0 or 3 or 5 or 10 or 12 : null;
                using var envs = new RecordVideoVector<NDArray, NDArray>(MakeRgb(10, autoresetMode), videoFolder, episodeTrigger: trigger, videoAspectRatio: (1, 1));
                envs.Reset(seed: 123);
                envs.ActionSpace.Seed(123);
                int episodeCount = 0;
                for (int i = 0; i < 199; i++) {
                    bool[] ended = StepAndReset(envs, autoresetMode);
                    if (ended[0])
                        episodeCount++;
                }

                envs.Close();

                Assert.IsTrue(Directory.Exists(videoFolder));
                Assert.IsNotNull(envs.EpisodeTrigger);
                Assert.AreEqual(Enumerable.Range(0, episodeCount + 1).Count(i => envs.EpisodeTrigger!(i)), Videos(videoFolder).Count);
            } finally {
                Delete(videoFolder);
            }
        }

        /// <summary>A step trigger every 100 steps over 199 steps records two videos.</summary>
        /// <param name="autoresetMode">The autoreset mode's value.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_record_video.py#L94">tests/wrappers/vector/test_record_video.py (test_step_trigger)</see>.</remarks>
        [TestMethod]
        [DataRow("NextStep")]
        [DataRow("SameStep")]
        [DataRow("Disabled")]
        public void TestStepTrigger(string autoresetMode) {
            string videoFolder = TemporaryFolder();
            try {
                using var envs = new RecordVideoVector<NDArray, NDArray>(MakeRgb(10, autoresetMode), videoFolder, stepTrigger: x => x % 100 == 0, videoAspectRatio: (1, 1));
                envs.Reset(seed: 123);
                envs.ActionSpace.Seed(123);
                for (int i = 0; i < 199; i++)
                    StepAndReset(envs, autoresetMode);

                envs.Close();
                Assert.IsTrue(Directory.Exists(videoFolder));
                Assert.AreEqual(2, Videos(videoFolder).Count);
            } finally {
                Delete(videoFolder);
            }
        }

        /// <summary>A step trigger at 100 and an episode trigger at 0 and 3 record three videos over 199 steps.</summary>
        /// <param name="autoresetMode">The autoreset mode's value.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_record_video.py#L123">tests/wrappers/vector/test_record_video.py (test_both_episodic_and_step_trigger)</see>.</remarks>
        [TestMethod]
        [DataRow("NextStep")]
        [DataRow("SameStep")]
        [DataRow("Disabled")]
        public void TestBothEpisodicAndStepTrigger(string autoresetMode) {
            string videoFolder = TemporaryFolder();
            try {
                using var envs = new RecordVideoVector<NDArray, NDArray>(MakeRgb(10, autoresetMode), videoFolder, stepTrigger: x => x == 100, episodeTrigger: x => x is 0 or 3,
                    videoAspectRatio: (1, 1));
                envs.Reset(seed: 123);
                envs.ActionSpace.Seed(123);
                for (int i = 0; i < 199; i++)
                    StepAndReset(envs, autoresetMode);

                envs.Close();
                Assert.IsTrue(Directory.Exists(videoFolder));
                Assert.AreEqual(3, Videos(videoFolder).Count);
            } finally {
                Delete(videoFolder);
            }
        }

        /// <summary>A video of 10 frames from step 0 is still recording after its tenth step (or the first sub-env's episode end), stops at the next, and is the only one.</summary>
        /// <param name="autoresetMode">The autoreset mode's value.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_record_video.py#L155">tests/wrappers/vector/test_record_video.py (test_video_length)</see>.</remarks>
        [TestMethod]
        [DataRow("NextStep")]
        [DataRow("SameStep")]
        [DataRow("Disabled")]
        public void TestVideoLength(string autoresetMode) {
            const int videoLength = 10;
            string videoFolder = TemporaryFolder();
            try {
                using var envs = new RecordVideoVector<NDArray, NDArray>(MakeRgb(10, autoresetMode), videoFolder, stepTrigger: x => x == 0, videoLength: videoLength,
                    videoAspectRatio: (1, 1));
                envs.Reset(seed: 123);
                envs.ActionSpace.Seed(123);
                for (int i = 0; i < videoLength; i++) {
                    VectorStepResult<NDArray> step = envs.Step(envs.ActionSpace.Sample());
                    bool[] terminations = step.Terminations.ToArray<bool>(), truncations = step.Truncations.ToArray<bool>();
                    if (terminations[0] || truncations[0])
                        break;

                    ResetEnded(envs, autoresetMode, terminations, truncations);
                }

                // check that the environment is still recording then take a step to take the number of steps > video length
                Assert.IsTrue(envs.Recording);
                envs.Step(envs.ActionSpace.Sample());
                Assert.IsFalse(envs.Recording);
                envs.Close();

                // check that only one video is recorded
                Assert.IsTrue(Directory.Exists(videoFolder));
                Assert.AreEqual(1, Videos(videoFolder).Count);
            } finally {
                Delete(videoFolder);
            }
        }

        /// <summary>The wrapper's step returns the vector env's arrays: rewards, terminations and truncations of shape <c>(4,)</c>, which combine as arrays.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_record_video.py#L195">tests/wrappers/vector/test_record_video.py (test_record_video_step_returns_arrays_not_scalars)</see>.</remarks>
        [TestMethod]
        public void TestRecordVideoStepReturnsArraysNotScalars() {
            string videoFolder = TemporaryFolder();
            try {
                using var envs = new RecordVideoVector<NDArray, NDArray>(MakeRgb(4, null), videoFolder, videoAspectRatio: (1, 1));
                envs.Reset(seed: 42);
                envs.ActionSpace.Seed(42);
                var (_, rewards, terminations, truncations, _) = envs.Step(envs.ActionSpace.Sample());

                try {
                    Assert.IsInstanceOfType<NDArray>(rewards);
                    CollectionAssert.AreEqual(new long[] { 4 }, rewards.shape);
                    Assert.IsInstanceOfType<NDArray>(terminations);
                    CollectionAssert.AreEqual(new long[] { 4 }, terminations.shape);
                    Assert.IsInstanceOfType<NDArray>(truncations);
                    CollectionAssert.AreEqual(new long[] { 4 }, truncations.shape);
                    // The caller pattern from the original bug report — calling array methods on the returned terminations/truncations — must work.
                    using NDArray done = terminations | truncations;
                    Assert.IsInstanceOfType<NDArray>(done);
                    CollectionAssert.AreEqual(new long[] { 4 }, done.shape);
                    _ = np.any(done);
                } finally {
                    envs.Close();
                }
            } finally {
                Delete(videoFolder);
            }
        }

        /// <summary>A vector env without a render mode is refused at construction with one message telling to use an image mode such as <c>rgb_array</c>.</summary>
        /// <remarks>
        ///     Gymnasium also calls <c>__del__</c> on a wrapper whose <c>__init__</c> never ran; Gym.NET's vector wrappers have
        ///     no finalizer, so a half-built one has nothing to run.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_record_video.py#L234">tests/wrappers/vector/test_record_video.py (test_incompatible_render_mode)</see>.
        /// </remarks>
        [TestMethod]
        public void TestIncompatibleRenderMode() {
            string videoFolder = TemporaryFolder();
            try {
                using VectorEnv<NDArray, NDArray> envs = MakeVec<NDArray, NDArray>("CartPole-v1", numEnvs: 2); // render_mode is None

                var error = Assert.ThrowsExactly<ValueError>(() => new RecordVideoVector<NDArray, NDArray>(envs, videoFolder));
                StringAssert.Contains(error.Message, "such as rgb_array");

                envs.Close();
            } finally {
                Delete(videoFolder);
            }
        }

        /// <summary>
        ///     <c>make_vec("CartPole-v1", num_envs, render_mode="rgb_array", vectorization_mode=SYNC,
        ///     vector_kwargs={"autoreset_mode": mode})</c>.
        /// </summary>
        /// <param name="numEnvs">The number of sub-envs.</param>
        /// <param name="autoresetMode">The autoreset mode's value, or <see langword="null"/> for the vector env's default.</param>
        /// <returns>The vector env; the wrapper built over it owns it.</returns>
        private static VectorEnv<NDArray, NDArray> MakeRgb(int numEnvs, string? autoresetMode) =>
            MakeVec<NDArray, NDArray>("CartPole-v1", numEnvs: numEnvs, vectorizationMode: "sync",
                vectorKwargs: autoresetMode is null ? null : new Info { ["autoreset_mode"] = autoresetMode }, kwargs: new { render_mode = "rgb_array" });

        /// <summary>One sampled step; under <c>DISABLED</c>, the ended sub-envs are reset through <c>reset_mask</c>.</summary>
        /// <param name="envs">The wrapped vector env.</param>
        /// <param name="autoresetMode">The autoreset mode's value.</param>
        /// <returns>Which sub-envs ended (terminated or truncated).</returns>
        private static bool[] StepAndReset(RecordVideoVector<NDArray, NDArray> envs, string autoresetMode) {
            VectorStepResult<NDArray> step = envs.Step(envs.ActionSpace.Sample());
            bool[] terminations = step.Terminations.ToArray<bool>(), truncations = step.Truncations.ToArray<bool>();
            return ResetEnded(envs, autoresetMode, terminations, truncations);
        }

        /// <summary>Under <c>DISABLED</c>, resets the ended sub-envs through <c>reset_mask</c> (Gymnasium's loop body).</summary>
        /// <param name="envs">The wrapped vector env.</param>
        /// <param name="autoresetMode">The autoreset mode's value.</param>
        /// <param name="terminations">The step's terminations.</param>
        /// <param name="truncations">The step's truncations.</param>
        /// <returns>Which sub-envs ended.</returns>
        private static bool[] ResetEnded(RecordVideoVector<NDArray, NDArray> envs, string autoresetMode, bool[] terminations, bool[] truncations) {
            bool[] autoresets = terminations.Zip(truncations, (terminated, truncated) => terminated || truncated).ToArray();
            if (autoresetMode == "Disabled" && autoresets.Any(ended => ended)) {
                using NDArray mask = np.array(autoresets);
                envs.Reset(options: new Info { ["reset_mask"] = mask });
            }

            return autoresets;
        }

        /// <summary>The videos in a folder: the <see cref="PngVideoWriter"/> folders (Gymnasium's <c>.mp4</c> files without the extension).</summary>
        /// <param name="videoFolder">The folder.</param>
        /// <returns>Their names.</returns>
        private static List<string> Videos(string videoFolder) =>
            Directory.GetDirectories(videoFolder).Select(path => Path.GetFileName(path)!).ToList();

        /// <summary>A new folder path under the temporary directory, not created.</summary>
        /// <returns>The path.</returns>
        private static string TemporaryFolder() => Path.Combine(Path.GetTempPath(), "gymnet-vvideos-" + Guid.NewGuid().ToString("N"));

        /// <summary>Deletes a folder and its contents, when it exists.</summary>
        /// <param name="folder">The folder.</param>
        private static void Delete(string folder) {
            if (Directory.Exists(folder))
                Directory.Delete(folder, recursive: true);
        }
    }
}
