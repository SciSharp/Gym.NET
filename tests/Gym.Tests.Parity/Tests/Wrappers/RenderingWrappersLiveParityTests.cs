using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Gym.Environments;
using Gym.Envs;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Pair;
using Gym.Tests.Parity.Session;
using Gym.Utils;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace Gym.Tests.Parity.Tests.Wrappers {
    /// <summary>
    ///     The rendering wrappers (Gymnasium's <c>AddWhiteNoise</c>, <c>ObstructView</c>, <c>RecordVideo</c>,
    ///     <c>HumanRendering</c>) and <c>gymnasium.utils.save_video</c> against Gymnasium's, live.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The noise wrappers draw from the env's generator at every render: their frames are compared byte for byte after
    ///     seeded steps, and the pair's generator check after each step proves the draws match in order and number.
    ///     </para>
    ///     <para>
    ///     <c>RecordVideo</c> and <c>save_video</c> encode with moviepy, which the reference stack does not install. Their
    ///     parity is on frames (DR-25), so the cells install a stand-in <c>moviepy.video.io.ImageSequenceClip</c> that
    ///     records each video's file name, frames and frame rate instead of encoding, and compare that with what Gym.NET
    ///     hands its <see cref="IVideoWriter"/>: Gymnasium's own triggers, start and stop rules run unchanged.
    ///     </para>
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/rendering.py">gymnasium/wrappers/rendering.py</see>,
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/save_video.py">gymnasium/utils/save_video.py</see>.
    /// </remarks>
    [TestClass]
    public sealed class RenderingWrappersLiveParityTests : ParityTestBase {
        /// <summary>
        ///     The moviepy stand-in, installed once into <c>sys.modules</c>: <c>ImageSequenceClip</c> keeps copies of the frames
        ///     and <c>write_videofile</c> appends <c>(filename, frames, fps)</c> to the module's <c>written</c> list. It refuses
        ///     a clip without <c>fps</c> or <c>durations</c> with moviepy's message.
        /// </summary>
        private const string MoviepyStub = """
            import sys, types
            if "moviepy.video.io.ImageSequenceClip" not in sys.modules:
                clip_module = types.ModuleType("moviepy.video.io.ImageSequenceClip")
                clip_module.written = []

                def clip_init(self, sequence, fps=None, durations=None, **kwargs):
                    if fps is None and durations is None:
                        raise ValueError("Please provide either 'fps' or 'durations'.")
                    self.frames = [frame.copy() for frame in sequence]
                    self.fps = fps

                def clip_write(self, filename, logger=None, **kwargs):
                    clip_module.written.append((filename, self.frames, self.fps))

                clip_module.ImageSequenceClip = type("ImageSequenceClip", (), {"__init__": clip_init, "write_videofile": clip_write})
                for name in ("moviepy", "moviepy.video", "moviepy.video.io"):
                    sys.modules[name] = types.ModuleType(name)
                sys.modules["moviepy.video.io.ImageSequenceClip"] = clip_module
            sys.modules["moviepy.video.io.ImageSequenceClip"].written.clear()
            result = None
            """;

        /// <summary>
        ///     The videos the stand-in recorded: <c>(name without .mp4, frames, float(fps))</c> each, in order. Shared with the
        ///     vector cells (<see cref="VectorWrappersLiveParityTests"/>), which compare it with
        ///     <see cref="CapturingVideoWriter.Describe"/>.
        /// </summary>
        internal const string WrittenStatements = """
            import os, sys
            result = tuple((os.path.basename(path)[:-len(".mp4")], tuple(frames), float(fps))
                           for (path, frames, fps) in sys.modules["moviepy.video.io.ImageSequenceClip"].written)
            """;

        /// <summary>
        ///     <c>AddWhiteNoise(make("CartPole-v1", render_mode="rgb_array"), p)</c>: the noisy frame after a seeded reset and
        ///     after every fifth of 40 steps, RGB noise (uint8 frames) and gray noise (uint64 frames, NumPy's <c>sum</c>).
        /// </summary>
        /// <param name="grayscale">Whether the noise is gray.</param>
        [TestMethod]
        [DataRow(false)]
        [DataRow(true)]
        public void AddWhiteNoise_Frames_MatchGymnasium(bool grayscale) {
            using var pair = new GymPair(PyGymEnv.Wrap($"AddWhiteNoise(make('CartPole-v1', render_mode='rgb_array'), 0.3, {grayscale})", () => {
                using PyObject made = Python.gymnasium.make("CartPole-v1", render_mode: "rgb_array");
                return Python.gymnasium.wrappers.AddWhiteNoise(made, 0.3, grayscale);
            }), new GymNetEnv(new AddWhiteNoise<NDArray, NDArray>(WrapperParity.Make("CartPole-v1", "rgb_array"), 0.3, grayscale)));
            StepAndCompareRenders(pair);
        }

        /// <summary>
        ///     <c>ObstructView(make("CartPole-v1", render_mode="rgb_array"), ratio, width)</c>: patch centres, the patches cut by
        ///     the frame's edges, and the noise, after a seeded reset and every fifth of 40 steps.
        /// </summary>
        /// <param name="ratio">The share of pixels to cover.</param>
        /// <param name="width">The patch side.</param>
        /// <param name="grayscale">Whether the noise is gray.</param>
        [TestMethod]
        [DataRow(0.5, 100L, false)]
        [DataRow(0.2, 7L, true)]
        [DataRow(0.0, 1L, false)]
        public void ObstructView_Frames_MatchGymnasium(double ratio, long width, bool grayscale) {
            using var pair = new GymPair(PyGymEnv.Wrap($"ObstructView(make('CartPole-v1', render_mode='rgb_array'), {ratio}, {width}, {grayscale})", () => {
                using PyObject made = Python.gymnasium.make("CartPole-v1", render_mode: "rgb_array");
                return Python.gymnasium.wrappers.ObstructView(made, ratio, width, grayscale);
            }), new GymNetEnv(new ObstructView<NDArray, NDArray>(WrapperParity.Make("CartPole-v1", "rgb_array"), ratio, width, grayscale)));
            StepAndCompareRenders(pair);
        }

        /// <summary>The noise wrappers' refusals: probabilities and ratios outside [0, 1), a patch width below 1.</summary>
        [TestMethod]
        public void NoiseWrappers_Refusals_MatchGymnasium() {
            AssertRefusal("AddWhiteNoise(p=1.0)", () => {
                using PyObject made = Python.gymnasium.make("CartPole-v1", render_mode: "rgb_array");
                return Python.gymnasium.wrappers.AddWhiteNoise(made, 1.0);
            }, () => new AddWhiteNoise<NDArray, NDArray>(WrapperParity.Make("CartPole-v1", "rgb_array"), 1.0));
            AssertRefusal("ObstructView(ratio=-0.5)", () => {
                using PyObject made = Python.gymnasium.make("CartPole-v1", render_mode: "rgb_array");
                return Python.gymnasium.wrappers.ObstructView(made, -0.5, 3);
            }, () => new ObstructView<NDArray, NDArray>(WrapperParity.Make("CartPole-v1", "rgb_array"), -0.5, 3));
            AssertRefusal("ObstructView(width=0)", () => {
                using PyObject made = Python.gymnasium.make("CartPole-v1", render_mode: "rgb_array");
                return Python.gymnasium.wrappers.ObstructView(made, 0.5, 0);
            }, () => new ObstructView<NDArray, NDArray>(WrapperParity.Make("CartPole-v1", "rgb_array"), 0.5, 0));
        }

        /// <summary>
        ///     <c>RecordVideo</c> through 300 steps of seeded episodes: which videos it writes (names), their frames and frame
        ///     rate, and its counters afterwards, under the default schedule, a step trigger with a fixed length, both triggers,
        ///     and over <c>RenderCollection</c>'s list mode.
        /// </summary>
        /// <param name="variant"><c>default</c>, <c>steps</c>, <c>both</c> or <c>list</c>.</param>
        [TestMethod]
        [DataRow("default")]
        [DataRow("steps")]
        [DataRow("both")]
        [DataRow("list")]
        public void RecordVideo_Videos_MatchGymnasium(string variant) {
            PythonRuntime.Run(() => {
                using PyObject _ = Python.evalStmts(MoviepyStub, "result");
            });
            string pythonFolder = TemporaryFolder(), dotnetFolder = TemporaryFolder();
            var writer = new CapturingVideoWriter();
            try {
                (string? episodeLambda, string? stepLambda, long length) = variant switch {
                    "steps" => (null, "lambda x: x % 40 == 0", 15L),
                    "both" => ("lambda x: x in (1, 3)", "lambda x: x == 50", 0L),
                    _ => ((string?)null, (string?)null, 0L),
                };
                Func<long, bool>? episodeTrigger = variant == "both" ? x => x is 1 or 3 : null;
                Func<long, bool>? stepTrigger = variant switch {
                    "steps" => x => x % 40 == 0,
                    "both" => x => x == 50,
                    _ => null,
                };

                using var pair = new GymPair(PyGymEnv.Wrap($"RecordVideo({variant})", () => {
                    using PyObject made = Python.gymnasium.make("CartPole-v1", render_mode: "rgb_array");
                    // The wrappers keep their own references, so both locals are released after construction.
                    using PyObject? collected = variant == "list" ? Python.evalStmts("from gymnasium.wrappers import RenderCollection\nresult = RenderCollection(env)\n", "result", ("env", made)) : null;
                    using PyObject? episode = episodeLambda is null ? null : WrapperParity.Lambda(episodeLambda);
                    using PyObject? step = stepLambda is null ? null : WrapperParity.Lambda(stepLambda);
                    return Python.gymnasium.wrappers.RecordVideo(collected ?? made, pythonFolder, episode, step, length);
                }), new GymNetEnv(new RecordVideo<NDArray, NDArray>(
                    variant == "list" ? new RenderCollection<NDArray, NDArray>(WrapperParity.Make("CartPole-v1", "rgb_array")) : WrapperParity.Make("CartPole-v1", "rgb_array"),
                    dotnetFolder, episodeTrigger, stepTrigger, videoLength: length, videoWriter: writer)));
                WrapperParity.RunEpisodes(pair, 300, TestContext);
                WrapperParity.AssertAttribute(pair, "counters", "result = (env.recording, env.episode_id, env.step_id, len(env.recorded_frames))", env => {
                    var record = (RecordVideo<NDArray, NDArray>)env;
                    return new object?[] { record.Recording, record.EpisodeId, record.StepId, (long)record.RecordedFrames.Count };
                });

                // close() writes the video still recording, on both sides.
                PythonRuntime.Run(() => {
                    using PyObject _ = Python.evalStmts("env.close()\nresult = None\n", "result", ("env", pair.Python.Stepped));
                });
                ((GymNetEnv)pair.DotNet).Env.Close();

                Outcome python = Outcome.CapturePython(() => Python.evalStmts(WrittenStatements, "result"));
                // Two empty lists would agree: every variant records at least one video, so require one.
                Assert.IsTrue(writer.Describe().Length > 0, "Gym.NET wrote no video");
                ByteContract.AssertSame("videos written (name, frames, fps)", python, Outcome.Returned(ValueFacts.Of(writer.Describe())));
            } finally {
                Delete(pythonFolder);
                Delete(dotnetFolder);
                writer.Dispose();
            }
        }

        /// <summary>
        ///     <c>HumanRendering(make("CartPole-v1", render_mode=mode))</c> through 100 sampled steps over pygame's headless
        ///     display and Gym.NET's null viewer: the env's results pass through, <c>render()</c> is <c>None</c>, and the
        ///     render mode and modes read <c>human</c>.
        /// </summary>
        /// <param name="mode"><c>rgb_array</c> or <c>rgb_array_list</c>.</param>
        [TestMethod]
        [DataRow("rgb_array")]
        [DataRow("rgb_array_list")]
        public void HumanRendering_Episodes_MatchGymnasium(string mode) {
            EnvViewerFactory? previous = Gymnasium.HumanViewerFactory;
            Gymnasium.HumanViewerFactory = NullEnvViewer.Factory;
            try {
                using var pair = new GymPair(PyGymEnv.Wrap($"HumanRendering(make('CartPole-v1', render_mode='{mode}'))", () => {
                    using PyObject made = Python.gymnasium.make("CartPole-v1", render_mode: mode);
                    return Python.gymnasium.wrappers.HumanRendering(made);
                }), new GymNetEnv(new HumanRendering<NDArray, NDArray>((Env<NDArray, NDArray>)Gymnasium.Make("CartPole-v1", new { render_mode = mode }))));
                WrapperParity.RunEpisodes(pair, 100, TestContext);
                WrapperParity.AssertAttribute(pair, "render()", "result = env.render()", env => env.Render().IsNone ? null : "a frame");
                WrapperParity.AssertAttribute(pair, "render_mode", "result = env.render_mode", env => env.RenderMode);
                WrapperParity.AssertAttribute(pair, "render_modes", "result = tuple(env.metadata['render_modes'])",
                    env => ((System.Collections.IEnumerable)env.Metadata[Metadata.RenderModesKey]!).Cast<object?>().ToArray());
                PythonRuntime.Run(() => {
                    using PyObject _ = Python.evalStmts("env.close()\nresult = None\n", "result", ("env", pair.Python.Stepped));
                });
            } finally {
                Gymnasium.HumanViewerFactory = previous;
            }
        }

        /// <summary><c>HumanRendering</c>'s refusals: a <c>human</c> env, and an <c>ansi</c> one.</summary>
        [TestMethod]
        public void HumanRendering_Refusals_MatchGymnasium() {
            AssertRefusal("HumanRendering(make('CartPole-v1', render_mode='human'))", () => {
                using PyObject made = Python.gymnasium.make("CartPole-v1", render_mode: "human");
                return Python.gymnasium.wrappers.HumanRendering(made);
            }, () => new HumanRendering<NDArray, NDArray>((Env<NDArray, NDArray>)Gymnasium.Make("CartPole-v1", new { render_mode = "human" })));
            AssertRefusal("HumanRendering(make('FrozenLake-v1', render_mode='ansi'))", () => {
                using PyObject made = Python.gymnasium.make("FrozenLake-v1", render_mode: "ansi");
                return Python.gymnasium.wrappers.HumanRendering(made);
            }, () => new HumanRendering<NDArray, NDArray>((Env<NDArray, NDArray>)Gymnasium.Make("FrozenLake-v1", new { render_mode = "ansi" })));
        }

        /// <summary>
        ///     <c>save_video</c> over the same 30 random frames: the default schedule at several episode numbers, step triggers
        ///     with no length, a length, a negative length (Python's slice end) and a starting step, and an episode trigger with
        ///     a length; the clips (names, frames, frame rate) compared.
        /// </summary>
        /// <param name="variant">The trigger case.</param>
        [TestMethod]
        [DataRow("default-0")]
        [DataRow("default-2")]
        [DataRow("default-8")]
        [DataRow("steps")]
        [DataRow("steps-length")]
        [DataRow("steps-negative")]
        [DataRow("episode-length")]
        public void SaveVideo_Clips_MatchGymnasium(string variant) {
            var rng = new Random(variant.Length * 7919);
            var frames = new List<NDArray>();
            for (int i = 0; i < 30; i++) {
                var bytes = new byte[6 * 8 * 3];
                rng.NextBytes(bytes);
                frames.Add(new NDArray(bytes, new Shape(6, 8, 3)));
            }

            (long EpisodeIndex, string? EpisodeLambda, string? StepLambda, long? Length, long Start) spec = variant switch {
                "default-0" => (0L, null, null, null, 0L),
                "default-2" => (2L, null, null, null, 0L),
                "default-8" => (8L, null, null, null, 0L),
                "steps" => (0L, null, "lambda x: x % 7 == 0", null, 10L),
                "steps-length" => (0L, null, "lambda x: x % 7 == 0", 5L, 10L),
                "steps-negative" => (0L, null, "lambda x: x % 9 == 0", -3L, 0L),
                _ => (4L, "lambda x: x % 2 == 0", null, 4L, 0L),
            };
            (long episodeIndex, string? episodeLambda, string? stepLambda, long? length, long start) = spec;
            Func<long, bool>? episodeTrigger = episodeLambda is null ? null : x => x % 2 == 0;
            Func<long, bool>? stepTrigger = stepLambda switch {
                "lambda x: x % 7 == 0" => x => x % 7 == 0,
                "lambda x: x % 9 == 0" => x => x % 9 == 0,
                _ => null,
            };

            string pythonFolder = TemporaryFolder(), dotnetFolder = TemporaryFolder();
            var writer = new CapturingVideoWriter();
            try {
                PythonRuntime.Run(() => {
                    using PyObject stub = Python.evalStmts(MoviepyStub, "result");
                    using var list = new PyList();
                    foreach (NDArray frame in frames) {
                        using PyObject array = frame.ToNumpyCopy();
                        list.Append(array);
                    }

                    using PyObject? episode = episodeLambda is null ? null : WrapperParity.Lambda(episodeLambda);
                    using PyObject? step = stepLambda is null ? null : WrapperParity.Lambda(stepLambda);
                    using PyObject _ = Python.gymnasium.utils.save_video.save_video_(list, pythonFolder, episode, step, length, episodeIndex, start, 30.0);
                });
                Video.SaveVideo(frames, dotnetFolder, episodeTrigger, stepTrigger, length, episodeIndex: episodeIndex, stepStartingIndex: start, fps: 30.0, videoWriter: writer);

                Outcome python = Outcome.CapturePython(() => Python.evalStmts(WrittenStatements, "result"));
                // Episode 2 is outside the default schedule; every other case saves at least one clip, so two empty lists
                // cannot pass for agreement there.
                Assert.AreEqual(variant != "default-2", writer.Describe().Length > 0, $"clips written: {writer.Describe().Length}");
                ByteContract.AssertSame($"save_video clips ({variant})", python, Outcome.Returned(ValueFacts.Of(writer.Describe())));
            } finally {
                Delete(pythonFolder);
                Delete(dotnetFolder);
                writer.Dispose();
                foreach (NDArray frame in frames)
                    frame.Dispose();
            }
        }

        /// <summary><c>capped_cubic_video_schedule</c> for every episode up to 3000 and a negative one (Python's <c>TypeError</c>).</summary>
        [TestMethod]
        public void CappedCubicVideoSchedule_MatchesGymnasium() {
            PythonRuntime.Run(() => {
                using PyObject _ = Python.evalStmts(MoviepyStub, "result");
            });
            for (long episode = 0; episode <= 3000; episode++) {
                long id = episode;
                Assert.AreEqual(PythonRuntime.Run(() => Python.gymnasium.utils.save_video.capped_cubic_video_schedule(id)), Video.CappedCubicVideoSchedule(id), $"episode {id}");
            }

            ByteContract.AssertSame("capped_cubic_video_schedule(-1)",
                Outcome.CapturePython(() => {
                    bool picked = Python.gymnasium.utils.save_video.capped_cubic_video_schedule(-1);
                    return Oracle.PyConvert.ToPython(picked);
                }),
                Outcome.Capture(() => Video.CappedCubicVideoSchedule(-1)));
        }

        /// <summary>Resets the pair with seed 0, then takes 40 alternating steps (resetting at episode ends), comparing the renders after the reset and every fifth step.</summary>
        /// <param name="pair">The pair.</param>
        /// <exception cref="ParityException">Thrown at the first difference.</exception>
        private void StepAndCompareRenders(GymPair pair) {
            pair.Reset(0);
            CompareRender(pair, "after reset");
            long seed = 1;
            for (int i = 1; i <= 40; i++) {
                using (ParityAction action = ParityAction.Discrete(i % 2))
                    pair.Step(action);
                if (pair.Ended)
                    pair.Reset(seed++);
                if (i % 5 == 0)
                    CompareRender(pair, $"after step {i}");
            }

            TestContext.WriteLine($"{pair.Case}: 40 steps, renders compared every fifth.");
        }

        /// <summary>Compares <c>env.render()</c> on both sides, as arrays (dtype, shape, bytes).</summary>
        /// <param name="pair">The pair.</param>
        /// <param name="label">Where in the run.</param>
        /// <exception cref="ParityException">Thrown when the renders differ.</exception>
        private static void CompareRender(GymPair pair, string label) =>
            WrapperParity.AssertAttribute(pair, $"render() {label}", "result = env.render()", env => env.Render() is RenderResult.Rgb rgb ? rgb.Frame : null);

        /// <summary>Compares a wrapper's refusal on both sides: the same exception type and message.</summary>
        /// <param name="label">The comparison's name in reports.</param>
        /// <param name="python">Builds Gymnasium's wrapper (under the GIL).</param>
        /// <param name="dotnet">Builds Gym.NET's wrapper.</param>
        /// <exception cref="ParityException">Thrown when the outcomes differ.</exception>
        private static void AssertRefusal(string label, Func<PyObject> python, Func<object> dotnet) =>
            ByteContract.AssertSame(label, Outcome.CapturePython(python), Outcome.Capture(dotnet));

        /// <summary>A new folder path under the temporary directory, not created.</summary>
        /// <returns>The path.</returns>
        private static string TemporaryFolder() => Path.Combine(Path.GetTempPath(), "gymnet-parity-videos-" + Guid.NewGuid().ToString("N"));

        /// <summary>Deletes a folder and its contents, when it exists.</summary>
        /// <param name="folder">The folder.</param>
        private static void Delete(string folder) {
            if (Directory.Exists(folder))
                Directory.Delete(folder, recursive: true);
        }

        /// <summary>
        ///     Installs the moviepy stand-in (<see cref="MoviepyStub"/>) and clears what it recorded, so a cell reads only its
        ///     own videos afterwards; the vector cells share it.
        /// </summary>
        /// <exception cref="PythonInteropException">Thrown when the stand-in cannot be installed.</exception>
        internal static void InstallMoviepyStub() =>
            PythonRuntime.Run(() => {
                using PyObject _ = Python.evalStmts(MoviepyStub, "result");
            });

        /// <summary>
        ///     An <see cref="IVideoWriter"/> that keeps each video's name, frame copies and frame rate instead of encoding; the
        ///     vector cells share it. Dispose it to release the copies.
        /// </summary>
        internal sealed class CapturingVideoWriter : IVideoWriter, IDisposable {
            /// <summary>The videos written, in order.</summary>
            private readonly List<(string Name, NDArray[] Frames, double Fps)> _written = new();

            /// <summary><c>.mp4</c>, so the names read as Gymnasium's files without the extension.</summary>
            public string Extension => ".mp4";

            /// <summary>Keeps the video's name, copies of its frames and its frame rate.</summary>
            /// <param name="path">The video's path without extension.</param>
            /// <param name="frames">The frames; copied.</param>
            /// <param name="fps">The frame rate.</param>
            public void Write(string path, IReadOnlyList<NDArray> frames, double fps) =>
                _written.Add((Path.GetFileName(path), frames.Select(frame => frame.copy()).ToArray(), fps));

            /// <summary>The videos as the stand-in's description: a tuple of <c>(name, frames tuple, fps)</c>.</summary>
            /// <returns>The description; its arrays stay this writer's.</returns>
            public object?[] Describe() =>
                _written.Select(video => (object?)new object?[] { video.Name, video.Frames.Cast<object?>().ToArray(), video.Fps }).ToArray();

            /// <summary>Releases the frame copies.</summary>
            public void Dispose() {
                foreach (NDArray frame in _written.SelectMany(video => video.Frames))
                    frame.Dispose();
                _written.Clear();
            }
        }
    }
}
