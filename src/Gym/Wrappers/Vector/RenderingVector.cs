#nullable enable
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Gym.Environments;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Utils;
using Gym.Vector;
using NumSharp;

namespace Gym.Wrappers.Vector {
    /// <summary>
    ///     Shows a vector env's sub-envs in one window, tiled, as Gymnasium's vector <c>HumanRendering</c> does: every reset
    ///     and step renders the sub-envs, scales each frame to fit the grid (OpenCV's bilinear <c>cv2.resize</c>), and shows
    ///     the mosaic at <c>metadata["render_fps"]</c>.
    /// </summary>
    /// <typeparam name="TObs">The batched observation type, passed through.</typeparam>
    /// <typeparam name="TAct">The batched action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     The window is a viewer from <c>Gymnasium.HumanViewerFactory</c> (DR-20), opened at the screen size (by default a
    ///     sub-env's frame size), with the grid's rows and columns chosen once, as Gymnasium chooses them. Frames are paced by
    ///     <see cref="Gym.Rendering.Clock"/>. <see cref="LastMosaic"/> keeps the last mosaic shown, <c>(width, height, 3)</c>
    ///     as pygame's surfaces are laid out.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/vector/rendering.py#L21">gymnasium/wrappers/vector/rendering.py</see>.</para>
    /// </remarks>
    [PyName("HumanRendering")]
    public class HumanRenderingVector<TObs, TAct> : VectorWrapper<TObs, TAct>, IRecordConstructorArgs {
        /// <summary>The render modes the wrapper can show (<c>ACCEPTED_RENDER_MODES</c>).</summary>
        public static readonly IReadOnlyList<string> AcceptedRenderModes = new[] { "rgb_array", "rgb_array_list", "depth_array", "depth_array_list" };

        /// <summary>The window, opened at the first frame.</summary>
        private IEnvViewer? _window;

        /// <summary>The frame pacer, created at the first frame.</summary>
        private Gym.Rendering.Clock? _clock;

        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The vector env; its render mode must be one of <see cref="AcceptedRenderModes"/>, and its metadata must have <c>render_fps</c>.</param>
        /// <param name="screenSize">The window's <c>(width, height)</c>; <see langword="null"/> for a sub-env's frame size.</param>
        /// <exception cref="ValueError">Thrown for another render mode, or metadata without <c>render_fps</c>, with Gymnasium's messages.</exception>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        public HumanRenderingVector(VectorEnv<TObs, TAct> env, (int Width, int Height)? screenSize = null) : base(env) {
            SavedKwargs = RecordConstructorArgs.Record(new Info { ["screen_size"] = screenSize });
            ScreenSize = screenSize;
            if (env.RenderMode is null || !AcceptedRenderModes.Contains(env.RenderMode))
                throw new ValueError($"Expected env.render_mode to be one of {PyBuiltins.Repr(AcceptedRenderModes.ToList())} but got '{PyBuiltins.Str(env.RenderMode)}'");
            if (!env.Metadata.ContainsKey(Metadata.RenderFpsKey))
                throw new ValueError("The base environment must specify 'render_fps' to be used with the HumanRendering wrapper");

            object? modes = Metadata[Metadata.RenderModesKey];
            if (!(modes is IEnumerable sequence and not string && sequence.Cast<object?>().Any(mode => mode is "human"))) {
                var metadata = (Metadata)env.Metadata.DeepCopy();
                object? copied = metadata[Metadata.RenderModesKey];
                metadata[Metadata.RenderModesKey] = (copied is IEnumerable items and not string ? items.Cast<object?>() : Enumerable.Empty<object?>())
                    .Select(item => item?.ToString() ?? "None").Append("human").ToList();
                WrapperMetadata = metadata;
            }
        }

        /// <summary>The recorded constructor arguments (Python's <c>_saved_kwargs</c>).</summary>
        [PyName("_saved_kwargs")]
        public Info SavedKwargs { get; }

        /// <summary>The window's <c>(width, height)</c> (<c>screen_size</c>), a sub-env frame's size once the first frame came when none was given.</summary>
        public (int Width, int Height)? ScreenSize { get; private set; }

        /// <summary>The size each sub-env's frame is scaled to, <c>(width, height)</c> (<c>scaled_subenv_size</c>), or <see langword="null"/> before the first frame.</summary>
        public (int Width, int Height)? ScaledSubenvSize { get; private set; }

        /// <summary>The grid's rows (<c>num_rows</c>), or <see langword="null"/> before the first frame.</summary>
        public int? NumRows { get; private set; }

        /// <summary>The grid's columns (<c>num_cols</c>), or <see langword="null"/> before the first frame.</summary>
        public int? NumCols { get; private set; }

        /// <summary>The last mosaic shown, <c>(width, height, 3)</c> uint8 (pygame's surface layout); owned by the wrapper.</summary>
        public NDArray? LastMosaic { get; private set; }

        /// <summary>Always <c>human</c>.</summary>
        /// <exception cref="AttributeError">Thrown when assigned, as for every wrapper.</exception>
        public override string? RenderMode {
            get => "human";
            protected set => base.RenderMode = value;
        }

        /// <summary>Steps the env, then shows its sub-envs' frames.</summary>
        /// <param name="actions">The actions, passed on.</param>
        /// <returns>The env's transition.</returns>
        /// <exception cref="DependencyNotInstalled">Thrown at the first frame when no viewer factory is set.</exception>
        public override VectorStepResult<TObs> Step(TAct actions) {
            VectorStepResult<TObs> result = base.Step(actions);
            RenderFrame();
            return result;
        }

        /// <summary>Resets the env, then shows its sub-envs' frames.</summary>
        /// <param name="seed">The seed, passed on.</param>
        /// <param name="options">The options, passed on.</param>
        /// <returns>The env's reset.</returns>
        /// <exception cref="DependencyNotInstalled">Thrown at the first frame when no viewer factory is set.</exception>
        public override VectorResetResult<TObs> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            VectorResetResult<TObs> result = base.Reset(seed, options);
            RenderFrame();
            return result;
        }

        /// <summary>Resets the env with one seed per sub-env, then shows its sub-envs' frames.</summary>
        /// <param name="seeds">The seeds, passed on.</param>
        /// <param name="options">The options, passed on.</param>
        /// <returns>The env's reset.</returns>
        /// <exception cref="DependencyNotInstalled">Thrown at the first frame when no viewer factory is set.</exception>
        public override VectorResetResult<TObs> Reset(IReadOnlyList<BigInteger?> seeds, IReadOnlyDictionary<string, object?>? options = null) {
            VectorResetResult<TObs> result = base.Reset(seeds, options);
            RenderFrame();
            return result;
        }

        /// <summary>Closes the window, then the env.</summary>
        public override void Close() {
            if (_window is not null) {
                _window.CloseEnvironment();
                _window = null;
            }

            base.Close();
        }

        /// <summary>Releases the last mosaic, then disposes as every vector wrapper does.</summary>
        /// <param name="disposing"><see langword="true"/> from <see cref="IDisposable.Dispose"/>.</param>
        protected override void Dispose(bool disposing) {
            base.Dispose(disposing);
            if (disposing)
                LastMosaic?.Dispose();
        }

        /// <summary>
        ///     Shows the sub-envs' frames (<c>_render_frame</c>): stacked and transposed to <c>(n, width, height, 3)</c>, the grid
        ///     chosen once for the screen's aspect, each frame scaled by <c>cv2.resize</c> into its cell, the mosaic paced and shown.
        /// </summary>
        /// <exception cref="AssertionError">Thrown when the env renders nothing, fewer frames than sub-envs, a frame that is no array, or a grid that does not fit, as Gymnasium asserts.</exception>
        /// <exception cref="DependencyNotInstalled">Thrown when the window must open and no viewer factory is set.</exception>
        private void RenderFrame() {
            IReadOnlyList<RenderResult>? renders = Env.Render();
            if (renders is null)
                throw new AssertionError("");
            if (renders.Count != NumEnvs)
                throw new AssertionError("");
            // The frames stay the renders', which the finally below releases; this only picks them out.
#pragma warning disable NDW012
            NDArray[] frames = renders.Select(render => render switch {
                RenderResult.Rgb rgb => rgb.Frame,
                RenderResult.Depth depth => depth.Frame,
                _ => throw new AssertionError($"Expected `env.render()` to return a numpy array, actually returned {PyBuiltins.Repr(renders.Select(r => r.ToString()).ToList())}"),
            }).ToArray();
#pragma warning restore NDW012
            try {
                // np.array(subenv_renders, dtype=np.uint8), then np.transpose(axes=(0, 2, 1, 3)): (n, width, height, channels).
                using NDArray stacked = np.stack(frames);
                using NDArray bytes = stacked.astype(np.uint8);
                using NDArray transposed = np.transpose(bytes, new[] { 0, 2, 1, 3 });
                using NDArray subenvRenders = transposed.copy();
                (int Width, int Height) subenvSize = ((int)subenvRenders.shape[1], (int)subenvRenders.shape[2]);
                ScreenSize ??= subenvSize;

                if (ScaledSubenvSize is null)
                    ChooseGrid(subenvSize);
                (int scaledWidth, int scaledHeight) = ScaledSubenvSize!.Value;

                var merged = np.zeros(new Shape(ScreenSize.Value.Width, ScreenSize.Value.Height, 3), np.uint8);
                for (int i = 0; i < NumEnvs; i++) {
                    int col = i % NumCols!.Value, row = i / NumCols.Value;
                    using NDArray subenv = subenvRenders[i];
                    // cv2.resize(subenv_renders[i], scaled_subenv_size[::-1]): the transposed frame's rows are its width.
                    using NDArray scaled = Cv2.Resize(subenv, (scaledHeight, scaledWidth));
                    int x = col * scaledWidth, y = row * scaledHeight;
                    merged[$"{x}:{x + scaledWidth}, {y}:{y + scaledHeight}"] = scaled;
                }

                LastMosaic?.Dispose();
                LastMosaic = merged;
                if (_window is null) {
                    IEnvironmentViewerFactoryDelegate factory = global::Gym.Gymnasium.ResolveHumanViewerFactory(null);
                    // TODO(async viewer path, DR-53): blocks on the factory; see IEnvironmentViewerFactoryDelegate.
                    _window = factory(ScreenSize.Value.Width, ScreenSize.Value.Height, "HumanRendering").GetAwaiter().GetResult();
                }

                _clock ??= new Gym.Rendering.Clock();
                _clock.Tick(PyBuiltins.Float(Metadata[Metadata.RenderFpsKey]));
                // The viewer takes (height, width, 3) pixels: pygame's make_surface reads the (width, height) array transposed.
                using NDArray image = np.transpose(merged, new[] { 1, 0, 2 });
                using NDArray contiguous = image.copy();
                using SixLabors.ImageSharp.Image frame = VideoFrames.ToImage(contiguous);
                _window.Render(frame);
            } finally {
                foreach (RenderResult render in renders)
                    render.Dispose();
            }
        }

        /// <summary>Chooses the grid and the scaled frame size for the screen, as Gymnasium does on the first frame.</summary>
        /// <param name="subenvSize">A sub-env frame's <c>(width, height)</c>.</param>
        /// <exception cref="AssertionError">Thrown when the grid does not fit the screen, as Gymnasium asserts.</exception>
        private void ChooseGrid((int Width, int Height) subenvSize) {
            (int screenWidth, int screenHeight) = ScreenSize!.Value;
            double widthRatio = subenvSize.Width / (double)screenWidth;
            double heightRatio = subenvSize.Height / (double)screenHeight;
            int numRows = 1, numCols = 1;
            while (numRows * numCols < NumEnvs) {
                double rowRatio = numRows * heightRatio, colRatio = numCols * widthRatio;
                if (rowRatio == colRatio) {
                    numRows += 1;
                    numCols += 1;
                } else if (rowRatio > colRatio) {
                    numCols += 1;
                } else {
                    numRows += 1;
                }
            }

            double scalingFactor = Math.Min(screenWidth / (double)(numCols * subenvSize.Width), screenHeight / (double)(numRows * subenvSize.Height));
            if (!(numCols * subenvSize.Width * scalingFactor == screenWidth || numRows * subenvSize.Height * scalingFactor == screenHeight))
                throw new AssertionError("");
            NumRows = numRows;
            NumCols = numCols;
            // int(subenv_size * scaling_factor): Python's int() truncates.
            ScaledSubenvSize = ((int)(subenvSize.Width * scalingFactor), (int)(subenvSize.Height * scalingFactor));
            if (!(numRows * numCols >= NumEnvs && ScaledSubenvSize.Value.Width * numCols <= screenWidth && ScaledSubenvSize.Value.Height * numRows <= screenHeight))
                throw new AssertionError("");
        }
    }

    /// <summary>
    ///     Records videos of a vector env's episodes, the sub-envs' frames tiled into one frame, as Gymnasium's vector
    ///     <c>RecordVideo</c> does: episodes are counted on the first sub-env, the grid's shape is the one closest to
    ///     <c>video_aspect_ratio</c>, and each video goes through an <see cref="IVideoWriter"/> (DR-25).
    /// </summary>
    /// <typeparam name="TObs">The batched observation type, passed through.</typeparam>
    /// <typeparam name="TAct">The batched action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     With <c>record_first_only</c> only the first sub-env is recorded. With <c>SAME_STEP</c> autoreset an episode's last
    ///     frame is not recorded (Gymnasium's warning). The triggers, names, lengths and frame rate are Gymnasium's; see the
    ///     single-env <see cref="RecordVideo{TObs, TAct}"/> for the writer, <c>gc_trigger</c> and closing. Gymnasium records
    ///     only some of its arguments for a spec; so does this wrapper, plus <c>video_writer</c>.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/vector/rendering.py#L233">gymnasium/wrappers/vector/rendering.py</see>.</para>
    /// </remarks>
    [PyName("RecordVideo")]
    public class RecordVideoVector<TObs, TAct> : VectorWrapper<TObs, TAct>, IRecordConstructorArgs {
        /// <summary>Wraps <paramref name="env"/>, creating <paramref name="videoFolder"/>.</summary>
        /// <param name="env">The vector env; it must render images.</param>
        /// <param name="videoFolder">The folder of the videos, created if needed (a warning when it exists).</param>
        /// <param name="videoAspectRatio">The mosaic's target <c>(width, height)</c> aspect; <see langword="null"/> is <c>(1, 1)</c>.</param>
        /// <param name="recordFirstOnly">Whether only the first sub-env is recorded.</param>
        /// <param name="episodeTrigger">Whether to start recording at an episode; with no trigger at all, <see cref="Video.CappedCubicVideoSchedule"/>.</param>
        /// <param name="stepTrigger">Whether to start recording at a step.</param>
        /// <param name="videoLength">The frames per video; 0 records whole episodes.</param>
        /// <param name="namePrefix">The prefix of the video names.</param>
        /// <param name="fps">The frame rate; <see langword="null"/> takes <c>metadata["render_fps"]</c>, else 30.</param>
        /// <param name="disableLogger">moviepy's logger switch, kept for Gymnasium's signature.</param>
        /// <param name="gcTrigger">Whether to collect garbage after a video; <see langword="null"/> always does.</param>
        /// <param name="videoWriter">The encoder; <see langword="null"/> for <see cref="PngVideoWriter"/>.</param>
        /// <exception cref="ValueError">Thrown when the env renders no images, with Gymnasium's message.</exception>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="videoFolder"/> is <see langword="null"/>.</exception>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        public RecordVideoVector(VectorEnv<TObs, TAct> env, string videoFolder, (int Width, int Height)? videoAspectRatio = null, bool recordFirstOnly = false,
            Func<long, bool>? episodeTrigger = null, Func<long, bool>? stepTrigger = null, long videoLength = 0, string namePrefix = "rl-video",
            double? fps = null, bool disableLogger = true, Func<long, bool>? gcTrigger = null, IVideoWriter? videoWriter = null) : base(env) {
            SavedKwargs = RecordConstructorArgs.Record(new Info {
                ["video_folder"] = videoFolder, ["episode_trigger"] = episodeTrigger, ["step_trigger"] = stepTrigger, ["video_length"] = videoLength,
                ["name_prefix"] = namePrefix, ["disable_logger"] = disableLogger, ["video_writer"] = videoWriter,
            });
            if (videoFolder is null)
                throw new ArgumentNullException(nameof(videoFolder));
            if (env.RenderMode is null or "human" or "ansi")
                throw new ValueError($"Render mode is {PyBuiltins.Str(env.RenderMode)}, which is incompatible with RecordVideo. " +
                                     "Initialize your environment with a render_mode that returns an image, such as rgb_array.");

            if (episodeTrigger is null && stepTrigger is null)
                episodeTrigger = Video.CappedCubicVideoSchedule;

            if (env.Metadata.TryGetValue("autoreset_mode", out object? mode) && mode is not null) {
                AutoresetMode = mode as AutoresetMode ?? AutoresetMode.NextStep;
            } else {
                Logger.Warn($"{env} metadata doesn't specify its autoreset mode ({PyBuiltins.Repr(env.Metadata)}), therefore, defaulting to next step.");
                AutoresetMode = AutoresetMode.NextStep;
            }

            if (ReferenceEquals(AutoresetMode, AutoresetMode.SameStep))
                Logger.Warn("Vector environment's autoreset mode is same-step (https://farama.org/Vector-Autoreset-Mode). Recorded episodes will not contain the last frame of the episode.");

            EpisodeTrigger = episodeTrigger;
            StepTrigger = stepTrigger;
            DisableLogger = disableLogger;
            GcTrigger = gcTrigger ?? (_ => true);
            RecordFirstOnly = recordFirstOnly;
            VideoAspectRatio = videoAspectRatio ?? (1, 1);
            VideoWriter = videoWriter ?? new PngVideoWriter();

            VideoFolder = System.IO.Path.GetFullPath(videoFolder);
            if (System.IO.Directory.Exists(VideoFolder))
                Logger.Warn($"Overwriting existing videos at {VideoFolder} folder (try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)");
            System.IO.Directory.CreateDirectory(VideoFolder);

            FramesPerSec = fps ?? (Metadata.TryGetValue(Metadata.RenderFpsKey, out object? renderFps) && renderFps is not null ? PyBuiltins.Float(renderFps) : 30);
            NamePrefix = namePrefix;
            VideoLength = videoLength != 0 ? videoLength : double.PositiveInfinity;
        }

        /// <summary>The recorded constructor arguments (Python's <c>_saved_kwargs</c>).</summary>
        [PyName("_saved_kwargs")]
        public Info SavedKwargs { get; }

        /// <summary>The env's autoreset mode (<c>autoreset_mode</c>).</summary>
        public AutoresetMode AutoresetMode { get; }

        /// <summary>Whether the first sub-env ended at the last step, so the next step starts its episode (<c>has_autoreset</c>).</summary>
        public bool HasAutoreset { get; private set; }

        /// <summary>When to start recording at an episode (<c>episode_trigger</c>).</summary>
        public Func<long, bool>? EpisodeTrigger { get; set; }

        /// <summary>When to start recording at a step (<c>step_trigger</c>).</summary>
        public Func<long, bool>? StepTrigger { get; set; }

        /// <summary>moviepy's logger switch (<c>disable_logger</c>).</summary>
        public bool DisableLogger { get; set; }

        /// <summary>When to collect garbage after a video (<c>gc_trigger</c>).</summary>
        public Func<long, bool>? GcTrigger { get; set; }

        /// <summary>Whether only the first sub-env is recorded (<c>record_first_only</c>).</summary>
        public bool RecordFirstOnly { get; set; }

        /// <summary>The mosaic's target aspect, <c>(width, height)</c> (<c>video_aspect_ratio</c>).</summary>
        public (int Width, int Height) VideoAspectRatio { get; set; }

        /// <summary>The mosaic's columns, -1 before the first frame (<c>frame_cols</c>).</summary>
        public int FrameCols { get; private set; } = -1;

        /// <summary>The mosaic's rows, -1 before the first frame (<c>frame_rows</c>).</summary>
        public int FrameRows { get; private set; } = -1;

        /// <summary>The encoder the videos go through.</summary>
        public IVideoWriter VideoWriter { get; set; }

        /// <summary>The absolute folder of the videos (<c>video_folder</c>).</summary>
        public string VideoFolder { get; }

        /// <summary>The videos' frame rate (<c>frames_per_sec</c>).</summary>
        public double FramesPerSec { get; set; }

        /// <summary>The prefix of the video names (<c>name_prefix</c>).</summary>
        public string NamePrefix { get; set; }

        /// <summary>The frames per video, infinite for whole episodes (<c>video_length</c>).</summary>
        public double VideoLength { get; set; }

        /// <summary>Whether a video is being recorded (<c>recording</c>).</summary>
        public bool Recording { get; private set; }

        /// <summary>The mosaics of the video being recorded (<c>recorded_frames</c>), owned by the wrapper.</summary>
        public List<NDArray> RecordedFrames { get; private set; } = new();

        /// <summary>The current video's name without extension (<c>_video_name</c>), or <see langword="null"/>.</summary>
        [PyName("_video_name")]
        public string? VideoName { get; private set; }

        /// <summary>The number of the last step, from 0 (<c>step_id</c>; -1 before the first).</summary>
        public long StepId { get; private set; } = -1;

        /// <summary>The number of the first sub-env's current episode, from 0 (<c>episode_id</c>; -1 before the first reset).</summary>
        public long EpisodeId { get; private set; } = -1;

        /// <summary>Resets the env; a full reset (or one masking the first sub-env) counts an episode and may start a recording.</summary>
        /// <param name="seed">The seed, passed on.</param>
        /// <param name="options">The options, passed on.</param>
        /// <returns>The env's reset.</returns>
        public override VectorResetResult<TObs> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            BeforeReset(options);
            VectorResetResult<TObs> result = base.Reset(seed, options);
            AfterReset();
            return result;
        }

        /// <summary>Resets the env with one seed per sub-env, as <see cref="Reset(BigInteger?, IReadOnlyDictionary{string, object})"/> does.</summary>
        /// <param name="seeds">The seeds, passed on.</param>
        /// <param name="options">The options, passed on.</param>
        /// <returns>The env's reset.</returns>
        public override VectorResetResult<TObs> Reset(IReadOnlyList<BigInteger?> seeds, IReadOnlyDictionary<string, object?>? options = null) {
            BeforeReset(options);
            VectorResetResult<TObs> result = base.Reset(seeds, options);
            AfterReset();
            return result;
        }

        /// <summary>Steps the env, counts the first sub-env's episodes by the autoreset mode, starts recordings and captures a frame.</summary>
        /// <param name="actions">The actions, passed on.</param>
        /// <returns>The env's transition.</returns>
        public override VectorStepResult<TObs> Step(TAct actions) {
            VectorStepResult<TObs> result = Env.Step(actions);
            StepId += 1;

            bool firstEnded = result.Terminations.GetBoolean(0) || result.Truncations.GetBoolean(0);
            if (ReferenceEquals(AutoresetMode, AutoresetMode.NextStep)) {
                if (HasAutoreset)
                    StartEpisode();
                HasAutoreset = firstEnded;
            } else if (ReferenceEquals(AutoresetMode, AutoresetMode.SameStep) && firstEnded) {
                StartEpisode();
            }

            if (StepTrigger is not null && StepTrigger(StepId))
                StartRecording($"{NamePrefix}-step-{StepId}");
            if (Recording) {
                CaptureFrame();
                if (RecordedFrames.Count > VideoLength)
                    StopRecording();
            }

            return result;
        }

        /// <summary>Closes the env, then writes the video still being recorded.</summary>
        public override void Close() {
            base.Close();
            if (Recording)
                StopRecording();
        }

        /// <summary>Starts a video (<c>start_recording</c>), first writing the one being recorded.</summary>
        /// <param name="videoName">The new video's name without extension.</param>
        public void StartRecording(string videoName) {
            if (Recording)
                StopRecording();
            Recording = true;
            VideoName = videoName;
        }

        /// <summary>Writes the video being recorded and stops (<c>stop_recording</c>); a video without frames is skipped with a warning.</summary>
        /// <exception cref="AssertionError">Thrown when nothing is being recorded: <c>stop_recording was called, but no recording was started</c>.</exception>
        public void StopRecording() {
            if (!Recording)
                throw new AssertionError("stop_recording was called, but no recording was started");
            if (RecordedFrames.Count == 0)
                Logger.Warn("Ignored saving a video as there were zero frames to save.");
            else
                VideoWriter.Write(System.IO.Path.Combine(VideoFolder, VideoName!), RecordedFrames, FramesPerSec);

            foreach (NDArray frame in RecordedFrames)
                frame.Dispose();
            RecordedFrames = new List<NDArray>();
            Recording = false;
            VideoName = null;
            if (GcTrigger is not null && GcTrigger(EpisodeId))
                GC.Collect();
        }

        /// <summary>Releases the recorded mosaics, then closes (writing a video still recording) and disposes as every vector wrapper does.</summary>
        /// <param name="disposing"><see langword="true"/> from <see cref="IDisposable.Dispose"/>.</param>
        protected override void Dispose(bool disposing) {
            base.Dispose(disposing);
            if (!disposing)
                return;
            foreach (NDArray frame in RecordedFrames)
                frame.Dispose();
            RecordedFrames = new List<NDArray>();
        }

        /// <summary>A full reset, or one whose mask includes the first sub-env, starts an episode (before the env resets, as Gymnasium orders it).</summary>
        /// <param name="options">The reset options.</param>
        private void BeforeReset(IReadOnlyDictionary<string, object?>? options) {
            bool firstResets = options is null || !options.TryGetValue("reset_mask", out object? mask) || mask is not NDArray array || array.GetBoolean(0);
            if (firstResets)
                StartEpisode();
        }

        /// <summary>After a reset: captures the first frame while recording, and clears the pending autoreset.</summary>
        private void AfterReset() {
            if (Recording) {
                CaptureFrame();
                if (RecordedFrames.Count > VideoLength)
                    StopRecording();
            }

            HasAutoreset = false;
        }

        /// <summary>Counts an episode of the first sub-env: ends a whole-episode recording and starts one when the episode trigger fires.</summary>
        private void StartEpisode() {
            EpisodeId += 1;
            if (Recording && double.IsPositiveInfinity(VideoLength))
                StopRecording();
            if (EpisodeTrigger is not null && EpisodeTrigger(EpisodeId))
                StartRecording($"{NamePrefix}-episode-{EpisodeId}");
        }

        /// <summary>Renders the sub-envs and records their mosaic (<c>_capture_frame</c>).</summary>
        /// <exception cref="AssertionError">Thrown when nothing is being recorded, or the env renders fewer frames than sub-envs.</exception>
        private void CaptureFrame() {
            if (!Recording)
                throw new AssertionError("Cannot capture a frame, recording wasn't started.");
            IReadOnlyList<RenderResult>? renders = Env.Render();
            if (renders is null)
                throw new AssertionError(PyBuiltins.TypeRepr(null));
            if (renders.Count != NumEnvs)
                throw new AssertionError("");
            try {
                NDArray[] frames = renders.Select(render => render is RenderResult.Rgb rgb ? rgb.Frame : throw new AssertionError("")).ToArray();
                if (RecordFirstOnly)
                    frames = new[] { frames[0] };
                if (FrameCols == -1 || FrameRows == -1)
                    ChooseGrid(frames.Length, (int)frames[0].shape[0], (int)frames[0].shape[1]);
                RecordedFrames.Add(ConcatFrames(frames));
            } finally {
                foreach (RenderResult render in renders)
                    render.Dispose();
            }
        }

        /// <summary>The grid closest to the target aspect among the divisors of the frame count (<c>_get_concat_frame_shape</c>).</summary>
        /// <param name="nFrames">The frames per mosaic.</param>
        /// <param name="h">A frame's height.</param>
        /// <param name="w">A frame's width.</param>
        private void ChooseGrid(int nFrames, int h, int w) {
            double target = VideoAspectRatio.Width / (double)VideoAspectRatio.Height;
            int bestRows = 1, bestCols = nFrames;
            double minDiff = double.PositiveInfinity;
            // range(1, int(n_frames ** 0.5) + 1)
            int limit = (int)Math.Pow(nFrames, 0.5);
            for (int rows = 1; rows <= limit; rows++) {
                if (nFrames % rows != 0)
                    continue;
                int cols = nFrames / rows;
                double aspect = cols * w / (double)(rows * h);
                double diff = Math.Abs(aspect - target);
                if (diff < minDiff) {
                    minDiff = diff;
                    bestRows = rows;
                    bestCols = cols;
                }
            }

            FrameRows = bestRows;
            FrameCols = bestCols;
        }

        /// <summary>The frames tiled row by row into one image (<c>_concat_frames</c>), the cells beyond the frames black.</summary>
        /// <param name="frames">The frames, all of one shape.</param>
        /// <returns>A new mosaic the wrapper owns.</returns>
        private NDArray ConcatFrames(NDArray[] frames) {
            using NDArray stacked = np.stack(frames);
            long h = stacked.shape[1], w = stacked.shape[2], c = stacked.shape[3];
            NDArray grid = np.zeros(new Shape(FrameRows * h, FrameCols * w, c), stacked.dtype);
            for (int idx = 0; idx < frames.Length; idx++) {
                long r = idx / FrameCols, col = idx % FrameCols;
                using NDArray frame = stacked[idx];
                grid[$"{r * h}:{(r + 1) * h}, {col * w}:{(col + 1) * w}"] = frame;
            }

            return grid;
        }
    }
}
