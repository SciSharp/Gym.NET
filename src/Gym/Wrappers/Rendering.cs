#nullable enable
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Gym.Environments;
using Gym.Envs;
using Gym.Observations;
using Gym.Utils;
using NumSharp;

namespace Gym.Wrappers {
    /// <summary>
    ///     Collects a frame after every reset and step, so <c>render()</c> returns the list of frames since the last call
    ///     (<see cref="RenderResult.Frames"/>) instead of the current one: the <c>*_list</c> render modes. <c>make</c> adds it
    ///     when <c>render_mode="rgb_array_list"</c> is asked of an env that renders <c>rgb_array</c> itself.
    /// </summary>
    /// <typeparam name="TObs">The observation type, passed through.</typeparam>
    /// <typeparam name="TAct">The action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     Every reset and step renders the env once more, which costs a frame's drawing each. The render mode reads as the
    ///     env's with <c>_list</c> appended, and the metadata is the env's (a deep copy) with that mode added.
    ///     </para>
    ///     <para>
    ///     Ownership (D19): with <see cref="PopFrames"/> (the default) <see cref="Render"/> hands the collected frames over and
    ///     the caller owns them; without it the frames stay the wrapper's, returned again by every render, and the caller must
    ///     not dispose them. Frames still held are released when the wrapper is disposed.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/rendering.py#L34">gymnasium/wrappers/rendering.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class RenderCollection<TObs, TAct> : Wrapper<TObs, TAct>, IRecordConstructorArgs {
        /// <summary>Wraps <paramref name="env"/>, whose render mode must be a single-frame one.</summary>
        /// <param name="env">The env to collect frames of.</param>
        /// <param name="popFrames">Clear the collection whenever <see cref="Render"/> returns it.</param>
        /// <param name="resetClean">Clear the collection on every reset.</param>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">
        ///     Thrown when the env has no render mode (<c>Expected env.render_mode to be not None</c>) or already collects
        ///     (<c>Expected env.render_mode to not end with '_list', got 'rgb_array_list'</c>).
        /// </exception>
        public RenderCollection(Env<TObs, TAct> env, bool popFrames = true, bool resetClean = true) : base(env) {
            SavedKwargs = RecordConstructorArgs.Record(new Info { ["pop_frames"] = popFrames, ["reset_clean"] = resetClean });

            string renderMode = env.RenderMode ?? throw new ValueError("Expected env.render_mode to be not None");
            if (renderMode.EndsWith("_list", StringComparison.Ordinal))
                throw new ValueError($"Expected env.render_mode to not end with '_list', got '{renderMode}'");

            FrameList = new List<RenderResult>();
            PopFrames = popFrames;
            ResetClean = resetClean;

            Metadata metadata = (Metadata)env.Metadata.DeepCopy();
            AddRenderMode(metadata, $"{renderMode}_list");
            WrapperMetadata = metadata;
        }

        /// <summary>The recorded constructor arguments, <c>{'pop_frames': True, 'reset_clean': True}</c> (Python's <c>_saved_kwargs</c>).</summary>
        [PyName("_saved_kwargs")]
        public Info SavedKwargs { get; }

        /// <summary>The frames collected since the last render or reset (<c>frame_list</c>), oldest first, owned by the wrapper.</summary>
        public List<RenderResult> FrameList { get; private set; }

        /// <summary>Whether <see cref="Render"/> clears the collection (<c>pop_frames</c>).</summary>
        public bool PopFrames { get; set; }

        /// <summary>Whether a reset clears the collection (<c>reset_clean</c>).</summary>
        public bool ResetClean { get; set; }

        /// <summary>The env's render mode with <c>_list</c> appended (<c>rgb_array_list</c>).</summary>
        /// <exception cref="AttributeError">Thrown when assigned, as for every wrapper.</exception>
        public override string? RenderMode {
            get => $"{Env.RenderMode}_list";
            protected set => base.RenderMode = value;
        }

        /// <summary>Steps the env, then collects a frame.</summary>
        /// <param name="action">The action, passed on.</param>
        /// <returns>The env's transition.</returns>
        public override StepResult<TObs> Step(TAct action) {
            StepResult<TObs> output = base.Step(action);
            // The frame outlives this call and any NumSharp scope the caller opened around it (DR-96).
            FrameList.Add(ScopeSafety.Keep(base.Render()));
            return output;
        }

        /// <summary>Resets the env, clears the collection when <see cref="ResetClean"/>, then collects the first frame.</summary>
        /// <param name="seed">The seed, passed on.</param>
        /// <param name="options">The options, passed on.</param>
        /// <returns>The env's first observation and info.</returns>
        /// <exception cref="Gym.Exceptions.Error">Thrown for a negative seed.</exception>
        public override ResetResult<TObs> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            ResetResult<TObs> output = base.Reset(seed, options);
            if (ResetClean)
                ReleaseFrames();
            // Kept past the caller's NumSharp scope, as every collected frame (DR-96).
            FrameList.Add(ScopeSafety.Keep(base.Render()));
            return output;
        }

        /// <summary>
        ///     The frames collected so far, as one <see cref="RenderResult.Frames"/>; with <see cref="PopFrames"/> the
        ///     collection then starts over and the caller owns the returned frames.
        /// </summary>
        /// <returns>The frames.</returns>
        public override RenderResult Render() {
            var frames = new RenderResult.Frames(FrameList);
            if (!PopFrames)
                return frames;
            // The frames now belong to the caller; a fresh list keeps the wrapper from releasing them, and the caller's NumSharp
            // scope, if one is open, adopts them as it adopts what it builds (they were detached when collected, DR-96).
            FrameList = new List<RenderResult>();
            return ScopeSafety.HandOver(frames);
        }

        /// <summary>Releases the frames still collected, then closes and disposes the env as every wrapper does.</summary>
        /// <param name="disposing"><see langword="true"/> from <see cref="Env{TObs, TAct}.Dispose()"/>.</param>
        protected override void Dispose(bool disposing) {
            if (disposing)
                ReleaseFrames();
            base.Dispose(disposing);
        }

        /// <summary>Clears the collection, releasing the frames it holds (Python's <c>self.frame_list = []</c>).</summary>
        private void ReleaseFrames() {
            foreach (RenderResult frame in FrameList)
                frame.Dispose();
            FrameList = new List<RenderResult>();
        }

        /// <summary>
        ///     Appends a render mode to <c>metadata["render_modes"]</c> unless it is there (Python's <c>append</c> on the
        ///     copied list). A list that cannot grow (a typed array, Python's list to Gym.NET) is replaced by a longer one.
        /// </summary>
        /// <param name="metadata">The wrapper's own metadata copy.</param>
        /// <param name="mode">The mode to add.</param>
        /// <exception cref="KeyError">Thrown when the metadata has no <c>render_modes</c>, as Python's <c>metadata["render_modes"]</c> raises.</exception>
        /// <exception cref="AttributeError">Thrown when <c>render_modes</c> is not a list (Python's <c>append</c> on another type).</exception>
        private static void AddRenderMode(Metadata metadata, string mode) {
            object? modes = metadata[Metadata.RenderModesKey];
            if (modes is IEnumerable sequence and not string && sequence.Cast<object?>().Any(item => item is string text && text == mode))
                return;
            switch (modes) {
                case IList { IsFixedSize: false, IsReadOnly: false } list:
                    list.Add(mode);
                    break;
                case IEnumerable items and not string:
                    metadata[Metadata.RenderModesKey] = items.Cast<object?>().Select(item => item?.ToString() ?? "None").Append(mode).ToList();
                    break;
                default:
                    throw new AttributeError($"'{PyBuiltins.TypeName(modes)}' object has no attribute 'append'");
            }
        }
    }

    /// <summary>
    ///     Records videos of episodes from the env's render, as Gymnasium's <c>RecordVideo</c> does: a recording starts when
    ///     the episode or step trigger fires (by default episodes 0, 1, 8, 27, ..., 729, then every 1000th), lasts the episode
    ///     or <c>video_length</c> frames, and is written through an <see cref="IVideoWriter"/> as
    ///     <c>{video_folder}/{name_prefix}-episode-{i}</c> or <c>-step-{i}</c>.
    /// </summary>
    /// <typeparam name="TObs">The observation type, passed through.</typeparam>
    /// <typeparam name="TAct">The action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     Gymnasium encodes with moviepy; here the encoder is pluggable (DR-25): <see cref="PngVideoWriter"/> by default (a
    ///     folder of PNG frames per video, no external program) or <see cref="FfmpegVideoWriter"/> for mp4. The triggers,
    ///     names, lengths and frame rate are Gymnasium's; a video's frames are the env's renders, so they equal Gymnasium's.
    ///     </para>
    ///     <para>
    ///     Every reset and step renders the env while recording, costing a frame's drawing each, and the frames stay in
    ///     memory until the video is written. Closing (or disposing) the wrapper writes a video still recording; Python only
    ///     warns from <c>__del__</c> about one left unsaved. <c>disable_logger</c> is kept for Gymnasium's signature: the
    ///     writers print no progress. <c>gc_trigger</c> left <see langword="null"/> is Gymnasium's default (collect after
    ///     every video); pass <c>_ =&gt; false</c> for Python's <c>None</c>.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/rendering.py#L166">gymnasium/wrappers/rendering.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class RecordVideo<TObs, TAct> : Wrapper<TObs, TAct>, IRecordConstructorArgs {
        /// <summary>The frames rendered for the caller in a <c>*_list</c> mode while capturing (<c>render_history</c>), handed out by the next render.</summary>
        private List<RenderResult> _renderHistory = new();

        /// <summary>Wraps <paramref name="env"/>, creating <paramref name="videoFolder"/>.</summary>
        /// <param name="env">The env; it must render images (<c>rgb_array</c> or a <c>*_list</c> mode).</param>
        /// <param name="videoFolder">The folder of the videos, created if needed (a warning when it exists).</param>
        /// <param name="episodeTrigger">Whether to start recording at an episode (by number from 0); with no trigger at all, <see cref="Video.CappedCubicVideoSchedule"/>.</param>
        /// <param name="stepTrigger">Whether to start recording at a step (counted across episodes from 0).</param>
        /// <param name="videoLength">The frames per video; 0 records whole episodes.</param>
        /// <param name="namePrefix">The prefix of the video names.</param>
        /// <param name="fps">The frame rate; <see langword="null"/> takes <c>metadata["render_fps"]</c>, else 30.</param>
        /// <param name="disableLogger">moviepy's logger switch, kept for Gymnasium's signature.</param>
        /// <param name="gcTrigger">Whether to collect garbage after a video of an episode; <see langword="null"/> always does.</param>
        /// <param name="videoWriter">The encoder; <see langword="null"/> for <see cref="PngVideoWriter"/>.</param>
        /// <exception cref="ValueError">Thrown when the env renders no images (no render mode, <c>human</c> or <c>ansi</c>), with Gymnasium's message.</exception>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="videoFolder"/> is <see langword="null"/>.</exception>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        public RecordVideo(Env<TObs, TAct> env, string videoFolder, Func<long, bool>? episodeTrigger = null, Func<long, bool>? stepTrigger = null,
            long videoLength = 0, string namePrefix = "rl-video", double? fps = null, bool disableLogger = true, Func<long, bool>? gcTrigger = null,
            IVideoWriter? videoWriter = null) : base(env) {
            SavedKwargs = RecordConstructorArgs.Record(new Info {
                ["video_folder"] = videoFolder, ["episode_trigger"] = episodeTrigger, ["step_trigger"] = stepTrigger,
                ["video_length"] = videoLength, ["name_prefix"] = namePrefix, ["fps"] = fps, ["disable_logger"] = disableLogger,
                ["gc_trigger"] = gcTrigger, ["video_writer"] = videoWriter,
            });
            if (videoFolder is null)
                throw new ArgumentNullException(nameof(videoFolder));

            if (env.RenderMode is null or "human" or "ansi")
                throw new ValueError($"Render mode is {PyBuiltins.Str(env.RenderMode)}, which is incompatible with RecordVideo. " +
                                     "Initialize your environment with a render_mode that returns an image, such as rgb_array.");

            if (episodeTrigger is null && stepTrigger is null)
                episodeTrigger = Video.CappedCubicVideoSchedule;

            EpisodeTrigger = episodeTrigger;
            StepTrigger = stepTrigger;
            DisableLogger = disableLogger;
            GcTrigger = gcTrigger ?? (_ => true);
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

        /// <summary>When to start recording at an episode (<c>episode_trigger</c>).</summary>
        public Func<long, bool>? EpisodeTrigger { get; set; }

        /// <summary>When to start recording at a step (<c>step_trigger</c>).</summary>
        public Func<long, bool>? StepTrigger { get; set; }

        /// <summary>moviepy's logger switch (<c>disable_logger</c>); the writers print nothing either way.</summary>
        public bool DisableLogger { get; set; }

        /// <summary>When to collect garbage after a video (<c>gc_trigger</c>).</summary>
        public Func<long, bool>? GcTrigger { get; set; }

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

        /// <summary>The frames of the video being recorded (<c>recorded_frames</c>), owned by the wrapper.</summary>
        public List<NDArray> RecordedFrames { get; private set; } = new();

        /// <summary>The current video's name without extension (<c>_video_name</c>), or <see langword="null"/>.</summary>
        [PyName("_video_name")]
        public string? VideoName { get; private set; }

        /// <summary>The number of the last step, across episodes, from 0 (<c>step_id</c>; -1 before the first).</summary>
        public long StepId { get; private set; } = -1;

        /// <summary>The number of the current episode, from 0 (<c>episode_id</c>; -1 before the first reset).</summary>
        public long EpisodeId { get; private set; } = -1;

        /// <summary>Resets the env, ends a whole-episode recording, and starts one when the episode trigger fires.</summary>
        /// <param name="seed">The seed, passed on.</param>
        /// <param name="options">The options, passed on.</param>
        /// <returns>The env's reset.</returns>
        /// <exception cref="Gym.Exceptions.Error">Thrown for a negative seed.</exception>
        public override ResetResult<TObs> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            ResetResult<TObs> output = base.Reset(seed, options);
            EpisodeId += 1;

            if (Recording && double.IsPositiveInfinity(VideoLength))
                StopRecording();

            if (EpisodeTrigger is not null && EpisodeTrigger(EpisodeId))
                StartRecording($"{NamePrefix}-episode-{EpisodeId}");
            if (Recording) {
                CaptureFrame();
                if (RecordedFrames.Count > VideoLength)
                    StopRecording();
            }

            return output;
        }

        /// <summary>Steps the env, starts a recording when the step trigger fires, and captures a frame while recording.</summary>
        /// <param name="action">The action, passed on.</param>
        /// <returns>The env's transition.</returns>
        public override StepResult<TObs> Step(TAct action) {
            StepResult<TObs> output = Env.Step(action);
            StepId += 1;

            if (StepTrigger is not null && StepTrigger(StepId))
                StartRecording($"{NamePrefix}-step-{StepId}");
            if (Recording) {
                CaptureFrame();
                if (RecordedFrames.Count > VideoLength)
                    StopRecording();
            }

            return output;
        }

        /// <summary>
        ///     The env's render; in a <c>*_list</c> mode the frames rendered while capturing come first, and while recording the
        ///     returned frames join the video too.
        /// </summary>
        /// <returns>The render, which the caller owns.</returns>
        public override RenderResult Render() {
            RenderResult renderOut = base.Render();
            if (Recording && renderOut is RenderResult.Frames frames) {
                // The caller owns the returned frames, so the video keeps copies, kept past the caller's NumSharp scope (DR-96).
                foreach (RenderResult item in frames.Items)
                    if (item is RenderResult.Rgb rgb)
                        RecordedFrames.Add(ScopeSafety.Keep(rgb.Frame.copy()));
            }

            if (_renderHistory.Count > 0) {
                List<RenderResult> history = _renderHistory;
                _renderHistory = new List<RenderResult>();
                // The history was kept past the caller's scopes; handing it over lets the caller's scope adopt it (DR-96).
                ScopeSafety.HandOver<IEnumerable<object?>>(history);
                IEnumerable<RenderResult> rest = renderOut is RenderResult.Frames list ? list.Items : new[] { renderOut };
                return new RenderResult.Frames(history.Concat(rest));
            }

            return renderOut;
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

            if (RecordedFrames.Count == 0) {
                Logger.Warn("Ignored saving a video as there were zero frames to save.");
            } else {
                // os.path.join(video_folder, f"{_video_name}.mp4"): the writer appends its own extension.
                VideoWriter.Write(System.IO.Path.Combine(VideoFolder, VideoName!), RecordedFrames, FramesPerSec);
            }

            ReleaseRecordedFrames();
            Recording = false;
            VideoName = null;

            if (GcTrigger is not null && GcTrigger(EpisodeId))
                GC.Collect();
        }

        /// <summary>Releases the frames still recorded and the history, then closes (writing a video still recording) and disposes as every wrapper does.</summary>
        /// <param name="disposing"><see langword="true"/> from <see cref="Env{TObs, TAct}.Dispose()"/>.</param>
        protected override void Dispose(bool disposing) {
            base.Dispose(disposing);
            if (!disposing)
                return;
            ReleaseRecordedFrames();
            foreach (RenderResult frame in _renderHistory)
                frame.Dispose();
            _renderHistory = new List<RenderResult>();
        }

        /// <summary>
        ///     Renders the env and keeps the frame (<c>_capture_frame</c>): in a <c>*_list</c> mode the rendered frames go to the
        ///     history for the caller and a copy of the last one to the video; a render that is not an array stops the
        ///     recording with a warning.
        /// </summary>
        /// <exception cref="AssertionError">Thrown when nothing is being recorded.</exception>
        private void CaptureFrame() {
            if (!Recording)
                throw new AssertionError("Cannot capture a frame, recording wasn't started.");

            RenderResult frame = Env.Render();
            if (frame is RenderResult.Frames list) {
                if (list.Items.Count == 0) {
                    Logger.Warn("Trying to capture render frame but 'env.render()' has just been called. The frame cannot be captured.");
                    return;
                }

                // The history outlives this call and the caller's NumSharp scope until Render hands it over (DR-96).
                _renderHistory.AddRange(ScopeSafety.Keep(list.Items));
                // The history hands the frames to the caller later, so the video keeps its own copy of the last one.
                frame = list.Items[list.Items.Count - 1] is RenderResult.Rgb last ? new RenderResult.Rgb(last.Frame.copy()) : list.Items[list.Items.Count - 1];
            }

            if (frame is RenderResult.Rgb rgb) {
                // The video's frames are kept until it is written: past any scope the caller opened (DR-96).
                RecordedFrames.Add(ScopeSafety.Keep(rgb.Frame));
            } else {
                StopRecording();
                Logger.Warn($"Recording stopped: expected type of frame returned by render to be a numpy array, got instead {RenderTypeRepr(frame)}.");
            }
        }

        /// <summary>Releases the recorded frames (Python's <c>del self.recorded_frames</c>).</summary>
        private void ReleaseRecordedFrames() {
            foreach (NDArray frame in RecordedFrames)
                frame.Dispose();
            RecordedFrames = new List<NDArray>();
        }

        /// <summary>Python's <c>type(frame)</c> of a render that is no array.</summary>
        /// <param name="frame">The render.</param>
        /// <returns><c>&lt;class 'NoneType'&gt;</c>, <c>&lt;class 'str'&gt;</c>, ...</returns>
        private static string RenderTypeRepr(RenderResult frame) => frame switch {
            RenderResult.NoneResult => "<class 'NoneType'>",
            RenderResult.Ansi => "<class 'str'>",
            RenderResult.Frames => "<class 'list'>",
            RenderResult.RgbDepth => "<class 'tuple'>",
            _ => "<class 'numpy.ndarray'>",
        };
    }

    /// <summary>
    ///     Shows an env that renders images in a window, as Gymnasium's <c>HumanRendering</c> does: every reset and step
    ///     renders the env and shows the frame at <c>metadata["render_fps"]</c>, so the wrapper's render mode is
    ///     <c>human</c>. <c>make</c> adds it when <c>render_mode="human"</c> is asked of an env that only renders images.
    /// </summary>
    /// <typeparam name="TObs">The observation type, passed through.</typeparam>
    /// <typeparam name="TAct">The action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     The window is a viewer from the wrapper's own factory, else <c>Gymnasium.HumanViewerFactory</c> (DR-20, DR-89),
    ///     opened at the first frame's size; without either the first frame raises
    ///     <see cref="Gym.Exceptions.DependencyNotInstalled"/>, as Gymnasium does without pygame. Frames are paced by <see cref="Gym.Rendering.Clock"/>, which sleeps to hold the frame rate. A frame of
    ///     another size than the first fails the assertion Gymnasium makes; frames of another dtype than uint8 are cast.
    ///     </para>
    ///     <para>
    ///     In a <c>*_list</c> render mode the env's own <c>render()</c> returns an empty list afterwards, since this wrapper
    ///     takes every frame (Gymnasium's warning).
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/rendering.py#L446">gymnasium/wrappers/rendering.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class HumanRendering<TObs, TAct> : Wrapper<TObs, TAct>, IRecordConstructorArgs, IHumanViewerFactoryHolder {
        /// <summary>The render modes the wrapper can show (<c>ACCEPTED_RENDER_MODES</c>).</summary>
        public static readonly IReadOnlyList<string> AcceptedRenderModes = new[] { "rgb_array", "rgb_array_list", "depth_array", "depth_array_list" };

        /// <summary>The window, opened at the first frame (<c>window</c>).</summary>
        private IEnvViewer? _window;

        /// <summary>The frame pacer, created at the first frame (<c>clock</c>).</summary>
        private Gym.Rendering.Clock? _clock;

        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env; its render mode must be one of <see cref="AcceptedRenderModes"/> and its metadata must have <c>render_fps</c>.</param>
        /// <exception cref="ValueError">Thrown for another render mode, or metadata without <c>render_fps</c>, with Gymnasium's messages.</exception>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        public HumanRendering(Env<TObs, TAct> env) : base(env) {
            SavedKwargs = RecordConstructorArgs.Record(new Info());

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

        /// <summary>The recorded constructor arguments, <c>{}</c> (Python's <c>_saved_kwargs</c>).</summary>
        [PyName("_saved_kwargs")]
        public Info SavedKwargs { get; }

        /// <summary>
        ///     Gym.NET's viewer factory for this wrapper's window (DR-20, DR-89): set it before the first frame, as
        ///     <c>new HumanRendering&lt;TObs, TAct&gt;(env) { ViewerFactory = factory }</c> or as <c>make</c> does with the factory
        ///     its render-mode overloads were given; <see langword="null"/> (the default) uses <c>Gymnasium.HumanViewerFactory</c>,
        ///     read when the window opens. A property, not a constructor argument, so the constructor and the arguments it
        ///     records stay Gymnasium's: a spec never carries the delegate.
        /// </summary>
        /// <remarks>Set after the window opened, it takes effect only if the window is closed and opens again.</remarks>
        public EnvViewerFactory? ViewerFactory { get; set; }

        /// <summary>The window's size, the first frame's <c>(width, height)</c> (<c>screen_size</c>), or <see langword="null"/> before it.</summary>
        public (int Width, int Height)? ScreenSize { get; private set; }

        /// <summary>Always <c>human</c>.</summary>
        /// <exception cref="AttributeError">Thrown when assigned, as for every wrapper.</exception>
        public override string? RenderMode {
            get => "human";
            protected set => base.RenderMode = value;
        }

        /// <summary>Steps the env, then shows its frame.</summary>
        /// <param name="action">The action, passed on.</param>
        /// <returns>The env's transition.</returns>
        /// <exception cref="Gym.Exceptions.DependencyNotInstalled">Thrown at the first frame when no viewer factory is set.</exception>
        public override StepResult<TObs> Step(TAct action) {
            StepResult<TObs> result = base.Step(action);
            RenderFrame();
            return result;
        }

        /// <summary>Resets the env, then shows its frame.</summary>
        /// <param name="seed">The seed, passed on.</param>
        /// <param name="options">The options, passed on.</param>
        /// <returns>The env's reset.</returns>
        /// <exception cref="Gym.Exceptions.Error">Thrown for a negative seed.</exception>
        /// <exception cref="Gym.Exceptions.DependencyNotInstalled">Thrown at the first frame when no viewer factory is set.</exception>
        public override ResetResult<TObs> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            ResetResult<TObs> result = base.Reset(seed, options);
            RenderFrame();
            return result;
        }

        /// <summary>Nothing: the frames are shown by <see cref="Step"/> and <see cref="Reset"/>.</summary>
        /// <returns><see cref="RenderResult.None"/>.</returns>
        public override RenderResult Render() => RenderResult.None;

        /// <summary>Closes the window, then the env.</summary>
        public override void Close() {
            if (_window is not null) {
                _window.Close();
                _window = null;
            }

            base.Close();
        }

        /// <summary>
        ///     Shows the env's last frame (<c>_render_frame</c>): the frame of <c>render()</c> (the last of a list), the window
        ///     opened at its size, paced to <c>render_fps</c>, then shown.
        /// </summary>
        /// <exception cref="AssertionError">Thrown when the render is no array, or a frame's size differs from the first's, with Gymnasium's messages.</exception>
        /// <exception cref="ValueError">Thrown for a frame without three axes (NumPy's <c>axes don't match array</c> from the transpose).</exception>
        /// <exception cref="IndexError">Thrown when a <c>*_list</c> render returned no frames (<c>list index out of range</c>).</exception>
        /// <exception cref="Gym.Exceptions.DependencyNotInstalled">Thrown when the window must open and no viewer factory is set.</exception>
        private void RenderFrame() {
            using RenderResult rendered = Env.Render();
            RenderResult last = rendered;
            if (Env.RenderMode!.EndsWith("_list", StringComparison.Ordinal)) {
                // assert isinstance(last_rgb_array, list), then last_rgb_array[-1], which an empty list refuses.
                if (rendered is not RenderResult.Frames frames)
                    throw new AssertionError("");
                if (frames.Items.Count == 0)
                    throw new IndexError("list index out of range");
                last = frames.Items[frames.Items.Count - 1];
            }

            NDArray frame = last switch {
                RenderResult.Rgb rgb => rgb.Frame,
                RenderResult.Depth depth => depth.Frame,
                _ => throw new AssertionError($"Expected `env.render()` to return a numpy array, actually returned {(last is RenderResult.NoneResult ? "<class 'NoneType'>" : "<class 'list'>")}"),
            };

            // np.transpose(frame, axes=(1, 0, 2)) makes the window (width, height); a frame without three axes fails there.
            if (frame.ndim != 3)
                throw new ValueError("axes don't match array");
            (int Width, int Height) size = ((int)frame.shape[1], (int)frame.shape[0]);
            ScreenSize ??= size;
            if (ScreenSize != size)
                throw new AssertionError($"The shape of the rgb array has changed from ({ScreenSize.Value.Width}, {ScreenSize.Value.Height}) to ({size.Width}, {size.Height})");

            if (_window is null) {
                EnvViewerFactory factory = global::Gym.Gymnasium.ResolveHumanViewerFactory(ViewerFactory);
                // The factory returns once its window is up (DR-85); frames are then shown synchronously, as pygame's.
                _window = factory(size.Width, size.Height, "HumanRendering");
            }

            _clock ??= new Gym.Rendering.Clock();
            _clock.Tick(PyBuiltins.Float(Metadata[Metadata.RenderFpsKey]));
            // The viewer contract's frame is uint8 (DR-85): another dtype is cast, as pygame's make_surface reads it as bytes.
            using NDArray? cast = frame.typecode == NPTypeCode.Byte ? null : frame.astype(np.uint8);
            _window.Render(cast ?? frame);
        }
    }

    /// <summary>
    ///     A wrapper that opens a human window from a viewer factory make can hand it (DR-89): both <c>HumanRendering</c>
    ///     wrappers, whose generic types make cannot name when it applies one.
    /// </summary>
    internal interface IHumanViewerFactoryHolder {
        /// <summary>The factory the window opens from; <see langword="null"/> for the app-wide one.</summary>
        EnvViewerFactory? ViewerFactory { get; set; }
    }

    /// <summary>
    ///     Replaces each pixel of the rendered frame by white noise with a given probability, as Gymnasium's
    ///     <c>AddWhiteNoise</c> does; with <c>AddRenderObservation</c> it makes pixel observations noisy, turning the MDP into a
    ///     POMDP.
    /// </summary>
    /// <typeparam name="TObs">The observation type, passed through.</typeparam>
    /// <typeparam name="TAct">The action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     Every render draws the noise and then the mask from the env's generator (<c>self.np_random</c>), so rendering
    ///     advances the env's stream. The frame must be an array (<c>rgb_array</c>). The result is a new array: uint8 for RGB
    ///     noise, uint64 for grayscale noise (NumPy's <c>sum</c> of uint8 promotes to uint64, and <c>np.where</c> keeps it).
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/rendering.py#L595">gymnasium/wrappers/rendering.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class AddWhiteNoise<TObs, TAct> : Wrapper<TObs, TAct>, IRecordConstructorArgs {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env; it must render arrays.</param>
        /// <param name="probabilityOfNoisePerPixel">The probability that a pixel is noise, in [0, 1).</param>
        /// <param name="isNoiseGrayscale">Whether the noise is gray (the RGB noise weighted and summed) rather than colored.</param>
        /// <exception cref="Gym.Exceptions.InvalidProbability">Thrown for a probability outside [0, 1): <c>probability_of_noise_per_pixel should be in the interval [0,1). Received 1.0</c>.</exception>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        public AddWhiteNoise(Env<TObs, TAct> env, double probabilityOfNoisePerPixel, bool isNoiseGrayscale = false)
            : base(CheckProbability(env, probabilityOfNoisePerPixel)) {
            SavedKwargs = RecordConstructorArgs.Record(new Info {
                ["probability_of_noise_per_pixel"] = probabilityOfNoisePerPixel, ["is_noise_grayscale"] = isNoiseGrayscale,
            });
            ProbabilityOfNoisePerPixel = probabilityOfNoisePerPixel;
            IsNoiseGrayscale = isNoiseGrayscale;
        }

        /// <summary>The recorded constructor arguments (Python's <c>_saved_kwargs</c>).</summary>
        [PyName("_saved_kwargs")]
        public Info SavedKwargs { get; }

        /// <summary>The probability that a pixel is noise (<c>probability_of_noise_per_pixel</c>).</summary>
        public double ProbabilityOfNoisePerPixel { get; set; }

        /// <summary>Whether the noise is gray (<c>is_noise_grayscale</c>).</summary>
        public bool IsNoiseGrayscale { get; set; }

        /// <summary>The env's frame with noise in place of each pixel the mask picks.</summary>
        /// <returns>A new frame (<see cref="RenderResult.Rgb"/>) the caller owns.</returns>
        /// <exception cref="AttributeError">Thrown when the env's render is not an array, as Python's <c>render_out.shape</c> raises.</exception>
        public override RenderResult Render() {
            using RenderResult renderOut = base.Render();
            NDArray frame = RenderNoise.FrameOf(renderOut);
            using NDArray noise = RenderNoise.Noise(NpRandom, frame.Shape, IsNoiseGrayscale);
            using NDArray uniform = NpRandom.random(new Shape(frame.shape[0], frame.shape[1]));
            using NDArray mask = uniform < ProbabilityOfNoisePerPixel;
            return new RenderResult.Rgb(RenderNoise.Where(mask, noise, frame));
        }

        /// <summary>Gymnasium's check of the probability, made before the wrapper is built as Python makes it before recording anything.</summary>
        /// <param name="env">The env, passed through.</param>
        /// <param name="probability">The probability.</param>
        /// <returns><paramref name="env"/>.</returns>
        /// <exception cref="Gym.Exceptions.InvalidProbability">Thrown for a probability outside [0, 1) or NaN.</exception>
        private static Env<TObs, TAct> CheckProbability(Env<TObs, TAct> env, double probability) {
            if (!(0 <= probability && probability < 1))
                throw new Gym.Exceptions.InvalidProbability($"probability_of_noise_per_pixel should be in the interval [0,1). Received {PyBuiltins.Str(probability)}");
            return env;
        }
    }

    /// <summary>
    ///     Covers random square patches of the rendered frame with white noise, as Gymnasium's <c>ObstructView</c> does: about
    ///     <c>obstructed_pixels_ratio</c> of the pixels in patches <c>obstruction_width</c> wide.
    /// </summary>
    /// <typeparam name="TObs">The observation type, passed through.</typeparam>
    /// <typeparam name="TAct">The action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     Every render draws the patch centres and then the noise from the env's generator, so rendering advances the env's
    ///     stream. Patches near the edge are cut by it and patches overlap, so the covered share is at most the ratio. The
    ///     frame must be an array (<c>rgb_array</c>); the result is a new array, uint8 for RGB noise and uint64 for grayscale
    ///     noise, as in Gymnasium.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/rendering.py#L671">gymnasium/wrappers/rendering.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class ObstructView<TObs, TAct> : Wrapper<TObs, TAct>, IRecordConstructorArgs {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env; it must render arrays.</param>
        /// <param name="obstructedPixelsRatio">The share of pixels to cover, in [0, 1).</param>
        /// <param name="obstructionWidth">The side of each square patch, at least 1.</param>
        /// <param name="isNoiseGrayscale">Whether the noise is gray rather than colored.</param>
        /// <exception cref="ValueError">Thrown for a ratio outside [0, 1) or a width below 1, with Gymnasium's messages.</exception>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        public ObstructView(Env<TObs, TAct> env, double obstructedPixelsRatio, long obstructionWidth, bool isNoiseGrayscale = false)
            : base(CheckArguments(env, obstructedPixelsRatio, obstructionWidth)) {
            SavedKwargs = RecordConstructorArgs.Record(new Info {
                ["obstructed_pixels_ratio"] = obstructedPixelsRatio, ["obstruction_width"] = obstructionWidth, ["is_noise_grayscale"] = isNoiseGrayscale,
            });
            // obstructed_pixels_ratio / obstruction_width**2: the int power is exact, the division Python's float one.
            ObstructionCentersRatio = obstructedPixelsRatio / (double)(obstructionWidth * obstructionWidth);
            ObstructionWidth = obstructionWidth;
            IsNoiseGrayscale = isNoiseGrayscale;
        }

        /// <summary>The recorded constructor arguments (Python's <c>_saved_kwargs</c>).</summary>
        [PyName("_saved_kwargs")]
        public Info SavedKwargs { get; }

        /// <summary>The patch centres per pixel (<c>obstruction_centers_ratio</c>): the ratio over the patch's area.</summary>
        public double ObstructionCentersRatio { get; set; }

        /// <summary>The side of each patch (<c>obstruction_width</c>).</summary>
        public long ObstructionWidth { get; set; }

        /// <summary>Whether the noise is gray (<c>is_noise_grayscale</c>).</summary>
        public bool IsNoiseGrayscale { get; set; }

        /// <summary>The env's frame with noise in the patches.</summary>
        /// <returns>A new frame (<see cref="RenderResult.Rgb"/>) the caller owns.</returns>
        /// <exception cref="AttributeError">Thrown when the env's render is not an array, as Python's <c>render_out.shape</c> raises.</exception>
        public override RenderResult Render() {
            using RenderResult renderOut = base.Render();
            NDArray frame = RenderNoise.FrameOf(renderOut);
            long height = frame.shape[0], width = frame.shape[1];
            long nPixels = height * width;
            // int(n_pixels * ratio): Python's float product, truncated.
            long nObstructions = (long)PyBuiltins.Int(nPixels * ObstructionCentersRatio);
            using NDArray centers = NpRandom.integers(0, nPixels, size: new Shape(nObstructions));
            using NDArray mask = np.zeros(new Shape(height, width), np.@bool);
            long low = ObstructionWidth / 2, high = ObstructionWidth - low;
            foreach (long center in centers.ToArray<long>()) {
                // np.unravel_index(center, (height, width)) in C order.
                long x = center / width, y = center % width;
                long x0 = Math.Max(x - low, 0), x1 = Math.Min(x + high, height);
                long y0 = Math.Max(y - low, 0), y1 = Math.Min(y + high, width);
                // An empty slice (start past stop never happens here, but a zero-width one may) writes nothing, as NumPy's.
                if (x0 < x1 && y0 < y1)
                    mask[$"{x0}:{x1}, {y0}:{y1}"] = true;
            }

            using NDArray noise = RenderNoise.Noise(NpRandom, frame.Shape, IsNoiseGrayscale);
            return new RenderResult.Rgb(RenderNoise.Where(mask, noise, frame));
        }

        /// <summary>Gymnasium's checks of the ratio and the width, made before the wrapper is built.</summary>
        /// <param name="env">The env, passed through.</param>
        /// <param name="ratio">The ratio.</param>
        /// <param name="width">The patch width.</param>
        /// <returns><paramref name="env"/>.</returns>
        /// <exception cref="ValueError">Thrown for a ratio outside [0, 1) or NaN, or a width below 1.</exception>
        private static Env<TObs, TAct> CheckArguments(Env<TObs, TAct> env, double ratio, long width) {
            if (!(0 <= ratio && ratio < 1))
                throw new ValueError($"obstructed_pixels_ratio should be in the interval [0,1). Received {PyBuiltins.Str(ratio)}");
            if (width < 1)
                throw new ValueError($"obstruction_width should be larger or equal than 1. Received {width}");
            return env;
        }
    }

    /// <summary>The noise and blending <see cref="AddWhiteNoise{TObs, TAct}"/> and <see cref="ObstructView{TObs, TAct}"/> share, in NumPy's dtypes.</summary>
    internal static class RenderNoise {
        /// <summary>The frame of a render, as Python's <c>cast(np.ndarray, render_out)</c> takes it.</summary>
        /// <param name="renderOut">The env's render; it keeps the frame.</param>
        /// <returns>The frame array.</returns>
        /// <exception cref="AttributeError">Thrown when the render carries no array (<c>'NoneType' object has no attribute 'shape'</c>).</exception>
        internal static NDArray FrameOf(RenderResult renderOut) => renderOut switch {
            RenderResult.Rgb rgb => rgb.Frame,
            RenderResult.Depth depth => depth.Frame,
            RenderResult.NoneResult => throw new AttributeError("'NoneType' object has no attribute 'shape'"),
            RenderResult.Ansi => throw new AttributeError("'str' object has no attribute 'shape'"),
            _ => throw new AttributeError($"'{(renderOut is RenderResult.Frames ? "list" : "tuple")}' object has no attribute 'shape'"),
        };

        /// <summary>
        ///     The noise of a frame's shape from the env's generator: <c>integers(0, 255, size=shape, dtype=uint8)</c>, or for
        ///     gray noise <c>integers((0, 0, 0), 255 * [0.2989, 0.5870, 0.1140], size=shape, dtype=uint8)</c> summed over the
        ///     channels (uint64) and repeated to three.
        /// </summary>
        /// <param name="rng">The env's generator.</param>
        /// <param name="shape">The frame's shape, <c>(H, W, 3)</c>.</param>
        /// <param name="grayscale">Whether the noise is gray.</param>
        /// <returns>A new array the caller owns.</returns>
        internal static NDArray Noise(Generator rng, Shape shape, bool grayscale) {
            if (!grayscale)
                return rng.integers(0, 255, size: shape, dtype: np.uint8);
            using NDArray low = np.array(new long[] { 0, 0, 0 });
            using NDArray weights = np.array(new[] { 0.2989, 0.5870, 0.1140 });
            using NDArray high = 255 * weights;
            using NDArray channels = rng.integers(low, high, size: shape, dtype: np.uint8);
            using NDArray summed = np.sum(channels, axis: -1, keepdims: true);
            return np.repeat(summed, 3, axis: -1);
        }

        /// <summary><c>np.where(mask[..., None], noise, frame)</c>: the noise where the mask is set, the frame elsewhere, in their promoted dtype.</summary>
        /// <param name="mask">The <c>(H, W)</c> bool mask.</param>
        /// <param name="noise">The noise, the frame's shape.</param>
        /// <param name="frame">The frame.</param>
        /// <returns>A new array the caller owns.</returns>
        internal static NDArray Where(NDArray mask, NDArray noise, NDArray frame) {
            using NDArray expanded = np.expand_dims(mask, -1);
            return np.where(expanded, noise, frame);
        }
    }
}
