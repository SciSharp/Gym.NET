#nullable enable
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Gym.Exceptions;
using NumSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Gym.Utils {
    /// <summary>
    ///     Encodes one video from rendered frames (decision DR-25): what <c>RecordVideo</c> and <see cref="Video.SaveVideo"/>
    ///     hand their frames to, where Gymnasium hands them to moviepy. Implement it to store videos another way.
    /// </summary>
    /// <remarks>
    ///     <see cref="PngVideoWriter"/> (the default) writes each video as a folder of PNG frames, with no external program;
    ///     <see cref="FfmpegVideoWriter"/> pipes the frames to <c>ffmpeg</c> for an mp4 like Gymnasium's. Parity with
    ///     Gymnasium is on the frames handed over, never on the encoded bytes.
    /// </remarks>
    public interface IVideoWriter {
        /// <summary>
        ///     What the writer appends to a video's path: <c>.mp4</c> for a file, or the empty string for a folder named like
        ///     Gymnasium's video file without its extension.
        /// </summary>
        string Extension { get; }

        /// <summary>Writes one video, replacing a video already at that path.</summary>
        /// <param name="path">The video's path without extension: Gymnasium's <c>{video_folder}/{name}</c>, the folder existing.</param>
        /// <param name="frames">The frames in order, <c>(H, W, 3)</c> RGB (or <c>(H, W, 4)</c>, <c>(H, W)</c>); they stay the caller's.</param>
        /// <param name="fps">The frame rate.</param>
        /// <exception cref="ArgumentException">Thrown for no frames, or frames of different shapes.</exception>
        void Write(string path, IReadOnlyList<NDArray> frames, double fps);
    }

    /// <summary>
    ///     The default <see cref="IVideoWriter"/>: each video a folder of PNG frames (<c>000000.png</c>, <c>000001.png</c>,
    ///     ...) and an <c>fps.txt</c> holding its frame rate. Needs no external program; <c>ffmpeg -framerate {fps} -i
    ///     %06d.png out.mp4</c> turns a folder into a video.
    /// </summary>
    /// <remarks>
    ///     A folder per video keeps Gymnasium's count of videos (<c>len(os.listdir(video_folder))</c>) and names. Writing
    ///     over an existing video first deletes the frames it wrote there (files named like its frames), so a shorter video
    ///     leaves none of a longer one's behind; other files in the folder are left alone. Frames of another dtype than uint8
    ///     are cast to uint8 first, as moviepy does.
    /// </remarks>
    public sealed class PngVideoWriter : IVideoWriter {
        /// <summary>The frame files this writer names: six digits, <c>.png</c>.</summary>
        private static readonly Regex FrameFile = new(@"^\d{6}\.png$", RegexOptions.CultureInvariant);

        /// <summary>The empty string: a video is a folder.</summary>
        public string Extension => "";

        /// <summary>Writes the frames as <c>{path}/000000.png</c>, ... and the frame rate as <c>{path}/fps.txt</c>.</summary>
        /// <param name="path">The video's folder, created if needed.</param>
        /// <param name="frames">The frames; they stay the caller's.</param>
        /// <param name="fps">The frame rate, written to <c>fps.txt</c>.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="path"/> or <paramref name="frames"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown for no frames, or a frame that is not an image of 2 or 3 dimensions.</exception>
        /// <exception cref="IOException">Thrown when a file cannot be written.</exception>
        public void Write(string path, IReadOnlyList<NDArray> frames, double fps) {
            if (path is null)
                throw new ArgumentNullException(nameof(path));
            if (frames is null)
                throw new ArgumentNullException(nameof(frames));
            if (frames.Count == 0)
                throw new ArgumentException("A video needs at least one frame.", nameof(frames));

            Directory.CreateDirectory(path);
            // Only this writer's own frame files are removed: anything else a user keeps there survives.
            foreach (string stale in Directory.EnumerateFiles(path).Where(file => FrameFile.IsMatch(Path.GetFileName(file))).ToList())
                File.Delete(stale);

            for (int i = 0; i < frames.Count; i++) {
                using Image image = VideoFrames.ToImage(frames[i]);
                image.SaveAsPng(Path.Combine(path, i.ToString("D6", CultureInfo.InvariantCulture) + ".png"));
            }

            File.WriteAllText(Path.Combine(path, "fps.txt"), PyBuiltins.Str(fps));
        }
    }

    /// <summary>
    ///     An <see cref="IVideoWriter"/> that pipes raw frames to an <c>ffmpeg</c> process and writes an H.264 mp4, as moviepy
    ///     does for Gymnasium: <c>libx264</c> at the <c>medium</c> preset, <c>yuv420p</c> when both sides are even.
    /// </summary>
    /// <remarks>
    ///     <c>ffmpeg</c> must be installed; it is looked up on <c>PATH</c> unless <see cref="Executable"/> names it. Each
    ///     video starts one process and waits for it; the frames stream through its standard input, so memory stays at one
    ///     frame. The bytes differ from moviepy's (another encoder build and options), which DR-25 accepts: parity is on
    ///     frames.
    /// </remarks>
    public sealed class FfmpegVideoWriter : IVideoWriter {
        /// <summary>The <c>ffmpeg</c> executable: a path, or a name looked up on <c>PATH</c>.</summary>
        public string Executable { get; init; } = "ffmpeg";

        /// <summary>The video codec (<c>-vcodec</c>), <c>libx264</c> as moviepy's default.</summary>
        public string Codec { get; init; } = "libx264";

        /// <summary>The encoder preset (<c>-preset</c>), <c>medium</c> as moviepy's default.</summary>
        public string Preset { get; init; } = "medium";

        /// <summary><c>.mp4</c>.</summary>
        public string Extension => ".mp4";

        /// <summary>Encodes the frames into <c>{path}.mp4</c>, replacing an existing file.</summary>
        /// <param name="path">The video's path without extension.</param>
        /// <param name="frames">The frames, all of one shape; they stay the caller's.</param>
        /// <param name="fps">The frame rate.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="path"/> or <paramref name="frames"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown for no frames, frames of different shapes, or a frame that is not an image.</exception>
        /// <exception cref="DependencyNotInstalled">Thrown when <c>ffmpeg</c> cannot be started.</exception>
        /// <exception cref="Error">Thrown when <c>ffmpeg</c> fails, with its error output.</exception>
        public void Write(string path, IReadOnlyList<NDArray> frames, double fps) {
            if (path is null)
                throw new ArgumentNullException(nameof(path));
            if (frames is null)
                throw new ArgumentNullException(nameof(frames));
            if (frames.Count == 0)
                throw new ArgumentException("A video needs at least one frame.", nameof(frames));

            long[] shape = frames[0].shape;
            if (shape.Length is not (2 or 3))
                throw new ArgumentException($"Expected (H, W) or (H, W, C) frames, got {frames[0].ndim} dimensions.", nameof(frames));
            long channels = shape.Length == 3 ? shape[2] : 1;
            string pixelFormat = channels switch {
                1 => "gray",
                3 => "rgb24",
                4 => "rgba",
                _ => throw new ArgumentException($"Expected 1, 3 or 4 channels, got {channels}.", nameof(frames)),
            };

            var arguments = new List<string> {
                "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", $"{shape[1]}x{shape[0]}", "-pix_fmt", pixelFormat, "-r", PyBuiltins.Str(fps), "-i", "-",
                "-an", "-vcodec", Codec, "-preset", Preset,
            };
            // moviepy asks libx264 for yuv420p only when both sides are even (the format's chroma needs them so).
            if (Codec == "libx264" && shape[0] % 2 == 0 && shape[1] % 2 == 0)
                arguments.AddRange(new[] { "-pix_fmt", "yuv420p" });
            arguments.Add(path + Extension);

            var start = new ProcessStartInfo(Executable) {
                RedirectStandardInput = true, RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true,
            };
            foreach (string argument in arguments)
                start.ArgumentList.Add(argument);

            Process process;
            try {
                process = Process.Start(start) ?? throw new DependencyNotInstalled($"ffmpeg could not be started ({Executable}).");
            } catch (System.ComponentModel.Win32Exception e) {
                throw new DependencyNotInstalled($"ffmpeg is not installed or not on PATH ({Executable}): {e.Message}");
            }

            using (process) {
                // Read the error stream concurrently: a full stderr pipe would block ffmpeg while we block on stdin.
                var stderr = process.StandardError.ReadToEndAsync();
                try {
                    using Stream input = process.StandardInput.BaseStream;
                    foreach (NDArray frame in frames) {
                        if (!frame.shape.SequenceEqual(shape))
                            throw new ArgumentException($"Every frame must have the first frame's shape ({string.Join(", ", shape)}).", nameof(frames));
                        byte[] bytes = VideoFrames.Bytes(frame);
                        input.Write(bytes, 0, bytes.Length);
                    }
                } finally {
                    process.WaitForExit();
                }

                if (process.ExitCode != 0)
                    throw new Error($"ffmpeg failed with exit code {process.ExitCode}: {stderr.GetAwaiter().GetResult().Trim()}");
            }
        }
    }

    /// <summary>
    ///     Gymnasium's <c>gymnasium.utils.save_video</c>: the default recording schedule, and saving videos from a list of
    ///     frames by episode and step triggers.
    /// </summary>
    public static class Video {
        /// <summary>
        ///     The default episode trigger (<c>capped_cubic_video_schedule</c>): episodes 0, 1, 8, 27, ..., 729 (the cubes
        ///     below 1000), then every 1000th.
        /// </summary>
        /// <param name="episodeId">The episode number.</param>
        /// <returns>Whether to record that episode.</returns>
        /// <remarks>
        ///     <c>int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id</c> in Python's float arithmetic: a negative id gives a
        ///     complex power, which Python's <c>round</c> refuses, so it raises <see cref="TypeError"/> here too.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/save_video.py#L20">gymnasium/utils/save_video.py</see>.
        /// </remarks>
        /// <exception cref="TypeError">Thrown for a negative episode id (<c>type complex doesn't define __round__ method</c>).</exception>
        public static bool CappedCubicVideoSchedule(long episodeId) {
            if (episodeId < 1000) {
                if (episodeId < 0)
                    throw new TypeError("type complex doesn't define __round__ method");
                // Python's round of a float is to even, and ** of floats is C's pow.
                long root = (long)Math.Round(Math.Pow(episodeId, 1.0 / 3), MidpointRounding.ToEven);
                return root * root * root == episodeId;
            }

            return episodeId % 1000 == 0;
        }

        /// <summary>
        ///     Saves videos from an episode's frames (<c>save_video</c>): the whole episode (up to <paramref name="videoLength"/>
        ///     frames) when the episode trigger fires, and from every step whose step trigger fires a clip of
        ///     <paramref name="videoLength"/> frames (to the end without it), the reset's frame skipped.
        /// </summary>
        /// <param name="frames">The episode's frames, the reset's first; they stay the caller's.</param>
        /// <param name="videoFolder">The folder, created if needed.</param>
        /// <param name="episodeTrigger">Whether to save the episode; with no trigger at all, <see cref="CappedCubicVideoSchedule"/>.</param>
        /// <param name="stepTrigger">Whether to save a clip from a step.</param>
        /// <param name="videoLength">The frames per video, a Python slice end (negative counts from the end); <see langword="null"/> for all.</param>
        /// <param name="namePrefix">The file name prefix.</param>
        /// <param name="episodeIndex">The episode's number, for the trigger and the name.</param>
        /// <param name="stepStartingIndex">The step number of the first frame after the reset's.</param>
        /// <param name="saveLogger">moviepy's progress logger (<c>"bar"</c>); kept for Gymnasium's signature, the writers report nothing.</param>
        /// <param name="fps">The frame rate; moviepy needs it (or durations, which Gym.NET does not take).</param>
        /// <param name="videoWriter">The encoder; <see langword="null"/> for <see cref="PngVideoWriter"/>.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="frames"/> or <paramref name="videoFolder"/> is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">Thrown when a video is due and <paramref name="fps"/> is <see langword="null"/>, with moviepy's message.</exception>
        /// <remarks>
        ///     The paths are Gymnasium's, <c>{abspath(video_folder)}/{name_prefix}-episode-{i}</c> and
        ///     <c>...-step-{i}</c>, with the writer's extension instead of <c>.mp4</c>.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/save_video.py#L37">gymnasium/utils/save_video.py</see>.
        /// </remarks>
        public static void SaveVideo(IReadOnlyList<NDArray> frames, string videoFolder, Func<long, bool>? episodeTrigger = null,
            Func<long, bool>? stepTrigger = null, long? videoLength = null, string namePrefix = "rl-video", long episodeIndex = 0,
            long stepStartingIndex = 0, string? saveLogger = null, double? fps = null, IVideoWriter? videoWriter = null) {
            if (frames is null)
                throw new ArgumentNullException(nameof(frames));
            if (videoFolder is null)
                throw new ArgumentNullException(nameof(videoFolder));
            if (episodeTrigger is null && stepTrigger is null)
                episodeTrigger = CappedCubicVideoSchedule;
            videoWriter ??= new PngVideoWriter();

            string folder = Path.GetFullPath(videoFolder);
            Directory.CreateDirectory(folder);
            string pathPrefix = $"{folder}/{namePrefix}";

            if (episodeTrigger is not null && episodeTrigger(episodeIndex))
                WriteClip(videoWriter, $"{pathPrefix}-episode-{episodeIndex}", PySlice(frames, 0, videoLength), fps);

            if (stepTrigger is not null) {
                // Skip the first frame since it comes from reset.
                long stepIndex = stepStartingIndex;
                for (int frameIndex = 1; frameIndex < frames.Count; frameIndex++, stepIndex++) {
                    if (!stepTrigger(stepIndex))
                        continue;
                    long? endIndex = videoLength is { } length ? frameIndex + length : null;
                    WriteClip(videoWriter, $"{pathPrefix}-step-{stepIndex}", PySlice(frames, frameIndex, endIndex), fps);
                }
            }
        }

        /// <summary>One clip: moviepy's <c>ImageSequenceClip(frames, fps=fps).write_videofile(path)</c> through the writer.</summary>
        /// <param name="writer">The writer.</param>
        /// <param name="path">The video's path without extension.</param>
        /// <param name="frames">The clip's frames.</param>
        /// <param name="fps">The frame rate.</param>
        /// <exception cref="ValueError">Thrown when <paramref name="fps"/> is <see langword="null"/>.</exception>
        private static void WriteClip(IVideoWriter writer, string path, IReadOnlyList<NDArray> frames, double? fps) {
            if (fps is not { } rate)
                throw new ValueError("Please provide either 'fps' or 'durations'.");
            writer.Write(path, frames, rate);
        }

        /// <summary><c>frames[start:end]</c> with Python's slice rules: negative ends count from the end, both are clipped.</summary>
        /// <param name="frames">The frames.</param>
        /// <param name="start">The first index.</param>
        /// <param name="end">The end index, or <see langword="null"/> for the end.</param>
        /// <returns>The slice; empty when the end is not past the start.</returns>
        private static IReadOnlyList<NDArray> PySlice(IReadOnlyList<NDArray> frames, long start, long? end) {
            long count = frames.Count;
            long Normalize(long index) => index < 0 ? Math.Max(index + count, 0) : Math.Min(index, count);
            long from = Normalize(start), to = end is { } e ? Normalize(e) : count;
            return to <= from ? Array.Empty<NDArray>() : frames.Skip((int)from).Take((int)(to - from)).ToList();
        }
    }

    /// <summary>The frame conversions the writers share.</summary>
    internal static class VideoFrames {
        /// <summary>A frame's bytes in C order, cast to uint8 when it is another dtype (moviepy's <c>astype("uint8")</c>).</summary>
        /// <param name="frame">The frame; not changed.</param>
        /// <returns>The bytes.</returns>
        internal static byte[] Bytes(NDArray frame) {
            if (frame.typecode == NPTypeCode.Byte)
                return frame.ToArray<byte>();
            using NDArray bytes = frame.astype(np.uint8);
            return bytes.ToArray<byte>();
        }

        /// <summary>A frame as an image: RGB for 3 channels, RGBA for 4, gray for none.</summary>
        /// <param name="frame">The frame, <c>(H, W)</c>, <c>(H, W, 3)</c> or <c>(H, W, 4)</c>.</param>
        /// <returns>A new image the caller disposes.</returns>
        /// <exception cref="ArgumentException">Thrown for any other shape.</exception>
        internal static Image ToImage(NDArray frame) {
            byte[] bytes = Bytes(frame);
            int height = (int)frame.shape[0], width = frame.ndim >= 2 ? (int)frame.shape[1] : 0;
            return frame.ndim switch {
                2 => Image.LoadPixelData<L8>(bytes, width, height),
                3 when frame.shape[2] == 3 => Image.LoadPixelData<Rgb24>(bytes, width, height),
                3 when frame.shape[2] == 4 => Image.LoadPixelData<Rgba32>(bytes, width, height),
                _ => throw new ArgumentException($"Expected an (H, W), (H, W, 3) or (H, W, 4) frame, got shape ({string.Join(", ", frame.shape)})."),
            };
        }
    }
}
