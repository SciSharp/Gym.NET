using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Threading.Tasks;

namespace Gym.Showreel;

/// <summary>
///     An ffmpeg process fed raw RGB24 frames on stdin, encoding to one file: the video, the README's WebP animation (or
///     its GIF) and the gallery's WebP loops all go through it, so the tool itself never encodes anything.
/// </summary>
/// <remarks>
///     ffmpeg must be on <c>PATH</c> (or named with <c>--ffmpeg</c>); this machine's is 7.0. The process runs with
///     <c>-v error</c>, and its stderr is collected so a failure reports ffmpeg's own message. <see cref="Finish"/> must be
///     called to close the input and wait for the file; <see cref="Dispose"/> alone kills an unfinished encode.
/// </remarks>
internal sealed class FfmpegPipe : IDisposable {
    /// <summary>The encoder process.</summary>
    private readonly Process _process;

    /// <summary>The process's stdin, where frames go.</summary>
    private readonly Stream _input;

    /// <summary>The frame size every <see cref="Write"/> must match.</summary>
    private readonly int _width, _height;

    /// <summary>ffmpeg's stderr, read concurrently so a chatty encoder never blocks on a full pipe.</summary>
    private readonly Task<string> _stderr;

    /// <summary>The output file.</summary>
    private readonly string _output;

    /// <summary>Whether <see cref="Finish"/> completed.</summary>
    private bool _finished;

    /// <summary>Starts ffmpeg reading raw frames of one size and rate, with the given output options.</summary>
    /// <param name="ffmpeg">The executable.</param>
    /// <param name="width">Frame width.</param>
    /// <param name="height">Frame height.</param>
    /// <param name="fps">Frame rate of the input.</param>
    /// <param name="outputArgs">Options between the input and the output file (codec, filters).</param>
    /// <param name="output">The file to write; overwritten.</param>
    /// <exception cref="InvalidOperationException">ffmpeg cannot be started.</exception>
    public FfmpegPipe(string ffmpeg, int width, int height, double fps, IEnumerable<string> outputArgs, string output) {
        _width = width;
        _height = height;
        _output = output;
        var start = new ProcessStartInfo(ffmpeg) {
            RedirectStandardInput = true,
            RedirectStandardError = true,
            RedirectStandardOutput = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        foreach (string a in new[] { "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", $"{width}x{height}",
                     "-r", fps.ToString(CultureInfo.InvariantCulture), "-i", "-" })
            start.ArgumentList.Add(a);
        foreach (string a in outputArgs)
            start.ArgumentList.Add(a);
        start.ArgumentList.Add(output);
        try {
            _process = Process.Start(start) ?? throw new InvalidOperationException($"could not start '{ffmpeg}'");
        } catch (System.ComponentModel.Win32Exception e) {
            throw new InvalidOperationException($"could not start '{ffmpeg}': {e.Message} (is ffmpeg on PATH?)", e);
        }
        _input = _process.StandardInput.BaseStream;
        _stderr = _process.StandardError.ReadToEndAsync();
        _ = _process.StandardOutput.ReadToEndAsync();
    }

    /// <summary>Sends one frame.</summary>
    /// <param name="frame">The frame; its size must be the pipe's.</param>
    /// <exception cref="ArgumentException">The frame's size differs.</exception>
    /// <exception cref="InvalidOperationException">ffmpeg exited early; carries its message.</exception>
    public void Write(RgbImage frame) {
        if (frame.Width != _width || frame.Height != _height)
            throw new ArgumentException($"frame is {frame.Width}x{frame.Height}, the pipe takes {_width}x{_height}", nameof(frame));
        try {
            _input.Write(frame.Pixels);
        } catch (IOException e) {
            // A broken pipe means ffmpeg died: its stderr says why, which matters more than the pipe error.
            _process.WaitForExit();
            throw new InvalidOperationException($"ffmpeg stopped while writing {_output}: {_stderr.Result}", e);
        }
    }

    /// <summary>Closes the input and waits for ffmpeg to finish the file.</summary>
    /// <exception cref="InvalidOperationException">ffmpeg failed; carries its message.</exception>
    public void Finish() {
        _input.Close();
        _process.WaitForExit();
        if (_process.ExitCode != 0)
            throw new InvalidOperationException($"ffmpeg failed ({_process.ExitCode}) writing {_output}: {_stderr.Result}");
        _finished = true;
    }

    /// <summary>Kills ffmpeg if <see cref="Finish"/> was not reached (an exception mid-encode), then releases it.</summary>
    public void Dispose() {
        if (!_finished && !_process.HasExited) {
            try {
                _process.Kill();
            } catch (InvalidOperationException) {
                // It exited between the check and the kill: nothing to stop.
            }
        }
        _process.Dispose();
    }

    /// <summary>
    ///     Options for the video: H.264 at CRF 20 (measured: about 8 MB for the minute, where CRF 18 made 10.4 MB with
    ///     no visible gain), in the pixel format every player reads, with the index first so browsers stream it.
    /// </summary>
    /// <returns>The output options.</returns>
    public static string[] Mp4() => new[] { "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart" };

    /// <summary>Writes one frame of a video as an image (the site's poster for the video), its format from the extension.</summary>
    /// <param name="ffmpeg">The ffmpeg executable.</param>
    /// <param name="video">The video.</param>
    /// <param name="seconds">The time of the frame.</param>
    /// <param name="output">The image file; overwritten.</param>
    /// <exception cref="InvalidOperationException">ffmpeg cannot start or fails; carries its message.</exception>
    public static void ExtractFrame(string ffmpeg, string video, double seconds, string output) {
        var start = new ProcessStartInfo(ffmpeg) {
            RedirectStandardError = true,
            RedirectStandardOutput = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        foreach (string a in new[] { "-v", "error", "-y", "-ss", seconds.ToString(CultureInfo.InvariantCulture), "-i", video, "-frames:v", "1", "-q:v", "3", output })
            start.ArgumentList.Add(a);
        using Process process = Process.Start(start) ?? throw new InvalidOperationException($"could not start '{ffmpeg}'");
        var stderr = process.StandardError.ReadToEndAsync();
        process.StandardOutput.ReadToEnd();
        process.WaitForExit();
        if (process.ExitCode != 0)
            throw new InvalidOperationException($"ffmpeg failed ({process.ExitCode}) extracting {output}: {stderr.Result}");
    }

    /// <summary>
    ///     Options for the README animation as a GIF (<c>hero --format gif</c>; the README itself shows the WebP of
    ///     <see cref="WebP"/>): area-scaled to a width, one 256-colour palette for the whole animation, ordered dithering
    ///     (it compresses far better than error diffusion and does not crawl between frames), and only the changed
    ///     rectangle stored per frame.
    /// </summary>
    /// <param name="width">Output width; the height follows the aspect.</param>
    /// <returns>The output options.</returns>
    /// <remarks>
    ///     One palette cannot cover the 51 s cut's 24 envs: measured on it, the GIF is 19 MB, dithered, with pink smears
    ///     where the grey MuJoCo skies find no fitting colour, which is why DR-73 moved the README to WebP. There is no
    ///     5 MB ceiling to respect: GitHub's camo proxy refuses larger images, but it only fronts images on other hosts,
    ///     and the README's come straight from <c>raw.githubusercontent.com</c> (checked with GitHub's own markdown
    ///     renderer, 2026-11-12). The previous edition dithered at <c>bayer_scale=4</c> to meet that ceiling; the
    ///     ceiling never applied, so this is DR-59's scale 3 again.
    /// </remarks>
    public static string[] Gif(int width) => new[] {
        "-vf", $"scale={width}:-2:flags=area,split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
        "-loop", "0",
    };

    /// <summary>
    ///     Options for the README animation (DR-73): area-scaled to a width, lossy animated WebP at quality 75 through
    ///     libwebp's animation encoder, looping. Every frame keeps its own colours, with no palette and no dithering.
    /// </summary>
    /// <param name="width">Output width; the height follows the aspect.</param>
    /// <returns>The output options.</returns>
    /// <remarks>
    ///     Measured on the 51 s cut at 800 px: 7.2 MB, against 19 MB for <see cref="Gif"/> and 17.4 MB for ffmpeg's
    ///     single-image <c>libwebp</c> encoder, which stores every frame whole; <c>libwebp_anim</c> stores what changed. At
    ///     quality 75 the MuJoCo skies and the sun's glow stay smooth, with faint block edges only around sharp neon lines.
    /// </remarks>
    public static string[] WebP(int width) =>
        new[] { "-vf", $"scale={width}:-2:flags=area", "-c:v", "libwebp_anim", "-lossless", "0", "-quality", "75", "-compression_level", "4", "-loop", "0" };

    /// <summary>Options for a gallery loop: area-scaled to the 8-bit size, lossless animated WebP, looping.</summary>
    /// <param name="width">Output width.</param>
    /// <param name="height">Output height.</param>
    /// <returns>The output options.</returns>
    public static string[] PixelWebP(int width, int height) =>
        new[] { "-vf", $"scale={width}:{height}:flags=area", "-c:v", "libwebp", "-lossless", "1", "-compression_level", "6", "-loop", "0" };
}
