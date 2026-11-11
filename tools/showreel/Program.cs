using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;

namespace Gym.Showreel;

/// <summary>
///     The showreel tool's command line.
/// </summary>
/// <remarks>
///     <code>
///     Showreel [all]       gallery and logo, README loop, then the video (the default)
///     Showreel video       docs/media/gym-net-showreel.mp4 and the site's poster for it (needs the parity stack unless --no-parity)
///     Showreel hero        docs/media/gym-net-hero.gif
///     Showreel gallery     docs/website-src/images/envs/*, the site's logo and icon
///     Showreel parity      only the Gymnasium comparison, printing what was compared
///     Showreel stills      PNG frames of a storyboard: --board video|hero --at 1.5,20,33 [--out DIR]
///     options: --repo DIR (default: the checkout this runs in), --ffmpeg EXE, --python EXE, --gymnasium DIR,
///              --work DIR (Python exchange files), --no-parity, --gif-width N (default 800)
///     </code>
/// </remarks>
internal static class Program {
    /// <summary>Runs a command.</summary>
    /// <param name="args">The command and its options.</param>
    /// <returns>0 on success, 1 on failure (the message goes to stderr), 2 on a usage error.</returns>
    public static int Main(string[] args) {
        try {
            var options = Options.Parse(args);
            string command = options.Command ?? "all";
            switch (command) {
                case "all":
                    Gallery(options);
                    Hero(options);
                    Video(options);
                    return 0;
                case "video":
                    Video(options);
                    return 0;
                case "hero":
                    Hero(options);
                    return 0;
                case "gallery":
                    Gallery(options);
                    return 0;
                case "parity":
                    using (ParityReplays replays = ParityReplays.Create(options.Python, Console.Out))
                        Console.WriteLine($"parity: {replays.Lander.Steps} + {replays.Racer.Steps} records identical");
                    return 0;
                case "stills":
                    Stills(options);
                    return 0;
                default:
                    Console.Error.WriteLine($"unknown command '{command}': all, video, hero, gallery, parity or stills");
                    return 2;
            }
        } catch (ArgumentException e) {
            Console.Error.WriteLine(e.Message);
            return 2;
        } catch (Exception e) when (e is InvalidOperationException or IOException) {
            Console.Error.WriteLine(e.Message);
            return 1;
        }
    }

    /// <summary>
    ///     Renders the video, with the parity segment unless <c>--no-parity</c>, and the docs site's poster for it (the
    ///     title, 3.3 s in). No scanlines: the site scales the video to its column, and resampled scanlines moiré.
    /// </summary>
    /// <param name="o">The options.</param>
    private static void Video(Options o) {
        string output = o.Output ?? Path.Combine(o.Repo, "docs", "media", "gym-net-showreel.mp4");
        using (ParityReplays? parity = o.NoParity ? null : ParityReplays.Create(o.Python, Console.Out))
            Render(Storyboards.Video(parity), output, o.Ffmpeg, FfmpegPipe.Mp4(), scanlines: false);
        if (o.Output is null)
            FfmpegPipe.ExtractFrame(o.Ffmpeg, output, 3.3, Path.Combine(o.Repo, "docs", "website-src", "images", "showreel-poster.jpg"));
    }

    /// <summary>Renders the README's loop as a GIF.</summary>
    /// <param name="o">The options.</param>
    private static void Hero(Options o) {
        string output = o.Output ?? Path.Combine(o.Repo, "docs", "media", "gym-net-hero.gif");
        Storyboard board = Storyboards.Hero() with { Fps = 15 };
        Render(board, output, o.Ffmpeg, FfmpegPipe.Gif(o.GifWidth), scanlines: false);
    }

    /// <summary>Writes the gallery's loops and manifest and the site's logo.</summary>
    /// <param name="o">The options.</param>
    private static void Gallery(Options o) {
        string images = Path.Combine(o.Repo, "docs", "website-src", "images");
        Showreel.Gallery.Write(o.Output ?? Path.Combine(images, "envs"), o.Ffmpeg, Console.Out);
        Showreel.Gallery.WriteLogo(Path.Combine(images, "gymnet-logo.png"), Path.Combine(images, "gymnet-icon.png"));
    }

    /// <summary>Writes PNG stills of a storyboard at chosen times, for checking a design without encoding.</summary>
    /// <param name="o">The options.</param>
    /// <exception cref="ArgumentException">No <c>--at</c> times, or an unknown board.</exception>
    private static void Stills(Options o) {
        if (o.At.Count == 0)
            throw new ArgumentException("stills needs --at with comma-separated seconds");
        if (o.Board is not ("video" or "hero"))
            throw new ArgumentException($"unknown board '{o.Board}': video or hero");
        // The replays own large temporary files: dispose them even when a still fails.
        using ParityReplays? parity = o.Board == "video" && !o.NoParity ? ParityReplays.Create(o.Python, Console.Out) : null;
        Storyboard board = o.Board == "video" ? Storyboards.Video(parity) : Storyboards.Hero();
        string folder = o.Output ?? Path.Combine(Path.GetTempPath(), "gymnet-showreel-stills");
        Directory.CreateDirectory(folder);
        using var compositor = new Compositor(board, scanlines: o.Board == "video");
        foreach (double t in o.At.OrderBy(t => t)) {
            string path = Path.Combine(folder, $"{o.Board}_{t.ToString("000.00", CultureInfo.InvariantCulture)}.png");
            compositor.Render(t).SavePng(path);
            Console.WriteLine(path);
        }
    }

    /// <summary>Renders a storyboard frame by frame into an ffmpeg pipe, reporting progress every few seconds.</summary>
    /// <param name="board">The timeline.</param>
    /// <param name="output">The file.</param>
    /// <param name="ffmpeg">The ffmpeg executable.</param>
    /// <param name="encoder">The output options.</param>
    /// <param name="scanlines">Whether to draw scanlines.</param>
    private static void Render(Storyboard board, string output, string ffmpeg, string[] encoder, bool scanlines) {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
        var clock = Stopwatch.StartNew();
        using var compositor = new Compositor(board, scanlines);
        using var pipe = new FfmpegPipe(ffmpeg, Slots.Width, Slots.Height, board.Fps, encoder, output);
        long lastReport = 0;
        for (int f = 0; f < board.Frames; f++) {
            pipe.Write(compositor.Render((double)f / board.Fps));
            if (clock.ElapsedMilliseconds - lastReport > 5000) {
                lastReport = clock.ElapsedMilliseconds;
                Console.WriteLine($"{Path.GetFileName(output)}: frame {f + 1}/{board.Frames}, {clock.Elapsed.TotalSeconds:0}s");
            }
        }
        pipe.Finish();
        Console.WriteLine($"{output}: {board.Frames} frames, {new FileInfo(output).Length / 1024} KB, {clock.Elapsed.TotalSeconds:0}s");
    }

    /// <summary>The parsed command line.</summary>
    private sealed class Options {
        /// <summary>The command, or null for the default.</summary>
        public string? Command { get; private set; }

        /// <summary>The repository root that default outputs go under.</summary>
        public string Repo { get; private set; } = "";

        /// <summary>An explicit output file or folder.</summary>
        public string? Output { get; private set; }

        /// <summary>The ffmpeg executable.</summary>
        public string Ffmpeg { get; private set; } = "ffmpeg";

        /// <summary>Where the parity segment's Python half runs.</summary>
        public PythonSide Python { get; private set; } = null!;

        /// <summary>Whether to leave the parity segment out.</summary>
        public bool NoParity { get; private set; }

        /// <summary>The README GIF's width.</summary>
        public int GifWidth { get; private set; } = 800;

        /// <summary>The storyboard for <c>stills</c>.</summary>
        public string Board { get; private set; } = "video";

        /// <summary>The times for <c>stills</c>.</summary>
        public List<double> At { get; } = new();

        /// <summary>Parses the arguments.</summary>
        /// <param name="args">The arguments.</param>
        /// <returns>The options.</returns>
        /// <exception cref="ArgumentException">An unknown option, a missing value, or no repository found.</exception>
        public static Options Parse(string[] args) {
            var o = new Options();
            string? repo = null, python = null, gymnasium = null, work = null;
            for (int i = 0; i < args.Length; i++) {
                string a = args[i];
                string Value() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{a} needs a value");
                switch (a) {
                    case "--repo": repo = Value(); break;
                    case "--out": o.Output = Value(); break;
                    case "--ffmpeg": o.Ffmpeg = Value(); break;
                    case "--python": python = Value(); break;
                    case "--gymnasium": gymnasium = Value(); break;
                    case "--work": work = Value(); break;
                    case "--no-parity": o.NoParity = true; break;
                    case "--gif-width": o.GifWidth = int.Parse(Value(), CultureInfo.InvariantCulture); break;
                    case "--board": o.Board = Value(); break;
                    case "--at":
                        o.At.AddRange(Value().Split(',', StringSplitOptions.RemoveEmptyEntries).Select(s => double.Parse(s, CultureInfo.InvariantCulture)));
                        break;
                    default:
                        if (a.StartsWith("--", StringComparison.Ordinal) || o.Command is not null)
                            throw new ArgumentException($"unknown argument '{a}'");
                        o.Command = a;
                        break;
                }
            }
            o.Repo = Path.GetFullPath(repo ?? FindRepo() ?? throw new ArgumentException("not inside a Gym.NET checkout: pass --repo"));
            o.Python = new PythonSide(python ?? "python", Path.GetFullPath(gymnasium ?? Path.Combine(o.Repo, "refs", "Gymnasium")),
                Path.Combine(AppContext.BaseDirectory, "gymnasium_replay.py"), Path.GetFullPath(work ?? Path.Combine(Path.GetTempPath(), "gymnet-showreel")));
            return o;
        }

        /// <summary>The nearest folder upward from the working directory that holds <c>Gym.NET.sln</c>.</summary>
        /// <returns>The folder, or null.</returns>
        private static string? FindRepo() {
            for (var d = new DirectoryInfo(Environment.CurrentDirectory); d is not null; d = d.Parent)
                if (File.Exists(Path.Combine(d.FullName, "Gym.NET.sln")))
                    return d.FullName;
            return null;
        }
    }
}
