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
///     Showreel [all]       gallery and logo, the README animation, then the video (the default; one parity run for both)
///     Showreel video       docs/media/gym-net-showreel.mp4 and the site's poster for it (needs the parity stack unless --no-parity),
///                          its scenes in the order tools/showreel/video-order.txt gives
///     Showreel scenes      where each scene of the order starts and how long the video runs, without rendering
///     Showreel hero        docs/media/gym-net-hero.webp, the README's 51 s cut (also needs the parity stack unless --no-parity);
///                          --format gif writes docs/media/gym-net-hero.gif instead
///     Showreel gallery     docs/website-src/images/envs/*, the site's logo and icon
///     Showreel parity      only the Gymnasium comparison, printing what was compared
///     Showreel stills      PNG frames of a storyboard: --board video|hero --at 1.5,20,33 [--out DIR]
///     Showreel train       trains MuJoCo policies with ARS into tools/showreel/policies: --env Hopper-v5,Ant-v5 [--minutes M]
///     options: --repo DIR (default: the checkout this runs in), --ffmpeg EXE, --python EXE, --gymnasium DIR,
///              --work DIR (Python exchange files), --no-parity, --width N (the README animation's, default 800;
///              --gif-width is the same option), --format webp|gif,
///              --order "A D B ..." (the video's scenes, in place of video-order.txt), --letters (each scene's letter
///              in the corner: a preview, written to the temp folder unless --out says where)
///     </code>
///     <para>
///     The MuJoCo recipes play the committed policies in <c>tools/showreel/policies</c>; <c>train</c> is only needed to
///     replace one. Training is deterministic for its seed but takes minutes per env (the budgets are in
///     <see cref="Ars.For"/>).
///     </para>
///     <para>
///     The video is 14 lettered scenes (DR-100, <see cref="Storyboards.Scenes"/>); the order file is the maintainer's to
///     edit, and the tool reads it from the checkout, so a new order needs no rebuild.
///     </para>
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
                    // A typo in the video's order should stop the run now, not after the gallery and the README cut.
                    VideoOrder(options);
                    Gallery(options);
                    // The README cut and the video show the same verified replays: one Python run serves both, and its
                    // temporary files are released when both are rendered, even if one fails.
                    using (ParityReplays? parity = options.NoParity ? null : ParityReplays.Create(options.Python, Console.Out)) {
                        Hero(options, parity);
                        Video(options, parity);
                    }
                    return 0;
                case "video":
                    // The order is read and checked before the parity run: a typo in it costs nothing.
                    VideoOrder(options);
                    using (ParityReplays? parity = options.NoParity ? null : ParityReplays.Create(options.Python, Console.Out))
                        Video(options, parity);
                    return 0;
                case "scenes":
                    Scenes(options);
                    return 0;
                case "hero":
                    using (ParityReplays? parity = options.NoParity ? null : ParityReplays.Create(options.Python, Console.Out))
                        Hero(options, parity);
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
                case "train":
                    Train(options);
                    return 0;
                default:
                    Console.Error.WriteLine($"unknown command '{command}': all, video, scenes, hero, gallery, parity, stills or train");
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
    ///     Trains a policy per <c>--env</c> id with Augmented Random Search and writes it to
    ///     <c>tools/showreel/policies/&lt;id&gt;.json</c> in the repository (rebuild the tool to ship it next to the program).
    /// </summary>
    /// <param name="o">The options.</param>
    /// <exception cref="ArgumentException">No <c>--env</c>, or an id without training settings.</exception>
    private static void Train(Options o) {
        if (o.Envs.Count == 0)
            throw new ArgumentException("train needs --env with comma-separated ids");
        string folder = Path.Combine(o.Repo, "tools", "showreel", "policies");
        Directory.CreateDirectory(folder);
        foreach (string id in o.Envs) {
            Ars.Settings settings = Ars.For(id);
            if (o.Minutes is double minutes)
                settings = settings with { Minutes = minutes };
            LinearPolicy policy = Ars.Train(id, settings, Console.Out);
            string path = Path.Combine(folder, id + ".json");
            policy.Save(path);
            Console.WriteLine($"{path}: evaluation mean {policy.Info["evaluation_mean"]}");
        }
    }

    /// <summary>
    ///     Renders the video, its scenes in <see cref="VideoOrder"/>'s order, with the parity scene when replays are given,
    ///     and the docs site's poster for it (the title, 3.3 s into its scene). No scanlines: the site scales the video to
    ///     its column, and resampled scanlines moiré.
    /// </summary>
    /// <param name="o">The options.</param>
    /// <param name="parity">The verified Gymnasium replays, or <see langword="null"/> for <c>--no-parity</c>; the caller disposes them.</param>
    /// <exception cref="ArgumentException">The order names a letter that is not a scene, or leaves no scene to play.</exception>
    /// <exception cref="IOException">The order file exists but cannot be read.</exception>
    /// <exception cref="InvalidOperationException">ffmpeg cannot start, or fails writing the video or the poster.</exception>
    /// <remarks>
    ///     With <c>--letters</c> and no <c>--out</c>, the lettered preview goes to the temp folder and the poster is left
    ///     alone: a preview must never overwrite the published video or its poster.
    /// </remarks>
    private static void Video(Options o, ParityReplays? parity) {
        string order = VideoOrder(o);
        string output = o.Output ?? (o.Letters
            ? Path.Combine(Path.GetTempPath(), "gymnet-showreel-letters.mp4")
            : Path.Combine(o.Repo, "docs", "media", "gym-net-showreel.mp4"));
        Console.WriteLine($"video: scenes {string.Join(" ", Storyboards.Layout(order, parity is not null).Placed.Select(p => p.Scene.Letter))}");
        Render(Storyboards.Video(parity, order, o.Letters), output, o.Ffmpeg, FfmpegPipe.Mp4(), scanlines: false);
        if (o.Output is null && !o.Letters) {
            // The poster is the title's frame wherever the order plays it (3.3 s in, the pitch typed on), else 3.3 s in.
            var (placed, _) = Storyboards.Layout(order, parity is not null);
            double title = placed.Where(p => p.Scene.Letter == 'A').Select(p => p.Start).DefaultIfEmpty(0).First();
            FfmpegPipe.ExtractFrame(o.Ffmpeg, output, title + 3.3, Path.Combine(o.Repo, "docs", "website-src", "images", "showreel-poster.jpg"));
        }
    }

    /// <summary>
    ///     The video's scene order: <c>--order</c> when given, else <c>tools/showreel/video-order.txt</c> in the checkout,
    ///     else <see cref="Storyboards.DefaultOrder"/>. Parsed once here, so a bad letter stops the tool before any
    ///     rendering or parity run.
    /// </summary>
    /// <param name="o">The options.</param>
    /// <returns>The order's text.</returns>
    /// <exception cref="ArgumentException">The order names a letter that is not a scene.</exception>
    /// <exception cref="IOException">The order file exists but cannot be read.</exception>
    private static string VideoOrder(Options o) {
        string file = Path.Combine(o.Repo, "tools", "showreel", "video-order.txt");
        string order = o.Order ?? (File.Exists(file) ? File.ReadAllText(file) : Storyboards.DefaultOrder);
        Storyboards.ParseOrder(order, Storyboards.Scenes());
        return order;
    }

    /// <summary>
    ///     Prints the order's scenes with their start times and the video's length: the times <c>stills --at</c> needs, at
    ///     no rendering cost.
    /// </summary>
    /// <param name="o">The options (<c>--order</c> and <c>--no-parity</c> count).</param>
    /// <exception cref="ArgumentException">The order names a letter that is not a scene, or leaves no scene to play.</exception>
    /// <exception cref="IOException">The order file exists but cannot be read.</exception>
    private static void Scenes(Options o) {
        string order = VideoOrder(o);
        var (placed, seconds) = Storyboards.Layout(order, withParity: !o.NoParity);
        foreach (var (scene, start) in placed)
            Console.WriteLine($"{scene.Letter}  {start.ToString("0.00", CultureInfo.InvariantCulture),6} s  {scene.Name}");
        Console.WriteLine($"   {seconds.ToString("0.00", CultureInfo.InvariantCulture),6} s  the end");
    }

    /// <summary>
    ///     Renders the README animation, the showreel's 51 s cut (DR-73), at 15 frames a second: an animated WebP, or a GIF
    ///     with <c>--format gif</c>. No scanlines, as for the video.
    /// </summary>
    /// <param name="o">The options.</param>
    /// <param name="parity">The verified Gymnasium replays, or <see langword="null"/> for <c>--no-parity</c>; the caller disposes them.</param>
    /// <exception cref="ArgumentException"><c>--format</c> is neither <c>webp</c> nor <c>gif</c>.</exception>
    private static void Hero(Options o, ParityReplays? parity) {
        // Checked before rendering, so a typo costs nothing rather than minutes of frames.
        string[] encoder = o.Format switch {
            "webp" => FfmpegPipe.WebP(o.Width),
            "gif" => FfmpegPipe.Gif(o.Width),
            _ => throw new ArgumentException($"unknown --format '{o.Format}': webp or gif"),
        };
        string output = o.Output ?? Path.Combine(o.Repo, "docs", "media", "gym-net-hero." + o.Format);
        Storyboard board = Storyboards.Hero(parity) with { Fps = 15 };
        Render(board, output, o.Ffmpeg, encoder, scanlines: false);
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
        // A bad order stops the tool before the parity run.
        string? order = o.Board == "video" ? VideoOrder(o) : null;
        // The replays own large temporary files: dispose them even when a still fails. Both boards hold the parity segment.
        using ParityReplays? parity = o.NoParity ? null : ParityReplays.Create(o.Python, Console.Out);
        Storyboard board = order is not null ? Storyboards.Video(parity, order, o.Letters) : Storyboards.Hero(parity);
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

        /// <summary>The README animation's width (<c>--width</c>, or its older name <c>--gif-width</c>).</summary>
        public int Width { get; private set; } = 800;

        /// <summary>The README animation's format: <c>webp</c> (what the README shows) or <c>gif</c>.</summary>
        public string Format { get; private set; } = "webp";

        /// <summary>The storyboard for <c>stills</c>.</summary>
        public string Board { get; private set; } = "video";

        /// <summary>The times for <c>stills</c>.</summary>
        public List<double> At { get; } = new();

        /// <summary>The ids for <c>train</c>.</summary>
        public List<string> Envs { get; } = new();

        /// <summary>A time budget per env for <c>train</c> replacing <see cref="Ars.For"/>'s, or null.</summary>
        public double? Minutes { get; private set; }

        /// <summary>The video's scene order from <c>--order</c>, or null to read <c>video-order.txt</c>.</summary>
        public string? Order { get; private set; }

        /// <summary>Whether to print each scene's letter over the video (<c>--letters</c>): a preview, never the published video.</summary>
        public bool Letters { get; private set; }

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
                    case "--width" or "--gif-width": o.Width = int.Parse(Value(), CultureInfo.InvariantCulture); break;
                    case "--format": o.Format = Value().ToLowerInvariant(); break;
                    case "--board": o.Board = Value(); break;
                    case "--at":
                        o.At.AddRange(Value().Split(',', StringSplitOptions.RemoveEmptyEntries).Select(s => double.Parse(s, CultureInfo.InvariantCulture)));
                        break;
                    case "--env": o.Envs.AddRange(Value().Split(',', StringSplitOptions.RemoveEmptyEntries)); break;
                    case "--minutes": o.Minutes = double.Parse(Value(), CultureInfo.InvariantCulture); break;
                    case "--order": o.Order = Value(); break;
                    case "--letters": o.Letters = true; break;
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
