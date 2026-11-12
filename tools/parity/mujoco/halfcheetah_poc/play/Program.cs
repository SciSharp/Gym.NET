using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Gym.Environments;
using Gym.Rendering;
using Gym.Rendering.Avalonia;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Gym.MuJoCo;

namespace MujocoPoc.Play;

/// <summary>
///     Plays HalfCheetah-v5 on the managed core in a window: the trained autopilot runs, Tab hands the six motors to the
///     keyboard. Also trains the autopilot, records it to a video, and prints its exact episode returns for comparison with
///     Gymnasium on MuJoCo (replay_in_gymnasium.py).
/// </summary>
/// <remarks>
///     Usage:
///     <code>
///     HalfCheetahPlay [play] [--seed N] [--manual]          the window (default)
///     HalfCheetahPlay train [--iterations N] [--threads N] [--seed N] [--out policy.json]
///     HalfCheetahPlay video FILE.mp4 [--seconds S] [--seed N]   needs ffmpeg on PATH
///     HalfCheetahPlay frames FOLDER [--steps N] [--every N] [--seed N]
///     HalfCheetahPlay score [--seeds A-B]
///     </code>
///     The model (half_cheetah.json, MuJoCo's compiled HalfCheetah as Gym.Environments embeds it) and the autopilot (policy.json) are read from
///     the program's folder; <c>--policy FILE</c> picks another autopilot.
/// </remarks>
internal static class Program {
    private const string Title = "HalfCheetah-v5 - Gym.NET managed core, bit-exact with MuJoCo 4.0.0";

    /// <summary>Dispatches the command.</summary>
    /// <param name="args">The command and its options (see the class remarks).</param>
    /// <returns>0 on success, 1 on a failure the output explains, 2 on bad usage.</returns>
    private static int Main(string[] args) {
        CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
        string command = args.Length > 0 && !args[0].StartsWith("--", StringComparison.Ordinal) ? args[0] : "play";
        var options = new Options(args.SkipWhile(a => !a.StartsWith("--", StringComparison.Ordinal)).ToArray());
        string folder = AppContext.BaseDirectory;
        MjModel model = MjModel.Load(Path.Combine(folder, "half_cheetah.json"));
        string policyPath = options.Text("policy", Path.Combine(folder, "policy.json"));
        try {
            switch (command) {
                case "play":
                    int code = Play(model, File.Exists(policyPath) ? LinearPolicy.Load(policyPath) : null, options.Int("seed", 0), options.Flag("manual"));
                    // the Avalonia UI thread keeps the process alive; end it explicitly
                    Environment.Exit(code);
                    return code;
                case "train":
                    return Train(model, options);
                case "video":
                    return Video(model, LinearPolicy.Load(policyPath), Argument(args, "video"), options.Double("seconds", 30), options.Int("seed", 0));
                case "frames":
                    return Frames(model, LinearPolicy.Load(policyPath), Argument(args, "frames"), options.Int("steps", 200), options.Int("every", 20), options.Int("seed", 0));
                case "score":
                    return Score(model, LinearPolicy.Load(policyPath), options.Text("seeds", "0-4"));
                default:
                    Console.Error.WriteLine($"unknown command '{command}': play, train, video, frames or score");
                    return 2;
            }
        } catch (ArgumentException e) {
            Console.Error.WriteLine(e.Message);
            return 2;
        }
    }

    /// <summary>The positional argument after a command (a file or folder).</summary>
    /// <param name="args">The arguments.</param>
    /// <param name="command">The command.</param>
    /// <returns>The argument.</returns>
    /// <exception cref="ArgumentException">It is missing.</exception>
    private static string Argument(string[] args, string command) =>
        args.Length > 1 && !args[1].StartsWith("--", StringComparison.Ordinal) ? args[1] : throw new ArgumentException($"{command} needs a path: {command} PATH");

    // ================================================================================== play

    /// <summary>
    ///     The window: the simulation runs in real time (or faster or slower with +/-); each env step reads the autopilot or
    ///     the keys, holds that action for five physics steps, and the view redraws up to 60 times a second.
    /// </summary>
    /// <param name="model">The model.</param>
    /// <param name="policy">The autopilot, or null when there is none (the keyboard drives).</param>
    /// <param name="seed">The first episode's Gymnasium seed; each new episode takes the next.</param>
    /// <param name="manual">Start with the keyboard driving.</param>
    /// <returns>0.</returns>
    /// <exception cref="InvalidOperationException">The viewer reports no keyboard input.</exception>
    private static int Play(MjModel model, LinearPolicy? policy, int seed, bool manual) {
        var view = new SideView();
        var game = new CheetahGame(model);
        game.Reset(seed);
        IEnvViewer viewer = AvaloniaEnvViewer.Factory(view.Width, view.Height, Title).GetAwaiter().GetResult();
        var input = viewer as IInteractiveEnvViewer ?? throw new InvalidOperationException("the viewer reports no keys");
        input.BeginInput();
        Console.WriteLine("Window open. Tab: autopilot on/off; keys Q/A W/S E/D drive the back leg, U/J I/K O/L the front; " +
                          "R: new episode; Space: pause; +/-: speed; Esc: quit.");
        var held = new HashSet<int>();
        bool autopilot = policy != null && !manual, paused = false, quit = false;
        double speed = 1, lastScore = double.NaN, bannerUntil = 0, lastFrame = -1;
        string? banner = null;
        int nextSeed = seed + 1;
        var clock = Stopwatch.StartNew();
        double due = 0;
        Span<float> action = stackalloc float[CheetahGame.ActionSize];

        void Show() {
            double now = clock.Elapsed.TotalSeconds;
            Hud hud = new(autopilot ? "AUTOPILOT" : "YOU", paused, speed, lastScore, now < bannerUntil ? banner : null);
            Surface frame = view.Render(game, hud);
            using Image<Rgb24> image = Image.LoadPixelData<Rgb24>(frame.ToRgbBytes(), frame.Width, frame.Height);
            viewer.Render(image);
            lastFrame = now;
        }

        while (!quit) {
            foreach (ViewerInput e in input.TakeInput()) {
                switch (e.Kind) {
                    case ViewerInputKind.Close:
                        quit = true;
                        break;
                    case ViewerInputKind.KeyUp:
                        held.Remove(e.Key);
                        break;
                    case ViewerInputKind.KeyDown:
                        held.Add(e.Key);
                        switch (e.Key) {
                            case KeyCode.Escape:
                                quit = true;
                                break;
                            case KeyCode.Tab when policy != null:
                                autopilot = !autopilot;
                                break;
                            case KeyCode.Space:
                                paused = !paused;
                                break;
                            case KeyCode.R:
                                game.Reset(nextSeed++);
                                break;
                            case KeyCode.EqualsSign or KeyCode.Plus or KeyCode.KeypadPlus:
                                speed = Math.Min(4, speed * 2);
                                break;
                            case KeyCode.Minus or KeyCode.KeypadMinus:
                                speed = Math.Max(0.125, speed / 2);
                                break;
                        }
                        break;
                }
            }
            if (quit)
                break;
            if (paused) {
                Show();
                Thread.Sleep(30);
                due = clock.Elapsed.TotalSeconds;
                continue;
            }
            if (autopilot && policy != null)
                policy.Act(game.Observation(), action);
            else
                KeysToAction(held, action);
            game.BeginStep(action);
            for (int k = 0; k < CheetahGame.FrameSkip; k++) {
                game.PhysicsStep();
                due += model.timestep / speed;
                double now = clock.Elapsed.TotalSeconds;
                if (now - due > 0.25) {
                    // far behind (the window was dragged, the machine is busy): skip the debt instead of racing to catch up
                    due = now;
                }
                if (now - lastFrame >= 1.0 / 60)
                    Show();
                while (clock.Elapsed.TotalSeconds < due)
                    Thread.Sleep(1);
            }
            game.EndStep();
            if (game.StepIndex >= CheetahGame.EpisodeSteps) {
                lastScore = game.EpisodeReturn;
                banner = $"EPISODE SCORE {lastScore.ToString("F0", CultureInfo.InvariantCulture)}";
                bannerUntil = clock.Elapsed.TotalSeconds + 3;
                game.Reset(nextSeed++);
            }
        }
        viewer.CloseEnvironment();
        return 0;
    }

    /// <summary>
    ///     The keyboard's action: per motor, +1 while its upper key is held, -1 while its lower key is, 0 for neither or both.
    ///     Q/A, W/S, E/D drive the back thigh, shin and foot; U/J, I/K, O/L the front ones.
    /// </summary>
    /// <param name="held">The SDL keycodes held down.</param>
    /// <param name="action">Receives the action.</param>
    private static void KeysToAction(HashSet<int> held, Span<float> action) {
        ReadOnlySpan<int> plus = [KeyCode.Q, KeyCode.W, KeyCode.E, KeyCode.U, KeyCode.I, KeyCode.O];
        ReadOnlySpan<int> minus = [KeyCode.A, KeyCode.S, KeyCode.D, KeyCode.J, KeyCode.K, KeyCode.L];
        for (int i = 0; i < CheetahGame.ActionSize; i++)
            action[i] = (held.Contains(plus[i]) ? 1f : 0f) - (held.Contains(minus[i]) ? 1f : 0f);
    }

    // ================================================================================== train, record, score

    /// <summary>Trains the autopilot with ARS and writes it as JSON.</summary>
    /// <param name="model">The model.</param>
    /// <param name="options">--iterations, --threads, --seed, --out.</param>
    /// <returns>0.</returns>
    private static int Train(MjModel model, Options options) {
        var settings = new Ars.Settings(Iterations: options.Int("iterations", 600), Threads: options.Int("threads", Math.Max(1, Environment.ProcessorCount / 2)),
                                        Seed: options.Int("seed", 1));
        string output = options.Text("out", "policy.json");
        Console.WriteLine($"Training with {settings}");
        LinearPolicy best = Ars.Train(model, settings, Console.Out);
        best.Save(output);
        Console.WriteLine($"best evaluation {best.Info["evaluation_mean"]} at iteration {best.Info["iteration"]}; written to {Path.GetFullPath(output)}");
        return 0;
    }

    /// <summary>
    ///     Records the autopilot to an H.264 video through ffmpeg: 50 frames a second (one every two physics steps), real
    ///     time, episodes chained with their score shown for two seconds.
    /// </summary>
    /// <param name="model">The model.</param>
    /// <param name="policy">The autopilot.</param>
    /// <param name="path">The video file, overwritten.</param>
    /// <param name="seconds">How many simulated seconds to record.</param>
    /// <param name="seed">The first episode's seed.</param>
    /// <returns>0, or ffmpeg's exit code when it failed.</returns>
    /// <exception cref="System.ComponentModel.Win32Exception">ffmpeg is not on PATH.</exception>
    private static int Video(MjModel model, LinearPolicy policy, string path, double seconds, int seed) {
        var view = new SideView();
        var game = new CheetahGame(model);
        game.Reset(seed);
        const int fps = 50;
        int physicsPerFrame = (int)Math.Round(1.0 / (fps * model.timestep));
        var start = new ProcessStartInfo("ffmpeg") { RedirectStandardInput = true, UseShellExecute = false };
        foreach (string a in new[] { "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", $"{view.Width}x{view.Height}",
                                     "-r", fps.ToString(CultureInfo.InvariantCulture), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                                     "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path })
            start.ArgumentList.Add(a);
        using Process ffmpeg = Process.Start(start)!;
        Stream sink = ffmpeg.StandardInput.BaseStream;
        Span<float> action = stackalloc float[CheetahGame.ActionSize];
        double lastScore = double.NaN, bannerUntil = -1;
        string? banner = null;
        long physics = 0, total = (long)Math.Round(seconds / model.timestep);
        int episodeSeed = seed;
        while (physics < total) {
            policy.Act(game.Observation(), action);
            game.BeginStep(action);
            for (int k = 0; k < CheetahGame.FrameSkip; k++) {
                game.PhysicsStep();
                if (++physics % physicsPerFrame == 0) {
                    double t = physics * model.timestep;
                    Surface frame = view.Render(game, new Hud("AUTOPILOT", false, 1, lastScore, t < bannerUntil ? banner : null));
                    sink.Write(frame.ToRgbBytes());
                }
            }
            game.EndStep();
            if (game.StepIndex >= CheetahGame.EpisodeSteps) {
                lastScore = game.EpisodeReturn;
                banner = $"EPISODE SCORE {lastScore.ToString("F0", CultureInfo.InvariantCulture)}";
                bannerUntil = physics * model.timestep + 2;
                game.Reset(++episodeSeed);
            }
        }
        sink.Close();
        ffmpeg.WaitForExit();
        Console.WriteLine(ffmpeg.ExitCode == 0 ? $"{seconds} s of autopilot written to {Path.GetFullPath(path)}" : $"ffmpeg failed ({ffmpeg.ExitCode})");
        return ffmpeg.ExitCode;
    }

    /// <summary>Writes PNG frames of the autopilot's first env steps, every few steps, for a quick look without a window.</summary>
    /// <param name="model">The model.</param>
    /// <param name="policy">The autopilot.</param>
    /// <param name="folder">The output folder (created).</param>
    /// <param name="steps">Env steps to play.</param>
    /// <param name="every">Env steps between frames.</param>
    /// <param name="seed">The episode's seed.</param>
    /// <returns>0.</returns>
    private static int Frames(MjModel model, LinearPolicy policy, string folder, int steps, int every, int seed) {
        Directory.CreateDirectory(folder);
        var view = new SideView();
        var game = new CheetahGame(model);
        game.Reset(seed);
        Span<float> action = stackalloc float[CheetahGame.ActionSize];
        for (int t = 0; t <= steps; t++) {
            if (t % every == 0) {
                Surface frame = view.Render(game, new Hud("AUTOPILOT", false, 1, double.NaN, null));
                using Image<Rgb24> image = Image.LoadPixelData<Rgb24>(frame.ToRgbBytes(), frame.Width, frame.Height);
                image.SaveAsPng(Path.Combine(folder, $"step{t:D4}.png"));
            }
            policy.Act(game.Observation(), action);
            game.Step(action);
        }
        Console.WriteLine($"frames written to {Path.GetFullPath(folder)}");
        return 0;
    }

    /// <summary>
    ///     Plays full episodes from Gymnasium's seeded resets and prints each return with its exact bits, to compare with
    ///     replay_in_gymnasium.py's run of the same policy in Gymnasium on MuJoCo.
    /// </summary>
    /// <param name="model">The model.</param>
    /// <param name="policy">The autopilot.</param>
    /// <param name="seeds">A range "A-B" or a single seed.</param>
    /// <returns>0.</returns>
    /// <exception cref="ArgumentException">The range is malformed.</exception>
    private static int Score(MjModel model, LinearPolicy policy, string seeds) {
        string[] parts = seeds.Split('-');
        if (!int.TryParse(parts[0], out int first) || !int.TryParse(parts[^1], out int last) || last < first)
            throw new ArgumentException($"--seeds wants A-B or A, got '{seeds}'");
        var results = new double[last - first + 1];
        Parallel.For(first, last + 1, s => results[s - first] = Ars.Play(model, policy, s));
        for (int s = first; s <= last; s++) {
            double r = results[s - first];
            Console.WriteLine($"seed {s}: return {r:R} (0x{BitConverter.DoubleToInt64Bits(r):X16})");
        }
        return 0;
    }

    /// <summary>The <c>--name value</c> options after the command.</summary>
    private sealed class Options {
        private readonly Dictionary<string, string?> values = new();

        /// <summary>Parses <c>--name value</c> pairs and bare <c>--flag</c>s.</summary>
        /// <param name="args">The option arguments.</param>
        public Options(string[] args) {
            for (int i = 0; i < args.Length; i++) {
                if (!args[i].StartsWith("--", StringComparison.Ordinal))
                    continue;
                bool hasValue = i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal);
                values[args[i][2..]] = hasValue ? args[++i] : null;
            }
        }

        /// <summary>Whether a flag was given.</summary>
        /// <param name="name">The flag without dashes.</param>
        /// <returns>Whether it is present.</returns>
        public bool Flag(string name) => values.ContainsKey(name);

        /// <summary>A text option.</summary>
        /// <param name="name">The option without dashes.</param>
        /// <param name="fallback">The value when it is absent.</param>
        /// <returns>The value.</returns>
        public string Text(string name, string fallback) => values.TryGetValue(name, out string? v) && v != null ? v : fallback;

        /// <summary>An integer option.</summary>
        /// <param name="name">The option without dashes.</param>
        /// <param name="fallback">The value when it is absent.</param>
        /// <returns>The value.</returns>
        /// <exception cref="ArgumentException">The value is not an integer.</exception>
        public int Int(string name, int fallback) =>
            values.TryGetValue(name, out string? v) && v != null
                ? int.TryParse(v, NumberStyles.Integer, CultureInfo.InvariantCulture, out int n) ? n : throw new ArgumentException($"--{name} wants an integer, got '{v}'")
                : fallback;

        /// <summary>A number option.</summary>
        /// <param name="name">The option without dashes.</param>
        /// <param name="fallback">The value when it is absent.</param>
        /// <returns>The value.</returns>
        /// <exception cref="ArgumentException">The value is not a number.</exception>
        public double Double(string name, double fallback) =>
            values.TryGetValue(name, out string? v) && v != null
                ? double.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out double x) ? x : throw new ArgumentException($"--{name} wants a number, got '{v}'")
                : fallback;
    }
}
