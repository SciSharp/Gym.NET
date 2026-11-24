#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using Gym.Envs;
using Gym.Observations;
using Gym.Rendering;
using Gym.Rendering.Avalonia;
using Gym.Utils;
using Gym.Wrappers;
using NumSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Gym.Examples.MountainCar {
    /// <summary>
    ///     Drives MountainCar-v0's car over the hill by pumping energy: it accelerates whichever way the car already
    ///     moves. <c>dotnet run --project examples/MountainCar -c Release</c> plays one seeded episode headless and
    ///     prints its steps and return; <c>-- --render</c> plays it in a window, <c>-- --play</c> hands the car to
    ///     the arrow keys, and <c>-- --record &lt;folder&gt;</c> records the episode for MountainCar's page on the docs
    ///     site (DR-112).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The engine is too weak to climb the right hill directly. Accelerating in the direction of motion adds energy
    ///     on every pass, as pumping a swing does, until the car carries over the top: 122 steps from seed 4.
    ///     </para>
    ///     <para>
    ///     The default seed, 4, is the episode the front-page showreel plays (<c>tools/showreel</c>, DR-74).
    ///     </para>
    /// </remarks>
    internal static class MountainCarProgram {
        /// <summary>The environment this example plays, by its registered id.</summary>
        private const string Id = "MountainCar-v0";

        /// <summary>The first reset seed when <c>--seed</c> is not given: the episode the showreel plays.</summary>
        private const long DefaultSeed = 4;

        /// <summary>
        ///     The moment How it works annotates on MountainCar's page (DR-114): the frame after this many steps of the
        ///     seed-4 episode, the top of the car's one back-swing, high on the left slope, where its velocity has just
        ///     turned positive and the hill starts pulling it back down toward the flag.
        /// </summary>
        private const int FigureStep = 80;

        /// <summary>
        ///     Plays the episodes the arguments ask for with the agent, or hands the env to the keyboard, or records the
        ///     docs page's media.
        /// </summary>
        /// <param name="args">
        ///     <c>--seed N</c> starts from reset seed N; <c>--episodes N</c> plays N episodes on consecutive seeds and
        ///     prints their mean; <c>--render</c> shows them in a window; <c>--play</c> lets you play the env from the
        ///     keyboard instead, until Escape or the window's close button; <c>--record &lt;folder&gt;</c> records the
        ///     page's episode, seed 4 whatever <c>--seed</c> says (<see cref="Record"/>). Other arguments are ignored.
        /// </param>
        /// <returns>
        ///     0 when the episodes were played, the keyboard game ended or the recording was written, 2 when an option's
        ///     value is not a whole number or <c>--record</c> names no folder.
        /// </returns>
        /// <exception cref="Gym.Exceptions.DependencyNotInstalled">
        ///     Not thrown in practice: <c>--render</c> and <c>--play</c> set Avalonia's viewer before the env first
        ///     draws.
        /// </exception>
        /// <exception cref="InvalidOperationException">
        ///     <c>--record</c> could not run ffmpeg, which <see cref="FfmpegVideoWriter"/> needs on PATH.
        /// </exception>
        private static int Main(string[] args) {
            if (!TryNumber(args, "--seed", DefaultSeed, out long seed) || !TryNumber(args, "--episodes", 1, out long episodes))
                return 2;
            int record = Array.IndexOf(args, "--record");
            if (record >= 0) {
                // The folder is required: the page's media live in the docs tree, and a default would write them where no
                // page reads them.
                if (record + 1 >= args.Length || args[record + 1].StartsWith("--", StringComparison.Ordinal)) {
                    Console.Error.WriteLine("--record needs a folder, e.g. --record docs/environments/classic_control/media");
                    return 2;
                }
                (int recordedSteps, double recordedReturn) = Record(args[record + 1]);
                Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                    $"{Id}, seed {DefaultSeed}: {recordedSteps} steps, return {recordedReturn:F1}, recorded into {args[record + 1]}"));
                return 0;
            }
            bool render = args.Contains("--render");
            if (args.Contains("--play")) {
                // play draws the env's frames into a window of the app-wide viewer and reads the keys from it.
                Gymnasium.HumanViewerFactory = AvaloniaEnvViewer.Factory;
                PlayWithKeyboard(seed);
                return 0;
            }
            // Human mode draws inside every reset and step into the app-wide viewer (DR-20); without one it raises
            // DependencyNotInstalled at the first frame, as Gymnasium does without pygame.
            if (render)
                Gymnasium.HumanViewerFactory = AvaloniaEnvViewer.Factory;
            double sum = 0;
            for (long episode = 0; episode < episodes; episode++) {
                (int steps, double total, string ending) = PlayEpisode(seed + episode, render);
                sum += total;
                Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                    $"{Id}, seed {seed + episode}: {steps} step{(steps == 1 ? "" : "s")}, return {total:F1}, {ending}"));
            }
            if (episodes > 1)
                Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"Mean return over {episodes} episodes: {sum / episodes:F1}"));
            return 0;
        }

        /// <summary>Plays one episode with the agent and reports how it went.</summary>
        /// <param name="seed">The reset seed, which draws the starting state.</param>
        /// <param name="render">
        ///     Whether to show the episode in a window, in human mode, paced at the env's frame rate.
        /// </param>
        /// <returns>
        ///     The steps taken, the return (float64, as every Gym.NET reward), and how the episode ended.
        /// </returns>
        /// <remarks>
        ///     The arrays a reset or a step returns belong to the caller (D19). The method is <c>[NDScoped]</c> (DR-80):
        ///     NumSharp.Build releases every array the episode creates (the results, the actions) when it returns. The env
        ///     is built and disposed inside it, the guides' rule for an env in a scoped method.
        /// </remarks>
        [NDScoped]
        private static (int Steps, double Return, string Ending) PlayEpisode(long seed, bool render) {
            // make's stack: TimeLimit over OrderEnforcing over PassiveEnvChecker over the env. The render mode is a
            // constructor argument, fixed for the env's life, as in Gymnasium.
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>(Id, render ? new { render_mode = "human" } : null);
            (NDArray observation, Info info) = env.Reset(seed: seed);
            int steps = 0;
            double total = 0;
            while (true) {
                StepResult<NDArray> step = env.Step(Act(observation));
                observation = step.Observation;
                steps++;
                total += step.Reward;
                if (step.Terminated || step.Truncated)
                    return (steps, total, step.Terminated ? "reached the flag" : "truncated by the time limit");
            }
        }

        /// <summary>
        ///     Reads an observation's values as float64, whatever its dtype (classic control's are float32).
        /// </summary>
        /// <param name="observation">The observation.</param>
        /// <returns>Its values in C order.</returns>
        private static double[] Values(NDArray observation) {
            using NDArray values = observation.astype(NPTypeCode.Double);
            return values.ToArray<double>();
        }

        /// <summary>The agent: accelerate the way the car is already moving, rightward from rest.</summary>
        /// <param name="observation">The observation <c>[position, velocity]</c>.</param>
        /// <returns>
        ///     Accelerate right (2) when the velocity is not negative, else left (0), as a 0-d int64 array.
        /// </returns>
        private static NDArray Act(NDArray observation) => NDArray.Scalar(Push(Values(observation)[1]));

        /// <summary>
        ///     The agent's rule as a number: right (2) when <paramref name="velocity"/> is not negative, else left (0).
        ///     <see cref="Act"/> and <see cref="Record"/> both decide through it, so the recorded episode is the one the
        ///     example plays.
        /// </summary>
        /// <param name="velocity">The observation's velocity, its element 1.</param>
        /// <returns>The action, 0 or 2, as MountainCar's <c>Discrete(3)</c> numbers them.</returns>
        private static long Push(double velocity) => velocity >= 0 ? 2L : 0L;

        /// <summary>
        ///     Records the seed-4 episode for MountainCar's page on the docs site (DR-112): the whole episode as
        ///     <c>MountainCar-v0.episode.mp4</c>, one frame per step at the env's 30 frames a second, through Gymnasium's
        ///     <see cref="RecordVideo{TObs, TAct}"/> and <see cref="FfmpegVideoWriter"/>; every step as
        ///     <c>MountainCar-v0.episode.json</c>, for the page's anatomy chart; and How it works' frame as
        ///     <c>MountainCar-v0.figure.png</c> with <c>MountainCar-v0.figure.json</c>, the pixels its labels point at.
        /// </summary>
        /// <param name="folder">The output folder, <c>docs/environments/classic_control/media</c> for the site; created if missing.</param>
        /// <returns>The steps taken and the return.</returns>
        /// <exception cref="InvalidOperationException">ffmpeg could not be run, or wrote no video.</exception>
        /// <exception cref="IOException">A file cannot be written.</exception>
        /// <remarks>
        ///     <para>
        ///     Each JSON row is the observation the agent acted on, then its action and the reward:
        ///     <c>step, position, velocity, action, reward</c>. The video's frame <c>t</c> shows the state of row <c>t</c>
        ///     (frame 0 is the reset), and its last frame the state after the last step, which the JSON keeps as
        ///     <c>final</c>.
        ///     </para>
        ///     <para>
        ///     RecordVideo names its file <c>MountainCar-v0-episode-0.mp4</c> and writes it when the env closes; it records
        ///     into a scratch folder inside <paramref name="folder"/> and the file is moved to the page's name, so a
        ///     recording that fails halfway leaves the published video as it was.
        ///     </para>
        /// </remarks>
        [NDScoped]   // releases every array the episode creates when it returns (DR-80); the envs are disposed inside
        private static (int Steps, double Return) Record(string folder) {
            Directory.CreateDirectory(folder);
            string scratch = Path.Combine(folder, ".recording");
            var rows = new List<double[]>();
            double total = 0;
            double[] final;
            using (Env<NDArray, NDArray> made = Gymnasium.Make<NDArray, NDArray>(Id, RenderMode.RgbArray))
            using (var env = new RecordVideo<NDArray, NDArray>(made, scratch, episodeTrigger: _ => true, namePrefix: Id,
                       videoWriter: new FfmpegVideoWriter())) {
                (NDArray observation, Info info) = env.Reset(seed: DefaultSeed);
                for (int t = 0; ; t++) {
                    double[] s = Values(observation);
                    long action = Push(s[1]);
                    // The frame now on screen shows the state the agent is about to act on (row t).
                    if (t == FigureStep)
                        SaveFigure(env, folder, s);
                    StepResult<NDArray> step = env.Step(NDArray.Scalar(action));
                    total += step.Reward;
                    rows.Add(new double[] { t, s[0], s[1], action, step.Reward });
                    observation = step.Observation;
                    if (step.Terminated || step.Truncated)
                        break;
                }
                final = Values(observation);
                // RecordVideo writes the episode's video when it stops recording, which closing the env does.
                env.Close();
            }
            string recorded = Path.Combine(scratch, $"{Id}-episode-0.mp4");
            if (!File.Exists(recorded))
                throw new InvalidOperationException($"RecordVideo wrote no {recorded}; is ffmpeg on PATH?");
            File.Move(recorded, Path.Combine(folder, $"{Id}.episode.mp4"), overwrite: true);
            Directory.Delete(scratch, recursive: true);
            File.WriteAllText(Path.Combine(folder, $"{Id}.episode.json"), JsonSerializer.Serialize(new {
                id = Id, seed = DefaultSeed, steps = rows.Count, @return = total, fps = 30,
                agent = "the energy-pumping rule (examples/MountainCar)",
                columns = new[] { "step", "position", "velocity", "action", "reward" },
                // Five decimals keep the chart exact to the pixel at a fraction of the file size; a velocity keeps three
                // significant digits at its smallest, as the readout shows it.
                rows = rows.Select(r => r.Select(v => Math.Round(v, 5)).ToArray()).ToArray(),
                final = final.Select(v => Math.Round(v, 5)).ToArray(),
            }) + "\n");
            return (rows.Count, total);
        }

        /// <summary>
        ///     Saves How it works' frame (the frame the env shows now) as <c>MountainCar-v0.figure.png</c>, and where its
        ///     labels point as <c>MountainCar-v0.figure.json</c>: the car's centre and the hill under it, the slope's
        ///     direction there, the flag, the starting range, the valley's floor and the left wall, in the frame's pixels.
        /// </summary>
        /// <param name="env">The recording env, in <c>rgb_array</c> mode; rendering it adds nothing to the video.</param>
        /// <param name="folder">The output folder.</param>
        /// <param name="s">The state the frame shows: position and velocity.</param>
        /// <exception cref="IOException">A file cannot be written.</exception>
        /// <remarks>
        ///     The pixels follow MountainCar's renderer: the 600 px frame spans the track's 1.8 units from -1.2 to 0.6
        ///     (333.3 px a unit); the hill's height <c>0.45 sin(3x) + 0.55</c> is measured up from the frame's bottom; the
        ///     car's bottom centre sits 10 px (the clearance) above the hill and the car is rotated by <c>cos(3x)</c>
        ///     radians, so its centre is 10 px further along the rotated up direction; the flag's pole stands 50 px tall
        ///     at column <c>int(1.7 × 333.3)</c> from the hill's height there, rounded down as <c>int()</c> does. Reference:
        ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/mountain_car.py#L175-L263">gymnasium/envs/classic_control/mountain_car.py</see>.
        /// </remarks>
        private static void SaveFigure(Env<NDArray, NDArray> env, string folder, double[] s) {
            using (RenderResult render = env.Render()) {
                NDArray frame = (NDArray)render;
                int height = (int)frame.shape[0], width = (int)frame.shape[1];
                using var image = Image.LoadPixelData<Rgb24>(frame.ToArray<byte>(), width, height);
                image.SaveAsPng(Path.Combine(folder, $"{Id}.figure.png"));
            }
            // The renderer draws in pygame's upward coordinates and flips the frame, so a height h above the bottom edge is
            // the frame's row 400 - h. The scale is the renderer's own expression: 0.6 - (-1.2) is 1.7999999999999998 in
            // float64, so 600 / 1.8 would be a different double, and int() of a column can land one pixel off with it
            // (the continuous flag at 0.45 is column 550, not 549).
            const double Scale = 600 / (0.6 - -1.2), Clearance = 10, HalfCar = 20 / 2.0, FrameHeight = 400;
            static double Column(double x) => (x + 1.2) * Scale;
            static double Row(double x) => FrameHeight - (Math.Sin(3 * x) * 0.45 + 0.55) * Scale;
            double position = s[0], angle = Math.Cos(3 * position);
            // pygame's rotate_rad turns (0, 10) into (-10 sin a, 10 cos a), upward; the flip negates the vertical part.
            double carX = Column(position) - HalfCar * Math.Sin(angle), carY = Row(position) - Clearance - HalfCar * Math.Cos(angle);
            long flagX = (long)Math.Floor(Column(0.5)), flagBottom = (long)Math.Floor((Math.Sin(3 * 0.5) * 0.45 + 0.55) * Scale);
            double valley = -Math.PI / 6;
            static double[] Point(double column, double row) => new[] { Math.Round(column, 2), Math.Round(row, 2) };
            File.WriteAllText(Path.Combine(folder, $"{Id}.figure.json"), JsonSerializer.Serialize(new {
                id = Id, step = FigureStep, width = 600, height = 400, pixelsPerUnit = Math.Round(Scale, 4),
                state = new { position = Math.Round(position, 5), velocity = Math.Round(s[1], 5) },
                angle = Math.Round(angle, 5),
                car = Point(carX, carY),
                ground = Point(Column(position), Row(position)),
                // The car's floor, a unit vector in the frame's pixels pointing toward the flag: the slope as drawn.
                slope = Point(Math.Cos(angle), -Math.Sin(angle)),
                flag = new { top = Point(flagX, FrameHeight - flagBottom - 50), bottom = Point(flagX, FrameHeight - flagBottom) },
                start = new[] { Point(Column(-0.6), Row(-0.6)), Point(Column(-0.4), Row(-0.4)) },
                valley = Point(Column(valley), Row(valley)),
                wall = Point(Column(-1.2), Row(-1.2)),
            }, new JsonSerializerOptions { WriteIndented = true }) + "\n");
        }

        /// <summary>
        ///     The keys for <c>--play</c>: the left and right arrows accelerate that way. MountainCar's own
        ///     <c>get_keys_to_action</c> is not used: it names pygame 1's arrow codes, which no pygame 2 window sends,
        ///     and maps the empty combination, which Gymnasium's play rejects.
        /// </summary>
        /// <returns>The map from key codes to MountainCar's actions.</returns>
        private static Dictionary<object, object?> Keys() => new() { [KeyCode.Left] = 0L, [KeyCode.Right] = 2L };

        /// <summary>
        ///     Hands the env to the keyboard, Gymnasium's play, until Escape or the window's close button.
        /// </summary>
        /// <param name="seed">
        ///     The seed of every reset: each new game starts as the last one did, as Gymnasium's play does.
        /// </param>
        /// <remarks>
        ///     With no key held the car coasts: the no-op is action 1, no acceleration.
        /// </remarks>
        private static void PlayWithKeyboard(long seed) {
            // play reads the env's rgb_array frames and draws them into its own window.
            using IEnv env = Gymnasium.Make(Id, new { render_mode = "rgb_array" });
            Dictionary<object, object?> keys = Keys();
            object noop = 1L;
            Play.Run(env, keysToAction: keys, noop: noop, seed: seed);
        }

        /// <summary>Reads a whole-number option's value, or the fallback when the option is absent.</summary>
        /// <param name="args">The command line.</param>
        /// <param name="name">The option, <c>--seed</c> or <c>--episodes</c>.</param>
        /// <param name="fallback">The value when the option is absent.</param>
        /// <param name="value">The value read, or <paramref name="fallback"/>.</param>
        /// <returns>
        ///     False, after printing why, when the option has no value or one that is not a non-negative whole number.
        /// </returns>
        private static bool TryNumber(string[] args, string name, long fallback, out long value) {
            value = fallback;
            int at = Array.IndexOf(args, name);
            if (at < 0)
                return true;
            // Gymnasium's seeds are non-negative, and so is a count of episodes: NumberStyles.None takes digits only.
            if (at + 1 < args.Length && long.TryParse(args[at + 1], NumberStyles.None, CultureInfo.InvariantCulture, out value))
                return true;
            Console.Error.WriteLine($"{name} needs a non-negative whole number");
            return false;
        }
    }
}
