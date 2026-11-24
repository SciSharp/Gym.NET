#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using Gym.Envs;
using Gym.Observations;
using Gym.Rendering.Avalonia;
using Gym.Utils;
using Gym.Wrappers;
using NumSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Gym.Examples.Pendulum {
    /// <summary>
    ///     Swings Pendulum-v1 up and holds it upright: energy pumping, then a PD controller near the top. <c>dotnet run
    ///     --project examples/Pendulum -c Release</c> plays one seeded episode headless and prints its steps and
    ///     return; <c>-- --render</c> plays it in a window, <c>-- --play</c> hands the motor to the A and D keys, and
    ///     <c>-- --record &lt;folder&gt;</c> records the episode for Pendulum's page on the docs site (DR-112).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The motor is too weak to lift the pendulum straight up. Far from the top, full torque along the swing adds
    ///     energy until the pendulum has the energy of the upright rest (<c>w^2/2 + 15 cos(theta)</c> for Gymnasium's g
    ///     = 10, m = l = 1); within about 32 degrees of the top a PD controller catches it and holds it there.
    ///     </para>
    ///     <para>
    ///     The default seed, 5, is the episode the front-page showreel plays (<c>tools/showreel</c>, DR-74).
    ///     </para>
    /// </remarks>
    internal static class PendulumProgram {
        /// <summary>The environment this example plays, by its registered id.</summary>
        private const string Id = "Pendulum-v1";

        /// <summary>The first reset seed when <c>--seed</c> is not given: the episode the showreel plays.</summary>
        private const long DefaultSeed = 5;

        /// <summary>
        ///     The moment How it works annotates on Pendulum's page (DR-114): the frame after this many steps of the
        ///     seed-5 episode, the rod swinging up the right side below the horizontal at nearly its top speed, the full
        ///     counterclockwise torque of the step before drawn as the env's arrow.
        /// </summary>
        private const int FigureStep = 10;

        /// <summary>
        ///     The controller's three regimes, as the page's anatomy chart colours them; the numbers are the trace's
        ///     <c>phase</c> column.
        /// </summary>
        private enum Phase {
            /// <summary>Full torque along the swing: the energy is below the upright rest's.</summary>
            Pump = 0,

            /// <summary>A gentle torque against the swing: the energy has passed the upright rest's.</summary>
            Brake = 1,

            /// <summary>The PD controller: within about 32 degrees of the top.</summary>
            Hold = 2,
        }

        /// <summary>
        ///     Plays the episodes the arguments ask for with the agent, or hands the env to the keyboard, or records the
        ///     docs page's media.
        /// </summary>
        /// <param name="args">
        ///     <c>--seed N</c> starts from reset seed N; <c>--episodes N</c> plays N episodes on consecutive seeds and
        ///     prints their mean; <c>--render</c> shows them in a window; <c>--play</c> lets you play the env from the
        ///     keyboard instead, until Escape or the window's close button; <c>--record &lt;folder&gt;</c> records the
        ///     page's episode, seed 5 whatever <c>--seed</c> says (<see cref="Record"/>). Other arguments are ignored.
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
                    return (steps, total, step.Terminated ? "terminated" : "truncated by the time limit");
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

        /// <summary>The agent: pump energy far from the top, catch and hold the pendulum near it.</summary>
        /// <param name="observation">The observation <c>[cos(theta), sin(theta), theta_dot]</c>.</param>
        /// <returns>A float32 torque in [-2, 2], as a one-element array.</returns>
        private static NDArray Act(NDArray observation) => np.array(new[] { Controller(Values(observation)).Torque });

        /// <summary>
        ///     The agent's decision as numbers: the torque and the regime that chose it. <see cref="Act"/> and
        ///     <see cref="Record"/> both decide through it, so the recorded episode is the one the example plays.
        /// </summary>
        /// <param name="s">The observation's values <c>[cos(theta), sin(theta), theta_dot]</c>, as float64.</param>
        /// <returns>The float32 torque in [-2, 2], and the <see cref="Phase"/> that produced it.</returns>
        /// <remarks>
        ///     The energy is <c>w^2/2 + 15 cos(theta)</c>: with Gymnasium's g = 10 and m = l = 1 the dynamics are
        ///     <c>theta'' = 15 sin(theta) + 3u</c>, so the energy changes by <c>3 u w</c> a second, and torque along the swing
        ///     (u and w of one sign) adds it. The upright rest has 15.
        /// </remarks>
        private static (float Torque, Phase Phase) Controller(double[] s) {
            double theta = Math.Atan2(s[1], s[0]), w = s[2];
            double energy = 0.5 * w * w + 15.0 * Math.Cos(theta);
            // Near the top (cos > 0.85) a PD controller holds it; below, full torque along the swing until the energy
            // reaches the upright rest's 15, then a gentle brake against overshooting.
            (double u, Phase phase) = Math.Cos(theta) > 0.85 ? (-(12.0 * theta + 2.5 * w), Phase.Hold)
                : energy < 15.0 ? (2.0 * (w >= 0 ? 1 : -1), Phase.Pump)
                : (-0.5 * Math.Sign(w), Phase.Brake);
            return ((float)Math.Clamp(u, -2.0, 2.0), phase);
        }

        /// <summary>
        ///     Records the seed-5 episode for Pendulum's page on the docs site (DR-112): the whole episode as
        ///     <c>Pendulum-v1.episode.mp4</c>, one frame per step at the env's 30 frames a second, through Gymnasium's
        ///     <see cref="RecordVideo{TObs, TAct}"/> and <see cref="FfmpegVideoWriter"/>; every step as
        ///     <c>Pendulum-v1.episode.json</c>, for the page's anatomy chart; and How it works' frame as
        ///     <c>Pendulum-v1.figure.png</c> with <c>Pendulum-v1.figure.json</c>, the pixels its labels point at.
        /// </summary>
        /// <param name="folder">The output folder, <c>docs/environments/classic_control/media</c> for the site; created if missing.</param>
        /// <returns>The steps taken and the return.</returns>
        /// <exception cref="InvalidOperationException">ffmpeg could not be run, or wrote no video.</exception>
        /// <exception cref="IOException">A file cannot be written.</exception>
        /// <remarks>
        ///     <para>
        ///     Each JSON row is the observation the agent acted on, as the angle from upright
        ///     (<c>atan2(sin, cos)</c> of the observation, in radians) and the angular velocity, then the torque, the
        ///     controller's regime (<see cref="Phase"/>: 0 pump, 1 brake, 2 hold) and the reward:
        ///     <c>step, theta, theta_dot, torque, phase, reward</c>. The video's frame <c>t</c> shows the state of row
        ///     <c>t</c> (frame 0 is the reset) with the torque of row <c>t - 1</c> as its arrow, and its last frame the
        ///     state after the last step, which the JSON keeps as <c>final</c>.
        ///     </para>
        ///     <para>
        ///     RecordVideo names its file <c>Pendulum-v1-episode-0.mp4</c> and writes it when the env closes; it records
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
                float shown = 0;
                for (int t = 0; ; t++) {
                    double[] s = Values(observation);
                    (float torque, Phase phase) = Controller(s);
                    // The frame now on screen shows the state the agent is about to act on (row t), and the arrow of
                    // the torque it applied last (row t - 1).
                    if (t == FigureStep)
                        SaveFigure(env, folder, s, shown);
                    StepResult<NDArray> step = env.Step(np.array(new[] { torque }));
                    total += step.Reward;
                    rows.Add(new double[] { t, Math.Atan2(s[1], s[0]), s[2], torque, (int)phase, step.Reward });
                    shown = torque;
                    observation = step.Observation;
                    if (step.Terminated || step.Truncated)
                        break;
                }
                double[] last = Values(observation);
                final = new[] { Math.Atan2(last[1], last[0]), last[2] };
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
                agent = "energy pumping, then a PD controller near the top (examples/Pendulum)",
                columns = new[] { "step", "theta", "theta_dot", "torque", "phase", "reward" },
                // Five decimals keep the chart exact to the pixel at a fraction of the file size.
                rows = rows.Select(r => r.Select(v => Math.Round(v, 5)).ToArray()).ToArray(),
                final = final.Select(v => Math.Round(v, 5)).ToArray(),
            }) + "\n");
            return (rows.Count, total);
        }

        /// <summary>
        ///     Saves How it works' frame (the frame the env shows now) as <c>Pendulum-v1.figure.png</c>, and where its
        ///     labels point as <c>Pendulum-v1.figure.json</c>: the pivot, the rod's end, the arrow's box, in the frame's
        ///     pixels, with the state and the torque the arrow shows.
        /// </summary>
        /// <param name="env">The recording env, in <c>rgb_array</c> mode; rendering it adds nothing to the video.</param>
        /// <param name="folder">The output folder.</param>
        /// <param name="s">The state the frame shows: <c>[cos(theta), sin(theta), theta_dot]</c>.</param>
        /// <param name="torque">The torque of the step before, which the frame's arrow shows (Gymnasium's <c>last_u</c>).</param>
        /// <exception cref="IOException">A file cannot be written.</exception>
        /// <remarks>
        ///     The pixels follow Pendulum's renderer: the 500 px frame spans ±2.2 m around the pivot at its centre
        ///     (113.6 px a metre); the rod, 1 m long, is rotated by <c>theta + pi/2</c> from the x axis, so it points up at
        ///     theta = 0 and turns counterclockwise as theta grows; its end is drawn at the rod's length, rounded down by
        ///     <c>int()</c>; the arrow image is scaled to <c>113.6 |u| / 2</c> px square, centred on the pivot, and
        ///     mirrored for a positive torque, which turns counterclockwise. Reference:
        ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/pendulum.py#L173-L262">gymnasium/envs/classic_control/pendulum.py</see>.
        /// </remarks>
        private static void SaveFigure(Env<NDArray, NDArray> env, string folder, double[] s, float torque) {
            using (RenderResult render = env.Render()) {
                NDArray frame = (NDArray)render;
                int height = (int)frame.shape[0], width = (int)frame.shape[1];
                using var image = Image.LoadPixelData<Rgb24>(frame.ToArray<byte>(), width, height);
                image.SaveAsPng(Path.Combine(folder, $"{Id}.figure.png"));
            }
            // The renderer draws in pygame's upward coordinates around the centre and flips the frame, so a point (x, y)
            // above the pivot is the frame's pixel (250 + x, 250 - y).
            const double Scale = 500 / (2.2 * 2), Pivot = 500 / 2;
            double theta = Math.Atan2(s[1], s[0]);
            // rotate_rad(theta + pi/2) of (Scale, 0): (Scale cos(theta + pi/2), Scale sin(theta + pi/2)), upward.
            double endX = Pivot + Scale * Math.Cos(theta + Math.PI / 2), endY = Pivot - Scale * Math.Sin(theta + Math.PI / 2);
            double arrow = Scale * Math.Abs(torque) / 2;
            static double[] Point(double column, double row) => new[] { Math.Round(column, 2), Math.Round(row, 2) };
            File.WriteAllText(Path.Combine(folder, $"{Id}.figure.json"), JsonSerializer.Serialize(new {
                id = Id, step = FigureStep, width = 500, height = 500, pixelsPerMetre = Math.Round(Scale, 4),
                state = new { theta = Math.Round(theta, 5), theta_dot = Math.Round(s[2], 5) },
                torque = Math.Round(torque, 5),
                pivot = Point(Pivot, Pivot),
                end = Point(endX, endY),
                // The arrow image's box: a square of this side centred on the pivot.
                arrowSide = Math.Round(arrow, 2),
            }, new JsonSerializerOptions { WriteIndented = true }) + "\n");
        }

        /// <summary>
        ///     The keys for <c>--play</c>: A turns the motor one way at full torque (-2), D the other way (+2).
        /// </summary>
        /// <returns>The map from keys to float32 torques; the caller disposes the actions.</returns>
        private static Dictionary<object, object?> Keys() => new() {
            ["a"] = np.array(new[] { -2f }),
            ["d"] = np.array(new[] { 2f }),
        };

        /// <summary>
        ///     Hands the env to the keyboard, Gymnasium's play, until Escape or the window's close button.
        /// </summary>
        /// <param name="seed">
        ///     The seed of every reset: each new game starts as the last one did, as Gymnasium's play does.
        /// </param>
        private static void PlayWithKeyboard(long seed) {
            // play reads the env's rgb_array frames and draws them into its own window.
            using IEnv env = Gymnasium.Make(Id, new { render_mode = "rgb_array" });
            Dictionary<object, object?> keys = Keys();
            NDArray noop = np.array(new[] { 0f });
            try {
                Play.Run(env, keysToAction: keys, noop: noop, seed: seed);
            } finally {
                // play only borrows a map it is given: its actions, and the no-op, stay ours to release.
                foreach (object? action in keys.Values)
                    (action as IDisposable)?.Dispose();
                noop.Dispose();
            }
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
