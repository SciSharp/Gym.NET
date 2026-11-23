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

namespace Gym.Examples.Acrobot {
    /// <summary>
    ///     Swings Acrobot-v1's tip over the line by torquing the second joint the way it already turns. <c>dotnet run
    ///     --project examples/Acrobot -c Release</c> plays one seeded episode headless and prints its steps and return;
    ///     <c>-- --render</c> plays it in a window, <c>-- --play</c> hands the joint to the A and D keys, and
    ///     <c>-- --record &lt;folder&gt;</c> records the episode for Acrobot's page on the docs site (DR-112).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Torque in the direction the second link swings feeds energy into the swing on every pass. Over seeds 0-9 the
    ///     tip crosses the line in 83 steps on average; seed 1 in 65.
    ///     </para>
    ///     <para>
    ///     The default seed, 1, is the episode the front-page showreel plays (<c>tools/showreel</c>, DR-74).
    ///     </para>
    /// </remarks>
    internal static class AcrobotProgram {
        /// <summary>The environment this example plays, by its registered id.</summary>
        private const string Id = "Acrobot-v1";

        /// <summary>The first reset seed when <c>--seed</c> is not given: the episode the showreel plays.</summary>
        private const long DefaultSeed = 1;

        /// <summary>
        ///     The moment How it works annotates on Acrobot's page (DR-114): the frame after this many steps of the seed-1
        ///     episode, the last state below the line, the tip 0.80 m above the pivot and rising; the next step ends the
        ///     episode.
        /// </summary>
        private const int FigureStep = 64;

        /// <summary>
        ///     Plays the episodes the arguments ask for with the agent, or hands the env to the keyboard, or records the
        ///     docs page's media.
        /// </summary>
        /// <param name="args">
        ///     <c>--seed N</c> starts from reset seed N; <c>--episodes N</c> plays N episodes on consecutive seeds and
        ///     prints their mean; <c>--render</c> shows them in a window; <c>--play</c> lets you play the env from the
        ///     keyboard instead, until Escape or the window's close button; <c>--record &lt;folder&gt;</c> records the
        ///     page's episode, seed 1 whatever <c>--seed</c> says (<see cref="Record"/>). Other arguments are ignored.
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
            NDArray observation = env.Reset(seed: seed).Observation;
            int steps = 0;
            double total = 0;
            while (true) {
                StepResult<NDArray> step = env.Step(Act(observation));
                observation = step.Observation;
                steps++;
                total += step.Reward;
                if (step.Terminated || step.Truncated)
                    return (steps, total, step.Terminated ? "swung the tip over the line" : "truncated by the time limit");
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

        /// <summary>The agent: torque the second joint the way it is already turning.</summary>
        /// <param name="observation">
        ///     The observation: the cosines and sines of both joint angles, then both angular velocities (index 5 is
        ///     the second joint's).
        /// </param>
        /// <returns>
        ///     +1 torque (2) when the second joint turns positively, else -1 torque (0), as a 0-d int64 array.
        /// </returns>
        private static NDArray Act(NDArray observation) => NDArray.Scalar(Push(Values(observation)[5]));

        /// <summary>
        ///     The agent's rule as a number: +1 torque (action 2) when the second joint turns positively, else -1 (action
        ///     0). <see cref="Act"/> and <see cref="Record"/> both decide through it, so the recorded episode is the one
        ///     the example plays.
        /// </summary>
        /// <param name="secondSpeed">The second joint's angular velocity, the observation's element 5.</param>
        /// <returns>The action, 0 or 2, as Acrobot's <c>Discrete(3)</c> numbers them.</returns>
        private static long Push(double secondSpeed) => secondSpeed > 0 ? 2L : 0L;

        /// <summary>
        ///     The joint angles and the tip's height an observation describes: <c>theta1</c> (the first link from straight
        ///     down), <c>theta2</c> (the second link relative to the first), both from their cosine and sine, and the tip's
        ///     height above the pivot, <c>-cos(theta1) - cos(theta1 + theta2)</c>, the quantity the episode ends on.
        /// </summary>
        /// <param name="s">The observation's values, float64.</param>
        /// <returns>The two angles in (-pi, pi], and the height in link lengths (metres).</returns>
        private static (double Theta1, double Theta2, double Height) Geometry(double[] s) {
            double theta1 = Math.Atan2(s[1], s[0]), theta2 = Math.Atan2(s[3], s[2]);
            return (theta1, theta2, -Math.Cos(theta1) - Math.Cos(theta1 + theta2));
        }

        /// <summary>
        ///     Records the seed-1 episode for Acrobot's page on the docs site (DR-112): the whole episode as
        ///     <c>Acrobot-v1.episode.mp4</c>, one frame per step at the env's 15 frames a second, through Gymnasium's
        ///     <see cref="RecordVideo{TObs, TAct}"/> and <see cref="FfmpegVideoWriter"/>; every step as
        ///     <c>Acrobot-v1.episode.json</c>, for the page's anatomy chart; and How it works' frame as
        ///     <c>Acrobot-v1.figure.png</c> with <c>Acrobot-v1.figure.json</c>, the pixels its labels point at.
        /// </summary>
        /// <param name="folder">The output folder, <c>docs/environments/classic_control/media</c> for the site; created if missing.</param>
        /// <returns>The steps taken and the return.</returns>
        /// <exception cref="InvalidOperationException">ffmpeg could not be run, or wrote no video.</exception>
        /// <exception cref="IOException">A file cannot be written.</exception>
        /// <remarks>
        ///     <para>
        ///     Each JSON row is the observation the agent acted on, as both joint angles (from their cosines and sines),
        ///     both angular velocities and the tip's height (<see cref="Geometry"/>), then the action and the reward:
        ///     <c>step, theta1, theta2, theta1_dot, theta2_dot, height, action, reward</c>. The video's frame <c>t</c>
        ///     shows the state of row <c>t</c> (frame 0 is the reset), and its last frame the state after the last step,
        ///     which the JSON keeps as <c>final</c> (the angles, the speeds and the height).
        ///     </para>
        ///     <para>
        ///     RecordVideo names its file <c>Acrobot-v1-episode-0.mp4</c> and writes it when the env closes; it records
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
                NDArray observation = env.Reset(seed: DefaultSeed).Observation;
                for (int t = 0; ; t++) {
                    double[] s = Values(observation);
                    long action = Push(s[5]);
                    (double theta1, double theta2, double height) = Geometry(s);
                    // The frame now on screen shows the state the agent is about to act on (row t).
                    if (t == FigureStep)
                        SaveFigure(env, folder, s);
                    StepResult<NDArray> step = env.Step(NDArray.Scalar(action));
                    total += step.Reward;
                    rows.Add(new double[] { t, theta1, theta2, s[4], s[5], height, action, step.Reward });
                    observation = step.Observation;
                    if (step.Terminated || step.Truncated)
                        break;
                }
                double[] last = Values(observation);
                (double lastTheta1, double lastTheta2, double lastHeight) = Geometry(last);
                final = new[] { lastTheta1, lastTheta2, last[4], last[5], lastHeight };
                // RecordVideo writes the episode's video when it stops recording, which closing the env does.
                env.Close();
            }
            string recorded = Path.Combine(scratch, $"{Id}-episode-0.mp4");
            if (!File.Exists(recorded))
                throw new InvalidOperationException($"RecordVideo wrote no {recorded}; is ffmpeg on PATH?");
            File.Move(recorded, Path.Combine(folder, $"{Id}.episode.mp4"), overwrite: true);
            Directory.Delete(scratch, recursive: true);
            File.WriteAllText(Path.Combine(folder, $"{Id}.episode.json"), JsonSerializer.Serialize(new {
                id = Id, seed = DefaultSeed, steps = rows.Count, @return = total, fps = 15,
                agent = "torque the second joint the way it turns (examples/Acrobot)",
                columns = new[] { "step", "theta1", "theta2", "theta1_dot", "theta2_dot", "height", "action", "reward" },
                // Five decimals keep the chart exact to the pixel at a fraction of the file size.
                rows = rows.Select(r => r.Select(v => Math.Round(v, 5)).ToArray()).ToArray(),
                final = final.Select(v => Math.Round(v, 5)).ToArray(),
            }) + "\n");
            return (rows.Count, total);
        }

        /// <summary>
        ///     Saves How it works' frame (the frame the env shows now) as <c>Acrobot-v1.figure.png</c>, and where its
        ///     labels point as <c>Acrobot-v1.figure.json</c>: the pivot, the elbow (the actuated joint), the tip and the
        ///     target line, in the frame's pixels, with the state they show.
        /// </summary>
        /// <param name="env">The recording env, in <c>rgb_array</c> mode; rendering it adds nothing to the video.</param>
        /// <param name="folder">The output folder.</param>
        /// <param name="s">The state the frame shows: the observation's values, float64.</param>
        /// <exception cref="IOException">A file cannot be written.</exception>
        /// <remarks>
        ///     The pixels follow Acrobot's renderer: the 500 px frame spans ±2.2 m around the pivot at its centre (113.6 px
        ///     a metre, the bound being the two links' 2 m and 0.2 m more); theta1 = 0 hangs the first link straight down,
        ///     so the elbow is at <c>(250 + 113.6 sin(theta1), 250 + 113.6 cos(theta1))</c> and the tip one link further
        ///     along <c>theta1 + theta2</c>; the line is drawn 1 m above the pivot, row <c>250 - 113.6</c>. Reference:
        ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/acrobot.py#L295-L367">gymnasium/envs/classic_control/acrobot.py</see>.
        /// </remarks>
        private static void SaveFigure(Env<NDArray, NDArray> env, string folder, double[] s) {
            using (RenderResult render = env.Render()) {
                NDArray frame = (NDArray)render;
                int height = (int)frame.shape[0], width = (int)frame.shape[1];
                using var image = Image.LoadPixelData<Rgb24>(frame.ToArray<byte>(), width, height);
                image.SaveAsPng(Path.Combine(folder, $"{Id}.figure.png"));
            }
            // The renderer draws in pygame's upward coordinates around the centre and flips the frame, so a point x right
            // and y up of the pivot is the frame's pixel (250 + x, 250 - y); a link hanging at angle a from straight down
            // reaches x = l sin(a), y = -l cos(a).
            const double Scale = 500 / ((1.0 + 1.0 + 0.2) * 2), Pivot = 500 / 2.0;
            (double theta1, double theta2, double tipHeight) = Geometry(s);
            double elbowX = Pivot + Scale * Math.Sin(theta1), elbowY = Pivot + Scale * Math.Cos(theta1);
            double tipX = elbowX + Scale * Math.Sin(theta1 + theta2), tipY = elbowY + Scale * Math.Cos(theta1 + theta2);
            static double[] Point(double column, double row) => new[] { Math.Round(column, 2), Math.Round(row, 2) };
            File.WriteAllText(Path.Combine(folder, $"{Id}.figure.json"), JsonSerializer.Serialize(new {
                id = Id, step = FigureStep, width = 500, height = 500, pixelsPerMetre = Math.Round(Scale, 4),
                state = new {
                    theta1 = Math.Round(theta1, 5), theta2 = Math.Round(theta2, 5),
                    theta1_dot = Math.Round(s[4], 5), theta2_dot = Math.Round(s[5], 5), height = Math.Round(tipHeight, 5),
                },
                pivot = Point(Pivot, Pivot),
                elbow = Point(elbowX, elbowY),
                tip = Point(tipX, tipY),
                // The target line's row: 1 m above the pivot.
                line = Math.Round(Pivot - Scale, 2),
            }, new JsonSerializerOptions { WriteIndented = true }) + "\n");
        }

        /// <summary>The keys for <c>--play</c>: A applies -1 torque to the second joint, D +1.</summary>
        /// <returns>The map from keys to Acrobot's actions.</returns>
        private static Dictionary<object, object?> Keys() => new() { ["a"] = 0L, ["d"] = 2L };

        /// <summary>
        ///     Hands the env to the keyboard, Gymnasium's play, until Escape or the window's close button.
        /// </summary>
        /// <param name="seed">
        ///     The seed of every reset: each new game starts as the last one did, as Gymnasium's play does.
        /// </param>
        /// <remarks>
        ///     With no key held the joint is free: the no-op is action 1, no torque.
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
