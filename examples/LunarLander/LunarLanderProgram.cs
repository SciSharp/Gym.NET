#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using Gym.Environments.Envs.Box2D;
using Gym.Envs;
using Gym.Observations;
using Gym.Rendering.Avalonia;
using Gym.Utils;
using Gym.Wrappers;
using NumSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Gym.Examples.LunarLander {
    /// <summary>
    ///     Lands LunarLander-v3 between the flags with Gymnasium's own heuristic, <c>LunarLanderEnv.Heuristic</c>.
    ///     <c>dotnet run --project examples/LunarLander -c Release</c> plays one seeded episode headless and prints its
    ///     steps and return; <c>-- --render</c> plays it in a window, <c>-- --play</c> hands the engines to the W, A and
    ///     D keys, and <c>-- --record &lt;folder&gt;</c> records the episode for Lunar Lander's page on the docs site
    ///     (DR-112).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The heuristic is Gymnasium's <c>heuristic</c> function, kept bit for bit: a PD controller aims the lander's
    ///     angle at a target that leans it toward the pad, fires the main engine to hold a descent rate that slows with
    ///     height, and cuts the engines once both legs touch. Seed 7 lands in 299 steps.
    ///     </para>
    ///     <para>
    ///     The default seed, 7, is the episode the front-page showreel plays (<c>tools/showreel</c>, DR-74).
    ///     </para>
    /// </remarks>
    internal static class LunarLanderProgram {
        /// <summary>The environment this example plays, by its registered id.</summary>
        private const string Id = "LunarLander-v3";

        /// <summary>The first reset seed when <c>--seed</c> is not given: the episode the showreel plays.</summary>
        private const long DefaultSeed = 7;

        /// <summary>
        ///     The moment How it works annotates on Lunar Lander's page (DR-114): the frame after this many steps of the
        ///     seed-7 episode, the lander left of the pad and tilted 17 degrees, firing its left orientation engine to
        ///     turn upright again, its exhaust particles trailing.
        /// </summary>
        private const int FigureStep = 60;

        /// <summary>
        ///     Plays the episodes the arguments ask for with the agent, or hands the env to the keyboard, or records the
        ///     docs page's media.
        /// </summary>
        /// <param name="args">
        ///     <c>--seed N</c> starts from reset seed N; <c>--episodes N</c> plays N episodes on consecutive seeds and
        ///     prints their mean; <c>--render</c> shows them in a window; <c>--play</c> lets you play the env from the
        ///     keyboard instead, until Escape or the window's close button; <c>--record &lt;folder&gt;</c> records the
        ///     page's episode, seed 7 whatever <c>--seed</c> says (<see cref="Record"/>). Other arguments are ignored.
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
        /// <exception cref="System.IO.IOException">
        ///     <c>--record</c> could not create the folder or write one of its files.
        /// </exception>
        private static int Main(string[] args) {
            if (!TryNumber(args, "--seed", DefaultSeed, out long seed) || !TryNumber(args, "--episodes", 1, out long episodes))
                return 2;
            int record = Array.IndexOf(args, "--record");
            if (record >= 0) {
                // The folder is required: the page's media live in the docs tree, and a default would write them where no
                // page reads them.
                if (record + 1 >= args.Length || args[record + 1].StartsWith("--", StringComparison.Ordinal)) {
                    Console.Error.WriteLine("--record needs a folder, e.g. --record docs/environments/box2d/media");
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
                StepResult<NDArray> step = env.Step(LunarLanderEnv.Heuristic(env, observation));
                observation = step.Observation;
                steps++;
                total += step.Reward;
                if (step.Terminated || step.Truncated)
                    return (steps, total, step.Terminated ? total > 100 ? "landed" : "crashed or flew off" : "truncated by the time limit");
            }
        }

        /// <summary>
        ///     Records the seed-7 episode for Lunar Lander's page on the docs site (DR-112): the whole episode as
        ///     <c>LunarLander-v3.episode.mp4</c>, one frame per step at the env's 50 frames a second, through Gymnasium's
        ///     <see cref="RecordVideo{TObs, TAct}"/> and <see cref="FfmpegVideoWriter"/>; every step as
        ///     <c>LunarLander-v3.episode.json</c>; and How it works' frame as <c>LunarLander-v3.figure.png</c> with
        ///     <c>LunarLander-v3.figure.json</c>.
        /// </summary>
        /// <param name="folder">The output folder, <c>docs/environments/box2d/media</c> for the site; created if missing.</param>
        /// <returns>The steps taken and the return.</returns>
        /// <exception cref="InvalidOperationException">ffmpeg could not be run, or wrote no video.</exception>
        /// <exception cref="IOException">A file cannot be written.</exception>
        /// <remarks>
        ///     <para>
        ///     Each JSON row is the observation the agent acted on, its action (0 nothing, 1 the left orientation engine, 2
        ///     the main engine, 3 the right one) and the reward, values rounded to 5 decimals: <c>step, x, y, vx, vy,
        ///     angle, angular_velocity, right_leg, left_leg, action, reward</c>. The observation's two contact flags are
        ///     the env's <c>legs[0]</c> and <c>legs[1]</c>, which stand right and left of the lander as the frame draws
        ///     it. The video's frame <c>t</c> shows the state of row <c>t</c>, and its last frame the state after the last
        ///     step, which the JSON keeps as <c>final</c>.
        ///     </para>
        ///     <para>
        ///     RecordVideo names its file <c>LunarLander-v3-episode-0.mp4</c> and writes it when the env closes; it records
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
                var lander = (LunarLanderEnv)env.Unwrapped;
                NDArray observation = env.Reset(seed: DefaultSeed).Observation;
                for (int t = 0; ; t++) {
                    NDArray action = LunarLanderEnv.Heuristic(env, observation);
                    long chosen = Convert.ToInt64(action.GetValue(), CultureInfo.InvariantCulture);
                    // The frame now on screen shows the state the agent is about to act on (row t): the last one the
                    // recorder captured, taken from it rather than rendered again, since a render ages the particles.
                    if (t == FigureStep)
                        SaveFigure(env, folder, lander, observation, chosen);
                    StepResult<NDArray> step = env.Step(action);
                    total += step.Reward;
                    rows.Add(Observed(observation).Concat(new[] { (double)chosen, Math.Round(step.Reward, 5) }).Prepend(t).ToArray());
                    observation = step.Observation;
                    if (step.Terminated || step.Truncated)
                        break;
                }
                final = Observed(observation);
                // RecordVideo writes the episode's video when it stops recording, which closing the env does.
                env.Close();
            }
            string recorded = Path.Combine(scratch, $"{Id}-episode-0.mp4");
            if (!File.Exists(recorded))
                throw new InvalidOperationException($"RecordVideo wrote no {recorded}; is ffmpeg on PATH?");
            File.Move(recorded, Path.Combine(folder, $"{Id}.episode.mp4"), overwrite: true);
            Directory.Delete(scratch, recursive: true);
            File.WriteAllText(Path.Combine(folder, $"{Id}.episode.json"), JsonSerializer.Serialize(new {
                id = Id, seed = DefaultSeed, steps = rows.Count, @return = total, fps = LunarLanderEnv.Fps,
                agent = "Gymnasium's heuristic (examples/LunarLander)",
                columns = new[] { "step", "x", "y", "vx", "vy", "angle", "angular_velocity", "right_leg", "left_leg", "action", "reward" },
                rows,
                final,
            }) + "\n");
            return (rows.Count, total);
        }

        /// <summary>An observation's eight values as float64, rounded to 5 decimals for the trace.</summary>
        /// <param name="observation">The float32 observation, shape <c>(8,)</c>.</param>
        /// <returns>The values: x, y, the two velocities, the angle, the angular velocity and the two contact flags.</returns>
        private static double[] Observed(NDArray observation) =>
            observation.ToArray<float>().Select(v => Math.Round((double)v, 5)).ToArray();

        /// <summary>
        ///     Saves How it works' frame (the frame the recorder captured last, the state now) as
        ///     <c>LunarLander-v3.figure.png</c>, and what its labels point at as <c>LunarLander-v3.figure.json</c>, in the
        ///     frame's pixels: the lander's centre and angle, each leg's centre, the pad's flags and height, and the point
        ///     the observation's x and y are measured from.
        /// </summary>
        /// <param name="env">
        ///     The recording env, whose last recorded frame is the figure. It is not rendered again: LunarLander's render
        ///     ages the exhaust particles (their time to live drops 0.15 a render, and spent ones leave the world), so a
        ///     second render would change this frame and every later one.
        /// </param>
        /// <param name="folder">The output folder.</param>
        /// <param name="lander">The unwrapped env, for its bodies and its pad.</param>
        /// <param name="observation">The observation the frame shows.</param>
        /// <param name="action">The action the agent takes from it.</param>
        /// <exception cref="IOException">A file cannot be written.</exception>
        /// <remarks>
        ///     The renderer draws the world at <c>SCALE</c> pixels a unit and then flips the frame upside down, so a world
        ///     point <c>(x, y)</c> is the pixel <c>(30 x, 400 - 30 y)</c>. The observation's origin is the pad's centre at
        ///     the legs' height above it: <c>x = (pos.x - W / 2) / (W / 2)</c>, <c>y = (pos.y - (helipad_y + LEG_DOWN /
        ///     SCALE)) / (H / 2)</c>. Reference:
        ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/lunar_lander.py#L624-L634">gymnasium/envs/box2d/lunar_lander.py</see>.
        /// </remarks>
        private static void SaveFigure(RecordVideo<NDArray, NDArray> env, string folder, LunarLanderEnv lander, NDArray observation, long action) {
            NDArray frame = env.RecordedFrames[^1];
            int height = (int)frame.shape[0], width = (int)frame.shape[1];
            using (var image = Image.LoadPixelData<Rgb24>(frame.ToArray<byte>(), width, height))
                image.SaveAsPng(Path.Combine(folder, $"{Id}.figure.png"));
            // World units to the frame's pixels, as the renderer places them.
            double[] Pixel(double x, double y) => new[] { Math.Round(x * LunarLanderEnv.Scale, 2), Math.Round(height - y * LunarLanderEnv.Scale, 2) };
            var body = lander.Lander!;
            File.WriteAllText(Path.Combine(folder, $"{Id}.figure.json"), JsonSerializer.Serialize(new {
                id = Id, step = FigureStep, width, height, scale = LunarLanderEnv.Scale,
                observation = Observed(observation), action,
                lander = Pixel(body.Position.x, body.Position.y), angle = Math.Round((double)body.Angle, 5),
                legs = lander.Legs.Select(l => Pixel(l.Position.x, l.Position.y)).ToArray(),
                flags = new[] { Pixel(lander.HelipadX1, lander.HelipadY), Pixel(lander.HelipadX2, lander.HelipadY) },
                origin = Pixel(LunarLanderEnv.ViewportW / LunarLanderEnv.Scale / 2, lander.HelipadY + LunarLanderEnv.LegDown / LunarLanderEnv.Scale),
            }, new JsonSerializerOptions { WriteIndented = true }) + "\n");
        }

        /// <summary>
        ///     The keys for <c>--play</c>: W fires the main engine, A the left orientation engine, D the right one (one
        ///     engine at a time: LunarLander's action is a single choice of four).
        /// </summary>
        /// <returns>The map from keys to LunarLander's actions.</returns>
        private static Dictionary<object, object?> Keys() => new() { ["w"] = 2L, ["a"] = 1L, ["d"] = 3L };

        /// <summary>
        ///     Hands the env to the keyboard, Gymnasium's play, until Escape or the window's close button.
        /// </summary>
        /// <param name="seed">
        ///     The seed of every reset: each new game starts as the last one did, as Gymnasium's play does.
        /// </param>
        /// <remarks>
        ///     With no key held every engine is off: the no-op is action 0.
        /// </remarks>
        private static void PlayWithKeyboard(long seed) {
            // play reads the env's rgb_array frames and draws them into its own window.
            using IEnv env = Gymnasium.Make(Id, new { render_mode = "rgb_array" });
            Dictionary<object, object?> keys = Keys();
            object noop = 0L;
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
