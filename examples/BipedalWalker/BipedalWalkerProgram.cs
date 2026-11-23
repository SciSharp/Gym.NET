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
using Gym.Spaces;
using Gym.Utils;
using Gym.Wrappers;
using NumSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Gym.Examples.BipedalWalker {
    /// <summary>
    ///     Walks BipedalWalker-v3's whole course with Gymnasium's own gait, <c>BipedalWalkerHeuristics</c>. <c>dotnet
    ///     run --project examples/BipedalWalker -c Release</c> plays one seeded episode headless and prints its steps
    ///     and return; <c>-- --render</c> plays it in a window, <c>-- --play</c> hands the four motors to the keyboard,
    ///     and <c>-- --record &lt;folder&gt;</c> records the episode for Bipedal Walker's page on the docs site
    ///     (DR-112).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The gait is Gymnasium's walking <c>heuristic</c>, a state machine kept bit for bit: one leg stays straight
    ///     to carry the body while the other lifts, swings forward and comes down, then they swap; a PD controller
    ///     keeps the hull level. Seed 0 walks the whole course in 1,598 steps.
    ///     </para>
    ///     <para>
    ///     The default seed, 0, is the episode the front-page showreel plays (<c>tools/showreel</c>, DR-74).
    ///     </para>
    /// </remarks>
    internal static class BipedalWalkerProgram {
        /// <summary>The environment this example plays, by its registered id.</summary>
        private const string Id = "BipedalWalker-v3";

        /// <summary>The first reset seed when <c>--seed</c> is not given: the episode the showreel plays.</summary>
        private const long DefaultSeed = 0;

        /// <summary>
        ///     The moment How it works annotates on Bipedal Walker's page (DR-114): the frame after this many steps of the
        ///     seed-0 episode, the walker mid-stride on the course's uneven ground, a quarter of the way along.
        /// </summary>
        private const int FigureStep = 400;

        /// <summary>
        ///     Plays the episodes the arguments ask for with the agent, or hands the env to the keyboard, or records the
        ///     docs page's media.
        /// </summary>
        /// <param name="args">
        ///     <c>--seed N</c> starts from reset seed N; <c>--episodes N</c> plays N episodes on consecutive seeds and
        ///     prints their mean; <c>--render</c> shows them in a window; <c>--play</c> lets you play the env from the
        ///     keyboard instead, until Escape or the window's close button; <c>--record &lt;folder&gt;</c> records the
        ///     page's episode, seed 0 whatever <c>--seed</c> says (<see cref="Record"/>). Other arguments are ignored.
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
            // The gait is a state machine with a phase: a fresh one for every episode.
            var gait = new BipedalWalkerHeuristics();
            NDArray observation = env.Reset(seed: seed).Observation;
            int steps = 0;
            double total = 0;
            while (true) {
                StepResult<NDArray> step = env.Step(gait.StepHeuristic(observation));
                observation = step.Observation;
                steps++;
                total += step.Reward;
                if (step.Terminated || step.Truncated)
                    return (steps, total, step.Terminated ? total > 0 ? "reached the end of the course" : "fell over" : "truncated by the time limit");
            }
        }

        /// <summary>
        ///     Records the seed-0 episode for Bipedal Walker's page on the docs site (DR-112): the whole episode as
        ///     <c>BipedalWalker-v3.episode.mp4</c>, one frame per step at the env's 50 frames a second, through Gymnasium's
        ///     <see cref="RecordVideo{TObs, TAct}"/> and <see cref="FfmpegVideoWriter"/>; every step as
        ///     <c>BipedalWalker-v3.episode.json</c>; and How it works' frame as <c>BipedalWalker-v3.figure.png</c> with
        ///     <c>BipedalWalker-v3.figure.json</c>.
        /// </summary>
        /// <param name="folder">The output folder, <c>docs/environments/box2d/media</c> for the site; created if missing.</param>
        /// <returns>The steps taken and the return.</returns>
        /// <exception cref="InvalidOperationException">ffmpeg could not be run, or wrote no video.</exception>
        /// <exception cref="IOException">A file cannot be written.</exception>
        /// <remarks>
        ///     <para>
        ///     Each JSON row is the state the agent acted on, its four commands and the reward: <c>step, hull_angle, vx, vy,
        ///     hip1, knee1, contact1, hip2, knee2, contact2, x, a0, a1, a2, a3, reward</c>. The first ten values are the
        ///     observation's items 0, 2, 3, 4, 6 (less the env's +1), 8, 9, 11 (less its +1) and 13, rounded to 5
        ///     decimals, and the hull's world x, which the observation leaves out; the commands are kept whole, float64 as
        ///     the gait computes them, so the trace replays the episode exactly (<c>action_dtype</c> says so). The video's
        ///     frame <c>t</c> shows the state of row <c>t</c>, and its last frame the state after the last step, which the
        ///     JSON keeps as <c>final</c> (the same ten values).
        ///     </para>
        ///     <para>
        ///     RecordVideo names its file <c>BipedalWalker-v3-episode-0.mp4</c> and writes it when the env closes; it
        ///     records into a scratch folder inside <paramref name="folder"/> and the file is moved to the page's name, so a
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
                var walker = (BipedalWalkerEnv)env.Unwrapped;
                var gait = new BipedalWalkerHeuristics();
                NDArray observation = env.Reset(seed: DefaultSeed).Observation;
                for (int t = 0; ; t++) {
                    NDArray action = gait.StepHeuristic(observation);
                    double[] commands = action.ToArray<double>();
                    // The frame now on screen shows the state the agent is about to act on (row t): the recorder's last,
                    // not rendered again, since every render moves the lidar's animation on.
                    if (t == FigureStep)
                        SaveFigure(env, folder, walker, observation);
                    double[] state = State(observation, walker);
                    StepResult<NDArray> step = env.Step(action);
                    total += step.Reward;
                    rows.Add(state.Prepend(t).Concat(commands).Append(Math.Round(step.Reward, 5)).ToArray());
                    observation = step.Observation;
                    if (step.Terminated || step.Truncated)
                        break;
                }
                final = State(observation, walker);
                // RecordVideo writes the episode's video when it stops recording, which closing the env does.
                env.Close();
            }
            string recorded = Path.Combine(scratch, $"{Id}-episode-0.mp4");
            if (!File.Exists(recorded))
                throw new InvalidOperationException($"RecordVideo wrote no {recorded}; is ffmpeg on PATH?");
            File.Move(recorded, Path.Combine(folder, $"{Id}.episode.mp4"), overwrite: true);
            Directory.Delete(scratch, recursive: true);
            File.WriteAllText(Path.Combine(folder, $"{Id}.episode.json"), JsonSerializer.Serialize(new {
                id = Id, seed = DefaultSeed, steps = rows.Count, @return = total, fps = BipedalWalkerEnv.Fps,
                agent = "Gymnasium's walking heuristic (examples/BipedalWalker)", action_dtype = "float64",
                columns = new[] { "step", "hull_angle", "vx", "vy", "hip1", "knee1", "contact1", "hip2", "knee2", "contact2", "x", "a0", "a1", "a2", "a3", "reward" },
                rows,
                final,
            }) + "\n");
            return (rows.Count, total);
        }

        /// <summary>
        ///     The trace's view of a state: the observation's hull angle, velocities, joint angles and contacts, rounded to
        ///     5 decimals, the knees without the env's +1, and the hull's world x.
        /// </summary>
        /// <param name="observation">The float32 observation, shape <c>(24,)</c>.</param>
        /// <param name="walker">The unwrapped env, whose hull is where the observation was taken.</param>
        /// <returns>Ten values: hull_angle, vx, vy, hip1, knee1, contact1, hip2, knee2, contact2, x.</returns>
        private static double[] State(NDArray observation, BipedalWalkerEnv walker) {
            float[] o = observation.ToArray<float>();
            double R(double v) => Math.Round(v, 5);
            return new[] { R(o[0]), R(o[2]), R(o[3]), R(o[4]), R(o[6] - 1.0), o[8], R(o[9]), R(o[11] - 1.0), o[13], R(walker.Hull!.Position.x) };
        }

        /// <summary>
        ///     Saves How it works' frame (the recorder's last, the state now) as <c>BipedalWalker-v3.figure.png</c>, and
        ///     what its labels point at as <c>BipedalWalker-v3.figure.json</c>, in the frame's pixels: the hull and its
        ///     angle, the four leg segments, and where each of the ten lidar rays stops.
        /// </summary>
        /// <param name="env">
        ///     The recording env, whose last recorded frame is the figure. It is not rendered again: each render moves the
        ///     lidar's animation on (one ray is drawn a frame, in turn), so a second render would change this frame's ray and
        ///     every later one.
        /// </param>
        /// <param name="folder">The output folder.</param>
        /// <param name="walker">The unwrapped env, for its bodies and its scroll.</param>
        /// <param name="observation">The observation the frame shows, whose items 14 to 23 are the lidar's fractions.</param>
        /// <exception cref="IOException">A file cannot be written.</exception>
        /// <remarks>
        ///     The frame is the last 600 columns of a surface drawn at <c>SCALE</c> pixels a unit and flipped upside down,
        ///     so a world point <c>(x, y)</c> is the pixel <c>((x - scroll) 30, 400 - 30 y)</c>. Ray <c>i</c> runs from the
        ///     hull's centre toward <c>(x + sin(0.15 i) R, y - cos(0.15 i) R)</c>, <c>R = 160 / 30</c>, and stops at the
        ///     observation's fraction <c>14 + i</c> of the way. Reference:
        ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/bipedal_walker.py#L556-L563">gymnasium/envs/box2d/bipedal_walker.py</see>.
        /// </remarks>
        private static void SaveFigure(RecordVideo<NDArray, NDArray> env, string folder, BipedalWalkerEnv walker, NDArray observation) {
            NDArray frame = env.RecordedFrames[^1];
            int height = (int)frame.shape[0], width = (int)frame.shape[1];
            using (var image = Image.LoadPixelData<Rgb24>(frame.ToArray<byte>(), width, height))
                image.SaveAsPng(Path.Combine(folder, $"{Id}.figure.png"));
            double scroll = walker.Scroll;
            double[] Pixel(double x, double y) => new[] { Math.Round((x - scroll) * BipedalWalkerEnv.Scale, 2), Math.Round(height - y * BipedalWalkerEnv.Scale, 2) };
            var hull = walker.Hull!;
            float[] o = observation.ToArray<float>();
            File.WriteAllText(Path.Combine(folder, $"{Id}.figure.json"), JsonSerializer.Serialize(new {
                id = Id, step = FigureStep, width, height, scale = BipedalWalkerEnv.Scale, scroll = Math.Round(scroll, 5),
                observation = o.Select(v => Math.Round((double)v, 5)).ToArray(),
                hull = Pixel(hull.Position.x, hull.Position.y), angle = Math.Round((double)hull.Angle, 5),
                legs = walker.Legs.Select(l => Pixel(l.Position.x, l.Position.y)).ToArray(),
                lidar = Enumerable.Range(0, 10).Select(i => {
                    double fraction = o[14 + i];
                    double ex = hull.Position.x + Math.Sin(1.5 * i / 10.0) * BipedalWalkerEnv.LidarRange * fraction;
                    double ey = hull.Position.y - Math.Cos(1.5 * i / 10.0) * BipedalWalkerEnv.LidarRange * fraction;
                    return Pixel(ex, ey);
                }).ToArray(),
            }, new JsonSerializerOptions { WriteIndented = true }) + "\n");
        }

        /// <summary>
        ///     The action space's upper bound per motor, as float64: the policy scales its [-1, 1] outputs by it.
        /// </summary>
        /// <param name="env">The env, whose action space is a <see cref="Box"/>.</param>
        /// <returns>One bound per motor.</returns>
        private static double[] ActionHigh(IEnv env) {
            // High is the space's own array: read a converted copy and leave the space's untouched.
            using NDArray high = ((Box)env.ActionSpace).High.astype(NPTypeCode.Double);
            return high.ToArray<double>();
        }

        /// <summary>
        ///     The keys for <c>--play</c>: Q and A the first hip, W and S the first knee, U and J the second hip, I and
        ///     K the second knee. The first key of a pair turns its motor one way at full torque, the second the other
        ///     way; keys of different motors combine.
        /// </summary>
        /// <param name="high">The action space's upper bound per motor.</param>
        /// <returns>
        ///     The map from the keys held together (a string of them) to a float32 action; the caller disposes the
        ///     actions.
        /// </returns>
        /// <remarks>
        ///     Gymnasium's play looks up the exact set of keys held, so the map lists every combination: 3^4 - 1 = 80
        ///     entries, one per motor being off, forward or back. Holding both keys of one pair matches nothing and
        ///     plays the no-op, every motor off.
        /// </remarks>
        private static Dictionary<object, object?> Keys(double[] high) {
            string[] pairs = { "qa", "ws", "uj", "ik" };
            var map = new Dictionary<object, object?>();
            int combinations = (int)Math.Pow(3, pairs.Length);
            // Each combination numbers the motors' states in base 3: 0 off, 1 the pair's first key, 2 its second.
            for (int combination = 1; combination < combinations; combination++) {
                var action = new float[pairs.Length];
                string held = "";
                for (int motor = 0, rest = combination; motor < pairs.Length; motor++, rest /= 3) {
                    int state = rest % 3;
                    if (state == 0)
                        continue;
                    held += pairs[motor][state - 1];
                    action[motor] = (float)(state == 1 ? high[motor] : -high[motor]);
                }
                map[held] = np.array(action);
            }

            return map;
        }

        /// <summary>
        ///     Hands the env to the keyboard, Gymnasium's play, until Escape or the window's close button.
        /// </summary>
        /// <param name="seed">
        ///     The seed of every reset: each new game starts as the last one did, as Gymnasium's play does.
        /// </param>
        private static void PlayWithKeyboard(long seed) {
            // play reads the env's rgb_array frames and draws them into its own window.
            using IEnv env = Gymnasium.Make(Id, new { render_mode = "rgb_array" });
            double[] high = ActionHigh(env);
            Dictionary<object, object?> keys = Keys(high);
            NDArray noop = np.array(new float[4]);
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
