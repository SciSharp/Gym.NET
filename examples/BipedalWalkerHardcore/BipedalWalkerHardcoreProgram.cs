#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;
using Gym.Box2D;
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

namespace Gym.Examples.BipedalWalkerHardcore {
    /// <summary>
    ///     Plays BipedalWalkerHardcore-v3 with RL Zoo's TQC expert, a published Stable-Baselines3 agent. <c>dotnet run
    ///     --project examples/BipedalWalkerHardcore -c Release</c> plays one seeded episode headless and prints its
    ///     steps and return; <c>-- --render</c> plays it in a window, <c>-- --play</c> hands the four motors to the
    ///     keyboard, and <c>-- --record &lt;folder&gt;</c> records the episode for the hardcore section of Bipedal
    ///     Walker's page on the docs site (DR-112).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <c>policy.json</c> holds the agent (<see cref="ExpertPolicy"/>): RL Zoo's TQC expert, the agent RL
    ///     Baselines3 Zoo trained and published for this env, imported from Hugging Face
    ///     (<c>sb3/tqc-BipedalWalkerHardcore-v3</c> at revision <c>4eb282a</c>) by
    ///     <c>tools/agents/import_agents.py</c>. A Stable-Baselines3 network of dense layers of 400 and 300 with relu:
    ///     each step it casts the observation to float32 and squashes the result into the motors' range with tanh. Over
    ///     seeds 0-9 it scores 243.8 in Gym.NET (six seeds of ten finish the course; seed 7, the default, in 968 steps
    ///     for 308); the model card reports 208.05 +/- 121.38.
    ///     </para>
    ///     <para>
    ///     The model repository states no licence; its training code is MIT. Gym.NET commits the weights with this
    ///     attribution, on the maintainer's decision (DR-109).
    ///     </para>
    ///     <para>
    ///     The default seed, 7, is the episode the front-page showreel plays (<c>tools/showreel</c>, DR-74).
    ///     </para>
    /// </remarks>
    internal static class BipedalWalkerHardcoreProgram {
        /// <summary>The environment this example plays, by its registered id.</summary>
        private const string Id = "BipedalWalkerHardcore-v3";

        /// <summary>The first reset seed when <c>--seed</c> is not given: the episode the showreel plays.</summary>
        private const long DefaultSeed = 7;

        /// <summary>
        ///     The moment the hardcore section of Bipedal Walker's page shows (DR-114): the frame after this many steps of
        ///     the seed-7 episode, the walker at the foot of a staircase with a stump and a pit ahead, the course's three
        ///     obstacles in one frame.
        /// </summary>
        private const int FigureStep = 110;

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
        ///     <c>--record</c> could not create the folder or write one of its files, or <c>policy.json</c> is missing.
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
            // The agent first: its file names the constructor arguments the env needs for the agent to see what it
            // was trained on.
            ExpertPolicy policy = ExpertPolicy.Load(Path.Combine(AppContext.BaseDirectory, "policy.json"));
            // make's stack: TimeLimit over OrderEnforcing over PassiveEnvChecker over the env. The render mode is a
            // constructor argument, fixed for the env's life, as in Gymnasium.
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>(Id, policy.MakeArguments(render));
            (NDArray observation, Info info) = env.Reset(seed: seed);
            int steps = 0;
            double total = 0;
            while (true) {
                StepResult<NDArray> step = env.Step(policy.Act(observation));
                observation = step.Observation;
                steps++;
                total += step.Reward;
                if (step.Terminated || step.Truncated)
                    return (steps, total, step.Terminated ? total > 0 ? "reached the end of the course" : "fell over" : "truncated by the time limit");
            }
        }

        /// <summary>
        ///     Records the seed-7 episode for the hardcore section of Bipedal Walker's page on the docs site (DR-112): the
        ///     whole episode as <c>BipedalWalkerHardcore-v3.episode.mp4</c>, one frame per step at the env's 50 frames a
        ///     second, through Gymnasium's <see cref="RecordVideo{TObs, TAct}"/> and <see cref="FfmpegVideoWriter"/>;
        ///     every step as <c>BipedalWalkerHardcore-v3.episode.json</c>; and the section's frame as
        ///     <c>BipedalWalkerHardcore-v3.figure.png</c> with <c>BipedalWalkerHardcore-v3.figure.json</c>.
        /// </summary>
        /// <param name="folder">The output folder, <c>docs/environments/box2d/media</c> for the site; created if missing.</param>
        /// <returns>The steps taken and the return.</returns>
        /// <exception cref="InvalidOperationException">ffmpeg could not be run, or wrote no video.</exception>
        /// <exception cref="IOException">A file cannot be written, or <c>policy.json</c> cannot be read.</exception>
        /// <remarks>
        ///     <para>
        ///     Each JSON row is the state the agent acted on, its four commands and the reward: <c>step, hull_angle, vx, vy,
        ///     hip1, knee1, contact1, hip2, knee2, contact2, x, a0, a1, a2, a3, reward</c>, the columns of BipedalWalker's
        ///     own trace (<c>examples/BipedalWalker</c>), so the page's two charts read alike. The first ten values are the
        ///     observation's items 0, 2, 3, 4, 6 (less the env's +1), 8, 9, 11 (less its +1) and 13, rounded to 5
        ///     decimals, and the hull's world x, which the observation leaves out. The commands are the network's float32
        ///     values, each written as the float64 that holds it exactly, so the trace replays the episode bit for bit
        ///     (<c>action_dtype</c> says to cast them back). The video's frame <c>t</c> shows the state of row <c>t</c>,
        ///     and its last frame the state after the last step, which the JSON keeps as <c>final</c>.
        ///     </para>
        ///     <para>
        ///     RecordVideo names its file <c>BipedalWalkerHardcore-v3-episode-0.mp4</c> and writes it when the env closes;
        ///     it records into a scratch folder inside <paramref name="folder"/> and the file is moved to the page's name,
        ///     so a recording that fails halfway leaves the published video as it was.
        ///     </para>
        /// </remarks>
        [NDScoped]   // releases every array the episode creates when it returns (DR-80); the envs are disposed inside
        private static (int Steps, double Return) Record(string folder) {
            Directory.CreateDirectory(folder);
            string scratch = Path.Combine(folder, ".recording");
            var rows = new List<double[]>();
            double total = 0;
            double[] final;
            ExpertPolicy policy = ExpertPolicy.Load(Path.Combine(AppContext.BaseDirectory, "policy.json"));
            // The agent's own constructor arguments (none for this agent) with the recorder's render mode: RecordVideo
            // reads rgb_array frames.
            Dictionary<string, object?> arguments = policy.MakeArguments(human: false);
            arguments["render_mode"] = "rgb_array";
            using (Env<NDArray, NDArray> made = Gymnasium.Make<NDArray, NDArray>(Id, arguments))
            using (var env = new RecordVideo<NDArray, NDArray>(made, scratch, episodeTrigger: _ => true, namePrefix: Id,
                       videoWriter: new FfmpegVideoWriter())) {
                var walker = (BipedalWalkerEnv)env.Unwrapped;
                (NDArray observation, Info info) = env.Reset(seed: DefaultSeed);
                for (int t = 0; ; t++) {
                    NDArray action = policy.Act(observation);
                    // float32 to float64 is exact, so the JSON's number casts back to the very float32 the agent chose.
                    double[] commands = action.ToArray<float>().Select(v => (double)v).ToArray();
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
                agent = "RL Zoo's TQC expert (examples/BipedalWalkerHardcore)", action_dtype = "float32",
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
        ///     Saves the hardcore section's frame (the recorder's last, the state now) as
        ///     <c>BipedalWalkerHardcore-v3.figure.png</c>, and what its labels point at as
        ///     <c>BipedalWalkerHardcore-v3.figure.json</c>, in the frame's pixels: the hull and its angle, the four leg
        ///     segments, where each of the ten lidar rays stops, and the corners of every obstacle in view.
        /// </summary>
        /// <param name="env">
        ///     The recording env, whose last recorded frame is the figure. It is not rendered again: each render moves the
        ///     lidar's animation on (one ray is drawn a frame, in turn), so a second render would change this frame's ray and
        ///     every later one.
        /// </param>
        /// <param name="folder">The output folder.</param>
        /// <param name="walker">The unwrapped env, for its bodies, its terrain and its scroll.</param>
        /// <param name="observation">The observation the frame shows, whose items 14 to 23 are the lidar's fractions.</param>
        /// <exception cref="IOException">A file cannot be written.</exception>
        /// <remarks>
        ///     <para>
        ///     The frame is the last 600 columns of a surface drawn at <c>SCALE</c> pixels a unit and flipped upside down,
        ///     so a world point <c>(x, y)</c> is the pixel <c>((x - scroll) 30, 400 - 30 y)</c>. Ray <c>i</c> runs from the
        ///     hull's centre toward <c>(x + sin(0.15 i) R, y - cos(0.15 i) R)</c>, <c>R = 160 / 30</c>, and stops at the
        ///     observation's fraction <c>14 + i</c> of the way. Reference:
        ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/bipedal_walker.py#L556-L563">gymnasium/envs/box2d/bipedal_walker.py</see>.
        ///     </para>
        ///     <para>
        ///     The obstacles are the terrain's polygon bodies (the ground's segments are edges): a stump's square, the two
        ///     walls of a pit, a staircase's steps. Their kind is not stored by the env (Gymnasium's generator keeps none),
        ///     so the JSON gives each one's corners and the page tells them apart by their sizes. Reference:
        ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/bipedal_walker.py#L283-L391">gymnasium/envs/box2d/bipedal_walker.py</see>.
        ///     </para>
        /// </remarks>
        private static void SaveFigure(RecordVideo<NDArray, NDArray> env, string folder, BipedalWalkerEnv walker, NDArray observation) {
            NDArray frame = env.RecordedFrames[^1];
            int height = (int)frame.shape[0], width = (int)frame.shape[1];
            using (var image = Image.LoadPixelData<Rgb24>(frame.ToArray<byte>(), width, height))
                image.SaveAsPng(Path.Combine(folder, $"{Id}.figure.png"));
            double scroll = walker.Scroll;
            double[] Pixel(double x, double y) => new[] { Math.Round((x - scroll) * BipedalWalkerEnv.Scale, 2), Math.Round(height - y * BipedalWalkerEnv.Scale, 2) };
            b2Body hull = walker.Hull!;
            float[] o = observation.ToArray<float>();
            // Every obstacle with a corner on screen, left to right: a body's polygon is in its own frame, so each corner
            // goes through the body's transform (an identity for the terrain's static bodies, applied all the same).
            double[][][] obstacles = walker.Terrain
                .SelectMany(body => body.Fixtures.Select(f => f.Shape).OfType<b2PolygonShape>()
                    .Select(polygon => polygon.Vertices.Select(v => body.Transform * v).Select(w => Pixel(w.x, w.y)).ToArray()))
                .Where(corners => corners.Any(c => c[0] >= 0 && c[0] <= width))
                .OrderBy(corners => corners.Min(c => c[0]))
                .ToArray();
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
                obstacles,
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

    /// <summary>
    ///     A published expert agent read from <c>policy.json</c> (DR-109): the deterministic actor network of a
    ///     Stable-Baselines3 model, which maps an observation to the action its <c>predict(deterministic=True)</c>
    ///     takes.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The file holds the network's dense layers (base64 of little-endian float32 values, each weight row-major
    ///     <c>outputs x inputs</c> as PyTorch's <c>nn.Linear</c> stores it), the activation after each, how the last
    ///     layer's values become the action (<c>tanh-rescale</c> for SAC and TQC, <c>clip</c> for PPO), and the
    ///     constructor arguments the env must be made with for the agent to see what it was trained on
    ///     (<c>make_kwargs</c>). It is a copy of the agent in <c>tests/Gym.Tests/Agents</c>, written by
    ///     <c>tools/agents/import_agents.py</c>, and this reader a copy of that folder's <c>ExpertPolicy</c>: every
    ///     example keeps its agent's code to itself (DR-74).
    ///     </para>
    ///     <para>
    ///     It computes as PyTorch does: the observation is cast to float32, each dot product is summed in float64, and
    ///     every layer's output is rounded to float32. An action can still differ from PyTorch's in its last bit, and
    ///     over a long episode of chaotic physics a last bit grows into a different trajectory, as good as the first.
    ///     </para>
    /// </remarks>
    internal sealed class ExpertPolicy {
        /// <summary>The widest layer the stack scratch holds; the published agents' widest is 400.</summary>
        private const int MaxWidth = 1024;

        /// <summary>The dense layers, input layer first.</summary>
        private readonly Layer[] _layers;

        /// <summary>The action space's lower bounds, as the model saved them.</summary>
        private readonly float[] _low;

        /// <summary>The action space's upper bounds.</summary>
        private readonly float[] _high;

        /// <summary>
        ///     Whether the output is squashed by tanh and rescaled to the bounds (SAC, TQC), rather than clipped (PPO).
        /// </summary>
        private readonly bool _squash;

        /// <summary>
        ///     The constructor arguments the agent's env needs (<c>make_kwargs</c>); none for most agents.
        /// </summary>
        private readonly Dictionary<string, object?> _makeKwargs;

        /// <summary>Creates an agent from its parsed parts; <see cref="Load"/> is the way in.</summary>
        /// <param name="layers">The dense layers.</param>
        /// <param name="low">The lower bounds.</param>
        /// <param name="high">The upper bounds.</param>
        /// <param name="squash">Whether the output is tanh-squashed.</param>
        /// <param name="makeKwargs">The constructor arguments.</param>
        private ExpertPolicy(Layer[] layers, float[] low, float[] high, bool squash, Dictionary<string, object?> makeKwargs) {
            _layers = layers;
            _low = low;
            _high = high;
            _squash = squash;
            _makeKwargs = makeKwargs;
        }

        /// <summary>Reads the agent's file.</summary>
        /// <param name="path">The <c>policy.json</c> file.</param>
        /// <returns>The agent.</returns>
        /// <exception cref="IOException">The file cannot be read.</exception>
        /// <exception cref="JsonException">It is not JSON.</exception>
        /// <exception cref="InvalidDataException">
        ///     It is not a <c>gym.net-agent/1</c> file, a layer's sizes do not chain, an array has the wrong length, or
        ///     an activation, output mode or constructor argument is of an unknown kind.
        /// </exception>
        public static ExpertPolicy Load(string path) {
            JsonNode root = JsonNode.Parse(File.ReadAllText(path)) ?? throw new JsonException($"{path}: empty");
            if (root["format"]?.GetValue<string>() != "gym.net-agent/1")
                throw new InvalidDataException($"{path}: not a gym.net-agent/1 file");
            Layer[] layers = root["layers"]!.AsArray().Select(n => Layer.Parse(n!, path)).ToArray();
            for (int i = 1; i < layers.Length; i++)
                if (layers[i].Inputs != layers[i - 1].Outputs)
                    throw new InvalidDataException($"{path}: layer {i} takes {layers[i].Inputs} inputs, layer {i - 1} gives {layers[i - 1].Outputs}");
            if (layers.Any(l => l.Outputs > MaxWidth))
                throw new InvalidDataException($"{path}: a layer is wider than {MaxWidth}");
            float[] Bounds(string name) {
                if (root[name] is not JsonArray array || array.Count != layers[^1].Outputs)
                    throw new InvalidDataException($"{path}: '{name}' must hold {layers[^1].Outputs} numbers");
                return array.Select(v => (float)v!.GetValue<double>()).ToArray();
            }

            bool squash = root["output"]?.GetValue<string>() switch {
                "tanh-rescale" => true,
                "clip" => false,
                var other => throw new InvalidDataException($"{path}: unknown output '{other}'"),
            };
            var kwargs = new Dictionary<string, object?>();
            if (root["make_kwargs"] is JsonObject make)
                foreach ((string name, JsonNode? value) in make)
                    kwargs[name] = value?.GetValueKind() switch {
                        JsonValueKind.True => true,
                        JsonValueKind.False => false,
                        JsonValueKind.Number => value.GetValue<double>(),
                        JsonValueKind.String => value.GetValue<string>(),
                        _ => throw new InvalidDataException($"{path}: make_kwargs '{name}' must be a bool, number or string"),
                    };
            return new ExpertPolicy(layers, Bounds("action_low"), Bounds("action_high"), squash, kwargs);
        }

        /// <summary>
        ///     The keyword arguments to make the env with: the agent's own (<c>make_kwargs</c>: HumanoidStandup's
        ///     expert sees positions and velocities only), and for a window, human mode and the view's arguments.
        /// </summary>
        /// <param name="human">Whether the env renders into a window (<c>render_mode="human"</c>).</param>
        /// <param name="view">
        ///     Arguments that change only what the window shows (a camera), added in human mode; or null.
        /// </param>
        /// <returns>A new dictionary for <c>Gymnasium.Make</c>.</returns>
        public Dictionary<string, object?> MakeArguments(bool human, IReadOnlyDictionary<string, object?>? view = null) {
            var arguments = new Dictionary<string, object?>(_makeKwargs);
            if (human) {
                arguments["render_mode"] = "human";
                if (view is not null)
                    foreach ((string name, object? value) in view)
                        arguments[name] = value;
            }
            return arguments;
        }

        /// <summary>The action for an observation.</summary>
        /// <param name="observation">The env's observation (float64 for MuJoCo, float32 for Box2D).</param>
        /// <returns>A new float32 array of one command per motor.</returns>
        /// <exception cref="ArgumentException">The observation's length is not the network's.</exception>
        public NDArray Act(NDArray observation) {
            float[] input;
            // Cast to float32 first, as Stable-Baselines3 casts what the env returns.
            using (NDArray values = observation.astype(NPTypeCode.Single))
                input = values.ToArray<float>();
            if (input.Length != _layers[0].Inputs)
                throw new ArgumentException($"the agent reads {_layers[0].Inputs} numbers, the observation has {input.Length}", nameof(observation));
            // Two stack buffers alternate as layer input and output: a forward pass allocates nothing but the action.
            Span<float> a = stackalloc float[MaxWidth], b = stackalloc float[MaxWidth];
            scoped ReadOnlySpan<float> x = input;
            for (int l = 0; l < _layers.Length; l++) {
                Span<float> y = (l % 2 == 0 ? a : b)[.._layers[l].Outputs];
                _layers[l].Forward(x, y);
                x = y;
            }
            var action = new float[_low.Length];
            // Stable-Baselines3's deterministic predict: unscale a tanh-squashed output, or clip an unsquashed one.
            for (int i = 0; i < action.Length; i++)
                action[i] = _squash ? _low[i] + 0.5f * (MathF.Tanh(x[i]) + 1f) * (_high[i] - _low[i]) : Math.Clamp(x[i], _low[i], _high[i]);
            return np.array(action);
        }

        /// <summary>One dense layer: <c>y = activation(W x + b)</c> in float32, each sum taken in float64.</summary>
        /// <param name="Inputs">Its input width.</param>
        /// <param name="Outputs">Its output width.</param>
        /// <param name="Weight">The weights, row-major <c>Outputs x Inputs</c>.</param>
        /// <param name="Bias">The biases.</param>
        /// <param name="Activation"><c>relu</c>, <c>tanh</c> or <c>none</c>.</param>
        private sealed record Layer(int Inputs, int Outputs, float[] Weight, float[] Bias, string Activation) {
            /// <summary>Computes the layer's output.</summary>
            /// <param name="input">The input, <see cref="Inputs"/> values.</param>
            /// <param name="output">Receives <see cref="Outputs"/> values.</param>
            public void Forward(ReadOnlySpan<float> input, Span<float> output) {
                for (int o = 0; o < Outputs; o++) {
                    double sum = Bias[o];
                    for (int i = 0; i < Inputs; i++)
                        sum += (double)Weight[o * Inputs + i] * input[i];
                    // Rounded to float32 before the activation, as a float32 tensor holds the pre-activation.
                    float y = (float)sum;
                    output[o] = Activation switch {
                        "relu" => y > 0 ? y : 0,
                        "tanh" => MathF.Tanh(y),
                        _ => y,
                    };
                }
            }

            /// <summary>Reads a layer from its JSON object.</summary>
            /// <param name="node">The layer object.</param>
            /// <param name="path">The file, for errors.</param>
            /// <returns>The layer.</returns>
            /// <exception cref="InvalidDataException">
            ///     The sizes and the arrays disagree, or the activation is unknown.
            /// </exception>
            /// <exception cref="FormatException">An array is not base64.</exception>
            public static Layer Parse(JsonNode node, string path) {
                int inputs = node["inputs"]!.GetValue<int>(), outputs = node["outputs"]!.GetValue<int>();
                string activation = node["activation"]!.GetValue<string>();
                if (activation is not ("relu" or "tanh" or "none"))
                    throw new InvalidDataException($"{path}: unknown activation '{activation}'");
                float[] Decode(string field, int count) {
                    byte[] bytes = Convert.FromBase64String(node[field]!.GetValue<string>());
                    if (bytes.Length != count * 4)
                        throw new InvalidDataException($"{path}: '{field}' holds {bytes.Length / 4} values, not {count}");
                    var values = new float[count];
                    // Little-endian in the file; .NET runs little-endian on every platform it supports.
                    Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
                    return values;
                }

                return new Layer(inputs, outputs, Decode("weight", inputs * outputs), Decode("bias", outputs), activation);
            }
        }
    }
}
