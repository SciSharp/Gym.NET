#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using Gym.Environments.Envs.ToyText;
using Gym.Envs;
using Gym.Observations;
using Gym.Rendering;
using Gym.Rendering.Avalonia;
using Gym.Utils;
using Gym.Wrappers;
using NumSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Gym.Examples.Taxi {
    /// <summary>
    ///     Picks up and delivers Taxi-v4's passenger by value iteration over the env's own transition table. <c>dotnet
    ///     run --project examples/Taxi -c Release</c> plays one seeded episode headless and prints its steps and
    ///     return; <c>-- --render</c> plays it in a window, <c>-- --play</c> lets you play it with the keyboard, and
    ///     <c>-- --record &lt;folder&gt;</c> records the episode for Taxi's page on the docs site (DR-112).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     500 states: the taxi's square, the passenger's place (one of four stands, or in the taxi) and the
    ///     destination. Every step costs 1, a delivery pays 20, and a pick-up or drop-off in the wrong place costs 10.
    ///     Value iteration over the transition table <c>P</c> finds the shortest delivery from every state: 16 steps
    ///     from seed 12.
    ///     </para>
    ///     <para>
    ///     The default seed, 12, is the episode the front-page showreel plays (<c>tools/showreel</c>, DR-74).
    ///     </para>
    /// </remarks>
    internal static class TaxiProgram {
        /// <summary>The environment this example plays, by its registered id.</summary>
        private const string Id = "Taxi-v4";

        /// <summary>The first reset seed when <c>--seed</c> is not given: the episode the showreel plays.</summary>
        private const long DefaultSeed = 12;

        /// <summary>
        ///     The moment How it works annotates on Taxi's page (DR-114, DR-125): the frame after this many steps of the
        ///     seed-12 episode, the taxi on the passenger's stand, G, about to pick them up, with the destination, R,
        ///     framed at the other end of the top row.
        /// </summary>
        private const int FigureStep = 6;

        /// <summary>
        ///     Gymnasium's <c>GRID_ORIGIN</c>: the pixel where the 5x5 grid's top-left cell starts in the 536x456 frame
        ///     (the env keeps it private, as Gymnasium's module constant is not an attribute of the env). Reference:
        ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/taxi.py#L28">gymnasium/envs/toy_text/taxi.py</see>.
        /// </summary>
        private static readonly int[] GridOrigin = { 48, 128 };

        /// <summary>
        ///     Plays the episodes the arguments ask for with the agent, or hands the env to the keyboard, or records the
        ///     docs page's media.
        /// </summary>
        /// <param name="args">
        ///     <c>--seed N</c> starts from reset seed N; <c>--episodes N</c> plays N episodes on consecutive seeds and
        ///     prints their mean; <c>--render</c> shows them in a window; <c>--play</c> lets you play the env from the
        ///     keyboard instead, until Escape or the window's close button; <c>--record &lt;folder&gt;</c> records the
        ///     page's episode, seed 12 whatever <c>--seed</c> says (<see cref="Record"/>). Other arguments are ignored.
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
                    Console.Error.WriteLine("--record needs a folder, e.g. --record docs/environments/toy_text/media");
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
            // The plan comes from the env's own model of itself: the transition table of the unwrapped env.
            long[] plan = Plan(((TaxiEnv)env.Unwrapped).P);
            NDArray observation = env.Reset(seed: seed).Observation;
            int steps = 0;
            double total = 0;
            while (true) {
                StepResult<NDArray> step = env.Step(NDArray.Scalar(plan[State(observation)]));
                observation = step.Observation;
                steps++;
                total += step.Reward;
                if (step.Terminated || step.Truncated)
                    return (steps, total, step.Terminated ? "delivered the passenger" : "truncated by the time limit");
            }
        }

        /// <summary>
        ///     Value iteration over the env's own transition table: the best action in every state, for a discount of
        ///     0.99, computed once per episode before the first step.
        /// </summary>
        /// <param name="p">The transition table <c>P</c>: for each state and action, the possible transitions.</param>
        /// <returns>
        ///     The greedy action per state; ties go to the lowest action, as NumPy's argmax would choose.
        /// </returns>
        private static long[] Plan(IReadOnlyDictionary<long, IReadOnlyDictionary<long, IReadOnlyList<ToyTextTransition>>> p) {
            const double gamma = 0.99;
            int states = p.Count, actions = p[0].Count;
            var v = new double[states];
            // The expected discounted return of an action: no value is carried past a terminal transition.
            double Q(int s, int a) {
                double q = 0;
                foreach (ToyTextTransition t in p[s][a])
                    q += t.Probability * (t.Reward + (t.Terminated ? 0 : gamma * v[t.NextState]));
                return q;
            }

            // Sweep until no state's value moves by more than 1e-10 (a few hundred sweeps at most).
            for (int sweep = 0; sweep < 10_000; sweep++) {
                double delta = 0;
                for (int s = 0; s < states; s++) {
                    double best = double.NegativeInfinity;
                    for (int a = 0; a < actions; a++)
                        best = Math.Max(best, Q(s, a));
                    delta = Math.Max(delta, Math.Abs(best - v[s]));
                    v[s] = best;
                }
                if (delta < 1e-10)
                    break;
            }
            var plan = new long[states];
            for (int s = 0; s < states; s++) {
                double best = double.NegativeInfinity;
                for (int a = 0; a < actions; a++) {
                    double q = Q(s, a);
                    // A strict improvement by more than rounding noise: equal actions keep the lowest index.
                    if (q > best + 1e-12) {
                        best = q;
                        plan[s] = a;
                    }
                }
            }

            return plan;
        }

        /// <summary>The state an observation names.</summary>
        /// <param name="observation">A 0-d int64 array.</param>
        /// <returns>The state's index.</returns>
        private static long State(NDArray observation) => Convert.ToInt64(observation.GetValue(), CultureInfo.InvariantCulture);

        /// <summary>
        ///     Records the seed-12 episode for Taxi's page on the docs site (DR-112): the whole episode as
        ///     <c>Taxi-v4.episode.mp4</c>, one frame per step at the env's 4 frames a second, through Gymnasium's
        ///     <see cref="RecordVideo{TObs, TAct}"/> and <see cref="FfmpegVideoWriter"/>; every step as
        ///     <c>Taxi-v4.episode.json</c>; the grid as the reset shows it, <c>Taxi-v4.board.png</c>, the passenger
        ///     waiting on their stand and the destination framed, which the page's anatomy draws the taxi's path over
        ///     (DR-131); and How it works' frame as <c>Taxi-v4.figure.png</c> with <c>Taxi-v4.figure.json</c>.
        /// </summary>
        /// <param name="folder">The output folder, <c>docs/environments/toy_text/media</c> for the site; created if missing.</param>
        /// <returns>The steps taken and the return.</returns>
        /// <exception cref="InvalidOperationException">ffmpeg could not be run, or wrote no video.</exception>
        /// <exception cref="IOException">A file cannot be written.</exception>
        /// <remarks>
        ///     <para>
        ///     Each JSON row is the state the agent acted on, decoded (the taxi's row and column, the passenger's place:
        ///     0 to 3 a stand, R, G, Y, B, or 4 in the taxi, and the destination), then whether the passenger is aboard,
        ///     the observation's state number, the action and the reward: <c>step, row, col, passenger, destination,
        ///     aboard, state, action, reward</c>. <c>aboard</c> repeats <c>passenger == 4</c> as 0 or 1, so the anatomy
        ///     can colour the path by it with two colours. The video's frame <c>t</c> shows the state of row <c>t</c>, and
        ///     its last frame the state after the last step, which the JSON keeps as <c>final</c>: the decoded state,
        ///     <c>aboard</c> and the state number, in the columns' order, so the page's readout can show them on the last
        ///     frame too.
        ///     </para>
        ///     <para>
        ///     RecordVideo names its file <c>Taxi-v4-episode-0.mp4</c> and writes it when the env closes; it records into
        ///     a scratch folder inside <paramref name="folder"/> and the file is moved to the page's name, so a recording
        ///     that fails halfway leaves the published video as it was.
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
                var taxi = (TaxiEnv)env.Unwrapped;
                long[] plan = Plan(taxi.P);
                NDArray observation = env.Reset(seed: DefaultSeed).Observation;
                // The board the anatomy draws on: the reset's frame, the passenger on their stand, the destination framed.
                SaveFrame(env, Path.Combine(folder, $"{Id}.board.png"));
                for (int t = 0; ; t++) {
                    long state = State(observation), action = plan[state];
                    // The frame now on screen shows the state the agent is about to act on (row t).
                    if (t == FigureStep)
                        SaveFigure(env, folder, taxi, state, action);
                    StepResult<NDArray> step = env.Step(NDArray.Scalar(action));
                    total += step.Reward;
                    (long row, long col, long passenger, long destination) = taxi.Decode(state);
                    rows.Add(new double[] { t, row, col, passenger, destination, passenger == 4 ? 1 : 0, state, action, step.Reward });
                    observation = step.Observation;
                    if (step.Terminated || step.Truncated)
                        break;
                }
                long last = State(observation);
                (long lastRow, long lastCol, long lastPassenger, long lastDestination) = taxi.Decode(last);
                final = new double[] { lastRow, lastCol, lastPassenger, lastDestination, lastPassenger == 4 ? 1 : 0, last };
                // RecordVideo writes the episode's video when it stops recording, which closing the env does.
                env.Close();
            }
            string recorded = Path.Combine(scratch, $"{Id}-episode-0.mp4");
            if (!File.Exists(recorded))
                throw new InvalidOperationException($"RecordVideo wrote no {recorded}; is ffmpeg on PATH?");
            File.Move(recorded, Path.Combine(folder, $"{Id}.episode.mp4"), overwrite: true);
            Directory.Delete(scratch, recursive: true);
            File.WriteAllText(Path.Combine(folder, $"{Id}.episode.json"), JsonSerializer.Serialize(new {
                id = Id, seed = DefaultSeed, steps = rows.Count, @return = total, fps = 4,
                agent = "value iteration over the env's transition table (examples/Taxi)",
                columns = new[] { "step", "row", "col", "passenger", "destination", "aboard", "state", "action", "reward" },
                rows,
                final,
            }) + "\n");
            return (rows.Count, total);
        }

        /// <summary>Saves the frame the env shows now as a PNG.</summary>
        /// <param name="env">The recording env, in <c>rgb_array</c> mode; rendering it adds nothing to the video.</param>
        /// <param name="path">The file to write.</param>
        /// <returns>The frame's width and height in pixels, which the figure's JSON records.</returns>
        /// <exception cref="IOException">The file cannot be written.</exception>
        private static (int Width, int Height) SaveFrame(Env<NDArray, NDArray> env, string path) {
            using RenderResult render = env.Render();
            NDArray frame = (NDArray)render;
            int height = (int)frame.shape[0], width = (int)frame.shape[1];
            using var image = Image.LoadPixelData<Rgb24>(frame.ToArray<byte>(), width, height);
            image.SaveAsPng(path);
            return (width, height);
        }

        /// <summary>
        ///     Saves How it works' frame (the frame the env shows now) as <c>Taxi-v4.figure.png</c>, and what its labels
        ///     point at as <c>Taxi-v4.figure.json</c>: the frame's size, the grid (its origin and 88x56 px cells), the
        ///     state and its decoding, the action the plan takes, the action mask, the four stands and the walls.
        /// </summary>
        /// <param name="env">The recording env, in <c>rgb_array</c> mode.</param>
        /// <param name="folder">The output folder.</param>
        /// <param name="taxi">The unwrapped env, for its decoding, stands, mask and map.</param>
        /// <param name="state">The state in the frame.</param>
        /// <param name="action">The action the agent takes from it.</param>
        /// <exception cref="IOException">A file cannot be written.</exception>
        /// <remarks>
        ///     The walls are read from the env's own map (<see cref="TaxiEnv.Map"/>), as the env reads them: a <c>|</c> at
        ///     <c>Map[1 + row][2 * col + 2]</c> stands east of the cell <c>(row, col)</c> and blocks the moves between it
        ///     and its east neighbour, which each wall entry lists as <c>[row, col]</c>. The mask is the env's
        ///     <see cref="TaxiEnv.ActionMask"/> of the state, the one the reset or step that reached it returned. The
        ///     frame is Gymnasium's 536 by 456 pixels, its grid starting at <see cref="GridOrigin"/>. Reference:
        ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/taxi.py#L417-L436">gymnasium/envs/toy_text/taxi.py</see>.
        /// </remarks>
        private static void SaveFigure(Env<NDArray, NDArray> env, string folder, TaxiEnv taxi, long state, long action) {
            (int width, int height) = SaveFrame(env, Path.Combine(folder, $"{Id}.figure.png"));
            (long row, long col, long passenger, long destination) = taxi.Decode(state);
            int rows = taxi.MaxRow + 1, columns = taxi.MaxCol + 1;
            var walls = new List<int[]>();
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns - 1; c++) {
                    if (TaxiEnv.Map[1 + r][2 * c + 2] == '|')
                        walls.Add(new[] { r, c });
                }
            }
            using NDArray mask = taxi.ActionMask(state);
            File.WriteAllText(Path.Combine(folder, $"{Id}.figure.json"), JsonSerializer.Serialize(new {
                id = Id, step = FigureStep, width, height,
                origin = GridOrigin, cell = new[] { taxi.CellSize.Width, taxi.CellSize.Height }, nrow = rows, ncol = columns,
                state, row, col, passenger, destination, action,
                mask = mask.ToArray<sbyte>().Select(m => (int)m).ToArray(),
                locs = taxi.Locs.Select(l => new[] { l.Row, l.Col }).ToArray(),
                walls,
            }, new JsonSerializerOptions { WriteIndented = true }) + "\n");
        }

        /// <summary>
        ///     The keys for <c>--play</c>: the arrow keys drive the taxi, P picks the passenger up, D drops them off.
        /// </summary>
        /// <returns>The map from key codes and keys to Taxi's actions.</returns>
        private static Dictionary<object, object?> Keys() => new() { [KeyCode.Down] = 0L, [KeyCode.Up] = 1L, [KeyCode.Right] = 2L, [KeyCode.Left] = 3L, ["p"] = 4L, ["d"] = 5L };

        /// <summary>
        ///     Hands the env to the keyboard, Gymnasium's play, until Escape or the window's close button.
        /// </summary>
        /// <param name="seed">
        ///     The seed of every reset: each new game starts as the last one did, as Gymnasium's play does.
        /// </param>
        /// <remarks>
        ///     The game is turn-based (<c>waitOnPlayer</c>): it steps only while a mapped key is held.
        /// </remarks>
        private static void PlayWithKeyboard(long seed) {
            // play reads the env's rgb_array frames and draws them into its own window.
            using IEnv env = Gymnasium.Make(Id, new { render_mode = "rgb_array" });
            Dictionary<object, object?> keys = Keys();
            Play.Run(env, keysToAction: keys, seed: seed, waitOnPlayer: true);
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
