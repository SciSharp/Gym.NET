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

namespace Gym.Examples.CliffWalking {
    /// <summary>
    ///     Walks CliffWalking-v1 to the goal along the cliff's edge, by value iteration over the env's own transition
    ///     table. <c>dotnet run --project examples/CliffWalking -c Release</c> plays one seeded episode headless and
    ///     prints its steps and return; <c>-- --render</c> plays it in a window, <c>-- --play</c> lets you play it
    ///     with the keyboard, and <c>-- --record &lt;folder&gt;</c> records the episode for Cliff Walking's page on the
    ///     docs site (DR-112).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Every step costs 1 and stepping off the cliff costs 100 and sends the walker back to the start. The shortest
    ///     safe path runs along the cliff's edge, which value iteration over the transition table <c>P</c> finds: 13
    ///     steps.
    ///     </para>
    ///     <para>
    ///     The default seed, 0, is the episode the front-page showreel plays (<c>tools/showreel</c>, DR-74).
    ///     </para>
    /// </remarks>
    internal static class CliffWalkingProgram {
        /// <summary>The environment this example plays, by its registered id.</summary>
        private const string Id = "CliffWalking-v1";

        /// <summary>The first reset seed when <c>--seed</c> is not given: the episode the showreel plays.</summary>
        private const long DefaultSeed = 0;

        /// <summary>
        ///     The moment How it works annotates on Cliff Walking's page (DR-114, DR-125): the frame after this many steps
        ///     of the seed-0 episode, the walker halfway along the cliff's edge on cell 29, where each of its four moves
        ///     costs 1 but the one down, which drops it off the cliff for 100 and puts it back on the start.
        /// </summary>
        private const int FigureStep = 6;

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
            long[] plan = Plan(((CliffWalkingEnv)env.Unwrapped).P);
            NDArray observation = env.Reset(seed: seed).Observation;
            int steps = 0;
            double total = 0;
            while (true) {
                StepResult<NDArray> step = env.Step(NDArray.Scalar(plan[State(observation)]));
                observation = step.Observation;
                steps++;
                total += step.Reward;
                if (step.Terminated || step.Truncated)
                    return (steps, total, step.Terminated ? "reached the goal" : "truncated by the time limit");
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
        ///     The cell an action aims at from a state, as Gymnasium's <c>POSITION_MAPPING</c> and
        ///     <c>_limit_coordinates</c> move on the grid: one cell up, right, down or left, held at the grid's edges. It
        ///     is the cell the walker lands on unless that cell is the cliff, which sends it back to the start instead.
        /// </summary>
        /// <param name="state">The state, <c>row * 12 + col</c>.</param>
        /// <param name="action">0 up, 1 right, 2 down, 3 left.</param>
        /// <param name="rows">The grid's rows, 4.</param>
        /// <param name="columns">The grid's columns, 12.</param>
        /// <returns>The aimed state.</returns>
        private static long Aimed(long state, long action, int rows, int columns) {
            long row = state / columns, col = state % columns;
            (long r, long c) = action switch {
                0 => (Math.Max(row - 1, 0), col),
                1 => (row, Math.Min(col + 1, columns - 1)),
                2 => (Math.Min(row + 1, rows - 1), col),
                _ => (row, Math.Max(col - 1, 0)),
            };
            return r * columns + c;
        }

        /// <summary>
        ///     Records the seed-0 episode for Cliff Walking's page on the docs site (DR-112): the whole episode as
        ///     <c>CliffWalking-v1.episode.mp4</c>, one frame per step at the env's 4 frames a second, through Gymnasium's
        ///     <see cref="RecordVideo{TObs, TAct}"/> and <see cref="FfmpegVideoWriter"/>; every step as
        ///     <c>CliffWalking-v1.episode.json</c>; the grid as the reset shows it, <c>CliffWalking-v1.board.png</c>,
        ///     which the page's anatomy draws the path over (DR-131); and How it works' frame as
        ///     <c>CliffWalking-v1.figure.png</c> with <c>CliffWalking-v1.figure.json</c>.
        /// </summary>
        /// <param name="folder">The output folder, <c>docs/environments/toy_text/media</c> for the site; created if missing.</param>
        /// <returns>The steps taken and the return.</returns>
        /// <exception cref="InvalidOperationException">ffmpeg could not be run, or wrote no video.</exception>
        /// <exception cref="IOException">A file cannot be written.</exception>
        /// <remarks>
        ///     <para>
        ///     Each JSON row is the state the agent acted on, as its row and column and as the observation's state number,
        ///     then its action and the reward: <c>step, row, col, state, action, reward</c>. A reward of -100 is a fall,
        ///     after which the next row starts on the start cell again. The video's frame <c>t</c> shows the state of row
        ///     <c>t</c>, and its last frame the state after the last step, which the JSON keeps as <c>final</c> (its row,
        ///     column and state).
        ///     </para>
        ///     <para>
        ///     RecordVideo names its file <c>CliffWalking-v1-episode-0.mp4</c> and writes it when the env closes; it
        ///     records into a scratch folder inside <paramref name="folder"/> and the file is moved to the page's name, so
        ///     a recording that fails halfway leaves the published video as it was. CliffWalking-v1 has no step limit, so
        ///     the episode ends only at the goal, which the plan reaches in 13 steps.
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
                var grid = (CliffWalkingEnv)env.Unwrapped;
                int columns = grid.Shape.Columns;
                long[] plan = Plan(grid.P);
                NDArray observation = env.Reset(seed: DefaultSeed).Observation;
                // The board the anatomy draws on: the reset's frame, the walker on the start.
                SaveFrame(env, Path.Combine(folder, $"{Id}.board.png"));
                for (int t = 0; ; t++) {
                    long state = State(observation), action = plan[state];
                    // The frame now on screen shows the state the agent is about to act on (row t).
                    if (t == FigureStep)
                        SaveFigure(env, folder, grid, state, action);
                    StepResult<NDArray> step = env.Step(NDArray.Scalar(action));
                    total += step.Reward;
                    rows.Add(new double[] { t, state / columns, state % columns, state, action, step.Reward });
                    observation = step.Observation;
                    if (step.Terminated || step.Truncated)
                        break;
                }
                long last = State(observation);
                final = new double[] { last / columns, last % columns, last };
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
                agent = "value iteration over the env's transition table (examples/CliffWalking)",
                columns = new[] { "step", "row", "col", "state", "action", "reward" },
                rows,
                final,
            }) + "\n");
            return (rows.Count, total);
        }

        /// <summary>Saves the frame the env shows now as a PNG.</summary>
        /// <param name="env">The recording env, in <c>rgb_array</c> mode; rendering it adds nothing to the video.</param>
        /// <param name="path">The file to write.</param>
        /// <exception cref="IOException">The file cannot be written.</exception>
        private static void SaveFrame(Env<NDArray, NDArray> env, string path) {
            using RenderResult render = env.Render();
            NDArray frame = (NDArray)render;
            int height = (int)frame.shape[0], width = (int)frame.shape[1];
            using var image = Image.LoadPixelData<Rgb24>(frame.ToArray<byte>(), width, height);
            image.SaveAsPng(path);
        }

        /// <summary>
        ///     Saves How it works' frame (the frame the env shows now) as <c>CliffWalking-v1.figure.png</c>, and what its
        ///     labels point at as <c>CliffWalking-v1.figure.json</c>: the grid (64 px cells), the walker's cell and the
        ///     action the plan takes there, each of the four moves (the cell it aims at, where it lands and its reward),
        ///     the start, the goal and the cliff's cells.
        /// </summary>
        /// <param name="env">The recording env, in <c>rgb_array</c> mode.</param>
        /// <param name="folder">The output folder.</param>
        /// <param name="grid">The unwrapped env, for its transition table and grid.</param>
        /// <param name="state">The walker's state in the frame.</param>
        /// <param name="action">The action the agent takes from it.</param>
        /// <exception cref="IOException">A file cannot be written.</exception>
        /// <remarks>
        ///     Everything is read from the env's own table rather than restated: the cliff is every cell a move aims at
        ///     whose outcome costs 100, and the goal every cell a terminal outcome lands on, so the labels mark what the
        ///     table does. The frame is Gymnasium's: 64-pixel cells, 768 by 256 pixels. Reference:
        ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/cliffwalking.py#L166-L197">gymnasium/envs/toy_text/cliffwalking.py</see>.
        /// </remarks>
        private static void SaveFigure(Env<NDArray, NDArray> env, string folder, CliffWalkingEnv grid, long state, long action) {
            SaveFrame(env, Path.Combine(folder, $"{Id}.figure.png"));
            (int rows, int columns) = grid.Shape;
            // Every (aimed cell, outcome) pair of the solid grid's table: one certain outcome per state and action.
            var outcomes = grid.P.SelectMany(byState => byState.Value.Select(byAction =>
                (Aimed: Aimed(byState.Key, byAction.Key, rows, columns), Outcome: byAction.Value.Single()))).ToList();
            File.WriteAllText(Path.Combine(folder, $"{Id}.figure.json"), JsonSerializer.Serialize(new {
                id = Id, step = FigureStep, width = grid.WindowSize.Width, height = grid.WindowSize.Height,
                cell = new[] { grid.CellSize.Width, grid.CellSize.Height }, nrow = rows, ncol = columns,
                state, action,
                moves = Enumerable.Range(0, grid.NA).Select(a => {
                    ToyTextTransition t = grid.P[state][a].Single();
                    return new { action = a, aimed = Aimed(state, a, rows, columns), next = t.NextState, reward = t.Reward };
                }).ToArray(),
                start = grid.StartStateIndex,
                goal = outcomes.Where(o => o.Outcome.Terminated).Select(o => o.Outcome.NextState).Distinct().Single(),
                cliff = outcomes.Where(o => o.Outcome.Reward <= -100).Select(o => o.Aimed).Distinct().OrderBy(s => s).ToArray(),
            }, new JsonSerializerOptions { WriteIndented = true }) + "\n");
        }

        /// <summary>The keys for <c>--play</c>: the arrow keys move the walker.</summary>
        /// <returns>The map from key codes and keys to CliffWalking's actions.</returns>
        private static Dictionary<object, object?> Keys() => new() { [KeyCode.Up] = 0L, [KeyCode.Right] = 1L, [KeyCode.Down] = 2L, [KeyCode.Left] = 3L };

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
