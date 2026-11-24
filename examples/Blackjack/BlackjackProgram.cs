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
using Gym.Rendering.Avalonia;
using Gym.Utils;
using Gym.Wrappers;
using NumSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Gym.Examples.Blackjack {
    /// <summary>
    ///     Plays Blackjack-v1 by basic strategy against Gymnasium's dealer, who stands on 17. <c>dotnet run --project
    ///     examples/Blackjack -c Release</c> plays one seeded episode headless and prints its steps and return; <c>--
    ///     --render</c> plays it in a window, <c>-- --play</c> deals you the hands, and <c>-- --record &lt;folder&gt;</c>
    ///     records the hand Blackjack's page on the docs site plays (DR-112, DR-128).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Basic strategy for these rules: hit a soft total below 18, and a soft 18 against a 9, a 10 or an ace; hit a
    ///     hard total below 12; hit 12 to 16 against a 7 or better or an ace; hit 12 against a 2 or a 3; otherwise
    ///     stick. One episode is one hand: <c>--episodes 1000</c> shows the strategy's long-run return.
    ///     </para>
    ///     <para>
    ///     The default seed, 40, is the episode the front-page showreel plays (<c>tools/showreel</c>, DR-74).
    ///     </para>
    /// </remarks>
    internal static class BlackjackProgram {
        /// <summary>The environment this example plays, by its registered id.</summary>
        private const string Id = "Blackjack-v1";

        /// <summary>The first reset seed when <c>--seed</c> is not given: the episode the showreel plays.</summary>
        private const long DefaultSeed = 40;

        /// <summary>
        ///     The hand Blackjack's page plays (DR-128), which <c>--record</c> records instead of <see cref="DefaultSeed"/>:
        ///     the longest hand of this strategy that wins on exactly 21 among seeds 0 to 1,999,999. Two 2s against the
        ///     dealer's ace, six hits (an ace, a 2, a 6, a 2, an ace, a 5) through a soft 15 and a soft 17 that turns hard,
        ///     21, a stick, and the dealer stands on a soft 17: 7 steps, 8 frames, return 1.
        /// </summary>
        /// <remarks>
        ///     A one-step hand, the showreel's seed 40, would make the page's video a single frame and its chart a point; the
        ///     env draws every card from its seeded generator, so choosing the seed chooses the hand and nothing is altered.
        /// </remarks>
        private const long PageSeed = 1_574_758;

        /// <summary>
        ///     The moment How it works annotates on Blackjack's page (DR-114, DR-125): the frame after this many steps of
        ///     the page's hand, where it shows every part of the observation at once: the player's soft 17, the dealer's
        ///     ace and the usable ace.
        /// </summary>
        private const int FigureStep = 2;

        /// <summary>
        ///     Gymnasium's table layout, in the frame's pixels: <c>SCREEN_SIZE</c>, the dealer's label, face-up card and
        ///     hidden card, the player's label, the top of the player's sum and of the usable-ace label (the env keeps
        ///     them private, as Gymnasium's module constants are not attributes of the env), and the cards' size, the big
        ///     digits' size and the gap between them, from Gymnasium's images. Reference:
        ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/blackjack.py#L10-L21">gymnasium/envs/toy_text/blackjack.py</see>.
        /// </summary>
        private static readonly object Layout = new {
            screen = new[] { 384, 376 }, dealerLabel = new[] { 40, 32 }, dealerCard = new[] { 96, 72 }, hiddenCard = new[] { 208, 72 },
            playerLabel = new[] { 40, 208 }, playerSumTop = 248, usableAceTop = 328, card = new[] { 80, 112 }, bigDigit = new[] { 36, 60 },
            bigDigitGap = 12, artPixel = 4,
        };

        /// <summary>
        ///     Plays the episodes the arguments ask for with the agent, or hands the env to the keyboard, or records the
        ///     docs page's media.
        /// </summary>
        /// <param name="args">
        ///     <c>--seed N</c> starts from reset seed N; <c>--episodes N</c> plays N episodes on consecutive seeds and
        ///     prints their mean; <c>--render</c> shows them in a window; <c>--play</c> lets you play the env from the
        ///     keyboard instead, until Escape or the window's close button; <c>--record &lt;folder&gt;</c> records the
        ///     page's hand, seed <see cref="PageSeed"/> whatever <c>--seed</c> says (<see cref="Record"/>). Other
        ///     arguments are ignored.
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
                    $"{Id}, seed {PageSeed}: {recordedSteps} steps, return {recordedReturn:F1}, recorded into {args[record + 1]}"));
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
                Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"Mean return over {episodes} episodes: {sum / episodes:F3}"));
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
            using Env<object?[], NDArray> env = Gymnasium.Make<object?[], NDArray>(Id, render ? new { render_mode = "human" } : null);
            (object?[] observation, Info info) = env.Reset(seed: seed);
            int steps = 0;
            double total = 0;
            while (true) {
                StepResult<object?[]> step = env.Step(Act(observation));
                observation = step.Observation;
                steps++;
                total += step.Reward;
                if (step.Terminated || step.Truncated)
                    return (steps, total, step.Terminated ? total > 0 ? "won" : total < 0 ? "lost" : "a draw" : "truncated by the time limit");
            }
        }

        /// <summary>The agent: basic strategy against a dealer who stands on 17.</summary>
        /// <param name="observation">
        ///     Gym.NET's tuple for Gymnasium's <c>(player_sum, dealer_card, usable_ace)</c>: three 0-d int64 arrays
        ///     (the dealer's ace is 1).
        /// </param>
        /// <returns>Hit (1) or stick (0), as a 0-d int64 array.</returns>
        private static NDArray Act(object?[] observation) {
            long sum = Convert.ToInt64(((NDArray)observation[0]!).GetValue(), CultureInfo.InvariantCulture);
            long dealer = Convert.ToInt64(((NDArray)observation[1]!).GetValue(), CultureInfo.InvariantCulture);
            bool soft = Convert.ToInt64(((NDArray)observation[2]!).GetValue(), CultureInfo.InvariantCulture) != 0;
            bool hit = soft
                ? sum < 18 || (sum == 18 && (dealer >= 9 || dealer == 1))
                : sum < 12 || (sum < 17 && (dealer >= 7 || dealer == 1)) || (sum == 12 && dealer <= 3);
            return NDArray.Scalar(hit ? 1L : 0L);
        }

        /// <summary>
        ///     Records the page's hand (<see cref="PageSeed"/>) for Blackjack's page on the docs site (DR-112, DR-128): the
        ///     whole hand as <c>Blackjack-v1.episode.mp4</c>, one frame per step at the env's 4 frames a second, through
        ///     Gymnasium's <see cref="RecordVideo{TObs, TAct}"/> and <see cref="FfmpegVideoWriter"/>; every step as
        ///     <c>Blackjack-v1.episode.json</c>; and How it works' frame as <c>Blackjack-v1.figure.png</c> with
        ///     <c>Blackjack-v1.figure.json</c>.
        /// </summary>
        /// <param name="folder">The output folder, <c>docs/environments/toy_text/media</c> for the site; created if missing.</param>
        /// <returns>The steps taken and the return.</returns>
        /// <exception cref="InvalidOperationException">ffmpeg could not be run, or wrote no video.</exception>
        /// <exception cref="IOException">A file cannot be written.</exception>
        /// <remarks>
        ///     <para>
        ///     Each JSON row is the observation the agent acted on (the player's sum, the dealer's face-up card, 1 for an
        ///     ace, and the usable ace), then its action, 1 to hit and 0 to stick, and the reward: <c>step, sum, dealer,
        ///     usable, action, reward</c>. The video's frame <c>t</c> shows the observation of row <c>t</c>, and its last
        ///     frame the observation after the stick, which the JSON keeps as <c>final</c>. Blackjack keeps the time-series
        ///     anatomy (DR-131), so there is no board; the cards themselves, which the frame never shows (it shows the
        ///     player's sum), go into the figure's JSON.
        ///     </para>
        ///     <para>
        ///     RecordVideo names its file <c>Blackjack-v1-episode-0.mp4</c> and writes it when the env closes; it records
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
            using (Env<object?[], NDArray> made = Gymnasium.Make<object?[], NDArray>(Id, RenderMode.RgbArray))
            using (var env = new RecordVideo<object?[], NDArray>(made, scratch, episodeTrigger: _ => true, namePrefix: Id,
                       videoWriter: new FfmpegVideoWriter())) {
                var table = (BlackjackEnv)env.Unwrapped;
                (object?[] observation, Info info) = env.Reset(seed: PageSeed);
                for (int t = 0; ; t++) {
                    // The frame now on screen shows the observation the agent is about to act on (row t).
                    if (t == FigureStep)
                        SaveFigure(env, folder, table, observation);
                    using NDArray action = Act(observation);
                    long chosen = Number(action);
                    StepResult<object?[]> step = env.Step(action);
                    total += step.Reward;
                    rows.Add(new double[] { t, Number(observation[0]), Number(observation[1]), Number(observation[2]), chosen, step.Reward });
                    observation = step.Observation;
                    if (step.Terminated || step.Truncated)
                        break;
                }
                final = new double[] { Number(observation[0]), Number(observation[1]), Number(observation[2]) };
                // RecordVideo writes the episode's video when it stops recording, which closing the env does.
                env.Close();
            }
            string recorded = Path.Combine(scratch, $"{Id}-episode-0.mp4");
            if (!File.Exists(recorded))
                throw new InvalidOperationException($"RecordVideo wrote no {recorded}; is ffmpeg on PATH?");
            File.Move(recorded, Path.Combine(folder, $"{Id}.episode.mp4"), overwrite: true);
            Directory.Delete(scratch, recursive: true);
            File.WriteAllText(Path.Combine(folder, $"{Id}.episode.json"), JsonSerializer.Serialize(new {
                id = Id, seed = PageSeed, steps = rows.Count, @return = total, fps = 4,
                agent = "basic strategy (examples/Blackjack)",
                columns = new[] { "step", "sum", "dealer", "usable", "action", "reward" },
                rows,
                final,
            }) + "\n");
            return (rows.Count, total);
        }

        /// <summary>
        ///     Saves How it works' frame (the frame the env shows now) as <c>Blackjack-v1.figure.png</c>, and what its
        ///     labels point at as <c>Blackjack-v1.figure.json</c>: the frame's size, the observation, the cards behind it
        ///     (the player's hand and the dealer's two, the second face down), the face-up card's name, and Gymnasium's
        ///     table layout (<see cref="Layout"/>).
        /// </summary>
        /// <param name="env">The recording env, in <c>rgb_array</c> mode; rendering it adds nothing to the video.</param>
        /// <param name="folder">The output folder.</param>
        /// <param name="table">The unwrapped env, for the hands and the face-up card's name.</param>
        /// <param name="observation">The observation the frame shows.</param>
        /// <exception cref="IOException">A file cannot be written.</exception>
        /// <remarks>
        ///     The hands are the env's own lists (<see cref="BlackjackEnv.Player"/>, <see cref="BlackjackEnv.Dealer"/>),
        ///     copied at the moment of the frame: the player's keeps growing with each hit.
        /// </remarks>
        private static void SaveFigure(Env<object?[], NDArray> env, string folder, BlackjackEnv table, object?[] observation) {
            using RenderResult render = env.Render();
            NDArray frame = (NDArray)render;
            int height = (int)frame.shape[0], width = (int)frame.shape[1];
            using (var image = Image.LoadPixelData<Rgb24>(frame.ToArray<byte>(), width, height))
                image.SaveAsPng(Path.Combine(folder, $"{Id}.figure.png"));
            File.WriteAllText(Path.Combine(folder, $"{Id}.figure.json"), JsonSerializer.Serialize(new {
                id = Id, seed = PageSeed, step = FigureStep, width, height,
                observation = observation.Select(o => Number((NDArray)o!)).ToArray(),
                player = table.Player!.ToArray(), dealer = table.Dealer!.ToArray(),
                faceUp = table.DealerTopCardValueStr + table.DealerTopCardSuit,
                layout = Layout,
            }, new JsonSerializerOptions { WriteIndented = true }) + "\n");
        }

        /// <summary>A 0-d int64 array's value, or an observation item's.</summary>
        /// <param name="value">The array, or the <see langword="object"/> holding it (an item of the observation tuple).</param>
        /// <returns>Its value.</returns>
        /// <exception cref="InvalidCastException">The value is not an <see cref="NDArray"/>.</exception>
        private static long Number(object? value) => Convert.ToInt64(((NDArray)value!).GetValue(), CultureInfo.InvariantCulture);

        /// <summary>The keys for <c>--play</c>: H hits, S sticks.</summary>
        /// <returns>The map from keys to Blackjack's actions.</returns>
        private static Dictionary<object, object?> Keys() => new() { ["h"] = 1L, ["s"] = 0L };

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
