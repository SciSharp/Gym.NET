#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Gym.Envs;
using Gym.Observations;
using Gym.Rendering.Avalonia;
using Gym.Utils;
using NumSharp;

namespace Gym.Examples.Blackjack {
    /// <summary>
    ///     Plays Blackjack-v1 by basic strategy against Gymnasium's dealer, who stands on 17. <c>dotnet run --project
    ///     examples/Blackjack -c Release</c> plays one seeded episode headless and prints its steps and return; <c>--
    ///     --render</c> plays it in a window, and <c>-- --play</c> deals you the hands.
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
    internal static class Program {
        /// <summary>The environment this example plays, by its registered id.</summary>
        private const string Id = "Blackjack-v1";

        /// <summary>The first reset seed when <c>--seed</c> is not given: the episode the showreel plays.</summary>
        private const long DefaultSeed = 40;

        /// <summary>
        ///     Plays the episodes the arguments ask for with the agent, or hands the env to the keyboard.
        /// </summary>
        /// <param name="args">
        ///     <c>--seed N</c> starts from reset seed N; <c>--episodes N</c> plays N episodes on consecutive seeds and
        ///     prints their mean; <c>--render</c> shows them in a window; <c>--play</c> lets you play the env from the
        ///     keyboard instead, until Escape or the window's close button. Other arguments are ignored.
        /// </param>
        /// <returns>
        ///     0 when the episodes were played or the keyboard game ended, 2 when an option's value is not a whole
        ///     number.
        /// </returns>
        /// <exception cref="Gym.Exceptions.DependencyNotInstalled">
        ///     Not thrown in practice: <c>--render</c> and <c>--play</c> set Avalonia's viewer before the env first
        ///     draws.
        /// </exception>
        private static int Main(string[] args) {
            if (!TryNumber(args, "--seed", DefaultSeed, out long seed) || !TryNumber(args, "--episodes", 1, out long episodes))
                return 2;
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
        ///     The arrays a reset or a step returns belong to the caller (D19): each result is released once the agent
        ///     has acted on its observation, and the last one when the episode ends.
        /// </remarks>
        private static (int Steps, double Return, string Ending) PlayEpisode(long seed, bool render) {
            // make's stack: TimeLimit over OrderEnforcing over PassiveEnvChecker over the env. The render mode is a
            // constructor argument, fixed for the env's life, as in Gymnasium.
            using Env<object?[], NDArray> env = Gymnasium.Make<object?[], NDArray>(Id, render ? new { render_mode = "human" } : null);
            ResetResult<object?[]> reset = env.Reset(seed: seed);
            // The result whose observation the agent reads next; a boxed copy shares its arrays, so disposing it frees them.
            IDisposable held = reset;
            object?[] observation = reset.Observation;
            int steps = 0;
            double total = 0;
            try {
                while (true) {
                    StepResult<object?[]> step;
                    using (NDArray action = Act(observation))
                        step = env.Step(action);
                    // The agent is done with the previous observation: release it with its info, and hold the new result.
                    held.Dispose();
                    held = step;
                    observation = step.Observation;
                    steps++;
                    total += step.Reward;
                    if (step.Terminated || step.Truncated)
                        return (steps, total, step.Terminated ? total > 0 ? "won" : total < 0 ? "lost" : "a draw" : "truncated by the time limit");
                }
            } finally {
                held.Dispose();
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
