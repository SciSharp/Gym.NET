#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Gym.Environments.Envs.ToyText;
using Gym.Envs;
using Gym.Observations;
using Gym.Rendering;
using Gym.Rendering.Avalonia;
using Gym.Utils;
using NumSharp;

namespace Gym.Examples.Taxi {
    /// <summary>
    ///     Picks up and delivers Taxi-v4's passenger by value iteration over the env's own transition table. <c>dotnet
    ///     run --project examples/Taxi -c Release</c> plays one seeded episode headless and prints its steps and
    ///     return; <c>-- --render</c> plays it in a window, and <c>-- --play</c> lets you play it with the keyboard.
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
    internal static class Program {
        /// <summary>The environment this example plays, by its registered id.</summary>
        private const string Id = "Taxi-v4";

        /// <summary>The first reset seed when <c>--seed</c> is not given: the episode the showreel plays.</summary>
        private const long DefaultSeed = 12;

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
        ///     The arrays a reset or a step returns belong to the caller (D19): each result is released once the agent
        ///     has acted on its observation, and the last one when the episode ends.
        /// </remarks>
        private static (int Steps, double Return, string Ending) PlayEpisode(long seed, bool render) {
            // make's stack: TimeLimit over OrderEnforcing over PassiveEnvChecker over the env. The render mode is a
            // constructor argument, fixed for the env's life, as in Gymnasium.
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>(Id, render ? new { render_mode = "human" } : null);
            // The plan comes from the env's own model of itself: the transition table of the unwrapped env.
            long[] plan = Plan(((TaxiEnv)env.Unwrapped).P);
            ResetResult<NDArray> reset = env.Reset(seed: seed);
            // The result whose observation the agent reads next; a boxed copy shares its arrays, so disposing it frees them.
            IDisposable held = reset;
            NDArray observation = reset.Observation;
            int steps = 0;
            double total = 0;
            try {
                while (true) {
                    StepResult<NDArray> step;
                    using (NDArray action = NDArray.Scalar(plan[State(observation)]))
                        step = env.Step(action);
                    // The agent is done with the previous observation: release it with its info, and hold the new result.
                    held.Dispose();
                    held = step;
                    observation = step.Observation;
                    steps++;
                    total += step.Reward;
                    if (step.Terminated || step.Truncated)
                        return (steps, total, step.Terminated ? "delivered the passenger" : "truncated by the time limit");
                }
            } finally {
                held.Dispose();
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
