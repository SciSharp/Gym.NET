#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Gym.Environments.Envs.Box2D;
using Gym.Envs;
using Gym.Observations;
using Gym.Rendering.Avalonia;
using Gym.Spaces;
using Gym.Utils;
using NumSharp;

namespace Gym.Examples.BipedalWalkerHardcore {
    /// <summary>
    ///     Walks BipedalWalkerHardcore-v3 with Gymnasium's gait for the plain course, which trips over the first stump.
    ///     <c>dotnet run --project examples/BipedalWalkerHardcore -c Release</c> plays one seeded episode headless and
    ///     prints its steps and return; <c>-- --render</c> plays it in a window, and <c>-- --play</c> hands the four
    ///     motors to the keyboard.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The gait is Gymnasium's walking <c>heuristic</c>, the same state machine as on the plain course. It was
    ///     written for that course: on the hardcore one it walks into the first obstacle and falls, at step 157 from
    ///     seed 10. Solving this course takes a learned policy that reads the lidar.
    ///     </para>
    ///     <para>
    ///     The default seed, 10, is the episode the front-page showreel plays (<c>tools/showreel</c>, DR-74).
    ///     </para>
    /// </remarks>
    internal static class Program {
        /// <summary>The environment this example plays, by its registered id.</summary>
        private const string Id = "BipedalWalkerHardcore-v3";

        /// <summary>The first reset seed when <c>--seed</c> is not given: the episode the showreel plays.</summary>
        private const long DefaultSeed = 10;

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
