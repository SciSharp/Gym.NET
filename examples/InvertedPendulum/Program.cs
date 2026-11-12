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

namespace Gym.Examples.InvertedPendulum {
    /// <summary>
    ///     Balances InvertedPendulum-v5, MuJoCo's cart-pole, with linear state feedback. <c>dotnet run --project
    ///     examples/InvertedPendulum -c Release</c> plays one seeded episode headless and prints its steps and return;
    ///     <c>-- --render</c> plays it in a window, and <c>-- --play</c> hands the cart to the A and D keys.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The force is <c>u = 5.330 x + 8.469 theta + 1.109 x_dot + 0.634 theta_dot</c>, clipped to the motor's range
    ///     of ±3: four gains found by random search, which hold the pole for all 1,000 steps on seeds 0-9 with the cart
    ///     near the centre (mean |x| 0.003). A policy trained by Augmented Random Search balanced as well, but let the
    ///     cart drift to the end of its rail.
    ///     </para>
    ///     <para>
    ///     The default seed, 0, is the episode the front-page showreel plays (<c>tools/showreel</c>, DR-74).
    ///     </para>
    /// </remarks>
    internal static class Program {
        /// <summary>The environment this example plays, by its registered id.</summary>
        private const string Id = "InvertedPendulum-v5";

        /// <summary>The first reset seed when <c>--seed</c> is not given: the episode the showreel plays.</summary>
        private const long DefaultSeed = 0;

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
            ResetResult<NDArray> reset = env.Reset(seed: seed);
            // The result whose observation the agent reads next; a boxed copy shares its arrays, so disposing it frees them.
            IDisposable held = reset;
            NDArray observation = reset.Observation;
            int steps = 0;
            double total = 0;
            try {
                while (true) {
                    StepResult<NDArray> step;
                    using (NDArray action = Act(observation))
                        step = env.Step(action);
                    // The agent is done with the previous observation: release it with its info, and hold the new result.
                    held.Dispose();
                    held = step;
                    observation = step.Observation;
                    steps++;
                    total += step.Reward;
                    if (step.Terminated || step.Truncated)
                        return (steps, total, step.Terminated ? "the pole fell" : "truncated by the time limit");
                }
            } finally {
                held.Dispose();
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

        /// <summary>The agent: a force on the cart, linear in the state.</summary>
        /// <param name="observation">The observation <c>[x, theta, x_dot, theta_dot]</c>.</param>
        /// <returns>A float32 force in [-3, 3], as a one-element array.</returns>
        private static NDArray Act(NDArray observation) {
            double[] s = Values(observation);
            double u = 5.330 * s[0] + 8.469 * s[1] + 1.109 * s[2] + 0.634 * s[3];
            return np.array(new[] { (float)Math.Clamp(u, -3.0, 3.0) });
        }

        /// <summary>The keys for <c>--play</c>: A pushes the cart left at full force (-3), D right (+3).</summary>
        /// <returns>The map from keys to float32 forces; the caller disposes the actions.</returns>
        private static Dictionary<object, object?> Keys() => new() {
            ["a"] = np.array(new[] { -3f }),
            ["d"] = np.array(new[] { 3f }),
        };

        /// <summary>
        ///     Hands the env to the keyboard, Gymnasium's play, until Escape or the window's close button.
        /// </summary>
        /// <param name="seed">
        ///     The seed of every reset: each new game starts as the last one did, as Gymnasium's play does.
        /// </param>
        private static void PlayWithKeyboard(long seed) {
            // play reads the env's rgb_array frames and draws them into its own window.
            using IEnv env = Gymnasium.Make(Id, new { render_mode = "rgb_array" });
            Dictionary<object, object?> keys = Keys();
            NDArray noop = np.array(new[] { 0f });
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
