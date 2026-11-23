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

namespace Gym.Examples.Acrobot {
    /// <summary>
    ///     Swings Acrobot-v1's tip over the line by torquing the second joint the way it already turns. <c>dotnet run
    ///     --project examples/Acrobot -c Release</c> plays one seeded episode headless and prints its steps and return;
    ///     <c>-- --render</c> plays it in a window, and <c>-- --play</c> hands the joint to the A and D keys.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Torque in the direction the second link swings feeds energy into the swing on every pass. Over seeds 0-9 the
    ///     tip crosses the line in 83 steps on average; seed 1 in 65.
    ///     </para>
    ///     <para>
    ///     The default seed, 1, is the episode the front-page showreel plays (<c>tools/showreel</c>, DR-74).
    ///     </para>
    /// </remarks>
    internal static class AcrobotProgram {
        /// <summary>The environment this example plays, by its registered id.</summary>
        private const string Id = "Acrobot-v1";

        /// <summary>The first reset seed when <c>--seed</c> is not given: the episode the showreel plays.</summary>
        private const long DefaultSeed = 1;

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
            NDArray observation = env.Reset(seed: seed).Observation;
            int steps = 0;
            double total = 0;
            while (true) {
                StepResult<NDArray> step = env.Step(Act(observation));
                observation = step.Observation;
                steps++;
                total += step.Reward;
                if (step.Terminated || step.Truncated)
                    return (steps, total, step.Terminated ? "swung the tip over the line" : "truncated by the time limit");
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

        /// <summary>The agent: torque the second joint the way it is already turning.</summary>
        /// <param name="observation">
        ///     The observation: the cosines and sines of both joint angles, then both angular velocities (index 5 is
        ///     the second joint's).
        /// </param>
        /// <returns>
        ///     +1 torque (2) when the second joint turns positively, else -1 torque (0), as a 0-d int64 array.
        /// </returns>
        private static NDArray Act(NDArray observation) => NDArray.Scalar(Values(observation)[5] > 0 ? 2L : 0L);

        /// <summary>The keys for <c>--play</c>: A applies -1 torque to the second joint, D +1.</summary>
        /// <returns>The map from keys to Acrobot's actions.</returns>
        private static Dictionary<object, object?> Keys() => new() { ["a"] = 0L, ["d"] = 2L };

        /// <summary>
        ///     Hands the env to the keyboard, Gymnasium's play, until Escape or the window's close button.
        /// </summary>
        /// <param name="seed">
        ///     The seed of every reset: each new game starts as the last one did, as Gymnasium's play does.
        /// </param>
        /// <remarks>
        ///     With no key held the joint is free: the no-op is action 1, no torque.
        /// </remarks>
        private static void PlayWithKeyboard(long seed) {
            // play reads the env's rgb_array frames and draws them into its own window.
            using IEnv env = Gymnasium.Make(Id, new { render_mode = "rgb_array" });
            Dictionary<object, object?> keys = Keys();
            object noop = 1L;
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
