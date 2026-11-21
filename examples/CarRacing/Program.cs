#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Gym.Environments.Envs.Box2D;
using Gym.Envs;
using Gym.Observations;
using Gym.Rendering.Avalonia;
using Gym.Utils;
using NumSharp;

namespace Gym.Examples.CarRacing {
    /// <summary>
    ///     Drives CarRacing-v3 by pure pursuit on the track's centre line. <c>dotnet run --project examples/CarRacing
    ///     -c Release</c> plays one seeded episode headless and prints its steps and return; <c>-- --render</c> plays
    ///     it in a window, and <c>-- --play</c> hands the car to the W, A, S and D keys.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The agent reads the track and the car from the unwrapped env rather than from the 96x96 pixels: it aims at
    ///     the centre-line point five track tiles past the nearest one, steers by the angle to it, keeps the throttle
    ///     while the speed is under a limit that drops with that angle, and brakes into sharp bends. Seed 11 drives the
    ///     full 1,000 steps.
    ///     </para>
    ///     <para>
    ///     The default seed, 11, is the episode the front-page showreel plays (<c>tools/showreel</c>, DR-74).
    ///     </para>
    /// </remarks>
    internal static class Program {
        /// <summary>The environment this example plays, by its registered id.</summary>
        private const string Id = "CarRacing-v3";

        /// <summary>The first reset seed when <c>--seed</c> is not given: the episode the showreel plays.</summary>
        private const long DefaultSeed = 11;

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
                StepResult<NDArray> step = env.Step(Act(env));
                observation = step.Observation;
                steps++;
                total += step.Reward;
                if (step.Terminated || step.Truncated)
                    return (steps, total, step.Terminated ? total > 0 ? "finished the lap" : "left the playfield" : "truncated by the time limit");
            }
        }

        /// <summary>The agent: pure pursuit of a point on the centre line, five tiles ahead.</summary>
        /// <param name="env">The env; its unwrapped <see cref="CarRacingEnv"/> exposes the car and the track.</param>
        /// <returns>A float32 <c>[steer, gas, brake]</c>.</returns>
        private static NDArray Act(IEnv env) {
            var race = (CarRacingEnv)env.Unwrapped;
            var hull = race.Car!.Hull!;
            var p = hull.Position;
            double angle = hull.Angle;
            var track = race.Track;
            int nearest = 0;
            double best = double.MaxValue;
            for (int i = 0; i < track.Count; i++) {
                double dx0 = track[i].X - p.x, dy0 = track[i].Y - p.y, d = dx0 * dx0 + dy0 * dy0;
                if (d < best) {
                    best = d;
                    nearest = i;
                }
            }
            var target = track[(nearest + 5) % track.Count];
            double dx = target.X - p.x, dy = target.Y - p.y;
            // The hull's forward axis is its local +y: (-sin a, cos a) in world coordinates.
            double fx = -Math.Sin(angle), fy = Math.Cos(angle);
            double error = Math.Atan2(fx * dy - fy * dx, fx * dx + fy * dy);
            var v = hull.LinearVelocity;
            double speed = Math.Sqrt(v.x * v.x + v.y * v.y);
            float steer = (float)Math.Clamp(-error * 2.5, -1, 1);
            float gas = speed < 45 - 30 * Math.Min(1, Math.Abs(error)) ? 0.6f : 0f;
            float brake = Math.Abs(error) > 0.6 && speed > 30 ? 0.4f : 0f;
            return np.array(new[] { steer, gas, brake });
        }

        /// <summary>
        ///     The keys for <c>--play</c>, the map Gymnasium's own <c>play</c> documentation gives for CarRacing: W
        ///     gas, A and D steer, S brake, and the four two-key combinations.
        /// </summary>
        /// <returns>
        ///     The map from keys to float32 <c>[steer, gas, brake]</c> actions; the caller disposes them.
        /// </returns>
        private static Dictionary<object, object?> Keys() => new() {
            ["w"] = np.array(new[] { 0f, 0.7f, 0f }),
            ["a"] = np.array(new[] { -1f, 0f, 0f }),
            ["s"] = np.array(new[] { 0f, 0f, 1f }),
            ["d"] = np.array(new[] { 1f, 0f, 0f }),
            ["wa"] = np.array(new[] { -1f, 0.7f, 0f }),
            ["dw"] = np.array(new[] { 1f, 0.7f, 0f }),
            ["ds"] = np.array(new[] { 1f, 0f, 1f }),
            ["as"] = np.array(new[] { -1f, 0f, 1f }),
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
            NDArray noop = np.array(new float[3]);
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
