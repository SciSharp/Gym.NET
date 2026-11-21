#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using Gym.Envs;
using Gym.Observations;
using Gym.Rendering.Avalonia;
using Gym.Spaces;
using NumSharp;

namespace Gym.Examples.Swimmer {
    /// <summary>
    ///     Makes Swimmer-v5's three-link swimmer swim forward with a linear policy trained by Augmented Random Search.
    ///     <c>dotnet run --project examples/Swimmer -c Release</c> plays one seeded episode headless and prints its
    ///     steps and return; <c>-- --render</c> plays it in a window.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <c>policy.json</c> holds the policy: the observation, normalized by the mean and scale seen in training,
    ///     weighted into one command per motor and clipped to the motor's range (<see cref="LinearPolicy"/>).
    ///     <c>tools/showreel</c>'s <c>train</c> command found it in 2 minutes on 32 threads (Augmented Random Search
    ///     V2-t, Mania, Guy and Recht 2018); it swims all 1,000 steps on seeds 0-4.
    ///     </para>
    ///     <para>
    ///     The default seed, 0, is the episode the front-page showreel plays (<c>tools/showreel</c>, DR-74).
    ///     </para>
    /// </remarks>
    internal static class Program {
        /// <summary>The environment this example plays, by its registered id.</summary>
        private const string Id = "Swimmer-v5";

        /// <summary>The first reset seed when <c>--seed</c> is not given: the episode the showreel plays.</summary>
        private const long DefaultSeed = 0;

        /// <summary>Plays the episodes the arguments ask for with the agent.</summary>
        /// <param name="args">
        ///     <c>--seed N</c> starts from reset seed N; <c>--episodes N</c> plays N episodes on consecutive seeds and
        ///     prints their mean; <c>--render</c> shows them in a window; <c>--play</c> is refused, with the reason
        ///     this env cannot be played from the keyboard. Other arguments are ignored.
        /// </param>
        /// <returns>
        ///     0 when the episodes were played, 2 when an option's value is not a whole number, or for <c>--play</c>.
        /// </returns>
        /// <exception cref="Gym.Exceptions.DependencyNotInstalled">
        ///     Not thrown in practice: <c>--render</c> sets Avalonia's viewer before the env first draws.
        /// </exception>
        private static int Main(string[] args) {
            if (!TryNumber(args, "--seed", DefaultSeed, out long seed) || !TryNumber(args, "--episodes", 1, out long episodes))
                return 2;
            bool render = args.Contains("--render");
            if (args.Contains("--play")) {
                Console.Error.WriteLine($"{Id}: its frames take about 0.3 s each to draw in software and a step is 40 ms of simulated time, so keyboard play would run about 7 times slower than real time; --render shows the agent instead.");
                return 2;
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
            LinearPolicy policy = LinearPolicy.Load(Path.Combine(AppContext.BaseDirectory, "policy.json"));
            double[] high = ActionHigh(env);
            NDArray observation = env.Reset(seed: seed).Observation;
            int steps = 0;
            double total = 0;
            while (true) {
                StepResult<NDArray> step = env.Step(policy.Act(observation, high));
                observation = step.Observation;
                steps++;
                total += step.Reward;
                if (step.Terminated || step.Truncated)
                    return (steps, total, step.Terminated ? "terminated" : "truncated by the time limit");
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
    ///     A linear policy over normalized observations, the kind Augmented Random Search trains:
    ///     <c>a_i = clip(sum_j W_ij (s_j - mean_j) / scale_j, -1, 1) * high_i</c>, rounded to float32.
    /// </summary>
    /// <remarks>
    ///     The arithmetic is the trainer's own, term by term in the same order (<c>tools/showreel/LinearPolicy.cs</c>):
    ///     a different summation order, a BLAS dot product say, rounds differently, and over a 1,000-step episode of
    ///     chaotic physics the trajectory drifts away from the one the policy was scored on.
    /// </remarks>
    internal sealed class LinearPolicy {
        /// <summary>The weights, row-major: one row of <see cref="_inputs"/> per motor.</summary>
        private readonly double[] _weights;

        /// <summary>The observation mean, subtracted before the weights.</summary>
        private readonly double[] _mean;

        /// <summary>The observation scale, divided by before the weights (a standard deviation plus 1e-8).</summary>
        private readonly double[] _scale;

        /// <summary>The observation size.</summary>
        private readonly int _inputs;

        /// <summary>The number of motors.</summary>
        private readonly int _outputs;

        /// <summary>Creates a policy from its parts.</summary>
        /// <param name="inputs">The observation size.</param>
        /// <param name="outputs">The number of motors.</param>
        /// <param name="weights">The row-major weights.</param>
        /// <param name="mean">The observation mean.</param>
        /// <param name="scale">The observation scale.</param>
        private LinearPolicy(int inputs, int outputs, double[] weights, double[] mean, double[] scale) {
            _inputs = inputs;
            _outputs = outputs;
            _weights = weights;
            _mean = mean;
            _scale = scale;
        }

        /// <summary>
        ///     Reads a policy file: <c>inputs</c>, <c>outputs</c>, the row-major <c>weights</c>, <c>mean</c> and
        ///     <c>scale</c>.
        /// </summary>
        /// <param name="path">The JSON file.</param>
        /// <returns>The policy.</returns>
        /// <exception cref="System.IO.IOException">The file cannot be read.</exception>
        /// <exception cref="JsonException">It is not JSON.</exception>
        /// <exception cref="KeyNotFoundException">A field is missing.</exception>
        /// <exception cref="InvalidOperationException">A field has the wrong JSON type.</exception>
        /// <exception cref="InvalidDataException">An array's length does not match the sizes.</exception>
        public static LinearPolicy Load(string path) {
            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
            JsonElement root = document.RootElement;
            int inputs = root.GetProperty("inputs").GetInt32(), outputs = root.GetProperty("outputs").GetInt32();
            double[] Read(string name, int length) {
                double[] values = root.GetProperty(name).EnumerateArray().Select(v => v.GetDouble()).ToArray();
                return values.Length == length ? values : throw new InvalidDataException($"{path}: {name} has {values.Length} numbers, {length} expected");
            }

            return new LinearPolicy(inputs, outputs, Read("weights", inputs * outputs), Read("mean", inputs), Read("scale", inputs));
        }

        /// <summary>The action for an observation.</summary>
        /// <param name="observation">The env's observation, read as float64.</param>
        /// <param name="high">The action space's upper bound per motor.</param>
        /// <returns>A new float32 array of one torque per motor; the caller disposes it.</returns>
        /// <exception cref="ArgumentException">The observation's size is not the policy's.</exception>
        public NDArray Act(NDArray observation, double[] high) {
            double[] s;
            using (NDArray values = observation.astype(NPTypeCode.Double))
                s = values.ToArray<double>();
            if (s.Length != _inputs)
                throw new ArgumentException($"the policy reads {_inputs} numbers, the observation has {s.Length}", nameof(observation));
            var action = new float[_outputs];
            for (int i = 0; i < _outputs; i++) {
                double sum = 0;
                for (int j = 0; j < _inputs; j++)
                    sum += _weights[i * _inputs + j] * ((s[j] - _mean[j]) / _scale[j]);
                // Comparisons rather than Math.Clamp, so a NaN passes through as the trainer's arithmetic passes it.
                double clipped = sum < -1.0 ? -1.0 : sum > 1.0 ? 1.0 : sum;
                action[i] = (float)(clipped * high[i]);
            }

            return np.array(action);
        }
    }
}
