using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json.Nodes;
using System.Threading;
using System.Threading.Tasks;
using Gym.Envs;
using Gym.Observations;
using NumSharp;

namespace Gym.Showreel;

/// <summary>
///     Augmented Random Search, V2-t (Mania, Guy and Recht, "Simple random search provides a competitive approach to
///     reinforcement learning", 2018): trains a <see cref="LinearPolicy"/> on any registered env with a Box action space,
///     through <c>Gymnasium.Make</c>, rollouts in parallel. The showreel's MuJoCo policies come from it.
/// </summary>
/// <remarks>
///     <para>
///     Each iteration perturbs the weights along <c>N</c> random directions both ways, plays one episode per
///     perturbation, keeps the <c>b</c> directions whose better side scored highest, and steps along their return
///     differences scaled by the step size over the kept returns' standard deviation. Observations are normalized by the
///     running mean and standard deviation of every state visited, frozen during an iteration. As in the paper, the
///     locomotion envs train with their survival bonus subtracted (otherwise standing still is a strong local optimum);
///     the evaluation that picks the best snapshot always uses the env's true return, on Gymnasium seeds 0 to 4.
///     </para>
///     <para>
///     Deterministic for a seed: the directions and reset seeds come from one seeded generator, and the parallel
///     rollouts write to fixed slots. Each worker thread keeps its own env (envs are not thread-safe). Training stops at
///     the iteration count or the time budget, whichever comes first, and returns the best evaluated snapshot.
///     </para>
/// </remarks>
internal static class Ars {
    /// <summary>One env's training settings.</summary>
    /// <param name="Directions">Directions sampled per iteration (the paper's N).</param>
    /// <param name="Top">Directions kept for the update (b).</param>
    /// <param name="StepSize">The step size (alpha).</param>
    /// <param name="Noise">The perturbation's standard deviation (nu).</param>
    /// <param name="AliveBonus">The per-step survival bonus subtracted from the training reward (0 to keep it).</param>
    /// <param name="Minutes">The time budget.</param>
    /// <param name="Iterations">The most iterations.</param>
    /// <param name="Seed">The generator's seed.</param>
    /// <param name="EvalEvery">Iterations between evaluations.</param>
    internal sealed record Settings(int Directions, int Top, double StepSize, double Noise, double AliveBonus, double Minutes, int Iterations = 100_000,
        int Seed = 1, int EvalEvery = 10);

    /// <summary>Seeds the evaluation plays (Gymnasium's 0 to 4).</summary>
    private const int EvalSeeds = 5;

    /// <summary>
    ///     The settings per id: the paper's V2-t hyperparameters for the locomotion envs it reports (Swimmer, Hopper,
    ///     HalfCheetah, Walker2d, Ant, Humanoid; Humanoid with fewer directions so an iteration fits the budget), and
    ///     comparable ones for the envs it does not.
    /// </summary>
    /// <param name="id">The env id.</param>
    /// <returns>The settings.</returns>
    /// <exception cref="ArgumentException">An id the showreel does not train.</exception>
    public static Settings For(string id) => id switch {
        "InvertedDoublePendulum-v5" => new(16, 8, 0.01, 0.025, 0, Minutes: 2),
        "Swimmer-v5" => new(1, 1, 0.02, 0.01, 0, Minutes: 2),
        "HalfCheetah-v5" => new(32, 4, 0.02, 0.03, 0, Minutes: 8),
        "Hopper-v5" => new(8, 4, 0.01, 0.025, 1.0, Minutes: 5),
        "Walker2d-v5" => new(40, 30, 0.03, 0.025, 1.0, Minutes: 8),
        "Ant-v5" => new(60, 20, 0.015, 0.025, 1.0, Minutes: 10),
        // Humanoid keeps its survival bonus, unlike the paper: with it removed the policy still fell within ~100 steps after
        // 5M steps (the paper needs on the order of 100M), while keeping it teaches standing first, which is what a clip needs.
        "Humanoid-v5" => new(96, 48, 0.02, 0.0075, 0, Minutes: 12),
        "HumanoidStandup-v5" => new(32, 16, 0.02, 0.0075, 0, Minutes: 6),
        _ => throw new ArgumentException($"no ARS settings for '{id}'", nameof(id)),
    };

    /// <summary>Trains a policy for an id.</summary>
    /// <param name="id">The registered id (a Box action space).</param>
    /// <param name="settings">The settings.</param>
    /// <param name="log">Progress lines, one per evaluation.</param>
    /// <returns>The best evaluated snapshot, its <see cref="LinearPolicy.Info"/> filled with the provenance.</returns>
    /// <exception cref="ArgumentException">Inconsistent settings (Top outside 1..Directions).</exception>
    public static LinearPolicy Train(string id, Settings settings, TextWriter log) {
        if (settings.Top < 1 || settings.Top > settings.Directions)
            throw new ArgumentException("ARS needs 1 <= Top <= Directions", nameof(settings));
        using var envs = new ThreadLocal<IEnv>(() => Gymnasium.Make(id), trackAllValues: true);
        try {
            IEnv probe = envs.Value!;
            double[] high = LinearPolicy.ActionHigh(probe);
            int inputs = Observation(probe.Reset(seed: 0).Observation).Length, outputs = high.Length, dim = inputs * outputs;
            var policy = new LinearPolicy(inputs, outputs);
            var stats = new RunningStats(inputs);
            var rng = new Random(settings.Seed);
            LinearPolicy best = policy.Clone();
            double bestScore = double.NegativeInfinity;
            long steps = 0;
            var clock = Stopwatch.StartNew();
            for (int iteration = 1; iteration <= settings.Iterations; iteration++) {
                // Normalization is frozen during an iteration and refreshed after the update, as in the paper.
                double[] mean = (double[])policy.Mean.Clone(), scale = (double[])policy.Scale.Clone(), weights = policy.Weights;
                int n = settings.Directions;
                var deltas = new double[n][];
                for (int d = 0; d < n; d++) {
                    deltas[d] = new double[dim];
                    for (int i = 0; i < dim; i++)
                        deltas[d][i] = Gaussian(rng);
                }
                int[] resetSeeds = Enumerable.Range(0, 2 * n).Select(_ => rng.Next()).ToArray();
                var returns = new double[2 * n];
                var visits = new RunningStats[2 * n];
                var counts = new int[2 * n];
                Parallel.For(0, 2 * n, k => {
                    double sign = k % 2 == 0 ? 1.0 : -1.0;
                    var w = new double[dim];
                    for (int i = 0; i < dim; i++)
                        w[i] = weights[i] + sign * settings.Noise * deltas[k / 2][i];
                    visits[k] = new RunningStats(inputs);
                    (returns[k], _, counts[k]) = Rollout(envs.Value!, high, w, mean, scale, resetSeeds[k], settings.AliveBonus, visits[k]);
                });
                steps += counts.Sum();
                foreach (RunningStats v in visits)
                    stats.Merge(v);
                int[] top = Enumerable.Range(0, n).OrderByDescending(d => Math.Max(returns[2 * d], returns[2 * d + 1])).Take(settings.Top).ToArray();
                double[] kept = top.SelectMany(d => new[] { returns[2 * d], returns[2 * d + 1] }).ToArray();
                double keptMean = kept.Average();
                double sigma = Math.Sqrt(kept.Select(r => (r - keptMean) * (r - keptMean)).Average());
                // All kept returns equal (a dead start): no gradient information, so the step is skipped, not divided by 0.
                if (sigma > 1e-12) {
                    double rate = settings.StepSize / (settings.Top * sigma);
                    for (int i = 0; i < dim; i++) {
                        double g = 0;
                        foreach (int d in top)
                            g += (returns[2 * d] - returns[2 * d + 1]) * deltas[d][i];
                        weights[i] += rate * g;
                    }
                }
                for (int j = 0; j < inputs; j++) {
                    policy.Mean[j] = stats.Mean[j];
                    policy.Scale[j] = stats.Count > 1 ? Math.Sqrt(stats.M2[j] / (stats.Count - 1)) + 1e-8 : 1.0;
                }
                bool timeUp = clock.Elapsed.TotalMinutes >= settings.Minutes;
                if (iteration % settings.EvalEvery != 0 && !timeUp && iteration != settings.Iterations)
                    continue;
                double[] scores = Evaluate(envs, high, policy);
                double score = scores.Average();
                log.WriteLine($"{id} iteration {iteration,5}: {steps / 1e6,7:F2}M steps, {clock.Elapsed.TotalSeconds,6:F0} s, "
                              + $"training {returns.Average(),9:F1}, evaluation {score,9:F1}{(score > bestScore ? "  best" : "")}");
                if (score > bestScore) {
                    bestScore = score;
                    best = policy.Clone();
                    best.Info = new JsonObject {
                        ["trainer"] = "Augmented Random Search V2-t (Mania, Guy, Recht 2018), tools/showreel/Ars.cs",
                        ["env"] = id,
                        ["iteration"] = iteration,
                        ["env_steps"] = steps,
                        ["evaluation_seeds"] = $"0-{EvalSeeds - 1}",
                        ["evaluation_returns"] = new JsonArray(scores.Select(v => (JsonNode)v).ToArray()),
                        ["evaluation_mean"] = score,
                        ["settings"] = $"directions {settings.Directions}, top {settings.Top}, step {settings.StepSize}, noise {settings.Noise}, "
                                       + $"alive bonus removed {settings.AliveBonus}, seed {settings.Seed}",
                    };
                }
                if (timeUp)
                    break;
            }
            return best;
        } finally {
            foreach (IEnv env in envs.Values)
                env.Dispose();
        }
    }

    /// <summary>Plays the policy without noise on seeds 0 to 4 with the env's true reward.</summary>
    /// <param name="envs">The per-thread envs.</param>
    /// <param name="high">The action bounds.</param>
    /// <param name="policy">The policy.</param>
    /// <returns>One return per seed.</returns>
    private static double[] Evaluate(ThreadLocal<IEnv> envs, double[] high, LinearPolicy policy) {
        var scores = new double[EvalSeeds];
        Parallel.For(0, EvalSeeds, seed => (_, scores[seed], _) = Rollout(envs.Value!, high, policy.Weights, policy.Mean, policy.Scale, seed, 0, null));
        return scores;
    }

    /// <summary>Plays one episode to its end (termination or the registered step limit).</summary>
    /// <param name="env">The env (this thread's).</param>
    /// <param name="high">The action bounds.</param>
    /// <param name="w">The weights to play.</param>
    /// <param name="mean">The normalization mean.</param>
    /// <param name="scale">The normalization scale.</param>
    /// <param name="seed">The reset seed.</param>
    /// <param name="alive">The per-step bonus subtracted from the training return.</param>
    /// <param name="visited">Collects the observations seen, or null.</param>
    /// <returns>The training return (bonus removed), the true return, and the steps taken.</returns>
    private static (double Training, double True, int Steps) Rollout(IEnv env, double[] high, double[] w, double[] mean, double[] scale, int seed,
        double alive, RunningStats? visited) {
        double[] s;
        using (ResetResult<object?> reset = env.Reset(seed: seed))
            s = Observation(reset.Observation);
        double training = 0, total = 0;
        int t = 0;
        var action = new float[high.Length];
        while (true) {
            visited?.Push(s);
            LinearPolicy.Act(w, mean, scale, s, action, high);
            // A fresh array per step: the env must never see a buffer the next step overwrites.
            using NDArray a = np.array((float[])action.Clone());
            using StepResult<object?> step = env.Step(a);
            t++;
            total += step.Reward;
            training += step.Reward - alive;
            if (step.Terminated || step.Truncated)
                return (training, total, t);
            s = Observation(step.Observation);
        }
    }

    /// <summary>An observation's values as float64.</summary>
    /// <param name="observation">An <see cref="NDArray"/> observation.</param>
    /// <returns>The values.</returns>
    private static double[] Observation(object? observation) {
        using NDArray d = ((NDArray)observation!).astype(NPTypeCode.Double);
        return d.ToArray<double>();
    }

    /// <summary>A standard normal sample (Box-Muller).</summary>
    /// <param name="rng">The generator.</param>
    /// <returns>The sample.</returns>
    private static double Gaussian(Random rng) {
        double u = 1.0 - rng.NextDouble(), v = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u)) * Math.Cos(2.0 * Math.PI * v);
    }

    /// <summary>Welford's running mean and sum of squared deviations, mergeable (Chan's parallel update).</summary>
    private sealed class RunningStats {
        /// <summary>Samples seen.</summary>
        public long Count;

        /// <summary>The running mean.</summary>
        public readonly double[] Mean;

        /// <summary>The running sum of squared deviations from the mean.</summary>
        public readonly double[] M2;

        /// <summary>Empty statistics of a dimension.</summary>
        /// <param name="size">The dimension.</param>
        public RunningStats(int size) {
            Mean = new double[size];
            M2 = new double[size];
        }

        /// <summary>Adds one sample.</summary>
        /// <param name="x">The sample.</param>
        public void Push(ReadOnlySpan<double> x) {
            Count++;
            for (int j = 0; j < Mean.Length; j++) {
                double d = x[j] - Mean[j];
                Mean[j] += d / Count;
                M2[j] += d * (x[j] - Mean[j]);
            }
        }

        /// <summary>Folds another set of statistics into this one.</summary>
        /// <param name="other">The other statistics.</param>
        public void Merge(RunningStats other) {
            if (other.Count == 0)
                return;
            long n = Count + other.Count;
            for (int j = 0; j < Mean.Length; j++) {
                double d = other.Mean[j] - Mean[j];
                M2[j] += other.M2[j] + d * d * Count * other.Count / n;
                Mean[j] += d * other.Count / n;
            }
            Count = n;
        }
    }
}
