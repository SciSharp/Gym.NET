using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json.Nodes;
using System.Threading.Tasks;
using Gym.MuJoCo;

namespace MujocoPoc.Play;

/// <summary>
///     Trains the autopilot with Augmented Random Search (Mania, Guy and Recht, 2018, "V2-t": a linear policy on observations
///     normalized by running statistics, the top directions only). Every rollout runs on the managed core, which steps
///     exactly as MuJoCo does, so the trained policy is a policy for Gymnasium's HalfCheetah-v5 itself.
/// </summary>
/// <remarks>
///     Each iteration perturbs the weights in <see cref="Settings.Directions"/> random directions, plays one episode with
///     each sign of each, keeps the <see cref="Settings.Top"/> directions whose better return is highest, and steps along
///     their return differences scaled by the returns' spread. Rollouts run in parallel, one game per rollout; the result is
///     deterministic for a seed, since every rollout's noise is drawn before the parallel part.
/// </remarks>
public static class Ars {
    /// <summary>The training hyperparameters; the defaults are the ARS paper's for HalfCheetah.</summary>
    /// <param name="Iterations">Update steps.</param>
    /// <param name="Directions">Random directions per step (two episodes each).</param>
    /// <param name="Top">Directions kept for the update.</param>
    /// <param name="StepSize">The update's step size, alpha.</param>
    /// <param name="Noise">The exploration noise, nu.</param>
    /// <param name="Horizon">Env steps per training episode.</param>
    /// <param name="Threads">Parallel rollouts.</param>
    /// <param name="Seed">The seed of the directions and the training resets.</param>
    /// <param name="EvalEvery">Iterations between evaluations of the current policy.</param>
    /// <param name="EvalSeeds">Gymnasium seeds 0..EvalSeeds-1 an evaluation plays.</param>
    public sealed record Settings(int Iterations = 600, int Directions = 32, int Top = 4, double StepSize = 0.02,
                                  double Noise = 0.03, int Horizon = 1000, int Threads = 8, int Seed = 1, int EvalEvery = 10,
                                  int EvalSeeds = 5);

    /// <summary>
    ///     Runs the training and returns the best policy an evaluation saw (ARS's returns are noisy, the last policy is not
    ///     always the best).
    /// </summary>
    /// <param name="model">The HalfCheetah model.</param>
    /// <param name="settings">The hyperparameters.</param>
    /// <param name="log">Receives one progress line per evaluation.</param>
    /// <returns>The best policy, with its score and the settings in <see cref="LinearPolicy.Info"/>.</returns>
    /// <exception cref="ArgumentException">A setting is out of range (Top above Directions, a non-positive count).</exception>
    public static LinearPolicy Train(MjModel model, Settings settings, TextWriter log) {
        if (settings.Top < 1 || settings.Top > settings.Directions || settings.Iterations < 1 || settings.Horizon < 1 || settings.Threads < 1)
            throw new ArgumentException("ARS needs 1 <= Top <= Directions and positive Iterations, Horizon and Threads", nameof(settings));
        int inputs = CheetahGame.ObservationSize, outputs = CheetahGame.ActionSize, dim = inputs * outputs;
        var policy = new LinearPolicy(inputs, outputs);
        var stats = new RunningStats(inputs);
        var rng = new Random(settings.Seed);
        var options = new ParallelOptions { MaxDegreeOfParallelism = settings.Threads };
        LinearPolicy best = policy;
        double bestScore = double.NegativeInfinity;
        long steps = 0;
        var clock = Stopwatch.StartNew();
        for (int iteration = 1; iteration <= settings.Iterations; iteration++) {
            // the normalization stays frozen during an iteration, as in ARS, and is refreshed after the update
            double[] mean = (double[])policy.Mean.Clone(), scale = (double[])policy.Scale.Clone();
            int n = settings.Directions;
            var deltas = new double[n][];
            for (int d = 0; d < n; d++) {
                deltas[d] = new double[dim];
                for (int i = 0; i < dim; i++)
                    deltas[d][i] = CheetahGame.Gaussian(rng);
            }
            int[] resetSeeds = Enumerable.Range(0, 2 * n).Select(_ => rng.Next()).ToArray();
            var returns = new double[2 * n];
            var partial = new RunningStats[2 * n];
            double[] weights = policy.Weights;
            Parallel.For(0, 2 * n, options, k => {
                double sign = k % 2 == 0 ? 1.0 : -1.0;
                var w = new double[dim];
                for (int i = 0; i < dim; i++)
                    w[i] = weights[i] + sign * settings.Noise * deltas[k / 2][i];
                var game = new CheetahGame(model);
                var visited = new RunningStats(inputs);
                double[] obs = game.Reset(new Random(resetSeeds[k]));
                Span<float> action = stackalloc float[outputs];
                for (int t = 0; t < settings.Horizon; t++) {
                    visited.Push(obs);
                    LinearPolicy.Act(w, mean, scale, obs, action);
                    game.Step(action);
                    obs = game.Observation();
                }
                returns[k] = game.EpisodeReturn;
                partial[k] = visited;
            });
            steps += 2L * n * settings.Horizon;
            foreach (RunningStats visited in partial)
                stats.Merge(visited);

            int[] top = Enumerable.Range(0, n).OrderByDescending(d => Math.Max(returns[2 * d], returns[2 * d + 1]))
                                  .Take(settings.Top).ToArray();
            double[] kept = top.SelectMany(d => new[] { returns[2 * d], returns[2 * d + 1] }).ToArray();
            double sigma = Math.Sqrt(kept.Select(r => (r - kept.Average()) * (r - kept.Average())).Average());
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

            if (iteration % settings.EvalEvery == 0 || iteration == settings.Iterations) {
                double[] scores = Evaluate(model, policy, settings.EvalSeeds, options);
                double score = scores.Average();
                log.WriteLine($"iteration {iteration,4}: {steps / 1e6,6:F2}M env steps, {clock.Elapsed.TotalSeconds,6:F1} s, "
                              + $"training returns mean {returns.Average(),8:F1} max {returns.Max(),8:F1}; "
                              + $"evaluation (Gymnasium seeds 0-{settings.EvalSeeds - 1}) {score,8:F1}");
                if (score > bestScore) {
                    bestScore = score;
                    best = Clone(policy);
                    best.Info = new JsonObject {
                        ["trainer"] = "Augmented Random Search V2-t (Mania, Guy, Recht 2018) on the managed HalfCheetah core",
                        ["iteration"] = iteration,
                        ["env_steps"] = steps,
                        ["evaluation_seeds"] = $"0-{settings.EvalSeeds - 1}",
                        ["evaluation_returns"] = new JsonArray(scores.Select(v => (JsonNode)v).ToArray()),
                        ["evaluation_mean"] = score,
                        ["settings"] = $"iterations {settings.Iterations}, directions {settings.Directions}, top {settings.Top}, "
                                       + $"step {settings.StepSize}, noise {settings.Noise}, horizon {settings.Horizon}, seed {settings.Seed}",
                    };
                }
            }
        }
        return best;
    }

    /// <summary>The policy's episode return from Gymnasium's resets for seeds 0..count-1, played in parallel.</summary>
    /// <param name="model">The model.</param>
    /// <param name="policy">The policy.</param>
    /// <param name="count">How many seeds.</param>
    /// <param name="options">The parallelism.</param>
    /// <returns>The return per seed.</returns>
    public static double[] Evaluate(MjModel model, LinearPolicy policy, int count, ParallelOptions options) {
        var scores = new double[count];
        Parallel.For(0, count, options, seed => scores[seed] = Play(model, policy, seed));
        return scores;
    }

    /// <summary>One full Gymnasium episode (1,000 env steps) of a policy from Gymnasium's reset for a seed.</summary>
    /// <param name="model">The model.</param>
    /// <param name="policy">The policy.</param>
    /// <param name="seed">The reset seed.</param>
    /// <returns>The episode return, summed as Gymnasium's caller sums it.</returns>
    public static double Play(MjModel model, LinearPolicy policy, int seed) {
        var game = new CheetahGame(model);
        double[] obs = game.Reset(seed);
        Span<float> action = stackalloc float[CheetahGame.ActionSize];
        for (int t = 0; t < CheetahGame.EpisodeSteps; t++) {
            policy.Act(obs, action);
            game.Step(action);
            obs = game.Observation();
        }
        return game.EpisodeReturn;
    }

    /// <summary>A copy of a policy's numbers (its info is not copied).</summary>
    /// <param name="policy">The policy.</param>
    /// <returns>The copy.</returns>
    private static LinearPolicy Clone(LinearPolicy policy) {
        var copy = new LinearPolicy(policy.Inputs, policy.Outputs);
        policy.Weights.CopyTo(copy.Weights, 0);
        policy.Mean.CopyTo(copy.Mean, 0);
        policy.Scale.CopyTo(copy.Scale, 0);
        return copy;
    }

    /// <summary>Running mean and sum of squared deviations per observation entry (Welford), mergeable across threads.</summary>
    private sealed class RunningStats {
        /// <summary>How many observations were pushed.</summary>
        public long Count;

        /// <summary>The running mean.</summary>
        public readonly double[] Mean;

        /// <summary>The running sum of squared deviations from the mean.</summary>
        public readonly double[] M2;

        /// <summary>Empty statistics.</summary>
        /// <param name="size">The observation size.</param>
        public RunningStats(int size) {
            Mean = new double[size];
            M2 = new double[size];
        }

        /// <summary>Adds one observation.</summary>
        /// <param name="x">The observation.</param>
        public void Push(ReadOnlySpan<double> x) {
            Count++;
            for (int j = 0; j < Mean.Length; j++) {
                double d = x[j] - Mean[j];
                Mean[j] += d / Count;
                M2[j] += d * (x[j] - Mean[j]);
            }
        }

        /// <summary>Adds another set of statistics (Chan et al.'s parallel update).</summary>
        /// <param name="other">The statistics to add.</param>
        public void Merge(RunningStats other) {
            if (other.Count == 0)
                return;
            long total = Count + other.Count;
            for (int j = 0; j < Mean.Length; j++) {
                double d = other.Mean[j] - Mean[j];
                Mean[j] += d * other.Count / total;
                M2[j] += other.M2[j] + d * d * Count * other.Count / total;
            }
            Count = total;
        }
    }
}
