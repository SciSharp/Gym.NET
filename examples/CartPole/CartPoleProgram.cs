#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Gym.Envs;
using Gym.Observations;
using Gym.Rendering.Avalonia;
using Gym.Utils;
using Gym.Vector;
using NumSharp;

namespace Gym.Examples.CartPole {
    /// <summary>
    ///     Learns to balance CartPole-v1 with the cross-entropy method over linear policies, then plays the learned policy
    ///     for one episode: <c>dotnet run --project examples/CartPole -c Release</c>, with <c>-- --render</c> to watch that
    ///     episode in a window. <c>-- --controller</c> plays a hand-written controller instead of learning one (the cart
    ///     the front-page showreel shows), and <c>-- --play</c> hands the cart to the keyboard.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A policy is <see cref="ParameterCount"/> numbers: a weight for each of the four observation entries (cart
    ///     position, cart velocity, pole angle, pole angular velocity) and a bias. It pushes the cart right when the
    ///     weighted sum is positive. Each generation draws <see cref="Population"/> policies from a Gaussian, plays one
    ///     episode with each, and refits the Gaussian to the <see cref="EliteCount"/> best. The Gaussian's mean is the
    ///     learned policy; training stops once it is solved by Gymnasium's measure, an average return of at least the
    ///     reward threshold CartPole-v1 is registered with (475 of a possible 500) over 100 episodes it has not seen.
    ///     </para>
    ///     <para>
    ///     The whole population plays at once in one vector env: <c>Gymnasium.MakeVec("CartPole-v1", n)</c> builds
    ///     CartPole's own vectorized env, which steps every cart in one call. Every draw comes from a seeded generator, so
    ///     two runs print the same numbers on the same platform.
    ///     </para>
    /// </remarks>
    internal static class CartPoleProgram {
        /// <summary>Policy parameters: four observation weights and a bias.</summary>
        private const int ParameterCount = 5;

        /// <summary>The observation entries a policy weighs (CartPole's observation space is a <c>Box</c> of shape <c>(4,)</c>).</summary>
        private const int ObservationSize = 4;

        /// <summary>
        ///     Policies drawn per generation, which is also the number of carts the vector env runs side by side and the
        ///     number of episodes the learned policy is judged on (Gymnasium's "solved" averages 100).
        /// </summary>
        private const int Population = 100;

        /// <summary>The best policies of a generation, which the next generation's Gaussian is fitted to.</summary>
        private const int EliteCount = 10;

        /// <summary>Generations to try before giving up; the method usually solves CartPole in fewer than ten.</summary>
        private const int MaxGenerations = 50;

        /// <summary>
        ///     Added to every standard deviation after each refit, so the search keeps exploring instead of collapsing onto
        ///     the first elite it finds (a Gaussian refitted to a handful of points shrinks fast).
        /// </summary>
        private const double ExtraNoise = 0.05;

        /// <summary>The seed of the generator the policies are drawn from.</summary>
        private const long SearchSeed = 0;

        /// <summary>
        ///     The first reset seed of the episodes that judge the learned policy, far from the training generations' seeds
        ///     (0, 1, 2, ...) so it is judged on starting states it was not selected on.
        /// </summary>
        private const long JudgeSeed = 1_000_000;

        /// <summary>The reset seed of the episode played at the end, again one neither training nor judging used.</summary>
        private const long PlaySeed = 2_000_000;

        /// <summary>The reset seed of the hand-written controller's episode: the one the front-page showreel plays (DR-74).</summary>
        private const long ControllerSeed = 3;

        /// <summary>Trains a policy, then plays one episode with it; or plays the hand-written controller; or the keyboard.</summary>
        /// <param name="args">
        ///     <c>--render</c> plays the final episode in a window (Avalonia); <c>--controller</c> plays the hand-written
        ///     controller instead of training; <c>--play</c> hands the cart to the keyboard (A and D) until Escape or the
        ///     window's close button. Other arguments are ignored.
        /// </param>
        /// <returns>
        ///     The process exit code: 0 when training solved CartPole-v1 (or the controller or the keyboard game ran), 1 when
        ///     <see cref="MaxGenerations"/> ran out first.
        /// </returns>
        /// <exception cref="Gym.Exceptions.DependencyNotInstalled">
        ///     Not thrown in practice: <c>--render</c> and <c>--play</c> set the Avalonia viewer before the env renders, and
        ///     without them the env never renders.
        /// </exception>
        private static int Main(string[] args) {
            bool render = args.Contains("--render");
            if (args.Contains("--play")) {
                PlayWithKeyboard();
                return 0;
            }
            if (args.Contains("--controller")) {
                (int steps, double total, string ending) = PlayController(render);
                Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"CartPole-v1, seed {ControllerSeed}: {steps} steps, return {total:F1}, {ending}"));
                return 0;
            }
            // The threshold comes from the registry, as gymnasium.spec("CartPole-v1").reward_threshold does in Python.
            double threshold = Gymnasium.Spec("CartPole-v1").RewardThreshold ?? 475.0;

            (double[] policy, bool solved) = Train(threshold);
            double score = Play(policy, render);
            Console.WriteLine($"{(solved ? "Solved" : "Not solved")}: policy {Describe(policy)} played a new episode for a return of " +
                              $"{score.ToString(CultureInfo.InvariantCulture)} (threshold {threshold.ToString(CultureInfo.InvariantCulture)}).");
            return solved ? 0 : 1;
        }

        /// <summary>
        ///     Runs the cross-entropy method until the learned policy averages <paramref name="threshold"/> over
        ///     <see cref="Population"/> new episodes, or <see cref="MaxGenerations"/> run out, printing one line per generation.
        /// </summary>
        /// <param name="threshold">The average return that counts as solved.</param>
        /// <returns>The mean of the final Gaussian (the learned policy) and whether it was solved.</returns>
        private static (double[] Policy, bool Solved) Train(double threshold) {
            // Gymnasium's seeding (gymnasium.utils.seeding.np_random): a NumPy-exact PCG64 generator.
            (Generator rng, _) = Seeding.NpRandom(SearchSeed);
            var mean = new double[ParameterCount];
            double[] deviation = Enumerable.Repeat(1.0, ParameterCount).ToArray();

            // One cart per policy. With no vectorization mode, make_vec picks the env's own vector entry point, which for
            // CartPole is CartPoleVectorEnv; "sync" would instead step Population copies of make("CartPole-v1") in a loop.
            using VectorEnv<NDArray, NDArray> envs = Gymnasium.MakeVec<NDArray, NDArray>("CartPole-v1", numEnvs: Population);
            for (int generation = 0; generation < MaxGenerations; generation++) {
                double[][] candidates = Sample(rng, mean, deviation);
                var returns = new double[Population];
                Evaluate(envs, candidates, seed: generation, returns);
                // OrderByDescending is stable, so equal returns keep their draw order and the run stays reproducible.
                int[] elite = Enumerable.Range(0, Population).OrderByDescending(i => returns[i]).Take(EliteCount).ToArray();
                Refit(candidates, elite, mean, deviation);

                // The elite are the lucky draws of this generation's starting states; the refitted mean is what is kept, so
                // it is the one judged, on every cart at once from starting states no generation was selected on.
                double[][] judged = Enumerable.Repeat(mean, Population).ToArray();
                var judgedReturns = new double[Population];
                Evaluate(envs, judged, seed: JudgeSeed + generation, judgedReturns);
                double score = judgedReturns.Average();
                Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                    $"generation {generation,2}: population {returns.Average(),6:F1}, elite {elite.Average(i => returns[i]),6:F1}, learned policy {score,6:F1}"));
                if (score >= threshold)
                    return (mean, true);
            }

            return (mean, false);
        }

        /// <summary>Draws <see cref="Population"/> policies from <c>N(mean, deviation²)</c>, each parameter independently.</summary>
        /// <param name="rng">The search's generator; each call advances it.</param>
        /// <param name="mean">The Gaussian's mean per parameter.</param>
        /// <param name="deviation">The Gaussian's standard deviation per parameter.</param>
        /// <returns>One array of <see cref="ParameterCount"/> parameters per policy.</returns>
        private static double[][] Sample(Generator rng, double[] mean, double[] deviation) {
            // One standard-normal block per generation, read row by row (NumPy's C order), then scaled and shifted.
            using NDArray noise = rng.standard_normal(new Shape(Population, ParameterCount));
            double[] z = noise.ToArray<double>();
            var candidates = new double[Population][];
            for (int i = 0; i < Population; i++) {
                candidates[i] = new double[ParameterCount];
                for (int k = 0; k < ParameterCount; k++)
                    candidates[i][k] = mean[k] + deviation[k] * z[i * ParameterCount + k];
            }

            return candidates;
        }

        /// <summary>
        ///     Plays one episode with every candidate at once, cart <c>i</c> following candidate <c>i</c>, and writes each
        ///     episode's return into <paramref name="returns"/>.
        /// </summary>
        /// <param name="envs">The vector env, with one cart per candidate.</param>
        /// <param name="candidates">The policies.</param>
        /// <param name="seed">The reset seed, which draws every cart's starting state.</param>
        /// <param name="returns">
        ///     Receives the return of each candidate's episode (1 per step the pole stayed up, at most 500); one slot per
        ///     candidate, zeros on entry.
        /// </param>
        /// <remarks>
        ///     <para>
        ///     A vector env never waits: in its default autoreset mode (<c>NEXT_STEP</c>) a cart whose episode ended is reset
        ///     by the following step, which ignores that cart's action and rewards 0. So the loop keeps stepping until every
        ///     cart has finished its first episode and stops adding a cart's rewards once it has.
        ///     </para>
        ///     <para>
        ///     The arrays a vector env returns belong to the caller. The method is <c>[NDScoped]</c> (DR-80): NumSharp.Build
        ///     releases every batch and action the generation creates when it returns, at most 500 steps' worth; the vector
        ///     env is built outside it, the guides' rule. It fills the caller's array rather than returning one because
        ///     NumSharp.Build refuses an <c>[NDScoped]</c> method that returns a collection (NDW003), whatever it holds.
        ///     </para>
        /// </remarks>
        [NDScoped]
        private static void Evaluate(VectorEnv<NDArray, NDArray> envs, double[][] candidates, long seed, double[] returns) {
            var finished = new bool[candidates.Length];
            NDArray observations = envs.Reset(seed: seed).Observations;
            while (finished.Contains(false)) {
                // (Population, 4) float32 observations, one row per cart.
                float[] rows = observations.ToArray<float>();
                var actions = new long[candidates.Length];
                for (int i = 0; i < candidates.Length; i++)
                    actions[i] = Act(candidates[i], rows.AsSpan(i * ObservationSize, ObservationSize));

                VectorStepResult<NDArray> step = envs.Step(np.array(actions));
                observations = step.Observations;
                double[] reward = step.Rewards.ToArray<double>();
                bool[] terminated = step.Terminations.ToArray<bool>(), truncated = step.Truncations.ToArray<bool>();
                for (int i = 0; i < candidates.Length; i++) {
                    if (finished[i])
                        continue;
                    returns[i] += reward[i];
                    // Terminated: the pole fell or the cart left the track. Truncated: the 500-step limit ran out.
                    finished[i] = terminated[i] || truncated[i];
                }
            }
        }

        /// <summary>Refits the Gaussian to the elite: each parameter's mean and standard deviation, plus <see cref="ExtraNoise"/>.</summary>
        /// <param name="candidates">The generation's policies.</param>
        /// <param name="elite">The indices of the best policies.</param>
        /// <param name="mean">The mean per parameter, overwritten.</param>
        /// <param name="deviation">The standard deviation per parameter, overwritten.</param>
        private static void Refit(double[][] candidates, int[] elite, double[] mean, double[] deviation) {
            for (int k = 0; k < ParameterCount; k++) {
                double average = elite.Average(i => candidates[i][k]);
                double variance = elite.Average(i => (candidates[i][k] - average) * (candidates[i][k] - average));
                mean[k] = average;
                deviation[k] = Math.Sqrt(variance) + ExtraNoise;
            }
        }

        /// <summary>A policy's action: push right (1) when the weighted observation plus the bias is positive, left (0) otherwise.</summary>
        /// <param name="policy">Four weights and a bias.</param>
        /// <param name="observation">Cart position, cart velocity, pole angle, pole angular velocity.</param>
        /// <returns>The action, 0 or 1, as CartPole's <c>Discrete(2)</c> action space numbers them.</returns>
        private static long Act(double[] policy, ReadOnlySpan<float> observation) {
            double sum = policy[ObservationSize];
            for (int k = 0; k < ObservationSize; k++)
                sum += policy[k] * observation[k];
            return sum > 0 ? 1 : 0;
        }

        /// <summary>Plays one episode of CartPole-v1 with a policy, in a window when <paramref name="render"/> is set.</summary>
        /// <param name="policy">The policy.</param>
        /// <param name="render">Whether to render the episode in human mode: an Avalonia window at the env's 50 frames per second.</param>
        /// <returns>The episode's return: 500 when the pole stays up until the time limit truncates the episode.</returns>
        /// <exception cref="Gym.Exceptions.DependencyNotInstalled">
        ///     Thrown at the first human frame when no viewer is set; <paramref name="render"/> sets Avalonia's first.
        /// </exception>
        /// <remarks>
        ///     The method is <c>[NDScoped]</c> (DR-80): NumSharp.Build releases every array the episode creates (the results,
        ///     the actions) when it returns. The env is built and disposed inside it, the guides' rule for an env in a
        ///     scoped method.
        /// </remarks>
        [NDScoped]
        private static double Play(double[] policy, bool render) {
            // Human mode renders inside every reset and step into the app-wide viewer (DR-20); without one it raises
            // DependencyNotInstalled at the first frame, as Gymnasium does without pygame.
            if (render)
                Gymnasium.HumanViewerFactory = AvaloniaEnvViewer.Factory;

            // make's stack, TimeLimit(500) over OrderEnforcing over PassiveEnvChecker over CartPoleEnv. The render mode is
            // a constructor argument, fixed for the env's life, as in Gymnasium.
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("CartPole-v1", kwargs: render ? new { render_mode = "human" } : null);
            NDArray observation = env.Reset(seed: PlaySeed).Observation;
            double total = 0;
            while (true) {
                // A discrete action is a 0-d int64 array, the type Discrete(2).Sample() returns.
                StepResult<NDArray> step = env.Step(NDArray.Scalar(Act(policy, observation.ToArray<float>())));
                observation = step.Observation;
                total += step.Reward;
                if (step.Terminated || step.Truncated)
                    return total;
            }
        }

        /// <summary>
        ///     Plays one episode with a hand-written controller instead of a learned policy: linear state feedback around a
        ///     set point that sways 1.2 either side of the centre every 250 steps, so the cart glides from side to side with
        ///     the pole balanced. It is the controller the front-page showreel plays (DR-74).
        /// </summary>
        /// <param name="render">Whether to render the episode in human mode, in an Avalonia window at 50 frames per second.</param>
        /// <returns>The steps taken, the return (1 per step the pole stays up), and how the episode ended.</returns>
        /// <remarks>
        ///     The push is right when <c>0.6 (x - target) + 1.2 x_dot + 12 theta + 2 theta_dot</c> is positive: the pole's
        ///     angle dominates, so the cart moves under the pole to keep it up, and the position term drags the whole
        ///     balancing act toward the moving set point. From seed 3 it lasts all 500 steps.
        /// </remarks>
        /// <exception cref="Gym.Exceptions.DependencyNotInstalled">
        ///     Not thrown in practice: <paramref name="render"/> sets Avalonia's viewer first.
        /// </exception>
        [NDScoped]   // NumSharp.Build releases every array the episode creates when it returns (DR-80); the env is disposed inside
        private static (int Steps, double Return, string Ending) PlayController(bool render) {
            if (render)
                Gymnasium.HumanViewerFactory = AvaloniaEnvViewer.Factory;
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("CartPole-v1", kwargs: render ? new { render_mode = "human" } : null);
            NDArray observation = env.Reset(seed: ControllerSeed).Observation;
            int steps = 0;
            double total = 0;
            while (true) {
                float[] s = observation.ToArray<float>();
                // The set point's phase is the step count, which is what TimeLimit's _elapsed_steps holds at this point.
                double target = 1.2 * Math.Sin(steps * 2 * Math.PI / 250.0);
                double u = 0.6 * (s[0] - target) + 1.2 * s[1] + 12.0 * s[2] + 2.0 * s[3];
                StepResult<NDArray> step = env.Step(NDArray.Scalar(u > 0 ? 1L : 0L));
                observation = step.Observation;
                steps++;
                total += step.Reward;
                if (step.Terminated || step.Truncated)
                    return (steps, total, step.Terminated ? "the pole fell" : "truncated by the time limit");
            }
        }

        /// <summary>
        ///     Hands the cart to the keyboard, Gymnasium's play, until Escape or the window's close button: A pushes the cart
        ///     left, D right.
        /// </summary>
        /// <remarks>
        ///     CartPole has no neutral action, so with no key held play's default no-op, action 0, pushes left; keep a key
        ///     down, alternating as the pole leans. Every reset uses <see cref="PlaySeed"/>, so each new game starts as the
        ///     last one did, as Gymnasium's play does with a seed.
        /// </remarks>
        private static void PlayWithKeyboard() {
            // play draws the env's rgb_array frames into a window of the app-wide viewer, which also reads the keys.
            Gymnasium.HumanViewerFactory = AvaloniaEnvViewer.Factory;
            using IEnv env = Gymnasium.Make("CartPole-v1", new { render_mode = "rgb_array" });
            // Qualified: inside this class, Play names the method that plays the learned policy.
            Gym.Utils.Play.Run(env, keysToAction: new Dictionary<object, object?> { ["a"] = 0L, ["d"] = 1L }, seed: PlaySeed);
        }

        /// <summary>Formats a policy's parameters for the console.</summary>
        /// <param name="policy">The policy.</param>
        /// <returns><c>[w0, w1, w2, w3, b]</c> with two decimals, culture-invariant.</returns>
        private static string Describe(double[] policy) =>
            "[" + string.Join(", ", policy.Select(p => p.ToString("F2", CultureInfo.InvariantCulture))) + "]";
    }
}
