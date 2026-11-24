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
using Gym.Utils;
using Gym.Vector;
using Gym.Wrappers;
using NumSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Gym.Examples.CartPole {
    /// <summary>
    ///     Learns to balance CartPole-v1 with the cross-entropy method over linear policies, then plays the learned policy
    ///     for one episode: <c>dotnet run --project examples/CartPole -c Release</c>, with <c>-- --render</c> to watch that
    ///     episode in a window. <c>-- --controller</c> plays a hand-written controller instead of learning one (the cart
    ///     the front-page showreel shows), <c>-- --play</c> hands the cart to the keyboard, and
    ///     <c>-- --record &lt;folder&gt;</c> records the controller's episode for CartPole's page on the docs site (DR-112).
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

        /// <summary>
        ///     The moment How it works annotates on CartPole's page (DR-114): the frame after this many steps of the
        ///     controller's episode, mid-sway, with the cart right of the centre and the pole leaning back toward it.
        /// </summary>
        private const int FigureStep = 125;

        /// <summary>The id the media files are named after, <c>&lt;id&gt;.episode.mp4</c> and the rest (DR-112).</summary>
        private const string EnvId = "CartPole-v1";

        /// <summary>Trains a policy, then plays one episode with it; or plays the hand-written controller; or the keyboard.</summary>
        /// <param name="args">
        ///     <c>--render</c> plays the final episode in a window (Avalonia); <c>--controller</c> plays the hand-written
        ///     controller instead of training; <c>--play</c> hands the cart to the keyboard (A and D) until Escape or the
        ///     window's close button; <c>--record &lt;folder&gt;</c> records the controller's episode for the docs site
        ///     (<see cref="Record"/>). Other arguments are ignored.
        /// </param>
        /// <returns>
        ///     The process exit code: 0 when training solved CartPole-v1 (or the controller, the keyboard game or the
        ///     recording ran), 1 when <see cref="MaxGenerations"/> ran out first, 2 when <c>--record</c> names no folder.
        /// </returns>
        /// <exception cref="Gym.Exceptions.DependencyNotInstalled">
        ///     Not thrown in practice: <c>--render</c> and <c>--play</c> set the Avalonia viewer before the env renders, and
        ///     without them the env never renders in human mode.
        /// </exception>
        /// <exception cref="InvalidOperationException">
        ///     <c>--record</c> could not run ffmpeg, which <see cref="FfmpegVideoWriter"/> needs on PATH.
        /// </exception>
        private static int Main(string[] args) {
            bool render = args.Contains("--render");
            if (args.Contains("--play")) {
                PlayWithKeyboard();
                return 0;
            }
            int record = Array.IndexOf(args, "--record");
            if (record >= 0) {
                // The folder is required: the page's media live in the docs tree, and a default would write them where no
                // page reads them.
                if (record + 1 >= args.Length || args[record + 1].StartsWith("--", StringComparison.Ordinal)) {
                    Console.Error.WriteLine("--record needs a folder, e.g. --record docs/environments/classic_control/media");
                    return 2;
                }
                (int steps, double total) = Record(args[record + 1]);
                Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                    $"{EnvId}, seed {ControllerSeed}: {steps} steps, return {total:F1}, recorded into {args[record + 1]}"));
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
            double[] mean = new double[ParameterCount];
            double[] deviation = Enumerable.Repeat(1.0, ParameterCount).ToArray();

            // One cart per policy. With no vectorization mode, make_vec picks the env's own vector entry point, which for
            // CartPole is CartPoleVectorEnv; "sync" would instead step Population copies of make("CartPole-v1") in a loop.
            using VectorEnv<NDArray, NDArray> envs = Gymnasium.MakeVec<NDArray, NDArray>("CartPole-v1", numEnvs: Population);
            for (int generation = 0; generation < MaxGenerations; generation++) {
                double[][] candidates = Sample(rng, mean, deviation);
                double[] returns = new double[Population];
                Evaluate(envs, candidates, seed: generation, returns);
                // OrderByDescending is stable, so equal returns keep their draw order and the run stays reproducible.
                int[] elite = Enumerable.Range(0, Population).OrderByDescending(i => returns[i]).Take(EliteCount).ToArray();
                Refit(candidates, elite, mean, deviation);

                // The elite are the lucky draws of this generation's starting states; the refitted mean is what is kept, so
                // it is the one judged, on every cart at once from starting states no generation was selected on.
                double[][] judged = Enumerable.Repeat(mean, Population).ToArray();
                double[] judgedReturns = new double[Population];
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
            double[][] candidates = new double[Population][];
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
            bool[] finished = new bool[candidates.Length];
            (NDArray observations, Info infos) = envs.Reset(seed: seed);
            while (finished.Contains(false)) {
                // (Population, 4) float32 observations, one row per cart.
                float[] rows = observations.ToArray<float>();
                long[] actions = new long[candidates.Length];
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
            (NDArray observation, Info info) = env.Reset(seed: PlaySeed);
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
            (NDArray observation, Info info) = env.Reset(seed: ControllerSeed);
            int steps = 0;
            double total = 0;
            while (true) {
                float[] s = observation.ToArray<float>();
                StepResult<NDArray> step = env.Step(NDArray.Scalar(ControllerAction(s, SwayTarget(steps))));
                observation = step.Observation;
                steps++;
                total += step.Reward;
                if (step.Terminated || step.Truncated)
                    return (steps, total, step.Terminated ? "the pole fell" : "truncated by the time limit");
            }
        }

        /// <summary>The controller's set point before a step: 1.2 either side of the centre, a full sway every 250 steps.</summary>
        /// <param name="steps">Steps already taken in the episode (the phase is the step count, which TimeLimit's
        ///     <c>_elapsed_steps</c> holds at this point).</param>
        /// <returns>The cart position the controller steers the balancing act toward.</returns>
        private static double SwayTarget(int steps) => 1.2 * Math.Sin(steps * 2 * Math.PI / 250.0);

        /// <summary>
        ///     The controller's push for an observation: right (1) when <c>0.6 (x - target) + 1.2 x_dot + 12 theta + 2
        ///     theta_dot</c> is positive, else left (0). <see cref="PlayController"/> and <see cref="Record"/> both act through
        ///     it, so the recorded episode is the one <c>--controller</c> plays.
        /// </summary>
        /// <param name="s">The observation: cart position, cart velocity, pole angle, pole angular velocity (float32).</param>
        /// <param name="target">The set point, <see cref="SwayTarget"/>.</param>
        /// <returns>The action, 0 or 1.</returns>
        private static long ControllerAction(float[] s, double target) {
            double u = 0.6 * (s[0] - target) + 1.2 * s[1] + 12.0 * s[2] + 2.0 * s[3];
            return u > 0 ? 1L : 0L;
        }

        /// <summary>
        ///     Records the controller's episode for CartPole's page on the docs site (DR-112): the whole episode as
        ///     <c>CartPole-v1.episode.mp4</c>, one frame per step at the env's 50 frames a second, through Gymnasium's
        ///     <see cref="RecordVideo{TObs, TAct}"/> and <see cref="FfmpegVideoWriter"/>; every step as
        ///     <c>CartPole-v1.episode.json</c>, for the page's anatomy chart; and How it works' frame as
        ///     <c>CartPole-v1.figure.png</c> with <c>CartPole-v1.figure.json</c>, the pixels its labels point at.
        /// </summary>
        /// <param name="folder">The output folder, <c>docs/environments/classic_control/media</c> for the site; created if missing.</param>
        /// <returns>The steps taken and the return.</returns>
        /// <exception cref="InvalidOperationException">ffmpeg could not be run, or wrote no video.</exception>
        /// <exception cref="IOException">A file cannot be written.</exception>
        /// <remarks>
        ///     <para>
        ///     Each JSON row is the observation the controller acted on, then its action, the reward and the sway's set
        ///     point: <c>step, x, x_dot, theta, theta_dot, action, reward, target</c>. The video's frame <c>t</c> shows the
        ///     state of row <c>t</c> (frame 0 is the reset), and its last frame the state after the last step, which the
        ///     JSON keeps as <c>final</c>.
        ///     </para>
        ///     <para>
        ///     RecordVideo names its file <c>CartPole-v1-episode-0.mp4</c> and writes it when the env closes; it records into
        ///     a scratch folder inside <paramref name="folder"/> and the file is moved to the page's name, so a recording
        ///     that fails halfway leaves the published video as it was.
        ///     </para>
        /// </remarks>
        [NDScoped]   // releases every array the episode creates when it returns (DR-80); the envs are disposed inside
        private static (int Steps, double Return) Record(string folder) {
            Directory.CreateDirectory(folder);
            string scratch = Path.Combine(folder, ".recording");
            var rows = new List<double[]>();
            double total = 0;
            float[] final;
            using (Env<NDArray, NDArray> made = Gymnasium.Make<NDArray, NDArray>(EnvId, RenderMode.RgbArray))
            using (var env = new RecordVideo<NDArray, NDArray>(made, scratch, episodeTrigger: _ => true, namePrefix: EnvId,
                       videoWriter: new FfmpegVideoWriter())) {
                (NDArray observation, Info info) = env.Reset(seed: ControllerSeed);
                for (int t = 0; ; t++) {
                    float[] s = observation.ToArray<float>();
                    double target = SwayTarget(t);
                    long action = ControllerAction(s, target);
                    // The frame now on screen shows the state the controller is about to act on (row t).
                    if (t == FigureStep)
                        SaveFigure(env, folder, s);
                    StepResult<NDArray> step = env.Step(NDArray.Scalar(action));
                    total += step.Reward;
                    rows.Add(new double[] { t, s[0], s[1], s[2], s[3], action, step.Reward, target });
                    observation = step.Observation;
                    if (step.Terminated || step.Truncated)
                        break;
                }
                final = observation.ToArray<float>();
                // RecordVideo writes the episode's video when it stops recording, which closing the env does.
                env.Close();
            }
            string recorded = Path.Combine(scratch, $"{EnvId}-episode-0.mp4");
            if (!File.Exists(recorded))
                throw new InvalidOperationException($"RecordVideo wrote no {recorded}; is ffmpeg on PATH?");
            File.Move(recorded, Path.Combine(folder, $"{EnvId}.episode.mp4"), overwrite: true);
            Directory.Delete(scratch, recursive: true);
            var json = new JsonSerializerOptions { WriteIndented = false };
            File.WriteAllText(Path.Combine(folder, $"{EnvId}.episode.json"), JsonSerializer.Serialize(new {
                id = EnvId, seed = ControllerSeed, steps = rows.Count, @return = total, fps = 50,
                agent = "the swaying controller (examples/CartPole --controller)",
                columns = new[] { "step", "x", "x_dot", "theta", "theta_dot", "action", "reward", "target" },
                // Five decimals keep the chart exact to the pixel at a fraction of the file size.
                rows = rows.Select(r => r.Select(v => Math.Round(v, 5)).ToArray()).ToArray(),
                final = final.Select(v => Math.Round((double)v, 5)).ToArray(),
            }, json) + "\n");
            return (rows.Count, total);
        }

        /// <summary>
        ///     Saves How it works' frame (the frame the env shows now) as <c>CartPole-v1.figure.png</c>, and where its labels
        ///     point as <c>CartPole-v1.figure.json</c>: the cart's centre, the pole's axle and tip, in the frame's pixels.
        /// </summary>
        /// <param name="env">The recording env, in <c>rgb_array</c> mode; rendering it adds nothing to the video.</param>
        /// <param name="folder">The output folder.</param>
        /// <param name="s">The state the frame shows: cart position, cart velocity, pole angle, pole angular velocity.</param>
        /// <exception cref="IOException">A file cannot be written.</exception>
        /// <remarks>
        ///     The pixels follow CartPole's renderer: 125 px a metre (the 600 px frame spans the ±2.4 m track), the cart's
        ///     centre at <c>300 + 125 x</c>, the axle 107.5 px above the bottom edge (the cart's centre at 100 px plus a
        ///     quarter of its 30 px height), the pole's tip 120 px along the pole from the axle (its 125 px length less half
        ///     its 10 px width). Reference:
        ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/cartpole.py#L279-L322">gymnasium/envs/classic_control/cartpole.py</see>.
        /// </remarks>
        private static void SaveFigure(Env<NDArray, NDArray> env, string folder, float[] s) {
            using (RenderResult render = env.Render()) {
                NDArray frame = (NDArray)render;
                int height = (int)frame.shape[0], width = (int)frame.shape[1];
                using var image = Image.LoadPixelData<Rgb24>(frame.ToArray<byte>(), width, height);
                image.SaveAsPng(Path.Combine(folder, $"{EnvId}.figure.png"));
            }
            // The renderer draws in pygame's upward coordinates and flips the frame, so a height h above the bottom edge is
            // the frame's row 400 - h.
            const double Scale = 600 / 4.8, CartY = 400 - 100, AxleY = 400 - (100 + 30 / 4.0), TipLength = 125 - 10 / 2.0;
            double cartX = 300 + s[0] * Scale, theta = s[2];
            File.WriteAllText(Path.Combine(folder, $"{EnvId}.figure.json"), JsonSerializer.Serialize(new {
                id = EnvId, step = FigureStep, width = 600, height = 400, pixelsPerMetre = Scale,
                state = new { x = Math.Round(s[0], 5), x_dot = Math.Round(s[1], 5), theta = Math.Round(s[2], 5), theta_dot = Math.Round(s[3], 5) },
                cart = new[] { Math.Round(cartX, 2), CartY },
                axle = new[] { Math.Round(cartX, 2), AxleY },
                tip = new[] { Math.Round(cartX + TipLength * Math.Sin(theta), 2), Math.Round(AxleY - TipLength * Math.Cos(theta), 2) },
            }, new JsonSerializerOptions { WriteIndented = true }) + "\n");
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
