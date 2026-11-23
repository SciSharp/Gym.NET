using Gym;
using Gym.Envs;
using NumSharp;

using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Environments.Envs.ToyText;
using Gym.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;

// Not under Gym.Tests: there the namespace Gym.Tests.Gymnasium (DR-06) shadows the Gymnasium facade, and the samples
// must read Gymnasium.Make(...) exactly as a reader's code does.
namespace Gym.Docs.EnvPageSamples {
    /// <summary>
    ///     The C# samples of Cliff Walking's page on the docs site (<c>docs/environments/toy_text/cliff_walking.md</c>),
    ///     each the body of a test, so the unit gate compiles and runs every line the page shows (DR-105). The page
    ///     includes them by DocFX's code-snippet syntax: a region by name, or for the quick start this file's first four
    ///     lines (its usings and a blank) and the quick start's region, by line range.
    /// </summary>
    /// <remarks>
    ///     Footgun: the quick start's include names line numbers (<c>?range=1-4,…</c>), so the first four lines of this file
    ///     must stay <c>using Gym; using Gym.Envs; using NumSharp;</c> and a blank line, and moving the quick start's lines
    ///     moves the page's range; <c>EnvDocPagesTests</c> checks both, so the gate fails before the page shows the wrong
    ///     lines. Each test asserts after its region only, so the assertions never appear on the page; where a region's
    ///     comment states a number (a step count, a return, a probability), the assertion pins it, so the page cannot
    ///     drift from what the env computes.
    /// </remarks>
    [TestClass]
    public class CliffWalkingSamples {
        /// <summary>The page's quick start: make the env, reset it with a seed, take one random step.</summary>
        [TestMethod]
        public void QuickStart() {
            #region QuickStart
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("CliffWalking-v1");
            // the cell, row * 12 + col: 36 is the start
            NDArray observation = env.Reset(seed: 42).Observation;
            // 0 up, 1 right, 2 down, 3 left
            using NDArray action = env.ActionSpace.Sample();
            var (next, reward, terminated, truncated, info) = env.Step(action);
            #endregion
            Assert.AreEqual(36L, observation.GetInt64());
            // One step from the start: up to 24, right off the cliff (back to 36, -100), or into the edge (stays on 36).
            Assert.IsTrue(next.GetInt64() is 24 or 36);
            Assert.IsTrue(reward is -1.0 or -100.0);
            Assert.IsFalse(terminated);
            Assert.IsFalse(truncated);
            // Every move on the solid grid is certain.
            Assert.AreEqual(1.0, info.Get<double>("prob"));
        }

        /// <summary>
        ///     Run an episode: random moves under a step limit of the reader's own, since CliffWalking-v1 registers none
        ///     and a random walk takes thousands of steps to reach the goal.
        /// </summary>
        [TestMethod]
        public void RunAnEpisode() {
            int steps = 0;
            #region RunAnEpisode
            // CliffWalking-v1 has no step limit: give it one
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>(
                "CliffWalking-v1", maxEpisodeSteps: 500);
            NDArray observation = env.Reset(seed: 42).Observation;
            double episodeReturn = 0;
            while (true) {
                using NDArray action = env.ActionSpace.Sample();
                var (next, reward, terminated, truncated, _) = env.Step(action);
                observation.Dispose();
                observation = next;
                episodeReturn += reward;
                steps++;
                if (terminated || truncated)
                    break;
            }
            // -1 a step, -100 a fall: random moves rarely reach the goal
            Console.WriteLine($"return {episodeReturn}");
            #endregion
            Assert.IsTrue(steps <= 500);
            // Every step costs at least 1, a fall 100.
            Assert.IsTrue(episodeReturn <= -steps && episodeReturn >= -100.0 * steps);
        }

        /// <summary>
        ///     Plan with the transition table: value iteration over the env's own <c>P</c>, the agent of the page's video,
        ///     which walks the cliff's edge from the start to the goal in 13 steps.
        /// </summary>
        [TestMethod]
        public void PlanWithTheTable() {
            int steps = 0;
            double episodeReturn = 0;
            #region PlanWithTheTable
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("CliffWalking-v1");
            var grid = (CliffWalkingEnv)env.Unwrapped;
            int n = grid.P.Count;
            var v = new double[n];
            // the expected discounted return of action a in state s, from P
            double Q(int s, int a) => grid.P[s][a].Sum(t => t.Probability
                * (t.Reward + (t.Terminated ? 0 : 0.99 * v[t.NextState])));
            for (double delta = 1; delta > 1e-10;) {
                delta = 0;
                for (int s = 0; s < n; s++) {
                    double best = Enumerable.Range(0, 4).Max(a => Q(s, a));
                    delta = Math.Max(delta, Math.Abs(best - v[s]));
                    v[s] = best;
                }
            }
            // the greedy plan: the best action in every state, ties to the lowest
            long[] plan = Enumerable.Range(0, n).Select(s => (long)Enumerable.Range(0, 4)
                .Aggregate((b, a) => Q(s, a) > Q(s, b) + 1e-12 ? a : b)).ToArray();

            NDArray observation = env.Reset(seed: 0).Observation;
            while (true) {
                using NDArray action = NDArray.Scalar(plan[observation.GetInt64()]);
                var (next, reward, terminated, truncated, _) = env.Step(action);
                observation.Dispose();
                observation = next;
                (steps, episodeReturn) = (steps + 1, episodeReturn + reward);
                if (terminated || truncated)
                    break;
            }
            // 13 steps, return -13: up, along the cliff's edge, down
            Console.WriteLine($"{steps} steps, return {episodeReturn}");
            #endregion
            Assert.AreEqual(13, steps);
            Assert.AreEqual(-13.0, episodeReturn);
        }

        /// <summary>
        ///     Learn it: tabular Q-learning as Sutton and Barto's Example 6.6 runs it (epsilon-greedy with epsilon 0.1, no
        ///     discount), with a step size of 0.5, which the example leaves unstated, and NumPy's generator for the
        ///     exploration, so the Python twin, with the same seed, makes the same draws, learns the same table and prints the
        ///     same average.
        /// </summary>
        /// <remarks>
        ///     The greedy walk over the learned table is asserted after the region: the 13-step path along the edge, which
        ///     Q-learning learned from every one of seeds 0 to 99 (measured on Gymnasium 1.4.0). The generator's draws are
        ///     0-d arrays, each disposed once read, so the thousands of draws the 500 episodes make leave nothing to the
        ///     finalizer.
        /// </remarks>
        [TestMethod]
        public void QLearning() {
            #region QLearning
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("CliffWalking-v1");
            var q = new double[48, 4];
            Generator rng = np.random.default_rng(0);
            // the best move known, ties to the lowest, as NumPy's argmax
            long Best(long s) =>
                Enumerable.Range(0, 4).Aggregate((b, a) => q[s, a] > q[s, b] ? a : b);
            // epsilon-greedy: a random move one time in ten
            long Act(long s) {
                using NDArray coin = rng.random();
                if ((double)coin >= 0.1)
                    return Best(s);
                using NDArray move = rng.integers(4);
                return (long)move;
            }

            var returns = new List<double>();
            for (int episode = 0; episode < 500; episode++) {
                using var start = env.Reset();
                long s = start.Observation.GetInt64();
                double total = 0;
                while (true) {
                    long a = Act(s);
                    using NDArray action = NDArray.Scalar(a);
                    using var step = env.Step(action);
                    long next = step.Observation.GetInt64();
                    // Q-learning: towards the reward plus the next cell's best value
                    double best = step.Terminated ? 0 : q[next, Best(next)];
                    q[s, a] += 0.5 * (step.Reward + best - q[s, a]);
                    (s, total) = (next, total + step.Reward);
                    if (step.Terminated)
                        break;
                }
                returns.Add(total);
            }
            // still exploring, it falls now and then: -30.19 over the last 100
            Console.WriteLine($"{returns.TakeLast(100).Average():F2}");
            #endregion
            Assert.AreEqual(-30.19, Math.Round(returns.TakeLast(100).Average(), 2));
            // The whole run is the Python twin's, draw for draw: its 500 returns sum to -27,360.
            Assert.AreEqual(-27360.0, returns.Sum());
            // The greedy walk over the learned table: up, along the edge, down.
            using var reset = env.Reset();
            long cell = reset.Observation.GetInt64();
            int walked = 0;
            double greedyReturn = 0;
            for (; walked < 200; walked++) {
                using NDArray action = NDArray.Scalar(Best(cell));
                using var step = env.Step(action);
                (cell, greedyReturn) = (step.Observation.GetInt64(), greedyReturn + step.Reward);
                if (step.Terminated) {
                    walked++;
                    break;
                }
            }
            Assert.AreEqual(13, walked);
            Assert.AreEqual(-13.0, greedyReturn);
        }

        /// <summary>
        ///     A slippery cliff: CliffWalkingSlippery-v1's table, where every move goes as aimed or slides a quarter turn
        ///     either way, a third each; the first step up from the start already risks the cliff.
        /// </summary>
        [TestMethod]
        public void Slippery() {
            #region Slippery
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("CliffWalkingSlippery-v1");
            var grid = (CliffWalkingEnv)env.Unwrapped;
            // up from the start: a third each to the left, up and right
            foreach (ToyTextTransition t in grid.P[36][CliffWalkingEnv.Up])
                Console.WriteLine($"{t.Probability:F2} -> {t.NextState}, {t.Reward}");
            // 0.33 -> 36, -1      a slide left, into the edge: stays
            // 0.33 -> 24, -1      up, as aimed
            // 0.33 -> 36, -100    a slide right, off the cliff: back to the start
            #endregion
            IReadOnlyList<ToyTextTransition> outcomes = grid.P[36][CliffWalkingEnv.Up];
            Assert.AreEqual(3, outcomes.Count);
            CollectionAssert.AreEqual(new long[] { 36, 24, 36 }, outcomes.Select(t => t.NextState).ToArray());
            CollectionAssert.AreEqual(new[] { -1.0, -1.0, -100.0 }, outcomes.Select(t => t.Reward).ToArray());
            Assert.IsTrue(outcomes.All(t => t.Probability == 1.0 / 3.0 && !t.Terminated));
        }

        /// <summary>The grid as text: the <c>ansi</c> render mode, the walker, the cliff and the goal marked.</summary>
        [TestMethod]
        public void AsText() {
            #region AsText
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("CliffWalking-v1",
                    new { render_mode = "ansi" });
            using var start = env.Reset(seed: 0);
            using NDArray up = NDArray.Scalar(CliffWalkingEnv.Up);
            using var step = env.Step(up);
            using RenderResult result = env.Render();
            string text = ((RenderResult.Ansi)result).Text;
            // x the walker, C the cliff, T the goal, o the rest
            Console.WriteLine(text);
            #endregion
            string open = string.Join("  ", Enumerable.Repeat("o", 12));
            string walker = "x  " + string.Join("  ", Enumerable.Repeat("o", 11));
            string cliff = "o  " + string.Join("  ", Enumerable.Repeat("C", 10)) + "  T";
            Assert.AreEqual($"{open}\n{open}\n{walker}\n{cliff}\n\n", text);
        }

        /// <summary>
        ///     Many cliffs at once: Cliff Walking has no vector env of its own, so <c>MakeVec</c> steps separate envs in turn
        ///     (<c>sync</c>, its default here).
        /// </summary>
        [TestMethod]
        public void ManyCliffs() {
            #region ManyCliffs
            // No vector env of its own: MakeVec steps 8 envs in turn ("sync")
            using VectorEnv<NDArray, NDArray> envs =
                Gymnasium.MakeVec<NDArray, NDArray>("CliffWalking-v1", numEnvs: 8);
            using var start = envs.Reset(seed: 42);
            for (int t = 0; t < 100; t++) {
                using NDArray actions = envs.ActionSpace.Sample();   // one per grid
                using var step = envs.Step(actions);   // finished grids reset themselves
            }
            #endregion
            Assert.AreEqual(8, envs.NumEnvs);
            Assert.IsInstanceOfType(envs, typeof(SyncVectorEnv<NDArray, NDArray>));
        }

        /// <summary>
        ///     The Arguments section's calls: kwargs by Gymnasium's names, or the generated options record by C# names; both
        ///     build the same env, here the slippery one.
        /// </summary>
        [TestMethod]
        public void Arguments() {
            #region Arguments
            using IEnv byName = Gymnasium.Make("CliffWalking-v1",
                new { is_slippery = true });
            using IEnv byRecord = Gymnasium.Make("CliffWalking-v1",
                new CliffWalkingEnvOptions { IsSlippery = true });
            #endregion
            Assert.IsTrue(((CliffWalkingEnv)byName.Unwrapped).IsSlippery);
            Assert.IsTrue(((CliffWalkingEnv)byRecord.Unwrapped).IsSlippery);
        }
    }
}
