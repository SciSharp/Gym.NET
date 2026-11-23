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
    ///     The C# samples of Frozen Lake's page on the docs site (<c>docs/environments/toy_text/frozen_lake.md</c>), each the
    ///     body of a test, so the unit gate compiles and runs every line the page shows (DR-105). The page includes them by
    ///     DocFX's code-snippet syntax: a region by name, or for the quick start this file's first four lines (its usings
    ///     and a blank) and the quick start's region, by line range.
    /// </summary>
    /// <remarks>
    ///     Footgun: the quick start's include names line numbers (<c>?range=1-4,…</c>), so the first four lines of this file
    ///     must stay <c>using Gym; using Gym.Envs; using NumSharp;</c> and a blank line, and moving the quick start's lines
    ///     moves the page's range; <c>EnvDocPagesTests</c> checks both, so the gate fails before the page shows the wrong
    ///     lines. Each test asserts after its region only, so the assertions never appear on the page; where a region's
    ///     comment states a number (a step count, a state, a map), the assertion pins it, so the page cannot drift from
    ///     what the env computes.
    /// </remarks>
    [TestClass]
    public class FrozenLakeSamples {
        /// <summary>The page's quick start: make the env, reset it with a seed, take one random step.</summary>
        [TestMethod]
        public void QuickStart() {
            #region QuickStart
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("FrozenLake-v1");
            // the cell, row * 4 + col: 0 is the start
            NDArray observation = env.Reset(seed: 42).Observation;
            // 0 left, 1 down, 2 right, 3 up
            using NDArray action = env.ActionSpace.Sample();
            var (next, reward, terminated, truncated, info) = env.Step(action);
            #endregion
            Assert.AreEqual(0L, observation.GetInt64());
            Assert.IsTrue(next.GetInt64() is >= 0 and < 16);
            // One step from the start cannot reach the goal: the reward is 0.
            Assert.AreEqual(0.0, reward);
            Assert.IsFalse(truncated);
            // The transition's probability: one of the three moves the slippery ice allows, each 1/3.
            Assert.AreEqual(1.0 / 3.0, info.Get<double>("prob"), 1e-12);
            _ = terminated;
        }

        /// <summary>Run an episode: random moves until the elf reaches the goal or falls through a hole.</summary>
        [TestMethod]
        public void RunAnEpisode() {
            #region RunAnEpisode
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("FrozenLake-v1");
            NDArray observation = env.Reset(seed: 42).Observation;
            double episodeReturn = 0;
            while (true) {
                using NDArray action = env.ActionSpace.Sample();
                var (next, reward, terminated, truncated, _) = env.Step(action);
                observation.Dispose();
                observation = next;
                episodeReturn += reward;
                if (terminated || truncated)
                    break;
            }
            Console.WriteLine($"return {episodeReturn}");   // 1 at the goal, else 0
            #endregion
            Assert.IsTrue(episodeReturn is 0.0 or 1.0);
        }

        /// <summary>
        ///     Plan with the transition table: value iteration over the env's own <c>P</c>, the agent of the page's video,
        ///     which reaches the goal from seed 1 in 41 steps.
        /// </summary>
        [TestMethod]
        public void PlanWithTheTable() {
            int steps = 0;
            double episodeReturn = 0;
            #region PlanWithTheTable
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("FrozenLake-v1");
            var lake = (FrozenLakeEnv)env.Unwrapped;
            int n = lake.P.Count;
            var v = new double[n];
            // the expected discounted return of action a in state s, from P
            double Q(int s, int a) => lake.P[s][a].Sum(t => t.Probability
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

            NDArray observation = env.Reset(seed: 1).Observation;
            while (true) {
                using NDArray action = NDArray.Scalar(plan[observation.GetInt64()]);
                var (next, reward, terminated, truncated, _) = env.Step(action);
                observation.Dispose();
                observation = next;
                (steps, episodeReturn) = (steps + 1, episodeReturn + reward);
                if (terminated || truncated)
                    break;
            }
            // 41 steps, return 1
            Console.WriteLine($"{steps} steps, return {episodeReturn}");
            #endregion
            Assert.AreEqual(41, steps);
            Assert.AreEqual(1.0, episodeReturn);
        }

        /// <summary>A lake that does not slip: every move goes where it aims, so a fixed path reaches the goal.</summary>
        [TestMethod]
        public void NotSlippery() {
            long cell = -1;
            double total = 0;
            #region NotSlippery
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("FrozenLake-v1",
                    new { is_slippery = false });
            env.Reset(seed: 0);
            // right, right, down, down, down, right: around the holes to the goal
            foreach (long move in new long[] { 2, 2, 1, 1, 1, 2 }) {
                using NDArray action = NDArray.Scalar(move);
                using var step = env.Step(action);
                (cell, total) = (step.Observation.GetInt64(), total + step.Reward);
            }
            Console.WriteLine($"cell {cell}, return {total}");   // cell 15, return 1
            #endregion
            Assert.AreEqual(15L, cell);
            Assert.AreEqual(1.0, total);
        }

        /// <summary>A random map: Gymnasium's generator of solvable lakes, seeded, then the lake made from it.</summary>
        [TestMethod]
        public void ARandomMap() {
            #region ARandomMap
            // an 8x8 lake with a path from S to G, 80% frozen
            IReadOnlyList<string> map = FrozenLakeEnv.GenerateRandomMap(size: 8, seed: 7);
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("FrozenLake-v1", new { desc = map });
            Console.WriteLine(string.Join("\n", map));   // SHFFFHFH ... HFHFFFFG
            #endregion
            // Gymnasium's generate_random_map(size=8, seed=7) returns this map.
            CollectionAssert.AreEqual(new[] { "SHFFFHFH", "FFFFFFFF", "HFFHFFFF", "FFFHFFFF", "FFFFFFHF", "FHFHFFFF", "FHFFFFFF", "HFHFFFFG" },
                map.ToArray());
            Assert.AreEqual("Discrete(64)", env.ObservationSpace.ToString());
        }

        /// <summary>The lake as text: the <c>ansi</c> render mode, the last move's name over the map, the elf's cell marked.</summary>
        [TestMethod]
        public void AsText() {
            string? text = null;
            #region AsText
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("FrozenLake-v1",
                    new { render_mode = "ansi" });
            env.Reset(seed: 0);
            using NDArray down = NDArray.Scalar(1L);
            using var step = env.Step(down);
            using RenderResult result = env.Render();
            text = ((RenderResult.Ansi)result).Text;
            Console.WriteLine(text);   // "  (Down)" over the map, the elf's cell in red
            #endregion
            StringAssert.StartsWith(text, "  (Down)\n");
            StringAssert.Contains(text, "FHFH");
        }

        /// <summary>
        ///     Many lakes at once: Frozen Lake has no vector env of its own, so <c>MakeVec</c> steps separate envs in turn
        ///     (<c>sync</c>, its default here).
        /// </summary>
        [TestMethod]
        public void ManyLakes() {
            #region ManyLakes
            // No vector env of its own: MakeVec steps 8 envs in turn ("sync")
            using VectorEnv<NDArray, NDArray> envs =
                Gymnasium.MakeVec<NDArray, NDArray>("FrozenLake-v1", numEnvs: 8);
            using var start = envs.Reset(seed: 42);
            for (int t = 0; t < 100; t++) {
                using NDArray actions = envs.ActionSpace.Sample();   // one per lake
                using var step = envs.Step(actions);   // finished lakes reset themselves
            }
            #endregion
            Assert.AreEqual(8, envs.NumEnvs);
            Assert.IsInstanceOfType(envs, typeof(SyncVectorEnv<NDArray, NDArray>));
        }

        /// <summary>
        ///     The Arguments section's calls: kwargs by Gymnasium's names, or the generated options record by C# names; both
        ///     build the same env.
        /// </summary>
        [TestMethod]
        public void Arguments() {
            #region Arguments
            using IEnv byName = Gymnasium.Make("FrozenLake-v1",
                new { map_name = "8x8", is_slippery = true, success_rate = 0.75 });
            using IEnv byRecord = Gymnasium.Make("FrozenLake-v1",
                new FrozenLakeEnvOptions { MapName = "8x8", SuccessRate = 0.75 });
            #endregion
            Assert.AreEqual(8, ((FrozenLakeEnv)byName.Unwrapped).Nrow);
            Assert.AreEqual(8, ((FrozenLakeEnv)byRecord.Unwrapped).Ncol);
        }
    }
}
