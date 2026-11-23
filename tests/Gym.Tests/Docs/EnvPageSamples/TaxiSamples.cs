using Gym;
using Gym.Envs;
using NumSharp;

using System;
using System.Linq;
using Gym.Environments.Envs.ToyText;
using Gym.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;

// Not under Gym.Tests: there the namespace Gym.Tests.Gymnasium (DR-06) shadows the Gymnasium facade, and the samples
// must read Gymnasium.Make(...) exactly as a reader's code does.
namespace Gym.Docs.EnvPageSamples {
    /// <summary>
    ///     The C# samples of Taxi's page on the docs site (<c>docs/environments/toy_text/taxi.md</c>), each the body of a
    ///     test, so the unit gate compiles and runs every line the page shows (DR-105). The page includes them by DocFX's
    ///     code-snippet syntax: a region by name, or for the quick start this file's first four lines (its usings and a
    ///     blank) and the quick start's region, by line range.
    /// </summary>
    /// <remarks>
    ///     Footgun: the quick start's include names line numbers (<c>?range=1-4,…</c>), so the first four lines of this file
    ///     must stay <c>using Gym; using Gym.Envs; using NumSharp;</c> and a blank line, and moving the quick start's lines
    ///     moves the page's range; <c>EnvDocPagesTests</c> checks both, so the gate fails before the page shows the wrong
    ///     lines. Each test asserts after its region only, so the assertions never appear on the page; where a region's
    ///     comment states a number (a step count, a return, a state, a probability), the assertion pins it, so the page
    ///     cannot drift from what the env computes.
    /// </remarks>
    [TestClass]
    public class TaxiSamples {
        /// <summary>The page's quick start: make the env, reset it with a seed, take one random step.</summary>
        [TestMethod]
        public void QuickStart() {
            #region QuickStart
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("Taxi-v4");
            // one of 500 states: the taxi's cell, the passenger, the destination
            NDArray observation = env.Reset(seed: 42).Observation;
            // 0 south, 1 north, 2 east, 3 west, 4 pick up, 5 drop off
            using NDArray action = env.ActionSpace.Sample();
            var (next, reward, terminated, truncated, info) = env.Step(action);
            #endregion
            Assert.IsTrue(observation.GetInt64() is >= 0 and < 500);
            Assert.IsTrue(next.GetInt64() is >= 0 and < 500);
            // A first step can cost 1 or, as a pick-up or drop-off where none can happen, 10; it cannot deliver.
            Assert.IsTrue(reward is -1.0 or -10.0);
            Assert.IsFalse(terminated);
            Assert.IsFalse(truncated);
            Assert.AreEqual(1.0, info.Get<double>("prob"));
        }

        /// <summary>
        ///     Run an episode: random actions drawn from the info's action mask, so every one changes the state (no bump
        ///     into a wall, no pick-up or drop-off that costs 10), until the delivery or the 200-step limit.
        /// </summary>
        /// <remarks>
        ///     The mask is the step result's own array, released with it (<see cref="TaxiInfo"/>), so the next action is
        ///     drawn while the step that holds the mask is still alive, and the action, a new array, outlives it.
        /// </remarks>
        [TestMethod]
        public void RunAnEpisode() {
            int steps = 0;
            #region RunAnEpisode
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("Taxi-v4");
            using var start = env.Reset(seed: 42);
            // only the moves that change something: the info's action mask
            NDArray action = env.ActionSpace.Sample(((TaxiInfo)start.Info).ActionMask);
            double episodeReturn = 0;
            while (true) {
                using var step = env.Step(action);
                action.Dispose();
                episodeReturn += step.Reward;
                steps++;
                if (step.Terminated || step.Truncated)
                    break;
                // drawn while the step, which owns the mask, is alive
                TaxiInfo info = step.Info;
                action = env.ActionSpace.Sample(info.ActionMask);
            }
            // -1 a step, 20 for the delivery: masked moves never cost 10
            Console.WriteLine($"return {episodeReturn}");
            #endregion
            Assert.IsTrue(steps <= 200);
            // Every masked step but a delivery costs exactly 1: the return is minus the steps, plus 21 for a delivery.
            Assert.IsTrue(episodeReturn == -steps || episodeReturn == 21 - steps);
        }

        /// <summary>
        ///     Plan with the transition table: value iteration over the env's own <c>P</c>, the agent of the page's video,
        ///     which delivers seed 12's passenger in 16 steps.
        /// </summary>
        [TestMethod]
        public void PlanWithTheTable() {
            int steps = 0;
            double episodeReturn = 0;
            #region PlanWithTheTable
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("Taxi-v4");
            var taxi = (TaxiEnv)env.Unwrapped;
            int n = taxi.P.Count;
            var v = new double[n];
            // the expected discounted return of action a in state s, from P
            double Q(int s, int a) => taxi.P[s][a].Sum(t => t.Probability
                * (t.Reward + (t.Terminated ? 0 : 0.99 * v[t.NextState])));
            for (double delta = 1; delta > 1e-10;) {
                delta = 0;
                for (int s = 0; s < n; s++) {
                    double best = Enumerable.Range(0, 6).Max(a => Q(s, a));
                    delta = Math.Max(delta, Math.Abs(best - v[s]));
                    v[s] = best;
                }
            }
            // the greedy plan: the best action in every state, ties to the lowest
            long[] plan = Enumerable.Range(0, n).Select(s => (long)Enumerable.Range(0, 6)
                .Aggregate((b, a) => Q(s, a) > Q(s, b) + 1e-12 ? a : b)).ToArray();

            NDArray observation = env.Reset(seed: 12).Observation;
            while (true) {
                using NDArray action = NDArray.Scalar(plan[observation.GetInt64()]);
                var (next, reward, terminated, truncated, _) = env.Step(action);
                observation.Dispose();
                observation = next;
                (steps, episodeReturn) = (steps + 1, episodeReturn + reward);
                if (terminated || truncated)
                    break;
            }
            // 16 steps, return 5: 15 steps at -1, then 20 for the delivery
            Console.WriteLine($"{steps} steps, return {episodeReturn}");
            #endregion
            Assert.AreEqual(16, steps);
            Assert.AreEqual(5.0, episodeReturn);
        }

        /// <summary>Encode and decode a state: the observation's number from the taxi's cell, the passenger and the destination, and back.</summary>
        [TestMethod]
        public void EncodeAndDecode() {
            #region EncodeAndDecode
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("Taxi-v4");
            var taxi = (TaxiEnv)env.Unwrapped;
            // row 0, col 4, the passenger at G (1), going to R (0)
            long state = taxi.Encode(0, 4, 1, 0);   // ((0 * 5 + 4) * 5 + 1) * 4 + 0
            var (row, col, passenger, destination) = taxi.Decode(state);
            // 84: 0, 4, 1, 0
            Console.WriteLine($"{state}: {row}, {col}, {passenger}, {destination}");
            #endregion
            Assert.AreEqual(84L, state);
            Assert.AreEqual((0L, 4L, 1L, 0L), (row, col, passenger, destination));
        }

        /// <summary>
        ///     Rain: the rainy table, where a move goes as aimed with probability 0.8 and a tenth of the time is blown to
        ///     either side.
        /// </summary>
        [TestMethod]
        public void Rain() {
            #region Rain
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("Taxi-v4", new { is_rainy = true });
            var taxi = (TaxiEnv)env.Unwrapped;
            // south from the middle, the passenger waiting at R, going to G
            long state = taxi.Encode(2, 2, 0, 1);
            foreach (ToyTextTransition t in taxi.P[state][0]) {
                var (row, col, _, _) = taxi.Decode(t.NextState);
                Console.WriteLine($"{t.Probability:F1} -> row {row}, col {col}");
            }
            // 0.8 -> row 3, col 2    south, as aimed
            // 0.1 -> row 2, col 3    blown east
            // 0.1 -> row 2, col 1    blown west
            #endregion
            var outcomes = taxi.P[state][0];
            CollectionAssert.AreEqual(new[] { (3L, 2L), (2L, 3L), (2L, 1L) },
                outcomes.Select(t => taxi.Decode(t.NextState)).Select(d => (d.TaxiRow, d.TaxiCol)).ToArray());
            CollectionAssert.AreEqual(new[] { 0.8, (1.0 - 0.8) / 2.0, (1.0 - 0.8) / 2.0 }, outcomes.Select(t => t.Probability).ToArray());
        }

        /// <summary>The grid as text: the <c>ansi</c> render mode, the map with the taxi, passenger and destination in colour.</summary>
        [TestMethod]
        public void AsText() {
            #region AsText
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("Taxi-v4",
                    new { render_mode = "ansi" });
            using var start = env.Reset(seed: 12);
            using NDArray south = NDArray.Scalar(0L);
            using var step = env.Step(south);
            using RenderResult result = env.Render();
            string text = ((RenderResult.Ansi)result).Text;
            // the map; the taxi in yellow, the passenger in blue, the
            // destination in magenta, then the last action: "  (South)"
            Console.WriteLine(text);
            #endregion
            Assert.AreEqual("+---------+\n|\u001b[35mR\u001b[0m: | : :\u001b[34;1mG\u001b[0m|\n| : | : : |\n| :\u001b[43m \u001b[0m: : : |\n"
                + "| | : | : |\n|Y| : |B: |\n+---------+\n  (South)\n", text);
        }

        /// <summary>
        ///     Many taxis at once: Taxi has no vector env of its own, so <c>MakeVec</c> steps separate envs in turn
        ///     (<c>sync</c>, its default here).
        /// </summary>
        [TestMethod]
        public void ManyTaxis() {
            #region ManyTaxis
            // No vector env of its own: MakeVec steps 8 envs in turn ("sync")
            using VectorEnv<NDArray, NDArray> envs =
                Gymnasium.MakeVec<NDArray, NDArray>("Taxi-v4", numEnvs: 8);
            using var start = envs.Reset(seed: 42);
            for (int t = 0; t < 100; t++) {
                using NDArray actions = envs.ActionSpace.Sample();   // one per taxi
                using var step = envs.Step(actions);   // finished taxis reset themselves
            }
            #endregion
            Assert.AreEqual(8, envs.NumEnvs);
            Assert.IsInstanceOfType(envs, typeof(SyncVectorEnv<NDArray, NDArray>));
        }

        /// <summary>
        ///     The Arguments section's calls: kwargs by Gymnasium's names, or the generated options record by C# names; both
        ///     build the same env, here a rainy one with a fickle passenger.
        /// </summary>
        [TestMethod]
        public void Arguments() {
            #region Arguments
            using IEnv byName = Gymnasium.Make("Taxi-v4",
                new { is_rainy = true, fickle_passenger = true, rainy_probability = 0.9 });
            using IEnv byRecord = Gymnasium.Make("Taxi-v4",
                new TaxiEnvOptions { IsRainy = true, FicklePassenger = true, RainyProbability = 0.9 });
            #endregion
            foreach (IEnv env in new[] { byName, byRecord }) {
                var taxi = (TaxiEnv)env.Unwrapped;
                Assert.IsTrue(taxi.FicklePassenger);
                Assert.AreEqual(0.9, taxi.RainyProbability);
            }
        }
    }
}
