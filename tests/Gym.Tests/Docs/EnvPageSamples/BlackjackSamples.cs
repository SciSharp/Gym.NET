using Gym;
using Gym.Envs;
using NumSharp;

// Below the four lines the quick start shows: Blackjack's observation type is object?[], which the test project's
// default (no nullable context) would flag at every use.
#nullable enable
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
    ///     The C# samples of Blackjack's page on the docs site (<c>docs/environments/toy_text/blackjack.md</c>), each the
    ///     body of a test, so the unit gate compiles and runs every line the page shows (DR-105). The page includes them by
    ///     DocFX's code-snippet syntax: a region by name, or for the quick start this file's first four lines (its usings
    ///     and a blank) and the quick start's region, by line range.
    /// </summary>
    /// <remarks>
    ///     Footgun: the quick start's include names line numbers (<c>?range=1-4,…</c>), so the first four lines of this file
    ///     must stay <c>using Gym; using Gym.Envs; using NumSharp;</c> and a blank line, and moving the quick start's lines
    ///     moves the page's range; <c>EnvDocPagesTests</c> checks both, so the gate fails before the page shows the wrong
    ///     lines. Each test asserts after its region only, so the assertions never appear on the page; where a region's
    ///     comment states a number (a return, a mean, a value, a card), the assertion pins it, so the page cannot drift
    ///     from what the env computes. Blackjack's observation is a tuple of three 0-d arrays that the result owns:
    ///     <see cref="BlackjackObservation"/> shares them, so the samples read the numbers out before the result that
    ///     holds them is disposed.
    /// </remarks>
    [TestClass]
    public class BlackjackSamples {
        /// <summary>The page's quick start: make the env, deal a hand with a seed, take one random action.</summary>
        [TestMethod]
        public void QuickStart() {
            #region QuickStart
            using Env<object?[], NDArray> env =
                Gymnasium.Make<object?[], NDArray>("Blackjack-v1");
            // (the player's sum, the dealer's face-up card, a usable ace)
            object?[] observation = env.Reset(seed: 42).Observation;
            // 0 stick, 1 hit
            using NDArray action = env.ActionSpace.Sample();
            var (next, reward, terminated, truncated, info) = env.Step(action);
            #endregion
            // Gymnasium deals seed 42 a hard 15 against the dealer's 2.
            CollectionAssert.AreEqual(new long[] { 15, 2, 0 }, observation.Select(o => ((NDArray)o!).GetInt64()).ToArray());
            Assert.IsTrue(reward is -1.0 or 0.0 or 1.0);
            Assert.IsFalse(truncated);
            Assert.AreEqual(0, info.Count);
            _ = (next, terminated);
        }

        /// <summary>Run an episode: random hits and sticks until the hand ends, one hand an episode.</summary>
        [TestMethod]
        public void RunAnEpisode() {
            double episodeReturn = 0;
            #region RunAnEpisode
            using Env<object?[], NDArray> env =
                Gymnasium.Make<object?[], NDArray>("Blackjack-v1");
            using var start = env.Reset(seed: 42);
            while (true) {
                using NDArray action = env.ActionSpace.Sample();
                using var step = env.Step(action);
                episodeReturn += step.Reward;
                if (step.Terminated || step.Truncated)
                    break;
            }
            Console.WriteLine($"return {episodeReturn}");   // 1 a win, -1 a loss, 0 a draw
            #endregion
            Assert.IsTrue(episodeReturn is -1.0 or 0.0 or 1.0);
        }

        /// <summary>
        ///     Basic strategy over 10,000 seeded hands, the observation read by name: the example's agent, and what it
        ///     loses a hand on average.
        /// </summary>
        [TestMethod]
        public void BasicStrategy() {
            double total = 0;
            #region BasicStrategy
            using Env<object?[], NDArray> env =
                Gymnasium.Make<object?[], NDArray>("Blackjack-v1");
            // hit or stick by the player's sum, the dealer's card and a soft hand
            long Basic(long sum, long dealer, bool soft) => (soft
                ? sum < 18 || (sum == 18 && (dealer >= 9 || dealer == 1))
                : sum < 12 || (sum < 17 && (dealer >= 7 || dealer == 1))
                    || (sum == 12 && dealer <= 3)) ? 1 : 0;
            for (int seed = 0; seed < 10_000; seed++) {
                using var start = env.Reset(seed: seed);
                // the tuple read by name; its arrays go with the result
                BlackjackObservation hand = start.Observation;
                long dealer = hand.DealerCard.GetInt64();
                (long sum, bool soft) = (hand.PlayerSum.GetInt64(), hand.UsableAce.GetInt64() == 1);
                while (true) {
                    using NDArray action = NDArray.Scalar(Basic(sum, dealer, soft));
                    using var step = env.Step(action);
                    BlackjackObservation next = step.Observation;
                    (sum, soft) = (next.PlayerSum.GetInt64(), next.UsableAce.GetInt64() == 1);
                    total += step.Reward;
                    if (step.Terminated)
                        break;
                }
            }
            // -0.0476 a hand: basic strategy still loses, as in a casino
            Console.WriteLine($"{total / 10_000:F4}");
            #endregion
            Assert.AreEqual(-476.0, total);
        }

        /// <summary>
        ///     The hands behind the table: the cards the env keeps, which the observation sums up and the frame never shows;
        ///     here the page's own hand, seed 1,574,758.
        /// </summary>
        [TestMethod]
        public void TheHands() {
            #region TheHands
            using Env<object?[], NDArray> env =
                Gymnasium.Make<object?[], NDArray>("Blackjack-v1");
            using var start = env.Reset(seed: 1_574_758);
            var table = (BlackjackEnv)env.Unwrapped;
            // aces are 1; the dealer's second card is the face-down one
            Console.WriteLine($"player {string.Join(" ", table.Player!)}");   // 2 2
            Console.WriteLine($"dealer {string.Join(" ", table.Dealer!)}");   // 1 6
            #endregion
            CollectionAssert.AreEqual(new long[] { 2, 2 }, table.Player!.ToArray());
            CollectionAssert.AreEqual(new long[] { 1, 6 }, table.Dealer!.ToArray());
        }

        /// <summary>
        ///     Casino rules: <c>sab=false, natural=true</c>, where a natural (an ace and a ten) that wins pays 1.5, against
        ///     Blackjack-v1's book rules (<c>sab=true</c>), where it pays 1.
        /// </summary>
        [TestMethod]
        public void CasinoRules() {
            double reward = 0;
            #region CasinoRules
            // Blackjack-v1 plays the book's rules (sab=true): a natural wins 1
            using Env<object?[], NDArray> env =
                Gymnasium.Make<object?[], NDArray>("Blackjack-v1",
                    new { sab = false, natural = true });
            using var start = env.Reset(seed: 5);   // an ace and a ten
            using NDArray stick = NDArray.Scalar(0L);
            using var step = env.Step(stick);
            reward = step.Reward;
            Console.WriteLine(reward);   // 1.5
            #endregion
            Assert.AreEqual(1.5, reward);
            // The same hand under the registered rules pays 1.
            using Env<object?[], NDArray> book = Gymnasium.Make<object?[], NDArray>("Blackjack-v1");
            using var bookStart = book.Reset(seed: 5);
            using NDArray bookStick = NDArray.Scalar(0L);
            using var bookStep = book.Step(bookStick);
            Assert.AreEqual(1.0, bookStep.Reward);
        }

        /// <summary>
        ///     First-visit Monte Carlo prediction, Sutton and Barto's Example 5.1: each state's value under the policy that
        ///     sticks on 20 or 21, the mean return of the hands that pass through it, over 10,000 seeded hands.
        /// </summary>
        /// <remarks>
        ///     A Blackjack state never repeats within a hand (a hit raises the hard sum, or turns a soft hand hard), so every
        ///     visit is a first visit. The only reward comes at the end, so it is every state's return.
        /// </remarks>
        [TestMethod]
        public void MonteCarlo() {
            var returns = new Dictionary<(long, long, long), List<double>>();
            #region MonteCarlo
            using Env<object?[], NDArray> env =
                Gymnasium.Make<object?[], NDArray>("Blackjack-v1");
            for (int seed = 0; seed < 10_000; seed++) {
                using var start = env.Reset(seed: seed);
                BlackjackObservation hand = start.Observation;
                var state = (hand.PlayerSum.GetInt64(), hand.DealerCard.GetInt64(),
                    hand.UsableAce.GetInt64());
                var seen = new List<(long, long, long)>();
                while (true) {
                    seen.Add(state);
                    // the book's policy: stick on 20 or 21, else hit
                    using NDArray action = NDArray.Scalar(state.Item1 >= 20 ? 0L : 1L);
                    using var step = env.Step(action);
                    BlackjackObservation next = step.Observation;
                    state = (next.PlayerSum.GetInt64(), next.DealerCard.GetInt64(),
                        next.UsableAce.GetInt64());
                    if (step.Terminated) {
                        // the only reward comes at the end: every state seen earns it
                        foreach (var s in seen)
                            (returns.TryGetValue(s, out var r) ? r : returns[s] = new()).Add(step.Reward);
                        break;
                    }
                }
            }
            // 20 against a 10: 0.413 from 610 hands; 13 against a 10: -0.597 from 365
            foreach (var key in new[] { (20L, 10L, 0L), (13L, 10L, 0L) })
                Console.WriteLine($"{key}: {returns[key].Average():F3} from {returns[key].Count} hands");
            #endregion
            Assert.AreEqual(0.413, Math.Round(returns[(20, 10, 0)].Average(), 3));
            Assert.AreEqual(610, returns[(20, 10, 0)].Count);
            Assert.AreEqual(-0.597, Math.Round(returns[(13, 10, 0)].Average(), 3));
            Assert.AreEqual(365, returns[(13, 10, 0)].Count);
        }

        /// <summary>
        ///     Many tables at once: Blackjack has no vector env of its own, so <c>MakeVec</c> steps separate envs in turn
        ///     (<c>sync</c>, its default here); the batched observation is a tuple of three arrays, one value per table.
        /// </summary>
        [TestMethod]
        public void ManyTables() {
            #region ManyTables
            // No vector env of its own: MakeVec steps 8 envs in turn ("sync")
            using VectorEnv<object?[], NDArray> envs =
                Gymnasium.MakeVec<object?[], NDArray>("Blackjack-v1", numEnvs: 8);
            using var start = envs.Reset(seed: 42);
            for (int t = 0; t < 100; t++) {
                using NDArray actions = envs.ActionSpace.Sample();   // one per table
                using var step = envs.Step(actions);   // finished hands deal again
            }
            #endregion
            Assert.AreEqual(8, envs.NumEnvs);
            Assert.IsInstanceOfType(envs, typeof(SyncVectorEnv<object?[], NDArray>));
            // The batched tuple: three arrays of eight values each.
            Assert.AreEqual(3, start.Observations.Length);
            Assert.AreEqual(8, ((NDArray)start.Observations[0]!).size);
        }

        /// <summary>
        ///     The Arguments section's calls: kwargs by Gymnasium's names, or the generated options record by C# names; both
        ///     build the same env, here with casino rules.
        /// </summary>
        [TestMethod]
        public void Arguments() {
            #region Arguments
            using IEnv byName = Gymnasium.Make("Blackjack-v1",
                new { sab = false, natural = true });
            using IEnv byRecord = Gymnasium.Make("Blackjack-v1",
                new BlackjackEnvOptions { Sab = false, Natural = true });
            #endregion
            foreach (IEnv env in new[] { byName, byRecord }) {
                var table = (BlackjackEnv)env.Unwrapped;
                Assert.IsTrue(table.Natural);
                Assert.IsFalse(table.Sab);
            }
        }
    }
}
