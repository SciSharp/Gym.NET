using Gym;
using Gym.Envs;
using NumSharp;

// Below the four lines the quick start shows: the variations' kwargs are object?, which the test project's default (no
// nullable context) would flag.
#nullable enable
using System;
using System.Collections.Generic;
using System.IO;
using Gym.Environments.Envs.Box2D;
using Gym.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SixLabors.ImageSharp;

// Not under Gym.Tests: there the namespace Gym.Tests.Gymnasium (DR-06) shadows the Gymnasium facade, and the samples
// must read Gymnasium.Make(...) exactly as a reader's code does.
namespace Gym.Docs.EnvPageSamples {
    /// <summary>
    ///     The C# samples of Lunar Lander's page on the docs site (<c>docs/environments/box2d/lunar_lander.md</c>), each
    ///     the body of a test, so the unit gate compiles and runs every line the page shows (DR-105). The page includes
    ///     them by DocFX's code-snippet syntax: a region by name, or for the quick start this file's first four lines (its
    ///     usings and a blank) and the quick start's region, by line range.
    /// </summary>
    /// <remarks>
    ///     Footgun: the quick start's include names line numbers (<c>?range=1-4,…</c>), so the first four lines of this file
    ///     must stay <c>using Gym; using Gym.Envs; using NumSharp;</c> and a blank line, and moving the quick start's lines
    ///     moves the page's range; <c>EnvDocPagesTests</c> checks both, so the gate fails before the page shows the wrong
    ///     lines. Each test asserts after its region only, so the assertions never appear on the page; where a region's
    ///     comment states a number (a step count, a return), the assertion pins it: those numbers are Gymnasium 1.4.0's own
    ///     for the same seeds, measured with its heuristic, so the page cannot drift from what either computes.
    /// </remarks>
    [TestClass]
    public class LunarLanderSamples {
        /// <summary>The page's quick start: make the env, reset it with a seed, take one random step.</summary>
        [TestMethod]
        public void QuickStart() {
            #region QuickStart
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("LunarLander-v3");
            // x, y, vx, vy, angle, angular velocity, right leg, left leg
            NDArray observation = env.Reset(seed: 42).Observation;
            // 0 nothing, 1 left engine, 2 main engine, 3 right engine
            using NDArray action = env.ActionSpace.Sample();
            var (next, reward, terminated, truncated, info) = env.Step(action);
            #endregion
            Assert.AreEqual(8, observation.size);
            // Gymnasium's reset of seed 42 (one free step inside the reset): the lander high above the pad, at 1.418.
            Assert.AreEqual(1.4181306f, observation.GetSingle(1));
            Assert.IsFalse(terminated);
            Assert.IsFalse(truncated);
            Assert.AreEqual(0, info.Count);
            _ = reward;
            next.Dispose();
            observation.Dispose();
        }

        /// <summary>Run an episode: random engines until the lander crashes, flies off, rests, or the 1,000-step limit.</summary>
        [TestMethod]
        public void RunAnEpisode() {
            int steps = 0;
            #region RunAnEpisode
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("LunarLander-v3");
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
            Console.WriteLine($"return {episodeReturn:F1}");   // random engines crash: -191 on average
            #endregion
            Assert.IsTrue(steps <= 1000);
        }

        /// <summary>
        ///     The heuristic: Gymnasium's own controller, <see cref="LunarLanderEnv.Heuristic"/>, which lands the page's
        ///     episode from seed 7.
        /// </summary>
        [TestMethod]
        public void TheHeuristic() {
            int steps = 0;
            double episodeReturn = 0;
            #region TheHeuristic
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("LunarLander-v3");
            NDArray observation = env.Reset(seed: 7).Observation;
            while (true) {
                // Gymnasium's own controller: lean toward the pad, slow the fall
                using NDArray action = LunarLanderEnv.Heuristic(env, observation);
                var (next, reward, terminated, truncated, _) = env.Step(action);
                observation.Dispose();
                observation = next;
                (steps, episodeReturn) = (steps + 1, episodeReturn + reward);
                if (terminated || truncated)
                    break;
            }
            // 299 steps, return 248.6: landed and at rest, +100
            Console.WriteLine($"{steps} steps, return {episodeReturn:F1}");
            #endregion
            Assert.AreEqual(299, steps);
            Assert.AreEqual(248.60156139138357, episodeReturn);
        }

        /// <summary>
        ///     The heuristic under three variations, from seed 7: continuous engines (LunarLanderContinuous-v3), wind and
        ///     turbulence, and the Moon's own gravity, each its own numbers, Gymnasium's for the same seed.
        /// </summary>
        [TestMethod]
        public void Variations() {
            var results = new List<(int Steps, double Return)>();
            #region Variations
            // continuous engines, wind and turbulence, the Moon's own gravity
            var variations = new (string Id, object? Kwargs)[] {
                ("LunarLanderContinuous-v3", null),
                ("LunarLander-v3", new { enable_wind = true }),
                ("LunarLander-v3", new { gravity = -1.62 }),
            };
            foreach (var (id, kwargs) in variations) {
                using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>(id, kwargs);
                NDArray observation = env.Reset(seed: 7).Observation;
                int steps = 0;
                double episodeReturn = 0;
                while (true) {
                    using NDArray action = LunarLanderEnv.Heuristic(env, observation);
                    var (next, reward, terminated, truncated, _) = env.Step(action);
                    observation.Dispose();
                    observation = next;
                    (steps, episodeReturn) = (steps + 1, episodeReturn + reward);
                    if (terminated || truncated)
                        break;
                }
                // 187 steps, 273.5; 379 steps, 197.8; 564 steps, 285.7
                Console.WriteLine($"{id}: {steps} steps, return {episodeReturn:F1}");
                results.Add((steps, episodeReturn));
            }
            #endregion
            Assert.AreEqual((187, 273.54317884234985), results[0]);
            Assert.AreEqual((379, 197.75352916816047), results[1]);
            Assert.AreEqual((564, 285.7146133876994), results[2]);
        }

        /// <summary>
        ///     Throttle the engines by hand: one step of LunarLanderContinuous-v3 with the main engine at 75% and the left
        ///     one at 80%, the continuous action's mapping from throttle to power.
        /// </summary>
        [TestMethod]
        public void Throttles() {
            double reward = 0;
            #region Throttles
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("LunarLanderContinuous-v3");
            using var start = env.Reset(seed: 7);
            // main 0.5: 75% (0 to 1 maps to 50% to 100%); side -0.8: the left
            // engine at 80% (beyond -0.5 the left fires, beyond 0.5 the right)
            using NDArray action = np.array(new[] { 0.5f, -0.8f });
            using var step = env.Step(action);
            reward = step.Reward;
            Console.WriteLine($"{reward:F4}");   // -0.1445, the fall slowed, the fuel paid
            #endregion
            Assert.AreEqual(-0.1445076316071572, reward);
        }

        /// <summary>
        ///     Save a frame: an <c>rgb_array</c> render as an image file, the reset's frame of seed 42, the lander high above
        ///     the pad; its exhaust is drawn only in a render mode, as in Gymnasium.
        /// </summary>
        [TestMethod]
        public void SaveFrame() {
            #region SaveFrame
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("LunarLander-v3", RenderMode.RgbArray);
            using var start = env.Reset(seed: 42);
            using RenderResult result = env.Render();
            using Image frame = (Image)result;   // 600x400, Gymnasium's pixels
            frame.SaveAsPng("lunar_lander.png");
            #endregion
            // The test's working folder is its output folder; the file is removed so runs leave nothing behind.
            Assert.AreEqual(600, frame.Width);
            Assert.AreEqual(400, frame.Height);
            Assert.IsTrue(File.Exists("lunar_lander.png"));
            File.Delete("lunar_lander.png");
        }

        /// <summary>
        ///     Many landers at once: Lunar Lander has no vector env of its own, so <c>MakeVec</c> steps separate envs in turn
        ///     (<c>sync</c>, its default here).
        /// </summary>
        [TestMethod]
        public void ManyLanders() {
            #region ManyLanders
            // No vector env of its own: MakeVec steps 8 envs in turn ("sync")
            using VectorEnv<NDArray, NDArray> envs =
                Gymnasium.MakeVec<NDArray, NDArray>("LunarLander-v3", numEnvs: 8);
            using var start = envs.Reset(seed: 42);
            for (int t = 0; t < 100; t++) {
                using NDArray actions = envs.ActionSpace.Sample();   // one per lander
                using var step = envs.Step(actions);   // finished landers reset themselves
            }
            #endregion
            Assert.AreEqual(8, envs.NumEnvs);
            Assert.IsInstanceOfType(envs, typeof(SyncVectorEnv<NDArray, NDArray>));
        }

        /// <summary>
        ///     The Arguments section's calls: kwargs by Gymnasium's names, or the generated options record by C# names; both
        ///     build the same env, here with wind.
        /// </summary>
        [TestMethod]
        public void Arguments() {
            #region Arguments
            using IEnv byName = Gymnasium.Make("LunarLander-v3",
                new { enable_wind = true, wind_power = 10.0, turbulence_power = 1.0 });
            using IEnv byRecord = Gymnasium.Make("LunarLander-v3",
                new LunarLanderEnvOptions { EnableWind = true, WindPower = 10.0, TurbulencePower = 1.0 });
            #endregion
            foreach (IEnv env in new[] { byName, byRecord }) {
                var lander = (LunarLanderEnv)env.Unwrapped;
                Assert.IsTrue(lander.EnableWind);
                Assert.AreEqual(10.0, lander.WindPower);
                Assert.AreEqual(1.0, lander.TurbulencePower);
            }
        }
    }
}
