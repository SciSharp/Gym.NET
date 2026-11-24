using Gym;
using Gym.Envs;
using NumSharp;

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Gym.Environments.Envs.MuJoCo;
using Gym.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;

// Not under Gym.Tests: there the namespace Gym.Tests.Gymnasium (DR-06) shadows the Gymnasium facade, and the samples
// must read Gymnasium.Make(...) exactly as a reader's code does.
namespace Gym.Docs.EnvPageSamples {
    /// <summary>
    ///     The C# samples of InvertedDoublePendulum's page on the docs site
    ///     (<c>docs/environments/mujoco/inverted_double_pendulum.md</c>), each the body of a test, so the unit gate compiles
    ///     and runs every line the page shows (DR-105). The page includes them by DocFX's code-snippet syntax: a region by
    ///     name, or for the quick start this file's first four lines (its usings and a blank) and the quick start's region,
    ///     by line range.
    /// </summary>
    /// <remarks>
    ///     Footgun: the quick start's include names line numbers (<c>?range=1-4,…</c>), so the first four lines of this file
    ///     must stay <c>using Gym; using Gym.Envs; using NumSharp;</c> and a blank line, and moving the quick start's lines
    ///     moves the page's range; <c>EnvDocPagesTests</c> checks both, so the gate fails before the page shows the wrong
    ///     lines. Each test asserts after its region only, so the assertions never appear on the page; where a region's
    ///     comment states a number, the assertion pins it: those numbers are Gymnasium 1.4.0's own on MuJoCo 4.0.0 for the
    ///     same seeds and actions (the physics is bit for bit), so the page cannot drift from what either computes. The
    ///     depth sample's number is rounded to three decimals: Gym.NET draws MuJoCo's frames itself, and its depth values
    ///     differ from Gymnasium's in their last bits (DR-98).
    /// </remarks>
    [TestClass]
    public class InvertedDoublePendulumSamples {
        /// <summary>The page's quick start: make the env, reset it with a seed, take one random step.</summary>
        [TestMethod]
        public void QuickStart() {
            #region QuickStart
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("InvertedDoublePendulum-v5");
            // 9 numbers: the cart's position, the poles' sines and cosines, 3 speeds, the rail's force
            NDArray observation = env.Reset(seed: 42).Observation;
            // one force, from -1 to 1: push the cart left or right
            using NDArray action = env.ActionSpace.Sample();
            var (next, reward, terminated, truncated, info) = env.Step(action);
            #endregion
            Assert.AreEqual(9, observation.size);
            // Gymnasium's reset of seed 42: the cart at 0.0548 m, the reset's uniform noise of 0.1.
            Assert.AreEqual(0.05479120971119267, observation.GetDouble(0));
            _ = (reward, terminated, truncated, info);
            next.Dispose();
            observation.Dispose();
        }

        /// <summary>Run an episode: random pushes until the tip drops to 1 m or the 1,000-step limit.</summary>
        [TestMethod]
        public void RunAnEpisode() {
            int steps = 0;
            #region RunAnEpisode
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("InvertedDoublePendulum-v5");
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
            // random pushes drop the tip within about 6 steps: 44.6 on average
            Console.WriteLine($"{steps} steps, return {episodeReturn:F1}");
            #endregion
            Assert.IsTrue(steps <= 1000);
        }

        /// <summary>Read the reward's terms: one step with a half push from seed 42, the reward and its three terms.</summary>
        [TestMethod]
        public void RewardTerms() {
            double reward = 0, alive = 0, distance = 0, velocity = 0;
            #region RewardTerms
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("InvertedDoublePendulum-v5");
            using var start = env.Reset(seed: 42);
            using NDArray action = np.array(new[] { 0.5f });   // half a push to the right
            using var step = env.Step(action);
            (reward, alive, distance, velocity) = (step.Reward, step.Info.Get<double>("reward_survive"),
                step.Info.Get<double>("distance_penalty"), step.Info.Get<double>("velocity_penalty"));
            // 9.3082 = 10.0000 alive - 0.6448 distance - 0.0469 velocity
            Console.WriteLine($"{reward:F4} = {alive:F4} alive {distance:F4} distance {velocity:F4} velocity");
            #endregion
            Assert.AreEqual(9.308231516039633, reward);
            Assert.AreEqual(10.0, alive);
            Assert.AreEqual(-0.64483360357413, distance);
            Assert.AreEqual(-0.046934880386237145, velocity);
        }

        /// <summary>
        ///     The best a step can pay: from the reset without noise, with no push, both poles stand still and upright, and the
        ///     step pays 9.36: the 10 for staying up less the distance penalty, which measures the 1.2 m tip from 2 m.
        /// </summary>
        [TestMethod]
        public void BestStep() {
            double reward = 0;
            #region BestStep
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("InvertedDoublePendulum-v5",
                new InvertedDoublePendulumEnvV5Options { ResetNoiseScale = 0.0 });
            using var start = env.Reset(seed: 42);   // both poles upright, at rest
            using NDArray rest = np.array(new[] { 0f });
            using var step = env.Step(rest);
            reward = step.Reward;
            // 9.3600: 10 - (1.2 - 2)^2, the tip 1.2 m up measured from 2 m
            Console.WriteLine($"{reward:F4}");
            #endregion
            Assert.AreEqual(9.36, reward);
        }

        /// <summary>
        ///     Start without noise: <c>reset_noise_scale</c> 0 resets every seed to the cart in the middle and both poles
        ///     upright: every value 0 but the angles' cosines, 1.
        /// </summary>
        [TestMethod]
        public void ResetWithoutNoise() {
            string cosines = "";
            bool restZero = false;
            #region ResetWithoutNoise
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("InvertedDoublePendulum-v5",
                new InvertedDoublePendulumEnvV5Options { ResetNoiseScale = 0.0 });
            using var start = env.Reset(seed: 42);
            double[] o = start.Observation.ToArray<double>();
            cosines = string.Create(CultureInfo.InvariantCulture, $"{o[3]:F1}, {o[4]:F1}");
            restZero = o.Where((v, i) => i != 3 && i != 4).All(v => v == 0);
            // 1.0, 1.0, True: both poles upright (their cosines 1), every other value 0
            Console.WriteLine($"{cosines}, {restZero}");
            #endregion
            Assert.AreEqual("1.0, 1.0", cosines);
            Assert.IsTrue(restZero);
        }

        /// <summary>
        ///     Cameras and depth: Gymnasium's v5 sets no camera for this env, so the sample places the free camera as the video
        ///     does, 2.6 m from a point 0.55 m above the rail; <c>rgbd_tuple</c> draws colour and depth together.
        /// </summary>
        [TestMethod]
        public void CamerasAndDepth() {
            string nearest = "";
            #region CamerasAndDepth
            // no camera set for v5: place the free camera as the video does; rgbd_tuple draws colour and depth together
            var view = new Dictionary<string, object?> {
                ["trackbodyid"] = 0L, ["distance"] = 2.6, ["lookat"] = new[] { 0.0, 0.0, 0.55 }, ["elevation"] = -5.0,
            };
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("InvertedDoublePendulum-v5",
                new Dictionary<string, object?> { ["render_mode"] = "rgbd_tuple", ["default_camera_config"] = view });
            using var start = env.Reset(seed: 42);
            using var frames = (RenderResult.RgbDepth)env.Render();
            Console.WriteLine($"{frames.RgbFrame.Shape} {frames.DepthFrame.Shape}");   // (480, 480, 3) (480, 480)
            // MuJoCo's raw depth buffer, 0 at the near plane and 1 at the far one,
            // not metres: the nearest point, on the upper pole, is 0.967
            using NDArray closest = frames.DepthFrame.min();
            nearest = closest.GetSingle().ToString("F3", CultureInfo.InvariantCulture);
            Console.WriteLine(nearest);   // 0.967
            #endregion
            // The printed line is the page's comment, as Shape prints itself.
            Assert.AreEqual("(480, 480, 3) (480, 480)", $"{frames.RgbFrame.Shape} {frames.DepthFrame.Shape}");
            Assert.AreEqual("0.967", nearest);
        }

        /// <summary>
        ///     Many pendulums at once: InvertedDoublePendulum has no vector env of its own, so <c>MakeVec</c> steps separate
        ///     envs in turn (<c>sync</c>, its default here), each with its own simulation and its own resets.
        /// </summary>
        [TestMethod]
        public void ManyPendulums() {
            #region ManyPendulums
            // No vector env of its own: MakeVec steps 8 envs in turn ("sync")
            using VectorEnv<NDArray, NDArray> envs =
                Gymnasium.MakeVec<NDArray, NDArray>("InvertedDoublePendulum-v5", numEnvs: 8);
            using var start = envs.Reset(seed: 42);   // 8 x 9 observations
            for (int t = 0; t < 100; t++) {
                using NDArray actions = envs.ActionSpace.Sample();   // 8 x 1 forces
                using var step = envs.Step(actions);   // fallen pendulums reset themselves
            }
            #endregion
            Assert.AreEqual(8, envs.NumEnvs);
            Assert.IsInstanceOfType(envs, typeof(SyncVectorEnv<NDArray, NDArray>));
            CollectionAssert.AreEqual(new long[] { 8, 9 }, start.Observations.shape.Select(d => (long)d).ToArray());
        }

        /// <summary>
        ///     The Arguments section's calls: kwargs by Gymnasium's names, or the generated options record by C# names; both
        ///     build the same env, here with half the alive bonus and half the starting noise.
        /// </summary>
        [TestMethod]
        public void Arguments() {
            #region Arguments
            using IEnv byName = Gymnasium.Make("InvertedDoublePendulum-v5",
                new { healthy_reward = 5.0, reset_noise_scale = 0.05 });
            using IEnv byRecord = Gymnasium.Make("InvertedDoublePendulum-v5",
                new InvertedDoublePendulumEnvV5Options { HealthyReward = 5.0, ResetNoiseScale = 0.05 });
            #endregion
            foreach (IEnv env in new[] { byName, byRecord }) {
                var pendulum = (InvertedDoublePendulumEnvV5)env.Unwrapped;
                Assert.AreEqual(5.0, pendulum.HealthyReward);
                Assert.AreEqual(0.05, pendulum.ResetNoiseScale);
            }
        }
    }
}
