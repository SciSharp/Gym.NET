using Gym;
using Gym.Envs;
using NumSharp;

using System;
using System.Globalization;
using System.Linq;
using Gym.Environments.Envs.MuJoCo;
using Gym.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;

// Not under Gym.Tests: there the namespace Gym.Tests.Gymnasium (DR-06) shadows the Gymnasium facade, and the samples
// must read Gymnasium.Make(...) exactly as a reader's code does.
namespace Gym.Docs.EnvPageSamples {
    /// <summary>
    ///     The C# samples of Half Cheetah's page on the docs site (<c>docs/environments/mujoco/half_cheetah.md</c>), each the
    ///     body of a test, so the unit gate compiles and runs every line the page shows (DR-105). The page includes them by
    ///     DocFX's code-snippet syntax: a region by name, or for the quick start this file's first four lines (its usings
    ///     and a blank) and the quick start's region, by line range.
    /// </summary>
    /// <remarks>
    ///     Footgun: the quick start's include names line numbers (<c>?range=1-4,…</c>), so the first four lines of this file
    ///     must stay <c>using Gym; using Gym.Envs; using NumSharp;</c> and a blank line, and moving the quick start's lines
    ///     moves the page's range; <c>EnvDocPagesTests</c> checks both, so the gate fails before the page shows the wrong
    ///     lines. Each test asserts after its region only, so the assertions never appear on the page; where a region's
    ///     comment states a number, the assertion pins it: those numbers are Gymnasium 1.4.0's own on MuJoCo 4.0.0 for the
    ///     same seeds and actions (the physics is bit for bit), so the page cannot drift from what either computes. The
    ///     depth sample's number is rounded to two decimals: Gym.NET draws MuJoCo's frames itself, and its depth values
    ///     differ from Gymnasium's in their last bits (DR-98).
    /// </remarks>
    [TestClass]
    public class HalfCheetahSamples {
        /// <summary>The page's quick start: make the env, reset it with a seed, take one random step.</summary>
        [TestMethod]
        public void QuickStart() {
            #region QuickStart
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("HalfCheetah-v5");
            // 17 numbers: the torso's height and pitch, 6 joint angles, 9 velocities
            NDArray observation = env.Reset(seed: 42).Observation;
            // six torques, each from -1 to 1: back thigh, shin, foot, then the front's
            using NDArray action = env.ActionSpace.Sample();
            var (next, reward, terminated, truncated, info) = env.Step(action);
            #endregion
            Assert.AreEqual(17, observation.size);
            // Gymnasium's reset of seed 42: the torso 1.2 cm below its rest height (the reset's uniform noise of 0.1).
            Assert.AreEqual(-0.012224312049589542, observation.GetDouble(0));
            Assert.IsFalse(terminated);
            _ = (reward, truncated, info);
            next.Dispose();
            observation.Dispose();
        }

        /// <summary>Run an episode: random torques for the 1,000 steps of the time limit (Half Cheetah never terminates).</summary>
        [TestMethod]
        public void RunAnEpisode() {
            int steps = 0;
            #region RunAnEpisode
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("HalfCheetah-v5");
            NDArray observation = env.Reset(seed: 42).Observation;
            double episodeReturn = 0;
            while (true) {
                using NDArray action = env.ActionSpace.Sample();
                var (next, reward, terminated, truncated, _) = env.Step(action);
                observation.Dispose();
                observation = next;
                episodeReturn += reward;
                steps++;
                if (terminated || truncated)   // never terminated: 1,000 steps
                    break;
            }
            Console.WriteLine($"return {episodeReturn:F1}");   // random torques: -295.8 on average
            #endregion
            Assert.AreEqual(1000, steps);
        }

        /// <summary>
        ///     Read the reward's terms: one step with every motor at half torque, the reward and the info that splits it, from
        ///     seed 42.
        /// </summary>
        [TestMethod]
        public void RewardTerms() {
            double reward = 0, forward = 0, control = 0;
            #region RewardTerms
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("HalfCheetah-v5");
            using var start = env.Reset(seed: 42);
            using NDArray action = np.full(6, 0.5f);   // every motor at half torque
            using var step = env.Step(action);
            (reward, forward, control) = (step.Reward, step.Info.Get<double>("reward_forward"),
                step.Info.Get<double>("reward_ctrl"));
            // 0.1357 = 0.2857 forward (the speed in m/s) - 0.1500 (0.1 x the torques squared)
            Console.WriteLine($"{reward:F4} = {forward:F4} forward {control:F4} control");
            #endregion
            Assert.AreEqual(0.13566419251323042, reward);
            Assert.AreEqual(0.2856641984736949, forward);
            Assert.AreEqual(-0.15000000596046448, control);
        }

        /// <summary>Start without noise: <c>reset_noise_scale</c> 0 resets every seed to the model's own pose, at rest.</summary>
        [TestMethod]
        public void ResetWithoutNoise() {
            bool allZero = false;
            #region ResetWithoutNoise
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("HalfCheetah-v5",
                new HalfCheetahEnvV5Options { ResetNoiseScale = 0.0 });
            using var start = env.Reset(seed: 42);
            // the model's pose at rest: every angle and velocity 0, the same for every seed
            allZero = start.Observation.ToArray<double>().All(v => v == 0);
            Console.WriteLine(allZero);   // True
            #endregion
            Assert.IsTrue(allZero);
        }

        /// <summary>
        ///     Reweight the reward: twice the forward term and no control cost, so the step's reward is twice the speed.
        /// </summary>
        [TestMethod]
        public void RewardWeights() {
            double reward = 0, velocity = 0;
            #region RewardWeights
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("HalfCheetah-v5",
                new { forward_reward_weight = 2.0, ctrl_cost_weight = 0.0 });
            using var start = env.Reset(seed: 42);
            using NDArray action = np.full(6, 0.5f);
            using var step = env.Step(action);
            (reward, velocity) = (step.Reward, step.Info.Get<double>("x_velocity"));
            Console.WriteLine($"{reward:F4} = 2 x {velocity:F4}");   // 0.5713 = 2 x 0.2857
            #endregion
            Assert.AreEqual(0.5713283969473898, reward);
            Assert.AreEqual(2 * velocity, reward);
        }

        /// <summary>
        ///     Cameras and depth: <c>rgbd_tuple</c> through the model's <c>track</c> camera, the colour frame and the depth of
        ///     the same view; the depth is the raw depth buffer, whose smallest value here is the floor's at the bottom edge.
        /// </summary>
        [TestMethod]
        public void CamerasAndDepth() {
            string nearest = "";
            #region CamerasAndDepth
            // the model's "track" camera follows the torso; rgbd_tuple draws colour and depth together
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("HalfCheetah-v5",
                new { render_mode = "rgbd_tuple", camera_name = "track" });
            using var start = env.Reset(seed: 42);
            using var frames = (RenderResult.RgbDepth)env.Render();
            Console.WriteLine($"{frames.RgbFrame.Shape} {frames.DepthFrame.Shape}");   // (480, 480, 3) (480, 480)
            // MuJoCo's raw depth buffer, 0 at the near plane and 1 at the far one,
            // not metres: the nearest point, the floor at the bottom edge, is 0.97
            using NDArray closest = frames.DepthFrame.min();
            nearest = closest.GetSingle().ToString("F2", CultureInfo.InvariantCulture);
            Console.WriteLine(nearest);   // 0.97
            #endregion
            CollectionAssert.AreEqual(new long[] { 480, 480, 3 }, frames.RgbFrame.shape.Select(d => (long)d).ToArray());
            CollectionAssert.AreEqual(new long[] { 480, 480 }, frames.DepthFrame.shape.Select(d => (long)d).ToArray());
            // The printed line is the page's comment, as Shape prints itself.
            Assert.AreEqual("(480, 480, 3) (480, 480)", $"{frames.RgbFrame.Shape} {frames.DepthFrame.Shape}");
            Assert.AreEqual("0.97", nearest);
        }

        /// <summary>
        ///     Many cheetahs at once: Half Cheetah has no vector env of its own, so <c>MakeVec</c> steps separate envs in turn
        ///     (<c>sync</c>, its default here), each with its own simulation.
        /// </summary>
        [TestMethod]
        public void ManyCheetahs() {
            #region ManyCheetahs
            // No vector env of its own: MakeVec steps 8 envs in turn ("sync")
            using VectorEnv<NDArray, NDArray> envs =
                Gymnasium.MakeVec<NDArray, NDArray>("HalfCheetah-v5", numEnvs: 8);
            using var start = envs.Reset(seed: 42);   // 8 x 17 observations
            for (int t = 0; t < 100; t++) {
                using NDArray actions = envs.ActionSpace.Sample();   // 8 x 6 torques
                using var step = envs.Step(actions);
            }
            #endregion
            Assert.AreEqual(8, envs.NumEnvs);
            Assert.IsInstanceOfType(envs, typeof(SyncVectorEnv<NDArray, NDArray>));
            CollectionAssert.AreEqual(new long[] { 8, 17 }, start.Observations.shape.Select(d => (long)d).ToArray());
        }

        /// <summary>
        ///     The Arguments section's calls: kwargs by Gymnasium's names, or the generated options record by C# names; both
        ///     build the same env, here with the x position kept in the observation and a lighter control cost.
        /// </summary>
        [TestMethod]
        public void Arguments() {
            #region Arguments
            using IEnv byName = Gymnasium.Make("HalfCheetah-v5",
                new { ctrl_cost_weight = 0.05, exclude_current_positions_from_observation = false });
            using IEnv byRecord = Gymnasium.Make("HalfCheetah-v5",
                new HalfCheetahEnvV5Options { CtrlCostWeight = 0.05, ExcludeCurrentPositionsFromObservation = false });
            #endregion
            foreach (IEnv env in new[] { byName, byRecord }) {
                var cheetah = (HalfCheetahEnvV5)env.Unwrapped;
                Assert.AreEqual(0.05, cheetah.CtrlCostWeight);
                Assert.AreEqual("Box(-inf, inf, (18,), float64)", env.ObservationSpace.ToString());
            }
        }
    }
}
