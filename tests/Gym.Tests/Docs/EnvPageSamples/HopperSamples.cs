using Gym;
using Gym.Envs;
using NumSharp;

using System;
using System.Globalization;
using System.Linq;
using Gym.Environments.Envs.MuJoCo;
using Gym.Observations;
using Gym.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;

// Not under Gym.Tests: there the namespace Gym.Tests.Gymnasium (DR-06) shadows the Gymnasium facade, and the samples
// must read Gymnasium.Make(...) exactly as a reader's code does.
namespace Gym.Docs.EnvPageSamples {
    /// <summary>
    ///     The C# samples of Hopper's page on the docs site (<c>docs/environments/mujoco/hopper.md</c>), each the body of a
    ///     test, so the unit gate compiles and runs every line the page shows (DR-105). The page includes them by DocFX's
    ///     code-snippet syntax: a region by name, or for the quick start this file's first four lines (its usings and a
    ///     blank) and the quick start's region, by line range.
    /// </summary>
    /// <remarks>
    ///     Footgun: the quick start's include names line numbers (<c>?range=1-4,…</c>), so the first four lines of this file
    ///     must stay <c>using Gym; using Gym.Envs; using NumSharp;</c> and a blank line, and moving the quick start's lines
    ///     moves the page's range; <c>EnvDocPagesTests</c> checks both, so the gate fails before the page shows the wrong
    ///     lines. Each test asserts after its region only, so the assertions never appear on the page; where a region's
    ///     comment states a number, the assertion pins it: those numbers are Gymnasium 1.4.0's own on MuJoCo 4.0.0 for the
    ///     same seeds and actions (the physics, and the seeded action space's samples, are bit for bit), so the page cannot
    ///     drift from what either computes. The depth sample's number is rounded to two decimals: Gym.NET draws MuJoCo's
    ///     frames itself, and its depth values differ from Gymnasium's in their last bits (DR-98).
    /// </remarks>
    [TestClass]
    public class HopperSamples {
        /// <summary>The page's quick start: make the env, reset it with a seed, take one random step.</summary>
        [TestMethod]
        public void QuickStart() {
            #region QuickStart
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Hopper-v5");
            // 11 numbers: the torso's height and angle, 3 joint angles, 6 velocities
            (NDArray observation, Info info) = env.Reset(seed: 42);
            // three torques, each from -1 to 1: thigh, leg, foot
            using NDArray action = env.ActionSpace.Sample();
            observation.Dispose();
            (observation, double reward, bool terminated, bool truncated, info)
                = env.Step(action);
            #endregion
            // The quick start's first observation, read again: the same seeded reset gives the same one.
            using ResetResult<NDArray> again = env.Reset(seed: 42);
            NDArray first = again.Observation;
            Assert.AreEqual(11, first.size);
            // Gymnasium's reset of seed 42: the torso at 1.2494 m, the model's 1.25 less the reset's noise.
            Assert.AreEqual(1.2493887843975204, first.GetDouble(0));
            _ = (reward, terminated, truncated, info);
            observation.Dispose();
        }

        /// <summary>Run an episode: random torques until the hopper falls or the 1,000-step limit.</summary>
        [TestMethod]
        public void RunAnEpisode() {
            int steps = 0;
            #region RunAnEpisode
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Hopper-v5");
            (NDArray observation, Info info) = env.Reset(seed: 42);
            double episodeReturn = 0;
            while (true) {
                using NDArray action = env.ActionSpace.Sample();
                observation.Dispose();
                (observation, double reward, bool terminated, bool truncated, info)
                    = env.Step(action);
                episodeReturn += reward;
                steps++;
                if (terminated || truncated)
                    break;
            }
            // random torques fall within about 21 steps: 17.0 on average
            Console.WriteLine($"{steps} steps, return {episodeReturn:F1}");
            #endregion
            Assert.IsTrue(steps <= 1000);
        }

        /// <summary>
        ///     Read the reward's terms: one step with every motor at half torque, the reward and the info that splits it, from
        ///     seed 42.
        /// </summary>
        [TestMethod]
        public void RewardTerms() {
            double reward = 0, forward = 0, healthy = 0, control = 0;
            #region RewardTerms
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Hopper-v5");
            using ResetResult<NDArray> start = env.Reset(seed: 42);
            using NDArray action = np.full(3, 0.5f);   // every motor at half torque
            using StepResult<NDArray> step = env.Step(action);
            (reward, forward, healthy, control) = (step.Reward, step.Info.Get<double>("reward_forward"),
                step.Info.Get<double>("reward_survive"), step.Info.Get<double>("reward_ctrl"));
            // 1.0381 = 0.0388 forward + 1.0000 healthy - 0.0008 control
            Console.WriteLine($"{reward:F4} = {forward:F4} forward + {healthy:F4} healthy {control:F4} control");
            #endregion
            Assert.AreEqual(1.0380970222960657, reward);
            Assert.AreEqual(0.038847022302584966, forward);
            Assert.AreEqual(1.0, healthy);
            Assert.AreEqual(-0.000750000006519258, control);
        }

        /// <summary>Start without noise: <c>reset_noise_scale</c> 0 resets every seed to the model's pose, at rest.</summary>
        [TestMethod]
        public void ResetWithoutNoise() {
            double height = 0;
            bool restZero = false;
            #region ResetWithoutNoise
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Hopper-v5",
                new HopperEnvV5Options { ResetNoiseScale = 0.0 });
            using ResetResult<NDArray> start = env.Reset(seed: 42);
            (NDArray observation, Info info) = start;
            double[] o = observation.ToArray<double>();
            (height, restZero) = (o[0], o.Skip(1).All(v => v == 0));
            // 1.25, True: the torso 1.25 m up, every angle and velocity 0
            Console.WriteLine($"{height}, {restZero}");
            #endregion
            Assert.AreEqual(1.25, height);
            Assert.IsTrue(restZero);
        }

        /// <summary>
        ///     Keep going after a fall: <c>terminate_when_unhealthy</c> false runs all 1,000 steps, and the healthy reward is
        ///     paid only on the steps the hopper is healthy; seeded random torques.
        /// </summary>
        [TestMethod]
        public void KeepGoing() {
            int steps = 0, healthy = 0;
            double episodeReturn = 0;
            #region KeepGoing
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Hopper-v5",
                new { terminate_when_unhealthy = false });
            var hopper = (HopperEnvV5)env.Unwrapped;
            using ResetResult<NDArray> start = env.Reset(seed: 0);
            env.ActionSpace.Seed(0);
            while (true) {
                using NDArray action = env.ActionSpace.Sample();
                using StepResult<NDArray> step = env.Step(action);
                (steps, episodeReturn) = (steps + 1, episodeReturn + step.Reward);
                healthy += hopper.IsHealthy ? 1 : 0;
                if (step.Terminated || step.Truncated)
                    break;
            }
            // 1000 steps, healthy for 98, return 236.0: it falls at step 26
            Console.WriteLine($"{steps} steps, healthy for {healthy}, return {episodeReturn:F1}");
            #endregion
            Assert.AreEqual(1000, steps);
            Assert.AreEqual(98, healthy);
            Assert.AreEqual(235.9526454534169, episodeReturn);
        }

        /// <summary>
        ///     Cameras and depth: <c>rgbd_tuple</c> through the model's <c>track</c> camera, the colour frame and the depth of
        ///     the same view; the depth is the raw depth buffer, whose smallest value here is the floor's at the bottom edge.
        /// </summary>
        [TestMethod]
        public void CamerasAndDepth() {
            string nearest = "";
            #region CamerasAndDepth
            // the model's "track" camera follows the hopper; rgbd_tuple draws colour and depth together
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Hopper-v5",
                new { render_mode = "rgbd_tuple", camera_name = "track" });
            using ResetResult<NDArray> start = env.Reset(seed: 42);
            using var frames = (RenderResult.RgbDepth)env.Render();
            Console.WriteLine($"{frames.RgbFrame.Shape} {frames.DepthFrame.Shape}");   // (480, 480, 3) (480, 480)
            // MuJoCo's raw depth buffer, 0 at the near plane and 1 at the far one,
            // not metres: the nearest point, the floor at the bottom edge, is 0.97
            using NDArray closest = frames.DepthFrame.min();
            nearest = closest.GetSingle().ToString("F2", CultureInfo.InvariantCulture);
            Console.WriteLine(nearest);   // 0.97
            #endregion
            // The printed line is the page's comment, as Shape prints itself.
            Assert.AreEqual("(480, 480, 3) (480, 480)", $"{frames.RgbFrame.Shape} {frames.DepthFrame.Shape}");
            Assert.AreEqual("0.97", nearest);
        }

        /// <summary>
        ///     Many hoppers at once: Hopper has no vector env of its own, so <c>MakeVec</c> steps separate envs in turn
        ///     (<c>sync</c>, its default here), each with its own simulation; a fallen hopper resets itself.
        /// </summary>
        [TestMethod]
        public void ManyHoppers() {
            #region ManyHoppers
            // No vector env of its own: MakeVec steps 8 envs in turn ("sync")
            using VectorEnv<NDArray, NDArray> envs
                = Gymnasium.MakeVec<NDArray, NDArray>("Hopper-v5", numEnvs: 8);
            using VectorResetResult<NDArray> start
                = envs.Reset(seed: 42);   // 8 x 11 observations
            (NDArray observations, Info infos) = start;
            for (int t = 0; t < 100; t++) {
                using NDArray actions = envs.ActionSpace.Sample();   // 8 x 3 torques
                using VectorStepResult<NDArray> step
                    = envs.Step(actions);   // fallen hoppers reset themselves
            }
            #endregion
            Assert.AreEqual(8, envs.NumEnvs);
            Assert.IsInstanceOfType(envs, typeof(SyncVectorEnv<NDArray, NDArray>));
            CollectionAssert.AreEqual(new long[] { 8, 11 }, observations.shape.Select(d => (long)d).ToArray());
        }

        /// <summary>
        ///     The Arguments section's calls: kwargs by Gymnasium's names, or the generated options record by C# names; both
        ///     build the same env, here with half the healthy reward and a lower floor for a healthy height.
        /// </summary>
        [TestMethod]
        public void Arguments() {
            #region Arguments
            using IEnv byName = Gymnasium.Make("Hopper-v5",
                new { healthy_reward = 0.5, healthy_z_range = (0.5, double.PositiveInfinity) });
            using IEnv byRecord = Gymnasium.Make("Hopper-v5",
                new HopperEnvV5Options { HealthyReward = 0.5, HealthyZRange = (0.5, double.PositiveInfinity) });
            #endregion
            foreach (IEnv env in new[] { byName, byRecord }) {
                var hopper = (HopperEnvV5)env.Unwrapped;
                Assert.AreEqual(0.5, hopper.HealthyRewardWeight);
                Assert.AreEqual((0.5, double.PositiveInfinity), hopper.HealthyZRange);
            }
        }
    }
}
