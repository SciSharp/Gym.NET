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
    ///     The C# samples of Humanoid's page on the docs site (<c>docs/environments/mujoco/humanoid.md</c>), each the body of
    ///     a test, so the unit gate compiles and runs every line the page shows (DR-105). The page includes them by DocFX's
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
    public class HumanoidSamples {
        /// <summary>The page's quick start: make the env, reset it with a seed, take one random step.</summary>
        [TestMethod]
        public void QuickStart() {
            #region QuickStart
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Humanoid-v5");
            // 348 numbers: the pose and velocities, then the bodies' inertias, velocities and forces
            (NDArray observation, Info info) = env.Reset(seed: 42);
            // seventeen torques, each from -0.4 to 0.4: abdomen, legs, arms
            using NDArray action = env.ActionSpace.Sample();
            observation.Dispose();
            (observation, double reward, bool terminated, bool truncated, info)
                = env.Step(action);
            #endregion
            // The quick start's first observation, read again: the same seeded reset gives the same one.
            using ResetResult<NDArray> again = env.Reset(seed: 42);
            NDArray first = again.Observation;
            Assert.AreEqual(348, first.size);
            // Gymnasium's reset of seed 42: the torso at 1.4072 m, the model's 1.4 plus the reset's uniform noise.
            Assert.AreEqual(1.4071719583982276, first.GetDouble(0));
            _ = (reward, terminated, truncated, info);
            observation.Dispose();
        }

        /// <summary>Run an episode: random torques until the humanoid falls or the 1,000-step limit.</summary>
        [TestMethod]
        public void RunAnEpisode() {
            int steps = 0;
            #region RunAnEpisode
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Humanoid-v5");
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
            // random torques fall within about 25 steps: 117.8 on average
            Console.WriteLine($"{steps} steps, return {episodeReturn:F1}");
            #endregion
            Assert.IsTrue(steps <= 1000);
        }

        /// <summary>
        ///     Read the reward's terms: every motor at half torque from seed 42; the eleventh step is the one the feet land
        ///     on, so the contact cost shows its size.
        /// </summary>
        [TestMethod]
        public void RewardTerms() {
            double reward = 0, forward = 0, healthy = 0, control = 0, contact = 0;
            #region RewardTerms
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Humanoid-v5");
            using ResetResult<NDArray> start = env.Reset(seed: 42);
            using NDArray action = np.full(17, 0.2f);   // every motor at half torque
            for (int t = 1; t < 11; t++) {
                using StepResult<NDArray> falling
                    = env.Step(action);   // the humanoid starts 0.1 m above the floor
            }
            using StepResult<NDArray> step = env.Step(action);
            (reward, forward, healthy, control, contact) = (step.Reward, step.Info.Get<double>("reward_forward"),
                step.Info.Get<double>("reward_survive"), step.Info.Get<double>("reward_ctrl"),
                step.Info.Get<double>("reward_contact"));
            // the feet land: 1.3454 = 0.0354 forward + 5.0000 healthy - 0.0680 control - 3.6220 contact
            Console.WriteLine($"{reward:F4} = {forward:F4} forward + {healthy:F4} healthy {control:F4} control {contact:F4} contact");
            #endregion
            Assert.AreEqual(1.3454175243067148, reward);
            Assert.AreEqual(0.035445558454855544, forward);
            Assert.AreEqual(5.0, healthy);
            Assert.AreEqual(-0.06800000202655794, control);
            Assert.AreEqual(-3.622028032121583, contact);
        }

        /// <summary>
        ///     Start without noise: <c>reset_noise_scale</c> 0 resets every seed to the model's pose, the torso 1.4 m up and
        ///     unrotated, every joint at 0 and at rest; the bodies' inertias in the observation are never 0.
        /// </summary>
        [TestMethod]
        public void ResetWithoutNoise() {
            double height = 0, w = 0;
            bool still = false;
            #region ResetWithoutNoise
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Humanoid-v5",
                new HumanoidEnvV5Options { ResetNoiseScale = 0.0 });
            using ResetResult<NDArray> start = env.Reset(seed: 42);
            (NDArray observation, Info info) = start;
            double[] o = observation.ToArray<double>();
            (height, w, still) = (o[0], o[1], o.Skip(2).Take(43).All(v => v == 0));
            // 1.40, 1.00, True: the torso 1.4 m up and unrotated, every joint at 0 and at rest (items 2 to 44)
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"{height:F2}, {w:F2}, {still}"));
            #endregion
            Assert.AreEqual(1.4, height);
            Assert.AreEqual(1.0, w);
            Assert.IsTrue(still);
        }

        /// <summary>
        ///     Keep going after a fall: <c>terminate_when_unhealthy</c> false runs the whole time limit and pays the healthy
        ///     bonus only on healthy steps; a seeded action space makes the run repeatable.
        /// </summary>
        [TestMethod]
        public void KeepGoing() {
            int steps = 0, healthy = 0;
            double episodeReturn = 0;
            #region KeepGoing
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Humanoid-v5",
                new { terminate_when_unhealthy = false });
            var humanoid = (HumanoidEnvV5)env.Unwrapped;
            using ResetResult<NDArray> start = env.Reset(seed: 0);
            env.ActionSpace.Seed(0);
            while (true) {
                using NDArray action = env.ActionSpace.Sample();
                using StepResult<NDArray> step = env.Step(action);
                (steps, episodeReturn) = (steps + 1, episodeReturn + step.Reward);
                healthy += humanoid.IsHealthy ? 1 : 0;
                if (step.Terminated || step.Truncated)
                    break;
            }
            // 1000 steps, healthy for 20, return -276.3: it falls at step 21 and lies on the floor
            Console.WriteLine($"{steps} steps, healthy for {healthy}, return {episodeReturn:F1}");
            #endregion
            Assert.AreEqual(1000, steps);
            Assert.AreEqual(20, healthy);
            Assert.AreEqual(-276.33562749039413, episodeReturn);
        }

        /// <summary>
        ///     Cameras and depth: <c>rgbd_tuple</c> through the model's <c>track</c> camera, the colour frame and the depth of
        ///     the same view; the depth is the raw depth buffer, whose smallest value here is the floor's at the bottom edge.
        /// </summary>
        [TestMethod]
        public void CamerasAndDepth() {
            string nearest = "";
            #region CamerasAndDepth
            // the model's "track" camera follows the humanoid from the side; rgbd_tuple draws colour and depth together
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Humanoid-v5",
                new { render_mode = "rgbd_tuple", camera_name = "track" });
            using ResetResult<NDArray> start = env.Reset(seed: 42);
            using var frames = (RenderResult.RgbDepth)env.Render();
            Console.WriteLine($"{frames.RgbFrame.Shape} {frames.DepthFrame.Shape}");   // (480, 480, 3) (480, 480)
            // MuJoCo's raw depth buffer, 0 at the near plane and 1 at the far one,
            // not metres: the nearest point, the floor at the bottom edge, is 0.99
            using NDArray closest = frames.DepthFrame.min();
            nearest = closest.GetSingle().ToString("F2", CultureInfo.InvariantCulture);
            Console.WriteLine(nearest);   // 0.99
            #endregion
            // The printed line is the page's comment, as Shape prints itself.
            Assert.AreEqual("(480, 480, 3) (480, 480)", $"{frames.RgbFrame.Shape} {frames.DepthFrame.Shape}");
            Assert.AreEqual("0.99", nearest);
        }

        /// <summary>
        ///     Many humanoids at once: Humanoid has no vector env of its own, so <c>MakeVec</c> steps separate envs in turn
        ///     (<c>sync</c>, its default here), each with its own simulation and its own resets.
        /// </summary>
        [TestMethod]
        public void ManyHumanoids() {
            #region ManyHumanoids
            // No vector env of its own: MakeVec steps 8 envs in turn ("sync")
            using VectorEnv<NDArray, NDArray> envs
                = Gymnasium.MakeVec<NDArray, NDArray>("Humanoid-v5", numEnvs: 8);
            using VectorResetResult<NDArray> start
                = envs.Reset(seed: 42);   // 8 x 348 observations
            (NDArray observations, Info infos) = start;
            for (int t = 0; t < 100; t++) {
                using NDArray actions = envs.ActionSpace.Sample();   // 8 x 17 torques
                using VectorStepResult<NDArray> step
                    = envs.Step(actions);   // fallen humanoids reset themselves
            }
            #endregion
            Assert.AreEqual(8, envs.NumEnvs);
            Assert.IsInstanceOfType(envs, typeof(SyncVectorEnv<NDArray, NDArray>));
            CollectionAssert.AreEqual(new long[] { 8, 348 }, observations.shape.Select(d => (long)d).ToArray());
        }

        /// <summary>
        ///     The Arguments section's calls: kwargs by Gymnasium's names, or the generated options record by C# names; both
        ///     build the same env, here observing neither the bodies' inertias nor their velocities: 140 numbers.
        /// </summary>
        [TestMethod]
        public void Arguments() {
            #region Arguments
            using IEnv byName = Gymnasium.Make("Humanoid-v5",
                new { include_cinert_in_observation = false, include_cvel_in_observation = false });
            using IEnv byRecord = Gymnasium.Make("Humanoid-v5",
                new HumanoidEnvV5Options { IncludeCinertInObservation = false, IncludeCvelInObservation = false });
            #endregion
            foreach (IEnv env in new[] { byName, byRecord }) {
                var humanoid = (HumanoidEnvV5)env.Unwrapped;
                CollectionAssert.AreEqual(new long[] { 140 }, humanoid.ObservationSpace.Shape!.Value.dimensions.Select(d => (long)d).ToArray());
            }
        }
    }
}
