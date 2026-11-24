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
    ///     The C# samples of Ant's page on the docs site (<c>docs/environments/mujoco/ant.md</c>), each the body of a test,
    ///     so the unit gate compiles and runs every line the page shows (DR-105). The page includes them by DocFX's
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
    public class AntSamples {
        /// <summary>The page's quick start: make the env, reset it with a seed, take one random step.</summary>
        [TestMethod]
        public void QuickStart() {
            #region QuickStart
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("Ant-v5");
            // 105 numbers: the torso's height and orientation, 8 joint angles, 14 velocities, 78 contact forces
            NDArray observation = env.Reset(seed: 42).Observation;
            // eight torques, each from -1 to 1: a hip and an ankle per leg
            using NDArray action = env.ActionSpace.Sample();
            var (next, reward, terminated, truncated, info) = env.Step(action);
            #endregion
            Assert.AreEqual(105, observation.size);
            // Gymnasium's reset of seed 42: the torso at 0.8217 m, the model's 0.75 plus the reset's uniform noise.
            Assert.AreEqual(0.8217195839822765, observation.GetDouble(0));
            _ = (reward, terminated, truncated, info);
            next.Dispose();
            observation.Dispose();
        }

        /// <summary>
        ///     Run an episode: random torques until the torso leaves its healthy heights or the 1,000-step limit; random
        ///     torques mostly end it by throwing the torso above 1 m.
        /// </summary>
        [TestMethod]
        public void RunAnEpisode() {
            int steps = 0;
            #region RunAnEpisode
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("Ant-v5");
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
            // random torques mostly throw the torso above 1 m: 193 steps, -76.5 on average
            Console.WriteLine($"{steps} steps, return {episodeReturn:F1}");
            #endregion
            Assert.IsTrue(steps <= 1000);
        }

        /// <summary>
        ///     Read the reward's terms: every motor at half torque for ten steps from seed 42; by the tenth the ant has dropped
        ///     onto its feet, so the contact cost is no longer 0.
        /// </summary>
        [TestMethod]
        public void RewardTerms() {
            double reward = 0, forward = 0, healthy = 0, control = 0, contact = 0;
            #region RewardTerms
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("Ant-v5");
            using var start = env.Reset(seed: 42);
            using NDArray action = np.full(8, 0.5f);   // every motor at half torque
            for (int t = 1; t < 10; t++) {
                using var landing = env.Step(action);   // the ant drops onto its feet
            }
            using var step = env.Step(action);
            (reward, forward, healthy, control, contact) = (step.Reward, step.Info.Get<double>("reward_forward"),
                step.Info.Get<double>("reward_survive"), step.Info.Get<double>("reward_ctrl"),
                step.Info.Get<double>("reward_contact"));
            // the tenth step: 0.0051 = 0.0083 forward + 1.0000 healthy - 1.0000 control - 0.0031 contact
            Console.WriteLine($"{reward:F4} = {forward:F4} forward + {healthy:F4} healthy {control:F4} control {contact:F4} contact");
            #endregion
            Assert.AreEqual(0.005142415105552134, reward);
            Assert.AreEqual(0.008266182387716148, forward);
            Assert.AreEqual(1.0, healthy);
            Assert.AreEqual(-1.0, control);
            Assert.AreEqual(-0.003123767282163981, contact);
        }

        /// <summary>
        ///     Start without noise: <c>reset_noise_scale</c> 0 resets every seed to the model's pose, the torso 0.75 m up and
        ///     unrotated, the legs straight out, at rest.
        /// </summary>
        [TestMethod]
        public void ResetWithoutNoise() {
            double height = 0, w = 0;
            bool restZero = false;
            #region ResetWithoutNoise
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Ant-v5",
                new AntEnvV5Options { ResetNoiseScale = 0.0 });
            using var start = env.Reset(seed: 42);
            double[] o = start.Observation.ToArray<double>();
            (height, w, restZero) = (o[0], o[1], o.Skip(2).All(v => v == 0));
            // 0.75, 1.00, True: the torso 0.75 m up, unrotated (its quaternion 1, 0, 0, 0), every other value 0
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"{height:F2}, {w:F2}, {restZero}"));
            #endregion
            Assert.AreEqual(0.75, height);
            Assert.AreEqual(1.0, w);
            Assert.IsTrue(restZero);
        }

        /// <summary>
        ///     Keep going outside the healthy heights: <c>terminate_when_unhealthy</c> false runs the whole time limit and pays
        ///     the healthy bonus only on healthy steps; a seeded action space makes the run repeatable.
        /// </summary>
        [TestMethod]
        public void KeepGoing() {
            int steps = 0, healthy = 0;
            double episodeReturn = 0;
            #region KeepGoing
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Ant-v5",
                new { terminate_when_unhealthy = false });
            var ant = (AntEnvV5)env.Unwrapped;
            using var start = env.Reset(seed: 0);
            env.ActionSpace.Seed(0);
            while (true) {
                using NDArray action = env.ActionSpace.Sample();
                using var step = env.Step(action);
                (steps, episodeReturn) = (steps + 1, episodeReturn + step.Reward);
                healthy += ant.IsHealthy ? 1 : 0;
                if (step.Terminated || step.Truncated)
                    break;
            }
            // 1000 steps, healthy for 988, return -342.0: the torso first above 1 m at step 45
            Console.WriteLine($"{steps} steps, healthy for {healthy}, return {episodeReturn:F1}");
            #endregion
            Assert.AreEqual(1000, steps);
            Assert.AreEqual(988, healthy);
            Assert.AreEqual(-341.9500668809291, episodeReturn);
        }

        /// <summary>
        ///     Cameras and depth: <c>rgbd_tuple</c> through the model's <c>track</c> camera, the colour frame and the depth of
        ///     the same view; the depth is the raw depth buffer, whose smallest value here is on the leg nearest the camera.
        /// </summary>
        [TestMethod]
        public void CamerasAndDepth() {
            string nearest = "";
            #region CamerasAndDepth
            // the model's "track" camera follows the ant from the side; rgbd_tuple draws colour and depth together
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Ant-v5",
                new { render_mode = "rgbd_tuple", camera_name = "track" });
            using var start = env.Reset(seed: 42);
            using var frames = (RenderResult.RgbDepth)env.Render();
            Console.WriteLine($"{frames.RgbFrame.Shape} {frames.DepthFrame.Shape}");   // (480, 480, 3) (480, 480)
            // MuJoCo's raw depth buffer, 0 at the near plane and 1 at the far one,
            // not metres: the nearest point, on the near left leg, is 0.96
            using NDArray closest = frames.DepthFrame.min();
            nearest = closest.GetSingle().ToString("F2", CultureInfo.InvariantCulture);
            Console.WriteLine(nearest);   // 0.96
            #endregion
            // The printed line is the page's comment, as Shape prints itself.
            Assert.AreEqual("(480, 480, 3) (480, 480)", $"{frames.RgbFrame.Shape} {frames.DepthFrame.Shape}");
            Assert.AreEqual("0.96", nearest);
        }

        /// <summary>
        ///     Many ants at once: Ant has no vector env of its own, so <c>MakeVec</c> steps separate envs in turn (<c>sync</c>,
        ///     its default here), each with its own simulation and its own resets.
        /// </summary>
        [TestMethod]
        public void ManyAnts() {
            #region ManyAnts
            // No vector env of its own: MakeVec steps 8 envs in turn ("sync")
            using VectorEnv<NDArray, NDArray> envs =
                Gymnasium.MakeVec<NDArray, NDArray>("Ant-v5", numEnvs: 8);
            using var start = envs.Reset(seed: 42);   // 8 x 105 observations
            for (int t = 0; t < 100; t++) {
                using NDArray actions = envs.ActionSpace.Sample();   // 8 x 8 torques
                using var step = envs.Step(actions);   // ants out of the healthy heights reset themselves
            }
            #endregion
            Assert.AreEqual(8, envs.NumEnvs);
            Assert.IsInstanceOfType(envs, typeof(SyncVectorEnv<NDArray, NDArray>));
            CollectionAssert.AreEqual(new long[] { 8, 105 }, start.Observations.shape.Select(d => (long)d).ToArray());
        }

        /// <summary>
        ///     The Arguments section's calls: kwargs by Gymnasium's names, or the generated options record by C# names; both
        ///     build the same env, here without the contact cost and with a healthy band up to 1.5 m.
        /// </summary>
        [TestMethod]
        public void Arguments() {
            #region Arguments
            using IEnv byName = Gymnasium.Make("Ant-v5",
                new { contact_cost_weight = 0.0, healthy_z_range = (0.2, 1.5) });
            using IEnv byRecord = Gymnasium.Make("Ant-v5",
                new AntEnvV5Options { ContactCostWeight = 0.0, HealthyZRange = (0.2, 1.5) });
            #endregion
            foreach (IEnv env in new[] { byName, byRecord }) {
                var ant = (AntEnvV5)env.Unwrapped;
                Assert.AreEqual(0.0, ant.ContactCostWeight);
                Assert.AreEqual((0.2, 1.5), ant.HealthyZRange);
            }
        }
    }
}
