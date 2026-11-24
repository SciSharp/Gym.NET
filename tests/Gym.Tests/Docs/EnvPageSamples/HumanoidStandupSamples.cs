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
    ///     The C# samples of HumanoidStandup's page on the docs site (<c>docs/environments/mujoco/humanoid_standup.md</c>),
    ///     each the body of a test, so the unit gate compiles and runs every line the page shows (DR-105). The page includes
    ///     them by DocFX's code-snippet syntax: a region by name, or for the quick start this file's first four lines (its
    ///     usings and a blank) and the quick start's region, by line range.
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
    public class HumanoidStandupSamples {
        /// <summary>The page's quick start: make the env, reset it with a seed, take one random step.</summary>
        [TestMethod]
        public void QuickStart() {
            #region QuickStart
            using Env<NDArray, NDArray> env
                = Gymnasium.Make<NDArray, NDArray>("HumanoidStandup-v5");
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
            // Gymnasium's reset of seed 42: the torso at 0.1122 m, the model's 0.105 plus the reset's uniform noise.
            Assert.AreEqual(0.11217195839822765, first.GetDouble(0));
            Assert.IsFalse(terminated);
            _ = (reward, truncated, info);
            observation.Dispose();
        }

        /// <summary>Run an episode: random torques for the 1,000 steps of the time limit (HumanoidStandup never terminates).</summary>
        [TestMethod]
        public void RunAnEpisode() {
            int steps = 0;
            #region RunAnEpisode
            using Env<NDArray, NDArray> env
                = Gymnasium.Make<NDArray, NDArray>("HumanoidStandup-v5");
            (NDArray observation, Info info) = env.Reset(seed: 42);
            double episodeReturn = 0;
            while (true) {
                using NDArray action = env.ActionSpace.Sample();
                observation.Dispose();
                (observation, double reward, bool terminated, bool truncated, info)
                    = env.Step(action);
                episodeReturn += reward;
                steps++;
                if (terminated || truncated)   // never terminated: 1,000 steps
                    break;
            }
            Console.WriteLine($"return {episodeReturn:F1}");   // random torques: 32,836.4 on average
            #endregion
            Assert.AreEqual(1000, steps);
        }

        /// <summary>
        ///     Read the reward's terms: one step with every motor at half torque from seed 42, the reward and the info that
        ///     splits it; the constant 1 a step is in the reward only.
        /// </summary>
        [TestMethod]
        public void RewardTerms() {
            double reward = 0, up = 0, control = 0, impact = 0;
            #region RewardTerms
            using Env<NDArray, NDArray> env
                = Gymnasium.Make<NDArray, NDArray>("HumanoidStandup-v5");
            using ResetResult<NDArray> start = env.Reset(seed: 42);
            using NDArray action = np.full(17, 0.2f);   // every motor at half torque
            using StepResult<NDArray> step = env.Step(action);
            (reward, up, control, impact) = (step.Reward, step.Info.Get<double>("reward_linup"),
                step.Info.Get<double>("reward_quadctrl"), step.Info.Get<double>("reward_impact"));
            // 38.9265 = 38.0847 up (the height over 0.003 s) + 1 - 0.0680 control - 0.0902 impact
            Console.WriteLine($"{reward:F4} = {up:F4} up + 1 {control:F4} control {impact:F4} impact");
            #endregion
            Assert.AreEqual(38.92653391465571, reward);
            Assert.AreEqual(38.08474734344501, up);
            Assert.AreEqual(-0.06800000202655794, control);
            Assert.AreEqual(-0.09021342676274735, impact);
        }

        /// <summary>
        ///     Start without noise: <c>reset_noise_scale</c> 0 resets every seed to the model's pose, lying flat with the
        ///     torso 0.105 m up and unrotated, every joint at 0 and at rest.
        /// </summary>
        [TestMethod]
        public void ResetWithoutNoise() {
            double height = 0, w = 0;
            bool still = false;
            #region ResetWithoutNoise
            using Env<NDArray, NDArray> env
                = Gymnasium.Make<NDArray, NDArray>("HumanoidStandup-v5",
                    new HumanoidStandupEnvV5Options { ResetNoiseScale = 0.0 });
            using ResetResult<NDArray> start = env.Reset(seed: 42);
            (NDArray observation, Info info) = start;
            double[] o = observation.ToArray<double>();
            (height, w, still) = (o[0], o[1], o.Skip(2).Take(43).All(v => v == 0));
            // 0.105, 1.00, True: lying, the torso 0.105 m up and unrotated, every joint at 0 and at rest
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"{height:F3}, {w:F2}, {still}"));
            #endregion
            Assert.AreEqual(0.105, height);
            Assert.AreEqual(1.0, w);
            Assert.IsTrue(still);
        }

        /// <summary>
        ///     Lie still: zero torques for the whole episode still earn the height term every step, about 29.4 once the
        ///     body settles, less than random torques earn.
        /// </summary>
        [TestMethod]
        public void LieStill() {
            int steps = 0;
            double episodeReturn = 0;
            #region LieStill
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("HumanoidStandup-v5");
            using ResetResult<NDArray> start = env.Reset(seed: 0);
            using NDArray rest = np.zeros(new Shape(17), NPTypeCode.Single);   // no torque at all
            while (true) {
                using StepResult<NDArray> step = env.Step(rest);
                (steps, episodeReturn) = (steps + 1, episodeReturn + step.Reward);
                if (step.Terminated || step.Truncated)
                    break;
            }
            // 1000 steps, return 29802.0: lying still pays about 29.4 a step
            Console.WriteLine($"{steps} steps, return {episodeReturn:F1}");
            #endregion
            Assert.AreEqual(1000, steps);
            Assert.AreEqual(29801.9519051827, episodeReturn);
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
            using Env<NDArray, NDArray> env
                = Gymnasium.Make<NDArray, NDArray>("HumanoidStandup-v5",
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
        ///     Many humanoids at once: HumanoidStandup has no vector env of its own, so <c>MakeVec</c> steps separate envs in
        ///     turn (<c>sync</c>, its default here), each with its own simulation.
        /// </summary>
        [TestMethod]
        public void ManyHumanoids() {
            #region ManyHumanoids
            // No vector env of its own: MakeVec steps 8 envs in turn ("sync")
            using VectorEnv<NDArray, NDArray> envs
                = Gymnasium.MakeVec<NDArray, NDArray>("HumanoidStandup-v5", numEnvs: 8);
            using VectorResetResult<NDArray> start
                = envs.Reset(seed: 42);   // 8 x 348 observations
            (NDArray observations, Info infos) = start;
            for (int t = 0; t < 100; t++) {
                using NDArray actions = envs.ActionSpace.Sample();   // 8 x 17 torques
                using VectorStepResult<NDArray> step
                    = envs.Step(actions);   // no episode ends early
            }
            #endregion
            Assert.AreEqual(8, envs.NumEnvs);
            Assert.IsInstanceOfType(envs, typeof(SyncVectorEnv<NDArray, NDArray>));
            CollectionAssert.AreEqual(new long[] { 8, 348 }, observations.shape.Select(d => (long)d).ToArray());
        }

        /// <summary>
        ///     The Arguments section's calls: kwargs by Gymnasium's names, or the generated options record by C# names; both
        ///     build the same env, here the one the expert was trained on, which observes positions and velocities only:
        ///     45 numbers.
        /// </summary>
        [TestMethod]
        public void Arguments() {
            #region Arguments
            using IEnv byName = Gymnasium.Make("HumanoidStandup-v5", new {
                include_cinert_in_observation = false, include_cvel_in_observation = false,
                include_qfrc_actuator_in_observation = false, include_cfrc_ext_in_observation = false,
            });
            using IEnv byRecord
                = Gymnasium.Make("HumanoidStandup-v5", new HumanoidStandupEnvV5Options {
                    IncludeCinertInObservation = false, IncludeCvelInObservation = false,
                    IncludeQfrcActuatorInObservation = false, IncludeCfrcExtInObservation = false,
                });
            #endregion
            foreach (IEnv env in new[] { byName, byRecord }) {
                var humanoid = (HumanoidStandupEnvV5)env.Unwrapped;
                CollectionAssert.AreEqual(new long[] { 45 }, humanoid.ObservationSpace.Shape!.Value.dimensions.Select(d => (long)d).ToArray());
            }
        }
    }
}
