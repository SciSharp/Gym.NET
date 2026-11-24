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
    ///     The C# samples of Swimmer's page on the docs site (<c>docs/environments/mujoco/swimmer.md</c>), each the body of a
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
    ///     same seeds and actions (the physics is bit for bit), so the page cannot drift from what either computes. The
    ///     depth sample's number is rounded to two decimals: Gym.NET draws MuJoCo's frames itself, and its depth values
    ///     differ from Gymnasium's in their last bits (DR-98).
    /// </remarks>
    [TestClass]
    public class SwimmerSamples {
        /// <summary>The page's quick start: make the env, reset it with a seed, take one random step.</summary>
        [TestMethod]
        public void QuickStart() {
            #region QuickStart
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("Swimmer-v5");
            // 8 numbers: the heading, the 2 rotors' angles, 5 velocities
            NDArray observation = env.Reset(seed: 42).Observation;
            // two torques, each from -1 to 1: the front rotor, the back rotor
            using NDArray action = env.ActionSpace.Sample();
            var (next, reward, terminated, truncated, info) = env.Step(action);
            #endregion
            Assert.AreEqual(8, observation.size);
            // Gymnasium's reset of seed 42: the heading, 0.0717 rad, the reset's uniform noise of 0.1.
            Assert.AreEqual(0.07171958398227649, observation.GetDouble(0));
            Assert.IsFalse(terminated);
            _ = (reward, truncated, info);
            next.Dispose();
            observation.Dispose();
        }

        /// <summary>Run an episode: random torques for the 1,000 steps of the time limit (Swimmer never terminates).</summary>
        [TestMethod]
        public void RunAnEpisode() {
            int steps = 0;
            #region RunAnEpisode
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("Swimmer-v5");
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
            Console.WriteLine($"return {episodeReturn:F1}");   // random torques: 9.4 on average
            #endregion
            Assert.AreEqual(1000, steps);
        }

        /// <summary>
        ///     Read the reward's terms: one step with both rotors at half torque, the reward and the info that splits it, from
        ///     seed 42.
        /// </summary>
        [TestMethod]
        public void RewardTerms() {
            double reward = 0, forward = 0, control = 0;
            #region RewardTerms
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("Swimmer-v5");
            using var start = env.Reset(seed: 42);
            using NDArray action = np.full(2, 0.5f);   // both rotors at half torque
            using var step = env.Step(action);
            (reward, forward, control) = (step.Reward, step.Info.Get<double>("reward_forward"),
                step.Info.Get<double>("reward_ctrl"));
            // 0.08428 = 0.08433 forward (the speed in m/s) - 0.00005 (0.0001 x the torques squared)
            Console.WriteLine($"{reward:F5} = {forward:F5} forward {control:F5} control");
            #endregion
            Assert.AreEqual(0.08427849322192656, reward);
            Assert.AreEqual(0.08432849322066345, forward);
            Assert.AreEqual(-4.999999873689376e-05, control);
        }

        /// <summary>Start without noise: <c>reset_noise_scale</c> 0 resets every seed to the model's pose, at rest.</summary>
        [TestMethod]
        public void ResetWithoutNoise() {
            bool allZero = false;
            #region ResetWithoutNoise
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Swimmer-v5",
                new SwimmerEnvV5Options { ResetNoiseScale = 0.0 });
            using var start = env.Reset(seed: 42);
            // the links in a straight line, at rest: every observed value 0
            allZero = start.Observation.ToArray<double>().All(v => v == 0);
            Console.WriteLine(allZero);   // True
            #endregion
            Assert.IsTrue(allZero);
        }

        /// <summary>
        ///     Observe the position: <c>exclude_current_positions_from_observation</c> false puts the root's x and y first (the
        ///     point the sliders move, the torso body's origin), ten numbers instead of eight.
        /// </summary>
        [TestMethod]
        public void ObserveThePosition() {
            double x = 0, y = 0;
            long size = 0;
            #region ObserveThePosition
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Swimmer-v5",
                new { exclude_current_positions_from_observation = false });
            using var start = env.Reset(seed: 42);
            (size, x, y) = (start.Observation.size, start.Observation.GetDouble(0), start.Observation.GetDouble(1));
            // 10 numbers, x 0.0548 and y -0.0122 first: the root's position
            Console.WriteLine($"{size} numbers, x {x:F4} and y {y:F4} first");
            #endregion
            Assert.AreEqual(10, size);
            Assert.AreEqual(0.05479120971119267, x);
            Assert.AreEqual(-0.012224312049589542, y);
        }

        /// <summary>
        ///     Cameras and depth: <c>rgbd_tuple</c> through the model's <c>track</c> camera, the colour frame and the depth of
        ///     the same view; the depth is the raw depth buffer, whose smallest value here is the floor's at the bottom edge.
        /// </summary>
        [TestMethod]
        public void CamerasAndDepth() {
            string nearest = "";
            #region CamerasAndDepth
            // the model's "track" camera follows the swimmer from above; rgbd_tuple draws colour and depth together
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Swimmer-v5",
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
            // The printed line is the page's comment, as Shape prints itself.
            Assert.AreEqual("(480, 480, 3) (480, 480)", $"{frames.RgbFrame.Shape} {frames.DepthFrame.Shape}");
            Assert.AreEqual("0.97", nearest);
        }

        /// <summary>
        ///     Many swimmers at once: Swimmer has no vector env of its own, so <c>MakeVec</c> steps separate envs in turn
        ///     (<c>sync</c>, its default here), each with its own simulation.
        /// </summary>
        [TestMethod]
        public void ManySwimmers() {
            #region ManySwimmers
            // No vector env of its own: MakeVec steps 8 envs in turn ("sync")
            using VectorEnv<NDArray, NDArray> envs =
                Gymnasium.MakeVec<NDArray, NDArray>("Swimmer-v5", numEnvs: 8);
            using var start = envs.Reset(seed: 42);   // 8 x 8 observations
            for (int t = 0; t < 100; t++) {
                using NDArray actions = envs.ActionSpace.Sample();   // 8 x 2 torques
                using var step = envs.Step(actions);
            }
            #endregion
            Assert.AreEqual(8, envs.NumEnvs);
            Assert.IsInstanceOfType(envs, typeof(SyncVectorEnv<NDArray, NDArray>));
            CollectionAssert.AreEqual(new long[] { 8, 8 }, start.Observations.shape.Select(d => (long)d).ToArray());
        }

        /// <summary>
        ///     The Arguments section's calls: kwargs by Gymnasium's names, or the generated options record by C# names; both
        ///     build the same env, here with a ten times dearer control cost and twice the starting noise.
        /// </summary>
        [TestMethod]
        public void Arguments() {
            #region Arguments
            using IEnv byName = Gymnasium.Make("Swimmer-v5",
                new { ctrl_cost_weight = 1e-3, reset_noise_scale = 0.2 });
            using IEnv byRecord = Gymnasium.Make("Swimmer-v5",
                new SwimmerEnvV5Options { CtrlCostWeight = 1e-3, ResetNoiseScale = 0.2 });
            #endregion
            foreach (IEnv env in new[] { byName, byRecord }) {
                var swimmer = (SwimmerEnvV5)env.Unwrapped;
                Assert.AreEqual(1e-3, swimmer.CtrlCostWeight);
                Assert.AreEqual(0.2, swimmer.ResetNoiseScale);
            }
        }
    }
}
