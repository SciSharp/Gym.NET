using Gym;
using Gym.Envs;
using NumSharp;

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Gym.Environments.Envs.MuJoCo;
using Gym.Exceptions;
using Gym.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;

// Not under Gym.Tests: there the namespace Gym.Tests.Gymnasium (DR-06) shadows the Gymnasium facade, and the samples
// must read Gymnasium.Make(...) exactly as a reader's code does.
namespace Gym.Docs.EnvPageSamples {
    /// <summary>
    ///     The C# samples of Pusher's page on the docs site (<c>docs/environments/mujoco/pusher.md</c>), each the body of a
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
    ///     depth sample's number is rounded to three decimals: Gym.NET draws MuJoCo's frames itself, and its depth values
    ///     differ from Gymnasium's in their last bits (DR-98).
    /// </remarks>
    [TestClass]
    public class PusherSamples {
        /// <summary>The page's quick start: make the env, reset it with a seed, take one random step.</summary>
        [TestMethod]
        public void QuickStart() {
            #region QuickStart
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("Pusher-v5");
            // 23 numbers: 7 joint angles, 7 speeds, then the fingertip's, the puck's and the goal's positions
            NDArray observation = env.Reset(seed: 42).Observation;
            // seven torques, each from -2 to 2: shoulder to wrist
            using NDArray action = env.ActionSpace.Sample();
            var (next, reward, terminated, truncated, info) = env.Step(action);
            #endregion
            Assert.AreEqual(23, observation.size);
            // Gymnasium's reset of seed 42: the puck at x 0.6402, its model place 0.45 plus the seed's offset.
            Assert.AreEqual(0.6402489406547024, observation.GetDouble(17));
            Assert.IsFalse(terminated);
            _ = (reward, truncated, info);
            next.Dispose();
            observation.Dispose();
        }

        /// <summary>Run an episode: random torques for the 100 steps of the time limit (Pusher never terminates).</summary>
        [TestMethod]
        public void RunAnEpisode() {
            int steps = 0;
            #region RunAnEpisode
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("Pusher-v5");
            NDArray observation = env.Reset(seed: 42).Observation;
            double episodeReturn = 0;
            while (true) {
                using NDArray action = env.ActionSpace.Sample();
                var (next, reward, terminated, truncated, _) = env.Step(action);
                observation.Dispose();
                observation = next;
                episodeReturn += reward;
                steps++;
                if (terminated || truncated)   // never terminated: 100 steps
                    break;
            }
            Console.WriteLine($"return {episodeReturn:F1}");   // random torques: -146.5 on average
            #endregion
            Assert.AreEqual(100, steps);
        }

        /// <summary>Read the reward's terms: one step with every motor at half torque from seed 42, the reward and its three terms.</summary>
        [TestMethod]
        public void RewardTerms() {
            double reward = 0, distance = 0, near = 0, control = 0;
            #region RewardTerms
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("Pusher-v5");
            using var start = env.Reset(seed: 42);
            using NDArray action = np.full(7, 1.0f);   // every motor at half torque
            using var step = env.Step(action);
            (reward, distance, near, control) = (step.Reward, step.Info.Get<double>("reward_dist"),
                step.Info.Get<double>("reward_near"), step.Info.Get<double>("reward_ctrl"));
            // -1.2499 = -0.3352 distance - 0.2147 near - 0.7000 control
            Console.WriteLine($"{reward:F4} = {distance:F4} distance {near:F4} near {control:F4} control");
            #endregion
            Assert.AreEqual(-1.2499215786139528, reward);
            Assert.AreEqual(-0.33517902732334426, distance);
            Assert.AreEqual(-0.2147425632115375, near);
            Assert.AreEqual(-0.699999988079071, control);
        }

        /// <summary>
        ///     Where the puck starts: the seed places it anywhere from 0.17 to 0.36 m of the goal, read here from the
        ///     observation's positions.
        /// </summary>
        [TestMethod]
        public void WhereThePuckStarts() {
            string gap = "";
            #region WhereThePuckStarts
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Pusher-v5");
            using var start = env.Reset(seed: 42);
            double[] o = start.Observation.ToArray<double>();
            // the puck's x and y are items 17 and 18, the goal's 20 and 21
            double distance = Math.Sqrt(Math.Pow(o[17] - o[20], 2) + Math.Pow(o[18] - o[21], 2));
            gap = distance.ToString("F3", CultureInfo.InvariantCulture);
            Console.WriteLine($"{gap} m");   // 0.332 m: every seed puts it between 0.17 and 0.36 m away
            #endregion
            Assert.AreEqual("0.332", gap);
        }

        /// <summary>
        ///     Pusher-v4 refuses to be made, as Gymnasium's does on MuJoCo 3 and later: Gym.NET's engine is MuJoCo 4.0.0's,
        ///     so the refusal is Gymnasium's own error and message.
        /// </summary>
        [TestMethod]
        public void PusherV4Refuses() {
            string message = "";
            #region PusherV4Refuses
            try {
                using IEnv env = Gymnasium.Make("Pusher-v4");
            } catch (ImportError error) {
                message = error.Message;
                // `Pusher-v4` is only supported on `mujoco<3`, for more information ...
                Console.WriteLine(message);
            }
            #endregion
            Assert.AreEqual("`Pusher-v4` is only supported on `mujoco<3`, for more information https://github.com/Farama-Foundation/Gymnasium/issues/950", message);
        }

        /// <summary>
        ///     Cameras and depth: Pusher's model has no camera of its own, so the sample places the free camera as the video
        ///     does, 2.4 m off at azimuth 135°, and draws colour and depth together; the depth is the raw depth buffer.
        /// </summary>
        [TestMethod]
        public void CamerasAndDepth() {
            string nearest = "";
            #region CamerasAndDepth
            // no camera in the model: place the free camera as the video does; rgbd_tuple draws colour and depth together
            var view = new Dictionary<string, object?> { ["trackbodyid"] = -1L, ["distance"] = 2.4, ["azimuth"] = 135.0 };
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Pusher-v5",
                new Dictionary<string, object?> { ["render_mode"] = "rgbd_tuple", ["default_camera_config"] = view });
            using var start = env.Reset(seed: 42);
            using var frames = (RenderResult.RgbDepth)env.Render();
            Console.WriteLine($"{frames.RgbFrame.Shape} {frames.DepthFrame.Shape}");   // (480, 480, 3) (480, 480)
            // MuJoCo's raw depth buffer, 0 at the near plane and 1 at the far one, not metres
            using NDArray closest = frames.DepthFrame.min();
            nearest = closest.GetSingle().ToString("F3", CultureInfo.InvariantCulture);
            Console.WriteLine(nearest);   // 0.993
            #endregion
            // The printed line is the page's comment, as Shape prints itself.
            Assert.AreEqual("(480, 480, 3) (480, 480)", $"{frames.RgbFrame.Shape} {frames.DepthFrame.Shape}");
            Assert.AreEqual("0.993", nearest);
        }

        /// <summary>
        ///     Many arms at once: Pusher has no vector env of its own, so <c>MakeVec</c> steps separate envs in turn
        ///     (<c>sync</c>, its default here), each with its own simulation.
        /// </summary>
        [TestMethod]
        public void ManyArms() {
            #region ManyArms
            // No vector env of its own: MakeVec steps 8 envs in turn ("sync")
            using VectorEnv<NDArray, NDArray> envs =
                Gymnasium.MakeVec<NDArray, NDArray>("Pusher-v5", numEnvs: 8);
            using var start = envs.Reset(seed: 42);   // 8 x 23 observations
            for (int t = 0; t < 100; t++) {
                using NDArray actions = envs.ActionSpace.Sample();   // 8 x 7 torques
                using var step = envs.Step(actions);   // after 100 steps each arm resets itself
            }
            #endregion
            Assert.AreEqual(8, envs.NumEnvs);
            Assert.IsInstanceOfType(envs, typeof(SyncVectorEnv<NDArray, NDArray>));
            CollectionAssert.AreEqual(new long[] { 8, 23 }, start.Observations.shape.Select(d => (long)d).ToArray());
        }

        /// <summary>
        ///     The Arguments section's calls: kwargs by Gymnasium's names, or the generated options record by C# names; both
        ///     build the same env, here with the puck's distance weighed twice and a tenth of the control cost.
        /// </summary>
        [TestMethod]
        public void Arguments() {
            #region Arguments
            using IEnv byName = Gymnasium.Make("Pusher-v5",
                new { reward_dist_weight = 2.0, reward_control_weight = 0.01 });
            using IEnv byRecord = Gymnasium.Make("Pusher-v5",
                new PusherEnvV5Options { RewardDistWeight = 2.0, RewardControlWeight = 0.01 });
            #endregion
            foreach (IEnv env in new[] { byName, byRecord }) {
                var pusher = (PusherEnvV5)env.Unwrapped;
                Assert.AreEqual(2.0, pusher.RewardDistWeight);
                Assert.AreEqual(0.01, pusher.RewardControlWeight);
            }
        }
    }
}
