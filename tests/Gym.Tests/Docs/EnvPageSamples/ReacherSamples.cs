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
    ///     The C# samples of Reacher's page on the docs site (<c>docs/environments/mujoco/reacher.md</c>), each the body of a
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
    public class ReacherSamples {
        /// <summary>The page's quick start: make the env, reset it with a seed, take one random step.</summary>
        [TestMethod]
        public void QuickStart() {
            #region QuickStart
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Reacher-v5");
            // 10 numbers: the joints' cosines and sines, the target, 2 speeds, the fingertip's offset
            (NDArray observation, Info info) = env.Reset(seed: 42);
            // two torques, each from -1 to 1: shoulder and elbow
            using NDArray action = env.ActionSpace.Sample();
            observation.Dispose();
            (observation, double reward, bool terminated, bool truncated, info)
                = env.Step(action);
            #endregion
            // The quick start's first observation, read again: the same seeded reset gives the same one.
            using ResetResult<NDArray> again = env.Reset(seed: 42);
            NDArray first = again.Observation;
            Assert.AreEqual(10, first.size);
            // Gymnasium's reset of seed 42: the shoulder's cosine, the reset's uniform turn of at most 0.1 rad.
            Assert.AreEqual(0.9984993371509613, first.GetDouble(0));
            Assert.IsFalse(terminated);
            _ = (reward, truncated, info);
            observation.Dispose();
        }

        /// <summary>Run an episode: random torques for the 50 steps of the time limit (Reacher never terminates).</summary>
        [TestMethod]
        public void RunAnEpisode() {
            int steps = 0;
            #region RunAnEpisode
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Reacher-v5");
            (NDArray observation, Info info) = env.Reset(seed: 42);
            double episodeReturn = 0;
            while (true) {
                using NDArray action = env.ActionSpace.Sample();
                observation.Dispose();
                (observation, double reward, bool terminated, bool truncated, info)
                    = env.Step(action);
                episodeReturn += reward;
                steps++;
                if (terminated || truncated)   // never terminated: 50 steps
                    break;
            }
            Console.WriteLine($"return {episodeReturn:F1}");   // random torques: -43.8 on average
            #endregion
            Assert.AreEqual(50, steps);
        }

        /// <summary>Read the reward's terms: one step with both motors at half torque from seed 42, the reward and its two terms.</summary>
        [TestMethod]
        public void RewardTerms() {
            double reward = 0, distance = 0, control = 0;
            #region RewardTerms
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Reacher-v5");
            using ResetResult<NDArray> start = env.Reset(seed: 42);
            using NDArray action = np.full(2, 0.5f);   // both motors at half torque
            using StepResult<NDArray> step = env.Step(action);
            (reward, distance, control) = (step.Reward, step.Info.Get<double>("reward_dist"), step.Info.Get<double>("reward_ctrl"));
            // -0.6435 = -0.1435 distance - 0.5000 control
            Console.WriteLine($"{reward:F4} = {distance:F4} distance {control:F4} control");
            #endregion
            Assert.AreEqual(-0.6435178197483826, reward);
            Assert.AreEqual(-0.14351781974838265, distance);
            Assert.AreEqual(-0.5, control);
        }

        /// <summary>
        ///     Read the distance from the observation: its last two items are the fingertip's offset from the target, whose
        ///     length is the reward's distance term, to the last bit.
        /// </summary>
        [TestMethod]
        public void DistanceFromTheObservation() {
            double fromObservation = 0, fromReward = 0;
            #region DistanceFromTheObservation
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Reacher-v5");
            using ResetResult<NDArray> start = env.Reset(seed: 42);
            using NDArray action = np.full(2, 0.5f);
            using StepResult<NDArray> step = env.Step(action);
            double[] o = step.Observation.ToArray<double>();
            // items 8 and 9: the fingertip's offset from the target, in x and y
            fromObservation = Math.Sqrt(o[8] * o[8] + o[9] * o[9]);
            fromReward = -step.Info.Get<double>("reward_dist");
            // 0.1435 m, True: the same number the reward charges
            Console.WriteLine($"{fromObservation:F4} m, {fromObservation == fromReward}");
            #endregion
            Assert.AreEqual(0.14351781974838265, fromObservation);
            Assert.AreEqual(fromReward, fromObservation);
        }

        /// <summary>Where the target starts: the seed places it anywhere within 0.2 m of the arm's base, here 0.155 m away.</summary>
        [TestMethod]
        public void WhereTheTargetStarts() {
            string gap = "";
            #region WhereTheTargetStarts
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Reacher-v5");
            using ResetResult<NDArray> start = env.Reset(seed: 42);
            (NDArray observation, Info info) = start;
            double[] o = observation.ToArray<double>();
            // the target's x and y are items 4 and 5, from the arm's base
            gap = Math.Sqrt(o[4] * o[4] + o[5] * o[5]).ToString("F3", CultureInfo.InvariantCulture);
            Console.WriteLine($"{gap} m");   // 0.155 m: every seed puts it within 0.2 m of the base
            #endregion
            Assert.AreEqual("0.155", gap);
        }

        /// <summary>
        ///     Cameras and depth: the model has no camera of its own, so the frames come from Gymnasium's free camera for this
        ///     env; <c>rgbd_tuple</c> draws colour and depth together, and the depth is the raw depth buffer.
        /// </summary>
        [TestMethod]
        public void CamerasAndDepth() {
            string nearest = "";
            #region CamerasAndDepth
            // no camera in the model: Gymnasium's free camera, 1.24 m off at 45°; rgbd_tuple draws colour and depth together
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Reacher-v5",
                new { render_mode = "rgbd_tuple" });
            using ResetResult<NDArray> start = env.Reset(seed: 42);
            using var frames = (RenderResult.RgbDepth)env.Render();
            Console.WriteLine($"{frames.RgbFrame.Shape} {frames.DepthFrame.Shape}");   // (480, 480, 3) (480, 480)
            // MuJoCo's raw depth buffer, 0 at the near plane and 1 at the far one,
            // not metres: the nearest point, the floor at the bottom edge, is 0.986
            using NDArray closest = frames.DepthFrame.min();
            nearest = closest.GetSingle().ToString("F3", CultureInfo.InvariantCulture);
            Console.WriteLine(nearest);   // 0.986
            #endregion
            // The printed line is the page's comment, as Shape prints itself.
            Assert.AreEqual("(480, 480, 3) (480, 480)", $"{frames.RgbFrame.Shape} {frames.DepthFrame.Shape}");
            Assert.AreEqual("0.986", nearest);
        }

        /// <summary>
        ///     Many arms at once: Reacher has no vector env of its own, so <c>MakeVec</c> steps separate envs in turn
        ///     (<c>sync</c>, its default here), each with its own simulation.
        /// </summary>
        [TestMethod]
        public void ManyArms() {
            #region ManyArms
            // No vector env of its own: MakeVec steps 8 envs in turn ("sync")
            using VectorEnv<NDArray, NDArray> envs
                = Gymnasium.MakeVec<NDArray, NDArray>("Reacher-v5", numEnvs: 8);
            using VectorResetResult<NDArray> start
                = envs.Reset(seed: 42);   // 8 x 10 observations
            (NDArray observations, Info infos) = start;
            for (int t = 0; t < 100; t++) {
                using NDArray actions = envs.ActionSpace.Sample();   // 8 x 2 torques
                using VectorStepResult<NDArray> step
                    = envs.Step(actions);   // every 50 steps each arm gets a new target
            }
            #endregion
            Assert.AreEqual(8, envs.NumEnvs);
            Assert.IsInstanceOfType(envs, typeof(SyncVectorEnv<NDArray, NDArray>));
            CollectionAssert.AreEqual(new long[] { 8, 10 }, observations.shape.Select(d => (long)d).ToArray());
        }

        /// <summary>
        ///     The Arguments section's calls: kwargs by Gymnasium's names, or the generated options record by C# names; both
        ///     build the same env, here charging a tenth of the control cost.
        /// </summary>
        [TestMethod]
        public void Arguments() {
            #region Arguments
            using IEnv byName = Gymnasium.Make("Reacher-v5",
                new { reward_control_weight = 0.1 });
            using IEnv byRecord = Gymnasium.Make("Reacher-v5",
                new ReacherEnvV5Options { RewardControlWeight = 0.1 });
            #endregion
            foreach (IEnv env in new[] { byName, byRecord }) {
                var reacher = (ReacherEnvV5)env.Unwrapped;
                Assert.AreEqual(0.1, reacher.RewardControlWeight);
            }
        }
    }
}
