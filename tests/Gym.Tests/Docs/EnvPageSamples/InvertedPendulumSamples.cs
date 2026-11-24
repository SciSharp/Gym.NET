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
    ///     The C# samples of InvertedPendulum's page on the docs site (<c>docs/environments/mujoco/inverted_pendulum.md</c>),
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
    ///     depth sample's number is rounded to three decimals: Gym.NET draws MuJoCo's frames itself, and its depth values
    ///     differ from Gymnasium's in their last bits (DR-98).
    /// </remarks>
    [TestClass]
    public class InvertedPendulumSamples {
        /// <summary>The page's quick start: make the env, reset it with a seed, take one random step.</summary>
        [TestMethod]
        public void QuickStart() {
            #region QuickStart
            using Env<NDArray, NDArray> env
                = Gymnasium.Make<NDArray, NDArray>("InvertedPendulum-v5");
            // 4 numbers: the cart's position, the pole's angle, and their speeds
            (NDArray observation, Info info) = env.Reset(seed: 42);
            // one force, from -3 to 3: push the cart left or right
            using NDArray action = env.ActionSpace.Sample();
            observation.Dispose();
            (observation, double reward, bool terminated, bool truncated, info)
                = env.Step(action);
            #endregion
            // The quick start's first observation, read again: the same seeded reset gives the same one.
            using ResetResult<NDArray> again = env.Reset(seed: 42);
            NDArray first = again.Observation;
            Assert.AreEqual(4, first.size);
            // Gymnasium's reset of seed 42: the cart at 0.0055 m, the reset's uniform noise of 0.01.
            Assert.AreEqual(0.005479120971119267, first.GetDouble(0));
            _ = (reward, terminated, truncated, info);
            observation.Dispose();
        }

        /// <summary>Run an episode: random pushes until the pole falls or the 1,000-step limit.</summary>
        [TestMethod]
        public void RunAnEpisode() {
            int steps = 0;
            #region RunAnEpisode
            using Env<NDArray, NDArray> env
                = Gymnasium.Make<NDArray, NDArray>("InvertedPendulum-v5");
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
            // random pushes topple the pole within about 6 steps: 5.2 on average
            Console.WriteLine($"{steps} steps, return {episodeReturn:F1}");
            #endregion
            Assert.IsTrue(steps <= 1000);
        }

        /// <summary>
        ///     The falling step pays nothing: a full push to the right from seed 42 topples the pole on the third step, and
        ///     that step's reward is 0 (Gymnasium's v4 paid 1 on it).
        /// </summary>
        [TestMethod]
        public void FallingStep() {
            int steps = 0;
            double episodeReturn = 0, last = -1;
            #region FallingStep
            using Env<NDArray, NDArray> env
                = Gymnasium.Make<NDArray, NDArray>("InvertedPendulum-v5");
            using ResetResult<NDArray> start = env.Reset(seed: 42);
            using NDArray push = np.array(new[] { 3f });   // full force to the right, every step
            while (true) {
                using StepResult<NDArray> step = env.Step(push);
                (steps, episodeReturn, last) = (steps + 1, episodeReturn + step.Reward, step.Reward);
                if (step.Terminated || step.Truncated)
                    break;
            }
            // 3 steps, return 2.0: the step the pole falls on pays 0
            Console.WriteLine($"{steps} steps, return {episodeReturn:F1}");
            #endregion
            Assert.AreEqual(3, steps);
            Assert.AreEqual(2.0, episodeReturn);
            Assert.AreEqual(0.0, last);
        }

        /// <summary>
        ///     The same push in InvertedPendulum-v4: v4 paid the step the pole falls on too, so the same three steps return
        ///     3.0; Gymnasium warns that v4 is out of date, and Gym.NET does the same.
        /// </summary>
        [TestMethod]
        public void FallingStepInV4() {
            int steps = 0;
            double episodeReturn = 0;
            #region FallingStepInV4
            using Env<NDArray, NDArray> env
                = Gymnasium.Make<NDArray, NDArray>("InvertedPendulum-v4");
            using ResetResult<NDArray> start = env.Reset(seed: 42);
            using NDArray push = np.array(new[] { 3f });
            while (true) {
                using StepResult<NDArray> step = env.Step(push);
                (steps, episodeReturn) = (steps + 1, episodeReturn + step.Reward);
                if (step.Terminated || step.Truncated)
                    break;
            }
            // 3 steps, return 3.0: v4 also paid the falling step
            Console.WriteLine($"{steps} steps, return {episodeReturn:F1}");
            #endregion
            Assert.AreEqual(3, steps);
            Assert.AreEqual(3.0, episodeReturn);
        }

        /// <summary>Start without noise: <c>reset_noise_scale</c> 0 resets every seed to the cart in the middle, the pole upright, at rest.</summary>
        [TestMethod]
        public void ResetWithoutNoise() {
            bool allZero = false;
            #region ResetWithoutNoise
            using Env<NDArray, NDArray> env
                = Gymnasium.Make<NDArray, NDArray>("InvertedPendulum-v5",
                    new InvertedPendulumEnvV5Options { ResetNoiseScale = 0.0 });
            using ResetResult<NDArray> start = env.Reset(seed: 42);
            (NDArray observation, Info info) = start;
            // the cart in the middle, the pole upright, both at rest: every observed value 0
            allZero = observation.ToArray<double>().All(v => v == 0);
            Console.WriteLine(allZero);   // True
            #endregion
            Assert.IsTrue(allZero);
        }

        /// <summary>
        ///     Cameras and depth: the model has no camera of its own, so the frames come from Gymnasium's free camera for this
        ///     env; <c>rgbd_tuple</c> draws colour and depth together, and the depth is the raw depth buffer.
        /// </summary>
        [TestMethod]
        public void CamerasAndDepth() {
            string nearest = "";
            #region CamerasAndDepth
            // no camera in the model: Gymnasium's free camera, 2.04 m off; rgbd_tuple draws colour and depth together
            using Env<NDArray, NDArray> env
                = Gymnasium.Make<NDArray, NDArray>("InvertedPendulum-v5",
                    new { render_mode = "rgbd_tuple" });
            using ResetResult<NDArray> start = env.Reset(seed: 42);
            using var frames = (RenderResult.RgbDepth)env.Render();
            Console.WriteLine($"{frames.RgbFrame.Shape} {frames.DepthFrame.Shape}");   // (480, 480, 3) (480, 480)
            // MuJoCo's raw depth buffer, 0 at the near plane and 1 at the far one,
            // not metres: the nearest point, the top of the pole, is 0.987
            using NDArray closest = frames.DepthFrame.min();
            nearest = closest.GetSingle().ToString("F3", CultureInfo.InvariantCulture);
            Console.WriteLine(nearest);   // 0.987
            #endregion
            // The printed line is the page's comment, as Shape prints itself.
            Assert.AreEqual("(480, 480, 3) (480, 480)", $"{frames.RgbFrame.Shape} {frames.DepthFrame.Shape}");
            Assert.AreEqual("0.987", nearest);
        }

        /// <summary>
        ///     Many pendulums at once: InvertedPendulum has no vector env of its own, so <c>MakeVec</c> steps separate envs in
        ///     turn (<c>sync</c>, its default here), each with its own simulation and its own resets.
        /// </summary>
        [TestMethod]
        public void ManyPendulums() {
            #region ManyPendulums
            // No vector env of its own: MakeVec steps 8 envs in turn ("sync")
            using VectorEnv<NDArray, NDArray> envs
                = Gymnasium.MakeVec<NDArray, NDArray>("InvertedPendulum-v5", numEnvs: 8);
            using VectorResetResult<NDArray> start
                = envs.Reset(seed: 42);   // 8 x 4 observations
            (NDArray observations, Info infos) = start;
            for (int t = 0; t < 100; t++) {
                using NDArray actions = envs.ActionSpace.Sample();   // 8 x 1 forces
                using VectorStepResult<NDArray> step
                    = envs.Step(actions);   // fallen pendulums reset themselves
            }
            #endregion
            Assert.AreEqual(8, envs.NumEnvs);
            Assert.IsInstanceOfType(envs, typeof(SyncVectorEnv<NDArray, NDArray>));
            CollectionAssert.AreEqual(new long[] { 8, 4 }, observations.shape.Select(d => (long)d).ToArray());
        }

        /// <summary>
        ///     The Arguments section's calls: kwargs by Gymnasium's names, or the generated options record by C# names; both
        ///     build the same env, here with ten times the starting noise and half the steps a second.
        /// </summary>
        [TestMethod]
        public void Arguments() {
            #region Arguments
            using IEnv byName = Gymnasium.Make("InvertedPendulum-v5",
                new { reset_noise_scale = 0.1, frame_skip = 4 });
            using IEnv byRecord = Gymnasium.Make("InvertedPendulum-v5",
                new InvertedPendulumEnvV5Options { ResetNoiseScale = 0.1, FrameSkip = 4 });
            #endregion
            foreach (IEnv env in new[] { byName, byRecord }) {
                var pendulum = (InvertedPendulumEnvV5)env.Unwrapped;
                Assert.AreEqual(0.1, pendulum.ResetNoiseScale);
                Assert.AreEqual(4, pendulum.FrameSkip);
            }
        }
    }
}
