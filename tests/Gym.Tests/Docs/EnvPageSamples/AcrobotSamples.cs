using Gym;
using Gym.Envs;
using NumSharp;

using System;
using System.Collections.Generic;
using System.IO;
using Gym.Environments.Envs.Classic;
using Gym.Observations;
using Gym.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SixLabors.ImageSharp;

// Not under Gym.Tests: there the namespace Gym.Tests.Gymnasium (DR-06) shadows the Gymnasium facade, and the samples
// must read Gymnasium.Make(...) exactly as a reader's code does.
namespace Gym.Docs.EnvPageSamples {
    /// <summary>
    ///     The C# samples of Acrobot's page on the docs site (<c>docs/environments/classic_control/acrobot.md</c>), each the
    ///     body of a test, so the unit gate compiles and runs every line the page shows (DR-105). The page includes them by
    ///     DocFX's code-snippet syntax: a region by name, or for the quick start this file's first four lines (its usings
    ///     and a blank) and the quick start's region, by line range.
    /// </summary>
    /// <remarks>
    ///     Footgun: the quick start's include names line numbers (<c>?range=1-4,…</c>), so the first four lines of this file
    ///     must stay <c>using Gym; using Gym.Envs; using NumSharp;</c> and a blank line, and moving the quick start's lines
    ///     moves the page's range; <c>EnvDocPagesTests</c> checks both, so the gate fails before the page shows the wrong
    ///     lines. Each test asserts after its region only, so the assertions never appear on the page; where a region's
    ///     comment or the page states a number (a return, a step count, an observation), the assertion pins it, so the
    ///     page cannot drift from what the env computes.
    /// </remarks>
    [TestClass]
    public class AcrobotSamples {
        /// <summary>The page's quick start: make the env, reset it with a seed, take one random step.</summary>
        [TestMethod]
        public void QuickStart() {
            #region QuickStart
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Acrobot-v1");
            // cos and sin of both joint angles, then both speeds: float32
            (NDArray observation, Info info) = env.Reset(seed: 42);
            // 0: -1 N m on the elbow, 1: none, 2: +1 N m
            using NDArray action = env.ActionSpace.Sample();
            observation.Dispose();
            (observation, double reward, bool terminated, bool truncated, info)
                = env.Step(action);
            #endregion
            // The quick start's first observation, read again: the same seeded reset gives the same one.
            using ResetResult<NDArray> again = env.Reset(seed: 42);
            NDArray first = again.Observation;
            Assert.AreEqual(6, first.size);
            Assert.AreEqual(6, observation.size);
            // From hanging, one step cannot lift the tip a metre above the pivot.
            Assert.AreEqual(-1.0, reward);
            Assert.IsFalse(terminated);
            Assert.IsFalse(truncated);
            Assert.IsNotNull(info);
        }

        /// <summary>
        ///     Run an episode: the example's rule, which torques the elbow the way the second link turns, swings the tip
        ///     over the line from seed 1 in 65 steps (the episode the page's video plays).
        /// </summary>
        [TestMethod]
        public void RunAnEpisode() {
            #region RunAnEpisode
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Acrobot-v1");
            (NDArray observation, Info info) = env.Reset(seed: 1);
            double episodeReturn = 0;
            while (true) {
                // torque the elbow the way the second link turns: +1 (2) or -1 (0)
                long push = observation.GetSingle(5) > 0 ? 2 : 0;
                using NDArray action = NDArray.Scalar(push);
                observation.Dispose();
                (observation, double reward, bool terminated, bool truncated, info)
                    = env.Step(action);
                episodeReturn += reward;
                if (terminated || truncated)
                    break;
            }
            Console.WriteLine($"return {episodeReturn}");   // -64: over the line in 65 steps
            #endregion
            Assert.AreEqual(-64.0, episodeReturn);
        }

        /// <summary>Start elsewhere: the reset's <c>low</c> and <c>high</c> options widen the starting state's range.</summary>
        [TestMethod]
        public void StartElsewhere() {
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Acrobot-v1");
            #region StartElsewhere
            // both angles and both speeds from [-0.2, 0.2) instead of [-0.1, 0.1)
            var options = new Dictionary<string, object?> { ["low"] = -0.2, ["high"] = 0.2 };
            (NDArray observation, Info info) = env.Reset(seed: 123, options: options);
            #endregion
            // Gymnasium's docstring prints this observation for the same call.
            CollectionAssert.AreEqual(new[] { 0.997341f, 0.07287608f, 0.9841162f, -0.17752565f, -0.11185605f, -0.12625128f },
                observation.ToArray<float>());
        }

        /// <summary>
        ///     Change the physics: the NIPS paper's equations instead of the book's, and a finer integration step. The page
        ///     states what each does to the example's rule from seed 1 (352 and 305 steps), which this test measures.
        /// </summary>
        [TestMethod]
        public void ChangePhysics() {
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Acrobot-v1");
            #region ChangePhysics
            var acrobot = (AcrobotEnv)env.Unwrapped;
            acrobot.BookOrNips = "nips";   // the NIPS paper's equations, not the book's
            acrobot.Dt = 0.05;   // RK4 over 0.05 s a step, not 0.2
            #endregion
            Assert.AreEqual("nips", acrobot.BookOrNips);
            Assert.AreEqual(0.05, acrobot.Dt);
            // Each change alone, on the rule of RunAnEpisode from seed 1, as Gymnasium measures it.
            Assert.AreEqual(352, StepsToTheLine(nips: true, dt: 0.2));
            Assert.AreEqual(305, StepsToTheLine(nips: false, dt: 0.05));
        }

        /// <summary>Plays the example's rule from seed 1 on an Acrobot with the given equations and step.</summary>
        /// <param name="nips">Whether to use the NIPS paper's equations.</param>
        /// <param name="dt">The integration step, in seconds.</param>
        /// <returns>The steps until the tip passed the line (or the 500-step limit).</returns>
        private static int StepsToTheLine(bool nips, double dt) {
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Acrobot-v1");
            var acrobot = (AcrobotEnv)env.Unwrapped;
            acrobot.BookOrNips = nips ? "nips" : "book";
            acrobot.Dt = dt;
            (NDArray observation, Info info) = env.Reset(seed: 1);
            for (int steps = 1; ; steps++) {
                using NDArray action = NDArray.Scalar(observation.GetSingle(5) > 0 ? 2L : 0L);
                observation.Dispose();
                (observation, _, bool terminated, bool truncated, info) = env.Step(action);
                if (terminated || truncated)
                    return steps;
            }
        }

        /// <summary>Save a frame: an <c>rgb_array</c> render as an image file.</summary>
        [TestMethod]
        public void SaveFrame() {
            #region SaveFrame
            using Env<NDArray, NDArray> env
                = Gymnasium.Make<NDArray, NDArray>("Acrobot-v1", RenderMode.RgbArray);
            env.Reset(seed: 1);
            using RenderResult result = env.Render();
            using Image frame = (Image)result;   // 500x500, Gymnasium's pixels
            frame.SaveAsPng("acrobot.png");
            #endregion
            // The test's working folder is its output folder; the file is removed so runs leave nothing behind.
            Assert.AreEqual(500, frame.Width);
            Assert.AreEqual(500, frame.Height);
            Assert.IsTrue(File.Exists("acrobot.png"));
            File.Delete("acrobot.png");
        }

        /// <summary>
        ///     The tip's height from an observation: the quantity the episode ends on, from the cosines and sines the
        ///     observation gives, without an arctangent.
        /// </summary>
        [TestMethod]
        public void TheTipsHeight() {
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Acrobot-v1");
            (NDArray observation, Info info) = env.Reset(seed: 42);
            #region TheTipsHeight
            float c1 = observation.GetSingle(0), s1 = observation.GetSingle(1);
            float c2 = observation.GetSingle(2), s2 = observation.GetSingle(3);
            // cos(theta1 + theta2) = c1 c2 - s1 s2; the episode ends when height > 1
            double height = -c1 - (c1 * c2 - s1 * s2);   // -2.00: hanging straight down
            #endregion
            // Gymnasium's own terminal test reads -1.9975934687264207 from the same state in float64.
            Assert.AreEqual(-1.9976, height, 1e-4);
        }

        /// <summary>
        ///     Many acrobots at once: Acrobot has no vector env of its own, so <c>MakeVec</c> steps separate envs in turn
        ///     (<c>sync</c>, its default here).
        /// </summary>
        [TestMethod]
        public void ManyAcrobots() {
            #region ManyAcrobots
            // No vector env of its own: MakeVec steps 8 envs in turn ("sync")
            using VectorEnv<NDArray, NDArray> envs
                = Gymnasium.MakeVec<NDArray, NDArray>("Acrobot-v1", numEnvs: 8);
            using VectorResetResult<NDArray> start = envs.Reset(seed: 42);
            for (int t = 0; t < 200; t++) {
                using NDArray actions = envs.ActionSpace.Sample();   // one per acrobot
                using VectorStepResult<NDArray> step
                    = envs.Step(actions);   // finished ones reset themselves
            }
            #endregion
            Assert.AreEqual(8, envs.NumEnvs);
            Assert.IsInstanceOfType(envs, typeof(SyncVectorEnv<NDArray, NDArray>));
        }

        /// <summary>
        ///     The Arguments section's calls: kwargs by Gymnasium's names, or the generated options record by C# names; both
        ///     build the same env.
        /// </summary>
        [TestMethod]
        public void Arguments() {
            #region Arguments
            using IEnv byName = Gymnasium.Make("Acrobot-v1",
                new { render_mode = "rgb_array" });
            using IEnv byRecord = Gymnasium.Make("Acrobot-v1",
                new AcrobotEnvOptions { RenderMode = "rgb_array" });
            #endregion
            Assert.AreEqual("rgb_array", byName.RenderMode);
            Assert.AreEqual("rgb_array", byRecord.RenderMode);
        }
    }
}
