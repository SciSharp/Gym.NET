using Gym;
using Gym.Envs;
using NumSharp;

using System;
using System.Collections.Generic;
using System.IO;
using Gym.Environments.Envs.Classic;
using Gym.Observations;
using Gym.Vector;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SixLabors.ImageSharp;

// Not under Gym.Tests: there the namespace Gym.Tests.Gymnasium (DR-06) shadows the Gymnasium facade, and the samples
// must read Gymnasium.Make(...) exactly as a reader's code does.
namespace Gym.Docs.EnvPageSamples {
    /// <summary>
    ///     The C# samples of CartPole's page on the docs site (<c>docs/environments/classic_control/cart_pole.md</c>), each
    ///     the body of a test, so the unit gate compiles and runs every line the page shows (DR-105). The page includes them
    ///     by DocFX's code-snippet syntax: a region by name, or for the quick start this file's first three lines (its
    ///     usings) and the quick start's region, by line range.
    /// </summary>
    /// <remarks>
    ///     Footgun: the quick start's include names line numbers (<c>?range=1-4,…</c>), so the first four lines of this file
    ///     must stay <c>using Gym; using Gym.Envs; using NumSharp;</c> and a blank line, and moving the quick start's lines
    ///     moves the page's range; <c>EnvDocPagesTests</c> checks both, so the gate fails before the page shows the wrong
    ///     lines. Each test asserts after its region only, so the assertions never appear on the page.
    /// </remarks>
    [TestClass]
    public class CartPoleSamples {
        /// <summary>The page's quick start: make the env, reset it with a seed, take one random step.</summary>
        [TestMethod]
        public void QuickStart() {
            #region QuickStart
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("CartPole-v1");
            // [x, x_dot, theta, theta_dot], float32
            (NDArray observation, Info info) = env.Reset(seed: 42);
            // 0 pushes the cart left, 1 right
            using NDArray action = env.ActionSpace.Sample();
            observation.Dispose();
            (observation, double reward, bool terminated, bool truncated, info)
                = env.Step(action);
            #endregion
            // The quick start's first observation, read again: the same seeded reset gives the same one.
            using ResetResult<NDArray> again = env.Reset(seed: 42);
            NDArray first = again.Observation;
            Assert.AreEqual(4, first.size);
            Assert.AreEqual(4, observation.size);
            Assert.AreEqual(1.0, reward);
            Assert.IsFalse(truncated);
            Assert.IsNotNull(info);
        }

        /// <summary>Run an episode: random actions until it terminates or the time limit truncates it.</summary>
        [TestMethod]
        public void RunAnEpisode() {
            #region RunAnEpisode
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("CartPole-v1");
            (NDArray observation, Info info) = env.Reset(seed: 42);
            double episodeReturn = 0;
            while (true) {
                using NDArray action = env.ActionSpace.Sample();
                observation.Dispose();
                (observation, double reward, bool terminated, bool truncated, info)
                    = env.Step(action);
                episodeReturn += reward;
                if (terminated || truncated)
                    break;
            }
            Console.WriteLine($"return {episodeReturn}");
            #endregion
            // A random policy keeps the pole up for about 22 steps on average; every step scores 1.
            Assert.IsTrue(episodeReturn >= 1 && episodeReturn <= 500);
            Assert.AreEqual(Math.Floor(episodeReturn), episodeReturn);
        }

        /// <summary>Start from wider bounds: the reset's <c>low</c> and <c>high</c> options widen the starting state's range.</summary>
        [TestMethod]
        public void WiderBounds() {
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("CartPole-v1");
            #region WiderBounds
            var options = new Dictionary<string, object?> { ["low"] = -0.2, ["high"] = 0.2 };
            (NDArray observation, Info info) = env.Reset(seed: 7, options: options);
            #endregion
            foreach (float value in observation.ToArray<float>())
                Assert.IsTrue(value >= -0.2f && value <= 0.2f);
        }

        /// <summary>Change the physics: the env's own properties, read by every later step.</summary>
        [TestMethod]
        public void ChangePhysics() {
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("CartPole-v1");
            #region ChangePhysics
            var cartPole = (CartPoleEnv)env.Unwrapped;
            cartPole.Gravity = 1.62;   // the Moon: the pole falls slower
            #endregion
            Assert.AreEqual(1.62, ((CartPoleEnv)env.Unwrapped).Gravity);
        }

        /// <summary>Save a frame: an <c>rgb_array</c> render as an image file.</summary>
        [TestMethod]
        public void SaveFrame() {
            #region SaveFrame
            using Env<NDArray, NDArray> env
                = Gymnasium.Make<NDArray, NDArray>("CartPole-v1", RenderMode.RgbArray);
            env.Reset(seed: 42);
            using RenderResult result = env.Render();
            using Image frame = (Image)result;   // 600x400, Gymnasium's pixels
            frame.SaveAsPng("cartpole.png");
            #endregion
            // The test's working folder is its output folder; the file is removed so runs leave nothing behind.
            Assert.AreEqual(600, frame.Width);
            Assert.AreEqual(400, frame.Height);
            Assert.IsTrue(File.Exists("cartpole.png"));
            File.Delete("cartpole.png");
        }

        /// <summary>Many carts at once: CartPole's own vector env, what <c>MakeVec("CartPole-v1", 8)</c> builds.</summary>
        [TestMethod]
        public void ManyCarts() {
            #region ManyCarts
            // What Gymnasium.MakeVec("CartPole-v1", 8) builds: eight carts in one env
            using var envs = new CartPoleVectorEnv(numEnvs: 8);
            using VectorResetResult<NDArray> start = envs.Reset(seed: 42);
            for (int t = 0; t < 1000; t++) {
                using NDArray actions = envs.ActionSpace.Sample();   // one per cart
                using VectorStepResult<NDArray> step
                    = envs.Step(actions);   // finished carts reset themselves
            }
            #endregion
            Assert.AreEqual(8, envs.NumEnvs);
        }

        /// <summary>Episode statistics: <c>RecordEpisodeStatistics</c> puts the return, length and time in the last step's info.</summary>
        [TestMethod]
        public void EpisodeStatistics() {
            Info? last = null;
            #region EpisodeStatistics
            using var env = new RecordEpisodeStatistics<NDArray, NDArray>(
                Gymnasium.Make<NDArray, NDArray>("CartPole-v1"));
            env.Reset(seed: 42);
            while (true) {
                using NDArray action = env.ActionSpace.Sample();
                (_, _, bool terminated, bool truncated, Info info) = env.Step(action);
                if (terminated || truncated) {
                    // r: the return, l: the length, t: the seconds it took
                    Console.WriteLine(info["episode"]);
                    last = info;
                    break;
                }
            }
            #endregion
            Assert.IsNotNull(last);
            Assert.IsTrue(last!.ContainsKey("episode"));
        }

        /// <summary>
        ///     The Arguments section's calls: kwargs by Gymnasium's names, or the generated options record by C# names; both
        ///     build the same env.
        /// </summary>
        [TestMethod]
        public void Arguments() {
            #region Arguments
            using IEnv byName = Gymnasium.Make("CartPole-v1",
                new { render_mode = "rgb_array", sutton_barto_reward = true });
            using IEnv byRecord = Gymnasium.Make("CartPole-v1",
                new CartPoleEnvOptions { RenderMode = "rgb_array", SuttonBartoReward = true });
            #endregion
            Assert.IsTrue(((CartPoleEnv)byName.Unwrapped).SuttonBartoReward);
            Assert.IsTrue(((CartPoleEnv)byRecord.Unwrapped).SuttonBartoReward);
            Assert.AreEqual("rgb_array", byRecord.RenderMode);
        }
    }
}
