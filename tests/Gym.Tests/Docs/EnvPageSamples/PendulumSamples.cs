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
    ///     The C# samples of Pendulum's page on the docs site (<c>docs/environments/classic_control/pendulum.md</c>), each
    ///     the body of a test, so the unit gate compiles and runs every line the page shows (DR-105). The page includes them
    ///     by DocFX's code-snippet syntax: a region by name, or for the quick start this file's first four lines (its
    ///     usings and a blank) and the quick start's region, by line range.
    /// </summary>
    /// <remarks>
    ///     Footgun: the quick start's include names line numbers (<c>?range=1-4,…</c>), so the first four lines of this file
    ///     must stay <c>using Gym; using Gym.Envs; using NumSharp;</c> and a blank line, and moving the quick start's lines
    ///     moves the page's range; <c>EnvDocPagesTests</c> checks both, so the gate fails before the page shows the wrong
    ///     lines. Each test asserts after its region only, so the assertions never appear on the page; where a region's
    ///     comment states a number (a return, an observation, a space), the assertion pins it, so the page cannot drift
    ///     from what the env computes.
    /// </remarks>
    [TestClass]
    public class PendulumSamples {
        /// <summary>The page's quick start: make the env, reset it with a seed, take one random step.</summary>
        [TestMethod]
        public void QuickStart() {
            #region QuickStart
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Pendulum-v1");
            // [cos(theta), sin(theta), theta_dot], float32
            (NDArray observation, Info info) = env.Reset(seed: 42);
            // one float32 torque in [-2, 2]
            using NDArray action = env.ActionSpace.Sample();
            observation.Dispose();
            (observation, double reward, bool terminated, bool truncated, info)
                = env.Step(action);
            #endregion
            // The quick start's first observation, read again: the same seeded reset gives the same one.
            using ResetResult<NDArray> again = env.Reset(seed: 42);
            NDArray first = again.Observation;
            Assert.AreEqual(3, first.size);
            Assert.AreEqual(3, observation.size);
            Assert.AreEqual(1, action.size);
            Assert.AreEqual(3, first.size);
            // The cost is at most pi^2 + 0.1 * 8^2 + 0.001 * 2^2 = 16.2736044, and never negative.
            Assert.IsTrue(reward <= 0 && reward >= -16.2736045);
            Assert.IsFalse(terminated);
            Assert.IsFalse(truncated);
            Assert.IsNotNull(info);
        }

        /// <summary>
        ///     Run an episode: the example's controller, which pumps energy into the swing, brakes past the upright rest's
        ///     energy and holds the top with a PD law, from seed 5 (the episode the page's video plays).
        /// </summary>
        [TestMethod]
        public void RunAnEpisode() {
            #region RunAnEpisode
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Pendulum-v1");
            (NDArray observation, Info info) = env.Reset(seed: 5);
            double episodeReturn = 0;
            while (true) {
                float cos = observation.GetSingle(0), sin = observation.GetSingle(1);
                double theta = Math.Atan2(sin, cos), w = observation.GetSingle(2);
                double energy = 0.5 * w * w + 15 * Math.Cos(theta);   // upright rest: 15
                double u = Math.Cos(theta) > 0.85 ? -(12 * theta + 2.5 * w)   // hold
                    : energy < 15 ? 2.0 * (w >= 0 ? 1 : -1)   // pump
                    : -0.5 * Math.Sign(w);   // brake
                using NDArray action = np.array(new[] { (float)Math.Clamp(u, -2.0, 2.0) });
                observation.Dispose();
                (observation, double reward, bool terminated, bool truncated, info)
                    = env.Step(action);
                episodeReturn += reward;
                if (terminated || truncated)
                    break;
            }
            Console.WriteLine($"return {episodeReturn:F1}");   // -114.1
            #endregion
            // examples/Pendulum's own episode from seed 5, which Gymnasium computes as -114.07403165309204.
            Assert.AreEqual(-114.07403165309204, episodeReturn, 1e-9);
        }

        /// <summary>Start elsewhere: the reset's <c>x_init</c> and <c>y_init</c> options bound the starting angle and speed.</summary>
        [TestMethod]
        public void StartElsewhere() {
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Pendulum-v1");
            #region StartElsewhere
            // theta from [-pi/2, pi/2), theta_dot from [-0.5, 0.5)
            var options = new Dictionary<string, object?> {
                ["x_init"] = Math.PI / 2, ["y_init"] = 0.5,
            };
            (NDArray observation, Info info) = env.Reset(seed: 123, options: options);
            // [0.8403459, 0.54205054, -0.44617897]
            #endregion
            // Gymnasium's docstring prints this observation for the same call.
            CollectionAssert.AreEqual(new[] { 0.8403459f, 0.54205054f, -0.44617897f }, observation.ToArray<float>());
        }

        /// <summary>Change the physics: the env's own properties, read by every later step.</summary>
        [TestMethod]
        public void ChangePhysics() {
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("Pendulum-v1");
            #region ChangePhysics
            var pendulum = (PendulumEnv)env.Unwrapped;
            pendulum.Gravity = 9.81;   // Gymnasium's g; the default is 10
            pendulum.MaxTorque = 3.0;   // a stronger motor: clips at 3, not 2
            #endregion
            Assert.AreEqual(9.81, ((PendulumEnv)env.Unwrapped).Gravity);
            // The action space keeps its constructor bounds: only the step's clip moves.
            Assert.AreEqual("Box(-2.0, 2.0, (1,), float32)", env.ActionSpace.ToString());
        }

        /// <summary>Save a frame: an <c>rgb_array</c> render as an image file.</summary>
        [TestMethod]
        public void SaveFrame() {
            #region SaveFrame
            using Env<NDArray, NDArray> env
                = Gymnasium.Make<NDArray, NDArray>("Pendulum-v1", RenderMode.RgbArray);
            env.Reset(seed: 5);
            using RenderResult result = env.Render();
            using Image frame = (Image)result;   // 500x500, Gymnasium's pixels
            frame.SaveAsPng("pendulum.png");
            #endregion
            // The test's working folder is its output folder; the file is removed so runs leave nothing behind.
            Assert.AreEqual(500, frame.Width);
            Assert.AreEqual(500, frame.Height);
            Assert.IsTrue(File.Exists("pendulum.png"));
            File.Delete("pendulum.png");
        }

        /// <summary>
        ///     A torque in [-1, 1]: <c>RescaleAction</c> maps the space most algorithms expect onto the env's [-2, 2], the
        ///     unnormalised space Gymnasium's env checker warns about.
        /// </summary>
        [TestMethod]
        public void RescaleTheTorque() {
            #region RescaleTheTorque
            using var env = new RescaleAction<NDArray, NDArray>(
                Gymnasium.Make<NDArray, NDArray>("Pendulum-v1"), -1f, 1f);
            Console.WriteLine(env.ActionSpace);   // Box(-1.0, 1.0, (1,), float32)
            env.Reset(seed: 42);
            using NDArray full = np.array(new[] { 1f });   // the motor's full 2 N m
            using StepResult<NDArray> step = env.Step(full);
            #endregion
            Assert.AreEqual("Box(-1.0, 1.0, (1,), float32)", env.ActionSpace.ToString());
            // The same step as a torque of 2 on the unwrapped env: Gymnasium's reward for it.
            Assert.AreEqual(-2.968425241430033, step.Reward, 1e-12);
        }

        /// <summary>
        ///     Many pendulums at once: Pendulum has no vector env of its own, so <c>MakeVec</c> steps separate envs in
        ///     turn (<c>sync</c>, its default here).
        /// </summary>
        [TestMethod]
        public void ManyPendulums() {
            #region ManyPendulums
            // No vector env of its own: MakeVec steps 8 envs in turn ("sync")
            using VectorEnv<NDArray, NDArray> envs
                = Gymnasium.MakeVec<NDArray, NDArray>("Pendulum-v1", numEnvs: 8);
            using VectorResetResult<NDArray> start = envs.Reset(seed: 42);
            for (int t = 0; t < 200; t++) {
                using NDArray actions = envs.ActionSpace.Sample();   // (8, 1) torques
                using VectorStepResult<NDArray> step
                    = envs.Step(actions);   // only the time limit ends one
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
            using IEnv byName = Gymnasium.Make("Pendulum-v1",
                new { render_mode = "rgb_array", g = 9.81 });
            using IEnv byRecord = Gymnasium.Make("Pendulum-v1",
                new PendulumEnvOptions { RenderMode = "rgb_array", G = 9.81 });
            #endregion
            Assert.AreEqual(9.81, ((PendulumEnv)byName.Unwrapped).Gravity);
            Assert.AreEqual(9.81, ((PendulumEnv)byRecord.Unwrapped).Gravity);
            Assert.AreEqual("rgb_array", byRecord.RenderMode);
        }
    }
}
