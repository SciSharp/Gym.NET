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
    ///     The C# samples of MountainCarContinuous's page on the docs site
    ///     (<c>docs/environments/classic_control/mountain_car_continuous.md</c>), each the body of a test, so the unit gate
    ///     compiles and runs every line the page shows (DR-105). The page includes them by DocFX's code-snippet syntax: a
    ///     region by name, or for the quick start this file's first four lines (its usings and a blank) and the quick
    ///     start's region, by line range.
    /// </summary>
    /// <remarks>
    ///     Footgun: the quick start's include names line numbers (<c>?range=1-4,…</c>), so the first four lines of this file
    ///     must stay <c>using Gym; using Gym.Envs; using NumSharp;</c> and a blank line, and moving the quick start's lines
    ///     moves the page's range; <c>EnvDocPagesTests</c> checks both, so the gate fails before the page shows the wrong
    ///     lines. Each test asserts after its region only, so the assertions never appear on the page; where a region's
    ///     comment states a number (a return, a reward, a step count), the assertion pins it, so the page cannot drift
    ///     from what the env computes.
    /// </remarks>
    [TestClass]
    public class MountainCarContinuousSamples {
        /// <summary>The page's quick start: make the env, reset it with a seed, take one random step.</summary>
        [TestMethod]
        public void QuickStart() {
            #region QuickStart
            using Env<NDArray, NDArray> env
                = Gymnasium.Make<NDArray, NDArray>("MountainCarContinuous-v0");
            // [position, velocity], float32
            (NDArray observation, Info info) = env.Reset(seed: 42);
            // one float32 force in [-1, 1]
            using NDArray action = env.ActionSpace.Sample();
            observation.Dispose();
            (observation, double reward, bool terminated, bool truncated, info)
                = env.Step(action);
            #endregion
            // The quick start's first observation, read again: the same seeded reset gives the same one.
            using ResetResult<NDArray> again = env.Reset(seed: 42);
            NDArray first = again.Observation;
            Assert.AreEqual(2, first.size);
            Assert.AreEqual(2, observation.size);
            // The same start as MountainCar-v0's from seed 42: both draw the position from one generator alike.
            Assert.AreEqual(-0.4452088f, first.GetSingle(0));
            Assert.AreEqual(1, action.size);
            Assert.AreEqual(2, first.size);
            // A step that does not reach the flag costs 0.1 times the squared force, at most 0.1.
            Assert.IsTrue(reward <= 0 && reward >= -0.1);
            Assert.IsFalse(terminated);
            Assert.IsFalse(truncated);
            Assert.IsNotNull(info);
        }

        /// <summary>
        ///     Run an episode: the left-first pumping rule at full force, the example's, reaches the flag from seed 4 in 78
        ///     steps (the episode the page's video plays).
        /// </summary>
        [TestMethod]
        public void RunAnEpisode() {
            #region RunAnEpisode
            using Env<NDArray, NDArray> env
                = Gymnasium.Make<NDArray, NDArray>("MountainCarContinuous-v0");
            (NDArray observation, Info info) = env.Reset(seed: 4);
            double episodeReturn = 0;
            int steps = 0;
            while (true) {
                // full force the way the car moves; left while it is at rest
                float force = observation.GetSingle(1) > 0 ? 1f : -1f;
                using NDArray action = np.array(new[] { force });
                observation.Dispose();
                (observation, double reward, bool terminated, bool truncated, info)
                    = env.Step(action);
                episodeReturn += reward;
                steps++;
                if (terminated || truncated)
                    break;
            }
            // 78 steps, return 92.2: the flag's 100 less 0.1 a step for fuel
            Console.WriteLine($"{steps} steps, return {episodeReturn:F1}");
            #endregion
            Assert.AreEqual(78, steps);
            Assert.AreEqual(92.2, episodeReturn, 1e-9);
        }

        /// <summary>Start elsewhere: the reset's <c>low</c> and <c>high</c> options move the starting position's range.</summary>
        [TestMethod]
        public void StartElsewhere() {
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("MountainCarContinuous-v0");
            #region StartElsewhere
            var options = new Dictionary<string, object?> { ["low"] = -0.7, ["high"] = -0.5 };
            (NDArray observation, Info info) = env.Reset(seed: 123, options: options);
            // [-0.5635296, 0]: the position drawn from [-0.7, -0.5)
            #endregion
            // Gymnasium's docstring prints this observation for the same call.
            Assert.AreEqual(-0.5635296f, observation.GetSingle(0));
            Assert.AreEqual(0f, observation.GetSingle(1));
        }

        /// <summary>
        ///     Change the physics: the env's own properties, read by every later step. An engine stronger than the hill's
        ///     steepest pull climbs straight to the flag.
        /// </summary>
        [TestMethod]
        public void ChangePhysics() {
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("MountainCarContinuous-v0");
            #region ChangePhysics
            var mountainCar = (MountainCarEnv)env.Unwrapped;
            mountainCar.Power = 0.003;   // stronger than the hill: it drives straight up
            #endregion
            // Full force right every step from seed 42 now reaches the flag in 34 steps, as Gymnasium's does.
            env.Reset(seed: 42);
            int steps = 0;
            bool terminated = false, truncated = false;
            while (!terminated && !truncated) {
                using NDArray right = np.array(new[] { 1f });
                using StepResult<NDArray> step = env.Step(right);
                (terminated, truncated) = (step.Terminated, step.Truncated);
                steps++;
            }
            Assert.IsTrue(terminated);
            Assert.AreEqual(34, steps);
        }

        /// <summary>Save a frame: an <c>rgb_array</c> render as an image file.</summary>
        [TestMethod]
        public void SaveFrame() {
            #region SaveFrame
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>(
                "MountainCarContinuous-v0", RenderMode.RgbArray);
            env.Reset(seed: 4);
            using RenderResult result = env.Render();
            using Image frame = (Image)result;   // 600x400, Gymnasium's pixels
            frame.SaveAsPng("mountaincarcontinuous.png");
            #endregion
            // The test's working folder is its output folder; the file is removed so runs leave nothing behind.
            Assert.AreEqual(600, frame.Width);
            Assert.AreEqual(400, frame.Height);
            Assert.IsTrue(File.Exists("mountaincarcontinuous.png"));
            File.Delete("mountaincarcontinuous.png");
        }

        /// <summary>
        ///     Clip the force before it is charged: the env clips a force outside [-1, 1] for the push but charges its
        ///     square in full; <c>ClipAction</c> clips it first, so the same push costs what a force of 1 costs.
        /// </summary>
        [TestMethod]
        public void ClipTheForce() {
            #region ClipTheForce
            using Env<NDArray, NDArray> env
                = Gymnasium.Make<NDArray, NDArray>("MountainCarContinuous-v0");
            using var clipped = new ClipAction<NDArray, NDArray>(
                Gymnasium.Make<NDArray, NDArray>("MountainCarContinuous-v0"));
            using NDArray tooStrong = np.array(new[] { 2f });
            env.Reset(seed: 42);
            clipped.Reset(seed: 42);
            using StepResult<NDArray> raw = env.Step(tooStrong);       // pushes as 1, costs as 2
            using StepResult<NDArray> fixedUp = clipped.Step(tooStrong);   // clipped to 1 first
            Console.WriteLine($"{raw.Reward} {fixedUp.Reward}");   // -0.4 -0.1
            #endregion
            Assert.AreEqual(-0.4, raw.Reward, 1e-12);
            Assert.AreEqual(-0.1, fixedUp.Reward, 1e-12);
            // The push itself was the same: both cars moved alike.
            Assert.AreEqual(raw.Observation.GetSingle(1), fixedUp.Observation.GetSingle(1));
        }

        /// <summary>
        ///     Many cars at once: MountainCarContinuous has no vector env of its own, so <c>MakeVec</c> steps separate envs
        ///     in turn (<c>sync</c>, its default here).
        /// </summary>
        [TestMethod]
        public void ManyCars() {
            #region ManyCars
            // No vector env of its own: MakeVec steps 8 envs in turn ("sync")
            using VectorEnv<NDArray, NDArray> envs
                = Gymnasium.MakeVec<NDArray, NDArray>("MountainCarContinuous-v0", numEnvs: 8);
            using VectorResetResult<NDArray> start = envs.Reset(seed: 42);
            for (int t = 0; t < 200; t++) {
                using NDArray actions = envs.ActionSpace.Sample();   // (8, 1) forces
                using VectorStepResult<NDArray> step
                    = envs.Step(actions);   // finished cars reset themselves
            }
            #endregion
            Assert.AreEqual(8, envs.NumEnvs);
            Assert.IsInstanceOfType(envs, typeof(SyncVectorEnv<NDArray, NDArray>));
        }

        /// <summary>
        ///     The Arguments section's calls: kwargs by Gymnasium's names, or the generated options record by C# names; both
        ///     build the same env. The options record must say <c>Continuous = true</c> itself only when it builds the class
        ///     directly; through the id, the registration supplies it.
        /// </summary>
        [TestMethod]
        public void Arguments() {
            #region Arguments
            using IEnv byName = Gymnasium.Make("MountainCarContinuous-v0",
                new { render_mode = "rgb_array", goal_velocity = 0.05 });
            using IEnv byRecord = Gymnasium.Make("MountainCarContinuous-v0",
                new MountainCarEnvOptions { RenderMode = "rgb_array", GoalVelocity = 0.05 });
            #endregion
            Assert.AreEqual(0.05, ((MountainCarEnv)byName.Unwrapped).GoalVelocity);
            Assert.AreEqual(0.05, ((MountainCarEnv)byRecord.Unwrapped).GoalVelocity);
            // The id's registered kwargs (continuous=True) survive the caller's record, which leaves Continuous unset.
            Assert.IsTrue(((MountainCarEnv)byRecord.Unwrapped).Continuous);
            Assert.AreEqual("rgb_array", byRecord.RenderMode);
        }
    }
}
