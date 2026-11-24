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
    ///     The C# samples of MountainCar's page on the docs site (<c>docs/environments/classic_control/mountain_car.md</c>),
    ///     each the body of a test, so the unit gate compiles and runs every line the page shows (DR-105). The page includes
    ///     them by DocFX's code-snippet syntax: a region by name, or for the quick start this file's first four lines (its
    ///     usings and a blank) and the quick start's region, by line range.
    /// </summary>
    /// <remarks>
    ///     Footgun: the quick start's include names line numbers (<c>?range=1-4,…</c>), so the first four lines of this file
    ///     must stay <c>using Gym; using Gym.Envs; using NumSharp;</c> and a blank line, and moving the quick start's lines
    ///     moves the page's range; <c>EnvDocPagesTests</c> checks both, so the gate fails before the page shows the wrong
    ///     lines. Each test asserts after its region only, so the assertions never appear on the page; where a region's
    ///     comment states a number (a return, a cell, a step count), the assertion pins it, so the page cannot drift from
    ///     what the env computes.
    /// </remarks>
    [TestClass]
    public class MountainCarSamples {
        /// <summary>The page's quick start: make the env, reset it with a seed, take one random step.</summary>
        [TestMethod]
        public void QuickStart() {
            #region QuickStart
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("MountainCar-v0");
            // [position, velocity], float32
            (NDArray observation, Info info) = env.Reset(seed: 42);
            // 0 accelerates left, 1 not at all, 2 right
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
            // Gymnasium's own example: seed 42 starts the car at -0.4452088, at rest.
            Assert.AreEqual(-0.4452088f, first.GetSingle(0));
            Assert.AreEqual(2, first.size);
            Assert.AreEqual(-1.0, reward);
            Assert.IsFalse(terminated);
            Assert.IsFalse(truncated);
            Assert.IsNotNull(info);
        }

        /// <summary>
        ///     Run an episode: the pumping rule, which accelerates the way the car already moves, reaches the flag from
        ///     seed 4 in 122 steps (the episode the page's video plays).
        /// </summary>
        [TestMethod]
        public void RunAnEpisode() {
            #region RunAnEpisode
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("MountainCar-v0");
            (NDArray observation, Info info) = env.Reset(seed: 4);
            double episodeReturn = 0;
            while (true) {
                // accelerate the way the car moves: right (2) or left (0)
                long push = observation.GetSingle(1) >= 0 ? 2 : 0;
                using NDArray action = NDArray.Scalar(push);
                observation.Dispose();
                (observation, double reward, bool terminated, bool truncated, info)
                    = env.Step(action);
                episodeReturn += reward;
                if (terminated || truncated)
                    break;
            }
            Console.WriteLine($"return {episodeReturn}");   // -122: the flag in 122 steps
            #endregion
            Assert.AreEqual(-122.0, episodeReturn);
        }

        /// <summary>Start elsewhere: the reset's <c>low</c> and <c>high</c> options move the starting position's range.</summary>
        [TestMethod]
        public void StartElsewhere() {
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("MountainCar-v0");
            #region StartElsewhere
            var options = new Dictionary<string, object?> { ["low"] = -0.7, ["high"] = 0.5 };
            (NDArray observation, Info info) = env.Reset(seed: 123, options: options);
            // [0.11882224, 0]: the position drawn from [-0.7, 0.5)
            #endregion
            // Gymnasium's docstring prints this observation for the same call.
            Assert.AreEqual(0.11882224f, observation.GetSingle(0));
            Assert.AreEqual(0f, observation.GetSingle(1));
        }

        /// <summary>
        ///     Change the physics: the env's own properties, read by every later step. An engine stronger than the hill's
        ///     steepest pull climbs straight to the flag.
        /// </summary>
        [TestMethod]
        public void ChangePhysics() {
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("MountainCar-v0");
            #region ChangePhysics
            var mountainCar = (MountainCarEnv)env.Unwrapped;
            mountainCar.Force = 0.003;   // stronger than the hill: it drives straight up
            #endregion
            // Pushing right every step from seed 42 now reaches the flag in 35 steps, as Gymnasium's does.
            env.Reset(seed: 42);
            int steps = 0;
            bool terminated = false, truncated = false;
            while (!terminated && !truncated) {
                using NDArray right = NDArray.Scalar(2L);
                using StepResult<NDArray> step = env.Step(right);
                (terminated, truncated) = (step.Terminated, step.Truncated);
                steps++;
            }
            Assert.IsTrue(terminated);
            Assert.AreEqual(35, steps);
        }

        /// <summary>Save a frame: an <c>rgb_array</c> render as an image file.</summary>
        [TestMethod]
        public void SaveFrame() {
            #region SaveFrame
            using Env<NDArray, NDArray> env
                = Gymnasium.Make<NDArray, NDArray>("MountainCar-v0", RenderMode.RgbArray);
            env.Reset(seed: 4);
            using RenderResult result = env.Render();
            using Image frame = (Image)result;   // 600x400, Gymnasium's pixels
            frame.SaveAsPng("mountaincar.png");
            #endregion
            // The test's working folder is its output folder; the file is removed so runs leave nothing behind.
            Assert.AreEqual(600, frame.Width);
            Assert.AreEqual(400, frame.Height);
            Assert.IsTrue(File.Exists("mountaincar.png"));
            File.Delete("mountaincar.png");
        }

        /// <summary>
        ///     A grid for tabular learning: <c>DiscretizeObservation</c> turns the two continuous numbers into one of 100
        ///     cells, the form a Q-table indexes; seed 42 lands in cell 45, as Gymnasium's own docstring shows.
        /// </summary>
        [TestMethod]
        public void TabularGrid() {
            #region TabularGrid
            // 10 bins of position times 10 of velocity: one of 100 cells
            using var env = new DiscretizeObservation<NDArray, NDArray>(
                Gymnasium.Make<NDArray, NDArray>("MountainCar-v0"), bins: 10);
            Console.WriteLine(env.ObservationSpace);   // Discrete(100)
            (NDArray cell, Info info) = env.Reset(seed: 42);   // 45
            #endregion
            Assert.AreEqual("Discrete(100)", env.ObservationSpace.ToString());
            Assert.AreEqual(45L, cell.GetInt64());
        }

        /// <summary>
        ///     Many cars at once: MountainCar has no vector env of its own, so <c>MakeVec</c> steps separate envs in turn
        ///     (<c>sync</c>, its default here).
        /// </summary>
        [TestMethod]
        public void ManyCars() {
            #region ManyCars
            // No vector env of its own: MakeVec steps 8 envs in turn ("sync")
            using VectorEnv<NDArray, NDArray> envs
                = Gymnasium.MakeVec<NDArray, NDArray>("MountainCar-v0", numEnvs: 8);
            using VectorResetResult<NDArray> start = envs.Reset(seed: 42);
            for (int t = 0; t < 200; t++) {
                using NDArray actions = envs.ActionSpace.Sample();   // one per car
                using VectorStepResult<NDArray> step
                    = envs.Step(actions);   // finished cars reset themselves
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
            using IEnv byName = Gymnasium.Make("MountainCar-v0",
                new { render_mode = "rgb_array", goal_velocity = 0.05 });
            using IEnv byRecord = Gymnasium.Make("MountainCar-v0",
                new MountainCarEnvOptions { RenderMode = "rgb_array", GoalVelocity = 0.05 });
            #endregion
            Assert.AreEqual(0.05, ((MountainCarEnv)byName.Unwrapped).GoalVelocity);
            Assert.AreEqual(0.05, ((MountainCarEnv)byRecord.Unwrapped).GoalVelocity);
            Assert.AreEqual("rgb_array", byRecord.RenderMode);
        }
    }
}
