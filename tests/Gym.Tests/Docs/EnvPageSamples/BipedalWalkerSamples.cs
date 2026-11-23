using Gym;
using Gym.Envs;
using NumSharp;

using System;
using System.Globalization;
using System.IO;
using System.Linq;
using Gym.Environments.Envs.Box2D;
using Gym.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SixLabors.ImageSharp;

// Not under Gym.Tests: there the namespace Gym.Tests.Gymnasium (DR-06) shadows the Gymnasium facade, and the samples
// must read Gymnasium.Make(...) exactly as a reader's code does.
namespace Gym.Docs.EnvPageSamples {
    /// <summary>
    ///     The C# samples of Bipedal Walker's page on the docs site (<c>docs/environments/box2d/bipedal_walker.md</c>),
    ///     each the body of a test, so the unit gate compiles and runs every line the page shows (DR-105). The page
    ///     includes them by DocFX's code-snippet syntax: a region by name, or for the quick start this file's first four
    ///     lines (its usings and a blank) and the quick start's region, by line range.
    /// </summary>
    /// <remarks>
    ///     Footgun: the quick start's include names line numbers (<c>?range=1-4,…</c>), so the first four lines of this file
    ///     must stay <c>using Gym; using Gym.Envs; using NumSharp;</c> and a blank line, and moving the quick start's lines
    ///     moves the page's range; <c>EnvDocPagesTests</c> checks both, so the gate fails before the page shows the wrong
    ///     lines. Each test asserts after its region only, so the assertions never appear on the page; where a region's
    ///     comment states a number (a step count, a return, a distance), the assertion pins it: those numbers are
    ///     Gymnasium 1.4.0's own for the same seeds and actions, measured with its gait, so the page cannot drift from what
    ///     either computes.
    /// </remarks>
    [TestClass]
    public class BipedalWalkerSamples {
        /// <summary>The page's quick start: make the env, reset it with a seed, take one random step.</summary>
        [TestMethod]
        public void QuickStart() {
            #region QuickStart
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("BipedalWalker-v3");
            // the hull, 4 joints, 2 feet's contacts, 10 lidar readings
            NDArray observation = env.Reset(seed: 42).Observation;
            // hip, knee, hip, knee: the sign turns a motor, the size is its torque
            using NDArray action = env.ActionSpace.Sample();
            var (next, reward, terminated, truncated, info) = env.Step(action);
            #endregion
            Assert.AreEqual(24, observation.size);
            // Gymnasium's reset of seed 42 (one step with the motors off inside the reset): the first lidar ray, straight
            // down, stops at 0.4408 of its range, and both feet stand on the start pad.
            Assert.AreEqual(0.4408141076564789f, observation.GetSingle(14));
            Assert.AreEqual(1f, observation.GetSingle(8));
            Assert.AreEqual(1f, observation.GetSingle(13));
            Assert.AreEqual(0, info.Count);
            _ = (reward, terminated, truncated);
            next.Dispose();
            observation.Dispose();
        }

        /// <summary>Run an episode: random motors until the walker falls, leaves the left edge, or the 1,600-step limit.</summary>
        [TestMethod]
        public void RunAnEpisode() {
            int steps = 0;
            #region RunAnEpisode
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("BipedalWalker-v3");
            NDArray observation = env.Reset(seed: 42).Observation;
            double episodeReturn = 0;
            while (true) {
                using NDArray action = env.ActionSpace.Sample();
                var (next, reward, terminated, truncated, _) = env.Step(action);
                observation.Dispose();
                observation = next;
                episodeReturn += reward;
                steps++;
                if (terminated || truncated)
                    break;
            }
            Console.WriteLine($"return {episodeReturn:F1}");   // random motors: -98.9 on average
            #endregion
            Assert.IsTrue(steps <= 1600);
        }

        /// <summary>
        ///     Gymnasium's gait, <see cref="BipedalWalkerHeuristics"/>, which walks the page's episode, seed 0, to the end of
        ///     the course.
        /// </summary>
        [TestMethod]
        public void TheGait() {
            int steps = 0;
            double episodeReturn = 0;
            #region TheGait
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("BipedalWalker-v3");
            NDArray observation = env.Reset(seed: 0).Observation;
            // Gymnasium's own gait: a state machine with a phase, one per episode
            var gait = new BipedalWalkerHeuristics();
            while (true) {
                using NDArray action = gait.StepHeuristic(observation);
                var (next, reward, terminated, truncated, _) = env.Step(action);
                observation.Dispose();
                observation = next;
                (steps, episodeReturn) = (steps + 1, episodeReturn + reward);
                if (terminated || truncated)
                    break;
            }
            // 1598 steps, return 314.5: the whole course
            Console.WriteLine($"{steps} steps, return {episodeReturn:F1}");
            #endregion
            Assert.AreEqual(1598, steps);
            Assert.AreEqual(314.52605747085136, episodeReturn);
        }

        /// <summary>
        ///     The same gait on the hardcore course from seed 7, the hardcore section's episode: it never reads the lidar, so
        ///     it walks into the first staircase and falls, Gymnasium's numbers to the last bit.
        /// </summary>
        [TestMethod]
        public void TheHardcoreCourse() {
            int steps = 0;
            double episodeReturn = 0;
            #region TheHardcoreCourse
            // stumps, stairs and pits: the same gait, blind to them
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("BipedalWalkerHardcore-v3");
            NDArray observation = env.Reset(seed: 7).Observation;
            var gait = new BipedalWalkerHeuristics();
            while (true) {
                using NDArray action = gait.StepHeuristic(observation);
                var (next, reward, terminated, truncated, _) = env.Step(action);
                observation.Dispose();
                observation = next;
                (steps, episodeReturn) = (steps + 1, episodeReturn + reward);
                if (terminated || truncated)
                    break;
            }
            // 248 steps, return -68.8: it falls on the first staircase
            Console.WriteLine($"{steps} steps, return {episodeReturn:F1}");
            #endregion
            Assert.AreEqual(248, steps);
            Assert.AreEqual(-68.83078918909483, episodeReturn);
        }

        /// <summary>
        ///     Read the lidar: the reset's ten readings as distances, the fraction of each ray's range times
        ///     <see cref="BipedalWalkerEnv.LidarRange"/>; on the flat start pad they grow with the angle, and the last two
        ///     rays reach nothing.
        /// </summary>
        [TestMethod]
        public void ReadTheLidar() {
            #region ReadTheLidar
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("BipedalWalker-v3");
            using var start = env.Reset(seed: 42);
            // items 14 to 23: ten rays from the hull, from straight down to
            // 77° ahead, each the fraction of its range before it hits ground
            for (int i = 0; i < 10; i++)
                Console.Write($"{start.Observation.GetSingle(14 + i) * BipedalWalkerEnv.LidarRange:F2} ");
            // 2.35 2.38 2.46 2.61 2.85 3.21 3.78 4.72 5.33 5.33: flat ground
            // ahead, and nothing within 5.33 units for the last two rays
            #endregion
            // The comment's numbers, formatted as the loop prints them but in the invariant culture, so the check holds on
            // a machine whose decimal separator is a comma.
            var distances = Enumerable.Range(0, 10)
                .Select(i => string.Create(CultureInfo.InvariantCulture, $"{start.Observation.GetSingle(14 + i) * BipedalWalkerEnv.LidarRange:F2}"))
                .ToArray();
            CollectionAssert.AreEqual(new[] { "2.35", "2.38", "2.46", "2.61", "2.85", "3.21", "3.78", "4.72", "5.33", "5.33" }, distances);
        }

        /// <summary>
        ///     Drive the motors by hand: one step with the first hip at full torque and the other motors at none; the hip
        ///     reaches its full speed within the step, and the reward pays for the torque.
        /// </summary>
        [TestMethod]
        public void DriveTheMotors() {
            float hipSpeed = 0;
            double reward = 0;
            #region DriveTheMotors
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("BipedalWalker-v3");
            using var start = env.Reset(seed: 42);
            // the first hip at full torque, forward; the other motors hold nothing
            using NDArray action = np.array(new[] { 1f, 0f, 0f, 0f });
            using var step = env.Step(action);
            (hipSpeed, reward) = (step.Observation.GetSingle(5), step.Reward);
            // 1.00, -0.1611: item 5, the hip's speed in units of its full 4 rad/s,
            // there in one step; the reward pays 0.028 for the torque
            Console.WriteLine($"{hipSpeed:F2}, {reward:F4}");
            #endregion
            Assert.AreEqual(0.9999800324440002f, hipSpeed);
            Assert.AreEqual(-0.16114844381809235, reward);
        }

        /// <summary>
        ///     Save a frame: an <c>rgb_array</c> render as an image file, the reset's frame of seed 42, the walker on the start
        ///     pad by the flag.
        /// </summary>
        [TestMethod]
        public void SaveFrame() {
            #region SaveFrame
            using Env<NDArray, NDArray> env =
                Gymnasium.Make<NDArray, NDArray>("BipedalWalker-v3", RenderMode.RgbArray);
            using var start = env.Reset(seed: 42);
            using RenderResult result = env.Render();
            using Image frame = (Image)result;   // 600x400, Gymnasium's pixels
            frame.SaveAsPng("bipedal_walker.png");
            #endregion
            // The test's working folder is its output folder; the file is removed so runs leave nothing behind.
            Assert.AreEqual(600, frame.Width);
            Assert.AreEqual(400, frame.Height);
            Assert.IsTrue(File.Exists("bipedal_walker.png"));
            File.Delete("bipedal_walker.png");
        }

        /// <summary>
        ///     Many walkers at once: Bipedal Walker has no vector env of its own, so <c>MakeVec</c> steps separate envs in turn
        ///     (<c>sync</c>, its default here), each with its own Box2D world.
        /// </summary>
        [TestMethod]
        public void ManyWalkers() {
            #region ManyWalkers
            // No vector env of its own: MakeVec steps 8 envs in turn ("sync")
            using VectorEnv<NDArray, NDArray> envs =
                Gymnasium.MakeVec<NDArray, NDArray>("BipedalWalker-v3", numEnvs: 8);
            using var start = envs.Reset(seed: 42);
            for (int t = 0; t < 100; t++) {
                using NDArray actions = envs.ActionSpace.Sample();   // 8 x 4 commands
                using var step = envs.Step(actions);   // fallen walkers reset themselves
            }
            #endregion
            Assert.AreEqual(8, envs.NumEnvs);
            Assert.IsInstanceOfType(envs, typeof(SyncVectorEnv<NDArray, NDArray>));
            Assert.AreEqual("Box(-1.0, 1.0, (8, 4), float32)", envs.ActionSpace.ToString());
        }

        /// <summary>
        ///     The Arguments section's calls: kwargs by Gymnasium's names, or the generated options record by C# names; both
        ///     build the hardcore course, as BipedalWalkerHardcore-v3 does.
        /// </summary>
        [TestMethod]
        public void Arguments() {
            #region Arguments
            using IEnv byName = Gymnasium.Make("BipedalWalker-v3", new { hardcore = true });
            using IEnv byRecord = Gymnasium.Make("BipedalWalker-v3",
                new BipedalWalkerEnvOptions { Hardcore = true });
            #endregion
            foreach (IEnv env in new[] { byName, byRecord })
                Assert.IsTrue(((BipedalWalkerEnv)env.Unwrapped).Hardcore);
        }
    }
}
