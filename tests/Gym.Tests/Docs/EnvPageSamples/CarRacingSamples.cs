using Gym;
using Gym.Envs;
using NumSharp;

// Below the four lines the quick start shows: the reset's options are a Dictionary<string, object?>, which the test
// project's default (no nullable context) would flag.
#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Gym.Box2D;
using Gym.Environments.Envs.Box2D;
using Gym.Observations;
using Gym.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SixLabors.ImageSharp;

// Not under Gym.Tests: there the namespace Gym.Tests.Gymnasium (DR-06) shadows the Gymnasium facade, and the samples
// must read Gymnasium.Make(...) exactly as a reader's code does.
namespace Gym.Docs.EnvPageSamples {
    /// <summary>
    ///     The C# samples of Car Racing's page on the docs site (<c>docs/environments/box2d/car_racing.md</c>), each the
    ///     body of a test, so the unit gate compiles and runs every line the page shows (DR-105). The page includes them
    ///     by DocFX's code-snippet syntax: a region by name, or for the quick start this file's first four lines (its
    ///     usings and a blank) and the quick start's region, by line range.
    /// </summary>
    /// <remarks>
    ///     Footgun: the quick start's include names line numbers (<c>?range=1-4,…</c>), so the first four lines of this file
    ///     must stay <c>using Gym; using Gym.Envs; using NumSharp;</c> and a blank line, and moving the quick start's lines
    ///     moves the page's range; <c>EnvDocPagesTests</c> checks both, so the gate fails before the page shows the wrong
    ///     lines. Each test asserts after its region only, so the assertions never appear on the page; where a region's
    ///     comment states a number (a return, a speed, a colour), the assertion pins it: those numbers are Gymnasium
    ///     1.4.0's own for the same seeds and actions, so the page cannot drift from what either computes. Car Racing
    ///     draws a 1000×800 frame every step, the slowest of the Box2D envs, so the samples keep their episodes short but
    ///     for the pure pursuit's.
    /// </remarks>
    [TestClass]
    public class CarRacingSamples {
        /// <summary>The page's quick start: make the env, reset it with a seed, take one random step.</summary>
        [TestMethod]
        public void QuickStart() {
            #region QuickStart
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("CarRacing-v3");
            // a 96x96 RGB picture of the car, the track and the indicators
            (NDArray observation, Info info) = env.Reset(seed: 42);
            // steer from -1 (left) to 1 (right), gas and brake from 0 to 1
            using NDArray action = env.ActionSpace.Sample();
            observation.Dispose();
            (observation, double reward, bool terminated, bool truncated, info)
                = env.Step(action);
            #endregion
            // The quick start's first observation, read again: the same seeded reset gives the same one.
            using ResetResult<NDArray> again = env.Reset(seed: 42);
            NDArray first = again.Observation;
            CollectionAssert.AreEqual(new long[] { 96, 96, 3 }, first.shape.Select(d => (long)d).ToArray());
            Assert.AreEqual(NPTypeCode.Byte, first.typecode);
            // Gymnasium's reset of seed 42: the camera starts far above and zooms in over the first second, so the first
            // picture is the whole playfield, small, on black: its mean is 32.7.
            Assert.AreEqual(32.72562210648148, first.astype(NPTypeCode.Double).mean().GetDouble(), 1e-12);
            Assert.AreEqual(0, info.Count);
            _ = (reward, terminated, truncated);
            observation.Dispose();
        }

        /// <summary>Run an episode: random driving until the 1,000-step limit or the car leaves the playfield.</summary>
        [TestMethod]
        public void RunAnEpisode() {
            int steps = 0;
            #region RunAnEpisode
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("CarRacing-v3");
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
            Console.WriteLine($"return {episodeReturn:F1}");   // random driving: -31.8 on average
            #endregion
            Assert.IsTrue(steps <= 1000);
        }

        /// <summary>
        ///     Drive from the track's geometry: pure pursuit of the centre-line point five tiles past the nearest one, the
        ///     page's video's agent written out in double arithmetic (the example reads the speed in float32), from seed 11.
        /// </summary>
        [TestMethod]
        public void PurePursuit() {
            double episodeReturn = 0;
            #region PurePursuit
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("CarRacing-v3");
            var race = (CarRacingEnv)env.Unwrapped;   // the track and the car
            using ResetResult<NDArray> start = env.Reset(seed: 11);
            while (true) {
                b2Body hull = race.Car!.Hull!;
                double px = hull.Position.x, py = hull.Position.y, angle = hull.Angle;
                IReadOnlyList<(double Alpha, double Beta, double X, double Y)> track = race.Track;
                // aim at the centre-line point five tiles past the nearest one
                int nearest = Enumerable.Range(0, track.Count).MinBy(i =>
                    (track[i].X - px) * (track[i].X - px) + (track[i].Y - py) * (track[i].Y - py));
                (_, _, double tx, double ty) = track[(nearest + 5) % track.Count];
                // the car's nose is its local +y: the angle from it to the target
                double error
                    = Math.Atan2(-Math.Sin(angle) * (ty - py) - Math.Cos(angle) * (tx - px),
                                 -Math.Sin(angle) * (tx - px) + Math.Cos(angle) * (ty - py));
                double vx = hull.LinearVelocity.x, vy = hull.LinearVelocity.y;
                double speed = Math.Sqrt(vx * vx + vy * vy);
                using NDArray action = np.array(new[] {
                    (float)Math.Clamp(-2.5 * error, -1, 1),                        // steer
                    speed < 45 - 30 * Math.Min(1, Math.Abs(error)) ? 0.6f : 0f,    // gas
                    Math.Abs(error) > 0.6 && speed > 30 ? 0.4f : 0f,               // brake
                });
                using StepResult<NDArray> step = env.Step(action);
                episodeReturn += step.Reward;
                if (step.Terminated || step.Truncated)
                    break;
            }
            Console.WriteLine($"return {episodeReturn:F1}");   // 880.7: 254 of 259 tiles in 1,000 steps
            #endregion
            Assert.AreEqual(880.6949806949622, episodeReturn);
        }

        /// <summary>
        ///     Discrete actions: <c>continuous = false</c> gives <c>Discrete(5)</c>; fifty steps of gas (action 3, 0.2 of the
        ///     pedal) from seed 42.
        /// </summary>
        [TestMethod]
        public void DiscreteActions() {
            double speed = 0, episodeReturn = 0;
            long tiles = 0;
            #region DiscreteActions
            // 0 nothing, 1 steer right, 2 steer left, 3 gas, 4 brake
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("CarRacing-v3",
                new CarRacingEnvOptions { Continuous = false });
            using ResetResult<NDArray> start = env.Reset(seed: 42);
            for (int t = 0; t < 50; t++) {
                using NDArray gas = np.array(3L);
                using StepResult<NDArray> step = env.Step(gas);
                episodeReturn += step.Reward;
            }
            var race = (CarRacingEnv)env.Unwrapped;
            b2Vec2 v = race.Car!.Hull!.LinearVelocity;
            (speed, tiles) = (Math.Sqrt((double)v.x * v.x + (double)v.y * v.y), race.TileVisitedCount);
            // speed 28.9, 7 tiles, return 19.7: a discrete gas is 0.2 of the pedal
            Console.WriteLine($"speed {speed:F1}, {tiles} tiles, return {episodeReturn:F1}");
            #endregion
            Assert.AreEqual(28.92299974319275, speed, 1e-12);
            Assert.AreEqual(7, tiles);
            Assert.AreEqual(19.734982332155475, episodeReturn);
        }

        /// <summary>
        ///     Domain randomization: random road, background and grass colours, drawn at each reset unless the reset's
        ///     <c>randomize</c> option says to keep them.
        /// </summary>
        [TestMethod]
        public void DomainRandomization() {
            var printed = new List<string>();
            #region DomainRandomization
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("CarRacing-v3",
                new { domain_randomize = true });
            var race = (CarRacingEnv)env.Unwrapped;
            string Colours() => string.Join(" ", new[] { race.RoadColor, race.BgColor, race.GrassColor }
                .Select(c => string.Join(",", c.ToArray<double>().Select(v => v.ToString("F0", CultureInfo.InvariantCulture)))));
            using ResetResult<NDArray> first
                = env.Reset(seed: 42);   // new colours at every reset
            Console.WriteLine(Colours());   // road 163,92,180 background 146,20,205 grass 146,20,225
            using ResetResult<NDArray> kept
                = env.Reset(options: new Dictionary<string, object?> { ["randomize"] = false });
            Console.WriteLine(Colours());   // the same three
            #endregion
            printed.Add(Colours());
            Assert.AreEqual("163,92,180 146,20,205 146,20,225", printed[0]);
        }

        /// <summary>
        ///     Save a frame: an <c>rgb_array</c> render as an image file, 600×400 with the indicators and the score, after a
        ///     second of driving straight, when the camera has finished zooming in.
        /// </summary>
        [TestMethod]
        public void SaveFrame() {
            #region SaveFrame
            using Env<NDArray, NDArray> env
                = Gymnasium.Make<NDArray, NDArray>("CarRacing-v3", RenderMode.RgbArray);
            using ResetResult<NDArray> start = env.Reset(seed: 42);
            for (int t = 0; t < 50; t++) {   // the camera zooms in over the first second
                using NDArray gas = np.array(new[] { 0f, 0.3f, 0f });
                using StepResult<NDArray> step = env.Step(gas);
            }
            using RenderResult result = env.Render();
            using Image frame = (Image)result;   // 600x400, Gymnasium's pixels
            frame.SaveAsPng("car_racing.png");
            #endregion
            // The test's working folder is its output folder; the file is removed so runs leave nothing behind.
            Assert.AreEqual(600, frame.Width);
            Assert.AreEqual(400, frame.Height);
            Assert.IsTrue(File.Exists("car_racing.png"));
            File.Delete("car_racing.png");
        }

        /// <summary>
        ///     Many cars at once: Car Racing has no vector env of its own, so <c>MakeVec</c> steps separate envs in turn
        ///     (<c>sync</c>, its default here), each with its own world and frame.
        /// </summary>
        [TestMethod]
        public void ManyCars() {
            #region ManyCars
            // No vector env of its own: MakeVec steps 4 envs in turn ("sync")
            using VectorEnv<NDArray, NDArray> envs
                = Gymnasium.MakeVec<NDArray, NDArray>("CarRacing-v3", numEnvs: 4);
            using VectorResetResult<NDArray> start
                = envs.Reset(seed: 42);   // 4 x 96 x 96 x 3 pictures
            (NDArray observations, Info infos) = start;
            for (int t = 0; t < 100; t++) {
                using NDArray actions = envs.ActionSpace.Sample();   // 4 x 3
                using VectorStepResult<NDArray> step = envs.Step(actions);
            }
            #endregion
            Assert.AreEqual(4, envs.NumEnvs);
            Assert.IsInstanceOfType(envs, typeof(SyncVectorEnv<NDArray, NDArray>));
            CollectionAssert.AreEqual(new long[] { 4, 96, 96, 3 }, observations.shape.Select(d => (long)d).ToArray());
        }

        /// <summary>
        ///     The Arguments section's calls: kwargs by Gymnasium's names, or the generated options record by C# names; both
        ///     build the same env, here with discrete actions and a lap that counts at 90% of the tiles.
        /// </summary>
        [TestMethod]
        public void Arguments() {
            #region Arguments
            using IEnv byName = Gymnasium.Make("CarRacing-v3",
                new { continuous = false, lap_complete_percent = 0.9 });
            using IEnv byRecord = Gymnasium.Make("CarRacing-v3",
                new CarRacingEnvOptions { Continuous = false, LapCompletePercent = 0.9 });
            #endregion
            foreach (IEnv env in new[] { byName, byRecord }) {
                var race = (CarRacingEnv)env.Unwrapped;
                Assert.IsFalse(race.Continuous);
                Assert.AreEqual(0.9, race.LapCompletePercent);
            }
        }
    }
}
