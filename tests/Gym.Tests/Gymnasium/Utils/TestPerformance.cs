#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Gym.Envs;
using Gym.Observations;
using Gym.Spaces;
using Gym.Utils;
using Gym.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Utils {
    /// <summary>Gymnasium's benchmark tests (<c>tests/utils/test_performance.py</c>), run against <see cref="Performance"/>.</summary>
    /// <remarks>
    ///     Gymnasium scripts <c>time.time</c> with <c>monkeypatch</c> and records calls with <c>unittest.mock.Mock</c>; the
    ///     port scripts <see cref="Performance.Time"/> (restored after each test) and records calls with
    ///     <see cref="RecordingEnv"/>, a hand-written stand-in for the mock env.
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_performance.py">tests/utils/test_performance.py</see>.
    /// </remarks>
    [TestClass]
    [DoNotParallelize]
    public class TestPerformance {
        /// <summary>The steps per second of two one-step CartPoles over one scripted second, under each autoreset mode.</summary>
        /// <param name="autoresetMode">The autoreset mode's value.</param>
        /// <param name="expectedStepsPerSecond">2 under <c>NEXT_STEP</c> (the autoreset steps do not count), else 4.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_performance.py#L9">tests/utils/test_performance.py (test_benchmark_vector_step)</see>.</remarks>
        [TestMethod]
        [DataRow("NextStep", 2.0)]
        [DataRow("SameStep", 4.0)]
        [DataRow("Disabled", 4.0)]
        public void TestBenchmarkVectorStep(string autoresetMode, double expectedStepsPerSecond) {
            using var env = new SyncVectorEnv(Enumerable.Range(0, 2).Select(_ => (Func<IEnv>)(() => Make("CartPole-v1", maxEpisodeSteps: 1))), autoresetMode: autoresetMode);
            using (ScriptTime(0.0, 0.0, 1.0)) {
                try {
                    Assert.AreEqual(expectedStepsPerSecond, Performance.BenchmarkVectorStep(env, targetDuration: 0, seed: 123));
                } finally {
                    env.Close();
                }
            }
        }

        /// <summary>A step that ends the episode makes the benchmark reset (unseeded), and every step samples an action.</summary>
        /// <param name="terminated">Whether the second step terminates.</param>
        /// <param name="truncated">Whether it truncates.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_performance.py#L35">tests/utils/test_performance.py (test_benchmark_step_resets_finished_env)</see>.</remarks>
        [TestMethod]
        [DataRow(true, false)]
        [DataRow(false, true)]
        public void TestBenchmarkStepResetsFinishedEnv(bool terminated, bool truncated) {
            var env = new RecordingEnv(new[] { (false, false), (terminated, truncated) });
            using (ScriptTime(0.0, 0.0, 1.0, 1.0))
                Assert.AreEqual(2.0, Performance.BenchmarkStep(env, targetDuration: 0, seed: 123));
            CollectionAssert.AreEqual(new BigInteger?[] { 123, null }, env.ResetSeeds);
            Assert.AreEqual(3, env.CountingActionSpace.Samples);
        }

        /// <summary>Two constructions in one scripted second, each reset once with the seed.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_performance.py#L50">tests/utils/test_performance.py (test_benchmark_init)</see>.</remarks>
        [TestMethod]
        public void TestBenchmarkInit() {
            var envs = new List<RecordingEnv>();
            using (ScriptTime(0.0, 0.0, 1.0, 1.0)) {
                Assert.AreEqual(2.0, Performance.BenchmarkInit(() => {
                    var env = new RecordingEnv(Array.Empty<(bool, bool)>());
                    envs.Add(env);
                    return env;
                }, targetDuration: 0, seed: 123));
            }

            Assert.AreEqual(2, envs.Count);
            foreach (RecordingEnv env in envs)
                CollectionAssert.AreEqual(new BigInteger?[] { 123 }, env.ResetSeeds);
        }

        /// <summary>Two renders in one scripted second.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_performance.py#L60">tests/utils/test_performance.py (test_benchmark_render)</see>.</remarks>
        [TestMethod]
        public void TestBenchmarkRender() {
            var env = new RecordingEnv(Array.Empty<(bool, bool)>());
            using (ScriptTime(0.0, 0.0, 1.0, 1.0))
                Assert.AreEqual(2.0, Performance.BenchmarkRender(env, targetDuration: 0));
            Assert.AreEqual(2, env.Renders);
        }

        /// <summary>Scripts <see cref="Performance.Time"/> with the timestamps (1.0 once they run out, as the test's <c>next(timestamps, 1.0)</c>).</summary>
        /// <param name="timestamps">The timestamps, in order.</param>
        /// <returns>A scope that restores the clock.</returns>
        private static IDisposable ScriptTime(params double[] timestamps) {
            Func<double> previous = Performance.Time;
            var queue = new Queue<double>(timestamps);
            Performance.Time = () => queue.Count > 0 ? queue.Dequeue() : 1.0;
            return new Restore(() => Performance.Time = previous);
        }

        /// <summary>Runs an action when disposed.</summary>
        private sealed class Restore : IDisposable {
            /// <summary>The action.</summary>
            private readonly Action _action;

            /// <summary>A scope that runs <paramref name="action"/> when disposed.</summary>
            /// <param name="action">The action.</param>
            public Restore(Action action) => _action = action;

            /// <summary>Runs the action.</summary>
            public void Dispose() => _action();
        }

        /// <summary>A space whose samples count themselves, the stand-in for the mock's <c>action_space.sample</c>.</summary>
        private sealed class CountingSpace : Space<object?> {
            /// <summary>A space with no shape or dtype.</summary>
            public CountingSpace() : base(null, null, null) { }

            /// <summary>How many samples were drawn.</summary>
            public int Samples { get; private set; }

            /// <summary>Not flattenable.</summary>
            public override bool IsNpFlattenable => false;

            /// <summary>Counts the call and returns <see langword="null"/>.</summary>
            /// <param name="mask">Ignored.</param>
            /// <param name="probability">Ignored.</param>
            /// <returns><see langword="null"/>.</returns>
            public override object? Sample(object? mask = null, object? probability = null) {
                Samples++;
                return null;
            }

            /// <summary>Contains anything.</summary>
            /// <param name="x">Ignored.</param>
            /// <returns><see langword="true"/>.</returns>
            public override bool Contains(object? x) => true;
        }

        /// <summary>The stand-in for the test's <c>Mock()</c> env: scripted step flags, and the resets' seeds and the renders recorded.</summary>
        private sealed class RecordingEnv : Env<object?, object?> {
            /// <summary>The flags of the steps, in order.</summary>
            private readonly Queue<(bool Terminated, bool Truncated)> _steps;

            /// <summary>An env whose steps end as scripted.</summary>
            /// <param name="steps">Each step's <c>(terminated, truncated)</c>.</param>
            public RecordingEnv(IEnumerable<(bool, bool)> steps) {
                _steps = new Queue<(bool, bool)>(steps);
                CountingActionSpace = new CountingSpace();
                ActionSpace = CountingActionSpace;
                ObservationSpace = new CountingSpace();
            }

            /// <summary>The action space, counting its samples.</summary>
            public CountingSpace CountingActionSpace { get; }

            /// <summary>The seed of every reset, in order.</summary>
            public List<BigInteger?> ResetSeeds { get; } = new();

            /// <summary>How many renders were asked for.</summary>
            public int Renders { get; private set; }

            /// <summary>Records the seed.</summary>
            /// <param name="seed">The seed.</param>
            /// <param name="options">Ignored.</param>
            /// <returns><see langword="null"/> and an empty info.</returns>
            public override ResetResult<object?> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
                ResetSeeds.Add(seed);
                return new ResetResult<object?>(null, new Info());
            }

            /// <summary>Returns the next scripted flags.</summary>
            /// <param name="action">Ignored.</param>
            /// <returns><c>(None, 0.0, terminated, truncated, {})</c>.</returns>
            public override StepResult<object?> Step(object? action) {
                (bool terminated, bool truncated) = _steps.Dequeue();
                return new StepResult<object?>(null, 0.0, terminated, truncated, new Info());
            }

            /// <summary>Counts the render.</summary>
            /// <returns>No frame.</returns>
            public override RenderResult Render() {
                Renders++;
                return RenderResult.None;
            }
        }
    }
}
