#nullable enable
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Gym.Environments;
using Gym.Environments.Envs.Box2D;
using Gym.Envs;
using Gym.Observations;
using Gym.Rendering.Avalonia;
using Gym.Rendering.WinForm;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Envs.Box2D {
    /// <summary>
    ///     <see cref="LunarLanderEnv"/> landed by Gymnasium's heuristic (<see cref="LunarLanderEnv.Heuristic"/>) in every viewer:
    ///     human mode into the WinForms and Avalonia windows (two at once in Avalonia) and into the headless viewer, and
    ///     without a render mode against Gymnasium's own totals, which pin the whole episode.
    /// </summary>
    /// <remarks>
    ///     The tests named <c>*WinFormEnv</c> and <c>*AvaloniaEnv</c> open windows; gated runs exclude them by name. Human mode
    ///     paces its frames at 50 per second, so a landing there takes a few seconds; the baselines run without a render mode.
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/lunar_lander.py#L805">gymnasium/envs/box2d/lunar_lander.py</see>.
    /// </remarks>
    [TestClass]
    public class LunarLanderEnvironment {
        /// <summary>
        ///     Gymnasium's heuristic landings, recorded from <c>refs/Gymnasium</c> on the parity stack (2026-11-11): per seed and
        ///     action mode, the steps until the lander rests (+100) and the float64 bits of the summed rewards. Gym.NET's env
        ///     must reproduce both exactly, on the reference platform (Box2D's rotations and the env's trigonometry call the C
        ///     runtime there).
        /// </summary>
        private static readonly Dictionary<(int Seed, bool Continuous), (int Steps, ulong TotalBits)> GymnasiumBaselines = new() {
            [(0, false)] = (152, 0x407295a68486e799ul),    // 297.35315373131886
            [(1, false)] = (205, 0x40704f190f2a0d72ul),    // 260.9436179773976
            [(42, false)] = (235, 0x406fffc19143ac85ul),   // 255.9923788377529
            [(1000, false)] = (424, 0x4071d93606abe00eul), // 285.57568995608096
            [(0, true)] = (200, 0x40731f6859dae7a6ul),     // 305.9629763174984
            [(1, true)] = (190, 0x40711e45443ba730ul),     // 273.89191077520354
            [(42, true)] = (203, 0x40712554f460d7beul),    // 274.33324086980053
            [(1000, true)] = (230, 0x4072fd37138db055ul),  // 303.8259463820189
        };

        /// <summary>The most steps an episode may take before the run fails; Gymnasium's landings above take at most 424.</summary>
        private const int MaxSteps = 5000;

        /// <summary>
        ///     Lands one episode with the heuristic and, for a seed in <see cref="GymnasiumBaselines"/> without a render mode,
        ///     asserts Gymnasium's step count and total exactly.
        /// </summary>
        /// <param name="renderMode"><c>"human"</c> to render every reset and step into <paramref name="factory"/>'s viewer, or <see langword="null"/>.</param>
        /// <param name="factory">The viewer factory of human mode; unused without a render mode.</param>
        /// <param name="seed">The reset seed.</param>
        /// <param name="continuous">Selects the continuous variant.</param>
        /// <param name="assertBaseline">Whether to assert the recorded baseline (only meaningful for the seeds it lists).</param>
        /// <exception cref="AssertFailedException">Thrown when the episode does not end within <see cref="MaxSteps"/> or misses its baseline.</exception>
        private static void Run(string? renderMode, IEnvironmentViewerFactoryDelegate? factory, int seed, bool continuous, bool assertBaseline) {
            using var env = new LunarLanderEnv(renderMode: renderMode, continuous: continuous, viewerFactory: factory);
            try {
                double total = 0;
                int steps = 0;
                NDArray s = env.Reset(seed).Observation;
                try {
                    while (true) {
                        StepResult<NDArray> result;
                        using (NDArray a = LunarLanderEnv.Heuristic(env, s))
                            result = env.Step(a);
                        s.Dispose();
                        s = result.Observation;
                        total += result.Reward;
                        steps++;
                        if (result.Terminated || result.Truncated || steps > MaxSteps)
                            break;
                    }
                } finally {
                    s.Dispose();
                }

                Assert.IsTrue(steps <= MaxSteps, "The heuristic must land the lander.");
                if (assertBaseline) {
                    (int expectedSteps, ulong expectedBits) = GymnasiumBaselines[(seed, continuous)];
                    Assert.AreEqual(expectedSteps, steps, $"Gymnasium's heuristic lands seed {seed} (continuous={continuous}) in {expectedSteps} steps.");
                    ulong bits = unchecked((ulong)BitConverter.DoubleToInt64Bits(total));
                    Assert.AreEqual(expectedBits, bits, $"Gymnasium's total for seed {seed} (continuous={continuous}) is " +
                                                        $"{BitConverter.Int64BitsToDouble(unchecked((long)expectedBits)):R}; Gym.NET's is {total:R}.");
                }
            } finally {
                env.Close();
            }
        }

        /// <summary>Shuts the shared Avalonia app down after each test, so the next one starts its own.</summary>
        [TestCleanup]
        public void Cleanup() => StaticAvaloniaApp.Shutdown();

        /// <summary>A discrete landing in human mode, rendered into a WinForms window.</summary>
        [TestMethod]
        public void Run_Discrete_WinFormEnv() => Run("human", WinFormEnvViewer.Factory, 0, false, assertBaseline: false);

        /// <summary>A discrete landing in human mode, rendered into an Avalonia window.</summary>
        [TestMethod]
        public void Run_Discrete_AvaloniaEnv() => Run("human", AvaloniaEnvViewer.Factory, 0, false, assertBaseline: false);

        /// <summary>A discrete landing in human mode, rendered into the headless viewer.</summary>
        [TestMethod]
        public void Run_Discrete_NullEnv() => Run("human", NullEnvViewer.Factory, 0, false, assertBaseline: false);

        /// <summary>A continuous landing in human mode, rendered into a WinForms window.</summary>
        [TestMethod]
        public void Run_Continuous_WinFormEnv() => Run("human", WinFormEnvViewer.Factory, 0, true, assertBaseline: false);

        /// <summary>A continuous landing in human mode, rendered into an Avalonia window.</summary>
        [TestMethod]
        public void Run_Continuous_AvaloniaEnv() => Run("human", AvaloniaEnvViewer.Factory, 0, true, assertBaseline: false);

        /// <summary>Two continuous landings at once on pool threads, each in its own Avalonia window of the shared app.</summary>
        /// <returns>The task of both runs.</returns>
        [TestMethod]
        public async Task Run_TwoInstances_Continuous_AvaloniaEnv() {
            Task first = Task.Run(() => Run("human", AvaloniaEnvViewer.Factory, 0, true, assertBaseline: false));
            Task second = Task.Run(() => Run("human", AvaloniaEnvViewer.Factory, 1, true, assertBaseline: false));
            await Task.WhenAll(first, second);
        }

        /// <summary>A continuous landing in human mode, rendered into the headless viewer.</summary>
        [TestMethod]
        public void Run_Continuous_NullEnv() => Run("human", NullEnvViewer.Factory, 0, true, assertBaseline: false);

        /// <summary>
        ///     Rendering does not change the landing: in human mode the engines also emit particles, bodies of the world that
        ///     never touch the lander, and the episode still ends at Gymnasium's step with Gymnasium's total (seed 1000, discrete).
        /// </summary>
        [TestMethod]
        public void Run_Discrete_NullEnv_ConsistencyCheck() => Run("human", NullEnvViewer.Factory, 1000, false, assertBaseline: true);

        /// <summary>
        ///     Pins all eight of Gymnasium's heuristic landings (four seeds, both action modes) without a render mode: any change
        ///     to a draw, its order, the physics or the reward arithmetic fails here, without Python.
        /// </summary>
        /// <param name="seed">The reset seed.</param>
        /// <param name="continuous">Selects the continuous variant.</param>
        [TestMethod]
        [DataRow(0, false)]
        [DataRow(1, false)]
        [DataRow(42, false)]
        [DataRow(1000, false)]
        [DataRow(0, true)]
        [DataRow(1, true)]
        [DataRow(42, true)]
        [DataRow(1000, true)]
        public void Run_Baseline_MatchesGymnasium(int seed, bool continuous) => Run(null, null, seed, continuous, assertBaseline: true);
    }
}
