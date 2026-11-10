using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Gym.Environments;
using Gym.Envs;
using Gym.Observations;
using Gym.Tests.Parity.Frames;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Pair;
using Gym.Tests.Parity.Session;
using Gym.Tests.Parity.Tests.Vector;
using Gym.Utils;
using Gym.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Python.Runtime;
using SixLabors.ImageSharp;

namespace Gym.Tests.Parity.Tests.Envs {
    /// <summary>
    ///     The coverage every registered env gets, whatever its own suite covers: each id through <c>make_vec</c>'s sync
    ///     vector env in two autoreset modes and through Gymnasium's own <c>AsyncVectorEnv</c> with and without shared
    ///     memory, its <c>human</c> frames against pygame's display surface, and its <c>rgb_array_list</c> frame lists; and
    ///     for CartPole-v0, which shares CartPole-v1's class, its own <c>make</c> stack through its 200-step limit.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The ids are every one Gym.NET registers; the suite's coverage matrix (<c>tests/Gym.Tests.Parity/CLAUDE.md</c> §5)
    ///     names the cells each id has. Every id's frames are compared whole, MountainCar's two included since their hill's
    ///     anti-aliased polyline is drawn by pygame's own algorithm (DR-37).
    ///     </para>
    ///     <para>
    ///     Human mode renders into pygame's display under SDL's dummy driver on the Python side and into a recording viewer
    ///     given as <c>viewer_factory</c> on Gym.NET's (DR-20); both pace their frames at <c>render_fps</c>, so the toy_text
    ///     cells (4 frames per second) take fewer steps than the classic ones.
    ///     </para>
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/__init__.py">gymnasium/envs/__init__.py</see>,
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/registration.py">gymnasium/envs/registration.py</see> and
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/rendering.py">gymnasium/wrappers/rendering.py</see>.
    /// </remarks>
    [TestClass]
    public sealed class EnvCoverageLiveParityTests : ParityTestBase {
        /// <summary>Every id Gym.NET registers, in Gymnasium's registration order.</summary>
        private static readonly string[] Ids = {
            "CartPole-v0", "CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0", "Pendulum-v1", "Acrobot-v1",
            "LunarLander-v3", "LunarLanderContinuous-v3",
            "FrozenLake-v1", "FrozenLake8x8-v1", "CliffWalking-v1", "CliffWalkingSlippery-v1", "Taxi-v4", "Blackjack-v1",
        };

        /// <summary>
        ///     Gymnasium's side of <see cref="AssertHumanFrames"/>: the calls made, the display flips and updates counted
        ///     after each (pygame's two functions are wrapped for the run and restored after it), and the display surface
        ///     after each as an <c>(H, W, 3)</c> uint8 array. The cell binds <c>env_id</c>, <c>seed</c> and <c>steps</c>.
        /// </summary>
        private const string HumanFramesStatements = """
            import gymnasium as gym
            import numpy as np
            import pygame
            shown = [0]
            display_flip, display_update = pygame.display.flip, pygame.display.update
            def counted(show):
                def call(*args, **kwargs):
                    shown[0] += 1
                    return show(*args, **kwargs)
                return call
            pygame.display.flip, pygame.display.update = counted(display_flip), counted(display_update)
            calls, counts, frames = [], [], []
            def grab(made):
                calls.append(made)
                counts.append(shown[0])
                frames.append(np.transpose(pygame.surfarray.array3d(pygame.display.get_surface()), (1, 0, 2)).copy())
            try:
                env = gym.make(env_id, render_mode="human")
                env.reset(seed=seed)
                grab(f"reset(seed={seed})")
                env.action_space.seed(seed)
                for i in range(1, steps + 1):
                    _, _, terminated, truncated, _ = env.step(env.action_space.sample())
                    grab(f"step #{i}")
                    if terminated or truncated:
                        env.reset()
                        grab(f"reset() after step #{i}")
                env.close()
            finally:
                pygame.display.flip, pygame.display.update = display_flip, display_update
            result = (tuple(calls), tuple(counts), tuple(frames))
            """;

        /// <summary>
        ///     Gymnasium's side of <see cref="AssertFrameLists"/>: the lists <c>render()</c> returns after 4 and 3 steps and once
        ///     more, each as a tuple of <c>(H, W, 3)</c> uint8 frames. The cell binds <c>env_id</c> and <c>seed</c>.
        /// </summary>
        private const string FrameListsStatements = """
            import gymnasium as gym
            env = gym.make(env_id, render_mode="rgb_array_list")
            env.reset(seed=seed)
            env.action_space.seed(seed)
            lists = []
            for count in (4, 3):
                for _ in range(count):
                    _, _, terminated, truncated, _ = env.step(env.action_space.sample())
                    if terminated or truncated:
                        env.reset()
                lists.append(tuple(env.render()))
            lists.append(tuple(env.render()))
            env.close()
            result = tuple(lists)
            """;

        /// <summary>Every id with each autoreset mode the sync cells run.</summary>
        public static IEnumerable<object[]> IdsAndAutoresetModes => from id in Ids from mode in new[] { "NextStep", "SameStep" } select new object[] { id, mode };

        /// <summary>Every id with shared memory on and off.</summary>
        public static IEnumerable<object[]> IdsAndSharedMemory => from id in Ids from sharedMemory in new[] { true, false } select new object[] { id, sharedMemory };

        /// <summary>Every id, one per row, for the cells that take the id alone; all render pixel for pixel.</summary>
        public static IEnumerable<object[]> EveryId => Ids.Select(id => new object[] { id });

        /// <summary>
        ///     <c>make_vec(id, 3, "sync")</c> in lockstep with <c>Gymnasium.MakeVec</c> for 220 steps of the batched action
        ///     space's seeded samples: every reset and step result whole (episode ends, autoresets, <c>final_obs</c> and
        ///     <c>final_info</c> under <c>SAME_STEP</c>, truncation where an id's limit falls inside the run).
        /// </summary>
        /// <param name="id">The env.</param>
        /// <param name="autoresetMode">The autoreset mode's value.</param>
        /// <exception cref="ParityException">Thrown at the first difference.</exception>
        [TestMethod]
        [DynamicData(nameof(IdsAndAutoresetModes))]
        public void SyncVector_EveryEnv_MatchesGymnasium(string id, string autoresetMode) {
            SkipClassicOffReferencePlatform(id);
            var vectorKwargs = new Dictionary<string, object?> { ["autoreset_mode"] = autoresetMode };
            using PyRef python = PyRef.Of(() => Python.gymnasium.make_vec(id, 3, "sync", vectorKwargs));
            using IVectorEnv dotnet = Gymnasium.MakeVec(id, numEnvs: 3, vectorizationMode: "sync", vectorKwargs: new Info { ["autoreset_mode"] = autoresetMode });
            VectorEnvLiveParityTests.Lockstep($"{id} x3 sync {autoresetMode}", python, dotnet, seed: 11, steps: 220);
        }

        /// <summary>
        ///     Gymnasium's own <c>AsyncVectorEnv</c> (worker processes spawned from the embedded interpreter) in lockstep with
        ///     Gym.NET's for 120 steps, with and without shared memory.
        /// </summary>
        /// <param name="id">The env.</param>
        /// <param name="sharedMemory">Whether the observations go through the shared buffer.</param>
        /// <exception cref="ParityException">Thrown at the first difference.</exception>
        [TestMethod]
        [DynamicData(nameof(IdsAndSharedMemory))]
        public void AsyncVector_EveryEnv_MatchesGymnasium(string id, bool sharedMemory) {
            SkipClassicOffReferencePlatform(id);
            AsyncVectorEnvLiveParityTests.RequireSpawn();
            var vectorKwargs = new Dictionary<string, object?> { ["shared_memory"] = sharedMemory };
            using PyRef python = PyRef.Of(() => Python.gymnasium.make_vec(id, 3, "async", vectorKwargs));
            try {
                using var dotnet = (IVectorEnv)Gymnasium.MakeVec(id, numEnvs: 3, vectorizationMode: "async", vectorKwargs: new Info { ["shared_memory"] = sharedMemory });
                AsyncVectorEnvLiveParityTests.Lockstep($"{id} x3 async (shared_memory={sharedMemory})", python, dotnet, 5, 120, resetEnded: false);
            } finally {
                AsyncVectorEnvLiveParityTests.Close(python);
            }
        }

        /// <summary>
        ///     <c>render_mode="human"</c>: the seeded reset, steps of seeded samples and a reset after each episode end; after
        ///     every call, the number of frames shown so far (pygame's <c>display.flip</c> and <c>display.update</c> calls
        ///     against the viewer's renders) and the frame on display, pixel for pixel.
        /// </summary>
        /// <param name="id">The env.</param>
        /// <exception cref="ParityException">Thrown at the first difference.</exception>
        [TestMethod]
        [DynamicData(nameof(EveryId))]
        public void HumanMode_EveryEnv_FramesMatchPygame(string id) => AssertHumanFrames(id);

        /// <summary>
        ///     <c>render_mode="rgb_array_list"</c> (<c>make</c>'s <c>RenderCollection</c>): the frames collected since the reset
        ///     or the last render, after 4 and then 3 more steps of seeded samples (an episode end resets and restarts the
        ///     list), and the empty list a third render returns.
        /// </summary>
        /// <param name="id">The env.</param>
        /// <exception cref="ParityException">Thrown at the first difference.</exception>
        [TestMethod]
        [DynamicData(nameof(EveryId))]
        public void RgbArrayList_EveryEnv_MatchesGymnasium(string id) => AssertFrameLists(id);

        /// <summary>
        ///     CartPole-v0's <c>make</c> stack against Gym.NET's for a whole episode of the balancing policy: every transition
        ///     identical, both truncating at v0's 200 steps (where v1 runs to 500).
        /// </summary>
        /// <param name="seed">The reset seed.</param>
        /// <exception cref="ParityException">Thrown at the first difference.</exception>
        [TestMethod]
        [DataRow(0L)]
        [DataRow(7L)]
        public void CartPoleV0_MadeStack_TruncatesAt200(long seed) {
            SkipOffReferencePlatform("CartPole's cos/sin");
            using var pair = new GymPair(PyGymEnv.Make("CartPole-v0", viaMake: true), new GymNetEnv(Gymnasium.Make("CartPole-v0")), stateAttribute: "state");
            PairRun run = pair.Run(seed, ActionSources.Policy("cartpole_balance"), maxSteps: 300);
            TestContext.WriteLine(run.ToString());
            Assert.AreEqual(200, run.Steps, "Both sides must truncate at CartPole-v0's max_episode_steps.");
            Assert.IsTrue(run.Ended);
        }

        /// <summary>
        ///     The comparison of <see cref="HumanMode_EveryEnv_FramesMatchPygame"/>: both sides run the same calls with
        ///     <c>render_mode="human"</c>; the calls and the frames shown by the end of each are compared as values, then each
        ///     frame on display pixel for pixel (with PNG artifacts on a difference).
        /// </summary>
        /// <param name="id">The env.</param>
        /// <exception cref="ParityException">Thrown at the first difference.</exception>
        /// <exception cref="AssertFailedException">Thrown when Gymnasium's side raises, or Gym.NET's showed no frame by a grab.</exception>
        private void AssertHumanFrames(string id) {
            SkipClassicOffReferencePlatform(id);
            const long seed = 3;
            // Both sides pace human frames at render_fps: toy_text's 4 frames per second make each call cost a quarter second.
            int steps = IsToyText(id) ? 6 : 20;
            ValueFacts python = Outcome.CapturePython(() => {
                using var envId = new PyString(id);
                using var seedValue = new PyInt(seed);
                using var stepCount = new PyInt(steps);
                return Python.evalStmts(HumanFramesStatements, "result", ("env_id", envId), ("seed", seedValue), ("steps", stepCount));
            }).Expect($"{id}: Gymnasium's human-mode run");

            var viewer = new RecordingViewer();
            var calls = new List<string>();
            var counts = new List<long>();
            var frames = new List<Frame>();
            void Grab(string made) {
                calls.Add(made);
                counts.Add(viewer.Shown);
                frames.Add(viewer.Last ?? throw new AssertFailedException($"{id}: Gym.NET showed no frame by {made}."));
            }

            var kwargs = new Dictionary<string, object?> { ["render_mode"] = "human", ["viewer_factory"] = (IEnvironmentViewerFactoryDelegate)viewer.Factory };
            using (IEnv env = Gymnasium.Make(id, kwargs)) {
                Release(env.Reset(seed));
                Grab($"reset(seed={seed})");
                env.ActionSpace.Seed(seed);
                for (int i = 1; i <= steps; i++) {
                    object? action = env.ActionSpace.Sample();
                    StepResult<object?> step = env.Step(action);
                    // The cell keeps none of these: the sampled action, and the step's observation and info arrays, are released.
                    Outcome.Release(action);
                    Outcome.Release(step.Observation);
                    Outcome.Release(step.Info);
                    Grab($"step #{i}");
                    if (step.Terminated || step.Truncated) {
                        Release(env.Reset());
                        Grab($"reset() after step #{i}");
                    }
                }
            }

            ByteContract.AssertSame($"{id}: the calls made in human mode", python.Items[0], ValueFacts.Of(calls));
            ByteContract.AssertSame($"{id}: the frames shown by the end of each call", python.Items[1], ValueFacts.Of(counts));
            for (int i = 0; i < frames.Count; i++)
                FrameDiff.AssertSame($"{id}: the human frame after {calls[i]}", Frame.FromFacts(python.Items[2].Items[i]), frames[i], "human render");
        }

        /// <summary>
        ///     The comparison of <see cref="RgbArrayList_EveryEnv_MatchesGymnasium"/>: both sides run the same calls with
        ///     <c>render_mode="rgb_array_list"</c>; the number of frames in each returned list is compared as values, then each
        ///     frame pixel for pixel (with PNG artifacts on a difference).
        /// </summary>
        /// <param name="id">The env.</param>
        /// <exception cref="ParityException">Thrown at the first difference.</exception>
        /// <exception cref="AssertFailedException">Thrown when either side raises.</exception>
        private void AssertFrameLists(string id) {
            SkipClassicOffReferencePlatform(id);
            const long seed = 9;
            ValueFacts python = Outcome.CapturePython(() => {
                using var envId = new PyString(id);
                using var seedValue = new PyInt(seed);
                return Python.evalStmts(FrameListsStatements, "result", ("env_id", envId), ("seed", seedValue));
            }).Expect($"{id}: Gymnasium's rgb_array_list run");

            // Described, then released by Capture: the frames are the cell's to drop once their bytes are read.
            ValueFacts dotnet = Outcome.Capture(() => {
                var lists = new List<object?>();
                using IEnv env = Gymnasium.Make(id, new { render_mode = "rgb_array_list" });
                Release(env.Reset(seed));
                env.ActionSpace.Seed(seed);
                foreach (int count in new[] { 4, 3 }) {
                    for (int i = 0; i < count; i++) {
                        object? action = env.ActionSpace.Sample();
                        StepResult<object?> step = env.Step(action);
                        Outcome.Release(action);
                        Outcome.Release(step.Observation);
                        Outcome.Release(step.Info);
                        if (step.Terminated || step.Truncated)
                            Release(env.Reset());
                    }

                    lists.Add(FrameList(env.Render()));
                }

                lists.Add(FrameList(env.Render()));
                return lists.ToArray();
            }).Expect($"{id}: Gym.NET's rgb_array_list run");

            string what = $"{id}: rgb_array_list renders after 4 and 3 steps, then once more";
            ByteContract.AssertSame($"{what}: frames per render", ValueFacts.Of(python.Items.Select(list => (long)list.Items.Count).ToArray()),
                ValueFacts.Of(dotnet.Items.Select(list => (long)list.Items.Count).ToArray()));
            for (int render = 0; render < python.Items.Count; render++) {
                for (int frame = 0; frame < python.Items[render].Items.Count; frame++) {
                    FrameDiff.AssertSame($"{what}: render #{render + 1}, frame #{frame + 1}", Frame.FromFacts(python.Items[render].Items[frame]),
                        Frame.FromFacts(dotnet.Items[render].Items[frame]), "rgb_array_list frame");
                }
            }
        }

        /// <summary>
        ///     An <c>rgb_array_list</c> render as Gymnasium's list of frames: the arrays of its <see cref="RenderResult.Frames"/>
        ///     items, which whoever releases the list releases with it.
        /// </summary>
        /// <param name="rendered">The render.</param>
        /// <returns>The frames, oldest first.</returns>
        /// <exception cref="InvalidCastException">Thrown when the render is not a list of RGB frames, which the cell then reports as Gym.NET's error.</exception>
        private static object?[] FrameList(RenderResult rendered) =>
            ((RenderResult.Frames)rendered).Items.Select(frame => (object?)((RenderResult.Rgb)frame).Frame).ToArray();

        /// <summary>Releases what a reset returned and the cell drops: its observation's and info's arrays.</summary>
        /// <param name="reset">The reset.</param>
        private static void Release(ResetResult<object?> reset) {
            Outcome.Release(reset.Observation);
            Outcome.Release(reset.Info);
        }

        /// <summary>Pins a classic-control or box2d id's cells to the reference platform (their cos/sin, and Box2D's sinf/cosf, come from the C runtime).</summary>
        /// <param name="id">The env.</param>
        private void SkipClassicOffReferencePlatform(string id) {
            if (IsToyText(id))
                return;
            SkipOffReferencePlatform(id.StartsWith("LunarLander", StringComparison.Ordinal) ? "Box2D's sinf/cosf" : "the classic envs' cos/sin");
        }

        /// <summary>Whether an id is one of toy_text's: no transcendental math, and 4 frames per second in human mode.</summary>
        /// <param name="id">The env.</param>
        /// <returns><see langword="true"/> for FrozenLake, CliffWalking, Taxi and Blackjack.</returns>
        private static bool IsToyText(string id) =>
            id.StartsWith("FrozenLake", StringComparison.Ordinal) || id.StartsWith("CliffWalking", StringComparison.Ordinal) || id.StartsWith("Taxi", StringComparison.Ordinal)
            || id.StartsWith("Blackjack", StringComparison.Ordinal);

        /// <summary>
        ///     A viewer that keeps the last frame it was shown and counts the frames, so a cell can compare what a human-mode
        ///     env displays with pygame's display surface (Gym.NET's counterpart of reading that surface).
        /// </summary>
        private sealed class RecordingViewer : IEnvViewer {
            /// <summary>How many frames were shown, the counterpart of pygame's display flips and updates.</summary>
            public long Shown { get; private set; }

            /// <summary>The last frame shown, or <see langword="null"/> before the first.</summary>
            public Frame? Last { get; private set; }

            /// <summary>The factory the env is given as <c>viewer_factory</c>: hands out this viewer for any size.</summary>
            /// <param name="width">The window's width; the frames carry their own size, which the comparison checks.</param>
            /// <param name="height">The window's height.</param>
            /// <param name="title">The window's title; not compared (pygame's caption is cosmetic).</param>
            /// <returns>This viewer, already completed.</returns>
            public Task<IEnvViewer> Factory(int width, int height, string? title = null) => Task.FromResult<IEnvViewer>(this);

            /// <summary>Keeps a copy of the frame (the env releases its image right after) and counts it.</summary>
            /// <param name="img">The frame.</param>
            public void Render(Image img) {
                Last = Frame.FromImage(img);
                Shown++;
            }

            /// <summary>Does nothing: there is no window to close.</summary>
            public void CloseEnvironment() { }

            /// <summary>Does nothing: the viewer holds no resources.</summary>
            public void Dispose() { }
        }
    }
}
