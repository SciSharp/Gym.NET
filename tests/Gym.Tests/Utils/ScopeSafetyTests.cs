#nullable enable
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Gym.Environments;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Spaces;
using Gym.Utils;
using Gym.Vector;
using Gym.Wrappers;
using Gym.Wrappers.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Utils {
    /// <summary>
    ///     Pins DR-96: Gym.NET's envs, wrappers, vector envs and renderers stay correct when the caller steps them inside
    ///     NumSharp scopes (an <c>[NDScoped]</c> method, or <see cref="NDScope.Open"/> by hand), the way the guides show
    ///     releasing what an env returns (DR-80).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A scope disposes, when it closes, every array constructed on its thread while it was open and not returned. A class
    ///     that keeps such an array past the call that built it (running moments, a frame stack, collected frames) must detach
    ///     it where it stores it; one that does not reads freed memory at its next step: wrong values in silence, or an
    ///     access violation.
    ///     </para>
    ///     <para>
    ///     Each case runs the same seeded stream twice, once with no scope and once in chunks, each chunk inside its own scope,
    ///     so the env's state outlives every scope; the two runs must match exactly (every observation's bits, rewards, flags,
    ///     infos, frames). After each scope closes, the env's object graph is walked: any array it still references that the
    ///     scope disposed fails the case at once, even when the freed memory still held the right values (a read that is
    ///     correct only by luck).
    ///     </para>
    ///     <para>
    ///     An env or space BUILT inside a scope and kept after it is outside this contract (DR-96): the guides say to build
    ///     them outside the scoped method, and NumSharp.Build refuses an <c>[NDScoped]</c> method returning one (NDW003).
    ///     </para>
    /// </remarks>
    [TestClass]
    public class ScopeSafetyTests {
        /// <summary>Steps per chunk: one scope each in the scoped run.</summary>
        private const int Chunk = 25;

        /// <summary>Chunks per run: three scope boundaries, each crossed with the queues, stacks and statistics mid-stream.</summary>
        private const int Chunks = 4;

        /// <summary>How deep the walk of an env's object graph goes; Box2D's bodies and the wrapper chains stay well inside it.</summary>
        private const int MaxDepth = 40;

        /// <summary>The MuJoCo envs render at this size here: scope safety does not depend on the frame's size, the run time does.</summary>
        private static readonly object MujocoFrameSize = new { width = 64, height = 48 };

        /// <summary>Every id Gym.NET.Environments registers, but Pusher-v4, which refuses at construction as Gymnasium's does on MuJoCo 3.</summary>
        private static readonly string[] BuiltInIds = {
            "CartPole-v0", "CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0", "Pendulum-v1", "Acrobot-v1",
            "Blackjack-v1", "FrozenLake-v1", "FrozenLake8x8-v1", "CliffWalking-v1", "CliffWalkingSlippery-v1", "Taxi-v4",
            "LunarLander-v3", "LunarLanderContinuous-v3", "BipedalWalker-v3", "BipedalWalkerHardcore-v3", "CarRacing-v3",
            "Ant-v4", "Ant-v5", "HalfCheetah-v4", "HalfCheetah-v5", "Hopper-v4", "Hopper-v5", "Humanoid-v4", "Humanoid-v5",
            "HumanoidStandup-v4", "HumanoidStandup-v5", "InvertedDoublePendulum-v4", "InvertedDoublePendulum-v5",
            "InvertedPendulum-v4", "InvertedPendulum-v5", "Pusher-v5", "Reacher-v4", "Reacher-v5", "Swimmer-v4", "Swimmer-v5",
            "Walker2d-v4", "Walker2d-v5",
        };

        /// <summary>The ids, one row each.</summary>
        public static IEnumerable<object[]> Ids => BuiltInIds.Select(id => new object[] { id });

        /// <summary>
        ///     The MuJoCo id whose rendering runs here: the 22 models render through one renderer, whose software frame costs a
        ///     tenth of a second to more than a second at any size (its shadow maps), so one model stands for them all.
        /// </summary>
        private const string RenderedMujocoId = "HalfCheetah-v5";

        /// <summary>
        ///     Every id with every render mode it lists but <c>human</c>, and <c>rgb_array_list</c> (make's RenderCollection, whose
        ///     frames are kept across steps) for every id that renders <c>rgb_array</c>; of the MuJoCo ids, only
        ///     <see cref="RenderedMujocoId"/>'s three modes (RenderCollection's frames do not depend on the env, and a frame per
        ///     step of a MuJoCo scene would take most of a minute).
        /// </summary>
        public static IEnumerable<object[]> IdsAndRenderModes {
            get {
                foreach (string id in BuiltInIds.Where(id => !IsMujoco(id) || id == RenderedMujocoId)) {
                    foreach (string mode in RenderModesOf(id).Where(mode => !IsMujoco(id) || mode != "rgb_array_list"))
                        yield return new object[] { id, mode };
                }
            }
        }

        /// <summary>The non-human render modes an id's class lists, read from its registered class's metadata, plus <c>rgb_array_list</c>.</summary>
        /// <param name="id">The id.</param>
        /// <returns>The modes.</returns>
        private static IEnumerable<string> RenderModesOf(string id) {
            string family = id.Split('-')[0];
            string[] modes = family switch {
                "CartPole" or "MountainCar" or "MountainCarContinuous" or "Pendulum" or "Acrobot" or "Blackjack" or "LunarLander" or "LunarLanderContinuous"
                    or "BipedalWalker" or "BipedalWalkerHardcore" => new[] { "rgb_array" },
                "FrozenLake" or "FrozenLake8x8" or "CliffWalking" or "CliffWalkingSlippery" or "Taxi" => new[] { "ansi", "rgb_array" },
                "CarRacing" => new[] { "rgb_array", "state_pixels" },
                _ => new[] { "rgb_array", "depth_array", "rgbd_tuple" },
            };
            return modes.Append("rgb_array_list");
        }

        /// <summary>Whether an id is one of the MuJoCo envs, which take a frame size.</summary>
        /// <param name="id">The id.</param>
        /// <returns><see langword="true"/> for a MuJoCo id.</returns>
        private static bool IsMujoco(string id) => RenderModesOf(id).Contains("depth_array");

        /// <summary>The wrapper cases by name: each builds a wrapped env over a fresh <c>make</c>.</summary>
        private static readonly Dictionary<string, Func<IEnv>> WrapperCases = new() {
            ["TimeLimit"] = () => new TimeLimit<NDArray, NDArray>(CartPole(), 40),
            ["Autoreset"] = () => new Autoreset<NDArray, NDArray>(CartPole()),
            ["RecordEpisodeStatistics"] = () => new RecordEpisodeStatistics<NDArray, NDArray>(CartPole()),
            ["RenderCollection"] = () => new RenderCollection<NDArray, NDArray>(CartPole(RenderMode.RgbArray)),
            ["RenderCollection(popFrames: false)"] = () => new RenderCollection<NDArray, NDArray>(CartPole(RenderMode.RgbArray), popFrames: false),
            ["RecordVideo"] = () => new RecordVideo<NDArray, NDArray>(CartPole(RenderMode.RgbArray), VideoFolder(), episodeTrigger: _ => true, videoWriter: new ChecksumVideoWriter()),
            ["HumanRendering"] = () => new HumanRendering<NDArray, NDArray>(Unpaced(CartPole(RenderMode.RgbArray))) { ViewerFactory = NullEnvViewer.Factory },
            ["AddWhiteNoise"] = () => new AddWhiteNoise<NDArray, NDArray>(CartPole(RenderMode.RgbArray), 0.1),
            ["ObstructView"] = () => new ObstructView<NDArray, NDArray>(CartPole(RenderMode.RgbArray), 0.1, 10),
            ["StickyAction"] = () => new StickyAction<NDArray, NDArray>(CartPole(), 0.5),
            ["StickyAction(Pendulum)"] = () => new StickyAction<NDArray, NDArray>(Pendulum(), 0.5),
            ["RepeatAction"] = () => new RepeatAction<NDArray, NDArray>(CartPole(), 2),
            ["TimeAwareObservation"] = () => new TimeAwareObservation<NDArray, NDArray>(CartPole()),
            ["NormalizeObservation"] = () => new NormalizeObservation<NDArray, NDArray>(CartPole()),
            ["DelayObservation"] = () => new DelayObservation<NDArray, NDArray>(CartPole(), 3),
            ["FrameStackObservation"] = () => new FrameStackObservation<NDArray, NDArray>(CartPole(), 4),
            ["MaxAndSkipObservation"] = () => new MaxAndSkipObservation<NDArray, NDArray>(CartPole(), 2),
            ["NormalizeReward"] = () => new NormalizeReward<NDArray, NDArray>(Pendulum()),
            ["TransformAction"] = () => new TransformAction<NDArray, NDArray, NDArray>(Pendulum(), a => a * 0.5, null),
            ["ClipAction"] = () => new ClipAction<NDArray, NDArray>(Pendulum()),
            ["RescaleAction"] = () => new RescaleAction<NDArray, NDArray>(Pendulum(), -1.0, 1.0),
            ["DiscretizeAction"] = () => new DiscretizeAction<NDArray, NDArray>(Pendulum(), 5L),
            ["TransformObservation"] = () => new TransformObservation<NDArray, NDArray, NDArray>(CartPole(), o => o * 2.0, null),
            ["FilterObservation"] = () => new FilterObservation<object?, NDArray>(new AddRenderObservation<NDArray, NDArray>(CartPole(RenderMode.RgbArray), renderOnly: false), new[] { "state" }),
            ["FlattenObservation"] = () => new FlattenObservation<object?, NDArray>(new AddRenderObservation<NDArray, NDArray>(CartPole(RenderMode.RgbArray), renderOnly: false)),
            ["GrayscaleObservation"] = () => new GrayscaleObservation<object?, NDArray>(new AddRenderObservation<NDArray, NDArray>(CartPole(RenderMode.RgbArray))),
            ["ReshapeObservation"] = () => new ReshapeObservation<NDArray, NDArray>(CartPole(), new long[] { 2, 2 }),
            ["ResizeObservation"] = () => new ResizeObservation<object?, NDArray>(new AddRenderObservation<NDArray, NDArray>(CartPole(RenderMode.RgbArray)), new long[] { 40, 60 }),
            ["RescaleObservation"] = () => new RescaleObservation<NDArray, NDArray>(Pendulum(), 0.0, 1.0),
            ["DtypeObservation"] = () => new DtypeObservation<NDArray, NDArray>(CartPole(), np.float64),
            ["AddRenderObservation"] = () => new AddRenderObservation<NDArray, NDArray>(CartPole(RenderMode.RgbArray)),
            ["DiscretizeObservation"] = () => new DiscretizeObservation<NDArray, NDArray>(Pendulum(), 10L),
            ["TransformReward"] = () => new TransformReward<NDArray, NDArray>(CartPole(), r => r * 2.0),
            ["ClipReward"] = () => new ClipReward<NDArray, NDArray>(CartPole(), 0.0, 0.5),
        };

        /// <summary>The wrapper cases' names, one row each.</summary>
        public static IEnumerable<object[]> Wrappers => WrapperCases.Keys.Select(name => new object[] { name });

        /// <summary>The vector cases by name: the three vectorization modes, the autoreset modes, and every vector wrapper.</summary>
        private static readonly Dictionary<string, Func<IVectorEnv>> VectorCases = new() {
            ["sync, next step"] = () => MakeVec("CartPole-v1", 3, VectorizeMode.Sync),
            ["sync, same step"] = () => MakeVec("CartPole-v1", 3, VectorizeMode.Sync, AutoresetMode.SameStep),
            ["sync, disabled"] = () => MakeVec("CartPole-v1", 3, VectorizeMode.Sync, AutoresetMode.Disabled),
            ["async"] = () => MakeVec("CartPole-v1", 3, VectorizeMode.Async),
            ["async, same step"] = () => MakeVec("CartPole-v1", 3, VectorizeMode.Async, AutoresetMode.SameStep),
            ["vector entry point"] = () => MakeVec("CartPole-v1", 3, VectorizeMode.VectorEntryPoint),
            ["sync HalfCheetah-v5"] = () => MakeVec("HalfCheetah-v5", 2, VectorizeMode.Sync),
            ["async LunarLander-v3"] = () => MakeVec("LunarLander-v3", 2, VectorizeMode.Async),
            ["RecordEpisodeStatisticsVector"] = () => new RecordEpisodeStatisticsVector<NDArray, NDArray>(CartPoles()),
            ["DictInfoToList"] = () => new DictInfoToList<NDArray, NDArray>(CartPoles()),
            ["HumanRenderingVector"] = () => new HumanRenderingVector<NDArray, NDArray>(Unpaced(CartPoles(RenderMode.RgbArray))) { ViewerFactory = NullEnvViewer.Factory },
            ["RecordVideoVector"] = () => new RecordVideoVector<NDArray, NDArray>(CartPoles(RenderMode.RgbArray), VideoFolder(), episodeTrigger: _ => true, videoWriter: new ChecksumVideoWriter()),
            ["NormalizeObservationVector"] = () => new NormalizeObservationVector<NDArray>(CartPoles()),
            ["NormalizeRewardVector"] = () => new NormalizeRewardVector<NDArray, NDArray>(Pendulums()),
            ["TransformObservationVector"] = () => new TransformObservationVector<NDArray, NDArray, NDArray>(CartPoles(), o => o * 2.0),
            ["FlattenObservationVector"] = () => new FlattenObservationVector<NDArray, NDArray>(CartPoles()),
            ["ReshapeObservationVector"] = () => new ReshapeObservationVector<NDArray, NDArray>(CartPoles(), new long[] { 2, 2 }),
            ["RescaleObservationVector"] = () => new RescaleObservationVector<NDArray, NDArray>(Pendulums(), 0.0, 1.0),
            ["DtypeObservationVector"] = () => new DtypeObservationVector<NDArray, NDArray>(CartPoles(), np.float64),
            ["GrayscaleObservationVector"] = () => new GrayscaleObservationVector<NDArray, NDArray>(CarRacings()),
            ["ResizeObservationVector"] = () => new ResizeObservationVector<NDArray, NDArray>(CarRacings(), new long[] { 32, 32 }),
            ["TransformActionVector"] = () => new TransformActionVector<NDArray, NDArray, NDArray>(Pendulums(), a => a * 0.5),
            ["ClipActionVector"] = () => new ClipActionVector<NDArray, NDArray>(Pendulums()),
            ["RescaleActionVector"] = () => new RescaleActionVector<NDArray, NDArray>(Pendulums(), -1.0, 1.0),
            ["TransformRewardVector"] = () => new TransformRewardVector<NDArray, NDArray>(CartPoles(), r => r * 2.0),
            ["ClipRewardVector"] = () => new ClipRewardVector<NDArray, NDArray>(CartPoles(), 0.0, 0.5),
        };

        /// <summary>The vector cases' names, one row each.</summary>
        public static IEnumerable<object[]> VectorEnvs => VectorCases.Keys.Select(name => new object[] { name });

        /// <summary>The folders the video cases wrote, deleted after the class runs.</summary>
        private static readonly List<string> VideoFolders = new();

        /// <summary>Deletes the video cases' folders.</summary>
        [ClassCleanup]
        public static void DeleteVideoFolders() {
            lock (VideoFolders) {
                foreach (string folder in VideoFolders.Where(Directory.Exists))
                    Directory.Delete(folder, recursive: true);
                VideoFolders.Clear();
            }
        }

        /// <summary>A new folder under the temp directory for a video case.</summary>
        /// <returns>The folder's path (RecordVideo creates it).</returns>
        private static string VideoFolder() {
            string folder = Path.Combine(Path.GetTempPath(), "gymnet-scope-" + Guid.NewGuid().ToString("N"));
            lock (VideoFolders)
                VideoFolders.Add(folder);
            return folder;
        }

        /// <summary>CartPole-v1 through make, typed.</summary>
        /// <param name="renderMode">The render mode, or <see langword="null"/>.</param>
        /// <returns>The env.</returns>
        private static Env<NDArray, NDArray> CartPole(RenderMode? renderMode = null) =>
            renderMode is { } mode ? Make<NDArray, NDArray>("CartPole-v1", mode) : Make<NDArray, NDArray>("CartPole-v1");

        /// <summary>
        ///     Raises an env's <c>render_fps</c> so far that the human-rendering wrappers' clock never waits: their pacing is
        ///     not under test, and at CartPole's 50 frames a second it would be most of the case's time.
        /// </summary>
        /// <typeparam name="TEnv">The env or vector env type.</typeparam>
        /// <param name="env">The env; its metadata is its own copy, so no other env changes.</param>
        /// <returns><paramref name="env"/>.</returns>
        /// <exception cref="ArgumentException">Thrown when <paramref name="env"/> is neither an <see cref="IEnv"/> nor an <see cref="IVectorEnv"/>.</exception>
        private static TEnv Unpaced<TEnv>(TEnv env) where TEnv : class {
            Metadata metadata = env switch {
                IEnv single => single.Metadata,
                IVectorEnv vector => vector.Metadata,
                _ => throw new ArgumentException("Not an env.", nameof(env)),
            };
            metadata[Metadata.RenderFpsKey] = 1_000_000L;
            return env;
        }

        /// <summary>Pendulum-v1 through make, typed: a continuous action and a bounded observation.</summary>
        /// <returns>The env.</returns>
        private static Env<NDArray, NDArray> Pendulum() => Make<NDArray, NDArray>("Pendulum-v1");

        /// <summary>Three CartPole-v1 envs, synchronous, typed.</summary>
        /// <param name="renderMode">The render mode, or <see langword="null"/>.</param>
        /// <returns>The vector env.</returns>
        private static VectorEnv<NDArray, NDArray> CartPoles(RenderMode? renderMode = null) =>
            MakeVec<NDArray, NDArray>("CartPole-v1", 3, "sync", kwargs: renderMode is { } mode ? new { render_mode = RenderModes.Name(mode) } : null);

        /// <summary>Two Pendulum-v1 envs, synchronous, typed.</summary>
        /// <returns>The vector env.</returns>
        private static VectorEnv<NDArray, NDArray> Pendulums() => MakeVec<NDArray, NDArray>("Pendulum-v1", 2, "sync");

        /// <summary>Two CarRacing-v3 envs, synchronous, typed: image observations.</summary>
        /// <returns>The vector env.</returns>
        private static VectorEnv<NDArray, NDArray> CarRacings() => MakeVec<NDArray, NDArray>("CarRacing-v3", 2, "sync");

        /// <summary>Every built-in env stepped inside caller scopes gives what it gives with no scope.</summary>
        /// <param name="id">The id.</param>
        [TestMethod]
        [DynamicData(nameof(Ids))]
        public void Env_SteppedInsideCallerScopes_IsUnchanged(string id) => AssertScopeSafe(id, () => Make(id), render: false);

        /// <summary>
        ///     Every built-in env rendered inside caller scopes gives the frames (or text) it gives with no scope; each chunk
        ///     renders first, so a frame or cache kept from the scope before is read after that scope closed.
        /// </summary>
        /// <param name="id">The id.</param>
        /// <param name="mode">The render mode.</param>
        [TestMethod]
        [DynamicData(nameof(IdsAndRenderModes))]
        public void Rendering_InsideCallerScopes_IsUnchanged(string id, string mode) =>
            AssertScopeSafe($"{id} {mode}", () => Make(id, mode, kwargs: IsMujoco(id) ? MujocoFrameSize : null), render: true);

        /// <summary>Every wrapper stepped (and, for the rendering ones, rendered) inside caller scopes gives what it gives with no scope.</summary>
        /// <param name="name">The case.</param>
        [TestMethod]
        [DynamicData(nameof(Wrappers))]
        public void Wrapper_SteppedInsideCallerScopes_IsUnchanged(string name) {
            Func<IEnv> make = WrapperCases[name];
            bool render;
            using (IEnv probe = make())
                render = probe.RenderMode is not null && probe.RenderMode != "human";
            AssertScopeSafe(name, make, render);
        }

        /// <summary>Every vectorization and autoreset mode, and every vector wrapper, stepped inside caller scopes gives what it gives with no scope.</summary>
        /// <param name="name">The case.</param>
        [TestMethod]
        [DynamicData(nameof(VectorEnvs))]
        public void VectorEnv_SteppedInsideCallerScopes_IsUnchanged(string name) {
            Func<IVectorEnv> make = VectorCases[name];
            List<string> expected = RunVector(make, scoped: false, name);
            List<string> actual = RunVector(make, scoped: true, name);
            AssertSameRun(name, expected, actual);
        }

        /// <summary>Runs a single-env case both ways and compares the runs.</summary>
        /// <param name="name">The case, for the messages.</param>
        /// <param name="make">Builds the env (a fresh one per run).</param>
        /// <param name="render">Whether each chunk renders first.</param>
        /// <exception cref="AssertFailedException">Thrown when the runs differ, or a scope left the env holding a disposed array.</exception>
        private static void AssertScopeSafe(string name, Func<IEnv> make, bool render) {
            List<string> expected = RunSingle(make, scoped: false, render, name);
            List<string> actual = RunSingle(make, scoped: true, render, name);
            AssertSameRun(name, expected, actual);
        }

        /// <summary>Fails at the first line where two runs differ.</summary>
        /// <param name="name">The case.</param>
        /// <param name="expected">The unscoped run.</param>
        /// <param name="actual">The scoped run.</param>
        /// <exception cref="AssertFailedException">Thrown when the runs differ.</exception>
        private static void AssertSameRun(string name, List<string> expected, List<string> actual) {
            for (int i = 0; i < Math.Min(expected.Count, actual.Count); i++)
                Assert.AreEqual(expected[i], actual[i], $"{name}: line {i} differs under caller scopes");
            Assert.AreEqual(expected.Count, actual.Count, $"{name}: the runs' lengths");
        }

        /// <summary>One seeded stream of a single env, in chunks, each chunk scoped or not.</summary>
        /// <param name="make">Builds the env.</param>
        /// <param name="scoped">Whether each chunk runs inside its own scope.</param>
        /// <param name="render">Whether each chunk renders first (after the first reset).</param>
        /// <param name="name">The case, for the messages.</param>
        /// <returns>One line per reset, step and render: exact texts of everything the env returned.</returns>
        /// <exception cref="AssertFailedException">Thrown when a scope left the env holding a disposed array.</exception>
        private static List<string> RunSingle(Func<IEnv> make, bool scoped, bool render, string name) {
            var lines = new List<string>();
            ChecksumVideoWriter.LastCreated = null;
            using (IEnv env = make()) {
                env.ActionSpace.Seed(0);
                bool reset = true;
                for (int c = 0; c < Chunks; c++) {
                    if (!scoped) {
                        reset = SingleChunk(env, lines, reset, render);
                        continue;
                    }

                    // The scope a caller's [NDScoped] method opens, by hand: everything the chunk builds and drops is disposed here.
                    using (NDScope.Open())
                        reset = SingleChunk(env, lines, reset, render);
                    AssertHoldsNoDisposedArray(env, $"{name}, after scope {c}");
                }
            }

            // The videos RecordVideo wrote, the last one at close: frames it kept across scopes are read there.
            if (ChecksumVideoWriter.LastCreated is { } writer)
                lines.AddRange(writer.Videos);
            return lines;
        }

        /// <summary>A chunk of steps; resets with seed 0 first, without a seed after a finished episode.</summary>
        /// <param name="env">The env.</param>
        /// <param name="lines">The lines so far.</param>
        /// <param name="reset">Whether the next step must reset first.</param>
        /// <param name="render">Whether the chunk renders first.</param>
        /// <returns>Whether the chunk ended on a finished episode.</returns>
        private static bool SingleChunk(IEnv env, List<string> lines, bool reset, bool render) {
            if (render && lines.Count > 0)
                lines.Add("render " + Canonical(env.Render()));
            for (int i = 0; i < Chunk; i++) {
                if (reset) {
                    var result = env.Reset(seed: lines.Count == 0 ? 0 : null);
                    lines.Add($"reset {Canonical(result.Observation)} {Canonical(result.Info)}");
                    reset = false;
                }

                var step = env.Step(env.ActionSpace.Sample());
                lines.Add($"{Canonical(step.Reward)} {step.Terminated} {step.Truncated} {Canonical(step.Observation)} {Canonical(step.Info)}");
                reset = step.Terminated || step.Truncated;
            }

            return reset;
        }

        /// <summary>One seeded stream of a vector env, in chunks, each chunk scoped or not.</summary>
        /// <param name="make">Builds the vector env.</param>
        /// <param name="scoped">Whether each chunk runs inside its own scope.</param>
        /// <param name="name">The case, for the messages.</param>
        /// <returns>One line per reset and step.</returns>
        /// <exception cref="AssertFailedException">Thrown when a scope left the vector env holding a disposed array.</exception>
        private static List<string> RunVector(Func<IVectorEnv> make, bool scoped, string name) {
            var lines = new List<string>();
            ChecksumVideoWriter.LastCreated = null;
            using (IVectorEnv envs = make()) {
                envs.ActionSpace.Seed(0);
                var first = envs.Reset(seed: 0);
                lines.Add($"reset {Canonical(first.Observations)} {Canonical(first.Infos)}");
                for (int c = 0; c < Chunks; c++) {
                    if (!scoped) {
                        VectorChunk(envs, lines);
                        continue;
                    }

                    using (NDScope.Open())
                        VectorChunk(envs, lines);
                    AssertHoldsNoDisposedArray(envs, $"{name}, after scope {c}");
                }
            }

            // The videos RecordVideoVector wrote, the last one at close.
            if (ChecksumVideoWriter.LastCreated is { } writer)
                lines.AddRange(writer.Videos);
            return lines;
        }

        /// <summary>A chunk of vector steps (the vector env resets finished sub-envs itself, per its autoreset mode).</summary>
        /// <param name="envs">The vector env.</param>
        /// <param name="lines">The lines so far.</param>
        private static void VectorChunk(IVectorEnv envs, List<string> lines) {
            for (int i = 0; i < Chunk; i++) {
                var step = envs.Step(envs.ActionSpace.Sample());
                lines.Add($"{Canonical(step.Rewards)} {Canonical(step.Terminations)} {Canonical(step.Truncations)} {Canonical(step.Observations)} {Canonical(step.Infos)}");
                if (envs.Metadata.TryGetValue("autoreset_mode", out object? mode) && mode is AutoresetMode.Disabled) {
                    // With autoreset disabled the caller resets the finished sub-envs, Gymnasium's way: a mask in the options.
                    using NDArray done = np.logical_or(step.Terminations, step.Truncations);
                    if (np.any(done)) {
                        var reset = envs.Reset(options: new Info { ["reset_mask"] = done.copy() });
                        lines.Add($"reset {Canonical(reset.Observations)} {Canonical(reset.Infos)}");
                    }
                }
            }
        }

        /// <summary>
        ///     Fails when the env (or vector env) still references an array that was disposed: a scope reclaimed an array the
        ///     object kept, so its next read is of freed memory, right or wrong only by luck.
        /// </summary>
        /// <param name="root">The env or vector env.</param>
        /// <param name="where">The case and the scope, for the message.</param>
        /// <exception cref="AssertFailedException">Thrown with the paths of the disposed arrays it found.</exception>
        private static void AssertHoldsNoDisposedArray(object root, string where) {
            List<string> found = DisposedArrays(root);
            Assert.AreEqual(0, found.Count, $"{where}: the object graph holds disposed arrays at {string.Join(", ", found.Take(8))}");
        }

        /// <summary>
        ///     Walks an object graph through its fields and collections and returns the path of every reachable array that is
        ///     disposed. Gym.NET's own objects, the runtime's collections, tuples and key/value pairs are opened; NumSharp's
        ///     internals, strings, types, delegates and threading objects are not.
        /// </summary>
        /// <param name="root">Where the walk starts.</param>
        /// <returns>The paths, empty when every reachable array is alive.</returns>
        private static List<string> DisposedArrays(object root) {
            var found = new List<string>();
            var seen = new HashSet<object>(ReferenceEqualityComparer.Instance);
            Visit(root, root.GetType().Name, 0);
            return found;

            void Visit(object? value, string path, int depth) {
                if (value is null || depth > MaxDepth)
                    return;
                Type type = value.GetType();
                if (type.IsPrimitive || type.IsEnum || type.IsPointer || value is string or Type or Delegate or MemberInfo or Thread or Task or WaitHandle
                        or CancellationTokenSource or Generator)
                    return;
                if (!type.IsValueType && !seen.Add(value))
                    return;
                if (value is NDArray array) {
                    // A disposed array still referenced is the defect; its views and storage are NumSharp's business.
                    if (array.IsDisposed)
                        found.Add(path);
                    return;
                }

                if (value is Array items) {
                    if (type.GetElementType()!.IsPrimitive)
                        return;
                    int index = 0;
                    foreach (object? item in items)
                        Visit(item, $"{path}[{index++}]", depth + 1);
                    return;
                }

                string ns = type.Namespace ?? "";
                if (ns.StartsWith("System.Collections", StringComparison.Ordinal) && value is IEnumerable sequence) {
                    // The runtime's collections are read through their enumerators (snapshots for the concurrent ones).
                    int index = 0;
                    foreach (object? item in sequence)
                        Visit(item, $"{path}[{index++}]", depth + 1);
                    return;
                }

                bool open = ns.StartsWith("Gym", StringComparison.Ordinal) || value is ITuple
                            || (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(KeyValuePair<,>));
                if (!open)
                    return;
                for (Type? t = type; t is not null && t != typeof(object); t = t.BaseType) {
                    foreach (FieldInfo field in t.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
                        Visit(field.GetValue(value), $"{path}.{field.Name}", depth + 1);
                }
            }
        }

        /// <summary>
        ///     An exact text of a value an env returned: every array's dtype, shape and element bits, dict keys in order, tuples
        ///     and lists in order, numbers by their bits. RecordEpisodeStatistics' elapsed time (<c>t</c>) is left out: it is
        ///     the wall clock, never equal between two runs.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>The text.</returns>
        private static string Canonical(object? value) {
            switch (value) {
                case null:
                    return "None";
                case NDArray array:
                    return CanonicalArray(array);
                case RenderResult render:
                    return CanonicalRender(render);
                case string text:
                    return PyBuiltins.Repr(text);
                case bool flag:
                    return flag ? "True" : "False";
                case double number:
                    return BitConverter.DoubleToInt64Bits(number).ToString("X", CultureInfo.InvariantCulture);
                case float number:
                    return BitConverter.SingleToInt32Bits(number).ToString("X", CultureInfo.InvariantCulture);
                case IReadOnlyDictionary<string, object?> dict:
                    bool statistics = dict.ContainsKey("r") && dict.ContainsKey("l") && dict.ContainsKey("t");
                    return "{" + string.Join(", ", dict.Where(p => !(statistics && p.Key is "t" or "_t")).Select(p => $"{p.Key}: {Canonical(p.Value)}")) + "}";
                case ITuple tuple:
                    return "(" + string.Join(", ", Enumerable.Range(0, tuple.Length).Select(i => Canonical(tuple[i]))) + ")";
                case IEnumerable items:
                    return "[" + string.Join(", ", items.Cast<object?>().Select(Canonical)) + "]";
                default:
                    return Convert.ToString(value, CultureInfo.InvariantCulture) ?? "";
            }
        }

        /// <summary>
        ///     An array's exact text: floats widened to float64 (exact), everything else read as int64, by bits; an array of
        ///     more than 64 elements (a frame) by a 64-bit FNV-1a-style hash over those 64-bit words, which any single changed
        ///     bit changes.
        /// </summary>
        /// <param name="array">The array.</param>
        /// <returns>The text.</returns>
        private static string CanonicalArray(NDArray array) {
            long[] bits;
            if (array.dtype.kind == 'f') {
                using NDArray wide = array.astype(np.float64);
                bits = wide.ToArray<double>().Select(BitConverter.DoubleToInt64Bits).ToArray();
            } else {
                using NDArray wide = array.astype(np.int64);
                bits = wide.ToArray<long>();
            }

            if (bits.Length <= 64)
                return $"{array.dtype.name}{array.Shape}[{string.Join(",", bits.Select(b => b.ToString("X", CultureInfo.InvariantCulture)))}]";
            // A frame's text would be megabytes a step; its hash still differs for any differing element.
            unchecked {
                ulong hash = 14695981039346656037UL;
                foreach (long b in bits) {
                    hash ^= (ulong)b;
                    hash *= 1099511628211UL;
                }

                return $"{array.dtype.name}{array.Shape}#{hash:X16}";
            }
        }

        /// <summary>A render result's exact text: its frames' (or text's) exact texts.</summary>
        /// <param name="render">The result.</param>
        /// <returns>The text.</returns>
        private static string CanonicalRender(RenderResult render) => render switch {
            RenderResult.Rgb rgb => "Rgb " + CanonicalArray(rgb.Frame),
            RenderResult.Depth depth => "Depth " + CanonicalArray(depth.Frame),
            RenderResult.RgbDepth pair => $"RgbDepth {CanonicalArray(pair.RgbFrame)} {CanonicalArray(pair.DepthFrame)}",
            RenderResult.Frames frames => "Frames [" + string.Join(", ", frames.Items.Select(CanonicalRender)) + "]",
            _ => render.ToString() ?? "",
        };

        /// <summary>
        ///     A video writer that keeps an exact text of each video's frames instead of writing files, so a frame RecordVideo
        ///     kept across scopes is read (and compared) when the video is written.
        /// </summary>
        private sealed class ChecksumVideoWriter : IVideoWriter {
            /// <summary>Registers the writer as this thread's latest, where the run that built it reads its videos.</summary>
            public ChecksumVideoWriter() => LastCreated = this;

            /// <summary>The writer the current thread built last (each run builds its env, and so its writer, on its own thread).</summary>
            [ThreadStatic]
            public static ChecksumVideoWriter? LastCreated;

            /// <summary>The texts, one per video written.</summary>
            public List<string> Videos { get; } = new();

            /// <summary>A folder per video, as the default writer's; nothing is written into it.</summary>
            public string Extension => "";

            /// <summary>Records the video's frames.</summary>
            /// <param name="path">The video's path.</param>
            /// <param name="frames">The frames; read, not kept.</param>
            /// <param name="fps">The frame rate.</param>
            public void Write(string path, IReadOnlyList<NDArray> frames, double fps) =>
                Videos.Add($"{Path.GetFileName(path)} {fps.ToString(CultureInfo.InvariantCulture)} [{string.Join(", ", frames.Select(CanonicalArray))}]");
        }
    }
}
