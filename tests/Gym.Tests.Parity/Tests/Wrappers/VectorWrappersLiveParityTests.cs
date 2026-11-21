using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Gym.Environments;
using Gym.Envs;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Pair;
using Gym.Tests.Parity.Session;
using Gym.Utils;
using Gym.Vector;
using Gym.Wrappers;
using Gym.Wrappers.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Python.Runtime;

namespace Gym.Tests.Parity.Tests.Wrappers {
    /// <summary>
    ///     The vector wrappers (<c>gymnasium.wrappers.vector</c>, DR-24's suffixed names) against Gymnasium's, live: each over
    ///     a sync vector env, a seeded reset and sampled steps across episode ends compared whole (observations, rewards,
    ///     flags, infos), and what the stateful ones keep afterwards.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The vectorized transforms apply the single-env wrappers per sub-env, so their cells pick envs whose spaces each
    ///     wrapper takes: CartPole's state, its rendered frames (through <c>AddRenderObservation</c> on every sub-env), a dict
    ///     of state and time, Pendulum's actions. <c>RecordEpisodeStatistics</c>' <c>t</c> entry is wall-clock time and is
    ///     left out on both sides. <c>RecordVideo</c> runs over the moviepy stand-in of
    ///     <see cref="RenderingWrappersLiveParityTests"/>. Reference:
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/vector/__init__.py">gymnasium/wrappers/vector</see>.
    ///     </para>
    /// </remarks>
    [TestClass]
    public sealed class VectorWrappersLiveParityTests : ParityTestBase {
        /// <summary>A step with <c>infos["episode"]["t"]</c> removed (wall-clock time), as the tuple Gymnasium returns.</summary>
        private const string StepWithoutTimeStatements =
            "o, r, te, tr, i = env.step(actions)\n" +
            "if isinstance(i, dict) and 'episode' in i:\n" +
            "    i = dict(i)\n" +
            "    i['episode'] = {k: v for k, v in i['episode'].items() if k != 't'}\n" +
            "result = (o, r, te, tr, i)\n";

        /// <summary><c>NormalizeObservation(make_vec(id, 3))</c> through 200 steps, and <c>obs_rms</c> afterwards.</summary>
        /// <param name="id">The env.</param>
        [TestMethod]
        [DataRow("CartPole-v1")]
        [DataRow("Pendulum-v1")]
        public void NormalizeObservationVector_MatchesGymnasium(string id) {
            using PyRef python = PyRef.Of(() => {
                using PyObject envs = Python.gymnasium.make_vec(id, 3, "sync");
                return Python.gymnasium.wrappers.vector.NormalizeObservation(envs);
            });
            using var dotnet = new NormalizeObservationVector<NDArray>(Gymnasium.MakeVec<NDArray, NDArray>(id, numEnvs: 3, vectorizationMode: "sync"));
            Lockstep($"NormalizeObservation({id} x3)", python, dotnet, seed: 7, steps: 200);
            AssertAttribute(python, dotnet, "obs_rms", "result = (env.obs_rms.mean, env.obs_rms.var, env.obs_rms.count)",
                env => new object?[] { dotnet.ObsRms.Mean.copy(), dotnet.ObsRms.Var.copy(), dotnet.ObsRms.Count });
        }

        /// <summary><c>NormalizeReward(make_vec("CartPole-v1", 3, autoreset_mode=mode), gamma=0.9)</c> through 200 steps, and the statistics afterwards.</summary>
        /// <param name="mode">The autoreset mode's value.</param>
        [TestMethod]
        [DataRow("NextStep")]
        [DataRow("SameStep")]
        public void NormalizeRewardVector_MatchesGymnasium(string mode) {
            var vectorKwargs = new Dictionary<string, object?> { ["autoreset_mode"] = mode };
            using PyRef python = PyRef.Of(() => {
                using PyObject envs = Python.gymnasium.make_vec("CartPole-v1", 3, "sync", vectorKwargs);
                return Python.gymnasium.wrappers.vector.NormalizeReward(envs, 0.9);
            });
            using var dotnet = new NormalizeRewardVector<NDArray, NDArray>(
                Gymnasium.MakeVec<NDArray, NDArray>("CartPole-v1", numEnvs: 3, vectorizationMode: "sync", vectorKwargs: new Info { ["autoreset_mode"] = mode }), gamma: 0.9);
            Lockstep($"NormalizeReward(CartPole-v1 x3 {mode})", python, dotnet, seed: 3, steps: 200);
            AssertAttribute(python, dotnet, "return_rms and accumulated_reward", "result = (env.return_rms.mean, env.return_rms.var, env.return_rms.count, env.accumulated_reward)",
                env => new object?[] { dotnet.ReturnRms.Mean.copy(), dotnet.ReturnRms.Var.copy(), dotnet.ReturnRms.Count, dotnet.AccumulatedReward.copy() });
        }

        /// <summary>
        ///     <c>RecordEpisodeStatistics(make_vec("CartPole-v1", 3, autoreset_mode=mode))</c> through 250 steps: the
        ///     <c>episode</c> infos (time left out), and the counters, returns and lengths afterwards.
        /// </summary>
        /// <param name="mode">The autoreset mode's value.</param>
        [TestMethod]
        [DataRow("NextStep")]
        [DataRow("SameStep")]
        public void RecordEpisodeStatisticsVector_MatchesGymnasium(string mode) {
            var vectorKwargs = new Dictionary<string, object?> { ["autoreset_mode"] = mode };
            using PyRef python = PyRef.Of(() => {
                using PyObject envs = Python.gymnasium.make_vec("CartPole-v1", 3, "sync", vectorKwargs);
                return Python.gymnasium.wrappers.vector.RecordEpisodeStatistics(envs, 5);
            });
            using var dotnet = new RecordEpisodeStatisticsVector<NDArray, NDArray>(
                Gymnasium.MakeVec<NDArray, NDArray>("CartPole-v1", numEnvs: 3, vectorizationMode: "sync", vectorKwargs: new Info { ["autoreset_mode"] = mode }), bufferLength: 5);
            Lockstep($"RecordEpisodeStatistics(CartPole-v1 x3 {mode})", python, dotnet, seed: 11, steps: 250, withoutTime: true);
            AssertAttribute(python, dotnet, "statistics",
                "result = (int(env.episode_count), env.episode_returns, env.episode_lengths, env.prev_dones, tuple(env.return_queue), tuple(int(x) for x in env.length_queue))",
                env => new object?[] {
                    dotnet.EpisodeCount, dotnet.EpisodeReturns.copy(), dotnet.EpisodeLengths.copy(), dotnet.PrevDones.copy(),
                    dotnet.ReturnQueue.Cast<object?>().ToArray(), dotnet.LengthQueue.Cast<object?>().ToArray(),
                });
        }

        /// <summary>
        ///     The vectorized observation transforms over CartPole x3: <c>FlattenObservation</c> of the state,
        ///     <c>ReshapeObservation</c> to <c>(2, 2)</c>, <c>DtypeObservation</c> to float64, and <c>TransformObservation</c>
        ///     with a batch function.
        /// </summary>
        /// <param name="wrapper">The wrapper.</param>
        [TestMethod]
        [DataRow("flatten")]
        [DataRow("reshape")]
        [DataRow("dtype")]
        [DataRow("transform")]
        public void ObservationTransformsVector_State_MatchGymnasium(string wrapper) {
            using PyRef python = PyRef.Of(() => {
                using PyObject envs = Python.gymnasium.make_vec("CartPole-v1", 3, "sync");
                switch (wrapper) {
                    case "flatten":
                        return Python.gymnasium.wrappers.vector.FlattenObservation(envs);
                    case "reshape":
                        return Python.gymnasium.wrappers.vector.ReshapeObservation(envs, new[] { 2L, 2L });
                    case "dtype":
                        return Python.gymnasium.wrappers.vector.DtypeObservation(envs, "float64");
                    default: {
                        using PyObject func = WrapperParity.Lambda("lambda obs: obs * 2 + 1");
                        return Python.gymnasium.wrappers.vector.TransformObservation(envs, func);
                    }
                }
            });
            VectorEnv<NDArray, NDArray> made = Gymnasium.MakeVec<NDArray, NDArray>("CartPole-v1", numEnvs: 3, vectorizationMode: "sync");
            using IVectorEnv dotnet = wrapper switch {
                "flatten" => new FlattenObservationVector<NDArray, NDArray>(made),
                "reshape" => new ReshapeObservationVector<NDArray, NDArray>(made, new[] { 2L, 2L }),
                "dtype" => new DtypeObservationVector<NDArray, NDArray>(made, "float64"),
                _ => new TransformObservationVector<NDArray, NDArray, NDArray>(made, obs => {
                    using NDArray doubled = obs * 2;
                    return doubled + 1;
                }),
            };
            Lockstep($"{wrapper}(CartPole-v1 x3)", python, dotnet, seed: 5, steps: 120);
            AssertAttribute(python, dotnet, "spaces", "result = (repr(env.observation_space), repr(env.single_observation_space))",
                env => new object?[] { env.ObservationSpace.ToString(), env.SingleObservationSpace.ToString() });
        }

        /// <summary>
        ///     The vectorized image transforms over CartPole x2's rendered frames (<c>AddRenderObservation</c> on each sub-env):
        ///     <c>GrayscaleObservation</c> with and without the channel axis, and <c>ResizeObservation</c> to <c>(40, 60)</c>.
        /// </summary>
        /// <param name="wrapper">The wrapper.</param>
        [TestMethod]
        [DataRow("grayscale")]
        [DataRow("grayscale-keep")]
        [DataRow("resize")]
        public void ObservationTransformsVector_Frames_MatchGymnasium(string wrapper) {
            using PyRef python = PyRef.Of(() => {
                using PyObject addRender = Python.evalStmts("from gymnasium.wrappers import AddRenderObservation\nresult = [AddRenderObservation]\n", "result");
                var kwargs = new Dictionary<string, object?> { ["render_mode"] = "rgb_array" };
                using PyObject envs = Python.gymnasium.make_vec("CartPole-v1", 2, "sync", null, kwargs, addRender);
                return wrapper switch {
                    "grayscale" => Python.gymnasium.wrappers.vector.GrayscaleObservation(envs),
                    "grayscale-keep" => Python.gymnasium.wrappers.vector.GrayscaleObservation(envs, true),
                    _ => Python.gymnasium.wrappers.vector.ResizeObservation(envs, new[] { 40L, 60L }),
                };
            });
            VectorEnv<NDArray, NDArray> made = Gymnasium.MakeVec<NDArray, NDArray>("CartPole-v1", numEnvs: 2, vectorizationMode: "sync",
                wrappers: new Func<IEnv, IEnv>[] { env => new AddRenderObservation<NDArray, NDArray>((Env<NDArray, NDArray>)env) }, kwargs: new { render_mode = "rgb_array" });
            using IVectorEnv dotnet = wrapper switch {
                "grayscale" => new GrayscaleObservationVector<NDArray, NDArray>(made),
                "grayscale-keep" => new GrayscaleObservationVector<NDArray, NDArray>(made, keepDim: true),
                _ => new ResizeObservationVector<NDArray, NDArray>(made, new[] { 40L, 60L }),
            };
            Lockstep($"{wrapper}(CartPole-v1 frames x2)", python, dotnet, seed: 2, steps: 15);
        }

        /// <summary><c>FilterObservation</c> over CartPole x3 made with <c>TimeAwareObservation(flatten=False)</c> on each sub-env: the <c>obs</c> key kept.</summary>
        [TestMethod]
        public void FilterObservationVector_MatchesGymnasium() {
            using PyRef python = PyRef.Of(() => {
                using PyObject timed = Python.evalStmts("from gymnasium.wrappers import TimeAwareObservation\nresult = [lambda env: TimeAwareObservation(env, flatten=False)]\n", "result");
                using PyObject envs = Python.gymnasium.make_vec("CartPole-v1", 3, "sync", null, null, timed);
                using PyObject keys = Python.evalStmts("result = ['obs']\n", "result");
                return Python.gymnasium.wrappers.vector.FilterObservation(envs, keys);
            });
            VectorEnv<Info, NDArray> made = Gymnasium.MakeVec<Info, NDArray>("CartPole-v1", numEnvs: 3, vectorizationMode: "sync",
                wrappers: new Func<IEnv, IEnv>[] { env => new TimeAwareObservation<NDArray, NDArray>((Env<NDArray, NDArray>)env, flatten: false) });
            using var dotnet = new FilterObservationVector<Info, NDArray>(made, new List<object?> { "obs" });
            Lockstep("FilterObservation(TimeAware CartPole-v1 x3, ['obs'])", python, dotnet, seed: 4, steps: 120);
        }

        /// <summary>
        ///     The action transforms over Pendulum x3: <c>ClipAction</c> (the samples of an unbounded box clipped),
        ///     <c>RescaleAction</c> from [-1, 1], and <c>TransformAction</c> with a batch function.
        /// </summary>
        /// <param name="wrapper">The wrapper.</param>
        [TestMethod]
        [DataRow("clip")]
        [DataRow("rescale")]
        [DataRow("transform")]
        public void ActionTransformsVector_MatchGymnasium(string wrapper) {
            using PyRef python = PyRef.Of(() => {
                using PyObject envs = Python.gymnasium.make_vec("Pendulum-v1", 3, "sync");
                switch (wrapper) {
                    case "clip":
                        return Python.gymnasium.wrappers.vector.ClipAction(envs);
                    case "rescale":
                        return Python.gymnasium.wrappers.vector.RescaleAction(envs, -1.0, 1.0);
                    default: {
                        using PyObject func = WrapperParity.Lambda("lambda action: action * 0.5");
                        return Python.gymnasium.wrappers.vector.TransformAction(envs, func);
                    }
                }
            });
            VectorEnv<NDArray, NDArray> made = Gymnasium.MakeVec<NDArray, NDArray>("Pendulum-v1", numEnvs: 3, vectorizationMode: "sync");
            using IVectorEnv dotnet = wrapper switch {
                "clip" => new ClipActionVector<NDArray, NDArray>(made),
                "rescale" => new RescaleActionVector<NDArray, NDArray>(made, -1.0, 1.0),
                _ => new TransformActionVector<NDArray, NDArray, NDArray>(made, action => action * 0.5),
            };
            Lockstep($"{wrapper}(Pendulum-v1 x3)", python, dotnet, seed: 9, steps: 120);
            AssertAttribute(python, dotnet, "action spaces", "result = (repr(env.action_space), repr(env.single_action_space))",
                env => new object?[] { env.ActionSpace.ToString(), env.SingleActionSpace.ToString() });
        }

        /// <summary>The reward transforms over CartPole x3: <c>ClipReward(0, 0.5)</c> per sub-env, and <c>TransformReward</c> with a batch function.</summary>
        /// <param name="wrapper">The wrapper.</param>
        [TestMethod]
        [DataRow("clip")]
        [DataRow("transform")]
        public void RewardTransformsVector_MatchGymnasium(string wrapper) {
            using PyRef python = PyRef.Of(() => {
                using PyObject envs = Python.gymnasium.make_vec("CartPole-v1", 3, "sync");
                if (wrapper == "clip")
                    return Python.gymnasium.wrappers.vector.ClipReward(envs, 0.0, 0.5);
                using PyObject func = WrapperParity.Lambda("lambda reward: reward * 2 - 1");
                return Python.gymnasium.wrappers.vector.TransformReward(envs, func);
            });
            VectorEnv<NDArray, NDArray> made = Gymnasium.MakeVec<NDArray, NDArray>("CartPole-v1", numEnvs: 3, vectorizationMode: "sync");
            using IVectorEnv dotnet = wrapper == "clip"
                ? new ClipRewardVector<NDArray, NDArray>(made, 0.0, 0.5)
                : new TransformRewardVector<NDArray, NDArray>(made, rewards => {
                    using NDArray doubled = rewards * 2;
                    return doubled - 1;
                });
            Lockstep($"{wrapper}(CartPole-v1 x3)", python, dotnet, seed: 6, steps: 120);
        }

        /// <summary>
        ///     <c>DictInfoToList(make_vec("FrozenLake-v1", 3))</c> through 100 steps: each step's infos as one dict per sub-env
        ///     (FrozenLake's <c>prob</c>), compared through Gym.NET's list-returning <c>Step</c>.
        /// </summary>
        [TestMethod]
        public void DictInfoToList_MatchesGymnasium() {
            using PyRef python = PyRef.Of(() => {
                using PyObject envs = Python.gymnasium.make_vec("FrozenLake-v1", 3, "sync");
                return Python.gymnasium.wrappers.vector.DictInfoToList(envs);
            });
            using var dotnet = new DictInfoToList<NDArray, NDArray>(Gymnasium.MakeVec<NDArray, NDArray>("FrozenLake-v1", numEnvs: 3, vectorizationMode: "sync"));
            ByteContract.AssertSame("DictInfoToList: reset(seed=1)", Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, 1)),
                Outcome.Capture(() => {
                    VectorListResetResult<NDArray> reset = dotnet.Reset(1);
                    return new object?[] { reset.Observations, reset.Infos.Cast<object?>().ToList() };
                }));
            using PyRef pythonSpace = PyRef.Of(() => Python.gymnasium.Env.action_space(python.Object));
            Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.seed(pythonSpace.Object, 1)).Expect("action_space.seed(1)");
            dotnet.ActionSpace.Seed(1);
            for (int i = 1; i <= 100; i++) {
                using ParityAction action = PythonRuntime.Run(() => ParityAction.Sampled(Python.gymnasium.spaces.space.Space.sample(pythonSpace.Object), dotnet.ActionSpace.Sample()!));
                ByteContract.AssertSame($"DictInfoToList: step #{i}", Outcome.CapturePython(() => Python.gymnasium.Env.step(python.Object, action.PythonValue())),
                    Outcome.Capture(() => {
                        VectorListStepResult<NDArray> step = dotnet.Step((NDArray)action.DotNet);
                        return new object?[] { step.Observations, step.Rewards, step.Terminations, step.Truncations, step.Infos.Cast<object?>().ToList() };
                    }));
            }
        }

        /// <summary>
        ///     <c>HumanRendering(make_vec("CartPole-v1", n, render_mode="rgb_array"))</c> through 30 steps (pygame's dummy display
        ///     against <see cref="NullEnvViewer"/>): the results, and the grid Gymnasium chose for the screen.
        /// </summary>
        /// <param name="numEnvs">The sub-envs.</param>
        /// <param name="screenWidth">The screen's width, or 0 for the sub-env frame's.</param>
        /// <param name="screenHeight">The screen's height.</param>
        [TestMethod]
        [DataRow(3, 0, 0)]
        [DataRow(4, 1200, 800)]
        public void HumanRenderingVector_MatchesGymnasium(int numEnvs, int screenWidth, int screenHeight) {
            EnvViewerFactory? previous = Gymnasium.HumanViewerFactory;
            Gymnasium.HumanViewerFactory = NullEnvViewer.Factory;
            try {
                long[]? screen = screenWidth == 0 ? null : new long[] { screenWidth, screenHeight };
                var kwargs = new Dictionary<string, object?> { ["render_mode"] = "rgb_array" };
                using PyRef python = PyRef.Of(() => {
                    using PyObject envs = Python.gymnasium.make_vec("CartPole-v1", numEnvs, "sync", null, kwargs);
                    return Python.gymnasium.wrappers.vector.HumanRendering(envs, screen);
                });
                using var dotnet = new HumanRenderingVector<NDArray, NDArray>(
                    Gymnasium.MakeVec<NDArray, NDArray>("CartPole-v1", numEnvs: numEnvs, vectorizationMode: "sync", kwargs: new { render_mode = "rgb_array" }),
                    screen is null ? null : (screenWidth, screenHeight));
                Lockstep($"HumanRendering(CartPole-v1 x{numEnvs})", python, dotnet, seed: 8, steps: 30);
                AssertAttribute(python, dotnet, "grid", "result = (tuple(int(x) for x in env.screen_size), env.scaled_subenv_size, env.num_rows, env.num_cols, env.render_mode)",
                    env => new object?[] {
                        new object?[] { (long)dotnet.ScreenSize!.Value.Width, (long)dotnet.ScreenSize.Value.Height },
                        new object?[] { (long)dotnet.ScaledSubenvSize!.Value.Width, (long)dotnet.ScaledSubenvSize.Value.Height },
                        (long)dotnet.NumRows!.Value, (long)dotnet.NumCols!.Value, dotnet.RenderMode,
                    });
                PythonRuntime.Run(() => {
                    using PyObject _ = Python.evalStmts("env.close()\nresult = None\n", "result", ("env", python.Object));
                });
            } finally {
                Gymnasium.HumanViewerFactory = previous;
            }
        }

        /// <summary>
        ///     <c>RecordVideo(make_vec("CartPole-v1", 3, render_mode="rgb_array"), ...)</c> through 200 steps over the moviepy
        ///     stand-in: the videos' names, mosaics and frame rate, under the default schedule, only the first sub-env, a 16:9
        ///     aspect, and a step trigger with a length.
        /// </summary>
        /// <param name="variant">The case.</param>
        [TestMethod]
        [DataRow("default")]
        [DataRow("first-only")]
        [DataRow("wide")]
        [DataRow("steps")]
        public void RecordVideoVector_MatchesGymnasium(string variant) {
            RenderingWrappersLiveParityTests.InstallMoviepyStub();
            string pythonFolder = Path.Combine(Path.GetTempPath(), "gymnet-parity-vvideos-" + Guid.NewGuid().ToString("N"));
            string dotnetFolder = Path.Combine(Path.GetTempPath(), "gymnet-parity-vvideos-" + Guid.NewGuid().ToString("N"));
            var writer = new RenderingWrappersLiveParityTests.CapturingVideoWriter();
            try {
                long[] aspect = variant == "wide" ? new long[] { 16, 9 } : new long[] { 1, 1 };
                bool firstOnly = variant == "first-only";
                long length = variant == "steps" ? 12 : 0;
                var kwargs = new Dictionary<string, object?> { ["render_mode"] = "rgb_array" };
                using PyRef python = PyRef.Of(() => {
                    using PyObject envs = Python.gymnasium.make_vec("CartPole-v1", 3, "sync", null, kwargs);
                    using PyObject? step = variant == "steps" ? WrapperParity.Lambda("lambda x: x % 50 == 0") : null;
                    return Python.gymnasium.wrappers.vector.RecordVideo(envs, pythonFolder, aspect, firstOnly, null, step, length);
                });
                using var dotnet = new RecordVideoVector<NDArray, NDArray>(
                    Gymnasium.MakeVec<NDArray, NDArray>("CartPole-v1", numEnvs: 3, vectorizationMode: "sync", kwargs: new { render_mode = "rgb_array" }), dotnetFolder,
                    videoAspectRatio: ((int)aspect[0], (int)aspect[1]), recordFirstOnly: firstOnly, stepTrigger: variant == "steps" ? x => x % 50 == 0 : null,
                    videoLength: length, videoWriter: writer);
                Lockstep($"RecordVideo(CartPole-v1 x3, {variant})", python, dotnet, seed: 12, steps: 200);
                PythonRuntime.Run(() => {
                    using PyObject _ = Python.evalStmts("env.close()\nresult = None\n", "result", ("env", python.Object));
                });
                dotnet.Close();
                Assert.IsTrue(writer.Describe().Length > 0, "Gym.NET wrote no video");
                ByteContract.AssertSame($"videos written ({variant})", Outcome.CapturePython(() => Python.evalStmts(RenderingWrappersLiveParityTests.WrittenStatements, "result")),
                    Outcome.Returned(ValueFacts.Of(writer.Describe())));
            } finally {
                if (Directory.Exists(pythonFolder))
                    Directory.Delete(pythonFolder, recursive: true);
                if (Directory.Exists(dotnetFolder))
                    Directory.Delete(dotnetFolder, recursive: true);
                writer.Dispose();
            }
        }

        /// <summary>
        ///     Resets both sides with <paramref name="seed"/>, seeds both action spaces, then steps both with the same sampled
        ///     actions, comparing every result whole.
        /// </summary>
        /// <param name="label">What runs, for the reports.</param>
        /// <param name="python">Gymnasium's vector env.</param>
        /// <param name="dotnet">Gym.NET's vector env.</param>
        /// <param name="seed">The reset seed and the action spaces' seed.</param>
        /// <param name="steps">How many steps.</param>
        /// <param name="withoutTime">Whether <c>infos["episode"]["t"]</c> (wall-clock time) is left out on both sides.</param>
        /// <exception cref="ParityException">Thrown at the first difference.</exception>
        private static void Lockstep(string label, PyRef python, IVectorEnv dotnet, long seed, int steps, bool withoutTime = false) {
            ByteContract.AssertSame($"{label}: reset(seed={seed})", Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, seed)),
                Outcome.Capture(() => {
                    VectorResetResult<object?> reset = dotnet.Reset(seed);
                    return new[] { reset.Observations, reset.Infos };
                }));
            using PyRef pythonSpace = PyRef.Of(() => Python.gymnasium.Env.action_space(python.Object));
            Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.seed(pythonSpace.Object, seed)).Expect($"{label}: action_space.seed({seed})");
            dotnet.ActionSpace.Seed(seed);
            for (int i = 1; i <= steps; i++) {
                using ParityAction action = PythonRuntime.Run(() => ParityAction.Sampled(Python.gymnasium.spaces.space.Space.sample(pythonSpace.Object), dotnet.ActionSpace.Sample()!));
                ByteContract.AssertSame($"{label}: action_space.sample() #{i}", Outcome.Returned(action.Facts), Outcome.Returned(ValueFacts.Of(action.DotNet)));
                Outcome pythonStep = withoutTime
                    ? Outcome.CapturePython(() => Python.evalStmts(StepWithoutTimeStatements, "result", ("env", python.Object), ("actions", action.PythonValue())))
                    : Outcome.CapturePython(() => Python.gymnasium.Env.step(python.Object, action.PythonValue()));
                ByteContract.AssertSame($"{label}: step #{i}", pythonStep, Outcome.Capture(() => {
                    VectorStepResult<object?> result = dotnet.Step(action.DotNet);
                    Info infos = withoutTime ? WithoutTime(result.Infos) : result.Infos;
                    return new object?[] { result.Observations, result.Rewards, result.Terminations, result.Truncations, infos };
                }));
            }
        }

        /// <summary>A copy of the infos with <c>episode["t"]</c> removed, as the Python side's snippet removes it.</summary>
        /// <param name="infos">The infos.</param>
        /// <returns>The copy; the values are shared.</returns>
        private static Info WithoutTime(Info infos) {
            var copy = new Info(infos);
            if (copy.TryGetValue("episode", out object? episode) && episode is Info statistics) {
                var trimmed = new Info(statistics);
                trimmed.Remove("t");
                copy["episode"] = trimmed;
            }

            return copy;
        }

        /// <summary>Compares an attribute read by a snippet on Gymnasium's wrapper with the value Gym.NET's gives.</summary>
        /// <param name="python">Gymnasium's wrapper.</param>
        /// <param name="dotnet">Gym.NET's wrapper.</param>
        /// <param name="label">The comparison's name.</param>
        /// <param name="statements">A snippet setting <c>result</c> from <c>env</c>.</param>
        /// <param name="read">Reads the value from Gym.NET's wrapper.</param>
        /// <exception cref="ParityException">Thrown when the values differ.</exception>
        private static void AssertAttribute(PyRef python, IVectorEnv dotnet, string label, string statements, Func<IVectorEnv, object?> read) =>
            ByteContract.AssertSame(label, Outcome.CapturePython(() => Python.evalStmts(statements, "result", ("env", python.Object))), Outcome.Capture(() => read(dotnet)));
    }
}
