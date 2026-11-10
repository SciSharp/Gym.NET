using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Pair;
using Gym.Tests.Parity.Session;
using Gym.Utils;
using Gym.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Python.Runtime;

namespace Gym.Tests.Parity.Tests.Vector {
    /// <summary>
    ///     Gym.NET's <see cref="AsyncVectorEnv{TObs, TAct}"/> (worker threads, DR-22) against Gymnasium's <c>AsyncVectorEnv</c>
    ///     (worker processes), live: lockstep episodes in every autoreset mode with and without shared memory and in every
    ///     execution mode, the quirks of Gymnasium's two copy paths, masked resets, member calls, the call protocol's errors,
    ///     and a sub-env's failure.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Gymnasium spawns its workers with <c>multiprocessing</c>; embedded, <c>sys.executable</c> is the test host, so the
    ///     cells name the session's interpreter with <c>multiprocessing.set_executable</c> first, and are Inconclusive when the
    ///     session found only a library (<c>PYTHONNET_PYDLL</c>). Each cell closes Gymnasium's env, which joins its processes.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/vector/async_vector_env.py">gymnasium/vector/async_vector_env.py</see>.</para>
    /// </remarks>
    [TestClass]
    public sealed class AsyncVectorEnvLiveParityTests : ParityTestBase {
        /// <summary>Whether <c>multiprocessing.set_executable</c> already named the interpreter in this process.</summary>
        private static bool _spawnConfigured;

        /// <summary>
        ///     <c>make_vec("CartPole-v1", 3, "async", vector_kwargs={"shared_memory": ..., "autoreset_mode": ...})</c> through
        ///     150 sampled steps across episode ends (<c>DISABLED</c> resetting through <c>reset_mask</c>), every result compared
        ///     whole, Gym.NET in the given execution mode.
        /// </summary>
        /// <param name="executionMode">Gym.NET's execution mode.</param>
        /// <param name="sharedMemory">Whether the observations go through the shared buffer.</param>
        /// <param name="autoresetMode">The autoreset mode's value.</param>
        [TestMethod]
        [DataRow("ThreadPool", true, "NextStep")]
        [DataRow("ThreadPool", true, "SameStep")]
        [DataRow("ThreadPool", true, "Disabled")]
        [DataRow("ThreadPool", false, "NextStep")]
        [DataRow("ThreadPool", false, "SameStep")]
        [DataRow("ThreadPool", false, "Disabled")]
        [DataRow("DedicatedThreads", true, "NextStep")]
        [DataRow("DedicatedThreads", false, "SameStep")]
        [DataRow("ThreadPerEnv", true, "SameStep")]
        [DataRow("ThreadPerEnv", false, "NextStep")]
        public void CartPole_Lockstep_MatchesGymnasium(string executionMode, bool sharedMemory, string autoresetMode) {
            RequireSpawn();
            var vectorKwargs = new Dictionary<string, object?> { ["shared_memory"] = sharedMemory, ["autoreset_mode"] = autoresetMode };
            using PyRef python = PyRef.Of(() => Python.gymnasium.make_vec("CartPole-v1", 3, "async", vectorKwargs));
            try {
                using VectorEnv<NDArray, NDArray> dotnet = MakeAsync<NDArray, NDArray>("CartPole-v1", 3, sharedMemory, autoresetMode, executionMode);
                Lockstep($"CartPole-v1 x3 async ({executionMode}, shared_memory={sharedMemory}, {autoresetMode})", python, dotnet, 7, 150, resetEnded: autoresetMode == "Disabled");
            } finally {
                Close(python);
            }
        }

        /// <summary>
        ///     FrozenLake's int observations and float infos, and Blackjack's tuple observations, through 120 sampled steps in
        ///     both copy paths: the shared buffer's int64 and tuple layouts.
        /// </summary>
        /// <param name="id">The env.</param>
        /// <param name="sharedMemory">Whether the observations go through the shared buffer.</param>
        [TestMethod]
        [DataRow("FrozenLake-v1", true)]
        [DataRow("FrozenLake-v1", false)]
        [DataRow("Blackjack-v1", true)]
        [DataRow("Blackjack-v1", false)]
        public void ToyText_Lockstep_MatchesGymnasium(string id, bool sharedMemory) {
            RequireSpawn();
            var vectorKwargs = new Dictionary<string, object?> { ["shared_memory"] = sharedMemory };
            using PyRef python = PyRef.Of(() => Python.gymnasium.make_vec(id, 3, "async", vectorKwargs));
            try {
                using var dotnet = (IVectorEnv)Gymnasium.MakeVec(id, numEnvs: 3, vectorizationMode: "async", vectorKwargs: new Info { ["shared_memory"] = sharedMemory });
                Lockstep($"{id} x3 async (shared_memory={sharedMemory})", python, dotnet, 3, 120, resetEnded: false);
            } finally {
                Close(python);
            }
        }

        /// <summary>
        ///     A full reset right after an episode ended, under <c>NEXT_STEP</c>: with shared memory the reset clears the
        ///     pending autoreset; without it Gymnasium's worker keeps it, so the next step resets that sub-env again with a
        ///     reward of 0. Gym.NET reproduces both paths.
        /// </summary>
        /// <param name="sharedMemory">Whether the observations go through the shared buffer.</param>
        [TestMethod]
        [DataRow(true)]
        [DataRow(false)]
        public void ResetAfterAnEpisodeEnd_KeepsGymnasiumsAutoreset(bool sharedMemory) {
            RequireSpawn();
            var vectorKwargs = new Dictionary<string, object?> { ["shared_memory"] = sharedMemory };
            using PyRef python = PyRef.Of(() => Python.gymnasium.make_vec("CartPole-v1", 2, "async", vectorKwargs));
            try {
                using VectorEnv<NDArray, NDArray> dotnet = MakeAsync<NDArray, NDArray>("CartPole-v1", 2, sharedMemory, "NextStep", "ThreadPool");
                string label = $"CartPole-v1 x2 async (shared_memory={sharedMemory})";
                ByteContract.AssertSame($"{label}: reset(seed=3)", Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, 3)), Outcome.Capture(() => ResetTuple(dotnet.Reset(3))));

                // Push right until a cart falls: the episode ends with the pole down.
                bool ended = false;
                for (int i = 1; i <= 100 && !ended; i++) {
                    using ParityAction push = ParityAction.Continuous(np.array(new long[] { 1, 1 }));
                    bool[] flags = Array.Empty<bool>();
                    ByteContract.AssertSame($"{label}: step([1, 1]) #{i}", Outcome.CapturePython(() => Python.gymnasium.Env.step(python.Object, push.PythonValue())),
                        Outcome.Capture(() => {
                            VectorStepResult<NDArray> result = dotnet.Step((NDArray)push.DotNet);
                            using NDArray endedArray = np.logical_or(result.Terminations, result.Truncations);
                            flags = endedArray.ToArray<bool>();
                            return StepTuple(result);
                        }));
                    ended = flags.Any(flag => flag);
                }

                Assert.IsTrue(ended, "no episode ended within 100 pushes");
                ByteContract.AssertSame($"{label}: reset(seed=4) after the end", Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, 4)),
                    Outcome.Capture(() => ResetTuple(dotnet.Reset(4))));
                for (int i = 1; i <= 5; i++) {
                    using ParityAction action = ParityAction.Continuous(np.array(new long[] { 0, 1 }));
                    ByteContract.AssertSame($"{label}: step([0, 1]) #{i} after the reset", Outcome.CapturePython(() => Python.gymnasium.Env.step(python.Object, action.PythonValue())),
                        Outcome.Capture(() => StepTuple(dotnet.Step((NDArray)action.DotNet))));
                }
            } finally {
                Close(python);
            }
        }

        /// <summary>
        ///     <c>reset(options={"reset_mask": [True, False, True]})</c> after 10 steps, both copy paths: the masked-out
        ///     sub-env answers with its last observation and no info.
        /// </summary>
        /// <param name="sharedMemory">Whether the observations go through the shared buffer.</param>
        [TestMethod]
        [DataRow(true)]
        [DataRow(false)]
        public void MaskedReset_MatchesGymnasium(bool sharedMemory) {
            RequireSpawn();
            var vectorKwargs = new Dictionary<string, object?> { ["shared_memory"] = sharedMemory };
            using PyRef python = PyRef.Of(() => Python.gymnasium.make_vec("FrozenLake-v1", 3, "async", vectorKwargs));
            try {
                using var dotnet = (IVectorEnv)Gymnasium.MakeVec("FrozenLake-v1", numEnvs: 3, vectorizationMode: "async", vectorKwargs: new Info { ["shared_memory"] = sharedMemory });
                string label = $"FrozenLake-v1 x3 async (shared_memory={sharedMemory})";
                Lockstep(label, python, dotnet, 11, 10, resetEnded: false);
                using NDArray mask = np.array(new[] { true, false, true });
                var options = new Dictionary<string, object?> { ["reset_mask"] = mask };
                ByteContract.AssertSame($"{label}: reset(seed=20, options={{'reset_mask': [True, False, True]}})",
                    Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, 20, options)),
                    Outcome.Capture(() => ResetTuple(dotnet.Reset(20, new Info { ["reset_mask"] = mask }))));
                Lockstep($"{label} after the masked reset", python, dotnet, null, 10, resetEnded: false);
            } finally {
                Close(python);
            }
        }

        /// <summary><c>set_attr("gravity", [...])</c>, <c>get_attr("gravity")</c>, and <c>call("render")</c>'s frames of three CartPoles.</summary>
        [TestMethod]
        public void Attributes_And_Render_MatchGymnasium() {
            RequireSpawn();
            var kwargs = new Dictionary<string, object?> { ["render_mode"] = "rgb_array" };
            using PyRef python = PyRef.Of(() => Python.gymnasium.make_vec("CartPole-v1", 3, "async", null, kwargs));
            try {
                using var dotnet = (AsyncVectorEnv)Gymnasium.MakeVec("CartPole-v1", numEnvs: 3, vectorizationMode: "async", kwargs: new { render_mode = "rgb_array" });
                ByteContract.AssertSame("reset(seed=2)", Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, 2)), Outcome.Capture(() => ResetTuple(dotnet.Reset(2))));
                ByteContract.AssertSame("set_attr('gravity', [9.81, 3.72, 8.87])",
                    Outcome.CapturePython(() => {
                        using PyObject values = Oracle.PyConvert.ToPython(new List<object?> { 9.81, 3.72, 8.87 });
                        return Python.gymnasium.vector.SyncVectorEnv.set_attr(python.Object, "gravity", values);
                    }),
                    Outcome.Capture(() => {
                        dotnet.SetAttr("gravity", new List<object?> { 9.81, 3.72, 8.87 });
                        return (object?)null;
                    }));
                ByteContract.AssertSame("get_attr('gravity')", Outcome.CapturePython(() => Python.gymnasium.vector.SyncVectorEnv.get_attr(python.Object, "gravity")),
                    Outcome.Capture(() => dotnet.GetAttr("gravity").ToArray()));
                ByteContract.AssertSame("call('render')", Outcome.CapturePython(() => Python.gymnasium.vector.SyncVectorEnv.call(python.Object, "render")),
                    Outcome.Capture(() => dotnet.Call("render").Select(frame => frame is RenderResult.Rgb rgb ? (object?)rgb.Frame : null).ToArray()));
            } finally {
                Close(python);
            }
        }

        /// <summary>
        ///     The call protocol's errors with Gymnasium's texts: a wait with nothing pending, a call while another is pending,
        ///     the warning of a close that interrupts one, and any call after the close.
        /// </summary>
        [TestMethod]
        public void ProtocolErrors_MatchGymnasium() {
            RequireSpawn();
            using PyRef python = PyRef.Of(() => Python.gymnasium.make_vec("CartPole-v1", 2, "async"));
            using var dotnet = (AsyncVectorEnv)Gymnasium.MakeVec("CartPole-v1", numEnvs: 2, vectorizationMode: "async");
            try {
                ByteContract.AssertSame("reset_wait() with nothing pending", Outcome.CapturePython(() => Python.gymnasium.vector.AsyncVectorEnv.reset_wait(python.Object)),
                    Outcome.Capture(() => ResetTuple(dotnet.ResetWait())));
                ByteContract.AssertSame("step_wait() with nothing pending", Outcome.CapturePython(() => Python.gymnasium.vector.AsyncVectorEnv.step_wait(python.Object)),
                    Outcome.Capture(() => StepTuple(dotnet.StepWait())));
                ByteContract.AssertSame("call_wait() with nothing pending", Outcome.CapturePython(() => Python.gymnasium.vector.AsyncVectorEnv.call_wait(python.Object)),
                    Outcome.Capture(() => dotnet.CallWait().ToArray()));

                ByteContract.AssertSame("reset_async(seed=0)", Outcome.CapturePython(() => Python.gymnasium.vector.AsyncVectorEnv.reset_async(python.Object, 0)),
                    Outcome.Capture(() => {
                        dotnet.ResetAsync(0);
                        return (object?)null;
                    }));
                using (ParityAction actions = ParityAction.Continuous(np.array(new long[] { 0, 1 }))) {
                    ByteContract.AssertSame("step_async(...) while a reset is pending",
                        Outcome.CapturePython(() => Python.gymnasium.vector.AsyncVectorEnv.step_async(python.Object, actions.PythonValue())),
                        Outcome.Capture(() => {
                            dotnet.StepAsync((object?)actions.DotNet);
                            return (object?)null;
                        }));
                }

                ByteContract.AssertSame("call_async('gravity') while a reset is pending",
                    Outcome.CapturePython(() => Python.gymnasium.vector.AsyncVectorEnv.call_async(python.Object, "gravity")),
                    Outcome.Capture(() => {
                        dotnet.CallAsync("gravity");
                        return (object?)null;
                    }));

                // close(terminate=True) while the reset is pending: the warning, then the workers stop.
                const string closeStatements =
                    "import warnings\n" +
                    "with warnings.catch_warnings(record=True) as caught:\n" +
                    "    warnings.simplefilter('always')\n" +
                    "    env.close(terminate=True)\n" +
                    "result = [(w.category.__name__, str(w.message)) for w in caught]\n";
                ByteContract.AssertSame("close(terminate=True) while a reset is pending: the warnings",
                    Outcome.CapturePython(() => Python.evalStmts(closeStatements, "result", ("env", python.Object))),
                    Outcome.Capture(() => {
                        using WarningCatcher catcher = Warnings.Catch();
                        dotnet.Close(null, terminate: true);
                        return catcher.Records.Select(w => (object?)new object?[] { w.Category.ToString(), w.Message }).ToList();
                    }));

                ByteContract.AssertSame("reset() after close", Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, 0)),
                    Outcome.Capture(() => ResetTuple(dotnet.Reset(0))));
                ByteContract.AssertSame("call('gravity') after close", Outcome.CapturePython(() => Python.gymnasium.vector.SyncVectorEnv.call(python.Object, "gravity")),
                    Outcome.Capture(() => dotnet.Call("gravity").ToArray()));
            } finally {
                Close(python);
            }
        }

        /// <summary><c>max_concurrency</c>'s refusals with Gymnasium's texts: non-integers (a float, a string, a whole float) and non-positive integers.</summary>
        [TestMethod]
        public void MaxConcurrency_Refusals_MatchGymnasium() {
            RequireSpawn();
            foreach (object value in new object[] { 2.5, "2", 2.0, 0L, -1L, -10L }) {
                var kwargs = new Dictionary<string, object?> { ["max_concurrency"] = value };
                ByteContract.AssertSame($"AsyncVectorEnv(..., max_concurrency={PyBuiltins.Repr(value)})",
                    Outcome.CapturePython(() => Python.gymnasium.make_vec("CartPole-v1", 2, "async", kwargs)),
                    Outcome.Capture(() => Gymnasium.MakeVec("CartPole-v1", numEnvs: 2, vectorizationMode: "async", vectorKwargs: new Info { ["max_concurrency"] = value })));
            }
        }

        /// <summary>
        ///     <c>max_concurrency</c> of 1 and 2 over 4 CartPoles: the same results as unlimited, through 60 sampled steps
        ///     (the limit changes when sub-envs run, never what they return).
        /// </summary>
        /// <param name="maxConcurrency">The limit.</param>
        [TestMethod]
        [DataRow(1L)]
        [DataRow(2L)]
        public void MaxConcurrency_Lockstep_MatchesGymnasium(long maxConcurrency) {
            RequireSpawn();
            var vectorKwargs = new Dictionary<string, object?> { ["max_concurrency"] = maxConcurrency };
            using PyRef python = PyRef.Of(() => Python.gymnasium.make_vec("CartPole-v1", 4, "async", vectorKwargs));
            try {
                using var dotnet = (IVectorEnv)Gymnasium.MakeVec("CartPole-v1", numEnvs: 4, vectorizationMode: "async", vectorKwargs: new Info { ["max_concurrency"] = maxConcurrency });
                Lockstep($"CartPole-v1 x4 async (max_concurrency={maxConcurrency})", python, dotnet, 5, 60, resetEnded: false);
            } finally {
                Close(python);
            }
        }

        /// <summary>
        ///     A sub-env that raises (CartPole given the invalid action 2): the step raises Gymnasium's <c>AssertionError</c>
        ///     with its text, logged with the worker's index and the closing line; that worker is dropped, so the next step
        ///     fails as Python's does on its <c>None</c> pipe; closing still works.
        /// </summary>
        [TestMethod]
        public void SubEnvError_MatchesGymnasium() {
            RequireSpawn();
            using PyRef python = PyRef.Of(() => Python.gymnasium.make_vec("CartPole-v1", 3, "async"));
            using var dotnet = (AsyncVectorEnv)Gymnasium.MakeVec("CartPole-v1", numEnvs: 3, vectorizationMode: "async");
            try {
                ByteContract.AssertSame("reset(seed=0)", Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, 0)), Outcome.Capture(() => ResetTuple(dotnet.Reset(0))));

                using ParityAction actions = ParityAction.Continuous(np.array(new long[] { 0, 2, 1 }));
                const string stepStatements =
                    "import warnings\n" +
                    "with warnings.catch_warnings(record=True) as caught:\n" +
                    "    warnings.simplefilter('always')\n" +
                    "    try:\n" +
                    "        env.step(actions)\n" +
                    "        error = None\n" +
                    "    except Exception as e:\n" +
                    "        error = (type(e).__name__, str(e))\n" +
                    "messages = [str(w.message) for w in caught]\n" +
                    "result = (error, messages[0], messages[-1], len(messages))\n";
                ByteContract.AssertSame("step([0, 2, 1]): the error, the first and last error lines, and their count",
                    Outcome.CapturePython(() => Python.evalStmts(stepStatements, "result", ("env", python.Object), ("actions", actions.PythonValue()))),
                    Outcome.Capture(() => {
                        using WarningCatcher catcher = Warnings.Catch();
                        object? error = null;
                        try {
                            // The step is expected to throw; a result, had it come, is released.
                            Outcome.Release(StepTuple(dotnet.Step(actions.DotNet)));
                        } catch (Exception e) {
                            error = new object?[] { ExceptionMap.PythonTypeFor(e), e.Message };
                        }

                        List<string> messages = catcher.Records.Select(w => w.Message).ToList();
                        return new object?[] { error, messages[0], messages[^1], (long)messages.Count };
                    }));

                using ParityAction again = ParityAction.Continuous(np.array(new long[] { 0, 0, 0 }));
                ByteContract.AssertSame("step([0, 0, 0]) after the failure", Outcome.CapturePython(() => Python.gymnasium.Env.step(python.Object, again.PythonValue())),
                    Outcome.Capture(() => StepTuple(dotnet.Step(again.DotNet))));
            } finally {
                Close(python);
            }
        }

        /// <summary>
        ///     Names the session's interpreter as the one <c>multiprocessing</c> spawns, once per process; Inconclusive when the
        ///     session has only a library.
        /// </summary>
        private static void RequireSpawn() {
            string? interpreter = GymPythonSession.Installation?.Interpreter;
            if (interpreter is null)
                Assert.Inconclusive("Gymnasium's AsyncVectorEnv spawns worker processes, which needs the interpreter's path; this session found only a library (PYTHONNET_PYDLL).");
            if (_spawnConfigured)
                return;
            // The session may know the interpreter by a bare name found on PATH ("python"); CreateProcess needs its file.
            string executable = ResolveOnPath(interpreter);
            PythonRuntime.Run(() => Python.multiprocessing.set_executable(executable));
            _spawnConfigured = true;
        }

        /// <summary>An executable's full path: as given when rooted, else the first <c>PATH</c> directory holding it (with <c>.exe</c> on Windows).</summary>
        /// <param name="name">The path or bare name.</param>
        /// <returns>The full path; the name itself when nothing on <c>PATH</c> holds it, so the spawn reports the missing file.</returns>
        private static string ResolveOnPath(string name) {
            if (System.IO.Path.IsPathRooted(name))
                return name;
            string[] candidates = OperatingSystem.IsWindows() && !name.EndsWith(".exe", StringComparison.OrdinalIgnoreCase) ? new[] { name + ".exe", name } : new[] { name };
            foreach (string directory in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(System.IO.Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries)) {
                foreach (string candidate in candidates) {
                    string path = System.IO.Path.Combine(directory.Trim('"'), candidate);
                    if (System.IO.File.Exists(path))
                        return path;
                }
            }

            return name;
        }

        /// <summary>Gym.NET's typed async vector env of an id.</summary>
        /// <typeparam name="TObs">The batched observation type.</typeparam>
        /// <typeparam name="TAct">The batched action type.</typeparam>
        /// <param name="id">The env.</param>
        /// <param name="numEnvs">The number of sub-envs.</param>
        /// <param name="sharedMemory">Whether the observations go through the shared buffer.</param>
        /// <param name="autoresetMode">The autoreset mode's value.</param>
        /// <param name="executionMode">The execution mode's name.</param>
        /// <returns>The vector env; the caller owns it.</returns>
        private static VectorEnv<TObs, TAct> MakeAsync<TObs, TAct>(string id, int numEnvs, bool sharedMemory, string autoresetMode, string executionMode) {
            AsyncExecutionMode mode = Enum.Parse<AsyncExecutionMode>(executionMode);
            var vectorKwargs = new Info { ["shared_memory"] = sharedMemory, ["autoreset_mode"] = autoresetMode, ["execution_mode"] = mode };
            if (mode == AsyncExecutionMode.DedicatedThreads)
                vectorKwargs["thread_count"] = 2;
            return Gymnasium.MakeVec<TObs, TAct>(id, numEnvs: numEnvs, vectorizationMode: "async", vectorKwargs: vectorKwargs);
        }

        /// <summary>
        ///     Runs both vector envs in lockstep: an optional seeded reset, then <paramref name="steps"/> steps of the batched
        ///     action spaces' samples (seeded with the reset), compared before each step, every result compared whole.
        /// </summary>
        /// <param name="label">What runs, for the reports.</param>
        /// <param name="python">Gymnasium's vector env.</param>
        /// <param name="dotnet">Gym.NET's vector env.</param>
        /// <param name="seed">The reset seed and the action spaces' seed; <see langword="null"/> continues without a reset or a reseed.</param>
        /// <param name="steps">How many steps.</param>
        /// <param name="resetEnded">Whether ended sub-envs are reset through <c>reset_mask</c> after each step (<c>DISABLED</c> mode).</param>
        /// <exception cref="ParityException">Thrown at the first difference.</exception>
        private static void Lockstep(string label, PyRef python, IVectorEnv dotnet, long? seed, int steps, bool resetEnded) {
            if (seed is { } given) {
                ByteContract.AssertSame($"{label}: reset(seed={given})", Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, given)),
                    Outcome.Capture(() => ResetTuple(dotnet.Reset(given))));
            }

            using PyRef pythonSpace = PyRef.Of(() => Python.gymnasium.Env.action_space(python.Object));
            if (seed is { } spaceSeed) {
                Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.seed(pythonSpace.Object, spaceSeed)).Expect($"{label}: action_space.seed({spaceSeed})");
                dotnet.ActionSpace.Seed(spaceSeed);
            }

            for (int i = 1; i <= steps; i++) {
                using ParityAction action = PythonRuntime.Run(() => ParityAction.Sampled(Python.gymnasium.spaces.space.Space.sample(pythonSpace.Object), dotnet.ActionSpace.Sample()!));
                ByteContract.AssertSame($"{label}: action_space.sample() #{i}", Outcome.Returned(action.Facts), Outcome.Returned(ValueFacts.Of(action.DotNet)));
                bool[] ended = Array.Empty<bool>();
                ByteContract.AssertSame($"{label}: step #{i}", Outcome.CapturePython(() => Python.gymnasium.Env.step(python.Object, action.PythonValue())),
                    Outcome.Capture(() => {
                        VectorStepResult<object?> result = dotnet.Step(action.DotNet);
                        if (resetEnded) {
                            using NDArray endedArray = np.logical_or(result.Terminations, result.Truncations);
                            ended = endedArray.ToArray<bool>();
                        }

                        return StepTuple(result);
                    }));
                if (resetEnded && ended.Any(flag => flag)) {
                    using NDArray mask = np.array(ended);
                    var options = new Dictionary<string, object?> { ["reset_mask"] = mask };
                    ByteContract.AssertSame($"{label}: reset(options={{'reset_mask': ...}}) after step #{i}", Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, null, options)),
                        Outcome.Capture(() => ResetTuple(dotnet.Reset(seed: null, options: new Info { ["reset_mask"] = mask }))));
                }
            }
        }

        /// <summary>Closes Gymnasium's vector env (joining its worker processes); a failure to close is not the cell's verdict.</summary>
        /// <param name="python">The vector env.</param>
        private static void Close(PyRef python) =>
            PythonRuntime.Run(() => {
                try {
                    using PyObject _ = Python.gymnasium.vector.AsyncVectorEnv.close(python.Object, terminate: true);
                } catch (PythonException) {
                    // Already closed by the cell, or a pending call's error: the processes are terminated either way.
                }
            });

        /// <summary>A typed Gym.NET reset as the tuple Gymnasium returns.</summary>
        /// <typeparam name="TObs">The batched observation type.</typeparam>
        /// <param name="result">The reset.</param>
        /// <returns><c>(observations, infos)</c>.</returns>
        private static object?[] ResetTuple<TObs>(VectorResetResult<TObs> result) => new object?[] { result.Observations, result.Infos };

        /// <summary>A typed Gym.NET step as the tuple Gymnasium returns.</summary>
        /// <typeparam name="TObs">The batched observation type.</typeparam>
        /// <param name="result">The step.</param>
        /// <returns><c>(observations, rewards, terminations, truncations, infos)</c>.</returns>
        private static object?[] StepTuple<TObs>(VectorStepResult<TObs> result) => new object?[] { result.Observations, result.Rewards, result.Terminations, result.Truncations, result.Infos };
    }
}
