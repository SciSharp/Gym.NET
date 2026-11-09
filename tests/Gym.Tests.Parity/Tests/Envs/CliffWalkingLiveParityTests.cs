using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Environments.Envs.ToyText;
using Gym.Envs;
using Gym.Observations;
using Gym.Spaces;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Pair;
using Gym.Tests.Parity.Session;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace Gym.Tests.Parity.Tests.Envs {
    /// <summary>
    ///     CliffWalking-v1 and CliffWalkingSlippery-v1 against Gymnasium's <c>CliffWalkingEnv</c>, live and in lockstep with
    ///     Gym.NET's <see cref="CliffWalkingEnv"/>: the transition table, resets, episodes on dry and slippery ground, Python's
    ///     dict and list lookups of every kind of action, the <c>ansi</c> text and the <c>rgb_array</c> frames at every cell,
    ///     spaces, metadata and <c>make</c>'s stack.
    /// </summary>
    /// <remarks>
    ///     Every lockstep compares <c>s</c> after each call. The dynamics are table lookups and one generator draw per call,
    ///     so nothing here depends on the platform's math. Reference:
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/cliffwalking.py#L28">gymnasium/envs/toy_text/cliffwalking.py</see>.
    /// </remarks>
    [TestClass]
    public sealed class CliffWalkingLiveParityTests : ParityTestBase {
        /// <summary>The constructor arguments compared, each in Python spelling and as Gym.NET's constructor call.</summary>
        private static readonly (string Python, Dictionary<string, object?> Kwargs, Func<string?, CliffWalkingEnv> DotNet)[] Variants = {
            ("is_slippery=False", new Dictionary<string, object?> { ["is_slippery"] = false }, mode => new CliffWalkingEnv(renderMode: mode)),
            ("is_slippery=True", new Dictionary<string, object?> { ["is_slippery"] = true }, mode => new CliffWalkingEnv(renderMode: mode, isSlippery: true)),
        };

        /// <summary>The transition table <c>P</c>: every state's every action's outcomes, probability, next state, reward and flag, dry and slippery.</summary>
        [TestMethod]
        public void GymNet_TransitionTable_MatchesGymnasium() {
            foreach ((string text, Dictionary<string, object?> kwargs, Func<string?, CliffWalkingEnv> create) in Variants) {
                ValueFacts python = Outcome.CapturePython(() => {
                    using PyObject made = Python.gymnasium.make("CliffWalking-v1", kwargs: kwargs);
                    using PyObject unwrapped = Python.gymnasium.Env.unwrapped(made);
                    // Each outcome as (float(p), int(s), float(r), bool(t)): the table's values, independent of their Python types.
                    const string table = "rows = [(s, a, float(p), int(n), float(r), bool(t)) for s in sorted(env.P) for a in sorted(env.P[s]) for (p, n, r, t) in env.P[s][a]]\n";
                    return Python.evalStmts(table, "rows", ("env", unwrapped));
                }).Expect($"CliffWalking({text}).P");
                CliffWalkingEnv dotnet = create(null);
                var rows = new List<object?>();
                foreach (long s in dotnet.P.Keys.Order())
                    foreach (long a in dotnet.P[s].Keys.Order())
                        foreach (ToyTextTransition t in dotnet.P[s][a])
                            rows.Add(new object?[] { s, a, t.Probability, t.NextState, t.Reward, t.Terminated });
                ByteContract.AssertSame($"CliffWalking({text}).P", Outcome.Returned(python), Outcome.Returned(ValueFacts.Of(rows)));
            }
        }

        /// <summary>Episodes of seeded sampled actions, 200 steps each (past the goal, whose row the table keeps), dry and slippery, three seeds each.</summary>
        [TestMethod]
        public void GymNet_SampledEpisodes_MatchGymnasium() {
            foreach ((string text, Dictionary<string, object?> kwargs, Func<string?, CliffWalkingEnv> create) in Variants) {
                foreach (long seed in new long[] { 0, 42, 7 }) {
                    using var pair = new GymPair(PyGymEnv.Make("CliffWalking-v1", kwargs), new GymNetEnv(create(null)), stateAttribute: "s");
                    PairRun run = pair.Run(seed, ActionSources.Sampled(seed), maxSteps: 200);
                    TestContext.WriteLine($"{text}: {run}");
                }
            }
        }

        /// <summary>
        ///     A walk along the cliff's edge to the goal and on past it, dry: up, eleven steps right, down onto the goal; then a
        ///     step into the cliff from the start, which sends the agent back with -100.
        /// </summary>
        [TestMethod]
        public void GymNet_GoalAndCliff_MatchGymnasium() {
            long[] path = new long[] { CliffWalkingEnv.Up }.Concat(Enumerable.Repeat(CliffWalkingEnv.Right, 11)).Append(CliffWalkingEnv.Down)
                .Append(CliffWalkingEnv.Down).Append(CliffWalkingEnv.Left).ToArray();
            using var pair = new GymPair(PyGymEnv.Make("CliffWalking-v1"), new GymNetEnv(new CliffWalkingEnv()), stateAttribute: "s");
            pair.Reset(3);
            foreach (long action in path) {
                using var parity = ParityAction.Discrete(action);
                pair.Step(parity);
            }

            pair.Reset(4);
            using var right = ParityAction.Discrete(CliffWalkingEnv.Right);
            pair.Step(right);
        }

        /// <summary>
        ///     Actions as Python's dict lookup takes them, one step each: <c>True</c>, NumPy integer scalars of several dtypes,
        ///     and floats equal to a key; and the ones it refuses, with Python's <c>KeyError</c>/<c>TypeError</c>, before and after
        ///     a reset (the state's <c>AttributeError</c> first).
        /// </summary>
        [TestMethod]
        public void GymNet_Actions_AreLookedUpAsPython() {
            var accepted = new (string Python, Func<PyObject> Gymnasium, Func<object?> DotNet)[] {
                ("True", () => PyConvert.ToPython(true), () => true),
                ("2.0", () => PyConvert.ToPython(2.0), () => 2.0),
                ("np.float32(1.0)", () => Python.numpy.float32(1.0f), () => 1.0f),
                ("np.int8(3)", () => Python.numpy.int8(3), () => NDArray.Scalar<sbyte>(3)),
                ("np.uint8(0)", () => Python.numpy.uint8(0), () => NDArray.Scalar<byte>(0)),
                ("np.float64(-0.0)", () => Python.numpy.float64(-0.0), () => NDArray.Scalar<double>(-0.0)),
            };
            var refused = new (string Python, Func<PyObject> Gymnasium, Func<object?> DotNet)[] {
                ("4", () => PyConvert.ToPython(4L), () => 4L),
                ("-1", () => PyConvert.ToPython(-1L), () => -1L),
                ("1.5", () => PyConvert.ToPython(1.5), () => 1.5),
                ("np.int64(7)", () => Python.numpy.int64(7), () => NDArray.Scalar<long>(7L)),
                ("'1'", () => PyConvert.ToPython("1"), () => "1"),
                ("None", () => PyConvert.ToPython(null), () => null),
                ("np.array([1])", () => {
                    using NDArray one = np.array(new[] { 1L });
                    return one.ToNumpyCopy();
                }, () => np.array(new[] { 1L })),
            };

            using PyRef python = PyRef.Of(() => {
                using PyObject made = Python.gymnasium.make("CliffWalking-v1");
                return Python.gymnasium.Env.unwrapped(made);
            });
            var dotnet = new CliffWalkingEnv();
            IEnv untyped = dotnet;
            foreach ((string text, Func<PyObject> gymnasium, Func<object?> value) in refused.Take(2)) {
                Outcome before = Outcome.CapturePython(() => {
                    using PyObject action = gymnasium();
                    return Python.gymnasium.Env.step(python.Object, action);
                });
                ByteContract.AssertSame($"step({text}) before reset", before, Outcome.Capture(() => untyped.Step(value())));
            }

            Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, 3)).Expect("reset(seed=3)");
            ((IDisposable)dotnet.Reset(3L).Observation).Dispose();
            foreach ((string text, Func<PyObject> gymnasium, Func<object?> value) in accepted.Concat(refused)) {
                Outcome pythonStep = Outcome.CapturePython(() => {
                    using PyObject action = gymnasium();
                    return Python.gymnasium.Env.step(python.Object, action);
                });
                FrozenLakeLiveParityTests.AssertStepSame($"step({text})", pythonStep, Outcome.Capture(() => FrozenLakeLiveParityTests.StepTuple(untyped.Step(value()))));
                ByteContract.AssertSame($"s after step({text})", Outcome.CapturePython(() => Python.gymnasium.envs.toy_text.cliffwalking.CliffWalkingEnv.s(python.Object)),
                    Outcome.Returned(ValueFacts.Of(dotnet.S)));
            }
        }

        /// <summary>
        ///     The <c>ansi</c> text before a reset (Python's <c>AttributeError</c>), after the reset, along seeded episodes, and
        ///     with the agent injected at every cell (the cliff and the goal drawn under it), byte for byte.
        /// </summary>
        [TestMethod]
        public void GymNet_AnsiText_MatchesGymnasium() {
            foreach ((string text, Dictionary<string, object?> kwargs, Func<string?, CliffWalkingEnv> create) in Variants) {
                using PyRef python = PyRef.Of(() => {
                    using PyObject made = Python.gymnasium.make("CliffWalking-v1", render_mode: "ansi", kwargs: kwargs);
                    return Python.gymnasium.Env.unwrapped(made);
                });
                CliffWalkingEnv dotnet = create("ansi");
                ByteContract.AssertSame($"{text}: render() before reset", Outcome.CapturePython(() => Python.gymnasium.Env.render(python.Object)), Outcome.Capture(() => AnsiText(dotnet)));
                Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, 5)).Expect("reset(seed=5)");
                ((IDisposable)dotnet.Reset(5L).Observation).Dispose();
                ByteContract.AssertSame($"{text}: render() after reset", Outcome.CapturePython(() => Python.gymnasium.Env.render(python.Object)), Outcome.Capture(() => AnsiText(dotnet)));
                for (int i = 0; i < 30; i++) {
                    long action = (i * 5 + 1) % 4;
                    Outcome.CapturePython(() => {
                        using PyObject a = PyConvert.ToPython(action);
                        return Python.gymnasium.Env.step(python.Object, a);
                    }).Expect($"step({action})");
                    using (NDArray a = NDArray.Scalar<long>(action))
                        ((IDisposable)dotnet.Step(a).Observation).Dispose();
                    ByteContract.AssertSame($"{text}: render() after step {i + 1}", Outcome.CapturePython(() => Python.gymnasium.Env.render(python.Object)), Outcome.Capture(() => AnsiText(dotnet)));
                }

                for (long s = 0; s < dotnet.NS; s++) {
                    long state = s;
                    PythonRuntime.Run(() => {
                        using PyObject value = PyConvert.ToPython(state);
                        Python.gymnasium.envs.toy_text.cliffwalking.CliffWalkingEnv.s(python.Object, value);
                    });
                    dotnet.S = state;
                    ByteContract.AssertSame($"{text}: render() at s={state}", Outcome.CapturePython(() => Python.gymnasium.Env.render(python.Object)), Outcome.Capture(() => AnsiText(dotnet)));
                }
            }
        }

        /// <summary>
        ///     <c>rgb_array</c> frames after the reset and along seeded episodes, dry and slippery, and with the agent injected at
        ///     every one of the 48 cells: the scaled tiles, the cliff's edge and middle tiles, the letters, the flag, the agent.
        /// </summary>
        [TestMethod]
        public void GymNet_Frames_MatchGymnasium() {
            foreach ((string text, Dictionary<string, object?> kwargs, Func<string?, CliffWalkingEnv> create) in Variants) {
                CliffWalkingEnv dotnet = create("rgb_array");
                using var pair = new GymPair(PyGymEnv.Make("CliffWalking-v1", kwargs, renderMode: "rgb_array"), new GymNetEnv(dotnet), stateAttribute: "s");
                PairRun run = pair.Run(9, ActionSources.Sampled(9), maxSteps: 30, frameEvery: 1);
                TestContext.WriteLine($"{text}: {run}");
                if (text != "is_slippery=False")
                    continue;
                for (long s = 0; s < dotnet.NS; s++) {
                    long state = s;
                    pair.Python.WithUnwrapped(env => {
                        using PyObject value = PyConvert.ToPython(state);
                        Python.gymnasium.envs.toy_text.cliffwalking.CliffWalkingEnv.s(env, value);
                    });
                    dotnet.S = state;
                    pair.CompareFrame($"agent at s={state}");
                }
            }
        }

        /// <summary>
        ///     Rendering before the first reset: <c>ansi</c> raises <c>AttributeError</c> at once, <c>rgb_array</c> after drawing
        ///     the first cell's ground; after a reset both render normally.
        /// </summary>
        [TestMethod]
        public void GymNet_RenderBeforeReset_RaisesAsGymnasium() {
            using PyRef python = PyRef.Of(() => {
                using PyObject made = Python.gymnasium.make("CliffWalking-v1", render_mode: "rgb_array");
                return Python.gymnasium.Env.unwrapped(made);
            });
            var dotnet = new CliffWalkingEnv(renderMode: "rgb_array");
            ByteContract.AssertSame("rgb_array render() before reset", Outcome.CapturePython(() => Python.gymnasium.Env.render(python.Object)),
                Outcome.Capture(() => ((RenderResult.Rgb)dotnet.Render()).Frame));
        }

        /// <summary>The spaces: <c>Discrete(48)</c> observations and <c>Discrete(4)</c> actions with seeded samples, both ids.</summary>
        [TestMethod]
        public void GymNet_Spaces_MatchGymnasium() {
            foreach (string id in new[] { "CliffWalking-v1", "CliffWalkingSlippery-v1" }) {
                IEnv env = Gymnasium.Make(id).Unwrapped;
                using PyRef observation = PyRef.Of(() => CartPoleLiveParityTests.MadeSpace(id, Python.gymnasium.Env.observation_space));
                var space = (Discrete)env.ObservationSpace;
                ByteContract.AssertSame($"{id} observation_space.n", Outcome.CapturePython(() => Python.gymnasium.spaces.discrete.Discrete.n(observation.Object)), Outcome.Returned(ValueFacts.Of(space.N)));
                ClassicSpaces.AssertActionSpaceSame(id, env.ActionSpace);
            }
        }

        /// <summary><c>metadata</c>: <c>{'render_modes': ['human', 'rgb_array', 'ansi'], 'render_fps': 4}</c>.</summary>
        [TestMethod]
        public void GymNet_Metadata_MatchesGymnasium() {
            Outcome python = Outcome.CapturePython(() => {
                using PyObject made = Python.gymnasium.make("CliffWalking-v1");
                using PyObject unwrapped = Python.gymnasium.Env.unwrapped(made);
                return Python.gymnasium.Env.metadata(unwrapped);
            });
            ByteContract.AssertSame("CliffWalking metadata", python, Outcome.Returned(ValueFacts.Of(new CliffWalkingEnv().Metadata)));
        }

        /// <summary><c>make</c>'s whole stack for both ids (no time limit) against <c>Gymnasium.Make</c> with 300 sampled actions.</summary>
        [TestMethod]
        public void GymNet_MakeStack_MatchesGymnasium() {
            foreach (string id in new[] { "CliffWalking-v1", "CliffWalkingSlippery-v1" }) {
                using var pair = new GymPair(PyGymEnv.Make(id, viaMake: true), new GymNetEnv(Gymnasium.Make(id)), stateAttribute: "s");
                PairRun run = pair.Run(11, ActionSources.Sampled(11), maxSteps: 300);
                TestContext.WriteLine(run.ToString());
            }
        }

        /// <summary>Gym.NET's <c>ansi</c> render as its text.</summary>
        /// <param name="env">The env, built with <c>render_mode="ansi"</c>.</param>
        /// <returns>The text.</returns>
        /// <exception cref="InvalidCastException">Thrown when the render is not text.</exception>
        private static string AnsiText(CliffWalkingEnv env) => ((RenderResult.Ansi)env.Render()).Text;
    }
}
