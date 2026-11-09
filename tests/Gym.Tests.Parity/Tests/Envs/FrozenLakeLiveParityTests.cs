using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Gym.Environments.Envs.ToyText;
using Gym.Envs;
using Gym.Observations;
using Gym.Spaces;
using Gym.Tests.Parity.Frames;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Pair;
using Gym.Tests.Parity.Session;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace Gym.Tests.Parity.Tests.Envs {
    /// <summary>
    ///     FrozenLake-v1 and FrozenLake8x8-v1 against Gymnasium's <c>FrozenLakeEnv</c>, live and in lockstep with Gym.NET's
    ///     <see cref="FrozenLakeEnv"/>: the transition table, resets, episodes on slippery and solid ice, other success rates
    ///     and reward schedules, custom and random maps, Python's dict and list lookups of every kind of action, the
    ///     <c>ansi</c> text and the <c>rgb_array</c> frames, spaces, metadata and <c>make</c>'s stack.
    /// </summary>
    /// <remarks>
    ///     Every lockstep compares <c>s</c> after each call. The dynamics are table lookups and one generator draw per call,
    ///     so nothing here depends on the platform's math. Reference:
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/frozen_lake.py#L83">gymnasium/envs/toy_text/frozen_lake.py</see>.
    /// </remarks>
    [TestClass]
    public sealed class FrozenLakeLiveParityTests : ParityTestBase {
        /// <summary>The constructor arguments compared, each in Python spelling and as Gym.NET's constructor call.</summary>
        private static readonly (string Python, Dictionary<string, object?> Kwargs, Func<string?, FrozenLakeEnv> DotNet)[] Variants = {
            ("map_name='4x4'", new Dictionary<string, object?> { ["map_name"] = "4x4" }, mode => new FrozenLakeEnv(renderMode: mode, mapName: "4x4")),
            ("map_name='8x8'", new Dictionary<string, object?> { ["map_name"] = "8x8" }, mode => new FrozenLakeEnv(renderMode: mode, mapName: "8x8")),
            ("is_slippery=False", new Dictionary<string, object?> { ["is_slippery"] = false }, mode => new FrozenLakeEnv(renderMode: mode, isSlippery: false)),
            ("success_rate=0.75", new Dictionary<string, object?> { ["success_rate"] = 0.75 }, mode => new FrozenLakeEnv(renderMode: mode, successRate: 0.75)),
            ("reward_schedule=(10.0, -1.0, -0.25)", new Dictionary<string, object?> { ["reward_schedule"] = new object?[] { 10.0, -1.0, -0.25 } },
                mode => new FrozenLakeEnv(renderMode: mode, rewardSchedule: new[] { 10.0, -1.0, -0.25 })),
            ("desc=['SFFG']", new Dictionary<string, object?> { ["desc"] = new List<object?> { "SFFG" } }, mode => new FrozenLakeEnv(renderMode: mode, desc: new[] { "SFFG" })),
            ("desc=['SFH', 'FFF', 'HFG']", new Dictionary<string, object?> { ["desc"] = new List<object?> { "SFH", "FFF", "HFG" } },
                mode => new FrozenLakeEnv(renderMode: mode, desc: new[] { "SFH", "FFF", "HFG" })),
            ("desc=9 columns x 2 rows", new Dictionary<string, object?> { ["desc"] = new List<object?> { "SFFFHFFFF", "FFHFFFFFG" } },
                mode => new FrozenLakeEnv(renderMode: mode, desc: new[] { "SFFFHFFFF", "FFHFFFFFG" })),
            ("desc=16 rows x 2 columns", new Dictionary<string, object?> { ["desc"] = new List<object?>(new object?[] { "SF" }.Concat(Enumerable.Repeat<object?>("FF", 14)).Append("FG")) },
                mode => new FrozenLakeEnv(renderMode: mode, desc: new[] { "SF" }.Concat(Enumerable.Repeat("FF", 14)).Append("FG").ToArray())),
        };

        /// <summary>The transition table <c>P</c>: every state's every action's outcomes, probability, next state, reward and flag, for each variant.</summary>
        [TestMethod]
        public void GymNet_TransitionTable_MatchesGymnasium() {
            foreach ((string text, Dictionary<string, object?> kwargs, Func<string?, FrozenLakeEnv> create) in Variants) {
                ValueFacts python = Outcome.CapturePython(() => {
                    using PyObject made = Python.gymnasium.make("FrozenLake-v1", kwargs: kwargs);
                    using PyObject unwrapped = Python.gymnasium.Env.unwrapped(made);
                    // Each outcome as (float(p), int(s), float(r), bool(t)): the table's values, independent of their Python types.
                    const string table = "rows = [(s, a, float(p), int(n), float(r), bool(t)) for s in sorted(env.P) for a in sorted(env.P[s]) for (p, n, r, t) in env.P[s][a]]\n";
                    return Python.evalStmts(table, "rows", ("env", unwrapped));
                }).Expect($"FrozenLake({text}).P");
                FrozenLakeEnv dotnet = create(null);
                var rows = new List<object?>();
                foreach (long s in dotnet.P.Keys.Order())
                    foreach (long a in dotnet.P[s].Keys.Order())
                        foreach (ToyTextTransition t in dotnet.P[s][a])
                            rows.Add(new object?[] { s, a, t.Probability, t.NextState, t.Reward, t.Terminated });
                ByteContract.AssertSame($"FrozenLake({text}).P", Outcome.Returned(python), Outcome.Returned(ValueFacts.Of(rows)));
            }
        }

        /// <summary>Episodes of seeded sampled actions, 120 steps each (past every termination), for every variant, two seeds each.</summary>
        [TestMethod]
        public void GymNet_SampledEpisodes_MatchGymnasium() {
            foreach ((string text, Dictionary<string, object?> kwargs, Func<string?, FrozenLakeEnv> create) in Variants) {
                foreach (long seed in new long[] { 0, 42 }) {
                    using var pair = new GymPair(PyGymEnv.Make("FrozenLake-v1", kwargs), new GymNetEnv(create(null)), stateAttribute: "s");
                    PairRun run = pair.Run(seed, ActionSources.Sampled(seed), maxSteps: 120);
                    TestContext.WriteLine($"{text}: {run}");
                    // The table keeps a terminal state's self-loop, so stepping on after termination is compared too.
                    for (int i = 0; i < 3 && run.Ended; i++) {
                        using var action = ParityAction.Discrete(i % 4);
                        pair.Step(action);
                    }
                }
            }
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
                ("2**70", () => PyConvert.ToPython(BigInteger.Pow(2, 70)), () => BigInteger.Pow(2, 70)),
                ("'1'", () => PyConvert.ToPython("1"), () => "1"),
                ("None", () => PyConvert.ToPython(null), () => null),
                ("np.array([1])", () => {
                    using NDArray one = np.array(new[] { 1L });
                    return one.ToNumpyCopy();
                }, () => np.array(new[] { 1L })),
            };

            using PyRef python = PyRef.Of(() => {
                using PyObject made = Python.gymnasium.make("FrozenLake-v1");
                return Python.gymnasium.Env.unwrapped(made);
            });
            var dotnet = new FrozenLakeEnv();
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
                AssertStepSame($"step({text})", pythonStep, Outcome.Capture(() => StepTuple(untyped.Step(value()))));
                ByteContract.AssertSame($"s after step({text})", Outcome.CapturePython(() => Python.gymnasium.envs.toy_text.frozen_lake.FrozenLakeEnv.s(python.Object)),
                    Outcome.Returned(ValueFacts.Of(dotnet.S)));
            }
        }

        /// <summary>
        ///     The <c>ansi</c> text after the reset and after every step of seeded episodes (the last action's caption, the agent in
        ///     red), byte for byte, and its errors before a reset and after a float action.
        /// </summary>
        [TestMethod]
        public void GymNet_AnsiText_MatchesGymnasium() {
            foreach (string mapName in new[] { "4x4", "8x8" }) {
                var kwargs = new Dictionary<string, object?> { ["map_name"] = mapName };
                using PyRef python = PyRef.Of(() => {
                    using PyObject made = Python.gymnasium.make("FrozenLake-v1", render_mode: "ansi", kwargs: kwargs);
                    return Python.gymnasium.Env.unwrapped(made);
                });
                var dotnet = new FrozenLakeEnv(renderMode: "ansi", mapName: mapName);
                ByteContract.AssertSame($"{mapName}: render() before reset", Outcome.CapturePython(() => Python.gymnasium.Env.render(python.Object)), Outcome.Capture(() => AnsiText(dotnet)));
                Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, 5)).Expect("reset(seed=5)");
                ((IDisposable)dotnet.Reset(5L).Observation).Dispose();
                ByteContract.AssertSame($"{mapName}: render() after reset", Outcome.CapturePython(() => Python.gymnasium.Env.render(python.Object)), Outcome.Capture(() => AnsiText(dotnet)));
                for (int i = 0; i < 25; i++) {
                    long action = (i * 7 + 3) % 4;
                    Outcome.CapturePython(() => {
                        using PyObject a = PyConvert.ToPython(action);
                        return Python.gymnasium.Env.step(python.Object, a);
                    }).Expect($"step({action})");
                    using (NDArray a = NDArray.Scalar<long>(action))
                        ((IDisposable)dotnet.Step(a).Observation).Dispose();
                    ByteContract.AssertSame($"{mapName}: render() after step {i + 1}", Outcome.CapturePython(() => Python.gymnasium.Env.render(python.Object)), Outcome.Capture(() => AnsiText(dotnet)));
                }

                // A float action is a key (2.0 == 2) but not a list index: the caption fails at render time.
                Outcome.CapturePython(() => {
                    using PyObject a = PyConvert.ToPython(2.0);
                    return Python.gymnasium.Env.step(python.Object, a);
                }).Expect("step(2.0)");
                (((IEnv)dotnet).Step(2.0).Observation as IDisposable)?.Dispose();
                ByteContract.AssertSame($"{mapName}: render() after step(2.0)", Outcome.CapturePython(() => Python.gymnasium.Env.render(python.Object)), Outcome.Capture(() => AnsiText(dotnet)));
            }
        }

        /// <summary>
        ///     <c>rgb_array</c> frames after the reset and along seeded episodes, for every variant: the tiles scaled to each map's
        ///     cell size, the overlays sized by Python's <c>round</c> and placed at truncated positions, the grid, the agent.
        /// </summary>
        [TestMethod]
        public void GymNet_Frames_MatchGymnasium() {
            foreach ((string text, Dictionary<string, object?> kwargs, Func<string?, FrozenLakeEnv> create) in Variants) {
                using var pair = new GymPair(PyGymEnv.Make("FrozenLake-v1", kwargs, renderMode: "rgb_array"), new GymNetEnv(create("rgb_array")), stateAttribute: "s");
                PairRun run = pair.Run(7, ActionSources.Sampled(7), maxSteps: 30, frameEvery: 1);
                TestContext.WriteLine($"{text}: {run}");
            }
        }

        /// <summary>
        ///     <c>generate_random_map(size, p, seed)</c> against <see cref="FrozenLakeEnv.GenerateRandomMap"/>: the same map for
        ///     each seed, size and frozen probability (redraws until a path exists included), and the same refusals.
        /// </summary>
        [TestMethod]
        public void GymNet_GenerateRandomMap_MatchesGymnasium() {
            var cases = new (long Size, double P, long Seed)[] { (8, 0.8, 0), (8, 0.8, 1), (4, 0.5, 2), (5, 0.3, 3), (12, 0.9, 4), (3, 1.0, 5), (6, 2.0, 6), (6, -0.5, 7) };
            foreach ((long size, double p, long seed) in cases) {
                Outcome python = Outcome.CapturePython(() => Python.gymnasium.envs.toy_text.frozen_lake.generate_random_map(size, p, seed));
                Outcome dotnet = Outcome.Capture(() => FrozenLakeEnv.GenerateRandomMap((int)size, p, seed).ToList());
                ByteContract.AssertSame($"generate_random_map(size={size}, p={p}, seed={seed})", python, dotnet);
            }
        }

        /// <summary>The spaces: <c>Discrete(nrow * ncol)</c> observations and <c>Discrete(4)</c> actions with seeded samples, both maps.</summary>
        [TestMethod]
        public void GymNet_Spaces_MatchGymnasium() {
            foreach (string id in new[] { "FrozenLake-v1", "FrozenLake8x8-v1" }) {
                IEnv env = Gymnasium.Make(id).Unwrapped;
                using PyRef observation = PyRef.Of(() => CartPoleLiveParityTests.MadeSpace(id, Python.gymnasium.Env.observation_space));
                var space = (Discrete)env.ObservationSpace;
                ByteContract.AssertSame($"{id} observation_space.n", Outcome.CapturePython(() => Python.gymnasium.spaces.discrete.Discrete.n(observation.Object)), Outcome.Returned(ValueFacts.Of(space.N)));
                ClassicSpaces.AssertActionSpaceSame(id, env.ActionSpace);
            }
        }

        /// <summary><c>metadata</c>: <c>{'render_modes': ['human', 'ansi', 'rgb_array'], 'render_fps': 4}</c>.</summary>
        [TestMethod]
        public void GymNet_Metadata_MatchesGymnasium() {
            Outcome python = Outcome.CapturePython(() => {
                using PyObject made = Python.gymnasium.make("FrozenLake-v1");
                using PyObject unwrapped = Python.gymnasium.Env.unwrapped(made);
                return Python.gymnasium.Env.metadata(unwrapped);
            });
            ByteContract.AssertSame("FrozenLake metadata", python, Outcome.Returned(ValueFacts.Of(new FrozenLakeEnv().Metadata)));
        }

        /// <summary>The map attribute <c>desc</c>: NumPy's <c>(nrow, ncol)</c> <c>S1</c> array, byte for byte, for each variant.</summary>
        [TestMethod]
        public void GymNet_Desc_MatchesGymnasium() {
            foreach ((string text, Dictionary<string, object?> kwargs, Func<string?, FrozenLakeEnv> create) in Variants) {
                Outcome python = Outcome.CapturePython(() => {
                    using PyObject made = Python.gymnasium.make("FrozenLake-v1", kwargs: kwargs);
                    using PyObject unwrapped = Python.gymnasium.Env.unwrapped(made);
                    return Python.gymnasium.envs.toy_text.frozen_lake.FrozenLakeEnv.desc(unwrapped);
                });
                using NDArray desc = create(null).Desc;
                ByteContract.AssertSame($"FrozenLake({text}).desc", python, Outcome.Returned(ValueFacts.Of(desc)));
            }
        }

        /// <summary><c>make</c>'s whole stack for both ids against <c>Gymnasium.Make</c> with sampled actions through truncation (100 and 200 steps).</summary>
        [TestMethod]
        public void GymNet_MakeStack_MatchesGymnasiumThroughTruncation() {
            foreach (string id in new[] { "FrozenLake-v1", "FrozenLake8x8-v1" }) {
                using var pair = new GymPair(PyGymEnv.Make(id, viaMake: true), new GymNetEnv(Gymnasium.Make(id)), stateAttribute: "s");
                PairRun run = pair.Run(11, ActionSources.Sampled(11), maxSteps: 300);
                TestContext.WriteLine(run.ToString());
            }
        }

        /// <summary>A Gym.NET step as the 5-tuple Gymnasium returns, for describing it like Python's.</summary>
        /// <param name="step">The step.</param>
        /// <returns><c>(observation, reward, terminated, truncated, info)</c>.</returns>
        internal static object?[] StepTuple(StepResult<object?> step) => new[] { step.Observation, step.Reward, step.Terminated, step.Truncated, step.Info };

        /// <summary>
        ///     Compares two step outcomes as <see cref="GymPair"/> does: an exception exactly, a 5-tuple field by field with the
        ///     reward as a float64 value (Gymnasium's toy_text rewards are Python ints, Gym.NET's doubles).
        /// </summary>
        /// <param name="what">The call, for the report.</param>
        /// <param name="python">Gymnasium's outcome.</param>
        /// <param name="dotnet">Gym.NET's outcome, its value from <see cref="StepTuple"/>.</param>
        /// <exception cref="ParityException">Thrown at the first difference.</exception>
        internal static void AssertStepSame(string what, Outcome python, Outcome dotnet) {
            if (python.Raised != null || dotnet.Raised != null) {
                ByteContract.AssertSame(what, python, dotnet);
                return;
            }

            string[] fields = { "observation", "reward", "terminated", "truncated", "info" };
            for (int i = 0; i < fields.Length; i++) {
                FieldPolicy policy = i == 1 ? FieldPolicy.Float64Value : FieldPolicy.Exact;
                ByteContract.AssertSame($"{what}: {fields[i]}", Outcome.Returned(python.Value!.Items[i]), Outcome.Returned(dotnet.Value!.Items[i]), policy);
            }
        }

        /// <summary>Gym.NET's <c>ansi</c> render as its text.</summary>
        /// <param name="env">The env, built with <c>render_mode="ansi"</c>.</param>
        /// <returns>The text.</returns>
        /// <exception cref="InvalidCastException">Thrown when the render is not text.</exception>
        private static string AnsiText(FrozenLakeEnv env) => ((RenderResult.Ansi)env.Render()).Text;
    }
}
