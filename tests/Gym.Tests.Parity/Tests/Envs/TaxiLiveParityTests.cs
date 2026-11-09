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
    ///     Taxi-v4 against Gymnasium's <c>TaxiEnv</c>, live and in lockstep with Gym.NET's <see cref="TaxiEnv"/>: the
    ///     transition table dry and rainy, the start distribution, episodes, the fickle passenger's draws, Python's dict and
    ///     list lookups of every kind of action, <c>encode</c>/<c>decode</c>/<c>action_mask</c> over every state, the
    ///     <c>ansi</c> text at every state, the <c>rgb_array</c> frames with every taxi sprite, spaces, metadata and
    ///     <c>make</c>'s stack.
    /// </summary>
    /// <remarks>
    ///     Every lockstep compares <c>s</c> after each call. The dynamics are table lookups and generator draws, so nothing
    ///     here depends on the platform's math. Reference:
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/taxi.py#L35">gymnasium/envs/toy_text/taxi.py</see>.
    /// </remarks>
    [TestClass]
    public sealed class TaxiLiveParityTests : ParityTestBase {
        /// <summary>The constructor arguments compared, each in Python spelling and as Gym.NET's constructor call.</summary>
        private static readonly (string Python, Dictionary<string, object?> Kwargs, Func<string?, TaxiEnv> DotNet)[] Variants = {
            ("dry", new Dictionary<string, object?>(), mode => new TaxiEnv(renderMode: mode)),
            ("is_rainy=True", new Dictionary<string, object?> { ["is_rainy"] = true }, mode => new TaxiEnv(renderMode: mode, isRainy: true)),
            ("is_rainy=True, rainy_probability=0.6", new Dictionary<string, object?> { ["is_rainy"] = true, ["rainy_probability"] = 0.6 },
                mode => new TaxiEnv(renderMode: mode, isRainy: true, rainyProbability: 0.6)),
            ("fickle_passenger=True", new Dictionary<string, object?> { ["fickle_passenger"] = true }, mode => new TaxiEnv(renderMode: mode, ficklePassenger: true)),
            ("fickle_passenger=True, fickle_probability=1.0, is_rainy=True", new Dictionary<string, object?> { ["fickle_passenger"] = true, ["fickle_probability"] = 1.0, ["is_rainy"] = true },
                mode => new TaxiEnv(renderMode: mode, ficklePassenger: true, fickleProbability: 1.0, isRainy: true)),
        };

        /// <summary>The transition table <c>P</c> (every state's every action's outcomes) and <c>initial_state_distrib</c>, for each variant.</summary>
        [TestMethod]
        public void GymNet_TransitionTable_MatchesGymnasium() {
            foreach ((string text, Dictionary<string, object?> kwargs, Func<string?, TaxiEnv> create) in Variants) {
                ValueFacts python = Outcome.CapturePython(() => {
                    using PyObject made = Python.gymnasium.make("Taxi-v4", kwargs: kwargs);
                    using PyObject unwrapped = Python.gymnasium.Env.unwrapped(made);
                    // Each outcome as (float(p), int(s), float(r), bool(t)): the table's values, independent of their Python types.
                    const string table = "rows = [(s, a, float(p), int(n), float(r), bool(t)) for s in sorted(env.P) for a in sorted(env.P[s]) for (p, n, r, t) in env.P[s][a]]\n";
                    return Python.evalStmts(table, "rows", ("env", unwrapped));
                }).Expect($"Taxi({text}).P");
                TaxiEnv dotnet = create(null);
                var rows = new List<object?>();
                foreach (long s in dotnet.P.Keys.Order())
                    foreach (long a in dotnet.P[s].Keys.Order())
                        foreach (ToyTextTransition t in dotnet.P[s][a])
                            rows.Add(new object?[] { s, a, t.Probability, t.NextState, t.Reward, t.Terminated });
                ByteContract.AssertSame($"Taxi({text}).P", Outcome.Returned(python), Outcome.Returned(ValueFacts.Of(rows)));

                Outcome distribution = Outcome.CapturePython(() => {
                    using PyObject made = Python.gymnasium.make("Taxi-v4", kwargs: kwargs);
                    using PyObject unwrapped = Python.gymnasium.Env.unwrapped(made);
                    return Python.gymnasium.envs.toy_text.taxi.TaxiEnv.initial_state_distrib(unwrapped);
                });
                using NDArray initial = np.array(dotnet.InitialStateDistrib);
                ByteContract.AssertSame($"Taxi({text}).initial_state_distrib", distribution, Outcome.Returned(ValueFacts.Of(initial)));
            }
        }

        /// <summary>Episodes of seeded sampled actions, 200 steps each, for every variant (the fickle resets' extra draw included), two seeds each.</summary>
        [TestMethod]
        public void GymNet_SampledEpisodes_MatchGymnasium() {
            foreach ((string text, Dictionary<string, object?> kwargs, Func<string?, TaxiEnv> create) in Variants) {
                foreach (long seed in new long[] { 0, 42 }) {
                    using var pair = new GymPair(PyGymEnv.Make("Taxi-v4", kwargs), new GymNetEnv(create(null)), stateAttribute: "s");
                    PairRun run = pair.Run(seed, ActionSources.Sampled(seed), maxSteps: 200);
                    TestContext.WriteLine($"{text}: {run}");
                    ByteContract.AssertSame($"{text}: fickle_step after seed {seed}", pair.Python.GetAttribute("fickle_step"), Outcome.Returned(pair.DotNet.GetAttribute("fickle_step")!));
                }
            }
        }

        /// <summary>
        ///     A delivery driven to the end, dry: from an injected state the taxi drives to the passenger, picks them up, drives to
        ///     the destination and drops them off (20, terminated); a wrong pickup and a wrong dropoff (-10) on the way.
        /// </summary>
        [TestMethod]
        public void GymNet_Delivery_MatchesGymnasium() {
            TaxiEnv dotnet = new();
            using var pair = new GymPair(PyGymEnv.Make("Taxi-v4"), new GymNetEnv(dotnet), stateAttribute: "s");
            pair.Reset(1);
            // Taxi at (2, 2), passenger at R (0, 0), destination B (4, 3).
            Inject(pair, dotnet, dotnet.Encode(2, 2, 0, 3));
            // Wrong pickup and dropoff, then north twice and west twice (row 2 has no walls) to R, pickup, then to B.
            long[] actions = { 4, 5, 1, 1, 3, 3, 4, 5, 0, 0, 2, 2, 0, 0, 2, 5 };
            foreach (long action in actions) {
                using var parity = ParityAction.Discrete(action);
                pair.Step(parity);
            }
        }

        /// <summary>
        ///     The fickle passenger: with <c>fickle_step</c> armed and the passenger aboard, a move draws the new destination
        ///     (<c>np_random.choice</c> over the other three); a pickup, a dropoff or a bump into a wall draws nothing; a passenger
        ///     waiting at a stand draws nothing. Every taxi cell and destination, dry and rainy.
        /// </summary>
        [TestMethod]
        public void GymNet_FicklePassenger_MatchesGymnasium() {
            foreach (bool rainy in new[] { false, true }) {
                var kwargs = new Dictionary<string, object?> { ["fickle_passenger"] = true, ["is_rainy"] = rainy };
                var dotnet = new TaxiEnv(ficklePassenger: true, isRainy: rainy);
                using var pair = new GymPair(PyGymEnv.Make("Taxi-v4", kwargs), new GymNetEnv(dotnet), stateAttribute: "s");
                pair.Reset(21);
                for (long row = 0; row < 5; row++) {
                    for (long col = 0; col < 5; col++) {
                        for (long dest = 0; dest < 4; dest++) {
                            foreach (long passLoc in new long[] { 4, (dest + 1) % 4 }) {
                                foreach (long action in new long[] { row < 4 ? 0 : 1, 2, 3, 4, 5 }) {
                                    Inject(pair, dotnet, dotnet.Encode(row, col, passLoc, dest), fickleStep: true);
                                    using var parity = ParityAction.Discrete(action);
                                    pair.Step(parity);
                                    ByteContract.AssertSame($"fickle_step after ({row}, {col}, {passLoc}, {dest}) step({action})", pair.Python.GetAttribute("fickle_step"),
                                        Outcome.Returned(pair.DotNet.GetAttribute("fickle_step")!));
                                }
                            }
                        }
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
                ("5", () => PyConvert.ToPython(5L), () => 5L),
                ("np.int64(4)", () => Python.numpy.int64(4), () => NDArray.Scalar<long>(4L)),
            };
            var refused = new (string Python, Func<PyObject> Gymnasium, Func<object?> DotNet)[] {
                ("6", () => PyConvert.ToPython(6L), () => 6L),
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
                using PyObject made = Python.gymnasium.make("Taxi-v4");
                return Python.gymnasium.Env.unwrapped(made);
            });
            var dotnet = new TaxiEnv();
            IEnv untyped = dotnet;
            foreach ((string text, Func<PyObject> gymnasium, Func<object?> value) in refused.Take(2)) {
                Outcome before = Outcome.CapturePython(() => {
                    using PyObject action = gymnasium();
                    return Python.gymnasium.Env.step(python.Object, action);
                });
                ByteContract.AssertSame($"step({text}) before reset", before, Outcome.Capture(() => untyped.Step(value())));
            }

            Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, 3)).Expect("reset(seed=3)");
            DisposeObservation(dotnet.Reset(3L).Observation);
            foreach ((string text, Func<PyObject> gymnasium, Func<object?> value) in accepted.Concat(refused)) {
                Outcome pythonStep = Outcome.CapturePython(() => {
                    using PyObject action = gymnasium();
                    return Python.gymnasium.Env.step(python.Object, action);
                });
                FrozenLakeLiveParityTests.AssertStepSame($"step({text})", pythonStep, Outcome.Capture(() => FrozenLakeLiveParityTests.StepTuple(untyped.Step(value()))));
                ByteContract.AssertSame($"s after step({text})", Outcome.CapturePython(() => Python.gymnasium.envs.toy_text.taxi.TaxiEnv.s(python.Object)),
                    Outcome.Returned(ValueFacts.Of(dotnet.S)));
            }
        }

        /// <summary><c>encode</c> of every part combination, and <c>decode</c> and <c>action_mask</c> of every state, byte for byte.</summary>
        [TestMethod]
        public void GymNet_EncodeDecodeActionMask_MatchGymnasium() {
            using PyRef python = PyRef.Of(() => {
                using PyObject made = Python.gymnasium.make("Taxi-v4");
                return Python.gymnasium.Env.unwrapped(made);
            });
            var dotnet = new TaxiEnv();
            for (long row = 0; row < 5; row++)
                for (long col = 0; col < 5; col++)
                    for (long passLoc = 0; passLoc < 5; passLoc++)
                        for (long dest = 0; dest < 4; dest++) {
                            long r = row, c = col, p = passLoc, d = dest;
                            ByteContract.AssertSame($"encode({r}, {c}, {p}, {d})", Outcome.CapturePython(() => Python.gymnasium.envs.toy_text.taxi.TaxiEnv.encode(python.Object, r, c, p, d)),
                                Outcome.Returned(ValueFacts.Of(dotnet.Encode(r, c, p, d))));
                        }

            for (long state = 0; state < 500; state++) {
                long s = state;
                ByteContract.AssertSame($"decode({s})", Outcome.CapturePython(() => Python.gymnasium.envs.toy_text.taxi.TaxiEnv.decode(python.Object, s)),
                    Outcome.Capture(() => dotnet.Decode(s)));
                ByteContract.AssertSame($"action_mask({s})", Outcome.CapturePython(() => Python.gymnasium.envs.toy_text.taxi.TaxiEnv.action_mask(python.Object, s)),
                    Outcome.Capture(() => dotnet.ActionMask(s)));
            }
        }

        /// <summary>
        ///     <c>decode</c> of a state beyond the grid raises Gymnasium's bare <c>AssertionError</c>, whose empty message
        ///     NumSharp's parameterless <c>AssertionError</c> does not carry yet (N-1).
        /// </summary>
        [TestMethod]
        [OpenBugs("N-1")]
        public void GymNet_DecodeBeyondTheGrid_RaisesTheBareAssertion() {
            using PyRef python = PyRef.Of(() => {
                using PyObject made = Python.gymnasium.make("Taxi-v4");
                return Python.gymnasium.Env.unwrapped(made);
            });
            var dotnet = new TaxiEnv();
            foreach (long state in new long[] { 500, 1000, -1 }) {
                ByteContract.AssertSame($"decode({state})", Outcome.CapturePython(() => Python.gymnasium.envs.toy_text.taxi.TaxiEnv.decode(python.Object, state)),
                    Outcome.Capture(() => dotnet.Decode(state)));
            }
        }

        /// <summary>
        ///     The <c>ansi</c> text before a reset (Python's <c>AttributeError</c>), after the reset, along seeded episodes, at
        ///     every one of the 500 states (taxi, passenger and destination coloured, nested where they share a cell), and the
        ///     caption's <c>TypeError</c> after a float action, byte for byte.
        /// </summary>
        [TestMethod]
        public void GymNet_AnsiText_MatchesGymnasium() {
            foreach ((string text, Dictionary<string, object?> kwargs, Func<string?, TaxiEnv> create) in Variants.Take(2)) {
                using PyRef python = PyRef.Of(() => {
                    using PyObject made = Python.gymnasium.make("Taxi-v4", render_mode: "ansi", kwargs: kwargs);
                    return Python.gymnasium.Env.unwrapped(made);
                });
                TaxiEnv dotnet = create("ansi");
                ByteContract.AssertSame($"{text}: render() before reset", Outcome.CapturePython(() => Python.gymnasium.Env.render(python.Object)), Outcome.Capture(() => AnsiText(dotnet)));
                Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, 5)).Expect("reset(seed=5)");
                DisposeObservation(dotnet.Reset(5L).Observation);
                ByteContract.AssertSame($"{text}: render() after reset", Outcome.CapturePython(() => Python.gymnasium.Env.render(python.Object)), Outcome.Capture(() => AnsiText(dotnet)));
                for (long state = 0; state < 500; state++) {
                    long s = state;
                    PythonRuntime.Run(() => {
                        using PyObject value = PyConvert.ToPython(s);
                        Python.gymnasium.envs.toy_text.taxi.TaxiEnv.s(python.Object, value);
                    });
                    dotnet.S = s;
                    ByteContract.AssertSame($"{text}: render() at s={s}", Outcome.CapturePython(() => Python.gymnasium.Env.render(python.Object)), Outcome.Capture(() => AnsiText(dotnet)));
                }

                for (int i = 0; i < 30; i++) {
                    long action = (i * 5 + 2) % 6;
                    Outcome.CapturePython(() => {
                        using PyObject a = PyConvert.ToPython(action);
                        return Python.gymnasium.Env.step(python.Object, a);
                    }).Expect($"step({action})");
                    using (NDArray a = NDArray.Scalar<long>(action))
                        DisposeObservation(dotnet.Step(a).Observation);
                    ByteContract.AssertSame($"{text}: render() after step {i + 1}", Outcome.CapturePython(() => Python.gymnasium.Env.render(python.Object)), Outcome.Capture(() => AnsiText(dotnet)));
                }

                // A float action is a key (2.0 == 2) but not a list index: the caption fails at render time.
                Outcome.CapturePython(() => {
                    using PyObject a = PyConvert.ToPython(2.0);
                    return Python.gymnasium.Env.step(python.Object, a);
                }).Expect("step(2.0)");
                (((IEnv)dotnet).Step(2.0).Observation as IDisposable)?.Dispose();
                ByteContract.AssertSame($"{text}: render() after step(2.0)", Outcome.CapturePython(() => Python.gymnasium.Env.render(python.Object)), Outcome.Capture(() => AnsiText(dotnet)));
            }
        }

        /// <summary>
        ///     <c>rgb_array</c> frames after the reset and along seeded episodes (dry and rainy), at every seventh state, and with
        ///     each of the eight taxi sprites (four orientations, empty and full) after a move from the map's open middle.
        /// </summary>
        [TestMethod]
        public void GymNet_Frames_MatchGymnasium() {
            foreach ((string text, Dictionary<string, object?> kwargs, Func<string?, TaxiEnv> create) in Variants.Take(2)) {
                TaxiEnv dotnet = create("rgb_array");
                using var pair = new GymPair(PyGymEnv.Make("Taxi-v4", kwargs, renderMode: "rgb_array"), new GymNetEnv(dotnet), stateAttribute: "s");
                PairRun run = pair.Run(9, ActionSources.Sampled(9), maxSteps: 30, frameEvery: 1);
                TestContext.WriteLine($"{text}: {run}");
                if (text != "dry")
                    continue;

                for (long state = 0; state < 500; state += 7) {
                    Inject(pair, dotnet, state);
                    pair.CompareFrame($"at s={state}");
                }

                foreach (long passLoc in new long[] { 2, 4 }) {
                    for (long action = 0; action < 4; action++) {
                        Inject(pair, dotnet, dotnet.Encode(2, 2, passLoc, 1));
                        using (var parity = ParityAction.Discrete(action))
                            pair.Step(parity);
                        pair.CompareFrame($"after step({action}) with pass_loc={passLoc}");
                        ByteContract.AssertSame($"taxi_orientation after step({action})", pair.Python.GetAttribute("taxi_orientation"),
                            Outcome.Returned(pair.DotNet.GetAttribute("taxi_orientation")!));
                        // A pickup or dropoff keeps the orientation the last move gave.
                        using (var pickup = ParityAction.Discrete(4))
                            pair.Step(pickup);
                        pair.CompareFrame($"after step(4) following step({action})");
                    }
                }
            }
        }

        /// <summary>
        ///     A float movement (<c>2.0</c>) is stored as the orientation by the next render, whose sprite lookup then raises
        ///     Python's <c>TypeError</c> after the background was drawn; a pickup does not replace it, so the next render raises
        ///     too; an integer move does, and the frames match again.
        /// </summary>
        [TestMethod]
        public void GymNet_FloatMove_BreaksTheSpriteLookupAsGymnasium() {
            using PyRef python = PyRef.Of(() => {
                using PyObject made = Python.gymnasium.make("Taxi-v4", render_mode: "rgb_array");
                return Python.gymnasium.Env.unwrapped(made);
            });
            var dotnet = new TaxiEnv(renderMode: "rgb_array");
            IEnv untyped = dotnet;
            Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, 2)).Expect("reset(seed=2)");
            DisposeObservation(dotnet.Reset(2L).Observation);
            foreach ((string text, object action) in new (string, object)[] { ("2.0", 2.0), ("4", 4L), ("1", 1L) }) {
                Outcome.CapturePython(() => {
                    using PyObject a = PyConvert.ToPython(action);
                    return Python.gymnasium.Env.step(python.Object, a);
                }).Expect($"step({text})");
                (untyped.Step(action).Observation as IDisposable)?.Dispose();
                ByteContract.AssertSame($"render() after step({text})", Outcome.CapturePython(() => Python.gymnasium.Env.render(python.Object)),
                    Outcome.Capture(() => ((RenderResult.Rgb)dotnet.Render()).Frame));
            }
        }

        /// <summary>Rendering before the first reset: <c>ansi</c> and <c>rgb_array</c> raise Python's <c>AttributeError</c>.</summary>
        [TestMethod]
        public void GymNet_RenderBeforeReset_RaisesAsGymnasium() {
            using PyRef python = PyRef.Of(() => {
                using PyObject made = Python.gymnasium.make("Taxi-v4", render_mode: "rgb_array");
                return Python.gymnasium.Env.unwrapped(made);
            });
            var dotnet = new TaxiEnv(renderMode: "rgb_array");
            ByteContract.AssertSame("rgb_array render() before reset", Outcome.CapturePython(() => Python.gymnasium.Env.render(python.Object)),
                Outcome.Capture(() => ((RenderResult.Rgb)dotnet.Render()).Frame));
        }

        /// <summary>The spaces: <c>Discrete(500)</c> observations and <c>Discrete(6)</c> actions with seeded samples.</summary>
        [TestMethod]
        public void GymNet_Spaces_MatchGymnasium() {
            const string id = "Taxi-v4";
            IEnv env = Gymnasium.Make(id).Unwrapped;
            using PyRef observation = PyRef.Of(() => CartPoleLiveParityTests.MadeSpace(id, Python.gymnasium.Env.observation_space));
            var space = (Discrete)env.ObservationSpace;
            ByteContract.AssertSame($"{id} observation_space.n", Outcome.CapturePython(() => Python.gymnasium.spaces.discrete.Discrete.n(observation.Object)), Outcome.Returned(ValueFacts.Of(space.N)));
            ClassicSpaces.AssertActionSpaceSame(id, env.ActionSpace);
        }

        /// <summary><c>metadata</c>: <c>{'render_modes': ['human', 'ansi', 'rgb_array'], 'render_fps': 4}</c>.</summary>
        [TestMethod]
        public void GymNet_Metadata_MatchesGymnasium() {
            Outcome python = Outcome.CapturePython(() => {
                using PyObject made = Python.gymnasium.make("Taxi-v4");
                using PyObject unwrapped = Python.gymnasium.Env.unwrapped(made);
                return Python.gymnasium.Env.metadata(unwrapped);
            });
            ByteContract.AssertSame("Taxi metadata", python, Outcome.Returned(ValueFacts.Of(new TaxiEnv().Metadata)));
        }

        /// <summary>The map attribute <c>desc</c>: NumPy's <c>(7, 11)</c> <c>S1</c> array, byte for byte.</summary>
        [TestMethod]
        public void GymNet_Desc_MatchesGymnasium() {
            Outcome python = Outcome.CapturePython(() => {
                using PyObject made = Python.gymnasium.make("Taxi-v4");
                using PyObject unwrapped = Python.gymnasium.Env.unwrapped(made);
                return Python.gymnasium.envs.toy_text.taxi.TaxiEnv.desc(unwrapped);
            });
            using NDArray desc = new TaxiEnv().Desc;
            ByteContract.AssertSame("Taxi desc", python, Outcome.Returned(ValueFacts.Of(desc)));
        }

        /// <summary><c>make</c>'s whole stack against <c>Gymnasium.Make</c> with sampled actions through truncation at 200 steps, two seeds.</summary>
        [TestMethod]
        public void GymNet_MakeStack_MatchesGymnasiumThroughTruncation() {
            foreach (long seed in new long[] { 11, 12 }) {
                using var pair = new GymPair(PyGymEnv.Make("Taxi-v4", viaMake: true), new GymNetEnv(Gymnasium.Make("Taxi-v4")), stateAttribute: "s");
                PairRun run = pair.Run(seed, ActionSources.Sampled(seed), maxSteps: 300);
                TestContext.WriteLine(run.ToString());
            }
        }

        /// <summary>Writes the same state (and optionally an armed fickle passenger) into both envs.</summary>
        /// <param name="pair">The pair; Gymnasium's env is reached through it.</param>
        /// <param name="dotnet">Gym.NET's env in the pair.</param>
        /// <param name="state">The state.</param>
        /// <param name="fickleStep">Whether to also set <c>fickle_step = True</c>.</param>
        /// <exception cref="PythonException">Thrown when an assignment fails.</exception>
        private static void Inject(GymPair pair, TaxiEnv dotnet, long state, bool fickleStep = false) {
            pair.Python.WithUnwrapped(env => {
                using PyObject value = PyConvert.ToPython(state);
                Python.gymnasium.envs.toy_text.taxi.TaxiEnv.s(env, value);
                if (fickleStep) {
                    using PyObject armed = PyConvert.ToPython(true);
                    Python.gymnasium.envs.toy_text.taxi.TaxiEnv.fickle_step(env, armed);
                }
            });
            dotnet.S = state;
            if (fickleStep)
                dotnet.FickleStep = true;
        }

        /// <summary>Releases an observation the test does not compare.</summary>
        /// <param name="observation">The observation.</param>
        private static void DisposeObservation(NDArray observation) => observation.Dispose();

        /// <summary>Gym.NET's <c>ansi</c> render as its text.</summary>
        /// <param name="env">The env, built with <c>render_mode="ansi"</c>.</param>
        /// <returns>The text.</returns>
        /// <exception cref="InvalidCastException">Thrown when the render is not text.</exception>
        private static string AnsiText(TaxiEnv env) => ((RenderResult.Ansi)env.Render()).Text;
    }
}
