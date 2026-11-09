using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
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
    ///     MountainCar-v0 and MountainCarContinuous-v0 against Gymnasium's <c>MountainCarEnv</c> and
    ///     <c>Continuous_MountainCarEnv</c>, live and in lockstep with Gym.NET's one <see cref="MountainCarEnv"/> (DR-32): resets
    ///     and their options, episodes of sampled actions and of a momentum policy through the flag and past it, the
    ///     continuous mode's NumPy typing (float32 and float64 forces, clamps, NaN), unsigned discrete actions, injected and
    ///     malformed states, every error, spaces, metadata, <c>repr</c>, <c>make</c>'s stack, and <c>rgb_array</c> frames.
    /// </summary>
    /// <remarks>
    ///     Every lockstep compares the state's float64 values after each call (Gymnasium's state is a float64 array after a
    ///     reset, a tuple after a discrete step and a float32 array after a continuous step; see
    ///     <see cref="PyGymEnv.GetStateValues"/>). Cells that step are pinned to the reference platform: the dynamics rest on
    ///     the platform's <c>cos</c>. References: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/mountain_car.py#L16">mountain_car.py</see>,
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/continuous_mountain_car.py#L26">continuous_mountain_car.py</see>.
    /// </remarks>
    [TestClass]
    public sealed class MountainCarLiveParityTests : ParityTestBase {
        /// <summary>The discrete id.</summary>
        private const string DiscreteId = "MountainCar-v0";

        /// <summary>The continuous id.</summary>
        private const string ContinuousId = "MountainCarContinuous-v0";

        /// <summary>
        ///     States written into both envs before a step or a frame: against both walls (the left one with leftward velocity,
        ///     which the step zeroes), at and around both flags, at the speed limits, a negative zero, and NaN.
        /// </summary>
        public static readonly double[][] InjectedStates = {
            new[] { -1.2, -0.05 },
            new[] { -1.19, -0.07 },
            new[] { -1.2, 0.0 },
            new[] { 0.49, 0.07 },
            new[] { 0.44, 0.02 },
            new[] { 0.45, 0.0 },
            new[] { 0.6, 0.07 },
            new[] { -0.5236, 0.0 },
            new[] { -0.0, -0.0 },
            new[] { 0.0, 0.07 },
            new[] { -0.4, -0.07 },
            new[] { double.NaN, 0.0 },
        };

        /// <summary>The reset options compared: valid bounds, every kind of rejected bound, and bounds the generator refuses.</summary>
        private static readonly (string Python, Dictionary<string, object?> Options)[] ResetOptions = {
            ("{}", new Dictionary<string, object?>()),
            ("{'low': -0.5, 'high': -0.45}", new Dictionary<string, object?> { ["low"] = -0.5, ["high"] = -0.45 }),
            ("{'low': -0.45}", new Dictionary<string, object?> { ["low"] = -0.45 }),
            ("{'high': -0.7}", new Dictionary<string, object?> { ["high"] = -0.7 }),
            ("{'low': 'abc'}", new Dictionary<string, object?> { ["low"] = "abc" }),
            ("{'low': None}", new Dictionary<string, object?> { ["low"] = null }),
            ("{'low': nan}", new Dictionary<string, object?> { ["low"] = double.NaN }),
            ("{'low': -inf}", new Dictionary<string, object?> { ["low"] = double.NegativeInfinity }),
            ("{'low': -1, 'high': 0}", new Dictionary<string, object?> { ["low"] = -1L, ["high"] = 0L }),
            ("{'low': '-0.5', 'high': '-0.4'}", new Dictionary<string, object?> { ["low"] = "-0.5", ["high"] = "-0.4" }),
            ("{'low': -10**400}", new Dictionary<string, object?> { ["low"] = -BigInteger.Pow(10, 400) }),
            ("{'low': -0.5, 'high': -0.5}", new Dictionary<string, object?> { ["low"] = -0.5, ["high"] = -0.5 }),
            ("{'other': 1}", new Dictionary<string, object?> { ["other"] = 1L }),
        };

        /// <summary><c>reset(seed=seed)</c> on both ids: observation, info, the state's values and the generator.</summary>
        /// <param name="id">The id.</param>
        /// <param name="seed">The reset seed.</param>
        [TestMethod]
        [DataRow(DiscreteId, 0L)]
        [DataRow(DiscreteId, 1L)]
        [DataRow(DiscreteId, 42L)]
        [DataRow(ContinuousId, 0L)]
        [DataRow(ContinuousId, 1234L)]
        public void GymNet_Reset_MatchesGymnasium(string id, long seed) {
            using GymPair pair = Pair(id);
            pair.Reset(seed);
        }

        /// <summary>
        ///     Episodes of seeded sampled actions (both action spaces sampled and compared), 200 steps: in continuous mode every
        ///     force is a float32 sample, so each step exercises the float32 arithmetic of a float32 state.
        /// </summary>
        /// <param name="id">The id.</param>
        /// <param name="seed">The reset seed, also the action spaces' seed.</param>
        [TestMethod]
        [DataRow(DiscreteId, 0L)]
        [DataRow(DiscreteId, 7L)]
        [DataRow(DiscreteId, 42L)]
        [DataRow(ContinuousId, 0L)]
        [DataRow(ContinuousId, 7L)]
        [DataRow(ContinuousId, 42L)]
        public void GymNet_SampledEpisodes_MatchGymnasium(string id, long seed) {
            SkipOffReferencePlatform("MountainCar's cos");
            using GymPair pair = Pair(id);
            PairRun run = pair.Run(seed, ActionSources.Sampled(seed), maxSteps: 200);
            TestContext.WriteLine(run.ToString());
        }

        /// <summary>
        ///     The momentum policy drives the car to the flag (termination, and in continuous mode the 100 reward), then five
        ///     more steps past it, which Gymnasium does not refuse.
        /// </summary>
        /// <param name="id">The id.</param>
        /// <param name="seed">The reset seed.</param>
        [TestMethod]
        [DataRow(DiscreteId, 0L)]
        [DataRow(DiscreteId, 5L)]
        [DataRow(ContinuousId, 0L)]
        [DataRow(ContinuousId, 5L)]
        public void GymNet_MomentumEpisodes_MatchGymnasiumThroughTheFlag(string id, long seed) {
            SkipOffReferencePlatform("MountainCar's cos");
            using GymPair pair = Pair(id);
            IActionSource policy = Momentum(id);
            PairRun run = pair.Run(seed, policy, maxSteps: 999);
            Assert.IsTrue(run.Ended, $"The momentum policy must reach the flag: {run}");
            for (int i = 0; i < 5; i++) {
                using ParityAction action = policy.Next(pair);
                pair.Step(action);
            }

            TestContext.WriteLine($"{run}; then 5 more steps past the flag, identical");
        }

        /// <summary>
        ///     A <c>goal_velocity</c> above zero (a Python float), which the termination test compares against in each mode's
        ///     precision: the momentum policy runs to the flag on both sides.
        /// </summary>
        /// <param name="id">The id.</param>
        [TestMethod]
        [DataRow(DiscreteId)]
        [DataRow(ContinuousId)]
        public void GymNet_GoalVelocity_MatchesGymnasium(string id) {
            SkipOffReferencePlatform("MountainCar's cos");
            var kwargs = new Dictionary<string, object?> { ["goal_velocity"] = 0.01 };
            using var pair = new GymPair(PyGymEnv.Make(id, kwargs), new GymNetEnv(new MountainCarEnv(goalVelocity: 0.01, continuous: id == ContinuousId)), stateAttribute: "state");
            PairRun run = pair.Run(3, Momentum(id), maxSteps: 999);
            TestContext.WriteLine(run.ToString());
        }

        /// <summary>
        ///     The continuous mode's NumPy typing, one step each from the same episode: float32 forces inside and beyond the
        ///     clamp, float64 forces (a float64 state follows), integer and boolean forces, a two-element action (only its first
        ///     element counts), NumPy's errors for a 0-d and an empty action, <c>math.pow</c>'s overflow for a huge float64
        ///     force (the state stays as it was, which the next step's state comparison shows), and a NaN force last.
        /// </summary>
        [TestMethod]
        public void GymNet_ContinuousActions_MatchGymnasium() {
            SkipOffReferencePlatform("MountainCar's cos");
            using GymPair pair = Pair(ContinuousId);
            pair.Reset(3);
            var actions = new Func<NDArray>[] {
                () => np.array(new[] { 0.5f }),
                () => np.array(new[] { 5.0f }),
                () => np.array(new[] { -5.0f }),
                () => np.array(new[] { 0.3 }),
                () => np.array(new[] { -1.0 }),
                () => np.array(new[] { 1L }),
                () => np.array(new[] { true }),
                () => np.array(new[] { 0.25f, 0.75f }),
                () => NDArray.Scalar<float>(0.5f),
                () => np.array(Array.Empty<float>()),
                () => np.array(new[] { 1e200 }),
                () => np.array(new[] { 0.125f }),
                () => np.array(new[] { 1.0f }),
                () => np.array(new[] { float.NaN }),
                () => np.array(new[] { 0.5f }),
            };
            foreach (Func<NDArray> action in actions) {
                using var step = ParityAction.Continuous(action());
                pair.Step(step);
            }
        }

        /// <summary>
        ///     Unsigned discrete actions: an unsigned 0 is a valid <c>Discrete(3)</c> element, and NumPy's <c>action - 1</c>
        ///     wraps it to the type's maximum, so the car is pushed right at full speed, on both sides.
        /// </summary>
        [TestMethod]
        public void GymNet_UnsignedActions_WrapLikeNumPy() {
            SkipOffReferencePlatform("MountainCar's cos");
            using GymPair pair = Pair(DiscreteId);
            pair.Reset(11);
            var actions = new Func<NDArray>[] {
                () => NDArray.Scalar<byte>(0), () => NDArray.Scalar<ushort>(0), () => NDArray.Scalar<uint>(0), () => NDArray.Scalar<ulong>(0),
                () => NDArray.Scalar<byte>(2), () => NDArray.Scalar<ulong>(1), () => NDArray.Scalar<sbyte>(0), () => NDArray.Scalar<int>(2),
            };
            foreach (Func<NDArray> action in actions) {
                // NumPy scalars on Gymnasium's side (np.uint16(0)), which a 0-d NumSharp array stands for.
                using var step = ParityAction.Scalar(action());
                pair.Step(step);
            }
        }

        /// <summary>
        ///     <c>reset(seed=..., options=...)</c> for every case of <see cref="ResetOptions"/>, seeded and unseeded, on both ids:
        ///     the result or the exception, then the state's values and the generator state.
        /// </summary>
        /// <param name="id">The id.</param>
        [TestMethod]
        [DataRow(DiscreteId)]
        [DataRow(ContinuousId)]
        public void GymNet_ResetOptions_MatchGymnasium(string id) {
            using PyRef python = PyRef.Of(() => {
                using PyObject made = Python.gymnasium.make(id);
                return Python.gymnasium.Env.unwrapped(made);
            });
            MountainCarEnv dotnet = Create(id);
            long seed = 100;
            foreach ((string text, Dictionary<string, object?> options) in ResetOptions) {
                foreach (bool seeded in new[] { true, false }) {
                    long? resetSeed = seeded ? seed++ : null;
                    if (!seeded) {
                        // An unseeded reset draws from the generator as it stands: bring both sides to the same one first.
                        Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, seed)).Expect($"reset(seed={seed})");
                        dotnet.Reset(seed).Observation.Dispose();
                        seed++;
                    }

                    string call = $"{id}: reset(seed={(resetSeed?.ToString(CultureInfo.InvariantCulture) ?? "None")}, options={text})";
                    Outcome pythonReset = Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, resetSeed, options));
                    Outcome dotnetReset = Outcome.Capture(() => {
                        var (observation, info) = dotnet.Reset(resetSeed, options);
                        return (observation, info);
                    });
                    ByteContract.AssertSame(call, pythonReset, dotnetReset);
                    ByteContract.AssertSame($"{call}: state", PythonState(python.Object), Outcome.Returned(StateFacts(dotnet.State)));
                    ByteContract.AssertSame($"{call}: np_random state", Outcome.CapturePython(() => {
                        using PyObject generator = Python.gymnasium.Env.np_random(python.Object);
                        using PyObject bitGenerator = Python.numpy.random.Generator.bit_generator(generator);
                        return Python.numpy.random.BitGenerator.state(bitGenerator);
                    }), Outcome.Returned(RngFacts.Of(dotnet.NpRandom)));
                }
            }
        }

        /// <summary>
        ///     Actions outside <c>Discrete(3)</c> raise <c>AssertionError</c> with Python's rendering of the action and its type,
        ///     before the state is read: ints, floats, NumPy scalars, an array, a string, <c>None</c>, big ints, a NumPy bool.
        /// </summary>
        [TestMethod]
        public void GymNet_InvalidActions_RaiseGymnasiumsAssertion() {
            var cases = new (string Python, Func<PyObject> Gymnasium, Func<object?> DotNet)[] {
                ("3", () => PyConvert.ToPython(3L), () => 3L),
                ("-1", () => PyConvert.ToPython(-1L), () => -1L),
                ("0.5", () => PyConvert.ToPython(0.5), () => 0.5),
                ("2.0", () => PyConvert.ToPython(2.0), () => 2.0),
                ("np.int64(5)", () => Python.numpy.int64(5), () => NDArray.Scalar<long>(5L)),
                ("np.float64(1.0)", () => Python.numpy.float64(1.0), () => NDArray.Scalar<double>(1.0)),
                ("np.float32(1.0)", () => Python.numpy.float32(1.0f), () => 1.0f),
                ("np.True_", () => {
                    using NDArray flag = NDArray.Scalar<bool>(true);
                    using PyObject array = flag.ToNumpyCopy();
                    // The 0-d array's item, np.True_, the NumPy scalar a 0-d NumSharp array stands for.
                    return Python.evalStmts("scalar = array[()]\n", "scalar", ("array", array));
                }, () => NDArray.Scalar<bool>(true)),
                ("np.array([1])", () => {
                    using NDArray one = np.array(new[] { 1L });
                    return one.ToNumpyCopy();
                }, () => np.array(new[] { 1L })),
                ("'1'", () => PyConvert.ToPython("1"), () => "1"),
                ("None", () => PyConvert.ToPython(null), () => null),
                ("2**70", () => PyConvert.ToPython(BigInteger.Pow(2, 70)), () => BigInteger.Pow(2, 70)),
            };

            using PyRef python = PyRef.Of(() => {
                using PyObject made = Python.gymnasium.make(DiscreteId);
                return Python.gymnasium.Env.unwrapped(made);
            });
            Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, 0)).Expect("reset(seed=0)");
            var dotnet = new MountainCarEnv();
            IEnv untyped = dotnet;
            dotnet.Reset(0L).Observation.Dispose();

            foreach ((string text, Func<PyObject> gymnasium, Func<object?> value) in cases) {
                Outcome pythonStep = Outcome.CapturePython(() => {
                    using PyObject action = gymnasium();
                    return Python.gymnasium.Env.step(python.Object, action);
                });
                Assert.IsNotNull(pythonStep.Raised, $"Gymnasium must reject step({text}).");
                ByteContract.AssertSame($"step({text})", pythonStep, Outcome.Capture(() => untyped.Step(value())));
            }
        }

        /// <summary>
        ///     Stepping before any reset: the state attribute does not exist yet, so a valid action raises Python's
        ///     <c>AttributeError</c> naming each mode's class; an invalid discrete action is asserted first.
        /// </summary>
        [TestMethod]
        public void GymNet_StepBeforeReset_RaisesLikeGymnasium() {
            var cases = new (string Id, string Python, Func<PyObject> Gymnasium, Func<object?> DotNet)[] {
                (DiscreteId, "1", () => PyConvert.ToPython(1L), () => 1L),
                (DiscreteId, "3", () => PyConvert.ToPython(3L), () => 3L),
                (ContinuousId, "np.array([0.5], dtype=float32)", () => {
                    using NDArray force = np.array(new[] { 0.5f });
                    return force.ToNumpyCopy();
                }, () => np.array(new[] { 0.5f })),
                (ContinuousId, "np.array(0.5, dtype=float32)", () => {
                    using NDArray force = NDArray.Scalar<float>(0.5f);
                    return force.ToNumpyCopy();
                }, () => NDArray.Scalar<float>(0.5f)),
            };

            foreach ((string id, string text, Func<PyObject> gymnasium, Func<object?> value) in cases) {
                using PyRef python = PyRef.Of(() => {
                    using PyObject made = Python.gymnasium.make(id);
                    return Python.gymnasium.Env.unwrapped(made);
                });
                IEnv dotnet = Create(id);
                Outcome pythonStep = Outcome.CapturePython(() => {
                    using PyObject action = gymnasium();
                    return Python.gymnasium.Env.step(python.Object, action);
                });
                Assert.IsNotNull(pythonStep.Raised, $"Gymnasium must refuse {id}'s step({text}) before reset.");
                ByteContract.AssertSame($"{id}: step({text}) before reset", pythonStep, Outcome.Capture(() => dotnet.Step(value())));
            }
        }

        /// <summary>
        ///     One step, then three more, from each of <see cref="InjectedStates"/>, on both ids: the wall rule, the clamps, the
        ///     flag's comparison in each mode's precision, and NaN carried through.
        /// </summary>
        /// <param name="id">The id.</param>
        [TestMethod]
        [DataRow(DiscreteId)]
        [DataRow(ContinuousId)]
        public void GymNet_InjectedStates_MatchGymnasium(string id) {
            SkipOffReferencePlatform("MountainCar's cos");
            foreach (double[] state in InjectedStates) {
                using GymPair pair = Pair(id);
                pair.Reset(0);
                pair.InjectState("state", state);
                for (int i = 0; i < 4; i++) {
                    using ParityAction action = id == DiscreteId ? ParityAction.Discrete(i % 3) : ParityAction.Continuous(np.array(new[] { i % 2 == 0 ? -1.0f : 0.75f }));
                    pair.Step(action);
                }
            }
        }

        /// <summary>
        ///     States Gymnasium's code trips on, with its exception and message on both ids: <c>None</c> (unpacking or
        ///     subscripting it), a one-value state, a three-value state (too many to unpack in discrete mode, the first two used
        ///     in continuous mode), and an empty state for a render.
        /// </summary>
        /// <param name="id">The id.</param>
        [TestMethod]
        [DataRow(DiscreteId)]
        [DataRow(ContinuousId)]
        public void GymNet_MalformedStates_RaiseLikeGymnasium(string id) {
            SkipOffReferencePlatform("MountainCar's cos");
            bool[] gap = HillGap(id);
            foreach (double[]? state in new[] { null, new[] { 0.1 }, new[] { -0.5, 0.01, 0.3 }, Array.Empty<double>() }) {
                using PyGymEnv python = PyGymEnv.Make(id, renderMode: "rgb_array");
                MountainCarEnv dotnet = Create(id, "rgb_array");
                if (state is null) {
                    python.WithUnwrapped(env => {
                        using PyObject none = PyConvert.ToPython(null);
                        if (id == DiscreteId)
                            Python.gymnasium.envs.classic_control.mountain_car.MountainCarEnv.state(env, none);
                        else
                            Python.gymnasium.envs.classic_control.continuous_mountain_car.Continuous_MountainCarEnv.state(env, none);
                    });
                } else {
                    python.SetAttribute("state", state);
                }

                dotnet.State = state;
                string what = state is null ? "None" : $"[{string.Join(", ", state.Select(v => PyBuiltins.FloatRepr(v)))}]";
                using (ParityAction action = id == DiscreteId ? ParityAction.Discrete(1) : ParityAction.Continuous(np.array(new[] { 0.5f }))) {
                    CallResult pythonStep = python.Step(action);
                    CallResult dotnetStep = new GymNetEnv(dotnet).Step(action.DotNet);
                    ByteContract.AssertSame($"{id}: step from state {what}", pythonStep.ToOutcome(), dotnetStep.ToOutcome());
                }

                AssertRenderSameOutsideHillGap($"{id}: render() at state {what}", python.TryRender(), RenderOutcome.Capture(() => RenderGymNet(dotnet)), gap);
            }
        }

        /// <summary><c>render()</c> before any reset raises the missing state's <c>AttributeError</c>, after creating the screen, on both ids.</summary>
        /// <param name="id">The id.</param>
        [TestMethod]
        [DataRow(DiscreteId)]
        [DataRow(ContinuousId)]
        public void GymNet_RenderBeforeReset_RaisesLikeGymnasium(string id) {
            using PyGymEnv python = PyGymEnv.Make(id, renderMode: "rgb_array");
            MountainCarEnv dotnet = Create(id, "rgb_array");
            RenderOutcome pythonOutcome = python.TryRender();
            Assert.IsNotNull(pythonOutcome.Raised, $"Gymnasium's render() before reset gave {pythonOutcome}.");
            RenderOutcome.AssertSame($"{id}: render() before reset", pythonOutcome, RenderOutcome.Capture(() => RenderGymNet(dotnet)));
            Assert.IsNotNull(dotnet.Screen, "The screen exists before the state is read, as in Gymnasium.");
        }

        /// <summary>
        ///     Whole frames along episodes of the momentum policy (a frame every 7 steps and after the reset). Open: the hill's
        ///     anti-aliased polyline differs from pygame's at a few of its joints (P7, 16 pixels per frame).
        /// </summary>
        /// <param name="id">The id.</param>
        [TestMethod]
        [OpenBugs("P7")]
        [DataRow(DiscreteId)]
        [DataRow(ContinuousId)]
        public void GymNet_FramesAlongEpisodes_MatchGymnasium(string id) {
            SkipOffReferencePlatform("MountainCar's cos/sin");
            using GymPair pair = Pair(id, "rgb_array");
            PairRun run = pair.Run(0, Momentum(id), maxSteps: 999, frameEvery: 7);
            TestContext.WriteLine(run.ToString());
        }

        /// <summary>
        ///     Whole frames at each of <see cref="InjectedStates"/>, on both ids. Open: the hill's joints (P7), as in
        ///     <see cref="GymNet_FramesAlongEpisodes_MatchGymnasium"/>.
        /// </summary>
        /// <param name="id">The id.</param>
        [TestMethod]
        [OpenBugs("P7")]
        [DataRow(DiscreteId)]
        [DataRow(ContinuousId)]
        public void GymNet_FramesAtInjectedStates_MatchGymnasium(string id) {
            SkipOffReferencePlatform("MountainCar's cos/sin");
            foreach (double[] state in InjectedStates) {
                using PyGymEnv python = PyGymEnv.Make(id, renderMode: "rgb_array");
                MountainCarEnv dotnet = Create(id, "rgb_array");
                python.SetAttribute("state", state);
                dotnet.State = (double[])state.Clone();
                string what = $"{id}: render() at state [{string.Join(", ", state.Select(v => PyBuiltins.FloatRepr(v)))}]";
                RenderOutcome.AssertSame(what, python.TryRender(), RenderOutcome.Capture(() => RenderGymNet(dotnet)));
            }
        }

        /// <summary>
        ///     The frames of <see cref="GymNet_FramesAlongEpisodes_MatchGymnasium"/> pixel for pixel everywhere but the hill's
        ///     known gap (<see cref="HillGap"/>): the car, its wheels and the flag at every position on the way to the flag, in
        ///     both modes (float32 coordinates in continuous mode).
        /// </summary>
        /// <param name="id">The id.</param>
        [TestMethod]
        [DataRow(DiscreteId)]
        [DataRow(ContinuousId)]
        public void GymNet_FramesAlongEpisodes_MatchGymnasiumOutsideTheHillGap(string id) {
            SkipOffReferencePlatform("MountainCar's cos/sin");
            bool[] gap = HillGap(id);
            using GymPair pair = Pair(id, "rgb_array");
            IActionSource policy = Momentum(id);
            pair.Reset(0);
            AssertSameOutsideHillGap($"{id}: after reset", pair.Python.Render(), pair.DotNet.Render(), gap);
            int frames = 1;
            while (pair.StepsTaken < 999 && !pair.Ended) {
                using (ParityAction action = policy.Next(pair))
                    pair.Step(action);
                if (pair.StepsTaken % 7 != 0 && !pair.Ended)
                    continue;
                AssertSameOutsideHillGap($"{id}: after step {pair.StepsTaken}", pair.Python.Render(), pair.DotNet.Render(), gap);
                frames++;
            }

            TestContext.WriteLine($"{id}: {frames} frames identical outside the hill's {gap.Count(g => g)} gap pixels, through step {pair.StepsTaken}");
        }

        /// <summary>
        ///     <c>render()</c> at each of <see cref="InjectedStates"/> everywhere but the hill's known gap: the car against both
        ///     walls, at the flags, and NaN's pygame error, on both ids.
        /// </summary>
        /// <param name="id">The id.</param>
        [TestMethod]
        [DataRow(DiscreteId)]
        [DataRow(ContinuousId)]
        public void GymNet_FramesAtInjectedStates_MatchGymnasiumOutsideTheHillGap(string id) {
            SkipOffReferencePlatform("MountainCar's cos/sin");
            bool[] gap = HillGap(id);
            foreach (double[] state in InjectedStates) {
                using PyGymEnv python = PyGymEnv.Make(id, renderMode: "rgb_array");
                MountainCarEnv dotnet = Create(id, "rgb_array");
                python.SetAttribute("state", state);
                dotnet.State = (double[])state.Clone();
                string what = $"{id}: render() at state [{string.Join(", ", state.Select(v => PyBuiltins.FloatRepr(v)))}]";
                AssertRenderSameOutsideHillGap(what, python.TryRender(), RenderOutcome.Capture(() => RenderGymNet(dotnet)), gap);
            }
        }

        /// <summary>
        ///     The observation space, <c>Box([-1.2, -0.07], [0.6, 0.07], float32)</c>, on both ids, and the action space:
        ///     <c>Discrete(3)</c> or <c>Box(-1.0, 1.0, (1,), float32)</c>, with seeded samples and the generator behind them.
        /// </summary>
        /// <param name="id">The id.</param>
        [TestMethod]
        [DataRow(DiscreteId)]
        [DataRow(ContinuousId)]
        public void GymNet_Spaces_MatchGymnasium(string id) {
            MountainCarEnv env = Create(id);
            ClassicSpaces.AssertBoxSame($"{id} observation_space", id, Python.gymnasium.Env.observation_space, (Box)env.ObservationSpace);
            ClassicSpaces.AssertActionSpaceSame(id, env.ActionSpace);
        }

        /// <summary><c>metadata</c> on both ids: <c>{'render_modes': ['human', 'rgb_array'], 'render_fps': 30}</c>.</summary>
        /// <param name="id">The id.</param>
        [TestMethod]
        [DataRow(DiscreteId)]
        [DataRow(ContinuousId)]
        public void GymNet_Metadata_MatchesGymnasium(string id) {
            Outcome python = Outcome.CapturePython(() => {
                using PyObject made = Python.gymnasium.make(id);
                using PyObject unwrapped = Python.gymnasium.Env.unwrapped(made);
                return Python.gymnasium.Env.metadata(unwrapped);
            });
            ByteContract.AssertSame($"{id} metadata", python, Outcome.Returned(ValueFacts.Of(Create(id).Metadata)));
        }

        /// <summary>
        ///     <c>str()</c> of the unwrapped env and of <c>make</c>'s stack: the merged class prints as Gymnasium's class of its
        ///     mode (<c>Continuous_MountainCarEnv</c>), with and without a spec.
        /// </summary>
        /// <param name="id">The id.</param>
        [TestMethod]
        [DataRow(DiscreteId)]
        [DataRow(ContinuousId)]
        public void GymNet_Repr_MatchesGymnasium(string id) {
            using PyRef python = PyRef.Of(() => Python.gymnasium.make(id));
            using IEnv dotnet = Gymnasium.Make(id);
            ByteContract.AssertSame($"str(make('{id}'))", Outcome.CapturePython(() => Python.builtins.str(python.Object)), Outcome.Returned(ValueFacts.FromText(dotnet.ToString()!)));
            ByteContract.AssertSame($"str(make('{id}').unwrapped)", Outcome.CapturePython(() => {
                using PyObject unwrapped = Python.gymnasium.Env.unwrapped(python.Object);
                return Python.builtins.str(unwrapped);
            }), Outcome.Returned(ValueFacts.FromText(dotnet.Unwrapped.ToString()!)));

            string bareClass = id == DiscreteId ? "MountainCarEnv" : "Continuous_MountainCarEnv";
            Assert.AreEqual($"<{bareClass} instance>", Create(id).ToString(), "A bare env prints as Gymnasium's class without a spec.");
        }

        /// <summary>
        ///     <c>make(id)</c>'s whole stack (<c>TimeLimit</c>, <c>OrderEnforcing</c>, <c>PassiveEnvChecker</c>) against
        ///     <c>Gymnasium.Make(id)</c> with sampled actions through truncation: 200 steps, and 999 for the continuous id.
        /// </summary>
        /// <param name="id">The id.</param>
        [TestMethod]
        [DataRow(DiscreteId)]
        [DataRow(ContinuousId)]
        public void GymNet_MakeStack_MatchesGymnasiumThroughTruncation(string id) {
            SkipOffReferencePlatform("MountainCar's cos");
            using var pair = new GymPair(PyGymEnv.Make(id, viaMake: true), new GymNetEnv(Gymnasium.Make(id)), stateAttribute: "state");
            PairRun run = pair.Run(21, ActionSources.Sampled(21), maxSteps: 1000);
            Assert.IsTrue(run.Ended, $"make's TimeLimit must end the episode: {run}");
            TestContext.WriteLine(run.ToString());
        }

        /// <summary>The .NET env standing for an id: the merged class in that id's mode.</summary>
        /// <param name="id">The id.</param>
        /// <param name="renderMode">The render mode, or <see langword="null"/>.</param>
        /// <returns>The env.</returns>
        private static MountainCarEnv Create(string id, string? renderMode = null) => new(renderMode: renderMode, continuous: id == ContinuousId);

        /// <summary>A lockstep pair of Gymnasium's unwrapped env and Gym.NET's for an id, comparing the state's values.</summary>
        /// <param name="id">The id.</param>
        /// <param name="renderMode">The render mode of both, or <see langword="null"/>.</param>
        /// <returns>The pair; the caller disposes it.</returns>
        private static GymPair Pair(string id, string? renderMode = null) =>
            new(PyGymEnv.Make(id, renderMode: renderMode), new GymNetEnv(Create(id, renderMode)), stateAttribute: "state");

        /// <summary>The momentum policy of an id's mode.</summary>
        /// <param name="id">The id.</param>
        /// <returns>The action source.</returns>
        private static IActionSource Momentum(string id) => ActionSources.Policy(id == DiscreteId ? "mountaincar_momentum" : "mountaincar_continuous_momentum");

        /// <summary>Gymnasium's state as its float64 values (or the missing attribute's error), as <see cref="GymPair"/> compares it.</summary>
        /// <param name="env">The unwrapped env; borrowed.</param>
        /// <returns>The outcome.</returns>
        private static Outcome PythonState(PyObject env) => Outcome.CapturePython(() => {
            using PyObject state = Python.gymnasium.envs.classic_control.mountain_car.MountainCarEnv.state(env);
            return Python.numpy.asarray(state, "float64");
        });

        /// <summary>Gym.NET's state values as NumPy's float64 array (or <c>None</c>).</summary>
        /// <param name="state">The values.</param>
        /// <returns>The description.</returns>
        private static ValueFacts StateFacts(double[]? state) {
            if (state is null)
                return ValueFacts.NoneValue();
            using NDArray values = np.array(state);
            return ValueFacts.Of(values);
        }

        /// <summary>
        ///     The pixels where Gym.NET's hill is not yet pygame's (P7): both sides draw the scene with the car parked off-screen
        ///     (position 1.8, right of the window), and every pixel that differs is in the gap. The hill depends on neither the
        ///     state nor the mode, so these are the only pixels its gap can reach in any frame.
        /// </summary>
        /// <param name="id">The id.</param>
        /// <returns>One flag per pixel, row-major.</returns>
        /// <exception cref="AssertFailedException">Thrown when the gap is larger than the P7 report's 16 pixels allow for (64), which would hide more than the hill.</exception>
        private static bool[] HillGap(string id) {
            using PyGymEnv python = PyGymEnv.Make(id, renderMode: "rgb_array");
            MountainCarEnv dotnet = Create(id, "rgb_array");
            double[] parked = { 1.8, 0.0 };
            python.SetAttribute("state", parked);
            dotnet.State = (double[])parked.Clone();
            Frame pythonFrame = python.Render();
            Frame dotnetFrame = RenderGymNet(dotnet) ?? throw new AssertFailedException("Gym.NET rendered nothing for the parked car.");
            Assert.AreEqual(pythonFrame.Height * pythonFrame.Width, dotnetFrame.Height * dotnetFrame.Width, "Both sides render frames of one size.");
            var gap = new bool[pythonFrame.Height * pythonFrame.Width];
            int count = 0;
            for (int pixel = 0; pixel < gap.Length; pixel++) {
                int at = pixel * 3;
                gap[pixel] = pythonFrame.Rgb[at] != dotnetFrame.Rgb[at] || pythonFrame.Rgb[at + 1] != dotnetFrame.Rgb[at + 1] || pythonFrame.Rgb[at + 2] != dotnetFrame.Rgb[at + 2];
                if (gap[pixel])
                    count++;
            }

            // A guard on the mask itself: a gap beyond the hill's few joint pixels would hide real differences.
            Assert.IsTrue(count <= 64, $"The hill's gap is {count} pixels; P7 left 16, so something beyond the hill differs.");
            return gap;
        }

        /// <summary>
        ///     Compares two frames exactly (<see cref="FrameDiff.AssertSame"/>) after blanking the hill gap's pixels to black on
        ///     both sides; every other pixel must be identical.
        /// </summary>
        /// <param name="what">What was rendered, for the report.</param>
        /// <param name="python">Gymnasium's frame.</param>
        /// <param name="dotnet">Gym.NET's frame.</param>
        /// <param name="gap">The pixels to leave out, from <see cref="HillGap"/>.</param>
        /// <exception cref="ParityException">Thrown when any pixel outside the gap differs.</exception>
        private static void AssertSameOutsideHillGap(string what, Frame python, Frame dotnet, bool[] gap) =>
            FrameDiff.AssertSame(what, Masked(python, gap), Masked(dotnet, gap), "frame outside the hill's P7 gap");

        /// <summary>
        ///     Compares two render outcomes: frames by <see cref="AssertSameOutsideHillGap"/>, anything else (an exception,
        ///     <c>None</c>) exactly.
        /// </summary>
        /// <param name="what">What was rendered, for the report.</param>
        /// <param name="python">Gymnasium's outcome.</param>
        /// <param name="dotnet">Gym.NET's outcome.</param>
        /// <param name="gap">The pixels to leave out of frame comparisons.</param>
        /// <exception cref="ParityException">Thrown when the outcomes differ.</exception>
        private static void AssertRenderSameOutsideHillGap(string what, RenderOutcome python, RenderOutcome dotnet, bool[] gap) {
            if (python.Frame is { } pythonFrame && dotnet.Frame is { } dotnetFrame)
                AssertSameOutsideHillGap(what, pythonFrame, dotnetFrame, gap);
            else
                RenderOutcome.AssertSame(what, python, dotnet);
        }

        /// <summary>A copy of a frame with the gap's pixels set to black.</summary>
        /// <param name="frame">The frame; not modified.</param>
        /// <param name="gap">The pixels to blank.</param>
        /// <returns>The copy.</returns>
        private static Frame Masked(Frame frame, bool[] gap) {
            byte[] rgb = (byte[])frame.Rgb.Clone();
            for (int pixel = 0; pixel < gap.Length && pixel * 3 < rgb.Length; pixel++) {
                if (!gap[pixel])
                    continue;
                rgb[pixel * 3] = 0;
                rgb[pixel * 3 + 1] = 0;
                rgb[pixel * 3 + 2] = 0;
            }

            return new Frame(frame.Height, frame.Width, rgb);
        }

        /// <summary>Gym.NET's <c>render()</c> of an <c>rgb_array</c> env as a frame, or <see langword="null"/> when it returned <c>None</c>.</summary>
        /// <param name="env">The env, built with <c>render_mode="rgb_array"</c>.</param>
        /// <returns>The frame, or <see langword="null"/>.</returns>
        /// <exception cref="InvalidOperationException">Thrown when the rendering is neither a frame nor <c>None</c>.</exception>
        private static Frame? RenderGymNet(IEnv env) {
            switch (env.Render()) {
                case RenderResult.NoneResult:
                    return null;
                case RenderResult.Rgb rgb: {
                    using NDArray frame = rgb.Frame;
                    return new Frame((int)frame.shape[0], (int)frame.shape[1], frame.ToArray<byte>());
                }
                case var other:
                    throw new InvalidOperationException($"{env} in rgb_array mode rendered {other}.");
            }
        }
    }
}
