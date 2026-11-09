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
    ///     Acrobot-v1 against Gymnasium's <c>AcrobotEnv</c>, live and in lockstep with Gym.NET's <see cref="AcrobotEnv"/>: resets
    ///     (float32 states and their float32 trigonometry), episodes of sampled actions and of an energy-pumping policy through
    ///     termination, Python's list indexing of the action, the NIPS equations and torque noise, injected and malformed
    ///     states, every error, spaces, metadata, <c>make</c>'s stack, and <c>rgb_array</c> frames.
    /// </summary>
    /// <remarks>
    ///     Every lockstep compares the state's float64 values after each call (Gymnasium's state is a float32 array after a
    ///     reset and a float64 one after a step). Cells that step or draw are pinned to the reference platform: the dynamics
    ///     rest on the platform's <c>sin</c>/<c>cos</c>/<c>pow</c>. Reference:
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/acrobot.py#L26">gymnasium/envs/classic_control/acrobot.py</see>.
    /// </remarks>
    [TestClass]
    public sealed class AcrobotLiveParityTests : ParityTestBase {
        /// <summary>The id.</summary>
        private const string Id = "Acrobot-v1";

        /// <summary>
        ///     Python's <c>float('nan')</c>, the positive quiet NaN (0x7FF8000000000000); .NET's <see cref="double.NaN"/> is the
        ///     negative one (0xFFF8000000000000), which no Python code injects.
        /// </summary>
        private static readonly double PythonNaN = BitConverter.Int64BitsToDouble(0x7FF8000000000000);

        /// <summary>
        ///     A NaN state, for the render cells only. Stepping it is not compared: <c>_dsdt</c> negates a NaN
        ///     (<c>ddtheta1 = -(...) / d1</c>), so NaNs of both signs meet in the Runge-Kutta sums, and two NaNs in one
        ///     operation keep the sign of whichever operand the compiled code puts first, an order NumPy's build, .NET 8's JIT
        ///     and .NET 10's do not share (measured: the state's sign bit differed on net8.0 only, either way round depending
        ///     on the injected NaN's sign). Neither library defines that bit; rendering the state raises pygame's error first.
        /// </summary>
        private static readonly double[] NaNState = { PythonNaN, 0.0, 0.0, 0.0 };

        /// <summary>
        ///     States written into both envs before steps or a frame: hanging still, near the swing-up, angles past ±π (which
        ///     the step wraps), velocities beyond both bounds, and a negative zero (NaN is <see cref="NaNState"/>, rendered only).
        /// </summary>
        public static readonly double[][] InjectedStates = {
            new[] { 0.0, 0.0, 0.0, 0.0 },
            new[] { Math.PI - 0.1, 0.2, 0.5, -0.5 },
            new[] { 3.2, -3.3, 1.0, 2.0 },
            new[] { -0.3, 0.4, 14.0, -30.0 },
            new[] { 1.0, -1.0, -13.0, 29.0 },
            new[] { -0.0, -0.0, -0.0, -0.0 },
            new[] { 2.5, 2.5, 3.0, 3.0 },
        };

        /// <summary>The reset options compared, around the defaults of ±0.1.</summary>
        private static readonly (string Python, Dictionary<string, object?> Options)[] ResetOptions = {
            ("{}", new Dictionary<string, object?>()),
            ("{'low': -0.5, 'high': 0.5}", new Dictionary<string, object?> { ["low"] = -0.5, ["high"] = 0.5 }),
            ("{'high': 3}", new Dictionary<string, object?> { ["high"] = 3L }),
            ("{'low': 0.2}", new Dictionary<string, object?> { ["low"] = 0.2 }),
            ("{'low': 'x'}", new Dictionary<string, object?> { ["low"] = "x" }),
            ("{'high': None}", new Dictionary<string, object?> { ["high"] = null }),
            ("{'low': nan}", new Dictionary<string, object?> { ["low"] = double.NaN }),
            ("{'high': inf}", new Dictionary<string, object?> { ["high"] = double.PositiveInfinity }),
            ("{'low': 1e-40, 'high': 1e-39}", new Dictionary<string, object?> { ["low"] = 1e-40, ["high"] = 1e-39 }),
            ("{'low': 0.05, 'high': 0.05}", new Dictionary<string, object?> { ["low"] = 0.05, ["high"] = 0.05 }),
        };

        /// <summary><c>reset(seed=seed)</c>: the float32 state's observation (NumPy's float32 cos/sin), info, state and generator.</summary>
        /// <param name="seed">The reset seed.</param>
        [TestMethod]
        [DataRow(0L)]
        [DataRow(1L)]
        [DataRow(42L)]
        [DataRow(1234L)]
        public void GymNet_Reset_MatchesGymnasium(long seed) {
            using GymPair pair = Pair();
            pair.Reset(seed);
        }

        /// <summary>Whole episodes of seeded sampled actions (both action spaces sampled and compared), 500 steps.</summary>
        /// <param name="seed">The reset seed, also the action spaces' seed.</param>
        [TestMethod]
        [DataRow(0L)]
        [DataRow(7L)]
        [DataRow(42L)]
        public void GymNet_SampledEpisodes_MatchGymnasium(long seed) {
            SkipOffReferencePlatform("Acrobot's sin/cos/pow");
            using GymPair pair = Pair();
            PairRun run = pair.Run(seed, ActionSources.Sampled(seed), maxSteps: 500);
            TestContext.WriteLine(run.ToString());
        }

        /// <summary>The energy-pumping policy swings the chain up to termination (reward 0.0), then five more steps past it.</summary>
        /// <param name="seed">The reset seed.</param>
        [TestMethod]
        [DataRow(0L)]
        [DataRow(3L)]
        public void GymNet_PumpingEpisodes_MatchGymnasiumThroughTermination(long seed) {
            SkipOffReferencePlatform("Acrobot's sin/cos/pow");
            using GymPair pair = Pair();
            IActionSource policy = ActionSources.Policy("acrobot_pump");
            PairRun run = pair.Run(seed, policy, maxSteps: 2000);
            Assert.IsTrue(run.Ended, $"The pumping policy must swing the chain up: {run}");
            for (int i = 0; i < 5; i++) {
                using ParityAction action = policy.Next(pair);
                pair.Step(action);
            }

            TestContext.WriteLine($"{run}; then 5 more steps past termination, identical");
        }

        /// <summary>
        ///     Actions as Python indexes <c>AVAIL_TORQUE</c> with them: the negative indices -1, -2 and -3 (valid), NumPy
        ///     integer scalars of several dtypes, and a Python <c>bool</c>, one step each.
        /// </summary>
        [TestMethod]
        public void GymNet_ListIndexedActions_MatchGymnasium() {
            SkipOffReferencePlatform("Acrobot's sin/cos/pow");
            using GymPair pair = Pair();
            pair.Reset(9);
            foreach (long action in new long[] { -1, -2, -3, 2, 0, 1 }) {
                using var step = ParityAction.Discrete(action);
                pair.Step(step);
            }

            var scalars = new Func<NDArray>[] {
                () => NDArray.Scalar<byte>(2), () => NDArray.Scalar<sbyte>(-1), () => NDArray.Scalar<ushort>(0), () => NDArray.Scalar<int>(-3), () => NDArray.Scalar<ulong>(1),
            };
            foreach (Func<NDArray> scalar in scalars) {
                using var step = ParityAction.Scalar(scalar());
                pair.Step(step);
            }
        }

        /// <summary>
        ///     Actions Python's list indexing refuses, through the untyped step, with Python's exception and message: out of range,
        ///     beyond an index-sized integer, floats (Python's and NumPy's), a NumPy bool, an array, a string, <c>None</c>; and
        ///     before any reset, the state's assertion first whatever the action.
        /// </summary>
        [TestMethod]
        public void GymNet_InvalidActions_RaiseLikeGymnasium() {
            var cases = new (string Python, Func<PyObject> Gymnasium, Func<object?> DotNet)[] {
                ("3", () => PyConvert.ToPython(3L), () => 3L),
                ("-4", () => PyConvert.ToPython(-4L), () => -4L),
                ("2**70", () => PyConvert.ToPython(BigInteger.Pow(2, 70)), () => BigInteger.Pow(2, 70)),
                ("2**63", () => PyConvert.ToPython(1UL << 63), () => 1UL << 63),
                ("1.0", () => PyConvert.ToPython(1.0), () => 1.0),
                ("np.float32(1.0)", () => Python.numpy.float32(1.0f), () => 1.0f),
                ("np.float64(1.0)", () => Python.numpy.float64(1.0), () => NDArray.Scalar<double>(1.0)),
                ("np.int64(5)", () => Python.numpy.int64(5), () => NDArray.Scalar<long>(5L)),
                ("np.uint64(2**63)", () => Python.numpy.uint64(1UL << 63), () => NDArray.Scalar<ulong>(1UL << 63)),
                ("np.True_", () => {
                    using NDArray flag = NDArray.Scalar<bool>(true);
                    using PyObject array = flag.ToNumpyCopy();
                    return Python.numpy.ndarray.item_scalar(array);
                }, () => NDArray.Scalar<bool>(true)),
                ("np.array([1])", () => {
                    using NDArray one = np.array(new[] { 1L });
                    return one.ToNumpyCopy();
                }, () => np.array(new[] { 1L })),
                ("'1'", () => PyConvert.ToPython("1"), () => "1"),
                ("None", () => PyConvert.ToPython(null), () => null),
            };

            using PyRef python = PyRef.Of(() => {
                using PyObject made = Python.gymnasium.make(Id);
                return Python.gymnasium.Env.unwrapped(made);
            });
            var dotnet = new AcrobotEnv();
            IEnv untyped = dotnet;

            // Before any reset: the state's assertion, whatever the action.
            foreach ((string text, Func<PyObject> gymnasium, Func<object?> value) in cases.Take(3).Concat(cases.Skip(10))) {
                Outcome pythonStep = Outcome.CapturePython(() => {
                    using PyObject action = gymnasium();
                    return Python.gymnasium.Env.step(python.Object, action);
                });
                ByteContract.AssertSame($"step({text}) before reset", pythonStep, Outcome.Capture(() => untyped.Step(value())));
            }

            Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, 0)).Expect("reset(seed=0)");
            dotnet.Reset(0L).Observation.Dispose();
            foreach ((string text, Func<PyObject> gymnasium, Func<object?> value) in cases) {
                Outcome pythonStep = Outcome.CapturePython(() => {
                    using PyObject action = gymnasium();
                    return Python.gymnasium.Env.step(python.Object, action);
                });
                Assert.IsNotNull(pythonStep.Raised, $"Gymnasium must refuse step({text}).");
                ByteContract.AssertSame($"step({text})", pythonStep, Outcome.Capture(() => untyped.Step(value())));
            }
        }

        /// <summary>
        ///     <c>reset(seed=..., options=...)</c> for every case of <see cref="ResetOptions"/>, seeded and unseeded: the result or
        ///     the exception, then the state's values and the generator.
        /// </summary>
        [TestMethod]
        public void GymNet_ResetOptions_MatchGymnasium() {
            using PyRef python = PyRef.Of(() => {
                using PyObject made = Python.gymnasium.make(Id);
                return Python.gymnasium.Env.unwrapped(made);
            });
            var dotnet = new AcrobotEnv();
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

                    string call = $"reset(seed={(resetSeed?.ToString(CultureInfo.InvariantCulture) ?? "None")}, options={text})";
                    Outcome pythonReset = Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, resetSeed, options));
                    Outcome dotnetReset = Outcome.Capture(() => {
                        var (observation, info) = dotnet.Reset(resetSeed, options);
                        return (observation, info);
                    });
                    ByteContract.AssertSame(call, pythonReset, dotnetReset);
                    ByteContract.AssertSame($"{call}: state", Outcome.CapturePython(() => {
                        using PyObject state = Python.gymnasium.envs.classic_control.acrobot.AcrobotEnv.state(python.Object);
                        return Python.numpy.asarray(state, "float64");
                    }), Outcome.Returned(StateFacts(dotnet.State)));
                    ByteContract.AssertSame($"{call}: np_random state", Outcome.CapturePython(() => {
                        using PyObject generator = Python.gymnasium.Env.np_random(python.Object);
                        using PyObject bitGenerator = Python.numpy.random.Generator.bit_generator(generator);
                        return Python.numpy.random.BitGenerator.state(bitGenerator);
                    }), Outcome.Returned(RngFacts.Of(dotnet.NpRandom)));
                }
            }
        }

        /// <summary>Five steps from each of <see cref="InjectedStates"/>: the wrap of both angles, the velocity bounds, termination.</summary>
        [TestMethod]
        public void GymNet_InjectedStates_MatchGymnasium() {
            SkipOffReferencePlatform("Acrobot's sin/cos/pow");
            foreach (double[] state in InjectedStates) {
                using GymPair pair = Pair();
                pair.Reset(0);
                pair.InjectState("state", state);
                for (int i = 0; i < 5; i++) {
                    using var action = ParityAction.Discrete(i % 3);
                    pair.Step(action);
                }
            }
        }

        /// <summary>
        ///     States Gymnasium's code trips on, stepped: three values and none (NumPy's <c>IndexError</c> in <c>_dsdt</c>), five
        ///     values (NumPy's broadcast <c>ValueError</c> in <c>rk4</c>), and <c>None</c> (the assertion).
        /// </summary>
        [TestMethod]
        public void GymNet_MalformedStates_RaiseLikeGymnasium() {
            foreach (double[]? state in new[] { new[] { 0.1, 0.2, 0.3 }, Array.Empty<double>(), new[] { 0.1, 0.2, 0.3, 0.4, 0.5 }, null }) {
                using PyGymEnv python = PyGymEnv.Make(Id, renderMode: "rgb_array");
                var dotnet = new AcrobotEnv(renderMode: "rgb_array");
                python.Reset(1);
                dotnet.Reset(1L).Observation.Dispose();
                if (state is null) {
                    python.WithUnwrapped(env => {
                        using PyObject none = PyConvert.ToPython(null);
                        Python.gymnasium.envs.classic_control.acrobot.AcrobotEnv.state(env, none);
                    });
                } else {
                    python.SetAttribute("state", state);
                }

                dotnet.State = state;
                string what = state is null ? "None" : $"[{string.Join(", ", state.Select(v => PyBuiltins.FloatRepr(v)))}]";
                using (var action = ParityAction.Discrete(1)) {
                    CallResult pythonStep = python.Step(action);
                    CallResult dotnetStep = new GymNetEnv(dotnet).Step(action.DotNet);
                    Assert.IsNotNull(pythonStep.Raised, $"Gymnasium must refuse a step from state {what}.");
                    ByteContract.AssertSame($"step from state {what}", pythonStep.ToOutcome(), dotnetStep.ToOutcome());
                }
            }
        }

        /// <summary>The NIPS paper's equation for the second joint (<c>book_or_nips = "nips"</c>): an episode of sampled actions.</summary>
        [TestMethod]
        public void GymNet_NipsDynamics_MatchGymnasium() {
            SkipOffReferencePlatform("Acrobot's sin/cos/pow");
            PyGymEnv python = PyGymEnv.Make(Id);
            python.WithUnwrapped(env => Python.gymnasium.envs.classic_control.acrobot.AcrobotEnv.book_or_nips(env, "nips"));
            using var pair = new GymPair(python, new GymNetEnv(new AcrobotEnv { BookOrNips = "nips" }), stateAttribute: "state");
            pair.Run(17, ActionSources.Sampled(17), maxSteps: 300);
        }

        /// <summary>Torque noise (<c>torque_noise_max = 0.25</c>): one draw from the env's generator per step, compared with the generator's state.</summary>
        [TestMethod]
        public void GymNet_TorqueNoise_MatchesGymnasium() {
            SkipOffReferencePlatform("Acrobot's sin/cos/pow");
            PyGymEnv python = PyGymEnv.Make(Id);
            python.WithUnwrapped(env => Python.gymnasium.envs.classic_control.acrobot.AcrobotEnv.torque_noise_max(env, 0.25));
            using var pair = new GymPair(python, new GymNetEnv(new AcrobotEnv { TorqueNoiseMax = 0.25 }), stateAttribute: "state");
            pair.Run(19, ActionSources.Sampled(19), maxSteps: 300);
        }

        /// <summary>
        ///     Frames along episodes of the pumping policy: after the reset (the float32 state, NumPy's float32 cos/sin in the
        ///     drawing) and every fifth step, the links at every angle they pass through.
        /// </summary>
        /// <param name="seed">The reset seed.</param>
        [TestMethod]
        [DataRow(0L)]
        [DataRow(3L)]
        public void GymNet_FramesAlongEpisodes_MatchGymnasium(long seed) {
            SkipOffReferencePlatform("Acrobot's rotation (cos/sin)");
            using GymPair pair = Pair("rgb_array");
            PairRun run = pair.Run(seed, ActionSources.Policy("acrobot_pump"), maxSteps: 2000, frameEvery: 5);
            TestContext.WriteLine(run.ToString());
        }

        /// <summary>
        ///     <c>render()</c> at each of <see cref="InjectedStates"/>, at <see cref="NaNState"/> (pygame's error), at a one-value
        ///     state (NumPy's <c>IndexError</c>), and before any reset (<c>None</c>, after the screen is made).
        /// </summary>
        [TestMethod]
        public void GymNet_FramesAtInjectedStates_MatchGymnasium() {
            SkipOffReferencePlatform("Acrobot's rotation (cos/sin)");
            using (PyGymEnv python = PyGymEnv.Make(Id, renderMode: "rgb_array")) {
                var dotnet = new AcrobotEnv(renderMode: "rgb_array");
                RenderOutcome pythonOutcome = python.TryRender();
                Assert.IsTrue(pythonOutcome.ReturnedNone, $"Gymnasium's render() before reset gave {pythonOutcome}.");
                RenderOutcome.AssertSame("render() before reset", pythonOutcome, RenderOutcome.Capture(() => RenderGymNet(dotnet)));
                Assert.IsNotNull(dotnet.Screen, "The screen exists before the state is looked at, as in Gymnasium.");
            }

            foreach (double[] state in InjectedStates.Append(NaNState).Append(new[] { 0.4 })) {
                using PyGymEnv python = PyGymEnv.Make(Id, renderMode: "rgb_array");
                var dotnet = new AcrobotEnv(renderMode: "rgb_array");
                python.SetAttribute("state", state);
                dotnet.State = (double[])state.Clone();
                string what = $"render() at state [{string.Join(", ", state.Select(v => PyBuiltins.FloatRepr(v)))}]";
                RenderOutcome.AssertSame(what, python.TryRender(), RenderOutcome.Capture(() => RenderGymNet(dotnet)));
            }
        }

        /// <summary>
        ///     The spaces: <c>Box(-high, high, float32)</c> with <c>high = [1, 1, 1, 1, 4π, 9π]</c> observations and
        ///     <c>Discrete(3)</c> actions, with seeded samples and the generator behind them.
        /// </summary>
        [TestMethod]
        public void GymNet_Spaces_MatchGymnasium() {
            var env = new AcrobotEnv();
            ClassicSpaces.AssertBoxSame($"{Id} observation_space", Id, Python.gymnasium.Env.observation_space, (Box)env.ObservationSpace);
            ClassicSpaces.AssertActionSpaceSame(Id, env.ActionSpace);
        }

        /// <summary><c>metadata</c>: <c>{'render_modes': ['human', 'rgb_array'], 'render_fps': 15}</c>.</summary>
        [TestMethod]
        public void GymNet_Metadata_MatchesGymnasium() {
            Outcome python = Outcome.CapturePython(() => {
                using PyObject made = Python.gymnasium.make(Id);
                using PyObject unwrapped = Python.gymnasium.Env.unwrapped(made);
                return Python.gymnasium.Env.metadata(unwrapped);
            });
            ByteContract.AssertSame($"{Id} metadata", python, Outcome.Returned(ValueFacts.Of(new AcrobotEnv().Metadata)));
        }

        /// <summary><c>make("Acrobot-v1")</c>'s whole stack against <c>Gymnasium.Make</c> with sampled actions through truncation at 500 steps.</summary>
        [TestMethod]
        public void GymNet_MakeStack_MatchesGymnasiumThroughTruncation() {
            SkipOffReferencePlatform("Acrobot's sin/cos/pow");
            using var pair = new GymPair(PyGymEnv.Make(Id, viaMake: true), new GymNetEnv(Gymnasium.Make(Id)), stateAttribute: "state");
            PairRun run = pair.Run(23, ActionSources.Sampled(23), maxSteps: 600);
            Assert.IsTrue(run.Ended, $"make's TimeLimit must end the episode: {run}");
        }

        /// <summary>A lockstep pair of Gymnasium's unwrapped env and Gym.NET's, comparing the state's float64 values.</summary>
        /// <param name="renderMode">The render mode of both, or <see langword="null"/>.</param>
        /// <returns>The pair; the caller disposes it.</returns>
        private static GymPair Pair(string? renderMode = null) =>
            new(PyGymEnv.Make(Id, renderMode: renderMode), new GymNetEnv(new AcrobotEnv(renderMode: renderMode)), stateAttribute: "state");

        /// <summary>Gym.NET's state values as NumPy's float64 array (or <c>None</c>).</summary>
        /// <param name="state">The values.</param>
        /// <returns>The description.</returns>
        private static ValueFacts StateFacts(double[]? state) {
            if (state is null)
                return ValueFacts.NoneValue();
            using NDArray values = np.array(state);
            return ValueFacts.Of(values);
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
