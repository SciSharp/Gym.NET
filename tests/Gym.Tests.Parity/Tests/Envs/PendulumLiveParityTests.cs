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
    ///     Pendulum-v1 against Gymnasium's <c>PendulumEnv</c>, live and in lockstep with Gym.NET's <see cref="PendulumEnv"/>:
    ///     resets and their options, episodes of sampled float32 torques and of float64, integer and out-of-range ones, the
    ///     changed <c>g</c>, injected and malformed states, every error, spaces, metadata, <c>make</c>'s stack, and
    ///     <c>rgb_array</c> frames with the torque arrow at every size it takes.
    /// </summary>
    /// <remarks>
    ///     Every lockstep compares the float64 state after each call. Cells that step or draw are pinned to the reference
    ///     platform: the dynamics and the rod's rotation rest on the platform's <c>sin</c>/<c>cos</c>/<c>pow</c>. Reference:
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/pendulum.py#L16">gymnasium/envs/classic_control/pendulum.py</see>.
    /// </remarks>
    [TestClass]
    public sealed class PendulumLiveParityTests : ParityTestBase {
        /// <summary>The id.</summary>
        private const string Id = "Pendulum-v1";

        /// <summary>
        ///     States written into both envs before steps or a frame: upright, hanging, past ±π (the cost wraps the angle), at
        ///     and beyond the speed limit, a negative zero, a large angle, and NaN.
        /// </summary>
        public static readonly double[][] InjectedStates = {
            new[] { 0.0, 0.0 },
            new[] { Math.PI, 0.0 },
            new[] { -Math.PI, 0.0 },
            new[] { 3.5, 1.0 },
            new[] { -3.5, -1.0 },
            new[] { 0.3, 8.0 },
            new[] { -0.3, -9.0 },
            new[] { -0.0, -0.0 },
            new[] { 100.0, 0.5 },
            new[] { 1e-7, 0.0 },
            new[] { double.NaN, 0.0 },
        };

        /// <summary>The reset options compared: every key's defaults and overrides, rejected values, and bounds the generator refuses.</summary>
        private static readonly (string Python, Dictionary<string, object?> Options)[] ResetOptions = {
            ("{}", new Dictionary<string, object?>()),
            ("{'x_init': 0.5}", new Dictionary<string, object?> { ["x_init"] = 0.5 }),
            ("{'y_init': 2.0}", new Dictionary<string, object?> { ["y_init"] = 2.0 }),
            ("{'x_init': 0.1, 'y_init': 0.1}", new Dictionary<string, object?> { ["x_init"] = 0.1, ["y_init"] = 0.1 }),
            ("{'x_init': 0.0, 'y_init': 0.0}", new Dictionary<string, object?> { ["x_init"] = 0.0, ["y_init"] = 0.0 }),
            ("{'x_init': None}", new Dictionary<string, object?> { ["x_init"] = null }),
            ("{'x_init': 'abc'}", new Dictionary<string, object?> { ["x_init"] = "abc" }),
            ("{'x_init': -1.0}", new Dictionary<string, object?> { ["x_init"] = -1.0 }),
            ("{'y_init': nan}", new Dictionary<string, object?> { ["y_init"] = double.NaN }),
            ("{'x_init': inf}", new Dictionary<string, object?> { ["x_init"] = double.PositiveInfinity }),
            ("{'x_init': 2, 'y_init': True}", new Dictionary<string, object?> { ["x_init"] = 2L, ["y_init"] = true }),
            ("{'x_init': '1.5'}", new Dictionary<string, object?> { ["x_init"] = "1.5" }),
            ("{'x_init': 10**400}", new Dictionary<string, object?> { ["x_init"] = BigInteger.Pow(10, 400) }),
            ("{'low': -0.1}", new Dictionary<string, object?> { ["low"] = -0.1 }),
        };

        /// <summary><c>reset(seed=seed)</c>: observation, info, state and generator.</summary>
        /// <param name="seed">The reset seed.</param>
        [TestMethod]
        [DataRow(0L)]
        [DataRow(1L)]
        [DataRow(42L)]
        public void GymNet_Reset_MatchesGymnasium(long seed) {
            SkipOffReferencePlatform("Pendulum's cos/sin");
            using GymPair pair = Pair();
            pair.Reset(seed);
        }

        /// <summary>Whole episodes of seeded sampled float32 torques (both action spaces sampled and compared), 200 steps.</summary>
        /// <param name="seed">The reset seed, also the action spaces' seed.</param>
        [TestMethod]
        [DataRow(0L)]
        [DataRow(7L)]
        [DataRow(42L)]
        public void GymNet_SampledEpisodes_MatchGymnasium(long seed) {
            SkipOffReferencePlatform("Pendulum's sin/pow");
            using GymPair pair = Pair();
            PairRun run = pair.Run(seed, ActionSources.Sampled(seed), maxSteps: 200);
            TestContext.WriteLine(run.ToString());
        }

        /// <summary>
        ///     Torques of other dtypes and values, one step each from one episode: float64 inside and beyond the clamp (clipped
        ///     in float64), float32 beyond it, int64 and boolean torques (clipped as float64), a two-element torque (the first
        ///     counts), and NaN last.
        /// </summary>
        [TestMethod]
        public void GymNet_TorqueDtypes_MatchGymnasium() {
            SkipOffReferencePlatform("Pendulum's sin/pow");
            using GymPair pair = Pair();
            pair.Reset(5);
            var actions = new Func<NDArray>[] {
                () => np.array(new[] { 0.7 }), () => np.array(new[] { 3.0 }), () => np.array(new[] { -2.5f }), () => np.array(new[] { 1.999999f }),
                () => np.array(new[] { 1L }), () => np.array(new[] { -7L }), () => np.array(new[] { true }), () => np.array(new[] { 0.25f, 9.0f }),
                () => np.array(new[] { -0.0f }), () => np.array(new[] { 0.0 }), () => np.array(new[] { 1e-30 }), () => np.array(new[] { float.NaN }),
                () => np.array(new[] { 1.0f }),
            };
            foreach (Func<NDArray> action in actions) {
                using var step = ParityAction.Continuous(action());
                pair.Step(step);
            }
        }

        /// <summary>
        ///     Actions NumPy cannot index after clipping (a 0-d torque: <c>invalid index to scalar variable.</c>; an empty one),
        ///     and a step before any reset (the missing state's <c>AttributeError</c>, raised before the action is looked at).
        /// </summary>
        [TestMethod]
        public void GymNet_ActionAndOrderErrors_MatchGymnasium() {
            using (GymPair pair = Pair()) {
                using (var zeroD = ParityAction.Scalar(NDArray.Scalar<float>(0.5f)))
                    Assert.IsNotNull(pair.Step(zeroD).Raised, "Gymnasium must refuse a step before reset.");
                pair.Reset(3);
                using (var zeroD = ParityAction.Continuous(NDArray.Scalar<float>(0.5f)))
                    Assert.IsNotNull(pair.Step(zeroD).Raised, "Gymnasium must refuse a 0-d torque.");
                using (var empty = ParityAction.Continuous(np.array(Array.Empty<float>())))
                    Assert.IsNotNull(pair.Step(empty).Raised, "Gymnasium must refuse an empty torque.");
                using (var scalar = ParityAction.Scalar(NDArray.Scalar<double>(0.5)))
                    Assert.IsNotNull(pair.Step(scalar).Raised, "Gymnasium must refuse a NumPy scalar torque.");
            }
        }

        /// <summary>
        ///     <c>reset(seed=..., options=...)</c> for every case of <see cref="ResetOptions"/>, seeded and unseeded: the result or
        ///     the exception (<c>ValueError</c> from the cast or the generator, <c>OverflowError</c>), then the state and the
        ///     generator.
        /// </summary>
        [TestMethod]
        public void GymNet_ResetOptions_MatchGymnasium() {
            SkipOffReferencePlatform("Pendulum's cos/sin");
            using PyRef python = PyRef.Of(() => {
                using PyObject made = Python.gymnasium.make(Id);
                return Python.gymnasium.Env.unwrapped(made);
            });
            var dotnet = new PendulumEnv();
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
                    ByteContract.AssertSame($"{call}: state", Outcome.CapturePython(() => Python.gymnasium.envs.classic_control.pendulum.PendulumEnv.state(python.Object)),
                        Outcome.Returned(StateFacts(dotnet.State)));
                    ByteContract.AssertSame($"{call}: np_random state", Outcome.CapturePython(() => {
                        using PyObject generator = Python.gymnasium.Env.np_random(python.Object);
                        using PyObject bitGenerator = Python.numpy.random.Generator.bit_generator(generator);
                        return Python.numpy.random.BitGenerator.state(bitGenerator);
                    }), Outcome.Returned(RngFacts.Of(dotnet.NpRandom)));
                }
            }
        }

        /// <summary>Five steps from each of <see cref="InjectedStates"/>, with torques of both signs: the angle wrap in the cost, the speed clamp, NaN carried through.</summary>
        [TestMethod]
        public void GymNet_InjectedStates_MatchGymnasium() {
            SkipOffReferencePlatform("Pendulum's sin/pow");
            foreach (double[] state in InjectedStates) {
                using GymPair pair = Pair();
                pair.Reset(0);
                pair.InjectState("state", state);
                for (int i = 0; i < 5; i++) {
                    using var action = ParityAction.Continuous(np.array(new[] { i % 2 == 0 ? 2.0f : -0.75f }));
                    pair.Step(action);
                }
            }
        }

        /// <summary>
        ///     States Gymnasium's code trips on, stepped and rendered: <c>None</c>, one value, three values (too many to unpack),
        ///     and an empty state, each with Python's exception and message.
        /// </summary>
        [TestMethod]
        public void GymNet_MalformedStates_RaiseLikeGymnasium() {
            SkipOffReferencePlatform("Pendulum's sin/pow");
            foreach (double[]? state in new[] { null, new[] { 0.1 }, new[] { 0.1, 0.2, 0.3 }, Array.Empty<double>() }) {
                using PyGymEnv python = PyGymEnv.Make(Id, renderMode: "rgb_array");
                var dotnet = new PendulumEnv(renderMode: "rgb_array");
                python.Reset(1);
                dotnet.Reset(1L).Observation.Dispose();
                if (state is null) {
                    python.WithUnwrapped(env => {
                        using PyObject none = PyConvert.ToPython(null);
                        Python.gymnasium.envs.classic_control.pendulum.PendulumEnv.state(env, none);
                    });
                } else {
                    python.SetAttribute("state", state);
                }

                dotnet.State = state;
                string what = state is null ? "None" : $"[{string.Join(", ", state.Select(v => PyBuiltins.FloatRepr(v)))}]";
                using (var action = ParityAction.Continuous(np.array(new[] { 0.5f }))) {
                    CallResult pythonStep = python.Step(action);
                    CallResult dotnetStep = new GymNetEnv(dotnet).Step(action.DotNet);
                    ByteContract.AssertSame($"step from state {what}", pythonStep.ToOutcome(), dotnetStep.ToOutcome());
                }

                RenderOutcome.AssertSame($"render() at state {what}", python.TryRender(), RenderOutcome.Capture(() => RenderGymNet(dotnet)));
            }
        }

        /// <summary>A changed <c>g</c> (9.81, through <c>make</c>'s kwargs): an episode of sampled torques.</summary>
        [TestMethod]
        public void GymNet_ChangedGravity_MatchesGymnasium() {
            SkipOffReferencePlatform("Pendulum's sin/pow");
            var kwargs = new Dictionary<string, object?> { ["g"] = 9.81 };
            using var pair = new GymPair(PyGymEnv.Make(Id, kwargs), new GymNetEnv(new PendulumEnv(g: 9.81)), stateAttribute: "state");
            pair.Run(13, ActionSources.Sampled(13), maxSteps: 200);
        }

        /// <summary>
        ///     Frames along episodes of sampled torques: after the reset (no arrow), then every step for 30 steps and every
        ///     tenth after, so the arrow is shrunk to every size and mirrored both ways.
        /// </summary>
        /// <param name="seed">The reset seed, also the action spaces' seed.</param>
        [TestMethod]
        [DataRow(0L)]
        [DataRow(42L)]
        public void GymNet_FramesAlongEpisodes_MatchGymnasium(long seed) {
            SkipOffReferencePlatform("Pendulum's rotation (cos/sin)");
            using GymPair pair = Pair("rgb_array");
            pair.Reset(seed);
            pair.CompareFrame("after reset");
            IActionSource actions = ActionSources.Sampled(seed);
            actions.Start(pair);
            while (pair.StepsTaken < 200) {
                using (ParityAction action = actions.Next(pair))
                    pair.Step(action);
                if (pair.StepsTaken <= 30 || pair.StepsTaken % 10 == 0)
                    pair.CompareFrame($"after step {pair.StepsTaken}");
            }
        }

        /// <summary>
        ///     <c>render()</c> after a reset into each of <see cref="InjectedStates"/> (no arrow), and after one step with a torque
        ///     of each sign and size from there (the arrow): the same frame or pygame's exception (NaN).
        /// </summary>
        [TestMethod]
        public void GymNet_FramesAtInjectedStates_MatchGymnasium() {
            SkipOffReferencePlatform("Pendulum's rotation (cos/sin)");
            float[] torques = { 2.0f, -2.0f, 0.013f, -0.5f, 0.0f };
            foreach (double[] state in InjectedStates) {
                foreach (float torque in torques) {
                    using PyGymEnv python = PyGymEnv.Make(Id, renderMode: "rgb_array");
                    var dotnet = new PendulumEnv(renderMode: "rgb_array");
                    python.Reset(2);
                    dotnet.Reset(2L).Observation.Dispose();
                    python.SetAttribute("state", state);
                    dotnet.State = (double[])state.Clone();
                    string at = $"[{string.Join(", ", state.Select(v => PyBuiltins.FloatRepr(v)))}]";
                    RenderOutcome.AssertSame($"render() at state {at}", python.TryRender(), RenderOutcome.Capture(() => RenderGymNet(dotnet)));
                    using (var action = ParityAction.Continuous(np.array(new[] { torque }))) {
                        CallResult pythonStep = python.Step(action);
                        CallResult dotnetStep = new GymNetEnv(dotnet).Step(action.DotNet);
                        ByteContract.AssertSame($"step({torque.ToString("R", CultureInfo.InvariantCulture)}) from state {at}", pythonStep.ToOutcome(), dotnetStep.ToOutcome());
                    }

                    RenderOutcome.AssertSame($"render() after step({torque.ToString("R", CultureInfo.InvariantCulture)}) from state {at}", python.TryRender(), RenderOutcome.Capture(() => RenderGymNet(dotnet)));
                }
            }
        }

        /// <summary>
        ///     The render errors of a missing attribute: before any reset (no <c>state</c>), and with a state injected before any
        ///     reset or step (no <c>last_u</c>, raised after the rod was drawn), as in Gymnasium.
        /// </summary>
        [TestMethod]
        public void GymNet_RenderWithoutAttributes_RaisesLikeGymnasium() {
            using (PyGymEnv python = PyGymEnv.Make(Id, renderMode: "rgb_array")) {
                var dotnet = new PendulumEnv(renderMode: "rgb_array");
                RenderOutcome pythonOutcome = python.TryRender();
                Assert.IsNotNull(pythonOutcome.Raised, $"Gymnasium's render() before reset gave {pythonOutcome}.");
                RenderOutcome.AssertSame("render() before reset", pythonOutcome, RenderOutcome.Capture(() => RenderGymNet(dotnet)));
            }

            using (PyGymEnv python = PyGymEnv.Make(Id, renderMode: "rgb_array")) {
                var dotnet = new PendulumEnv(renderMode: "rgb_array");
                python.SetAttribute("state", new[] { 0.5, 0.1 });
                dotnet.State = new[] { 0.5, 0.1 };
                RenderOutcome pythonOutcome = python.TryRender();
                Assert.IsNotNull(pythonOutcome.Raised, $"Gymnasium's render() without last_u gave {pythonOutcome}.");
                RenderOutcome.AssertSame("render() of an injected state before any reset", pythonOutcome, RenderOutcome.Capture(() => RenderGymNet(dotnet)));
            }
        }

        /// <summary>
        ///     The spaces: <c>Box([-1, -1, -8], [1, 1, 8], float32)</c> observations and <c>Box(-2.0, 2.0, (1,), float32)</c>
        ///     torques, with seeded samples and the generator behind them.
        /// </summary>
        [TestMethod]
        public void GymNet_Spaces_MatchGymnasium() {
            var env = new PendulumEnv();
            ClassicSpaces.AssertBoxSame($"{Id} observation_space", Id, Python.gymnasium.Env.observation_space, (Box)env.ObservationSpace);
            ClassicSpaces.AssertActionSpaceSame(Id, env.ActionSpace);
        }

        /// <summary><c>metadata</c>: <c>{'render_modes': ['human', 'rgb_array'], 'render_fps': 30}</c>.</summary>
        [TestMethod]
        public void GymNet_Metadata_MatchesGymnasium() {
            Outcome python = Outcome.CapturePython(() => {
                using PyObject made = Python.gymnasium.make(Id);
                using PyObject unwrapped = Python.gymnasium.Env.unwrapped(made);
                return Python.gymnasium.Env.metadata(unwrapped);
            });
            ByteContract.AssertSame($"{Id} metadata", python, Outcome.Returned(ValueFacts.Of(new PendulumEnv().Metadata)));
        }

        /// <summary><c>make("Pendulum-v1")</c>'s whole stack against <c>Gymnasium.Make</c> with sampled torques through truncation at 200 steps.</summary>
        [TestMethod]
        public void GymNet_MakeStack_MatchesGymnasiumThroughTruncation() {
            SkipOffReferencePlatform("Pendulum's sin/pow");
            using var pair = new GymPair(PyGymEnv.Make(Id, viaMake: true), new GymNetEnv(Gymnasium.Make(Id)), stateAttribute: "state");
            PairRun run = pair.Run(21, ActionSources.Sampled(21), maxSteps: 300);
            Assert.AreEqual(200, run.Steps, $"make's TimeLimit must truncate at 200: {run}");
        }

        /// <summary>A lockstep pair of Gymnasium's unwrapped env and Gym.NET's, comparing the float64 state.</summary>
        /// <param name="renderMode">The render mode of both, or <see langword="null"/>.</param>
        /// <returns>The pair; the caller disposes it.</returns>
        private static GymPair Pair(string? renderMode = null) =>
            new(PyGymEnv.Make(Id, renderMode: renderMode), new GymNetEnv(new PendulumEnv(renderMode: renderMode)), stateAttribute: "state");

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
