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
using Gym.Tests.Parity.Reference;
using Gym.Tests.Parity.Session;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace Gym.Tests.Parity.Tests.Envs {
    /// <summary>
    ///     CartPole-v1 against Gymnasium's <c>CartPoleEnv</c> (<c>gymnasium/envs/classic_control/cartpole.py</c> @ 9e04324),
    ///     live and in lockstep: first the float64 reference transcription (the harness's positive control), then Gym.NET's
    ///     own <see cref="CartPoleEnv"/>, the port, across its whole surface: reset (with options), dynamics (both
    ///     integrators, a changed <c>tau</c>), rewards (both reward schemes), termination and what follows it, the warning,
    ///     assertion messages, spaces, metadata, and <c>rgb_array</c> frames, including states pygame cannot draw.
    /// </summary>
    /// <remarks>
    ///     Every lockstep here compares the float64 <c>state</c> after each call too (Gymnasium's <c>env.unwrapped.state</c>,
    ///     Gym.NET's <see cref="CartPoleEnv.State"/>), so dynamics are compared at full precision even where the float32
    ///     observation would round a difference away. Cells that step or draw a rotated pole are pinned to the reference
    ///     platform: they rest on the platform's <c>cos</c>/<c>sin</c>.
    /// </remarks>
    [TestClass]
    public sealed class CartPoleLiveParityTests : ParityTestBase {
        /// <summary>The CartPole states of the port plan's pixel differential (A.9): centered, tilted, and left of center.</summary>
        public static readonly double[][] FrameStates = {
            new[] { 0.0, 0.0, 0.0, 0.0 },
            new[] { 0.5, 0.0, 0.15, 0.0 },
            new[] { -1.2, 0.0, -0.1, 0.0 },
        };

        /// <summary>
        ///     States at the edges of what can be drawn: near the thresholds, off-screen, the pole at exact and nearly exact
        ///     quarter turns (pygame snaps rotations within 1e-6 of one), a cart whose polygon wraps around Sint16 (x ≈ 259.6),
        ///     and states pygame refuses: a circle beyond Sint16 (x = 300), a polygon beyond a C long (x = 1e10), NaN and
        ///     infinite coordinates.
        /// </summary>
        public static readonly double[][] ExtremeStates = {
            new[] { 2.39, 0.0, 0.2, 0.0 },
            new[] { -2.4, 0.0, -0.2094, 0.0 },
            new[] { 0.0, 0.0, Math.PI / 2, 0.0 },
            new[] { 0.0, 0.0, -Math.PI / 2, 0.0 },
            new[] { 0.0, 0.0, Math.PI, 0.0 },
            new[] { 0.0, 0.0, 1e-7, 0.0 },
            new[] { 0.0, 0.0, -1e-7, 0.0 },
            new[] { 0.0, 0.0, 2e-6, 0.0 },
            new[] { 0.0, 0.0, 100.0, 0.0 },
            new[] { -0.0, 0.0, -0.0, 0.0 },
            new[] { 3.0, 0.0, 0.5, 0.0 },
            new[] { -2.6, 0.0, -0.3, 0.0 },
            new[] { 259.6, 0.0, 0.0, 0.0 },
            new[] { -264.5, 0.0, 0.05, 0.0 },
            new[] { 300.0, 0.0, 0.0, 0.0 },
            new[] { 1e10, 0.0, 0.0, 0.0 },
            new[] { double.NaN, 0.0, 0.0, 0.0 },
            new[] { 0.0, 0.0, double.NaN, 0.0 },
            new[] { 0.0, 0.0, double.PositiveInfinity, 0.0 },
            new[] { double.NegativeInfinity, 0.0, 0.0, 0.0 },
        };

        /// <summary>The reset options compared: valid bounds, every kind of rejected bound, and bounds the generator refuses.</summary>
        private static readonly (string Python, Dictionary<string, object?> Options)[] ResetOptions = {
            ("{}", new Dictionary<string, object?>()),
            ("{'low': -0.1, 'high': 0.1}", new Dictionary<string, object?> { ["low"] = -0.1, ["high"] = 0.1 }),
            ("{'low': 0.2}", new Dictionary<string, object?> { ["low"] = 0.2 }),
            ("{'low': 'abc'}", new Dictionary<string, object?> { ["low"] = "abc" }),
            ("{'low': None}", new Dictionary<string, object?> { ["low"] = null }),
            ("{'high': -1.0}", new Dictionary<string, object?> { ["high"] = -1.0 }),
            ("{'low': 0.1, 'high': -0.1}", new Dictionary<string, object?> { ["low"] = 0.1, ["high"] = -0.1 }),
            ("{'low': nan}", new Dictionary<string, object?> { ["low"] = double.NaN }),
            ("{'low': -inf}", new Dictionary<string, object?> { ["low"] = double.NegativeInfinity }),
            ("{'low': 1e-05, 'high': 1e+16}", new Dictionary<string, object?> { ["low"] = 1e-5, ["high"] = 1e16 }),
            ("{'low': '-0.5', 'high': '1_0'}", new Dictionary<string, object?> { ["low"] = "-0.5", ["high"] = "1_0" }),
            ("{'low': ' 1e-3 ', 'high': 'Infinity'}", new Dictionary<string, object?> { ["low"] = " 1e-3 ", ["high"] = "Infinity" }),
            ("{'low': 10**400}", new Dictionary<string, object?> { ["low"] = BigInteger.Pow(10, 400) }),
            ("{'low': 0.05, 'high': 0.05}", new Dictionary<string, object?> { ["low"] = 0.05, ["high"] = 0.05 }),
            ("{'low': -1, 'high': 1}", new Dictionary<string, object?> { ["low"] = -1L, ["high"] = 1L }),
            ("{'low': True, 'high': 2}", new Dictionary<string, object?> { ["low"] = true, ["high"] = 2L }),
            ("{'other': 1}", new Dictionary<string, object?> { ["other"] = 1L }),
        };

        /// <summary>
        ///     The reference transcription runs a whole episode in lockstep with Gymnasium on each of A.7's seeds: 500 steps of
        ///     the balancing policy, every observation, reward, flag, info, float64 state and PCG64 state identical.
        /// </summary>
        /// <param name="seed">The reset seed.</param>
        [TestMethod]
        [DataRow(0L)]
        [DataRow(1L)]
        [DataRow(42L)]
        [DataRow(1234L)]
        public void Reference_MatchesGymnasium_ForAWholeEpisode(long seed) {
            SkipOffReferencePlatform("CartPole's cos/sin");
            using var pair = new GymPair(PyGymEnv.Make("CartPole-v1"), CartPoleReference.Faithful(), stateAttribute: "state");
            PairRun run = pair.Run(seed, ActionSources.Policy("cartpole_balance"), maxSteps: 500);
            TestContext.WriteLine(run.ToString());
            Assert.AreEqual(500, run.Steps);
        }

        /// <summary>
        ///     Termination and what follows it, for the transcription: pushing right every step ends the episode, with reward
        ///     1.0 on the terminating step, and every further step returns reward 0.0 and <c>terminated</c>.
        /// </summary>
        [TestMethod]
        public void Reference_MatchesGymnasium_ThroughAndPastTermination() {
            SkipOffReferencePlatform("CartPole's cos/sin");
            using var pair = new GymPair(PyGymEnv.Make("CartPole-v1"), CartPoleReference.Faithful(), stateAttribute: "state");
            PairRun run = pair.Run(3, ActionSources.Constant(1), maxSteps: 500);
            Assert.IsTrue(run.Ended, "Pushing right every step must topple the pole.");
            StepPastTermination(pair, 5);
            TestContext.WriteLine($"{run}; then 5 more steps past termination, identical");
        }

        /// <summary>Gym.NET's <c>Reset(seed)</c> against <c>reset(seed=seed)</c>: observation, info, state and generator.</summary>
        /// <param name="seed">The reset seed.</param>
        [TestMethod]
        [DataRow(0L)]
        [DataRow(1L)]
        [DataRow(42L)]
        [DataRow(1234L)]
        public void GymNet_Reset_MatchesGymnasium(long seed) {
            using var pair = new GymPair(PyGymEnv.Make("CartPole-v1"), new GymNetEnv(new CartPoleEnv()), stateAttribute: "state");
            pair.Reset(seed);
        }

        /// <summary>Gym.NET's CartPole in lockstep with Gymnasium's for a whole episode of the balancing policy, on each of A.7's seeds.</summary>
        /// <param name="seed">The reset seed.</param>
        [TestMethod]
        [DataRow(0L)]
        [DataRow(1L)]
        [DataRow(42L)]
        [DataRow(1234L)]
        public void GymNet_Episode_MatchesGymnasium(long seed) {
            SkipOffReferencePlatform("CartPole's cos/sin");
            using var pair = new GymPair(PyGymEnv.Make("CartPole-v1"), new GymNetEnv(new CartPoleEnv()), stateAttribute: "state");
            PairRun run = pair.Run(seed, ActionSources.Policy("cartpole_balance"), maxSteps: 500);
            Assert.AreEqual(500, run.Steps);
        }

        /// <summary>
        ///     Episodes of seeded random actions (both action spaces sampled and compared), which topple the pole within tens of
        ///     steps at varied angles, and then five steps past termination, for many seeds.
        /// </summary>
        /// <param name="seed">The reset seed, also the action spaces' seed.</param>
        [TestMethod]
        [DataRow(0L)]
        [DataRow(3L)]
        [DataRow(17L)]
        [DataRow(99L)]
        [DataRow(2026L)]
        public void GymNet_RandomEpisodes_MatchGymnasium(long seed) {
            SkipOffReferencePlatform("CartPole's cos/sin");
            using var pair = new GymPair(PyGymEnv.Make("CartPole-v1"), new GymNetEnv(new CartPoleEnv()), stateAttribute: "state");
            PairRun run = pair.Run(seed, ActionSources.Sampled(seed), maxSteps: 500);
            Assert.IsTrue(run.Ended, "Random actions must topple the pole.");
            StepPastTermination(pair, 5);
        }

        /// <summary>
        ///     Termination and past it: pushing right every step ends the episode with reward 1.0, and further steps return
        ///     0.0 and <c>terminated</c>, with <c>steps_beyond_terminated</c> counting up alongside.
        /// </summary>
        [TestMethod]
        public void GymNet_ThroughAndPastTermination_MatchesGymnasium() {
            SkipOffReferencePlatform("CartPole's cos/sin");
            using var pair = new GymPair(PyGymEnv.Make("CartPole-v1"), new GymNetEnv(new CartPoleEnv()), stateAttribute: "state");
            PairRun run = pair.Run(3, ActionSources.Constant(1), maxSteps: 500);
            Assert.IsTrue(run.Ended);
            for (int i = 0; i < 5; i++) {
                using var action = ParityAction.Discrete(i % 2);
                pair.Step(action);
                ByteContract.AssertSame($"steps_beyond_terminated after {i + 1} more steps", pair.Python.GetAttribute("steps_beyond_terminated"), Outcome.Returned(pair.DotNet.GetAttribute("steps_beyond_terminated")!));
            }
        }

        /// <summary>
        ///     <c>sutton_barto_reward=True</c>: 0.0 per surviving step, -1.0 on the terminating step and after it, in lockstep
        ///     through termination and five steps beyond.
        /// </summary>
        [TestMethod]
        public void GymNet_SuttonBartoReward_MatchesGymnasium() {
            SkipOffReferencePlatform("CartPole's cos/sin");
            var kwargs = new Dictionary<string, object?> { ["sutton_barto_reward"] = true };
            using var pair = new GymPair(PyGymEnv.Make("CartPole-v1", kwargs), new GymNetEnv(new CartPoleEnv(suttonBartoReward: true)), stateAttribute: "state");
            PairRun run = pair.Run(5, ActionSources.Constant(0), maxSteps: 500);
            Assert.IsTrue(run.Ended);
            StepPastTermination(pair, 5);
        }

        /// <summary>
        ///     The semi-implicit Euler integrator (<c>kinematics_integrator</c> set to anything but <c>"euler"</c>), which
        ///     updates the velocities before the positions: a whole episode of the balancing policy in lockstep.
        /// </summary>
        [TestMethod]
        public void GymNet_SemiImplicitEuler_MatchesGymnasium() {
            SkipOffReferencePlatform("CartPole's cos/sin");
            PyGymEnv python = PyGymEnv.Make("CartPole-v1");
            python.WithUnwrapped(env => Python.gymnasium.envs.classic_control.cartpole.CartPoleEnv.kinematics_integrator(env, "semi-implicit"));
            var cartPole = new CartPoleEnv { KinematicsIntegrator = "semi-implicit" };
            using var pair = new GymPair(python, new GymNetEnv(cartPole), stateAttribute: "state");
            pair.Run(42, ActionSources.Policy("cartpole_balance"), maxSteps: 500);
        }

        /// <summary>A changed <c>tau</c> (0.05 s): the attribute is read at every step, as in Python.</summary>
        [TestMethod]
        public void GymNet_ChangedTau_MatchesGymnasium() {
            SkipOffReferencePlatform("CartPole's cos/sin");
            PyGymEnv python = PyGymEnv.Make("CartPole-v1");
            python.WithUnwrapped(env => Python.gymnasium.envs.classic_control.cartpole.CartPoleEnv.tau(env, 0.05));
            var cartPole = new CartPoleEnv { Tau = 0.05 };
            using var pair = new GymPair(python, new GymNetEnv(cartPole), stateAttribute: "state");
            pair.Run(7, ActionSources.Sampled(7), maxSteps: 500);
        }

        /// <summary>
        ///     One step of Gym.NET's dynamics from a state written into both envs (state injection), isolated from reset and the
        ///     generator, and then 50 more open-loop steps.
        /// </summary>
        [TestMethod]
        public void GymNet_Dynamics_MatchGymnasium_FromAnInjectedState() {
            SkipOffReferencePlatform("CartPole's cos/sin");
            using var pair = new GymPair(PyGymEnv.Make("CartPole-v1"), new GymNetEnv(new CartPoleEnv()), stateAttribute: "state");
            pair.InjectState("state", 0.01, -0.02, 0.03, 0.04);
            for (int i = 0; i < 51; i++) {
                using var action = ParityAction.Discrete((i / 3) % 2);
                pair.Step(action);
            }
        }

        /// <summary>
        ///     <c>reset(seed=..., options=...)</c> against Gym.NET's <see cref="CartPoleEnv.Reset(BigInteger?, IReadOnlyDictionary{string, object})"/>
        ///     for every case of <see cref="ResetOptions"/>, seeded and unseeded: the observation and info, or the exception
        ///     (<c>ValueError</c> for rejected options, <c>OverflowError</c> from the generator or an unconvertible int), then
        ///     the float64 state and the generator state, which a failed reset leaves freshly seeded and undrawn.
        /// </summary>
        [TestMethod]
        public void GymNet_ResetOptions_MatchGymnasium() {
            using PyRef python = PyRef.Of(() => {
                using PyObject made = Python.gymnasium.make("CartPole-v1");
                return Python.gymnasium.Env.unwrapped(made);
            });
            var dotnet = new CartPoleEnv();
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
                    ByteContract.AssertSame($"{call}: state", Outcome.CapturePython(() => Python.gymnasium.envs.classic_control.cartpole.CartPoleEnv.state(python.Object)), Outcome.Returned(StateFacts(dotnet)));
                    ByteContract.AssertSame($"{call}: np_random state", Outcome.CapturePython(() => {
                        using PyObject generator = Python.gymnasium.Env.np_random(python.Object);
                        using PyObject bitGenerator = Python.numpy.random.Generator.bit_generator(generator);
                        return Python.numpy.random.BitGenerator.state(bitGenerator);
                    }), Outcome.Returned(RngFacts.Of(dotnet.NpRandom)));
                }
            }
        }

        /// <summary>
        ///     Actions outside <c>Discrete(2)</c> raise <c>AssertionError</c> with Python's rendering of the action and its type,
        ///     <c>f"{action!r} ({type(action)}) invalid"</c>: ints, a float, NumPy scalars (NumSharp's 0-d arrays), an array, a
        ///     string, <c>None</c>, an int beyond int64, and a NumPy bool.
        /// </summary>
        [TestMethod]
        public void GymNet_InvalidActions_RaiseGymnasiumsAssertion() {
            var cases = new (string Python, Func<PyObject> Gymnasium, Func<object?> DotNet)[] {
                ("2", () => PyConvert.ToPython(2L), () => 2L),
                ("-1", () => PyConvert.ToPython(-1L), () => -1L),
                ("0.5", () => PyConvert.ToPython(0.5), () => 0.5),
                ("1.0", () => PyConvert.ToPython(1.0), () => 1.0),
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
                ("2**63", () => PyConvert.ToPython(1UL << 63), () => 1UL << 63),
            };

            using PyRef python = PyRef.Of(() => {
                using PyObject made = Python.gymnasium.make("CartPole-v1");
                return Python.gymnasium.Env.unwrapped(made);
            });
            Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, 0)).Expect("reset(seed=0)");
            var dotnet = new CartPoleEnv();
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
        ///     Stepping before any reset: a valid action raises <c>Call reset before using step method.</c>, an invalid one the
        ///     action's assertion first, as the source asserts the action before the state.
        /// </summary>
        [TestMethod]
        public void GymNet_StepBeforeReset_RaisesGymnasiumsAssertion() {
            var cases = new (string Python, Func<PyObject> Gymnasium, object DotNet)[] {
                ("1", () => PyConvert.ToPython(1L), 1L),
                ("True", () => PyConvert.ToPython(true), true),
                ("2", () => PyConvert.ToPython(2L), 2L),
            };

            foreach ((string text, Func<PyObject> gymnasium, object value) in cases) {
                using PyRef python = PyRef.Of(() => {
                    using PyObject made = Python.gymnasium.make("CartPole-v1");
                    return Python.gymnasium.Env.unwrapped(made);
                });
                IEnv dotnet = new CartPoleEnv();
                Outcome pythonStep = Outcome.CapturePython(() => {
                    using PyObject action = gymnasium();
                    return Python.gymnasium.Env.step(python.Object, action);
                });
                ByteContract.AssertSame($"step({text}) before reset", pythonStep, Outcome.Capture(() => dotnet.Step(value)));
            }
        }

        /// <summary>
        ///     The warning for stepping after termination: one <c>UserWarning</c> with Gymnasium's colourised text on the first
        ///     step past termination and none after it, captured on both sides as <c>catch_warnings(record=True)</c> records it.
        /// </summary>
        [TestMethod]
        public void GymNet_StepAfterTermination_WarnsLikeGymnasium() {
            SkipOffReferencePlatform("CartPole's cos/sin");
            using PyRef python = PyRef.Of(() => {
                using PyObject made = Python.gymnasium.make("CartPole-v1");
                return Python.gymnasium.Env.unwrapped(made);
            });
            var dotnet = new CartPoleEnv();
            Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, 3)).Expect("reset(seed=3)");
            dotnet.Reset(3L).Observation.Dispose();

            // Push right until the pole falls, on both sides (the dynamics are compared elsewhere; here they only lead there).
            bool terminated = false;
            while (!terminated) {
                ValueFacts step = Outcome.CapturePython(() => {
                    using PyObject action = PyConvert.ToPython(1L);
                    return Python.gymnasium.Env.step(python.Object, action);
                }).Expect("step(1)");
                terminated = step.Items[2].Boolean;
                Assert.AreEqual(terminated, dotnet.Step(1L).Terminated, "Both sides must terminate on the same step.");
            }

            const string stepsAfterTermination =
                "import warnings\n" +
                "with warnings.catch_warnings(record=True) as caught:\n" +
                "    warnings.simplefilter('always')\n" +
                "    for _ in range(3):\n" +
                "        env.step(1)\n" +
                "recorded = [(w.category.__name__, str(w.message)) for w in caught]\n";
            ValueFacts pythonWarnings = Outcome.CapturePython(() => Python.evalStmts(stepsAfterTermination, "recorded", ("env", python.Object))).Expect("three steps after termination");

            IReadOnlyList<WarningMessage> records;
            using (WarningCatcher catcher = Warnings.Catch()) {
                for (int i = 0; i < 3; i++)
                    dotnet.Step(1L);
                records = catcher.Records;
            }

            ValueFacts dotnetWarnings = ValueFacts.FromSequence(records.Select(w => ValueFacts.FromSequence(new[] { ValueFacts.FromText(w.Category.ToString()), ValueFacts.FromText(w.Message) }, "tuple")), "list");
            ByteContract.AssertSame("warnings of three steps after termination", pythonWarnings, dotnetWarnings);
            Assert.AreEqual(1, records.Count, "Exactly one warning: on the first step past termination.");
        }

        /// <summary>Gym.NET's <c>rgb_array</c> frames against pygame's at A.9's states, pixel for pixel.</summary>
        /// <param name="state">The index into <see cref="FrameStates"/>.</param>
        [TestMethod]
        [DataRow(0)]
        [DataRow(1)]
        [DataRow(2)]
        public void GymNet_Frame_MatchesGymnasium(int state) {
            SkipOffReferencePlatform("the pole's rotation (cos/sin)");
            using var pair = new GymPair(PyGymEnv.Make("CartPole-v1", renderMode: "rgb_array"), new GymNetEnv(new CartPoleEnv(renderMode: "rgb_array")), stateAttribute: "state");
            pair.InjectState("state", FrameStates[state]);
            pair.CompareFrame($"state {state}");
        }

        /// <summary>
        ///     Frames along whole episodes: the balancing policy with a frame every 10 steps, and random actions with a frame
        ///     at every step until the pole is down, the pole at every angle it passes through.
        /// </summary>
        /// <param name="seed">The reset seed.</param>
        [TestMethod]
        [DataRow(0L)]
        [DataRow(42L)]
        public void GymNet_FramesAlongEpisodes_MatchGymnasium(long seed) {
            SkipOffReferencePlatform("CartPole's cos/sin");
            using (var pair = new GymPair(PyGymEnv.Make("CartPole-v1", renderMode: "rgb_array"), new GymNetEnv(new CartPoleEnv(renderMode: "rgb_array")), stateAttribute: "state"))
                pair.Run(seed, ActionSources.Policy("cartpole_balance"), maxSteps: 500, frameEvery: 10);
            using (var pair = new GymPair(PyGymEnv.Make("CartPole-v1", renderMode: "rgb_array"), new GymNetEnv(new CartPoleEnv(renderMode: "rgb_array")), stateAttribute: "state"))
                pair.Run(seed, ActionSources.Sampled(seed), maxSteps: 500, frameEvery: 1);
        }

        /// <summary>
        ///     <c>render()</c> at the <see cref="ExtremeStates"/>: the same frame (clipped, wrapped, snapped) or the same
        ///     exception (<c>OverflowError</c>, <c>ValueError</c>) with pygame's message.
        /// </summary>
        [TestMethod]
        public void GymNet_FramesAtExtremeStates_MatchGymnasium() {
            SkipOffReferencePlatform("the pole's rotation (cos/sin)");
            foreach (double[] state in ExtremeStates) {
                using PyGymEnv python = PyGymEnv.Make("CartPole-v1", renderMode: "rgb_array");
                var dotnet = new CartPoleEnv(renderMode: "rgb_array");
                python.SetAttribute("state", state);
                dotnet.State = (double[])state.Clone();
                string what = $"render() at state [{string.Join(", ", state.Select(v => PyBuiltins.FloatRepr(v)))}]";
                RenderOutcome.AssertSame(what, python.TryRender(), RenderOutcome.Capture(() => RenderGymNet(dotnet)));
            }
        }

        /// <summary><c>render()</c> before any reset returns <c>None</c>; Gym.NET returns <see cref="RenderResult.None"/>.</summary>
        [TestMethod]
        public void GymNet_RenderBeforeReset_ReturnsNone() {
            using PyGymEnv python = PyGymEnv.Make("CartPole-v1", renderMode: "rgb_array");
            var dotnet = new CartPoleEnv(renderMode: "rgb_array");
            RenderOutcome pythonOutcome = python.TryRender();
            Assert.IsTrue(pythonOutcome.ReturnedNone, $"Gymnasium's render() before reset returned {pythonOutcome}.");
            RenderOutcome.AssertSame("render() before reset", pythonOutcome, RenderOutcome.Capture(() => RenderGymNet(dotnet)));
        }

        /// <summary>The observation space: <c>Box(-high, high, float32)</c>, <c>high = [4.8, inf, 0.41887903, inf]</c>, and its boundedness.</summary>
        [TestMethod]
        public void GymNet_ObservationSpace_MatchesGymnasium() {
            using PyRef python = PyRef.Of(() => MadeSpace("CartPole-v1", Python.gymnasium.Env.observation_space));
            var space = (Box)new CartPoleEnv().ObservationSpace;
            ByteContract.AssertSame("CartPole observation_space.high", Outcome.CapturePython(() => Python.gymnasium.spaces.box.Box.high(python.Object)), Outcome.Returned(ValueFacts.Of(space.High)));
            ByteContract.AssertSame("CartPole observation_space.low", Outcome.CapturePython(() => Python.gymnasium.spaces.box.Box.low(python.Object)), Outcome.Returned(ValueFacts.Of(space.Low)));
            ByteContract.AssertSame("CartPole observation_space.bounded_above", Outcome.CapturePython(() => Python.gymnasium.spaces.box.Box.bounded_above(python.Object)), Outcome.Returned(ValueFacts.Of(space.BoundedHigh)));
            ByteContract.AssertSame("CartPole observation_space.bounded_below", Outcome.CapturePython(() => Python.gymnasium.spaces.box.Box.bounded_below(python.Object)), Outcome.Returned(ValueFacts.Of(space.BoundedLow)));
            ByteContract.AssertSame("CartPole observation_space.dtype", Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.dtype(python.Object)), Outcome.Returned(ValueFacts.Of(space.DType)));
            ByteContract.AssertSame("CartPole observation_space.shape", Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.shape(python.Object)), Outcome.Returned(Spaces.DiscreteLiveParityTests.ShapeFacts(space.Shape)));
        }

        /// <summary>
        ///     The action space, <c>Discrete(2)</c>: its <c>n</c>, <c>start</c> and dtype, and seeded samples with the generator
        ///     state behind them, are Gymnasium's (the shape is covered by <see cref="Spaces.DiscreteLiveParityTests.Shape_MatchesGymnasium"/>).
        /// </summary>
        [TestMethod]
        public void GymNet_ActionSpace_MatchesGymnasium() {
            using PyRef python = PyRef.Of(() => MadeSpace("CartPole-v1", Python.gymnasium.Env.action_space));
            var space = (Discrete)new CartPoleEnv().ActionSpace;
            ByteContract.AssertSame("CartPole action_space.n", Outcome.CapturePython(() => Python.gymnasium.spaces.discrete.Discrete.n(python.Object)), Outcome.Returned(ValueFacts.Of(space.N)));
            ByteContract.AssertSame("CartPole action_space.start", Outcome.CapturePython(() => Python.gymnasium.spaces.discrete.Discrete.start(python.Object)), Outcome.Returned(ValueFacts.Of(space.Start)));
            ByteContract.AssertSame("CartPole action_space.dtype", Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.dtype(python.Object)), Outcome.Returned(ValueFacts.Of(space.DType)));

            Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.seed(python.Object, 5)).Expect("action_space.seed(5)");
            space.Seed(5);
            for (int i = 1; i <= 20; i++)
                ByteContract.AssertSame($"CartPole action_space.sample() #{i} after seed(5)", Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.sample(python.Object)), Outcome.Capture(() => space.Sample()));
            ByteContract.AssertSame("CartPole action_space generator state", RngFacts.OfPythonSpace(python.Object), Outcome.Returned(GeneratorProbe.StateOf(space)));
        }

        /// <summary><c>metadata</c>: <c>{'render_modes': ['human', 'rgb_array'], 'render_fps': 50}</c>, keys in that order.</summary>
        [TestMethod]
        public void GymNet_Metadata_MatchesGymnasium() {
            Outcome python = Outcome.CapturePython(() => {
                using PyObject made = Python.gymnasium.make("CartPole-v1");
                using PyObject unwrapped = Python.gymnasium.Env.unwrapped(made);
                return Python.gymnasium.Env.metadata(unwrapped);
            });
            ByteContract.AssertSame("CartPole metadata", python, Outcome.Returned(ValueFacts.Of(new CartPoleEnv().Metadata)));
        }

        /// <summary>
        ///     <c>gymnasium.make(env_id).unwrapped.&lt;space&gt;</c>: a space of the bare env, the counterpart of a Gym.NET env's
        ///     own space. Must run under the GIL.
        /// </summary>
        /// <param name="envId">The registered id.</param>
        /// <param name="space">The facade row that reads the space off the env (<c>Env.action_space</c> or <c>Env.observation_space</c>).</param>
        /// <param name="kwargs">Constructor keyword arguments in Python spelling, or <see langword="null"/>.</param>
        /// <returns>A new owned reference to the space; the envs themselves are released.</returns>
        /// <exception cref="PythonException">Thrown when making the env or reading the space raises.</exception>
        internal static PyObject MadeSpace(string envId, Func<PyObject, PyObject> space, IReadOnlyDictionary<string, object?>? kwargs = null) {
            using PyObject made = Python.gymnasium.make(envId, kwargs: kwargs);
            using PyObject unwrapped = Python.gymnasium.Env.unwrapped(made);
            return space(unwrapped);
        }

        /// <summary>Steps a pair that has terminated a few more times with alternating actions, compared as every step is.</summary>
        /// <param name="pair">The pair.</param>
        /// <param name="steps">How many steps.</param>
        private static void StepPastTermination(GymPair pair, int steps) {
            for (int i = 0; i < steps; i++) {
                using var action = ParityAction.Discrete(i % 2);
                pair.Step(action);
            }
        }

        /// <summary>Gym.NET's <see cref="CartPoleEnv.State"/> described as NumPy's float64 array (or <c>None</c>).</summary>
        /// <param name="env">The env.</param>
        /// <returns>The description.</returns>
        private static ValueFacts StateFacts(CartPoleEnv env) {
            if (env.State is null)
                return ValueFacts.NoneValue();
            using NDArray state = np.array(env.State);
            return ValueFacts.Of(state);
        }

        /// <summary>Gym.NET's <c>render()</c> of an <c>rgb_array</c> env as a frame, or <see langword="null"/> when it returned <c>None</c>.</summary>
        /// <param name="env">The env, built with <c>render_mode="rgb_array"</c>.</param>
        /// <returns>The frame, or <see langword="null"/>.</returns>
        /// <exception cref="InvalidOperationException">Thrown when the rendering is neither a frame nor <c>None</c>.</exception>
        private static Frame? RenderGymNet(CartPoleEnv env) {
            switch (env.Render()) {
                case RenderResult.NoneResult:
                    return null;
                case RenderResult.Rgb rgb: {
                    using NDArray frame = rgb.Frame;
                    return new Frame((int)frame.shape[0], (int)frame.shape[1], frame.ToArray<byte>());
                }
                case var other:
                    throw new InvalidOperationException($"CartPole in rgb_array mode rendered {other}.");
            }
        }
    }
}
