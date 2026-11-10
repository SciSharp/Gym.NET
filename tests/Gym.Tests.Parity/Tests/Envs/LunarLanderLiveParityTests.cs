using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Environments.Envs.Box2D;
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
    ///     LunarLander-v3 and LunarLanderContinuous-v3 against Gymnasium's <c>LunarLander</c>, live and in lockstep with
    ///     Gym.NET's <see cref="LunarLanderEnv"/> on its managed Box2D: resets, whole episodes of Gymnasium's heuristic and of
    ///     sampled actions through termination, wind and turbulence, other gravities, every refused action and constructor
    ///     argument with Gymnasium's message and warnings, <c>heuristic</c> itself, <c>rgb_array</c> frames with the exhaust
    ///     particles, spaces, metadata and <c>make</c>'s stacks.
    /// </summary>
    /// <remarks>
    ///     Every lockstep compares, after each call, the float32 observation, the float64 reward (the difference of two shaping
    ///     terms computed from the bodies' float32 state in float64, so it pins that state bit for bit), the flags, the info and
    ///     the generator's state. Cells that step or draw are pinned to the reference platform: Box2D's rotations call the C
    ///     runtime's <c>sinf</c>/<c>cosf</c> and the env its <c>sin</c>/<c>cos</c>/<c>tanh</c>. Reference:
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/lunar_lander.py#L79">gymnasium/envs/box2d/lunar_lander.py</see>.
    /// </remarks>
    [TestClass]
    public sealed class LunarLanderLiveParityTests : ParityTestBase {
        /// <summary>The discrete id.</summary>
        private const string Id = "LunarLander-v3";

        /// <summary>The platform qualification of every cell that steps or draws.</summary>
        private const string Platform = "Box2D's sinf/cosf and the env's sin/cos/tanh";

        /// <summary><c>reset(seed=seed)</c> of both variants: the no-op step's observation, the info and the generator after the reset's 16 draws.</summary>
        /// <param name="continuous">Which variant.</param>
        /// <param name="seed">The reset seed.</param>
        [TestMethod]
        [DataRow(false, 0L)]
        [DataRow(false, 1L)]
        [DataRow(false, 42L)]
        [DataRow(true, 0L)]
        [DataRow(true, 1234L)]
        public void GymNet_Reset_MatchesGymnasium(bool continuous, long seed) {
            SkipOffReferencePlatform(Platform);
            using GymPair pair = Pair(continuous);
            pair.Reset(seed);
        }

        /// <summary>
        ///     Whole episodes of Gymnasium's own heuristic (<c>lunar_lander.heuristic</c> on Gymnasium's observations, fed to
        ///     both sides) through the landing and the lander's sleep (+100), then five no-op steps past the end.
        /// </summary>
        /// <param name="continuous">Which variant.</param>
        /// <param name="seed">The reset seed.</param>
        [TestMethod]
        [DataRow(false, 0L)]
        [DataRow(false, 42L)]
        [DataRow(false, 7L)]
        [DataRow(true, 0L)]
        [DataRow(true, 42L)]
        [DataRow(true, 7L)]
        public void GymNet_HeuristicEpisodes_MatchGymnasiumThroughTermination(bool continuous, long seed) {
            SkipOffReferencePlatform(Platform);
            using GymPair pair = Pair(continuous);
            PairRun run = pair.Run(seed, ActionSources.Policy("lunar_lander_heuristic"), maxSteps: 1000);
            Assert.IsTrue(run.Ended, $"The heuristic must end the episode: {run}");
            for (int i = 0; i < 5; i++) {
                using ParityAction noop = continuous ? ParityAction.Continuous(np.array(new[] { 0f, 0f })) : ParityAction.Discrete(0);
                pair.Step(noop);
            }

            TestContext.WriteLine($"{run}; then 5 no-op steps past the end, identical");
        }

        /// <summary>Episodes of seeded sampled actions (both action spaces sampled and compared) until the crash or 400 steps.</summary>
        /// <param name="continuous">Which variant.</param>
        /// <param name="seed">The reset seed, also the action spaces' seed.</param>
        [TestMethod]
        [DataRow(false, 0L)]
        [DataRow(false, 3L)]
        [DataRow(true, 0L)]
        [DataRow(true, 5L)]
        public void GymNet_SampledEpisodes_MatchGymnasium(bool continuous, long seed) {
            SkipOffReferencePlatform(Platform);
            using GymPair pair = Pair(continuous);
            PairRun run = pair.Run(seed, ActionSources.Sampled(seed), maxSteps: 400);
            TestContext.WriteLine(run.ToString());
        }

        /// <summary>
        ///     Wind and turbulence (<c>enable_wind=True</c>): two more draws at the reset, the force and torque functions every
        ///     step until a leg touches down, at the default powers and at stronger ones, heuristic episodes of both variants.
        /// </summary>
        /// <param name="continuous">Which variant.</param>
        /// <param name="windPower">The wind power.</param>
        /// <param name="turbulencePower">The turbulence power.</param>
        [TestMethod]
        [DataRow(false, 15.0, 1.5)]
        [DataRow(true, 15.0, 1.5)]
        [DataRow(false, 19.5, 1.9)]
        [DataRow(true, 2.0, 0.25)]
        public void GymNet_WindEpisodes_MatchGymnasium(bool continuous, double windPower, double turbulencePower) {
            SkipOffReferencePlatform(Platform);
            var kwargs = new Dictionary<string, object?> { ["enable_wind"] = true, ["wind_power"] = windPower, ["turbulence_power"] = turbulencePower };
            using GymPair pair = Pair(continuous, kwargs: kwargs);
            PairRun run = pair.Run(11, ActionSources.Policy("lunar_lander_heuristic"), maxSteps: 1000);
            TestContext.WriteLine(run.ToString());
        }

        /// <summary>Other gravities (a light -3.7 and a heavy -11.5): heuristic episodes in each world.</summary>
        /// <param name="gravity">The gravity.</param>
        [TestMethod]
        [DataRow(-3.7)]
        [DataRow(-11.5)]
        public void GymNet_Gravity_MatchesGymnasium(double gravity) {
            SkipOffReferencePlatform(Platform);
            using GymPair pair = Pair(false, kwargs: new Dictionary<string, object?> { ["gravity"] = gravity });
            PairRun run = pair.Run(5, ActionSources.Policy("lunar_lander_heuristic"), maxSteps: 1000);
            TestContext.WriteLine(run.ToString());
        }

        /// <summary>
        ///     Frames along heuristic episodes of both variants, every fourth step: the terrain and sky, the hull and legs, the
        ///     flags redrawn per fixture, and the exhaust particles fading (a render mode makes the engines emit them, and every
        ///     render shortens their lives and destroys the expired ones, changing the world on both sides alike).
        /// </summary>
        /// <param name="continuous">Which variant.</param>
        /// <param name="seed">The reset seed.</param>
        [TestMethod]
        [DataRow(false, 0L)]
        [DataRow(true, 3L)]
        public void GymNet_FramesAlongEpisodes_MatchGymnasium(bool continuous, long seed) {
            SkipOffReferencePlatform(Platform);
            using GymPair pair = Pair(continuous, "rgb_array");
            PairRun run = pair.Run(seed, ActionSources.Policy("lunar_lander_heuristic"), maxSteps: 1000, frameEvery: 4);
            TestContext.WriteLine(run.ToString());
        }

        /// <summary>
        ///     Frames along sampled episodes, every third step: many particles at once, some not yet rendered, the circles of
        ///     every life.
        /// </summary>
        /// <param name="continuous">Which variant.</param>
        [TestMethod]
        [DataRow(false)]
        [DataRow(true)]
        public void GymNet_FramesOfSampledEpisodes_MatchGymnasium(bool continuous) {
            SkipOffReferencePlatform(Platform);
            using GymPair pair = Pair(continuous, "rgb_array");
            PairRun run = pair.Run(13, ActionSources.Sampled(13), maxSteps: 300, frameEvery: 3);
            TestContext.WriteLine(run.ToString());
        }

        /// <summary>
        ///     Discrete actions <c>Discrete(4)</c> refuses, through the untyped step, each with Gymnasium's message (trailing
        ///     space included): out of range both ways, a float, NumPy integers, a string, <c>None</c>.
        /// </summary>
        [TestMethod]
        public void GymNet_InvalidDiscreteActions_RaiseLikeGymnasium() {
            var cases = new (string Python, Func<PyObject> Gymnasium, Func<object?> DotNet)[] {
                ("4", () => PyConvert.ToPython(4L), () => 4L),
                ("-1", () => PyConvert.ToPython(-1L), () => -1L),
                ("1.0", () => PyConvert.ToPython(1.0), () => 1.0),
                ("np.int64(7)", () => Python.numpy.int64(7), () => NDArray.Scalar<long>(7L)),
                ("'1'", () => PyConvert.ToPython("1"), () => "1"),
                ("None", () => PyConvert.ToPython(null), () => null),
            };

            using PyRef python = PyRef.Of(() => {
                using PyObject made = Python.gymnasium.make(Id);
                return Python.gymnasium.Env.unwrapped(made);
            });
            var dotnet = new LunarLanderEnv();
            IEnv untyped = dotnet;
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
        ///     A step before any reset: Gymnasium's bare <c>assert self.lander is not None</c>, whose empty message NumSharp's
        ///     <c>AssertionError()</c> does not give yet (N-1).
        /// </summary>
        [TestMethod]
        [OpenBugs("N-1")]
        public void GymNet_StepBeforeReset_RaisesLikeGymnasium() {
            using PyRef python = PyRef.Of(() => {
                using PyObject made = Python.gymnasium.make(Id);
                return Python.gymnasium.Env.unwrapped(made);
            });
            IEnv dotnet = new LunarLanderEnv();
            Outcome pythonStep = Outcome.CapturePython(() => {
                using PyObject action = PyConvert.ToPython(0L);
                return Python.gymnasium.Env.step(python.Object, action);
            });
            ByteContract.AssertSame("step(0) before reset", pythonStep, Outcome.Capture(() => dotnet.Step(0L)));
        }

        /// <summary>
        ///     Continuous actions of every form, each followed by an ordinary step so their effect is compared too: float32 and
        ///     float64 throttles inside and beyond the clip, integer and boolean arrays, NaN (no engine fires), three elements
        ///     (the third ignored), and the refused ones, a 0-d array and a one-element array, the second after the main engine
        ///     already fired (NumPy's <c>IndexError</c> at <c>action[1]</c>), which the next step shows on both sides.
        /// </summary>
        [TestMethod]
        public void GymNet_ContinuousActionForms_MatchGymnasium() {
            SkipOffReferencePlatform(Platform);
            var actions = new (string What, Func<NDArray> Make)[] {
                ("float32 [0.7, -0.8]", () => np.array(new[] { 0.7f, -0.8f })),
                ("float64 [2.5, 3.0]", () => np.array(new[] { 2.5, 3.0 })),
                ("float64 [-5.0, -0.51]", () => np.array(new[] { -5.0, -0.51 })),
                ("int64 [1, -1]", () => np.array(new[] { 1L, -1L })),
                ("bool [True, False]", () => np.array(new[] { true, false })),
                ("float64 [nan, nan]", () => np.array(new[] { double.NaN, double.NaN })),
                ("float32 [0.3, 0.9, 7.0]", () => np.array(new[] { 0.3f, 0.9f, 7.0f })),
                ("0-d float64 0.5", () => NDArray.Scalar<double>(0.5)),
                ("float64 [0.9]", () => np.array(new[] { 0.9 })),
            };
            using GymPair pair = Pair(true);
            pair.Reset(21);
            foreach ((string what, Func<NDArray> make) in actions) {
                using (ParityAction action = ParityAction.Continuous(make()))
                    pair.Step(action);
                using ParityAction follow = ParityAction.Continuous(np.array(new[] { 0.4f, 0.6f }));
                pair.Step(follow);
                TestContext.WriteLine($"{what}: compared, and the step after it");
            }
        }

        /// <summary>
        ///     The constructor's checks with Gymnasium's texts: gravities outside (-12, 0) raise its <c>AssertionError</c>, and
        ///     wind or turbulence powers outside their recommended ranges warn through <c>gymnasium.logger</c> (text, category,
        ///     count) and are kept.
        /// </summary>
        [TestMethod]
        public void GymNet_ConstructorChecks_MatchGymnasium() {
            foreach (double gravity in new[] { -12.0, 0.0, 1.5, -20.25 }) {
                var kwargs = new Dictionary<string, object?> { ["gravity"] = gravity };
                Outcome python = Outcome.CapturePython(() => {
                    using PyObject made = Python.gymnasium.make(Id, kwargs: kwargs);
                    return PyConvert.ToPython(null);
                });
                Assert.IsNotNull(python.Raised, $"Gymnasium must refuse gravity={gravity}.");
                ByteContract.AssertSame($"LunarLander(gravity={gravity})", python, Outcome.Capture(() => {
                    // Refused in the constructor, so nothing is left to dispose; one built by mistake would be released.
                    new LunarLanderEnv(gravity: gravity).Dispose();
                    return null;
                }));
            }

            foreach ((double wind, double turbulence) in new[] { (25.0, 1.5), (15.0, -0.5), (-1.0, 3.0), (20.0, 2.0) }) {
                string statements =
                    "import warnings\n" +
                    "import gymnasium\n" +
                    "from gymnasium.envs.box2d.lunar_lander import LunarLander\n" +
                    "with warnings.catch_warnings(record=True) as caught:\n" +
                    "    warnings.simplefilter('always')\n" +
                    $"    LunarLander(wind_power={PyBuiltins.FloatRepr(wind)}, turbulence_power={PyBuiltins.FloatRepr(turbulence)})\n" +
                    "recorded = [(w.category.__name__, str(w.message)) for w in caught]\n";
                ValueFacts pythonWarnings = Outcome.CapturePython(() => Python.evalStmts(statements, "recorded")).Expect($"LunarLander(wind_power={wind}, turbulence_power={turbulence})");
                IReadOnlyList<WarningMessage> records;
                using (WarningCatcher catcher = Warnings.Catch()) {
                    new LunarLanderEnv(windPower: wind, turbulencePower: turbulence).Dispose();
                    records = catcher.Records;
                }

                ValueFacts dotnetWarnings = ValueFacts.FromSequence(records.Select(w => ValueFacts.FromSequence(new[] { ValueFacts.FromText(w.Category.ToString()), ValueFacts.FromText(w.Message) }, "tuple")), "list");
                ByteContract.AssertSame($"warnings of LunarLander(wind_power={wind}, turbulence_power={turbulence})", pythonWarnings, dotnetWarnings);
            }
        }

        /// <summary><c>render()</c> of an <c>rgb_array</c> env before any reset: the <c>AttributeError</c> for the missing <c>sky_polys</c>, both variants.</summary>
        [TestMethod]
        public void GymNet_RenderBeforeReset_RaisesLikeGymnasium() {
            foreach (bool continuous in new[] { false, true }) {
                using PyGymEnv python = PyGymEnv.Make(Id, Kwargs(continuous), renderMode: "rgb_array");
                var dotnet = new LunarLanderEnv(renderMode: "rgb_array", continuous: continuous);
                RenderOutcome.AssertSame($"render() before reset (continuous={continuous})", python.TryRender(), RenderOutcome.Capture(() => RenderGymNet(dotnet)));
            }
        }

        /// <summary>
        ///     Gymnasium's <c>heuristic(env, s)</c> against <see cref="LunarLanderEnv.Heuristic"/> on 2,000 float32 observations
        ///     of both variants: random states around the pad, beyond the ±0.4 angle clamp, with and without leg contact (the
        ///     continuous action is then float64, NumPy's promotion of the int 0), and NaNs.
        /// </summary>
        [TestMethod]
        public void GymNet_Heuristic_MatchesGymnasium() {
            var rng = new Random(20261111);
            foreach (bool continuous in new[] { false, true }) {
                using PyRef python = PyRef.Of(() => {
                    using PyObject made = Python.gymnasium.make(Id, kwargs: Kwargs(continuous));
                    return Python.gymnasium.Env.unwrapped(made);
                });
                var dotnet = new LunarLanderEnv(continuous: continuous);
                for (int i = 0; i < 1000; i++) {
                    float[] values = new float[8];
                    for (int k = 0; k < 6; k++)
                        values[k] = (float)((rng.NextDouble() * 2 - 1) * (i % 5 == 0 ? 3.0 : 1.0));
                    values[6] = rng.Next(3) == 0 ? 1f : 0f;
                    values[7] = rng.Next(3) == 0 ? 1f : 0f;
                    if (i % 97 == 5)
                        values[rng.Next(8)] = float.NaN;
                    using NDArray s = np.array(values);
                    Outcome pythonAction = Outcome.CapturePython(() => {
                        using PyObject observation = s.ToNumpyCopy();
                        using PyObject action = Python.gymnasium.envs.box2d.lunar_lander.heuristic(python.Object, observation);
                        // A discrete action is a Python int; Gym.NET's is a 0-d int64 array (DR-08), compared by value.
                        return continuous ? Python.numpy.asarray(action) : PyConvert.ToPython(action.As<long>());
                    });
                    Outcome dotnetAction = Outcome.Capture(() => {
                        NDArray action = LunarLanderEnv.Heuristic(dotnet, s);
                        if (continuous)
                            return action;
                        using (action)
                            return action.GetAtIndex<long>(0);
                    });
                    ByteContract.AssertSame($"heuristic(continuous={continuous}, [{string.Join(", ", values.Select(v => v.ToString("R", System.Globalization.CultureInfo.InvariantCulture)))}])", pythonAction, dotnetAction);
                }
            }
        }

        /// <summary>
        ///     The spaces: the v3 observation bounds (±2.5, ±2.5, ±10, ±10, ±2π, ±10, [-0, 1], [-0, 1], float32) of both
        ///     variants, <c>Discrete(4)</c>, and <c>Box(-1, 1, (2,), float32)</c> with seeded samples.
        /// </summary>
        /// <param name="continuous">Which variant.</param>
        [TestMethod]
        [DataRow(false)]
        [DataRow(true)]
        public void GymNet_Spaces_MatchGymnasium(bool continuous) {
            var env = new LunarLanderEnv(continuous: continuous);
            Dictionary<string, object?> kwargs = Kwargs(continuous);
            ClassicSpaces.AssertBoxSame($"{Id}(continuous={continuous}) observation_space", Id, Python.gymnasium.Env.observation_space, (Box)env.ObservationSpace, kwargs);
            ClassicSpaces.AssertActionSpaceSame(Id, env.ActionSpace, kwargs);
        }

        /// <summary><c>metadata</c>: <c>{'render_modes': ['human', 'rgb_array'], 'render_fps': 50}</c>.</summary>
        [TestMethod]
        public void GymNet_Metadata_MatchesGymnasium() {
            Outcome python = Outcome.CapturePython(() => {
                using PyObject made = Python.gymnasium.make(Id);
                using PyObject unwrapped = Python.gymnasium.Env.unwrapped(made);
                return Python.gymnasium.Env.metadata(unwrapped);
            });
            ByteContract.AssertSame($"{Id} metadata", python, Outcome.Returned(ValueFacts.Of(new LunarLanderEnv().Metadata)));
        }

        /// <summary>
        ///     <c>make</c>'s whole stack of both ids (<c>TimeLimit</c>, <c>OrderEnforcing</c>, <c>PassiveEnvChecker</c>) against
        ///     <c>Gymnasium.Make</c>, heuristic episodes through termination.
        /// </summary>
        /// <param name="id">The id.</param>
        [TestMethod]
        [DataRow("LunarLander-v3")]
        [DataRow("LunarLanderContinuous-v3")]
        public void GymNet_MakeStack_MatchesGymnasium(string id) {
            SkipOffReferencePlatform(Platform);
            using var pair = new GymPair(PyGymEnv.Make(id, viaMake: true), new GymNetEnv(Gymnasium.Make(id)));
            PairRun run = pair.Run(31, ActionSources.Policy("lunar_lander_heuristic"), maxSteps: 1100);
            Assert.IsTrue(run.Ended, $"The episode must end: {run}");
        }

        /// <summary><c>LunarLanderContinuous()</c>: the class only refuses, with Gymnasium's <c>gymnasium.error.Error</c> and its message.</summary>
        [TestMethod]
        public void GymNet_LunarLanderContinuousClass_RefusesLikeGymnasium() {
            Outcome python = Outcome.CapturePython(() => Python.evalStmts(
                "from gymnasium.envs.box2d.lunar_lander import LunarLanderContinuous\n" +
                "LunarLanderContinuous()\n" +
                "result = None\n", "result"));
            ByteContract.AssertSame("LunarLanderContinuous()", python, Outcome.Capture(() => {
                _ = new LunarLanderContinuousEnv();
                return null;
            }));
        }

        /// <summary>The kwargs of a variant, as <c>make</c> takes them.</summary>
        /// <param name="continuous">Which variant.</param>
        /// <param name="extra">More kwargs, or <see langword="null"/>.</param>
        /// <returns>The kwargs.</returns>
        private static Dictionary<string, object?> Kwargs(bool continuous, IReadOnlyDictionary<string, object?>? extra = null) {
            var kwargs = new Dictionary<string, object?> { ["continuous"] = continuous };
            if (extra != null) {
                foreach (KeyValuePair<string, object?> pair in extra)
                    kwargs[pair.Key] = pair.Value;
            }

            return kwargs;
        }

        /// <summary>
        ///     A lockstep pair of Gymnasium's unwrapped env and Gym.NET's, built with the same arguments.
        /// </summary>
        /// <param name="continuous">Which variant.</param>
        /// <param name="renderMode">The render mode of both, or <see langword="null"/>.</param>
        /// <param name="kwargs">More constructor arguments by Python name: <c>enable_wind</c>, <c>wind_power</c>, <c>turbulence_power</c>, <c>gravity</c>.</param>
        /// <returns>The pair; the caller disposes it.</returns>
        private static GymPair Pair(bool continuous, string? renderMode = null, IReadOnlyDictionary<string, object?>? kwargs = null) {
            Dictionary<string, object?> all = Kwargs(continuous, kwargs);
            double Get(string name, double fallback) => all.TryGetValue(name, out object? value) && value is double d ? d : fallback;
            var dotnet = new LunarLanderEnv(renderMode: renderMode, continuous: continuous, gravity: Get("gravity", -10.0),
                                            enableWind: all.TryGetValue("enable_wind", out object? wind) && wind is true,
                                            windPower: Get("wind_power", 15.0), turbulencePower: Get("turbulence_power", 1.5));
            return new GymPair(PyGymEnv.Make(Id, all, renderMode: renderMode), new GymNetEnv(dotnet));
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
