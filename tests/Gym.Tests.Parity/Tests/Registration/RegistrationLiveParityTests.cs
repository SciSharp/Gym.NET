using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Spaces;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Session;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Python.Runtime;
using Tuple = Gym.Spaces.Tuple;

namespace Gym.Tests.Parity.Tests.Registration {
    /// <summary>
    ///     The registry, <c>make</c> and the checkers (port plan P3) against Gymnasium's, live: the stack and spec
    ///     <c>make("CartPole-v1")</c> builds, the registry's errors (names, namespaces, versions, malformed ids) with their
    ///     suggestions, <c>EnvSpec</c>'s JSON both ways, <c>pprint_registry</c>'s text, and the passive observation check's
    ///     warnings and assertions.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Entry points differ by design (a .NET class against a Python module path), so the cells compare what does not name
    ///     one: the stack's <c>str</c>, <c>pprint</c> without entry points, and specs built with the same entry-point strings on
    ///     both sides. The error cells pick ids whose answer does not depend on which other envs are registered: Gymnasium's
    ///     registry holds every family, Gym.NET's the ones ported so far, and the "Did you mean" suggestions are drawn from all
    ///     of them.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/registration.py">gymnasium/envs/registration.py</see>, <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/passive_env_checker.py">gymnasium/utils/passive_env_checker.py</see>.</para>
    /// </remarks>
    [TestClass]
    public sealed class RegistrationLiveParityTests : ParityTestBase {
        /// <summary>
        ///     <c>make("CartPole-v1")</c>'s stack (<c>str(env)</c>: <c>TimeLimit</c>, <c>OrderEnforcing</c>,
        ///     <c>PassiveEnvChecker</c>, the env) and the specs of the stack and of the unwrapped env, printed.
        /// </summary>
        [TestMethod]
        public void Make_CartPole_StackAndSpecs_MatchGymnasium() {
            using PyRef python = PyRef.Of(() => Python.gymnasium.make("CartPole-v1"));
            using IEnv dotnet = Gymnasium.Make("CartPole-v1");

            ByteContract.AssertSame("str(make('CartPole-v1'))", Outcome.CapturePython(() => Python.builtins.str(python.Object)), Outcome.Returned(ValueFacts.FromText(dotnet.ToString()!)));
            ByteContract.AssertSame("env.spec.pprint()", Outcome.CapturePython(() => {
                using PyObject spec = Python.gymnasium.Env.spec(python.Object);
                return Python.gymnasium.envs.registration.EnvSpec.pprint(spec);
            }), Outcome.Returned(ValueFacts.FromText(dotnet.Spec!.Pprint(disablePrint: true)!)));
            ByteContract.AssertSame("env.unwrapped.spec.pprint()", Outcome.CapturePython(() => {
                using PyObject unwrapped = Python.gymnasium.Env.unwrapped(python.Object);
                using PyObject spec = Python.gymnasium.Env.spec(unwrapped);
                return Python.gymnasium.envs.registration.EnvSpec.pprint(spec);
            }), Outcome.Returned(ValueFacts.FromText(dotnet.Unwrapped.Spec!.Pprint(disablePrint: true)!)));
        }

        /// <summary>
        ///     The classic-control ids of P4: <c>str(make(id))</c> (the merged MountainCar prints as <c>Continuous_MountainCarEnv</c>
        ///     in continuous mode) and the printed specs of the stack and of the unwrapped env, which leave out the kwargs that
        ///     differ for <c>MountainCarContinuous-v0</c> (see <see cref="Spec_MountainCarContinuousKwargs_CarryContinuous_AsDr32Accepts"/>).
        /// </summary>
        /// <param name="id">The id.</param>
        [TestMethod]
        [DataRow("MountainCar-v0")]
        [DataRow("MountainCarContinuous-v0")]
        [DataRow("Pendulum-v1")]
        [DataRow("Acrobot-v1")]
        public void Make_ClassicControl_StackAndSpecs_MatchGymnasium(string id) {
            using PyRef python = PyRef.Of(() => Python.gymnasium.make(id));
            using IEnv dotnet = Gymnasium.Make(id);

            ByteContract.AssertSame($"str(make('{id}'))", Outcome.CapturePython(() => Python.builtins.str(python.Object)), Outcome.Returned(ValueFacts.FromText(dotnet.ToString()!)));
            ByteContract.AssertSame($"{id}: env.spec.pprint()", Outcome.CapturePython(() => {
                using PyObject spec = Python.gymnasium.Env.spec(python.Object);
                return Python.gymnasium.envs.registration.EnvSpec.pprint(spec);
            }), Outcome.Returned(ValueFacts.FromText(dotnet.Spec!.Pprint(disablePrint: true)!)));
            ByteContract.AssertSame($"{id}: env.unwrapped.spec.pprint()", Outcome.CapturePython(() => {
                using PyObject unwrapped = Python.gymnasium.Env.unwrapped(python.Object);
                using PyObject spec = Python.gymnasium.Env.spec(unwrapped);
                return Python.gymnasium.envs.registration.EnvSpec.pprint(spec);
            }), Outcome.Returned(ValueFacts.FromText(dotnet.Unwrapped.Spec!.Pprint(disablePrint: true)!)));
        }

        /// <summary>
        ///     The toy_text ids of P5: <c>str(make(id))</c> and the printed specs of the stack and of the unwrapped env (Blackjack's
        ///     kwargs in Gymnasium's order, <c>sab</c> then <c>natural</c>; FrozenLake's map names and thresholds; CliffWalking's
        ///     missing limit). Taxi-v4's printed threshold differs by type (see <see cref="Spec_TaxiRewardThreshold_IsFloat64"/>).
        /// </summary>
        /// <param name="id">The id.</param>
        [TestMethod]
        [DataRow("Blackjack-v1")]
        [DataRow("FrozenLake-v1")]
        [DataRow("FrozenLake8x8-v1")]
        [DataRow("CliffWalking-v1")]
        [DataRow("CliffWalkingSlippery-v1")]
        public void Make_ToyText_StackAndSpecs_MatchGymnasium(string id) {
            using PyRef python = PyRef.Of(() => Python.gymnasium.make(id));
            using IEnv dotnet = Gymnasium.Make(id);

            ByteContract.AssertSame($"str(make('{id}'))", Outcome.CapturePython(() => Python.builtins.str(python.Object)), Outcome.Returned(ValueFacts.FromText(dotnet.ToString()!)));
            ByteContract.AssertSame($"{id}: env.spec.pprint()", Outcome.CapturePython(() => {
                using PyObject spec = Python.gymnasium.Env.spec(python.Object);
                return Python.gymnasium.envs.registration.EnvSpec.pprint(spec);
            }), Outcome.Returned(ValueFacts.FromText(dotnet.Spec!.Pprint(disablePrint: true)!)));
            ByteContract.AssertSame($"{id}: env.unwrapped.spec.pprint()", Outcome.CapturePython(() => {
                using PyObject unwrapped = Python.gymnasium.Env.unwrapped(python.Object);
                using PyObject spec = Python.gymnasium.Env.spec(unwrapped);
                return Python.gymnasium.envs.registration.EnvSpec.pprint(spec);
            }), Outcome.Returned(ValueFacts.FromText(dotnet.Unwrapped.Spec!.Pprint(disablePrint: true)!)));
        }

        /// <summary>
        ///     Taxi-v4, pinned so it cannot drift: <c>str(make('Taxi-v4'))</c> is Gymnasium's, and so is the printed spec except
        ///     its threshold. Gymnasium registers <c>reward_threshold=8</c>, an <c>int</c> (its annotation says <c>float</c>);
        ///     Gym.NET's thresholds are float64, like its rewards, so the spec prints <c>reward_threshold=8.0</c>. The value is
        ///     the same: <c>float(spec.reward_threshold) == 8.0</c>.
        /// </summary>
        [TestMethod]
        public void Spec_TaxiRewardThreshold_IsFloat64() {
            using PyRef python = PyRef.Of(() => Python.gymnasium.make("Taxi-v4"));
            using IEnv dotnet = Gymnasium.Make("Taxi-v4");
            ByteContract.AssertSame("str(make('Taxi-v4'))", Outcome.CapturePython(() => Python.builtins.str(python.Object)), Outcome.Returned(ValueFacts.FromText(dotnet.ToString()!)));

            ValueFacts printed = Outcome.CapturePython(() => {
                using PyObject spec = Python.gymnasium.Env.spec(python.Object);
                return Python.gymnasium.envs.registration.EnvSpec.pprint(spec);
            }).Expect("Taxi-v4: env.spec.pprint()");
            StringAssert.Contains(printed.Text, "\nreward_threshold=8\n", "Gymnasium's Taxi-v4 registers an int threshold.");
            Assert.AreEqual(printed.Text!.Replace("\nreward_threshold=8\n", "\nreward_threshold=8.0\n", StringComparison.Ordinal), dotnet.Spec!.Pprint(disablePrint: true),
                "Gym.NET's spec prints Gymnasium's text with the threshold as a float64.");
            Assert.AreEqual(8.0, dotnet.Spec.RewardThreshold);
        }

        /// <summary>
        ///     DR-32's accepted deviation, pinned so it cannot drift: <c>MountainCarContinuous-v0</c>'s <c>EnvSpec.kwargs</c> is
        ///     empty in Gymnasium (its entry point is a class of its own) and <c>{'continuous': True}</c> in Gym.NET (one class
        ///     for both MountainCars); <c>MountainCar-v0</c>'s are equal (empty).
        /// </summary>
        [TestMethod]
        public void Spec_MountainCarContinuousKwargs_CarryContinuous_AsDr32Accepts() {
            ByteContract.AssertSame("spec('MountainCar-v0').kwargs", Outcome.CapturePython(() => {
                using PyObject spec = Python.gymnasium.spec("MountainCar-v0");
                return Python.gymnasium.envs.registration.EnvSpec.kwargs(spec);
            }), Outcome.Returned(ValueFacts.Of(Gymnasium.Spec("MountainCar-v0").Kwargs)));

            ValueFacts python = Outcome.CapturePython(() => {
                using PyObject spec = Python.gymnasium.spec("MountainCarContinuous-v0");
                return Python.gymnasium.envs.registration.EnvSpec.kwargs(spec);
            }).Expect("spec('MountainCarContinuous-v0').kwargs");
            Assert.AreEqual("{}", python.Render(), "Gymnasium's MountainCarContinuous-v0 has no kwargs.");
            Info kwargs = Gymnasium.Spec("MountainCarContinuous-v0").Kwargs;
            Assert.AreEqual(1, kwargs.Count, "Gym.NET's carries exactly one kwarg.");
            Assert.AreEqual(true, kwargs["continuous"], "...and it is continuous=True (DR-32).");
        }

        /// <summary>
        ///     <c>make</c>'s errors with Gymnasium's types and messages: a mistyped name (with its suggestion), a missing version,
        ///     a lower-case name, an unknown namespace, and malformed ids.
        /// </summary>
        [TestMethod]
        public void Make_Errors_MatchGymnasium() {
            foreach (string id in new[] { "CartPol-v1", "CartPole-v9", "cartpole-v1", "Nope/CartPole-v1", "“Breakout-v0”", "CartPole v1", "a:b:c" })
                ByteContract.AssertSame($"make({PyBuiltins.Repr(id)})", Outcome.CapturePython(() => Python.gymnasium.make(id)), Outcome.Capture(() => Gymnasium.Make(id)));
        }

        /// <summary>
        ///     <c>spec</c>'s version errors over ids registered on both sides for the cell: deprecated, missing and newer
        ///     versions, a default version, and a newer version in a namespace.
        /// </summary>
        [TestMethod]
        public void Spec_VersionErrors_MatchGymnasium() {
            string[] registered = { "ParityTestEnv-v0", "ParityTestEnv-v15", "ParityTestEnv-v9", "paritytest/TestEnv", "paritytest/VersionedEnv-v5" };
            try {
                foreach (string id in registered) {
                    ByteContract.AssertSame($"register({PyBuiltins.Repr(id)})", Outcome.CapturePython(() => Python.gymnasium.register(id, "no-entry-point")),
                        Outcome.Capture(() => {
                            Gymnasium.Register(id, "no-entry-point");
                            return null;
                        }));
                }

                foreach (string id in new[] { "ParityTestEnv-v1", "ParityTestEnv-v1000", "ParityTestEnv-v15", "paritytest/TestEnv-v0", "paritytest/VersionedEnv-v9", "paritytest/VersionedEnv-v4" }) {
                    ByteContract.AssertSame($"spec({PyBuiltins.Repr(id)}).pprint()", Outcome.CapturePython(() => {
                        using PyObject spec = Python.gymnasium.spec(id);
                        return Python.gymnasium.envs.registration.EnvSpec.pprint(spec, include_entry_points: true, print_all: true);
                    }), Outcome.Capture(() => Gymnasium.Spec(id).Pprint(disablePrint: true, includeEntryPoints: true, printAll: true)));
                }

                // Registering a name both ways is refused, with the same message.
                ByteContract.AssertSame("register('paritytest/TestEnv-v1')", Outcome.CapturePython(() => Python.gymnasium.register("paritytest/TestEnv-v1", "no-entry-point")),
                    Outcome.Capture(() => {
                        Gymnasium.Register("paritytest/TestEnv-v1", "no-entry-point");
                        return null;
                    }));
            } finally {
                foreach (string id in registered) {
                    PythonRuntime.Run(() => {
                        using PyObject removed = Python.gymnasium.registry.pop(id);
                    });
                    Gymnasium.Registry.Remove(id);
                }
            }
        }

        /// <summary>
        ///     <c>EnvSpec.to_json</c> byte for byte, and <c>from_json</c> back to the same spec, for specs with every kind of
        ///     JSON value among their kwargs, wrapper specs, and defaults.
        /// </summary>
        [TestMethod]
        public void EnvSpec_Json_MatchesGymnasium() {
            const string setup =
                "from gymnasium.envs.registration import EnvSpec, WrapperSpec\n" +
                "specs = [\n" +
                "    EnvSpec('CartPole-v1', 'gymnasium.envs.classic_control.cartpole:CartPoleEnv', reward_threshold=475.0, max_episode_steps=500),\n" +
                "    EnvSpec('ns/Name-v3', 'module:Class', nondeterministic=True, order_enforce=False, disable_env_checker=True,\n" +
                "            kwargs={'int': 7, 'big': 2**70, 'float': 0.1, 'neg': -1e-300, 'str': 'é\\n\"quoted\"\\t\\u0001', 'none': None,\n" +
                "                    'list': [1, 2.5, 'x', None, True, False], 'nested': {'a': {'b': [{}]}}},\n" +
                "            additional_wrappers=(WrapperSpec('TimeLimit', 'gymnasium.wrappers.common:TimeLimit', {'max_episode_steps': 10}),\n" +
                "                                 WrapperSpec('Other', 'x:Other', None)),\n" +
                "            vector_entry_point='module:VectorClass'),\n" +
                "    EnvSpec('Plain'),\n" +
                "]\n";
            var dotnetSpecs = new[] {
                new EnvSpec("CartPole-v1", "gymnasium.envs.classic_control.cartpole:CartPoleEnv", rewardThreshold: 475.0, maxEpisodeSteps: 500),
                new EnvSpec("ns/Name-v3", "module:Class", nondeterministic: true, orderEnforce: false, disableEnvChecker: true,
                    kwargs: new Info {
                        ["int"] = 7L, ["big"] = System.Numerics.BigInteger.Pow(2, 70), ["float"] = 0.1, ["neg"] = -1e-300, ["str"] = "é\n\"quoted\"\t\u0001", ["none"] = null,
                        ["list"] = new List<object?> { 1L, 2.5, "x", null, true, false },
                        ["nested"] = new Info { ["a"] = new Info { ["b"] = new List<object?> { new Info() } } },
                    },
                    additionalWrappers: new[] {
                        new WrapperSpec("TimeLimit", "gymnasium.wrappers.common:TimeLimit", new Info { ["max_episode_steps"] = 10L }),
                        new WrapperSpec("Other", "x:Other", null),
                    },
                    vectorEntryPoint: "module:VectorClass"),
                new EnvSpec("Plain"),
            };

            for (int i = 0; i < dotnetSpecs.Length; i++) {
                int index = i;
                Outcome pythonJson = Outcome.CapturePython(() => {
                    using PyObject specs = Python.evalStmts(setup, "specs");
                    using PyObject spec = specs[index];
                    return Python.gymnasium.envs.registration.EnvSpec.to_json(spec);
                });
                string json = dotnetSpecs[index].ToJson();
                ByteContract.AssertSame($"specs[{index}].to_json()", pythonJson, Outcome.Returned(ValueFacts.FromText(json)));

                ByteContract.AssertSame($"repr(EnvSpec.from_json(specs[{index}].to_json()))", Outcome.CapturePython(() => {
                    using PyObject spec = Python.gymnasium.envs.registration.EnvSpec.from_json(json);
                    return Python.builtins.repr(spec);
                }), Outcome.Capture(() => EnvSpec.FromJson(json).ToString()));
            }

            // Malformed JSON and fields that are not a spec's fail alike.
            foreach (string json in new[] { "{\"id\": \"X\"}", "{\"id\": \"X\", \"additional_wrappers\": [], \"bogus\": 1}", "[1, 2" })
                ByteContract.AssertSame($"EnvSpec.from_json({PyBuiltins.Repr(json)})", Outcome.CapturePython(() => Python.gymnasium.envs.registration.EnvSpec.from_json(json)),
                    Outcome.Capture(() => EnvSpec.FromJson(json)));
        }

        /// <summary><c>pprint_registry</c>'s text over Gymnasium's example registries: namespaces from entry points, columns, exclusions.</summary>
        [TestMethod]
        public void PprintRegistry_MatchesGymnasium() {
            const string setup =
                "from gymnasium.envs.registration import EnvSpec\n" +
                "ep = 'gymnasium.envs.classic_control.cartpole:CartPoleEnv'\n" +
                "registry = {\n" +
                "    'CartPole-v0': EnvSpec('CartPole-v0', ep),\n" +
                "    'CartPole-v1': EnvSpec('CartPole-v1', 'gymnasium.envs.classic_control:CartPoleEnv'),\n" +
                "    'CartPole-v2': EnvSpec('CartPole-v2', 'gymnasium.cartpole:CartPoleEnv'),\n" +
                "    'Single-v0': EnvSpec('Single-v0', 'single:Env'),\n" +
                "    'Test/CartPole-v0': EnvSpec('Test/CartPole-v0', ep),\n" +
                "    'Test/LongerName-v1': EnvSpec('Test/LongerName-v1', ep),\n" +
                "    'Test/A-v1': EnvSpec('Test/A-v1', ep),\n" +
                "    'Example/CartPole-v4': EnvSpec('Example/CartPole-v4', ep),\n" +
                "}\n";
            const string ep = "gymnasium.envs.classic_control.cartpole:CartPoleEnv";
            var dotnet = new EnvSpecRegistry {
                ["CartPole-v0"] = new EnvSpec("CartPole-v0", ep),
                ["CartPole-v1"] = new EnvSpec("CartPole-v1", "gymnasium.envs.classic_control:CartPoleEnv"),
                ["CartPole-v2"] = new EnvSpec("CartPole-v2", "gymnasium.cartpole:CartPoleEnv"),
                ["Single-v0"] = new EnvSpec("Single-v0", "single:Env"),
                ["Test/CartPole-v0"] = new EnvSpec("Test/CartPole-v0", ep),
                ["Test/LongerName-v1"] = new EnvSpec("Test/LongerName-v1", ep),
                ["Test/A-v1"] = new EnvSpec("Test/A-v1", ep),
                ["Example/CartPole-v4"] = new EnvSpec("Example/CartPole-v4", ep),
            };

            foreach ((long columns, string[]? excluded) in new (long, string[]?)[] { (3, null), (1, null), (2, new[] { "Test" }), (5, new[] { "classic_control", "Example" }) }) {
                ByteContract.AssertSame($"pprint_registry(num_cols={columns}, exclude_namespaces={PyBuiltins.Repr(excluded?.ToList())})", Outcome.CapturePython(() => {
                    using PyObject registry = Python.evalStmts(setup, "registry");
                    return Python.gymnasium.envs.registration.pprint_registry(registry, columns, excluded);
                }), Outcome.Returned(ValueFacts.FromText(Gymnasium.PprintRegistry(dotnet, (int)columns, excluded, disablePrint: true)!)));
            }
        }

        /// <summary>
        ///     <c>check_obs</c>'s warnings (type, dtype, membership) and assertions (tuple length, dict keys), captured as
        ///     <c>catch_warnings(record=True)</c> records them, over observations and spaces built alike on both sides.
        /// </summary>
        [TestMethod]
        public void CheckObs_MatchesGymnasium() {
            var cases = new (string PythonSetup, Func<(object? Obs, Space Space)> DotNet)[] {
                ("obs, space = 3, spaces.Discrete(2)", () => (3L, new Discrete(2))),
                ("obs, space = np.uint8(0), spaces.Discrete(1)", () => (NDArray.Scalar((byte)0), new Discrete(1))),
                ("obs, space = np.int64(1), spaces.Discrete(3)", () => (NDArray.Scalar(1L), new Discrete(3))),
                ("obs, space = [0, 1], spaces.Tuple([spaces.Discrete(1), spaces.Discrete(2)])", () => (new List<object?> { 0L, 1L }, new Tuple(new Discrete(1), new Discrete(2)))),
                ("obs, space = (1, 2, 3), spaces.Tuple([spaces.Discrete(1), spaces.Discrete(2)])", () => (new object?[] { 1L, 2L, 3L }, new Tuple(new Discrete(1), new Discrete(2)))),
                ("obs, space = {'a': 1, 'c': 2}, spaces.Dict(a=spaces.Discrete(1), b=spaces.Discrete(2))",
                    () => (new Info { ["a"] = 1L, ["c"] = 2L }, new Dict(new Dictionary<string, Space> { ["a"] = new Discrete(1), ["b"] = new Discrete(2) }))),
                ("obs, space = np.array([0.5], dtype=np.float64), spaces.Box(0, 1, (1,), np.float32)",
                    () => (np.array(new[] { 0.5 }), new Box(0, 1, new Shape(1L), dtype: np.float32))),
                ("obs, space = 0.5, spaces.Box(0, 1, (2,), np.float32)", () => (0.5, new Box(0, 1, new Shape(2L), dtype: np.float32))),
                ("obs, space = 0.5, spaces.Box(0, 1, (), np.float32)", () => (0.5, new Box(0, 1, Shape.Scalar, dtype: np.float32))),
                ("obs, space = np.array([1, 0], dtype=np.int8), spaces.MultiBinary(2)", () => (np.array(new sbyte[] { 1, 0 }), new MultiBinary(new long[] { 2 }))),
                ("obs, space = [1, 0], spaces.MultiBinary(2)", () => (new List<object?> { 1L, 0L }, new MultiBinary(new long[] { 2 }))),
            };

            foreach ((string pythonSetup, Func<(object? Obs, Space Space)> dotnet) in cases) {
                string program =
                    "import warnings\n" +
                    "import numpy as np\n" +
                    "from gymnasium import spaces\n" +
                    "from gymnasium.utils.passive_env_checker import check_obs\n" +
                    pythonSetup + "\n" +
                    "with warnings.catch_warnings(record=True) as caught:\n" +
                    "    warnings.simplefilter('always')\n" +
                    "    try:\n" +
                    "        check_obs(obs, space, 'testing')\n" +
                    "        error = None\n" +
                    "    except AssertionError as e:\n" +
                    "        error = str(e)\n" +
                    "result = (error, [str(w.message) for w in caught])\n";
                Outcome pythonOutcome = Outcome.CapturePython(() => Python.evalStmts(program, "result"));
                Outcome dotnetOutcome = Outcome.Capture(() => {
                    (object? obs, Space space) = dotnet();
                    using (space) {
                        using WarningCatcher caught = Warnings.Catch();
                        string? error = null;
                        try {
                            PassiveChecker.CheckObs(obs, space, "testing");
                        } catch (AssertionError e) {
                            error = e.Message;
                        }

                        (obs as IDisposable)?.Dispose();
                        return (error, caught.Records.Select(record => (object?)record.Message).ToList());
                    }
                });
                ByteContract.AssertSame($"check_obs: {pythonSetup}", pythonOutcome, dotnetOutcome);
            }
        }
    }
}
