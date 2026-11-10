using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Gym.Environments;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Observations;
using Gym.Spaces;
using Gym.Tests.Parity.Bridge;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Session;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Python.Runtime;
using Tuple = Gym.Spaces.Tuple;

namespace Gym.Tests.Parity.Tests.Bridge {
    /// <summary>
    ///     Gymnasium's own tools on Gym.NET's envs, through the reverse bridge (<see cref="ReverseBridge"/>, completion plan
    ///     P3.5): <c>check_env</c>, <c>check_environments_match</c> and <c>make</c>'s stack run unchanged on a Gym.NET env, and
    ///     each verdict is held to the verdict the same tool gives on Gymnasium's own env. Gym.NET's ports of the tools
    ///     (<c>EnvChecker.CheckEnv</c>) are held to it too, so both checkers agree on CartPole (the phase's exit).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The bridge itself is proven to have teeth (plan §11): a planted dynamics defect, a planted buffer reuse and a
    ///     planted unseeded reset each reach Gymnasium's tools as the failure a Gymnasium env with that defect produces, with
    ///     the same message Gym.NET's own checker gives.
    ///     </para>
    ///     <para>
    ///     Human mode renders headless on both sides: SDL's dummy driver for pygame (the session's environment), and
    ///     <see cref="NullEnvViewer"/> as <see cref="Gymnasium.HumanViewerFactory"/> for Gym.NET while a cell needs it.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/env_checker.py">gymnasium/utils/env_checker.py</see>, <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/env_match.py">gymnasium/utils/env_match.py</see>.</para>
    /// </remarks>
    [TestClass]
    public sealed class GymnasiumToolsLiveParityTests : ParityTestBase {
        /// <summary>
        ///     Gymnasium's <c>check_env</c> on <c>env</c>, its warnings recorded as <c>catch_warnings(record=True)</c> records
        ///     them: <c>(None, warnings)</c> when the env passes, <c>(message, warnings)</c> when a check asserts.
        /// </summary>
        private const string CheckEnvStatements =
            "import warnings\n" +
            "from gymnasium.utils.env_checker import check_env\n" +
            "with warnings.catch_warnings(record=True) as caught:\n" +
            "    warnings.simplefilter('always')\n" +
            "    try:\n" +
            "        check_env(env)\n" +
            "        error = None\n" +
            "    except AssertionError as e:\n" +
            "        error = str(e)\n" +
            "result = (error, [str(w.message) for w in caught])\n";

        /// <summary>
        ///     Steps <c>env</c> with action 1 from <c>reset(seed=0)</c> until the pole falls, then twice more, recording the
        ///     warnings of the steps past termination (CartPole warns once).
        /// </summary>
        private const string StepPastTerminationStatements =
            "import warnings\n" +
            "env.reset(seed=0)\n" +
            "terminated = False\n" +
            "while not terminated:\n" +
            "    _, _, terminated, _, _ = env.step(1)\n" +
            "with warnings.catch_warnings(record=True) as caught:\n" +
            "    warnings.simplefilter('always')\n" +
            "    env.step(1)\n" +
            "    env.step(1)\n" +
            "result = [(w.category.__name__, str(w.message)) for w in caught]\n";

        /// <summary>Gymnasium's spaces of <see cref="SpaceMirror_EqualsGymnasiumsSpaces"/>, in its order.</summary>
        private const string SpaceCatalogStatements =
            "import numpy as np\n" +
            "from gymnasium import spaces\n" +
            "catalog = [\n" +
            "    spaces.Box(-1.0, 2.0, (3,), np.float64),\n" +
            "    spaces.Box(np.array([0, -5], dtype=np.int32), np.array([10, 5], dtype=np.int32), dtype=np.int32),\n" +
            "    spaces.Discrete(5, start=-2),\n" +
            "    spaces.Discrete(3, dtype=np.int32),\n" +
            "    spaces.Tuple((spaces.Discrete(2), spaces.Box(0, 1, (2,), np.float32))),\n" +
            "    spaces.Dict({'b': spaces.Discrete(3), 'a': spaces.Box(0, 1, (2,), np.float32)}),\n" +
            "]\n";

        /// <summary>
        ///     The Gymnasium-side stack over Gym.NET's CartPole-v1 (<c>gymnasium.make(spec)</c> on the mirrored spec) prints,
        ///     equals and describes itself as Gymnasium's own <c>make("CartPole-v1")</c>: <c>str</c> of the stack, both spaces
        ///     (<c>repr</c> and <c>==</c>), the metadata, and the spec <c>make</c> leaves on the unwrapped env.
        /// </summary>
        [TestMethod]
        public void Shim_MirrorsCartPole_AsGymnasiumsMakeBuildsIt() {
            using var bridge = new ReverseBridge();
            using PyRef python = PyRef.Of(() => Python.gymnasium.make("CartPole-v1"));
            using PyRef shim = PyRef.Of(() => {
                using PyObject spec = bridge.Spec(Gymnasium.Spec("CartPole-v1"));
                return Python.gymnasium.make(spec);
            });

            ByteContract.AssertSame("str(make('CartPole-v1'))", Outcome.CapturePython(() => Python.builtins.str(python.Object)),
                ReverseBridge.AsBridgedSide(Outcome.CapturePython(() => Python.builtins.str(shim.Object))));
            foreach ((string name, Func<PyObject, PyObject> space) in new (string, Func<PyObject, PyObject>)[] {
                         ("observation_space", Python.gymnasium.Env.observation_space), ("action_space", Python.gymnasium.Env.action_space),
                     }) {
                ByteContract.AssertSame($"repr(env.{name})", Outcome.CapturePython(() => {
                    using PyObject value = space(python.Object);
                    return Python.builtins.repr(value);
                }), ReverseBridge.AsBridgedSide(Outcome.CapturePython(() => {
                    using PyObject value = space(shim.Object);
                    return Python.builtins.repr(value);
                })));
                // Gymnasium's own __eq__ across the two: what check_environments_match asserts first.
                ByteContract.AssertSame($"gymnasium env.{name} == bridged env.{name}", Outcome.Capture(() => true), ReverseBridge.AsBridgedSide(Outcome.CapturePython(() => {
                    using PyObject left = space(python.Object);
                    using PyObject right = space(shim.Object);
                    return Python.@operator.eq(left, right);
                })));
            }

            ByteContract.AssertSame("env.metadata", Outcome.CapturePython(() => Python.gymnasium.Env.metadata(python.Object)),
                ReverseBridge.AsBridgedSide(Outcome.CapturePython(() => Python.gymnasium.Env.metadata(shim.Object))));
            ByteContract.AssertSame("env.unwrapped.spec.pprint()", Outcome.CapturePython(() => UnwrappedSpecText(python.Object)),
                ReverseBridge.AsBridgedSide(Outcome.CapturePython(() => UnwrappedSpecText(shim.Object))));
        }

        /// <summary>
        ///     The bridge's space mirror equals Gymnasium's own construction of the same space (<c>==</c> and <c>repr</c>) for
        ///     every space type it mirrors: float and int boxes, discrete spaces with a start and a dtype, a tuple, and a dict
        ///     whose keys Gymnasium sorts.
        /// </summary>
        [TestMethod]
        public void SpaceMirror_EqualsGymnasiumsSpaces() {
            using NDArray low = np.array(new[] { 0, -5 });
            using NDArray high = np.array(new[] { 10, 5 });
            var dotnet = new Space[] {
                new Box(-1.0, 2.0, new Shape(3L), dtype: np.float64),
                new Box(low, high, dtype: np.int32),
                new Discrete(5, start: -2),
                new Discrete(3, dtype: np.int32),
                new Tuple(new Discrete(2), new Box(0, 1, new Shape(2L), dtype: np.float32)),
                new Dict(new Dictionary<string, Space> { ["b"] = new Discrete(3), ["a"] = new Box(0, 1, new Shape(2L), dtype: np.float32) }),
            };
            try {
                for (int i = 0; i < dotnet.Length; i++) {
                    int index = i;
                    ByteContract.AssertSame($"repr(catalog[{index}])", Outcome.CapturePython(() => {
                        using PyObject catalog = Python.evalStmts(SpaceCatalogStatements, "catalog");
                        using PyObject expected = catalog[index];
                        return Python.builtins.repr(expected);
                    }), ReverseBridge.AsBridgedSide(Outcome.CapturePython(() => {
                        using PyObject mirror = ReverseBridge.Space(dotnet[index]);
                        return Python.builtins.repr(mirror);
                    })));
                    ByteContract.AssertSame($"catalog[{index}] == mirror", Outcome.Capture(() => true), ReverseBridge.AsBridgedSide(Outcome.CapturePython(() => {
                        using PyObject catalog = Python.evalStmts(SpaceCatalogStatements, "catalog");
                        using PyObject expected = catalog[index];
                        using PyObject mirror = ReverseBridge.Space(dotnet[index]);
                        return Python.@operator.eq(expected, mirror);
                    })));
                }
            } finally {
                foreach (Space space in dotnet)
                    space.Dispose();
            }
        }

        /// <summary>
        ///     Gymnasium's <c>check_env</c> passes Gym.NET's CartPole exactly as it passes its own (no error, the same
        ///     warnings), for <c>make("CartPole-v1").unwrapped</c> (spec set: every declared render mode is made and checked, and
        ///     a made env is closed twice), the bare env (no spec) and the bare <c>rgb_array</c> env; and Gym.NET's own
        ///     <c>EnvChecker.CheckEnv</c> gives the same verdict on the same envs.
        /// </summary>
        [TestMethod]
        public void CheckEnv_PassesGymNetsCartPole_AsGymnasiumsOwn() {
            IEnvironmentViewerFactoryDelegate? previous = Gymnasium.HumanViewerFactory;
            // check_env makes the env in human mode too; Gym.NET's renders into a headless viewer, as pygame does under SDL's dummy driver.
            Gymnasium.HumanViewerFactory = NullEnvViewer.Factory;
            try {
                using var bridge = new ReverseBridge();
                using (PyRef python = PyRef.Of(() => MadeUnwrapped(Python.gymnasium.make("CartPole-v1"))))
                using (PyRef shim = PyRef.Of(() => {
                           using PyObject spec = bridge.Spec(Gymnasium.Spec("CartPole-v1"));
                           return MadeUnwrapped(Python.gymnasium.make(spec));
                       }))
                using (IEnv dotnet = Gymnasium.Make("CartPole-v1")) {
                    const string label = "check_env(make('CartPole-v1').unwrapped)";
                    Outcome expected = ExpectPassed(label, GymnasiumCheckEnv(python.Object));
                    ByteContract.AssertSame($"{label}: Gymnasium's tool on Gym.NET's env", expected, ReverseBridge.AsBridgedSide(GymnasiumCheckEnv(shim.Object)));
                    ByteContract.AssertSame($"{label}: Gym.NET's tool", expected, GymNetCheckEnv(dotnet.Unwrapped));
                    // The made shim, then spec.make() once per declared render mode and once for the double-close check: every env
                    // check_env built went through the bridge, so those checks ran on Gym.NET envs too.
                    Assert.AreEqual(4, bridge.Envs.Count, $"{label}: the Gym.NET envs Gymnasium's check_env built through the bridge.");
                }

                foreach (string? renderMode in new[] { null, "rgb_array" }) {
                    var kwargs = new Dictionary<string, object?>();
                    if (renderMode != null)
                        kwargs["render_mode"] = renderMode;
                    string label = $"check_env(CartPoleEnv(render_mode={PyBuiltins.Repr(renderMode)}))";
                    using PyRef python = PyRef.Of(() => Python.gymnasium.envs.classic_control.CartPoleEnv(renderMode));
                    using PyRef shim = PyRef.Of(() => bridge.Shim(typeof(CartPoleEnv), kwargs));
                    using var dotnet = new CartPoleEnv(renderMode: renderMode);
                    Outcome expected = ExpectPassed(label, GymnasiumCheckEnv(python.Object));
                    ByteContract.AssertSame($"{label}: Gymnasium's tool on Gym.NET's env", expected, ReverseBridge.AsBridgedSide(GymnasiumCheckEnv(shim.Object)));
                    ByteContract.AssertSame($"{label}: Gym.NET's tool", expected, GymNetCheckEnv(dotnet));
                }
            } finally {
                Gymnasium.HumanViewerFactory = previous;
            }
        }

        /// <summary>
        ///     Gymnasium's <c>check_environments_match</c> finds Gym.NET's CartPole-v1 equal to its own, frames included, over
        ///     three seeds of 200 sampled steps (resets after every fall), with either env sampling the actions: the tool's
        ///     verdict on (Gymnasium, Gym.NET) is its verdict on (Gymnasium, Gymnasium).
        /// </summary>
        [TestMethod]
        public void CheckEnvironmentsMatch_GymNetsCartPole_AsGymnasiumsOwn() {
            SkipOffReferencePlatform("CartPole's cos/sin");
            using var bridge = new ReverseBridge();
            foreach (long seed in new long[] { 0, 1, 123 }) {
                using PyRef python = PyRef.Of(() => Python.gymnasium.make("CartPole-v1", render_mode: "rgb_array"));
                using PyRef twin = PyRef.Of(() => Python.gymnasium.make("CartPole-v1", render_mode: "rgb_array"));
                using PyRef shim = PyRef.Of(() => {
                    using PyObject spec = bridge.Spec(Gymnasium.Spec("CartPole-v1"));
                    return Python.gymnasium.make(spec, render_mode: "rgb_array");
                });
                Outcome expected = Outcome.CapturePython(() => Python.gymnasium.utils.env_match.check_environments_match(python.Object, twin.Object, 200, seed));
                // The reference must pass, or two equal failures would pass the comparison.
                expected.Expect($"check_environments_match(gymnasium, gymnasium, 200, seed={seed})");
                ByteContract.AssertSame($"check_environments_match(gymnasium, gym.net, 200, seed={seed})", expected,
                    ReverseBridge.AsBridgedSide(Outcome.CapturePython(() => Python.gymnasium.utils.env_match.check_environments_match(python.Object, shim.Object, 200, seed))));
                // Gymnasium samples from env_a's action space, so the other order samples from the bridge's mirror.
                ByteContract.AssertSame($"check_environments_match(gym.net, gymnasium, 200, seed={seed})", expected,
                    ReverseBridge.AsBridgedSide(Outcome.CapturePython(() => Python.gymnasium.utils.env_match.check_environments_match(shim.Object, python.Object, 200, seed))));
            }
        }

        /// <summary>
        ///     Gymnasium's <c>check_env</c> on <c>make(id).unwrapped</c> for the classic-control envs of P4 and CartPole-v0 (whose
        ///     v1 twin has its own cells above): Gym.NET's env built through the bridge passes exactly as Gymnasium's own (no
        ///     error, the same warnings, such as Pendulum's advice on its unnormalised action space), every declared render
        ///     mode made and checked; and Gym.NET's own <c>EnvChecker.CheckEnv</c> gives the same verdict.
        /// </summary>
        /// <param name="id">The id.</param>
        [TestMethod]
        [DataRow("CartPole-v0")]
        [DataRow("MountainCar-v0")]
        [DataRow("MountainCarContinuous-v0")]
        [DataRow("Pendulum-v1")]
        [DataRow("Acrobot-v1")]
        public void CheckEnv_PassesGymNetsClassicEnvs_AsGymnasiumsOwn(string id) {
            SkipOffReferencePlatform("the classic envs' cos/sin");
            IEnvironmentViewerFactoryDelegate? previous = Gymnasium.HumanViewerFactory;
            // check_env makes the env in human mode too; Gym.NET's renders into a headless viewer, as pygame does under SDL's dummy driver.
            Gymnasium.HumanViewerFactory = NullEnvViewer.Factory;
            try {
                using var bridge = new ReverseBridge();
                using PyRef python = PyRef.Of(() => MadeUnwrapped(Python.gymnasium.make(id)));
                using PyRef shim = PyRef.Of(() => {
                    using PyObject spec = bridge.Spec(Gymnasium.Spec(id));
                    return MadeUnwrapped(Python.gymnasium.make(spec));
                });
                using IEnv dotnet = Gymnasium.Make(id);
                string label = $"check_env(make({PyBuiltins.Repr(id)}).unwrapped)";
                Outcome expected = ExpectPassed(label, GymnasiumCheckEnv(python.Object));
                ByteContract.AssertSame($"{label}: Gymnasium's tool on Gym.NET's env", expected, ReverseBridge.AsBridgedSide(GymnasiumCheckEnv(shim.Object)));
                ByteContract.AssertSame($"{label}: Gym.NET's tool", expected, GymNetCheckEnv(dotnet.Unwrapped));
            } finally {
                Gymnasium.HumanViewerFactory = previous;
            }
        }

        /// <summary>
        ///     Gymnasium's <c>check_environments_match</c> finds Gym.NET's classic-control envs of P4 and CartPole-v0 equal to its
        ///     own over two seeds of 200 sampled steps, in both orders, every env's <c>rgb_array</c> frames included (the tool
        ///     compares each step's render). The tool compares reward types, so the bridge returns Pendulum's as the
        ///     <c>np.float64</c> Gymnasium's does (<see cref="BridgeReward"/>).
        /// </summary>
        /// <param name="id">The id.</param>
        [TestMethod]
        [DataRow("CartPole-v0")]
        [DataRow("MountainCar-v0")]
        [DataRow("MountainCarContinuous-v0")]
        [DataRow("Pendulum-v1")]
        [DataRow("Acrobot-v1")]
        public void CheckEnvironmentsMatch_GymNetsClassicEnvs_AsGymnasiumsOwn(string id) {
            SkipOffReferencePlatform("the classic envs' cos/sin");
            const string renderMode = "rgb_array";
            // Pendulum's reward is an np.float64 (-costs of NumPy scalars); the others' are Python floats.
            BridgeReward reward = id == "Pendulum-v1" ? BridgeReward.NumpyFloat64 : BridgeReward.Float;
            using var bridge = new ReverseBridge();
            foreach (long seed in new long[] { 0, 123 }) {
                using PyRef python = PyRef.Of(() => Python.gymnasium.make(id, render_mode: renderMode));
                using PyRef twin = PyRef.Of(() => Python.gymnasium.make(id, render_mode: renderMode));
                using PyRef shim = PyRef.Of(() => {
                    using PyObject spec = bridge.Spec(Gymnasium.Spec(id), reward: reward);
                    return Python.gymnasium.make(spec, render_mode: renderMode);
                });
                Outcome expected = Outcome.CapturePython(() => Python.gymnasium.utils.env_match.check_environments_match(python.Object, twin.Object, 200, seed));
                // The reference must pass, or two equal failures would pass the comparison.
                expected.Expect($"check_environments_match({id}: gymnasium, gymnasium, 200, seed={seed})");
                ByteContract.AssertSame($"check_environments_match({id}: gymnasium, gym.net, 200, seed={seed})", expected,
                    ReverseBridge.AsBridgedSide(Outcome.CapturePython(() => Python.gymnasium.utils.env_match.check_environments_match(python.Object, shim.Object, 200, seed))));
                // Gymnasium samples from env_a's action space, so the other order samples from the bridge's mirror.
                ByteContract.AssertSame($"check_environments_match({id}: gym.net, gymnasium, 200, seed={seed})", expected,
                    ReverseBridge.AsBridgedSide(Outcome.CapturePython(() => Python.gymnasium.utils.env_match.check_environments_match(shim.Object, python.Object, 200, seed))));
            }
        }

        /// <summary>
        ///     Gymnasium's <c>check_env</c> on <c>make(id).unwrapped</c> for the toy_text envs of P5: Gym.NET's env built through
        ///     the bridge passes exactly as Gymnasium's own (no error, the same warnings), every declared render mode
        ///     (<c>ansi</c> text included) made and checked; and Gym.NET's own <c>EnvChecker.CheckEnv</c> gives the same verdict.
        ///     Their observations cross as the Python ints Gymnasium's envs return (<see cref="BridgeObservation.PythonInt"/>).
        /// </summary>
        /// <param name="id">The id.</param>
        [TestMethod]
        [DataRow("Blackjack-v1")]
        [DataRow("FrozenLake-v1")]
        [DataRow("FrozenLake8x8-v1")]
        [DataRow("CliffWalking-v1")]
        [DataRow("CliffWalkingSlippery-v1")]
        [DataRow("Taxi-v4")]
        public void CheckEnv_PassesGymNetsToyTextEnvs_AsGymnasiumsOwn(string id) {
            IEnvironmentViewerFactoryDelegate? previous = Gymnasium.HumanViewerFactory;
            // check_env makes the env in human mode too; Gym.NET's renders into a headless viewer, as pygame does under SDL's dummy driver.
            Gymnasium.HumanViewerFactory = NullEnvViewer.Factory;
            try {
                using var bridge = new ReverseBridge();
                using PyRef python = PyRef.Of(() => MadeUnwrapped(Python.gymnasium.make(id)));
                using PyRef shim = PyRef.Of(() => {
                    using PyObject spec = bridge.Spec(Gymnasium.Spec(id), reward: ToyTextReward(id), observation: BridgeObservation.PythonInt);
                    return MadeUnwrapped(Python.gymnasium.make(spec));
                });
                using IEnv dotnet = Gymnasium.Make(id);
                string label = $"check_env(make({PyBuiltins.Repr(id)}).unwrapped)";
                Outcome expected = ExpectPassed(label, GymnasiumCheckEnv(python.Object));
                ByteContract.AssertSame($"{label}: Gymnasium's tool on Gym.NET's env", expected, ReverseBridge.AsBridgedSide(GymnasiumCheckEnv(shim.Object)));
                ByteContract.AssertSame($"{label}: Gym.NET's tool", expected, GymNetCheckEnv(dotnet.Unwrapped));
            } finally {
                Gymnasium.HumanViewerFactory = previous;
            }
        }

        /// <summary>
        ///     Gymnasium's <c>check_environments_match</c> finds Gym.NET's toy_text envs of P5 equal to its own over two seeds of
        ///     200 sampled steps, in both orders, <c>rgb_array</c> frames included. The tool compares types, so the bridge returns
        ///     the observations as Python ints and the rewards as the type Gymnasium's env returns (an <c>int</c> from the tables,
        ///     Blackjack's <c>float</c>).
        /// </summary>
        /// <param name="id">The id.</param>
        [TestMethod]
        [DataRow("Blackjack-v1")]
        [DataRow("FrozenLake-v1")]
        [DataRow("FrozenLake8x8-v1")]
        [DataRow("CliffWalking-v1")]
        [DataRow("CliffWalkingSlippery-v1")]
        [DataRow("Taxi-v4")]
        public void CheckEnvironmentsMatch_GymNetsToyTextEnvs_AsGymnasiumsOwn(string id) {
            using var bridge = new ReverseBridge();
            foreach (long seed in new long[] { 0, 123 }) {
                using PyRef python = PyRef.Of(() => Python.gymnasium.make(id, render_mode: "rgb_array"));
                using PyRef twin = PyRef.Of(() => Python.gymnasium.make(id, render_mode: "rgb_array"));
                using PyRef shim = PyRef.Of(() => {
                    using PyObject spec = bridge.Spec(Gymnasium.Spec(id), reward: ToyTextReward(id), observation: BridgeObservation.PythonInt);
                    return Python.gymnasium.make(spec, render_mode: "rgb_array");
                });
                Outcome expected = Outcome.CapturePython(() => Python.gymnasium.utils.env_match.check_environments_match(python.Object, twin.Object, 200, seed));
                // The reference must pass, or two equal failures would pass the comparison.
                expected.Expect($"check_environments_match({id}: gymnasium, gymnasium, 200, seed={seed})");
                ByteContract.AssertSame($"check_environments_match({id}: gymnasium, gym.net, 200, seed={seed})", expected,
                    ReverseBridge.AsBridgedSide(Outcome.CapturePython(() => Python.gymnasium.utils.env_match.check_environments_match(python.Object, shim.Object, 200, seed))));
                // Gymnasium samples from env_a's action space, so the other order samples from the bridge's mirror.
                ByteContract.AssertSame($"check_environments_match({id}: gym.net, gymnasium, 200, seed={seed})", expected,
                    ReverseBridge.AsBridgedSide(Outcome.CapturePython(() => Python.gymnasium.utils.env_match.check_environments_match(shim.Object, python.Object, 200, seed))));
            }
        }

        /// <summary>
        ///     The Python type of a toy_text env's reward in Gymnasium: Blackjack's <c>float</c> (<c>-1.0</c>, <c>cmp</c>'s
        ///     result), the tables' <c>int</c> (FrozenLake's default schedule <c>(1, 0, 0)</c>, CliffWalking's and Taxi's).
        /// </summary>
        /// <param name="id">The id.</param>
        /// <returns>The bridge's reward type for it.</returns>
        private static BridgeReward ToyTextReward(string id) => id == "Blackjack-v1" ? BridgeReward.Float : BridgeReward.Int;

        /// <summary>
        ///     What a Gym.NET env raises reaches Gymnasium as the Python type it stands for: <c>make</c> with a keyword the
        ///     constructor does not take wraps the creator's <c>TypeError</c> in the same text for both envs.
        /// </summary>
        [TestMethod]
        public void Make_ConstructorError_ReachesGymnasiumAsItsOwn() {
            using var bridge = new ReverseBridge();
            var kwargs = new Dictionary<string, object?> { ["bogus"] = 1L };
            Outcome expected = Outcome.CapturePython(() => Python.gymnasium.make("CartPole-v1", kwargs: kwargs));
            Assert.AreEqual("TypeError", expected.Raised?.Type, "Gymnasium's own make must refuse the keyword, or the comparison proves nothing.");
            ByteContract.AssertSame("make('CartPole-v1', bogus=1)", expected,
                ReverseBridge.AsBridgedSide(Outcome.CapturePython(() => {
                    using PyObject spec = bridge.Spec(Gymnasium.Spec("CartPole-v1"));
                    return Python.gymnasium.make(spec, kwargs: kwargs);
                })));
        }

        /// <summary>
        ///     What a Gym.NET env warns reaches Python's <c>warnings</c> as a Gymnasium env's warning: stepping past
        ///     termination records the same category and text through the bridge as on Gymnasium's CartPole.
        /// </summary>
        [TestMethod]
        public void Warnings_ReachPythonAsGymnasiumsOwn() {
            using var bridge = new ReverseBridge();
            using PyRef python = PyRef.Of(() => Python.gymnasium.envs.classic_control.CartPoleEnv());
            using PyRef shim = PyRef.Of(() => bridge.Shim(typeof(CartPoleEnv)));
            Outcome expected = Outcome.CapturePython(() => Python.evalStmts(StepPastTerminationStatements, "result", ("env", python.Object)));
            // Gymnasium's CartPole warns on the first step past termination only: one record, so the comparison is not of two empty lists.
            Assert.AreEqual(1, expected.Expect("Gymnasium's warnings of steps past termination").Items.Count);
            ByteContract.AssertSame("warnings of steps past termination", expected,
                ReverseBridge.AsBridgedSide(Outcome.CapturePython(() => Python.evalStmts(StepPastTerminationStatements, "result", ("env", shim.Object)))));
        }

        /// <summary>
        ///     Negative control (plan §11): a CartPole with a planted dynamics defect (τ = 0.03 instead of 0.02) fails
        ///     Gymnasium's <c>check_environments_match</c> at the first step, so a passing cell above is not the bridge hiding
        ///     differences.
        /// </summary>
        [TestMethod]
        public void PlantedDynamicsDefect_FailsGymnasiumsEnvMatch() {
            using var bridge = new ReverseBridge();
            using PyRef python = PyRef.Of(() => Python.gymnasium.make("CartPole-v1"));
            using PyRef shim = PyRef.Of(() => {
                using PyObject spec = bridge.Spec(Gymnasium.Spec("CartPole-v1"), kwargs => {
                    var env = (CartPoleEnv)PyKwargs.Construct(typeof(CartPoleEnv), kwargs);
                    env.Tau = 0.03;
                    return env;
                });
                return Python.gymnasium.make(spec);
            });

            Outcome outcome = Outcome.CapturePython(() => Python.gymnasium.utils.env_match.check_environments_match(python.Object, shim.Object, 50, 0));
            Assert.IsNotNull(outcome.Raised, "The planted defect went unnoticed.");
            Assert.AreEqual("AssertionError", outcome.Raised.Type);
            StringAssert.StartsWith(outcome.Raised.Message, "stepping observation is not equivalent in step = 0");
        }

        /// <summary>
        ///     Negative controls (plan §11) for <c>check_env</c>: an env whose observations are views of one reused buffer, and
        ///     an env whose reset ignores its seed, fail Gymnasium's checks through the bridge with Gymnasium's messages (the
        ///     zero-copy crossing keeps the shared memory visible, the generator mirror keeps the missing generator visible), and
        ///     Gym.NET's own checker fails them with the same messages.
        /// </summary>
        [TestMethod]
        public void PlantedEnvDefects_FailGymnasiumsCheckEnv_AsGymNetsOwn() {
            foreach ((bool reuseObservation, bool seedOnReset, string expectedStart) in new[] {
                         (true, true, "The observations returned by `env.reset() (call 1)` and `env.step() (call 2)` share an object"),
                         (false, false, "Expects the random number generator to have been generated given a seed was passed to reset."),
                     }) {
                using var bridge = new ReverseBridge();
                using PyRef shim = PyRef.Of(() => bridge.Shim(typeof(PlantedDefectEnv), create: _ => new PlantedDefectEnv(reuseObservation, seedOnReset)));
                using var dotnet = new PlantedDefectEnv(reuseObservation, seedOnReset);
                Outcome gymnasium = GymnasiumCheckEnv(shim.Object);
                string label = $"check_env(PlantedDefectEnv(reuse_observation={reuseObservation}, seed_on_reset={seedOnReset}))";
                ByteContract.AssertSame(label, gymnasium, GymNetCheckEnv(dotnet));
                // The verdict's first item is the assertion text, an empty description when check_env passed (None).
                string error = gymnasium.Expect(label).Items[0].Text;
                StringAssert.StartsWith(error, expectedStart, $"{label}: the planted defect went unnoticed.");
            }
        }

        /// <summary>
        ///     Requires Gymnasium's own <c>check_env</c> verdict to be a pass (it returned, and its error item is <c>None</c>),
        ///     then prints it: two equal failures would otherwise pass every comparison against it.
        /// </summary>
        /// <param name="label">The checked call, for the failure message.</param>
        /// <param name="verdict">Gymnasium's verdict on its own env.</param>
        /// <returns><paramref name="verdict"/>.</returns>
        /// <exception cref="AssertFailedException">Thrown when Gymnasium's check raised or asserted on its own env.</exception>
        private static Outcome ExpectPassed(string label, Outcome verdict) {
            ValueFacts facts = verdict.Expect(label);
            Assert.AreEqual(FactKind.None, facts.Items[0].Kind, $"{label} failed on Gymnasium's own env: {facts.Render()}");
            CurrentContext?.WriteLine($"{label}: {facts.Render()}");
            return verdict;
        }

        /// <summary>Gymnasium's <c>check_env</c> on a Python env, described: <c>(error or None, [warning texts])</c>.</summary>
        /// <param name="env">The env; borrowed.</param>
        /// <returns>The outcome; an exception other than <c>AssertionError</c> is the outcome's raised error.</returns>
        private static Outcome GymnasiumCheckEnv(PyObject env) => Outcome.CapturePython(() => Python.evalStmts(CheckEnvStatements, "result", ("env", env)));

        /// <summary>Gym.NET's <c>EnvChecker.CheckEnv</c> on a Gym.NET env, described as <see cref="GymnasiumCheckEnv"/> describes Gymnasium's.</summary>
        /// <param name="env">The env; stays owned by the caller.</param>
        /// <returns>The outcome; an exception other than <see cref="AssertionError"/> is the outcome's raised error.</returns>
        private static Outcome GymNetCheckEnv(IEnv env) => Outcome.Capture(() => {
            using WarningCatcher caught = Warnings.Catch();
            string? error = null;
            try {
                EnvChecker.CheckEnv(env);
            } catch (AssertionError e) {
                error = e.Message;
            }

            return (error, caught.Records.Select(record => (object?)record.Message).ToList());
        });

        /// <summary><c>made.unwrapped</c>, releasing the made stack (the unwrapped env stays alive through the returned reference).</summary>
        /// <param name="made">A new reference to a made env; disposed here.</param>
        /// <returns>A new owned reference to the unwrapped env. Must run under the GIL.</returns>
        /// <exception cref="PythonException">Thrown when the attribute lookup fails.</exception>
        private static PyObject MadeUnwrapped(PyObject made) {
            using (made)
                return Python.gymnasium.Env.unwrapped(made);
        }

        /// <summary><c>env.unwrapped.spec.pprint()</c>'s text (without entry points, which differ by design: a class against a module path).</summary>
        /// <param name="env">The made env; borrowed.</param>
        /// <returns>A new owned reference to the text. Must run under the GIL.</returns>
        /// <exception cref="PythonException">Thrown when printing fails.</exception>
        private static PyObject UnwrappedSpecText(PyObject env) {
            using PyObject unwrapped = Python.gymnasium.Env.unwrapped(env);
            using PyObject spec = Python.gymnasium.Env.spec(unwrapped);
            return Python.gymnasium.envs.registration.EnvSpec.pprint(spec);
        }

        /// <summary>
        ///     A valid little env with one planted defect at a time, for the negative controls: its observations can be views
        ///     of one reused buffer (Gymnasium's data-reuse check must see the sharing), and its reset can ignore the seed
        ///     (Gymnasium's seeding check must see no generator). Otherwise <c>Box(0, 1, (2,), float32)</c> observations,
        ///     <c>Discrete(2)</c> actions, and deterministic: action <c>a</c> adds 0.125 to entry <c>a</c>.
        /// </summary>
        private sealed class PlantedDefectEnv : Env<NDArray, NDArray> {
            /// <summary>The state; returned as a view when <see cref="_reuseObservation"/>, else copied.</summary>
            private readonly NDArray _buffer = np.zeros(new Shape(2L), np.float32);

            /// <summary>The first planted defect: every observation is a view of <see cref="_buffer"/>.</summary>
            private readonly bool _reuseObservation;

            /// <summary>Whether reset seeds the generator; <see langword="false"/> is the second planted defect.</summary>
            private readonly bool _seedOnReset;

            /// <summary>An env with the chosen defects.</summary>
            /// <param name="reuseObservation">Return views of one buffer (the data-reuse defect).</param>
            /// <param name="seedOnReset">Seed on reset; <see langword="false"/> plants the seeding defect.</param>
            public PlantedDefectEnv(bool reuseObservation, bool seedOnReset) {
                _reuseObservation = reuseObservation;
                _seedOnReset = seedOnReset;
                ObservationSpace = new Box(0, 1, new Shape(2L), dtype: np.float32);
                ActionSpace = new Discrete(2);
            }

            /// <summary>Zeroes the state, seeding first unless the seeding defect is planted.</summary>
            /// <param name="seed">The seed.</param>
            /// <param name="options">Ignored.</param>
            /// <returns>The zero observation and an empty info.</returns>
            /// <exception cref="Gym.Exceptions.Error">Thrown for a negative seed.</exception>
            public override ResetResult<NDArray> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
                if (_seedOnReset)
                    SeedNpRandom(seed);
                _buffer.SetAtIndex(0f, 0);
                _buffer.SetAtIndex(0f, 1);
                return new ResetResult<NDArray>(Observation(), new Info());
            }

            /// <summary>Adds 0.125 to the entry the action names.</summary>
            /// <param name="action">A 0-d int64 array, 0 or 1.</param>
            /// <returns>The new observation, reward 0, never terminated or truncated.</returns>
            public override StepResult<NDArray> Step(NDArray action) {
                long index = action.GetAtIndex<long>(0);
                _buffer.SetAtIndex(_buffer.GetAtIndex<float>(index) + 0.125f, index);
                return new StepResult<NDArray>(Observation(), 0.0, false, false, new Info());
            }

            /// <summary>Releases the state, then disposes as every env does.</summary>
            /// <param name="disposing"><see langword="true"/> from <see cref="Env{TObs, TAct}.Dispose()"/>.</param>
            protected override void Dispose(bool disposing) {
                if (disposing)
                    _buffer.Dispose();
                base.Dispose(disposing);
            }

            /// <summary>The observation: a view of the one buffer (the planted reuse) or a copy of it.</summary>
            /// <returns>A new array object, owned by the caller.</returns>
            private NDArray Observation() => _reuseObservation ? _buffer[":"] : _buffer.copy();
        }
    }
}
