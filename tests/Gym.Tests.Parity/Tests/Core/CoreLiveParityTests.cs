using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Observations;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Pair;
using Gym.Tests.Parity.Session;
using Gym.Utils;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Python.Runtime;

namespace Gym.Tests.Parity.Tests.Core {
    /// <summary>
    ///     The core API (port plan P1) against Gymnasium's, live: the
    ///     wrapper stack <c>make</c> builds against Gym.NET's through truncation, <c>TimeLimit</c> and <c>OrderEnforcing</c>
    ///     call by call, the attribute protocol (<c>get_wrapper_attr</c>, plain <c>getattr</c>/<c>setattr</c>, read-only
    ///     properties), <c>np_random</c>/<c>np_random_seed</c>, <c>render()</c> without a render mode, <c>EnvSpec</c>'s
    ///     repr and pprint and the id grammar, the step-API conversions, and dict repr.
    /// </summary>
    /// <remarks>
    ///     Gym.NET has no <c>make</c> or <c>PassiveEnvChecker</c> until P3, so its side of a <c>make</c>-built stack is built by
    ///     hand, <c>TimeLimit(OrderEnforcing(CartPoleEnv))</c>, with the spec <c>make</c> gives the unwrapped env; the checker only
    ///     validates, so every compared value is unaffected. Entry points name modules, which differ by design
    ///     (<c>gymnasium.core:Wrapper</c> against <c>Gym.Envs:Wrapper</c>), so spec cells use stacks whose wrappers edit the spec
    ///     rather than append a <c>WrapperSpec</c>, or build <c>WrapperSpec</c>s with the same strings on both sides.
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/core.py">gymnasium/core.py</see>, <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/wrappers/common.py">gymnasium/wrappers/common.py</see>,
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/envs/registration.py">gymnasium/envs/registration.py</see> and <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/utils/step_api_compatibility.py">gymnasium/utils/step_api_compatibility.py</see>.
    /// </remarks>
    [TestClass]
    public sealed class CoreLiveParityTests : ParityTestBase {
        /// <summary>
        ///     Gymnasium's <c>make("CartPole-v1")</c> stack against Gym.NET's hand-built one for a whole episode of the balancing
        ///     policy: every transition identical, and both truncate at step 500, where <c>TimeLimit</c> ends the episode.
        /// </summary>
        /// <param name="seed">The reset seed.</param>
        [TestMethod]
        [DataRow(0L)]
        [DataRow(42L)]
        public void MadeStack_MatchesGymnasiumsMake_ThroughTruncation(long seed) {
            SkipOffReferencePlatform("CartPole's cos/sin");
            using var pair = new GymPair(PyGymEnv.Make("CartPole-v1", viaMake: true), new GymNetEnv(MadeCartPole()), stateAttribute: "state");
            PairRun run = pair.Run(seed, ActionSources.Policy("cartpole_balance"), maxSteps: 600);
            TestContext.WriteLine(run.ToString());
            Assert.AreEqual(500, run.Steps, "Both sides must truncate at max_episode_steps.");
            Assert.IsTrue(run.Ended);
        }

        /// <summary>
        ///     <c>OrderEnforcing</c> call by call: <c>ResetNeeded</c> (type and message) for step and render before a reset,
        ///     <c>has_reset</c>, the transitions and frame after it, <c>str(env)</c>, and <c>disable_render_order_enforcing</c>.
        /// </summary>
        [TestMethod]
        public void OrderEnforcing_MatchesGymnasium() {
            using PyRef python = PyRef.Of(() => {
                using PyObject cartPole = Python.gymnasium.envs.classic_control.CartPoleEnv(render_mode: "rgb_array");
                return Python.gymnasium.wrappers.OrderEnforcing(cartPole);
            });
            var dotnet = new OrderEnforcing<NDArray, long>(new CartPoleEnv(renderMode: "rgb_array"));

            ByteContract.AssertSame("str(env)", Outcome.CapturePython(() => Python.builtins.str(python.Object)), Outcome.Returned(ValueFacts.FromText(dotnet.ToString())));
            ByteContract.AssertSame("has_reset before reset", Outcome.CapturePython(() => Python.gymnasium.wrappers.common.OrderEnforcing.has_reset(python.Object)), Outcome.Returned(ValueFacts.Of(dotnet.HasReset)));
            ByteContract.AssertSame("step(0) before reset", CapturePythonStep(python.Object, 0), Outcome.Capture(() => TupleOf(dotnet.Step(0))));
            ByteContract.AssertSame("render() before reset", Outcome.CapturePython(() => Python.gymnasium.Env.render(python.Object)), Outcome.Capture(() => FrameOf(dotnet.Render())));

            ByteContract.AssertSame("reset(seed=0)", Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, 0)), Outcome.Capture(() => TupleOf(dotnet.Reset(0))));
            ByteContract.AssertSame("has_reset after reset", Outcome.CapturePython(() => Python.gymnasium.wrappers.common.OrderEnforcing.has_reset(python.Object)), Outcome.Returned(ValueFacts.Of(dotnet.HasReset)));
            ByteContract.AssertSame("step(1) after reset", CapturePythonStep(python.Object, 1), Outcome.Capture(() => TupleOf(dotnet.Step(1))));
            ByteContract.AssertSame("render() after reset", Outcome.CapturePython(() => Python.gymnasium.Env.render(python.Object)), Outcome.Capture(() => FrameOf(dotnet.Render())));

            using PyRef lenient = PyRef.Of(() => {
                using PyObject cartPole = Python.gymnasium.envs.classic_control.CartPoleEnv(render_mode: "rgb_array");
                return Python.gymnasium.wrappers.OrderEnforcing(cartPole, disable_render_order_enforcing: true);
            });
            var dotnetLenient = new OrderEnforcing<NDArray, long>(new CartPoleEnv(renderMode: "rgb_array"), disableRenderOrderEnforcing: true);
            ByteContract.AssertSame("render() before reset, render order enforcing disabled", Outcome.CapturePython(() => Python.gymnasium.Env.render(lenient.Object)),
                Outcome.Capture(() => FrameOf(dotnetLenient.Render())));
        }

        /// <summary>
        ///     <c>TimeLimit</c> call by call: its description, the inner env's assertion when stepped before a reset, truncation
        ///     at and after the limit, <c>_elapsed_steps</c> and <c>_max_episode_steps</c>, a missing spec, and the argument
        ///     errors (<c>ValueError</c> for -1 and 0, <c>TypeError</c> for <c>None</c>) with Gymnasium's messages.
        /// </summary>
        [TestMethod]
        public void TimeLimit_MatchesGymnasium() {
            SkipOffReferencePlatform("CartPole's cos/sin");
            using PyRef python = PyRef.Of(() => {
                using PyObject cartPole = Python.gymnasium.envs.classic_control.CartPoleEnv();
                using PyObject limit = PyConvert.ToPython(3L);
                return Python.gymnasium.wrappers.TimeLimit(cartPole, limit);
            });
            var dotnet = new TimeLimit<NDArray, long>(new CartPoleEnv(), 3);

            ByteContract.AssertSame("str(env)", Outcome.CapturePython(() => Python.builtins.str(python.Object)), Outcome.Returned(ValueFacts.FromText(dotnet.ToString())));
            ByteContract.AssertSame("step(0) before reset", CapturePythonStep(python.Object, 0), Outcome.Capture(() => TupleOf(dotnet.Step(0))));
            ByteContract.AssertSame("reset(seed=5)", Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, 5)), Outcome.Capture(() => TupleOf(dotnet.Reset(5))));
            for (int step = 1; step <= 5; step++) {
                ByteContract.AssertSame($"step #{step}", CapturePythonStep(python.Object, step % 2), Outcome.Capture(() => TupleOf(dotnet.Step(step % 2))));
                ByteContract.AssertSame($"_elapsed_steps after step #{step}", Outcome.CapturePython(() => Python.gymnasium.Env.get_wrapper_attr(python.Object, "_elapsed_steps")),
                    Outcome.Capture(() => dotnet.GetWrapperAttr("_elapsed_steps")));
            }

            ByteContract.AssertSame("_max_episode_steps", Outcome.CapturePython(() => Python.gymnasium.Env.get_wrapper_attr(python.Object, "_max_episode_steps")),
                Outcome.Capture(() => dotnet.GetWrapperAttr("_max_episode_steps")));
            ByteContract.AssertSame("spec without a spec beneath", Outcome.CapturePython(() => Python.gymnasium.Env.spec(python.Object)), Outcome.Capture(() => dotnet.Spec));

            foreach (long? limit in new long?[] { -1, 0, null }) {
                string call = $"TimeLimit(env, {(limit?.ToString() ?? "None")})";
                Outcome pythonOutcome = Outcome.CapturePython(() => {
                    using PyObject cartPole = Python.gymnasium.envs.classic_control.CartPoleEnv();
                    using PyObject value = PyConvert.ToPython(limit);
                    return Python.gymnasium.wrappers.TimeLimit(cartPole, value);
                });
                Assert.IsNotNull(pythonOutcome.Raised, $"Gymnasium must refuse {call}.");
                ByteContract.AssertSame(call, pythonOutcome, Outcome.Capture(() => new TimeLimit<NDArray, long>(new CartPoleEnv(), limit)));
            }
        }

        /// <summary>
        ///     The attribute protocol through a stack, as <c>test_get_set_wrapper_attr</c> exercises it and beyond: plain
        ///     <c>getattr</c>/<c>setattr</c> (a wrapper forwards nothing), <c>has_wrapper_attr</c>, <c>get_wrapper_attr</c>,
        ///     <c>set_wrapper_attr</c> (where it lands, and what it returns), run-time attributes, read-only properties
        ///     (<c>np_random_seed</c>, <c>spec</c>) and a wrapper's <c>_np_random</c>, every result or error message compared.
        /// </summary>
        [TestMethod]
        public void WrapperAttributes_MatchGymnasium() {
            using PyRef python = PyRef.Of(() => {
                using PyObject cartPole = Python.gymnasium.envs.classic_control.CartPoleEnv();
                using PyObject ordered = Python.gymnasium.wrappers.OrderEnforcing(cartPole);
                using PyObject limit = PyConvert.ToPython(500L);
                return Python.gymnasium.wrappers.TimeLimit(ordered, limit);
            });
            using PyRef pythonUnwrapped = PyRef.Of(() => Python.gymnasium.Env.unwrapped(python.Object));
            using PyRef pythonInner = PyRef.Of(() => Python.builtins.getattr(python.Object, "env"));
            var cartPole = new CartPoleEnv();
            var dotnet = new TimeLimit<NDArray, long>(new OrderEnforcing<NDArray, long>(cartPole), 500);
            IEnv dotnetInner = dotnet.Env;

            void GetAttr(string label, PyObject pythonTarget, object dotnetTarget, string name) =>
                ByteContract.AssertSame($"getattr({label}, '{name}')", Outcome.CapturePython(() => Python.builtins.getattr(pythonTarget, name)),
                    Outcome.Capture(() => PyAttributes.GetAttr(dotnetTarget, name), disposeResult: false));

            void SetAttr(string name, object? value) =>
                ByteContract.AssertSame($"setattr(env, '{name}', {PyBuiltins.Repr(value)})", Outcome.CapturePython(() => {
                    using PyObject pythonValue = PyConvert.ToPython(value);
                    return Python.builtins.setattr(python.Object, name, pythonValue);
                }), Outcome.Capture(() => {
                    PyAttributes.SetAttr(dotnet, name, value);
                    return null;
                }));

            void Has(string name) =>
                ByteContract.AssertSame($"env.has_wrapper_attr('{name}')", Outcome.CapturePython(() => Python.gymnasium.Env.has_wrapper_attr(python.Object, name)),
                    Outcome.Capture(() => dotnet.HasWrapperAttr(name)));

            void Get(string name) =>
                ByteContract.AssertSame($"env.get_wrapper_attr('{name}')", Outcome.CapturePython(() => Python.gymnasium.Env.get_wrapper_attr(python.Object, name)),
                    Outcome.Capture(() => dotnet.GetWrapperAttr(name), disposeResult: false));

            void Set(string name, object? value, bool force = true) =>
                ByteContract.AssertSame($"env.set_wrapper_attr('{name}', {PyBuiltins.Repr(value)}, force={PyBuiltins.Repr(force)})", Outcome.CapturePython(() => {
                    using PyObject pythonValue = PyConvert.ToPython(value);
                    return Python.gymnasium.Env.set_wrapper_attr(python.Object, name, pythonValue, force);
                }), Outcome.Capture(() => dotnet.SetWrapperAttr(name, value, force)));

            // get_wrapper_attr
            GetAttr("env", python.Object, dotnet, "gravity");
            GetAttr("env.unwrapped", pythonUnwrapped.Object, cartPole, "gravity");
            Has("gravity");
            Get("gravity");
            GetAttr("env", python.Object, dotnet, "unknown_attr");
            Has("unknown_attr");
            Get("unknown_attr");

            // set_wrapper_attr
            Set("gravity", 10.0);
            GetAttr("env", python.Object, dotnet, "gravity");
            GetAttr("env.unwrapped", pythonUnwrapped.Object, cartPole, "gravity");
            Get("gravity");
            SetAttr("gravity", 5.0);
            GetAttr("env", python.Object, dotnet, "gravity");
            Get("gravity");
            ByteContract.AssertSame("env.env.get_wrapper_attr('gravity')", Outcome.CapturePython(() => Python.gymnasium.Env.get_wrapper_attr(pythonInner.Object, "gravity")),
                Outcome.Capture(() => dotnetInner.GetWrapperAttr("gravity")));

            // An intermediate wrapper's private attribute.
            GetAttr("env", python.Object, dotnet, "_disable_render_order_enforcing");
            GetAttr("env.unwrapped", pythonUnwrapped.Object, cartPole, "_disable_render_order_enforcing");
            Has("_disable_render_order_enforcing");
            Get("_disable_render_order_enforcing");
            Set("_disable_render_order_enforcing", true);
            Get("_disable_render_order_enforcing");
            GetAttr("env", python.Object, dotnet, "_disable_render_order_enforcing");

            // Run-time attributes on the outermost wrapper.
            SetAttr("MY_ATTRIBUTE_1", true);
            Get("MY_ATTRIBUTE_1");
            Set("MY_ATTRIBUTE_1", false);
            Get("MY_ATTRIBUTE_1");
            Set("MY_ATTRIBUTE_2", true);
            GetAttr("env", python.Object, dotnet, "MY_ATTRIBUTE_2");
            Set("MY_ATTRIBUTE_3", 1L, force: false);
            Has("MY_ATTRIBUTE_3");

            // Read-only properties and a wrapper's _np_random.
            Set("np_random_seed", 1L);
            SetAttr("spec", null);
            SetAttr("render_mode", "human");
            GetAttr("env", python.Object, dotnet, "_np_random");
            // The generator itself is found beneath (its value, a Generator object, is not comparable across languages).
            Has("_np_random");
            ByteContract.AssertSame("reset(seed=3)", Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, 3)), Outcome.Capture(() => TupleOf(dotnet.Reset(3))));
            Get("np_random_seed");
            Get("_elapsed_steps");
            Get("has_reset");
        }

        /// <summary>
        ///     <c>np_random</c> and <c>np_random_seed</c>: lazy (<c>_np_random</c> is <c>None</c> before use), set by a seeded
        ///     reset and kept by an unseeded one, -1 once a generator is assigned (also through a wrapper), read-only, and a
        ///     negative seed's <c>Error</c>.
        /// </summary>
        [TestMethod]
        public void NpRandomSeed_MatchesGymnasium() {
            using PyRef python = PyRef.Of(() => Python.gymnasium.envs.classic_control.CartPoleEnv());
            var dotnet = new CartPoleEnv();

            ByteContract.AssertSame("_np_random before use", Outcome.CapturePython(() => Python.builtins.getattr(python.Object, "_np_random")),
                Outcome.Capture(() => PyAttributes.GetAttr(dotnet, "_np_random")));
            ByteContract.AssertSame("reset(seed=42)", Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, 42)), Outcome.Capture(() => TupleOf(dotnet.Reset(42))));
            ByteContract.AssertSame("np_random_seed after reset(seed=42)", Outcome.CapturePython(() => Python.gymnasium.Env.np_random_seed(python.Object)), Outcome.Capture(() => dotnet.NpRandomSeed));
            ByteContract.AssertSame("reset()", Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object)), Outcome.Capture(() => TupleOf(dotnet.Reset())));
            ByteContract.AssertSame("np_random_seed after reset()", Outcome.CapturePython(() => Python.gymnasium.Env.np_random_seed(python.Object)), Outcome.Capture(() => dotnet.NpRandomSeed));
            ByteContract.AssertSame("np_random state after reset()", Outcome.CapturePython(() => {
                using PyObject generator = Python.gymnasium.Env.np_random(python.Object);
                using PyObject bitGenerator = Python.numpy.random.Generator.bit_generator(generator);
                return Python.numpy.random.BitGenerator.state(bitGenerator);
            }), Outcome.Returned(RngFacts.Of(dotnet.NpRandom)));

            ByteContract.AssertSame("setattr(env, 'np_random_seed', 1)", Outcome.CapturePython(() => {
                using PyObject one = PyConvert.ToPython(1L);
                return Python.builtins.setattr(python.Object, "np_random_seed", one);
            }), Outcome.Capture(() => {
                PyAttributes.SetAttr(dotnet, "np_random_seed", 1L);
                return null;
            }));

            PythonRuntime.Run(() => {
                using PyObject seven = PyConvert.ToPython(7L);
                using PyObject pair = Python.gymnasium.utils.seeding.np_random(seven);
                using PyObject generator = pair[0];
                Python.gymnasium.Env.np_random(python.Object, generator);
            });
            dotnet.NpRandom = Seeding.NpRandom(7).Rng;
            ByteContract.AssertSame("np_random_seed after assigning np_random", Outcome.CapturePython(() => Python.gymnasium.Env.np_random_seed(python.Object)), Outcome.Capture(() => dotnet.NpRandomSeed));
            ByteContract.AssertSame("reset(seed=-1)", Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, -1)), Outcome.Capture(() => TupleOf(dotnet.Reset(-1))));

            using PyRef pythonWrapped = PyRef.Of(() => {
                using PyObject cartPole = Python.gymnasium.envs.classic_control.CartPoleEnv();
                using PyObject limit = PyConvert.ToPython(10L);
                return Python.gymnasium.wrappers.TimeLimit(cartPole, limit);
            });
            var dotnetWrapped = new TimeLimit<NDArray, long>(new CartPoleEnv(), 10);
            ByteContract.AssertSame("wrapped reset(seed=9)", Outcome.CapturePython(() => Python.gymnasium.Env.reset(pythonWrapped.Object, 9)), Outcome.Capture(() => TupleOf(dotnetWrapped.Reset(9))));
            ByteContract.AssertSame("wrapped np_random_seed", Outcome.CapturePython(() => Python.gymnasium.Env.np_random_seed(pythonWrapped.Object)), Outcome.Capture(() => dotnetWrapped.NpRandomSeed));
            PythonRuntime.Run(() => {
                using PyObject seven = PyConvert.ToPython(7L);
                using PyObject pair = Python.gymnasium.utils.seeding.np_random(seven);
                using PyObject generator = pair[0];
                Python.gymnasium.Env.np_random(pythonWrapped.Object, generator);
            });
            dotnetWrapped.NpRandom = Seeding.NpRandom(7).Rng;
            ByteContract.AssertSame("wrapped np_random_seed after assigning np_random", Outcome.CapturePython(() => Python.gymnasium.Env.np_random_seed(pythonWrapped.Object)),
                Outcome.Capture(() => dotnetWrapped.NpRandomSeed));
            ByteContract.AssertSame("unwrapped np_random_seed after assigning through the wrapper", Outcome.CapturePython(() => {
                using PyObject unwrapped = Python.gymnasium.Env.unwrapped(pythonWrapped.Object);
                return Python.gymnasium.Env.np_random_seed(unwrapped);
            }), Outcome.Capture(() => dotnetWrapped.Unwrapped.NpRandomSeed));
        }

        /// <summary>
        ///     <c>render()</c> without a render mode: a bare <c>AssertionError</c> when the env has no spec (an env built with its
        ///     constructor), and with a spec the warning naming its id and <c>None</c>.
        /// </summary>
        [TestMethod]
        public void RenderWithoutRenderMode_MatchesGymnasium() {
            using PyRef python = PyRef.Of(() => Python.gymnasium.envs.classic_control.CartPoleEnv());
            var dotnet = new CartPoleEnv();
            ByteContract.AssertSame("reset(seed=0)", Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, 0)), Outcome.Capture(() => TupleOf(dotnet.Reset(0))));
            ByteContract.AssertSame("render() without a spec", Outcome.CapturePython(() => Python.gymnasium.Env.render(python.Object)), Outcome.Capture(() => FrameOf(dotnet.Render())));

            const string renderWithSpec =
                "import warnings\n" +
                "from gymnasium.envs.registration import EnvSpec\n" +
                "env.spec = EnvSpec('CartPole-v1')\n" +
                "with warnings.catch_warnings(record=True) as caught:\n" +
                "    warnings.simplefilter('always')\n" +
                "    rendered = env.render()\n" +
                "result = (rendered, [(w.category.__name__, str(w.message)) for w in caught])\n";
            Outcome pythonOutcome = Outcome.CapturePython(() => Python.evalStmts(renderWithSpec, "result", ("env", python.Object)));

            dotnet.Spec = new EnvSpec("CartPole-v1");
            Outcome dotnetOutcome = Outcome.Capture(() => {
                using WarningCatcher catcher = Warnings.Catch();
                object? rendered = FrameOf(dotnet.Render());
                var warnings = catcher.Records.Select(w => (object?)(w.Category.ToString(), w.Message)).ToList();
                return (rendered, warnings);
            });
            ByteContract.AssertSame("render() with a spec: the result and the warnings", pythonOutcome, dotnetOutcome);
        }

        /// <summary>
        ///     <c>EnvSpec</c>'s dataclass repr and <c>pprint</c> (three flag combinations) for specs with every field set and
        ///     with defaults, the spec a <c>TimeLimit(OrderEnforcing(env))</c> stack reports, and <c>WrapperSpec</c>'s repr.
        /// </summary>
        [TestMethod]
        public void EnvSpec_MatchesGymnasium() {
            var cases = new (string Id, string? EntryPoint, double? RewardThreshold, bool Nondeterministic, long? MaxEpisodeSteps, bool OrderEnforce, bool DisableEnvChecker,
                Dictionary<string, object?> Kwargs, (string Name, string EntryPoint, Dictionary<string, object?>? Kwargs)[] Wrappers, string? VectorEntryPoint)[] {
                ("CartPole-v1", "gymnasium.envs.classic_control.cartpole:CartPoleEnv", 475.0, false, 500, true, false, new Dictionary<string, object?>(),
                    Array.Empty<(string, string, Dictionary<string, object?>?)>(), "gymnasium.envs.classic_control.cartpole:CartPoleVectorEnv"),
                ("ns/Name-v3", null, null, true, null, false, true, new Dictionary<string, object?> { ["a"] = 1.5, ["b"] = new List<object?> { 1L, "x" }, ["c"] = null },
                    new[] { ("TimeLimit", "gymnasium.wrappers.common:TimeLimit", (Dictionary<string, object?>?)new Dictionary<string, object?> { ["max_episode_steps"] = 10L }) }, null),
                ("MyEnv", "my.module:MyEnv", -1e-5, false, 1L, true, false, new Dictionary<string, object?> { ["render_mode"] = "rgb_array", ["t"] = (1L, 2.5) },
                    new[] { ("A", "m:A", (Dictionary<string, object?>?)null), ("B", "m:B", new Dictionary<string, object?>()) }, null),
            };

            foreach (var c in cases) {
                using PyRef python = PyRef.Of(() => {
                    using PyObject envId = PyConvert.ToPython(c.Id);
                    using PyObject entryPoint = PyConvert.ToPython(c.EntryPoint);
                    using PyObject rewardThreshold = PyConvert.ToPython(c.RewardThreshold);
                    using PyObject nondeterministic = PyConvert.ToPython(c.Nondeterministic);
                    using PyObject maxEpisodeSteps = PyConvert.ToPython(c.MaxEpisodeSteps);
                    using PyObject orderEnforce = PyConvert.ToPython(c.OrderEnforce);
                    using PyObject disableEnvChecker = PyConvert.ToPython(c.DisableEnvChecker);
                    using PyObject kwargs = PyConvert.ToPython(c.Kwargs);
                    using PyObject vectorEntryPoint = PyConvert.ToPython(c.VectorEntryPoint);
                    var wrapperSpecs = new List<PyObject>();
                    try {
                        foreach (var w in c.Wrappers) {
                            using PyObject wrapperKwargs = PyConvert.ToPython(w.Kwargs);
                            wrapperSpecs.Add(Python.gymnasium.envs.registration.WrapperSpec(w.Name, w.EntryPoint, wrapperKwargs));
                        }

                        using var additionalWrappers = new PyTuple(wrapperSpecs.ToArray());
                        return Python.evalStmts(
                            "from gymnasium.envs.registration import EnvSpec\n" +
                            "spec = EnvSpec(env_id, entry_point=entry_point, reward_threshold=reward_threshold, nondeterministic=nondeterministic,\n" +
                            "               max_episode_steps=max_episode_steps, order_enforce=order_enforce, disable_env_checker=disable_env_checker,\n" +
                            "               kwargs=kwargs, additional_wrappers=additional_wrappers, vector_entry_point=vector_entry_point)\n",
                            "spec", ("env_id", envId), ("entry_point", entryPoint), ("reward_threshold", rewardThreshold), ("nondeterministic", nondeterministic),
                            ("max_episode_steps", maxEpisodeSteps), ("order_enforce", orderEnforce), ("disable_env_checker", disableEnvChecker), ("kwargs", kwargs),
                            ("additional_wrappers", additionalWrappers), ("vector_entry_point", vectorEntryPoint));
                    } finally {
                        foreach (PyObject spec in wrapperSpecs)
                            spec.Dispose();
                    }
                });
                var dotnet = new EnvSpec(c.Id, c.EntryPoint, c.RewardThreshold, c.Nondeterministic, c.MaxEpisodeSteps, c.OrderEnforce, c.DisableEnvChecker, new Info(c.Kwargs),
                    c.Wrappers.Select(w => new WrapperSpec(w.Name, w.EntryPoint, w.Kwargs is null ? null : new Info(w.Kwargs))), c.VectorEntryPoint);

                ByteContract.AssertSame($"repr(EnvSpec({c.Id}))", Outcome.CapturePython(() => Python.builtins.repr(python.Object)), Outcome.Returned(ValueFacts.FromText(dotnet.ToString())));
                foreach ((bool entryPoints, bool all) in new[] { (false, false), (true, false), (false, true), (true, true) })
                    ByteContract.AssertSame($"EnvSpec({c.Id}).pprint(include_entry_points={entryPoints}, print_all={all})",
                        Outcome.CapturePython(() => Python.gymnasium.envs.registration.EnvSpec.pprint(python.Object, entryPoints, all)),
                        Outcome.Returned(ValueFacts.FromText(dotnet.Pprint(disablePrint: true, includeEntryPoints: entryPoints, printAll: all)!)));
            }

            // The spec of a TimeLimit(OrderEnforcing(env)) stack over the spec make gives the unwrapped env.
            using PyRef pythonStack = PyRef.Of(() => {
                using PyObject cartPole = Python.gymnasium.envs.classic_control.CartPoleEnv();
                using PyObject spec = Python.evalStmts(
                    "from gymnasium.envs.registration import EnvSpec\n" +
                    "spec = EnvSpec('CartPole-v1', 'gymnasium.envs.classic_control.cartpole:CartPoleEnv', reward_threshold=475.0, order_enforce=False,\n" +
                    "               disable_env_checker=True)\n", "spec");
                Python.gymnasium.Env.spec(cartPole, spec);
                using PyObject ordered = Python.gymnasium.wrappers.OrderEnforcing(cartPole);
                using PyObject limit = PyConvert.ToPython(500L);
                return Python.gymnasium.wrappers.TimeLimit(ordered, limit);
            });
            var dotnetStack = new TimeLimit<NDArray, long>(new OrderEnforcing<NDArray, long>(new CartPoleEnv {
                Spec = new EnvSpec("CartPole-v1", "gymnasium.envs.classic_control.cartpole:CartPoleEnv", rewardThreshold: 475.0, orderEnforce: false, disableEnvChecker: true)
            }), 500);
            ByteContract.AssertSame("repr(TimeLimit(OrderEnforcing(env)).spec)", Outcome.CapturePython(() => {
                using PyObject spec = Python.gymnasium.Env.spec(pythonStack.Object);
                return Python.builtins.repr(spec);
            }), Outcome.Returned(ValueFacts.FromText(dotnetStack.Spec!.ToString())));
            ByteContract.AssertSame("str(TimeLimit(OrderEnforcing(env)))", Outcome.CapturePython(() => Python.builtins.str(pythonStack.Object)), Outcome.Returned(ValueFacts.FromText(dotnetStack.ToString())));
        }

        /// <summary><c>parse_env_id</c> over valid and malformed ids (the <c>Error</c> message included), and <c>get_env_id</c>.</summary>
        [TestMethod]
        public void EnvIds_MatchGymnasium() {
            string[] ids = {
                "CartPole-v1", "ALE/Pong-v5", "MyEnv", "ns/My.Env-v0", "my:ns/Name-v12", "Name-v1-v2", "Name-v", "a-b_c.d:e-v007", "A/B-v1",
                "a/b/c-v1", "", "CartPole v1", "CartPole-v1\n", "/Name-v1", "ns/", "Name-v-1", "CartPole-v٣",
            };
            foreach (string id in ids)
                ByteContract.AssertSame($"parse_env_id({PyBuiltins.Repr(id)})", Outcome.CapturePython(() => Python.gymnasium.envs.registration.parse_env_id(id)),
                    Outcome.Capture(() => EnvRegistry.ParseEnvId(id)));

            foreach ((string? ns, string name, long? version) in new (string?, string, long?)[] { (null, "CartPole", 1), ("ALE", "Pong", 5), (null, "MyEnv", null), ("ns", "Name", 0) })
                ByteContract.AssertSame($"get_env_id({PyBuiltins.Repr(ns)}, {PyBuiltins.Repr(name)}, {PyBuiltins.Repr(version)})",
                    Outcome.CapturePython(() => Python.gymnasium.envs.registration.get_env_id(ns, name, version)), Outcome.Capture(() => EnvRegistry.GetEnvId(ns, name, version)));
        }

        /// <summary>
        ///     <c>convert_to_done_step_api</c> and <c>convert_to_terminated_truncated_step_api</c> for single envs: every
        ///     combination of the flags and of the <c>"TimeLimit.truncated"</c> key, the returned tuple and the info as the
        ///     conversion leaves it.
        /// </summary>
        /// <remarks>
        ///     The key holds booleans here, as Gymnasium's envs store it. A non-boolean value is a documented deviation, not a
        ///     gap: Python's <c>dones and truncated</c> returns the operand itself (an int <c>0</c> stays <c>0</c>), while the
        ///     flags of <see cref="StepResult{TObs}"/> are <see langword="bool"/>, the operand's truth value.
        /// </remarks>
        [TestMethod]
        public void StepApiCompatibility_MatchesGymnasium() {
            var infos = new[] {
                new Dictionary<string, object?>(),
                new Dictionary<string, object?> { ["Test-info"] = true },
                new Dictionary<string, object?> { ["TimeLimit.truncated"] = false },
                new Dictionary<string, object?> { ["TimeLimit.truncated"] = true, ["x"] = 1L },
            };

            foreach (bool terminated in new[] { false, true })
            foreach (bool truncated in new[] { false, true })
            foreach (Dictionary<string, object?> info in infos) {
                string call = $"convert_to_done_step_api((0, 1.5, {PyBuiltins.Repr(terminated)}, {PyBuiltins.Repr(truncated)}, {PyBuiltins.Repr(info)}))";
                ByteContract.AssertSame(call, Outcome.CapturePython(() => {
                    using PyObject returns = PyConvert.ToPython((0L, 1.5, terminated, truncated, info));
                    return Python.gymnasium.utils.step_api_compatibility.convert_to_done_step_api(returns);
                }), Outcome.Capture(() => {
                    DoneStepResult<object?> done = StepApiCompatibility.ConvertToDoneStepApi(new StepResult<object?>(0L, 1.5, terminated, truncated, new Info(info)));
                    return (done.Observation, done.Reward, done.Done, done.Info);
                }));
            }

            foreach (bool done in new[] { false, true })
            foreach (Dictionary<string, object?> info in infos) {
                string call = $"convert_to_terminated_truncated_step_api((0, 1.5, {PyBuiltins.Repr(done)}, {PyBuiltins.Repr(info)}))";
                ByteContract.AssertSame(call, Outcome.CapturePython(() => {
                    using PyObject returns = PyConvert.ToPython((0L, 1.5, done, info));
                    return Python.gymnasium.utils.step_api_compatibility.convert_to_terminated_truncated_step_api(returns);
                }), Outcome.Capture(() => {
                    StepResult<object?> step = StepApiCompatibility.ConvertToTerminatedTruncatedStepApi(new DoneStepResult<object?>(0L, 1.5, done, new Info(info)));
                    return (step.Observation, step.Reward, step.Terminated, step.Truncated, step.Info);
                }));
            }
        }

        /// <summary>
        ///     <c>repr(dict)</c> against <see cref="Info.ToString"/> (<see cref="PyBuiltins.Repr"/>): nested dicts, lists and
        ///     tuples, strings that need escaping, and floats in every notation Python prints.
        /// </summary>
        [TestMethod]
        public void InfoRepr_MatchesGymnasium() {
            var cases = new[] {
                new Dictionary<string, object?>(),
                new Dictionary<string, object?> { ["render_modes"] = new List<object?> { "human", "rgb_array" }, ["render_fps"] = 50L },
                new Dictionary<string, object?> { ["a"] = 0.1, ["b"] = 1e16, ["c"] = 1e-5, ["d"] = -0.0, ["e"] = double.NaN, ["f"] = double.NegativeInfinity, ["g"] = 1.5e300, ["h"] = 123456789.125 },
                new Dictionary<string, object?> { ["quote's"] = "it's", ["both"] = "'\"", ["esc"] = "tab\tnew\nline\\", ["uni"] = "éÿĀ", ["none"] = null, ["t"] = true, ["f"] = false },
                new Dictionary<string, object?> { ["big"] = System.Numerics.BigInteger.Pow(2, 70), ["neg"] = -7L, ["one"] = (1L, 0), ["pair"] = ("x", 2.0), ["empty"] = new List<object?>() },
                new Dictionary<string, object?> { ["nested"] = new Dictionary<string, object?> { ["inner"] = new List<object?> { new Dictionary<string, object?> { ["k"] = (1L, 2L, 3L) } } } },
            };

            foreach (Dictionary<string, object?> entries in cases) {
                var info = new Info(entries);
                ByteContract.AssertSame($"repr({info})", Outcome.CapturePython(() => {
                    using PyObject dictionary = PyConvert.ToPython(entries);
                    return Python.builtins.repr(dictionary);
                }), Outcome.Returned(ValueFacts.FromText(info.ToString())));
            }
        }

        /// <summary>
        ///     Gym.NET's counterpart of <c>gym.make("CartPole-v1")</c> until P3: <c>TimeLimit(OrderEnforcing(CartPoleEnv), 500)</c>
        ///     with the spec <c>make</c> gives the unwrapped env (the wrappers' settings cleared, so each puts its own back).
        /// </summary>
        /// <returns>The stack.</returns>
        private static TimeLimit<NDArray, long> MadeCartPole() {
            var cartPole = new CartPoleEnv {
                Spec = new EnvSpec("CartPole-v1", "gymnasium.envs.classic_control.cartpole:CartPoleEnv", rewardThreshold: 475.0, orderEnforce: false, disableEnvChecker: true,
                    vectorEntryPoint: "gymnasium.envs.classic_control.cartpole:CartPoleVectorEnv")
            };
            return new TimeLimit<NDArray, long>(new OrderEnforcing<NDArray, long>(cartPole), 500);
        }

        /// <summary>Gymnasium's <c>env.step(action)</c> with an int action, captured.</summary>
        /// <param name="env">The Python env.</param>
        /// <param name="action">The action.</param>
        /// <returns>The outcome.</returns>
        private static Outcome CapturePythonStep(PyObject env, long action) {
            return Outcome.CapturePython(() => {
                using PyObject pythonAction = PyConvert.ToPython(action);
                return Python.gymnasium.Env.step(env, pythonAction);
            });
        }

        /// <summary>A reset result as the tuple Python returns, <c>(obs, info)</c>, so it is described like Gymnasium's.</summary>
        /// <param name="result">The result.</param>
        /// <returns>The tuple.</returns>
        private static (NDArray, Info) TupleOf(ResetResult<NDArray> result) => (result.Observation, result.Info);

        /// <summary>A step result as the tuple Python returns, <c>(obs, reward, terminated, truncated, info)</c>.</summary>
        /// <param name="result">The result.</param>
        /// <returns>The tuple.</returns>
        private static (NDArray, double, bool, bool, Info) TupleOf(StepResult<NDArray> result) => (result.Observation, result.Reward, result.Terminated, result.Truncated, result.Info);

        /// <summary>A rendering as Python returns it: the frame array for <c>rgb_array</c>, <see langword="null"/> (<c>None</c>) otherwise.</summary>
        /// <param name="result">The rendering.</param>
        /// <returns>The frame or <see langword="null"/>.</returns>
        /// <exception cref="InvalidOperationException">Thrown for a rendering that is neither.</exception>
        private static NDArray? FrameOf(RenderResult result) {
            return result switch {
                RenderResult.NoneResult => null,
                RenderResult.Rgb rgb => rgb.Frame,
                _ => throw new InvalidOperationException($"Unexpected rendering {result}.")
            };
        }
    }
}
