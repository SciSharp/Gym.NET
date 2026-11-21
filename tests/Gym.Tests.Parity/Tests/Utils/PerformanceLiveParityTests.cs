using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Envs;
using Gym.Experimental.Functional;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Session;
using Gym.Utils;
using Gym.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Python.Runtime;

namespace Gym.Tests.Parity.Tests.Utils {
    /// <summary>
    ///     <c>gymnasium.utils.performance</c> and <c>gymnasium.experimental.functional.FuncEnv</c> against Gym.NET's
    ///     <see cref="Performance"/> and <see cref="FuncEnv"/>, live, with both clocks scripted so the rates are
    ///     deterministic.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Gymnasium's benchmarks read <c>time.time</c>, which the cells replace for the call (and restore), as Gymnasium's
    ///     own tests monkeypatch it; Gym.NET's read <see cref="Performance.Time"/>. Only the vector benchmark seeds the
    ///     action space, so it is the one whose run is compared beyond its rate: the sub-envs' states afterwards.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/performance.py">gymnasium/utils/performance.py</see>, <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/experimental/functional.py">gymnasium/experimental/functional.py</see>.</para>
    /// </remarks>
    [TestClass]
    public sealed class PerformanceLiveParityTests : ParityTestBase {
        /// <summary>
        ///     <c>benchmark_vector_step</c> over two one-step CartPoles with a clock scripted to <c>0, 0, 1, 1.5, 3</c>: the rate
        ///     and the sub-envs' states afterwards (as float64 values), under each autoreset mode.
        /// </summary>
        /// <param name="autoresetMode">The autoreset mode's value.</param>
        [TestMethod]
        [DataRow("NextStep")]
        [DataRow("SameStep")]
        [DataRow("Disabled")]
        public void BenchmarkVectorStep_MatchesGymnasium(string autoresetMode) {
            const string statements = """
                import gymnasium as gym
                import numpy as np
                from gymnasium.utils import performance
                from gymnasium.vector import SyncVectorEnv
                env = SyncVectorEnv([lambda: gym.make("CartPole-v1", max_episode_steps=1) for _ in range(2)], autoreset_mode=mode)
                timestamps = iter([0.0, 0.0, 1.0, 1.5, 3.0])
                original = performance.time.time
                performance.time.time = lambda: next(timestamps)
                try:
                    rate = performance.benchmark_vector_step(env, target_duration=2, seed=123)
                finally:
                    performance.time.time = original
                result = (rate, tuple(np.asarray(state, dtype=np.float64) for state in env.get_attr("state")))
                env.close()
                """;
            Outcome python = Outcome.CapturePython(() => {
                using var mode = new PyString(autoresetMode);
                return Python.evalStmts(statements, "result", ("mode", mode));
            });

            Func<double> previous = Performance.Time;
            var timestamps = new Queue<double>(new[] { 0.0, 0.0, 1.0, 1.5, 3.0 });
            Performance.Time = () => timestamps.Dequeue();
            try {
                using var env = new SyncVectorEnv(Enumerable.Range(0, 2).Select(_ => (Func<IEnv>)(() => Gymnasium.Make("CartPole-v1", maxEpisodeSteps: 1))), autoresetMode: autoresetMode);
                // The states compare as float64 values, the parity suite's rule for a state attribute (a double[] here, an
                // ndarray in Gymnasium); the arrays made for that are released with the result.
                Outcome dotnet = Outcome.Capture(() => new object?[] {
                    Performance.BenchmarkVectorStep(env, targetDuration: 2, seed: 123), env.GetAttr("state").Select(state => (object?)np.array((double[])state!)).ToArray(),
                });
                env.Close();
                ByteContract.AssertSame($"benchmark_vector_step(CartPole x2, {autoresetMode}): the rate and the states", python, dotnet);
            } finally {
                Performance.Time = previous;
            }
        }

        /// <summary>
        ///     <c>benchmark_step</c>, <c>benchmark_init</c> and <c>benchmark_render</c> over CartPole with a clock scripted to
        ///     <c>0, 0, 0.5, 2.5</c>: three iterations each, the same rate.
        /// </summary>
        [TestMethod]
        public void BenchmarkRates_MatchGymnasium() {
            const string statements = """
                import gymnasium as gym
                from gymnasium.utils import performance
                original = performance.time.time
                rates = []
                try:
                    for run in ("step", "init", "render"):
                        timestamps = iter([0.0, 0.0, 0.5, 2.5])
                        performance.time.time = lambda: next(timestamps)
                        if run == "step":
                            rates.append(performance.benchmark_step(gym.make("CartPole-v1"), target_duration=1, seed=3))
                        elif run == "init":
                            rates.append(performance.benchmark_init(lambda: gym.make("CartPole-v1"), target_duration=1, seed=3))
                        else:
                            env = gym.make("CartPole-v1", render_mode="rgb_array")
                            env.reset(seed=3)
                            rates.append(performance.benchmark_render(env, target_duration=1))
                finally:
                    performance.time.time = original
                result = tuple(rates)
                """;
            Outcome python = Outcome.CapturePython(() => Python.evalStmts(statements, "result"));

            Func<double> previous = Performance.Time;
            try {
                var rates = new List<object?>();
                foreach (string run in new[] { "step", "init", "render" }) {
                    var timestamps = new Queue<double>(new[] { 0.0, 0.0, 0.5, 2.5 });
                    Performance.Time = () => timestamps.Dequeue();
                    if (run == "step") {
                        using IEnv env = Gymnasium.Make("CartPole-v1");
                        rates.Add(Performance.BenchmarkStep(env, targetDuration: 1, seed: 3));
                    } else if (run == "init") {
                        rates.Add(Performance.BenchmarkInit(() => Gymnasium.Make("CartPole-v1"), targetDuration: 1, seed: 3));
                    } else {
                        using IEnv env = Gymnasium.Make("CartPole-v1", new { render_mode = "rgb_array" });
                        Outcome.Release(env.Reset(3).Observation);
                        rates.Add(Performance.BenchmarkRender(env, targetDuration: 1));
                    }
                }

                ByteContract.AssertSame("benchmark_step, benchmark_init, benchmark_render: the rates", python, Outcome.Returned(ValueFacts.Of(rates.ToArray())));
            } finally {
                Performance.Time = previous;
            }
        }

        /// <summary>
        ///     A <c>FuncEnv</c> subclass that overrides nothing: its options set attributes (declared or not), the functions
        ///     raise <c>NotImplementedError</c>, the info functions give empty dicts, the default params are none, and
        ///     <c>transform</c> stores its functions under their names (<c>step_info</c> included).
        /// </summary>
        [TestMethod]
        public void FuncEnv_Base_MatchesGymnasium() {
            const string statements = """
                from gymnasium.experimental.functional import FuncEnv
                Bare = type("Bare", (FuncEnv,), {})
                env = Bare({"scale": 2.0})
                outcomes = []
                for call in (lambda: env.initial(None), lambda: env.transition(None, None, None), lambda: env.observation(None, None),
                             lambda: env.reward(None, None, None, None), lambda: env.terminal(None, None), lambda: env.render_init(),
                             lambda: env.render_image(None, None), lambda: env.render_close(None)):
                    try:
                        call()
                        outcomes.append("returned")
                    except NotImplementedError as e:
                        outcomes.append(("NotImplementedError", str(e)))
                names = []
                env.transform(lambda f: (names.append(f.__name__), f)[1])
                result = (env.scale, env.default_params, env.get_default_params(), env.state_info(None), env.transition_info(None, None, None),
                          tuple(outcomes), tuple(names), hasattr(env, "step_info"))
                """;
            Outcome python = Outcome.CapturePython(() => Python.evalStmts(statements, "result"));

            Outcome dotnet = Outcome.Capture(() => {
                var env = new BareFuncEnv(new Dictionary<string, object?> { ["scale"] = 2.0 });
                var outcomes = new List<object?>();
                foreach (Action call in new Action[] {
                             () => env.Initial(null), () => env.Transition(null, null, null), () => env.Observation(null, null), () => env.Reward(null, null, null, null),
                             () => env.Terminal(null, null), () => env.RenderInit(), () => Outcome.Release(env.RenderImage(null, null).Frame), () => env.RenderClose(null),
                         }) {
                    try {
                        call();
                        outcomes.Add("returned");
                    } catch (NumSharp.NotImplementedError e) {
                        outcomes.Add(new object?[] { "NotImplementedError", e.Message });
                    }
                }

                var names = new List<object?>();
                env.Transform(function => {
                    // Each function starts bound to its …Core method (DR-43), whose name less the suffix is Gymnasium's __name__.
                    string method = function.Method.Name;
                    names.Add(PyAttributes.SnakeCase(method.EndsWith("Core", StringComparison.Ordinal) ? method.Substring(0, method.Length - 4) : method));
                    return function;
                });
                return new object?[] {
                    PyAttributes.GetAttr(env, "scale"), env.DefaultParams, env.GetDefaultParams(), env.StateInfo(null), env.TransitionInfo(null, null, null),
                    outcomes.ToArray(), names.ToArray(), PyAttributes.HasAttr(env, "step_info"),
                };
            });
            ByteContract.AssertSame("Bare({'scale': 2.0}): attributes, NotImplementedError, infos, params, transform", python, dotnet);
        }

        /// <summary>A functional env overriding nothing, as <c>type("Bare", (FuncEnv,), {})</c>.</summary>
        private sealed class BareFuncEnv : FuncEnv {
            /// <summary>Sets the options as attributes.</summary>
            /// <param name="options">Attribute values by Python name.</param>
            public BareFuncEnv(IReadOnlyDictionary<string, object?> options) : base(options) { }
        }
    }
}
