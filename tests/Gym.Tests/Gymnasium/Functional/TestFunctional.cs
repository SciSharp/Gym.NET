#nullable enable
using System.Collections.Generic;
using System.Linq;
using Gym.Experimental.Functional;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Functional {
    /// <summary>Gymnasium's functional API tests (<c>tests/functional/test_functional.py</c>), run against <see cref="FuncEnv"/>.</summary>
    /// <remarks>
    ///     <c>test_func_jax_env.py</c> and the JAX Blackjack and CliffWalking tests need JAX, which Gym.NET has no counterpart
    ///     for (DR-36).
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/functional/test_functional.py">tests/functional/test_functional.py</see>.
    /// </remarks>
    [TestClass]
    public class TestFunctional {
        /// <summary>Tests the core functional api specification using a generic testing environment.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/functional/test_functional.py#L49">tests/functional/test_functional.py (test_functional_api)</see>.</remarks>
        [TestMethod]
        public void TestFunctionalApi() => FunctionalApi.Check(new GenericTestFuncEnv());

        /// <summary>
        ///     The constructor's options set attributes by Python name, declared or not, and <see cref="FuncEnv{TState, TObs, TAct, TReward, TTerminal, TRenderState, TParams}.Transform(System.Func{System.Delegate, System.Delegate})"/>
        ///     visits the functions in Gymnasium's order, each bound to its <c>…Core</c> method (DR-43), and creates
        ///     <c>step_info</c> from <c>transition_info</c>; every function is an attribute, as in Python.
        /// </summary>
        [TestMethod]
        public void TestOptionsAndTransform() {
            var env = new GenericTestFuncEnv(new Dictionary<string, object?> { ["scale"] = 2.0, ["default_params"] = "given" });
            Assert.AreEqual(2.0, PyAttributes.GetAttr(env, "scale"));
            // get_default_params runs after the options, as Python's __init__ orders them.
            Assert.IsNull(env.DefaultParams);
            Assert.IsFalse(PyAttributes.HasAttr(env, "step_info"), "step_info exists only once a transform made it");

            var names = new List<string>();
            env.Transform(function => {
                names.Add(function.Method.Name);
                return function;
            });
            CollectionAssert.AreEqual(new[] { "InitialCore", "TransitionCore", "ObservationCore", "RewardCore", "TerminalCore", "StateInfoCore", "TransitionInfoCore" }, names);
            foreach (string name in new[] { "initial", "transition", "observation", "reward", "terminal", "state_info", "transition_info", "step_info" })
                Assert.IsTrue(PyAttributes.HasAttr(env, name), name);
        }

        /// <summary>
        ///     DR-43: a transform replaces the functions for every caller, C# calls included, and Gymnasium's quirk holds:
        ///     <c>transition_info</c> stays as it was while the transformed one is <c>step_info</c>. A typed transform wraps
        ///     every signature (an expression tree around each function, counting its calls).
        /// </summary>
        [TestMethod]
        public void Transform_ReplacesTheFunctionsForEveryCaller() {
            var env = new GenericTestFuncEnv();
            var counting = new CountingTransform();
            env.Transform(counting);

            using var state = (NDArray)env.Initial(null)!;
            Assert.AreEqual(1, counting.Calls, "Initial runs the transformed function");
            using var next = (NDArray)env.Transition(state, 1, null)!;
            Assert.AreEqual(2, counting.Calls, "Transition runs the transformed function");
            env.TransitionInfo(state, 1, next);
            Assert.AreEqual(2, counting.Calls, "transition_info is left as it was");
            env.StepInfo(state, 1, next);
            Assert.AreEqual(3, counting.Calls, "step_info is the transformed transition_info");
        }

        /// <summary>A typed transform that wraps each function in a delegate of its own type, counting the calls.</summary>
        private sealed class CountingTransform : IFunctionTransform {
            /// <summary>How many wrapped calls ran.</summary>
            public int Calls { get; private set; }

            /// <summary>Wraps <paramref name="function"/>: a delegate of the same type that counts, then calls it.</summary>
            /// <typeparam name="TDelegate">The function's delegate type.</typeparam>
            /// <param name="function">The function.</param>
            /// <returns>The counting wrapper.</returns>
            public TDelegate Apply<TDelegate>(TDelegate function) where TDelegate : System.Delegate {
                var parameters = typeof(TDelegate).GetMethod("Invoke")!.GetParameters()
                                                   .Select(p => System.Linq.Expressions.Expression.Parameter(p.ParameterType, p.Name)).ToArray();
                var body = System.Linq.Expressions.Expression.Block(
                    System.Linq.Expressions.Expression.Call(System.Linq.Expressions.Expression.Constant(this), typeof(CountingTransform).GetMethod(nameof(Count))!),
                    System.Linq.Expressions.Expression.Invoke(System.Linq.Expressions.Expression.Constant(function), parameters));
                return System.Linq.Expressions.Expression.Lambda<TDelegate>(body, parameters).Compile();
            }

            /// <summary>Counts one call.</summary>
            public void Count() => Calls++;
        }

        /// <summary>Generic testing functional environment (<c>GenericTestFuncEnv</c>): a float32 pair whose second entry accumulates the actions.</summary>
        internal sealed class GenericTestFuncEnv : FuncEnv {
            /// <summary>Constructor that allows generic options to be set on the environment.</summary>
            /// <param name="options">Attribute values by Python name, or <see langword="null"/>.</param>
            public GenericTestFuncEnv(IReadOnlyDictionary<string, object?>? options = null) : base(options) { }

            /// <summary>Testing initial function: <c>[0, 0]</c>, float32.</summary>
            /// <param name="rng">Ignored.</param>
            /// <param name="params">Ignored.</param>
            /// <returns>The state.</returns>
            protected override object? InitialCore(object? rng, object? @params) => np.array(new float[] { 0, 0 });

            /// <summary>Testing observation function: the state itself.</summary>
            /// <param name="state">The state.</param>
            /// <param name="rng">Ignored.</param>
            /// <param name="params">Ignored.</param>
            /// <returns>The state.</returns>
            protected override object? ObservationCore(object? state, object? rng, object? @params) => state;

            /// <summary>Testing transition function: adds <c>[0, action]</c> to the state.</summary>
            /// <param name="state">The state.</param>
            /// <param name="action">The action, an integer.</param>
            /// <param name="rng">Ignored.</param>
            /// <param name="params">Ignored.</param>
            /// <returns>The next state, float32.</returns>
            protected override object? TransitionCore(object? state, object? action, object? rng, object? @params) {
                using NDArray step = np.array(new float[] { 0, (float)PyBuiltins.Float(action) });
                return (NDArray)state! + step;
            }

            /// <summary>Testing reward function: 1 when the next state's second entry is positive, else 0.</summary>
            /// <param name="state">Ignored.</param>
            /// <param name="action">Ignored.</param>
            /// <param name="nextState">The state after.</param>
            /// <param name="rng">Ignored.</param>
            /// <param name="params">Ignored.</param>
            /// <returns>The reward.</returns>
            protected override object? RewardCore(object? state, object? action, object? nextState, object? rng, object? @params) =>
                ((NDArray)nextState!).GetSingle(1) > 0 ? 1.0 : 0.0;

            /// <summary>Testing terminal function: the second entry is positive.</summary>
            /// <param name="state">The state.</param>
            /// <param name="rng">Ignored.</param>
            /// <param name="params">Ignored.</param>
            /// <returns>Whether the state is terminal.</returns>
            protected override object? TerminalCore(object? state, object? rng, object? @params) => ((NDArray)state!).GetSingle(1) > 0;
        }
    }

    /// <summary>The functional API's checks shared by <c>test_functional.py</c> and <c>tests/envs/functional/test_core.py</c>.</summary>
    internal static class FunctionalApi {
        /// <summary>
        ///     The initial state and its observation are float32 pairs; six actions move the second entry, each transition's
        ///     state, observation, reward and terminal flag as Gymnasium's tests expect (terminal only after the last).
        /// </summary>
        /// <param name="env">The env under test.</param>
        /// <exception cref="AssertFailedException">Thrown at the first difference.</exception>
        internal static void Check(FuncEnv env) {
            var state = (NDArray)env.Initial(null)!;

            var obs = (NDArray)env.Observation(state, null)!;

            CollectionAssert.AreEqual(new long[] { 2 }, state.shape);
            Assert.IsTrue(state.typecode == NPTypeCode.Single);
            CollectionAssert.AreEqual(new long[] { 2 }, obs.shape);
            Assert.IsTrue(obs.typecode == NPTypeCode.Single);
            Assert.IsTrue(np.allclose(obs, state));

            long[] actions = { -1, -2, -5, 3, 5, 2 };
            for (int i = 0; i < actions.Length; i++) {
                long action = actions[i];
                var nextState = (NDArray)env.Transition(state, action, null)!;
                CollectionAssert.AreEqual(new long[] { 2 }, nextState.shape);
                Assert.IsTrue(nextState.typecode == NPTypeCode.Single);
                using (NDArray step = np.array(new long[] { 0, action }))
                using (NDArray expected = state + step)
                    Assert.IsTrue(np.allclose(nextState, expected));

                var observation = (NDArray)env.Observation(nextState, null)!;
                CollectionAssert.AreEqual(new long[] { 2 }, observation.shape);
                Assert.IsTrue(observation.typecode == NPTypeCode.Single);
                Assert.IsTrue(np.allclose(observation, nextState));

                object? reward = env.Reward(state, action, nextState, null);
                Assert.AreEqual(nextState.GetSingle(1) > 0 ? 1.0 : 0.0, reward);

                object? terminal = env.Terminal(nextState, null);
                Assert.AreEqual(i == 5, terminal); // terminal state is in the final action

                state = nextState;
            }
        }
    }
}
