#nullable enable
using System.Collections.Generic;
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
        ///     The constructor's options set attributes by Python name, declared or not, and <see cref="FuncEnv{TState, TObs, TAct, TReward, TTerminal, TRenderState, TParams}.Transform"/>
        ///     stores each transformed function under its Python name (<c>step_info</c> for <c>transition_info</c>).
        /// </summary>
        [TestMethod]
        public void TestOptionsAndTransform() {
            var env = new GenericTestFuncEnv(new Dictionary<string, object?> { ["scale"] = 2.0, ["default_params"] = "given" });
            Assert.AreEqual(2.0, PyAttributes.GetAttr(env, "scale"));
            // get_default_params runs after the options, as Python's __init__ orders them.
            Assert.IsNull(env.DefaultParams);

            var names = new List<string>();
            env.Transform(function => {
                names.Add(function.Method.Name);
                return function;
            });
            CollectionAssert.AreEqual(new[] { "Initial", "Transition", "Observation", "Reward", "Terminal", "StateInfo", "TransitionInfo" }, names);
            foreach (string name in new[] { "initial", "transition", "observation", "reward", "terminal", "state_info", "step_info" })
                Assert.IsTrue(PyAttributes.HasAttr(env, name), name);
            Assert.IsFalse(PyAttributes.HasAttr(env, "transition_info"));
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
            public override object? Initial(object? rng, object? @params = null) => np.array(new float[] { 0, 0 });

            /// <summary>Testing observation function: the state itself.</summary>
            /// <param name="state">The state.</param>
            /// <param name="rng">Ignored.</param>
            /// <param name="params">Ignored.</param>
            /// <returns>The state.</returns>
            public override object? Observation(object? state, object? rng, object? @params = null) => state;

            /// <summary>Testing transition function: adds <c>[0, action]</c> to the state.</summary>
            /// <param name="state">The state.</param>
            /// <param name="action">The action, an integer.</param>
            /// <param name="rng">Ignored.</param>
            /// <param name="params">Ignored.</param>
            /// <returns>The next state, float32.</returns>
            public override object? Transition(object? state, object? action, object? rng, object? @params = null) {
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
            public override object? Reward(object? state, object? action, object? nextState, object? rng, object? @params = null) =>
                ((NDArray)nextState!).GetSingle(1) > 0 ? 1.0 : 0.0;

            /// <summary>Testing terminal function: the second entry is positive.</summary>
            /// <param name="state">The state.</param>
            /// <param name="rng">Ignored.</param>
            /// <param name="params">Ignored.</param>
            /// <returns>Whether the state is terminal.</returns>
            public override object? Terminal(object? state, object? rng, object? @params = null) => ((NDArray)state!).GetSingle(1) > 0;
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
