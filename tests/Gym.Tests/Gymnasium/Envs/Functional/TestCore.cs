#nullable enable
using System.Collections.Generic;
using Gym.Experimental.Functional;
using Gym.Tests.Gymnasium.Functional;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Envs.Functional {
    /// <summary>Gymnasium's functional env core test (<c>tests/envs/functional/test_core.py</c>), run against <see cref="FuncEnv"/>.</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/functional/test_core.py">tests/envs/functional/test_core.py</see>.</remarks>
    [TestClass]
    public class TestCore {
        /// <summary>The functional API over a basic env: the same checks as <c>test_functional.py</c>'s.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/functional/test_core.py#L29">tests/envs/functional/test_core.py (test_api)</see>.</remarks>
        [TestMethod]
        public void TestApi() => FunctionalApi.Check(new BasicTestEnv());

        /// <summary>The basic functional env of the test (<c>BasicTestEnv</c>), the same dynamics as <see cref="TestFunctional.GenericTestFuncEnv"/>.</summary>
        private sealed class BasicTestEnv : FuncEnv {
            /// <summary>A basic env.</summary>
            /// <param name="options">Attribute values by Python name, or <see langword="null"/>.</param>
            public BasicTestEnv(IReadOnlyDictionary<string, object?>? options = null) : base(options) { }

            /// <summary><c>[0, 0]</c>, float32.</summary>
            /// <param name="rng">Ignored.</param>
            /// <param name="params">Ignored.</param>
            /// <returns>The state.</returns>
            public override object? Initial(object? rng, object? @params = null) => np.array(new float[] { 0, 0 });

            /// <summary>The state itself.</summary>
            /// <param name="state">The state.</param>
            /// <param name="rng">Ignored.</param>
            /// <param name="params">Ignored.</param>
            /// <returns>The state.</returns>
            public override object? Observation(object? state, object? rng, object? @params = null) => state;

            /// <summary>Adds <c>[0, action]</c>, float32.</summary>
            /// <param name="state">The state.</param>
            /// <param name="action">The action.</param>
            /// <param name="rng">Ignored.</param>
            /// <param name="params">Ignored.</param>
            /// <returns>The next state.</returns>
            public override object? Transition(object? state, object? action, object? rng, object? @params = null) {
                using NDArray step = np.array(new float[] { 0, (float)PyBuiltins.Float(action) });
                return (NDArray)state! + step;
            }

            /// <summary>1 when the next state's second entry is positive, else 0.</summary>
            /// <param name="state">Ignored.</param>
            /// <param name="action">Ignored.</param>
            /// <param name="nextState">The state after.</param>
            /// <param name="rng">Ignored.</param>
            /// <param name="params">Ignored.</param>
            /// <returns>The reward.</returns>
            public override object? Reward(object? state, object? action, object? nextState, object? rng, object? @params = null) =>
                ((NDArray)nextState!).GetSingle(1) > 0 ? 1.0 : 0.0;

            /// <summary>Whether the second entry is positive.</summary>
            /// <param name="state">The state.</param>
            /// <param name="rng">Ignored.</param>
            /// <param name="params">Ignored.</param>
            /// <returns>Whether the state is terminal.</returns>
            public override object? Terminal(object? state, object? rng, object? @params = null) => ((NDArray)state!).GetSingle(1) > 0;
        }
    }
}
