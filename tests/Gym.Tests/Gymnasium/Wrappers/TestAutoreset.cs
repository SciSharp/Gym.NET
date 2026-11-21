#nullable enable
using Gym.Envs;
using Gym.Observations;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Gym.Tests.Helpers;

namespace Gym.Tests.Gymnasium.Wrappers {
    /// <summary>
    ///     Gymnasium's <c>Autoreset</c> tests (<c>tests/wrappers/test_autoreset.py</c>): the step after an episode ends resets
    ///     the env instead of stepping it, with reward 0 and no signals.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_autoreset.py">tests/wrappers/test_autoreset.py</see>.</remarks>
    [TestClass]
    public class TestAutoreset {
        /// <summary>
        ///     An env that counts its steps and terminates at the third (reward 1 then): the fourth step is the reset, the count
        ///     back at 0.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_autoreset.py#L26">test_autoreset_wrapper_autoreset</see>.</remarks>
        [TestMethod]
        public void TestAutoresetWrapperAutoreset() {
            long count = 0;
            var genericEnv = new GenericTestEnv(
                resetFunc: (self, seed, options) => {
                    count = 0;
                    return new ResetResult<object?>(np.array(new[] { count }), new Info { ["count"] = count });
                },
                stepFunc: (self, action) => {
                    count++;
                    // Gymnasium's reward is the bool `count > 2`, which Python adds as 0 or 1.
                    return new StepResult<object?>(np.array(new[] { count }), count > 2 ? 1.0 : 0.0, count > 2, false, new Info { ["count"] = count });
                });
            using var env = new Autoreset<object?, object?>(genericEnv);
            using NDArray action = NDArray.Scalar(0L);

            (object? obs, Info info) = env.Reset();
            AssertObs(0, obs);
            PyAssert.AreEqual(new Info { ["count"] = 0L }, info);

            StepResult<object?> step = env.Step(action);
            AssertObs(1, step.Observation);
            Assert.AreEqual(0.0, step.Reward);
            Assert.IsFalse(step.Terminated || step.Truncated);
            PyAssert.AreEqual(new Info { ["count"] = 1L }, step.Info);

            step = env.Step(action);
            AssertObs(2, step.Observation);
            Assert.IsFalse(step.Terminated || step.Truncated);
            Assert.AreEqual(0.0, step.Reward);
            PyAssert.AreEqual(new Info { ["count"] = 2L }, step.Info);

            step = env.Step(action);
            AssertObs(3, step.Observation);
            Assert.IsTrue(step.Terminated || step.Truncated);
            Assert.AreEqual(1.0, step.Reward);
            PyAssert.AreEqual(new Info { ["count"] = 3L }, step.Info);

            step = env.Step(action);
            AssertObs(0, step.Observation);
            Assert.AreEqual(0.0, step.Reward);
            Assert.IsFalse(step.Terminated || step.Truncated);
            PyAssert.AreEqual(new Info { ["count"] = 0L }, step.Info);

            env.Close();
        }

        /// <summary>Asserts the observation is the one-element array <c>[expected]</c> and releases it.</summary>
        /// <param name="expected">The count it should hold.</param>
        /// <param name="observation">The observation.</param>
        private static void AssertObs(long expected, object? observation) {
            using var array = (NDArray)observation!;
            Assert.AreEqual(1L, array.size);
            Assert.AreEqual(expected, array.GetInt64(0));
        }
    }
}
