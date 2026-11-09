#nullable enable
using System.Collections.Generic;
using System.Linq;
using Gym.Envs;
using Gym.Tests.Gymnasium.Spaces;
using Gym.Utils;
using Gym.Vector;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Wrappers {
    /// <summary>Gymnasium's <c>FrameStackObservation</c> tests (<c>test_frame_stack_observation.py</c>), run against <see cref="FrameStackObservation{TObs, TAct}"/>.</summary>
    /// <remarks>
    ///     <c>test_stack_size_failures</c>' <c>TypeError</c> for a float stack size has no counterpart: the size is a
    ///     <see cref="long"/>, so the compiler refuses a float. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_frame_stack_observation.py">tests/wrappers/test_frame_stack_observation.py</see>.
    /// </remarks>
    [TestClass]
    public class TestFrameStackObservation {
        /// <summary>Every space of Gymnasium's testing catalog, by position (<c>TESTING_OBS_ENVS</c>).</summary>
        public static IEnumerable<object[]> Spaces => TestingSpaces.AllIndexes;

        /// <summary>
        ///     Over an env of every testing space: each stack is a member of the batched space, and iterating it gives the
        ///     last three observations of the unwrapped env, the reset's repeated before the first step.
        /// </summary>
        /// <param name="index">The space's position in <see cref="TestingSpaces.All"/>.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_frame_stack_observation.py#L16">tests/wrappers/test_frame_stack_observation.py (test_different_obs_spaces)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Spaces))]
        public void TestDifferentObsSpaces(int index) {
            const int stackSize = 3;
            var env = new GenericTestEnv(observationSpace: TestingSpaces.All[index]());
            var (obs, _) = env.Reset(seed: WrapperUtils.Seed);
            env.ActionSpace.Seed(WrapperUtils.Seed);

            var unstackedObs = Enumerable.Repeat(obs, stackSize).ToList();
            for (int i = 0; i < stackSize * 2; i++) {
                (obs, _, _, _, _) = env.Step(env.ActionSpace.Sample());
                unstackedObs.Add(obs);
            }

            using var wrapped = new FrameStackObservation<object?, object?>(env, stackSize);
            wrapped.ActionSpace.Seed(WrapperUtils.Seed);

            (obs, _) = wrapped.Reset(seed: WrapperUtils.Seed);
            var stackedObs = new List<object?> { obs };
            Assert.IsTrue(wrapped.ObservationSpace.Contains(obs));

            for (int i = 0; i < stackSize * 2; i++) {
                (obs, _, _, _, _) = wrapped.Step(wrapped.ActionSpace.Sample());
                stackedObs.Add(obs);
                Assert.IsTrue(wrapped.ObservationSpace.Contains(obs));
            }

            Assert.AreEqual(unstackedObs.Count, stackedObs.Count + stackSize - 1);
            for (int i = 0; i < stackedObs.Count; i++) {
                List<object?> frames = VectorUtils.Iterate(wrapped.ObservationSpace, stackedObs[i]).ToList();
                Assert.IsTrue(EnvChecker.DataEquivalence(unstackedObs.GetRange(i, stackSize), frames), $"stack {i}: {PyBuiltins.Repr(stackedObs[i])}");
            }
        }

        /// <summary>CartPole stacked by 2 to 4: every frame of every stack is the observation it should be, the reset's padding first.</summary>
        /// <param name="stackSize">The stack size.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_frame_stack_observation.py#L47">tests/wrappers/test_frame_stack_observation.py (test_stack_size)</see>.</remarks>
        [TestMethod]
        [DataRow(2L)]
        [DataRow(3L)]
        [DataRow(4L)]
        public void TestStackSize(long stackSize) {
            var env = (Env<NDArray, NDArray>)Make("CartPole-v1");
            env.ActionSpace.Seed(WrapperUtils.Seed);

            // Perform a series of actions and store the resulting observations
            var unstackedObs = new List<NDArray>();
            NDArray obs = env.Reset(seed: WrapperUtils.Seed).Observation;
            unstackedObs.Add(obs);
            NDArray firstObs = obs;
            for (int i = 0; i < 5; i++) {
                (obs, _, _, _, _) = env.Step(env.ActionSpace.Sample());
                unstackedObs.Add(obs);
            }

            using var wrapped = new FrameStackObservation<NDArray, NDArray>(env, stackSize);
            wrapped.ActionSpace.Seed(WrapperUtils.Seed);

            // Perform the same series of actions and store the resulting stacked observations
            var stackedObs = new List<NDArray> { wrapped.Reset(seed: WrapperUtils.Seed).Observation };
            for (int i = 0; i < 5; i++)
                stackedObs.Add(wrapped.Step(wrapped.ActionSpace.Sample()).Observation);

            // Check that the frames in each stacked observation match the corresponding observations
            for (int i = 0; i < stackedObs.Count; i++) {
                List<object?> frames = VectorUtils.Iterate(wrapped.ObservationSpace, stackedObs[i]).ToList();
                for (int j = 0; j < stackSize; j++) {
                    // Use the first observation instead of a zero observation
                    NDArray expectedObs = i - j < 0 ? firstObs : unstackedObs[i - j];
                    Assert.IsTrue(EnvChecker.DataEquivalence(expectedObs, frames[(int)stackSize - 1 - j]), $"stack {i}, frame {j}");
                }
            }
        }

        /// <summary>The three paddings over CartPole: the reset's observation, zeros, and a given array.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_frame_stack_observation.py#L84">tests/wrappers/test_frame_stack_observation.py (test_padding_type)</see>.</remarks>
        [TestMethod]
        public void TestPaddingType() {
            var env = (Env<NDArray, NDArray>)Make("CartPole-v1");
            using NDArray resetObs = env.Reset(seed: 123).Observation;
            using NDArray action = env.ActionSpace.Sample();
            using NDArray stepObs = env.Step(action).Observation;
            using NDArray zeros = np.zeros(new Shape(4L));
            using NDArray custom = np.array(new float[] { 1, -1, 0, 2 });

            var stackedEnv = new FrameStackObservation<NDArray, NDArray>(env, 3); // default = "reset"
            AssertStack(stackedEnv.Reset(seed: 123).Observation, resetObs, resetObs, resetObs);
            AssertStack(stackedEnv.Step(action).Observation, resetObs, resetObs, stepObs);

            stackedEnv = new FrameStackObservation<NDArray, NDArray>(env, 3, paddingType: "zero");
            AssertStack(stackedEnv.Reset(seed: 123).Observation, zeros, zeros, resetObs);
            AssertStack(stackedEnv.Step(action).Observation, zeros, resetObs, stepObs);

            stackedEnv = new FrameStackObservation<NDArray, NDArray>(env, 3, paddingType: custom);
            AssertStack(stackedEnv.Reset(seed: 123).Observation, custom, custom, resetObs);
            AssertStack(stackedEnv.Step(action).Observation, custom, resetObs, stepObs);
            // The three wrappers share the env, so only the last one closes it.
            stackedEnv.Dispose();
        }

        /// <summary>A stack of 0, a misspelt padding mode, and an array outside the space are refused with Gymnasium's messages.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_frame_stack_observation.py#L123">tests/wrappers/test_frame_stack_observation.py (test_stack_size_failures)</see>.</remarks>
        [TestMethod]
        public void TestStackSizeFailures() {
            using var env = (Env<NDArray, NDArray>)Make("CartPole-v1");

            var error = Assert.ThrowsExactly<ValueError>(() => new FrameStackObservation<NDArray, NDArray>(env, stackSize: 0));
            StringAssert.Contains(error.Message, "The stack_size needs to be greater than zero, actual value: 0");

            error = Assert.ThrowsExactly<ValueError>(() => new FrameStackObservation<NDArray, NDArray>(env, stackSize: 3, paddingType: "unknown"));
            StringAssert.Contains(error.Message, "Unexpected `padding_type`, expected 'reset', 'zero' or a custom observation space, actual value: 'unknown'");

            using NDArray invalidPadding = np.array(new long[] { 1, 2, 3, 4, 5 });
            Assert.IsFalse(env.ObservationSpace.Contains(invalidPadding));
            error = Assert.ThrowsExactly<ValueError>(() => new FrameStackObservation<NDArray, NDArray>(env, stackSize: 3, paddingType: invalidPadding));
            StringAssert.Contains(error.Message, "Unexpected `padding_type`, expected 'reset', 'zero' or a custom observation space, actual value: array([1, 2, 3, 4, 5])");
        }

        /// <summary><c>np.all(np.stack(expected) == stacked)</c>.</summary>
        /// <param name="stacked">The wrapper's stack.</param>
        /// <param name="expected">The frames it should hold, oldest first.</param>
        /// <exception cref="AssertFailedException">Thrown when an element differs.</exception>
        private static void AssertStack(NDArray stacked, params NDArray[] expected) {
            using NDArray stack = np.stack(expected);
            using NDArray equal = stack == stacked;
            Assert.IsTrue(np.all(equal), $"{PyBuiltins.Repr(stacked)} is not {PyBuiltins.Repr(stack)}");
        }
    }
}
