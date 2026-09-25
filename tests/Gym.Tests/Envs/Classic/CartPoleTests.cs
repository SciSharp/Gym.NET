using System;
using System.Linq;
using Gym.Environments;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Envs.Classic {
    /// <summary>
    ///     Headless contract tests of <see cref="CartPoleEnv"/> after the NumSharp master migration: it steps with what its
    ///     own action space samples (trap 1), accepts actions of any integer width, and its observations are members of its
    ///     observation space.
    /// </summary>
    /// <remarks>
    ///     The tests dispose the initial observations they do not keep. Every observation is a fresh float32 copy of the state
    ///     (the port fixed B-13's aliasing), so disposing one never touches the env.
    /// </remarks>
    [TestClass]
    public class CartPoleTests {
        /// <summary>
        ///     Stepping with <c>ActionSpace.Sample()</c> works. The sample is a 0-d int64 array, which the former
        ///     <c>(int)action</c> could not unbox (<see cref="InvalidCastException"/>).
        /// </summary>
        [TestMethod]
        public void Step_WithSampledActions_RunsEpisodes()
        {
            var env = new CartPoleEnv(NullEnvViewer.Factory, Seeding.NpRandom(0).Rng);
            env.ActionSpace.Seed(0);
            env.Reset().Dispose();

            for (int i = 0; i < 200; i++)
            {
                var (_, _, done, _) = env.Step(env.ActionSpace.Sample());
                if (done)
                    env.Reset().Dispose();
            }
        }

        /// <summary>
        ///     The same push results from the same action value whatever boxed it: <see langword="int"/>, <see langword="long"/>
        ///     (what <see cref="VecEnv.Step(long)"/> sends), <see langword="byte"/>, a 0-d array, or <see langword="true"/>.
        /// </summary>
        [TestMethod]
        public void Step_ActionOfAnyIntegerWidth_GivesTheSameTransition()
        {
            var expected = StepOnce(1);
            foreach (object action in new object[] { 1L, (byte)1, (short)1, 1u, NDArray.Scalar<long>(1L), true })
                Assert.IsTrue(np.array_equal(expected, StepOnce(action)), $"Action {action} ({action.GetType().Name}) must push right like 1.");
        }

        /// <summary>
        ///     A value that is not an element of <c>Discrete(2)</c> is rejected as Gymnasium's <c>assert</c> rejects it: an
        ///     <see cref="AssertionError"/> (in Release builds too, B-14) with Python's rendering of the action and its type.
        /// </summary>
        [TestMethod]
        public void Step_ActionOutsideTheActionSpace_ThrowsGymnasiumsAssertionError()
        {
            var env = new CartPoleEnv(NullEnvViewer.Factory, Seeding.NpRandom(0).Rng);
            env.Reset().Dispose();

            Assert.AreEqual("0.5 (<class 'float'>) invalid", Assert.ThrowsExactly<AssertionError>(() => env.Step(0.5)).Message);
            Assert.AreEqual("2 (<class 'int'>) invalid", Assert.ThrowsExactly<AssertionError>(() => env.Step(2L)).Message);
            Assert.AreEqual("np.int64(5) (<class 'numpy.int64'>) invalid", Assert.ThrowsExactly<AssertionError>(() => env.Step(NDArray.Scalar<long>(5L))).Message);
        }

        /// <summary>Stepping before the first reset raises Gymnasium's <c>Call reset before using step method.</c></summary>
        [TestMethod]
        public void Step_BeforeReset_ThrowsGymnasiumsAssertionError()
        {
            var env = new CartPoleEnv(NullEnvViewer.Factory, Seeding.NpRandom(0).Rng);

            Assert.AreEqual("Call reset before using step method.", Assert.ThrowsExactly<AssertionError>(() => env.Step(1)).Message);
        }

        /// <summary>
        ///     Observations are float32 copies of the float64 <see cref="CartPoleEnv.State"/>, as Gymnasium's
        ///     <c>np.array(self.state, dtype=np.float32)</c>: writing to one leaves the env alone (B-13).
        /// </summary>
        [TestMethod]
        public void Observations_AreFloat32Copies_OfTheFloat64State()
        {
            var env = new CartPoleEnv(NullEnvViewer.Factory, Seeding.NpRandom(0).Rng);
            using NDArray observation = env.Reset();
            double[] state = (double[]) env.State.Clone();

            Assert.AreEqual(np.float32, observation.dtype);
            CollectionAssert.AreEqual(state.Select(v => (float) v).ToArray(), observation.ToArray<float>());
            observation.SetSingle(123f, 0);
            CollectionAssert.AreEqual(state, env.State, "Writing to the observation must not move the env.");
        }

        /// <summary>
        ///     Stepping after termination keeps going with reward 0.0, and warns exactly once (Gymnasium's
        ///     <c>logger.warn</c>), captured here as Python's <c>catch_warnings(record=True)</c> would capture it.
        /// </summary>
        [TestMethod]
        public void Step_AfterTermination_WarnsOnceAndRewardsZero()
        {
            var env = new CartPoleEnv(NullEnvViewer.Factory, Seeding.NpRandom(3).Rng);
            env.Reset().Dispose();
            while (!env.Step(1).Done) { }

            using WarningCatcher catcher = Warnings.Catch();
            var rewards = Enumerable.Range(0, 3).Select(_ => env.Step(1).Reward).ToArray();

            CollectionAssert.AreEqual(new[] { 0f, 0f, 0f }, rewards);
            Assert.AreEqual(1, catcher.Records.Count);
            StringAssert.StartsWith(catcher.Records[0].Message, "\u001b[33mWARN: You are calling 'step()' even though this environment has already returned terminated = True.");
            Assert.AreEqual(3, env.StepsBeyondTerminated);
        }

        /// <summary>
        ///     Every step hands out a new array object. (The returned array is still the env's own state until the ownership
        ///     step removes B-13's aliasing.)
        /// </summary>
        [TestMethod]
        public void Step_ConsecutiveObservations_AreDistinctObjects()
        {
            var env = new CartPoleEnv(NullEnvViewer.Factory, Seeding.NpRandom(0).Rng);
            env.Reset().Dispose();

            var first = env.Step(0).Observation;
            var second = env.Step(1).Observation;

            Assert.IsFalse(ReferenceEquals(first, second));
        }

        /// <summary>The initial observation, float32 like the space, is a member of the observation space.</summary>
        [TestMethod]
        public void ObservationSpace_ContainsTheResetObservation()
        {
            var env = new CartPoleEnv(NullEnvViewer.Factory, Seeding.NpRandom(0).Rng);

            Assert.IsTrue(env.ObservationSpace.Contains(env.Reset()));
        }

        /// <summary>Resets a fresh seed-0 env and returns the observation after one step with <paramref name="action"/>.</summary>
        /// <param name="action">The action to step with.</param>
        /// <returns>The observation of that step.</returns>
        private static NDArray StepOnce(object action)
        {
            var env = new CartPoleEnv(NullEnvViewer.Factory, Seeding.NpRandom(0).Rng);
            env.Reset().Dispose();
            return env.Step(action).Observation;
        }
    }
}
