using System;
using Gym.Environments;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
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
    ///     The tests dispose the initial observations they do not keep; <see cref="CartPoleEnv.Reset"/> hands out a copy of the
    ///     state. They never dispose a step's observation, which is still the env's own state array (B-13).
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
            var env = new CartPoleEnv(NullEnvViewer.Factory, LegacySeeding.RandomState(0));
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

        /// <summary>A value that is not an integer is rejected with <see cref="ArgumentException"/>, not an unboxing error.</summary>
        [TestMethod]
        public void Step_NonIntegerAction_ThrowsArgumentException()
        {
            var env = new CartPoleEnv(NullEnvViewer.Factory, LegacySeeding.RandomState(0));
            env.Reset().Dispose();

            Assert.ThrowsExactly<ArgumentException>(() => env.Step(0.5));
        }

        /// <summary>
        ///     Every step hands out a new array object. (The returned array is still the env's own state until the ownership
        ///     step removes B-13's aliasing.)
        /// </summary>
        [TestMethod]
        public void Step_ConsecutiveObservations_AreDistinctObjects()
        {
            var env = new CartPoleEnv(NullEnvViewer.Factory, LegacySeeding.RandomState(0));
            env.Reset().Dispose();

            var first = env.Step(0).Observation;
            var second = env.Step(1).Observation;

            Assert.IsFalse(ReferenceEquals(first, second));
        }

        /// <summary>
        ///     The initial observation is a member of the observation space. Membership checks shape and bounds only: the
        ///     observation is float64 while the space is float32 (B-13, P4), which Gymnasium's <c>np.can_cast</c> would reject.
        /// </summary>
        [TestMethod]
        public void ObservationSpace_ContainsTheResetObservation()
        {
            var env = new CartPoleEnv(NullEnvViewer.Factory, LegacySeeding.RandomState(0));

            Assert.IsTrue(env.ObservationSpace.Contains(env.Reset()));
        }

        /// <summary>Resets a fresh seed-0 env and returns the observation after one step with <paramref name="action"/>.</summary>
        /// <param name="action">The action to step with.</param>
        /// <returns>The observation of that step.</returns>
        private static NDArray StepOnce(object action)
        {
            var env = new CartPoleEnv(NullEnvViewer.Factory, LegacySeeding.RandomState(0));
            env.Reset().Dispose();
            return env.Step(action).Observation;
        }
    }
}
