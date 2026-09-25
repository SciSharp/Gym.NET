using System;
using System.Threading;
using System.Threading.Tasks;
using Gym.Environments;
using Gym.Environments.Envs.Aether;
using Gym.Environments.Envs.Classic;
using Gym.Spaces;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Envs {
    /// <summary>
    ///     Envs and spaces created with default generators step and sample on parallel threads without exceptions. Before the
    ///     per-instance generators, every unseeded space and LunarLander shared the process-wide <c>np.random</c>, and two
    ///     threads drawing from one unsynchronized generator (NumSharp's MT19937 then) threw
    ///     <see cref="IndexOutOfRangeException"/> (trap 5, measured with 2 threads x 2,000,000 draws).
    /// </summary>
    /// <remarks>
    ///     A race only fails when it happens to interleave badly, so these tests are smoke tests. The deterministic guarantee,
    ///     that no two unseeded instances share a generator, is pinned by <c>EntropySeedingTests</c>.
    /// </remarks>
    [TestClass]
    public class ConcurrencyTests {
        /// <summary>
        ///     Two unseeded LunarLanders and two unseeded CartPoles step on four threads that start together, sampling their
        ///     action and observation spaces every step.
        /// </summary>
        [TestMethod]
        public void UnseededEnvs_SteppingInParallel_DoNotThrow()
        {
            using var start = new Barrier(4);
            var runs = new[] {
                Task.Run(() => RunLunarLander(start)),
                Task.Run(() => RunLunarLander(start)),
                Task.Run(() => RunCartPole(start)),
                Task.Run(() => RunCartPole(start)),
            };

            Task.WaitAll(runs);
        }

        /// <summary>Two unseeded spaces are sampled 250,000 times each on two threads that start together.</summary>
        [TestMethod]
        public void UnseededSpaces_SampledInParallel_DoNotThrow()
        {
            using var start = new Barrier(2);
            var runs = new[] {
                Task.Run(() => SampleMany(new Discrete(1000), start)),
                Task.Run(() => SampleMany(new Discrete(1000), start)),
            };

            Task.WaitAll(runs);
        }

        /// <summary>Runs 500 steps of an unseeded discrete LunarLander, resetting whenever an episode ends.</summary>
        /// <param name="start">Released once every thread is ready, so the threads overlap.</param>
        private static void RunLunarLander(Barrier start)
        {
            var env = new LunarLanderEnv(NullEnvViewer.Factory);
            start.SignalAndWait();
            // LunarLander's observations are fresh arrays every step, so this loop owns and releases them.
            env.Reset().Dispose();
            for (int i = 0; i < 500; i++)
            {
                env.ObservationSpace.Sample().Dispose();
                using NDArray action = env.ActionSpace.Sample();
                var step = env.Step(action);
                step.Observation.Dispose();
                if (step.Done)
                    env.Reset().Dispose();
            }
        }

        /// <summary>Runs 2,000 steps of an unseeded CartPole, resetting whenever an episode ends.</summary>
        /// <param name="start">Released once every thread is ready, so the threads overlap.</param>
        private static void RunCartPole(Barrier start)
        {
            var env = new CartPoleEnv(NullEnvViewer.Factory);
            start.SignalAndWait();
            // Reset hands out a copy; a step's observation is still the env's own state (B-13), so it is not disposed.
            env.Reset().Dispose();
            for (int i = 0; i < 2000; i++)
            {
                env.ObservationSpace.Sample().Dispose();
                using NDArray action = env.ActionSpace.Sample();
                if (env.Step(action).Done)
                    env.Reset().Dispose();
            }
        }

        /// <summary>Samples <paramref name="space"/> 250,000 times.</summary>
        /// <param name="space">An unseeded space owned by the calling thread.</param>
        /// <param name="start">Released once every thread is ready, so the threads overlap.</param>
        private static void SampleMany(Discrete space, Barrier start)
        {
            start.SignalAndWait();
            for (int i = 0; i < 250_000; i++)
                space.Sample().Dispose();
        }
    }
}
