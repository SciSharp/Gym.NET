#nullable enable
using System;
using System.Linq;
using System.Threading;
using Gym.Environments;
using Gym.Environments.Envs.Aether;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
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
    ///     that no two unseeded instances share a generator, is pinned by <c>EntropySeedingTests</c>. Each worker runs on a
    ///     dedicated <see cref="Thread"/>, never a thread-pool task: a <see cref="Barrier"/> of N releases only once N
    ///     threads run at the same time, which <c>Task.Run</c> does not promise (a pool saturated by a parallel test run
    ///     injects threads slowly, so the barrier would stall the test instead of making its threads overlap).
    /// </remarks>
    [TestClass]
    public class ConcurrencyTests {
        /// <summary>
        ///     How long <see cref="RunOnThreads"/> waits for each thread before failing the test. The runs take about a
        ///     second; the margin is for slow CI machines, and the limit turns a stalled barrier into a failure instead of a
        ///     test run that never ends.
        /// </summary>
        private static readonly TimeSpan ThreadTimeout = TimeSpan.FromMinutes(2);

        /// <summary>
        ///     Two unseeded LunarLanders and two unseeded CartPoles step on four threads that start together, sampling their
        ///     action and observation spaces every step.
        /// </summary>
        [TestMethod]
        public void UnseededEnvs_SteppingInParallel_DoNotThrow()
        {
            using var start = new Barrier(4);
            RunOnThreads(
                () => RunLunarLander(start),
                () => RunLunarLander(start),
                () => RunCartPole(start),
                () => RunCartPole(start));
        }

        /// <summary>Two unseeded spaces are sampled 250,000 times each on two threads that start together.</summary>
        [TestMethod]
        public void UnseededSpaces_SampledInParallel_DoNotThrow()
        {
            using var start = new Barrier(2);
            RunOnThreads(
                () => SampleMany(new Discrete(1000), start),
                () => SampleMany(new Discrete(1000), start));
        }

        /// <summary>
        ///     Runs each body on a dedicated thread of its own, waits for all of them, and fails with whatever any body threw.
        /// </summary>
        /// <param name="bodies">
        ///     The work of each thread. Every body gets its own thread, so a <see cref="Barrier"/> sized to the number of
        ///     bodies is always reached.
        /// </param>
        /// <exception cref="AggregateException">
        ///     Thrown when any body threw, holding every body's exception, or when a thread outlived
        ///     <see cref="ThreadTimeout"/>, holding a <see cref="TimeoutException"/> and the failures recorded so far (a body
        ///     that fails before its barrier leaves the other threads waiting there, so the timeout reports its exception).
        /// </exception>
        private static void RunOnThreads(params Action[] bodies)
        {
            var failures = new Exception?[bodies.Length];
            var threads = new Thread[bodies.Length];
            for (int i = 0; i < bodies.Length; i++)
            {
                int index = i;
                threads[i] = new Thread(() => {
                    // An exception escaping a raw thread would terminate the test host, so each thread records its own
                    // failure in its own slot (no two threads share one, so no lock is needed) for the test thread to rethrow.
                    try {
                        bodies[index]();
                    } catch (Exception exception) {
                        failures[index] = exception;
                    }
                }) {
                    // Background, so a thread stuck at its barrier after a timeout cannot keep the test host alive.
                    IsBackground = true,
                    Name = $"{nameof(ConcurrencyTests)} worker {index}",
                };
            }

            foreach (var thread in threads)
                thread.Start();
            foreach (var thread in threads)
            {
                // Join is a full memory barrier, so the failures the thread recorded are visible once it returns true.
                if (!thread.Join(ThreadTimeout))
                    throw new AggregateException(
                        new Exception?[] { new TimeoutException($"{thread.Name} was still running after {ThreadTimeout}.") }
                            .Concat(failures).OfType<Exception>());
            }

            var thrown = failures.OfType<Exception>().ToArray();
            if (thrown.Length > 0)
                throw new AggregateException(thrown);
        }

        /// <summary>Runs 500 steps of an unseeded discrete LunarLander, resetting whenever an episode ends.</summary>
        /// <param name="start">Released once every thread is ready, so the threads overlap.</param>
        private static void RunLunarLander(Barrier start)
        {
            var env = new LunarLanderEnv(NullEnvViewer.Factory);
            start.SignalAndWait();
            // LunarLander's observations are fresh arrays every step, so this loop owns and releases them.
            env.Reset().Observation.Dispose();
            for (int i = 0; i < 500; i++)
            {
                env.ObservationSpace.Sample().Dispose();
                using NDArray action = env.ActionSpace.Sample();
                var step = env.Step(action);
                step.Observation.Dispose();
                if (step.Terminated || step.Truncated)
                    env.Reset().Observation.Dispose();
            }
        }

        /// <summary>Runs 2,000 steps of an unseeded CartPole, resetting whenever an episode ends.</summary>
        /// <param name="start">Released once every thread is ready, so the threads overlap.</param>
        private static void RunCartPole(Barrier start)
        {
            IEnv env = new CartPoleEnv();
            start.SignalAndWait();
            // Every observation is a fresh float32 copy of the state (B-13 is fixed), so the loop releases them all.
            ((NDArray)env.Reset().Observation!).Dispose();
            for (int i = 0; i < 2000; i++)
            {
                env.ObservationSpace.Sample().Dispose();
                using NDArray action = env.ActionSpace.Sample();
                var step = env.Step(action);
                ((NDArray)step.Observation!).Dispose();
                if (step.Terminated || step.Truncated)
                    ((NDArray)env.Reset().Observation!).Dispose();
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
