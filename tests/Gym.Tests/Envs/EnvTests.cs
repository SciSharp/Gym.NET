using System;
using Gym.Environments;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Observations;
using Gym.Spaces;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using SixLabors.ImageSharp;

namespace Gym.Tests.Envs {
    /// <summary>
    ///     Pins the action conversion of <see cref="Env{TAction}"/> and the <see langword="long"/> surface of
    ///     <see cref="VecEnv"/>. No env in the repository derives from <see cref="Env{TAction}"/>, so a minimal recording env
    ///     stands in for one.
    /// </summary>
    [TestClass]
    public class EnvTests {
        /// <summary>An action enum backed by <see cref="long"/>: the former <c>(TAction)(object)action</c> could not unbox even a boxed int into it.</summary>
        private enum Move : long { Left = 0, Right = 1 }

        /// <summary>The smallest <see cref="Env{TAction}"/>: it records the last typed action it was stepped with.</summary>
        private sealed class RecordingEnv : Env<Move> {
            /// <summary>The typed action of the last <see cref="Step(Move)"/> call.</summary>
            public Move Last { get; private set; }

            /// <summary>Two moves, drawn from a fixed seed so tests are reproducible.</summary>
            public RecordingEnv() {
                ActionSpace = new Discrete(2, seed: 0);
            }

            /// <summary>Records <paramref name="action"/>.</summary>
            /// <param name="action">The converted action.</param>
            /// <returns>An empty step.</returns>
            public override Step Step(Move action) {
                Last = action;
                return new Step();
            }

            /// <summary>Not used by these tests.</summary>
            /// <returns>Nothing; always throws.</returns>
            /// <exception cref="NotSupportedException">Always.</exception>
            public override NDArray Reset() => throw new NotSupportedException();

            /// <summary>Not used by these tests.</summary>
            /// <param name="mode">Ignored.</param>
            /// <returns>Nothing; always throws.</returns>
            /// <exception cref="NotSupportedException">Always.</exception>
            public override Image Render(string mode = "human") => throw new NotSupportedException();

            /// <summary>Nothing to release.</summary>
            public override void CloseEnvironment() { }

            /// <summary>Seeds the action space; the env itself draws nothing.</summary>
            /// <param name="seed">A NumPy seed in <c>[0, 2**32 - 1]</c>.</param>
            public override void Seed(long seed) => ActionSpace.Seed(seed);
        }

        /// <summary>
        ///     Trap 1: an untyped action of any integer width, a sampled 0-d int64 array, and a typed member all reach
        ///     <see cref="Env{TAction}.Step(TAction)"/> as the member with that value.
        /// </summary>
        [TestMethod]
        public void Step_UntypedAction_ConvertsToTheEnumMember()
        {
            var env = new RecordingEnv();

            foreach (object action in new object[] { 1, 1L, (byte)1, NDArray.Scalar<long>(1L), Move.Right })
            {
                env.Step(Move.Left);
                env.Step(action);
                Assert.AreEqual(Move.Right, env.Last, $"{action} ({action.GetType().Name})");
            }

            using var sampled = env.ActionSpace.Sample();
            env.Step((object) sampled);
            Assert.AreEqual((Move) (long) sampled, env.Last);
        }

        // Env.StepAsync shares ToAction with Step(object) but is not tested here: DistributedScheduler.Default throws on
        // machines with 32 or more logical CPUs (DistributedThread._validateProcessorAffinity overflows int computing
        // 1 << ProcessorCount), before any work is scheduled. That threading is deleted in P1 (B-34).

        /// <summary>
        ///     <see cref="VecEnv.Step(long)"/> boxes the action as <see langword="long"/>, which an env that casts
        ///     <c>(int)action</c> could not unbox; with <see cref="Discrete.ToInt64"/> it steps. Seeds above
        ///     <see cref="int.MaxValue"/> reach every sub-env.
        /// </summary>
        [TestMethod]
        public void DummyVecEnv_StepsWithLongActionAndSeedsAboveInt32()
        {
            var vec = new DummyVecEnv(new CartPoleEnv(NullEnvViewer.Factory, LegacySeeding.RandomState(0)));

            vec.Seed((1L << 31) + 5);
            vec.Seed(new[] { (long) uint.MaxValue });
            // CartPole's Reset hands out a copy of its state, so the initial observations are safe to release.
            foreach (var observation in vec.Reset())
                observation.Dispose();
            var steps = vec.Step(1L);

            Assert.AreEqual(1, steps.Length);
            Assert.IsNotNull(steps[0].Observation);
        }
    }
}
