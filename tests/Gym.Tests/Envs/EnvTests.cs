#nullable enable
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Threading.Tasks;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Legacy;
using Gym.Observations;
using Gym.Spaces;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Envs {
    /// <summary>
    ///     Pins the action conversion of the untyped <see cref="IEnv.Step"/> (formerly <c>Env&lt;TAction&gt;</c>'s), the
    ///     thread-pool <see cref="EnvExtensions.StepAsync{TObs, TAct}"/> that replaced the <c>DistributedScheduler</c>
    ///     (B-34), and the <see langword="long"/> surface of the legacy <see cref="VecEnv"/> behind
    ///     <see cref="LegacyEnvAdapter"/>. A minimal recording env stands in for an enum-typed env.
    /// </summary>
    [TestClass]
    public class EnvTests {
        /// <summary>An action enum backed by <see cref="long"/>: the former <c>(TAction)(object)action</c> could not unbox even a boxed int into it.</summary>
        private enum Move : long { Left = 0, Right = 1 }

        /// <summary>The smallest enum-typed env: it records the last typed action it was stepped with.</summary>
        private sealed class RecordingEnv : Env<NDArray, Move> {
            /// <summary>Two moves, drawn from a fixed seed so tests are reproducible.</summary>
            public RecordingEnv() {
                ActionSpace = new Discrete(2, seed: 0);
            }

            /// <summary>The typed action of the last <see cref="Step(Move)"/> call.</summary>
            public Move Last { get; private set; }

            /// <summary>Records <paramref name="action"/>.</summary>
            /// <param name="action">The converted action.</param>
            /// <returns>A transition with no observation and an empty info.</returns>
            public override StepResult<NDArray> Step(Move action) {
                Last = action;
                return new StepResult<NDArray>(null!, 0.0, false, false, new Info());
            }

            /// <summary>Not used by these tests.</summary>
            /// <param name="seed">Ignored.</param>
            /// <param name="options">Ignored.</param>
            /// <returns>Nothing; always throws.</returns>
            /// <exception cref="NotSupportedException">Always.</exception>
            public override ResetResult<NDArray> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) => throw new NotSupportedException();
        }

        /// <summary>
        ///     Trap 1: an untyped action of any integer width, a sampled 0-d int64 array, and a typed member all reach
        ///     <see cref="Env{TObs, TAct}.Step"/> as the member with that value.
        /// </summary>
        [TestMethod]
        public void Step_UntypedAction_ConvertsToTheEnumMember() {
            var env = new RecordingEnv();
            IEnv untyped = env;

            foreach (object action in new object[] { 1, 1L, (byte)1, NDArray.Scalar<long>(1L), Move.Right }) {
                env.Step(Move.Left);
                untyped.Step(action);
                Assert.AreEqual(Move.Right, env.Last, $"{action} ({action.GetType().Name})");
            }

            using var sampled = env.ActionSpace.Sample();
            untyped.Step(sampled);
            Assert.AreEqual((Move)(long)sampled, env.Last);
        }

        /// <summary>An untyped action that denotes no integer is refused with Python's <c>TypeError</c>, naming the value.</summary>
        [TestMethod]
        public void Step_UntypedAction_NotAnInteger_RaisesTypeError() {
            IEnv env = new RecordingEnv();

            var error = Assert.ThrowsExactly<TypeError>(() => env.Step("left"));

            StringAssert.Contains(error.Message, "'left' (<class 'str'>)");
        }

        /// <summary>
        ///     <see cref="EnvExtensions.StepAsync{TObs, TAct}"/> steps on the thread pool, typed and untyped; the removed
        ///     <c>DistributedScheduler</c> threw on machines with 32 or more logical CPUs before scheduling anything (B-34).
        /// </summary>
        /// <returns>The test's task.</returns>
        [TestMethod]
        public async Task StepAsync_StepsOnTheThreadPool() {
            var env = new RecordingEnv();

            await env.StepAsync(Move.Right);
            Assert.AreEqual(Move.Right, env.Last);
            await ((IEnv)env).StepAsync(0L);
            Assert.AreEqual(Move.Left, env.Last);
        }

#pragma warning disable CS0618 // The legacy API is what these tests pin.
        /// <summary>
        ///     <see cref="VecEnv.Step(long)"/> boxes the action as <see langword="long"/>, which reaches CartPole through
        ///     <see cref="LegacyEnvAdapter"/> and the untyped step's conversion. Seeds above <see cref="int.MaxValue"/> reach
        ///     every sub-env, as the seed of their next reset.
        /// </summary>
        [TestMethod]
        public void DummyVecEnv_StepsWithLongActionAndSeedsAboveInt32() {
            var vec = new DummyVecEnv(new LegacyEnvAdapter(new CartPoleEnv()));

            vec.Seed((1L << 31) + 5);
            vec.Seed(new[] { (long)uint.MaxValue });
            // CartPole's Reset hands out a copy of its state, so the initial observations are safe to release.
            foreach (var observation in vec.Reset())
                observation.Dispose();
            var steps = vec.Step(1L);

            Assert.AreEqual(1, steps.Length);
            Assert.IsNotNull(steps[0].Observation);
            Assert.AreEqual(uint.MaxValue, (long)((LegacyEnvAdapter)vec.Environments[0]).Env.NpRandomSeed);
        }
#pragma warning restore CS0618
    }
}
