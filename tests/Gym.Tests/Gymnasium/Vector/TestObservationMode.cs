#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Envs;
using Gym.Spaces;
using Gym.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Dict = Gym.Spaces.Dict;

namespace Gym.Tests.Gymnasium.Vector {
    /// <summary>
    ///     Gymnasium's observation-mode tests (<c>test_observation_mode.py</c>, class <c>TestVectorEnvObservationModes</c>),
    ///     run against <see cref="SyncVectorEnv"/> and <see cref="AsyncVectorEnv"/> with and without shared memory.
    /// </summary>
    /// <remarks>
    ///     Every sub-env gets a space of its own, where Gymnasium's lambdas may share one: a Gym.NET env owns and disposes its
    ///     spaces (D19), and <see cref="AsyncVectorEnv"/> disposes the envs it builds only to read their spaces.
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_observation_mode.py">tests/vector/test_observation_mode.py</see>.
    /// </remarks>
    [TestClass]
    public class TestObservationMode {
        /// <summary>Gymnasium's three vectorisers, by its ids.</summary>
        public static IEnumerable<object[]> VectorEnvFns => new[] { "SyncVectorEnv", "AsyncVectorEnv(shared_memory=True)", "AsyncVectorEnv(shared_memory=False)" }.Select(id => new object[] { id });

        /// <summary>The vectorisers with Gymnasium's two "different" modes: the string, and an explicit pair of spaces built per case.</summary>
        public static IEnumerable<object[]> VectorEnvFnsAndDifferentModes =>
            from id in VectorEnvFns.Select(row => (string)row[0]) from mode in new[] { "different", "pair" } select new object[] { id, mode };

        /// <summary>The vectorisers with every mode of <c>test_mixed_observation_spaces</c>.</summary>
        public static IEnumerable<object[]> VectorEnvFnsAndAllModes =>
            from id in VectorEnvFns.Select(row => (string)row[0]) from mode in new[] { "same", "different", "pair" } select new object[] { id, mode };

        /// <summary>An unknown mode is refused with Gymnasium's message.</summary>
        /// <param name="vectorEnvFn">The vectoriser.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_observation_mode.py#L28">tests/vector/test_observation_mode.py (test_invalid_observation_mode)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(VectorEnvFns))]
        public void TestInvalidObservationMode(string vectorEnvFn) {
            var error = Assert.ThrowsExactly<ValueError>(() =>
                Vectorise(vectorEnvFn, Enumerable.Range(0, 3).Select(_ => CreateEnv(() => new Box(0, 1, new Shape(5)))), "invalid"));
            StringAssert.Contains(error.Message, "Invalid `observation_mode`, expected: 'same' or 'different' or tuple of single and batch observation space, actual got invalid");
        }

        /// <summary><c>"same"</c> over spaces that differ is refused, and the message suggests <c>"different"</c>.</summary>
        /// <param name="vectorEnvFn">The vectoriser.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_observation_mode.py#L39">tests/vector/test_observation_mode.py (test_obs_mode_same_different_spaces)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(VectorEnvFns))]
        public void TestObsModeSameDifferentSpaces(string vectorEnvFn) {
            var error = Assert.ThrowsExactly<RuntimeError>(() =>
                Vectorise(vectorEnvFn, Enumerable.Range(1, 3).Select(i => CreateEnv(() => new Box(0, i, new Shape(2)))), "same"));
            StringAssert.Matches(error.Message, new System.Text.RegularExpressions.Regex("the sub-environments observation spaces are not equivalent. .*If this is intentional, use `observation_mode='different'` instead."));
        }

        /// <summary>
        ///     <c>"different"</c> (or the explicit pair) over boxes of different bounds batches them with
        ///     <c>batch_differing_spaces</c>, keeps the first as the single space, and resets and steps.
        /// </summary>
        /// <param name="vectorEnvFn">The vectoriser.</param>
        /// <param name="observationMode"><c>different</c>, or <c>pair</c> for Gymnasium's tuple of spaces.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_observation_mode.py#L48">tests/vector/test_observation_mode.py (test_obs_mode_different_different_spaces)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(VectorEnvFnsAndDifferentModes))]
        public void TestObsModeDifferentDifferentSpaces(string vectorEnvFn, string observationMode) {
            Space[] spaces = Enumerable.Range(1, 3).Select(i => (Space)new Box(0, i, new Shape(5))).ToArray();
            object mode = observationMode == "different"
                ? "different"
                : (new Box(0, np.repeat(np.arange(1, 4), 5).reshape(3, 5), new Shape(3, 5)), new Box(0, 1, new Shape(5)));
            // Each sub-env gets its own box (an env owns and disposes its spaces, D19); the list is the expectation only.
            using IVectorEnv envs = Vectorise(vectorEnvFn, Enumerable.Range(1, 3).Select(i => CreateEnv(() => new Box(0, i, new Shape(5)))), mode);
            using Space batched = VectorUtils.BatchDifferingSpaces(spaces);
            Assert.AreEqual(batched, ObjectSpace.Unview(envs.ObservationSpace));
            Assert.AreEqual(spaces[0], ObjectSpace.Unview(envs.SingleObservationSpace));

            envs.Reset();
            envs.Step(envs.ActionSpace.Sample());
            envs.Close();
        }

        /// <summary><c>"different"</c> (or the pair) over boxes of different shapes cannot batch them and is refused.</summary>
        /// <param name="vectorEnvFn">The vectoriser.</param>
        /// <param name="observationMode"><c>different</c>, or <c>pair</c> for Gymnasium's tuple of spaces.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_observation_mode.py#L75">tests/vector/test_observation_mode.py (test_obs_mode_different_different_shapes)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(VectorEnvFnsAndDifferentModes))]
        public void TestObsModeDifferentDifferentShapes(string vectorEnvFn, string observationMode) {
            object mode = observationMode == "different" ? "different" : (new Box(0, 4, new Shape(3, 5)), new Box(0, 4, new Shape(5)));
            Exception error = AssertThrowsAnyOf(() => Vectorise(vectorEnvFn, Enumerable.Range(0, 3).Select(i => CreateEnv(() => new Box(0, 1, new Shape(i + 1)))), mode),
                typeof(AssertionError), typeof(RuntimeError), typeof(ValueError));
            Assert.IsNotNull(error);
        }

        /// <summary>A box, a <c>Discrete</c> and a <c>Dict</c> together are refused in every mode.</summary>
        /// <param name="vectorEnvFn">The vectoriser.</param>
        /// <param name="observationMode"><c>same</c>, <c>different</c>, or <c>pair</c> for Gymnasium's tuple of spaces.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_observation_mode.py#L96">tests/vector/test_observation_mode.py (test_mixed_observation_spaces)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(VectorEnvFnsAndAllModes))]
        public void TestMixedObservationSpaces(string vectorEnvFn, string observationMode) {
            object mode = observationMode == "pair" ? (new Box(0, 4, new Shape(3, 5)), new Box(0, 4, new Shape(5))) : observationMode;
            Func<Space>[] spaces = {
                () => new Box(0, 1, new Shape(3)),
                () => new Discrete(5),
                () => Spaces.TestingSpaces.DictOf(("a", new Discrete(2)), ("b", new Box(0, 1, new Shape(2)))),
            };
            Exception error = AssertThrowsAnyOf(() => Vectorise(vectorEnvFn, spaces.Select(CreateEnv), mode), typeof(AssertionError), typeof(RuntimeError), typeof(TypeError));
            Assert.IsNotNull(error);
        }

        /// <summary><c>create_env(obs_space)</c>: a function making a <see cref="GenericTestEnv"/> with the space.</summary>
        /// <param name="space">Makes the space, called once per env made.</param>
        /// <returns>The function.</returns>
        private static Func<IEnv> CreateEnv(Func<Space> space) => () => new GenericTestEnv(observationSpace: space());

        /// <summary>Builds the vectoriser of the id over the functions with the observation mode.</summary>
        /// <param name="id">Gymnasium's vectoriser id.</param>
        /// <param name="envFns">The sub-envs' functions.</param>
        /// <param name="observationMode">The observation mode.</param>
        /// <returns>The vector env; the caller disposes it.</returns>
        private static IVectorEnv Vectorise(string id, IEnumerable<Func<IEnv>> envFns, object observationMode) => id switch {
            "SyncVectorEnv" => new SyncVectorEnv(envFns, observationMode: observationMode),
            "AsyncVectorEnv(shared_memory=True)" => new AsyncVectorEnv(envFns, sharedMemory: true, observationMode: observationMode),
            _ => new AsyncVectorEnv(envFns, sharedMemory: false, observationMode: observationMode),
        };

        /// <summary>pytest's <c>raises((A, B, C))</c>: the call throws an exception of one of the types (or a subclass).</summary>
        /// <param name="call">The call.</param>
        /// <param name="types">The accepted types.</param>
        /// <returns>The exception.</returns>
        /// <exception cref="AssertFailedException">Thrown when the call returns, or throws another type.</exception>
        private static Exception AssertThrowsAnyOf(Func<object?> call, params Type[] types) {
            try {
                (call() as IDisposable)?.Dispose();
            } catch (Exception e) when (types.Any(type => type.IsInstanceOfType(e))) {
                return e;
            }

            throw new AssertFailedException($"Expected one of {string.Join(", ", types.Select(t => t.Name))}, but nothing was thrown.");
        }
    }
}
