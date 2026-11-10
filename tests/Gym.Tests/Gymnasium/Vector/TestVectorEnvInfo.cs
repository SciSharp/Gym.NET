#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Gym.Envs;
using Gym.Observations;
using Gym.Spaces;
using Gym.Utils;
using Gym.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Vector {
    /// <summary>Gymnasium's vector info tests (<c>test_vector_env_info.py</c>), run against the info merge of <see cref="VectorEnv{TObs, TAct}"/>.</summary>
    /// <remarks>
    ///     Gymnasium calls the bare <c>VectorEnv()</c>'s private <c>_add_info</c> after setting <c>num_envs</c>; the port does
    ///     the same through <see cref="AddInfoProbe"/>, a vector env exposing the protected <c>AddInfo</c>. <c>None</c> is
    ///     <see langword="null"/>, and object arrays are Gym.NET's <see langword="object"/>-dtype arrays of NumSharp, compared
    ///     with Gymnasium's <c>data_equivalence</c> (<see cref="EnvChecker.DataEquivalence"/>).
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_vector_env_info.py">tests/vector/test_vector_env_info.py</see>.
    /// </remarks>
    [TestClass]
    public class TestVectorEnvInfo {
        /// <summary>
        ///     <c>_add_info</c> batches each key across sub-envs with its <c>_key</c> mask: one sub-env, three alike, three with
        ///     different keys (zeros and <c>None</c> where a key is missing), and nested dicts batched recursively.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_vector_env_info.py#L16">tests/vector/test_vector_env_info.py (test_vector_add_info)</see>.</remarks>
        [TestMethod]
        public void TestVectorAddInfo() {
            // Test num-envs==1 then expand_dims(sub-env-info) == vector-infos
            var env = new AddInfoProbe(1);
            Info subEnvInfo = new() { ["a"] = 0L, ["b"] = 0.0, ["c"] = null, ["d"] = np.zeros(new Shape(2)), ["e"] = new Discrete(1) };
            Info vectorInfos = env.Merge(new Info(), subEnvInfo, 0);
            var expected = new Info {
                ["a"] = np.array(new[] { 0L }), ["b"] = np.array(new[] { 0.0 }), ["c"] = new object?[] { null }, ["d"] = np.zeros(new Shape(1, 2)),
                ["e"] = ObjectArray(new Discrete(1)), ["_a"] = Mask(true), ["_b"] = Mask(true), ["_c"] = Mask(true), ["_d"] = Mask(true), ["_e"] = Mask(true),
            };
            Assert.IsTrue(EnvChecker.DataEquivalence(vectorInfos, expected), $"one sub-env: {Describe(vectorInfos)}");

            // Thought: num-envs>1 then vector-infos should have the same structure as sub-env-info
            env = new AddInfoProbe(3);
            Info[] subEnvInfos = {
                new() { ["a"] = 0L, ["b"] = 0.0, ["c"] = null, ["d"] = np.zeros(new Shape(2)), ["e"] = new Discrete(1) },
                new() { ["a"] = 1L, ["b"] = 1.0, ["c"] = null, ["d"] = np.zeros(new Shape(2)), ["e"] = new Discrete(2) },
                new() { ["a"] = 2L, ["b"] = 2.0, ["c"] = null, ["d"] = np.zeros(new Shape(2)), ["e"] = new Discrete(3) },
            };
            vectorInfos = MergeAll(env, subEnvInfos);
            expected = new Info {
                ["a"] = np.array(new[] { 0L, 1L, 2L }), ["b"] = np.array(new[] { 0.0, 1.0, 2.0 }), ["c"] = ObjectArray(null, null, null),
                ["d"] = np.zeros(new Shape(3, 2)), ["e"] = ObjectArray(new Discrete(1), new Discrete(2), new Discrete(3)),
                ["_a"] = Mask(true, true, true), ["_b"] = Mask(true, true, true), ["_c"] = Mask(true, true, true), ["_d"] = Mask(true, true, true),
                ["_e"] = Mask(true, true, true),
            };
            Assert.IsTrue(EnvChecker.DataEquivalence(vectorInfos, expected), "three alike");

            // Test different structures of sub-infos
            subEnvInfos = new Info[] {
                new() { ["a"] = 1L, ["b"] = 1.0 },
                new() { ["c"] = null, ["d"] = np.zeros(new Shape(2)) },
                new() { ["e"] = new Discrete(3) },
            };
            vectorInfos = MergeAll(env, subEnvInfos);
            expected = new Info {
                ["a"] = np.array(new[] { 1L, 0L, 0L }), ["b"] = np.array(new[] { 1.0, 0.0, 0.0 }), ["c"] = ObjectArray(null, null, null),
                ["d"] = np.zeros(new Shape(3, 2)), ["e"] = ObjectArray(null, null, new Discrete(3)),
                ["_a"] = Mask(true, false, false), ["_b"] = Mask(true, false, false), ["_c"] = Mask(false, true, false), ["_d"] = Mask(false, true, false),
                ["_e"] = Mask(false, false, true),
            };
            Assert.IsTrue(EnvChecker.DataEquivalence(vectorInfos, expected), "different keys");

            // Test recursive structure
            subEnvInfos = new Info[] {
                new() { ["episode"] = new Info { ["a"] = 1L, ["b"] = 1.0 } },
                new() { ["episode"] = new Info { ["a"] = 2L, ["b"] = 2.0 }, ["a"] = 1L },
                new() { ["a"] = 2L },
            };
            vectorInfos = MergeAll(env, subEnvInfos);
            expected = new Info {
                ["episode"] = new Info {
                    ["a"] = np.array(new[] { 1L, 2L, 0L }), ["b"] = np.array(new[] { 1.0, 2.0, 0.0 }), ["_a"] = Mask(true, true, false), ["_b"] = Mask(true, true, false),
                },
                ["_episode"] = Mask(true, true, false), ["a"] = np.array(new[] { 0L, 1L, 2L }), ["_a"] = Mask(false, true, true),
            };
            Assert.IsTrue(EnvChecker.DataEquivalence(vectorInfos, expected), "nested");
        }

        /// <summary>A vectoriser of the id: Gymnasium's two, <c>AsyncVectorEnv</c> and <c>SyncVectorEnv</c>.</summary>
        public static IEnumerable<object[]> Vectorizers => new[] { new object[] { "AsyncVectorEnv" }, new object[] { "SyncVectorEnv" } };

        /// <summary>
        ///     The infos of a reset and a step through a vector env: keys missing in a sub-env get zeros and a false mask, and
        ///     array values are stacked.
        /// </summary>
        /// <param name="vectorizer">The vectoriser.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/test_vector_env_info.py#L143">tests/vector/test_vector_env_info.py (test_vector_return_info)</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Vectorizers))]
        public void TestVectorReturnInfo(string vectorizer) {
            Func<IEnv>[] envFns = {
                () => new ReturnInfoEnv(new Info { ["a"] = 1L }, new Info { ["c"] = np.array(new[] { 1L, 2L }) }),
                () => new ReturnInfoEnv(new Info { ["a"] = 2L, ["b"] = 3L }, new Info { ["c"] = np.array(new[] { 3L, 4L }) }),
            };
            using IVectorEnv vecEnv = vectorizer == "SyncVectorEnv" ? new SyncVectorEnv(envFns) : new AsyncVectorEnv(envFns);

            var resetExpectedInfos = new Info {
                ["a"] = np.array(new[] { 1L, 2L }), ["b"] = np.array(new[] { 0L, 3L }), ["_a"] = Mask(true, true), ["_b"] = Mask(false, true),
            };
            var stepExpectedInfos = new Info { ["c"] = np.array(new[,] { { 1L, 2L }, { 3L, 4L } }), ["_c"] = Mask(true, true) };

            (_, Info resetInfo) = vecEnv.Reset();
            Assert.IsTrue(EnvChecker.DataEquivalence(resetInfo, resetExpectedInfos), $"reset infos: {resetInfo}");
            (_, _, _, _, Info stepInfo) = vecEnv.Step(vecEnv.ActionSpace.Sample());
            Assert.IsTrue(EnvChecker.DataEquivalence(stepInfo, stepExpectedInfos), $"step infos: {stepInfo}");
        }

        /// <summary>Describes infos for a failure message: each key with its value's .NET type and Python repr.</summary>
        /// <param name="infos">The infos.</param>
        /// <returns>The text.</returns>
        private static string Describe(Info infos) =>
            string.Join("; ", infos.Select(pair => $"{pair.Key}: {pair.Value?.GetType().Name ?? "null"} {PyBuiltins.Repr(pair.Value)}"));

        /// <summary>Merges each sub-env's info in turn, as Gymnasium's loop does.</summary>
        /// <param name="env">The probe.</param>
        /// <param name="infos">The sub-envs' infos.</param>
        /// <returns>The vector infos.</returns>
        private static Info MergeAll(AddInfoProbe env, Info[] infos) {
            var vectorInfos = new Info();
            for (int i = 0; i < infos.Length; i++)
                vectorInfos = env.Merge(vectorInfos, infos[i], i);
            return vectorInfos;
        }

        /// <summary>A boolean mask, <c>np.array([...])</c> of bools.</summary>
        /// <param name="values">The flags.</param>
        /// <returns>A new array.</returns>
        private static NDArray Mask(params bool[] values) => np.array(values);

        /// <summary><c>np.array([...], dtype=object)</c>; a single <c>None</c> needs <c>new object?[] { null }</c>, as C# binds a lone <see langword="null"/> to the array itself.</summary>
        /// <param name="values">The items.</param>
        /// <returns>A new object array.</returns>
        private static object?[] ObjectArray(params object?[] values) => values;

        /// <summary>
        ///     <c>VectorEnv()</c> with <c>num_envs</c> set: a vector env of no sub-envs that exposes the protected info merge
        ///     (<c>_add_info</c>).
        /// </summary>
        private sealed class AddInfoProbe : VectorEnv<NDArray, NDArray> {
            /// <summary>Sets <c>num_envs</c>.</summary>
            /// <param name="numEnvs">The batches' length.</param>
            public AddInfoProbe(int numEnvs) => NumEnvs = numEnvs;

            /// <summary><c>env._add_info(vector_infos, env_info, env_num)</c>.</summary>
            /// <param name="vectorInfos">The infos so far.</param>
            /// <param name="envInfo">The sub-env's info.</param>
            /// <param name="envNum">The sub-env.</param>
            /// <returns>The merged infos.</returns>
            public Info Merge(Info vectorInfos, Info envInfo, int envNum) => AddInfo(vectorInfos, envInfo, envNum);

            /// <summary>Not used: Gymnasium's bare <c>VectorEnv</c> raises <c>NotImplementedError</c>.</summary>
            /// <param name="seed">Unused.</param>
            /// <param name="options">Unused.</param>
            /// <returns>Never.</returns>
            /// <exception cref="NotSupportedException">Always.</exception>
            public override VectorResetResult<NDArray> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) => throw new NotSupportedException();

            /// <summary>Not used: Gymnasium's bare <c>VectorEnv</c> raises <c>NotImplementedError</c>.</summary>
            /// <param name="actions">Unused.</param>
            /// <returns>Never.</returns>
            /// <exception cref="NotSupportedException">Always.</exception>
            public override VectorStepResult<NDArray> Step(NDArray actions) => throw new NotSupportedException();
        }

        /// <summary>
        ///     Gymnasium's <c>ReturnInfoEnv</c>: <c>Box(0, 1)</c> spaces, a reset returning the first info, and a step that
        ///     terminates with the second.
        /// </summary>
        private sealed class ReturnInfoEnv : Env<object?, object?> {
            /// <summary>The reset's and the step's infos.</summary>
            private readonly Info[] _infos;

            /// <summary>An env returning the given infos.</summary>
            /// <param name="resetInfo">The reset's info.</param>
            /// <param name="stepInfo">The step's info.</param>
            public ReturnInfoEnv(Info resetInfo, Info stepInfo) {
                ((IEnv)this).ObservationSpace = new Box(0, 1);
                ((IEnv)this).ActionSpace = new Box(0, 1);
                _infos = new[] { resetInfo, stepInfo };
            }

            /// <summary>A sampled observation and the first info.</summary>
            /// <param name="seed">Unused, as in Gymnasium's (it does not seed).</param>
            /// <param name="options">Unused.</param>
            /// <returns>The reset.</returns>
            public override ResetResult<object?> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) =>
                new(ObservationSpace.Sample(), _infos[0]);

            /// <summary>A sampled observation, reward 0, terminated, and the second info.</summary>
            /// <param name="action">Unused.</param>
            /// <returns>The step.</returns>
            public override StepResult<object?> Step(object? action) => new(ObservationSpace.Sample(), 0, true, false, _infos[1]);
        }
    }
}
