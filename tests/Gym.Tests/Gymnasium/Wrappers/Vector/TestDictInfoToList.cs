#nullable enable
using System.Collections.Generic;
using System.Numerics;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Spaces;
using Gym.Utils;
using Gym.Vector;
using Gym.Wrappers.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Wrappers.Vector {
    /// <summary>Gymnasium's <c>DictInfoToList</c> tests (<c>test_dict_info_to_list.py</c>), run against <see cref="DictInfoToList{TObs, TAct}"/>.</summary>
    /// <remarks>
    ///     The list comes from the wrapper's own <c>Reset</c>, which hides the vector env's (see
    ///     <see cref="DictInfoToList{TObs, TAct}"/>). NumPy's scalars of the split arrays are 0-d arrays here (DR-08), so
    ///     the expected entries are <see cref="NDArray.Scalar{T}(T)"/> values of the arrays' dtypes.
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_dict_info_to_list.py">tests/wrappers/vector/test_dict_info_to_list.py</see>.
    /// </remarks>
    [TestClass]
    public class TestDictInfoToList {
        /// <summary>The wrapper takes a vector env and refuses a single env with a <see cref="TypeError"/>.</summary>
        /// <remarks>
        ///     Both are built through <see cref="PyKwargs.Construct"/>, as Python's call checks the argument at run time; a
        ///     typed C# call would not compile with a single env.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_dict_info_to_list.py#L18">tests/wrappers/vector/test_dict_info_to_list.py (test_usage_in_vector_env)</see>.
        /// </remarks>
        [TestMethod]
        public void TestUsageInVectorEnv() {
            const string envId = "CartPole-v1";
            const int numEnvs = 3;
            using IEnv env = Make(envId, disableEnvChecker: true);
            using IVectorEnv vectorEnv = MakeVec(envId, numEnvs);

            PyKwargs.Construct(typeof(DictInfoToList<,>), new Info { ["env"] = vectorEnv });

            Assert.ThrowsExactly<TypeError>(() => PyKwargs.Construct(typeof(DictInfoToList<,>), new Info { ["env"] = env }));
        }

        /// <summary>
        ///     Batched infos split into one info per sub-env: one sub-env, three, masks keeping a key for some sub-envs only,
        ///     a nested dict with and without its own masks.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_dict_info_to_list.py#L43">tests/wrappers/vector/test_dict_info_to_list.py (test_update_info)</see>.</remarks>
        [TestMethod]
        public void TestUpdateInfo() {
            var unwrapped = new ResetOptionAsInfo();
            using var env = new DictInfoToList<object?, object?>(unwrapped);

            // Test num-envs==1 then expand_dims(sub-env-info) == vector-infos
            unwrapped.SetNumEnvs(1);

            var vectorInfos = new Info {
                ["a"] = np.array(new long[] { 0 }),
                ["b"] = np.array(new[] { 0.0 }),
                ["c"] = new object?[] { null },
                ["d"] = np.zeros(new Shape(1, 2)),
                ["e"] = new object?[] { new Discrete(1) },
                ["_a"] = np.array(new[] { true }),
                ["_b"] = np.array(new[] { true }),
                ["_c"] = np.array(new[] { true }),
                ["_d"] = np.array(new[] { true }),
                ["_e"] = np.array(new[] { true }),
            };
            var (_, listInfo) = env.Reset(options: vectorInfos);

            var expectedListInfo = new List<Info> {
                new() { ["a"] = NDArray.Scalar(0L), ["b"] = NDArray.Scalar(0.0), ["c"] = null, ["d"] = np.zeros(new Shape(2)), ["e"] = new Discrete(1) },
            };
            Assert.IsTrue(EnvChecker.DataEquivalence(listInfo, expectedListInfo));

            // Thought: num-envs>1 then vector-infos should have the same structure as sub-env-info
            unwrapped.SetNumEnvs(3);

            vectorInfos = new Info {
                ["a"] = np.array(new long[] { 0, 1, 2 }),
                ["b"] = np.array(new[] { 0.0, 1.0, 2.0 }),
                ["c"] = new object?[] { null, null, null },
                ["d"] = np.zeros(new Shape(3, 2)),
                ["e"] = new object?[] { new Discrete(1), new Discrete(2), new Discrete(3) },
                ["_a"] = np.array(new[] { true, true, true }),
                ["_b"] = np.array(new[] { true, true, true }),
                ["_c"] = np.array(new[] { true, true, true }),
                ["_d"] = np.array(new[] { true, true, true }),
                ["_e"] = np.array(new[] { true, true, true }),
            };
            (_, listInfo) = env.Reset(options: vectorInfos);
            expectedListInfo = new List<Info> {
                new() { ["a"] = NDArray.Scalar(0L), ["b"] = NDArray.Scalar(0.0), ["c"] = null, ["d"] = np.zeros(new Shape(2)), ["e"] = new Discrete(1) },
                new() { ["a"] = NDArray.Scalar(1L), ["b"] = NDArray.Scalar(1.0), ["c"] = null, ["d"] = np.zeros(new Shape(2)), ["e"] = new Discrete(2) },
                new() { ["a"] = NDArray.Scalar(2L), ["b"] = NDArray.Scalar(2.0), ["c"] = null, ["d"] = np.zeros(new Shape(2)), ["e"] = new Discrete(3) },
            };

            CollectionAssert.AreEqual(new List<string>(expectedListInfo[0].Keys), new List<string>(listInfo[0].Keys));
            foreach (string key in listInfo[0].Keys)
                Assert.IsTrue(EnvChecker.DataEquivalence(listInfo[0][key], expectedListInfo[0][key]), key);
            Assert.IsTrue(EnvChecker.DataEquivalence(listInfo, expectedListInfo));

            // Test different structures of sub-infos
            unwrapped.SetNumEnvs(3);

            vectorInfos = new Info {
                ["a"] = np.array(new long[] { 1, 0, 0 }),
                ["_a"] = np.array(new[] { true, false, false }),
                ["b"] = np.array(new[] { 1.0, 0.0, 0.0 }),
                ["_b"] = np.array(new[] { true, false, false }),
                ["c"] = new object?[] { null, null, null },
                ["_c"] = np.array(new[] { false, true, false }),
                ["_d"] = np.array(new[] { false, true, false }),
                ["d"] = np.zeros(new Shape(3, 2)),
                ["e"] = new object?[] { null, null, new Discrete(3) },
                ["_e"] = np.array(new[] { false, false, true }),
            };
            (_, listInfo) = env.Reset(options: vectorInfos);
            expectedListInfo = new List<Info> {
                new() { ["a"] = NDArray.Scalar(1L), ["b"] = NDArray.Scalar(1.0) },
                new() { ["c"] = null, ["d"] = np.zeros(new Shape(2)) },
                new() { ["e"] = new Discrete(3) },
            };
            Assert.IsTrue(EnvChecker.DataEquivalence(listInfo, expectedListInfo));

            // Test recursive structure
            unwrapped.SetNumEnvs(3);

            vectorInfos = new Info {
                ["episode"] = new Info {
                    ["a"] = np.array(new long[] { 1, 2, 0 }),
                    ["b"] = np.array(new[] { 1.0, 2.0, 0.0 }),
                    ["_a"] = np.array(new[] { true, true, false }),
                    ["_b"] = np.array(new[] { true, true, false }),
                },
                ["_episode"] = np.array(new[] { true, true, false }),
                ["a"] = np.array(new long[] { 0, 1, 2 }),
                ["_a"] = np.array(new[] { false, true, true }),
            };
            (_, listInfo) = env.Reset(options: vectorInfos);
            expectedListInfo = new List<Info> {
                new() { ["episode"] = new Info { ["a"] = NDArray.Scalar(1L), ["b"] = NDArray.Scalar(1.0) } },
                new() { ["episode"] = new Info { ["a"] = NDArray.Scalar(2L), ["b"] = NDArray.Scalar(2.0) }, ["a"] = NDArray.Scalar(1L) },
                new() { ["a"] = NDArray.Scalar(2L) },
            };
            Assert.IsTrue(EnvChecker.DataEquivalence(listInfo, expectedListInfo));

            // Test without binary array (can happen with custom vector environment)
            vectorInfos = new Info {
                ["episode"] = new Info {
                    ["a"] = np.array(new long[] { 1, 2, 0 }),
                    ["b"] = np.array(new[] { 1.0, 2.0, 0.0 }),
                },
                ["a"] = np.array(new long[] { 0, 1, 2 }),
            };
            (_, listInfo) = env.Reset(options: vectorInfos);
            expectedListInfo = new List<Info> {
                new() { ["episode"] = new Info { ["a"] = NDArray.Scalar(1L), ["b"] = NDArray.Scalar(1.0) }, ["a"] = NDArray.Scalar(0L) },
                new() { ["episode"] = new Info { ["a"] = NDArray.Scalar(2L), ["b"] = NDArray.Scalar(2.0) }, ["a"] = NDArray.Scalar(1L) },
                new() { ["episode"] = new Info { ["a"] = NDArray.Scalar(0L), ["b"] = NDArray.Scalar(0.0) }, ["a"] = NDArray.Scalar(2L) },
            };
            Assert.IsTrue(EnvChecker.DataEquivalence(listInfo, expectedListInfo));
        }

        /// <summary>A value or mask whose length is not the number of sub-envs fails the conversion's assertion, nested or not.</summary>
        /// <param name="variant">The case, in Gymnasium's order.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_dict_info_to_list.py#L194">tests/wrappers/vector/test_dict_info_to_list.py (test_errors)</see>.</remarks>
        [TestMethod]
        [DataRow(0)]
        [DataRow(1)]
        [DataRow(2)]
        [DataRow(3)]
        [DataRow(4)]
        [DataRow(5)]
        public void TestErrors(int variant) {
            Info vectorInfos = variant switch {
                0 => new Info { ["a"] = np.array(new long[] { 1, 2 }), ["_a"] = np.array(new[] { true, true, false }) },
                1 => new Info { ["a"] = np.array(new long[] { 1, 2 }) },
                2 => new Info {
                    ["episode"] = new Info { ["a"] = np.array(new long[] { 1, 2 }), ["_a"] = np.array(new[] { true, true, false }) },
                    ["a"] = np.array(new long[] { 0, 1, 2 }),
                    ["_a"] = np.array(new[] { false, true, true }),
                },
                3 => new Info {
                    ["episode"] = new Info { ["a"] = np.array(new long[] { 1, 2 }) },
                    ["a"] = np.array(new long[] { 0, 1, 2 }),
                },
                4 => new Info {
                    ["episode"] = new Info { ["a"] = np.array(new long[] { 1, 2, 0 }), ["_a"] = np.array(new[] { true, true, false }) },
                    ["a"] = np.array(new long[] { 1, 2 }),
                    ["_a"] = np.array(new[] { true, true }),
                },
                _ => new Info {
                    ["episode"] = new Info { ["a"] = np.array(new long[] { 1, 2, 0 }) },
                    ["a"] = np.array(new long[] { 1, 2 }),
                },
            };
            var unwrapped = new ResetOptionAsInfo();
            using var env = new DictInfoToList<object?, object?>(unwrapped);
            unwrapped.SetNumEnvs(3);

            Assert.ThrowsExactly<AssertionError>(() => env.Reset(options: vectorInfos));
        }

        /// <summary>A minimal vector env whose reset returns its options as the infos, to test the conversion (Gymnasium's <c>ResetOptionAsInfo</c>).</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_dict_info_to_list.py#L29">tests/wrappers/vector/test_dict_info_to_list.py (ResetOptionAsInfo)</see>.</remarks>
        private sealed class ResetOptionAsInfo : VectorEnv<object?, object?> {
            /// <summary>Sets <c>num_envs</c>, as the test assigns <c>env.unwrapped.num_envs</c>.</summary>
            /// <param name="numEnvs">The number of sub-envs the conversion expects.</param>
            public void SetNumEnvs(int numEnvs) => NumEnvs = numEnvs;

            /// <summary>Returns no observations and the options as the infos.</summary>
            /// <param name="seed">Ignored.</param>
            /// <param name="options">The infos to return; copied into an <see cref="Info"/>, an empty one for <see langword="null"/>.</param>
            /// <returns><c>(None, options)</c>.</returns>
            public override VectorResetResult<object?> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) =>
                new(null, options is null ? new Info() : new Info(options));

            /// <summary>Not implemented, as <c>VectorEnv.step</c> is not.</summary>
            /// <param name="actions">Ignored.</param>
            /// <returns>Never.</returns>
            /// <exception cref="NotImplementedError">Always.</exception>
            public override VectorStepResult<object?> Step(object? actions) => throw new NotImplementedError();
        }
    }
}
