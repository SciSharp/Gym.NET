#nullable enable
using System;
using System.Linq;
using Gym.Envs;
using Gym.Spaces;
using Gym.Utils;
using Gym.Vector;
using Gym.Wrappers.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Wrappers.Vector {
    /// <summary>
    ///     Gymnasium's tests of the vector <c>TransformAction</c> (<c>test_transform_action.py</c>), run against
    ///     <see cref="TransformActionVector{TObs, TWAct, TAct}"/>.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_transform_action.py">tests/wrappers/vector/test_transform_action.py</see>.</remarks>
    [TestClass]
    public class TestTransformAction {
        /// <summary>A single action space alone gives the batched space: the single box stacked five times, float32.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_transform_action.py#L20">tests/wrappers/vector/test_transform_action.py (test_action_space_from_single_action_space)</see>.</remarks>
        [TestMethod]
        public void TestActionSpaceFromSingleActionSpace() {
            const int nEnvs = 5;
            using var vecEnv = new TransformActionVector<object?, object?, object?>(Create(nEnvs), x => (NDArray)x! + 100, singleActionSpace: Bounds(100));

            // Check action space
            var batched = (Box)ObjectSpace.Unview(vecEnv.ActionSpace)!;
            CollectionAssert.AreEqual(new long[] { nEnvs, 3 }, batched.Shape.dimensions);
            Assert.IsTrue(batched.DType == np.float32);
            using (NDArray low = np.array(Enumerable.Repeat(new float[] { 100, 90, 95 }, nEnvs).ToArray()))
                Assert.IsTrue(np.array_equal(batched.Low, low));
            using (NDArray high = np.array(Enumerable.Repeat(new float[] { 110, 95, 110 }, nEnvs).ToArray()))
                Assert.IsTrue(np.array_equal(batched.High, high));

            // Check single action space
            var single = (Box)ObjectSpace.Unview(vecEnv.SingleActionSpace)!;
            CollectionAssert.AreEqual(new long[] { 3 }, single.Shape.dimensions);
            Assert.IsTrue(single.DType == np.float32);
            using (NDArray low = np.array(new float[] { 100, 90, 95 }))
                Assert.IsTrue(np.array_equal(single.Low, low));
            using (NDArray high = np.array(new float[] { 110, 95, 110 }))
                Assert.IsTrue(np.array_equal(single.High, high));
        }

        /// <summary>A batched action space alone, against the env's single space it does not batch, warns.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_transform_action.py#L56">tests/wrappers/vector/test_transform_action.py (test_warning_on_mismatched_single_action_space)</see>.</remarks>
        [TestMethod]
        public void TestWarningOnMismatchedSingleActionSpace() {
            const int nEnvs = 2;
            SyncVectorEnv vecEnv = Create(nEnvs);
            using NDArray low = np.array(Enumerable.Repeat(new float[] { 100, 90, 95 }, nEnvs).ToArray());
            using NDArray high = np.array(Enumerable.Repeat(new float[] { 110, 95, 110 }, nEnvs).ToArray());
            using WarningCatcher caught = Warnings.Catch();
            using var wrapped = new TransformActionVector<object?, object?, object?>(vecEnv, x => (NDArray)x! + 100, actionSpace: new ObjectSpace(new Box(low, high)));
            Assert.IsTrue(caught.Records.Any(warning => warning.Message.Contains("the action space and the batched single action space don't match as expected")),
                string.Join(Environment.NewLine, caught.Records.Select(warning => warning.Message)));
        }

        /// <summary>A sync vector env of <paramref name="nEnvs"/> test envs acting in <c>Box([0, -10, -5], [10, -5, 10])</c>, float32 (<c>create_env</c>).</summary>
        /// <param name="nEnvs">The number of sub-envs.</param>
        /// <returns>The vector env.</returns>
        private static SyncVectorEnv Create(int nEnvs) => new(Enumerable.Range(0, nEnvs).Select(_ => (Func<IEnv>)(() => new GenericTestEnv(actionSpace: Bounds(0)))));

        /// <summary><c>Box([0, -10, -5] + offset, [10, -5, 10] + offset)</c>, float32.</summary>
        /// <param name="offset">Added to both bounds.</param>
        /// <returns>The box.</returns>
        private static Box Bounds(float offset) {
            using NDArray low = np.array(new[] { 0 + offset, -10 + offset, -5 + offset });
            using NDArray high = np.array(new[] { 10 + offset, -5 + offset, 10 + offset });
            return new Box(low, high);
        }
    }
}
