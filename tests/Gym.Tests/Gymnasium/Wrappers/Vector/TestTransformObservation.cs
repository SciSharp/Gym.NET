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
    ///     Gymnasium's tests of the vector <c>TransformObservation</c> (<c>test_transform_observation.py</c>), run against
    ///     <see cref="TransformObservationVector{TWObs, TObs, TAct}"/>.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_transform_observation.py">tests/wrappers/vector/test_transform_observation.py</see>.</remarks>
    [TestClass]
    public class TestTransformObservation {
        /// <summary>Adding 100 to every observation of the batch moves the reset's and a step's observations into the shifted bounds.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_transform_observation.py#L20">tests/wrappers/vector/test_transform_observation.py (test_transform)</see>.</remarks>
        [TestMethod]
        public void TestTransform() {
            const int nEnvs = 2;
            using var vecEnv = new TransformObservationVector<object?, object?, object?>(Create(nEnvs), x => (NDArray)x! + 100, singleObservationSpace: Bounds(0));

            var (obs, _) = vecEnv.Reset(seed: 123);
            vecEnv.ObservationSpace.Seed(123);
            vecEnv.ActionSpace.Seed(123);
            AssertWithin((NDArray)obs!, 100);

            obs = vecEnv.Step(vecEnv.ActionSpace.Sample()).Observations;
            AssertWithin((NDArray)obs!, 100);
        }

        /// <summary>A single space alone gives the batched space: the single box stacked five times, float32.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_transform_observation.py#L45">tests/wrappers/vector/test_transform_observation.py (test_observation_space_from_single_observation_space)</see>.</remarks>
        [TestMethod]
        public void TestObservationSpaceFromSingleObservationSpace() {
            const int nEnvs = 5;
            using var vecEnv = new TransformObservationVector<object?, object?, object?>(Create(nEnvs), x => (NDArray)x! + 100, singleObservationSpace: Bounds(100));

            // Check observation space
            var batched = (Box)ObjectSpace.Unview(vecEnv.ObservationSpace)!;
            CollectionAssert.AreEqual(new long[] { nEnvs, 3 }, batched.Shape.dimensions);
            Assert.IsTrue(batched.DType == np.float32);
            using (NDArray low = np.array(Enumerable.Repeat(new float[] { 100, 90, 95 }, nEnvs).ToArray()))
                Assert.IsTrue(np.array_equal(batched.Low, low));
            using (NDArray high = np.array(Enumerable.Repeat(new float[] { 110, 95, 110 }, nEnvs).ToArray()))
                Assert.IsTrue(np.array_equal(batched.High, high));

            // Check single observation space
            var single = (Box)ObjectSpace.Unview(vecEnv.SingleObservationSpace)!;
            CollectionAssert.AreEqual(new long[] { 3 }, single.Shape.dimensions);
            Assert.IsTrue(single.DType == np.float32);
            using (NDArray low = np.array(new float[] { 100, 90, 95 }))
                Assert.IsTrue(np.array_equal(single.Low, low));
            using (NDArray high = np.array(new float[] { 110, 95, 110 }))
                Assert.IsTrue(np.array_equal(single.High, high));
        }

        /// <summary>A batched space alone, against the env's single space it does not batch, warns.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_transform_observation.py#L83">tests/wrappers/vector/test_transform_observation.py (test_warning_on_mismatched_single_observation_space)</see>.</remarks>
        [TestMethod]
        public void TestWarningOnMismatchedSingleObservationSpace() {
            const int nEnvs = 2;
            SyncVectorEnv vecEnv = Create(nEnvs);
            using NDArray low = np.array(Enumerable.Repeat(new float[] { 100, 90, 95 }, nEnvs).ToArray());
            using NDArray high = np.array(Enumerable.Repeat(new float[] { 110, 95, 110 }, nEnvs).ToArray());
            using WarningCatcher caught = Warnings.Catch();
            using var wrapped = new TransformObservationVector<object?, object?, object?>(vecEnv, x => (NDArray)x! + 100, observationSpace: new ObjectSpace(new Box(low, high)));
            Assert.IsTrue(caught.Records.Any(warning => warning.Message.Contains("the observation space and the batched single observation space don't match as expected")),
                string.Join(Environment.NewLine, caught.Records.Select(warning => warning.Message)));
        }

        /// <summary>A sync vector env of <paramref name="nEnvs"/> test envs observing <c>Box([0, -10, -5], [10, -5, 10])</c>, float32 (<c>create_env</c>).</summary>
        /// <param name="nEnvs">The number of sub-envs.</param>
        /// <returns>The vector env.</returns>
        private static SyncVectorEnv Create(int nEnvs) => new(Enumerable.Range(0, nEnvs).Select(_ => (Func<IEnv>)(() => new GenericTestEnv(observationSpace: Bounds(0)))));

        /// <summary><c>Box([0, -10, -5] + offset, [10, -5, 10] + offset)</c>, float32.</summary>
        /// <param name="offset">Added to both bounds.</param>
        /// <returns>The box.</returns>
        private static Box Bounds(float offset) {
            using NDArray low = np.array(new[] { 0 + offset, -10 + offset, -5 + offset });
            using NDArray high = np.array(new[] { 10 + offset, -5 + offset, 10 + offset });
            return new Box(low, high);
        }

        /// <summary>Asserts every row of the batch lies within <c>[[100, 90, 95], [110, 95, 110]]</c> shifted by <c>offset - 100</c>.</summary>
        /// <param name="obs">The batched observations, <c>(n, 3)</c> float32.</param>
        /// <param name="offset">The shift the transform applied.</param>
        /// <exception cref="AssertFailedException">Thrown when an entry is outside.</exception>
        private static void AssertWithin(NDArray obs, float offset) {
            using NDArray low = np.array(new[] { offset, offset - 10, offset - 5 });
            using NDArray high = np.array(new[] { offset + 10, offset - 5, offset + 10 });
            using NDArray above = obs >= low;
            using NDArray below = obs <= high;
            Assert.IsTrue(np.all(above), $"{obs} below {low}");
            Assert.IsTrue(np.all(below), $"{obs} above {high}");
        }
    }
}
