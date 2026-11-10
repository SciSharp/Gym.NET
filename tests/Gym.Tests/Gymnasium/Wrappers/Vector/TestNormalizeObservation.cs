#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Spaces;
using Gym.Vector;
using Gym.Wrappers;
using Gym.Wrappers.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Wrappers.Vector {
    /// <summary>
    ///     Gymnasium's tests of the vector <c>NormalizeObservation</c> (<c>test_normalize_observation.py</c>), run against
    ///     <see cref="NormalizeObservationVector{TAct}"/>.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_normalize_observation.py">tests/wrappers/vector/test_normalize_observation.py</see>.</remarks>
    [TestClass]
    public class TestNormalizeObservation {
        /// <summary>After 250 steps of two envs' uniform observations, the next 100 normalized batches have a mean near 0 and a variance near 1.</summary>
        /// <remarks>
        ///     Gymnasium averages the <c>(100, 2, 3)</c> stack over its first two axes; the port reshapes it to <c>(200, 3)</c>
        ///     and averages over the first, the same entries.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_normalize_observation.py#L21">tests/wrappers/vector/test_normalize_observation.py (test_normalization)</see>.
        /// </remarks>
        [TestMethod]
        public void TestNormalization() {
            const int nEnvs = 2, convergenceSteps = 250, testingSteps = 100;
            using var vecEnv = new NormalizeObservationVector<NDArray>(Create(nEnvs));

            vecEnv.Reset(seed: 123);
            vecEnv.ObservationSpace.Seed(123);
            vecEnv.ActionSpace.Seed(123);
            for (int i = 0; i < convergenceSteps; i++)
                vecEnv.Step(vecEnv.ActionSpace.Sample());

            var observations = new List<NDArray>();
            for (int i = 0; i < testingSteps; i++)
                observations.Add(vecEnv.Step(vecEnv.ActionSpace.Sample()).Observations);
            using NDArray stacked = np.stack(observations.ToArray()); // (100, 2, 3)
            foreach (NDArray observation in observations)
                observation.Dispose();
            using NDArray rows = stacked.reshape(testingSteps * nEnvs, 3);

            using NDArray meanObs = np.mean(rows, axis: 0);
            using NDArray varObs = np.var(rows, axis: 0);
            CollectionAssert.AreEqual(new long[] { 3 }, meanObs.shape);
            CollectionAssert.AreEqual(new long[] { 3 }, varObs.shape);

            using NDArray zeros = np.zeros(new Shape(3L));
            using NDArray ones = np.ones(new Shape(3L));
            Assert.IsTrue(np.allclose(meanObs, zeros, atol: 0.15), $"mean {meanObs}");
            Assert.IsTrue(np.allclose(varObs, ones, atol: 0.2), $"var {varObs}");
        }

        /// <summary>
        ///     The vector wrapper's statistics after 250 steps of three envs are close to the single-env wrapper's after 83
        ///     steps of one autoresetting env, within Gymnasium's per-entry relative tolerances.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_normalize_observation.py#L47">tests/wrappers/vector/test_normalize_observation.py (test_wrapper_equivalence)</see>.</remarks>
        [TestMethod]
        public void TestWrapperEquivalence() {
            const int nEnvs = 3, nSteps = 250;
            double[] meanRtol = { 0.1, 0.4, 0.25 }, varRtol = { 0.15, 0.15, 0.18 };

            using var vecEnv = new NormalizeObservationVector<NDArray>(Create(nEnvs));
            vecEnv.Reset(seed: 123);
            vecEnv.ObservationSpace.Seed(123);
            vecEnv.ActionSpace.Seed(123);
            for (int i = 0; i < nSteps; i++)
                vecEnv.Step(vecEnv.ActionSpace.Sample());

            using var env = new NormalizeObservation<object?, object?>(new Autoreset<object?, object?>(CreateEnv()));
            env.Reset(seed: 123);
            env.ActionSpace.Seed(123);
            for (int i = 0; i < nSteps / nEnvs; i++)
                env.Step(env.ActionSpace.Sample());

            Assert.IsTrue(AllClose(env.ObsRms.Mean, vecEnv.ObsRms.Mean, meanRtol), $"mean {env.ObsRms.Mean} against {vecEnv.ObsRms.Mean}");
            Assert.IsTrue(AllClose(env.ObsRms.Var, vecEnv.ObsRms.Var, varRtol), $"var {env.ObsRms.Var} against {vecEnv.ObsRms.Var}");
        }

        /// <summary><c>update_running_mean</c> freezes the statistics through 10 steps and resumes them for the next 10.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_normalize_observation.py#L75">tests/wrappers/vector/test_normalize_observation.py (test_update_running_mean)</see>.</remarks>
        [TestMethod]
        public void TestUpdateRunningMean() {
            using var env = new NormalizeObservationVector<NDArray>(Create(2));

            // Default value is True
            Assert.IsTrue(env.UpdateRunningMean);

            env.Reset();
            for (int i = 0; i < 100; i++)
                env.Step(env.ActionSpace.Sample());

            // Disable updating the running mean
            env.UpdateRunningMean = false;
            using NDArray copiedRmsMean = np.copy(env.ObsRms.Mean);
            using NDArray copiedRmsVar = np.copy(env.ObsRms.Var);

            // Continue stepping through the environment and check that the running mean is not effected
            for (int i = 0; i < 10; i++)
                env.Step(env.ActionSpace.Sample());

            Assert.IsTrue(np.array_equal(copiedRmsMean, env.ObsRms.Mean));
            Assert.IsTrue(np.array_equal(copiedRmsVar, env.ObsRms.Var));

            // Re-enable updating the running mean
            env.UpdateRunningMean = true;

            for (int i = 0; i < 10; i++)
                env.Step(env.ActionSpace.Sample());

            using (NDArray changed = copiedRmsMean != env.ObsRms.Mean)
                Assert.IsTrue(np.any(changed));
            using (NDArray changed = copiedRmsVar != env.ObsRms.Var)
                Assert.IsTrue(np.any(changed));
        }

        /// <summary>The wrapper's single space is an unbounded float32 box, and its observations are float32.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_normalize_observation.py#L105">tests/wrappers/vector/test_normalize_observation.py (test_observation_space_and_dtype)</see>.</remarks>
        [TestMethod]
        public void TestObservationSpaceAndDtype() {
            using var vecEnv = new NormalizeObservationVector<NDArray>(Create(2));

            var single = (Box)ObjectSpace.Unview(vecEnv.SingleObservationSpace)!;
            Assert.IsTrue(single.DType == np.float32);
            Assert.IsTrue(single.Low.ToArray<float>().All(float.IsNegativeInfinity));
            Assert.IsTrue(single.High.ToArray<float>().All(float.IsPositiveInfinity));

            NDArray obs = vecEnv.Reset(seed: 123).Observations;
            Assert.IsTrue(obs.dtype == typeof(float));

            obs = vecEnv.Step(vecEnv.ActionSpace.Sample()).Observations;
            Assert.IsTrue(obs.dtype == typeof(float));
        }

        /// <summary>A non-positive <c>epsilon</c> is refused, as the single-env wrapper refuses it.</summary>
        /// <param name="epsilon">The epsilon.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_normalize_observation.py#L121">tests/wrappers/vector/test_normalize_observation.py (test_non_positive_epsilon_is_rejected)</see>.</remarks>
        [TestMethod]
        [DataRow(0.0)]
        [DataRow(-1e-8)]
        [DataRow(-1.0)]
        public void TestNonPositiveEpsilonIsRejected(double epsilon) {
            using SyncVectorEnv<NDArray, NDArray> vecEnv = Create(1);
            var error = Assert.ThrowsExactly<InvalidBound>(() => new NormalizeObservationVector<NDArray>(vecEnv, epsilon: epsilon));
            StringAssert.Contains(error.Message, "`epsilon` should be strictly positive");
            vecEnv.Close();
        }

        /// <summary>A test env observing <c>Box([0, -10, -5], [10, -5, 10])</c>, float32 (<c>create_env</c>).</summary>
        /// <returns>The env.</returns>
        private static GenericTestEnv CreateEnv() {
            using NDArray low = np.array(new float[] { 0, -10, -5 });
            using NDArray high = np.array(new float[] { 10, -5, 10 });
            return new GenericTestEnv(observationSpace: new Box(low, high));
        }

        /// <summary>A sync vector env of <paramref name="nEnvs"/> <see cref="CreateEnv"/> envs, typed by its float32 observation batches.</summary>
        /// <param name="nEnvs">The number of sub-envs.</param>
        /// <returns>The vector env.</returns>
        private static SyncVectorEnv<NDArray, NDArray> Create(int nEnvs) => new(Enumerable.Range(0, nEnvs).Select(_ => (Func<IEnv>)CreateEnv));

        /// <summary><c>np.allclose(a, b, rtol=rtol)</c> with a per-entry relative tolerance: <c>|a - b| &lt;= 1e-8 + rtol * |b|</c> for every entry.</summary>
        /// <param name="a">The first array, one entry per tolerance.</param>
        /// <param name="b">The second array, the reference of the relative tolerance.</param>
        /// <param name="rtol">The relative tolerance of each entry.</param>
        /// <returns>Whether every entry is close.</returns>
        private static bool AllClose(NDArray a, NDArray b, double[] rtol) {
            using NDArray wideA = a.astype(NPTypeCode.Double);
            using NDArray wideB = b.astype(NPTypeCode.Double);
            double[] left = wideA.ToArray<double>(), right = wideB.ToArray<double>();
            return left.Select((value, i) => Math.Abs(value - right[i]) <= 1e-8 + rtol[i] * Math.Abs(right[i])).All(close => close);
        }
    }
}
