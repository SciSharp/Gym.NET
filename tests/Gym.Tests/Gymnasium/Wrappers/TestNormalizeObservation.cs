#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Spaces;
using Gym.Vector;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Wrappers {
    /// <summary>Gymnasium's <c>NormalizeObservation</c> tests (<c>test_normalize_observation.py</c>), run against <see cref="NormalizeObservation{TObs, TAct}"/>.</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_normalize_observation.py">tests/wrappers/test_normalize_observation.py</see>.</remarks>
    [TestClass]
    public class TestNormalizeObservation {
        /// <summary>After 1,000 steps of uniform observations, the next 100 normalized ones have a mean near 0 and a variance near 1.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_normalize_observation.py#L13">tests/wrappers/test_normalize_observation.py (test_normalization)</see>.</remarks>
        [TestMethod]
        public void TestNormalization() {
            const int convergenceSteps = 1000, testingSteps = 100;
            using NDArray low = np.array(new float[] { 0, -10, -5 });
            using NDArray high = np.array(new float[] { 10, -5, 10 });
            var env = new GenericTestEnv(observationSpace: new Box(low, high));
            using var wrapped = new NormalizeObservation<object?, object?>(env);

            wrapped.Reset(seed: 123);
            wrapped.ObservationSpace.Seed(123);
            wrapped.ActionSpace.Seed(123);
            for (int i = 0; i < convergenceSteps; i++)
                wrapped.Step(wrapped.ActionSpace.Sample()).Observation.Dispose();

            var observations = new List<NDArray>();
            for (int i = 0; i < testingSteps; i++)
                observations.Add(wrapped.Step(wrapped.ActionSpace.Sample()).Observation);
            using NDArray stacked = np.stack(observations.ToArray()); // (100, 3)
            foreach (NDArray observation in observations)
                observation.Dispose();

            using NDArray meanObs = np.mean(stacked, axis: 0);
            using NDArray varObs = np.var(stacked, axis: 0);
            CollectionAssert.AreEqual(new long[] { 3 }, meanObs.shape);
            CollectionAssert.AreEqual(new long[] { 3 }, varObs.shape);

            using NDArray zeros = np.zeros(new Shape(3L));
            using NDArray ones = np.ones(new Shape(3L));
            Assert.IsTrue(np.allclose(meanObs, zeros, atol: 0.15), $"mean {meanObs}");
            Assert.IsTrue(np.allclose(varObs, ones, atol: 0.15), $"var {varObs}");
        }

        /// <summary><c>update_running_mean</c> freezes and resumes the statistics.</summary>
        /// <remarks>
        ///     The statistics' arrays are replaced (and the old ones released) by each update, so the test keeps copies where
        ///     Python keeps the replaced arrays alive by reference.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_normalize_observation.py#L42">tests/wrappers/test_normalize_observation.py (test_update_running_mean_property)</see>.
        /// </remarks>
        [TestMethod]
        public void TestUpdateRunningMeanProperty() {
            var env = new GenericTestEnv();
            using var wrappedEnv = new NormalizeObservation<object?, object?>(env);

            // Default value is True
            Assert.IsTrue(wrappedEnv.UpdateRunningMean);

            wrappedEnv.Reset();
            using NDArray rmsVarInit = wrappedEnv.ObsRms.Var.copy();
            using NDArray rmsMeanInit = wrappedEnv.ObsRms.Mean.copy();

            // Statistics are updated when env.step()
            wrappedEnv.Step(null);
            using NDArray rmsVarUpdated = wrappedEnv.ObsRms.Var.copy();
            using NDArray rmsMeanUpdated = wrappedEnv.ObsRms.Mean.copy();
            Assert.IsFalse(np.array_equal(rmsVarInit, rmsVarUpdated));
            Assert.IsFalse(np.array_equal(rmsMeanInit, rmsMeanUpdated));

            // Assure property is set
            wrappedEnv.UpdateRunningMean = false;
            Assert.IsFalse(wrappedEnv.UpdateRunningMean);

            // Statistics are frozen
            wrappedEnv.Step(null);
            Assert.IsTrue(np.array_equal(rmsVarUpdated, wrappedEnv.ObsRms.Var));
            Assert.IsTrue(np.array_equal(rmsMeanUpdated, wrappedEnv.ObsRms.Mean));
        }

        /// <summary>Four <c>NormalizeObservation(GrayscaleObservation(make("CarRacing-v3")))</c> in a sync vector env reset.</summary>
        /// <remarks>
        ///     CarRacing comes with P8 (Box2D); until then the id is not registered.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_normalize_observation.py#L71">tests/wrappers/test_normalize_observation.py (test_normalize_obs_with_vector)</see>.
        /// </remarks>
        [TestMethod]
        [OpenBugs("P8")]
        public void TestNormalizeObsWithVector() {
            static IEnv Thunk() {
                var env = (Env<NDArray, NDArray>)Make("CarRacing-v3");
                var grayscale = new GrayscaleObservation<NDArray, NDArray>(env);
                return new NormalizeObservation<NDArray, NDArray>(grayscale);
            }

            using var envs = new SyncVectorEnv(Enumerable.Range(0, 4).Select(_ => (Func<IEnv>)Thunk));
            envs.Reset();
        }

        /// <summary>A non-positive <c>epsilon</c> is refused: it sits under a square root, and past the variance it makes every observation NaN.</summary>
        /// <param name="epsilon">The epsilon.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_normalize_observation.py#L83">tests/wrappers/test_normalize_observation.py (test_non_positive_epsilon_is_rejected)</see>.</remarks>
        [TestMethod]
        [DataRow(0.0)]
        [DataRow(-1e-8)]
        [DataRow(-1.0)]
        public void TestNonPositiveEpsilonIsRejected(double epsilon) {
            var error = Assert.ThrowsExactly<InvalidBound>(() => new NormalizeObservation<object?, object?>(new GenericTestEnv(), epsilon: epsilon));
            StringAssert.Contains(error.Message, "`epsilon` should be strictly positive");
        }

        /// <summary>A positive <c>epsilon</c> is kept as given.</summary>
        /// <param name="epsilon">The epsilon.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_normalize_observation.py#L94">tests/wrappers/test_normalize_observation.py (test_positive_epsilon_is_accepted)</see>.</remarks>
        [TestMethod]
        [DataRow(1e-8)]
        [DataRow(1e-4)]
        [DataRow(1.0)]
        public void TestPositiveEpsilonIsAccepted(double epsilon) {
            using var env = new NormalizeObservation<object?, object?>(new GenericTestEnv(), epsilon: epsilon);
            Assert.AreEqual(epsilon, env.Epsilon);
        }
    }
}
