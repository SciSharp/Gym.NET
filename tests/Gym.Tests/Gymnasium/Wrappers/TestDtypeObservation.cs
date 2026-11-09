#nullable enable
using System.Collections.Generic;
using System.Linq;
using Gym.Envs;
using Gym.Spaces;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Tuple = Gym.Spaces.Tuple;

namespace Gym.Tests.Gymnasium.Wrappers {
    /// <summary>Gymnasium's <c>DtypeObservation</c> tests (<c>test_dtype_observation.py</c>), run against <see cref="DtypeObservation{TObs, TAct}"/>.</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_dtype_observation.py">tests/wrappers/test_dtype_observation.py</see>.</remarks>
    [TestClass]
    public class TestDtypeObservation {
        /// <summary>The observation's dtype is changed.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_dtype_observation.py#L12">tests/wrappers/test_dtype_observation.py (test_dtype_observation)</see>.</remarks>
        [TestMethod]
        public void TestDtypeObservationWrapper() {
            var env = new GenericTestEnv(resetFunc: WrapperUtils.RecordRandomObsReset, stepFunc: WrapperUtils.RecordRandomObsStep);
            using var wrappedEnv = new DtypeObservation<object?, object?>(env, np.uint8);
            var (obs, info) = wrappedEnv.Reset();
            Assert.AreNotEqual(((NDArray)info["obs"]!).dtype, obs.dtype);
            Assert.AreEqual(np.uint8, obs.dtype);
            (obs, _, _, _, info) = wrappedEnv.Step(null);
            Assert.AreNotEqual(((NDArray)info["obs"]!).dtype, obs.dtype);
            Assert.AreEqual(np.uint8, obs.dtype);
        }

        /// <summary>
        ///     Equivalent dtype descriptions (a type, a dtype, a name) give the same observations and spaces, for every space
        ///     kind the wrapper takes.
        /// </summary>
        /// <param name="spaceName">The space: <c>box2</c>, <c>box0</c>, <c>discrete</c>, <c>multidiscrete</c> or <c>multibinary</c>.</param>
        /// <param name="target">The target dtype's name.</param>
        /// <param name="form">How the dtype is given: <c>type</c> (a .NET type), <c>dtype</c> or <c>string</c>.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_dtype_observation.py#L40">tests/wrappers/test_dtype_observation.py (test_dtype_observation_descriptors)</see>.</remarks>
        [TestMethod]
        [DataRow("box2", "float64", "type"), DataRow("box2", "float64", "dtype"), DataRow("box2", "float64", "string")]
        [DataRow("box0", "float64", "type"), DataRow("box0", "float64", "dtype"), DataRow("box0", "float64", "string")]
        [DataRow("discrete", "float64", "type"), DataRow("discrete", "float64", "dtype"), DataRow("discrete", "float64", "string")]
        [DataRow("discrete", "int32", "type"), DataRow("discrete", "int32", "dtype"), DataRow("discrete", "int32", "string")]
        [DataRow("multidiscrete", "int32", "type"), DataRow("multidiscrete", "int32", "dtype"), DataRow("multidiscrete", "int32", "string")]
        [DataRow("multibinary", "int32", "type"), DataRow("multibinary", "int32", "dtype"), DataRow("multibinary", "int32", "string")]
        public void TestDtypeObservationDescriptors(string spaceName, string target, string form) {
            DType targetDtype = np.dtype(target);
            object dtype = form switch {
                "type" => target == "float64" ? typeof(double) : typeof(int),
                "dtype" => targetDtype,
                _ => targetDtype.name,
            };
            Space space = spaceName switch {
                "box2" => new Box(-2, 2, new Shape(2), np.int16),
                "box0" => new Box(-2, 2, Shape.Scalar, np.int16),
                "discrete" => new Discrete(3, start: -1),
                "multidiscrete" => new MultiDiscrete(new[] { 3L, 4L }, start: new[] { -1L, 2L }),
                _ => new MultiBinary(3),
            };
            var env = new GenericTestEnv(observationSpace: space, resetFunc: WrapperUtils.RecordRandomObsReset, stepFunc: WrapperUtils.RecordRandomObsStep);
            using var wrappedEnv = new DtypeObservation<object?, object?>(env, dtype);
            Assert.AreEqual(targetDtype, wrappedEnv.ObservationSpace.DType);

            var (obs, info) = wrappedEnv.Reset();
            AssertObservation(obs, info, space, targetDtype, wrappedEnv);
            (obs, _, _, _, info) = wrappedEnv.Step(null);
            AssertObservation(obs, info, space, targetDtype, wrappedEnv);
        }

        /// <summary>The wrapped <c>Discrete</c> space has the original's closed interval <c>[start, start + n - 1]</c>.</summary>
        /// <param name="n">The space's size.</param>
        /// <param name="start">The space's start.</param>
        /// <param name="spaceDtype">The space's own dtype.</param>
        /// <param name="dtype">The target dtype.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_dtype_observation.py#L80">tests/wrappers/test_dtype_observation.py (test_dtype_observation_discrete_space_matches_original_support)</see>.</remarks>
        [TestMethod]
        [DataRow(2L, 0L, "int64", "int64")]
        [DataRow(3L, -1L, "int64", "int32")]
        [DataRow(4L, 10L, "int32", "int16")]
        public void TestDtypeObservationDiscreteSpaceMatchesOriginalSupport(long n, long start, string spaceDtype, string dtype) {
            var space = new Discrete(n, start: start, dtype: np.dtype(spaceDtype));
            var env = new GenericTestEnv(observationSpace: space);
            DType target = np.dtype(dtype);
            using var wrappedEnv = new DtypeObservation<object?, object?>(env, target);
            using (var expected = new Box((double)start, (double)(start + n - 1), Shape.Scalar, target))
                Assert.IsTrue(wrappedEnv.ObservationSpace.Equals(expected));

            using NDArray last = NDArray.Scalar(start + n).astype(target);
            Assert.IsFalse(wrappedEnv.ObservationSpace.Contains(last));
            Assert.IsFalse(space.Contains(last));
            for (int i = 0; i < 50; i++) {
                using NDArray sample = wrappedEnv.ObservationSpace.Sample();
                Assert.IsTrue(wrappedEnv.ObservationSpace.Contains(sample));
                Assert.IsTrue(space.Contains(System.Convert.ToInt64(sample.GetAtIndex(0))));
            }
        }

        /// <summary>Changing a <c>MultiDiscrete</c>'s dtype keeps its non-zero start.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_dtype_observation.py#L106">tests/wrappers/test_dtype_observation.py (test_dtype_observation_multidiscrete_keeps_start)</see>.</remarks>
        [TestMethod]
        public void TestDtypeObservationMultidiscreteKeepsStart() {
            var space = new MultiDiscrete(new[] { 3L, 4L }, start: new[] { -1L, 2L });
            var env = new GenericTestEnv(observationSpace: space);
            using var wrappedEnv = new DtypeObservation<object?, object?>(env, np.int32);
            using (var expected = new MultiDiscrete(new[] { 3L, 4L }, np.int32, start: new[] { -1L, 2L }))
                Assert.IsTrue(wrappedEnv.ObservationSpace.Equals(expected));
            var (obs, _) = wrappedEnv.Reset();
            Assert.IsTrue(wrappedEnv.ObservationSpace.Contains(obs));
            Assert.AreEqual(np.int32, obs.dtype);
        }

        /// <summary>The checks of <see cref="TestDtypeObservationDescriptors"/> on one observation.</summary>
        /// <param name="obs">The wrapper's observation.</param>
        /// <param name="info">The info, holding the env's observation.</param>
        /// <param name="space">The env's space.</param>
        /// <param name="target">The target dtype.</param>
        /// <param name="wrappedEnv">The wrapper.</param>
        private static void AssertObservation(NDArray obs, Info info, Space space, DType target, IEnv wrappedEnv) {
            Assert.AreEqual(target, obs.dtype);
            Assert.AreEqual(space.Shape, obs.Shape);
            Assert.IsTrue(wrappedEnv.ObservationSpace.Contains(obs));
            // Every space here samples arrays (a Discrete sample is a 0-d array), so the recorded observation is one.
            var original = (NDArray)info["obs"]!;
            Assert.IsTrue(np.array_equal(obs, original));
            if (space is Discrete)
                Assert.AreEqual(0, obs.ndim, "A Discrete observation becomes a NumPy scalar, Gym.NET's 0-d array.");
        }
    }
}
