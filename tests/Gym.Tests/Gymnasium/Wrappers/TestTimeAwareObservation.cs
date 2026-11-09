#nullable enable
using System.Collections.Generic;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Spaces;
using Gym.Utils;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Tuple = Gym.Spaces.Tuple;

namespace Gym.Tests.Gymnasium.Wrappers {
    /// <summary>Gymnasium's <c>TimeAwareObservation</c> tests (<c>test_time_aware_observation.py</c>), run against <see cref="TimeAwareObservation{TObs, TAct}"/>.</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_time_aware_observation.py">tests/wrappers/test_time_aware_observation.py</see>.</remarks>
    [TestClass]
    public class TestTimeAwareObservation {
        /// <summary>The flattened default over CartPole and Pendulum: one more element, the step count, reset to 0 by a reset.</summary>
        /// <param name="envId">The env.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_time_aware_observation.py#L18">tests/wrappers/test_time_aware_observation.py (test_default)</see>.</remarks>
        [TestMethod]
        [DataRow("CartPole-v1")]
        [DataRow("Pendulum-v1")]
        public void TestDefault(string envId) {
            var env = (Env<NDArray, NDArray>)Make(envId, disableEnvChecker: true);
            using var wrappedEnv = new TimeAwareObservation<NDArray, NDArray>(env);

            Assert.IsInstanceOfType<Box>(env.ObservationSpace);
            Assert.IsInstanceOfType<Box>(ObjectSpace.Unview(wrappedEnv.ObservationSpace));
            Assert.AreEqual(env.ObservationSpace.Shape!.Value[0] + 1, wrappedEnv.ObservationSpace.Shape!.Value[0]);

            NDArray obs = env.Reset().Observation;
            var wrappedObs = (NDArray)wrappedEnv.Reset().Observation!;
            Assert.AreEqual(0L, wrappedEnv.Timesteps);
            Assert.AreEqual(0.0, (double)wrappedObs[-1], PyBuiltins.Repr(wrappedObs));
            Assert.AreEqual(obs.shape[0] + 1, wrappedObs.shape[0]);

            wrappedObs = (NDArray)wrappedEnv.Step(env.ActionSpace.Sample()).Observation!;
            Assert.AreEqual(1L, wrappedEnv.Timesteps);
            Assert.AreEqual(1.0, (double)wrappedObs[-1]);
            Assert.AreEqual(obs.shape[0] + 1, wrappedObs.shape[0]);

            wrappedObs = (NDArray)wrappedEnv.Step(env.ActionSpace.Sample()).Observation!;
            Assert.AreEqual(2L, wrappedEnv.Timesteps);
            Assert.AreEqual(2.0, (double)wrappedObs[-1]);
            Assert.AreEqual(obs.shape[0] + 1, wrappedObs.shape[0]);

            wrappedObs = (NDArray)wrappedEnv.Reset().Observation!;
            Assert.AreEqual(0L, wrappedEnv.Timesteps);
            Assert.AreEqual(0.0, (double)wrappedObs[-1]);
            Assert.AreEqual(obs.shape[0] + 1, wrappedObs.shape[0]);
        }

        /// <summary>Without a spec the step limit must come from a <c>TimeLimit</c>, which then serves without a warning.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_time_aware_observation.py#L48">tests/wrappers/test_time_aware_observation.py (test_no_spec)</see>.</remarks>
        [TestMethod]
        public void TestNoSpec() {
            var env = new CartPoleEnv();

            var error = Assert.ThrowsExactly<ValueError>(() => new TimeAwareObservation<NDArray, NDArray>(env));
            StringAssert.Contains(error.Message, "The environment must be wrapped by a TimeLimit wrapper or the spec specify a `max_episode_steps`.");

            var limited = new TimeLimit<NDArray, NDArray>(env, 100);
            using (WarningCatcher caughtWarnings = Warnings.Catch()) {
                using var wrapped = new TimeAwareObservation<NDArray, NDArray>(limited);
                Assert.AreEqual(100L, wrapped.MaxTimesteps);
                Assert.AreEqual(0, caughtWarnings.Records.Count);
            }
        }

        /// <summary>A <c>Box</c> env's observations, flattened by default, gain the time as a second element.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_time_aware_observation.py#L67">tests/wrappers/test_time_aware_observation.py (test_no_flatten)</see>.</remarks>
        [TestMethod]
        public void TestNoFlatten() {
            var env = new GenericTestEnv(observationSpace: new Box(0, 1));
            using var wrappedEnv = new TimeAwareObservation<object?, object?>(env);
            Assert.IsInstanceOfType<Box>(ObjectSpace.Unview(wrappedEnv.ObservationSpace));
            var resetObs = (NDArray)wrappedEnv.Reset().Observation!;
            var stepObs = (NDArray)wrappedEnv.Step(null).Observation!;
            CollectionAssert.AreEqual(new long[] { 2 }, resetObs.shape);
            CollectionAssert.AreEqual(new long[] { 2 }, stepObs.shape);

            Assert.IsTrue(wrappedEnv.ObservationSpace.Contains(resetObs));
            Assert.IsTrue(wrappedEnv.ObservationSpace.Contains(stepObs));
        }

        /// <summary>Unflattened, a <c>Dict</c> gains a <c>time</c> key, a <c>Tuple</c> a last item, and anything else becomes <c>{"obs", "time"}</c>.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_time_aware_observation.py#L80">tests/wrappers/test_time_aware_observation.py (test_with_flatten)</see>.</remarks>
        [TestMethod]
        public void TestWithFlatten() {
            var env = new GenericTestEnv(observationSpace: new Dict(new Dictionary<string, Space> { ["arm_1"] = new Box(0, 1), ["arm_2"] = new Box(2, 3) }));
            var wrappedEnv = new TimeAwareObservation<object?, object?>(env, flatten: false);
            Assert.IsInstanceOfType<Dict>(ObjectSpace.Unview(wrappedEnv.ObservationSpace));
            var resetObs = (IReadOnlyDictionary<string, object?>)wrappedEnv.Reset().Observation!;
            var stepObs = (IReadOnlyDictionary<string, object?>)wrappedEnv.Step(null).Observation!;
            Assert.IsTrue(resetObs.ContainsKey("time") && stepObs.ContainsKey("time"), $"{PyBuiltins.Repr(resetObs)}, {PyBuiltins.Repr(stepObs)}");

            Assert.IsTrue(wrappedEnv.ObservationSpace.Contains(resetObs));
            Assert.IsTrue(wrappedEnv.ObservationSpace.Contains(stepObs));
            wrappedEnv.Dispose();

            env = new GenericTestEnv(observationSpace: new Tuple(new Space[] { new Box(0, 1), new Box(2, 3) }));
            wrappedEnv = new TimeAwareObservation<object?, object?>(env, flatten: false);
            Assert.IsInstanceOfType<Tuple>(ObjectSpace.Unview(wrappedEnv.ObservationSpace));
            var resetTuple = (object?[])wrappedEnv.Reset().Observation!;
            var stepTuple = (object?[])wrappedEnv.Step(null).Observation!;
            Assert.IsTrue(resetTuple.Length == 3 && stepTuple.Length == 3);

            Assert.IsTrue(wrappedEnv.ObservationSpace.Contains(resetTuple));
            Assert.IsTrue(wrappedEnv.ObservationSpace.Contains(stepTuple));
            wrappedEnv.Dispose();

            env = new GenericTestEnv(observationSpace: new Box(0, 1));
            wrappedEnv = new TimeAwareObservation<object?, object?>(env, flatten: false);
            Assert.IsInstanceOfType<Dict>(ObjectSpace.Unview(wrappedEnv.ObservationSpace));
            object? reset = wrappedEnv.Reset().Observation;
            object? step = wrappedEnv.Step(null).Observation;
            Assert.IsTrue(reset is IReadOnlyDictionary<string, object?> && step is IReadOnlyDictionary<string, object?>);
            resetObs = (IReadOnlyDictionary<string, object?>)reset!;
            stepObs = (IReadOnlyDictionary<string, object?>)step!;
            Assert.IsTrue(resetObs.ContainsKey("obs") && stepObs.ContainsKey("obs"));
            Assert.IsTrue(resetObs.ContainsKey("time") && stepObs.ContainsKey("time"));

            Assert.IsTrue(wrappedEnv.ObservationSpace.Contains(reset));
            Assert.IsTrue(wrappedEnv.ObservationSpace.Contains(step));
            wrappedEnv.Dispose();
        }

        /// <summary>The time as int32 step counts, or with <c>normalize_time</c> as float32 fractions of the 100-step limit.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_time_aware_observation.py#L115">tests/wrappers/test_time_aware_observation.py (test_normalize_time)</see>.</remarks>
        [TestMethod]
        public void TestNormalizeTime() {
            var env = new GenericTestEnv(observationSpace: new Box(0, 1));
            var wrappedEnv = new TimeAwareObservation<object?, object?>(env, flatten: false, normalizeTime: false);
            var resetObs = (IReadOnlyDictionary<string, object?>)wrappedEnv.Reset().Observation!;
            var stepObs = (IReadOnlyDictionary<string, object?>)wrappedEnv.Step(null).Observation!;
            using (NDArray zero = np.array(new[] { 0 }), one = np.array(new[] { 1 })) {
                Assert.IsTrue(np.array_equal((NDArray)resetObs["time"]!, zero) && np.array_equal((NDArray)stepObs["time"]!, one));
                Assert.AreEqual(NPTypeCode.Int32, ((NDArray)resetObs["time"]!).typecode);
            }

            Assert.IsTrue(wrappedEnv.ObservationSpace.Contains(resetObs));
            Assert.IsTrue(wrappedEnv.ObservationSpace.Contains(stepObs));
            wrappedEnv.Dispose();

            env = new GenericTestEnv(observationSpace: new Box(0, 1));
            wrappedEnv = new TimeAwareObservation<object?, object?>(env, flatten: false, normalizeTime: true);
            resetObs = (IReadOnlyDictionary<string, object?>)wrappedEnv.Reset().Observation!;
            stepObs = (IReadOnlyDictionary<string, object?>)wrappedEnv.Step(null).Observation!;
            // reset_obs["time"] == 0.0 and step_obs["time"] == 0.01: float32 values against Python floats.
            Assert.AreEqual(0.0f, ((NDArray)resetObs["time"]!).GetSingle(0));
            Assert.AreEqual(0.01f, ((NDArray)stepObs["time"]!).GetSingle(0));

            Assert.IsTrue(wrappedEnv.ObservationSpace.Contains(resetObs));
            Assert.IsTrue(wrappedEnv.ObservationSpace.Contains(stepObs));
            wrappedEnv.Dispose();
        }
    }
}
