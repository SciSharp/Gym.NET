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
    /// <summary>
    ///     Gymnasium's <c>TransformObservation</c> tests (<c>test_lambda_observation.py</c>), run against
    ///     <see cref="TransformObservation{TWObs, TObs, TAct}"/>.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_lambda_observation.py">tests/wrappers/test_lambda_observation.py</see>.</remarks>
    [TestClass]
    public class TestLambdaObservation {
        /// <summary>The function is applied to both the reset and the step observation.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/test_lambda_observation.py#L11">tests/wrappers/test_lambda_observation.py (test_lambda_observation_wrapper)</see>.</remarks>
        [TestMethod]
        public void TestLambdaObservationWrapper() {
            var env = new GenericTestEnv(resetFunc: WrapperUtils.RecordObsReset, stepFunc: WrapperUtils.RecordActionAsObsStep);
            // The function must not dispose its argument: it is the env's observation, recorded in the info too.
            using var wrappedEnv = new TransformObservation<NDArray, object?, object?>(env, obs => (NDArray)obs! + 2, new Box(2, 3));

            using NDArray resetObs = np.array(new[] { 0f });
            var (obs, info) = wrappedEnv.Reset(options: new Dictionary<string, object?> { ["obs"] = resetObs });
            WrapperUtils.CheckObs(env, wrappedEnv, obs, info["obs"]);

            using NDArray action = np.array(new[] { 1f });
            (obs, _, _, _, info) = wrappedEnv.Step(action);
            WrapperUtils.CheckObs(env, wrappedEnv, obs, info["obs"]);
        }
    }
}
