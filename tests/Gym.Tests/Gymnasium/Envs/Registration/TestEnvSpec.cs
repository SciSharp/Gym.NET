#nullable enable
using Gym.Envs;
using Gym.Envs.Registration;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests.Gymnasium.Envs.Registration {
    /// <summary>
    ///     Gymnasium's <c>EnvSpec</c> tests (<c>tests/envs/registration/test_env_spec.py</c>): a spec read back from its JSON
    ///     equals the original, for a registered spec, a made env's unwrapped spec and its wrapped one.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Waiting for P6's wrappers: <c>test_full_integration</c> and <c>test_env_spec_pprint</c> (<c>TimeAwareObservation</c>,
    ///     <c>NormalizeReward</c>), and the <c>NormalizeReward</c> case of <c>test_env_spec_to_from_json</c>. Python-only:
    ///     <c>test_pickling_env_stack</c> (pickle) and <c>test_spec_with_unpickleable_object</c>, whose deep copy fails through
    ///     the pickle protocol; Gym.NET's deep copy shares an object it has no way to copy instead of failing.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_env_spec.py">tests/envs/registration/test_env_spec.py</see>.</para>
    /// </remarks>
    [TestClass]
    public class TestEnvSpec {
        /// <summary>
        ///     <c>to_json</c> then <c>from_json</c> gives an equal spec; the class entry point comes back as the
        ///     <c>"Namespace:Class"</c> string, which equals it.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_env_spec.py#L86">test_env_spec_to_from_json</see>.</remarks>
        [TestMethod]
        public void TestEnvSpecToFromJson() {
            using IEnv unwrappedSource = Make("CartPole-v1");
            using IEnv wrappedSource = Make("CartPole-v1");
            EnvSpec[] specs = { Spec("CartPole-v1"), unwrappedSource.Unwrapped.Spec!, wrappedSource.Spec! };
            foreach (EnvSpec envSpec in specs) {
                string jsonSpec = envSpec.ToJson();
                EnvSpec recreatedEnvSpec = EnvSpec.FromJson(jsonSpec);
                Assert.AreEqual(envSpec, recreatedEnvSpec, jsonSpec);
            }
        }
    }
}
