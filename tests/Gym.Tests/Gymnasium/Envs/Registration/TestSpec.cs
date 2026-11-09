#nullable enable
using System.Text.RegularExpressions;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Exceptions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Envs.Registration {
    /// <summary>
    ///     Gymnasium's <c>spec</c> tests (<c>tests/envs/registration/test_spec.py</c>): the registered spec itself, and the
    ///     errors of missing, malformed, versioned and default lookups, and of serializing a function entry point.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_spec.py">tests/envs/registration/test_spec.py</see>.</remarks>
    [TestClass]
    public class TestSpec {
        /// <summary><c>spec</c> returns the very spec the registry holds.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_spec.py#L10">test_spec</see>.</remarks>
        [TestMethod]
        public void TestSpecIsTheRegisteredOne() {
            EnvSpec spec = Spec("CartPole-v1");
            Assert.AreEqual("CartPole-v1", spec.Id);
            Assert.AreSame(Registry["CartPole-v1"], spec);
        }

        /// <summary>An older version is deprecated, a newer one does not exist, an unknown name does not exist.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_spec.py#L16">test_spec_missing_lookup</see>.</remarks>
        [TestMethod]
        public void TestSpecMissingLookup() {
            Register("TestEnv-v0", "no-entry-point");
            Register("TestEnv-v15", "no-entry-point");
            Register("TestEnv-v9", "no-entry-point");
            Register("OtherEnv-v100", "no-entry-point");
            try {
                StringAssert.Contains(Assert.ThrowsExactly<DeprecatedEnv>(() => Spec("TestEnv-v1")).Message,
                    "Environment version v1 for `TestEnv` is deprecated. Please use `TestEnv-v15` instead.");
                StringAssert.Contains(Assert.Throws<UnregisteredEnv>(() => Spec("TestEnv-v1000")).Message,
                    "Environment version `v1000` for environment `TestEnv` doesn't exist. It provides versioned environments: [ `v0`, `v9`, `v15` ].");
                StringAssert.Contains(Assert.Throws<UnregisteredEnv>(() => Spec("UnknownEnv-v1")).Message, "Environment `UnknownEnv` doesn't exist.");
            } finally {
                Registry.Remove("TestEnv-v0");
                Registry.Remove("TestEnv-v15");
                Registry.Remove("TestEnv-v9");
                Registry.Remove("OtherEnv-v100");
            }
        }

        /// <summary>A malformed id is refused with Gymnasium's full message.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_spec.py#L47">test_spec_malformed_lookup</see>.</remarks>
        [TestMethod]
        public void TestSpecMalformedLookup() {
            StringAssert.Contains(Assert.ThrowsExactly<Error>(() => Spec("“Breakout-v0”")).Message,
                "Malformed environment ID: “Breakout-v0”. (Currently all IDs must be of the form [namespace/](env-name)-v(version). (namespace is optional))");
        }

        /// <summary>In a namespace, a newer version does not exist and an older one is deprecated.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_spec.py#L57">test_spec_versioned_lookups</see>.</remarks>
        [TestMethod]
        public void TestSpecVersionedLookups() {
            Register("test/TestEnv-v5", "no-entry-point");
            try {
                StringAssert.Contains(Assert.ThrowsExactly<VersionNotFound>(() => Spec("test/TestEnv-v9")).Message,
                    "Environment version `v9` for environment `test/TestEnv` doesn't exist. It provides versioned environments: [ `v5` ].");
                StringAssert.Contains(Assert.ThrowsExactly<DeprecatedEnv>(() => Spec("test/TestEnv-v4")).Message,
                    "Environment version v4 for `test/TestEnv` is deprecated. Please use `test/TestEnv-v5` instead.");
                Assert.IsNotNull(Spec("test/TestEnv-v5"));
            } finally {
                Registry.Remove("test/TestEnv-v5");
            }
        }

        /// <summary>A version of an unversioned env names the default version.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_spec.py#L78">test_spec_default_lookups</see>.</remarks>
        [TestMethod]
        public void TestSpecDefaultLookups() {
            Register("test/TestEnv", "no-entry-point");
            try {
                StringAssert.Contains(Assert.ThrowsExactly<DeprecatedEnv>(() => Spec("test/TestEnv-v0")).Message,
                    "Environment version `v0` for environment `test/TestEnv` doesn't exist. It provides the default version `test/TestEnv`.");
                Assert.IsNotNull(Spec("test/TestEnv"));
            } finally {
                Registry.Remove("test/TestEnv");
            }
        }

        /// <summary>A function entry point cannot be serialized, with Gymnasium's message (the function printed as <c>&lt;...&gt;</c>).</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_spec.py#L93">test_check_can_jsonify</see>.</remarks>
        [TestMethod]
        public void TestCheckCanJsonify() {
            // Python's `def no_entry_point(): pass`: never called, it returns None.
            EnvCreator noEntryPoint = kwargs => null!;
            Register("test/TestEnv-v0", noEntryPoint);
            try {
                ValueError error = Assert.ThrowsExactly<ValueError>(() => Spec("test/TestEnv-v0").ToJson());
                Assert.IsTrue(Regex.IsMatch(error.Message,
                    "^Callable found in test/TestEnv-v0 for entry_point attribute with value=<.*>. Currently, Gymnasium does not support serialising callables.$"), error.Message);
            } finally {
                Registry.Remove("test/TestEnv-v0");
            }
        }
    }
}
