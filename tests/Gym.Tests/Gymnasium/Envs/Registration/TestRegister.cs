#nullable enable
using System;
using System.Collections.Generic;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Exceptions;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests.Gymnasium.Envs.Registration {
    /// <summary>
    ///     Gymnasium's <c>register</c> tests (<c>tests/envs/registration/test_register.py</c>): the id grammar, malformed ids,
    ///     the "Did you mean" suggestions for names, namespaces and versions, versioned against unversioned ids, the latest
    ///     version for an unversioned <c>make</c>, and the namespace scope.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The suggestion cases that name envs of later phases (Blackjack and Taxi, P5; MountainCarContinuous, P4) are the
    ///     <c>[OpenBugs]</c> methods below, which pass once those envs register.
    ///     </para>
    ///     <para>
    ///     The entry point <c>tests.envs.registration.utils_envs:ArgumentEnv</c> is this namespace's <c>ArgumentEnv</c>:
    ///     <c>Gym.Tests.Gymnasium.Envs.Registration:ArgumentEnv</c>.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_register.py">tests/envs/registration/test_register.py</see>.</para>
    /// </remarks>
    [TestClass]
    public class TestRegister {
        /// <summary>The namespace of the fixture's envs.</summary>
        private const string Namespace = "MyAwesomeNamespace";

        /// <summary>The name of the fixture's versioned envs.</summary>
        private const string VersionedName = "MyAwesomeVersionedEnv";

        /// <summary>The name of the fixture's unversioned env.</summary>
        private const string UnversionedName = "MyAwesomeUnversionedEnv";

        /// <summary>The versions the fixture registers.</summary>
        private static readonly long[] Versions = { 1, 3, 5 };

        /// <summary>
        ///     Gymnasium's <c>register_registration_testing_envs</c> fixture: <c>ArgumentEnv</c> under three versions and one
        ///     unversioned id; disposing the result unregisters them.
        /// </summary>
        /// <returns>The fixture's teardown.</returns>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_register.py#L10">tests/envs/registration/test_register.py (register_registration_testing_envs)</see>.</remarks>
        private static IDisposable RegisterRegistrationTestingEnvs() {
            foreach (long version in Versions) {
                Register($"{Namespace}/{VersionedName}-v{version}", "Gym.Tests.Gymnasium.Envs.Registration:ArgumentEnv",
                    kwargs: new Info { ["arg1"] = "arg1", ["arg2"] = "arg2", ["arg3"] = "arg3" });
            }

            // Gymnasium's fixture misspells the module (tests.env.utils_envs); nothing makes that id, so it never matters.
            Register($"{Namespace}/{UnversionedName}", "Gym.Tests.Gymnasium.Env:ArgumentEnv",
                kwargs: new Info { ["arg1"] = "arg1", ["arg2"] = "arg2", ["arg3"] = "arg3" });

            return new Teardown(() => {
                foreach (long version in Versions)
                    Registry.Remove($"{Namespace}/{VersionedName}-v{version}");
                Registry.Remove($"{Namespace}/{UnversionedName}");
            });
        }

        /// <summary>Registering an id parses it into the expected namespace, name and version, and registers it under the full id.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_register.py#L44">test_register</see>.</remarks>
        [TestMethod]
        public void TestRegisterParsesIds() {
            (string EnvId, string? Namespace, string Name, long? Version)[] cases = {
                ("MyAwesomeNamespace/MyAwesomeEnv-v0", "MyAwesomeNamespace", "MyAwesomeEnv", 0),
                ("MyAwesomeEnv-v0", null, "MyAwesomeEnv", 0),
                ("MyAwesomeEnv", null, "MyAwesomeEnv", null),
                ("MyAwesomeEnv-vfinal-v0", null, "MyAwesomeEnv-vfinal", 0),
                ("MyAwesomeEnv-vfinal", null, "MyAwesomeEnv-vfinal", null),
                ("MyAwesomeEnv--", null, "MyAwesomeEnv--", null),
                ("MyAwesomeEnv-v", null, "MyAwesomeEnv-v", null),
            };
            foreach ((string envId, string? ns, string name, long? version) in cases) {
                Register(envId, "no-entry-point");
                try {
                    Assert.AreEqual(envId, Spec(envId).Id, envId);

                    string fullName = name;
                    if (!string.IsNullOrEmpty(ns))
                        fullName = $"{ns}/{fullName}";
                    if (version is not null)
                        fullName = $"{fullName}-v{version}";
                    Assert.IsTrue(Registry.ContainsKey(fullName), fullName);
                } finally {
                    Registry.Remove(envId);
                }
            }
        }

        /// <summary>Malformed ids are refused with Gymnasium's message, which quotes the id.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_register.py#L74">test_register_error</see>.</remarks>
        [TestMethod]
        public void TestRegisterError() {
            foreach (string envId in new[] { "“Breakout-v0”", "MyNotSoAwesomeEnv-vNone\n", "MyNamespace///MyNotSoAwesomeEnv-vNone" }) {
                Error error = Assert.ThrowsExactly<Error>(() => Register(envId, "no-entry-point"));
                Assert.IsTrue(error.Message.StartsWith($"Malformed environment ID: {envId}", StringComparison.Ordinal), error.Message);
            }
        }

        /// <summary>A mistyped name or namespace is answered with the closest registered one.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_register.py#L80">test_env_suggestions</see>.</remarks>
        [TestMethod]
        public void TestEnvSuggestions() {
            using IDisposable fixture = RegisterRegistrationTestingEnvs();
            (string Input, string Suggested)[] cases = {
                ("cartpole-v1", "CartPole"),
                ("MyAwesomeNamspce/MyAwesomeVersionedEnv-v1", "MyAwesomeNamespace"),
                ("MyAwesomeNamspce/MyAwesomeUnversionedEnv", "MyAwesomeNamespace"),
                ("MyAwesomeNamespace/MyAwesomeUnversioneEnv", "MyAwesomeUnversionedEnv"),
                ("MyAwesomeNamespace/MyAwesomeVersioneEnv", "MyAwesomeVersionedEnv"),
            };
            foreach ((string input, string suggested) in cases) {
                UnregisteredEnv error = Assert.Throws<UnregisteredEnv>(() => Make(input));
                StringAssert.Contains(error.Message, $"Did you mean: `{suggested}`", input);
            }
        }

        /// <summary>The suggestions that name envs of later phases (Blackjack and Taxi in P5, MountainCarContinuous in P4).</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_register.py#L80">test_env_suggestions</see>.</remarks>
        [TestMethod]
        [OpenBugs("P4", "P5")]
        public void TestEnvSuggestions_LaterPhaseEnvs() {
            (string Input, string Suggested)[] cases = {
                ("blackjack-v1", "Blackjack"),
                ("Blackjock-v1", "Blackjack"),
                ("mountaincarcontinuous-v0", "MountainCarContinuous"),
                ("taxi-v4", "Taxi"),
                ("taxi-v40", "Taxi"),
            };
            foreach ((string input, string suggested) in cases) {
                UnregisteredEnv error = Assert.Throws<UnregisteredEnv>(() => Make(input));
                StringAssert.Contains(error.Message, $"Did you mean: `{suggested}`", input);
            }
        }

        /// <summary>An unregistered version lists the registered ones, or names the default version of an unversioned env.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_register.py#L103">test_env_version_suggestions</see>.</remarks>
        [TestMethod]
        public void TestEnvVersionSuggestions() {
            using IDisposable fixture = RegisterRegistrationTestingEnvs();
            (string Input, string Suggested)[] cases = {
                ("CartPole-v12", "`v0`, `v1`"),
                ("MyAwesomeNamespace/MyAwesomeVersionedEnv-v6", "`v1`, `v3`, `v5`"),
            };
            foreach ((string input, string suggested) in cases) {
                UnregisteredEnv error = Assert.Throws<UnregisteredEnv>(() => Make(input));
                StringAssert.Contains(error.Message, $"It provides versioned environments: [ {suggested} ]", input);
            }

            DeprecatedEnv deprecated = Assert.ThrowsExactly<DeprecatedEnv>(() => Make("MyAwesomeNamespace/MyAwesomeUnversionedEnv-v6"));
            StringAssert.Contains(deprecated.Message, "It provides the default version");
        }

        /// <summary>The version suggestions that name envs of later phases.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_register.py#L103">test_env_version_suggestions</see>.</remarks>
        [TestMethod]
        [OpenBugs("P4", "P5")]
        public void TestEnvVersionSuggestions_LaterPhaseEnvs() {
            (string Input, string Suggested)[] cases = {
                ("Blackjack-v10", "`v1`"),
                ("MountainCarContinuous-v100", "`v0`"),
                ("Taxi-v40", "`v4`"),
            };
            foreach ((string input, string suggested) in cases) {
                UnregisteredEnv error = Assert.Throws<UnregisteredEnv>(() => Make(input));
                StringAssert.Contains(error.Message, $"It provides versioned environments: [ {suggested} ]", input);
            }
        }

        /// <summary>A name is registered either with versions or without, never both, in either order.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_register.py#L134">test_register_versioned_unversioned</see>.</remarks>
        [TestMethod]
        public void TestRegisterVersionedUnversioned() {
            // Register versioned then unversioned.
            const string versionedEnv = "Test/MyEnv-v0";
            Register(versionedEnv, "no-entry-point");
            Assert.AreEqual(versionedEnv, Spec(versionedEnv).Id);

            const string unversionedEnv = "Test/MyEnv";
            try {
                StringAssert.Contains(Assert.ThrowsExactly<RegistrationError>(() => Register(unversionedEnv, "no-entry-point")).Message,
                    "Can't register the unversioned environment `Test/MyEnv` when the versioned environment `Test/MyEnv-v0` of the same name already exists");
            } finally {
                Registry.Remove(versionedEnv);
            }

            // Register unversioned then versioned.
            Register(unversionedEnv, "no-entry-point");
            try {
                Assert.AreEqual(unversionedEnv, Spec(unversionedEnv).Id);
                StringAssert.Contains(Assert.ThrowsExactly<RegistrationError>(() => Register(versionedEnv, "no-entry-point")).Message,
                    "Can't register the versioned environment `Test/MyEnv-v0` when the unversioned environment `Test/MyEnv` of the same name already exists.");
            } finally {
                Registry.Remove(unversionedEnv);
            }
        }

        /// <summary>An unversioned <c>make</c> of a versioned name builds the latest version, with Gymnasium's warning.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_register.py#L168">test_make_latest_versioned_env</see>.</remarks>
        [TestMethod]
        public void TestMakeLatestVersionedEnv() {
            using IDisposable fixture = RegisterRegistrationTestingEnvs();
            using WarningCatcher caught = Warnings.Catch();
            using IEnv env = Make("MyAwesomeNamespace/MyAwesomeVersionedEnv");
            CollectionAssert.Contains(Messages(caught),
                "\u001b[33mWARN: Using the latest versioned environment `MyAwesomeNamespace/MyAwesomeVersionedEnv-v5` instead of the unversioned environment `MyAwesomeNamespace/MyAwesomeVersionedEnv`.\u001b[0m");
            Assert.IsNotNull(env.Spec);
            Assert.AreEqual("MyAwesomeNamespace/MyAwesomeVersionedEnv-v5", env.Spec.Id);
        }

        /// <summary>The namespace scope puts the ids registered inside it in its namespace, and only those.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_register.py#L181">test_namespace</see>.</remarks>
        [TestMethod]
        public void TestNamespace() {
            using (EnvRegistry.Namespace("MyDefaultNamespace"))
                Register("MyDefaultEnvironment-v0", "no-entry-point");
            Register("MyDefaultEnvironment-v1", "no-entry-point");
            try {
                Assert.IsTrue(Registry.ContainsKey("MyDefaultNamespace/MyDefaultEnvironment-v0"));
                Assert.IsTrue(Registry.ContainsKey("MyDefaultEnvironment-v1"));
            } finally {
                Registry.Remove("MyDefaultNamespace/MyDefaultEnvironment-v0");
                Registry.Remove("MyDefaultEnvironment-v1");
            }
        }

        /// <summary>The messages a catcher recorded.</summary>
        /// <param name="caught">The catcher.</param>
        /// <returns>The messages, in order.</returns>
        private static List<string> Messages(WarningCatcher caught) {
            var messages = new List<string>();
            foreach (WarningMessage warning in caught.Records)
                messages.Add(warning.Message);
            return messages;
        }

        /// <summary>A fixture's teardown, run once when disposed.</summary>
        private sealed class Teardown : IDisposable {
            /// <summary>The cleanup, cleared once run.</summary>
            private Action? _cleanup;

            /// <summary>Holds the cleanup.</summary>
            /// <param name="cleanup">What to run on dispose.</param>
            public Teardown(Action cleanup) => _cleanup = cleanup;

            /// <summary>Runs the cleanup once.</summary>
            public void Dispose() {
                Action? cleanup = _cleanup;
                _cleanup = null;
                cleanup?.Invoke();
            }
        }
    }
}
