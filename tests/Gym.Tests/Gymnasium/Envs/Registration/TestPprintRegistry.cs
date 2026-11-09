#nullable enable
using System.Collections.Generic;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Envs.Registration;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests.Gymnasium.Envs.Registration {
    /// <summary>
    ///     Gymnasium's <c>pprint_registry</c> tests (<c>tests/envs/registration/test_pprint_registry.py</c>): the default
    ///     registry prints, and example registries print Gymnasium's exact text, by namespace and in columns, with namespaces
    ///     excluded.
    /// </summary>
    /// <remarks>
    ///     The example registries use Gymnasium's own entry-point strings: the namespaces come from their module paths
    ///     (<c>classic_control</c>), which is what the printing logic reads. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_pprint_registry.py">tests/envs/registration/test_pprint_registry.py</see>.
    /// </remarks>
    [TestClass]
    public class TestPprintRegistry {
        /// <summary>Gymnasium's CartPole entry point, whose module path names the <c>classic_control</c> namespace.</summary>
        private const string ExampleEntryPoint = "gymnasium.envs.classic_control.cartpole:CartPoleEnv";

        /// <summary>The default registry prints something.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_pprint_registry.py#L14">test_pprint_default_registry</see>.</remarks>
        [TestMethod]
        public void TestPprintDefaultRegistry() {
            string? output = PprintRegistry(disablePrint: true);
            Assert.IsFalse(string.IsNullOrEmpty(output));
        }

        /// <summary>A registry other than the default prints its ids under their namespace, three to a line.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_pprint_registry.py#L19">test_pprint_example_registry</see>.</remarks>
        [TestMethod]
        public void TestPprintExampleRegistry() {
            var exampleRegistry = new EnvSpecRegistry {
                ["CartPole-v0"] = new EnvSpec("CartPole-v0", ExampleEntryPoint),
                ["CartPole-v1"] = new EnvSpec("CartPole-v1", ExampleEntryPoint),
                ["CartPole-v2"] = new EnvSpec("CartPole-v2", ExampleEntryPoint),
                ["CartPole-v3"] = new EnvSpec("CartPole-v3", ExampleEntryPoint),
            };

            Assert.AreEqual("===== classic_control =====\nCartPole-v0 CartPole-v1 CartPole-v2\nCartPole-v3", PprintRegistry(exampleRegistry, disablePrint: true));
        }

        /// <summary>
        ///     The namespace of an id without one comes from its entry point's module path (its third or second part), and is
        ///     <c>None</c> for a function entry point.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_pprint_registry.py#L33">test_pprint_namespace</see>.</remarks>
        [TestMethod]
        public void TestPprintNamespace() {
            var exampleRegistry = new EnvSpecRegistry {
                ["CartPole-v0"] = new EnvSpec("CartPole-v0", "gymnasium.envs.classic_control.cartpole:CartPoleEnv"),
                ["CartPole-v1"] = new EnvSpec("CartPole-v1", "gymnasium.envs.classic_control:CartPoleEnv"),
                ["CartPole-v2"] = new EnvSpec("CartPole-v2", "gymnasium.cartpole:CartPoleEnv"),
                ["CartPole-v3"] = new EnvSpec("CartPole-v3", (EnvCreator)(_ => new CartPoleEnv())),
                ["ExampleNamespace/CartPole-v2"] = new EnvSpec("ExampleNamespace/CartPole-v2", "gymnasium.envs.classic_control:CartPoleEnv"),
            };

            Assert.AreEqual("===== classic_control =====\nCartPole-v0 CartPole-v1\n===== cartpole =====\nCartPole-v2\n===== None =====\nCartPole-v3\n"
                            + "===== ExampleNamespace =====\nExampleNamespace/CartPole-v2", PprintRegistry(exampleRegistry, disablePrint: true));
        }

        /// <summary>The number of columns per line.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_pprint_registry.py#L58">test_pprint_n_columns</see>.</remarks>
        [TestMethod]
        public void TestPprintNColumns() {
            var exampleRegistry = new EnvSpecRegistry {
                ["CartPole-v0"] = new EnvSpec("CartPole-v0", ExampleEntryPoint),
                ["CartPole-v1"] = new EnvSpec("CartPole-v1", ExampleEntryPoint),
                ["CartPole-v2"] = new EnvSpec("CartPole-v2", ExampleEntryPoint),
                ["CartPole-v3"] = new EnvSpec("CartPole-v3", ExampleEntryPoint),
            };

            Assert.AreEqual("===== classic_control =====\nCartPole-v0 CartPole-v1\nCartPole-v2 CartPole-v3", PprintRegistry(exampleRegistry, numCols: 2, disablePrint: true));
            Assert.AreEqual("===== classic_control =====\nCartPole-v0 CartPole-v1 CartPole-v2 CartPole-v3", PprintRegistry(exampleRegistry, numCols: 5, disablePrint: true));
        }

        /// <summary>Excluded namespaces are left out, whether given or derived from the entry point.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_pprint_registry.py#L80">test_pprint_exclude_namespace</see>.</remarks>
        [TestMethod]
        public void TestPprintExcludeNamespace() {
            var exampleRegistry = new EnvSpecRegistry {
                ["Test/CartPole-v0"] = new EnvSpec("Test/CartPole-v0", ExampleEntryPoint),
                ["Test/CartPole-v1"] = new EnvSpec("Test/CartPole-v1", ExampleEntryPoint),
                ["CartPole-v2"] = new EnvSpec("CartPole-v2", ExampleEntryPoint),
                ["CartPole-v3"] = new EnvSpec("CartPole-v3", ExampleEntryPoint),
            };

            Assert.AreEqual("===== classic_control =====\nCartPole-v2 CartPole-v3", PprintRegistry(exampleRegistry, excludeNamespaces: new[] { "Test" }, disablePrint: true));
            Assert.AreEqual("===== Test =====\nTest/CartPole-v0 Test/CartPole-v1", PprintRegistry(exampleRegistry, excludeNamespaces: new[] { "classic_control" }, disablePrint: true));

            exampleRegistry["Example/CartPole-v4"] = new EnvSpec("Example/CartPole-v4", ExampleEntryPoint);
            Assert.AreEqual("===== classic_control =====\nCartPole-v2 CartPole-v3",
                PprintRegistry(exampleRegistry, excludeNamespaces: new List<string?> { "Test", "Example" }, disablePrint: true));
        }
    }
}
