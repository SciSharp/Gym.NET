#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Gym.Environments;
using Gym.Environments.Envs.Classic;
using Gym.Envs.Registration;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests.Envs.Registration {
    /// <summary>
    ///     Pins DR-88: <c>Gymnasium.Envs</c> holds one constant per id Gym.NET.Environments registers, each named after its
    ///     id (<c>"CartPole-v1"</c> is <c>CartPoleV1</c>) and holding the id's exact string, and nothing else.
    /// </summary>
    [TestClass]
    public class EnvIdsTests {
        /// <summary>The constants, by name; read through reflection so a constant added or removed shows here without a list to keep.</summary>
        /// <returns>Each constant's name and value.</returns>
        private static Dictionary<string, string> Constants() =>
            typeof(global::Gym.Gymnasium.Envs).GetFields(BindingFlags.Public | BindingFlags.Static)
                .Where(f => f.IsLiteral && f.FieldType == typeof(string))
                .ToDictionary(f => f.Name, f => (string)f.GetRawConstantValue()!);

        /// <summary>
        ///     The ids whose entry point is one of Gym.NET.Environments' classes: the built-in registrations, whatever other
        ///     tests registered beside them.
        /// </summary>
        /// <returns>The ids.</returns>
        private static HashSet<string> BuiltInIds() {
            GymEnvironments.Register();
            Assembly environments = typeof(CartPoleEnv).Assembly;
            // Ids other tests register live in namespaces ("test/..."); the built-in ones have none.
            return EnvRegistry.Registry.Where(pair => !pair.Key.Contains('/') && pair.Value.EntryPoint is Type type && type.Assembly == environments)
                .Select(pair => pair.Key).ToHashSet(StringComparer.Ordinal);
        }

        /// <summary>Every constant names a registered built-in id, and every built-in id has its constant.</summary>
        [TestMethod]
        public void TheConstants_AreTheBuiltInIds() {
            HashSet<string> ids = BuiltInIds();
            HashSet<string> values = Constants().Values.ToHashSet(StringComparer.Ordinal);
            CollectionAssert.AreEquivalent(ids.OrderBy(id => id, StringComparer.Ordinal).ToList(), values.OrderBy(id => id, StringComparer.Ordinal).ToList(),
                $"missing constants: {string.Join(", ", ids.Except(values))}; constants of no built-in id: {string.Join(", ", values.Except(ids))}");
            Assert.AreEqual(39, values.Count, "the 39 ids, Pusher-v4 included (it refuses at construction, as Gymnasium's does)");
        }

        /// <summary>Each constant is named after its id: the hyphen dropped and the version's <c>v</c> capitalised.</summary>
        [TestMethod]
        public void EachConstant_IsNamedAfterItsId() {
            foreach ((string name, string id) in Constants()) {
                int dash = id.LastIndexOf("-v", StringComparison.Ordinal);
                Assert.IsTrue(dash > 0, $"{id} has a version");
                Assert.AreEqual(id.Substring(0, dash) + "V" + id.Substring(dash + 2), name, $"the constant for {id}");
            }
        }

        /// <summary>A constant is the id itself: <c>Make</c> builds the same stack from it as from the string.</summary>
        [TestMethod]
        public void AConstant_MakesTheEnvItNames() {
            using var env = global::Gym.Gymnasium.Make(global::Gym.Gymnasium.Envs.CartPoleV1);
            Assert.AreEqual("CartPole-v1", env.Spec!.Id);
            // The type compared, not cast: the env owns its inner env and disposes it with itself.
            Assert.AreEqual(typeof(CartPoleEnv), env.Unwrapped.GetType());
        }
    }
}
