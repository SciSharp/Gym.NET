#nullable enable
using System;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Exceptions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests.Envs.Registration {
    /// <summary>
    ///     Pins <see cref="EnvSpec"/>, <see cref="WrapperSpec"/> and the id grammar to Gymnasium 1.4.0: every expected string
    ///     below was printed by the Python source (<c>repr</c>, <c>pprint</c>, <c>parse_env_id</c>).
    /// </summary>
    [TestClass]
    public class EnvSpecTests {
        /// <summary>
        ///     <c>ParseEnvId</c> splits namespaced, versioned and unversioned ids as Gymnasium's <c>parse_env_id</c> does, and
        ///     <c>GetEnvId</c> joins them back.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/envs/registration.py#L259">gymnasium/envs/registration.py (parse_env_id)</see>.</remarks>
        [TestMethod]
        public void ParseEnvId_SplitsLikeGymnasium() {
            Assert.AreEqual((null, "CartPole", 1L), EnvRegistry.ParseEnvId("CartPole-v1"));
            Assert.AreEqual(("ALE", "Pong", 5L), EnvRegistry.ParseEnvId("ALE/Pong-v5"));
            Assert.AreEqual((null, "MyEnv", null), EnvRegistry.ParseEnvId("MyEnv"));
            Assert.AreEqual(("ns", "My.Env", 0L), EnvRegistry.ParseEnvId("ns/My.Env-v0"));
            Assert.AreEqual(("my:ns", "Name", 12L), EnvRegistry.ParseEnvId("my:ns/Name-v12"));
            Assert.AreEqual((null, "Name-v1", 2L), EnvRegistry.ParseEnvId("Name-v1-v2"));
            Assert.AreEqual((null, "Name-v", null), EnvRegistry.ParseEnvId("Name-v"));

            Assert.AreEqual("ALE/Pong-v5", EnvRegistry.GetEnvId("ALE", "Pong", 5));
            Assert.AreEqual("MyEnv", EnvRegistry.GetEnvId(null, "MyEnv", null));
        }

        /// <summary>A malformed id raises <see cref="Error"/> with Gymnasium's message.</summary>
        /// <param name="id">The id.</param>
        [TestMethod]
        [DataRow("a/b/c-v1")]
        [DataRow("")]
        [DataRow("CartPole v1")]
        [DataRow("CartPole-v1\n")]
        public void ParseEnvId_Malformed_RaisesError(string id) {
            var error = Assert.ThrowsExactly<Error>(() => EnvRegistry.ParseEnvId(id));
            Assert.AreEqual($"Malformed environment ID: {id}. (Currently all IDs must be of the form [namespace/](env-name)-v(version). (namespace is optional))", error.Message);
        }

        /// <summary>The dataclass repr, field for field, and <c>pprint</c>'s three forms.</summary>
        [TestMethod]
        public void Repr_And_Pprint_AreGymnasiums() {
            var spec = new EnvSpec("CartPole-v1", "gymnasium.envs.classic_control.cartpole:CartPoleEnv", rewardThreshold: 475.0, maxEpisodeSteps: 500);

            Assert.AreEqual("EnvSpec(id='CartPole-v1', entry_point='gymnasium.envs.classic_control.cartpole:CartPoleEnv', reward_threshold=475.0, nondeterministic=False, max_episode_steps=500, order_enforce=True, disable_env_checker=False, kwargs={}, namespace=None, name='CartPole', version=1, additional_wrappers=(), vector_entry_point=None)", spec.ToString());
            Assert.AreEqual("id=CartPole-v1\nreward_threshold=475.0\nmax_episode_steps=500", spec.Pprint(disablePrint: true));
            Assert.AreEqual("id=CartPole-v1\nentry_point=gymnasium.envs.classic_control.cartpole:CartPoleEnv\nreward_threshold=475.0\nnondeterministic=False\nmax_episode_steps=500\norder_enforce=True\ndisable_env_checker=False\nadditional_wrappers=[]",
                spec.Pprint(disablePrint: true, printAll: true));

            var wrapper = new WrapperSpec("TimeLimit", "gymnasium.wrappers.common:TimeLimit", new Info { ["max_episode_steps"] = 10L });
            var spec2 = new EnvSpec("ns/Name-v3", additionalWrappers: new[] { wrapper }, kwargs: new Info { ["a"] = 1.5, ["b"] = new System.Collections.Generic.List<object?> { 1L, "x" } },
                nondeterministic: true, orderEnforce: false);
            Assert.AreEqual("EnvSpec(id='ns/Name-v3', entry_point=None, reward_threshold=None, nondeterministic=True, max_episode_steps=None, order_enforce=False, disable_env_checker=False, kwargs={'a': 1.5, 'b': [1, 'x']}, namespace='ns', name='Name', version=3, additional_wrappers=(WrapperSpec(name='TimeLimit', entry_point='gymnasium.wrappers.common:TimeLimit', kwargs={'max_episode_steps': 10}),), vector_entry_point=None)",
                spec2.ToString());
            Assert.AreEqual("id=ns/Name-v3\nentry_point=None\nnondeterministic=True\norder_enforce=False\nadditional_wrappers=[\n\tname=TimeLimit, entry_point=gymnasium.wrappers.common:TimeLimit, kwargs={'max_episode_steps': 10}\n]",
                spec2.Pprint(disablePrint: true, includeEntryPoints: true));
            Assert.AreEqual("WrapperSpec(name='TimeLimit', entry_point='gymnasium.wrappers.common:TimeLimit', kwargs={'max_episode_steps': 10})", wrapper.ToString());
        }

        /// <summary>
        ///     Equality is the dataclass's (kwargs as dicts); a deep copy is equal but shares nothing mutable, and the parsed
        ///     fields are not re-parsed when the id changes, as in Python.
        /// </summary>
        [TestMethod]
        public void Equality_And_DeepCopy() {
            var spec = new EnvSpec("CartPole-v1", kwargs: new Info { ["x"] = 1 }, additionalWrappers: new[] { new WrapperSpec("W", "m:W", new Info { ["k"] = 2 }) });
            EnvSpec copy = spec.DeepCopy();

            Assert.AreEqual(spec, copy);
            Assert.AreEqual(spec, new EnvSpec("CartPole-v1", kwargs: new Info { ["x"] = 1.0 }, additionalWrappers: new[] { new WrapperSpec("W", "m:W", new Info { ["k"] = 2L }) }));
            Assert.AreNotSame(spec.Kwargs, copy.Kwargs);
            Assert.AreNotSame(spec.AdditionalWrappers[0].Kwargs, copy.AdditionalWrappers[0].Kwargs);
            copy.Kwargs["x"] = 5;
            copy.MaxEpisodeSteps = 3;
            Assert.AreEqual(1, spec.Kwargs["x"]);
            Assert.IsNull(spec.MaxEpisodeSteps);
            Assert.AreNotEqual(spec, copy);

            copy.Id = "Other-v7";
            Assert.AreEqual("CartPole", copy.Name, "__post_init__ ran once, at construction.");
            Assert.AreEqual(1L, copy.Version);
        }

        /// <summary><see cref="WrapperSpec.For"/> names a wrapper type by its class and namespace, without generic arity.</summary>
        [TestMethod]
        public void WrapperSpecFor_NamesTheType() {
            WrapperSpec spec = WrapperSpec.For(typeof(Gym.Wrappers.TimeLimit<,>), new Info { ["max_episode_steps"] = 5L });

            Assert.AreEqual("TimeLimit", spec.Name);
            Assert.AreEqual("Gym.Wrappers:TimeLimit", spec.EntryPoint);
        }
    }
}
