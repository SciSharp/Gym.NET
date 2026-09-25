#nullable enable
using System;
using System.Collections.Generic;
using System.Numerics;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Observations;
using Gym.Tests.Ported;
using Gym.Utils;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Envs {
    /// <summary>
    ///     Pins the <see cref="Wrapper{TWObs, TWAct, TObs, TAct}"/> behaviour <c>test_core.py</c> does not reach: spec
    ///     propagation (Gymnasium 1.4.0's output for the same stacks is quoted), read-only properties, descriptions, the
    ///     pass-through of mismatched types, and <c>TimeLimit</c>/<c>OrderEnforcing</c>'s recorded arguments.
    /// </summary>
    [TestClass]
    public class WrapperTests {
        /// <summary>
        ///     A plain wrapper appends <c>WrapperSpec(name, "Namespace:Class", kwargs=None)</c> to a deep copy of the env's spec
        ///     (Python: <c>(WrapperSpec(name='W', entry_point='__main__:W', kwargs=None),)</c>); <c>TimeLimit</c> and
        ///     <c>OrderEnforcing</c> edit the spec instead of appending.
        /// </summary>
        [TestMethod]
        public void Spec_PropagatesUpTheStack() {
            var made = new TimeLimit<NDArray, long>(new OrderEnforcing<NDArray, long>(new CartPoleEnv { Spec = MadeSpec() }), 500);

            var plain = new PlainWrapper(made);
            EnvSpec spec = plain.Spec!;
            Assert.AreEqual(1, spec.AdditionalWrappers.Count);
            Assert.AreEqual(new WrapperSpec("PlainWrapper", "Gym.Tests.Envs:PlainWrapper", null), spec.AdditionalWrappers[0]);
            Assert.AreEqual(500L, spec.MaxEpisodeSteps);
            Assert.IsTrue(spec.OrderEnforce);
            Assert.AreEqual(0, made.Spec!.AdditionalWrappers.Count, "The wrapped env's spec is a copy, untouched.");
            Assert.AreSame(spec, plain.Spec, "The spec is cached (_cached_spec).");

            Assert.AreEqual(7L, new TimeLimit<NDArray, long>(made, 7).Spec!.MaxEpisodeSteps);
            Assert.AreEqual(0, new OrderEnforcing<NDArray, long>(made).Spec!.AdditionalWrappers.Count);
            Assert.IsNull(new PlainWrapper(new TimeLimit<NDArray, long>(new CartPoleEnv(), 5)).Spec, "No spec beneath, no spec.");
        }

        /// <summary>A wrapper that records its arguments puts them in its spec, without <c>env</c>.</summary>
        [TestMethod]
        public void Spec_RecordsConstructorArguments() {
            var recording = new RecordingWrapper(new CartPoleEnv { Spec = MadeSpec() }, scale: 2.5);

            WrapperSpec wrapperSpec = recording.Spec!.AdditionalWrappers[0];
            Assert.AreEqual("{'scale': 2.5}", wrapperSpec.Kwargs!.ToString());
            Assert.IsTrue(recording.SavedKwargs.ContainsKey("env"), "The saved arguments keep env; only the spec drops it.");
            Assert.AreEqual("{'max_episode_steps': 10}", new TimeLimit<NDArray, long>(new CartPoleEnv(), 10).SavedKwargs.ToString());
            Assert.AreEqual("{'disable_render_order_enforcing': False}", new OrderEnforcing<NDArray, long>(new CartPoleEnv()).SavedKwargs.ToString());
        }

        /// <summary><c>spec</c> and <c>render_mode</c> are read-only on a wrapper, with Python's messages.</summary>
        [TestMethod]
        public void ReadOnlyProperties_RaiseAttributeError() {
            var wrapper = new TimeLimit<NDArray, long>(new CartPoleEnv(), 5);

            Assert.AreEqual("property 'spec' of 'TimeLimit' object has no setter", Assert.ThrowsExactly<AttributeError>(() => wrapper.Spec = null).Message);
            Assert.AreEqual("property 'render_mode' of 'TimeLimit' object has no setter", Assert.ThrowsExactly<AttributeError>(() => PyAttributes.SetAttr(wrapper, "render_mode", "human")).Message);
            Assert.AreEqual("property 'spec' of 'TimeLimit' object has no setter", Assert.ThrowsExactly<AttributeError>(() => PyAttributes.SetAttr(wrapper, "spec", null)).Message);
        }

        /// <summary>
        ///     <c>str(env)</c> as Gymnasium prints it (<c>&lt;TimeLimit&lt;OrderEnforcing&lt;CartPoleEnv instance&gt;&gt;&gt;</c>),
        ///     the class name, and what forwards to the innermost env.
        /// </summary>
        [TestMethod]
        public void Descriptions_And_Forwarding() {
            var cartPole = new CartPoleEnv();
            var env = new TimeLimit<NDArray, long>(new OrderEnforcing<NDArray, long>(cartPole), 10);

            Assert.AreEqual("<TimeLimit<OrderEnforcing<CartPoleEnv instance>>>", env.ToString());
            Assert.AreEqual("TimeLimit", env.ClassName());
            Assert.AreSame(cartPole, env.Unwrapped);
            env.Reset(seed: 3);
            Assert.AreEqual(new BigInteger(3), cartPole.NpRandomSeed);
            Assert.AreSame(cartPole.NpRandom, env.NpRandom);
            Assert.AreSame(cartPole.Metadata, env.Metadata);
            Assert.AreSame(cartPole.ActionSpace, env.ActionSpace);
            cartPole.Spec = MadeSpec();
            Assert.AreEqual("<CartPoleEnv<CartPole-v1>>", cartPole.ToString());
        }

        /// <summary>A <c>TimeLimit</c> stepped before any reset steps the env, then raises Python's <c>None += 1</c> error.</summary>
        [TestMethod]
        public void TimeLimit_StepBeforeReset_RaisesTypeErrorAfterStepping() {
            int steps = 0;
            var counting = new GenericTestEnv(stepFunc: (self, action) => {
                steps++;
                return TestingEnv.BasicStepFunc(self, action);
            });
            var env = new TimeLimit<object?, object?>(counting, 3);

            Assert.AreEqual("unsupported operand type(s) for +=: 'NoneType' and 'int'", Assert.ThrowsExactly<TypeError>(() => env.Step(0)).Message);
            Assert.AreEqual(1, steps);
            Assert.IsNull(env.ElapsedSteps);
            env.Reset();
            env.Step(0);
            Assert.AreEqual(1L, env.ElapsedSteps);
            Assert.AreEqual(1L, env.GetWrapperAttr("_elapsed_steps"));
        }

        /// <summary>A wrapper whose types differ from its env's must convert; the default pass-through says so instead of corrupting data.</summary>
        [TestMethod]
        public void PassThrough_OfMismatchedTypes_Throws() {
            var mismatched = new Wrapper<string, long, NDArray, long>(new CartPoleEnv());

            var error = Assert.ThrowsExactly<InvalidCastException>(() => mismatched.Reset(seed: 0));
            StringAssert.Contains(error.Message, "override Step and Reset");
        }

        /// <summary>A missing env raises Gymnasium's <c>TypeError</c>.</summary>
        [TestMethod]
        public void NullEnv_RaisesTypeError() {
            Assert.AreEqual("Expected env to be a `gymnasium.Env` but got <class 'NoneType'>", Assert.ThrowsExactly<TypeError>(() => new Wrapper<NDArray, long>(null!)).Message);
        }

        /// <summary>The <c>make</c>-style spec of the unwrapped env: the wrappers' settings cleared, so each wrapper puts its own back.</summary>
        /// <returns>The spec.</returns>
        private static EnvSpec MadeSpec() => new("CartPole-v1", "Gym.Environments.Envs.Classic:CartPoleEnv", rewardThreshold: 475.0, orderEnforce: false, disableEnvChecker: true);

        /// <summary>A wrapper that changes nothing and records no arguments.</summary>
        private sealed class PlainWrapper : Wrapper<NDArray, long> {
            /// <summary>Wraps <paramref name="env"/>.</summary>
            /// <param name="env">The env.</param>
            public PlainWrapper(Env<NDArray, long> env) : base(env) { }
        }

        /// <summary>A wrapper that records its arguments, including <c>env</c>, as some Gymnasium wrappers do.</summary>
        private sealed class RecordingWrapper : Wrapper<NDArray, long>, IRecordConstructorArgs {
            /// <summary>Wraps <paramref name="env"/>.</summary>
            /// <param name="env">The env.</param>
            /// <param name="scale">An argument to record.</param>
            public RecordingWrapper(Env<NDArray, long> env, double scale) : base(env) {
                SavedKwargs = RecordConstructorArgs.Record(new Info { ["env"] = env, ["scale"] = scale }, disableDeepcopy: true);
            }

            /// <summary>The recorded arguments.</summary>
            [PyName("_saved_kwargs")]
            public Info SavedKwargs { get; }
        }
    }
}
