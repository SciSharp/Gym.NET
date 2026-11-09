#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Exceptions;
using Gym.Tests.Gymnasium.Wrappers;
using Gym.Utils;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Envs.Registration {
    /// <summary>
    ///     Gymnasium's <c>make</c> tests (<c>tests/envs/registration/test_make.py</c>): ids and specs alike, the
    ///     <c>max_episode_steps</c>, <c>disable_env_checker</c> and order-enforcing switches, render modes (native,
    ///     <c>RenderCollection</c>, the old rendering API), kwargs, re-making from a spec, and the id errors.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Parts that need later phases are the <c>[OpenBugs]</c> methods: the <c>HumanRendering</c>, <c>NormalizeReward</c>,
    ///     <c>TimeAwareObservation</c> and <c>FlattenObservation</c> wrappers (P6) and CarRacing (P8). Gymnasium's
    ///     <c>Humanoid-v0</c> deprecation case is not ported: MuJoCo is postponed beyond 2.0 (DR-35).
    ///     </para>
    ///     <para>
    ///     <c>make("tests.envs.registration.utils_unregistered_env:RegisterDuringMake-v0")</c> imports a module that registers
    ///     an id; here the module is the <c>Gym.Tests.Plugin</c> assembly, which <c>make</c> loads and whose module initializer
    ///     registers it.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_make.py">tests/envs/registration/test_make.py</see>.</para>
    /// </remarks>
    [TestClass]
    public class TestMake {
        /// <summary>An id and its spec make the same env, with the id's spec.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_make.py#L45">test_no_arguments</see>.</remarks>
        [TestMethod]
        public void TestNoArguments() {
            const string envId = "CartPole-v1";
            using IEnv envFromId = Make(envId);
            Assert.IsNotNull(envFromId.Spec);
            Assert.AreEqual(envId, envFromId.Spec.Id);
            Assert.IsInstanceOfType<CartPoleEnv>(envFromId.Unwrapped);

            EnvSpec envSpec = Spec(envId);
            using IEnv envFromSpec = Make(envSpec);
            Assert.IsNotNull(envFromSpec.Spec);
            Assert.AreEqual(envId, envFromSpec.Spec.Id);
            Assert.IsInstanceOfType<CartPoleEnv>(envFromSpec.Unwrapped);

            Assert.AreEqual(envFromId.Spec, envFromSpec.Spec);
        }

        /// <summary>
        ///     <c>max_episode_steps</c>: the spec's limit by default, the call's when given, none for <c>-1</c> or when neither
        ///     has one.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_make.py#L62">test_max_episode_steps</see>.</remarks>
        [TestMethod]
        public void TestMaxEpisodeSteps() {
            using IDisposable fixture = RegisterParameterEnvs();
            foreach (object makeId in new object[] { "CartPole-v1", Spec("CartPole-v1") }) {
                EnvSpec envSpec = makeId as EnvSpec ?? Spec((string)makeId);

                // Use the spec's value.
                using (IEnv env = MakeFrom(makeId)) {
                    Assert.IsTrue(WrapperUtils.HasWrapper(env, typeof(TimeLimit<,>)));
                    Assert.AreEqual(envSpec.MaxEpisodeSteps, env.Spec!.MaxEpisodeSteps);
                }

                // Set a custom max episode steps value.
                Assert.AreNotEqual(100L, envSpec.MaxEpisodeSteps);
                using (IEnv env = MakeFrom(makeId, maxEpisodeSteps: 100)) {
                    Assert.IsTrue(WrapperUtils.HasWrapper(env, typeof(TimeLimit<,>)));
                    Assert.AreEqual(100L, env.Spec!.MaxEpisodeSteps, makeId.ToString());
                }
            }

            foreach (object makeId in new object[] { "NoMaxEpisodeStepsEnv-v0", Spec("NoMaxEpisodeStepsEnv-v0") }) {
                EnvSpec envSpec = makeId as EnvSpec ?? Spec((string)makeId);

                // The env spec has no max episode steps.
                Assert.IsNull(envSpec.MaxEpisodeSteps);
                using (IEnv env = MakeFrom(makeId)) {
                    Assert.IsNull(env.Spec!.MaxEpisodeSteps);
                    Assert.IsFalse(WrapperUtils.HasWrapper(env, typeof(TimeLimit<,>)));
                }

                // Set a custom max episode steps value.
                using (IEnv env = MakeFrom(makeId, maxEpisodeSteps: 100)) {
                    Assert.AreEqual(100L, env.Spec!.MaxEpisodeSteps);
                    Assert.IsTrue(WrapperUtils.HasWrapper(env, typeof(TimeLimit<,>)));
                }
            }

            // Override max_episode_steps to prevent applying the wrapper.
            foreach (object envId in new object[] { "CartPole-v1", Spec("CartPole-v1"), "NoMaxEpisodeStepsEnv-v0", Spec("NoMaxEpisodeStepsEnv-v0") }) {
                using IEnv env = MakeFrom(envId, maxEpisodeSteps: -1);
                Assert.IsNull(env.Spec!.MaxEpisodeSteps);
                Assert.IsFalse(WrapperUtils.HasWrapper(env, typeof(TimeLimit<,>)));
            }
        }

        /// <summary>
        ///     <c>disable_env_checker</c>: the call's value when it is a bool, else the registration's, for ids and specs alike.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_make.py#L113">test_disable_env_checker</see>.</remarks>
        [TestMethod]
        public void TestDisableEnvChecker() {
            (bool RegistrationDisabled, bool? MakeDisabled, bool IfDisabled)[] cases = {
                (false, false, false),
                (false, true, true),
                (true, false, false),
                (true, true, true),
                (false, null, false),
                (true, null, true),
            };
            foreach ((bool registrationDisabled, bool? makeDisabled, bool ifDisabled) in cases) {
                Register("DisableEnvCheckerEnv-v0", (EnvCreator)(_ => new GenericTestEnv()), disableEnvChecker: registrationDisabled);
                try {
                    using (IEnv env = Make("DisableEnvCheckerEnv-v0", disableEnvChecker: makeDisabled))
                        Assert.AreEqual(!ifDisabled, WrapperUtils.HasWrapper(env, typeof(PassiveEnvChecker<,>)), $"{registrationDisabled}, {makeDisabled}");

                    EnvSpec envSpec = Spec("DisableEnvCheckerEnv-v0");
                    using (IEnv env = Make(envSpec, disableEnvChecker: makeDisabled))
                        Assert.AreEqual(!ifDisabled, WrapperUtils.HasWrapper(env, typeof(PassiveEnvChecker<,>)), $"{registrationDisabled}, {makeDisabled}");
                } finally {
                    Registry.Remove("DisableEnvCheckerEnv-v0");
                }
            }
        }

        /// <summary>
        ///     <c>OrderEnforcing</c> follows the spec's <c>order_enforce</c>; the minimal spec of a made env (its unwrapped
        ///     env's) never enforces order.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_make.py#L146">test_order_enforcing</see>.</remarks>
        [TestMethod]
        public void TestOrderEnforcing() {
            using IDisposable fixture = RegisterParameterEnvs();
            Assert.IsTrue(EnvTestingUtils.AllTestingEnvSpecs().All(spec => !spec.OrderEnforce));

            foreach (object makeId in new object[] { "CartPole-v1", Spec("CartPole-v1") }) {
                using IEnv env = MakeFrom(makeId);
                Assert.IsTrue(WrapperUtils.HasWrapper(env, typeof(OrderEnforcing<,>)));
            }

            foreach (object makeId in new object[] { "OrderlessEnv-v0", Spec("OrderlessEnv-v0") }) {
                using IEnv env = MakeFrom(makeId);
                Assert.IsFalse(WrapperUtils.HasWrapper(env, typeof(OrderEnforcing<,>)));
            }
        }

        /// <summary>
        ///     <c>render_mode</c> reaches the env; a mode its metadata does not list is passed on with Gymnasium's warning
        ///     (CartPole does not validate it).
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_make.py#L160">test_make_with_render_mode</see>.</remarks>
        [TestMethod]
        public void TestMakeWithRenderMode() {
            IEnv env = Make("CartPole-v1", new Info { ["render_mode"] = null });
            Assert.IsNull(env.RenderMode);
            env.Dispose();

            Assert.IsTrue(env.Metadata.RenderModes!.Contains("rgb_array"));
            env = Make("CartPole-v1", new { render_mode = "rgb_array" });
            Assert.AreEqual("rgb_array", env.RenderMode);
            env.Dispose();

            Assert.IsFalse(env.Metadata.RenderModes!.Contains("no-render-mode"));
            // CartPole is special in that it doesn't check the render_mode passed at initialisation.
            using WarningCatcher caught = Warnings.Catch();
            using (env = Make("CartPole-v1", new { render_mode = "no-render-mode" }))
                Assert.AreEqual("no-render-mode", env.RenderMode);
            CollectionAssert.Contains(caught.Records.Select(record => record.Message).ToList(),
                "\u001b[33mWARN: The environment is being initialised with render_mode='no-render-mode' that is not in the possible render_modes (['human', 'rgb_array']).\u001b[0m");
        }

        /// <summary>A <c>*_list</c> mode the env renders singly gets <c>RenderCollection</c>, whose render returns the frames.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_make.py#L184">test_make_render_collection</see>.</remarks>
        [TestMethod]
        public void TestMakeRenderCollection() {
            // Make sure that render_mode is applied correctly.
            using IEnv env = Make("CartPole-v1", new { render_mode = "rgb_array_list" });
            Assert.IsTrue(WrapperUtils.HasWrapper(env, typeof(RenderCollection<,>)));
            Assert.AreEqual("rgb_array_list", env.RenderMode);
            Assert.AreEqual("rgb_array", env.Unwrapped.RenderMode);

            env.Reset();
            using RenderResult renders = env.Render();
            // Make sure that the render method does what it is supposed to.
            Assert.IsInstanceOfType<RenderResult.Frames>(renders);
            Assert.IsInstanceOfType<RenderResult.Rgb>(((RenderResult.Frames)renders).Items[0]);
        }

        /// <summary>
        ///     Human rendering: native when the env has it; an env on the old rendering API is refused, with the clearer error
        ///     when <c>human</c> was asked for; an env whose metadata lists no modes warns.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_make.py#L200">test_make_human_rendering</see>.</remarks>
        [TestMethod]
        public void TestMakeHumanRendering() {
            using IDisposable fixture = RegisterRenderingTestingEnvs();

            // Make sure that native rendering is used when possible: no viewer is needed until the first render.
            using (IEnv env = Make("CartPole-v1", new { render_mode = "human" })) {
                Assert.AreEqual("human", env.RenderMode);
                Assert.AreEqual("<TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPoleEnv<CartPole-v1>>>>>", env.ToString());
            }

            StringAssert.Contains(Assert.Throws<TypeError>(() => Make("NoHumanRenderingOldAPI-v0", new { render_mode = "rgb_array_list" })).Message,
                "got an unexpected keyword argument 'render_mode'");

            // An additional error is thrown when a user tries to use the wrapper on an environment with the old API.
            using (Warnings.Catch()) {
                StringAssert.Contains(Assert.ThrowsExactly<Error>(() => Make("NoHumanRenderingOldAPI-v0", new { render_mode = "human" })).Message,
                    "You passed render_mode='human' although NoHumanRenderingOldAPI-v0 doesn't implement human-rendering natively.");
            }

            // A user can create an environment without the metadata including the render mode.
            using (WarningCatcher caught = Warnings.Catch()) {
                using IEnv env = Make("NoRenderModesMetadata-v0", new { render_mode = "rgb_array" });
                CollectionAssert.Contains(caught.Records.Select(record => record.Message).ToList(),
                    "\u001b[33mWARN: The environment is being initialised with render_mode='rgb_array' that is not in the possible render_modes ([]).\u001b[0m");
            }
        }

        /// <summary>The <c>HumanRendering</c> wrapper for an env without native human rendering (P6), and CarRacing's kwargs error (P8).</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_make.py#L200">test_make_human_rendering</see>.</remarks>
        [TestMethod]
        [OpenBugs("P6", "P8")]
        public void TestMakeHumanRendering_LaterPhases() {
            using IDisposable fixture = RegisterRenderingTestingEnvs();
            using (WarningCatcher caught = Warnings.Catch()) {
                using IEnv env = Make("NoHumanRendering-v0", new { render_mode = "human" });
                Assert.AreEqual("human", env.RenderMode);
                CollectionAssert.Contains(caught.Records.Select(record => record.Message).ToList(),
                    "\u001b[33mWARN: You are trying to use 'human' rendering for an environment that doesn't natively support it. The HumanRendering wrapper is being applied to your environment.\u001b[0m");
            }

            // The extra error is not raised for a TypeError about another keyword.
            StringAssert.Contains(Assert.Throws<TypeError>(() => Make("CarRacing-v3", new { render = "human" })).Message, "got an unexpected keyword argument 'render'");
        }

        /// <summary>The call's kwargs override the spec's, and the made env's spec records the merged kwargs.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_make.py#L258">test_make_kwargs</see>.</remarks>
        [TestMethod]
        public void TestMakeKwargs() {
            Register("test.ArgumentEnv-v0", "Gym.Tests.Gymnasium.Envs.Registration:ArgumentEnv", kwargs: new Info { ["arg1"] = "arg1", ["arg2"] = "arg2" });
            try {
                using IEnv env = Make("test.ArgumentEnv-v0", new { arg2 = "override_arg2", arg3 = "override_arg3" });
                Assert.IsNotNull(env.Spec);
                Assert.AreEqual("test.ArgumentEnv-v0", env.Spec.Id);
                Assert.AreEqual(new Info { ["arg1"] = "arg1", ["arg2"] = "override_arg2", ["arg3"] = "override_arg3" }, env.Spec.Kwargs);

                var unwrapped = (ArgumentEnv)env.Unwrapped;
                Assert.AreEqual("arg1", unwrapped.Arg1);
                Assert.AreEqual("override_arg2", unwrapped.Arg2);
                Assert.AreEqual("override_arg3", unwrapped.Arg3);
            } finally {
                Registry.Remove("test.ArgumentEnv-v0");
            }
        }

        /// <summary>An <c>"Assembly:Id"</c> id loads the assembly, whose module initializer registers the id, then makes it.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_make.py#L279">test_import_module_during_make</see>.</remarks>
        [TestMethod]
        public void TestImportModuleDuringMake() {
            // Test custom environment which is registered at make.
            Assert.IsFalse(Registry.ContainsKey("RegisterDuringMake-v0"));
            using IEnv env = Make("Gym.Tests.Plugin:RegisterDuringMake-v0");
            try {
                Assert.IsTrue(Registry.ContainsKey("RegisterDuringMake-v0"));
                Assert.AreEqual("Gym.Tests.Plugin.RegisterDuringMakeEnv", env.Unwrapped.GetType().FullName);
            } finally {
                Registry.Remove("RegisterDuringMake-v0");
            }
        }

        /// <summary>
        ///     Re-making from specs: an id and its spec agree; a function entry point; and a wrapper that does not record its
        ///     arguments cannot be re-created from a spec, though it can be an entry point's own.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_make.py#L302">test_make_with_env_spec</see>.</remarks>
        [TestMethod]
        public void TestMakeWithEnvSpec() {
            // make
            using (IEnv idEnv = Make("CartPole-v1"))
            using (IEnv specEnv = Make(Spec("CartPole-v1")))
                Assert.AreEqual(idEnv.Spec, specEnv.Spec);

            // make with a callable entry point
            Register("CartPole-v2", (EnvCreator)(_ => new CartPoleEnv()));
            try {
                using (IEnv env3 = Make("CartPole-v2"))
                    Assert.IsInstanceOfType<CartPoleEnv>(env3.Unwrapped);

                // make with a wrapper that does not record its arguments
                using (var env6 = new NoRecordArgsWrapper((Env<NDArray, NDArray>)Make("CartPole-v1"))) {
                    Assert.AreEqual("NoRecordArgsWrapper wrapper does not inherit from `gymnasium.utils.RecordConstructorArgs`, therefore, the wrapper cannot be recreated.",
                        Assert.ThrowsExactly<ValueError>(() => Make(env6.Spec!)).Message);
                }

                // make with such a wrapper in the entry point
                Register("CartPole-v5", (EnvCreator)(_ => new NoRecordArgsWrapper(new CartPoleEnv())), disableEnvChecker: true, orderEnforce: false);
                using (IEnv env7 = Make(Spec("CartPole-v5"))) {
                    Assert.IsInstanceOfType<NoRecordArgsWrapper>(env7);
                    Assert.IsInstanceOfType<CartPoleEnv>(env7.Unwrapped);
                }

                Register("CartPole-v6", (EnvCreator)(_ => new CartPoleEnv()), disableEnvChecker: true, orderEnforce: false,
                    additionalWrappers: new[] { WrapperSpec.For(typeof(NoRecordArgsWrapper), null) });
            } finally {
                Registry.Remove("CartPole-v2");
                Registry.Remove("CartPole-v5");
                Registry.Remove("CartPole-v6");
            }
        }

        /// <summary>
        ///     The parts of re-making from specs that need <c>NormalizeReward</c>, <c>TimeAwareObservation</c> and
        ///     <c>FlattenObservation</c> (P6): a stack of the three is re-made from its spec with an equal spec; a wrapper made by
        ///     the entry point stays the entry point's; a wrapper listed in <c>additional_wrappers</c> is added by <c>make</c>.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_make.py#L302">test_make_with_env_spec</see>.</remarks>
        [TestMethod]
        public void TestMakeWithEnvSpec_ObservationAndRewardWrappers() {
            // make with applied wrappers
            using (IEnv env2 = WrappedCartPole()) {
                using IEnv env2Recreated = Make(env2.Spec!);
                Assert.AreEqual(env2.Spec, env2Recreated.Spec);
            }

            // make with wrapper in env-creator
            Register("CartPole-v3", (EnvCreator)(_ => new NormalizeReward<NDArray, NDArray>(new CartPoleEnv())), disableEnvChecker: true, orderEnforce: false);
            Register("CartPole-v4", (EnvCreator)(_ => new CartPoleEnv()), disableEnvChecker: true, orderEnforce: false,
                additionalWrappers: new[] { WrapperSpec.For(typeof(NormalizeReward<,>), new Info()) });
            try {
                using (IEnv env4 = Make(Spec("CartPole-v3"))) {
                    Assert.IsInstanceOfType<NormalizeReward<NDArray, NDArray>>(env4);
                    Assert.IsInstanceOfType<CartPoleEnv>(((IWrapper)env4).Env);
                }

                using (IEnv env5 = Make(Spec("CartPole-v4"))) {
                    Assert.IsInstanceOfType<NormalizeReward<NDArray, NDArray>>(env5);
                    Assert.IsInstanceOfType<CartPoleEnv>(((IWrapper)env5).Env);
                }
            } finally {
                Registry.Remove("CartPole-v3");
                Registry.Remove("CartPole-v4");
            }
        }

        /// <summary>Each level of a wrapper stack is re-made from its own spec with an equal spec.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_make.py#L388">test_make_with_env_spec_levels</see>.</remarks>
        [TestMethod]
        public void TestMakeWithEnvSpecLevels() {
            using IEnv stack = WrappedCartPole();
            for (IEnv env = stack; !ReferenceEquals(env, env.Unwrapped); env = ((IWrapper)env).Env) {
                using IEnv recreatedEnv = Make(env.Spec!);
                Assert.AreEqual(env.Spec, recreatedEnv.Spec, $"Re-making {env}");
            }
        }

        /// <summary>
        ///     An entry point that wraps the env itself (<c>FlattenObservation</c>), wrapped further by the caller: the stack
        ///     re-made from its spec resets and steps exactly as the original.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_make.py#L404">test_wrapped_env_entry_point</see>.</remarks>
        [TestMethod]
        public void TestWrappedEnvEntryPoint() {
            Register("TestingEnv-v0", (EnvCreator)(_ => new FlattenObservation<NDArray, NDArray>((Env<NDArray, NDArray>)Make("CartPole-v1", new { render_mode = "rgb_array" }))));
            try {
                var made = (Env<NDArray, NDArray>)Make("TestingEnv-v0");
                using IEnv env = new NormalizeReward<object?, NDArray>(new TimeAwareObservation<NDArray, NDArray>(made), gamma: 0.8);
                using IEnv recreatedEnv = Make(env.Spec!);

                (object? obs, Info info) = env.Reset(seed: 42);
                (object? recreatedObs, Info recreatedInfo) = recreatedEnv.Reset(seed: 42);
                Assert.IsTrue(EnvChecker.DataEquivalence(obs, recreatedObs));
                Assert.IsTrue(EnvChecker.DataEquivalence(info, recreatedInfo));

                object action = env.ActionSpace.Sample()!;
                (obs, double reward, bool terminated, bool truncated, info) = env.Step(action);
                (recreatedObs, double recreatedReward, bool recreatedTerminated, bool recreatedTruncated, recreatedInfo) = recreatedEnv.Step(action);
                Assert.IsTrue(EnvChecker.DataEquivalence(obs, recreatedObs));
                Assert.AreEqual(reward, recreatedReward);
                Assert.AreEqual(terminated, recreatedTerminated);
                Assert.AreEqual(truncated, recreatedTruncated);
                Assert.IsTrue(EnvChecker.DataEquivalence(info, recreatedInfo));
            } finally {
                Registry.Remove("TestingEnv-v0");
            }
        }

        /// <summary>
        ///     <c>NormalizeReward(TimeAwareObservation(FlattenObservation(make("CartPole-v1", render_mode="rgb_array"))),
        ///     gamma=0.8)</c>, the stack Gymnasium's make-with-spec tests re-make.
        /// </summary>
        /// <returns>The stack; the caller owns it.</returns>
        private static IEnv WrappedCartPole() {
            var made = (Env<NDArray, NDArray>)Make("CartPole-v1", new { render_mode = "rgb_array" });
            return new NormalizeReward<object?, NDArray>(new TimeAwareObservation<NDArray, NDArray>(new FlattenObservation<NDArray, NDArray>(made)), gamma: 0.8);
        }

        /// <summary>An unknown name is reported with Gymnasium's message.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_make.py#L440">test_make_errors</see>.</remarks>
        [TestMethod]
        public void TestMakeErrors() {
            StringAssert.Contains(Assert.ThrowsExactly<NameNotFound>(() => Make("NonExistenceEnv-v0")).Message, "Environment `NonExistenceEnv` doesn't exist.");
        }

        /// <summary><c>make</c> from an id or a spec, as the Python tests pass either to the same call.</summary>
        /// <param name="makeId">An id or an <see cref="EnvSpec"/>.</param>
        /// <param name="maxEpisodeSteps">The call's step limit.</param>
        /// <returns>The env; the caller owns it.</returns>
        private static IEnv MakeFrom(object makeId, long? maxEpisodeSteps = null) =>
            makeId is EnvSpec spec ? Make(spec, maxEpisodeSteps: maxEpisodeSteps) : Make((string)makeId, maxEpisodeSteps: maxEpisodeSteps);

        /// <summary>
        ///     Gymnasium's <c>register_parameter_envs</c> fixture: an env without a step limit and one without order enforcing;
        ///     disposing the result unregisters them.
        /// </summary>
        /// <returns>The teardown.</returns>
        private static IDisposable RegisterParameterEnvs() {
            Register("NoMaxEpisodeStepsEnv-v0", (EnvCreator)(_ => new GenericTestEnv()), maxEpisodeSteps: null);
            Register("OrderlessEnv-v0", (EnvCreator)(_ => new GenericTestEnv()), orderEnforce: false);
            return new Unregister("NoMaxEpisodeStepsEnv-v0", "OrderlessEnv-v0");
        }

        /// <summary>Gymnasium's <c>register_rendering_testing_envs</c> fixture: the rendering test envs by entry-point string.</summary>
        /// <returns>The teardown.</returns>
        private static IDisposable RegisterRenderingTestingEnvs() {
            Register("NoHumanRendering-v0", "Gym.Tests.Gymnasium.Envs.Registration:NoHuman");
            Register("NoHumanRenderingOldAPI-v0", "Gym.Tests.Gymnasium.Envs.Registration:NoHumanOldApi");
            Register("NoHumanRenderingNoRGB-v0", "Gym.Tests.Gymnasium.Envs.Registration:NoHumanNoRgb");
            Register("NoRenderModesMetadata-v0", "Gym.Tests.Gymnasium.Envs.Registration:NoRenderModesMetadata");
            return new Unregister("NoHumanRendering-v0", "NoHumanRenderingOldAPI-v0", "NoHumanRenderingNoRGB-v0", "NoRenderModesMetadata-v0");
        }

        /// <summary>A wrapper that does not record its constructor arguments (Gymnasium's <c>NoRecordArgsWrapper</c>), so no spec can re-create it.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/test_make.py#L294">tests/envs/registration/test_make.py (NoRecordArgsWrapper)</see>.</remarks>
        public sealed class NoRecordArgsWrapper : ObservationWrapper<NDArray, NDArray, NDArray> {
            /// <summary>Wraps <paramref name="env"/>.</summary>
            /// <param name="env">The env.</param>
            public NoRecordArgsWrapper(Env<NDArray, NDArray> env) : base(env) { }

            /// <summary>Replaces the observation with a sample of the observation space.</summary>
            /// <param name="observation">The env's observation, released here.</param>
            /// <returns>The sample.</returns>
            public override NDArray Observation(NDArray observation) {
                observation.Dispose();
                return ObservationSpace.Sample();
            }
        }

        /// <summary>A fixture teardown: unregisters ids when disposed.</summary>
        private sealed class Unregister : IDisposable {
            /// <summary>The ids to unregister.</summary>
            private readonly string[] _ids;

            /// <summary>Holds the ids.</summary>
            /// <param name="ids">The ids.</param>
            public Unregister(params string[] ids) => _ids = ids;

            /// <summary>Unregisters the ids that are still registered.</summary>
            public void Dispose() {
                foreach (string id in _ids)
                    Registry.Remove(id);
            }
        }
    }
}
