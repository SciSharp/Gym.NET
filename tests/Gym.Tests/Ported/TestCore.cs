#nullable enable
using System;
using System.Collections.Generic;
using System.Numerics;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Observations;
using Gym.Spaces;
using Gym.Utils;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Ported {
    /// <summary>
    ///     Gymnasium's core tests, run against Gym.NET: the exit criterion of the core-API phase (P1), covering the
    ///     <c>Env</c> and <c>Wrapper</c> contract, the three single-hook wrappers, Python-name attribute access through a
    ///     wrapper stack, and <c>np_random</c>/<c>np_random_seed</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Python attribute syntax maps to <see cref="PyAttributes"/>: <c>env.gravity</c> is
    ///     <c>PyAttributes.GetAttr(env, "gravity")</c> and <c>env.gravity = 5.0</c> is <c>PyAttributes.SetAttr(env,
    ///     "gravity", 5.0)</c>, so the test keeps Python's semantics (a wrapper does not forward attributes, and assigning an
    ///     unknown one creates it on that layer).
    ///     </para>
    ///     <para>
    ///     <c>test_get_set_wrapper_attr</c> builds <c>gym.make("CartPole-v1")</c>'s stack by hand, <c>TimeLimit(OrderEnforcing(
    ///     CartPoleEnv))</c> with the spec <c>make</c> attaches: <c>make</c> and the <c>PassiveEnvChecker</c> it also adds
    ///     arrive with the registry (P3), and the checker takes no part in what the test asserts.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/tests/test_core.py">tests/test_core.py</see>.</para>
    /// </remarks>
    [TestClass]
    public class TestCore {
        /// <summary>The <c>example_env</c> fixture: a fresh <see cref="ExampleEnv"/> per test, so no test sees another's state.</summary>
        /// <returns>A new <see cref="ExampleEnv"/>.</returns>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/tests/test_core.py#L46">tests/test_core.py (example_env)</see>.</remarks>
        private static ExampleEnv CreateExampleEnv() => new();

        /// <summary>An env's defaults are Gymnasium's.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/tests/test_core.py#L51">tests/test_core.py (test_example_env)</see>.</remarks>
        [TestMethod]
        public void TestExampleEnv() {
            var exampleEnv = CreateExampleEnv();

            Assert.IsTrue(exampleEnv.Metadata.Equals(new Info { ["render_modes"] = new List<object>() }));
            Assert.IsNull(exampleEnv.RenderMode);
            Assert.IsNull(exampleEnv.Spec);
            Assert.IsNull(PyAttributes.GetAttr(exampleEnv, "_np_random"));
        }

        /// <summary>What a wrapper forwards, what it can override, and its <c>_np_random</c> error.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/tests/test_core.py#L87">tests/test_core.py (test_example_wrapper)</see>.</remarks>
        [TestMethod]
        public void TestExampleWrapper() {
            var env = CreateExampleEnv();
            var wrapperEnv = new ExampleWrapper(env);

            Assert.AreEqual(env.Metadata, wrapperEnv.Metadata);
            wrapperEnv.Metadata = new Metadata { ["render_modes"] = new List<string> { "rgb_array" } };
            Assert.AreNotEqual(env.Metadata, wrapperEnv.Metadata);

            Assert.AreEqual(env.RenderMode, wrapperEnv.RenderMode);

            Assert.AreEqual(env.Spec, wrapperEnv.Spec);

            env.ObservationSpace = new Box(0f, 1f, Shape.Scalar);
            env.ActionSpace = new Box(0f, 1f, Shape.Scalar);
            Assert.AreEqual(env.ObservationSpace, wrapperEnv.ObservationSpace);
            Assert.AreEqual(env.ActionSpace, wrapperEnv.ActionSpace);
            wrapperEnv.ObservationSpace = new Box(1f, 2f, Shape.Scalar);
            wrapperEnv.ActionSpace = new Box(1f, 2f, Shape.Scalar);
            Assert.AreNotEqual(env.ObservationSpace, wrapperEnv.ObservationSpace);
            Assert.AreNotEqual(env.ActionSpace, wrapperEnv.ActionSpace);

            (wrapperEnv.NpRandom, _) = Seeding.NpRandom();
            Assert.AreSame(PyAttributes.GetAttr(env, "_np_random"), env.NpRandom);
            Assert.AreSame(env.NpRandom, wrapperEnv.NpRandom);
            using (NDArray uniform = wrapperEnv.NpRandom.uniform()) {
                double value = (double)uniform;
                Assert.IsTrue(0 <= value && value <= 1);
            }

            var error = Assert.ThrowsExactly<AttributeError>(() => wrapperEnv.AccessHiddenNpRandom());
            Assert.AreEqual("Can't access `_np_random` of a wrapper, use `.unwrapped._np_random` or `.np_random`.", error.Message);
        }

        /// <summary>Each hook wrapper transforms what it is named after.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/tests/test_core.py#L149">tests/test_core.py (test_reward_observation_action_wrapper)</see>.</remarks>
        [TestMethod]
        public void TestRewardObservationActionWrapper() {
            var env = new GenericTestEnv();

            var rewardEnv = new ExampleRewardWrapper(env);
            rewardEnv.Reset();
            (_, double reward, _, _, _) = rewardEnv.Step(0);
            Assert.AreEqual(1.0, reward);

            var observationEnv = new ExampleObservationWrapper(env);
            (object? obs, _) = observationEnv.Reset();
            Assert.IsTrue(PyEquality.AreEqual(obs, np.array(new long[] { 1 })));
            (obs, _, _, _, _) = observationEnv.Step(0);
            Assert.IsTrue(PyEquality.AreEqual(obs, np.array(new long[] { 1 })));

            env = new GenericTestEnv(stepFunc: (self, action) => new StepResult<object?>(action, 0, false, false, new Info()));
            var actionEnv = new ExampleActionWrapper(env);
            (obs, _, _, _, _) = actionEnv.Step(0);
            Assert.IsTrue(PyEquality.AreEqual(obs, np.array(new long[] { 1 })));
        }

        /// <summary>
        ///     Attributes are reached through the stack only by <c>get_wrapper_attr</c>/<c>set_wrapper_attr</c>/
        ///     <c>has_wrapper_attr</c>, never forwarded by plain access.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/tests/test_core.py#L170">tests/test_core.py (test_get_set_wrapper_attr)</see>.</remarks>
        [TestMethod]
        public void TestGetSetWrapperAttr() {
            IEnv env = MakeCartPole();
            Assert.AreNotSame(env, env.Unwrapped);

            // Test get_wrapper_attr
            Assert.ThrowsExactly<AttributeError>(() => PyAttributes.GetAttr(env, "gravity"));
            Assert.IsNotNull(PyAttributes.GetAttr(env.Unwrapped, "gravity"));
            Assert.IsTrue(env.HasWrapperAttr("gravity"));
            Assert.IsNotNull(env.GetWrapperAttr("gravity"));

            Assert.ThrowsExactly<AttributeError>(() => PyAttributes.GetAttr(env, "unknown_attr"));
            Assert.IsFalse(env.HasWrapperAttr("unknown_attr"));
            Assert.ThrowsExactly<AttributeError>(() => env.GetWrapperAttr("unknown_attr"));

            // Test set_wrapper_attr
            env.SetWrapperAttr("gravity", 10.0);
            // Checks the top level wrapper hasn't been assigned.
            Assert.ThrowsExactly<AttributeError>(() => PyAttributes.GetAttr(env, "gravity"));
            Assert.AreEqual(10.0, PyAttributes.GetAttr(env.Unwrapped, "gravity"));
            Assert.AreEqual(10.0, env.GetWrapperAttr("gravity"));

            PyAttributes.SetAttr(env, "gravity", 5.0);
            Assert.AreEqual(5.0, PyAttributes.GetAttr(env, "gravity"));
            Assert.AreEqual(5.0, env.GetWrapperAttr("gravity"));
            Assert.AreEqual(10.0, ((IWrapper)env).Env.GetWrapperAttr("gravity"));

            // Test with OrderEnforcing (intermediate wrapper)
            Assert.IsFalse(env is OrderEnforcing<NDArray, long>);

            // Show that the base and top level objects don't contain the attribute.
            Assert.ThrowsExactly<AttributeError>(() => PyAttributes.GetAttr(env, "_disable_render_order_enforcing"));
            Assert.ThrowsExactly<AttributeError>(() => PyAttributes.GetAttr(env.Unwrapped, "_disable_render_order_enforcing"));
            Assert.IsTrue(env.HasWrapperAttr("_disable_render_order_enforcing"));
            Assert.AreEqual(false, env.GetWrapperAttr("_disable_render_order_enforcing"));

            env.SetWrapperAttr("_disable_render_order_enforcing", true);

            Assert.ThrowsExactly<AttributeError>(() => PyAttributes.GetAttr(env, "_disable_render_order_enforcing"));
            Assert.ThrowsExactly<AttributeError>(() => PyAttributes.GetAttr(env.Unwrapped, "_disable_render_order_enforcing"));
            Assert.AreEqual(true, env.GetWrapperAttr("_disable_render_order_enforcing"));

            // Test with top-most wrapper
            PyAttributes.SetAttr(env, "MY_ATTRIBUTE_1", true);
            Assert.AreEqual(true, env.GetWrapperAttr("MY_ATTRIBUTE_1"));
            env.SetWrapperAttr("MY_ATTRIBUTE_1", false);
            Assert.AreEqual(false, env.GetWrapperAttr("MY_ATTRIBUTE_1"));

            // Test with non-existing attribute
            env.SetWrapperAttr("MY_ATTRIBUTE_2", true);
            Assert.AreEqual(true, PyAttributes.GetAttr(env, "MY_ATTRIBUTE_2"));
        }

        /// <summary>Reading the seed creates the generator.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/tests/test_core.py#L232">tests/test_core.py (TestRandomSeeding.test_nonempty_seed_retrieved_when_not_set)</see>.</remarks>
        [TestMethod]
        public void TestRandomSeeding_NonemptySeedRetrievedWhenNotSet() {
            var exampleEnv = CreateExampleEnv();

            BigInteger seed = exampleEnv.NpRandomSeed;
            Assert.IsTrue(seed.Sign >= 0);
            Assert.IsNotNull(PyAttributes.GetAttr(exampleEnv, "_np_random"));
        }

        /// <summary>A seeded reset records its seed; an unseeded one keeps it.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/tests/test_core.py#L237">tests/test_core.py (TestRandomSeeding.test_seed_set_at_reset_and_retrieved)</see>.</remarks>
        [TestMethod]
        public void TestRandomSeeding_SeedSetAtResetAndRetrieved() {
            var exampleEnv = CreateExampleEnv();
            const int seed = 42;

            exampleEnv.Reset(seed: seed);
            Assert.AreEqual(new BigInteger(seed), exampleEnv.NpRandomSeed);
            // Resetting with seed=None means seed remains the same.
            exampleEnv.Reset(seed: null);
            Assert.AreEqual(new BigInteger(seed), exampleEnv.NpRandomSeed);
        }

        /// <summary><c>np_random_seed</c> is read-only.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/tests/test_core.py#L246">tests/test_core.py (TestRandomSeeding.test_seed_cannot_be_set_directly)</see>.</remarks>
        [TestMethod]
        public void TestRandomSeeding_SeedCannotBeSetDirectly() {
            var exampleEnv = CreateExampleEnv();

            var error = Assert.ThrowsExactly<AttributeError>(() => PyAttributes.SetAttr(exampleEnv, "np_random_seed", 42));
            Assert.AreEqual("property 'np_random_seed' of 'ExampleEnv' object has no setter", error.Message);
        }

        /// <summary>An assigned generator's seed is -1.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/tests/test_core.py#L251">tests/test_core.py (TestRandomSeeding.test_negative_seed_retrieved_when_seed_unknown)</see>.</remarks>
        [TestMethod]
        public void TestRandomSeeding_NegativeSeedRetrievedWhenSeedUnknown() {
            var exampleEnv = CreateExampleEnv();
            (Generator rng, _) = Seeding.NpRandom();

            exampleEnv.NpRandom = rng;
            // The seed is unknown.
            Assert.AreEqual(new BigInteger(-1), exampleEnv.NpRandomSeed);
        }

        /// <summary>Seeding through a wrapper seeds the env beneath.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/tests/test_core.py#L258">tests/test_core.py (TestRandomSeeding.test_seeding_works_in_wrapped_envs)</see>.</remarks>
        [TestMethod]
        public void TestRandomSeeding_SeedingWorksInWrappedEnvs() {
            const int seed = 42;
            var wrapperEnv = new ExampleWrapper(CreateExampleEnv());

            wrapperEnv.Reset(seed: seed);
            Assert.AreEqual(new BigInteger(seed), wrapperEnv.NpRandomSeed);
            // Resetting with seed=None means seed remains the same.
            wrapperEnv.Reset(seed: null);
            Assert.AreEqual(new BigInteger(seed), wrapperEnv.NpRandomSeed);
            // Setting np_random directly makes seed unknown.
            (Generator rng, _) = Seeding.NpRandom();
            wrapperEnv.NpRandom = rng;
            Assert.AreEqual(new BigInteger(-1), wrapperEnv.NpRandomSeed);
        }

        /// <summary>
        ///     <c>gym.make("CartPole-v1")</c>'s stack without the passive env checker (P3):
        ///     <c>TimeLimit(OrderEnforcing(CartPoleEnv), 500)</c>, the unwrapped env carrying the spec <c>make</c> gives it
        ///     (the wrappers' settings cleared, so that each wrapper puts its own back).
        /// </summary>
        /// <returns>The outermost wrapper.</returns>
        private static TimeLimit<NDArray, long> MakeCartPole() {
            var cartPole = new CartPoleEnv {
                Spec = new EnvSpec("CartPole-v1", "Gym.Environments.Envs.Classic:CartPoleEnv", rewardThreshold: 475.0, maxEpisodeSteps: null, orderEnforce: false,
                    disableEnvChecker: true)
            };
            return new TimeLimit<NDArray, long>(new OrderEnforcing<NDArray, long>(cartPole), 500);
        }

        /// <summary>The minimal env of the core tests: <c>Box(0, 1)</c> spaces, every step returns <c>(0, 0, False, False, {})</c>, reset seeds and returns <c>(0, {})</c>.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/tests/test_core.py#L21">tests/test_core.py (ExampleEnv)</see>.</remarks>
        private sealed class ExampleEnv : Env<object?, object?> {
            /// <summary>Constructor for example environment.</summary>
            public ExampleEnv() {
                ObservationSpace = new Box(0f, 1f, Shape.Scalar);
                ActionSpace = new Box(0f, 1f, Shape.Scalar);
            }

            /// <summary>Steps through the environment.</summary>
            /// <param name="action">Ignored.</param>
            /// <returns><c>(0, 0, False, False, {})</c>.</returns>
            public override StepResult<object?> Step(object? action) => new(0L, 0, false, false, new Info());

            /// <summary>Resets the environment.</summary>
            /// <param name="seed">Seeds the env's generator when given.</param>
            /// <param name="options">Ignored.</param>
            /// <returns><c>(0, {})</c>.</returns>
            public override ResetResult<object?> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
                SeedNpRandom(seed);
                return new ResetResult<object?>(0L, new Info());
            }
        }

        /// <summary>A wrapper that rewards every step with <c>new_reward</c> (3).</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/tests/test_core.py#L60">tests/test_core.py (ExampleWrapper)</see>.</remarks>
        private sealed class ExampleWrapper : Wrapper<object?, object?> {
            /// <summary>Constructor that sets the reward.</summary>
            /// <param name="env">The env to wrap.</param>
            public ExampleWrapper(Env<object?, object?> env) : base(env) {
                NewReward = 3;
            }

            /// <summary>Python's <c>new_reward</c>.</summary>
            public double NewReward { get; set; }

            /// <summary>Resets the environment.</summary>
            /// <param name="seed">The seed, passed on.</param>
            /// <param name="options">The options, passed on.</param>
            /// <returns>The env's reset.</returns>
            public override ResetResult<object?> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) => base.Reset(seed, options);

            /// <summary>Steps through the environment.</summary>
            /// <param name="action">The action, passed on.</param>
            /// <returns>The env's transition with <see cref="NewReward"/>.</returns>
            public override StepResult<object?> Step(object? action) {
                (object? obs, _, bool termination, bool truncation, Info info) = Env.Step(action);
                return new StepResult<object?>(obs, NewReward, termination, truncation, info);
            }

            /// <summary>Reads the wrapper's own <c>_np_random</c>, which raises: wrappers use the unwrapped env's.</summary>
            /// <returns>Nothing; always throws.</returns>
            /// <exception cref="AttributeError">Always, with Gymnasium's message.</exception>
            public Generator? AccessHiddenNpRandom() => RawNpRandom;
        }

        /// <summary>A reward wrapper under which every reward becomes 1.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/tests/test_core.py#L125">tests/test_core.py (ExampleRewardWrapper)</see>.</remarks>
        private sealed class ExampleRewardWrapper : RewardWrapper<object?, object?> {
            /// <summary>Wraps <paramref name="env"/>.</summary>
            /// <param name="env">The env.</param>
            public ExampleRewardWrapper(Env<object?, object?> env) : base(env) { }

            /// <summary>Reward function.</summary>
            /// <param name="reward">Ignored.</param>
            /// <returns>1.</returns>
            public override double Reward(double reward) => 1;
        }

        /// <summary>An observation wrapper under which every observation becomes <c>np.array([1])</c>.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/tests/test_core.py#L133">tests/test_core.py (ExampleObservationWrapper)</see>.</remarks>
        private sealed class ExampleObservationWrapper : ObservationWrapper<object?, object?, object?> {
            /// <summary>Wraps <paramref name="env"/>.</summary>
            /// <param name="env">The env.</param>
            public ExampleObservationWrapper(Env<object?, object?> env) : base(env) { }

            /// <summary>Observation function.</summary>
            /// <param name="observation">Ignored.</param>
            /// <returns><c>np.array([1])</c>.</returns>
            public override object? Observation(object? observation) => np.array(new long[] { 1 });
        }

        /// <summary>An action wrapper under which every action becomes <c>np.array([1])</c>.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/tests/test_core.py#L141">tests/test_core.py (ExampleActionWrapper)</see>.</remarks>
        private sealed class ExampleActionWrapper : ActionWrapper<object?, object?, object?> {
            /// <summary>Wraps <paramref name="env"/>.</summary>
            /// <param name="env">The env.</param>
            public ExampleActionWrapper(Env<object?, object?> env) : base(env) { }

            /// <summary>Action function.</summary>
            /// <param name="action">Ignored.</param>
            /// <returns><c>np.array([1])</c>.</returns>
            public override object? Action(object? action) => np.array(new long[] { 1 });
        }
    }
}
