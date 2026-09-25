#nullable enable
using System;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Ported.Wrappers {
    /// <summary>Port of <c>tests/wrappers/test_order_enforcing.py</c> @ 9e04324.</summary>
    [TestClass]
    public class TestOrderEnforcing {
        /// <summary>
        ///     Port of <c>test_order_enforcing</c>: the order enforcing works as expected, raising an error before reset is
        ///     called and not after. <c>gym.make</c> is not used because every env it makes is already order-enforced.
        /// </summary>
        [TestMethod]
        public void TestOrderEnforcingWrapper() {
            var env = new CartPoleEnv(renderMode: "rgb_array_list");
            Assert.IsFalse(WrapperTestUtils.HasWrapper(env, typeof(OrderEnforcing<,>)));

            // Assert that the order enforcing works for step and render before reset.
            var orderEnforcedEnv = new OrderEnforcing<NDArray, long>(env);
            Assert.IsFalse(orderEnforcedEnv.HasReset);
            Assert.ThrowsExactly<ResetNeeded>(() => orderEnforcedEnv.Step(0));
            Assert.ThrowsExactly<ResetNeeded>(() => orderEnforcedEnv.Render());
            Assert.IsFalse(orderEnforcedEnv.HasReset);

            // Assert that the errors are not raised after reset.
            orderEnforcedEnv.Reset();
            Assert.IsTrue(orderEnforcedEnv.HasReset);
            orderEnforcedEnv.Step(0);
            orderEnforcedEnv.Render();

            // Assert that disable_render_order_enforcing works; the environment has not been reset.
            env = new CartPoleEnv(renderMode: "rgb_array_list");
            var renderEnforcingDisabled = new OrderEnforcing<NDArray, long>(env, disableRenderOrderEnforcing: true);
            renderEnforcingDisabled.Render(); // no error
        }

        /// <summary>The messages of both errors are Gymnasium's, verbatim (the source's docstring shows them).</summary>
        [TestMethod]
        public void TestOrderEnforcing_Messages() {
            var env = new OrderEnforcing<NDArray, long>(new CartPoleEnv(renderMode: "rgb_array"));

            Assert.AreEqual("Cannot call env.step() before calling env.reset()", Assert.ThrowsExactly<ResetNeeded>(() => env.Step(0)).Message);
            Assert.AreEqual("Cannot call `env.render()` before calling `env.reset()`, if this is an intended action, set `disable_render_order_enforcing=True` on the OrderEnforcer wrapper.",
                Assert.ThrowsExactly<ResetNeeded>(() => env.Render()).Message);
            Assert.IsInstanceOfType<Error>(new ResetNeeded(), "ResetNeeded is a gymnasium.error.Error.");
        }
    }

    /// <summary>Port of the helpers of <c>tests/wrappers/utils.py</c> @ 9e04324 that the ported wrapper tests use.</summary>
    internal static class WrapperTestUtils {
        /// <summary>
        ///     Port of <c>has_wrapper</c>: whether a wrapper of <paramref name="wrapperType"/> is anywhere in the stack above
        ///     the unwrapped env.
        /// </summary>
        /// <param name="wrappedEnv">The outermost env.</param>
        /// <param name="wrapperType">The wrapper type; an open generic (<c>typeof(OrderEnforcing&lt;,&gt;)</c>) matches any of its closed types.</param>
        /// <returns><see langword="true"/> when some layer is such a wrapper.</returns>
        internal static bool HasWrapper(IEnv wrappedEnv, Type wrapperType) {
            while (wrappedEnv is IWrapper wrapper) {
                Type type = wrapper.GetType();
                if (wrapperType.IsAssignableFrom(type) || (wrapperType.IsGenericTypeDefinition && type.IsGenericType && type.GetGenericTypeDefinition() == wrapperType))
                    return true;
                wrappedEnv = wrapper.Env;
            }

            return false;
        }
    }
}
