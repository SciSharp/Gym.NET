using System;
using System.Collections.Generic;
using System.Numerics;
using Gym.Environments;
using Gym.Environments.Envs.Classic;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Envs.Classic {
    /// <summary>
    ///     <see cref="ClassicControlUtils"/> and CartPole's <c>Reset(seed, options)</c> against Gymnasium's answers (recorded
    ///     from <c>refs/Gymnasium</c> @ 9e04324): which bounds are accepted, the messages of the rejected ones, and the order
    ///     of seeding and parsing.
    /// </summary>
    [TestClass]
    public class ClassicControlUtilsTests {
        /// <summary>Without options, or without the keys, the defaults are used; present keys win, numeric strings included.</summary>
        [TestMethod]
        public void MaybeParseResetBounds_UsesOptionsOverDefaults() {
            Assert.AreEqual((-0.05, 0.05), ClassicControlUtils.MaybeParseResetBounds(null, -0.05, 0.05));
            Assert.AreEqual((-0.05, 0.05), ClassicControlUtils.MaybeParseResetBounds(new Dictionary<string, object> { ["other"] = 1 }, -0.05, 0.05));
            Assert.AreEqual((-0.1, 0.05), ClassicControlUtils.MaybeParseResetBounds(new Dictionary<string, object> { ["low"] = -0.1 }, -0.05, 0.05));
            Assert.AreEqual((-0.5, 10.0), ClassicControlUtils.MaybeParseResetBounds(new Dictionary<string, object> { ["low"] = "-0.5", ["high"] = "1_0" }, -0.05, 0.05));
            Assert.AreEqual((1.0, 2.0), ClassicControlUtils.MaybeParseResetBounds(new Dictionary<string, object> { ["low"] = true, ["high"] = 2L }, -0.05, 0.05));
            Assert.AreEqual((0.05, 0.05), ClassicControlUtils.MaybeParseResetBounds(new Dictionary<string, object> { ["low"] = 0.05, ["high"] = 0.05 }, -0.05, 0.05), "Equal bounds are allowed.");
        }

        /// <summary>Rejected bounds raise <see cref="ValueError"/> with Gymnasium's messages, <c>{x}</c> printed as Python's <c>str</c>.</summary>
        [TestMethod]
        public void MaybeParseResetBounds_RejectsLikeGymnasium() {
            Assert.AreEqual("Lower bound (0.2) must be lower than higher bound (0.05).", Reject(new Dictionary<string, object> { ["low"] = 0.2 }));
            Assert.AreEqual("Lower bound (0.1) must be lower than higher bound (-0.1).", Reject(new Dictionary<string, object> { ["low"] = 0.1, ["high"] = -0.1 }));
            Assert.AreEqual("Lower bound (-0.05) must be lower than higher bound (-1.0).", Reject(new Dictionary<string, object> { ["high"] = -1.0 }));
            Assert.AreEqual("An option (abc) could not be converted to a float.", Reject(new Dictionary<string, object> { ["low"] = "abc" }));
            Assert.AreEqual("An option (None) could not be converted to a float.", Reject(new Dictionary<string, object> { ["low"] = null }));
            using NDArray vector = np.array(new[] { 0.1 });
            Assert.AreEqual("An option ([0.1]) could not be converted to a float.", Reject(new Dictionary<string, object> { ["low"] = vector }));
        }

        /// <summary>An integer beyond the float range raises Python's <c>OverflowError</c> unchanged: the source catches only ValueError and TypeError.</summary>
        [TestMethod]
        public void MaybeParseResetBounds_LetsOverflowThrough() {
            var options = new Dictionary<string, object> { ["low"] = BigInteger.Pow(10, 400) };
            Assert.AreEqual("int too large to convert to float", Assert.ThrowsExactly<OverflowException>(() => ClassicControlUtils.MaybeParseResetBounds(options, -0.05, 0.05)).Message);
        }

        /// <summary>
        ///     CartPole's seeded reset with options draws the state Gymnasium draws: the docstring's own example,
        ///     <c>reset(seed=123, options={"low": -0.1, "high": 0.1})</c>, printed there as
        ///     <c>[0.03647037, -0.0892358, -0.05592803, -0.06312564]</c>; compared here by the float32 bits Gymnasium returns.
        /// </summary>
        [TestMethod]
        public void CartPoleReset_WithOptions_DrawsGymnasiumsState() {
            var env = new CartPoleEnv(NullEnvViewer.Factory);
            (NDArray observation, _) = env.Reset(123, new Dictionary<string, object> { ["low"] = -0.1, ["high"] = 0.1 });
            using (observation) {
                int[] bits = Array.ConvertAll(observation.ToArray<float>(), BitConverter.SingleToInt32Bits);
                CollectionAssert.AreEqual(new[] { 0x3d1561f5, unchecked((int)0xbdb6c142), unchecked((int)0xbd6514c9), unchecked((int)0xbd814804) }, bits);
            }
        }

        /// <summary>
        ///     A reset that fails on its options has already reseeded: the generator is the seed's, undrawn, and the state is
        ///     the previous one, as <c>super().reset(seed=seed)</c> runs before the options are parsed.
        /// </summary>
        [TestMethod]
        public void CartPoleReset_ThatFails_HasAlreadyReseeded() {
            var env = new CartPoleEnv(NullEnvViewer.Factory);
            env.Reset(1).Observation.Dispose();
            double[] before = env.State;

            Assert.ThrowsExactly<ValueError>(() => env.Reset(7, new Dictionary<string, object> { ["low"] = "abc" }));

            Assert.AreSame(before, env.State, "The state is untouched.");
            using NDArray fromFresh = Seeding.NpRandom(7).Rng.uniform(-0.05, 0.05, new Shape(4));
            using NDArray fromEnv = env.NpRandom.uniform(-0.05, 0.05, new Shape(4));
            Assert.IsTrue(np.array_equal(fromFresh, fromEnv), "The env's generator is freshly seeded with 7.");
        }

        /// <summary>The message of the <see cref="ValueError"/> a set of options raises.</summary>
        /// <param name="options">The options.</param>
        /// <returns>The message.</returns>
        private static string Reject(Dictionary<string, object> options) {
            return Assert.ThrowsExactly<ValueError>(() => ClassicControlUtils.MaybeParseResetBounds(options, -0.05, 0.05)).Message;
        }
    }
}
