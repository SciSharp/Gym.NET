#nullable enable
using System;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Spaces;
using Gym.Tests.Gymnasium;
using Gym.Vector;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Spaces {
    /// <summary>
    ///     DR-40: one owner per space, enforced. An env owns and disposes the spaces it is given (D19), so handing one space
    ///     to a second env that is not yet disposed fails at once, where it used to fail late (the first env disposing the
    ///     space under the second, the corrupted boxes P6.8 met).
    /// </summary>
    [TestClass]
    public class SpaceOwnershipTests {
        /// <summary>A typed env's space handed to a second typed env is refused, with a message naming both envs and the fix.</summary>
        /// <exception cref="AssertFailedException">Thrown when the second env takes the space, or the message lacks the fix.</exception>
        [TestMethod]
        public void SecondEnv_TakingAnOwnedSpace_IsRefused() {
            using var first = new CartPoleEnv();
            using var second = new CartPoleEnv();
            var error = Assert.ThrowsExactly<InvalidOperationException>(() => second.ActionSpace = first.ActionSpace);
            StringAssert.Contains(error.Message, "cannot become the action_space of CartPoleEnv: it already belongs to CartPoleEnv");
            StringAssert.Contains(error.Message, "space.DeepCopy()");
            Assert.AreNotSame(first.ActionSpace, second.ActionSpace, "the refused space did not replace the second env's own");
        }

        /// <summary>
        ///     Two untyped envs sharing one space through the views they wrap it in (Gymnasium's generic test env taking the
        ///     same box twice, P6.8's case): the view takes the space it wraps, so the second env is refused too.
        /// </summary>
        /// <exception cref="AssertFailedException">Thrown when the second env takes the shared box.</exception>
        [TestMethod]
        public void UntypedEnvs_SharingOneSpaceThroughViews_AreRefused() {
            var box = new Box(-1, 1, new Shape(2));
            using var first = new GenericTestEnv(actionSpace: box);
            Assert.ThrowsExactly<InvalidOperationException>(() => new GenericTestEnv(actionSpace: box));
        }

        /// <summary>One env may hold one space as both its action and its observation space: the same owner takes it twice.</summary>
        [TestMethod]
        public void OneEnv_MayHoldOneSpaceTwice() {
            var box = new Box(0, 1, new Shape(1));
            using var env = new GenericTestEnv(actionSpace: box, observationSpace: box);
            Assert.IsTrue(env.ActionSpace.Contains(env.ObservationSpace.Sample()));
        }

        /// <summary>A space an env replaced is no longer that env's: another env may take it (the replaced space is not disposed).</summary>
        [TestMethod]
        public void ReplacedSpace_IsReleased_ForAnotherEnv() {
            using var first = new CartPoleEnv();
            using var second = new CartPoleEnv();
            Space<NDArray> released = first.ActionSpace;
            first.ActionSpace = new Discrete(3);
            second.ActionSpace = released;
            Assert.AreSame(released, second.ActionSpace);
        }

        /// <summary>A deep copy belongs to no env, so it is the fix the refusal names.</summary>
        [TestMethod]
        public void DeepCopy_BelongsToNoEnv() {
            using var first = new CartPoleEnv();
            using var second = new CartPoleEnv();
            second.ObservationSpace = (Space<NDArray>)first.ObservationSpace.DeepCopy();
            Assert.AreEqual(first.ObservationSpace, second.ObservationSpace);
        }

        /// <summary>
        ///     A wrapper handed its own env's space (Gymnasium's <c>TransformObservation(env, f, env.observation_space)</c>) passes
        ///     it through instead of owning it: no refusal, and disposing the stack releases the space once, through the env.
        /// </summary>
        [TestMethod]
        public void Wrapper_HandedItsEnvsSpace_PassesItThrough() {
            var env = new CartPoleEnv();
            using var wrapper = new TransformObservation<NDArray, NDArray, NDArray>(env, observation => observation.copy(), env.ObservationSpace);
            Assert.AreSame(env.ObservationSpace, wrapper.ObservationSpace);
        }

        /// <summary>
        ///     Sub-env functions that share one space fail when the vector env builds them, where the first sub-env's disposal
        ///     would otherwise release the space under the others.
        /// </summary>
        [TestMethod]
        public void SyncVectorEnv_SubEnvsSharingASpace_AreRefused() {
            var shared = new Box(0, 1, new Shape(1));
            Func<IEnv>[] envFns = { () => new GenericTestEnv(observationSpace: shared), () => new GenericTestEnv(observationSpace: shared) };
            Assert.ThrowsExactly<InvalidOperationException>(() => new SyncVectorEnv(envFns));
        }
    }
}
