#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Environments;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Exceptions;
using Gym.Observations;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using SixLabors.ImageSharp;

namespace Gym.Tests.Envs.Classic {
    /// <summary>
    ///     Headless contract tests of <see cref="CartPoleEnv"/> on Gymnasium's API: it steps with what its own action space
    ///     samples (trap 1), accepts actions of any integer width through the untyped <see cref="IEnv.Step"/>, its
    ///     observations are members of its observation space, and its render mode, fixed at construction, decides what
    ///     <see cref="CartPoleEnv.Render"/> does.
    /// </summary>
    /// <remarks>
    ///     The tests dispose the initial observations they do not keep. Every observation is a fresh float32 copy of the state
    ///     (the port fixed B-13's aliasing), so disposing one never touches the env.
    /// </remarks>
    [TestClass]
    public class CartPoleTests {
        /// <summary>
        ///     Stepping with <c>ActionSpace.Sample()</c> works. The sample is a 0-d int64 array, which the former
        ///     <c>(int)action</c> could not unbox (<see cref="InvalidCastException"/>); the untyped step converts it.
        /// </summary>
        [TestMethod]
        public void Step_WithSampledActions_RunsEpisodes() {
            var env = new CartPoleEnv();
            IEnv untyped = env;
            env.ActionSpace.Seed(0);
            env.Reset(seed: 0).Observation.Dispose();

            for (int i = 0; i < 200; i++) {
                using NDArray action = env.ActionSpace.Sample();
                var (_, _, terminated, truncated, _) = untyped.Step(action);
                if (terminated || truncated)
                    env.Reset().Observation.Dispose();
            }
        }

        /// <summary>
        ///     The same push results from the same action value whatever boxed it: <see langword="int"/>, <see langword="long"/>
        ///     (what the legacy <c>VecEnv.Step(long)</c> sends), <see langword="byte"/>, a 0-d array, or <see langword="true"/>.
        /// </summary>
        [TestMethod]
        public void Step_ActionOfAnyIntegerWidth_GivesTheSameTransition() {
            var expected = StepOnce(1);
            foreach (object action in new object[] { 1L, (byte)1, (short)1, 1u, NDArray.Scalar<long>(1L), true })
                Assert.IsTrue(np.array_equal(expected, StepOnce(action)), $"Action {action} ({action.GetType().Name}) must push right like 1.");
        }

        /// <summary>
        ///     A value that is not an element of <c>Discrete(2)</c> is rejected as Gymnasium's <c>assert</c> rejects it: an
        ///     <see cref="AssertionError"/> (in Release builds too, B-14) with Python's rendering of the action and its type,
        ///     through the typed and the untyped step alike.
        /// </summary>
        [TestMethod]
        public void Step_ActionOutsideTheActionSpace_ThrowsGymnasiumsAssertionError() {
            var env = new CartPoleEnv();
            IEnv untyped = env;
            env.Reset(seed: 0).Observation.Dispose();

            Assert.AreEqual("0.5 (<class 'float'>) invalid", Assert.ThrowsExactly<AssertionError>(() => untyped.Step(0.5)).Message);
            Assert.AreEqual("2 (<class 'int'>) invalid", Assert.ThrowsExactly<AssertionError>(() => env.Step(2L)).Message);
            Assert.AreEqual("2 (<class 'int'>) invalid", Assert.ThrowsExactly<AssertionError>(() => untyped.Step(2)).Message);
            Assert.AreEqual("np.int64(5) (<class 'numpy.int64'>) invalid", Assert.ThrowsExactly<AssertionError>(() => untyped.Step(NDArray.Scalar<long>(5L))).Message);
            Assert.AreEqual("'x' (<class 'str'>) invalid", Assert.ThrowsExactly<AssertionError>(() => untyped.Step("x")).Message);
            Assert.AreEqual("None (<class 'NoneType'>) invalid", Assert.ThrowsExactly<AssertionError>(() => untyped.Step(null)).Message);
        }

        /// <summary>Stepping before the first reset raises Gymnasium's <c>Call reset before using step method.</c></summary>
        [TestMethod]
        public void Step_BeforeReset_ThrowsGymnasiumsAssertionError() {
            var env = new CartPoleEnv();

            Assert.AreEqual("Call reset before using step method.", Assert.ThrowsExactly<AssertionError>(() => env.Step(1)).Message);
        }

        /// <summary>
        ///     Observations are float32 copies of the float64 <see cref="CartPoleEnv.State"/>, as Gymnasium's
        ///     <c>np.array(self.state, dtype=np.float32)</c>: writing to one leaves the env alone (B-13).
        /// </summary>
        [TestMethod]
        public void Observations_AreFloat32Copies_OfTheFloat64State() {
            var env = new CartPoleEnv();
            using NDArray observation = env.Reset(seed: 0).Observation;
            double[] state = (double[])env.State!.Clone();

            Assert.AreEqual(np.float32, observation.dtype);
            CollectionAssert.AreEqual(state.Select(v => (float)v).ToArray(), observation.ToArray<float>());
            observation.SetSingle(123f, 0);
            CollectionAssert.AreEqual(state, env.State, "Writing to the observation must not move the env.");
        }

        /// <summary>
        ///     Stepping after termination keeps going with reward 0.0, and warns exactly once (Gymnasium's
        ///     <c>logger.warn</c>), captured here as Python's <c>catch_warnings(record=True)</c> would capture it.
        /// </summary>
        [TestMethod]
        public void Step_AfterTermination_WarnsOnceAndRewardsZero() {
            var env = new CartPoleEnv();
            env.Reset(seed: 3).Observation.Dispose();
            while (!env.Step(1).Terminated) { }

            using WarningCatcher catcher = Warnings.Catch();
            var rewards = Enumerable.Range(0, 3).Select(_ => env.Step(1).Reward).ToArray();

            CollectionAssert.AreEqual(new[] { 0.0, 0.0, 0.0 }, rewards);
            Assert.AreEqual(1, catcher.Records.Count);
            StringAssert.StartsWith(catcher.Records[0].Message, "\u001b[33mWARN: You are calling 'step()' even though this environment has already returned terminated = True.");
            Assert.AreEqual(3, env.StepsBeyondTerminated);
        }

        /// <summary>The unwrapped env never truncates and returns an empty info, as Gymnasium's (<c>TimeLimit</c> truncates).</summary>
        [TestMethod]
        public void Step_NeverTruncates_AndReturnsAnEmptyInfo() {
            var env = new CartPoleEnv();
            var (_, info) = env.Reset(seed: 0);
            Assert.AreEqual(0, info.Count);

            for (int i = 0; i < 600; i++) {
                StepResult<NDArray> step = env.Step(i % 2);
                Assert.IsFalse(step.Truncated);
                Assert.AreEqual(0, step.Info.Count);
            }
        }

        /// <summary>Every step hands out a new array object, never the env's state (B-13).</summary>
        [TestMethod]
        public void Step_ConsecutiveObservations_AreDistinctObjects() {
            var env = new CartPoleEnv();
            env.Reset(seed: 0).Observation.Dispose();

            var first = env.Step(0).Observation;
            var second = env.Step(1).Observation;

            Assert.IsFalse(ReferenceEquals(first, second));
        }

        /// <summary>The initial observation, float32 like the space, is a member of the observation space.</summary>
        [TestMethod]
        public void ObservationSpace_ContainsTheResetObservation() {
            var env = new CartPoleEnv();

            Assert.IsTrue(env.ObservationSpace.Contains(env.Reset(seed: 0).Observation));
        }

        /// <summary>
        ///     The generator is created lazily, as Gymnasium's <c>np_random</c> is: none exists after construction, a seeded
        ///     reset records its seed, and an unseeded reset keeps both the generator and the seed.
        /// </summary>
        [TestMethod]
        public void NpRandom_IsLazy_AndSeededByReset() {
            var env = new CartPoleEnv();
            Assert.IsNull(PyAttributes.GetAttr(env, "_np_random"));

            env.Reset(seed: 42).Observation.Dispose();
            var generator = env.NpRandom;
            Assert.AreEqual(42, (int)env.NpRandomSeed);
            env.Reset().Observation.Dispose();
            Assert.AreSame(generator, env.NpRandom);
            Assert.AreEqual(42, (int)env.NpRandomSeed);
        }

        /// <summary>Metadata, spaces and attributes by their Python names: <c>metadata</c>, <c>gravity</c>, <c>_sutton_barto_reward</c>, <c>isopen</c>.</summary>
        [TestMethod]
        public void Attributes_HaveGymnasiumsNamesAndValues() {
            var env = new CartPoleEnv(suttonBartoReward: true);

            Assert.AreEqual("{'render_modes': ['human', 'rgb_array'], 'render_fps': 50}", env.Metadata.ToString());
            Assert.AreEqual(9.8, env.GetWrapperAttr("gravity"));
            Assert.AreEqual(true, env.GetWrapperAttr("_sutton_barto_reward"));
            Assert.AreEqual(true, env.GetWrapperAttr("isopen"));
            Assert.IsNull(env.GetWrapperAttr("state"));
            Assert.IsNull(env.RenderMode);
            Assert.IsNull(env.Spec);
            Assert.AreEqual("<CartPoleEnv instance>", env.ToString());
            env.Spec = new EnvSpec("CartPole-v1", "Gym.Environments.Envs.Classic:CartPoleEnv", maxEpisodeSteps: 500);
            Assert.AreEqual("<CartPoleEnv<CartPole-v1>>", env.ToString());
        }

        /// <summary><c>rgb_array</c> returns a fresh <c>(400, 600, 3)</c> uint8 frame after a reset and nothing before it.</summary>
        [TestMethod]
        public void Render_RgbArray_ReturnsTheFrame() {
            var env = new CartPoleEnv(renderMode: "rgb_array");
            Assert.IsTrue(env.Render().IsNone, "Gymnasium returns None before the first reset.");
            env.Reset(seed: 0).Observation.Dispose();

            var frame = (RenderResult.Rgb)env.Render();
            using NDArray pixels = frame.Frame;

            CollectionAssert.AreEqual(new long[] { 400, 600, 3 }, pixels.shape);
            Assert.AreEqual(np.uint8, pixels.dtype);
        }

        /// <summary>
        ///     Without a render mode, Gymnasium asserts a spec exists (a bare <see cref="AssertionError"/> for an env built
        ///     with its constructor), and with a spec it warns, naming the id, and returns <c>None</c>.
        /// </summary>
        [TestMethod]
        public void Render_WithoutRenderMode_AssertsSpecThenWarns() {
            var env = new CartPoleEnv();
            env.Reset(seed: 0).Observation.Dispose();

            Assert.AreEqual("", Assert.ThrowsExactly<AssertionError>(() => env.Render()).Message);

            env.Spec = new EnvSpec("CartPole-v1");
            using WarningCatcher catcher = Warnings.Catch();
            Assert.IsTrue(env.Render().IsNone);
            Assert.AreEqual(1, catcher.Records.Count);
            StringAssert.Contains(catcher.Records[0].Message, "You are calling render method without specifying any render mode. You can specify the render_mode at initialization, e.g. gym.make(\"CartPole-v1\", render_mode=\"rgb_array\")");
        }

        /// <summary>
        ///     <c>human</c> mode renders into the viewer on every reset and step, paced by the clock at <c>render_fps</c>, and
        ///     <c>render()</c> returns nothing; <see cref="CartPoleEnv.Close"/> closes the window and keeps the screen.
        /// </summary>
        [TestMethod]
        public void Render_Human_RendersEveryStepIntoTheViewer() {
            var viewer = new CountingViewer();
            var env = new CartPoleEnv(viewer);
            // A high rate keeps the pacing short; Gymnasium's tests raise render_fps the same way.
            env.Metadata[Metadata.RenderFpsKey] = 1000L;

            env.Reset(seed: 0).Observation.Dispose();
            env.Step(1).Observation.Dispose();
            env.Step(0).Observation.Dispose();
            Assert.IsTrue(env.Render().IsNone);

            Assert.AreEqual(4, viewer.Frames);
            Assert.IsNotNull(env.Screen);
            Assert.IsNotNull(env.Clock);
            env.Close();
            Assert.IsTrue(viewer.Closed);
            Assert.IsFalse(env.IsOpen);
            Assert.IsNotNull(env.Screen, "Python's close keeps self.screen.");
        }

        /// <summary>A render mode CartPole does not list (<c>rgb_array_list</c>) draws each frame and returns nothing, as Python's <c>render</c> does.</summary>
        [TestMethod]
        public void Render_UnlistedMode_DrawsAndReturnsNothing() {
            var env = new CartPoleEnv(renderMode: "rgb_array_list");
            env.Reset(seed: 0).Observation.Dispose();

            Assert.IsTrue(env.Render().IsNone);
            Assert.IsNotNull(env.Surf);
        }

        /// <summary>Resets a fresh seed-0 env and returns the observation after one untyped step with <paramref name="action"/>.</summary>
        /// <param name="action">The action to step with.</param>
        /// <returns>The observation of that step.</returns>
        private static NDArray StepOnce(object action) {
            var env = new CartPoleEnv();
            env.Reset(seed: 0).Observation.Dispose();
            return (NDArray)((IEnv)env).Step(action).Observation!;
        }

        /// <summary>A viewer that counts the frames it is shown and records being closed.</summary>
        private sealed class CountingViewer : IEnvViewer {
            /// <summary>The frames rendered so far.</summary>
            public int Frames { get; private set; }

            /// <summary>Whether <see cref="CloseEnvironment"/> ran.</summary>
            public bool Closed { get; private set; }

            /// <summary>Counts the frame.</summary>
            /// <param name="img">The frame; not kept.</param>
            public void Render(Image img) => Frames++;

            /// <summary>Records the close.</summary>
            public void CloseEnvironment() => Closed = true;

            /// <summary>Nothing to release.</summary>
            public void Dispose() { }
        }
    }
}
