#nullable enable
#pragma warning disable CS0618 // The obsolete shims are what these tests pin.
using System;
using System.Collections.Generic;
using System.Reflection;
using Gym.Environments;
using Gym.Environments.Envs.Box2D;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Observations;
using Gym.Spaces;
using Gym.Vector;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Legacy {
    /// <summary>
    ///     Pins DR-58: the <c>[Obsolete]</c> shims that keep 0.2 code compiling until 2.0, each found by compiling 0.2.0's own
    ///     tests against 1.0. Every shim must mean what the 0.2 line meant and spell out 1.0's equivalent; the limits the docs
    ///     state (what a shim does not catch or keep) are pinned too.
    /// </summary>
    [TestClass]
    public class CompatibilityShimTests {
        /// <summary><c>Info.Get&lt;T&gt;</c> reads as the legacy <c>Dict.Get&lt;T&gt;</c> did: the value cast, <c>default(T)</c> when missing, and only the exact boxed type.</summary>
        [TestMethod]
        public void InfoGetT_ReadsLikeTheLegacyDict() {
            var info = new Info { ["steps"] = 5L, ["name"] = "cart" };

            Assert.AreEqual(5L, info.Get<long>("steps"));
            Assert.AreEqual("cart", info.Get<string>("name"));
            Assert.AreEqual(0L, info.Get<long>("missing"));
            Assert.IsNull(info.Get<string>("missing"));
            // A boxed long is not an int, as NumPy's integers arrive: the documented cast failure.
            Assert.ThrowsExactly<InvalidCastException>(() => info.Get<int>("steps"));
        }

        /// <summary>
        ///     The reason <c>Get&lt;T&gt;(key, T fallback)</c> was not added: <c>Get("k", 5)</c> still binds to Python's
        ///     <c>get</c>, returning the stored <see cref="long"/> rather than casting it to <see cref="int"/>.
        /// </summary>
        [TestMethod]
        public void InfoGetWithFallback_StillBindsToPythonsGet() {
            var info = new Info { ["k"] = 7L };

            object? stored = info.Get("k", 5);
            object? fallback = info.Get("missing", 5);

            Assert.IsInstanceOfType<long>(stored);
            Assert.AreEqual(7L, stored);
            Assert.AreEqual(5, fallback);
        }

        /// <summary><c>Info.Clone</c> is <c>Copy</c>: a new info of the same type (a <see cref="Metadata"/> stays one), the same entries in order, the values shared.</summary>
        [TestMethod]
        public void InfoClone_IsAShallowCopyOfTheSameType() {
            var modes = new List<string> { "human", "rgb_array" };
            var metadata = new Metadata { [Metadata.RenderModesKey] = modes, [Metadata.RenderFpsKey] = 50L };

            Info clone = metadata.Clone();

            Assert.IsInstanceOfType<Metadata>(clone);
            Assert.AreNotSame(metadata, clone);
            CollectionAssert.AreEqual(new List<string>(metadata.Keys), new List<string>(clone.Keys));
            Assert.AreSame(modes, clone[Metadata.RenderModesKey]);
        }

        /// <summary>
        ///     0.2's three exceptions are obsolete bases of Gymnasium's: each Gymnasium error is still what it was (an
        ///     <see cref="Error"/> or not, the same message), and 0.2's parameterless constructors keep 0.2's messages.
        /// </summary>
        [TestMethod]
        public void LegacyExceptions_AreBasesOfGymnasiums() {
            Assert.IsInstanceOfType<InvalidActionError>(new InvalidAction("m"));
            Assert.IsInstanceOfType<Error>(new InvalidAction("m"));
            Assert.AreEqual("", new InvalidAction().Message);
            Assert.IsInstanceOfType<AlreadySteppingError>(new AlreadyPendingCallError("m", "step"));
            Assert.IsNotInstanceOfType<Error>(new AlreadyPendingCallError("m", "step"));
            Assert.IsInstanceOfType<NotSteppingError>(new NoAsyncCallError("m", "step"));
            Assert.IsNotInstanceOfType<Error>(new NoAsyncCallError("m", "step"));

            Assert.AreEqual("Action is outside of the configured action space.", new InvalidActionError().Message);
            Assert.AreEqual("already running an async step", new AlreadySteppingError().Message);
            Assert.AreEqual("not running an async step", new NotSteppingError().Message);
        }

        /// <summary>
        ///     <c>catch (InvalidActionError)</c> catches CarRacing's <see cref="InvalidAction"/>, and, the documented limit,
        ///     not LunarLander's <see cref="NumSharp.AssertionError"/>, which Gymnasium's envs raise by asserting the action.
        /// </summary>
        [TestMethod]
        public void CatchInvalidActionError_CatchesInvalidAction_NotTheEnvsAssertions() {
            using (var carRacing = new CarRacingEnv(continuous: false)) {
                carRacing.Reset(seed: 0).Dispose();
                using NDArray outside = NDArray.Scalar(9L);
                string? caught = null;
                try {
                    carRacing.Step(outside).Dispose();
                } catch (InvalidActionError e) {
                    caught = e.Message;
                }

                Assert.AreEqual("you passed the invalid action `9`. The supported action_space is `Discrete(5)`", caught);
            }

            using var lunarLander = new LunarLanderEnv();
            lunarLander.Reset(seed: 0).Dispose();
            using NDArray refused = NDArray.Scalar(7L);
            Exception thrown = Assert.ThrowsExactly<AssertionError>(() => lunarLander.Step(refused).Dispose());
            Assert.IsNotInstanceOfType<InvalidActionError>(thrown);
        }

        /// <summary>
        ///     0.2's async errors catch <c>AsyncVectorEnv</c>'s: a second <c>StepAsync</c> while one is pending raises
        ///     <see cref="AlreadyPendingCallError"/>, a <c>StepWait</c> with none pending <see cref="NoAsyncCallError"/>.
        /// </summary>
        [TestMethod]
        public void CatchLegacyAsyncErrors_CatchesAsyncVectorEnvsErrors() {
            using var vector = new AsyncVectorEnv(new Func<IEnv>[] { () => new CartPoleEnv(), () => new CartPoleEnv() });
            vector.Reset(seed: 0).Dispose();
            using NDArray actions = np.array(new long[] { 0, 1 });

            string? notStepping = null;
            try {
                vector.StepWait().Dispose();
            } catch (NotSteppingError e) {
                notStepping = e.Message;
            }

            vector.StepAsync(actions);
            string? alreadyStepping = null;
            try {
                vector.StepAsync(actions);
            } catch (AlreadySteppingError e) {
                alreadyStepping = e.Message;
            }

            vector.StepWait().Dispose();
            Assert.AreEqual("Calling `step_wait` without any prior call to `step_async`.", notStepping);
            Assert.AreEqual("Calling `step_async` while waiting for a pending call to `step` to complete.", alreadyStepping);
        }

        /// <summary>0.2's four-element deconstruction: <c>done</c> is <c>terminated || truncated</c>, the other three the result's own.</summary>
        [TestMethod]
        public void StepResult_DeconstructsIntoZeroPointTwosFour() {
            using NDArray observation = np.array(new float[] { 1f, 2f });
            var info = new Info { ["k"] = 1L };
            foreach ((bool terminated, bool truncated) in new[] { (false, false), (true, false), (false, true), (true, true) }) {
                var step = new StepResult<NDArray>(observation, 0.5, terminated, truncated, info);

                var (obs, reward, done, stepInfo) = step;

                Assert.AreSame(observation, obs);
                Assert.AreEqual(0.5, reward);
                Assert.AreEqual(terminated || truncated, done);
                Assert.AreSame(info, stepInfo);
            }
        }

        /// <summary>0.2's loop over a real env: the four-tuple's <c>done</c> ends the episode where the five-tuple's flags do.</summary>
        [TestMethod]
        public void StepResult_FourTuple_EndsCartPolesEpisode() {
            using var env = new CartPoleEnv();
            env.Reset(seed: 3).Dispose();
            using NDArray push = NDArray.Scalar(1L);
            int steps = 0;
            bool done = false;
            while (!done) {
                using StepResult<NDArray> step = env.Step(push);
                (_, _, done, _) = step;
                Assert.AreEqual(step.Terminated || step.Truncated, done);
                steps++;
            }

            Assert.IsTrue(steps is > 0 and < 500);
        }

        /// <summary>0.2's <c>NDArray obs = env.Reset();</c>: the result converts to its own observation, the same instance.</summary>
        [TestMethod]
        public void ResetResult_ConvertsToItsObservation() {
            using var env = new CartPoleEnv();
            ResetResult<NDArray> reset = env.Reset(seed: 0);

            NDArray observation = reset;

            Assert.AreSame(reset.Observation, observation);
            reset.Dispose();
        }

        /// <summary><c>CloseEnvironment</c> calls <c>Close</c>, the override included, from the class and through <see cref="IEnv"/>.</summary>
        [TestMethod]
        public void CloseEnvironment_CallsClose() {
            using var wrapper = new CloseCounter(new CartPoleEnv());

            wrapper.CloseEnvironment();
            ((IEnv)wrapper).CloseEnvironment();

            Assert.AreEqual(2, wrapper.Closes);
        }

        /// <summary>0.2's <c>new Box(low, high, dtype)</c>, positional and with 0.2's <c>dType:</c> name, builds the box <c>dtype:</c> builds in 1.0.</summary>
        [TestMethod]
        public void Box_ZeroPointTwosDtypeThird_BuildsTheSameBox() {
            using var expected = new Box(-5.0, 5.0, dtype: np.float32);
            using var positional = new Box(-5.0f, 5.0f, np.float32);
            using var named = new Box(-5.0, 5.0, dType: np.float32);
            Assert.AreEqual(expected.ToString(), positional.ToString());
            Assert.AreEqual(expected.ToString(), named.ToString());

            using NDArray low = np.array(new double[] { -1, -2 });
            using NDArray high = np.array(new double[] { 1, 2 });
            using var expectedArrays = new Box(low, high, dtype: np.float64);
            using var positionalArrays = new Box(low, high, np.float64);
            Assert.AreEqual(expectedArrays.ToString(), positionalArrays.ToString());
        }

        /// <summary>0.2's <c>new CartPoleEnv(factory)</c>: <c>NullEnvViewer.Factory</c> or none is headless, any other factory human mode.</summary>
        [TestMethod]
        public void CartPoleEnvWithFactory_IsHeadlessForNullViewer_HumanOtherwise() {
            using (var headless = new CartPoleEnv(NullEnvViewer.Factory))
                Assert.IsNull(headless.RenderMode);
            using (var none = new CartPoleEnv((IEnvironmentViewerFactoryDelegate?)null))
                Assert.IsNull(none.RenderMode);

            IEnvironmentViewerFactoryDelegate window = (width, height, title) => NullEnvViewer.Run(width, height, title);
            using var human = new CartPoleEnv(window);
            Assert.AreEqual("human", human.RenderMode);
        }

        /// <summary>LunarLander's 0.2 names: the action enum in both namespaces, <c>ContinuousMode</c>, and 0.2's float constants.</summary>
        [TestMethod]
        public void LunarLanderZeroPointTwoNames_MapToGymnasiums() {
            Assert.AreEqual(0, (int)LunarLanderDiscreteActions.Nothing);
            Assert.AreEqual(1, (int)LunarLanderDiscreteActions.FireLeftThruster);
            Assert.AreEqual(2, (int)LunarLanderDiscreteActions.FireMainEngine);
            Assert.AreEqual(3, (int)LunarLanderDiscreteActions.FireRightThruster);
            Assert.AreEqual(2, (int)Gym.Environments.Envs.Aether.LunarLanderDiscreteActions.FireMainEngine);

            // Read by reflection: as constants they would fold at compile time, and the field type is part of the point
            // (0.2's constants were floats).
            foreach ((string name, float value) in new[] { ("SCALE", 30f), ("MAIN_ENGINE_POWER", 13f), ("SIDE_ENGINE_POWER", 0.6f), ("INITIAL_RANDOM", 1000f) }) {
                FieldInfo field = typeof(LunarLanderEnv).GetField(name)!;
                Assert.AreEqual(typeof(float), field.FieldType, name);
                Assert.AreEqual(value, (float)field.GetValue(null)!, name);
            }

            using var continuous = new LunarLanderEnv(continuous: true);
            using var discrete = new LunarLanderEnv();
            Assert.IsTrue(continuous.ContinuousMode);
            Assert.IsFalse(discrete.ContinuousMode);
        }

        /// <summary>A wrapper that counts its <c>Close</c> calls, to see <c>CloseEnvironment</c> reach the override.</summary>
        private sealed class CloseCounter : Wrapper<NDArray, NDArray> {
            /// <summary>Wraps an env.</summary>
            /// <param name="env">The env to wrap.</param>
            public CloseCounter(Env<NDArray, NDArray> env) : base(env) { }

            /// <summary>How many times <see cref="Close"/> ran.</summary>
            public int Closes { get; private set; }

            /// <summary>Counts, then closes the wrapped env.</summary>
            public override void Close() {
                Closes++;
                base.Close();
            }
        }
    }
}
