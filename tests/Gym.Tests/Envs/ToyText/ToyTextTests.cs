#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using Gym.Environments;
using Gym.Environments.Envs.ToyText;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Exceptions;
using Gym.Observations;
using Gym.Rendering;
using Gym.Spaces;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using SixLabors.ImageSharp;

namespace Gym.Tests.Envs.ToyText {
    /// <summary>
    ///     Headless contract tests of the toy_text envs of P5 (<see cref="BlackjackEnv"/>, <see cref="FrozenLakeEnv"/>,
    ///     <see cref="CliffWalkingEnv"/>, <see cref="TaxiEnv"/>) that need no Python: their registration and options records,
    ///     the Blackjack observation record, the errors before a reset, determinism, render modes and sizes, human mode, and
    ///     the public module functions. Byte-for-byte agreement with Gymnasium is the parity suite's job
    ///     (<c>tests/Gym.Tests.Parity/Tests/Envs</c>).
    /// </summary>
    [TestClass]
    public class ToyTextTests {
        /// <summary>
        ///     The six ids are registered in Gymnasium's order with its step limits, thresholds and kwargs (Blackjack's in
        ///     Gymnasium's order, <c>sab</c> then <c>natural</c>), and <c>make</c> builds each over the right class.
        /// </summary>
        [TestMethod]
        public void Registry_HoldsTheSixIds_WithGymnasiumsLimits() {
            GymEnvironments.Register();
            var expected = new (string Id, long? Steps, double? Threshold, Type Class, Info Kwargs)[] {
                ("Blackjack-v1", null, null, typeof(BlackjackEnv), new Info { ["sab"] = true, ["natural"] = false }),
                ("FrozenLake-v1", 100, 0.70, typeof(FrozenLakeEnv), new Info { ["map_name"] = "4x4" }),
                ("FrozenLake8x8-v1", 200, 0.85, typeof(FrozenLakeEnv), new Info { ["map_name"] = "8x8" }),
                ("CliffWalking-v1", null, null, typeof(CliffWalkingEnv), new Info()),
                ("CliffWalkingSlippery-v1", null, null, typeof(CliffWalkingEnv), new Info { ["is_slippery"] = true }),
                ("Taxi-v4", 200, 8.0, typeof(TaxiEnv), new Info()),
            };
            foreach ((string id, long? steps, double? threshold, Type type, Info kwargs) in expected) {
                EnvSpec spec = Spec(id);
                Assert.AreEqual(steps, spec.MaxEpisodeSteps, $"{id} max_episode_steps");
                Assert.AreEqual(threshold, spec.RewardThreshold, $"{id} reward_threshold");
                CollectionAssert.AreEqual(kwargs.Keys.ToList(), spec.Kwargs.Keys.ToList(), $"{id} kwargs keys, in order");
                Assert.AreEqual(kwargs, spec.Kwargs, $"{id} kwargs");
                using IEnv env = Make(id);
                Assert.IsInstanceOfType(env.Unwrapped, type, $"{id} builds a {type.Name}");
            }

            Assert.IsTrue(((BlackjackEnv)Make("Blackjack-v1").Unwrapped).Sab);
            Assert.AreEqual(8, ((FrozenLakeEnv)Make("FrozenLake8x8-v1").Unwrapped).Nrow);
            Assert.IsTrue(((CliffWalkingEnv)Make("CliffWalkingSlippery-v1").Unwrapped).IsSlippery);
        }

        /// <summary>The generated options records carry each constructor's Python kwargs in constructor order, and <c>make</c> takes them.</summary>
        [TestMethod]
        public void OptionsRecords_CarryThePythonKwargs() {
            Assert.AreEqual(new Info { ["natural"] = true, ["sab"] = false }, new BlackjackEnvOptions { Sab = false, Natural = true }.ToKwargs());
            Assert.AreEqual(new Info { ["map_name"] = "8x8", ["is_slippery"] = false }, new FrozenLakeEnvOptions { IsSlippery = false, MapName = "8x8" }.ToKwargs());
            Assert.AreEqual(new Info { ["is_slippery"] = true }, new CliffWalkingEnvOptions { IsSlippery = true }.ToKwargs());
            Assert.AreEqual(new Info { ["is_rainy"] = true, ["fickle_passenger"] = true, ["fickle_probability"] = 1.0 },
                new TaxiEnvOptions { FicklePassenger = true, IsRainy = true, FickleProbability = 1.0 }.ToKwargs());

            using IEnv taxi = Make("Taxi-v4", new TaxiEnvOptions { IsRainy = true, RainyProbability = 0.6 });
            Assert.AreEqual(0.6, ((TaxiEnv)taxi.Unwrapped).RainyProbability);
            Assert.AreEqual(0.2, ((TaxiEnv)taxi.Unwrapped).RainyLateralProbability, 1e-15);
        }

        /// <summary>
        ///     A Blackjack observation is Gym.NET's tuple of three 0-d int64 arrays, which <see cref="BlackjackObservation"/>
        ///     reads by name, sharing the arrays; the record is a member of the observation space (it is an
        ///     <see cref="ITuple"/>) and converts back.
        /// </summary>
        [TestMethod]
        public void BlackjackObservation_ReadsAnObservationByName() {
            var env = new BlackjackEnv(sab: true);
            object?[] items = env.Reset(seed: 3).Observation;
            BlackjackObservation observation = items;
            try {
                Assert.AreSame(items[0], observation.PlayerSum);
                Assert.AreSame(items[1], observation.DealerCard);
                Assert.AreSame(items[2], observation.UsableAce);
                Assert.AreEqual(BlackjackEnv.SumHand(env.Player!), observation.PlayerSum.GetAtIndex<long>(0), "player_sum is sum_hand of the player's cards.");
                Assert.AreEqual(env.Dealer![0], observation.DealerCard.GetAtIndex<long>(0), "dealer_card is the face-up card.");
                Assert.AreEqual(BlackjackEnv.UsableAce(env.Player!), observation.UsableAce.GetAtIndex<long>(0));
                Assert.AreEqual(0, observation.PlayerSum.ndim);
                Assert.AreEqual(NPTypeCode.Int64, observation.PlayerSum.typecode);

                Assert.IsTrue(env.ObservationSpace.Contains(items), "The generic tuple is a member.");
                Assert.IsTrue(env.ObservationSpace.Contains(observation), "So is the record, an ITuple.");
                ITuple tuple = observation;
                Assert.AreEqual(3, tuple.Length);
                Assert.AreSame(observation.UsableAce, tuple[2]);
                Assert.ThrowsExactly<IndexOutOfRangeException>(() => tuple[3]);

                object?[] back = observation;
                CollectionAssert.AreEqual(items, back, "The conversions share the arrays both ways.");
                Assert.AreEqual(observation, (BlackjackObservation)back, "Python's == on the items.");
            } finally {
                observation.Dispose();
            }
        }

        /// <summary>
        ///     The errors Python raises before the first reset, where its attributes do not exist yet: each step and each
        ///     image or text render names the attribute Python reaches first.
        /// </summary>
        [TestMethod]
        public void BeforeReset_RaisesPythonsAttributeErrors() {
            IEnv frozen = new FrozenLakeEnv(renderMode: "ansi");
            Assert.AreEqual("'FrozenLakeEnv' object has no attribute 's'", Assert.ThrowsExactly<AttributeError>(() => frozen.Step(0L)).Message);
            Assert.AreEqual("'FrozenLakeEnv' object has no attribute 's'", Assert.ThrowsExactly<AttributeError>(() => frozen.Render()).Message);

            IEnv cliff = new CliffWalkingEnv(renderMode: "rgb_array");
            Assert.AreEqual("'CliffWalkingEnv' object has no attribute 's'", Assert.ThrowsExactly<AttributeError>(() => cliff.Step(0L)).Message);
            Assert.AreEqual("'CliffWalkingEnv' object has no attribute 's'", Assert.ThrowsExactly<AttributeError>(() => cliff.Render()).Message);

            IEnv taxi = new TaxiEnv(renderMode: "ansi");
            Assert.AreEqual("'TaxiEnv' object has no attribute 's'", Assert.ThrowsExactly<AttributeError>(() => taxi.Step(0L)).Message);
            Assert.AreEqual("'TaxiEnv' object has no attribute 's'", Assert.ThrowsExactly<AttributeError>(() => taxi.Render()).Message);

            IEnv blackjack = new BlackjackEnv(renderMode: "rgb_array");
            Assert.AreEqual("'BlackjackEnv' object has no attribute 'player'", Assert.ThrowsExactly<AttributeError>(() => blackjack.Step(1L)).Message);
            Assert.AreEqual("'BlackjackEnv' object has no attribute 'dealer'", Assert.ThrowsExactly<AttributeError>(() => blackjack.Step(0L)).Message);
            Assert.AreEqual("'BlackjackEnv' object has no attribute 'player'", Assert.ThrowsExactly<AttributeError>(() => blackjack.Render()).Message);
        }

        /// <summary>The dict lookup of the table envs: an action no key equals raises Python's <c>KeyError</c> with its <c>repr</c>.</summary>
        [TestMethod]
        public void TableEnvs_LookActionsUpAsPythonsDict() {
            var frozen = new FrozenLakeEnv();
            frozen.Reset(seed: 0).Observation.Dispose();
            Assert.AreEqual("4", Assert.ThrowsExactly<KeyError>(() => ((IEnv)frozen).Step(4L)).Message);
            Assert.AreEqual("1.5", Assert.ThrowsExactly<KeyError>(() => ((IEnv)frozen).Step(1.5)).Message);
            using (NDArray vector = np.array(new[] { 1L }))
                Assert.AreEqual("unhashable type: 'numpy.ndarray'", Assert.ThrowsExactly<TypeError>(() => frozen.Step(vector)).Message);

            var taxi = new TaxiEnv();
            taxi.Reset(seed: 0).Observation.Dispose();
            Assert.AreEqual("6", Assert.ThrowsExactly<KeyError>(() => ((IEnv)taxi).Step(6L)).Message);
            // A float equal to a key is a key: 5.0 drops off.
            (((IEnv)taxi).Step(5.0).Observation as IDisposable)?.Dispose();
            Assert.AreEqual(5L, taxi.LastAction);
        }

        /// <summary>Two envs reset with the same seed and fed the same actions produce the same observations, rewards and flags.</summary>
        [TestMethod]
        public void Episodes_AreDeterministicGivenTheSeed() {
            foreach (Func<IEnv> create in new Func<IEnv>[] {
                         () => new FrozenLakeEnv(), () => new CliffWalkingEnv(isSlippery: true), () => new TaxiEnv(isRainy: true, ficklePassenger: true),
                         () => new BlackjackEnv(sab: true),
                     }) {
                IEnv first = create(), second = create();
                string[] a = Trajectory(first, 77), b = Trajectory(second, 77);
                CollectionAssert.AreEqual(a, b, $"{first}: same seed, same trajectory.");
            }
        }

        /// <summary><c>rgb_array</c> renders a frame of each env's size, and <c>ansi</c> renders the text maps.</summary>
        [TestMethod]
        public void Render_ReturnsFramesAndTextOfEachEnv() {
            var frames = new (IEnv Env, long Height, long Width)[] {
                (new FrozenLakeEnv(renderMode: "rgb_array"), 256, 256), (new FrozenLakeEnv(renderMode: "rgb_array", mapName: "8x8"), 512, 512),
                (new CliffWalkingEnv(renderMode: "rgb_array"), 256, 768), (new TaxiEnv(renderMode: "rgb_array"), 456, 536),
                (new BlackjackEnv(renderMode: "rgb_array"), 376, 384),
            };
            foreach ((IEnv env, long height, long width) in frames) {
                ResetAndRelease(env, 0);
                using NDArray frame = ((RenderResult.Rgb)env.Render()).Frame;
                CollectionAssert.AreEqual(new[] { height, width, 3L }, frame.shape, $"{env} frame shape");
                Assert.AreEqual(NPTypeCode.Byte, frame.typecode);
            }

            var frozen = new FrozenLakeEnv(renderMode: "ansi");
            frozen.Reset(seed: 0).Observation.Dispose();
            // No caption before a step (an empty line), then the map with the agent on the start tile, highlighted red.
            Assert.AreEqual("\n\u001b[41mS\u001b[0mFFF\nFHFH\nFFFH\nHFFG\n", ((RenderResult.Ansi)frozen.Render()).Text);

            var cliff = new CliffWalkingEnv(renderMode: "ansi");
            cliff.Reset(seed: 0).Observation.Dispose();
            string grid = ((RenderResult.Ansi)cliff.Render()).Text;
            StringAssert.EndsWith(grid, "x  C  C  C  C  C  C  C  C  C  C  T\n\n");

            var taxi = new TaxiEnv(renderMode: "ansi");
            taxi.Reset(seed: 0).Observation.Dispose();
            string map = ((RenderResult.Ansi)taxi.Render()).Text;
            StringAssert.StartsWith(map, "+---------+\n");
            StringAssert.EndsWith(map, "+---------+\n\n");
        }

        /// <summary>
        ///     <c>human</c> mode renders every reset and step into the viewer from <c>viewer_factory</c>, and <c>Close</c> shuts it,
        ///     for each env; without any factory the first human render raises Gymnasium's missing-pygame error (DR-20).
        /// </summary>
        [TestMethod]
        public void Render_Human_RendersIntoTheFactorysViewer() {
            foreach (Func<IEnvironmentViewerFactoryDelegate, IEnv> create in new Func<IEnvironmentViewerFactoryDelegate, IEnv>[] {
                         factory => new FrozenLakeEnv(renderMode: "human", viewerFactory: factory),
                         factory => new CliffWalkingEnv(renderMode: "human", viewerFactory: factory),
                         factory => new TaxiEnv(renderMode: "human", viewerFactory: factory),
                         factory => new BlackjackEnv(renderMode: "human", viewerFactory: factory),
                     }) {
                var viewer = new CountingViewer();
                IEnv env = create((_, _, _) => System.Threading.Tasks.Task.FromResult<IEnvViewer>(viewer));
                // A high rate keeps the pacing short; Gymnasium's tests raise render_fps the same way.
                env.Metadata[Metadata.RenderFpsKey] = 1000L;
                ResetAndRelease(env, 0);
                ReleaseObservation(env.Step(1L).Observation);
                Assert.IsTrue(env.Render().IsNone);
                Assert.AreEqual(3, viewer.Frames, $"{env}: reset, step and render each show a frame.");
                env.Close();
                Assert.IsTrue(viewer.Closed, $"{env}: close shuts the window.");
            }

            IEnvironmentViewerFactoryDelegate? previous = HumanViewerFactory;
            HumanViewerFactory = null;
            try {
                Assert.ThrowsExactly<DependencyNotInstalled>(() => new FrozenLakeEnv(renderMode: "human").Reset(seed: 0));
                Assert.ThrowsExactly<DependencyNotInstalled>(() => new CliffWalkingEnv(renderMode: "human").Reset(seed: 0));
                Assert.ThrowsExactly<DependencyNotInstalled>(() => new TaxiEnv(renderMode: "human").Reset(seed: 0));
                Assert.ThrowsExactly<DependencyNotInstalled>(() => new BlackjackEnv(renderMode: "human").Reset(seed: 0));
            } finally {
                HumanViewerFactory = previous;
            }
        }

        /// <summary>
        ///     Taxi's <c>encode</c> and <c>decode</c> are inverse over the 500 states, its action masks are int8 arrays of six,
        ///     and its start distribution is uniform over the 300 states with a waiting passenger not at the destination.
        /// </summary>
        [TestMethod]
        public void Taxi_EncodesDecodesAndMasks() {
            var taxi = new TaxiEnv();
            for (long state = 0; state < 500; state++) {
                (long row, long col, long passLoc, long destIdx) = taxi.Decode(state);
                Assert.AreEqual(state, taxi.Encode(row, col, passLoc, destIdx));
                using NDArray mask = taxi.ActionMask(state);
                Assert.AreEqual(NPTypeCode.SByte, mask.typecode);
                CollectionAssert.AreEqual(new[] { 6L }, mask.shape);
            }

            double[] distribution = taxi.InitialStateDistrib;
            Assert.AreEqual(300, distribution.Count(p => p > 0));
            Assert.AreEqual(1.0 / 300, distribution.Max());
            Assert.ThrowsExactly<AssertionError>(() => taxi.Decode(500));
        }

        /// <summary>
        ///     The module functions: FrozenLake's random maps have a path from start to goal; Blackjack's hand rules; the
        ///     <c>'c'</c> map builder refuses ragged rows with NumPy's message.
        /// </summary>
        [TestMethod]
        public void ModuleFunctions_FollowGymnasiumsRules() {
            IReadOnlyList<string> map = FrozenLakeEnv.GenerateRandomMap(size: 6, p: 0.7, seed: 4);
            Assert.AreEqual(6, map.Count);
            Assert.IsTrue(map.All(row => row.Length == 6));
            Assert.AreEqual('S', map[0][0]);
            Assert.AreEqual('G', map[5][5]);
            Assert.IsTrue(FrozenLakeEnv.IsValid(map, 6));
            CollectionAssert.AreEqual(map.ToList(), FrozenLakeEnv.GenerateRandomMap(size: 6, p: 0.7, seed: 4).ToList(), "A seed fixes the map.");

            Assert.IsTrue(BlackjackEnv.IsNatural(new long[] { 10, 1 }));
            Assert.IsFalse(BlackjackEnv.IsNatural(new long[] { 10, 1, 10 }));
            Assert.AreEqual(21, BlackjackEnv.SumHand(new long[] { 1, 10 }));
            Assert.AreEqual(12, BlackjackEnv.SumHand(new long[] { 1, 1 }), "Only one ace counts as 11.");
            Assert.AreEqual(0, BlackjackEnv.Score(new long[] { 10, 10, 5 }), "A bust scores 0.");
            Assert.AreEqual(-1.0, BlackjackEnv.Cmp(17, 20));
            Assert.AreEqual(0.0, BlackjackEnv.Cmp(double.NaN, 1), "NaN compares neither way.");

            Assert.AreEqual("setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (2,) + inhomogeneous part.",
                Assert.ThrowsExactly<ValueError>(() => ToyTextUtils.CharMap(new[] { "SF", "FFG" })).Message);
            using NDArray chars = ToyTextUtils.CharMap(new[] { "SF", "HG" });
            CollectionAssert.AreEqual(new[] { 2L, 2L }, chars.shape);
            Assert.AreEqual("|S1", chars.dtype.str);
        }

        /// <summary>A seeded trajectory of 40 steps (resetting after each episode), each step as text.</summary>
        /// <param name="env">The env.</param>
        /// <param name="seed">The seed of the first reset and of the action space.</param>
        /// <returns>One line per step: observation, reward, flags.</returns>
        private static string[] Trajectory(IEnv env, long seed) {
            var lines = new List<string>();
            ResetAndRelease(env, seed);
            env.ActionSpace.Seed(seed);
            for (int i = 0; i < 40; i++) {
                object action = env.ActionSpace.Sample()!;
                StepResult<object?> step = env.Step(action);
                lines.Add($"{Gym.Utils.PyBuiltins.Repr(step.Observation)} {step.Reward} {step.Terminated} {step.Truncated}");
                ReleaseObservation(step.Observation);
                (action as IDisposable)?.Dispose();
                if (step.Terminated || step.Truncated)
                    ResetAndRelease(env, null);
            }

            return lines.ToArray();
        }

        /// <summary>Resets an env and releases the observation's arrays.</summary>
        /// <param name="env">The env.</param>
        /// <param name="seed">The seed, or <see langword="null"/>.</param>
        private static void ResetAndRelease(IEnv env, long? seed) => ReleaseObservation(env.Reset(seed).Observation);

        /// <summary>Releases an observation's arrays: an array, or each array of a tuple.</summary>
        /// <param name="observation">The observation.</param>
        private static void ReleaseObservation(object? observation) {
            if (observation is object?[] items) {
                foreach (object? item in items)
                    (item as IDisposable)?.Dispose();
            } else {
                (observation as IDisposable)?.Dispose();
            }
        }

        /// <summary>A viewer that counts the frames it is shown.</summary>
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
