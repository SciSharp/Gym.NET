using System;
using System.Globalization;
using System.Linq;
using System.Numerics;
using Gym.Environments;
using Gym.Environments.Envs.Aether;
using Gym.Environments.Envs.Classic;
using Gym.Exceptions;
using Gym.Spaces;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Utils {
    /// <summary>
    ///     Pins Gym.NET's seeding, which follows Gymnasium's: any non-negative seed is valid, a negative one raises
    ///     Gymnasium's <see cref="Error"/> with its message, and the reported seed is the one given, or 128-bit entropy for an
    ///     unseeded generator. The legacy <c>RandomState</c> rules (a 2^32 - 1 ceiling, NumPy's <c>ValueError</c>) no longer
    ///     apply.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/seeding.py">gymnasium/utils/seeding.py</see>.</remarks>
    [TestClass]
    public class SeedTests {
        /// <summary>Gymnasium's message for a negative seed, before the value.</summary>
        private const string NegativeSeedMessage = "Seed must be greater or equal to zero, actual value: ";

        /// <summary>Seeds that NumPy's legacy RandomState rejected or could not take from an <see cref="int"/>, all valid for Gymnasium.</summary>
        private static readonly long[] LargeSeeds = { (1L << 31) + 5, (1L << 32) - 1, 1L << 32, 1L << 40, long.MaxValue };

        /// <summary>
        ///     Gymnasium's <c>test_valid_seeds</c>, extended with large seeds: <c>np_random(seed)</c> reports the seed it was
        ///     given, including one beyond <see cref="long"/>.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_seeding.py#L17">tests/utils/test_seeding.py (test_valid_seeds)</see>.</remarks>
        [TestMethod]
        public void NpRandom_ValidSeeds_ReportTheSeed()
        {
            foreach (long seed in new[] { 0L, 1L }.Concat(LargeSeeds))
                Assert.AreEqual(new BigInteger(seed), Seeding.NpRandom(seed).Seed, $"Seed {seed}");

            var beyondLong = BigInteger.Pow(2, 100) + 7;
            Assert.AreEqual(beyondLong, Seeding.NpRandom(beyondLong).Seed);
        }

        /// <summary>
        ///     Gymnasium's <c>test_invalid_seeds</c> for -1 (a string cannot reach a typed seed): a negative seed raises
        ///     <see cref="Error"/> with Gymnasium's message.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_seeding.py#L7">tests/utils/test_seeding.py (test_invalid_seeds)</see>.</remarks>
        [TestMethod]
        public void NpRandom_NegativeSeed_ThrowsGymnasiumError()
        {
            foreach (long seed in new[] { -1L, long.MinValue })
            {
                var error = Assert.ThrowsExactly<Error>(() => Seeding.NpRandom(seed));
                Assert.AreEqual(NegativeSeedMessage + seed.ToString(CultureInfo.InvariantCulture), error.Message);
            }
        }

        /// <summary>
        ///     An unseeded generator reports the 128-bit entropy it drew, as Gymnasium's <c>np_random()</c> does, different on
        ///     every call; passing it back rebuilds the same stream. There is no exception for an entropy whose top 32-bit word
        ///     is zero any more: NumSharp 0.80.0 builds the unseeded pool from the reported integer itself, as NumPy does.
        /// </summary>
        [TestMethod]
        public void NpRandom_Unseeded_ReportsItsEntropyAndRebuildsFromIt()
        {
            var (rng, seed) = Seeding.NpRandom();

            Assert.IsTrue(seed.Sign >= 0 && seed < BigInteger.One << 128, $"{seed} is not a 128-bit value.");
            Assert.AreNotEqual(seed, Seeding.NpRandom().Seed, "Two unseeded generators must draw different entropy.");
            var (rebuilt, _) = Seeding.NpRandom(seed);
            using NDArray original = rng.random(8);
            using NDArray again = rebuilt.random(8);
            Assert.IsTrue(np.array_equal(original, again), "The reported seed must rebuild the unseeded generator's stream.");
        }

        /// <summary>
        ///     Seeds beyond the former 2^32 - 1 ceiling are accepted by every space and env (an env through
        ///     <c>Reset(seed)</c>, even beyond <see cref="long"/>), and the same seed replays the same stream (the Gymnasium
        ///     golden pins the values themselves for seeds up to 2^40).
        /// </summary>
        [TestMethod]
        public void Seed_LargeSeeds_AreAcceptedAndReproducible()
        {
            foreach (long seed in LargeSeeds)
            {
                var first = new Discrete(1_000_000, seed: seed);
                var second = new Discrete(1_000_000, seed: 5);
                second.Seed(seed);
                using (NDArray a = first.Sample())
                using (NDArray b = second.Sample())
                    Assert.AreEqual((long) a, (long) b, $"Seed {seed}: construction and Seed must give the same stream.");

                new Box(-1f, 1f, new Shape(2), np.float32, seed: seed).Seed(seed);
                AssertSameReset(seed);
                new LunarLanderEnv(NullEnvViewer.Factory, random_state: Seeding.NpRandom(0).Rng).Reset(seed: seed).Observation.Dispose();
            }

            AssertSameReset(BigInteger.Pow(2, 100) + 7);
        }

        /// <summary>Every space and env rejects a negative seed with Gymnasium's error and message.</summary>
        [TestMethod]
        public void Seed_Negative_ThrowsGymnasiumErrorEverywhere()
        {
            foreach (long seed in new[] { -1L, long.MinValue })
            {
                AssertSeedRejected(() => new Discrete(4).Seed(seed), seed);
                AssertSeedRejected(() => new Box(-1f, 1f, new Shape(2), np.float32).Seed(seed), seed);
                AssertSeedRejected(() => new CartPoleEnv().Reset(seed: seed), seed);
                AssertSeedRejected(() => new LunarLanderEnv(NullEnvViewer.Factory, random_state: Seeding.NpRandom(0).Rng).Reset(seed: seed), seed);
            }
        }

        /// <summary>
        ///     A constructor seed is Gymnasium's: <see langword="null"/> (Python's <c>None</c>) means no seed, and every negative
        ///     seed, -1 included, is rejected with Gymnasium's message. Gym.NET's earlier spaces read -1 as "no seed"; P2's
        ///     <see cref="SpaceSeed"/> parameter has <c>None</c> for that.
        /// </summary>
        [TestMethod]
        public void ConstructorSeed_NoneMeansUnseeded_EveryNegativeThrows()
        {
            using (var unseeded = new Discrete(4, seed: null))
                Assert.IsNull(unseeded.NpRandomOrNull, "No seed creates no generator until the first draw.");
            using (new Box(-1f, 1f, new Shape(2), np.float32, seed: null)) { }
            AssertSeedRejected(() => new Discrete(4, seed: -1), -1);
            AssertSeedRejected(() => new Discrete(4, seed: -2), -2);
            AssertSeedRejected(() => new Box(-1f, 1f, new Shape(2), np.float32, seed: -1), -1);
        }

        /// <summary>
        ///     A seeded reset gives LunarLander a new generator, as Gymnasium's <c>super().reset(seed=seed)</c> does: an action
        ///     space created with the constructor's <c>random_state</c> keeps drawing from that generator, whose next sample is
        ///     what it would have been without the reset. (Before the core-API work, <c>Seed(s)</c> reseeded the shared
        ///     generator in place, which Gymnasium never does.)
        /// </summary>
        [TestMethod]
        public void LunarLanderSeededReset_ReplacesOnlyTheEnvsGenerator()
        {
            var shared = Seeding.NpRandom(1).Rng;
            var env = new LunarLanderEnv(NullEnvViewer.Factory, random_state: shared);
            // The constructor draws the wind and turbulence phases from the shared generator (B-18); the twin makes the same draws.
            var twinGenerator = Seeding.NpRandom(1).Rng;
            twinGenerator.integers(-9999, 9999).Dispose();
            twinGenerator.integers(-9999, 9999).Dispose();
            var twin = new Discrete(4, seed: twinGenerator);

            env.Reset(seed: 7).Observation.Dispose();

            Assert.AreEqual(new BigInteger(7), env.NpRandomSeed);
            Assert.AreNotSame(shared, env.NpRandom);
            using NDArray fromShared = env.ActionSpace.Sample();
            using NDArray fromTwin = twin.Sample();
            Assert.AreEqual((long) fromTwin, (long) fromShared);
        }

        /// <summary>Two CartPoles reset with <paramref name="seed"/> start from the same state.</summary>
        /// <param name="seed">A non-negative seed of any size.</param>
        private static void AssertSameReset(BigInteger seed)
        {
            using NDArray first = new CartPoleEnv().Reset(seed: seed).Observation;
            using NDArray second = new CartPoleEnv().Reset(seed: seed).Observation;
            Assert.IsTrue(np.array_equal(first, second), $"Seed {seed}: the same seed must give the same initial state.");
        }

        /// <summary>Asserts that <paramref name="seeding"/> throws Gymnasium's <see cref="Error"/> with its negative-seed message.</summary>
        /// <param name="seeding">The call that seeds with <paramref name="seed"/>.</param>
        /// <param name="seed">The rejected seed.</param>
        private static void AssertSeedRejected(Action seeding, long seed)
        {
            var error = Assert.ThrowsExactly<Error>(seeding, $"Seed {seed} must be rejected.");
            Assert.AreEqual(NegativeSeedMessage + seed.ToString(CultureInfo.InvariantCulture), error.Message, $"Seed {seed}");
        }
    }
}
