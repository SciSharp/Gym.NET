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
    ///     Pins Gym.NET's port of Gymnasium's seeding (<c>gymnasium/utils/seeding.py</c>): any non-negative seed is valid, a
    ///     negative one raises Gymnasium's <see cref="Error"/> with its message, and the reported seed is the one given, or
    ///     128-bit entropy for an unseeded generator. The legacy <c>RandomState</c> rules (a 2^32 - 1 ceiling, NumPy's
    ///     <c>ValueError</c>) no longer apply.
    /// </summary>
    [TestClass]
    public class SeedTests {
        /// <summary>Gymnasium's message for a negative seed, before the value.</summary>
        private const string NegativeSeedMessage = "Seed must be greater or equal to zero, actual value: ";

        /// <summary>Seeds that NumPy's legacy RandomState rejected or could not take from an <see cref="int"/>, all valid for Gymnasium.</summary>
        private static readonly long[] LargeSeeds = { (1L << 31) + 5, (1L << 32) - 1, 1L << 32, 1L << 40, long.MaxValue };

        /// <summary>
        ///     Port of Gymnasium's <c>test_valid_seeds</c> (<c>tests/utils/test_seeding.py</c>), extended with large seeds:
        ///     <c>np_random(seed)</c> reports the seed it was given, including one beyond <see cref="long"/>.
        /// </summary>
        [TestMethod]
        public void NpRandom_ValidSeeds_ReportTheSeed()
        {
            foreach (long seed in new[] { 0L, 1L }.Concat(LargeSeeds))
                Assert.AreEqual(new BigInteger(seed), Seeding.NpRandom(seed).Seed, $"Seed {seed}");

            var beyondLong = BigInteger.Pow(2, 100) + 7;
            Assert.AreEqual(beyondLong, Seeding.NpRandom(beyondLong).Seed);
        }

        /// <summary>
        ///     Port of Gymnasium's <c>test_invalid_seeds</c> for -1 (a string cannot reach a typed seed): a negative seed
        ///     raises <see cref="Error"/> with Gymnasium's message.
        /// </summary>
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
        ///     every call; passing it back rebuilds the same stream.
        /// </summary>
        [TestMethod]
        public void NpRandom_Unseeded_ReportsItsEntropyAndRebuildsFromIt()
        {
            var (rng, seed) = Seeding.NpRandom();

            Assert.IsTrue(seed.Sign >= 0 && seed < BigInteger.One << 128, $"{seed} is not a 128-bit value.");
            Assert.AreNotEqual(seed, Seeding.NpRandom().Seed, "Two unseeded generators must draw different entropy.");
            if (seed < BigInteger.One << 96)
                Assert.Inconclusive("The entropy's top 32-bit word is zero (1 in 2^32): NumPy would rebuild from three words, not four.");
            var (rebuilt, _) = Seeding.NpRandom(seed);
            using NDArray original = rng.random(8);
            using NDArray again = rebuilt.random(8);
            Assert.IsTrue(np.array_equal(original, again), "The reported seed must rebuild the unseeded generator's stream.");
        }

        /// <summary>
        ///     Seeds beyond the former 2^32 - 1 ceiling are accepted by every space and env, and the same seed replays the
        ///     same stream (the Gymnasium golden pins the values themselves for seeds up to 2^40).
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
                new CartPoleEnv(NullEnvViewer.Factory).Seed(seed);
                new LunarLanderEnv(NullEnvViewer.Factory, random_state: Seeding.NpRandom(0).Rng).Seed(seed);
            }
        }

        /// <summary>Every space and env rejects a negative seed with Gymnasium's error and message.</summary>
        [TestMethod]
        public void Seed_Negative_ThrowsGymnasiumErrorEverywhere()
        {
            foreach (long seed in new[] { -1L, long.MinValue })
            {
                AssertSeedRejected(() => new Discrete(4).Seed(seed), seed);
                AssertSeedRejected(() => new Box(-1f, 1f, new Shape(2), np.float32).Seed(seed), seed);
                AssertSeedRejected(() => new CartPoleEnv(NullEnvViewer.Factory).Seed(seed), seed);
                AssertSeedRejected(() => new LunarLanderEnv(NullEnvViewer.Factory, random_state: Seeding.NpRandom(0).Rng).Seed(seed), seed);
            }
        }

        /// <summary>A constructor seed of -1 still means "no seed"; any other negative constructor seed is rejected.</summary>
        [TestMethod]
        public void ConstructorSeed_MinusOneMeansUnseeded_OtherNegativesThrow()
        {
            new Discrete(4, seed: -1);
            new Box(-1f, 1f, new Shape(2), np.float32, seed: -1);
            AssertSeedRejected(() => new Discrete(4, seed: -2), -2);
            AssertSeedRejected(() => new Box(-1f, 1f, new Shape(2), np.float32, seed: -2), -2);
        }

        /// <summary>
        ///     LunarLander reseeds its generator in place, so an action space created with the same generator follows: after
        ///     <c>Seed(7)</c> its next sample is what a <c>Discrete(4)</c> seeded with 7 draws first.
        /// </summary>
        [TestMethod]
        public void LunarLanderSeed_ReseedsTheSharedGeneratorInPlace()
        {
            var env = new LunarLanderEnv(NullEnvViewer.Factory, random_state: Seeding.NpRandom(1).Rng);
            var expected = new Discrete(4, seed: 7);

            env.Seed(7);

            using NDArray shared = env.ActionSpace.Sample();
            using NDArray fresh = expected.Sample();
            Assert.AreEqual((long) fresh, (long) shared);
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
