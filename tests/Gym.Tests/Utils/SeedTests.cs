using System;
using Gym.Environments;
using Gym.Environments.Envs.Aether;
using Gym.Environments.Envs.Classic;
using Gym.Spaces;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Utils {
    /// <summary>
    ///     Pins the <see langword="long"/> seed path: every seed NumPy accepts (<c>[0, 2**32 - 1]</c>) works through
    ///     <see cref="LegacySeeding"/>, spaces and envs, and everything outside it raises NumPy's <c>ValueError</c>.
    ///     <c>np.random.RandomState(int)</c> could take neither half of that contract: it has no overload for seeds above
    ///     <see cref="int.MaxValue"/> and silently maps -1 to 4294967295.
    /// </summary>
    [TestClass]
    public class SeedTests {
        /// <summary>NumPy's own message for a seed outside <c>[0, 2**32 - 1]</c>.</summary>
        private const string SeedRangeMessage = "Seed must be between 0 and 2**32 - 1";

        /// <summary>2^31 + 5: a valid NumPy seed that does not fit in an <see cref="int"/>.</summary>
        private const long AboveInt32 = (1L << 31) + 5;

        /// <summary>2^32 - 1: the largest seed NumPy accepts.</summary>
        private const long MaxSeed = uint.MaxValue;

        /// <summary>Seeds above <see cref="int.MaxValue"/> are accepted everywhere a seed is taken.</summary>
        [TestMethod]
        public void Seed_AboveInt32_IsAccepted()
        {
            foreach (long seed in new[] { AboveInt32, MaxSeed })
            {
                LegacySeeding.RandomState(seed);
                new Discrete(4, seed: seed).Seed(seed);
                new Box(-1f, 1f, new Shape(2), np.float32, seed: seed).Seed(seed);
                new CartPoleEnv(NullEnvViewer.Factory).Seed(seed);
                new LunarLanderEnv(NullEnvViewer.Factory, random_state: LegacySeeding.RandomState(0)).Seed(seed);
            }
        }

        /// <summary>
        ///     The high bits of a large seed matter: 2^31 + 5 and 5 are different seeds, so their streams must differ, while
        ///     the same large seed replays the same stream.
        /// </summary>
        [TestMethod]
        public void Seed_AboveInt32_ProducesItsOwnReproducibleStream()
        {
            var large = LegacySeeding.RandomState(AboveInt32);
            var largeAgain = LegacySeeding.RandomState(AboveInt32);
            var small = LegacySeeding.RandomState(5);

            using var drawn = large.randint(0, 1_000_000, new Shape(8), np.int64);
            using var replayed = largeAgain.randint(0, 1_000_000, new Shape(8), np.int64);
            using var other = small.randint(0, 1_000_000, new Shape(8), np.int64);

            Assert.IsTrue(np.array_equal(drawn, replayed), "The same seed must replay the same stream.");
            Assert.IsFalse(np.array_equal(drawn, other), "Seeds 2^31 + 5 and 5 must not share a stream.");
        }

        /// <summary>
        ///     -1, 2^32 and <see cref="long.MinValue"/> raise <see cref="ValueError"/> with NumPy's message from the factory,
        ///     from spaces and from envs, instead of wrapping into some other valid seed.
        /// </summary>
        [TestMethod]
        public void Seed_OutsideNumPyRange_ThrowsValueError()
        {
            foreach (long seed in new[] { -1L, 1L << 32, long.MinValue })
            {
                AssertSeedRejected(() => LegacySeeding.RandomState(seed), seed);
                AssertSeedRejected(() => new Discrete(4).Seed(seed), seed);
                AssertSeedRejected(() => new Box(-1f, 1f, new Shape(2), np.float32).Seed(seed), seed);
                AssertSeedRejected(() => new CartPoleEnv(NullEnvViewer.Factory).Seed(seed), seed);
                AssertSeedRejected(() => new LunarLanderEnv(NullEnvViewer.Factory, random_state: LegacySeeding.RandomState(0)).Seed(seed), seed);
            }
        }

        /// <summary>
        ///     A constructor seed of -1 still means "no seed", while any other negative constructor seed is rejected; before
        ///     this migration, -2 silently became seed 4294967294.
        /// </summary>
        [TestMethod]
        public void ConstructorSeed_MinusOneMeansUnseeded_OtherNegativesThrow()
        {
            new Discrete(4, seed: -1);
            new Box(-1f, 1f, new Shape(2), np.float32, seed: -1);
            AssertSeedRejected(() => new Discrete(4, seed: -2), -2);
            AssertSeedRejected(() => new Box(-1f, 1f, new Shape(2), np.float32, seed: -2), -2);
        }

        /// <summary>Asserts that <paramref name="seeding"/> throws <see cref="ValueError"/> with NumPy's range message.</summary>
        /// <param name="seeding">The call that seeds with <paramref name="seed"/>.</param>
        /// <param name="seed">The rejected seed, for the failure message.</param>
        private static void AssertSeedRejected(Action seeding, long seed)
        {
            var error = Assert.ThrowsExactly<ValueError>(seeding, $"Seed {seed} must be rejected.");
            Assert.AreEqual(SeedRangeMessage, error.Message, $"Seed {seed}");
        }
    }
}
