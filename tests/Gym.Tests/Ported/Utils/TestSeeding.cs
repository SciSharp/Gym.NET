#nullable enable
using System.Numerics;
using Gym.Exceptions;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests.Ported.Utils {
    /// <summary>Port of <c>tests/utils/test_seeding.py</c> @ 9e04324.</summary>
    /// <remarks>
    ///     <c>test_invalid_seeds</c>' string seed (<c>"test"</c>) cannot reach a typed seed, and <c>test_rng_pickle</c> has
    ///     no .NET counterpart (there is no pickle); the generator's state round trip is compared with NumPy's by the live
    ///     parity suite (<c>RngFacts</c>). <c>SeedTests</c> extends these with large seeds.
    /// </remarks>
    [TestClass]
    public class TestSeeding {
        /// <summary>Port of <c>test_invalid_seeds</c> for -1: a negative seed raises <see cref="Error"/>.</summary>
        [TestMethod]
        public void TestInvalidSeeds() {
            foreach (int seed in new[] { -1 })
                Assert.ThrowsExactly<Error>(() => Seeding.NpRandom(seed), $"Invalid seed {seed} passed validation");
        }

        /// <summary>Port of <c>test_valid_seeds</c>: the reported seed is the one given.</summary>
        [TestMethod]
        public void TestValidSeeds() {
            foreach (int seed in new[] { 0, 1 }) {
                var (_, seed1) = Seeding.NpRandom(seed);
                Assert.AreEqual(new BigInteger(seed), seed1);
            }
        }
    }
}
