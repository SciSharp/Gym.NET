#nullable enable
using System.Numerics;
using Gym.Exceptions;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests.Ported.Utils {
    /// <summary>Gymnasium's seeding tests, run against <see cref="Seeding"/>.</summary>
    /// <remarks>
    ///     <c>test_invalid_seeds</c>' string seed (<c>"test"</c>) cannot reach a typed seed, and <c>test_rng_pickle</c> has
    ///     no .NET counterpart (there is no pickle); the generator's state round trip is compared with NumPy's by the live
    ///     parity suite (<c>RngFacts</c>). <c>SeedTests</c> extends these with large seeds. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/tests/utils/test_seeding.py">tests/utils/test_seeding.py</see>.
    /// </remarks>
    [TestClass]
    public class TestSeeding {
        /// <summary>A negative seed (-1) raises <see cref="Error"/>.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/tests/utils/test_seeding.py#L7">tests/utils/test_seeding.py (test_invalid_seeds)</see>.</remarks>
        [TestMethod]
        public void TestInvalidSeeds() {
            foreach (int seed in new[] { -1 })
                Assert.ThrowsExactly<Error>(() => Seeding.NpRandom(seed), $"Invalid seed {seed} passed validation");
        }

        /// <summary>The reported seed is the one given.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/tests/utils/test_seeding.py#L17">tests/utils/test_seeding.py (test_valid_seeds)</see>.</remarks>
        [TestMethod]
        public void TestValidSeeds() {
            foreach (int seed in new[] { 0, 1 }) {
                var (_, seed1) = Seeding.NpRandom(seed);
                Assert.AreEqual(new BigInteger(seed), seed1);
            }
        }
    }
}
