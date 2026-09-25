using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Session;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Parity.Tests.Rng {
    /// <summary>
    ///     <c>Gym.Utils.Seeding.NpRandom</c> against <c>gymnasium.utils.seeding.np_random</c> (port of <c>seeding.py</c>
    ///     @ 9e04324), live: the reported seed, the full PCG64 state, the draw patterns Gymnasium's envs and spaces make
    ///     (port plan Appendix A.6), and the error for a negative seed.
    /// </summary>
    [TestClass]
    public sealed class SeedingLiveParityTests : ParityTestBase {
        /// <summary>Seeds of A.6 and of the F2 golden, from 0 to beyond 64 bits' worth of Python int.</summary>
        public static IEnumerable<object[]> Seeds =>
            new[] { 0L, 1L, 42L, 1000L, 1234L, int.MaxValue, 1L << 32, 1L << 40, long.MaxValue }.Select(seed => new object[] { seed });

        /// <summary>
        ///     The draw patterns Gymnasium's envs and spaces make, in the order A.6 lists them, each run on the generator the
        ///     previous ones advanced. The Python expression has the generator bound as <c>x</c>.
        /// </summary>
        /// <remarks>
        ///     NumPy's <c>choice</c> without a size returns a scalar where NumSharp's returns a one-element array (Gym.NET's
        ///     <c>Discrete</c> reads it through <c>SingleValue</c>), so those two patterns compare through
        ///     <c>np.atleast_1d</c>: same draw, same value, NumSharp's shape. <c>geometric</c> (<c>Sequence</c>) is absent:
        ///     NumSharp's <c>Generator</c> does not have it yet.
        /// </remarks>
        private static readonly (string Python, Func<Generator, NDArray> DotNet)[] Patterns = {
            // CartPole reset.
            ("x.uniform(-0.05, 0.05, size=4)", g => g.uniform(-0.05, 0.05, new Shape(4))),
            // LunarLander: terrain heights (H / 2 with H = VIEWPORT_H / SCALE), initial force, wind phases, dispersion.
            ("x.uniform(0, 400 / 30 / 2, size=12)", g => g.uniform(0, 400.0 / 30 / 2, new Shape(12))),
            ("x.uniform(-1000, 1000)", g => g.uniform(-1000, 1000)),
            ("x.integers(-9999, 9999)", g => g.integers(-9999, 9999)),
            ("x.uniform(-1.0, 1.0)", g => g.uniform(-1.0, 1.0)),
            // Discrete.sample, plain, masked and with a probability mask.
            ("x.integers(4, dtype=np.int64)", g => g.integers(4, dtype: np.int64)),
            ("np.atleast_1d(x.choice(np.array([0, 2, 3])))", g => {
                using NDArray valid = np.array(new long[] { 0, 2, 3 });
                return g.choice(valid);
            }),
            ("np.atleast_1d(x.choice(np.arange(4), p=np.array([0.1, 0.2, 0.3, 0.4])))", g => {
                using NDArray values = np.arange(4L);
                using NDArray probability = np.array(new[] { 0.1, 0.2, 0.3, 0.4 });
                return g.choice(values, p: probability);
            }),
            // Box: unbounded, half-bounded and bounded draws.
            ("x.normal(size=3)", g => g.normal(size: new Shape(3))),
            ("x.exponential(size=3)", g => g.exponential(size: new Shape(3))),
            ("x.random(5)", g => g.random(new Shape(5))),
            ("x.standard_normal(2)", g => g.standard_normal(new Shape(2))),
            // MultiBinary.sample, Dict/Tuple subseeds, FrozenLake's random map.
            ("x.integers(0, 2, size=3, dtype=np.int8)", g => g.integers(0, 2, new Shape(3), np.int8)),
            ("x.integers(np.iinfo(np.int32).max, size=3)", g => g.integers(int.MaxValue, size: new Shape(3))),
            ("x.choice(2, size=(4, 4), p=[0.8, 0.2])", g => {
                using NDArray probability = np.array(new[] { 0.8, 0.2 });
                return g.choice(2, new Shape(4, 4), p: probability);
            }),
        };

        /// <summary>
        ///     <c>NpRandom(seed)</c> reports the seed Gymnasium reports and builds the same generator: equal PCG64
        ///     <c>state</c>, <c>inc</c> and buffered half-word.
        /// </summary>
        /// <param name="seed">The seed.</param>
        [TestMethod]
        [DynamicData(nameof(Seeds))]
        public void NpRandom_ReportsTheSeedAndBuildsTheGeneratorGymnasiumDoes(long seed) {
            (Generator rng, BigInteger reported) = Seeding.NpRandom(seed);
            string call = $"seeding.np_random({seed.ToString(CultureInfo.InvariantCulture)})";
            ByteContract.AssertSame($"{call}[1]", PyGym.Invoke($"{call}[1]"), Outcome.Returned(ValueFacts.FromInteger(reported, "", "BigInteger")));
            ByteContract.AssertSame($"{call}[0].bit_generator.state", PyGym.Invoke($"{call}[0].bit_generator.state"), Outcome.Returned(RngFacts.Of(rng)));
        }

        /// <summary>
        ///     Every draw pattern of A.6, in sequence on one generator per seed, matches Gymnasium's generator value for value,
        ///     and the generator state matches after each draw, so a pattern that consumes one word more or less is caught at
        ///     that pattern.
        /// </summary>
        /// <param name="seed">The seed.</param>
        [TestMethod]
        [DynamicData(nameof(Seeds))]
        public void DrawPatterns_MatchGymnasiumValueForValue(long seed) {
            Generator dotnet = Seeding.NpRandom(seed).Rng;
            using PyHandle python = PyGym.Evaluate($"seeding.np_random({seed.ToString(CultureInfo.InvariantCulture)})[0]");
            foreach ((string expression, Func<Generator, NDArray> draw) in Patterns) {
                ByteContract.AssertSame($"seed {seed}: {expression}", python.Evaluate(expression), Outcome.Capture(() => draw(dotnet)));
                ByteContract.AssertSame($"seed {seed}: state after {expression}", python.Evaluate("x.bit_generator.state"), Outcome.Returned(RngFacts.Of(dotnet)));
            }
        }

        /// <summary>
        ///     A seed beyond 64 bits, which Gymnasium accepts as a Python int, reports back unchanged and builds the same
        ///     generator.
        /// </summary>
        [TestMethod]
        public void NpRandom_AcceptsSeedsBeyond64Bits() {
            BigInteger seed = BigInteger.Pow(2, 100) + 7;
            (Generator rng, BigInteger reported) = Seeding.NpRandom(seed);
            ByteContract.AssertSame("seeding.np_random(2**100 + 7)[1]", PyGym.Invoke("seeding.np_random(2**100 + 7)[1]"), Outcome.Returned(ValueFacts.FromInteger(reported, "", "BigInteger")));
            ByteContract.AssertSame("seeding.np_random(2**100 + 7)[0].bit_generator.state", PyGym.Invoke("seeding.np_random(2**100 + 7)[0].bit_generator.state"), Outcome.Returned(RngFacts.Of(rng)));
        }

        /// <summary>
        ///     A negative seed raises Gymnasium's <c>gymnasium.error.Error</c> with its message, and Gym.NET throws the port,
        ///     <c>Gym.Exceptions.Error</c>, with the same message.
        /// </summary>
        /// <param name="seed">A negative seed, as Python source.</param>
        [TestMethod]
        [DataRow("-1")]
        [DataRow("-9223372036854775808")]
        [DataRow("-(2**100)")]
        public void NpRandom_RejectsNegativeSeeds_WithGymnasiumsError(string seed) {
            BigInteger value = seed == "-(2**100)" ? -BigInteger.Pow(2, 100) : BigInteger.Parse(seed, CultureInfo.InvariantCulture);
            Outcome python = PyGym.Invoke($"seeding.np_random({seed})");
            Assert.IsNotNull(python.Raised, "Gymnasium must reject a negative seed.");
            ByteContract.AssertSame($"seeding.np_random({seed})", python, Outcome.Capture(() => Seeding.NpRandom(value)));
        }

        /// <summary>
        ///     An unseeded generator reports its 128-bit entropy as the seed, and Gymnasium, given that seed, builds the same
        ///     generator: the entropy path of <c>NpRandom()</c> crosses over to NumPy's <c>SeedSequence</c> intact.
        /// </summary>
        [TestMethod]
        public void NpRandom_Unseeded_ReportsASeedGymnasiumRebuildsTheGeneratorFrom() {
            (Generator rng, BigInteger reported) = Seeding.NpRandom();
            // Seeding's documented 1-in-2^32 exception: with a zero top word NumPy derives the pool from three words.
            if (reported >> 96 == BigInteger.Zero)
                Assert.Inconclusive("The drawn entropy's top 32-bit word is zero; NumPy would rebuild a different pool from it.");
            string literal = reported.ToString(CultureInfo.InvariantCulture);
            ByteContract.AssertSame($"seeding.np_random({literal})[0].bit_generator.state", PyGym.Invoke($"seeding.np_random({literal})[0].bit_generator.state"), Outcome.Returned(RngFacts.Of(rng)));
        }
    }
}
