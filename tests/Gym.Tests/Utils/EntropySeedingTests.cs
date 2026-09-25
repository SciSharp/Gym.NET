using System.Collections.Generic;
using System.Linq;
using Gym.Environments;
using Gym.Environments.Envs.Classic;
using Gym.Spaces;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Utils {
    /// <summary>
    ///     Pins the unseeded fallback (<see cref="LegacySeeding.FromEntropy"/>): every env and space created without a seed
    ///     or generator draws from a generator of its own, whose stream differs from every other instance's even when they are
    ///     created back to back. The two defects it replaces: the process-wide <c>np.random</c>, which every unseeded space
    ///     shared (trap 5, B-29), and <c>np.random.RandomState()</c>, which NumSharp seeds from
    ///     <see cref="System.Environment.TickCount"/>, so instances created in the same millisecond repeated one stream
    ///     (trap 6).
    /// </summary>
    [TestClass]
    public class EntropySeedingTests {
        /// <summary>A <see cref="Discrete"/> that exposes the generator it draws from.</summary>
        private sealed class ProbeDiscrete : Discrete {
            /// <summary>An unseeded space of <paramref name="n"/> elements.</summary>
            /// <param name="n">The number of elements.</param>
            public ProbeDiscrete(long n) : base(n) { }

            /// <summary>The space's generator.</summary>
            public NumPyRandom Generator => RandomState;
        }

        /// <summary>A <see cref="Box"/> that exposes the generator it draws from.</summary>
        private sealed class ProbeBox : Box {
            /// <summary>An unseeded box of shape (3,) in [-1, 1].</summary>
            public ProbeBox() : base(-1f, 1f, new Shape(3)) { }

            /// <summary>The box's generator.</summary>
            public NumPyRandom Generator => RandomState;
        }

        /// <summary>
        ///     Generators made back to back, well within one millisecond, all produce different streams. With TickCount
        ///     seeding all 64 would be identical.
        /// </summary>
        [TestMethod]
        public void FromEntropy_BackToBack_ProducesIndependentStreams()
        {
            var generators = Enumerable.Range(0, 64).Select(_ => LegacySeeding.FromEntropy()).ToList();

            var streams = generators.Select(FirstDraws).ToList();

            Assert.AreEqual(streams.Count, streams.Distinct().Count(), "Every generator must produce its own stream.");
        }

        /// <summary>
        ///     Unseeded spaces never share a generator, with each other or with the process-wide <c>np.random</c>, so drawing
        ///     from one cannot shift another and two threads never draw from one unsynchronized MT19937.
        /// </summary>
        [TestMethod]
        public void UnseededSpaces_EachOwnTheirGenerator()
        {
            var first = new ProbeDiscrete(4);
            var second = new ProbeDiscrete(4);
            var box = new ProbeBox();

            Assert.AreNotSame(first.Generator, second.Generator);
            Assert.AreNotSame(first.Generator, box.Generator);
            foreach (var generator in new[] { first.Generator, second.Generator, box.Generator })
                Assert.AreNotSame(np.random, generator, "No space may fall back to the process-wide np.random.");
        }

        /// <summary>Two unseeded spaces created back to back draw different streams.</summary>
        [TestMethod]
        public void UnseededSpaces_CreatedBackToBack_DrawDifferentStreams()
        {
            var first = new Discrete(1_000_000);
            var second = new Discrete(1_000_000);

            CollectionAssert.AreNotEqual(Samples(first, 16), Samples(second, 16));
        }

        /// <summary>
        ///     Behavior change: seeding the process-wide <c>np.random</c> no longer makes unseeded spaces reproducible, as in
        ///     Gymnasium; <see cref="Space.Seed"/> does.
        /// </summary>
        [TestMethod]
        public void GlobalSeed_DoesNotMakeSpacesReproducible_SpaceSeedDoes()
        {
            np.random.seed(123);
            var first = new Discrete(1_000_000);
            var firstDraws = Samples(first, 16);
            np.random.seed(123);
            var second = new Discrete(1_000_000);
            var secondDraws = Samples(second, 16);
            CollectionAssert.AreNotEqual(firstDraws, secondDraws, "np.random.seed must not reach an unseeded space.");

            first.Seed(123);
            second.Seed(123);
            CollectionAssert.AreEqual(Samples(first, 16), Samples(second, 16), "Space.Seed must make spaces reproducible.");
        }

        /// <summary>
        ///     Two unseeded envs created back to back start from different initial states (trap 6: CartPole's former
        ///     <c>np.random.RandomState()</c> fallback gave both the same stream).
        /// </summary>
        [TestMethod]
        public void UnseededEnvs_CreatedBackToBack_ResetDifferently()
        {
            var first = new CartPoleEnv(NullEnvViewer.Factory);
            var second = new CartPoleEnv(NullEnvViewer.Factory);

            using NDArray firstState = first.Reset();
            using NDArray secondState = second.Reset();

            Assert.IsFalse(np.array_equal(firstState, secondState), "Unseeded envs must not share an initial state.");
        }

        /// <summary>Draws 8 values in [0, 2^31 - 1) from <paramref name="generator"/>, joined into one comparable string.</summary>
        /// <param name="generator">The generator to draw from.</param>
        /// <returns>The draws, comma-separated.</returns>
        private static string FirstDraws(NumPyRandom generator)
        {
            using NDArray draws = generator.randint(0, int.MaxValue, new Shape(8), np.int64);
            return string.Join(",", draws.ToArray<long>());
        }

        /// <summary>Samples <paramref name="space"/> <paramref name="count"/> times.</summary>
        /// <param name="space">The space to sample.</param>
        /// <param name="count">The number of samples.</param>
        /// <returns>The samples, in draw order.</returns>
        private static List<long> Samples(Discrete space, int count)
        {
            var samples = new List<long>(count);
            for (int i = 0; i < count; i++)
            {
                using NDArray sample = space.Sample();
                samples.Add((long) sample);
            }
            return samples;
        }
    }
}
