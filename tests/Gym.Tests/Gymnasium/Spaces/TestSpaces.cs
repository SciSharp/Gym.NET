#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Text.Json;
using Gym.Spaces;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Tuple = Gym.Spaces.Tuple;

namespace Gym.Tests.Gymnasium.Spaces {
    /// <summary>
    ///     Gymnasium's tests of every space (<c>tests/spaces/test_spaces.py</c>), over <see cref="TestingSpaces"/>: the JSON round
    ///     trip, equality, the sample distributions (chi-square and Kolmogorov-Smirnov), masked sample distributions, seed
    ///     reproducibility, a shared generator, membership, <c>repr</c>, copying and seeding.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <c>test_space_pickling</c> has no pickle to test: its C# counterpart is <see cref="Space.DeepCopy"/>, which must
    ///     give an equal space drawing the same samples, the two properties the Python test checks of the unpickled space.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_spaces.py">tests/spaces/test_spaces.py</see>.</para>
    /// </remarks>
    [TestClass]
    public class TestSpaces {
        /// <summary>The number of draws <c>test_sample</c> takes.</summary>
        private const int SampleTrials = 1_000;

        /// <summary>The number of draws <c>test_space_sample_mask</c> takes.</summary>
        private const int MaskTrials = 100;

        /// <summary>Every space of the catalog, by position.</summary>
        public static IEnumerable<object[]> Spaces => TestingSpaces.AllIndexes;

        /// <summary>Every fundamental space, by position.</summary>
        public static IEnumerable<object[]> FundamentalSpaces => TestingSpaces.FundamentalIndexes;

        /// <summary>Two samples survive <c>to_jsonable</c>, a JSON dump and load, and <c>from_jsonable</c>.</summary>
        /// <param name="index">The space's position in <see cref="TestingSpaces.All"/>.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_spaces.py#L77">test_roundtripping</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Spaces))]
        public void TestRoundtripping(int index) {
            using Space space = TestingSpaces.All[index]();
            object? sample1 = space.Sample();
            object? sample2 = space.Sample();

            object sampleJson = space.ToJsonable(new[] { sample1, sample2 });
            object roundtripped = JsonRoundTrip(sampleJson);
            IList<object?> primes = space.FromJsonable(roundtripped);

            Assert.IsTrue(EnvChecker.DataEquivalence(sample1, primes[0]), $"{space}: sample 1: {PyBuiltins.Repr(sample1)}, prime: {PyBuiltins.Repr(primes[0])}");
            Assert.IsTrue(EnvChecker.DataEquivalence(sample2, primes[1]), $"{space}: sample 2: {PyBuiltins.Repr(sample2)}, prime: {PyBuiltins.Repr(primes[1])}");
        }

        /// <summary>Every pair of distinct spaces of one type: each equals itself and not the other.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_spaces.py#L97">test_space_equality</see>.</remarks>
        [TestMethod]
        public void TestSpaceEquality() {
            Space[] spaces = TestingSpaces.All.Select(create => create()).ToArray();
            try {
                // itertools.groupby: runs of consecutive spaces of one type, then every ordered pair within a run.
                for (int start = 0; start < spaces.Length;) {
                    int end = start;
                    while (end < spaces.Length && spaces[end].GetType() == spaces[start].GetType())
                        end++;
                    for (int i = start; i < end; i++) {
                        for (int j = start; j < end; j++) {
                            if (i == j)
                                continue;
                            Assert.IsTrue(spaces[i].Equals(spaces[i]), $"{spaces[i]} == itself");
                            Assert.IsFalse(spaces[i].Equals(spaces[j]), $"{spaces[i]} != {spaces[j]}");
                        }
                    }

                    start = end;
                }
            } finally {
                foreach (Space space in spaces)
                    space.Dispose();
            }
        }

        /// <summary>
        ///     1000 samples of each fundamental space follow its distribution: uniform per category (chi-square), uniform,
        ///     exponential or normal per float box entry by its bounds (Kolmogorov-Smirnov), the matching discrete
        ///     distributions per integer box entry, and fair coins for booleans.
        /// </summary>
        /// <param name="index">The space's position in <see cref="TestingSpaces.Fundamental"/>.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_spaces.py#L117">test_sample</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(FundamentalSpaces))]
        public void TestSample(int index) {
            using Space space = TestingSpaces.Fundamental[index]();
            space.Seed(0);
            object?[] samples = Enumerable.Range(0, SampleTrials).Select(_ => space.Sample()).ToArray();
            try {
                Assert.AreEqual(SampleTrials, samples.Length);
                switch (space) {
                    case Box box:
                        TestBoxSample(box, samples.Cast<NDArray>().ToArray());
                        break;
                    case Discrete discrete: {
                        var observed = new double[discrete.N];
                        foreach (NDArray sample in samples.Cast<NDArray>())
                            observed[ToLong(sample) - discrete.Start]++;
                        double expected = (double)SampleTrials / discrete.N;
                        Assert.AreEqual(SampleTrials, observed.Sum());
                        double variance = observed.Sum(o => (expected - o) * (expected - o) / expected);
                        Assert.IsTrue(variance < SpaceStatistics.Chi2Isf(discrete.N - 1), $"{space}: variance {variance}");
                        break;
                    }
                    case MultiBinary binary: {
                        long size = binary.Shape.size;
                        var observed = new double[size];
                        foreach (NDArray sample in samples.Cast<NDArray>()) {
                            for (long i = 0; i < size; i++)
                                observed[i] += ToLong(sample.GetAtIndex(i));
                        }

                        double expected = SampleTrials / 2.0;
                        foreach (double o in observed) {
                            // A binary space: the variance is symmetric for the 0 and 1 categories.
                            double variance = 2 * (o - expected) * (o - expected) / expected;
                            Assert.IsTrue(variance < SpaceStatistics.Chi2Isf(1), $"{space}: variance {variance}");
                        }

                        break;
                    }
                    case MultiDiscrete multi: {
                        long[] nvec = Longs(multi.Nvec);
                        long[] start = Longs(multi.Start);
                        for (int entry = 0; entry < nvec.Length; entry++) {
                            var observed = new double[nvec[entry]];
                            foreach (NDArray sample in samples.Cast<NDArray>())
                                observed[ToLong(sample.GetAtIndex(entry)) - start[entry]]++;
                            double expected = (double)SampleTrials / nvec[entry];
                            Assert.AreEqual(SampleTrials, observed.Sum());
                            double variance = observed.Sum(o => (expected - o) * (expected - o) / expected);
                            Assert.IsTrue(variance < SpaceStatistics.Chi2Isf(nvec[entry] - 1), $"{space}: entry {entry} variance {variance}");
                        }

                        break;
                    }
                    case Text text: {
                        int characters = text.CharacterSet.Count;
                        double expected = SampleTrials * (text.MinLength + (text.MaxLength - text.MinLength) / 2.0) / characters;
                        var observed = new double[characters];
                        long total = 0;
                        foreach (string sample in samples.Cast<string>()) {
                            foreach (string character in CodePoints(sample)) {
                                observed[text.CharacterIndex(character)]++;
                                total++;
                            }
                        }

                        Assert.AreEqual(total, (long)observed.Sum());
                        double variance = observed.Sum(o => (expected - o) * (expected - o) / expected);
                        Assert.IsTrue(variance < SpaceStatistics.Chi2Isf(characters - 1), $"{space}: variance {variance}");
                        break;
                    }
                    default:
                        Assert.Fail($"Unknown sample testing for {space.GetType().Name}");
                        break;
                }
            } finally {
                foreach (object? sample in samples)
                    (sample as IDisposable)?.Dispose();
            }
        }

        /// <summary>
        ///     100 masked samples of each fundamental space follow the masked distribution: only allowed categories (or the
        ///     start when none is), uniformly; a box takes no mask.
        /// </summary>
        /// <param name="index">The space's position in <see cref="TestingSpaces.Fundamental"/>.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_spaces.py#L344">test_space_sample_mask</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(FundamentalSpaces))]
        public void TestSpaceSampleMask(int index) {
            using Space space = TestingSpaces.Fundamental[index]();
            object?[] masks = TestingSpaces.FundamentalSampleMasks();
            object? mask = masks[index];
            try {
                if (space is Box) {
                    Assert.IsNull(mask, "The box space can't have a sample mask.");
                    return;
                }

                Assert.IsNotNull(mask);
                space.Seed(1);
                object?[] samples = Enumerable.Range(0, MaskTrials).Select(_ => space.Sample(mask)).ToArray();
                try {
                    switch (space) {
                        case Discrete discrete:
                            CheckMaskedCategories(Longs((NDArray)mask!), samples.Select(s => ToLong(s) - discrete.Start).ToArray(), $"{space}");
                            break;
                        case MultiBinary binary: {
                            long size = binary.Shape.size;
                            long[] maskValues = Longs((NDArray)mask!);
                            for (long i = 0; i < size; i++) {
                                double expected = (maskValues[i] == 2 ? 0.5 : maskValues[i]) * MaskTrials;
                                double observed = samples.Sum(s => (double)ToLong(((NDArray)s!).GetAtIndex(i)));
                                double variance = 2 * (observed - expected) * (observed - expected) / Math.Max(expected, 1);
                                Assert.IsTrue(variance < SpaceStatistics.Chi2Isf(1), $"{space}: entry {i} variance {variance}");
                            }

                            break;
                        }
                        case MultiDiscrete multi: {
                            long[] start = Longs(multi.Start);
                            NDArray[] entryMasks = FlattenMaskTuple(mask!).ToArray();
                            for (int entry = 0; entry < entryMasks.Length; entry++) {
                                long[] values = samples.Select(s => ToLong(((NDArray)s!).GetAtIndex(entry)) - start[entry]).ToArray();
                                CheckMaskedCategories(Longs(entryMasks[entry]), values, $"{space} entry {entry}");
                            }

                            break;
                        }
                        case Text text: {
                            var pair = (object?[])mask!;
                            long[] charMask = Longs((NDArray)pair[1]!);
                            int characters = text.CharacterSet.Count;
                            double expectedLength = pair[0] is null ? text.MinLength + (text.MaxLength - text.MinLength) / 2.0 : Convert.ToDouble(pair[0], CultureInfo.InvariantCulture);
                            long allowed = charMask.Count(m => m == 1);
                            var expected = new double[characters];
                            if (allowed > 0) {
                                for (int c = 0; c < characters; c++)
                                    expected[c] = (double)MaskTrials * expectedLength / allowed * charMask[c];
                            }

                            var observed = new double[characters];
                            foreach (string sample in samples.Cast<string>()) {
                                foreach (string character in CodePoints(sample))
                                    observed[text.CharacterIndex(character)]++;
                            }

                            long degrees = Math.Max(allowed - 1, 0);
                            double variance = observed.Zip(expected).Sum(p => (p.Second - p.First) * (p.Second - p.First) / Math.Max(p.Second, 1));
                            if (degrees == 0)
                                Assert.AreEqual(0.0, variance, $"{space}");
                            else
                                Assert.IsTrue(variance < SpaceStatistics.Chi2Isf(degrees), $"{space}: variance {variance}");
                            break;
                        }
                        default:
                            Assert.Fail($"Unknown mask testing for {space.GetType().Name}");
                            break;
                    }
                } finally {
                    foreach (object? sample in samples)
                        (sample as IDisposable)?.Dispose();
                }
            } finally {
                foreach (object? m in masks)
                    DisposeAll(m);
            }
        }

        /// <summary>Two copies seeded alike sample alike, and seeded apart they do not, for five seeds.</summary>
        /// <param name="index">The space's position in <see cref="TestingSpaces.All"/>.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_spaces.py#L507">test_seed_reproducibility</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Spaces))]
        public void TestSeedReproducibility(int index) {
            using Space space1 = TestingSpaces.All[index]();
            using Space space2 = space1.DeepCopy();
            for (long seed = 0; seed < 5; seed++) {
                Assert.IsTrue(EnvChecker.DataEquivalence(space1.Seed(seed), space2.Seed(seed)), $"{space1}: seed({seed})");
                for (int i = 0; i < 10; i++)
                    Assert.IsTrue(SamplesEquivalent(space1, space2), $"{space1}: sample {i} after seed({seed})");
            }

            Assert.IsFalse(EnvChecker.DataEquivalence(space1.Seed(123), space2.Seed(456)), $"{space1}: seeds 123 and 456");
            // Due to randomness, it is difficult to test that random seeds produce different answers: not all of 10 match.
            Assert.IsFalse(Enumerable.Range(0, 10).All(_ => SamplesEquivalent(space1, space2)), $"{space1}: differently seeded samples all equal");
        }

        /// <summary>A generator given at construction is the space's <c>np_random</c>, for every space class.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_spaces.py#L544">test_seed_np_random</see>.</remarks>
        [TestMethod]
        public void TestSeedNpRandom() {
            var classes = new Func<SpaceSeed?, Space>[] {
                seed => new Discrete(3, seed),
                seed => new Box(1, 10, seed: seed),
                seed => new MultiDiscrete(new long[] { 3, 2 }, seed: seed),
                seed => new MultiBinary(2, seed),
                seed => new Text(5, seed: seed),
                seed => new Tuple(new Space[] { new Discrete(3), new Discrete(2) }, seed),
                seed => new Dict(new Dictionary<string, Space> { ["a"] = new Discrete(3), ["b"] = new Discrete(2) }, seed),
                seed => new Graph(new Discrete(4), new Discrete(3), seed),
                seed => new Sequence(new Discrete(4), seed),
                seed => new OneOf(new Space[] { new Discrete(3), new Discrete(5) }, seed),
            };
            foreach (Func<SpaceSeed?, Space> create in classes) {
                (Generator rng, _) = Seeding.NpRandom(123);
                using Space space = create(rng);
                Assert.AreSame(rng, space.NpRandom, $"{space}");
            }
        }

        /// <summary>A space contains its own samples, and every other space's membership test of its own sample answers a bool.</summary>
        /// <param name="index">The space's position in <see cref="TestingSpaces.All"/>.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_spaces.py#L557">test_sample_contains</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Spaces))]
        public void TestSampleContains(int index) {
            using Space space = TestingSpaces.All[index]();
            for (int i = 0; i < 10; i++) {
                object? sample = space.Sample();
                try {
                    Assert.IsTrue(space.Contains(sample), $"{space} contains its sample {PyBuiltins.Repr(sample)}");
                } finally {
                    DisposeAll(sample);
                }
            }

            foreach (Func<Space> createOther in TestingSpaces.All) {
                using Space other = createOther();
                object? sample = other.Sample();
                try {
                    // In C# Contains is typed bool; the Python test checks that the call answers rather than raising.
                    _ = other.Contains(sample);
                } finally {
                    DisposeAll(sample);
                }
            }
        }

        /// <summary><c>str(space)</c> is a string.</summary>
        /// <param name="index">The space's position in <see cref="TestingSpaces.All"/>.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_spaces.py#L578">test_repr</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Spaces))]
        public void TestRepr(int index) {
            using Space space = TestingSpaces.All[index]();
            Assert.IsFalse(string.IsNullOrEmpty(space.ToString()));
        }

        /// <summary>A deep copy (Gym.NET's counterpart of an unpickled space) equals the space and draws the same next sample.</summary>
        /// <param name="index">The space's position in <see cref="TestingSpaces.All"/>.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_spaces.py#L583">test_space_pickling</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Spaces))]
        public void TestSpacePickling(int index) {
            using Space space = TestingSpaces.All[index]();
            space.Seed(0);
            using Space copy = space.DeepCopy();
            Assert.IsTrue(space.Equals(copy), $"{space} == its copy");
            Assert.IsTrue(SamplesEquivalent(space, copy), $"{space}: the copy draws the same sample");
        }

        /// <summary>Reseeding with the seeds a seeding returned reproduces the same samples, from <c>None</c> and from 123.</summary>
        /// <param name="index">The space's position in <see cref="TestingSpaces.All"/>.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_spaces.py#L611">test_space_seeding_output</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Spaces))]
        public void TestSpaceSeedingOutput(int index) {
            foreach (long? initialSeed in new long?[] { null, 123 }) {
                using Space space = TestingSpaces.All[index]();
                object seedingValues = initialSeed is null ? space.Seed((BigInteger?)null) : space.Seed(initialSeed.Value);
                object?[] samples = Enumerable.Range(0, 5).Select(_ => space.Sample()).ToArray();

                object reseededValues = space.SeedWith(seedingValues);
                object?[] resamples = Enumerable.Range(0, 5).Select(_ => space.Sample()).ToArray();
                try {
                    Assert.IsTrue(EnvChecker.DataEquivalence(seedingValues, reseededValues), $"{space}: seeds {PyBuiltins.Repr(seedingValues)} vs {PyBuiltins.Repr(reseededValues)}");
                    Assert.IsTrue(EnvChecker.DataEquivalence(samples.ToList(), resamples.ToList()), $"{space}: samples after reseeding");
                } finally {
                    DisposeAll(samples);
                    DisposeAll(resamples);
                }
            }
        }

        /// <summary><c>space.seed("abc")</c> raises a <c>ValueError</c>, a <c>TypeError</c> or Gymnasium's <c>Error</c>.</summary>
        /// <param name="index">The space's position in <see cref="TestingSpaces.All"/>.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_spaces.py#L624">test_invalid_space_seed</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(Spaces))]
        public void TestInvalidSpaceSeed(int index) {
            using Space space = TestingSpaces.All[index]();
            try {
                space.SeedWith("abc");
                Assert.Fail($"{space}.seed('abc') did not raise");
            } catch (Exception e) when (e is ValueError or TypeError or Gym.Exceptions.Error) {
                // Expected: the space's own refusal.
            }
        }

        /// <summary>
        ///     Gymnasium's box branch of <c>test_sample</c>: per entry, a Kolmogorov-Smirnov test for floats, a chi-square
        ///     test for integers and booleans, against the distribution the entry's bounds select.
        /// </summary>
        /// <param name="box">The box.</param>
        /// <param name="samples">The samples.</param>
        private static void TestBoxSample(Box box, NDArray[] samples) {
            Assert.IsTrue(box.Shape.dimensions.SequenceEqual(box.Low.Shape.dimensions) && box.Shape.dimensions.SequenceEqual(box.High.Shape.dimensions));
            long size = box.Shape.size;
            for (long entry = 0; entry < size; entry++) {
                double low = Convert.ToDouble(box.Low.GetAtIndex(entry), CultureInfo.InvariantCulture);
                double high = Convert.ToDouble(box.High.GetAtIndex(entry), CultureInfo.InvariantCulture);
                bool below = box.BoundedBelow.GetAtIndex(entry) is true;
                bool above = box.BoundedAbove.GetAtIndex(entry) is true;
                double[] values = samples.Select(s => Convert.ToDouble(s.GetAtIndex(entry), CultureInfo.InvariantCulture)).ToArray();
                switch (box.DType.kind) {
                    case 'f':
                        KsTest(values, low, high, below, above, $"{box} entry {entry}");
                        break;
                    case 'i' or 'u':
                        Chi2Test(values.Select(v => (long)v).ToArray(), (long)low, (long)high, below, above, $"{box} entry {entry}");
                        break;
                    case 'b':
                        BinaryChi2Test(values, low, high, below, above, $"{box} entry {entry}");
                        break;
                    default:
                        Assert.Fail($"Unknown test for Box(dtype={box.DType})");
                        break;
                }
            }
        }

        /// <summary>Gymnasium's <c>ks_test</c>: uniform, shifted exponential or normal by the bounds.</summary>
        /// <param name="sample">The entry's samples.</param>
        /// <param name="low">The lower bound.</param>
        /// <param name="high">The upper bound.</param>
        /// <param name="boundedBelow">Whether bounded below.</param>
        /// <param name="boundedAbove">Whether bounded above.</param>
        /// <param name="context">The entry, for messages.</param>
        private static void KsTest(double[] sample, double low, double high, bool boundedBelow, bool boundedAbove, string context) {
            double p;
            if (boundedBelow && boundedAbove)
                p = SpaceStatistics.KsTestPValue(sample, x => x < low ? 0 : x > high ? 1 : (x - low) / (high - low));
            else if (boundedBelow)
                p = SpaceStatistics.KsTestPValue(sample.Select(x => x - low).ToArray(), SpaceStatistics.ExponCdf);
            else if (boundedAbove)
                p = SpaceStatistics.KsTestPValue(sample.Select(x => high - x).ToArray(), SpaceStatistics.ExponCdf);
            else
                p = SpaceStatistics.KsTestPValue(sample, SpaceStatistics.NormalCdf);
            Assert.IsTrue(p >= SpaceStatistics.Alpha, $"{context}: KS p-value {p}");
        }

        /// <summary>Gymnasium's <c>chi2_test</c>: uniform, geometric or floored normal frequencies by the bounds.</summary>
        /// <param name="sample">The entry's samples.</param>
        /// <param name="low">The lower bound.</param>
        /// <param name="high">The upper bound.</param>
        /// <param name="boundedBelow">Whether bounded below.</param>
        /// <param name="boundedAbove">Whether bounded above.</param>
        /// <param name="context">The entry, for messages.</param>
        private static void Chi2Test(long[] sample, long low, long high, bool boundedBelow, bool boundedAbove, string context) {
            int trials = sample.Length;
            double[] observed;
            double[] expected;
            if (boundedBelow && boundedAbove) {
                int degrees = (int)(high - low + 1);
                observed = BinCount(sample.Select(x => x - low), degrees);
                expected = Enumerable.Repeat((double)trials / degrees, degrees).ToArray();
            } else if (boundedBelow || boundedAbove) {
                // X - low (or high - X) ~ Geom(1 - e^-1), counted from 0.
                observed = BinCount(sample.Select(x => boundedBelow ? x - low : high - x), 0);
                double p = 1 - 1 / Math.E;
                expected = Enumerable.Range(0, observed.Length).Select(x => SpaceStatistics.GeomPmf(p, x + 1) * trials).ToArray();
                expected[^1] += trials - expected.Sum();
            } else {
                // X ~ floor(N(0, 1)): pmf(x) = cdf(x + 1) - cdf(x), the tails folded into the end bins.
                long lowest = sample.Min();
                observed = BinCount(sample.Select(x => x - lowest), 0);
                expected = Enumerable.Range(0, observed.Length).Select(i => SpaceStatistics.NormalCdf(lowest + i + 1) - SpaceStatistics.NormalCdf(lowest + i)).ToArray();
                expected[0] += SpaceStatistics.NormalCdf(lowest);
                for (int i = 0; i < expected.Length; i++)
                    expected[i] *= trials;
                expected[^1] += trials - expected.Sum();
            }

            Assert.AreEqual(expected.Length, observed.Length, context);
            double variance = observed.Zip(expected).Sum(p => (p.Second - p.First) * (p.Second - p.First) / p.Second);
            Assert.IsTrue(variance < SpaceStatistics.Chi2Isf(observed.Length - 1), $"{context}: variance {variance}");
        }

        /// <summary>Gymnasium's <c>binary_chi2_test</c>: constant entries stay constant, others are fair coins.</summary>
        /// <param name="sample">The entry's samples (0 or 1).</param>
        /// <param name="low">The lower bound.</param>
        /// <param name="high">The upper bound.</param>
        /// <param name="boundedBelow">Whether bounded below (always, for bool).</param>
        /// <param name="boundedAbove">Whether bounded above (always, for bool).</param>
        /// <param name="context">The entry, for messages.</param>
        private static void BinaryChi2Test(double[] sample, double low, double high, bool boundedBelow, bool boundedAbove, string context) {
            Assert.IsTrue(boundedBelow && boundedAbove, context);
            if (low == 0 && high == 0) {
                Assert.IsTrue(sample.All(x => x == 0), context);
            } else if (low == 1 && high == 1) {
                Assert.IsTrue(sample.All(x => x == 1), context);
            } else {
                double expected = sample.Length / 2.0;
                double observed = sample.Sum();
                double variance = 2 * (observed - expected) * (observed - expected) / expected;
                Assert.IsTrue(variance < SpaceStatistics.Chi2Isf(1), $"{context}: variance {variance}");
            }
        }

        /// <summary>The masked-category check of <c>test_space_sample_mask</c> for one categorical entry.</summary>
        /// <param name="mask">The entry's 0/1 mask.</param>
        /// <param name="values">The samples, as offsets from the start.</param>
        /// <param name="context">The entry, for messages.</param>
        private static void CheckMaskedCategories(long[] mask, long[] values, string context) {
            int n = mask.Length;
            long allowed = mask.Count(m => m == 1);
            var expected = new double[n];
            if (allowed > 0) {
                for (int i = 0; i < n; i++)
                    expected[i] = (double)values.Length / allowed * mask[i];
            } else {
                expected[0] = values.Length;
            }

            double[] observed = BinCount(values, n);
            Assert.AreEqual(values.Length, (int)observed.Sum(), context);
            long degrees = Math.Max(allowed - 1, 0);
            double variance = observed.Zip(expected).Sum(p => (p.Second - p.First) * (p.Second - p.First) / Math.Max(p.Second, 1));
            if (degrees == 0)
                Assert.AreEqual(0.0, variance, context);
            else
                Assert.IsTrue(variance < SpaceStatistics.Chi2Isf(degrees), $"{context}: variance {variance}");
        }

        /// <summary>The leaf masks of a (nested) tuple of masks, in C order (one per <c>MultiDiscrete</c> entry).</summary>
        /// <param name="mask">The mask tuple.</param>
        /// <returns>The leaf arrays.</returns>
        private static IEnumerable<NDArray> FlattenMaskTuple(object mask) {
            if (mask is NDArray array) {
                yield return array;
                yield break;
            }

            foreach (object? item in (object?[])mask) {
                foreach (NDArray leaf in FlattenMaskTuple(item!))
                    yield return leaf;
            }
        }

        /// <summary>NumPy's <c>np.bincount(values, minlength=minLength)</c>, as doubles.</summary>
        /// <param name="values">Non-negative integers.</param>
        /// <param name="minLength">The shortest result.</param>
        /// <returns>The counts.</returns>
        private static double[] BinCount(IEnumerable<long> values, int minLength) {
            long[] all = values.ToArray();
            int length = Math.Max(minLength, all.Length == 0 ? 0 : (int)all.Max() + 1);
            var counts = new double[length];
            foreach (long value in all)
                counts[value]++;
            return counts;
        }

        /// <summary>Whether the next samples of two spaces are equivalent; both samples are disposed.</summary>
        /// <param name="space1">The first space.</param>
        /// <param name="space2">The second space.</param>
        /// <returns><see langword="true"/> when equivalent.</returns>
        private static bool SamplesEquivalent(Space space1, Space space2) {
            object? sample1 = space1.Sample();
            object? sample2 = space2.Sample();
            try {
                return EnvChecker.DataEquivalence(sample1, sample2);
            } finally {
                DisposeAll(sample1);
                DisposeAll(sample2);
            }
        }

        /// <summary>
        ///     Python's <c>json.loads(json.dumps(data))</c>: the data written as JSON and read back as plain values (integers as
        ///     <see langword="long"/> or <see cref="BigInteger"/>, other numbers as <see langword="double"/>, lists, string-keyed
        ///     dictionaries).
        /// </summary>
        /// <param name="data">The JSON-friendly data.</param>
        /// <returns>The data read back.</returns>
        private static object JsonRoundTrip(object data) {
            var options = new JsonSerializerOptions { NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals };
            string text = JsonSerializer.Serialize(data, options);
            using JsonDocument document = JsonDocument.Parse(text);
            return FromJson(document.RootElement)!;
        }

        /// <summary>A JSON element as the plain value Python's <c>json</c> module reads.</summary>
        /// <param name="element">The element.</param>
        /// <returns>The value.</returns>
        private static object? FromJson(JsonElement element) {
            switch (element.ValueKind) {
                case JsonValueKind.Object: {
                    var dictionary = new Dictionary<string, object?>();
                    foreach (JsonProperty property in element.EnumerateObject())
                        dictionary[property.Name] = FromJson(property.Value);
                    return dictionary;
                }
                case JsonValueKind.Array:
                    return element.EnumerateArray().Select(FromJson).ToList();
                case JsonValueKind.String: {
                    // System.Text.Json writes NaN and the infinities as strings under AllowNamedFloatingPointLiterals.
                    string text = element.GetString()!;
                    return text switch {
                        "NaN" => double.NaN,
                        "Infinity" => double.PositiveInfinity,
                        "-Infinity" => double.NegativeInfinity,
                        _ => text,
                    };
                }
                case JsonValueKind.Number: {
                    string raw = element.GetRawText();
                    // Python reads a number without a fraction or exponent as an int, any other as a float.
                    if (raw.IndexOfAny(new[] { '.', 'e', 'E' }) < 0)
                        // Each arm boxed: a long arm and a BigInteger arm would type the ?: as BigInteger for every int.
                        return element.TryGetInt64(out long integer) ? (object)integer : BigInteger.Parse(raw, CultureInfo.InvariantCulture);
                    return element.GetDouble();
                }
                case JsonValueKind.True:
                    return true;
                case JsonValueKind.False:
                    return false;
                default:
                    return null;
            }
        }

        /// <summary>A string's code points, as Python iterates a <c>str</c> (a surrogate pair is one character).</summary>
        /// <param name="text">The text.</param>
        /// <returns>Each code point as a string.</returns>
        internal static IEnumerable<string> CodePoints(string text) {
            for (int i = 0; i < text.Length; i += char.IsSurrogatePair(text, i) ? 2 : 1)
                yield return char.IsSurrogatePair(text, i) ? text.Substring(i, 2) : text[i].ToString();
        }

        /// <summary>An integer sample (a 0-d array or a .NET integer) as a <see langword="long"/>.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The integer.</returns>
        private static long ToLong(object? value) => value is NDArray array ? Convert.ToInt64(array.GetAtIndex(0), CultureInfo.InvariantCulture) : Convert.ToInt64(value, CultureInfo.InvariantCulture);

        /// <summary>Every entry of an integer array as a <see langword="long"/>, in C order.</summary>
        /// <param name="array">The array.</param>
        /// <returns>The values.</returns>
        private static long[] Longs(NDArray array) => Enumerable.Range(0, (int)array.size).Select(i => ToLong(array.GetAtIndex(i))).ToArray();

        /// <summary>Disposes the arrays inside a sample, a mask or a list of them.</summary>
        /// <param name="value">The value.</param>
        private static void DisposeAll(object? value) {
            switch (value) {
                case IDisposable disposable:
                    disposable.Dispose();
                    break;
                case string:
                    break;
                case IEnumerable<KeyValuePair<string, object?>> pairs:
                    foreach (KeyValuePair<string, object?> pair in pairs)
                        DisposeAll(pair.Value);
                    break;
                case System.Collections.IEnumerable items:
                    foreach (object? item in items)
                        DisposeAll(item);
                    break;
            }
        }
    }
}
