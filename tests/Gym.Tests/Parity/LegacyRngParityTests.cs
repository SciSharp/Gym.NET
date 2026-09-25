using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Gym.Environments;
using Gym.Environments.Envs.Aether;
using Gym.Environments.Envs.Classic;
using Gym.Spaces;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Parity {
    /// <summary>
    ///     Replays the draw patterns of <c>tools/parity/legacy_rng.py</c> through Gym.NET and compares them, bit for bit, with
    ///     the committed NumPy 2.4.2 golden (<c>Parity/legacy_rng.golden.txt</c>, 602 values: 10 patterns under 7 seeds,
    ///     docs/plans/NUMSHARP_MIGRATION_PLAN.md §9.1). The seeds 2^31 - 1, 2^31 + 5 and 2^32 - 1 exercise the
    ///     <see langword="long"/> seed path.
    /// </summary>
    /// <remarks>
    ///     This is parity with NumPy's legacy <c>RandomState</c>, which Gym.NET's envs and spaces use until F2. Gymnasium's own
    ///     streams come from <c>Generator(PCG64)</c> and arrive with F2 and the P4/P8 re-ports.
    /// </remarks>
    [TestClass]
    public class LegacyRngParityTests {
        /// <summary>The seeds of <c>legacy_rng.py</c>, in its order.</summary>
        private static readonly long[] Seeds = { 0, 1, 42, 1000, (1L << 31) - 1, (1L << 31) + 5, (1L << 32) - 1 };

        /// <summary>The logical name the test project embeds the golden under (see Gym.Tests.csproj).</summary>
        private const string GoldenResource = "Gym.Tests.Parity.legacy_rng.golden.txt";

        /// <summary>
        ///     Every one of the 602 draws equals NumPy's. A failure names how many lines differ and the first difference, whose
        ///     case and index identify the draw pattern and seed.
        /// </summary>
        [TestMethod]
        public void Draws_MatchNumPyRandomStateBitForBit()
        {
            List<string> expected = ReadGolden();
            List<string> actual = Record();

            Assert.AreEqual(602, expected.Count, "The golden must hold the 602 draws of legacy_rng.py.");
            Assert.AreEqual(expected.Count, actual.Count, "Gym.NET must record as many draws as the golden holds.");
            var mismatches = Enumerable.Range(0, expected.Count).Where(i => expected[i] != actual[i]).ToList();
            if (mismatches.Count > 0)
            {
                int first = mismatches[0];
                Assert.Fail($"{mismatches.Count} of {expected.Count} draws differ from NumPy. First at line {first + 1}: expected '{expected[first]}', got '{actual[first]}'.");
            }
        }

        /// <summary>
        ///     The real <see cref="LunarLanderEnv"/> consumes exactly the pattern the golden's <c>lunar.*</c> cases record:
        ///     two <c>randint</c> in the constructor, then 2 + 12 <c>uniform</c> in <see cref="LunarLanderEnv.Reset"/> and two
        ///     per step for the zero step and 20 more. After that, its generator's state equals the state of a generator that
        ///     made those draws by hand, so the hand replay in <see cref="Record"/> is the env's own draw sequence.
        /// </summary>
        [TestMethod]
        public void LunarLander_ConsumesExactlyTheRecordedDrawPattern()
        {
            foreach (long seed in Seeds)
            {
                var envGenerator = LegacySeeding.RandomState(seed);
                var env = new LunarLanderEnv(NullEnvViewer.Factory, random_state: envGenerator);
                // The zero step inside Reset is the first of the 21 dispersion pairs; 20 explicit steps make the rest.
                env.Reset().Dispose();
                for (int i = 0; i < 20; i++)
                    env.Step((long) LunarLanderDiscreteActions.Nothing).Observation.Dispose();

                var replayed = LegacySeeding.RandomState(seed);
                ReplayLunarLander(replayed, seed, lines: null);

                var envState = envGenerator.get_state();
                var replayedState = replayed.get_state();
                Assert.AreEqual(replayedState.Pos, envState.Pos, $"Seed {seed}: the env consumed a different number of words.");
                CollectionAssert.AreEqual(replayedState.Key, envState.Key, $"Seed {seed}: the env's generator state differs.");
            }
        }

        /// <summary>Produces Gym.NET's lines in the golden's order and format.</summary>
        /// <returns>One <c>"&lt;case&gt; &lt;index&gt; &lt;value&gt;"</c> line per draw.</returns>
        private static List<string> Record()
        {
            var lines = new List<string>();
            foreach (long seed in Seeds)
            {
                ReplayLunarLander(LegacySeeding.RandomState(seed), seed, lines);

                // CartPoleEnv.Reset draws uniform(-0.05, 0.05, 4) and hands out a copy of that state.
                var cartPole = new CartPoleEnv(NullEnvViewer.Factory, LegacySeeding.RandomState(seed));
                using (NDArray reset = cartPole.Reset())
                    Emit(lines, $"cartpole.reset.s{seed}", reset.ToArray<double>().Select(F64));

                // Discrete(4) seeded like RandomState(seed): 10 plain samples, then 5 masked over [0, 2, 3].
                var discrete = new Discrete(4, seed: seed);
                Emit(lines, $"discrete.sample.s{seed}", Enumerable.Range(0, 10).Select(_ => Draw(discrete)));
                using (NDArray mask = np.array(new sbyte[] { 1, 0, 1, 1 }))
                    Emit(lines, $"discrete.masked.s{seed}", Enumerable.Range(0, 5).Select(_ => Draw(discrete, mask)));

                // Box(-1, 1, (3,), float32).Sample(): the float64 uniform computation, compared as float32 bits.
                using (NDArray bounded = new Box(-1f, 1f, new Shape(3), np.float32, seed: seed).Sample())
                    Emit(lines, $"box.bounded.s{seed}", bounded.ToArray<float>().Select(F32));

                // An unbounded and then a low-bounded (at 0) float64 box on one generator: the normal and the exponential
                // draws in the golden's order. Adding the bound 0.0 leaves an exponential draw's bits unchanged.
                var shared = LegacySeeding.RandomState(seed);
                using (NDArray normal = new Box(float.NegativeInfinity, float.PositiveInfinity, new Shape(3), np.float64, random_state: shared).Sample())
                    Emit(lines, $"box.normal.s{seed}", normal.ToArray<double>().Select(F64));
                using (NDArray exponential = new Box(0f, float.PositiveInfinity, new Shape(3), np.float64, random_state: shared).Sample())
                    Emit(lines, $"box.exponential.s{seed}", exponential.ToArray<double>().Select(F64));
            }
            return lines;
        }

        /// <summary>
        ///     Draws what <see cref="LunarLanderEnv"/> draws from its generator, in its order: two <c>randint(-9999, 9999)</c>
        ///     in the constructor; in <see cref="LunarLanderEnv.Reset"/> the force (x, y) and then the 12 terrain heights
        ///     (Gymnasium draws the terrain first, B-17); and two dispersion values per step for 21 steps.
        /// </summary>
        /// <param name="rs">The generator to draw from, seeded with <paramref name="seed"/>.</param>
        /// <param name="seed">The seed, for the case names.</param>
        /// <param name="lines">Receives the lines, or <see langword="null"/> to only advance <paramref name="rs"/>.</param>
        private static void ReplayLunarLander(NumPyRandom rs, long seed, List<string> lines)
        {
            // LunarLanderEnv computes VIEWPORT_H / SCALE in float32 (VIEWPORT_H = 400) and halves it in float32 before the
            // float -> double widening at the call.
            float h = 400 / LunarLanderEnv.SCALE;
            var wind = new[] { TakeInt64(rs.randint(-9999, 9999)), TakeInt64(rs.randint(-9999, 9999)) };
            var force = new[] { TakeDouble(rs.uniform(-LunarLanderEnv.INITIAL_RANDOM, LunarLanderEnv.INITIAL_RANDOM)), TakeDouble(rs.uniform(-LunarLanderEnv.INITIAL_RANDOM, LunarLanderEnv.INITIAL_RANDOM)) };
            var heights = Enumerable.Range(0, 12).Select(_ => TakeDouble(rs.uniform(0f, h / 2f))).ToList();
            var dispersion = Enumerable.Range(0, 42).Select(_ => TakeDouble(rs.uniform(-1f, 1f))).ToList();
            if (lines is null)
                return;
            Emit(lines, $"lunar.wind.s{seed}", wind.Select(v => v.ToString(CultureInfo.InvariantCulture)));
            Emit(lines, $"lunar.force.s{seed}", force.Select(F64));
            Emit(lines, $"lunar.height.s{seed}", heights.Select(F64));
            Emit(lines, $"lunar.disp.s{seed}", dispersion.Select(F64));
        }

        /// <summary>Samples <paramref name="space"/> once and formats the value as the golden prints integers.</summary>
        /// <param name="space">The space to sample.</param>
        /// <param name="mask">An optional sampling mask.</param>
        /// <returns>The sample in decimal.</returns>
        private static string Draw(Discrete space, NDArray mask = null)
        {
            using NDArray sample = space.Sample(mask);
            return ((long) sample).ToString(CultureInfo.InvariantCulture);
        }

        /// <summary>Converts and releases a 0-d floating-point draw.</summary>
        /// <param name="draw">A 0-d array; it is disposed.</param>
        /// <returns>The drawn float64 value.</returns>
        private static double TakeDouble(NDArray draw)
        {
            using (draw)
                return (double) draw;
        }

        /// <summary>Converts and releases a 0-d integer draw.</summary>
        /// <param name="draw">A 0-d array; it is disposed.</param>
        /// <returns>The drawn value.</returns>
        private static long TakeInt64(NDArray draw)
        {
            using (draw)
                return (long) draw;
        }

        /// <summary>Formats a float64 as the 16 hex digits of its IEEE bit pattern, like the script's <c>f64</c>.</summary>
        /// <param name="value">The value.</param>
        /// <returns>Lowercase hex, most significant digit first.</returns>
        private static string F64(double value) => BitConverter.DoubleToInt64Bits(value).ToString("x16", CultureInfo.InvariantCulture);

        /// <summary>Formats a float32 as the 8 hex digits of its IEEE bit pattern, like the script's <c>f32</c>.</summary>
        /// <param name="value">The value.</param>
        /// <returns>Lowercase hex, most significant digit first.</returns>
        private static string F32(float value) => BitConverter.SingleToInt32Bits(value).ToString("x8", CultureInfo.InvariantCulture);

        /// <summary>Appends <c>"&lt;case&gt; &lt;index&gt; &lt;value&gt;"</c> lines, like the script's <c>emit</c>.</summary>
        /// <param name="lines">The destination.</param>
        /// <param name="caseName">The case name, which includes the seed.</param>
        /// <param name="values">The formatted values, in draw order.</param>
        private static void Emit(List<string> lines, string caseName, IEnumerable<string> values)
        {
            int index = 0;
            foreach (string value in values)
                lines.Add($"{caseName} {index++} {value}");
        }

        /// <summary>Reads the embedded golden, ignoring line-ending differences and a trailing empty line.</summary>
        /// <returns>The golden's lines.</returns>
        /// <exception cref="InvalidOperationException">Thrown when the golden is not embedded in the test assembly.</exception>
        private static List<string> ReadGolden()
        {
            using Stream stream = typeof(LegacyRngParityTests).Assembly.GetManifestResourceStream(GoldenResource)
                ?? throw new InvalidOperationException($"Embedded resource {GoldenResource} is missing.");
            using var reader = new StreamReader(stream);
            return reader.ReadToEnd().Replace("\r\n", "\n").Split('\n').Where(line => line.Length > 0).ToList();
        }
    }
}
