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
    ///     L3 byte parity with Gymnasium itself: Gym.NET's seeded spaces and CartPole reset replay the committed golden that
    ///     <c>tools/parity/gymnasium_rng.py</c> recorded from Gymnasium @ 9e04324 (<c>Parity/gymnasium_rng.golden.txt</c>, 783
    ///     values over 9 seeds up to 2^40), and must match it bit for bit.
    /// </summary>
    /// <remarks>
    ///     Both sides draw from NumPy's <c>Generator(PCG64(SeedSequence(seed)))</c>: Gymnasium through
    ///     <c>seeding.np_random</c>, Gym.NET through <see cref="Seeding.NpRandom"/> and NumSharp's port of the generator. What
    ///     the golden does not cover yet: CartPole's dynamics (float32 constants, P4), LunarLander (another physics engine
    ///     and draw order, P8), and the spaces Gym.NET does not have.
    /// </remarks>
    [TestClass]
    public class GymnasiumRngParityTests {
        /// <summary>The seeds of <c>gymnasium_rng.py</c>, in its order.</summary>
        private static readonly long[] Seeds = { 0, 1, 42, 1000, (1L << 31) - 1, (1L << 31) + 5, (1L << 32) - 1, 1L << 32, 1L << 40 };

        /// <summary>The logical name the test project embeds the golden under (see Gym.Tests.csproj).</summary>
        private const string GoldenResource = "Gym.Tests.Parity.gymnasium_rng.golden.txt";

        /// <summary>
        ///     Every one of the 783 values equals Gymnasium's. A failure names how many lines differ and the first difference,
        ///     whose case and index identify the space, the draw and the seed.
        /// </summary>
        [TestMethod]
        public void SpacesAndCartPoleReset_MatchGymnasiumBitForBit()
        {
            List<string> expected = ReadGolden();
            List<string> actual = Record();

            Assert.AreEqual(783, expected.Count, "The golden must hold the 783 values of gymnasium_rng.py.");
            Assert.AreEqual(expected.Count, actual.Count, "Gym.NET must record as many values as the golden holds.");
            var mismatches = Enumerable.Range(0, expected.Count).Where(i => expected[i] != actual[i]).ToList();
            if (mismatches.Count > 0)
            {
                int first = mismatches[0];
                Assert.Fail($"{mismatches.Count} of {expected.Count} values differ from Gymnasium. First at line {first + 1}: expected '{expected[first]}', got '{actual[first]}'.");
            }
        }

        /// <summary>
        ///     The real <see cref="LunarLanderEnv"/> consumes exactly the draws its code is written to make: two
        ///     <c>integers(-9999, 9999)</c> in the constructor, then 2 + 12 <c>uniform</c> in <see cref="LunarLanderEnv.Reset"/>
        ///     and two per step for the zero step and 20 more. Afterwards its generator's complete PCG64 state equals that of a
        ///     generator that made those draws by hand. A draw added, removed or reordered by kind fails it.
        /// </summary>
        /// <remarks>
        ///     LunarLander is not compared with Gymnasium's values: Gymnasium draws the terrain before the force and the wind in
        ///     reset (B-17, B-18), and its physics is Box2D, not Aether.
        /// </remarks>
        [TestMethod]
        public void LunarLander_ConsumesExactlyItsDrawPattern()
        {
            foreach (long seed in Seeds)
            {
                var envGenerator = Seeding.NpRandom(seed).Rng;
                var env = new LunarLanderEnv(NullEnvViewer.Factory, random_state: envGenerator);
                // The zero step inside Reset is the first of the 21 dispersion pairs; 20 explicit steps make the rest.
                env.Reset().Dispose();
                for (int i = 0; i < 20; i++)
                    env.Step((long) LunarLanderDiscreteActions.Nothing).Observation.Dispose();

                var replayed = Seeding.NpRandom(seed).Rng;
                // LunarLanderEnv computes VIEWPORT_H / SCALE in float32 (VIEWPORT_H = 400) and halves it in float32.
                float h = 400 / LunarLanderEnv.SCALE;
                for (int i = 0; i < 2; i++)
                    replayed.integers(-9999, 9999).Dispose();
                for (int i = 0; i < 2; i++)
                    replayed.uniform(-LunarLanderEnv.INITIAL_RANDOM, LunarLanderEnv.INITIAL_RANDOM).Dispose();
                for (int i = 0; i < 12; i++)
                    replayed.uniform(0f, h / 2f).Dispose();
                for (int i = 0; i < 42; i++)
                    replayed.uniform(-1f, 1f).Dispose();

                var envState = ((PCG64) envGenerator.bit_generator).state;
                var replayedState = ((PCG64) replayed.bit_generator).state;
                Assert.AreEqual(replayedState.state, envState.state, $"Seed {seed}: the env's PCG64 state differs.");
                Assert.AreEqual(replayedState.inc, envState.inc, $"Seed {seed}: the env's PCG64 increment differs.");
                Assert.AreEqual(replayedState.has_uint32, envState.has_uint32, $"Seed {seed}: the buffered 32-bit half differs.");
                Assert.AreEqual(replayedState.uinteger, envState.uinteger, $"Seed {seed}: the buffered 32-bit value differs.");
            }
        }

        /// <summary>Produces Gym.NET's lines in the golden's order and format.</summary>
        /// <returns>One <c>"&lt;case&gt; &lt;index&gt; &lt;value&gt;"</c> line per value.</returns>
        private static List<string> Record()
        {
            var lines = new List<string>();
            foreach (long seed in Seeds)
            {
                var (rng, reportedSeed) = Seeding.NpRandom(seed);
                Emit(lines, $"seeding.seed.s{seed}", new[] { reportedSeed.ToString(CultureInfo.InvariantCulture) });
                using (NDArray draws = rng.random(8))
                    Emit(lines, $"seeding.random.s{seed}", draws.ToArray<double>().Select(F64));

                var space = new Discrete(4, seed: seed);
                Emit(lines, $"discrete.sample.s{seed}", Enumerable.Range(0, 10).Select(_ => Draw(space)));
                using (NDArray mask = np.array(new sbyte[] { 1, 0, 1, 1 }))
                    Emit(lines, $"discrete.masked.s{seed}", Enumerable.Range(0, 5).Select(_ => Draw(space, mask)));
                var started = new Discrete(3, seed: seed, start: 5);
                Emit(lines, $"discrete.start.s{seed}", Enumerable.Range(0, 5).Select(_ => Draw(started)));
                var large = new Discrete(3_000_000_000, seed: seed);
                Emit(lines, $"discrete.large.s{seed}", Enumerable.Range(0, 5).Select(_ => Draw(large)));
                var huge = new Discrete(1L << 40, seed: seed);
                Emit(lines, $"discrete.huge.s{seed}", Enumerable.Range(0, 5).Select(_ => Draw(huge)));

                using (NDArray s = new Box(-1f, 1f, new Shape(3), np.float32, seed: seed).Sample())
                    Emit(lines, $"box.bounded.s{seed}", s.ToArray<float>().Select(F32));
                using (NDArray low = np.array(new[] { float.NegativeInfinity, 0f, float.NegativeInfinity, -1f }))
                using (NDArray high = np.array(new[] { float.PositiveInfinity, float.PositiveInfinity, 0f, 1f }))
                using (NDArray s = new Box(low, high, np.float32, seed: seed).Sample())
                    Emit(lines, $"box.mixed.s{seed}", s.ToArray<float>().Select(F32));
                using (NDArray s = new Box(float.NegativeInfinity, float.PositiveInfinity, new Shape(3), np.float64, seed: seed).Sample())
                    Emit(lines, $"box.normal.s{seed}", s.ToArray<double>().Select(F64));
                using (NDArray s = new Box(0f, float.PositiveInfinity, new Shape(3), np.float64, seed: seed).Sample())
                    Emit(lines, $"box.exponential.s{seed}", s.ToArray<double>().Select(F64));
                using (NDArray s = new Box(float.NegativeInfinity, 0f, new Shape(3), np.float64, seed: seed).Sample())
                    Emit(lines, $"box.upper.s{seed}", s.ToArray<double>().Select(F64));
                using (NDArray s = new Box(0f, 5f, new Shape(8), np.int32, seed: seed).Sample())
                    Emit(lines, $"box.int32.s{seed}", s.ToArray<int>().Select(v => v.ToString(CultureInfo.InvariantCulture)));
                using (NDArray s = new Box(-5f, 5f, new Shape(8), np.int8, seed: seed).Sample())
                    Emit(lines, $"box.int8.s{seed}", s.ToArray<sbyte>().Select(v => v.ToString(CultureInfo.InvariantCulture)));
                using (NDArray s = new Box(0f, 255f, new Shape(8), np.uint8, seed: seed).Sample())
                    Emit(lines, $"box.uint8.s{seed}", s.ToArray<byte>().Select(v => v.ToString(CultureInfo.InvariantCulture)));

                // Gym.NET's CartPole hands out its float64 state (B-13, P4); Gymnasium's observation is that state in float32.
                var cartPole = new CartPoleEnv(NullEnvViewer.Factory, Seeding.NpRandom(seed).Rng);
                using (NDArray reset = cartPole.Reset())
                {
                    double[] state = reset.ToArray<double>();
                    Emit(lines, $"cartpole.reset.state.s{seed}", state.Select(F64));
                    Emit(lines, $"cartpole.reset.obs.s{seed}", state.Select(v => F32((float) v)));
                }
            }
            return lines;
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
            using Stream stream = typeof(GymnasiumRngParityTests).Assembly.GetManifestResourceStream(GoldenResource)
                ?? throw new InvalidOperationException($"Embedded resource {GoldenResource} is missing.");
            using var reader = new StreamReader(stream);
            return reader.ReadToEnd().Replace("\r\n", "\n").Split('\n').Where(line => line.Length > 0).ToList();
        }
    }
}
