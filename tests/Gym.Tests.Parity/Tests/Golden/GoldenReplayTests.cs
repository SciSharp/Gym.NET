using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Tests.Parity.Golden;
using Gym.Tests.Parity.Pair;
using Gym.Tests.Parity.Reference;
using Gym.Tests.Parity.Session;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests.Parity.Tests.Golden {
    /// <summary>
    ///     The Python-free half of the golden catalog (DR-04): every committed golden replays against Gym.NET, so machines
    ///     without the reference stack (the Linux CI job, a contributor's laptop) still check trajectories and frames. Category
    ///     <c>Golden</c>, which <c>--filter "TestCategory=Golden"</c> runs alone.
    /// </summary>
    /// <remarks>
    ///     Deliberately not a <see cref="ParityTestBase"/>: it must not require the embedded Python. A platform-dependent
    ///     case is strict on the manifest's reference platform; elsewhere a divergence is reported Inconclusive, with the
    ///     report, because another libm may round a <c>cos</c> one ulp differently (port plan §14.9).
    /// </remarks>
    [TestClass]
    [TestCategory("Golden")]
    public sealed class GoldenReplayTests {
        /// <summary>MSTest's context, for the run summary.</summary>
        public TestContext TestContext { get; set; } = null!;

        /// <summary>A case's golden is present, current, and Gym.NET reproduces it call for call and frame for frame.</summary>
        /// <param name="key">The case's key.</param>
        [TestMethod]
        [DynamicData(nameof(GoldenCatalog.Keys), typeof(GoldenCatalog))]
        public void Replay_MatchesTheGolden(string key) {
            GoldenCase @case = GoldenCatalog.Get(key);
            try {
                TestContext.WriteLine(GoldenAssert.Replay(@case).ToString());
            } catch (ParityException divergence) when (@case.PlatformDependence != null && !GoldenStore.OnReferencePlatform()) {
                Assert.Inconclusive($"{@case.PlatformDependence} is strict only on the reference platform; on {GymPythonSession.Platform} the replay diverged:{Environment.NewLine}{divergence.Message}");
            }
        }

        /// <summary>
        ///     The replay's positive control: the float64 reference transcription replays the policy golden's calls without a
        ///     difference, so the replay accepts a correct env (frames are left out: the reference does not draw).
        /// </summary>
        [TestMethod]
        public void Replay_AcceptsTheFaithfulReference() {
            GoldenCase @case = GoldenCatalog.Get("classic_control/CartPole-v1.policy.seed0");
            SkipOffReferencePlatform(@case);
            GoldenTrajectory calls = CallsOnly(GoldenAssert.LoadCurrent(@case));
            using var reference = CartPoleReference.Faithful();
            GoldenReplayRun run = GoldenAssert.Replay(@case, calls, reference);
            TestContext.WriteLine(run.ToString());
            Assert.AreEqual(500, run.Steps, "The policy golden balances the pole for all 500 steps.");
        }

        /// <summary>
        ///     The replay's negative control: a transcription with a float32 <c>tau</c> drifts by an ulp, and the replay
        ///     reports it at the first step, through the float64 state, before the float32 observation could hide it.
        /// </summary>
        [TestMethod]
        public void Replay_CatchesAFloat32Tau_AtTheFirstStep() {
            GoldenCase @case = GoldenCatalog.Get("classic_control/CartPole-v1.policy.seed0");
            GoldenTrajectory calls = CallsOnly(GoldenAssert.LoadCurrent(@case));
            using var broken = CartPoleReference.WithFloat32Tau();
            var divergence = Assert.ThrowsExactly<ParityException>(() => GoldenAssert.Replay(@case, calls, broken)).Divergence;
            Assert.AreEqual(1, divergence.Step, divergence.Format());
            Assert.IsTrue(divergence.Has("state"), divergence.Format());
            Assert.IsFalse(divergence.Has("rng"), "tau does not touch the generator.");
        }

        /// <summary>
        ///     The replay's other negative control: one extra draw after the reset changes no value yet, and the replay still
        ///     reports it at the reset, through the recorded generator state alone.
        /// </summary>
        [TestMethod]
        public void Replay_CatchesAnExtraDraw_AtTheReset() {
            GoldenCase @case = GoldenCatalog.Get("classic_control/CartPole-v1.policy.seed0");
            GoldenTrajectory calls = CallsOnly(GoldenAssert.LoadCurrent(@case));
            using var broken = CartPoleReference.WithExtraDraw();
            var divergence = Assert.ThrowsExactly<ParityException>(() => GoldenAssert.Replay(@case, calls, broken)).Divergence;
            Assert.AreEqual(0, divergence.Step, divergence.Format());
            CollectionAssert.AreEqual(new[] { "rng" }, divergence.Fields.Select(f => f.Field).Distinct().ToList(), divergence.Format());
        }

        /// <summary>
        ///     Every committed golden survives a read and a re-write byte for byte: the codec loses nothing (dtypes, shapes,
        ///     NaN payloads, 128-bit generator states, entry order), so a replay compares exactly what the live cell compared.
        /// </summary>
        /// <param name="key">The case's key.</param>
        [TestMethod]
        [DynamicData(nameof(GoldenCatalog.Keys), typeof(GoldenCatalog))]
        public void Codec_RoundTripsTheGolden_ByteForByte(string key) {
            GoldenCase @case = GoldenCatalog.Get(key);
            byte[] committed = GoldenTrajectory.Decompress(System.IO.File.ReadAllBytes(GoldenStore.TrajectoryPath(@case)));
            byte[] rewritten = GoldenTrajectory.FromJsonLines(committed).ToJsonLines();
            CollectionAssert.AreEqual(committed, rewritten, "Reading and re-writing a golden must reproduce its text exactly.");
        }

        /// <summary>
        ///     A golden recorded for other case parameters is refused as stale, field by field, rather than replayed against a
        ///     case it does not record.
        /// </summary>
        [TestMethod]
        public void Header_ReportsWhatMakesAGoldenStale() {
            GoldenCase @case = GoldenCatalog.Get("classic_control/CartPole-v1.sampled.seed42");
            GoldenHeader recorded = GoldenAssert.LoadCurrent(@case).Header;
            Assert.AreEqual(0, recorded.DifferencesFrom(@case).Count, "The committed golden records the catalog's case.");

            IReadOnlyList<string> differences = (recorded with { Seed = 7, FrameEvery = 0 }).DifferencesFrom(@case);
            CollectionAssert.AreEqual(
                new[] { "seed: recorded 7, the catalog asks for 42", "frame every: recorded 0, the catalog asks for 5" },
                differences.ToList());
        }

        /// <summary>The golden without its frames, for the reference transcriptions, which do not draw.</summary>
        /// <param name="golden">The golden.</param>
        /// <returns>The same header and calls.</returns>
        private static GoldenTrajectory CallsOnly(GoldenTrajectory golden) =>
            new(golden.Header, golden.Entries.OfType<GoldenEntry.Call>().ToList<GoldenEntry>());

        /// <summary>Reports Inconclusive off the reference platform for a case whose values rest on the platform's libm.</summary>
        /// <param name="case">The case.</param>
        /// <exception cref="AssertInconclusiveException">Thrown off the reference platform for a platform-dependent case.</exception>
        private static void SkipOffReferencePlatform(GoldenCase @case) {
            if (@case.PlatformDependence != null && !GoldenStore.OnReferencePlatform())
                Assert.Inconclusive($"{@case.PlatformDependence} is strict only on the reference platform; this is {GymPythonSession.Platform}.");
        }
    }
}
