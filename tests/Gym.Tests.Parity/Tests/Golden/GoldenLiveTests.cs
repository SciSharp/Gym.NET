using Gym.Tests.Parity.Golden;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Pair;
using Gym.Tests.Parity.Session;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests.Parity.Tests.Golden {
    /// <summary>
    ///     The live half of the golden catalog (DR-04): every case runs in lockstep against Gymnasium, and with
    ///     <c>GYMNET_PARITY_RECORD=1</c> the verified trajectory is written as the case's golden. A golden is therefore always
    ///     the recording of a cell that just passed; the replay cells (<c>GoldenReplayTests</c>) check it later without Python.
    /// </summary>
    /// <remarks>
    ///     Record from one target framework (<c>-f net10.0</c>): both would write the same content, and the store leaves an
    ///     unchanged golden untouched, but one recording keeps the run simple. Cases that depend on the platform's libm record
    ///     only on the reference platform, like every such live cell.
    /// </remarks>
    [TestClass]
    public sealed class GoldenLiveTests : ParityTestBase {
        /// <summary>
        ///     A catalog case matches Gymnasium call for call and frame for frame; in record mode its golden is (re)written,
        ///     and left untouched when the recording is unchanged.
        /// </summary>
        /// <param name="key">The case's key.</param>
        [TestMethod]
        [DynamicData(nameof(GoldenCatalog.Keys), typeof(GoldenCatalog))]
        public void Live_MatchesGymnasium_AndRecordsOnRequest(string key) {
            GoldenCase @case = GoldenCatalog.Get(key);
            if (@case.PlatformDependence != null)
                SkipOffReferencePlatform(@case.PlatformDependence);

            GoldenRecorder? recorder = GoldenStore.RecordRequested ? new GoldenRecorder(@case) : null;
            PyGymEnv python = @case.CreatePython();
            IDotNetEnv dotnet;
            try {
                dotnet = @case.CreateDotNet(@case.RenderMode);
            } catch {
                // The pair owns both sides once it exists; until then the Python env is this test's to release.
                python.Dispose();
                throw;
            }

            PairRun run;
            using (var pair = new GymPair(python, dotnet, @case.StateAttribute) { Recorder = recorder })
                run = pair.Run(@case.Seed, @case.Actions.CreateLive(), @case.MaxSteps, @case.FrameEvery);
            TestContext.WriteLine(run.ToString());

            if (recorder is null)
                return;
            var changed = recorder.Save();
            TestContext.WriteLine(changed.Count == 0 ? $"golden {@case.Key} unchanged" : $"golden {@case.Key}: {string.Join(", ", changed)}");
        }
    }
}
