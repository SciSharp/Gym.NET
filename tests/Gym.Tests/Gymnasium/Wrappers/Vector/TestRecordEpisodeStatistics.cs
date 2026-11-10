#nullable enable
using System;
using System.Linq;
using Gym.Envs;
using Gym.Observations;
using Gym.Spaces;
using Gym.Utils;
using Gym.Vector;
using Gym.Wrappers;
using Gym.Wrappers.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Wrappers.Vector {
    /// <summary>
    ///     Gymnasium's tests of the vector <c>RecordEpisodeStatistics</c> (<c>test_record_episode_statistics.py</c>), run
    ///     against <see cref="RecordEpisodeStatisticsVector{TObs, TAct}"/>.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_record_episode_statistics.py">tests/wrappers/vector/test_record_episode_statistics.py</see>.</remarks>
    [TestClass]
    public class TestRecordEpisodeStatistics {
        /// <summary>
        ///     The vector wrapper over CartPole and the single-env wrapper on every sub-env give the same spaces and, through
        ///     100 seeded steps, the same data, the elapsed times compared by shape and dtype only and the sub-env wrappers'
        ///     own <c>_r</c>, <c>_l</c>, <c>_t</c> masks left out.
        /// </summary>
        /// <param name="numEnvs">The number of sub-envs.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_record_episode_statistics.py#L10">tests/wrappers/vector/test_record_episode_statistics.py (test_record_episode_statistics)</see>.</remarks>
        [TestMethod]
        [DataRow(1)]
        [DataRow(3)]
        public void TestRecordEpisodeStatisticsEquivalence(int numEnvs) {
            const string envId = "CartPole-v1";
            const int numSteps = 100;
            using var wrapperVectorEnv = new RecordEpisodeStatisticsVector<object?, object?>((VectorEnv<object?, object?>)MakeVec(envId, numEnvs, "sync"));
            using IVectorEnv vectorWrapperEnv = MakeVec(envId, numEnvs, "sync",
                wrappers: new Func<IEnv, IEnv>[] { env => new RecordEpisodeStatistics<NDArray, NDArray>((Env<NDArray, NDArray>)env) });

            Assert.AreEqual(ObjectSpace.Unview(vectorWrapperEnv.ActionSpace), ObjectSpace.Unview(wrapperVectorEnv.ActionSpace));
            Assert.AreEqual(ObjectSpace.Unview(vectorWrapperEnv.ObservationSpace), ObjectSpace.Unview(wrapperVectorEnv.ObservationSpace));
            Assert.AreEqual(ObjectSpace.Unview(vectorWrapperEnv.SingleActionSpace), ObjectSpace.Unview(wrapperVectorEnv.SingleActionSpace));
            Assert.AreEqual(ObjectSpace.Unview(vectorWrapperEnv.SingleObservationSpace), ObjectSpace.Unview(wrapperVectorEnv.SingleObservationSpace));
            Assert.AreEqual(vectorWrapperEnv.NumEnvs, wrapperVectorEnv.NumEnvs);

            VectorResetResult<object?> wrapperVectorReset = wrapperVectorEnv.Reset(123);
            VectorResetResult<object?> vectorWrapperReset = vectorWrapperEnv.Reset(123);
            Assert.IsTrue(EnvChecker.DataEquivalence(wrapperVectorReset.Observations, vectorWrapperReset.Observations));
            Assert.IsTrue(EnvChecker.DataEquivalence(wrapperVectorReset.Infos, vectorWrapperReset.Infos));

            for (int step = 1; step <= numSteps; step++) {
                object? action = wrapperVectorEnv.ActionSpace.Sample();
                VectorStepResult<object?> wrapperVector = wrapperVectorEnv.Step(action);
                VectorStepResult<object?> vectorWrapper = vectorWrapperEnv.Step(action);

                Assert.IsTrue(EnvChecker.DataEquivalence(wrapperVector.Observations, vectorWrapper.Observations), $"observations at step {step}");
                Assert.IsTrue(EnvChecker.DataEquivalence(wrapperVector.Rewards, vectorWrapper.Rewards), $"rewards at step {step}");
                Assert.IsTrue(EnvChecker.DataEquivalence(wrapperVector.Terminations, vectorWrapper.Terminations), $"terminations at step {step}");
                Assert.IsTrue(EnvChecker.DataEquivalence(wrapperVector.Truncations, vectorWrapper.Truncations), $"truncations at step {step}");

                if (wrapperVector.Infos.TryGetValue("episode", out object? wrapperEpisode)) {
                    var wrapperStatistics = (Info)wrapperEpisode!;
                    var vectorStatistics = (Info)vectorWrapper.Infos["episode"]!;
                    // The elapsed times are wall-clock: only their shape and dtype can agree.
                    var wrapperVectorTime = (NDArray)wrapperStatistics["t"]!;
                    var vectorWrapperTime = (NDArray)vectorStatistics["t"]!;
                    wrapperStatistics.Remove("t");
                    vectorStatistics.Remove("t");
                    CollectionAssert.AreEqual(vectorWrapperTime.shape, wrapperVectorTime.shape, $"t's shape at step {step}");
                    Assert.AreEqual(vectorWrapperTime.dtype, wrapperVectorTime.dtype, $"t's dtype at step {step}");

                    vectorStatistics.Remove("_l");
                    vectorStatistics.Remove("_r");
                    vectorStatistics.Remove("_t");
                }

                Assert.IsTrue(EnvChecker.DataEquivalence(wrapperVector.Infos, vectorWrapper.Infos), $"infos at step {step}");
            }

            wrapperVectorEnv.Close();
            vectorWrapperEnv.Close();
        }

        /// <summary>
        ///     Three-step episodes rewarding 1 per step, under each autoreset mode (<c>DISABLED</c> resetting the ended
        ///     sub-envs through <c>reset_mask</c>): every ended episode reports a return and a length of 3, and the queues hold
        ///     only those.
        /// </summary>
        /// <param name="autoresetMode">The autoreset mode's value.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_record_episode_statistics.py#L77">tests/wrappers/vector/test_record_episode_statistics.py (test_episode_statistics_with_autoreset_mode)</see>.</remarks>
        [TestMethod]
        [DataRow("NextStep")]
        [DataRow("SameStep")]
        [DataRow("Disabled")]
        public void TestEpisodeStatisticsWithAutoresetMode(string autoresetMode) {
            const int numEnvs = 2, episodeLength = 3, numSteps = 25;
            AutoresetMode mode = AutoresetMode.From(autoresetMode);

            using var envs = new RecordEpisodeStatisticsVector<object?, object?>(new SyncVectorEnv(
                Enumerable.Range(0, numEnvs).Select(_ => (Func<IEnv>)(() => FixedLengthEnv(episodeLength))), autoresetMode: mode));

            envs.Reset(seed: 123);
            for (int i = 0; i < numSteps; i++) {
                VectorStepResult<object?> step = envs.Step(envs.ActionSpace.Sample());
                bool[] terminations = step.Terminations.ToArray<bool>(), truncations = step.Truncations.ToArray<bool>();
                bool[] dones = terminations.Zip(truncations, (terminated, truncated) => terminated || truncated).ToArray();
                if (!dones.Any(done => done))
                    continue;

                Assert.IsTrue(step.Infos.ContainsKey("episode"), $"no episode statistics after step {i}");
                var statistics = (Info)step.Infos["episode"]!;
                using NDArray returns = ((NDArray)statistics["r"]!).astype(NPTypeCode.Double);
                using NDArray lengths = ((NDArray)statistics["l"]!).astype(NPTypeCode.Int64);
                double[] r = returns.ToArray<double>();
                long[] l = lengths.ToArray<long>();
                for (int j = 0; j < numEnvs; j++) {
                    if (!dones[j])
                        continue;
                    Assert.AreEqual(episodeLength, r[j], $"return of sub-env {j} after step {i}");
                    Assert.AreEqual(episodeLength, l[j], $"length of sub-env {j} after step {i}");
                }

                if (ReferenceEquals(mode, AutoresetMode.Disabled)) {
                    using NDArray resetMask = np.array(dones);
                    envs.Reset(options: new Info { ["reset_mask"] = resetMask });
                }
            }

            Assert.IsTrue(envs.EpisodeCount > numEnvs, $"episode_count {envs.EpisodeCount}");
            Assert.IsTrue(envs.ReturnQueue.All(value => value == episodeLength), string.Join(", ", envs.ReturnQueue));
            Assert.IsTrue(envs.LengthQueue.All(value => value == episodeLength), string.Join(", ", envs.LengthQueue));
            envs.Close();
        }

        /// <summary>A test env whose episodes last <paramref name="episodeLength"/> steps, rewarding 1 per step (the test's <c>reset_func</c> and <c>step_func</c>).</summary>
        /// <param name="episodeLength">The steps per episode.</param>
        /// <returns>The env.</returns>
        private static GenericTestEnv FixedLengthEnv(int episodeLength) {
            int timestep = 0;
            return new GenericTestEnv(
                resetFunc: (self, _, _) => {
                    timestep = 0;
                    return new ResetResult<object?>(self.ObservationSpace.Sample(), new Info());
                },
                stepFunc: (self, _) => {
                    timestep++;
                    return new StepResult<object?>(self.ObservationSpace.Sample(), 1.0, timestep >= episodeLength, false, new Info());
                });
        }
    }
}
