#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Observations;
using Gym.Vector;
using Gym.Wrappers;
using Gym.Wrappers.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Wrappers.Vector {
    /// <summary>
    ///     Gymnasium's tests of the vector <c>NormalizeReward</c> (<c>test_normalize_reward.py</c>), run against
    ///     <see cref="NormalizeRewardVector{TObs, TAct}"/>.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_normalize_reward.py">tests/wrappers/vector/test_normalize_reward.py</see>.</remarks>
    [TestClass]
    public class TestNormalizeReward {
        /// <summary>
        ///     After 100 steps of three ten-step episodes rewarding 1 at their end, the discounted sums of the next 100
        ///     normalized rewards have a standard deviation near 0.89 in every sub-env.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_normalize_reward.py#L27">tests/wrappers/vector/test_normalize_reward.py (test_functionality)</see>.</remarks>
        [TestMethod]
        public void TestFunctionality() {
            const int nEnvs = 3, nSteps = 100;
            using var env = new NormalizeRewardVector<object?, object?>(new SyncVectorEnv(Enumerable.Range(0, nEnvs).Select(_ => (Func<IEnv>)Thunk)));

            env.Reset();
            for (int i = 0; i < nSteps; i++)
                env.Step(env.ActionSpace.Sample());

            env.Reset();
            var forwardRets = new List<double[]>();
            double[] accumulatedRew = new double[nEnvs];
            for (int i = 0; i < nSteps; i++) {
                VectorStepResult<object?> step = env.Step(env.ActionSpace.Sample());
                bool[] terminated = step.Terminations.ToArray<bool>(), truncated = step.Truncations.ToArray<bool>();
                double[] reward = step.Rewards.ToArray<double>();
                // accumulated_rew = accumulated_rew * 0.9 * dones + reward, as Gymnasium's test writes it.
                accumulatedRew = accumulatedRew.Select((value, j) => value * 0.9 * (terminated[j] || truncated[j] ? 1 : 0) + reward[j]).ToArray();
                forwardRets.Add(accumulatedRew);
            }

            env.Close();

            for (int j = 0; j < nEnvs; j++) {
                double[] column = forwardRets.Select(row => row[j]).ToArray();
                double mean = column.Average();
                double std = Math.Sqrt(column.Select(value => (value - mean) * (value - mean)).Average());
                Assert.AreEqual(0.89, std, 0.1 + 1e-5 * 0.89, $"sub-env {j}");
            }
        }

        /// <summary>The vector wrapper's return variance after 100 steps of three envs is within 10% of the single-env wrapper's after 100 steps of one autoresetting env.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_normalize_reward.py#L55">tests/wrappers/vector/test_normalize_reward.py (test_against_wrapper)</see>.</remarks>
        [TestMethod]
        public void TestAgainstWrapper() {
            const int nEnvs = 3, nSteps = 100;
            const double rtol = 0.1, atol = 0;
            using var vecEnv = new NormalizeRewardVector<object?, object?>(new SyncVectorEnv(Enumerable.Range(0, nEnvs).Select(_ => (Func<IEnv>)Thunk)));
            vecEnv.Reset();
            for (int i = 0; i < nSteps; i++)
                vecEnv.Step(vecEnv.ActionSpace.Sample());

            using var env = new NormalizeReward<object?, object?>(new Autoreset<object?, object?>(Thunk()));
            env.Reset();
            for (int i = 0; i < nSteps; i++)
                env.Step(env.ActionSpace.Sample());

            double single = env.ReturnRms.Var.GetDouble(), vector = vecEnv.ReturnRms.Var.GetDouble();
            Assert.IsTrue(Math.Abs(single - vector) <= atol + rtol * Math.Abs(vector), $"return_rms.var {single} against {vector}");
        }

        /// <summary>One env: the vector wrapper's return statistics after 50 seeded steps equal those of the single-env wrapper inside a sync vector env.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_normalize_reward.py#L73">tests/wrappers/vector/test_normalize_reward.py (test_equivalence_with_wrapper)</see>.</remarks>
        [TestMethod]
        public void TestEquivalenceWithWrapper() {
            const int nSteps = 50;
            static IEnv ThunkWithNormalize() => new NormalizeReward<object?, object?>(Thunk());

            using var perEnv = new SyncVectorEnv(new Func<IEnv>[] { ThunkWithNormalize });
            perEnv.Reset(seed: 42);
            for (int i = 0; i < nSteps; i++)
                perEnv.Step(perEnv.ActionSpace.Sample());

            using var vecEnv = new NormalizeRewardVector<object?, object?>(new SyncVectorEnv(new Func<IEnv>[] { Thunk }));
            vecEnv.Reset(seed: 42);
            for (int i = 0; i < nSteps; i++)
                vecEnv.Step(vecEnv.ActionSpace.Sample());

            var normalized = (NormalizeReward<object?, object?>)perEnv.Envs[0];
            Assert.AreEqual(normalized.ReturnRms.Count, vecEnv.ReturnRms.Count);
            Assert.IsTrue(np.allclose(vecEnv.ReturnRms.Mean, normalized.ReturnRms.Mean, rtol: 1e-4), $"mean {vecEnv.ReturnRms.Mean} against {normalized.ReturnRms.Mean}");
            Assert.IsTrue(np.allclose(vecEnv.ReturnRms.Var, normalized.ReturnRms.Var, rtol: 1e-4), $"var {vecEnv.ReturnRms.Var} against {normalized.ReturnRms.Var}");
            perEnv.Close();
            vecEnv.Close();
        }

        /// <summary>A <c>gamma</c> outside [0, 1] is refused, as the single-env wrapper refuses it.</summary>
        /// <param name="gamma">The discount.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_normalize_reward.py#L98">tests/wrappers/vector/test_normalize_reward.py (test_gamma_outside_unit_interval_is_rejected)</see>.</remarks>
        [TestMethod]
        [DataRow(-1.0)]
        [DataRow(1.01)]
        [DataRow(2.0)]
        [DataRow(99.0)]
        public void TestGammaOutsideUnitIntervalIsRejected(double gamma) {
            using var vecEnv = new SyncVectorEnv(new Func<IEnv>[] { Thunk });
            var error = Assert.ThrowsExactly<InvalidBound>(() => new NormalizeRewardVector<object?, object?>(vecEnv, gamma: gamma));
            StringAssert.Contains(error.Message, "`gamma` should be in the interval");
            vecEnv.Close();
        }

        /// <summary>A non-positive <c>epsilon</c> is refused.</summary>
        /// <param name="epsilon">The epsilon.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_normalize_reward.py#L107">tests/wrappers/vector/test_normalize_reward.py (test_non_positive_epsilon_is_rejected)</see>.</remarks>
        [TestMethod]
        [DataRow(0.0)]
        [DataRow(-1e-8)]
        [DataRow(-1.0)]
        public void TestNonPositiveEpsilonIsRejected(double epsilon) {
            using var vecEnv = new SyncVectorEnv(new Func<IEnv>[] { Thunk });
            var error = Assert.ThrowsExactly<InvalidBound>(() => new NormalizeRewardVector<object?, object?>(vecEnv, epsilon: epsilon));
            StringAssert.Contains(error.Message, "`epsilon` should be strictly positive");
            vecEnv.Close();
        }

        /// <summary>
        ///     Under <c>SAME_STEP</c>, the step that ends an episode resets its accumulated return to 0, and every step (12
        ///     steps of 2 four-step episodes) updates <c>return_rms</c>.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_normalize_reward.py#L115">tests/wrappers/vector/test_normalize_reward.py (test_same_step_autoreset_updates_return_rms)</see>.</remarks>
        [TestMethod]
        public void TestSameStepAutoresetUpdatesReturnRms() {
            const int nEnvs = 2, episodeLength = 4, nSteps = 12;
            var timesteps = new Dictionary<GenericTestEnv, int>();

            ResetResult<object?> ResetFunc(GenericTestEnv self, BigInteger? seed, IReadOnlyDictionary<string, object?>? options) {
                timesteps[self] = 0;
                return new ResetResult<object?>(self.ObservationSpace.Sample(), new Info());
            }

            StepResult<object?> StepFunc(GenericTestEnv self, object? action) {
                timesteps[self]++;
                return new StepResult<object?>(self.ObservationSpace.Sample(), 1.0, timesteps[self] >= episodeLength, false, new Info());
            }

            using var env = new NormalizeRewardVector<object?, object?>(new SyncVectorEnv(
                Enumerable.Range(0, nEnvs).Select(_ => (Func<IEnv>)(() => new GenericTestEnv(resetFunc: ResetFunc, stepFunc: StepFunc))), autoresetMode: AutoresetMode.SameStep));

            env.Reset(seed: 123);
            for (int i = 0; i < nSteps; i++) {
                VectorStepResult<object?> step = env.Step(env.ActionSpace.Sample());
                bool[] terminated = step.Terminations.ToArray<bool>(), truncated = step.Truncations.ToArray<bool>();
                // accumulated_reward is float32, as Gymnasium's.
                using NDArray wide = env.AccumulatedReward.astype(NPTypeCode.Double);
                double[] accumulated = wide.ToArray<double>();
                for (int j = 0; j < nEnvs; j++) {
                    if (terminated[j] || truncated[j])
                        Assert.AreEqual(0.0, accumulated[j], $"accumulated_reward[{j}] after step {i}");
                }
            }

            // Every step is a real environment step under SAME_STEP; np.isclose's default tolerances.
            Assert.AreEqual(nSteps * nEnvs, env.ReturnRms.Count, 1e-8 + 1e-5 * nSteps * nEnvs);
            env.Close();
        }

        /// <summary>
        ///     A test env whose episodes last ten steps and reward 1 at their last (<c>thunk</c>): its reset zeroes a step
        ///     counter, its step counts and terminates at ten.
        /// </summary>
        /// <returns>The env.</returns>
        private static GenericTestEnv Thunk() {
            int stepId = 0;
            return new GenericTestEnv(
                resetFunc: (self, _, _) => {
                    stepId = 0;
                    return new ResetResult<object?>(self.ObservationSpace.Sample(), new Info());
                },
                stepFunc: (self, _) => {
                    stepId++;
                    bool terminated = stepId == 10;
                    return new StepResult<object?>(self.ObservationSpace.Sample(), terminated ? 1.0 : 0.0, terminated, false, new Info());
                });
        }
    }
}
