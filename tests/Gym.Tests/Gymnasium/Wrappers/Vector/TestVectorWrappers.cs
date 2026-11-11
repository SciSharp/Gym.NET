#nullable enable
using System;
using System.Collections.Generic;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Spaces;
using Gym.Utils;
using Gym.Vector;
using Gym.Wrappers;
using Gym.Wrappers.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Wrappers.Vector {
    /// <summary>
    ///     Gymnasium's equivalence tests of the vector wrappers (<c>test_vector_wrappers.py</c>): each vector wrapper over a
    ///     sync vector env against the single-env wrapper on every sub-env (<c>make_vec(..., wrappers=...)</c>), their spaces,
    ///     the seeded reset and 50 sampled steps compared with <see cref="EnvChecker.DataEquivalence"/>.
    /// </summary>
    /// <remarks>
    ///     Gymnasium parametrizes the test with an autoreset mode its body never passes on, so each case runs twice over the
    ///     same vector envs; the port keeps that. The image cases run over <c>CarRacing-v3</c>, as Gymnasium's do.
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_vector_wrappers.py">tests/wrappers/vector/test_vector_wrappers.py</see>.
    /// </remarks>
    [TestClass]
    public class TestVectorWrappers {
        /// <summary>The vector wrapper and the single-env wrapper on every sub-env give the same spaces and the same data, for 1 and 3 sub-envs.</summary>
        /// <param name="wrapperName">The wrapper's Gymnasium name.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_vector_wrappers.py#L41">tests/wrappers/vector/test_vector_wrappers.py (test_vector_wrapper_equivalence)</see>.</remarks>
        [TestMethod]
        [DataRow("FilterObservation")]
        [DataRow("FlattenObservation")]
        [DataRow("RescaleObservation")]
        [DataRow("ClipAction")]
        [DataRow("RescaleAction")]
        [DataRow("ClipReward")]
        public void TestVectorWrapperEquivalence(string wrapperName) => RunEquivalence(wrapperName);

        /// <summary>The image cases of <see cref="TestVectorWrapperEquivalence"/>, over <c>CarRacing-v3</c>.</summary>
        /// <param name="wrapperName">The wrapper's Gymnasium name.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/vector/test_vector_wrappers.py#L41">tests/wrappers/vector/test_vector_wrappers.py (test_vector_wrapper_equivalence)</see>.</remarks>
        [TestMethod]
        [DataRow("GrayscaleObservation")]
        [DataRow("ResizeObservation")]
        [DataRow("ReshapeObservation")]
        [DataRow("DtypeObservation")]
        public void TestVectorWrapperEquivalence_CarRacing(string wrapperName) => RunEquivalence(wrapperName);

        /// <summary>
        ///     Registers the fixture's <c>DictObsEnv-v0</c> (<c>Dict({"a": Box(0, 1), "b": Discrete(5)})</c> observations), then
        ///     compares the case for both autoreset modes and 1 and 3 sub-envs, removing the id afterwards.
        /// </summary>
        /// <param name="wrapperName">The wrapper's Gymnasium name.</param>
        /// <exception cref="AssertFailedException">Thrown at the first difference.</exception>
        private static void RunEquivalence(string wrapperName) {
            (string envId, Type vectorWrapper, Type envWrapper, Info kwargs) = Case(wrapperName);
            Register("DictObsEnv-v0", (EnvCreator)(_ => new GenericTestEnv(observationSpace: new Dict(new Dictionary<string, Space> {
                ["a"] = new Box(0, 1),
                ["b"] = new Discrete(5),
            }))));
            try {
                // Gymnasium's body never passes the autoreset mode on; the loop keeps its number of runs.
                foreach (string autoresetMode in new[] { "NextStep", "SameStep" }) {
                    foreach (int numEnvs in new[] { 1, 3 })
                        AssertEquivalent($"{wrapperName}, {autoresetMode}, num_envs={numEnvs}", envId, numEnvs, vectorWrapper, envWrapper, kwargs);
                }
            } finally {
                Registry.Remove("DictObsEnv-v0");
                // The bound arrays are shared by every wrapper built from the case, so they go only after the envs.
                foreach (object? value in kwargs.Values)
                    (value as NDArray)?.Dispose();
            }
        }

        /// <summary>The case's env id, both wrapper classes and keyword arguments (Gymnasium's parametrization).</summary>
        /// <param name="wrapperName">The wrapper's Gymnasium name.</param>
        /// <returns>The case; its arrays are the caller's to release.</returns>
        /// <exception cref="ArgumentException">Thrown for a name outside the table.</exception>
        private static (string EnvId, Type VectorWrapper, Type EnvWrapper, Info Kwargs) Case(string wrapperName) => wrapperName switch {
            "FilterObservation" => ("DictObsEnv-v0", typeof(FilterObservationVector<,>), typeof(FilterObservation<,>), new Info { ["filter_keys"] = new List<object?> { "a" } }),
            "FlattenObservation" => ("CartPole-v1", typeof(FlattenObservationVector<,>), typeof(FlattenObservation<,>), new Info()),
            "GrayscaleObservation" => ("CarRacing-v3", typeof(GrayscaleObservationVector<,>), typeof(GrayscaleObservation<,>), new Info()),
            "ResizeObservation" => ("CarRacing-v3", typeof(ResizeObservationVector<,>), typeof(ResizeObservation<,>), new Info { ["shape"] = new long[] { 35, 45 } }),
            "ReshapeObservation" => ("CarRacing-v3", typeof(ReshapeObservationVector<,>), typeof(ReshapeObservation<,>), new Info { ["shape"] = new long[] { 96, 48, 6 } }),
            "RescaleObservation" => ("CartPole-v1", typeof(RescaleObservationVector<,>), typeof(RescaleObservation<,>), new Info {
                ["min_obs"] = np.array(new[] { 0, double.NegativeInfinity, 0, double.NegativeInfinity }),
                ["max_obs"] = np.array(new[] { 1, double.PositiveInfinity, 1, double.PositiveInfinity }),
            }),
            "DtypeObservation" => ("CarRacing-v3", typeof(DtypeObservationVector<,>), typeof(DtypeObservation<,>), new Info { ["dtype"] = np.int32 }),
            "ClipAction" => ("MountainCarContinuous-v0", typeof(ClipActionVector<,>), typeof(ClipAction<,>), new Info()),
            "RescaleAction" => ("MountainCarContinuous-v0", typeof(RescaleActionVector<,>), typeof(RescaleAction<,>), new Info { ["min_action"] = 1L, ["max_action"] = 2L }),
            "ClipReward" => ("CartPole-v1", typeof(ClipRewardVector<,>), typeof(ClipReward<,>), new Info { ["min_reward"] = -0.25, ["max_reward"] = 0.75 }),
            _ => throw new ArgumentException($"no case {wrapperName}", nameof(wrapperName)),
        };

        /// <summary>
        ///     Builds the vector wrapper over <c>make_vec(env_id, num_envs, "sync")</c> and <c>make_vec</c> with the single-env
        ///     wrapper on every sub-env, then compares their spaces, <c>num_envs</c>, the reset with seed 123, and 50 steps of
        ///     the first one's samples given to both.
        /// </summary>
        /// <param name="label">The case, for the messages.</param>
        /// <param name="envId">The env id.</param>
        /// <param name="numEnvs">The number of sub-envs.</param>
        /// <param name="vectorWrapper">The vector wrapper class, an open generic.</param>
        /// <param name="envWrapper">The single-env wrapper class, an open generic.</param>
        /// <param name="kwargs">Both wrappers' other arguments by Python name.</param>
        /// <exception cref="AssertFailedException">Thrown at the first difference.</exception>
        private static void AssertEquivalent(string label, string envId, int numEnvs, Type vectorWrapper, Type envWrapper, Info kwargs) {
            using var wrapperVectorEnv = (IVectorEnv)PyKwargs.Construct(vectorWrapper, With(kwargs, MakeVec(envId, numEnvs, "sync")));
            using IVectorEnv vectorWrapperEnv = MakeVec(envId, numEnvs, "sync", wrappers: new Func<IEnv, IEnv>[] { env => (IEnv)PyKwargs.Construct(envWrapper, With(kwargs, env)) });

            Assert.AreEqual(ObjectSpace.Unview(vectorWrapperEnv.ActionSpace), ObjectSpace.Unview(wrapperVectorEnv.ActionSpace), $"{label}: action_space");
            Assert.AreEqual(ObjectSpace.Unview(vectorWrapperEnv.ObservationSpace), ObjectSpace.Unview(wrapperVectorEnv.ObservationSpace), $"{label}: observation_space");
            Assert.AreEqual(ObjectSpace.Unview(vectorWrapperEnv.SingleActionSpace), ObjectSpace.Unview(wrapperVectorEnv.SingleActionSpace), $"{label}: single_action_space");
            Assert.AreEqual(ObjectSpace.Unview(vectorWrapperEnv.SingleObservationSpace), ObjectSpace.Unview(wrapperVectorEnv.SingleObservationSpace), $"{label}: single_observation_space");
            Assert.AreEqual(vectorWrapperEnv.NumEnvs, wrapperVectorEnv.NumEnvs, $"{label}: num_envs");

            VectorResetResult<object?> wrapperVectorReset = wrapperVectorEnv.Reset(123);
            VectorResetResult<object?> vectorWrapperReset = vectorWrapperEnv.Reset(123);
            Assert.IsTrue(EnvChecker.DataEquivalence(wrapperVectorReset.Observations, vectorWrapperReset.Observations), $"{label}: reset observations");
            Assert.IsTrue(EnvChecker.DataEquivalence(wrapperVectorReset.Infos, vectorWrapperReset.Infos), $"{label}: reset infos");

            for (int step = 0; step < 50; step++) {
                object? action = wrapperVectorEnv.ActionSpace.Sample();
                VectorStepResult<object?> wrapperVectorStep = wrapperVectorEnv.Step(action);
                VectorStepResult<object?> vectorWrapperStep = vectorWrapperEnv.Step(action);
                Assert.IsTrue(EnvChecker.DataEquivalence(wrapperVectorStep.Observations, vectorWrapperStep.Observations), $"{label}: observations at step {step}");
                Assert.IsTrue(EnvChecker.DataEquivalence(wrapperVectorStep.Rewards, vectorWrapperStep.Rewards), $"{label}: rewards at step {step}");
                Assert.IsTrue(EnvChecker.DataEquivalence(wrapperVectorStep.Terminations, vectorWrapperStep.Terminations), $"{label}: terminations at step {step}");
                Assert.IsTrue(EnvChecker.DataEquivalence(wrapperVectorStep.Truncations, vectorWrapperStep.Truncations), $"{label}: truncations at step {step}");
                Assert.IsTrue(EnvChecker.DataEquivalence(wrapperVectorStep.Infos, vectorWrapperStep.Infos), $"{label}: infos at step {step}");
            }

            wrapperVectorEnv.Close();
            vectorWrapperEnv.Close();
        }

        /// <summary>A copy of the case's arguments with <c>env</c> added, as <c>wrapper(env, **kwargs)</c> passes them.</summary>
        /// <param name="kwargs">The case's arguments; their values are shared, not copied.</param>
        /// <param name="env">The env to wrap.</param>
        /// <returns>The arguments.</returns>
        private static Info With(Info kwargs, object env) => new(kwargs) { ["env"] = env };
    }
}
