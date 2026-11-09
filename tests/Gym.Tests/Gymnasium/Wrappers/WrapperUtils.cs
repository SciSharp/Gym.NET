#nullable enable
using System;
using System.Collections.Generic;
using System.Numerics;
using Gym.Envs;
using Gym.Observations;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests.Gymnasium.Wrappers {
    /// <summary>
    ///     The helpers of Gymnasium's <c>tests/wrappers/utils.py</c>: its constants, the reset and step functions that record
    ///     what a wrapper passed to the env (bound to a <see cref="GenericTestEnv"/>), <c>check_obs</c> and
    ///     <c>has_wrapper</c>.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/utils.py">tests/wrappers/utils.py</see>.</remarks>
    public static class WrapperUtils {
        /// <summary>The seed the wrapper tests use (<c>SEED</c>).</summary>
        public const long Seed = 42;

        /// <summary>The env the wrapper tests make (<c>ENV_ID</c>).</summary>
        public const string EnvId = "CartPole-v1";

        /// <summary>A valid action of <see cref="EnvId"/> (<c>DISCRETE_ACTION</c>).</summary>
        public const long DiscreteAction = 0;

        /// <summary>The number of sub-envs of the vector wrapper tests (<c>NUM_ENVS</c>).</summary>
        public const int NumEnvs = 3;

        /// <summary>The steps the wrapper tests take (<c>NUM_STEPS</c>).</summary>
        public const int NumSteps = 20;

        /// <summary>A reset that returns <c>options["obs"]</c> as the observation and records it in the info (<c>record_obs_reset</c>).</summary>
        /// <param name="self">The env.</param>
        /// <param name="seed">Ignored.</param>
        /// <param name="options">Must hold <c>obs</c>.</param>
        /// <returns><c>(options["obs"], {"obs": options["obs"]})</c>, the same object twice.</returns>
        /// <exception cref="KeyNotFoundException">Thrown when the options lack <c>obs</c>.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/utils.py#L16">tests/wrappers/utils.py (record_obs_reset)</see>.</remarks>
        public static ResetResult<object?> RecordObsReset(GenericTestEnv self, BigInteger? seed, IReadOnlyDictionary<string, object?>? options) {
            object? obs = options!["obs"];
            return new ResetResult<object?>(obs, new Info { ["obs"] = obs });
        }

        /// <summary>A reset that samples the observation space and records the sample in the info (<c>record_random_obs_reset</c>).</summary>
        /// <param name="self">The env.</param>
        /// <param name="seed">Ignored.</param>
        /// <param name="options">Ignored.</param>
        /// <returns><c>(obs, {"obs": obs})</c>.</returns>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/utils.py#L21">tests/wrappers/utils.py (record_random_obs_reset)</see>.</remarks>
        public static ResetResult<object?> RecordRandomObsReset(GenericTestEnv self, BigInteger? seed, IReadOnlyDictionary<string, object?>? options) {
            object? obs = self.ObservationSpace.Sample();
            return new ResetResult<object?>(obs, new Info { ["obs"] = obs });
        }

        /// <summary>A step that records the action it got (<c>record_action_step</c>).</summary>
        /// <param name="self">The env.</param>
        /// <param name="action">The action.</param>
        /// <returns><c>(0, 0, False, False, {"action": action})</c>.</returns>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/utils.py#L27">tests/wrappers/utils.py (record_action_step)</see>.</remarks>
        public static StepResult<object?> RecordActionStep(GenericTestEnv self, object? action) =>
            new(0L, 0, false, false, new Info { ["action"] = action });

        /// <summary>A step that samples the observation space and records the sample (<c>record_random_obs_step</c>).</summary>
        /// <param name="self">The env.</param>
        /// <param name="action">Ignored.</param>
        /// <returns><c>(obs, 0, False, False, {"obs": obs})</c>.</returns>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/utils.py#L32">tests/wrappers/utils.py (record_random_obs_step)</see>.</remarks>
        public static StepResult<object?> RecordRandomObsStep(GenericTestEnv self, object? action) {
            object? obs = self.ObservationSpace.Sample();
            return new StepResult<object?>(obs, 0, false, false, new Info { ["obs"] = obs });
        }

        /// <summary>A step that returns its action as the observation (<c>record_action_as_obs_step</c>).</summary>
        /// <param name="self">The env.</param>
        /// <param name="action">The action, which becomes the observation.</param>
        /// <returns><c>(action, 0, False, False, {"obs": action})</c>.</returns>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/utils.py#L38">tests/wrappers/utils.py (record_action_as_obs_step)</see>.</remarks>
        public static StepResult<object?> RecordActionAsObsStep(GenericTestEnv self, object? action) =>
            new(action, 0, false, false, new Info { ["obs"] = action });

        /// <summary>A step that returns its action as the reward (<c>record_action_as_record_step</c>).</summary>
        /// <param name="self">The env.</param>
        /// <param name="action">A number, which becomes the reward.</param>
        /// <returns><c>(0, action, False, False, {"reward": action})</c>.</returns>
        /// <exception cref="InvalidCastException">Thrown when the action is no number.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/utils.py#L43">tests/wrappers/utils.py (record_action_as_record_step)</see>.</remarks>
        public static StepResult<object?> RecordActionAsRecordStep(GenericTestEnv self, object? action) =>
            new(0L, Convert.ToDouble(action), false, false, new Info { ["reward"] = action });

        /// <summary>
        ///     Checks a wrapper's observation against the env's (<c>check_obs</c>): each belongs to its own env's space and,
        ///     when <paramref name="strict"/>, not to the other's.
        /// </summary>
        /// <param name="env">The base env.</param>
        /// <param name="wrappedEnv">The wrapper.</param>
        /// <param name="transformedObs">The wrapper's observation.</param>
        /// <param name="originalObs">The base env's observation.</param>
        /// <param name="strict">Whether each must also be outside the other's space.</param>
        /// <exception cref="AssertFailedException">Thrown when a check fails.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/utils.py#L48">tests/wrappers/utils.py (check_obs)</see>.</remarks>
        public static void CheckObs(IEnv env, IEnv wrappedEnv, object? transformedObs, object? originalObs, bool strict = true) {
            Assert.IsTrue(wrappedEnv.ObservationSpace.Contains(transformedObs), $"{transformedObs}, {wrappedEnv.ObservationSpace}");
            Assert.IsTrue(env.ObservationSpace.Contains(originalObs), $"{originalObs}, {env.ObservationSpace}");
            if (strict) {
                Assert.IsFalse(env.ObservationSpace.Contains(transformedObs), $"{transformedObs}, {env.ObservationSpace}");
                Assert.IsFalse(wrappedEnv.ObservationSpace.Contains(originalObs), $"{originalObs}, {wrappedEnv.ObservationSpace}");
            }
        }

        /// <summary>
        ///     Whether a wrapper class is anywhere in an env's wrapper stack (Gymnasium's <c>has_wrapper</c>), walking
        ///     <c>wrapped_env.env</c> down to the unwrapped env.
        /// </summary>
        /// <param name="wrappedEnv">The outermost env.</param>
        /// <param name="wrapperType">The wrapper class, open (<c>typeof(TimeLimit&lt;,&gt;)</c>, any type arguments) or closed.</param>
        /// <returns><see langword="true"/> when a layer is of that class.</returns>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/utils.py#L87">tests/wrappers/utils.py (has_wrapper)</see>.</remarks>
        public static bool HasWrapper(IEnv wrappedEnv, Type wrapperType) {
            IEnv? env = wrappedEnv;
            while (env is IWrapper wrapper) {
                if (IsOfClass(env.GetType(), wrapperType))
                    return true;
                env = wrapper.Env;
            }

            return false;
        }

        /// <summary>Python's <c>isinstance</c> for an open generic class: the type or one of its bases is a closed form of it.</summary>
        /// <param name="type">The env's class.</param>
        /// <param name="wrapperType">The wrapper class, open or closed.</param>
        /// <returns><see langword="true"/> when the env is one.</returns>
        private static bool IsOfClass(Type type, Type wrapperType) {
            if (!wrapperType.IsGenericTypeDefinition)
                return wrapperType.IsAssignableFrom(type);
            for (Type? level = type; level != null; level = level.BaseType) {
                if (level.IsGenericType && level.GetGenericTypeDefinition() == wrapperType)
                    return true;
            }

            return false;
        }
    }
}
