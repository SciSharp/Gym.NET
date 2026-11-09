#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Environments;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Exceptions;
using Gym.Utils;
using NumSharp;

namespace Gym.Tests.Gymnasium.Envs {
    /// <summary>
    ///     Gymnasium's <c>tests/envs/utils.py</c>: every built-in env that can be made here, unwrapped, and their specs, for
    ///     the tests that run over all envs. The list grows as each phase registers its ids.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/utils.py">tests/envs/utils.py</see>.</remarks>
    public static class EnvTestingUtils {
        /// <summary>
        ///     Makes a built-in env without <c>make</c>'s wrappers (Gymnasium's <c>try_make_env</c>): only ids whose entry point
        ///     is a class of <c>Gym.NET.Environments</c> (Python: <c>"gymnasium.envs."</c> in the entry point), so ids the tests
        ///     register never count; envs that cannot be made here are skipped with a warning.
        /// </summary>
        /// <param name="envSpec">The spec.</param>
        /// <returns>The unwrapped env, which the caller owns, or <see langword="null"/>.</returns>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/utils.py#L8">tests/envs/utils.py (try_make_env)</see>.</remarks>
        public static IEnv? TryMakeEnv(EnvSpec envSpec) {
            bool builtIn = envSpec.EntryPoint is Type type && type.Assembly == typeof(GymEnvironments).Assembly
                           || envSpec.EntryPoint is string entryPoint && entryPoint.StartsWith("Gym.Environments.Envs.", StringComparison.Ordinal);
            if (!builtIn)
                return null;
            try {
                return envSpec.Make(disableEnvChecker: true).Unwrapped;
            } catch (Exception e) when (e is ImportError or AttributeError or DependencyNotInstalled or MissingArgument) {
                Logger.Warn($"Not testing {envSpec.Id} due to error: {e.Message}");
                return null;
            }
        }

        /// <summary>Every built-in env that can be made, unwrapped (Gymnasium's <c>all_testing_initialised_envs</c>); the caller owns them.</summary>
        /// <returns>A new list of new envs.</returns>
        public static List<IEnv> AllTestingInitialisedEnvs() => Registry.Values.Select(TryMakeEnv).Where(env => env is not null).Select(env => env!).ToList();

        /// <summary>
        ///     The specs of <see cref="AllTestingInitialisedEnvs"/> (Gymnasium's <c>all_testing_env_specs</c>): the minimal specs
        ///     <c>make</c> gives unwrapped envs, so <c>order_enforce</c> is <see langword="false"/> in every one.
        /// </summary>
        /// <returns>The specs; the envs they came from are disposed.</returns>
        public static List<EnvSpec> AllTestingEnvSpecs() {
            var specs = new List<EnvSpec>();
            foreach (IEnv env in AllTestingInitialisedEnvs()) {
                using (env)
                    specs.Add(env.Spec!);
            }

            return specs;
        }
    }
}
