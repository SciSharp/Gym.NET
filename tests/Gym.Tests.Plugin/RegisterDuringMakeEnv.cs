using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Exceptions;
using Gym.Observations;
using Gym.Spaces;
using NumSharp;

namespace Gym.Tests.Plugin {
    /// <summary>
    ///     The env this plugin registers on load (Gymnasium's <c>RegisterDuringMakeEnv</c>): <c>Discrete(1)</c> spaces and
    ///     nothing else, as <c>make</c> only needs to build it.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/utils_unregistered_env.py">tests/envs/registration/utils_unregistered_env.py</see>.</remarks>
    public sealed class RegisterDuringMakeEnv : Env<NDArray, NDArray> {
        /// <summary>Sets the <c>Discrete(1)</c> spaces.</summary>
        public RegisterDuringMakeEnv() {
            ActionSpace = new Discrete(1);
            ObservationSpace = new Discrete(1);
        }

        /// <summary>Not implemented, as Gymnasium's base <c>Env.step</c>.</summary>
        /// <param name="action">Ignored.</param>
        /// <returns>Nothing; always throws.</returns>
        /// <exception cref="NotImplementedError">Always.</exception>
        public override StepResult<NDArray> Step(NDArray action) => throw new NotImplementedError();

        /// <summary>Not implemented, as Gymnasium's base <c>Env.reset</c> only seeds.</summary>
        /// <param name="seed">Ignored.</param>
        /// <param name="options">Ignored.</param>
        /// <returns>Nothing; always throws.</returns>
        /// <exception cref="NotImplementedError">Always.</exception>
        public override ResetResult<NDArray> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) => throw new NotImplementedError();
    }

    /// <summary>The plugin's registration, Python's module-level <c>gym.register(...)</c>, run when the assembly is loaded.</summary>
    internal static class PluginRegistration {
        /// <summary>Registers <c>RegisterDuringMake-v0</c> when <c>make</c> loads the assembly and runs its module initializer.</summary>
        [ModuleInitializer]
        [SuppressMessage("Usage", "CA2255:The 'ModuleInitializer' attribute should not be used in libraries",
            Justification = "Registering on load is what this test plugin exists to show: Gymnasium's module that registers when imported.")]
        internal static void Register() => EnvRegistry.Register("RegisterDuringMake-v0", typeof(RegisterDuringMakeEnv));
    }
}
