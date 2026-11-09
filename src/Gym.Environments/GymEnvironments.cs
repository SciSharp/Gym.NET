#nullable enable
using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Threading;
using Gym.Environments.Envs.Classic;
using Gym.Environments.Envs.ToyText;
using Gym.Envs;
using Gym.Envs.Registration;

namespace Gym.Environments {
    /// <summary>
    ///     Registers Gym.NET's built-in env ids with the registry, as importing <c>gymnasium</c> registers Gymnasium's
    ///     (<c>gymnasium/envs/__init__.py</c>): the same ids, step limits and reward thresholds, with the env classes as entry
    ///     points (DR-12). After it, <c>Gymnasium.Make("CartPole-v1")</c> works.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Nothing needs to call it in a C# app: the package compiles a module initializer into the app that calls
    ///     <see cref="RegisterUnlessDisabled"/> before <c>Main</c> (DR-17), and this assembly's own module initializer does the
    ///     same when it is loaded first. F# and Visual Basic apps call <see cref="Register"/> at start-up.
    ///     </para>
    ///     <para>
    ///     Setting the environment variable <see cref="DisableRegistrationVariable"/> (to anything but <c>0</c> or
    ///     <c>false</c>) turns both initializers off (DR-14), for an app that registers its own ids under the same names;
    ///     <see cref="Register"/> still registers when called.
    ///     </para>
    ///     <para>
    ///     Ids register in Gymnasium's order as their phases port them: classic_control's five now (CartPole-v0/v1,
    ///     MountainCar-v0, MountainCarContinuous-v0, Pendulum-v1, Acrobot-v1). The vectorized entry points
    ///     (<c>CartPoleVectorEnv</c>) come with the vector envs.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/__init__.py">gymnasium/envs/__init__.py</see>.</para>
    /// </remarks>
    public static class GymEnvironments {
        /// <summary>The environment variable that disarms the automatic registration (DR-14): <c>GYMNET_DISABLE_ENV_REGISTRATION</c>.</summary>
        public const string DisableRegistrationVariable = "GYMNET_DISABLE_ENV_REGISTRATION";

        /// <summary>1 once <see cref="Register"/> ran, so each id registers once whatever calls it how often.</summary>
        private static int _registered;

        /// <summary>
        ///     Whether <see cref="DisableRegistrationVariable"/> disarms the automatic registration: it is set to anything but
        ///     empty, <c>0</c> or <c>false</c> (case-insensitive).
        /// </summary>
        public static bool IsAutomaticRegistrationDisabled {
            get {
                string? value = Environment.GetEnvironmentVariable(DisableRegistrationVariable);
                return !string.IsNullOrWhiteSpace(value) && value.Trim() != "0" && !string.Equals(value.Trim(), "false", StringComparison.OrdinalIgnoreCase);
            }
        }

        /// <summary>
        ///     Registers every built-in id, once per process: later calls do nothing, so an id an app removed from the
        ///     registry stays removed. Registering an id the app already registered replaces it with Gymnasium's
        ///     <c>Overriding environment ... already in registry.</c> warning.
        /// </summary>
        /// <exception cref="Gym.Exceptions.RegistrationError">Thrown when the app registered a name both ways (versioned and not) first.</exception>
        public static void Register() {
            if (Interlocked.Exchange(ref _registered, 1) == 1)
                return;

            // Classic
            // ----------------------------------------
            EnvRegistry.Register("CartPole-v0", typeof(CartPoleEnv), maxEpisodeSteps: 200, rewardThreshold: 195.0);
            EnvRegistry.Register("CartPole-v1", typeof(CartPoleEnv), maxEpisodeSteps: 500, rewardThreshold: 475.0);
            EnvRegistry.Register("MountainCar-v0", typeof(MountainCarEnv), maxEpisodeSteps: 200, rewardThreshold: -110.0);
            // One class for both MountainCars (DR-32): the continuous id carries continuous=True in its kwargs, where
            // Gymnasium's entry point is a class of its own and its kwargs are empty.
            EnvRegistry.Register("MountainCarContinuous-v0", typeof(MountainCarEnv), maxEpisodeSteps: 999, rewardThreshold: 90.0,
                kwargs: new MountainCarEnvOptions { Continuous = true });
            EnvRegistry.Register("Pendulum-v1", typeof(PendulumEnv), maxEpisodeSteps: 200);
            EnvRegistry.Register("Acrobot-v1", typeof(AcrobotEnv), rewardThreshold: -100.0, maxEpisodeSteps: 500);

            // Toy Text
            // ----------------------------------------
            // An Info rather than the options record: the kwargs keep Gymnasium's order (sab, then natural), which the spec prints.
            EnvRegistry.Register("Blackjack-v1", typeof(BlackjackEnv), kwargs: new Info { ["sab"] = true, ["natural"] = false });
            EnvRegistry.Register("FrozenLake-v1", typeof(FrozenLakeEnv), maxEpisodeSteps: 100, rewardThreshold: 0.70, kwargs: new FrozenLakeEnvOptions { MapName = "4x4" });
            EnvRegistry.Register("FrozenLake8x8-v1", typeof(FrozenLakeEnv), maxEpisodeSteps: 200, rewardThreshold: 0.85, kwargs: new FrozenLakeEnvOptions { MapName = "8x8" });
            EnvRegistry.Register("CliffWalking-v1", typeof(CliffWalkingEnv));
            EnvRegistry.Register("CliffWalkingSlippery-v1", typeof(CliffWalkingEnv), kwargs: new CliffWalkingEnvOptions { IsSlippery = true });
            // Gymnasium registers reward_threshold=8, an int; Gym.NET's thresholds are float64 like its rewards, so the spec prints 8.0.
            EnvRegistry.Register("Taxi-v4", typeof(TaxiEnv), rewardThreshold: 8, maxEpisodeSteps: 200);
        }

        /// <summary>
        ///     What the module initializers call: <see cref="Register"/>, unless <see cref="DisableRegistrationVariable"/>
        ///     disarms it. The app-side initializer the package compiles into a C# app is this one call (DR-17).
        /// </summary>
        /// <exception cref="Gym.Exceptions.RegistrationError">Thrown when the app registered a name both ways first.</exception>
        public static void RegisterUnlessDisabled() {
            if (!IsAutomaticRegistrationDisabled)
                Register();
        }

        /// <summary>This assembly's module initializer: registers the ids when the assembly is loaded before the app's initializer ran.</summary>
        [ModuleInitializer]
        [SuppressMessage("Usage", "CA2255:The 'ModuleInitializer' attribute should not be used in libraries",
            Justification = "Registering the built-in ids on load is the approved design (DR-12); the environment variable disarms it (DR-14).")]
        internal static void Initialize() => RegisterUnlessDisabled();
    }
}
