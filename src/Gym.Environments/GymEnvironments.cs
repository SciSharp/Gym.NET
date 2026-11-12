#nullable enable
using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Threading;
using Gym.Environments.Envs.Box2D;
using Gym.Environments.Envs.Classic;
using Gym.Environments.Envs.MuJoCo;
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
    ///     Ids register in Gymnasium's order as their phases port them: classic_control's (CartPole-v0/v1, MountainCar-v0,
    ///     MountainCarContinuous-v0, Pendulum-v1, Acrobot-v1), box2d's (LunarLander-v3, LunarLanderContinuous-v3,
    ///     BipedalWalker-v3, BipedalWalkerHardcore-v3, CarRacing-v3), toy_text's (Blackjack-v1,
    ///     FrozenLake-v1, FrozenLake8x8-v1, CliffWalking-v1, CliffWalkingSlippery-v1, Taxi-v4) and mujoco's 22 (Reacher,
    ///     Pusher, InvertedPendulum, InvertedDoublePendulum, HalfCheetah, Hopper, Swimmer, Walker2d, Ant, Humanoid and
    ///     HumanoidStandup, each v4 and v5; Pusher-v4 refuses at construction, as Gymnasium's does on MuJoCo 3 and later).
    ///     CartPole's two register <c>CartPoleVectorEnv</c> as their vectorized entry point.
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
            // Both CartPoles vectorize natively: MakeVec builds CartPoleVectorEnv unless another mode is asked for.
            EnvRegistry.Register("CartPole-v0", typeof(CartPoleEnv), maxEpisodeSteps: 200, rewardThreshold: 195.0, vectorEntryPoint: typeof(CartPoleVectorEnv));
            EnvRegistry.Register("CartPole-v1", typeof(CartPoleEnv), maxEpisodeSteps: 500, rewardThreshold: 475.0, vectorEntryPoint: typeof(CartPoleVectorEnv));
            EnvRegistry.Register("MountainCar-v0", typeof(MountainCarEnv), maxEpisodeSteps: 200, rewardThreshold: -110.0);
            // One class for both MountainCars (DR-32): the continuous id carries continuous=True in its kwargs, where
            // Gymnasium's entry point is a class of its own and its kwargs are empty.
            EnvRegistry.Register("MountainCarContinuous-v0", typeof(MountainCarEnv), maxEpisodeSteps: 999, rewardThreshold: 90.0,
                kwargs: new MountainCarEnvOptions { Continuous = true });
            EnvRegistry.Register("Pendulum-v1", typeof(PendulumEnv), maxEpisodeSteps: 200);
            EnvRegistry.Register("Acrobot-v1", typeof(AcrobotEnv), rewardThreshold: -100.0, maxEpisodeSteps: 500);

            // Box2d
            // ----------------------------------------

            EnvRegistry.Register("LunarLander-v3", typeof(LunarLanderEnv), maxEpisodeSteps: 1000, rewardThreshold: 200);
            // The same class with continuous=True, as Gymnasium registers it (no DR-32 deviation here).
            EnvRegistry.Register("LunarLanderContinuous-v3", typeof(LunarLanderEnv), kwargs: new LunarLanderEnvOptions { Continuous = true },
                maxEpisodeSteps: 1000, rewardThreshold: 200);

            EnvRegistry.Register("BipedalWalker-v3", typeof(BipedalWalkerEnv), maxEpisodeSteps: 1600, rewardThreshold: 300);
            // The same class with hardcore=True, as Gymnasium registers it.
            EnvRegistry.Register("BipedalWalkerHardcore-v3", typeof(BipedalWalkerEnv), kwargs: new BipedalWalkerEnvOptions { Hardcore = true },
                maxEpisodeSteps: 2000, rewardThreshold: 300);

            EnvRegistry.Register("CarRacing-v3", typeof(CarRacingEnv), maxEpisodeSteps: 1000, rewardThreshold: 900);

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

            // MuJoCo
            // ----------------------------------------
            // Gym.NET's own engine, bit for bit with MuJoCo 4.0.0 (DR-61, DR-64), on Gymnasium's embedded models (DR-63).
            // Gymnasium's v2/v3 ids, which only point at gymnasium-robotics, are not registered.

            // manipulation
            EnvRegistry.Register("Reacher-v4", typeof(ReacherEnvV4), maxEpisodeSteps: 50, rewardThreshold: -3.75);
            EnvRegistry.Register("Reacher-v5", typeof(ReacherEnvV5), maxEpisodeSteps: 50, rewardThreshold: -3.75);
            // Pusher-v4 registers as in Gymnasium, and refuses at construction as Gymnasium does on MuJoCo 3 and later.
            EnvRegistry.Register("Pusher-v4", typeof(PusherEnvV4), maxEpisodeSteps: 100, rewardThreshold: 0.0);
            EnvRegistry.Register("Pusher-v5", typeof(PusherEnvV5), maxEpisodeSteps: 100, rewardThreshold: 0.0);

            // balance
            EnvRegistry.Register("InvertedPendulum-v4", typeof(InvertedPendulumEnvV4), maxEpisodeSteps: 1000, rewardThreshold: 950.0);
            EnvRegistry.Register("InvertedPendulum-v5", typeof(InvertedPendulumEnvV5), maxEpisodeSteps: 1000, rewardThreshold: 950.0);
            EnvRegistry.Register("InvertedDoublePendulum-v4", typeof(InvertedDoublePendulumEnvV4), maxEpisodeSteps: 1000, rewardThreshold: 9100.0);
            EnvRegistry.Register("InvertedDoublePendulum-v5", typeof(InvertedDoublePendulumEnvV5), maxEpisodeSteps: 1000, rewardThreshold: 9100.0);

            // runners
            EnvRegistry.Register("HalfCheetah-v4", typeof(HalfCheetahEnvV4), maxEpisodeSteps: 1000, rewardThreshold: 4800.0);
            EnvRegistry.Register("HalfCheetah-v5", typeof(HalfCheetahEnvV5), maxEpisodeSteps: 1000, rewardThreshold: 4800.0);
            EnvRegistry.Register("Hopper-v4", typeof(HopperEnvV4), maxEpisodeSteps: 1000, rewardThreshold: 3800.0);
            EnvRegistry.Register("Hopper-v5", typeof(HopperEnvV5), maxEpisodeSteps: 1000, rewardThreshold: 3800.0);
            EnvRegistry.Register("Swimmer-v4", typeof(SwimmerEnvV4), maxEpisodeSteps: 1000, rewardThreshold: 360.0);
            EnvRegistry.Register("Swimmer-v5", typeof(SwimmerEnvV5), maxEpisodeSteps: 1000, rewardThreshold: 360.0);
            EnvRegistry.Register("Walker2d-v4", typeof(Walker2dEnvV4), maxEpisodeSteps: 1000);
            EnvRegistry.Register("Walker2d-v5", typeof(Walker2dEnvV5), maxEpisodeSteps: 1000);
            EnvRegistry.Register("Ant-v4", typeof(AntEnvV4), maxEpisodeSteps: 1000, rewardThreshold: 6000.0);
            EnvRegistry.Register("Ant-v5", typeof(AntEnvV5), maxEpisodeSteps: 1000, rewardThreshold: 6000.0);
            EnvRegistry.Register("Humanoid-v4", typeof(HumanoidEnvV4), maxEpisodeSteps: 1000);
            EnvRegistry.Register("Humanoid-v5", typeof(HumanoidEnvV5), maxEpisodeSteps: 1000);
            EnvRegistry.Register("HumanoidStandup-v4", typeof(HumanoidStandupEnvV4), maxEpisodeSteps: 1000);
            EnvRegistry.Register("HumanoidStandup-v5", typeof(HumanoidStandupEnvV5), maxEpisodeSteps: 1000);
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
