#nullable enable
namespace Gym {
    public static partial class Gymnasium {
        /// <summary>
        ///     The ids Gym.NET.Environments registers, as constants (DR-88): <c>Gymnasium.Make(Gymnasium.Envs.CartPoleV1)</c>,
        ///     or <c>Make(Envs.CartPoleV1)</c> with <c>using static Gym.Gymnasium;</c>, so the compiler catches a mistyped
        ///     id and the IDE lists the built-in envs.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     Each constant is the id's exact string, Gymnasium's own (<c>"CartPole-v1"</c>); the strings keep working
        ///     everywhere. The ids register when Gym.NET.Environments loads (DR-12, DR-17): a constant names an id, it does
        ///     not register one, so an app without that package gets Gymnasium's <c>NameNotFound</c> for it as for the string.
        ///     A test keeps this list equal to the registered built-in ids.
        ///     </para>
        ///     <para>
        ///     Footgun: inside a <c>Gym.*</c> namespace the simple name <c>Envs</c> is the <c>Gym.Envs</c> namespace, so write
        ///     <c>Gymnasium.Envs.CartPoleV1</c> there (and <c>global::Gym.Gymnasium.Envs.CartPoleV1</c> where a
        ///     <c>Gymnasium</c> namespace hides the class, as in Gym.NET's own tests).
        ///     </para>
        /// </remarks>
        public static class Envs {
            /// <summary><c>CartPole-v0</c>: <c>CartPoleEnv</c>, the pole balanced for up to 200 steps, solved at a return of 195.</summary>
            public const string CartPoleV0 = "CartPole-v0";

            /// <summary><c>CartPole-v1</c>: <c>CartPoleEnv</c>, the pole balanced for up to 500 steps, solved at a return of 475.</summary>
            public const string CartPoleV1 = "CartPole-v1";

            /// <summary><c>MountainCar-v0</c>: <c>MountainCarEnv</c> with three discrete pushes, 200 steps, solved at -110.</summary>
            public const string MountainCarV0 = "MountainCar-v0";

            /// <summary><c>MountainCarContinuous-v0</c>: <c>MountainCarEnv</c> with a continuous force (<c>continuous=True</c>), 999 steps, solved at 90.</summary>
            public const string MountainCarContinuousV0 = "MountainCarContinuous-v0";

            /// <summary><c>Pendulum-v1</c>: <c>PendulumEnv</c>, swinging the pendulum up, 200 steps, no threshold.</summary>
            public const string PendulumV1 = "Pendulum-v1";

            /// <summary><c>Acrobot-v1</c>: <c>AcrobotEnv</c>, swinging the two links up, 500 steps, solved at -100.</summary>
            public const string AcrobotV1 = "Acrobot-v1";

            /// <summary><c>Blackjack-v1</c>: <c>BlackjackEnv</c> with Sutton and Barto's rules (<c>sab=True</c>, <c>natural=False</c>), no step limit.</summary>
            public const string BlackjackV1 = "Blackjack-v1";

            /// <summary><c>FrozenLake-v1</c>: <c>FrozenLakeEnv</c> on the 4x4 map, 100 steps, solved at 0.70.</summary>
            public const string FrozenLakeV1 = "FrozenLake-v1";

            /// <summary><c>FrozenLake8x8-v1</c>: <c>FrozenLakeEnv</c> on the 8x8 map, 200 steps, solved at 0.85.</summary>
            public const string FrozenLake8x8V1 = "FrozenLake8x8-v1";

            /// <summary><c>CliffWalking-v1</c>: <c>CliffWalkingEnv</c>, no step limit.</summary>
            public const string CliffWalkingV1 = "CliffWalking-v1";

            /// <summary><c>CliffWalkingSlippery-v1</c>: <c>CliffWalkingEnv</c> with slippery moves (<c>is_slippery=True</c>), no step limit.</summary>
            public const string CliffWalkingSlipperyV1 = "CliffWalkingSlippery-v1";

            /// <summary><c>Taxi-v4</c>: <c>TaxiEnv</c>, 200 steps, solved at 8.</summary>
            public const string TaxiV4 = "Taxi-v4";

            /// <summary><c>LunarLander-v3</c>: <c>LunarLanderEnv</c> with four discrete engine actions, 1000 steps, solved at 200.</summary>
            public const string LunarLanderV3 = "LunarLander-v3";

            /// <summary><c>LunarLanderContinuous-v3</c>: <c>LunarLanderEnv</c> with continuous throttles (<c>continuous=True</c>), 1000 steps, solved at 200.</summary>
            public const string LunarLanderContinuousV3 = "LunarLanderContinuous-v3";

            /// <summary><c>BipedalWalker-v3</c>: <c>BipedalWalkerEnv</c>, 1600 steps, solved at 300.</summary>
            public const string BipedalWalkerV3 = "BipedalWalker-v3";

            /// <summary><c>BipedalWalkerHardcore-v3</c>: <c>BipedalWalkerEnv</c> with obstacles (<c>hardcore=True</c>), 2000 steps, solved at 300.</summary>
            public const string BipedalWalkerHardcoreV3 = "BipedalWalkerHardcore-v3";

            /// <summary><c>CarRacing-v3</c>: <c>CarRacingEnv</c>, driving a random track from pixels, 1000 steps, solved at 900.</summary>
            public const string CarRacingV3 = "CarRacing-v3";

            /// <summary><c>Reacher-v4</c>: <c>ReacherEnvV4</c>, a two-joint arm reaching a random goal, 50 steps, solved at -3.75.</summary>
            public const string ReacherV4 = "Reacher-v4";

            /// <summary><c>Reacher-v5</c>: <c>ReacherEnvV5</c>, 50 steps, solved at -3.75.</summary>
            public const string ReacherV5 = "Reacher-v5";

            /// <summary>
            ///     <c>Pusher-v4</c>: <c>PusherEnvV4</c>, registered as Gymnasium registers it but refusing at construction, as
            ///     Gymnasium's does on MuJoCo 3 and later; use <see cref="PusherV5"/>.
            /// </summary>
            public const string PusherV4 = "Pusher-v4";

            /// <summary><c>Pusher-v5</c>: <c>PusherEnvV5</c>, an arm pushing a cylinder to a goal, 100 steps, solved at 0.</summary>
            public const string PusherV5 = "Pusher-v5";

            /// <summary><c>InvertedPendulum-v4</c>: <c>InvertedPendulumEnvV4</c>, 1000 steps, solved at 950.</summary>
            public const string InvertedPendulumV4 = "InvertedPendulum-v4";

            /// <summary><c>InvertedPendulum-v5</c>: <c>InvertedPendulumEnvV5</c>, 1000 steps, solved at 950.</summary>
            public const string InvertedPendulumV5 = "InvertedPendulum-v5";

            /// <summary><c>InvertedDoublePendulum-v4</c>: <c>InvertedDoublePendulumEnvV4</c>, 1000 steps, solved at 9100.</summary>
            public const string InvertedDoublePendulumV4 = "InvertedDoublePendulum-v4";

            /// <summary><c>InvertedDoublePendulum-v5</c>: <c>InvertedDoublePendulumEnvV5</c>, 1000 steps, solved at 9100.</summary>
            public const string InvertedDoublePendulumV5 = "InvertedDoublePendulum-v5";

            /// <summary><c>HalfCheetah-v4</c>: <c>HalfCheetahEnvV4</c>, running forward, 1000 steps, solved at 4800.</summary>
            public const string HalfCheetahV4 = "HalfCheetah-v4";

            /// <summary><c>HalfCheetah-v5</c>: <c>HalfCheetahEnvV5</c>, 1000 steps, solved at 4800.</summary>
            public const string HalfCheetahV5 = "HalfCheetah-v5";

            /// <summary><c>Hopper-v4</c>: <c>HopperEnvV4</c>, hopping forward without falling, 1000 steps, solved at 3800.</summary>
            public const string HopperV4 = "Hopper-v4";

            /// <summary><c>Hopper-v5</c>: <c>HopperEnvV5</c>, 1000 steps, solved at 3800.</summary>
            public const string HopperV5 = "Hopper-v5";

            /// <summary><c>Swimmer-v4</c>: <c>SwimmerEnvV4</c>, swimming forward in a viscous fluid, 1000 steps, solved at 360.</summary>
            public const string SwimmerV4 = "Swimmer-v4";

            /// <summary><c>Swimmer-v5</c>: <c>SwimmerEnvV5</c>, 1000 steps, solved at 360.</summary>
            public const string SwimmerV5 = "Swimmer-v5";

            /// <summary><c>Walker2d-v4</c>: <c>Walker2dEnvV4</c>, walking forward on two legs, 1000 steps, no threshold.</summary>
            public const string Walker2dV4 = "Walker2d-v4";

            /// <summary><c>Walker2d-v5</c>: <c>Walker2dEnvV5</c>, 1000 steps, no threshold.</summary>
            public const string Walker2dV5 = "Walker2d-v5";

            /// <summary><c>Ant-v4</c>: <c>AntEnvV4</c>, a four-legged walker, 1000 steps, solved at 6000.</summary>
            public const string AntV4 = "Ant-v4";

            /// <summary><c>Ant-v5</c>: <c>AntEnvV5</c>, 1000 steps, solved at 6000.</summary>
            public const string AntV5 = "Ant-v5";

            /// <summary><c>Humanoid-v4</c>: <c>HumanoidEnvV4</c>, walking forward without falling, 1000 steps, no threshold.</summary>
            public const string HumanoidV4 = "Humanoid-v4";

            /// <summary><c>Humanoid-v5</c>: <c>HumanoidEnvV5</c>, 1000 steps, no threshold.</summary>
            public const string HumanoidV5 = "Humanoid-v5";

            /// <summary><c>HumanoidStandup-v4</c>: <c>HumanoidStandupEnvV4</c>, standing up from the ground, 1000 steps, no threshold.</summary>
            public const string HumanoidStandupV4 = "HumanoidStandup-v4";

            /// <summary><c>HumanoidStandup-v5</c>: <c>HumanoidStandupEnvV5</c>, 1000 steps, no threshold.</summary>
            public const string HumanoidStandupV5 = "HumanoidStandup-v5";
        }
    }
}
