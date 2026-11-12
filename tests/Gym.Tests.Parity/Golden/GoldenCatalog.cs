using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Environments.Envs.Box2D;
using Gym.Environments.Envs.Classic;
using Gym.Environments.Envs.MuJoCo;
using Gym.Environments.Envs.ToyText;
using Gym.Tests.Parity.Pair;

namespace Gym.Tests.Parity.Golden {
    /// <summary>
    ///     Every golden case (plan §9's case catalog): the live golden cells run each one against Gymnasium and record it on
    ///     request, the replay cells check each one without Python. A phase that ports an env adds its cases here, and its
    ///     goldens are recorded from the live run that proves it.
    /// </summary>
    /// <remarks>
    ///     Cases are listed, not generated, so a case's key, and with it its golden's path, never changes by accident. Each
    ///     case builds its own .NET side from the render mode, exactly as its live counterpart in the env's parity tests
    ///     does.
    /// </remarks>
    public static class GoldenCatalog {
        /// <summary>The cases, by key.</summary>
        private static readonly IReadOnlyDictionary<string, GoldenCase> Cases = new[] {
            // CartPole-v1 balanced for the whole 500 steps: dynamics at full precision through the state attribute, the
            // generator after every call, and a frame every 100 steps.
            new GoldenCase {
                Suite = "classic_control",
                Name = "CartPole-v1.policy.seed0",
                EnvId = "CartPole-v1",
                Seed = 0,
                Actions = new GoldenActions.Policy("cartpole_balance"),
                MaxSteps = 500,
                FrameEvery = 100,
                StateAttribute = "state",
                PlatformDependence = "CartPole's cos/sin",
                CreateDotNet = renderMode => new GymNetEnv(new CartPoleEnv(renderMode: renderMode)),
            },
            // CartPole-v1 under seeded random actions until the pole falls: the replay samples Gym.NET's own action space,
            // so Discrete sampling is checked without Python too, and the termination step is part of the golden.
            new GoldenCase {
                Suite = "classic_control",
                Name = "CartPole-v1.sampled.seed42",
                EnvId = "CartPole-v1",
                Seed = 42,
                Actions = new GoldenActions.Sampled(42),
                MaxSteps = 500,
                FrameEvery = 5,
                StateAttribute = "state",
                PlatformDependence = "CartPole's cos/sin",
                CreateDotNet = renderMode => new GymNetEnv(new CartPoleEnv(renderMode: renderMode)),
            },
            // CartPole-v0 balanced through make's stack: the same class as v1, so the case records what only the
            // registration gives, TimeLimit's truncation at step 200, with a frame every 50 steps.
            new GoldenCase {
                Suite = "classic_control",
                Name = "CartPole-v0.policy.seed7",
                EnvId = "CartPole-v0",
                Seed = 7,
                Actions = new GoldenActions.Policy("cartpole_balance"),
                MaxSteps = 300,
                FrameEvery = 50,
                StateAttribute = "state",
                PlatformDependence = "CartPole's cos/sin",
                ViaMake = true,
                CreateDotNet = renderMode => new GymNetEnv(Gymnasium.Make("CartPole-v0", new Dictionary<string, object?> { ["render_mode"] = renderMode })),
            },
            // MountainCar-v0 driven to the flag by the momentum policy: the float64 state through the wall rule and
            // termination, with a frame every 20 steps (the hill's anti-aliased polyline is pygame's own since DR-37).
            new GoldenCase {
                Suite = "classic_control",
                Name = "MountainCar-v0.momentum.seed0",
                EnvId = "MountainCar-v0",
                Seed = 0,
                Actions = new GoldenActions.Policy("mountaincar_momentum"),
                MaxSteps = 200,
                FrameEvery = 20,
                StateAttribute = "state",
                PlatformDependence = "MountainCar's cos",
                CreateDotNet = renderMode => new GymNetEnv(new MountainCarEnv(renderMode: renderMode)),
            },
            // MountainCarContinuous-v0 under seeded float32 forces: the NEP 50 float32 arithmetic of every step, checked
            // without Python, and Box sampling in the replay, with a frame every 30 steps (the car's float32 position).
            new GoldenCase {
                Suite = "classic_control",
                Name = "MountainCarContinuous-v0.sampled.seed42",
                EnvId = "MountainCarContinuous-v0",
                Seed = 42,
                Actions = new GoldenActions.Sampled(42),
                MaxSteps = 300,
                FrameEvery = 30,
                StateAttribute = "state",
                PlatformDependence = "MountainCar's cos",
                CreateDotNet = renderMode => new GymNetEnv(new MountainCarEnv(renderMode: renderMode, continuous: true)),
            },
            // Pendulum-v1 under seeded float32 torques for a whole episode, with a frame every 20 steps: the rod and the
            // torque arrow (smoothscale, flip, alpha blit) at the sizes the episode takes.
            new GoldenCase {
                Suite = "classic_control",
                Name = "Pendulum-v1.sampled.seed42",
                EnvId = "Pendulum-v1",
                Seed = 42,
                Actions = new GoldenActions.Sampled(42),
                MaxSteps = 200,
                FrameEvery = 20,
                StateAttribute = "state",
                PlatformDependence = "Pendulum's sin/cos/pow",
                CreateDotNet = renderMode => new GymNetEnv(new PendulumEnv(renderMode: renderMode)),
            },
            // Acrobot-v1 swung up by the pumping policy through termination, a frame every 10 steps: the float32 reset state,
            // the Runge-Kutta dynamics in float64, and the links' drawing.
            new GoldenCase {
                Suite = "classic_control",
                Name = "Acrobot-v1.pump.seed0",
                EnvId = "Acrobot-v1",
                Seed = 0,
                Actions = new GoldenActions.Policy("acrobot_pump"),
                MaxSteps = 500,
                FrameEvery = 10,
                StateAttribute = "state",
                PlatformDependence = "Acrobot's sin/cos/pow",
                CreateDotNet = renderMode => new GymNetEnv(new AcrobotEnv(renderMode: renderMode)),
            },
            // The toy_text envs are table lookups and generator draws, with no libm, so their goldens are strict everywhere.
            // FrozenLake-v1 on slippery ice under seeded actions, a frame every 2 steps: the slippery table's draws, the
            // agent's moves and the hole or goal that ends the episode.
            new GoldenCase {
                Suite = "toy_text",
                Name = "FrozenLake-v1.sampled.seed42",
                EnvId = "FrozenLake-v1",
                Seed = 42,
                Actions = new GoldenActions.Sampled(42),
                MaxSteps = 100,
                FrameEvery = 2,
                StateAttribute = "s",
                CreateDotNet = renderMode => new GymNetEnv(new FrozenLakeEnv(renderMode: renderMode)),
            },
            // FrozenLake8x8-v1 under seeded actions, a frame every 5 steps: the 8x8 map's cell size.
            new GoldenCase {
                Suite = "toy_text",
                Name = "FrozenLake8x8-v1.sampled.seed7",
                EnvId = "FrozenLake8x8-v1",
                Seed = 7,
                Actions = new GoldenActions.Sampled(7),
                MaxSteps = 200,
                FrameEvery = 5,
                StateAttribute = "s",
                CreateDotNet = renderMode => new GymNetEnv(new FrozenLakeEnv(renderMode: renderMode, mapName: "8x8")),
            },
            // CliffWalking-v1 under seeded actions for 200 steps (it has no time limit and seldom reaches the goal), a frame
            // every 20: falls off the cliff back to the start, with -100.
            new GoldenCase {
                Suite = "toy_text",
                Name = "CliffWalking-v1.sampled.seed0",
                EnvId = "CliffWalking-v1",
                Seed = 0,
                Actions = new GoldenActions.Sampled(0),
                MaxSteps = 200,
                FrameEvery = 20,
                StateAttribute = "s",
                CreateDotNet = renderMode => new GymNetEnv(new CliffWalkingEnv(renderMode: renderMode)),
            },
            // CliffWalkingSlippery-v1 under seeded actions: the slippery table's three-way draws, no frames.
            new GoldenCase {
                Suite = "toy_text",
                Name = "CliffWalkingSlippery-v1.sampled.seed3",
                EnvId = "CliffWalkingSlippery-v1",
                Seed = 3,
                Actions = new GoldenActions.Sampled(3),
                MaxSteps = 200,
                StateAttribute = "s",
                CreateDotNet = renderMode => new GymNetEnv(new CliffWalkingEnv(renderMode: renderMode, isSlippery: true)),
            },
            // Taxi-v4 under seeded actions for 200 steps, a frame every 25: the infos' action masks, wrong pickups and
            // dropoffs, and the taxi sprite's orientation.
            new GoldenCase {
                Suite = "toy_text",
                Name = "Taxi-v4.sampled.seed11",
                EnvId = "Taxi-v4",
                Seed = 11,
                Actions = new GoldenActions.Sampled(11),
                MaxSteps = 200,
                FrameEvery = 25,
                StateAttribute = "s",
                CreateDotNet = renderMode => new GymNetEnv(new TaxiEnv(renderMode: renderMode)),
            },
            // Blackjack-v1 (registered sab=True) for one seeded game, a frame after every call: the dealt cards, the face-up
            // card's image and the table's digits. The observation is a tuple, so there is no state attribute.
            new GoldenCase {
                Suite = "toy_text",
                Name = "Blackjack-v1.sampled.seed5",
                EnvId = "Blackjack-v1",
                Seed = 5,
                Actions = new GoldenActions.Sampled(5),
                MaxSteps = 20,
                FrameEvery = 1,
                CreateDotNet = renderMode => new GymNetEnv(new BlackjackEnv(renderMode: renderMode, sab: true)),
            },
            // The box2d envs run on Gym.NET's managed Box2D, whose rotations call the C runtime's sinf/cosf, so their goldens
            // are strict on the reference platform only. LunarLander-v3 landed by Gymnasium's heuristic, a frame every 10
            // steps: the terrain draws, the push, the engines' dispersed impulses and particles, the legs' contacts, and the
            // +100 when the lander falls asleep.
            new GoldenCase {
                Suite = "box2d",
                Name = "LunarLander-v3.heuristic.seed0",
                EnvId = "LunarLander-v3",
                Seed = 0,
                Actions = new GoldenActions.Policy("lunar_lander_heuristic"),
                MaxSteps = 1000,
                FrameEvery = 10,
                PlatformDependence = "Box2D's sinf/cosf and LunarLander's sin/cos",
                CreateDotNet = renderMode => new GymNetEnv(new LunarLanderEnv(renderMode: renderMode)),
            },
            // LunarLanderContinuous-v3 (registered continuous=True) under the heuristic's throttles, a frame every 10 steps.
            new GoldenCase {
                Suite = "box2d",
                Name = "LunarLanderContinuous-v3.heuristic.seed3",
                EnvId = "LunarLanderContinuous-v3",
                Seed = 3,
                Actions = new GoldenActions.Policy("lunar_lander_heuristic"),
                MaxSteps = 1000,
                FrameEvery = 10,
                PlatformDependence = "Box2D's sinf/cosf and LunarLander's sin/cos",
                CreateDotNet = renderMode => new GymNetEnv(new LunarLanderEnv(renderMode: renderMode, continuous: true)),
            },
            // BipedalWalker-v3 walked by Gymnasium's gait for 800 steps, a frame every 40: the terrain and clouds a seed
            // draws, the motors, the lidar's closest hits, the float64 rewards of the gait's float64 actions, and the frame
            // scrolling with the walker.
            new GoldenCase {
                Suite = "box2d",
                Name = "BipedalWalker-v3.heuristic.seed0",
                EnvId = "BipedalWalker-v3",
                Seed = 0,
                Actions = new GoldenActions.Policy("bipedal_walker_heuristic"),
                MaxSteps = 800,
                FrameEvery = 40,
                PlatformDependence = "Box2D's sinf/cosf and BipedalWalker's sin/cos",
                CreateDotNet = renderMode => new GymNetEnv(new BipedalWalkerEnv(renderMode: renderMode)),
            },
            // BipedalWalkerHardcore-v3 (registered hardcore=True) under seeded float32 actions until the fall or 300 steps, a
            // frame every 30: the obstacles' draws and bodies, and the float32 rewards.
            new GoldenCase {
                Suite = "box2d",
                Name = "BipedalWalkerHardcore-v3.sampled.seed2",
                EnvId = "BipedalWalkerHardcore-v3",
                Seed = 2,
                Actions = new GoldenActions.Sampled(2),
                MaxSteps = 300,
                FrameEvery = 30,
                PlatformDependence = "Box2D's sinf/cosf and BipedalWalker's sin/cos",
                CreateDotNet = renderMode => new GymNetEnv(new BipedalWalkerEnv(renderMode: renderMode, hardcore: true)),
            },
            // CarRacing-v3 under seeded continuous actions for 60 steps, a frame every 30: the track a seed builds, the
            // opening second's zoom, tiles visited and paid, and the score's text (DR-48's embedded glyphs) in every
            // observation, each a whole 96×96 frame (about 27 KB a step, which keeps the case short).
            new GoldenCase {
                Suite = "box2d",
                Name = "CarRacing-v3.sampled.seed0",
                EnvId = "CarRacing-v3",
                Seed = 0,
                Actions = new GoldenActions.Sampled(0),
                MaxSteps = 60,
                FrameEvery = 30,
                PlatformDependence = "Box2D's sinf/cosf and the track's atan2/sin/cos",
                CreateDotNet = renderMode => new GymNetEnv(new CarRacingEnv(renderMode: renderMode)),
            },
            // CarRacing-v3's discrete mode (continuous=False) under seeded actions for 40 steps, a frame every 20: the five
            // fixed controls and another seed's track.
            new GoldenCase {
                Suite = "box2d",
                Name = "CarRacing-v3.discrete.sampled.seed4",
                EnvId = "CarRacing-v3",
                Kwargs = new Dictionary<string, object?> { ["continuous"] = false },
                Seed = 4,
                Actions = new GoldenActions.Sampled(4),
                MaxSteps = 40,
                FrameEvery = 20,
                PlatformDependence = "Box2D's sinf/cosf and the track's atan2/sin/cos",
                CreateDotNet = renderMode => new GymNetEnv(new CarRacingEnv(renderMode: renderMode, continuous: false)),
            },
            // The mujoco envs run on Gym.NET's managed MuJoCo 4.0.0 (DR-61, DR-64), whose joint rotations, gains and camera
            // math call the C runtime's sin/cos/tan/pow, so their goldens are strict on the reference platform only. Every
            // id that runs on MuJoCo 3 and later (Pusher-v4 refuses, as Gymnasium's does) under seeded float32 actions,
            // without frames (DR-62, DR-67): the reset's noise draws, the observation layout, the reward and every info
            // term with its NumPy type, and the termination of the ids that fall. The v5 ids use seed 0, the v4 ids seed 1;
            // the ids that never terminate stop at a step count that keeps each golden small (Humanoid's 348 and 376
            // float64 observation entries are about 3 KB a step).
            // Reacher-v4 for its whole 50-step limit: the goal drawn at the reset, and the reward priced before the step.
            new GoldenCase {
                Suite = "mujoco",
                Name = "Reacher-v4.sampled.seed1",
                EnvId = "Reacher-v4",
                Seed = 1,
                Actions = new GoldenActions.Sampled(1),
                MaxSteps = 50,
                PlatformDependence = MujocoPlatformDependence,
                CreateDotNet = renderMode => new GymNetEnv(new ReacherEnvV4(renderMode: renderMode)),
            },
            // Reacher-v5 for its whole 50-step limit: the reward priced after the step, its int control weight.
            new GoldenCase {
                Suite = "mujoco",
                Name = "Reacher-v5.sampled.seed0",
                EnvId = "Reacher-v5",
                Seed = 0,
                Actions = new GoldenActions.Sampled(0),
                MaxSteps = 50,
                PlatformDependence = MujocoPlatformDependence,
                CreateDotNet = renderMode => new GymNetEnv(new ReacherEnvV5(renderMode: renderMode)),
            },
            // Pusher-v5 for its whole 100-step limit: the cylinder redrawn at the reset until it lies 0.17 from the fixed goal
            // (and written into init_qpos itself, as Gymnasium's reset does), the arm's distances to it and to the goal.
            new GoldenCase {
                Suite = "mujoco",
                Name = "Pusher-v5.sampled.seed0",
                EnvId = "Pusher-v5",
                Seed = 0,
                Actions = new GoldenActions.Sampled(0),
                MaxSteps = 100,
                PlatformDependence = MujocoPlatformDependence,
                CreateDotNet = renderMode => new GymNetEnv(new PusherEnvV5(renderMode: renderMode)),
            },
            // InvertedPendulum-v4 until the pole falls.
            new GoldenCase {
                Suite = "mujoco",
                Name = "InvertedPendulum-v4.sampled.seed1",
                EnvId = "InvertedPendulum-v4",
                Seed = 1,
                Actions = new GoldenActions.Sampled(1),
                MaxSteps = 1000,
                PlatformDependence = MujocoPlatformDependence,
                CreateDotNet = renderMode => new GymNetEnv(new InvertedPendulumEnvV4(renderMode: renderMode)),
            },
            // InvertedPendulum-v5 until the pole falls: v5's reward of 0 at the terminating step.
            new GoldenCase {
                Suite = "mujoco",
                Name = "InvertedPendulum-v5.sampled.seed0",
                EnvId = "InvertedPendulum-v5",
                Seed = 0,
                Actions = new GoldenActions.Sampled(0),
                MaxSteps = 1000,
                PlatformDependence = MujocoPlatformDependence,
                CreateDotNet = renderMode => new GymNetEnv(new InvertedPendulumEnvV5(renderMode: renderMode)),
            },
            // InvertedDoublePendulum-v4 until the tip drops: the reward's distance and velocity penalties, and the three
            // constraint forces in its 11-entry observation.
            new GoldenCase {
                Suite = "mujoco",
                Name = "InvertedDoublePendulum-v4.sampled.seed1",
                EnvId = "InvertedDoublePendulum-v4",
                Seed = 1,
                Actions = new GoldenActions.Sampled(1),
                MaxSteps = 1000,
                PlatformDependence = MujocoPlatformDependence,
                CreateDotNet = renderMode => new GymNetEnv(new InvertedDoublePendulumEnvV4(renderMode: renderMode)),
            },
            // InvertedDoublePendulum-v5 until the tip drops: v5's info terms (survival, distance and velocity penalties) and
            // its 9-entry observation.
            new GoldenCase {
                Suite = "mujoco",
                Name = "InvertedDoublePendulum-v5.sampled.seed0",
                EnvId = "InvertedDoublePendulum-v5",
                Seed = 0,
                Actions = new GoldenActions.Sampled(0),
                MaxSteps = 1000,
                PlatformDependence = MujocoPlatformDependence,
                CreateDotNet = renderMode => new GymNetEnv(new InvertedDoublePendulumEnvV5(renderMode: renderMode)),
            },
            // HalfCheetah-v4 for 150 steps (it never terminates): the forward reward and the control cost.
            new GoldenCase {
                Suite = "mujoco",
                Name = "HalfCheetah-v4.sampled.seed1",
                EnvId = "HalfCheetah-v4",
                Seed = 1,
                Actions = new GoldenActions.Sampled(1),
                MaxSteps = 150,
                PlatformDependence = MujocoPlatformDependence,
                CreateDotNet = renderMode => new GymNetEnv(new HalfCheetahEnvV4(renderMode: renderMode)),
            },
            // HalfCheetah-v5 for 150 steps: the uniform and standard-normal reset noise, the ground contacts.
            new GoldenCase {
                Suite = "mujoco",
                Name = "HalfCheetah-v5.sampled.seed0",
                EnvId = "HalfCheetah-v5",
                Seed = 0,
                Actions = new GoldenActions.Sampled(0),
                MaxSteps = 150,
                PlatformDependence = MujocoPlatformDependence,
                CreateDotNet = renderMode => new GymNetEnv(new HalfCheetahEnvV5(renderMode: renderMode)),
            },
            // Hopper-v4 until it falls: the health ranges that end the episode.
            new GoldenCase {
                Suite = "mujoco",
                Name = "Hopper-v4.sampled.seed1",
                EnvId = "Hopper-v4",
                Seed = 1,
                Actions = new GoldenActions.Sampled(1),
                MaxSteps = 1000,
                PlatformDependence = MujocoPlatformDependence,
                CreateDotNet = renderMode => new GymNetEnv(new HopperEnvV4(renderMode: renderMode)),
            },
            // Hopper-v5 until it falls: the survival reward as a Python float, the info's positions and velocity.
            new GoldenCase {
                Suite = "mujoco",
                Name = "Hopper-v5.sampled.seed0",
                EnvId = "Hopper-v5",
                Seed = 0,
                Actions = new GoldenActions.Sampled(0),
                MaxSteps = 1000,
                PlatformDependence = MujocoPlatformDependence,
                CreateDotNet = renderMode => new GymNetEnv(new HopperEnvV5(renderMode: renderMode)),
            },
            // Swimmer-v4 for 150 steps (it never terminates): the viscous fluid's drag.
            new GoldenCase {
                Suite = "mujoco",
                Name = "Swimmer-v4.sampled.seed1",
                EnvId = "Swimmer-v4",
                Seed = 1,
                Actions = new GoldenActions.Sampled(1),
                MaxSteps = 150,
                PlatformDependence = MujocoPlatformDependence,
                CreateDotNet = renderMode => new GymNetEnv(new SwimmerEnvV4(renderMode: renderMode)),
            },
            // Swimmer-v5 for 150 steps.
            new GoldenCase {
                Suite = "mujoco",
                Name = "Swimmer-v5.sampled.seed0",
                EnvId = "Swimmer-v5",
                Seed = 0,
                Actions = new GoldenActions.Sampled(0),
                MaxSteps = 150,
                PlatformDependence = MujocoPlatformDependence,
                CreateDotNet = renderMode => new GymNetEnv(new SwimmerEnvV5(renderMode: renderMode)),
            },
            // Walker2d-v4 until it falls.
            new GoldenCase {
                Suite = "mujoco",
                Name = "Walker2d-v4.sampled.seed1",
                EnvId = "Walker2d-v4",
                Seed = 1,
                Actions = new GoldenActions.Sampled(1),
                MaxSteps = 1000,
                PlatformDependence = MujocoPlatformDependence,
                CreateDotNet = renderMode => new GymNetEnv(new Walker2dEnvV4(renderMode: renderMode)),
            },
            // Walker2d-v5 until it falls: v5's own model (walker2d_v5.xml: both feet at friction 1.9, where v4's right foot
            // has 0.9) and the survival reward as np.float64.
            new GoldenCase {
                Suite = "mujoco",
                Name = "Walker2d-v5.sampled.seed0",
                EnvId = "Walker2d-v5",
                Seed = 0,
                Actions = new GoldenActions.Sampled(0),
                MaxSteps = 1000,
                PlatformDependence = MujocoPlatformDependence,
                CreateDotNet = renderMode => new GymNetEnv(new Walker2dEnvV5(renderMode: renderMode)),
            },
            // Ant-v4 for 100 steps: the 27-entry observation without contact forces, the health check.
            new GoldenCase {
                Suite = "mujoco",
                Name = "Ant-v4.sampled.seed1",
                EnvId = "Ant-v4",
                Seed = 1,
                Actions = new GoldenActions.Sampled(1),
                MaxSteps = 100,
                PlatformDependence = MujocoPlatformDependence,
                CreateDotNet = renderMode => new GymNetEnv(new AntEnvV4(renderMode: renderMode)),
            },
            // Ant-v5 for at most 60 steps (seed 0's torso leaves the healthy height at step 46): the clipped contact forces
            // in its 105-entry observation, the contact cost and the termination.
            new GoldenCase {
                Suite = "mujoco",
                Name = "Ant-v5.sampled.seed0",
                EnvId = "Ant-v5",
                Seed = 0,
                Actions = new GoldenActions.Sampled(0),
                MaxSteps = 60,
                PlatformDependence = MujocoPlatformDependence,
                CreateDotNet = renderMode => new GymNetEnv(new AntEnvV5(renderMode: renderMode)),
            },
            // Humanoid-v4 until it falls: the 376-entry observation (inertias, velocities, actuator and external forces)
            // and the mass centre's velocity.
            new GoldenCase {
                Suite = "mujoco",
                Name = "Humanoid-v4.sampled.seed1",
                EnvId = "Humanoid-v4",
                Seed = 1,
                Actions = new GoldenActions.Sampled(1),
                MaxSteps = 1000,
                PlatformDependence = MujocoPlatformDependence,
                CreateDotNet = renderMode => new GymNetEnv(new HumanoidEnvV4(renderMode: renderMode)),
            },
            // Humanoid-v5 until it falls: the 348-entry observation (the world body's rows and the free joint's actuator
            // forces left out) and the contact cost.
            new GoldenCase {
                Suite = "mujoco",
                Name = "Humanoid-v5.sampled.seed0",
                EnvId = "Humanoid-v5",
                Seed = 0,
                Actions = new GoldenActions.Sampled(0),
                MaxSteps = 1000,
                PlatformDependence = MujocoPlatformDependence,
                CreateDotNet = renderMode => new GymNetEnv(new HumanoidEnvV5(renderMode: renderMode)),
            },
            // HumanoidStandup-v4 for 40 steps (it never terminates): the height reward and the quadratic costs.
            new GoldenCase {
                Suite = "mujoco",
                Name = "HumanoidStandup-v4.sampled.seed1",
                EnvId = "HumanoidStandup-v4",
                Seed = 1,
                Actions = new GoldenActions.Sampled(1),
                MaxSteps = 40,
                PlatformDependence = MujocoPlatformDependence,
                CreateDotNet = renderMode => new GymNetEnv(new HumanoidStandupEnvV4(renderMode: renderMode)),
            },
            // HumanoidStandup-v5 for 40 steps: the impact cost clipped to its range.
            new GoldenCase {
                Suite = "mujoco",
                Name = "HumanoidStandup-v5.sampled.seed0",
                EnvId = "HumanoidStandup-v5",
                Seed = 0,
                Actions = new GoldenActions.Sampled(0),
                MaxSteps = 40,
                PlatformDependence = MujocoPlatformDependence,
                CreateDotNet = renderMode => new GymNetEnv(new HumanoidStandupEnvV5(renderMode: renderMode)),
            },
        }.ToDictionary(c => c.Key, StringComparer.Ordinal);

        /// <summary>
        ///     What makes every mujoco golden depend on the platform: Gym.NET's engine computes MuJoCo 4.0.0's win-x64 bits,
        ///     whose <c>sin</c>/<c>cos</c>/<c>tan</c>/<c>pow</c> are the C runtime's, so another libm can round a step one ulp
        ///     apart and the trajectory drifts from there. Its AVX kernels' summation order is reproduced in scalar code and
        ///     does not depend on the machine.
        /// </summary>
        private const string MujocoPlatformDependence = "MuJoCo's sin/cos/tan/pow through the C runtime";

        /// <summary>Every case's key, as MSTest data rows: one row per case, named by its key in test results.</summary>
        public static IEnumerable<object[]> Keys => Cases.Keys.Order(StringComparer.Ordinal).Select(key => new object[] { key });

        /// <summary>The case with a key.</summary>
        /// <param name="key">The key, <c>suite/name</c>.</param>
        /// <returns>The case.</returns>
        /// <exception cref="KeyNotFoundException">Thrown for a key the catalog does not list.</exception>
        public static GoldenCase Get(string key) {
            if (!Cases.TryGetValue(key, out GoldenCase? found))
                throw new KeyNotFoundException($"The golden catalog has no case '{key}'.");
            return found;
        }
    }
}
