using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Environments.Envs.Box2D;
using Gym.Environments.Envs.Classic;
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
        }.ToDictionary(c => c.Key, StringComparer.Ordinal);

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
