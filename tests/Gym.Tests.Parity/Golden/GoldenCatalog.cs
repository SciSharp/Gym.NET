using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Environments.Envs.Classic;
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
            // MountainCar-v0 driven to the flag by the momentum policy: the float64 state through the wall rule and
            // termination. No frames: the hill's joints are the open P7 gap.
            new GoldenCase {
                Suite = "classic_control",
                Name = "MountainCar-v0.momentum.seed0",
                EnvId = "MountainCar-v0",
                Seed = 0,
                Actions = new GoldenActions.Policy("mountaincar_momentum"),
                MaxSteps = 200,
                StateAttribute = "state",
                PlatformDependence = "MountainCar's cos",
                CreateDotNet = renderMode => new GymNetEnv(new MountainCarEnv(renderMode: renderMode)),
            },
            // MountainCarContinuous-v0 under seeded float32 forces: the NEP 50 float32 arithmetic of every step, checked
            // without Python, and Box sampling in the replay. No frames (P7).
            new GoldenCase {
                Suite = "classic_control",
                Name = "MountainCarContinuous-v0.sampled.seed42",
                EnvId = "MountainCarContinuous-v0",
                Seed = 42,
                Actions = new GoldenActions.Sampled(42),
                MaxSteps = 300,
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
