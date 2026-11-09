# Gym.NET

[![NuGet](https://img.shields.io/nuget/v/Gym.NET.Environments?label=Gym.NET.Environments)](https://www.nuget.org/packages/Gym.NET.Environments)
[![NuGet downloads](https://img.shields.io/nuget/dt/Gym.NET)](https://www.nuget.org/packages/Gym.NET)

Reinforcement-learning environments for .NET with [Farama Gymnasium](https://github.com/Farama-Foundation/Gymnasium)'s
API and behavior. Given the same seed and actions, Gym.NET's environments return the same observations, rewards,
`terminated`/`truncated` flags and infos as Gymnasium's, bit for bit, and render the same frames, pixel for pixel. A test
suite that runs Gymnasium itself inside the test process checks this, call by call.

Gym.NET is part of the [SciSharp STACK](https://scisharpstack.org) and builds on [NumSharp](https://github.com/SciSharp/NumSharp)
for arrays and NumPy-exact random generators.

## Installation

```sh
dotnet add package Gym.NET.Environments          # the environments, with Gym.NET's core (Gym.NET)
dotnet add package Gym.NET.Rendering.Avalonia    # optional: a window for render_mode "human" (Windows, Linux, macOS)
dotnet add package Gym.NET.Rendering.WinForm     # optional: the same on Windows Forms
```

Gym.NET 2.0 targets .NET 8 and .NET 10.

## Example

```csharp
using System;
using Gym;
using Gym.Envs;
using NumSharp;

// make builds the registered env inside Gymnasium's wrappers: TimeLimit(500), OrderEnforcing, PassiveEnvChecker.
using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("CartPole-v1");
env.ActionSpace.Seed(42);
NDArray observation = env.Reset(seed: 42).Observation;
double episodeReturn = 0;
while (true) {
    using NDArray action = env.ActionSpace.Sample();   // 0 (push left) or 1 (push right)
    var (next, reward, terminated, truncated, info) = env.Step(action);
    observation.Dispose();                            // the arrays an env returns belong to the caller
    observation = next;
    episodeReturn += reward;
    if (terminated || truncated)
        break;
}

observation.Dispose();
Console.WriteLine($"Episode return: {episodeReturn}");
```

`Reset(seed)` returns `(observation, info)`. `Step(action)` returns `(observation, reward, terminated, truncated, info)`:
`terminated` means the task ended (the pole fell), and `truncated` means a limit cut the episode short (`TimeLimit`'s 500
steps). The render mode is chosen when the env is built: `Gymnasium.Make("CartPole-v1", new { render_mode = "rgb_array" })`.

[examples/CartPole](https://github.com/SciSharp/Gym.NET/tree/master/examples/CartPole) trains a policy that balances
the pole, playing 100 carts at once through `Gymnasium.MakeVec`.

## What is in 2.0

| Area | Contents |
|---|---|
| Core | `Env<TObs, TAct>`, `IEnv`, `Wrapper` and the observation, action and reward wrapper bases, `StepResult`/`ResetResult`, `Info`, `Metadata`, `RenderResult`, every Gymnasium error |
| Spaces | `Box`, `Discrete`, `MultiBinary`, `MultiDiscrete`, `Tuple`, `Dict`, `Text`, `Sequence`, `Graph`, `OneOf`, and `SpaceUtils` (flatten, unflatten, flatdim) |
| Registry | `Gymnasium.Make`, `Gymnasium.MakeVec`, `Gymnasium.Register`, `Gymnasium.Spec`, `Gymnasium.PprintRegistry`; constructor arguments by Python name, from an anonymous object, a dictionary or a generated options record |
| Wrappers | `TimeLimit`, `OrderEnforcing`, `PassiveEnvChecker`, `Autoreset`, `RecordEpisodeStatistics`, `RenderCollection`, `NormalizeObservation`, `NormalizeReward`, `TimeAwareObservation`, `FlattenObservation` |
| Vector envs | `VectorEnv<TObs, TAct>`, `SyncVectorEnv` in every autoreset mode, `CartPoleVectorEnv`, the vector wrapper bases |
| Checkers | `EnvChecker.CheckEnv`, `PassiveChecker`, `EnvMatch` |
| Environments | classic_control: CartPole-v0/v1, MountainCar-v0, MountainCarContinuous-v0, Pendulum-v1, Acrobot-v1. toy_text: FrozenLake-v1, FrozenLake8x8-v1, CliffWalking-v1, CliffWalkingSlippery-v1, Taxi-v4, Blackjack-v1 |
| Rendering | `rgb_array` and `ansi` frames equal to Gymnasium's; `human` mode in an Avalonia or Windows Forms window |

Box2D's environments (LunarLander, BipedalWalker, CarRacing) are in progress. MuJoCo's are postponed beyond 2.0. The
release notes list what 2.0 leaves out and every known difference from Gymnasium:
[docs/releases/RELEASE_2.0.0.md](https://github.com/SciSharp/Gym.NET/blob/master/docs/releases/RELEASE_2.0.0.md).

## Documentation

- [Getting started](https://github.com/SciSharp/Gym.NET/blob/master/docs/guides/getting_started.md): making envs,
  seeding, spaces, rendering, wrappers, vector envs.
- [Creating custom environments](https://github.com/SciSharp/Gym.NET/blob/master/docs/guides/creating_custom_environments.md):
  an `Env<TObs, TAct>`, its registration, and checking it.
- [Migrating from 0.2 to 2.0](https://github.com/SciSharp/Gym.NET/blob/master/docs/guides/migrating_to_2.0.md): the old
  calls and their replacements, and `Gym.Legacy` for code that cannot move yet.
- [Architecture](https://github.com/SciSharp/Gym.NET/blob/master/docs/architecture.md): the packages, and how Gym.NET
  is held to Gymnasium's behavior.
- [Development and testing](https://github.com/SciSharp/Gym.NET/blob/master/docs/guides/development_and_testing.md):
  building, the unit suite, and the live parity suite.

## A note on SixLabors.ImageSharp

Gym.NET depends on SixLabors.ImageSharp 2.1.13, the last release under the Apache-2.0 license. GitHub published five
advisories against it on 2026-10-07, which NuGet reports in your restore as NU1902/NU1903 (and which fail it under
`TreatWarningsAsErrors`). None of them is reachable through Gym.NET: it never encodes or decodes TIFF, never runs
histogram equalization, and decodes only its own embedded PNG assets. If you don't use ImageSharp yourself on untrusted
input, you can suppress them; the release notes list the five advisories and the `NuGetAuditSuppress` lines.

## License

Apache-2.0. Gymnasium's environment assets are used under Gymnasium's MIT license; see
[THIRD-PARTY-NOTICES.md](https://github.com/SciSharp/Gym.NET/blob/master/THIRD-PARTY-NOTICES.md).
