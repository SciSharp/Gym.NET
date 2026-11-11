---
uid: api-index
---

# Gym.NET API reference

Generated from the XML documentation of the five packages' source on every build of this site.

| Namespace | Holds |
|---|---|
| @Gym | `Gymnasium`, the facade: `Make`, `MakeVec`, `Register`, `Spec`, `PprintRegistry`, `HumanViewerFactory` |
| @Gym.Envs | `Env<TObs, TAct>`, `IEnv`, the wrapper bases, `Info`, `Metadata`, `RenderResult`, `RenderModes` |
| @Gym.Envs.Registration | `EnvSpec`, `EnvRegistry`, `EnvCreator`, `WrapperSpec`, `VectorizeMode` |
| @Gym.Spaces | `Box`, `Discrete`, `MultiBinary`, `MultiDiscrete`, `Tuple`, `Dict`, `Text`, `Sequence`, `Graph`, `OneOf`, `SpaceUtils` |
| @Gym.Observations | `StepResult<TObs>`, `ResetResult<TObs>` |
| @Gym.Vector | `VectorEnv<TObs, TAct>`, `SyncVectorEnv`, `AsyncVectorEnv`, `AutoresetMode`, the vector wrapper bases |
| @Gym.Wrappers | `TimeLimit`, `RecordEpisodeStatistics`, `NormalizeObservation`, `RecordVideo` and the rest of Gymnasium's wrappers |
| @Gym.Environments.Envs.Classic | CartPole, MountainCar, Pendulum, Acrobot |
| @Gym.Environments.Envs.Box2D | LunarLander, BipedalWalker, CarRacing |
| @Gym.Environments.Envs.ToyText | Blackjack, FrozenLake, CliffWalking, Taxi |
| @Gym.Rendering | The pygame-compatible rasterizer: `Surface`, `Draw`, `GfxDraw`, `Transform`, `ImageIO` |

The [getting started guide](../../guides/getting_started.md) walks through the same types with runnable examples.
