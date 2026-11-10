# Migrating from Gym.NET 0.2 to 2.0

[Documentation index](../README.md) · [Getting started](getting_started.md)

Gym.NET 2.0 takes Gymnasium's API, the contract gym adopted in 0.26 and Gymnasium kept: `Reset(seed)` returns
`(observation, info)`, `Step` returns `(observation, reward, terminated, truncated, info)`, the render mode is fixed at
construction, and envs are made through a registry. It is a breaking release: the package ids
(`Gym.NET`, `Gym.NET.Environments`, `Gym.NET.Rendering.Avalonia`, `Gym.NET.Rendering.WinForm`) and the namespaces stay,
so package references and most `using` directives keep working, but the members inside them change.

Code that cannot move at once can keep the 0.2 API through `Gym.Legacy.LegacyEnvAdapter` (below), which is obsolete and
will be removed in 3.0.

## Requirements

- .NET 8 or .NET 10 (0.2 targeted .NET Core 3.1 and .NET 6).
- NumSharp 0.80.0, which the packages bring, in place of 0.2's NumSharp.Lite 0.1.12. NumSharp's own API differs from
  NumSharp.Lite's: dtypes are `DType`s (`np.float32` still reads the same), indices are `long`, and random generators
  are NumPy's `Generator`.

## The calls, old and new

| 0.2 | 2.0 |
|---|---|
| `new CartPoleEnv()` | `Gymnasium.Make("CartPole-v1")`, which adds Gymnasium's wrappers (`TimeLimit` of 500 steps, `OrderEnforcing`, `PassiveEnvChecker`); `new CartPoleEnv()` still builds the bare env |
| `new CartPoleEnv(WinFormEnvViewer.Factory)` | `Gymnasium.HumanViewerFactory = WinFormEnvViewer.Factory;` then `Gymnasium.Make("CartPole-v1", new { render_mode = "human" })`, or `new CartPoleEnv(renderMode: "human", viewerFactory: WinFormEnvViewer.Factory)` |
| `env.Seed(s); env.Reset()` | `env.Reset(seed: s)` |
| `NDArray obs = env.Reset()` | `var (obs, info) = env.Reset()`, or `env.Reset().Observation` |
| `Step step = env.Step(a)` | `var (obs, reward, terminated, truncated, info) = env.Step(a)` (a `StepResult<TObs>`) |
| `step.Done` | `terminated \|\| truncated`: `terminated` when the task ended (the pole fell), `truncated` when a limit cut the episode short |
| `step.Reward` (`float`) | `reward` (`double`) |
| `step.Information` (`Gym.Collections.Dict`, or `null`) | `info` (`Gym.Envs.Info`, never `null`) |
| `env.Render("rgb_array")` returning an ImageSharp `Image` | build the env with `render_mode = "rgb_array"`; `env.Render()` returns `RenderResult.Rgb`, whose `Frame` is a `(height, width, 3)` uint8 array |
| `env.Render("human")` after each step | build the env with `render_mode = "human"`; every `Reset` and `Step` draws the frame itself, paced at the env's frame rate |
| `env.CloseEnvironment()` | `env.Close()`, or `Dispose()` (`using`) |
| `Task<Step> env.StepAsync(a)` | `ValueTask<StepResult<TObs>> env.StepAsync(a, cancellationToken)`, which runs the step on the calling thread |
| `env.Metadata` (`Dict`, keys `render.modes`, `video.frames_per_second`) | `env.Metadata` (`Gym.Envs.Metadata`, keys `render_modes`, `render_fps`) |
| `env.RewardRange` | removed, as in Gymnasium 1.0 |
| `class MyEnv : Env`, `class MyEnv : Env<TAction>` | `class MyEnv : Env<TObs, TAct>` ([Creating custom environments](creating_custom_environments.md)) |
| `GoalEnv` | removed (Gymnasium-Robotics' concern) |
| `IVecEnv`, `VecEnv`, `DummyVecEnv`, `VecEnvWrapper` | `Gym.Vector`: `Gymnasium.MakeVec("CartPole-v1", n)`, `SyncVectorEnv`, `VectorEnv<TObs, TAct>`; the old types moved to `Gym.Legacy`, obsolete |
| `space.Seed(int)` | `space.Seed(seed)`, which also takes Gymnasium's lists and dicts of seeds for composite spaces |
| `new Discrete(n)` samples | 0-d int64 arrays, `start` to `start + n - 1` |
| `new Box(low, high, shape, typeof(float))` | `new Box(low, high, shape, dtype: np.float32)` |

## An episode, before and after

0.2:

```csharp
using var env = new CartPoleEnv(WinFormEnvViewer.Factory);
env.Seed(42);
NDArray observation = env.Reset();
while (true) {
    Step step = env.Step(env.ActionSpace.Sample());
    env.Render();
    if (step.Done)
        break;
    observation = step.Observation;
}
env.CloseEnvironment();
```

2.0:

```csharp
Gymnasium.HumanViewerFactory = WinFormEnvViewer.Factory;    // once, at start-up
using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("CartPole-v1", new { render_mode = "human" });
env.ActionSpace.Seed(42);
NDArray observation = env.Reset(seed: 42).Observation;
while (true) {
    using NDArray action = env.ActionSpace.Sample();
    var (next, reward, terminated, truncated, info) = env.Step(action);   // draws the frame
    observation.Dispose();                                               // arrays belong to the caller
    observation = next;
    if (terminated || truncated)
        break;
}

observation.Dispose();
```

What behaves differently, beyond the spelling:

- **Episodes end on either flag.** `Gymnasium.Make("CartPole-v1")` truncates at 500 steps, as Gymnasium's CartPole-v1;
  0.2's `CartPoleEnv` ran until the pole fell.
- **The numbers are Gymnasium's.** A seed now selects Gymnasium's random stream (NumPy's
  `Generator(PCG64(SeedSequence(seed)))`, not 0.2's `RandomState(seed)`), and CartPole returns float32 observations, as
  Gymnasium's does. The same seed gives the same trajectory as `gymnasium.make("CartPole-v1")` in Python, but not as
  0.2 did.
- **Order is enforced.** A `Step` before the first `Reset` raises `ResetNeeded`; stepping on after an episode ended warns
  with Gymnasium's message.
- **Arrays belong to the caller.** NumSharp arrays live in unmanaged memory; dispose what an env returns. NumSharp's
  analyzer, which comes with the packages, points at arrays that are dropped (`NDW012`).
- **Spaces belong to their env.** An env disposes the spaces it was given, so one space cannot serve two envs: assigning
  a space another undisposed env owns throws. Give each env its own, a new one or `space.DeepCopy()`.

## Rendering to an image

0.2 returned an ImageSharp `Image`; 2.0 returns the frame as an array, the way Gymnasium does. To get an image:

```csharp
using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("CartPole-v1", new { render_mode = "rgb_array" });
using NDArray observation = env.Reset(seed: 0).Observation;
using RenderResult result = env.Render();
var frame = ((RenderResult.Rgb)result).Frame;                       // (400, 600, 3) uint8
using Image<Rgb24> image = Image.LoadPixelData<Rgb24>(frame.ToArray<byte>(), (int)frame.shape[1], (int)frame.shape[0]);
```

## Keeping 0.2 code running: Gym.Legacy

`LegacyEnvAdapter` exposes any 2.0 env through the 0.2 interface (`ILegacyEnv`):

```csharp
#pragma warning disable CS0618 // Gym.Legacy is obsolete on purpose: it keeps 0.2 code running while it moves
using var env = new Gym.Legacy.LegacyEnvAdapter(Gymnasium.Make("CartPole-v1"));
env.Seed(42);                                 // recorded for the next Reset, as Reset(seed: 42)
NDArray observation = env.Reset();            // the observation alone
Gym.Legacy.Step step = env.Step(1);           // (Observation, Reward, Done, Information)
while (!step.Done)                            // Done = terminated || truncated
    step = env.Step(1);
#pragma warning restore CS0618
```

- `Seed(s)` records the seed for the next `Reset()`; `Reset()` drops the info.
- `Step` reports `terminated || truncated` as `Done` and, when a time limit ended the episode, sets
  `Information["TimeLimit.truncated"]`, as gym 0.21 did. The reward stays a `double`.
- `Render(mode)` accepts only the mode the env was built with, and returns an `Image` for `rgb_array`.
- The 0.2 vector types, `IVecEnv`, `VecEnv`, `DummyVecEnv` and `VecEnvWrapper`, are in `Gym.Legacy` too, over
  `ILegacyEnv`.

Everything in `Gym.Legacy` is `[Obsolete]`, with the replacement named in the warning, and is removed in 3.0.

## LunarLander

The `LunarLanderEnv` of 0.2 (`Gym.Environments.Envs.Aether`, Aether physics) is gone, with its
`LunarLanderDiscreteActions` enum and the Aether.Physics2D, ImageSharp.Drawing and SixLabors.Fonts dependencies. Its
replacement is Gymnasium's LunarLander on Gym.NET's Box2D engine: `Gymnasium.Make("LunarLander-v3")` or
`Gymnasium.Make("LunarLanderContinuous-v3")`, or `Gym.Environments.Envs.Box2D.LunarLanderEnv` built directly.

| 0.2 | 2.0 |
|---|---|
| `new LunarLanderEnv(viewerFactory, random_state: rng, continuous: c, render_mode: m)` | `new LunarLanderEnv(renderMode: m, continuous: c, viewerFactory: f)`; seed through `Reset(seed)`, or assign `env.NpRandom` |
| `(int)LunarLanderDiscreteActions.FireMainEngine` | the action `2` (0 nothing, 1 left engine, 2 main engine, 3 right engine), as Gymnasium's |
| `env.ContinuousMode` | `env.Continuous` |
| `LunarLanderEnv.SCALE`, `INITIAL_RANDOM`, ... (mutable fields) | `LunarLanderEnv.Scale`, `InitialRandom`, ... (constants) |
| v2 observation bounds, a 0-d continuous action space | v3's bounds, `Box(-1, 1, (2,), float32)` |

The episodes differ from 0.2's: the physics is Box2D 2.3.0, bit for bit as Gymnasium's pybox2d, and the draws follow
Gymnasium's order, so `reset(seed)` gives Gymnasium's episode.
