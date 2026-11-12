# Getting started

[Documentation index](../README.md)

Gym.NET's API is Gymnasium's, spelled in C#: `gymnasium.make("CartPole-v1")` is `Gymnasium.Make("CartPole-v1")`,
`env.reset(seed=42)` is `env.Reset(seed: 42)`, and an env built with the same seed and fed the same actions returns
the same numbers as Gymnasium's. Every sample on this page compiles and runs as shown, warning-free.

## Install

```sh
dotnet add package Gym.NET.Environments        # the environments and Gym.NET's core
dotnet add package Gym.NET.Rendering.Avalonia  # only for render_mode "human"
```

The package registers its environment ids before your `Main` runs (a C# module initializer the package adds to your
project), so `Gymnasium.Make` works with no set-up call. F# and Visual Basic projects call
`Gym.Environments.GymEnvironments.Register()` once instead.

Gym.NET needs the JIT: NumSharp compiles its array kernels at run time, so Native AOT and trimmed publishing are not
supported. A .NET 10 file-based app (`dotnet run app.cs`) publishes AOT by default, and its first env throws
`PlatformNotSupportedException` until the app opts out with `#:property PublishAot=false`.

## Make an env and run an episode

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

The later samples also use `System.Collections.Generic`, `Gym.Observations`, `Gym.Spaces`, `Gym.Wrappers`, `Gym.Vector`,
`Gym.Environments.Envs.Classic` and `Gym.Rendering.Avalonia`.

- `Reset(seed, options)` returns a `ResetResult<TObs>`, `(Observation, Info)`.
- `Step(action)` returns a `StepResult<TObs>`, `(Observation, Reward, Terminated, Truncated, Info)`. `Terminated`
  means the task reached a terminal state (the pole fell); `Truncated` means something outside the task cut the
  episode short (`TimeLimit`'s step limit). Reset when either is set.
- Rewards are `double` everywhere, also where Gymnasium returns an `int` or a float32.
- `Info` is a Python dict: insertion-ordered, keyed by strings, compared and printed as Python does.

### Releasing what an env returns

The arrays an env returns belong to the caller, and they hold native memory. Both results are `IDisposable`: disposing
one releases its observation and the arrays in its info, so `using` does it at the end of the scope:

```csharp
using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("CartPole-v1");
using ResetResult<NDArray> reset = env.Reset(seed: 0);
using NDArray action = env.ActionSpace.Sample();
using StepResult<NDArray> step = env.Step(action);
Console.WriteLine($"{step.Observation.Shape} {step.Reward}");   // (4) 1
```

A copy of a result shares its arrays: dispose one of them, after the last use, and `.copy()` an observation you keep
longer. The vector results (`VectorStepResult`, `VectorResetResult`) work the same way.

### Typed and untyped

`Gymnasium.Make<TObs, TAct>` returns an `Env<TObs, TAct>` and raises `TypeError` when the env has other types.
`Gymnasium.Make` returns an `IEnv`, whose observations are `object`s and whose `Step(object)` converts the action:

```csharp
using IEnv env = Gymnasium.Make("CartPole-v1");           // untyped: observations are objects
var (observation, info) = env.Reset(seed: 0);
var step = env.Step(1);                                   // the env converts the action (here to a 0-d int64 array)
Console.WriteLine($"{observation} {step.Reward}");
(observation as IDisposable)?.Dispose();
(step.Observation as IDisposable)?.Dispose();
```

| Ids | Env | Observation | Action |
|---|---|---|---|
| CartPole-v0, CartPole-v1 | `CartPoleEnv` | `NDArray` float32 `(4,)` | `NDArray` 0-d int64, `Discrete(2)` |
| MountainCar-v0 | `MountainCarEnv` | `NDArray` float32 `(2,)` | `NDArray` 0-d int64, `Discrete(3)` |
| MountainCarContinuous-v0 | `MountainCarEnv(continuous: true)` | `NDArray` float32 `(2,)` | `NDArray` float32 `(1,)`, `Box(-1, 1)` |
| Pendulum-v1 | `PendulumEnv` | `NDArray` float32 `(3,)` | `NDArray` float32 `(1,)`, `Box(-2, 2)` |
| Acrobot-v1 | `AcrobotEnv` | `NDArray` float32 `(6,)` | `NDArray` 0-d int64, `Discrete(3)` |
| FrozenLake-v1, FrozenLake8x8-v1 | `FrozenLakeEnv` | `NDArray` 0-d int64 (the cell) | `NDArray` 0-d int64, `Discrete(4)` |
| CliffWalking-v1, CliffWalkingSlippery-v1 | `CliffWalkingEnv` | `NDArray` 0-d int64 | `NDArray` 0-d int64, `Discrete(4)` |
| Taxi-v4 | `TaxiEnv` | `NDArray` 0-d int64 (the encoded state) | `NDArray` 0-d int64, `Discrete(6)` |
| Blackjack-v1 | `BlackjackEnv` | `object?[]`, Python's tuple `(player sum, dealer card, usable ace)`; `BlackjackObservation` reads it by name | `NDArray` 0-d int64, `Discrete(2)` |

A discrete action or observation is a 0-d int64 `NDArray`, the shape NumPy gives `Discrete.sample()`. `Gymnasium.PprintRegistry()`
prints every id; `Gymnasium.Spec(id)` returns its `EnvSpec`:

```csharp
Gymnasium.PprintRegistry();                               // every registered id, grouped by namespace
var spec = Gymnasium.Spec("CartPole-v1");
Console.WriteLine($"{spec.MaxEpisodeSteps} {spec.RewardThreshold}");   // 500 475
```

## Constructor arguments

`make`'s keyword arguments are the env's constructor arguments by their Python names. Pass them as an anonymous object,
a dictionary, or the options record generated for the env class:

```csharp
using IEnv a = Gymnasium.Make("CartPole-v1", new { render_mode = "rgb_array", sutton_barto_reward = true });
using IEnv b = Gymnasium.Make("CartPole-v1", new CartPoleEnvOptions { RenderMode = "rgb_array", SuttonBartoReward = true });
using IEnv c = Gymnasium.Make("CartPole-v1", new Dictionary<string, object?> { ["render_mode"] = "rgb_array" });
using IEnv d = Gymnasium.Make("CartPole-v1", maxEpisodeSteps: 1000);   // instead of the spec's 500
Console.WriteLine(a);   // <TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPoleEnv<CartPole-v1>>>>>
```

`CartPoleEnvOptions` lives next to `CartPoleEnv` in `Gym.Environments.Envs.Classic`; it passes only the properties you
set, so the registered defaults still apply. An unknown name raises Gymnasium's `TypeError`. Building an env class
directly works too (`new CartPoleEnv(renderMode: "rgb_array")`), without `make`'s wrappers and spec.

## Seeding

`Reset(seed: n)` seeds the env's generator, `NpRandom`, with Gymnasium's `Generator(PCG64(SeedSequence(n)))`, the
generator NumPy builds for `np.random.default_rng(n)`; NumSharp implements it bit for bit. So `Reset(seed: 42)` in
Gym.NET and `env.reset(seed=42)` in Gymnasium draw the same starting state. `Reset()` without a seed continues the
current stream. Spaces have their own generators: `env.ActionSpace.Seed(42)` makes `Sample()` repeat
`env.action_space.seed(42)` and `sample()`. `Gym.Utils.Seeding.NpRandom(seed)` returns such a generator, with its seed,
for your own draws (`gymnasium.utils.seeding.np_random`).

## Spaces

```csharp
var box = new Box(-1.0, 1.0, new Shape(3), dtype: np.float32, seed: 0);
using NDArray point = box.Sample();                  // float32 (3,), the values NumPy's generator draws for seed 0
bool inside = box.Contains(point);                   // true
var discrete = new Discrete(4, seed: 0);
using NDArray choice = discrete.Sample();            // a 0-d int64 array
var tuple = new Gym.Spaces.Tuple(new Space[] { discrete, box }, seed: 0);
object?[] sample = tuple.Sample();                   // Python's tuple: object?[] { NDArray, NDArray }
Console.WriteLine($"{box} {inside} {discrete} {tuple}");   // Box(-1.0, 1.0, (3,), float32) True Discrete(4) Tuple(Discrete(4), Box(...))
```

All ten of Gymnasium's spaces are in `Gym.Spaces`: `Box`, `Discrete`, `MultiBinary`, `MultiDiscrete`, `Tuple` (samples
are `object?[]`), `Dict` (samples are `Info`), `Text` (strings), `Sequence`, `Graph` (`GraphInstance`) and `OneOf`.
`SpaceUtils` holds `flatdim`, `flatten_space`, `flatten` and `unflatten`. Spaces own unmanaged memory (their bounds and
generators): dispose the ones you build yourself; an env disposes its own.

## Rendering

The render mode is a constructor argument, fixed for the env's life, as in Gymnasium. `Gym.Envs.RenderModes` names
them as constants (`RenderModes.RgbArray` is `"rgb_array"`), so either form works:

- `rgb_array`: `Render()` returns `RenderResult.Rgb`, whose `Frame` is a `(height, width, 3)` uint8 array with the
  pixels Gymnasium's env draws.

  ```csharp
  using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("CartPole-v1", new { render_mode = "rgb_array" });
  using NDArray observation = env.Reset(seed: 0).Observation;
  using RenderResult result = env.Render();
  if (result is RenderResult.Rgb rgb)
      Console.WriteLine(rgb.Frame.Shape);   // (400, 600, 3), uint8, the pixels Gymnasium's CartPole draws
  ```

- `ansi` (toy_text): `Render()` returns `RenderResult.Ansi` with the text Gymnasium prints, escape codes included.

  ```csharp
  using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("FrozenLake-v1", new { render_mode = "ansi" });
  using NDArray start = env.Reset(seed: 0).Observation;   // a 0-d int64 array: the cell, 0 to 15
  using RenderResult text = env.Render();
  Console.WriteLine(((RenderResult.Ansi)text).Text);        // the 4x4 map, the agent's cell highlighted
  ```

- `human`: every `Reset` and `Step` draws into a window, paced at `Metadata["render_fps"]`, and `Render()` returns
  `RenderResult.None`. Set the window type once, at start-up; without one, the first frame raises
  `DependencyNotInstalled`, as Gymnasium does without pygame.

  ```csharp
  Gymnasium.HumanViewerFactory = AvaloniaEnvViewer.Factory;   // once, at start-up (WinFormEnvViewer.Factory on Windows)
  using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("CartPole-v1", new { render_mode = "human" });
  using NDArray observation = env.Reset(seed: 0).Observation; // the window opens and shows the first frame
  // Every Step draws the next frame, paced at metadata["render_fps"] (50 for CartPole); Render() returns None.
  ```

An env class also takes its own viewer: `new CartPoleEnv(renderMode: "human", viewerFactory: WinFormEnvViewer.Factory)`.

A MuJoCo env's `human` window is Gymnasium's MuJoCo viewer, with its controls (DR-70): drag with the left button to orbit
the camera and with the right one to pan it (Shift: horizontally), scroll to zoom; Space pauses (each `Step` then waits
in the window until Space or → is pressed), → advances one step, Tab cycles the model's cameras, C shows contact
forces, E the bodies' frames, R makes the bodies transparent, 0 to 4 switch geom groups, T saves the frame as a PNG
(`/tmp/frame_0000000.png` and on, on the current drive), H hides the menu, S and F slow down and speed up, D switches
between drawing every step and real time, Esc closes the window. Its frames come from Gym.NET's software renderer, a
few to a few dozen a second.

## Wrappers

A wrapper is an env around an env: `Wrapper<TObs, TAct>` and the `ObservationWrapper`, `ActionWrapper` and
`RewardWrapper` bases, as in Gymnasium. 1.0 has `TimeLimit`, `OrderEnforcing`, `PassiveEnvChecker`, `Autoreset`,
`RecordEpisodeStatistics`, `RenderCollection`, `NormalizeObservation`, `NormalizeReward`, `TimeAwareObservation` and
`FlattenObservation` (`Gym.Wrappers`):

```csharp
using var env = new RecordEpisodeStatistics<NDArray, NDArray>(
    new NormalizeObservation<NDArray, NDArray>(Gymnasium.Make<NDArray, NDArray>("CartPole-v1")));
NDArray observation = env.Reset(seed: 0).Observation;
while (true) {
    using NDArray action = env.ActionSpace.Sample();
    var (next, _, terminated, truncated, info) = env.Step(action);
    observation.Dispose();
    observation = next;
    if (terminated || truncated) {
        var episode = (Info)info["episode"]!;   // {"r": return, "l": length, "t": seconds}
        Console.WriteLine($"return {episode["r"]}, length {episode["l"]}");
        break;
    }
}

observation.Dispose();
```

A wrapper owns the env it wraps: disposing the outermost one closes the stack. `env.Unwrapped` is the innermost env, and
Python attributes are reachable by their Python names through a stack (`env.GetWrapperAttr("gravity")`).

## Vector envs

A vector env steps several copies of an env at once and returns batched arrays:

```csharp
// 8 carts in one env: CartPole's own vector env (its vector entry point), or "sync" for 8 copies of make's stack.
using VectorEnv<NDArray, NDArray> envs = Gymnasium.MakeVec<NDArray, NDArray>("CartPole-v1", numEnvs: 8);
envs.ActionSpace.Seed(42);
using NDArray observations = envs.Reset(seed: 42).Observations;   // (8, 4) float32
using NDArray actions = envs.ActionSpace.Sample();                 // (8,) int64
var (next, rewards, terminations, truncations, infos) = envs.Step(actions);
using (next)
using (rewards)
using (terminations)
using (truncations)
    Console.WriteLine($"{next.Shape} {rewards.dtype.name} {terminations.dtype.name}");   // (8, 4) float64 bool
// A sub-env whose episode ended is reset by the next Step (AutoresetMode.NextStep, Gymnasium's default).
```

- `Gymnasium.MakeVec(id, n)` uses the env's own vectorized class when it has one (CartPole's `CartPoleVectorEnv`), else
  `SyncVectorEnv`; `vectorizationMode: "sync"` asks for `SyncVectorEnv` (n copies of `make`'s stack, stepped in turn).
- `MakeVec(id, n, vectorizationMode: "sync", vectorKwargs: new { autoreset_mode = "SameStep" })` picks the autoreset
  mode of a `SyncVectorEnv`: `NextStep` (the default) resets an ended sub-env at the following step, whose reward is 0;
  `SameStep` resets it at once and returns the last observation and info under `final_obs`/`final_info`; `Disabled`
  leaves it to `Reset(options: new Dictionary<string, object?> { ["reset_mask"] = mask })`, with `mask` a bool array of
  one flag per sub-env. A vectorized env class such as `CartPoleVectorEnv` takes no `vectorKwargs` (Gymnasium's `Error`)
  and has its own mode, `NextStep` for CartPole.
- Infos are merged as Gymnasium merges them: one array per key, and a boolean `_key` array saying which sub-envs set it.
- `vectorizationMode: "async"` builds `AsyncVectorEnv`, which steps the sub-envs in parallel on worker threads and
  returns what `SyncVectorEnv` returns, bit for bit. `vectorKwargs` choose where the calls run (`execution_mode`:
  `AsyncExecutionMode.ThreadPool`, the default, `DedicatedThreads` with `thread_count`, or `ThreadPerEnv`), cap how many
  sub-envs run at once (`max_concurrency`) and keep Gymnasium's `shared_memory` and `copy`. Each sub-env runs on another
  thread, so an env function must build its own spaces and buffers rather than share one object among sub-envs.

## Arrays belong to the caller

Observations, rewards, actions and frames are NumSharp arrays in unmanaged memory. What an env or a space returns is
yours: dispose it when done (`using`, or `Dispose()` when the next one replaces it), and dispose the arrays you create.
Arrays you drop are reclaimed by the garbage collector eventually, and NumSharp's analyzer, which comes with the
packages, warns about them (`NDW012`). The samples above dispose everything and build without a warning.

## The pre-0.26 API

Code written for Gym.NET 0.2 (`Reset()` returning only the observation, the 4-tuple `Step` with `Done`, `Seed(s)`,
`Render("rgb_array")`) can run against 1.0 through `Gym.Legacy.LegacyEnvAdapter`, obsolete and removed in 2.0. The
[migration guide](migrating_to_1.0.md) maps every 0.2 call to its replacement.
