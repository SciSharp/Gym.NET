# Migrating from Gym.NET 0.2 to 1.0

[Documentation index](../README.md) · [Getting started](getting_started.md)

Gym.NET 1.0 takes Gymnasium's API, the contract gym adopted in 0.26 and Gymnasium kept: `Reset(seed)` returns
`(observation, info)`, `Step` returns `(observation, reward, terminated, truncated, info)`, the render mode is fixed at
construction, and envs are made through a registry. It is a breaking release: the package ids
(`Gym.NET`, `Gym.NET.Environments`, `Gym.NET.Rendering.Avalonia`, `Gym.NET.Rendering.WinForm`) and the namespaces stay,
so most `using` directives keep working, but the members inside them change, and every package moves to 1.0.0 at once
(see Requirements). `Gym.NET.Rendering`, new, comes with `Gym.NET.Environments`.

Code that cannot move at once can keep the 0.2 API through `Gym.Legacy.LegacyEnvAdapter` (below), which is obsolete and
will be removed in 2.0.

## Requirements

- .NET 8, .NET 10 or .NET 11 (0.2 targeted .NET Core 3.1 and .NET 6).
- Every Gym.NET package at 1.0.0, moved together. 0.2.0 declared its dependencies as "0.2.0 or later", so NuGet
  accepts a mix of 0.2.0 and 1.0.0 packages without a warning, and the mix fails when it runs: Gym.NET.Environments
  0.2.0 over Gym.NET 1.0.0 cannot load its envs (`Could not load type 'Gym.Envs.Env'`), and a 0.2.0 viewer over
  Gym.NET.Environments 1.0.0 cannot load its window (`Could not load type 'Gym.Environments.IEnvViewer'`).
- NumSharp 0.81.0, which the packages bring, in place of 0.2's NumSharp.Lite 0.1.12. Remove any NumSharp.Lite
  reference: with both, code naming `NDArray` stops compiling (CS0433, the type is in both). NumSharp's own API differs
  from NumSharp.Lite's: dtypes are `DType`s (`np.float32` still reads the same), indices are `long`, and random
  generators are NumPy's `Generator`.
- Avalonia 12 for Gym.NET.Rendering.Avalonia (0.2's was built on Avalonia 0.10).
- In a build that treats warnings as errors, two suppressions, both explained, with the lines to add, under Breaking
  Changes in the [release notes](../releases/RELEASE_1.0.0.md): ImageSharp 2.1.13's five advisories, and, with
  Gym.NET.Rendering.WinForm, NU1701 for the .NET Framework packages ScottPlot.WinForms brings.

## The calls, old and new

| 0.2 | 1.0 |
|---|---|
| `new CartPoleEnv()` | `Gymnasium.Make("CartPole-v1")`, which adds Gymnasium's wrappers (`TimeLimit` of 500 steps, `OrderEnforcing`, `PassiveEnvChecker`); `new CartPoleEnv()` still builds the bare env |
| `new CartPoleEnv(WinFormEnvViewer.Factory)` | `Gymnasium.HumanViewerFactory = WinFormEnvViewer.Factory;` then `Gymnasium.Make("CartPole-v1", new { render_mode = "human" })`, or `new CartPoleEnv(renderMode: "human", viewerFactory: WinFormEnvViewer.Factory)` |
| `env.Seed(s); env.Reset()` | `env.Reset(seed: s)` |
| `NDArray obs = env.Reset()` | `var (obs, info) = env.Reset()`, or `env.Reset().Observation` |
| `Step step = env.Step(a)` | `var (obs, reward, terminated, truncated, info) = env.Step(a)` (a `StepResult<TObs>`) |
| `step.Done` | `terminated \|\| truncated`: `terminated` when the task ended (the pole fell), `truncated` when a limit cut the episode short |
| `step.Reward` (`float`) | `reward` (`double`) |
| `step.Information` (`Gym.Collections.Dict`, or `null`) | `info` (`Gym.Envs.Info`, never `null`); `step.Information` still compiles, obsolete, until 2.0 (DR-78). `info.Get<T>(key)` reads a value as `T`, converting NumPy scalars, and a built-in env's info also reads as a typed record (`HalfCheetahInfoV5 typed = step.Info;`) |
| `dict1.Equals(dict2)` (0.2's `Dict` compared by value) | `info1.PyEquals(info2)`: Python's dict `==`; an `Info`'s `Equals` and `==` compare references (DR-92) |
| `env.Render("rgb_array")` returning an ImageSharp `Image` | build the env with `render_mode = "rgb_array"` (or `RenderMode.RgbArray`); `env.Render()` returns `RenderResult.Rgb`, whose `Frame` is a `(height, width, 3)` uint8 array; `(Image)env.Render()` makes an ImageSharp image of it |
| `env.Render("human")` after each step | build the env with `render_mode = "human"` (or `RenderMode.Human`); every `Reset` and `Step` draws the frame itself, paced at the env's frame rate |
| A viewer of your own: `IEnvViewer.Render(Image img)`, `CloseEnvironment()`, a factory returning `Task<IEnvViewer>` (`IEnvironmentViewerFactoryDelegate`) | `IEnvViewer.Render(NDArray frame)` (the `(H, W, 3)` uint8 frame, the caller's), `Close()`, and an `EnvViewerFactory` that returns the viewer once its window is up; `IAsyncEnvViewer` adds `RenderAsync`/`CloseAsync` (DR-85). `CloseEnvironment()` stays, obsolete, until 2.0 |
| `env.CloseEnvironment()` | `env.Close()`, or `Dispose()` (`using`) |
| `Task<Step> env.StepAsync(a)` | `ValueTask<StepResult<TObs>> env.StepAsync(a, cancellationToken)`, which runs the step on the calling thread |
| `env.Metadata` (`Dict`, keys `render.modes`, `video.frames_per_second`) | `env.Metadata` (`Gym.Envs.Metadata`, keys `render_modes`, `render_fps`) |
| `env.RewardRange` | removed, as in Gymnasium 1.0 |
| `class MyEnv : Env`, `class MyEnv : Env<TAction>` | `class MyEnv : Env<TObs, TAct>` ([Creating custom environments](creating_custom_environments.md)) |
| `GoalEnv` | removed (Gymnasium-Robotics' concern) |
| `Gym.Threading` (`DistributedScheduler`, `DistributedThread`, `DistributedQueuedTaskScheduler`, `ThreadAffinityDistrubution`, `CancellationSourceExtensions`), which 0.2's `StepAsync` ran on | removed: built on `Thread.Abort`, which throws on .NET 5 and later. `StepAsync` runs the step on the calling thread; `Gymnasium.MakeVec(id, n, "async")` steps envs on worker threads |
| `IVecEnv`, `VecEnv`, `DummyVecEnv`, `VecEnvWrapper` | `Gym.Vector`: `Gymnasium.MakeVec("CartPole-v1", n)`, `SyncVectorEnv`, `VectorEnv<TObs, TAct>`; the old types moved to `Gym.Legacy`, obsolete |
| `catch (InvalidActionError)` | compiles, with a warning: `InvalidActionError` is now the obsolete base of Gymnasium's `InvalidAction` (DR-58), which CarRacing raises and a custom env should throw. The envs that check their action assert it, as Gymnasium's do, and raise `AssertionError` (LunarLander, CartPole, ...), which this `catch` does not see |
| `AlreadySteppingError`, `NotSteppingError` | compile, with a warning: the obsolete bases of the `AlreadyPendingCallError` and `NoAsyncCallError` that `AsyncVectorEnv` raises (DR-58), so a 0.2 `catch` now catches them (0.2 itself threw neither) |
| `Gym.Collections.CircularQueue<T>`, `CircularQueueThreadSafe<T>` | removed: a `Queue<T>` cycled with `queue.Enqueue(queue.Dequeue())`, or `ConcurrentQueue<T>` |
| `Gym.Reflection.GenericCallFactory` | removed: `method.MakeGenericMethod(type).Invoke(target, args)` |
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

1.0:

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

0.2 returned an ImageSharp `Image`; 1.0 returns the frame as an array, the way Gymnasium does, and converts it to an
image on request:

```csharp
using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("CartPole-v1", RenderMode.RgbArray);
using NDArray observation = env.Reset(seed: 0).Observation;
using RenderResult result = env.Render();
NDArray frame = (NDArray)result;                                     // (400, 600, 3) uint8, the result's own array
using Image image = (Image)result;                                   // a new Image<Rgb24> of the frame, yours to dispose
```

## Keeping 0.2 code running: Gym.Legacy

`LegacyEnvAdapter` exposes any 1.0 env through the 0.2 interface (`ILegacyEnv`):

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

Everything in `Gym.Legacy` is `[Obsolete]`, with the replacement named in the warning, and is removed in 2.0.

### The legacy `Dict` and its helpers

`Gym.Legacy` still hands out 0.2's `Gym.Collections.Dict` (`Step.Information`, `ILegacyEnv.Metadata`,
`VecEnv.Metadata`), so `Dict` and the fifteen types behind it stay in 1.0, `[Obsolete]` (DR-56), and go in 2.0 with
`Gym.Legacy`. Each warning names the replacement:

| 0.2 type | 1.0 replacement |
|---|---|
| `Gym.Collections.Dict` | `Gym.Envs.Info` for infos, `Gym.Envs.Metadata` for metadata |
| `Gym.Dynamic.Expando` | `System.Dynamic.ExpandoObject` |
| `Gym.Merging.Merger`, `IMergable`, `IMergable<T>`, `MergePrefer` | `Info.Update`, Python's `dict.update`: the new value wins, one level deep |
| `Gym.Collections.ExpandableList`, `ExpandableList<T>`, `IExpandableList` | `List<T>` |
| `Gym.Reflection.DefaultConstructor`, `DefaultConstructor<T>`, `DefaultConstructorNotFoundException` | `Activator.CreateInstance<T>()`, or a `new()` constraint |
| `Gym.Reflection.ReflectionHelper` | `Type.GetConstructor(Type.EmptyTypes)`, `Type.IsAssignableTo`, `System.Runtime.CompilerServices.ITuple` |
| `DictInlineInitializationException`, `DictMergingException` | none needed |

Moving code from `Dict` to `Info`, member by member:

| `Dict` | `Info` |
|---|---|
| `new Dict("key", value, ...)`, `Dict.Empty` | `new Info { ["key"] = value, ... }`, `new Info()` |
| `d["key"]`, `ContainsKey`, `TryGetValue`, `Remove`, `foreach` | the same |
| `d.Get<T>("key")` (`default(T)` when missing) | `info.Get<T>("key")`, the same read (`default(T)` when missing), which also converts NumPy scalars and numbers (DR-78); or `(T)info["key"]!`, or `info.Get("key") is T value` (Python's `get`: `null` when missing) |
| `d.Get<T>("key", fallback)` | `info.Get("key", fallback)`, which returns `object?` |
| `d.Update(other)`, `Append(other)`, `Override(other)` | `info.Update(other)` |
| `d.Merge(other, MergePrefer.New)` | `info.Update(other)`, one level deep; nothing replaces the deep merge |
| `d.Clone()` (shallow) | `info.Copy()` (shallow) or `info.DeepCopy()` |
| `d.RemoveWhere(predicate)` | `foreach (string key in info.Keys.Where(...).ToList()) info.Remove(key);` |
| `d.AsDynamic()` | a `System.Dynamic.ExpandoObject` filled from the info |

`Info` keeps Python's insertion order through removals, which `Dict` (a `Dictionary`) does not promise.

`Info` also takes `Dict`'s two reads (DR-58): `info.Get<T>("key")` (`default(T)` when missing) and `info.Clone()`
(the same as `Copy()`). `Get<T>` is an extension in `Gym.Envs` (`InfoExtensions`, DR-78), so it reads any string-keyed
dict and converts what Gymnasium's envs store: `info.Get<double>("x_position")` reads a NumPy scalar (a 0-d array),
which a plain cast refuses; an integer is range-checked, and a float reads as an integer only when it is whole. There
is no `Get<T>("key", fallback)`: it would capture `Get("key", fallback)` calls, which must keep returning Python's
`object`.

### 0.2 lines that still compile

Compiling 0.2's own tests against 1.0 found the lines below broken; each now compiles with an `[Obsolete]` warning that
names the 1.0 spelling (DR-58), and goes in 2.0 with `Gym.Legacy`:

| 0.2 line | What it does in 1.0 | 1.0 spelling |
|---|---|---|
| `var (obs, reward, done, info) = env.Step(a);` | `done` is `terminated \|\| truncated`, as gym 0.21 reported it | `var (obs, reward, terminated, truncated, info) = env.Step(a);` |
| `NDArray obs = env.Reset();` | the observation; the info is dropped, and arrays in it are never disposed (Taxi's action mask) | `env.Reset().Observation`, or `var (obs, info) = env.Reset();` |
| `env.CloseEnvironment();` | `Close()` | `env.Close()`, or `using` |
| `new Box(-5f, 5f, np.float32)`, `new Box(low, high, dType: np.float32)` | the box `dtype:` builds; `new Box(low, high, null)` is then ambiguous | `new Box(-5, 5, dtype: np.float32)` |
| `new CartPoleEnv(WinFormEnvViewer.Factory)` | human mode in that window, drawing every step itself (drop 0.2's per-step `Render()`, which would draw twice) | `new CartPoleEnv(renderMode: "human", viewerFactory: WinFormEnvViewer.Factory)` |
| `new CartPoleEnv(NullEnvViewer.Factory)` | headless, no render mode (`Render()` warns once and returns nothing) | `new CartPoleEnv()` |
| `(int)LunarLanderDiscreteActions.FireMainEngine` | 2, Gymnasium's main engine (0 nothing, 1 left, 3 right); the enum is in `Gym.Environments.Envs.Box2D`, and in 0.2's `Gym.Environments.Envs.Aether`, revived for it alone (importing both is ambiguous) | `2` |
| `env.ContinuousMode`, `LunarLanderEnv.SCALE`, `MAIN_ENGINE_POWER`, `SIDE_ENGINE_POWER`, `INITIAL_RANDOM` | the same values, 0.2's float constants | `env.Continuous`, `LunarLanderEnv.Scale`, ... (doubles) |

Still broken: `env.Seed(s)` (`Reset(seed: s)`; a `Seed` method would make `check_env` warn on every env, as Gymnasium's
does), `env.RewardRange` (Gymnasium 1.0 removed it), `env.Render("rgb_array")` returning an `Image` (see Rendering to an
image), NumSharp's now-explicit `(float)array`, and 0.2's Aether `LunarLanderEnv`.

## LunarLander

The `LunarLanderEnv` of 0.2 (`Gym.Environments.Envs.Aether`, Aether physics) is gone, with its
`LunarLanderDiscreteActions` enum and the Aether.Physics2D, ImageSharp.Drawing and SixLabors.Fonts dependencies. Its
replacement is Gymnasium's LunarLander on Gym.NET's Box2D engine: `Gymnasium.Make("LunarLander-v3")` or
`Gymnasium.Make("LunarLanderContinuous-v3")`, or `Gym.Environments.Envs.Box2D.LunarLanderEnv` built directly.

| 0.2 | 1.0 |
|---|---|
| `new LunarLanderEnv(viewerFactory, random_state: rng, continuous: c, render_mode: m)` | `new LunarLanderEnv(renderMode: m, continuous: c, viewerFactory: f)`; seed through `Reset(seed)`, or assign `env.NpRandom` |
| `(int)LunarLanderDiscreteActions.FireMainEngine` | the action `2` (0 nothing, 1 left engine, 2 main engine, 3 right engine), as Gymnasium's |
| `env.ContinuousMode` | `env.Continuous` |
| `LunarLanderEnv.SCALE`, `INITIAL_RANDOM`, ... (mutable fields) | `LunarLanderEnv.Scale`, `InitialRandom`, ... (constants) |
| v2 observation bounds, a 0-d continuous action space | v3's bounds, `Box(-1, 1, (2,), float32)` |

The episodes differ from 0.2's: the physics is Box2D 2.3.0, bit for bit as Gymnasium's pybox2d, and the draws follow
Gymnasium's order, so `reset(seed)` gives Gymnasium's episode.
