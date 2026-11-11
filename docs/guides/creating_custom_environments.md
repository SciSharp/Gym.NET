# Creating custom environments

[Documentation index](../README.md) · [Getting started](getting_started.md)

A custom environment is a class deriving from `Env<TObs, TAct>`, the base every Gym.NET env shares. It follows
Gymnasium's `gymnasium.Env` contract, so Gym.NET's wrappers, vector envs, registry and checkers work on it unchanged.
The sample on this page compiles and passes `EnvChecker.CheckEnv` as shown.

## The contract

| Member | What it must do |
|---|---|
| Constructor | Set `ObservationSpace`, `ActionSpace`, `Metadata` (usually the class's `ClassMetadata`) and `RenderMode`. `make` passes its keyword arguments here by Python name: a parameter `renderMode` receives `render_mode`. |
| `ClassMetadata` | A `static new Metadata` property: the render modes the env supports (`render_modes`) and its frame rate (`render_fps`), as Gymnasium's class attribute `metadata`. `make` reads it before building the env. |
| `Reset(seed, options)` | Start with `SeedNpRandom(seed)`, which reseeds `NpRandom` when a seed is given and keeps the stream otherwise. Draw every random value from `NpRandom`. Return `ResetResult<TObs>(observation, info)`. |
| `Step(action)` | Advance one step and return `StepResult<TObs>(observation, reward, terminated, truncated, info)`. Leave time limits to `TimeLimit` (`maxEpisodeSteps` at registration), so `truncated` is usually `false` here. |
| `Render()` | Render in the mode fixed at construction: a frame (`RenderResult.Rgb`), text (`RenderResult.Ansi`), or `RenderResult.None`. In `human` mode, draw inside `Reset` and `Step` instead. |
| `Close()` / `Dispose(bool)` | Release what the env holds (windows, files) in `Close`, which may run more than once. `Dispose()` calls `Close()`, then releases the env's spaces; a wrapper also disposes the env it wraps. Because the env owns its spaces, give every env its own: assigning a space another env still owns throws (`space.DeepCopy()` makes an unowned copy). |

Return new arrays from `Reset` and `Step`: the caller owns them and may dispose them, and Gymnasium's checker refuses an
env that hands out the same buffer twice.

## An example: a corridor

```csharp
using System;
using System.Collections.Generic;
using System.Numerics;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Observations;
using Gym.Spaces;
using Gym.Utils;
using NumSharp;

/// <summary>
///     A corridor of <c>size</c> cells. The agent starts on a random cell and moves left (0) or right (1); reaching the
///     right end ends the episode with reward 1, every other step costs 0.01.
/// </summary>
[GymOptions]
public sealed partial class CorridorEnv : Env<NDArray, NDArray> {
    private readonly long _size;
    private long _position = -1;

    /// <summary>Builds the corridor; <c>make</c> passes its kwargs here by Python name (<c>size</c>, <c>render_mode</c>).</summary>
    public CorridorEnv(long size = 10, string? renderMode = null) {
        if (size < 2)
            throw new ArgumentOutOfRangeException(nameof(size), "The corridor needs at least two cells.");
        _size = size;
        ObservationSpace = new Discrete(size);   // the agent's cell
        ActionSpace = new Discrete(2);           // 0: left, 1: right
        Metadata = ClassMetadata;
        RenderMode = renderMode;
    }

    /// <summary>The render modes and frame rate, as Gymnasium's class attribute <c>metadata</c>.</summary>
    public static new Metadata ClassMetadata => new() {
        [Metadata.RenderModesKey] = new List<string> { "ansi" },
        [Metadata.RenderFpsKey] = 4L,
    };

    /// <inheritdoc />
    public override ResetResult<NDArray> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
        SeedNpRandom(seed);   // Gymnasium's super().reset(seed=seed): a seed reseeds NpRandom, null keeps its stream
        using NDArray start = NpRandom.integers(0, _size - 1);
        _position = (long)start;
        return new ResetResult<NDArray>(NDArray.Scalar(_position), new Info());
    }

    /// <inheritdoc />
    public override StepResult<NDArray> Step(NDArray action) {
        if (_position < 0)
            throw new ResetNeeded("Call Reset before Step.");
        if (!ActionSpace.Contains(action))
            throw new ArgumentException($"{action} is not an action of {ActionSpace}.", nameof(action));
        _position = Math.Clamp(_position + ((long)action == 1 ? 1 : -1), 0, _size - 1);
        bool terminated = _position == _size - 1;
        double reward = terminated ? 1.0 : -0.01;
        return new StepResult<NDArray>(NDArray.Scalar(_position), reward, terminated, false, new Info());
    }

    /// <inheritdoc />
    public override RenderResult Render() {
        if (RenderMode != "ansi")
            return RenderResult.None;
        var cells = new char[_size];
        for (long i = 0; i < _size; i++)
            cells[i] = i == _position ? 'A' : i == _size - 1 ? 'G' : '.';
        return new RenderResult.Ansi(new string(cells));
    }
}
```

Notes on the choices:

- **Types.** A `Discrete` space holds 0-d int64 arrays, so both types are `NDArray`. An env with a `Tuple` observation
  space uses `object?[]` (Python's tuple), one with a `Dict` space uses `Info`, and one with a `Text` space uses `string`.
- **Randomness.** Every draw comes from `NpRandom`, so a seeded reset repeats exactly. `NpRandom` is NumPy's generator
  (`Generator(PCG64(SeedSequence(seed)))`), which is what lets a Gym.NET env repeat a Gymnasium env's draws.
- **`[GymOptions]`** on a `partial` class asks Gym.NET's source generator (shipped in the Gym.NET package) for
  `CorridorEnvOptions`, a record with one property per constructor parameter, so
  `Gymnasium.Make("Corridor-v0", new CorridorEnvOptions { Size = 8, RenderMode = "ansi" })` is checked by the
  compiler.
- **Errors.** Gymnasium's exception types are in `Gym.Exceptions` (`ResetNeeded`, `DependencyNotInstalled`, ...);
  Python's built-in ones (`ValueError`, `TypeError`, `KeyError`) come from NumSharp.

## Register it, make it, check it

```csharp
Gymnasium.Register("Corridor-v0", entryPoint: typeof(CorridorEnv), maxEpisodeSteps: 100);
using IEnv env = Gymnasium.Make("Corridor-v0", new { size = 8, render_mode = "ansi" });
EnvChecker.CheckEnv(env.Unwrapped);   // Gymnasium's check_env: spaces, seeding, determinism, render modes
```

- `Gymnasium.Register` takes Gymnasium's `register` arguments: `entryPoint` (the class, an `EnvCreator` delegate, or a
  `"Namespace:Class"` string), `maxEpisodeSteps` (adds `TimeLimit`), `rewardThreshold`, `kwargs` (default constructor
  arguments), `orderEnforce`, `disableEnvChecker`, `additionalWrappers` and `vectorEntryPoint` (a vectorized class for
  `MakeVec`). Ids follow Gymnasium's grammar, `[namespace/]name[-vN]`. To put several envs in one namespace, register
  them inside `using (EnvRegistry.Namespace("MyEnvs")) { ... }` (`Gym.Envs.Registration`): `"Grid-v0"` becomes
  `"MyEnvs/Grid-v0"`. The scope applies to the thread and async flow that opens it (DR-57), so plugins registering on
  other threads keep their own.
- `Gymnasium.Make` wraps the env as Gymnasium's `make` does: `PassiveEnvChecker` (it checks the first reset, step and
  render, and warns with Gymnasium's texts), `OrderEnforcing` (a step before the first reset raises `ResetNeeded`) and
  `TimeLimit`.
- `EnvChecker.CheckEnv` runs Gymnasium's `check_env` and raises or warns with Gymnasium's messages: the spaces, the
  types and membership of what `Reset` and `Step` return, that the same seed gives the same observations and a new
  object each time, the render modes, and closing twice. Run it on the unwrapped env, as Gymnasium recommends.

A registry entry lives for the process: register your ids at start-up, before the first `Make`. Gym.NET.Environments
registers its own from a module initializer, and its package compiles a one-line initializer into your app (through a
`buildTransitive` .targets file) that runs it before `Main`; a module initializer inside a library alone would wait until
something first touched that library.

## Vector envs of a custom env

`Gymnasium.MakeVec("Corridor-v0", 8, vectorizationMode: "sync")` runs 8 copies of `make`'s stack in a `SyncVectorEnv`
with no further code: the batched spaces come from `VectorUtils.BatchSpace` of the env's spaces. For speed, derive a
class from `VectorEnv<TObs, TAct>` that steps every copy in one call (as `CartPoleVectorEnv` does) and register it as the
`vectorEntryPoint`.
