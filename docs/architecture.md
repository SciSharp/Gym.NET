# Architecture

[Documentation index](README.md)

How Gym.NET 2.0 is put together, and how it is held to Gymnasium's behavior. The reasons behind each choice are recorded
in the [completion plan](plans/COMPLETION_PLAN.md) (decisions D1–D22) and the [design review](plans/DESIGN_REVIEW.md)
(designs DR-01…DR-36); the references in parentheses point there.

## What Gym.NET is

Gym.NET is a C# codebase whose public behavior is Gymnasium's: for the same inputs, seed and actions, its environments
return the same dtypes, shapes and bytes, the same IEEE-754 bits, the same info keys in the same order, the same
exceptions with the same messages, the same generator states and the same frames. Gymnasium is pinned as the
`refs/Gymnasium` submodule (v1.4.0, `8ba8c3a6`), and every member based on Gymnasium's source links the exact lines it
follows. Gym.NET adds what .NET users expect: typed generic APIs, desktop viewers, `ValueTask` stepping, generated
option records.

## Packages and projects

| Package | Project | Holds | Depends on |
|---|---|---|---|
| Gym.NET | `src/Gym` | The env contract, wrappers, spaces, registry and `Gymnasium.Make`/`MakeVec`, vector envs, checkers, seeding, errors, the Python-semantics helpers, `Gym.Legacy`; the source generator ships inside it as an analyzer | NumSharp 0.80.0, SixLabors.ImageSharp 2.1.13 |
| Gym.NET.Rendering | `src/Gym.Rendering` | The rasterizer the envs draw `rgb_array` frames with: `Surface`, `GfxDraw`, `Draw`, `Transform`, `ImageIO`, `SurfArray`, `Vector2`, `Clock` | Gym.NET, ImageSharp (PNG decoding) |
| Gym.NET.Environments | `src/Gym.Environments` | classic_control and toy_text, `CartPoleVectorEnv`, the embedded images, the registration of the built-in ids, the 0.2 `LunarLanderEnv` | Gym.NET, Gym.NET.Rendering; Aether.Physics2D, ImageSharp.Drawing and SixLabors.Fonts for the 0.2 LunarLander only |
| Gym.NET.Rendering.Avalonia | `src/Gym.Rendering.Avalonia` | `AvaloniaEnvViewer`, a window for `human` mode | Gym.NET.Environments, Avalonia 12 |
| Gym.NET.Rendering.WinForm | `src/Gym.Rendering.WinForm` | `WinFormEnvViewer`, the same on Windows Forms | Gym.NET.Environments, Windows Forms |
| (inside Gym.NET) | `src/Gym.SourceGenerators` | The generator behind `[GymOptions]`, `[GymTuple]` and `[GymDict]` (DR-13, DR-34) | Roslyn 4.8 |

Gymnasium's modules map to namespaces by area, with Gymnasium's class names kept (D1, D16):

| Gymnasium | Gym.NET |
|---|---|
| `gymnasium.make`, `register`, `spec`, `pprint_registry` | `Gym.Gymnasium` (a static class) |
| `gymnasium.core` (`Env`, `Wrapper`, ...) | `Gym.Envs`; results (`StepResult`, `ResetResult`) in `Gym.Observations` |
| `gymnasium.envs.registration` | `Gym.Envs.Registration` |
| `gymnasium.spaces` | `Gym.Spaces` |
| `gymnasium.wrappers` | `Gym.Wrappers` |
| `gymnasium.vector` | `Gym.Vector` |
| `gymnasium.error` | `Gym.Exceptions`; Python's built-in exceptions come from NumSharp (D17) |
| `gymnasium.utils` | `Gym.Utils` |
| `gymnasium.envs.classic_control`, `toy_text` | `Gym.Environments.Envs.Classic`, `Gym.Environments.Envs.ToyText` |

## The env contract

`Env<TObs, TAct>` is the base of every env and wrapper:

- `Reset(BigInteger? seed, options)` → `ResetResult<TObs>(Observation, Info)`. Implementations start with
  `SeedNpRandom(seed)`, Gymnasium's `super().reset(seed=seed)`.
- `Step(TAct)` → `StepResult<TObs>(Observation, Reward, Terminated, Truncated, Info)`; rewards are `double` everywhere.
- `StepAsync` is a `ValueTask` that runs `Step` on the calling thread; an env with real asynchronous work overrides it.
  Wrappers do not forward it, so a wrapper's `Step` always applies.
- `Render()` renders in the mode fixed at construction; `human` mode draws inside `Reset` and `Step`.
- `NpRandom` is the env's generator, created lazily from entropy, `NpRandomSeed` its seed (-1 once a generator is
  assigned); `Metadata`, `Spec`, `Unwrapped`, `Close()`/`Dispose()`.
- Python attributes by Python name: `GetWrapperAttr`/`SetWrapperAttr`/`HasWrapperAttr`, through `PyAttributes`
  (`[PyName]`, or the snake_case of the property).

`IEnv` is the same contract untyped: observations are `object`s, and `Step(object)` converts the action through the env's
`ConvertAction`. Wrappers derive from `Wrapper<TWObs, TWAct, TObs, TAct>` (or `Wrapper<TObs, TAct>`) and the
`ObservationWrapper`, `ActionWrapper` and `RewardWrapper` bases.

## Typed and untyped

Gym.NET keeps Gymnasium's dynamic behavior and adds static types where they hold:

- **Spaces** are `Space<T>` over an untyped `Space` (DR-07). An array space holds `NDArray`s, and a discrete sample is a
  0-d int64 array, the shape NumPy gives it (DR-08). A `Tuple` holds `object?[]` (Python's tuple), a `Dict` holds
  `Info` (Python's dict), `Text` holds strings (DR-13).
- **`Make` and `MakeVec`** come typed (`Make<TObs, TAct>`, which raises `TypeError` for other types) and untyped
  (`IEnv`, `IVectorEnv`) (DR-10, DR-21). A vector env is typed by its batched types: `VectorEnv<NDArray, NDArray>` for
  CartPole.
- **Generated records**: `[GymTuple]`/`[GymDict]` records convert to and from the generic tuple and dict, compare and
  print as Python does, and read observations by name (`BlackjackObservation`); `[GymOptions]` gives an env class a
  `<Class>Options` record of its constructor arguments, checked by the compiler (DR-13, DR-34).
- Where the type depends on an argument at run time, the API stays `object` (`TimeAwareObservation`), as recorded in the
  design review.

## Python's semantics, in .NET

Matching Gymnasium byte for byte means matching Python and NumPy where Gymnasium leans on them:

- **NumPy**: arrays, dtypes, NEP 50 promotion and the `Generator(PCG64(SeedSequence))` streams come from NumSharp,
  verified against NumPy 2.4.2. Where Gymnasium mixes float32 and float64 scalars, `NpScalar` carries NumPy's scalar
  typing.
- **Python**: `PyBuiltins` (`repr`, `str`), `PyEquality` (`==` across ints, floats, tuples, dicts and arrays),
  `PyMath` (`//`, `%`, `divmod`, `round`, `np.clip`, `np.sign`), `PyKwargs` (binding keyword arguments to constructor
  parameters by Python name), `PyJson` (`json.dumps`/`loads` for `EnvSpec`), `Difflib` (the registry's "Did you mean"),
  `Warnings` and `Logger` (warnings as `gymnasium.logger` issues them).
- **Errors**: every Gymnasium error type and message. Python's built-in exceptions are NumSharp's types (`ValueError`,
  `TypeError`, `KeyError`, `AssertionError`, ...); `OverflowError` and `ZeroDivisionError` are .NET's
  `OverflowException` and `DivideByZeroException`, as NumSharp throws them; `NotImplementedError` and `ImportError` live
  in `Gym.Exceptions` until NumSharp has them (N-3).
- **Platform**: Windows x64 is the reference platform. Where Gymnasium's arithmetic goes through the C runtime (CartPole's
  `cos`/`sin` in float64, `pow`), .NET calls the same CRT functions there; on Linux and macOS the platform's C library may
  differ in the last bit, so those results are verified on Windows and tracked elsewhere.

## Registration

The built-in ids are registered by `GymEnvironments.Register()`, from a module initializer of Gym.NET.Environments
(DR-12). Because a library's initializer only runs once something touches the library, the package also compiles a
one-line initializer into each consuming C# app through `buildTransitive/Gym.NET.Environments.targets`, which runs before
`Main` (DR-17). The environment variable `GYMNET_DISABLE_ENV_REGISTRATION` disarms it (DR-14), and
`GymNetDisableEnvRegistrationInitializer=true` leaves it out of a build. F# and Visual Basic apps call `Register()`
themselves.

## Rendering

`rgb_array` frames are drawn by `Gym.Rendering`, which reproduces what pygame-ce 2.5.8 draws for Gymnasium, pixel for
pixel: a `Surface` with pygame's semantics (opaque or per-pixel alpha), SDL_gfx 2.0.26's `gfxdraw` primitives in C#
(`GfxDraw`), and Gym.NET's own `Draw`, `Transform` and `ImageIO`, derived from pygame-ce's output (D4). The images the envs
draw are Gymnasium's own, embedded in Gym.NET.Environments (DR-19).

`human` mode renders into a viewer: the env's own `viewerFactory`, else the app-wide `Gymnasium.HumanViewerFactory`, else
`DependencyNotInstalled` at the first frame, as Gymnasium does without pygame (DR-20). Frames are paced by
`Gym.Rendering.Clock` at the env's `render_fps`.

## Array ownership

NumSharp arrays live in unmanaged memory (D19). Whatever an env, space or wrapper returns belongs to the caller; spaces
and envs dispose what they own. A space has one owner (DR-40): the env it was given to, which disposes it with itself.
Handing it to a second env that is not yet disposed throws at once, naming the fix: give each env its own space, a new
one or `space.DeepCopy()`. A wrapper handed its own env's space passes it through without owning it, and a space an env
replaces is let go, undisposed. Results that carry arrays (`StepResult`, `ResetResult`, the vector results) implement
`INDArrayCarrier`, so NumSharp.Build's `[NDScoped]` weaving keeps them alive while it reclaims a method's temporaries.
NumSharp's analyzer reports arrays that are created and dropped (`NDW012`).

## How parity is proven

- **Gymnasium's own tests**: `tests/Gym.Tests/Gymnasium` rewrites Gymnasium's pytest files in C#, file for file.
- **The live parity suite** (`tests/Gym.Tests.Parity`): CPython runs inside the test process through pythonnet, imports
  Gymnasium from the submodule, and runs it next to Gym.NET with the same seeds and actions. `GymPair` steps both in
  lockstep and stops at the first divergence; `ByteContract` compares values by their bytes and bits, exceptions by type
  and message, generators by state; `FrameDiff` compares frames pixel for pixel. The suite checks that its Python stack
  matches `tools/parity/manifest.json`, and its harness is tested by planting defects it must catch.
- **The reverse bridge**: Gym.NET envs exposed to Python as Gymnasium envs, so Gymnasium's own `check_env` and
  `check_environments_match` run on them.
- **Goldens** (`tests/Golden`): trajectories and frames the live suite verified, replayed without Python, so every
  platform checks them.
- **The inventory**: every public Gymnasium name is present in Gym.NET or listed, with its reason, in a ratchet file.
- **`[OpenBugs]`**: a known gap is a test that asserts Gymnasium's answer and fails until the gap closes, so the gaps are
  enumerable, not hidden.
