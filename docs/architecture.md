# Architecture

[Documentation index](README.md)

How Gym.NET 1.0 is put together, and how it is held to Gymnasium's behavior. The reasons behind each choice are recorded
in the [completion plan](plans/COMPLETION_PLAN.md) (decisions D1–D22) and the [design review](plans/DESIGN_REVIEW.md)
(designs DR-01…DR-95); the references in parentheses point there.

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
| Gym.NET | `src/Gym` | The env contract, wrappers, spaces, registry and `Gymnasium.Make`/`MakeVec`, vector envs, checkers, seeding, errors, the Python-semantics helpers, `Gym.Legacy`; the source generator ships inside it as an analyzer | NumSharp 0.81.0, SixLabors.ImageSharp 2.1.13 |
| Gym.NET.Rendering | `src/Gym.Rendering` | The rasterizer the envs draw `rgb_array` frames with: `Surface`, `GfxDraw`, `Draw`, `Transform`, `ImageIO`, `SurfArray`, `Vector2`, `Clock` | Gym.NET, ImageSharp (PNG decoding) |
| Gym.NET.Environments | `src/Gym.Environments` | classic_control, box2d (LunarLander, BipedalWalker, CarRacing), toy_text and mujoco (the 22 ids' classes in `Envs/MuJoCo`), `CartPoleVectorEnv`, the embedded images, CarRacing's score glyphs, Gymnasium's 14 compiled MuJoCo models and MuJoCo's overlay font, the registration of the built-in ids; the Box2D engine in `Box2D/` (namespace `Gym.Box2D`, DR-65) and the MuJoCo engine in `MuJoCo/` (namespace `Gym.MuJoCo`, DR-65, DR-66) | Gym.NET, Gym.NET.Rendering, SixLabors.ImageSharp 2.1.13 (the human window's images) |
| Gym.NET.Rendering.Avalonia | `src/Gym.Rendering.Avalonia` | `AvaloniaEnvViewer`, a window for `human` mode; `AvaloniaPlotFigure`, `PlayPlot`'s ScottPlot figure (DR-39) | Gym.NET.Environments, Avalonia 12, ScottPlot.Avalonia 5.1 |
| Gym.NET.Rendering.WinForm | `src/Gym.Rendering.WinForm` | `WinFormEnvViewer`, the same on Windows Forms; `WinFormPlotFigure` | Gym.NET.Environments, Windows Forms, ScottPlot.WinForms 5.1 |
| (inside Gym.NET) | `src/Gym.SourceGenerators` | The generator behind `[GymOptions]`, `[GymTuple]` and `[GymDict]` (DR-13, DR-34) | Roslyn 4.8 |
| (inside Gym.NET.Environments) | `src/Gym.Environments/Box2D` | Box2D 2.3.0 in managed C#, float32 and bit for bit as pybox2d 2.3.10 computes it, with pybox2d's surface (`b2World`, `CreateDynamicBody`, `b2FixtureDef`, `b2PolygonShape`, `b2RevoluteJointDef`, contact listeners, ray casts) for the box2d envs (D5, DR-26, DR-29): namespace `Gym.Box2D`, compiled into Gym.Environments.dll since DR-65 | NumSharp 0.81.0 (Python's exceptions) |

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
| `gymnasium.envs.box2d`, `mujoco` | `Gym.Environments.Envs.Box2D`, `Gym.Environments.Envs.MuJoCo` (the classes `AntEnvV4`/`AntEnvV5`... with Gymnasium's names as `[PyName]`, DR-66) |
| `mujoco` (the bindings Gymnasium's MuJoCo envs call) | `Gym.MuJoCo`: `MjModel`, `MjData`, `Mujoco.MjStep`/`MjForward`/..., `MjtState`, `MjtObj` (DR-68) |

## The env contract

`Env<TObs, TAct>` is the base of every env and wrapper:

- `Reset(BigInteger? seed, options)` → `ResetResult<TObs>(Observation, Info)`. Implementations start with
  `SeedNpRandom(seed)`, Gymnasium's `super().reset(seed=seed)`.
- `Step(TAct)` → `StepResult<TObs>(Observation, Reward, Terminated, Truncated, Info)`; rewards are `double` everywhere.
- `StepAsync` is a `ValueTask` that runs `Step` on the calling thread; an env with real asynchronous work overrides it.
  Wrappers do not forward it, so a wrapper's `Step` always applies.
- `Render()` renders in the mode fixed at construction; `human` mode draws inside `Reset` and `Step`. A single frame
  converts explicitly: `(NDArray)env.Render()` is the result's own frame, `(Image)env.Render()` a new ImageSharp image
  (DR-84).
- `RenderMode` is Gymnasium's string (`"rgb_array"`), any string Gymnasium accepts, a custom env's own modes included
  (DR-93). The `RenderMode` enum is how a mode is passed: each built-in env has its constructor taking the string and a
  hand-written twin taking the enum first (`new CartPoleEnv(RenderMode.RgbArray)`, DR-84); `RenderModes` names the strings
  and converts (`Name`, `Parse`, `TryParse`).
- `NpRandom` is the env's generator, created lazily from entropy, `NpRandomSeed` its seed (-1 once a generator is
  assigned); `Metadata`, `Spec`, `Unwrapped`, `Close()`/`Dispose()`.
- `Info` is Python's dict. `InfoExtensions.Get<T>` reads a value as a .NET type, converting NumPy scalars (0-d arrays)
  and numbers without losing them (DR-78). Its `Equals`, `==` and hash are by reference, as for any .NET collection, and
  Python's `==` is `PyEquals` (DR-92). Each built-in env class that sends info keys has a generated typed record,
  deriving from `EnvInfo` as the env classes derive from `Env` (`MujocoInfo`, `ToyTextInfo`; DR-91, DR-95): `double`
  members for scalars, `bool` for flags, `NDArray` for arrays, nullable where a reset or a step leaves the key out, the
  dict kept inside for its indexer, and value equality over the members (DR-94). The results' `Information` is 0.2's
  name for `Info`, obsolete until 2.0.
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
- **Enums**: Gymnasium's `AutoresetMode` and `VectorizeMode` are C# enums marked `[PyEnum]` (DR-84); `AutoresetModes` and
  `VectorizeModes` give a member's Python name, value, `str` and `repr` and parse `AutoresetMode("NextStep")` by value, and
  `PyBuiltins` and `PyEquality` treat a member as Python treats its counterpart (equal only to itself). The `RenderMode`
  enum is marked as standing for strings: `make`'s kwargs and the specs hold `"rgb_array"`, never the member.
- **Errors**: every Gymnasium error type and message. Every Python and NumPy exception is NumSharp 0.81.0's own class
  (`ValueError`, `TypeError`, `KeyError`, `AssertionError`, `OverflowError`, `ZeroDivisionError`, `NotImplementedError`,
  `ImportError`, ...; DR-76), each under its .NET twin where .NET has one (`ValueError` is an `ArgumentException`,
  `OverflowError` an `ArithmeticException`). `Gym.Exceptions` holds `gymnasium.error`'s classes only. .NET's own
  `OverflowException` remains for Gym.NET's limits where Python would not raise (a value beyond a C# `int` or `long`).
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
`Gym.Rendering.Clock` at the env's `render_fps`. `Gymnasium.Make` and `MakeVec` take the render mode and a viewer factory
(DR-89): the factory reaches the env, or the `HumanRendering` wrapper `make` applies to an env that only renders images,
and is never recorded in the spec.

The viewer contract is synchronous where the envs use it (DR-85): `EnvViewerFactory` returns the viewer once its window
is up, `IEnvViewer.Render(NDArray frame)` shows the `(H, W, 3)` uint8 frame (copied before it returns) and `Close()`
closes the window. A viewer that can also render without blocking implements `IAsyncEnvViewer` (`RenderAsync`,
`CloseAsync`), which an inherently asynchronous env awaits (`EnvViewerExtensions` falls back to the synchronous path for
any other viewer). Nothing blocks on a task, and ImageSharp is not part of the contract: the Avalonia and WinForms viewers
convert frames themselves (`src/Shared/ViewerFrames.cs`, compiled into both). 0.2's `CloseEnvironment()` is `[Obsolete]`
until 2.0.

The MuJoCo envs draw their frames with a renderer of their own (`Envs/MuJoCo/Rendering`, DR-62, DR-69): MuJoCo's classic
renderer in software, with the reference GPU's measured rounding, measured against Gymnasium's frames (DR-71). Their
`human` window is Gymnasium's MuJoCo viewer (DR-70): the mouse orbits, pans and zooms the camera, keys pause, step,
switch cameras, contact forces, frames, transparency and geom groups, save a frame or hide the menu MuJoCo's overlay
draws. It reads the viewer's input through `IInteractiveEnvViewer` (the Avalonia and WinForms viewers; another viewer
only shows the frames) and is paced at the display's 60 Hz, as GLFW's swap is, rather than at `render_fps`.

## Array ownership

NumSharp arrays live in unmanaged memory (D19). Whatever an env, space or wrapper returns belongs to the caller; spaces
and envs dispose what they own. A space has one owner (DR-40): the env it was given to, which disposes it with itself.
Handing it to a second env that is not yet disposed throws at once, naming the fix: give each env its own space, a new
one or `space.DeepCopy()`. A wrapper handed its own env's space passes it through without owning it, and a space an env
replaces is let go, undisposed. Results that carry arrays (`StepResult`, `ResetResult`, the vector results) implement
`INDArrayCarrier`, so NumSharp.Build's `[NDScoped]` weaving keeps them alive while it reclaims a method's temporaries.
NumSharp's analyzer reports arrays that are created and dropped (`NDW012`).

A caller may step an env inside its own NumSharp scope (an `[NDScoped]` method, the way the guides release what an env
returns, DR-80), which releases every array built while it was open when it closes. So every env, wrapper and vector
env that keeps an array it built during `Reset`, `Step` or `Render` (running statistics, frame stacks and delay queues,
collected frames and videos, a reset's goal, the last state pixels, a vector env's kept observations and flags) detaches
it from that scope where it stores it, and hands back through the caller's scope what it stops keeping and returns (the
popped frames): `ScopeSafety.Keep`/`HandOver` in Gym, `NDScope.Detach` in Gym.Environments (DR-96). `ScopeSafetyTests`
runs every registered id, render mode, wrapper and vector env in caller scopes against no scope, bit for bit, and fails
on any array an env still references after a scope disposed it. Envs and spaces are built outside the scoped method:
one built inside it and kept after it is not covered.

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
