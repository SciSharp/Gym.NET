# Gym.NET 1.0.0+rc1

Gym.NET 1.0 brings Farama Gymnasium v1.4.0's API and 39 of its environments (classic_control, box2d, toy_text and MuJoCo) to .NET: for the same seed and actions, they return Gymnasium's observations, rewards, flags and infos bit for bit and draw its frames pixel for pixel (MuJoCo's within a measured margin).
Box2D and MuJoCo run as managed C# inside the package, with no native code to install, and a parity suite that runs Gymnasium inside the test process checks every value and every frame.
It is the first release since 0.2.0 (March 2023), and it replaces 0.2's API, which followed gym before 0.26, with Gymnasium's.

### 🧭 Overview
- **Gymnasium v1.4.0 is the reference** - every registered id returns the same observations, rewards, flags and infos as Gymnasium's, bit for bit, and the classic_control, box2d and toy_text frames equal Gymnasium's pixel for pixel; a live parity suite runs Gymnasium inside the test process and compares every value and frame.
  - MuJoCo - the 22 MuJoCo ids step bit for bit with MuJoCo 4.0.0 on Gym.NET's own engine; Gym.NET draws their frames itself, and 0.07% to 16% of a frame's pixels differ from Gymnasium's, nearly all by one level.
- **39 environment ids** - classic_control (6), box2d (5), toy_text (6) and mujoco (22: eleven envs, each `-v4` and `-v5`), with Gymnasium's names and versions, registered for `Gymnasium.Make` before `Main` runs and named by constants (`Gymnasium.Envs.CartPoleV1`); each name below links to its page in Gymnasium v1.4.0's documentation.
  - classic_control - [CartPole](https://gymnasium.farama.org/v1.4.0/environments/classic_control/cart_pole/) (v0, v1), [MountainCar](https://gymnasium.farama.org/v1.4.0/environments/classic_control/mountain_car/), [MountainCarContinuous](https://gymnasium.farama.org/v1.4.0/environments/classic_control/mountain_car_continuous/), [Pendulum](https://gymnasium.farama.org/v1.4.0/environments/classic_control/pendulum/), [Acrobot](https://gymnasium.farama.org/v1.4.0/environments/classic_control/acrobot/); box2d - [LunarLander / LunarLanderContinuous](https://gymnasium.farama.org/v1.4.0/environments/box2d/lunar_lander/), [BipedalWalker / BipedalWalkerHardcore](https://gymnasium.farama.org/v1.4.0/environments/box2d/bipedal_walker/), [CarRacing](https://gymnasium.farama.org/v1.4.0/environments/box2d/car_racing/).
  - toy_text - [FrozenLake / FrozenLake8x8](https://gymnasium.farama.org/v1.4.0/environments/toy_text/frozen_lake/), [CliffWalking / CliffWalkingSlippery](https://gymnasium.farama.org/v1.4.0/environments/toy_text/cliff_walking/), [Taxi](https://gymnasium.farama.org/v1.4.0/environments/toy_text/taxi/), [Blackjack](https://gymnasium.farama.org/v1.4.0/environments/toy_text/blackjack/).
  - mujoco, each `-v4` and `-v5` - [Ant](https://gymnasium.farama.org/v1.4.0/environments/mujoco/ant/), [HalfCheetah](https://gymnasium.farama.org/v1.4.0/environments/mujoco/half_cheetah/), [Hopper](https://gymnasium.farama.org/v1.4.0/environments/mujoco/hopper/), [Humanoid](https://gymnasium.farama.org/v1.4.0/environments/mujoco/humanoid/), [HumanoidStandup](https://gymnasium.farama.org/v1.4.0/environments/mujoco/humanoid_standup/), [InvertedDoublePendulum](https://gymnasium.farama.org/v1.4.0/environments/mujoco/inverted_double_pendulum/), [InvertedPendulum](https://gymnasium.farama.org/v1.4.0/environments/mujoco/inverted_pendulum/), [Pusher](https://gymnasium.farama.org/v1.4.0/environments/mujoco/pusher/), [Reacher](https://gymnasium.farama.org/v1.4.0/environments/mujoco/reacher/), [Swimmer](https://gymnasium.farama.org/v1.4.0/environments/mujoco/swimmer/), [Walker2d](https://gymnasium.farama.org/v1.4.0/environments/mujoco/walker2d/).
- **Gymnasium's API in C#, 237 of Gymnasium v1.4.0's 301 public names** - `Env<TObs, TAct>`, `Reset(seed)` returning `(observation, info)` and `Step` returning `(observation, reward, terminated, truncated, info)`, the registry, all ten spaces, every wrapper but Atari's and the JAX/PyTorch conversions, sync and async vector envs, the env checkers and `play`.
  - The other 64 - 56 are Python-only (the JAX functional envs' 32 names, the JAX/PyTorch conversions' 21, the process workers' 2 and pickling's 1), 7 are provided another way by design, and 1 is out of scope (`AtariPreprocessing`).
- **.NET-typed where Gymnasium's shape is fixed** - `RenderMode`, `AutoresetMode` and `VectorizeMode` enums beside Gymnasium's strings, a typed info record per env class (`HalfCheetahInfoV5 info = step.Info;`), typed `Sample` masks, `Make` overloads taking a render mode and a viewer factory, and viewers with a synchronous and an asynchronous contract.
- **No native code** - Box2D 2.3.0 and a MuJoCo engine that computes what MuJoCo 4.0.0 computes are managed C#, inside **Gym.NET.Environments**; nothing to install beside the packages.
- **Faster than Gymnasium** - per step, Acrobot `x14.4`, Taxi `x4.5`, Pendulum `x3.8`, FrozenLake and BipedalWalker `x3.6`, MountainCar `x3.3`, CarRacing `x2.2`, LunarLander `x1.6` and CartPole `x1.4`, against Gymnasium on the same machine.
- **Memory released with `[NDScoped]`** - the guides and the 24 examples release what an env returns with NumSharp.Build's attribute, and every env, wrapper and vector env stays correct when stepped inside such a scope.
- **.NET 8, .NET 10 and .NET 11** - every package carries a build for each (the Windows Forms viewer's for Windows).
- **Release candidate 1, `1.0.0+rc1`** - NuGet ignores the `+rc1` build metadata, so to every NuGet client these are the stable 1.0.0 packages; whatever follows, a fix or the final, ships as 1.0.1.
- **14 deliberate differences from Gymnasium**, **1 known gap**, **9 environments measured faster than Gymnasium** and **16 breaking changes from 0.2.0**, across the 122 commits since 0.2.0; the migration guide maps every 0.2 call, and `Gym.Legacy` keeps 0.2's commonest lines compiling, `[Obsolete]`, until 2.0.
- **1 new NuGet package** - **Gym.NET.Rendering**.
- **4,097 unit tests and 2,922 live parity cells** pass on net8.0, net10.0 and net11.0, Gymnasium's own tests, rewritten in C#, among them.

### 📦 New NuGet Packages
- **Gym.NET.Rendering** - pygame's drawing in managed C#, so `rgb_array` frames equal Gymnasium's pixel for pixel, and Gymnasium's `play`.

## Detailed Breakdown
<details>
  <summary>Show All</summary>

### 📦 New NuGet Packages

One package ships for the first time, beside the four 0.2.0 shipped: **Gym.NET** (the API), **Gym.NET.Environments** (the environments, with the Box2D and MuJoCo engines), **Gym.NET.Rendering.Avalonia** and **Gym.NET.Rendering.WinForm** (the `human` windows). All five are co-versioned at `1.0.0+rc1` (decision D25), target .NET 8, .NET 10 and .NET 11, depend on [NumSharp](https://www.nuget.org/packages/NumSharp/0.81.0) 0.81.0, and are Apache-2.0, the repository's license (decision DR-49). Each carries THIRD-PARTY-NOTICES.md, which credits, with their notices, the sources Gym.NET translates (Box2D 2.3.0 and pybox2d, SDL_gfx, SDL_ttf, OpenCV's resize, CPython's difflib, json and math routines, pygame-ce's `draw.aaline` / `aalines`) and the data it carries (Gymnasium's assets, CarRacing's score glyphs, MuJoCo's font and compiled models).

- **Gym.NET.Rendering** - pygame's drawing in managed C#: the rasterizer the classic_control, box2d and toy_text envs draw their `rgb_array` frames with, equal to [pygame-ce](https://github.com/pygame-community/pygame-ce)'s pixels; it comes with **Gym.NET.Environments**.
  - Surfaces - a `Surface` with pygame's semantics, opaque or with per-pixel alpha, alpha blitting, `surfarray` (with `make_surface`) and `Vector2`.
  - Drawing - SDL_gfx 2.0.26's `gfxdraw` primitives, `draw.line`, `draw.rect`, `draw.polygon`, `draw.circle` and `draw.lines`, and `draw.aaline` / `draw.aalines`, translated from pygame-ce's draw.c.
  - Transforms and images - `transform.flip`, `transform.scale`, `transform.smoothscale`, `image.load` and `image.frombytes`.
  - Display and events - pygame's display and event queue as `PygameContext` instances, whose default the static `Display` and `EventQueue` forward to, process-wide like pygame's.
  - Playing - Gymnasium's `play`: `Play.Run`, `Play.RunAsync`, `Play.DisplayArr`, `PlayableGame` and `PlayPlot` (New APIs & Modules has the details).
  - Depends - **Gym.NET**, NumSharp 0.81.0, and [SixLabors.ImageSharp](https://www.nuget.org/packages/SixLabors.ImageSharp/2.1.13) 2.1.13, which decodes the PNG assets `image.load` reads (its advisories are under Breaking Changes).

### 📊 Dashboards & Docs
- **[Documentation site](https://scisharp.github.io/Gym.NET)** - the guides, the API reference, and every environment playing in its 8-bit gallery; every code sample in the guides was compiled and run against the tree.
  - [Getting started](https://scisharp.github.io/Gym.NET/docs/guides/getting_started.html) - making envs, seeding, spaces, rendering, wrappers and vector envs, and releasing arrays with `[NDScoped]`.
  - [Creating custom environments](https://scisharp.github.io/Gym.NET/docs/guides/creating_custom_environments.html) - an `Env<TObs, TAct>`, its registration, and checking it.
  - [Migrating from 0.2 to 1.0](https://scisharp.github.io/Gym.NET/docs/guides/migrating_to_1.0.html) - every 0.2 call and its replacement, and `Gym.Legacy` for code that cannot move yet.
  - [Architecture](https://scisharp.github.io/Gym.NET/docs/architecture.html) - the packages, and how Gym.NET is held to Gymnasium's behavior.
  - [Development and testing](https://scisharp.github.io/Gym.NET/docs/guides/development_and_testing.html) - building, the unit suite and the live parity suite.
- **[24 examples](https://github.com/SciSharp/Gym.NET/tree/master/examples)** - one project per environment the README's showreel plays, each playing a seeded episode with its agent (hand-written controllers, Gymnasium's heuristics, value iteration, ARS-trained linear policies), in a window with `--render`, and from the keyboard with `--play` where a few keys and the frame cost allow; CartPole's trains a cross-entropy method over 100 carts at once through `MakeVec`.

### ✨ New APIs & Modules
- `Env<TObs, TAct>` - Gymnasium's env contract, typed by observation and action: `Reset(seed, options)` returns `(observation, info)`, `Step` returns `(observation, reward, terminated, truncated, info)`, and `Render()` returns what the render mode fixed at construction asks for.
  - `IEnv`, `Wrapper` and the observation, action and reward wrapper bases - the untyped env and the wrapper family, with Python attributes read by Python name through wrapper stacks.
  - `StepResult`, `ResetResult` - disposable: `using var step = env.Step(action);` releases the arrays the result holds.
  - `Info`, `Metadata`, `RenderResult`, `RenderModes` - infos, metadata, frames, and the render modes as constants.
  - `StepAsync` - `ValueTask` stepping.
  - Every Gymnasium error with its messages, Gymnasium's logger warnings, and nullable annotations throughout.
- `Reset(seed)`, `Space.Seed` - Gymnasium's `Generator(PCG64(SeedSequence(seed)))`, bit for bit, for envs and spaces.
- `Box`, `Discrete`, `MultiBinary`, `MultiDiscrete`, `Tuple`, `Dict`, `Text`, `Sequence`, `Graph`, `OneOf` - all ten spaces, typed as `Space<T>`.
  - `Gym.Spaces.Spaces` - the `gymnasium.spaces` module as a static class (`Spaces.Discrete(4, seed: 0)`, `Spaces.Flatdim(space)`; DR-79, DR-86).
  - `SpaceUtils` - flatten, unflatten, flatdim and `is_space_dtype_shape_equiv`.
  - `batch_space`, `batch_differing_spaces`, `iterate`, `concatenate`, `create_empty_array`, `create_shared_memory`, `read_from_shared_memory`, `write_to_shared_memory` - the vector space utilities.
  - `Sample` - typed masks where Gymnasium's shape is fixed (`Discrete`, `MultiBinary`, `MultiDiscrete` as one array per entry, `Box.Sample()`; DR-97); a simple space's `Seed` returns a `BigInteger`, and `Clone()` is `DeepCopy()` (DR-81).
- `Gymnasium.Make`, `Gymnasium.MakeVec` - the registry, typed and untyped.
  - `Register`, `Spec`, `PprintRegistry` - namespaces and versions with Gymnasium's errors and "Did you mean" suggestions, and `EnvSpec` to and from JSON.
  - Constructor arguments by Python name - from anonymous objects, dictionaries, or generated `<Class>Options` records (`[GymOptions]`).
  - `Gymnasium.Envs` - the built-in ids, registered before `Main`, as constants (`Gymnasium.Make(Gymnasium.Envs.CartPoleV1)`, DR-88).
  - `Make(id, renderMode, viewerFactory)` - the render mode and the viewer factory per env (`Gymnasium.Make("CartPole-v1", RenderMode.Human, factory)`, DR-89).
- Every Gymnasium wrapper but `AtariPreprocessing` and the JAX/PyTorch conversions -
  - `TimeLimit`, `OrderEnforcing`, `PassiveEnvChecker`, `Autoreset`, `RecordEpisodeStatistics`, `RenderCollection` - the common wrappers.
  - `TransformObservation`, `FilterObservation`, `FlattenObservation`, `GrayscaleObservation`, `ResizeObservation`, `ReshapeObservation`, `RescaleObservation`, `DtypeObservation`, `AddRenderObservation`, `DiscretizeObservation` - the observation transforms, `ResizeObservation` over OpenCV's resize translated to C#.
  - `TimeAwareObservation`, `NormalizeObservation`, `DelayObservation`, `FrameStackObservation`, `MaxAndSkipObservation` - the stateful observation wrappers.
  - `TransformAction`, `ClipAction`, `RescaleAction`, `DiscretizeAction`, `StickyAction`, `RepeatAction` - the action wrappers.
  - `TransformReward`, `ClipReward`, `NormalizeReward` - the reward wrappers.
  - `RecordVideo`, `HumanRendering`, `AddWhiteNoise`, `ObstructView`, `save_video` - the rendering wrappers; `RecordVideo` writes through a pluggable `IVideoWriter`, PNG frames by default or mp4 through ffmpeg.
- `VectorEnv<TObs, TAct>` - vector envs, typed by the batched types.
  - `SyncVectorEnv` - in the three autoreset modes (`NextStep`, `SameStep`, `Disabled`).
  - `AsyncVectorEnv` - parallel sub-envs on worker threads, in three execution modes, with Gymnasium's `*_async` / `*_wait` protocol, `max_concurrency` and shared memory.
  - The vector wrapper bases and every vector wrapper of `gymnasium.wrappers.vector`, named `...Vector` and printing Gymnasium's names; `CartPoleVectorEnv`, CartPole's own vector env.
- `EnvChecker.CheckEnv`, `PassiveChecker`, `EnvMatch` - the checkers: Gymnasium's `check_env`, its passive checks and `check_environments_match`.
- `StepApiCompatibility`, `Performance`, `Video.SaveVideo`, `Cv2.Resize`, `FuncEnv` - the utilities: `step_api_compatibility` (single and vector), Gymnasium's benchmarks, `save_video`, OpenCV's resize, and the functional protocol (without JAX).
- `Play.Run` - Gymnasium's `play`, in a window from the viewer factory.
  - `PlayableGame`, `Play.DisplayArr`, `PlayPlot` - the game, `display_arr` and the live plot.
  - Input - the Avalonia and WinForms viewers report keys, their close button, resizes and the mouse (which a MuJoCo env's human window reads for Gymnasium's camera controls, DR-70); headless runs are driven by events posted to `Gym.Rendering.EventQueue`, as Gymnasium's tests post them.
  - `Play.RunAsync` - awaits a game, on a thread of its own for a window and on the thread pool headless; a `PygameContext` gives a game its own display and event queue, so two games can run at once.
  - `AvaloniaPlotFigure.Subplots()`, `WinFormPlotFigure.Subplots()` - `PlayPlot`'s charts with titles, ticks and labels from ScottPlot in the viewer packages; the core's plain `PlotFigure` needs no chart library.
- `Gymnasium.HumanViewerFactory` - the `human` viewers, chosen once for the app or per env, rendering synchronously, and the Avalonia and WinForms ones asynchronously too (`IAsyncEnvViewer`, DR-85).
  - `AvaloniaEnvViewer` - a window on Avalonia 12 (Windows, Linux, macOS), in **Gym.NET.Rendering.Avalonia**.
  - `WinFormEnvViewer` - the same on Windows Forms, in **Gym.NET.Rendering.WinForm**.
- `RenderMode`, `AutoresetMode`, `VectorizeMode` - C# enums that parse from and print as Gymnasium's strings (DR-84): every env takes its render mode as the enum or the string, `MakeVec` takes the vectorization and autoreset modes as enums (`MakeVec("CartPole-v1", 4, VectorizeMode.Async, AutoresetMode.SameStep)`), and the strings keep working everywhere Gymnasium takes them, specs and kwargs included.
- `HalfCheetahInfoV5` and 22 more info records - every built-in env class that sends info keys has a record of its info (DR-91): `HalfCheetahInfoV5 info = step.Info;` gives `info.XPosition` as a `double` and `info.XVelocity` as a `double?` (`null` on a reset's info), while `info["reward_ctrl"]` keeps the value as the env sent it; the records are named like their env classes and inherit as they do (`MujocoInfo`, `ToyTextInfo`).
- `[NDScoped]` - the guides release what an env returns with NumSharp.Build's attribute on the method that steps it (DR-80), `using` beside it; every built-in env, wrapper and vector env is safe to step in such a scope (DR-96): what it keeps from one step to the next (running statistics, frame stacks, delay queues, collected frames, a reset's goal) is detached from the caller's scope, and envs and spaces are built outside the scoped method.
- `Gym.Legacy` - 0.2's API for code that cannot move yet: `LegacyEnvAdapter` runs any 1.0 env behind the 0.2 interface, and everything in `Gym.Legacy` is `[Obsolete]` and is removed in 2.0.
  - 0.2's commonest lines still compile, each with an `[Obsolete]` warning naming the 1.0 spelling (DR-58), until 2.0: `var (obs, reward, done, info) = env.Step(a)` (done is terminated or truncated), `NDArray obs = env.Reset()`, `step.Information`, `env.CloseEnvironment()`, `new Box(low, high, np.float32)`, `new CartPoleEnv(viewerFactory)`, LunarLander's 0.2 names, and `catch` blocks of 0.2's three exceptions, now the bases of Gymnasium's `InvalidAction`, `AlreadyPendingCallError` and `NoAsyncCallError`; the migration guide lists each.
  - `Info` also reads like the legacy `Dict` (`Get<T>(key)`, `Clone()`).

### 🧩 Environments
- Every environment is registered for `Gymnasium.Make`, with `rgb_array` frames, `ansi` text for toy_text, and `human` mode.
- `CartPole-v0`, `CartPole-v1`, `MountainCar-v0`, `MountainCarContinuous-v0`, `Pendulum-v1`, `Acrobot-v1` - classic_control, frames pixel for pixel; CartPole-v1 also as `CartPoleVectorEnv`, its vector entry point for `Gymnasium.MakeVec`.
- `LunarLander-v3`, `LunarLanderContinuous-v3`, `BipedalWalker-v3`, `BipedalWalkerHardcore-v3`, `CarRacing-v3` - box2d, frames pixel for pixel, on `Gym.Box2D`: Box2D 2.3.0 in managed C#, bit for bit as pybox2d 2.3.10 computes it, with pybox2d's surface.
- `FrozenLake-v1`, `FrozenLake8x8-v1`, `CliffWalking-v1`, `CliffWalkingSlippery-v1`, `Taxi-v4`, `Blackjack-v1` - toy_text, frames pixel for pixel.
- `Ant`, `HalfCheetah`, `Hopper`, `Humanoid`, `HumanoidStandup`, `InvertedDoublePendulum`, `InvertedPendulum`, `Pusher`, `Reacher`, `Swimmer`, `Walker2d`, each `-v4` and `-v5` - mujoco, on `Gym.MuJoCo`: Gym.NET's own engine, bit for bit with MuJoCo 4.0.0 on Gymnasium's 14 bundled models, which it embeds as MuJoCo compiled them; observations, rewards, infos and their NumPy types match Gymnasium's bit for bit, and Pusher-v4 refuses as Gymnasium's does on MuJoCo 3 and later.
  - Frames - drawn on the CPU as MuJoCo's classic renderer draws its scene, reproducing the reference GPU's measured rounding and filtering: 0.07% (InvertedPendulum) to 16% (Humanoid's distant floor) of a frame's pixels differ from Gymnasium's, nearly all by one level, and a frame takes 0.05 to 0.5 s (DR-62, DR-67, DR-71).
  - `human` - a window with Gymnasium's mouse and key controls and its menu (DR-70), showing a few to a few dozen frames a second, single-sampled as Gymnasium's GLFW window's frames are.
  - `xml_file` - takes only Gymnasium's 14 bundled model names (DR-63): Gym.NET has no MJCF compiler.

### ⚡ Performance
*Ratios are Gymnasium ÷ Gym.NET time per step - higher is better (`x2` = twice Gymnasium's speed); `xLOW->xHIGH` spans the slowest→fastest env of a family. Measured 2026-11-11, before MuJoCo landed, on one machine: net10.0 Release builds against Gymnasium v1.4.0 on CPython 3.12.12 (NumPy 2.4.2, pygame-ce 2.5.8, Box2D 2.3.10), the same loop of sampled actions, five rounds of Gym.NET and four of Gymnasium interleaved, medians; the C# code review's [§4](https://github.com/SciSharp/Gym.NET/blob/master/docs/plans/CSHARP_CODE_REVIEW.md#4-measurements) has the table. MuJoCo's speed against native MuJoCo is not measured.*
- `x1.4->x14.4` - classic_control: Acrobot `x14.4`, Pendulum `x3.8`, MountainCar `x3.3`, CartPole `x1.4`.
- `x3.6->x4.5` - toy_text: Taxi `x4.5`, FrozenLake `x3.6`.
- `x1.6->x3.6` - box2d, on Gym.NET's managed Box2D: BipedalWalker `x3.6`, CarRacing `x2.2`, LunarLander `x1.6`.

### 🎯 Parity & Fixes
- Deliberate differences from Gymnasium - each documented on its member and pinned by the parity suite, so a change shows:
  - `double` rewards - everywhere: where Gymnasium returns a Python `int` (toy_text), an `np.float64` (Pendulum) or a float32 array (`CartPoleVectorEnv`), Gym.NET returns the same value as a `double`, or a float64 array.
  - `Taxi-v4` - its `reward_threshold` prints `8.0` where Gymnasium registers the `int` `8`.
  - `MountainCarContinuous-v0` - is `MountainCarEnv(continuous: true)` (one class for both MountainCars, DR-32): its spec's entry point and kwargs differ from Gymnasium's, while its `repr`, `Continuous_MountainCarEnv`, matches.
  - `Blackjack-v1` - its observations are Gym.NET's generic tuple, `object?[]`, which `BlackjackObservation` reads by name.
  - `SyncVectorEnv.NpRandom`, `NpRandomSeed` - the vector env's own generator and seed, where Gymnasium returns tuples of the sub-envs' (`GetAttr("np_random")` returns those); a `reset_mask` stays in the caller's options dictionary, where Python pops it.
  - `TimeAwareObservation` - its observations are `object`s, since their type depends on `flatten`.
  - `AsyncVectorEnv` - its workers are threads, not processes: `context` is accepted and has nothing to choose, `daemon` decides whether the worker threads are background threads, and a custom `worker` is an `AsyncVectorWorker` with Gymnasium's eight arguments, running on a thread of its own and talking through a `PipeConnection`, which passes values by reference instead of pickling them (`AsyncVectorEnv.AsyncWorker` is the default one to wrap); `close(terminate=True)` stops waiting instead of killing (each worker closes its sub-env as soon as its current call returns); a reset right after an episode end clears that sub-env's pending autoreset in both copy paths (DR-42), where Gymnasium's worker clears it only with shared memory, so without it Gymnasium's next step resets the sub-env a second time, with a reward of 0, while Gym.NET's sync and async envs agree in every sequence.
  - `DictInfoToList` - returns its list of infos from its own `Step` and `Reset`; through `IVectorEnv` the infos stay a dict.
  - `EnvRegistry.Namespace` - Gymnasium's `namespace` context manager applies to the thread and async flow that opens it (DR-57), where Gymnasium's `current_namespace` is one module global: two threads registering in scopes of their own at once each get their own namespace; on one thread nothing differs.
  - `RecordVideo` - writes PNG frames into a folder per video by default; `FfmpegVideoWriter` writes Gymnasium's mp4.
  - `Draw.Aalines` with two points - pygame-ce reads one float past its buffer there, so for a steep segment ending off a whole pixel its drawing varies from run to run, by one blend of the end column; Gym.NET takes that float as 0, what a fresh interpreter usually holds there, and no Gymnasium env draws a two-point polyline.
  - MuJoCo's `human` window - follows Gymnasium's `WindowViewer` where GLFW leaves nothing to choose, and decides the rest (DR-70): it draws at the viewer's client size, paces frames with a 60 Hz clock, opens at 960x540 when no size is given, writes T's capture under `/tmp` on the current drive (creating the folder), and after Esc or the close button draws nothing more.
  - Exceptions - Python's and NumPy's exceptions are NumSharp 0.81.0's own classes, `OverflowError` and `ZeroDivisionError` included, each under its .NET twin where .NET has one (`ValueError` is an `ArgumentException`); `InvalidAction`, `AlreadyPendingCallError` and `NoAsyncCallError` each have one of 0.2's obsolete exceptions as their base (DR-58), a level Gymnasium's hierarchy does not have, while their messages, and the Python types they stand for, are Gymnasium's.
  - Windows x64 is the reference platform - where Gymnasium's float64 arithmetic goes through the C runtime (CartPole's `cos` / `sin`, Box2D's and MuJoCo's math), results are verified on Windows, and on Linux and macOS the C library may differ in the last bit; the CPU counts too: on a CPU without FMA, .NET's `Math.Sin`, `Cos`, `Exp`, `Log`, `Atan2` and `Tanh` return other last bits (measured on a Xeon E5-2690 v2), so bit for bit is proven on the reference machine, an Intel i9-13900K.
- `Cv2.Resize` of float32 images with `INTER_LINEAR` - the one known gap, with a test that asserts Gymnasium's answer and fails until the gap closes (`[OpenBugs]`): Gym.NET gives OpenCV's own bilinear kernel, what `cv2` computes with `cv2.ipp.setUseIPP(False)` or on a build without Intel IPP, while the `cv2` wheels hand that case to IPP by default, whose algorithm is not published, and its results differ by a few units in the last place (DR-46); uint8 images, which Gymnasium's wrappers use, and float32 `INTER_AREA` are exact either way.
- NumSharp 0.81.0 - closed the four NumSharp gaps the 1.0 work found (N-1 to N-4): a bare `assert` fails with Python's empty message, `NotImplementedError` and `ImportError` are NumSharp's, every exception takes an inner exception, and `np.linspace` and the `'c'` dtype behave as NumPy's.
- Not in 1.0, by decision -
  - MuJoCo's frames - measured, not equal (DR-62, DR-67, DR-71), and a MuJoCo env's `xml_file` takes only the 14 bundled models (DR-63); Environments has the numbers.
  - `AtariPreprocessing` - it needs the Arcade Learning Environment (DR-36).
  - Native AOT and trimming (DR-55) - NumSharp compiles its array kernels at run time, so Gym.NET needs the JIT; a .NET 10 file-based app (`dotnet run app.cs`), which publishes AOT by default, sets `#:property PublishAot=false`.
  - Python-only stacks - JAX and PyTorch environments and converters, and pickling; Gymnasium also registers ids that only raise an `ImportError` pointing elsewhere (Shimmy's `GymV21Environment-v0` and `GymV26Environment-v0`, and mujoco-py's `-v2` / `-v3` MuJoCo ids), which Gym.NET does not register (port plan §1.4).
- `WinFormEnvViewer` - the window's client area is the frame's size, where 0.2's form cropped the frame's bottom rows behind the title bar.

### 🧰 Testing & Tooling
*Verified on 2026-11-21 on Windows x64, in Release, on net8.0, net10.0 and net11.0 alike, built with the .NET 11 SDK (11.0.100) against NumSharp 0.81.0, unless a line gives another date.*
- The live parity suite - embedded CPython runs Gymnasium v1.4.0 (Python 3.12, NumPy 2.4.2, pygame-ce 2.5.8, Box2D 2.3.10, OpenCV 4.13.0, MuJoCo 4.0.0) inside the test process and compares Gym.NET with it byte for byte and pixel for pixel: 2,922 of 2,922 gated cells pass, every registered id at every layer, and the one open cell fails only at its documented divergence, Intel IPP's float32 bilinear (DR-46).
  - `check_env`, `check_environments_match` - Gymnasium's own checkers, run through the reverse bridge on the 38 ids that build.
  - Goldens - 40 golden trajectories, replayed without Python.
  - MuJoCo - the frame and window cells, on the reference GPU (NVIDIA GeForce RTX 4060 Ti, driver 616.56).
- The unit suite - Gymnasium's own tests rewritten in C#, and Gym.NET's: 4,097 of 4,097 pass (24 data rows skip where Gymnasium's own tests call `pytest.skip`), and none is open since NumSharp 0.81.0 closed N-1 to N-4; six of Gymnasium's `tests/envs` modules are not mirrored yet (`test_make_vec.py`, `test_taxi.py`, `test_envs.py`, `test_action_dim_check.py`, `test_rendering.py`, `test_frozen_lake.py`).
- Scope safety - 141 of the unit cases step every registered id, render mode, wrapper and vector env inside a caller's NumSharp scope, against no scope, bit for bit (DR-96).
- The API inventory - Gymnasium v1.4.0's 301 public names, checked against Gym.NET's by reflection as a ratchet: 237 present, and each of the other 64 listed with its reason.
- The five packages at `1.0.0+rc1` - built from a fresh clone of the release commit with `-p:ContinuousIntegrationBuild=true` on SDK 11.0.100 (building packs them): each manifest carries `1.0.0+rc1` and the commit, the files are named `<id>.1.0.0.nupkg` (NuGet drops the metadata from file names), the packages depend on each other at 1.0.0, and the assemblies are 1.0.0.0 (assembly and file version) with the product version `1.0.0+rc1.<commit>` (the SDK appends the commit). Restored into an app on .NET 10 that references `Version="1.0.0"`, with an empty package cache, they resolve to this candidate, and CartPole, LunarLander, FrozenLake and HalfCheetah are made, stepped and rendered through them; the guides' samples of the 1.0 review (rendering, typed infos, `[NDScoped]`, the spaces module, the id constants) ran against the candidate's earlier build.
- The packages' earlier check (2026-11-12, at 1.0.0 with NumSharp 0.80.0) - restored into apps on .NET 10 and .NET 11, 12 ids from every family, MuJoCo's HalfCheetah, Humanoid, Pusher and Ant among them, were made, stepped and rendered, with the same frames on both runtimes; installed all together, every type of the five assemblies loads, both viewers implement the window and input contracts, and 0.2's common lines compile and run through the shims.
- Windows and examples - the 11 cells that open windows pass in a Windows 11 desktop (both toolkits, the viewer contract's exact pixels, sync and async); each of the 24 examples runs headless and prints the same steps and returns as before its `[NDScoped]` conversion, and CartPole's cross-entropy method solves CartPole-v1.
- The documentation site builds, here and on GitHub's runner (the Docs workflow), with Gym.NET's source generator built first, as DocFX needs it.

### 💥 Breaking Changes
- Every Gym.NET package moves to 1.0.0 at once - 0.2.0 declared its dependencies as "0.2.0 or later", so NuGet accepts a mix of 0.2.0 and 1.0.0 packages without a warning, and the mix fails when it runs: **Gym.NET.Environments** 0.2.0 over **Gym.NET** 1.0.0 cannot load its envs (`TypeLoadException: Could not load type 'Gym.Envs.Env'`), and a 0.2.0 viewer over **Gym.NET.Environments** 1.0.0 cannot load its window (`Could not load type 'Gym.Environments.IEnvViewer'`); **Gym.NET.Rendering**, new, comes with **Gym.NET.Environments**.
- NumSharp 0.81.0 replaces NumSharp.Lite 0.1.12 - remove NumSharp.Lite: with it still referenced, code naming `NDArray` stops compiling (CS0433: the type exists in both); `Discrete` samples are 0-d int64 arrays.
- .NET 8 or later - the packages carry .NET 8, .NET 10 and .NET 11 builds, where 0.2's targeted .NET Core 3.1 and .NET 6.
- Avalonia 12 - **Gym.NET.Rendering.Avalonia** is built on Avalonia 12, where 0.2's viewer was built on Avalonia 0.10.
- `NU1701` with **Gym.NET.Rendering.WinForm** - ScottPlot.WinForms, which draws `PlayPlot`'s charts, brings OpenTK 3.1.0, OpenTK.GLControl 3.1.0 and SkiaSharp.Views.WindowsForms 3.119.0, which ship .NET Framework builds only; a `NoWarn` on the package's own reference does not reach them, so a build that treats warnings as errors keeps the warning, but not as an error (the `WarningsNotAsErrors` line below).
- `NU1902`, `NU1903` from SixLabors.ImageSharp 2.1.13 - Gym.NET stays on ImageSharp's last release under the Apache-2.0 license (later versions ship under the Six Labors Split License), against which GitHub published five advisories on 2026-10-07, fixed only in 3.2.0 and 4.1.2; Gym.NET's own build suppresses them (decision D18), but your restore still reports them, and a build with `TreatWarningsAsErrors` fails on them. None is reachable through Gym.NET:
  - [GHSA-j3p4-wp97-rph4](https://github.com/advisories/GHSA-j3p4-wp97-rph4) (High) - `HistogramEqualization` uses an unvalidated luminance as a histogram index; Gym.NET never applies histogram equalization.
  - [GHSA-j9gm-c75j-xc9q](https://github.com/advisories/GHSA-j9gm-c75j-xc9q) (High) - the TIFF CCITT T4 encoder can write past its output buffer; Gym.NET never encodes TIFF.
  - [GHSA-jjfr-hcj7-qf5w](https://github.com/advisories/GHSA-jjfr-hcj7-qf5w) (High) - the TIFF CCITT T6 encoder writes beyond an undersized buffer; Gym.NET never encodes TIFF.
  - [GHSA-gwg2-r3hj-4w44](https://github.com/advisories/GHSA-gwg2-r3hj-4w44) (Moderate) - ICC CLUT parsing allocates from unvalidated dimensions; it is reached only by decoding an image with an embedded ICC profile, and Gym.NET decodes only its own embedded PNG assets.
  - [GHSA-wmxv-xphr-5c9g](https://github.com/advisories/GHSA-wmxv-xphr-5c9g) (Moderate) - a BigTIFF IFD count can stall a decoder thread; Gym.NET never decodes TIFF.
  - If your own code does not use ImageSharp on untrusted input, suppress them in your project (or `Directory.Build.props`) with the `NuGetAuditSuppress` lines below; a new advisory against 2.1.13 gets the same review before Gym.NET suppresses it.
- `Reset(seed, options)` returns `(observation, info)` - `Seed(s)` is gone; `Reset(seed: s)` replaces it.
- `Step` returns `(observation, reward, terminated, truncated, info)` - instead of `(observation, reward, done, info)`.
- `render_mode` - the render mode is a constructor argument (Gymnasium's string, or the `RenderMode` enum through each env's twin constructor and `Gymnasium.Make`'s overloads), and `Render()` returns a `RenderResult` (a frame as a uint8 array, or text) instead of an ImageSharp `Image`; `(Image)env.Render()` converts a frame when an image is wanted, and `human` mode draws inside `Reset` and `Step`.
- `IEnvViewer` - the viewer contract (DR-85): `IEnvViewer.Render(NDArray frame)` takes the `(H, W, 3)` uint8 frame instead of an `Image`, `Close()` replaces `CloseEnvironment()` (obsolete until 2.0), and the factory, `EnvViewerFactory`, is synchronous where 0.2's `IEnvironmentViewerFactoryDelegate` returned a task; a viewer that can render without blocking also implements `IAsyncEnvViewer`. Code that only passes `AvaloniaEnvViewer.Factory` or `WinFormEnvViewer.Factory` is unaffected; a viewer of your own changes its signatures.
- `Env<TObs, TAct>` - envs are `Env<TObs, TAct>`; `Env`, `Env<TAction>` and `GoalEnv` are gone, and `Gymnasium.Make` builds registered envs.
- `double` rewards, `Info` dicts - rewards are `double`; infos are `Info` dicts, compared by reference with Python's `==` as `PyEquals` (DR-92) and read typed through `info.Get<T>(key)`, which converts NumPy scalars (DR-78); metadata keys are `render_modes` and `render_fps`; 0.2's `step.Information` still compiles, obsolete, until 2.0.
- Seeds - they select Gymnasium's random streams, so seeded runs give Gymnasium's numbers, not 0.2's.
- `IVecEnv`, `VecEnv`, `DummyVecEnv`, `VecEnvWrapper` - the baselines-style vector types moved to `Gym.Legacy`.
- `CircularQueue<T>`, `CircularQueueThreadSafe<T>`, `GenericCallFactory` - 0.2's unused public types are gone (DR-56); the legacy `Dict`, which `Gym.Legacy` still returns, and its helpers in `Gym.Collections`, `Gym.Dynamic`, `Gym.Merging` and `Gym.Reflection` are `[Obsolete]`, each warning naming its replacement, and go in 2.0 with `Gym.Legacy`.
- `LunarLanderEnv` - 0.2's Aether-physics lander (`Gym.Environments.Envs.Aether`) and its `LunarLanderDiscreteActions` are gone, with the Aether.Physics2D, ImageSharp.Drawing and SixLabors.Fonts dependencies; Gymnasium's LunarLander-v3 on Gym.NET's Box2D replaces it (`Gym.Environments.Envs.Box2D.LunarLanderEnv`).

The lines for a build that treats warnings as errors, in your project or `Directory.Build.props` (the `NuGetAuditSuppress` items only if your own code does not use ImageSharp on untrusted input):

```xml
<PropertyGroup>
  <WarningsNotAsErrors>$(WarningsNotAsErrors);NU1701</WarningsNotAsErrors>
</PropertyGroup>
<ItemGroup>
  <NuGetAuditSuppress Include="https://github.com/advisories/GHSA-j3p4-wp97-rph4" />
  <NuGetAuditSuppress Include="https://github.com/advisories/GHSA-j9gm-c75j-xc9q" />
  <NuGetAuditSuppress Include="https://github.com/advisories/GHSA-jjfr-hcj7-qf5w" />
  <NuGetAuditSuppress Include="https://github.com/advisories/GHSA-gwg2-r3hj-4w44" />
  <NuGetAuditSuppress Include="https://github.com/advisories/GHSA-wmxv-xphr-5c9g" />
</ItemGroup>
```

</details>

---

### Install via NuGet

```
dotnet add package Gym.NET --version 1.0.0
dotnet add package Gym.NET.Environments --version 1.0.0
dotnet add package Gym.NET.Rendering --version 1.0.0
dotnet add package Gym.NET.Rendering.Avalonia --version 1.0.0
dotnet add package Gym.NET.Rendering.WinForm --version 1.0.0
```

### Packages

| Package | NuGet |
|---------|-------|
| Gym.NET | [![NuGet](https://img.shields.io/nuget/v/Gym.NET.svg)](https://www.nuget.org/packages/Gym.NET/1.0.0) |
| Gym.NET.Environments | [![NuGet](https://img.shields.io/nuget/v/Gym.NET.Environments.svg)](https://www.nuget.org/packages/Gym.NET.Environments/1.0.0) |
| Gym.NET.Rendering | [![NuGet](https://img.shields.io/nuget/v/Gym.NET.Rendering.svg)](https://www.nuget.org/packages/Gym.NET.Rendering/1.0.0) |
| Gym.NET.Rendering.Avalonia | [![NuGet](https://img.shields.io/nuget/v/Gym.NET.Rendering.Avalonia.svg)](https://www.nuget.org/packages/Gym.NET.Rendering.Avalonia/1.0.0) |
| Gym.NET.Rendering.WinForm | [![NuGet](https://img.shields.io/nuget/v/Gym.NET.Rendering.WinForm.svg)](https://www.nuget.org/packages/Gym.NET.Rendering.WinForm/1.0.0) |
