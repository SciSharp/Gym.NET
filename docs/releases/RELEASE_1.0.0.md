# Gym.NET 1.0.0

> **Prepared, not published yet.** Verified on 2026-11-21 on the working branch `journey1`; it ships when the
> maintainer merges it into `master`, tags it and publishes the packages (completion plan P10.5). Remove this note then.

Gym.NET 1.0 gives .NET reinforcement-learning environments with Farama Gymnasium's API and behavior. For the same seed
and actions, its environments return the same observations, rewards, flags and infos as Gymnasium v1.4.0's, bit for
bit, and draw the same frames, pixel for pixel (MuJoCo's within a measured margin, below); a test suite that runs
Gymnasium inside the test process checks it. It is the first release since 0.2.0 (March 2023), and it replaces 0.2's
API, which followed gym before 0.26, with Gymnasium's.

## Highlights

- **39 environment ids**, with Gymnasium's names and versions: classic_control (6), box2d (5), toy_text (6) and mujoco
  (22), registered for `Gymnasium.Make` before `Main` runs.
- **Gymnasium's API in C#**: `Env<TObs, TAct>`, `Reset(seed)` returning `(observation, info)`, `Step` returning
  `(observation, reward, terminated, truncated, info)`, the registry, all ten spaces, every wrapper but Atari's and the
  JAX/PyTorch conversions, and sync and async vector envs.
- **.NET 8, .NET 10 and .NET 11**: every package carries a build for each (the Windows Forms viewer's for Windows).
- **No native code**: Box2D 2.3.0 and a MuJoCo engine that computes what MuJoCo 4.0.0 computes are managed C#, inside
  Gym.NET.Environments; nothing to install beside the packages.
- **Checked against Gymnasium itself**: a live parity suite runs Gymnasium inside the test process and compares every
  value bit for bit and every frame pixel for pixel, and Gymnasium's own tests run in C# beside Gym.NET's (Verification,
  below).

## Packages

| Package | Contents |
|---|---|
| Gym.NET | The env contract, wrappers, spaces, registry (`Gymnasium.Make`, `MakeVec`), vector envs, checkers, seeding, errors; `Gym.Legacy` |
| Gym.NET.Rendering (new) | The rasterizer `rgb_array` frames are drawn with, equal to pygame-ce's pixels |
| Gym.NET.Environments | classic_control, toy_text, box2d (LunarLander, BipedalWalker, CarRacing) and mujoco (Ant, HalfCheetah, Hopper, Humanoid, HumanoidStandup, the two inverted pendulums, Pusher, Reacher, Swimmer, Walker2d), CartPole's vector env, their registration; inside its assembly, the `Gym.Box2D` namespace: Box2D 2.3.0 in managed C#, bit for bit as pybox2d 2.3.10, with pybox2d's surface, and the `Gym.MuJoCo` namespace: Gym.NET's own engine, bit for bit with MuJoCo 4.0.0 on Gymnasium's 14 bundled models, embedded as MuJoCo compiled them |
| Gym.NET.Rendering.Avalonia | A window for `human` mode on Avalonia 12 (Windows, Linux, macOS), and `PlayPlot`'s live charts on ScottPlot (`AvaloniaPlotFigure`) |
| Gym.NET.Rendering.WinForm | The same on Windows Forms (`WinFormPlotFigure`); its window's client area is the frame's size, where 0.2's form cropped the frame's bottom rows behind the title bar |

All target .NET 8, .NET 10 and .NET 11, depend on NumSharp 0.81.0, and are Apache-2.0, the repository's license (decision
DR-49). Each package carries THIRD-PARTY-NOTICES.md, which credits, with their notices, the sources Gym.NET translates
(Box2D 2.3.0 and pybox2d, SDL_gfx, SDL_ttf, OpenCV's resize, CPython's difflib, json and math routines, pygame-ce's
`draw.aaline`/`aalines`) and the data it carries (Gymnasium's assets, CarRacing's score glyphs, MuJoCo's font and
compiled models).

## Upgrading from 0.2

1. **Move every Gym.NET package to 1.0.0 at once.** 0.2.0 declared its dependencies as "0.2.0 or later", so NuGet
   accepts a mix of 0.2.0 and 1.0.0 packages without a warning, and the mix fails when it runs: Gym.NET.Environments
   0.2.0 over Gym.NET 1.0.0 cannot load its envs (`TypeLoadException: Could not load type 'Gym.Envs.Env'`), and a
   0.2.0 viewer over Gym.NET.Environments 1.0.0 cannot load its window (`Could not load type
   'Gym.Environments.IEnvViewer'`). Gym.NET.Rendering, new, comes with Gym.NET.Environments.
2. **Remove NumSharp.Lite.** 1.0 depends on NumSharp 0.81.0; with NumSharp.Lite 0.1.12 still referenced, code naming
   `NDArray` stops compiling (CS0433: the type exists in both).
3. **Target .NET 8 or later.** The packages carry .NET 8, .NET 10 and .NET 11 builds; 0.2's targeted .NET Core 3.1
   and .NET 6.
4. **Avalonia 12** for Gym.NET.Rendering.Avalonia; 0.2's viewer was built on Avalonia 0.10.
5. **Builds that treat warnings as errors** need two suppressions:
   - ImageSharp 2.1.13's five advisories, which NuGet reports as NU1902/NU1903 ([below](#sixlaborsimagesharp-2113-and-its-advisories)).
   - With Gym.NET.Rendering.WinForm, NU1701: ScottPlot.WinForms, which draws `PlayPlot`'s charts, brings OpenTK 3.1.0,
     OpenTK.GLControl 3.1.0 and SkiaSharp.Views.WindowsForms 3.119.0, which ship .NET Framework builds only. A `NoWarn`
     on the package's own reference does not reach them; keep the warning, but not as an error:

     ```xml
     <PropertyGroup>
       <WarningsNotAsErrors>$(WarningsNotAsErrors);NU1701</WarningsNotAsErrors>
     </PropertyGroup>
     ```
6. **Then the code.** The [migration guide](../guides/migrating_to_1.0.md) maps every 0.2 call; the commonest 0.2 lines
   still compile, with `[Obsolete]` warnings, until 2.0.

## Breaking changes

1.0 is Gymnasium's API, and 0.2 code needs changes; the [migration guide](../guides/migrating_to_1.0.md) maps every call.
In short:

- `Reset(seed, options)` returns `(observation, info)`; `Seed(s)` is gone, `Reset(seed: s)` replaces it.
- `Step` returns `(observation, reward, terminated, truncated, info)` instead of `(observation, reward, done, info)`.
- The render mode is a constructor argument (`render_mode`), and `Render()` returns a `RenderResult` (a frame as a
  uint8 array, or text) instead of an ImageSharp `Image`; `human` mode draws inside `Reset` and `Step`.
- Envs are `Env<TObs, TAct>`; `Env`, `Env<TAction>` and `GoalEnv` are gone; `Gymnasium.Make` builds registered envs.
- Rewards are `double`; infos are `Info` dicts; metadata keys are `render_modes` and `render_fps`.
- Seeds select Gymnasium's random streams, so seeded runs give Gymnasium's numbers, not 0.2's.
- NumSharp 0.81.0 replaces NumSharp.Lite 0.1.12; `Discrete` samples are 0-d int64 arrays.
- The baselines-style vector types (`IVecEnv`, `VecEnv`, `DummyVecEnv`, `VecEnvWrapper`) moved to `Gym.Legacy`.
- 0.2's unused public types are gone (DR-56): `CircularQueue<T>`, `CircularQueueThreadSafe<T>` and
  `GenericCallFactory`. The legacy `Dict`, which `Gym.Legacy` still returns, and its helpers in `Gym.Collections`,
  `Gym.Dynamic`, `Gym.Merging` and `Gym.Reflection` are `[Obsolete]`, each warning naming its replacement, and go in
  2.0 with `Gym.Legacy`.
- Common 0.2 lines still compile, each with an `[Obsolete]` warning naming the 1.0 spelling (DR-58), until 2.0:
  `var (obs, reward, done, info) = env.Step(a)` (done is terminated or truncated), `NDArray obs = env.Reset()`,
  `env.CloseEnvironment()`, `new Box(low, high, np.float32)`, `new CartPoleEnv(viewerFactory)`, LunarLander's 0.2
  names, and `catch` blocks of 0.2's three exceptions, now the bases of Gymnasium's `InvalidAction`,
  `AlreadyPendingCallError` and `NoAsyncCallError`. `Info` also reads like the legacy `Dict` (`Get<T>(key)`,
  `Clone()`). The migration guide lists each.
- The 0.2 `LunarLanderEnv` (Aether physics, `Gym.Environments.Envs.Aether`) and its `LunarLanderDiscreteActions` are
  gone, with the Aether.Physics2D, ImageSharp.Drawing and SixLabors.Fonts dependencies; Gymnasium's LunarLander-v3 on
  Gym.NET's Box2D replaces it (`Gym.Environments.Envs.Box2D.LunarLanderEnv`).

`Gym.Legacy.LegacyEnvAdapter` runs any 1.0 env behind the 0.2 interface for code that cannot move yet. Everything in
`Gym.Legacy` is `[Obsolete]` and is removed in 2.0.

## What 1.0 contains

- **Core**: `Env<TObs, TAct>` and `IEnv`, `Wrapper` with the observation, action and reward wrapper bases,
  `StepResult`/`ResetResult` (disposable: `using var step = env.Step(action);` releases the arrays the result holds),
  `Info`, `Metadata`, `RenderResult`, `RenderModes` (the render modes as constants), every Gymnasium error with its
  messages, Gymnasium's logger warnings, `ValueTask` stepping, Python attributes by Python name through wrapper stacks,
  nullable annotations throughout.
- **Seeding**: Gymnasium's `Generator(PCG64(SeedSequence(seed)))`, bit for bit, for envs and spaces.
- **Spaces**: all ten (`Box`, `Discrete`, `MultiBinary`, `MultiDiscrete`, `Tuple`, `Dict`, `Text`, `Sequence`, `Graph`,
  `OneOf`), typed as `Space<T>`, with `SpaceUtils` (flatten, unflatten, flatdim, `is_space_dtype_shape_equiv`) and the
  vector space utilities (`batch_space`, `batch_differing_spaces`, `iterate`, `concatenate`, `create_empty_array`,
  `create_shared_memory`, `read_from_shared_memory`, `write_to_shared_memory`).
- **Registry**: `Gymnasium.Make` and `Gymnasium.MakeVec`, typed and untyped; `Register`, `Spec`, `PprintRegistry`,
  namespaces and versions with Gymnasium's errors and "Did you mean" suggestions; `EnvSpec` to and from JSON; constructor
  arguments by Python name from anonymous objects, dictionaries, or generated `<Class>Options` records
  (`[GymOptions]`); the built-in ids registered before `Main`.
- **Wrappers**: every Gymnasium wrapper but `AtariPreprocessing` and the JAX/PyTorch conversions: `TimeLimit`,
  `OrderEnforcing`, `PassiveEnvChecker`, `Autoreset`, `RecordEpisodeStatistics`, `RenderCollection`; the observation
  transforms (`TransformObservation`, `FilterObservation`, `FlattenObservation`, `GrayscaleObservation`,
  `ResizeObservation` over OpenCV's resize translated to C#, `ReshapeObservation`, `RescaleObservation`,
  `DtypeObservation`, `AddRenderObservation`, `DiscretizeObservation`), the stateful ones (`TimeAwareObservation`,
  `NormalizeObservation`, `DelayObservation`, `FrameStackObservation`, `MaxAndSkipObservation`), the action and reward
  transforms (`TransformAction`, `ClipAction`, `RescaleAction`, `DiscretizeAction`, `StickyAction`, `RepeatAction`,
  `TransformReward`, `ClipReward`, `NormalizeReward`), and the rendering wrappers (`RecordVideo` over a pluggable
  `IVideoWriter`, PNG frames by default or mp4 through ffmpeg; `HumanRendering`; `AddWhiteNoise`; `ObstructView`), with
  `save_video`.
- **Vector envs**: `VectorEnv<TObs, TAct>` typed by the batched types, `SyncVectorEnv` in the three autoreset modes
  (`NextStep`, `SameStep`, `Disabled`), `AsyncVectorEnv` (parallel sub-envs on worker threads, in three execution modes,
  with Gymnasium's `*_async`/`*_wait` protocol, `max_concurrency` and shared memory), the vector wrapper bases, every
  vector wrapper of `gymnasium.wrappers.vector` (named `...Vector`, printing Gymnasium's names), `CartPoleVectorEnv`.
- **Checkers**: `EnvChecker.CheckEnv` (Gymnasium's `check_env`), `PassiveChecker`, `EnvMatch`.
- **Utilities**: `StepApiCompatibility` (`step_api_compatibility`, single and vector), `Performance` (Gymnasium's
  benchmarks), `Video.SaveVideo`, `Cv2.Resize`, and the functional protocol `FuncEnv` (without JAX).
- **Playing from the keyboard**: `Play.Run` (Gymnasium's `play`), `PlayableGame`, `Play.DisplayArr` and `PlayPlot`, in
  a window from the viewer factory; the Avalonia and WinForms viewers report keys, their close button, resizes and the
  mouse (which a MuJoCo env's human window reads for Gymnasium's camera controls, DR-70), and headless runs are driven
  by events posted to `Gym.Rendering.EventQueue`, as Gymnasium's tests post them. `Play.RunAsync` awaits a game, on a
  thread of its own for a window and on the thread pool headless; a `PygameContext` gives a game its own display and
  event queue, so two games can run at once. `PlayPlot`'s charts get titles, ticks and labels from ScottPlot in the
  viewer packages (`AvaloniaPlotFigure.Subplots()`, `WinFormPlotFigure.Subplots()`); the core's plain `PlotFigure` needs
  no chart library.
- **Environments** (all registered for `Gymnasium.Make`, each with `rgb_array` frames, `ansi` text for toy_text, and
  `human` mode):

  | Family | Ids |
  |---|---|
  | classic_control | CartPole-v0, CartPole-v1 (and its vector env), MountainCar-v0, MountainCarContinuous-v0, Pendulum-v1, Acrobot-v1 |
  | box2d | LunarLander-v3, LunarLanderContinuous-v3, BipedalWalker-v3, BipedalWalkerHardcore-v3, CarRacing-v3 (on `Gym.Box2D`, Box2D 2.3.0 in managed C#) |
  | toy_text | FrozenLake-v1, FrozenLake8x8-v1, CliffWalking-v1, CliffWalkingSlippery-v1, Taxi-v4, Blackjack-v1 |
  | mujoco | Ant, HalfCheetah, Hopper, Humanoid, HumanoidStandup, InvertedDoublePendulum, InvertedPendulum, Pusher, Reacher, Swimmer and Walker2d, each -v4 and -v5 (on `Gym.MuJoCo`, Gym.NET's engine following MuJoCo 4.0.0; Pusher-v4 refuses as Gymnasium's does on MuJoCo 3 and later). Observations, rewards, infos and their NumPy types match Gymnasium's bit for bit; Gym.NET draws the frames itself, and 0.07% (InvertedPendulum) to 16% (Humanoid's distant floor) of a frame's pixels differ from Gymnasium's, nearly all by one level (DR-62, DR-71); `human` opens a window with Gymnasium's mouse and key controls and its menu (DR-70) |

  The classic_control, toy_text and box2d frames equal Gymnasium's pixel for pixel. The nine of those envs measured
  (2026-11-11, before MuJoCo landed) step faster than Gymnasium's on the same machine (Release builds, sampled actions,
  the medians of interleaved runs): 1.4× CartPole to 14× Acrobot, LunarLander 1.6×, BipedalWalker 3.6×, CarRacing 2.2×
  (the C# code review's §4 has the table). MuJoCo's speed against native MuJoCo is not measured.
- **Rendering**: `Gym.Rendering`, a rasterizer with pygame's semantics (SDL_gfx's `gfxdraw` primitives, `draw.line`,
  `draw.rect`, `draw.polygon`, `draw.circle`, `draw.lines`, alpha blitting, `transform.flip`/`scale`/`smoothscale`,
  `image.load`), MuJoCo's scene drawn in software as its classic renderer draws it, and the `human` viewers, chosen once
  with `Gymnasium.HumanViewerFactory`.

## Differences from Gymnasium, by design

Each is documented on the member, and the parity suite pins it, so a change shows:

- **Rewards are `double`** everywhere: where Gymnasium returns a Python `int` (toy_text), an `np.float64` (Pendulum) or a
  float32 array (`CartPoleVectorEnv`), Gym.NET returns the same value as a `double`, or a float64 array.
- **Taxi-v4's `reward_threshold`** prints `8.0` where Gymnasium registers the `int` `8`.
- **MountainCarContinuous-v0** is `MountainCarEnv(continuous: true)` (one class for both MountainCars, DR-32): its spec's
  entry point and kwargs differ from Gymnasium's, while its `repr`, `Continuous_MountainCarEnv`, matches.
- **Blackjack's observations** are Gym.NET's generic tuple, `object?[]`, which `BlackjackObservation` reads by name.
- **`SyncVectorEnv.NpRandom`/`NpRandomSeed`** are the vector env's own generator and seed, where Gymnasium returns tuples
  of the sub-envs' (`GetAttr("np_random")` returns those); a `reset_mask` stays in the caller's options dictionary,
  where Python pops it.
- **`TimeAwareObservation`'s observations** are `object`s, since their type depends on `flatten`.
- **`AsyncVectorEnv`'s workers are threads**, not processes. `context` is accepted and has nothing to choose, and
  `daemon` decides whether the worker threads are background threads. A custom `worker` is an `AsyncVectorWorker` with
  Gymnasium's eight arguments; it runs on a thread of its own and talks through a `PipeConnection`, which passes values
  by reference instead of pickling them (`AsyncVectorEnv.AsyncWorker` is the default one to wrap). `close(terminate=True)`
  stops waiting instead of killing (each worker closes its sub-env as soon as its current call returns). A reset right
  after an episode end clears that sub-env's pending autoreset in both copy paths (DR-42). Gymnasium's worker clears it
  only with shared memory, so without it Gymnasium's next step resets the sub-env a second time, with a reward of 0;
  Gym.NET's sync and async envs agree in every sequence instead.
- **`DictInfoToList`** returns its list of infos from its own `Step` and `Reset`; through `IVectorEnv` the infos stay a
  dict.
- **The registry's namespace scope** (`EnvRegistry.Namespace`, Gymnasium's `namespace` context manager) applies to the
  thread and async flow that opens it (DR-57), where Gymnasium's `current_namespace` is one module global: two threads
  registering in scopes of their own at once each get their own namespace. On one thread nothing differs.
- **`RecordVideo` writes PNG frames** into a folder per video by default; `FfmpegVideoWriter` writes Gymnasium's mp4.
- **`Draw.Aalines` with two points**: pygame-ce reads one float past its buffer there, so for a steep segment ending off
  a whole pixel its drawing varies from run to run, by one blend of the end column. Gym.NET takes that float as 0, what a
  fresh interpreter usually holds there. No Gymnasium env draws a two-point polyline.
- **MuJoCo's human window** follows Gymnasium's `WindowViewer` where GLFW leaves nothing to choose, and decides the rest
  (DR-70): it draws at the viewer's client size, paces frames with a 60 Hz clock, opens at 960 × 540 when no size is
  given, writes T's capture under `/tmp` on the current drive (creating the folder), and after Esc or the close button
  draws nothing more.
- **Exceptions**: Python's and NumPy's exceptions are NumSharp 0.81.0's own classes, `OverflowError` and
  `ZeroDivisionError` included, each under its .NET twin where .NET has one (`ValueError` is an `ArgumentException`).
  `InvalidAction`, `AlreadyPendingCallError` and `NoAsyncCallError` each have one of 0.2's obsolete exceptions as their
  base (DR-58), a level Gymnasium's hierarchy does not have; their messages, and the Python types they stand for, are
  Gymnasium's.
- **The reference platform is Windows x64.** Where Gymnasium's float64 arithmetic goes through the C runtime (CartPole's
  `cos`/`sin`, Box2D's and MuJoCo's math), results are verified on Windows; on Linux and macOS the C library may differ
  in the last bit.

## Known gaps

One, with a test that asserts Gymnasium's answer and fails until the gap closes (`[OpenBugs]`):

- **`Cv2.Resize` of float32 images with `INTER_LINEAR`** gives OpenCV's own bilinear kernel, what `cv2` computes with
  `cv2.ipp.setUseIPP(False)` or on a build without Intel IPP. The `cv2` wheels hand that case to IPP by default, whose
  algorithm is not published, and its results differ by a few units in the last place (DR-46). uint8 images, which
  Gymnasium's wrappers use, and float32 `INTER_AREA` are exact either way.

NumSharp 0.81.0 closed the four NumSharp gaps the 1.0 work found (N-1 to N-4): a bare `assert` fails with Python's
empty message, `NotImplementedError` and `ImportError` are NumSharp's, every exception takes an inner exception, and
`np.linspace` and the `'c'` dtype behave as NumPy's.

## Not in 1.0

Out of 1.0 by decision:

- **MuJoCo's frames are measured, not equal** (DR-62, DR-67, DR-71): Gym.NET draws them on the CPU, reproducing the
  reference GPU's measured rounding and filtering, so a frame differs from Gymnasium's in 0.07% (InvertedPendulum) to
  16% (Humanoid's distant floor) of its pixels, nearly all by one level, and takes 0.05 to 0.5 s; so the `human` window
  (DR-70) shows a few to a few dozen frames a second, and its frames are single-sampled as Gymnasium's GLFW window's
  are. A MuJoCo env's `xml_file` takes only Gymnasium's 14 bundled model names (DR-63): Gym.NET has no MJCF compiler.
- `AtariPreprocessing`, which needs the Arcade Learning Environment (DR-36).
- **Native AOT and trimming** (DR-55): NumSharp compiles its array kernels at run time, so Gym.NET needs the JIT. A .NET
  10 file-based app (`dotnet run app.cs`), which publishes AOT by default, sets `#:property PublishAot=false`.
- Python-only stacks: JAX and PyTorch environments and converters, pickling. Gymnasium also registers ids that only
  raise an `ImportError` pointing elsewhere (Shimmy's `GymV21Environment-v0` and `GymV26Environment-v0`, and mujoco-py's
  `-v2`/`-v3` MuJoCo ids); Gym.NET does not register them (port plan §1.4).

## SixLabors.ImageSharp 2.1.13 and its advisories

Gym.NET depends on SixLabors.ImageSharp 2.1.13, its last release under the Apache-2.0 license; later versions ship under
the Six Labors Split License. On 2026-10-07 GitHub published five advisories against 2.1.13, fixed only in 3.2.0 and
4.1.2:

| Advisory | Severity | Why Gym.NET does not reach it |
|---|---|---|
| [GHSA-j3p4-wp97-rph4](https://github.com/advisories/GHSA-j3p4-wp97-rph4): `HistogramEqualization` uses an unvalidated luminance as a histogram index | High | Gym.NET never applies histogram equalization |
| [GHSA-j9gm-c75j-xc9q](https://github.com/advisories/GHSA-j9gm-c75j-xc9q): the TIFF CCITT T4 encoder can write past its output buffer | High | Gym.NET never encodes TIFF |
| [GHSA-jjfr-hcj7-qf5w](https://github.com/advisories/GHSA-jjfr-hcj7-qf5w): the TIFF CCITT T6 encoder writes beyond an undersized buffer | High | Gym.NET never encodes TIFF |
| [GHSA-gwg2-r3hj-4w44](https://github.com/advisories/GHSA-gwg2-r3hj-4w44): ICC CLUT parsing allocates from unvalidated dimensions | Moderate | Reached only by decoding an image with an embedded ICC profile; Gym.NET decodes only its own embedded PNG assets |
| [GHSA-wmxv-xphr-5c9g](https://github.com/advisories/GHSA-wmxv-xphr-5c9g): a BigTIFF IFD count can stall a decoder thread | Moderate | Gym.NET never decodes TIFF |

Gym.NET's own build suppresses them (decision D18), but your restore still reports them as NU1902/NU1903 warnings, and a
build with `TreatWarningsAsErrors` fails on them. If your own code does not use ImageSharp on untrusted input, suppress
them in your project (or `Directory.Build.props`):

```xml
<ItemGroup>
  <NuGetAuditSuppress Include="https://github.com/advisories/GHSA-j3p4-wp97-rph4" />
  <NuGetAuditSuppress Include="https://github.com/advisories/GHSA-j9gm-c75j-xc9q" />
  <NuGetAuditSuppress Include="https://github.com/advisories/GHSA-jjfr-hcj7-qf5w" />
  <NuGetAuditSuppress Include="https://github.com/advisories/GHSA-gwg2-r3hj-4w44" />
  <NuGetAuditSuppress Include="https://github.com/advisories/GHSA-wmxv-xphr-5c9g" />
</ItemGroup>
```

A new advisory against 2.1.13 gets the same review before Gym.NET suppresses it.

## Verification

On 2026-11-21, on Windows x64, in Release, on net8.0, net10.0 and net11.0 alike (built with the .NET 11 SDK, 11.0.100,
against NumSharp 0.81.0):

- **The unit suite** (Gymnasium's own tests rewritten in C#, and Gym.NET's): 3,904 of 3,904 pass (24 data rows skip
  where Gymnasium's own tests call `pytest.skip`); none is open since NumSharp 0.81.0 closed N-1 to N-4.
- **The live parity suite**, against Gymnasium v1.4.0 on Python 3.12, NumPy 2.4.2, pygame-ce 2.5.8, Box2D 2.3.10,
  OpenCV 4.13.0 and MuJoCo 4.0.0: 2,922 of 2,922 gated cells pass, every registered id at every layer. The one open cell
  fails only at its documented divergence, Intel IPP's float32 bilinear (DR-46). The gate includes:
  - Gymnasium's own `check_env` and `check_environments_match`, run through the reverse bridge on the 38 ids that build;
  - the 40 golden trajectories, replayed without Python;
  - the MuJoCo frame and window cells, on the reference GPU (NVIDIA GeForce RTX 4060 Ti, driver 616.56).
- **The five packages** (checked on 2026-11-12, with NumSharp 0.80.0; redone at the release), packed with a build for
  each framework, and restored from a local feed into apps of their own on .NET 10 and on .NET 11:
  - 12 ids from every family, MuJoCo's HalfCheetah, Humanoid, Pusher and Ant among them, made, stepped and rendered,
    the same frames on both runtimes;
  - installed all together, every type of the five assemblies loads, both viewers implement the window and input
    contracts, and 0.2's common lines compile and run through the shims.
- **The rest** (2026-11-12): the CartPole example solves CartPole-v1 in seven generations, and the documentation site
  builds.
