# Gym.NET 2.0.0

> **Draft, not released.** Prepared on 2026-11-09 on the working branch `journey1`; nothing is merged into `master`,
> tagged or published. Before the release (completion plan P10.5):
>
> - **P8, the Box2D environments** (LunarLander-v3, LunarLanderContinuous-v3, BipedalWalker-v3,
>   BipedalWalkerHardcore-v3, CarRacing-v3), which the approved 2.0 scope includes (D21).
> - The exit check: the inventory's list of missing names holds only MuJoCo's names and `AtariPreprocessing`.
>
> Update this page when any of these lands; the numbers under "Verification" are those of the preparing commit.

Gym.NET 2.0 gives .NET reinforcement-learning environments with Farama Gymnasium's API and behavior. For the same
seed and actions, its environments return the same observations, rewards, flags and infos as Gymnasium v1.4.0's, bit
for bit, and draw the same frames, pixel for pixel; a test suite that runs Gymnasium inside the test process checks it.
It replaces 0.2's API, which followed gym before 0.26, with Gymnasium's.

## Packages

| Package | Contents |
|---|---|
| Gym.NET | The env contract, wrappers, spaces, registry (`Gymnasium.Make`, `MakeVec`), vector envs, checkers, seeding, errors; `Gym.Legacy` |
| Gym.NET.Rendering (new) | The rasterizer `rgb_array` frames are drawn with, equal to pygame-ce's pixels; licensed `Apache-2.0 AND LGPL-2.0-or-later` (below) |
| Gym.NET.Environments | classic_control, toy_text and box2d (LunarLander, BipedalWalker, CarRacing), CartPole's vector env, their registration; inside it, `Gym.Box2D`: Box2D 2.3.0 in managed C#, bit for bit as pybox2d 2.3.10, with pybox2d's surface (zlib notices in THIRD-PARTY-NOTICES.md) |
| Gym.NET.Rendering.Avalonia | A window for `human` mode on Avalonia 12 (Windows, Linux, macOS), and `PlayPlot`'s live charts on ScottPlot (`AvaloniaPlotFigure`) |
| Gym.NET.Rendering.WinForm | The same on Windows Forms (`WinFormPlotFigure`) |

All target .NET 8 and .NET 10, and depend on NumSharp 0.80.0. All are Apache-2.0 except Gym.NET.Rendering, which is
`Apache-2.0 AND LGPL-2.0-or-later`: its `draw.aaline`/`aalines` (MountainCar's hill) are translated from pygame-ce's
`draw.c` and keep pygame-ce's license, the GNU Library GPL version 2 or later (decision DR-37; the notice is in
THIRD-PARTY-NOTICES.md).

## Breaking changes

2.0 is Gymnasium's API, and 0.2 code needs changes; the [migration guide](../guides/migrating_to_2.0.md) maps every call.
In short:

- `Reset(seed, options)` returns `(observation, info)`; `Seed(s)` is gone, `Reset(seed: s)` replaces it.
- `Step` returns `(observation, reward, terminated, truncated, info)` instead of `(observation, reward, done, info)`.
- The render mode is a constructor argument (`render_mode`), and `Render()` returns a `RenderResult` (a frame as a
  uint8 array, or text) instead of an ImageSharp `Image`; `human` mode draws inside `Reset` and `Step`.
- Envs are `Env<TObs, TAct>`; `Env`, `Env<TAction>` and `GoalEnv` are gone; `Gymnasium.Make` builds registered envs.
- Rewards are `double`; infos are `Info` dicts; metadata keys are `render_modes` and `render_fps`.
- Seeds select Gymnasium's random streams, so seeded runs give Gymnasium's numbers, not 0.2's.
- NumSharp 0.80.0 replaces NumSharp.Lite 0.1.12; `Discrete` samples are 0-d int64 arrays.
- The baselines-style vector types (`IVecEnv`, `VecEnv`, `DummyVecEnv`, `VecEnvWrapper`) moved to `Gym.Legacy`.
- The 0.2 `LunarLanderEnv` (Aether physics, `Gym.Environments.Envs.Aether`) and its `LunarLanderDiscreteActions` are
  gone, with the Aether.Physics2D, ImageSharp.Drawing and SixLabors.Fonts dependencies; Gymnasium's LunarLander-v3 on
  Gym.NET's Box2D replaces it (`Gym.Environments.Envs.Box2D.LunarLanderEnv`).

`Gym.Legacy.LegacyEnvAdapter` runs any 2.0 env behind the 0.2 interface for code that cannot move yet. Everything in
`Gym.Legacy` is `[Obsolete]` and is removed in 3.0.

## What 2.0 contains

- **Core**: `Env<TObs, TAct>` and `IEnv`, `Wrapper` with the observation, action and reward wrapper bases,
  `StepResult`/`ResetResult`, `Info`, `Metadata`, `RenderResult`, every Gymnasium error with its messages, Gymnasium's
  logger warnings, `ValueTask` stepping, Python attributes by Python name through wrapper stacks.
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
  a window from the viewer factory; the Avalonia and WinForms viewers report keys, their close button and resizes, and
  headless runs are driven by events posted to `Gym.Rendering.EventQueue`, as Gymnasium's tests post them.
  `Play.RunAsync` awaits a game, on a thread of its own for a window and on the thread pool headless; a
  `PygameContext` gives a game its own display and event queue, so two games can run at once. `PlayPlot`'s charts get
  titles, ticks and labels from ScottPlot in the viewer packages (`AvaloniaPlotFigure.Subplots()`,
  `WinFormPlotFigure.Subplots()`); the core's plain `PlotFigure` needs no chart library.
- **Environments** (all registered for `Gymnasium.Make`, each with `rgb_array` frames equal to Gymnasium's, `ansi` text
  for toy_text, and `human` mode):

  | Family | Ids |
  |---|---|
  | classic_control | CartPole-v0, CartPole-v1 (and its vector env), MountainCar-v0, MountainCarContinuous-v0, Pendulum-v1, Acrobot-v1 |
  | box2d | LunarLander-v3, LunarLanderContinuous-v3, BipedalWalker-v3, BipedalWalkerHardcore-v3, CarRacing-v3 (on `Gym.Box2D`, Box2D 2.3.0 in managed C#) |
  | toy_text | FrozenLake-v1, FrozenLake8x8-v1, CliffWalking-v1, CliffWalkingSlippery-v1, Taxi-v4, Blackjack-v1 |

- **Rendering**: `Gym.Rendering`, a rasterizer with pygame's semantics (SDL_gfx's `gfxdraw` primitives, `draw.line`,
  `draw.rect`, `draw.polygon`, `draw.circle`, `draw.lines`, alpha blitting, `transform.flip`/`scale`/`smoothscale`, `image.load`), and the `human` viewers, chosen
  once with `Gymnasium.HumanViewerFactory`.

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
  stops waiting instead of killing (each worker closes its sub-env as soon as its current call returns). A reset right after an episode end clears that sub-env's pending autoreset in
  both copy paths (DR-42). Gymnasium's worker clears it only with shared memory, so without it Gymnasium's next step
  resets the sub-env a second time, with a reward of 0; Gym.NET's sync and async envs agree in every sequence instead.
- **`DictInfoToList`** returns its list of infos from its own `Step` and `Reset`; through `IVectorEnv` the infos stay a
  dict.
- **`RecordVideo` writes PNG frames** into a folder per video by default; `FfmpegVideoWriter` writes Gymnasium's mp4.
- **`Draw.Aalines` with two points**: pygame-ce reads one float past its buffer there, so for a steep segment ending off
  a whole pixel its drawing varies from run to run, by one blend of the end column. Gym.NET takes that float as 0, what a
  fresh interpreter usually holds there. No Gymnasium env draws a two-point polyline.
- **Exceptions**: Python's built-in exceptions are NumSharp's types; `OverflowError` and `ZeroDivisionError` are .NET's
  `OverflowException` and `DivideByZeroException`.
- **The reference platform is Windows x64.** Where Gymnasium's float64 arithmetic goes through the C runtime (CartPole's
  `cos`/`sin`), results are verified on Windows; on Linux and macOS the C library may differ in the last bit.

## Known gaps

Each has a test that asserts Gymnasium's answer and fails until the gap closes (`[OpenBugs]`):

- **`Cv2.Resize` of float32 images with `INTER_LINEAR`** gives OpenCV's own bilinear kernel, what `cv2` computes with
  `cv2.ipp.setUseIPP(False)` or on a build without Intel IPP. The `cv2` wheels hand that case to IPP by default, whose
  algorithm is not published, and its results differ by a few units in the last place (DR-46). uint8 images, which
  Gymnasium's wrappers use, and float32 `INTER_AREA` are exact either way.

- **A bare `assert`'s message**: where Gymnasium fails a message-less `assert` (Taxi's `decode` beyond the grid,
  Blackjack's refused actions, `render()` without a render mode or spec), Gym.NET raises `AssertionError` too, but its
  message is NumSharp's `"AssertionError"` instead of Python's empty one (NumSharp gap N-1).
- **`NotImplementedError` and `ImportError`** are `Gym.Exceptions` types until NumSharp defines them, and NumSharp's
  exceptions take no inner exception yet (N-3).
- **CarRacing-v3's score text** is drawn from glyph data of pygame's default font (GNU FreeFont's FreeSans Bold),
  recorded from live pygame and laid out as SDL_ttf 2.24 lays it out, exact on every score string. The data does not
  ship until the maintainer decides it may (the font's embedded licence is the GPL; DR-27's licence check), so the
  packaged env leaves the score out: its observations and frames match Gymnasium's everywhere but the score's few
  pixels. The parity suite installs the recorded data and proves the rest.

## Not in 2.0 yet

Nothing of P6 is deferred any more; the Box2D environments (P8) are still to come, as the note at the top says.

Out of 2.0 by decision:

- **MuJoCo's environments** are postponed (DR-35). When they resume, they become a fully managed port inside
  Gym.NET.Environments, with no native binaries (DR-30).
- `AtariPreprocessing`, which needs the Arcade Learning Environment (DR-36).
- Python-only stacks: JAX and PyTorch environments and converters, pickling, Shimmy's compatibility ids.

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

At the preparing commit, on Windows x64, in Release, on net8.0 and net10.0 alike:

- the unit suite (Gymnasium's own tests rewritten in C#, and Gym.NET's): 1225 of 1225 pass, and the 10 open tests fail
  only at their documented gaps (NumSharp's N-1, N-2 and N-3, and `make`'s human rendering);
- the live parity suite against Gymnasium v1.4.0 on Python 3.12, NumPy 2.4.2 and pygame-ce 2.5.8: 1783 of 1783 gated
  cells pass, and the 15 open cells fail only at their documented divergences (the gaps above and the P8 LunarLander);
- Gymnasium's own `check_env` and `check_environments_match` pass through the reverse bridge on CartPole-v1, the four
  other classic_control ids and the six toy_text ids;
- the golden trajectories replay without Python.
