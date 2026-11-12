# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 1. Project Overview

Gym.NET is a native C# (.NET) reinforcement learning environment toolkit and benchmark suite, API-compatible with Farama Gymnasium (the successor of OpenAI Gym) and aligned with it byte for byte. It is its own codebase: Gymnasium is the reference it measures itself against, not an identity it claims. It is part of the SciSharp STACK ecosystem and uses `NumSharp` for multidimensional tensor/array operations and `SixLabors.ImageSharp` for rendering.

- **Canonical SOT Submodule:** `refs/Gymnasium` (Farama Foundation Gymnasium Git submodule for behavioral, mathematical, and algorithmic parity).
- **Roadmap:** `docs/plans/COMPLETION_PLAN.md` orders the remaining work to one 2.0 release (phases R0, P0r, P2, P3, P7, P4, P5, P6, P8, P10; MuJoCo's P9, postponed until DR-61, is under way) and records the 2026-11-08 decisions; `docs/plans/GYMNASIUM_PORT_PLAN.md` holds the per-area specifications.
- **User docs (P10):** `docs/README.md` indexes the guides, `docs/architecture.md` and the draft release notes `docs/releases/RELEASE_2.0.0.md`; every code sample in them was compiled and run against the tree. Keep them true when the API changes.
- **Design gate:** `docs/plans/DESIGN_REVIEW.md` holds the approved implementation designs (DR-01…DR-71). Implement only what an approved design covers; a design question no DR-xx answers goes back to the maintainer for review before the step continues.
- **Working branch:** `journey1` until 2.0 (decision D20); it merges into `master` once, at the release.

---

## 2. Solution & Project Architecture

The repository is structured around the `Gym.NET.sln` solution:

```
refs/Gym.NET/
├── src/
│   ├── Gym/                       # Core API (env contract, wrappers, results, errors), spaces, utilities, legacy shim
│   │   ├── Envs/                  # IEnv, Env<TObs, TAct> (Step, ValueTask StepAsync), Wrapper family (+ IWrapper), Info, Metadata, RenderResult
│   │   │   ├── Registration/      # EnvSpec (Make, ToJson/FromJson), WrapperSpec, EnvCreator, EnvSpecRegistry, EnvRegistry (register, spec, make, pprint_registry),
│   │   │   │                      #   EnvRegistry.MakeVec.cs (make_vec: sync, async and vector entry point), VectorizeMode
│   │   │   └── Viewers/           # IEnvViewer, IEnvironmentViewerFactoryDelegate, IInteractiveEnvViewer + ViewerInput (keys, close, resizes, mouse: DR-70),
│   │   │                          #   NullEnvViewer (namespace Gym.Environments; DR-20); Clock (Gym.Rendering)
│   │   ├── Spaces/                # Space (base), Box (bounded/continuous), Discrete
│   │   ├── Observations/          # StepResult<TObs> (5-tuple), ResetResult<TObs>, DoneStepResult<TObs> (pre-0.26 4-tuple)
│   │   ├── Vector/                # IVectorEnv, VectorEnv<TObs, TAct> (batched types, DR-21; _add_info), VectorStepResult/VectorResetResult, AutoresetMode,
│   │   │                          #   SyncVectorEnv (typed, and untyped as SyncVectorEnv), AsyncVectorEnv (worker threads, AsyncExecutionMode, AsyncState;
│   │   │                          #   DR-22; custom workers: AsyncVectorWorker over PipeConnection, AsyncVectorEnv.AsyncWorker, DR-38), VectorWrapper family, VectorUtils (batch_space, concatenate, iterate, the shared-memory functions)
│   │   ├── Wrappers/              # Common.cs: TimeLimit, OrderEnforcing, PassiveEnvChecker, Autoreset, RecordEpisodeStatistics;
│   │   │                          #   Rendering.cs: RenderCollection, RecordVideo (IVideoWriter, DR-25), HumanRendering, AddWhiteNoise, ObstructView;
│   │   │                          #   StatefulObservation.cs: TimeAwareObservation, NormalizeObservation, DelayObservation, FrameStackObservation,
│   │   │                          #   MaxAndSkipObservation; StatefulAction.cs: StickyAction, RepeatAction; StatefulReward.cs: NormalizeReward;
│   │   │                          #   TransformObservation.cs: TransformObservation and its 9 subclasses (Filter, Flatten, Grayscale, Resize, Reshape,
│   │   │                          #   Rescale, Dtype, AddRender, Discretize); TransformAction.cs, TransformReward.cs; WrapperFunctions(.Spaces).cs:
│   │   │                          #   RunningMeanStd, update_mean_var_count_from_moments, create_zero_array, rescale_box
│   │   │   └── Vector/            # The vector wrappers (DR-24: `...Vector` names, [PyName] = Gymnasium's): VectorizeTransform.cs (Transform*Vector,
│   │   │                          #   VectorizeTransform{Observation,Action,Reward} and their Filter/Flatten/Grayscale/Resize/Reshape/Rescale/Dtype/Clip
│   │   │                          #   twins), StatefulVector.cs (NormalizeObservation/NormalizeReward), CommonVector.cs (RecordEpisodeStatistics),
│   │   │                          #   DictInfoToList.cs, RenderingVector.cs (HumanRendering, RecordVideo)
│   │   ├── Legacy/                # [Obsolete] pre-0.26 API: Step, ILegacyEnv, LegacyEnvAdapter, VecEnv family
│   │   ├── Internal/              # [Obsolete] until 3.0 (DR-56): Dict (Gym.Legacy's info type) and its Dynamic/Merging/Reflection helpers
│   │   ├── Utils/                 # Seeding (PCG64), PyBuiltins, PyMath (F3), PyAttributes, PyEquality, PyCopy, PyKwargs (Python kwargs binding), PyJson, Deque,
│   │   │                          # PassiveChecker, EnvChecker (check_env, data_equivalence), EnvMatch, StepApiCompatibility, RecordConstructorArgs, Warnings,
│   │   │                          # Cv2 (OpenCV's uint8 and float32 cv2.resize, INTER_AREA and INTER_LINEAR, bit for bit; DR-23, DR-46: float32 bilinear is
│   │   │                          # OpenCV's own kernel, not IPP's), Video (save_video; PNG/ffmpeg writers),
│   │   │                          # Performance (benchmark_step/vector_step/init/render; DR-36)
│   │   ├── Experimental/Functional/ # FuncEnv (Gymnasium's functional protocol, no JAX; DR-36)
│   │   ├── Gymnasium.cs           # The `gymnasium` facade: Make, MakeVec, Spec, Register, Registry, PprintRegistry, HumanViewerFactory (DR-20)
│   │   ├── Logger.cs              # Warnings as gymnasium.logger issues them (colourised, Python's default filters)
│   │   └── Exceptions/            # Every gymnasium.error class (Error, ResetNeeded, ...), AssertionError, NotImplementedError, ImportError; 0.2's three as [Obsolete] bases of InvalidAction, AlreadyPendingCallError, NoAsyncCallError (DR-58)
│   │
│   ├── Gym.SourceGenerators/      # Roslyn generator (DR-13, DR-34): <Class>Options records of [GymOptions] envs/wrappers; [GymTuple]/[GymDict] records
│   ├── Gym.Rendering/             # pygame-compatible rasterizer: Surface (opaque or per-pixel alpha), GfxDraw (SDL_gfx 2.0.26 port), Draw (line, rect;
│   │   │                          #   polygon, circle, lines (widths 1 and 2) in Draw.Polygon.cs, black-box from pygame like line and rect, D4;
│   │   │                          #   aaline/aalines in Draw.Aaline.cs, translated from pygame-ce's draw.c with its notice, DR-37; the package is
│   │   │                          #   Apache-2.0 like every package, DR-49), Transform (flip, scale, smoothscale), ImageIO (image.load/frombytes),
│   │   │                          #   SurfArray (incl. make_surface), Vector2;
│   │   │                          #   pygame's display and event queue as PygameContext instances (PygameDisplay, PygameEventQueue; DR-41), whose
│   │   │                          #   Default the static Display/EventQueue forward to, process-wide like pygame's; Event, EventType, KeyCode
│   │   └── Utils/                 # namespace Gym.Utils: Play (play, display_arr, RunAsync: DR-44), PlayableGame, MissingKeysToAction, PlayPlot
│   │                              #   (IPlotFigure, PlotFigure); DR-36
│   ├── Gym.Environments/          # Concrete environment implementations
│   │   ├── Box2D/                 # namespace Gym.Box2D, compiled into Gym.Environments (DR-65): Box2D 2.3.0 in managed C# (D5, DR-26, DR-29),
│   │   │                          #   float32 and bit for bit as pybox2d 2.3.10 computes it: Common/, Collision/, Dynamics/ (the C++-faithful
│   │   │                          #   core, statement for statement; internal where pybox2d hides it) and Facade/ (pybox2d's surface:
│   │   │                          #   b2World(gravity, doSleep), CreateDynamicBody, b2FixtureDef, b2PolygonShape(vertices/box),
│   │   │                          #   b2RevoluteJointDef, contact listener, ray casts, SWIG's conversions and messages). Each file keeps Erin
│   │   │                          #   Catto's or Ken Lauer's zlib notice
│   │   ├── MuJoCo/                # namespace Gym.MuJoCo, compiled into Gym.Environments (DR-65, DR-66): Gym.NET's own engine, written from
│   │   │                          #   scratch, bit for bit with MuJoCo 4.0.0 on win-x64 (DR-61, DR-64) on Gymnasium's 14 bundled models; it
│   │   │                          #   refuses, by name, any model feature they do not use. MjModel (FromXmlPath over the embedded models,
│   │   │                          #   DR-63), MjData (the simulation: Euler and RK4, the colliders, Newton and PGS), Ccd (the convex
│   │   │                          #   collider), Kernels; Gymnasium's reach under .NET names (Mujoco.MjStep/MjForward/..., MjtState, MjtObj,
│   │   │                          #   writable NDArray views; DR-68). Proven by tools/parity/mujoco/halfcheetah_poc's 50 traces
│   │   ├── Envs/Classic/          # CartPoleEnv (+ CartPoleVectorEnv, CartPole's vector entry point), MountainCarEnv (MountainCar-v0 and
│   │   │                          #   MountainCarContinuous-v0, DR-32), PendulumEnv, AcrobotEnv, ClassicControlUtils
│   │   ├── Envs/ToyText/          # BlackjackEnv (+ BlackjackObservation, DR-13), FrozenLakeEnv, CliffWalkingEnv, TaxiEnv; ToyTextUtils (CategoricalSample,
│   │   │                          #   CharMap: the 'c' map, N-2's workaround), PyKeys (Python's dict/list lookups of an action), ToyTextTransition
│   │   ├── Envs/MuJoCo/           # MujocoEnv (the base: model by bundled name, SetState, DoSimulation, the resets' noise) and Gymnasium's 22
│   │   │                          #   classes as <Name>EnvV4/<Name>EnvV5, each [PyName]d to Gymnasium's (DR-24, DR-66): Ant, HalfCheetah,
│   │   │                          #   Hopper, Humanoid (+ MassCenter), HumanoidStandup, InvertedDoublePendulum, InvertedPendulum, Pusher
│   │   │                          #   (PusherEnvV4 refuses, as Gymnasium's on MuJoCo 3+), Reacher, Swimmer, Walker2d; MujocoMath (internal:
│   │   │                          #   NumPy's pairwise sums, norms and the control cost in the action's dtype; DR-08 info scalars);
│   │   │                          #   MujocoRenderer (Gymnasium's camera rules, rgb_array/depth_array/rgbd_tuple, and human: the window
│   │   │                          #   with Gymnasium's full WindowViewer controls, DR-70); MujocoUtils (utils' 3 functions)
│   │   ├── Envs/MuJoCo/Rendering/ # internal (DR-69): MuJoCo's classic renderer drawn in software (DR-62): MujocoScene (mjv_updateScene,
│   │   │                          #   the C/E keys' contact and frame decorations; MjvCamera.Move = mjv_moveCamera), MujocoMeshes (the
│   │   │                          #   builtin meshes, the arrows' cone), MujocoFrame (mjr_render's passes: reflection, shadow maps, skybox,
│   │   │                          #   haze, transparency; mjr_readPixels), GlContext (the fixed-function OpenGL MuJoCo uses: lighting,
│   │   │                          #   texgen, clipping, 4x MSAA, stencil, blending), GlTexture (mipmaps, filtering), GlMath. Where the GPU
│   │   │                          #   decides (rounding, LOD, centroid, resolve), the reference GPU's measured behaviour: gpu_probe.py.
│   │   │                          #   MujocoOverlay (mjr_overlay with MuJoCo's normal150 font, Assets/mujoco/fonts) and MujocoWindow
│   │   │                          #   (Gymnasium's WindowViewer: render loop, pause, keys, mouse through IInteractiveEnvViewer, menu; DR-70)
│   │   ├── Envs/                  # NpScalar (NEP 50 scalars), HumanWindow (DR-20), EnvAssets (DR-19), RenderMath
│   │   ├── Assets/                # Embedded env images copied from Gymnasium (classic_control, toy_text), CarRacing's recorded score glyphs
│   │   │                          #   (fonts/freesansbold-42.json, DR-48) and Gymnasium's 14 MuJoCo models as MuJoCo 4.0.0 compiles them
│   │   │                          #   (mujoco/*.json and their builtin textures, recorded by tools/parity/mujoco/record_models.py, DR-63);
│   │   │                          #   README lists sources and hashes
│   │   ├── Envs/Box2D/            # LunarLanderEnv (LunarLander-v3 and LunarLanderContinuous-v3, [PyName("LunarLander")], DR-32; Heuristic,
│   │   │                          #   DemoHeuristicLander, a private ContactDetector, DR-28; LunarLanderContinuousEnv, the refusing stub) on Gym.Box2D;
│   │   │                          #   BipedalWalkerEnv (BipedalWalker-v3 and BipedalWalkerHardcore-v3, [PyName("BipedalWalker")]; one world per env,
│   │   │                          #   lidar ray casts, NEP 50 rewards; private ContactDetector), BipedalWalkerHeuristics, BipedalWalkerHardcoreEnv (stub);
│   │   │                          #   CarRacingEnv (CarRacing-v3, [PyName("CarRacing")]; RoadTile, public FrictionDetector, RegisterInput), CarDynamics.cs
│   │   │                          #   (Car, CarWheel, Particle), ScoreFont (internal: the score's text as SDL_ttf 2.24 lays it out, zlib translation, from
│   │   │                          #   glyphs recorded from pygame by tools/parity/record_score_font.py and embedded as Assets/fonts/freesansbold-42.json:
│   │   │                          #   every text f"{reward:04.0f}" can show, NaN and the infinities included, DR-27, DR-48)
│   │   ├── GymEnvironments.cs     # Register(): the built-in ids, from a module initializer (DR-12; GYMNET_DISABLE_ENV_REGISTRATION disarms it, DR-14)
│   │   └── buildTransitive/       # Gym.NET.Environments.targets: the app-side registration initializer (DR-17)
│   │
│   ├── Gym.Rendering.Avalonia/    # Cross-platform GUI rendering (AvaloniaEnvViewer, StaticAvaloniaApp); keyboard and mouse input (IInteractiveEnvViewer: play; MuJoCo's window, DR-70);
│   │                              #   AvaloniaPlotFigure, PlayPlot's ScottPlot figure (DR-39)
│   ├── Gym.Rendering.WinForm/     # Windows Forms GUI rendering (WinFormEnvViewer); keyboard and mouse input (IInteractiveEnvViewer: play; MuJoCo's window, DR-70);
│   │                              #   WinFormPlotFigure, PlayPlot's ScottPlot figure (DR-39)
│   └── Shared/                    # ScottPlotAxes.cs: the IPlotAxes adapter both viewer packages compile (DR-39)
│
├── tests/
│   ├── Gym.Tests/                 # MSTest test suite for environments, spaces, and multi-instance concurrency
│   │   └── Gymnasium/             # refs/Gymnasium/tests mirrored one to one (test_core.py → TestCore.cs), namespace Gym.Tests.Gymnasium.* (DR-06)
│   ├── Gym.Tests.Parity/          # Live parity gate: embedded CPython runs refs/Gymnasium, compared byte/pixel-exact (see its CLAUDE.md)
│   └── Gym.Tests.Plugin/          # A module that registers an id when loaded, for make("Gym.Tests.Plugin:RegisterDuringMake-v0")
│
├── tools/parity/                  # Parity stack: manifest.json (pins), requirements.txt, setup_venv.{ps1,sh}; audit probes
├── tools/parity/mujoco/           # record_models.py (the embedded MuJoCo models, --check); gpu_probe.py (re-measures the GPU behaviours
│                                  #   the renderer reproduces, scoring its models); halfcheetah_poc/: the trace checker over the
│                                  #   product engine (oracle.py records MuJoCo's state per mj_step; verify/: traces.txt, record_traces.py,
│                                  #   negative_controls.py, branch_search.py, cs_coverage.py), and play/, a playable HalfCheetah
├── tools/showreel/                # The front-page media (DR-59, DR-60): the synthwave video and README GIF (docs/media), the site's 8-bit
│                                  #   gallery (docs/website-src/images/envs), from the envs' own frames under scripted policies; its parity
│                                  #   segment is checked against Gymnasium in Python (gymnasium_replay.py). Not in the sln; README.md has the commands
│
├── examples/
│   └── CartPole/                  # The one example (D13): the cross-entropy method over linear policies, 100 carts at once through
│                                  #   MakeVec, then a played episode (--render: Avalonia window); in Gym.NET.sln, so API changes break its build
│
├── docs/                          # README.md (index), architecture.md, guides/ (getting_started, creating_custom_environments,
│                                  #   migrating_to_2.0, development_and_testing), releases/RELEASE_2.0.0.md (draft), plans/;
│                                  #   media/ (the showreel); docfx.json + website-src/ (the DocFX site: landing page, template
│                                  #   templates/gymnet, gallery images; `docfx docs/docfx.json`, deployed by .github/workflows/docs.yml)
│
└── refs/
    └── Gymnasium/                 # Submodule: Farama-Foundation/Gymnasium (Python Master SOT)
```

### Core Design Patterns & Mechanisms

- **Environment Contract (`Env<TObs, TAct>` / `IEnv`), Gymnasium's shape since the core-API work (P1):**
  - `ResetResult<TObs> Reset(BigInteger? seed = null, options = null)`: `(obs, info)`; a seed reseeds `NpRandom` with Gymnasium's `Generator(PCG64(SeedSequence(seed)))`, `null` keeps the stream. Implementations start with `SeedNpRandom(seed)` (Gymnasium's `super().reset(seed=seed)`).
  - `StepResult<TObs> Step(TAct action)`: `(obs, reward (float64), terminated, truncated, info)`; `IEnv.Step(object)` converts untyped actions (`ConvertAction`). The result structs (and the vector ones) are `IDisposable`: `using var step = env.Step(a);` releases the arrays they hold (DR-51). 0.2's spellings compile until 3.0 through `[Obsolete]` shims (DR-58): a four-element `Deconstruct` (done = terminated or truncated), `ResetResult` → observation, `CloseEnvironment()`, `Box(low, high, dType)`, `CartPoleEnv(viewerFactory)`, LunarLander's 0.2 names; never call them from Gym.NET's own code (the build shows any such use as CS0618).
  - `ValueTask<StepResult<TObs>> StepAsync(TAct action, CancellationToken ct = default)` (and `IEnv.StepAsync(object)`): runs `Step` on the calling thread and completes synchronously, never through `Task.Run`; a failure or a cancellation before the step surfaces through the task. Override it only for an env with real asynchronous work; wrappers do not forward it, so a wrapper's `Step` override always applies.
  - `RenderResult Render()`: per the `render_mode` fixed at construction (`RenderResult.Rgb` for `rgb_array`; `human` renders inside reset/step and returns `None`); `RenderModes` names the modes as constants (DR-52).
  - `NpRandom` (lazy, entropy-seeded on first use) / `NpRandomSeed` (-1 once a generator is assigned), `Metadata`, `Spec`, `Unwrapped`, `Close()` / `Dispose()`.
  - Python attributes by Python name: `HasWrapperAttr`/`GetWrapperAttr`/`SetWrapperAttr` and `Gym.Utils.PyAttributes` (`[PyName("_x")]` or snake_case of the property).
- **Wrappers:** `Wrapper<TWObs, TWAct, TObs, TAct>` (and `Wrapper<TObs, TAct>`) forward to `Env`; `ObservationWrapper`, `RewardWrapper`, `ActionWrapper` add the hooks; `TimeLimit`, `OrderEnforcing`, `PassiveEnvChecker`, `Autoreset`, `RecordEpisodeStatistics`, `NormalizeObservation`, `NormalizeReward`, `TimeAwareObservation` and `FlattenObservation` live in `Gym.Wrappers` (the other wrappers are P6, deferred).
- **Vector envs (P6.1, DR-21):** `VectorEnv<TObs, TAct>` is typed by the batched types (`SyncVectorEnv<NDArray, NDArray>` steps an `NDArray` of actions into a `VectorStepResult<NDArray>` of `(observations, rewards, terminations, truncations, infos)`); `Gymnasium.MakeVec("CartPole-v1", 3, "sync")` builds `SyncVectorEnv` over `make`'s stacks, `"vector_entry_point"` (the default when registered) CartPole's `CartPoleVectorEnv`; `AutoresetMode`/`VectorizeMode` are Python-enum-like classes whose values convert from strings as Python's `Enum` call takes them (`"NextStep"`, `"sync"`). `"async"` builds `AsyncVectorEnv` (P6.2, DR-22): Gymnasium's protocol on worker threads (`ThreadPool` by default,
  `DedicatedThreads`, `ThreadPerEnv`), the same bits as sync. The vector wrappers (P6.6) live in `Gym.Wrappers.Vector`.
- **Registry and `make` (P3):** `Gymnasium.Make("CartPole-v1")` (or typed, `Make<NDArray, NDArray>(...)`, DR-10) builds Gymnasium's stack, `TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPoleEnv>>>`; kwargs go as a dictionary, an anonymous object (`new { render_mode = "rgb_array" }`) or a generated options record (`new CartPoleEnvOptions { RenderMode = "rgb_array" }`, DR-11/DR-34), by Python name, bound to constructor parameters by `PyKwargs`. Entry points are a class (`typeof(CartPoleEnv)`), an `EnvCreator`, or a `"Namespace:Class"` string. The built-in ids register before `Main` (DR-12/14/17); the unit tests import the facade with `global using static Gym.Gymnasium` (DR-06's shadowing). `EnvRegistry.Namespace(ns)`, Gymnasium's `namespace` scope, applies to the thread and async flow that opens it (an `AsyncLocal`, DR-57), not to the whole process as Python's global does.
- **Pluggable Viewer Architecture (DR-20):** `render_mode="human"` renders into the env's `viewer_factory`, else the app-wide `Gymnasium.HumanViewerFactory` (`AvaloniaEnvViewer.Factory`, `WinFormEnvViewer.Factory`, or `NullEnvViewer.Factory` headless), else raises `DependencyNotInstalled` at the first human render, as Gymnasium does without pygame; nothing falls back to a headless viewer silently. Human frames are paced by `Gym.Rendering.Clock` at `metadata["render_fps"]`.
- **Legacy API (`Gym.Legacy`, `[Obsolete]`):** `LegacyEnvAdapter` exposes any env through the pre-0.26 4-tuple API (`Seed(s)` + `Reset()` is `Reset(seed: s)`; the reward stays float64, as everywhere in Gym.NET); the baselines-style `IVecEnv` / `VecEnv` / `DummyVecEnv` work over `ILegacyEnv`; new code uses `Gym.Vector` (P6.1).
- **Observation & Action Spaces:** Strongly-typed bounds checking and random sampling via `Box` (multi-dimensional continuous floats/bounds) and `Discrete` (categorical integer actions); `Seed(BigInteger)`.
- **Documentation voice:** XML docs describe Gym.NET's own members (what they do, what they cost, their footguns) and never say "Port of ...". A member based on Gymnasium carries its source as a pinned GitHub permalink in `<remarks>`, e.g. `Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/core.py#L28">gymnasium/core.py</see>.`; pygame-ce, CPython and NumPy sources are referenced the same way at their pinned tags (port plan §3.1 rule 2).

---

## 3. Build & Test Commands

### 3.1 Building the Solution & Projects

```powershell
# Build entire solution (Debug / Release)
dotnet build Gym.NET.sln -c Debug
dotnet build Gym.NET.sln -c Release

# Build specific subprojects
dotnet build src/Gym/Gym.csproj -c Release
dotnet build src/Gym.Environments/Gym.Environments.csproj -c Release
dotnet build src/Gym.Rendering.Avalonia/Gym.Rendering.Avalonia.csproj -c Release
dotnet build src/Gym.Rendering.WinForm/Gym.Rendering.WinForm.csproj -c Release

# The example: trains CartPole headless in about two seconds, exits 0 when solved (--render opens a window: desktops only)
dotnet run --project examples/CartPole -c Release

# The packages (versions 2.0.0, P10.4) into a folder of your choice; building also packs (GeneratePackageOnBuild)
dotnet pack Gym.NET.sln -c Release -o <folder>
```

### 3.2 Running Automated Tests

Tests are located in `tests/Gym.Tests/` using MSTest. Tests that assert a reference behavior Gym.NET or NumSharp doesn't
provide yet are `[OpenBugs("N-x")]` (decision DR-01; `tests/Gym.Tests/NumSharpGaps` proves each NumSharp gap): gated runs
exclude them, as in the parity suite. The viewer tests (`*AvaloniaEnv`, `*WinFormEnv`) open windows; exclude them on a
machine whose screen must not be used.

```powershell
# The gate: every unit test that must pass, headless
dotnet test tests/Gym.Tests/Gym.Tests.csproj --filter "FullyQualifiedName!~AvaloniaEnv&FullyQualifiedName!~WinFormEnv&TestCategory!=OpenBugs"

# The open gaps (NumSharp items N-1, N-3, ...), each failing until its fix lands
dotnet test tests/Gym.Tests/Gym.Tests.csproj --filter "TestCategory=OpenBugs"

# The API inventory: Gymnasium's public names (tools/parity/gymnasium_inventory.json) against Gym.NET, by reflection.
# Porting a name means removing its line from tests/Gym.Tests/Gymnasium/inventory.expected-missing.txt (a ratchet);
# after a re-pin, regenerate the inventory with: python tools/parity/inventory.py --write
dotnet test tests/Gym.Tests/Gym.Tests.csproj --filter "FullyQualifiedName~InventoryTests" --logger "console;verbosity=detailed"

# Run tests targeting a specific framework
dotnet test tests/Gym.Tests/Gym.Tests.csproj -f net10.0-windows

# Run a specific test class
dotnet test tests/Gym.Tests/Gym.Tests.csproj --filter "FullyQualifiedName~CartpoleEnvironment"
dotnet test tests/Gym.Tests/Gym.Tests.csproj --filter "FullyQualifiedName~BoxTest"
dotnet test tests/Gym.Tests/Gym.Tests.csproj --filter "FullyQualifiedName~LunarLanderEnvironment"

# Run a single individual test method
dotnet test tests/Gym.Tests/Gym.Tests.csproj --filter "FullyQualifiedName~CartpoleEnvironment.Run_NullEnv"
dotnet test tests/Gym.Tests/Gym.Tests.csproj --filter "FullyQualifiedName~BoxTest.TestBoxBoundedTest"
dotnet test tests/Gym.Tests/Gym.Tests.csproj --filter "FullyQualifiedName~LunarLanderEnvironment.Run_Discrete_NullEnv"
dotnet test tests/Gym.Tests/Gym.Tests.csproj --filter "FullyQualifiedName~LunarLanderEnvironment.Run_TwoInstances_Continuous_AvaloniaEnv"
```

### 3.3 Gymnasium Parity Tests (live)

`tests/Gym.Tests.Parity` embeds CPython (pythonnet) and compares Gym.NET with `refs/Gymnasium` byte for byte and pixel for
pixel, on the reference stack pinned in `tools/parity/manifest.json` (Python 3.12, NumPy 2.4.2, pygame-ce 2.5.8, Box2D
2.3.10, OpenCV 4.13.0, MuJoCo 4.0.0). Cells Gym.NET does not pass yet are `[OpenBugs("B-xx")]` (NumSharp's convention): gated runs exclude them. Full
guide: `tests/Gym.Tests.Parity/CLAUDE.md`; design: `docs/plans/GYM_PARITY_INTEROP_TESTING.md`. An interpreter without the
`mujoco` package fails the stack check: create the venv with `tools/parity/setup_venv.ps1`, which installs it.

```powershell
# Once per machine without the stack: creates tools/parity/.venv, which the suite finds on its own
pwsh tools/parity/setup_venv.ps1

# The gate (what must pass)
dotnet test tests/Gym.Tests.Parity/Gym.Tests.Parity.csproj --filter "TestCategory!=OpenBugs"

# The open gaps, each printing its first divergence from Gymnasium
dotnet test tests/Gym.Tests.Parity/Gym.Tests.Parity.csproj --filter "TestCategory=OpenBugs" --logger "console;verbosity=detailed"

# The goldens (tests/Golden), replayed without Python; record them with GYMNET_PARITY_RECORD=1 (see its CLAUDE.md)
dotnet test tests/Gym.Tests.Parity/Gym.Tests.Parity.csproj --filter "TestCategory=Golden"

# The MuJoCo frame cells (DR-71) gate on tests/Gym.Tests.Parity/Baselines/mujoco_frames.json, recorded on the reference GPU;
# re-record it after a deliberate change to the renderer (tools/parity/mujoco/gpu_probe.py re-checks a new GPU first)
GYMNET_PARITY_RECORD_BASELINE=1 dotnet test tests/Gym.Tests.Parity/Gym.Tests.Parity.csproj -f net10.0 --filter "FullyQualifiedName~MujocoFrameLiveParityTests"
```

CI (`.github/workflows/ci.yml`, informational on `journey1` until 2.0, decision D20): a Windows job builds Debug and
Release, runs the headless unit gate and the live parity gate on the pinned stack; an Ubuntu job replays the goldens
without Python.

### 3.4 Submodule Management

```powershell
# Initialize the Gymnasium and NumSharp SOT submodules (refs/NumSharp at its v0.80.0 tag, the agents' NumSharp reference).
# Not --recursive: that would also clone NumSharp's own submodules (numpy, OpenBLAS, onnxruntime, ...), which nothing needs.
git submodule update --init refs/Gymnasium refs/NumSharp

# The build uses NuGet NumSharp 0.80.0 by default (eng/NumSharp.props, decision D3). To try a NumSharp fix before it is
# released, check it out in refs/NumSharp and build from source:
dotnet build Gym.NET.sln -p:NumSharpSource=Submodule

# Re-pinning Gymnasium (refs/Gymnasium is at v1.4.0, 8ba8c3a6): check out the new commit, move every permalink with the
# script (it reads the old commit from tools/parity/manifest.json, so run it before updating the manifest; it refuses
# any link whose cited lines changed, for review by hand), then set "gymnasium.commit" in the manifest and rerun the gates.
git -C refs/Gymnasium checkout <commit>
python tools/parity/repin_references.py           # dry run: the report
python tools/parity/repin_references.py --write   # rewrite the safe links
```
