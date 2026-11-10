# Gym.NET Design Review

| | |
|---|---|
| **Status** | Reviewed on 2026-11-08 in ten rounds, on branch `journey1`: 36 designs, all decided; none pending. [Coverage by phase](#coverage-by-phase) says which phases can start |
| **Rule** | A step of [`COMPLETION_PLAN.md`](COMPLETION_PLAN.md) starts only when every design it depends on is **Approved** here. A design that isn't listed is not approved: a new design question found while implementing goes through the same review before the step that needs it continues |
| **How** | The maintainer reviewed the implementation design choices four at a time, before any phase from R0 to P9 started. Each answer is recorded with the options that were on the table and what it changes in the plans. Where a later round refined an earlier answer, both rows stay and the earlier one points to the later |
| **Earlier decisions** | D1–D22 in [`GYMNASIUM_PORT_PLAN.md`](GYMNASIUM_PORT_PLAN.md) §20 and [`COMPLETION_PLAN.md`](COMPLETION_PLAN.md) §3 still stand unless a design here overrides them |

---

## Approved

### Round 1: R0 and P0r (2026-11-08)

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-01 | NumSharp gaps | Every gap is documented as an N-item (COMPLETION_PLAN §7) and **proven by a test**: an `[OpenBugs("N-x")]` cell that asserts NumPy's behavior and fails until NumSharp provides it. A workaround in Gym.NET comes only when the port is stuck without it | A workaround for every gap plus an upstream fix and a canary test; fixing NumSharp first with no workaround; local workarounds only | Every phase |
| DR-02 | Python exceptions | **NumSharp owns every Python and NumPy exception.** Missing ones are added to NumSharp (`NotImplementedError`, `ImportError`, and any other a port needs), and every one has public, customizable constructors, including `(message, innerException)`. NumSharp's parameterless `AssertionError()` changes to Python's `str()` text, `""` (N-1). Gym.NET throws NumSharp's types directly; `Gym.Exceptions` keeps only `gymnasium.error`'s classes | A `PyAssert.That` helper in Gym.NET; inline `throw new AssertionError("")` at each site | R0.1 and every phase |
| DR-03 | Static rules | **None**: no analyzers and no rule scripts. The port plan's §14.7 bans and F3's analyzer item are dropped | A Roslyn analyzer; a CI regex script; the script first and the analyzer later | P0r (P0r.2 dropped) |
| DR-04 | Goldens | **Recorded from the live cells**, with the code that regenerates them in the live suite (`tests/Gym.Tests.Parity`, a record mode). A Python-free replay checks them anywhere | The live suite only; golden frames only | P0r.3 |

### Round 2: standards, test layout, spaces (2026-11-08)

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-05 | Warnings | **Warnings are ignored.** No `Nullable`/`TreatWarningsAsErrors` rollout, new projects included, and no warning ever gates a step. The projects that already set `TreatWarningsAsErrors` (`Gym.Rendering`, `Gym.Tests.Parity`) keep their settings | Standards in every `src` project from R0; per project on rewrite; new projects only | Every phase |
| DR-06 | Test layout | **`tests/Gym.Tests/Gymnasium/`** (renamed from `Ported/`) mirrors `refs/Gymnasium/tests` one to one: `spaces/test_box.py` → `Gymnasium/Spaces/TestBox.cs`, namespace `Gym.Tests.Gymnasium.*`. Gym.NET's own tests stay beside it | A separate `tests/Gym.Tests.Gymnasium` project; keeping `Ported/` | R0.5, every phase |
| DR-07 | Space typing | **`Space<T>` over an untyped `Space` base, and `Env<TObs, TAct>` exposes `Space<TObs> ObservationSpace` and `Space<TAct> ActionSpace`**; `IEnv` keeps the untyped view | `Space<T>` with untyped env spaces; untyped spaces only | P2.1 |
| DR-08 | Discrete samples | **A 0-d int64 `NDArray`**, Gym.NET's stand-in for `np.int64`: `Discrete : Space<NDArray>`. With DR-07, discrete-action envs become `Env<NDArray, NDArray>` (CartPole and LunarLander change in P2.1), and discrete observations (FrozenLake, CliffWalking, Taxi) are 0-d int64 arrays | `long` | P2, P4, P5, P8 |

### Round 3: composites, `make`, registry (2026-11-08)

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-09 | Composite samples (`Tuple`, `Dict`, `Sequence`, `OneOf`, `Graph`) | **Typed POCOs**: a class per composite shape, with a property for each tuple item or dict key, a `Deconstruct`, and implicit casts for interop with the generic form. The concrete sketch (what sits underneath for spaces built at run time) is DR-13 | Python-shaped Gym.NET types only; .NET collections; typed generics with ValueTuples; Python containers in NumSharp | P2, P5, P6 |
| DR-10 | `Gymnasium.Make` return types | **Both**: `IEnv Make(...)` for registry-wide tooling and `Env<TObs, TAct> Make<TObs, TAct>(...)`, which checks the env's types before building it; `MakeVec` follows the same pattern | `IEnv` only; typed only | P3, P6 |
| DR-11 | Env kwargs | **Typed options records** per env, **plus dictionary and anonymous-object overloads**. All three bind to constructor parameters by Python name (snake_case or `[PyName]`); unknown keys raise Python's `TypeError` text; `EnvSpec.Kwargs` keeps Python names | Dictionary and anonymous object only; dictionary only | P3 and every env phase |
| DR-12 | Built-in registry | **Module initializers**: each env assembly registers its own ids, **and a switch skips that registration** (an MSBuild property in this round; an environment variable since DR-14). What runs the initializer before the first `Make` is DR-17 | Every id registered up front with string entry points; explicit registration by the app | P3 and every env phase |

### Round 4: composite sketch, registration, names, packages (2026-11-08)

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-13 | Composite sketch (DR-09 made concrete) | **Source-generated records**: `[GymTuple] public partial record BlackjackObservation(NDArray PlayerSum, NDArray DealerCard, NDArray UsableAce);` and a dict counterpart. A Roslyn source generator (a generator, not an analyzer, so DR-03 doesn't cover it) writes the implicit casts to and from the generic form, Python `==` (element-wise through `PyEquality`, overriding record equality) and Python `repr`. The generic form underneath is `object?[]` for tuples and an insertion-ordered map for dicts | POCOs hand-written over `PyTuple`/`PyDict` with static-abstract conversion | P2 (generator in P2.1), P5, P6 |
| DR-14 | Registration skip (DR-12 made concrete) | **An environment variable disarms the module initializers**: when it is set, an env assembly's `[ModuleInitializer]` registers no ids. It replaces DR-12's MSBuild property. What runs the initializer before the first `Make` is DR-17 | A buildTransitive-injected app initializer with MSBuild opt-outs; loading env assemblies on a registry miss; the app touching the family first | P3 |
| DR-15 | Names | **C# naming conventions for Gym.NET's own types and members; PascalCase static methods for Gymnasium's module functions** (`FrozenLake.GenerateRandomMap`). The `[GymName]` attribute proposed in this round was dropped: DR-18, DR-24, DR-28 and DR-32 settle which classes are renamed and how their Gymnasium names are kept (`[PyName]` on the renamed classes) | Gymnasium's names verbatim with modules mapped to containers; Python names everywhere | P3–P9 |
| DR-16 | Packages | **The managed env families together**: classic_control, toy_text and box2d (with the Box2D engine) in `Gym.NET.Environments`. MuJoCo would join them once it is ported to managed code 100% (DR-30), and is postponed for now (DR-35) | One package per family; everything in one package | P4–P10 |

### Round 5: registration trigger, names, assets, human mode (2026-11-08)

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-17 | What runs the registration (DR-12, DR-14) | **An app-side initializer injected by the env package**: its `buildTransitive` targets compile a one-line `[ModuleInitializer]` into the consuming C# app that calls the family's `Register()`, so registration runs before `Main`. The environment variable of DR-14 disarms it. F#/VB apps call `Register()` themselves (documented). In-repo projects (tests, examples) import the same targets | The registry loading known env assemblies by name; the app touching the family first | P3 |
| DR-18 | Name attributes | **Classes are identified by their C# types; no class-name attribute. `[PyName]` stays for Python attribute names** that snake_case can't produce (`get_wrapper_attr("_disable_render_order_enforcing")` is Gymnasium API; 19 uses). Gymnasium's ids stay exactly as they are. Round 6 added a constraint: **every class has a unique name** across Gym.NET; namespaces or nesting may not be what tells two classes apart. The scheme for Gymnasium's 30 duplicated names is DR-24 | One `[GymName]` attribute for types and members; no attributes at all | P3–P9 |
| DR-19 | Assets | **Embedded resources, copied once from a Gymnasium installation into the repo** (re-copied on a re-pin), not built from `refs/Gymnasium` at build time. PNGs are decoded from the resource stream (ImageSharp, D18); MuJoCo XML is handed to MuJoCo from memory. Their license is confirmed when they are copied (D14) | Content files next to the assembly; loading from a Gymnasium install at run time | P4, P5, P9 |
| DR-20 | `render_mode="human"` | **An app-wide default, a per-env override, and Gymnasium's error otherwise**: `Gymnasium.HumanViewerFactory` set once at app start, a `viewer_factory` kwarg (or options-record property) per env, and `DependencyNotInstalled` with Gymnasium-style text when neither is set. The silent `NullEnvViewer` fallback goes; headless tests pass `NullEnvViewer.Factory` explicitly | Auto-discovering a deployed viewer; today's per-env factory with a silent headless fallback | P3, P4, P5, P6 (`HumanRendering`), P8 |

### Round 6: names made exact, vector, async, resize (2026-11-08)

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-21 | `VectorEnv` typing | **`VectorEnv<TObs, TAct>` with batched types, plus an untyped view** (`IVectorEnv`), as `Make` has both (DR-10); `MakeVec` has typed and untyped forms. Batched spaces are typed; rewards float64, flags bool, Gymnasium's info layout | Untyped only; NDArray-only batches | P6 |
| DR-22 | `AsyncVectorEnv` execution | **Three execution modes, ThreadPool by default** (the plan approved in round 10): `AsyncExecutionMode.ThreadPool` runs each sub-env call as a thread-pool task; `DedicatedThreads` keeps `threadCount` long-lived threads (default `Environment.ProcessorCount`) with env *i* pinned to thread *i* % `threadCount` for life; `ThreadPerEnv` gives each env its own thread and command channel. Gymnasium's API stays (`reset_async`/`reset_wait`, `step_async`/`step_wait`, `call_async`/`call_wait`, `close(terminate)`). In every mode results are gathered in env order, so async equals sync bit for bit, and the tests run all three; `max_concurrency` (v1.4.0) is a semaphore over running `reset`/`step` calls; a sub-env's exception returns from `*_wait` with Gymnasium's text and the env index; `shared_memory` is accepted and has nothing to do, since threads share memory; `copy` keeps its meaning. **A thread-bound sub-env (a human viewer) in ThreadPool mode automatically gets its own dedicated thread**, while the others stay on the pool | A thread per env only; thread-pool only; processes; thread-bound envs raising an error in ThreadPool mode | P6.2 |
| DR-23 | `cv2.resize(INTER_AREA)` (F8) | **A managed port of OpenCV's area-resize paths** (integer and fractional scale, uint8 and float32), under OpenCV's Apache-2.0 notice, proven byte for byte against live `cv2` at a pinned version | OpenCvSharp's native OpenCV; an injectable, inexact resizer | P6.4 |

### Round 7: unique names, video, Box2D, CarRacing text (2026-11-08)

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-24 | Unique class names | **Suffix scheme, with `[PyName]` on each renamed class** so `repr` and messages print Gymnasium's name: the 11 MuJoCo pairs become `AntEnvV4` / `AntEnvV5`, `HalfCheetahEnvV4`, …; the 18 vector-wrapper twins become `RecordEpisodeStatisticsVector`, `NormalizeObservationVector`, `ClipActionVector`, …, each `[PyName("RecordEpisodeStatistics")]`. `[PyName]` is then the one attribute, on members and on the classes that had to change. The maintainer questioned renaming `ContactDetector` and `Continuous_MountainCarEnv`; how they are named is DR-28 | `AntV4Env` / `VectorRecordEpisodeStatistics` prefixes; the same names with no class attribute | P6, P8, P9 |
| DR-25 | Video (F9) | **A pluggable `IVideoWriter`**; Gymnasium's triggers, file names and fps stay; the default writes PNG frame sequences, and **an ffmpeg writer ships as a pluggable implementation** (raw RGB piped to an ffmpeg process, mp4 like Gymnasium's). Parity is on frames, never on video bytes | ffmpeg as the built-in default; a managed MJPEG/GIF encoder | P6.5 |
| DR-26 | Box2D API (D5's managed port) | **Two layers: a core faithful to Box2D 2.3.x's C++** (float32, operation order) **and a thin pybox2d-shaped facade** the envs call, so the engine diffs against Box2D and the envs against Gymnasium's Python. The NuGet id `Box2D.NET` is taken by an unrelated Box2D v3 port. The maintainer added "NumSharp if possible"; what that covers is DR-29 | Core only; facade only | P8.1 |
| DR-27 | CarRacing text (D22) | **Glyph data recorded once from live pygame** (`'0'`–`'9'` and `'-'` of the default font at size 42: coverage bitmaps, bearings, advances, kerning) and SDL_ttf's shaded layout re-implemented, managed and exact for every reward string, and the fastest route. The font is GNU FreeFont (GPL with a font exception), so the recorded data gets a license check before P8.4. D22's FreeType default is replaced | Native FreeType; a managed FreeType port | P8.4 |

### Round 8: remaining names, Box2D and NumSharp, MuJoCo (2026-11-08)

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-28 | `ContactDetector`, `Continuous_MountainCarEnv` | **Keep Gymnasium's names where they aren't public twins**: each Box2D env has a private nested `ContactDetector`, which no outside code can name, so public class names stay unique. The maintainer added "Ending in Env and Continuous_MountainCarEnv -> MountainCarEnv"; the env-class naming this implies is DR-32 (round 9) | Nested detectors with a C#-style env name; suffix-renaming both | P4, P8 |
| DR-29 | Box2D engine and NumSharp | **Engine math on NumSharp `NDArray`s where it makes sense, by amount of code and by performance.** P8.1 starts with a measurement of the solver's hot loops (contact solver, broad-phase) written both ways; each subsystem uses NDArrays only where they are no slower and no larger in code, otherwise Box2D's float32 structs. The facade keeps DR-26's shape | The facade alone speaking NumSharp; hosting the engine under NumSharp | P8.1 |
| DR-30 | MuJoCo engine | **A managed port, 100%**, so MuJoCo joins `Gym.NET.Environments` with no native binaries (DR-16's condition is met). NumSharp assists where it fits: `np.evaluate` fused expressions and accelerated `np.*` calls. Its scope and parity target are DR-33; MuJoCo as a whole is postponed (DR-35) | Native bindings in a separate package; native for 2.0 and a managed port later | P9 (postponed) |
| DR-31 | MuJoCo frames | **MuJoCo's own renderer, offscreen**: with DR-30, its `mjr_*` rendering is ported too and drives OpenGL through a managed GL binding in a hidden WGL (Windows) or EGL (Linux) context; `rgb_array`, `depth_array` and human mode; frames best-effort, as the port plan's §1.2 allows for MuJoCo | No MuJoCo rendering in 2.0; software GL | P9 |

### Round 9: env names, options records, MuJoCo, optional scope (2026-11-08)

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-32 | Env class names | **Every env class ends in `Env`, and the two MountainCars merge into one `MountainCarEnv(continuous: bool)`**, registered for `MountainCar-v0` (`continuous: false`) and `MountainCarContinuous-v0` (`continuous: true`), the way LunarLander handles its continuous mode. The Box2D envs become `LunarLanderEnv`, `BipedalWalkerEnv`, `CarRacingEnv` (and the error-only stubs `LunarLanderContinuousEnv`, `BipedalWalkerHardcoreEnv`), each with `[PyName]` carrying Gymnasium's class name. Accepted consequences: `MountainCarContinuous-v0`'s `EnvSpec.kwargs` carries `continuous=True` where Gymnasium's is empty, and the merged class's constructor takes a kwarg Gymnasium's doesn't; to keep `repr` exact, the merged class prints `Continuous_MountainCarEnv` in continuous mode | `MountainCarContinuousEnv` named after its id; `ContinuousMountainCarEnv` | P4, P8 |
| DR-33 | MuJoCo scope (for when MuJoCo is done) | **All that Gymnasium's 22 envs reach**: the engine paths, the MJCF compiler subset and the renderer path the 14 bundled models (primitive geoms only) and v5's documented options exercise, proven bitwise against live Python `mujoco`, with fused multiply-adds in native builds measured first and any non-bitwise quantity documented with its ulp bound. **MuJoCo itself is postponed (DR-35)** | The whole library; the engine with precompiled `.mjb` models | P9 (postponed) |
| DR-34 | Options records (DR-11) | **Generated**: `[GymOptions]` on an env or wrapper, and DR-13's source generator emits a `<Class>Options` record from its constructor (one property per parameter, the Python kwarg names, `ToKwargs()`), always in sync; envs and wrappers become `partial` | Hand-written records; one common record plus a dictionary | P3 and every env/wrapper phase |
| DR-35 | MuJoCo in 2.0 | **Postponed entirely**: P9 leaves the 2.0 path. D21's "one 2.0 with everything" becomes everything except MuJoCo (core, spaces, registry, wrappers, vector, classic_control, toy_text, box2d). DR-30, DR-31 and DR-33 stand for when MuJoCo is done | — | P9, P10 |
| DR-36 | Optional scope | **In 2.0: `utils.play` and `PlayPlot`, the `FuncEnv` protocol (no JAX), and the `utils.performance` benchmarks. Out: `AtariPreprocessing`** (native ALE emulator and user ROMs) | — | P6, P10 |

### Round 10: `AsyncVectorEnv` (2026-11-08)

DR-22's three-mode plan was approved, with the auto-switch for thread-bound envs; the row above holds it.

---

## Coverage by phase

Every design each phase of [`COMPLETION_PLAN.md`](COMPLETION_PLAN.md) depends on, as of 2026-11-08:

| Phase | Designs it rests on | Can start |
|---|---|---|
| R0 Re-baseline | DR-01, DR-02, DR-05, DR-06; D2, D3, D18 | Yes |
| P0r P0's rest | DR-03 (no static rules), DR-04 (goldens), D20 (CI informational) | Yes |
| P2 Spaces | DR-07, DR-08, DR-09, DR-13 (the source generator), D19 (ownership) | Yes |
| P3 Registry and checkers | DR-10, DR-11, DR-12, DR-14, DR-17, DR-20, DR-34 | Yes |
| P7 Rasterizer | D4; DR-19 (assets) for `image.load` | Yes |
| P4 classic_control | DR-08, DR-19, DR-20, DR-32 (`MountainCarEnv(continuous)`) | Yes |
| P5 toy_text | DR-08, DR-13, DR-19, DR-20 | Yes |
| P6 Vector and wrappers | DR-21, DR-22, DR-23, DR-24, DR-25, DR-36 | Yes |
| P8 Box2D | D5, DR-26, DR-27, DR-28, DR-29 (its measurement opens P8.1), DR-32 | Yes |
| P9 MuJoCo | DR-30, DR-31, DR-33 | **Postponed** (DR-35) |
| P10 Release | DR-35, DR-36; D9, D13 | Yes |

## Interpretations to confirm

Three answers were recorded in the reading below; a correction from the maintainer replaces the row before the step that depends on it starts:

- **DR-02**: until NumSharp releases its changes, `NotImplementedError` and `ImportError` stay in `Gym.Exceptions` as DR-01's stuck-case workaround, each proven by an `[OpenBugs("N-3")]` test, and the bare-assert cells that need NumSharp's `""` message are `[OpenBugs("N-1")]`.
- **DR-29**: "if it makes sense both by amount of code and by performance" is read as a measured gate per subsystem in P8.1, not a requirement to use NDArrays everywhere.
- **DR-32**: the merged `MountainCarEnv` prints `Continuous_MountainCarEnv` in continuous mode, so `repr` stays exact; only `EnvSpec.kwargs` and the constructor's extra kwarg deviate.

## Consequences found while implementing

Effects of approved designs that only showed up in code. None changes a design; each records how the code lives with it.

- **DR-06 with DR-10/DR-20** (found in R0.5, 2026-11-09): the namespace `Gym.Tests.Gymnasium` shadows the
  `Gymnasium` facade class (`Gymnasium.Make`, `Gymnasium.HumanViewerFactory`) in every `Gym.Tests.*` namespace. C#
  resolves a simple name by walking the enclosing namespaces outward, and finds the namespace `Gym.Tests.Gymnasium` (a
  member of `Gym.Tests`) before it reaches the class, so `Gymnasium.Make(...)` does not compile in the unit tests
  (CS0234), and no global alias can win (global usings are consulted last). Both approved names stay: when P3 adds the
  facade, `Gym.Tests.csproj` imports it with a `global using static`, so tests call `Make("CartPole-v1")`, as Python's
  `from gymnasium import make` reads; the facade's own tests may also spell `global::Gym.Gymnasium`. Consumers'
  projects are unaffected unless they declare a `Gymnasium` namespace of their own.
- **DR-13 with DR-08 on Blackjack** (found in P5.5, 2026-11-09): `BlackjackEnv` is `Env<object?[], NDArray>`, and
  `BlackjackObservation` reads its observations by name through the generated conversion
  (`BlackjackObservation obs = env.Reset().Observation;`). Typing the env by the record would need a
  `Space<BlackjackObservation>`, a typed tuple space that every space tool (flatten, the vector utils, both checkers,
  the reverse bridge) recognizes beside `Tuple`; no DR covers one, so the observation space stays Gymnasium's `Tuple` and
  the observation its generic form, which every tool already reads. A maintainer who wants the env typed by the record
  gets a design question for that space first.
- **DR-13, the generated tuple records** (found in P5.5, 2026-11-09): a `[GymTuple]` record also implements
  `System.Runtime.CompilerServices.ITuple`, the shape every tuple reader in Gym.NET accepts (a `Tuple` space's
  `contains`, `PyEquality`, `repr`, the results' array walk), so a record held as an `object` is still the tuple it
  stands for: `env.ObservationSpace.Contains(obs)` holds for a `BlackjackObservation`. The generated indexer casts each
  member to `object?`, because NumSharp's implicit `long` → `NDArray` conversion would otherwise make the arms' common
  type `NDArray` and return a 0-d array for a `long` member (caught by `GymRecordsTests.TupleRecord_IsAnITuple`).
- **DR-21 in P6.1** (2026-11-09): a sync vector env is typed by the batched types, so the untyped `MakeVec` builds
  `SyncVectorEnv` (`SyncVectorEnv<object?, object?>`, its typed spaces viewed through `ObjectSpace`) and the typed one
  `SyncVectorEnv<TObs, TAct>`, refusing types the batched spaces do not hold. The rewards being float64 makes
  `CartPoleVectorEnv`'s a float64 array where Gymnasium's is float32 (equal values). `AutoresetMode` and `VectorizeMode`
  are classes with three instances rather than C# enums, so they print, parse and compare as Python's `Enum` members.
- **Typed wrappers whose type depends on an argument** (found in P6.3, 2026-11-09): `TimeAwareObservation` returns an
  array when flattening and a dict or tuple otherwise, so its observations are `object`s; `FlattenObservation` is typed
  for arrays and refuses spaces whose flat form is not one (`Graph`, `Sequence`).
- **The stateless wrappers** (found in P6.3, 2026-11-10): Python's `func` and duck-typed spaces become typed delegates
  and `Space<T>`s, so `TransformObservation` without a space keeps the env's and refuses (`TypeError`) an env whose
  observations are not of the wrapper's type. `FilterObservation` and `AddRenderObservation` build their spaces from
  copies of the env's subspaces, where Python shares the objects: the wrapper owns and disposes what it builds (D19), and
  sampling its space leaves the env's subspaces' generators where they are. `ReshapeObservation` takes a `long[]` (Python
  an int or a tuple). `ClipReward`'s array bounds must clip a reward to one value (`TypeError` otherwise) where Python
  would return an array reward, since rewards are float64. An untyped env's spaces are `ObjectSpace` views, so every
  space tool that dispatches on the space's class unviews it first (`ObjectSpace.Unview`). The discretizing wrappers need
  NumPy's float32 `linspace`, which NumSharp lacks (N-4, COMPLETION_PLAN §7).
- **DR-25 and DR-20, the rendering wrappers** (found in P6.5, 2026-11-10): `IVideoWriter` (`Gym.Utils`) takes a path
  without extension and adds its own; `PngVideoWriter`, the default, writes each video as a folder of that name holding
  `000000.png`, ... and `fps.txt`, so a folder of videos counts and names its videos as Gymnasium's folder of mp4 files
  does; `FfmpegVideoWriter` pipes raw frames to `ffmpeg` with moviepy's codec choices. `RecordVideo` takes the writer as
  an extra `video_writer` argument (recorded with the others, so a spec rebuild keeps it); `gc_trigger` left `null` is
  Gymnasium's default, every video, and Python's `None` is `_ => false`; in a `*_list` mode the video keeps copies of the
  frames the caller also receives (D19); closing writes a pending video, so Python's `__del__` warning has no
  counterpart. `HumanRendering` lives in the core assembly beside `RenderCollection`, so `make` applies it directly:
  `Gym.Rendering.Clock`, the frame pacer it shares with the envs, moved from the Gym.Rendering assembly into Gym
  (namespace unchanged); its window comes from `Gymnasium.HumanViewerFactory` (DR-20). The parity cells replace the
  absent moviepy with a stand-in `ImageSequenceClip` that records what it would encode, Gymnasium's own wrapper logic
  running unchanged.
- **DR-23, the OpenCV resize** (found in P6.4, 2026-11-10): `Gym.Utils.Cv2.Resize` translates the uint8 paths of
  OpenCV 4.13.0's `cv::resize` (the integer-factor area average, the area decimation, and the fixed-point bilinear
  kernel, which `INTER_AREA` also takes when an axis grows), and `INTER_LINEAR` with them, which the vector
  `HumanRendering` calls. Measured before porting: on the pinned `cv2` (IPP 2022.2, AVX2 dispatch) these results do not
  change with IPP or the optimizations switched off, so the scalar C++ is the reference, with one exception the port
  keeps: the 2×2 area kernel of 1, 3 and 4 channels rounds half up where the other area kernels round half to even.
  The float32 paths DR-23 names are not translated: Gymnasium's only callers (`ResizeObservation` and its vector twin)
  refuse every dtype but uint8. `cv2.error` has no .NET type (port plan §3.4); the one OpenCV refusal these paths
  reach (more than 4 channels by area) is a `NotSupportedException` with OpenCV's assertion text. `ResizeObservation`
  takes its shape as a `long[]`, as `ReshapeObservation` does.
- **The stateful wrappers with D19** (found in P6.3, 2026-11-10): Python's wrappers keep references to the objects they
  were given, and the garbage collector decides when those die; under D19 the caller owns its actions and the
  observations it receives, and may dispose them when a call returns (the parity harness does). `StickyAction` therefore
  keeps a deep copy of the last action and hands the env a fresh copy at every repeated step (keeping the caller's
  object, as Python does, repeated a disposed array: measured over Pendulum and MountainCarContinuous). The wrappers
  that keep the env's observations (`DelayObservation`'s queue, `FrameStackObservation`'s deque and reset padding)
  release only what they created (`FrameStackObservation`'s buffer and zero padding, `MaxAndSkipObservation`'s buffer):
  the env's info may hold an observation, so the ones they drop are left to NumSharp's finalizer. `FrameStackObservation`
  is typed by the env's observation type, which the stacks of an array, `Dict` or `Tuple` space share; a space whose
  stack is another type (`Text`, `Graph`, `Sequence`, `OneOf` stack into tuples) needs the untyped form. The vector
  utils (`batch_space`, `iterate`, `concatenate`, `create_empty_array`) unview `ObjectSpace` as the space utils do.
- **DR-24, the vector wrappers** (found in P6.6, 2026-11-10): `[PyName]` now also goes on a class, and
  `PyAttributes.PyTypeName` reads it without inheritance, so `ClipRewardVector` prints and is found as `ClipReward` while a
  class derived from it keeps its own name unless it declares one. The vectorizing wrappers build the single-env wrapper
  over a stand-in env holding the sub-envs' single space (Python's `_SingleEnv`) and apply its function to each sub-env's
  element through an internal untyped view of it (`IVectorizableTransform`, implemented by the transform wrappers). The
  stand-in is untyped, so an open generic class closes over `object` (`typeof(ClipAction<,>)`), and a space argument whose
  elements are of another type is passed as an `ObjectSpace` view (a `Dict` action space for `TransformAction<,,>`):
  `PyKwargs` binds by Python name and does not view spaces itself. `DictInfoToList` returns Gymnasium's list of infos from
  its own `Step` and `Reset`, which hide the base ones (`new`): through `IVectorEnv` or a `VectorEnv` reference the infos
  stay a dict, since `VectorStepResult.Infos` is one, so it belongs outermost, as in Gymnasium. The vector
  `NormalizeObservation` and `NormalizeReward` keep Gymnasium's float32 statistics and float32 accumulated returns,
  updated with NumPy 2's promotions (the discounted return computed in float64 through the integer `1 - terminated`, then
  stored as float32). The vector `HumanRendering` resizes each sub-env's frame with `Cv2.Resize(INTER_LINEAR)` (DR-23)
  into one mosaic shown through the viewer factory (DR-20), kept as `LastMosaic` in pygame's `(width, height, 3)`
  layout; the vector `RecordVideo` writes through `IVideoWriter` as the single-env one does (DR-25). The parity cells
  leave `RecordEpisodeStatistics`' `t` (wall-clock time) out on both sides. The unit tests of `RecordVideo` over ten
  CartPoles take about 100 s per framework: each recorded step renders ten frames and `PngVideoWriter` writes their
  3000 × 800 mosaic (two rows of five) as a PNG.
- **DR-22, `AsyncVectorEnv` on threads** (found in P6.2, 2026-11-10): each sub-env has a worker, the counterpart of
  Gymnasium's process and pipe: the worker builds its sub-env from its function on its own thread and runs Gymnasium's
  worker loop (`reset`, `reset-noop`, `step` by autoreset mode, `_call`, `_setattr`, `_check_spaces`, `close`), and the
  caller's end keeps the results queued in send order, as a pipe does, so a timed-out call's result is the next one read,
  as in Python. `ThreadPool` chains a sub-env's calls as pool tasks; `DedicatedThreads` and `ThreadPerEnv` run them on
  long-lived background threads; a sub-env whose `render_mode` is `human` moves to a dedicated thread in pool mode, before
  its first call. A worker's exception ends it (its sub-env closed) and comes back from the wait after Gymnasium's three
  error lines, the trace line being .NET's stack trace; the worker is then detached and any later call through it fails as
  Python's dropped pipe does (`AttributeError`). The constructor still builds Gymnasium's dummy env from the first function
  and copies its spaces (disposing an env disposes its spaces); a function that throws makes the constructor throw.
  `shared_memory` keeps Gymnasium's two copy paths through the new `create_shared_memory`/`read_from_shared_memory`/
  `write_to_shared_memory` (an ordinary array buffer; threads write their slots under a lock), so Gymnasium's quirks stay:
  without shared memory a reset after an episode end keeps the pending autoreset (measured: the next step's rewards
  `[0, 0]` there and `[1, 1]` with shared memory, in both libraries), and `Text`/`OneOf` buffers are read once.
  `close(terminate=True)` cannot kill threads: it stops waiting, each worker closes its sub-env once its current call
  returns, and disposal waits for that. Not covered by DR-22 and left out, as questions for the maintainer: Gymnasium's
  `context` and `daemon` (process start options) and the custom `worker` function (a pipe protocol), whose two tests are
  `[OpenBugs("P6.2-worker")]`. Gymnasium's live `AsyncVectorEnv` runs in the parity suite: its workers are spawned from
  the embedded interpreter once `multiprocessing.set_executable` names a real `python.exe`.
- **DR-36, `FuncEnv`, the benchmarks and the step-API dispatcher** (found in P6.7, 2026-11-10): `FuncEnv<TState, TObs,
  TAct, TReward, TTerminal, TRenderState, TParams>` (namespace `Gym.Experimental.Functional`, with an untyped `FuncEnv`)
  keeps Gymnasium's functions as virtual methods raising `NotImplementedError`, sets its options by Python name
  (`IDynamicAttributes` holds the undeclared ones) and reads `get_default_params` after them. `transform(func)` rebinds
  Python's attributes, which C# methods cannot be: Gym.NET's stores the transformed functions as run-time attributes of
  the same Python names (`step_info` for `transition_info`, Gymnasium's own naming), so `PyAttributes.GetAttr` sees them
  while direct C# calls run the originals; no Gymnasium test exercises it. `Gym.Utils.Performance` reads its clock through
  a replaceable `Performance.Time` (Python's `time.time`, which Gymnasium's tests monkeypatch) and disposes what its loops
  drop, so collection does not leak into the measurement; a zero elapsed time raises Python's `float division by zero`.
  `step_api_compatibility` over tuples is `StepApiCompatibility.Dispatch` (the class takes the name), marked
  `[PyName("step_api_compatibility")]`, which now also targets methods; its single-env flags keep Python's `and`/`or`
  operands (an int `done` comes back as itself), where the typed conversions return `bool`s.
- **DR-34's generated docs** (found in P10.4, 2026-11-09): an options record copies each constructor `<param>` onto its
  property, and a `<paramref>` inside it is a broken reference on a property (CS1734 wherever documentation is generated,
  so in a consumer's build too). The generator now turns it into a `<see cref>` to the sibling property, or `<c>` text for
  a parameter with no property (a wrapper's `env`), and drops `<typeparamref>`s the record does not declare.
- **D13 without a learning library** (found in P10.3, 2026-11-09): "rewrite one CartPole example on the new API" is read
  as one example that learns CartPole with Gym.NET alone. The old runners' NeuralNetwork.NET (last released in 2020)
  depends on ImageSharp 1.0.0-beta0005 and the Alea GPU runtime; the cross-entropy method over linear policies solves CartPole-v1 in seconds
  and shows `MakeVec`, `Make`, seeding and `human` rendering. A maintainer who wants a neural-network example gets a
  design question for its library first.
- **Float64 rewards with the registry** (found in P5.4, 2026-11-09): `EnvSpec.RewardThreshold` is a `double?`, so
  Taxi-v4's threshold, which Gymnasium registers as the `int` 8 (its annotation says `float`), prints `8.0` in
  `pprint` and `to_json`; the value is equal. Pinned by `RegistrationLiveParityTests.Spec_TaxiRewardThreshold_IsFloat64`,
  as rewards are float64 everywhere in Gym.NET.
