# Gym.NET Design Review

| | |
|---|---|
| **Status** | Reviewed on 2026-11-08 in ten rounds, on branch `journey1`: 36 designs, all decided. Four more rounds on 2026-11-10 settled the questions P6 and the per-env parity coverage raised: DR-37…DR-47. Round 15 on 2026-11-11, after P8, decided CarRacing's score glyphs (DR-48) and the packages' licence (DR-49), and the maintainer confirmed the ten [interpretations](#interpretations-to-confirm). Round 16, the same day, took the C# code review's first findings (DR-50…DR-54; DR-50's R1 was withdrawn before commit, its gain having been measured on Debug builds), and round 17 closed AOT (DR-55). Round 18 settled the 0.2 leftovers (DR-56) and the registry's namespace (DR-57), and round 19 the obsolete shims that keep 0.2 code compiling (DR-58). Rounds 20 to 25 (2026-11-11 and 2026-11-12) settled the showreel and MuJoCo (DR-59…DR-74). Rounds 26 to 29 on 2026-11-21 are the 1.0 architecture review and the questions implementing it raised (DR-75…DR-97); they also decided the code review's R7–R9 (DR-85), R12 (DR-92), R13 (DR-75) and R21 (DR-79). Round 30, the same day, made every frame byte perfect the goal: MuJoCo's family by family (DR-98, amending DR-67 and DR-71) and the viewers one screen pixel per frame pixel (DR-99). On 2026-11-22, round 31 lettered the showreel's scenes (DR-100); round 32 decided the docs site's "now playing" screen (DR-101) and round 33 the environment docs ([`ENV_DOCS_PLAN.md`](ENV_DOCS_PLAN.md), DR-102…DR-106), both on `master`; round 34 the showreel's shape, backdrop and expert agents (DR-107…DR-110). On 2026-11-23, round 35 designed the environment page whole, CartPole's and then HalfCheetah's, as the template of all 23 (DR-111…DR-116), on `master`; rounds 36 and 37 (DR-117…DR-123, `Make`'s wrappers and the JAX tabular envs) are `journey1`'s, and round 38 (DR-124, DR-125), on `master`, gave MountainCarContinuous an example and confirmed toy text's figure. Round 39 (DR-128…DR-131, `master`) chose Blackjack's hand by a seed search, drew the grid worlds' anatomy on the board, left the notices as they are, and settled the ids: `journey1`'s round 38 (DR-124…DR-127, which reuses `master`'s two) is renumbered when it merges. On 2026-11-24, round 42 (DR-138…DR-141, `master`, after `journey1`'s rounds 40 and 41 took DR-132…DR-137) made the sidebar remember its sections as NumSharp's does, opened the env families by default, wrote the samples in Gymnasium's shape with their types spelled out, and gave every C# sample its Python twin. Pending: `DictInfoToList`'s result shape (deferred, with a direction), and the code review's R15, R19 and R23. [Coverage by phase](#coverage-by-phase) says which phases can start |
| **Rule** | A step of [`COMPLETION_PLAN.md`](COMPLETION_PLAN.md) starts only when every design it depends on is **Approved** here. A design that isn't listed is not approved: a new design question found while implementing goes through the same review before the step that needs it continues |
| **How** | The maintainer reviewed the implementation design choices four at a time, before any phase from R0 to P9 started. Each answer is recorded with the options that were on the table and what it changes in the plans. Where a later round refined an earlier answer, both rows stay and the earlier one points to the later |
| **Earlier decisions** | D1–D23 in [`GYMNASIUM_PORT_PLAN.md`](GYMNASIUM_PORT_PLAN.md) §20 and [`COMPLETION_PLAN.md`](COMPLETION_PLAN.md) §3 still stand unless a design here overrides them |
| **Versions** | The rows below call the release 2.0, its number until 2026-11-12; it ships as 1.0.0 ([`COMPLETION_PLAN.md`](COMPLETION_PLAN.md)'s D23): read 2.0 (2.0.0) as 1.0 (1.0.0), 2.x as 1.x, and 3.0, where the `[Obsolete]` shims and `Gym.Legacy` go, as 2.0 (DR-56, DR-58) |

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

### Round 11: the questions of P6 and the parity coverage (2026-11-10)

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-37 | `draw.aaline`/`draw.aalines` (D4, P7) | **Read pygame-ce 2.5.8's `src_c/draw.c` for these two functions only** and translate their algorithm, so MountainCar's frames become pixel-exact and the P7 cells (8 MountainCar frame cells, 2 draw corpus cells) become gates. D4's black-box rule stays for every other drawing function. The translated code derives from LGPL-2.1 source: it carries pygame-ce's notice in THIRD-PARTY-NOTICES. Asked on 2026-11-10 where it lives, the maintainer chose **inside `Gym.Rendering`**: the translation replaces the model there, and the `Gym.NET.Rendering` package's license expression gains the LGPL term named by `draw.c`'s own header, `LGPL-2.0-or-later` (the Library GPL version 2 or any later version), making it `Apache-2.0 AND LGPL-2.0-or-later` (not taken: an LGPL assembly of its own behind an interface). *Implemented 2026-11-10 in `Draw.Aaline.cs`; pygame's two-point read past its buffer is taken as 0 and accepted either way by the parity corpus. The license expression was replaced on 2026-11-11 by DR-49: the package is Apache-2.0* | A documented deviation gated outside the measured gap; more black-box probing | P7, P4 (MountainCar) |
| DR-38 | `AsyncVectorEnv`'s process options (DR-22) | **`worker` as a public delegate** over the per-sub-env channel, the counterpart of Gymnasium's pipe, with Gymnasium's eight arguments (index, env function, pipe, parent pipe, shared memory, error queue, autoreset mode, semaphore); the default worker becomes public as Gymnasium's worker loop. `daemon` maps to background threads; `context` is accepted and has nothing to choose (documented). Gymnasium's two custom-worker tests port; the arity check becomes a compile-time fact | Accepting `context`/`daemon` and refusing a custom worker; leaving all three out | P6.2 |
| DR-39 | `PlayPlot`'s chart backend (DR-36) | **ScottPlot adapters in the viewer packages**: `Gym.NET.Rendering.Avalonia` and `Gym.NET.Rendering.WinForm` each ship a ScottPlot-backed `IPlotFigure` (titles, ticks, labels, live updates). The core keeps the plain built-in `PlotFigure` and gains no chart dependency | The built-in figure only; the built-in figure with an embedded bitmap font | P6.7 |
| DR-40 | Shared spaces (D19) | **One owner per space, enforced**: a space records the env that owns it, and handing it to a second live env throws at once (`InvalidOperationException` naming the fix: give each env its own space). Ownership clears when the env disposes the space; a wrapper that passes its env's space through is not an owner | Envs copying the spaces they are given; keeping the rule documented only | P2 (spaces), every env and wrapper |

### Round 12: `play`'s display, `transform`, `DictInfoToList`, the async autoreset (2026-11-10)

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-41 | `play`'s display and event queue (DR-36) | **Instances with a process-wide default**: a `PygameContext` holds a display and its event queue; `Play.Run`, `PlayableGame` and the display helpers take an optional context, and the static `Display`/`EventQueue` forward to the default context, so line-for-line ports keep pygame's spelling. The maintainer added "consider also `Play.RunAsync`": its shape is DR-44 | Keeping pygame's process-wide statics only | P6.7 |
| DR-42 | `AsyncVectorEnv(shared_memory=False)` and a full reset after an episode end (DR-22) | **Fixed in Gym.NET, as a documented deviation**: a full reset clears the pending autoreset in both copy paths, so sync and async agree in every sequence. The parity cell pins Gym.NET's `[1, 1]` against Gymnasium's `[0, 0]`; the release notes list it | Keeping Gymnasium's quirk bit for bit | P6.2 |

Two answers came back for a second look. `FuncEnv.transform` ("undecided, rethink the design and suggest again") was
re-asked in round 13 (DR-43). On `DictInfoToList` the maintainer asked "can we inherit NumSharp's
`ConcurrentOrderedDictionary`, and can we use `virtual` instead of `new`": the class is `sealed` in NumSharp 0.80.0 and
enumerates its values (it is an `IReadOnlyList<TValue>`), and `VectorStepResult` is a struct holding an `Info`, so an
override needs the list form to be an `Info`. The options were re-asked in round 13.

### Round 13: `transform`, `DictInfoToList`, `Play.RunAsync`, `get_keys_to_action` (2026-11-10)

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-43 | `FuncEnv`'s functions and `transform` (DR-36) | **Python's function attributes as delegate-typed properties**: `Initial`, `Transition`, `Observation`, `Reward`, `Terminal`, `StateInfo`, `TransitionInfo` (and `StepInfo`, which Gymnasium's `transform` creates) are settable delegate properties under Gymnasium's names, so a call reads like a method call and `transform` is Python's seven assignments. Implementers override protected `…Core` methods (the properties start bound to them) or assign the properties. `Transform` takes a typed `IFunctionTransform` (`TDelegate Apply<TDelegate>(TDelegate function) where TDelegate : Delegate`, generic over every signature as `jax.jit` is) or an untyped `Func<Delegate, Delegate>`; Gymnasium's `step_info` quirk is kept | A functions record replaced by `transform`; run-time attributes (P6.7's first reading); dropping `transform` | P6.7 |
| DR-44 | `Play.RunAsync` (DR-41) | **`Run`'s loop on a thread, and the thread pool where the env allows it**: `RunAsync` returns a `Task` that completes when the game ends; cancellation acts like Escape or the close button. The maintainer added "certain envs might be async-friendly and can run on the thread pool, especially envs that render nothing; see if we can support both": one loop body, written against an awaitable frame pacer and the env's `StepAsync`, runs on a dedicated thread (blocking pacing, as pygame's clock) for a thread-bound env and on the thread pool (awaited pacing, no thread held) otherwise, as DR-22's auto-switch decides | A second, separately written async loop | P6.7 |
| — | `DictInfoToList`'s results | **Deferred.** The maintainer picked "unseal NumSharp's `ConcurrentOrderedDictionary` and derive `Info` from it" with the note "Defer, consider wrapping `ConcurrentOrderedDictionary` and exposing what we need": the direction is composition (an `Info` built over the collection, exposing what Gym.NET needs) rather than inheritance. Until it is designed, `DictInfoToList` keeps its hiding `Step`/`Reset` | `InfoList : Info` with overriding methods; keeping the hiding methods for good | P6.6 |

### Round 14: `get_keys_to_action`, OpenCV's float32 paths, key layouts, readings (2026-11-10)

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-45 | `get_keys_to_action` (DR-36) | **A virtual method on `Env`, absent until overridden** (the maintainer's "consider inheritance"): `Env` declares `GetKeysToAction()` with `[PyName("get_keys_to_action")]`, whose base raises Python's `AttributeError`; an env with a mapping overrides it. `PyAttributes` counts a `[PyName]` virtual method as present when the runtime type overrides it, unless the env's overridable `HasPythonAttribute(name)` hook says no (MountainCar's continuous mode). Wrappers inherit the absent base, as Gymnasium 1.x's wrappers forward no attributes; `play` checks the env and then its unwrapped env, as Gymnasium does. The maintainer asked "are you sure `Dictionary<object, object>`?": the return type is typed by what Gymnasium returns, key-code combinations to the env's actions, and is listed for confirmation | The delegate-typed property P6.7 built; a method with a presence attribute | P6.7 |
| DR-46 | OpenCV's float32 resize (DR-23) | **Translated too, as DR-23 says**: OpenCV 4.13.0's float32 `INTER_AREA` (integer and fractional scales) and `INTER_LINEAR`, bit for bit against live `cv2` on a float32 corpus, after measuring whether IPP and the SIMD dispatch change float32 results. `ResizeObservation` still refuses non-uint8 frames, as Gymnasium's does | Leaving float32 out, documented | P6.4 |
| DR-47 | Keys from the viewers (`IInteractiveEnvViewer`) | **The layout's key, as pygame reports it**: both viewers report SDL's layout-dependent keycode (the key labeled A gives `K_a` on any layout). Avalonia switches from `PhysicalKey` to its layout key; WinForms' Windows virtual keys already follow the layout, and its docs, which claimed US positions, are corrected | The key's US position in both viewers | P6.7 |

The fourth question of round 14 listed six readings recorded earlier (DR-02, DR-29, DR-32, P5's Blackjack and Taxi
consequences, `HumanRendering` in the core, D13's one example). The maintainer answered "after we finished
implementing, remind me of all these": they stay as built until then ([Interpretations to confirm](#interpretations-to-confirm)).

### Round 15: CarRacing's score glyphs, the licences, the readings (2026-11-11)

Asked when P8's implementing was finished. The first question offered four ways to treat the glyph data after DR-27's
licence check; the maintainer answered "Ask this question again as if license/GPL has no importance", and the question
was asked again on technical grounds alone (embedded recorded data, a managed FreeType over the font file, native FreeType
over it). Upstream PR #13 (a 2023 Aether CarRacing that P8.4 supersedes) is left to the maintainer: nothing is posted.

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-48 | CarRacing's score glyphs (DR-27) | **Embed the recorded data, DR-27 as designed**: `Assets/fonts/freesansbold-42.json` in Gym.NET.Environments, read by `ScoreFont` on the first score render, so every CarRacing frame and observation is Gymnasium's out of the box and CarRacing gets goldens. The maintainer added "Make sure to handle all cases of what might be plotted in that box": the data covers every character `f"{self.reward:04.0f}"` can show (the digits, `-`, and the letters of `nan` and `inf`: 15 glyphs, 109 kerned pairs), `FormatScore` gives Python's text for every double (`0nan` for a NaN of either sign, `0inf`, `-inf`, every digit of the integer a large double holds), and a box wider than the frame is clipped as pygame clips it. The font's one ligature among those characters (`fi`, which no score contains) is refused rather than drawn wrong. A parity cell compares the asset with a live recording | A managed FreeType translation over the embedded font file; native FreeType over it (both rejected by DR-27 before); first asked with the licence weighed: no glyph data with a public opt-in loader, a separate GPL-labelled package, the score left out | P8.4 |
| DR-49 | The packages' licences (DR-37's term, interpretation 6) | **Every package under the repository's base licence, Apache-2.0**: the maintainer's note on confirming the readings, "all of our licenses are unified and same as base license of the repo and not random, set it all to the base license". `Gym.NET.Rendering` drops DR-37's `Apache-2.0 AND LGPL-2.0-or-later`; translated sources and recorded data keep their sources' notices as attribution in their files and in THIRD-PARTY-NOTICES.md | DR-37's combined expression for `Gym.NET.Rendering` | P10 (packaging) |

The third question reminded the maintainer of the ten readings under [Interpretations to confirm](#interpretations-to-confirm):
"All ten stand", with the licence note DR-49 records.

### Round 16: the C# code review (2026-11-11)

The maintainer asked whether the design is sound C#; [`CSHARP_CODE_REVIEW.md`](CSHARP_CODE_REVIEW.md) answers with 23
findings, and this round decided the first four groups of them.

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-50 | Performance (review R1–R3) | **All three, results unchanged**: Box2D's small math members (every operator of `b2Vec2`, `b2Vec3`, `b2Mat22`, `b2Mat33`, `b2Rot`, the static `b2*` helpers, 53 in all) are `[MethodImpl(AggressiveInlining)]`, measured 5.9× (LunarLander) and 6.4× (BipedalWalker); `Transform.Smoothscale`'s shrink keeps its four channel sums in locals (3.0×) and rents its temporary from `ArrayPool`; CarRacing reuses its pre-flip canvas across renders, while the flipped surface stays a new object per render, as Gymnasium's `self.surf` is. *Corrected the same day, before commit:* R1's 5.9× and 6.4× compared a Debug build of the engine with a Release one (the benchmark's libraries were Debug builds, review §4). Measured in Release, R1 gains 1.05× and 1.12×, inside the runs' spread, so it was reverted; R2 and R3 stand, CarRacing stepping 1.6× as fast at 3.6 MB a step instead of 7.1 | R1 and R2 only; no change | Gym.Box2D (withdrawn), Gym.Rendering, CarRacing |
| DR-51 | Results that own their arrays (R10) | **`StepResult`, `ResetResult`, `DoneStepResult`, `VectorStepResult` and `VectorResetResult` are `IDisposable`**: `Dispose` releases the arrays the result holds, the observation's and the info's (nested ones included), the same arrays an `NDScope` would keep, so `using var step = env.Step(action);` replaces the hand-written `step.Observation.Dispose()` | Callers disposing by hand | Gym |
| DR-52 | Render modes (R11) | **`Gym.Envs.RenderModes` constants** (`Human`, `RgbArray`, `Ansi`, `DepthArray`, `RgbdTuple`, `StatePixels` and the `_list` modes), each Gymnasium's own string; the literals stay valid | Strings only | Gym |
| DR-53 | The human-mode viewer contract (R8, R9) | **Kept as it is**: the asynchronous `IEnvironmentViewerFactoryDelegate` and `IEnvViewer.Render(Image)`, with the maintainer's "Add a TODO to support via async path": a TODO at the delegate and at each of the six call sites that block on it | A synchronous factory with frames instead of ImageSharp images; a synchronous factory only | Gym, the viewers |
| DR-54 | Nullability (R16) | **`<Nullable>enable</Nullable>` in Gym and Gym.Environments, warnings staying warnings**: the files without a nullability context are annotated (`NullEnvViewer`, `Gym.Legacy`'s `Step`, `VecEnv` and `IVecEnv`), and `Gym.Internal` keeps an explicit `#nullable disable` while its deletion is reviewed | Nullable with warnings as errors; per-file opt-in as before | Gym, Gym.Environments |

The second question also offered deleting the 0.2 leftovers (R5, R6) and renaming the factory delegate (R7). The
maintainer did not take the rename, and asked to see each leftover double-checked for use and value before deciding:
[`CSHARP_CODE_REVIEW.md`](CSHARP_CODE_REVIEW.md) §3.7 lists all 21. Six are used by nothing; fifteen are reachable only
through `Gym.Legacy`'s info type `Gym.Collections.Dict` (a first count said seven and fourteen, missing that
`DefaultConstructor` calls `ReflectionHelper`'s extension methods).

### Round 17: the leftovers, AOT, the registry's namespace (2026-11-11)

Asked with the leftovers' double-check. The maintainer deferred the leftovers, asking that each answer show what it
costs backward compatibility and what replaces each type: §3.7 of the review now has both, per type and per way of
removing them, and the question goes back with them. On the registry's namespace (R14) the maintainer asked what a
namespace is and what Gymnasium does with it; the review's R14 row answers, and that question goes back too.

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-55 | AOT and trimming (review R17) | **Not supported**, the maintainer: "We can't AOT, NumSharp can't." NumSharp compiles its array kernels at run time (`DynamicMethod`), so Gym.NET adds no trimming or AOT annotations; the release notes list the limitation, and the getting-started guide says what a .NET 10 file-based app needs (`#:property PublishAot=false`), since those publish AOT by default | Document and annotate (`[RequiresUnreferencedCode]`, `[RequiresDynamicCode]`, `IsAotCompatible=false`); document only; later | Gym, Gym.Environments (docs only) |

### Round 18: the leftovers and the namespace, asked again (2026-11-11)

Asked with what round 17 requested: each way of removing the leftovers with what it costs 0.2 code and what replaces
each type (re-checked: six unused, fifteen reachable through `Dict`), and the namespace explained, with Gymnasium's
use of it, on the question's pages.

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-56 | The 0.2 leftovers (review R5, R6) | **Remove the six unused, `[Obsolete]` on the fifteen**: `CircularQueue<T>`, `CircularQueueThreadSafe<T>`, `GenericCallFactory`, `InvalidActionError`, `AlreadySteppingError` and `NotSteppingError` are deleted; `Dict` and its machinery (`Expando`, `Merger`, `IMergable`, `IMergable<T>`, `MergePrefer`, `ExpandableList` and its generic and interface, `DefaultConstructor` and its generic and exception, `ReflectionHelper`, `DictInlineInitializationException`, `DictMergingException`) carry `[Obsolete]` naming each replacement and go in 3.0 with `Gym.Legacy`; their members are documented, footguns included, and their code is unchanged (compared with the commit before, token by token). The two ambiguous `cref`s are fixed. The maintainer added "Then consider each if we could allow an api with backwards compatability": the review's §3.7 answers type by type. Only keeping a type keeps code naming it compiling. `InvalidActionError` could come back only as `InvalidAction`'s base, catching CarRacing's errors but not the envs' `AssertionError`; `Info` could take `Dict`'s `Get<T>(key)` and `Clone()`, though a `Get<T>(key, T)` overload would capture today's `Get(key, fallback)` calls and throw (measured). None was added; the migration guide maps each type and each `Dict` member | `[Obsolete]` on all 21; remove all 21 and type the legacy info as `Info` (the port plan's §4.4); keep all 21 unchanged | Gym |
| DR-57 | The registry's namespace scope (review R14) | **Per thread and async flow**: `EnvRegistry.CurrentNamespace` is stored in an `AsyncLocal<string?>`, so two threads registering inside scopes of their own at once each get their own namespace; threads and tasks started inside a scope inherit it, and single-threaded code behaves as Gymnasium's global. The lock that guarded the static goes with it. Footgun, documented: a value set inside an `async` method does not flow back to its caller. Four tests pin it, each failing against a process-wide static (checked) | One namespace per process, as Gymnasium's module global | Gym (registry) |

### Round 19: backward-compatible APIs (2026-11-11)

DR-56's follow-up ("consider each if we could allow an api with backwards compatability"), asked twice. First the
three APIs the type-by-type look found: the maintainer took both, with "What else? similar". Then the "similar" ones,
found by compiling 0.2.0's own tests (`666746d`: the CartPole, Aether LunarLander and Box tests) against 2.0, where
the compiler listed every break: the maintainer took all six. Left out, with the reason on the question's page:
`Env.Seed(int)` (Gym.NET's `check_env` mirrors Gymnasium's seed-deprecation warning, so every env would start
warning), `RewardRange` (FrozenLake has its own; Gymnasium 1.0 removed it), `Render("rgb_array")` (0.2 returned an
`Image`), `float s = box.Sample()` (NumSharp's own conversion) and the Aether engine.

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-58 | `[Obsolete]` shims for 0.2 code, each naming the 2.0 spelling and removed in 3.0 with `Gym.Legacy` | **Eleven shims.** `Info.Get<T>(key)` (the legacy `Dict`'s typed read; no `Get<T>(key, T)` overload, which would capture today's `Get(key, fallback)` calls, measured) and `Info.Clone()` (= `Copy()`), these two not obsolete. `InvalidActionError` back as `InvalidAction`'s base, `AlreadySteppingError` as `AlreadyPendingCallError`'s, `NotSteppingError` as `NoAsyncCallError`'s: 0.2's catches compile and catch Gymnasium's errors (one level each in Gymnasium's hierarchy; messages and mapped Python types unchanged), never the envs' `AssertionError`. `StepResult`'s four-element `Deconstruct` (done = terminated or truncated). `ResetResult<TObs>`'s implicit conversion to `TObs` (drops the info, leaving its arrays undisposed). `Env.CloseEnvironment()` (and a default member on `IEnv`), calling `Close()`. `Box(low, high, dType)` for doubles and arrays: the parameter keeps 0.2's name `dType`, since with `dtype` a named `dtype:` call bound to it or turned ambiguous (found by the build: four calls in Gym). `CartPoleEnv(viewerFactory)`: `NullEnvViewer.Factory` or `null` headless, another factory human mode. `LunarLanderDiscreteActions` in `Gym.Environments.Envs.Box2D` and in a revived `Gym.Environments.Envs.Aether` (a twin under DR-24's unique-names rule: a file importing both gets CS0104), `LunarLanderEnv.ContinuousMode`, and 0.2's float constants `SCALE`, `MAIN_ENGINE_POWER`, `SIDE_ENGINE_POWER`, `INITIAL_RANDOM`. With them 0.2's CartPole test compiles whole, its Box test but for NumSharp's conversion, and its LunarLander test gets past the namespace to the Aether engine | Nothing more (the guide maps it); a `Get<T>(key, T)` overload; `Env.Seed(int)`, `RewardRange`, `Render(mode)` | Gym, Gym.Environments |

### Round 20: the front-page video (2026-11-11)

The maintainer asked for "a retro video that will be on the front page" now that so many envs are supported. Four
questions, each option a page built from real frames of the envs played by the policies the video would use, with the
delivery sizes measured on this machine (10 s of all envs at 960x480: GIF 2.9 MB at 15 fps, WebP 3.1 MB, MP4 0.44 MB).
The maintainer's answers are quoted in the rows.

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-59 | The front-page video | **Synthwave, playful one/two/four layouts, a GIF in the README with the MP4 linked, the visually distinct ids, a parity segment.** A synthwave look (neon grid floor, striped sun, chrome pixel lettering); a layout that switches between one, two and four envs ("A playful animation, showing sometimes one, somtimes two or 4 games, we have many envs"); the README opens on a 12 s GIF loop (800x450, 15 fps, 3.9 MB) linking the 60 s MP4 (1280x720, 8.4 MB), both under `docs/media`; the 13 ids that look different ("Except for duplicates, continous and non continous are looking same, dedup": CartPole-v0, MountainCarContinuous-v0, LunarLanderContinuous-v3 and CliffWalkingSlippery-v1 left out, FrozenLake-v1 and BipedalWalkerHardcore-v3 kept); and Gymnasium beside Gym.NET on LunarLander-v3 and CarRacing-v3 with the count of differing pixels, measured against Gymnasium running in Python (0 over 300 and 331 records; any difference stops the tool). `tools/showreel` makes it from the envs' own `rgb_array` frames under scripted policies (Gymnasium's heuristics for LunarLander and BipedalWalker, value iteration over the toy-text envs' `P`, pure pursuit for CarRacing). *Read as* (to confirm): both parity columns replay the recorded every-step renders, since LunarLander ages its particles in `render()` and a live panel would draw others | Arcade attract mode, terminal boot or 8-bit cartridge as the video's look; a 30 s trailer, a 60 s stage-by-stage tour, a wall-only loop; a WebP hero; an MP4 player uploaded to GitHub; all 17 ids; a HalfCheetah-v5 teaser; a chiptune soundtrack | README, `docs/media`, `tools/showreel` |
| DR-60 | The docs site's main page | **A DocFX site like NumSharp's, its main page the video with the 8-bit cartridge select below it** ("This should be part of the docfx main page", "And below it 8-bit cartridge"). `docs/docfx.json` builds the `docs/` pages, the API reference of Gym, Gym.Environments, Gym.Box2D, Gym.Rendering and Gym.Rendering.Avalonia, and the landing page in `docs/website-src`. The cartridge select is the site's environment gallery: 13 looping WebPs at a fifth of the frame size and `envs.json` from `tools/showreel`, driven by `templates/gymnet/public/main.js` (attract mode, arrow keys, the video's 5x7 font). `.github/workflows/docs.yml` deploys it to GitHub Pages from master. *Read as* (to confirm): no Gym.NET site existed, so this round creates one; `docfx.json` sits in `docs/`, not NumSharp's `website-src`, because DocFX resolves links by source path and the docs pages must be inside its folder; WinForms is left out of the API reference (Ubuntu cannot build `net*-windows`); links to files the site does not publish (plans, notices, CLAUDE.md) go to GitHub; Pages has to be enabled once in the repository settings. *Since 2026-11-22 the "now playing" screen plays each env's loop at full size, and the panel stays in view while the reader scrolls the cartridges: DR-101* | The README only | `docs/`, `.github/workflows/docs.yml` |

### Round 21: MuJoCo as supported envs (2026-11-12)

The maintainer asked to "Establish the poc port to a stable version to be well put envs we support normally like
other envs", and settled the route before the questions: "we don't care about porting, we only care about env working
as expected and being byte exact". Four questions followed, on frames, models, the MuJoCo pin and where the code lives,
with what each costs from `MUJOCO_ASSESSMENT.md`. The maintainer's answers and comments are quoted in the rows.

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-61 | MuJoCo's route (DR-30, DR-33, DR-35) | **The proof of concept's specialized core becomes the MuJoCo envs; MuJoCo itself is not ported.** The 21 ids that run on the pinned MuJoCo, and `Pusher-v4`'s refusal, become registered envs like the others, byte for byte with Gymnasium at every layer. The core keeps refusing, by name, any model feature Gymnasium's bundled models do not use. DR-30's general managed port and DR-33's compiler subset and renderer port are replaced. DR-35's postponement ends with this round | DR-30's general port of everything the envs reach; native MuJoCo through its bindings | P9 |
| DR-62 | MuJoCo frames (DR-31) | **Gym.NET draws the scene itself, aiming at byte-exact frames** ("Try to make it byte-exact. after all its basic draws"). A managed rasterizer draws what MuJoCo's renderer draws: geoms, lights, the default camera, the floor's texture. It serves `rgb_array`, `depth_array`, `rgbd_tuple` and human mode through Gym.NET's viewers. Every frame is compared with Gymnasium's in the live suite, and whatever stays inexact is measured and documented | Refusing the render modes for now; MuJoCo's own renderer ported (DR-31) | P9 |
| DR-63 | MuJoCo models (DR-19) | **Gymnasium's 14 bundled models, embedded as MuJoCo's compiled models**, recorded once from the pinned MuJoCo as DR-48 recorded CarRacing's glyphs. `xml_file` accepts their file names (`walker2d.xml` gives Walker2d-v5 the v4 model); any other path is refused with the list of names | Users' own compiled models as well; an MJCF compiler subset | P9 |
| DR-64 | The MuJoCo pin (D6) | **MuJoCo 4.0.0** (2026-11-06). Gymnasium v1.4.0 runs all 21 ids on it, measured on 2026-11-12. The parity stack pins `mujoco==4.0.0`, and the core follows 4.0.0's engine | 3.15.0, the version every proof-of-concept trace verified | P9 |
| DR-65 | Where the engines live (DR-16) | **Inside Gym.Environments, without projects of their own.** MuJoCo's core is part of Gym.Environments. The maintainer added "Also, Box2D should just be in Gym.Environments", so Gym.Box2D's sources move there too, keeping their `Gym.Box2D` namespace | A Gym.MuJoCo project packed like Gym.Box2D; a separate Gym.NET.MuJoCo package | P8, P9 |

### Round 22: the MuJoCo product's names, frames and surface (2026-11-12)

Three questions the Round 21 rows left open, asked before the env classes are written: how users reach MuJoCo's model
and data, the names, and what MuJoCo's frames are measured against.

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-66 | MuJoCo's names | **MuJoCo's own capitalization for the namespaces, Gymnasium's for the classes.** The engine is namespace `Gym.MuJoCo` in `src/Gym.Environments/MuJoCo`, the envs `Gym.Environments.Envs.MuJoCo` in `Envs/MuJoCo`: `MujocoEnv`, and per DR-24 `AntEnvV4`/`AntEnvV5` and their siblings, each `[PyName]` with Gymnasium's class name | `Gym.Mujoco`, the Python module's spelling | P9 |
| DR-67 | What MuJoCo's frames are measured against (amends DR-62) | **Measure only** ("Because at the bottom line they all draw simple drawings"). Gym.NET draws its own frames (DR-62) and compares them with the frames Gymnasium draws in its default setup; the differences (pixels, largest channel delta) are measured and documented, and no cell requires equal frames. DR-62's byte-exact aim stays a direction, not a gate | Mesa's llvmpipe pinned in the parity stack as a byte-exact reference; the parity machine's GPU as the reference, its cells pinned to that machine | P9 |
| DR-68 | How users reach MuJoCo's model and data | **Gymnasium's reach under .NET names.** First deferred ("Defer, compare to python apis and suggest these again"); `MUJOCO_ASSESSMENT.md` §15 then compared the Python APIs: Gymnasium's envs, `utils` and tests read about 45 members of `MjModel`/`MjData` and call 9 `mj_*` functions, of MuJoCo's 625 + 238 attributes and 485 callables. `Gym.MuJoCo` exposes exactly that reach in PascalCase, each member `[PyName]`d to Python's name: `MjModel` (`Nq`, `Nv`, `Opt.Timestep`, `Stat.Extent`, `ActuatorCtrlrange`, `BodyMass`, `GeomRgba`, ...), `MjData` (`Qpos`, `Qvel`, `Ctrl`, `CfrcExt`, `Cinert`, `Body(name).Xpos`, ...), the static `Mujoco` (`MjStep(model, data, nstep)`, `MjForward`, `MjResetData`, `MjRnePostConstraint`, `MjKinematics`, `MjName2Id`, `MjStateSize`, `MjGetState`, `MjSetState`), the enums `MjtState` and `MjtObj`, and `utils`' three functions. Arrays are writable views of the core's own arrays, as Python's are. More members can be added later without a breaking change | A: the same reach with MuJoCo's own names (`data.qpos`); B: everything the core holds (about 110 model fields and 85 data arrays); D: the env's own methods only | P9 |

### Round 23: MuJoCo's renderer (2026-11-12)

Three questions DR-62 and DR-67 left open, asked once the 22 ids had every parity layer but frames (`5371db1`): how
much of Gymnasium's rendering surface Gym.NET exposes, what the human window does, and what the frame cells assert.

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-69 | The MuJoCo renderer's surface (DR-62) | **`MujocoRenderer` only, with everything it needs inlined into it** ("We inline into MujocoRenderer all we need, reproduce simply but with parity"). `env.Render()` serves `rgb_array`, `depth_array` (the raw depth buffer, flipped, as Gymnasium returns it), `rgbd_tuple` and `human`; the scene it builds and the rasterizer that draws it are internal to `MujocoRenderer`, reproducing what MuJoCo's renderer draws for Gymnasium's scenes without exposing MuJoCo's visualization API. Gymnasium's `BaseRender`, `OffScreenViewer` and `WindowViewer` are not ported: the inventory keeps them with this row as the reason, and markers, overlays and segmentation renders have no .NET counterpart. Adding them later is not breaking | Gymnasium's three classes under .NET names (`[PyName]`); those plus MuJoCo's scene API (`MjvScene`, `MjvCamera`, `MjrRender`, ...) | P9 |
| DR-70 | `render_mode="human"` for the MuJoCo envs | **Gymnasium's full `WindowViewer` controls**, in Gym.NET's viewers: mouse orbit, pan and zoom (`mjv_moveCamera`'s math), Tab through the model's cameras, Space to pause and → to advance one step (a paused `render()` waits, as Gymnasium's does), S/F run speed, D render every frame, C contact forces, E frames, R transparency, 0–4 geom groups, T capture, H the text menu, Esc close. Mouse input joins the viewer contract (keyboard only before), the menu is drawn as `mjr_overlay` draws it, and contact forces and frames are drawn as MuJoCo's decorations. Inside `MujocoRenderer` (DR-69) | Frames only, as the other envs; camera controls only | P9 |
| DR-71 | What the MuJoCo frame cells assert (amends DR-67) | **A measured baseline, plus Gym.NET's own frames as goldens.** Each frame cell measures Gym.NET's frame against Gymnasium's (differing pixels, largest channel delta, a histogram, diff PNGs) and fails when the frame is worse than the checked-in baseline for that id and mode by more than Gymnasium's own render-to-render noise (measured 2026-11-12 on the reference GPU: Humanoid-v5 15 to 21 pixels, Hopper-v5 1 to 19); the cells are pinned to the reference machine and Inconclusive elsewhere, and a deliberate improvement re-records the baseline. The MuJoCo goldens also record Gym.NET's own frames, which the Python-free replay requires bit for bit (strict on win-x64). Equality with Gymnasium is still never required | Report only; the baseline gate without frame goldens | P9 |

### Round 24: the new envs in the showreel (2026-11-12)

With the MuJoCo envs done (P9), the maintainer asked: "We finished up working on so many new envs, recreate the video and
insert between existing envs the new envs". No question was put: DR-59 settled the look and the rules, and the
request settled where the new envs go. The choices below apply DR-59 and are read as stated.

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-72 | The new envs in the showreel (amends DR-59) | **The eleven MuJoCo envs, one v5 id each, inserted between the existing sections.** The video runs intro, classic control, MuJoCo's control tasks (CartPole stays on beside InvertedPendulum, then InvertedDoublePendulum, Reacher, Pusher), Box2D, MuJoCo's locomotion (CarRacing stays on beside HalfCheetah, then Hopper and Walker2d, then Ant, Humanoid, Swimmer, HumanoidStandup), toy text, the parity segment, a finale flipping through all 24 ids, the outro: 85 s. DR-59's rule drops the v4 twins (they look the same). Policies: HalfCheetah plays the POC's ARS autopilot; Augmented Random Search (`tools/showreel/Ars.cs`, the paper's V2-t) trains InvertedDoublePendulum, Swimmer, Hopper, Walker2d, Ant, Humanoid and HumanoidStandup, committed as JSON under `tools/showreel/policies`; InvertedPendulum (state feedback), Reacher (inverse kinematics) and Pusher (operational-space control) are scripted, each because its trained policy looked wrong on screen. The seven locomotion envs name their model's `track` camera, since with no camera named Gymnasium 1.4 forces the free camera, which never follows a body; InvertedDoublePendulum gets a closer `default_camera_config`. The captions say what holds for MuJoCo ("MUJOCO PHYSICS · BIT FOR BIT, IN C#"); "pixel for pixel" leaves the captions that now cover MuJoCo frames, and the parity segment stays on LunarLander and CarRacing. The README loop mixes HalfCheetah, Ant and Hopper in; its ordered dithering moves from Bayer scale 3 to 4 so the GIF stays under the 5 MB that GitHub's image proxy serves (4.6 MB at DR-59's 800x450, where scale 3 made 5.4 MB with the MuJoCo panels). The site's gallery has 24 cartridges, each showing the exact `Gymnasium.Make` call. *Read as* (to confirm): insertion by section with a panel carried across each boundary, rather than shuffling old and new envs one by one; v5 ids only; no MuJoCo parity row, since its frames are within one level of Gymnasium's, not equal | Shuffling the new envs among the old one by one; appending them after toy text; a MuJoCo parity row (observations bit for bit, frames within one level); Gymnasium's default MuJoCo cameras | `tools/showreel`, `docs/media`, the site's gallery, README |

### Round 25: the README's showreel cut, and the agents as examples (2026-11-12)

The maintainer asked: "1. the gif is not great, redo the showreel but -2/5 of the time of the mp4 2. then in examples/ i
want all of the playing examples you did". Four questions were put as HTML pages built from a 51 s cut of the showreel,
encoded every candidate way and measured (sizes and zoomed crops of the same frame), and from the examples' layouts.

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-73 | The README animation (amends DR-59 and DR-72) | **The whole showreel cut to 51 s, as an animated WebP.** "-2/5 of the time of the mp4" is 85 s minus two fifths, and it is the README's animation that becomes the showreel; the 85 s MP4 stays as it is, linked from the README and on the site. `docs/media/gym-net-hero.webp` runs every section of the video in its order and with its hand-overs, all 24 envs, the parity segment and the finale, each panel on screen about 60% as long; a panel too short for the moment its episode is known for joins mid-episode so that moment still lands (MountainCar's summit, Acrobot's crossing, the toy-text goals); LunarLander, whose story is its descent (touchdown at step 117 of 299), plays from its start at twice its speed (corrected after the maintainer's "The lunar is broken in the part its shown alone": the first cut joined it at step 125, after the descent, and its unrendered steps left a cloud of unaged exhaust particles; LunarLander and the BipedalWalkers now render every step, as human mode does). 800x450, 15 fps, `libwebp_anim` quality 75: full colour, where one GIF palette for 24 envs dithers and smears pink into the grey MuJoCo skies (measured on the 51 s cut: WebP 7.2 MB, GIF 19 MB). It replaces DR-59's 12 s GIF loop, and is the site's fallback where the video does not play; `hero --format gif` still writes a GIF. This corrects DR-72's GIF reasoning: GitHub's camo proxy refuses images over 5 MB, but it only fronts images on other hosts, and the README's come straight from `raw.githubusercontent.com` (checked with GitHub's own markdown renderer), so there was no 5 MB ceiling; the GIF encoder is DR-59's Bayer scale 3 again | Both cut to 51 s (the MP4 too); the MP4 cut to 51 s with a short README loop; a 34 s cut (two fifths of the MP4); animated AVIF (3.0 MB at 1280x720, but older browsers show one frame); the GIF as it was (19 MB for 51 s); a 640x360 GIF (12.5 MB) | README, `docs/media`, `tools/showreel`, the site's landing page |
| DR-74 | The playing agents as examples (amends D13) | **One project per env the showreel plays, each with its agent, a window and the keyboard.** The 24 ids get 24 projects under `examples/`: 23 new ones named after their env (`examples/Hopper`, `examples/FrozenLake8x8`, ...), and CartPole's hand-written controller joins `examples/CartPole` as a second mode beside the cross-entropy method. Each project plays one seeded episode with the agent the showreel uses (the same seeds) and prints its steps and return; `--render` plays it in a window (Avalonia; the MuJoCo envs with DR-70's viewer); `--play` hands the env to the keyboard through Gymnasium's play utility (`Play.Run`), with a key map per env. The playable HalfCheetah moves from `tools/parity/mujoco/halfcheetah_poc/play` to `examples/HalfCheetah`, its window, autopilot and commands kept. Each project is self-contained (its own agent code, and the trained ones their policy file), and they are in `Gym.NET.sln` as CartPole's is, so an API change that breaks one fails the build. *Read as* (to confirm): keyboard play where a few keys can drive the actions and the frames keep up: not Pusher, Ant, Humanoid and HumanoidStandup, whose 7, 8 and 17 motors would need 14 to 34 keys held in combinations, and, measured while implementing, not Hopper, Walker2d and Swimmer either, whose frames Gym.NET draws in 0.1 to 0.3 s against 8 to 40 ms of simulated time per step, so Gymnasium's play, a frame per step, would run 7 to 37 times slower than real time (those examples print why; HalfCheetah keeps its own real-time game); each motor of an arm or a walker on a key pair, as the playable HalfCheetah's; the turn-based toy texts wait for a key; the trained policies ship as `policy.json` beside each program | Grouping by env family; grouping by method; one Play project for all 24; the showreel playing the examples' agents (one copy; each keeps its own); an ARS `train` command in the examples (the showreel's `train` retrains the policies); `--record` | `examples/`, `Gym.NET.sln`, `tools/parity/mujoco/halfcheetah_poc` |

### Round 26: the 1.0 architecture review (2026-11-21)

The maintainer asked, before the release PR: "Run me down on what changes beside new envs, show me that using ask html
questions tool. I want to find architectural/abstractional decisions i would disagree with. Gym.NET is a base
abstraction that must comply with gymnasium and also (maybe) maintain compatability." Twenty-four decisions beyond the
environments were put as HTML pages in four questions (the core contract, how Pythonic the surface is, registration and
memory, vector/rendering/compatibility/packaging), and after his "Formatted code, showing also diffs before and after"
each page showed the same task in 0.2 and in 1.0 as a highlighted side-by-side diff, Gymnasium's Python, the reasons,
the costs and each alternative as a diff from 1.0. He disagreed with thirteen and added two requests. The directions he
gave are below; the shapes they leave open (rewards, the env contract, `make`'s viewer and `new` paths, the viewer
contract, the unwrapped env by type, the env-id constants, seeds, typed infos and `Info`'s equality, and the details of
DR-77 and DR-81) are round 27's.

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-75 | What the review keeps | **Eleven decisions stand as built**: discrete values as 0-d int64 arrays (DR-08); disposable record-struct results (DR-51); `StepAsync` completing on the caller's thread, not forwarded by wrappers (2026-09-25); `make`'s kwargs by Python name, with generated options records (DR-11, DR-34); the string attribute protocol (`GetWrapperAttr`, DR-18); the registration compiled into consuming projects (DR-17); `EnvSpec` as built (R13 not taken); the pygame rendering layer (D4); one environments package with both engines, the viewers on it (DR-16, DR-65); the 0.2 bridges until 2.0 (DR-58); Gymnasium's class names with suffix twins (DR-24) | Each page's alternatives | — |
| DR-76 | NumSharp 0.81.0 (review B4, C6) | **Gym.NET moves to NumSharp 0.81.0 and uses NumSharp directly**: "NumSharp 0.81.0 has ALL NumPy/Python exceptions so we should use NumSharp exceptions and if its not there then either inherit or create our own"; "Use NumSharp directly, latest 0.81.0". Every Python and NumPy exception Gym.NET raises is NumSharp's, `OverflowError` and `ZeroDivisionError` included (0.80.0 threw .NET's `OverflowException` and `DivideByZeroException` for them); `Gym.Exceptions` keeps only `gymnasium.error`'s classes, the stop-gap `NotImplementedError`, `ImportError` and `ModuleNotFoundError` go, and a class NumSharp lacks derives from NumSharp's nearest or is created. NumSharp's gaps N-1 to N-4, fixed in 0.81.0, close. How seeds reach `np.random.default_rng` is round 27's | NumSharp 0.80.0 for 1.0 | Gym, Gym.Environments, the tests, `eng/NumSharp.props` |
| DR-77 | Predetermined values as enums (review B1, B6, D1) | **Enums that parse from and print as Gymnasium's strings**: "RenderMode should be string parsed to Enum and have overloads taking enum or string"; "An enum of Render is supposed to be supported. And RenderResult should be castable to NDArray or Image to simplify."; for the vector envs, "support string but use enums for predetermined values". The render mode, `AutoresetMode` and `VectorizeMode` become enums (`AsyncExecutionMode` is one already); every API that takes one takes the enum or Gymnasium's string; `RenderResult` converts to its frame (`NDArray`) or an ImageSharp `Image`. Round 27 settles the property types, the conversions and the overloads | Strings with constants (DR-52, as built); Python-enum classes (as built) | Gym, Gym.Environments, the viewers |
| DR-78 | `Info` (review A5) | **A converting `Get<T>` as an extension, and 0.2's `Information` kept obsolete**: "1. Have static extension with same namespace that will allow Get<int>("x_position") 2. Keep Information obsoleted to migrate away from old. 3. Conside ralso Typed infos. Generated [GymDict] records per env over the dict, as BlackjackObservation does for tuples." `Get<T>` (today a plain cast, which throws on the 0-d arrays NumPy scalars arrive as) becomes an extension in a static class of `Gym.Envs` that converts NumPy scalars and numbers; the results get 0.2's `Information` back, [Obsolete] and naming `Info`. Typed infos and `Info`'s equality (R12) are round 27's | `Info` as built | Gym, Gym.Environments |
| DR-79 | Spaces (review B5, R21) | **.NET-typed, few new types, no `object` returns, and a static `Spaces` class**: "I think I the solution more .NET typed and less new types or object returns. Also whats spaces? a class? We should have static class Spaces". Typed overloads per space beside the untyped entry, and a static class mirroring the `gymnasium.spaces` module. Where `Spaces` can live beside the `Gym.Spaces` namespace, and which members get typed forms, are round 27's | Python's dynamic signatures (as built); typed composite spaces (`Tuple<T1, T2>`) | Gym |
| DR-80 | Array lifetimes in user code (review C3) | **`[NDScoped]` first, `using` beside it**: "I prefer using NDScoped attribute but using is also working". The guides and examples show NumSharp.Build's `[NDScoped]` (and `[NDScopedAsync]` for async methods) as the way to release what an env returns, with `using` as the alternative. Refined by DR-96 (round 29): seven stateful classes kept arrays a caller's scope disposed, so every class that keeps an array detaches it first, and the guides add the rule "build envs and spaces outside the scoped method" | `using` only (as the docs show it) | docs, examples |
| DR-81 | Copying a space (review C4) | **`Clone()` does what `DeepCopy()` does**: ".DeepCopy() or .Clone() should do same". The one-owner rule (DR-40) stays | — | Gym |

### Round 27: the shapes round 26 left open (2026-11-21)

Twelve questions in three batches, each option an HTML page with its change as highlighted side-by-side diffs from 1.0
as built, its gains and its costs. Where the maintainer asked a question back, the answer is in the row.

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-82 | Rewards (review A4) | **`double`, as built**: the decision of 2026-09-25 stands. Gymnasium's int and float32 rewards arrive as doubles of the same value | The reward as a NumPy scalar (`NDArray`) in the env's dtype; `double` plus a `RewardValue` scalar | — |
| DR-83 | The env contract (review A1) | **As built** ("Skip"): `Env<TObs, TAct>` stays the typed contract; `Dispose()` closes the env (it calls `Close()`), and `CloseEnvironment()` stays 0.2's [Obsolete] alias of `Close()` until 2.0 | A typed `IEnv<TObs, TAct>` beside the class; that interface with `CloseEnvironment()` kept for good | — |
| DR-84 | DR-77's shape | **C# enums, with hand-written string twins**: `RenderMode`, `AutoresetMode` and `VectorizeMode` are enums that parse from and print as Gymnasium's strings (`RenderMode.RgbArray` and "rgb_array"); the env constructors take the enum, and each has a twin taking the string, written by hand (the maintainer: "But why generated, have hardcoded is not better?": hand-written twins are explicit, documented and debuggable, at the cost of one more constructor per env with a render mode); `Make` and `MakeVec` have overloads for both; the `RenderMode` properties become the enum; `RenderResult` converts explicitly to its frame (`NDArray`) and to an ImageSharp `Image`; specs, metadata, `make`'s kwargs and the attribute protocol keep Gymnasium's strings. Refined by DR-93 (round 28): the `RenderMode` properties stay `string?`, since Gymnasium accepts render modes no enum lists, and the enum constructor is the twin | Constructors taking only the enum (strings through `RenderMode.Parse`); one struct that is both enum and string; generated twins | Gym, Gym.Environments, the viewers |
| DR-85 | The viewer contract (review D2, R7–R9; replaces DR-53) | **Asynchronous end to end, with a separate synchronous path written for synchronous use**: "BUT we have complete separate sync overload optimized for sync"; then "IAsyncEnvViewer inherits IEnvViewer, theres one factory and some envs are inherietly async. Will that work out?" It does: `IEnvViewer` is the synchronous contract (`Render(NDArray frame)`, the `(H, W, 3)` uint8 frame, and `Close()`); `IAsyncEnvViewer : IEnvViewer` adds `RenderAsync` and `CloseAsync`; one factory creates either. Every viewer therefore has the synchronous path, which the envs' human mode uses inside `Step` and `Reset`; an inherently asynchronous env awaits `RenderAsync` when its viewer is an `IAsyncEnvViewer` and falls back to `Render` otherwise. ImageSharp leaves the contract (R9), nothing blocks on a task (R8), the factory follows .NET's naming (R7), and `CloseEnvironment()` stays an [Obsolete] default member until 2.0 | Frames with a synchronous factory only; ImageSharp kept with a synchronous factory; one interface carrying both paths | Gym, the viewers, every env with `human` mode |
| DR-86 | DR-79's shape | **A static `Spaces` class inside `Gym.Spaces`**, so `using Gym.Spaces;` gives `Spaces.Discrete(4, seed: 0)` as `from gymnasium import spaces` gives `spaces.Discrete(4)`, with typed overloads per space beside the untyped entry (`Discrete.Sample(NDArray? mask)`, `Box.Sample()`, `MultiDiscrete`/`MultiBinary.Sample(NDArray? mask)`) and `Seed` typed where Gymnasium's shape is fixed (an int, `BigInteger`, for the simple spaces; `Tuple` and `Dict` keep their nested forms). The class shares its namespace's name (CA1724), which only code inside namespace `Gym` sees. Refined by DR-97 (round 29): the typed overloads take Gymnasium's whole parameter list (`mask` and `probability`), and MultiDiscrete's mask is Gymnasium's tuple of arrays as `IReadOnlyList<NDArray>`, not one `NDArray` | `Spaces` nested in the `Gymnasium` facade | Gym |
| DR-87 | Reaching the unwrapped env by type (round 26's request) | **As built** ("Keep as is"): `(CartPoleEnv)env.Unwrapped`. A C# cast on an `IEnv` or `Env<,>` variable cannot be user-defined (CS0552, CS0553) | `env.Unwrapped<TEnv>()`; a typed `Make<TEnv>` whose result casts | — |
| DR-88 | Env ids as constants (round 26's request) | **`Gymnasium.Envs`**: string constants for the ids Gym.NET.Environments registers ("We should have this as an assistant for the default coming envs"): `Gymnasium.Make(Gymnasium.Envs.CartPoleV1)`, or `Make(Envs.CartPoleV1)` with `using static Gym.Gymnasium;`. A top-level `Envs` class cannot sit beside the `Gym.Envs` namespace | `EnvIds` in Gym.Environments | Gym |
| DR-89 | `make` and viewers (review C2) | **`Make` and `MakeVec` take the render mode and a viewer factory**: `Make("CartPole-v1", RenderMode.Human, viewerFactory: ...)`; "We can define factory still for when passing viewerFactory as null": a null factory falls back to the app-wide `Gymnasium.HumanViewerFactory`, then to `DependencyNotInstalled`, as DR-20 resolves it | `Gymnasium.Wrap(new Env(...), id)` (make's wrappers around an env built with `new`) | Gym |
| DR-90 | Seeds (DR-76's remainder) | **`Reset(BigInteger? seed)` stays, over NumSharp's `np.random.default_rng`**. The maintainer asked "But what value do we get from long vs BigInteger?": at a call site none, since `Reset(seed: 42)` already compiles (an int converts implicitly), while `BigInteger` carries Python's unbounded seeds and the 128-bit entropy seed Gymnasium reports for an unseeded env, which `NpRandomSeed` must hold. Generators come from `np.random.default_rng` directly; `Seeding.NpRandom` remains only as Gymnasium's `seeding.np_random` | `Reset(long? seed)` with a `BigInteger` overload | Gym |
| DR-91 | Typed infos (DR-78) | **Now, for every env**: a generated `[GymDict]` record per env (`HalfCheetahInfo`, `TaxiInfo`, ...) with a property per info key, converted from the `Info` an env returns (`HalfCheetahInfo info = step.Info;`), and keeping the dict's access too ("But HalfCheetahInfo also supports (double)(NDArray)step.Info["x_position"];"). `StepResult.Info` stays an `Info`. Shaped by DR-94 (the members) and DR-95 (which envs, the names, the inheritance) in round 28 | Typed infos later, additively | Gym, Gym.SourceGenerators, Gym.Environments |
| DR-92 | `Info`'s equality (R12) | **`Info` compares by reference, and Python's equality is `PyEquals`**; the generated info records, being records, compare by value through both `==` and `PyEquals` ("But now that Info might be generated type, we can have proper equality through PyEquals and =="). `GetHashCode()` stops being `Count` | Python's value equality on `Info` (as built) | Gym, the tests |

### Round 28: what implementing rounds 26 and 27 raised (2026-11-21)

Three questions no row above answered, found while implementing DR-84 and while preparing DR-91, each option an HTML
page with its change as highlighted side-by-side diffs from 1.0 as built (`ba05397`), its gains and its costs. The
facts on the pages were measured the same day: the info keys by running every registered Gymnasium v1.4.0 env (a reset
with seed 0 and 300 random steps, on the pinned stack), the render-mode strings by searching `refs/Gymnasium`.

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-93 | Render modes outside DR-84's enum (Q1) | **The `RenderMode` properties stay `string?`; the enum is how a mode is passed and parsed**. Gymnasium takes any string as a render mode: `make("CartPole-v1", render_mode="no-render-mode")` warns and builds the env, whose `render_mode` keeps the string (Gymnasium's `test_make_render_mode` asserts it), its tests use "Testing mode", "Test", "Non mode" and "ascii", and a user's env lists modes of its own; an enum-typed property cannot hold them. So `Env`, `IEnv`, the wrappers and the vector envs keep `string? RenderMode`, Gymnasium's exact string, and DR-84's "the `RenderMode` properties become the enum" is withdrawn. The `RenderMode` enum (`Gym.Envs`, the eleven modes `RenderModes` names) is what the constructors, `Make` and `MakeVec` take beside the string, and `RenderModes.Parse`, `TryParse` and `Name` convert between them. With the string the canonical form, each env's constructor that takes the string stays as built and the hand-written twin takes the enum: `RenderMode renderMode`, required and first, since C# refuses two constructors whose parameters all have defaults for `new CartPoleEnv()` (CS0121), and this way every existing call, positional ones included, compiles unchanged and the options records and `make`'s kwargs keep the string | The enum with a `Custom` member and the exact string in a second property, `RenderModeName`; refusing strings outside the enum with `ValueError` | Gym, Gym.Environments |
| DR-94 | Typed info records' members (Q2) | **.NET types, nullable where a key can be missing, with the dict's values kept inside**. Measured: the MuJoCo infos are float64 scalars and a float32 `reward_ctrl` (the action's dtype), with Python floats and ints for some survival rewards; toy_text's `prob` is an int at reset and a float after; Taxi's `action_mask` and Humanoid-v5's tendon values are arrays; CarRacing's `lap_finished` comes only on the step that ends an episode; a reset sends fewer keys, or none. A member is `double` for every scalar (float32 and int values widen exactly), `bool` for a flag, `NDArray` for an array, and nullable when the env does not always send its key; the record keeps the values the env sent, so its indexer (`info["reward_ctrl"]`) returns the float32 scalar as sent and `(double)(NDArray)info["x_position"]` reads as it does on `step.Info`. Array members are the step result's own arrays, disposed with it | NumPy's own values as members (0-d arrays, a cast at every read); the .NET types without the dict inside | Gym, Gym.SourceGenerators, Gym.Environments |
| DR-95 | Which envs get typed info records, and their names (Q3) | **One record per env class that sends keys, named like the class, and inheriting as the env classes do**. 23 records: `FrozenLakeInfo`, `CliffWalkingInfo`, `TaxiInfo`, `CarRacingInfo`, and the MuJoCo `<Name>InfoV4`/`<Name>InfoV5` (DR-24's suffixes; v4 and v5 send different keys, HalfCheetah-v4 `reward_run` where v5 sends `reward_forward`). One record serves a reset's info and a step's, the keys only a step sends being nullable; the 15 ids whose info is always `{}` get none, and their docs say so. The maintainer: "But do inherit logically like MujocoEnv": the MuJoCo records derive from `MujocoInfo` as the envs derive from `MujocoEnv`, the three toy_text records from `ToyTextInfo`, which holds the `prob` all three send, and every record from `EnvInfo`, which keeps the dict's values | One record per env family with the versions' keys merged (15); member-less records for the envs whose info is always empty; separate reset and step records (30) | Gym, Gym.SourceGenerators, Gym.Environments |

### Round 29: what preparing DR-80 and DR-86 found (2026-11-21)

Two questions, each option an HTML page. Q1 came from a probe written to check DR-80 before documenting it (scratchpad,
not committed): 34 cases, each run unscoped and then in chunks of 25 steps, each chunk inside a caller's `[NDScoped]`
method woven by NumSharp.Build 0.81.0, compared bit for bit. Q2 came from reading Gymnasium's `MultiDiscrete.sample`
while implementing DR-86.

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-96 | Gym.NET under a caller's NDScope (Q1; DR-80's premise) | **"Fix the wrappers now": every class that keeps an array it built during `Reset`, `Step` or `Render` detaches it where it stores it, and `[NDScoped]` is then documented first**. NumSharp's scope tracks every array constructed on the thread while it is open and disposes, when it closes, each one the method did not return; NumSharp's rule for an array cached into a long-lived field from inside a scoped call is `NDScope.Detach` at the caching site. Measured before the fix: the 10 envs tried (classic control, Box2D, MuJoCo, Taxi), the sync, entry-point and async vector envs, rendering, RecordEpisodeStatistics (and its vector form) and the stateless wrappers give identical results; NormalizeObservation, NormalizeReward, FrameStackObservation, DelayObservation, NormalizeObservationVector and NormalizeRewardVector corrupt silently from the first step after a scope closes (NormalizeReward returns NaN), and RenderCollection's kept frames crash the process (AccessViolationException). The fix: an audit of every class that stores an `NDArray` outside its constructor, a detach at each storing line, and `ScopeSafetyTests` in the unit gate (every registered id, wrapper and vector wrapper, the render modes; a hand-opened `NDScope` against no scope, bit for bit). An env or space BUILT inside a scope and kept after it stays unsafe (HalfCheetah-v5 and a `Box` were corrupt in the probe): the guides' rule is to build envs and spaces outside the scoped method, or dispose them inside it, and NumSharp.Build already refuses an `[NDScoped]` method that returns one (NDW003) | Every constructor detaching what it keeps too, so anything may be built inside a scope (a much larger change); `using` first for 1.0 and scope safety after it (DR-80 deferred) | Gym, Gym.Environments, docs, examples |
| DR-97 | The typed `Sample` overloads (Q2; DR-86's list) | **"Overloads to reach parity with gymnasium but also clear api"**: each typed overload takes Gymnasium's whole parameter list, typed, beside the untyped entry. Gymnasium's `MultiDiscrete.sample` takes a TUPLE of int8 arrays, one per entry (nested for a multi-dimensional space), and asserts it, so DR-86's `MultiDiscrete.Sample(NDArray? mask)` could never work. The overloads: `Discrete.Sample(NDArray? mask = null, NDArray? probability = null)`, `MultiBinary.Sample(NDArray? mask = null, NDArray? probability = null)`, `MultiDiscrete.Sample(IReadOnlyList<NDArray>? mask = null, IReadOnlyList<NDArray>? probability = null)` (one array per entry: Gymnasium's tuple for a one-dimensional space, the entries in C order for a multi-dimensional one, which Gym.NET groups into Gymnasium's nested tuples) and `Box.Sample()` (Gymnasium's box takes no mask). The untyped `Sample(object? mask, object? probability)` keeps Gymnasium's exact forms, nested tuples included; a call on the space's own type binds the typed overload unless an argument is typed `object` | A typed list for one-dimensional spaces only, nested ones untyped; MultiDiscrete untyped only | Gym |

### Round 30: every env in every renderer, byte perfect (2026-11-21)

The maintainer's instruction: "ensure every env is able to render in every renderer and byte perfect". Measured first:
every one of the 38 ids that build renders in every mode its metadata lists (Gymnasium's `tests/envs/test_rendering.py`,
mirrored the same day as `TestRendering`), and every frame but MuJoCo's equals Gymnasium's already. Two questions, each
option an HTML page. Q1's pages carried a measurement taken for it: Gymnasium rendering one state eight times on the
reference GPU (RTX 4060 Ti, driver 616.56), in two processes, gives one frame for InvertedPendulum-v5 and HalfCheetah-v5,
three or four for Hopper-v5 (1 to 8 pixels apart) and three to five for Humanoid-v5 (6 to 18 pixels apart; the first
render was the same in both processes).

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-98 | MuJoCo frames byte perfect (Q1; amends DR-67 and DR-71) | **"Like onces we already did, we port mujoco envs completely one by one"**: Gym.NET keeps drawing MuJoCo's frames itself, in managed code (DR-62), and the frames become equal to Gymnasium's, one env family at a time, the way the engine was brought to bit for bit (HalfCheetah first, then the others, each closed against the oracle before the next). DR-67's "measure only" and DR-71's "equality is never required" end: a family whose frames are equal has its frame cells gated on equality, and the measured baseline stays only for the families not yet done. The oracle is Gymnasium's frames on the reference GPU (D4's way, as pygame's `draw.line` was derived from its output): the GPU's rounding, sampling and coverage rules are measured and reproduced. Reading to confirm: where Gymnasium's own frames of one state vary between renders (Humanoid, HumanoidStandup, Hopper and Ant on the reference GPU), "equal" is "one of the frames Gymnasium draws for that state", the only byte-exact target such an oracle has | MuJoCo's GL commands sent to the machine's own OpenGL, equal to Gymnasium on any GPU, with the software renderer as the fallback; Mesa's software OpenGL pinned as the reference and translated bit for bit | Gym.Environments (`Envs/MuJoCo/Rendering`), the parity frame cells, the baseline |
| DR-99 | The viewers on scaled displays (Q2) | **One screen pixel per frame pixel at any scaling**: the Avalonia and WinForms viewers size their client area in physical pixels and draw the frame unscaled, without interpolation, so the window shows the frame's own bytes at 100%, 125%, 150% or 200%, on every monitor it is moved to (on a scaled display it looks smaller than a scaled app's window). Before, Avalonia laid the window out in device-independent pixels and smoothed the frame up, and WinForms was stretched by Windows unless the app declared DPI awareness, as pygame's window is (python.exe declares none). Checked by capturing each window's client area as drawn, for every env | A whole-number zoom (each frame pixel a k×k block, k the largest whole number not above the scale); scaling with the display as other apps do, byte-exact at 100% only | Gym.Rendering.Avalonia, Gym.Rendering.WinForm, MuJoCo's human window (DR-70's pointer coordinates) |

### Round 31: the showreel's scenes, lettered (2026-11-22)

The maintainer asked: "Kindly split into segments lettered so i can reorder and shorten time". Three questions were put
as HTML pages showing a frame of the video for each candidate segment: how finely to cut it (14 scenes, 9 sections or 18
layouts), how a letter shortens the video (dropping and reordering only, a short form per letter, or a length per
letter), and which outputs follow the letters (the MP4 alone, or the README's cut too).

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-100 | The video as lettered scenes (amends DR-72) | **"14 scenes": the MP4 is 14 self-contained scenes, lettered A to N, played in the order a line of letters gives**. The order is `tools/showreel/video-order.txt`, committed, the maintainer's to edit (`--order` overrides it for one run); the tool reads it from the checkout, so a new order needs no rebuild. A title, B classic control, C MuJoCo control, D LunarLander, E the walkers, F CarRacing, G HalfCheetah, H Hopper and Walker2d, I MuJoCo's four, J toy text's four, K FrozenLake, L parity, M finale, N outro. The maintainer's note on the question, "What letter, what i meant is the order in which we show each scene": a letter names a scene, and the line of letters is the order the scenes play in. Each scene starts and ends on an empty stage, so any order plays and leaving a letter out drops that scene alone; a letter may come twice. The timings inside each scene are the 85 s video's. The two hand-overs that crossed a section boundary become scene-local: CartPole enters C afresh beside InvertedPendulum (0.3 s after it, so it never meets B's CartPole on its way out), CarRacing leaves at the end of F, and G is HalfCheetah alone, joined at step 40. A caption that consecutive scenes share types on once; the header logo spans every scene but the title and the outro; the parity scene drops out with `--no-parity`. In letter order the video is 82.8 s (the CarRacing and HalfCheetah pair is gone). `scenes` prints where each scene of the order starts and the video's length; `video --letters` renders a preview with each scene's letter in the corner, to the temp folder unless `--out` says where, never over the published video or its poster. Outputs, "The MP4 only": the README's 51 s cut (DR-73) keeps the 85 s video's timeline, hand-overs included. Shortening, "Ill say when i need": *Read as* (to confirm): for now an order only reorders and drops; a short form or a length per scene waits for the maintainer's word, and lower-case letters are refused rather than folded, kept free for a short form | 9 sections, A to I (dropping a letter drops a whole family); 18 layouts, A to R (each layout change its own letter, panels flying out and back in between letters); each letter long or short (upper case the MP4's timing, lower case the README cut's); a length per letter (holds stretched or shrunk to fit); both outputs, each its own order; both outputs, one order | `tools/showreel`, `docs/media` |

### Round 32: the gallery's "now playing" screen (2026-11-22)

Rounds 30 and 31 (DR-98…DR-100) were decided on `journey1` and reach master with it. The maintainer, working on
master: "The gif is pixelated, it needs to be high quality (preview is okay to be pixelated)" and "When you scroll down
the right side with NOW PLAYING should be moving with your scroll as it needs to stay in view inside the SELECT YOUR
ENVIRONMENT area." The second needed no question. The first raised one, the screen loops' format, put as three HTML
pages that played each candidate's real loops (CartPole-v1, CarRacing-v3, HalfCheetah-v5) at the screen's size, with
2x zooms and the old 8-bit loop beside them; every candidate had been rendered for all 24 recipes and measured.

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-101 | The "now playing" screen (amends DR-60) | **"MP4, 30 fps", "but it should be autoplayed and not a video with a play button or pause"**. The screen plays each env's loop at the env's own frame size as H.264 (CRF 20, 4:2:0, the matrix tagged BT.601), 30 fps, over the same 6.5 s of play and 1.2 s hold as its cartridge; `tools/showreel`'s `gallery` command writes it beside the cartridge's WebP in the same pass (`images/envs/<id>.mp4`, the manifest's `screen`), and the cartridges keep DR-60's 8-bit WebPs frame for frame. The video is muted, looping and inline, with no controls and no pointer input; where it cannot play by itself (autoplay refused, no H.264 decoder, a missing file) the screen shows the cartridge's 8-bit loop instead, so it never shows a play button. Measured on the 24 recipes: 4.2 MB in all against 4.9 MB for the 8-bit set, Hopper-v5's the largest at 861 KB, 38 to 50 dB against the exact frames (colour edges softened by 4:2:0). Side by side, the panel is `position: sticky` 76 px from the top, under the landing page's 60 px header, and stays inside the gallery body, on windows at least 660 px tall (the tallest panel measured 516 px; on shorter ones a stuck panel would hide its own make() line). *Read as*: the request was about the screen's loop, so the cartridges stay 8-bit; in one column (under 992 px) the panel sits under the shelf and does not stick. *The same day the panel's content became a game manual, with its own sticky floor: DR-102* | Animated WebP, lossy at quality 80, 15 fps (7.7 MB; 13.4 MB at 30 fps); lossless animated WebP, 15 fps (28.9 MB, up to 4.9 MB for one MuJoCo loop) | `tools/showreel` (`Gallery.cs`, `Ffmpeg.cs`), `docs/website-src` (`index.md`, `templates/gymnet/public/main.js` and `main.css`, `images/envs`) |

### Round 33: the environment docs (2026-11-22)

The maintainer: "We need to plan for envs docs properly. the SELECT YOUR ENV side menu needs to be improved AND a button
linking to proper docfx env doc page. We will need to decide how env doc page looks and contains. including indepth,
external links, original source ported from (you can find the sessions using am cli), A GIF playing 1/2 width of the
doc article, api, code examples, observable, reward, alllllll other". Seven questions in three calls, every option an
HTML page built from CartPole's real facts (Gymnasium 1.4.0's docstring, Gym.NET's registry, class, example and
loops): the page's look, the side panel, the page set and how pages are produced; the side panel again after "Try
again, other options", with what the navbar's Environments opens; then which reading of "but with sidemenu" was meant.
The sources were gathered from the code's `Reference:` permalinks and the sessions that built each family (Agentmaster
CLI). The plan is [`ENV_DOCS_PLAN.md`](ENV_DOCS_PLAN.md).

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-102 | The gallery's side panel (amends DR-101's panel) | **"Game manual"**, chosen in the second round. Under the screen and its HUD, a HOW TO PLAY box in plain game words: CONTROLS (the actions), YOU SEE (the observation), SCORE (the reward), GAME OVER (the episode's end); a READ THE MANUAL button that opens the env's page; small links to its `examples/` project, Gymnasium's page and the API; Enter opens the manual while the gallery has focus. The blurb and the make line leave the panel (the page's infobox has the make call). Each cartridge's four lines are written beside its blurb in `tools/showreel/Episodes.cs` and reach `envs.json` with the page's address (and the anchor for BipedalWalkerHardcore-v3 and FrozenLake8x8-v1). The panel grows, so DR-101's 660 px sticky floor is measured again | First round ("Try again, other options"): a spec card with DOCS, EXAMPLE and GYMNASIUM buttons; the panel as it was with one big docs button; PLAY/SPACES/CODE tabs. Second round: a character-select screen with stat bars and START; a HUD on the loop's corners with a button row; an 8-bit pause menu | `docs/website-src/templates/gymnet/public/main.js` and `main.css`, the arcade's markup, `tools/showreel` (`Episodes.cs`, `Gallery.cs`, `envs.json`) |
| DR-103 | An env page's look and contents | **"Floating infobox"**: a box floated right at half the article's width holds the env's loop (the gallery's full-size MP4, playing as DR-101 has it: muted, looping, no controls), a caption naming the agent with its seed, steps and return, and the facts table (ids with their limits, both spaces, reward and threshold, render modes, the make call, class, options and info records, vector env, parity, Gymnasium's page). The sections follow Gymnasium's docstrings and add what Gym.NET has: description, action space, observation space, rewards, starting state, episode end, arguments (Gymnasium's keyword beside the C# parameter), info, rendering, C# beside Python code, agents and examples, vectorized environment, variants, parity with Gymnasium, performance, reference sources, version history, external links, API reference (`ENV_DOCS_PLAN.md` §2). *Read as* (to confirm): tables beside the box shrink to fit next to it rather than drop below it (the mockup's gap); under 768 px the box spans the article above the text; "original source ported from" is the section **Reference sources**, in the docs' own voice (CLAUDE.md: never "ported"), listing every upstream file the env reproduces as a pinned permalink; Gymnasium's MuJoCo action-space figures are linked, not copied, since their credits grant Gymnasium's repository alone | "Loop beside facts": the loop on the left half, the facts on the right, the sections below; "Arcade header": the gallery's 8-bit screen as the page's header with a pixel HUD and PREV/NEXT | `docs/environments/**`, `main.css` |
| DR-104 | The page set | **"23, like Gymnasium"**: one page per page of Gymnasium's environment docs, under `docs/environments/<family>/`, each paired with its Gymnasium twin. The other ids are sections of their page: CartPole-v0, LunarLanderContinuous-v3, CliffWalkingSlippery-v1, each MuJoCo env's v4 (frames measured identical to v5's from the same seed; InvertedDoublePendulum's through v4's camera, which Gymnasium's v5 leaves unused; Pusher-v4 refuses on MuJoCo 3+); BipedalWalkerHardcore-v3 (`#hardcore`) and FrozenLake8x8-v1 (`#8x8`) with their own loops, which their cartridges open. MountainCarContinuous-v0 keeps a page and gets a loop (a docs-only recipe) but no cartridge | "24, one per cartridge" (Hardcore and 8x8 as pages of their own, Mountain Car Continuous folded into Mountain Car); "39, one per id" | `docs/environments/**`, `tools/showreel/Episodes.cs` |
| DR-105 | How the pages are produced | **"Hand-written pages"**: each page written whole in Markdown (HTML only for the infobox), and a unit test, `tests/Gym.Tests/Docs/EnvDocPagesTests.cs`, checks each page's facts table and arguments table against the code: the registered ids, limits and thresholds, both spaces, render modes and fps, the constructor's keywords and defaults, the info record's keys, the loop files. *Read as* (to confirm): the C# samples are written in compiled files under `tests/Gym.Tests/Docs/EnvPageSamples/` and included into the pages by DocFX's code-snippet syntax, so the unit gate runs every one (the guides' rule: every sample compiled and run); the Python twins are run against Gymnasium in the parity venv when written | "Generated + prose": a tool filling the facts in around hand-written prose, checked in CI; "XML docs as source": the prose in each env class's XML docs, built into pages by a DocFX template step | `docs/environments/**`, `tests/Gym.Tests/Docs` |
| DR-106 | What the navbar's Environments opens | **"The landing gallery", "but with sidemenu"**, confirmed as **"Environments page + sidebar"**: `docs/environments/index.md` holds the same SELECT YOUR ENVIRONMENT gallery (the arcade's markup in one include shared with the landing page; `main.js` fills any `.gn-arcade`), with the Environments sidebar (the 23 pages by family) on its left; the navbar's Environments and the sidebar's Overview open it. The landing page keeps its gallery under the video | An overview table (loop thumbnail, ids, both spaces per page); family card grids, as Gymnasium's family pages; the readings not meant: an 8-bit menu inside the gallery's screen, the landing page taking the docs layout with the sidebar | `docs/website-src/toc.yml`, `docs/docfx.json`, `docs/environments/index.md` and `toc.yml`, `docs/website-src/index.md` |

### Round 34: the showreel's shape, backdrop and agents (2026-11-22)

The maintainer asked: "1. Remove the 2 least common vertical pole, keep rest (atleast 1) 2. Shuffle the envs scenes.
mixing the order and groupings - Title reduce to 2.5s - lunar lander landing cut earlier, 4s total - car racing 4s (the
loop part) 3. Let me pick another retro background 4. All of our envs must be good or state of the art. some are really
not good looking. so humanoid, swimmer and so on all must be perfect working - you can find online sota or ready models
to run. Slightly imperfect is also good.", then corrected point 2: "a. Show the humanoid and its kind 2 in scene and
another scene with other 2 - instead of 4 per scene. b. dont shuffle envs order c. move reacher and pusher to a 2 in a
scene". Four questions were put as HTML pages: the scene plan for each pair of vertical poles (gallery stills of the
five pole-like envs, the resulting scenes A to O), eight backdrops animated live in the browser under the video's real
panels, caption and chrome logo, three 4 s windows of CarRacing's episode playing at real speed, and the README
animation's scope. A second question pair followed the agents' research: Farama's own experts for every MuJoCo v5 env
were found on Hugging Face (the agents Farama trained to record the Minari datasets), their repositories state no
licence, and their training code is MIT.

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-107 | The scenes reshaped (amends DR-100) | **"Pendulum + InvertedPendulum" leave the video; the order stays; pairs where there were fours; the title 2.5 s, LunarLander 4 s, CarRacing "The S-bends"**. Fifteen scenes, A to O, in the old order ("dont shuffle envs order"): A title, B classic control (CartPole, then beside MountainCar, then four with InvertedDoublePendulum in Pendulum's old place and Acrobot), C Reacher and Pusher ("move reacher and pusher to a 2 in a scene"), D LunarLander, E the walkers, F CarRacing, G HalfCheetah, H Hopper and Walker2d, I Humanoid and HumanoidStandup, J Ant and Swimmer ("Show the humanoid and its kind 2 in scene and another scene with other 2"), K toy text's four, L FrozenLake, M parity, N the finale (22 ids, its last row two), O outro: 78.4 s in letter order. The title drops its logo in 0.6 s and types the pitch at 64 letters a second. LunarLander plays its descent at its own speed and its 180 steps of rest on the pad eight times faster, so LANDED! still lands in 4 s ("landing cut earlier"). CarRacing plays steps 430-630, the S-bends, picked over the hairpin and the opening zoom. *Read as* (to confirm): a MuJoCo env in the classic four is said by the caption, which retypes "CLASSIC CONTROL + MUJOCO" when it joins (`Scene.CaptionUntil`); Humanoid plays at three-quarter speed and HumanoidStandup lies still while its panel lands, then gets up | Removing InvertedPendulum + InvertedDoublePendulum (the MuJoCo cart-poles, Reacher and Pusher then MuJoCo control's whole scene); Pendulum + InvertedDoublePendulum; the hairpin (steps 96-296) or the opening zoom (0-200) for CarRacing; a shuffled order with mixed groupings (withdrawn by the maintainer) | `tools/showreel`, `docs/media` |
| DR-108 | The backdrop (amends DR-59's look) | **"16-bit dusk beach"**: pixel art drawn at 320x180 and scaled up four times without smoothing, an ordered-dithered (4x4 Bayer) sunset sky in seven bands, the sun half-sunk in the sea with gaps rolling down its lower half, a far skyline with blinking lights, the sun's reflection shimmering on five sea bands, two swaying palm silhouettes (`DuskBeach.cs`, a pure function of time). The synthwave grid stays selectable (`--backdrop synthwave`), DR-59's look kept rather than deleted | Neon city drive; vector arcade; 8-bit dusk; space shooter; demoscene copper bars; vaporwave pastel; green phosphor terminal; today's synthwave | `tools/showreel`, `docs/media`, the site's poster |
| DR-109 | State-of-the-art agents, committed (the maintainer's point 4) | **"Commit, attributed", and "i want all sota weights to be used and served in the tests project or wherever fit"; the examples get them too, "New agents, own copies"**. Twelve published agents: Farama's experts for the eleven MuJoCo v5 envs (SAC for Ant, Hopper, HumanoidStandup, InvertedDoublePendulum, InvertedPendulum, Pusher, Reacher, Walker2d; TQC for HalfCheetah and Humanoid; PPO for Swimmer) and RL Zoo's TQC agent for BipedalWalkerHardcore-v3. `tools/agents/import_agents.py` downloads each `policy.pth` at a pinned Hugging Face revision and writes the deterministic actor as `gym.net-agent/1` JSON into `tests/Gym.Tests/Agents` (layers, activations, the output mode, the action bounds, `make_kwargs`, the source, four PyTorch reference actions); `ExpertPolicy.cs` there reads it in float32 as PyTorch computes. `ExpertAgentTests` holds the reader to PyTorch (6e-7 of the range) and plays each agent's expert episode on Gym.NET's engines (Humanoid 10,563.5 over 1,000 steps, HumanoidStandup 399,827.7, the hardcore walker the whole course); measured over seeds 0-9 they score what their model cards report. The showreel links the folder; the eleven generated examples and HalfCheetah's carry their own copies of the file and the reader (HalfCheetah's autopilot stays for `--autopilot` and its game, Gymnasium reproducing its returns to the last bit). Each file names its source and the licence situation | Downloading at run time (nothing committed); training our own with Farama's MIT scripts (their budgets were 20-100 million steps per env); the examples reading the showreel's files; the examples keeping their agents | `tests/Gym.Tests/Agents`, `tools/agents`, `tools/showreel`, `examples/` |
| DR-110 | The README animation (amends DR-73) | **"Re-cut from the new video"**: the README's WebP plays the video's fifteen scenes in their short form (`Storyboards.ShortScenes`, about 60% as long), in the video's order (`video-order.txt`), over the beach, with the experts: 49.4 s. As DR-73's cut did, a panel too short for its episode's moment joins mid-episode (MountainCar's summit, Acrobot's crossing, Pusher's push, the toy-text goals); LunarLander plays at one and a half times its speed and fast-forwards its rest | The README keeping its own timeline with the new look only; leaving it as it was | `README.md`, `docs/media`, `tools/showreel` |

### Round 35: the environment page, whole (2026-11-23)

The maintainer: "Kindly design with me the env page as it should be full page about the env". Four calls put ten
questions, each option a complete page built from real facts. First CartPole's page, in three drafts, written from
Gymnasium 1.4.0's `cartpole.py`, Gym.NET's registry, class and example, the example's swaying controller recorded
whole (seed 3, 500 steps) and the random policy measured over 1,000 seeded episodes (22.6 steps on average). Then
HalfCheetah's page with the same template, to see whether it holds for a MuJoCo env, with about a dozen arguments,
info keys, v4 and v5, cameras and depth images. Its episode is Farama's TQC expert recorded whole (seed 0, 1,000 steps,
return 17,648.0, as `examples/HalfCheetah` prints it). Its figure is Gym.NET's own frame of the reset pose, with each
joint projected onto its pixel. The specification is [`ENV_DOCS_PLAN.md`](ENV_DOCS_PLAN.md) §2.

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-111 | The page's top (amends DR-103) | **"8-bit banner", "no mention of BIT FOR BIT"; in the infobox, "Keep the badges"**. The page opens with a banner in the gallery's 5x7 pixel font. Its first line is the env's name (white, with an orange shadow); its second line gives the family and the sizes of the spaces (`CARTPOLE`, then `CLASSIC CONTROL  2 ACTIONS  4 OBSERVATIONS`). The lead follows. The banner makes no parity claim. The infobox's last row keeps parity badges, each saying only what its family's tests prove. CartPole shows "bit for bit" and "pixel for pixel". A MuJoCo env shows "physics bit for bit" beside an amber badge with the frames' measured gap (HalfCheetah: 1.8% of the pixels differ at reset) until DR-98 closes it. Its parity section says that bit for bit holds on the reference machine, an x64 CPU with FMA. Without FMA, .NET's math functions differ in their last bits, and a contact-rich episode grows that into another episode (HalfCheetah's autopilot, seed 0: 5,895.8 instead of 5,797.0) | The title with a row of badges (family, actions, observations, ids, bit for bit, pixel for pixel) and buttons to Gymnasium's page, the source, the example and the API; the title and the lead alone. For the infobox: parity in its own section only; a neutral row pointing to that section | `docs/environments/**`, the `gymnet` template (`main.js` draws the pixel lettering) |
| DR-112 | The infobox's episode and the pages' media (amends DR-103's loop) | **"Each example records", "But name the class CartPoleProgram.cs"**. The infobox plays the whole seeded episode that the env's example plays, one frame per step at the env's render fps, not the gallery's 6.5 s loop. CartPole's is 501 frames at 50 fps (40 KB); HalfCheetah's is 1,001 frames at 20 fps (976 KB). It plays as DR-101's screen does: muted, looping, inline, no controls. The caption names the agent, the seed, the steps and the return. Each `examples/<Env>` gets `--record <folder>`, which plays its seeded episode and writes into `docs/environments/<family>/media`: `<id>.episode.mp4` through Gym.NET's `RecordVideo` and `FfmpegVideoWriter`, `<id>.episode.json` (every step's state, action, reward and reward terms, for the anatomy chart, DR-115), and How it works' frame with the pixel of each point its labels mark (DR-114). ffmpeg is needed only to record. Each example's `Program.cs` becomes `<Env>Program.cs`, with its class `<Env>Program` (`CartPoleProgram.cs`). *Read as* (to confirm): the rename applies to all 24 examples, not only CartPole's | One recorder written once in `examples/Shared/PageMedia.cs` and compiled into every example; each example keeping a `media.json` (seed, fps, window) and `tools/showreel` rendering every page's media | `examples/**`, `docs/environments/**/media` |
| DR-113 | The section order and the quick start (amends DR-103's order) | **"Quick start after description", "If we can play the env, then have a command to launch playable / Also Quickstart has to be below description and right now it is now the full width"**. After the banner and the description comes Quick start. It sits below the infobox and spans the article's full width, clearing the float. It holds the make, reset and step lines in C# and Python (Gymnasium) tabs and the package to add. Its part "Play it, or watch the agent" gives the example's `--play` command with its keys where the example is playable from the keyboard, and the command that shows the infobox's agent in a window (`--render`). The other sections follow in this order: how it works, action space, observation space, rewards, starting state, episode end, arguments, info, rendering, episode anatomy, code examples, agents and examples, vectorized environment, variants, parity with Gymnasium, performance, reference sources, version history, references, external links, API reference | The drafted order (no quick start; how it works right after the description); Gymnasium's docstring sections first, in Gymnasium's order, with Gym.NET's additions after them | `docs/environments/**`, `main.css` |
| DR-114 | How it works | **"Diagram + equations"; for MuJoCo and Box2D, "Labels over a real frame"**. The section opens with a frame that Gym.NET's renderer returns, with SVG labels over it in the frame's own pixels. CartPole's frame is step 125 of the infobox's episode, labelled with x, θ, both pushes, and the 12° and ±2.4 m limits. HalfCheetah's is the reset pose without noise, seen from a free side camera 1.6 m away (760x380). Each hinge is a ring on its projected pixel with its motor's gear, and the labels also mark the root's slide, rise and pitch and the floor's friction. The caption says how the frame was rendered. Next come the mechanics as math (DocFX's own math support; its modern template ships MathJax): CartPole's equations of motion and Euler step, HalfCheetah's reward. A table follows with the constants or the joints, Gym.NET's and Gymnasium's names, and footgun notes (CartPole's `TotalMass` and `PolemassLength`, computed once in the constructor). For an engine, the section ends with what simulates the env (MuJoCo's integrator, constraint solver and colliders). *Read as* (to confirm): a toy-text page's figure is the env's own frame of its grid, with the cells labelled the same way | The equations and the constants without the figure; the figure with plain words and no math. For the 3D families: the model drawn flat from its XML (the HalfCheetah mockup's first figure: cleaner lines, but not what the env renders, and each model transcribed by hand) | `docs/environments/**`, `examples/**` (the frame and its pixels, DR-112) |
| DR-115 | Code examples and the episode's anatomy | **"Tabs: C# \| Python"** and **"Chart synced to the video"**. Each code example shows its C#, with a "Python (Gymnasium)" tab holding the same code in Gymnasium; the quick start uses the same tabs. CartPole has six examples: run an episode, start from wider bounds, change the physics, save a frame, many carts at once, episode statistics. HalfCheetah has six as well: run an episode, weigh the costs, follow it with the camera, depth images, read the info typed, many cheetahs at once. A new section, Episode anatomy, plays the infobox's episode again. Beside the video sits an SVG chart of every step: for CartPole, the cart's position, the controller's target, the pole's angle, and each push as a strip; for HalfCheetah, the forward speed, the control cost, and each motor's torque as a row of colour. A cursor follows the playing video, and a live readout gives the step, its state, its reward and the return so far. The chart is drawn at the width it is shown, so its text reads at about 11 px | Code side by side, lines aligned; C# only. For the anatomy: nine key frames 62 steps apart; no such section | `docs/environments/**`, `examples/**` (the trace, DR-112), the `gymnet` template (the tabs, the chart's script) |
| DR-116 | One template for every family | **"Approve for all families"**: HalfCheetah's page, built with CartPole's template, makes it the template of all 23 pages. What the MuJoCo page needed becomes the rule on every page. The Arguments table has three columns: Gymnasium's keyword over the C# parameter, the type over the default, and the effect with its footguns (`xml_file` takes only a bundled model's name: Gym.NET compiles no MJCF, so any other file throws `NotSupportedException`). The anatomy chart is drawn at its own width. The parity badges follow DR-111. Rendering says which camera draws by default: Gymnasium 1.4.0's free camera, which a running cheetah leaves, unless the model's `track` camera is asked for. Version history lists every Gymnasium version and Gym.NET's | Building FrozenLake's page (toy text: a grid, discrete actions, an ANSI render) before recording | `docs/environments/**`, [`ENV_DOCS_PLAN.md`](ENV_DOCS_PLAN.md) §2 |

### Round 38: before the classic-control and toy-text pages (2026-11-23)

E3 built CartPole's page and the template, and the maintainer answered "Lets continue". Two questions were left. The
plan held MountainCarContinuous-v0's media open: it has a page of its own (DR-104) but no example, and every other
page's media come from its example's `--record` (DR-112). DR-114 had read toy text's figure, which round 35 did not
show, as the env's frame labelled. Both were asked in one call before E4, so that E5 does not stop either. Rounds 36
and 37 are `journey1`'s (DR-117…DR-123, `fe6d361`, `c3c4f80`); this round takes the ids after theirs.

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-124 | MountainCarContinuous's media (amends DR-104's docs-only recipe) | **"A new example"**: `examples/MountainCarContinuous` (`MountainCarContinuousProgram.cs`) plays an energy-pumping controller with the continuous force. Like the other examples, it has `--render`, `--play` where the keys allow, and `--record`, so all 23 pages get their media the same way (DR-112). It is the 25th example, and the gallery still has no cartridge for it (DR-104) | DR-104's docs-only recipe in `tools/showreel/Episodes.cs`, written by the `gallery` command without a cartridge entry: no new example, but this page alone would take its media from the showreel | `examples/MountainCarContinuous`, `Gym.NET.sln`, `examples/README.md`, `docs/environments/classic_control` |
| DR-125 | Toy text's How it works figure (confirms DR-114's reading) | **"The env's frame, labelled"**: the `rgb_array` frame Gym.NET renders, for example FrozenLake's 4x4 lake, with SVG labels on its cells. They mark the start, the holes and the goal, the cell indices the observation uses, and the slip directions. It is the same kind of figure as CartPole's and the 3D families' | The env's `ansi` text grid, annotated: closer to how the state is numbered, but not the picture the env shows, and Blackjack has no `ansi` mode | `docs/environments/toy_text`, the toy-text examples' `--record` |

### Round 39: before the toy-text pages (2026-11-23)

E4 finished classic control, and the maintainer answered "Continue, any questions use html questions tool". Four
questions were asked in one call. Two concern E5: Blackjack's default hand is one step, two frames at 4 fps, and a
time series of a grid world's row and column hardly shows where the agent went. Two were left from E4: the ids
`journey1`'s round 38 reuses, and RLPy's notice. `journey1`'s round 38 took DR-124…DR-127 after `master`'s took DR-124 and
DR-125, so this round takes the ids after both.

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-128 | Blackjack's episode (amends DR-112 for Blackjack) | **"Find a handcrafted seed of incredible few gameplay seconds or even emulate a interesting scenario"**: the page plays one hand of the example's basic strategy, chosen by searching Blackjack-v1's seeds 0 to 1,999,999 for the longest hand that wins on exactly 21. Seed 1,574,758 wins it: eight cards against the dealer's ace, a soft 17 turning hard on the way, the dealer standing on a soft 17. That makes 8 frames, 2 s at 4 fps: 7 steps, six hits and a stick (corrected in E5; the row first said 9 frames and 2.25 s, counting the eight cards instead of the steps). `examples/Blackjack --record` plays that seed; the example's default seed stays the showreel's 40. *Read as* (to confirm): "emulate" as choosing the seed that deals the scenario, since the env draws every card from its seeded generator and nothing in it is altered | The example's five default hands back to back (seeds 40-44); one hand from seed 40 (2 frames); a longer hand from seed 81 (4 frames) | `examples/Blackjack`, `docs/environments/toy_text` |
| DR-129 | The round-38 ids both branches use | **"Renumber journey1's at the merge"**: `master`'s DR-124 and DR-125 (`1f613ed`, 14:43, first) stay. When `journey1` merges, its DR-124…DR-127 (`3024093`, 15:13) and every reference to them in its code, tests and docs move above `master`'s highest id | Renumbering `master`'s two to DR-128 and DR-129 now | `DESIGN_REVIEW.md` on both branches, the merge |
| DR-130 | RLPy's notice for Acrobot | **"Leave as is"**: `THIRD-PARTY-NOTICES.md` keeps Gymnasium's MIT notice for every environment, and Acrobot's page names RLPy as the origin of Gymnasium's `acrobot.py`, with its BSD 3-Clause licence | A section with RLPy's notice, pinned to `rlpy/rlpy` `af25d20` | `THIRD-PARTY-NOTICES.md`, `docs/environments/classic_control/acrobot.md` |
| DR-131 | Toy text's Episode anatomy (extends DR-115) | **"The path on the board"**: for the grid worlds (FrozenLake, CliffWalking, Taxi) the chart draws the episode's moves over the env's own first frame, a slip in orange, and a ring on the current cell follows the video; the step strips stay beside it. `main.js` gains a "board" band. Blackjack keeps the time series | The time series of the row and the column, with the strips | `main.js`, the toy-text pages |

### Round 42: the sidebar, and the samples beside Gymnasium's (2026-11-24)

With every env page live, the maintainer asked for four changes, and "Dont deploy or push yet as we are still working
on this": "Have sidemenu work like K:/source/NumSharp sidemenu does when it comes to persisting the
collapsed/uncollapsed state"; "have all environments in sidemenu default state to be open submenus (once persisted and
if changed, remembered)"; "Have all examples `NDArray observation = env.Reset(seed: 42).Observation;` to actually
destruct into two variables to match NumPy counterpart"; and "There are large piece of code not presenting a
counterpart python vs C# with tabs". The first two and the direction of the last two are the maintainer's own; three
HTML questions in one call settled what they left open: how a step reads once the reset is deconstructed (its `info`
cannot be declared twice in C#), how a sample that released the reset with `using` keeps that release, and where the
Python twins go. `journey1`'s rounds 40 and 41 took DR-132…DR-137, so this round takes the ids after them.

| ID | Design | Approved | Options not taken | Applies to |
|---|---|---|---|---|
| DR-138 | The sidebar's state | **The maintainer's**: the sidebar remembers each section the reader opens or closes and restores it on the next page, for 48 hours that slide forward with every visit, with NumSharp's code (`installTocStatePersistence`, `SciSharp/NumSharp` `4deab504`; ported with Gym.NET's comments, its logic identical): a record per toc.json in `localStorage`, toc.yml's `expanded` as each section's default, the current page's own branch never collapsed by a remembered choice. The Environments sidebar's four families open by default (`expanded: true`) | — | `docs/website-src/templates/gymnet/public/main.js`, `docs/environments/toc.yml` |
| DR-139 | Samples that read like Gymnasium's | **"Python's shape"**, with the maintainer's "no var, use the actual types in all examples unless its obvious which is not here. And if line long, \n before the =": a reset deconstructs into `(observation, info)` (`(NDArray observation, Info info) = env.Reset(seed: 42);`) and each step assigns back into the same names after disposing the observation it replaces (`(observation, double reward, bool terminated, bool truncated, info) = env.Step(action);`), as `observation, reward, terminated, truncated, info = env.step(action)` reads; every declaration names its type, except where the right-hand side already does (`new`, a cast); a declaration longer than 85 columns as shown (what a code block shows at 1280 px before it scrolls) breaks before its `=`. In the env pages' samples, the guides, the README, the landing page, `examples/` and the API's examples | Only Reset changes: each step keeps `var (next, ...)` and its hand-over, its info renamed | the samples, the guides, `examples/` |
| DR-140 | A reset released with `using` | **"Dispose the result"**: C# has no `using var (a, b) = ...`, so where a sample released the reset with `using`, the result takes the `using` (`using ResetResult<NDArray> reset = env.Reset(seed: 0);`) and the next line deconstructs it (`(NDArray observation, Info info) = reset;`), which releases the info's arrays too (Taxi's action mask). A result whose observation the sample never reads is not deconstructed | Deconstruct on one line and dispose the observation at the end (the info's arrays dropped) | the samples, the guides |
| DR-141 | Python counterparts | **All four places**: every C# sample with a Gymnasium counterpart has a Python (Gymnasium) tab beside it: the guides (Getting Started, Creating Custom Environments, Migrating's 1.0 samples), the env pages' Arguments sections, and the landing page's quick start (DocFX's own tab-group markup in the hero, which DocFX's script wires like any other group: the same look and keys, and `?tabs=python` in the URL, where DocFX keeps the choice, opens it on Python); the README, which GitHub shows without tabs, has the Python episode below the C# one, and its C# is written on `IEnv` (the maintainer's "In README use IEnv"). Shell commands, NumSharp.Build's package line, 0.2's code and `Gym.Legacy` have no counterpart and stay single blocks. Every Python twin is run against the pinned Gymnasium when written | — (each place was an option; all four taken) | the guides, the env pages, `docs/website-src/index.md`, `README.md` |

---

## Coverage by phase

Every design each phase of [`COMPLETION_PLAN.md`](COMPLETION_PLAN.md) depends on, as of 2026-11-08:

| Phase | Designs it rests on | Can start |
|---|---|---|
| R0 Re-baseline | DR-01, DR-02, DR-05, DR-06; D2, D3, D18 | Yes |
| P0r P0's rest | DR-03 (no static rules), DR-04 (goldens), D20 (CI informational) | Yes |
| P2 Spaces | DR-07, DR-08, DR-09, DR-13 (the source generator), D19 (ownership); since 2026-11-10 DR-40 (one owner per space, enforced) | Yes |
| P3 Registry and checkers | DR-10, DR-11, DR-12, DR-14, DR-17, DR-20, DR-34 | Yes |
| P7 Rasterizer | D4; DR-19 (assets) for `image.load`; since 2026-11-10 DR-37 (`aaline`/`aalines` from pygame-ce's source, inside `Gym.Rendering`, implemented) | Yes |
| P4 classic_control | DR-08, DR-19, DR-20, DR-32 (`MountainCarEnv(continuous)`) | Yes |
| P5 toy_text | DR-08, DR-13, DR-19, DR-20 | Yes |
| P6 Vector and wrappers | DR-21, DR-22, DR-23, DR-24, DR-25, DR-36; since 2026-11-10 DR-38, DR-39, DR-41…DR-47 (`DictInfoToList`'s result shape deferred) | Yes |
| P8 Box2D | D5, DR-26, DR-27, DR-28, DR-29 (its measurement opens P8.1), DR-32; since 2026-11-11 DR-48 (the score's glyphs embedded) | Yes |
| P9 MuJoCo | DR-30, DR-31, DR-33; postponed by DR-35 until 2026-11-12, then DR-61…DR-68 (the engine, the env classes, the parity stage) and DR-69…DR-71 (the renderer); since 2026-11-21 DR-98 (byte-perfect frames, family by family) | Yes (resumed by DR-61) |
| P10 Release | DR-35, DR-36; D9, D13; since 2026-11-11 DR-49 (every package Apache-2.0) | Yes |

## Interpretations to confirm

**Confirmed on 2026-11-11 (round 15): all ten stand as built.** The maintainer added that every package takes the
repository's base licence, which replaces item 6's term (DR-49).

On 2026-11-10 (round 14) the maintainer asked to be reminded of these once implementing is finished, so each stayed as
built until then:

1. The three readings below (DR-02, DR-29, DR-32).
2. P5's consequences: `BlackjackEnv` is `Env<object?[], NDArray>` read by name through `BlackjackObservation`;
   Taxi-v4's threshold is the float64 `8.0` where Gymnasium registers the int `8` (both under "Consequences found while
   implementing").
3. `HumanRendering` in the core assembly, with the frame `Clock` moved into `Gym` (same section, DR-25 and DR-20).
4. D13's one example: the cross-entropy method with Gym.NET alone (same section).
5. DR-45's return type: key-code combinations to the env's actions, in place of P6.7's `Dictionary<object, object?>`.
6. DR-37's license term (added when it was implemented): the question page said `MIT AND LGPL-2.1-or-later`, but
   `Gym.NET.Rendering` was Apache-2.0 and `draw.c`'s own header grants the GNU Library GPL version 2 or any later
   version, so the package says `Apache-2.0 AND LGPL-2.0-or-later`. It does not pack THIRD-PARTY-NOTICES or the LGPL
   text yet (P10's packaging).
7. DR-44's thread choice (added when it was implemented): DR-22's rule (a human-mode env is thread-bound) never applies
   to `play`, which accepts only `rgb_array` envs, so `RunAsync` reads a real window as what binds a game to a thread:
   a game whose display is any viewer but `NullEnvViewer.Factory` (or whose env renders in human mode) runs on a
   dedicated background thread with blocking pacing, and a headless game runs on the thread pool with awaited pacing.
   That keeps the recommended option's thread for a person playing in a window, and the pool for what "renders
   nothing", as the maintainer's note put it.
8. DR-38's custom worker (added when it was implemented): DR-38 does not say how a custom worker meets DR-22's
   execution modes. Each custom worker runs on a thread of its own, whatever the mode, as each of Gymnasium's runs in a
   process of its own, and the built-in workers keep their executors. Python's pipe errors map to `EndOfStreamException`
   (`EOFError`, which NumSharp lacks) and NumSharp's `OSError` (a broken pipe), and values cross the pipe by reference.
9. DR-46's measurement (added when it was implemented): the `cv2` wheels hand float32 `INTER_LINEAR` to Intel IPP by
   default, and IPP's results differ from OpenCV's own kernel (255 of 400 cases, a few units in the last place), while
   float32 `INTER_AREA` does not depend on IPP. IPP's bilinear is not published, so `Cv2.Resize` gives OpenCV's own
   kernel, exact against `cv2` with IPP off, and the gap against default `cv2` is an open cell. To decide: keep that
   reference, or try harder to reproduce IPP's arithmetic black-box.
10. DR-26's one facade deviation (added in P8.1): a vector coordinate beyond float32 (or infinite) in a keyword
    constructor (`CreateDynamicBody(position=(0, 1e39))`, a polygon vertex). pybox2d converts it inside the constructor
    and prefixes the error (`Failed on kwargs for b2BodyDef.position: Converting from sequence to b2Vec2, ...`; a
    polygon vertex even fails as SWIG's `OverflowError`). C# converts the tuple where the caller builds the `b2Vec2`,
    before the constructor runs, so Gym.NET raises the conversion's own `TypeError`, unprefixed. Both sides are pinned
    exactly (`VectorOverflowInKeywords_DeviatesAsDocumented`). Matching it would take an argument type that defers the
    conversion (tuples and `b2Vec2`s alike) through every facade signature; no Gymnasium env passes such a value.

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
- **D19 and Gymnasium's shared spaces** (found in P6.8, 2026-11-10): a Gym.NET env owns and disposes the spaces it
  is given (documented on `Env.ActionSpace`), where Gymnasium's tests share one space between envs, which processes
  copy. The ported tests build a space per env; sharing one is a caller error that shows up late: `AsyncVectorEnv`
  disposes the envs it builds only to read their spaces, which released a test's boxes while the test still held
  them. Since DR-40 it shows up early: the second env's setter refuses the space.
- **DR-36, `play`, `PlayableGame` and `PlayPlot`** (found in P6.7, 2026-11-10): pygame's module-level display and
  event queue are mirrored, process-wide, by `Gym.Rendering.Display` (one window from the viewer factory, DR-20; its
  surface is replaced when the user resizes a resizable display) and `Gym.Rendering.EventQueue` (posted events, plus
  the window's input moved in when `Get` pumps it), with `Event`, `EventType` and `KeyCode` (SDL keycodes read from
  pygame-ce 2.5.8). Gymnasium's tests therefore translate line for line (a callback posts events), and the footgun is
  pygame's: one game per process (since DR-41, one per `PygameContext`; the statics are the default context's). Viewers report input through the new `IInteractiveEnvViewer`. Keys were SDL keycodes by
  physical position on a US layout in the Avalonia viewer (WinForms' virtual keys followed the layout already); since
  DR-47 both report the layout's key, as pygame does, with no auto-repeat (pygame's default). While input is on, the close button
  posts `WINDOWCLOSE` then `QUIT` and keeps the window open, and any resize not made by the viewer's own code (a drag, maximizing, snapping, another program) posts `WINDOWRESIZED`, as SDL reports every size change. Before
  `BeginInput` a viewer behaves as it always has. The
  Avalonia and WinForms viewers implement it (WinForms reports Shift, Ctrl and Alt as the left keys). `play`'s callback
  arguments are borrowed (D19): observations and infos are released after their last callback, so a callback copies
  what it keeps; the first `obs_t` of each episode is the reset's `(obs, info)`, Gymnasium's quirk kept.
  `get_keys_to_action` was a delegate-typed property marked `[PyName]` rather than a method, because a property can be
  absent: MountainCar's raises `AttributeError` in continuous mode, where Gymnasium's continuous class has no such
  method. DR-45 replaced it with a virtual method on `Env`, absent until overridden, with a presence hook. MountainCar's mapping keeps Gymnasium's footguns (pygame 1's arrow codes 275 and 276, which no window
  produces, and the empty combination `()` that makes `play` refuse the mapping). `PlayPlot` draws through
  `IPlotFigure`, matplotlib's calls as `PlayPlot` makes them (compared call for call through a stand-in in the parity
  suite); the default `PlotFigure` is a plain viewer window in matplotlib's layout without text, not matplotlib's
  rendering. The port plan names "a .NET chart library", and that choice is left for the maintainer. `keys_to_action`
  takes any dictionary and compares keys as Python does (`PyKeyComparer`). A display of size 0, which pygame opens
  desktop-sized, is refused. The parity harness compares a NumSharp argument exception by NumPy's text, without the
  ` (Parameter '...')` and `Actual value was` suffixes .NET appends (self-tested).
- **DR-36, `FuncEnv`, the benchmarks and the step-API dispatcher** (found in P6.7, 2026-11-10): `FuncEnv<TState, TObs,
  TAct, TReward, TTerminal, TRenderState, TParams>` (namespace `Gym.Experimental.Functional`, with an untyped `FuncEnv`)
  keeps Gymnasium's functions as virtual methods raising `NotImplementedError`, sets its options by Python name
  (`IDynamicAttributes` holds the undeclared ones) and reads `get_default_params` after them. `transform(func)` rebinds
  Python's attributes, which C# methods cannot be: Gym.NET's stored the transformed functions as run-time attributes of
  the same Python names (`step_info` for `transition_info`, Gymnasium's own naming), so `PyAttributes.GetAttr` saw them
  while direct C# calls ran the originals; no Gymnasium test exercises it. DR-43 replaced this: the functions are
  delegate properties under Gymnasium's names, bound to protected `…Core` methods, and `Transform` reassigns them for
  every caller. `Gym.Utils.Performance` reads its clock through
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
- **DR-29, measured** (P8.1, 2026-11-10): the contact solver's velocity and position loops and the dynamic tree's query
  were written both ways and run on the same scenes, with bit-identical results. On NumSharp `NDArray`s the solver ran
  6.2× slower than on float32 structs and the tree query 1.6× slower, and both took more code (element access and
  scalar extraction around every vector operation). No subsystem passed the gate, so the engine is float32 structs from
  end to end, as Box2D's C++ is; NumSharp supplies Python's exceptions (`AssertionError`, `TypeError`, `ValueError`).
- **DR-26, the pybox2d facade** (P8.1, 2026-11-10): "pybox2d-shaped" is held to pybox2d's behavior, measured against
  the live wheel, where it is surprising:
  - `b2World(doSleep=False)` does not stop bodies sleeping. pybox2d's constructor runs `self.allowSleeping = doSleep`,
    and its world has no such property (only `GetAllowSleeping`/`SetAllowSleeping`), so the value lands in a plain
    attribute. Gym.NET records it as `DoSleep` and leaves the engine alone; `SetAllowSleeping` is public and is the
    switch. No Gymnasium env passes `doSleep`.
  - Keyword constructors wrap a refused argument's error as pybox2d's `_init_kwargs` does, keeping its type:
    `Failed on kwargs for b2BodyDef.angle: in method 'b2BodyDef_angle_set', argument 2 of type 'float32'`; the body's
    fixture shortcuts use pybox2d's other format, `Failed on kwargs, class="b2CircleShape" key="radius": ...`. C# binds
    named arguments to parameters, so when two arguments fail, the one reported first follows the parameter order, not
    the call's.
  - Box2D's `b2Assert` raises `AssertionError` with C++'s stringified condition (`#A`), as pybox2d compiles it; where C#
    spells the condition differently, the call passes C++'s text (44 conditions).
- **DR-32 and DR-28 on LunarLander** (P8.2, 2026-11-11):
  - Gymnasium hangs attributes on its Box2D bodies (`legs[i].ground_contact`, `legs[i].joint`, `particles[i].ttl`, the
    colours). `b2Body` has no attribute bag, so `LunarLanderEnv` keeps them beside the bodies, exposed as
    `LegGroundContact`, `LegJoints` and `ParticleTtl`.
  - Rewards are float64 as everywhere (Gymnasium's are an `np.float64`, and the int -100/+100 on termination). The
    registered threshold `200` prints `200.0` in the spec, as Taxi-v4's `8` does.
  - `heuristic` returns DR-08's 0-d int64 array where Python returns an `int`.
- **DR-32 and DR-28 on BipedalWalker** (P8.3, 2026-11-11):
  - One class, `BipedalWalkerEnv` (`[PyName("BipedalWalker")]`), for both ids; `BipedalWalkerHardcoreEnv` only refuses,
    as Gymnasium's `BipedalWalkerHardcore` does. `ContactDetector` is private and nested (DR-28), and so is the lidar's
    ray-cast callback, which Gymnasium defines inside `reset`.
  - The body attributes live beside the bodies, as for LunarLander: `TerrainColors` for `terrain[i].color1/color2`, and
    `LegGroundContact` with `null` for the upper legs, which have no `ground_contact` attribute in Python.
  - Rewards are float64 values, as everywhere; their precision is the action's, as NumPy 2 computes them (float32 for
    float32 actions). The float64 carries the float32 value exactly.
  - `BipedalWalkerHeuristics.StepHeuristic` returns a new array per call. Python returns the instance's own `a` array,
    which its next call overwrites in place; Gym.NET keeps a copy in `A`. The values are the same.
- **DR-32 and DR-28 on CarRacing** (P8.4, 2026-11-11):
  - `CarRacingEnv` (`[PyName("CarRacing")]`). `FrictionDetector` is public: it is not a twin, which is DR-28's rule for
    keeping a name. `Car` and `Particle` keep `car_dynamics`' names.
  - The attributes Python hangs on bodies live in `RoadTile` (a tile's `color`, `road_visited`, `road_friction`,
    `idx`) and `CarWheel` (a wheel's `gas`, `brake`, `steer`, `phase`, `omega`, skid state, `joint`, `tiles`). Each is
    the body's user data, as Python's `t.userData = t` makes the listener find it.
  - `register_input`, which Gymnasium defines only when the module runs as a script, is
    `CarRacingEnv.RegisterInput(a, ref quit, ref restart)`: Python's module globals become the caller's variables.
  - Rewards are float64 values; Gymnasium's are Python floats, and the int -100 when the car leaves the playfield.
- **DR-27's licence check** (P8.4, 2026-11-11): pygame-ce 2.5.8's default font (`freesansbold.ttf`) is GNU FreeFont's
  FreeSans Bold of 2002, licensed "subject to GNU General Public License" with no font exception. So, as DR-27 provides,
  the recorded data does not ship until the maintainer decides. The layout (`ScoreFont`, an SDL_ttf 2.24 translation)
  and the recorder are in; without data the env leaves the score out, and the parity suite installs data recorded live.
  None of GNU FreeFont's eight archived releases (2003–2012, the later ones with the font exception) renders the same
  glyphs: 0 of 4,633 score strings match pygame's, the vertical metrics and most digit outlines differ. The maintainer
  then decided to ship the recorded data (DR-48, round 15).
- **D4's rasterizer and non-finite coordinates** (P8.2, 2026-11-11): pygame's C code converts a NaN, infinite or
  out-of-range coordinate to `INT_MIN` and draws the visible part. P7's `Draw.Line(double…)` raised `ValueError` there
  instead; it now converts as pygame does, like the new `Polygon`, `Circle` and `Lines`. `Aaline`/`Aalines` (DR-37's
  translation) still refuse non-finite coordinates; pygame draws nothing for them.
- **DR-61 with DR-63 on point.xml** (P9, 2026-11-12): one of the 14 embedded models, point.xml (Gymnasium's custom-env
  test model, used by no id), has a box geom, which the proof of concept's loader refused, so `MjModel.FromXmlPath`
  would have listed a name it cannot load. The engine now takes the box and runs `mjc_PlaneBox`'s test against the floor;
  the test never finds a contact on that model (its box cannot reach the floor), and a box that would touch refuses at
  that step rather than approximate. `oracle.py` records point.xml as `Point-v0`, and the trace checker steps it
  bit-exact, so all 14 names load.
- **DR-11 with C# defaults on the MuJoCo envs** (P9, 2026-11-12): Gymnasium's v5 kwargs default to tuples
  (`healthy_z_range=(0.2, 1.0)`), a camera dictionary, or an `int | str` (`main_body=1`), none of which a C# optional
  parameter can hold. Those parameters are nullable and `null` stands for Gymnasium's default; Python's explicit `None`
  (which Gymnasium fails on) has no counterpart. Reacher-v5's `reward_control_weight`, `null` by default, uses the same
  rule to keep Gymnasium's default an int 1, so an integer action's cost stays an `np.int64` as there.
- **DR-70, the human window, where Gymnasium leaves GLFW to decide** (P9, 2026-11-12): the window draws its frame at
  the viewer's client size in the viewer's own unit (Avalonia's device-independent pixels, WinForms' pixels), so the
  pointer needs no framebuffer scale (GLFW's `_scale` is 1 on Windows too). Its buffer is GLFW's default framebuffer:
  one sample, 24-bit depths and an 8-bit stencil, where the offscreen modes keep 4x multisampling and float depths.
  `swap_interval(1)`'s wait for the display is a 60 Hz `Clock` tick. A missing width or height opens a 960 × 540 window
  (half a 1920 × 1080 monitor; Gymnasium halves the monitor's size, which a viewer factory cannot report). T writes to
  Gymnasium's `/tmp/frame_%07d.png`, which Windows resolves on the current drive, and creates the folder when it is
  missing, where imageio fails. After Esc (still printing `Pressed ESC` and `Quitting.`) or the close button the window
  draws nothing more and a paused `render()` returns, where Gymnasium's viewer keeps calling GLFW on its destroyed window
  (warnings, then MuJoCo drawing without a context) and spins while paused. A visual option outside `mjtVisFlag` fails
  before the window opens (Gymnasium's opens first and leaks it). Keys are the viewer's layout keys (DR-47), where GLFW
  names keys by their US position. The mouse reaches only the MuJoCo window: `PygameDisplay` does not pump the mouse
  kinds into pygame's queue. Kept because the frames show them: the menu's lines pile up over the frames of one
  `render()` (a paused window repeats the menu until `mjr_overlay`'s 500-character cap), the run speed's arithmetic uses
  the model's timestep, not the env's `dt`, the camera starts at MuJoCo's default free camera (not the offscreen modes'
  median look-at point) and ignores `camera_id`, and R changes the env's own model in float32, so the alphas need not
  round-trip. The frames come from the software renderer: a few to a few dozen a second, and a paused window keeps a
  core busy. The parity cells run Gymnasium's own `WindowViewer` (render loop, callbacks, menu, `mjv_moveCamera`) with
  only GLFW, the GL calls, the clock and imageio replaced by scripted stand-ins, since its real window would take the
  screen; MuJoCo's own `mjr_overlay` over Gymnasium's single-sampled frames (byte-exact, pinned to the reference GPU);
  and MuJoCo's own `mjv_updateScene` decorations (float32 bit for bit). They found two defects before the commit:
  `mjv_moveCamera`'s final clamps (elevation within ±89°, the azimuth wrap, the distance range) were missing, and
  `mju_quatZ2Vec`'s cross product with +z has signed zeros of its own, which reach the frames' and arrows' matrices.
- **DR-106, the gallery in the docs layout** (E1, 2026-11-23): the Environments page holds the same gallery inside the
  docs layout's article, with the sidebar on its left and no "In this article" column (`_disableAffix`), so the shelf
  keeps two cartridge columns beside the 440 px panel from about 1260 px wide (one below; the article is about 860 px
  at 1280). Each cartridge is then 165 px wide, and its label's room (139 px) is narrower than the two longest ids at
  one canvas pixel per font pixel: InvertedDoublePendulum-v5 needs 149 px, BipedalWalkerHardcore-v3 143 px. Scaled
  down by `max-width`, their 5x7 letters lost whole columns. Those two labels now wrap onto two lines at the capital
  nearest the middle (INVERTEDDOUBLE / PENDULUM-V5, BIPEDALWALKER / HARDCORE-V3), measured against the cartridge's
  width whenever the shelf resizes; on the landing page's three columns (209 px each at 1440) every id still fits on
  one line. The sidebar lists Overview alone until the family steps add their pages (E3 to E7), so it never links to a
  page that does not exist.
- **DR-102, the game manual in the panel** (E2, 2026-11-23): READ THE MANUAL leads to pages written family by family
  (E3 to E7), so the panel shows it, and Enter's hint, only for a cartridge whose page the Environments sidebar lists
  (`docs/environments/toc.json`, read beside the manifest); until then a cartridge shows its four manual lines and its
  three links alone. The pages take Gymnasium's slugs, `docs/environments/<family>/<page>.html` (`EpisodeRecipe.Page`,
  and `Anchor` for the two variants). Since DR-101 the NOW PLAYING label was 454 px wide for InvertedDoublePendulum-v5
  and 442 px for BipedalWalkerHardcore-v3 in the 440 px panel, scaled down and short of letter columns, on the
  published site too; it now puts the id on a second line where the label does not fit. The blurb left the panel but
  describes the loop to screen readers, and the panel is a live region only once the reader picks a cartridge (attract
  mode would have read a whole manual aloud every 4.5 s). Measured over the 24 envs, the panel was 570 px at most
  before the manual (round 34's blurbs had already outgrown DR-101's 516 px) and is 724 px at most once READ THE MANUAL
  shows (BipedalWalkerHardcore-v3, its label and every row on two lines; 671 px without the button), so the sticky
  floor moved from 660 px to 840 px of window height; between them the panel scrolls with the page. Each manual row
  keeps to two lines, ranges and numbers with their units joined by no-break spaces, and the one row that took three
  (the hardcore walker's YOU SEE) was shortened. The tool's regeneration left every loop, still and logo byte for byte
  as it was; only `envs.json` changed.
- **DR-103 and DR-111…DR-116, the page on the site** (E3, 2026-11-23). CartPole's page follows the approved draft. The
  site builds it as DocFX builds any page, so the draft's hand-made parts took DocFX's own:
  - The C# | Python switches are DocFX's tab groups. Choosing Python in one example chooses it in all.
  - The footgun note is DocFX's `[!WARNING]` alert.
  - The infobox, figure caption and readout use Bootstrap's theme colours, so the page reads in the light theme too.
    The banner and the chart stay dark like the gallery.

  The article is narrower than the draft's (689 px at 1280 beside DocFX's 230 px "In this article" column, against
  792), so the samples keep their lines within about 80 characters, with comments on their own lines.

  The quick start shows its usings through a two-part line range of the samples file, which the facts test pins. The
  samples live in the namespace `Gym.Docs.EnvPageSamples`, where DR-06's `Gym.Tests.Gymnasium` cannot shadow the facade.

  The anatomy chart is drawn by one generic renderer in `main.js`:
  - the page declares the chart's bands (lines over trace columns with ranges and ticks, or a strip of a discrete
    column) and the readout's fields as JSON beside its video;
  - the script draws them from the recorded trace at the chart's own width, so later pages need no script of their
    own.

  The episode, figure and trace come from `examples/CartPole --record` (DR-112). They equal the approved draft's data
  cell for cell and its step-125 frame pixel for pixel; the draft's "±0.62 m" was imprecise (the cart reaches 0.62 m
  right of the centre and 0.47 m left), and the page says so.

  The facts test (`EnvDocPagesTests`) compares a Box's printed bounds with runs of spaces collapsed, as HTML shows
  them: NumPy aligns the bounds in columns. It caught each of three planted errors, one at a time: a step limit, the
  quick start's range and an argument's default. All seven Python twins ran against Gymnasium 1.4.0 (a venv with the
  parity stack's pins).
- **DR-112, DR-114 and DR-124, classic control's other pages** (E4, 2026-11-23). Acrobot, Mountain Car, Mountain
  Car Continuous and Pendulum follow CartPole's template, and each example gained `--record`; MountainCarContinuous is
  the new example DR-124 asked for, the 25th. Every figure equals Gymnasium's own frame of the same moment, 0 differing
  pixels. What the four pages found:
  - Gymnasium's MountainCar renderer scales by `600 / (0.6 - (-1.2))`, and 0.6 + 1.2 is 1.7999999999999998 in float64:
    with `600 / 1.8` instead, `int()` puts the continuous flag in column 549, not 550. The examples' figure code uses
    the renderer's own expression.
  - The facts test read only positive thresholds, so MountainCar's −110 and Acrobot's −100 would have passed unchecked.
    It now reads the typographic minus, and a registered threshold the Reward row leaves out fails.
  - `Gymnasium.Make("MountainCarContinuous-v0")` has no break opportunity before its id's hyphen and pushed the
    half-width infobox's table past the article. A `<wbr>` after `Make(` lets it wrap, and the facts test compares the
    call without it. The longer ids of E6 and E7 will need the same.
  - The 500x500 frames (Pendulum, Acrobot) keep their own size (`.gn-figure-narrow`) instead of the wide frames' 640 px.
  - MountainCarContinuous's controller pushes left from rest, away from the flag: over seeds 0-99 it reaches the flag in
    75 to 86 steps, mean return 92.03, above the registry's 90. Pushing right from rest, as MountainCar's rule does,
    averages 89.37; at half force the rule averages 96.10, since the cost is the force's square.
  - The Arguments sections' make calls are compiled regions on every page, CartPole's included.
  - Gymnasium's `acrobot.py` carries RLPy's notice (Copyright 2013, RLPy, BSD 3-Clause), which `THIRD-PARTY-NOTICES.md`
    does not list beside Gymnasium's MIT notice. The page names it; the notices file is the maintainer's to amend.
- **DR-125, DR-128 and DR-131, toy text's pages** (E5, 2026-11-23). Blackjack, Cliff Walking, Frozen Lake (with its
  8x8 section) and Taxi follow the template, and their five examples gained `--record`. Every figure and board equals
  Gymnasium's own frame of the same moment, 0 differing pixels. What the four pages found:
  - The board band (DR-131) first followed the video's `currentTime`, which runs up to a frame ahead of the picture on
    screen: at toy text's 4 fps that put the ring a whole cell ahead of the sprite. The cursor, the ring and the readout
    now follow `requestVideoFrameCallback`'s presented frame. A headless check matched, in every presented frame, the
    sprite's cell found from its pixels against the trace's cell and the ring's: 0 mismatches on all four boards.
  - The frames are pixel art, and a non-integer scale blurs it, so each page shows its frame at a scale that keeps the
    art pixels whole: Frozen Lake's 256 px at 2x (`.gn-figure-pixel`), Cliff Walking's 768x256 strip up to its own size
    (`.gn-figure-wide`, its anatomy's video above the chart with `.gn-anatomy-wide`), Taxi's 536x456 at its own size
    (`.gn-figure-medium`), Blackjack's 384x376 table, drawn in 4-pixel art pixels, at 1.25x (`.gn-figure-art`).
  - Taxi-v4's registered threshold, 8 ("optimum = 8.46" in Gymnasium's registration), is above what any policy
    averages: over the 300 starting states the reset draws, the shortest deliveries average a return of 7.93
    (793/100), and only 163 starts can return 8 or more. The page says so in its infobox and a warning; the facts test
    still checks the registered 8.
  - DR-128's hand is 7 steps, so 8 frames and 2 s, not the 9 frames and 2.25 s the row first said; the row is corrected.
  - Cliff Walking's Q-learning sample draws its exploration from NumSharp's `np.random.default_rng(0)`, which is
    NumPy's generator bit for bit, so the C# and the Python twin learn the same table: both print −30.19, and the test
    pins the whole run (its 500 returns sum to −27,360). Example 6.6 states epsilon 0.1 and no discount; the step size of
    0.5 is not the example's (the book's 0.5 is the windy gridworld's), and the page says so. SARSA, in the prose, learns
    paths that climb to the top row (17 steps from 76 of 100 seeds), and its online returns (−26.0 against Q-learning's
    −49.9) are the book's figure's.
  - CliffWalkingSlippery-v1's value-iteration plan never falls: it aims so that no slide can reach the cliff (left into
    the edge at the start), and averages 65 steps over 1,000 seeds, where the solid cliff's plan falls 34 times an
    episode on slippery ground.
  - The facts test learned four things: "no step limit" for ids registered without a `TimeLimit` (Cliff Walking's,
    Blackjack's), thresholds stated per id (Frozen Lake's two lakes), defaults written as fractions (`1/3`), and a
    section's own trace (FrozenLake8x8's), whose seed and steps the page must state.
  - Taxi's action mask is the step result's own array, released with it, so the masked random sample draws the next
    action while that step is alive. Blackjack's observation is a tuple of arrays the result owns, which
    `BlackjackObservation` shares: its samples read the numbers out before the result goes. Blackjack's samples file
    enables nullable annotations below the quick start's four lines, for `object?[]`.
  - Gymnasium's Cliff Walking docstring names the info key `p`; the env sends `prob`, as the page says.
- **DR-112 and DR-114, Box2D's pages** (E6, 2026-11-24). Bipedal Walker (with its hardcore section), Car Racing and
  Lunar Lander (with its continuous section) follow the template, and their four examples gained `--record`. Every
  figure equals Gymnasium's own frame after replaying the trace's actions in Gymnasium, one render a step (0 differing
  pixels, and the replay's return equals the trace's bit for bit). What the three pages found:
  - A render changes Box2D's envs: Lunar Lander's ages the exhaust particles (and destroys spent ones), Bipedal Walker's
    moves its lidar sweep on. A figure rendered a second time changes that frame and every later one, so the recorders
    take the figure from `RecordVideo.RecordedFrames`, the frame the video already holds.
  - The legs' (Lunar Lander) and feet's (Bipedal Walker) contact flags are one boolean each, set by `BeginContact` and
    cleared by any `EndContact`, over a ground of separate edges: a leg crossing a joint clears its flag while it
    stands. Replayed with pybox2d's contact lists, Bipedal Walker's leg 2 reads 0 after 209 of the 923 steps its foot
    touches the ground. Both pages warn, and their charts call the strips flags.
  - Gymnasium's walking gait never reads the lidar, and seed 0, the showreel's, is one of its few full courses: over
    seeds 0-99 it reaches the end 4 times and falls 91 times (mean 48.8), and on the hardcore course it gets no further
    than 33 units. The example's README said the gait timed its steps by the lidar; corrected.
  - Bipedal Walker's hardcore section plays the TQC expert's episode with its own figure, a frame with a staircase, a
    stump and a pit, which the figure's script tells apart by their sizes (the env keeps no kinds). The expert hops: in
    486 of its 968 steps neither foot is on the ground.
  - Gymnasium's CarRacing docstring lists the change of 1.0.0, finishing the lap as a termination, under v2; that
    change registered CarRacing-v3. The page's version history follows the registry's history instead (v1 February
    2022, v2 July 2022, v3 Gymnasium 1.0.0).
  - Car Racing's page shows the observation beside the labelled frame: the 96x96 picture at five times its size, in
    `.gn-figure-art`'s 480 px.
  - The anatomy readout showed "−0°" for a level hull: `toFixed` keeps a small negative value's sign. `formatReadout`
    drops it when every digit is a zero.
  - A float32 action makes Gymnasium's reward an `np.float32` (NEP 50), and a Python float plus it stays float32; the
    replay checker sums `float(reward)`, as Gym.NET's double rewards do.
- **DR-112 and DR-114, MuJoCo's pages** (E7, 2026-11-24). All eleven families follow the template, each with its v4
  section, and the eleven examples gained `--record`. Every trace replays in Gymnasium on MuJoCo 4.0.0 to the
  recording's return bit for bit. The figures are Gym.NET's own drawing of MuJoCo's renderer (DR-98), so each page
  states its figure's measured difference from Gymnasium's frame of the same state, from none of 364,800 pixels
  (InvertedDoublePendulum) to 45,452 of 456,000 (Humanoid, nearly all by a level or two), and its badge follows
  MUJOCO_ASSESSMENT 18.3's frame cells. What the eleven pages found:
  - Gymnasium 1.4's default camera does not follow a body: with no camera named the renderer keeps a free camera and
    writes the env's `default_camera_config` into it, `trackbodyid` included, which MuJoCo reads only for a tracking
    camera. Moving Hopper's root 1.5 m changes 9,344 pixels of the default view. Hopper's and Walker2d's pages (Hopper's
    live) had said the camera tracked the thigh; corrected, and Gym.NET's renderer does the same as Gymnasium's.
  - A figure projects the joints through `mjv_updateCamera`'s free-camera geometry; where the model turns a body at the
    reset pose (Humanoid's `lwaist` and `pelvis`, 0.23 degrees each), body-plus-offset is within a millimetre, and the
    checker compares the points with MuJoCo's own `xanchor` (within 0.3 px on every page). Points that are no joint
    (Pusher's fingertip, puck and goal; the rail ends) are body origins plus the recorded offset.
  - Pusher and Reacher take no `reset_noise_scale`, so their figures are the episode's own reset; Pusher's reset gives
    the puck's first draw to the y slide, not x as its docstring's order suggests; Pusher's observed "fingertip" is the
    wrist's centre, 0.1 m behind the gripper's tips, and its goal sits 0.048 m below the puck, so the distance term
    never reaches 0.
  - Reacher's 50-step episode would last one second at its 50 fps: its video is recorded at 25 fps (`RecordVideo`'s
    `fps`), and the trace says 25, keeping the chart's cursor on the picture.
  - HumanoidStandup's `uph_cost_weight` is stored and read nowhere, in Gymnasium as in Gym.NET (2.0 leaves the rewards
    unchanged); InvertedDoublePendulum's model leans gravity 1e-5 m/s^2 toward +x, which topples the noiseless upright
    poles at step 53 (with it set to 0 they stand all 1,000 steps); Humanoid's expert straightens its right knee past
    its -2 degree stop on 169 steps (MuJoCo's soft limits); random torques end Ant's episodes by throwing the torso
    above its healthy band in 9 of seeds 0-9, not by a fall.
  - The parity tables named a "noise" keyword variant the live tests do not run for Hopper, Walker2d, Swimmer and Ant;
    each table now names the variants `MujocoLiveParityTests.KeywordVariants` lists.
  - The software frame times were re-measured (`frametime.cs`, twelve runs with the machine 64-100% busy): HalfCheetah
    about 0.2 s, Hopper 0.5, Walker2d 0.2, Swimmer 0.55, Ant 0.25, Humanoid and HumanoidStandup 0.6, Pusher 0.2,
    Reacher 0.05, the pendulums 0.02 and 0.03. The earlier pages' 0.1 and 0.3 s are corrected, with the "times slower
    than real time" they implied.
  - The generator of the uniform examples' `--record` (`e7/gen-record.py`, scratch) gained options for these cases:
    the figure without the noise argument, the video's frame rate, and the doc patches of the examples that also offer
    `--play`.
