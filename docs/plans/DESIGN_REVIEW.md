# Gym.NET Design Review

| | |
|---|---|
| **Status** | Reviewed on 2026-11-08 in ten rounds, on branch `journey1`: 36 designs, all decided. Four more rounds on 2026-11-10 settled the questions P6 and the per-env parity coverage raised: DR-37…DR-47. Round 15 on 2026-11-11, after P8, decided CarRacing's score glyphs (DR-48) and the packages' licence (DR-49), and the maintainer confirmed the ten [interpretations](#interpretations-to-confirm). Round 16, the same day, took the C# code review's first findings (DR-50…DR-54; DR-50's R1 was withdrawn before commit, its gain having been measured on Debug builds), and round 17 closed AOT (DR-55). Round 18 settled the 0.2 leftovers (DR-56) and the registry's namespace (DR-57), and round 19 the obsolete shims that keep 0.2 code compiling (DR-58). Rounds 20 to 25 (2026-11-11 and 2026-11-12) settled the showreel and MuJoCo (DR-59…DR-74). Rounds 26 to 29 on 2026-11-21 are the 1.0 architecture review and the questions implementing it raised (DR-75…DR-97); they also decided the code review's R7–R9 (DR-85), R12 (DR-92), R13 (DR-75) and R21 (DR-79). Pending: `DictInfoToList`'s result shape (deferred, with a direction), and the code review's R15, R19 and R23. [Coverage by phase](#coverage-by-phase) says which phases can start |
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
| DR-60 | The docs site's main page | **A DocFX site like NumSharp's, its main page the video with the 8-bit cartridge select below it** ("This should be part of the docfx main page", "And below it 8-bit cartridge"). `docs/docfx.json` builds the `docs/` pages, the API reference of Gym, Gym.Environments, Gym.Box2D, Gym.Rendering and Gym.Rendering.Avalonia, and the landing page in `docs/website-src`. The cartridge select is the site's environment gallery: 13 looping WebPs at a fifth of the frame size and `envs.json` from `tools/showreel`, driven by `templates/gymnet/public/main.js` (attract mode, arrow keys, the video's 5x7 font). `.github/workflows/docs.yml` deploys it to GitHub Pages from master. *Read as* (to confirm): no Gym.NET site existed, so this round creates one; `docfx.json` sits in `docs/`, not NumSharp's `website-src`, because DocFX resolves links by source path and the docs pages must be inside its folder; WinForms is left out of the API reference (Ubuntu cannot build `net*-windows`); links to files the site does not publish (plans, notices, CLAUDE.md) go to GitHub; Pages has to be enabled once in the repository settings | The README only | `docs/`, `.github/workflows/docs.yml` |

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
| P9 MuJoCo | DR-30, DR-31, DR-33; postponed by DR-35 until 2026-11-12, then DR-61…DR-68 (the engine, the env classes, the parity stage) and DR-69…DR-71 (the renderer) | Yes (resumed by DR-61) |
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
