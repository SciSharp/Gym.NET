# C# Code Review

| | |
|---|---|
| **Date** | 2026-11-11, at `b87d845` on `journey1` (P1–P8 done) |
| **Question** | The maintainer's: "is our design C# sound? We are a faithful port of Gymnasium, but we want to be using correct C# and utilize C# lang to the best" |
| **Scope** | `src/`: 61k lines in 8 projects (Gym 31.4k, Gym.Box2D 12.7k, Gym.Environments 10.2k, Gym.Rendering 5.4k, the generator, the two viewers) |
| **Method** | The public contract read in full (`IEnv`, `Env<TObs, TAct>`, wrappers, results, `Info`, spaces, registry, viewers); representative envs, the rasterizer, the Box2D facade and the generator sampled. Measured: build warnings by code, nullability coverage, exception hierarchies (by reflection), reflection and AOT exposure, step throughput and allocation against Gymnasium, CPU profiles (`dotnet-trace`). Two fixes were tried on scratch copies, outside the tree, and the parity suite checked one of them bit for bit (§4) |
| **Status** | Findings only; the changes they suggest are design decisions for the maintainer (DESIGN_REVIEW's gate) |

---

## 1. Verdict

**The design is sound C#.** Gymnasium's shapes became idiomatic .NET types where it mattered:
- **Results:** `StepResult<TObs>` and `ResetResult<TObs>` are `readonly record struct`s that deconstruct like Python's tuples, and allocate nothing.
- **Renders:** `RenderResult` is a closed union callers pattern-match.
- **Wrappers:** their generic parameters are Gymnasium's own type variables.
- **`StepAsync`:** a `ValueTask` that faults its task rather than throwing, with no `Task.Run`.
- **Infos:** `Info` keeps Python's key order through removals, which `Dictionary` does not promise.
- **Typed kwargs:** come from an incremental source generator.
- **Exceptions:** Gym's own carry the standard constructors.
- **Envs:** read as plain, statement-for-statement C#.

**What it does not yet do is use the runtime well where it matters most, and it carries 0.2's leftovers in its public
surface.** Ranked by value:

1. **Performance (measured).** Box2D's vector math is not inlined: LunarLander and BipedalWalker run 3× slower than
   Gymnasium (Python driving C++ Box2D). An attribute on 53 small members makes them 5.9× and 6.4× faster, 2.2× and 2.6×
   faster than Gymnasium, with all 176 Box2D parity cells passing unchanged. CarRacing spends 68% of a step in
   `smoothscale`, which runs 3× faster with its four channels unrolled (identical output), and it allocates 7.1 MB a step.
2. **Public surface for 2.0.** 18 unused public types in `Gym.Internal` and 3 unused 0.2 exceptions (69 of the core
   build's 71 warnings), a delegate named like an interface (`IEnvironmentViewerFactoryDelegate`), an `async` factory
   every caller blocks on, and ImageSharp's `Image` in the core viewer contract.
3. **.NET contracts against Python semantics.** Value equality with `==` on mutable types (`Info`, `Metadata`, the
   mutable record `EnvSpec`), a process-wide registry namespace, static hiding for class metadata.
4. **Reach.** Nullability is off project-wide in the two largest projects (29 files, `Space`/`Box`/`Discrete` among
   them). Nothing is annotated for trimming or AOT, and NumSharp's IL kernels crash where dynamic code is off, which
   .NET 10's file-based apps (`dotnet run app.cs`) turn on by default.

## 2. What is idiomatic and should stay

| Design | Why it is right in C# |
|---|---|
| `readonly record struct StepResult<TObs>` / `ResetResult<TObs>` with positional deconstruction | Value semantics, no allocation per step, `var (obs, reward, terminated, truncated, info) = env.Step(a)` reads as Gymnasium's tuple |
| `RenderResult`, a closed hierarchy (private constructor, nested sealed cases) | C#'s discriminated union: exhaustive `switch` over `Rgb`, `Ansi`, `Frames`, `None`; disposable because frames own native memory |
| `Env<TObs, TAct>` + `IEnv`, `Wrapper<TWObs, TWAct, TObs, TAct>`, `ObservationWrapper`/`ActionWrapper`/`RewardWrapper` | Gymnasium's own type variables as generic parameters; the untyped `IEnv` for heterogeneous code |
| `ValueTask<StepResult<TObs>> StepAsync`, completing synchronously | No thread-pool hop, no allocation; a failure faults the task instead of throwing from the call (the async contract) |
| `Info`, an ordered dictionary (list + position index) | Python's insertion order survives `Remove`, which .NET's `Dictionary` does not guarantee |
| The incremental generator (`IIncrementalGenerator`, `ForAttributeWithMetadataName`) emitting `sealed partial record` options | Typed, discoverable kwargs (`new CartPoleEnvOptions { RenderMode = "rgb_array" }`) at no run-time cost |
| Gym's exceptions (`Error`, `ResetNeeded`, `UnregisteredEnv`, ...) with the three standard constructors; `NotImplementedError : NotImplementedException` | .NET's exception conventions, Gymnasium's names |
| `IDisposable` envs and spaces with the standard dispose pattern; the registry's namespace as a `using` scope | Deterministic release of native memory and windows; Python's `with` maps to `using` |
| `EnvSpecRegistry`: an ordered list under a lock, entry points cached in a `ConcurrentDictionary` | Safe registration from module initializers on any thread |
| Envs written as plain C# over `double`s (`CartPoleEnv.Step`), with the parity rules commented where they bite | Readable without knowing Gymnasium; the float64/float32 decisions visible |
| Pattern matching throughout (370 `switch` expressions and property patterns) | Current C#, used naturally |

## 3. Findings

`Breaking` means a public API change; 2.0 is a breaking release, so now is when such changes are cheapest.

### 3.1 Performance (measured, §4)

| # | Finding | Evidence | Recommendation | Breaking |
|---|---|---|---|---|
| R1 | Box2D's vector, matrix and rotation operators and `b2Math` helpers are not inlined in the big solver methods (the JIT's budget runs out there), so every `a - b` is a call | LunarLander's profile: `b2Vec2.op_Subtraction`, `op_Addition`, `op_Multiply`, `..ctor`, `b2Cross`, `b2Mat33.Solve33` each a frame; 52% of the step in `b2RevoluteJoint.SolveVelocityConstraints` (180 velocity iterations a step, as Gymnasium's `world.Step(1/50, 6*30, 2*30)` asks) | `[MethodImpl(MethodImplOptions.AggressiveInlining)]` on the 53 small math members. RyuJIT never fuses or reorders float operations, so inlining keeps every rounding: measured 5.9× (LunarLander) and 6.4× (BipedalWalker), 176 of 176 Box2D parity cells passing | No |
| R2 | `Transform.Smoothscale`'s shrink keeps its four channel sums in a `stackalloc`'d `Span<long>`, indexed per channel per pixel | CarRacing: 68% of the step in `ShrinkLine`; 13.2 ms per 1000×800 → 96×96 | Four locals, the channels unrolled: 4.4 ms (3.0×), identical on 200 random frames | No |
| R3 | CarRacing allocates 7.1 MB a step: a new 1000×800 surface, a new flipped copy, the shrink's temporary, the frame's byte array | `GC.GetAllocatedBytesForCurrentThread`; both 3.2 MB buffers land on the Large Object Heap, which only gen-2 collections reclaim | Reuse env-owned buffers across steps (a cached surface, an in-place flip into a second cached one, `ArrayPool` for the temporary) | No |
| R4 | The small envs are already faster than Gymnasium (1.2× CartPole to 11.9× Acrobot) and allocate 3–11 KB a step, mostly NumSharp temporaries | §4 | Nothing now; the arrays per step are the price of `NDArray` observations | — |

### 3.2 The public surface for 2.0

| # | Finding | Evidence | Recommendation | Breaking |
|---|---|---|---|---|
| R5 | `Gym.Internal`: 18 public types from 0.2 (`Dict`, `Expando`, `Merger`, `ReflectionHelper`, `ExpandableList`, `CircularQueue`, `DefaultConstructor`, `GenericCallFactory`, ...) used by nothing | No reference outside the folder, in `src`, `tests` or `examples`; 62 of the core build's 71 warnings are theirs (43 missing docs, 18 malformed docs, 1 unused field) | Delete them | Yes, to code no one should call |
| R6 | Three 0.2 exceptions, `InvalidActionError`, `AlreadySteppingError`, `NotSteppingError`, thrown by nothing | Referenced only by their own files (Gymnasium's `InvalidAction` replaced the first); 7 more of the warnings. The last 2 are ambiguous `cref`s (`Legacy/VecEnvWrapper.cs`, `Spaces/Discrete.cs`), one-line fixes | Delete them; fix the two `cref`s | Yes |
| R7 | `IEnvironmentViewerFactoryDelegate` is a delegate with an interface's `I` prefix (.NET naming rules), its `string title = null` unannotated | `src/Gym/Envs/Viewers/IEnvViewer.cs` | Rename (`EnvViewerFactory`) | Yes |
| R8 | The viewer factory returns `Task<IEnvViewer>`, but rendering is synchronous and all six call sites block on it (`.GetAwaiter().GetResult()`), a deadlock risk wherever a factory resumes on a UI synchronization context | `CartPoleEnv`, `HumanWindow`, `HumanRendering` (both), `PygameDisplay`, `PlayPlot`; the plot figures block the same way | A synchronous factory (`IEnvViewer Create(int width, int height, string? title)`); the viewer packages keep their UI-thread startup inside | Yes |
| R9 | `IEnvViewer.Render(Image img)` puts ImageSharp's type in the core contract | Every viewer and the core depend on it for that one parameter | Pass the frame (`NDArray` of `(H, W, 3)` uint8, or a `ReadOnlySpan<byte>` with its size); ImageSharp stays where it decodes and encodes (D18 unchanged) | Yes |
| R10 | `StepResult`/`ResetResult` are not `IDisposable`, so the ownership rule ("the arrays an env returns belong to the caller") needs `step.Observation.Dispose()` by hand | `docs/guides/getting_started.md` | Implement `IDisposable` on both (disposing the observation), so `using var step = env.Step(a);` works | No (additive) |
| R11 | Render modes are magic strings: 172 `"human"`/`"rgb_array"` literals, no constants | `grep` | `RenderModes.Human`, `.RgbArray`, ... constants, the strings unchanged | No (additive) |

### 3.3 .NET contracts against Python semantics

| # | Finding | Evidence | Recommendation | Breaking |
|---|---|---|---|---|
| R12 | `Info` (and `Metadata`, which derives from it) overloads `==` and `Equals` by value on a mutable type, with `GetHashCode() => Count`; `info.Equals(dictionary)` can be true while `dictionary.Equals(info)` is false | .NET's guidelines: no value `==` on mutable reference types; `Equals` must be symmetric. A mutated `Info` inside a `HashSet` is lost | Reference equality for the type, Python's `==` as `PyEquality.AreEqual` (already there) or an `IEqualityComparer<Info>` | Yes |
| R13 | `EnvSpec` is a `sealed record` with public setters: value equality and a hash over mutable state | Python's non-frozen dataclass is unhashable; a C# record is hashed | `init` accessors, derived specs through `with` | Yes |
| R14 | `EnvRegistry.CurrentNamespace` is process-wide: two threads registering under different namespaces interfere, the lock notwithstanding | `EnvRegistry.Namespace(ns)` swaps one static | Keep the `using` scope, store the namespace in an `AsyncLocal<string?>` | No |
| R15 | Class-level metadata is a `public new static Metadata ClassMetadata` that each env hides | `Env.cs` | C# 11's static abstract interface member, or leave it (reflection finds it either way) | Minor |

### 3.4 Nullability, analysis and reach

| # | Finding | Evidence | Recommendation | Breaking |
|---|---|---|---|---|
| R16 | `<Nullable>` is not enabled in Gym and Gym.Environments; 29 files have no nullability context, the space types (`Space.cs`, `Box.cs`, `Discrete.cs`), `Seeding.cs`, the viewers and the legacy among them; Gym.Rendering and Gym.Box2D enable it and treat warnings as errors | Per-file `#nullable enable` count | Enable it project-wide, annotate the 29 files, and treat warnings as errors in all four projects | No |
| R17 | Nothing is annotated for trimming or NativeAOT, though `make`, kwargs binding and the attribute protocol use reflection (`GetConstructors`, `MakeGenericType`, `Assembly.Load`); and NumSharp emits IL kernels (`DynamicMethod`), which throw `PlatformNotSupportedException` where dynamic code is off | Measured: a .NET 10 file-based app (`dotnet run app.cs`) defaults to AOT settings, and constructing `CartPoleEnv` throws there; `#:property PublishAot=false` is the workaround | Document it now (getting started); mark the reflection entry points `[RequiresUnreferencedCode]`/`[RequiresDynamicCode]` and the `Type` parameters `[DynamicallyAccessedMembers]`; the generated options records already offer the reflection-free path; NumSharp needs a non-emitting fallback (upstream) | No |
| R18 | `Gym.Box2D` ships inside Gym.NET.Environments as a `PrivateAssets` project reference, yet the envs' public API returns its types (`LunarLanderEnv.World` is a `b2World`) | A project-reference consumer cannot compile `env.World` (CS0012); the NuGet package works, since it packs the dll beside the envs' | Fine for package consumers; note it for project-reference users | — |

### 3.5 Naming and typing

| # | Finding | Recommendation |
|---|---|---|
| R19 | Box2D's public types keep C++'s and pybox2d's names (`b2World`, `b2Vec2`), lowercase against .NET's PascalCase rule; the engine's structs expose camelCase fields (`separation`, `proxyIdA`) while the facade uses PascalCase properties | A decision: keep the names everyone who knows Box2D recognizes (DR-26), or `B2World`/`B2Vec2` (PascalCase, prefix kept) with `[PyName]` for parity reports |
| R20 | Every env is `Env<NDArray, NDArray>` (DR-08: Discrete samples are 0-d arrays), so the type parameters carry little; NumSharp's implicit conversions keep `env.Step(1)` compiling | Decided (DR-08); typed convenience is possible later without changing the contract |
| R21 | `Space.Sample(object? mask, object? probability)` and `Seed(...)` returning `object` keep Python's dynamic signatures | Typed overloads per space (`Discrete.Sample(NDArray? mask)`), the untyped entry kept for generic code |
| R22 | Python-named members (`Pprint`, `PprintRegistry`, `GetWrapperAttr`) | Keep: a consistent, documented mapping (port plan §3.8) |
| R23 | 154 block-scoped namespaces, 1 file-scoped | Style only: file-scoped namespaces would remove a level of indentation everywhere; mechanical, low value |

### 3.6 Upstream: NumSharp (same maintainer)

| # | Finding | Recommendation |
|---|---|---|
| U1 | `ValueError : ArgumentException`, `AxisError : ArgumentOutOfRangeException` and `OSError : IOException` are idiomatic, but `TypeError`, `KeyError`, `IndexError`, `AttributeError` and `AssertionError` derive only from `NumSharpException`: `catch (KeyNotFoundException)` or `catch (ArgumentException)` misses them | Derive them from `ArgumentException`, `KeyNotFoundException`, `IndexOutOfRangeException` (or `ArgumentOutOfRangeException`), `MissingMemberException` and `InvalidOperationException` |
| U2 | The IL kernel generator has no fallback when `RuntimeFeature.IsDynamicCodeSupported` is false | An interpreted or generic-math path, so NumSharp (and Gym.NET) run under NativeAOT and in file-based apps |

### 3.7 Stale documentation found

- `Env<TObs, TAct>`'s remarks still say spaces are "the non-generic `Space` until the spaces work (P2) makes them
  `Space<T>`"; they are `Space<TAct>`/`Space<TObs>`.
- Its `TObs` doc says Discrete observations are `long`; since DR-08 they are 0-d `NDArray`s.

## 4. Measurements

**Throughput** (Release, net10.0, sampled actions, observations disposed; Gymnasium v1.4.0 on Python 3.12 the same
way, on the same machine):

| Env | Gym.NET steps/s | Gymnasium steps/s | Ratio | Gym.NET bytes/step |
|---|---:|---:|---:|---:|
| CartPole | 116,148 | 95,558 | 1.2× faster | 3,269 |
| MountainCar | 173,398 | 63,740 | 2.7× faster | 3,192 |
| Pendulum | 64,629 | 20,083 | 3.2× faster | 11,088 |
| Acrobot | 227,650 | 19,096 | 11.9× faster | 4,019 |
| FrozenLake | 272,377 | 96,050 | 2.8× faster | 4,283 |
| Taxi | 187,941 | 75,246 | 2.5× faster | 4,681 |
| LunarLander | 7,792 | 23,936 | 3.1× slower | 9,038 |
| BipedalWalker | 1,719 | 4,686 | 2.7× slower | 35,199 |
| CarRacing | 50 | 104 | 2.1× slower | 7,069,106 |

**R1's experiment.** A copy of `src/Gym.Box2D` with `[MethodImpl(MethodImplOptions.AggressiveInlining)]` on 53 members of
`Common/b2Math.cs` (every operator of `b2Vec2`, `b2Vec3`, `b2Mat22`, `b2Mat33`, `b2Rot`, every static `b2*` helper, the
one-line readonly members) was swapped into a copy of the app (the change is metadata only, so the binary interface is
the same). Steps per second, two interleaved rounds each:

| Env | Original | Inlined | Gain |
|---|---:|---:|---:|
| LunarLander | 8,780 | 52,130 | 5.9× |
| BipedalWalker | 1,924 | 12,232 | 6.4× |
| CarRacing | 65 | 62 | none (rendering-bound) |

The same copy, swapped into a copy of the parity suite's Release output: 176 of 176 cells pass. That covers
`Box2DEngineLiveParityTests` (scenes, events, ray casts, errors against live pybox2d, float32 bits), the LunarLander,
BipedalWalker and CarRacing suites, every golden replay and the bridge's Box2D rows.

**R2's experiment.** `ShrinkLine` as it is, against the same arithmetic with the four channel sums in locals: identical
output on 200 random frames of random sizes, and 13.2 ms against 4.4 ms for CarRacing's 1000×800 → 96×96 (100 runs
after 20 warm-up runs, twice).

**Profiles** (`dotnet-trace`, sampled thread time, 6 s of sampled actions; inclusive time on the stepping thread):

| LunarLander | | CarRacing | |
|---|---:|---|---:|
| `LunarLanderEnv.Step` | 69% | `CarRacingEnv.RenderFrame` | 91% |
| `b2World.Step` | 66% | `Transform.Smoothscale` / `ShrinkLine` | 68% |
| `b2RevoluteJoint.SolveVelocityConstraints` | 52% | `CarRacingEnv.RenderRoad` | 16% |
| `b2Mat33.Solve33` | 14.5% | `GfxDraw.FilledPolygon` | 11% |
| `b2Vec2` operators and constructor, `b2Cross` | 32% together | `Transform.Flip` | 4% |
| NumSharp (the observation, the samples) | ~8% | `b2World.Step` | 1.4% |

## 5. Suggested order

1. R1, R2, R3: no API change, measured, verified by the existing gates; R1 is a one-file change.
2. R5, R6, R7, R8, R9, R10, R11: the 2.0 surface, while breaking is free; R5 and R6, with the two `cref` fixes, clear
   every build warning.
3. R16: nullability project-wide, then warnings as errors everywhere.
4. R12, R13, R14: the equality and scoping semantics, each a short design decision.
5. R17 and U1/U2: documentation now, annotations and the NumSharp items after.
6. R19, R21, R23: naming and typing decisions with no urgency.
