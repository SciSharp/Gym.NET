# C# Code Review

| | |
|---|---|
| **Date** | 2026-11-11, at `b87d845` on `journey1` (P1–P8 done) |
| **Question** | The maintainer's: "is our design C# sound? We are a faithful port of Gymnasium, but we want to be using correct C# and utilize C# lang to the best" |
| **Scope** | `src/`: 61k lines in 8 projects (Gym 31.4k, Gym.Box2D 12.7k, Gym.Environments 10.2k, Gym.Rendering 5.4k, the generator, the two viewers) |
| **Method** | The public contract read in full (`IEnv`, `Env<TObs, TAct>`, wrappers, results, `Info`, spaces, registry, viewers); representative envs, the rasterizer, the Box2D facade and the generator sampled. Measured: build warnings by code, nullability coverage, exception hierarchies (by reflection), reflection and AOT exposure, step throughput and allocation against Gymnasium, CPU profiles (`dotnet-trace`). Two fixes were tried on scratch copies, outside the tree, and the parity suite checked one of them bit for bit (§4) |
| **Status** | Findings; the changes they suggest are design decisions for the maintainer (DESIGN_REVIEW's gate). Decided so far: R2, R3, R10, R11 and R16 implemented (DR-50…DR-54), R8 and R9 kept with TODOs (DR-53), R7 not taken, R17 closed (DR-55), R5 and R6 (DR-56: the six unused types removed, the fifteen `Dict` types `[Obsolete]` until 3.0) and R14 (DR-57: the namespace scope per thread and async flow) implemented |
| **Second edition** | The same day. The first edition's throughput and profiles were measured on Debug builds of the libraries (§4 says how), and R1 rested on them: measured again in Release, R1 gains nothing and is withdrawn, and Gym.NET is faster than Gymnasium on every env measured. §3.7's leftovers were checked again, extension-method calls included, and each now says what removing it costs 0.2 code and what replaces it |

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

1. **Performance (measured in Release, §4).** Gym.NET steps faster than Gymnasium on all nine envs measured: 1.4×
   (CartPole) to 14× (Acrobot), LunarLander 1.6×, BipedalWalker 3.6×. The one hot spot was CarRacing, which spent most
   of a step in `smoothscale` and allocated 7.1 MB a step; unrolling the shrink's four channels (identical output) and
   reusing its buffers (R2, R3) make it 1.6× faster at 3.6 MB a step, 2.2× Gymnasium's speed. *Corrected:* the first
   edition found Box2D 3× slower than Gymnasium and an inlining fix 6× faster (R1); both came from Debug builds of the
   libraries, and R1 is withdrawn.
2. **Public surface for 2.0.** 21 public types left from 0.2, six used by nothing and fifteen reachable only through
   the obsolete legacy API's `Dict` (69 of the core build's 71 warnings), a delegate named like an interface
   (`IEnvironmentViewerFactoryDelegate`), an `async` factory every caller blocks on, and ImageSharp's `Image` in the
   core viewer contract.
3. **.NET contracts against Python semantics.** Value equality with `==` on mutable types (`Info`, `Metadata`, the
   mutable record `EnvSpec`), a process-wide registry namespace, static hiding for class metadata.
4. **Reach.** Nullability is off project-wide in the two largest projects (26 files without it, the viewers,
   `Seeding`, `Gym.Legacy` and `Gym.Internal`). Nothing is annotated for trimming or AOT, and NumSharp's IL kernels crash
   where dynamic code is off, which .NET 10's file-based apps (`dotnet run app.cs`) turn on by default.

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
| R1 | *Withdrawn.* The first edition read Box2D's small math members (`b2Vec2`'s operators, `b2Cross`, ...) as not inlined in the solver's big methods, every `a - b` a call | Its profile showed each operator as a frame, but it was a Debug build, where the JIT inlines nothing. In Release, `[MethodImpl(MethodImplOptions.AggressiveInlining)]` on the 53 members measures 1.05× (LunarLander) and 1.12× (BipedalWalker), inside the rounds' spread (§4) | None. The change passed every gate (it cannot change a float: RyuJIT never fuses or reorders float operations) and was reverted before it was committed | — |
| R2 | `Transform.Smoothscale`'s shrink keeps its four channel sums in a `stackalloc`'d `Span<long>`, indexed per channel per pixel | CarRacing's profile; 13.2 ms per 1000×800 → 96×96 (§4) | Four locals, the channels unrolled: 4.4 ms (3.0×), identical on 200 random frames. *Done (DR-50)*: with R3, CarRacing steps 1.6× as fast; `ShrinkLine` is still 36% of its step | No |
| R3 | CarRacing allocates 7.1 MB a step: a new 1000×800 surface, a new flipped copy, the shrink's temporary, the frame's byte array | `GC.GetAllocatedBytesForCurrentThread`; both 3.2 MB buffers land on the Large Object Heap, which only gen-2 collections reclaim | Reuse buffers across steps. *Done (DR-50)*: the pre-flip canvas is the env's, the shrink's temporary comes from `ArrayPool`; the flipped surface stays a new object per render, as Gymnasium's `self.surf` is: 3.6 MB a step | No |
| R4 | The small envs are faster than Gymnasium (1.4× CartPole to 14× Acrobot) and allocate 3–11 KB a step, mostly NumSharp temporaries | §4 | Nothing now; the arrays per step are the price of `NDArray` observations | — |

### 3.2 The public surface for 2.0

| # | Finding | Evidence | Recommendation | Breaking |
|---|---|---|---|---|
| R5 | `Gym.Internal`: 18 public types from 0.2 (`Dict`, `Expando`, `Merger`, `ReflectionHelper`, `ExpandableList`, `CircularQueue`, `DefaultConstructor`, `GenericCallFactory`, ...); 62 of the core build's 71 warnings are theirs (43 missing docs, 18 malformed docs, 1 unused field) | *Corrected twice after the maintainer's double-check request (§3.7):* three are used by nothing; the other fifteen are reachable through `Gym.Collections.Dict`, the info type of the `[Obsolete]` `Gym.Legacy` API (`Step.Information`, `VecEnv.Metadata`). The first search looked for `Gym.Internal` and the type names, not for `Dict` in namespace `Gym.Collections`; the second missed that `DefaultConstructor` calls `ReflectionHelper`'s extension methods, which never name their class | See §3.7: what each way of removing them costs 0.2 code, and what replaces each. *Decided (DR-56)*: the six unused removed, the fifteen `[Obsolete]` until 3.0, documented; the core build's warnings went from 71 to 1 | Yes |
| R6 | Three 0.2 exceptions, `InvalidActionError`, `AlreadySteppingError`, `NotSteppingError`, thrown by nothing | Referenced only by their own files (Gymnasium's `InvalidAction` replaced the first); 7 more of the warnings. The last 2 are ambiguous `cref`s (`Legacy/VecEnvWrapper.cs`, `Spaces/Discrete.cs`), one-line fixes | See §3.7; fix the two `cref`s. *Decided (DR-56)*: all three removed, the two `cref`s fixed | Yes |
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
| R14 | `EnvRegistry.CurrentNamespace` is process-wide: two threads registering under different namespaces interfere, the lock notwithstanding | `EnvRegistry.Namespace(ns)` swaps one static. *What the namespace is:* an env id's optional prefix, `[namespace/](name)[-v(version)]`, as in `ALE/Breakout-v5` or Gymnasium's own `phys2d/CartPole-v1`; `make`'s errors (`NamespaceNotFound`) and `pprint_registry`'s groups use it. Gymnasium's `namespace(ns)` context manager sets the module global `current_namespace`, and while it is set `register` puts every id in that namespace, replacing one written in the id. Gymnasium 1.x calls it nowhere itself (its plugin loader, which did, went in 1.0); one test does. Gym.NET registers no namespaced ids | Keep the `using` scope, store the namespace in an `AsyncLocal<string?>`: each thread and async flow sees its own scope, single-threaded code behaves as now. *Done (DR-57)*, pinned by four tests that a process-wide static fails | No |
| R15 | Class-level metadata is a `public new static Metadata ClassMetadata` that each env hides | `Env.cs` | C# 11's static abstract interface member, or leave it (reflection finds it either way) | Minor |

### 3.4 Nullability, analysis and reach

| # | Finding | Evidence | Recommendation | Breaking |
|---|---|---|---|---|
| R16 | `<Nullable>` is not enabled in Gym and Gym.Environments; 26 files of Gym have no nullability context: the viewers, `Seeding.cs`, the four 0.2-era exception files, `Gym.Legacy` and `Gym.Internal` (the space types do have it; a first count missed files whose directive follows a byte-order mark). Gym.Rendering and Gym.Box2D enable it and treat warnings as errors | Per-file `#nullable enable` count | Enable it project-wide, annotate the 26 files, and treat warnings as errors in all four projects | No |
| R17 | Nothing is annotated for trimming or NativeAOT, though `make`, kwargs binding and the attribute protocol use reflection (`GetConstructors`, `MakeGenericType`, `Assembly.Load`); and NumSharp emits IL kernels (`DynamicMethod`), which throw `PlatformNotSupportedException` where dynamic code is off | Measured: a .NET 10 file-based app (`dotnet run app.cs`) defaults to AOT settings, and constructing `CartPoleEnv` throws there; `#:property PublishAot=false` is the workaround | Document it now (getting started); mark the reflection entry points `[RequiresUnreferencedCode]`/`[RequiresDynamicCode]` and the `Type` parameters `[DynamicallyAccessedMembers]`; the generated options records already offer the reflection-free path; NumSharp needs a non-emitting fallback (upstream). *Closed (DR-55)*, the maintainer: "We can't AOT, NumSharp can't." No annotations; the guide and the release notes state the limitation | No |
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

### 3.7 The 0.2 leftovers, one by one (the maintainer's double-check)

Every public type of `src/Gym/Internal` and the three 0.2 exceptions, searched by name in every C# file of `src`,
`tests` and `examples` (code apart from comments), each namespace import traced, the calls of their extension methods
followed (they never name their class), and what live code reaches closed over. `Gym.Legacy` counts as live: it ships
in 2.0, `[Obsolete]`, and goes in 3.0. "0.2.0" is the last release, commit `666746d` (2023-03-10): all 21 types were
public in it.

*Corrected twice.* The first edition called 18 + 3 of them unused; the first double-check called seven unused, missing
that `DefaultConstructor` calls two of `ReflectionHelper`'s extension methods, and described `CircularQueue<T>` as a
ring buffer.

**Used by nothing (6).** No 2.0 code reaches them, and none was part of 0.2's env API.

| Type | What it is | In 0.2.0 | Removed, what stops compiling | What replaces it |
|---|---|---|---|---|
| `CircularQueue<T>` (`Gym.Collections`) | A round-robin queue: `Dequeue` returns the next item and puts it back at the end, `CircleCompleted` fires after each full circle (from QuantConnect's Lean) | Unused | Code that names it | A `Queue<T>` (`queue.Enqueue(queue.Dequeue())`), or an index modulo the count (Python's `itertools.cycle`) |
| `CircularQueueThreadSafe<T>` (same file) | The same under a lock | Unused | Code that names it | The same under a `lock`, or `ConcurrentQueue<T>` |
| `GenericCallFactory` (`Gym.Reflection`) | Caches `MethodInfo.MakeGenericMethod` per type and invokes the result | Unused since the 2019 initial commit | Code that names it | `method.MakeGenericMethod(type).Invoke(target, args)` |
| `InvalidActionError` (`Gym.Exceptions`) | 0.2's invalid-action error | Thrown by 0.2's Aether `LunarLanderEnv` for an action outside its space | `catch (InvalidActionError)` and `throw new InvalidActionError(...)`. Kept, such a `catch` compiles and never fires: nothing in 2.0 throws it | `AssertionError` from the envs that assert their action, as Gymnasium's do (`assert self.action_space.contains(action)`: LunarLander, CartPole, ...); `InvalidAction` from CarRacing, and for custom envs (Gymnasium's `error.InvalidAction`) |
| `AlreadySteppingError` (`Gym.Exceptions`) | "already running an async step" | Thrown by nothing | Code that names it | `AlreadyPendingCallError`: `AsyncVectorEnv`, a call while another is pending |
| `NotSteppingError` (`Gym.Exceptions`) | "not running an async step" | Thrown by nothing | Code that names it | `NoAsyncCallError`: `AsyncVectorEnv`, a `*_wait` without its `*_async` |

**Reachable only through the legacy API's info type (15).** `Gym.Collections.Dict` was 0.2's info and metadata type
(`Env.Metadata`, `Step.Information`, `IVecEnv.Metadata`, `GoalEnv.ComputeReward`). In 2.0 only `Gym.Legacy` uses it:
`Step.Information`, `ILegacyEnv.Metadata`, `IVecEnv`/`VecEnv.Metadata`, filled by `LegacyEnvAdapter`. The other
fourteen are its machinery.

| Type | What it is | Reached through | What replaces it |
|---|---|---|---|
| `Dict` (`Gym.Collections`) | `Dictionary<string, object>` with inline pairs (`new Dict("k", v, ...)`), `Get<T>`, `AsDynamic()`, merges, `Clone`, `RemoveWhere`/`SetWhere` | `Gym.Legacy` (5 files) | `Gym.Envs.Info` for infos (Gymnasium's dict: insertion order, `Get(key, default)`, `Update`, `Copy`/`DeepCopy`, `Pop`) and `Gym.Envs.Metadata` for metadata, the types 2.0's `Env` uses |
| `Expando` (`Gym.Dynamic`) | A dynamic object over a dictionary (Rick Strahl's West Wind Expando) | `Dict.AsDynamic()` | `System.Dynamic.ExpandoObject` |
| `Merger`, `IMergable`, `IMergable<T>`, `MergePrefer` (`Gym.Merging`) | A deep merge of dictionaries and lists, the old or the new value winning | `Dict`'s `Merge`, `Append`, `Override`, `Update` | `Info.Update` (Python's `dict.update`: the new value wins, one level deep) |
| `ExpandableList`, `ExpandableList<T>`, `IExpandableList` (`Gym.Collections`) | One value that a merge expands into a list | `Merger` | `List<T>` |
| `DefaultConstructor`, `DefaultConstructor<T>`, `DefaultConstructorNotFoundException` (`Gym.Reflection`) | Cached delegates to parameterless constructors | `Merger` | `Activator.CreateInstance<T>()`, or a `new()` constraint |
| `ReflectionHelper` (`Gym.Reflection`) | Reflection extension methods (`GetDefaultConstructor`, `CreateDefaultConstructor`, `HasInterface`, tuple helpers, ...) | `DefaultConstructor` calls two of them | `Type.GetConstructor(Type.EmptyTypes)`, `Type.IsAssignableTo`, `System.Runtime.CompilerServices.ITuple` |
| `DictInlineInitializationException`, `DictMergingException` (`Gym.Collections`) | `new Dict(...)`'s and the merges' errors | `Dict`, `Merger` | None needed: `new Info { ["k"] = v }` is checked when it compiles |

**What each way of removing them costs 0.2 code.** In 2.0, 0.2 code reaches these types only by naming them or
through `Gym.Legacy`; 0.2's `Env` base and its envs are gone already.

| Way | What stops compiling | What changes silently |
|---|---|---|
| Keep all 21 until 3.0 | Nothing | A `catch (InvalidActionError)` never fires |
| `[Obsolete]` on all 21 now, removed in 3.0 with `Gym.Legacy` | Nothing: each use warns, naming its replacement | Nothing: the exceptions' warnings say nothing throws them |
| Remove the six unused now, `[Obsolete]` on the fifteen | Code naming one of the six; a dead `catch` becomes an error where the behavior changed | Nothing |
| Remove all 21, the legacy info typed `Info` | The above, and `Dict`-only members used through the legacy API: `new Dict(...)`, `Get<T>`, `AsDynamic()`, `Merge`/`Append`/`Override`, `Clone`, `RemoveWhere`/`SetWhere`, `Dict` passed as a `Dictionary<string, object>`. Indexing, `ContainsKey`, `TryGetValue` and `foreach` keep compiling | Nothing |

The port plan had set their direction before any of this (§4.4): `Dict` replaced by `Info`; `Internal/Dynamic`,
`Merging` and `Reflection` deleted or made internal (for 0.2 code, internal costs what removal costs);
`AlreadySteppingError` and `NotSteppingError` moved to `Gym.Legacy` for one major version; `InvalidActionError` kept
"for LunarLander until P8", and P8's LunarLander raises `AssertionError`. Keeping `Dict` public has one more cost: it
shares its name with the Dict space, `Gym.Spaces.Dict`, so a file importing both namespaces gets `CS0104`, which is why
`Gym.Legacy`'s files alias it. Marking the types `[Obsolete]` edits their files, which then need documentation for their
61 undocumented or malformed members (the repository's rule for edited code).

The same check found 0.2's `Gym.Threading` (`DistributedScheduler`, `DistributedThread`,
`DistributedQueuedTaskScheduler`, `ThreadAffinityDistrubution`, `CancellationSourceExtensions`), public in 0.2.0,
already removed by P1 (`da428ab`, built on `Thread.Abort`, which throws on .NET 5+); the migration guide now lists it.

**Decided (DR-56):** the six removed, the fifteen `[Obsolete]` (each warning names its replacement) and documented,
removed in 3.0 with `Gym.Legacy`. The maintainer added: "consider each if we could allow an api with backwards
compatability". Type by type, what an API keeping 0.2 code compiling could be:

| Type | A backward-compatible API? | Done |
|---|---|---|
| `CircularQueue<T>`, `CircularQueueThreadSafe<T>`, `GenericCallFactory` | Only by keeping the type: nothing in Gymnasium or Gym.NET has their shape, and the BCL's replacements differ in name and members. 0.2's env API exposed none of them | Removed |
| `InvalidActionError` | Partly. Restored as the base class of `InvalidAction`, `catch (InvalidActionError)` would catch CarRacing's and custom envs' invalid actions. It would still miss the `AssertionError` most envs raise (NumSharp's type, which cannot derive from it), and it would put a class Gymnasium does not have into Gymnasium's error hierarchy | Removed; the migration guide names both replacements |
| `AlreadySteppingError`, `NotSteppingError` | Only as base classes of `AlreadyPendingCallError` and `NoAsyncCallError`, which would make old `catch` blocks fire where they never did (0.2 threw neither): new behavior, not compatibility | Removed |
| `Dict` | It is the compatible API through 2.x. For code moving to `Info` before 3.0, `Info` could take `Dict`'s reads: `Get<T>(key)` and `Clone()` are safe additions. A `Get<T>(key, T fallback)` overload is not: overload resolution prefers it to today's `Get(key, object? fallback)` (an identity conversion beats boxing), so `info.Get("k", 5)` would turn into a cast that throws `InvalidCastException` for a stored `long`. Either way `Info` grows beyond Python's dict | Not added; the migration guide maps each member |
| `Expando`, `Merger`, `IMergable`, `IMergable<T>`, `MergePrefer`, `ExpandableList` (3), `DefaultConstructor` (3), `ReflectionHelper`, the two `Dict` exceptions | Their replacements are the BCL's own types (`ExpandoObject`, `List<T>`, `Activator`, reflection), whose API Gym.NET cannot reshape; none was part of 0.2's env API | Nothing to add |

### 3.8 Stale documentation found (corrected)

- `Env<TObs, TAct>`'s remarks still said spaces are "the non-generic `Space` until the spaces work (P2) makes them
  `Space<T>`"; they are `Space<TAct>`/`Space<TObs>`.
- Its `TObs` doc said Discrete observations are `long`; since DR-08 they are 0-d `NDArray`s.

## 4. Measurements

**How the first edition went wrong.** Its benchmark was a .NET 10 file-based app with `#:property Configuration=Release`
and `#:project` references to the libraries. The property compiles the app in Release but does not reach the
references, which receive only a build's global properties: Gym, Gym.Environments, Gym.Box2D and Gym.Rendering ran as
Debug builds (the copies beside the app hashed equal to `bin/Debug`), with the JIT's optimizations, inlining among them,
switched off. Re-measuring found two more traps: `dotnet run app.cs` reuses its last build while the app's own file is
unchanged, even after a referenced project's sources change (`--no-cache` rebuilds), and the machine's load moved
Gymnasium's CartPole from 95,558 steps/s in the morning to the 65,464 below. Every number here was taken with
`dotnet run -c Release app.cs`, each library's hash checked against its `bin/Release` build.

**Throughput** (net10.0, Release; sampled actions, observations disposed; Gymnasium v1.4.0 on CPython 3.12.12 with the
pinned NumPy 2.4.2, pygame-ce 2.5.8 and Box2D 2.3.10, the same loop; five rounds of the three Gym.NET builds below and
four of Gymnasium, interleaved; medians). Single runs strayed up to 25% from the median, the ratios less:

| Env | Gym.NET steps/s | Gymnasium steps/s | Ratio | Gym.NET bytes/step |
|---|---:|---:|---:|---:|
| CartPole | 93,898 | 65,464 | 1.4× faster | 3,266 |
| MountainCar | 145,364 | 43,440 | 3.3× faster | 3,192 |
| Pendulum | 48,601 | 12,930 | 3.8× faster | 11,088 |
| Acrobot | 168,049 | 11,696 | 14.4× faster | 4,019 |
| FrozenLake | 213,280 | 59,436 | 3.6× faster | 4,283 |
| Taxi | 164,917 | 36,414 | 4.5× faster | 4,681 |
| LunarLander | 19,744 | 12,503 | 1.6× faster | 8,993 |
| BipedalWalker | 9,024 | 2,499 | 3.6× faster | 35,071 |
| CarRacing | 164 | 74 | 2.2× faster | 3,631,032 |

Gym.NET's column is the tree with R2 and R3. The other two builds were the committed sources (`src` as it is from
`489d589` to `610d912`), exported with `git archive`, as they were and with R1's `b2Math.cs`.

**R1 in Release.** LunarLander 19,697 steps/s at HEAD and 20,777 with R1 (1.05×), BipedalWalker 8,110 and 9,122
(1.12×), while each build's five rounds spread wider than that (LunarLander 15,728–23,827, BipedalWalker
6,028–10,511). In Release the JIT already inlines the small operators; the first edition's 5.9× and 6.4× compared a
Debug engine with a Release one, and its "3× slower than Gymnasium" was the Debug engine too.

**R2 and R3.** CarRacing in the same rounds: 103 steps/s at HEAD, 164 with both (1.6×), and 7,065,020 bytes a step
down to 3,631,032. R2's microbenchmark, an app with both shrinks compiled into it (so the trap did not touch it):
identical output on 200 random frames of random sizes, 13.2 ms against 4.4 ms for CarRacing's 1000×800 → 96×96 (100
runs after 20 warm-up runs, twice).

**Correctness.** With R2, R3, R10, R11 and R16 in the tree, the unit gate (3,549 tests) and the live parity gate
(2,451 cells) pass on net8.0 and net10.0. R1 passed both before it was reverted, and after the revert the Box2D,
LunarLander, BipedalWalker and CarRacing tests of both suites passed again.

**Profiles** (`dotnet-trace`, sampled thread time, 6 s of sampled actions, Release; inclusive share of the stepping
thread). CarRacing, with R2 and R3: `CarRacingEnv.RenderFrame` 84%, `Transform.Smoothscale` 38% (`ShrinkLine` 36%),
`CreateImageArray` (the frame copied into an array) 20%, `RenderRoad` 19%, `Transform.Flip` 11%, SDL_gfx's filled
polygons 10%. LunarLander's profile is rougher, the sampler putting 38% of the thread under GC-poll frames: `LunarLanderEnv.Step`
46%, of it `b2World.Solve` 22% and `b2RevoluteJoint.SolveVelocityConstraints` 5%; building the observation
(`np.array`) 18%; `Discrete.Sample` 9%. The first edition's profiles were Debug builds and are withdrawn: they showed
every vector operator as a frame because Debug code inlines nothing.

## 5. Suggested order

1. R2, R3: no API change, measured, verified by the existing gates. *Done (DR-50)*; R1 withdrawn.
2. R5, R6, R7, R8, R9, R10, R11: the 2.0 surface, while breaking is free; R5 and R6, with the two `cref` fixes, clear
   every build warning. *R10 and R11 done (DR-51, DR-52), R8 and R9 kept with TODOs (DR-53), R7 not taken; R5 and R6
   done (DR-56), one warning left (CS0169, a never-used field of 0.2's `DefaultConstructor`, kept as it was).*
3. R16: nullability project-wide, then warnings as errors everywhere. *Done, warnings staying warnings (DR-54).*
4. R12, R13, R14: the equality and scoping semantics, each a short design decision. *R14 done (DR-57).*
5. R17 and U1/U2: documentation now, annotations and the NumSharp items after. *R17 closed (DR-55); U1 and U2 are
   NumSharp's.*
6. R19, R21, R23: naming and typing decisions with no urgency.
