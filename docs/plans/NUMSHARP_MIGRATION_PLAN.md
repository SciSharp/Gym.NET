# NumSharp Migration Plan: NumSharp.Lite 0.1.12 → NumSharp master

| | |
|---|---|
| **Status** | Draft 1, 2026-09-23 |
| **Goal** | Replace `NumSharp.Lite 0.1.12` with current NumSharp, adopt NumSharp's `long` (int64) indexing across Gym.NET's own API, and keep every test green |
| **Target** | `refs/NumSharp` master @ `93967990` (v0.70.0 + 46). The published NuGet 0.70.0 is not enough (§3.2) |
| **Audited Gym.NET** | the code of `pr/24` @ `2639785`, as carried by `docs/gymnasium-port-plan` @ `48e095d` |
| **Parent plan** | [`GYMNASIUM_PORT_PLAN.md`](GYMNASIUM_PORT_PLAN.md) §5.1 (F1), phase P0 |
| **Evidence** | A complete trial migration of a scratch copy of the repository, built, tested, probed and benchmarked on 2026-09-23 (Windows 11 x64, .NET SDK 10.0.101 with runtimes 8.0.29 and 10.0.1, Python 3.12.12 with NumPy 2.4.2). Appendix D says how to reproduce every number. |

---

## Contents

1. [Summary](#1-summary)
2. [Scope](#2-scope)
3. [What changed in NumSharp](#3-what-changed-in-numsharp)
4. [Consumption and build plumbing](#4-consumption-and-build-plumbing)
5. [The int → long policy](#5-the-int--long-policy)
6. [Repairs, file by file](#6-repairs-file-by-file)
7. [Runtime traps the compiler does not catch](#7-runtime-traps-the-compiler-does-not-catch)
8. [Ownership: disposable arrays and the NDW analyzer](#8-ownership-disposable-arrays-and-the-ndw-analyzer)
9. [RNG streams and baselines](#9-rng-streams-and-baselines)
10. [NumSharp.Lite workarounds to delete](#10-numsharplite-workarounds-to-delete)
11. [Defect register delta](#11-defect-register-delta)
12. [Verification](#12-verification)
13. [Execution order](#13-execution-order)
14. [Upstream NumSharp items](#14-upstream-numsharp-items)
15. [Decisions for the maintainer](#15-decisions-for-the-maintainer)
- [Appendix A: Trial compile log](#appendix-a-trial-compile-log)
- [Appendix B: NumSharp API map for every call Gym.NET makes](#appendix-b-numsharp-api-map-for-every-call-gymnet-makes)
- [Appendix C: Measurements](#appendix-c-measurements)
- [Appendix D: Reproduction](#appendix-d-reproduction)

---

## 1. Summary

The migration is small in lines and large in consequences. Gym.NET touches NumSharp in 13 files (1,961 lines, 905 of them LunarLander), yet current NumSharp changes the meaning of nearly every call those files make:

- indices, sizes and shapes are `long`;
- `np.float32` is a `DType` descriptor, not a `System.Type`;
- array→scalar conversions are explicit and work on 0-d arrays only;
- comparisons no longer collapse to `bool`;
- `np.random` is NumPy's MT19937;
- `NDArray` is disposable, and the package ships an ownership analyzer.

A trial migration of a scratch copy established the full picture:

| Measure | Result |
|---|---|
| Compile errors against master | 28 distinct sites (56 diagnostics over net8.0 and net10.0): `Gym` 13, `Gym.Environments` 14, tests 1. `examples/`, which does not build today, has 7 more by inspection |
| Code that compiles but misbehaves | 11 runtime traps (§7). For example, `(int)action` throws `InvalidCastException` on a sampled action, `NDArray.GetHashCode()` throws, and `np.byte` is now int8 |
| RNG | 602 of 602 draws (10 patterns × 7 seeds up to 2^32−1) are bit-identical to NumPy 2.4.2's `np.random.RandomState` |
| Seeded baselines | All change once. LunarLander seed 1000 goes from 245 steps / 35.515747 to 230 steps / 40.93515, identical in Debug and Release on both TFMs |
| Ownership analyzer | 23 Gym.NET sites (20 NDW012, 2 NDW016, 1 NDW017), all removed without changing one result bit |
| After the repair | 0 errors and 0 NDW sites. The 6 headless tests pass in Debug and Release on net8.0 and net10.0 (24 of 24 runs) |
| Throughput | LunarLander about 20% faster. CartPole about 2× slower per step (0.72–0.89 µs vs 0.34–0.37 µs) because the observation arrays it hands out are reclaimed by the finalizer (§8.4) |
| Defects fixed on the way | B-01, B-02, B-04, B-05, B-06, B-07 and B-37; B-13's aliasing; B-29 with step 3 of §13 |
| Blocker | NumSharp master is unreleased. Against NuGet 0.70.0 the migrated code fails with 4 errors (§3.2), so Gym.NET consumes `refs/NumSharp` as a submodule until the next NumSharp release |

---

## 2. Scope

In scope:

- dependency swap and build plumbing (§4);
- the int → long policy for every value that crosses into or out of NumSharp (§5);
- compile repairs (§6), compile-silent runtime traps (§7), ownership (§8), RNG and baselines (§9);
- deleting the NumSharp.Lite workarounds (§10), which is the exit criterion of the port plan's F1;
- tests that pin all of the above (§12).

Out of scope, because the port plan's phases own them (§11 lists each defect): the Gymnasium re-port of `Env`, spaces and envs; `Generator(PCG64)` seeding (F2); float64 physics constants; float32 observations; Discrete's `()` shape; Gymnasium's validation messages.

The boundary rule: when a line must change for NumSharp, it changes to the Gymnasium-correct form if that form is unambiguous and local. For example, a masked `Discrete.Sample` now draws from the valid set. Everything else keeps today's behavior, so this migration can land before P1 without pre-empting it.

---

## 3. What changed in NumSharp

### 3.1 Changes that reach Gym.NET

| # | Change | Since | Effect on Gym.NET |
|---|---|---|---|
| 1 | int64 indexing. `nd.shape` is `long[]`, `nd.size` and `Shape.Size` are `long`, `Shape.Dimensions` is `long[]`, indexers and `GetDouble`/`GetSingle`/… take `params long[]`. `ndim` and `axis` stay `int` | 0.50.0 | `int` arguments widen silently; reading a size into an `int` no longer compiles (examples) |
| 2 | `Shape` is a `readonly struct` (in Lite it is a class) | 0.40.0 | `base(null, …)` fails (CS0037); `Equals(shape, null)` compiles and is always false |
| 3 | Array→scalar conversions are explicit and throw `IncorrectShapeException("only 0-d arrays can be converted to scalar")` when `ndim != 0`; scalar→array stays implicit | 0.50.0 | 12 LunarLander sites and one test (CS0266, CS1503) |
| 4 | Comparisons return `NDArray<bool>`, which has no implicit `bool` and no `operator true`/`false` | 0.41.0 / 0.50.0 | `if (a != null)` (CS0266); `x >= Low && x <= High` (CS0218) |
| 5 | `DType` is the one dtype spelling: `np.float32` and friends are `DType`, `NDArray.dtype` is `DType`, and `DType → Type` is explicit | master only (`c916c579`, `68a02258`, 2026-09-09) | `Type dType = dType ?? np.float32` (CS0266) three times |
| 6 | `np.full(shape, fill_value, dtype)`: the argument order flipped | 0.60.0 | `np.full(value, shape)` (CS1503) twice |
| 7 | `np.random` rebuilt on MT19937, byte-identical to NumPy's legacy `RandomState`. Scalar draws return 0-d `NDArray`s; `randint` defaults to int32 (NumPy 2: int64); the array-bound `uniform(NDArray, NDArray, DType)` computes in the operand dtype and takes no size; only `RandomState(int)` exists; `seed(long)` validates `[0, 2^32−1]`; an unseeded `RandomState()` is seeded from `Environment.TickCount` | 0.60.0 | Every seeded sequence changes. `uniform(Low[m], High[m], shape)` fails (CS1503) twice |
| 8 | `np.byte` is int8 (NumPy's C `char`) | 0.60.0 | `DType == np.@byte` in `Box.Sample` silently stops meaning uint8 |
| 9 | `NDArray.GetHashCode()` throws (NumPy arrays are unhashable); `Equals(object)` is element-wise with broadcasting | 0.50.0 | Compiles; `Box` and `Step` hashing throws; `Box == Box` of unbroadcastable shapes throws |
| 10 | `NDArray` is `IDisposable` (atomic refcount, buffer pool, finalizer backstop); the package ships an ownership analyzer (NDW012/016/017) | 0.60.0 / 0.70.0 | 23 warning sites (§8) |
| 11 | `NDArray.ToString()` prints NumPy's `array_str` | 0.60.0 | Cosmetic (`Step.ToString`) |
| 12 | Every assembly is strong-named (token `cc7b13ffcd2ddd51`) | 0.70.0 | None; an unsigned Gym.NET may reference it |
| 13 | TFMs `net8.0;net10.0` only | 0.40.0 | Matches `src/` since PR #24; the netstandard2.0/netcoreapp3.0 examples cannot reference it |

### 3.2 Why master and not NuGet 0.70.0

The only NumSharp on NuGet newer than Lite is 0.70.0 (2026-09-06). Change 5 landed after it. Built against the 0.70.0 package, the migrated code fails with 4 errors:

- three CS0121: `astype(Type, bool)` and `astype(NPTypeCode, bool)` are ambiguous for a `DType` argument;
- one CS1061: `Type` has no `kind`.

The code could be bent to compile against both versions, but only by reintroducing the `Type`/`NPTypeCode` spellings that Stage B deleted. Target master, and cut the next NumSharp release before Gym.NET ships (U1 in §14).

---

## 4. Consumption and build plumbing

### 4.1 Submodule

`refs/NumSharp` is an untracked clone today; `.gitmodules` lists only `refs/Gymnasium`. Register it as a submodule pinned at `93967990`.

NumSharp has six submodules of its own: `refs/numpy`, `refs/pocketfft`, `refs/OpenBLAS`, `refs/onnxruntime`, `refs/onnxruntime-inference-examples` and `refs/data`. `NumSharp.Core` builds with none of them; all six were uninitialized throughout the trial. Initialize non-recursively:

```bash
git submodule update --init refs/Gymnasium refs/NumSharp
```

`CLAUDE.md:102` and `docs/guides/development_and_testing.md:13` currently say `git submodule update --init --recursive`. That would clone NumPy, OpenBLAS and ONNX Runtime. Change both.

### 4.2 One switch for source vs package

Add `eng/NumSharp.props` and import it from `Gym.csproj`, `Gym.Environments.csproj` and `Gym.Tests.csproj`:

```xml
<Project>
  <!--
    Where Gym.NET gets NumSharp from. Imported by every project that compiles against NumSharp or must be
    analyzed by NumSharp's ownership analyzer (src/Gym, src/Gym.Environments, tests/Gym.Tests).

    NumSharpSource=Submodule (default): builds refs/NumSharp/src/NumSharp.Core from source. Required while Gym.NET
      depends on NumSharp master, because DType Stage B is not in NuGet 0.70.0. The analyzer is referenced
      explicitly because a ProjectReference does not carry it (NumSharp.Core references it with
      PrivateAssets="all"); without it, submodule builds would silently skip the NDW diagnostics that package
      consumers get.
    NumSharpSource=Package: consumes the published package, which ships the analyzer and the NDW013 guard itself.
      Flip only after setting NumSharpPackageVersion to the first release that contains Stage B.
  -->
  <PropertyGroup>
    <NumSharpSource Condition="'$(NumSharpSource)' == ''">Submodule</NumSharpSource>
    <NumSharpRoot>$(MSBuildThisFileDirectory)..\refs\NumSharp\</NumSharpRoot>
    <!-- Deliberately empty until a NumSharp release contains Stage B; the Package branch is unusable before that. -->
    <NumSharpPackageVersion></NumSharpPackageVersion>
  </PropertyGroup>

  <ItemGroup Condition="'$(NumSharpSource)' == 'Submodule'">
    <ProjectReference Include="$(NumSharpRoot)src\NumSharp.Core\NumSharp.Core.csproj" />
    <ProjectReference Include="$(NumSharpRoot)tools\NumSharp.Build.Analyzer\NumSharp.Build.Analyzer.csproj"
                      OutputItemType="Analyzer" ReferenceOutputAssembly="false" PrivateAssets="all" />
  </ItemGroup>

  <ItemGroup Condition="'$(NumSharpSource)' == 'Package'">
    <PackageReference Include="NumSharp" Version="$(NumSharpPackageVersion)" />
  </ItemGroup>

  <!-- A missing submodule otherwise surfaces as dozens of CS0246 errors that do not name the cause. -->
  <Target Name="GymNetRequireNumSharpSubmodule" BeforeTargets="ResolveProjectReferences"
          Condition="'$(NumSharpSource)' == 'Submodule' and !Exists('$(NumSharpRoot)src\NumSharp.Core\NumSharp.Core.csproj')">
    <Error Text="refs/NumSharp is missing. Run: git submodule update --init refs/NumSharp (not --recursive)." />
  </Target>
</Project>
```

### 4.3 Build hygiene

- Building NumSharp.Core from source also builds its IL weaver (`tools/NumSharp.Build`) and its analyzer. It prints about 1,260 NumSharp-internal warnings (1,108 of them CS8632, the rest CS1718, CS3021, CS0219 and others). They are NumSharp's, not Gym.NET's.
- Never pass `-warnaserror` on the command line, because it also applies to NumSharp's projects. Set `TreatWarningsAsErrors` per project, as the port plan's §3.7 already requires. Neither repository's `Directory.Build.props` reaches the other's projects: MSBuild stops at the first file it finds walking upward, and `refs/NumSharp` has its own.
- Strong naming needs nothing from Gym.NET.

### 4.4 Flip to the package

When a NumSharp release contains Stage B:

1. set `NumSharpPackageVersion` and `NumSharpSource=Package`;
2. drop the analyzer `ProjectReference`, because the package ships the analyzer.

Keep the submodule for development and for agents, per the port plan's decision D3.

---

## 5. The int → long policy

NumSharp settled "int or long" in 0.50.0. Every element count, dimension, stride, offset and index is `long`; only `ndim` and `axis` stay `int`. Gym.NET should mirror that in two places:

- wherever a value crosses into or out of NumSharp;
- wherever Gymnasium's value is a Python `int` that NumPy stores as `int64`.

### 5.1 Rules

| Category | Type | Examples |
|---|---|---|
| NumSharp shapes, sizes, strides, offsets, element indices | `long`, NumSharp's own type | `Shape`, `nd.shape` (`long[]`), `nd.size`, `GetDouble(params long[])` |
| Discrete `n`, `start` and elements | `long` (Gymnasium: `np.int64`) | `Discrete.N`, `Discrete.Start`, `Sample()` returning a 0-d int64 array, `Contains(long)` |
| Actions of discrete envs | `long`, obtained through `Discrete.ToInt64(object)` | `CartPoleEnv.Step`, `LunarLanderEnv.Step`, `VecEnv.Step(long)` |
| Seeds | `long`, range `[0, 2^32−1]` enforced by NumSharp with NumPy's message | `IEnv.Seed(long)`, `Space.Seed(long)`, `VecEnv.Seed(long[])` |
| Env state NumPy produces as int64 | `long` | LunarLander `_wind_idx`, `_torque_idx` |
| `ndim`, `axis` | `int`, as NumSharp keeps them | none in Gym.NET today |
| Counts of managed objects, loop indices over managed arrays, pixel sizes, FPS, enum backing types | `int`, the .NET API boundary | `VecEnv.NumberOfEnvironments`, `VIEWPORT_W`, `LunarLanderDiscreteActions : int` |
| A NumSharp `long` passed to an `int` API | `checked((int)x)` at that boundary, never a silent `(int)` | the examples' `Shape.Size` passed to `NetworkLayers.Softmax(int)` |
| Python `int` counters that never touch NumSharp | stay `int` here; the P4 and P8 re-ports decide | CartPole `steps_beyond_done` (P4: `steps_beyond_terminated`) |

### 5.2 Public API changes

| Member | Before | After |
|---|---|---|
| `Space.DType` | `System.Type` | `NumSharp.DType` |
| `Space.Seed` | `(int)` | `(long)` |
| `Space` | not disposable | `IDisposable` (§8) |
| `Discrete(…)` | `(int n, Type dType = null, int seed = -1, int start = 0, NumPyRandom random_state = null)` | `(long n, DType dType = null, long seed = -1, long start = 0, NumPyRandom random_state = null)` |
| `Discrete.N`, `Discrete.Start` | `int` | `long` |
| `Discrete` default dtype | float32 | int64 |
| `Discrete.Contains` | `(int)` | `(long)`; honors `Start` |
| `Discrete.ToInt64`, `TryToInt64` | none | new |
| `Box(…)` constructors | `Type dType`, `int seed` | `DType dType`, `long seed` |
| `IEnv.Seed`, `Env.Seed` | `(int)` | `(long)` |
| `IVecEnv.Step`, `VecEnv.Step`, `VecEnv.StepAsync` | `(int action)` | `(long action)` |
| `IVecEnv.Seed`, `VecEnv.Seed` | `(int)`, `(int[])` | `(long)`, `(long[])` |
| `Env.Dispose` | calls `CloseEnvironment()` | `Dispose(bool)` pattern that also disposes the spaces |
| `Step.Observation` | `NDArray` | `[NDBorrowed] NDArray` |
| `Gym.Utils.LegacySeeding` | none | new (§9.3) |

Existing callers keep compiling where C# widens implicitly (`Seed(5)`, `new Discrete(2)`, `space.DType == typeof(float)`). `Type t = space.DType` needs `(Type)` or `.type`. Everything is a binary break: consumers recompile.

### 5.3 Pitfalls the compiler will not catch

- An `int[]` does not convert to `long[]`. `Shape` converts implicitly from both, so pass shapes as `Shape` or `long[]`.
- Unboxing is exact: `(int)(object)5L` throws. Every `object action` goes through `Discrete.ToInt64`.
- `NumPyRandom.Seed` is an `int` property and wraps for seeds of 2^31 and above (U7). Do not read it back.
- Narrow with `checked`, so a size above `int.MaxValue` fails loudly instead of wrapping.
- NumSharp's explicit array→scalar conversions are C-style casts: `(long)` of a 0-d uint64 array holding `ulong.MaxValue` returns −1 (measured). Convert uint64 through `checked((long)(ulong)nd)`.

---

## 6. Repairs, file by file

"CSxxxx" references the errors listed in Appendix A. "Trap n" references §7.

### 6.1 Project files

- `src/Gym/Gym.csproj`, `src/Gym.Environments/Gym.Environments.csproj`: remove `<PackageReference Include="NumSharp.Lite" Version="0.1.12" />`; import `eng/NumSharp.props` (§4.2).
- `tests/Gym.Tests/Gym.Tests.csproj`: import `eng/NumSharp.props`, so the analyzer also covers the tests.

### 6.2 `src/Gym/Spaces/Space.cs`

- `Type DType` becomes `DType DType`. This fixes the three CS0266 at the constructors that default to `np.float32`.
- `Seed(int)` becomes `Seed(long)`.
- `Space : IDisposable`, with `Dispose()` and a `protected virtual Dispose(bool)` (§8).

### 6.3 `src/Gym/Spaces/Box.cs` (10 errors, traps 2, 3, 4, 7 and 11)

- Constructor dtype parameters become `DType`, seeds become `long`, and generators come from `LegacySeeding` (§9.3).
- `Box(NDArray low, NDArray high, …)`: `base(null, …)` (CS0037) becomes `base(RequireSameShape(low, high), …)`, a static check for null and equal shapes. It replaces a `Debug.Assert` that vanished in Release. The dead `Equals(shape, null)` guard in the scalar constructor goes (trap 7).
- `Box(float low, float high, Shape shape, …)`: `Low = np.full(shape, low, DType)` replaces `(low + np.zeros(shape, dType)).astype(dType)`.
- `CheckBounded`: `BoundedLow = Low > double.NegativeInfinity` and `BoundedHigh = High < double.PositiveInfinity` replace comparisons against `np.full(±inf, shape)` arrays (CS1503 twice).
- `IsBounded`: `np.all(...)` directly; the 0-d `Any`/`All` helpers go (§10).
- `Sample`:
  - delete the 0-d scalar path (§10) and use `mask is not null`;
  - count each category with `np.count_nonzero(mask)` instead of materializing `mask[mask]`;
  - wrap temporaries in `using`;
  - integer check `DType.kind is 'i' or 'u' or 'b'` (Gymnasium's `dtype.kind in ["i", "u", "b"]`), not `np.int32`/`np.uint32`/`np.@byte` (trap 4).
  - The draw order stays unbounded, low-bounded, upper-bounded, bounded. The bounded branch uses NumPy's own float64 computation (CS1503 twice; trap 11):

  ```csharp
  // numpy.random.RandomState.uniform(low_array, high_array) computes lower + range * next_double() per element with
  // both bounds cast to float64 (numpy/random/mtrand.pyx, random_uniform). NumSharp's uniform(NDArray, NDArray, DType)
  // computes in the operand dtype and takes no size, so it is not used. Verified bit-identical to NumPy (§9.1).
  using var lows = Low[bounded];
  using var highs = High[bounded];
  using var low64 = lows.astype(np.float64);
  using var high64 = highs.astype(np.float64);
  using var range = high64 - low64;
  using var u = RandomState.random_sample(n);
  using var scaled = range * u;
  using var drawn = low64 + scaled;
  sample[bounded] = drawn;
  ```

  Defects B-08 (unbounded dims draw `normal(0.5, 1)`) and B-09 (integer `high` never drawn) stay for P2.
- `Contains`: return false on `x.Shape != Shape`, then `np.all(x >= Low) && np.all(x <= High)` with scoped temporaries (CS0218).
- Equality and hashing (CS0266 twice; traps 2 and 3):

  ```csharp
  /// <summary>
  /// Structural equality: same shape, same dtype, element-identical bounds. Arrays are compared with
  /// <see cref="np.array_equal(NDArray, NDArray)"/>, never with <see cref="NDArray.Equals(object)"/>, which broadcasts
  /// (a 0-d array equals every array holding that value) and throws for shapes that cannot broadcast.
  /// </summary>
  /// <param name="other">The box to compare with; may be <see langword="null"/>.</param>
  /// <returns><see langword="true"/> when both boxes describe the same set of points.</returns>
  /// <remarks>Gymnasium's <c>Box.__eq__</c> compares bounds with <c>np.allclose</c>; the exact comparison stays until P2 ports it.</remarks>
  public bool Equals(Box other)
  {
      if (other is null) return false;
      if (ReferenceEquals(this, other)) return true;
      return Shape == other.Shape && DType == other.DType
          && np.array_equal(Low, other.Low) && np.array_equal(High, other.High);
  }

  /// <summary>Hashes shape and dtype only, because <see cref="NDArray.GetHashCode"/> throws (arrays are mutable).</summary>
  /// <returns>A hash consistent with <see cref="Equals(Box)"/>: equal boxes always share shape and dtype.</returns>
  public override int GetHashCode() => HashCode.Combine(Shape, DType);
  ```

- `Dispose(bool)` disposes `Low`, `High`, `BoundedLow` and `BoundedHigh` (§8).

### 6.4 `src/Gym/Spaces/Discrete.cs` (2 errors, trap 10, defects B-02 and B-04)

- `N` and `Start` become `long`; constructor per §5.2; the default dtype becomes int64.
- `Sample`:
  - `mask != null` becomes `mask is not null` (CS0266).
  - The masked branch draws `RandomState.choice(np.nonzero(mask == 1)[0])`. It was `choice((int)np.nonzero(...)[0])`, which converted the whole index array to a single `int` (B-04).
  - The unmasked branch draws `randint(0, N, dtype: np.int64)`, with an explicit dtype because NumSharp defaults to int32 (trap 10).
  - Results are built as `NDArray.Scalar(Start + value)`, a 0-d int64 array, and temporaries are disposed.
- `Contains(object)` goes through `TryToInt64`. `Contains(long)` honors `Start` (B-02). `Contains(Enum)` uses `Convert.ToInt64` instead of `(int)(object)x`, which threw for enums not backed by `int`.
- New action conversion, used by every env:

  ```csharp
  /// <summary>
  /// Converts an action handed to <c>Env.Step(object)</c> into the discrete value it denotes. Envs must call this instead
  /// of casting: <c>(int)action</c> unboxes exactly and throws <see cref="InvalidCastException"/> as soon as the action is a
  /// <see langword="long"/>, or the 0-d int64 <see cref="NDArray"/> that <see cref="Sample"/> returns.
  /// </summary>
  /// <param name="action">A boxed integer of any width, an enum member, or a 0-d integer <see cref="NDArray"/>.</param>
  /// <returns>The action as <see langword="long"/>, the .NET spelling of Gymnasium's <c>np.int64</c>.</returns>
  /// <exception cref="ArgumentException">Thrown when <paramref name="action"/> is <see langword="null"/>, not an integer, or an array that is not 0-d.</exception>
  /// <exception cref="OverflowException">Thrown when an unsigned 64-bit action exceeds <see cref="long.MaxValue"/> (see <see cref="TryToInt64"/>).</exception>
  public static long ToInt64(object action)
  {
      if (TryToInt64(action, out long value))
          return value;
      throw new ArgumentException(
          $"{action ?? "null"} ({action?.GetType().Name ?? "null"}) is not a discrete action: expected an integer, an enum or a 0-d integer NDArray.",
          nameof(action));
  }

  /// <summary>
  /// Non-throwing form of <see cref="ToInt64"/> for membership tests, so <see cref="Contains(object)"/> decides what an
  /// unconvertible input means instead of surfacing a conversion error.
  /// </summary>
  /// <param name="action">The candidate action.</param>
  /// <param name="value">The converted value when the method returns <see langword="true"/>; otherwise 0.</param>
  /// <returns><see langword="true"/> when <paramref name="action"/> denotes an integer.</returns>
  /// <exception cref="OverflowException">
  /// Thrown when an unsigned 64-bit action (a <see langword="ulong"/>, a 0-d uint64 array or an enum backed by
  /// <see langword="ulong"/>) exceeds <see cref="long.MaxValue"/>.
  /// </exception>
  public static bool TryToInt64(object action, out long value)
  {
      switch (action)
      {
          case long l: value = l; return true;
          case int i: value = i; return true;
          case short s: value = s; return true;
          case sbyte sb: value = sb; return true;
          case byte b: value = b; return true;
          case ushort us: value = us; return true;
          case uint ui: value = ui; return true;
          case ulong ul: value = checked((long)ul); return true;
          case Enum e: value = Convert.ToInt64(e); return true;
          // Gymnasium's Discrete.contains accepts 0-d integer arrays (np.int64 scalars) as well as Python ints.
          case NDArray nd when nd.ndim == 0 && nd.dtype.kind is 'i' or 'u':
              // NumSharp's (long) is a C-style cast: a uint64 above long.MaxValue wraps silently (ulong.MaxValue
              // becomes -1, measured), so that dtype takes a checked conversion.
              value = nd.typecode == NPTypeCode.UInt64 ? checked((long)(ulong)nd) : (long)nd;
              return true;
          default:
              value = 0;
              return false;
      }
  }
  ```

- `Seed(long)`.
- The shape stays `(n)` (B-03, P2), because the examples read `ActionSpace.Shape.Size` as the action count.

### 6.5 `src/Gym/Observations/Step.cs` (1 error, traps 2 and 3)

- `GetHashCode` hashes `Observation.Shape`, not the data (CS0266, and `NDArray.GetHashCode()` throws).
- `Equals` compares observations by shape, dtype and `np.array_equal`, without broadcasting.
- `Observation` gains `[NDBorrowed]` (§8.2).

### 6.6 `src/Gym/Envs/IEnv.cs`, `src/Gym/Envs/Env.cs`

- `Seed(long)`.
- `Env<TAction>` replaces `(TAction)(object)action` with `(TAction)Enum.ToObject(typeof(TAction), Discrete.ToInt64(action))` (trap 1).
- `Dispose` follows the `Dispose(bool)` pattern: `CloseEnvironment()`, then dispose `ActionSpace` and `ObservationSpace` (§8).

### 6.7 `src/Gym/Envs/IVecEnv.cs`, `VecEnv.cs`, `VecEnvWrapper.cs`

`Step(long)`, `StepAsync(long)`, `Seed(long)` and `Seed(long[])`. These types still send one action to every env (B-33) until P6 deletes them.

### 6.8 New `src/Gym/Utils/LegacySeeding.cs`

The single factory for the legacy generators (§9.3, §9.4). Envs and spaces never touch `np.random` or `np.random.RandomState()` directly again.

### 6.9 `src/Gym.Environments/Envs/Classic/CartPoleEnv.cs` (trap 1, trap 6, trap 9, B-13 aliasing)

- `Step`: `long iaction = Discrete.ToInt64(action)` replaces `(int)action` (trap 1).
- `Seed(long)` through `LegacySeeding.RandomState`. NumSharp offers only `RandomState(int)`; passing a `long` is CS1503.
- State lives in a `double[]` field. `Reset` returns `np.array(state)`, and `Step` returns a fresh `np.array(state)` observation. Before, it returned the internal state array itself (B-13's aliasing). That aliasing becomes a use-after-free hazard as soon as the env disposes its own arrays (trap 9). `Render` reads the doubles.
- Constructor: `randomState ?? LegacySeeding.FromEntropy()` replaces the `Environment.TickCount`-seeded `np.random.RandomState()` (trap 6).
- Unchanged for P4: the float32 constants (B-11), `float.MaxValue` bounds (B-12), float64 observations (B-13's dtype), and `Debug.Assert` validation (B-14).

### 6.10 `src/Gym.Environments/Envs/Aether/LunarLanderEnv.cs` (12 errors, trap 1, trap 5)

- `_wind_idx` and `_torque_idx` become `long`, read from the 0-d `randint` results through a small disposing helper.
- Force, terrain heights and dispersion convert the 0-d `uniform` draws with `(double)`. Dispersion is divided in float64 and narrowed once, which is bit-identical to the NumSharp expression it replaces.
- `i_action` becomes `long`, through `Discrete.ToInt64` (trap 1).
- The continuous branch reads the two throttles once as float32 scalars:

  ```csharp
  // The NDArray expressions this replaces (c_action[0] > 0f, np.clip(c_action[0], 0f, 1f), np.abs(c_action[1]), ...)
  // allocated about five 0-d arrays per step and no longer compile (explicit conversions, NDArray<bool>). Scalar
  // float32 math is bit-identical: the same IEEE operations, and NaN propagates through Math.Clamp and np.clip alike.
  // All 16 recorded baselines (§9.2) are unchanged by this rewrite.
  float mainThrottle = c_action.GetSingle(0);
  float lateralThrottle = c_action.GetSingle(1);
  ```

- Shaping: `-100f * (float)Math.Sqrt(…)` replaces `-100f * np.sqrt(…)` (CS0266 twice). The float64 square root narrowed to float32 is correctly rounded, so it equals NumSharp's float32 `sqrt`.
- `random_state ?? LegacySeeding.FromEntropy()` replaces the fallback to the shared global `np.random` (B-29, trap 5).
- `Seed(long)`.
- Unchanged for P8: v2 observation bounds, Aether physics, the 0-d continuous action space (B-20 now throws `IndexError` instead of `OverflowException`), and the rest of B-17 through B-30.

### 6.11 Tests

- `Spaces/BoxTest.cs:40`: `float sample = (float)box.Sample(null);` (CS0266).
- `Envs/Aether/LunarLanderEnvironment.cs`:
  - the baseline for seed 1000 becomes 230 steps / 40.93515;
  - the `#if DEBUG` guard around the assertions goes (B-37), because the result is identical in Release;
  - the PID controller's temporary array gets a `using`;
  - add the other 15 baselines of §9.2.
- `Envs/Classic/CartpoleEnvironment.cs`: dispose the result of `Reset()` (NDW012).
- New tests: §12.2.

### 6.12 `examples/` (not built; decision D13)

Seven NumSharp sites, found by inspection:

- six `ActionSpace.Shape.Size` reads, now `long`, passed to `int` parameters (`CartPoleConfiguration.cs` in both runners, `ImageGameEngine.cs:15,18`, `ParameterGameEngine.cs:14,17`). Use `checked((int)((Discrete)ActionSpace).N)`;
- `ParameterReplayMemory.cs:16`: `Observation == null` becomes `Observation is null`.

They also need retargeting from netcoreapp3.0/netstandard2.0 and dropping their own `NumSharp 0.20.4` reference. Do this only if D13 keeps the examples.

### 6.13 Documentation

- `CLAUDE.md` §3.3 and `docs/guides/development_and_testing.md`: non-recursive submodule init (§4.1). Check `AGENTS.md` and `GEMINI.md` for the same command.
- The port plan's §5.1 (F1) points here.
- Release notes: the §5.2 table, the RNG stream change (§9), per-instance generators (§9.3), and `IDisposable` spaces and envs (§8).

---

## 7. Runtime traps the compiler does not catch

Each trap compiles after the migration. The "Measured" column comes from the trial's behavior probe (Appendix C.6) unless noted.

| # | Trap | Measured | Where | Fix |
|---|---|---|---|---|
| 1 | `(int)action` unboxes exactly | `InvalidCastException: Unable to cast object of type 'NumSharp.NDArray' to type 'System.Int32'` on a sampled action | `CartPoleEnv.Step`, `LunarLanderEnv.Step`, `Env<TAction>` | `Discrete.ToInt64` (§6.4) |
| 2 | `NDArray.GetHashCode()` throws | `NotSupportedException: NDArray is unhashable because it is mutable…` | `Box.GetHashCode`, `Step.GetHashCode` | hash shape and dtype |
| 3 | `NDArray.Equals(object)` broadcasts | `np.zeros(2).Equals(np.zeros(3))` throws `IncorrectShapeException: operands could not be broadcast together with shapes (2,) (3,)` instead of returning false; `NDArray.Scalar(0.0).Equals(np.zeros(3))` is true | `Box.Equals`, `Step.Equals` | shape, dtype, then `np.array_equal` |
| 4 | `np.byte` is int8 | the integer check in `Box.Sample` silently skips uint8 and floors int8 | `Box.Sample` | `DType.kind` |
| 5 | the global `np.random` is not thread-safe | two threads × 2,000,000 draws: `IndexOutOfRangeException` inside MT19937 | `Box`, `Discrete` and `LunarLander` fallbacks | per-instance generators (§9.3) |
| 6 | an unseeded `RandomState()` is seeded from `Environment.TickCount` | 1,000 of 1,000 back-to-back pairs produce identical streams; NumPy's pairs are independent | CartPole's fallback, and any naive per-instance fix | `LegacySeeding.FromEntropy` (§9.4) |
| 7 | `Shape` is a struct | `Equals(shape, null)` is always false: the null guard is dead | `Box(float, float, Shape, …)` | delete the guard; validate in `RequireSameShape` |
| 8 | the RNG streams changed | every seeded result differs (LunarLander seed 1000: 245 → 230 steps) | tests and users' own baselines | re-record once; pin with the NumPy golden (§12.2) |
| 9 | dispose plus alias | `NDArray.Dispose()` frees the buffer when its refcount reaches zero, and the same object keeps reading freed or pooled memory (source read: no access check after dispose) | CartPole handing out its internal state | hand out fresh arrays; dispose only arrays you own |
| 10 | `randint` defaults to int32 | NumPy 2 defaults to int64; the values fit but the dtype differs | `Discrete.Sample` | pass `dtype: np.int64` |
| 11 | `uniform(NDArray, NDArray, DType)` is not NumPy's `uniform(low, high, size)` | computes `low + rand(shape).astype(dtype) * (high - low)` in the operand dtype; NumPy computes in float64 | a naive `Box.Sample` port | the explicit float64 computation (§6.3) |

---

## 8. Ownership: disposable arrays and the NDW analyzer

### 8.1 Findings

With the analyzer attached (§4.2), the migrated code draws 23 diagnostics before this step:

| Rule | Count | Sites |
|---|---:|---|
| NDW012: a created array is never returned, stored, disposed or scoped | 20 | 18 in `src`: `Box.Sample` masks and draws (9), `Discrete.Sample` (2), LunarLander per-step scalar expressions (7). 2 in tests: `CartpoleEnvironment` drops `Reset()`'s result; the LunarLander PID drops its temporary |
| NDW016: a type stores arrays but is not disposable | 2 | `Box` (`Low`, `High`, `BoundedLow`, `BoundedHigh`) and `Step` (`Observation`) |
| NDW017: a disposable type never disposes a stored array | 1 | `CartPoleEnv.state`: `Env.Dispose` never reaches it |

### 8.2 Policy

- **Owned** arrays, which a type creates and keeps (`Box` bounds and masks): the type is `IDisposable` and disposes them.
- **Borrowed** arrays, which a type carries for someone else (`Step.Observation`): marked `[NDBorrowed]`, NumSharp's analyzer-only ownership statement. The same holds for the spaces a vector env references: they belong to its sub-envs, and `VecEnv` must never dispose them. The analyzer does not flag `VecEnv`, because the declared type `Space` stores no arrays itself.
- **Transients**: `using var`.
- **Hand-outs** are always fresh arrays. An env never hands out an array it will later dispose (trap 9).
- Severity is decision M6.

### 8.3 Fixes

1. `Space : IDisposable`; `Box.Dispose(bool)` disposes its four arrays.
2. `Env.Dispose(bool)` disposes both spaces after `CloseEnvironment()`.
3. CartPole keeps managed `double` state and returns fresh observations.
4. `Box.Sample`, `Box.Contains` and `Discrete.Sample` scope their temporaries.
5. LunarLander reads scalars (§6.10).
6. `Step.Observation` is `[NDBorrowed]`.
7. The two tests dispose what they create.

Result in the trial: 0 NDW sites in `src` and `tests`, with no change to any result. All 32 baseline runs and all 602 NumPy draws were identical before and after this step.

### 8.4 What finalizer-backed arrays cost

The minimum of 5 × 1,000,000 calls of `np.array(double[4])`:

| NumSharp | Undisposed | Disposed promptly |
|---|---:|---:|
| Lite 0.1.12 | 0.330 µs | n/a |
| master | 0.634 µs | 0.217 µs |

CartPole's per-step slowdown (Appendix C.2) is this undisposed path: it hands every observation to a caller that never disposes it. A caller that does dispose runs faster than on Lite. This goes into the port plan's §6.2 as decision M5: `StepResult` and `ResetResult` should implement NumSharp's `INDArrayCarrier`, so training loops can `using` them, and the docs should say "dispose observations you do not keep".

### 8.5 Why aliasing is now dangerous

`NDArray.Dispose()` drops a reference, and the last reference frees the buffer synchronously. Views keep their parent alive; the disposed object itself is not guarded. An env that disposed its state array while the caller still held that same object as its last observation would hand the caller freed memory. This is why CartPole returns copies (§6.9) and why hand-outs are always fresh (§8.2).

---

## 9. RNG streams and baselines

### 9.1 Parity with NumPy

After the migration, Gym.NET's draws are bit-identical to NumPy 2.4.2's `np.random.RandomState`. The probe (Appendix D) recorded 10 draw patterns under 7 seeds (0, 1, 42, 1000, 2^31−1, 2^31+5 and 2^32−1) in both languages and compared raw bits:

| Pattern | Draws per seed | Mirrors |
|---|---:|---|
| `randint(-9999, 9999)` × 2 | 2 | LunarLander wind and torque indices |
| `uniform(-1000, 1000)` × 2 | 2 | LunarLander initial force |
| `uniform(0, H/2)` × 12, with H/2 exactly as the env computes it in float32 | 12 | LunarLander terrain |
| `uniform(-1, 1)` × 42 | 42 | LunarLander dispersion over 21 steps |
| `uniform(-0.05, 0.05, 4)` | 4 | CartPole reset |
| `Discrete(4).Sample()` × 10 | 10 | unmasked sampling |
| `Discrete(4).Sample(mask)` × 5 | 5 | masked sampling (legacy `choice` over the valid actions) |
| `Box(-1, 1, (3,), float32).Sample()` | 3 | the float64 `uniform` computation, compared as float32 bits |
| `normal(0.5, 1, 3)` | 3 | unbounded `Box` dims |
| `exponential(1, 3)` | 3 | half-bounded `Box` dims |

Result: 602 of 602 values identical. The seeds 2^31+5 and 2^32−1 exercise the `long` seed path (§5).

This is parity with NumPy's legacy `RandomState`, which is what Gym.NET's current envs use. Gymnasium itself uses `Generator(PCG64)`, and its envs draw with `integers`, not `randint`. Parity with Gymnasium's streams arrives with F2 and the P4/P8 re-ports.

### 9.2 New baselines

LunarLander under `NullEnvViewer` with the test's PID controller. Every value was identical in Debug and Release on .NET 8.0.29 and 10.0.1 (32 runs):

| Seed | Discrete | Continuous |
|---:|---|---|
| 0 | 192 steps, 17.233437 (`0x4189de14`) | 176 steps, 23.801888 (`0x41be6a44`) |
| 1 | 251 steps, 36.771927 (`0x42131674`) | 157 steps, 41.866196 (`0x422776fc`) |
| 42 | 182 steps, −46.32995 (`0xc23951de`) | 221 steps, 47.117523 (`0x423c7858`) |
| 1000 | 230 steps, 40.93515 (`0x4223bd98`) | 355 steps, 35.39447 (`0x420d93f0`) |

On Lite, seed 1000 discrete was 245 steps / 35.515747, asserted in Debug builds only. These are Gym.NET regression baselines, not Gymnasium goldens: the physics engine is Aether, not Box2D. Assert steps exactly and the reward's float32 bits exactly, because the platform and configuration spread is zero.

### 9.3 Per-instance generators

Today, `Box`, `Discrete` and `LunarLanderEnv` fall back to the process-wide `np.random`. NumSharp's MT19937 is not synchronized, and two threads drawing from it crash (trap 5). NumPy's `RandomState` takes a lock, and Gymnasium gives every env and space its own `np_random`. Replace every fallback with a per-instance generator from a single factory:

```csharp
namespace Gym.Utils;

/// <summary>
/// Creates the legacy <see cref="NumPyRandom"/> generators (NumPy's <c>RandomState</c>, MT19937) that Gym.NET's envs and
/// spaces draw from until F2 replaces them with <c>Generator(PCG64)</c>. Use this instead of <c>np.random</c> or
/// <c>np.random.RandomState()</c>: the global <c>np.random</c> is shared and not thread-safe, and an unseeded
/// <c>RandomState()</c> is seeded from <see cref="Environment.TickCount"/>, so instances created in the same tick produce
/// identical streams.
/// </summary>
public static class LegacySeeding
{
    /// <summary>Creates a generator whose stream is bit-identical to <c>numpy.random.RandomState(seed)</c>.</summary>
    /// <param name="seed">
    /// A Python-int seed in <c>[0, 2**32 - 1]</c>. It is <see langword="long"/> because values above
    /// <see cref="int.MaxValue"/> are valid.
    /// </param>
    /// <returns>A new generator that no other instance shares.</returns>
    /// <exception cref="ValueError">Thrown when <paramref name="seed"/> is outside <c>[0, 2**32 - 1]</c>, with NumPy's message.</exception>
    /// <remarks>NumSharp offers only <c>RandomState(int)</c>, so a fresh instance is reseeded through <c>seed(long)</c> (upstream U3).</remarks>
    public static NumPyRandom RandomState(long seed)
    {
        var rng = np.random.RandomState();
        rng.seed(seed);
        return rng;
    }

    /// <summary>Creates a generator seeded from the operating system's CSPRNG, which is what NumPy does for <c>RandomState(None)</c>.</summary>
    /// <returns>A new generator whose stream is independent of every other instance and deliberately not reproducible.</returns>
    /// <remarks>
    /// Seeds MT19937 with <c>init_by_array</c> over four OS-random 32-bit words (128 bits), the routine NumPy uses for an
    /// array seed. NumPy fills the whole state from <c>SeedSequence</c> entropy instead; the stream cannot be reproduced
    /// either way, so only independence matters (upstream U2).
    /// </remarks>
    public static NumPyRandom FromEntropy()
    {
        var key = new uint[4];
        RandomNumberGenerator.Fill(MemoryMarshal.AsBytes(key.AsSpan()));
        var rng = np.random.RandomState();
        rng.seed(key);
        return rng;
    }
}
```

Behavior change for users: `np.random.seed(x)` no longer makes spaces deterministic. `space.Seed(x)` does, as in Gymnasium. Put this in the release notes.

### 9.4 Why the factory needs entropy

`new MT19937()` seeds itself with `(uint)Environment.TickCount`. In the trial, 1,000 of 1,000 pairs of `np.random.RandomState()` created back to back produced identical streams; NumPy's `RandomState()` pairs are independent. A naive `?? np.random.RandomState()` fallback would give an env's action space and observation space, or all the envs a loop creates, the same stream. `FromEntropy` avoids that until upstream U2 fixes the default.

---

## 10. NumSharp.Lite workarounds to delete

The port plan's F1 exit criterion is "no Lite-specific code remains". These are all of it:

| Workaround | Where | Replacement |
|---|---|---|
| `Any`/`All` helpers that special-case 0-d arrays | `Box` | `np.any`/`np.all`, which return `bool` for any rank |
| The 0-d scalar path in `Sample`, with its `GetSingle(0)` reads | `Box.Sample` | the masked path, which handles 0-d |
| `!Equals(mask, null)`, `mask != null` | `Box.Sample`, `Discrete.Sample` | `mask is not null` |
| `np.full(±inf, shape)` comparison arrays | `Box.CheckBounded` | comparisons with scalar infinities |
| `(low + np.zeros(shape, dType)).astype(dType)` | `Box(float, float, …)` | `np.full(shape, low, dtype)` |
| `(int)np.nonzero(bmask)[0]` | `Discrete.Sample` | `choice(np.nonzero(...)[0])` |

---

## 11. Defect register delta

Status of the port plan's Appendix B defects after this migration:

| ID | After | Note |
|---|---|---|
| B-01 | fixed | `Discrete.Sample()` works; returns a 0-d int64 array |
| B-02 | fixed | `Contains` honors `Start` |
| B-03 | partly | dtype is int64; shape stays `(n)` until P2 (the examples depend on it) |
| B-04 | fixed | masked samples come from the valid set |
| B-05, B-06 | fixed | `Sample`/`Contains` work for every shape; `Contains` still has no dtype check (P2) |
| B-07 | fixed | integer boxes construct and sample |
| B-08, B-09, B-10 | open | P2 |
| B-11, B-12, B-14 | open | P4 |
| B-13 | partly | no more aliasing (fresh observation arrays); still float64 (P4) |
| B-15 | changed | the unseeded fallback draws from OS entropy instead of `TickCount`; lazy seeding waits for P4 |
| B-16 | open | P7 |
| B-17 to B-19, B-21 to B-28, B-30 | open | P8 |
| B-20 | open | P8; now `IndexError: too many indices for array: array is 0-dimensional, but 1 were indexed` instead of `OverflowException` |
| B-29 | fixed by step 3 of §13 | per-instance generators |
| B-31, B-32 | open | P1 |
| B-33 | open | `Step(long)` still sends one action to every env; P6 deletes the type |
| B-34, B-35 | open | P1, P7 |
| B-36 | open | the examples also gain 7 NumSharp sites (§6.12) |
| B-37 | fixed | baselines assert in every configuration |
| B-38 | open | P0's other items (CI, nullable, warnings) |

---

## 12. Verification

### 12.1 Gates

| Gate | Requirement |
|---|---|
| Build | 0 errors on net8.0 and net10.0, Debug and Release, submodule mode (and package mode after §4.4) |
| Ownership | 0 NDW diagnostics in `src/` and `tests/` |
| Tests | every headless test green in all four configurations |
| RNG | the committed NumPy golden matches bit for bit (§12.2) |
| GUI | the WinForms and Avalonia viewer tests pass when run manually (they open windows) |

### 12.2 New tests

| Test | Pins |
|---|---|
| `LegacyRngParityTests` | the 602 NumPy draws of §9.1, committed as a text golden recorded by `tools/parity/legacy_rng.py` (Appendix D.2), replayed through `LegacySeeding`, `Discrete`, `Box` and LunarLander's draw order |
| `DiscreteTests` | `ToInt64` over int, long, short, byte, uint, ulong, enum and 0-d int32/int64/uint64 arrays; `OverflowException` for a `ulong` or 0-d uint64 above `long.MaxValue`; rejection of float, 1-D and null; `Contains` with a start (B-02); masked sampling stays in the mask (B-04); `Sample()` is 0-d int64 |
| `BoxTests` | non-0-d `Sample`/`Contains` (B-05, B-06); integer boxes (B-07); equality of different shapes is false; `GetHashCode` does not throw; `Dispose` disposes the bounds |
| `SeedTests` | seeds 2^31+5 and 2^32−1 are accepted; −1 and 2^32 raise `ValueError: Seed must be between 0 and 2**32 - 1` |
| `LunarLanderEnvironment` | the 16 baselines of §9.2, in every configuration |
| `CartPoleTests` | `Step(ActionSpace.Sample())` works; consecutive observations are distinct objects; `ObservationSpace.Contains(Reset())` |
| `ConcurrencyTests` | two envs and their spaces stepping on parallel threads with default generators finish without exceptions |
| `EntropySeedingTests` | two unseeded spaces created back to back produce different streams |

---

## 13. Execution order

Each step is one commit that builds and passes every gate that exists at that point.

| # | Commit | Content | Exit |
|---|---|---|---|
| 1 | `build(deps): consume NumSharp master through refs/NumSharp` | submodule at `93967990`; `eng/NumSharp.props`; the 28 compile repairs (§6, without the `long` surface); `Equals`/`GetHashCode`; the `DType.kind` check; §10's deletions; `BoxTest` cast; LunarLander baseline 230 / 40.93515 without `#if DEBUG`; non-recursive submodule commands in the docs | 0 errors; 6 of 6 headless tests green in four configurations |
| 2 | `refactor: long indices, actions and seeds` | the §5.2 table; `Discrete.ToInt64`; `LegacySeeding.RandomState`; `DiscreteTests`, `SeedTests`, `CartPoleTests` | 0 errors; new tests green |
| 3 | `fix(rng): per-instance entropy-seeded generators` | `LegacySeeding.FromEntropy`; every `np.random` fallback and CartPole's `TickCount` one replaced; `legacy_rng.py` and its golden; `LegacyRngParityTests`, `ConcurrencyTests`, `EntropySeedingTests` | the golden matches 602 of 602 |
| 4 | `refactor: NDArray ownership` | §8.3 | 0 NDW; all baselines unchanged |
| 5 | `docs: NumSharp migration` | `CLAUDE.md`, `AGENTS.md`, `GEMINI.md`, the development guide, the port plan's F1 status, release notes | none |
| 6 | examples | per decision D13 and M7 | none |
| later | `build(deps): consume the NumSharp package` | §4.4, once U1 ships | package mode green |

Work on a branch cut from PR #24's head, or from master once PR #24 is merged trimmed (port plan decision D11).

---

## 14. Upstream NumSharp items

The maintainer owns NumSharp. These items would remove the workarounds above:

| # | Item | Evidence | Gym.NET workaround until then |
|---|---|---|---|
| U1 | Release a NumSharp that contains Stage B | NuGet 0.70.0 fails with 4 errors (§3.2) | submodule |
| U2 | Seed an unseeded `RandomState()`/`NumPyRandom()` from OS entropy (NumPy: `SeedSequence()`), not `Environment.TickCount` | 1,000 of 1,000 identical pairs | `LegacySeeding.FromEntropy` |
| U3 | Add `RandomState(long)` or `RandomState(uint)`: NumPy accepts `0` to `2**32 - 1` | only `RandomState(int)` exists | `LegacySeeding.RandomState(long)` |
| U4 | Give `NumPyRandom` a per-instance lock like NumPy's `RandomState.lock`; streams are unaffected | two threads crash with `IndexOutOfRangeException` | per-instance generators |
| U5 | Make `randint` default to int64, NumPy 2's `int_` | defaults to int32 | explicit `dtype` |
| U6 | Add NumPy-faithful `uniform`/`normal`/`exponential(low, high, size)` with array parameters, computed in float64 with broadcasting, for both the legacy API and `Generator` (the port plan's F2 item 2) | `uniform(NDArray, NDArray, DType)` computes in the operand dtype | explicit formula |
| U7 | Make `NumPyRandom.Seed` hold the full seed; it is an `int` and wraps at 2^31 | source read | do not read it |
| U8 | Lower the cost of small arrays reclaimed by the finalizer | 0.634 µs undisposed vs 0.217 µs disposed | disposable results (M5) |
| U9 | Clear the warnings NumSharp.Core prints when built as a project reference | about 1,260, 1,108 of them CS8632 | ignore them |

---

## 15. Decisions for the maintainer

| # | Decision | Options | Recommendation |
|---|---|---|---|
| M1 | NumSharp source | submodule on master now; or wait for a release | Submodule now (§4); ship U1 before Gym.NET's next release |
| M2 | Discrete element type | int64/`long` now; or at P2 | Now: it is the NumSharp index type and Gymnasium's dtype. Keep the `(n)` shape until P2 |
| M3 | Seed type | `long` now; or wait for P1's `Reset(long? seed)` | Now: it costs one signature per type and unlocks seeds above 2^31−1 |
| M4 | Default generators | per-instance entropy-seeded; or the global `np.random` | Per-instance: thread-safe and Gymnasium's model. It changes behavior for users who call `np.random.seed` |
| M5 | Result ownership | `Step.Observation` `[NDBorrowed]` now. P1's `StepResult`/`ResetResult`: `INDArrayCarrier` or borrowed | Borrowed now. Carrier in P1 (§8.4): callers that dispose run faster than on Lite |
| M6 | NDW severity | warnings; errors in `src`; `.editorconfig` tuning | Errors in `src` once §8.3 lands; warnings in `tests` |
| M7 | Examples | migrate the 7 sites; or remove (port plan D13) | Follow D13: rewrite one CartPole example on the new API and remove the rest |

---

## Appendix A: Trial compile log

Against `refs/NumSharp` master with the unmodified Gym.NET code. Line numbers refer to the files at `48e095d`. Each error was reported once per TFM.

### A.1 `Gym` (13)

| Site | Code | Cause |
|---|---|---|
| `Spaces/Discrete.cs:11` | CS0266 | `DType` to `Type` (`dType ?? np.float32`) |
| `Spaces/Discrete.cs:18` | CS0266 | `NDArray<bool>` to `bool` (`mask != null`) |
| `Spaces/Box.cs:25` | CS0266 | `DType` to `Type` |
| `Spaces/Box.cs:35` | CS0266 | `DType` to `Type` |
| `Spaces/Box.cs:35` | CS0037 | `null` to `Shape` (`base(null, …)`) |
| `Spaces/Box.cs:47`, `:49` | CS1503 | `float` to `int[]`: `np.full`'s argument order |
| `Spaces/Box.cs:117` (×2) | CS1503 | `NDArray` to `double`: `uniform(Low[m], High[m], shape)` |
| `Spaces/Box.cs:128` | CS0218 | `&&` on `NDArray` (`x >= Low && x <= High`) |
| `Spaces/Box.cs:169` (×2) | CS0266 | `NDArray<bool>` to `bool` (`Low != null`, `High != null`) |
| `Observations/Step.cs:56` | CS0266 | `NDArray<bool>` to `bool` (`Observation != null`) |

### A.2 `Gym.Environments` (14)

All in `Envs/Aether/LunarLanderEnv.cs`:

| Site | Code | Cause |
|---|---|---|
| `:409`, `:410` | CS0266 | `NDArray` to `int` (`randint`) |
| `:496` (×2) | CS1503 | `NDArray` to `float` (`uniform` into `Vector2`) |
| `:507` | CS0266 | `NDArray` to `float` (terrain heights) |
| `:611`, `:612` | CS0266 | `NDArray` to `float` (dispersion) |
| `:618` | CS0266 | `NDArray<bool>` to `bool` (`c_action[0] > 0f`) |
| `:620` | CS0266 | `NDArray<bool>` to `bool` (`np.abs(c_action[1]) > 0.5f`) |
| `:640` | CS0266 | `NDArray` to `float` (`m_power`) |
| `:683` | CS0266 | `NDArray<bool>` to `bool` (direction) |
| `:684` | CS0266 | `NDArray` to `float` (`s_power`) |
| `:749`, `:750` | CS0266 | `NDArray` to `float` (shaping) |

`CartPoleEnv.cs` compiles unchanged. The `long` seed change of step 2 adds `CartPoleEnv.cs:197` (CS1503, `RandomState` takes `int`) and four override errors (CS0534/CS0115) that the same step fixes.

### A.3 Tests (1)

`Spaces/BoxTest.cs:40`: CS0266, `NDArray` to `float`.

### A.4 Against NuGet 0.70.0 (4, after the repair)

`Box.cs`: three CS0121 (`astype(Type, bool)` vs `astype(NPTypeCode, bool)`). `Discrete.cs`: one CS1061 (`Type` has no `kind`).

---

## Appendix B: NumSharp API map for every call Gym.NET makes

| Gym.NET call (Lite) | NumSharp master | Note |
|---|---|---|
| `np.float32`, `np.int32`, … as `Type` | the same names as `DType` | `DType` to `Type` is explicit; `Type`, `NPTypeCode` and strings convert to `DType` implicitly |
| `Type dType` parameters | `DType dtype` | |
| `np.zeros(shape, dtype)` | `np.zeros(Shape, DType)` | |
| `np.full(value, shape)` | `np.full(shape, value, dtype)` | order flipped |
| `nd.shape` (`int[]`), `nd.size` (`int`) | `long[]`, `long` | |
| `new Shape(int)` | `new Shape(params long[])`; implicit from `int`, `long`, `int[]`, `long[]` | |
| implicit `NDArray` to scalar | explicit, 0-d only | `(double)nd`, `(long)nd` |
| comparison used as `bool` | `NDArray<bool>` | `np.all`/`np.any`, or `(bool)` on 0-d |
| `a != null` | `a is not null` | `!=` is element-wise |
| `np.any(a)`, `np.all(a)` | return `bool` | 0-d supported |
| `np.random.RandomState(int)` | same, MT19937 | streams equal NumPy's legacy `RandomState` |
| `RandomState.seed(int)` | `seed(int)`, `(uint)`, `(long)`, `(ulong)`, `(uint[])` | range-checked |
| `uniform(low, high)` | returns a 0-d `NDArray` | `(double)` |
| `uniform(low, high, size)` | `uniform(double, double, Shape)` | |
| `uniform(NDArray, NDArray, size)` | gone; `uniform(NDArray, NDArray, DType)` differs | §6.3 formula |
| `normal(loc, scale, size)` | `normal(double, double, Shape)` | |
| `exponential(scale, size)` | `exponential(double, Shape)` | |
| `randint(low, high, size)` | 0-d `NDArray`, int32 by default | pass `dtype` |
| `choice(int)` | `choice(long a, Shape size = default, …)`, `choice(int a, …)`, `choice(NDArray a, …)` | returns a 0-d `NDArray` for the default size |
| `np.nonzero(a)` | `NDArray<long>[]` | |
| `astype(NPTypeCode.Float)` | `astype(DType)`; `NPTypeCode.Float` is still an alias of `Single` | |
| `np.clip(a, 0f, 1f)` | float32 stays float32 for C# literals (NEP 50) | |
| `np.@byte` (uint8) | int8 | use `np.uint8` |
| `NDArray.GetHashCode()` | throws | |
| `NDArray.Equals(object)` | element-wise with broadcasting | |
| `NDArray.ToString()` | NumPy's `array_str` | |

---

## Appendix C: Measurements

### C.1 Tests

| Build | Result |
|---|---|
| Lite (baseline) | 6 of 6 headless tests pass (Debug) |
| master, compile repairs only | 5 of 6: the consistency check expects 245 steps and gets 230 |
| master, full repair | 6 of 6 in Debug and Release on net8.0-windows and net10.0-windows (24 runs) |

### C.2 Throughput

Headless, no rendering, fixed action sequences, minimum of 5 repetitions per run. The host is unpinned hybrid-core hardware, so runs vary by up to 1.4×.

| Run | Lite CartPole (200k steps) | master CartPole | Lite LunarLander (20k steps) | master LunarLander |
|---|---:|---:|---:|---:|
| 1 | 0.373 µs/step | 0.776 µs/step | 87.63 µs/step | 68.94 µs/step |
| 2 | 0.374 | 0.890 | 84.01 | 70.60 |
| 3 | 0.335 | 0.722 | 116.79 | 67.32 |

CartPole's gap is the finalizer path of its per-step observation array (§8.4); LunarLander is dominated by physics.

### C.3 Allocation

See §8.4: `np.array(double[4])` costs 0.330 µs on Lite, 0.634 µs on master undisposed, and 0.217 µs on master disposed.

### C.4 Warnings

Gym.NET's own warning families are unchanged by the migration: CS0219, CS0618, SYSLIB0006, CS0162, CA1416, SYSLIB0003, CS0649, CS8073, CS0169, WFDEV004, CS0672, plus the NuGet vulnerability and packaging warnings. The migration adds the 23 NDW sites until step 4 removes them. A submodule build adds about 1,260 NumSharp-internal warnings (§4.3).

### C.5 NuGet 0.70.0

See §3.2 and A.4.

### C.6 Behavior probe

| Probe | Migrated result |
|---|---|
| `Discrete(4).Sample()` | `2`, int64, 0-d |
| `Discrete(3, start: 5).Contains(5 / 7 / 0)` | true / true / false |
| masked sample over `[0, 0, 1, 1]`, 200 draws | only 2 and 3 |
| `Box(-1, 1, (3,)).Sample()`, then `Contains` | float32 (3,) sample; `Contains` true |
| `Box(0, 5, (2000,), int32).Sample()` | min 0, max 4 (B-09 open) |
| CartPole `Step(ActionSpace.Sample())` | works |
| CartPole `ObservationSpace.Contains(Reset())` | true (float64 observation, float32 space) |
| `Seed(2^31 + 5)` | works |
| `Seed(2^32)` | `ValueError: Seed must be between 0 and 2**32 - 1` |
| continuous LunarLander `Step(ActionSpace.Sample())` | `IndexError` (B-20 open) |
| `Box((2,)) == Box((3,))` | false |
| raw `NDArray.GetHashCode()` | `NotSupportedException` |
| `(int)(object)discrete.Sample()` | `InvalidCastException` |
| global `np.random`, 2 threads × 2,000,000 draws | `IndexOutOfRangeException` |
| 1,000 pairs of unseeded `RandomState()` | 1,000 identical |
| `(long)` of a 0-d uint64 array holding `ulong.MaxValue` | −1 (C-style cast; the §6.4 snippet range-checks it) |
| `Discrete.ToInt64` over the §12.2 input matrix | integers and 0-d integer arrays convert; out-of-range uint64 throws `OverflowException`; double, 1-D array and null throw `ArgumentException` |

---

## Appendix D: Reproduction

### D.1 Trial setup

1. Export the code: `git archive HEAD src tests Gym.NET.sln | tar -x -C <scratch>`.
2. In the copy's `Gym.csproj` and `Gym.Environments.csproj`, replace the `NumSharp.Lite` reference with a `ProjectReference` to `refs/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj`.
3. Build `src/Gym`, then `src/Gym.Environments`, then the solution, fixing each layer before the next (Appendix A). Add the analyzer `ProjectReference` from §4.2 to see the NDW diagnostics.
4. For C.1, run the headless subset: `dotnet test tests/Gym.Tests -f <tfm> -c <config> --filter "FullyQualifiedName~NullEnv|FullyQualifiedName~BoxTest"`.
5. For §3.2, replace the `ProjectReference` with `<PackageReference Include="NumSharp" Version="0.70.0" />`.

### D.2 RNG golden recorder

This is the oracle behind §9.1 and `LegacyRngParityTests`; commit it as `tools/parity/legacy_rng.py`. The C# twin prints the same `<case> <index> <value>` lines through Gym.NET's code, and a `diff` of the two outputs (after normalizing line endings) must be empty.

```python
"""Record the legacy numpy.random.RandomState draw patterns Gym.NET's envs and spaces use, as raw bits.

Each line is "<case> <index> <value>" so a C# twin can be diffed line by line: float64 values print as 16 hex digits
of their IEEE bit pattern, float32 as 8, integers in decimal. Requires numpy==2.4.2 (the reference stack).
"""
import struct
import numpy as np

SEEDS = [0, 1, 42, 1000, 2**31 - 1, 2**31 + 5, 2**32 - 1]
# LunarLanderEnv computes VIEWPORT_H / SCALE / 2 in float32 before widening it to double.
H_HALF = float(np.float32(np.float32(400.0) / np.float32(30.0)) / np.float32(2.0))


def f64(x):
    return struct.pack(">d", float(x)).hex()


def f32(x):
    return struct.pack(">f", np.float32(x)).hex()


def emit(case, values, fmt):
    for i, v in enumerate(values):
        print(f"{case} {i} {fmt(v)}")


for seed in SEEDS:
    rs = np.random.RandomState(seed)  # LunarLander: constructor, reset, zero step, 20 steps
    emit(f"lunar.wind.s{seed}", [rs.randint(-9999, 9999), rs.randint(-9999, 9999)], str)
    emit(f"lunar.force.s{seed}", [rs.uniform(-1000.0, 1000.0), rs.uniform(-1000.0, 1000.0)], f64)
    emit(f"lunar.height.s{seed}", [rs.uniform(0.0, H_HALF) for _ in range(12)], f64)
    emit(f"lunar.disp.s{seed}", [rs.uniform(-1.0, 1.0) for _ in range(42)], f64)

    rs = np.random.RandomState(seed)  # CartPole reset
    emit(f"cartpole.reset.s{seed}", rs.uniform(-0.05, 0.05, 4), f64)

    rs = np.random.RandomState(seed)  # Discrete(4): 10 samples, then 5 masked samples over [0, 2, 3]
    emit(f"discrete.sample.s{seed}", [rs.randint(0, 4) for _ in range(10)], str)
    emit(f"discrete.masked.s{seed}", [rs.choice(np.array([0, 2, 3])) for _ in range(5)], str)

    rs = np.random.RandomState(seed)  # Box(-1, 1, (3,), float32).Sample()
    low, high = np.full((3,), -1.0, np.float32), np.full((3,), 1.0, np.float32)
    emit(f"box.bounded.s{seed}", rs.uniform(low=low, high=high, size=(3,)).astype(np.float32), f32)

    rs = np.random.RandomState(seed)  # unbounded and half-bounded Box draws
    emit(f"box.normal.s{seed}", rs.normal(0.5, 1.0, (3,)), f64)
    emit(f"box.exponential.s{seed}", rs.exponential(1.0, (3,)), f64)
```

### D.3 Other probes

- **Baselines (§9.2):** replay `LunarLanderEnvironment.Run`'s PID loop with `NullEnvViewer` for seeds 0, 1, 42 and 1000, discrete and continuous. Print the steps and the reward's float32 bits. Run the script under `dotnet run -c Debug` and `-c Release`, with `TargetFramework` set to net8.0 and to net10.0.
- **Behavior probe (C.6):** one C# file-based app with one `try` block per row, printing the result or the exception type and message.
- **Throughput (C.2):** a file-based app compiled once against the Lite copy and once against the migrated copy. CartPole runs `Step(i % 2)` with a reset on done; LunarLander runs `Step((i / 10) % 4)` with `RandomState(0)`. Warm up first, then take the minimum of 5.
- **Allocation (§8.4):** 1,000,000 calls of `np.array(double[4])`, with and without `Dispose()`, minimum of 5.
- **Seeding (§9.4):** create 1,000 pairs of `np.random.RandomState()` back to back and compare their first draws; the NumPy counterpart is `np.random.RandomState().uniform() == np.random.RandomState().uniform()`.

P0 turns these probes into `tools/parity/` scripts, as the port plan's Appendix A already foresees for the gap-scan prototypes.
