# Gymnasium → .NET Porting Plan

| | |
|---|---|
| **Status** | Draft 1, 2026-09-23 |
| **Goal** | Gym.NET becomes a byte-perfect clone of `refs/Gymnasium`, plus its own additions |
| **Source of truth** | `refs/Gymnasium` @ `9e04324` (v1.3.0 + 75 commits, `__version__ = "1.4.0"` dev, 2026-08-29) |
| **Array engine** | `refs/NumSharp` master @ `93967990` (v0.70.0 + 46) |
| **Audited Gym.NET** | branch `pr/24` @ `2639785` (PR #24: .NET 8/10 upgrade) |
| **Contributor fork** | `HalfTick/Gym.NET` master @ `cb9dcb6` (unmerged; assessed in §18) |
| **Supersedes** | the porting direction in PR #24's `docs/PRD.md`, `docs/ROADMAP.md` and `docs/plans/expressive-nibbling-goose.md` |

Every number in this document was measured on 2026-09-23 against the revisions above, on Windows 11 x64 with .NET 10.0.1, Python 3.12.12, NumPy 2.4.2, pygame-ce 2.5.8 (SDL 2.32.10) and Box2D 2.3.10. Appendix A says how to reproduce each measurement.

---

## Contents

1. [Goal, scope and definition of done](#1-goal-scope-and-definition-of-done)
2. [Where Gym.NET stands](#2-where-gymnet-stands)
3. [Porting rules](#3-porting-rules)
4. [Target architecture](#4-target-architecture)
5. [Foundations workstream](#5-foundations-workstream)
6. [Core API](#6-core-api)
7. [Spaces](#7-spaces)
8. [Registration and make()](#8-registration-and-make)
9. [Wrappers](#9-wrappers)
10. [Vector environments](#10-vector-environments)
11. [Environments](#11-environments)
12. [Utilities, errors, logging, functional API](#12-utilities-errors-logging-functional-api)
13. [Rendering](#13-rendering)
14. [Verification strategy](#14-verification-strategy)
15. [Gym.NET additions and legacy API](#15-gymnet-additions-and-legacy-api)
16. [Migration for existing users](#16-migration-for-existing-users)
17. [Phased roadmap](#17-phased-roadmap)
18. [External contributions](#18-external-contributions)
19. [Risks and mitigations](#19-risks-and-mitigations)
20. [Decisions for the maintainer](#20-decisions-for-the-maintainer)
- [Appendix A: Gap-scan methods and results](#appendix-a-gap-scan-methods-and-results)
- [Appendix B: Defect register](#appendix-b-defect-register)
- [Appendix C: Gymnasium surface map](#appendix-c-gymnasium-surface-map)
- [Appendix D: NumPy and NumSharp coverage](#appendix-d-numpy-and-numsharp-coverage)
- [Appendix E: Rasterizer scope (pygame APIs)](#appendix-e-rasterizer-scope-pygame-apis)
- [Appendix F: Python → C# semantics cheat-sheet](#appendix-f-python--c-semantics-cheat-sheet)
- [Appendix G: Assets and licenses](#appendix-g-assets-and-licenses)

---

## 1. Goal, scope and definition of done

### 1.1 Identity

Gym.NET is a byte-perfect .NET clone of Farama Gymnasium, as pinned in `refs/Gymnasium`, with additions that Gymnasium doesn't have (typed .NET APIs, desktop viewers, async stepping). Where Gym.NET and Gymnasium disagree, Gymnasium is right, unless this plan records a deliberate, documented deviation.

It stays Gym.NET. The product name, the NuGet ids (`Gym.NET.*`) and the root namespace (`Gym`) don't change, and today's namespaces keep every type that survives the port (decision D1, §3.8). Parity covers what the code does and how its API is shaped, not package or namespace names.

### 1.2 Parity levels

Every ported component must reach a defined parity level. "Byte-perfect" is not a slogan here; it's a test oracle.

| Level | Definition | Oracle |
|---|---|---|
| **L1 API** | Same components, same semantics, same argument validation, same error types; names follow the mapping in §3.8 | Reflection inventory vs Gymnasium inventory (§14.6); ported unit tests |
| **L2 Behavioral** | Same results up to documented tolerances | Ported Gymnasium tests (§14.2) |
| **L3 Byte** | For identical inputs, seeds and action sequences: bit-identical observations, rewards (float64), `terminated`/`truncated`, deterministic `info` entries, space samples and RNG state | Golden trajectories recorded from Python (§14.3), compared bitwise |
| **L4 Pixel** | `render()` in `rgb_array` mode returns byte-identical `(H, W, 3) uint8` frames | Golden frames (§14.4) |

Targets per area:

| Area | Target |
|---|---|
| Core, registration, errors, logger, utils, functional protocol | L1 + L2 |
| Spaces (sampling, seeding, flatten) | L3 |
| Wrappers, vector envs | L3 where outputs are deterministic (time-based `info` excluded) |
| classic_control, toy_text | L3 dynamics, L4 frames |
| box2d | L3 dynamics (requires §5.6), L4 frames |
| mujoco | L3 dynamics with the pinned native MuJoCo build; frames best-effort (GPU/driver-dependent) |

### 1.3 Reference stack

Golden data is recorded once, from one pinned stack, and every CI run compares against it. Recommended reference (see decision D2):

| Component | Pin |
|---|---|
| Gymnasium | `refs/Gymnasium` @ `9e04324` |
| Python | 3.12.x |
| NumPy | 2.4.2 (NumSharp's `Generator` is verified byte-exact against it, Appendix A.6) |
| pygame-ce | 2.5.8 (SDL 2.32.10) |
| Box2D | `box2d==2.3.10` |
| MuJoCo | to be pinned in P9 (decision D6) |
| OS / CPU | Windows 11 x64 |

Why Windows x64: NumPy on Windows has no `np.float128` (its `longdouble` is float64), and Gymnasium's own `rescale_box` (`gymnasium/wrappers/utils.py:229`) switches to 80-bit arithmetic when `float128` exists. On Linux x86-64, Gymnasium's `RescaleAction`/`RescaleObservation` therefore compute differently from Windows. .NET has no 80-bit type, so only the Windows code path is reproducible. In addition, a float64 transcription of CartPole using .NET `Math.Cos/Sin` was bit-identical to Gymnasium for 4 × 500 steps on this platform (Appendix A.7). Linux runs as a tracked secondary platform (§14.9).

### 1.4 Scope

In scope: everything in Appendix C except the items below.

Out of scope (Gymnasium features tied to Python-only stacks):

- JAX and PyTorch: `phys2d/*`, `tabular/*`, `FunctionalJaxEnv`, `FunctionalJaxVectorEnv`, `ArrayConversion`, `JaxToNumpy`, `JaxToTorch`, `NumpyToTorch` (single and vector variants).
- Python plumbing: `EzPickle`, `CloudpickleWrapper`, `clear_mpi_env_vars`.
- Shimmy shims `GymV21Environment-v0` and `GymV26Environment-v0`, and the 17 error-only `mujoco-py` ids (`-v2`/`-v3`).

Optional (decide later): `AtariPreprocessing` requires ALE (`ale_py`) and ROMs; it can be ported against any ALE binding that exposes the same frames.

### 1.5 Definition of done (for the whole port)

1. Every in-scope Gymnasium public name has a .NET counterpart (reflection inventory test passes).
2. Gymnasium's test suite for in-scope areas is ported and green.
3. Golden trajectories: every in-scope registered env id passes bitwise comparison for the golden seed and action set on the reference platform.
4. Golden frames: classic_control, toy_text and box2d `rgb_array` frames are byte-identical.
5. No code path in `src/` uses a legacy pre-0.26 contract except the opt-in `Legacy` shim (§16).

---

## 2. Where Gym.NET stands

### 2.1 Headline

Of **258** public Gymnasium names in scope for mapping (Appendix C), Gym.NET on `pr/24` has:

| Status | Names | Meaning |
|---|---:|---|
| Matches | 3 | Same contract: `Env.action_space`, `Env.observation_space` (properties only), context-manager close via `IDisposable` |
| Diverges | 12 | Exists but is shaped or behaves differently |
| Broken | 4 | Exists but crashes or misbehaves by its own intent: `Box`, `Discrete`, `LunarLander-v3`, `LunarLanderContinuous-v3` |
| Missing | 218 | No counterpart |
| N/A | 21 | Python-only stacks (§1.4) |

No environment, space or wrapper is bit-identical. What exists (the core contract, `Box`, `Discrete`, CartPole, LunarLander) is a port of pre-0.26 OpenAI Gym, not of Gymnasium.

### 2.2 Coverage by area

| Area | Total | Has (M/D/B) | Missing | N/A | New in fork |
|---|---:|---:|---:|---:|---:|
| Core API (`gymnasium.core`) | 20 | 9 (3/6/0) | 11 | 0 | 4 |
| Spaces | 17 | 3 (0/1/2) | 14 | 0 | 9 |
| Registration and `make()` | 16 | 0 | 16 | 0 | 0 |
| Environments | 57 | 4 (0/2/2) | 45 | 8 | 4 |
| Wrappers | 42 | 0 | 38 | 4 | 8 |
| Vector | 19 | 0 | 17 | 2 | 3 |
| Vector wrappers | 26 | 0 | 22 | 4 | 0 |
| Utilities | 35 | 0 | 34 | 1 | 0 |
| Errors and logging | 23 | 3 (0/3/0) | 20 | 0 | 0 |
| Functional API | 3 | 0 | 1 | 2 | 0 |
| **All** | **258** | **19 (3/12/4)** | **218** | **21** | **28** |

Counting rule: one per public name (class, function or registered id); env-module helper functions count once per module; the 17 error-only mujoco-py ids count once.

### 2.3 What the gap scan found

Twelve techniques were applied (Appendix A). The findings that shape this plan:

1. **The spaces are unusable for real shapes.** On `pr/24`, `Discrete.Sample()` throws `NullReferenceException` even without a mask (`src/Gym/Spaces/Discrete.cs:18`, `mask != null` goes through NDArray's overloaded operator). `Box.Sample()` and `Box.Contains()` throw `IncorrectShapeException` for any box that isn't 0-d (`Box.cs:96`, `Box.cs:128`: `~BoundedLow` and `x >= Low && …` implicitly convert an array to a scalar). Integer boxes can't be constructed (`Box.cs:48`). Consequences: `CartPoleEnv.ActionSpace.Sample()` throws, and CartPole's `ObservationSpace.Contains(obs)` throws on its own observations. The existing tests pass only because they exercise a 0-d box.
2. **CartPole diverges at step 1.** Replaying Gymnasium's actions from Gymnasium's initial states, Gym.NET's state differs at step 1 on all 4 seeds, reaches |Δstate| ≈ 27 by step 500, and terminates at steps 247–254 where Gymnasium never terminates. Root cause: float32 constants (`tau = 0.02f` is 0.019999999552965164 as a double; `x_threshold = 2.4f` moves the termination boundary to 2.4000000953674316).
3. **Byte-perfect is achievable in .NET.** A float64 transcription of Gymnasium's `cartpole.py` step, using `Math.Cos/Math.Sin`, is **bit-identical for all 4 × 500 steps**.
4. **NumSharp's RNG is byte-exact.** `np.random.default_rng(seed)` in NumSharp reproduced NumPy 2.4.2 for 16 of the 17 draw patterns Gymnasium uses (80 of 85 results over 5 seeds). The 17th, `Generator.geometric` (used by `spaces.Sequence`), doesn't exist on NumSharp's `Generator`.
5. **NumPy API coverage is essentially complete.** Of the 93 applicable `np.*` symbols Gymnasium uses, 86 resolve directly, 5 resolve through NumSharp spellings (`np.round` → `np.round_`, abstract dtypes → `np.issubdtype(dtype, "integer")` etc., verified equal to NumPy), and 2 are real gaps: the `'c'` byte dtype (`np.bytes_`, FrozenLake/Taxi maps) and `np.float128` (absent on Windows NumPy too). All 24 ndarray method names used exist on `NDArray`.
6. **Aether is not Box2D.** An identical scene (lander hull, push, fall onto an edge) diverges from Box2D 2.3.10 **at step 1, even in free fall**, and ends 1.75 m apart after 300 steps. Byte-perfect Box2D environments need a Box2D 2.3.x port (§5.6).
7. **Rendering is close but not pixel-exact.** CartPole frames differ from pygame-ce in 1.07–1.14% of pixels (max channel delta 255): wrong colours, different rasterizer.
8. **LunarLander is broken beyond parity.** 169 of 1,770 random-play observations (9.5%) fall outside its own (v2) observation space. Envs without an explicit `random_state` share the global `np.random`, so a second env changes the first one's trajectory. Continuous mode crashes on `Step(ActionSpace.Sample())` (`OverflowException`, 0-d action space). `wind_power: 25` throws where Gymnasium only warns.
9. **Precision rules are subtle.** Under NumPy 2 promotion (NEP 50), `0.001 * (u**2)` with a float32 `u` stays float32 (0.00169 vs 0.0016899998760223412 in float64). Pendulum's reward mixes float32 and float64 terms this way. Ports must mirror dtype per expression (§3.2).
10. **The contributor fork repeats the root causes.** Static rules find 83 float32 constants in the fork (40 on `pr/24`); its `SyncVectorEnv` writes `info["final_observation"]`, which Gymnasium removed in 1.0.
11. **Repository health.** The examples don't build (`NU1201`: netcoreapp3.0/netstandard2.0 vs net8/10). There is no CI and no nullable annotations. The build has 16 warning codes, including `SYSLIB0006` (`Thread.Abort`, throws on .NET 5+), `CA1416` and `CS8073` (a dead `!= null` check on a struct).

Appendix B lists every verified defect with file and line.

### 2.4 Consequence

Patching the existing code can't reach L3: the numeric types, RNG, spaces, contract, physics engine and rasterizer are all wrong at the root. The plan therefore **re-ports every component from Gymnasium source**, keeping only the pieces listed in §15 as Gym.NET additions.

---

## 3. Porting rules

These rules apply to every ported file. They exist because each one was violated somewhere in the current code or the fork.

### 3.1 Source fidelity

1. **One C# file per Python module.** `gymnasium/spaces/box.py` → `src/Gym/Spaces/Box.cs`. Keep definition order, helper functions and branch structure. A reviewer should be able to read both files side by side.
2. **Cite the source.** Every ported type and member carries `<remarks>Port of <c>gymnasium/spaces/box.py:64</c> @ 9e04324.</remarks>`. When Gymnasium is re-pinned, the citations make a diff-driven update possible (§14.10).
3. **Port behavior, including quirks.** Python's `height[-1]` wraps; `min`/`max` compare the way Python compares; asserts raise. Don't fix upstream behavior. If upstream is buggy, port the bug and file it upstream; record the fix as an explicit, flagged deviation only if the maintainer approves.
4. **No restructuring for style.** PR #24's PRD mandates Object Calisthenics (no `else`, methods of at most 15 lines, one dot per line). That rule is rejected for ported code: it makes line-by-line verification against the source impossible. It may apply to Gym.NET-only additions.

### 3.2 Numeric rules

| Python construct | C# port |
|---|---|
| Python `float` scalar | `double` |
| NumPy array with `dtype=np.float32` | `NDArray` of `float32`; cast at exactly the points the source casts |
| `np.float32(x)` / `.astype(np.float32)` | `(float)x` / `.astype(np.float32)` at that point only |
| Python scalar mixed with a NumPy float32 scalar (NEP 50: Python scalars are weak) | Result stays float32. Example: Pendulum's `0.001 * (u**2)` |
| Python `int` | `long` (checked arithmetic where the source can exceed 64 bits) |
| `a / b` | true division in `double` |
| `a // b` | floor division: `Math.Floor(a / b)` for floats, `PyMath.FloorDiv` for integers (C# `/` truncates toward zero) |
| `a % b` | `PyMath.Mod` (result takes the divisor's sign; C# `%` takes the dividend's) |
| `x ** 2` | Mirror the source's function: `np.square(x)` → `x * x`; `x ** 2` → `Math.Pow(x, 2)` unless proven equal on the reference platform |
| `round(x)`, `np.round` | `Math.Round(x, MidpointRounding.ToEven)` / `np.round_` (both half-to-even, verified) |
| `min(a, b)`, `max(a, b)` | `PyMath.Min/Max`, which reproduce Python's comparison order for NaN. `Math.Min/Max` propagate NaN differently |
| `math.sin`, `np.sin`, … | `Math.Sin`, … (bit-identical on the reference platform, Appendix A.7). Keep track of which of the two the source calls |
| `np.clip` | `np.clip` (NumSharp) or `PyMath.Clip` with NumPy's NaN semantics |

**Banned in ported math:** `float` literals (`0.02f`), `MathF`, `System.Numerics.Vector*`, `Math.FusedMultiplyAdd`, and any reassociation of the source's expression order. Enforced by an analyzer (§14.7).

### 3.3 Randomness

1. Each env owns `NpRandom` (a NumSharp `Generator`); each space owns its own. **No global RNG**, and no entropy-seeded fallback except where Gymnasium seeds from entropy (`seeding.np_random(None)`).
2. Every draw mirrors the source exactly: the same method, arguments, `size`, `dtype` and order. Reordering two draws changes every later value; that is defect B-17 in LunarLander.
3. NumSharp's `Generator.uniform/normal/exponential` take scalar parameters only. Where Gymnasium passes arrays (`Box.sample` bounded dims, `Pendulum.reset`), use `low + (high - low) * rng.random(size)` element-wise. That is NumPy's own computation and was verified bit-identical (Appendix A.6). Replace it with a NumSharp overload once one exists (§5.2).

### 3.4 Errors and warnings

- `assert` → throw `AssertionError`, not `Debug.Assert`, which vanishes in Release builds. Gymnasium tests assert `AssertionError` in places. NumSharp has no `AssertionError` today, so it lives in `Gym.Exceptions` (`AssertionError : Exception`) unless NumSharp adds it first (D17).
- `raise error.X` → throw the ported `Gym.Exceptions.X` (Appendix C.9). `Gym.Exceptions` is today's exceptions namespace (§3.8).
- Python built-in exceptions raised by the source (`ValueError`, `TypeError`, `KeyError`, `NotImplementedError`, …) map to distinct types, never to one shared .NET exception. Gymnasium's tests tell `ValueError` and `TypeError` apart, so collapsing both into `ArgumentException` would lose that. Gym.NET reuses NumSharp's ports of Python's built-ins (decision D17). Namespace `NumSharp` already has `ValueError : ArgumentException`, `TypeError`, `KeyError`, `IndexError`, `AttributeError`, `RuntimeError` and `NameError`, and in Python, NumPy and Gymnasium raise the same built-in classes. Built-ins that NumSharp lacks (`AssertionError`, `NotImplementedError`, …) go into `Gym.Exceptions`, each deriving from the closest .NET exception, unless NumSharp adds them first. Never define a second copy of a type NumSharp has: a second `ValueError` in `Gym.Exceptions` would be an ambiguous reference (CS0104, checked with .NET SDK 10.0.101) in every file that imports both namespaces, which is nearly every ported file. Messages are ported verbatim, because tests match on them.
- `logger.warn` → `Logger.Warn`, which reproduces Python's default warning filter (once per call site), the `WARN: ` prefix and colour. Never `Console.WriteLine`, and never throw where Gymnasium only warns.

### 3.5 Collections and types

| Python | C# |
|---|---|
| `dict` (insertion-ordered) | `Info` / `OrderedDictionary<string, object?>`; never `Dictionary<,>` when order is observable |
| `tuple` | positional `record struct` or `ValueTuple` |
| `list` | `List<T>` |
| `None` | `null` with nullable annotations |
| `dataclass` (`EnvSpec`) | `sealed record` |
| `Enum` (`AutoresetMode`) | `enum` plus string conversion for the values Gymnasium stores in metadata |

### 3.6 Constants

Module constants become `const double`/`const int`, never public mutable fields (defect B-28). Instance attributes Gymnasium sets in `__init__` (`self.gravity = 9.8`) become `public double Gravity { get; }`, so users can read them the way they read Python attributes.

### 3.7 Code and documentation standards

- XML documentation on every public and internal member: a summary that states consequence or trade-off rather than restating the name; `<param>`, `<returns>`, `<typeparam>`, `<exception>` for everything; `<remarks>` with the source citation and any behavioural consequence. Inline comments explain *why* on non-obvious bodies (validation, dispatch, ownership, precision).
- `<Nullable>enable</Nullable>`, `<TreatWarningsAsErrors>true</TreatWarningsAsErrors>` for new projects, `<Deterministic>true</Deterministic>`.
- Target `net8.0;net10.0`, the same as NumSharp.

### 3.8 Naming

Gym.NET keeps its own names (decision D1). The product, the repository and the NuGet ids stay `Gym.NET` / `Gym.NET.*`, and the root namespace stays `Gym`. Every namespace that exists today keeps the types that survive the port, so existing `using Gym.Envs;` and `using Gym.Spaces;` directives still compile. Gymnasium modules without a home yet get namespaces in the same layout:

| Gymnasium module | Gym.NET namespace | Today |
|---|---|---|
| `gymnasium.core`: `Env`, `Wrapper`, `ObservationWrapper`, `ActionWrapper`, `RewardWrapper`; plus `IEnv`, `Info`, `Metadata`, `RenderResult` | `Gym.Envs` | exists: `Env`, `IEnv` |
| the `reset`/`step` return tuples: `ResetResult`, `StepResult` | `Gym.Observations` | exists: `Step` |
| `gymnasium.spaces` | `Gym.Spaces` | exists: `Space`, `Box`, `Discrete` |
| `gymnasium.error` | `Gym.Exceptions`; Python's built-in exceptions come from `NumSharp` (§3.4) | exists: 3 exception types |
| `gymnasium.logger` | `Gym`, as the static class `Logger` | new |
| `gymnasium.envs.registration` | `Gym.Envs.Registration`; the top-level registration API sits on the static class `Env` (see below) | new |
| `gymnasium.envs.classic_control` | `Gym.Environments.Envs.Classic` | exists: `CartPoleEnv` |
| `gymnasium.envs.toy_text` | `Gym.Environments.Envs.ToyText` | new |
| `gymnasium.envs.box2d` | `Gym.Environments.Envs.Box2D` | new; replaces `Gym.Environments.Envs.Aether` |
| `gymnasium.envs.mujoco` | `Gym.Environments.Envs.MuJoCo` | new |
| `gymnasium.wrappers` | `Gym.Wrappers` | new |
| `gymnasium.wrappers.vector` | `Gym.Wrappers.Vector` | new |
| `gymnasium.vector`, including `vector.utils` | `Gym.Vector` | new |
| `gymnasium.utils`, including `seeding`; plus `PyMath` (F3) | `Gym.Utils` | new |
| `gymnasium.experimental.functional` | `Gym.Experimental.Functional` | new |
| pygame's drawing, replaced by the F5 rasterizer | `Gym.Rendering` | new, as the parent of the viewer namespaces |
| human-mode viewers and their contract | `Gym.Rendering.Avalonia`, `Gym.Rendering.WinForm`; `IEnvViewer` in `Gym.Environments` | exist |
| the pre-0.26 shim (§16) | `Gym.Legacy` | new |

The env families keep today's `Gym.Environments.Envs.<Family>` pattern, and `Gym.Envs` keeps holding the core contract. So `Gym.Envs` doesn't mean what `gymnasium.envs` means: in Gymnasium that package holds the registry and the env families.

- **No type is named `Gym`.** A static class `Gym` inside namespace `Gym` can't provide `Gym.Make(…)`:
  - in caller code the name `Gym` binds to the namespace, so `Gym.Make` fails with CS0234;
  - inside any `Gym.*` namespace the type hides the namespace, so a `using Gym.Envs;` there fails with CS0426.

  Both were checked with .NET SDK 10.0.101. Gymnasium's top-level registration API (`make`, `make_vec`, `spec`, `register`, `registry`, `pprint_registry`, `register_envs`) therefore sits on a non-generic static class `Env` next to `Env<TObs, TAct>`: `Env.Make("CartPole-v1")`. That is the BCL's `Tuple`/`Tuple<T1>` pattern, and it compiles from caller code and from inside an env subclass (decision D16).
- Class names are kept **exactly** when they're valid C#, including `Continuous_MountainCarEnv`, so users can find them from Gymnasium docs.
- Methods and properties use PascalCase with a 1:1 table: `step` → `Step`, `reset` → `Reset`, `np_random` → `NpRandom`, `np_random_seed` → `NpRandomSeed`, `render_mode` → `RenderMode`, `action_space` → `ActionSpace`, `unwrapped` → `Unwrapped`, `get_wrapper_attr` → `GetWrapperAttr`.
- **Data keeps Python spelling byte-for-byte**: env ids, `info` keys (`"episode"`, `"final_obs"`), metadata keys (`"render_modes"`, `"render_fps"`, `"autoreset_mode"`), kwargs names in `EnvSpec`, error messages.

---

## 4. Target architecture

### 4.1 Projects

Project and package names stay in Gym.NET's existing families (decision D1). The four existing projects keep their names and NuGet ids; new projects extend the same prefixes.

| Project (package) | Contents | Depends on |
|---|---|---|
| `Gym` (`Gym.NET`), existing | core, spaces, registration, wrappers, vector, utils, errors, logger, experimental.functional; the viewer contract `IEnvViewer`, moved here from the `Gym.Environments` assembly while keeping its namespace `Gym.Environments` | NumSharp |
| `Gym.Rendering` (`Gym.NET.Rendering`) | pygame-compatible `Surface` rasterizer (§5.5) | `Gym` |
| `Gym.Environments` (`Gym.NET.Environments`), existing | classic_control: CartPole, MountainCar, Continuous_MountainCar, Pendulum, Acrobot, `CartPoleVectorEnv` | `Gym`, `Gym.Rendering` |
| `Gym.Environments.ToyText` (`Gym.NET.Environments.ToyText`) | Blackjack, FrozenLake, CliffWalking, Taxi, assets | `Gym`, `Gym.Rendering` |
| `Box2D.NET` | exact Box2D 2.3.x port (§5.6); could live in its own repo | — |
| `Gym.Environments.Box2D` (`Gym.NET.Environments.Box2D`) | LunarLander, BipedalWalker, CarRacing | `Gym`, `Gym.Rendering`, `Box2D.NET` |
| `MuJoCo.NET` | P/Invoke bindings for the pinned MuJoCo (§5.7) | native mujoco |
| `Gym.Environments.MuJoCo` (`Gym.NET.Environments.MuJoCo`) | `MujocoEnv`, `MujocoRenderer`, 11 envs × v4/v5, XML assets | `Gym`, `MuJoCo.NET` |
| `Gym.Rendering.Avalonia`, `Gym.Rendering.WinForm` (`Gym.NET.Rendering.Avalonia`, `Gym.NET.Rendering.WinForm`), existing | human-mode windows (Gym.NET additions, §15) | `Gym` |
| `Gym.Legacy` (`Gym.NET.Legacy`) | pre-0.26 compatibility shim (§16) | `Gym` |
| `tests/Gym.Tests`, existing | ported pytest suites, golden tests | all |
| `tests/Gym.Tests.Parity` | live interop parity suite ([`GYM_PARITY_INTEROP_TESTING.md`](GYM_PARITY_INTEROP_TESTING.md)) | all |
| `tools/golden` | Python recorder, manifest, golden data tooling | `refs/Gymnasium` venv |

Registration of the bundled envs happens in each env assembly (a module initializer or an explicit `RegisterEnvs()`), mirroring `gymnasium/envs/__init__.py`.

### 4.2 NumSharp consumption

Consume NumSharp from NuGet at a pinned version ≥ 0.70.0 for releases, and keep `refs/NumSharp` as a git submodule for development and agent reference (decision D3). 0.60.0 is not enough: `PCG64`, `SeedSequence` and `default_rng` first ship in 0.70.0 (commit `e868d8ae`).

NuGet 0.70.0 is not enough either. The `DType` API that F1 adopts ("Stage B", commits `c916c579` and `68a02258`) landed on NumSharp master after 0.70.0 was released, so Gym.NET builds `refs/NumSharp` from source until the next NumSharp release. [`NUMSHARP_MIGRATION_PLAN.md`](NUMSHARP_MIGRATION_PLAN.md) §3.2 and §4 give the measurements and the source/package switch.

### 4.3 Licensing

Gym.NET is Apache-2.0; ported code keeps Gymnasium's MIT notice. Add `THIRD-PARTY-NOTICES.md` listing:

- Gymnasium (MIT)
- Box2D (zlib)
- MuJoCo (Apache-2.0)
- OpenCV (Apache-2.0, `INTER_AREA` port)
- SDL_gfx (zlib, `gfxdraw` algorithms)
- pygame-ce (LGPL-2.1), if its `draw` code is ported (decision D4)
- the toy_text assets (Appendix G)

### 4.4 Fate of existing code

| Existing | Action |
|---|---|
| `Gym/Envs/IEnv.cs`, `Env.cs` | Replace with §6, still in `Gym.Envs`. The non-generic `Env` base class goes away; its name becomes the static registration class (D16), and subclasses move to `Env<TObs, TAct>` |
| `GoalEnv` | Delete; it now lives in Gymnasium-Robotics, and Gym.NET's version is a stub whose `Reset()` returns `null` |
| `IVecEnv`, `VecEnv`, `VecEnvWrapper`, `DummyVecEnv` | Delete; replaced by `gymnasium.vector` (§10) |
| `Observations/Step.cs` | Replace with `StepResult`/`ResetResult`, still in `Gym.Observations`; the legacy 4-tuple moves to `Gym.Legacy` |
| `Spaces/*` | Re-port from Gymnasium, still in `Gym.Spaces` |
| `Exceptions/*` | Replace with the ported `error.py` types, still in `Gym.Exceptions` |
| `Internal/Collections/Dict.cs` | Replace with `Info`; keep other collections only if a port needs them |
| `Internal/Dynamic`, `Merging`, `Reflection` | Delete or make internal; no Gymnasium counterpart |
| `Internal/Threading/*` | Delete (`Thread.Abort` family throws on .NET 5+); async stepping moves to §15 |
| `CartPoleEnv` | Re-port, still `Gym.Environments.Envs.Classic.CartPoleEnv` |
| `LunarLanderEnv` (Aether) | Re-port on Box2D.NET in `Gym.Environments.Envs.Box2D`; drop the Aether dependency and the `Gym.Environments.Envs.Aether` namespace |
| Viewers and `IEnvViewer` | Keep as human-mode backends (§13, §15), with their namespaces. `IEnvViewer` and `IEnvironmentViewerFactoryDelegate` move into the `Gym` assembly so that `HumanRendering` can use them and the viewer projects depend on `Gym` alone |
| `NullEnvViewer` | Obsolete: `render_mode=None` covers it |
| Tests | Replace with ported suites and goldens; keep viewer smoke tests |
| `examples/` | Rewrite on the new API or remove (decision D13) |
| PR #24 docs | Replace with docs derived from this plan; remove `docs/plans/expressive-nibbling-goose.md` |

---

## 5. Foundations workstream

Everything above depends on these. Each item lists scope, verification and exit criteria.

### 5.1 F1: NumSharp migration

- Replace `NumSharp.Lite 0.1.12` with NumSharp ≥ 0.70.0 in every project.
- Delete Lite workarounds: the `Any(...)` guards in `Box.Sample`, `!Equals(mask, null)`, the 0-d scalar path. They exist only because Lite's boolean masks crash (`AccessViolationException` on an all-false mask) and its operators convert arrays to scalars.
- Adopt NumSharp 0.70 semantics: `dtype` is `DType`; `np.float32` etc. are `DType` statics.
- Adopt NumSharp's `long` (int64) indexing wherever a value crosses into or out of NumSharp: Discrete elements, actions and seeds become `long`.
- **Detailed plan:** [`NUMSHARP_MIGRATION_PLAN.md`](NUMSHARP_MIGRATION_PLAN.md), measured by a trial migration of the whole repository. It covers every compile site, 11 runtime traps the compiler does not catch, the ownership analyzer's findings, the new RNG baselines and the commit order.
- **Exit:** `src/` builds against NumSharp master (`refs/NumSharp`, §4.2) and no Lite-specific code remains.

### 5.2 F2: Seeding and RNG

- Port `gymnasium/utils/seeding.py` as `Seeding.NpRandom(long? seed) → (Generator rng, long seed)`:
  - validate like the source (`error.Error` for negative seeds);
  - build `SeedSequence(seed)`, return `seed_seq.entropy`, construct `Generator(PCG64(seed_seq))`;
  - for `seed = null`, draw OS entropy through `SeedSequence()`.
- Env and space `NpRandom` are lazily seeded, with setters that record seed `-1`, all as in `core.py`/`space.py`.
- NumSharp upstream work (the maintainer owns both repos):
  1. `Generator.geometric(p, size)`, byte-exact with NumPy's `random_geometric`, needed by `spaces.Sequence`;
  2. array-parameter overloads for `uniform`, `normal` and `exponential` (NumPy broadcasting semantics);
  3. optionally the `'S1'`/`'c'` byte dtype, so FrozenLake/Taxi `desc` can be an `NDArray` rather than `byte[,]`;
  4. optionally the Python built-in exceptions that Gymnasium raises and NumSharp lacks (`AssertionError`, `NotImplementedError`), so that every built-in comes from NumSharp and Gym.NET defines none (D17).
- **Exit:** RNG stream tests (§14.5) pass for all draw patterns in Appendix A.6, including `geometric`.

### 5.3 F3: Python-semantics numerics

- `Gym.Utils.PyMath`: `FloorDiv`, `Mod`, `Min`, `Max` (Python comparison semantics), `Clip`, `Round`, `Sign`, `Float32(double)` (IEEE round-to-nearest-even).
- Analyzer (Roslyn, or a CI script at first) enforcing the bans in §3.2 in `Gym.Environments.Envs.*`, `Gym.Spaces`, `Gym.Wrappers` and `Gym.Vector`.
- **Exit:** analyzer in CI, zero violations in ported code.

### 5.4 F4: Golden harness

- `tools/golden/record.py` runs the pinned Gymnasium in a pinned venv and writes `tests/Golden/<suite>/<case>.jsonl.gz`. Floats are stored as IEEE bit patterns (int64 or int32 by dtype), with shape and dtype, per step. The prototype used for Appendix A.7 already records this way.
- `tests/Golden/manifest.json` records Gymnasium SHA, Python, NumPy, pygame-ce, Box2D and MuJoCo versions, OS and CPU. Tests refuse to compare against a manifest from a different stack.
- C# `GoldenTrajectory` loader and a `GoldenAssert` that reports the first diverging step and field, the ulp distance and both values.
- Two action modes per env:
  - **open-loop:** actions recorded from Python and replayed, which isolates dynamics;
  - **sampled:** `action_space.seed(s)` then `sample()` every step, which also tests space sampling.
- Frames: PNG (lossless) plus SHA-256 in the manifest.
- **Exit:** CartPole-v1 golden passes against the new port (the proof of concept already passes as a transcription).

### 5.5 F5: Rasterizer (`Gym.Rendering`)

- A `Surface` (RGB, `(W, H)` addressing like pygame) with exactly the operations in Appendix E:
  - `gfxdraw`: `aapolygon`, `filled_polygon`, `aacircle`, `filled_circle`, `hline`, `vline`;
  - `draw`: `polygon`, `line`, `lines`, `aaline`, `aalines`, `circle`, `rect`;
  - `transform`: `flip`, `scale`, `smoothscale`;
  - `image.load` (PNG);
  - `font` (FreeType);
  - `surfarray.pixels3d` returning `(W, H, 3)`, transposed by envs to `(H, W, 3)` as they do;
  - `math.Vector2.rotate_rad`, including pygame's special-casing of angles near multiples of 90°.
- `gfxdraw` is SDL_gfx (zlib): port freely. `draw` is pygame's own C code (LGPL-2.1); decision D4 chooses between an LGPL sub-package, a clean-room port verified only by pixel tests, or native SDL interop.
- Fonts (Blackjack's `Minecraft.ttf`, pygame's default font in CarRacing): pixel parity requires FreeType-identical rasterization; plan on FreeType interop (FreeType is FTL/GPLv2 dual-licensed).
- **Exit:** golden frames for all classic_control envs pass (L4).

### 5.6 F6: Box2D 2.3.x (`Box2D.NET`)

- Port the Box2D C++ core that `box2d==2.3.10` compiles, in float32, preserving operation order: `b2Settings`, `b2Math`, collision (broad-phase dynamic tree, pair ordering, distance, TOI, manifolds), dynamics (world, island, contact solver, joints: revolute, prismatic, weld, …), and the contact listener callback order.
- Expose the subset of pybox2d's API Gymnasium uses: `b2World`, bodies, fixtures (`fixtureDef`, `polygonShape`, `edgeShape`, `circleShape`), `revoluteJointDef`, `contactListener`, `ApplyForceToCenter`, `ApplyLinearImpulse`, `ApplyTorque`, `raycast` (BipedalWalker lidar), `awake`, `transform`, …
- Verification: engine-level goldens recorded from pybox2d (the Appendix A.10 scene is the first case), then per-env goldens.
- Fallback (decision D5): native Box2D 2.3.x via P/Invoke if the port stalls. That is byte-identical by construction but platform-bound.
- **Exit:** engine goldens pass, then LunarLander-v3 goldens (L3).

### 5.7 F7: MuJoCo interop (`MuJoCo.NET`)

- Generate P/Invoke bindings (e.g. with ClangSharp) from the pinned `mujoco.h`.
- Cover the model and data fields Gymnasium's envs read (`qpos`, `qvel`, `ctrl`, `cfrc_ext`, `xpos`, sensor data, …).
- Offscreen rendering via MuJoCo's own renderer on an EGL/OSMesa/WGL context.
- Ship the 14 XML models from `gymnasium/envs/mujoco/assets`.
- Pin the MuJoCo version for goldens (decision D6).
- **Exit:** Hopper-v5 golden passes (L3).

### 5.8 F8: OpenCV area resize

- Port `cv2.resize(…, INTER_AREA)` for the dtypes `ResizeObservation` and `AtariPreprocessing` use.
- **Exit:** byte-identical resize outputs against recorded `cv2` outputs.

### 5.9 F9: Video and human mode

- `RecordVideo`/`save_video` need an encoder (FFmpeg via a managed wrapper). Byte parity of video files is not a goal; frame parity is.
- Human mode reuses the existing Avalonia/WinForms viewers (§13).

---

## 6. Core API

### 6.1 Shape of the API

Gymnasium's `Env[ObsType, ActType]` becomes a generic `Env<TObs, TAct>` plus a non-generic `IEnv` for code that handles arbitrary envs (registry, wrappers, vector). Observation and action element types follow the spaces:

| Space | Element type |
|---|---|
| `Box`, `MultiDiscrete`, `MultiBinary` | `NDArray` |
| `Discrete` | `long` |
| `Text` | `string` |
| `Tuple` | `SpaceTuple` (immutable, positional) |
| `Dict` | `SpaceDict` (insertion-ordered) |
| `Sequence` | `SpaceSequence` or `NDArray` when `stack=True` |
| `Graph` | `GraphInstance` |
| `OneOf` | `(long Index, object Value)` |

### 6.2 Result types

```csharp
using Gym.Envs;

namespace Gym.Observations;

/// <summary>
/// The first observation of an episode and its diagnostics, as returned by <c>reset</c>.
/// Port of the <c>(obs, info)</c> tuple returned by <c>gymnasium.Env.reset</c> (gymnasium/core.py:25 @ 9e04324).
/// </summary>
/// <typeparam name="TObs">Observation type; always an element of the env's observation space.</typeparam>
/// <param name="Observation">The initial observation.</param>
/// <param name="Info">Auxiliary diagnostics. Insertion order is observable (Python dict semantics).</param>
public readonly record struct ResetResult<TObs>(TObs Observation, Info Info);

/// <summary>
/// One environment transition. Port of the <c>(obs, reward, terminated, truncated, info)</c> tuple.
/// </summary>
/// <typeparam name="TObs">Observation type; always an element of the env's observation space.</typeparam>
/// <param name="Observation">Observation after the action was applied.</param>
/// <param name="Reward">
/// Reward as float64. Gymnasium rewards are Python floats or <c>np.float64</c>; storing them as float32
/// (as Gym.NET did) changes every accumulated return.
/// </param>
/// <param name="Terminated">True when the MDP reached a terminal state; do not bootstrap past it.</param>
/// <param name="Truncated">True when an outside limit (e.g. <c>TimeLimit</c>) ended the episode; bootstrapping stays valid.</param>
/// <param name="Info">Auxiliary diagnostics with Python dict ordering.</param>
public readonly record struct StepResult<TObs>(TObs Observation, double Reward, bool Terminated, bool Truncated, Info Info);
```

Positional record structs deconstruct like the Python tuples: `var (obs, reward, terminated, truncated, info) = env.Step(action);`.

### 6.3 `Env<TObs, TAct>`

```csharp
using Gym.Envs.Registration;
using Gym.Exceptions;
using Gym.Observations;
using Gym.Spaces;
using Gym.Utils;
using NumSharp;

namespace Gym.Envs;

/// <summary>
/// Base class of every environment. Port of <c>gymnasium.Env</c> (gymnasium/core.py:25 @ 9e04324).
/// </summary>
/// <typeparam name="TObs">Observation element type (see the space table in the port plan §6.1).</typeparam>
/// <typeparam name="TAct">Action element type.</typeparam>
/// <remarks>
/// Implementations must call <see cref="SeedNpRandom"/> at the start of <see cref="Reset"/>, the port of
/// <c>super().reset(seed=seed)</c>. Skipping it breaks seeding determinism, the property every golden test relies on.
/// </remarks>
public abstract class Env<TObs, TAct> : IEnv, IDisposable
{
    private Generator? _npRandom;
    private long? _npRandomSeed;

    /// <summary>Rendering and vectorization metadata (<c>"render_modes"</c>, <c>"render_fps"</c>, <c>"autoreset_mode"</c>, …) with Python key spelling.</summary>
    public virtual Metadata Metadata { get; } = new();

    /// <summary>The render mode fixed at construction, or <c>null</c> when the env doesn't render; changing modes means building a new env.</summary>
    public string? RenderMode { get; protected init; }

    /// <summary>The spec attached by <c>Env.Make</c>, or <c>null</c> for an env built with its constructor.</summary>
    public EnvSpec? Spec { get; set; }

    /// <summary>The space every action passed to <see cref="Step"/> must belong to.</summary>
    public Space<TAct> ActionSpace { get; protected set; } = null!;

    /// <summary>The space every observation returned by <see cref="Step"/> and <see cref="Reset"/> belongs to.</summary>
    public Space<TObs> ObservationSpace { get; protected set; } = null!;

    /// <summary>
    /// The env's random generator. Lazily seeded from OS entropy on first use, exactly like Gymnasium.
    /// Setting it records <see cref="NpRandomSeed"/> as <c>-1</c> because the seed is no longer known.
    /// </summary>
    public Generator NpRandom
    {
        get { if (_npRandom is null) (_npRandom, _npRandomSeed) = Seeding.NpRandom(null); return _npRandom; }
        set { _npRandom = value; _npRandomSeed = -1; }
    }

    /// <summary>The seed <see cref="NpRandom"/> was created from; draws an entropy seed first if none exists yet.</summary>
    public long NpRandomSeed
    {
        get { if (_npRandomSeed is null) (_npRandom, _npRandomSeed) = Seeding.NpRandom(null); return _npRandomSeed.Value; }
    }

    /// <summary>The innermost environment; wrappers override this to walk down the stack.</summary>
    public virtual IEnv Unwrapped => this;

    /// <summary>Runs one timestep of the dynamics.</summary>
    /// <param name="action">An element of <see cref="ActionSpace"/>.</param>
    /// <returns>The transition caused by <paramref name="action"/>.</returns>
    /// <exception cref="AssertionError">Thrown when the source asserts on the action or on a missing <see cref="Reset"/>.</exception>
    public abstract StepResult<TObs> Step(TAct action);

    /// <summary>Starts a new episode.</summary>
    /// <param name="seed">When set, reseeds <see cref="NpRandom"/> before anything else happens.</param>
    /// <param name="options">Env-specific reset options (e.g. <c>"low"</c>/<c>"high"</c> for classic_control).</param>
    /// <returns>The first observation and its info.</returns>
    /// <exception cref="Error">Thrown when <paramref name="seed"/> is negative (mirrors <c>seeding.np_random</c>).</exception>
    public abstract ResetResult<TObs> Reset(long? seed = null, Info? options = null);

    /// <summary>Renders according to <see cref="RenderMode"/>.</summary>
    /// <returns>An RGB frame, a frame list, ANSI text, or <see cref="RenderResult.None"/> for human mode.</returns>
    public virtual RenderResult Render() => RenderResult.None;

    /// <summary>Releases rendering and physics resources. Safe to call more than once.</summary>
    public virtual void Close() { }

    /// <summary>Calls <see cref="Close"/>; lets <c>using</c> play the role of Python's context manager.</summary>
    public void Dispose() { Close(); GC.SuppressFinalize(this); }

    /// <summary>Port of <c>super().reset(seed=seed)</c>: reseeds only when a seed is given.</summary>
    /// <param name="seed">The seed passed to <see cref="Reset"/>.</param>
    /// <exception cref="Error">Thrown for negative seeds.</exception>
    protected void SeedNpRandom(long? seed)
    {
        // Gymnasium reseeds only when a seed is supplied; an unseeded reset keeps the stream going.
        if (seed is not null) (_npRandom, _npRandomSeed) = Seeding.NpRandom(seed);
    }

    /// <summary>Returns <c>&lt;CartPoleEnv&lt;CartPole-v1&gt;&gt;</c>, or <c>&lt;CartPoleEnv instance&gt;</c> without a spec, like <c>Env.__str__</c>.</summary>
    /// <returns>The Gymnasium-formatted description.</returns>
    public override string ToString() => Spec is null ? $"<{GetType().Name} instance>" : $"<{GetType().Name}<{Spec.Id}>>";
}
```

`IEnv` exposes the same members over `object` (`StepResult<object> Step(object action)`) so the registry, wrappers and vector envs can hold any env. `Env.Make<TObs, TAct>(id)` returns the typed view (§3.8, D16).

### 6.4 `Info`, `Metadata`, `RenderResult`

- `Info` is an insertion-ordered `string → object?` map with Python equality semantics for tests. Vector envs use Gymnasium's layout: one array per key plus an `"_" + key` boolean mask (port `VectorEnv._add_info`).
- `Metadata` is an `Info` with typed accessors (`RenderModes`, `RenderFps`, `AutoresetMode`).
- `RenderResult` is a small sealed union: `None`, `Rgb(NDArray)` with shape `(H, W, 3)` and dtype uint8, `Frames(IReadOnlyList<NDArray>)` for `rgb_array_list`, `Ansi(string)`, and `Depth(NDArray)` for MuJoCo `depth_array`.

### 6.5 Wrappers

`Wrapper<TWObs, TWAct, TObs, TAct> : Env<TWObs, TWAct>` forwards every member to the wrapped env, with overridable spaces, metadata and render mode, exactly like `gymnasium.Wrapper` (core.py:290). `ObservationWrapper`, `ActionWrapper` and `RewardWrapper` add the `Observation(obs)`, `Action(act)` and `Reward(r)` hooks. `WrapperSpec`, `ClassName()` and spec propagation (`EnvSpec.AdditionalWrappers`) are ported.

`GetWrapperAttr`, `SetWrapperAttr` and `HasWrapperAttr` walk the stack by name. Names resolve through a `[PyName("…")]` attribute (default: the snake_case form of the PascalCase name), so Python attribute names used in Gymnasium tests keep working.

---

## 7. Spaces

Port `gymnasium/spaces/*` (3,752 lines). Rules specific to spaces:

- **Seeding.** `Seed(long? seed)` returns the seed like Gymnasium. Composite spaces derive subseeds with `NpRandom.integers(int32.MaxValue, size: n)`, which is `Dict.seed`'s exact draw and is verified byte-exact. `Dict` also accepts a dict of seeds; `Tuple` accepts a list.
- **Sampling** mirrors every draw (Appendix A.6 lists them):
  - `Box`: standard normal for unbounded dims, exponential for half-bounded, uniform on `[low, high]` for bounded, with `high + 1` for integer dtypes and `int64` clipping.
  - `Discrete`: `integers(n)`, `choice(np.where(mask))`, `choice(arange(n), p=probability)`.
  - `MultiBinary`: `integers(0, 2, dtype=int8)`.
  - `Sequence`: `geometric(0.25)`, which needs F2.
  - `mask`/`probability` validation errors, with exact messages.
- **`contains`** mirrors dtype castability (`np.can_cast`), shape and bounds checks, and the special cases (`Discrete` accepts Python ints and 0-d integer arrays within `[start, start + n)`).
- **`dtype`, shape and validation.** `Discrete` has shape `()` and dtype int64 (Gym.NET: `(n)` and float32). `Box` validates an explicit dtype, shape consistency, NaN-free bounds, `low <= high` and precision loss exactly like `spaces/box.py`, raising `ValueError`/`TypeError` with the source's messages (§3.4). `InvalidBound` belongs to the wrappers (`StickyAction`, `NormalizeObservation`, `NormalizeReward`, `RescaleAction`, `ClipReward`), not to `Box`.
- **`__repr__` and `__eq__`** are ported (`ToString`/`Equals`): wrappers and `EnvSpec` printing depend on them.
- **`to_jsonable`/`from_jsonable`** are ported so spaces round-trip through JSON the same way.
- **`spaces.utils`:** `flatdim`, `flatten`, `unflatten`, `flatten_space` for every space type (dispatch by type), plus `is_space_dtype_shape_equiv`.
- **Tests:** port `tests/spaces` (13 files, 107 test functions), then golden sampling streams per space.

---

## 8. Registration and make()

Port `gymnasium/envs/registration.py`:

- **Where it lives.** `EnvSpec`, `WrapperSpec`, `VectorizeMode` and the registry go in `Gym.Envs.Registration`. The top-level names (`make`, `make_vec`, `spec`, `register`, `registry`, `pprint_registry`, `register_envs`) become members of the static class `Env` in `Gym.Envs`: `Env.Make`, `Env.MakeVec`, `Env.Spec`, `Env.Register`, `Env.Registry`, `Env.PprintRegistry`, `Env.RegisterEnvs` (§3.8, D16).
- **Registry.** A process-wide ordered registry of `EnvSpec` keyed by id. The id grammar is the source's regex: `^(?:(?P<namespace>[\w:-]+)\/)?(?:(?P<name>[\w:.-]+?))(?:-v(?P<version>\d+))?$`. .NET uses named groups the same way.
- **Version resolution and errors:** `find_highest_version`, plus the exact messages of `NamespaceNotFound`, `NameNotFound`, `VersionNotFound` and `DeprecatedEnv`, which tests assert on.
- **Entry points.** Gymnasium uses `"module:Class"` strings. .NET accepts a `Type`, a factory delegate, or an assembly-qualified type name string. `EnvSpec.EntryPoint` stores the .NET form; `to_json` output differs only in that field (documented deviation).
- **kwargs** are bound to constructor parameters by name (Python spelling via `[PyName]`). An unknown kwarg raises the ported `TypeError` with Python's message (`… got an unexpected keyword argument 'x'`). `make()` matches that exact message for `'render_mode'` and re-raises it as a clearer `error.Error` when `HumanRendering` was requested for an env on the old rendering API, so the message text is part of the contract.
- **`make()` default stack**, in this order:
  1. build the env;
  2. attach `EnvSpec` to `env.unwrapped.spec`;
  3. apply `PassiveEnvChecker` unless disabled;
  4. apply `OrderEnforcing` when `spec.order_enforce`;
  5. apply `TimeLimit(max_episode_steps)` when set;
  6. apply `HumanRendering` when `render_mode="human"` isn't natively supported, or `RenderCollection` for `*_list` modes.

  This stack is part of what an episode *is*: without it CartPole-v1 has no 500-step limit.
- **Also ported:** `make_vec` with `VectorizeMode` (`sync`, `async`, `vector_entry_point`), `spec`, `pprint_registry` (same text layout), `register_envs`, `namespace()`, and `EnvSpec.to_json`/`from_json`/`pprint`.
- **Registrations:** every in-scope id with Gymnasium's `max_episode_steps`, `reward_threshold` and kwargs (Appendix C.4).

---

## 9. Wrappers

Port all 35 in-scope wrappers plus `wrappers/utils.py`. Order follows dependencies; tests come from `tests/wrappers` (50 files, 136 test functions).

| Group | Wrappers | Notes |
|---|---|---|
| make() stack | `PassiveEnvChecker`, `OrderEnforcing`, `TimeLimit` | Needed by P3; `PassiveEnvChecker` needs `utils/passive_env_checker.py` |
| Common | `Autoreset`, `RecordEpisodeStatistics` | `Autoreset` since 1.0 resets on the step *after* termination and sets no `final_observation` (`wrappers/common.py:182`). `RecordEpisodeStatistics` puts `{r, l, t}` in `info["episode"]`; `t` is wall-clock time and is excluded from goldens |
| Observation, stateless | `TransformObservation`, `FilterObservation`, `FlattenObservation`, `GrayscaleObservation`, `ResizeObservation`, `ReshapeObservation`, `RescaleObservation`, `DtypeObservation`, `AddRenderObservation`, `DiscretizeObservation` | `ResizeObservation` needs F8. `RescaleObservation` uses `rescale_box`, which is platform-dependent upstream (§1.3) |
| Observation, stateful | `DelayObservation`, `TimeAwareObservation`, `FrameStackObservation`, `NormalizeObservation`, `MaxAndSkipObservation` | `NormalizeObservation` uses `RunningMeanStd` in float64 |
| Action | `TransformAction`, `ClipAction`, `RescaleAction`, `DiscretizeAction`, `StickyAction`, `RepeatAction` | `StickyAction` draws from the env's `np_random`: mirror the draw |
| Reward | `TransformReward`, `ClipReward`, `NormalizeReward` | |
| Rendering | `RenderCollection`, `RecordVideo`, `HumanRendering`, `AddWhiteNoise`, `ObstructView` | `RecordVideo` needs F9; `HumanRendering` uses the viewers |
| Atari | `AtariPreprocessing` | Optional (§1.4); needs F8 and an ALE binding |
| Helpers | `RunningMeanStd`, `update_mean_var_count_from_moments`, `create_zero_array` | |

Out of scope: `ArrayConversion`, `JaxToNumpy`, `JaxToTorch`, `NumpyToTorch`.

---

## 10. Vector environments

Port `gymnasium/vector` (1,960 lines plus 1,028 of utils) and `gymnasium/wrappers/vector` (2,580 lines).

- **`VectorEnv`** exposes `NumEnvs`, `SingleObservationSpace`, `SingleActionSpace`, batched spaces (`batch_space`), `metadata["autoreset_mode"]`, and `Reset(seed: long | long[] | null, options)`. `Step(actions)` returns batched observations, `float64` rewards, and `bool` terminations and truncations as `NDArray`s, plus the vector `Info` layout.
- **`AutoresetMode`**, ported exactly from `sync_vector_env.py`:
  - `NEXT_STEP` (default): on the step after an episode ends, the sub-env is reset and its slot returns the reset observation with reward `0.0`, `terminated=False`, `truncated=False`;
  - `SAME_STEP`: the sub-env is reset in the same step and `info` gets `final_obs`/`final_info`;
  - `DISABLED`: an autoreset is a programming error (assert).
- **`SyncVectorEnv`** first; it's the reference for the async version.
- **`AsyncVectorEnv`**: decision D7 chooses threads (default recommendation) or processes. Either way, results must equal `SyncVectorEnv` bit for bit. Port `AsyncState`, `reset_async`/`step_async`/`call_async`, and the error forwarding. Shared-memory utilities are needed only in process mode.
- **Utils:** `batch_space`, `batch_differing_spaces`, `iterate`, `concatenate`, `create_empty_array` for every space type.
- **Vector wrappers:** `VectorizeTransform*`, `DictInfoToList`, the vectorized observation, action and reward wrappers, `RecordEpisodeStatistics`, `RecordVideo`, `HumanRendering`.
- **`CartPoleVectorEnv`:** the native vector entry point for CartPole-v0/v1.
- **Tests:** port `tests/vector` (9 files, 68 test functions) and `tests/wrappers/vector`.

---

## 11. Environments

Common requirements for every env port:

- Constructor arguments and defaults, with validation and warnings as in the source; warnings are never exceptions.
- `metadata` with the exact keys and values.
- Spaces with the exact bounds and dtypes.
- `reset(seed, options)` with the exact draw order; `step` with the exact dtype flow (§3.2).
- Rendering through `Gym.Rendering` for `rgb_array` and `human`, plus `ansi` where the source supports it.
- Registration with Gymnasium's limits.
- Goldens: open-loop and sampled trajectories for at least 4 seeds, and frames for 3 fixed states.

### 11.1 classic_control (`gymnasium/envs/classic_control`, 2,012 lines)

| Env | Source | Notes |
|---|---|---|
| `CartPoleEnv` (CartPole-v0: 200 steps / 195; v1: 500 / 475) | `cartpole.py:20` | Re-port. Constants as float64 (`tau = 0.02`, …); `np.inf` bounds; return a float32 copy of the float64 state; `sutton_barto_reward`; `options` through `maybe_parse_reset_bounds`; reward and warning text per source. The proof-of-concept transcription is already bit-identical over 500 steps. |
| `CartPoleVectorEnv` | `cartpole.py:355` | Native vector version. |
| `MountainCarEnv` (v0: 200 / −110) | `mountain_car.py` | |
| `Continuous_MountainCarEnv` (v0: 999 / 90) | `continuous_mountain_car.py` | |
| `PendulumEnv` (v1: 200) | `pendulum.py` | Uses `np.sin(th)`, not the old gym `sin(th + π)`. Its reward mixes float32 and float64 (NEP 50). Array-bound `uniform` in `reset`. `clockwise.png` in rendering. |
| `AcrobotEnv` (v1: 500 / −100) | `acrobot.py` | RK4 integrator, `wrap` and `bound` helpers: port them verbatim. |
| `utils` | `classic_control/utils.py` | `verify_number_and_cast`, `maybe_parse_reset_bounds`. |

### 11.2 toy_text (`gymnasium/envs/toy_text`, 1,917 lines)

| Env | Notes |
|---|---|
| `BlackjackEnv` (Blackjack-v1, `sab=True`) | Card sprites and `Minecraft.ttf` text: needs fonts (§5.5). |
| `FrozenLakeEnv` (v1 4×4: 100 / 0.7; FrozenLake8x8-v1: 200 / 0.85) | `generate_random_map` (seeded `choice` with `p`, verified byte-exact), `is_slippery` transition tables, `ansi` rendering, `desc` as a `'c'` dtype array (F2 or `byte[,]`). |
| `CliffWalkingEnv` (v1, and Slippery-v1 with `is_slippery=True`) | `ansi` rendering. |
| `TaxiEnv` (Taxi-v4: 200 / 8) | `desc` as a `'c'` dtype array; `ansi` rendering; action masks in `info`. |
| helpers | `categorical_sample` (`toy_text/utils.py`) is the transition sampler for FrozenLake, CliffWalking and Taxi; port its draw exactly. |

toy_text needs no physics engine, so it's the second byte-perfect family (P5).

### 11.3 box2d (`gymnasium/envs/box2d`, 2,984 lines; requires F6)

| Env | Notes |
|---|---|
| `LunarLander` (v3 and Continuous-v3: 1000 / 200) | Full re-port on Box2D.NET. v3 changes to honour: wind and turbulence indices are reset in `reset` and drawn only when `enable_wind`; the world is recreated on reset; observation bounds are ±2.5/±10/±2π/±10. Also: particles only when `render_mode` is set; the draw order in `reset` (terrain before force); `ContactDetector` semantics; termination `abs(x) >= 1`; float64 impulse math. Port `heuristic` and `demo_heuristic_lander` into the library. |
| `BipedalWalker` (v3: 1600 / 300; Hardcore-v3: 2000 / 300) | Needs lidar ray casts. |
| `CarRacing` (v3: 1000 / 900) + `car_dynamics.Car` | Track generation, wheel friction, pygame text with the default font. Coordinate with PR #13 (§18). |

### 11.4 mujoco (`gymnasium/envs/mujoco`, 6,180 lines; requires F7)

`MujocoEnv` and `MujocoRenderer`, then Ant, HalfCheetah, Hopper, Humanoid, HumanoidStandup, InvertedDoublePendulum, InvertedPendulum, Pusher, Reacher, Swimmer and Walker2d, each as v4 and v5 (22 ids), with the 14 XML assets. v5 constructor options (`xml_file`, weights, reset noise scale, …) are part of the API.

---

## 12. Utilities, errors, logging, functional API

| Module | Port |
|---|---|
| `utils/seeding.py` | F2 |
| `utils/passive_env_checker.py` | Full port; `make()` runs it by default |
| `utils/env_checker.py` | `check_env` and its checks (reset seed determinism, reset options, step determinism, return types, deprecations, space limits, `data_equivalence`) |
| `utils/env_match.py` | `check_environments_match`; also useful to compare a legacy and a new env during migration |
| `utils/step_api_compatibility.py` | Full port; the bridge used by `Gym.Legacy` (§16) |
| `utils/performance.py` | `benchmark_step`, `benchmark_vector_step`, `benchmark_init`, `benchmark_render` |
| `utils/play.py` | `play`, `PlayableGame`, `PlayPlot` on the viewers (keyboard input); plotting through a .NET chart library |
| `utils/save_video.py` | F9 |
| `utils/colorize.py`, `utils/record_constructor.py` | Port |
| `utils/ezpickle.py` | N/A |
| `error.py` | 20 exception types with the same hierarchy (`Error` base; `UnregisteredEnv` → `NamespaceNotFound`/`NameNotFound`/`VersionNotFound`; …) plus `AssertionError` (§3.4) |
| `logger.py` | `Warn`, `Deprecation`, `Error`, `min_level`, once-per-call-site filtering, colours |
| `experimental/functional.py` | `FuncEnv<TState, TObs, TAct, TReward, TTerminal, TRenderState, TParams>` protocol (no JAX adapters) |

---

## 13. Rendering

- `rgb_array` frames come from `Gym.Rendering` and must be pixel-identical (L4), including the vertical flip and `(H, W, 3)` transpose the envs perform.
- `human` mode draws the same frame into a viewer window, pacing by `metadata["render_fps"]` like `pygame.time.Clock.tick`. The existing `IEnvViewer`, Avalonia and WinForms viewers become human-mode backends, selected by configuration and defaulting to Avalonia.
- `rgb_array_list` and `ansi` follow the source (`RenderCollection`; toy_text text rendering).
- CartPole's pole rotation goes through `pygame.math.Vector2.rotate_rad`. Port pygame's rotation helper, including its snapping of angles within epsilon of multiples of 90°, or frames will differ at those angles.

---

## 14. Verification strategy

### 14.1 Layers

| Layer | What | When |
|---|---|---|
| Unit | Ported Gymnasium tests | Every PR |
| RNG streams | NumSharp vs recorded NumPy streams | Every PR touching seeding or spaces |
| Golden trajectories | Bitwise env dynamics | Every PR touching an env, wrapper or vector code |
| Golden frames | Bitwise `rgb_array` | Every PR touching rendering |
| API inventory | Reflection vs Gymnasium inventory | Nightly and on re-pin |
| Static rules | Analyzer bans (§3.2) | Every build |

### 14.2 Porting the pytest suite

Gymnasium ships 490 test functions (plus 265 `@pytest.mark.parametrize` decorators):

| Area | Test functions |
|---|---:|
| `tests/envs` | 113 |
| `tests/spaces` | 107 |
| `tests/wrappers` | 136 |
| `tests/vector` | 68 |
| `tests/utils` | 46 |
| `tests/functional` | 11 |
| `tests/test_core.py` | 9 |

Mirror each file under `tests/Gym.Tests/Ported/…` with a source citation. Map `parametrize` to `[DynamicData]`, fixtures to helpers, `pytest.raises(X, match=…)` to `Assert.ThrowsExactly<X>` plus a message check, and `pytest.warns` to a captured `Logger` sink. Gymnasium's env tests iterate the registry (`all_testing_env_specs`); the .NET registry supports the same enumeration.

### 14.3 Golden trajectories

- **Content per step:** action; observation (dtype, shape, bit patterns); reward (float64 bits); `terminated`; `truncated`; and the deterministic `info` entries (time-based entries such as `RecordEpisodeStatistics.t` are excluded by an allowlist per wrapper).
- **Cases per env:** 4 seeds × {open-loop, sampled}, running to termination or `max_episode_steps`, with and without the `make()` stack.
- **Failure output:** first diverging step, field, element index, both values and their ulp distance.

### 14.4 Golden frames

Three fixed states per env, set through the source's own attributes (e.g. `env.unwrapped.state` for CartPole) and recorded as PNGs. The comparison is exact.

### 14.5 RNG streams

Record NumPy draws for every pattern in Appendix A.6 (plus new ones as ports add them) and compare bit patterns. This already passes for 16 of 17 patterns.

### 14.6 API inventory

`tools/parity/inventory.py` (an AST scan of `refs/Gymnasium`) produces the list of public names. A reflection test maps them through §3.8 and fails on unmapped in-scope names. Appendix C is its current output.

### 14.7 Static rules

Encode the Appendix A.3 hazards as analyzer diagnostics:

- float32 literals and `MathF` in ported math;
- `Debug.Assert` in `src/`;
- `Console.WriteLine`/`Debug.WriteLine` in `src/`;
- sync-over-async (`.GetAwaiter().GetResult()`, `.Result`, `.Wait()`);
- `lock (this)`;
- `Thread.Abort` family;
- global or unseeded RNG in envs and spaces;
- generic `throw new Exception`;
- public mutable constant-like fields;
- `NDArray == null`/`!= null`.

### 14.8 CI

GitHub Actions (none exists today):

1. **windows-latest (reference):** build with warnings as errors, unit tests, RNG, goldens, frames, analyzers.
2. **ubuntu-latest (tracking):** build and unit tests. Golden comparisons report ulp-level differences without failing (§14.9).
3. **golden-recorder:** rebuild goldens from the pinned venv and fail if they differ from the committed ones. This guards against a drifting reference.

### 14.9 Platform policy

The reference platform is authoritative. Other platforms must pass L1 and L2. L3 differences there are tracked per function (expected sources: libm transcendentals and upstream `float128` use in `rescale_box`). A difference that appears on the reference platform is always a bug.

### 14.10 Re-pinning Gymnasium

Bump the submodule, then:

1. rerun the inventory diff to list added and removed names;
2. `git diff old..new` on `refs/Gymnasium`, mapped through the source citations (§3.1) to the affected C# files;
3. re-record the goldens;
4. port the changes;
5. review every golden change individually.

---

## 15. Gym.NET additions and legacy API

**Keep, as documented additions:**

- Typed generic API (`Env<TObs, TAct>`) and C# idioms (`IDisposable`, record results).
- Human-mode viewers (Avalonia, WinForms), behind `HumanRendering`.
- Async stepping. `StepAsync` becomes an extension over `Task.Run`, or is served by `AsyncVectorEnv`; the `DistributedScheduler` family is removed because it relies on `Thread.Abort`.
- Discrete-action enums such as `LunarLanderDiscreteActions`, as helpers that map to `long` actions.

**Drop:**

- `GoalEnv`
- Baselines-style `VecEnv`/`DummyVecEnv`, whose `Step(int)` sends one action to every env
- `Internal/Dynamic` and `Merging`
- The LunarLander extras that change behaviour or `info`: `MainExhaustRGB`, `ThrusterRGB`, `RenderEngineParticles`, and `info` keys `pos`/`velocity`/`angle`/`omega`/`LeftContact`/`RightContact`, since Gymnasium returns an empty `info`

**Legacy members** (`Seed(int)`, `RewardRange`, `Step.Done`, `Render(string mode)`, the `render.modes`/`video.frames_per_second` metadata keys, `AlreadySteppingError`, `NotSteppingError`) move to `Gym.Legacy` for one major version (§16).

---

## 16. Migration for existing users

- Release the port as **2.0.0**, a breaking change. Package ids and namespaces don't change (D1), so package references and `using` directives keep working; the API inside them changes.
- `Gym.Legacy` provides a `LegacyEnvAdapter` that exposes a new env through the old 4-tuple API, built on the ported `step_api_compatibility`. Everything in it is marked `[Obsolete]` with the replacement named.
- A migration guide maps old calls to new ones: `Reset()` → `Reset(seed)`, `Step(a).Done` → `terminated || truncated`, `Seed(s)` → `Reset(seed: s)`, `Render("rgb_array")` → `render_mode` at construction, `new CartPoleEnv()` → `Env.Make("CartPole-v1")`, `class MyEnv : Env` → `class MyEnv : Env<TObs, TAct>`.
- Remove `Gym.Legacy` in 3.0.

---

## 17. Phased roadmap

Sizes are relative (S < M < L < XL). The phases in the second group can run in parallel with the first once their prerequisites land.

| Phase | Deliverables | Prerequisites | Exit criteria | Size |
|---|---|---|---|---|
| **P0 Groundwork** | CI (§14.8); analyzers (§14.7); golden recorder and manifest (F4); inventory tool; NumSharp swap (F1); `refs/NumSharp` submodule; THIRD-PARTY-NOTICES | — | CI green on the empty new projects; RNG stream test passes 16/17; inventory test runs | M |
| **P1 Core** | `Env`, results, `Info`, `Metadata`, `RenderResult`, `Wrapper` family, `Error`, `Logger`, seeding (F2), `PyMath` (F3) | P0 | `tests/test_core.py` ported and green | M |
| **P2 Spaces** | All 10 spaces, `spaces.utils` | P1, NumSharp `geometric` | `tests/spaces` green; space RNG goldens pass | L |
| **P3 Registry and checkers** | Registration, `make`, `make_vec` scaffold, `PassiveEnvChecker`, `OrderEnforcing`, `TimeLimit`, `Autoreset`, `RecordEpisodeStatistics`, `env_checker`, `step_api_compatibility` | P2 | registration and common-wrapper tests green | M |
| **P4 classic_control** | 5 envs and `CartPoleVectorEnv` (dynamics) | P3 | L3 goldens for all 6 ids | M |
| **P5 toy_text** | 4 envs (dynamics and `ansi`) | P3 | L3 goldens for all 6 ids | M |
| **P6 Vector and wrappers** | `VectorEnv`, `SyncVectorEnv`, `AsyncVectorEnv`, utils, remaining wrappers, vector wrappers | P3 | `tests/vector` and `tests/wrappers` green; async equals sync bitwise | L |
| **P10 Release** | Docs, migration guide, `Gym.Legacy`, examples, packages 2.0.0 | P4–P6 at minimum | Definition of done (§1.5) for the released scope | M |
| **P7 Rasterizer** | F5; frames for classic_control and toy_text; fonts | P1 | L4 goldens for classic_control, then toy_text | L |
| **P8 Box2D** | F6; LunarLander, BipedalWalker, CarRacing | P3 (for envs); the engine can start at P0 | Engine goldens, then env goldens | XL |
| **P9 MuJoCo** | F7; `MujocoEnv`, `MujocoRenderer`, 22 ids | P3; bindings can start at P0 | Hopper-v5 golden, then all | XL |

A reasonable first release (2.0) is P0–P7: core, spaces, registry, wrappers, vector, classic_control and toy_text, all byte-perfect with pixel-perfect frames. Box2D and MuJoCo families follow in minor releases.

---

## 18. External contributions

- **PR #24 (open).** Worth keeping:
  - the TFM upgrade to `net8.0;net10.0` and the package bumps;
  - the WinForms non-interactive fallback;
  - the `refs/Gymnasium` submodule;
  - the corrected LunarLander baseline value, but it isn't a Gymnasium golden and its `#if DEBUG` guard should go.

  Not worth keeping: the PRD's Object Calisthenics mandate, the committed plan file, docs that describe non-existent APIs, and the claims that `Box`/`Discrete` are complete. Its `Box` scalar-path fix becomes irrelevant after F1. Suggested path: merge a trimmed version (TFMs, packages, submodule, WinForms fix), and replace its docs with this plan (decision D11).
- **Fork `HalfTick/Gym.NET@cb9dcb6` (unmerged, 6 commits).** It implements many missing names, but repeats the root causes: 83 float32 constants, float32 rewards, NumSharp 0.60.0 (no PCG64), and `SyncVectorEnv` writing the pre-1.0 `final_observation`. Treat it as a reference for scope, not as mergeable code; new ports follow §3 and must pass goldens (decision D12).
- **PR #13 (bojake, CarRacing).** Re-target to CarRacing-v3 on Box2D.NET (P8); reuse the track-generation and rendering work where it matches the v3 source.
- **Rules for any contribution:**
  - one Gymnasium module per PR, with source citations;
  - goldens for everything it touches;
  - analyzer-clean;
  - XML documentation per §3.7;
  - `refs/Gymnasium` and `refs/NumSharp` present for review, and for agents, so ports are made from current sources.

---

## 19. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Transcendental functions differ across platforms (libm, NumPy SIMD kernels) | L3 fails off the reference platform | Pinned reference platform (§1.3); per-function tracking on Linux (§14.9); verified bit-identical on Windows for CartPole |
| NEP 50 promotion subtleties (float32 × Python float stays float32) | Silent divergence | §3.2 table; per-expression dtype review; goldens catch every miss |
| Box2D port size and exactness | Box2D envs delayed | Engine-level goldens from day one; native interop fallback (D5); schedule as parallel track |
| pygame `draw` is LGPL-2.1 | Licensing of `Gym.Rendering` | Decision D4 before P7 starts |
| FreeType text rendering parity | Blackjack and CarRacing frames | FreeType interop; fonts limited to two envs |
| MuJoCo native distribution and version drift | L3 depends on the exact native build | Pin version (D6); ship per-RID native packages |
| NumSharp gaps (`geometric`, array-parameter distributions, `'c'` dtype) | Blocks `Sequence`; forces emulations | Upstream in NumSharp (same maintainer); exact emulation until then |
| Gymnasium keeps moving (pinned to an unreleased 1.4.0 dev commit) | Re-work on every bump | Pin to a release tag when one covers these changes (D2); re-pin procedure (§14.10) |
| Asset licenses (Minecraft.ttf, sprites) | Can't redistribute in packages | Decision D14: ship with permission, or load from a user-provided Gymnasium checkout |
| Contributors reintroduce float32 math or global RNG | Silent parity loss | Analyzers (§14.7), goldens required in PRs |
| .NET runtime changes float behaviour | Rare, but possible | .NET doesn't contract or reassociate floating point; keep `Vector*` and FMA out of ported math; run goldens on both TFMs (verified identical between .NET 6, 8 and 10 for the existing LunarLander baseline) |

---

## 20. Decisions for the maintainer

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | Root namespace and package ids | `Gym` (current) or `Gymnasium` | **Decided 2026-09-23: it stays Gym.NET.** Root namespace `Gym` with today's namespaces kept (§3.8), NuGet ids `Gym.NET.*`, and project names in the existing `Gym.*` families (§4.1). No type can be named `Gym` (§3.8), so the `make` entry point is D16 |
| D2 | Reference platform and Gymnasium pin | Windows x64 or Linux x64; current SHA or next release tag | Windows x64 (§1.3); move to the next Gymnasium release tag once it exists |
| D3 | NumSharp consumption | Project reference to `refs/NumSharp`, or NuGet ≥ 0.70 | NuGet pinned for releases; submodule for development |
| D4 | Rasterizer licensing | LGPL sub-package; clean-room port verified by pixel tests; native SDL + pygame interop | Port `gfxdraw` (zlib) freely; clean-room `draw` verified by golden frames; LGPL sub-package only if clean-room parity fails |
| D5 | Box2D | C# port or native interop | C# port (portable, debuggable) with engine goldens; native interop as fallback |
| D6 | MuJoCo version | Any 3.x | Pin the version Gymnasium's CI uses at the pinned SHA |
| D7 | `AsyncVectorEnv` model | Threads or processes | Threads by default; processes later if isolation is needed |
| D8 | Generic API | Generic `Env<TObs, TAct>` or object-typed only | Generic plus non-generic `IEnv` (§6) |
| D9 | Legacy shim | Keep one major version, or hard break | Keep in 2.x (§16) |
| D10 | Aether-based LunarLander in 2.0 | Ship it as an extra, or drop it until P8 | Drop; it's broken (Appendix B) and not Gymnasium |
| D11 | PR #24 | Merge as is, merge trimmed, or close | Merge trimmed (§18) |
| D12 | Fork cb9dcb6 | Merge or re-port | Re-port (§18) |
| D13 | `examples/` | Update or remove | Rewrite one CartPole example on the new API; remove the rest |
| D14 | toy_text assets | Ship in packages, or load from a Gymnasium checkout | Seek permission; until then load from `refs/Gymnasium` |
| D15 | Golden storage | In repo, Git LFS, or CI artifacts | In repo, gzip-compressed JSONL (small for classic and toy_text); LFS for MuJoCo |
| D16 | Home of the top-level registration API (`make`, `make_vec`, `spec`, `register`, `registry`, `pprint_registry`, `register_envs`); follows from D1 | A static class `Env` next to `Env<TObs, TAct>` (`Env.Make(…)`); a static class `Gymnasium` in `Gym` (`Gymnasium.Make(…)`, spelled like the Python module). A static class `Gym` is ruled out: CS0234 and CS0426 (§3.8) | Static `Env` in `Gym.Envs`. It follows the BCL's `Tuple`/`Tuple<T1>` and `ImmutableArray`/`ImmutableArray<T>` pattern, sits in the namespace users already import, and adds no second brand to the API. The current non-generic `Env` base class goes away under either option (§4.4) |
| D17 | Python built-in exceptions (`ValueError`, `TypeError`, `KeyError`, …) | Throw NumSharp's ports (namespace `NumSharp`), or define Gym.NET's own in `Gym.Exceptions` | **Decided 2026-09-23: reuse NumSharp's** (§3.4). Python has one `ValueError` for NumPy and Gymnasium alike, and a second definition is ambiguous (CS0104) wherever both namespaces are imported. `Gym.Exceptions` holds `gymnasium.error`'s types plus the built-ins NumSharp lacks (`AssertionError`, `NotImplementedError`, …). Recommendation: add those to NumSharp as well (§5.2 item 4). A later NumSharp copy, for example for `np.testing` (whose asserts raise `AssertionError`), would otherwise bring the collision back |

---

## Appendix A: Gap-scan methods and results

All scans ran on 2026-09-23 against the pinned revisions. The prototype scripts lived in a session scratchpad; P0 turns them into `tools/parity/`.

### A.1 Gymnasium surface inventory (AST)

An AST scan of all 121 modules (33,841 lines) of `refs/Gymnasium/gymnasium` collected every `__all__`, every public class with its public members, and every public function. A separate scan of the `register()` calls in `gymnasium/envs/__init__.py` collected the 63 registrations. The output is Appendix C.

### A.2 Gym.NET public API inventory (reflection)

Reflection over the built `Gym` and `Gym.Environments` assemblies: **45 public types, 278 public members**. Public namespaces:

- `Gym.Envs`: `Env`, `Env<T>`, `GoalEnv`, `IEnv`, `IVecEnv`, `VecEnv`, `VecEnvWrapper`, `DummyVecEnv`
- `Gym.Spaces`: `Space`, `Box`, `Discrete`, `BoundedMannerEnum`
- `Gym.Observations`: `Step`
- `Gym.Exceptions`: 3 types
- `Gym.Environments`: `IEnvViewer`, `IEnvironmentViewerFactoryDelegate`, `NullEnvViewer`
- `Gym.Environments.Envs.Classic`: `CartPoleEnv`
- `Gym.Environments.Envs.Aether`: `LunarLanderEnv`, `LunarLanderDiscreteActions`

Infrastructure exposed as public API with no Gymnasium counterpart: `Gym.Collections` (8 types), `Gym.Dynamic`, `Gym.Merging` (4), `Gym.Reflection` (5), `Gym.Threading` (5).

### A.3 Static hazard scan

Regex rules over `src/**/*.cs`, run on `pr/24` and on the fork:

| Rule | Why it matters | pr/24 | fork |
|---|---|---:|---:|
| float32 constant | Python scalars are float64 | 40 | 83 |
| float32 literal | Same expression evaluates in float64 upstream | 224 | 305 |
| `(float)` narrowing | Precision loss inside ported math | 20 | 60 |
| `Debug.Assert` | Python `assert` raises; this vanishes in Release | 3 | 2 |
| `Console`/`Debug.WriteLine` | Not `logger.warn` semantics | 7 | 7 |
| Sync-over-async | Deadlock risk in render paths | 5 | 10 |
| `lock (this)` | Locks a public object | 2 | 6 |
| `Thread.Abort` family | Throws on .NET 5+ | 7 | 7 |
| Global or unseeded RNG | Breaks per-env and per-space seeding | 4 | 6 |
| `throw new Exception` | Untyped errors | 5 | 5 |
| `WarningException` thrown | Gymnasium only warns | 2 | 2 |
| Public mutable constant | Silent dynamics changes | 5 | 5 |
| TODO / not implemented | Unfinished paths | 5 | 3 |
| `mask != null` on NDArray | Element-wise operator throws | 1 | 0 |

### A.4 Constant audit

Every numeric constant compared with the same-named Gymnasium value; errors are the exact float32 widening.

| Env | Constant | C# | As double | Error |
|---|---|---|---|---:|
| CartPole | `gravity` | `9.8f` | 9.800000190734863 | 1.9e-7 |
| CartPole | `masspole` | `0.1f` | 0.10000000149011612 | 1.5e-9 |
| CartPole | `tau` | `0.02f` | 0.019999999552965164 | 4.5e-10 |
| CartPole | `x_threshold` | `2.4f` | 2.4000000953674316 | 9.5e-8 |
| CartPole | `theta_threshold_radians` | `(float)(12·2π/360)` | float32-rounded | — |
| LunarLander | `SIDE_ENGINE_POWER` | `0.6f` | 0.6000000238418579 | 2.4e-8 |

The other CartPole and LunarLander constants are exactly representable. The fork adds float32 constants to every new env: Pendulum `dt = 0.05f`, `g = 10.0f`; MountainCar `force = 0.001f`, `gravity = 0.0025f`; Acrobot `Dt = 0.2f`, `MaxVel1 = 4.0f * (float)Math.PI`.

### A.5 NumPy API coverage

An AST scan of Gymnasium's in-scope modules collected **95** `np.*` symbols (2 are `np.typing`, so 93 apply) and **24** ndarray method names. They were checked against NumSharp's reflected surface (`np`: 428 members; `NDArray`: 168; 385 exported types):

- **86** resolve directly; all 24 method names exist on `NDArray`; `np.linalg` exists.
- **5** resolve through NumSharp spellings, each verified against NumPy:
  - `np.round` → `np.round_`/`np.around`, half-to-even, including `-0.`;
  - `np.integer`, `np.floating`, `np.number`, `np.generic` → `np.issubdtype(dtype, "integer" | "floating" | "number" | "generic")`, identical results for int32, int64, float32, float64 and bool.
- **2** real gaps:
  - `np.bytes_`: the `'c'` dtype of FrozenLake's and Taxi's `desc`;
  - `np.float128`: `rescale_box`; absent on Windows NumPy as well.

Details in Appendix D.

### A.6 RNG differential

NumPy 2.4.2 `np.random.default_rng(seed)` (identical to `gymnasium.utils.seeding.np_random(seed)[0]`) versus NumSharp `np.random.default_rng(seed)`. One generator per seed with the calls in sequence, so state advancement is covered; seeds 0, 1, 42, 1234 and 2147483647; raw IEEE bits compared.

| Draw pattern | Used by | Result |
|---|---|---|
| `uniform(-0.05, 0.05, size=4)` | CartPole reset | 5/5 identical |
| `uniform(0, H/2, size=12)` | LunarLander terrain | 5/5 |
| `uniform(-1000, 1000)` × 2 | LunarLander initial force | 5/5 |
| `integers(-9999, 9999)` × 2 | LunarLander wind indices | 5/5 |
| `uniform(-1, 1)` × 2 | LunarLander dispersion | 5/5 |
| `integers(4, dtype=int64)` | `Discrete.sample` | 5/5 |
| `choice([0, 2, 3])` | `Discrete.sample(mask)` | 5/5 |
| `choice(arange(4), p)` | `Discrete.sample(probability)` | 5/5 |
| `normal(size=3)` | `Box` unbounded | 5/5 |
| `exponential(size=3)` | `Box` half-bounded | 5/5 |
| `random(5)` | various | 5/5 |
| `standard_normal(2)` | various | 5/5 |
| `integers(0, 2, size=3, dtype=int8)` | `MultiBinary.sample` | 5/5 |
| `integers(int32.max, size=3)` | `Dict`/`Tuple` subseeds | 5/5 |
| `choice(2, size=(4, 4), p=[.8, .2])` | FrozenLake maps | 5/5 |
| `uniform(low=[…], high=[…])` via `low + (high - low) * random()` | `Box` bounded, Pendulum reset | 5/5 |
| `geometric(0.25)` | `Sequence` length | missing on `Generator` |

Total: **80 of 85** results bit-identical; the 5 failures are the missing method.

### A.7 Dynamics differential (CartPole)

Gymnasium's `CartPoleEnv` recorded 4 open-loop trajectories (seeds 0, 1, 42, 1234; 500 steps each, actions from a balancing heuristic evaluated on Python's own states). C# replayed the identical initial float64 state and actions:

| Seed | Gym.NET `CartPoleEnv` | float64 transcription (`Math.Cos/Sin`) |
|---|---|---|
| 0 | differs at step 1; max \|Δstate\| 27.3; terminates at step 247 (Gymnasium: never) | bit-identical, 500/500 |
| 1 | differs at step 1; 26.2; terminates at step 254 | bit-identical, 500/500 |
| 42 | differs at step 1; 26.6; terminates at step 251 | bit-identical, 500/500 |
| 1234 | differs at step 1; 26.9; terminates at step 251 | bit-identical, 500/500 |

Gym.NET also returns float64 observations where Gymnasium returns float32.

### A.8 Env-checker battery

Checks modelled on `gymnasium.utils.env_checker` and `passive_env_checker`, run against `pr/24`:

| Check | Result |
|---|---|
| `Discrete(4).Sample()` | **throws** `NullReferenceException` (`Discrete.cs:18`) |
| `Discrete(4).Contains(3)` | pass |
| `Box(-1, 1, (3,)).Sample()` | **throws** `IncorrectShapeException` (`Box.cs:96`) |
| `Box(-1, 1, (3,)).Contains(Sample())` | **throws** (`Box.cs:96`) |
| `Box(0, 5, (2000,), int32)` construction | **throws** `NotImplementedException` (`Box.cs:48`) |
| `Box(NDArray, NDArray)` sample | **throws** `IncorrectShapeException` |
| Two `Box`es seeded 7 sample identically | **throws** before comparing |
| CartPole `ActionSpace.Sample()` as a step action | **throws** `NullReferenceException` |
| CartPole reset obs ∈ observation space, dtype matches | **throws** (`Box.cs:128`); obs float64 vs space float32 |
| CartPole `Seed(5)` + `Reset()` repeatable | pass |
| LunarLander step determinism with equal seeds | pass |
| LunarLander obs inside its own observation space | **fail**: 169/1,770 random-play observations outside |
| LunarLander without `random_state` isolated from a second env | **fail**: shared global `np.random` |
| LunarLanderContinuous `Step(ActionSpace.Sample())` | **throws** `OverflowException` (`LunarLanderEnv.cs:618`) |
| `LunarLanderEnv(wind_power: 25)` | **throws** `WarningException` (Gymnasium warns) |
| CartPole `Render("rgb_array")` returns an RGB array | **fail**: returns an ImageSharp `Image` |

### A.9 Pixel differential (CartPole)

pygame-ce 2.5.8 `rgb_array` frames versus Gym.NET `Render()` at the same states (600 × 400):

| State | Differing pixels | Max channel delta |
|---|---:|---:|
| centered `[0, 0, 0, 0]` | 2,573 (1.07%) | 255 |
| tilted `[0.5, 0, 0.15, 0]` | 2,747 (1.14%) | 255 |
| left `[-1.2, 0, -0.1, 0]` | 2,659 (1.11%) | 255 |

Causes: pole colour `(204,153,102)` vs `(202,152,101)`; axle drawn in the pole colour instead of `(129,132,203)`; ImageSharp anti-aliasing vs SDL_gfx.

### A.10 Physics engine differential

The same scene in Box2D 2.3.10 (pybox2d) and Aether.Physics2D 1.6.1, with LunarLander's settings: gravity (0, −10); the hull `LANDER_POLY / 30` with density 5, friction 0.1, restitution 0; an edge ground; a 300 N push; dt 1/50; 180 velocity and 60 position iterations; 300 steps.

- The first difference is at **step 1**, during free fall: x 10.0264845 (Aether) vs 10.024914 (Box2D).
- After 300 steps the bodies rest at (12.80359, 0.3424233) vs (11.054015, 0.3483163), 1.75 m apart.

### A.11 Platform and promotion probes

- NumPy 2.4.2 on Windows x64: `hasattr(np, "float128")` is `False`; `longdouble` is float64.
- NEP 50: with `u = np.float32(1.3)`, `0.001 * (u**2)` is `np.float32(0.00169)`; float64 math gives 0.0016899998760223412. Pendulum rewards are `np.float64` overall, CartPole rewards are Python `float`, and observations are float32.

### A.12 Repository health

- **Release build, non-incremental (16 warning codes):**
  - `CS0219` ×20, `CS0618` ×16, `SYSLIB0006` ×12, `CS0162` ×12, `CA1416` ×12, `SYSLIB0003` ×8, `CS0649` ×8, `CS8073` ×4, `CS0169` ×4, `WFDEV004` ×2, `CS0672` ×2;
  - NuGet: `NU1903` ×16, `NU1904` ×6, `NU5125` ×8, `NU5048` ×8, `NETSDK1206` ×8.
- **Examples:** the build fails with `NU1201` (netcoreapp3.0/netstandard2.0 projects referencing net8/net10 libraries); they aren't in the solution.
- **Project settings:** no `.github/workflows`; no `<Nullable>`, `<TreatWarningsAsErrors>` or `<Deterministic>` settings.

### A.13 Contributor fork

The same rules applied to `cb9dcb6` (A.3 fork column). A source read confirmed float32 constants in all four new classic envs, `float` rewards in `StepResult`, NumSharp 0.60.0, and `info["final_observation"]` in `SyncVectorEnv` (line 46).

---

## Appendix B: Defect register

Verified on `pr/24`. The last column names the phase that removes the defect (all are fixed by re-porting, not patching).

| ID | Location | Defect | Evidence | Removed in |
|---|---|---|---|---|
| B-01 | `src/Gym/Spaces/Discrete.cs:18` | `Sample()` throws `NullReferenceException` for any call: `mask != null` is element-wise | A.8 probe | P2 |
| B-02 | `Discrete.cs:39` | `Contains` ignores `Start`: `Discrete(3, start: 5)` accepts 0 and rejects 5, 7 | Probe | P2 |
| B-03 | `Discrete.cs:11` | Shape `(n)` and dtype float32 instead of `()` and int64 | Probe | P2 |
| B-04 | `Discrete.cs:23` | Mask path would sample from `[0, firstValidIndex)` | Source read | P2 |
| B-05 | `src/Gym/Spaces/Box.cs:96` | `Sample()` throws for every non-scalar box | A.8 probe | P2 |
| B-06 | `Box.cs:128` | `Contains()` throws for every non-scalar box | A.8 probe | P2 |
| B-07 | `Box.cs:48` | Integer boxes can't be constructed (mixed-dtype comparison) | A.8 probe | P2 |
| B-08 | `Box.cs:88`, `Box.cs:105` | Unbounded dims draw `normal(0.5, 1)`; Gymnasium uses a standard normal | Source read | P2 |
| B-09 | `Box.cs:117` | Integer boxes never sample `high` (no `+ 1`) | Source read | P2 |
| B-10 | `Box.cs:25`–`44` | No validation of dtype, NaN bounds or `low <= high` (Gymnasium raises `ValueError`/`TypeError`) | Source read | P2 |
| B-11 | `src/Gym.Environments/Envs/Classic/CartPoleEnv.cs:24`–`36` | float32 physics constants: diverges at step 1 | A.7 | P4 |
| B-12 | `CartPoleEnv.cs:46` | `float.MaxValue` bounds instead of `inf` (changes `Box` boundedness) | Source read | P4 |
| B-13 | `CartPoleEnv.cs:185` | Returns the internal float64 state (aliased) instead of a float32 copy | A.7 | P4 |
| B-14 | `CartPoleEnv.cs:139` | Action validation is `Debug.Assert` only; Release treats any non-1 action as "push left" | Source read | P4 |
| B-15 | `CartPoleEnv.cs:49` | Unseeded `RandomState()` fallback | A.3 | P4 |
| B-16 | `CartPoleEnv.cs:115`–`121` | Wrong colours; 1.07–1.14% of pixels differ | A.9 | P7 |
| B-17 | `src/Gym.Environments/Envs/Aether/LunarLanderEnv.cs:496` vs `:505`–`508` | Reset draws force before terrain (Gymnasium: terrain first) | Source read | P8 |
| B-18 | `LunarLanderEnv.cs:409`–`410` | Wind indices drawn in the constructor, even without wind | Source read | P8 |
| B-19 | `LunarLanderEnv.cs:412`–`413` | v2 observation bounds; 9.5% of random-play observations fall outside | A.8 | P8 |
| B-20 | `LunarLanderEnv.cs:417`, `:618` | Continuous action space is 0-d; `Step(Sample())` throws `OverflowException` | A.8 | P8 |
| B-21 | `LunarLanderEnv.cs:526`–`531` | Terrain smoothing uses 0 where Python's `height[-1]` wraps | Source read | P8 |
| B-22 | `LunarLanderEnv.cs:259`–`277` | Legs are a quarter of Gymnasium's area and start ≈ 1.33 m from their joint anchors | Source read | P8 |
| B-23 | `LunarLanderEnv.cs:316`–`329` | Contact listener clears leg flags on every new contact; crash contact not sticky | Source read | P8 |
| B-24 | `LunarLanderEnv.cs:762` | No left-edge termination (`x > 1` vs `abs(x) >= 1`) | Source read | P8 |
| B-25 | `LunarLanderEnv.cs:658`, `:703` | Particles created even when nothing renders | Source read | P8 |
| B-26 | `LunarLanderEnv.cs:600` | Continuous actions cast to float32 (Gymnasium: float64) | Source read | P8 |
| B-27 | `LunarLanderEnv.cs:402`, `:406` | Throws `WarningException` where Gymnasium warns | A.8 | P8 |
| B-28 | `LunarLanderEnv.cs:155`–`164` | FPS, viewport and engine geometry are public mutable fields | A.3 | P8 |
| B-29 | `LunarLanderEnv.cs:384`–`387` | Falls back to the global `np.random`; a second env perturbs the first | A.8 | P8 |
| B-30 | `LunarLanderEnv.cs:806` | `p.RenderColor != null` on a struct is always true (`CS8073`) | A.12 | P8 |
| B-31 | `src/Gym/Envs/Env.cs:34` | `Unwrapped` is protected; no wrapper attribute access | Source read | P1 |
| B-32 | `Env.cs:56`–`59` | `GoalEnv.Reset()` returns `null` | Source read | P1 (deleted) |
| B-33 | `src/Gym/Envs/VecEnvWrapper.cs:22`–`24` | `Step(int)` sends the same action to every sub-env | Source read | P6 (deleted) |
| B-34 | `src/Gym/Internal/Threading/DistributedThread.cs:114`–`137`, `DistributedQueuedTaskScheduler.cs:194` | `Thread.Abort`/`ResetAbort`/compressed stack throw on .NET 5+ | A.12 | P1 (deleted) |
| B-35 | `CartPoleEnv.cs:83`–`88`, `LunarLanderEnv.cs:779`–`784` | Viewer creation blocks on async inside `lock (this)` | A.3 | P7 |
| B-36 | `examples/ReinforcementLearning/*` | Doesn't build (`NU1201`) | A.12 | P10 |
| B-37 | `tests/Gym.Tests/Envs/Aether/LunarLanderEnvironment.cs:65` | The only determinism assertion runs in Debug builds only | Source read | P0 |
| B-38 | Repository | No CI; no nullable annotations; 16 warning codes; vulnerable transitive packages through Avalonia 0.10.22 | A.12 | P0 |

---

## Appendix C: Gymnasium surface map

Status legend: **M** matches, **D** diverges, **B** broken, **—** missing, **n/a** out of scope. "Fork" marks names the unmerged fork adds (new) or rewrites (changed); presence only, not fidelity.

### C.1 Core (`gymnasium/core.py`)

| Gymnasium | Gym.NET today | Status | Fork |
|---|---|---|---|
| `Env.step` → 5-tuple | `Step Step(object)` → 4-tuple, float reward | D | changed |
| `Env.reset(*, seed, options)` → `(obs, info)` | `NDArray Reset()` | D | changed |
| `Env.render()` + `render_mode` | `Image Render(string mode)` | D | |
| `Env.close()` | `CloseEnvironment()`, `Dispose()` | D | |
| `__enter__`/`__exit__` | `IDisposable` | M | |
| `np_random`, `np_random_seed` | private fields | — | |
| `action_space`, `observation_space` | `ActionSpace`, `ObservationSpace` | M | |
| `metadata` | `Dict Metadata` with legacy keys | D | |
| `spec` | — | — | |
| `unwrapped` | `protected Unwrapped()` | D | |
| `get_wrapper_attr`, `set_wrapper_attr`, `has_wrapper_attr` | — | — | |
| `__str__` | default | — | |
| `Wrapper`, `ObservationWrapper`, `ActionWrapper`, `RewardWrapper` | — | — | new |

### C.2 Spaces (`gymnasium/spaces`)

| Gymnasium | Status | Fork |
|---|---|---|
| `Space` | D | changed |
| `Box` | B | changed |
| `Discrete` | B | changed |
| `MultiDiscrete`, `MultiBinary`, `Tuple`, `Dict`, `Text`, `Sequence`, `Graph` + `GraphInstance`, `OneOf` | — | new |
| `flatdim`, `flatten`, `unflatten`, `flatten_space`, `is_space_dtype_shape_equiv` | — | |

### C.3 Registration (`gymnasium/envs/registration.py`)

`register`, `make`, `make_vec`, `VectorizeMode`, `registry`, `spec`, `pprint_registry`, `EnvSpec`, `WrapperSpec`, `register_envs`, `namespace`, `current_namespace`, `parse_env_id`, `get_env_id`, `find_highest_version`, `load_env_creator`: all missing.

### C.4 Registered environments (63 `register()` calls)

| Family | Ids (steps / reward threshold) | Gym.NET | Fork |
|---|---|---|---|
| classic_control | CartPole-v0 (200/195), CartPole-v1 (500/475) | D | changed |
| | MountainCar-v0 (200/−110), MountainCarContinuous-v0 (999/90), Pendulum-v1 (200), Acrobot-v1 (500/−100) | — | new |
| | `CartPoleVectorEnv` (vector entry point) | — | |
| box2d | LunarLander-v3, LunarLanderContinuous-v3 (1000/200) | B | changed |
| | BipedalWalker-v3 (1600/300), BipedalWalkerHardcore-v3 (2000/300), CarRacing-v3 (1000/900) | — | |
| toy_text | Blackjack-v1, FrozenLake-v1 (100/0.7), FrozenLake8x8-v1 (200/0.85), CliffWalking-v1, CliffWalkingSlippery-v1, Taxi-v4 (200/8) | — | |
| mujoco | Ant, HalfCheetah, Hopper, Humanoid, HumanoidStandup, InvertedDoublePendulum, InvertedPendulum, Pusher, Reacher, Swimmer, Walker2d, each as -v4 and -v5 | — | |
| n/a | 17 mujoco-py ids (-v2/-v3, raise on make); phys2d/CartPole-v0, phys2d/CartPole-v1, phys2d/Pendulum-v0; tabular/Blackjack-v0, tabular/CliffWalking-v0; GymV21Environment-v0, GymV26Environment-v0 | n/a | |

Env helpers, all missing:

- `classic_control.utils`: `verify_number_and_cast`, `maybe_parse_reset_bounds`
- `lunar_lander.heuristic`, `demo_heuristic_lander`
- `car_dynamics.Car`
- `toy_text.utils.categorical_sample`, `frozen_lake.generate_random_map`, Blackjack card helpers
- `MujocoEnv`, `MujocoRenderer`

### C.5 Wrappers (39 exported)

All missing on `pr/24`. The fork adds `TimeLimit`, `OrderEnforcing`, `Autoreset`, `RecordEpisodeStatistics`, `TransformObservation`, `ClipAction`, `RescaleAction` and `TransformReward`. `ArrayConversion`, `JaxToNumpy`, `JaxToTorch` and `NumpyToTorch` are n/a. The full list with notes is in §9.

### C.6 Vector (`gymnasium/vector`)

`VectorEnv`, `VectorWrapper`, `VectorObservationWrapper`, `VectorActionWrapper`, `VectorRewardWrapper`, `SyncVectorEnv`, `AsyncVectorEnv`, `AsyncState`, `AutoresetMode`; utils `batch_space`, `batch_differing_spaces`, `iterate`, `concatenate`, `create_empty_array`, `create_shared_memory`, `read_from_shared_memory`, `write_to_shared_memory`: all missing. The fork adds `VectorEnv`, `SyncVectorEnv` and `AsyncVectorEnv`. `CloudpickleWrapper` and `clear_mpi_env_vars` are n/a.

### C.7 Vector wrappers (`gymnasium/wrappers/vector`, 26 exported)

`VectorizeTransformObservation`, `VectorizeTransformAction`, `VectorizeTransformReward`, `DictInfoToList`, 9 observation, 3 action and 3 reward wrappers, `RecordEpisodeStatistics`, `RecordVideo`, `HumanRendering`: all missing. The 4 array-conversion wrappers are n/a.

### C.8 Utilities (`gymnasium/utils`)

`seeding.np_random`; `check_env` and 8 checks; `passive_env_checker` (7 functions); `check_environments_match`; 4 benchmarks; `play`, `PlayableGame`, `PlayPlot`, `display_arr`, `MissingKeysToAction`; `save_video`, `capped_cubic_video_schedule`; 3 step-API converters; `colorize`; `RecordConstructorArgs`: all missing. `EzPickle` is n/a.

### C.9 Errors and logger

| Gymnasium | Gym.NET | Status |
|---|---|---|
| `InvalidAction` | `InvalidActionError` | D |
| `AlreadyPendingCallError` | `AlreadySteppingError` | D |
| `NoAsyncCallError` | `NotSteppingError` | D |
| `Error`, `UnregisteredEnv`, `NamespaceNotFound`, `NameNotFound`, `VersionNotFound`, `DeprecatedEnv`, `RegistrationError`, `DependencyNotInstalled`, `UnsupportedMode`, `InvalidMetadata`, `ResetNeeded`, `MissingArgument`, `InvalidProbability`, `InvalidBound`, `DeprecatedWrapper`, `ClosedEnvironmentError`, `CustomSpaceError` | — | — |
| `logger.warn`, `logger.deprecation`, `logger.error` | `Console`/`Debug.WriteLine` | — |

### C.10 Functional

`FuncEnv`: missing. `FunctionalJaxEnv`, `FunctionalJaxVectorEnv`: n/a.

---

## Appendix D: NumPy and NumSharp coverage

`np.*` symbols used by in-scope Gymnasium code, by frequency:

- **100 uses or more:** `np.ndarray` 160, `np.array` 127
- **30–99:** `np.inf` 73, `np.integer` 73, `np.float32` 62, `np.float64` 59, `np.zeros` 59, `np.all` 38, `np.concatenate` 37, `np.random.Generator` 36, `np.bool_` 34, `np.clip` 34, `np.square` 34, `np.any` 32, `np.dtype` 30
- **10–29:** `np.int64` 28, `np.sum` 28, `np.asarray` 27, `np.issubdtype` 27, `np.uint8` 26, `np.int32` 24, `np.floating` 19, `np.linalg.norm` 19, `np.int8` 17, `np.abs` 16, `np.full` 16, `np.transpose` 15, `np.logical_or` 14, `np.round` 14, `np.logical_and` 13, `np.sign` 13, `np.where` 12, `np.sin` 10, `np.sqrt` 10
- **Fewer than 10:** `np.iinfo`, `np.isinf`, `np.copy`, `np.cos`, `np.prod`, `np.empty`, `np.frombuffer`, `np.isfinite`, `np.linspace`, `np.pi`, `np.cumsum`, `np.finfo`, `np.isnan`, `np.logical_not`, `np.ones`, `np.arange`, `np.int_`, `np.max`, `np.min`, `np.tile`, `np.unravel_index`, `np.allclose`, `np.can_cast`, `np.copyto`, `np.generic`, `np.isclose`, `np.ravel_multi_index`, `np.reshape`, `np.result_type`, `np.einsum`, `np.equal`, `np.expand_dims`, `np.eye`, `np.isneginf`, `np.isposinf`, `np.multiply`, `np.nonzero`, `np.ones_like`, `np.ravel`, `np.split`, `np.stack`, `np.zeros_like`, `np.append`, `np.argmax`, `np.argmin`, `np.bytes_`, `np.count_nonzero`, `np.digitize`, `np.flipud`, `np.float128`, `np.floor`, `np.maximum`, `np.mean`, `np.median`, `np.meshgrid`, `np.number`, `np.random.PCG64`, `np.random.SeedSequence`, `np.var`
- **Not applicable (static typing):** `np.typing.NDArray`, `np.typing.ArrayLike`

| Gymnasium spelling | NumSharp spelling | Verified |
|---|---|---|
| `np.round(x)` | `np.round_(x)` / `np.around(x)` | Half-to-even, `-0.` preserved |
| `np.issubdtype(t, np.integer)` (also `floating`, `number`, `generic`) | `np.issubdtype(t, "integer")` | Identical for int32, int64, float32, float64, bool |
| `rng.uniform(low=array, high=array, size)` | `low + (high - low) * rng.random(size)` until an overload exists | Bit-identical |
| `np.random.Generator(np.random.PCG64(np.random.SeedSequence(s)))` | `np.random.default_rng(s)` | Bit-identical |

Gaps to close in NumSharp (F2):

1. `Generator.geometric`.
2. Array-parameter `uniform`/`normal`/`exponential`.
3. The `'S1'`/`'c'` dtype (optional).

`np.float128` has no .NET counterpart and follows the Windows float64 path.

---

## Appendix E: Rasterizer scope (pygame APIs)

Every pygame and `gfxdraw` call made by in-scope Gymnasium code, with the families that use it:

| API | Used by | License of algorithm |
|---|---|---|
| `gfxdraw.aapolygon`, `gfxdraw.filled_polygon` | box2d, classic_control | SDL_gfx (zlib) |
| `gfxdraw.aacircle`, `gfxdraw.filled_circle`, `gfxdraw.hline`, `gfxdraw.vline` | classic_control | SDL_gfx (zlib) |
| `draw.polygon`, `draw.circle`, `draw.aaline`, `draw.lines` | box2d | pygame (LGPL-2.1) |
| `draw.line`, `draw.aalines` | box2d, classic_control | pygame (LGPL-2.1) |
| `draw.rect` | box2d, toy_text | pygame (LGPL-2.1) |
| `Surface`, `surfarray.pixels3d` | box2d, classic_control, toy_text | pygame |
| `surfarray.make_surface` | utils, wrappers | pygame |
| `transform.flip` | box2d, classic_control | pygame |
| `transform.scale` | box2d, toy_text, utils | pygame / SDL |
| `transform.smoothscale` | box2d, classic_control | pygame |
| `image.load` | classic_control (Pendulum arrow), toy_text sprites | SDL_image / libpng |
| `font.Font`, `font.init`, `font.get_default_font` | box2d (CarRacing), toy_text (Blackjack) | SDL_ttf / FreeType |
| `math.Vector2` (`rotate_rad`) | box2d, classic_control | pygame (LGPL-2.1) |
| `display.*`, `event.*`, `time.Clock`, `quit` | human mode only | replaced by viewers |

---

## Appendix F: Python → C# semantics cheat-sheet

| Python | Pitfall | C# |
|---|---|---|
| `-7 // 2` = `-4` | C# `/` truncates toward zero: `-7 / 2 == -3` | `PyMath.FloorDiv(-7, 2)` |
| `-7 % 2` = `1` | C# `%` keeps the dividend's sign: `-7 % 2 == -1` | `PyMath.Mod(-7, 2)` |
| `a[-1]` | Wraps to the last element | `a[^1]` (as data, not as an "i − 1 with a floor at 0" rewrite) |
| `round(2.5)` = `2` | Half-to-even | `Math.Round(2.5, MidpointRounding.ToEven)` (the default) |
| `int(-2.7)` = `-2` | Truncation toward zero | `(long)(-2.7)` |
| `min(nan, 1)` = `nan`; `min(1, nan)` = `1` | Argument-order-dependent with NaN | `PyMath.Min`, not `Math.Min` |
| `0.001 * np.float32(x)` | NEP 50: stays float32 | Compute in `float`, then widen |
| `np.float32(x)` | Round-to-nearest-even from float64 | `(float)x` |
| `x ** 2` vs `np.square(x)` | Different functions | Mirror the source |
| `a == b` on arrays | Element-wise | Never use `==`/`!=` with `null` on `NDArray` |
| `dict` iteration | Insertion order | Ordered dictionary |
| `assert cond, msg` | Raises `AssertionError` | `throw new AssertionError(msg)` |
| `warnings.warn` | Once per call site by default | `Logger.Warn` with call-site dedup |
| Python `int` | Unbounded | `long`, with checks where the source can overflow |
| `True + 1` | Bools are ints | Explicit conversion |

---

## Appendix G: Assets and licenses

| Asset | Location in `refs/Gymnasium` | Used by | Status |
|---|---|---|---|
| 81 PNG sprites | `gymnasium/envs/toy_text/img/` | Blackjack, FrozenLake, CliffWalking, Taxi | Credits: Mel Tillery (cyaneus.com); Taxi rider from franuka.itch.io RPG asset pack. Redistribution terms to confirm (D14) |
| `Minecraft.ttf` | `gymnasium/envs/toy_text/font/` | Blackjack | License to confirm (D14) |
| `clockwise.png` | `gymnasium/envs/classic_control/assets/` | Pendulum | Ships with Gymnasium (MIT); confirm |
| 14 MuJoCo XML models | `gymnasium/envs/mujoco/assets/` | MuJoCo envs | Ships with Gymnasium (MIT); confirm per-file headers |
| pygame default font (`freesansbold.ttf`) | pygame-ce package | CarRacing text | GPL with font exception; confirm |

Code licenses:

- Gymnasium (MIT), Box2D (zlib), SDL_gfx (zlib), MuJoCo (Apache-2.0), OpenCV (Apache-2.0), NumSharp (Apache-2.0): compatible with Gym.NET's Apache-2.0 with notices.
- pygame-ce (LGPL-2.1): requires decision D4 before any port of its `draw` or `math` code.
