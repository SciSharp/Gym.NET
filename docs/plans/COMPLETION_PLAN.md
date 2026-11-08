# Gym.NET Completion Plan: from the core API to 2.0

| | |
|---|---|
| **Status** | Draft 1, 2026-11-08 |
| **Scope** | Everything [`GYMNASIUM_PORT_PLAN.md`](GYMNASIUM_PORT_PLAN.md) still has open, up to one 2.0 release that includes Box2D and MuJoCo (decision D21) |
| **Starting point** | branch `feat/numsharp-master-migration` @ `ed68964` (13 commits ahead of `master`, pushed 2026-09-27, no PR) |
| **Source of truth** | `refs/Gymnasium` @ `9e04324` today; `v1.4.0` (`8ba8c3a6`) after step R0.4 (D2) |
| **Array engine** | NuGet NumSharp 0.80.0 (released 2026-11-08); `refs/NumSharp` @ `v0.80.0` (`d523eb48`) after step R0.1 (D3) |
| **Per-area specs** | [`GYMNASIUM_PORT_PLAN.md`](GYMNASIUM_PORT_PLAN.md) §6–§13 (what each area must do), [`NUMSHARP_MIGRATION_PLAN.md`](NUMSHARP_MIGRATION_PLAN.md) §8 (array ownership), [`GYM_PARITY_INTEROP_TESTING.md`](GYM_PARITY_INTEROP_TESTING.md) (the live suite) |

This plan doesn't restate what the port plan specifies per area. It records where the work stands, what changed
around it since the last session, the decisions taken on 2026-11-08, and the order of everything left, with an exit
gate per step. Every number below was measured on 2026-11-08 on Windows 11 x64, .NET SDK 10.0.101, Python 3.12.12,
NumPy 2.4.2, pygame-ce 2.5.8 (SDL 2.32.10) and Box2D 2.3.10, unless it says otherwise.

---

## Contents

1. [Where the work stands](#1-where-the-work-stands)
2. [What changed since 2026-09-25](#2-what-changed-since-2026-09-25)
3. [Decisions](#3-decisions)
4. [The road to 2.0](#4-the-road-to-20)
5. [Phases](#5-phases)
6. [How every step is done](#6-how-every-step-is-done)
7. [NumSharp upstream items](#7-numsharp-upstream-items)
8. [Risks](#8-risks)

---

## 1. Where the work stands

Eight working sessions between 2026-09-23 and 2026-09-25 produced the plans and these 13 commits on
`feat/numsharp-master-migration`:

| Commit | What it did |
|---|---|
| `801f123`, `c110cbc`, `c141717`, `684ca1d` | NumSharp migration steps 1–3: built against NumSharp master through `refs/NumSharp` (`eng/NumSharp.props`), `long` seeds, indices and actions, a generator per env and space, the NuGet reference declared next to the import |
| `2a1b827`, `ede57bd`, `cf0dd77` | The byte-parity audit ([`GYMNASIUM_PARITY_AUDIT.md`](GYMNASIUM_PARITY_AUDIT.md)) and the LunarLander memory-safety fix it found |
| `b51689d` | Gymnasium's seeding: `Generator(PCG64(SeedSequence(seed)))`, `BigInteger` seeds |
| `c08a72b`, `f8a6e11` | The live parity suite `tests/Gym.Tests.Parity` (embedded CPython over pythonnet), rebuilt on the pythonic skill |
| `0fa7bb2` | CartPole byte for byte, frames pixel for pixel, through the new `Gym.Rendering` (an SDL_gfx 2.0.26 port) |
| `da428ab` | P1, the core API: `Env<TObs, TAct>`, wrappers, `TimeLimit`, `OrderEnforcing`, `EnvSpec`, `Info`, every `error.py` class, `PyMath`, `Gym.Legacy` |
| `ed68964` | float64 rewards everywhere, `ValueTask StepAsync`, documentation in Gym.NET's own voice |

State at `ed68964` (reported 2026-09-25): unit tests 212/212 (203 headless plus 9 viewer tests) on net8.0 and net10.0;
live parity 163/163 gated cells on both; 14 `[OpenBugs]` cells, 6 for spaces (P2) and 8 for LunarLander (P8).

Done in the port plan's terms: P1, the CartPole slice of P4 and P7, F2 (seeding) and F3 (`PyMath`). Not done: the rest
of P0 (CI, static rules, the golden harness, the inventory tool), migration step 4 (array ownership; no `Space` is
`IDisposable` and nothing in `src/` is `[NDBorrowed]`, checked 2026-11-08), and P2–P10.

## 2. What changed since 2026-09-25

### 2.1 NumSharp 0.80.0 is released

NumSharp 0.80.0 reached nuget.org on 2026-11-08, 1,625 commits after the `93967990` that `refs/NumSharp` pins. It
carries the `DType` API that kept Gym.NET on a source build, so the port plan's §4.4 flip to the package is possible.
A trial in a scratch export of `ed68964` (Package mode, Release, net8.0 and net10.0) measured what the switch costs:

| Gate | Result against 0.80.0 | Cause and fix |
|---|---|---|
| Compile | 4 × CS0104 | `AssertionError` is now also `NumSharp.AssertionError`, the collision D17 predicted. Delete `Gym.Exceptions.AssertionError` and throw NumSharp's |
| Compile | 1 × CS0426 | `PCG64.Pcg64StateData` is now `PCG64.State`, whose `has_uint32` is an `int` (`tests/Gym.Tests.Parity/Oracle/RngFacts.cs`) |
| Compile | after both fixes, with NuGet audit and packing off | 0 errors, 0 warnings; the 0.80.0 ownership analyzer reports no NDW site |
| Pack | NU5048, NU5125 as errors | Package mode packs on build, and the deprecated `PackageIconUrl`/`PackageLicenseUrl` fail it (R0.3) |
| Unit tests (headless) | 202/203 on both frameworks | NumSharp's parameterless `AssertionError()` says `"AssertionError"`; Python's `str(AssertionError())` is `""`. Bare-assert sites pass `""` |
| Parity gate | 150/163 on both frameworks | 9 `DrawPatterns` cells still wrapped Python's `choice()` result in `np.atleast_1d` to match NumSharp's old `(1,)` shape, which 0.80.0 fixed: the workaround goes. 4 cells map Python's `AssertionError` to `Gym.Exceptions.AssertionError` |
| `[OpenBugs]` | 14/14 fail, each at its documented divergence | unchanged |

A probe of the calls the coming phases need, NumSharp 0.80.0 against NumPy 2.4.2 for seeds 0, 1, 42 and 1234, matched
on every value: `Generator.geometric(0.25)` (unblocks `spaces.Sequence`), `uniform` with array bounds (`Box.sample`,
Pendulum's reset), `integers` over the 2^31–2^32 range, and `choice` without a size returning a 0-d scalar.
`SeedSequence().entropy` is a `BigInteger` now (it was four 32-bit words). One gap remains: `np.array(string[], "c")`
gives shape `(4,)` where NumPy's `np.asarray(desc, dtype="c")` splits each string into characters, `(4, 4)` of `S1`.
FrozenLake and Taxi build their maps that way (§7, N-2).

### 2.2 Gymnasium v1.4.0 is released

Gymnasium tagged v1.4.0 (`8ba8c3a6`) on 2026-10-05, 24 commits after the `9e04324` dev commit we pin; `main` is 41
commits further. What the 24 commits change, by our phase:

| Area | Change | Effect on Gym.NET |
|---|---|---|
| `core.py`, `cartpole.py` | `TypeVar` defaults, type hints, and a docs note that `reset`/`step` must return new objects (#1683) | No behavior change; P1 and CartPole stay as they are |
| `registration.py` | `_check_metadata`'s message names the right variable; wrapper classes as entry points (#1672) | P3 ports the fixed version |
| toy_text | New Farama pixel art for all four envs (#1713, #1732, #1733, #1744); `Minecraft.ttf` and the 52 card PNGs deleted (77 assets removed, 97 added); Blackjack draws its numbers from digit images | P5 needs no font, so FreeType is needed only for CarRacing (P8) |
| `spaces/utils.py`, `env_checker.py`, `passive_env_checker.py`, `env_match.py` | `OneOf` flatten agreement, Tuple-length compatibility, determinism snapshots, RNG-state comparison | P2 and P3 port the fixed versions |
| `vector/*`, `wrappers/*` | `AsyncVectorEnv(max_concurrency)`, `DiscretizeObservation`/`DiscretizeAction` on n-d boxes, `ResizeObservation` singleton channels, `DtypeObservation` descriptors | P6 ports the fixed versions |

Ported code cites Gymnasium in 142 permalinks across 35 files, all at `9e04324`. Four of the cited sources changed
(`core.py`, `cartpole.py`, `registration.py`, `wrappers/common.py`).

Gymnasium's own test counts at v1.4.0, the exit gates of the coming phases: `tests/spaces` 109 test functions in 13
files, `tests/wrappers` 146 in 50, `tests/vector` 76 in 9, `tests/envs` 116 in 18, `tests/utils` 61 in 9,
`tests/functional` 11 in 4, `tests/test_core.py` 9 (ported).

### 2.3 ImageSharp 2.1.13 has five advisories, and the build fails

On 2026-10-07 GitHub published five advisories against `SixLabors.ImageSharp` 2.1.13, the last Apache-2.0 release:
GHSA-j3p4-wp97-rph4, GHSA-j9gm-c75j-xc9q and GHSA-jjfr-hcj7-qf5w (high), GHSA-gwg2-r3hj-4w44 and GHSA-wmxv-xphr-5c9g
(moderate). They sit in HistogramEqualization, the TIFF CCITT T4/T6 encoders, ICC CLUT parsing and BigTIFF IFD
parsing; the fixes are in 3.2.0 and 4.1.2, both under the Six Labors Split License. NuGet audit turns them into
NU1902/NU1903, which `TreatWarningsAsErrors` makes errors in `Gym.Rendering` and `Gym.Tests.Parity`: **restore fails
today, on `ed68964` and on `master` alike**, independent of NumSharp.

Gym.NET uses ImageSharp to hand frames to the Avalonia and WinForms viewers, for the legacy `Render()` result, for the
Aether LunarLander's drawing (ImageSharp.Drawing) and to write the parity suite's PNG artifacts. None of these encodes
TIFF, reads BigTIFF or runs HistogramEqualization, and Gym.NET decodes no image file today. P7.4 will decode
Gymnasium's own PNG assets, which is the one route to the ICC-profile parser, and only for trusted files shipped with
Gymnasium.

## 3. Decisions

### 3.1 Taken on 2026-11-08

| # | Decision | Taken | Consequence for this plan |
|---|---|---|---|
| D2 | Gymnasium pin | **Re-pin to v1.4.0 now** | R0.4, before any further port |
| D3 | NumSharp consumption | **NuGet 0.80.0 by default; `-p:NumSharpSource=Submodule` kept** for co-developing NumSharp fixes, with `refs/NumSharp` at `v0.80.0` | R0.1 |
| D4 | pygame `draw.*` (LGPL-2.1) | **Gym.NET's own implementations**, proven pixel-exact against live pygame-ce by the parity suite; no LGPL code enters Gym.NET | P7 |
| D5 | Box2D | **Managed C# port of the Box2D 2.3.x core** that `box2d==2.3.10` compiles, float32, C++ operation order kept, proven by engine goldens first | P8 |
| D10 | The Aether LunarLander | Removed when LunarLander-v3 lands in P8, which is before 2.0 (follows from D21) | P8 |
| D18 | ImageSharp advisories | **Suppress the five advisories and keep ImageSharp 2.1.13** | R0.2; consumers of the packages still see the advisories in their own restores (§8) |
| D19 | Array ownership (migration step 4, and NUMSHARP_MIGRATION_PLAN's M5) | **Explicit ownership plus `[NDScoped]`**: spaces own and dispose their arrays, carried arrays are `[NDBorrowed]`, `StepResult`/`ResetResult` implement `INDArrayCarrier`, and ported methods' temporaries are reclaimed by `[NDScoped]`, woven by NumSharp.Build (a development dependency, never in consumers' graphs) | P2.1, then every phase |
| D20 | Branch and CI | **One branch until 2.0**: all work stays on `feat/numsharp-master-migration` and merges into `master` once, at the release. CI runs on the branch, informational | P0r.1, P10.5 |
| D21 | Release cut | **One 2.0 with everything**: P2–P9, Box2D and MuJoCo included, then P10 | §4 |

### 3.2 Still open, with the default that applies unless the maintainer overrides it

| # | Decision | Default | Needed by |
|---|---|---|---|
| D6 | MuJoCo version | The newest MuJoCo 3.x release at P9's start that Gymnasium v1.4.0's mujoco tests pass with (its `pyproject.toml` only asks `mujoco >=2.1.5`), recorded in `tools/parity/manifest.json` | P9 |
| D7 | `AsyncVectorEnv` model | Threads; results equal `SyncVectorEnv` bit for bit | P6 |
| D9 | Legacy shim | `Gym.Legacy` stays through 2.x, `[Obsolete]` | P10 |
| D13 | `examples/` | Rewrite one CartPole example on the new API, remove the rest | P10 |
| D14 | toy_text assets | Ship v1.4.0's Farama pixel art once its license is confirmed in R0.4; until then load it from `refs/Gymnasium` | P5 |
| D15 | Golden storage | In the repo as gzip JSONL; Git LFS only for MuJoCo | P0r.3 |
| D22 | Text rendering for CarRacing (pygame's default font) | FreeType through P/Invoke at the version pygame-ce 2.5.8 bundles (FreeType License, permissive with attribution) | P8 |
| — | The last session's three open points | (1) The two concurrency tests that start threads with `Task.Run` move to explicit `Thread`s; (2) the SDL_gfx header keeps "C# port of", which zlib's clause 2 requires of altered versions; (3) the `Gym.Tests.Ported` namespace and folder become `Gym.Tests.Gymnasium` (they mirror Gymnasium's test suite), and the README/PRD tagline "A port of openai/gym" is rewritten in P10 | R0.5, P10 |

## 4. The road to 2.0

| Phase | Deliverables | Exit gate | Size |
|---|---|---|---|
| **R0 Re-baseline** | NumSharp 0.80.0, the ImageSharp suppressions, package metadata, Gymnasium v1.4.0, housekeeping | Builds clean; unit 212/212; parity 163/163 gated, 14 `[OpenBugs]` failing as documented, all on both frameworks | S |
| **P0r P0's rest** | CI (informational), static rules, golden harness (F4), inventory tool | CI green on the branch; CartPole-v1 golden passes; the inventory test reports the missing-names list | M |
| **P2 Spaces** | 10 spaces, `spaces.utils`, `Space<T>`, array ownership (D19) | `tests/spaces` (109) green; live cells for every space; 0 space `[OpenBugs]` | L |
| **P3 Registry and checkers** | `Gymnasium.Make` and friends, `PassiveEnvChecker`, `check_env`, `env_match`, `Autoreset`, `RecordEpisodeStatistics` | Registration and checker tests green; Gymnasium's `make("CartPole-v1")` stack equals ours, built by `Gymnasium.Make` | M |
| **P7 Rasterizer** | pygame `draw.*` (D4), `transform.scale`/`smoothscale`, `image.load`, the frame path for every family | 150+ random shapes per primitive pixel-exact against pygame | L |
| **P4 classic_control** | MountainCar, Continuous_MountainCar, Pendulum, Acrobot, `CartPoleVectorEnv` | L3 and L4 for all 6 ids | M |
| **P5 toy_text** | Blackjack, FrozenLake, CliffWalking, Taxi with `ansi` and v1.4.0's art | L3 and L4 for all 6 ids | M |
| **P6 Vector and wrappers** | `VectorEnv`, `SyncVectorEnv`, `AsyncVectorEnv`, vector utils, 35 wrappers, vector wrappers, F8, F9 | `tests/vector` (76) and `tests/wrappers` (146) green; async equals sync bitwise | L |
| **P8 Box2D** | Box2D.NET (D5), LunarLander, BipedalWalker, CarRacing, fonts (D22); Aether removed | Engine goldens; L3 and L4 for all 5 ids; 0 `[OpenBugs]` | XL |
| **P9 MuJoCo** | MuJoCo.NET bindings, `MujocoEnv`, `MujocoRenderer`, 22 ids | Hopper-v5 golden, then all 22 at L3 | XL |
| **P10 Release 2.0.0** | Docs, migration guide, examples, packages; merge into `master` | The port plan's definition of done (§1.5), whole | M |

The order is sequential. P7 moves in front of P4 because MountainCar and Acrobot already call `pygame.draw`, and P5
needs `draw.rect`, `image.load` and `transform.scale`.

## 5. Phases

Each numbered step is one commit, unless it says otherwise.

### R0 Re-baseline

- **R0.1 NumSharp 0.80.0.**
  - Raise `NumSharp` to 0.80.0 in the five projects that declare it, and `NumSharp.Interop.pythonnet` to 0.80.0 in
    the parity suite.
  - Make `NumSharpSource=Package` the default in `eng/NumSharp.props`. Submodule mode stays, with its guards and the
    analyzer reference, and its comments say why it now exists: to build a NumSharp fix before NumSharp releases it.
  - Check out `refs/NumSharp` at `v0.80.0` (`d523eb48`) and record the new gitlink.
  - Delete `Gym.Exceptions.AssertionError`; every site throws `NumSharp.AssertionError`, and bare asserts pass `""`
    (N-1).
  - Switch `RngFacts` to `PCG64.State`. Map Python's `AssertionError` to NumSharp's type in the parity harness.
  - Remove the workarounds for the bugs 0.80.0 fixed:
    - `Discrete`'s unwrapping of the one-element array `choice()` used to return;
    - `Seeding.Entropy`'s four-word branch, now that an unseeded `SeedSequence` holds a `BigInteger`, and with it
      `SeedTests`' inconclusive case for a zero top word;
    - the parity harness's `np.atleast_1d` around `choice()` (`SeedingLiveParityTests`, the 9 failing cells).
  - Update `NUMSHARP_MIGRATION_PLAN.md`'s status and §14, and the root `CLAUDE.md`.
- **R0.2 ImageSharp advisories (D18).** A root `Directory.Build.props`, new and holding only the five
  `NuGetAuditSuppress` items, each with its advisory's title and the reason it doesn't apply. The comment also records
  that the suppression covers only our build: consumers' restores still report the advisories. A new advisory against
  2.1.13 fails the build again, by design, and gets the same review.
- **R0.3 Package metadata.** `PackageLicenseExpression` `Apache-2.0`, a packed `PackageIcon`, and a `PackageReadmeFile`
  replace `PackageLicenseUrl`/`PackageIconUrl` in the five packable projects (NU5048/NU5125). `JetBrains.Annotations`
  gets `PrivateAssets="all"`, and `Gym.csproj` drops `SixLabors.ImageSharp.Drawing`, which nothing in `src/Gym` uses
  (both found by the 2026-09-23 NuGet review).
- **R0.4 Gymnasium v1.4.0 (D2), by the port plan's §14.10.**
  - Check out `refs/Gymnasium` at `v1.4.0` and update `tools/parity/manifest.json`.
  - A script, `tools/parity/repin_references.py`, moves every permalink at `9e04324` to the new commit. It maps line
    numbers through `git diff 9e04324..v1.4.0` and refuses any link whose target line's text changed, so those get
    reviewed by hand.
  - Port what the four changed sources mean for ported code: `_check_metadata`'s message, and the core docs note.
  - Confirm the license of the new toy_text art (D14).
  - Re-run every gate. A cell whose verdict moves is reviewed, never re-recorded blindly.
- **R0.5 Housekeeping.** The two `Task.Run` concurrency tests move to explicit threads; `Gym.Tests.Ported` becomes
  `Gym.Tests.Gymnasium` (§3.2).
- **Exit:** Debug and Release build with 0 errors on net8.0 and net10.0; unit 212/212 (203 headless + 9 viewer);
  parity 163/163 gated; the 14 `[OpenBugs]` cells fail at their documented divergences and nowhere else.

### P0r P0's rest

- **P0r.1 CI (D20, informational).** GitHub Actions on the branch:
  - `windows-latest` (the reference) builds Debug and Release, runs the headless unit suite, sets up the pinned
    Python stack with `tools/parity/setup_venv.ps1`, and runs the parity gate with `GYMNET_PYTHONNET_REQUIRE_ENGINE=1`.
  - `ubuntu-latest` builds and runs the unit suite.
  - Both set `AVALONIA_TELEMETRY_OPTOUT=1` and exclude the viewer tests.
- **P0r.2 Static rules (port plan §14.7)** as a script that CI runs: float32 literals and `MathF` in ported math,
  `Debug.Assert`, `Console.WriteLine`, sync-over-async, `lock (this)`, the `Thread.Abort` family, global or unseeded
  RNG, `throw new Exception`, public mutable constant-like fields, `NDArray == null`. Each rule gets a fixture that must
  trip it.
- **P0r.3 Golden harness (F4).**
  - `tools/golden/record.py` records trajectories and frames from the pinned stack into `tests/Golden` (D15).
  - The C# side gets `GoldenTrajectory`/`GoldenAssert`, with open-loop and sampled action modes.
  - One case catalog drives the live, record and replay modes (the interop design's §9), so a golden is always a
    recording of a live cell.
  - Goldens let the ubuntu job and machines without Python check L3/L4. The live suite stays the oracle on the
    reference job.
- **P0r.4 Inventory (port plan §14.6).**
  - `tools/parity/inventory.py` lists Gymnasium's public names by AST.
  - A reflection test maps them through §3.8 and reports the missing ones against a committed expected-missing list.
    Each phase shrinks the list, and 2.0 needs it empty.
- **Exit:** both CI jobs green on the branch; the CartPole-v1 golden passes; the inventory test runs.

### P2 Spaces

Specs: the port plan's §7; sources `gymnasium/spaces` (3,816 lines at v1.4.0); tests `tests/spaces`.

- **P2.1 `Space<T>` and ownership (D19).**
  - The generic base: lazy `NpRandom`, `Seed` returning the seed list, `ToString`/`Equals`, `to_jsonable`/`from_jsonable`.
  - `Space : IDisposable`; `Env.Dispose` disposes both spaces after `Close`.
  - `[NDBorrowed]` on carried arrays; `StepResult`/`ResetResult` implement `INDArrayCarrier`.
  - NumSharp.Build added as a development dependency, so `[NDScoped]` methods are woven.
  - Measure what `[NDScoped]` saves on `Box.Sample` and CartPole's step: NUMSHARP_MIGRATION_PLAN §8.4 measured
    0.634 µs per undisposed small array against 0.217 µs disposed.
  - NDW findings become errors in `src/` (the migration plan's M6).
- **P2.2 `Box`**: validation with Gymnasium's messages, float64 bounds kept in float64 (closes the `Box(float, float)`
  defect), sampling per bound category, `contains` with `np.can_cast` and lists, `repr`.
- **P2.3 `Discrete`**: shape `()`, int64, `start`, masks and probabilities, `repr`, `n > 0` validation (B-03, and
  B-10 as `Discrete`'s cell tags it).
- **P2.4 `MultiBinary`, `MultiDiscrete`.**
- **P2.5 `Tuple`, `Dict`, `Sequence`**: subseeds through `integers(int32.MaxValue, size: n)`, `geometric(0.25)` for
  `Sequence`, which NumSharp 0.80.0 now provides.
- **P2.6 `Text`, `Graph`, `OneOf`.**
- **P2.7 `spaces.utils`**: `flatdim`, `flatten`, `unflatten`, `flatten_space`, `is_space_dtype_shape_equiv`, with
  v1.4.0's `OneOf` and `Tuple` fixes.
- **P2.8 Tests**: port `tests/spaces` (109 functions, 13 files) under `tests/Gym.Tests/Gymnasium/Spaces`. Add live
  cells for every space (samples with generator state, `contains`, `repr`, `flatten`, JSON) and remove the `[OpenBugs]`
  attribute from the six space cells (tagged B-03, B-10 and P2) as they turn green.
- **Exit:** `tests/spaces` green on both frameworks; every space has live cells; no space `[OpenBugs]` left; the
  inventory list loses every `gymnasium.spaces` name.

### P3 Registry and checkers

Specs: the port plan's §8, §9 (the `make()` stack) and §12.

- **P3.1 Registration**: `Gymnasium.Make`, `MakeVec` (scaffold), `Spec`, `Register`, `Registry`, `PprintRegistry`,
  `RegisterEnvs`, with v1.4.0's entry-point and metadata fixes and exact error texts. CartPole-v0/v1 register now;
  each later phase registers its ids.
- **P3.2 Checkers**: `utils/passive_env_checker.py` and the `PassiveEnvChecker` wrapper, `utils/env_checker.py`
  (`check_env`, including v1.4.0's distinct-object and RNG-state checks), `utils/env_match.py`.
- **P3.3 Common wrappers**: `Autoreset`, `RecordEpisodeStatistics` (its time entry excluded from goldens), vector
  `step_api_compatibility`.
- **P3.4 Tests**: port `tests/envs/registration`, the checker tests in `tests/utils`, and the common-wrapper tests.
  Parity cells switch from the hand-built `TimeLimit(OrderEnforcing(env))` to `Gymnasium.Make`.
- **P3.5 The reverse bridge** (the interop design's §8): a Python env class over a Gym.NET env, so Gymnasium's own
  `check_env` and `check_environments_match` run against Gym.NET's envs in the live suite. Every later env phase adds
  its ids to it.
- **Exit:** those tests green; Gymnasium's `make("CartPole-v1")` and `Gymnasium.Make("CartPole-v1")` agree for whole
  episodes, `PassiveEnvChecker` included; both our `check_env` and Gymnasium's pass on CartPole.

### P7 Rasterizer, ahead of P4

Specs: the port plan's §5.5 and Appendix E. pygame-ce calls at v1.4.0:

| Family | `pygame.draw` (D4: our own code) | Other |
|---|---|---|
| classic_control | `aalines` ×2, `line` ×1 | `transform.smoothscale`, `image.load` (Pendulum), `transform.flip`, `Vector2` |
| toy_text | `rect` ×1 | `image.load` ×5, `transform.scale` ×3, `transform.flip` |
| box2d | `polygon` ×13, `circle` ×4, `lines`, `line`, `aaline` ×2 | `transform.scale`, `transform.smoothscale`, `font` |

- **P7.1** `line`, `aaline`, `lines`, `aalines`.
- **P7.2** `polygon`, `circle`, `rect`.
- **P7.3** `transform.scale` and `transform.smoothscale` (also pygame's own code, so also D4).
- **P7.4** `image.load`: PNG decoded through ImageSharp, which stays per D18. PNG is lossless, so the pixels are the
  file's; a cell proves it against pygame for every asset the envs load.
- Each primitive is proven the way gfxdraw was: 150+ random shapes per primitive, with clipping, translucency, widths
  and Sint16 wrap-around, compared pixel for pixel with live pygame-ce 2.5.8. A primitive that won't converge is
  reported back to the maintainer before any other route is taken (D4).
- **Exit:** every primitive above is pixel-exact on its random corpus.

### P4 classic_control

Specs: the port plan's §11.1; sources 2,028 lines.

- **P4.1** MountainCar-v0 and MountainCarContinuous-v0.
- **P4.2** Pendulum-v1: `np.sin(th)`, the NEP 50 float32/float64 mix in its reward, array-bound `uniform` in `reset`,
  `clockwise.png`.
- **P4.3** Acrobot-v1: the RK4 integrator, `wrap` and `bound`.
- **P4.4** `CartPoleVectorEnv` (with P6's `VectorEnv` base, pulled forward if needed, as P1 pulled `TimeLimit`
  forward).
- Each env: live cells for whole episodes (policy and random actions), reset options, invalid actions with Python's
  messages, frames along episodes and at extreme states; goldens recorded; registration.
- **Exit:** L3 and L4 for all six ids, live and golden.

### P5 toy_text

Specs: the port plan's §11.2, with v1.4.0's sources (1,868 lines) and art.

- **P5.1** `categorical_sample`, and the `'c'` map helper: N-2 fixed upstream, or a local builder from per-character
  rows until it is.
- **P5.2** FrozenLake-v1 and FrozenLake8x8-v1: `generate_random_map`, slippery transitions, `ansi`.
- **P5.3** CliffWalking-v1 and CliffWalkingSlippery-v1.
- **P5.4** Taxi-v4: action masks in `info`.
- **P5.5** Blackjack-v1: digit images, no font.
- **Exit:** L3 and L4 for all six ids; `ansi` output identical to Gymnasium's text, byte for byte.

### P6 Vector and wrappers

Specs: the port plan's §9 and §10; sources `vector` 3,113 lines and `wrappers` 7,569 lines at v1.4.0.

- **P6.1** `VectorEnv`, `AutoresetMode` (NEXT_STEP, SAME_STEP, DISABLED), `SyncVectorEnv`, the vector utils
  (`batch_space`, `batch_differing_spaces`, `iterate`, `concatenate`, `create_empty_array`).
- **P6.2** `AsyncVectorEnv` on threads (D7), with v1.4.0's `max_concurrency`; bit-equal to sync.
- **P6.3** Observation, action and reward wrappers, in the port plan's §9 groups.
- **P6.4** F8: `cv2.resize(INTER_AREA)` for `ResizeObservation`, byte-exact against recorded OpenCV outputs.
- **P6.5** Rendering wrappers; F9: `RecordVideo` over an FFmpeg wrapper (frame parity, not file parity),
  `HumanRendering` on the viewers.
- **P6.6** Vector wrappers.
- **P6.7** Tests: port `tests/vector` (76) and `tests/wrappers` (146).
- **Exit:** both suites green; every vector cell equal between sync and async, bit for bit.

### P8 Box2D

Specs: the port plan's §5.6 and §11.3; sources 2,979 lines.

- **P8.1 Box2D.NET (D5):** settings, math, the dynamic tree and pair ordering, distance, TOI, manifolds, contact
  solver, island, joints, ray casts, and the contact-listener callback order. Engine goldens come from live pybox2d
  scenes (free fall, stacking, joints, contacts, ray casts), compared per step in float32 bits. The first goldens also
  settle whether .NET's `MathF` trigonometry matches the CRT calls Box2D makes (§8).
- **P8.2** LunarLander-v3 and LunarLanderContinuous-v3, with `heuristic`; the 8 LunarLander `[OpenBugs]` cells turn
  green.
- **P8.3** BipedalWalker-v3 and BipedalWalkerHardcore-v3 (lidar ray casts).
- **P8.4** CarRacing-v3 with `car_dynamics` and text (D22); coordinate with PR #13.
- **P8.5** Delete the Aether LunarLander (D10) and the `Aether.Physics2D`, `ImageSharp.Drawing` and `Fonts`
  references in `Gym.Environments`; B-17…B-30 close.
- **Exit:** engine goldens; L3 and L4 for all five ids; no `[OpenBugs]` cell left in the suite.

### P9 MuJoCo

Specs: the port plan's §5.7 and §11.4; sources 6,153 lines.

- **P9.1 MuJoCo.NET:** P/Invoke bindings generated from the pinned `mujoco.h` (D6), native packages per RID from the
  official release, and an offscreen GL context (WGL on Windows, EGL on Linux).
- **P9.2** `MujocoEnv` and `MujocoRenderer`; the 14 XML assets.
- **P9.3** Hopper-v5 golden first, then the other 21 ids (11 envs × v4/v5), with v5's constructor options.
- **Exit:** all 22 ids at L3 on the pinned native build; frames best-effort, as the port plan's §1.2 allows.

### P10 Release 2.0.0

- **P10.1 Docs.** Replace PR #24's `PRD.md` and `ROADMAP.md` (root and `docs/`), `docs/architecture`, `docs/sot` and
  `docs/plans/expressive-nibbling-goose.md` with docs derived from the plans, in Gym.NET's own voice; rewrite the
  README.
- **P10.2** The migration guide (port plan §16); `Gym.Legacy` stays (D9).
- **P10.3** One CartPole example on the new API; the rest removed (D13, B-36).
- **P10.4** Packages at 2.0.0: Gym.NET, .Rendering, .Environments, .Environments.ToyText, .Environments.Box2D,
  .Environments.MuJoCo, .Rendering.Avalonia, .Rendering.WinForm, Box2D.NET, MuJoCo.NET. Release notes include D18's
  advisory note.
- **P10.5** Merge `feat/numsharp-master-migration` into `master` (D20), tag, publish.
- **Exit:** the port plan's definition of done (§1.5) in full; the inventory's expected-missing list is empty.

## 6. How every step is done

- **Source first.** Port from the pinned `refs/Gymnasium` (and pygame-ce, CPython, NumPy, Box2D, MuJoCo sources at
  their pins), following the port plan's §3 rules; each member based on a source carries its pinned GitHub
  permalink in `<remarks>`, in Gym.NET's own voice.
- **Documented.** Every public and internal member gets full XML documentation (summary that states consequences,
  every parameter, return and exception), as the repository's rules require.
- **Proven twice.** The ported Gymnasium tests, and live parity cells against Gymnasium itself. Every new cell family
  is shown to catch a planted defect (one at a time, then reverted), as P1 did.
- **Every configuration.** Debug and Release, net8.0 and net10.0: the unit suite, the parity gate, and the
  `[OpenBugs]` run, whose cells must fail only at their documented divergence.
- **Recorded.** A gap that can't be closed in the step becomes an `[OpenBugs("B-xx")]` cell with a register row. The
  commit message carries the gate numbers. The port plan's progress section and this plan's step get a status line,
  and the catalog's changelog gets a line.

## 7. NumSharp upstream items

What the earlier plans asked of NumSharp, and what 0.80.0 did with it:

| Item | Origin | 0.80.0 | Evidence |
|---|---|---|---|
| `Generator.geometric` | port plan F2.1 | Fixed | Probe: identical to NumPy 2.4.2 for 4 seeds |
| Array-parameter `uniform`/`normal`/`exponential` | F2.2, U6 | Fixed | Probe: `uniform` identical; the release notes cover all 29 samplers |
| `'S1'`/`'c'` dtype | F2.3 | `S1` exists; the `'c'` split from strings does not (N-2) | Probe: shape `(4,)` against NumPy's `(4, 4)` |
| `AssertionError`, `NotImplementedError` in NumSharp | F2.4, D17 | `AssertionError` added (N-1); `NotImplementedError` still absent, so Gym.NET keeps its own | Trial build: 4 × CS0104 |
| A release with the `DType` API | U1 | Fixed | Trial build |
| Unseeded `RandomState` from OS entropy | U2 | Fixed (release notes) | Not re-measured: Gym.NET no longer uses `RandomState` |
| `RandomState(long)` | U3 | Moot | Gym.NET seeds through `SeedSequence` |
| A lock per generator | U4 | Fixed for `Generator` (release notes) | Not re-measured |
| `randint` defaults to int64 | U5 | Fixed (release notes) | — |
| `NumPyRandom.Seed` wraps at 2^31 | U7 | Fixed: `uint` (release notes) | — |
| Finalizer cost of small arrays | U8 | Not re-measured | P2.1 measures it with `[NDScoped]` |
| NumSharp.Core's warnings in a source build | U9 | Moot in Package mode | — |
| `Generator.choice(array)` returns shape `(1,)` | 2026-09-25 audit | Fixed | Probe: 0-d; 9 parity cells now need their workaround removed |
| `SeedSequence()` entropy as four 32-bit words | 2026-09-25 | Fixed: `BigInteger` | Probe |
| `GetSingle`/`GetDouble` unchecked in Release | 2026-09-25 | Fixed: element accessors check every coordinate (release notes) | — |
| Legacy `choice` ignores `replace`; `randint(-5, -1)` read as "no high" | 2026-09-23 | `replace` fixed (release notes); `high = -1` not re-checked | Legacy API, unused by Gym.NET |
| Legacy `randint` diverges for 2^31–2^32 | 2026-09-25 | Moot for Gym.NET | Probe: `Generator.integers` identical over that range |

New items:

- **N-1.** NumSharp's parameterless `AssertionError()` carries the message `"AssertionError"`; Python's
  `str(AssertionError())` is `""`. Gym.NET passes `""` at its bare asserts (R0.1); whether NumSharp should match
  `str()` is NumSharp's call.
- **N-2.** `np.array(string[], "c")` (and `np.asarray`) should split each string into characters, as NumPy does for the
  `'c'` code: `np.asarray(["SFFF", "FHFH", "FFFH", "HFFG"], dtype="c")` is a `(4, 4)` `S1` array. P5 needs it, or
  works around it.
- **N-3.** `NotImplementedError` (D17's recommendation that NumSharp define every Python built-in), low priority.

## 8. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| ImageSharp 2.1.13 stays flagged (D18) | Consumers' restores report three high and two moderate advisories; under `TreatWarningsAsErrors` their builds fail | Release notes and README name the five advisories and why Gym.NET's use doesn't reach them; a new advisory gets the same review (R0.2) |
| One branch until 2.0 (D20) | A long-lived branch with many commits merges once; contributors keep targeting `master` | The branch sits on `master`'s tip (13 ahead, 0 behind), so while `master` doesn't move the final merge is a fast-forward; CI runs on the branch from P0r |
| pygame `draw` from our own code (D4) | A primitive that won't converge stalls P7 | The random-shape harness reports the first differing pixel and its neighbourhood; the maintainer decides before any other route |
| Box2D float32 trigonometry (D5) | Box2D calls `sinf`/`cosf`/`atan2f` from MSVC's CRT; .NET's `MathF` must agree on win-x64 | The first engine goldens include rotation-heavy scenes; a mismatch surfaces at P8.1, before any env |
| MuJoCo native builds (D6) | L3 depends on the exact native binary | Pin the version, ship the official binaries per RID, record them in the manifest |
| FreeType text for CarRacing (D22) | Text pixels depend on FreeType's version and hinting | Bind the FreeType version pygame-ce 2.5.8 ships; limited to one env |
| Gymnasium moves past v1.4.0 | Work against a superseded pin | Re-pin only at a release tag, by the port plan's §14.10, between phases |
