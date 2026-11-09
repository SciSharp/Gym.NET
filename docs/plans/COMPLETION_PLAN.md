# Gym.NET Completion Plan: from the core API to 2.0

| | |
|---|---|
| **Status** | Draft 2, 2026-11-08: the implementation designs are reviewed and approved in [`DESIGN_REVIEW.md`](DESIGN_REVIEW.md) (DR-01…DR-36). **A step starts only when every design it depends on is approved there** |
| **Progress** | Approved to run on 2026-11-09: R0, P0r, P2, P3, P7, P4, P5, P6 (what is necessary; the rest deferred and listed), then P10 prepared without publishing or pushing. Done: R0.2 (`e4e5215`), R0.1 + R0.3 (`631e0cf`), R0.4 (`0b3dee8`), R0.5 (`c80bc2d`, `f9d4dc4`), P0r.3 (`339b3b3`), P0r.4 and P0r.1 (2026-11-09) |
| **Scope** | Everything [`GYMNASIUM_PORT_PLAN.md`](GYMNASIUM_PORT_PLAN.md) still has open, up to one 2.0 release that includes Box2D (D21); MuJoCo is postponed out of 2.0 (DR-35) |
| **Starting point** | branch `journey1` @ `7b7c919`, cut from `feat/numsharp-master-migration` (13 commits ahead of `master`, pushed 2026-09-27, no PR); all work continues on `journey1` until 2.0 (D20) |
| **Source of truth** | `refs/Gymnasium` @ `v1.4.0` (`8ba8c3a6`) since step R0.4 (D2, 2026-11-09); `9e04324` before |
| **Array engine** | NuGet NumSharp 0.80.0 (released 2026-11-08); `refs/NumSharp` @ `v0.80.0` (`d523eb48`) after step R0.1 (D3) |
| **Per-area specs** | [`GYMNASIUM_PORT_PLAN.md`](GYMNASIUM_PORT_PLAN.md) §6–§13 (what each area must do), [`NUMSHARP_MIGRATION_PLAN.md`](NUMSHARP_MIGRATION_PLAN.md) §8 (array ownership), [`GYM_PARITY_INTEROP_TESTING.md`](GYM_PARITY_INTEROP_TESTING.md) (the live suite) |

This plan doesn't restate what the port plan specifies per area. It records where the work stands, what changed
around it since the last session, the decisions taken on 2026-11-08, and the order of everything left, with an exit
gate per step. How each piece is built (types, names, packaging, threading, engines) is fixed by the approved designs
in [`DESIGN_REVIEW.md`](DESIGN_REVIEW.md); the steps below cite them by id. Every number below was measured on 2026-11-08 on Windows 11 x64, .NET SDK 10.0.101, Python 3.12.12,
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
| Compile | after both fixes, with NuGet audit and packing off | 0 errors. The 0.80.0 ownership analyzer reports 10 sites, all migration step 4 work (P2.1): 6 × NDW016 in `src` (`Box`, `RenderResult.Rgb`/`Depth`/`RgbDepth`, the legacy `Step`, LunarLander's `ContactDetector`) and 4 × NDW012 in `tests/Gym.Tests` |
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
| toy_text | New Farama pixel art for all four envs (#1713, #1732, #1733, #1744); `Minecraft.ttf` and the 52 card PNGs deleted (77 assets removed, 97 added); Blackjack draws its numbers from digit images | P5 needs no font; CarRacing's score is the only text left (P8, DR-27) |
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
| D10 | The Aether LunarLander | Removed when LunarLander-v3 lands in P8, which is before 2.0 (follows from D21) | P8.5 |
| D18 | ImageSharp advisories | **Suppress the five advisories and keep ImageSharp 2.1.13** | R0.2; consumers of the packages still see the advisories in their own restores (§8) |
| D19 | Array ownership (migration step 4, and NUMSHARP_MIGRATION_PLAN's M5) | **Explicit ownership plus `[NDScoped]`**: spaces own and dispose their arrays, carried arrays are `[NDBorrowed]`, `StepResult`/`ResetResult` implement `INDArrayCarrier`, and ported methods' temporaries are reclaimed by `[NDScoped]`, woven by NumSharp.Build (a development dependency, never in consumers' graphs) | P2.1, then every phase |
| D20 | Branch and CI | **One branch until 2.0**: all work continues on `journey1` (cut from `feat/numsharp-master-migration` on 2026-11-08) and merges into `master` once, at the release. CI runs on the branch, informational | P0r.1, P10.5 |
| D21 | Release cut | **One 2.0 with everything except MuJoCo**: P2–P8, then P10. MuJoCo (P9) was postponed out of 2.0 by DR-35 | §4 |
| DR-01…DR-36 | The implementation designs | Approved in [`DESIGN_REVIEW.md`](DESIGN_REVIEW.md) in ten rounds; its "Coverage by phase" table lists which designs each phase rests on | Every phase |

### 3.2 Still open, with the default that applies unless the maintainer overrides it

| # | Decision | Default | Needed by |
|---|---|---|---|
| D6 | MuJoCo version | Decided when MuJoCo resumes (DR-35); the managed port (DR-30) is checked against a pinned Python `mujoco` | P9 (postponed) |
| D9 | Legacy shim | `Gym.Legacy` stays through 2.x, `[Obsolete]` | P10 |
| D13 | `examples/` | Rewrite one CartPole example on the new API, remove the rest | P10 |
| D15 | Golden storage | In the repo as gzip JSONL (DR-04 decides how they are made) | P0r.3 |
| — | Two of the last session's open points | (1) The two concurrency tests that start threads with `Task.Run` move to explicit `Thread`s; (2) the SDL_gfx header keeps "C# port of", which zlib's clause 2 requires of altered versions. The third (renaming `Ported`) is DR-06, approved; the README/PRD tagline "A port of openai/gym" is rewritten in P10 | R0.5, P10 |

D7 (`AsyncVectorEnv`) is settled by DR-22, D14 (toy_text assets) by DR-19, and D22 (CarRacing's text) by DR-27.

## 4. The road to 2.0

| Phase | Deliverables | Exit gate | Size |
|---|---|---|---|
| **R0 Re-baseline** | NumSharp 0.80.0, the ImageSharp suppressions, package metadata, Gymnasium v1.4.0, housekeeping | Builds clean; unit 212/212; parity 163/163 gated, 14 `[OpenBugs]` failing as documented, all on both frameworks | S |
| **P0r P0's rest** | CI (informational), golden record/replay from the live suite (F4, DR-04), inventory tool; no static rules (DR-03) | CI green on the branch; the CartPole-v1 golden replays; the inventory test reports the missing-names list | M |
| **P2 Spaces** | 10 spaces, `spaces.utils`, `Space<T>` with typed env spaces (DR-07), 0-d `NDArray` discrete samples (DR-08), generated composite records (DR-13), array ownership (D19) | `tests/spaces` (109) green; live cells for every space; 0 space `[OpenBugs]` | L |
| **P3 Registry and checkers** | `Gymnasium.Make` typed and untyped (DR-10), generated options records and dictionary/anonymous kwargs (DR-11, DR-34), registration by module initializers (DR-12, DR-14, DR-17), the human-viewer default (DR-20), `PassiveEnvChecker`, `check_env`, `env_match`, `Autoreset`, `RecordEpisodeStatistics` | Registration and checker tests green; Gymnasium's `make("CartPole-v1")` stack equals ours, built by `Gymnasium.Make` | M |
| **P7 Rasterizer** | pygame `draw.*` (D4), `transform.scale`/`smoothscale`, `image.load`, the frame path for every family | 150+ random shapes per primitive pixel-exact against pygame | L |
| **P4 classic_control** | `MountainCarEnv(continuous)` for both MountainCar ids (DR-32), Pendulum, Acrobot, `CartPoleVectorEnv` | L3 and L4 for all 6 ids | M |
| **P5 toy_text** | Blackjack, FrozenLake, CliffWalking, Taxi with `ansi` and v1.4.0's art embedded (DR-19) | L3 and L4 for all 6 ids | M |
| **P6 Vector and wrappers** | `VectorEnv` typed and untyped (DR-21), `SyncVectorEnv`, `AsyncVectorEnv` in three modes (DR-22), vector utils, 35 wrappers, vector wrappers named by DR-24, F8 (DR-23), F9 (DR-25), `play`/`PlayPlot`, `FuncEnv`, `performance` (DR-36); `AtariPreprocessing` out | `tests/vector` (76) and `tests/wrappers` (146) green; async equals sync bitwise in every mode | L |
| **P8 Box2D** | The managed Box2D 2.3.x engine, core plus pybox2d facade (D5, DR-26, DR-29), `LunarLanderEnv`, `BipedalWalkerEnv`, `CarRacingEnv` (DR-32), CarRacing's text from recorded glyphs (DR-27); Aether removed | Engine goldens; L3 and L4 for all 5 ids; 0 `[OpenBugs]` | XL |
| **P9 MuJoCo** | **Postponed out of 2.0 (DR-35).** When it resumes: a managed port (DR-30) of what Gymnasium's 22 envs reach (DR-33), with MuJoCo's own renderer offscreen (DR-31) | Hopper-v5 bitwise against Python `mujoco`, then all 22 at L3 | XXL |
| **P10 Release 2.0.0** | Docs, migration guide, examples, packages; merge into `master` | The port plan's definition of done (§1.5) for everything but MuJoCo | M |

The order is sequential. P7 moves in front of P4 because MountainCar and Acrobot already call `pygame.draw`, and P5
needs `draw.rect`, `image.load` and `transform.scale`.

## 5. Phases

Each numbered step is one commit, unless it says otherwise.

### R0 Re-baseline

- **R0.1 NumSharp 0.80.0.** *Done 2026-11-09, with R0.3 in one commit (R0.1 alone cannot pack in Package mode):
  unit 207/207 gated (203 + the 4 gated inner-exception rows), 8 `[OpenBugs]` (N-1, N-3) failing as designed; parity
  163/163 gated, 15 `[OpenBugs]` (the 14 + one N-1 cell), all on net8.0 and net10.0, Debug and Release.*
  - Raise `NumSharp` to 0.80.0 in the five projects that declare it, and `NumSharp.Interop.pythonnet` to 0.80.0 in
    the parity suite.
  - Make `NumSharpSource=Package` the default in `eng/NumSharp.props`. Submodule mode stays, with its guards and the
    analyzer reference, and its comments say why it now exists: to build a NumSharp fix before NumSharp releases it.
  - Check out `refs/NumSharp` at `v0.80.0` (`d523eb48`) and record the new gitlink.
  - Delete `Gym.Exceptions.AssertionError`; every site throws `NumSharp.AssertionError` (DR-02). Bare asserts throw
    `new AssertionError()`; until NumSharp releases the `""` message (N-1), the cells and tests that compare a bare
    assert's text are `[OpenBugs("N-1")]` (DR-01). `NotImplementedError` and `ImportError` stay in `Gym.Exceptions`
    until NumSharp has them, each gap proven by an `[OpenBugs("N-3")]` test (DR-01, DR-02).
  - Switch `RngFacts` to `PCG64.State`. Map Python's `AssertionError` to NumSharp's type in the parity harness.
  - Remove the workarounds for the bugs 0.80.0 fixed:
    - `Discrete`'s unwrapping of the one-element array `choice()` used to return;
    - `Seeding.Entropy`'s four-word branch, now that an unseeded `SeedSequence` holds a `BigInteger`, and with it
      `SeedTests`' inconclusive case for a zero top word;
    - the parity harness's `np.atleast_1d` around `choice()` (`SeedingLiveParityTests`, the 9 failing cells).
  - Update `NUMSHARP_MIGRATION_PLAN.md`'s status and §14, and the root `CLAUDE.md`.
- **R0.2 ImageSharp advisories (D18).** *Done (`e4e5215`).* A root `Directory.Build.props`, new and holding only the five
  `NuGetAuditSuppress` items, each with its advisory's title and the reason it doesn't apply. The comment also records
  that the suppression covers only our build: consumers' restores still report the advisories. A new advisory against
  2.1.13 fails the build again, by design, and gets the same review.
- **R0.3 Package metadata.** *Done 2026-11-09, with R0.1.* `PackageLicenseExpression` `Apache-2.0`, a packed `PackageIcon`, and a `PackageReadmeFile`
  replace `PackageLicenseUrl`/`PackageIconUrl` in the five packable projects (NU5048/NU5125). `JetBrains.Annotations`
  gets `PrivateAssets="all"`, and `Gym.csproj` drops `SixLabors.ImageSharp.Drawing`, which nothing in `src/Gym` uses
  (both found by the 2026-09-23 NuGet review).
- **R0.4 Gymnasium v1.4.0 (D2), by the port plan's §14.10.** *Done 2026-11-09. `repin_references.py` moved all 146
  permalinks in 38 files and refused none: every line-anchored target kept its text, only line numbers moved (core.py
  +3 to +12, cartpole.py +1, registration.py -1, wrappers/common.py +1 and +27); 4 whole-file links cite files that
  changed, reviewed. `Env.Step`/`Env.Reset` document v1.4.0's new-objects contract (Gym.NET's envs already build a
  fresh observation and `Info` per call); `_check_metadata` and `PassiveEnvChecker`'s data-reuse check are not ported
  yet, so P3 ports v1.4.0's versions. The toy_text art (D14) is the Farama Foundation's own pixel art, shipped in
  Gymnasium's MIT package (FrozenLake credits "Farama Foundation's shared environment art set"; Taxi, CliffWalking and
  Blackjack carry an unfilled "Pixel art assets by ____"), so P5 needs only Gymnasium's MIT notice. The
  `gymnasium_rng` golden re-recorded at v1.4.0 is byte-identical. Every gate's verdict is unchanged: unit 207/207,
  parity 163/163, the same 8 + 15 `[OpenBugs]`, net8.0 and net10.0, Debug and Release.*
  - Check out `refs/Gymnasium` at `v1.4.0` and update `tools/parity/manifest.json`.
  - A script, `tools/parity/repin_references.py`, moves every permalink at `9e04324` to the new commit. It maps line
    numbers through `git diff 9e04324..v1.4.0` and refuses any link whose target line's text changed, so those get
    reviewed by hand.
  - Port what the four changed sources mean for ported code: `_check_metadata`'s message, and the core docs note.
  - Confirm the license of the new toy_text art (D14).
  - Re-run every gate. A cell whose verdict moves is reviewed, never re-recorded blindly.
- **R0.5 Housekeeping.** The two `Task.Run` concurrency tests move to explicit threads (§3.2); `tests/Gym.Tests/Ported`
  becomes `tests/Gym.Tests/Gymnasium`, namespace `Gym.Tests.Gymnasium.*`, mirroring `refs/Gymnasium/tests` (DR-06).
  *Done 2026-11-09. `ConcurrencyTests` runs each worker on a dedicated background thread with a 2-minute join limit and
  rethrows every worker's failure as one `AggregateException`. The rename surfaced that the namespace shadows the
  `Gymnasium` facade class in `Gym.Tests.*`; P3 imports the facade with a `global using static` (DESIGN_REVIEW.md,
  "Consequences found while implementing").*
- **Exit:** Debug and Release build with 0 errors on net8.0 and net10.0; unit 212/212 (203 headless + 9 viewer), the
  N-1 tests aside; parity 163/163 gated, the N-1 cells aside; the 14 `[OpenBugs]` cells fail at their documented
  divergences and nowhere else. *Met 2026-11-09: 0 errors in both configurations; unit 207/207 headless (the 9 viewer
  tests not run: they open windows), parity 163/163, `[OpenBugs]` 8 unit (N-1, N-3) and 15 parity (the 14 + the N-1
  cell), on net8.0 and net10.0, Debug and Release.*

### P0r P0's rest

- **P0r.1 CI (D20, informational).** *Written 2026-11-09 (`.github/workflows/ci.yml`), not run: running it needs a push,
  which this run does not do. The YAML parses and every command in it was run locally on Windows. One change from the
  plan: the unit suite targets `net*-windows` (WinForms viewer tests), so the ubuntu job cannot run it, and replays the
  goldens without Python instead.* GitHub Actions on the branch:
  - `windows-latest` (the reference) builds Debug and Release, runs the headless unit suite, sets up the pinned
    Python stack with `tools/parity/setup_venv.ps1`, and runs the parity gate with `GYMNET_PYTHONNET_REQUIRE_ENGINE=1`.
  - `ubuntu-latest` builds and runs the unit suite (as written: builds the parity suite and replays the goldens).
  - Both set `AVALONIA_TELEMETRY_OPTOUT=1` and exclude the viewer tests.
- **P0r.2 Static rules: none (DR-03).** No analyzers and no rule scripts; the port plan's §14.7 bans are not enforced
  by tooling.
- **P0r.3 Golden harness (F4, DR-04).** *Done 2026-11-09. `GymPair.Recorder` receives each call and frame after the
  pair verified it; `GYMNET_PARITY_RECORD=1` makes `GoldenLiveTests` save it to `tests/Golden/<suite>/<case>.jsonl.gz`
  plus PNG frames, written only when the content changed (atomically; the gzip bytes of .NET 8 and 10 differ). Replay
  (`GoldenReplayTests`, category `Golden`) needs no Python: it feeds the recorded actions back or re-samples a sampled
  case, compares through the same `ByteContract`, and refuses a stale golden (other case parameters or another Gymnasium
  commit). Two CartPole-v1 goldens (balanced 500 steps with 6 frames; seeded random actions to termination with 7
  frames; 104 KB) replay 8/8 with Python disabled, and the replay's controls catch a float32 `tau` at step 1 and an
  extra draw at the reset. Format: `tests/Golden/README.md`.*
  - The live suite (`tests/Gym.Tests.Parity`) holds the code that records and regenerates goldens: one case catalog
    drives its live, record and replay modes (the interop design's §9), so a golden is always a recording of a live
    cell. No separate Python recorder script.
  - Record mode writes trajectories and frames into `tests/Golden` as gzip JSONL plus PNG, with a manifest of the
    pinned stack (D15). The C# side gets `GoldenTrajectory`/`GoldenAssert`, with open-loop and sampled action modes.
  - Replay needs no Python: the ubuntu job and machines without the stack check L3/L4 from the files. The live suite
    stays the oracle on the reference job.
- **P0r.4 Inventory (port plan §14.6).** *Done 2026-11-09: 301 public names at v1.4.0
  (`tools/parity/gymnasium_inventory.json`); `InventoryTests` finds 43 in Gym.NET and checks the other 258 against
  `tests/Gym.Tests/Gymnasium/inventory.expected-missing.txt`, each line with its phase (P2 16, P3 28, P4 8, P5 16,
  P6 87, P8 12, P9 33) or reason (n/a 54: JAX, PyTorch, pickling; by design 3: DR-28, DR-32; Atari 1).*
  - `tools/parity/inventory.py` lists Gymnasium's public names by AST.
  - A reflection test maps them through §3.8 and reports the missing ones against a committed expected-missing list.
    Each phase shrinks the list, and 2.0 needs it empty.
- **Exit:** both CI jobs green on the branch; the CartPole-v1 golden, recorded by the live suite, replays without
  Python; the inventory test runs. *Met 2026-11-09 except "CI green", which waits for a push: the goldens replay with
  Python disabled, and the inventory test runs and passes.*

### P2 Spaces

Specs: the port plan's §7; sources `gymnasium/spaces` (3,816 lines at v1.4.0); tests `tests/spaces`.

- **P2.1 `Space<T>`, typed env spaces and ownership (DR-07, DR-08, D19).** *Done 2026-11-09 with P2.2 and P2.3, one
  commit: `Space`/`Space<T>`, `SpaceSeed` (int, BigInteger or a shared generator; a null generator is `None`),
  `ObjectSpace` (any space as a `Space<object?>`, for object-typed envs), typed `Env`/`Wrapper` spaces with the untyped
  `IEnv` view, CartPole and the Aether lander on `Env<NDArray, NDArray>`, `Env.Dispose` releasing the env's spaces (a
  wrapper releases its own and its env), `INDArrayCarrier` results, disposable `RenderResult`, NumSharp.Build
  referenced; the analyzer's src sites are all fixed. Not yet: measuring `[NDScoped]` (no ported method needs it yet:
  every temporary is disposed by hand) and the source generator, which starts with P2.5.*
  - The generic `Space<T>` over an untyped `Space` base: lazy `NpRandom`, `Seed` returning the seed list,
    `ToString`/`Equals`, `to_jsonable`/`from_jsonable`, `Contains(object?)`.
  - `Env<TObs, TAct>` exposes `Space<TObs> ObservationSpace` and `Space<TAct> ActionSpace`; `IEnv` keeps the untyped
    view; wrappers follow their `TWObs`/`TWAct` (DR-07).
  - `Discrete : Space<NDArray>`, a 0-d int64 array standing for `np.int64` (DR-08), so the discrete-action envs move to
    `Env<NDArray, NDArray>`: CartPole and LunarLander change here, with their tests and parity cells.
  - `Space : IDisposable`; `Env.Dispose` disposes both spaces after `Close`; `[NDBorrowed]` on carried arrays;
    `StepResult`/`ResetResult` implement `INDArrayCarrier`; NumSharp.Build added as a development dependency, so
    `[NDScoped]` methods are woven (D19). The 0.80.0 analyzer's 10 NDW sites are fixed by this work; warnings never
    gate a step (DR-05).
  - Measure what `[NDScoped]` saves on `Box.Sample` and CartPole's step: NUMSHARP_MIGRATION_PLAN §8.4 measured
    0.634 µs per undisposed small array against 0.217 µs disposed.
  - The source-generator project (DR-13, DR-34) starts here, with the `[GymTuple]`/dict records P2.5 needs.
- **P2.2 `Box`**: *Done 2026-11-09 (with P2.1).* validation with Gymnasium's messages, float64 bounds kept in float64 (closes the `Box(float, float)`
  defect), sampling per bound category, `contains` with `np.can_cast` and lists, `repr`.
- **P2.3 `Discrete`**: *Done 2026-11-09 (with P2.1).* shape `()`, int64 0-d samples (DR-08), `start`, masks and probabilities, `repr`, `n > 0`
  validation (B-03, and B-10 as `Discrete`'s cell tags it).
- **P2.4 `MultiBinary`, `MultiDiscrete`.** *Done 2026-11-09 with P2.5 and P2.6 (the spaces themselves), one commit.*
- **P2.5 `Tuple`, `Dict`, `Sequence`**: the generic forms (`object?[]`, an insertion-ordered map) with
  source-generated typed records over them (DR-09, DR-13); subseeds through `integers(int32.MaxValue, size: n)`,
  `geometric(0.25)` for `Sequence`, which NumSharp 0.80.0 now provides. *Spaces done 2026-11-09: `Tuple`
  (`object?[]` samples), `Dict` (`Info` samples, `sort_keys` kept for batching), `Sequence` (tuple or stacked samples),
  seeding by int, list, dict or `None` with each space's own refusal (`Space.SeedTypeError`), and
  `gymnasium.vector.utils`' space functions (`VectorUtils`: `batch_space`, `batch_differing_spaces`, `iterate`,
  `concatenate`, `create_empty_array`), which `Sequence(stack=True)` and `Graph` sample through. An `object?[]` is now
  Python's tuple in `PyBuiltins` and `PyEquality` too (DR-13). The source generator is still to come (P2.9).*
- **P2.6 `Text`, `Graph`, `OneOf`.** *Done 2026-11-09 (with P2.4): `Text` (code points, `charset` as a string or set),
  `Graph` and `GraphInstance` (an `ITuple` that owns its arrays), `OneOf` (`(np.int64 index, value)` pairs). Live
  cells: 24 spaces × repr, seeded samples with every generator state, `seed(int)`, own samples as members,
  `is_np_flattenable`, JSON round trip, equality and `batch_space` (`SpaceSpecLiveParityTests`); masks, probabilities,
  list/dict/refused seeds, membership matrices, `Graph.sample` with node and edge counts, constructor errors and
  `MultiDiscrete` indexing (`SpaceDetailsLiveParityTests`); `create_empty_array`, `concatenate`, `iterate` and
  `batch_differing_spaces` (`VectorUtilsLiveParityTests`). 393 new cells, all gated.*
- **P2.7 `spaces.utils`**: `flatdim`, `flatten`, `unflatten`, `flatten_space`, `is_space_dtype_shape_equiv`, with
  v1.4.0's `OneOf` and `Tuple` fixes. *Done 2026-11-09: `SpaceUtils` (the shared flat dtype that must hold a `OneOf`
  index, Graph and Sequence flattening through `VectorUtils`, Gymnasium's errors, including its own `TypeError` when a
  text flattened inside a float tuple is unflattened). The live space cells now also run over Gymnasium's own test
  catalog (`tests/spaces/utils.py`, 60 spaces), 1415 space cells in all. Two `Box` fixes found on the way: a 0-d bound
  is a NumPy scalar (shape `(1,)` by default), and lists are packed as NumPy packs them (`[300]` into int8 raises, so
  `contains` answers `False` instead of wrapping to 44; int64 stays exact).*
- **P2.8 Tests**: port `tests/spaces` (109 functions, 13 files) under `tests/Gym.Tests/Gymnasium/Spaces`. Add live
  cells for every space (samples with generator state, `contains`, `repr`, `flatten`, JSON) and remove the `[OpenBugs]`
  attribute from the six space cells (tagged B-03, B-10 and P2) as they turn green. *Done 2026-11-09: the 13 files
  ported (900 test cases over Gymnasium's own space catalog, `TestingSpaces`), chi-square critical values recorded from
  scipy 1.16.3, `EnvChecker.DataEquivalence` (`gymnasium.utils.env_checker.data_equivalence`) ported for them. Python-only
  parametrizations (dtypes that are not dtypes, shapes of the wrong type, strings where spaces belong, pickling,
  hash-seed independence) are listed in each class's remarks; `test_all_space_pairs_for_is_space_dtype_shape_equiv`
  needs the vector shared memory and moves to P6. Found on the way: `MultiBinary`/`MultiDiscrete` read JSON and list
  candidates through NumSharp's Python-list coercion (a `BigInteger` or an exact int64 no longer breaks them);
  `MultiDiscrete` takes NumPy slices (`space["0:2, :"]`); `Tuple` gains Python's `count`/`index` (`CountOf`,
  `IndexOf`). Two Gymnasium test quirks are documented where ported: an `is not` identity check that cannot fail, and an
  int64 case that passes only because an earlier case mutated the shared bounds.*
- **P2.9 The source generator (DR-13, DR-34)**: `[GymTuple]`/dict records over the generic forms, with implicit casts,
  Python `==` and `repr`; the same project later emits DR-34's options records (P3). Split out of P2.1/P2.5, which
  shipped the generic forms the records convert to. *Moved next to P3's options records (2026-11-09): nothing in P2
  needs it (every space works on its generic form), and building it once for both record kinds avoids a second pass.
  Done 2026-11-09 (with P3): `src/Gym.SourceGenerators` (an incremental generator, netstandard2.0, Roslyn 4.8) writes,
  for `[GymOptions]` classes, a `<Class>Options` record (properties from the constructor make binds, documented with its
  `<param>` texts, each remembering whether it was set; `ToKwargs()`; a constructor of the class taking the record that
  refuses unset required arguments with Python's text), and for `[GymTuple]`/`[GymDict]` positional records the implicit
  conversions to and from `object?[]`/`Info`, Python's `==` through `PyEquality`, a consistent hash and Python's `repr`.
  CartPole and the wrappers with parameters (`TimeLimit`, `OrderEnforcing`, `RecordEpisodeStatistics`,
  `RenderCollection`) carry `[GymOptions]`; the Gym package ships the generator under `analyzers/dotnet/cs`.*
- **Exit:** `tests/spaces` green on both frameworks; every space has live cells; no space `[OpenBugs]` left; the
  inventory list loses every `gymnasium.spaces` name.

### P3 Registry and checkers

Specs: the port plan's §8, §9 (the `make()` stack) and §12.

- **P3.1 Registration**: `Gymnasium.Make` in an untyped and a typed form (DR-10), `MakeVec` (scaffold), `Spec`,
  `Register`, `Registry`, `PprintRegistry`, `RegisterEnvs`, with v1.4.0's entry-point and metadata fixes and exact
  error texts.
  - Each env assembly registers its ids from a module initializer (DR-12), with `typeof(...)` entry points; an
    environment variable disarms it (DR-14); the env package's `buildTransitive` targets compile a one-line app-side
    initializer into consuming C# apps so registration runs before `Main` (DR-17). CartPole-v0/v1 register now; each
    later phase registers its ids.
  - kwargs: generated `<Class>Options` records (`[GymOptions]`, DR-34), plus dictionary and anonymous-object overloads,
    all bound by Python name (DR-11). *Options records done 2026-11-09 with P2.9: `Make("CartPole-v1", new
    CartPoleEnvOptions { RenderMode = "rgb_array" })`.*
  - `render_mode="human"`: `Gymnasium.HumanViewerFactory`, a `viewer_factory` override, and `DependencyNotInstalled`
    when neither is set (DR-20).
  - *Done 2026-11-09 except the options records:* `EnvRegistry` (`gymnasium.envs.registration`: `Registry`, an ordered,
    synchronized `EnvSpecRegistry`; `Register`, `Spec`, `Make` untyped and typed, `PprintRegistry`, `RegisterEnvs`,
    `Namespace`, `CurrentNamespace`, `FindHighestVersion`, `LoadEnvCreator`, every `_check_*` with Gymnasium's texts and
    CPython's `difflib.get_close_matches` for the suggestions) re-exported by the `Gym.Gymnasium` facade. Entry points
    are a class (`typeof(CartPoleEnv)`), an `EnvCreator`, or a `"Namespace:Class"` string resolved over the loaded
    assemblies (loading the one named after the namespace when needed); an `"Assembly:Id"` id loads the assembly and runs
    its module initializer, Python's import. kwargs bind by Python name through `PyKwargs.Construct` (dictionary and
    anonymous-object forms; the generated options records are P2.9 with DR-34), which also closes the generic wrappers
    over the env's types. `EnvSpec.Make`/`ToJson`/`FromJson` (Python's `json` text through `PyJson`; a class entry point
    is written as its `"Namespace:Class"` string and equals it). `GymEnvironments.Register()` registers CartPole-v0/v1
    from a module initializer, `GYMNET_DISABLE_ENV_REGISTRATION` disarms it, and `buildTransitive/Gym.NET.Environments.targets`
    compiles the app-side initializer (imported by path in the test projects; the examples, still on netcoreapp3.0, are
    refreshed in P10). DR-20: the viewer contract moved into `Gym` (namespace `Gym.Environments` kept), envs resolve
    their factory with `Gymnasium.ResolveHumanViewerFactory`. `RenderCollection` came early for `make`'s `*_list` modes;
    `HumanRendering` stays in P6 (`make` raises `NotImplementedError` where it would apply), and so do `make_vec` and
    `VectorizeMode`, which need `VectorEnv`. LunarLander stays unregistered until P8 replaces the Aether port.*
- **P3.2 Checkers**: `utils/passive_env_checker.py` and the `PassiveEnvChecker` wrapper, `utils/env_checker.py`
  (`check_env`, including v1.4.0's distinct-object and RNG-state checks), `utils/env_match.py`. *Done 2026-11-09:
  `PassiveChecker` (every check, plus untyped `IEnv` overloads; data reuse compares retained views, which keep the
  memory alive so a disposed and re-pooled buffer cannot pass for sharing), `PassiveEnvChecker`, `EnvChecker.CheckEnv`
  and its checks (the checks of Python signatures that C# rules out are documented, not ported), `EnvMatch`.*
- **P3.3 Common wrappers**: `Autoreset`, `RecordEpisodeStatistics` (its time entry excluded from goldens), vector
  `step_api_compatibility`. *Done 2026-11-09 for `Autoreset` and `RecordEpisodeStatistics` (with `Deque<T>`, Python's
  `deque(maxlen)`); the vector form of `step_api_compatibility` moves to P6 with the vector info layout.*
- **P3.4 Tests**: port `tests/envs/registration`, the checker tests in `tests/utils`, and the common-wrapper tests.
  Parity cells switch from the hand-built `TimeLimit(OrderEnforcing(env))` to `Gymnasium.Make`. *Done 2026-11-09: 83
  ported tests (registration, `spec`, `make`, `EnvSpec`, `pprint_registry`, both `PassiveEnvChecker` files,
  `env_checker`, `env_match`, `Autoreset`, `RecordEpisodeStatistics`), a `Gym.Tests.Plugin` assembly standing for the
  module a `"module:id"` make imports, and `[OpenBugs]` cells for what later phases bring (Blackjack/Taxi P5,
  MountainCar/Pendulum P4, the P6 wrappers, CarRacing P8); Humanoid's case is out with MuJoCo (DR-35). Live cells:
  `make`'s stack through truncation, its `str` and specs, the registry's errors, `EnvSpec` JSON both ways,
  `pprint_registry`, `check_obs`.*
- **P3.5 The reverse bridge** (the interop design's §8): a Python env class over a Gym.NET env, so Gymnasium's own
  `check_env` and `check_environments_match` run against Gym.NET's envs in the live suite. Every later env phase adds
  its ids to it. *Done 2026-11-09: `tests/Gym.Tests.Parity/Bridge` (a `GymNetShim(gymnasium.Env)` snippet class
  subclassed per .NET env class, a mirrored `EnvSpec` so `gymnasium.make` builds Gymnasium's stack over a Gym.NET env,
  mirrored spaces, zero-copy observations, the PCG64 state mirrored after every call, Gym.NET's exceptions and
  warnings raised as Python's). Cells: `check_env` on CartPole made, bare and `rgb_array` (held to Gymnasium's verdict
  on its own env and to `EnvChecker.CheckEnv`), `check_environments_match` against Gymnasium's CartPole with frames,
  `make`'s wrapped `TypeError`, warnings, and three negative controls (a planted tau, a reused buffer, an unseeded
  reset). A later env phase adds a cell per id; spaces beyond `Box`/`Discrete`/`Tuple`/`Dict` get a mirror when an env
  needs one. Gymnasium's pytest modules through the bridge (the design's third item) stay unbuilt.*
- **Exit:** those tests green; Gymnasium's `make("CartPole-v1")` and `Gymnasium.Make("CartPole-v1")` agree for whole
  episodes, `PassiveEnvChecker` included; both our `check_env` and Gymnasium's pass on CartPole. *Met 2026-11-09: the
  live `MadeStack` cells (P3.4), and `CheckEnv_PassesGymNetsCartPole_AsGymnasiumsOwn`, where Gymnasium's `check_env`
  on the bridged CartPole and Gym.NET's own both give Gymnasium's verdict on its own CartPole.*

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
- *Done 2026-11-09 for what this run's envs need (P4 classic_control, P5 toy_text); every behaviour was derived
  black-box from live pygame-ce 2.5.8 with probe scripts, no pygame source read (D4):*
  - *`draw.line` width 1 (Acrobot): an error term starting at half the major extent, ties toward the start, floats
    truncated; exact on 5,000 random lines.*
  - *`draw.rect` widths 0/1/-1 (FrozenLake): the original rectangle's edges clipped, except a clipped sliver at most 2
    pixels across is filled; exact on 8,000 random rectangles.*
  - *`image.load` (8-bit RGB/RGBA PNG through ImageSharp, D18) and `image.frombytes`: per-pixel-alpha surfaces;
    every env asset equals pygame's, alpha included.*
  - *Alpha blit onto a frame: `d + (((s - d) * a + s) >> 8)`, exact on all 2^24 (source, alpha, destination) triples.*
  - *`transform.scale`: 16.16 steps sampled at the half step; exact on 2,011 size pairs.*
  - *`transform.smoothscale` shrinking (Pendulum): a 2.14 fixed-point area filter, horizontal pass first; exact on 800
    random rows and 300 random images. Growing is refused (`NotSupportedException`) until an env needs it (P8).*
  - *`transform.flip` keeps per-pixel alpha.*
  - ***Not converged (reported to the maintainer per D4): `draw.aaline`/`draw.aalines` (MountainCar's hill).*** *The
    single-segment model (float32 coordinates, major-axis stepping with float32 accumulation, `1 - frac`/`frac`
    coverage, `trunc(b * c + (1 - b) * old)` blending, first column weighted `1 - frac(x0)`, last `frac(x1)`) reproduces
    all but 21 pixels of 2,500 random segments, each off by one from a 1-3 ulp difference in y that a dozen arithmetic
    variants did not explain; the joint rule of `aalines` (which endpoint columns are drawn where segments meet) is not
    settled either. Both ship as the best model, documented as not pixel-exact, with `[OpenBugs("P7")]` cells; MountainCar's
    frames inherit the gap. The maintainer's options: allow consulting pygame-ce's `draw.c` for these two functions only,
    accept a documented tolerance for MountainCar's frames, or continue black-box probing.*
    *Probing continued during P4 (2026-11-09, still black-box): MountainCar's hill differed in 196 pixels per frame, every
    one within 2 pixels of a joint, each an extra partial-weight sample; leaving out the partial column of both segments
    at a joint (instead of one) brings it to 16 pixels and is exact along smooth runs (24 of 150 random smooth curves
    exact, against 0). Two further measurements for the maintainer: pygame's `aalines` is not `aaline` segment by
    segment (a 2-point open `aalines` blends the segment's interior twice and its end columns once, 251 of 300 cases
    exact under that model), and joints between segments drawn against their major axis, or turning sharply, follow
    another rule still.*
  - *Deferred to P8 (box2d only): `polygon`, `circle`, `lines`, `aaline` beyond MountainCar, `smoothscale` growing, rect
    widths above 1; `font`.*
- **Exit:** every primitive above is pixel-exact on its random corpus.

### P4 classic_control

Specs: the port plan's §11.1; sources 2,028 lines.

- **P4.1** MountainCar-v0 and MountainCarContinuous-v0 through one `MountainCarEnv(continuous: bool)` (DR-32): both
  sources' dynamics, spaces and rewards behind the flag; `repr` prints `Continuous_MountainCarEnv` in continuous mode;
  the continuous id's `EnvSpec.kwargs` carries `continuous=True` (accepted deviation).
- **P4.2** Pendulum-v1: `np.sin(th)`, the NEP 50 float32/float64 mix in its reward, array-bound `uniform` in `reset`,
  `clockwise.png`.
- **P4.3** Acrobot-v1: the RK4 integrator, `wrap` and `bound`.
- **P4.4** `CartPoleVectorEnv` (with P6's `VectorEnv` base, pulled forward if needed, as P1 pulled `TimeLimit`
  forward).
- Each env: live cells for whole episodes (policy and random actions), reset options, invalid actions with Python's
  messages, frames along episodes and at extreme states; goldens recorded; registration.
- **Exit:** L3 and L4 for all six ids, live and golden.
- *Done 2026-11-09 for P4.1–P4.3; P4.4 (`CartPoleVectorEnv`) moves to P6, where `VectorEnv` lands:*
  - *`MountainCarEnv(continuous)`, `PendulumEnv` and `AcrobotEnv` in `Gym.Environments.Envs.Classic`, registered as
    `MountainCar-v0`, `MountainCarContinuous-v0` (kwargs `continuous=True`, DR-32), `Pendulum-v1` and `Acrobot-v1` with
    Gymnasium's limits and thresholds. Shared support: `NpScalar` (NumPy 2's NEP 50 scalar rules for the continuous
    MountainCar, whose state is float64 after a reset and float32 after a step, and Pendulum's float32 torque),
    `HumanWindow` (DR-20's window, sized at the first human render), `EnvAssets` (DR-19: `clockwise.png` embedded,
    byte-identical to Gymnasium's), `RenderMath` (Python's `int()` and pygame's C `(int)` of a float size).*
  - *Byte-exact details the ports carry: NumPy's float32 `sin`/`cos` (through NumSharp, bit for bit) wherever a float32
    scalar meets them (Acrobot's reset observation and frame, the continuous car's height); the C runtime's `pow` for
    every `x ** 2`; an unsigned discrete action 0 wrapping in MountainCar's `action - 1` (it pushes right);
    `math.pow`'s `OverflowError`; Acrobot's action indexing a Python list (negative indices valid, Python's
    `TypeError`/`IndexError` texts); Python's `AttributeError`, `TypeError`, `ValueError` and NumPy's `IndexError` for
    missing, `None`, short and long states. Refused rather than approximated (`NotSupportedException`): continuous
    actions of more than one dimension or of float16, which NumPy would carry into a state of another shape or precision.*
  - *State: each env's `State` is `double[]` (CartPole's form) holding exactly Python's values, though Python's
    container changes between calls; the parity harness compares the state's float64 values
    (`PyGymEnv.GetStateValues`).*
  - *Parity: 41 MountainCar cells (both modes), 19 Pendulum, 22 Acrobot, 8 reverse-bridge cells (Gymnasium's
    `check_env` and `check_environments_match` on all four ids), 5 registration cells, and 4 goldens (with frames for
    Pendulum and Acrobot), all passing; inventory ratchet −7 names. MountainCar's whole frames are `[OpenBugs("P7")]`
    (16 hill pixels per frame); gated cells compare every pixel outside the hill's measured gap, so the car, wheels and
    flag are pixel-exact. The bridge returns Pendulum's reward as the `np.float64` Gymnasium's does
    (`BridgeReward`), since `check_environments_match` compares types.*
  - *Deferred: `MountainCarEnv.get_keys_to_action` arrives with `utils.play` (DR-36); Gymnasium's construction-time
    `low`/`high` (`low_state`/`high_state`) attributes are read from the observation space instead; RLPy's unused
    `action_arrow`/`domain_fig`/`actions_num` on Acrobot are not carried.*

### P5 toy_text

Specs: the port plan's §11.2, with v1.4.0's sources (1,868 lines) and art.

- **P5.0 Assets (DR-19)**: v1.4.0's toy_text images copied once from a Gymnasium installation into the repo and
  embedded as resources (re-copied on a re-pin), their license confirmed when copied.
- **P5.1** `categorical_sample`, and the `'c'` map: N-2 is proven by an `[OpenBugs("N-2")]` test (DR-01); P5 can't
  proceed without the map, so a local builder from per-character rows is the stuck-case workaround until NumSharp splits
  strings itself.
- **P5.2** FrozenLake-v1 and FrozenLake8x8-v1: `generate_random_map`, slippery transitions, `ansi`.
- **P5.3** CliffWalking-v1 and CliffWalkingSlippery-v1.
- **P5.4** Taxi-v4: action masks in `info`.
- **P5.5** Blackjack-v1: digit images, no font.
- **Exit:** L3 and L4 for all six ids; `ansi` output identical to Gymnasium's text, byte for byte.
- *Done 2026-11-09:*
  - *`FrozenLakeEnv`, `CliffWalkingEnv`, `TaxiEnv` and `BlackjackEnv` in `Gym.Environments.Envs.ToyText`, registered in
    Gymnasium's order as `Blackjack-v1` (kwargs `sab=True, natural=False`, in that order), `FrozenLake-v1`,
    `FrozenLake8x8-v1`, `CliffWalking-v1`, `CliffWalkingSlippery-v1` and `Taxi-v4`, with Gymnasium's limits and
    thresholds. The module functions are statics (DR-15): `ToyTextUtils.CategoricalSample` (the draw before the empty
    check, as NumPy's), `FrozenLakeEnv.GenerateRandomMap`/`IsValid`, and Blackjack's `Cmp`, `DrawCard`, `DrawHand`,
    `UsableAce`, `SumHand`, `IsBust`, `Score`, `IsNatural`, `DrawTable`. 97 images embedded (DR-19, hashes in
    `Assets/README.md`); the maps through `ToyTextUtils.CharMap`, P5.1's workaround for N-2.*
  - *Byte-exact details the ports carry: Python's dict lookup of the action (`True`, floats and NumPy scalars equal to a
    key are actions; `KeyError(repr)` otherwise; `unhashable type` for arrays), Python's list indexing where the source
    indexes a list with the action (FrozenLake's and Taxi's captions, Taxi's sprite: a float action raises at render
    time, as in Python), `int` reset infos (`{"prob": 1}`), Taxi's int8 `action_mask`, the fickle passenger's extra
    draws, Blackjack's draws in Gymnasium's order (dealer, player, suit, face) and its `np.str_` suit, the
    `AttributeError`s before a reset at the point Python reaches them (after the first ground tile for CliffWalking's
    frame, after the background for Taxi's).*
  - *Typing (DR-13): `BlackjackEnv` is `Env<object?[], NDArray>`; `BlackjackObservation` reads an observation by name
    through the generated conversion, and every `[GymTuple]` record is now an `ITuple`, so a `Tuple` space contains it
    (DESIGN_REVIEW, consequences). Rewards are float64 as everywhere (Gymnasium's tables return ints); Taxi-v4's
    threshold prints `8.0` where Gymnasium registers the int `8` (pinned).*
  - *Parity: 10 FrozenLake cells, 10 CliffWalking, 15 Taxi, 12 Blackjack (one bare-assert cell each for Taxi and
    Blackjack is `[OpenBugs("N-1")]`), 12 reverse-bridge cells (Gymnasium's `check_env` and `check_environments_match`
    on all six ids, the bridge returning Python ints where Gymnasium's envs do: `BridgeObservation.PythonInt`,
    `BridgeReward.Int`), 6 registration cells, and 6 goldens with frames; all gated cells passing. Unit: `ToyTextTests`
    (10), `TupleRecord_IsAnITuple`, the N-2 proof `[OpenBugs("N-2")]`, the two registry suggestion tests promoted from
    `[OpenBugs("P4", "P5")]`; inventory ratchet −16 names.*
  - *Deferred: nothing of P5. Known non-exact corners, all documented on the members: a float action's
    `taxi_orientation` reads as its key (2) where Python stores the float; an int `rainy_probability` kwarg becomes a
    float64 probability in the table and info.*

### P6 Vector and wrappers

Specs: the port plan's §9 and §10; sources `vector` 3,113 lines and `wrappers` 7,569 lines at v1.4.0.

- **P6.1** `VectorEnv<TObs, TAct>` with batched types and the untyped `IVectorEnv`, `MakeVec` in both forms (DR-21),
  `AutoresetMode` (NEXT_STEP, SAME_STEP, DISABLED), `SyncVectorEnv`, the vector utils (`batch_space`,
  `batch_differing_spaces`, `iterate`, `concatenate`, `create_empty_array`).
- **P6.2** `AsyncVectorEnv` with the three execution modes of DR-22: `ThreadPool` (default), `DedicatedThreads` (env
  *i* pinned to thread *i* % `threadCount`), `ThreadPerEnv`; Gymnasium's `*_async`/`*_wait` API, `max_concurrency` as a
  semaphore, errors returned with the env index, and a thread-bound sub-env given its own thread automatically in
  ThreadPool mode. Results gathered in env order, so every mode is bit-equal to sync; the tests run all three.
- **P6.3** Observation, action and reward wrappers, in the port plan's §9 groups. The 18 vector twins take the suffix
  names of DR-24 (`RecordEpisodeStatisticsVector`, …), each with `[PyName]` carrying Gymnasium's name.
- **P6.4** F8: a managed port of OpenCV's `INTER_AREA` paths (DR-23) for `ResizeObservation`, byte-exact against live
  `cv2` at a pinned version.
- **P6.5** Rendering wrappers; F9: `RecordVideo` over a pluggable `IVideoWriter`, PNG frame sequences by default and
  an ffmpeg writer as a pluggable implementation (DR-25; frame parity, not file parity); `HumanRendering` on the
  viewers (DR-20).
- **P6.6** Vector wrappers.
- **P6.7** `utils.play` and `PlayPlot` (keyboard input and a plot window in the viewers), the `FuncEnv` protocol
  without JAX, and the `utils.performance` benchmarks (DR-36). `AtariPreprocessing` is out of 2.0.
- **P6.8** Tests: port `tests/vector` (76) and `tests/wrappers` (146).
- **Exit:** both suites green; every vector cell equal between sync and async, bit for bit, in every execution mode.
- *Done 2026-11-09, the necessary part (the run's approval: "P6 (only necessary, defer for later whats not)"):*
  - *Necessary was read as: the vector API's core, which P4.4 and `make_vec` need, and the wrappers that `make`'s
    spec re-creation (P3's own Gymnasium tests) needs. Built: `AutoresetMode`, `IVectorEnv` and `VectorEnv<TObs, TAct>`
    of batched types with `VectorStepResult`/`VectorResetResult` (DR-21), the info merging of `_add_info`, the vector
    wrapper bases (`VectorWrapper`, `VectorObservationWrapper`, `VectorActionWrapper`, `VectorRewardWrapper`),
    `SyncVectorEnv` (typed, and untyped as `SyncVectorEnv`) in every autoreset mode with the observation modes,
    `reset_mask`, `copy`, `call`/`get_attr`/`set_attr`, `VectorizeMode` and `Gymnasium.MakeVec` typed and untyped (sync
    and vector entry point), `CartPoleVectorEnv` (P4.4) as CartPole-v0/v1's vector entry point, and the wrappers
    `NormalizeObservation`, `NormalizeReward`, `TimeAwareObservation`, `FlattenObservation` with `RunningMeanStd` and
    `WrapperFunctions.UpdateMeanVarCountFromMoments` (NumPy 2's dtype rules kept: float32 statistics stay float32,
    integer observations promote them, weak Python counts cast first).*
  - *Parity: 23 vector cells (`make_vec` in both modes, every autoreset mode over arrays, toy_text ints and int/float
    infos, Taxi's masks, Blackjack's tuples, seeds and masks, observation modes, attributes, descriptions, errors,
    frames, and `CartPoleVectorEnv` step for step with its state) and 16 wrapper cells (episodes and statistics of the
    four wrappers over float32, integer and tuple observations, a stack of three, the refusals), all passing. Unit:
    `VectorEnvTests`, and Gymnasium's three make-with-spec tests ported in place of P3's placeholder; inventory ratchet
    −16 names, the 65 P6 names left marked "deferred in this run".*
  - *Deferred, each with its design still approved: P6.2 `AsyncVectorEnv` (DR-22; `MakeVec(..., "async")` raises
    `NotImplementedError` naming P6.2), the other 24 single-env wrappers (P6.3's transforms and stateful wrappers,
    P6.4's `ResizeObservation`, P6.5's rendering wrappers; `create_zero_array` with them), P6.4's F8 (DR-23), P6.5's rendering wrappers
    (`HumanRendering`, so `make`'s human-rendering path still raises, and `RecordVideo`, DR-25), P6.6's vector wrappers
    (DR-24's names), P6.7 (`play`, `PlayPlot`, `FuncEnv`, `performance`, DR-36), and P6.8's ports of `tests/vector` and
    `tests/wrappers`.*
  - *Deviations, documented on the members: `CartPoleVectorEnv`'s rewards are float64 (DR-21; Gymnasium's are float32,
    the values equal); `SyncVectorEnv.NpRandom`/`NpRandomSeed` keep the base's meaning (the vector env's own generator
    and seed) where Gymnasium's return tuples of the sub-envs' (`GetAttr` gives those); a `reset_mask` stays in the caller's options dict (Python pops it); `TimeAwareObservation`'s observations
    are `object`s (the type depends on `flatten` at run time); `FlattenObservation` is typed for array results and
    refuses spaces whose flat form is not an array.*
- *Done 2026-11-10, the rest of P6.3 (the maintainer's "continue with P6"):*
  - *The stateless wrappers (`e63168e`): `TransformObservation` and its eight subclasses (`FilterObservation`,
    `FlattenObservation` rebased on it, `GrayscaleObservation`, `ReshapeObservation`, `RescaleObservation`,
    `DtypeObservation`, `AddRenderObservation`, `DiscretizeObservation`), `TransformAction` with `ClipAction`,
    `RescaleAction`, `DiscretizeAction`, `TransformReward` with `ClipReward`, and the helpers `create_zero_array` and
    `rescale_box`; 137 parity cells; 15 test files mirrored. NumSharp gap N-4 (`np.linspace` over float32 bounds, §7).*
  - *The stateful wrappers: `DelayObservation`, `FrameStackObservation`, `MaxAndSkipObservation`, `StickyAction`,
    `RepeatAction`; 34 parity cells (`StatefulObservationLiveParityTests`, `StatefulActionLiveParityTests`); their five
    test files mirrored, and the four test files of the wrappers built earlier (`test_normalize_observation`,
    `test_normalize_reward`, `test_time_aware_observation`, `test_record_constructor_args`; CarRacing's case is
    `[OpenBugs("P8")]`). `test_import_wrappers` is Python's module protocol: the inventory test covers its names.*
  - *Consequences recorded in DESIGN_REVIEW: typed delegates for `func`, subspace copies, `StickyAction`'s copy of the
    last action (D19), dropped observations left to the finalizer, `ObjectSpace.Unview` in the space and vector utils.*
- *Done 2026-11-10, P6.4 (F8, DR-23):* `Gym.Utils.Cv2.Resize`, the uint8 `INTER_AREA` and `INTER_LINEAR` paths of
  OpenCV 4.13.0's `cv::resize` in C#, byte for byte against live `cv2` (1,250 random images and Gymnasium-sized frames;
  OpenCV pinned in the parity stack as `opencv-python-headless==4.13.0.92`, `opencv.version` in the manifest, and in
  THIRD-PARTY-NOTICES); `ResizeObservation` on it, with its parity cells and `test_resize_observation.py` mirrored
  (CarRacing's cases `[OpenBugs("P8")]`). The float32 paths are not translated (no Gymnasium caller; DESIGN_REVIEW).
- *Done 2026-11-10, P6.5 (F9, DR-25, DR-20):* `RecordVideo` over `IVideoWriter` (`PngVideoWriter` by default,
  `FfmpegVideoWriter` for mp4), `HumanRendering` (on the viewers; `make`'s human-rendering path now applies it, so
  `TestMakeHumanRendering_LaterPhases` waits only for CarRacing), `AddWhiteNoise`, `ObstructView`, and
  `gymnasium.utils.save_video` as `Video.SaveVideo`/`Video.CappedCubicVideoSchedule`; 21 parity cells (frames, videos,
  clips, schedule) and `test_record_video.py`, `test_human_rendering.py`, `test_white_noise_rendering.py` and
  `tests/utils/test_save_video.py` mirrored.

### P8 Box2D

Specs: the port plan's §5.6 and §11.3; sources 2,979 lines.

- **P8.1 The Box2D engine (D5, DR-26, DR-29):** an assembly `Gym.Box2D` inside `Gym.NET.Environments` (DR-16; the
  NuGet id `Box2D.NET` is taken). It opens with DR-29's measurement: the contact solver's and broad-phase's hot loops
  written on NumSharp `NDArray`s and on float32 structs; each subsystem uses NDArrays only where they are no slower and
  no larger in code. Then the core faithful to Box2D 2.3.x's C++ (settings, math, the dynamic tree and pair ordering,
  distance, TOI, manifolds, contact solver, island, joints, ray casts, the contact-listener callback order) and the
  pybox2d-shaped facade the envs call. Engine goldens come from live pybox2d scenes (free fall, stacking, joints,
  contacts, ray casts), compared per step in float32 bits; the first ones also settle whether .NET's `MathF`
  trigonometry matches the CRT calls Box2D makes (§8).
- **P8.2** LunarLander-v3 and LunarLanderContinuous-v3 as `LunarLanderEnv` (DR-32, `[PyName("LunarLander")]`), with
  `heuristic` and a private nested `ContactDetector` (DR-28); the 8 LunarLander `[OpenBugs]` cells turn green.
- **P8.3** BipedalWalker-v3 and BipedalWalkerHardcore-v3 as `BipedalWalkerEnv` (lidar ray casts), with its own private
  `ContactDetector`.
- **P8.4** CarRacing-v3 as `CarRacingEnv` with `car_dynamics`; its score text drawn from glyph data recorded once from
  live pygame and SDL_ttf's shaded layout (DR-27), after the font-data license check; coordinate with PR #13.
- **P8.5** Delete the Aether LunarLander (D10) and the `Aether.Physics2D`, `ImageSharp.Drawing` and `Fonts`
  references in `Gym.Environments`; B-17…B-30 close.
- **Exit:** engine goldens; L3 and L4 for all five ids; no `[OpenBugs]` cell left in the suite.

### P9 MuJoCo (postponed out of 2.0, DR-35)

Specs: the port plan's §5.7 and §11.4; sources 6,153 lines. Nothing here starts before MuJoCo resumes; the approved
designs for that time are:

- **A managed port, 100%** (DR-30), inside `Gym.NET.Environments` with no native binaries, using NumSharp where it fits
  (`np.evaluate`, accelerated `np.*`).
- **Scope** (DR-33): the engine paths, MJCF compiler subset and renderer path that Gymnasium's 22 envs and their 14
  bundled models (primitive geoms only) reach. MuJoCo's engine alone is 2.56 MB of C in 90 files, and its compiler and
  renderer another 2.63 MB of C/C++, so this is the largest phase (XXL).
- **Proof**: bitwise against live Python `mujoco` at a pinned version (D6), after measuring whether native builds fuse
  multiply-adds; any non-bitwise quantity documented with its ulp bound.
- **Frames** (DR-31): MuJoCo's own `mjr_*` renderer, ported, drawing through a managed OpenGL binding in a hidden WGL or
  EGL context; best-effort, as the port plan's §1.2 allows.
- **Names** (DR-24): `AntEnvV4` / `AntEnvV5` and the other 10 pairs, each `[PyName("AntEnv")]`; models embedded
  (DR-19).

### P10 Release 2.0.0

- **P10.1 Docs.** Replace PR #24's `PRD.md` and `ROADMAP.md` (root and `docs/`), `docs/architecture`, `docs/sot` and
  `docs/plans/expressive-nibbling-goose.md` with docs derived from the plans, in Gym.NET's own voice; rewrite the
  README.
- **P10.2** The migration guide (port plan §16); `Gym.Legacy` stays (D9).
- **P10.3** One CartPole example on the new API; the rest removed (D13, B-36).
- **P10.4** Packages at 2.0.0 (DR-16): Gym.NET, Gym.NET.Rendering, Gym.NET.Environments (classic_control, toy_text
  and box2d with the `Gym.Box2D` engine), Gym.NET.Rendering.Avalonia, Gym.NET.Rendering.WinForm. Release notes include
  D18's advisory note and MuJoCo's postponement (DR-35).
- **P10.5** Merge `journey1` into `master` (D20), tag, publish.
- **Exit:** the port plan's definition of done (§1.5) for everything but MuJoCo; the inventory's expected-missing list
  holds only MuJoCo's names and `AtariPreprocessing` (DR-36).
- *Prepared 2026-11-09, P10.1–P10.4, with nothing merged, tagged, pushed or published (the run's approval: "prepare
  P10 but no publish or push"):*
  - *P10.1: PR #24's `PRD.md` and `ROADMAP.md` (root and `docs/`), `docs/architecture/`, `docs/sot/` and
    `expressive-nibbling-goose.md` removed. New: `docs/README.md` (index), `docs/architecture.md`,
    `docs/guides/getting_started.md`, `creating_custom_environments.md` and `development_and_testing.md` (both
    rewritten), the root README (also the packages' readme) and `THIRD-PARTY-NOTICES.md` brought to 2.0's scope
    (Gymnasium's areas and assets, CPython's PSF notice for `Difflib`/`PyJson`/`PyMath`, P7's pygame-derived drawing).
    Every code sample was compiled and run from a consumer project outside the tree (NumSharp's analyzer active, no
    warning); the guide's custom env passes `EnvChecker.CheckEnv`.*
  - *P10.2: `docs/guides/migrating_to_2.0.md`, from 0.2: the last published release (2023, netcoreapp3.1/net6.0 on
    NumSharp.Lite 0.1.12), since Gym.NET never had a 1.x.*
  - *P10.3: `examples/CartPole`, the cross-entropy method over linear policies with 100 carts at once through
    `MakeVec` (CartPoleVectorEnv), solved by Gymnasium's measure (475 averaged over 100 unseen episodes) in 7 generations,
    about 1.6 s, reproducible; part of `Gym.NET.sln`, so an API change breaks its build. It needs no learning library:
    the old runners' NeuralNetwork.NET (last released in 2020) depends on ImageSharp 1.0.0-beta0005 and the Alea GPU
    runtime. The
    `ReinforcementLearning` examples are removed (B-36). Its `--render` path compiles but was not run (it opens a window).*
  - *P10.4: the five packages at 2.0.0 (assembly and file versions 2.0.0.0), descriptions in Gym.NET's own voice, tags,
    a release-notes link, and the XML documentation (`GenerateDocumentationFile` on the four that lacked it). `dotnet
    pack Gym.NET.sln` gives exactly the five, depending on each other at 2.0.0, the generator under
    `analyzers/dotnet/cs`, the registration targets under `buildTransitive`. Draft release notes:
    `docs/releases/RELEASE_2.0.0.md` (D18's five advisories with the consumer's suppression lines, DR-35, every
    deviation, gap and deferral).*
  - *Found while packaging: the generator copied `<paramref>`s into the options records' property docs, a broken
    reference (CS1734) in any project that generates documentation, consumers' included; they now link the sibling
    property, pinned by `GymRecordsTests.OptionsRecord_DocumentationLinksTheOtherOptions` (shown to fail with the defect
    planted). Three crefs to `np.array_equal` fixed. The other documentation warnings are in pre-2.0 code
    (`Gym.Internal`, the 0.2 LunarLander, the viewers): CS1591 and about a dozen broken crefs, tolerated (DR-05).*
  - *Not done: P10.5. Before it: P8, the deferred part of P6 (69 inventory names) and the `aaline` decision (D4); the
    exit's inventory check is unmet until then (69 P6 and 12 P8 names besides MuJoCo's 33, `AtariPreprocessing` and the
    n/a and by-design entries).*

## 6. How every step is done

- **Designs first.** The step's designs are approved in [`DESIGN_REVIEW.md`](DESIGN_REVIEW.md). A design question
  found while implementing (a choice no DR-xx covers) stops the step until the maintainer reviews it the same way.
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
| `AssertionError`, `NotImplementedError` in NumSharp | F2.4, D17, DR-02 | `AssertionError` added (its parameterless message is N-1); `NotImplementedError` and `ImportError` still absent (N-3) | Trial build: 4 × CS0104 |
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

New items. Under DR-01 each is documented here and proven by an `[OpenBugs("N-x")]` test that asserts NumPy's or
Python's behavior and fails until NumSharp provides it; a Gym.NET workaround exists only where a step is stuck without
one:

- **N-1.** NumSharp's parameterless `AssertionError()` carries the message `"AssertionError"`; Python's
  `str(AssertionError())` is `""`. DR-02: NumSharp changes it. Gym.NET throws `new AssertionError()` at bare asserts,
  and the tests and cells that compare such a message are `[OpenBugs("N-1")]` until the NumSharp release (R0.1).
- **N-2.** `np.array(string[], "c")` (and `np.asarray`) should split each string into characters, as NumPy does for the
  `'c'` code: `np.asarray(["SFFF", "FHFH", "FFFH", "HFFG"], dtype="c")` is a `(4, 4)` `S1` array. P5 is stuck without
  it, so it carries a local map builder until NumSharp has it (P5.1): `ToyTextUtils.CharMap`, which builds NumPy's shape,
  dtype and bytes from one-character strings. Proven by `NumSharpGapTests.N2_CharCode_SplitsStringsIntoCharacters`
  (`[OpenBugs("N-2")]`).
- **N-3.** DR-02: NumSharp owns every Python and NumPy exception. It lacks `NotImplementedError` and `ImportError`
  today (and any other a later port needs, such as `ModuleNotFoundError`), and each needs public constructors
  `()`, `(message)` and `(message, innerException)`. `AttributeError` can't carry an inner exception today, so
  `get_wrapper_attr` loses Python's `raise ... from e` chain (found in P1). Gym.NET keeps its own two until NumSharp
  has them.
- **N-4.** `np.linspace` takes only C# scalars, which stand for Python floats, so it has no form of NumPy's
  `np.linspace(low[i], high[i], n)` over float32 NumPy scalars or arrays, which computes in float32: four of the eleven
  samples of `np.linspace(np.float32(0.1), np.float32(0.7), 11)` differ from the float64 samples cast to float32 (found
  in P6.3, 2026-11-10). `DiscretizeObservation` and `DiscretizeAction` were stuck without NumPy's float32 bin edges, so
  they compute them through `WrapperFunctions.Linspace` until NumSharp has array bounds. Proven by
  `NumSharpGapTests.N4_Linspace_TakesArrayBoundsAndKeepsTheirDtype` (`[OpenBugs("N-4")]`).

## 8. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| ImageSharp 2.1.13 stays flagged (D18) | Consumers' restores report three high and two moderate advisories; under `TreatWarningsAsErrors` their builds fail | Release notes and README name the five advisories and why Gym.NET's use doesn't reach them; a new advisory gets the same review (R0.2) |
| One branch until 2.0 (D20) | A long-lived branch with many commits merges once; contributors keep targeting `master` | The branch sits on `master`'s tip (13 ahead, 0 behind), so while `master` doesn't move the final merge is a fast-forward; CI runs on the branch from P0r |
| pygame `draw` from our own code (D4) | A primitive that won't converge stalls P7 | The random-shape harness reports the first differing pixel and its neighbourhood; the maintainer decides before any other route |
| Box2D float32 trigonometry (D5) | Box2D calls `sinf`/`cosf`/`atan2f` from MSVC's CRT; .NET's `MathF` must agree on win-x64 | The first engine goldens include rotation-heavy scenes; a mismatch surfaces at P8.1, before any env |
| Box2D on NumSharp (DR-29) | `NDArray`s in the solver's hot loops could cost performance or code size | P8.1 measures both forms per subsystem before choosing; float32 structs where NDArrays lose |
| CarRacing's recorded glyphs (DR-27) | The font is GNU FreeFont (GPL with a font exception); its glyph data ships in Gym.NET | License check before P8.4; if the data can't ship, the text design goes back to review before P8.4 continues |
| The source generator (DR-13, DR-34) | One generator project feeds composite records and options records; a defect there reaches every env | Its output is covered by the same parity cells as hand-written code (`repr`, `==`, kwargs binding) |
| MuJoCo postponed (DR-35) | 2.0 ships without Gymnasium's mujoco family | Release notes say so; DR-30, DR-31 and DR-33 keep the approved design for when it resumes (a managed port is XXL, and its bitwise parity with native MuJoCo is uncertain until measured) |
| Gymnasium moves past v1.4.0 | Work against a superseded pin | Re-pin only at a release tag, by the port plan's §14.10, between phases |
