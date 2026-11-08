# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 1. Project Overview

Gym.NET is a native C# (.NET) reinforcement learning environment toolkit and benchmark suite, API-compatible with Farama Gymnasium (the successor of OpenAI Gym) and aligned with it byte for byte. It is its own codebase: Gymnasium is the reference it measures itself against, not an identity it claims. It is part of the SciSharp STACK ecosystem and uses `NumSharp` for multidimensional tensor/array operations and `SixLabors.ImageSharp` for rendering.

- **Canonical SOT Submodule:** `refs/Gymnasium` (Farama Foundation Gymnasium Git submodule for behavioral, mathematical, and algorithmic parity).
- **Roadmap:** `docs/plans/COMPLETION_PLAN.md` orders the remaining work to one 2.0 release (phases R0, P0r, P2, P3, P7, P4, P5, P6, P8, P10; MuJoCo's P9 is postponed) and records the 2026-11-08 decisions; `docs/plans/GYMNASIUM_PORT_PLAN.md` holds the per-area specifications.
- **Design gate:** `docs/plans/DESIGN_REVIEW.md` holds the approved implementation designs (DR-01…DR-36). Implement only what an approved design covers; a design question no DR-xx answers goes back to the maintainer for review before the step continues.
- **Working branch:** `journey1` until 2.0 (decision D20); it merges into `master` once, at the release.

---

## 2. Solution & Project Architecture

The repository is structured around the `Gym.NET.sln` solution:

```
refs/Gym.NET/
├── src/
│   ├── Gym/                       # Core API (env contract, wrappers, results, errors), spaces, utilities, legacy shim
│   │   ├── Envs/                  # IEnv, Env<TObs, TAct> (Step, ValueTask StepAsync), Wrapper family (+ IWrapper), Info, Metadata, RenderResult
│   │   │   └── Registration/      # EnvSpec, WrapperSpec, EnvCreator, EnvRegistry (parse_env_id/get_env_id; make & registry in P3)
│   │   ├── Spaces/                # Space (base), Box (bounded/continuous), Discrete
│   │   ├── Observations/          # StepResult<TObs> (5-tuple), ResetResult<TObs>, DoneStepResult<TObs> (pre-0.26 4-tuple)
│   │   ├── Wrappers/              # Common.cs: TimeLimit, OrderEnforcing (reference: gymnasium/wrappers/common.py)
│   │   ├── Legacy/                # [Obsolete] pre-0.26 API: Step, ILegacyEnv, LegacyEnvAdapter, VecEnv family
│   │   ├── Internal/              # Dict (legacy info type), CircularQueue, Dynamic/Merging/Reflection helpers
│   │   ├── Utils/                 # Seeding (PCG64), PyBuiltins, PyMath (F3), PyAttributes, PyEquality, PyCopy, StepApiCompatibility, RecordConstructorArgs, Warnings
│   │   ├── Logger.cs              # Warnings as gymnasium.logger issues them (colourised, Python's default filters)
│   │   └── Exceptions/            # Every gymnasium.error class (Error, ResetNeeded, ...), AssertionError, NotImplementedError, ImportError; legacy InvalidActionError etc.
│   │
│   ├── Gym.Rendering/             # pygame-compatible rasterizer: Surface, GfxDraw (SDL_gfx 2.0.26 port), Transform, SurfArray, Vector2
│   ├── Gym.Environments/          # Concrete environment implementations
│   │   ├── Envs/Classic/          # CartPoleEnv (CartPole-v1, byte-for-byte port), ClassicControlUtils
│   │   ├── Envs/Aether/           # LunarLanderEnv (Aether.Physics2D 2D physics simulation)
│   │   └── Rendering/             # IEnvViewer, IEnvironmentViewerFactoryDelegate, NullEnvViewer
│   │
│   ├── Gym.Rendering.Avalonia/    # Cross-platform GUI rendering (AvaloniaEnvViewer, StaticAvaloniaApp)
│   └── Gym.Rendering.WinForm/     # Windows Forms GUI rendering (WinFormEnvViewer)
│
├── tests/
│   ├── Gym.Tests/                 # MSTest test suite for environments, spaces, and multi-instance concurrency
│   │   └── Gymnasium/             # refs/Gymnasium/tests mirrored one to one (test_core.py → TestCore.cs), namespace Gym.Tests.Gymnasium.* (DR-06)
│   └── Gym.Tests.Parity/          # Live parity gate: embedded CPython runs refs/Gymnasium, compared byte/pixel-exact (see its CLAUDE.md)
│
├── tools/parity/                  # Parity stack: manifest.json (pins), requirements.txt, setup_venv.{ps1,sh}; audit probes
│
├── examples/                      # Standalone reinforcement learning sample runners
│   └── ReinforcementLearning/     # CartPole neural network training via parameters vs image observations
│
└── refs/
    └── Gymnasium/                 # Submodule: Farama-Foundation/Gymnasium (Python Master SOT)
```

### Core Design Patterns & Mechanisms

- **Environment Contract (`Env<TObs, TAct>` / `IEnv`), Gymnasium's shape since the core-API work (P1):**
  - `ResetResult<TObs> Reset(BigInteger? seed = null, options = null)`: `(obs, info)`; a seed reseeds `NpRandom` with Gymnasium's `Generator(PCG64(SeedSequence(seed)))`, `null` keeps the stream. Implementations start with `SeedNpRandom(seed)` (Gymnasium's `super().reset(seed=seed)`).
  - `StepResult<TObs> Step(TAct action)`: `(obs, reward (float64), terminated, truncated, info)`; `IEnv.Step(object)` converts untyped actions (`ConvertAction`).
  - `ValueTask<StepResult<TObs>> StepAsync(TAct action, CancellationToken ct = default)` (and `IEnv.StepAsync(object)`): runs `Step` on the calling thread and completes synchronously, never through `Task.Run`; a failure or a cancellation before the step surfaces through the task. Override it only for an env with real asynchronous work; wrappers do not forward it, so a wrapper's `Step` override always applies.
  - `RenderResult Render()`: per the `render_mode` fixed at construction (`RenderResult.Rgb` for `rgb_array`; `human` renders inside reset/step and returns `None`).
  - `NpRandom` (lazy, entropy-seeded on first use) / `NpRandomSeed` (-1 once a generator is assigned), `Metadata`, `Spec`, `Unwrapped`, `Close()` / `Dispose()`.
  - Python attributes by Python name: `HasWrapperAttr`/`GetWrapperAttr`/`SetWrapperAttr` and `Gym.Utils.PyAttributes` (`[PyName("_x")]` or snake_case of the property).
- **Wrappers:** `Wrapper<TWObs, TWAct, TObs, TAct>` (and `Wrapper<TObs, TAct>`) forward to `Env`; `ObservationWrapper`, `RewardWrapper`, `ActionWrapper` add the hooks; `TimeLimit` and `OrderEnforcing` live in `Gym.Wrappers`. Until `make` lands (P3), build its stack by hand: `new TimeLimit<NDArray, long>(new OrderEnforcing<NDArray, long>(new CartPoleEnv()), 500)`.
- **Pluggable Viewer Architecture:** Environments accept an `IEnvironmentViewerFactoryDelegate` used for `render_mode="human"` (`NullEnvViewer.Factory` headless, `AvaloniaEnvViewer.Factory`, `WinFormEnvViewer.Factory`); human frames are paced by `Gym.Rendering.Clock` at `metadata["render_fps"]`.
- **Legacy API (`Gym.Legacy`, `[Obsolete]`):** `LegacyEnvAdapter` exposes any env through the pre-0.26 4-tuple API (`Seed(s)` + `Reset()` is `Reset(seed: s)`; the reward stays float64, as everywhere in Gym.NET); the baselines-style `IVecEnv` / `VecEnv` / `DummyVecEnv` work over `ILegacyEnv` until `gymnasium.vector` (P6).
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
2.3.10). Cells Gym.NET does not pass yet are `[OpenBugs("B-xx")]` (NumSharp's convention): gated runs exclude them. Full
guide: `tests/Gym.Tests.Parity/CLAUDE.md`; design: `docs/plans/GYM_PARITY_INTEROP_TESTING.md`.

```powershell
# Once per machine without the stack: creates tools/parity/.venv, which the suite finds on its own
pwsh tools/parity/setup_venv.ps1

# The gate (what must pass)
dotnet test tests/Gym.Tests.Parity/Gym.Tests.Parity.csproj --filter "TestCategory!=OpenBugs"

# The open gaps, each printing its first divergence from Gymnasium
dotnet test tests/Gym.Tests.Parity/Gym.Tests.Parity.csproj --filter "TestCategory=OpenBugs" --logger "console;verbosity=detailed"

# The goldens (tests/Golden), replayed without Python; record them with GYMNET_PARITY_RECORD=1 (see its CLAUDE.md)
dotnet test tests/Gym.Tests.Parity/Gym.Tests.Parity.csproj --filter "TestCategory=Golden"
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
