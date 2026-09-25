# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 1. Project Overview

Gym.NET is a native C# (.NET) port of OpenAI Gym / Farama Gymnasium, providing a standardized reinforcement learning environment toolkit and benchmark suite. It is part of the SciSharp STACK ecosystem and uses `NumSharp` for multidimensional tensor/array operations and `SixLabors.ImageSharp` for rendering.

- **Canonical SOT Submodule:** `refs/Gymnasium` (Farama Foundation Gymnasium Git submodule for behavioral, mathematical, and algorithmic parity).

---

## 2. Solution & Project Architecture

The repository is structured around the `Gym.NET.sln` solution:

```
refs/Gym.NET/
├── src/
│   ├── Gym/                       # Core API (port of gymnasium/core.py), spaces, wrappers, utilities, legacy shim
│   │   ├── Envs/                  # IEnv, Env<TObs, TAct>, Wrapper family (+ IWrapper), Info, Metadata, RenderResult, EnvExtensions (StepAsync)
│   │   │   └── Registration/      # EnvSpec, WrapperSpec, EnvCreator, EnvRegistry (parse_env_id/get_env_id; make & registry in P3)
│   │   ├── Spaces/                # Space (base), Box (bounded/continuous), Discrete
│   │   ├── Observations/          # StepResult<TObs> (5-tuple), ResetResult<TObs>, DoneStepResult<TObs> (pre-0.26 4-tuple)
│   │   ├── Wrappers/              # Common.cs: TimeLimit, OrderEnforcing (port of gymnasium/wrappers/common.py)
│   │   ├── Legacy/                # [Obsolete] pre-0.26 API: Step, ILegacyEnv, LegacyEnvAdapter, VecEnv family
│   │   ├── Internal/              # Dict (legacy info type), CircularQueue, Dynamic/Merging/Reflection helpers
│   │   ├── Utils/                 # Seeding (PCG64), PyBuiltins, PyMath (F3), PyAttributes, PyEquality, PyCopy, StepApiCompatibility, RecordConstructorArgs, Warnings
│   │   ├── Logger.cs              # Port of gymnasium.logger (colourised warnings, Python's default filters)
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

- **Environment Contract (`Env<TObs, TAct>` / `IEnv`), Gymnasium's since the core-API port (P1):**
  - `ResetResult<TObs> Reset(BigInteger? seed = null, options = null)`: `(obs, info)`; a seed reseeds `NpRandom` with Gymnasium's `Generator(PCG64(SeedSequence(seed)))`, `null` keeps the stream. Implementations start with `SeedNpRandom(seed)`, the port of `super().reset(seed=seed)`.
  - `StepResult<TObs> Step(TAct action)`: `(obs, reward (float64), terminated, truncated, info)`; `IEnv.Step(object)` converts untyped actions (`ConvertAction`).
  - `RenderResult Render()`: per the `render_mode` fixed at construction (`RenderResult.Rgb` for `rgb_array`; `human` renders inside reset/step and returns `None`).
  - `NpRandom` (lazy, entropy-seeded on first use) / `NpRandomSeed` (-1 once a generator is assigned), `Metadata`, `Spec`, `Unwrapped`, `Close()` / `Dispose()`.
  - Python attributes by Python name: `HasWrapperAttr`/`GetWrapperAttr`/`SetWrapperAttr` and `Gym.Utils.PyAttributes` (`[PyName("_x")]` or snake_case of the property).
- **Wrappers:** `Wrapper<TWObs, TWAct, TObs, TAct>` (and `Wrapper<TObs, TAct>`) forward to `Env`; `ObservationWrapper`, `RewardWrapper`, `ActionWrapper` add the hooks; `TimeLimit` and `OrderEnforcing` live in `Gym.Wrappers`. Until `make` lands (P3), build its stack by hand: `new TimeLimit<NDArray, long>(new OrderEnforcing<NDArray, long>(new CartPoleEnv()), 500)`.
- **Pluggable Viewer Architecture:** Environments accept an `IEnvironmentViewerFactoryDelegate` used for `render_mode="human"` (`NullEnvViewer.Factory` headless, `AvaloniaEnvViewer.Factory`, `WinFormEnvViewer.Factory`); human frames are paced by `Gym.Rendering.Clock` at `metadata["render_fps"]`.
- **Legacy API (`Gym.Legacy`, `[Obsolete]`):** `LegacyEnvAdapter` exposes a ported env through the pre-0.26 4-tuple API (`Seed(s)` + `Reset()` is `Reset(seed: s)`); the baselines-style `IVecEnv` / `VecEnv` / `DummyVecEnv` work over `ILegacyEnv` until `gymnasium.vector` (P6).
- **Observation & Action Spaces:** Strongly-typed bounds checking and random sampling via `Box` (multi-dimensional continuous floats/bounds) and `Discrete` (categorical integer actions); `Seed(BigInteger)`.

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

Tests are located in `tests/Gym.Tests/` using MSTest.

```powershell
# Run full test suite
dotnet test tests/Gym.Tests/Gym.Tests.csproj

# Run tests targeting a specific framework
dotnet test tests/Gym.Tests/Gym.Tests.csproj -f net6.0-windows

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
```

### 3.4 Submodule Management

```powershell
# Initialize the Gymnasium and NumSharp SOT submodules. The build compiles NumSharp from refs/NumSharp (eng/NumSharp.props).
# Not --recursive: that would also clone NumSharp's own submodules (numpy, OpenBLAS, onnxruntime, ...), which the build does not need.
git submodule update --init refs/Gymnasium refs/NumSharp
```
