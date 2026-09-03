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
│   ├── Gym/                       # Core abstractions, spaces, vector environments, threading
│   │   ├── Envs/                  # IEnv, Env, GoalEnv, VecEnv, DummyVecEnv, VecEnvWrapper
│   │   ├── Spaces/                # Space (base), Box (bounded/continuous), Discrete
│   │   ├── Observations/          # Step (Observation, Reward, Done, Information)
│   │   ├── Internal/              # Dict, CircularQueue, DistributedScheduler, Threading
│   │   └── Exceptions/            # InvalidActionError, AlreadySteppingError, NotSteppingError
│   │
│   ├── Gym.Environments/          # Concrete environment implementations
│   │   ├── Envs/Classic/          # CartPoleEnv (CartPole-v1)
│   │   ├── Envs/Aether/           # LunarLanderEnv (Aether.Physics2D 2D physics simulation)
│   │   └── Rendering/             # IEnvViewer, IEnvironmentViewerFactoryDelegate, NullEnvViewer
│   │
│   ├── Gym.Rendering.Avalonia/    # Cross-platform GUI rendering (AvaloniaEnvViewer, StaticAvaloniaApp)
│   └── Gym.Rendering.WinForm/     # Windows Forms GUI rendering (WinFormEnvViewer)
│
├── tests/
│   └── Gym.Tests/                 # MSTest test suite for environments, spaces, and multi-instance concurrency
│
├── examples/                      # Standalone reinforcement learning sample runners
│   └── ReinforcementLearning/     # CartPole neural network training via parameters vs image observations
│
└── refs/
    └── Gymnasium/                 # Submodule: Farama-Foundation/Gymnasium (Python Master SOT)
```

### Core Design Patterns & Mechanisms

- **Environment Contract (`IEnv` / `Env`):** Standard RL lifecycle methods:
  - `NDArray Reset()`: Resets the state and returns the initial observation.
  - `Step Step(object action)`: Steps environment dynamics; returns `(Observation, Reward, Done, Information)`.
  - `Image Render(string mode = "human")`: Generates/renders frame via `SixLabors.ImageSharp`.
  - `void Seed(int seed)`: Seeds pseudorandom number generators via `NumPyRandom`.
  - `void CloseEnvironment()` / `Dispose()`: Releases rendering and physics resources.
- **Pluggable Viewer Architecture:** Environments accept an `IEnvironmentViewerFactoryDelegate` allowing execution in headless mode (`NullEnvViewer.Factory`) or interactive GUI mode (`AvaloniaEnvViewer.Factory`, `WinFormEnvViewer.Factory`).
- **Vectorized Environments (`IVecEnv` / `VecEnv` / `DummyVecEnv`):** Synchronous/asynchronous batch stepping across multiple parallel environment instances.
- **Observation & Action Spaces:** Strongly-typed bounds checking and random sampling via `Box` (multi-dimensional continuous floats/bounds) and `Discrete` (categorical integer actions).

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

### 3.3 Submodule Management

```powershell
# Initialize and synchronize Gymnasium SOT submodule
git submodule update --init --recursive
```
