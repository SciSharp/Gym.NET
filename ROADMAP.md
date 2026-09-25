# Gym.NET: Live Milestone Roadmap & Deliverables Matrix

This document tracks the live milestone progress, implementation status, and deliverables matrix for **Gym.NET**, migrating from legacy OpenAI Gym to 100% **Farama Gymnasium** (`refs/Gymnasium`) parity.

---

## 1. High-Level Milestone Overview

| Milestone | Focus Area | Status | Target Frameworks | Verification Gate |
| :--- | :--- | :---: | :--- | :--- |
| **M0** | .NET 8/10 Modernization & Security | **COMPLETED** | .NET 8.0, .NET 10.0 | 15/15 Automated Tests Passing |
| **M1** | Core Lifecycle & Spaces Suite | **PLANNED** | .NET 8.0, .NET 10.0 | Golden Space Sampling & Bounds Tests |
| **M2** | Wrapper Pipeline & Vector Environments | **PLANNED** | .NET 8.0, .NET 10.0 | Wrapper Truncation & Autoreset Tests |
| **M3** | Classical Control & Physics Environments | **PLANNED** | .NET 8.0, .NET 10.0 | SOT Golden Seed Trajectory Comparisons |
| **M4** | Live Documentation Hub & Agent Ontology | **IN PROGRESS** | Markdown / JSON | PRD, Architecture Specs, Ontology Graph |

---

## 2. Deliverables Matrix

### Milestone 0: .NET 8/10 Modernization & Security (Completed)
- [x] Multi-target projects to `net8.0` and `net10.0` (with `net8.0-windows;net10.0-windows` for UI).
- [x] Upgrade `SixLabors.ImageSharp` to `2.1.13` resolving high/moderate CVE security advisories.
- [x] Upgrade `SixLabors.ImageSharp.Drawing` to stable `1.0.0` and `SixLabors.Fonts` to `1.0.1`.
- [x] Upgrade `Avalonia` & `Avalonia.Desktop` to `0.10.22`.
- [x] Upgrade test framework to `MSTest 3.8.2` and `Microsoft.NET.Test.Sdk 17.14.1`.
- [x] Fix 0D scalar shape bounds and sampling logic in `Box.cs`.
- [x] Fix Avalonia application lifetime multi-instance reuse in `StaticAvaloniaApp.cs`.
- [x] Guard `WinFormEnvViewer.cs` against headless modal dialog crashes.

### Milestone 1: Core Lifecycle & Spaces Suite
- [ ] Implement modern `(NDArray Obs, Dict Info) Reset(int? seed = null, Dict options = null)`.
- [ ] Implement modern 5-tuple `StepResult Step(object action)` with `(Obs, Reward, Terminated, Truncated, Info)` and tuple deconstruction.
- [ ] Implement `MultiDiscrete` space with independent dimension bounds.
- [ ] Implement `MultiBinary` space with multi-dimensional binary arrays.
- [ ] Implement `TupleSpace` for Cartesian products of heterogeneous subspaces.
- [ ] Implement `DictSpace` for named composite key-value subspaces.
- [ ] Implement `TextSpace`, `SequenceSpace`, `GraphSpace`, `OneOfSpace`.
- [ ] Complete TDD test suite for all spaces verifying sampling, masking, bounds, and containment.

### Milestone 2: Wrapper Pipeline & Vector Environments
- [ ] Implement `Wrapper`, `ObservationWrapper`, `ActionWrapper`, `RewardWrapper` base abstractions.
- [ ] Implement `TimeLimit` wrapper managing truncation limits.
- [ ] Implement `TransformObservation` and `TransformReward` functional transformation wrappers.
- [ ] Implement `ClipAction` and `RescaleAction` action space adapters.
- [ ] Implement `RecordEpisodeStatistics` monitoring return and episode length.
- [ ] Implement `Autoreset` and `OrderEnforcing` wrappers.
- [ ] Modernize `VectorEnv`, `SyncVectorEnv`, `AsyncVectorEnv` with batched stepping and `final_observation` tracking.

### Milestone 3: Classical Control & Physics Environments
- [ ] Implement `CartPole-v1` with Gymnasium Euler kinematics and termination boundaries.
- [ ] Implement `Pendulum-v1` with continuous torque physics and trigonometry observation vectors.
- [ ] Implement `MountainCar-v0` (discrete) and `MountainCarContinuous-v0` (continuous).
- [ ] Implement `Acrobot-v1` two-link double pendulum.
- [ ] Modernize `LunarLander-v3` with discrete and continuous physics simulation via `Aether.Physics2D`.
- [ ] Implement `BipedalWalker-v3` 4-joint locomotion.
- [ ] Author automated Golden SOT test suites comparing seeded rollout trajectories against Python Gymnasium.

### Milestone 4: Live Documentation Hub & Agent Ontology
- [x] Author `PRD.md` capturing functional, architectural, and quality requirements.
- [x] Author `ROADMAP.md` tracking milestone deliverables.
- [ ] Author `docs/architecture/` specs (Core Lifecycle, Spaces, Wrappers, Vector Environments, Decoupled Rendering).
- [ ] Author `docs/architecture/ontology.json` and Mermaid dependency graph for autonomous guide agents.
- [ ] Author `docs/sot/` mapping and golden trajectory baselines.
- [ ] Author `docs/guides/` for development, testing, and custom environment authoring.
