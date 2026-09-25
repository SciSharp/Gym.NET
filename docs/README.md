# Gym.NET Documentation Hub

Welcome to the central documentation hub for **Gym.NET**, the native C# (.NET 8/10) port of Farama Gymnasium.

---

## 1. Documentation Index

### Core Specifications & Roadmap
- [**Product Requirements Document (PRD)**](./PRD.md) — Complete functional, architectural, and quality specifications.
- [**Milestone Roadmap**](./ROADMAP.md) — Live deliverables matrix and implementation phases.

### System Architecture
- [**Core Lifecycle & Spaces**](./architecture/core_lifecycle_and_spaces.md) — Modern `Reset(seed, options)`, 5-tuple `StepResult`, and space hierarchy.
- [**Wrapper Architecture**](./architecture/wrapper_architecture.md) — Transformation pipeline, `TimeLimit`, monitoring, and wrappers.
- [**Vector Environments**](./architecture/vector_environments.md) — Synchronous (`SyncVectorEnv`) and asynchronous (`AsyncVectorEnv`) vectorization.
- [**Decoupled Rendering**](./architecture/decoupled_rendering.md) — Headless `NullEnvViewer` vs. Avalonia / WinForms UI rendering.
- [**Agent Architecture & Ontology Graph**](./architecture/ontology.json) — Machine-readable component ontology and dependency graph for guide agents.

### Source of Truth (SOT)
- [**Gymnasium SOT Mapping**](./sot/gymnasium_sot_mapping.md) — Canonical mapping to Farama Gymnasium (`refs/Gymnasium`).
- [**Golden Trajectory Baselines**](./sot/golden_trajectory_baseline.md) — Deterministic seed test vectors and trajectory verification.

### Developer Guides
- [**Development & Testing Guide**](./guides/development_and_testing.md) — Building, running tests, single test filtering, and TDD workflow.
- [**Creating Custom Environments**](./guides/creating_custom_environments.md) — Step-by-step guide for implementing new Gymnasium-compliant C# environments.

---

## 2. Quickstart Example

```csharp
using Gym.Environments.Envs.Classic;
using Gym.Rendering.Avalonia;
using NumSharp;

// 1. Instantiate environment with render mode and viewer factory
var env = new CartPoleEnv(renderMode: "human", viewerFactory: AvaloniaEnvViewer.Factory);

// 2. Reset environment with deterministic seed
var (observation, info) = env.Reset(seed: 42);

bool terminated = false;
bool truncated = false;

while (!terminated && !truncated) {
    // 3. Sample an action from the action space
    var action = env.ActionSpace.Sample();

    // 4. Step dynamics (returns 5-tuple)
    var step = env.Step(action);
    observation = step.Observation;
    terminated = step.Terminated;
    truncated = step.Truncated;

    // 5. Render frame
    env.Render();
}

env.Close();
```
