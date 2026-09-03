# Architecture: Core Lifecycle & Spaces Hierarchy

[Back to Documentation Index](../README.md) · [View PRD](../PRD.md)

---

## 1. Core Lifecycle Overview

The core environment abstraction in `Gym.NET` centers upon `IEnv` and the base `Env` class, establishing the standardized reinforcement learning interaction loop.

```mermaid
sequenceDiagram
    autonumber
    actor Agent as RL Agent / PPO Core
    participant Env as Gymnasium Env (IEnv)
    participant Space as Action/Obs Space
    participant Viewer as IEnvViewer

    Agent->>Env: Reset(seed, options)
    Env->>Env: Seed PRNG (np.random.RandomState)
    Env-->>Agent: (Observation, Information)

    loop Rollout Step Loop
        Agent->>Space: Sample() or Compute Action
        Space-->>Agent: Action
        Agent->>Env: Step(action)
        Env->>Env: Compute Dynamics / Physics
        Env-->>Agent: StepResult(Obs, Reward, Terminated, Truncated, Info)
        opt If render_mode enabled
            Agent->>Env: Render()
            Env->>Viewer: Render(Image)
        end
    end

    Agent->>Env: Close()
```

---

## 2. The 5-Tuple Step & 2-Tuple Reset Contract

### 2.1 `ResetResult` (2-Tuple)
```csharp
namespace Gym.Core {
    public readonly record struct ResetResult(
        NDArray Observation,
        Dict Information
    ) {
        public void Deconstruct(out NDArray observation, out Dict information) {
            observation = Observation;
            information = Information;
        }
    }
}
```

### 2.2 `StepResult` (5-Tuple)
```csharp
namespace Gym.Core {
    public readonly record struct StepResult(
        NDArray Observation,
        float Reward,
        bool Terminated,
        bool Truncated,
        Dict Information
    ) {
        public void Deconstruct(
            out NDArray observation,
            out float reward,
            out bool terminated,
            out bool truncated,
            out Dict information
        ) {
            observation = Observation;
            reward = Reward;
            terminated = Terminated;
            truncated = Truncated;
            information = Information;
        }

        public bool Done => Terminated || Truncated;
    }
}
```

---

## 3. Spaces Hierarchy

Gymnasium provides 10 standard spaces for observation and action representation:

```mermaid
classDiagram
    class Space {
        <<abstract>>
        +Shape Shape
        +Type DType
        +Sample(NDArray mask) NDArray
        +Contains(object x) bool
        +Seed(int seed) void
    }

    class Box {
        +NDArray Low
        +NDArray High
        +NDArray BoundedLow
        +NDArray BoundedHigh
        +IsBounded(manner) bool
    }

    class Discrete {
        +int N
        +int Start
    }

    class MultiDiscrete {
        +int[] Nvec
        +int[] Start
    }

    class MultiBinary {
        +int[] Dimensions
    }

    class TupleSpace {
        +IReadOnlyList~Space~ Spaces
    }

    class DictSpace {
        +IReadOnlyDictionary~string, Space~ Spaces
    }

    class TextSpace {
        +int MinLength
        +int MaxLength
        +string Charset
    }

    class SequenceSpace {
        +Space Subspace
    }

    Space <|-- Box
    Space <|-- Discrete
    Space <|-- MultiDiscrete
    Space <|-- MultiBinary
    Space <|-- TupleSpace
    Space <|-- DictSpace
    Space <|-- TextSpace
    Space <|-- SequenceSpace
```

### 3.1 Space Implementation Invariants
- **Deterministic Sampling**: `space.Seed(seed)` enforces reproducible random generation.
- **Type Bounds Safety**: Values sampled from `Box` are guaranteed to reside within $[Low, High]$.
- **Action Masking**: `Discrete.Sample(mask)` accepts a boolean selector vector masking out invalid actions.
