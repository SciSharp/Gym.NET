# Architecture: Vectorized Environments & Batched Execution

[Back to Documentation Index](../README.md) · [View PRD](../PRD.md)

---

## 1. Vectorized Environments Overview

Vectorized environments (`IVectorEnv`) allow batching state transitions across $N$ parallel sub-environments, scaling RL sample collection throughput.

```mermaid
graph TD
    Agent["PPO Core / RL Agent"] -->|Batched Actions [N, ...]| VectorEnv["VectorEnv (IVectorEnv)"]
    
    subgraph Execution ["Execution Engine"]
        VectorEnv --> Sync["SyncVectorEnv (Sequential)"]
        VectorEnv --> Async["AsyncVectorEnv (Task Pool)"]
    end

    Sync --> Env1["Env 1"]
    Sync --> Env2["Env 2"]
    Sync --> EnvN["Env N"]

    Async --> Worker1["Worker Thread 1"] --> Env1
    Async --> Worker2["Worker Thread 2"] --> Env2
    Async --> WorkerN["Worker Thread N"] --> EnvN

    VectorEnv -->|Batched Obs [N, ...], Rewards [N], Term [N], Trunc [N]| Agent
```

---

## 2. Autoreset Semantics & Final Observation Tracking

In Farama Gymnasium vectorized environments:
1. When sub-environment $i$ encounters `Terminated || Truncated`, `VectorEnv` immediately calls `Reset()` on sub-environment $i$.
2. The observation vector returned in the batch `batched_obs[i]` contains the **new initial observation** of the subsequent episode.
3. The true terminal observation of the completed episode is preserved in `infos["final_observation"][i]`, ensuring replay buffers and GAE advantage calculators have access to the exact terminal transition.

---

## 3. `SyncVectorEnv` vs `AsyncVectorEnv`

| Characteristic | `SyncVectorEnv` | `AsyncVectorEnv` |
| :--- | :--- | :--- |
| **Execution Model** | Single thread sequential loop | Multithreaded `Task.WhenAll` / `DistributedScheduler` |
| **Overhead** | Minimum (zero context switching) | Moderate (thread synchronization) |
| **Best Used For** | Fast math environments (`CartPole`, `MountainCar`) | Heavy physics / computation (`LunarLander`, `BipedalWalker`) |
| **Determinism** | Strict single-threaded order | Guaranteed deterministic by seed |
