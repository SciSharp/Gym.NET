# SOT: Farama Gymnasium API & Architecture Mapping

[Back to Documentation Index](../README.md) · [View PRD](../PRD.md)

---

## 1. Master Source of Truth (SOT) Reference

The canonical Source of Truth for `Gym.NET` is the **Farama Foundation Gymnasium** repository, maintained locally as a nested Git submodule at `refs/Gymnasium` (`https://github.com/Farama-Foundation/Gymnasium`).

---

## 2. API Mapping & Parity Matrix

| Feature / Contract | Farama Gymnasium (`refs/Gymnasium`) | Target Gym.NET C# Implementation | Parity Status |
| :--- | :--- | :--- | :---: |
| **`reset()`** | `tuple[Obs, dict] reset(seed=None, options=None)` | `(NDArray Obs, Dict Info) Reset(int? seed = null, Dict options = null)` | Planned (M1) |
| **`step()`** | `tuple[Obs, float, bool, bool, dict] step(action)` | `StepResult Step(object action)` with `(Obs, Rew, Term, Trunc, Info)` | Planned (M1) |
| **`Box`** | `gymnasium.spaces.Box` | `Gym.Spaces.Box` (Continuous interval with bounded tracking) | Complete (M0) |
| **`Discrete`** | `gymnasium.spaces.Discrete` | `Gym.Spaces.Discrete` (Categorical integer with start offset) | Complete (M0) |
| **`MultiDiscrete`** | `gymnasium.spaces.MultiDiscrete` | `Gym.Spaces.MultiDiscrete` | Planned (M1) |
| **`MultiBinary`** | `gymnasium.spaces.MultiBinary` | `Gym.Spaces.MultiBinary` | Planned (M1) |
| **`Tuple`** | `gymnasium.spaces.Tuple` | `Gym.Spaces.TupleSpace` | Planned (M1) |
| **`Dict`** | `gymnasium.spaces.Dict` | `Gym.Spaces.DictSpace` | Planned (M1) |
| **`Text`** | `gymnasium.spaces.Text` | `Gym.Spaces.TextSpace` | Planned (M1) |
| **`Sequence`** | `gymnasium.spaces.Sequence` | `Gym.Spaces.SequenceSpace` | Planned (M1) |
| **`Wrapper`** | `gymnasium.Wrapper` | `Gym.Wrappers.Wrapper` | Planned (M2) |
| **`TimeLimit`** | `gymnasium.wrappers.TimeLimit` | `Gym.Wrappers.TimeLimit` | Planned (M2) |
| **`RecordStats`** | `gymnasium.wrappers.RecordEpisodeStatistics` | `Gym.Wrappers.RecordEpisodeStatistics` | Planned (M2) |
| **`SyncVectorEnv`**| `gymnasium.vector.SyncVectorEnv` | `Gym.Vector.SyncVectorEnv` | Planned (M2) |
| **`AsyncVectorEnv`**| `gymnasium.vector.AsyncVectorEnv` | `Gym.Vector.AsyncVectorEnv` | Planned (M2) |
| **`CartPole-v1`** | `gymnasium.envs.classic_control.CartPoleEnv` | `Gym.Environments.Envs.Classic.CartPoleEnv` | Upgraded (M0) |
| **`Pendulum-v1`** | `gymnasium.envs.classic_control.PendulumEnv` | `Gym.Environments.Envs.Classic.PendulumEnv` | Planned (M3) |
| **`MountainCar`** | `gymnasium.envs.classic_control.MountainCarEnv` | `Gym.Environments.Envs.Classic.MountainCarEnv` | Planned (M3) |
| **`Acrobot-v1`** | `gymnasium.envs.classic_control.AcrobotEnv` | `Gym.Environments.Envs.Classic.AcrobotEnv` | Planned (M3) |
| **`LunarLander`** | `gymnasium.envs.box2d.LunarLander` | `Gym.Environments.Envs.Aether.LunarLanderEnv` | Upgraded (M0) |
| **`BipedalWalker`**| `gymnasium.envs.box2d.BipedalWalker` | `Gym.Environments.Envs.Aether.BipedalWalkerEnv` | Planned (M3) |

---

## 3. Behavioral Invariants

1. **Deterministic Random Seeding**:
   - `env.Reset(seed: S)` must synchronize all sub-generators such that calling identical action sequences on two separate environment instances produces identical state transitions and rewards.
2. **Terminal vs Truncation Semantics**:
   - `Terminated`: Environment physics or goal conditions naturally ended the episode.
   - `Truncated`: External constraints (step horizons, boundaries) halted execution.
3. **Autoreset Integration**:
   - Replay buffers must receive the true final transition state via `info["final_observation"]` when vectorized environments autoreset on step boundaries.
