# Gym.NET: Product Requirements Document (PRD)

## 1. Executive Summary & Vision

**Gym.NET** is a high-performance, native C# (.NET 8/10) port of OpenAI Gym / Farama Gymnasium designed for reinforcement learning algorithmic development, benchmarking, and quantitative financial microstructure simulations. Operating as part of the SciSharp STACK ecosystem, it provides a standardized, strongly-typed environment suite with `NumSharp` as the underlying multidimensional array/tensor engine.

The goal of this project is to achieve complete architectural, behavioral, and mathematical parity with **Farama Gymnasium** (`refs/Gymnasium`), replacing legacy OpenAI Gym patterns with modern lifecycle contracts, comprehensive observation/action spaces, extensible wrapper hierarchies, and vectorized execution engines.

---

## 2. Core Functional Requirements (FR)

### FR-01: Modernized `Reset` Contract
- **Contract**: `(NDArray Observation, Dict Information) Reset(int? seed = null, Dict options = null)`.
- **Behavior**:
  - Reseeds environment PRNG (`np.random.RandomState(seed)`) when `seed` is provided.
  - Accepts domain-specific options (`Dict options`) for custom initial states.
  - Returns initial observation tensor accompanied by diagnostic information metadata.

### FR-02: Modernized 5-Tuple `Step` Contract
- **Contract**: `StepResult Step(object action)` supporting 5-tuple deconstruction:
  ```csharp
  var (observation, reward, terminated, truncated, info) = env.Step(action);
  ```
- **Behavior**:
  - `terminated` (`bool`): Signals MDP terminal condition (e.g. agent succeeded or failed).
  - `truncated` (`bool`): Signals out-of-bounds termination or time limit expiration (e.g. `TimeLimit` wrapper reached max steps).
  - Deprecates legacy `Done` flag to ensure proper Generalized Advantage Estimation ($\text{GAE}$) bootstrapping in downstream RL engines like `PPO.Core`.

### FR-03: Complete Observation & Action Spaces Suite
- **`Box`**: Continuous multi-dimensional bounded intervals with vectorized Gaussian, exponential, and uniform sampling.
- **`Discrete`**: Categorical integer space `{start, ..., start + n - 1}` with action masking support.
- **`MultiDiscrete`**: Vector of discrete categorical dimensions, each with independent bounds.
- **`MultiBinary`**: Multi-dimensional binary arrays with values in $\{0, 1\}$.
- **`TupleSpace`**: Cartesian product of heterogeneous subspaces.
- **`DictSpace`**: Key-value mapped composite subspaces.
- **`TextSpace`**: Bounded/variable-length character string space.
- **`SequenceSpace`**: Variable-length sequences of subspace elements.
- **`GraphSpace`**: Structured graph spaces containing node, edge, and link features.
- **`OneOfSpace`**: Exclusive union of alternative subspaces.

### FR-04: Standard Wrapper Architecture
- **Base Contracts**: `Wrapper`, `ObservationWrapper`, `ActionWrapper`, `RewardWrapper`.
- **Standard Suite**:
  - `TimeLimit`: Enforces maximum episode step bounds and marks `truncated = true`.
  - `TransformObservation`: Functional transformation applied to observation tensors.
  - `TransformReward`: Functional scaling/clipping applied to rewards.
  - `ClipAction`: Clips continuous actions to action space bounds.
  - `RescaleAction`: Maps continuous actions affine-transformed to custom ranges (e.g. $[-1, 1]$).
  - `RecordEpisodeStatistics`: Collects episode return, length, and execution time in `info["episode"]`.
  - `Autoreset`: Automatically invokes `Reset()` when `terminated or truncated` is encountered in `Step()`.
  - `OrderEnforcing`: Enforces that `Reset()` is invoked prior to `Step()`.

### FR-05: Vectorized Environments
- **`VectorEnv`**: Base vectorized environment abstraction.
- **`SyncVectorEnv`**: Sequential batch execution across $N$ environment instances.
- **`AsyncVectorEnv`**: Parallel batch execution across worker threads/tasks.
- **Autoreset Semantics**: Automatically captures `final_observation` in `info` when sub-environments terminate while continuing seamless rollout batching.

### FR-06: Classical Control & Physics Environments
- **`CartPole-v1`**: Discrete classic control balancing cart and pole via Euler kinematics.
- **`Pendulum-v1`**: Continuous torque control pendulum swing-up.
- **`MountainCar-v0`** & **`MountainCarContinuous-v0`**: Underpowered car mountain ascent.
- **`Acrobot-v1`**: Two-link double pendulum.
- **`LunarLander-v3`**: Discrete and Continuous 2D lunar lander physics simulation (via `Aether.Physics2D`).
- **`BipedalWalker-v3`**: 4-joint bipedal locomotion robot.

### FR-07: Headless & Decoupled Rendering Pipeline
- Environment constructor parameter: `string render_mode = null` (`"human"`, `"rgb_array"`, `null`).
- Rendering decoupled from core physics/math via `IEnvViewer` and `IEnvironmentViewerFactoryDelegate`.
- Implementations:
  - `NullEnvViewer`: High-speed headless simulation.
  - `AvaloniaEnvViewer`: Cross-platform hardware-accelerated GUI (`Gym.Rendering.Avalonia`).
  - `WinFormEnvViewer`: Windows Forms GUI (`Gym.Rendering.WinForm`).

---

## 3. Non-Functional Requirements & Engineering Standards

### 3.1 SOLID Principles
- **Single Responsibility (SRP)**: Segregate observation calculation, physics integration, reward calculation, and viewer rendering into focused classes.
- **Open/Closed (OCP)**: Extend environment capabilities via `Wrapper` composition rather than modifying concrete environment classes.
- **Liskov Substitution (LSP)**: All concrete environments and wrappers must satisfy `IEnv` and generic `IEnv<TObs, TAct>` contracts without surprising side effects.
- **Interface Segregation (ISP)**: Focused interfaces (`IEnv`, `ISpace`, `IWrapper`, `IVectorEnv`, `IEnvViewer`).
- **Dependency Inversion (DIP)**: Environments depend upon abstract viewers via `IEnvironmentViewerFactoryDelegate`, enabling headless testing.

### 3.2 Object Calisthenics Rules
1. **One level of indentation per method**: Extract nested loops/conditionals into private descriptive methods.
2. **Never use the `else` keyword**: Guard clauses, early returns, and polymorphic strategy dispatch.
3. **Wrap domain primitives**: Wrap scalars and raw arrays in strongly typed Value Objects (`Observation`, `Action`, `Reward`, `EpisodeStats`).
4. **First-class collections**: Classes containing collections must encapsulate collection behavior without extraneous properties.
5. **One dot per line**: Demeter compliance across all submodules.
6. **No abbreviations**: Use explicit identifiers (`observation`, `terminated`, `truncated`, `actionSpace`).
7. **Keep entities small**: Target classes $\le 100$ lines, methods $\le 15$ lines.
8. **No bare getters/setters**: Expose domain behavior instead of mutable data structures (*Tell, Don't Ask*).

### 3.3 Test-Driven Development (TDD)
- Comprehensive test coverage with MSTest / FluentAssertions.
- Golden comparison tests validating against deterministic Python Gymnasium trajectories and bounds.
- Tests serve as immutable behavioral contracts.
