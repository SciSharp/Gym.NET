# Gymnasium Parity Audit

| | |
|---|---|
| **Edition** | Second, 2026-11-11, after P8. The first edition (2026-09-25, before the port) is kept as §10 |
| **Gym.NET** | branch `journey1` @ `793c398`: P1–P8 done; MuJoCo (P9) postponed by DR-35 |
| **Gymnasium** | `refs/Gymnasium` @ `8ba8c3a6` (v1.4.0), imported from source by the parity suite |
| **Reference stack** | Windows 11 x64, .NET SDK 10.0.101, Release, net10.0; Python 3.12.12, NumPy 2.4.2, pygame-ce 2.5.8 (SDL 2.32.10), Box2D 2.3.10, OpenCV 4.13.0 |
| **Method** | Both test suites run whole, open gaps included: 3,579 unit results and 2,458 parity results, each data row a result. [`tools/parity/audit/parity_matrix.py`](../../tools/parity/audit/parity_matrix.py) places every result by the env it exercises and the criterion it checks (§9). net8.0 gives the same results (the gates of `793c398`) |
| **Parity levels** | L1 API, L2 behavioral, L3 byte, L4 pixel, as defined in [`GYMNASIUM_PORT_PLAN.md`](GYMNASIUM_PORT_PLAN.md) §1.2 |

---

## 1. Verdict

**All 17 registered ids are at byte (L3) and pixel (L4) parity with Gymnasium v1.4.0, and no gated test fails.** For
the same seeds and actions, each env returns Gymnasium's observations, float64 rewards, flags, infos and generator
states bit for bit, and draws Gymnasium's frames pixel for pixel. That holds in every layer the suites cover: the env's
own suite, `make`'s stack, registration, Gymnasium's own `check_env` and `check_environments_match`, a golden replayed
without Python, sync and async vector envs, `human` mode and `rgb_array_list`.

| Suite | Results | Passed | Open (`[OpenBugs]`) | Skipped | Failed |
|---|---:|---:|---:|---:|---:|
| Unit (`tests/Gym.Tests`) | 3,579 | 3,549 | 11 | 19 | 0 |
| Parity (`tests/Gym.Tests.Parity`, live Gymnasium) | 2,458 | 2,451 | 7 | 0 | 0 |

- **The 18 open results change no observation, reward or frame** (§7.1). Eight are NumSharp's N-1: where Gymnasium
  fails a bare `assert`, Gym.NET raises the same `AssertionError` with NumSharp's message `"AssertionError"` instead of
  Python's empty one. That covers a step before a reset in the three Box2D envs, Taxi's `decode` beyond the grid,
  Blackjack's refused actions, and `render()` without a render mode or spec (twice), plus N-1's own proof. Eight more
  prove NumSharp's other gaps (N-2 to N-4). One is Intel IPP's float32 bilinear resize in `cv2` (DR-46), which no env
  uses. One is MuJoCo's Ant (P9, postponed).
- **The deviations are deliberate and pinned** by passing cells (§7.2): six ids print an int reward threshold as
  float64; MountainCarContinuous-v0's spec carries `continuous=True` (DR-32); Blackjack's observation is a .NET tuple
  read by name (DR-13).
- **The skipped 19** are the data rows where Gymnasium's own tests call `pytest.skip` (spaces with a dynamic shape,
  wrappers that record no constructor arguments).
- **Against the first edition** (§10): then, nothing algorithmic was byte-identical beyond seeding and space samples.
  CartPole diverged at step 1, LunarLander was another simulation, 0 of 12 frames matched, 216 of 258 names were missing,
  and none of Gymnasium's 490 test functions was ported. Now every id is exact. The inventory misses only MuJoCo's names
  and `AtariPreprocessing`, besides 4 names merged or kept private by design and names that do not apply to .NET (JAX,
  pickling, process workers). 440 test methods in 83 classes mirror Gymnasium's own test modules (3,258 results with
  their data rows).

## 2. The criteria

| Criterion | What a cell compares | Where |
|---|---|---|
| Reset | `reset(seed, options)`: the observation, the info and the generator state; options and their errors; the world a reset builds (Box2D terrains and tracks) | The env's suite; every episode cell also starts with a compared seeded reset |
| Dynamics | Trajectories under seeded sampled actions and Gymnasium's own policies and heuristics, call by call: the observation's bytes, the reward's float64 bits, `terminated`, `truncated`, the info, the generator's state; the env's own functions (transition tables, heuristics, hand functions, map generation) | The env's suite |
| Actions & errors | Every action form (dtypes, bounds, NaN, shapes, Python's dict lookups), refused actions (the exception's type mapped, its message verbatim), steps before a reset or after termination | The env's suite |
| Injected states | A state set on both sides, then stepped or rendered; malformed states and their errors | The env's suite |
| Spaces, metadata, repr | Both spaces (`repr`, dtype, bounds, seeded samples), `metadata`, `repr`/`str`, `desc`, the stub classes that refuse construction | The env's suite |
| `rgb_array` frames | Frames pixel for pixel along episodes and at injected and extreme states; `render()` before a reset, and its errors | The env's suite |
| `ansi` text | The text render, character for character | The env's suite |
| `make` stack | `gymnasium.make(id)` against `Gymnasium.Make(id)`: `TimeLimit(OrderEnforcing(PassiveEnvChecker(env)))` through the step limit | The env's suite, `EnvCoverageLiveParityTests`, `CoreLiveParityTests` |
| Registration | `str(make(id))` and the printed spec of the stack and of the unwrapped env | `RegistrationLiveParityTests` |
| `check_env` / `env_match` | Gymnasium's own `check_env` and `check_environments_match`, run on Gym.NET's env through the reverse bridge | `GymnasiumToolsLiveParityTests` |
| Golden | A trajectory the live cell verified, recorded with its frames and replayed without Python (the Linux CI job) | `GoldenLiveTests`, `GoldenReplayTests` |
| Sync / async vector | `make_vec(id, 3, "sync")` in lockstep for 220 steps under `NEXT_STEP` and `SAME_STEP`; Gymnasium's own `AsyncVectorEnv` against Gym.NET's for 120 steps, with and without shared memory | `EnvCoverageLiveParityTests`, `VectorEnvLiveParityTests`, `AsyncVectorEnvLiveParityTests` |
| `human` | `render_mode="human"`: the calls, the frames shown, and the frame on display against pygame's display | `EnvCoverageLiveParityTests` |
| `rgb_array_list` | `make`'s `RenderCollection`: the frames per render | `EnvCoverageLiveParityTests` |

**Notation:**

| Mark | Meaning |
|---|---|
| `✓ n` | n results passed; each data row counts |
| `open n` | n `[OpenBugs]` results failing as expected (§7.1) |
| `in every episode cell` | The suite has no reset cell of its own, and every episode cell compares its seeded reset |
| `n/a` | A mode or state the env does not have |
| `—` | No dedicated cell |

## 3. Per env family: the env itself

| Env | Ids | Reset | Dynamics | Actions & errors | Injected states | Spaces, metadata, repr | `rgb_array` frames | `ansi` text |
|---|---|---|---|---|---|---|---|---|
| CartPole (classic_control) | CartPole-v0, CartPole-v1 | ✓ 5 | ✓ 13 | ✓ 3 | ✓ 1 | ✓ 3 | ✓ 7 | n/a (no `ansi` mode) |
| MountainCar (classic_control) | MountainCar-v0, MountainCarContinuous-v0 | ✓ 7 | ✓ 12 | ✓ 4 | ✓ 4 | ✓ 6 | ✓ 6 | n/a (no `ansi` mode) |
| Pendulum (classic_control) | Pendulum-v1 | ✓ 4 | ✓ 4 | ✓ 2 | ✓ 2 | ✓ 2 | ✓ 4 | n/a (no `ansi` mode) |
| Acrobot (classic_control) | Acrobot-v1 | ✓ 5 | ✓ 7 | ✓ 2 | ✓ 2 | ✓ 2 | ✓ 3 | n/a (no `ansi` mode) |
| LunarLander (box2d) | LunarLander-v3, LunarLanderContinuous-v3 | ✓ 5 | ✓ 17 | ✓ 2 · open 1 | n/a (a Box2D world) | ✓ 5 | ✓ 5 | n/a (no `ansi` mode) |
| BipedalWalker (box2d) | BipedalWalker-v3, BipedalWalkerHardcore-v3 | ✓ 13 | ✓ 20 | ✓ 1 · open 1 | n/a (a Box2D world) | ✓ 4 | ✓ 5 | n/a (no `ansi` mode) |
| CarRacing (box2d) | CarRacing-v3 | ✓ 11 | ✓ 7 | ✓ 1 · open 1 | n/a (a Box2D world) | ✓ 3 | ✓ 6 | n/a (no `ansi` mode) |
| FrozenLake (toy_text) | FrozenLake-v1, FrozenLake8x8-v1 | in every episode cell | ✓ 3 | ✓ 1 | — | ✓ 3 | ✓ 1 | ✓ 1 |
| CliffWalking (toy_text) | CliffWalking-v1, CliffWalkingSlippery-v1 | in every episode cell | ✓ 3 | ✓ 1 | — | ✓ 2 | ✓ 2 | ✓ 1 |
| Taxi (toy_text) | Taxi-v4 | in every episode cell | ✓ 5 | ✓ 2 · open 1 | — | ✓ 3 | ✓ 2 | ✓ 1 |
| Blackjack (toy_text) | Blackjack-v1 | in every episode cell | ✓ 5 | ✓ 1 · open 1 | — | ✓ 2 | ✓ 2 | n/a (no `ansi` mode) |

The counts are results, so they measure how a suite is cut into cells, not how much it checks. One CarRacing dynamics
cell drives two full laps of 1,700-odd steps; one Taxi text cell covers all 500 states. Toy_text has no injected-state
cells of its own, but CliffWalking's, Taxi's and Blackjack's text, frame and reward cells set their states (§6).

## 4. Per registered id: integration layers

| Id | `make` stack | Registration | `check_env` / `env_match` | Golden | Sync vector | Async vector | `human` | `rgb_array_list` |
|---|---|---|---|---|---|---|---|---|
| CartPole-v0 | ✓ 2 | ✓ 1 | ✓ 2 | ✓ 1 case | ✓ 3 | ✓ 2 | ✓ 1 | ✓ 1 |
| CartPole-v1 | ✓ 2 | ✓ 1 | ✓ 3 | ✓ 2 cases | ✓ 11 | ✓ 12 | ✓ 1 | ✓ 1 |
| MountainCar-v0 | ✓ 1 | ✓ 1 | ✓ 2 | ✓ 1 case | ✓ 2 | ✓ 2 | ✓ 1 | ✓ 1 |
| MountainCarContinuous-v0 | ✓ 1 | ✓ 2 | ✓ 2 | ✓ 1 case | ✓ 2 | ✓ 2 | ✓ 1 | ✓ 1 |
| Pendulum-v1 | ✓ 1 | ✓ 1 | ✓ 2 | ✓ 1 case | ✓ 2 | ✓ 2 | ✓ 1 | ✓ 1 |
| Acrobot-v1 | ✓ 1 | ✓ 1 | ✓ 2 | ✓ 1 case | ✓ 2 | ✓ 2 | ✓ 1 | ✓ 1 |
| LunarLander-v3 | ✓ 1 | ✓ 1 | ✓ 2 | ✓ 1 case | ✓ 2 | ✓ 2 | ✓ 1 | ✓ 1 |
| LunarLanderContinuous-v3 | ✓ 1 | ✓ 1 | ✓ 2 | ✓ 1 case | ✓ 2 | ✓ 2 | ✓ 1 | ✓ 1 |
| BipedalWalker-v3 | ✓ 1 | ✓ 1 | ✓ 2 | ✓ 1 case | ✓ 2 | ✓ 2 | ✓ 1 | ✓ 1 |
| BipedalWalkerHardcore-v3 | ✓ 1 | ✓ 1 | ✓ 2 | ✓ 1 case | ✓ 2 | ✓ 2 | ✓ 1 | ✓ 1 |
| CarRacing-v3 | ✓ 1 | ✓ 1 | ✓ 2 | ✓ 2 cases | ✓ 2 | ✓ 2 | ✓ 1 | ✓ 1 |
| FrozenLake-v1 | ✓ 1 | ✓ 1 | ✓ 2 | ✓ 1 case | ✓ 5 | ✓ 4 | ✓ 1 | ✓ 1 |
| FrozenLake8x8-v1 | ✓ 1 | ✓ 1 | ✓ 2 | ✓ 1 case | ✓ 2 | ✓ 2 | ✓ 1 | ✓ 1 |
| CliffWalking-v1 | ✓ 1 | ✓ 1 | ✓ 2 | ✓ 1 case | ✓ 2 | ✓ 2 | ✓ 1 | ✓ 1 |
| CliffWalkingSlippery-v1 | ✓ 1 | ✓ 1 | ✓ 2 | ✓ 1 case | ✓ 2 | ✓ 2 | ✓ 1 | ✓ 1 |
| Taxi-v4 | ✓ 1 | ✓ 1 | ✓ 2 | ✓ 1 case | ✓ 3 | ✓ 2 | ✓ 1 | ✓ 1 |
| Blackjack-v1 | ✓ 1 | ✓ 1 | ✓ 2 | ✓ 1 case | ✓ 5 | ✓ 4 | ✓ 1 | ✓ 1 |

- **Goldens:** each case is three results: the live cell that verifies and records it, its replay without Python, and
  its codec round trip.
- **`make` stack:** a cell that loops over both of a family's ids counts for each (FrozenLake's, CliffWalking's).
- **`check_env` / `env_match`:** "2" is one `check_env` and one `check_environments_match` cell, the latter over two
  seeds in both orders, frames included. CartPole-v1 has a third: `make(spec)` over the bridged env.

## 5. Per env family: wider coverage

| Env | Wrapper cells naming it | Gymnasium's own tests (C#) | Gym.NET's unit tests |
|---|---|---|---|
| CartPole | ✓ 26 | ✓ 12 | ✓ 22 · open 1 |
| MountainCar | ✓ 7 | ✓ 17 | ✓ 3 |
| Pendulum | ✓ 15 | ✓ 6 | ✓ 1 |
| Acrobot | ✓ 1 | ✓ 8 | — |
| LunarLander | — | ✓ 9 | ✓ 14 |
| BipedalWalker | — | ✓ 7 | — |
| CarRacing | — | ✓ 7 | ✓ 5 |
| FrozenLake | ✓ 12 | ✓ 8 | — |
| CliffWalking | — | ✓ 4 | — |
| Taxi | ✓ 2 | ✓ 1 | ✓ 1 |
| Blackjack | ✓ 4 | ✓ 1 | ✓ 1 |

Only cells that name an env count here: 173 more wrapper cells and the rest of Gymnasium's tests (3,177 results) run
envs or spaces without naming them, and `play`'s 38 cells run CartPole, MountainCar and Gymnasium's own test env (§8).
The unit suite's viewer tests (`*AvaloniaEnv`, `*WinFormEnv`) open windows and are not part of this run; the parity
suite's `human` cells cover human mode headless.

## 6. Each env: levels and notes

| Env | L1 API | L2 behavioral | L3 byte | L4 pixel | Open | Deviation |
|---|---|---|---|---|---|---|
| CartPole-v0, CartPole-v1 | ✓ | ✓ | ✓ | ✓ | `render()` without mode or spec (N-1) | — |
| MountainCar-v0, MountainCarContinuous-v0 | ✓ | ✓ | ✓ | ✓ | — | the continuous id's spec kwargs (DR-32) |
| Pendulum-v1 | ✓ | ✓ | ✓ | ✓ | — | — |
| Acrobot-v1 | ✓ | ✓ | ✓ | ✓ | — | — |
| LunarLander-v3, LunarLanderContinuous-v3 | ✓ | ✓ | ✓ | ✓ | step before reset (N-1) | threshold `200` prints `200.0` |
| BipedalWalker-v3, BipedalWalkerHardcore-v3 | ✓ | ✓ | ✓ | ✓ | step before reset (N-1) | threshold `300` prints `300.0` |
| CarRacing-v3 | ✓ | ✓ | ✓ | ✓ | step before reset (N-1) | threshold `900` prints `900.0` |
| FrozenLake-v1, FrozenLake8x8-v1 | ✓ | ✓ | ✓ | ✓ | — | — |
| CliffWalking-v1, CliffWalkingSlippery-v1 | ✓ | ✓ | ✓ | ✓ | — | — |
| Taxi-v4 | ✓ | ✓ | ✓ | ✓ | `decode` beyond the grid (N-1) | threshold `8` prints `8.0` |
| Blackjack-v1 | ✓ | ✓ | ✓ | ✓ | refused actions (N-1) | its observation's .NET type (DR-13) |

L1 is the inventory and the cells for spaces, metadata, `repr` and error types. L2 is Gymnasium's own tests that run the
env. L3 is §3's reset, dynamics, actions and states with §4's `make`, vector and golden layers. L4 is the frames, `ansi`,
`human` and `rgb_array_list`. An "open" entry is an error message, not a value (§7.1).

**CartPole-v0, CartPole-v1** (`CartPoleEnv`)
- **Episodes:** the balancing policy and random actions through termination and past it.
- **Variations:** Sutton and Barto's reward, the semi-implicit Euler integrator, a changed `tau`.
- **Resets and refusals:** 17 reset-option cases, 13 invalid actions, the post-termination warning.
- **Frames:** along episodes and at 20 extreme states (clipped, Sint16-wrapped, NaN).
- **CartPole-v0:** the same class with its 200-step limit and 195 threshold, checked through `make`.
- **`CartPoleVectorEnv`** (the vector entry point) matches Gymnasium's over 600 steps of 4 carts, its state compared
  after every step.
- **The harness:** the suite's float64 transcription of `cartpole.py` is its positive control.

**MountainCar-v0, MountainCarContinuous-v0** (`MountainCarEnv(continuous)`, DR-32)
- **Episodes:** both modes, the continuous one in NumPy's float32 arithmetic (NEP 50), and the momentum policy to the
  flag; `goal_velocity`.
- **Actions:** continuous forces of every dtype with NumPy's errors; unsigned discrete actions wrap as NumPy's
  `action - 1`.
- **States:** 12 injected ones, and malformed ones with Python's errors.
- **Frames:** the hill is pygame's own `aaline` (DR-37).
- **Deviation:** the continuous id's `EnvSpec.kwargs` carry `continuous=True`, which the constructor takes. The class
  prints as `Continuous_MountainCarEnv`, so its `repr` is Gymnasium's.

**Pendulum-v1**
- **Actions:** torques of every dtype and value, and NumPy's errors for 0-d and empty torques.
- **Resets:** 14 reset-option cases (`x_init`/`y_init` and the generator's range errors).
- **States:** 11 injected ones; a changed gravity, `g=9.81`.
- **Frames:** the arrow at every size, and the render errors of a missing `state` or `last_u`.

**Acrobot-v1**
- **Episodes:** the pumping policy through termination.
- **Actions:** list-indexed actions (negative indices, NumPy integer scalars), and 13 refused actions with Python's texts.
- **States:** 8 injected ones, and malformed ones (NumPy's `IndexError` in `_dsdt`, the broadcast `ValueError` in `rk4`).
- **Variants:** `book_or_nips='nips'` and `torque_noise_max`.

**LunarLander-v3, LunarLanderContinuous-v3** (`LunarLanderEnv(continuous)`)
- **Resets:** both variants, 16 draws each (terrain, push, legs).
- **Episodes:** Gymnasium's heuristic through the landing (3 seeds each variant), sampled episodes, wind and turbulence
  at four power pairs, two gravities.
- **Actions:** every continuous action form, 6 refused discrete actions, the constructor's checks.
- **Heuristic function:** matches Gymnasium's on 2,000 observations.
- **Engine:** Box2D 2.3.0 translated to C#, bit for bit with pybox2d 2.3.10 (§8).

**BipedalWalker-v3, BipedalWalkerHardcore-v3** (`BipedalWalkerEnv(hardcore)`)
- **Resets:** the world each builds, value for value (7 seeds: stumps, stairs both ways, pits).
- **Episodes:** Gymnasium's gait through 1,600 steps or the fall, the lidar's closest hits, and
  `check_environments_match`'s loop twice on the same envs.
- **Finding, Gymnasium's own behaviour reproduced:** an episode depends on the env's past, because Box2D's broad phase
  reuses freed tree nodes. The cells compare envs with equal pasts.

**CarRacing-v3** (`CarRacingEnv`). Every observation is a rendered 96×96 frame, so every compared step compares pixels.
- **Resets:** the tracks they build (5 seeds, with and without randomization).
- **Episodes:** sampled episodes of both action modes, two full laps of a track-following driver (1,709 and 1,767
  steps), skid marks, leaving the playfield.
- **Actions:** 10 action forms.
- **Score:** its glyphs (embedded, DR-48) against pygame on 3,663 strings, and its box for 32 kinds of value: NaN, the
  infinities, and 309-digit boxes.

**FrozenLake-v1, FrozenLake8x8-v1**
- **Dynamics:** the transition table of 9 variants (both maps, `success_rate`, `reward_schedule`, custom maps), and
  episodes past termination.
- **Actions:** looked up as Python's dicts look them up (`True`, floats equal to a key, NumPy scalars), with their
  `KeyError`/`TypeError`.
- **Rest:** the `ansi` text, `generate_random_map` (8 cases), and `desc` as NumPy's `S1` array.

**CliffWalking-v1, CliffWalkingSlippery-v1**
- **Dynamics:** the dry and slippery tables, a walk to the goal, the cliff's -100.
- **Text and frames:** with the agent at every one of the 48 cells; render before a reset.

**Taxi-v4**
- **Dynamics:** the tables dry, rainy and at `rainy_probability=0.6`, and the fickle passenger at every cell.
- **Functions:** `encode`, `decode` and `action_mask` of all 500 states.
- **Text and frames:** the text at all 500 states, and every taxi sprite.

**Blackjack-v1**
- **Games:** 150 sampled and 150 strategy games under each of the 4 rule sets.
- **Rules and draws:** every reward rule on 12 injected hands; `draw_card`/`draw_hand` with the generator's state; the
  hand functions over every hand of up to 3 cards.
- **Observation type:** Gym.NET's `object?[]` tuple, read by name through `BlackjackObservation` (DR-13); the values are
  Gymnasium's.

## 7. Open gaps and deviations

### 7.1 Open (`[OpenBugs]`: the expectation is Gymnasium's, Gym.NET does not meet it yet)

| Test | Suite | Waits on | What differs |
|---|---|---|---|
| `LunarLander`, `BipedalWalker`, `CarRacing`: `GymNet_StepBeforeReset_RaisesLikeGymnasium` | Parity | NumSharp N-1 | The bare `assert self.<body> is not None`: same type, NumSharp's message |
| `TaxiLiveParityTests.GymNet_DecodeBeyondTheGrid_RaisesTheBareAssertion` | Parity | N-1 | `decode` beyond the grid: the bare assertion's message |
| `BlackjackLiveParityTests.GymNet_RefusedActions_RaiseTheBareAssertion` | Parity | N-1 | Refused actions: the bare assertion's message |
| `CoreLiveParityTests.RenderWithoutRenderMode_WithoutSpec_RaisesGymnasiumsBareAssertion` | Parity | N-1 | `render()` without a render mode or spec (Gymnasium's `assert self.spec is not None`) |
| `CartPoleTests.Render_WithoutRenderModeOrSpec_HasPythonsEmptyMessage` | Unit | N-1 | The same, without Python |
| `NumSharpGapTests.N1_…`, `N2_…`, `N3_…` (6), `N4_…` | Unit | N-1 to N-4 | Each proves a NumSharp gap: the empty `AssertionError` message, the `'c'` dtype's string splitting, the missing Python exceptions and inner exceptions, `linspace` over arrays |
| `Cv2ResizeLiveParityTests.Resize_Float32_Linear_MatchesDefaultOpenCv` | Parity | DR-46 | `cv2`'s default float32 `INTER_LINEAR` goes to Intel IPP, whose unpublished bilinear differs by a few units in the last place; Gym.NET gives OpenCV's own kernel, exact with IPP off. No env resizes float32 frames |
| `TestHumanRendering.TestNumEnvsScreenSize_Ant` | Unit | P9 | MuJoCo's Ant, postponed (DR-35) |

### 7.2 Documented deviations (each pinned by a passing cell)

| Deviation | Ids or scope | Pinned by | Decision |
|---|---|---|---|
| An int reward threshold prints as float64 (`200` → `200.0`), since Gym.NET's thresholds are float64 like its rewards | Taxi-v4 (8), LunarLander-v3 and LunarLanderContinuous-v3 (200), BipedalWalker-v3 and BipedalWalkerHardcore-v3 (300), CarRacing-v3 (900) | `RegistrationLiveParityTests` | Interpretations confirmed in round 15 |
| MountainCarContinuous-v0's `EnvSpec.kwargs` carry `continuous=True`, and the merged constructor takes it | MountainCarContinuous-v0 | `Spec_MountainCarContinuousKwargs_CarryContinuous_AsDr32Accepts` | DR-32 |
| Blackjack's observation is an `object?[]` tuple, read by name through `BlackjackObservation` | Blackjack-v1 | The suite's own cells, which compare the values | DR-13 |
| `AsyncVectorEnv(shared_memory=False)`: a full reset after an episode end clears the pending autoreset, so sync and async agree | Every id under async vectorization | `AsyncVectorEnvLiveParityTests` | DR-42 |
| A vector coordinate beyond float32 in a pybox2d keyword constructor raises the conversion's own `TypeError`, without pybox2d's prefix; no Gymnasium env passes such a value | The Box2D facade | `Box2DEngineLiveParityTests` | DR-26 |
| LunarLander's and BipedalWalker's `ContactDetector` classes are private | LunarLander, BipedalWalker | The inventory (by design) | DR-28 |

## 8. What the envs rest on

Results that name no env, by area:

| Area | Results |
|---|---|
| Spaces (parity: `Box`, `Discrete`, the space catalog, flatten, vector utils) | ✓ 1,501 |
| Spaces (Gymnasium's tests: `tests/spaces`) | ✓ 900 |
| Vector (Gymnasium's tests: `tests/vector`) | ✓ 1,736 · skip 19 |
| Wrappers (Gymnasium's tests: `tests/wrappers`) | ✓ 397 |
| Wrappers (parity, no env named) | ✓ 173 |
| Gymnasium's other tests (envs, core, utils, functional, inventory) | ✓ 125 |
| `play` (parity) | ✓ 38 |
| Seeding (PCG64 against NumPy) | ✓ 23 |
| Vector API (parity, no env named) | ✓ 19 |
| Rasterizer (SDL_gfx and pygame's `draw`, parity) | ✓ 27 |
| Box2D engine (scenes, ray casts and error texts against pybox2d) | ✓ 13 |
| OpenCV resize | ✓ 16 · open 1 |
| Core API (wrappers, attributes, `EnvSpec`, step API) | ✓ 10 · open 1 |
| Registry (`make`'s errors, `spec`, `pprint_registry`) | ✓ 5 |
| Reverse bridge (the space mirror, negative controls) | ✓ 5 |
| `PyMath` (Python's arithmetic) and benchmarks | ✓ 11 |
| Harness self-tests, session, golden harness, CartPole's reference transcription | ✓ 27 |
| Gym.NET's own unit tests (spaces, utils, vector, rendering, Box2D, envs in general, legacy) | ✓ 260 |
| NumSharp gaps (proofs) | ✓ 4 · open 9 |
| MuJoCo (P9, postponed) | open 1 |

## 9. Method and reproduction

```bash
OUT=<scratch directory>
export AVALONIA_TELEMETRY_OPTOUT=1
dotnet build Gym.NET.sln -c Release
dotnet test tests/Gym.Tests/Gym.Tests.csproj -c Release --no-build -f net10.0-windows \
    --filter "FullyQualifiedName!~AvaloniaEnv&FullyQualifiedName!~WinFormEnv" \
    --logger "trx;LogFileName=unit.trx" --results-directory "$OUT"
dotnet test tests/Gym.Tests.Parity/Gym.Tests.Parity.csproj -c Release --no-build -f net10.0 \
    --logger "trx;LogFileName=parity.trx" --results-directory "$OUT"
python tools/parity/audit/parity_matrix.py "$OUT/unit.trx" "$OUT/parity.trx" --json "$OUT/audit.json"
```

Neither run filters out `[OpenBugs]`, so they exit with a failure: the open results are part of the audit. The script
prints the tables of §3–§5, the areas of §8, every open test with its defect ids (read from the `[OpenBugs(...)]`
attributes, since TRX keeps only the category), any gated failure, and every result its rules cannot place
(none in this run).

How it places a result:
- **The env:** an id in the data row (`"CarRacing-v3"`); else a family named in the class, method or data row
  (`CarRacingLiveParityTests`, `CartPole_Lockstep`). MuJoCo's names are removed first, since `InvertedPendulum` contains
  `Pendulum`.
- **The criterion:** the cross-cutting classes give theirs (`EnvCoverageLiveParityTests`' four methods, registration,
  bridge, goldens, vector, wrappers). In an env's own suite, the method name decides, through an ordered list of
  patterns (`Frame|Render` → frames, `Space|Metadata|Repr|Desc` → API, `Inject|Malformed` → states, actions, resets,
  then dynamics).
- **Cells naming no id:** their ids come from a table read off the test bodies (FrozenLake's `make` cell runs both maps),
  else from a variant the method names (`MountainCarContinuous`), else from the family's primary id.

Limits, all visible in the tables:
- A result counts once per family it names, so a cell over two envs counts for both.
- Cells that run an env without naming it count only under §8.
- Results measure how a suite is cut into cells, not how much each checks (§3).

---

## 10. History: the first edition (2026-09-25)

The first edition, as it was written, before the port began. Its section numbers (§1 to §7) and its "today" are its own; its probes still live in `tools/parity/audit/` (`gymnasium_side.py`, `gymnet_side.cs`, `compare.py`).

| | |
|---|---|
| **Date** | 2026-09-25 |
| **Gym.NET** | branch `feat/numsharp-master-migration` @ `b51689d` (NumSharp migration steps 1–3, then Gymnasium's PCG64 seeding; ownership step 4 not started). The first measurement, at `2a1b827` before the PCG64 upgrade, is kept in §6 |
| **Gymnasium** | `refs/Gymnasium` @ `9e04324` (1.4.0 dev), run from source |
| **Reference stack** | Windows 11 x64; Python 3.12.12, NumPy 2.4.2, pygame-ce 2.5.8 (SDL 2.32.10), Box2D 2.3.10; .NET SDK 10.0.101 |
| **Parity levels** | as defined in [`GYMNASIUM_PORT_PLAN.md`](GYMNASIUM_PORT_PLAN.md) §1.2: L1 API, L2 behavioral, L3 byte, L4 pixel |
| **Method** | the same cases run live in Gymnasium and in Gym.NET and compared bit for bit (§7 reproduces every number) |

---

### 1. Verdict

**Gym.NET's seeded draws are byte-identical to Gymnasium's.** Since `b51689d`, envs and spaces draw from Gymnasium's generator, `Generator(PCG64(SeedSequence(seed)))`. For the same seed, Gym.NET's `Discrete` and `Box` samples and CartPole's reset state equal Gymnasium's bit for bit:

- 4 of 4 seeds in every case of this audit;
- 783 of 783 values over 9 seeds up to 2^40 in the committed golden, which `GymnasiumRngParityTests` enforces on every test run.

These are the first L3 results against Gymnasium itself. Before the upgrade, every case was 0 of 4 (§6).

**Everything else is not byte-identical yet.**

| Area | Status | What closes it |
|---|---|---|
| CartPole observations | Gym.NET hands out its float64 state; Gymnasium's observation is that state as float32 (B-13) | a cast (P4) |
| CartPole dynamics | 0 of 2,000 steps identical, from step 1 on, because of float32 constants (B-11) | a float64 transcription, already bit-identical for all 2,000 steps (P4) |
| CartPole frames | 0 of 12 identical | the rasterizer (F5, P7) |
| LunarLander | a different simulation (Aether, v2 geometry) | a Box2D 2.3.x port (F6, P8) |
| The API contract | 216 of 258 Gymnasium names missing | P1 to P6 |

### 2. Scorecard

| Level | Area | Target (port plan §1.2) | Today | Projection: P2 + P4 done as planned |
|---|---|---|---|---|
| L1 API | Public surface (258 in-scope names) | every name | 4 match, 15 diverge, 2 broken, 216 missing (21 N/A) | unchanged; the projection is numerics only |
| L1 API | `Env` contract | `reset(seed, options) -> (obs, info)`, `step -> 5-tuple` | `Reset() -> obs`, `Step -> (obs, reward, done, info)` | P1 |
| L2 behavioral | Gymnasium's test suite | ported and green | 0 of 490 test functions ported (102 files) | P1–P6 |
| L3 byte | Seeding | `Generator(PCG64(SeedSequence(seed)))` | 4 / 4 seeds identical | 4 / 4 |
| L3 byte | `Discrete` samples (plain, masked, with start) | bit-identical | 4 / 4 seeds, in each case | 4 / 4 |
| L3 byte | `Box` samples (bounded, unbounded, low-bounded, integer) | bit-identical | 4 / 4 seeds, in each case | 4 / 4 |
| L3 byte | CartPole reset | bit-identical | state 4 / 4; observation 0 / 4 (float64, not float32) | 4 / 4 |
| L3 byte | CartPole dynamics, same initial state | bit-identical | 0 / 2,000 steps (diverges at step 1, every seed) | 2,000 / 2,000 |
| L3 byte | LunarLander | bit-identical | 0 / 4 reset observations; unrelated episodes | out of reach without Box2D (P8) |
| L4 pixel | CartPole frames | byte-identical | 0 / 12 frames (1.14–1.24% of pixels differ, max delta 255) | out of reach without the rasterizer (P7) |

### 3. Measurements

Seeds 0, 1, 42 and 1000 throughout. "Today" is Gym.NET's own code. "Projection" is Gymnasium's algorithm transcribed in C# on NumSharp (§7.2); it measures how far the planned ports would carry and is not shipped code.

#### 3.1 Seeding

| Case | Today | Projection |
|---|---:|---:|
| First 8 `random()` doubles of the seeded generator | 32 / 32 values, 4 / 4 seeds | 32 / 32, 4 / 4 |

- **How both sides seed:** Gymnasium goes through `seeding.np_random(seed)`, and Gym.NET through its own `Gym.Utils.Seeding.NpRandom`. Both use the same validation, the same message ("Seed must be greater or equal to zero, actual value: …", raised as `Gym.Exceptions.Error`, Gym.NET's counterpart of `gymnasium.error.Error`), and the same generator.
- **Seed range:** `Seeding.NpRandom` takes any non-negative integer as a `BigInteger?`, because Gymnasium's seeds are Python ints of any size, and the seed it reports for an unseeded generator is 128-bit entropy. The envs' and spaces' `Seed(long)` take any non-negative `long`; the legacy ceiling of 2^32 − 1 is gone.
- **Upstream NumSharp item:** NumSharp holds that unseeded entropy as four 32-bit words, where NumPy holds the integer they encode. It is reported as that integer, and passing it back rebuilds the stream, except in the 1-in-2^32 case where the top word is zero. *Fixed in NumSharp 0.80.0 (2026-11-08): the unseeded sequence holds the integer and builds its pool from it, so the exception and its workaround are gone (completion plan R0.1).*
- **Still missing (P1/P2):** lazy seeding, the `np_random`/`np_random_seed` properties, and `Space.seed` returning its seed.

#### 3.2 Spaces

| Case | Values | Today: seeds identical | Projection: seeds identical |
|---|---:|---:|---:|
| `Discrete(4).sample()` × 10 | 40 | 4 / 4 | 4 / 4 |
| `Discrete(4).sample(mask=[1,0,1,1])` × 5 | 20 | 4 / 4 | 4 / 4 |
| `Discrete(3, start=5).sample()` × 5 | 20 | 4 / 4 | 4 / 4 |
| `Box(-1, 1, (3,), float32).sample()` | 12 | 4 / 4 | 4 / 4 |
| `Box(-inf, inf, (3,), float64).sample()` | 12 | 4 / 4 | 4 / 4 |
| `Box(0, inf, (3,), float64).sample()` | 12 | 4 / 4 | 4 / 4 |
| `Box(0, 5, (8,), int32).sample()` | 32 | 4 / 4 | 4 / 4 |

The committed golden also covers:
- 9 seeds, including 2^32 and 2^40;
- `Discrete` with n = 3×10⁹ and 2^40;
- all four bound categories in one `Box`, an upper-bounded box, and int8 and uint8 boxes.

It is identical on all 783 values. `Box.sample` follows `box.py:404` line for line. That port also removed B-08 (unbounded elements drew `normal(0.5, 1)`) and B-09 (integer boxes never reached `high`), and it clips integer dtypes as Gymnasium does.

What still differs, outside the tested values:
- `Discrete`'s shape is `(n,)` and Gymnasium's `()` (B-03);
- a sample is a 0-d `NDArray` where Gymnasium returns a NumPy scalar;
- Gymnasium validates bounds, dtypes and masks and raises its own errors (B-10);
- probability masks are missing;
- `Discrete` always samples in int64 (Gymnasium samples in the space's dtype, int64 by default);
- `Box.contains` has no dtype check.

All of these are P2.

#### 3.3 CartPole-v1

| Case | Today | Projection |
|---|---|---|
| `reset(seed)` internal state | 4 / 4 seeds | 4 / 4 |
| `reset(seed)` observation | 0 / 4: Gymnasium's state values, but handed out as float64 where Gymnasium hands out float32 (B-13) | 4 / 4, float32 |
| 500 open-loop steps from Gymnasium's initial state, per seed | 0 / 500 steps identical; first divergence at step 1 | 500 / 500, including reward and `terminated` |
| Observation-space bounds | differ: `±float.MaxValue` where Gymnasium has `±inf` (B-12) | not projected |

- **Why dynamics diverge:** the constants are float32 (`tau = 0.02f` is 0.019999999552965164 as a double, B-11), so the state differs in its last bits from step 1 on every seed.
- **How far it drifts:** with this open-loop action sequence (0, 0, 1, 1, ...) the drift stays below 4e-6 over 500 steps, and termination happens on the same step as in Gymnasium. The port plan's A.7, replaying Gymnasium's own random actions, saw the drift grow to 27 and termination move.
- **The fix is proven:** the float64 transcription of `cartpole.py:164` with `Math.Cos`/`Math.Sin` is bit-identical for all 2,000 steps.
- **Still missing for full L3:** float32 observations, the `TimeLimit` truncation at 500 steps, and the 5-tuple `step`.

#### 3.4 LunarLander-v3 and LunarLanderContinuous-v3

| Case | Gymnasium (Box2D) | Gym.NET today (Aether) |
|---|---|---|
| `reset(seed)` observation | | 0 / 4 seeds identical; only the two leg-contact zeros coincide |
| Heuristic episode, discrete, seeds 0 / 1 / 42 / 1000 | 152 / 205 / 235 / 424 steps, rewards 297.4 / 260.9 / 256.0 / 285.6 (lands) | 192 / 499 / 200 / 209 steps, rewards 52.2 / 12.4 / 19.3 / −0.5 |
| Heuristic episode, continuous | 200 / 190 / 203 / 230 steps, rewards 306.0 / 273.9 / 274.3 / 303.8 | 156 / 156 / 184 / 167 steps, rewards 46.9 / 39.6 / 36.4 / 15.9 |
| Random play, observations outside the env's own observation space | 0 / 2,000 for every seed | 221, 185, 474, 156 of 2,000 (7.8–23.7%, B-19) |
| Observation-space bounds | v3: ±2.5, ±2.5, ±10, ±10, ±2π, ±10, [−0, 1], [−0, 1] | v2: ±1.5, ±1.5, ±5, ±5, ±π, ±5, [0, 1], [0, 1] |
| Continuous `step(action_space.sample())` | works | `IndexError`: the action space is 0-d (B-20) |

Gym.NET's LunarLander draws from Gymnasium's generator now, but it remains a different simulation, not a divergent copy:
- another physics engine (Aether, which leaves Box2D 2.3.10's trajectory at step 1 even in free fall; port plan A.10);
- v2 geometry and bounds;
- its own draw order (B-17, B-18);
- contact-listener and termination differences (B-21 to B-24).

Its episodes don't resemble Gymnasium's. The random-play share moved with the new stream (it was 6.5–16.3% on the legacy generator), but the defect is the same.

#### 3.5 Rendering (CartPole, `rgb_array`)

| Frames | Identical | Pixels that differ | Max channel delta |
|---:|---:|---:|---:|
| 12 (4 seeds × steps 0, 10, 50, rendered from the same state) | 0 | 2,729–2,981 of 240,000 (1.14–1.24%) | 255 |

Gymnasium draws with pygame's `gfxdraw` (antialiased polygon outlines, pole colour (202, 152, 101)). Gym.NET fills ImageSharp polygons with (204, 153, 102) (B-16). The share matches the port plan's A.9 (1.07–1.14%) and rises slightly with the pole angle. LunarLander frames weren't compared: its states can't be made equal.

#### 3.6 API surface (L1)

Gym.NET exposes 47 public types. The ones with a Gymnasium counterpart:
- `Env`/`IEnv`, `Space`, `Box`, `Discrete`;
- the step result `Step`;
- `Seeding`, and four exceptions including the new `Error`;
- `CartPoleEnv`, `LunarLanderEnv`;
- the baselines-style `VecEnv` family.

The rest are Gym.NET internals: collections, merging, reflection, a threading scheduler that throws on machines with 32 or more logical CPUs, and the viewer types.

The port plan's inventory (§2.1, same Gymnasium pin) counted 258 in-scope names. The statuses of four of them have changed since:
- `Box` and `Discrete` sample and check membership for every shape, so they move from Broken to Diverges.
- `seeding.np_random` has a faithful port (`Seeding.NpRandom`), so it moves from Missing to Matches.
- `error.Error` exists (`Gym.Exceptions.Error`), but Gym.NET's older exceptions don't derive from it yet, so it moves from Missing to Diverges.

LunarLander stays Broken: observations leave its own space, and continuous sampling throws.

| Status | Port plan (`pr/24`) | Now |
|---|---:|---:|
| Matches | 3 | 4 |
| Diverges | 12 | 15 |
| Broken | 4 | 2 |
| Missing | 218 | 216 |
| N/A | 21 | 21 |

Largest contract gaps (P1):

| Contract | Gymnasium | Gym.NET |
|---|---|---|
| Reset | `reset(*, seed=None, options=None) -> (obs, info)` | `Reset()` returns only the observation |
| Step | `step(action) -> (obs, reward, terminated, truncated, info)` | `Step` returns one `Done` flag, so no truncation |
| Rendering | `render_mode` constructor argument | `Render(mode)` argument |
| Metadata keys | `render_modes`, `render_fps` | `render.modes`, `video.frames_per_second` |
| Seeding | `np_random` / `np_random_seed` properties | none |
| Registration | a registry (63 ids) and `make` | no registry, no `make` |
| Wrappers | yes | none |
| Vectorization | `gymnasium.vector` | none |

### 4. What closes each gap

| Gap | Cause | Closed by | Proven feasible? |
|---|---|---|---|
| Seeded streams | the legacy MT19937 generator | done in `b51689d` (the part of F2 that today's code needs) | Yes: 783 / 783 golden values. Still open in F2: lazy seeding, `np_random_seed`, and `Generator.geometric` (needed by `spaces.Sequence`, missing on NumSharp master) |
| Remaining `Discrete`/`Box` semantics | pre-0.26 Gym port (B-03, B-10) | P2 (spaces re-port) | Yes: the sampling itself is already bit-identical |
| CartPole dynamics | float32 constants (B-11) | P4 | Yes: 2,000 / 2,000 steps |
| CartPole observations | float64 hand-outs (B-13) | P4 | Trivial once the step is float64: `(float)state` |
| Truncation, 5-tuple, `(obs, info)` | pre-0.26 contract | P1 (core), P3 (`TimeLimit`) | API design, no numerics |
| Frames | ImageSharp instead of pygame's rasterizer and colours | F5 rasterizer, P7 | Not yet: needs a port of pygame-ce's `draw`/`gfxdraw` behaviour (decision D4) |
| LunarLander | Aether instead of Box2D 2.3.10; v2 env code | F6 (Box2D port), P8 | Not yet: needs a Box2D 2.3.x port and its own goldens |
| 216 missing names | not ported | P1–P10 | — |

### 5. Found by this audit

| # | Finding | Status |
|---|---|---|
| 1 | Continuous LunarLander read its two throttles with NumSharp's unchecked `GetSingle`. A 0-d action (what its action space samples, B-20) made a Debug build of NumSharp fail an assertion that terminates the process, and a Release build read past the buffer. The bug came in with migration step 1 | Fixed in `2a1b827`: such actions now raise NumPy's `IndexError`, with a regression test |
| 2 | NumSharp `Generator.choice(array)` with `size=None` returns a one-element 1-D array where NumPy returns a scalar (the draw is the same) | Upstream NumSharp item; `Discrete.Sample` reads the single element |
| 3 | NumSharp `GetSingle`/`GetDouble`/… `(params long[])` don't bounds-check in Release | Upstream NumSharp item (cause of finding 1) |
| 4 | NumSharp's unseeded `SeedSequence().entropy` is the four drawn uint32 words, where NumPy's is the integer they encode | Upstream NumSharp item; `Seeding.NpRandom` reports the integer |
| 5 | NumSharp's legacy `randint` diverges from NumPy for `high` in [2^31, 2^32]; its `Generator.integers` does not (the golden's n = 3×10⁹ and 2^40 cases match) | Upstream item for the legacy API only; Gym.NET no longer uses it |
| 6 | Gym.NET's LunarLander leaves its own observation space in 7.8–23.7% of random-play steps (6.5–16.3% on the legacy stream; the plan measured 9.5% on `pr/24`) | Open, B-19, P8 |
| 7 | `DistributedScheduler` throws on machines with 32 or more logical CPUs, so `StepAsync` can't run there | B-34 area, deleted in P1 |

### 6. History: the first measurement (`2a1b827`, before the PCG64 upgrade)

Envs and spaces then drew from NumPy's legacy `RandomState` (MT19937), through `LegacySeeding`. That stream matched NumPy's `RandomState` on 602 of 602 draws (the golden of the since-replaced `LegacyRngParityTests`), but it is a different generator family from Gymnasium's. The results:

- **Seeding, spaces and CartPole reset:** 0 of 4 seeds identical in every case. Some individual integer draws matched by chance.
- **CartPole dynamics and frames:** as today.
- **LunarLander:** unrelated episodes. Random play left the observation space in 6.5–16.3% of steps.
- **L1:** 3 match, 14 diverge, 2 broken, 218 missing; `Seeding` and `Error` didn't exist yet.

The projection column, computed the way Gymnasium computes, was already 4 of 4 on every space and reset case. The upgrade has since reproduced that on Gym.NET's own code.

### 7. Reproduction

#### 7.1 Commands

The probes live in `tools/parity/audit/`. From the repository root, with the reference stack of the table above (`pip install pygame-ce==2.5.8 box2d==2.3.10`; Gymnasium runs from `refs/Gymnasium`):

```bash
OUT=<scratch directory>
PYTHONPATH=refs/Gymnasium python tools/parity/audit/gymnasium_side.py "$OUT"
dotnet run --no-cache -p:ReferenceNDArrayAnalyzer=false tools/parity/audit/gymnet_side.cs -- "$OUT"
python tools/parity/audit/compare.py "$OUT"
```

`--no-cache` matters: a file-based app otherwise reuses its last build even after the Gym.NET projects it references have changed.

The committed unit-test golden has its own recorder. It imports neither pygame nor Box2D and renders nothing, so it needs Gymnasium's core dependencies only, none of its extras:

```bash
PYTHONPATH=refs/Gymnasium python tools/parity/gymnasium_rng.py | tr -d '\r' > tests/Gym.Tests/Parity/gymnasium_rng.golden.txt
```

#### 7.2 What each probe does

- **`gymnasium_side.py`:** runs the cases in Gymnasium and writes `gymnasium.txt`, one `<case> <index> <value>` line per value, float64 as 16 hex digits and float32 as 8. It also writes CartPole `rgb_array` frames as raw `(400, 600, 3)` bytes, and the states it rendered them from.
- **`gymnet_side.cs`:** runs the same cases through Gym.NET into `gymnet.txt`. It injects Gymnasium's initial CartPole state and frame states into `CartPoleEnv` by reflection, so dynamics and frames are compared from identical states. It writes the projection into `projection.txt`, computed with NumSharp's `Generator` and a float64 transcription of `cartpole.py`. It also lists Gym.NET's public types.
- **`compare.py`:** scores the three recordings. A value matches only when its text is identical, so a float64 value never matches a float32 one.
