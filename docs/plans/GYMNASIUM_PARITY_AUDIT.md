# Gymnasium Parity Audit

| | |
|---|---|
| **Date** | 2026-09-25 |
| **Gym.NET** | branch `feat/numsharp-master-migration` @ `2a1b827` (NumSharp migration steps 1–3 done, step 4 not started) |
| **Gymnasium** | `refs/Gymnasium` @ `9e04324` (1.4.0 dev), run from source |
| **Reference stack** | Windows 11 x64; Python 3.12.12, NumPy 2.4.2, pygame-ce 2.5.8 (SDL 2.32.10), Box2D 2.3.10; .NET SDK 10.0.101 |
| **Parity levels** | as defined in [`GYMNASIUM_PORT_PLAN.md`](GYMNASIUM_PORT_PLAN.md) §1.2: L1 API, L2 behavioral, L3 byte, L4 pixel |
| **Method** | the same cases run live in Gymnasium and in Gym.NET and compared bit for bit (§6 reproduces every number) |

---

## 1. Verdict

**Nothing Gymnasium produces is byte-identical in Gym.NET yet.** Every space sample, CartPole reset, CartPole step, CartPole frame and LunarLander episode measured here differs from Gymnasium. That holds for all 4 seeds.

**What is byte-exact is the layer underneath.** NumSharp reproduces NumPy: the legacy `RandomState` stream Gym.NET draws from today matches NumPy on 602 of 602 draws (the committed golden). NumSharp's `Generator(PCG64)`, the generator Gymnasium draws from, matches it on every value measured here.

**The gap in the spaces and classic-control dynamics is porting work, not numerics.** A projection computes the measured cases the way Gymnasium does, on NumSharp's `Generator` and with a float64 CartPole step, and it is bit-identical on all of them:

- 4 of 4 seeds for every space and reset case;
- 2,000 of 2,000 CartPole steps, including reward and termination.

What still needs whole subsystems:

| Gap | Needs |
|---|---|
| Pixel-exact frames | the rasterizer (F5, P7) |
| LunarLander | a Box2D 2.3.x port (F6, P8) |
| The API contract | P1 to P6; 218 of 258 Gymnasium names are missing |

## 2. Scorecard

| Level | Area | Target (port plan §1.2) | Today | Projection: F2 + P2 + P4 done as planned |
|---|---|---|---|---|
| L1 API | Public surface (258 in-scope names) | every name | 3 match, 14 diverge, 2 broken, 218 missing (21 N/A) | unchanged; the projection is numerics only |
| L1 API | `Env` contract | `reset(seed, options) -> (obs, info)`, `step -> 5-tuple` | `Reset() -> obs`, `Step -> (obs, reward, done, info)` | P1 |
| L2 behavioral | Gymnasium's test suite | ported and green | 0 of 490 test functions ported (102 files) | P1–P6 |
| L3 byte | Seeding | `Generator(PCG64(SeedSequence(seed)))` | legacy MT19937: 0 / 4 seeds identical | 4 / 4 |
| L3 byte | `Discrete` samples (plain, masked, with start) | bit-identical | 0 / 4 seeds, in each case | 4 / 4 |
| L3 byte | `Box` samples (bounded, unbounded, low-bounded, integer) | bit-identical | 0 / 4 seeds, in each case | 4 / 4 |
| L3 byte | CartPole reset | bit-identical | 0 / 4 seeds (and float64 instead of float32) | 4 / 4 |
| L3 byte | CartPole dynamics, same initial state | bit-identical | 0 / 2,000 steps (diverges at step 1, every seed) | 2,000 / 2,000 |
| L3 byte | LunarLander | bit-identical | 0 / 4 reset observations; unrelated episodes | out of reach without Box2D (P8) |
| L4 pixel | CartPole frames | byte-identical | 0 / 12 frames (1.14–1.24% of pixels differ, max delta 255) | out of reach without the rasterizer (P7) |

## 3. Measurements

Seeds 0, 1, 42 and 1000 throughout. "Today" is Gym.NET's own code. "Projection" is Gymnasium's algorithm transcribed in C# on NumSharp (§6.2); it measures how far the planned ports would carry and is not shipped code.

### 3.1 Seeding

| Case | Today | Projection |
|---|---:|---:|
| First 8 `random()` doubles of the seeded generator | 0 / 32 values, 0 / 4 seeds | 32 / 32, 4 / 4 |

- **Gymnasium** seeds every env and space through `seeding.np_random(seed)`, i.e. `Generator(PCG64(SeedSequence(seed)))`.
- **Gym.NET** seeds NumPy's legacy `RandomState`: MT19937 through `LegacySeeding`, bit-exact with NumPy's `RandomState` (602 / 602, `LegacyRngParityTests`), but a different generator family.
- **Consequence:** no seeded Gym.NET stream can match Gymnasium until F2 replaces the family. The types differ too. Gymnasium takes any non-negative Python int as a seed and reports a 128-bit seed for unseeded envs. Gym.NET takes a `long` in `[0, 2**32 - 1]`, which is the legacy `RandomState` rule.

### 3.2 Spaces

| Case | Values | Today: values | Today: seeds identical | Projection: seeds identical |
|---|---:|---:|---:|---:|
| `Discrete(4).sample()` × 10 | 40 | 9 | 0 / 4 | 4 / 4 |
| `Discrete(4).sample(mask=[1,0,1,1])` × 5 | 20 | 9 | 0 / 4 | 4 / 4 |
| `Discrete(3, start=5).sample()` × 5 | 20 | 4 | 0 / 4 | 4 / 4 |
| `Box(-1, 1, (3,), float32).sample()` | 12 | 0 | 0 / 4 | 4 / 4 |
| `Box(-inf, inf, (3,), float64).sample()` | 12 | 0 | 0 / 4 | 4 / 4 |
| `Box(0, inf, (3,), float64).sample()` | 12 | 0 | 0 / 4 | 4 / 4 |
| `Box(0, 5, (8,), int32).sample()` | 32 | 6 | 0 / 4 | 4 / 4 |

Today's individual matches in the integer cases are chance: each draw has only 3 to 6 possible values. What else differs, beyond the generator:

- **Unbounded `Box` draws:** Gym.NET draws `normal(0.5, 1)`, Gymnasium a standard normal (B-08).
- **Integer `Box` samples:** Gym.NET never samples `high`, because Gymnasium's `high + 1` is missing (B-09). It also doesn't clip to the dtype's range, as Gymnasium's `sample` does.
- **`Discrete` shape:** Gym.NET's is `(4,)`, Gymnasium's `()` (B-03).
- **`Discrete` sample type:** Gym.NET returns a 0-d int64 `NDArray` where Gymnasium returns an `np.int64` scalar.
- **Validation and messages:** Gymnasium validates bounds and dtypes and raises its own errors; Gym.NET doesn't (B-10).

The projection applies Gymnasium's formulas (`start + integers(n)`, `choice(where(mask))`, a float64 `low + (high - low) * random()`, a standard normal, `exponential + low`, and `high + 1` / floor / clip for integers), and every case becomes bit-identical.

### 3.3 CartPole-v1

| Case | Today | Projection |
|---|---|---|
| `reset(seed)` observation | 0 / 4 seeds; dtype float64 (Gymnasium float32) | 4 / 4, float32 |
| `reset(seed)` internal state | 0 / 4 seeds | 4 / 4 |
| 500 open-loop steps from Gymnasium's initial state, per seed | 0 / 500 steps identical; first divergence at step 1 | 500 / 500, including reward and `terminated` |
| Observation-space bounds | differ: `±float.MaxValue` where Gymnasium has `±inf` (B-12) | not projected |

- **Why dynamics diverge today:** CartPole's constants are float32 (`tau = 0.02f` is 0.019999999552965164 as a double, B-11), so the state differs in its last bits from step 1 on every seed.
- **How far it drifts:** with this open-loop action sequence (0, 0, 1, 1, ...) the drift stays below 4e-6 over 500 steps, and termination happens on the same step as in Gymnasium. The port plan's A.7, replaying Gymnasium's own random actions, saw the drift grow to 27 and termination move.
- **The fix is proven:** the float64 transcription of `cartpole.py:164` with `Math.Cos`/`Math.Sin` is bit-identical for all 2,000 steps, which confirms A.7 on the current tree.
- **Still missing for full L3:** float32 observations (B-13), the `TimeLimit` truncation at 500 steps, and the 5-tuple `step`.

### 3.4 LunarLander-v3 and LunarLanderContinuous-v3

| Case | Gymnasium (Box2D) | Gym.NET today (Aether) |
|---|---|---|
| `reset(seed)` observation | | 0 / 4 seeds identical; only the two leg-contact zeros coincide |
| Heuristic episode, discrete, seeds 0 / 1 / 42 / 1000 | 152 / 205 / 235 / 424 steps, rewards 297.4 / 260.9 / 256.0 / 285.6 (lands) | 192 / 251 / 182 / 230 steps, rewards 17.2 / 36.8 / −46.3 / 40.9 |
| Heuristic episode, continuous | 200 / 190 / 203 / 230 steps, rewards 306.0 / 273.9 / 274.3 / 303.8 | 176 / 157 / 221 / 355 steps, rewards 23.8 / 41.9 / 47.1 / 35.4 |
| Random play, observations outside the env's own observation space | 0 / 2,000 for every seed | 166, 275, 326, 131 of 2,000 (6.5–16.3%, B-19) |
| Observation-space bounds | v3: ±2.5, ±2.5, ±10, ±10, ±2π, ±10, [−0, 1], [−0, 1] | v2: ±1.5, ±1.5, ±5, ±5, ±π, ±5, [0, 1], [0, 1] |
| Continuous `step(action_space.sample())` | works | `IndexError`: the action space is 0-d (B-20) |

Gym.NET's LunarLander is a different simulation, not a divergent copy. It uses another physics engine (Aether, which leaves Box2D 2.3.10's trajectory at step 1 even in free fall; port plan A.10), v2 geometry and bounds, and its own draw order (B-17). It also has contact-listener and termination differences (B-21 to B-24). Its episodes don't resemble Gymnasium's.

### 3.5 Rendering (CartPole, `rgb_array`)

| Frames | Identical | Pixels that differ | Max channel delta |
|---:|---:|---:|---:|
| 12 (4 seeds × steps 0, 10, 50, rendered from the same state) | 0 | 2,729–2,981 of 240,000 (1.14–1.24%) | 255 |

Gymnasium draws with pygame's `gfxdraw` (antialiased polygon outlines, pole colour (202, 152, 101)). Gym.NET fills ImageSharp polygons with (204, 153, 102) (B-16). The share matches the port plan's A.9 (1.07–1.14%) and rises slightly with the pole angle. LunarLander frames weren't compared: its states can't be made equal.

### 3.6 API surface (L1)

Gym.NET exposes 46 public types. The ones with a Gymnasium counterpart:
- `Env`/`IEnv`, `Space`, `Box`, `Discrete`;
- the step result `Step`;
- three exceptions;
- `CartPoleEnv`, `LunarLanderEnv`;
- the baselines-style `VecEnv` family.

The rest are Gym.NET internals: collections, merging, reflection, a threading scheduler that throws on machines with 32 or more logical CPUs, the viewer types, and `LegacySeeding`.

The port plan's inventory (§2.1, same Gymnasium pin) counted 258 in-scope names. The migration added no Gymnasium name and removed none. It did change two statuses: `Box` and `Discrete` sample and check membership for every shape now, so they move from Broken to Diverges. LunarLander stays Broken: observations leave its own space, and continuous sampling throws.

| Status | Port plan (`pr/24`) | Now |
|---|---:|---:|
| Matches | 3 | 3 |
| Diverges | 12 | 14 |
| Broken | 4 | 2 |
| Missing | 218 | 218 |
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

## 4. What closes each gap

| Gap | Cause | Closed by | Proven feasible? |
|---|---|---|---|
| Every seeded stream | MT19937 instead of `Generator(PCG64)` | F2 (seeding port) | Yes: projection 4 / 4 on every case; NumSharp's `Generator` covers 16 of the 17 draw patterns Gymnasium uses (A.6); `geometric` is still missing on NumSharp master (needed by `spaces.Sequence`) |
| `Discrete`/`Box` sampling semantics | pre-0.26 Gym port (B-03, B-08 to B-10) | P2 (spaces re-port) | Yes: projection 4 / 4 |
| CartPole dynamics | float32 constants (B-11) | P4 | Yes: 2,000 / 2,000 steps |
| CartPole observations | float64 hand-outs (B-13) | P4 | Trivial once the step is float64: `(float)state` |
| Truncation, 5-tuple, `(obs, info)` | pre-0.26 contract | P1 (core), P3 (`TimeLimit`) | API design, no numerics |
| Frames | ImageSharp instead of pygame's rasterizer and colours | F5 rasterizer, P7 | Not yet: needs a port of pygame-ce's `draw`/`gfxdraw` behaviour (decision D4) |
| LunarLander | Aether instead of Box2D 2.3.10; v2 env code | F6 (Box2D port), P8 | Not yet: needs a Box2D 2.3.x port and its own goldens |
| 218 missing names | not ported | P1–P10 | — |

## 5. Found by this audit

| # | Finding | Status |
|---|---|---|
| 1 | Continuous LunarLander read its two throttles with NumSharp's unchecked `GetSingle`. A 0-d action (what its action space samples, B-20) made a Debug build of NumSharp fail an assertion that terminates the process, and a Release build read past the buffer. The bug came in with migration step 1 | Fixed in `2a1b827`: such actions now raise NumPy's `IndexError`, with a regression test |
| 2 | NumSharp `Generator.choice(array)` with `size=None` returns a one-element 1-D array where NumPy returns a scalar (the draw is the same) | Upstream NumSharp item; matters for P2's masked `Discrete.sample` |
| 3 | NumSharp `GetSingle`/`GetDouble`/… `(params long[])` don't bounds-check in Release | Upstream NumSharp item (cause of finding 1) |
| 4 | Gym.NET's LunarLander leaves its own observation space in 6.5–16.3% of random-play steps (the plan measured 9.5% on `pr/24`) | Open, B-19, P8 |
| 5 | Earlier this session: NumSharp's legacy `randint` diverges from NumPy for `high` in [2^31, 2^32]; `DistributedScheduler` throws on machines with 32 or more logical CPUs, so `StepAsync` can't run there | Upstream NumSharp item; B-34 area, deleted in P1 |

## 6. Reproduction

### 6.1 Commands

The probes live in `tools/parity/audit/`. From the repository root, with the reference stack of the table above (`pip install pygame-ce==2.5.8 box2d==2.3.10`; Gymnasium runs from `refs/Gymnasium`):

```bash
OUT=<scratch directory>
PYTHONPATH=refs/Gymnasium python tools/parity/audit/gymnasium_side.py "$OUT"
dotnet run --no-cache -p:ReferenceNDArrayAnalyzer=false tools/parity/audit/gymnet_side.cs -- "$OUT"
python tools/parity/audit/compare.py "$OUT"
```

`--no-cache` matters: a file-based app otherwise reuses its last build even after the Gym.NET projects it references have changed.

### 6.2 What each probe does

- **`gymnasium_side.py`:** runs the cases in Gymnasium and writes `gymnasium.txt`, one `<case> <index> <value>` line per value, float64 as 16 hex digits and float32 as 8. It also writes CartPole `rgb_array` frames as raw `(400, 600, 3)` bytes, and the states it rendered them from.
- **`gymnet_side.cs`:** runs the same cases through Gym.NET into `gymnet.txt`. It injects Gymnasium's initial CartPole state and frame states into `CartPoleEnv` by reflection, so dynamics and frames are compared from identical states. It writes the projection into `projection.txt`, computed with NumSharp's `Generator` and a float64 transcription of `cartpole.py`. It also lists Gym.NET's public types.
- **`compare.py`:** scores the three recordings. A value matches only when its text is identical, so a float64 value never matches a float32 one.
