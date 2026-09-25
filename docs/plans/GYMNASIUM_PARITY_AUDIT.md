# Gymnasium Parity Audit

| | |
|---|---|
| **Date** | 2026-09-25 |
| **Gym.NET** | branch `feat/numsharp-master-migration` @ `b51689d` (NumSharp migration steps 1–3, then Gymnasium's PCG64 seeding; ownership step 4 not started). The first measurement, at `2a1b827` before the PCG64 upgrade, is kept in §6 |
| **Gymnasium** | `refs/Gymnasium` @ `9e04324` (1.4.0 dev), run from source |
| **Reference stack** | Windows 11 x64; Python 3.12.12, NumPy 2.4.2, pygame-ce 2.5.8 (SDL 2.32.10), Box2D 2.3.10; .NET SDK 10.0.101 |
| **Parity levels** | as defined in [`GYMNASIUM_PORT_PLAN.md`](GYMNASIUM_PORT_PLAN.md) §1.2: L1 API, L2 behavioral, L3 byte, L4 pixel |
| **Method** | the same cases run live in Gymnasium and in Gym.NET and compared bit for bit (§7 reproduces every number) |

---

## 1. Verdict

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

## 2. Scorecard

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

## 3. Measurements

Seeds 0, 1, 42 and 1000 throughout. "Today" is Gym.NET's own code. "Projection" is Gymnasium's algorithm transcribed in C# on NumSharp (§7.2); it measures how far the planned ports would carry and is not shipped code.

### 3.1 Seeding

| Case | Today | Projection |
|---|---:|---:|
| First 8 `random()` doubles of the seeded generator | 32 / 32 values, 4 / 4 seeds | 32 / 32, 4 / 4 |

- **How both sides seed:** Gymnasium goes through `seeding.np_random(seed)`, and Gym.NET through its port, `Gym.Utils.Seeding.NpRandom`. Both use the same validation, the same message ("Seed must be greater or equal to zero, actual value: …", raised as a port of `gymnasium.error.Error`), and the same generator.
- **Seed range:** `Seeding.NpRandom` takes any non-negative integer as a `BigInteger?`, because Gymnasium's seeds are Python ints of any size, and the seed it reports for an unseeded generator is 128-bit entropy. The envs' and spaces' `Seed(long)` take any non-negative `long`; the legacy ceiling of 2^32 − 1 is gone.
- **Upstream NumSharp item:** NumSharp holds that unseeded entropy as four 32-bit words, where NumPy holds the integer they encode. It is reported as that integer, and passing it back rebuilds the stream, except in the 1-in-2^32 case where the top word is zero.
- **Still missing (P1/P2):** lazy seeding, the `np_random`/`np_random_seed` properties, and `Space.seed` returning its seed.

### 3.2 Spaces

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

### 3.3 CartPole-v1

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

### 3.4 LunarLander-v3 and LunarLanderContinuous-v3

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

### 3.5 Rendering (CartPole, `rgb_array`)

| Frames | Identical | Pixels that differ | Max channel delta |
|---:|---:|---:|---:|
| 12 (4 seeds × steps 0, 10, 50, rendered from the same state) | 0 | 2,729–2,981 of 240,000 (1.14–1.24%) | 255 |

Gymnasium draws with pygame's `gfxdraw` (antialiased polygon outlines, pole colour (202, 152, 101)). Gym.NET fills ImageSharp polygons with (204, 153, 102) (B-16). The share matches the port plan's A.9 (1.07–1.14%) and rises slightly with the pole angle. LunarLander frames weren't compared: its states can't be made equal.

### 3.6 API surface (L1)

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

## 4. What closes each gap

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

## 5. Found by this audit

| # | Finding | Status |
|---|---|---|
| 1 | Continuous LunarLander read its two throttles with NumSharp's unchecked `GetSingle`. A 0-d action (what its action space samples, B-20) made a Debug build of NumSharp fail an assertion that terminates the process, and a Release build read past the buffer. The bug came in with migration step 1 | Fixed in `2a1b827`: such actions now raise NumPy's `IndexError`, with a regression test |
| 2 | NumSharp `Generator.choice(array)` with `size=None` returns a one-element 1-D array where NumPy returns a scalar (the draw is the same) | Upstream NumSharp item; `Discrete.Sample` reads the single element |
| 3 | NumSharp `GetSingle`/`GetDouble`/… `(params long[])` don't bounds-check in Release | Upstream NumSharp item (cause of finding 1) |
| 4 | NumSharp's unseeded `SeedSequence().entropy` is the four drawn uint32 words, where NumPy's is the integer they encode | Upstream NumSharp item; `Seeding.NpRandom` reports the integer |
| 5 | NumSharp's legacy `randint` diverges from NumPy for `high` in [2^31, 2^32]; its `Generator.integers` does not (the golden's n = 3×10⁹ and 2^40 cases match) | Upstream item for the legacy API only; Gym.NET no longer uses it |
| 6 | Gym.NET's LunarLander leaves its own observation space in 7.8–23.7% of random-play steps (6.5–16.3% on the legacy stream; the plan measured 9.5% on `pr/24`) | Open, B-19, P8 |
| 7 | `DistributedScheduler` throws on machines with 32 or more logical CPUs, so `StepAsync` can't run there | B-34 area, deleted in P1 |

## 6. History: the first measurement (`2a1b827`, before the PCG64 upgrade)

Envs and spaces then drew from NumPy's legacy `RandomState` (MT19937), through `LegacySeeding`. That stream matched NumPy's `RandomState` on 602 of 602 draws (the golden of the since-replaced `LegacyRngParityTests`), but it is a different generator family from Gymnasium's. The results:

- **Seeding, spaces and CartPole reset:** 0 of 4 seeds identical in every case. Some individual integer draws matched by chance.
- **CartPole dynamics and frames:** as today.
- **LunarLander:** unrelated episodes. Random play left the observation space in 6.5–16.3% of steps.
- **L1:** 3 match, 14 diverge, 2 broken, 218 missing; `Seeding` and `Error` didn't exist yet.

The projection column, computed the way Gymnasium computes, was already 4 of 4 on every space and reset case. The upgrade has since reproduced that on Gym.NET's own code.

## 7. Reproduction

### 7.1 Commands

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

### 7.2 What each probe does

- **`gymnasium_side.py`:** runs the cases in Gymnasium and writes `gymnasium.txt`, one `<case> <index> <value>` line per value, float64 as 16 hex digits and float32 as 8. It also writes CartPole `rgb_array` frames as raw `(400, 600, 3)` bytes, and the states it rendered them from.
- **`gymnet_side.cs`:** runs the same cases through Gym.NET into `gymnet.txt`. It injects Gymnasium's initial CartPole state and frame states into `CartPoleEnv` by reflection, so dynamics and frames are compared from identical states. It writes the projection into `projection.txt`, computed with NumSharp's `Generator` and a float64 transcription of `cartpole.py`. It also lists Gym.NET's public types.
- **`compare.py`:** scores the three recordings. A value matches only when its text is identical, so a float64 value never matches a float32 one.
