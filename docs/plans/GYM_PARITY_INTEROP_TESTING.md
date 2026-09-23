# Gymnasium Parity via Live Interop Testing

| | |
|---|---|
| **Status** | Draft 1, 2026-09-23 (see §14 for what's verified and what's next) |
| **Companion to** | `docs/plans/GYMNASIUM_PORT_PLAN.md` (§14 verification strategy); this document designs its live half |
| **Modelled on** | `refs/NumSharp`: `src/NumSharp.Interop.pythonnet` + `test/NumSharp.Tests.Interop` (and its `CLAUDE.md`) |
| **Oracle** | `refs/Gymnasium` @ `9e04324`, imported **in-process** through pythonnet |

## 1. Purpose

Prove, test by test, that Gym.NET is a **byte-for-byte and pixel-for-pixel** clone of Gymnasium. The test process embeds CPython, imports the pinned Gymnasium, and asks it to compute the answer *right now*: the same env, the same seed, the same actions. Every observation, reward, flag, info entry, space sample, RNG state and `rgb_array` frame must then be byte-identical to what the .NET port produced.

NumSharp already runs this model. Its offline corpus bit-compares against committed NumPy output with no Python at test time. Its `NumSharp.Tests.Interop` embeds live NumPy and asserts `NumSharp.np.OP(x)` and `numpy.OP(x)` produce the same bytes over literally the same buffer. Gym.NET adopts both halves:

| Oracle | Needs Python | Proves | Runs where |
|---|---|---|---|
| **Live interop** (this document) | yes, embedded | parity against the installed Gymnasium, over the actual bytes produced, with no serialization in between | reference CI job and developer boxes |
| **Offline goldens** (port plan §5.4, §14.3) | only to record | parity against a committed snapshot | every CI job |

Both are driven by **the same case definitions** (§9). The live harness runs a case against Python; record mode writes the case's golden; replay mode compares against the file.

## 2. What NumSharp's interop gives us (reuse, don't rebuild)

From `refs/NumSharp` master (v0.70.0 + 46):

- **`NumSharp.Interop.pythonnet`** (NuGet, co-versioned with NumSharp). It crosses arrays zero-copy between `NDArray` and numpy: `nd.ToNumpy()` is a view, `ToNumpyCopy()` an independent copy, `py.ToNDArray()` a copy, `py.AsNDArray()` a view. It also brings buffer leases, GIL-safe release, and a codec. It depends only on `pythonnet [3.0.5, 4.0.0)`; 3.0.5 drives Python 3.7–3.13, and 3.1.0 is needed for 3.14.
- **The live-test pattern** (`test/NumSharp.Tests.Interop/CLAUDE.md`):
  - **`PythonSession`:** one embedded engine per process, started in `[AssemblyInitialize]`. CPython and numpy can't re-initialize after `Py_Finalize`, so the lifecycle is assembly-scoped. Discovery order: `PYTHONNET_PYDLL`, then `python`/`python3`, then `~/.claude/python`, then Windows `Programs\Python3*`.
  - **`EnsureOrInconclusive()`:** by default a missing engine **fails**, because "a self-skipping suite that proves nothing is worse than a red one". `…_REQUIRE_ENGINE=0` downgrades it to Inconclusive on a dev box without Python.
  - **`[assembly: DoNotParallelize]`:** there is one interpreter and one GIL.
  - **`InteropTestBase`:** a `Scope` with numpy imported, `Gil()`, export/import helpers, `SkipUnless(module)`, and a **leak gate**. It snapshots `LiveExports`/`LiveImports` on entry and fails the test unless both return to baseline, so every test is also a no-leak check.
  - **`ByteContract`:** canonical C-order bytes on both sides (`np.ascontiguousarray(a).tobytes('C')` vs NumSharp's own `ascontiguousarray` plus a raw copy, read independently so a match isn't a tautology).
  - **Host-pinned cells go Inconclusive off the reference architecture** (`SkipByteExactOnArm64`), never red.
- **Pitfalls it already paid for:**
  - Every `PyObject` must be created **and disposed** under the GIL.
  - A .NET method invoked *from* Python does **not** hold the GIL.
  - A numpy scalar isn't a Python `int` (use `int(x)` or `.item()`).
  - A `dynamic` read keeps a CLR wrapper alive until the GC runs.
  - Call `RuntimeData.FormatterType = typeof(NoopFormatter)` before `PythonEngine.Shutdown()` on .NET 8+.
  - On Windows, `numpy.sum(int32)` accumulates in the 32-bit C `long`, so integer reductions compare at int64.

Gym.NET references `NumSharp.Interop.pythonnet` at the same version as NumSharp (≥ 0.70.0, per the port plan's decision D3), and ports the harness pieces above rather than reinventing them.

## 3. The reference Python stack

A pinned venv, created by `tools/parity/setup_venv.{ps1,sh}` from `tools/parity/requirements.txt`:

| Package | Pin | Why |
|---|---|---|
| Python | 3.12.x | Inside pythonnet 3.0.5's range; matches the port plan's reference |
| numpy | 2.4.2 | NumSharp is verified byte-exact against it (port plan, Appendix A.6) |
| pygame-ce | 2.5.8 (SDL 2.32.10) | Rasterizer oracle; Gymnasium only sets a floor (`>=2.1.3`), so the pin matters |
| box2d | 2.3.10 | Gymnasium's pin for Python < 3.14 |
| mujoco | per port plan D6 | Native physics |
| cloudpickle, typing-extensions, farama-notifications | Gymnasium runtime deps | |

**Gymnasium itself is imported from `refs/Gymnasium` via `sys.path`, never pip-installed.** An editable install would write build artifacts into the submodule, and a wheel could drift from the pinned SHA.

Verified on 2026-09-23: this venv installs cleanly on Windows 11 x64 from Python 3.12.12 (numpy 2.4.2, pygame-ce 2.5.8, Box2D 2.3.10), and `refs/Gymnasium` imports from `sys.path` and runs CartPole, Pendulum and `rgb_array` rendering.

**Session manifest check.** At `[AssemblyInitialize]` the session reads `gymnasium.__file__`, the `refs/Gymnasium` HEAD SHA, `numpy.__version__`, `pygame.version.ver`, `Box2D.__version__`, `sys.version`, and the OS and CPU. A mismatch with `tools/parity/manifest.json` fails the run: a live oracle on the wrong stack proves nothing. `GYMNET_PARITY_ALLOW_STACK_DRIFT=1` downgrades the mismatch to Inconclusive for local exploration.

## 4. Project layout

```
tests/Gym.Tests.Parity/                 live-interop test project (net8.0;net10.0)
  AssemblyInfo.cs                       [assembly: DoNotParallelize]
  Session/GymPythonSession.cs           engine start, venv + refs/Gymnasium on sys.path, manifest check
  Session/ParityTestBase.cs             InteropTestBase port: Scope, Gil(), leak gate, SkipUnless
  Oracle/ByteContract.cs                NDArray/scalar/Info vs PyObject, bitwise
  Oracle/PyGym.cs                       live builders: make, reset, step, render, space ops
  Pair/GymPair.cs                       lockstep differential runner (§5)
  Pair/Divergence.cs                    first-divergence report and shrinking
  Frames/FrameDiff.cs                   pixel diff and PNG artifacts (§6)
  Bridge/PyEnvShim.py, DotNetEnv.cs     reverse bridge: a .NET env seen by Python (§8)
  Cases/*.cs                            ParityCase catalog shared with golden record/replay (§9)
  Tests/Core|Spaces|Registry|Wrappers|Vector|Envs/*LiveParityTests.cs
tools/parity/requirements.txt, setup_venv.ps1, setup_venv.sh, manifest.json
```

Environment variables, mirroring NumSharp's:

| Variable | Default | Effect |
|---|---|---|
| `PYTHONNET_PYDLL` | discovered | The Python DLL to host (the base interpreter's `python312.dll`, also when a venv is used) |
| `GYMNET_PARITY_VENV` | `tools/parity/.venv` | Its `site-packages` is prepended to `sys.path` |
| `GYMNET_PYTHONNET_REQUIRE_ENGINE` | `1` | `0`: no engine means Inconclusive instead of fail |
| `GYMNET_PARITY_ALLOW_STACK_DRIFT` | `0` | `1`: manifest mismatch means Inconclusive instead of fail |
| `GYMNET_PARITY_ARTIFACTS` | test results dir | Where divergence dumps and frame PNGs go |

The session also sets deterministic process state before importing anything:

- `PYTHONHASHSEED=0` (Python `set` ordering)
- `SDL_VIDEODRIVER=dummy` and `PYGAME_HIDE_SUPPORT_PROMPT=1` (headless pygame)
- `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`

## 5. Lockstep differential: `GymPair`

The core runner creates the same environment on both sides and drives them in lockstep.

```csharp
/// <summary>
/// Drives one Gymnasium env (live, in-process) and its Gym.NET port in lockstep and asserts that every
/// transition is byte-identical. The first divergence fails the test with a full report (see <see cref="Divergence"/>).
/// </summary>
/// <remarks>
/// Both sides are created from the same id and kwargs, reset with the same seed, and fed the same actions.
/// Comparison order per step is fixed (obs, reward, terminated, truncated, info, RNG state) so reports are stable.
/// All Python calls run under the GIL; the .NET side runs outside it.
/// </remarks>
public sealed class GymPair : IDisposable
{
    /// <summary>Creates both environments.</summary>
    /// <param name="id">A registered id, e.g. <c>"CartPole-v1"</c>; used for <c>gymnasium.make</c> and <c>Gymnasium.Make</c>.</param>
    /// <param name="kwargs">Constructor kwargs with Python spelling; passed verbatim to both registries.</param>
    /// <param name="viaMake">True to include the make() wrapper stack on both sides; false to construct the bare env class.</param>
    /// <exception cref="ParityException">Thrown when either side fails to construct, or when they fail differently.</exception>
    public GymPair(string id, IReadOnlyDictionary<string, object?>? kwargs = null, bool viaMake = true) { /* … */ }

    /// <summary>Resets both sides with <paramref name="seed"/> and compares the returned (obs, info).</summary>
    /// <param name="seed">The seed passed to both <c>reset(seed=…)</c> calls.</param>
    /// <returns>The .NET reset result, after it has been verified against Python.</returns>
    /// <exception cref="ParityException">Thrown on the first byte difference.</exception>
    public ResetResult<object> Reset(long seed) { /* … */ }

    /// <summary>Steps both sides with the same action and compares the full transition bitwise.</summary>
    /// <param name="action">The action; converted to a Python object once and reused, so both sides see identical bits.</param>
    /// <returns>The .NET transition, after it has been verified against Python.</returns>
    /// <exception cref="ParityException">Thrown on the first byte difference, including a difference in raised exceptions.</exception>
    public StepResult<object> Step(object action) { /* … */ }

    /// <summary>Closes both environments.</summary>
    public void Dispose() { /* … */ }
}
```

**What is compared, per call:**

| Field | Comparison |
|---|---|
| Observation | dtype + shape + C-order bytes (`ByteContract`); for `Discrete`: int value and Python type (`int` vs `np.int64`) |
| Reward | float64 bit pattern (a Python `float` or `np.float64`; `np.float32` rewards are widened exactly) |
| `terminated`, `truncated` | equality |
| `info` | same keys **in the same order**; values compared recursively (arrays by bytes, scalars by bits); a per-wrapper allowlist excludes wall-clock entries (`RecordEpisodeStatistics` `"t"`) |
| RNG state | `env.unwrapped.np_random.bit_generator.state` (PCG64 `state`/`inc`) vs NumSharp's PCG64 state. This catches a missing or extra draw at the step it happens, even before any value differs |
| Exceptions | same type (mapped through port plan §3.4) and same message |

**Action sources:**
- **Open-loop:** actions come from a fixed recorded list, or from Python's policy evaluated on Python's own observations. This isolates dynamics from the policy.
- **Sampled:** both sides call `action_space.seed(s)`, then `sample()` each step. The two samples must first match each other, which also tests space sampling.
- **Heuristic:** an env-provided controller such as `lunar_lander.heuristic`, evaluated on the Python observation and fed to both sides.

**State injection** is for unit-level dynamics checks. The pair writes the same state into both envs (Python `env.unwrapped.state = …`; .NET the equivalent public property), steps once, and compares. This isolates one step of dynamics from reset and RNG. In the 2026-09-23 scan this technique separated "wrong float32 constants" from "platform math" in CartPole (port plan Appendix A.7).

**Divergence report and shrinking** (`Divergence`): the report gives the first step, field, element index, both values (hex bits and decimal), the ulp distance, the action history, and a dump of both envs' full state at steps *k−1* and *k*, written to `GYMNET_PARITY_ARTIFACTS`. The shrinker replays from reset to find the shortest action prefix that still diverges, then tries state injection at *k−1* to confirm whether the fault lies in `step` alone.

## 6. Pixel parity

`FrameDiff.AssertSame(pythonFrame, netFrame)` compares two `rgb_array` renders:
- **Python side:** `env.render()` returns an `(H, W, 3) uint8` ndarray. It crosses zero-copy with `AsNDArray()` and is compared by bytes.
- **.NET side:** `Render()` returns `RenderResult.Rgb(NDArray)`.

Frames are rendered at:
- fixed states injected on both sides (3 per env, as in the golden plan);
- every *n*-th step of a paired trajectory;
- edge states: pole at ±12°, cart at ±2.4, lander touching down, leg contact, particles alive.

On failure the test writes `python.png`, `dotnet.png` and an amplified diff PNG, with statistics:
- differing-pixel count and fraction;
- bounding box;
- maximum channel delta;
- a histogram of per-pixel deltas.

The histogram tells anti-aliasing noise (delta ≤ 2 along edges) from geometry or colour errors (large deltas in bands). The 2026-09-23 scan measured the current CartPole at 1.07–1.14% of pixels differing, with max delta 255 (port plan A.9); that is exactly the output this report is designed to make actionable.

pygame's own primitives are also tested directly, because the rasterizer (port plan F5) is shared by every env:
- `Surface` calls (`gfxdraw.aapolygon`, `draw.polygon`, `transform.smoothscale`, `math.Vector2.rotate_rad`, …) run on both sides with identical arguments, over a generated corpus of shapes, angles and sizes;
- the resulting surfaces are compared by bytes.

## 7. Spaces, RNG, wrappers, vector

| Area | Live checks |
|---|---|
| RNG | `seeding.np_random(seed)` on both sides: identical seed-sequence entropy and PCG64 state, then N draws of every pattern Gymnasium uses (port plan A.6), compared by bits after each call |
| Spaces | For every space type and a generated grid of constructor args: `seed(s)` return values (incl. Dict/Tuple subseeds); `sample()` × N; `sample(mask=…)`/`sample(probability=…)`; `contains` over samples and adversarial inputs (wrong dtype, shape, bounds, NaN); `flatten`/`unflatten`/`flatdim`/`flatten_space`; `to_jsonable`/`from_jsonable`; `repr`; equality; validation errors (type and message) |
| Registry | Enumerate `gymnasium.envs.registry` live and assert the .NET registry has every in-scope id with identical `EnvSpec` fields (`max_episode_steps`, `reward_threshold`, `nondeterministic`, `order_enforce`, `kwargs`, `disable_env_checker`); `pprint_registry` text; id parsing and version resolution over a generated set of valid and invalid ids, compared by result or by error type and message |
| Wrappers | Apply the same wrapper stack on both sides (single wrappers and random compositions) and run a `GymPair`; stateful wrappers (`FrameStack`, `Normalize*`, `DelayObservation`, `StickyAction`) run long episodes |
| Vector | `make_vec(id, num_envs, vectorization_mode)` on both sides for sync (and async, which must equal sync); every `AutoresetMode`; compare batched observations, rewards, terminations and truncations bytewise, plus the vector `info` layout including `"_" + key` masks |

## 8. Reverse bridge: run Gymnasium's own tools on the .NET envs

Gymnasium ships verification utilities: `check_env`, `passive_env_checker`, `check_environments_match`, `data_equivalence`, and a pytest suite of about 490 functions that iterate the registry. Rather than trusting only the ported copies, the harness also runs **the originals** against the .NET envs.

`PyEnvShim.py` defines a `gymnasium.Env` subclass whose methods delegate to a CLR `DotNetEnv` object through pythonnet. Observations are exported zero-copy, and spaces are mirrored as real Gymnasium spaces built from the .NET space parameters. Then:
- `gymnasium.utils.env_checker.check_env(shim)` must pass with no warnings, exactly as it does for the Python env;
- `check_environments_match(python_env, shim, num_steps, seed)` is Gymnasium's own lockstep comparator, run on both envs;
- selected pytest modules (the env API and determinism tests) can run in-process with the registry temporarily pointed at shim factories.

GIL trap (from NumSharp): pythonnet releases the GIL around managed method bodies. Any conversion performed *inside* a shim callback must therefore acquire it itself. NumSharp's interop does this by default (`RequireGIL = true`); keep it on.

## 9. One case catalog, three modes

```csharp
/// <summary>
/// A parity case: everything needed to reproduce one comparison on either stack.
/// The same instance drives live interop, golden recording and golden replay, so the offline gate
/// can never test something the live gate does not.
/// </summary>
/// <param name="Id">Registered env id.</param>
/// <param name="Kwargs">Constructor kwargs, Python spelling.</param>
/// <param name="Seed">Reset seed.</param>
/// <param name="Actions">How actions are produced (open-loop list, sampled with a seed, or a named heuristic).</param>
/// <param name="Steps">Maximum steps; the case stops earlier at termination or truncation.</param>
/// <param name="ViaMake">Whether the make() wrapper stack is included.</param>
/// <param name="FrameEvery">Render and compare a frame every N steps (0 = never).</param>
public sealed record ParityCase(string Id, IReadOnlyDictionary<string, object?> Kwargs, long Seed,
                                ActionSource Actions, int Steps, bool ViaMake, int FrameEvery);
```

| Mode | Runs | Oracle |
|---|---|---|
| `Live` | `GymPair` against embedded Gymnasium | Python, now |
| `Record` | Python side only, through the same bridge | writes `tests/Golden/<suite>/<case>.jsonl.gz` and PNGs (port plan §5.4) |
| `Replay` | .NET side only | the committed golden files |

CI layout (port plan §14.8), plus:
- **Windows reference job:** venv, then `Live` with `GYMNET_PYTHONNET_REQUIRE_ENGINE=1`, then `Replay`.
- **Other jobs:** `Replay` only.
- **Nightly soak:** `Live` over randomized cases (§10).

## 10. Differential fuzzing

A seeded generator draws:
- env ids and valid kwargs within documented ranges (plus a slice of out-of-range values to exercise warnings and errors);
- seeds;
- action sources and random wrapper stacks;
- vector configurations.

Each generated case runs in `Live` mode. On failure the shrinker (§5) minimizes it, and the minimized case is written to `Cases/Regressions/` as a `ParityCase` with its golden. From then on every CI run replays it, the same loop as NumSharp's nightly `fuzz-soak` and its committed `regressions/`.

## 11. Harness self-tests (proving it has teeth)

Following NumSharp's `HarnessSelfTests`, a parity harness that can't fail is worthless. Negative controls:

| Control | Must report |
|---|---|
| CartPole with `tau = 0.02f` (the current Gym.NET bug) | divergence at step 1, field `observation`, via `GymPair` |
| A space that skips one RNG draw | divergence at the RNG-state check, one step before any value differs |
| A frame with one pixel changed by 1 | `FrameDiff` failure, 1 pixel, delta 1 |
| An `info` with keys in a different order | `info` key-order divergence |
| A reward computed in float32 | reward-bits divergence with the ulp distance |
| A leaked `PyObject` | leak-gate failure |

## 12. Tolerances and exclusions

**Byte-exact is the only default; there is no tolerance fallback** (same stance as NumSharp's LAPACK tier). Exclusions are explicit, listed in a `ParityExclusions` registry (the analogue of `MisalignedRegistry`), each with a reason and an owner:
- wall-clock `info` values;
- cells known to be platform-dependent (libm transcendentals and the `float128` path in `rescale_box` off Windows) go **Inconclusive off the reference platform**, never red, and stay strict on it;
- MuJoCo frames (GPU and driver dependent) are compared on dynamics only unless offscreen rendering is pinned.

## 13. Costs and limits

- One interpreter per test process and `[DoNotParallelize]`: the live suite is slower than unit tests. Keep cases short (hundreds of steps); long episodes belong to `Replay`.
- Each paired step costs a few Python calls. Batch comparisons per step, not per field.
- CPython can't be re-initialized: never call `PythonEngine.Shutdown()` between tests. Shut down once at assembly cleanup, with `NoopFormatter` set.
- pythonnet 3.0.5 caps Python at 3.13. Moving the reference to 3.14 requires pythonnet 3.1.0 across the harness and NumSharp's interop.

## 14. Status and continuation

**Verified on 2026-09-23** (this session):
- NumSharp's interop design, test base, pitfalls and CI conventions were read in `refs/NumSharp` (§2).
- The reference venv installs on Windows 11 x64 / Python 3.12.12 with numpy 2.4.2, pygame-ce 2.5.8 and Box2D 2.3.10. `refs/Gymnasium` imports from `sys.path`, and CartPole trajectories and `rgb_array` frames were recorded from it.
- NumSharp master's `default_rng` is bit-identical to NumPy 2.4.2 for 16/17 of Gymnasium's draw patterns (all but `geometric`).
- The state-injection technique detects the CartPole float32-constant divergence at step 1, and a float64 transcription is bit-identical for 4 × 500 steps. These were run offline, through JSON, not yet through pythonnet.

**Not yet done:** booting pythonnet inside a Gym.NET test project. First steps for the next session:
1. Create `tests/Gym.Tests.Parity` with `pythonnet` `[3.0.5, 4.0.0)` and `NumSharp.Interop.pythonnet` (use a ProjectReference to `refs/NumSharp/src/NumSharp.Interop.pythonnet` until the Gym.NET NumSharp migration lands, port plan F1).
2. Port `PythonSession` + `InteropTestBase`. Point `PYTHONNET_PYDLL` at the base `python312.dll`, then prepend the venv's `site-packages` and `refs/Gymnasium` to `sys.path`. Confirm `import gymnasium` and the manifest check.
3. Implement `ByteContract` + `GymPair` for CartPole and reproduce, **live**, the offline result: the current `CartPoleEnv` diverges at step 1, and a float64 port matches for 500 steps.
4. Add the §11 self-tests before any further coverage.

**Open questions for the maintainer:**
- Whether the live suite should be required on PRs, or only on the reference job and nightly.
- Where golden files live (port plan D15).
- Whether to run a subset of Gymnasium's pytest suite through the reverse bridge (§8) as a required gate.
