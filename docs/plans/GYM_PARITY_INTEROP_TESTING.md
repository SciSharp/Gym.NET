# Gymnasium Parity via Live Interop Testing

| | |
|---|---|
| **Status** | Draft 2, 2026-09-25: the live harness is built and running as `tests/Gym.Tests.Parity` (see §14 for what's verified and what's next) |
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

Verified on 2026-09-25: `tools/parity/setup_venv.ps1` and `setup_venv.sh` (Git Bash) each create `tools/parity/.venv` from `tools/parity/requirements.txt` (the pins above, plus `cloudpickle` 3.1.2, `typing-extensions` 4.15.0 and `farama-notifications` 0.0.6), and the suite runs on it unconfigured: 103 of 103 gated cells passed with `python.source` reporting the venv. The reference interpreter itself (`~/.claude/python`, not a venv) carries the same stack, and the suite runs on it when no venv exists.

**Session manifest check.** At `[AssemblyInitialize]` the session reads `gymnasium.__file__`, the `refs/Gymnasium` HEAD SHA, `numpy.__version__`, `pygame.version.ver`, `Box2D.__version__`, `sys.version`, and the OS and CPU. A mismatch with `tools/parity/manifest.json` fails the run: a live oracle on the wrong stack proves nothing. `GYMNET_PARITY_ALLOW_STACK_DRIFT=1` downgrades the mismatch to Inconclusive for local exploration.

As built, the check also fails the run when `gymnasium` is imported from anywhere but `refs/Gymnasium`, when `git status` reports modified tracked files in the submodule (an edited oracle changes what "Gymnasium's answer" means; cleanliness is reported as unverified, not failed, when git is unavailable), and when `sys.flags.hash_randomization` is not 0. The report also records where Python and NumPy were loaded from and the platform.

## 4. Project layout

As built (2026-09-25); the project's own guide is `tests/Gym.Tests.Parity/CLAUDE.md`:

```
tests/Gym.Tests.Parity/                 live-interop test project (net8.0;net10.0; nullable, warnings as errors)
  AssemblyInfo.cs                       [assembly: DoNotParallelize]
  OpenBugsAttribute.cs                  NumSharp's [OpenBugs] category, with the port plan's defect ids (§12)
  Session/GymPythonSession.cs           engine start, venv + refs/Gymnasium on sys.path, manifest check
  Session/PythonDiscovery.cs            NumSharp's interpreter discovery, plus the parity venv
  Session/NativeEnvironment.cs          determinism variables through the C runtime (see below)
  Session/StackManifest.cs              tools/parity/manifest.json and the stack report
  Session/RepositoryLayout.cs           the checkout, the submodule's HEAD and its cleanliness
  Session/ParityTestBase.cs             InteropTestBase port: session check, leak gate, SkipUnless, SkipOffReferencePlatform
  Session/LeakGate.cs                   NumSharp's LiveExports/LiveImports gate, run after every test
  Interop/PythonRuntime.cs              the pythonic skill's backend: engine start/stop, Run (the GIL), cached Import
  Interop/Pythonic.cs                   the pythonic tree: static class Python, one facade row per fixed Python call
  Interop/PyRef.cs                      a Python object a test keeps across GIL sections, released under the GIL
  Oracle/ValueFacts.cs                  the side-neutral description of a value both halves reduce to (FromPython, Of)
  Oracle/PyConvert.cs                   CLR value -> Python value marshaling for facade arguments (None, ints, floats, arrays)
  Oracle/ByteContract.cs                the comparer: dtype, shape, bytes, float bits, ulps, key order, exceptions
  Oracle/Outcome.cs                     a call's value or exception; the Python -> .NET exception map (port plan §3.4)
  Oracle/PyGymEnv.cs                    a live gymnasium.make env driven through the tree; the balancing policy
  Oracle/RngFacts.cs, GeneratorProbe.cs PCG64 state in NumPy's bit_generator.state shape
  Pair/GymPair.cs                       lockstep differential runner (§5)
  Pair/Divergence.cs                    first-divergence report (ParityException)
  Pair/ParityAction.cs, ActionSources.cs  one action with identical bits on both sides; open-loop, policy, sampled
  Pair/IDotNetEnv.cs, GymNetEnv.cs      the .NET half; the adapter over Gym.NET's legacy env API
  Pair/Transition.cs                    a described reset/step result
  Frames/Frame.cs, FrameDiff.cs         pixel diff and PNG artifacts (§6)
  Frames/ParityArtifacts.cs             where artifacts go, attached to the test result
  Reference/CartPoleReference.cs        float64 transcription of cartpole.py: positive and negative controls (§11)
  Tests/Harness|Rng|Spaces|Envs/*.cs    the cells (§14)
tools/parity/requirements.txt, setup_venv.ps1, setup_venv.sh, manifest.json
```

Not built yet: `Bridge/` (§8) and `Cases/` (§9).

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

As built it adds `PYTHONDONTWRITEBYTECODE=1` (and sets `sys.dont_write_bytecode`), so the suite never writes `__pycache__` into the submodule, and `SDL_AUDIODRIVER=dummy`. These must be set in the **C runtime's** environment: `Environment.SetEnvironmentVariable` only changes the Win32 block, which CPython's `Py_Initialize` does not read (measured 2026-09-25: `PYTHONHASHSEED=0` stayed invisible and `sys.flags.hash_randomization` stayed 1). The session therefore also calls `_wputenv_s` (UCRT) on Windows and `setenv` (libc) elsewhere, and the stack check verifies the effect. No `.runsettings` is needed, so the suite behaves the same from `dotnet test`, an IDE or the test executable.

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

**As built (2026-09-25).** Constructing both sides from one id needs Gym.NET's registry (`Gymnasium.Make`, P1), so the pair takes the two envs instead: `new GymPair(PyGymEnv.Make("CartPole-v1"), new GymNetEnv(new CartPoleEnv(...)), stateAttribute: "state")`, with `Reset(seed)`, `Step(ParityAction)`, `Run(seed, actionSource, maxSteps, frameEvery)`, `InjectState(attribute, values)` and `CompareFrame(label)`. `PyGymEnv.Make` goes through `gymnasium.make` and steps its `unwrapped` env by default, the right counterpart while Gym.NET has no wrappers (`viaMake: true` keeps the stack). `GymNetEnv` bridges Gym.NET's legacy API without touching a value: `Seed(s)` + `Reset()` stands for `reset(seed=s)`, the missing reset info and a null step information are reported as `{}`, `done` as `terminated`, and `truncated` as `False`, which is what Gymnasium's unwrapped env returns. Every report prints those adaptations. Test-local envs implement `IDotNetEnv` directly; `Reference/CartPoleReference.cs` is the harness's positive control.

**What is compared, per call:**

| Field | Comparison |
|---|---|
| Observation | dtype + shape + C-order bytes (`ByteContract`). A NumPy scalar (`np.int64(3)`) equals a 0-d NumSharp array of the same dtype and bytes, because NumSharp has no scalar types; Python type names are reported, never compared |
| Reward | float64 bit pattern (a Python `float` or `np.float64`; `np.float32` rewards are widened exactly) |
| `terminated`, `truncated` | equality |
| `info` | same keys **in the same order**; values compared recursively (arrays by bytes, scalars by bits); a per-wrapper allowlist excludes wall-clock entries (`RecordEpisodeStatistics` `"t"`), to be added with the first wrapper that needs one |
| State attribute (as built, optional) | a float64 attribute both envs expose (`state` for classic control), compared exactly after every call, so dynamics are checked at full precision even where the float32 observation would round a difference away |
| RNG state | `env.unwrapped.np_random.bit_generator.state` (PCG64 `state`/`inc`) vs NumSharp's PCG64 state. This catches a missing or extra draw at the step it happens, even before any value differs. As built, only after the pair's own seeded reset: before one, both generators are entropy-seeded (and reading Gymnasium's `np_random` would create one), so the skip is noted in the report instead |
| Exceptions | same type (mapped through port plan §3.4) and same message |

**Action sources:**
- **Open-loop:** actions come from a fixed recorded list, or from Python's policy evaluated on Python's own observations. This isolates dynamics from the policy.
- **Sampled:** both sides call `action_space.seed(s)`, then `sample()` each step. The two samples must first match each other, which also tests space sampling.
- **Heuristic:** an env-provided controller such as `lunar_lander.heuristic`, evaluated on the Python observation and fed to both sides.

**State injection** is for unit-level dynamics checks. The pair writes the same state into both envs (Python `env.unwrapped.state = …`; .NET the equivalent public property), steps once, and compares. This isolates one step of dynamics from reset and RNG. In the 2026-09-23 scan this technique separated "wrong float32 constants" from "platform math" in CartPole (port plan Appendix A.7).

**Divergence report and shrinking** (`Divergence`): the report gives the first step, field, element index, both values (hex bits and decimal), the ulp distance, the action history, and a dump of both envs' full state at steps *k−1* and *k*, written to `GYMNET_PARITY_ARTIFACTS`. The shrinker replays from reset to find the shortest action prefix that still diverges, then tries state injection at *k−1* to confirm whether the fault lies in `step` alone.

As built, the report lists **every** differing field of the first diverging call (together they tell a missing cast from wrong values: CartPole's reset reports only `observation.dtype`, with "casting dotnet to <f4 would make them identical"), the action history, the pair's adaptations, and the state attribute before and after, in the exception message; frame artifacts go to `GYMNET_PARITY_ARTIFACTS`. The shrinker is not built: with open-loop and policy sources the first divergence already is the shortest prefix, so it waits for the fuzzer (§10).

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

| Control | Must report | As built (`Tests/Harness/HarnessSelfTests.cs`, all passing 2026-09-25) |
|---|---|---|
| CartPole with `tau = 0.02f` (the current Gym.NET bug) | divergence at step 1, field `observation`, via `GymPair` | `CartPoleReference.WithFloat32Tau()`: step 1, field `state`, 4 of 4 elements, no `rng`/`reward`/flag field. The float64 state, not the observation, is the field: a 1e-10 difference can round away in the float32 observation |
| A space that skips one RNG draw | divergence at the RNG-state check, one step before any value differs | `CartPoleReference.WithExtraDraw()` (one extra `random()` after the reset): reported at the reset, field `rng` only, the observation still identical |
| A frame with one pixel changed by 1 | `FrameDiff` failure, 1 pixel, delta 1 | on a real Gymnasium frame: 1 pixel, delta 1, bounds that pixel, histogram bucket 1 |
| An `info` with keys in a different order | `info` key-order divergence | `info.keys`: "same keys in a different order" |
| A reward computed in float32 | reward-bits divergence with the ulp distance | 0.1 against 0.1f: "107374182 ulp apart in float64" |
| A leaked `PyObject` | leak-gate failure | an export Python still holds keeps `LiveExports` above the baseline (0 → 1); `del` settles it |

The positive control is as important: the faithful `CartPoleReference` runs 500 steps in lockstep with nothing reported. Further self-tests pin the comparer's rules: a float32/float64 dtype difference with equal values is reported (with "casting would make them identical"); `np.int64(3)` equals a 0-d int64 NumSharp array but not an int32 one or a `(1,)` array; a Python `ValueError` matches `NumSharp.ValueError` with the same message and is reported against another type, another message, or a return; state injection without a reset compares no generators.

## 12. Tolerances and exclusions

**Byte-exact is the only default; there is no tolerance fallback** (same stance as NumSharp's LAPACK tier). Exclusions are explicit, listed in a `ParityExclusions` registry (the analogue of `MisalignedRegistry`), each with a reason and an owner:
- wall-clock `info` values;
- cells known to be platform-dependent (libm transcendentals and the `float128` path in `rescale_box` off Windows) go **Inconclusive off the reference platform**, never red, and stay strict on it;
- MuJoCo frames (GPU and driver dependent) are compared on dynamics only unless offscreen rendering is pinned.

**Known gaps are not exclusions.** A cell Gym.NET fails because a port has not landed yet asserts Gymnasium's answer and carries `[OpenBugs("B-xx", ...)]`, NumSharp's category for known failing reproductions, with the defect ids of the port plan's register (Appendix B) or its workstreams. Gated runs filter the category out (`--filter "TestCategory!=OpenBugs"`), a run of the category alone prints every gap's first divergence, and when the port lands the cell passes unchanged and the attribute is removed. NumSharp's triage rule applies: never excuse a real gap as an exclusion. As built, platform-dependent cells call `SkipOffReferencePlatform` (the reference is `referencePlatform` in the manifest, `win-x64`); `ParityExclusions` itself is not built, because no compared value needs one yet: it arrives with the first wrapper that reports wall-clock time.

## 13. Costs and limits

- One interpreter per test process and `[DoNotParallelize]`: the live suite is slower than unit tests. Keep cases short (hundreds of steps); long episodes belong to `Replay`.
- Each paired step costs a few Python calls. Batch comparisons per step, not per field.
- CPython can't be re-initialized: never call `PythonEngine.Shutdown()` between tests. Shut down once at assembly cleanup, with `NoopFormatter` set.
- pythonnet 3.0.5 caps Python at 3.13. Moving the reference to 3.14 requires pythonnet 3.1.0 across the harness and NumSharp's interop.
- Measured 2026-09-25 (Windows 11 x64): the 103 gated cells run in about 2 s; a 500-step CartPole lockstep with the state and generator compared after every call takes about 0.2 s; a 600 × 400 frame crosses in about 0.1 s (pythonnet's `As<byte[]>()`), fine for a few frames per test but not for a frame every step.

## 14. Status and continuation

**Verified on 2026-09-23** (this session):
- NumSharp's interop design, test base, pitfalls and CI conventions were read in `refs/NumSharp` (§2).
- The reference venv installs on Windows 11 x64 / Python 3.12.12 with numpy 2.4.2, pygame-ce 2.5.8 and Box2D 2.3.10. `refs/Gymnasium` imports from `sys.path`, and CartPole trajectories and `rgb_array` frames were recorded from it.
- NumSharp master's `default_rng` is bit-identical to NumPy 2.4.2 for 16/17 of Gymnasium's draw patterns (all but `geometric`).
- The state-injection technique detects the CartPole float32-constant divergence at step 1, and a float64 transcription is bit-identical for 4 × 500 steps. These were run offline, through JSON, not yet through pythonnet.

**Verified live on 2026-09-25** (`tests/Gym.Tests.Parity`, Gym.NET branch `feat/numsharp-master-migration` after the PCG64 seeding commit b51689d; Windows 11 x64, .NET SDK 10.0.101; the four first steps of Draft 1 are done):
- pythonnet 3.0.5 embeds CPython 3.12.12 in the test process; NumSharp.Interop.pythonnet is built from `refs/NumSharp` (`eng/NumSharp.props` swaps its declared package, as it does NumSharp's); `import gymnasium` resolves to `refs/Gymnasium` @ 9e04324, clean; the manifest check passes; hash randomization is off.
- **130 cells: 103 gated, all passing; 27 `[OpenBugs]`, each failing at its documented defect and nothing else**, in Debug and Release on net8.0 and net10.0. The existing `Gym.Tests` suite is unaffected (62 of 62 headless).
- **Byte-identical to Gymnasium today:** `Seeding.NpRandom` (reported seed and PCG64 state for 9 seeds up to `long.MaxValue` and for 2^100 + 7; the negative-seed `Error` and its message; an unseeded generator's 128-bit seed rebuilds it in NumPy); the 15 draw patterns of the port plan's A.6, value for value with the state after each; `Discrete` (samples, masked samples, membership, attributes); `Box` (bounds and their classification, `is_bounded`, samples in every bound category and dtype family, membership); CartPole's action space; LunarLander's discrete action space.
- **The live form of the offline results:** the float64 CartPole transcription is bit-identical for 4 seeds × 500 steps and through termination; Gym.NET's CartPole now matches Gymnasium's reset state and generator (F2) and differs only by the missing float32 cast (B-13: "casting dotnet to <f4 would make them identical"); from the same state its float32 constants move the state at step 1 (B-11); its frames differ in 2,573, 2,747 and 2,659 pixels at A.9's three states, exactly A.9's counts (B-16).
- **Found by the live suite:** Gym.NET's `Box(float low, float high, ...)` stores the bounds of a float64 box through `float`, so `Box(-0.1, 0.1, (2,), float64)` has `low` -0.10000000149011612 where Gymnasium has -0.1 (107374182 ulps; open, P2).

**Rebuilt on the pythonic skill on 2026-09-25** (NumSharp's and OptunaSharp's `pythonic`): the Python half is no longer
an embedded `gym_oracle.py` evaluated from expression strings, but a `static class Python` tree
(`tests/Gym.Tests.Parity/Interop/Pythonic.cs`) with one facade row per fixed call (`Python.gymnasium.make(...)`,
`Python.gymnasium.spaces.space.Space.sample(space)`, `Python.numpy.random.Generator.uniform(g, ...)`), an `evalStmts`
snippet runner for the two small programs that must read as Python (the session bootstrap, a strided NumPy probe), and a
`PythonRuntime` backend that owns the engine, the GIL entry point and the module cache. Values come back through
`ValueFacts.FromPython` (NumSharp's `ToNDArray` import for arrays, NumPy's own `dtype.str`/`int()` for scalars), and the
new session cell `ByteReaders_AgreeOnAnImportedView` pins that import against NumPy's own `tobytes()` for all twelve
dtype families on a strided, negative-step view. Same verdicts afterwards: 131 cells, 104 gated passing, the same 27
`[OpenBugs]` failing at the same divergences, net8.0 and net10.0.

**CartPole byte-for-byte on 2026-09-25.** Gym.NET's `CartPoleEnv` was re-ported from `cartpole.py` (float64 attributes as
settable properties, the source's operation order, float32 observation copies, `±inf` observation bounds,
`AssertionError`s with Python's messages, `sutton_barto_reward`, both integrators, `Reset(seed, options)` with
`maybe_parse_reset_bounds`, the post-termination warning through a new `Gym.Logger`), and renders with a new
`Gym.Rendering` project: a pygame-compatible `Surface` and a C# port of SDL_gfx 2.0.26's `gfxdraw` primitives (zlib),
plus `transform.flip`, `surfarray` and `Vector2.rotate_rad`. The live suite proves it: **160 cells, 146 gated, all
passing on net8.0 and net10.0**, among them 37 CartPole cells (whole episodes of the balancing policy and of random
actions, termination and past it, both reward schemes, the semi-implicit integrator, a changed `tau`, 17 reset-option
cases seeded and unseeded, 13 invalid actions and 3 pre-reset steps with their exact messages, the warning's text and
count, frames along episodes and at 20 extreme states, NaN/overflow errors included, metadata and spaces) and 11
rasterizer cells against pygame itself (150 random shapes per primitive with clipping, translucency and Sint16
wrap-around; `rotate_rad` bit for bit over 300 angles, NaN payloads included). The 14 open cells are spaces (P2, B-03,
B-10) and LunarLander (F6, B-17, B-19, B-20). Found on the way: .NET 8's `double %` returns its own sign-set NaN for an
infinite dividend where the C runtime's `fmod` keeps the infinity's sign, so `rotate_rad(inf)` differed in one bit on
net8.0 only; `Vector2.RotateRad` now reproduces `fmod`'s NaNs itself.

**Next steps:**
1. Wire the suite into CI (port plan §14.8): a Windows reference job that runs `tools/parity/setup_venv.ps1`, then `dotnet test tests/Gym.Tests.Parity --filter "TestCategory!=OpenBugs"` with `GYMNET_PYTHONNET_REQUIRE_ENGINE=1`.
2. The reverse bridge (§8): `PyEnvShim.py` over `IDotNetEnv`, then `check_env` and `check_environments_match` on Gym.NET's envs.
3. One case catalog for live, record and replay (§9), with the port plan's golden harness (F4).
4. As each port lands (P1 core API, P2 spaces, P4 CartPole, P7 rasterizer, P8 LunarLander), remove its `[OpenBugs]` and add the cells its new surface needs (`Gymnasium.Make`, wrappers, vector envs, the registry, §7).

**Open questions for the maintainer:**
- Whether the live suite should be required on PRs, or only on the reference job and nightly.
- Where golden files live (port plan D15).
- Whether to run a subset of Gymnasium's pytest suite through the reverse bridge (§8) as a required gate.
