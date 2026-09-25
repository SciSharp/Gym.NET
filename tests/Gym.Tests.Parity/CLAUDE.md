# Gym.Tests.Parity: Gym.NET against live Gymnasium, byte for byte

This project runs **real CPython, NumPy and Gymnasium inside the test process** (through pythonnet) and compares Gym.NET
with them bit for bit and pixel for pixel. Gymnasium is imported from the `refs/Gymnasium` submodule, so the oracle is
exactly the pinned commit, and it computes its answer **now**, for the same seed and the same actions, with nothing
serialized in between.

It is the live half of the parity strategy in [docs/plans/GYM_PARITY_INTEROP_TESTING.md](../../docs/plans/GYM_PARITY_INTEROP_TESTING.md),
modelled on NumSharp's `test/NumSharp.Tests.Interop` (whose `CLAUDE.md` it follows closely).

> **The claim a passing cell makes:** for the same inputs, seed and actions, Gym.NET produced the same dtypes, shapes
> and bytes as Gymnasium (arrays), the same IEEE-754 bits (floats), the same values (integers, booleans, strings), the
> same info keys in the same order, the same exceptions (type mapped per port plan §3.4, message verbatim), the same
> PCG64 generator state after the call, and the same `rgb_array` pixels.

---

## 1. Running

Prerequisites: the reference stack of [`tools/parity/manifest.json`](../../tools/parity/manifest.json) (Python 3.12,
NumPy 2.4.2, pygame-ce 2.5.8 with SDL 2.32.10, Box2D 2.3.10) and the submodules
(`git submodule update --init refs/Gymnasium refs/NumSharp`, not `--recursive`). Either install the stack into an
interpreter on `PATH`, or create the venv the suite finds on its own:

```bash
pwsh tools/parity/setup_venv.ps1          # Windows; creates tools/parity/.venv (gitignored)
bash tools/parity/setup_venv.sh           # Linux / macOS (needs a shared libpython3.12)
```

```bash
export AVALONIA_TELEMETRY_OPTOUT=1        # the solution's Avalonia projects ping telemetry otherwise

# The gate: every cell that must pass (what CI runs)
dotnet test tests/Gym.Tests.Parity/Gym.Tests.Parity.csproj --filter "TestCategory!=OpenBugs"

# The open gaps: every cell Gym.NET does not pass yet, each printing its first divergence
dotnet test tests/Gym.Tests.Parity/Gym.Tests.Parity.csproj --filter "TestCategory=OpenBugs" --logger "console;verbosity=detailed"

# One area
dotnet test tests/Gym.Tests.Parity/Gym.Tests.Parity.csproj -f net10.0 --filter "FullyQualifiedName~CartPoleLiveParityTests"
```

A plain `dotnet test` (also `dotnet test Gym.NET.sln`) runs the open gaps too, and they fail: that is NumSharp's
convention for `[OpenBugs]`, and the reason gated runs pass the filter. Do **not** move the filter into a `.runsettings`
file: a runsettings `TestCaseFilter` is combined with the command line's `--filter`, so
`--filter "TestCategory=OpenBugs"` would then silently run nothing (measured).

| Variable | Default | Effect |
|---|---|---|
| `PYTHONNET_PYDLL` | discovered | The Python shared library to embed (the base `python312.dll`, also with a venv) |
| `GYMNET_PARITY_VENV` | `tools/parity/.venv` when it exists | The venv whose interpreter is probed for `sys.path`; a broken configured venv fails, it never falls back |
| `GYMNET_PYTHONNET_REQUIRE_ENGINE` | `1` | `0`: no Python means Inconclusive instead of failure |
| `GYMNET_PARITY_ALLOW_STACK_DRIFT` | `0` | `1`: a stack that differs from the manifest no longer fails every test; only `Stack_MatchesTheManifest` reports Inconclusive |
| `GYMNET_PARITY_ARTIFACTS` | the test results directory | Where failed frame comparisons write `python.png`, `dotnet.png`, `diff.png`, `stats.txt` |

The first thing every run prints (test `Stack_MatchesTheManifest`) is the stack it ran on, including the Gymnasium commit,
where `gymnasium` and `numpy` were imported from, and whether `refs/Gymnasium` has local modifications.

---

## 2. The moving parts

| Piece | File | Role |
|---|---|---|
| `GymPythonSession` | `Session/GymPythonSession.cs` | One embedded CPython per process (`[AssemblyInitialize]`/`[AssemblyCleanup]`; CPython cannot re-initialize). Discovers Python (`Session/PythonDiscovery.cs`), fixes the process state through the C runtime (`Session/NativeEnvironment.cs`), runs the bootstrap snippet (`refs/Gymnasium` first on `sys.path`), reads the live stack through the tree and checks it against the manifest (`Session/StackManifest.cs`). |
| `PythonRuntime` | `Interop/PythonRuntime.cs` | The pythonic skill's backend: `Initialize`/`Shutdown` (once per process), `Run` (every Python-touching body holds the GIL through it), cached `Import` (never dispose a module handle), `Builtins`, `PythonInteropException`. |
| `Python` | `Interop/Pythonic.cs` | **The pythonic tree**: `Python.gymnasium.make(...)`, `Python.gymnasium.Env.step(env, a)`, `Python.gymnasium.spaces.Discrete(4, seed: 0)`, `Python.gymnasium.spaces.space.Space.sample(space)`, `Python.numpy.random.Generator.uniform(g, ...)`, `Python.pygame.gfxdraw.aapolygon(...)`. One row per fixed Python call; object members are nested classes taking the object; `evalStmts` runs a multi-statement snippet. |
| `ParityTestBase` | `Session/ParityTestBase.cs` | Base class of every cell: requires the session, runs NumSharp's **leak gate** (`Session/LeakGate.cs`) after every test, `SkipUnless`, `SkipOffReferencePlatform`. |
| `ValueFacts` | `Oracle/ValueFacts.cs` | The side-neutral description both halves reduce to: from Python via `FromPython` (arrays through NumSharp's `ToNDArray` import, scalars through NumPy's own `dtype.str`/`int()`, dtypes as their `dtype.str`), from .NET via `ValueFacts.Of` (NDArray bytes through NumSharp's own `data.tobytes`, never through the bridge). `SessionTests.ByteReaders_*` pin both directions against NumPy's own `tobytes()`. |
| `ByteContract` | `Oracle/ByteContract.cs` | The comparer: every difference with both values, their bits, the ulp distance, the element index; exceptions by mapped type (`Oracle/Outcome.cs`, `ExceptionMap`) and message. |
| `Outcome.CapturePython`, `PyRef`, `PyConvert` | `Oracle/Outcome.cs`, `Interop/PyRef.cs`, `Oracle/PyConvert.cs` | Run a tree call under the GIL and describe its result or its exception; hold a Python object across calls (released under the GIL); convert CLR arguments (`None`, ints of any width, floats, strings, `NDArray` as `ToNumpyCopy`, dicts, lists, tuples). |
| `PyGymEnv` | `Oracle/PyGymEnv.cs` | A live Gymnasium env (`gymnasium.make(...)`, optionally `.unwrapped`), driven through the tree; the `cartpole_balance` policy (Python's float64 arithmetic) and Gymnasium's own `lunar_lander.heuristic`. |
| `GymPair` | `Pair/GymPair.cs` | The lockstep runner (plan §5): same seed, same actions (`Pair/ParityAction.cs`), every call verified in a fixed field order, first divergence thrown as `ParityException` (`Pair/Divergence.cs`). |
| `IDotNetEnv`, `GymNetEnv` | `Pair/IDotNetEnv.cs`, `Pair/GymNetEnv.cs` | The .NET half. `GymNetEnv` drives any Gym.NET env, bare or wrapped, through `IEnv`, Gymnasium's own contract (5-tuple, float64 reward, `(obs, info)` resets), so nothing is adapted; attributes go through the unwrapped env by Python name (`PyAttributes`). |
| Action sources | `Pair/ActionSources.cs` | Open-loop, a Python policy on Gymnasium's own observations, or seeded sampling on both sides (samples compared before each step). |
| `RngFacts`, `GeneratorProbe` | `Oracle/RngFacts.cs`, `Oracle/GeneratorProbe.cs` | A NumSharp `Generator`'s PCG64 state in NumPy's `bit_generator.state` shape; finds the generator of a Gym.NET env or space. |
| `FrameDiff` | `Frames/FrameDiff.cs` | Pixel parity (plan §6): differing pixels, bounding box, max channel delta, delta histogram, and PNG artifacts. |
| `CartPoleReference` | `Reference/CartPoleReference.cs` | A test-local float64 transcription of `cartpole.py`: the harness's positive control, and with a knob its negative controls. Not product code. |
| `[OpenBugs]` | `OpenBugsAttribute.cs` | NumSharp's category for known gaps, with the port plan's defect ids attached. |

---

## 3. Recipes

### A direct cell (a space, a function)

```csharp
[TestMethod]
public void Sample_MatchesGymnasium() {
    using PyRef python = PyRef.Of(() => Python.gymnasium.spaces.Discrete(4, seed: 0));   // live Gymnasium object
    var dotnet = new Discrete(4, seed: 0);                                                // Gym.NET object
    for (int i = 1; i <= 10; i++)
        ByteContract.AssertSame($"Discrete(4, seed=0).sample() #{i}",
            Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.sample(python.Object)),  // Gymnasium, described
            Outcome.Capture(() => dotnet.Sample()));                                                // Gym.NET, described, disposed
    ByteContract.AssertSame("generator state",
        RngFacts.OfPythonSpace(python.Object),               // space.np_random.bit_generator.state
        Outcome.Returned(GeneratorProbe.StateOf(dotnet)));
}
```

A Python argument is built through the tree (`Python.numpy.int64(6)`), converted by `PyConvert.ToPython` (plain CLR
values), or crossed from the very NumSharp array the .NET side receives (`point.ToNumpyCopy()`), so both sides get
identical bits. A call the tree lacks gets a row in `Interop/Pythonic.cs` (Python spelling, XML docs, the caller owns
the returned `PyObject`), not a string: only a multi-statement program goes through `Python.evalStmts`.

### A lockstep cell (an env)

```csharp
using var pair = new GymPair(PyGymEnv.Make("CartPole-v1"),               // gymnasium.make(...).unwrapped
                             new GymNetEnv(new CartPoleEnv()),               // any IEnv, bare or wrapped
                             stateAttribute: "state");                     // also compare the float64 state
PairRun run = pair.Run(seed, ActionSources.Policy("cartpole_balance"), maxSteps: 500, frameEvery: 0);
```

State injection (one step of dynamics, isolated from reset and the generator): `pair.InjectState("state", 0.01, -0.02,
0.03, 0.04)` then `pair.Step(ParityAction.Discrete(1))`. Frames: `PyGymEnv.Make(..., renderMode: "rgb_array")`, a Gym.NET env
built with `renderMode: "rgb_array"` (the render mode is fixed at construction, as in Gymnasium), and
`pair.CompareFrame("label")`. A wrapper stack: `PyGymEnv.Make(id, viaMake: true)` keeps `make`'s stack, compared with
Gym.NET's `TimeLimit(OrderEnforcing(env))` until Gym.NET's own `make` (P3).

### Errors

`Outcome` carries exceptions as data, so an error is a compared outcome, not a test failure (`ZeroDivisionError` maps to
`System.DivideByZeroException`, as `OverflowError` maps to `System.OverflowException`):
`Outcome.CapturePython(() => Python.gymnasium.utils.seeding.np_random(seed))` (with `seed` a `PyInt` of -1) against
`Outcome.Capture(() => Seeding.NpRandom(-1))` passes when Gym.NET throws `Gym.Exceptions.Error` with Gymnasium's message. `ExceptionMap` holds the port plan's §3.4 mapping
(`ValueError` → `NumSharp.ValueError`, `gymnasium.error.X` → `Gym.Exceptions.X`, `AssertionError` →
`Gym.Exceptions.AssertionError`, `OverflowError` → `System.OverflowException`, the type NumSharp throws for NumPy's).

---

## 4. Rules

- **Byte-exact is the only default**; there is no tolerance (plan §12). The only equivalences are the ones NumSharp's
  model forces, all in `ByteContract`: a NumPy scalar equals a 0-d NumSharp array of the same dtype (NumSharp has no
  scalar types), and a Python `int` equals a .NET integer of any width with the same value (Python ints have no width).
  Rewards compare as float64 bits (`FieldPolicy.Float64Value`, plan §5), so a float32 reward still shows its ulp distance.
- **The expectation is always Gymnasium's**, never today's Gym.NET output. A cell Gym.NET fails today gets
  `[OpenBugs("B-xx", ...)]` with the ids from the port plan's defect register (Appendix B) or its workstreams (`F6`,
  `P2`). Lifecycle, as in NumSharp: the port lands, the cell passes, **remove the attribute** so it becomes a gate.
  A deliberate, documented deviation is not an open bug.
- **Split a cell where only part diverges**, so the part that matches stays gated (`Contains_MatchesGymnasium` gated,
  `Contains_RequiresCastableDtypes_LikeGymnasium` open).
- **Pin libm-dependent cells to the reference platform** with `SkipOffReferencePlatform(...)` (CartPole's `cos`/`sin`):
  strict on `win-x64`, Inconclusive elsewhere (port plan §14.9).
- **The pythonic skill** (`refs/NumSharp/.claude/skills/pythonic`): Python is reached through the `Python.*` tree only;
  each fixed call is a facade row, an object's members a nested class taking the object, a multi-statement program an
  `evalStmts` snippet. Runtime-dispatched calls (type reflection, `dict.items()` walks) stay in `ValueFacts.FromPython`.
  Inside `Gym.Tests.Parity` the class `Python` shadows the `Python.Runtime` namespace: reach pythonnet types through a
  file-level `using Python.Runtime;`, and never name a member `Python` (it would shadow the tree in its class).
- **GIL discipline** (NumSharp's): every `PyObject` is created and disposed inside `PythonRuntime.Run` (or
  `Outcome.CapturePython`, `PyRef`, which take it themselves); tests keep NumSharp arrays and CLR values outside it. Spell
  chains with `using` locals: an inline `a(b(x))` leaks `b(x)`. Never dispose `PyObject.None` (pythonnet shares it
  process-wide; `PyConvert.ToPython(null)` gives an owned `None`) nor a module from `PythonRuntime.Import`.
- **Dispose every NDArray.** The project builds with warnings as errors, so NumSharp's ownership analyzer (NDW012) fails
  the build on an array that is created and dropped.
- **Test the harness before trusting it.** `Tests/Harness/HarnessSelfTests.cs` plants known defects and checks each is
  reported at the exact call and field (plan §11). A new comparison rule gets a self-test.

---

## 5. What the suite covers (2026-09-25)

| Area | Gated (must pass) | Open (`[OpenBugs]`) |
|---|---|---|
| Session | stack = manifest; Gymnasium from the submodule; deterministic interpreter; both byte readers agree with NumPy's `tobytes()` (export: a strided NumSharp view; import: strided NumPy views of 12 dtypes) | |
| Harness self-tests (§11) | faithful transcription passes 500 steps; `tau = 0.02f` at step 1 in the state; an extra draw at the reset through the generator alone; injected state without unseeded generators; one pixel off by one; info key order; float32 reward ulps; the leak gate; dtype width; the scalar rule; exception mapping | |
| Seeding | `Seeding.NpRandom` = `seeding.np_random` for 9 seeds (0 … `long.MaxValue`) and 2^100 + 7: reported seed and PCG64 state; 15 A.6 draw patterns value for value with the state after each; negative seeds raise `Error` with Gymnasium's message; the unseeded 128-bit seed rebuilds the generator in NumPy | |
| `Discrete` | samples (7 cases, `n` up to 2^40, negative and positive `start`) and generator state; masked samples; all-masked returns `start` without a draw; membership (15 candidates); `n`/`start`/dtype; `repr` of a plain space | `can_cast` in `contains` (P2); shape `()` (B-03); `repr` with `start` (P2); `n <= 0` validation (B-10) |
| `Box` | bounds, their classification, dtype, shape, `is_bounded` (8 boxes: every bound category, float/int/uint dtypes); 5 samples and generator state for each × 4 seeds; membership | `can_cast` in `contains`, a list as a point, double-precision bounds of a float64 box (P2) |
| CartPole-v1 | reference transcription: 4 seeds × 500 steps, and through termination. **The port, byte for byte:** reset; 4 × 500-step episodes of the balancing policy and 5 random-action episodes, each past termination; `sutton_barto_reward`; semi-implicit Euler; `tau = 0.05`; 51 steps from an injected state; 17 reset-option cases (seeded and unseeded, `ValueError`/`OverflowError` messages, state and generator after a failed reset); 13 invalid actions and 3 pre-reset steps (`AssertionError` messages); the post-termination warning (text, category, count); frames at A.9's states, along 4 episodes, at 20 extreme states (clipped, Sint16-wrapped, quarter-turn snapped, and pygame's NaN/overflow errors) and before reset (`None`); observation and action spaces; metadata | |
| Rendering (`Gym.Rendering` vs pygame-ce 2.5.8) | `aapolygon`, `filled_polygon`, `aacircle`, `filled_circle` (150 random shapes each, clipping and translucency), `hline`/`vline`, compositions, Sint16 wrap-around polygons, CartPole's shapes, `transform.flip`, `Vector2.rotate_rad` bit for bit (NaN payloads included), pygame's argument errors | |
| LunarLander-v3 | Gym.NET discrete action space | continuous action space (B-20); observation space (B-19); heuristic episodes, both variants (F6, P8, B-17, B-19) |
| Core API (P1) | `make("CartPole-v1")`'s stack against `TimeLimit(OrderEnforcing(CartPoleEnv))` through truncation at step 500 (2 seeds); `OrderEnforcing` and `TimeLimit` call by call (`ResetNeeded`/`TypeError`/`ValueError` messages, `has_reset`, `_elapsed_steps`, truncation past the limit, `str`); the attribute protocol through a stack (plain `getattr`/`setattr`, `has/get/set_wrapper_attr`, run-time attributes, read-only properties, a wrapper's `_np_random`); `np_random`/`np_random_seed` (lazy, seeded, -1 once assigned, through a wrapper); `render()` without a render mode; `EnvSpec` repr and `pprint`; `parse_env_id`/`get_env_id`; the single-env step-API conversions; dict repr | |
| `PyMath` (F3) | `//`, `%`, `divmod` on ints and floats, `min`/`max`, `round` with and without digits, `np.clip` (512 triples), `np.sign`, bit for bit against CPython 3.12 and NumPy 2.4.2 | |

Not built yet (plan §8–§10): the reverse bridge (`check_env` on a Gym.NET env through a Python shim), the golden
record/replay modes sharing these cases, differential fuzzing, and the remaining wrapper, vector and registry areas, which
wait for their ports (P3–P6).

---

## 6. Pitfalls this project has already paid for

- **`Environment.SetEnvironmentVariable` does not reach CPython.** It changes the Win32 environment block; CPython's
  `Py_Initialize` reads the C runtime's copy (measured: `PYTHONHASHSEED=0` stayed invisible, `hash_randomization` 1).
  `NativeEnvironment` also calls `_wputenv_s` (UCRT) or `setenv` (libc), and the stack check verifies it took effect.
- **Reading Gymnasium's `np_random` creates a generator** when none exists. `GymPair` compares generator states only
  after a seeded reset; without one it notes the skip instead of reporting two unrelated entropy-seeded generators.
- **NumSharp's size-less `Generator.choice(array)` returns a one-element array** where NumPy returns a scalar; the
  draw-pattern cell compares it through `np.atleast_1d`, and Gym.NET's `Discrete` reads it through `SingleValue`.
- **MSTest 4.4.1's `TestContext.Current` is experimental** (MSTESTEXP, an error under warnings-as-errors);
  `ParityTestBase.CurrentContext` replaces it (tests run one at a time, `[DoNotParallelize]`).
- **`dotnet sln add` (SDK 10) adds referenced projects and x64/x86 platforms** to the solution; add a project to
  `Gym.NET.sln` by hand.
- **.NET 8's `double %` is not C's `fmod` for non-finite dividends**: it returns its own sign-set NaN for `±inf` and NaN,
  where the C runtime keeps the infinity's sign or the NaN operand. Measured through `Vector2.rotate_rad(inf)` differing
  in one bit on net8.0 only; code that must match C (`Gym.Rendering.Vector2`) handles non-finite values itself.
- **`Python.numpy.ndarray.tobytes_t` (pythonnet's `As<byte[]>()`) is slow for large buffers**: about 84 ms for a 600 × 400 × 3 frame (measured once, JIT
  warm-up included). Fine for a handful of frames per test; for a frame every step of long trajectories, read the bytes
  through NumSharp's buffer import instead.
