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
NumPy 2.4.2, pygame-ce 2.5.8 with SDL 2.32.10, Box2D 2.3.10, OpenCV 4.13.0) and the submodules
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

# The goldens, replayed without Python (what the Linux CI job runs; works with no stack and no refs/Gymnasium)
dotnet test tests/Gym.Tests.Parity/Gym.Tests.Parity.csproj --filter "TestCategory=Golden"

# Record (or re-record) the goldens from the live cells that verify them; unchanged goldens are left untouched
GYMNET_PARITY_RECORD=1 dotnet test tests/Gym.Tests.Parity/Gym.Tests.Parity.csproj -f net10.0 --filter "FullyQualifiedName~GoldenLiveTests"
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
| `GYMNET_PARITY_RECORD` | unset | `1`: the live golden cells (`GoldenLiveTests`) write what they verified into `tests/Golden` |

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
| Golden catalog | `Golden/GoldenCatalog.cs`, `Golden/GoldenCase.cs` | Plan §9's one case catalog (DR-04): each `GoldenCase` (id, kwargs, seed, actions, steps, frames, state attribute, the .NET factory) drives a live cell (`GoldenLiveTests`), its recording, and its Python-free replay (`GoldenReplayTests`, category `Golden`). |
| Recording and replay | `Golden/GoldenRecorder.cs`, `Golden/GoldenAssert.cs`, `Golden/GoldenStore.cs`, `Golden/GoldenTrajectory.cs`, `Golden/GoldenCodec.cs` | `GymPair.Recorder` receives each call and frame **after** it was verified; `GoldenStore` writes `tests/Golden/<suite>/<case>.jsonl.gz` and PNG frames only when their content changed, atomically; `GoldenAssert.Replay` drives the .NET env through the recording (recorded actions, or re-sampled and checked for a sampled case) with the same `ByteContract`, and refuses a stale golden (other case parameters, or another Gymnasium commit than the manifest pins). Format: `tests/Golden/README.md`. |
| `[OpenBugs]` | `OpenBugsAttribute.cs` | NumSharp's category for known gaps, with the port plan's defect ids attached. |
| `ReverseBridge`, `BridgedEnv` | `Bridge/ReverseBridge.cs`, `Bridge/BridgedEnv.cs` | The reverse bridge (plan §8, P3.5): Gym.NET envs as Gymnasium envs, so Gymnasium's own tools run on them. A `GymNetShim(gymnasium.Env)` snippet class, subclassed per .NET env class and named after it (`CartPoleEnv`), delegates every call to a `BridgedEnv`; `ReverseBridge.Spec` mirrors a Gym.NET spec with that class as its entry point, so `gymnasium.make(spec)` builds Gymnasium's stack over a Gym.NET env. Spaces mirror as the Gymnasium spaces they equal (`Box`, `Discrete`, `Tuple`, `Dict`); observations cross as zero-copy views (`PyConvert.ToPythonShared`); the PCG64 state is mirrored into a NumPy generator after every call; Gym.NET's exceptions are raised as the Python types they stand for (`ExceptionMap.PythonTypeFor`) and its warnings warned in Python (`BridgeCalls`). The bridge owns every .NET env built through it. |

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
Gym.NET's `Gymnasium.Make(id)` (`new GymNetEnv(Gymnasium.Make("CartPole-v1"))`), `PassiveEnvChecker` included.

### A golden case (a trajectory that also replays without Python)

Add a `GoldenCase` to `Golden/GoldenCatalog.cs`; the live cell, its recording and its replay come with it:

```csharp
new GoldenCase {
    Suite = "classic_control", Name = "CartPole-v1.policy.seed0", EnvId = "CartPole-v1", Seed = 0,
    Actions = new GoldenActions.Policy("cartpole_balance"),    // or Sampled(seed), Constant(action)
    MaxSteps = 500, FrameEvery = 100, StateAttribute = "state",
    PlatformDependence = "CartPole's cos/sin",                 // strict on the reference platform only
    CreateDotNet = renderMode => new GymNetEnv(new CartPoleEnv(renderMode: renderMode)),
},
```

A case whose id differs from another only in its registration (CartPole-v0, CartPole-v1's class with a 200-step
limit) sets `ViaMake = true` and builds its .NET side with `Gymnasium.Make(id, ...)`: both sides then run `make`'s
stack, and the golden records the truncation instead of the unwrapped dynamics another case already holds.

Then record it with `GYMNET_PARITY_RECORD=1` (§1) and commit `tests/Golden/<suite>/<name>.jsonl.gz` with its frame
directory. Changing a case's parameters, or re-pinning Gymnasium, makes its golden stale: the replay fails with the
re-record command instead of comparing against another trajectory.

### A Gymnasium tool on a Gym.NET env (the reverse bridge)

```csharp
using var bridge = new ReverseBridge();                                    // owns every .NET env built through it
using PyRef shim = PyRef.Of(() => {
    using PyObject spec = bridge.Spec(Gymnasium.Spec("CartPole-v1"));      // a Gymnasium EnvSpec over the Gym.NET env
    return Python.gymnasium.make(spec, render_mode: "rgb_array");         // Gymnasium's own stack over it
});
Outcome verdict = Outcome.CapturePython(() =>
    Python.gymnasium.utils.env_match.check_environments_match(python.Object, shim.Object, 200, seed));
```

`bridge.Shim(typeof(CartPoleEnv), kwargs)` builds a bare one (no spec, no wrappers); a `create` delegate plants a defect
(`env.Tau = 0.03`). Hold the tool's verdict on the bridged env to its verdict on Gymnasium's own env, and require that
reference to be the expected one first (two equal failures pass any comparison). Both outcomes are Python's, so wrap
the bridged one in `ReverseBridge.AsBridgedSide` (its exception type in .NET spelling, as `ByteContract` expects on the right). Human
mode needs `Gymnasium.HumanViewerFactory = NullEnvViewer.Factory` for the cell.

### Errors

`Outcome` carries exceptions as data, so an error is a compared outcome, not a test failure (`ZeroDivisionError` maps to
`System.DivideByZeroException`, as `OverflowError` maps to `System.OverflowException`):
`Outcome.CapturePython(() => Python.gymnasium.utils.seeding.np_random(seed))` (with `seed` a `PyInt` of -1) against
`Outcome.Capture(() => Seeding.NpRandom(-1))` passes when Gym.NET throws `Gym.Exceptions.Error` with Gymnasium's message. `ExceptionMap` holds the port plan's §3.4 mapping
(`ValueError` → `NumSharp.ValueError`, `gymnasium.error.X` → `Gym.Exceptions.X`, `AssertionError` →
`NumSharp.AssertionError` since NumSharp 0.80.0, `OverflowError` → `System.OverflowException`, the type NumSharp throws
for NumPy's). The built-ins NumSharp still lacks (`NotImplementedError`, `ImportError`: open item N-3) map to
`Gym.Exceptions` until NumSharp adds them (decisions DR-01, DR-02).

---

## 4. Rules

- **Byte-exact is the only default**; there is no tolerance (plan §12). The only equivalences are the ones NumSharp's
  model forces, all in `ByteContract`: a NumPy scalar equals a 0-d NumSharp array of the same dtype (NumSharp has no
  scalar types), and a Python `int` equals a .NET integer of any width with the same value (Python ints have no width).
  Rewards compare as float64 bits (`FieldPolicy.Float64Value`, plan §5), so a float32 reward still shows its ulp distance.
- **The expectation is always Gymnasium's**, never today's Gym.NET output. A cell Gym.NET fails today gets
  `[OpenBugs("B-xx", ...)]` with the ids from the port plan's defect register (Appendix B) or its workstreams (`F6`,
  `P2`); a cell that fails because of a NumSharp gap gets that gap's N-item from docs/plans/COMPLETION_PLAN.md §7
  (`[OpenBugs("N-1")]`, decision DR-01). Lifecycle, as in NumSharp: the port or the NumSharp release lands, the cell
  passes, **remove the attribute** so it becomes a gate. A deliberate, documented deviation is not an open bug.
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

## 5. What the suite covers (2026-11-10)

| Area | Gated (must pass) | Open (`[OpenBugs]`) |
|---|---|---|
| Session | stack = manifest; Gymnasium from the submodule; deterministic interpreter; both byte readers agree with NumPy's `tobytes()` (export: a strided NumSharp view; import: strided NumPy views of 12 dtypes) | |
| Harness self-tests (§11) | faithful transcription passes 500 steps; `tau = 0.02f` at step 1 in the state; an extra draw at the reset through the generator alone; injected state without unseeded generators; one pixel off by one; info key order; float32 reward ulps; the leak gate; dtype width; the scalar rule; exception mapping; NumSharp's argument exceptions compared by NumPy's text (a live `AxisError`) | |
| Seeding | `Seeding.NpRandom` = `seeding.np_random` for 9 seeds (0 … `long.MaxValue`) and 2^100 + 7: reported seed and PCG64 state; 15 A.6 draw patterns value for value with the state after each; negative seeds raise `Error` with Gymnasium's message; the unseeded 128-bit seed rebuilds the generator in NumPy | |
| `Discrete` | samples (7 cases, `n` up to 2^40, negative and positive `start`) and generator state; masked samples; all-masked returns `start` without a draw; membership (15 candidates); `n`/`start`/dtype; `repr`, with and without `start`; `can_cast` in `contains`, shape `()` (B-03) and `n <= 0` validation (B-10), gated since P2.3 | |
| `Box` | bounds, their classification, dtype, shape, `is_bounded` (8 boxes: every bound category, float/int/uint dtypes); 5 samples and generator state for each × 4 seeds; membership; `can_cast` in `contains`, a list as a point and the double-precision bounds of a float64 box, gated since P2.2 | |
| `MultiBinary`, `MultiDiscrete`, `Tuple`, `Dict`, `Text`, `Sequence`, `Graph`, `OneOf` (`Tests/Spaces/SpaceSpec*.cs`, `SpaceDetailsLiveParityTests`) | a catalog of 84 spaces (this suite's 24, each composite alone, nested and over the simple spaces, plus Gymnasium's own test catalog `tests/spaces/utils.py`, `GymnasiumTestingSpaces`), each: `repr`, 5 seeded samples × 2 seeds with the generator state of the space and of every subspace (`_np_random`, never created by the check), `seed(int)`'s return and stream, own samples as members, `is_np_flattenable`, JSON round trip, `==`, `batch_space`'s `repr`, samples and state; 22 mask/probability cases, 17 refused samples, 26 seed forms (lists, nested lists, dicts, refused forms, `np.int64`, `True`), membership matrices of 8 spaces, `Graph.sample` with node/edge counts and masks, `Graph.contains` of 9 graphs, constructor errors, `MultiDiscrete` indexing | |
| `gymnasium.vector.utils` (`VectorUtilsLiveParityTests`) | `create_empty_array`, `concatenate` into it, `list(iterate(...))` over a batched sample, for the catalog's spaces; `iterate`'s refusals; `batch_differing_spaces` for 7 space types and 6 refused lists; `create_shared_memory`, three seeded samples written with `write_to_shared_memory` and `read_from_shared_memory` over the catalog (views, decoded texts, `OneOf` pairs, the dynamic-shape `TypeError`) | |
| `gymnasium.spaces.utils` (`SpaceUtilsLiveParityTests`) | `flatdim` and `flatten_space` (or their errors), `flatten` of 3 seeded samples, their membership in the flattened space, `unflatten` of each, over the catalog and 10 more flattening cases (a `OneOf` whose index needs a wider dtype, text inside a float tuple); `is_space_dtype_shape_equiv` against every space of the same type and one of each other type | |
| CartPole-v1 | reference transcription: 4 seeds × 500 steps, and through termination. **The port, byte for byte:** reset; 4 × 500-step episodes of the balancing policy and 5 random-action episodes, each past termination; `sutton_barto_reward`; semi-implicit Euler; `tau = 0.05`; 51 steps from an injected state; 17 reset-option cases (seeded and unseeded, `ValueError`/`OverflowError` messages, state and generator after a failed reset); 13 invalid actions and 3 pre-reset steps (`AssertionError` messages); the post-termination warning (text, category, count); frames at A.9's states, along 4 episodes, at 20 extreme states (clipped, Sint16-wrapped, quarter-turn snapped, and pygame's NaN/overflow errors) and before reset (`None`); observation and action spaces; metadata | |
| Rendering (`Gym.Rendering` vs pygame-ce 2.5.8) | `aapolygon`, `filled_polygon`, `aacircle`, `filled_circle` (150 random shapes each, clipping and translucency), `hline`/`vline`, compositions, Sint16 wrap-around polygons, CartPole's shapes, `transform.flip`, `Vector2.rotate_rad` bit for bit (NaN payloads included), pygame's argument errors | |
| Rasterizer of P7 (`PygameDrawLiveParityTests`, D4: Gym.NET's own code, behaviour derived from pygame) | `draw.line` (600 lines: integer and float endpoints, both directions, to ±5,000 off-surface); `draw.rect` widths 0, 1 and -1 (600 rectangles across the edges, the clipped-sliver rule); `image.load` of every PNG the classic_control and toy_text envs load (RGB, alpha, size, per-pixel alpha flag); `image.frombytes` and its errors; blitting per-pixel alpha onto a frame (150 sources of every alpha, across the edges); `transform.scale` (157 size pairs, shrinking and growing) and its error; `transform.smoothscale` shrinking (150 random images on one or both axes, Pendulum's arrow at 26 sizes); `transform.flip` with alpha; every alpha byte compared | `draw.aaline` and `draw.aalines` (P7: the model misses about one drawn pixel in ten thousand by one, and the joint rule is not settled; reported to the maintainer per D4) |
| MountainCar-v0, MountainCarContinuous-v0 (P4, one `MountainCarEnv`, DR-32) | resets (5 seeds); 200-step sampled episodes (3 seeds each mode, the continuous mode's float32 NEP 50 arithmetic); the momentum policy to the flag and 5 steps past it; `goal_velocity`; the continuous mode's forces of every dtype (float32/float64 inside and beyond the clamp, int, bool, two elements, NaN), NumPy's `IndexError`s and `math.pow`'s `OverflowError`; unsigned discrete actions (0 wraps, as NumPy's `action - 1`); 13 reset-option cases × 2 modes, seeded and unseeded; 12 invalid discrete actions (`AssertionError` texts); steps before reset (`AttributeError` naming each mode's class); 12 injected states × 2 modes; `None`, short and long states stepped and rendered (Python's `TypeError`/`ValueError`, NumPy's `IndexError`); render before reset; frames along episodes and at injected states compared outside the hill's measured P7 gap; spaces with seeded samples; metadata; `repr` of the bare env and `make`'s stack; `make`'s stack through truncation (200, 999) | whole frames (P7: 16 pixels at the hill's joints) |
| Pendulum-v1 (P4) | resets; 200-step sampled episodes (3 seeds); torques of every dtype and value (float64, beyond the clamp, int, bool, two elements, NaN); NumPy's errors for 0-d and empty torques; step before reset; 14 reset-option cases (`x_init`/`y_init`, the generator's `high - low < 0` and `Range exceeds valid bounds`); 11 injected states; malformed states stepped and rendered; `g=9.81`; frames along 2 episodes (the arrow at every size and both mirrorings) and at injected states after a reset and after 5 torques each; the missing `state`/`last_u` render errors; spaces; metadata; `make`'s stack through truncation | |
| Acrobot-v1 (P4) | resets (the float32 state's NumPy float32 cos/sin); 500-step sampled episodes (3 seeds); the pumping policy through termination and past it; list-indexed actions (negative indices, NumPy integer scalars); 13 refused actions with Python's `TypeError`/`IndexError` texts, before and after reset; 10 reset-option cases; 8 injected states; malformed states (NumPy's `IndexError` in `_dsdt`, the broadcast `ValueError` in `rk4`, the assertion for `None`); `book_or_nips = 'nips'`; `torque_noise_max` (generator draws); frames along 2 episodes and at injected states, and `None` before reset; spaces; metadata; `make`'s stack through truncation | |
| FrozenLake-v1, FrozenLake8x8-v1 (P5) | the transition table `P` of 9 variants (both maps, solid ice, `success_rate`, `reward_schedule`, custom maps of 1x4, 3x3, 2x9 and 16x2); 120-step sampled episodes past termination (2 seeds per variant); actions as Python's dict lookup takes them (`True`, floats equal to a key, NumPy scalars) and refuses them (`KeyError`, `TypeError`), before and after reset; the `ansi` text after the reset and 25 steps, and the caption's `TypeError` after a float action; frames along episodes of every variant; `generate_random_map` (8 cases, redraws, `p` beyond [0, 1]); spaces; metadata; `desc` as NumPy's `S1` array; `make`'s stack through truncation (100, 200) | |
| CliffWalking-v1, CliffWalkingSlippery-v1 (P5) | the transition table, dry and slippery; 200-step sampled episodes (3 seeds each); a walk to the goal and on, and the cliff's -100; the dict lookup of actions; the `ansi` text before reset (`AttributeError`), after it, along episodes and with the agent at every cell; frames along episodes and at every one of the 48 cells; render before reset; spaces; metadata; `make`'s stack | |
| Taxi-v4 (P5) | the transition table dry, rainy and at `rainy_probability=0.6`, and `initial_state_distrib`; 200-step sampled episodes of 5 variants (fickle resets' extra draw, `fickle_step` after each); a scripted delivery with wrong pickups and dropoffs; the fickle passenger at every cell, destination and action, dry and rainy (`np_random.choice` draws); the dict lookup of actions; `encode`, `decode` and `action_mask` of every state; the `ansi` text at all 500 states and along episodes, the caption's `TypeError` after a float action; frames along episodes, at every seventh state, and with each of the 8 taxi sprites; a float move breaking the sprite lookup until an int move; render before reset; spaces; metadata; `desc`; `make`'s stack through truncation | `decode` beyond the grid: the bare assertion's empty message (N-1) |
| Blackjack-v1 (P5) | 150 sampled and 150 strategy games under each of 4 rule sets (`sab`, `natural`), the hands, the face-up card's face and its `np.str_` suit after every game; every reward rule on 12 injected hands, stick and hit; `draw_card`/`draw_hand` value for value with the generator's state (4 seeds); the module's hand functions over every hand of up to 3 cards, `cmp` with NaN; admitted actions and steps before reset; frames of 60 seeded games and of injected hands; render before reset; spaces (`Tuple` `repr` and samples); metadata; `make`'s stack (100 games) | refused actions: the bare assertion's empty message (N-1) |
| LunarLander-v3 | Gym.NET discrete action space | continuous action space (B-20); observation space (B-19); heuristic episodes, both variants (F6, P8, B-17, B-19) |
| Registry and checkers (P3) | `str(make("CartPole-v1"))` and the `pprint` of its stack's and unwrapped env's specs, and the same for CartPole-v0, P4's four ids and P5's six (Blackjack's kwargs in Gymnasium's order); MountainCarContinuous-v0's `kwargs` deviation pinned (DR-32); Taxi-v4's float64 threshold pinned (`8.0` for Gymnasium's int `8`); `make`'s errors (a mistyped name with its suggestion, a missing version, an unknown namespace, malformed ids, `a:b:c`); `spec`'s version errors over ids registered on both sides (deprecated, newer, default, namespaced) and a refused versioned/unversioned mix; `EnvSpec.to_json` byte for byte and `from_json` back (every JSON value kind, wrapper specs, defaults, malformed input); `pprint_registry` over example registries (namespaces from entry points, columns, exclusions); `check_obs`'s warnings and assertions over 11 observation/space pairs | |
| Gymnasium's tools on Gym.NET's envs (P3.5, `Tests/Bridge`) | `make(spec)` over the bridged CartPole-v1: `str` of the stack, both spaces (`repr`, `==`), metadata, the unwrapped spec; the space mirror against Gymnasium's own construction (float/int `Box`, `Discrete` with start and dtype, `Tuple`, sorted `Dict`); `check_env` on `make("CartPole-v1").unwrapped` (every render mode made through the bridge, the double close), the bare env and the bare `rgb_array` env, held to Gymnasium's verdict on its own env and to Gym.NET's `EnvChecker.CheckEnv`; `check_environments_match` against Gymnasium's CartPole, frames included, 3 seeds × 200 steps, both orders; `check_env` and `check_environments_match` (2 seeds × 200 steps, both orders, CartPole's, Pendulum's and Acrobot's frames included, Pendulum's `np.float64` reward through `BridgeReward`) on CartPole-v0, MountainCar-v0, MountainCarContinuous-v0, Pendulum-v1 and Acrobot-v1; the same on P5's six toy_text ids, frames included, the bridge returning the Python ints Gymnasium's envs return (`BridgeObservation.PythonInt`, `BridgeReward.Int`); `make`'s wrapped constructor `TypeError`; the step-past-termination warning crossing into Python's `warnings`; negative controls: a planted τ fails `env_match` at step 0, a reused observation buffer and an unseeded reset fail `check_env` with Gymnasium's messages, as Gym.NET's checker does | |
| Core API (P1) | `make("CartPole-v1")`'s stack against `Gymnasium.Make("CartPole-v1")` through truncation at step 500 (2 seeds); `OrderEnforcing` and `TimeLimit` call by call (`ResetNeeded`/`TypeError`/`ValueError` messages, `has_reset`, `_elapsed_steps`, truncation past the limit, `str`); the attribute protocol through a stack (plain `getattr`/`setattr`, `has/get/set_wrapper_attr`, run-time attributes, read-only properties, a wrapper's `_np_random`); `np_random`/`np_random_seed` (lazy, seeded, -1 once assigned, through a wrapper); `render()` without a render mode on an env with a spec (the warning and `None`); `EnvSpec` repr and `pprint`; `parse_env_id`/`get_env_id`; the single-env step-API conversions, and `step_api_compatibility` over 16 single and vector tuples (Python's operand semantics, both directions, the refusals); dict repr | `render()` without a render mode or spec: the bare `AssertionError`'s empty message (N-1, NumSharp) |
| `PyMath` (F3) | `//`, `%`, `divmod` on ints and floats, `min`/`max`, `round` with and without digits, `np.clip` (512 triples), `np.sign`, bit for bit against CPython 3.12 and NumPy 2.4.2 | |
| Vector envs (P6.1, `Tests/Vector`) | `make_vec` sync and vector entry point: `str` with and without a spec, both batched and both single spaces, `num_envs`, the metadata with its `autoreset_mode`, the spec `make_vec` records; its errors (an unknown vectorization or autoreset mode, vector kwargs with the vector entry point, an id without one); `SyncVectorEnv` lockstep (one seeded reset, the batched action space's seeded samples compared before every step, every result compared whole) over CartPole for 300 steps in each autoreset mode (`NEXT_STEP`, `SAME_STEP` with its `final_obs` object array and `final_info`, `DISABLED` resetting through `reset_mask`), FrozenLake's int observations and int/float infos (`_add_info`'s dtypes and `_key` masks), Taxi's int8 action masks, Blackjack's tuple observations batched into a tuple of arrays; seeds (`seed + i`, a list with `None`, the length `ValueError`); `reset_mask` (only the masked sub-envs reset) and its four refusals; `observation_mode="different"` over FrozenLake 4x4 and 8x8 and `"same"` refused; `get_attr`/`set_attr` (per env, broadcast, the length error)/`call` through the sub-envs' wrappers; `render` frames; `CartPoleVectorEnv` for 600 sampled steps of 4 carts with its `state` after every step (terminations, re-draws, truncations, Sutton and Barto's rewards, CartPole-v0), its reset bounds, refusals and frames | |
| `play` (P6.7, `PlayLiveParityTests`, DR-36) | `display_arr` pixel for pixel against pygame's on 12 combinations (RGB and 2-D palette frames, transposed, not and `None`, scaled up, down and to nothing, centred with odd and even margins, cropped with offsets truncated toward zero) and 4 refusals (pygame's newline-terminated shape message, NumPy's `swapaxes` `AxisError`); `PlayableGame` over CartPole event by event (5 scripts: relevant, irrelevant, repeated and unpressed keys, Escape mapped and not, `QUIT`, resizes rounded half to even) and 10 refusals or lookups (MountainCar-v0's relevant keys 275 and 276); `play` callback by callback over 10 scenarios with the display at each callback (each key form of Gymnasium's test, a wrapper's `get_keys_to_action`, episodes restarting on the seed with the reset's `(obs, info)` as `obs_t`, a held combination, `wait_on_player`, `rgb_array_list`, an explicit no-op without transpose, posted resizes) and 15 refusals; MountainCar's mapping in its order; `PlayPlot` over 7 scenarios through a matplotlib stand-in (titles, scatters, removals, limits, pauses, data, `t`, `zip(strict=True)`'s errors) | |
| Benchmarks and `FuncEnv` (P6.7, `PerformanceLiveParityTests`, DR-36) | `benchmark_vector_step` over two one-step CartPoles in each autoreset mode with both clocks scripted: the rate and the sub-envs' states (float64 values); `benchmark_step`, `benchmark_init` and `benchmark_render` rates; a `FuncEnv` overriding nothing: option attributes, `NotImplementedError` from every function, empty infos, no default params, `transform`'s names | |
| Async vector env (P6.2, `AsyncVectorEnvLiveParityTests`, DR-22) | Gym.NET's threads against Gymnasium's own `AsyncVectorEnv`, whose worker processes are spawned from the embedded interpreter (`multiprocessing.set_executable` names the session's `python.exe`; Inconclusive without one): CartPole x3 in lockstep for 150 steps in every autoreset mode, with and without shared memory, Gym.NET in every execution mode; FrozenLake's ints and Blackjack's tuples in both copy paths; the reset right after an episode end, which clears the pending autoreset in both of Gym.NET's copy paths as in Gymnasium's shared-memory path, and DR-42's deviation pinned (Gymnasium's path without shared memory keeps it: 0 for each sub-env that had just ended, where Gym.NET steps on); a masked reset's no-op answer; `set_attr`/`get_attr`/`call('render')`; the call protocol's errors (waits with nothing pending, calls while one is pending, the warning of a close that interrupts one, calls after the close); `max_concurrency`'s refusals and runs at 1 and 2; a sub-env failing (CartPole's invalid action): the error, the first and last error lines, the dropped worker's `AttributeError` | |
| Stateful wrappers (P6.3, `Tests/Wrappers`) | 300 sampled steps across episodes over `make`'s stacks: `NormalizeObservation` over CartPole's and Pendulum's float32 observations (float32 statistics) and FrozenLake's integer ones (promoted to float64), `obs_rms` afterwards; `NormalizeReward` over float and int rewards at two discounts, `return_rms` and `discounted_reward` afterwards; `TimeAwareObservation` flattened, normalized and as `{"obs", "time"}` through truncation; `FlattenObservation` of a `Box`, a `Discrete` (one-hot) and Blackjack's `Tuple`; Gymnasium's make-test stack of three; the refusals (`gamma` outside [0, 1], a non-positive `epsilon`, `TimeAwareObservation` without a step limit) | |
| Rendering wrappers and `save_video` (P6.5, `RenderingWrappersLiveParityTests`) | `AddWhiteNoise` (RGB uint8 and gray uint64 noise) and `ObstructView` (three ratio/width/noise cases, patches cut by the frame's edges) over CartPole: the frame after a seeded reset and every fifth of 40 steps, byte for byte, the pair's generator check proving the draws' order; `RecordVideo` through 300 steps under the default schedule, a step trigger with a length, both triggers, and `RenderCollection`'s list mode: each video's name, frames and frame rate and the counters afterwards (moviepy, absent from the stack, replaced by a stand-in `ImageSequenceClip` that records what it would encode); `HumanRendering` over `rgb_array` and `rgb_array_list` (pygame's dummy display against `NullEnvViewer`): results, `render()`, render mode and modes; `save_video` over 30 frames with the default schedule, step triggers, lengths (a negative one) and a starting step; `capped_cubic_video_schedule` for 0 … 3000 and -1's `TypeError`; the refusals of all four | |
| OpenCV's resize (P6.4, `Cv2ResizeLiveParityTests`, DR-23) and `ResizeObservation` | `Gym.Utils.Cv2.Resize` against `cv2.resize` 4.13.0 over 1,250 random uint8 images of 1 to 5 channels (one channel as `(H, W)` and `(H, W, 1)`, the dropped axis included): integer-factor area averages (the 2×2 kernel's round-half-up for 1, 3 and 4 channels, the float product otherwise), fractional area decimation, `INTER_AREA`'s bilinear emulation when an axis grows, `INTER_LINEAR`, and exact halving; CartPole- and Atari-sized frames to 84 × 84, 64 × 64 and smaller; five channels by area refused as OpenCV's assertion. `ResizeObservation` over CartPole's frames (shrunk, grown on one axis, grayscale with and without its channel axis), its space and `cv2_shape`, and its refusals (`Tuple` space, 1-D box, float32 and unbounded boxes, shapes of the wrong length or sign) | |
| Stateful observation and action wrappers (P6.3, `StatefulObservationLiveParityTests`, `StatefulActionLiveParityTests`) | 150 to 200 sampled steps across episodes: `DelayObservation` (no delay, CartPole's float32 zeros, FrozenLake's `np.int64(0)` beside its ints, Blackjack's tuple of zeros) and its queue afterwards; `FrameStackObservation` padded by the reset, by zeros and by a given array, over arrays, FrozenLake's ints, Blackjack's tuples and a dict, with the stacked space, `padding_type`, `padding_value` and the deque afterwards; `MaxAndSkipObservation` over CartPole (episodes ending inside a skip: the stale buffer), FrozenLake's ints, Pendulum, MountainCar and uint8 frames; `StickyAction` with fixed and drawn run lengths over Discrete and Box actions and FrozenLake's slippery draws (the order of the draws on the shared generator), with the run's state and `last_action` afterwards; `RepeatAction` over five envs; every refusal with its message (`StickyAction`'s probabilities, NaN included, and durations of every wrong kind) | |
| Vector wrappers (P6.6, `VectorWrappersLiveParityTests`, DR-24) | Sync vector envs in lockstep (the seeded reset, the batched action space's seeded samples compared before every step, every result compared whole) through each wrapper of `gymnasium.wrappers.vector`: `NormalizeObservation` over CartPole and Pendulum (200 steps, `obs_rms` afterwards); `NormalizeReward` at `gamma=0.9` under `NEXT_STEP` and `SAME_STEP` (`return_rms` and `accumulated_reward` afterwards); `RecordEpisodeStatistics` under both modes (250 steps, its `t` left out on both sides, the counters and queues afterwards); `FlattenObservation`, `ReshapeObservation`, `DtypeObservation` and `TransformObservation` over CartPole's state and their spaces; `GrayscaleObservation` (both `keep_dim`) and `ResizeObservation` over CartPole's frames (`AddRenderObservation` on each sub-env); `FilterObservation` over `TimeAwareObservation(flatten=False)` sub-envs; `ClipAction`, `RescaleAction` and `TransformAction` over Pendulum with their spaces; `ClipReward` and `TransformReward`; `DictInfoToList` over FrozenLake's infos (Gym.NET's list-returning `Step`/`Reset`); `HumanRendering` of 3 and 4 sub-envs (pygame's dummy display against `NullEnvViewer`) with its grid (`screen_size`, `scaled_subenv_size`, `num_rows`, `num_cols`); `RecordVideo` over the moviepy stand-in: names, mosaics and frame rate under the default schedule, `record_first_only`, a 16:9 aspect and a step trigger with a length | |
| Stateless wrappers (P6.3, `TransformWrappersLiveParityTests`) | Sampled episodes (20 to 200 steps) through each wrapper over the built-in envs whose spaces it takes, with the space's `repr` and the attributes it computes: `TransformObservation` with and without a space; `FilterObservation` of `TimeAwareObservation`'s dict (three key lists) and Blackjack's tuple (negative and reordered indices); `GrayscaleObservation` of CartPole's frames (`keep_dim` both ways, uint8 through float64 weights); `ReshapeObservation` of a state and of frames; `RescaleObservation` of Pendulum, MountainCar and CartPole (infinite bounds kept); `DtypeObservation` over float, int and Discrete spaces (5 pairs); `AddRenderObservation` (pixels only, and both keys over an array and a dict env); `DiscretizeObservation` (one and per-dimension bins, `multidiscrete`, `bin_edges`, `revert_observation`); `TransformAction`, `ClipAction`, `RescaleAction`, `DiscretizeAction` over Pendulum and MountainCarContinuous; `TransformReward`; `ClipReward` with one and two bounds over Sutton and Barto's `-0.0` rewards; every refusal with its message; `create_zero_array` over the space catalog and Gymnasium's testing spaces; `rescale_box` over 5 dtype and bound mixes | |

| Every registered env (`EnvCoverageLiveParityTests`) | each of the twelve ids: `make_vec(id, 3, "sync")` in lockstep with `Gymnasium.MakeVec` for 220 steps under `NEXT_STEP` and `SAME_STEP` (episode ends, autoresets, `final_obs`/`final_info`, truncations); Gymnasium's own `AsyncVectorEnv` against Gym.NET's for 120 steps with and without shared memory; `render_mode="human"` through the seeded reset, 20 steps (6 at toy_text's 4 frames per second) and every reset after an episode end: the calls, the frames shown by the end of each (pygame's `display.flip`/`display.update` counted against the viewer's renders) and the frame on display pixel for pixel, MountainCar's two outside the hill's P7 gap; `rgb_array_list` through `make`'s `RenderCollection`: the frames per render after 4 and 3 steps and the empty third render, each frame pixel for pixel, MountainCar's outside the gap; CartPole-v0's `make` stack through its truncation at step 200 (2 seeds) | MountainCar's whole `human` and `rgb_array_list` frames (P7: the hill's 16 or 17 joint pixels) |

Not built yet (plan §8–§10): Gymnasium's pytest modules run through the bridge (plan §8's third item),
and differential fuzzing.

### 5.1 Coverage by env

Every id Gym.NET registers, against the layers above. "Own suite" is the env's file under `Tests/Envs` (transition
tables or dynamics step by step, injected states, every refused action and malformed state with Gymnasium's message,
spaces, metadata, `rgb_array` frames); "`make` stack" runs Gymnasium's `make(id)` against `Gymnasium.Make(id)` through
the id's step limit; "bridge" is Gymnasium's own `check_env` and `check_environments_match` on Gym.NET's env (P3.5);
"golden" replays without Python (`tests/Golden`); the vector, `human` and `rgb_array_list` columns are
`EnvCoverageLiveParityTests`. "Gap" marks frames compared outside MountainCar's open P7 hill gap, with the whole frame
in an `[OpenBugs("P7")]` twin.

| Id | Own suite | `make` stack | Registration | Bridge | Golden | Sync / async vector | `human` | `rgb_array_list` | `ansi` |
|---|---|---|---|---|---|---|---|---|---|
| CartPole-v0 | CartPole-v1's (one class) | 200 steps | ✓ | ✓ | through `make`, truncating at 200 | ✓ / ✓ | ✓ | ✓ | no such mode |
| CartPole-v1 | ✓ (`CartPoleLiveParityTests`) | 500 steps (`CoreLiveParityTests`) | ✓ | ✓ | 2 cases | ✓ / ✓ | ✓ | ✓ | no such mode |
| MountainCar-v0 | ✓ (frames: gap) | 200 steps | ✓ | ✓ (no render mode) | no frames | ✓ / ✓ | gap | gap | no such mode |
| MountainCarContinuous-v0 | ✓ (frames: gap) | 999 steps | ✓ (DR-32's kwargs pinned) | ✓ (no render mode) | no frames | ✓ / ✓ | gap | gap | no such mode |
| Pendulum-v1 | ✓ | 200 steps | ✓ | ✓ | ✓ | ✓ / ✓ | ✓ | ✓ | no such mode |
| Acrobot-v1 | ✓ | 500 steps | ✓ | ✓ | ✓ | ✓ / ✓ | ✓ | ✓ | no such mode |
| FrozenLake-v1 | ✓ | 100 steps | ✓ | ✓ | ✓ | ✓ / ✓ | ✓ | ✓ | ✓ |
| FrozenLake8x8-v1 | ✓ | 200 steps | ✓ | ✓ | ✓ | ✓ / ✓ | ✓ | ✓ | ✓ |
| CliffWalking-v1 | ✓ | no limit registered | ✓ | ✓ | ✓ | ✓ / ✓ | ✓ | ✓ | ✓ |
| CliffWalkingSlippery-v1 | ✓ | no limit registered | ✓ | ✓ | no frames | ✓ / ✓ | ✓ | ✓ | ✓ |
| Taxi-v4 | ✓ (`decode` beyond the grid: N-1) | 200 steps | ✓ (float threshold pinned) | ✓ | ✓ | ✓ / ✓ | ✓ | ✓ | ✓ |
| Blackjack-v1 | ✓ (refused actions' message: N-1) | no limit registered (100 games) | ✓ | ✓ | ✓ | ✓ / ✓ | ✓ | ✓ | no such mode |

The wrapper suites add more per env, each on the envs whose spaces the wrapper takes: CartPole by far the most, then
Pendulum, Blackjack and FrozenLake, and a few cells over both MountainCars, Taxi and Acrobot. `play` runs over CartPole
and both MountainCars.

---

## 6. Pitfalls this project has already paid for

- **`Environment.SetEnvironmentVariable` does not reach CPython.** It changes the Win32 environment block; CPython's
  `Py_Initialize` reads the C runtime's copy (measured: `PYTHONHASHSEED=0` stayed invisible, `hash_randomization` 1).
  `NativeEnvironment` also calls `_wputenv_s` (UCRT) or `setenv` (libc), and the stack check verifies it took effect.
- **Reading Gymnasium's `np_random` creates a generator** when none exists. `GymPair` compares generator states only
  after a seeded reset; without one it notes the skip instead of reporting two unrelated entropy-seeded generators.
- **NumSharp 0.70's size-less `Generator.choice(array)` returned a one-element array** where NumPy returns a scalar.
  NumSharp 0.80.0 returns a 0-d array like `integers`, so the draw-pattern cells compare `choice` as returned (no
  `np.atleast_1d` any more) and `Discrete.SingleValue` now rejects anything but a 0-d draw, so a regression would surface.
- **MSTest 4.4.1's `TestContext.Current` is experimental** (MSTESTEXP, an error under warnings-as-errors);
  `ParityTestBase.CurrentContext` replaces it (tests run one at a time, `[DoNotParallelize]`).
- **`dotnet sln add` (SDK 10) adds referenced projects and x64/x86 platforms** to the solution; add a project to
  `Gym.NET.sln` by hand.
- **.NET 8's `double %` is not C's `fmod` for non-finite dividends**: it returns its own sign-set NaN for `±inf` and NaN,
  where the C runtime keeps the infinity's sign or the NaN operand. Measured through `Vector2.rotate_rad(inf)` differing
  in one bit on net8.0 only; code that must match C (`Gym.Rendering.Vector2`) handles non-finite values itself.
- **Gymnasium's state attributes change container between calls.** MountainCar's `state` is a float64 array after a
  reset, a tuple after a discrete step and a float32 array after a continuous step; Acrobot's is float32 after a reset
  and float64 after a step. `GymPair` therefore compares the state's float64 values (`PyGymEnv.GetStateValues`:
  `np.asarray(state, dtype=np.float64)` of a tuple, list or float array), exact for float32 and int elements; CartPole's
  float64 array comes back unchanged, so its goldens did not move.
- **A 0-d NumSharp array stands for a NumPy scalar, not a 0-d NumPy array.** Crossed with `ToNumpyCopy` it arrives as
  `array(0, dtype=uint64)`, whose `repr`, type and `contains` verdicts differ from `np.uint64(0)`'s; build such actions
  with `ParityAction.Scalar` (`a[()]` on the Python side).
- **NaN sign bits are not a contract when NaNs of both signs meet.** .NET's `double.NaN` is the negative NaN
  (0xFFF8...), Python's `float('nan')` the positive one; inject Python's. Two NaNs in one operation keep the sign of
  whichever operand the compiled code puts first, and NumPy's build, .NET 8's JIT and .NET 10's disagree: Acrobot's
  `_dsdt` negates a NaN, so a stepped NaN state differed in one sign bit on net8.0 only. Such states are rendered (an
  exception, not bits), not stepped.
- **`make` merges kwargs into the registered ones.** `gymnasium.make("Blackjack-v1", natural=True)` runs with the
  registration's `sab=True` too, which overrides `natural`; a cell that builds Gym.NET's env with the constructor
  must pass every registered kwarg explicitly on Gymnasium's side, or the two sides play different rules (measured:
  identical games for 150 seeds of one variant, then a 1.0 against a 1.5).
- **NumPy's `np.str_` is a `str` pythonnet will not convert.** `np_random.choice` over a list of strings returns one
  (Blackjack's suit); `ValueFacts.FromPython` reads any `str` subclass through `str()`, and compares it with a .NET
  string, as a 0-d array stands for a NumPy number.
- **`check_environments_match` compares reward types** (`data_equivalence`): Gym.NET's reward is always a `double`,
  which the bridge returns as a Python `float` unless told the env's Gymnasium twin returns an `np.float64`
  (`BridgeReward.NumpyFloat64`, Pendulum's `-costs`).
- **`multiprocessing` cannot spawn from the embedded interpreter on its own.** There `sys.executable` is the .NET test
  host, so Gymnasium's `AsyncVectorEnv` fails with `[WinError 2]`; the async cells call `multiprocessing.set_executable`
  with the session's interpreter first, resolved to a full path (the session may know it as a bare `python` from `PATH`,
  which `CreateProcess` does not search). The spawned workers get the parent's `sys.path`, `refs/Gymnasium` first.
- **A recording rewrites every golden once the manifest gains a key.** Each header carries the stack's versions (D15's
  per-file manifest), and the store compares whole content, so after P6.4 pinned `opencv.version`, recording only the
  new CartPole-v0 case rewrote all twelve older goldens with nothing but that header key changed (their calls and frames
  identical, checked line by line). Restore such files (`git checkout -- tests/Golden/...`) unless re-pinning every
  golden is the point of the change; the replay checks the case parameters and the Gymnasium commit, not `stack`.
- **`Python.numpy.ndarray.tobytes_t` (pythonnet's `As<byte[]>()`) is slow for large buffers**: about 84 ms for a 600 × 400 × 3 frame (measured once, JIT
  warm-up included). Fine for a handful of frames per test; for a frame every step of long trajectories, read the bytes
  through NumSharp's buffer import instead.
