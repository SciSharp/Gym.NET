# MuJoCo (P9): what completing it takes

| | |
|---|---|
| **Date** | 2026-11-11 |
| **Status** | Assessment for P9, written while DR-35 had postponed it out of 2.0 and DR-30, DR-31 and DR-33 asked for a managed port of MuJoCo and its renderer. Rounds 21 and 22 of `DESIGN_REVIEW.md` (2026-11-12) replaced them: the proof of concept's core becomes the MuJoCo envs (DR-61), on MuJoCo 4.0.0 (DR-64), with frames drawn by Gym.NET and measured only (DR-62, DR-67), and Gymnasium's reach of the Python API under .NET names (DR-68). §14 verifies the core on 4.0.0; §15 is the API comparison behind DR-68 |
| **Measured against** | Gymnasium `8ba8c3a6` (v1.4.0, committed 2026-10-05); MuJoCo 3.15.0 (released the same day) for §1–§13 and MuJoCo 4.0.0 (2026-11-06) for §14–§15: their sources, and their `win_amd64` wheels for Python 3.12 in isolated venvs |
| **Reproduce** | §8; the probes are in `tools/parity/mujoco/`, the proof of concept for HalfCheetah (§10), Hopper and Humanoid (§11) and all 21 ids (§13) in `tools/parity/mujoco/halfcheetah_poc/`, with its verification scripts (§13) in `halfcheetah_poc/verify/`, the engine's line coverage (§10.6, §12) in `tools/parity/mujoco/coverage/` |

---

## 1. Summary

- **What is to port:** 22 ids, 11 envs in a v4 and a v5. On MuJoCo 3, 21 of them run; Gymnasium's own `Pusher-v4` raises
  `ImportError` ("only supported on mujoco<3", Gymnasium #950), which Gym.NET would reproduce.
- **How much of MuJoCo they reach, measured:**

  | Part | Of | Reached |
  |---|---|---|
  | Physics | 59,095 lines of C | at most 26,028 lines (586 functions) |
  | MJCF compiler | 44,296 lines of C++ (`user` and `xml`) | 23 elements and 93 attributes; about 10,000 lines |
  | Visualizer | engine's `mjv_*` | at most 5,238 lines |
  | Classic OpenGL renderer | `mjr_*` | about 5,300 lines |

  Gymnasium's side adds 6,153 lines of Python env code and 1,138 lines of MuJoCo tests.
- **Bitwise parity is reachable on win-x64:**
  - **No fused multiply-adds:** MuJoCo's x64 builds use `-mavx` or `/arch:AVX` only, and its hand-written AVX kernels contain
    no FMA instructions.
  - **Scalar libm:** `mujoco.dll` imports the C runtime's scalar `sin`, `cos`, `acos`, `atan2`, `exp`, `pow` and `log`, with
    no vectorized math library. That is the arithmetic of .NET's `Math` on the same platform, proven bit-exact for
    CartPole and Box2D.
  - **AVX reductions:** four lanes in a fixed combine order, which C# reproduces exactly.
  - **Single-threaded:** Gymnasium attaches no thread pool.
- **Frames (DR-31):** a ported `mjr_*` issues MuJoCo's own sequence of OpenGL calls. On the same machine and driver, Gym.NET's
  frames can therefore be compared with Gymnasium's exactly; elsewhere they are best-effort, since drivers differ.
- **Size:** about four to five times P8 by source volume. P8 (Box2D, five ids, every parity layer) took about a day and a
  half of continuous sessions, which puts P9 on the order of one and a half to two weeks, frames included. The
  uncertainty is in four places:
  - bitwise convergence of the GJK/EPA convex collider;
  - the Newton solver's line search and Cholesky updates;
  - the compiler's derived quantities (inertias, `mj_setConst`);
  - OpenGL contexts.
- **A model-specialized core is far smaller (§10):** HalfCheetah-v5's, written from scratch in 1,670 lines of code,
  steps bit for bit with MuJoCo 3.15.0 over 80,000 physics steps, Newton line search and Cholesky updates included.
  Measured by line coverage, the five most-used envs together execute 4,337 engine lines, and Walker2d and Ant add none
  to HalfCheetah, Hopper and Humanoid. A shared core for the five is therefore about 2,200 lines (§10.6). The engine
  sizes above are static upper bounds for a general port.
- **Hopper-v5 and Humanoid-v5 now run on that core (§11):** 2,264 lines of code, 14% above the estimate, step all three
  envs bit for bit with MuJoCo 3.15.0 and Gymnasium over 249 episodes and 138,490 physics steps, PGS, RK4, the free
  joint and the env layers included. It took 36 minutes of one session, against an estimate of two to three hours.
- **The other 18 ids (§12), measured:** Walker2d, Ant and HumanoidStandup already run bit-exact on that core with no
  change. InvertedPendulum, InvertedDoublePendulum and Reacher need about 20 lines: two small generalizations, tried in
  a copy of the core, make all three match, and InvertedDoublePendulum's reward adds site positions. Swimmer needs about
  70 for fluid forces. Pusher needs about 1,050: MuJoCo's GJK/EPA convex collider and
  island-local solving. Together that is about 1,140 lines of core and 490 of env rules, about 3.5 to 6.5 hours, of which
  Pusher is 2 to 4.
- **All 21 now run on that core (§13):**
  - **Size:** 3,776 lines of code, 1,512 of them new (33% above §12's estimate, almost all in the convex collider).
  - **Parity:** every id that runs on MuJoCo 3 steps bit for bit with MuJoCo 3.15.0 and Gymnasium at every layer, over
    45 trace folders, 2,460 episodes and 555,514 physics steps.
  - **Coverage:** a search of about 90 million more physics steps found two branches the traces had missed; both were
    then recorded and are exact. What no trace executes is listed with the reason.
  - **Time:** 40 minutes of one session, against §12's 3.5 to 6.5 hours.
  - **Still a proof of concept:** P9 stays postponed.

## 2. What Gymnasium needs

### 2.1 The envs, run on MuJoCo 3.15.0

1,000 seeded random steps per id (`probe_envs.py`); the v4 and v5 of an env share their model except where noted.

| Ids | Model | Integrator / solver | nq / nv / nu | Contacts (max in a step), pairs | Constraint rows (max) | Islands (max) | Native env step |
|---|---|---|---|---|---|---|---|
| Ant-v4, v5 | `ant.xml` | RK4 / Newton | 15 / 14 / 8 | 4: capsule-plane, sphere-plane | 19 | 1 | ~370 µs |
| HalfCheetah-v4, v5 | `half_cheetah.xml` | Euler (default) / Newton | 9 / 9 / 6 | 5: capsule-plane | 21 | 1 | ~100 µs |
| Hopper-v4, v5 | `hopper.xml` | RK4 / Newton | 6 / 6 / 3 | 2: capsule-plane | 10 | 1 | ~180 µs |
| Humanoid-v4, v5 | `humanoid.xml` | RK4 / PGS, 50 iterations | 24 / 23 / 17 | 3: sphere-plane, capsule-sphere, sphere-sphere, capsule-capsule | 12 | 1 | ~355 µs |
| HumanoidStandup-v4, v5 | `humanoidstandup.xml` | RK4 / PGS, 50 iterations | 24 / 23 / 17 | 11: the same and capsule-plane | 32 | 1 | ~575 µs |
| InvertedDoublePendulum-v4, v5 | `inverted_double_pendulum.xml` | RK4 / Newton | 3 / 3 / 1 | none | 0 | 0 | ~135 µs |
| InvertedPendulum-v4, v5 | `inverted_pendulum.xml` | RK4 / Newton | 2 / 2 / 1 | none | 0 | 0 | ~80 µs |
| Pusher-v5 (v4 refuses) | `pusher_v5.xml` | Euler / Newton, 20 iterations | 11 / 11 / 7 | 5: cylinder-plane, capsule-plane | 6 | 2 | ~185 µs |
| Reacher-v4, v5 | `reacher.xml` | RK4 / Newton | 4 / 4 / 2 | none (a joint limit) | 1 | 1 | ~105 µs |
| Swimmer-v4, v5 | `swimmer.xml` | RK4 / Newton; fluid (density 4000, viscosity 0.1) | 5 / 5 / 2 | none (joint limits) | 2 | 1 | ~150 µs |
| Walker2d-v4, v5 | `walker2d.xml`, `walker2d_v5.xml` | RK4 / Newton | 9 / 9 / 6 | 3: capsule-plane | 15 | 1 | ~255 µs |

Every model uses pyramidal friction cones, MuJoCo's automatic Jacobian (dense below 60 DOFs, so dense here), no disable
or enable flags, and motor actuators only. The step times are this machine's, through MuJoCo's Python bindings, each
covering the env's `frame_skip` of `mj_step` calls.

### 2.2 The `mujoco` API the env code calls

| Area | Calls |
|---|---|
| Models | `MjModel.from_xml_path`, which is the MJCF compiler |
| Simulation | `MjData`, `mj_step`, `mj_forward`, `mj_resetData`, `mj_rnePostConstraint` (for `cfrc_ext`), `mj_name2id`, `mj_getState` / `mj_setState` / `mj_stateSize` |
| `MjData` fields | `qpos`, `qvel`, `ctrl`, `act`, `time`, `cfrc_ext`, `cvel`, `cinert`, `ten_length`, `ten_velocity`, `qfrc_actuator`, `qfrc_constraint`, `site_xpos`, `geom_xpos`, `body(name).xpos`, `solver_niter` |
| `MjModel` fields | `nq`, `nv`, `nu`, `na`, `ncam`, `opt.timestep`, `geom_rgba`, `actuator_ctrlrange`, `stat.extent` |
| Frames | `MjvScene`, `MjvOption`, `MjvCamera`, `MjvPerturb`, `mjv_updateScene`, `mjv_moveCamera`, `mjv_initGeom`; `MjrContext`, `mjr_render`, `mjr_overlay` (text), `mjr_setBuffer`, `mjr_readPixels`; GLFW, EGL or OSMesa contexts |

The env code reads `MjData`'s arrays as NumPy views, so Gym.NET's port exposes them as NumSharp views over the engine's own
buffers.

### 2.3 The models

The 14 bundled models (`mjcf_census.py`) use 23 MJCF elements with 93 element-attribute pairs. That covers:
- **Structure:** bodies; hinge, slide and free joints; geoms: plane, sphere, capsule (`fromto` in 72 of them), cylinder,
  and one box in `point.xml`, which only a custom-env test loads.
- **Properties:** `inertiafromgeom`, `settotalmass`, angles in degrees or radians; joint limits, stiffness, damping,
  armature; geom `solref`/`solimp`/`margin`/`user`; fixed tendons in both humanoids; `numeric` custom data.
- **Visual:** lights, cameras, materials, and three built-in texture kinds (2-D checker, gradient skybox, flat cube with
  random marks).
- **Options:** timestep, integrator, solver, iterations, gravity, and fluid density and viscosity.

No model uses meshes, height fields, sensors, equality constraints, flex or composites.

## 3. The engine subset

**The pipeline that runs** (`engine_forward.c`):
- **Islands are on by default:** `mj_island` runs in every position stage, and `mj_fwdConstraint` solves island by island
  unless `mjDSBL_ISLAND` is set. Island discovery and the solver's island paths are therefore in scope.
- **Off unless enabled:** sleeping bodies (`mjENBL_SLEEP`), the IPC mode (`mjENBL_IPC`) and energy, which no model enables.
- **Integrators:** RK4 and Euler.
- **Solvers:** Newton (with its line search and Cholesky updates) and PGS. CG, the implicit integrators and derivatives are
  never reached.

**Collision reach** (`probe_pairs.py`): every geom pair the models' filters let collide, and the collider MuJoCo's table
assigns it.

| Collider | Models |
|---|---|
| `mjc_PlaneCapsule` | ant, half_cheetah, hopper, humanoid, humanoidstandup, pusher, walker2d |
| `mjc_PlaneSphere` | ant, humanoid, humanoidstandup |
| `mjc_PlaneCylinder` | pusher |
| `mjc_CapsuleCapsule` | hopper, humanoid, humanoidstandup |
| `mjc_SphereCapsule` | humanoid, humanoidstandup |
| `mjc_SphereSphere` | humanoid, humanoidstandup |
| `mjc_Convex` (GJK/EPA) | pusher: the arm's capsules against the cylinder it pushes, which is Pusher's task |
| `mjc_PlaneBox` | point.xml only (no registered env) |

**Size** (`callgraph.py`): the static closure from the entry points of §2.2, with the colliders above as roots (the driver
calls them through a function-pointer table), and without the subsystems no model enables. That leaves sleep, IPC,
derivatives, SDF/flex/box/continuous collision, sensors, ray casting and printing out. The result is 586 of the engine's
1,153 functions, 26,028 lines of C:

| Lines | File |
|---:|---|
| 3,462 | `engine_core_constraint.c` (contacts, limits, constraint assembly) |
| 2,652 | `engine_solver.c` (Newton, PGS; CG is reachable but unused) |
| 2,608 | `engine_core_smooth.c` (kinematics, inertia, factorization, RNE) |
| 2,210 | `engine_collision_gjk.c` (the convex path) |
| 1,889 | `engine_collision_driver.c` (broad phase, filtering) |
| 1,843 | `engine_forward.c` (the pipeline, Euler, RK4) |
| 1,642 | `engine_core_util.c` |
| 1,478 | `engine_setconst.c` (the compiler's `mj_setConst`) |
| 1,331 | `engine_util_misc.c` |
| 1,233 | `engine_passive.c` (springs, damping, the fluid model) |
| 2,706 | the solve, sparse, BLAS, spatial, memory and error utilities |
| 2,974 | `engine_io.c` (585), `engine_collision_convex.c` (564), `engine_island.c` (554), `engine_support.c` (349), `engine_collision_primitive.c` (343, the six primitive colliders), `engine_inverse.c` (318), `engine_name.c` (247), `engine_init.c` (14) |

The closure is an upper bound: it follows every branch, so the sparse paths of models with 60 or more DOFs and the CG
solver count although no env takes them.

## 4. Floating point: why bitwise holds on win-x64

- **Compiler flags:** MuJoCo's build sets `MUJOCO_ENABLE_AVX` and `MUJOCO_ENABLE_AVX_INTRINSICS` (on by default): `-mavx`
  with GCC and Clang, `/arch:AVX` with MSVC. Neither emits fused multiply-adds, and the intrinsic kernels
  (`engine_util_blas_avx.h`, `engine_util_sparse_avx.h`) contain no `fmadd`.
- **The AVX kernels:** element-wise kernels round as scalar code does. The reductions don't: `mju_dot_avx` accumulates four
  lanes, combines them as `(l0 + l2) + (l1 + l3)`, then adds the remaining one to three products. C# reproduces this
  exactly with `System.Runtime.Intrinsics` (`Vector256<double>`), with a scalar fallback in the same order for CPUs without
  AVX.
- **The math library:** the wheel's `mujoco.dll` imports scalar `sin`, `cos`, `tan`, `acos`, `asin`, `atan2`, `exp`, `log`,
  `pow`, `sqrt` and `floor`, and contains no `__vdecl`/`__svml` vector routines. So no loop of math calls was vectorized,
  and the results are the C runtime's, as .NET's `Math` gives on Windows.
- **Elsewhere:** Linux pairs glibc with glibc. MuJoCo's own arm64 builds (Clang's default `-ffp-contract=on`) may fuse
  multiply-adds and differ from x64, so the reference platform is win-x64, as for CartPole and Box2D.

## 5. Frames (DR-31)

- **What to port:**
  - `mjv_updateScene` and its helpers (at most 5,238 lines: the abstract scene of geoms, lights, camera and decorations).
  - The classic renderer, about 5,300 lines: `render_context.c` (2,015), `render_gl3.c` (1,627), `render_gl2.c` (1,333) and
    `render_util.c` (287). This includes MuJoCo's built-in fonts for `mjr_overlay`.
  - The compiler's built-in texture generation, including the cube texture's random marks.

  The `glad` loader is not ported: a managed OpenGL binding replaces it. The newer Filament renderer in `render/filament`
  is not used by Gymnasium.
- **Where it draws:**
  - The OS's OpenGL through a hidden WGL window on Windows, or EGL on Linux. No native binaries ship (DR-30); the driver is
    the system's.
  - `rgb_array`, `depth_array`, and `human` mode through Gym.NET's viewers. Gymnasium's own window uses GLFW with mouse
    camera control; matching that interaction is a separate choice.
- **How exact:** the same GL calls on the same driver give the same pixels. So the live parity suite can compare
  Gym.NET's frames with Gymnasium's exactly on the reference machine.
  - **Different drivers:** across machines, frames differ for MuJoCo's own builds too, so goldens carry observations, not
    frames.
  - **No GPU:** a runner without a GPU has only Windows' OpenGL 1.1 software driver, which MuJoCo's renderer cannot use, or
    Mesa on Linux. Frame cells therefore run where a GPU or Mesa is present.

## 6. The work, in steps

Each step ends at a gate, like P8's.

| Step | Contents | Gate |
|---|---|---|
| P9.0 Stack and measurement | Pin MuJoCo in the parity stack (`requirements.txt`, manifest, venv setup), add `Python.mujoco` facade rows, dump compiled models and step data from live MuJoCo for bootstrapping, and measure the data layout as DR-29 did (flat `double[]` against NDArrays in the hot loops) | The stack check passes; the layout is decided by measurement |
| P9.1 Smooth dynamics | `mjModel`/`mjData` (flat arrays, NDArray views for the env API), the arena, the utilities with the AVX kernels' order, kinematics, centre of mass, composite inertia, factorization, RNE, passive forces (springs, damping, the fluid model), motor actuation, fixed tendons, Euler and RK4 | InvertedPendulum, InvertedDoublePendulum, Reacher and Swimmer step bitwise against live MuJoCo, from Gymnasium's compiled models loaded from the dumps |
| P9.2 Collision and constraints | Broad phase and filters, the six primitive colliders, the GJK/EPA convex path, contacts with pyramidal cones, joint limits, constraint assembly, islands, the Newton and PGS solvers with warm start, `mj_rnePostConstraint` | Every model steps bitwise under random and scripted controls (Pusher's arm driven into the cylinder): contacts, constraint rows and solver statistics equal |
| P9.3 MJCF compiler subset | The XML reader for the 23 elements and 93 attributes, defaults, the kinematic tree, frames (`quat`, `axisangle`, `xyaxes`, `fromto`, degrees), inertia from geoms, `settotalmass`, built-in textures, materials, lights, cameras, `numeric`/`custom`/`size`/`visual`; then `mj_setConst` | Every `mjModel` array of the 14 models equals MuJoCo's, and the bootstrap dumps are dropped; MJCF outside the subset is refused by name |
| P9.4 The envs | `MujocoEnv`, the 11 families as `AntEnvV4`/`AntEnvV5`… (DR-24) with their v5 options records (DR-34), registration of the 22 ids (`Pusher-v4` refusing as Gymnasium does on MuJoCo 3), and every parity layer: own suites, `make`, registration, the reverse bridge, goldens of observations, sync and async vector | L3 for every id, in the audit's terms |
| P9.5 Frames | `mjv_*`, `mjr_*`, the GL binding and contexts, `rgb_array`, `depth_array`, `human`, `rgb_array_list` | Frames equal to Gymnasium's on the same machine and driver |
| P9.6 Gymnasium's tests | `tests/envs/mujoco` (1,138 lines: v5's options, custom envs, rendering) and the MuJoCo rows of `test_make`, `test_action_dim_check`, `test_rendering`, the passive checker's tests and the vector human-rendering tests | The open `TestNumEnvsScreenSize_Ant` passes; the inventory's 33 MuJoCo lines go |

P9.1 and P9.3 can proceed side by side, since the dumps let the engine run before the compiler exists. The release
vehicle is the first minor release after 2.0 (DR-35).

## 7. Decisions to take when P9 resumes

These go to the design review before the step that needs them.

1. **The MuJoCo pin.** D6 says the version Gymnasium's CI used at the pinned commit: 3.15.0 or 3.14.0. MuJoCo 4.0.0 came
   out on 2026-11-06, and a Gymnasium re-pin would likely bring it; this assessment did not measure 4.0's changes.
   MuJoCo releases monthly, so the pin also fixes how often P9's code must follow upstream.
2. **The OpenGL binding.** A small hand-written binding (only the calls `mjr_*` makes, and WGL/EGL context creation)
   against a NuGet package (Silk.NET or OpenTK): code to own, or a dependency.
3. **Frames in goldens:** none (driver-dependent), with frames proven live on the reference machine instead.
4. **`xml_file` beyond the subset:** a refusal naming the unsupported element or attribute, as DR-33's boundary, or a
   wider compiler.
5. **Human mode's interaction:** Gymnasium's GLFW window rotates and zooms the camera with the mouse, while Gym.NET's
   viewers show frames.
6. **The data layout:** decided by P9.0's measurement, as DR-29 decided Box2D's.

## 8. How these numbers were measured

```bash
# A Python with numpy and the MuJoCo to measure, isolated from the parity stack:
python -m venv --system-site-packages <venv> && <venv>/bin/python -m pip install mujoco==3.15.0 imageio
<venv>/bin/python tools/parity/mujoco/probe_envs.py           # §2.1: the 22 ids at run time
<venv>/bin/python tools/parity/mujoco/probe_pairs.py          # §3: the collision reach
python tools/parity/mujoco/mjcf_census.py                       # §2.3: the MJCF surface
# MuJoCo's source at the same version (github.com/google-deepmind/mujoco, tag 3.15.0):
python tools/parity/mujoco/callgraph.py <mujoco>/src/engine \
  mj_step,mj_forward,mj_resetData,mj_rnePostConstraint,mj_name2id,mj_getState,mj_setState,mj_stateSize,mj_makeData,mj_deleteData,mj_setConst,mjc_PlaneSphere,mjc_PlaneCapsule,mjc_PlaneCylinder,mjc_SphereSphere,mjc_SphereCapsule,mjc_CapsuleCapsule,mjc_Convex \
  engine_sleep.c,engine_ipc.c,engine_derivative.c,engine_derivative_fd.c,engine_collision_sdf.c,engine_collision_flex.c,engine_collision_box.c,engine_collision_continuous.c,engine_sensor.c,engine_ray.c,engine_print.c,engine_vis_visualize.c,engine_vis_interact.c
```

The proof of concept (§10, §11) records a run of episodes per folder, about 40 MB per 1,000 HalfCheetah env steps and
437 MB for all eleven of §11.2's folders, so keep them outside the repo. These record §11.2's traces (`--debug-episodes`
bounds the size of the long ones):

```bash
oracle=tools/parity/mujoco/halfcheetah_poc/oracle.py
<venv>/bin/python $oracle --out <dir>/hc_s42 --seed 42
<venv>/bin/python $oracle --out <dir>/hc_n --seed 0 --episodes 10 --debug-episodes 1
<venv>/bin/python $oracle --out <dir>/hc_f --seed 0 --episodes 5 --flip --debug-episodes 1
<venv>/bin/python $oracle --out <dir>/hop_a --env Hopper-v5 --seed 0 --episodes 20
<venv>/bin/python $oracle --out <dir>/hop_b --env Hopper-v5 --seed 100 --episodes 100 --debug-episodes 10
<venv>/bin/python $oracle --out <dir>/hop_nt --env Hopper-v5 --seed 0 --episodes 5 --no-terminate --debug-episodes 1
<venv>/bin/python $oracle --out <dir>/hum_a --env Humanoid-v5 --seed 0 --episodes 5
<venv>/bin/python $oracle --out <dir>/hum_b --env Humanoid-v5 --seed 100 --episodes 100 --debug-episodes 5
<venv>/bin/python $oracle --out <dir>/hum_nt0 --env Humanoid-v5 --seed 0 --no-terminate --debug-episodes 0  # and seeds 1, 2
# the checker: one report per folder, the totals, then the core's speed on each env's first folder; exit code 0 when all exact
dotnet run --project tools/parity/mujoco/halfcheetah_poc -c Release -- <dir>/hc_s42 <dir>/hc_n ...   # --verbose: per-pass tallies
```

The line coverage of §10.6 needs Linux or WSL (gcc, make, curl) for the instrumented build, and the measuring Python for
the plans:

```bash
bash tools/parity/mujoco/coverage/build_mujoco_cov.sh <mujoco-3.15.0-source>      # once: ~/mjcov/build/lib/libmujoco.so
<venv>/bin/python tools/parity/mujoco/coverage/gen_plans.py HalfCheetah-v5 <plans>/hc_a --seeds 0:10
<venv>/bin/python tools/parity/mujoco/coverage/gen_plans.py HalfCheetah-v5 <plans>/hc_42 --seeds 42:43
<venv>/bin/python tools/parity/mujoco/coverage/gen_plans.py HalfCheetah-v5 <plans>/hc_flip --seeds 0:5 --flip
<venv>/bin/python tools/parity/mujoco/coverage/gen_plans.py Hopper-v5 <plans>/hop_term --budget 16000
<venv>/bin/python tools/parity/mujoco/coverage/gen_plans.py Hopper-v5 <plans>/hop_long --seeds 0:16 --no-terminate
<venv>/bin/python tools/parity/mujoco/coverage/gen_plans.py Humanoid-v5 <plans>/hum_term --budget 16000 --rnepost
<venv>/bin/python tools/parity/mujoco/coverage/gen_plans.py Humanoid-v5 <plans>/hum_long --seeds 0:16 --no-terminate --rnepost
# likewise Walker2d-v5 (wal_term, wal_long) and Ant-v5 with --rnepost (ant_term, ant_long)
bash tools/parity/mujoco/coverage/cov_measure.sh <plans> hc_a hc_42 hc_flip hop_term hop_long hum_term hum_long
python3 tools/parity/mujoco/coverage/cov_report.py HalfCheetah=hc_a,hc_42,hc_flip Hopper=hop_term,hop_long \
  Humanoid=hum_term,hum_long Walker2d=wal_term,wal_long Ant=ant_term,ant_long
```

§12's plans add the remaining envs (`--budget 16000` each; `--rnepost` for HumanoidStandup; Walker2d-v4 also with
`--seeds 0:16 --no-terminate`). The first group of `cov_report.py` is the baseline, so the core's three envs together
form one group:

```bash
<venv>/bin/python tools/parity/mujoco/coverage/gen_plans.py HumanoidStandup-v5 <plans>/hst_term --budget 16000 --rnepost
<venv>/bin/python tools/parity/mujoco/coverage/gen_plans.py Pusher-v5 <plans>/pus_term --budget 16000
# likewise InvertedPendulum-v5 (ip_term), InvertedDoublePendulum-v5 (idp_term), Reacher-v5 (rea_term),
# Swimmer-v5 (swi_term), Walker2d-v4 (wal4_term; wal4_long with --seeds 0:16 --no-terminate)
python3 tools/parity/mujoco/coverage/cov_report.py Core=hc_a,hc_42,hc_flip,hop_term,hop_long,hum_term,hum_long \
  Walker2d=wal_term,wal_long Walker2dV4=wal4_term,wal4_long Ant=ant_term,ant_long HumanoidStandup=hst_term \
  InvertedPendulum=ip_term InvertedDoublePendulum=idp_term Reacher=rea_term Pusher=pus_term Swimmer=swi_term
```

§12.3's physics-only traces come from the same oracle with any id. The checker checks an env without env rules physics
only, and reports a model the core refuses by the feature `Load` names:

```bash
<venv>/bin/python $oracle --out <dir>/ant --env Ant-v5 --seed 0 --episodes 20 --debug-episodes 2
<venv>/bin/python $oracle --out <dir>/ant_nt --env Ant-v5 --seed 0 --no-terminate --debug-episodes 0
# likewise Walker2d-v5/-v4 (50 episodes, and no-termination ones), HumanoidStandup-v5 (2 episodes),
# InvertedPendulum-v5 and InvertedDoublePendulum-v5 (200), Reacher-v5 (40), Swimmer-v5 (1), Pusher-v5 (10)
dotnet run --project tools/parity/mujoco/halfcheetah_poc -c Release -- <dir>/ant <dir>/ant_nt ...
```

The probe copy of §12.3 made the two edits described there in a copy of the project; since §13 both are in the core,
and every id has env rules, so these runs now check the env layer too.

§13 records its trace set from `verify/traces.txt`, about 1.6 GB, and checks it whole. The coverage needs
`dotnet-coverage`, a Debug build and a few minutes:

```bash
poc=tools/parity/mujoco/halfcheetah_poc
<venv>/bin/python $poc/verify/record_traces.py --out <set> --python <venv>/bin/python   # 45 folders, 8 at a time
dotnet run --project $poc -c Release -- <set>/*        # 45/45 traces bit-exact, then the speed of each id's first folder
dotnet tool install dotnet-coverage --tool-path <tools>
dotnet build $poc -c Debug
<tools>/dotnet-coverage collect -f cobertura -o <cov.xml> -- $poc/bin/Debug/net10.0/HalfCheetahPoc.exe <set>/*
python $poc/verify/cs_coverage.py <cov.xml> $poc        # lines executed per file, and every unexecuted run
# the branch search: first prove the probes change no bit (the probed copy built as the checker), then search
python $poc/verify/branch_search.py <cov.xml> $poc <probed> checker
dotnet run --project <probed> -c Release -- <set>/*    # must stay 45/45
python $poc/verify/branch_search.py <cov.xml> $poc <search>
dotnet build <search> -c Release
<search>/bin/Release/net10.0/HalfCheetahPoc.exe <set>/pus5_a 0 80000     # likewise each id: a folder, the seed range
python $poc/verify/negative_controls.py $poc <scratch> <set>             # every control gives its expected verdict
<venv>/bin/python $poc/verify/mj_speed.py <set>/hc5 <set>/hop5_nt <set>/wal5_nt <set>/ant5_nt <set>/hum5_nt \
  <set>/hst5 <set>/ip5 <set>/idp5 <set>/rea5 <set>/swi5 <set>/pus5_a
dotnet run --project $poc -c Release -- <the same folders, in the same order>   # the core's side; alternate the two
```

The visualizer's figure uses the roots `mjv_updateScene`, `mjv_moveCamera`, `mjv_initGeom`, `mjv_defaultScene`,
`mjv_makeScene`, `mjv_defaultOption`, `mjv_defaultCamera`, `mjv_defaultPerturb` and `mjv_freeScene`, and excludes the
physics files.

The flags (§4) are in MuJoCo's `cmake/MujocoOptions.cmake` and `cmake/CheckAvxSupport.cmake`. The imports were read from
the wheel's `mujoco.dll` with `strings`. Module sizes are line counts of `src/engine` (`.c` and `.h`), `src/user` and
`src/xml` (`.cc`), and `src/render/classic`.

## 9. The five most-used envs without core code (evaluated 2026-11-11)

The maintainer asked what it would take to port the four or five most-used envs directly, writing no core or shared code.

### 9.1 Which five

Models on the Hugging Face Hub whose id names the env in Gym's id form (`hf_usage.py`):

| Env | Models | | Env | Models |
|---|---:|---|---|---:|
| HalfCheetah | 106 | | Swimmer | 57 |
| Humanoid | 105 | | InvertedPendulum | 27 |
| Hopper | 98 | | Pusher | 26 |
| Ant | 81 | | HumanoidStandup | 20 |
| Walker2d | 69 | | InvertedDoublePendulum | 16 |
| | | | Reacher | 9 |

The top five are the classic locomotion benchmark set: 10 ids (each env's v4 and v5) on 6 models, since Walker2d-v5 has
its own `walker2d_v5.xml`.

### 9.2 What an env is, and what it needs from MuJoCo

The env files are thin (`env_logic.py`): 97 to 225 lines of code each, 1,518 for all ten, plus 129 for `MujocoEnv`; the
rest is documentation. No env calls a MuJoCo function. Each reads `qpos` and `qvel` (Ant and Humanoid also `cfrc_ext`;
Humanoid also `cinert`, `cvel` and `qfrc_actuator`) and computes its reward and termination from them. `MujocoEnv` steps
the simulation (`mj_step` × `frame_skip`, then `mj_rnePostConstraint`). So everything an env computes beyond reward
arithmetic is MuJoCo's.

| Env | Integrator / solver | Joints | Colliders its filters allow | Tendons |
|---|---|---|---|---|
| HalfCheetah | Euler / Newton | slide, hinge | plane-capsule | — |
| Hopper | RK4 / Newton | slide, hinge | plane-capsule, capsule-capsule | — |
| Walker2d | RK4 / Newton | slide, hinge | plane-capsule | — |
| Ant | RK4 / Newton | free, hinge | plane-capsule, plane-sphere | — |
| Humanoid | RK4 / PGS | free, hinge | plane-capsule, plane-sphere, capsule-capsule, sphere-capsule, sphere-sphere | 2 fixed |

**What only some of them need is small** (`callgraph.py`'s definitions): PGS is 296 lines, tendons 459, free-joint
integration 21. Swimmer's fluid model (464) and Pusher's convex path (2,774) are not needed. The rest of §3's 26,028
lines is common to all five: kinematics, inertia, the LDL factorization, RNE, the broad phase (whose order fixes the
contacts' order, and with it the constraint rows' order), contacts, limits, islands, the shared Newton/CG core, line
search, Cholesky updates, RK4 and Euler. For byte parity none of it can be replaced by simpler equations, since a
closed-form planar model rounds differently from MuJoCo's general algorithms.

### 9.3 Three routes

| Route | What is written | Bitwise with Gymnasium | Native binaries | Frames | v5's `xml_file` | Estimate |
|---|---|---|---|---|---|---|
| **A. Each env self-contained, managed** (the literal reading) | Each env carries its own copy of MuJoCo's pipeline for its model: the common core of about 22,000 lines (static upper bound; perhaps 11,000–15,000 after specializing to the model) plus its compiled model embedded as data | Yes, if each copy is exact | None | None, unless each carries a renderer | Refused (no compiler) | Four to five times route C's engine: three to five weeks, and five copies to keep exact through every MuJoCo release |
| **B. Native MuJoCo through MuJoCo's own C# bindings** (the only route that writes no engine code) | The env logic (1,647 lines of Python code for the ten ids and the base) and native packaging. `unity/Runtime/Bindings/MjBindings.cs` is reused as it stands: 7,782 generated lines, Apache-2.0, 348 `DllImport("mujoco")` functions, no Unity dependency, covering `mj_loadXML`, `mj_step`, `mj_forward`, `mj_resetData`, `mj_rnePostConstraint` and `mj_name2id` | By construction on win-x64: DeepMind's release `mujoco.dll` is the wheel's build (`pe_compare.py`: identical once the release's 10,392-byte Authenticode signature is stripped, same link timestamp) | Yes: `mujoco.dll` (5.5 MB) per platform, from DeepMind's release (22–33 MB archives) | MuJoCo's own `mjr_*`, given an OpenGL context to create | Works (the native compiler) | Two to three days with every parity layer, frames one to two more. The other 11 ids then cost about a day, since each is 100–200 lines over the same base |
| **C. One minimal shared managed engine for these five** | §3's subset without the convex path and the fluid model: about 22,800 lines (static upper bound), shared. Compiled models embedded as data (no compiler), renderer deferred | Yes | None | Deferred | Refused | One to one and a half weeks |

### 9.4 The verdict

- **Route A is the most expensive way to get these five.** For MuJoCo envs the core *is* the env: a self-contained port
  repeats the same 22,000-line core in each, so dropping "shared" multiplies the work instead of removing it.
- **Route B is the only one that writes no core code.** It is the fastest, exact by construction, and keeps `xml_file`
  and every model working. It reverses DR-30 (no native binaries, which DR-16's condition required for joining
  Gym.NET.Environments). That would need a new design decision, for example native MuJoCo as a separate package and an
  interim that the managed engine later replaces behind the same env classes. With route B, porting all 21 ids costs
  about one day more than porting five.
- **If DR-30 stands, route C is the cheapest managed one.** It is about half of P9's physics work and none of its compiler
  or renderer, at the price of refusing `xml_file` and having no frames until the renderer comes.

Measured with `hf_usage.py`, `env_logic.py` and `pe_compare.py` (in `tools/parity/mujoco/`; the release archive is
`mujoco-3.15.0-windows-x86_64.zip` from DeepMind's GitHub release), and with `callgraph.py`'s `definitions` for the
function sizes.

> **Corrected by §10.** Routes A and C above size a managed core by the static closure of the functions the models
> reach (about 22,000 lines). A core written for one model is an order of magnitude smaller: HalfCheetah's, built from
> scratch and bit-exact over 80,000 physics steps, is 1,670 lines of code. Route A's estimate and the first bullet of
> §9.4 do not hold; §10.5 gives the revised routes.

## 10. Proof of concept: HalfCheetah-v5 from scratch (2026-11-11)

The maintainer challenged §9.4 and asked for a proof of concept: the most-used env (HalfCheetah, §9.1), byte for byte,
with its core written from scratch and inlined, not converted line by line from the engine.

### 10.1 What was built

`tools/parity/mujoco/halfcheetah_poc/`, a .NET 10 project outside `Gym.NET.sln`; nothing ships from it.

| File | Code lines (with docs) | Contents |
|---|---:|---|
| `Kernels.cs` | 399 (692) | Dot products in the AVX lane order, element-wise kernels, dense Cholesky (factor, solve, rank-one update), MuJoCo's sparse `L'DL` factor and solve, quaternion and spatial helpers |
| `CheetahModel.cs` | 184 (317) | The compiled model, read from MuJoCo's own arrays (oracle.py's `model.json`); `Load` refuses any feature the core does not reproduce |
| `CheetahCore.cs` | 1,087 (1,491) | `Step()`: kinematics, centre of mass, composite inertia, plane-capsule contacts, joint limits, pyramidal contact rows, impedance, springs and damping, RNE, motors, the warm start, the island Newton solver with its exact line search and Cholesky updates, Euler with implicit damping |
| **Core** | **1,670 (2,500)** | |
| `Program.cs` | 273 (427) | The checker: the reset and actions drawn with NumSharp, two passes over a trace, the env layer, a benchmark |
| `oracle.py` | 146 (165) | Records MuJoCo 3.15.0's state after every `mj_step` (46 arrays) and Gymnasium's step results |

The core is specialized, not general. It covers hinge and slide joints, one ground plane against capsules, motors,
Euler, pyramidal cones and the dense Newton solver. It leaves out what HalfCheetah never executes: sparse paths,
sleeping, multi-tree islands, flex, tendons, sensors, the other colliders and solvers. It replaces no equation with a
simpler one. It performs only the operations MuJoCo executes for this model, in MuJoCo's order, and keeps MuJoCo's
footguns: zero-skipping that preserves +0, `x + 0` that turns -0 into +0, and the solver's states as the island path
leaves them.

### 10.2 Results

16 episodes of 1,000 env steps (5,000 physics steps each): seeds 0–9 and 42 as Gymnasium plays them, and seeds 0–4
started upside down (`qpos[2] += pi`), so that the torso and head land on the ground. Each episode ran two passes:
- **Closed loop:** the core runs on its own from its own reset, as an env would.
- **Resync:** each physics step starts from MuJoCo's recorded state.

| Check | Result |
|---|---|
| Reset noise (`uniform`, `standard_normal`) and the 16,000 sampled actions, through NumSharp's `Generator` | bit-exact |
| Closed loop: 80,000 physics steps × 46 arrays (state, `qacc`, forces, contacts, `efc_J` and the 14 `efc_*` arrays, frames, `cinert`, `cdof`, `M`, `qLD`, `cvel`, …) | 3,680,000 of 3,680,000 comparisons bit-exact |
| Resync: the same | bit-exact |
| Env layer: observation, reward, terminated, truncated, `x_position`, `x_velocity`, `reward_forward`, `reward_ctrl`, for 16,000 env steps | bit-exact |
| Speed (best of five) | 5.1–7.5 µs per physics step for the managed core; 8.1 µs for MuJoCo through Python (one `mj_step(nstep=5)` call per env step) |

**Coverage per pass**, from an instrumented copy of the core (since the core matches MuJoCo bit for bit, these are
MuJoCo's own paths):
- 47,478 steps had constraint rows: 52,572 active contacts and 12,044 joint-limit rows.
- Every capsule touched the ground. In 4,079 steps the torso and head did so at once, which exercises the contact order
  of a body with two geoms.
- 63,327 Newton iterations and as many line searches: 40,991 converged at the first Newton step, 22,121 in the one-sided
  phase and 215 in the bracketed phase.
- 6,480 certificate exits; 11,741 rank-one Hessian updates and 28,889 downdates.
- The warm start lost to `qacc_smooth` 32,266 times.

Never reached, so written from the source but untested:
- the Hessian refactorization after a lost rank;
- the line search's midpoint, bracket-failure and budget exits;
- a contact excluded at `dist >= margin`;
- the bad-state auto-reset;
- solref's direct form and contact priorities.

**Negative controls**, each on a copy of the core:

| Change | Effect |
|---|---|
| Sum the four AVX lanes as `((l0 + l1) + l2) + l3` instead of `(l0 + l2) + (l1 + l3)` | First difference at step 13 (2 ulp in `qpos[6]`); 2,401 of 5,000 resync steps differ |
| A fused multiply-add in the scaled-add kernel | Step 0 differs (1 ulp) |
| The monolithic solver scale `1/(meaninertia*nv)` instead of the island's `1/sum(diag M)` | No difference: the Newton solver ends far from its thresholds, so these traces do not discriminate the two; the core follows the source |
| Left-to-right 6-vector dot instead of the AVX order | No difference: a planar model's motion axes have zeros in the same three slots, so both orders add the same products |

The checker therefore sees single-ulp differences, and the AVX lane order and the absence of FMA are what the parity
rests on, as §4 measured. .NET's `Math.Sin` and `Math.Cos` (every hinge rotation) come from the same Universal CRT
import as `mujoco.dll`'s: `coreclr.dll` 10.0.1 and `mujoco.dll` both import `sin` and `cos` from
`api-ms-win-crt-math-l1-1-0.dll`.

**Time:** under an hour of one session, from the request to the 80,000 verified steps. That covers reading the source
paths HalfCheetah executes, writing the core, and checking it; the first run was already exact.

### 10.3 What it does not cover

- **The compiler:** the core reads MuJoCo's compiled arrays. A product would embed them per model, as route C planned;
  v5's `xml_file` stays refused until a compiler exists.
- **Frames:** nothing is rendered.
- **Other platforms:** like every parity claim here, exactness is on win-x64 (§4).
- **The MuJoCo pin:** a re-pin can change MuJoCo's arithmetic. The oracle and checker re-verify a pin in about a minute
  per episode, and a model outside the core's scope is refused by name at `Load`.

### 10.4 Why §9 was wrong

§9.2's premise holds: no equation can be simplified under byte parity. The error was in the size. §3's 26,028 lines
and route A's 22,000 are a static closure: every branch of every function a model can reach. The arithmetic one model
executes is about a tenth of that. MuJoCo's generality (sparse and dense variants, sleep, islands of many trees, flex,
tendons, the other solvers and colliders, derivative support) dominates its source but never runs for HalfCheetah, and
a specialized core omits it without changing a bit of the result.

### 10.5 The routes, revised

Measured by line coverage (§10.6) rather than extrapolated: the engine lines each of the five envs executes, what each
adds to HalfCheetah's, and the C# that takes at the ratios this proof of concept measured.

| Env | Executed engine lines | New beyond HalfCheetah | Adds to the core | C# lines added (estimate) |
|---|---:|---:|---|---:|
| HalfCheetah | 3,636 | | (this proof of concept: 1,670 lines) | |
| Hopper | 3,542 | 130 | RK4; capsule-capsule and sphere-sphere collision; contacts between two moving bodies | about 140 |
| Humanoid | 3,746 | 692 (571 beyond Hopper's) | PGS and its dual matrix; `mj_rnePostConstraint` for `cfrc_ext`; the free joint; sphere-capsule and plane-sphere collision; the midphase order of bodies with several geoms | about 380 beyond Hopper's |
| Walker2d | 3,483 | 56 | nothing beyond Hopper's additions | 0 |
| Ant | 3,601 | 200 | nothing beyond Humanoid's additions | 0 |
| **The five together** | **4,337** | | | **a core of about 2,200** |

- **A. Each env self-contained, managed:** about 1,700 to 1,900 lines each (Hopper about 1,800, Humanoid about 1,900),
  about 9,000 for the five; five copies to re-verify per MuJoCo pin.
- **C. One shared, specialized managed core for the five:** about 2,200 lines of code. That is HalfCheetah's core with
  Hopper's and Humanoid's additions, since Walker2d and Ant execute no engine line beyond them. The engine work is a
  few hours at this proof of concept's pace. The env classes and every parity layer cost the same as in route B, so
  the total is two to three days plus those hours, frames excluded. It keeps DR-30 (no native binaries) and refuses
  `xml_file` and any model outside its scope.
- **B. Native MuJoCo through MuJoCo's own bindings:** unchanged at two to three days. It costs about the same as C, and
  is still the only route with `xml_file` and MuJoCo's own frames by construction.

The decision P9 needs is therefore no longer about cost. It is whether `xml_file` and frames are worth native binaries
(B, which reverses DR-30), or whether the five envs ship managed and exact, refusing custom models (C).

### 10.6 The next two, measured: Humanoid and Hopper (2026-11-11)

After HalfCheetah (106 models on the Hugging Face Hub), the most-used envs are Humanoid (105) and Hopper (98). Their
estimate is measured, not extrapolated: the MuJoCo lines each env executes, compared with HalfCheetah's, and converted
to C# with the ratios the proof of concept measured.

#### 10.6.1 Method

The tools are in `tools/parity/mujoco/coverage/`:
- **The build** (`build_mujoco_cov.sh`): MuJoCo 3.15.0 built with gcov (Debug, `-O0`, no LTO) in WSL.
- **The plans** (`gen_plans.py`): each env's compiled model, saved as MJB from Gymnasium's env, so no compiler code is on
  the measured path. With it, the episodes as Gymnasium plays them: the seeded reset, then the seeded action space's
  samples until termination.
- **The replay** (`cov_run.c`): a C program steps the plans as `MujocoEnv` does: `mj_resetData`, `set_state` with
  `mj_forward`, the controls, `mj_step` × `frame_skip`, and `mj_rnePostConstraint` for the envs that read `cfrc_ext`.
- **The count** (`cov_measure.sh`, `cov_report.py`): a line counts as executed when its count in the full run exceeds
  its count in a setup-only run (loading the model and making the data).

Per env: 16,000 env steps as Gymnasium plays them, and 16 episodes of 1,000 steps with `terminate_when_unhealthy=False`
(fallen states); HalfCheetah: the proof of concept's 16 episodes.

| Env | Episodes | Physics steps | Executed engine lines | New beyond HalfCheetah | HalfCheetah's lines it does not run |
|---|---|---:|---:|---:|---|
| HalfCheetah | 16 | 80,000 | 3,636 | | |
| Hopper | 715 (median 19 env steps) + 16 | 128,080 | 3,542 | 130 | 224: the torso's midphase, Euler with implicit damping |
| Humanoid | 658 (median 22) + 16 | 160,105 | 3,746 | 692 | 582: the Newton solver and the dense Cholesky (Humanoid uses PGS), Euler |
| Walker2d | 742 (median 19) + 16 | 128,000 | 3,483 | 56 | 209 |
| Ant | 77 (median 82) + 16 | 160,290 | 3,601 | 200 | 235 |
| All five | | | 4,337 | | |

The fallen-state episodes add 76 lines for Hopper and 5 for Humanoid: the episodes Gymnasium plays already reach nearly
all of it.

**Calibration** (HalfCheetah: its executed lines by area, and the proof of concept's C# for them):

| Area | Executed C lines | C# lines | Ratio |
|---|---:|---:|---:|
| Utilities, inline helpers, AVX kernels, Jacobians | 841 | 399 (`Kernels.cs`) | 0.47 |
| Smooth dynamics, passive forces | 426 | 235 | 0.55 |
| Constraint assembly | 416 | 177 | 0.43 |
| The Newton solver | 383 | 246 | 0.64 |
| Collision driver and primitives | 695 | 134 | 0.19 |
| The pipeline (forward, step, Euler) | 380 | 106 | 0.28 |
| Islands, sleep, memory, I/O, sensors, and stubs of inactive features | 495 | 8 | ~0 |
| Fields, constructor, model loader | | 373 | |
| **Total** | **3,636** | **1,670** | **0.46** |

The low ratios are where a specialized core drops MuJoCo's general machinery: the broad phase's sweep and pair
bookkeeping, island discovery for a single tree, the arena.

#### 10.6.2 Hopper

| What it executes beyond HalfCheetah | Lines | C# (estimate) |
|---|---:|---:|
| `mj_RungeKutta` and the stage calls it makes | 52 | ~45 |
| Capsule-capsule collision (`mjc_CapsuleCapsule`, `mjraw_CapsuleCapsule`, and `mjraw_SphereSphere` through it) | 52 | ~50 |
| Pairs of two moving bodies (filters and order; the proof of concept walks world pairs only) | 9 | ~30 |
| New branches of functions the core already has (impedance, frame, line search) | 17 | 0 |
| Model loader and validation | | ~15 |
| **Total** | **130** | **about 140** |

On HalfCheetah's core that is about 1,810 lines; self-contained (RK4 instead of Euler), about 1,800. The env layer is
small:
- **Observation:** `qpos[1:]` and `clip(qvel, -10, 10)`.
- **Health:** z at least 0.7, torso angle within ±0.2, state within ±100.
- **Reward:** the healthy reward, plus the forward velocity, minus a float32 control cost `1e-3*sum(a²)`.

#### 10.6.3 Humanoid

| What it executes beyond HalfCheetah and Hopper | Lines | C# (estimate) | Note |
|---|---:|---:|---|
| PGS: `solPGS`, the dual states, residual, cost change, `AR` diagonal, the pyramid's projection, `dualFinish`, its warm start and island path, and the row shuffle | 188 | ~120 | Deterministic: the shuffle's `pcg32` restarts at state 0, increment 1, in every solve |
| The dual matrix `AR = Y*Y' + R` with `Y = J*M^-1/2`: `mj_projectConstraint`, `Y` and `AR` (symbolic and numeric), `mj_solveM2_impl`, sparse combine, transpose | 98 | ~60 | The sparse supports fix the summation order |
| `mj_rnePostConstraint` for `cfrc_ext`: contact forces from the pyramid rows, spatial transforms | 97 | ~55 | `cfrc_ext` is in the observation and the contact cost |
| The free joint: kinematics, centre of mass, velocities, quaternion integration | 48 | ~35 | |
| Sphere-capsule, sphere-sphere and plane-sphere collision; the midphase traversal | 65 | ~55 | The torso has three geoms and each forearm two, so the traversal's order decides the contact order when several touch |
| Two fixed tendons | 49 | ~5 | Inert in `humanoid.xml` (no limit, spring, damper, armature or actuator): a validation check, no tendon code |
| Early returns of inactive features; new branches of functions the core has | 26 | ~10 | |
| Model loader and validation | | ~40 | |
| **Total** | **571** | **about 380** | |

On the HalfCheetah and Hopper core that is about 2,200 lines (the five envs' shared core). Self-contained, with PGS
instead of Newton and RK4 instead of Euler, it is about 1,900. The env layer is larger:
- **Observation:** 348 numbers: `qpos[2:]`, `qvel`, `cinert[1:]`, `cvel[1:]`, `qfrc_actuator[6:]` and `cfrc_ext[1:]`.
  At reset they come from `mj_forward`, so `cfrc_ext` is still zero.
- **Forward velocity:** from the mass centre, whose NumPy sums over the bodies must be reproduced in NumPy's order.
- **Costs:** the contact cost `5e-7*sum(cfrc_ext²)`, capped at 10, and the float32 control cost `0.1*sum(a²)`.
- **Health:** the torso's z within [1, 2].

#### 10.6.4 Time and risks

At the proof of concept's pace (HalfCheetah took under an hour from nothing, oracle and checker included), and to the
same evidence (two passes over many episodes, the env layer, negative controls):

| Env | Session time | Why |
|---|---|---|
| Hopper | 30–45 minutes | 130 lines; the oracle and checker gain RK4, terminating episodes and Hopper's env layer |
| Humanoid | 1.5–2.5 hours | 571 lines in five subsystems, and the larger env layer |
| Both | about two to three hours | |

Productizing them (env classes, registration, parity cells, goldens) is the separate cost of §10.5.

The risks, in order:
1. **PGS:** its exactness rests on the construction order of `AR` (the sparse supports of `Y`) and on the sweeps. It is
   deterministic, but it carries more numerics than Newton.
2. **The midphase** (wrong, see §11.3: MuJoCo sorts the pairs the traversal finds)**:** its depth-first order for the
   torso and forearms decides the contact order when several of their geoms touch at once. `mj_collideTree` always descends the node whose static bounding box has the larger surface, so
   the order is fixed by the model. Its pose-dependent overlap tests only prune pairs that cannot touch, so the order
   can be precomputed per body pair at load, without reproducing the tests.
3. **The mass centre:** the velocity's NumPy reductions.
4. **Capsule-capsule collision:** its geometric branches (parallel segments, sphere fallbacks); the coverage shows
   which ones ran.

§11 measures these: risk 2 was wrong (§11.3), and the parallel-segments branch never ran (§11.4).

## 11. Hopper-v5 and Humanoid-v5 on the same core (2026-11-11)

The maintainer asked for both, built and checked the way HalfCheetah was. The proof of concept's core
(`tools/parity/mujoco/halfcheetah_poc/`) now steps all three envs. Its classes keep their first names (`CheetahModel`,
`CheetahCore`), because the playable HalfCheetah (`play/`) compiles them as they are.

### 11.1 What was added

The core grew by 594 lines of code, to 2,264 (3,376 with docs). §10.6 estimated about 520 (140 for Hopper, 380 for
Humanoid), so the estimate was 14% low. Counted per method against the previous commit:

| Area | Code lines added | §10.6 estimate | Contents |
|---|---:|---:|---|
| RK4 | 57 | ~45 | `RungeKutta` with MuJoCo's tableau; position integration, now shared with Euler |
| The free joint | 46 | ~35 | Kinematics, its six motion axes, velocities, quaternion integration |
| Pairs of two moving bodies | 71 | ~30, plus part of ~55 | The candidate pairs (the broad phase's filters, computed once at construction), the geom order within a body pair (§11.3), the bounding-sphere filter |
| Colliders | 106 | ~50, plus part of ~55 | Sphere-sphere, sphere-capsule, capsule-capsule (general and parallel axes); plane-sphere was already there, inside plane-capsule |
| PGS and its dual matrix | 147 | ~180 | Sweeps with Nesterov extrapolation and gradient restarts, the per-solve `pcg32` shuffle, `AR = Y*Y' + R` from `mj_solveM2`, the warm start's dual cost, `dualFinish` |
| `mj_rnePostConstraint` | 43 | ~55 | `cfrc_ext` from the pyramid and frictionless rows |
| Fixed tendons | 23 | ~5 | Lengths, Jacobians and velocities. The estimate assumed no tendon code, but Humanoid's info reports `tendon_length` and `tendon_velocity` |
| Frictionless contacts | 10 | ~10 | Humanoid's limbs touching each other (condim 1: one row per contact) |
| Fields, constructor, loader and validation | 93 | ~55 | The new arrays and their capacities; `Load` refuses, by name, anything beyond the three models' features |
| **Total** | **594** | **about 520** | |

The checker (`Program.cs`) went from 274 to 461 lines of code: a second trace format, and the env rules of all three
envs (reset noise, observation, reward, health, info). `oracle.py` records any of the three, terminating or not, many
episodes per folder.

### 11.2 Results

Eleven trace folders, recorded from Gymnasium's envs on MuJoCo 3.15.0:

| Trace | Episodes | Env steps | Physics steps | What it exercises |
|---|---:|---:|---:|---|
| HalfCheetah, seeds 0–9 and 42 | 11 | 11,000 | 55,000 | §10's runs, now on the extended core |
| HalfCheetah, seeds 0–4 upside down | 5 | 5,000 | 25,000 | Torso and head on the ground at once |
| Hopper, seeds 0–19 and 100–199, as Gymnasium plays them | 120 | 2,640 | 10,560 | RK4 with Newton; the episodes end at the fall (8 to 127 env steps) |
| Hopper, seeds 0–4, no termination | 5 | 5,000 | 20,000 | Fallen states: every capsule on the ground |
| Humanoid, seeds 0–4 and 100–199, as Gymnasium plays them | 105 | 2,586 | 12,930 | PGS, the free joint, the feet on the ground (16 to 46 env steps) |
| Humanoid, seeds 0–2, no termination | 3 | 3,000 | 15,000 | Lying on the ground: all 18 geoms touch something, limbs touch each other and the torso |
| **Total** | **249** | **29,226** | **138,490** | |

| Check | Result |
|---|---|
| Reset noise and the 29,226 sampled actions, drawn through NumSharp's `Generator` | bit-exact |
| Closed loop: per physics step 30 arrays (state, `qacc`, forces, contacts, `efc_J`, the 14 `efc_*` arrays, the solver's iteration count); in the debug episodes 17 more (`efc_AR`, frames, `cinert`, `cdof`, `M`, `qLD`, `cvel`, …); per env step `cfrc_ext`, `ten_length` and `ten_velocity` | 4,630,182 of 4,630,182 comparisons bit-exact |
| Resync, each physics step from MuJoCo's recorded state | bit-exact |
| Env layer: observation (Humanoid's 348 numbers), reward, terminated, truncated, and every info entry (Hopper's 6, Humanoid's 11) | bit-exact |
| Fresh seeds recorded with the final oracle (Humanoid 300–309, Hopper 300–319) | bit-exact |
| The playable HalfCheetah, built against the extended core: its autopilot's returns on seeds 0–9 | identical to the last bit with the previous core |

What the closed loop exercised: HalfCheetah had 47,478 steps with constraint rows and 63,327 Newton iterations, as in
§10. Hopper had 25,828 steps with constraints and 31,143 Newton iterations. Humanoid had 23,987 steps with constraints
and 393,278 PGS sweeps. Under RK4 these count the solve each step publishes (the last stage); the three stages before
it are checked through the state they produce.

### 11.3 The contact order of a body with several geoms: §10.6.4's risk 2 was wrong

§10.6.4 expected the midphase's depth-first traversal to decide the contact order of the torso and the forearms, and the
first version of the core reproduced that traversal from the model's bounding volume hierarchies. Hopper's and
Humanoid's first traces (short episodes) passed with it, and so did HalfCheetah's seed 42. The wider traces did not:
- HalfCheetah's flipped episodes first differed at step 38, by one ulp in `qpos[4]`. In the resync pass, every step that
  differed had its contacts in another order.
- Humanoid's long episodes broke near step 200, when the torso landed.

The traversal's order never reaches the contact list: `mj_collideTree` ends by sorting the pairs it found (`pairSort`),
by geom ids after `pushGeomGeom` has put the lower geom type first. The core now takes every geom pair of the two
bodies, puts the lower type first, and sorts, so the order depends on geom ids and types alone. The hierarchies are no
longer needed, and left the model dump. §10 passed with a plain ascending walk only because HalfCheetah's torso pairs
are the floor (a plane, always first) against two capsules.

The swap matters, as a negative control shows. Sorting the pairs before the swap gives a plain ascending walk, and it
breaks Humanoid's first lying-down episode at step 288. Humanoid's terminating episodes do not catch it: a hand's
sphere meets the torso's capsules only once the body is on the ground.

### 11.4 What the traces execute

Line coverage of the managed core over the eleven trace folders (`dotnet-coverage`, Debug build, executable lines):
- `CheetahCore.cs`: 1,361 of 1,447 lines;
- `Kernels.cs`: 394 of 421;
- `CheetahModel.cs`: 201 of 202.

Every collider case ran, including a direct sphere-sphere pair. The lines never executed are branches no trace took.
They follow MuJoCo's source but are unproven:
- the auto-reset of a bad state or bad controls;
- capsules with parallel axes, and coincident sphere centres;
- geom priorities, direct (negative) or mixed-sign solref, an impedance of power 1;
- a contact beyond its margin, and a body of zero subtree mass;
- PGS's revert of a row update that raised the cost by more than 1e-10. Minimizing a one-row block exactly cannot do
  that, so the branch is dead for these models;
- Newton's fallbacks: a zero step, an exhausted or stuck line search, a lost rank in the factor update (as in §10);
- the kernels' degenerate cases: tiny pivots, zero-length vectors and quaternions, a zero rotation angle.

### 11.5 Negative controls

Each on a copy of the project, against the traces named:

| Change | Traces | Effect |
|---|---|---|
| Sort a body pair's geom pairs before the type swap (a plain ascending walk) | Humanoid: 100 terminating episodes, 1 lying down | The lying-down episode differs at step 288; the terminating ones stay exact |
| PGS: one `pcg32` stream for the whole run, not restarted per solve | Humanoid | Step 0 differs |
| PGS: the row's diagonal `AR_ii` used as is, not as MuJoCo's `1/(1/AR_ii)` | Humanoid | No difference. The value feeds only the dead revert test (§11.4) and the stopping sum, where an ulp cannot cross the threshold, so no trace can tell the two apart; the core follows the source |
| RK4: the four stages added in reverse order | Hopper and Humanoid | Step 0 differs (2 ulp in `qvel[0]`) |
| Free joint: the quaternion not normalized before the rotation | Humanoid | Step 0 differs (1 ulp) |
| Humanoid's control cost summed left to right, not by NumPy's pairwise sum | Humanoid | Env layer, step 0: `reward_ctrl` differs by 1 ulp |
| The mass centre's sum over bodies by NumPy's pairwise sum, not `einsum`'s left-to-right | Humanoid | Env layer, step 1: `y_velocity` differs by 133 ulp |
| Hopper's control cost in float64, not NEP 50's float32 | Hopper | Env layer, step 0: the reward differs |

### 11.6 Speed

The minimum of five interleaved rounds. Each round is the best of five runs of one 1,000-step episode, the same
episode for both. Another session shared the machine, and single runs varied up to twofold.

| Episode | Managed core | MuJoCo 3.15.0, `mj_step(nstep=frame_skip)` from Python | Ratio |
|---|---:|---:|---:|
| HalfCheetah, seed 0 | 5.39 µs | 6.89 µs | 0.78 |
| Hopper, seed 0, no termination | 18.10 µs | 22.24 µs | 0.81 |
| Humanoid, seed 0, no termination (lying down) | 112.07 µs | 89.67 µs | 1.25 |

Humanoid's time goes to PGS: a dense `AR` of `nefc²` entries per step, and sweeps over it. The managed dot products are
scalar code that adds in the AVX lane order.

### 11.7 Time

The session clock, from the request to:
- the first exact Hopper and Humanoid runs: 28 minutes;
- all eleven folders exact, with §11.3's regression found and fixed: 36 minutes;
- coverage, negative controls, the speed comparison and the play check: 47 minutes.

§10.6.4 estimated two to three hours for the two envs. Its time estimate was three to four times too high, and its
line estimate 14% too low.

### 11.8 Still open

- **Walker2d and Ant:** by §10.6's coverage they execute no engine line beyond these three envs. Neither has run on the
  core yet. Each needs its env rules in the checker, and Ant's model must pass `Load`'s validation. (§12.3: both now run
  bit-exact on the core as it is.)
- **The unexecuted branches** of §11.4.
- **§10.3 still holds:** no compiler, no frames, exactness on win-x64 only, and re-verification at each MuJoCo pin.
- **Productizing:** env classes, registration, parity cells and goldens are §10.5's separate cost.

## 12. The remaining envs, measured (2026-11-11)

The maintainer asked for an estimate of the rest. It rests on two measurements:
- **Engine lines:** what each remaining env executes in MuJoCo, against the union of the three envs the core already
  runs (§10.6's method).
- **The core itself on their models:** physics only. The checker starts from Gymnasium's recorded reset, replays the
  recorded actions, and compares every recorded array, without env rules yet.

### 12.1 The ids

Of the 22 ids, 18 remain:
- **The v5s:** Walker2d, Ant, HumanoidStandup, InvertedPendulum, InvertedDoublePendulum, Reacher, Pusher and Swimmer.
- **The v4s of every env but Pusher:** Pusher-v4 raises `ImportError` on MuJoCo 3 (§1), and Gym.NET reproduces that
  refusal with no core at all.

A v4 shares its v5's model, except Walker2d-v4 (`walker2d.xml`). A v4 therefore adds env rules only, not core code.

### 12.2 Engine lines

The plans are Gymnasium's own episodes: 16,000 env steps per env, as in §10.6. The never-terminating envs played
16 × 1,000 steps. Walker2d-v5's and Ant's runs are §10.6's.

| Env | Episodes | Physics steps | Executed engine lines | Beyond the core | What is new |
|---|---:|---:|---:|---:|---|
| Walker2d-v5 | 742 + 16 | 128,000 | 3,483 | 0 | |
| Walker2d-v4 | 742 + 16 | 128,100 | 3,483 | 0 | |
| Ant | 77 + 16 | 160,290 | 3,601 | 0 | |
| HumanoidStandup | 16 | 80,000 | 3,746 | 0 | |
| InvertedPendulum | 2,618 (median 5 env steps) | 32,002 | 1,884 | 0 | |
| InvertedDoublePendulum | 2,491 (median 6) | 80,025 | 2,777 | 5 | Site positions (`mj_kinematics2`); the reward reads `site_xpos` |
| Reacher | 320 | 32,000 | 2,776 | 9 | A second kinematic tree (the islands' union-find, and the warm start of dofs in no island), and simple dofs' constant inertia (`mj_crb`) |
| Swimmer | 16 | 64,000 | 2,811 | 90 | Fluid forces: the inertia-box model (33), `mj_applyFT` (17), `mj_fluid` (14), the body's velocity (9), helpers (17) |
| Pusher | 160 | 80,000 | 4,489 | 928 | The gripper's capsules against the object's cylinder take MuJoCo's general convex collider: GJK and EPA (`engine_collision_gjk.c`, 650) and its glue (`engine_collision_convex.c`, 162). Also plane-cylinder (56), several islands at once (22) and utilities (38) |

The ten envs other than Pusher execute 4,440 engine lines, 103 beyond the core's 4,337. With Pusher, the eleven
execute 5,359.

### 12.3 The core on their models

The checker gained a physics-only mode for envs without env rules. Run on the core as committed:

| Env | Episodes | Physics steps | Result |
|---|---:|---:|---|
| Walker2d-v5 | 50, and 2 without termination | 12,380 | bit-exact |
| Walker2d-v4 | 50, and 1 without termination | 8,432 | bit-exact |
| Ant-v5 | 20, and 1 without termination | 30,995 | bit-exact, `cfrc_ext` included |
| HumanoidStandup-v5 | 2 × 1,000 | 10,000 | bit-exact, `cfrc_ext` included (181,753 PGS sweeps) |
| InvertedPendulum-v5 | 200 | 2,602 | refused: a capsule on the world body |
| InvertedDoublePendulum-v5 | 200 | 6,440 | refused: the same |
| Reacher-v5 | 40 | 4,000 | refused: simple dofs |
| Swimmer-v5 | 1 | 4,000 | refused: fluid forces |
| Pusher-v5 | 10 | 5,000 | refused: simple dofs, then the object's cylinder |

So Walker2d, Ant and HumanoidStandup need no core code: 61,807 physics steps match MuJoCo bit for bit.

A probe copy of the core, outside the repo, made two generalizations:
1. **Non-colliding geoms:** a geom that can never collide (no geom of another body has a compatible bitmask) may be of
   any type and on any body, the world body included. This covers InvertedPendulum's rail and cart and Reacher's arena.
   It changes about five lines of the loader's validation.
2. **Simple dofs:** their mass-matrix diagonal is the model's constant `dof_M0`, as `mj_crb` sets it. That is four lines
   in the core and two fields in the loader.

With them, InvertedPendulum, InvertedDoublePendulum and Reacher match bit for bit too: 13,042 more physics steps,
including Reacher's 246 steps with a joint-limit row. The probe still matches the earlier folders it was run on.

**Islands.** The core solves every constraint as one island over all dofs. Reacher has two kinematic trees, the arm and
the target, and MuJoCo solves the arm's island alone, with its own inertia scale. Reacher still matches for two
reasons:
- its only rows are one joint limit, which Newton solves in a single iteration;
- the target carries no force, so its dofs never move away from `qacc_smooth`.

That is a property of this model, not a proof. Pusher has the arm and the object in separate islands at once, so it
needs the island-local solve.

### 12.4 C# per env

The core estimates come from §11.1's measured ratios. Comparable dense arithmetic converted at 0.44 to 0.96, and
colliders at about 0.9. The env-rule estimates scale each Gymnasium file's code lines by the 0.16 to 0.28 that the
checker's rules for HalfCheetah, Hopper and Humanoid took (20, 30 and 64 lines).

| Ids | Core C# | Basis | Env rules (v5 + v4) |
|---|---:|---|---:|
| Walker2d | 0 | measured: runs as is | ~30 + ~28 |
| Ant | 0 | measured: runs as is | ~45 + ~40 |
| HumanoidStandup | 0 | measured: runs as is | ~50 + ~35 |
| HalfCheetah-v4, Hopper-v4, Humanoid-v4 | 0 | their v5's model | ~18, ~28, ~45 |
| InvertedPendulum | ~5 | measured: the geom generalization, which the next two need as well | ~12 + ~10 |
| InvertedDoublePendulum | ~11 | site positions for the reward, estimated from 5 engine lines; its physics measured | ~20 + ~18 |
| Reacher | ~6 | measured: simple dofs | ~20 + ~18 |
| Swimmer | ~70 | 90 engine lines of fluid arithmetic, no iteration | ~20 + ~18 |
| Pusher-v5 | ~1,050 | GJK, EPA and their glue (812 engine lines, at about 1.0: MuJoCo's code is already compact geometry), plane-cylinder (~60), the island-local solve (~150: its engine lines already run for one island, but the core specialized them away), utilities (~20) | ~30 |
| **Total** | **~1,140** | | **~490** |

Without Pusher, the core grows by about 90 lines, to about 2,360, and runs 20 of the 22 ids (Pusher-v4 refuses as
Gymnasium does). Pusher-v5 takes it to about 3,400.

The env layers hold no hidden BLAS order. Reacher and Pusher reward with `np.linalg.norm` of 3-vectors, which is the
square root of the left-to-right sum of squares, with no fused multiply-add, in 20,000 of 20,000 trials. Their float32
control costs of 2 and 7 actions also sum left to right.

### 12.5 Time

Estimated at §11.7's pace: Hopper and Humanoid, 594 core lines and 187 checker lines, were exact in 36 minutes. The
envs are grouped by the work they share:

| Work | Ids | Core C# | Env rules | Session time |
|---|---:|---:|---:|---|
| Walker2d, Ant, HumanoidStandup (v4 and v5); HalfCheetah-v4, Hopper-v4, Humanoid-v4 | 9 | 0 | ~320 | 45–90 minutes |
| InvertedPendulum, InvertedDoublePendulum, Reacher (v4 and v5) | 6 | ~20 | ~100 | 20–40 minutes |
| Swimmer (v4 and v5) | 2 | ~70 | ~40 | 20–40 minutes |
| Pusher-v5 | 1 | ~1,050 | ~30 | 2–4 hours |
| **All 18** | **18** | **~1,140** | **~490** | **about 3.5–6.5 hours** |

Without Pusher, the 17 other ids take about 1.5–3 hours.

### 12.6 Risks

1. **Pusher's convex collider:** GJK and EPA iterate on simplices and polytopes, with many branches. Exactness rests on
   reproducing each branch in MuJoCo's order. At 812 executed lines it is the largest single piece, and coverage shows
   only the branches 16,000 random steps reach.
2. **Islands:** the island-local solve is new C# for engine lines the core already reproduces for one island: the order
   of dofs and rows, the sparse sub-matrix, the per-island scale. Reacher shows the one-island shortcut survives only by
   a property of its model.
3. **Swimmer's fluid forces:** deterministic arithmetic with no iteration. Low risk.
4. **The env layers:** each v4 has its own observation and reward, and NumPy's reductions to match per env, as §11.5's
   controls showed for the first three. Pusher's reset also draws the object's position in a rejection loop of
   `uniform` calls.
5. **Productizing** (env classes, registration, parity cells, goldens, frames) remains §10.5's separate cost.

## 13. All eleven envs on the same core (2026-11-11)

The maintainer asked for the rest, implemented to completion, with parity covered to completion. The proof of concept's
core (`tools/parity/mujoco/halfcheetah_poc/`) now steps every id that runs on MuJoCo 3: the 22 ids but `Pusher-v4`,
which Gymnasium itself refuses on MuJoCo 3 (§1), so there is nothing to step. The checker holds the env rules of all 21.
The work stays a proof of concept outside `Gym.NET.sln`: P9 is still postponed (DR-35), and §7's decisions still
precede a product.

### 13.1 What was added

The core grew by 1,512 lines of code, to 3,776 (5,666 with docs). §12.4 estimated about 1,140, so the estimate was 33%
low, almost all of it in the convex collider. Counted per member against §12's commit:

| Area | Code lines added | §12.4 estimate | Contents |
|---|---:|---:|---|
| GJK and EPA (`Ccd` in `Kernels.cs`) | 1,012 | ~812, with the glue | The distance query (GJK with the signed-volume subalgorithm and the intersection fallback), the initial polytopes from 2, 3 or 4 vertices, EPA with its horizon, the witness points, the support functions of a point, segment, sphere, capsule and cylinder |
| The convex collider's glue | 76 | (in the line above) | `mjc_Convex`'s one query with a margin, the capsule shrunk to its segment and inflated back, MultiCCD's four perturbed queries and their distinctness test |
| Plane-cylinder | 72 | ~60 | `mjc_PlaneCylinder`: the near rim point, the far one, and for a disk flat on the plane the two side points |
| Islands | 148 | ~150 | The union-find over trees, the maps of dofs and rows, each island's problem gathered (the sparse mass matrix and its factor, renumbered), solved by Newton with its own scale, and scattered back; the dofs in no island reset to their unconstrained acceleration |
| Fluid forces | 86 | ~70 | The inertia-box model, the body's local velocity, the wind's transform, `mj_applyFT` |
| Sites, simple dofs, the collider dispatch, two 3x3 helpers | 24 | ~20 | InvertedDoublePendulum's reward reads a site; Reacher's and Pusher's slides take `dof_M0` |
| Fields, constructor, loader and validation | 94 | | The new arrays; `Load` admits cylinders, several trees, fluid and sites, and refuses, by name, what the core does not do (§13.8) |
| **Total** | **1,512** | **~1,140** | |

The checker's env rules moved into `EnvRules.cs`: 450 lines of code for all 21 ids, including §11's three, which
`Program.cs` held until now (it fell from 503 to 319 lines). §12.4 estimated about 490 lines for the 18 new ids alone.
The 21 took fewer because a v4 and its v5 share one class, and a base class holds the resets, NumPy's sums and the info
plumbing. `oracle.py` records format 3: site frames, body positions and the `TimeLimit`, which the rules read.

`verify/` holds what §13.2–§13.6 ran: the trace list and its recorder, the coverage summary, the branch search, the
negative controls and MuJoCo's timing.

### 13.2 Results

The trace set (`verify/traces.txt`, 45 folders, about 1.6 GB) plays every id as Gymnasium does. It adds long episodes
without termination for the four envs that can fall, the planar HalfCheetah upside down, and the eight targeted
episodes of §13.4:

| Id | Folders | Episodes | Env steps | Physics steps | Steps with constraint rows | Solver iterations |
|---|---:|---:|---:|---:|---:|---:|
| HalfCheetah-v5 | 2 | 7 | 7,000 | 35,000 | 20,803 | 28,110 |
| HalfCheetah-v4 | 2 | 4 | 4,000 | 20,000 | 11,799 | 16,036 |
| Hopper-v5 | 4 | 105 | 7,267 | 29,068 | 25,012 | 30,007 |
| Hopper-v4 | 2 | 102 | 4,267 | 17,068 | 14,294 | 16,675 |
| Walker2d-v5 | 2 | 103 | 5,147 | 20,588 | 18,425 | 29,413 |
| Walker2d-v4 | 2 | 102 | 4,164 | 16,656 | 14,917 | 24,047 |
| Ant-v5 | 2 | 32 | 9,158 | 45,790 | 45,001 | 89,248 |
| Ant-v4 | 2 | 31 | 8,158 | 40,790 | 40,077 | 79,423 |
| Humanoid-v5 | 2 | 103 | 5,347 | 26,735 | 22,998 | 379,659 (PGS) |
| Humanoid-v4 | 2 | 102 | 4,347 | 21,735 | 18,102 | 283,549 (PGS) |
| HumanoidStandup-v5 | 1 | 3 | 3,000 | 15,000 | 14,756 | 269,725 (PGS) |
| HumanoidStandup-v4 | 1 | 2 | 2,000 | 10,000 | 9,857 | 181,753 (PGS) |
| InvertedPendulum-v5 | 1 | 300 | 1,911 | 3,822 | 0 | 0 |
| InvertedPendulum-v4 | 1 | 300 | 1,911 | 3,822 | 0 | 0 |
| InvertedDoublePendulum-v5 | 1 | 300 | 1,944 | 9,720 | 0 | 0 |
| InvertedDoublePendulum-v4 | 1 | 300 | 1,944 | 9,720 | 0 | 0 |
| Reacher-v5 | 1 | 100 | 5,000 | 10,000 | 581 | 585 |
| Reacher-v4 | 1 | 100 | 5,000 | 10,000 | 581 | 585 |
| Swimmer-v5 | 1 | 5 | 5,000 | 20,000 | 1,591 | 1,556 |
| Swimmer-v4 | 1 | 3 | 3,000 | 12,000 | 956 | 940 |
| Pusher-v5 | 13 | 356 | 35,600 | 178,000 | 178,000 | 51,044 |
| **Total** | **45** | **2,460** | **125,165** | **555,514** | **437,750** | **1,482,355** |

The pendulums never reach a joint limit in these episodes, which end first.

| Check | Result |
|---|---|
| Reset noise and the 125,165 sampled actions, drawn through NumSharp's `Generator` (Pusher's rejection loop included) | bit-exact |
| Closed loop: per physics step 30 arrays, 47 in debug episodes; per env step `cfrc_ext`, tendons, site frames and body positions | 18,265,118 of 18,265,118 comparisons bit-exact |
| Resync, each physics step from MuJoCo's recorded state | bit-exact |
| Env layer of all 21 ids: observation, reward, terminated, truncated, every info entry | bit-exact |
| The playable HalfCheetah, built against the extended core: its autopilot's returns on seeds 0–9 | identical to the last bit |

### 13.3 What the new envs needed

- **Islands, as MuJoCo solves them.** Pusher has three kinematic trees: the arm, the object and the goal. MuJoCo
  solves each constrained island alone, with its own dofs, rows and inertia scale, so the core gathers each island's
  problem, solves it and scatters the result. Scaling Newton by the model's inertia instead breaks Pusher at physics
  step 436 (§13.5). MuJoCo also resets the dofs no island holds to their unconstrained acceleration after choosing the
  warm start. None of the searched episodes could show that reset (§13.5); the core does it anyway.
- **The convex collider.** Pusher's capsules meet its object's cylinder through `mjc_Convex`. Each pair makes one GJK
  query with the margin, on the capsule shrunk to its segment, inflated back at the end. A deep penetration goes on to
  EPA, from the polytope GJK's last simplex gives. MultiCCD then repeats the query four times with the geoms rotated by
  ±0.001 rad about the contact's tangents, and keeps each new contact far enough from the others. EPA runs in 1.6% of
  Pusher's episodes. EPA from a triangle (`polytope3`) runs in 0.18%, and its arithmetic shows in the contact in fewer
  still, so its seeds were found by search and recorded on their own (§13.4).
- **Plane-cylinder.** The object only slides, so its cylinder always stands upright on the table. That is MuJoCo's
  flat-disk case: three contacts on the bottom rim. The branches for a tilted or lifted cylinder are unreachable in
  Gymnasium's envs (§13.4).
- **Fluid forces** (Swimmer): MuJoCo's left-to-right products, and `mj_applyFT`'s order (the force's term, then the
  torque's). The order shows at physics step 1 (§13.5).
- **The env layers.** InvertedDoublePendulum's distance penalty squares with NumPy's float64 `**`, which is C's `pow`
  and differs from a product in 0.05% of values. Reacher-v4 prices the state before the step. Pusher's arm starts at its
  reference pose with no noise, and its object is drawn by rejection until it lies farther than 0.17 from the goal. Ant's health
  range is inclusive. A v4's healthy reward is `float(healthy or terminate)`. HumanoidStandup-v4 caps the impact cost
  with `min` where v5 clips it. The body positions an env reads after `mj_step` are those of RK4's last stage, or under
  Euler those from before the integration.

### 13.4 What the traces execute

Line coverage of the managed core over the 45 folders (`dotnet-coverage`, Debug build, executable lines):
- `CheetahCore.cs`: 1,781 of 1,868 lines;
- `Kernels.cs`: 1,149 of 1,339;
- `CheetahModel.cs`: 242 of 243;
- `EnvRules.cs`: 369 of 377. The eight are the arm for an id without rules, and NumPy's sums at lengths no env uses:
  a float32 sum beyond 8 entries, a float64 sum below 8.

Random actions reach rare branches rarely, so coverage alone cannot tell an unreachable branch from an unlucky set of
traces. A branch search settles it as far as search can. `verify/branch_search.py` puts a probe at the start of every
run of unexecuted lines in a copy of the core, 86 of the 92 runs. It cannot place the other six (statements under an
unbraced header, switch arms):
- two get probes on their conditions instead;
- one is covered by `polytope2`'s own probe;
- three cannot run here: a minimum over several contacts, a second shrunk geom, and an empty simplex.

It also probes conditions no single line shows, such as the island warm start's reset changing a value. Its `Main` then plays
seeded episodes as oracle.py plays them, through the core alone, without MuJoCo. Any seed that reaches a probe is then
recorded with oracle.py and checked. The probes must change nothing, so the probed copy is first built as the checker
and run over the trace set: 45 of 45 bit-exact. That check exists because the first version broke it: a probe placed
before a statement that a multi-line condition guarded made the statement unconditional, so `polytope3` always failed
in the probed copy. Its own output showed this, and Pusher was searched again after the fix.

| Env | Episodes searched | Physics steps | Unexecuted branches reached |
|---|---:|---:|---|
| HalfCheetah-v5, and upside down | 3,000 + 300 | 16.5 million | none |
| Hopper-v5, as played and lying down | 30,000 + 1,200 | 7.5 million | two capsules with parallel axes, in seeds 109, 280, 552, 668 and 697 lying down |
| Walker2d-v5 and -v4, as played and lying down | 60,800 | 8.3 million | none |
| Ant-v5, as played and lying down | 2,300 | 3.2 million | none |
| Humanoid-v5, as played and lying down | 20,300 | 3.9 million | none |
| HumanoidStandup-v5 | 150 | 0.75 million | none |
| InvertedPendulum-v5, InvertedDoublePendulum-v5 | 80,000 | 1.8 million | none |
| Reacher-v5 | 40,000 | 4.0 million | none |
| Swimmer-v5 | 1,000 | 4.0 million | none |
| Pusher-v5 | 80,000 | 40 million | none. EPA ran in 1,292 episodes and `polytope3` in 143; every one of its 663 calls went on into EPA |
| **Total** | | **about 90 million** | |

Two of the parallel-axes seeds (109 and 280) and six `polytope3` seeds (654, 1559, 2455, 20432, 40228 and 60390) were
recorded on their own: the targeted lines of `traces.txt`, bit-exact in §13.2. Coverage confirms that the parallel-axes
branch ran in them.

The lines still never executed fall in two groups. Each follows MuJoCo's source but is unproven:
- **No model here can reach them:**
  - geom priorities, direct (negative) or mixed-sign solref, and an impedance of power 1;
  - a body of zero subtree mass, the undamped Euler step, and a body welded to the world in the fluid;
  - a tilted or lifted cylinder;
  - the union-find's other merge orders. Pusher's arm (tree 0) only ever joins its object (tree 1), whose table
    contact has already activated it, and the goal never collides;
  - the convex collider's paths for spheres, for two shrunk geoms, for zero-margin multi-contact and for zero
    contacts;
  - the loader's own refusals.
- **No search reached them:**
  - the auto-reset of a bad state or bad controls;
  - coincident sphere centres, two parallel capsules touching at both ends, and a contact in the gap zone;
  - PGS's cost-rise revert (dead by construction, §11.4);
  - Newton's fallbacks: a zero step, an exhausted or stuck line search, a lost rank in the factor update;
  - the collider's degenerate exits: EPA from a segment (`polytope2`), a degenerate simplex or face, and EPA's
    failure exits;
  - the kernels' degenerate cases: tiny pivots, zero-length vectors and quaternions, a zero rotation angle.

### 13.5 Negative controls

`verify/negative_controls.py` applies each change to a copy of the project and runs the checker on the traces named.
Each control expects one of two verdicts. "Detected" means the checker reports a difference. "Same bits" applies to a
change that cannot move a recorded bit on these models; its reason was measured, not assumed.

| Change | Traces | Verdict |
|---|---|---|
| Newton's scale from the whole model's inertia, not the island's | Pusher (7), Reacher (2) | Detected: Pusher, physics step 436, `qacc[7]`. Reacher stays exact: its one limit row solves in one iteration |
| Without the upright cylinder's two side contacts | Pusher | Detected at step 0: 1 contact, MuJoCo 3 |
| Without MultiCCD's perturbed contacts | Pusher | Detected: step 418, `qpos[0]` |
| S2D's cofactor `m14` summed in another order | Pusher | Detected: step 290, 6 ulp in `qpos[7]` |
| `polytope3`'s normal from the triangle's other edge | Pusher, with the targeted seeds | Detected: seed 1559, step 479, 20 ulp. The broad set's three `polytope3` episodes stay exact |
| `mj_applyFT` adding the torque's term before the force's | Swimmer | Detected at step 1: 2 ulp in `qfrc_passive[2]` |
| InvertedDoublePendulum's squares as products, not NumPy's `**` | InvertedDoublePendulum | Detected: env layer, step 2, 1 ulp |
| Reacher-v4 priced after the step, like v5 | Reacher-v4 | Detected: env layer, step 0 |
| Ant's float32 control cost summed left to right | Ant | Detected: env layer, step 0 |
| Pusher's object kept at its first draw | Pusher | Detected at the reset |
| Pusher's reward summed in another order | Pusher | Detected: env layer, step 2, 1 ulp |
| The dofs in no island keep their warm start | Pusher, Reacher | Same bits. The search saw 7.3 million Pusher steps and 0.64 million Reacher steps keep the warm start while a tree was in no island. In every one, the free dofs' warm start already equalled `qacc_smooth`: the goal and the target never move. When Pusher's arm is free with another warm start, the smooth start wins and resets it anyway |
| S2D taking an exact tie | Pusher | Same bits. Ties are common, 59,516 of 425,129 edge comparisons in 20,000 episodes, but every one falls at the two edges' shared vertex, where both give the same weights |
| `polytope3`'s two support points taken in the other order | Pusher, with the targeted seeds | Same bits. The order only numbers the two vertices; the points and EPA's faces are the same |
| Swimmer's face drag regrouped, `abs(v)*v` first | Swimmer | Same bits. The drag's last bit changes in 32% of evaluations, but it is small against the viscous force it is subtracted from, which never changes |
| The zero wind not transformed | Swimmer | Same bits. The transform only signs zeros. 3% of the local velocity components are exactly zero, because Swimmer is planar, yet the subtraction changed no bit in 28.8 million |
| Simple dofs' diagonal computed, not `dof_M0` | Pusher, Reacher | Same bits: armature plus mass along an axis-aligned slide, compared in 561,200 evaluations |

### 13.6 Speed

The minimum of five interleaved rounds. Each round is the best of five runs of one episode, the same episode for both:
1,000 env steps for HalfCheetah and Swimmer, and for Hopper, Walker2d, Ant and Humanoid without termination. The others
use their first recorded episode.

| Episode | Managed core | MuJoCo 3.15.0, `mj_step(nstep=frame_skip)` from Python | Ratio |
|---|---:|---:|---:|
| HalfCheetah-v5, seed 0 | 5.47 µs | 6.84 µs | 0.80 |
| Hopper-v5, seed 0, no termination | 16.36 µs | 21.20 µs | 0.77 |
| Walker2d-v5, seed 0, no termination | 26.42 µs | 29.63 µs | 0.89 |
| Ant-v5, seed 0, no termination | 47.56 µs | 47.03 µs | 1.01 |
| Humanoid-v5, seed 0, no termination (lying down) | 85.43 µs | 88.43 µs | 0.97 |
| HumanoidStandup-v5, seed 0 | 86.40 µs | 87.24 µs | 0.99 |
| InvertedPendulum-v5, seed 0 (5 env steps) | 3.33 µs | 8.87 µs | 0.38 |
| InvertedDoublePendulum-v5, seed 0 (6 env steps) | 4.61 µs | 9.65 µs | 0.48 |
| Reacher-v5, seed 0 (50 env steps) | 5.74 µs | 10.75 µs | 0.53 |
| Swimmer-v5, seed 0 | 8.81 µs | 12.07 µs | 0.73 |
| Pusher-v5, seed 0 (100 env steps) | 5.91 µs | 7.26 µs | 0.81 |

On the smallest models, Python's one call per env step (about a microsecond, over two to five physics steps) is a large
share of MuJoCo's time. §11.6 measured Humanoid at 112 µs while another session shared the machine; these rounds
measured 85 µs.

### 13.7 Time

The session clock, from the request (16:54 UTC) to:
- the env rules of all 21 ids written, and the 18 ids that need neither fluid forces nor a new collider bit-exact at
  every layer: 15 minutes;
- Swimmer, with fluid forces: 19 minutes;
- the island-local solve: 27 minutes;
- Pusher, with GJK, EPA, MultiCCD and plane-cylinder: 40 minutes;
- the 37 folders of the broad set bit-exact: 46 minutes;
- coverage, the branch search, the targeted traces, the controls, the speed and the full documentation: 1 hour 52
  minutes.

§12.5 estimated 3.5 to 6.5 hours for the 18 ids. The implementation took 40 minutes, so the estimate was five to ten
times too high. Its line count was 33% too low.

### 13.8 Still open

- **Productizing.** No env classes, registration, parity cells, goldens or frames. P9 stays postponed (DR-35), and
  §7's decisions come first: the MuJoCo pin, specialized core or general port, the renderer.
- **§10.3 still holds.** There is no compiler: the models come from MuJoCo's own compiled dump. Exactness holds on
  win-x64 only, and needs re-verification at each MuJoCo pin.
- **The unexecuted branches** of §13.4.
- **What `Load` refuses:** colliding geoms other than planes, spheres, capsules and cylinders; a capsule-cylinder pair
  without a margin (MuJoCo's multi-contact polygon clipping); ball joints; equality constraints; active tendons;
  elliptic cones; the ellipsoid fluid model; PGS over several trees; and stateful actuators. Gymnasium's eleven models
  need none of them. Their own `xml_file` variants might.

## 14. The core on MuJoCo 4.0.0 (2026-11-12)

Round 21 of `DESIGN_REVIEW.md` made the proof of concept's core the MuJoCo envs (DR-61) and pinned MuJoCo 4.0.0
(DR-64), released 2026-11-06. Gymnasium v1.4.0 runs all 21 ids on it. The core had been written against 3.15.0, so it
was verified again. Traces were recorded with mujoco 4.0.0 in an isolated venv, 4.0.0's sources were compared with
3.15.0's function by function, and the core was changed wherever 4.0.0 rounds differently.

### 14.1 What changed

A first set of 18 traces, one or two per id, showed five changes outside the collider:
- **RK4** publishes the step's weighted acceleration as `qacc` and warm-starts the next step from the last stage's
  solve. A free joint's angular velocity gets the RKMK correction at each stage (`mju_derivExpmap`, with the Bortz
  coefficient series).
- **The AVX quaternion kernels:** `mju_mulQuat` sums its products in pairs, `mju_quat2Mat` groups the diagonal's
  squares, and `mju_normalize4` adds its squares in pairs. They move every model's kinematics in the last bit.
- **The dense `mju_cholSolve`** substitutes backward by columns: each solved entry is scattered into the rows above.
- **PGS** (Humanoid, HumanoidStandup) scales its improvement test by the island's inertia trace, where 3.15 used the
  model's mean inertia.
- **Capsule pairs** go through `mjc_segWeights`: Ericson's Voronoi regions, Lagrange's determinant and the
  Binet-Cauchy weight, with two pairs within sin² = 1e-8 of parallel. 3.15 solved the axes' 2×2 system.

After them, 15 of the 18 were exact. The three Pusher traces needed **4.0.0's convex collider**, rewritten as a whole:
- GJK with discrete geoms, the Frank-Wolfe gap, exact plane and line projections and a final separation check;
- the signed-volume subalgorithms in their new form;
- the initial polytopes, with three new helpers (`replaceSimplex2`, `recoverHexa`, `originOnEdge`), and EPA's
  recursive horizon;
- `inflate` with an explicit normal.

`maxContacts` now asks a capsule's edge against a face for two contacts in one pass, even with a margin.
`mjc_multicontact` then clips the capsule's edge against the cylinder's disk, a 16-gon, and MultiCCD's perturbations
no longer run for that pair. The new collider replaced the old one's 1,520 lines. The loader no longer refuses a
capsule-cylinder pair without a margin, which 4.0.0 treats the same way.

### 14.2 Results

The 45 folders of `verify/traces.txt`, recorded again with mujoco 4.0.0, and four new targeted ones (§14.3). Episodes
are the same seeds as §13.2, but their lengths follow 4.0.0's physics: Ant's falls come sooner.

| Id | Folders | Episodes | Env steps | Physics steps | Steps with constraint rows | Solver iterations |
|---|---:|---:|---:|---:|---:|---:|
| HalfCheetah-v5 | 2 | 7 | 7,000 | 35,000 | 21,146 | 28,566 |
| HalfCheetah-v4 | 2 | 4 | 4,000 | 20,000 | 12,347 | 16,842 |
| Hopper-v5 | 4 | 105 | 7,267 | 29,068 | 25,012 | 30,007 |
| Hopper-v4 | 2 | 102 | 4,267 | 17,068 | 14,294 | 16,675 |
| Walker2d-v5 | 2 | 103 | 5,147 | 20,588 | 18,445 | 29,428 |
| Walker2d-v4 | 2 | 102 | 4,164 | 16,656 | 14,922 | 24,034 |
| Ant-v5 | 2 | 32 | 5,470 | 27,350 | 26,754 | 52,154 |
| Ant-v4 | 2 | 31 | 4,470 | 22,350 | 21,825 | 42,400 |
| Humanoid-v5 | 2 | 103 | 5,347 | 26,735 | 22,982 | 377,329 (PGS) |
| Humanoid-v4 | 2 | 102 | 4,347 | 21,735 | 18,089 | 282,063 (PGS) |
| HumanoidStandup-v5 | 1 | 3 | 3,000 | 15,000 | 14,722 | 273,808 (PGS) |
| HumanoidStandup-v4 | 1 | 2 | 2,000 | 10,000 | 9,797 | 183,857 (PGS) |
| InvertedPendulum-v5 | 1 | 300 | 1,911 | 3,822 | 0 | 0 |
| InvertedPendulum-v4 | 1 | 300 | 1,911 | 3,822 | 0 | 0 |
| InvertedDoublePendulum-v5 | 1 | 300 | 1,944 | 9,720 | 0 | 0 |
| InvertedDoublePendulum-v4 | 1 | 300 | 1,944 | 9,720 | 0 | 0 |
| Reacher-v5 | 1 | 100 | 5,000 | 10,000 | 581 | 585 |
| Reacher-v4 | 1 | 100 | 5,000 | 10,000 | 581 | 585 |
| Swimmer-v5 | 1 | 5 | 5,000 | 20,000 | 1,591 | 1,556 |
| Swimmer-v4 | 1 | 3 | 3,000 | 12,000 | 956 | 940 |
| Pusher-v5 | 17 | 360 | 36,000 | 180,000 | 180,000 | 51,769 |
| **Total** | **49** | **2,464** | **118,189** | **520,634** | **404,044** | **1,412,598** |

| Check | Result |
|---|---|
| Reset noise and the 118,189 sampled actions, through NumSharp's `Generator` | bit-exact |
| Closed loop: 17,228,202 array comparisons per pass | bit-exact |
| Resync, each physics step from MuJoCo's recorded state | bit-exact |
| Env layer of all 21 ids: observation, reward, terminated, truncated, every info entry | bit-exact |

### 14.3 What the traces execute

Line coverage over the 49 folders (Debug build): `CheetahCore.cs` 1,761 of 1,906 lines, `Kernels.cs` 1,554 of 2,014,
`CheetahModel.cs` 240 of 241, `EnvRules.cs` 369 of 377.

The branch search of §13.4 was aimed at 4.0.0's code: new anchors, probes on the multi-contact pass, and probes on
`mjc_segWeights`' parallel test. The probed copy, built as the checker, gave 45 of 45 bit-exact first.

| Env | Episodes searched | Physics steps | Found |
|---|---:|---:|---|
| Pusher-v5 | 80,000 | 40 million | EPA in 1,232 episodes: from GJK's tetrahedron (`polytope4`) 32,418 times, from its triangle (`polytope3`) 141 times in 104 episodes. The multi-contact pass after 51,681 shallow contacts and after 32,559 of EPA's; it clipped the capsule's edge 3,778 times in 576 episodes, pruned a clipped point beyond the margin in 88, and kept two points or fewer as they were in 119 |
| Hopper-v5, no termination | 1,600 | 6.4 million | Two capsules within the 1e-8 parallel band, never exactly parallel: 2,630 times in 909 episodes |
| Hopper-v5, as played | 30,000 | 2.7 million | none |
| Walker2d-v5, no termination | 800 | 3.2 million | none |

No episode of the broad set clips an edge: the first seeds that do are 94, 418, 512 and 654. Four seeds were recorded
on their own and are bit-exact: 94 (all three clip paths, at physics step 384), 418, 2851 and 3016. Seed 654, a
targeted line of §13.4, clips its edge on 4.0.0. The broad set's `hop5_nt` (seeds 0 and 1) already lies in the
parallel band, so 3.15.0's targeted Hopper seeds are kept only as a record.

The lines no trace executes and no search reached, new with 4.0.0:
- EPA from a segment (`polytope2`, `replaceSimplex2`, `originOnEdge`);
- the branches of `recoverHexa`, and the failure exits and fallbacks of `polytope3` and `polytope4`;
- `hull4`, more than four clipped points;
- the degenerate segment weights, a capsule of zero length;
- the large angles of `DerivExpmap` and `BortzCoef`: a rotation beyond π, or beyond 1.75 rad, in one step.

Two parts are unreachable by construction:
- MultiCCD's perturbation code, since no pair of these models asks the collider for a single contact on 4.0.0;
- the multi-contact pass's face cases, since a capsule has no face.

§13.4's other unexecuted lines are unchanged.

### 14.4 Negative controls

`verify/negative_controls.py` now holds 32 controls, run on the 4.0.0 traces. All 32 gave their expected verdict.
§13.5's controls on code that 4.0.0 replaced were re-aimed, and 15 are new. Most of the new ones put back what 3.15.0
did:

| Change | Traces | Verdict |
|---|---|---|
| MuJoCo 3's one contact and MultiCCD for the capsule-cylinder pair | Pusher | Detected: `pus5_c`, step 426, `qpos[0]` |
| The multi-contact pass skipped after a shallow contact | Pusher, with the targeted seeds | Detected: seed 94, step 384 |
| The multi-contact pass skipped after EPA's contact | Pusher, with the targeted seeds | Detected: seed 654, step 270 |
| S2D's weights divided by the largest minor, as in 3.15 | Pusher | Detected: step 343, 1 ulp in `qvel[2]` |
| `polytope3`'s normal from the triangle's other edge | Pusher, targeted | Detected: `pus5_c`, step 386, 180 ulp |
| RK4 without the free joints' correction | Ant, Humanoid, HumanoidStandup | Detected at step 0 |
| RK4 publishing the last stage's `qacc`, as in 3.15 | The RK4 envs | Detected at step 0 |
| RK4 warm-starting from the first stage | The RK4 envs | Detected at step 0 |
| `mju_quat2Mat`'s diagonal left to right, as in 3.15 | Ant, HalfCheetah | Detected: step 0, 5 ulp |
| `mju_mulQuat` left to right, as in 3.15 | Ant, HalfCheetah | Detected: step 0, 1 ulp |
| `mju_normalize4` left to right, as in 3.15 | Ant, Humanoid, HumanoidStandup | Detected: step 387, 1 ulp |
| The dense `mju_cholSolve` substituting backward by rows, as in 3.15 | Hopper, Walker2d, Ant, Pusher | Detected: step 9, 3 ulp |
| PGS scaled by the mean inertia, as in 3.15 | Humanoid, HumanoidStandup | Detected: step 93 |
| One pair for parallel capsules | Hopper lying down | Detected: `hop5_nt`, step 3828 |
| The parallel band reduced to exact parallels | Hopper lying down | Detected: `hop5_nt`, step 3828 |
| The clipped contacts not inflated by the cylinder's half margin | the clipping seeds | Detected: seed 94, step 384 |
| A clipped edge keeping its first two points, not the two farthest apart | the clipping seeds | Detected: seed 654, step 267, 1 ulp |
| Clipped points beyond the margin kept | the clipping seeds | Detected: seed 94, step 384, 6 contacts where MuJoCo has 5 |

§13.5's other controls give the same verdicts on 4.0.0's traces. Their "same bits" reasons were measured on 3.15.0
and still hold, for two reasons:
- Two are structural: an exact S2D tie falls at the shared vertex, and swapping `polytope3`'s supports only renumbers
  two vertices.
- The search re-measured a third: the warm start was kept while a tree was in no island 7.3 million times in 80,000
  Pusher episodes, and a free dof's warm start never differed from `qacc_smooth`.

### 14.5 Speed

§13.6's method: the minimum of five interleaved rounds, each the best of five runs of the same episode on both sides,
against mujoco 4.0.0 from Python.

| Episode | Managed core | MuJoCo 4.0.0, `mj_step(nstep=frame_skip)` from Python | Ratio |
|---|---:|---:|---:|
| HalfCheetah-v5, seed 0 | 7.54 µs | 9.98 µs | 0.76 |
| Hopper-v5, seed 0, no termination | 21.76 µs | 30.43 µs | 0.72 |
| Walker2d-v5, seed 0, no termination | 34.64 µs | 40.93 µs | 0.85 |
| Ant-v5, seed 0, no termination | 59.46 µs | 60.59 µs | 0.98 |
| Humanoid-v5, seed 0, no termination | 121.29 µs | 115.07 µs | 1.05 |
| HumanoidStandup-v5, seed 0 | 131.79 µs | 114.96 µs | 1.15 |
| InvertedPendulum-v5, seed 0 | 5.41 µs | 12.67 µs | 0.43 |
| InvertedDoublePendulum-v5, seed 0 | 7.04 µs | 13.94 µs | 0.51 |
| Reacher-v5, seed 0 | 8.34 µs | 15.55 µs | 0.54 |
| Swimmer-v5, seed 0 | 12.09 µs | 16.46 µs | 0.73 |
| Pusher-v5, seed 0 | 7.76 µs | 10.10 µs | 0.77 |

Both sides ran 30 to 45% slower than in §13.6's rounds on the same machine, which was busier this time, so the ratios
are what compares. They moved little, except for the two Humanoids, whose PGS sweeps now take 4% and 15% longer than
MuJoCo's. The episodes also differ, because 4.0.0's physics changes where the falls happen.

### 14.6 Still open

- **The branches of §14.3** that no trace and no search reached.
- **Productizing**, in Round 21 and 22's terms: the env classes, the registration, the embedded models (DR-63), the
  `Gym.MuJoCo` surface (DR-68), the parity stack on mujoco 4.0.0 with its live cells, and the frames (DR-62, DR-67).
- **§10.3 still holds** for exactness: proven on win-x64, and to be verified again at each MuJoCo pin.

## 15. The Python API Gymnasium reaches, compared (2026-11-12)

Round 22 of `DESIGN_REVIEW.md` deferred how users reach MuJoCo's model and data: "Defer, compare to python apis and
suggest these again". This section is that comparison. It counts what Gymnasium v1.4.0 calls in MuJoCo's Python
bindings, what the bindings offer, and what the core holds. It ends with the options, revised.

### 15.1 What Gymnasium calls

Counted in `refs/Gymnasium`: the 22 env modules, `mujoco_env.py`, `utils.py`, the renderer `mujoco_rendering.py`, and
the three MuJoCo test modules.

| Where | `MjModel` | `MjData` | Functions and types |
|---|---|---|---|
| The envs and `MujocoEnv` | `nq`, `nv`, `nu`, `na`, `opt.timestep`, `actuator_ctrlrange`, `body_mass` (Humanoid's `mass_center`), `vis.global_.offwidth` and `offheight` (grown to the render size) | `qpos` (61 uses), `qvel`, `ctrl`, `act` (cleared when `na` is 0), `cfrc_ext`, `cvel`, `cinert`, `qfrc_actuator`, `qfrc_constraint` (InvertedDoublePendulum), `ten_length` and `ten_velocity` (Humanoid's and HumanoidStandup's info), `site_xpos`, `xipos`, `body(name).xpos` | `MjModel.from_xml_path`, `MjData`, `mj_step(model, data, nstep)`, `mj_forward`, `mj_resetData`, `mj_rnePostConstraint` |
| `utils.py` | | | `mj_stateSize`, `mj_getState`, `mj_setState`, `mjtState` (`mjSTATE_FULLPHYSICS`, `mjSTATE_INTEGRATION`) |
| The renderer | `ncam`, `stat.extent`, `geom_rgba` (written: made translucent and back), `opt.timestep` | `geom_xpos`, `time`, `solver_niter` | `MjvScene`, `MjvOption`, `MjvCamera`, `MjvPerturb`, `MjrContext`, `MjrRect`, `mjv_updateScene`, `mjv_moveCamera`, `mjv_initGeom`, `mjr_render`, `mjr_readPixels`, `mjr_setBuffer`, `mjr_overlay`, `mj_name2id`, 12 enum types, the OpenGL contexts (`glfw`, `egl`, `osmesa`) |
| The tests | `nbody`, `njnt`, `ngeom`, `ntendon`, `nbvh`, `nq`, `nv`, `nu`, `body_mass`, `opt.timestep`, `vis.global_` | `qpos`, `qvel`, `xipos`, `site_xpos`, `body(name).xpos`, and the shapes of `cfrc_ext`, `qfrc_actuator`, `cvel`, `cinert` | `mj_kinematics`, `mj_step`, `mj_forward`, `MjModel.from_xml_path` (custom models among them), `mujoco.Renderer` |

Gymnasium writes as well as reads. `set_state` assigns `data.qpos[:]` and `data.qvel[:]`, `do_simulation` assigns
`data.ctrl[:]`, and `MujocoEnv` grows `model.vis.global_.offwidth`. The renderer divides `model.geom_rgba[:, 3]` by 5
and back. Python's arrays are views of MuJoCo's own memory, so every such write reaches the simulation.

### 15.2 What the bindings offer, and what the core holds

MuJoCo 4.0.0's Python bindings, counted by introspection, are far larger:
- `MjModel` has 625 public attributes and `MjData` 238.
- The module has 485 callables: 133 `mj_*`, 110 `mju_*` utilities, 29 `mjv_*` and 25 `mjr_*` rendering functions,
  67 classes and 78 enum types.
- Both objects have named accessors per element kind: `model.body("torso")`, `data.joint("hip")`, `data.site("tip")`.

The core loads about 110 model fields and computes about 85 data arrays:
- the state: `qpos`, `qvel`, `ctrl`, `time`, `qacc`, `qacc_warmstart`;
- positions and frames: `xpos`, `xquat`, `xmat`, `xipos`, `ximat`, `xanchor`, `xaxis`, `geom_xpos`, `geom_xmat`,
  `site_xpos`, `site_xmat`, `subtree_com`;
- composite inertias and velocities: `cinert`, `cdof`, `crb`, `cvel`, `cdof_dot`, `cfrc_ext`;
- the mass matrix and its factor, the forces (`qfrc_*`, `actuator_force`), the tendons;
- the contacts, the constraint rows (`efc_*`), and the solver's statistics.

Of what Gymnasium reads, the core lacks a few model fields and two small items. The model fields are the renderer's
`ncam`, `stat.extent`, `vis.global_` and `geom_rgba`, and `nbvh`, which the tests compare to tell the `xml_file`
variants apart; the embedded models will carry them all (DR-63). The two small items are `na` and `act` (always 0 and
empty, since no model has a stateful actuator) and `body(name)`, a name lookup over arrays the core already holds.

### 15.3 The options, revised

| Option | What a user writes | Public surface | Gymnasium's code and tests |
|---|---|---|---|
| **A. Gymnasium's reach, MuJoCo's names** | `env.Unwrapped.data.qpos`, `model.opt.timestep`, `data.body("torso").xpos`, `Mujoco.mj_forward(model, data)` | `MjModel` and `MjData` with the members of §15.1 (about 25 and 20), the 9 functions the envs, `utils` and tests call, `mjtState` and `mjtObj`, `utils`' 3 functions | carried over with the same names; the tests mirror, except custom XML models (refused, DR-63) |
| **B. Everything the core holds** | as A, plus `data.efc_force`, `data.contact`, `model.dof_damping` | about 110 model fields and 85 data arrays | as A |
| **C. Gymnasium's reach, .NET names** | `env.Unwrapped.Data.Qpos`, `Model.Opt.Timestep`, `Mujoco.MjForward(model, data)` | as A, PascalCase, each `[PyName]`d | renamed member by member |
| **D. The env only** | `env.SetState(qpos, qvel)`, `env.StateVector()`, `env.GetBodyCom("torso")` | `MujocoEnv`'s own members | the tests that read `data` or call `mj_*` are not mirrored |

In all four, the arrays are writable views of the core's own arrays, as Python's are. NumSharp can wrap a managed
array without copying it (`ArraySlice.FromArray(array, copy: false)`), so `data.qpos[...] = ...` reaches the
simulation. The renderer's types (`Mjv*`, `Mjr*`, `mjv_*`, `mjr_*`, the OpenGL contexts) stay out of every option,
because DR-62 draws with Gym.NET's own rasterizer. Gymnasium's `MujocoRenderer` can keep its name and its `render`
and `close` methods over that rasterizer whichever option is chosen.

**Recommended: A.** Gymnasium's own envs, `utils` and tests read about 45 members and call 9 functions; A gives exactly
those, so they and users' Gymnasium code carry over unchanged in names (DR-06's mirrored tests included). MuJoCo's
names are its documentation's, and option C would give names MuJoCo never uses (`Xipos`, `CfrcExt`). B's other 150
members are inspection aids no env reads. Each would become a public contract to document and keep exact, and A
leaves room to add them later without a breaking change. D cannot run Gymnasium's `utils` or the tests that read
`data`.

**Decided: C** (DR-68). The maintainer chose Gymnasium's reach under .NET names, each member `[PyName]`d to Python's,
in line with how Gym.NET names its own members.
