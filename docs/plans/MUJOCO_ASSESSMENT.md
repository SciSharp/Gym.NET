# MuJoCo (P9): what completing it takes

| | |
|---|---|
| **Date** | 2026-11-11 |
| **Status** | Assessment for P9, which DR-35 postponed out of 2.0. The approved designs stand: DR-30 (a 100% managed port inside Gym.NET.Environments), DR-31 (MuJoCo's own renderer, ported, drawing through a managed OpenGL binding), DR-33 (scope: what Gymnasium's envs reach, proven bitwise) |
| **Measured against** | Gymnasium `8ba8c3a6` (v1.4.0, committed 2026-10-05); MuJoCo 3.15.0 (released the same day): its source, and its `win_amd64` wheel for Python 3.12 in an isolated venv |
| **Reproduce** | §8; the probes are in `tools/parity/mujoco/`, the HalfCheetah proof of concept (§10) in `tools/parity/mujoco/halfcheetah_poc/`, the line coverage (§10.6) in `tools/parity/mujoco/coverage/` |

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

The HalfCheetah proof of concept (§10) records each episode in its own folder, about 40 MB per 1,000 env steps, so keep
the folders outside the repo:

```bash
<venv>/bin/python tools/parity/mujoco/halfcheetah_poc/oracle.py --out <dir>/ep_s42 --seed 42
<venv>/bin/python tools/parity/mujoco/halfcheetah_poc/oracle.py --out <dir>/ep_s0_flip --seed 0 --flip
dotnet run --project tools/parity/mujoco/halfcheetah_poc -c Release -- <dir>/ep_s42 <dir>/ep_s0_flip   # --verbose: per-pass tallies
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
2. **The midphase:** its depth-first order for the torso and forearms decides the contact order when several of their
   geoms touch at once. `mj_collideTree` always descends the node whose static bounding box has the larger surface, so
   the order is fixed by the model. Its pose-dependent overlap tests only prune pairs that cannot touch, so the order
   can be precomputed per body pair at load, without reproducing the tests.
3. **The mass centre:** the velocity's NumPy reductions.
4. **Capsule-capsule collision:** its geometric branches (parallel segments, sphere fallbacks); the coverage shows
   which ones ran.
