"""What Gymnasium's MuJoCo envs reach at run time: the P9 assessment's measurement (docs/plans/MUJOCO_ASSESSMENT.md).

Runs every MuJoCo id Gymnasium registers (refs/Gymnasium, v4 and v5) on the installed `mujoco` for 1,000 seeded random
steps and prints, per id: the model's sizes and options (integrator, solver, cone, Jacobian, flags), the geom-type pairs
that made contact, the most contacts, constraint rows and islands in one step, the mean solver iterations per step, and
the time per env step. An id Gymnasium refuses on this mujoco (Pusher-v4 on mujoco 3) is reported as refused.

Usage (a Python with numpy, imageio and the mujoco to measure, e.g. `pip install mujoco==3.15.0 imageio`):
    python tools/parity/mujoco/probe_envs.py
"""
import os
import sys
import time
from collections import Counter

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "refs", "Gymnasium"))
import gymnasium as gym  # noqa: E402  (imported from the pinned submodule, as the parity suite does)
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

GEOM = {int(v): k.replace("mjGEOM_", "").lower() for k, v in mujoco.mjtGeom.__members__.items()}
SOLVER = {0: "PGS", 1: "CG", 2: "Newton"}
INTEGRATOR = {0: "Euler", 1: "RK4", 2: "implicit", 3: "implicitfast"}
CONE = {0: "pyramidal", 1: "elliptic"}
JAC = {0: "dense", 1: "sparse", 2: "auto"}


def main():
    """Measures every registered MuJoCo id and prints one line per id, then the contact pairs met by any.

    Returns:
        int: 0.
    """
    ids = sorted(i for i, s in gym.registry.items()
                 if isinstance(s.entry_point, str) and "mujoco." in s.entry_point and not i.endswith(("v2", "v3")))
    print(f"mujoco {mujoco.__version__}; {len(ids)} ids")
    union_pairs = Counter()
    for env_id in ids:
        try:
            env = gym.make(env_id)
        except Exception as e:  # Gymnasium refuses some ids on some mujoco versions (Pusher-v4 needs mujoco<3)
            print(f"{env_id:26} REFUSED: {type(e).__name__}: {str(e)[:120]}")
            continue
        u = env.unwrapped
        m, d = u.model, u.data
        env.action_space.seed(0)
        env.reset(seed=0)
        pairs = Counter()
        max_ncon = max_nefc = max_island = 0
        niter = []
        t0 = time.perf_counter()
        steps = 1000
        for _ in range(steps):
            _obs, _r, term, trunc, _info = env.step(env.action_space.sample())
            max_ncon = max(max_ncon, d.ncon)
            max_nefc = max(max_nefc, d.nefc)
            max_island = max(max_island, d.nisland)
            niter.append(int(d.solver_niter[: max(d.nisland, 1)].sum()))
            for k in range(d.ncon):
                c = d.contact[k]
                a, b = sorted((GEOM[int(m.geom_type[c.geom[0]])], GEOM[int(m.geom_type[c.geom[1]])]))
                pairs[f"{a}-{b}"] += 1
            if term or trunc:
                env.reset()
        dt = (time.perf_counter() - t0) / steps
        union_pairs.update({p: 1 for p in pairs})
        o = m.opt
        print(f"{env_id:26} nq={m.nq:2} nv={m.nv:2} nu={m.nu:2} ngeom={m.ngeom:2} nbody={m.nbody:2} ntendon={m.ntendon} "
              f"{INTEGRATOR[o.integrator]}/{SOLVER[o.solver]}/{CONE[o.cone]}/{JAC[o.jacobian]} dt={o.timestep} skip={u.frame_skip} "
              f"iters={o.iterations} dis={o.disableflags:#x} en={o.enableflags:#x} | max ncon={max_ncon} nefc={max_nefc} "
              f"islands={max_island} solver iters/step mean={np.mean(niter):.1f} | {dt * 1e6:.0f} us/step | pairs {dict(pairs)}")
        env.close()
    print("pairs met by any env:", sorted(union_pairs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
