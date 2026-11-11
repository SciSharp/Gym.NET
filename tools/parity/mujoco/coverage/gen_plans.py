"""Writes a Gymnasium MuJoCo env's compiled model (MJB) and an episode plan, for the line-coverage measurement
(MUJOCO_ASSESSMENT.md §10.6).

A plan replays what Gymnasium does: each seeded reset's qpos/qvel (after the optional flip), then the seeded action
space's samples until the env terminates or truncates. cov_run.c steps the plan on an instrumented MuJoCo. Loading the
MJB needs no MJCF compiler, so the coverage holds the engine's stepping only.

Plan format (little-endian): int32 nq, nv, nu, frame_skip, rnepost, n_episodes; per episode int32 n_steps, float64
qpos[nq], qvel[nv], actions[n_steps*nu].

Usage (a Python with numpy, imageio and the mujoco to measure, e.g. `pip install mujoco==3.15.0 imageio`):
    python tools/parity/mujoco/coverage/gen_plans.py Hopper-v5 <dir>/hop_term --budget 16000

Args (command line):
    env_id          the Gymnasium id, e.g. Hopper-v5.
    out             output prefix: writes <out>.mjb and <out>.plan.
    --seeds A:B     seed range of the episodes (default 0:10).
    --budget N      instead of a seed range: seeds 0, 1, ... until N env steps are recorded (for envs that terminate).
    --flip          turn the model upside down after reset (qpos[2] += pi), as the HalfCheetah proof of concept did.
    --no-terminate  pass terminate_when_unhealthy=False, so episodes run to the 1,000-step limit.
    --rnepost       mark the plan for mj_rnePostConstraint after every env step, as MujocoEnv runs it; only envs that read
                    cfrc_ext (Ant, Humanoid, HumanoidStandup) need its lines ported.

Raises:
    SystemExit: bad arguments.
"""
import argparse
import os
import struct
import sys

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "refs", "Gymnasium"))
import gymnasium as gym  # noqa: E402  (imported from the pinned submodule, as the parity suite does)
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("env_id")
ap.add_argument("out")
ap.add_argument("--seeds", default="0:10")
ap.add_argument("--budget", type=int, default=0)
ap.add_argument("--flip", action="store_true")
ap.add_argument("--no-terminate", action="store_true")
ap.add_argument("--rnepost", action="store_true")
args = ap.parse_args()

kwargs = {"terminate_when_unhealthy": False} if args.no_terminate else {}
env = gym.make(args.env_id, **kwargs)
u = env.unwrapped
mujoco.mj_saveModel(u.model, args.out + ".mjb", None)


def episode(seed):
    """Records one episode as Gymnasium plays it.

    Args:
        seed: the reset and action-space seed.
    Returns:
        (qpos, qvel, actions): the start, and every env step's action until termination or truncation.
    """
    env.reset(seed=seed)
    if args.flip:
        u.data.qpos[2] += np.pi
        mujoco.mj_forward(u.model, u.data)
    env.action_space.seed(seed)
    q0, v0, acts = u.data.qpos.copy(), u.data.qvel.copy(), []
    while True:
        a = env.action_space.sample()
        acts.append(a)
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    return q0, v0, np.asarray(acts, dtype=np.float64)


eps, total = [], 0
if args.budget:
    seed = 0
    while total < args.budget:
        eps.append(episode(seed))
        total += len(eps[-1][2])
        seed += 1
else:
    a, b = (int(x) for x in args.seeds.split(":"))
    for seed in range(a, b):
        eps.append(episode(seed))
        total += len(eps[-1][2])
with open(args.out + ".plan", "wb") as f:
    f.write(struct.pack("<6i", u.model.nq, u.model.nv, u.model.nu, u.frame_skip, int(args.rnepost), len(eps)))
    for q0, v0, acts in eps:
        f.write(struct.pack("<i", len(acts)))
        f.write(np.ascontiguousarray(q0, dtype="<f8").tobytes())
        f.write(np.ascontiguousarray(v0, dtype="<f8").tobytes())
        f.write(np.ascontiguousarray(acts, dtype="<f8").tobytes())
lengths = [len(e[2]) for e in eps]
print(f"{args.env_id}: {len(eps)} episodes, {total} env steps ({total * u.frame_skip} physics steps), "
      f"episode length min {min(lengths)} median {int(np.median(lengths))} max {max(lengths)}")
