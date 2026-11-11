"""Times MuJoCo's own mj_step on the episodes the checker benchmarks (MUJOCO_ASSESSMENT.md §13.6): per trace folder given,
its first episode (the id, seed, length and termination in run.json), the same reset and the same sampled actions, best
of five runs, with only the mj_step calls inside the timer (nstep=frame_skip, so Python adds one call per env step,
about a microsecond, which shows on the smallest models).

The checker times the managed core the same way on the first folder of each id it is given ("speed (...)" lines), so
give both the same folders in the same order and alternate the two a few times (§13.6 takes the minimum of five
interleaved rounds; another program on the machine can double a single run).

Args (command line): trace folders; the first folder of each id is the one timed.
Returns (stdout): one line per id: microseconds per physics step.
Raises:
    FileNotFoundError: a folder has no run.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..", "refs", "Gymnasium")))
import gymnasium as gym  # noqa: E402  (the pinned submodule, as oracle.py imports it)
import mujoco  # noqa: E402


def time_episode(run):
    """Times one recorded episode's physics.

    Args:
        run: the folder's run.json.
    Returns:
        (microseconds per physics step, env steps): the best of five runs.
    """
    kwargs = {} if run["terminate"] else {"terminate_when_unhealthy": False}
    env = gym.make(run["env"], **kwargs)
    u = env.unwrapped
    seed, steps = run["seeds"][0], run["steps"][0]
    env.reset(seed=seed)
    env.action_space.seed(seed)
    actions = [env.action_space.sample() for _ in range(steps)]
    best = float("inf")
    for _ in range(5):
        env.reset(seed=seed)
        m, d = u.model, u.data
        total = 0.0
        for a in actions:
            d.ctrl[:] = a
            t0 = time.perf_counter()
            mujoco.mj_step(m, d, nstep=u.frame_skip)
            total += time.perf_counter() - t0
        best = min(best, total / (len(actions) * u.frame_skip))
    env.close()
    return best * 1e6, steps


def main():
    """Times the first folder of each id, in the order given."""
    seen = set()
    for folder in sys.argv[1:]:
        run = json.load(open(os.path.join(folder, "run.json")))
        if run["env"] in seen:
            continue
        seen.add(run["env"])
        us, steps = time_episode(run)
        print(f"mujoco ({run['env']}): {us:.2f} us per physics step ({steps} env steps, MuJoCo {mujoco.__version__}, best of 5)")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
