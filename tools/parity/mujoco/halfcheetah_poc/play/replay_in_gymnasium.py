"""Plays the autopilot in Gymnasium's own HalfCheetah-v5 on MuJoCo, to compare with the managed core.

`HalfCheetahPlay score` prints the autopilot's episode returns on the managed core; this script plays the same policy from
the same seeded resets in Gymnasium (refs/Gymnasium) on the installed `mujoco`, and prints the same lines. Because the
core steps exactly as MuJoCo 3.15.0 does, and the policy is evaluated here in the same arithmetic order as in C# (plain
Python floats, summed in column order, clipped, then rounded to float32: NumPy's matrix product would sum differently),
the two outputs are identical to the last bit.

Usage (a Python with numpy, imageio and mujoco==3.15.0):
    python tools/parity/mujoco/halfcheetah_poc/play/replay_in_gymnasium.py [--policy FILE] [--seeds A-B]

Args (command line):
    --policy FILE   the policy JSON (default: policy.json next to this script).
    --seeds A-B     Gymnasium reset seeds, inclusive (default 0-4).

Raises:
    ValueError: the seed range is malformed.
"""
import argparse
import json
import os
import struct
import sys

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "refs", "Gymnasium"))
import gymnasium as gym  # noqa: E402  (imported from the pinned submodule, as the parity suite does)
import numpy as np  # noqa: E402


def act(policy, obs):
    """The policy's action, in LinearPolicy.Act's exact arithmetic order.

    Args:
        policy: the parsed policy JSON.
        obs: the observation (17 float64).

    Returns:
        A float32 array of the six actions.
    """
    n, m = policy["inputs"], policy["outputs"]
    weights, mean, scale = policy["weights"], policy["mean"], policy["scale"]
    action = np.empty(m, dtype=np.float32)
    for i in range(m):
        acc = 0.0
        for j in range(n):
            acc += weights[i * n + j] * ((float(obs[j]) - mean[j]) / scale[j])
        action[i] = np.float32(-1.0 if acc < -1.0 else (1.0 if acc > 1.0 else acc))
    return action


def episode(policy, seed):
    """One Gymnasium episode (1,000 steps) of the policy from reset(seed=seed).

    Args:
        policy: the parsed policy JSON.
        seed: the reset seed.

    Returns:
        The episode return, summed step by step as the C# side sums it.
    """
    env = gym.make("HalfCheetah-v5")
    obs, _ = env.reset(seed=seed)
    total = 0.0
    for _ in range(1000):
        obs, reward, terminated, truncated, _ = env.step(act(policy, obs))
        total += reward
        if terminated or truncated:
            break
    env.close()
    return float(total)


def main():
    """Parses the arguments and prints one line per seed, in HalfCheetahPlay score's format."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy.json"))
    parser.add_argument("--seeds", default="0-4")
    args = parser.parse_args()
    parts = args.seeds.split("-")
    first, last = int(parts[0]), int(parts[-1])
    if last < first:
        raise ValueError(f"--seeds wants A-B with A <= B, got {args.seeds}")
    with open(args.policy) as f:
        policy = json.load(f)
    for seed in range(first, last + 1):
        r = episode(policy, seed)
        bits = struct.unpack("<Q", struct.pack("<d", r))[0]
        print(f"seed {seed}: return {r!r} (0x{bits:016X})")


if __name__ == "__main__":
    main()
