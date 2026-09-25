"""Record what Gymnasium's seeded spaces and CartPole reset draw, as raw bits, for GymnasiumRngParityTests.

Each line is "<case> <index> <value>": float64 as 16 hex digits of the IEEE bit pattern, float32 as 8, integers in decimal.
The C# test replays every case through Gym.NET's Seeding, Discrete, Box and CartPoleEnv and requires identical lines.

Runs Gymnasium from the pinned source checkout (refs/Gymnasium @ 9e04324) on the reference stack (Python 3.12, NumPy
2.4.2); no pygame or Box2D is needed. Regenerate the golden after changing this script, with LF line endings:

    PYTHONPATH=refs/Gymnasium python tools/parity/gymnasium_rng.py | tr -d '\\r' > tests/Gym.Tests/Parity/gymnasium_rng.golden.txt
"""
import struct

import numpy as np

from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import seeding

# Gymnasium's seeds are Python ints with no upper bound; the large ones exercise Gym.NET's long and BigInteger paths.
SEEDS = [0, 1, 42, 1000, 2**31 - 1, 2**31 + 5, 2**32 - 1, 2**32, 2**40]


def f64(x):
    return struct.pack(">d", float(x)).hex()


def f32(x):
    return struct.pack(">f", np.float32(x)).hex()


def emit(case, values, fmt):
    for i, v in enumerate(values):
        print(f"{case} {i} {fmt(v)}")


for seed in SEEDS:
    # seeding.np_random: the reported seed and the first draws of the generator.
    rng, np_seed = seeding.np_random(seed)
    emit(f"seeding.seed.s{seed}", [np_seed], str)
    emit(f"seeding.random.s{seed}", rng.random(8), f64)

    # Discrete: 10 samples, then 5 masked over [0, 2, 3] from the same space; a space with start=5; spaces larger than 2**31.
    space = Discrete(4, seed=seed)
    emit(f"discrete.sample.s{seed}", [int(space.sample()) for _ in range(10)], str)
    mask = np.array([1, 0, 1, 1], dtype=np.int8)
    emit(f"discrete.masked.s{seed}", [int(space.sample(mask=mask)) for _ in range(5)], str)
    started = Discrete(3, start=5, seed=seed)
    emit(f"discrete.start.s{seed}", [int(started.sample()) for _ in range(5)], str)
    large = Discrete(3_000_000_000, seed=seed)
    emit(f"discrete.large.s{seed}", [int(large.sample()) for _ in range(5)], str)
    huge = Discrete(2**40, seed=seed)
    emit(f"discrete.huge.s{seed}", [int(huge.sample()) for _ in range(5)], str)

    # Box: every bound category, in one box and separately, and the integer paths (high + 1, floor, clip).
    emit(f"box.bounded.s{seed}", Box(-1.0, 1.0, (3,), np.float32, seed=seed).sample(), f32)
    mixed = Box(
        low=np.array([-np.inf, 0.0, -np.inf, -1.0], dtype=np.float32),
        high=np.array([np.inf, np.inf, 0.0, 1.0], dtype=np.float32),
        dtype=np.float32,
        seed=seed,
    )
    emit(f"box.mixed.s{seed}", mixed.sample(), f32)
    emit(f"box.normal.s{seed}", Box(-np.inf, np.inf, (3,), np.float64, seed=seed).sample(), f64)
    emit(f"box.exponential.s{seed}", Box(0.0, np.inf, (3,), np.float64, seed=seed).sample(), f64)
    emit(f"box.upper.s{seed}", Box(-np.inf, 0.0, (3,), np.float64, seed=seed).sample(), f64)
    emit(f"box.int32.s{seed}", Box(0, 5, (8,), np.int32, seed=seed).sample(), str)
    emit(f"box.int8.s{seed}", Box(-5, 5, (8,), np.int8, seed=seed).sample(), str)
    emit(f"box.uint8.s{seed}", Box(0, 255, (8,), np.uint8, seed=seed).sample(), str)

    # CartPole-v1 reset: the float64 state and the float32 observation made from it.
    env = CartPoleEnv()
    obs, _ = env.reset(seed=seed)
    emit(f"cartpole.reset.state.s{seed}", env.state, f64)
    emit(f"cartpole.reset.obs.s{seed}", obs, f32)
