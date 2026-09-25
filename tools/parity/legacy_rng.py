"""Record the legacy numpy.random.RandomState draw patterns Gym.NET's envs and spaces use, as raw bits.

Each line is "<case> <index> <value>" so a C# twin can be diffed line by line: float64 values print as 16 hex digits
of their IEEE bit pattern, float32 as 8, integers in decimal. Requires numpy==2.4.2 (the reference stack).

This is the oracle of docs/plans/NUMSHARP_MIGRATION_PLAN.md §9.1 (Appendix D.2): 10 draw patterns under 7 seeds, 602
values. Its output is committed as tests/Gym.Tests/Parity/legacy_rng.golden.txt, which LegacyRngParityTests replays
through Gym.NET's LegacySeeding, Discrete, Box, CartPoleEnv and LunarLanderEnv's draw order. Regenerate the golden after
changing this script, with LF line endings:

    python tools/parity/legacy_rng.py | tr -d '\\r' > tests/Gym.Tests/Parity/legacy_rng.golden.txt

The patterns mirror Gym.NET's current legacy-RandomState code, not Gymnasium's streams: Gymnasium draws from
Generator(PCG64), and its envs call `integers`, not `randint`. The LunarLander block also keeps Gym.NET's order of the
reset draws (force before terrain, defect B-17 of docs/plans/GYMNASIUM_PORT_PLAN.md).
"""
import struct
import numpy as np

SEEDS = [0, 1, 42, 1000, 2**31 - 1, 2**31 + 5, 2**32 - 1]
# LunarLanderEnv computes VIEWPORT_H / SCALE / 2 in float32 before widening it to double.
H_HALF = float(np.float32(np.float32(400.0) / np.float32(30.0)) / np.float32(2.0))


def f64(x):
    return struct.pack(">d", float(x)).hex()


def f32(x):
    return struct.pack(">f", np.float32(x)).hex()


def emit(case, values, fmt):
    for i, v in enumerate(values):
        print(f"{case} {i} {fmt(v)}")


for seed in SEEDS:
    rs = np.random.RandomState(seed)  # LunarLander: constructor, reset, zero step, 20 steps
    emit(f"lunar.wind.s{seed}", [rs.randint(-9999, 9999), rs.randint(-9999, 9999)], str)
    emit(f"lunar.force.s{seed}", [rs.uniform(-1000.0, 1000.0), rs.uniform(-1000.0, 1000.0)], f64)
    emit(f"lunar.height.s{seed}", [rs.uniform(0.0, H_HALF) for _ in range(12)], f64)
    emit(f"lunar.disp.s{seed}", [rs.uniform(-1.0, 1.0) for _ in range(42)], f64)

    rs = np.random.RandomState(seed)  # CartPole reset
    emit(f"cartpole.reset.s{seed}", rs.uniform(-0.05, 0.05, 4), f64)

    rs = np.random.RandomState(seed)  # Discrete(4): 10 samples, then 5 masked samples over [0, 2, 3]
    emit(f"discrete.sample.s{seed}", [rs.randint(0, 4) for _ in range(10)], str)
    emit(f"discrete.masked.s{seed}", [rs.choice(np.array([0, 2, 3])) for _ in range(5)], str)

    rs = np.random.RandomState(seed)  # Box(-1, 1, (3,), float32).Sample()
    low, high = np.full((3,), -1.0, np.float32), np.full((3,), 1.0, np.float32)
    emit(f"box.bounded.s{seed}", rs.uniform(low=low, high=high, size=(3,)).astype(np.float32), f32)

    rs = np.random.RandomState(seed)  # unbounded and half-bounded Box draws
    emit(f"box.normal.s{seed}", rs.normal(0.5, 1.0, (3,)), f64)
    emit(f"box.exponential.s{seed}", rs.exponential(1.0, (3,)), f64)
