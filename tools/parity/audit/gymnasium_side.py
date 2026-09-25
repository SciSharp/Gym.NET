"""Record Gymnasium's behavior for the Gym.NET parity audit (docs/plans/GYMNASIUM_PARITY_AUDIT.md).

Runs Gymnasium from the pinned source checkout (refs/Gymnasium @ 9e04324) and writes one "<case> <index> <value>" line
per observation, in the format of tools/parity/gymnasium_rng.py: float64 as 16 hex digits of the IEEE bit pattern, float32
as 8, integers and booleans in decimal. gymnet_side.cs prints the same cases through Gym.NET, and compare.py scores them.

Also writes, into the output directory:
  gymnasium.txt           the lines above
  frames/cartpole_*.rgb   CartPole rgb_array frames, raw (400, 600, 3) uint8, row-major
  cartpole_states.txt     the float64 states those frames were rendered from, so the C# side renders the same states

Usage (reference stack: Python 3.12, NumPy 2.4.2, pygame-ce 2.5.8, Box2D 2.3.10):
    PYTHONPATH=refs/Gymnasium python tools/parity/audit/gymnasium_side.py <output-dir>
"""
import os
import struct
import sys

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import numpy as np  # noqa: E402

import gymnasium  # noqa: E402
from gymnasium.envs.box2d.lunar_lander import LunarLander, heuristic  # noqa: E402
from gymnasium.envs.classic_control.cartpole import CartPoleEnv  # noqa: E402
from gymnasium.spaces import Box, Discrete  # noqa: E402
from gymnasium.utils import seeding  # noqa: E402

SEEDS = [0, 1, 42, 1000]
CARTPOLE_STEPS = 500
FRAME_STEPS = [0, 10, 50]
LUNAR_RANDOM_STEPS = 2000


def f64(x):
    return struct.pack(">d", float(x)).hex()


def f32(x):
    return struct.pack(">f", np.float32(x)).hex()


def action(i):
    """The open-loop CartPole action sequence both sides replay: 0, 0, 1, 1, 0, 0, ..."""
    return (i // 2) % 2


out_dir = sys.argv[1]
os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True)
lines = []
states = []


def emit(case, index, value):
    lines.append(f"{case} {index} {value}")


def dtype_name(a):
    return np.asarray(a).dtype.name


for seed in SEEDS:
    # Seeding: Gymnasium's generator is Generator(PCG64(SeedSequence(seed))).
    rng, np_seed = seeding.np_random(seed)
    for i, v in enumerate(rng.random(8)):
        emit(f"rng.random.s{seed}", i, f64(v))

    # Discrete: 10 plain samples, then 5 masked over [0, 2, 3]; and a space with start=5.
    space = Discrete(4, seed=seed)
    for i in range(10):
        emit(f"discrete.sample.s{seed}", i, int(space.sample()))
    mask = np.array([1, 0, 1, 1], dtype=np.int8)
    for i in range(5):
        emit(f"discrete.masked.s{seed}", i, int(space.sample(mask=mask)))
    started = Discrete(3, start=5, seed=seed)
    for i in range(5):
        emit(f"discrete.start.s{seed}", i, int(started.sample()))
    emit(f"discrete.meta.s{seed}", 0, f"shape={space.shape} dtype={space.dtype.name} sample_type={type(space.sample()).__name__}")

    # Box: bounded float32, unbounded and low-bounded float64, and an integer box.
    sample = Box(-1.0, 1.0, (3,), np.float32, seed=seed).sample()
    for i, v in enumerate(sample):
        emit(f"box.bounded.s{seed}", i, f32(v))
    emit(f"box.bounded.dtype.s{seed}", 0, dtype_name(sample))
    for i, v in enumerate(Box(-np.inf, np.inf, (3,), np.float64, seed=seed).sample()):
        emit(f"box.normal.s{seed}", i, f64(v))
    for i, v in enumerate(Box(0.0, np.inf, (3,), np.float64, seed=seed).sample()):
        emit(f"box.exponential.s{seed}", i, f64(v))
    sample = Box(0, 5, (8,), np.int32, seed=seed).sample()
    for i, v in enumerate(sample):
        emit(f"box.int.s{seed}", i, int(v))

    # CartPole-v1 (unwrapped: no TimeLimit, so all 500 steps run even past termination).
    env = CartPoleEnv(render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    emit(f"cartpole.reset.dtype.s{seed}", 0, dtype_name(obs))
    for i, v in enumerate(obs):
        emit(f"cartpole.reset.obs.s{seed}", i, f32(v))
    for i, v in enumerate(env.state):
        emit(f"cartpole.reset.state.s{seed}", i, f64(v))
    frame_states = {0: np.array(env.state)}
    for t in range(1, CARTPOLE_STEPS + 1):
        obs, reward, terminated, truncated, _ = env.step(action(t - 1))
        s = env.state
        emit(f"cartpole.traj.s{seed}", t, f"{f64(s[0])} {f64(s[1])} {f64(s[2])} {f64(s[3])} {f64(reward)} {int(terminated)}")
        if t in FRAME_STEPS:
            frame_states[t] = np.array(s)
    for t in FRAME_STEPS:
        env.state = frame_states[t]
        frame = env.render()
        frame.astype(np.uint8).tofile(os.path.join(out_dir, "frames", f"cartpole_s{seed}_t{t}.rgb"))
        states.append(f"{seed} {t} " + " ".join(f64(v) for v in frame_states[t]))
    env.close()

    # LunarLander-v3 through make (TimeLimit 1000), driven by Gymnasium's own heuristic, discrete and continuous.
    for continuous in (False, True):
        mode = "continuous" if continuous else "discrete"
        env = gymnasium.make("LunarLander-v3", continuous=continuous)
        obs, _ = env.reset(seed=seed)
        for i, v in enumerate(obs):
            emit(f"lunar.{mode}.reset.s{seed}", i, f32(v))
        total, steps = 0.0, 0
        while True:
            obs, reward, terminated, truncated, _ = env.step(heuristic(env.unwrapped, obs))
            total += reward
            steps += 1
            if terminated or truncated:
                break
        emit(f"lunar.{mode}.heuristic.s{seed}", 0, f"steps={steps} total={f64(total)} ({total:.6f}) terminated={int(terminated)}")
        env.close()

    # Random play (discrete): how many observations fall outside the env's own observation space (defect B-19).
    env = gymnasium.make("LunarLander-v3")
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    outside = 0
    for _ in range(LUNAR_RANDOM_STEPS):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        outside += 0 if env.observation_space.contains(obs) else 1
        if terminated or truncated:
            obs, _ = env.reset()
    emit(f"lunar.random.outside.s{seed}", 0, f"{outside}/{LUNAR_RANDOM_STEPS}")
    env.close()

# Observation space bounds (one line per bound vector).
cp = CartPoleEnv()
emit("cartpole.obs_space", 0, " ".join(f32(v) for v in cp.observation_space.low))
emit("cartpole.obs_space", 1, " ".join(f32(v) for v in cp.observation_space.high))
ll = LunarLander()
emit("lunar.obs_space", 0, " ".join(f32(v) for v in ll.observation_space.low))
emit("lunar.obs_space", 1, " ".join(f32(v) for v in ll.observation_space.high))

with open(os.path.join(out_dir, "gymnasium.txt"), "w", newline="\n") as fh:
    fh.write("\n".join(lines) + "\n")
with open(os.path.join(out_dir, "cartpole_states.txt"), "w", newline="\n") as fh:
    fh.write("\n".join(states) + "\n")
print(f"{len(lines)} lines, {len(states)} frames -> {out_dir}")
