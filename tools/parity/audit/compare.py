"""Score the Gym.NET parity audit: compare gymnasium_side.py's recording with gymnet_side.cs's two recordings.

Usage:
    python tools/parity/audit/compare.py <output-dir>

Reads gymnasium.txt (Gymnasium @ 9e04324), gymnet.txt (Gym.NET today) and projection.txt (the same cases computed the
way Gymnasium computes them, on NumSharp's Generator), plus the CartPole frames of both sides, and prints a Markdown
scorecard. A value matches only when its text is identical: the same bits, dtype width and integer, so a float64
observation never matches a float32 one.
"""
import os
import re
import sys
from collections import OrderedDict

import numpy as np

out_dir = sys.argv[1]


def load(name):
    values = OrderedDict()
    with open(os.path.join(out_dir, name), encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line:
                continue
            case, index, value = line.split(" ", 2)
            values[(case, int(index))] = value
    return values


def family(case):
    return re.sub(r"\.s\d+$", "", case)


def seed_of(case):
    m = re.search(r"\.s(\d+)$", case)
    return int(m.group(1)) if m else None


gym, cur, proj = load("gymnasium.txt"), load("gymnet.txt"), load("projection.txt")

print("## Value-by-value comparison with Gymnasium\n")
print("A seed's sequence is identical only when every value of the case matches for that seed. Single matching values of")
print("small integer draws (Discrete, integer Box) are mostly chance: with 3 to 6 possible values a random match is likely.\n")
print("| Case | Values | Gym.NET today: values | Gym.NET today: seeds identical | Projection: values | Projection: seeds identical |")
print("|---|---:|---:|---:|---:|---:|")
families = OrderedDict()
for key in gym:
    families.setdefault(family(key[0]), []).append(key)
for fam, keys in families.items():
    if fam in ("cartpole.traj",) or fam.startswith("lunar.") or fam.endswith(".meta"):
        continue

    def score(other):
        present = [k for k in keys if k in other]
        if not present:
            return "n/a", "n/a"
        values = f"{sum(1 for k in present if other[k] == gym[k])} / {len(present)}"
        by_case = OrderedDict()
        for k in present:
            by_case.setdefault(k[0], []).append(other[k] == gym[k])
        sequences = f"{sum(1 for flags in by_case.values() if all(flags))} / {len(by_case)}"
        return values, sequences

    cur_values, cur_sequences = score(cur)
    proj_values, proj_sequences = score(proj)
    print(f"| `{fam}` | {len(keys)} | {cur_values} | {cur_sequences} | {proj_values} | {proj_sequences} |")

print("\n## CartPole dynamics (500 open-loop steps per seed, from the same initial state)\n")
print("| Seed | Side | Identical steps (state bits) | Identical steps (state, reward, terminated) | First divergence | Terminates at (Gymnasium / side) | max abs(delta state) |")
print("|---:|---|---:|---:|---:|---|---:|")
seeds = sorted({seed_of(c) for c, _ in gym if c.startswith("cartpole.traj")})


def f64(h):
    return np.frombuffer(bytes.fromhex(h)[::-1], dtype="<f8")[0]


for seed in seeds:
    case = f"cartpole.traj.s{seed}"
    g_rows = [gym[(case, t)].split() for t in range(1, 501)]
    g_term = next((t for t, r in enumerate(g_rows, 1) if r[5] == "1"), None)
    for side_name, side in (("Gym.NET today", cur), ("Projection (float64 port)", proj)):
        rows = [side[(case, t)].split() for t in range(1, 501)]
        same = 0
        same_full = sum(1 for g, s in zip(g_rows, rows) if g == s)
        first_diff = None
        max_delta = 0.0
        for t, (g, s) in enumerate(zip(g_rows, rows), 1):
            if g[:4] == s[:4] and first_diff is None:
                same += 1
            elif first_diff is None:
                first_diff = t
            max_delta = max(max_delta, max(abs(f64(a) - f64(b)) for a, b in zip(g[:4], s[:4])))
        s_term = next((t for t, r in enumerate(rows, 1) if r[5] == "1"), None)
        print(f"| {seed} | {side_name} | {same} / 500 | {same_full} / 500 | {first_diff or '-'} | {g_term} / {s_term} | {max_delta:.3g} |")

print("\n## CartPole frames (same state rendered by both sides, 600 x 400)\n")
print("| Seed | Step | Pixels that differ | Share | Max channel delta |")
print("|---:|---:|---:|---:|---:|")
for name in sorted(os.listdir(os.path.join(out_dir, "frames"))):
    if not name.startswith("cartpole_"):
        continue
    g = np.fromfile(os.path.join(out_dir, "frames", name), dtype=np.uint8).reshape(400, 600, 3)
    n = np.fromfile(os.path.join(out_dir, "frames", "gymnet_" + name), dtype=np.uint8).reshape(400, 600, 3)
    diff = np.any(g != n, axis=2)
    delta = int(np.max(np.abs(g.astype(np.int16) - n.astype(np.int16))))
    m = re.match(r"cartpole_s(\d+)_t(\d+)\.rgb", name)
    print(f"| {m.group(1)} | {m.group(2)} | {int(diff.sum())} | {diff.mean() * 100:.2f}% | {delta} |")

print("\n## LunarLander (Gymnasium: Box2D, v3; Gym.NET: Aether)\n")
print("| Case | Gymnasium | Gym.NET today |")
print("|---|---|---|")
for key, value in gym.items():
    if key[0].startswith("lunar.") and not key[0].startswith("lunar.discrete.reset") and not key[0].startswith("lunar.continuous.reset"):
        print(f"| `{key[0]}` [{key[1]}] | {value} | {cur.get(key, 'n/a')} |")
for mode in ("discrete", "continuous"):
    keys = [k for k in gym if k[0].startswith(f"lunar.{mode}.reset")]
    by_seed = OrderedDict()
    for k in keys:
        by_seed.setdefault(k[0], []).append(cur.get(k) == gym[k])
    same_vectors = sum(1 for flags in by_seed.values() if all(flags))
    print(f"\n`lunar.{mode}.reset`: {sum(1 for k in keys if cur.get(k) == gym[k])} / {len(keys)} values and "
          f"{same_vectors} / {len(by_seed)} whole observations identical (the leg-contact flags start at 0 on both sides).")
if ("lunar.continuous.step_sample", 0) in cur:
    print(f"\nGym.NET continuous `Step(ActionSpace.Sample())`: {cur[('lunar.continuous.step_sample', 0)]}")

print("\n## Metadata\n")
for key, value in gym.items():
    if key[0].endswith(".meta.s0"):
        print(f"- `{key[0]}`: Gymnasium `{value}`; Gym.NET `{cur.get(key, 'n/a')}`")
for name in ("cartpole.obs_space", "lunar.obs_space"):
    for i, label in ((0, "low"), (1, "high")):
        same = gym[(name, i)] == cur.get((name, i))
        print(f"- `{name}` {label}: {'identical' if same else 'differs'} (Gymnasium `{gym[(name, i)]}`, Gym.NET `{cur.get((name, i))}`)")
