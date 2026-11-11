"""Turns cov_measure.sh's runs into the MuJoCo engine lines each env executes, and what each adds to a baseline env
(MUJOCO_ASSESSMENT.md §10.6).

A line counts as executed by a plan when its count in the full run exceeds its count in the setup-only run (loading the
MJB and making the data), so loading and allocation are excluded. Lines of headers (inline helpers, AVX kernels) are
summed over every object that compiles them. An env is the union of its plans; the first env given is the baseline.

Usage (Linux or WSL, after cov_measure.sh):
    python3 tools/parity/mujoco/coverage/cov_report.py HalfCheetah=hc_a,hc_42,hc_flip Hopper=hop_term,hop_long \
        Humanoid=hum_term,hum_long

Args (command line):
    groups: Name=label[,label...], one per env; the first is the baseline the others are compared with.

Raises:
    SystemExit: no groups given.
    FileNotFoundError: a label was not measured.
"""
import collections
import glob
import json
import os
import sys

OUT = os.path.join(os.environ.get("MJCOV", os.path.expanduser("~/mjcov")), "out")
SRC = os.path.join(os.environ.get("MJCOV", os.path.expanduser("~/mjcov")), "src") + "/"
FUNCS = {}


def load(label, mode):
    """Line counts of one run, recording each line's function in FUNCS.

    Args:
        label: the plan label.
        mode: "setup" or "full".
    Returns:
        {(file, line): count}.
    Raises:
        FileNotFoundError: the run's folder does not exist.
    """
    folder = os.path.join(OUT, label, mode)
    if not os.path.isdir(folder):
        raise FileNotFoundError(folder)
    counts = collections.Counter()
    for path in glob.glob(os.path.join(folder, "*.json")):
        text = open(path).read().strip()
        if not text:
            continue
        for f in json.loads(text)["files"]:
            name = f["file"].replace(SRC, "")
            for ln in f["lines"]:
                key = (name, ln["line_number"])
                counts[key] += ln["count"]
                if ln.get("function_name"):
                    FUNCS[key] = ln["function_name"]
    return counts


def executed(label):
    """The lines a plan's episodes executed beyond loading.

    Args:
        label: the plan label.
    Returns:
        A set of (file, line).
    """
    setup, full = load(label, "setup"), load(label, "full")
    return {k for k, c in full.items() if c > setup.get(k, 0)}


def grouped(lines, by_function):
    """Counts lines per source file or per (file, function), largest first.

    Args:
        lines: a set of (file, line).
        by_function: group by function within the file instead of by file.
    Returns:
        A list of (label, count).
    """
    def key(k):
        f = k[0].split("/")[-1]
        return f"{f}: {FUNCS.get(k, '?')}" if by_function else f
    return collections.Counter(key(k) for k in lines).most_common()


def main(argv):
    """Prints, per env, the executed lines by file and, against the baseline, what it adds by function.

    Args:
        argv: the groups, Name=label[,label...].
    """
    if not argv:
        sys.exit(__doc__)
    envs = []
    for g in argv:
        name, labels = g.split("=", 1)
        lines = set()
        for label in labels.split(","):
            lines |= executed(label)
        envs.append((name, lines))
    base_name, base = envs[0]
    print("executed engine lines (beyond loading):")
    for name, lines in envs:
        extra = "" if name == base_name else f"; new vs {base_name} {len(lines - base)}, not needed from {base_name} {len(base - lines)}"
        print(f"  {name}: {len(lines)}{extra}")
    if len(envs) > 2:
        union = set().union(*(lines for _, lines in envs))
        print(f"  all together: {len(union)}")
    for name, lines in envs:
        print(f"\n== {name} by file")
        for label, n in grouped(lines, False):
            print(f"  {n:5d}  {label}")
        if name != base_name:
            print(f"\n== {name} new vs {base_name}, by function")
            for label, n in grouped(lines - base, True):
                print(f"  {n:5d}  {label}")
            print(f"\n== {base_name} lines {name} does not execute, by file")
            for label, n in grouped(base - lines, False):
                print(f"  {n:5d}  {label}")


if __name__ == "__main__":
    main(sys.argv[1:])
