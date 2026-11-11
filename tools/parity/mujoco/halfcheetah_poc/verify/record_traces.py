"""Records the trace set of traces.txt (MUJOCO_ASSESSMENT.md §13): one oracle.py run per line, several at once, each into
its own folder with its log inside (oracle.log), then prints each run's summary line.

The set is about 1.6 GB; keep it outside the repo. A folder that already holds a run.json is skipped, so an interrupted
recording resumes where it stopped (delete a folder to record it again).

Args (command line):
    --out DIR       the folder that receives one subfolder per trace (required).
    --python PATH   the Python with numpy and mujoco 3.15.0 (default: this interpreter).
    --jobs N        concurrent oracle.py runs (default 8; each is single-threaded).
    --only NAMES    record only these folder names (default: every line).
    --list PATH     the trace list (default: traces.txt next to this script).

Returns (exit code): 0 when every run finished, 1 when any oracle.py run failed (its log says why).
Raises:
    FileNotFoundError: the trace list or oracle.py is missing.
"""
import argparse
import concurrent.futures
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.join(os.path.dirname(HERE), "oracle.py")


def parse(path):
    """Reads the trace list.

    Args:
        path: the list file.
    Returns:
        A list of (name, oracle.py arguments) in file order, comments and blank lines skipped.
    Raises:
        ValueError: a line has fewer than five columns.
    """
    traces = []
    for raw in open(path, encoding="utf-8"):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        cols = line.split()
        if len(cols) < 5:
            raise ValueError(f"{path}: expected name, env, seed, episodes, debug episodes: {raw.strip()}")
        name, env, seed, episodes, debug = cols[:5]
        traces.append((name, ["--env", env, "--seed", seed, "--episodes", episodes, "--debug-episodes", debug] + cols[5:]))
    return traces


def record(python, out, name, args):
    """Runs oracle.py for one trace.

    Args:
        python: the interpreter.
        out: the parent folder.
        name: the trace's folder name.
        args: oracle.py's arguments after --out.
    Returns:
        (name, exit code, the log's last line that is not a warning).
    """
    folder = os.path.join(out, name)
    if os.path.exists(os.path.join(folder, "run.json")):
        return name, 0, "already recorded"
    # the log goes inside the trace's own folder, so the set's parent folder holds only trace folders and a shell glob
    # over it (<out>/*) can be handed to the checker as is
    os.makedirs(folder, exist_ok=True)
    log = os.path.join(folder, "oracle.log")
    with open(log, "w", encoding="utf-8") as fh:
        # -I: the interpreter ignores the environment and the script's folder on sys.path, as for any downloaded package
        code = subprocess.run([python, "-I", ORACLE, "--out", folder] + args, stdout=fh, stderr=subprocess.STDOUT).returncode
    lines = [l for l in open(log, encoding="utf-8", errors="replace").read().splitlines()
             if l.strip() and "warn" not in l.lower() and "deprecat" not in l.lower()]
    return name, code, lines[-1] if lines else ""


def main():
    """Records the requested traces and reports each.

    Returns:
        The exit code: 0 when every run finished, 1 when any failed.
    Raises:
        FileNotFoundError: the trace list is missing.
        ValueError: the trace list has a malformed line.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--list", default=os.path.join(HERE, "traces.txt"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    traces = [t for t in parse(a.list) if not a.only or t[0] in a.only]
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        for name, code, last in pool.map(lambda t: record(a.python, a.out, *t), traces):
            failed += code != 0
            print(f"{name}: {'FAILED, see its oracle.log: ' if code else ''}{last[:150]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
