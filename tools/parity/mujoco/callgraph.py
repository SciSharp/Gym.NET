"""Static call-graph closure of MuJoCo's C sources from given entry points: the size of the engine code P9 must port
(docs/plans/MUJOCO_ASSESSMENT.md).

Finds every top-level function definition in the directory's .c files, follows direct calls from the roots, and prints
the reachable functions' lines by file. It is an upper bound for one run's reach: every branch counts, whatever the
model's flags. Calls through function pointers are invisible to it, so table-driven entry points (the colliders of
mjCOLLISIONFUNC) must be given as roots; files listed as excluded are not entered (subsystems the models never enable).

Usage:
    python tools/parity/mujoco/callgraph.py SRC_DIR ROOT[,ROOT...] [EXCLUDED_FILE[,EXCLUDED_FILE...]]
    e.g. SRC_DIR = <mujoco 3.15.0 source>/src/engine
"""
import os
import re
import sys
from collections import defaultdict, deque

DEFINITION = re.compile(r"^(?:static\s+)?(?:inline\s+)?(?:const\s+)?[A-Za-z_][\w\s\*]*?\b([A-Za-z_]\w*)\s*\(([^;]*)$")
CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
NOT_FUNCTIONS = {"if", "for", "while", "switch", "return", "sizeof"}


def definitions(src):
    """Every top-level function defined in a directory's .c files.

    Args:
        src: The directory.

    Returns:
        dict: Each function's name mapped to (file, first line, last line, body text).

    Raises:
        OSError: A file cannot be read.
    """
    funcs = {}
    for name in sorted(os.listdir(src)):
        if not name.endswith(".c"):
            continue
        with open(os.path.join(src, name), encoding="utf-8", errors="replace") as f:
            lines = f.read().split("\n")
        i = 0
        while i < len(lines):
            m = DEFINITION.match(lines[i])
            if m and not lines[i].startswith((" ", "\t", "#", "//", "}")) and m.group(1) not in NOT_FUNCTIONS:
                # The signature may span lines; the body opens at the first line ending in "{" (a ";" first: a prototype).
                j = i
                while j < len(lines) and not lines[j].rstrip().endswith("{"):
                    if lines[j].rstrip().endswith(";"):
                        break
                    j += 1
                if j < len(lines) and lines[j].rstrip().endswith("{"):
                    depth, k, body = 0, j, []
                    while k < len(lines):
                        depth += lines[k].count("{") - lines[k].count("}")
                        body.append(lines[k])
                        if depth == 0:
                            break
                        k += 1
                    funcs[m.group(1)] = (name, i + 1, k + 1, "\n".join(body))
                    i = k + 1
                    continue
            i += 1
    return funcs


def closure(funcs, roots, excluded):
    """The functions reachable from the roots by direct calls, not entering excluded files.

    Args:
        funcs: From definitions().
        roots: The entry points.
        excluded: File names whose functions are not entered (unless a root).

    Returns:
        set: The reachable function names.
    """
    seen, queue = set(), deque(r for r in roots if r in funcs)
    while queue:
        f = queue.popleft()
        if f in seen or (funcs[f][0] in excluded and f not in roots):
            continue
        seen.add(f)
        queue.extend(c for c in CALL.findall(funcs[f][3]) if c in funcs and c not in seen)
    return seen


def main(argv):
    """Prints the closure's size by file.

    Args:
        argv: The command line without the program name: source directory, roots, optional excluded files.

    Returns:
        int: 0, or 2 for a malformed command line.
    """
    if len(argv) < 2:
        print(__doc__)
        return 2
    funcs = definitions(argv[0])
    roots = argv[1].split(",")
    excluded = set(argv[2].split(",")) if len(argv) > 2 and argv[2] else set()
    seen = closure(funcs, roots, excluded)
    by_file = defaultdict(lambda: [0, 0])
    for f in seen:
        file, start, end, _body = funcs[f]
        by_file[file][0] += 1
        by_file[file][1] += end - start + 1
    total = sum(v[1] for v in by_file.values())
    print(f"{len(funcs)} functions defined; {len(seen)} reachable, {total} lines; roots not found: {[r for r in roots if r not in funcs]}")
    for file, (n, count) in sorted(by_file.items(), key=lambda kv: -kv[1][1]):
        print(f"  {count:6}  {n:4} functions  {file}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
