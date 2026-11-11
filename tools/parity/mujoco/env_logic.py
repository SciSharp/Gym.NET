"""How much of a Gymnasium MuJoCo env file is the env's own logic (docs/plans/MUJOCO_ASSESSMENT.md §9): its lines,
docstring lines and code lines, with the mujoco functions it calls and the model and data fields it reads. Everything
else an env computes is MuJoCo's.

Usage (any Python 3):
    python tools/parity/mujoco/env_logic.py [FILE ...]   (default: the five most-used envs' v4 and v5, and mujoco_env.py)
"""
import ast
import io
import os
import re
import sys
import tokenize

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
DEFAULT = [f"{env}_{version}.py" for env in ("half_cheetah", "hopper", "walker2d", "ant", "humanoid") for version in ("v5", "v4")]
DEFAULT.append("mujoco_env.py")


def measure(path):
    """One file's sizes and its use of mujoco.

    Args:
        path: The Python file.

    Returns:
        tuple: (total lines, docstring lines, code lines, mujoco functions called, model and data fields read).

    Raises:
        SyntaxError: The file is not valid Python.
        OSError: The file cannot be read.
    """
    with open(path, encoding="utf-8") as f:
        src = f.read()
    total = src.count("\n")
    doc_lines = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and ast.get_docstring(node) is not None:
            d = node.body[0]
            doc_lines.update(range(d.lineno, d.end_lineno + 1))
    comment_lines = {t.start[0] for t in tokenize.generate_tokens(io.StringIO(src).readline) if t.type == tokenize.COMMENT}
    blank = {i + 1 for i, line in enumerate(src.split("\n")) if not line.strip()}
    code = total - len(doc_lines | comment_lines | blank)
    calls = sorted(set(re.findall(r"mujoco\.(mj\w+)", src)))
    fields = sorted(set(re.findall(r"self\.data\.(\w+)", src)) | set(re.findall(r"self\.model\.(\w+)", src)))
    return total, len(doc_lines), code, calls, fields


def main(argv):
    """Prints one line per file.

    Args:
        argv: The command line without the program name: files, or none for the default set.

    Returns:
        int: 0.

    Raises:
        SyntaxError: A file is not valid Python.
        OSError: A file cannot be read.
    """
    folder = os.path.join(REPO, "refs", "Gymnasium", "gymnasium", "envs", "mujoco")
    for path in argv or [os.path.join(folder, name) for name in DEFAULT]:
        total, docs, code, calls, fields = measure(path)
        print(f"{os.path.basename(path):24} {total:4} lines, {docs:4} docstring, {code:4} code | calls {calls} | fields {fields}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
