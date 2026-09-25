#!/usr/bin/env bash
# Creates tools/parity/.venv, the reference Python stack of Gym.NET's live parity suite (tests/Gym.Tests.Parity).
#
# Builds a virtual environment next to this script from a CPython 3.12 interpreter and installs
# tools/parity/requirements.txt into it. The parity suite then finds the venv on its own: it asks the venv's interpreter
# for the base installation's shared library (pythonnet must load libpython3.12 or python312.dll, which a venv does not
# carry) and for the venv's sys.path, so nothing has to be configured afterwards. GYMNET_PARITY_VENV points the suite at
# a venv elsewhere.
#
# Gymnasium is not installed: the suite imports it from the refs/Gymnasium submodule, so the oracle is exactly the
# pinned commit. Every run deletes and recreates the venv, so a drifted or half-installed venv is replaced, never
# patched.
#
# On Linux the base interpreter must be a shared build (libpython3.12.so present), as the ones from actions/setup-python
# and most distributions are; pythonnet cannot embed a static build.
#
# Usage: tools/parity/setup_venv.sh [interpreter]
#   interpreter  a CPython 3.12 executable; defaults to python3.12 when it is on PATH, otherwise python3.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv="$here/.venv"

python="${1:-}"
if [[ -z "$python" ]]; then
    if command -v python3.12 >/dev/null 2>&1; then
        python=python3.12
    else
        python=python3
    fi
fi

# Refuse any other version up front: a venv built on it would install cleanly and then fail the manifest check
# (python.version 3.12.*) on every test run, far from the cause.
version="$("$python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [[ "$version" != "3.12" ]]; then
    echo "error: '$python' is Python $version; the parity stack needs CPython 3.12. Pass its path as the first argument." >&2
    exit 1
fi

rm -rf "$venv"
"$python" -m venv "$venv"

# A POSIX venv keeps its interpreter in bin/; a venv created by a Windows Python (for example from Git Bash) in Scripts/.
if [[ -x "$venv/bin/python" ]]; then
    venv_python="$venv/bin/python"
else
    venv_python="$venv/Scripts/python.exe"
fi

"$venv_python" -m pip install --disable-pip-version-check --upgrade pip
"$venv_python" -m pip install --disable-pip-version-check -r "$here/requirements.txt"

# Import what the manifest pins, so a broken wheel fails here rather than in the first test.
"$venv_python" -c 'import numpy, pygame, Box2D; print("numpy", numpy.__version__, "| pygame-ce", pygame.version.ver, "(SDL", str(pygame.version.SDL) + ") | Box2D", Box2D.__version__)'
echo "Created $venv. The parity suite uses it automatically; see tests/Gym.Tests.Parity/CLAUDE.md."
