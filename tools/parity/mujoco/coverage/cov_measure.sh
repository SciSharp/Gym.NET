#!/usr/bin/env bash
# Measures the engine lines each episode plan executes on the instrumented MuJoCo of build_mujoco_cov.sh: for every
# label, one --setup-only run and one full run of cov_run, each followed by gcov's JSON for every engine object, stored
# under $MJCOV/out/<label>/{setup,full}/. cov_report.py subtracts the two.
#
# Usage:   cov_measure.sh <plans-dir> <label>...     (each label names <plans-dir>/<label>.mjb and .plan)
# Exits non-zero when a build is missing or a run fails.
set -euo pipefail
if [ $# -lt 2 ]; then
  echo "usage: $0 <plans-dir> <label>..." >&2
  exit 2
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLANS="$1"
shift
ROOT="${MJCOV:-$HOME/mjcov}"
B="$ROOT/build"
OBJ="$B/CMakeFiles/mujoco.dir/src/engine"
gcc -O2 -I"$ROOT/src/include" -o "$ROOT/cov_run" "$HERE/cov_run.c" -L"$B/lib" -lmujoco -Wl,-rpath,"$B/lib"

# writes gcov's JSON for every engine object into the folder $1
collect() {
  mkdir -p "$1"
  for g in "$OBJ"/*.gcda; do
    name=$(basename "$g" .gcda)
    (cd "$OBJ" && gcov -j -t "$name.gcda" 2>/dev/null) > "$1/$name.json"
  done
}

for label in "$@"; do
  for mode in setup full; do
    # counters accumulate across runs, so each run starts from no .gcda at all
    find "$B" -name '*.gcda' -delete
    if [ "$mode" = setup ]; then
      "$ROOT/cov_run" "$PLANS/$label.mjb" "$PLANS/$label.plan" --setup-only
    else
      "$ROOT/cov_run" "$PLANS/$label.mjb" "$PLANS/$label.plan"
    fi
    rm -rf "${ROOT:?}/out/$label/$mode"
    collect "$ROOT/out/$label/$mode"
  done
  echo "$label measured"
done
