#!/usr/bin/env bash
# Builds MuJoCo's library with gcov line coverage, for measuring which engine lines a Gymnasium env executes
# (MUJOCO_ASSESSMENT.md §10.6). Linux or WSL with gcc, make and curl; everything goes under $MJCOV (default ~/mjcov):
# a copy of the source, Kitware's CMake binary release, and the build tree. Nothing is installed system-wide.
#
# Usage:   build_mujoco_cov.sh <mujoco-source-dir>     (a MuJoCo checkout or release tarball, e.g. tag 3.15.0)
# Result:  $MJCOV/build/lib/libmujoco.so, a Debug -O0 build with --coverage and without LTO, so line counts are exact.
# Exits non-zero on any failed step; the configure and build logs are $MJCOV/configure.log and $MJCOV/build.log.
set -euo pipefail
if [ $# -ne 1 ]; then
  echo "usage: $0 <mujoco-source-dir>" >&2
  exit 2
fi
SRC_IN="$1"
ROOT="${MJCOV:-$HOME/mjcov}"
CMAKE_VER=3.31.6
mkdir -p "$ROOT"
# copy the source once: a build on a Windows mount (/mnt/c) is several times slower than on the Linux file system
if [ ! -d "$ROOT/src" ]; then
  cp -r "$SRC_IN" "$ROOT/src"
fi
CMAKE="$ROOT/cmake-$CMAKE_VER-linux-x86_64/bin/cmake"
if [ ! -x "$CMAKE" ]; then
  curl -sSL -o "$ROOT/cmake.tgz" "https://github.com/Kitware/CMake/releases/download/v$CMAKE_VER/cmake-$CMAKE_VER-linux-x86_64.tar.gz"
  tar -xzf "$ROOT/cmake.tgz" -C "$ROOT"
fi
# MuJoCo's own dependencies (lodepng, qhull, tinyxml2, ...) are fetched by its CMake files
"$CMAKE" -S "$ROOT/src" -B "$ROOT/build" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_FLAGS="--coverage -O0" -DCMAKE_CXX_FLAGS="--coverage -O0" \
  -DCMAKE_EXE_LINKER_FLAGS=--coverage -DCMAKE_SHARED_LINKER_FLAGS=--coverage \
  -DMUJOCO_BUILD_EXAMPLES=OFF -DMUJOCO_BUILD_SIMULATE=OFF -DMUJOCO_BUILD_TESTS=OFF \
  -DMUJOCO_TEST_PYTHON_UTIL=OFF -DMUJOCO_ENABLE_LTO=OFF > "$ROOT/configure.log" 2>&1
"$CMAKE" --build "$ROOT/build" --target mujoco -j "$(nproc)" > "$ROOT/build.log" 2>&1
ls "$ROOT/build/lib/libmujoco.so"
