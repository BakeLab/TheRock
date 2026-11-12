#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# See corresponding linux_portable_build.py which invokes this within a
# container.
set -e
set -o pipefail
trap 'kill -TERM 0' INT

OUTPUT_DIR="/therock/output"
mkdir -p "$OUTPUT_DIR/caches"

# Set build profiling log directory to match the actual build output location
# This ensures resource_info.py writes logs to the correct path inside the container
export THEROCK_BUILD_PROF_LOG_DIR="$OUTPUT_DIR/build/logs/therock-build-prof"

export CCACHE_DIR="$OUTPUT_DIR/caches/container/ccache"
export PIP_CACHE_DIR="$OUTPUT_DIR/caches/container/pip"
mkdir -p "$CCACHE_DIR"
mkdir -p "$PIP_CACHE_DIR"

pip install -r /therock/src/requirements.txt

python /therock/src/build_tools/health_status.py

# Build compiler launcher: use extra launcher if provided (it handles ccache internally),
# otherwise fall back to ccache directly.
if [ -n "${EXTRA_C_COMPILER_LAUNCHER}" ]; then
  export CMAKE_C_COMPILER_LAUNCHER="${EXTRA_C_COMPILER_LAUNCHER}"
  export CMAKE_CXX_COMPILER_LAUNCHER="${EXTRA_CXX_COMPILER_LAUNCHER}"
else
  export CMAKE_C_COMPILER_LAUNCHER=ccache
  export CMAKE_CXX_COMPILER_LAUNCHER=ccache
fi

# Build manylinux Python executables and Python shared executables argument if MANYLINUX is set
PYTHON_EXECUTABLES_ARG=""
PYTHON_SHARED_EXECUTABLES_ARG=""
if [ "${MANYLINUX}" = "1" ] || [ "${MANYLINUX}" = "true" ]; then
  PYTHON_EXECUTABLES_ARG="-DTHEROCK_DIST_PYTHON_EXECUTABLES=/opt/python/cp314-cp314/bin/python"
  PYTHON_SHARED_EXECUTABLES_ARG="-DTHEROCK_SHARED_PYTHON_EXECUTABLES=/opt/python-shared/cp314-cp314/bin/python3"
fi

set -o xtrace
time cmake -GNinja -S /therock/src -B "$OUTPUT_DIR/build" \
  -DTHEROCK_BUNDLE_SYSDEPS=ON \
  -DTHEROCK_ENABLE_SYSDEPS_AMD_MESA=ON \
  -DTHEROCK_ENABLE_ROCDECODE=ON \
  -DTHEROCK_ENABLE_ROCJPEG=ON \
  -DTHEROCK_ENABLE_RPP=ON \
  ${PYTHON_EXECUTABLES_ARG} \
  ${PYTHON_SHARED_EXECUTABLES_ARG} \
  "$@"
time cmake --build "$OUTPUT_DIR/build" --target therock-artifacts therock-dist
