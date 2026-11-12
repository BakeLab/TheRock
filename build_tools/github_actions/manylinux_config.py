# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Manylinux container Python executable configuration.

These path lists correspond to the Python installations available in the
manylinux build container image used by CI
(ghcr.io/rocm/therock_build_manylinux_x86_64).

Used by:
  - build_configure.py (standard CI path)
  - configure_stage.py (multi-arch CI path)
"""

# Python executables for distribution packages (rocpd, roctx, etc.)
DIST_PYTHON_EXECUTABLES = "/opt/python/cp314-cp314/bin/python"

# Python executables with shared libpython, for embedded Python builds (rocgdb)
SHARED_PYTHON_EXECUTABLES = "/opt/python-shared/cp314-cp314/bin/python3"
