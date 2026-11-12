# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Python version policy shared by release matrix generators."""

import re


def validate_python_versions(versions: list[str]) -> None:
    for version in versions:
        if not re.fullmatch(r"3\.\d+(?:\.\d+)?", version) or tuple(
            map(int, version.split(".")[:2])
        ) < (3, 14):
            raise ValueError(f"Python 3.14+ required, got {version!r}")
