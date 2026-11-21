#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for package recognition in download_python_packages.py.

"""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from download_python_packages import (
    PackageEntry,
    categorize_package,
    core_tarball_prefix,
    is_allowed_multi_arch_package,
    list_packages_structured,
    list_tarball_for_package,
    parse_arguments,
)

# download_python_packages inserts build_tools onto sys.path on import, so
# _therock_utils is importable here too.
from _therock_utils.storage_location import StorageLocation

class FakePaginator:
    def __init__(self, objects_by_bucket_prefix):
        self.objects_by_bucket_prefix = objects_by_bucket_prefix
        self.calls = []

    def paginate(self, Bucket, Prefix):
        self.calls.append((Bucket, Prefix))
        return [{"Contents": self.objects_by_bucket_prefix.get((Bucket, Prefix), [])}]


class FakeS3Client:
    def __init__(self, objects_by_bucket_prefix):
        self.paginator = FakePaginator(objects_by_bucket_prefix)

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self.paginator


class CategorizePackageTest(unittest.TestCase):
    def test_jax_plugin_is_unknown(self):
        self.assertEqual(
            categorize_package("jax_rocm10_plugin-0.10.0-py3-none-any.whl"),
            "unknown",
        )


class IsAllowedMultiArchPackageTest(unittest.TestCase):
    def test_unknown_package_is_not_allowed(self):
        self.assertFalse(
            is_allowed_multi_arch_package("some_other_pkg-1.0-py3-none-any.whl")
        )


class StructuredDownloadTest(unittest.TestCase):
    def test_structured_lists_product_buckets_and_filters_artifacts(self):
        objects = {
            (
                "therock-repo-amd-rc-core",
                "v5/rocm/core/whl-next/",
            ): [
                {
                    "Key": "v5/rocm/core/whl-next/rocm-sdk-core/rocm_sdk_core-7.13.0rc1-py3-none-linux_x86_64.whl",
                    "Size": 10,
                },
                {
                    "Key": "v5/rocm/core/whl-next/rocm-sdk-core/index.html",
                    "Size": 1,
                },
                {
                    "Key": "v5/rocm/core/whl-next/rocm-sdk-core/rocm_sdk_core-7.13.0rc1-py3-none-linux_x86_64.whl.metadata",
                    "Size": 1,
                },
                {
                    "Key": "v5/rocm/core/whl-next/rocm-sdk-core/extra/rocm_sdk_core-7.13.0rc1.whl",
                    "Size": 1,
                },
                {
                    "Key": "v5/rocm/core/whl-next/rocm-sdk-core/rocm_sdk_core-7.13.0-py3-none-linux_x86_64.whl",
                    "Size": 1,
                },
            ],
            (
                "therock-repo-amd-rc-pytorch",
                "v5/rocm/pytorch/whl-next/",
            ): [
                {
                    "Key": "v5/rocm/pytorch/whl-next/torch/torch-2.10.0+rocm7.13.0rc1-cp312-cp312-linux_x86_64.whl",
                    "Size": 20,
                },
                {
                    "Key": "v5/rocm/pytorch/whl-next/amd-torch-device-gfx942/amd_torch_device_gfx942-2.10.0+rocm7.13.0rc1-py3-none-linux_x86_64.whl",
                    "Size": 30,
                },
            ],
        }
        client = FakeS3Client(objects)

        packages = list_packages_structured(
            client,
            "rc",
            ["core", "pytorch"],
            "whl-next",
            "7.13.0rc1",
            architectures=["gfx942"],
        )

        self.assertEqual(
            packages,
            [
                PackageEntry(
                    StorageLocation(
                        "therock-repo-amd-rc-core",
                        "v5/rocm/core/whl-next/rocm-sdk-core/rocm_sdk_core-7.13.0rc1-py3-none-linux_x86_64.whl",
                    ),
                    10,
                ),
                PackageEntry(
                    StorageLocation(
                        "therock-repo-amd-rc-pytorch",
                        "v5/rocm/pytorch/whl-next/torch/torch-2.10.0+rocm7.13.0rc1-cp312-cp312-linux_x86_64.whl",
                    ),
                    20,
                ),
                PackageEntry(
                    StorageLocation(
                        "therock-repo-amd-rc-pytorch",
                        "v5/rocm/pytorch/whl-next/amd-torch-device-gfx942/amd_torch_device_gfx942-2.10.0+rocm7.13.0rc1-py3-none-linux_x86_64.whl",
                    ),
                    30,
                ),
            ],
        )

    def test_parse_structured_args_defaults_to_whl_next_and_all_products(self):
        args = parse_arguments(
            ["--version=7.13.0rc1", "--structured", "--list-multi-arch-packages"]
        )

        self.assertTrue(args.multi_arch)
        self.assertEqual(args.python_index, "whl-next")
        self.assertEqual(args.repo_stream, "rc")
        self.assertEqual(args.product, ["core", "pytorch"])

    def test_parse_product_accepts_repeated_values(self):
        args = parse_arguments(
            [
                "--version=7.13.0rc1",
                "--structured",
                "--list-multi-arch-packages",
                "--product=core",
                "--product=pytorch",
            ]
        )

        self.assertEqual(args.product, ["core", "pytorch"])

    def test_parse_multi_arch_tarballs_keep_handler_default_output_dir(self):
        args = parse_arguments(
            [
                "--version=7.13.0rc1",
                "--multi-arch",
                "--include-tarballs",
                "--output-dir=downloads",
            ]
        )

        self.assertIsNone(args.tarball_output_dir)

    def test_parse_structured_asan_tarballs_use_asan_output_dir(self):
        args = parse_arguments(
            [
                "--version=7.13.0rc1",
                "--structured",
                "--include-tarballs",
                "--tarball-variant=asan",
                "--output-dir=downloads",
            ]
        )

        self.assertEqual(args.tarball_output_dir, Path("downloads") / "tarball-asan")

    def test_version_matching_rejects_numeric_prefix_collision(self):
        # Requesting "3.0.0" must not also match "13.0.0": a plain substring
        # check would incorrectly find "3.0.0" inside "13.0.0".
        objects = {
            (
                "therock-repo-amd-rc-core",
                "v5/rocm/core/whl-next/",
            ): [
                {
                    "Key": "v5/rocm/core/whl-next/rocm-sdk-core/rocm_sdk_core-3.0.0-py3-none-linux_x86_64.whl",
                    "Size": 10,
                },
                {
                    "Key": "v5/rocm/core/whl-next/rocm-sdk-core/rocm_sdk_core-13.0.0-py3-none-linux_x86_64.whl",
                    "Size": 20,
                },
            ],
        }
        client = FakeS3Client(objects)

        packages = list_packages_structured(
            client,
            "rc",
            ["core"],
            "whl-next",
            "3.0.0",
        )

        self.assertEqual(
            packages,
            [
                PackageEntry(
                    StorageLocation(
                        "therock-repo-amd-rc-core",
                        "v5/rocm/core/whl-next/rocm-sdk-core/rocm_sdk_core-3.0.0-py3-none-linux_x86_64.whl",
                    ),
                    10,
                ),
            ],
        )

    def test_version_matching_rejects_rc_suffix_collision(self):
        # Requesting "7.13.0rc1" must not also match "7.13.0rc10": a plain
        # substring check would incorrectly find "7.13.0rc1" as a prefix of
        # "7.13.0rc10".
        objects = {
            "Contents": [
                {
                    "Key": "v5/rocm/core/tarball/therock-dist-linux-multiarch-7.13.0rc1.tar.gz",
                    "Size": 10,
                },
                {
                    "Key": "v5/rocm/core/tarball/therock-dist-linux-multiarch-7.13.0rc10.tar.gz",
                    "Size": 20,
                },
            ]
        }
        client = FakeS3Client(
            {
                ("therock-repo-amd-rc-core", core_tarball_prefix("release")): objects[
                    "Contents"
                ]
            }
        )

        tarballs = list_tarball_for_package(
            client,
            "therock-repo-amd-rc-core",
            core_tarball_prefix("release"),
            None,
            "7.13.0rc1",
        )

        self.assertEqual(
            tarballs,
            [
                (
                    "v5/rocm/core/tarball/therock-dist-linux-multiarch-7.13.0rc1.tar.gz",
                    10,
                )
            ],
        )

    def test_version_matching_rejects_post_and_dev_suffix_collision(self):
        # Requesting "7.13.0" must not also match "7.13.0.post1" or
        # "7.13.0.dev0": the "." separator in these PEP 440 pre/post-release
        # suffixes is not a word-boundary character in regex, so a naive
        # right-boundary check would incorrectly treat "7.13.0" as ending
        # exactly before the ".post1"/".dev0" suffix.
        objects = {
            "Contents": [
                {
                    "Key": "v5/rocm/core/tarball/therock-dist-linux-multiarch-7.13.0.tar.gz",
                    "Size": 10,
                },
                {
                    "Key": "v5/rocm/core/tarball/therock-dist-linux-multiarch-7.13.0.post1.tar.gz",
                    "Size": 20,
                },
                {
                    "Key": "v5/rocm/core/tarball/therock-dist-linux-multiarch-7.13.0.dev0.tar.gz",
                    "Size": 30,
                },
            ]
        }
        client = FakeS3Client(
            {
                ("therock-repo-amd-rc-core", core_tarball_prefix("release")): objects[
                    "Contents"
                ]
            }
        )

        tarballs = list_tarball_for_package(
            client,
            "therock-repo-amd-rc-core",
            core_tarball_prefix("release"),
            None,
            "7.13.0",
        )

        self.assertEqual(
            tarballs,
            [
                (
                    "v5/rocm/core/tarball/therock-dist-linux-multiarch-7.13.0.tar.gz",
                    10,
                )
            ],
        )

    def test_version_matching_rejects_post_and_dev_suffix_collision_for_wheels(self):
        # Same collision as the tarball case above, but for wheel filenames,
        # where the version segment is followed by the platform/abi/python
        # tags (e.g. "-cp310-cp310-win_amd64.whl") instead of ".tar.gz".
        objects = {
            (
                "therock-repo-amd-rc-core",
                "v5/rocm/core/whl-next/",
            ): [
                {
                    "Key": "v5/rocm/core/whl-next/rocm-sdk-core/rocm_sdk_core-7.13.0-cp310-cp310-win_amd64.whl",
                    "Size": 10,
                },
                {
                    "Key": "v5/rocm/core/whl-next/rocm-sdk-core/rocm_sdk_core-7.13.0.post1-cp310-cp310-win_amd64.whl",
                    "Size": 20,
                },
                {
                    "Key": "v5/rocm/core/whl-next/rocm-sdk-core/rocm_sdk_core-7.13.0.dev0-py3-none-manylinux_2_27_x86_64.whl",
                    "Size": 30,
                },
            ],
        }
        client = FakeS3Client(objects)

        packages = list_packages_structured(
            client,
            "rc",
            ["core"],
            "whl-next",
            "7.13.0",
        )

        self.assertEqual(
            packages,
            [
                PackageEntry(
                    StorageLocation(
                        "therock-repo-amd-rc-core",
                        "v5/rocm/core/whl-next/rocm-sdk-core/rocm_sdk_core-7.13.0-cp310-cp310-win_amd64.whl",
                    ),
                    10,
                ),
            ],
        )

    def test_core_tarball_prefixes_reuse_legacy_tarball_listing(self):
        objects = {
            "Contents": [
                {
                    "Key": "v5/rocm/core/tarball/therock-dist-linux-multiarch-7.13.0rc1.tar.gz",
                    "Size": 10,
                },
                {
                    "Key": "v5/rocm/core/tarball/index.html",
                    "Size": 1,
                },
                {
                    "Key": "v5/rocm/core/tarball/other-7.13.0rc1.tar.gz",
                    "Size": 1,
                },
            ]
        }
        client = FakeS3Client(
            {
                ("therock-repo-amd-rc-core", core_tarball_prefix("release")): objects[
                    "Contents"
                ]
            }
        )

        tarballs = list_tarball_for_package(
            client,
            "therock-repo-amd-rc-core",
            core_tarball_prefix("release"),
            None,
            "7.13.0rc1",
        )

        self.assertEqual(
            tarballs,
            [
                (
                    "v5/rocm/core/tarball/therock-dist-linux-multiarch-7.13.0rc1.tar.gz",
                    10,
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
