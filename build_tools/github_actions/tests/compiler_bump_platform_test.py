# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Keep branch-writing automation off read-only repository mirrors."""

import unittest

from workflow_utils import WORKFLOWS_DIR, load_workflow


class CompilerBumpPlatformTest(unittest.TestCase):
    def test_mirror_skips_branch_creation_and_dependent_jobs(self):
        jobs = load_workflow(WORKFLOWS_DIR / "compiler_bump.yml")["jobs"]
        guard = "github.server_url == 'https://github.com'"
        self.assertEqual(jobs["create-branch"]["if"], "${{ " + guard + " }}")
        # Summary intentionally bypasses dependency status, but must not bypass
        # the platform guard or consume a runner for a skipped mirror workflow.
        self.assertEqual(
            jobs["summary"]["if"], "${{ always() && " + guard + " }}"
        )

        def depends_on_branch(job_id: str, visited: frozenset[str]) -> bool:
            if job_id == "create-branch":
                return True
            if job_id in visited:
                self.fail(f"Cyclic job dependency at {job_id}")
            needs = jobs[job_id].get("needs", [])
            if isinstance(needs, str):
                needs = [needs]
            return any(
                depends_on_branch(parent, visited | {job_id}) for parent in needs
            )

        for job_id, job in jobs.items():
            with self.subTest(job=job_id):
                self.assertTrue(depends_on_branch(job_id, frozenset()))
                if job_id not in ("create-branch", "summary"):
                    condition = job.get("if", "")
                    for status in ("always(", "cancelled(", "failure("):
                        self.assertNotIn(status, condition)
