"""Regression tests for torchtpu CI command authorization."""

import json
import unittest
from pathlib import Path

import yaml

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
REPO = Path(__file__).resolve().parents[4]


class TestSlashAuthorization(CustomTestCase):
    """Slash commands authorize the actual requester against trusted policy."""

    def test_slash_handler_does_not_execute_pr_code_or_use_original_author(self):
        # An upstream sync can bring back the PR-branch checkout, which lets a PR
        # edit the allowlist and handler that authorize it.
        workflow = yaml.safe_load(
            (REPO / ".github/workflows/slash-command-handler.yml").read_text()
        )
        steps = workflow["jobs"]["slash_command"]["steps"]
        checkout = next(
            s for s in steps if s.get("uses", "").startswith("actions/checkout@")
        )
        self.assertEqual(
            checkout["with"]["ref"], "${{ github.event.repository.default_branch }}"
        )
        handler = next(s for s in steps if s.get("name") == "Handle Slash Command")
        self.assertEqual(handler["env"]["USER_LOGIN"], "${{ github.triggering_actor }}")
        self.assertEqual(handler["env"]["CI_WRITE_ACCESS_CAN_TAG"], "false")


class TestPolicyContents(CustomTestCase):
    """The allowlist is chosen for this repository, never synced from upstream."""

    def test_every_entry_is_a_torchtpu_grant(self):
        # An upstream sync or update_ci_permission.py brings back entries with
        # upstream reasons such as "top contributor" or "custom override".
        policy = json.loads((REPO / ".github/CI_PERMISSIONS.json").read_text())
        self.assertTrue(policy)
        for login, entry in policy.items():
            with self.subTest(login=login):
                self.assertTrue(entry["reason"].startswith("sglang-torchtpu "))


if __name__ == "__main__":
    unittest.main()
