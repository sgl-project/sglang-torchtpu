# SPDX-License-Identifier: Apache-2.0
"""Select automatic subgroup compilation without admitting older stock wheels."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu import pp_stage_groups

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestStageGroupRuntime(unittest.TestCase):
    def test_automatic_subgroups_do_not_load_legacy_api(self):
        parallel = SimpleNamespace(pp_size=8, tp_size=2, pp_rank=0)
        for version in ("0.2.1.dev20261015025303", "0.2.1.dev20261028000335", "0.2.1"):
            with (
                self.subTest(version=version),
                patch.object(pp_stage_groups, "version", return_value=version),
                patch.object(pp_stage_groups, "_load_make_standalone") as load,
            ):
                pp_stage_groups.make_stage_groups_standalone(parallel)
                load.assert_not_called()

    def test_old_runtime_still_requires_explicit_api(self):
        parallel = SimpleNamespace(pp_size=8, tp_size=2, pp_rank=0)
        with (
            patch.object(pp_stage_groups, "version", return_value="0.2.0"),
            patch.object(
                pp_stage_groups,
                "_load_make_standalone",
                side_effect=RuntimeError("missing API"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "missing API"):
                pp_stage_groups.make_stage_groups_standalone(parallel)

    def test_single_stage_or_device_needs_no_runtime_gate(self):
        with patch.object(pp_stage_groups, "version") as version:
            for pp_size, tp_size in ((1, 16), (16, 1)):
                pp_stage_groups.make_stage_groups_standalone(
                    SimpleNamespace(pp_size=pp_size, tp_size=tp_size)
                )
            version.assert_not_called()


if __name__ == "__main__":
    unittest.main()
