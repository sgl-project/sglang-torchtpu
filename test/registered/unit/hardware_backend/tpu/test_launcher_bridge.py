# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TPU distributed launcher bridge."""

import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.hardware_backend.tpu.patches import (
    get_tpu_launcher_module,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTpuLauncherBridge(CustomTestCase):
    """Test TPU distributed launcher compatibility bridge."""

    def test_get_tpu_launcher_module_runtime(self):
        """Verify launcher module returns valid module in current environment."""
        valid_modules = {
            "torch.tpu.distributed",
            "torch_tpu._internal.distributed.launchers.singlehost_wrapper",
            "torch_tpu._internal.distributed.launchers.multihost_wrapper",
        }
        for multihost in (False, True):
            self.assertIn(get_tpu_launcher_module(multihost=multihost), valid_modules)

    def test_get_tpu_launcher_module_legacy(self):
        """Verify legacy wrapper selection when wrapper spec exists."""
        with patch("importlib.util.find_spec", return_value=Mock()):
            self.assertEqual(
                get_tpu_launcher_module(multihost=False),
                "torch_tpu._internal.distributed.launchers.singlehost_wrapper",
            )
            self.assertEqual(
                get_tpu_launcher_module(multihost=True),
                "torch_tpu._internal.distributed.launchers.multihost_wrapper",
            )

    def test_get_tpu_launcher_module_fallback(self):
        """Verify fallback to torch.tpu.distributed when legacy wrappers are missing."""
        for side_effect, return_value in [
            (None, None),
            (
                ModuleNotFoundError(
                    "No module named 'torch_tpu._internal.distributed.launchers'"
                ),
                None,
            ),
        ]:
            with (
                self.subTest(side_effect=side_effect),
                patch(
                    "importlib.util.find_spec",
                    return_value=return_value,
                    side_effect=side_effect,
                ),
            ):
                self.assertEqual(
                    get_tpu_launcher_module(multihost=False), "torch.tpu.distributed"
                )
                self.assertEqual(
                    get_tpu_launcher_module(multihost=True), "torch.tpu.distributed"
                )

    def test_coordinate_worker_rank_discovery(self):
        """Verify _coordinate_worker topological rank discovery on both new and legacy modules."""
        from sglang.srt.hardware_backend.tpu import node_normalization

        config = node_normalization.TopologyConfig(
            hosts=("source", "target"),
            topology=(2, 2, 2, 2),
            devices_per_host=8,
        )
        base_dist = ModuleType("torch_tpu._internal.distributed")
        base_dist.tpu_distributed = SimpleNamespace(
            local_device_coords=Mock(return_value=[0, 0, 1])
        )

        for use_new_api in (True, False):
            with self.subTest(use_new_api=use_new_api):
                mock_calc = Mock(return_value=1)
                modules = {base_dist.__name__: base_dist}
                if use_new_api:
                    new_dist = ModuleType("torch_tpu._internal.distributed.distributed")
                    new_dist._calculate_topological_node_rank = mock_calc
                    modules[new_dist.__name__] = new_dist
                else:
                    legacy_mod = ModuleType(
                        "torch_tpu._internal.distributed.launchers.multihost_wrapper"
                    )
                    legacy_mod.calculate_topological_node_rank = mock_calc
                    modules[legacy_mod.__name__] = legacy_mod

                with (
                    patch.dict(os.environ, RANK="stale", LOCAL_RANK="stale"),
                    patch.dict(sys.modules, modules),
                    patch("torch.distributed.init_process_group"),
                    patch("torch.distributed.barrier"),
                    patch("torch.distributed.destroy_process_group"),
                    patch("sglang.srt.utils.kill_itself_when_parent_died"),
                ):
                    writer = Mock()
                    node_normalization._coordinate_worker(0, config, 1, 10, writer)
                    mock_calc.assert_called_once_with(
                        [0, 0, 1],
                        "2,2,2,2",
                        fallback_index=None,
                        chips_per_host=config.chips_per_host,
                    )
                    writer.send.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
