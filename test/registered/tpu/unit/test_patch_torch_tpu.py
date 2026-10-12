"""Unit tests for TorchTPU worker initialization.

Runs anywhere (no TPU required) — exercises pure Python env-var logic.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit", chips=1)

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.platforms import current_platform
from sglang.srt.utils.patch_torch import initialize_torch_tpu_worker
from sglang.test.test_utils import CustomTestCase

_TORCHRUN_ENV_VARS = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TPU_WORKER_HOSTNAMES",
    "TPU_VISIBLE_DEVICES",
    "SGLANG_TPU_PD_COLOCATED",
)


class TestInitializeTorchTpuWorker(CustomTestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in _TORCHRUN_ENV_VARS}
        for k in _TORCHRUN_ENV_VARS:
            os.environ.pop(k, None)
        init_backend_patcher = patch.object(current_platform, "init_backend")
        self.init_backend = init_backend_patcher.start()
        self.addCleanup(init_backend_patcher.stop)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_skips_when_device_not_tpu(self):
        args = SimpleNamespace(
            device="cuda", tp_size=8, pp_size=1, nnodes=1, node_rank=0, use_ray=False
        )
        initialize_torch_tpu_worker(args, tp_rank=3)
        for k in _TORCHRUN_ENV_VARS:
            self.assertNotIn(k, os.environ)
        self.init_backend.assert_not_called()

    def test_sets_torchrun_vars_for_tpu(self):
        args = SimpleNamespace(
            device="tpu", tp_size=4, pp_size=1, nnodes=2, node_rank=1, use_ray=False
        )

        def assert_worker_env_is_ready():
            self.assertEqual(os.environ["RANK"], "2")
            self.assertEqual(os.environ["WORLD_SIZE"], "4")

        self.init_backend.side_effect = assert_worker_env_is_ready
        initialize_torch_tpu_worker(args, tp_rank=2)
        self.assertEqual(os.environ["RANK"], "2")
        self.assertEqual(os.environ["LOCAL_RANK"], "0")
        self.assertEqual(os.environ["WORLD_SIZE"], "4")
        self.assertEqual(os.environ["LOCAL_WORLD_SIZE"], "2")
        self.assertEqual(os.environ["MASTER_ADDR"], "127.0.0.1")
        self.assertEqual(os.environ["MASTER_PORT"], "29500")
        self.init_backend.assert_called_once_with()

    def test_single_node_rank0(self):
        args = SimpleNamespace(
            device="tpu", tp_size=8, pp_size=1, nnodes=1, node_rank=0, use_ray=False
        )
        initialize_torch_tpu_worker(args, tp_rank=0)
        self.assertEqual(os.environ["RANK"], "0")
        self.assertEqual(os.environ["WORLD_SIZE"], "8")
        self.assertEqual(os.environ["LOCAL_WORLD_SIZE"], "8")

    def test_pipeline_stages_share_one_slice(self):
        args = SimpleNamespace(
            device="tpu", tp_size=1, pp_size=8, nnodes=1, node_rank=0, use_ray=False
        )
        initialize_torch_tpu_worker(args, tp_rank=0, pp_rank=5)
        self.assertEqual(os.environ["RANK"], "5")
        self.assertEqual(os.environ["LOCAL_RANK"], "5")
        self.assertEqual(os.environ["WORLD_SIZE"], "8")
        self.assertEqual(os.environ["LOCAL_WORLD_SIZE"], "8")
        self.init_backend.assert_called_once_with()

    def test_pipeline_rank_is_pp_major(self):
        args = SimpleNamespace(
            device="tpu", tp_size=2, pp_size=2, nnodes=1, node_rank=0, use_ray=False
        )
        initialize_torch_tpu_worker(args, tp_rank=1, pp_rank=1)
        self.assertEqual(os.environ["RANK"], "3")
        self.assertEqual(os.environ["WORLD_SIZE"], "4")

    def test_master_addr_port_respect_existing(self):
        os.environ["MASTER_ADDR"] = "10.0.0.1"
        os.environ["MASTER_PORT"] = "12345"
        args = SimpleNamespace(
            device="tpu", tp_size=1, pp_size=1, nnodes=1, node_rank=0, use_ray=False
        )
        initialize_torch_tpu_worker(args, tp_rank=0)
        self.assertEqual(os.environ["MASTER_ADDR"], "10.0.0.1")
        self.assertEqual(os.environ["MASTER_PORT"], "12345")

    def test_colocation_is_disabled_by_default(self):
        args = SimpleNamespace(
            device="tpu",
            tp_size=4,
            pp_size=1,
            nnodes=1,
            node_rank=0,
            use_ray=False,
            base_gpu_id=4,
            gpu_id_step=1,
        )
        initialize_torch_tpu_worker(args, tp_rank=2)
        self.assertNotIn("TPU_VISIBLE_DEVICES", os.environ)

    def test_colocation_maps_rank_from_base_device_and_step(self):
        os.environ["SGLANG_TPU_PD_COLOCATED"] = "1"
        args = SimpleNamespace(
            device="tpu",
            tp_size=4,
            pp_size=1,
            nnodes=1,
            node_rank=0,
            use_ray=False,
            base_gpu_id=1,
            gpu_id_step=2,
        )
        initialize_torch_tpu_worker(args, tp_rank=2)
        self.assertEqual(os.environ["TPU_VISIBLE_DEVICES"], "5")

    def test_colocation_rejects_ray(self):
        os.environ["SGLANG_TPU_PD_COLOCATED"] = "1"
        args = SimpleNamespace(
            device="tpu",
            tp_size=4,
            pp_size=1,
            nnodes=1,
            node_rank=0,
            use_ray=True,
            base_gpu_id=0,
            gpu_id_step=1,
        )
        with self.assertRaisesRegex(
            AssertionError, "TPU P/D co-location does not support Ray"
        ):
            initialize_torch_tpu_worker(args, tp_rank=2)


if __name__ == "__main__":
    unittest.main()
