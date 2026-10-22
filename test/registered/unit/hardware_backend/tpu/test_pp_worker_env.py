"""Unit tests for the pipeline-parallel process layout on TPU.

Pure env-var and rank arithmetic; torch_tpu is never imported.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.arg_groups.overrides import declare_resolution
from sglang.srt.distributed import bootstrap, parallel_state
from sglang.test.test_utils import CustomTestCase, published_topology
from sglang_tpu.tpu_platform import TpuSRTPlatform
from sglang_tpu.worker_env import (
    initialize_torch_tpu_worker,
    monkey_patch_torch_tpu_env,
)

_ENV_VARS = (
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


def _args(**overrides):
    defaults = dict(
        device="tpu",
        tp_size=1,
        pp_size=8,
        dp_size=1,
        attn_dp_size=1,
        nnodes=1,
        node_rank=0,
        use_ray=False,
        base_gpu_id=0,
        gpu_id_step=1,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestWorkerEnvUnderPipelineParallelism(CustomTestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in _ENV_VARS}
        for k in _ENV_VARS:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_pp8_tp1_gives_each_stage_its_own_rank_in_one_slice(self):
        for pp_rank in range(8):
            monkey_patch_torch_tpu_env(_args(), tp_rank=0, pp_rank=pp_rank)
            self.assertEqual(os.environ["RANK"], str(pp_rank))
            self.assertEqual(os.environ["LOCAL_RANK"], str(pp_rank))
            self.assertEqual(os.environ["WORLD_SIZE"], "8")
            self.assertEqual(os.environ["LOCAL_WORLD_SIZE"], "8")

    def test_worker_bootstrap_honors_resolution_before_publication(self):
        args = _args(device=None, tp_size=1, pp_size=1, nnodes=1)
        declare_resolution(
            args, "test.worker_defaults", device="tpu", pp_size=8, nnodes=2
        )
        with patch("sglang.srt.platforms.current_platform.init_backend") as initialize:
            initialize_torch_tpu_worker(args, tp_rank=0, pp_rank=5)
        initialize.assert_called_once_with()
        self.assertIsNone(args.device)
        self.assertEqual(args.pp_size, 1)
        self.assertEqual(os.environ["RANK"], "5")
        self.assertEqual(os.environ["LOCAL_RANK"], "1")
        self.assertEqual(os.environ["WORLD_SIZE"], "8")
        self.assertEqual(os.environ["LOCAL_WORLD_SIZE"], "4")

    def test_rank_is_pp_major_then_tp(self):
        monkey_patch_torch_tpu_env(_args(tp_size=2, pp_size=2), tp_rank=1, pp_rank=1)
        self.assertEqual(os.environ["RANK"], "3")
        self.assertEqual(os.environ["WORLD_SIZE"], "4")
        self.assertEqual(os.environ["LOCAL_RANK"], "3")

    def test_local_rank_wraps_per_node(self):
        monkey_patch_torch_tpu_env(
            _args(tp_size=2, pp_size=4, nnodes=2), tp_rank=1, pp_rank=3
        )
        self.assertEqual(os.environ["RANK"], "7")
        self.assertEqual(os.environ["LOCAL_WORLD_SIZE"], "4")
        self.assertEqual(os.environ["LOCAL_RANK"], "3")

    def test_pp_rank_defaults_to_zero_for_tp_only_callers(self):
        monkey_patch_torch_tpu_env(_args(tp_size=8, pp_size=1), tp_rank=5)
        self.assertEqual(os.environ["RANK"], "5")
        self.assertEqual(os.environ["WORLD_SIZE"], "8")

    def test_colocated_pd_pins_the_stage_device(self):
        os.environ["SGLANG_TPU_PD_COLOCATED"] = "1"
        monkey_patch_torch_tpu_env(
            _args(base_gpu_id=2, gpu_id_step=1), tp_rank=0, pp_rank=3
        )
        self.assertEqual(os.environ["TPU_VISIBLE_DEVICES"], "5")


class TestResolveBackend(CustomTestCase):
    def _resolve(self, tp_size, pp_size):
        with patch.object(
            bootstrap, "get_default_distributed_backend", return_value="tpu_dist"
        ):
            return bootstrap._resolve_backend(
                device="tpu",
                tp_size=tp_size,
                pp_size=pp_size,
            )

    def test_single_process_world_uses_gloo(self):
        self.assertEqual(self._resolve(1, 1), "gloo")

    def test_pipeline_world_keeps_the_device_backend(self):
        self.assertEqual(self._resolve(1, 8), "tpu_dist")

    def test_tensor_parallel_world_keeps_the_device_backend(self):
        self.assertEqual(self._resolve(8, 1), "tpu_dist")

    def test_pp_size_defaults_to_one(self):
        with patch.object(
            bootstrap, "get_default_distributed_backend", return_value="tpu_dist"
        ):
            backend = bootstrap._resolve_backend(
                device="tpu",
                tp_size=1,
            )
        self.assertEqual(backend, "gloo")


class TestPipelineGroupCommunicators(CustomTestCase):
    def _pp_group_kwargs(self):
        captured = {}

        def fake_init_model_parallel_group(group_ranks, local_rank, backend, **kw):
            captured[kw.get("group_name")] = (group_ranks, kw)
            group = Mock()
            group.device_group = Mock()
            return group

        world = Mock()
        world.device_group = Mock()
        world.local_rank = 0
        with (
            # initialize_model_parallel reads the widths from the published config.
            published_topology(tp_size=1, pp_size=8),
            patch.object(
                parallel_state.platforms, "current_platform", TpuSRTPlatform()
            ),
            patch.object(parallel_state, "_WORLD", None),
            patch.object(parallel_state, "_TP", None),
            patch.object(parallel_state, "_ATTN_CP", None),
            patch.object(parallel_state, "_ATTN_TP", None),
            patch.object(parallel_state, "_PP", None),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=8),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_backend", return_value="gloo"),
            patch.object(
                parallel_state,
                "init_model_parallel_group",
                side_effect=fake_init_model_parallel_group,
            ),
            patch.object(parallel_state, "get_world_group", return_value=world),
        ):
            parallel_state.initialize_model_parallel()
            parallel_state.destroy_model_parallel()
        return captured["pp"]

    def test_pp8_tp1_builds_one_pipeline_group_over_all_ranks(self):
        group_ranks, kwargs = self._pp_group_kwargs()
        self.assertEqual(group_ranks, [list(range(8))])
        self.assertIs(kwargs["use_custom_allreduce"], False)


if __name__ == "__main__":
    unittest.main()
