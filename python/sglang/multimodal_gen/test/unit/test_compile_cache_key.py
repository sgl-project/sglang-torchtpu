"""Each worker scopes the persistent torch.compile cache key through its platform."""

import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch.compiler.config as compiler_config

from sglang.multimodal_gen.runtime.managers import gpu_worker
from sglang.multimodal_gen.runtime.platforms.interface import Platform
from sglang.multimodal_gen.runtime.platforms.tpu import TpuPlatform

_KERNEL_INPUTS_TAG = r"^tpu-kernel-inputs-v1:[0-9a-f]{64}$"
_RANK_ASSIGNMENT_TAG = r"^tpu-rank-assignment-v1:[0-9a-f]{64}$"


class TestCompileCacheHook(unittest.TestCase):
    def test_default_platform_leaves_the_cache_key_alone(self):
        with compiler_config.patch(cache_key_tag="user-tag"):
            Platform().configure_compile_cache()
            self.assertEqual(compiler_config.cache_key_tag, "user-tag")

    def test_worker_configures_the_cache_right_after_distributed_init(self):
        class StopAfterDistributedInit(Exception):
            pass

        calls = []
        platform = MagicMock()
        platform.is_mps.return_value = False
        platform.configure_compile_cache.side_effect = lambda: calls.append(
            "configure_compile_cache"
        )
        worker = SimpleNamespace(
            server_args=SimpleNamespace(
                nnodes=1,
                num_gpus=1,
                tp_size=1,
                cfg_parallel_degree=1,
                ulysses_degree=1,
                ring_degree=1,
                sp_degree=1,
                dp_size=1,
                dist_timeout=None,
            ),
            master_port=29500,
            local_rank=0,
            rank=0,
            _cap_device_memory_for_tests=lambda: None,
            _configure_persistent_torch_compile_cache=lambda: None,
        )
        with (
            patch.dict(os.environ),
            patch.object(gpu_worker, "current_platform", platform),
            patch.object(gpu_worker, "_worker_cpu_intra_op_threads", return_value=None),
            patch.object(
                gpu_worker,
                "maybe_init_distributed_environment_and_model_parallel",
                side_effect=lambda **_: calls.append("init_distributed"),
            ),
            # The next step publishes srt's context and builds the pipeline.
            patch(
                "sglang.srt.runtime_context.get_context",
                side_effect=StopAfterDistributedInit,
            ),
            self.assertRaises(StopAfterDistributedInit),
        ):
            gpu_worker.GPUWorker.init_device_and_model(worker)
        self.assertEqual(calls, ["init_distributed", "configure_compile_cache"])


class TestTpuCompileCacheKey(unittest.TestCase):
    """The TPU worker adds sglang_tpu's compile-cache tags; what each tag
    digests is covered by sglang_tpu's compile_cache tests."""

    def _tags(self, world_size=1, existing="user-tag"):
        device_mesh = ModuleType("torch_tpu._internal.distributed.device_mesh")
        device_mesh.get_device_assignment = lambda: list(range(world_size))
        with (
            patch.dict(sys.modules, {device_mesh.__name__: device_mesh}),
            patch("torch.distributed.get_world_size", return_value=world_size),
            patch("torch.distributed.get_rank", return_value=0),
            compiler_config.patch(cache_key_tag=existing),
        ):
            TpuPlatform().configure_compile_cache()
            return compiler_config.cache_key_tag.split("|")

    def test_one_worker_keeps_user_tags_and_adds_the_kernel_inputs(self):
        user, kernel_inputs = self._tags()
        self.assertEqual(user, "user-tag")
        self.assertRegex(kernel_inputs, _KERNEL_INPUTS_TAG)

    def test_multi_worker_launch_adds_the_rank_assignment(self):
        user, rank_assignment, kernel_inputs = self._tags(world_size=4)
        self.assertEqual(user, "user-tag")
        self.assertRegex(rank_assignment, _RANK_ASSIGNMENT_TAG)
        self.assertRegex(kernel_inputs, _KERNEL_INPUTS_TAG)

    def test_reconfiguring_keeps_one_tag_of_each_kind(self):
        tags = self._tags(world_size=4)
        self.assertEqual(self._tags(world_size=4, existing="|".join(tags)), tags)


if __name__ == "__main__":
    unittest.main()
