"""Each worker scopes the persistent torch.compile cache key through its platform."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch.compiler.config as compiler_config

from sglang.multimodal_gen.runtime.managers import gpu_worker
from sglang.multimodal_gen.runtime.platforms.interface import Platform


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


if __name__ == "__main__":
    unittest.main()
