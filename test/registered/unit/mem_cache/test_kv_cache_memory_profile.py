import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.mem_cache import kv_cache_configurator
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestKVCacheMemoryProfile(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)

        self.configurator = kv_cache_configurator.KVCacheConfigurator.__new__(
            kv_cache_configurator.KVCacheConfigurator
        )
        self.configurator.device = "tpu"
        self.configurator.gpu_id = 0
        self.configurator.mambaish_config = None
        self.configurator.model_config = SimpleNamespace(is_multimodal=False)

        stack.enter_context(
            # Argument resolution always publishes a graph config; the profile
            # reads its prefill phase.
            get_context().override_server_args(
                mem_fraction_static=0.75,
                mm_feature_transport=None,
                cuda_graph_config=CudaGraphConfig(),
            )
        )
        self.platform = stack.enter_context(
            patch.object(kv_cache_configurator, "current_platform")
        )
        self.platform.is_tpu.return_value = True
        self.collect = stack.enter_context(
            patch.object(kv_cache_configurator.gc, "collect")
        )
        self.available_memory = stack.enter_context(
            patch.object(
                kv_cache_configurator, "get_available_gpu_memory", return_value=60
            )
        )
        self.world_group = SimpleNamespace(world_size=16, cpu_group=object())
        stack.enter_context(
            patch.object(
                kv_cache_configurator,
                "get_parallel",
                return_value=SimpleNamespace(
                    launch_world_size=16, world_group=self.world_group
                ),
            )
        )

    def test_tpu_materializes_weights_before_profiling(self):
        events = []
        self.available_memory.return_value = 100

        def synchronize():
            events.append("synchronize")
            # Deferred target and draft allocations consume another 40 GiB.
            self.available_memory.return_value = 60

        def read_memory(*args, **kwargs):
            events.append("read_memory")
            return self.available_memory.return_value

        self.platform.synchronize.side_effect = synchronize
        self.collect.side_effect = lambda: events.append("collect")
        self.available_memory.side_effect = read_memory

        budget = self.configurator._profile_available_bytes(pre_model_load_memory=128)

        self.assertEqual(events, ["synchronize", "collect", "read_memory"])
        self.assertEqual(budget, 28 * (1 << 30))
        self.available_memory.assert_called_once_with(
            "tpu", 0, distributed=True, cpu_group=self.world_group.cpu_group
        )

    def test_non_tpu_does_not_add_synchronization(self):
        self.configurator.device = "cuda"
        self.platform.is_tpu.return_value = False

        budget = self.configurator._profile_available_bytes(pre_model_load_memory=128)

        self.assertEqual(budget, 28 * (1 << 30))
        self.platform.synchronize.assert_not_called()
        self.collect.assert_called_once_with()

    def test_synchronization_failure_prevents_memory_profiling(self):
        self.platform.synchronize.side_effect = RuntimeError("materialization failed")

        with self.assertRaisesRegex(RuntimeError, "materialization failed"):
            self.configurator._profile_available_bytes(pre_model_load_memory=128)

        self.available_memory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
