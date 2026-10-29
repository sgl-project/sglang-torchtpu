"""One host budget and one layer count size every Raiden engine of a rank."""

import sys
import types
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager
from sglang_tpu.disaggregation.raiden.host_pool import (
    host_pool_gb,
    split_staging_layers,
)
from sglang_tpu.envs import envs

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Buffer:
    # One Muse Spark BF16 page per layer: 256 tokens x K/V x 1 head x 128.
    shape = (32, 256, 2, 1, 128)

    def element_size(self):
        return 2


class TestStagingLayers(CustomTestCase):
    def test_split_follows_layer_counts_with_one_per_engine(self):
        for layers, total, expected in (
            ([16, 46], 8, [2, 6]),
            ([16, 46], 16, [4, 12]),
            ([16, 46], 3, [1, 2]),
            ([16, 46], 1, [1, 1]),
            ([16, 46], 0, [16, 46]),
            ([16, 46], 99, [16, 46]),
            ([94], 2, [2]),
            ([1, 3], 8, [1, 3]),
        ):
            self.assertEqual(
                split_staging_layers(layers, total), expected, (layers, total)
            )


class TestHostPoolDefault(CustomTestCase):
    def test_unset_budget_follows_the_staging_mode(self):
        option = envs.SGLANG_RAIDEN_HOST_POOL_GB
        with option.override(1):
            option.clear()
            self.assertEqual(host_pool_gb(staging_layers=True), 24)
            self.assertEqual(host_pool_gb(staging_layers=False), 128)

    def test_configured_budget_wins(self):
        with envs.SGLANG_RAIDEN_HOST_POOL_GB.override(64):
            self.assertEqual(host_pool_gb(staging_layers=True), 64)
            self.assertEqual(host_pool_gb(staging_layers=False), 64)


class TestBuildEngines(CustomTestCase):
    """Muse Spark shape at TP16 on two hosts: eight ranks share each budget."""

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        option = envs.SGLANG_RAIDEN_HOST_POOL_GB
        self.stack.enter_context(option.override(option.default))
        option.clear()
        self.engines = []
        engines = self.engines

        class Engine:
            def __init__(self, **kwargs):
                engines.append(kwargs)

        module = types.ModuleType("tpu_sync.api.torch.kv_cache_manager")
        module.KVCacheManager = Engine
        self.stack.enter_context(patch.dict(sys.modules, {module.__name__: module}))

    def _built(self, *, dynamic, layers=8, slot_ceiling=256):
        manager = object.__new__(RaidenKVManager)
        manager._engine = None
        manager._kv_caches = [_Buffer() for _ in range(16)]
        manager._swa_kv_caches = [_Buffer() for _ in range(46)]
        manager._state_kv_caches = ()
        manager._dynamic_host_staging = dynamic
        manager._staging_layers_total = layers
        manager._slot_ceiling = lambda: slot_ceiling
        manager._max_request_blocks = lambda: 1120
        manager._swa_max_request_blocks = lambda: 9
        manager.attn_tp_rank = 0
        manager.attn_tp_size = 16
        manager.attn_dp_size = manager.system_dp_size = manager.attn_cp_size = 1
        manager.pp_size = 1
        manager.server_args = SimpleNamespace(nnodes=2)
        manager.local_ip = "127.0.0.1"
        manager._local_rank = 0
        manager._control_port = 32000
        manager._timeout_s = 120
        manager._transfer_parallelism = 4
        manager.chunked_transfer = True
        manager._build_engine()
        sizes = [(e["num_slots"], e["num_staging_layers"]) for e in self.engines]
        return manager, sizes

    def test_fixed_slots_share_one_count(self):
        # 3 GiB per rank over a 280 MiB full plus a 6.75 MiB SWA slot.
        manager, sizes = self._built(dynamic=False)
        self.assertEqual(sizes, [(10, 2), (10, 6)])
        self.assertTrue(manager._bounded_staging)
        self.assertEqual(manager._pulling.capacity, 10 * 1120)
        self.assertEqual(manager._pulling.swa_capacity, 10 * 9)

    def test_dynamic_staging_gives_windows_eight_slots_per_full_slot(self):
        # 3 GiB per rank over 280 MiB plus eight 6.75 MiB slots.
        manager, sizes = self._built(dynamic=True)
        self.assertEqual(sizes, [(9, 2), (72, 6)])
        self.assertEqual(manager._pulling.swa_capacity, 72 * 9)

    def test_slots_stop_at_what_requests_can_claim(self):
        # The budget fits (9, 72), but a rank cannot claim more than 24 slots.
        _, sizes = self._built(dynamic=True, slot_ceiling=24)
        self.assertEqual(sizes, [(9, 2), (24, 6)])

    def test_zero_layers_keep_full_mirrors(self):
        # The default 16 GiB per rank over a 2240 MiB full plus eight 51.75 MiB
        # SWA mirrors.
        manager, sizes = self._built(dynamic=True, layers=0)
        self.assertEqual(sizes, [(6, 16), (48, 46)])
        self.assertFalse(manager._bounded_staging)

    def test_configured_budget_applies_to_full_mirrors(self):
        with envs.SGLANG_RAIDEN_HOST_POOL_GB.override(64):
            _, sizes = self._built(dynamic=True, layers=0)
        self.assertEqual(sizes, [(3, 16), (24, 46)])

    def test_pipeline_manager_keeps_the_full_mirror_budget(self):
        # A pipeline transfer never builds these engines, so staging stays unset.
        manager, _ = self._built(dynamic=True)
        manager._bounded_staging = RaidenKVManager._bounded_staging
        self.assertEqual(manager._rank_budget_bytes(), 16 * 1024**3)

    def test_bounded_decode_rejects_an_asymmetric_prefill(self):
        manager = object.__new__(RaidenKVManager)
        manager._state_engine = None
        manager._dynamic_host_staging = True
        manager._bounded_staging = True
        with self.assertRaisesRegex(
            NotImplementedError, "SGLANG_TPU_RAIDEN_STAGING_LAYERS=0"
        ):
            manager.register_decode_request(
                room=1,
                local_blocks=[1],
                local_swa_blocks=[],
                aux_index=None,
                bootstrap_infos=[{}, {}],
            )

    def test_budget_without_one_transfer_fails_startup(self):
        with envs.SGLANG_RAIDEN_HOST_POOL_GB.override(1):
            with self.assertRaisesRegex(ValueError, "SGLANG_RAIDEN_HOST_POOL_GB"):
                self._built(dynamic=True)


if __name__ == "__main__":
    unittest.main()
