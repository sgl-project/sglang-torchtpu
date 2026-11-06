"""The KV offload to decode DRAM refuses what it does not support.

Every unsupported setting must stop the engine at start with a message that
names the switch and the setting, on either role; a pair whose two engines
disagree must fail its requests with the reason. A setting that silently
runs without the offload fails here.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import functools
import sys
import unittest
from contextlib import ExitStack
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import pd_spill_validate
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager
from sglang_tpu.envs import envs as tpu_envs

SWITCH = tpu_envs.SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM


def _manager():
    mgr = RaidenKVManager.__new__(RaidenKVManager)
    mgr.pp_size = mgr.attn_cp_size = mgr.attn_dp_size = mgr.system_dp_size = 1
    mgr.attn_tp_size = 8
    mgr._pipeline_enabled = False
    mgr._state_kv_caches = ()
    mgr._swa_kv_caches = []
    mgr.kv_args = SimpleNamespace(page_size=64, kv_cache_dtype_str="fp8_e4m3")
    return mgr


def _tree_cache():
    cache = MagicMock(spec=UnifiedRadixCache)
    cache.tree_core = MagicMock(spec=UnifiedTreeCore)
    cache.cache_controller = object()
    return cache


def _engine():
    """A prefill or decode engine in a setting the offload supports."""
    return SimpleNamespace(
        kv_manager=_manager(),
        tree_cache=_tree_cache(),
        scheduler=SimpleNamespace(
            server_args=SimpleNamespace(disable_radix_cache=False),
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
        ),
        memory=SimpleNamespace(
            hicache_io_backend="raiden",
            hicache_storage_backend=None,
            enable_unified_memory=False,
            hicache_write_policy="write_back",
        ),
        disagg=SimpleNamespace(disaggregation_decode_enable_radix_cache=True),
        missing_api=[],
    )


def _validate(engine, role="prefill", switch=True):
    with ExitStack() as stack:
        # Not SWITCH.override: it does not restore the variable when the body
        # raises, and a refusal raises.
        stack.enter_context(patch.dict("os.environ", {SWITCH.name: str(int(switch))}))
        stack.enter_context(
            patch("sglang.srt.runtime_context.get_memory", return_value=engine.memory)
        )
        stack.enter_context(
            patch("sglang.srt.runtime_context.get_disagg", return_value=engine.disagg)
        )
        stack.enter_context(
            patch.object(
                pd_spill_validate,
                "missing_tpu_sync_host_api",
                return_value=engine.missing_api,
            )
        )
        pd_spill_validate.validate_engine(
            role=role,
            scheduler=engine.scheduler,
            kv_manager=engine.kv_manager,
            tree_cache=engine.tree_cache,
        )


# (attribute of a supported engine, the value that breaks it, text the refusal
# must contain)
_UNSUPPORTED = (
    ("kv_manager", object(), "--disaggregation-transfer-backend must be raiden"),
    ("scheduler.server_args.disable_radix_cache", True, "--disable-radix-cache"),
    ("tree_cache.tree_core", object(), "Python unified radix tree"),
    ("tree_cache.cache_controller", None, "--enable-hierarchical-cache"),
    ("memory.hicache_io_backend", "kernel", "--hicache-io-backend must be raiden"),
    ("memory.hicache_storage_backend", "file", "--hicache-storage-backend"),
    ("memory.enable_unified_memory", True, "--enable-unified-memory"),
    ("kv_manager.pp_size", 2, "pipeline parallelism"),
    ("kv_manager._pipeline_enabled", True, "pipeline transfer"),
    ("kv_manager.attn_cp_size", 2, "context parallelism"),
    ("kv_manager.attn_dp_size", 2, "data parallelism"),
    ("kv_manager.system_dp_size", 2, "data parallelism"),
    ("kv_manager._state_kv_caches", (object(),), "GDN/Mamba"),
    ("scheduler.spec_algorithm.is_none", lambda: False, "speculative decoding"),
    ("kv_manager._swa_kv_caches", [object()], "hybrid sliding-window models"),
    ("missing_api", ["register_read(source=)"], "(missing register_read(source=))"),
)


def _broken(path, value):
    engine = _engine()
    *parents, name = path.split(".")
    setattr(functools.reduce(getattr, parents, engine), name, value)
    return engine


class TestEngineRefusals(CustomTestCase):
    def test_supported_engine_starts_on_both_roles_and_both_write_policies(self):
        for role in ("prefill", "decode"):
            for policy in ("write_back", "write_through"):
                with self.subTest(role=role, policy=policy):
                    engine = _engine()
                    engine.memory.hicache_write_policy = policy
                    _validate(engine, role=role)

    def test_each_unsupported_setting_stops_the_engine_and_names_itself(self):
        for path, value, expected in _UNSUPPORTED:
            for role in ("prefill", "decode"):
                with self.subTest(setting=path, role=role):
                    with self.assertRaises(
                        pd_spill_validate.UnsupportedSpillConfig
                    ) as raised:
                        _validate(_broken(path, value), role=role)
                    message = str(raised.exception)
                    self.assertIn(expected, message)
                    self.assertIn(SWITCH.name, message)
                    self.assertIn(f"{role} engine", message)

    def test_decode_needs_its_radix_cache_and_the_prefill_does_not(self):
        engine = _engine()
        engine.disagg.disaggregation_decode_enable_radix_cache = False
        _validate(engine, role="prefill")
        with self.assertRaisesRegex(
            pd_spill_validate.UnsupportedSpillConfig,
            "--disaggregation-decode-enable-radix-cache",
        ):
            _validate(engine, role="decode")

    def test_switch_off_checks_nothing(self):
        for path, value, _ in _UNSUPPORTED:
            _validate(_broken(path, value), switch=False)


class TestTpuSyncHostApi(CustomTestCase):
    def _installed(self, manager):
        module = ModuleType("tpu_sync.api.torch.kv_cache_manager")
        module.KVCacheManager = manager
        torch_pkg = ModuleType("tpu_sync.api.torch")
        torch_pkg.kv_cache_manager = module
        return patch.dict(
            sys.modules,
            {
                "tpu_sync": ModuleType("tpu_sync"),
                "tpu_sync.api": ModuleType("tpu_sync.api"),
                "tpu_sync.api.torch": torch_pkg,
                module.__name__: module,
            },
        )

    def test_probe_names_exactly_the_host_parameters_a_build_lacks(self):
        class Current:
            def register_read(self, req_id, uuid, blocks, source="device"): ...

            def start_read(
                self,
                req_id,
                uuid,
                address,
                remote,
                local,
                local_host_block_ids=None,
                destination="device",
            ): ...

        class Older:
            def register_read(self, req_id, uuid, blocks): ...

            def start_read(self, req_id, uuid, address, remote, local): ...

        missing = [
            "register_read(source=)",
            "start_read(destination=)",
            "start_read(local_host_block_ids=)",
        ]
        for manager, expected in ((Current, []), (Older, missing)):
            with self.subTest(manager.__name__), self._installed(manager):
                self.assertEqual(
                    pd_spill_validate.missing_tpu_sync_host_api(), expected
                )


if __name__ == "__main__":
    unittest.main()
