"""The KV offload to decode DRAM refuses what it does not support.

Every unsupported setting must stop the engine at start with a message that
names the switch and the setting, on either role; a pair whose two engines
disagree must fail its requests with the reason. A setting that silently
runs without the offload fails here.
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.decode_dram_offload import (
    startup_validation,
)
from sglang_tpu.disaggregation.raiden import dram_offload_channel as raiden_offload
from sglang_tpu.testing.decode_dram_offload import (
    SWITCH,
    supported_engine,
    validate_engine_as,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import functools
import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.decode_dram_offload import (
    startup_validation,
)
from sglang_tpu.testing.decode_dram_offload import (
    SWITCH,
    supported_engine,
    validate_engine_as,
)

# (attribute of a supported engine, the value that breaks it, text the refusal
# must contain)
_UNSUPPORTED = (
    ("kv_manager", object(), "--disaggregation-transfer-backend must be raiden"),
    ("scheduler.server_args.disable_radix_cache", True, "--disable-radix-cache"),
    ("disagg.optimistic_prefill_attempts", 1, "optimistic prefill is not supported"),
    ("device.sleep_on_idle", True, "--sleep-on-idle is not supported"),
    ("tree_cache.tree_core", object(), "Python unified radix tree"),
    ("tree_cache.cache_controller", None, "--enable-hierarchical-cache"),
    ("memory.hicache_io_backend", "kernel", "--hicache-io-backend must be raiden"),
    ("memory.hicache_storage_backend", "file", "--hicache-storage-backend"),
    ("memory.enable_unified_memory", True, "--enable-unified-memory"),
    ("kv_manager.pp_size", 2, "pipeline parallelism is not supported (--pp-size 2)"),
    ("kv_manager._pipeline_enabled", True, "SGLANG_RAIDEN_PIPELINE_TRANSFER"),
    ("kv_manager.attn_cp_size", 2, "context parallelism"),
    ("kv_manager.attn_dp_size", 2, "data parallelism"),
    ("kv_manager.system_dp_size", 2, "data parallelism"),
    ("kv_manager._state_kv_caches", (object(),), "GDN/Mamba"),
    ("scheduler.spec_algorithm.is_none", lambda: False, "speculative decoding"),
    ("kv_manager._swa_kv_caches", [object()], "hybrid sliding-window models"),
    ("missing_api", ["register_read(source=)"], "(missing register_read(source=))"),
)


def _broken(path, value):
    engine = supported_engine()
    *parents, name = path.split(".")
    setattr(functools.reduce(getattr, parents, engine), name, value)
    return engine


class TestEngineRefusals(CustomTestCase):
    def test_supported_engine_starts_on_both_roles_and_both_write_policies(self):
        for role in ("prefill", "decode"):
            for policy in ("write_back", "write_through"):
                with self.subTest(role=role, policy=policy):
                    engine = supported_engine()
                    engine.memory.hicache_write_policy = policy
                    validate_engine_as(engine, role=role)

    def test_each_unsupported_setting_stops_the_engine_and_names_itself(self):
        for path, value, expected in _UNSUPPORTED:
            for role in ("prefill", "decode"):
                with self.subTest(setting=path, role=role):
                    with self.assertRaises(
                        startup_validation.UnsupportedDramOffloadConfig
                    ) as raised:
                        validate_engine_as(_broken(path, value), role=role)
                    message = str(raised.exception)
                    self.assertIn(expected, message)
                    self.assertIn(SWITCH.name, message)
                    self.assertIn(f"{role} engine", message)

    def test_decode_needs_its_radix_cache_and_the_prefill_does_not(self):
        engine = supported_engine()
        engine.disagg.disaggregation_decode_enable_radix_cache = False
        validate_engine_as(engine, role="prefill")
        with self.assertRaisesRegex(
            startup_validation.UnsupportedDramOffloadConfig,
            "--disaggregation-decode-enable-radix-cache",
        ):
            validate_engine_as(engine, role="decode")

    def test_switch_off_checks_nothing(self):
        for path, value, _ in _UNSUPPORTED:
            validate_engine_as(_broken(path, value), switch=False)


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
                self, req_id, uuid, address, remote, local, destination="device"
            ): ...

        class Older:
            def register_read(self, req_id, uuid, blocks): ...

            def start_read(self, req_id, uuid, address, remote, local): ...

        missing = [
            "register_read(source=)",
            "start_read(destination=)",
        ]
        for manager, expected in ((Current, []), (Older, missing)):
            with self.subTest(manager.__name__), self._installed(manager):
                self.assertEqual(
                    startup_validation.missing_tpu_sync_host_api(), expected
                )


# ---- Lower-priority cases: less common paths, wire round trips, wording ----


class TestEngineRefusalsExtra(CustomTestCase):
    def test_every_problem_is_reported_at_once(self):
        engine = supported_engine()
        engine.memory.enable_unified_memory = True
        engine.kv_manager.pp_size = 2
        with self.assertRaises(
            startup_validation.UnsupportedDramOffloadConfig
        ) as raised:
            validate_engine_as(engine)
        self.assertIn("--enable-unified-memory", str(raised.exception))
        self.assertIn("pipeline parallelism", str(raised.exception))


class TestPairRefusals(CustomTestCase):
    def test_switch_must_match_on_both_roles(self):
        for mine, theirs in ((True, False), (False, True)):
            with self.subTest(mine=mine, theirs=theirs), SWITCH.override(mine):
                reason = raiden_offload.peer_switch_mismatch(decode_enabled=theirs)
                self.assertIn("same value on both roles", reason)
                self.assertIn(
                    "on for " + ("this engine" if mine else "its decode"), reason
                )
        for value in (True, False):
            with SWITCH.override(value):
                self.assertIsNone(
                    raiden_offload.peer_switch_mismatch(decode_enabled=value)
                )

    def test_resharding_is_refused_only_with_the_switch_on(self):
        with SWITCH.override(True):
            self.assertIn("resharding", raiden_offload.resharding_refusal())
        with SWITCH.override(False):
            self.assertIsNone(raiden_offload.resharding_refusal())


if __name__ == "__main__":
    unittest.main()
