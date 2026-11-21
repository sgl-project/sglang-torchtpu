"""The KV offload to decode DRAM refuses what it does not support.

Every unsupported setting must stop the engine at start with a message that
names the switch and the setting, on either role, before the transfer manager
is built; a pair whose two engines disagree must fail its requests with the
reason. A setting that silently runs without the offload fails here.
"""

import functools
import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.decode_dram_offload import startup_validation
from sglang_tpu.disaggregation.raiden import dram_offload_channel as raiden_offload
from sglang_tpu.testing.decode_dram_offload import (
    SWITCH,
    supported_engine,
    validate_engine_as,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# (attribute of a supported engine, the value that breaks it, text the refusal
# must contain)
_UNSUPPORTED = (
    ("scheduler.server_args.disable_radix_cache", True, "--disable-radix-cache"),
    ("disagg.optimistic_prefill_attempts", 1, "optimistic prefill is not supported"),
    ("device.sleep_on_idle", True, "--sleep-on-idle is not supported"),
    ("scheduler.tree_cache.tree_core", object(), "Python unified radix tree"),
    ("memory.hicache_io_backend", "kernel", "--hicache-io-backend must be raiden"),
    ("memory.hicache_storage_backend", "file", "--hicache-storage-backend"),
    ("memory.enable_unified_memory", True, "--enable-unified-memory"),
    ("parallel.attn_cp_size", 2, "context parallelism"),
    ("mambaish", object(), "GDN/Mamba"),
    ("scheduler.spec_algorithm.is_none", lambda: False, "speculative decoding"),
    (
        "scheduler.model_config.is_hybrid_swa_compress",
        True,
        "compressed window state",
    ),
    ("missing_api", ["register_read(source=)"], "(missing register_read(source=))"),
    (
        "scheduler.server_args.disaggregation_transfer_backend",
        "mooncake",
        "must be raiden",
    ),
    (
        "scheduler.server_args.enable_hierarchical_cache",
        False,
        "--enable-hierarchical-cache",
    ),
    (
        "scheduler.server_args.pp_size",
        2,
        "pipeline parallelism is not supported (--pp-size 2)",
    ),
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

    def test_pipeline_transfer_env_is_refused_on_the_decode(self):
        engine = supported_engine()
        with patch.dict("os.environ", {"SGLANG_RAIDEN_PIPELINE_TRANSFER": "1"}):
            validate_engine_as(engine, role="prefill")
            with self.assertRaisesRegex(
                startup_validation.UnsupportedDramOffloadConfig,
                "SGLANG_RAIDEN_PIPELINE_TRANSFER",
            ):
                validate_engine_as(engine, role="decode")


class TestHybridModels(CustomTestCase):
    def _hybrid(self):
        engine = supported_engine()
        engine.scheduler.model_config.is_hybrid_swa = True
        return engine

    def test_hybrid_model_is_accepted_with_its_host_tier_of_windows(self):
        engine = self._hybrid()
        engine.scheduler.tree_cache.tree_core.has_swa_host_pool = True
        validate_engine_as(engine)
        # A window cannot be cut to whole pages of one token.
        engine.scheduler.server_args.page_size = 1
        with self.assertRaisesRegex(
            startup_validation.UnsupportedDramOffloadConfig, "--page-size > 1"
        ):
            validate_engine_as(engine)

    def test_hybrid_model_needs_the_host_tier_of_windows(self):
        engine = self._hybrid()
        with self.assertRaisesRegex(
            startup_validation.UnsupportedDramOffloadConfig,
            "host tier of SWA windows",
        ):
            validate_engine_as(engine)


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
                    startup_validation._missing_tpu_sync_host_api(), expected
                )


# ---- Lower-priority cases: less common paths, wire round trips, wording ----


class TestEngineRefusalsExtra(CustomTestCase):
    def test_every_problem_is_reported_at_once(self):
        engine = supported_engine()
        engine.memory.enable_unified_memory = True
        engine.scheduler.server_args.pp_size = 2
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

    def test_resharding_refuses_switch_mismatch_and_non_multiple_tp_ratios(self):
        with SWITCH.override(True):
            self.assertIn(
                "same value on both roles",
                raiden_offload.resharding_refusal(decode_enabled=False),
            )
            self.assertIsNone(
                raiden_offload.resharding_refusal(
                    decode_enabled=True, src_attn_tp=8, dst_attn_tp=4
                )
            )
            self.assertIn(
                "resharding",
                raiden_offload.resharding_refusal(
                    decode_enabled=True, src_attn_tp=4, dst_attn_tp=8
                ),
            )
            self.assertIn(
                "resharding",
                raiden_offload.resharding_refusal(
                    decode_enabled=True, src_attn_tp=6, dst_attn_tp=4
                ),
            )
        with SWITCH.override(False):
            self.assertIsNone(
                raiden_offload.resharding_refusal(
                    decode_enabled=False, src_attn_tp=8, dst_attn_tp=4
                )
            )
            # A decode with the switch on would otherwise run without the
            # offload and never say so.
            self.assertIn(
                "same value on both roles",
                raiden_offload.resharding_refusal(
                    decode_enabled=True, src_attn_tp=8, dst_attn_tp=4
                ),
            )

    def test_dp_attention_and_system_dp_engines_pass_startup_validation(self):
        for dp_size, attn_dp_size in ((2, 2), (2, 1)):
            for role in ("prefill", "decode"):
                with self.subTest(
                    dp_size=dp_size, attn_dp_size=attn_dp_size, role=role
                ):
                    engine = supported_engine()
                    engine.scheduler.server_args.dp_size = dp_size
                    engine.scheduler.server_args.attn_dp_size = attn_dp_size
                    validate_engine_as(engine, role=role)


if __name__ == "__main__":
    unittest.main()
