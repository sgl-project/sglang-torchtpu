"""Unit tests for TPU LIBTPU_INIT_ARGS env overrides.

Covers the override contract: default flags are appended, the VMEM
prefetch serialization applies only to small explicit KV pools, and a
caller-supplied value for the same flag name — even a different value —
is never clobbered or duplicated.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os
import unittest
from unittest.mock import patch

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.env_override import (
    _DECODE_CP_SPARSE_CORE_OVERRIDES,
    _FUSED_RS_SPARSE_CORE_OVERRIDES,
    _VMEM_PREFETCH_MIN_LAYERS,
    _VMEM_PREFETCH_OVERRIDES,
    _VMEM_PREFETCH_SAFE_MIN_TOKENS,
    apply_decode_cp_sparse_core_override,
    apply_fused_rs_sparse_core_override,
    apply_small_kv_pool_vmem_prefetch_override,
    apply_tpu_env_overrides,
)


class TestLibtpuInitArgsOverrides(CustomTestCase):
    def _apply(
        self,
        initial: str | None,
        max_total_tokens: int | None,
        num_hidden_layers: int = _VMEM_PREFETCH_MIN_LAYERS + 1,
    ) -> str:
        env = {} if initial is None else {"LIBTPU_INIT_ARGS": initial}
        with patch.dict(os.environ, env, clear=False):
            if initial is None:
                os.environ.pop("LIBTPU_INIT_ARGS", None)
            apply_tpu_env_overrides()
            apply_small_kv_pool_vmem_prefetch_override(
                max_total_tokens=max_total_tokens,
                num_hidden_layers=num_hidden_layers,
            )
            return os.environ.get("LIBTPU_INIT_ARGS", "")

    def test_small_pool_serializes_vmem_prefetches(self):
        # The device deadlocks large programs at small KV pools when VMEM
        # prefetches overlap; a small explicit pool must get the serializing
        # flags.
        args = self._apply(None, _VMEM_PREFETCH_SAFE_MIN_TOKENS - 1).split()
        for flag in _VMEM_PREFETCH_OVERRIDES:
            self.assertIn(flag, args)

    def test_large_or_unset_pool_keeps_default_schedule(self):
        # Unaffected deployments must keep the default prefetch schedule
        # (bit-identical numerics): no serializing flags at the boundary
        # value or when no explicit pool size is requested.
        for max_total_tokens in (None, _VMEM_PREFETCH_SAFE_MIN_TOKENS):
            args = self._apply(None, max_total_tokens).split()
            for flag in _VMEM_PREFETCH_OVERRIDES:
                self.assertNotIn(flag, args)

    def test_small_model_keeps_default_schedule(self):
        # Models below the measured layer threshold never deadlocked at any
        # pool size; capping their prefetches broke an unrelated small-pool
        # CI configuration, so the containment must not touch them.
        args = self._apply(
            None,
            _VMEM_PREFETCH_SAFE_MIN_TOKENS - 1,
            num_hidden_layers=_VMEM_PREFETCH_MIN_LAYERS - 1,
        ).split()
        for flag in _VMEM_PREFETCH_OVERRIDES:
            self.assertNotIn(flag, args)

    def test_user_value_for_same_flag_name_wins(self):
        # A user-tuned value must be preserved, and the default for the same
        # flag must not be appended alongside it (libtpu would see both).
        user_flag = "--xla_gf_vmem_max_outstanding_prefetches=4"
        args = self._apply(user_flag, _VMEM_PREFETCH_SAFE_MIN_TOKENS - 1).split()
        self.assertIn(user_flag, args)
        self.assertNotIn("--xla_gf_vmem_max_outstanding_prefetches=1", args)

    def test_original_hlo_is_discarded_unless_explicitly_requested(self):
        flag = "--xla_tpu_save_original_hlo_module"
        for initial, expected in ((None, "false"), (f"{flag}=true", "true")):
            with self.subTest(initial=initial):
                args = self._apply(initial, None)
                args = self._apply(args, None).split()
                self.assertEqual(
                    [arg for arg in args if arg.startswith(f"{flag}=")],
                    [f"{flag}={expected}"],
                )

    def test_unrelated_user_flags_are_preserved(self):
        user_flag = "--xla_tpu_enable_sparse_core_collective_offload_all_gather=false"
        args = self._apply(user_flag, _VMEM_PREFETCH_SAFE_MIN_TOKENS - 1).split()
        self.assertIn(user_flag, args)
        for flag in _VMEM_PREFETCH_OVERRIDES:
            self.assertIn(flag, args)

    def test_decode_cp_turns_off_sparse_core_offloads(self):
        # Under DCP the SparseCore all-gather offload returns a wrong first
        # token after long prompts; without DCP the libtpu defaults stay.
        for dcp_size, expected in ((2, True), (1, False)):
            with self.subTest(dcp_size=dcp_size):
                with patch.dict(os.environ, {"LIBTPU_INIT_ARGS": ""}, clear=False):
                    apply_decode_cp_sparse_core_override(dcp_size=dcp_size)
                    args = os.environ["LIBTPU_INIT_ARGS"].split()
                for flag in _DECODE_CP_SPARSE_CORE_OVERRIDES:
                    self.assertEqual(flag in args, expected)


    def test_fused_rs_turns_off_sparse_core_offloads(self):
        # libtpu 0.0.50 faults fused-RS EXTEND modules with the SparseCore
        # collective offloads on (Beam-1.0-501B-A23B NVFP4, TP=8): a
        # SparseCore halt in the distributed-top-k logits module, and a
        # silent execute wedge in the FULL_VOCAB 256-token module -- so the
        # offloads go off for fused-RS regardless of the top-k cutoff.
        cases = [
            # (moe_backend, sampling_backend, tp_size, min_batch_env, expected)
            ("pallas_tpu_fused_rs", None, 8, None, True),
            ("pallas_tpu_fused_rs", "tpu", 8, None, True),
            ("pallas_tpu_fused_rs", "tpu", 16, None, True),
            ("auto", "tpu", 8, None, False),
            ("pallas_tpu_fused_rs", "pytorch", 8, None, False),
            ("pallas_tpu_fused_rs", "tpu", 4, None, False),
            ("pallas_tpu_fused_rs", "tpu", 8, "100000", True),
            ("pallas_tpu_fused_rs", "tpu", 8, "optout", False),
        ]
        for moe_backend, backend, tp_size, min_batch, expected in cases:
            with self.subTest(
                moe_runner_backend=moe_backend,
                sampling_backend=backend,
                tp_size=tp_size,
                min_batch=min_batch,
            ):
                env = {"LIBTPU_INIT_ARGS": ""}
                if min_batch == "optout":
                    env["SGLANG_TPU_FUSED_RS_SPARSE_CORE_OVERRIDE"] = "0"
                    min_batch = None
                if min_batch is not None:
                    env["SGLANG_TPU_DISTRIBUTED_TOP_K_MIN_BATCH_SIZE"] = min_batch
                with patch.dict(os.environ, env, clear=False):
                    if min_batch is None:
                        os.environ.pop(
                            "SGLANG_TPU_DISTRIBUTED_TOP_K_MIN_BATCH_SIZE", None
                        )
                    apply_fused_rs_sparse_core_override(
                        sampling_backend=backend,
                        tp_size=tp_size,
                        moe_runner_backend=moe_backend,
                    )
                    args = os.environ["LIBTPU_INIT_ARGS"].split()
                for flag in _FUSED_RS_SPARSE_CORE_OVERRIDES:
                    self.assertEqual(flag in args, expected)


if __name__ == "__main__":
    unittest.main()
