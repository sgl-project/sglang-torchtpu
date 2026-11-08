# SPDX-License-Identifier: Apache-2.0
"""A DFlash draft on batched RPA never takes Muse's in-kernel attention output norm."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang_tpu import attention_pallas_rpa_batched as rpa  # noqa: E402

_ARGS = dict(
    queries="q",
    keys="k",
    values="v",
    kv_cache_fused="cache",
    kv_lens="kv_lens",
    page_indices="pages",
    cu_q_lens="cu_q",
    cu_kv_lens="cu_kv",
    distribution="dist",
    attention_sink=None,
    sm_scale=1.0,
    sliding_window=2048,
    soft_cap=None,
    k_scale=None,
    v_scale=None,
)


class TestDraftSpec(unittest.TestCase):
    def test_draft_kernel_turns_the_head_norm_off(self):
        kernel = Mock(return_value="out")
        with patch.object(rpa, "kv_layout", return_value="layout"):
            rpa.RPA_BATCHED_DRAFT_SPEC.invoke_kernel(kernel, **_ARGS)
        self.assertIsNone(kernel.call_args.kwargs["out_head_norm_eps"])

    def test_target_kernel_reads_the_env_as_before(self):
        kernel = Mock(return_value="out")
        with patch.object(rpa, "kv_layout", return_value="layout"):
            rpa.RPA_BATCHED_SPEC.invoke_kernel(kernel, **_ARGS)
        self.assertEqual(kernel.call_args.kwargs["out_head_norm_eps"], -1.0)

    def test_draft_op_is_compiled_apart(self):
        # The compile cache keys on the op name, not on the env the kernel reads.
        self.assertNotEqual(
            rpa.RPA_BATCHED_DRAFT_SPEC.op_name, rpa.RPA_BATCHED_SPEC.op_name
        )

    def test_backend_builds_the_draft_op_for_a_draft_worker(self):
        for is_draft, spec in (
            (True, rpa.RPA_BATCHED_DRAFT_SPEC),
            (False, rpa.RPA_BATCHED_SPEC),
        ):
            backend = object.__new__(rpa.TpuPallasRpaBatchedAttnBackend)
            backend._is_draft = is_draft
            with patch.object(rpa, "_build_pallas_kernel_op") as build:
                backend._build_kernel_op()
            build.assert_called_once_with(spec)


if __name__ == "__main__":
    unittest.main()
