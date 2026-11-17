"""CPU unit test for SRTPlatform.dcp_attention_backend_merges_mla.

A platform whose MLA attention backend gathers the query heads and merges the
partial outputs itself must keep forward_mla's DCP decode branch (q all-gather,
LSE merge) and the Kimi-K3 GPU decode-backend policy out of its way; every
other platform keeps both.

Usage:
    python -m pytest test_dcp_backend_merges_mla_seam.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.model_overrides import kimi_k3 as kimi_k3_overrides
from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mla
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _decode_batch():
    mode = SimpleNamespace(is_decode=lambda: True, is_target_verify=lambda: False)
    return SimpleNamespace(forward_mode=mode)


def _platform(merges: bool):
    return SimpleNamespace(dcp_attention_backend_merges_mla=lambda: merges)


class TestDcpBackendMergesMlaSeam(CustomTestCase):
    def _decode_phase(self, merges: bool) -> bool:
        with (
            patch.object(
                forward_mla, "get_parallel", lambda: SimpleNamespace(dcp_enabled=True)
            ),
            patch.object(forward_mla, "current_platform", _platform(merges)),
        ):
            return forward_mla.is_dcp_mla_decode_phase(_decode_batch())

    def test_forward_mla_dcp_decode_branch(self):
        self.assertTrue(self._decode_phase(merges=False))
        self.assertFalse(self._decode_phase(merges=True))

    def test_kimi_k3_dcp_backend_policy_skipped_only_when_backend_merges(self):
        cfg = SimpleNamespace(
            dcp_size=8,
            enable_symm_mem=False,
            speculative_algorithm=None,
            decode_attention_backend="pallas_tpu_mla_sparse",
            prefill_attention_backend=None,
            attention_backend="pallas_tpu_mla_sparse",
            dcp_replicate_q_proj=None,
            dcp_comm_backend="ag_rs",
            kv_cache_dtype="fp8_e4m3",
        )
        with patch.object(kimi_k3_overrides, "resolving_view", lambda _: cfg):
            with patch.object(kimi_k3_overrides, "current_platform", _platform(True)):
                self.assertEqual(kimi_k3_overrides._kimi_k3_overrides(None, None), {})
            with patch.object(kimi_k3_overrides, "current_platform", _platform(False)):
                with self.assertRaises(AssertionError):
                    kimi_k3_overrides._kimi_k3_overrides(None, None)


if __name__ == "__main__":
    unittest.main()
