"""CPU guards for what ``pallas_tpu_mla_sparse`` refuses to serve.

The backend builds its own top-k rows, every causal position up to
``index_topk``, and its kernels read an FP8 cache only. Outside those bounds it
would not fail on its own: rows would silently drop keys past ``index_topk``.
So the constructor has to refuse, before it touches the device.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.hardware_backend.tpu.attention_pallas_mla_sparse import (
    TpuPallasSparseMlaAttnBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _runner(*, architecture, context_len, kv_cache_dtype, index_topk=2048):
    hf_config = SimpleNamespace(architectures=[architecture], index_topk=index_topk)
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config, context_len=context_len),
        kv_cache_dtype=kv_cache_dtype,
    )


class TestSparseMlaBackendRefusals(CustomTestCase):
    def test_context_above_index_topk_is_refused(self):
        runner = _runner(
            architecture="GlmMoeDsaForCausalLM",
            context_len=2049,
            kv_cache_dtype=torch.float8_e4m3fn,
        )
        with self.assertRaisesRegex(ValueError, "at most index_topk"):
            TpuPallasSparseMlaAttnBackend(runner)

    def test_checkpoint_without_dsa_is_refused(self):
        runner = _runner(
            architecture="DeepseekV3ForCausalLM",
            context_len=2048,
            kv_cache_dtype=torch.float8_e4m3fn,
            index_topk=None,
        )
        with self.assertRaisesRegex(ValueError, "GlmMoeDsa"):
            TpuPallasSparseMlaAttnBackend(runner)

    def test_non_fp8_cache_is_refused(self):
        runner = _runner(
            architecture="GlmMoeDsaForCausalLM",
            context_len=2048,
            kv_cache_dtype=torch.bfloat16,
        )
        with self.assertRaisesRegex(ValueError, "fp8_e4m3"):
            TpuPallasSparseMlaAttnBackend(runner)


if __name__ == "__main__":
    unittest.main()
