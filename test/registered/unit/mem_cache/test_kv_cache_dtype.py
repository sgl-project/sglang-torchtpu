import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.kv_cache_dtype import configure_kv_cache_dtype
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestKVCacheDtype(unittest.TestCase):
    def test_explicit_draft_auto_does_not_inherit_target_format(self):
        for quant_algo, expected_format, expected_dtype in (
            (None, "auto", torch.bfloat16),
            ("FP8", "fp8_e4m3", torch.float8_e4m3fn),
        ):
            with (
                self.subTest(quant_algo=quant_algo),
                patch("sglang.srt.mem_cache.kv_cache_dtype._is_hip", False),
            ):
                resolved = configure_kv_cache_dtype(
                    server_args_kv_cache_dtype="mxint8",
                    model=SimpleNamespace(
                        quant_config=SimpleNamespace(kv_cache_quant_algo=quant_algo)
                    ),
                    model_dtype=torch.bfloat16,
                    is_draft_worker=True,
                    is_dflash=False,
                    speculative_draft_attention_backend="tpu",
                    speculative_draft_kv_cache_dtype="auto",
                )
                self.assertEqual(resolved, (expected_format, expected_dtype))

    def test_target_ignores_draft_override(self):
        resolved = configure_kv_cache_dtype(
            server_args_kv_cache_dtype="mxint8",
            model=None,
            model_dtype=torch.bfloat16,
            is_draft_worker=False,
            is_dflash=False,
            speculative_draft_attention_backend="tpu",
            speculative_draft_kv_cache_dtype="auto",
        )
        self.assertEqual(resolved, ("mxint8", torch.int8))

    def test_mxint8_requires_bfloat16_model(self):
        for model_dtype in (torch.float16, torch.float32):
            with (
                self.subTest(model_dtype=model_dtype),
                self.assertRaisesRegex(
                    ValueError, "MXINT8 KV cache requires a BF16 model"
                ),
            ):
                configure_kv_cache_dtype(
                    server_args_kv_cache_dtype="mxint8",
                    model=None,
                    model_dtype=model_dtype,
                    is_draft_worker=False,
                    is_dflash=False,
                    speculative_draft_attention_backend="tpu",
                )

    def test_dflash_fa4_uses_unquantized_draft_cache(self):
        resolved = configure_kv_cache_dtype(
            server_args_kv_cache_dtype="mxint8",
            model=None,
            model_dtype=torch.bfloat16,
            is_draft_worker=True,
            is_dflash=True,
            speculative_draft_attention_backend="fa4",
        )
        self.assertEqual(resolved, ("auto", torch.bfloat16))


if __name__ == "__main__":
    unittest.main()
