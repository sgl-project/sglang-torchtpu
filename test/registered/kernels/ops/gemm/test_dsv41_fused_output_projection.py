# SPDX-License-Identifier: Apache-2.0
"""Validate the DeepSeek-V4.1 output projection's bucket and numerical contract."""

import importlib.util
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import torch
from jax import export

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import pallas_export
from sglang_tpu.fp8_linear import (
    _quantize_activation_per_row,
    quantize_channelwise_fp8_weight,
)
from sglang_tpu.models.deepseek_v41.kernels import output_projection as kernel
from sglang_tpu.models.deepseek_v41.ops import output_projection as ops

register_tpu_ci(
    est_time=180,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "deepseek-v4.1-flash", "accuracy", "compile"],
)


class TestOutputProjectionDispatch(CustomTestCase):
    def test_rejects_incompatible_token_buckets_and_scale_layouts(self):
        def meta(shape, dtype):
            return torch.empty(shape, dtype=dtype, device="meta")

        wa = meta((4096, 8192), torch.float8_e4m3fn)
        sa = meta((8192,), torch.float32)
        wb = meta((8192, 5120), torch.float8_e4m3fn)
        sb = meta((5120, 1), torch.float32)
        layer = SimpleNamespace(
            attn_tp_size=1,
            wo_a=SimpleNamespace(
                _dsv41_output_projection_weight=wa,
                _dsv41_output_projection_scale=sa,
            ),
            wo_b=SimpleNamespace(
                bias=None,
                use_decode_attn_tp=False,
                weight=wb,
                weight_scale=sb,
            ),
        )

        def projection(output, wa_in, sa_in, wb_in, sb_in):
            self.assertIs(wa_in, wa)
            self.assertIs(sa_in, sa)
            self.assertIs(wb_in, wb)
            self.assertEqual(sb_in.shape, (5120,))
            return meta((output.shape[0], 5120), output.dtype)

        with patch.object(ops, "_OP", projection):
            for tokens in (0, 1, 320, 321, 384, 415, 416, 2048):
                with self.subTest(tokens=tokens):
                    output = meta((tokens, 64, 512), torch.bfloat16)
                    actual = ops.project_output(layer, output)
                    if tokens in (1, 320, 416, 2048):
                        self.assertEqual(actual.shape, (tokens, 5120))
                    else:
                        self.assertIsNone(actual)
            layer.wo_b.weight_block_size = [128, 128]
            self.assertIsNone(
                ops.project_output(layer, meta((64, 64, 512), torch.bfloat16))
            )

    def test_legacy_and_out_of_tree_dispatch_use_the_same_backend(self):
        from sglang.srt import platforms
        from sglang.srt.models import deepseek_v4
        from sglang_tpu.models.deepseek_v41.attention_backend import DSV41TPUAttnBackend
        from sglang_tpu.models.deepseek_v41.hooks import (
            tpu_dsv41_project_attention_output,
        )

        backend = DSV41TPUAttnBackend.__new__(DSV41TPUAttnBackend)
        layer = SimpleNamespace(
            attn_tp_size=1,
            wo_a=SimpleNamespace(_dsv41_output_projection_weight=object()),
            wo_b=SimpleNamespace(use_decode_attn_tp=False),
            _project_wo_a=lambda o, **kwargs: o + 1,
            _project_wo_b=lambda o, deferred: o * 2,
        )
        original = deepseek_v4.MQALayer._project_attention_output
        o = torch.tensor([1.0, 2.0])
        mock_platform = SimpleNamespace(is_out_of_tree=None)
        with (
            patch.object(deepseek_v4, "current_platform", mock_platform),
            patch.object(platforms, "current_platform", mock_platform),
        ):
            for oot in (False, True):
                mock_platform.is_out_of_tree = lambda: oot
                for candidate in (None, o + 11):
                    with self.subTest(oot=oot, uses_fusion=candidate is not None):
                        with patch.object(
                            ops, "project_output", return_value=candidate
                        ):
                            actual = tpu_dsv41_project_attention_output(
                                original,
                                layer,
                                backend,
                                o,
                                positions=torch.tensor([0]),
                                forward_batch=object(),
                                inv_rope=None,
                                wo_a_quant_freqs=None,
                                defer_all_reduce=False,
                            )
                        expected = (o + 1) * 2 if candidate is None else candidate
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_export_into_tpu_custom_call(self):
        # Check single-tile lowering and the two-call export that reuses the
        # first tile's output buffer in the tail call.
        with (
            patch.object(
                pallas_export, "get_torch_tpu_device_kind", return_value="TPU v7"
            ),
            patch.object(kernel, "get_tpu_version", return_value=7),
            pallas_export.tpu_info_for_cpu_export(),
        ):
            lower = export.export(
                jax.jit(kernel.fused_output_projection), platforms=("tpu",)
            )
            for tokens in (64, 416):
                with self.subTest(tokens=tokens):
                    compiled = lower(
                        jax.ShapeDtypeStruct((tokens, 64, 512), jnp.bfloat16),
                        jax.ShapeDtypeStruct((4096, 8192), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((8192,), jnp.float32),
                        jax.ShapeDtypeStruct((8192, 5120), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((5120,), jnp.float32),
                    )
                    (out,) = compiled.out_avals
                    self.assertEqual(out.shape, (tokens, 5120))
                    self.assertEqual(out.dtype, jnp.bfloat16)


@unittest.skipUnless(importlib.util.find_spec("torch_tpu"), "requires TorchTPU")
class TestOutputProjectionNumerics(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        import torch_tpu  # noqa: F401

        if kernel.get_tpu_version() != 7:
            raise unittest.SkipTest("fused output projection is qualified on TPU v7")
        op = ops.register_output_projection_op()

        @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
        def fused(x, wa, sa, wb, sb):
            return op(x, wa, sa, wb, sb)

        cls.fused = staticmethod(fused)

    def test_fused_matches_bf16_wo_a_and_fp8_wo_b(self):
        # Match the default SGLang fallback on a fixed set of decode, prefill
        # and tail rows. Reference math stays on the CPU.
        rng = torch.Generator().manual_seed(4919)
        wa_bf16 = torch.randn(8192, 4096, generator=rng).to(torch.bfloat16)
        wa, sa = quantize_channelwise_fp8_weight(wa_bf16, torch.float8_e4m3fn)
        wb = torch.randn(8192, 5120, generator=rng).to(torch.float8_e4m3fn)
        sb = torch.rand(5120, generator=rng) * 0.025 + 0.005
        device_weights = tuple(t.to("tpu") for t in (wa, sa.squeeze(-1), wb, sb))
        for tokens in (64, 416, 1024):
            with self.subTest(tokens=tokens):
                x = torch.randn(tokens, 64, 512, generator=rng).to(torch.bfloat16)
                x[0].zero_()
                actual = self.fused(x.to("tpu"), *device_weights).cpu()
                self.assertTrue(torch.isfinite(actual).all())
                self.assertEqual(torch.count_nonzero(actual[0]).item(), 0)
                rows = torch.linspace(0, tokens - 1, 16).long()
                y = (
                    torch.einsum(
                        "tgd,grd->tgr",
                        x[rows].reshape(-1, 8, 4096).float(),
                        wa_bf16.reshape(8, 1024, 4096).float(),
                    )
                    .to(torch.bfloat16)
                    .flatten(1)
                )
                y_fp8, y_scale = _quantize_activation_per_row(y, wb.dtype)
                reference = (y_fp8.float() @ wb.float() * y_scale * sb).bfloat16()
                rel_l2 = (
                    actual[rows].float() - reference
                ).norm() / reference.float().norm()
                self.assertLessEqual(rel_l2.item(), 0.05)

    def test_tail_preserves_exact_group_and_row_values(self):
        tokens = 416  # A full tile followed by a shorter tile.
        groups = torch.arange(8)
        wa = torch.zeros((4096, 8192), dtype=torch.bfloat16)
        wa[0, groups * 1024] = 1
        wb = torch.zeros((8192, 5120), dtype=torch.bfloat16)
        wb[groups * 1024, groups] = 1
        x = torch.zeros((tokens, 64, 512), dtype=torch.bfloat16)
        # All operands are exactly representable after the kernel's FP8 row
        # quantizers, and each output column reads only its own attention group.
        row_scale = (2.0 ** (torch.arange(tokens) % 7 - 3)).reshape(-1, 1)
        x[:, groups * 8, 0] = (row_scale * 2.0**groups).bfloat16()
        expected = torch.zeros((tokens, 5120), dtype=torch.bfloat16)
        expected[:, :8] = x[:, groups * 8, 0]
        actual = self.fused(
            x.to("tpu"),
            wa.to(torch.float8_e4m3fn).to("tpu"),
            torch.ones(8192, dtype=torch.float32, device="tpu"),
            wb.to(torch.float8_e4m3fn).to("tpu"),
            torch.ones(5120, dtype=torch.float32, device="tpu"),
        ).cpu()
        self.assertTrue(torch.equal(actual.view(torch.int16), expected.view(torch.int16)))


if __name__ == "__main__":
    unittest.main()
