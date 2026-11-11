# SPDX-License-Identifier: Apache-2.0
"""Real core-to-plugin early QA dispatch; only exported ops execute on CPU.

This checks seam/adapter wiring and tensor ownership, not accelerator lowering
or FP8 kernel numerical equivalence. Sparse canonical weights keep CPU refs small.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import sglang.srt.models.deepseek_v4 as core
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import env_gate
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

from sglang_tpu.models.deepseek_v41 import preprocess, qa_preprocess
from sglang_tpu.models.deepseek_v41.attention_backend import (
    DSV41TPUAttnBackend,
    rope_reference,
)
from sglang_tpu.models.deepseek_v41.ops import preprocess as qb_ops
from sglang_tpu.models.deepseek_v41.ops import qa_preprocess as qa_ops

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _project(rows, weight, scale):
    # Fixture weights are zero outside the first two input channels.
    return ((rows[:, :2].float() @ weight[:2].float()) * scale).to(torch.bfloat16)


def _norm(rows, gamma, eps):
    values = rows.float()
    return (values * (values.square().mean(-1, keepdim=True) + eps).rsqrt() * gamma).to(
        torch.bfloat16
    )


class _Linear(nn.Module):
    def __init__(self, rank, columns):
        super().__init__()
        weight = torch.zeros(rank, columns, dtype=torch.float8_e4m3fn)
        weight[0] = 1
        weight[1] = -0.5
        self.weight = nn.Parameter(weight, False)
        self.register_buffer("weight_scale", torch.linspace(0.5, 1.5, columns))
        self.bias = None

    def forward(self, rows):
        return _project(rows, self.weight, self.weight_scale), None


def _fixture():
    layer = core.MQALayer.__new__(core.MQALayer)
    nn.Module.__init__(layer)
    layer.is_dsv41 = True
    layer.q_head_norm = False
    layer.hidden_size = 5120
    layer.q_lora_rank = 1280
    layer.head_dim = 512
    layer.qk_rope_head_dim = 64
    layer.n_local_heads = 8
    layer.layer_id = 3
    layer.fuse_wqa_wkv = True
    layer.use_fused_qk_norm_rope = False
    layer.compress_ratio = 4
    layer.compressor = None
    layer.wqkv_a = _Linear(5120, 1792)
    layer.wq_b = _Linear(1280, 4096)
    layer.q_norm = core.RMSNorm(1280, eps=1e-6)
    layer.kv_norm = core.RMSNorm(512, eps=1e-6)
    for norm in (layer.q_norm, layer.kv_norm):
        norm.weight = nn.Parameter(
            torch.linspace(0.75, 1.25, norm.weight.numel()).to(torch.bfloat16), False
        )
    angles = torch.arange(6)[:, None] * torch.full((1, 32), 0.3)
    layer.freqs_cis = torch.cat((angles.cos(), angles.sin()), -1)
    indexed = []
    layer.indexer = lambda **kwargs: indexed.append(kwargs["q_lora"].clone())
    assert qa_preprocess.bind_model_qa(layer) == 1
    assert preprocess.bind_model_preprocessing(layer) == 1
    backend = DSV41TPUAttnBackend.__new__(DSV41TPUAttnBackend)
    backend.rope_tables = {3: layer.freqs_cis}
    return layer, backend, indexed


class TestEarlyPreprocessIntegration(CustomTestCase):
    def _run(self, layer, backend, x, positions, **kwargs):
        with (
            patch(
                "sglang.srt.platforms.current_platform",
                SimpleNamespace(is_out_of_tree=lambda: True),
            ),
            patch.object(core, "is_cp_active", return_value=False),
            patch.object(core, "_is_npu", False),
            patch.object(env_gate, "is_unified_kv_triton", return_value=False),
            patch.object(env_gate, "is_unified_kv_fp8", return_value=False),
        ):
            return layer._forward_prepare(
                x,
                positions,
                SimpleNamespace(forward_mode=ForwardMode.DECODE),
                backend,
                **kwargs,
            )

    def test_bound_qa_bypasses_original_projection_and_norms(self):
        layer, backend, indexed = _fixture()
        positions = torch.tensor([4, 1])
        x = torch.zeros(2, 5120, dtype=torch.bfloat16)
        x[:, :2] = torch.tensor([[2, 1], [3, -2]], dtype=torch.bfloat16)
        projected = _project(x, layer.wqkv_a.weight, layer.wqkv_a.weight_scale)
        expected_rows = _norm(projected[:, :1280], layer.q_norm.weight.float(), 1e-6)
        expected_kv = rope_reference(
            _norm(projected[:, 1280:], layer.kv_norm.weight.float(), 1e-6),
            positions,
            layer.freqs_cis,
        )
        expected_q = rope_reference(
            _project(expected_rows, layer.wq_b.weight, layer.wq_b.weight_scale).view(
                2, 8, 512
            ),
            positions,
            layer.freqs_cis,
        )
        output = torch.full_like(expected_q, -7)

        def qa_reference(rows, weight, scale, gamma_qr, gamma_kv, table, pos, eps):
            raw = _project(rows, weight, scale)
            return (
                _norm(raw[:, :1280], gamma_qr, eps),
                rope_reference(_norm(raw[:, 1280:], gamma_kv, eps), pos, table),
            )

        def qb_reference(rows, weight, scale, table, pos):
            return rope_reference(
                _project(rows, weight, scale).view(-1, 8, 512), pos, table
            )

        with (
            patch.object(qa_ops, "qa_rope", side_effect=qa_reference),
            patch.object(qb_ops, "qb_rope", side_effect=qb_reference),
            patch.object(
                layer.wqkv_a, "forward", side_effect=AssertionError("original QA")
            ),
            patch.object(
                layer.wq_b, "forward", side_effect=AssertionError("original QB")
            ),
            patch.object(
                layer.q_norm, "forward", side_effect=AssertionError("double Q norm")
            ),
            patch.object(
                layer.kv_norm, "forward", side_effect=AssertionError("double KV norm")
            ),
        ):
            q, kv = self._run(
                layer, backend, torch.zeros_like(x), positions, x_quant=x, q_out=output
            )
        self.assertIs(q, output)
        torch.testing.assert_close(q, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(kv, expected_kv, rtol=0, atol=0)
        self.assertEqual(len(indexed), 1)
        self.assertEqual(indexed[0].dtype, torch.bfloat16)
        torch.testing.assert_close(indexed[0], expected_rows, rtol=0, atol=0)

    def test_unsupported_adapters_and_buffers_retain_existing_route(self):
        for unsupported in ("qa", "qb", "packed_buffer", "input_dtype"):
            with self.subTest(unsupported=unsupported):
                layer, backend, indexed = _fixture()
                x = torch.ones(2, 5120, dtype=torch.bfloat16)
                x[:, 1] = -2
                positions = torch.tensor([3, 1])
                if unsupported in ("qa", "qb"):
                    setattr(layer, "_dsv41_" + unsupported, None)
                if unsupported == "input_dtype":
                    x = x.float()
                kwargs = (
                    {"q_rope_out": torch.empty(2, 8, 64)}
                    if unsupported == "packed_buffer"
                    else {}
                )
                raw = _project(x, layer.wqkv_a.weight, layer.wqkv_a.weight_scale)
                rows = layer.q_norm(raw[:, :1280])
                expected_q = rope_reference(
                    _project(rows, layer.wq_b.weight, layer.wq_b.weight_scale).view(
                        2, 8, 512
                    ),
                    positions,
                    layer.freqs_cis,
                )
                expected_kv = rope_reference(
                    layer.kv_norm(raw[:, 1280:]), positions, layer.freqs_cis
                )

                def qb_reference(values, weight, scale, table, pos):
                    return rope_reference(
                        _project(values, weight, scale).view(-1, 8, 512), pos, table
                    )

                with (
                    patch.object(
                        qa_ops, "qa_rope", side_effect=AssertionError("unsupported QA")
                    ),
                    patch.object(qb_ops, "qb_rope", side_effect=qb_reference),
                ):
                    q, kv = self._run(layer, backend, x, positions, **kwargs)
                torch.testing.assert_close(q, expected_q, rtol=0, atol=0)
                torch.testing.assert_close(kv, expected_kv, rtol=0, atol=0)
                self.assertEqual(len(indexed), 1)
                torch.testing.assert_close(indexed[0], rows, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
