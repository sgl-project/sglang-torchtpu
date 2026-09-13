# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for GLM's TPU text configuration, router and mHC."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.configs.glm5_next import Glm5NextConfig, Glm5NextTextConfig
from sglang.srt.configs.hybrid_arch import kimi_linear_config
from sglang.srt.disaggregation.utils import get_dsa_seed_metadata_dim
from sglang.srt.models.deepseek_v2 import MoEGate
from sglang.srt.models.glm4_moe import GlmMoeDsaForCausalLM
from sglang.srt.models.glm5_next_tpu import (
    Glm5NextLinearAttention,
    TpuRMSNormGated,
    hc_post,
    hc_pre,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.hf_transformers.config import HfModelConfigParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestGlm5NextTpu(unittest.TestCase):
    def test_full_glm_explicit_rotary_width_survives_head_dim_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "glm_moe_dsa",
                        "architectures": ["GlmMoeDsaForCausalLM"],
                        "head_dim": 192,
                        "qk_nope_head_dim": 192,
                        "qk_rope_head_dim": 64,
                    }
                )
            )
            config = HfModelConfigParser().parse(directory, trust_remote_code=False)
        self.assertEqual(config.qk_rope_head_dim, 64)
        self.assertEqual(config.qk_head_dim, 256)

    def test_full_glm_dense_dsa_rejects_context_beyond_selection_budget(self):
        config = SimpleNamespace(index_topk=2048, tpu_dense_dsa_max_context=2048)
        with patch(
            "sglang.srt.models.glm4_moe.get_server_args",
            return_value=SimpleNamespace(context_length=2049),
        ):
            with self.assertRaisesRegex(ValueError, "context-length"):
                GlmMoeDsaForCausalLM(config)
        config.tpu_dense_dsa_max_context = 4096
        with patch(
            "sglang.srt.models.glm4_moe.get_server_args",
            return_value=SimpleNamespace(context_length=4096),
        ):
            with self.assertRaisesRegex(ValueError, "index_topk"):
                GlmMoeDsaForCausalLM(config)

    def test_dense_dsa_does_not_allocate_sparse_seed_metadata(self):
        flash = SimpleNamespace(
            architectures=["Glm5NextForConditionalGeneration"],
            index_share_for_mtp_iteration=True,
            index_topk=2048,
        )
        full = SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"],
            index_share_for_mtp_iteration=True,
            index_topk=2048,
            tpu_dense_dsa_max_context=2048,
        )
        self.assertEqual(get_dsa_seed_metadata_dim(flash), 0)
        self.assertEqual(get_dsa_seed_metadata_dim(full), 0)
        del full.tpu_dense_dsa_max_context
        self.assertEqual(get_dsa_seed_metadata_dim(full), 2048)

    def test_kda_output_keeps_batch_one_token_axis(self):
        config = Glm5NextTextConfig(
            hidden_size=16, linear_num_heads=2, linear_head_dim=4
        )
        with get_parallel().override(
            tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0
        ):
            attn = Glm5NextLinearAttention(0, config.hidden_size, config)
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: False, is_decode=lambda: True)
        )
        with patch.object(
            attn.attn, "forward", return_value=torch.ones(1, 2, 4)
        ), patch("sglang.srt.layers.linear.get_tp_group", return_value=None):
            output = attn(torch.ones(1, config.hidden_size), batch)
        self.assertEqual(output.shape, (1, config.hidden_size))

    def test_nested_checkpoint_config_preserves_attention_types_and_gate(self):
        cfg = Glm5NextConfig(
            text_config={
                "num_hidden_layers": 4,
                "layer_types": ["linear_attention"] * 3 + ["deepseek_sparse_attention"],
                "linear_attn_config": {
                    "num_heads": 64,
                    "head_dim": 128,
                    "short_conv_kernel_size": 4,
                    "gate_lower_bound": -5.0,
                    "kda_layers": [0, 1, 2],
                    "full_attn_layers": [3],
                },
            }
        )
        text = kimi_linear_config(SimpleNamespace(hf_config=cfg))
        self.assertIs(text, cfg.text_config)
        self.assertEqual(text.linear_layer_ids, [0, 1, 2])
        self.assertEqual(text.full_attention_layer_ids, [3])
        self.assertEqual(text.layer_types[-1], "deepseek_sparse_attention")
        self.assertEqual(text.linear_attn_config["gate_lower_bound"], -5.0)

    def test_hyper_connection_collapse_and_residual_orientation(self):
        cfg = SimpleNamespace(
            hc_mult=3, hc_eps=1e-6, rms_norm_eps=1e-5, hc_sinkhorn_iters=20
        )
        x = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        fn = torch.zeros(15, 12)
        collapsed, post, comb = hc_pre(x, fn, torch.ones(3), torch.zeros(15), cfg)
        torch.testing.assert_close(collapsed, x.sum(1) * (0.5 + cfg.hc_eps))
        torch.testing.assert_close(post, torch.ones_like(post))
        torch.testing.assert_close(comb.sum(-1), torch.ones(1, 3), atol=1e-5, rtol=0)
        # A cyclic residual mixer catches accidentally transposing H_res.
        permutation = torch.tensor(
            [[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]]
        )
        output = hc_post(torch.zeros(1, 4), x, post, permutation)
        torch.testing.assert_close(output, x[:, [2, 0, 1]])

    def test_tpu_router_preserves_fp32_weights_and_logits(self):
        cfg = SimpleNamespace(
            n_routed_experts=3,
            hidden_size=4,
            topk_method="noaux_tc",
            moe_router_dtype="float32",
        )
        with patch("sglang.srt.models.deepseek_v2._is_tpu", True):
            old_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.bfloat16)
                gate = MoEGate(cfg, quant_config=None)
            finally:
                torch.set_default_dtype(old_dtype)
            self.assertEqual(gate.weight.dtype, torch.float32)
            with torch.no_grad():
                gate.weight.copy_(torch.arange(12).reshape(3, 4).float() / 10)
            x = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
            torch.testing.assert_close(gate(x), F.linear(x.float(), gate.weight))
            self.assertEqual(gate(x).dtype, torch.float32)

    def test_gated_rms_norm_matches_high_precision_reference(self):
        generator = torch.Generator().manual_seed(53)
        norm = TpuRMSNormGated(8, eps=1e-6)
        with torch.no_grad():
            norm.weight.copy_(torch.randn(8, generator=generator))
        for tokens in (1, 5):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(tokens=tokens, dtype=dtype):
                    x = torch.randn(tokens, 2, 8, generator=generator).to(dtype)
                    gate = torch.linspace(-20, 20, x.numel()).reshape_as(x).to(dtype)
                    xf = x.double()
                    expected = (
                        xf / (xf.square().mean(-1, keepdim=True) + norm.eps).sqrt()
                    )
                    expected = (
                        expected * norm.weight.double() / (1 + (-gate.double()).exp())
                    ).to(dtype)
                    output = norm(x, gate)
                    torch.testing.assert_close(output, expected)
                    self.assertEqual(output.dtype, dtype)

    def test_nonuniform_hyper_connections_match_numpy(self):
        rng = np.random.default_rng(53)
        h, d, tokens = 4, 8, 3
        cfg = SimpleNamespace(
            hc_mult=h, hc_eps=1e-6, rms_norm_eps=1e-5, hc_sinkhorn_iters=20
        )
        x = rng.normal(size=(tokens, h, d))
        fn = rng.normal(scale=0.1, size=((2 + h) * h, h * d))
        scale = np.array([0.8, 1.1, 1.3])
        base = rng.normal(scale=0.3, size=(2 + h) * h)
        branch = rng.normal(size=(tokens, d))
        expected, expected_post, expected_comb, expected_output = [], [], [], []
        for row, branch_row in zip(x, branch):
            normalized = row.ravel() / np.sqrt(np.mean(row**2) + cfg.rms_norm_eps)
            logits = fn @ normalized
            pre = 1 / (1 + np.exp(-(logits[:h] * scale[0] + base[:h]))) + cfg.hc_eps
            post = 2 / (1 + np.exp(-(logits[h : 2 * h] * scale[1] + base[h : 2 * h])))
            matrix = (logits[2 * h :] * scale[2] + base[2 * h :]).reshape(h, h)
            matrix = np.exp(matrix - matrix.max(axis=1, keepdims=True))
            matrix = matrix / matrix.sum(axis=1, keepdims=True) + cfg.hc_eps
            matrix /= matrix.sum(axis=0, keepdims=True) + cfg.hc_eps
            for _ in range(cfg.hc_sinkhorn_iters - 1):
                matrix /= matrix.sum(axis=1, keepdims=True) + cfg.hc_eps
                matrix /= matrix.sum(axis=0, keepdims=True) + cfg.hc_eps
            expected.append(pre @ row)
            expected_post.append(post)
            expected_comb.append(matrix)
            expected_output.append(matrix.T @ row + post[:, None] * branch_row)
        inputs = [
            torch.tensor(value, dtype=torch.float32) for value in (x, fn, scale, base)
        ]
        collapsed, post, comb = hc_pre(*inputs, cfg)
        output = hc_post(
            torch.tensor(branch, dtype=torch.float32), inputs[0], post, comb
        )
        for actual, reference in zip(
            (collapsed, post, comb, output),
            (expected, expected_post, expected_comb, expected_output),
        ):
            np.testing.assert_allclose(actual.numpy(), reference, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
