# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for what GLM-5.3-Flash runs on TPU: the torch mHC path, the
gated norm and the checkpoint config."""

import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated
from sglang.kernels.ops.layernorm.mhc import (
    _mhc_post_uses_torch,
    _mhc_pre_uses_torch,
    hc_post,
    hc_pre,
)
from sglang.srt.configs.glm5_next import Glm5NextConfig
from sglang.srt.configs.hybrid_arch import glm5_next_config
from sglang.srt.environ import envs
from sglang.srt.utils.hf_transformers.config import HfModelConfigParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_MHC_ENVS = (
    envs.SGLANG_OPT_USE_TILELANG_MHC_PRE,
    envs.SGLANG_OPT_USE_TILELANG_MHC_POST,
    envs.SGLANG_OPT_FUSE_MHC_POST_PRE,
)


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

    def test_mhc_envs_off_select_the_traceable_torch_path(self):
        with ExitStack() as stack:
            for env in _MHC_ENVS:
                stack.enter_context(env.override(False))
            self.assertTrue(_mhc_pre_uses_torch())
            self.assertTrue(_mhc_post_uses_torch())

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
        text = glm5_next_config(SimpleNamespace(hf_config=cfg, is_draft_model=False))
        self.assertIs(text, cfg.text_config)
        self.assertEqual(text.linear_layer_ids, [0, 1, 2])
        self.assertEqual(text.full_attention_layer_ids, [3])
        self.assertEqual(text.layer_types[-1], "deepseek_sparse_attention")
        self.assertEqual(text.linear_attn_config["gate_lower_bound"], -5.0)

    def test_gated_rms_norm_matches_high_precision_reference(self):
        generator = torch.Generator().manual_seed(53)
        norm = FusedRMSNormGated(8, eps=1e-6, activation="sigmoid")
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
                    output = norm.forward_native(x, gate)
                    torch.testing.assert_close(output, expected)
                    self.assertEqual(output.dtype, dtype)

    def test_torch_mhc_matches_numpy(self):
        rng = np.random.default_rng(53)
        h, d, tokens = 4, 8, 3
        rms_eps, hc_eps, iters = 1e-5, 1e-6, 20
        x = rng.normal(size=(tokens, h, d))
        fn = rng.normal(scale=0.1, size=((2 + h) * h, h * d))
        scale = np.array([0.8, 1.1, 1.3])
        base = rng.normal(scale=0.3, size=(2 + h) * h)
        branch = rng.normal(size=(tokens, d))
        expected, expected_post, expected_comb, expected_output = [], [], [], []
        for row, branch_row in zip(x, branch):
            normalized = row.ravel() / np.sqrt(np.mean(row**2) + rms_eps)
            logits = fn @ normalized
            pre = 1 / (1 + np.exp(-(logits[:h] * scale[0] + base[:h]))) + hc_eps
            post = 2 / (1 + np.exp(-(logits[h : 2 * h] * scale[1] + base[h : 2 * h])))
            matrix = (logits[2 * h :] * scale[2] + base[2 * h :]).reshape(h, h)
            matrix = np.exp(matrix - matrix.max(axis=1, keepdims=True))
            matrix = matrix / matrix.sum(axis=1, keepdims=True) + hc_eps
            matrix /= matrix.sum(axis=0, keepdims=True) + hc_eps
            for _ in range(iters - 1):
                matrix /= matrix.sum(axis=1, keepdims=True) + hc_eps
                matrix /= matrix.sum(axis=0, keepdims=True) + hc_eps
            expected.append(pre @ row)
            expected_post.append(post)
            expected_comb.append(matrix.ravel())
            # H_res mixes streams as matrix.T @ row: a transposed mixer fails here.
            expected_output.append(
                (matrix.T @ row + post[:, None] * branch_row).ravel()
            )
        x_t, fn_t, scale_t, base_t, branch_t = (
            torch.tensor(value, dtype=torch.float32)
            for value in (x, fn, scale, base, branch)
        )
        residual = x_t.reshape(tokens, h * d)
        with ExitStack() as stack:
            for env in _MHC_ENVS:
                stack.enter_context(env.override(False))
            collapsed, comb, post, norm_fused = hc_pre(
                residual,
                fn_t,
                scale_t,
                base_t,
                hc_mult=h,
                rms_eps=rms_eps,
                hc_eps=hc_eps,
                sinkhorn_iters=iters,
            )
            output = hc_post(branch_t, residual, post, comb, hc_mult=h)
        self.assertFalse(norm_fused)
        for actual, reference in zip(
            (collapsed, post, comb, output),
            (expected, expected_post, expected_comb, expected_output),
        ):
            np.testing.assert_allclose(actual.numpy(), reference, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
