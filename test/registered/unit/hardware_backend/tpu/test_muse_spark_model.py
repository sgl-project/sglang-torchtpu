"""CPU tests for the Muse Spark TPU model's checkpoint mapping, config and
attention construction."""

import unittest

import torch

from sglang.srt.configs.model_config import get_hybrid_layer_ids
from sglang.srt.configs.muse_spark import MuseSparkConfig
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.layer_ut_utils import init_single_process_dist
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.models.muse_spark import (
    MuseSparkAttention,
    MuseSparkInterleavedRope,
    _param_name,
    _to_interleaved_rope,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_LAYER = "model.language_model.layers.3."


def _rotate_half_rope(x, positions, base):
    """HF reference RoPE: rotates the pairs (i, i + d/2)."""
    head_dim = x.shape[-1]
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
    )
    angles = positions.double()[:, None] * inv_freq
    cos = torch.cat([angles.cos(), angles.cos()], -1).float()[:, None]
    sin = torch.cat([angles.sin(), angles.sin()], -1).float()[:, None]
    first, second = x.chunk(2, dim=-1)
    return x * cos + torch.cat([-second, first], -1) * sin


class TestMuseSparkCheckpointMapping(CustomTestCase):
    def test_param_names(self):
        cases = {
            "lm_head.weight": "lm_head.weight",
            "model.language_model.norm.weight": "model.norm.weight",
            _LAYER
            + "self_attn.q_proj.weight": "model.layers.3.self_attn.q_proj.weight",
            _LAYER
            + "self_attn.gate_proj.weight": "model.layers.3.self_attn.gate_proj.weight",
            _LAYER + "mlp.gate.weight": "model.layers.3.mlp.gate.weight",
            _LAYER
            + "mlp.gate.e_score_correction_bias": "model.layers.3.mlp.e_score_correction_bias",
            _LAYER
            + "mlp.experts.post_expert_norm.weight": "model.layers.3.mlp.post_expert_norm.weight",
            _LAYER
            + "mlp.experts.gate_up_proj": "model.layers.3.mlp.experts.w13_weight",
            _LAYER
            + "mlp.experts.gate_up_proj_weight_scale": "model.layers.3.mlp.experts.w13_weight_scale",
            _LAYER
            + "mlp.experts.gate_up_proj_weight_scale_2": "model.layers.3.mlp.experts.w13_weight_scale_2",
            _LAYER + "mlp.experts.down_proj": "model.layers.3.mlp.experts.w2_weight",
            _LAYER
            + "mlp.experts.down_proj_weight_scale_2": "model.layers.3.mlp.experts.w2_weight_scale_2",
            _LAYER + "mlp.experts.gate_up_proj_input_scale": None,
            "model.vision_tower.layers.0.attn.qkv.weight": None,
            "model.audio_tower.proj.weight": None,
        }
        for checkpoint_name, expected in cases.items():
            with self.subTest(checkpoint_name):
                self.assertEqual(_param_name(checkpoint_name), expected)

    def test_qk_permutation_matches_half_split_rope(self):
        torch.manual_seed(0)
        heads, head_dim, hidden, tokens, base = 3, 8, 16, 5, 500000.0
        weight = torch.randn(heads * head_dim, hidden)
        x = torch.randn(tokens, hidden)
        positions = torch.tensor([0, 1, 7, 100, 4095])

        reference = _rotate_half_rope(
            (x @ weight.t()).view(tokens, heads, head_dim), positions, base
        )
        rope = MuseSparkInterleavedRope(head_dim, max_position=4096, base=base)
        permuted = x @ _to_interleaved_rope(weight, heads).t()
        rotated, _ = rope(positions, permuted, permuted)
        # Interleaved pair i holds the reference's dims (i, i + d/2).
        expected = (
            reference.view(tokens, heads, 2, head_dim // 2)
            .transpose(2, 3)
            .reshape(tokens, heads * head_dim)
        )
        torch.testing.assert_close(rotated, expected, rtol=1e-5, atol=1e-5)


class TestMuseSparkConfig(CustomTestCase):
    def test_nested_config_and_swa_layers(self):
        layer_types = ["sliding_attention", "full_attention"] * 2
        config = MuseSparkConfig(
            architectures=["MuseSparkForConditionalGeneration"],
            text_config={
                "num_hidden_layers": 4,
                "layer_types": layer_types,
                "layer_rope_theta": [500000.0, 0, 500000.0, 0],
                "mlp_layer_types": ["sparse"] * 4,
            },
        )
        text = config.text_config
        self.assertEqual(text.model_type, "muse_spark_text")
        self.assertTrue(text.is_hybrid_swa)
        swa, full = get_hybrid_layer_ids(config.architectures, text)
        self.assertEqual(swa, [0, 2])
        self.assertEqual(full, [1, 3])


class TestMuseSparkAttentionConstruction(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist()
        override = get_context().override_server_args(tp_size=1, device="cpu")
        override.install()
        cls.addClassCleanup(override.restore)

    def test_projections_shard_over_attention_tp(self):
        heads, kv_heads, head_dim, hidden = 4, 2, 32, 128
        text = MuseSparkConfig(
            text_config={
                "hidden_size": hidden,
                "head_dim": head_dim,
                "num_attention_heads": heads,
                "num_key_value_heads": kv_heads,
                "num_hidden_layers": 2,
                "layer_types": ["sliding_attention", "full_attention"],
                "layer_rope_theta": [500000.0, 0],
                "mlp_layer_types": ["sparse"] * 2,
            }
        ).text_config
        rope = MuseSparkInterleavedRope(head_dim, 64, 500000.0)
        attn = MuseSparkAttention(text, 0, rope, prefix="model.layers.0.self_attn")
        self.assertEqual(
            tuple(attn.qkv_proj.weight.shape),
            ((heads + 2 * kv_heads) * head_dim, hidden),
        )
        self.assertEqual(tuple(attn.gate_proj.weight.shape), (heads * head_dim, hidden))
        self.assertEqual(tuple(attn.o_proj.weight.shape), (hidden, heads * head_dim))
        # The local head counts come from attention TP, so the weights split there.
        for proj in (attn.qkv_proj, attn.gate_proj, attn.o_proj):
            self.assertEqual(proj.parallel_group, "attn_tp")


if __name__ == "__main__":
    unittest.main()
