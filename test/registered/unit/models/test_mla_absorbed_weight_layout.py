"""Absorbed MLA weights (w_kc / w_vc) layout on TPU and its model call sites."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.models import kimi_k3
from sglang.srt.models.deepseek_common import deepseek_weight_loader
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    mla_absorbed_weight_layout,
)
from sglang.srt.models.kimi_k3 import KimiK3LinearForCausalLM
from sglang.srt.models.kimi_k25_eagle3 import Eagle3DeepseekV2ForCausalLM
from sglang.srt.models.kimi_linear import KimiLinearForCausalLM
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.layer_ut_utils import init_single_process_dist
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

NOPE, V, LORA = 32, 64, 128


def _kv_b_weight(heads: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(heads * (NOPE + V), LORA, generator=g).to(torch.bfloat16)


def _split(weight: torch.Tensor):
    return weight.unflatten(0, (-1, NOPE + V)).split([NOPE, V], dim=1)


def _attn(heads: int, seed: int = 0):
    return SimpleNamespace(
        kv_b_proj=SimpleNamespace(weight=_kv_b_weight(heads, seed)),
        qk_nope_head_dim=NOPE,
        v_head_dim=V,
        w_kc=None,
        w_vc=None,
    )


def assert_owned_layout(test, w_kc, w_vc, kv_b_weight):
    """Dense tensors with their own storage, values of the kv_b_proj split."""
    ref_kc, ref_vc = _split(kv_b_weight)
    for t in (w_kc, w_vc):
        test.assertIsNone(t._base)
        test.assertTrue(t.is_contiguous())
        test.assertEqual(t.untyped_storage().nbytes(), t.numel() * t.element_size())
        test.assertNotEqual(t.data_ptr(), kv_b_weight.data_ptr())
    torch.testing.assert_close(w_kc, ref_kc, rtol=0, atol=0)
    torch.testing.assert_close(w_vc, ref_vc.transpose(1, 2), rtol=0, atol=0)


class TestMlaAbsorbedWeightLayout(CustomTestCase):
    def test_tpu_layout_owns_contiguous_storage(self):
        # One head per rank: the w_kc slice is already contiguous there.
        for heads in (1, 4):
            with self.subTest(heads=heads):
                weight = _kv_b_weight(heads)
                with patch.object(deepseek_weight_loader, "_is_tpu", True):
                    w_kc, w_vc = mla_absorbed_weight_layout(*_split(weight))
                self.assertEqual(w_kc.shape, (heads, NOPE, LORA))
                self.assertEqual(w_vc.shape, (heads, LORA, V))
                assert_owned_layout(self, w_kc, w_vc, weight)

    def test_other_platforms_keep_transposed_layout(self):
        weight = _kv_b_weight(4)
        with patch.object(deepseek_weight_loader, "_is_tpu", False):
            w_kc, w_vc = mla_absorbed_weight_layout(*_split(weight))
        self.assertEqual(w_kc.stride(), (NOPE * LORA, 1, NOPE))
        self.assertEqual(w_vc.stride(), (V * LORA, 1, LORA))
        ref_kc, ref_vc = _split(weight)
        torch.testing.assert_close(w_kc, ref_kc, rtol=0, atol=0)
        torch.testing.assert_close(w_vc, ref_vc.transpose(1, 2), rtol=0, atol=0)

    def test_kimi_linear_and_eagle3_post_load(self):
        attn = _attn(4, seed=1)
        kimi_linear = SimpleNamespace(
            config=SimpleNamespace(full_attention_layer_ids=[0]),
            model=SimpleNamespace(
                start_layer=0, end_layer=1, layers=[SimpleNamespace(self_attn=attn)]
            ),
        )
        draft_attn = _attn(4, seed=2)
        eagle3 = SimpleNamespace(
            model=SimpleNamespace(midlayer=SimpleNamespace(self_attn=draft_attn))
        )
        with patch.object(deepseek_weight_loader, "_is_tpu", True):
            KimiLinearForCausalLM.post_load_weights(kimi_linear)
            Eagle3DeepseekV2ForCausalLM.post_load_weights(eagle3)
        for a in (attn, draft_attn):
            assert_owned_layout(self, a.w_kc, a.w_vc, a.kv_b_proj.weight)


def _k3_tiny_config() -> KimiLinearConfig:
    return KimiLinearConfig(
        vocab_size=1024,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=64,
        max_position_embeddings=2048,
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        q_lora_rank=128,
        kv_lora_rank=LORA,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=32,
        v_head_dim=V,
        moe_intermediate_size=256,
        num_experts_per_token=2,
        num_experts=4,
        num_shared_experts=1,
        first_k_dense_replace=1,
        routed_expert_hidden_size=256,
        latent_moe_use_norm=True,
        mla_use_output_gate=True,
        attn_res_block_size=2,
        linear_attn_config={
            "kda_layers": [0, 1, 2],
            "full_attn_layers": [3],
            "short_conv_kernel_size": 4,
            "head_dim": 64,
            "num_heads": 8,
            "num_key_value_heads": 8,
            "q_lora_rank": None,
            "use_full_rank_gate": True,
        },
    )


@unittest.skipIf(torch.cuda.is_available(), "builds on CPU with the TPU flag set")
class TestKimiK3MlaAbsorbedWeights(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist()
        override = get_context().override_server_args(tp_size=1, device="cpu")
        override.install()
        cls.addClassCleanup(override.restore)
        for module in (kimi_k3, deepseek_weight_loader):
            tpu_flag = patch.object(module, "_is_tpu", True)
            tpu_flag.start()
            cls.addClassCleanup(tpu_flag.stop)

    def test_k3_post_load(self):
        config = _k3_tiny_config()
        model = KimiK3LinearForCausalLM(config=config, quant_config=None)
        attns = [
            model.model.layers[i].self_attn for i in config.full_attention_layer_ids
        ]
        self.assertTrue(attns)
        for i, attn in enumerate(attns):
            weight = attn.kv_b_proj.weight
            with torch.no_grad():
                weight.copy_(_kv_b_weight(weight.shape[0] // (NOPE + V), seed=i))
        model.post_load_weights()
        for attn in attns:
            assert_owned_layout(self, attn.w_kc, attn.w_vc, attn.kv_b_proj.weight)


if __name__ == "__main__":
    unittest.main()
