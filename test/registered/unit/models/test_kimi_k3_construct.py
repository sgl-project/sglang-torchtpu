"""Kimi-K3 construction and post-load on TPU."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.models import kimi_k3
from sglang.srt.models.kimi_k3 import (
    KimiK3DeltaAttention,
    KimiK3LinearForCausalLM,
    KimiK3MoE,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.layer_ut_utils import init_single_process_dist
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _tiny_config() -> KimiLinearConfig:
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
        kv_lora_rank=128,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=64,
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
class TestKimiK3ConstructOnTpu(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist()
        override = get_context().override_server_args(tp_size=1, device="cpu")
        override.install()
        cls.addClassCleanup(override.restore)
        tpu_flag = patch.object(kimi_k3, "_is_tpu", True)
        tpu_flag.start()
        cls.addClassCleanup(tpu_flag.stop)

    def test_moe_pad_count_goes_to_topk_not_experts(self):
        """TopK masks padded rows on TPU; experts keep DeepEPMoE's signature."""
        import inspect
        from types import SimpleNamespace

        from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        class ExpertsCallSpy(torch.nn.Module):
            """Routed-experts stand-in: records each call, returns zeros."""

            def __init__(self, out_dim: int):
                super().__init__()
                self.out_dim = out_dim
                self.calls = []

            def forward(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return args[0].new_zeros(args[0].shape[0], self.out_dim)

        model = KimiK3LinearForCausalLM(config=_tiny_config(), quant_config=None)
        moe = next(
            layer.mlp
            for layer in model.model.layers
            if isinstance(layer.mlp, KimiK3MoE)
        )
        moe.experts = ExpertsCallSpy(moe.moe_hidden_size)
        hidden = torch.zeros(4, 512, dtype=moe.gate.weight.dtype)
        count = torch.tensor(3, dtype=torch.int32)
        forward_batch = SimpleNamespace(moe_num_token_non_padded=lambda: count)
        topk_output = StandardTopKOutput(
            torch.zeros(4, 2), torch.zeros(4, 2, dtype=torch.int32), torch.zeros(4, 4)
        )
        for on_tpu in (True, False):
            moe.experts.calls.clear()
            with (
                self.subTest(on_tpu=on_tpu),
                patch.object(kimi_k3, "_is_tpu", on_tpu),
                patch.object(moe.topk, "forward", return_value=topk_output) as topk,
            ):
                moe(hidden, forward_batch=forward_batch)
                self.assertIs(
                    topk.call_args.kwargs["num_token_non_padded"],
                    count if on_tpu else None,
                )
                ((args, kwargs),) = moe.experts.calls
                self.assertIs(args[1], topk_output)
                inspect.signature(DeepEPMoE.forward).bind(moe.experts, *args, **kwargs)

    def test_model_builds_without_side_streams(self):
        """No torch.cuda.Stream is created on TPU."""
        model = KimiK3LinearForCausalLM(config=_tiny_config(), quant_config=None)
        self.assertIsNone(model.model.alt_streams)

    def test_mla_binds_mha_kv_b_proj_after_load(self):
        """Loading reads the unaliased kv_b_proj key; the compiled forward then
        finds attn_mha.kv_b_proj bound instead of setting it in the trace."""
        from sglang.srt.model_loader.loader import ShardedStateLoader
        from sglang.srt.models.kimi_k3 import KimiK3MLAAttention

        model = KimiK3LinearForCausalLM(config=_tiny_config(), quant_config=None)
        mla = [
            layer.self_attn
            for layer in model.model.layers
            if isinstance(layer.self_attn, KimiK3MLAAttention)
        ]
        self.assertTrue(mla)
        load_keys = [
            key
            for key in ShardedStateLoader._filter_subtensors(model.state_dict())
            if "kv_b_proj" in key
        ]
        self.assertTrue(load_keys)
        self.assertFalse([key for key in load_keys if "attn_mha" in key])
        model.post_load_weights()
        for attn in mla:
            self.assertIs(attn.attn_mha.kv_b_proj, attn.kv_b_proj)

    def test_post_load_resolves_only_the_tpu_forward_properties(self):
        """A cached property first read inside the compiled trace recompiles
        while serving; the CUDA-only ones may probe CUDA kernels."""
        from functools import cached_property

        model = KimiK3LinearForCausalLM(config=_tiny_config(), quant_config=None)
        model.post_load_weights()
        cached = {
            name
            for name, attr in vars(KimiK3MoE).items()
            if isinstance(attr, cached_property)
        }
        moes = [
            layer.mlp
            for layer in model.model.layers
            if isinstance(layer.mlp, KimiK3MoE)
        ]
        self.assertTrue(moes)
        for moe in moes:
            self.assertEqual(
                cached & moe.__dict__.keys(),
                {
                    "_eligible_for_fused_front",
                    "_ep_front_eligible",
                    "_routed_needs_reduce",
                },
            )

    def test_post_load_keeps_unfused_weights(self):
        """CUDA-only merged layouts stay unset on TPU."""
        model = KimiK3LinearForCausalLM(config=_tiny_config(), quant_config=None)
        model.post_load_weights()
        layers = model.model.layers
        moes = [l.mlp for l in layers if isinstance(l.mlp, KimiK3MoE)]
        kdas = [
            l.self_attn for l in layers if isinstance(l.self_attn, KimiK3DeltaAttention)
        ]
        self.assertTrue(moes and kdas)
        for moe in moes:
            self.assertIsNone(moe._front_w)
        for kda in kdas:
            self.assertIsNone(kda._bfa_w)
            self.assertFalse(kda._kda_fused_decode_ready)


if __name__ == "__main__":
    unittest.main()
