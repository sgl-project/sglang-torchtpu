"""CPU contracts for the PD prefill's context-KV DFlash draft (sglang_tpu.models.dflash_context_kv)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Qwen3Config

from sglang.srt.models.dflash import DFlash2DraftModel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.dflash_draft_kv_write import _project_draft_kv
from sglang_tpu.experimental.speculative.dflash.dflash_draft_worker import (
    TpuDflashDraftWorker,
)
from sglang_tpu.models.dflash_context_kv import (
    DFlashContextKVModel,
    as_context_kv_model_config,
    context_kv_unsupported_reason,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_NUM_LAYERS = 2
# Every module that sizes a shard from the parallel config.
_PARALLEL_USERS = (
    "sglang.srt.models.dflash.get_parallel",
    "sglang.srt.layers.linear.get_parallel",
    "sglang_tpu.models.dflash_context_kv.get_parallel",
)


def _draft_hf_config(**dflash_overrides) -> Qwen3Config:
    """A small all-SWA DFlash2 draft shaped like Muse Spark's (GQA, non-causal, v5 rope)."""
    dflash_config = {
        "attention_mode": "gqa",
        "block_size": 8,
        "conv_group_size": 16,
        "conv_kernel_size": 2,
        "mask_token_id": 3,
        "selector_rank": 8,
        "selector_top_k": 4,
        "target_layer_ids": [0, 2, 3],
        **dflash_overrides,
    }
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=_NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        rms_norm_eps=1e-5,
        sliding_window=32,
        use_sliding_window=True,
        layer_types=["sliding_attention"] * _NUM_LAYERS,
        rope_theta=500000.0,
    )
    config.architectures = ["DFlash2DraftModel"]
    config.dflash_config = dflash_config
    config.num_target_layers = 4
    config.is_causal = False
    return config


def _checkpoint(config: Qwen3Config) -> dict[str, torch.Tensor]:
    """Random weights under the HF checkpoint names, including the ones KV never reads."""
    gen = torch.Generator().manual_seed(0)
    h, kv, q = 64, 2 * 16, 4 * 16

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen) * 0.1

    weights = {
        "fc.weight": rand(h, 3 * h),
        "hidden_norm.weight": 1 + rand(h),
        "norm.weight": 1 + rand(h),
        "candidate_selector.hidden_projection.weight": rand(8, h),
        "candidate_selector.predecessor_codebook": rand(64, 8),
        "candidate_selector.successor_codebook": rand(64, 8),
    }
    for i in range(_NUM_LAYERS):
        p = f"layers.{i}."
        weights |= {
            p + "self_attn.q_proj.weight": rand(q, h),
            p + "self_attn.k_proj.weight": rand(kv, h),
            p + "self_attn.v_proj.weight": rand(kv, h),
            p + "self_attn.o_proj.weight": rand(h, q),
            p + "self_attn.q_norm.weight": 1 + rand(16),
            p + "self_attn.k_norm.weight": 1 + rand(16),
            p + "input_layernorm.weight": 1 + rand(h),
            p + "post_attention_layernorm.weight": 1 + rand(h),
            p + "mlp.gate_proj.weight": rand(96, h),
            p + "mlp.up_proj.weight": rand(96, h),
            p + "mlp.down_proj.weight": rand(h, 96),
        }
    return weights


def _model_config(hf_config) -> SimpleNamespace:
    return SimpleNamespace(hf_config=hf_config, hf_text_config=hf_config)


class TestDFlashContextKVModel(CustomTestCase):
    def setUp(self):
        parallel = SimpleNamespace(tp_size=1, tp_rank=0)
        self._patches = [patch(name, return_value=parallel) for name in _PARALLEL_USERS]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def _models(self) -> tuple[DFlash2DraftModel, DFlashContextKVModel]:
        hf_config = _draft_hf_config()
        weights = _checkpoint(hf_config)
        full = DFlash2DraftModel(hf_config)
        full.load_weights(weights.items())
        context_kv = DFlashContextKVModel(hf_config)
        context_kv.load_weights(weights.items())
        return full, context_kv

    def test_holds_only_the_context_kv_parameters(self):
        _, model = self._models()
        names = set(dict(model.named_parameters()))
        expected = {"fc.weight", "hidden_norm.weight"}
        for i in range(_NUM_LAYERS):
            expected |= {
                f"layers.{i}.self_attn.kv_proj.weight",
                f"layers.{i}.self_attn.k_norm.weight",
            }
        self.assertEqual(names, expected)

    def test_context_kv_matches_the_full_draft(self):
        """Every layer's normed K, V and the rope rows equal the full DFlash2 draft's, bit for bit."""
        full, context_kv = self._models()
        target_hidden = torch.randn(
            11, 3 * 64, generator=torch.Generator().manual_seed(1)
        )
        positions = torch.tensor([0, 1, 2, 3, 4, 5, 6, 40, 41, 42, 200])
        with torch.no_grad():
            outs = [
                _project_draft_kv(
                    draft_model=model,
                    ctx_hidden=model.project_target_hidden(target_hidden),
                    ctx_positions=positions,
                )
                for model in (full, context_kv)
            ]
        (k_ref, v_ref, rope_ref), (k, v, rope) = outs
        torch.testing.assert_close(k, k_ref, rtol=0, atol=0)
        torch.testing.assert_close(v, v_ref, rtol=0, atol=0)
        assert rope is not None and rope_ref is not None
        for got, want in zip(rope, rope_ref):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        # apply_k_rope agrees too, for writers that rotate per layer.
        k0 = k[0].reshape(11, -1)
        torch.testing.assert_close(
            context_kv.layers[0].self_attn.apply_k_rope(positions, k0),
            full.layers[0].self_attn.apply_k_rope(positions, k0),
            rtol=0,
            atol=0,
        )

    def test_missing_context_kv_weight_raises(self):
        hf_config = _draft_hf_config()
        for name in ("fc.weight", "layers.1.self_attn.v_proj.weight"):
            with self.subTest(name=name):
                weights = _checkpoint(hf_config)
                del weights[name]
                with self.assertRaisesRegex(ValueError, "checkpoint lacks"):
                    DFlashContextKVModel(hf_config).load_weights(weights.items())

    def test_fc_shape_mismatch_raises(self):
        hf_config = _draft_hf_config()
        weights = _checkpoint(hf_config)
        weights["fc.weight"] = torch.zeros(64, 2 * 64)
        with self.assertRaisesRegex(ValueError, "fc.weight shape"):
            DFlashContextKVModel(hf_config).load_weights(weights.items())

    def test_forward_raises(self):
        _, model = self._models()
        with self.assertRaisesRegex(NotImplementedError, "never drafts"):
            model(torch.zeros(1, dtype=torch.long), torch.zeros(1), None)

    def test_kv_heads_must_shard_evenly(self):
        with patch(
            "sglang_tpu.models.dflash_context_kv.get_parallel",
            return_value=SimpleNamespace(tp_size=4, tp_rank=0),
        ):
            with self.assertRaisesRegex(NotImplementedError, "replicate KV heads"):
                DFlashContextKVModel(_draft_hf_config())


class TestContextKVModelConfig(CustomTestCase):
    def test_renames_a_copy_of_the_architecture(self):
        hf_config = _draft_hf_config()
        model_config = _model_config(hf_config)
        out = as_context_kv_model_config(model_config)
        self.assertEqual(out.hf_config.architectures, ["DFlashContextKVModel"])
        self.assertIs(out.hf_text_config, out.hf_config)
        # get_config caches the HF config: the shared object keeps its name.
        self.assertEqual(hf_config.architectures, ["DFlash2DraftModel"])
        self.assertIs(model_config.hf_config, hf_config)

    def test_rejects_drafts_whose_kv_it_cannot_compute(self):
        mla = _draft_hf_config(attention_mode="mla")
        laguna = _draft_hf_config()
        laguna.architectures = ["DFlashLagunaForCausalLM"]
        for hf_config, reason in ((mla, "MLA"), (laguna, "not a dense GQA")):
            with self.subTest(reason=reason):
                model_config = _model_config(hf_config)
                self.assertIn(reason, context_kv_unsupported_reason(model_config))
                with self.assertRaisesRegex(NotImplementedError, reason):
                    as_context_kv_model_config(model_config)
        self.assertIsNone(
            context_kv_unsupported_reason(_model_config(_draft_hf_config()))
        )


class TestDflashDraftWorkerModelConfig(CustomTestCase):
    def _prepare(self, mode: str, hf_config) -> SimpleNamespace:
        worker = TpuDflashDraftWorker.__new__(TpuDflashDraftWorker)
        with patch(
            "sglang_tpu.experimental.speculative.dflash.dflash_draft_worker.get_disagg",
            return_value=SimpleNamespace(disaggregation_mode=mode),
        ):
            return worker.prepare_draft_model_config(_model_config(hf_config))

    def test_only_a_pd_prefill_loads_the_context_kv_model(self):
        for mode, arch in (
            ("prefill", "DFlashContextKVModel"),
            ("decode", "DFlash2DraftModel"),
            ("null", "DFlash2DraftModel"),
        ):
            with self.subTest(mode=mode):
                out = self._prepare(mode, _draft_hf_config())
                self.assertEqual(out.hf_config.architectures, [arch])

    def test_prefill_keeps_a_draft_it_cannot_slice(self):
        with self.assertLogs(level="WARNING") as logs:
            out = self._prepare("prefill", _draft_hf_config(attention_mode="mla"))
        self.assertEqual(out.hf_config.architectures, ["DFlash2DraftModel"])
        self.assertIn("MLA", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
