"""CPU tests for the Muse Spark TPU model's checkpoint mapping, config, and PP."""

import unittest

import torch

from sglang.srt.configs.model_config import get_hybrid_layer_ids
from sglang.srt.configs.muse_spark import MuseSparkConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.models.muse_spark import (
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


class _StubAttn(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.proj = torch.nn.Linear(hidden, hidden, bias=False, dtype=torch.bfloat16)

    def forward(self, positions, x, forward_batch):
        return self.proj(x)


class _StubMlp(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.proj = torch.nn.Linear(hidden, hidden, bias=False, dtype=torch.bfloat16)

    def forward(self, x, router_input):
        return self.proj(x)


def _build_stub_text_stage(
    num_layers: int,
    start_layer: int,
    end_layer: int,
    *,
    is_first: bool,
    is_last: bool,
    ref_layers: list | None = None,
    ref_embed: torch.nn.Module | None = None,
    ref_norm: torch.nn.Module | None = None,
):
    from types import SimpleNamespace

    from sglang.srt.layers.utils import PPMissingLayer
    from sglang_tpu.models.muse_spark import (
        MuseSparkGatedResidual,
        MuseSparkRMSNorm,
        MuseSparkTextModel,
    )

    hidden, vocab = 16, 32
    stage = torch.nn.Module.__new__(MuseSparkTextModel)
    torch.nn.Module.__init__(stage)
    stage.config = SimpleNamespace(
        hidden_size=hidden,
        vocab_size=vocab,
        num_hidden_layers=num_layers,
        rms_norm_eps=1e-5,
        post_norm_eps=1e-8,
        residual_gate_temperature=1.0,
    )
    stage.pp_group = SimpleNamespace(is_first_rank=is_first, is_last_rank=is_last)
    stage.layers_to_capture = []
    stage._capture_slots_by_layer = {}
    stage.start_layer = start_layer
    stage.end_layer = end_layer
    stage.embed_tokens = (
        (
            ref_embed
            if ref_embed is not None
            else torch.nn.Embedding(vocab, hidden, dtype=torch.bfloat16)
        )
        if is_first
        else PPMissingLayer()
    )

    def _make_layer(idx: int):
        if ref_layers is not None:
            return ref_layers[idx]
        layer = torch.nn.Module()
        layer.input_layernorm = MuseSparkRMSNorm(hidden, eps=1e-5, gain_center=1.0)
        torch.nn.init.normal_(layer.input_layernorm.weight, std=0.1)
        layer.self_attn = _StubAttn(hidden)
        layer.post_attention_layernorm = MuseSparkRMSNorm(
            hidden, eps=1e-8, has_weight=False
        )
        layer.post_attention_residual_gate = MuseSparkGatedResidual(
            hidden, temperature=1.0
        )
        torch.nn.init.normal_(layer.post_attention_residual_gate.gate, std=0.5)
        layer.post_attention_residual_gate.precompute_coefficients()
        layer.pre_feedforward_layernorm = MuseSparkRMSNorm(
            hidden, eps=1e-5, gain_center=1.0
        )
        torch.nn.init.normal_(layer.pre_feedforward_layernorm.weight, std=0.1)
        layer.mlp = _StubMlp(hidden)
        layer.post_feedforward_layernorm = MuseSparkRMSNorm(
            hidden, eps=1e-8, gain_center=1.0, scale_input=True
        )
        torch.nn.init.normal_(layer.post_feedforward_layernorm.weight, std=0.1)
        layer.post_feedforward_residual_gate = MuseSparkGatedResidual(
            hidden, temperature=1.0
        )
        torch.nn.init.normal_(layer.post_feedforward_residual_gate.gate, std=0.5)
        layer.post_feedforward_residual_gate.precompute_coefficients()

        def _fwd(positions, stream, forward_batch, lyr=layer):
            attn_in = lyr.input_layernorm(stream).to(torch.bfloat16)
            stream = lyr.post_attention_residual_gate(
                stream,
                lyr.post_attention_layernorm(
                    lyr.self_attn(positions, attn_in, forward_batch)
                ),
            )
            router_in = lyr.pre_feedforward_layernorm(stream)
            return lyr.post_feedforward_residual_gate(
                stream,
                lyr.post_feedforward_layernorm(
                    lyr.mlp(router_in.to(torch.bfloat16), router_in)
                ),
            )

        layer.forward = _fwd
        return layer

    stage.layers = torch.nn.ModuleList(
        [
            _make_layer(i) if start_layer <= i < end_layer else PPMissingLayer()
            for i in range(num_layers)
        ]
    )
    if is_last:
        if ref_norm is not None:
            stage.norm = ref_norm
        else:
            stage.norm = MuseSparkRMSNorm(hidden, eps=1e-5)
            torch.nn.init.normal_(stage.norm.weight, std=0.1)
    else:
        stage.norm = PPMissingLayer()
    return stage


class TestMuseSparkPipelineParallel(CustomTestCase):
    def test_declares_fp32_pp_transfer_dtype(self):
        from sglang_tpu.models.muse_spark import (
            MuseSparkForConditionalGeneration as TextWrapper,
        )
        from sglang_tpu.models.muse_spark_omni import (
            MuseSparkForConditionalGeneration as OmniWrapper,
        )

        self.assertIs(TextWrapper.pp_transfer_dtype, torch.float32)
        self.assertIs(OmniWrapper.pp_transfer_dtype, torch.float32)
        self.assertTrue(TextWrapper.relays_pp_aux_hidden_states)
        self.assertTrue(OmniWrapper.relays_pp_aux_hidden_states)

    def test_fp32_pp_handoff_and_aux_relay_match_pp1_bit_for_bit(self):
        from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

        torch.manual_seed(42)
        num_layers = 6
        capture_ids = [5, 0, 3, 0]  # non-sorted + duplicate slot coverage
        pp1 = _build_stub_text_stage(
            num_layers, 0, num_layers, is_first=True, is_last=True
        )
        pp1._set_layers_to_capture(capture_ids)

        stages = [
            _build_stub_text_stage(
                num_layers,
                s,
                e,
                is_first=(s == 0),
                is_last=(e == num_layers),
                ref_layers=list(pp1.layers),
                ref_embed=pp1.embed_tokens,
                ref_norm=pp1.norm,
            )
            for s, e in ((0, 2), (2, 4), (4, 6))
        ]
        for st in stages:
            st._set_layers_to_capture(capture_ids)

        input_ids = torch.tensor([1, 7, 13, 29])
        positions = torch.arange(4)

        ref_hidden, ref_aux = pp1(input_ids, positions, None)

        proxy = None
        for st in stages[:-1]:
            proxy = st(input_ids, positions, None, pp_proxy_tensors=proxy)
            self.assertIsInstance(proxy, PPProxyTensors)
            self.assertEqual(proxy.tensors["hidden_states"].dtype, torch.float32)
            self.assertEqual(proxy.tensors["residual"].dtype, torch.float32)
            self.assertEqual(proxy.tensors["aux_hidden_states"].dtype, torch.bfloat16)
        pp_hidden, pp_aux = stages[-1](
            input_ids, positions, None, pp_proxy_tensors=proxy
        )
        self.assertTrue(torch.equal(pp_hidden, ref_hidden))
        self.assertTrue(torch.equal(pp_aux, ref_aux))

        # Truncating the inter-stage residual stream to bf16 breaks bit-exact parity.
        bf16_proxy = None
        for st in stages[:-1]:
            out = st(input_ids, positions, None, pp_proxy_tensors=bf16_proxy)
            bf16_proxy = PPProxyTensors(
                {
                    "hidden_states": out.tensors["hidden_states"].to(torch.bfloat16),
                    "residual": out.tensors["residual"].to(torch.bfloat16),
                    "aux_hidden_states": out.tensors["aux_hidden_states"],
                }
            )
        bf16_hidden, _ = stages[-1](
            input_ids, positions, None, pp_proxy_tensors=bf16_proxy
        )
        self.assertFalse(torch.equal(bf16_hidden, ref_hidden))

    def test_dflash_marks_every_stage_and_unpacks_on_last_stage_only(self):
        from types import SimpleNamespace

        from sglang_tpu.models.muse_spark import (
            MuseSparkForConditionalGeneration,
            MuseSparkTextModel,
        )

        for start, end, is_last, expected_slots in (
            (0, 2, False, {1: [0]}),
            (2, 4, False, {}),
            (4, 6, True, {5: [1]}),
        ):
            stage = _build_stub_text_stage(
                6, start, end, is_first=(start == 0), is_last=is_last
            )
            wrapper = SimpleNamespace(
                text_config=SimpleNamespace(num_hidden_layers=6),
                pp_group=SimpleNamespace(is_last_rank=is_last),
                model=stage,
                capture_aux_hidden_states=False,
            )
            MuseSparkForConditionalGeneration.set_dflash_layers_to_capture(
                wrapper, [1, 5]
            )
            self.assertEqual(stage.layers_to_capture, [1, 5])
            self.assertEqual(stage._capture_slots_by_layer, expected_slots)
            self.assertEqual(wrapper.capture_aux_hidden_states, is_last)
            self.assertIs(
                MuseSparkTextModel._set_layers_to_capture,
                stage._set_layers_to_capture.__func__,
            )

    def test_off_stage_param_filter(self):
        from types import SimpleNamespace

        from sglang_tpu.models.muse_spark import MuseSparkForConditionalGeneration

        for start, end, is_first, is_last in (
            (0, 7, True, False),
            (7, 15, False, False),
            (55, 62, False, True),
        ):
            stub = SimpleNamespace(
                start_layer=start,
                end_layer=end,
                pp_group=SimpleNamespace(is_first_rank=is_first, is_last_rank=is_last),
            )
            fn = MuseSparkForConditionalGeneration._is_off_stage_param
            self.assertEqual(fn(stub, "model.embed_tokens.weight"), not is_first)
            self.assertEqual(fn(stub, "lm_head.weight"), not is_last)
            self.assertEqual(fn(stub, "model.norm.weight"), not is_last)
            self.assertFalse(fn(stub, f"model.layers.{start}.mlp.gate.weight"))
            if start > 0:
                self.assertTrue(fn(stub, "model.layers.0.mlp.gate.weight"))
            if end < 62:
                self.assertTrue(fn(stub, "model.layers.61.mlp.gate.weight"))

    def test_make_pp_layers_builds_muse_spark_text_model_stages(self):
        import os
        from types import SimpleNamespace
        from unittest.mock import patch

        import sglang_tpu.models.muse_spark as muse_mod
        from sglang.srt.layers.utils import PPMissingLayer

        cfg = muse_mod.MuseSparkTextConfig(
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=62,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            moe_hidden_size=16,
            moe_intermediate_size=16,
            num_local_experts=4,
            num_experts_per_tok=2,
            layer_types=["sliding_attention", "full_attention"] * 31,
            layer_rope_theta=[10000.0, 0.0] * 31,
        )

        class _FakeEmbed(torch.nn.Module):
            def __init__(self, num_embeddings, embedding_dim, **kwargs):
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.zeros(num_embeddings, embedding_dim)
                )

        class _FakeLayer(torch.nn.Module):
            def __init__(self, config, layer_idx, rotary_emb, quant_config, prefix):
                super().__init__()
                self.layer_idx = layer_idx
                self.prefix = prefix

        expected_ranges = [
            (0, 7),
            (7, 15),
            (15, 23),
            (23, 31),
            (31, 39),
            (39, 47),
            (47, 55),
            (55, 62),
        ]
        for pp_rank, expected_range in enumerate(expected_ranges):
            exp_first = pp_rank == 0
            exp_last = pp_rank == 7
            fake_parallel = SimpleNamespace(
                pp_rank=pp_rank,
                pp_size=8,
                pp_group=SimpleNamespace(
                    world_size=8,
                    rank_in_group=pp_rank,
                    is_first_rank=exp_first,
                    is_last_rank=exp_last,
                ),
            )
            with (
                patch.dict(
                    os.environ,
                    {"SGLANG_PP_LAYER_PARTITION": "7,8,8,8,8,8,8,7"},
                    clear=False,
                ),
                patch.object(muse_mod, "get_parallel", return_value=fake_parallel),
                patch(
                    "sglang.srt.utils.common.get_parallel", return_value=fake_parallel
                ),
                patch.object(muse_mod, "VocabParallelEmbedding", _FakeEmbed),
                patch.object(muse_mod, "MuseSparkDecoderLayer", _FakeLayer),
            ):
                model = muse_mod.MuseSparkTextModel(
                    cfg, max_position=128, quant_config=None, prefix="model"
                )
            self.assertEqual((model.start_layer, model.end_layer), expected_range)
            self.assertEqual(
                not isinstance(model.embed_tokens, PPMissingLayer), exp_first
            )
            self.assertEqual(not isinstance(model.norm, PPMissingLayer), exp_last)
            for idx in range(62):
                layer = model.layers[idx]
                if expected_range[0] <= idx < expected_range[1]:
                    self.assertIsInstance(layer, _FakeLayer)
                    self.assertEqual(layer.layer_idx, idx)
                    self.assertEqual(layer.prefix, f"model.layers.{idx}")
                else:
                    self.assertIsInstance(layer, PPMissingLayer)

    def test_omni_wrapper_gates_vision_to_first_stage_and_forwards_proxy(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        import sglang_tpu.models.muse_spark as text_mod
        import sglang_tpu.models.muse_spark_omni as omni_mod
        from sglang.srt.layers.utils import PPMissingLayer
        from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

        def _fake_text_init(self, config, quant_config=None, prefix=""):
            torch.nn.Module.__init__(self)
            self.text_config = SimpleNamespace(rms_norm_eps=1e-5, hidden_size=16)
            self.pp_group = text_mod.get_parallel().pp_group
            self.logits_processor = SimpleNamespace()
            self.model = _build_stub_text_stage(
                4,
                0 if self.pp_group.is_first_rank else 2,
                2 if self.pp_group.is_first_rank else 4,
                is_first=self.pp_group.is_first_rank,
                is_last=self.pp_group.is_last_rank,
            )
            self.lm_head = (
                torch.nn.Linear(16, 32, bias=False)
                if self.pp_group.is_last_rank
                else PPMissingLayer()
            )
            self.capture_aux_hidden_states = False
            self._megakernel = None

        fake_non_first = SimpleNamespace(
            pp_group=SimpleNamespace(
                world_size=2, rank_in_group=1, is_first_rank=False, is_last_rank=False
            )
        )
        with (
            patch.object(
                text_mod.MuseSparkForConditionalGeneration, "__init__", _fake_text_init
            ),
            patch.object(text_mod, "get_parallel", return_value=fake_non_first),
            patch.object(
                omni_mod,
                "get_mm",
                return_value=SimpleNamespace(enable_multimodal=True),
            ),
        ):
            stage1 = omni_mod.MuseSparkForConditionalGeneration(SimpleNamespace())

        self.assertFalse(stage1.vision_enabled)
        self.assertIsInstance(stage1.get_input_embeddings(), PPMissingLayer)
        in_proxy = PPProxyTensors(
            {
                "hidden_states": torch.randn(3, 16, dtype=torch.float32),
                "residual": torch.zeros(3, 16, dtype=torch.float32),
            }
        )
        out_proxy = stage1(
            torch.tensor([1, 2, 3]),
            torch.arange(3),
            None,
            pp_proxy_tensors=in_proxy,
        )
        self.assertIsInstance(out_proxy, PPProxyTensors)
        self.assertEqual(out_proxy.tensors["hidden_states"].dtype, torch.float32)

        decoder = omni_mod._Decoder(stage1)
        self.assertIs(decoder.pp_group, stage1.pp_group)
        dec_proxy = decoder(
            torch.tensor([1, 2, 3]),
            torch.arange(3),
            None,
            pp_proxy_tensors=in_proxy,
        )
        self.assertIsInstance(dec_proxy, PPProxyTensors)

    def test_server_pp8_recipe_validates_under_launch_server(self):
        import pathlib
        import sys

        repo_root = str(pathlib.Path(__file__).resolve().parents[5])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from benchmark.tpu.launch_server import load_recipe, plan_launch

        plan = plan_launch(
            load_recipe("recipe/configs/muse_spark/server_pp8.py"),
            inherited={},
        )
        self.assertEqual(plan.config.server_args["tp_size"], 1)
        self.assertEqual(plan.config.server_args["pp_size"], 8)
        self.assertEqual(plan.config.server_args["ep_size"], 1)
        self.assertEqual(plan.config.server_args["quantization"], "modelopt_fp4")
        self.assertEqual(plan.config.server_args["moe_runner_backend"], "auto")
        self.assertTrue(plan.config.server_args["disable_overlap_schedule"])
        self.assertEqual(plan.env["SGLANG_PP_LAYER_PARTITION"], "7,8,8,8,8,8,8,7")


if __name__ == "__main__":
    unittest.main()
