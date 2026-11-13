"""CPU checkpoint-ABI tests; no TPU device, kernel execution, or model download.

Guard the explicit opt-in (ordinary/Flash FP8 must remain upstream), channel
rather than reciprocal/block scale semantics, and checkpoint parameter names.
The existing TPU MoE formatter is exercised on real CPU tensors as well.
"""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.parameter import ChannelQuantScaleParameter, ModelWeightParameter
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.srt.runtime_context import get_context
from sglang_tpu.glm52_channelwise_fp8 import (
    Glm52ChannelwiseFp8Config,
    Glm52ChannelwiseFp8LinearMethod,
    Glm52ChannelwiseFp8MoEMethod,
    TpuFp8Config,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _metadata(**overrides):
    return {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "weight_block_size": None,
        "tpu_checkpoint_layout": "glm52_channelwise_fp8",
        "hf_config": SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"]),
        **overrides,
    }


def _config():
    # Allocation tests isolate checkpoint packing from the factory's runtime gate.
    return Glm52ChannelwiseFp8Config.from_fp8_config(_metadata())


def _dense():
    layer = torch.nn.Module()
    method = Glm52ChannelwiseFp8LinearMethod(_config())
    method.create_weights(
        layer,
        input_size_per_partition=3,
        output_partition_sizes=[2, 3],
        input_size=3,
        output_size=5,
        params_dtype=torch.bfloat16,
    )
    return layer, method


class _LoaderModel(DeepseekV2WeightLoaderMixin, torch.nn.Module):
    """Minimal model tree; all checkpoint routing/post-load methods stay real."""

    def __init__(self, attention, mlp):
        super().__init__()
        self.config = SimpleNamespace(
            n_routed_experts=0,
            q_lora_rank=2,
            num_hidden_layers=1,
            num_nextn_predict_layers=1,
        )
        self.quant_config = _config()
        self.num_fused_shared_experts = 0
        self.pp_group = SimpleNamespace(is_last_rank=True)
        self.model = torch.nn.Module()
        self.model.start_layer = 0
        self.model.end_layer = 1
        layer = torch.nn.Module()
        layer.self_attn = attention
        layer.mlp = mlp
        self.model.layers = torch.nn.ModuleList([layer])


def _linear_for_loader(loader, widths, input_width=3):
    layer = torch.nn.Module()
    method = Glm52ChannelwiseFp8LinearMethod(_config())
    method.create_weights(
        layer,
        input_size_per_partition=input_width,
        output_partition_sizes=widths,
        input_size=input_width,
        output_size=sum(widths),
        params_dtype=torch.bfloat16,
        weight_loader=loader,
    )
    return layer, method


class TestGlm52Fp8OptIn(CustomTestCase):
    def test_ordinary_scalar_and_block_fp8_do_not_consult_runtime(self):
        # A broad "weight_block_size is None" predicate silently captures Flash.
        # Fail at the runtime boundary even if another test published a context.
        with (
            patch(
                "sglang.srt.runtime_context.get_model",
                side_effect=AssertionError("ordinary FP8 consulted model runtime"),
            ),
            patch(
                "sglang.srt.runtime_context.get_exec",
                side_effect=AssertionError("ordinary FP8 consulted execution runtime"),
            ),
        ):
            for block_size in (None, [128, 128]):
                for layout in ({}, {"tpu_checkpoint_layout": None}):
                    with self.subTest(block_size=block_size, layout=layout):
                        metadata = {
                            "quant_method": "fp8",
                            "activation_scheme": "dynamic",
                            "weight_block_size": block_size,
                            "hf_config": SimpleNamespace(
                                architectures=["Glm4MoeForCausalLM"]
                            ),
                            **layout,
                        }
                        config = TpuFp8Config.from_config(metadata)
                        self.assertIs(type(config), Fp8Config)
                        self.assertEqual(config.weight_block_size, block_size)
                        self.assertTrue(config.is_checkpoint_fp8_serialized)

    def test_explicit_layout_rejects_incompatible_checkpoint_metadata(self):
        cases = (
            ({"tpu_checkpoint_layout": "channelwise"}, "Unknown TPU FP8"),
            ({"hf_config": None}, "requires GlmMoeDsaForCausalLM"),
            (
                {"hf_config": SimpleNamespace(architectures=["Glm4MoeForCausalLM"])},
                "requires GlmMoeDsaForCausalLM",
            ),
            ({"quant_method": "mxfp8"}, "requires serialized fp8"),
            ({"activation_scheme": "static"}, "requires serialized fp8"),
            ({"fmt": "e5m2"}, "requires serialized fp8"),
            ({"weight_block_size": [128, 128]}, "requires serialized fp8"),
            ({"scale_fmt": "ue8m0"}, "requires serialized fp8"),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    TpuFp8Config.from_config(_metadata(**overrides))

    def test_opt_in_preserves_upstream_parsing_and_enforces_loader_contract(self):
        metadata = _metadata(
            ignored_layers=["model.layers.0.mlp"],
            packed_modules_mapping={"gate_up_proj": ["gate_proj", "up_proj"]},
        )
        original = copy.deepcopy(metadata)
        with get_context().override_server_args(load_format="safetensors"):
            config = TpuFp8Config.from_config(metadata)
        self.assertIs(type(config), Glm52ChannelwiseFp8Config)
        self.assertTrue(config.is_checkpoint_fp8_serialized)
        self.assertEqual(config.activation_scheme, "dynamic")
        self.assertIsNone(config.weight_block_size)
        self.assertEqual(config.ignored_layers, ["layers.0.mlp", "model.layers.0.mlp"])
        self.assertEqual(
            config.packed_modules_mapping, metadata["packed_modules_mapping"]
        )
        self.assertEqual(metadata, original)
        for runtime, message in (
            ({"load_format": "dummy"}, "default checkpoint loader"),
            ({"load_format": "auto", "cpu_offload_gb": 1}, "CPU offloading"),
        ):
            with self.subTest(runtime=runtime):
                with get_context().override_server_args(**runtime):
                    with self.assertRaisesRegex(ValueError, message):
                        TpuFp8Config.from_config(metadata)


class TestGlm52Fp8Packing(CustomTestCase):
    def test_dense_channel_scales_have_one_checkpoint_name_and_reconstruct(self):
        layer, method = _dense()
        self.assertIsInstance(layer.weight, ModelWeightParameter)
        self.assertIsInstance(layer.weight_scale_inv, ChannelQuantScaleParameter)
        self.assertEqual((layer.weight.input_dim, layer.weight.output_dim), (1, 0))
        self.assertEqual(layer.weight_scale_inv.output_dim, 0)
        self.assertFalse(hasattr(layer.weight_scale_inv, "input_dim"))
        self.assertIs(layer.weight_scale, layer.weight_scale_inv)
        self.assertEqual(
            set(dict(layer.named_parameters())), {"weight", "weight_scale_inv"}
        )
        self.assertEqual(set(layer.state_dict()), {"weight", "weight_scale_inv"})
        self.assertEqual(layer.logical_widths, [2, 3])

        checkpoint = torch.arange(-7, 8, dtype=torch.float32).reshape(5, 3)
        # Actual checkpoint scales are rank-one [N], not [N, 1].
        scales = torch.tensor([0.5, 2.0, 4.0, 0.25, 8.0])
        self.assertEqual(layer.weight_scale_inv.shape, (5,))
        layer.weight.data.copy_(checkpoint.to(torch.float8_e4m3fn))
        # Exercise the real legacy loader used by this new quant method.
        topology = SimpleNamespace(tp_rank=1, tp_size=2, use_presharded_weights=False)
        RowParallelLinear.weight_loader(topology, layer.weight_scale_inv, scales)
        x = torch.tensor([[1.0, -2.0, 3.0], [-1.0, 0.5, 2.0]])
        expected = torch.nn.functional.linear(x, checkpoint * scales[:, None])
        method.process_weights_after_loading(layer)

        self.assertEqual(layer.weight.shape, (3, 5))
        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
        self.assertTrue(layer.weight.is_contiguous())
        self.assertFalse(layer.weight.requires_grad)
        self.assertIs(layer.weight_scale, layer.weight_scale_inv)
        self.assertEqual(set(layer.state_dict()), {"weight", "weight_scale_inv"})
        torch.testing.assert_close(
            x @ (layer.weight.float() * layer.weight_scale_inv.flatten()), expected
        )

    def test_legacy_dense_loaders_shard_weights_but_replicate_row_scales(self):
        """Rank-one checkpoint scales shard only with output channels at TP=2."""
        checkpoint = torch.arange(60).reshape(10, 6).float()
        scales = torch.arange(1, 11).float()
        for rank in (0, 1):
            for column_parallel in (True, False):
                with self.subTest(rank=rank, column_parallel=column_parallel):
                    layer = torch.nn.Module()
                    method = Glm52ChannelwiseFp8LinearMethod(_config())
                    topology = SimpleNamespace(
                        tp_rank=rank, tp_size=2, use_presharded_weights=False
                    )
                    loader_cls = (
                        ColumnParallelLinear if column_parallel else RowParallelLinear
                    )
                    loader = loader_cls.weight_loader.__get__(topology)
                    method.create_weights(
                        layer,
                        input_size_per_partition=6 if column_parallel else 3,
                        output_partition_sizes=[5 if column_parallel else 10],
                        input_size=6,
                        output_size=10,
                        params_dtype=torch.bfloat16,
                        weight_loader=loader,
                    )
                    layer.weight.weight_loader(
                        layer.weight, checkpoint.to(torch.float8_e4m3fn)
                    )
                    layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, scales)
                    expected_weight = (
                        checkpoint[rank * 5 : (rank + 1) * 5]
                        if column_parallel
                        else checkpoint[:, rank * 3 : (rank + 1) * 3]
                    ).to(torch.float8_e4m3fn).float()
                    expected_scale = (
                        scales[rank * 5 : (rank + 1) * 5]
                        if column_parallel
                        else scales
                    )
                    torch.testing.assert_close(layer.weight.float(), expected_weight)
                    torch.testing.assert_close(layer.weight_scale_inv, expected_scale)
                    method.process_weights_after_loading(layer)
                    torch.testing.assert_close(
                        layer.weight.float() * layer.weight_scale_inv,
                        (expected_weight * expected_scale[:, None]).t(),
                    )

    def test_model_loader_pairs_qkv_scales_and_shards_merged_mlp(self):
        """Real model routing must concatenate rank-one scales, never collapse them."""
        replicated_loader = ReplicatedLinear.weight_loader.__get__(SimpleNamespace())
        q = torch.tensor([[1, 2, 3], [4, 5, 6]]).float()
        kv = torch.tensor([[-1, -2, -3], [2, 4, 6], [1, 3, 5]]).float()
        qs = torch.tensor([0.25, 4.0])
        kvs = torch.tensor([2.0, 0.5, 8.0])
        gate = torch.arange(12).reshape(4, 3).float()
        up = -gate - 1
        gs = torch.tensor([1.0, 2.0, 4.0, 8.0])
        us = torch.tensor([0.5, 0.25, 0.125, 0.0625])
        for rank in (0, 1):
            with self.subTest(rank=rank):
                attention = torch.nn.Module()
                fused, fused_method = _linear_for_loader(replicated_loader, [2, 3])
                attention.fused_qkv_a_proj_with_mqa = fused
                topology = SimpleNamespace(
                    tp_rank=rank,
                    tp_size=2,
                    use_presharded_weights=False,
                    output_sizes=[4, 4],
                )
                merged_loader = MergedColumnParallelLinear.weight_loader.__get__(
                    topology
                )
                merged, merged_method = _linear_for_loader(merged_loader, [2, 2])
                mlp = torch.nn.Module()
                mlp.gate_up_proj = merged
                model = _LoaderModel(attention, mlp)
                prefix = "model.layers.0."
                # Interleave pairs and deliver kv before q: caches for weights
                # and scales must remain independent and concatenate q then kv.
                weights = [
                    (prefix + "self_attn.kv_a_proj_with_mqa.weight_scale_inv", kvs),
                    (prefix + "self_attn.q_a_proj.weight", q.to(torch.float8_e4m3fn)),
                    (prefix + "mlp.up_proj.weight_scale_inv", us),
                    (prefix + "self_attn.q_a_proj.weight_scale_inv", qs),
                    (prefix + "mlp.gate_proj.weight", gate.to(torch.float8_e4m3fn)),
                    (
                        prefix + "self_attn.kv_a_proj_with_mqa.weight",
                        kv.to(torch.float8_e4m3fn),
                    ),
                    (prefix + "mlp.gate_proj.weight_scale_inv", gs),
                    (prefix + "mlp.up_proj.weight", up.to(torch.float8_e4m3fn)),
                    # An MTP/draft tensor has no target parameter and must not
                    # enter either fusion cache or overwrite layer zero.
                    (
                        "model.layers.1.self_attn.q_a_proj.weight_scale_inv",
                        torch.ones(99),
                    ),
                ]
                with get_context().override_server_args(tp_size=2):
                    model.do_load_weights(weights)
                torch.testing.assert_close(fused.weight.float(), torch.cat([q, kv]))
                torch.testing.assert_close(fused.weight_scale_inv, torch.cat([qs, kvs]))
                sl = slice(rank * 2, rank * 2 + 2)
                expected_weight = torch.cat([gate[sl], up[sl]])
                expected_scale = torch.cat([gs[sl], us[sl]])
                torch.testing.assert_close(merged.weight.float(), expected_weight)
                torch.testing.assert_close(merged.weight_scale_inv, expected_scale)
                fused_method.process_weights_after_loading(fused)
                merged_method.process_weights_after_loading(merged)
                torch.testing.assert_close(
                    fused.weight.float() * fused.weight_scale_inv,
                    (torch.cat([q, kv]) * torch.cat([qs, kvs])[:, None]).t(),
                )
                torch.testing.assert_close(
                    merged.weight.float() * merged.weight_scale_inv,
                    (expected_weight * expected_scale[:, None]).t(),
                )

    def test_early_kv_b_absorption_uses_channel_multipliers(self):
        """Before linear transpose, the model reader must rescale rows, not K."""
        replicated_loader = ReplicatedLinear.weight_loader.__get__(SimpleNamespace())
        projection, method = _linear_for_loader(replicated_loader, [6])
        values = torch.tensor(
            [[1, -2, 3], [2, 1, -1], [-3, 2, 1], [1, 3, -2], [2, -1, 1], [-1, 2, 3]]
        ).float()
        scales = torch.tensor([0.25, 2.0, 0.5, 4.0, 1.0, 0.125])
        attention = torch.nn.Module()
        attention.kv_b_proj = projection
        # The TPU reader also binds this projection into the MHA consumer before
        # absorption. Keep the real binding path when torch_tpu is importable.
        attention.attn_mha = torch.nn.Module()
        attention.attn_mha.kv_b_proj = None
        attention.qk_nope_head_dim = 2
        attention.v_head_dim = 1
        attention.w_kc = None
        attention.w_vc = None
        attention.w_scale = None
        model = _LoaderModel(attention, torch.nn.Module())
        prefix = "model.layers.0.self_attn.kv_b_proj."
        with get_context().override_server_args():
            model.do_load_weights(
                [(prefix + "weight", values.to(torch.float8_e4m3fn)),
                 (prefix + "weight_scale_inv", scales)]
            )
        expected = (values * scales[:, None]).reshape(2, 3, 3)
        # Absorption requantizes to tensor FP8, so allow its quantization error;
        # channel-order/reciprocal/axis mistakes are far larger than this bound.
        torch.testing.assert_close(
            attention.w_kc.float() * attention.w_scale,
            expected[:, :2, :], rtol=0.06, atol=0.02,
        )
        torch.testing.assert_close(
            attention.w_vc.float() * attention.w_scale,
            expected[:, 2:, :].transpose(1, 2), rtol=0.06, atol=0.02,
        )
        self.assertIs(projection.weight_scale, projection.weight_scale_inv)
        self.assertEqual(projection.weight.shape, (6, 3))
        method.process_weights_after_loading(projection)
        self.assertEqual(projection.weight.shape, (3, 6))

    def test_dense_rejects_stale_alias_and_nonchannel_scale_before_transpose(self):
        for stale_alias in (True, False):
            with self.subTest(stale_alias=stale_alias):
                layer, method = _dense()
                if stale_alias:
                    layer.weight_scale_inv = torch.nn.Parameter(
                        layer.weight_scale_inv.detach().clone(), requires_grad=False
                    )
                    message = "early kv_b scale alias"
                else:
                    layer.weight_scale_inv.data = torch.ones(5, 1)
                    message = "dense scales must have shape"
                with self.assertRaisesRegex(ValueError, message):
                    method.process_weights_after_loading(layer)
                self.assertEqual(layer.weight.shape, (5, 3))

    def test_moe_channel_metadata_survives_real_runtime_formatting(self):
        # Real upstream allocation plus real TPU formatter, all on CPU. No MoE
        # runner/kernel is constructed; is_gated is the allocation input only.
        with get_context().override_server_args(tp_size=1, pp_size=1):
            method = Glm52ChannelwiseFp8MoEMethod(_config())
            # Bypass only the distributed runner constructor, not loader logic.
            layer = FusedMoE.__new__(FusedMoE)
            torch.nn.Module.__init__(layer)
            layer.moe_runner_config = SimpleNamespace(is_gated=True)
            layer.quant_method = method
            layer.moe_tp_rank = 0
            layer.moe_tp_size = 1
            layer.use_padded_loading = False
            layer.use_presharded_weights = False
            layer.use_triton_kernels = False
            loader = object()
            method.create_weights(
                layer, 2, 4, 3, torch.bfloat16, weight_loader=loader
            )
            expected = {}
            for stem, shape in (("w13", (2, 6, 4)), ("w2", (2, 4, 3))):
                weight = getattr(layer, f"{stem}_weight")
                scale = getattr(layer, f"{stem}_weight_scale_inv")
                self.assertEqual(weight.shape, shape)
                self.assertEqual(weight.dtype, torch.float8_e4m3fn)
                self.assertEqual(scale.shape, shape[:2])
                self.assertEqual(scale.quant_method, "channel")
                self.assertIs(scale.weight_loader, loader)
                self.assertFalse(hasattr(layer, f"{stem}_weight_scale"))
                values = (torch.arange(weight.numel()).reshape(shape) % 13 - 6).float()
                # Powers of two stay exact in both bf16/fp32 formatter policies.
                scales = 2.0 ** torch.arange(shape[0] * shape[1]).reshape(shape[:2])
                weight.data.copy_(values.to(weight.dtype))
                # Feed actual checkpoint [N] tensors through the existing CHANNEL
                # loader: separate gate/up shards concatenate into w13; w2
                # replicates output-channel scales rather than slicing K.
                for expert in range(2):
                    if stem == "w13":
                        for shard_id, shard_scale in zip(
                            ("w1", "w3"), scales[expert].chunk(2)
                        ):
                            layer._load_per_channel_weight_scale(
                                scale.data[expert], 0, shard_id, shard_scale
                            )
                    else:
                        layer._load_per_channel_weight_scale(
                            scale.data[expert], 1, "w2", scales[expert]
                        )
                torch.testing.assert_close(scale, scales)
                expected[stem] = (
                    (values * scales[:, :, None]).transpose(1, 2).contiguous()
                )

            method.process_weights_after_loading(layer)

        for stem, reference in expected.items():
            weight = getattr(layer, f"{stem}_weight")
            scale = getattr(layer, f"{stem}_weight_scale")
            self.assertEqual(scale.shape, (2, 1, 1, reference.shape[-1]))
            self.assertIs(scale.weight_loader, loader)
            self.assertEqual(scale.quant_method, "channel")
            self.assertTrue(weight.is_contiguous())
            torch.testing.assert_close(
                weight.float() * scale.float().reshape(2, 1, -1), reference
            )
        self.assertIsNone(layer.weight_block_size)


if __name__ == "__main__":
    unittest.main()
