# SPDX-License-Identifier: Apache-2.0
"""CPU checkpoint-ABI tests; no TPU device, kernel execution, or model download.

Guard the explicit opt-in (ordinary/Flash FP8 must remain upstream), channel
rather than reciprocal/block scale semantics, and checkpoint parameter names.
The existing TPU MoE formatter is exercised on real CPU tensors as well.
"""

import copy
import importlib
import os
import tempfile
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import PretrainedConfig

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
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.parameter import ChannelQuantScaleParameter, ModelWeightParameter
from sglang.srt.layers.quantization import fp8
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.srt.plugins.hook_registry import _wrap_fn
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang_tpu.fp8 import (
    TpuChannelwiseFp8Config,
    TpuChannelwiseFp8LinearMethod,
    TpuFp8Config,
)
from sglang_tpu.hooks import HOOKS

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _glm_nextn_resolver():
    from sglang.srt.models.glm4_moe import GlmMoeDsaForCausalLMNextN

    resolver = GlmMoeDsaForCausalLMNextN._resolve_nextn_quant_config
    for target, hook, hook_type in HOOKS:
        if target == (
            "sglang.srt.models.glm4_moe."
            "GlmMoeDsaForCausalLMNextN._resolve_nextn_quant_config"
        ):
            resolver = _wrap_fn(resolver, hook, hook_type)
    return resolver


def _metadata(**overrides):
    return {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "weight_block_size": None,
        "weight_scale_granularity": "channel",
        "hf_config": SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"]),
        **overrides,
    }


def _config(**overrides):
    with get_context().override_server_args(load_format="safetensors"):
        return TpuFp8Config.from_config(_metadata(**overrides))


def _block_config(**overrides):
    return TpuFp8Config.from_config(
        _metadata(weight_scale_granularity=None, weight_block_size=[2, 2], **overrides)
    )


def _moe(config, *, intermediate=4, weight_loader=None):
    # Keep real allocation and expert shard loaders without distributed runners.
    layer = FusedMoE.__new__(FusedMoE)
    torch.nn.Module.__init__(layer)
    layer.moe_runner_config = SimpleNamespace(is_gated=True)
    layer.quant_config = config
    layer.quant_method = config.get_quant_method(layer, "model.decoder.mlp.experts")
    layer.num_local_experts = 2
    layer.num_experts = 2
    layer._num_local_routed = 2
    layer._num_global_routed = 2
    layer._expert_storage_rank = 0
    layer._has_fused_shared = False
    layer.num_fused_shared_experts = 0
    layer.moe_tp_rank = 0
    layer.moe_tp_size = 1
    layer.use_padded_loading = False
    layer.use_presharded_weights = False
    layer.use_triton_kernels = False
    layer.use_flashinfer_trtllm_moe = False
    layer.scheme = None
    layer.quant_method.create_weights(
        layer,
        2,
        4,
        intermediate,
        torch.bfloat16,
        weight_loader=layer.weight_loader if weight_loader is None else weight_loader,
    )
    return layer


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
    method = TpuChannelwiseFp8LinearMethod(_config())
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


class TestTpuFp8Config(CustomTestCase):
    def test_model_loader_covers_legacy_sources_and_offline_opt_in(self):
        from sglang.srt.model_loader.weight_utils import get_quant_config

        cases = (
            (None, TpuFp8Config, False),
            (
                _metadata(weight_scale_granularity=None, weight_block_size=[128, 128]),
                Fp8Config,
                True,
            ),
            (_metadata(), TpuChannelwiseFp8Config, True),
        )
        with tempfile.TemporaryDirectory() as directory:
            for metadata, expected_type, serialized in cases:
                with self.subTest(config=expected_type.__name__):
                    hf_config = PretrainedConfig(architectures=["GlmMoeDsaForCausalLM"])
                    if metadata is not None:
                        hf_config.quantization_config = {
                            k: v for k, v in metadata.items() if k != "hf_config"
                        }
                    model_config = SimpleNamespace(
                        quantization="fp8",
                        hf_config=hf_config,
                        model_path=directory,
                        is_draft_model=False,
                        is_draft_quantization_explicit=False,
                    )
                    with (
                        get_context().override_server_args(load_format="safetensors"),
                        patch(
                            "sglang.srt.model_loader.weight_utils.get_quantization_config",
                            return_value=TpuFp8Config,
                        ),
                    ):
                        result = get_quant_config(model_config, SimpleNamespace(), {})
                    self.assertIs(type(result), expected_type)
                    self.assertEqual(result.is_checkpoint_fp8_serialized, serialized)

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
                for scale_metadata in ({}, {"weight_scale_granularity": None}):
                    with self.subTest(
                        block_size=block_size, scale_metadata=scale_metadata
                    ):
                        metadata = {
                            "quant_method": "fp8",
                            "activation_scheme": "dynamic",
                            "weight_block_size": block_size,
                            "hf_config": SimpleNamespace(
                                architectures=["Glm4MoeForCausalLM"]
                            ),
                            **scale_metadata,
                        }
                        config = TpuFp8Config.from_config(metadata)
                        self.assertIs(type(config), Fp8Config)
                        self.assertEqual(config.weight_block_size, block_size)
                        self.assertTrue(config.is_checkpoint_fp8_serialized)
                        # An upstream-ignored source key alone must not opt
                        # into the offline ABI or change legacy parsing.
                        metadata["is_checkpoint_fp8_serialized"] = False
                        config = TpuFp8Config.from_config(metadata)
                        self.assertIs(type(config), Fp8Config)
                        self.assertTrue(config.is_checkpoint_fp8_serialized)

    def test_channel_scales_reject_incompatible_quantization_metadata(self):
        cases = (
            (
                {"weight_scale_granularity": "unknown"},
                "Unsupported FP8 weight scale granularity",
            ),
            (
                {"weight_scale_granularity": True},
                "Unsupported FP8 weight scale granularity",
            ),
            ({"is_checkpoint_fp8_serialized": False}, "require serialized FP8"),
            ({"quant_method": "mxfp8"}, "require serialized fp8"),
            ({"activation_scheme": "static"}, "require serialized fp8"),
            ({"fmt": "e5m2"}, "require serialized fp8"),
            ({"weight_block_size": [128, 128]}, "require serialized fp8"),
            ({"scale_fmt": "ue8m0"}, "require serialized fp8"),
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
        self.assertIs(type(config), TpuChannelwiseFp8Config)
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
            ({"load_format": "npcache"}, "npcache cannot serialize FP8"),
            ({"load_format": "auto", "cpu_offload_gb": 1}, "CPU offloading"),
        ):
            with self.subTest(runtime=runtime):
                with get_context().override_server_args(**runtime):
                    with self.assertRaisesRegex(ValueError, message):
                        TpuFp8Config.from_config(metadata)


class _TpuMoETestCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.contexts = ExitStack()
        self.addCleanup(self.contexts.close)
        self.contexts.enter_context(
            get_flags().moe.override(runner_backend=MoeRunnerBackend.PALLAS_TPU)
        )


class TestChannelwiseFp8Loading(_TpuMoETestCase):
    def test_dense_channel_scales_have_one_checkpoint_name_and_reconstruct(self):
        layer, method = _linear_for_loader(None, [2, 3])
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
                    method = TpuChannelwiseFp8LinearMethod(_config())
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
                        (
                            checkpoint[rank * 5 : (rank + 1) * 5]
                            if column_parallel
                            else checkpoint[:, rank * 3 : (rank + 1) * 3]
                        )
                        .to(torch.float8_e4m3fn)
                        .float()
                    )
                    expected_scale = (
                        scales[rank * 5 : (rank + 1) * 5] if column_parallel else scales
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
                [
                    (prefix + "weight", values.to(torch.float8_e4m3fn)),
                    (prefix + "weight_scale_inv", scales),
                ]
            )
        expected = (values * scales[:, None]).reshape(2, 3, 3)
        # Absorption requantizes to tensor FP8, so allow its quantization error;
        # channel-order/reciprocal/axis mistakes are far larger than this bound.
        torch.testing.assert_close(
            attention.w_kc.float() * attention.w_scale,
            expected[:, :2, :],
            rtol=0.06,
            atol=0.02,
        )
        torch.testing.assert_close(
            attention.w_vc.float() * attention.w_scale,
            expected[:, 2:, :].transpose(1, 2),
            rtol=0.06,
            atol=0.02,
        )
        self.assertIs(projection.weight_scale, projection.weight_scale_inv)
        self.assertEqual(projection.weight.shape, (6, 3))
        method.process_weights_after_loading(projection)
        self.assertEqual(projection.weight.shape, (3, 6))

    def test_dense_rejects_stale_alias_and_nonchannel_scale_before_transpose(self):
        for stale_alias in (True, False):
            with self.subTest(stale_alias=stale_alias):
                layer, method = _linear_for_loader(None, [2, 3])
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
        with get_context().override_server_args(
            tp_size=1, pp_size=1, moe_runner_backend="pallas_tpu"
        ):
            loader = object()
            layer = _moe(_config(), intermediate=3, weight_loader=loader)
            method = layer.quant_method
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


def _expert_fixture():
    # Build channelwise fixtures from the *same block-FP8 values*, not BF16
    # originals. Independent expansion defines the expected multiplicative ABI.
    result = {}
    for stem, shape in (("w13", (2, 8, 4)), ("w2", (2, 4, 4))):
        block_q = (
            torch.arange(torch.tensor(shape).prod()).reshape(shape) % 13 - 6
        ).float()
        block_q[:, -2:].zero_()
        block_scales = (
            2.0 ** (torch.arange(shape[0] * shape[1] * shape[2] // 4) % 7 - 8)
        ).reshape(shape[0], shape[1] // 2, shape[2] // 2)
        source = block_q * block_scales.repeat_interleave(2, 1).repeat_interleave(2, 2)
        channel_scales = (source.abs().amax(-1) / 448).clamp_min(1e-12)
        channel_q = (
            (source / channel_scales[:, :, None])
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
        )
        result[stem] = (
            block_q.to(torch.float8_e4m3fn),
            block_scales,
            channel_q,
            channel_scales,
        )
    return result


def _load_expert_fixture(layer, fixture, blockwise):
    # Both representations use the real gate/up/down expert loaders, including
    # packed scale metadata. Only checkpoint values and scale ranks differ.
    for stem, values in fixture.items():
        weight, scales = values[:2] if blockwise else values[2:]
        for expert in range(weight.shape[0]):
            projections = (
                (("w1", slice(0, 4)), ("w3", slice(4, 8)))
                if stem == "w13"
                else (("w2", slice(None)),)
            )
            for shard, rows in projections:
                scale_rows = (
                    slice(rows.start // 2, rows.stop // 2)
                    if blockwise and stem == "w13"
                    else rows
                )
                for suffix, value in (
                    ("weight", weight[expert, rows]),
                    ("weight_scale_inv", scales[expert, scale_rows]),
                ):
                    param = getattr(layer, f"{stem}_{suffix}")
                    param.weight_loader(param, value, suffix, shard, expert)


class TestChannelwiseMoeAndNextN(_TpuMoETestCase):
    def assert_moe_fixture(self, layer, fixture, *, check_dtype=True):
        for stem, (_, _, q, scales) in fixture.items():
            self.assertTrue(
                torch.equal(
                    getattr(layer, f"{stem}_weight").view(torch.uint8),
                    q.transpose(1, 2).contiguous().view(torch.uint8),
                )
            )
            torch.testing.assert_close(
                getattr(layer, f"{stem}_weight_scale").reshape(2, -1),
                scales.to(torch.bfloat16),
                rtol=0,
                atol=0,
                check_dtype=check_dtype,
            )

    def test_channel_scale_loading_is_independent_of_model_architecture(self):
        fixture = _expert_fixture()
        for hf_config in (
            None,
            SimpleNamespace(architectures=["DeepseekV3ForCausalLM"]),
        ):
            with (
                self.subTest(hf_config=hf_config),
                get_context().override_server_args(moe_runner_backend="pallas_tpu"),
            ):
                config = _config(hf_config=hf_config)
                self.assertEqual(config.weight_scale_granularity, "channel")
                layer = _moe(config)
                _load_expert_fixture(layer, fixture, blockwise=False)
                layer.quant_method.process_weights_after_loading(layer)
                self.assert_moe_fixture(layer, fixture, check_dtype=False)

    def test_block_conversion_and_offline_share_runtime(self):
        fixture = _expert_fixture()
        with (
            get_context().override_server_args(
                moe_runner_backend="pallas_tpu", tpu_fp8_runtime="channelwise"
            ),
            patch.object(fp8, "_is_tpu", True),
            patch.object(fp8, "_is_cpu", False),
            patch(
                "sglang_tpu.fp8_moe.current_platform.get_available_memory",
                return_value=(4 * 1024**3, 8 * 1024**3),
            ),
            patch("torch.accelerator.synchronize"),
        ):
            converted, offline = _moe(_block_config()), _moe(_config())
            _load_expert_fixture(converted, fixture, blockwise=True)
            _load_expert_fixture(offline, fixture, blockwise=False)
            for layer in (converted, offline):
                layer.quant_method.process_weights_after_loading(layer)
            for layer in (converted, offline):
                self.assert_moe_fixture(layer, fixture)
            self.assertIsNone(converted.weight_block_size)
            self.assertFalse(converted.quant_method.handles_moe_reduction)
            self.assertFalse(offline.quant_method.handles_moe_reduction)

    def test_block_checkpoint_keeps_fused_rs_selection_and_reduction_ownership(self):
        """Offline loading must preserve the old block-FP8 auto/explicit RS path.

        Real loading, formatting, eligibility and forward dispatch run on CPU;
        only device compilation/collectives are replaced at their boundaries.
        """
        fixture = _expert_fixture()
        parallel = SimpleNamespace(
            pp_size=1,
            tp_size=2,
            tp_rank=0,
            tp_group=SimpleNamespace(world_size=2),
            moe_tp_size=1,
            moe_tp_rank=0,
            moe_ep_size=2,
            moe_ep_rank=0,
            attn_tp_size=2,
            attn_dp_size=1,
            attn_dp_enabled=False,
        )

        def make_layer(offline=False):
            layer = _moe(_config() if offline else _block_config())
            if not offline:
                self.assertIs(type(layer.quant_config), Fp8Config)
            layer.num_experts = 4
            layer.moe_ep_size = 2
            layer.moe_tp_size = 1
            layer.intermediate_size_per_partition = 4
            layer.allow_backend_reduction = True
            layer.moe_runner_config = SimpleNamespace(
                is_gated=True,
                activation="silu",
                apply_router_weight_on_input=False,
                no_combine=False,
                swiglu_limit=None,
            )
            layer.quant_method.moe_runner_config = layer.moe_runner_config
            _load_expert_fixture(layer, fixture, blockwise=not offline)
            return layer

        x = torch.ones(2, 4, dtype=torch.bfloat16)
        dispatch = SimpleNamespace(
            hidden_states=x,
            topk_output=(
                torch.ones(2, 1),
                torch.zeros(2, 1, dtype=torch.int32),
                None,
            ),
        )
        self.contexts.enter_context(
            patch.dict(
                "sys.modules",
                {"torch_tpu._internal.pallas": SimpleNamespace(jax_op=None)},
            )
        )
        # Earlier import-boundary patches can leave stale parent-package attrs.
        rs_op = importlib.import_module("sglang_tpu.moe.pallas.fused_moe_rs_op")
        with (
            get_context().override_server_args(tpu_fp8_runtime="channelwise"),
            patch.object(fp8, "_is_tpu", True),
            patch.object(fp8, "_is_cpu", False),
            patch.dict(os.environ, WORLD_SIZE="2"),
            get_parallel().override(**vars(parallel)),
            patch(
                "sglang_tpu.fp8_moe.current_platform.get_available_memory",
                return_value=(4 * 1024**3, 8 * 1024**3),
            ),
            patch("torch.accelerator.synchronize"),
            patch.object(rs_op, "_build_fused_moe_rs_op"),
            patch(
                "sglang_tpu.fp8_moe.apply_tpu_fp8_moe",
                side_effect=AssertionError(
                    "fused-RS silently fell back to ordinary MoE"
                ),
            ),
        ):
            for backend in (None, MoeRunnerBackend.PALLAS_TPU_FUSED_RS):
                for offline in (False, True):
                    with (
                        self.subTest(backend=backend, offline=offline),
                        get_flags().moe.override(runner_backend=backend),
                    ):
                        layer = make_layer(offline)
                        layer.quant_method.process_weights_after_loading(layer)
                        self.assertTrue(layer.quant_method.handles_moe_reduction)
                        self.assertFalse(layer._tpu_fused_rs_expert_tp)
                        self.assertEqual(layer._tpu_fused_rs_group_offset.tolist(), [0])
                        self.assert_moe_fixture(layer, fixture)
                        with patch.object(
                            rs_op,
                            "fused_moe_rs_gather",
                            return_value=x + 7,
                        ):
                            result = layer.quant_method.apply(layer, dispatch)
                        torch.testing.assert_close(result.hidden_states, x + 7)

            with get_parallel().override(
                attn_tp_size=1, attn_dp_size=2, attn_dp_enabled=True
            ):
                with get_flags().moe.override(
                    runner_backend=MoeRunnerBackend.PALLAS_TPU_FUSED_RS
                ):
                    with self.assertRaisesRegex(ValueError, "DP attention"):
                        layer = make_layer()
                        layer.quant_method.process_weights_after_loading(layer)
                with get_flags().moe.override(runner_backend=None):
                    layer = make_layer()
                    layer.quant_method.process_weights_after_loading(layer)
                    self.assertFalse(layer.quant_method.handles_moe_reduction)
                    self.assertIn(
                        "DP attention", layer._tpu_fused_rs_unsupported_reason
                    )

    def test_nextn_quant_config_hook_preserves_other_models_and_formats(self):
        from sglang.srt.models.glm4_moe import GlmMoeDsaForCausalLMNextN

        class UnreviewedNextN(GlmMoeDsaForCausalLMNextN):
            pass

        resolver = _glm_nextn_resolver()
        for layers, checkpoint_layer in ((3, 3), (1, 0)):
            hf_config = SimpleNamespace(
                architectures=["GlmMoeDsaForCausalLMNextN"],
                num_hidden_layers=layers,
            )
            excluded = f"model.layers.{checkpoint_layer}.self_attn.o_proj"
            config = _config(hf_config=hf_config, ignored_layers=[excluded])
            original_exclusions = list(config.ignored_layers)
            model = GlmMoeDsaForCausalLMNextN.__new__(GlmMoeDsaForCausalLMNextN)
            with self.subTest(layers=layers):
                # Upstream normalizes prefixes, but only the GLM hook maps NextN.
                self.assertNotIn(
                    "model.decoder.self_attn.o_proj", config.ignored_layers
                )
                resolved = resolver(model, hf_config, config)
                self.assertIn("model.decoder.self_attn.o_proj", resolved.ignored_layers)
                self.assertEqual(config.ignored_layers, original_exclusions)
                self.assertIs(resolver(model, hf_config, None), None)
                legacy = _block_config()
                self.assertIs(resolver(model, hf_config, legacy), legacy)
                other = UnreviewedNextN.__new__(UnreviewedNextN)
                self.assertIs(resolver(other, hf_config, config), config)

    def test_nextn_loads_channel_experts_and_preserves_exclusions_and_bridge(self):
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        from sglang.srt.models.glm4_moe import GlmMoeDsaForCausalLMNextN

        hf_config = SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLMNextN"],
            num_hidden_layers=3,
            num_nextn_predict_layers=1,
            n_routed_experts=2,
            q_lora_rank=None,
        )
        fixture = _expert_fixture()
        with get_context().override_server_args(moe_runner_backend="pallas_tpu"):
            config = _config(
                hf_config=hf_config,
                ignored_layers=["model.layers.3.self_attn.o_proj"],
            )
            model = GlmMoeDsaForCausalLMNextN.__new__(GlmMoeDsaForCausalLMNextN)
            resolver = _glm_nextn_resolver()
            original = config
            original_exclusions = list(original.ignored_layers)
            config = resolver(model, hf_config, original)
            self.assertIsNot(config, original)
            self.assertEqual(original.ignored_layers, original_exclusions)
            skipped = ReplicatedLinear.__new__(ReplicatedLinear)
            torch.nn.Module.__init__(skipped)
            self.assertIsInstance(
                config.get_quant_method(skipped, "model.decoder.self_attn.o_proj"),
                UnquantizedLinearMethod,
            )
            experts = _moe(config)
            model = GlmMoeDsaForCausalLMNextN.__new__(GlmMoeDsaForCausalLMNextN)
            torch.nn.Module.__init__(model)
            model.config = hf_config
            model.quant_config = config
            model.num_fused_shared_experts = 0
            model.pp_group = SimpleNamespace(is_last_rank=True)
            model.model = torch.nn.Module()
            model.model.decoder = torch.nn.Module()
            model.model.decoder.self_attn = torch.nn.Module()
            model.model.decoder.mlp = torch.nn.Module()
            model.model.decoder.mlp.experts = experts
            model.model.eh_proj = torch.nn.Linear(
                8, 4, bias=False, dtype=torch.bfloat16
            )
            weights = [("model.layers.3.eh_proj.weight", torch.full((4, 8), 0.5))]
            for stem, (_, _, q, scales) in fixture.items():
                for expert in range(2):
                    projections = (
                        (("gate_proj", slice(0, 4)), ("up_proj", slice(4, 8)))
                        if stem == "w13"
                        else (("down_proj", slice(None)),)
                    )
                    for projection, rows in projections:
                        prefix = f"model.layers.3.mlp.experts.{expert}.{projection}."
                        weights.append((prefix + "weight", q[expert, rows]))
                        weights.append(
                            (prefix + "weight_scale_inv", scales[expert, rows])
                        )
            weights.insert(
                0,
                (
                    "model.layers.2.mlp.experts.0.gate_proj.weight",
                    torch.zeros(4, 4),
                ),
            )
            model.load_weights(iter(weights))
            torch.testing.assert_close(
                model.model.eh_proj.weight,
                torch.full((4, 8), 0.5, dtype=torch.bfloat16),
            )
            experts.quant_method.process_weights_after_loading(experts)
            self.assert_moe_fixture(experts, fixture, check_dtype=False)


if __name__ == "__main__":
    unittest.main()
