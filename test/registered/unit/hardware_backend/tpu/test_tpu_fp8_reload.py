"""Online RL weight reload tests for TPU FP8 dense linear and MoE layers."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from sglang.srt.layers.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization import fp8 as fp8_mod
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod, Fp8MoEMethod
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.plugins.hook_registry import HookRegistry
from sglang_tpu.fp8_linear import (
    prepare_tpu_online_fp8_linear_weights,
    record_tpu_fp8_linear_reload_state,
    restore_tpu_fp8_linear_weights_before_loading,
)
from sglang_tpu.rl.fp8_reload import hook_specs

_FP8 = torch.float8_e4m3fn


def _loader_fn(param: torch.Tensor, loaded: torch.Tensor) -> None:
    param.data.copy_(loaded)


class TestTpuFp8Reload(unittest.TestCase):
    def _install_reload_hooks(self) -> None:
        saved_hooks = {k: list(v) for k, v in HookRegistry._hooks.items()}
        saved_patched = set(HookRegistry._patched)
        saved_attrs: list[tuple[type, str, bool, object]] = []
        for cls in (Fp8LinearMethod, Fp8MoEMethod):
            for attr in (
                "process_weights_after_loading",
                "restore_weights_before_loading",
            ):
                owned = attr in cls.__dict__
                saved_attrs.append(
                    (cls, attr, owned, cls.__dict__.get(attr) if owned else None)
                )

        def _cleanup() -> None:
            HookRegistry._hooks.clear()
            HookRegistry._hooks.update(saved_hooks)
            HookRegistry._patched.clear()
            HookRegistry._patched.update(saved_patched)
            for cls, attr, owned, orig in saved_attrs:
                if owned:
                    setattr(cls, attr, orig)
                elif attr in cls.__dict__:
                    delattr(cls, attr)

        self.addCleanup(_cleanup)
        HookRegistry.reset()
        for target, hook, hook_type in hook_specs():
            HookRegistry.register(target, hook, hook_type)
        HookRegistry.apply_hooks()

    def test_online_fp8_linear_preserves_parameter_identity_and_weight_loader_across_reloads(
        self,
    ):
        layer = nn.Module()
        layer.orig_dtype = torch.bfloat16
        layer.weight = ModelWeightParameter(
            data=torch.randn(32, 64, dtype=torch.bfloat16),
            input_dim=1,
            output_dim=0,
            weight_loader=_loader_fn,
        )
        record_tpu_fp8_linear_reload_state(layer)
        orig_weight_param = layer.weight

        prepare_tpu_online_fp8_linear_weights(layer, dtype=_FP8)
        orig_scale_param = layer.weight_scale
        self.assertIs(layer.weight, orig_weight_param)
        self.assertIs(layer.weight.weight_loader, _loader_fn)
        self.assertEqual(tuple(layer.weight.shape), (64, 32))

        restore_tpu_fp8_linear_weights_before_loading(layer)
        self.assertIs(layer.weight, orig_weight_param)
        self.assertEqual(tuple(layer.weight.shape), (32, 64))
        self.assertEqual(layer.weight.dtype, torch.bfloat16)

        fresh = torch.randn(32, 64, dtype=torch.bfloat16) * 3.0
        layer.weight.weight_loader(layer.weight, fresh)
        prepare_tpu_online_fp8_linear_weights(layer, dtype=_FP8)

        self.assertIs(layer.weight, orig_weight_param)
        self.assertIs(layer.weight_scale, orig_scale_param)
        self.assertEqual(tuple(layer.weight.shape), (64, 32))
        dequant = layer.weight.t().float() * layer.weight_scale
        rel_err = (dequant - fresh.float()).abs().max() / fresh.float().abs().max()
        self.assertLess(rel_err.item(), 0.08)

    def test_block_fp8_linear_captures_post_create_wrapped_loader_for_channelwise_reload(
        self,
    ):
        self._install_reload_hooks()
        cfg = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
        with (
            patch.object(fp8_mod, "_is_tpu", True),
            patch.object(fp8_mod, "_is_cpu", False),
            patch.object(fp8_mod, "_is_fp8_fnuz", False),
            patch.object(fp8_mod, "dispatch_w8a8_block_fp8_linear", return_value=None),
            patch.object(fp8_mod, "can_serve_block_fp8_as_mxfp8", return_value=False),
            patch(
                "sglang_tpu.fp8_linear._get_tpu_fp8_runtime", return_value="channelwise"
            ),
        ):
            method = Fp8LinearMethod(cfg)
            layer = nn.Module()
            layer.quant_method = method
            method.create_weights(
                layer=layer,
                input_size_per_partition=128,
                output_partition_sizes=[128],
                input_size=128,
                output_size=128,
                params_dtype=torch.bfloat16,
                skip_block_quant_check=True,
                weight_loader=_loader_fn,
            )

            # Simulate Qwen3.5 GDN _bind_packed_weight_loaders wrapping _weight_loader
            # *after* create_weights returns.
            wrapped_calls: list[str] = []

            def _wrapped_scale_loader(
                param: torch.Tensor, loaded: torch.Tensor
            ) -> None:
                wrapped_calls.append("scale_inv")
                param.data.copy_(loaded)

            layer.weight_scale_inv._weight_loader = _wrapped_scale_loader
            layer.weight.data.fill_(1.0)
            layer.weight_scale_inv.data.fill_(0.5)

            method.process_weights_after_loading(layer)
            self.assertIsNone(layer.weight_scale_inv)

            DefaultModelLoader.restore_weights_before_loading(
                layer, torch.device("cpu")
            )
            self.assertIsInstance(layer.weight_scale_inv, BlockQuantScaleParameter)
            self.assertIs(layer.weight_scale_inv.weight_loader, _wrapped_scale_loader)

            layer.weight.weight_loader(
                layer.weight, torch.full((128, 128), 2.0, dtype=_FP8)
            )
            layer.weight_scale_inv.weight_loader(
                layer.weight_scale_inv, torch.full((1, 1), 4.0, dtype=torch.float32)
            )
            method.process_weights_after_loading(layer)

        self.assertEqual(wrapped_calls, ["scale_inv"])
        dequant = layer.weight.t().float() * layer.weight_scale
        self.assertTrue(torch.allclose(dequant, torch.full((128, 128), 8.0), atol=0.5))

    def test_pertensor_fp8_linear_restores_vllm_parameters_after_core_replaces_them(
        self,
    ):
        layer = nn.Module()
        layer.weight = ModelWeightParameter(
            data=torch.ones(32, 64, dtype=_FP8),
            input_dim=1,
            output_dim=0,
            weight_loader=_loader_fn,
        )
        layer.weight_scale = PerTensorScaleParameter(
            data=torch.tensor([0.5], dtype=torch.float32),
            weight_loader=_loader_fn,
        )
        record_tpu_fp8_linear_reload_state(layer)

        # Simulate core per-tensor FP8 post-processing replacing wrappers with plain Parameters.
        layer.weight = nn.Parameter(
            layer.weight.data.t().contiguous(), requires_grad=False
        )
        layer.weight_scale = nn.Parameter(
            torch.full((32,), 0.5, dtype=torch.float32), requires_grad=False
        )

        restore_tpu_fp8_linear_weights_before_loading(layer)
        self.assertIsInstance(layer.weight, ModelWeightParameter)
        self.assertIsInstance(layer.weight_scale, PerTensorScaleParameter)
        self.assertIs(layer.weight.weight_loader, _loader_fn)
        self.assertIs(layer.weight_scale.weight_loader, _loader_fn)
        self.assertEqual(tuple(layer.weight.shape), (32, 64))
        self.assertEqual(tuple(layer.weight_scale.shape), (1,))

    def test_online_fp8_moe_hook_runs_channelwise_quant_fused_rs_and_dispatcher_update(
        self,
    ):
        self._install_reload_hooks()
        cfg = Fp8Config(is_checkpoint_fp8_serialized=False)
        fake_parallel = SimpleNamespace(pp_size=1, tp_rank=0)
        fake_runner_backend = SimpleNamespace(
            is_auto=lambda: True,
            is_cutlass=lambda: False,
            is_pallas_tpu_fused_rs=lambda: False,
        )
        with (
            patch.object(
                fp8_mod, "get_moe_runner_backend", return_value=fake_runner_backend
            ),
            patch("sglang_tpu.fp8_moe.get_parallel", return_value=fake_parallel),
            patch(
                "sglang_tpu.fp8_moe._get_tpu_fp8_runtime", return_value="channelwise"
            ),
            patch(
                "sglang.srt.layers.moe.get_moe_runner_backend",
                return_value=fake_runner_backend,
            ),
            patch(
                "sglang_tpu.fp8_moe_fused_rs.prepare_fp8_fused_rs", return_value=True
            ) as mock_fused_rs,
            patch(
                "sglang.srt.layers.moe.token_dispatcher.base.build_dispatcher_quant_config",
                return_value={"quant_dtype": _FP8},
            ),
        ):
            method = Fp8MoEMethod(cfg)
            method._aiter_moe_quant_type = lambda: None

            layer = nn.Module()
            layer.quant_method = method
            layer.num_local_experts = 4
            w13 = nn.Parameter(
                torch.randn(4, 32, 64, dtype=torch.bfloat16), requires_grad=False
            )
            w2 = nn.Parameter(
                torch.randn(4, 64, 16, dtype=torch.bfloat16), requires_grad=False
            )
            w13.weight_loader = _loader_fn
            w2.weight_loader = _loader_fn
            layer.register_parameter("w13_weight", w13)
            layer.register_parameter("w2_weight", w2)
            layer.register_parameter(
                "w13_weight_scale", nn.Parameter(torch.ones(4, 2), requires_grad=False)
            )
            layer.register_parameter(
                "w2_weight_scale", nn.Parameter(torch.ones(4), requires_grad=False)
            )
            layer.dispatcher = MagicMock()

            method.process_weights_after_loading(layer)
            self.assertTrue(method.handles_moe_reduction)
            mock_fused_rs.assert_called_once_with(layer)
            layer.dispatcher.set_quant_config.assert_called_once_with(
                {"quant_dtype": _FP8, "use_mxfp8": False}
            )
            w13_scale_param = layer.w13_weight_scale
            w2_scale_param = layer.w2_weight_scale

            DefaultModelLoader.restore_weights_before_loading(
                layer, torch.device("cpu")
            )
            self.assertIs(layer.w13_weight, w13)
            self.assertIs(layer.w2_weight, w2)
            self.assertEqual(tuple(layer.w13_weight.shape), (4, 32, 64))
            self.assertEqual(layer.w13_weight.dtype, torch.bfloat16)

            fresh_w13 = torch.randn(4, 32, 64, dtype=torch.bfloat16) * 2.0
            fresh_w2 = torch.randn(4, 64, 16, dtype=torch.bfloat16) * 2.0
            layer.w13_weight.weight_loader(layer.w13_weight, fresh_w13)
            layer.w2_weight.weight_loader(layer.w2_weight, fresh_w2)
            method.process_weights_after_loading(layer)

        self.assertIs(layer.w13_weight, w13)
        self.assertIs(layer.w13_weight_scale, w13_scale_param)
        self.assertIs(layer.w2_weight_scale, w2_scale_param)
        self.assertEqual(mock_fused_rs.call_count, 2)
        dequant_w13 = (
            layer.w13_weight.transpose(1, 2).float()
            * layer.w13_weight_scale[:, 0, 0, :, None].float()
        )
        rel_err = (
            dequant_w13 - fresh_w13.float()
        ).abs().max() / fresh_w13.float().abs().max()
        self.assertLess(rel_err.item(), 0.08)

    def test_reload_fails_loud_where_core_post_processing_rebinds_parameters(self):
        """Per-tensor and non-channelwise FP8 reloads must raise instead of serving
        the stale parameters direct_compile captured; supported modes must not."""
        from sglang_tpu.rl.fp8_reload import (
            restore_fp8_linear_before_load,
            restore_fp8_moe_before_load,
        )

        def linear(*, serialized, block_quant):
            return SimpleNamespace(
                use_mxfp8=False,
                is_checkpoint_fp8_serialized=serialized,
                block_quant=block_quant,
            )

        def moe(*, serialized, block_quant):
            return SimpleNamespace(
                is_fp4_expert=False,
                use_mxfp8=False,
                block_quant=block_quant,
                quant_config=SimpleNamespace(is_checkpoint_fp8_serialized=serialized),
            )

        layer = nn.Module()
        with self.assertRaises(NotImplementedError):
            restore_fp8_linear_before_load(
                None, linear(serialized=True, block_quant=False), layer
            )
        restore_fp8_linear_before_load(
            None, linear(serialized=False, block_quant=False), layer
        )
        restore_fp8_linear_before_load(
            None, linear(serialized=True, block_quant=True), layer
        )

        with patch(
            "sglang_tpu.fp8_moe._get_tpu_fp8_runtime", return_value="channelwise"
        ):
            with self.assertRaises(NotImplementedError):
                restore_fp8_moe_before_load(
                    None, moe(serialized=True, block_quant=False), layer
                )
            restore_fp8_moe_before_load(
                None, moe(serialized=False, block_quant=False), layer
            )
            restore_fp8_moe_before_load(
                None, moe(serialized=True, block_quant=True), layer
            )
        with patch("sglang_tpu.fp8_moe._get_tpu_fp8_runtime", return_value="blockwise"):
            with self.assertRaises(NotImplementedError):
                restore_fp8_moe_before_load(
                    None, moe(serialized=False, block_quant=False), layer
                )

    def test_rl_fp8_reload_hook_specs(self):
        specs = hook_specs()
        self.assertEqual(len(specs), 4)


if __name__ == "__main__":
    unittest.main()
