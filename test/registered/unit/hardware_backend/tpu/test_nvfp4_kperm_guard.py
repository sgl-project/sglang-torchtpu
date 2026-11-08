"""KPERM-layout NVFP4 experts must not reach a native MoE GMM that misreads them (host-only)."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import fp4_moe
from sglang_tpu.muse_megakernel import ops

register_tpu_ci(est_time=2, suite="stage-a-test-tpu-unit", chips=1)


def _experts() -> torch.nn.Module:
    m = torch.nn.Module()
    # uint8 stand-ins (not packed fp4): K and N are one 256-block each.
    for name, shape in (
        ("w13_weight", (1, 256, 256)),
        ("w13_weight_scale", (1, 16, 256)),
        ("w2_weight", (1, 256, 256)),
        ("w2_weight_scale", (1, 16, 256)),
    ):
        data = torch.randint(0, 255, shape, dtype=torch.uint8)
        m.register_parameter(name, torch.nn.Parameter(data, requires_grad=False))
    return m


def _decoder(n_layers: int) -> SimpleNamespace:
    return SimpleNamespace(
        layers=[
            SimpleNamespace(mlp=SimpleNamespace(experts=_experts()))
            for _ in range(n_layers)
        ]
    )


def _refs(decoder) -> list:
    return [
        getattr(layer.mlp.experts, name)
        for layer in decoder.layers
        for name in ops._EXPERT_PARAMS
    ]


class TestKpermGuard(CustomTestCase):
    def test_megakernel_marks_permuted_layers(self):
        for groups in ((), ((0, 2),)):
            with (
                self.subTest(groups=groups),
                mock.patch("torch.accelerator.synchronize"),
            ):
                decoder = _decoder(3)
                ops._stack_expert_groups(
                    decoder,
                    _refs(decoder),
                    groups,
                    kinds=("fp4", "fp4", "fp8"),
                    kperm=True,
                )
                marked = [
                    fp4_moe.is_kperm_layout(layer.mlp.experts)
                    for layer in decoder.layers
                ]
                self.assertEqual(marked, [True, True, False])

    def test_no_kperm_leaves_layers_unmarked(self):
        decoder = _decoder(2)
        ops._stack_expert_groups(
            decoder, _refs(decoder), (), kinds=("fp4", "fp4"), kperm=False
        )
        self.assertFalse(
            any(fp4_moe.is_kperm_layout(l.mlp.experts) for l in decoder.layers)
        )

    def test_native_gmm_v2_raises_on_kperm_layout(self):
        layer = _experts()
        fp4_moe.check_native_kperm_layout(layer, use_fused_rs=False)
        fp4_moe.mark_kperm_layout(layer)
        fp4_moe.check_native_kperm_layout(layer, use_fused_rs=True)
        with self.assertRaisesRegex(RuntimeError, "KPERM layout"):
            fp4_moe.check_native_kperm_layout(layer, use_fused_rs=False)

    def test_apply_raises_before_computing(self):
        layer = _experts()
        fp4_moe.mark_kperm_layout(layer)
        with mock.patch.object(fp4_moe, "fused_moe_call") as gmm:
            with self.assertRaisesRegex(RuntimeError, "KPERM layout"):
                fp4_moe.apply_tpu_nvfp4_moe(
                    layer=layer,
                    dispatch_output=SimpleNamespace(),
                    activation="silu",
                    apply_router_weight_on_input=False,
                    use_fused_rs=False,
                )
            gmm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
