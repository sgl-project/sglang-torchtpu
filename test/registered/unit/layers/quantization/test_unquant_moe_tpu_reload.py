"""A TPU unquantized MoE layer must take a weight reload after its gmm_v2 layout transform."""

import types
import unittest
from unittest import mock

import torch

from sglang.srt.layers.quantization import unquant
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

EXPERTS = 2
HIDDEN_SIZE = 128


def _layer(intermediate: int) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.moe_runner_config = types.SimpleNamespace(is_gated=True)
    layer.tpu_gmm_pad = None
    layer.w13_weight = torch.nn.Parameter(
        torch.randn(EXPERTS, 2 * intermediate, HIDDEN_SIZE), requires_grad=False
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.randn(EXPERTS, HIDDEN_SIZE, intermediate), requires_grad=False
    )
    return layer


class TestUnquantizedMoeTpuReload(CustomTestCase):
    def _check_reload(self, intermediate: int) -> None:
        method = object.__new__(unquant.UnquantizedFusedMoEMethod)
        layer = _layer(intermediate)
        loaded = {
            name: getattr(layer, name).detach().clone()
            for name in ("w13_weight", "w2_weight")
        }
        params = {name: getattr(layer, name) for name in loaded}
        with mock.patch.object(unquant, "_is_tpu", True):
            method.process_weights_after_loading(layer)
            served = {name: getattr(layer, name).detach().clone() for name in loaded}

            method.restore_weights_before_loading(layer)
            for name, weight in loaded.items():
                # the FusedMoE weight loader writes this (E, n, k) layout, unpadded
                torch.testing.assert_close(
                    getattr(layer, name).data, weight, rtol=0, atol=0
                )

            method.process_weights_after_loading(layer)
            for name, weight in served.items():
                torch.testing.assert_close(
                    getattr(layer, name).data, weight, rtol=0, atol=0
                )
                self.assertIs(
                    getattr(layer, name),
                    params[name],
                    "the Parameter object must survive a reload",
                )

    def test_reload_restores_the_loader_layout(self):
        self._check_reload(intermediate=256)

    def test_reload_strips_the_128_lane_padding(self):
        self._check_reload(intermediate=96)

    def test_reload_keeps_a_square_w2(self):
        self._check_reload(intermediate=HIDDEN_SIZE)


if __name__ == "__main__":
    unittest.main()
