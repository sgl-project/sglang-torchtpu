"""Kimi-K3 KDA weight re-homing after load (sglang_tpu/hooks/kda_weights.py), on CPU."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import pkgutil
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.utils import PPMissingLayer
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.hooks import HOOKS
from sglang_tpu.hooks.kda_weights import rehome_kimi_k3_kda_weights


class _FakeKda(nn.Module):
    """The attributes rehome_kimi_k3_kda_weights reads from a bound KimiK3DeltaAttention."""

    def __init__(self, bound=True):
        super().__init__()
        conv = nn.Parameter(torch.arange(12.0).reshape(6, 2), requires_grad=False)
        conv.data = conv.data.unsqueeze(1)  # a view, as kimi_k3 builds it at load
        self.qkv_conv1d = SimpleNamespace(weight=conv)
        self.dt_bias = nn.Parameter(torch.ones(4), requires_grad=False)
        self.A_log = nn.Parameter(torch.zeros(1, 4), requires_grad=False)
        self.attn = SimpleNamespace(
            A_log=nn.Parameter(
                self.A_log.detach().reshape(-1).clone(), requires_grad=False
            ),
            packed_kernel_weights=torch.zeros(3) if bound else None,
        )
        self.packs = 0

    def _pack_tpu_kda_weights(self):
        self.packs += 1
        return torch.full((3,), float(self.packs))


class TestKimiK3KdaWeightRehome(CustomTestCase):
    def test_hook_targets_kimi_k3_post_load(self):
        targets = [t for t, fn, _ in HOOKS if fn is rehome_kimi_k3_kda_weights]
        self.assertEqual(
            targets,
            ["sglang.srt.models.kimi_k3.KimiK3LinearForCausalLM.post_load_weights"],
        )
        self.assertTrue(callable(pkgutil.resolve_name(targets[0])))

    def test_bound_kda_weights_are_rehomed_and_repacked(self):
        kda, unbound = _FakeKda(), _FakeKda(bound=False)
        layers = [
            PPMissingLayer(),
            SimpleNamespace(self_attn=nn.Identity()),
            SimpleNamespace(self_attn=unbound),
            SimpleNamespace(self_attn=kda),
        ]
        causal_lm = SimpleNamespace(model=SimpleNamespace(layers=layers))
        rehomed = []
        # On CPU, .cpu() returns the tensor itself, so record which tensors go
        # through the host instead of checking their storage.
        with (
            patch("sglang.srt.models.kimi_k3.KimiK3DeltaAttention", _FakeKda),
            patch("sglang_tpu.hooks.kda_weights._rehome", side_effect=rehomed.append),
        ):
            rehome_kimi_k3_kda_weights(None, causal_lm)

        expected = [kda.qkv_conv1d.weight, kda.dt_bias, kda.A_log, kda.attn.A_log]
        self.assertEqual([id(t) for t in rehomed], [id(t) for t in expected])
        self.assertEqual(kda.packs, 1)
        torch.testing.assert_close(kda.attn.packed_kernel_weights, torch.ones(3))
        self.assertEqual(unbound.packs, 0)

    def test_rehome_keeps_the_parameter_and_its_values(self):
        from sglang_tpu.hooks.kda_weights import _rehome

        conv = nn.Parameter(torch.arange(12.0).reshape(6, 1, 2), requires_grad=False)
        before = conv.detach().clone()
        _rehome(conv)
        torch.testing.assert_close(conv.data, before)
        self.assertEqual(conv.device, before.device)


if __name__ == "__main__":
    unittest.main()
