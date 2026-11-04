"""Inkling2 expert loading narrows to the EP rank's experts before converting them."""

import types
import unittest
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

import torch

from sglang.test.test_utils import CustomTestCase

E, EP, N, K = 8, 4, 256, 256  # w13 rows N = 2 * intermediate (interleaved gate/up); N, K: whole FP8 blocks


def _model(precision, ep_rank):
    import sglang_tpu.models.inkling2 as m

    self = types.SimpleNamespace(
        text_config=types.SimpleNamespace(n_routed_experts=E),
        quant_config=types.SimpleNamespace(expert_precision=lambda prefix: precision),
        _pending={},
    )
    for fn in ("_load_experts", "_load_packed_nvfp4", "_take_nvfp4"):
        setattr(self, fn, types.MethodType(getattr(m.Inkling2ForConditionalGeneration, fn), self))
    parallel = types.SimpleNamespace(moe_ep_size=EP, moe_ep_rank=ep_rank)
    return m, self, mock.patch.object(m, "get_parallel", return_value=parallel)


class TestInkling2ExpertLoad(CustomTestCase):
    prefix = "llm.layers.0.mlp.experts."

    def test_mxfp8_matches_convert_then_slice(self):
        g = torch.Generator().manual_seed(0)
        weight = (torch.randn(E, N, K, generator=g) * 3).to(torch.float8_e4m3fn)
        scale = torch.randint(120, 134, (E, N, K // 32), generator=g, dtype=torch.uint8)
        for rank in range(EP):
            m, model, patch = _model("mxfp8", rank)
            local = E // EP
            params = {
                self.prefix + "w13_weight": torch.empty(local, N, K, dtype=torch.float8_e4m3fn),
                self.prefix + "w13_weight_scale_inv": torch.empty(
                    local, N // m.FP8_BLOCK[0], K // m.FP8_BLOCK[1]
                ),
            }
            loaded = set()
            with patch:
                model._load_experts(params, self.prefix + "w13_weight_scale", scale, loaded)
                model._load_experts(params, self.prefix + "w13_weight", weight, loaded)
                full = m._deinterleave_gate_up(m._dequant_mxfp8(weight, scale))
                want_w, want_s = m._quant_block_fp8(full[rank * local : (rank + 1) * local])
            self.assertEqual(loaded, set(params))
            self.assertTrue(torch.equal(params[self.prefix + "w13_weight"].view(torch.uint8), want_w.view(torch.uint8)))
            self.assertTrue(torch.equal(params[self.prefix + "w13_weight_scale_inv"], want_s))
            self.assertEqual(model._pending, {})

    def test_packed_nvfp4_matches_slice_of_full(self):
        g = torch.Generator().manual_seed(1)
        weight = torch.randint(0, 256, (E, N, K // 2), generator=g, dtype=torch.uint8)
        amax = torch.rand(E, generator=g) + 0.5
        den = torch.tensor([2688.0])
        for rank in range(EP):
            m, model, patch = _model("nvfp4_packed", rank)
            local = E // EP
            params = {
                self.prefix + "w13_weight": torch.empty(local, N, K // 2, dtype=torch.uint8),
                self.prefix + "w13_weight_scale_2": torch.empty(local, 2),
            }
            loaded = set()
            with patch:
                model._load_experts(params, self.prefix + "w13_weight", weight, loaded)
                model._load_experts(params, self.prefix + "w13_weight_amax", amax, loaded)
                model._load_experts(params, self.prefix + "w13_weight_outer_scale_denominator", den, loaded)
                want = m._deinterleave_gate_up(weight)[rank * local : (rank + 1) * local]
            sl = slice(rank * local, (rank + 1) * local)
            self.assertTrue(torch.equal(params[self.prefix + "w13_weight"], want))
            self.assertTrue(torch.equal(params[self.prefix + "w13_weight_scale_2"], (amax[sl] / 2688.0)[:, None].expand(-1, 2)))


if __name__ == "__main__":
    unittest.main()
