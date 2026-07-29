"""Unit tests for ``UnquantizedFusedMoEMethod.process_weights_after_loading`` (TPU).

Regression coverage for the square-weight storage bug: the TPU branch transposes
each expert weight to gmm_v2's ``(num_experts, k, n)`` layout and then frees the
pre-transpose storage via ``old.untyped_storage().resize_(0)``. That is only
safe when ``copy_or_rebind_param`` rebound the param to a fresh storage. It
rebinds on a shape change but copies **in place** when the shape is unchanged, so
a shape-preserving (square) transpose — ``w2`` when ``intermediate == hidden`` —
leaves the param pointing at ``old``'s storage; the ``resize_(0)`` then used to
zero out the live weight, giving a valid shape but a 0-byte storage that later
crashed the pallas MoE kernel.

The TPU branch is device-agnostic (transpose / contiguous / copy_ / resize_ /
data_ptr), so these run on CPU by forcing ``_is_tpu``.
"""

import unittest

import torch

from sglang.srt.layers.quantization import unquant
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.test.test_utils import CustomTestCase


class TestUnquantizedMoEWeightPrepTPU(CustomTestCase):
    def setUp(self):
        # Exercise the `if _is_tpu:` branch on a CPU host.
        self._orig_is_tpu = unquant._is_tpu
        unquant._is_tpu = True

    def tearDown(self):
        unquant._is_tpu = self._orig_is_tpu

    def _make_layer(self, num_experts, intermediate, hidden, dtype=torch.bfloat16):
        # SGLang stores experts as (num_experts, n, k): w13 is gate||up
        # (n = 2*intermediate), w2 is down-proj (n = intermediate, k = hidden).
        layer = torch.nn.Module()
        w13 = torch.randn(num_experts, 2 * intermediate, hidden, dtype=dtype)
        w2 = torch.randn(num_experts, intermediate, hidden, dtype=dtype)
        layer.register_parameter(
            "w13_weight", torch.nn.Parameter(w13, requires_grad=False)
        )
        layer.register_parameter(
            "w2_weight", torch.nn.Parameter(w2, requires_grad=False)
        )
        return layer, {"w13_weight": w13.clone(), "w2_weight": w2.clone()}

    def _run_prep(self, layer):
        # self is unused in the TPU branch, so skip __init__ (which pulls in the
        # MoE runner backend / global server args).
        method = UnquantizedFusedMoEMethod.__new__(UnquantizedFusedMoEMethod)
        method.process_weights_after_loading(layer)

    def _assert_materialized_and_transposed(self, layer, ref):
        for name in ("w13_weight", "w2_weight"):
            got = getattr(layer, name).data
            self.assertNotEqual(
                got.untyped_storage().nbytes(),
                0,
                f"{name} was left with 0-byte storage (weight freed)",
            )
            expected = ref[name].transpose(1, 2).contiguous()
            self.assertEqual(tuple(got.shape), tuple(expected.shape), f"{name} shape")
            self.assertTrue(torch.equal(got, expected), f"{name} values changed")

    def test_square_w2_is_materialized(self):
        # intermediate == hidden -> w2's transpose is shape-preserving: the bug
        # case. Before the fix, w2_weight ended up with 0-byte storage.
        layer, ref = self._make_layer(num_experts=4, intermediate=128, hidden=128)
        self._run_prep(layer)
        self._assert_materialized_and_transposed(layer, ref)

    def test_nonsquare_w2_is_materialized(self):
        # intermediate != hidden -> w2 rebinds like w13 (was never broken).
        layer, ref = self._make_layer(num_experts=4, intermediate=256, hidden=128)
        self._run_prep(layer)
        self._assert_materialized_and_transposed(layer, ref)

    def test_pretranspose_storage_freed_when_rebound(self):
        # The memory optimization must still free the pre-transpose storage in
        # the rebind (non-square) case; only the in-place (square) case keeps it.
        layer, _ = self._make_layer(num_experts=4, intermediate=256, hidden=128)
        old_w13_storage = (
            layer.w13_weight.data.untyped_storage()
        )  # non-square -> rebind
        old_w2_storage = layer.w2_weight.data.untyped_storage()  # (256,128) -> rebind
        self._run_prep(layer)
        self.assertEqual(
            old_w13_storage.nbytes(), 0, "non-square w13 old storage not freed"
        )
        self.assertEqual(
            old_w2_storage.nbytes(), 0, "non-square w2 old storage not freed"
        )
        self.assertNotEqual(layer.w13_weight.data.untyped_storage().nbytes(), 0)
        self.assertNotEqual(layer.w2_weight.data.untyped_storage().nbytes(), 0)


if __name__ == "__main__":
    unittest.main()
