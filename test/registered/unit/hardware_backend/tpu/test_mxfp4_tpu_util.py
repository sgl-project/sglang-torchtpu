"""Unit tests for the TPU MXFP4 MoE weight preparation (CPU only)."""

from sglang.test.ci.ci_register import register_cpu_ci, register_tpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit")

import importlib.util
import sys
import types
import unittest
from unittest import mock

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.quantization import mxfp4_tpu_util
from sglang.srt.layers.quantization.mxfp4_tpu_util import (
    _EXPERTS_PER_CHUNK,
    prepare_tpu_mxfp4_moe_weights,
    process_mxfp4_moe_weights_bf16,
)
from sglang.test.test_utils import CustomTestCase

HIDDEN, INTER = 64, 96
# Wider than INTER's 128-lane width, as gpt-oss's 2880 pads to 3072 on tpu7x.
PADDED_INTER = 256


def _fake_layer(
    num_experts: int,
    seed: int = 0,
    activation: str = "swigluoai",
    hidden: int = HIDDEN,
) -> types.SimpleNamespace:
    g = torch.Generator().manual_seed(seed)
    u8 = lambda *shape: torch.randint(0, 256, shape, dtype=torch.uint8, generator=g)
    # e8m0 scales near 1.0 keep the dequantized values finite.
    e8 = lambda *shape: torch.randint(120, 134, shape, dtype=torch.uint8, generator=g)
    layer = types.SimpleNamespace()
    layer.w13_weight = Parameter(
        u8(num_experts, 2 * INTER, hidden // 2), requires_grad=False
    )
    layer.w13_weight_scale = Parameter(
        e8(num_experts, 2 * INTER, hidden // 32), requires_grad=False
    )
    layer.w13_weight_bias = Parameter(
        torch.randn(num_experts, 2 * INTER, generator=g, dtype=torch.bfloat16),
        requires_grad=False,
    )
    layer.w2_weight = Parameter(
        u8(num_experts, hidden, INTER // 2), requires_grad=False
    )
    layer.w2_weight_scale = Parameter(
        e8(num_experts, hidden, INTER // 32), requires_grad=False
    )
    layer.w2_weight_bias = Parameter(
        torch.randn(num_experts, hidden, generator=g, dtype=torch.bfloat16),
        requires_grad=False,
    )
    layer.moe_runner_config = types.SimpleNamespace(activation=activation)
    return layer


def _process_whole(layer, **kwargs):
    return process_mxfp4_moe_weights_bf16(
        layer.w13_weight.data,
        layer.w13_weight_scale.data,
        layer.w13_weight_bias.data,
        layer.w2_weight.data,
        layer.w2_weight_scale.data,
        layer.w2_weight_bias.data,
        w13_interleaved=True,
        **kwargs,
    )


def _gmm_width(width):
    """Stand in for the tiler query, which reads the TPU's metadata."""
    return mock.patch.object(
        mxfp4_tpu_util, "_gmm_padded_intermediate", return_value=width
    )


class TestChunkedExpertPreparation(CustomTestCase):
    def test_experts_prepared_in_groups_match_the_whole_tensor(self):
        """The experts are dequantized a group at a time; with a group count
        that does not divide the expert count, the assembled weights and
        biases must equal the single-tensor result expert for expert, at the
        width the loader pads to."""
        num_experts = 2 * _EXPERTS_PER_CHUNK + 8
        layer = _fake_layer(num_experts)
        whole = _process_whole(layer, padded_intermediate=PADDED_INTER)
        # A layer this small is under the chunking threshold; drop it so the
        # experts really are converted a group at a time.
        with (
            mock.patch.object(mxfp4_tpu_util, "_MAX_DEQUANT_ELEMENTS", 0),
            _gmm_width(PADDED_INTER),
        ):
            prepare_tpu_mxfp4_moe_weights(layer, with_bias=True)
        got = (
            layer.w13_weight,
            layer.w13_weight_bias,
            layer.w2_weight,
            layer.w2_weight_bias,
        )
        for name, a, b in zip(("w13", "w13_bias", "w2", "w2_bias"), whole, got):
            self.assertEqual(tuple(a.shape), tuple(b.shape), name)
            self.assertEqual(a.dtype, b.dtype, name)
            self.assertTrue(torch.equal(a, b), name)
        self.assertIsNone(layer.w13_weight_scale)
        self.assertIsNone(layer.w2_weight_scale)


class TestPackedPreparation(CustomTestCase):
    def test_non_interleaved_aligned_keeps_packed_fp4(self):
        """Kimi-K3-style checkpoints keep packed FP4 in gmm_v2's (E, K, N) layout."""
        layer = _fake_layer(num_experts=4, activation="situ", hidden=128)
        prepare_tpu_mxfp4_moe_weights(layer, with_bias=False)
        n_pad = 128  # INTER=96 pads each gate/up half to 128
        self.assertEqual(layer.w13_weight.dtype, torch.uint8)
        self.assertEqual(tuple(layer.w13_weight.shape), (4, 128 // 2, 2 * n_pad))
        self.assertEqual(layer.w13_weight_scale.dtype, torch.float32)
        self.assertEqual(
            tuple(layer.w13_weight_scale.shape), (4, 128 // 32, 1, 2 * n_pad)
        )
        self.assertEqual(tuple(layer.w2_weight.shape), (4, n_pad // 2, 128))
        self.assertEqual(tuple(layer.w2_weight_scale.shape), (4, n_pad // 32, 1, 128))
        self.assertIsNone(layer.w13_weight_bias)
        self.assertIsNone(layer.w2_weight_bias)

    def test_misaligned_hidden_falls_back_to_bf16(self):
        """The packed path cannot pad K, so a misaligned hidden size dequantizes."""
        layer = _fake_layer(num_experts=4, activation="situ")
        with _gmm_width(128):
            prepare_tpu_mxfp4_moe_weights(layer, with_bias=True)
        self.assertEqual(layer.w13_weight.dtype, torch.bfloat16)
        self.assertIsNone(layer.w13_weight_scale)
        self.assertIsNone(layer.w2_weight_scale)


class TestIntermediatePadding(CustomTestCase):
    def test_loader_pads_the_intermediate_to_the_gmm_width(self):
        """The loader asks the tiler for the width at which GMM1 runs the
        fewest N tiles and zero-pads gate, up and w2's K to it. The
        checkpoint's values are unchanged: gate stays at the front of w13's N
        and up starts at the padded width."""
        layer = _fake_layer(4)
        # By default the intermediate pads to the next multiple of 128.
        base_w13, base_w13_b, base_w2, base_w2_b = _process_whole(layer)
        self.assertEqual(base_w13.shape[-1], 2 * 128)
        with _gmm_width(PADDED_INTER) as width:
            prepare_tpu_mxfp4_moe_weights(layer, with_bias=True)
        width.assert_called_once_with(HIDDEN, INTER, "swigluoai")

        w13, w13_b = layer.w13_weight, layer.w13_weight_bias
        w2, w2_b = layer.w2_weight, layer.w2_weight_bias
        hidden_padded = 128
        self.assertEqual(tuple(w13.shape), (4, hidden_padded, 2 * PADDED_INTER))
        self.assertEqual(tuple(w13_b.shape), (4, 1, 2 * PADDED_INTER))
        self.assertEqual(tuple(w2.shape), (4, PADDED_INTER, hidden_padded))
        for name, got, base in (
            ("w13", w13, base_w13),
            ("w13_bias", w13_b, base_w13_b),
        ):
            with self.subTest(name):
                gate, up = got[..., :PADDED_INTER], got[..., PADDED_INTER:]
                self.assertTrue(torch.equal(gate[..., :INTER], base[..., :INTER]))
                self.assertTrue(
                    torch.equal(up[..., :INTER], base[..., 128 : 128 + INTER])
                )
                self.assertFalse(gate[..., INTER:].any())
                self.assertFalse(up[..., INTER:].any())
        self.assertTrue(torch.equal(w2[:, :INTER], base_w2[:, :INTER]))
        self.assertFalse(w2[:, INTER:].any())
        self.assertTrue(torch.equal(w2_b, base_w2_b))

    def test_width_must_be_lane_aligned_and_cover_the_intermediate(self):
        layer = _fake_layer(1)
        # 0 is short of INTER; 160 covers it but is off the 128-lane grid.
        for width in (0, 160):
            with self.subTest(width=width), self.assertRaises(AssertionError):
                _process_whole(layer, padded_intermediate=width)

    @unittest.skipUnless(importlib.util.find_spec("jax"), "requires JAX")
    def test_gmm_padded_intermediate(self):
        """Exercise ``_gmm_padded_intermediate`` end-to-end (without
        ``_gmm_width``): it rounds hidden up to 128 lanes, maps the SGLang
        activation name to gmm_v2's ``fuse_act`` spelling, swaps in the
        chip's Pallas metadata via ``tpu_info_for_cpu_export``, and queries
        ``pad_size_n_to_fewest_tiles``."""
        from sglang_tpu import pallas_export

        # fused_moe_op imports torch_tpu at module scope; stub it when
        # running on a CPU host that has JAX but not torch_tpu.
        stubs = {}
        if importlib.util.find_spec("torch_tpu") is None:
            for mod in (
                "torch_tpu",
                "torch_tpu._internal",
                "torch_tpu._internal.pallas",
            ):
                stubs[mod] = types.SimpleNamespace(jax_op=None)

        with mock.patch.dict(sys.modules, stubs):
            with mock.patch.object(
                pallas_export, "get_torch_tpu_device_kind", return_value="TPU v7"
            ):
                # GPT-OSS TP1 (hidden = inter = 2880) pads to 3072 on tpu7x
                # (64 MiB VMEM), for both swiglu_oai/swigluoai and situ.
                for act in ("swigluoai", "swiglu_oai", "situ"):
                    with self.subTest(chip="TPU v7", activation=act):
                        self.assertEqual(
                            mxfp4_tpu_util._gmm_padded_intermediate(2880, 2880, act),
                            3072,
                        )
                # End-to-end through prepare_tpu_mxfp4_moe_weights without
                # mocking _gmm_padded_intermediate (HIDDEN=64, INTER=96 -> 128).
                layer = _fake_layer(2)
                prepare_tpu_mxfp4_moe_weights(layer, with_bias=True)
                self.assertEqual(tuple(layer.w13_weight.shape), (2, 128, 2 * 128))
                self.assertEqual(tuple(layer.w2_weight.shape), (2, 128, 128))

            with mock.patch.object(
                pallas_export, "get_torch_tpu_device_kind", return_value="TPU v6e"
            ):
                # On v6e (128 MiB VMEM), 2944 fits in a single N tile.
                self.assertEqual(
                    mxfp4_tpu_util._gmm_padded_intermediate(2880, 2880, "swigluoai"),
                    2944,
                )


if __name__ == "__main__":
    unittest.main()
