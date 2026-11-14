"""Tests for the Muse prefill program's dispatch and config selection (host-only; needs jax)."""

import unittest
from types import SimpleNamespace

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.muse_megakernel import ops
from sglang_tpu.muse_megakernel.prefill.program import _layer_experts

register_tpu_ci(est_time=2, suite="stage-a-test-tpu-unit", chips=1)

_LAYERS = 62
_GEOMETRY = SimpleNamespace(
    hidden=8192,
    head_dim=64,
    q_heads=16,
    kv_heads=2,
    n_layers=_LAYERS,
    n_experts=256,
    top_k=8,
    expert_hidden=4096,
    inter_local=512,
    rms_eps=1e-5,
    post_eps=1e-8,
    scale_query_by=5.473,
    attn_scale=0.125,
    router_mult=0.1768,
    route_scale=1.0,
    router_eps=1e-15,
    window=2048,
    global_layers=tuple(range(1, _LAYERS, 4)),
)
# Layers 0 and 61 hold FP8 experts quantized at load; the rest are NVFP4.
_KINDS = ("fp8",) + ("fp4",) * (_LAYERS - 2) + ("fp8",)


def _megakernel(*, prefill=True, tp=8, qblock=0, expert_group=0, fp8_gather=False):
    return SimpleNamespace(
        geo=_GEOMETRY,
        tp_size=tp,
        kinds=_KINDS,
        max_rows=32,
        moe_mode="tp",
        prefill=prefill,
        prefill_fp8_gather=fp8_gather,
        expert_groups=ops.expert_groups_of(_KINDS, expert_group),
        w4a16_layers=ops.w4a16_layers_of(_KINDS, qblock),
    )


def _batch(mode):
    return SimpleNamespace(forward_mode=mode)


class TestMusePrefillDispatch(CustomTestCase):
    def test_extend_steps_take_the_program_only_when_enabled(self):
        on, off = _megakernel(prefill=True), _megakernel(prefill=False)
        for mode in (ForwardMode.EXTEND, ForwardMode.MIXED):
            self.assertTrue(ops.MuseMegakernel.handles(on, _batch(mode), 4096))
            self.assertFalse(ops.MuseMegakernel.handles(off, _batch(mode), 4096))

    def test_decode_and_verify_keep_the_decode_megakernel_rule(self):
        mk = _megakernel(prefill=True)
        self.assertTrue(ops.MuseMegakernel.handles(mk, _batch(ForwardMode.DECODE), 16))
        # past max_rows a verify step goes native even with the prefill program on
        verify = _batch(ForwardMode.TARGET_VERIFY)
        self.assertFalse(ops.MuseMegakernel.handles(mk, verify, 64))

    def test_extend_bucket_not_tiling_the_kernels_raises(self):
        mk = _megakernel(prefill=True)
        with self.assertRaisesRegex(ValueError, "multiples of 128"):
            ops.MuseMegakernel.handles(mk, _batch(ForwardMode.EXTEND), 1000)

    def test_idle_and_draft_extend_stay_native(self):
        mk = _megakernel(prefill=True)
        for mode in (ForwardMode.IDLE, ForwardMode.DRAFT_EXTEND_V2):
            self.assertFalse(ops.MuseMegakernel.handles(mk, _batch(mode), 4096))


class TestMusePrefillConfig(CustomTestCase):
    def test_config_takes_the_kernel_geometry_and_settings(self):
        mk = _megakernel(qblock=256, expert_group=15, fp8_gather=True)
        cfg = ops.MuseMegakernel.prefill_config(mk, rows=4096, capture=(1, 30))
        self.assertEqual(cfg.rows, 4096)
        self.assertEqual(cfg.capture_layers, (1, 30))
        self.assertEqual(cfg.global_layers, _GEOMETRY.global_layers)
        self.assertEqual(cfg.qkvg_dim, 2 * 16 * 64 + 2 * 2 * 64)
        self.assertTrue(cfg.fp8_gather)
        self.assertEqual(cfg.expert_groups, mk.expert_groups)

    def test_w4a16_only_on_requantized_fp4_layers(self):
        self.assertEqual(ops.w4a16_layers_of(_KINDS, 0), ())
        layers = ops.w4a16_layers_of(_KINDS, 256)
        self.assertEqual(layers, tuple(range(1, _LAYERS - 1)))
        cfg = ops.MuseMegakernel.prefill_config(
            _megakernel(qblock=256), rows=1024, capture=()
        )
        self.assertNotIn(0, cfg.w4a16_layers)
        self.assertNotIn(_LAYERS - 1, cfg.w4a16_layers)

    def test_uncovered_shapes_raise(self):
        with self.assertRaisesRegex(ValueError, "multiples of 128"):
            ops.MuseMegakernel.prefill_config(_megakernel(), rows=1000, capture=())
        with self.assertRaisesRegex(ValueError, "TP 8"):
            ops.MuseMegakernel.prefill_config(_megakernel(tp=4), rows=1024, capture=())

    def test_layers_map_to_their_expert_operands(self):
        cfg = ops.MuseMegakernel.prefill_config(
            _megakernel(expert_group=15), rows=1024, capture=()
        )
        units = _layer_experts(cfg)
        self.assertEqual(units[0], (0, None))
        # fp4 layers 1..15 share the first stacked group, indexed within it
        self.assertEqual(units[1], (1, 0))
        self.assertEqual(units[15], (1, 14))
        self.assertEqual(units[16], (2, 0))
        self.assertEqual(units[_LAYERS - 1][1], None)
        # fp8 layer 0, four groups of 15 fp4 layers, fp8 layer 61
        self.assertEqual(len({u for u, _ in units}), 6)


if __name__ == "__main__":
    unittest.main()
