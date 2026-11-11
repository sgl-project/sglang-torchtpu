"""Tests for the Muse megakernel's per-step config selection (host-only; needs jax)."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.muse_megakernel import kernel as K
from sglang_tpu.muse_megakernel import ops

register_tpu_ci(est_time=2, suite="stage-a-test-tpu-unit", chips=1)

_LAYERS = 62
# Muse Spark at tp8: per-device heads and expert-TP intermediate slice.
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
_FP8_SAL = {"kv_sal": True, "kv_sal_fp8": True}


def _megakernel(**flags):
    return SimpleNamespace(
        geo=_GEOMETRY,
        tp_size=8,
        kinds=_KINDS,
        max_position=262144,
        chunk_pages=4,
        kv_packed=flags.get("kv_packed", True),
        scale_fp8=False,
        expert_groups=flags.get("expert_groups", ()),
        kperm=flags.get("kperm", True),
        moe_mode="tp",
    )


def _config(mk, q, n_req, **pool):
    cfg = ops.MuseMegakernel.config(
        mk,
        rows=ops.step_rows(q * n_req, mk.tp_size, mk.moe_mode),
        q=q,
        n_req=n_req,
        n_page_entries=n_req * 2048,
        page_size=256 if pool else 128,
        capture=(1, 16, 30, 44, 59),
        **pool,
    )
    return cfg.validate()  # the kernel builder's own checks


class TestMuseMegakernelConfig(CustomTestCase):
    def test_served_shapes_fit_the_scratch_budget(self):
        # (rows per request, requests): decode and DFlash verify blocks up to the
        # 32-row cap; 8 rows of 4 is the shape whose Mosaic temporaries set the budget.
        for q, n_req in ((1, 1), (1, 8), (8, 1), (8, 2), (4, 8), (8, 4)):
            for flags, pool in (
                ({}, {}),
                ({"kv_packed": False}, {}),
                ({}, {"kv_sal": True}),
                ({}, _FP8_SAL),
            ):
                with self.subTest(q=q, n_req=n_req, **flags, **pool):
                    cfg = _config(_megakernel(**flags), q, n_req, **pool)
                    budget = (
                        ops._OPT_BASE_VMEM_BUDGET
                        if (cfg.alias_oy or cfg.alias_p or cfg.xfull_packed)
                        else ops._SCRATCH_VMEM_BUDGET
                    )
                    self.assertLessEqual(K.scratch_vmem_bytes(cfg), budget)
                    self.assertEqual(cfg.stream_depth, cfg.ring_slots - 1)
                    self.assertEqual(cfg.row_tile, 0)
                    # pipelined attention: two iterations of chunks in use, refills in flight
                    if cfg.attn_pipe:
                        self.assertGreater(cfg.kv_ring, 2 * cfg.attn_streams)

    def test_sal_pool_shape_detection(self):
        geo = _GEOMETRY
        self.assertTrue(ops.is_sal_pool_shape((4297, 4, 16, 4, 256), geo))  # fp8
        self.assertTrue(ops.is_sal_pool_shape((4297, 4, 32, 2, 256), geo))  # bf16
        self.assertFalse(ops.is_sal_pool_shape((4297, 128, 1, 2, 128), geo))  # fused

    def test_step_rows(self):
        # The TP size up to 32 rows, then the 16-row tile of the row-tiled program.
        for tokens, rows in ((1, 8), (8, 8), (20, 24), (32, 32), (33, 48), (40, 48)):
            self.assertEqual(ops.step_rows(tokens, 8, "tp"), rows)
        for tokens, rows in ((72, 80), (80, 80), (96, 96), (128, 128)):
            self.assertEqual(ops.step_rows(tokens, 8, "tp"), rows)
        # the EP layout runs every step row-tiled
        for tokens, rows in ((1, 16), (8, 16), (20, 32), (64, 64)):
            self.assertEqual(ops.step_rows(tokens, 8, "ep"), rows)

    def test_folded_verify_steps_keep_a_deep_enough_ring(self):
        # <=16 rows and 24 rows at q <= 4 fit the 10-slot ring unaliased in 48 MiB;
        # 24 rows at q >= 5 and 32-row steps turn on pad/ysend aliasing + packed xfull
        # so they also keep the full 10-slot folded ring inside 50 MiB.
        cfg_24_q4 = _config(_megakernel(), 4, 6, **_FP8_SAL)
        self.assertEqual(cfg_24_q4.ring_slots, 10)
        self.assertFalse(
            cfg_24_q4.alias_oy or cfg_24_q4.alias_p or cfg_24_q4.xfull_packed
        )
        self.assertLessEqual(K.scratch_vmem_bytes(cfg_24_q4), ops._SCRATCH_VMEM_BUDGET)
        for q, n_req in ((8, 3), (6, 4), (8, 4), (4, 8)):
            with self.subTest(q=q, n_req=n_req):
                cfg = _config(_megakernel(), q, n_req, **_FP8_SAL)
                self.assertTrue(cfg.fold_f and cfg.fold_o)
                self.assertTrue(K.fold_fits(cfg))
                self.assertEqual(cfg.ring_slots, 10)
                self.assertTrue(cfg.alias_oy and cfg.alias_p and cfg.xfull_packed)
                self.assertLessEqual(
                    K.scratch_vmem_bytes(cfg), ops._OPT_BASE_VMEM_BUDGET
                )

    def test_wave5_expert_split_swa_and_kperm0_bd_all(self):
        groups = ops.expert_groups_of(_KINDS, 15)
        cfg_24 = _config(_megakernel(expert_groups=groups), 8, 3, **_FP8_SAL)
        self.assertEqual(cfg_24.expert_split, 0)
        self.assertEqual(cfg_24.compact_arms, (1, 2, 4))
        self.assertEqual(cfg_24.swa_chunk_pages, 2)
        self.assertTrue(cfg_24.attn_skip_idle_streams and cfg_24.attn_col_mask)
        cfg_32 = _config(_megakernel(expert_groups=groups), 4, 8, **_FP8_SAL)
        self.assertEqual(cfg_32.expert_split, 4)
        self.assertEqual(cfg_32.compact_rows, 4)
        self.assertEqual(cfg_32.compact_arms, (1, 2, 4))
        self.assertTrue(cfg_32.ex_w2_stage)
        self.assertFalse(cfg_32.ex_bd_all)
        cfg_32_kperm0 = _config(
            _megakernel(expert_groups=groups, kperm=False), 4, 8, **_FP8_SAL
        )
        self.assertTrue(cfg_32_kperm0.ex_bd_all)

    def test_row_tiled_steps(self):
        # Steps past 32 rows run the row-tiled program: rows a multiple of the row tile,
        # folds enabled where they fit, scratch within the row-tiled VMEM budget.
        for groups in ((), ops.expert_groups_of(_KINDS, 15)):
            for q, n_req in ((8, 5), (8, 6), (4, 10), (8, 9), (8, 12), (8, 16)):
                with self.subTest(q=q, n_req=n_req, groups=bool(groups)):
                    cfg = _config(
                        _megakernel(expert_groups=groups), q, n_req, **_FP8_SAL
                    )
                    self.assertEqual(cfg.row_tile, 16)
                    self.assertEqual(cfg.rows % 16, 0)
                    if cfg.rows < 128:
                        self.assertTrue(cfg.fold_f and cfg.fold_o)
                    else:
                        # 48 MiB budget at 128 rows: a 3-slot ring, no stream depth for folds.
                        self.assertEqual((cfg.expert_depth, cfg.ring_slots), (3, 3))
                        self.assertFalse(cfg.fold_f or cfg.fold_o)
                    if cfg.fold_f or cfg.fold_o:
                        self.assertTrue(K.fold_fits(cfg))
                    self.assertEqual(cfg.sal_fp8_dot, 0)  # env default; recipes set 2
                    budget = (
                        ops._TILED_VY2_VMEM_BUDGET
                        if cfg.tiled_vmem_y >= 2
                        else (
                            (
                                ops._TILED_VPADS_SMALL_VMEM_BUDGET
                                if cfg.rows <= 128
                                else ops._TILED_VPADS_VMEM_BUDGET
                            )
                            if cfg.tiled_vmem_pads
                            else ops._SCRATCH_VMEM_BUDGET
                        )
                    )
                    self.assertLessEqual(K.scratch_vmem_bytes(cfg), budget)

    def test_expert_groups_of(self):
        groups = ops.expert_groups_of(_KINDS, 15)
        self.assertEqual(groups, ((1, 16), (16, 31), (31, 46), (46, 61)))
        self.assertEqual(ops.expert_groups_of(_KINDS, 0), ())


if __name__ == "__main__":
    unittest.main()
