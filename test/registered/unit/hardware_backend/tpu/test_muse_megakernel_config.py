"""Tests for the Muse megakernel's per-step config selection (host-only; needs jax)."""

import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.muse_megakernel import ops
from sglang_tpu.muse_megakernel.common.plan import fold_fits
from sglang_tpu.muse_megakernel.common.scratch import scratch_vmem_bytes
from sglang_tpu.muse_megakernel.decode.spep.config import ep_mk_config

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
        expert_groups=flags.get("expert_groups", ()),
        moe_mode=flags.get("moe_mode", "tp"),
    )


def _raw_config(mk, q, n_req, **pool):
    return ops.MuseMegakernel.config(
        mk,
        rows=ops.step_rows(q * n_req, mk.tp_size, mk.moe_mode),
        q=q,
        n_req=n_req,
        n_page_entries=n_req * 2048,
        page_size=256 if pool else 128,
        capture=(1, 16, 30, 44, 59),
        **pool,
    )


def _config(mk, q, n_req, **pool):
    return _raw_config(
        mk, q, n_req, **pool
    ).validate()  # the kernel builder's own checks


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
                    self.assertLessEqual(scratch_vmem_bytes(cfg), budget)
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
        self.assertLessEqual(scratch_vmem_bytes(cfg_24_q4), ops._SCRATCH_VMEM_BUDGET)
        for q, n_req in ((8, 3), (6, 4), (8, 4), (4, 8)):
            with self.subTest(q=q, n_req=n_req):
                cfg = _config(_megakernel(), q, n_req, **_FP8_SAL)
                self.assertTrue(cfg.fold_f and cfg.fold_o)
                self.assertTrue(fold_fits(cfg))
                self.assertEqual(cfg.ring_slots, 10)
                self.assertTrue(cfg.alias_oy and cfg.alias_p and cfg.xfull_packed)
                self.assertLessEqual(scratch_vmem_bytes(cfg), ops._OPT_BASE_VMEM_BUDGET)

    def test_wave5_expert_split_swa_and_bd_all(self):
        groups = ops.expert_groups_of(_KINDS, 15)
        cfg_24 = _config(_megakernel(expert_groups=groups), 8, 3, **_FP8_SAL)
        self.assertEqual(cfg_24.expert_split, 0)
        self.assertEqual(cfg_24.compact_arms, (1, 2, 4))
        self.assertEqual(cfg_24.swa_chunk_pages, 2)
        self.assertTrue(cfg_24.attn_skip_idle_streams and cfg_24.attn_col_mask)
        cfg_32 = _config(_megakernel(expert_groups=groups), 4, 8, **_FP8_SAL)
        self.assertEqual(cfg_32.expert_split, 4)
        self.assertEqual(cfg_32.compact_rows, 8)
        self.assertEqual(cfg_32.compact_arms, (1, 2, 4, 8))
        self.assertFalse(cfg_32.ex_w2_stage)
        self.assertTrue(cfg_32.ex_bd_all)

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
                        self.assertTrue(fold_fits(cfg))
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
                    self.assertLessEqual(scratch_vmem_bytes(cfg), budget)

    def test_ep_steps_validate(self):
        # the EP kernel validates ep_mk_config's config (no VMEM Y, no single-row collectives)
        for q, n_req in ((1, 1), (8, 2), (8, 8), (8, 16)):
            for pool in ({}, _FP8_SAL):
                with self.subTest(q=q, n_req=n_req, **pool):
                    cfg = _raw_config(_megakernel(moe_mode="ep"), q, n_req, **pool)
                    self.assertEqual(cfg.row_tile, 16)
                    ep_mk_config(cfg).validate()

    def test_expert_groups_of(self):
        groups = ops.expert_groups_of(_KINDS, 15)
        self.assertEqual(groups, ((1, 16), (16, 31), (31, 46), (46, 61)))
        self.assertEqual(ops.expert_groups_of(_KINDS, 0), ())


@contextmanager
def _mock_tpu7x():
    """Host-side pool construction asks jax for the TPU generation; answer TPU7x."""
    from jax._src.pallas.mosaic import tpu_info
    from jax.experimental.pallas import tpu as pltpu

    import sglang_tpu.pallas_export as pallas_export

    try:
        from jax._src import tpu_info as core_tpu_info

        modules = (tpu_info, core_tpu_info)
    except ImportError:
        modules = (tpu_info,)
    info = tpu_info.get_tpu_info_for_chip(
        tpu_info.chip_version_from_device_kind("TPU7x"), 1
    )
    with ExitStack() as stack:
        stack.enter_context(patch.object(pltpu, "get_tpu_info", lambda: info))
        for mod in modules:
            stack.enter_context(patch.object(mod, "get_tpu_info", lambda: info))
            stack.enter_context(patch.object(mod, "get_device_kind", lambda: "TPU7x"))
            stack.enter_context(patch.object(mod, "is_tpu_device", lambda: True))
        stack.enter_context(
            patch.object(pallas_export, "get_pallas_tpu_device_kind", lambda: "TPU7x")
        )
        if len(modules) == 1:
            stack.enter_context(
                patch.object(pallas_export, "tpu_info_for_cpu_export", nullcontext)
            )
        yield


class TestMuseMegakernelUnifiedKV(CustomTestCase):
    """TpuUnifiedSWAKVPool: the programs read the pool's shared buffers through kv_buf_map and
    the SWA groups' page tables back to back through swa_group_map."""

    def test_maps_match_the_unified_pool_layer_assignment(self):
        import torch

        from sglang_tpu.kv_pool import TpuSeqAlongLaneKVPool
        from sglang_tpu.unified_kv_pool import AttnLayerSpec, TpuUnifiedSWAKVPool

        gl = _GEOMETRY.global_layers
        swa_ids = [i for i in range(_LAYERS) if i not in set(gl)]
        buf_map, grp_map = ops.unified_kv_maps(_LAYERS, gl)
        self.assertEqual((max(buf_map) + 1, max(grp_map) + 1), (16, 3))
        with _mock_tpu7x():
            self._check_pools(gl, swa_ids, buf_map, grp_map)

    def _check_pools(self, gl, swa_ids, buf_map, grp_map):
        import torch

        from sglang_tpu.kv_pool import TpuSeqAlongLaneKVPool
        from sglang_tpu.unified_kv_pool import AttnLayerSpec, TpuUnifiedSWAKVPool

        pool = TpuUnifiedSWAKVPool(
            size=512,
            page_size=256,
            dtype=torch.float8_e4m3fn,
            full=AttnLayerSpec(2, 64, gl),
            swa=AttnLayerSpec(2, 64, swa_ids),
            device="cpu",
            token_to_kv_pool_class=TpuSeqAlongLaneKVPool,
        )
        self.assertEqual(buf_map, tuple(pool._buffer_idx(i) for i in range(_LAYERS)))
        self.assertEqual(grp_map, tuple(pool.swa_page_group(i) for i in range(_LAYERS)))
        self.assertEqual(ops._layout_pools(pool, _GEOMETRY), (pool.kv_pool,))
        # a pool whose layer assignment differs from the kernel geometry is refused
        moved = TpuUnifiedSWAKVPool(
            size=512,
            page_size=256,
            dtype=torch.float8_e4m3fn,
            full=AttnLayerSpec(2, 64, gl[:-1]),
            swa=AttnLayerSpec(2, 64, [*swa_ids, gl[-1]]),
            device="cpu",
            token_to_kv_pool_class=TpuSeqAlongLaneKVPool,
        )
        with self.assertRaises(ValueError):
            ops._layout_pools(moved, _GEOMETRY)

    def test_config_carries_the_maps_and_sizes_the_window_tables(self):
        cfg = _config(_megakernel(), 8, 2, unified_kv=True, **_FP8_SAL)
        self.assertEqual((cfg.n_kv_buffers, cfg.n_swa_groups), (16, 3))
        self.assertEqual(cfg.layer_kv_buf(0), 0)
        last_swa = max(i for i in range(_LAYERS) if i not in _GEOMETRY.global_layers)
        self.assertEqual(cfg.layer_swa_group(last_swa), 2)
        plain = _config(_megakernel(), 8, 2, **_FP8_SAL)
        self.assertEqual((plain.n_kv_buffers, plain.n_swa_groups), (_LAYERS, 1))
        # three SWA tables plus the full table outgrow SMEM sooner than one SWA table
        self.assertFalse(ops._swa_window_table(True, 40000, n_swa_groups=1))
        self.assertTrue(ops._swa_window_table(True, 40000, n_swa_groups=3))

    def test_swa_tables_concatenate_the_groups(self):
        import torch

        base = torch.tensor([3, 4], dtype=torch.int32)
        groups = torch.tensor([[7, 8], [9, 10]], dtype=torch.int32)
        md = SimpleNamespace(
            page_indices=None, swa_page_indices=base, swa_group_page_indices=groups
        )
        self.assertEqual(ops._swa_tables(md).tolist(), [3, 4, 7, 8, 9, 10])
        md.swa_group_page_indices = None
        self.assertIs(ops._swa_tables(md), base)

    def test_ep_decode_refuses_the_unified_pool(self):
        with self.assertRaises(NotImplementedError):
            ops.MuseMegakernel.check_kv_pools(
                SimpleNamespace(moe_mode="ep", geo=_GEOMETRY, kv_packed=False),
                object.__new__(ops.TpuUnifiedSWAKVPool),
            )


if __name__ == "__main__":
    unittest.main()
