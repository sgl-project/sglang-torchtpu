"""Tests for the Muse megakernel's per-step config selection (host-only; needs jax)."""

import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.muse_megakernel import kernel as K
from sglang_tpu.muse_megakernel import ops
from sglang_tpu.muse_megakernel.ep_mk.config import ep_mk_config

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

    def test_ep_steps_validate(self):
        # the EP kernel validates ep_mk_config's config (no VMEM Y, no single-row collectives)
        for q, n_req in ((1, 1), (8, 2), (8, 8), (8, 16)):
            for pool in ({}, _FP8_SAL, {"unified_kv": True, **_FP8_SAL}):
                with self.subTest(q=q, n_req=n_req, **pool):
                    cfg = _raw_config(_megakernel(moe_mode="ep"), q, n_req, **pool)
                    self.assertEqual(cfg.row_tile, 16)
                    ep_mk_config(cfg).validate()

    def test_expert_groups_of(self):
        groups = ops.expert_groups_of(_KINDS, 15)
        self.assertEqual(groups, ((1, 16), (16, 31), (31, 46), (46, 61)))
        self.assertEqual(ops.expert_groups_of(_KINDS, 0), ())

    @staticmethod
    @contextmanager
    def _mock_tpu7x():
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
                stack.enter_context(
                    patch.object(mod, "get_device_kind", lambda: "TPU7x")
                )
                stack.enter_context(patch.object(mod, "is_tpu_device", lambda: True))
            stack.enter_context(
                patch.object(
                    pallas_export, "get_pallas_tpu_device_kind", lambda: "TPU7x"
                )
            )
            if len(modules) == 1:
                stack.enter_context(
                    patch.object(pallas_export, "tpu_info_for_cpu_export", nullcontext)
                )
            yield

    def test_unified_kv_maps_and_pool_validation(self):
        import torch

        from sglang_tpu.attention_pallas_rpa_base import BaseTpuPallasRpaAttnBackend
        from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool, TpuSeqAlongLaneKVPool
        from sglang_tpu.unified_kv_pool import AttnLayerSpec, TpuUnifiedSWAKVPool

        gl = _GEOMETRY.global_layers
        swa_ids = [i for i in range(_LAYERS) if i not in set(gl)]
        buf_map, grp_map = ops.unified_kv_maps(_LAYERS, gl)
        self.assertEqual((max(buf_map) + 1, max(grp_map) + 1), (16, 3))

        def mk_pool(cls, dt, ps=16, fids=gl, sids=swa_ids):
            f_spec = AttnLayerSpec(2, 64, fids)
            s_spec = AttnLayerSpec(2, 64, sids)
            return TpuUnifiedSWAKVPool(
                size=2 * ps,
                page_size=ps,
                dtype=dt,
                full=f_spec,
                swa=s_spec,
                device="cpu",
                token_to_kv_pool_class=cls,
            )

        with self._mock_tpu7x():
            sal = mk_pool(TpuSeqAlongLaneKVPool, torch.bfloat16, 256)
            self.assertEqual(buf_map, tuple(sal._buffer_idx(i) for i in range(_LAYERS)))
            self.assertEqual(
                grp_map, tuple(sal.swa_page_group(i) for i in range(_LAYERS))
            )
            ops.MuseMegakernel.check_kv_pools(_megakernel(), sal)
            b = object.__new__(BaseTpuPallasRpaAttnBackend)
            b._swa_group_pages = sal.swa_group_pages
            sp = torch.tensor([1, 2], dtype=torch.int64)
            s_all = b._build_swa_all_page_indices_cpu(sp)
            self.assertTrue(torch.equal(s_all[0], sp.to(torch.int32)))
            self.assertTrue(torch.equal(s_all[1:], sal.swa_group_pages[:, sp]))
            self.assertEqual(s_all.reshape(-1).data_ptr(), s_all.data_ptr())
            ops.MuseMegakernel.check_kv_pools(
                _megakernel(), mk_pool(TpuSeqAlongLaneKVPool, torch.float8_e4m3fn, 256)
            )
            fused_u = mk_pool(TpuFusedMHATokenToKVPool, torch.bfloat16)
            ops.MuseMegakernel.check_kv_pools(_megakernel(kv_packed=False), fused_u)
            with self.assertRaises(NotImplementedError):
                ops.MuseMegakernel.check_kv_pools(_megakernel(kv_packed=True), fused_u)
            with self.assertRaises(NotImplementedError):
                ops.MuseMegakernel.check_kv_pools(
                    _megakernel(), mk_pool(TpuSeqAlongLaneKVPool, torch.float32, 256)
                )
            with self.assertRaises(ValueError):
                ops.MuseMegakernel.check_kv_pools(
                    _megakernel(),
                    mk_pool(
                        TpuSeqAlongLaneKVPool,
                        torch.bfloat16,
                        256,
                        gl[:-1],
                        [*swa_ids, gl[-1]],
                    ),
                )

    def test_unified_kv_config_tables_and_mosaic_export(self):
        import jax
        import jax.export
        import jax.numpy as jnp

        from sglang_tpu.muse_megakernel.mk.scratch import scratch_plan

        cfg = _config(_megakernel(kv_packed=False), 4, 2, unified_kv=True)
        self.assertEqual((cfg.n_kv_buffers, cfg.n_swa_groups), (16, 3))
        self.assertFalse(ops._swa_window_table(True, 40000, n_swa_groups=1))
        self.assertTrue(ops._swa_window_table(True, 40000, n_swa_groups=3))

        cfg_sal = ops.MuseMegakernel.config(
            _megakernel(),
            rows=16,
            q=8,
            n_req=2,
            n_page_entries=44020,
            page_size=256,
            capture=(1,),
            unified_kv=True,
            **_FP8_SAL,
        ).validate()
        self.assertTrue(cfg_sal.swa_window_table)
        tp_pad = K.table_entries_padded(cfg_sal)
        padded = K.pad_table(
            cfg_sal, jnp.arange(3 * cfg_sal.n_pages_entries, dtype=jnp.int32) + 1
        )
        self.assertEqual(padded.shape, (3 * tp_pad,))
        self.assertIs(K.pad_table(cfg_sal, padded), padded)
        for sg in range(3):
            self.assertEqual(int(padded[sg * tp_pad]), 1 + sg * cfg_sal.n_pages_entries)
        self.assertEqual(
            dict(scratch_plan(cfg_sal))["psw_win"].shape,
            (3 * 2 * K.swa_window_entries(cfg_sal),),
        )

        from jax.sharding import AbstractMesh, NamedSharding
        from jax.sharding import PartitionSpec as P

        mesh = AbstractMesh((8,), ("model",))
        sh = NamedSharding(mesh, P("model"))
        L = 5

        def sd(shape, dt):
            return jax.ShapeDtypeStruct((shape[0] * 8, *shape[1:]), dt, sharding=sh)

        geo = SimpleNamespace(
            **{**_GEOMETRY.__dict__, "n_layers": L, "global_layers": (1, 4)}
        )
        kinds = ("fp8",) + ("fp4",) * (L - 2) + ("fp8",)
        mk_obj = SimpleNamespace(
            geo=geo,
            tp_size=8,
            kinds=kinds,
            max_position=8192,
            chunk_pages=4,
            kv_packed=False,
            expert_groups=(),
            moe_mode="tp",
        )
        with self._mock_tpu7x():
            for n_ent, kw in (
                (66, {"unified_kv": True}),
                (22010, {"unified_kv": True, **_FP8_SAL}),
            ):
                c = ops.MuseMegakernel.config(
                    mk_obj,
                    rows=8,
                    q=4,
                    n_req=2,
                    n_page_entries=2 * n_ent,
                    page_size=256 if kw.get("kv_sal") else 128,
                    capture=(1, 4),
                    **kw,
                ).validate()
                B, qd, kd = (
                    c.n_kv_buffers,
                    c.q_heads * c.head_dim,
                    c.kv_heads * c.head_dim,
                )
                He, Il = c.expert_hidden, c.inter_local
                w = {
                    "rope": sd((c.max_position, 2 * c.head_dim), jnp.float32),
                    "final_norm": sd((8, c.hidden), jnp.float32),
                    "smalls": sd((L, K.SMALL_ROWS, c.hidden), jnp.float32),
                    "wrouter": sd((L, c.hidden, c.n_experts), jnp.bfloat16),
                    "wqkvg": sd((L, c.hidden, 2 * qd + 2 * kd), jnp.float8_e4m3fn),
                    "wo": sd((L, qd, c.hidden), jnp.float8_e4m3fn),
                    "wpool": sd((L, c.hidden, He // 8), jnp.float8_e4m3fn),
                    "wunpool": sd((L, He, c.hidden // 8), jnp.float8_e4m3fn),
                }
                ex_by_k = {
                    "fp4": [
                        sd((c.n_experts, He // 8, 2 * Il), jnp.int32),
                        sd((c.n_experts, He // 16, 2 * Il), jnp.bfloat16),
                        sd((c.n_experts, Il // 8, He), jnp.int32),
                        sd((c.n_experts, Il // 16, He), jnp.bfloat16),
                    ],
                    "fp8": [
                        sd((c.n_experts, He // 4, 2 * Il), jnp.int32),
                        sd((c.n_experts, 2, 2 * Il), jnp.bfloat16),
                        sd((c.n_experts, Il // 4, He), jnp.int32),
                        sd((c.n_experts, 2, He), jnp.bfloat16),
                    ],
                }
                kv_s = (
                    (32, 2 * c.kv_heads, c.head_dim // 4, 4, c.page_size)
                    if c.kv_sal
                    else (32, c.page_size, c.kv_heads, 2, K.LANE)
                )
                kv_dt = jnp.float8_e4m3fn if c.kv_sal_fp8 else jnp.bfloat16
                args = (
                    sd((2,), jnp.int32),
                    sd((c.rows,), jnp.int32),
                    sd((c.n_pages_entries,), jnp.int32),
                    sd((2 * c.n_pages_entries,), jnp.int32),
                    sd((c.rows, c.hidden), jnp.bfloat16),
                    *[w[n] for n in K.WEIGHT_NAMES],
                    *[t for k in kinds for t in ex_by_k[k]],
                    *[sd(kv_s, kv_dt) for _ in range(B)],
                )
                kv0, n_out = len(args) - B, 2 + B
                mk_c, kvw_c = K.megakernel_call(c), K.kv_write_call(c)

                def _body(*xs, c=c, mk_c=mk_c, kvw_c=kvw_c, kv0=kv0):
                    psw = K.pad_table(c, xs[3]) if c.swa_window_table else xs[3]
                    o = mk_c(*xs[:3], psw, *xs[4:])
                    return (*o[:2], *kvw_c(*xs[:3], psw, o[2], *xs[kv0:]))

                sharded = jax.shard_map(
                    _body,
                    mesh=mesh,
                    in_specs=(P("model"),) * len(args),
                    out_specs=(P("model"),) * n_out,
                    check_vma=False,
                )
                jit_fn = jax.jit(
                    lambda *xs, sharded=sharded: tuple(sharded(*xs)),
                    donate_argnums=tuple(range(kv0, len(args))),
                    out_shardings=(sh,) * n_out,
                )
                with jax._src.config.export_ignore_forward_compatibility(True):
                    exp = jax.export.export(jit_fn, platforms=["tpu"])(*args)
                self.assertEqual(len(exp.out_avals), n_out)


if __name__ == "__main__":
    unittest.main()
