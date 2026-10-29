"""Compare Raiden reshard byte ranges with the actual fused KV pool layout."""

import ast
import itertools
import pathlib
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu import kv_pool
from sglang_tpu.disaggregation.raiden import reshard_schedule as rs
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager
from sglang_tpu.disaggregation.raiden.reshard_schedule import (
    KVLayoutGeometry,
    LandRequired,
    ReshardGeometry,
)
from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool, TpuSeqAlongLaneKVPool

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE = 64
# SEQ_ALONG_LANE pages are a multiple of the 128-lane width (batched RPA
# floor 256).
SAL_PAGE = 256


def _sublane_tile_head_dim(*, head_dim, kv_packing):
    """kv_pool._seq_along_lane_head_dim at 8 sublanes; the real query needs
    JAX, which the CPU suite lacks."""
    return _cdiv(head_dim, _SUBLANES * kv_packing) * _SUBLANES * kv_packing


def _layer(dtype, heads, head_dim=128, sal=False):
    if sal:
        with mock.patch.object(
            kv_pool, "_seq_along_lane_head_dim", _sublane_tile_head_dim
        ):
            return _new_layer(TpuSeqAlongLaneKVPool, dtype, heads, head_dim, SAL_PAGE)
    return _new_layer(TpuFusedMHATokenToKVPool, dtype, heads, head_dim, PAGE)


def _new_layer(cls, dtype, heads, head_dim, page):
    pool = cls(
        size=2 * page,
        page_size=page,
        dtype=dtype,
        head_num=heads,
        head_dim=head_dim,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    return pool.get_fused_kv_buffer(0)


def _rank_heads(total, tp, rank):
    """Global kv-heads an attention-TP rank holds; replicated below tp."""
    if total >= tp:
        per_rank = total // tp
        return list(range(rank * per_rank, (rank + 1) * per_rank))
    return [rank // (tp // total)]


def _manager(dtype, full_heads, swa_heads, tp, sal=False):
    """A RaidenKVManager holding one full and one SWA layer for an attention-TP
    rank of a model with the given global kv-head counts."""
    mgr = object.__new__(RaidenKVManager)
    mgr.attn_tp_size = tp
    mgr._seq_along_lane = sal
    mgr._kv_caches = [_layer(dtype, max(1, full_heads // tp), sal=sal)]
    mgr._swa_kv_caches = [_layer(dtype, max(1, swa_heads // tp), sal=sal)]
    mgr.kv_args = SimpleNamespace(
        total_kv_head_num=full_heads,
        total_swa_kv_head_num=swa_heads,
        page_size=SAL_PAGE if sal else PAGE,
    )
    return mgr


def _fill(layer, heads, sal=False):
    """Tag every live byte of page 0 by (global head, k|v, token, byte) and
    return (page bytes, live mask). The fused layout puts K and V of local
    head i at kv-slots 2i, 2i+1, packed kv_packing slots per 32-bit word;
    SEQ_ALONG_LANE puts them at rows 2i, 2i+1, each spanning the page."""
    packing = int(layer.shape[3])
    raw = layer.view(torch.uint8)
    raw.fill_(0xFF)
    live = torch.zeros_like(raw, dtype=torch.bool)
    if sal:
        w, r, b = (torch.arange(n) for n in raw.shape[2:])
        pos = w[:, None, None] * 5 + r[None, :, None] * 3 + b[None, None, :] * 7
        for local, head in enumerate(heads):
            for kv in (0, 1):
                tag = ((2 * head + kv) * 131 + pos) % 251
                raw[0, 2 * local + kv] = tag.to(torch.uint8)
                live[0, 2 * local + kv] = True
        return raw[0].flatten(), live[0].flatten()
    token = torch.arange(PAGE)[:, None]
    byte = torch.arange(raw.shape[-1])[None, :]
    for local, head in enumerate(heads):
        for kv in (0, 1):
            slot = 2 * local + kv
            tag = ((2 * head + kv) * 131 + token * 7 + byte * 3) % 251
            raw[0, :, slot // packing, slot % packing, :] = tag.to(torch.uint8)
            live[0, :, slot // packing, slot % packing, :] = True
    return raw[0].flatten(), live[0].flatten()


def _replay(
    full_heads, swa_heads, src_tp, dst_tp, *, dtype, swa, sal=False, land=False
):
    """Replay a push reshard byte for byte over real pool pages, each side
    building its geometry with _make_geometry. True when every decode rank
    holds exactly its own heads with no out-of-bounds copy; LandRequired when
    no whole-granule schedule exists. With ``land``, LandEntry contributions
    write a scratch page that land_merge folds into the final page."""
    total = swa_heads if swa else full_heads
    sender = _manager(dtype, full_heads, swa_heads, src_tp, sal=sal)
    receiver = _manager(dtype, full_heads, swa_heads, dst_tp, sal=sal)
    page = SAL_PAGE if sal else PAGE

    def layer(mgr):
        return (mgr._swa_kv_caches if swa else mgr._kv_caches)[0]

    send_geom = sender._make_geometry(src_tp, dst_tp, swa=swa)
    recv_geom = receiver._make_geometry(src_tp, dst_tp, swa=swa)
    if land:
        send_geom, recv_geom = send_geom.with_land(), recv_geom.with_land()
    for dst_rank in range(dst_tp):
        want, live = _fill(
            layer(receiver).clone(), _rank_heads(total, dst_tp, dst_rank), sal
        )
        got = torch.full_like(want, 0xFF)
        scratch = torch.full_like(want, 0xFF)
        landed = False
        ratio = send_geom.ratio
        for src_rank in range(dst_rank * ratio, (dst_rank + 1) * ratio):
            src, _ = _fill(
                layer(sender).clone(), _rank_heads(total, src_tp, src_rank), sal
            )
            sent = send_geom.entry_bytes(src_rank, dst_rank)
            recv = recv_geom.entry_bytes(src_rank, dst_rank)
            if isinstance(sent, rs.LandEntry) != isinstance(recv, rs.LandEntry):
                return False
            src_off, size, src_stride, _, _ = sent
            _, _, _, dst_off, dst_stride = recv
            out = scratch if isinstance(recv, rs.LandEntry) else got
            landed |= out is scratch
            count = send_geom.count_for(page)
            if recv_geom.count_for(page) != count:
                return False
            for t in range(count):
                a, b = src_off + t * src_stride, dst_off + t * dst_stride
                if a + size > src.numel() or b + size > out.numel():
                    return False
                out[b : b + size] = src[a : a + size]
        if landed:
            shape = layer(receiver).shape[1:]
            rs.land_merge(got.view(shape), scratch.view(shape), recv_geom.layout)
        if not torch.equal(got[live], want[live]):
            return False
    return True


class TestTpuReshardLayout(unittest.TestCase):
    def test_bf16_writer_ranges_cover_each_destination_token(self):
        for total_heads in (4, 8, 16, 32):
            for swa in (False, True):
                with self.subTest(total_heads=total_heads, swa=swa):
                    self._check_layout(torch.bfloat16, total_heads, swa=swa)

    def _check_layout(self, dtype, total_heads, *, swa=False):
        src_tp, dst_tp = 8, 4
        source = _layer(dtype, max(1, total_heads // src_tp))
        destination = _layer(dtype, max(1, total_heads // dst_tp))
        mgr = object.__new__(RaidenKVManager)
        mgr.attn_tp_rank = 0
        mgr.attn_tp_size = dst_tp
        mgr._kv_caches = [destination]
        mgr._swa_kv_caches = [destination] if swa else []
        mgr.kv_args = SimpleNamespace(
            total_kv_head_num=total_heads, total_swa_kv_head_num=total_heads
        )
        geometry = mgr._make_geometry(src_tp, dst_tp, swa=swa)
        source_width = source.stride(1) * source.element_size()
        destination_width = destination.stride(1) * destination.element_size()
        for dst_rank in range(dst_tp):
            written = set()
            for src_rank in range(dst_rank * 2, dst_rank * 2 + 2):
                src_off, size, src_stride, dst_off, dst_stride = geometry.entry_bytes(
                    src_rank, dst_rank
                )
                self.assertEqual(src_stride, source_width)
                self.assertEqual(dst_stride, destination_width)
                self.assertLessEqual(src_off + size, source_width)
                self.assertLessEqual(dst_off + size, destination_width)
                written.update(range(dst_off, dst_off + size))
            self.assertEqual(written, set(range(destination_width)))

    def test_bf16_swa_lands_each_head_with_its_own_head_count(self):
        """Every decode rank ends up holding exactly its own SWA (and full)
        heads. The SWA head count differs from the full one on MiMo-V2 (4 vs 8)
        and Gemma-4 (4 vs 16)."""
        for full_heads, swa_heads in ((4, 4), (8, 8), (4, 8), (4, 16), (8, 4)):
            for src_tp, dst_tp in ((8, 4), (8, 2), (4, 2), (16, 8)):
                for swa in (False, True):
                    with self.subTest(
                        full=full_heads, swa_heads=swa_heads, tp=(src_tp, dst_tp)
                    ):
                        mgr = _manager(torch.bfloat16, full_heads, swa_heads, dst_tp)
                        self.assertEqual(
                            mgr._make_geometry(src_tp, dst_tp, swa=swa).total_kv_heads,
                            swa_heads if swa else full_heads,
                        )
                        self.assertTrue(
                            _replay(
                                full_heads,
                                swa_heads,
                                src_tp,
                                dst_tp,
                                dtype=torch.bfloat16,
                                swa=swa,
                            )
                        )

    def test_equal_head_counts_keep_the_full_attention_geometry(self):
        """GPT-OSS-style models (SWA count == full count) get the same SWA
        schedule as full attention."""
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            for heads, (src_tp, dst_tp) in ((8, (8, 4)), (8, (16, 8)), (4, (8, 2))):
                with self.subTest(dtype=dtype, heads=heads, tp=(src_tp, dst_tp)):
                    mgr = _manager(dtype, heads, heads, dst_tp)
                    full = mgr._make_geometry(src_tp, dst_tp)
                    swa = mgr._make_geometry(src_tp, dst_tp, swa=True)
                    for src_rank in range(src_tp):
                        dst_rank = src_rank // swa.ratio
                        try:
                            want = full.entry_bytes(src_rank, dst_rank)
                        except LandRequired:
                            with self.assertRaises(LandRequired):
                                swa.entry_bytes(src_rank, dst_rank)
                            continue
                        self.assertEqual(swa.entry_bytes(src_rank, dst_rank), want)

    def test_fp8_swa_geometry_uses_the_swa_head_count(self):
        """At fp8 a granule spans two kv-heads (#829). The SWA geometry takes
        the SWA count, and every whole-granule schedule it yields replays byte
        for byte; the rest need a landing buffer. The cells that were correct
        only by coincidence of the old head-pair unit stay correct."""
        fp8 = torch.float8_e4m3fn
        correct = set()
        for full_heads, swa_heads in itertools.permutations((2, 4, 8, 16), 2):
            for src_tp, dst_tp in (
                (2, 1),
                (4, 2),
                (4, 1),
                (8, 4),
                (8, 2),
                (16, 8),
                (16, 4),
            ):
                cell = (full_heads, swa_heads, src_tp, dst_tp)
                with self.subTest(cell=cell):
                    mgr = _manager(fp8, full_heads, swa_heads, dst_tp)
                    geom = mgr._make_geometry(src_tp, dst_tp, swa=True)
                    self.assertEqual(geom.total_kv_heads, swa_heads)
                    try:
                        exact = _replay(*cell, dtype=fp8, swa=True)
                    except LandRequired:
                        continue
                    self.assertTrue(exact)
                    correct.add(cell)
        self.assertLessEqual(
            {
                (2, 4, 2, 1),
                (4, 8, 2, 1),
                (4, 8, 4, 2),
                (4, 8, 4, 1),
                (8, 16, 2, 1),
                (8, 16, 4, 2),
                (8, 16, 4, 1),
                (8, 16, 8, 4),
                (8, 16, 8, 2),
            },
            correct,
        )
        # Full-attention geometry ignores the SWA count.
        mgr = _manager(fp8, 4, 8, 4)
        self.assertEqual(mgr._make_geometry(8, 4).total_kv_heads, 4)


class TestKVLayoutGeometry(unittest.TestCase):
    """The layout geometry against real pool tensors' strides."""

    def test_token_major_matches_pool_strides(self):
        for dtype, heads, head_dim in itertools.product(
            (torch.bfloat16, _FP8), (1, 2, 4, 8), (64, 128)
        ):
            with self.subTest(dtype=dtype, heads=heads, head_dim=head_dim):
                t = _layer(dtype, heads, head_dim)
                es = t.element_size()
                g = KVLayoutGeometry.from_tensor(
                    t.shape, es, heads, seq_along_lane=False
                )
                self.assertEqual(g.granule_bytes, t.stride(2) * es)
                self.assertEqual(g.row_bytes, t.stride(1) * es)
                self.assertEqual(g.block_bytes, t.stride(0) * es)
                self.assertEqual(g.granules_per_token, t.shape[2])
                self.assertEqual(g.slots_per_granule, t.shape[3])
                # B1: an fp8 head is half a granule, a bf16 head a whole one.
                self.assertEqual(g.heads_per_granule, 2 if dtype == _FP8 else 1)
                self.assertEqual(
                    g.granule_bytes / g.heads_per_granule, 2 * t.shape[4] * es
                )
                self.assertEqual(g.head_bytes, 2 * t.shape[4] * es)

    def test_fp8_head_count_is_not_read_from_the_shape(self):
        # B4: one and two fp8 heads share a shape; the count is the caller's.
        one, two = _layer(_FP8, 1), _layer(_FP8, 2)
        self.assertEqual(one.shape, two.shape)
        for heads in (1, 2):
            g = KVLayoutGeometry.from_tensor(one.shape, 1, heads, seq_along_lane=False)
            self.assertEqual(g.heads_per_rank, heads)
        with self.assertRaisesRegex(ValueError, "does not hold 4 kv heads"):
            KVLayoutGeometry.from_tensor(one.shape, 1, 4, seq_along_lane=False)

    def test_seq_along_lane_matches_pool_strides(self):
        """A head's K and V rows are adjacent and each spans the page, so a
        head is one contiguous run of 2 * head_dim * page bytes."""
        for dtype, heads, head_dim in itertools.product(
            (torch.bfloat16, _FP8), (1, 2, 4, 8), (64, 128, 256)
        ):
            with self.subTest(dtype=dtype, heads=heads, head_dim=head_dim):
                t = _layer(dtype, heads, head_dim, sal=True)
                es = t.element_size()
                g = KVLayoutGeometry.from_tensor(
                    t.shape, es, heads, seq_along_lane=True, page_size=SAL_PAGE
                )
                self.assertEqual(g.block_bytes, t.stride(0) * es)
                self.assertEqual(g.head_run_bytes, 2 * t.stride(1) * es)
                self.assertEqual(g.head_run_bytes, 2 * head_dim * es * SAL_PAGE)
                self.assertEqual(g.copy_unit_bytes, g.head_run_bytes)
                for name in ("granule_bytes", "row_bytes", "heads_per_granule"):
                    with self.assertRaises(ValueError):
                        getattr(g, name)

    def test_seq_along_lane_head_run_entries(self):
        g = KVLayoutGeometry.from_tensor(
            _shape("HND", 2, 1, 128, 256), 2, 1, seq_along_lane=True, page_size=256
        )
        run = 2 * 128 * 2 * 256
        # 8 kv heads at TP8 -> TP4: source rank 1 is decode rank 0's head 1.
        geom = ReshardGeometry.from_layout(8, 4, g, 8)
        self.assertEqual(geom.entry_bytes(1, 0), (0, run, run, run, 2 * run))
        self.assertEqual(geom.count_for(256), 1)
        # 4 kv heads at TP8 -> TP2: replicated sources, two heads per decode
        # rank; only the first replica of each head contributes.
        geom = ReshardGeometry.from_layout(8, 2, g, 4)
        self.assertEqual(geom.entry_bytes(2, 0), (0, run, run, run, 2 * run))
        with self.assertRaisesRegex(ValueError, "do not fit"):
            rs.head_run_copy(g, g, 0, 2, 0)
        token_major = KVLayoutGeometry.from_tensor(
            _shape("NHD", 2, 1, 128, 64), 2, 1, seq_along_lane=False
        )
        with self.assertRaisesRegex(ValueError, "differ in layout"):
            rs.head_run_copy(g, token_major, 0, 1, 0)

    def test_direct_and_land_contributions(self):
        fp8 = KVLayoutGeometry.from_tensor(
            _shape("NHD", 1, 2, 128, 64), 1, 2, seq_along_lane=False
        )
        # fp8 2 -> 4 heads per rank: prefill rank 1 fills granule 1.
        geom = ReshardGeometry.from_layout(8, 4, fp8, 16)
        self.assertEqual(geom.entry_bytes(1, 0), (0, 512, 512, 512, 1024))
        # fp8 1 -> 1 head per rank (4 kv heads at TP8 -> TP4): one padded
        # granule each, as before.
        geom = ReshardGeometry.from_layout(8, 4, fp8, 4)
        self.assertEqual(geom.entry_bytes(1, 0), (0, 512, 512, 0, 512))
        # fp8 1 -> 2 heads per rank (8 kv heads at TP8 -> TP4): the second
        # source's head starts mid-granule, and the first source's padding
        # slots would cover it.
        geom = ReshardGeometry.from_layout(8, 4, fp8, 8)
        for src_rank in (0, 1):
            with self.assertRaises(LandRequired):
                geom.entry_bytes(src_rank, 0)
        # With landing, the second source lands its granule in scratch
        # granule 0 and the first copies directly: its padding sits in final
        # slots 2-3, which the merge overwrites.
        land = ReshardGeometry.from_layout(8, 4, fp8, 8, land=True)
        direct, landed = land.entry_bytes(0, 0), land.entry_bytes(1, 0)
        self.assertNotIsInstance(direct, rs.LandEntry)
        self.assertEqual(direct, (0, 512, 512, 0, 512))
        self.assertEqual(landed, rs.LandEntry(0, 512, 512, 0, 512))
        self.assertTrue(rs.land_plan_exact(land))
        # 1 -> 4 (16 kv heads at TP16 -> TP4): head j lands in granule j // 2.
        land = ReshardGeometry.from_layout(16, 4, fp8, 16, land=True)
        self.assertEqual(
            [land.entry_bytes(p, 0) for p in range(4)],
            [(0, 512, 512, 0, 1024), rs.LandEntry(0, 512, 512, 0, 1024)]
            + [(0, 512, 512, 512, 1024), rs.LandEntry(0, 512, 512, 512, 1024)],
        )
        self.assertTrue(rs.land_plan_exact(land))
        # Landing does not change a cell that needs none.
        for heads in (4, 16):
            plain = ReshardGeometry.from_layout(8, 4, fp8, heads)
            land = ReshardGeometry.from_layout(8, 4, fp8, heads, land=True)
            self.assertEqual(
                rs._typed(rs._schedule(land)), rs._typed(rs._schedule(plain))
            )

    def test_land_scope(self):
        fp8 = KVLayoutGeometry(rs.TOKEN_MAJOR, 4, 1, 1, 128, 64)
        # 3-head sources (24 kv heads at TP8 -> TP4) still need more than the
        # merge's one slot move.
        land = ReshardGeometry.from_layout(8, 4, fp8.with_heads(3), 24, land=True)
        for src_rank in (0, 1):
            with self.assertRaises(LandRequired):
                land.entry_bytes(src_rank, 0)
        self.assertFalse(rs.land_plan_exact(land))
        # An odd decode head count leaves the last granule's slots 2-3 with
        # no landed head (6 kv heads at TP6 -> TP2).
        self.assertFalse(
            rs.land_plan_exact(ReshardGeometry.from_layout(6, 2, fp8, 6, land=True))
        )
        # bf16 and SEQ_ALONG_LANE never land.
        bf16 = KVLayoutGeometry(rs.TOKEN_MAJOR, 2, 2, 1, 128, 64)
        self.assertFalse(
            rs.land_plan_exact(ReshardGeometry.from_layout(8, 4, bf16, 8, land=True))
        )
        with self.assertRaises(ValueError):
            rs.land_merge_slots(bf16)
        self.assertEqual(rs.land_merge_slots(fp8), (slice(2, 4), slice(0, 2)))

    def test_land_merge_moves_one_slot_pair_per_granule(self):
        fp8 = KVLayoutGeometry(rs.TOKEN_MAJOR, 4, 1, 2, 128, 64)
        final = np.arange(3 * 64 * 1 * 4 * 128).reshape(3, 64, 1, 4, 128)
        scratch = -final
        merged = rs.land_merge(final.copy(), scratch, fp8)
        np.testing.assert_array_equal(merged[..., :2, :], final[..., :2, :])
        np.testing.assert_array_equal(merged[..., 2:, :], scratch[..., :2, :])
        pool = torch.arange(2 * 64 * 4 * 128, dtype=torch.int32).to(torch.uint8)
        pool = pool.view(2, 64, 1, 4, 128)
        want = pool.clone()
        want[0, :, :, 2:4] = pool[1, :, :, 0:2]
        rs.land_merge(pool[0], pool[1].clone(), fp8)
        self.assertTrue(torch.equal(pool, want))

    def test_rpa_batched_page_floor(self):
        """The SEQ_ALONG_LANE cells allowed here assume pages are a multiple of
        the 128-lane width and at least 128 (the HND padding audit); the
        batched RPA backend enforces its floor at server-arg resolution."""
        path = pathlib.Path(rs.__file__).parents[2] / "attention_pallas_rpa_batched.py"
        tree = ast.parse(path.read_text())
        floor = next(
            node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and [getattr(t, "id", None) for t in node.targets] == ["_MIN_PAGE_SIZE"]
        )
        self.assertGreaterEqual(floor, 128)
        self.assertEqual(floor % 128, 0)
        spec = next(
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and [getattr(t, "id", None) for t in node.targets] == ["RPA_BATCHED_SPEC"]
        )
        kw = {k.arg: k.value for k in spec.keywords}
        self.assertEqual(getattr(kw["min_page_size"], "id", None), "_MIN_PAGE_SIZE")


# ---- push-reshard refusal --------------------------------------------------

_FP8 = torch.float8_e4m3fn
_SUBLANES = 8


class _Tensor:
    def __init__(self, shape, itemsize):
        self.shape = tuple(shape)
        self._itemsize = itemsize

    def element_size(self):
        return self._itemsize


def _cdiv(a, b):
    return -(-a // b)


def _shape(layout, itemsize, heads, head_dim, page):
    pk = 4 // itemsize
    if layout == "NHD":
        return (2, page, _cdiv(2 * heads, pk), pk, _cdiv(head_dim, 128) * 128)
    lanes = _cdiv(head_dim, _SUBLANES * pk) * _SUBLANES * pk
    return (2, 2 * heads, lanes // pk, pk, page)


def _refusal_manager(
    tensors, heads, own_tp, *, total_heads, page_size=64, side="decode"
):
    mgr = object.__new__(RaidenKVManager)
    mgr.attn_tp_size = own_tp
    mgr.kv_args = SimpleNamespace(total_kv_head_num=total_heads, page_size=page_size)
    mgr.disaggregation_mode = (
        DisaggregationMode.DECODE if side == "decode" else DisaggregationMode.PREFILL
    )
    mgr._kv_caches = list(tensors)
    mgr._kv_head_nums = tuple(heads)
    mgr._swa_kv_caches = []
    mgr._seq_along_lane = False
    return mgr


def _allowed(layout, itemsize, src_tp, dst_tp, held, geom_heads, head_dim, page):
    """Both sides' verdict for one KV layer holding ``held`` global heads, pushed
    with a geometry built for ``geom_heads`` (they differ for draft and, before
    #1076, sliding-window layers)."""
    for own_tp, side in ((dst_tp, "decode"), (src_tp, "prefill")):
        heads = max(1, held // own_tp)
        mgr = _refusal_manager(
            [_Tensor(_shape(layout, itemsize, heads, head_dim, page), itemsize)],
            [heads],
            own_tp,
            total_heads=geom_heads,
            page_size=page,
            side=side,
        )
        mgr._seq_along_lane = layout == "HND"
        if mgr.reshard_refusal(src_tp, dst_tp) is not None:
            return False
    return True


def _land_allowed(layout, itemsize, src_tp, dst_tp, held, geom_heads, head_dim, page):
    """Both sides' push_reshard_refusal with land_ok, for a layer holding
    ``held`` global heads under a geometry built for ``geom_heads``."""
    for own_tp in (dst_tp, src_tp):
        heads = max(1, held // own_tp)
        t = _Tensor(_shape(layout, itemsize, heads, head_dim, page), itemsize)
        geom = _side_geometry(layout, t, heads, src_tp, dst_tp, geom_heads, page)
        reason = rs.push_reshard_refusal(
            geom,
            t.shape,
            itemsize,
            total_heads=held,
            own_tp=own_tp,
            page_size=page,
            seq_along_lane=layout == "HND",
            land_ok=True,
        )
        if reason is not None:
            return False
    return True


def _global_heads(rank, tp, total):
    if total >= tp:
        return [rank * (total // tp) + i for i in range(total // tp)]
    return [rank // (tp // total)]


def _labels(layout, itemsize, heads, ghead, head_dim, page):
    """Byte labels of one block in the CPU (logical) layout; 0 marks padding."""
    shape = _shape(layout, itemsize, heads, head_dim, page)
    pk = shape[3]
    if layout == "NHD":
        t, row, r, c = np.meshgrid(*(np.arange(n) for n in shape[1:]), indexing="ij")
        slot = row * pk + r
    else:
        slot, w, r, t = np.meshgrid(*(np.arange(n) for n in shape[1:]), indexing="ij")
        c = w * pk + r
    valid = (slot < 2 * heads) & (c < head_dim)
    g = np.take(np.array(ghead + [0]), np.minimum(slot // 2, heads))
    elem = np.arange(valid.size).reshape(valid.shape)
    out = np.zeros(valid.size * itemsize, dtype=np.int64)
    for b in range(itemsize):
        label = (((g * 2 + slot % 2) * 512 + c) * 2048 + t) * 4 + b + 1
        out[(elem * itemsize + b)[valid]] = label[valid]
    return out


def _chunks(offset, size, stride, count):
    return (offset + stride * np.arange(count)[:, None] + np.arange(size)).ravel()


def _side_geometry(layout, t, heads, src_tp, dst_tp, geom_heads, page):
    """The geometry _make_geometry builds from a side's tensor ``t``."""
    lay = KVLayoutGeometry.from_tensor(
        t.shape,
        t.element_size(),
        heads,
        seq_along_lane=layout == "HND",
        page_size=page,
    )
    return rs.ReshardGeometry.from_layout(src_tp, dst_tp, lay, geom_heads)


class _Tampered:
    """A geometry whose entries pass through ``tamper``."""

    def __init__(self, geom, tamper):
        self._geom, self._tamper = geom, tamper
        self.src_tp, self.dst_tp, self.ratio = geom.src_tp, geom.dst_tp, geom.ratio

    def entry_bytes(self, p, d):
        return self._tamper(self._geom.entry_bytes(p, d))

    def count_for(self, live_tokens):
        return self._geom.count_for(live_tokens)


def _push_correct(
    layout,
    itemsize,
    src_tp,
    dst_tp,
    held,
    geom_heads,
    head_dim,
    page,
    tamper=None,
    land=False,
    merge=True,
    scratch_pairs=None,
):
    """Replay the legacy push byte for byte: the sender sends at most its block
    of the entry's chunks, the receiver writes exactly what was sent. False
    when no whole-granule schedule exists. With ``land`` both sides classify
    landing contributions, which go to a scratch block merged afterwards
    (unless ``merge`` is False)."""
    s_h, d_h = max(1, held // src_tp), max(1, held // dst_tp)
    src_t = _Tensor(_shape(layout, itemsize, s_h, head_dim, page), itemsize)
    dst_t = _Tensor(_shape(layout, itemsize, d_h, head_dim, page), itemsize)
    gsrc = _side_geometry(layout, src_t, s_h, src_tp, dst_tp, geom_heads, page)
    gdst = _side_geometry(layout, dst_t, d_h, src_tp, dst_tp, geom_heads, page)
    if land:
        gsrc, gdst = gsrc.with_land(), gdst.with_land()
    if tamper is not None:
        gsrc, gdst = _Tampered(gsrc, tamper), _Tampered(gdst, tamper)
    try:
        return _replay_requests(
            layout,
            itemsize,
            src_tp,
            dst_tp,
            held,
            head_dim,
            page,
            gsrc,
            gdst,
            merge=merge,
            scratch_pairs=scratch_pairs,
        )
    except LandRequired:
        return False


# Decode block ids per page: finals ascend, scratch descends, so pairing a
# final with the wrong page's scratch block cannot pass by coincidence.
_FINAL, _SCRATCH = (3, 9), (11, 7)
_PAGE_LABEL = 1 << 40  # added to page i's nonzero labels, above any byte label


def _paged(labels, i):
    return np.where(labels != 0, labels + i * _PAGE_LABEL, 0)


def _replay_requests(
    layout,
    itemsize,
    src_tp,
    dst_tp,
    held,
    head_dim,
    page,
    gsrc,
    gdst,
    merge=True,
    scratch_pairs=None,
):
    """Replay built requests over len(_FINAL) pages: entries write the block
    their dst_block_id names, read the source page their src_block_id names,
    and a plan that wrote scratch is merged page by page with land_merge (the
    device merge's CPU stand-in), final i with scratch_pairs[i] (default
    _SCRATCH[i], the pairing land_prepass uses)."""
    pages = len(_FINAL)
    scratch_pairs = scratch_pairs or _SCRATCH
    ratio = rs.reshard_ratio(src_tp, dst_tp)
    s_h, d_h = max(1, held // src_tp), max(1, held // dst_tp)
    src_t = _Tensor(_shape(layout, itemsize, s_h, head_dim, page), itemsize)
    src_block = int(np.prod(src_t.shape[1:])) * itemsize
    dst_shape = _shape(layout, itemsize, d_h, head_dim, page)
    for d in range(dst_tp):
        base = _labels(
            layout, itemsize, d_h, _global_heads(d, dst_tp, held), head_dim, page
        )
        size = base.size
        blocks = {b: np.zeros(size * 8, np.int64) for b in _FINAL + _SCRATCH}
        srcs = rs.source_ranks_for(d, ratio)
        rreq = rs.build_receiver_request(
            _Pb2, uuid=1, req_id="x", peer="d", decode_rank=d, geom=gdst,
            sources=[(p, list(range(pages))) for p in srcs],
            dst_block_ids=list(_FINAL), live_tokens=[page] * pages,
            scratch_block_ids=list(_SCRATCH),
        )  # fmt: skip
        for p in srcs:
            labels = _labels(
                layout, itemsize, s_h, _global_heads(p, src_tp, held), head_dim, page
            )
            sent = [
                np.concatenate([_paged(labels, i), np.full(labels.size * 8, -1)])
                for i in range(pages)
            ]
            sreq = rs.build_sender_request(
                _Pb2, uuid=1, req_id="x", peer="d", prefill_rank=p, decode_rank=d,
                geom=gsrc, src_block_ids=list(range(pages)),
                dst_block_ids=list(_FINAL), live_tokens=[page] * pages,
                scratch_block_ids=list(_SCRATCH),
            )  # fmt: skip
            sent_entries = sreq.shard_push_schedules[0].entries
            recv_entries = rreq.shard_push_schedules[p].entries
            # tpu-sync pairs a pushed chunk with the receiver entry of the same
            # (dst block, src block).
            if [(e.dst_block_id, e.src_block_id) for e in sent_entries] != [
                (e.dst_block_id, e.src_block_id) for e in recv_entries
            ]:
                return False
            for se, re in zip(sent_entries, recv_entries):
                sidx = _chunks(
                    se.src_offset_bytes, se.size_bytes, se.src_stride_bytes, se.count
                )[:src_block]
                didx = _chunks(
                    re.dst_offset_bytes, re.size_bytes, re.dst_stride_bytes, re.count
                )
                if didx.size < sidx.size:
                    return False
                blocks[re.dst_block_id][didx[: sidx.size]] = sent[se.src_block_id][sidx]
        if any(blocks[b][size:].any() for b in blocks):
            return False
        landed = any(
            e.dst_block_id in _SCRATCH
            for s in rreq.shard_push_schedules.values()
            for e in s.entries
        )
        for i, (f, s) in enumerate(zip(_FINAL, scratch_pairs)):
            got = blocks[f]
            if landed and merge:
                rs.land_merge(
                    got[:size].reshape(dst_shape[1:]),
                    blocks[s][:size].reshape(dst_shape[1:]),
                    gdst.layout,
                )
            expect = _paged(base, i)
            mask = expect != 0
            if not np.array_equal(got[:size][mask], expect[mask]):
                return False
    return True


class _Entries(list):
    def add(self):
        entry = SimpleNamespace()
        self.append(entry)
        return entry


class _Schedules(dict):
    def __missing__(self, key):
        self[key] = SimpleNamespace(entries=_Entries())
        return self[key]


class _Pb2:
    MEMORY_TYPE_DRAM = 1
    MEMORY_TYPE_HBM = 2

    @staticmethod
    def StartTransferRequest():
        return SimpleNamespace(shard_push_schedules=_Schedules())


class TestPushReshardRefusal(unittest.TestCase):
    def _decode(self, tensors, heads, *, total_heads, src_tp=8, dst_tp=4, page=64):
        mgr = _refusal_manager(
            tensors, heads, dst_tp, total_heads=total_heads, page_size=page
        )
        return mgr.reshard_refusal(src_tp, dst_tp)

    def test_real_pool_matrix(self):
        bf16_2 = _layer(torch.bfloat16, 2)
        fp8_1, fp8_2 = _layer(_FP8, 1), _layer(_FP8, 2)
        # A fp8 tensor's shape cannot tell one head from two.
        self.assertEqual(fp8_1.shape, fp8_2.shape)
        self.assertIsNone(self._decode([bf16_2], [2], total_heads=8))
        self.assertIsNone(self._decode([fp8_1], [1], total_heads=4))
        self.assertIn("fp8", self._decode([fp8_2], [2], total_heads=8))
        # 2 -> 4 fp8 heads per rank: each source fills whole granules.
        fp8_4 = _layer(_FP8, 4)
        self.assertIsNone(self._decode([fp8_4], [4], total_heads=16))
        # 1 -> 2 fp8 heads per rank lands; the managers still refuse it.
        mgr = _refusal_manager([fp8_2], [2], 4, total_heads=8)
        kwargs = dict(total_heads=8, own_tp=4, page_size=64, seq_along_lane=False)
        geom = mgr._make_geometry(8, 4)
        refused = rs.push_reshard_refusal(geom, fp8_2.shape, 1, **kwargs)
        # The manager appends a hint when the landing arena would admit it.
        got = self._decode([fp8_2], [2], total_heads=8)
        self.assertTrue(got.startswith(refused), got)
        self.assertIn("SGLANG_RAIDEN_RESHARD_LAND_ARENA", got)
        self.assertIn("landing buffer", refused)
        self.assertEqual(
            rs.push_reshard_refusal(geom, fp8_2.shape, 1, land_ok=False, **kwargs),
            refused,
        )
        self.assertIsNone(
            rs.push_reshard_refusal(geom, fp8_2.shape, 1, land_ok=True, **kwargs)
        )

    def test_layout_and_limits(self):
        mla = torch.zeros(4, 64, 576, dtype=torch.bfloat16)
        self.assertIn("5-D", self._decode([mla], [1], total_heads=1))
        bf16 = _layer(torch.bfloat16, 2)
        self.assertIn("page_size", self._decode([bf16], [2], total_heads=8, page=2048))
        self.assertIn("multiple", self._decode([bf16], [2], total_heads=8, src_tp=6))
        self.assertIsNotNone(self._decode([bf16], [2], total_heads=8, src_tp=2))

    def test_draft_layers_with_other_head_dim_refused(self):
        target = _layer(torch.bfloat16, 2)
        draft = TpuFusedMHATokenToKVPool(
            size=128,
            page_size=64,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=256,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
        ).get_fused_kv_buffer(0)
        reason = self._decode([target, draft], [2, 2], total_heads=8)
        self.assertIn("head_dim or dtype", reason)

    def test_prefill_side_and_memo(self):
        mgr = _refusal_manager(
            [_layer(_FP8, 1)], [1], 8, total_heads=8, side="prefill"
        )  # fmt: skip
        reason = mgr.reshard_refusal(8, 4)
        self.assertIsNotNone(reason)
        self.assertIs(mgr.reshard_refusal(8, 4), reason)

    def test_gdn_state_refused_by_name(self):
        # Each side's fixture is servable 8 -> 4 until it holds GDN state.
        fixtures = {
            "decode": ([_layer(torch.bfloat16, 2)], [2], 4),
            "prefill": ([_layer(torch.bfloat16, 1)], [1], 8),
        }
        for side, (tensors, heads, own_tp) in fixtures.items():
            with self.subTest(side=side):
                plain = _refusal_manager(
                    tensors, heads, own_tp, total_heads=8, side=side
                )
                self.assertIsNone(plain.reshard_refusal(8, 4))
                hybrid = _refusal_manager(
                    tensors, heads, own_tp, total_heads=8, side=side
                )
                hybrid._state_kv_caches = (torch.zeros(4, 8),)
                self.assertEqual(
                    hybrid.reshard_refusal(8, 4),
                    f"{rs.push_reshard_where(8, 4)} does not move hybrid model "
                    "state (GDN/Mamba linear-attention rows or DSV4 compressor "
                    "state); use equal attention TP on prefill and decode",
                )

    def test_bounds_check_fires_for_fp8_one_to_two(self):
        geom = rs.ReshardGeometry(8, 4, 4 * 128, 8)
        # Each decode rank holds two fp8 heads in one 32-bit row; the default
        # unit schedule places the second source one row further.
        mgr = _refusal_manager([_layer(_FP8, 2)], [2], 4, total_heads=8)
        req = rs.build_receiver_request(
            _Pb2, uuid=1, req_id="x", peer="d", decode_rank=0, geom=geom,
            sources=[(0, [0]), (1, [0])], dst_block_ids=[0], live_tokens=[64],
        )  # fmt: skip
        with self.assertRaisesRegex(ValueError, "exceeds the block"):
            mgr._check_reshard_bounds(req, 8, 4)

    def test_packed_heads_refused_by_name(self):
        """SGLANG_TPU_KV_PACK_HEADS: a pool storing two 64-wide heads in one
        row reports one 128-wide head; the refusal must say so, not trip over
        the head count."""
        from sglang_tpu.disaggregation.raiden.common import get_raiden_kv_caches
        from sglang_tpu.envs import envs as tpu_envs

        for dtype in (torch.bfloat16, _FP8):
            with tpu_envs.SGLANG_TPU_KV_PACK_HEADS.override(True):
                pool = TpuFusedMHATokenToKVPool(
                    size=128, page_size=PAGE, dtype=dtype, head_num=2, head_dim=64,
                    layer_num=1, device="cpu", enable_memory_saver=False,
                )  # fmt: skip
            groups = get_raiden_kv_caches(pool)
            self.assertEqual(groups.kv_pack_heads, 2)
            # 8 heads: decode TP 4 holds 2 per rank (packed), prefill TP 8 holds 1.
            mgr = _refusal_manager(groups.full, groups.full_head_nums, 4, total_heads=8)
            mgr._kv_pack_heads = groups.kv_pack_heads
            reason = mgr.reshard_refusal(8, 4)
            self.assertIn("SGLANG_TPU_KV_PACK_HEADS", reason)
            self.assertIn("pack 2 heads per row", reason)


class TestPushReshardBytes(unittest.TestCase):
    """Byte-level pins: the predicate allows a cell iff the push is exact."""

    # (layout, itemsize, src_tp, dst_tp, held heads, geometry heads, head_dim,
    #  page, exact)
    CASES = [
        ("NHD", 2, 8, 4, 8, 8, 128, 64, True),
        ("NHD", 2, 4, 1, 8, 8, 64, 64, True),
        ("NHD", 2, 8, 2, 8, 8, 128, 64, True),
        ("NHD", 1, 8, 4, 4, 4, 128, 64, True),
        # fp8 sources holding whole granules (2 -> 4, 2 -> 8, 4 -> 16 heads).
        ("NHD", 1, 4, 2, 8, 8, 128, 64, True),
        ("NHD", 1, 8, 2, 16, 16, 128, 64, True),
        ("NHD", 1, 4, 1, 16, 16, 64, 64, True),
        # fp8 one head per source into more per decode rank needs a landing
        # buffer (#829): 8 kv heads at TP8 -> TP4, 1 -> 2, 1 -> 4.
        ("NHD", 1, 8, 4, 8, 8, 128, 64, False),
        ("NHD", 1, 2, 1, 2, 2, 128, 64, False),
        ("NHD", 1, 16, 4, 16, 16, 128, 64, False),
        # Issue #829 (jcc arm H): 4 kv heads at TP8 -> TP2, replicated sources.
        ("NHD", 1, 8, 2, 4, 4, 128, 64, False),
        ("NHD", 1, 8, 2, 4, 4, 128, 256, False),
        # 480B shape: 8 kv heads at TP8 -> TP4, head_dim 64, page 256.
        ("NHD", 1, 8, 4, 8, 8, 64, 256, False),
        # 3-head sources: no single slot move places them.
        ("NHD", 1, 8, 4, 24, 24, 128, 64, False),
        # Spec §3.2 counterexample: shape-only heads would allow it.
        ("NHD", 1, 8, 4, 8, 4, 128, 64, False),
        ("NHD", 2, 8, 4, 16, 8, 128, 64, False),
        # SEQ_ALONG_LANE moves whole per-head runs, so fp8 one head per source
        # into more per decode rank needs no landing buffer.
        # 1 -> 1 (jcc arm A): 4 kv heads at TP8 -> TP4.
        ("HND", 2, 8, 4, 4, 4, 128, 128, True),
        ("HND", 1, 8, 4, 4, 4, 128, 256, True),
        ("HND", 2, 8, 4, 4, 4, 256, 128, True),
        # 1 -> 2 concat, replicated sources (jcc arm G): 4 kv heads TP8 -> TP2.
        ("HND", 2, 8, 2, 4, 4, 128, 256, True),
        ("HND", 1, 8, 2, 4, 4, 128, 256, True),
        # 1 -> 2 concat (jcc arm I2): 8 kv heads at TP8 -> TP4.
        ("HND", 2, 8, 4, 8, 8, 128, 128, True),
        ("HND", 2, 8, 4, 8, 8, 128, 256, True),
        ("HND", 1, 8, 4, 8, 8, 128, 256, True),
        ("HND", 1, 8, 4, 8, 8, 64, 256, True),
        # 2 -> 4, 1 -> 4, and replication on both sides.
        ("HND", 1, 4, 2, 8, 8, 128, 256, True),
        ("HND", 1, 16, 4, 16, 16, 128, 256, True),
        ("HND", 2, 8, 4, 2, 2, 128, 128, True),
        ("HND", 1, 8, 4, 2, 2, 128, 256, True),
        # Shape-only heads: the geometry's count is not what the layer holds.
        ("HND", 1, 8, 4, 8, 4, 128, 256, False),
        ("HND", 2, 8, 4, 16, 8, 128, 256, False),
    ]

    # CASES that a scratch block and land_merge make exact.
    LANDS = {
        ("NHD", 1, 8, 4, 8, 8, 128, 64),
        ("NHD", 1, 2, 1, 2, 2, 128, 64),
        ("NHD", 1, 16, 4, 16, 16, 128, 64),
        ("NHD", 1, 8, 2, 4, 4, 128, 64),
        ("NHD", 1, 8, 2, 4, 4, 128, 256),
        ("NHD", 1, 8, 4, 8, 8, 64, 256),
    }

    def test_verdict_matches_bytes(self):
        for case in self.CASES:
            *args, exact = case
            with self.subTest(case=case):
                self.assertEqual(_push_correct(*args), exact)
                self.assertEqual(_allowed(*args), exact)

    def test_land_verdict_matches_merged_bytes(self):
        """With landing allowed, the predicate allows a cell iff the replay
        through scratch blocks and land_merge is exact; the managers (no
        land_ok yet) keep refusing the landing cells."""
        self.assertLessEqual(self.LANDS, {tuple(c[:-1]) for c in self.CASES})
        for case in self.CASES:
            *args, exact = case
            want = exact or tuple(args) in self.LANDS
            with self.subTest(case=case):
                self.assertEqual(_push_correct(*args, land=True), want)
                self.assertEqual(_land_allowed(*args), want)
                if tuple(args) in self.LANDS:
                    self.assertFalse(_allowed(*args))
                    # Negative control: skipping the merge leaves source
                    # padding on the odd heads.
                    self.assertFalse(_push_correct(*args, land=True, merge=False))
                    # Negative control: merging each final page with the
                    # other page's scratch block crosses page labels.
                    crossed = _SCRATCH[::-1]
                    self.assertFalse(
                        _push_correct(*args, land=True, scratch_pairs=crossed)
                    )

    def test_land_fp8_sweep(self):
        """Every fp8 token-major fan-in cell: allowed with land_ok iff the
        merged replay is exact, and allowed without it iff the plain replay
        is; the landing class is exactly the one-head sources into an even
        decode head count."""
        landed = 0
        for held, src_tp, dst_tp in itertools.product(
            (1, 2, 4, 8, 16, 32), (2, 4, 8, 16), (1, 2, 4, 8)
        ):
            if src_tp <= dst_tp or src_tp % dst_tp:
                continue
            for head_dim, page in ((128, 64), (64, 256)):
                args = ("NHD", 1, src_tp, dst_tp, held, held, head_dim, page)
                with self.subTest(args=args):
                    plain = _push_correct(*args)
                    merged = _push_correct(*args, land=True)
                    self.assertEqual(_allowed(*args), plain)
                    self.assertEqual(_land_allowed(*args), merged)
                    self.assertTrue(merged or not plain)
                    one_to_even = held <= src_tp and max(1, held // dst_tp) % 2 == 0
                    self.assertEqual(merged and not plain, one_to_even)
                    landed += merged and not plain
        self.assertGreater(landed, 0)

    def test_requests_without_landing_are_unchanged(self):
        """scratch_block_ids is ignored by plans with no LandEntry, and a
        LandEntry without scratch ids is an error, not a misplaced push."""
        for case in self.CASES:
            layout, itemsize, src_tp, dst_tp, held, geom_heads, head_dim, page = case[
                :-1
            ]
            d_h = max(1, held // dst_tp)
            t = _Tensor(_shape(layout, itemsize, d_h, head_dim, page), itemsize)
            geom = _side_geometry(layout, t, d_h, src_tp, dst_tp, geom_heads, page)
            with self.subTest(case=case):
                try:
                    rs._schedule(geom)
                except LandRequired:
                    continue

                def build(**extra):
                    return rs.build_receiver_request(
                        _Pb2, uuid=1, req_id="x", peer="d", decode_rank=0,
                        geom=geom, sources=[(0, [4, 5]), (1, [6, 7])],
                        dst_block_ids=[8, 9], live_tokens=[page, 3], **extra,
                    )  # fmt: skip

                def entries(req):
                    return {
                        k: [vars(e) for e in v.entries]
                        for k, v in req.shard_push_schedules.items()
                    }

                self.assertEqual(
                    entries(build()), entries(build(scratch_block_ids=[10, 11]))
                )
        land = _side_geometry(
            "NHD", _Tensor(_shape("NHD", 1, 2, 128, 64), 1), 2, 8, 4, 8, 64
        ).with_land()
        with self.assertRaisesRegex(ValueError, "scratch_block_ids"):
            rs.build_sender_request(
                _Pb2, uuid=1, req_id="x", peer="d", prefill_rank=1, decode_rank=0,
                geom=land, src_block_ids=[0], dst_block_ids=[1], live_tokens=[64],
            )  # fmt: skip

    def test_half_count_geometry_is_no_longer_built(self):
        """fp8 layers holding twice the geometry's heads, even heads per rank.
        Before #829 the push path sized the sliding-window geometry from the
        full count where the fp8 head-pair unit made that byte-correct
        (swa_geometry_kv_heads), so these pushes were exact. _make_geometry now
        sizes a token-major group from its own count, which gives the very
        same entries; a half-count token-major geometry is never built, and
        the predicate refuses it."""
        for src_tp, dst_tp, held in ((4, 2, 8), (8, 2, 16)):
            half = ("NHD", 1, src_tp, dst_tp, held, held // 2, 128, 64)
            real = ("NHD", 1, src_tp, dst_tp, held, held, 128, 64)
            with self.subTest(src_tp=src_tp, dst_tp=dst_tp, held=held):
                self.assertFalse(_allowed(*half))
                self.assertTrue(_allowed(*real))
                self.assertTrue(_push_correct(*real))
                layout = KVLayoutGeometry(rs.TOKEN_MAJOR, 4, 1, held // src_tp, 128, 64)
                ratio = rs.reshard_ratio(src_tp, dst_tp)
                legacy = ReshardGeometry(src_tp, dst_tp, 4 * 128, held // 2)
                new = ReshardGeometry.from_layout(src_tp, dst_tp, layout, held)
                for p in range(src_tp):
                    self.assertEqual(
                        new.entry_bytes(p, p // ratio),
                        legacy.entry_bytes(p, p // ratio),
                    )

    def test_replay_rejects_a_misplaced_schedule(self):
        """Negative control: the replay that passes the allowed fp8 2 -> 4
        cell fails once its stride or offset is off by one granule."""
        case = ("NHD", 1, 4, 2, 8, 8, 128, 64)
        self.assertTrue(_push_correct(*case, tamper=lambda e: e))
        for name, tamper in (
            ("dst stride", lambda e: (*e[:4], e[4] + 512)),
            ("dst offset", lambda e: (*e[:3], 0, e[4])),
            ("src stride", lambda e: (e[0], e[1], e[2] + 512, e[3], e[4])),
        ):
            with self.subTest(tamper=name):
                self.assertFalse(_push_correct(*case, tamper=tamper))

    def test_replay_rejects_a_misplaced_head_run(self):
        """Negative control for SEQ_ALONG_LANE: the fp8 1 -> 2 replay (8 kv
        heads at TP8 -> TP4) fails once a head run lands on the wrong head,
        or is cut short, and under the pre-#829 per-token row unit."""
        case = ("HND", 1, 8, 4, 8, 8, 128, 256)
        run = 2 * 128 * 256
        self.assertTrue(_push_correct(*case, tamper=lambda e: e))
        for name, tamper in (
            ("dst offset", lambda e: (*e[:3], 0, e[4])),
            ("src offset", lambda e: (e[0] + run, *e[1:])),
            ("size", lambda e: (e[0], e[1] - 4, *e[2:])),
        ):
            with self.subTest(tamper=name):
                self.assertFalse(_push_correct(*case, tamper=tamper))
        # The pre-#829 unit, one (kv_packing, page) row per head, per token.
        legacy = rs.ReshardGeometry(8, 4, 4 * 256, 8)
        self.assertFalse(_replay_requests("HND", 1, 8, 4, 8, 128, 256, legacy, legacy))


class TestRefusalWithSlidingWindow(unittest.TestCase):
    def test_refusal_matches_both_groups_bytes(self):
        """reshard_refusal checks the full and the sliding-window group. Over
        real pools it allows a cell iff both groups land byte for byte, with
        the SWA group sized by _make_geometry from its own head count."""

        def manager(dtype, full_heads, swa_heads, tp, mode):
            mgr = _manager(dtype, full_heads, swa_heads, tp)
            mgr.attn_tp_size = tp
            mgr.kv_args.page_size = PAGE
            mgr.disaggregation_mode = mode
            mgr._kv_head_nums = (max(1, full_heads // tp),)
            mgr._swa_kv_head_nums = (max(1, swa_heads // tp),)
            return mgr

        topologies = ((2, 1), (4, 2), (4, 1), (8, 4), (8, 2), (16, 8), (16, 4))
        for dtype in (torch.bfloat16, _FP8):
            for full_heads, swa_heads in itertools.product((2, 4, 8, 16), repeat=2):
                for src_tp, dst_tp in topologies:
                    cell = (dtype, full_heads, swa_heads, src_tp, dst_tp)
                    with self.subTest(cell=cell):
                        heads = (full_heads, swa_heads, src_tp, dst_tp)
                        try:
                            exact = _replay(*heads, dtype=dtype, swa=False) and _replay(
                                *heads, dtype=dtype, swa=True
                            )
                        except LandRequired:
                            exact = False
                        allowed = all(
                            manager(
                                dtype, full_heads, swa_heads, tp, mode
                            ).reshard_refusal(src_tp, dst_tp)
                            is None
                            for tp, mode in (
                                (dst_tp, DisaggregationMode.DECODE),
                                (src_tp, DisaggregationMode.PREFILL),
                            )
                        )
                        self.assertEqual(allowed, exact)

    def test_land_full_group_only(self):
        """The policy commit (iii) wires: land_ok for the full-attention
        group only. A cell is then allowed iff the full group's merged replay
        and the SWA group's plain replay are exact, so SWA cells that need
        landing stay refused."""
        swa_land = 0
        for full_heads, swa_heads in itertools.product((2, 4, 8, 16), repeat=2):
            for src_tp, dst_tp in ((2, 1), (4, 2), (8, 4), (8, 2), (16, 4)):
                cell = (full_heads, swa_heads, src_tp, dst_tp)
                with self.subTest(cell=cell):
                    full = _replay(*cell, dtype=_FP8, swa=False, land=True)
                    try:
                        swa = _replay(*cell, dtype=_FP8, swa=True)
                    except LandRequired:
                        swa = False
                        swa_land += _replay(*cell, dtype=_FP8, swa=True, land=True)
                    allowed = True
                    for tp in (dst_tp, src_tp):
                        mgr = _manager(_FP8, full_heads, swa_heads, tp)
                        for group, heads in ((False, full_heads), (True, swa_heads)):
                            t = (mgr._swa_kv_caches if group else mgr._kv_caches)[0]
                            reason = rs.push_reshard_refusal(
                                mgr._make_geometry(src_tp, dst_tp, swa=group),
                                t.shape,
                                1,
                                total_heads=heads,
                                own_tp=tp,
                                page_size=PAGE,
                                seq_along_lane=False,
                                land_ok=not group,
                            )
                            allowed &= reason is None
                    self.assertEqual(allowed, full and swa)
        # Some refused SWA groups would have landed exactly.
        self.assertGreater(swa_land, 0)

    def test_seq_along_lane_refusal_matches_both_groups_bytes(self):
        """The same over real SEQ_ALONG_LANE pools: every cell's whole head
        runs land byte for byte, so every multiple-of topology is allowed."""

        def manager(dtype, full_heads, swa_heads, tp, mode):
            mgr = _manager(dtype, full_heads, swa_heads, tp, sal=True)
            mgr.disaggregation_mode = mode
            mgr._kv_head_nums = (max(1, full_heads // tp),)
            mgr._swa_kv_head_nums = (max(1, swa_heads // tp),)
            return mgr

        topologies = ((2, 1), (4, 2), (8, 4), (8, 2), (16, 4))
        for dtype in (torch.bfloat16, _FP8):
            for full_heads, swa_heads in ((4, 8), (8, 4), (2, 16), (8, 8)):
                for src_tp, dst_tp in topologies:
                    cell = (dtype, full_heads, swa_heads, src_tp, dst_tp)
                    with self.subTest(cell=cell):
                        heads = (full_heads, swa_heads, src_tp, dst_tp)
                        exact = _replay(
                            *heads, dtype=dtype, swa=False, sal=True
                        ) and _replay(*heads, dtype=dtype, swa=True, sal=True)
                        allowed = all(
                            manager(
                                dtype, full_heads, swa_heads, tp, mode
                            ).reshard_refusal(src_tp, dst_tp)
                            is None
                            for tp, mode in (
                                (dst_tp, DisaggregationMode.DECODE),
                                (src_tp, DisaggregationMode.PREFILL),
                            )
                        )
                        self.assertTrue(exact)
                        self.assertEqual(allowed, exact)


if __name__ == "__main__":
    unittest.main()
