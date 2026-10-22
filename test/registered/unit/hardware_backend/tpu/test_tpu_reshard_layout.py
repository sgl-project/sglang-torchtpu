"""Compare Raiden reshard byte ranges with the actual fused KV pool layout."""

import itertools
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.raiden import reshard_schedule as rs
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager
from sglang_tpu.disaggregation.raiden.reshard_schedule import (
    ReshardGeometry,
    head_pair_bytes,
)
from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE = 64


def _layer(dtype, heads):
    pool = TpuFusedMHATokenToKVPool(
        size=128,
        page_size=PAGE,
        dtype=dtype,
        head_num=heads,
        head_dim=128,
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


def _manager(dtype, full_heads, swa_heads, tp):
    """A RaidenKVManager holding one full and one SWA layer for an attention-TP
    rank of a model with the given global kv-head counts."""
    mgr = object.__new__(RaidenKVManager)
    mgr._kv_caches = [_layer(dtype, max(1, full_heads // tp))]
    mgr._swa_kv_caches = [_layer(dtype, max(1, swa_heads // tp))]
    mgr.kv_args = SimpleNamespace(
        total_kv_head_num=full_heads, total_swa_kv_head_num=swa_heads
    )
    return mgr


def _fill(layer, heads):
    """Tag every live byte of page 0 by (global head, k|v, token, byte) and
    return (page bytes, live mask). The fused layout puts K and V of local
    head i at kv-slots 2i, 2i+1, packed kv_packing slots per 32-bit word."""
    packing = int(layer.shape[3])
    raw = layer.view(torch.uint8)
    raw.fill_(0xFF)
    live = torch.zeros_like(raw, dtype=torch.bool)
    token = torch.arange(PAGE)[:, None]
    byte = torch.arange(raw.shape[-1])[None, :]
    for local, head in enumerate(heads):
        for kv in (0, 1):
            slot = 2 * local + kv
            tag = ((2 * head + kv) * 131 + token * 7 + byte * 3) % 251
            raw[0, :, slot // packing, slot % packing, :] = tag.to(torch.uint8)
            live[0, :, slot // packing, slot % packing, :] = True
    return raw[0].flatten(), live[0].flatten()


def _replay(full_heads, swa_heads, src_tp, dst_tp, *, dtype, swa, heads=None):
    """Replay a push reshard byte for byte over real pool pages. Each side
    builds its geometry from its own tensor: via _make_geometry, or with an
    explicit kv-head count. True when every decode rank holds exactly its own
    heads with no out-of-bounds copy."""
    total = swa_heads if swa else full_heads
    sender = _manager(dtype, full_heads, swa_heads, src_tp)
    receiver = _manager(dtype, full_heads, swa_heads, dst_tp)

    def layer(mgr):
        return (mgr._swa_kv_caches if swa else mgr._kv_caches)[0]

    def geometry(mgr):
        if heads is None:
            return mgr._make_geometry(src_tp, dst_tp, swa=swa)
        return ReshardGeometry(src_tp, dst_tp, head_pair_bytes(layer(mgr)), heads)

    send_geom, recv_geom = geometry(sender), geometry(receiver)
    for dst_rank in range(dst_tp):
        want, live = _fill(
            layer(receiver).clone(), _rank_heads(total, dst_tp, dst_rank)
        )
        got = torch.full_like(want, 0xFF)
        ratio = send_geom.ratio
        for src_rank in range(dst_rank * ratio, (dst_rank + 1) * ratio):
            src, _ = _fill(layer(sender).clone(), _rank_heads(total, src_tp, src_rank))
            src_off, size, src_stride, _, _ = send_geom.entry_bytes(src_rank, dst_rank)
            _, _, _, dst_off, dst_stride = recv_geom.entry_bytes(src_rank, dst_rank)
            for t in range(PAGE):
                a, b = src_off + t * src_stride, dst_off + t * dst_stride
                if a + size > src.numel() or b + size > got.numel():
                    return False
                got[b : b + size] = src[a : a + size]
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
        schedule as before total_swa_kv_head_num existed."""
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            for heads, (src_tp, dst_tp) in ((8, (8, 4)), (8, (16, 8)), (4, (8, 2))):
                with self.subTest(dtype=dtype, heads=heads, tp=(src_tp, dst_tp)):
                    mgr = _manager(dtype, heads, heads, dst_tp)
                    old = ReshardGeometry(
                        src_tp,
                        dst_tp,
                        head_pair_bytes(mgr._swa_kv_caches[0]),
                        heads,
                    )
                    new = mgr._make_geometry(src_tp, dst_tp, swa=True)
                    for src_rank in range(src_tp):
                        dst_rank = src_rank // new.ratio
                        self.assertEqual(
                            new.entry_bytes(src_rank, dst_rank),
                            old.entry_bytes(src_rank, dst_rank),
                        )

    def test_fp8_swa_allows_exactly_the_byte_correct_cells(self):
        """At fp8 the head-pair granule spans two kv-heads (#829). A cell is
        allowed iff some kv-head count places every byte correctly, and then
        with that count; it is refused iff both the SWA and the full count are
        wrong. The full count is correct by coincidence only where it equals
        half the SWA count and every rank holds whole packs; those cells are
        pinned below. TODO(#829): drop the coincidence cells with B1."""
        fp8 = torch.float8_e4m3fn
        coincidences = set()
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
                    by_count = {
                        heads: _replay(*cell, dtype=fp8, swa=True, heads=heads)
                        for heads in (swa_heads, full_heads)
                    }
                    mgr = _manager(fp8, full_heads, swa_heads, dst_tp)
                    try:
                        heads = mgr._make_geometry(
                            src_tp, dst_tp, swa=True
                        ).total_kv_heads
                    except ValueError as error:
                        self.assertIn("sliding-window KV", str(error))
                        self.assertFalse(any(by_count.values()))
                        continue
                    self.assertTrue(_replay(*cell, dtype=fp8, swa=True))
                    self.assertTrue(by_count[heads])
                    if heads != swa_heads:
                        coincidences.add(cell)
        self.assertEqual(
            coincidences,
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
        )
        # Full-attention geometry ignores the SWA count.
        mgr = _manager(fp8, 4, 8, 4)
        self.assertEqual(mgr._make_geometry(8, 4).total_kv_heads, 4)


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


def _push_correct(layout, itemsize, src_tp, dst_tp, held, geom_heads, head_dim, page):
    """Replay the legacy push byte for byte: the sender sends at most its block
    of the entry's chunks, the receiver writes exactly what was sent."""
    ratio = rs.reshard_ratio(src_tp, dst_tp)
    s_h, d_h = max(1, held // src_tp), max(1, held // dst_tp)
    src_t = _Tensor(_shape(layout, itemsize, s_h, head_dim, page), itemsize)
    dst_t = _Tensor(_shape(layout, itemsize, d_h, head_dim, page), itemsize)
    gsrc = rs.ReshardGeometry(src_tp, dst_tp, rs.head_pair_bytes(src_t), geom_heads)
    gdst = rs.ReshardGeometry(src_tp, dst_tp, rs.head_pair_bytes(dst_t), geom_heads)
    src_block = int(np.prod(src_t.shape[1:])) * itemsize
    for d in range(dst_tp):
        expect = _labels(
            layout, itemsize, d_h, _global_heads(d, dst_tp, held), head_dim, page
        )
        got = np.zeros(expect.size * 8, dtype=np.int64)
        srcs = rs.source_ranks_for(d, ratio)
        rreq = rs.build_receiver_request(
            _Pb2, uuid=1, req_id="x", peer="d", decode_rank=d, geom=gdst,
            sources=[(p, [0]) for p in srcs], dst_block_ids=[0], live_tokens=[page],
        )  # fmt: skip
        for p in srcs:
            sent = _labels(
                layout, itemsize, s_h, _global_heads(p, src_tp, held), head_dim, page
            )
            sent = np.concatenate([sent, np.full(sent.size * 8, -1, np.int64)])
            sreq = rs.build_sender_request(
                _Pb2, uuid=1, req_id="x", peer="d", prefill_rank=p, decode_rank=d,
                geom=gsrc, src_block_ids=[0], dst_block_ids=[0], live_tokens=[page],
            )  # fmt: skip
            sidx = np.concatenate(
                [
                    _chunks(
                        e.src_offset_bytes, e.size_bytes, e.src_stride_bytes, e.count
                    )
                    for e in sreq.shard_push_schedules[0].entries
                ]
            )[:src_block]
            didx = np.concatenate(
                [
                    _chunks(
                        e.dst_offset_bytes, e.size_bytes, e.dst_stride_bytes, e.count
                    )
                    for e in rreq.shard_push_schedules[p].entries
                ]
            )
            if didx.size < sidx.size:
                return False
            got[didx[: sidx.size]] = sent[sidx]
        if got[expect.size :].any():
            return False
        mask = expect != 0
        if not np.array_equal(got[: expect.size][mask], expect[mask]):
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
        fp8_4 = _layer(_FP8, 4)
        self.assertIsNotNone(self._decode([fp8_4], [4], total_heads=16))

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
        ("NHD", 1, 8, 4, 4, 4, 128, 64, True),
        ("NHD", 1, 8, 4, 8, 8, 128, 64, False),
        ("NHD", 1, 4, 2, 8, 8, 128, 64, False),
        # TODO(#829) coincidence: fp8 layers holding twice the geometry's heads
        # with even heads per rank (sliding-window before #1076, draft).
        ("NHD", 1, 4, 2, 8, 4, 128, 64, True),
        ("NHD", 1, 8, 2, 16, 8, 128, 64, True),
        # Spec §3.2 counterexample: shape-only heads would allow it.
        ("NHD", 1, 8, 4, 8, 4, 128, 64, False),
        ("NHD", 2, 8, 4, 16, 8, 128, 64, False),
        # TODO(#829) coincidence: SEQ_ALONG_LANE at one head per rank.
        ("HND", 2, 8, 4, 4, 4, 128, 128, True),
        ("HND", 1, 8, 4, 4, 4, 128, 256, True),
        ("HND", 2, 8, 4, 4, 4, 256, 128, False),
        ("HND", 2, 8, 4, 8, 8, 128, 128, False),
        ("HND", 2, 8, 4, 2, 2, 128, 128, True),
    ]

    def test_verdict_matches_bytes(self):
        for case in self.CASES:
            *args, exact = case
            with self.subTest(case=case):
                self.assertEqual(_push_correct(*args), exact)
                self.assertEqual(_allowed(*args), exact)


class TestRefusalWithSlidingWindow(unittest.TestCase):
    def test_refusal_matches_both_groups_bytes(self):
        """reshard_refusal checks the full and the sliding-window group. Over
        real pools it allows a cell iff both groups land byte for byte, with
        the SWA group sized by _make_geometry (#1076), including the fp8
        cells where that keeps the full count."""

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
                        exact = _replay(*heads, dtype=dtype, swa=False)
                        try:
                            exact = exact and _replay(*heads, dtype=dtype, swa=True)
                        except ValueError:
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


if __name__ == "__main__":
    unittest.main()
