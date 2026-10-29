"""Compare Raiden reshard byte ranges with the actual fused KV pool layout."""

import itertools
import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
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


if __name__ == "__main__":
    unittest.main()
