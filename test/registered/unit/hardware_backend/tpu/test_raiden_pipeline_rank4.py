"""Rank-4 RPA-seq KV pages crossing a Raiden pipeline transfer.

Applies the spans head_slice_spans emits to per-rank destination pages, byte
for byte, and checks every rank ends up with exactly the kv-heads it owns.
"""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.raiden import pipeline as pl
from sglang_tpu.disaggregation.raiden import pp_common

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _groups(full, swa, full_ids, swa_ids):
    return SimpleNamespace(
        full=full,
        swa=swa,
        full_layer_ids=full_ids,
        swa_layer_ids=swa_ids,
        num_draft_full=0,
        num_draft_swa=0,
    )


def _apply(spans, src_pages, dst_pages_per_rank):
    """Copy span bytes from source pages into each rank's flat destination space."""
    for span in spans.spans:
        src = src_pages[span.src_block_ordinal]
        dst = dst_pages_per_rank[span.dst_unit_ordinal]
        for i in range(span.count):
            s = span.src_offset_bytes + i * span.src_stride_bytes
            d = span.dst_offset_bytes + i * span.dst_stride_bytes
            dst[d : d + span.size_bytes] = src[s : s + span.size_bytes]


@pytest.mark.parametrize(
    "total_kv_heads,dst_tp,dtype",
    [
        (16, 16, torch.bfloat16),  # one head per rank
        (16, 8, torch.bfloat16),  # two heads per rank
        (8, 16, torch.bfloat16),  # each head replicated on two ranks
        (16, 16, torch.float8_e4m3fn),
    ],
)
def test_rank4_head_slices_land_on_owning_rank(total_kv_heads, dst_tp, dtype):
    head_dim, page = 128, 512
    src = torch.zeros((6, 2 * total_kv_heads, head_dim, page), dtype=dtype)
    swa_src = torch.zeros((4, 2 * total_kv_heads, head_dim, page), dtype=dtype)
    pools = pl.pool_manifest(_groups([src], [swa_src], [1], [2]))
    full_ids, swa_ids = [4, 0, 2], [1, 3]
    spans = pp_common.head_slice_spans(
        pools, [src, swa_src], full_ids, swa_ids, dst_tp, total_kv_heads, total_kv_heads
    )
    rng = np.random.default_rng(0)
    heads_per_rank = max(1, total_kv_heads // dst_tp)
    replication = max(1, dst_tp // total_kv_heads)
    run = 2 * head_dim * page * src.element_size()
    for pool_spans, block_ids in zip(spans, (full_ids, swa_ids)):
        n = len(block_ids)
        pages = rng.integers(
            0, 255, size=(n, 2 * total_kv_heads * run // 2), dtype=np.uint8
        )
        dst_page_bytes = heads_per_rank * run
        dst = [np.zeros(n * dst_page_bytes, dtype=np.uint8) for _ in range(dst_tp)]
        # Every rank receives its heads, so replicated heads count once per rank.
        assert pool_spans.declared_bytes == n * dst_tp * dst_page_bytes
        _apply(pool_spans, pages, dst)
        for rank in range(dst_tp):
            first_head = rank // replication * heads_per_rank
            for index in range(n):
                want = pages[index][
                    first_head * run : (first_head + heads_per_rank) * run
                ]
                got = dst[rank][index * dst_page_bytes : (index + 1) * dst_page_bytes]
                assert np.array_equal(got, want), (rank, index)


def test_rank4_swa_pool_slices_over_swa_heads():
    """A sliding-window pool with fewer kv-heads than full attention is sliced
    over its own head count, not the full-attention one."""
    head_dim, page, dst_tp = 128, 512, 4
    full_heads, swa_heads = 16, 4
    src = torch.zeros((3, 2 * full_heads, head_dim, page), dtype=torch.bfloat16)
    swa_src = torch.zeros((3, 2 * swa_heads, head_dim, page), dtype=torch.bfloat16)
    pools = pl.pool_manifest(_groups([src], [swa_src], [1], [2]))
    full_spans, swa_spans = pp_common.head_slice_spans(
        pools, [src, swa_src], [0, 1], [2], dst_tp, full_heads, swa_heads
    )
    run = 2 * head_dim * page * src.element_size()
    assert {s.size_bytes for s in full_spans.spans} == {full_heads // dst_tp * run}
    assert {s.size_bytes for s in swa_spans.spans} == {swa_heads // dst_tp * run}
    # The SWA source page holds 2 * swa_heads rows; no span may read past it.
    assert max(s.src_offset_bytes + s.size_bytes for s in swa_spans.spans) == (
        swa_heads * run
    )


def test_rank4_fingerprint_ignores_per_rank_heads():
    """A PP producer holds every kv-head and a TP consumer a slice of them; the
    pools must still match, or the controller rejects the transfer."""
    getter = lambda t: ((3, 2, 1, 0), ((16, 128),), 16)  # noqa: E731
    prefill = torch.zeros((5, 32, 128, 512), dtype=torch.bfloat16)
    decode = torch.zeros((7, 2, 128, 512), dtype=torch.bfloat16)
    p_pools = pl.pool_manifest(_groups([prefill], [], [5], []))
    d_pools = pl.pool_manifest(_groups([decode], [], [5], []))
    assert pl.layout_fingerprint(p_pools, [prefill], getter) == pl.layout_fingerprint(
        d_pools, [decode], getter
    )


def test_unsupported_or_mixed_kv_layouts_are_rejected():
    fused = torch.zeros((4, 512, 16, 2, 128), dtype=torch.bfloat16)
    seq = torch.zeros((4, 32, 128, 512), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="share one page layout"):
        pl.pool_manifest(_groups([fused], [seq], [1], [0]))
    with pytest.raises(ValueError, match="rank-5 fused or rank-4"):
        pl.pool_manifest(_groups([torch.zeros((4, 8, 8))], [], [1], []))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
