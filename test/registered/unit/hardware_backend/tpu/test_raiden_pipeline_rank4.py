"""Head-major KV pages (rank-4 RPA-seq, SEQ_ALONG_LANE) crossing a Raiden
pipeline transfer.

Applies the spans head_slice_spans emits to per-rank destination pages, byte
for byte, and checks every rank ends up with exactly the kv-heads it owns.
SEQ_ALONG_LANE cases use real TpuSeqAlongLaneKVPool buffers built on CPU.
"""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.raiden import common as raiden_common
from sglang_tpu.disaggregation.raiden import pipeline as pl
from sglang_tpu.disaggregation.raiden import pp_common
from sglang_tpu.experimental.speculative import pd_hidden_pool
from sglang_tpu.experimental.speculative.pd_hidden_pool import PdHiddenPool
from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool, TpuSeqAlongLaneKVPool

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _groups(full, swa, full_ids, swa_ids):
    return SimpleNamespace(
        full=full,
        swa=swa,
        full_layer_ids=full_ids,
        swa_layer_ids=swa_ids,
        num_draft_full=0,
        num_hidden_full=0,
        num_draft_swa=0,
        seq_along_lane=False,
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
        pools,
        [src, swa_src],
        full_ids,
        swa_ids,
        dst_tp,
        total_kv_heads,
        seq_along_lane=False,
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


def test_rank4_fingerprint_ignores_per_rank_heads():
    """A PP producer holds every kv-head and a TP consumer a slice of them; the
    pools must still match, or the controller rejects the transfer."""
    getter = lambda t: ((3, 2, 1, 0), ((16, 128),), 16)  # noqa: E731
    prefill = torch.zeros((5, 32, 128, 512), dtype=torch.bfloat16)
    decode = torch.zeros((7, 2, 128, 512), dtype=torch.bfloat16)
    p_pools = pl.pool_manifest(_groups([prefill], [], [5], []))
    d_pools = pl.pool_manifest(_groups([decode], [], [5], []))
    assert pl.layout_fingerprint(
        p_pools, [prefill], getter, ships_pd_hidden=False, seq_along_lane=False
    ) == pl.layout_fingerprint(
        d_pools, [decode], getter, ships_pd_hidden=False, seq_along_lane=False
    )


def test_unsupported_or_mixed_kv_layouts_are_rejected():
    fused = torch.zeros((4, 512, 16, 2, 128), dtype=torch.bfloat16)
    seq = torch.zeros((4, 32, 128, 512), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="share one page layout"):
        pl.pool_manifest(_groups([fused], [seq], [1], [0]))
    with pytest.raises(ValueError, match="rank-5 fused or rank-4"):
        pl.pool_manifest(_groups([torch.zeros((4, 8, 8))], [], [1], []))


# Muse Spark: 16 KV heads, 128-wide heads, 256-token pages.
HEADS, HEAD_DIM, PAGE = 16, 128, 256
LAYERS = 2


def _head_dim_without_tpu(*, head_dim, kv_packing):
    # The pool asks Pallas for the sublane count, which needs a TPU; tpu7x has 8.
    tile = 8 * kv_packing
    return -(-head_dim // tile) * tile


def _pool(pool_cls, *, heads: int, dtype: torch.dtype, pages: int):
    with patch("sglang_tpu.kv_pool._seq_along_lane_head_dim", _head_dim_without_tpu):
        return pool_cls(
            size=pages * PAGE,
            page_size=PAGE,
            dtype=dtype,
            head_num=heads,
            head_dim=HEAD_DIM,
            layer_num=LAYERS,
            device="cpu",
            enable_memory_saver=False,
        )


def _page_bytes(tensor: torch.Tensor) -> np.ndarray:
    """[num_blocks, page bytes] uint8 view sharing the tensor's storage."""
    return tensor.view(torch.uint8).reshape(tensor.shape[0], -1).numpy()


def _apply_by_block(
    pool_spans, src: np.ndarray, dst_per_rank: list[np.ndarray]
) -> None:
    """Copy span bytes into each rank's flat destination space of the transfer."""
    for span in pool_spans.spans:
        page = src[pool_spans.block_ids[span.src_block_ordinal]]
        dst = dst_per_rank[span.dst_unit_ordinal]
        for i in range(span.count):
            s = span.src_offset_bytes + i * span.src_stride_bytes
            d = span.dst_offset_bytes + i * span.dst_stride_bytes
            dst[d : d + span.size_bytes] = page[s : s + span.size_bytes]


def _randomize(tensor: torch.Tensor, seed: int) -> None:
    rng = np.random.default_rng(seed)
    view = _page_bytes(tensor)
    view[...] = rng.integers(0, 256, size=view.shape, dtype=np.uint8)


def _land(pool_spans, src: torch.Tensor, consumers: list[torch.Tensor], dst_blocks):
    """Run one transfer of src's pages into each rank's consumer pages."""
    page_bytes = _page_bytes(consumers[0]).shape[1]
    n = len(pool_spans.block_ids)
    flat = [np.zeros(n * page_bytes, dtype=np.uint8) for _ in consumers]
    _apply_by_block(pool_spans, _page_bytes(src), flat)
    for consumer, staged in zip(consumers, flat):
        _page_bytes(consumer)[dst_blocks] = staged.reshape(n, page_bytes)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("dst_tp", [2, 4, 8])
def test_each_rank_gets_its_heads_of_every_page(dst_tp, dtype):
    producer = _pool(TpuSeqAlongLaneKVPool, heads=HEADS, dtype=dtype, pages=6)
    groups = raiden_common.get_raiden_kv_caches(producer)
    assert groups.seq_along_lane
    src = groups.full[0]
    _randomize(src, seed=dst_tp)
    pools = pl.pool_manifest(groups)
    src_blocks, dst_blocks = [4, 1, 3], [2, 5, 1]
    (pool_spans, _) = pp_common.head_slice_spans(
        pools, groups.full, src_blocks, [], dst_tp, HEADS, seq_along_lane=True
    )
    # One contiguous run per page and destination rank.
    assert len(pool_spans.spans) == dst_tp * len(src_blocks)
    assert all(span.count == 1 for span in pool_spans.spans)

    per_rank = HEADS // dst_tp
    consumers = [
        _pool(
            TpuSeqAlongLaneKVPool, heads=per_rank, dtype=dtype, pages=6
        ).get_fused_kv_buffer(0)
        for _ in range(dst_tp)
    ]
    _land(pool_spans, src, consumers, dst_blocks)
    for rank, consumer in enumerate(consumers):
        rows = slice(2 * rank * per_rank, 2 * (rank + 1) * per_rank)
        for s, d in zip(src_blocks, dst_blocks):
            want = src[s, rows].view(torch.uint8)
            assert torch.equal(consumer[d].view(torch.uint8), want), (rank, s)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_head_run_matches_the_pool_strides(dtype):
    buf = _pool(TpuSeqAlongLaneKVPool, heads=HEADS, dtype=dtype, pages=2)
    buf = buf.get_fused_kv_buffer(0)
    run = pp_common.seq_along_lane_head_run_bytes(buf)
    assert run == 2 * buf.stride(1) * buf.element_size()
    assert HEADS * run == buf.stride(0) * buf.element_size()


def test_sliding_window_pool_is_sliced_like_the_full_pool():
    full = _pool(TpuSeqAlongLaneKVPool, heads=HEADS, dtype=torch.bfloat16, pages=4)
    window = _pool(TpuSeqAlongLaneKVPool, heads=HEADS, dtype=torch.bfloat16, pages=3)
    groups = SimpleNamespace(
        full=[full.get_fused_kv_buffer(0)],
        swa=[window.get_fused_kv_buffer(0)],
        full_layer_ids=[1],
        swa_layer_ids=[0],
        num_draft_full=0,
        num_hidden_full=0,
        num_draft_swa=0,
    )
    src = groups.swa[0]
    _randomize(src, seed=7)
    pools = pl.pool_manifest(groups)
    _, swa_spans = pp_common.head_slice_spans(
        pools, groups.full + groups.swa, [0], [2, 1], 8, HEADS, seq_along_lane=True
    )
    assert swa_spans.tag == "swa.l0"
    consumers = [
        _pool(
            TpuSeqAlongLaneKVPool, heads=2, dtype=torch.bfloat16, pages=3
        ).get_fused_kv_buffer(0)
        for _ in range(8)
    ]
    _land(swa_spans, src, consumers, [0, 2])
    for rank, consumer in enumerate(consumers):
        rows = slice(4 * rank, 4 * rank + 4)
        for d, s in ((0, 2), (2, 1)):
            want = src[s, rows].view(torch.uint8)
            assert torch.equal(consumer[d].view(torch.uint8), want), rank


def test_pd_hidden_rows_round_trip_beside_seq_along_lane_kv():
    """The hidden pool stays fused next to SEQ_ALONG_LANE KV; its per-rank slices
    must rejoin into the prefill's rows on a TP8 decode."""
    feature, pages, dst_tp = 3 * 1024, 4, 8

    def hidden(attn_tp_size, group=None):
        return PdHiddenPool(
            # The KV pool keeps one padding page past size.
            num_pages=pages + 1,
            page_size=PAGE,
            feature_size=feature,
            total_kv_heads=HEADS,
            attn_tp_size=attn_tp_size,
            attn_tp_group=group,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )

    producer = _pool(
        TpuSeqAlongLaneKVPool, heads=HEADS, dtype=torch.float8_e4m3fn, pages=pages
    )
    prefill_hidden = hidden(1)
    slots = torch.tensor([PAGE * 2 + 3, PAGE * 2 + 4, PAGE + 9])
    rows = torch.randn(len(slots), feature).to(torch.bfloat16)
    prefill_hidden.write(
        rows, segments=prefill_hidden.write_segments(slots, len(slots))
    )
    with patch.object(pd_hidden_pool, "_pd_hidden_pool", prefill_hidden):
        groups = raiden_common.get_raiden_kv_caches(producer, draft_pool=producer)
    assert groups.seq_along_lane and groups.num_hidden_full == 1
    pools = pl.pool_manifest(groups)
    assert pools[-1]["tag"] == "pdhid.fa.l0"
    src_blocks = [2, 1]
    spans = pp_common.head_slice_spans(
        pools, groups.full, src_blocks, [], dst_tp, HEADS, seq_along_lane=True
    )
    decode = [hidden(dst_tp) for _ in range(dst_tp)]
    dst_blocks = [1, 3]
    _land(spans[-1], groups.full[-1], [d.buffer for d in decode], dst_blocks)

    # The decode's slots for the same tokens, after the page renumbering.
    remap = dict(zip(src_blocks, dst_blocks))
    loc = torch.tensor([remap[int(s) // PAGE] * PAGE + int(s) % PAGE for s in slots])
    locals_ = [
        d.buffer.flatten(0, 1).index_select(0, loc).reshape(len(slots), -1)
        for d in decode
    ]
    group = SimpleNamespace(
        all_gather=lambda tensor, dim: torch.cat(
            [t.unsqueeze(0) for t in locals_], dim=dim
        )
    )
    reader = hidden(dst_tp, group)
    reader.buffer = decode[0].buffer
    assert torch.equal(reader.read(loc), rows)


def test_fingerprint_names_the_layout_and_ignores_per_rank_heads():
    getter = lambda t: ((4, 3, 2, 1, 0), ((8, 128),), 8)  # noqa: E731

    def fingerprint(pool_cls, heads, seq_along_lane):
        dev = _pool(pool_cls, heads=heads, dtype=torch.float8_e4m3fn, pages=2)
        groups = raiden_common.get_raiden_kv_caches(dev)
        assert groups.seq_along_lane == seq_along_lane
        return pl.layout_fingerprint(
            pl.pool_manifest(groups),
            groups.full,
            getter,
            ships_pd_hidden=False,
            seq_along_lane=seq_along_lane,
        )

    prefill = fingerprint(TpuSeqAlongLaneKVPool, HEADS, True)
    assert prefill == fingerprint(TpuSeqAlongLaneKVPool, 2, True)
    assert prefill != fingerprint(TpuFusedMHATokenToKVPool, HEADS, False)
    # The same tensors read as fused pages must not fingerprint alike.
    sal = _pool(TpuSeqAlongLaneKVPool, heads=HEADS, dtype=torch.bfloat16, pages=2)
    groups = raiden_common.get_raiden_kv_caches(sal)
    pools = pl.pool_manifest(groups)
    assert pl.layout_fingerprint(
        pools, groups.full, getter, ships_pd_hidden=False, seq_along_lane=True
    ) != pl.layout_fingerprint(
        pools, groups.full, getter, ships_pd_hidden=False, seq_along_lane=False
    )


@pytest.mark.parametrize("producer_sal", [True, False])
def test_consumer_in_the_other_layout_is_refused_at_planning(producer_sal):
    pipe = object.__new__(pl.PipelineTransfer)
    pipe.seq_along_lane = producer_sal
    pipe.replies = {7: {"tcp://d:1": dict(attn_tp_size=8, seq_along_lane=producer_sal)}}
    pipe._check_consumer_layout(7)
    pipe.replies[7]["tcp://d:2"] = dict(attn_tp_size=8, seq_along_lane=not producer_sal)
    with pytest.raises(ValueError, match="SGLANG_TPU_USE_SEQ_ALONG_LANE_KV"):
        pipe._check_consumer_layout(7)


def test_producer_missing_heads_is_refused():
    dev = _pool(TpuSeqAlongLaneKVPool, heads=8, dtype=torch.bfloat16, pages=2)
    groups = raiden_common.get_raiden_kv_caches(dev)
    with pytest.raises(ValueError, match="does not hold all 16 kv-heads"):
        pp_common.head_slice_spans(
            pl.pool_manifest(groups),
            groups.full,
            [0],
            [],
            8,
            HEADS,
            seq_along_lane=True,
        )


def test_draft_pool_in_another_layout_is_refused():
    target = _pool(TpuSeqAlongLaneKVPool, heads=HEADS, dtype=torch.bfloat16, pages=2)
    draft = _pool(TpuFusedMHATokenToKVPool, heads=8, dtype=torch.bfloat16, pages=2)
    with patch.object(pd_hidden_pool, "_pd_hidden_pool", None):
        with pytest.raises(NotImplementedError, match="SEQ_ALONG_LANE"):
            raiden_common.get_raiden_kv_caches(target, draft_pool=draft)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
