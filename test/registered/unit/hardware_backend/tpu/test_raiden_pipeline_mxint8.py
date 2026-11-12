"""MXINT8 KV pages crossing a Raiden pipeline transfer.

The spans address the pages' tiled device bytes, so the checks model the int8
``T(32,128)(4,1)`` tiling in numpy: build each page logically, tile it, apply the
spans byte for byte, untile each destination page and compare it with the page
a rank holding only the kv-heads it is assigned would have.
"""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu import kv_mxint8
from sglang_tpu.disaggregation.raiden import common
from sglang_tpu.disaggregation.raiden import pipeline as pl
from sglang_tpu.disaggregation.raiden import pp_common, pp_mxint8
from sglang_tpu.disaggregation.raiden.reshard_schedule import head_slice_params

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

INT8_LAYOUT = ((2, 1, 0), ((32, 128), (4, 1)), 8)


def _getter(tensor):
    if len(tensor.shape) == 4:
        return ((3, 2, 1, 0), ((16, 128),), 16)
    return INT8_LAYOUT


def _groups(full, swa, full_ids, swa_ids, num_draft_full=0):
    return SimpleNamespace(
        full=full,
        swa=swa,
        full_layer_ids=full_ids,
        swa_layer_ids=swa_ids,
        num_draft_full=num_draft_full,
        num_hidden_full=0,
        num_draft_swa=0,
    )


def _tile(page):
    """Tiled bytes of a logical int8 [R, T] page: byte (b, c, s, l, j) holds
    row 32 b + 4 s + j of token 128 c + l."""
    rows, width = page.shape
    tiles = page.reshape(rows // 32, 8, 4, width // 128, 128)
    return tiles.transpose(0, 3, 1, 4, 2).reshape(-1)


def _untile(data, rows, width):
    tiles = data.reshape(rows // 32, width // 128, 8, 128, 4)
    return tiles.transpose(0, 2, 4, 1, 3).reshape(rows, width)


def _logical_page(fmt, payload, scales):
    """A page holding kv-heads whose payload rows [2A, page_size] and scale rows
    [S, page_size] are given; returns the page and a mask of its used rows."""
    heads = len(payload)
    rows, width = fmt.rows(heads), fmt.width
    page = np.full((rows, width), kv_mxint8.EXPONENT_PAD_BYTE, dtype=np.uint8)
    used = np.zeros(rows, dtype=bool)
    head_rows, head_scales = fmt.head_payload_rows, fmt.head_scale_rows
    page_payload = heads * head_rows
    for h in range(heads):
        for j in range(fmt.token_blocks):
            row = j * page_payload + h * head_rows
            page[row : row + head_rows] = payload[h][:, j * width : (j + 1) * width]
            used[row : row + head_rows] = True
        exp = fmt.token_blocks * page_payload
        if fmt.layout == "merged":
            row = exp + h * head_scales
            page[row : row + head_scales] = scales[h]
            used[row : row + head_scales] = True
            continue
        for k in range(head_scales):
            # Tile row 4 r + j holds scale row r of token block j; all four
            # rows of a scale row are kept, as the decode reads them.
            row = exp + 4 * (h * head_scales + k)
            used[row : row + 4] = True
            for j in range(fmt.token_blocks):
                page[row + j] = scales[h][k, j * 128 : (j + 1) * 128]
    return page, used


def _apply(pool_spans, src_pages, dst, writes):
    for span in pool_spans.spans:
        src = src_pages[span.src_block_ordinal]
        for i in range(span.count):
            s = span.src_offset_bytes + i * span.src_stride_bytes
            d = span.dst_offset_bytes + i * span.dst_stride_bytes
            dst[span.dst_unit_ordinal][d : d + span.size_bytes] = src[
                s : s + span.size_bytes
            ]
            writes[span.dst_unit_ordinal][d : d + span.size_bytes] += 1


@pytest.mark.parametrize(
    "layout,page_size", [("merged", 256), ("blocked", 512), ("blocked", 128)]
)
@pytest.mark.parametrize(
    "total_kv_heads,dst_tp,head_dim",
    [
        (16, 16, 128),  # one head per rank
        (8, 2, 128),  # several heads per rank
        (12, 4, 128),  # a rank's exponent rows cross a row band
        (4, 8, 128),  # each head replicated on two ranks
        (8, 4, 64),
    ],
)
def test_head_slices_rebuild_each_rank_page(
    layout, page_size, total_kv_heads, dst_tp, head_dim
):
    """Every destination byte is written exactly once, and each rank's page,
    padding aside, equals the page of a rank holding only its assigned kv-heads.
    The sliding-window pool is merged, with twice the full pool's kv-heads."""
    fmt = pp_mxint8.Mxint8PageFormat(
        layout=layout, head_dim=head_dim, group_size=32, page_size=page_size
    )
    swa_fmt = pp_mxint8.Mxint8PageFormat(
        layout="merged", head_dim=head_dim, group_size=32, page_size=page_size
    )
    draft_heads = 2
    swa_heads = 2 * total_kv_heads
    num_pages = 3
    src = torch.zeros((5, fmt.rows(total_kv_heads), fmt.width), dtype=torch.int8)
    draft = torch.zeros((5, fmt.rows(draft_heads), fmt.width), dtype=torch.int8)
    swa = torch.zeros((5, swa_fmt.rows(swa_heads), swa_fmt.width), dtype=torch.int8)
    pools = pl.pool_manifest(
        _groups([src, draft], [swa], [1, 0], [2], num_draft_full=1)
    )
    pages = list(range(num_pages))
    spans = pp_common.head_slice_spans(
        pools,
        [src, draft, swa],
        pages,
        pages,
        dst_tp,
        total_kv_heads,
        page_formats={"fa": fmt, "draft.fa": fmt, "swa": swa_fmt},
    )
    rng = np.random.default_rng(0)
    for pool_spans, heads, fmt in zip(
        spans,
        (total_kv_heads, draft_heads, swa_heads),
        (fmt, fmt, swa_fmt),
    ):
        # Payload bytes >= 128 and exponents < 128 tell filler sources apart.
        payload = rng.integers(
            128, 256, (num_pages, heads, fmt.head_payload_rows, page_size), np.uint8
        )
        scales = rng.integers(
            0, 128, (num_pages, heads, fmt.head_scale_rows, page_size), np.uint8
        )
        src_pages = [
            _tile(_logical_page(fmt, payload[i], scales[i])[0])
            for i in range(num_pages)
        ]
        per_rank = max(1, heads // dst_tp)
        dst_page_bytes = fmt.page_bytes(per_rank)
        dst = [np.zeros(num_pages * dst_page_bytes, np.uint8) for _ in range(dst_tp)]
        writes = [np.zeros(num_pages * dst_page_bytes, np.int32) for _ in range(dst_tp)]
        _apply(pool_spans, src_pages, dst, writes)
        assert pool_spans.declared_bytes == num_pages * dst_tp * dst_page_bytes
        for rank in range(dst_tp):
            assert (writes[rank] == 1).all(), rank
            first, count, _, _, _ = head_slice_params(
                src_tp=1, dst_tp=dst_tp, src_rank=0, dst_rank=rank, total_kv_heads=heads
            )
            for i in range(num_pages):
                want, used = _logical_page(
                    fmt,
                    payload[i][first : first + count],
                    scales[i][first : first + count],
                )
                got = _untile(
                    dst[rank][i * dst_page_bytes : (i + 1) * dst_page_bytes],
                    fmt.rows(per_rank),
                    fmt.width,
                )
                assert np.array_equal(got[used], want[used]), (rank, i)
                # Padding holds exponent bytes, never payload.
                assert (got[~used] < 128).all(), (rank, i)


def _fingerprint(heads, *, layout="blocked", group_size=32, page_size=512):
    fmt = pp_mxint8.Mxint8PageFormat(
        layout=layout, head_dim=128, group_size=group_size, page_size=page_size
    )
    tensor = torch.zeros((4, fmt.rows(heads), fmt.width), dtype=torch.int8)
    pools = pl.pool_manifest(_groups([tensor], [], [3], []))
    return pl.layout_fingerprint(
        pools, [tensor], _getter, ships_pd_hidden=False, page_formats={"fa": fmt}
    )


def test_fingerprint_matches_across_head_counts_and_refuses_other_formats():
    """A TP=1 producer and a TP consumer hold different rows per page and must
    match; a peer with another KV dtype, layout or group size must not."""
    producer = _fingerprint(16)
    assert producer == _fingerprint(1)
    assert producer != _fingerprint(16, layout="merged")
    assert producer != _fingerprint(16, group_size=64)
    assert producer != _fingerprint(16, page_size=256)
    bf16 = torch.zeros((4, 32, 128, 512), dtype=torch.bfloat16)
    pools = pl.pool_manifest(_groups([bf16], [], [3], []))
    assert producer != pl.layout_fingerprint(
        pools, [bf16], _getter, ships_pd_hidden=False
    )


def test_unsupported_mxint8_pages_are_refused():
    fmt = pp_mxint8.Mxint8PageFormat(
        layout="merged", head_dim=128, group_size=32, page_size=256
    )
    tensor = torch.zeros((4, fmt.rows(8), fmt.width), dtype=torch.int8)
    pools = pl.pool_manifest(_groups([tensor], [], [3], []))
    with pytest.raises(ValueError, match="needs an MXINT8 page format"):
        pl.layout_fingerprint(pools, [tensor], _getter, ships_pd_hidden=False)
    with pytest.raises(ValueError, match="needs an MXINT8 page format"):
        pp_common.head_slice_spans(pools, [tensor], [0], [], 2, 8)
    other_tiling = lambda t: ((2, 1, 0), ((8, 128), (4, 1)), 8)  # noqa: E731
    with pytest.raises(ValueError, match="int8 tiling"):
        pl.layout_fingerprint(
            pools,
            [tensor],
            other_tiling,
            ships_pd_hidden=False,
            page_formats={"fa": fmt},
        )
    # head_dim / group_size odd would put two kv-heads in one exponent sublane
    # group; such a pool still moves whole pages to a TP=1 decode.
    odd = pp_mxint8.Mxint8PageFormat(
        layout="merged", head_dim=64, group_size=64, page_size=256
    )
    odd_tensor = torch.zeros((4, odd.rows(8), odd.width), dtype=torch.int8)
    odd_pools = pl.pool_manifest(_groups([odd_tensor], [], [3], []))
    assert pp_common.part_spans(
        odd_pools, [odd_tensor], [0], [], 1, 8, page_formats={"fa": odd}, with_swa=True
    )
    with pytest.raises(ValueError, match="even head_dim / group_size"):
        pp_common.part_spans(
            odd_pools,
            [odd_tensor],
            [0],
            [],
            2,
            8,
            page_formats={"fa": odd},
            with_swa=True,
        )
    # A tensor-parallel decode refuses such a pool at startup.
    proto = SimpleNamespace(
        StartTransferRequest=SimpleNamespace(
            DESCRIPTOR=SimpleNamespace(fields_by_name={"wire_pool_indices": None})
        )
    )
    for attn_tp_size in (1, 2):
        pipe = pl.PipelineTransfer.__new__(pl.PipelineTransfer)
        pipe.manager = SimpleNamespace(
            attn_cp_size=1, attn_tp_size=attn_tp_size, pp_size=1
        )
        pipe.producer = False
        if attn_tp_size == 1:
            pipe._check_support(proto, page_formats={"fa": odd})
        else:
            with pytest.raises(ValueError, match="even head_dim / group_size"):
                pipe._check_support(proto, page_formats={"fa": odd})


def test_hybrid_pool_formats_come_from_each_sub_pool():
    """The full and sliding-window sub-pools carry their own MXINT8 layout; a
    pool without the MXINT8 arguments has no format."""

    def sub_pool(layout):
        return SimpleNamespace(
            kv_cache_dtype_str="mxint8",
            kv_layout=layout,
            kv_group_size=32,
            head_dim=128,
            page_size=256,
        )

    pool = SWAKVPool.__new__(SWAKVPool)
    pool.full_kv_pool = sub_pool("blocked")
    pool.swa_kv_pool = sub_pool("merged")
    formats = common._mxint8_page_formats(pool)
    assert {kind: fmt.layout for kind, fmt in formats.items()} == {
        "fa": "blocked",
        "swa": "merged",
    }
    assert common._mxint8_page_formats(SimpleNamespace(head_dim=128)) == {}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
