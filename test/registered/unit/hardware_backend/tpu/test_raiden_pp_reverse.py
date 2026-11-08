"""CPU checks of the reverse (prefix-pull) Raiden pipeline span builder.

A fake decode at attention TP 8 holds each layer as eight head slices; the
spans of all eight ranks are applied byte by byte to rebuild the TP=1 prefill
stage's pages, and checked against the native planner's coverage rules.
"""

import unittest

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.raiden.pp_common import parse_req_id
from sglang_tpu.disaggregation.raiden.pp_reverse import (
    destination_args,
    fan_in_spans,
    is_pull_req_id,
    parse_pull_req_id,
    pull_req_id,
    pull_uuid,
    split_native_status,
    stage_pools,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

SRC_TP = 8


class _Tensor:
    """Byte-level stand-in for a KV page tensor (fp8: one byte per element)."""

    def __init__(self, array: np.ndarray):
        self.array = array
        self.shape = array.shape

    def element_size(self) -> int:
        return 1

    def page(self, block: int) -> bytes:
        return self.array[block].tobytes()


def _pool(tag, tensor):
    stride = int(np.prod(tensor.shape[1:]))
    return dict(tag=tag, block_stride_bytes=stride, num_blocks=tensor.shape[0])


def _sal_full(heads, blocks=6, hd_words=2, packing=4, page=128, seed=0):
    rng = np.random.default_rng(seed)
    shape = (blocks, 2 * heads, hd_words, packing, page)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def _sal_rank(full, rank):
    rows = full.shape[1] // SRC_TP
    return _Tensor(np.ascontiguousarray(full[:, rank * rows : (rank + 1) * rows]))


def _fused_full(heads, blocks=6, page=16, hd=8, seed=1):
    # FP8 packs two heads per 32-bit word: [blocks, page, heads / 2, 4, hd].
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(blocks, page, heads // 2, 4, hd), dtype=np.uint8)


def _fused_rank(full, rank):
    packs = full.shape[2] // SRC_TP
    return _Tensor(np.ascontiguousarray(full[:, :, rank * packs : (rank + 1) * packs]))


def _apply(entries_by_rank, rank_tensors, dst_live, num_dst):
    """Rebuild destination pages from every rank's spans, enforcing the planner's
    rules: global-space spans split at page boundaries, each destination page
    covered exactly once, declared bytes equal to covered bytes."""
    dst = bytearray(dst_live * num_dst)
    covered = [[] for _ in range(num_dst)]
    for rank, entry in entries_by_rank.items():
        declared = 0
        for span in entry.spans:
            assert span.dst_block_index == 0 and span.dst_unit_ordinal == 0
            page = rank_tensors[rank].page(entry.block_ids[span.src_block_ordinal])
            for repeat in range(span.count):
                src = span.src_offset_bytes + repeat * span.src_stride_bytes
                out = span.dst_offset_bytes + repeat * span.dst_stride_bytes
                dst[out : out + span.size_bytes] = page[src : src + span.size_bytes]
                block, start = divmod(out, dst_live)
                assert start + span.size_bytes <= dst_live, "span crosses a page"
                covered[block].append((start, start + span.size_bytes))
                declared += span.size_bytes
        assert declared == entry.declared_bytes
    for intervals in covered:
        until = 0
        for start, end in sorted(intervals):
            assert start == until, f"coverage gap/overlap at {start}"
            until = end
        assert until == dst_live
    return bytes(dst)


class TestFanInSpans(unittest.TestCase):
    def _check(self, full, rank_of, tag, blocks, **kwargs):
        ranks = {r: rank_of(full, r) for r in range(SRC_TP)}
        entries = {}
        for rank, tensor in ranks.items():
            pools = [_pool(tag, tensor)]
            (entry,) = fan_in_spans(
                pools,
                [tensor],
                blocks,
                blocks,
                src_tp=SRC_TP,
                src_rank=rank,
                **kwargs,
            )
            self.assertEqual(entry.tag, tag)
            self.assertEqual(entry.block_ids, blocks)
            entries[rank] = entry
        dst_live = int(np.prod(full.shape[1:]))
        rebuilt = _apply(entries, ranks, dst_live, len(blocks))
        expected = b"".join(full[b].tobytes() for b in blocks)
        self.assertEqual(rebuilt, expected)

    def test_seq_along_lane_target_pages_reassemble(self):
        self._check(
            _sal_full(16),
            _sal_rank,
            "fa.l3",
            [4, 1, 5],
            total_kv_heads=16,
            seq_along_lane=True,
        )

    def test_seq_along_lane_draft_window_uses_the_draft_heads(self):
        self._check(
            _sal_full(8, seed=3),
            _sal_rank,
            "draft.swa.l0",
            [2, 0],
            total_kv_heads=16,
            seq_along_lane=True,
            draft_total_kv_heads=8,
        )

    def test_fused_token_major_pages_reassemble(self):
        self._check(
            _fused_full(16),
            _fused_rank,
            "swa.l1",
            [3, 2],
            total_kv_heads=16,
            seq_along_lane=False,
        )

    def test_replicated_heads_are_refused(self):
        full = _sal_full(4)
        tensor = _Tensor(full[:, :1])
        with self.assertRaisesRegex(ValueError, "do not split evenly"):
            fan_in_spans(
                [_pool("fa.l0", tensor)],
                [tensor],
                [0],
                [],
                src_tp=SRC_TP,
                src_rank=0,
                total_kv_heads=4,
                seq_along_lane=True,
            )

    def test_wrong_rank_slice_is_refused(self):
        tensor = _Tensor(_sal_full(16)[:, :8])
        with self.assertRaisesRegex(ValueError, "does not hold 2 of 16"):
            fan_in_spans(
                [_pool("fa.l0", tensor)],
                [tensor],
                [0],
                [],
                src_tp=SRC_TP,
                src_rank=0,
                total_kv_heads=16,
                seq_along_lane=True,
            )


class TestStageSelection(unittest.TestCase):
    def test_stage_pools_keep_decode_order_and_refuse_unknown_tags(self):
        pools = [dict(tag=t) for t in ("fa.l0", "fa.l1", "swa.l2", "swa.l3")]
        tensors = ["t0", "t1", "t2", "t3"]
        picked, picked_tensors = stage_pools(pools, tensors, ["swa.l3", "fa.l1"])
        self.assertEqual([p["tag"] for p in picked], ["fa.l1", "swa.l3"])
        self.assertEqual(picked_tensors, ["t1", "t3"])
        with self.assertRaisesRegex(ValueError, "fa.l9"):
            stage_pools(pools, tensors, ["fa.l9"])

    def test_destination_args_split_full_and_window_pages(self):
        args = destination_args(["fa.l1", "swa.l2", "swa.l3"], [7, 8, 9], [3])
        self.assertEqual(args["transfer_pool_tags"], ["fa.l1", "swa.l2", "swa.l3"])
        self.assertEqual(args["dst_device_block_ids"], [7, 8, 9, 3, 3])
        self.assertEqual(args["dst_block_counts"], [3, 1, 1])


class TestPullIds(unittest.TestCase):
    def test_ids_round_trip_and_stay_out_of_forward_parsing(self):
        req_id = pull_req_id(1234, 5)
        self.assertTrue(is_pull_req_id(req_id))
        self.assertEqual(parse_pull_req_id(req_id), (1234, 5))
        self.assertFalse(is_pull_req_id("1234.s5"))
        self.assertNotEqual(pull_uuid(1234, 5), pull_uuid(1234, 4))
        self.assertGreater(pull_uuid(1234, 5), 0)
        # Forward parsing never sees a reverse id.
        status = (
            [type("T", (), {"req_id": "7"})()],
            [type("T", (), {"req_id": req_id})()],
            [],
        )
        forward, reverse = split_native_status(status)
        self.assertEqual([t.req_id for t in forward[0]], ["7"])
        self.assertEqual([t.req_id for t in reverse[1]], [req_id])
        self.assertEqual(parse_req_id(forward[0][0].req_id), (7, None))


if __name__ == "__main__":
    unittest.main()
