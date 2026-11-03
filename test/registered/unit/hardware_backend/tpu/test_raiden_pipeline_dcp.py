"""A Raiden pipeline transfer into a decode-context-parallel (DCP) decode.

Applies the spans part_spans emits for a DCP consumer to per-rank destination
pages, byte for byte, and checks every live token lands where a DCP decode stores
it: DCP rank ``p % D`` of every kv-head group, row ``(p % (P*D)) // D`` of the DCP
page holding ``p``.
"""

import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.raiden.pipeline import PipelineTransfer
from sglang_tpu.disaggregation.raiden.pp_common import (
    dcp_page_count,
    dcp_token_spans,
    part_destination_blocks,
    part_spans,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

PAGE, HEADS, HEAD_DIM = 8, 4, 2


def _tensor(heads=HEADS):
    # Fused layout [pages, page_tokens, kv_heads * 2 / packing, packing, head_dim].
    return SimpleNamespace(shape=(64, PAGE, heads, 2, HEAD_DIM), element_size=lambda: 2)


class TestDcpTokenSpans(unittest.TestCase):
    def _case(self, tokens, dcp, tp, window=None, twin=False):
        """Move one request and compare every live token with the DCP layout.

        ``twin``: the window moves alone, as on its own engine (pp_swa), so its
        spans and destination are placed by the request's full page count.
        """
        rng = np.random.default_rng(tokens * 131 + dcp * 7 + tp)
        kv_tp = tp // dcp
        head = 2 * HEAD_DIM * 2
        src_row, dst_row = HEADS * head, HEADS // kv_tp * head
        n = -(-tokens // PAGE)
        first = max(0, (tokens - window) // PAGE) if window else 0
        full = list(range(100, 100 + n))
        swa = full[first:] if window else []
        if twin:
            tags = ["swa.l1"]
        else:
            tags = ["fa.l0"] + (["swa.l1"] if window else [])
        pools = [{"tag": t, "block_stride_bytes": PAGE * dst_row} for t in tags]
        tensors = [_tensor() for _ in tags]
        pages = {
            s: rng.integers(0, 256, PAGE * src_row, dtype=np.uint8) for s in range(n)
        }
        spans = part_spans(
            pools,
            tensors,
            [] if twin else full,
            swa,
            tp,
            HEADS,
            with_swa=True,
            dcp_size=dcp,
            full_count=n,
        )
        # The decode holds one more DCP page than the source covers (the next token).
        dst_full = list(range(-(-n // dcp) + 1))
        dst_swa = list(range((n - 1) // dcp - first // dcp + 2)) if window else []
        blocks = part_destination_blocks(
            pools,
            [] if twin else dst_full,
            dst_swa,
            start=0,
            count=0 if twin else n,
            source_swa_count=len(swa),
            dcp_size=dcp,
            full_count=n,
        )
        counts = dict(zip(blocks["transfer_pool_tags"], blocks["dst_block_counts"]))
        self.assertEqual(blocks["dst_skip_bytes"], [0] * len(tags))
        for pool in spans:
            is_swa = pool.tag.startswith("swa.")
            base = first if is_swa else 0
            space = counts[pool.tag] * PAGE * dst_row
            got = {u: np.zeros(space, dtype=np.uint8) for u in range(tp)}
            covered = {u: [] for u in range(tp)}
            for span in pool.spans:
                src = pages[base + span.src_block_ordinal]
                for k in range(span.count):
                    s = span.src_offset_bytes + k * span.src_stride_bytes
                    d = span.dst_offset_bytes + k * span.dst_stride_bytes
                    got[span.dst_unit_ordinal][d : d + span.size_bytes] = src[
                        s : s + span.size_bytes
                    ]
                    covered[span.dst_unit_ordinal].append((d, d + span.size_bytes))
            self.assertEqual(
                pool.declared_bytes,
                sum(e - b for intervals in covered.values() for b, e in intervals),
            )
            extents = set()
            for intervals in covered.values():
                intervals.sort()
                self.assertEqual(intervals[0][0], 0)
                for (_, end), (start, _) in zip(intervals, intervals[1:]):
                    self.assertEqual(end, start, "gap or overlap in a unit's coverage")
                extents.add(intervals[-1][1])
            self.assertEqual(len(extents), 1, "units must cover the same extent")
            self.assertLessEqual(extents.pop(), space)
            # Every live position must sit where the DCP decode stores it.
            live = range(max(0, tokens - window), tokens) if is_swa else range(tokens)
            for p in live:
                row = pages[p // PAGE][(p % PAGE) * src_row : (p % PAGE + 1) * src_row]
                for group in range(kv_tp):
                    u = group * dcp + p % dcp
                    j = p // (PAGE * dcp) - (first // dcp if is_swa else 0)
                    at = j * PAGE * dst_row + (p % (PAGE * dcp)) // dcp * dst_row
                    np.testing.assert_array_equal(
                        got[u][at : at + dst_row],
                        row[group * dst_row : (group + 1) * dst_row],
                        err_msg=f"{pool.tag} p={p} u={u}",
                    )

    def test_full_pool_matches_the_dcp_layout(self):
        for tokens in (1, 7, 8, 9, 15, 16, 17, 40, 64, 65):
            for dcp in (2, 4, 8):
                for tp in (dcp, 2 * dcp):
                    with self.subTest(tokens=tokens, dcp=dcp, tp=tp):
                        self._case(tokens, dcp, tp)

    def test_sliding_window_starting_mid_dcp_page(self):
        for tokens in (9, 17, 30, 41, 64, 70):
            for dcp in (2, 4):
                for tp in (dcp, 2 * dcp):
                    with self.subTest(tokens=tokens, dcp=dcp, tp=tp):
                        self._case(tokens, dcp, tp, window=6)

    def test_window_twin_is_placed_by_the_full_page_count(self):
        for tokens in (9, 17, 30, 41, 64, 70):
            for dcp in (2, 4):
                with self.subTest(tokens=tokens, dcp=dcp):
                    self._case(tokens, dcp, 2 * dcp, window=6, twin=True)

    def test_page_count(self):
        self.assertEqual(
            dcp_page_count(full_count=9, swa_count=0, is_swa=False, dcp_size=4), 3
        )
        # A window of pages 3..8 spans DCP pages 0..2.
        self.assertEqual(
            dcp_page_count(full_count=9, swa_count=6, is_swa=True, dcp_size=4), 3
        )
        self.assertEqual(
            dcp_page_count(full_count=9, swa_count=0, is_swa=True, dcp_size=4), 0
        )

    def test_page_not_a_multiple_of_dcp_is_rejected(self):
        pools = [{"tag": "fa.l0", "block_stride_bytes": 0}]
        with self.assertRaisesRegex(ValueError, "not a multiple of DCP"):
            dcp_token_spans(
                pools,
                [_tensor()],
                [1],
                [],
                full_count=1,
                dst_attn_tp=3,
                dcp_size=3,
                total_kv_heads=HEADS,
            )

    def test_other_kv_layouts_are_rejected(self):
        pools = [{"tag": "fa.l0", "block_stride_bytes": 0}]
        head_major = SimpleNamespace(shape=(64, 2 * HEADS, HEAD_DIM, PAGE))
        with self.assertRaisesRegex(ValueError, "fused KV layout"):
            dcp_token_spans(
                pools,
                [head_major],
                [1],
                [],
                full_count=1,
                dst_attn_tp=2,
                dcp_size=2,
                total_kv_heads=HEADS,
            )

    def test_early_parts_are_rejected_at_the_decode(self):
        pools = [{"tag": "fa.l0", "block_stride_bytes": 1}]
        with self.assertRaisesRegex(ValueError, "whole transfers"):
            part_destination_blocks(
                pools, [0, 1], [], start=2, count=2, source_swa_count=None, dcp_size=2
            )


class TestPipelineDcpHandshake(unittest.TestCase):
    """The decode reports its DCP size; the producer slices for it and refuses
    early parts."""

    def _producer(self, metas):
        pipe = PipelineTransfer.__new__(PipelineTransfer)
        pipe.replies = {7: dict(enumerate(metas))}
        pipe.pools = [{"tag": "fa.l0", "block_stride_bytes": PAGE * 64}]
        pipe.tensors = [_tensor()]
        pipe.manager = SimpleNamespace(kv_args=SimpleNamespace(total_kv_head_num=HEADS))
        pipe.draft_kv_heads = pipe.draft_swa_kv_heads = None
        pipe.swa = None
        pipe.unit = "unit"
        pipe.client = Mock()
        pipe._fail_room = Mock()
        return pipe

    def test_producer_slices_for_the_consumer_dcp(self):
        pipe = self._producer([dict(attn_tp_size=4, attn_tp_rank=0, dcp_size=2)])
        ok = pipe._register_blocks(
            7, req_id="7", uuid=7, full=[3, 4, 5], swa=[], with_swa=True, start=0
        )
        self.assertTrue(ok)
        (pool,) = pipe.client.register_request_blocks.call_args.kwargs["pool_spans"]
        # One span per source page and destination rank, PAGE // dcp rows each.
        self.assertEqual(len(pool.spans), 3 * 4)
        self.assertEqual({span.count for span in pool.spans}, {PAGE // 2})

    def test_producer_refuses_early_parts_for_a_dcp_consumer(self):
        pipe = self._producer([dict(attn_tp_size=4, attn_tp_rank=0, dcp_size=2)])
        ok = pipe._register_blocks(
            7, req_id="7.p0", uuid=1, full=[3], swa=[], with_swa=False, start=2
        )
        self.assertFalse(ok)
        self.assertIn(
            "SGLANG_RAIDEN_PIPELINE_EARLY_PARTS", pipe._fail_room.call_args[0][1]
        )
        pipe.client.register_request_blocks.assert_not_called()

    def test_consumers_must_agree_on_dcp(self):
        pipe = self._producer(
            [
                dict(attn_tp_size=4, attn_tp_rank=0, dcp_size=2),
                dict(attn_tp_size=4, attn_tp_rank=1),
            ]
        )
        with self.assertRaisesRegex(ValueError, "disagree on attention TP or DCP"):
            pipe._dst_layout(7)

    def test_decode_reports_its_dcp_size(self):
        pipe = PipelineTransfer.__new__(PipelineTransfer)
        pipe.dcp_size = 2
        pipe.lock = threading.RLock()
        pipe.receives = {}
        pipe.manager = SimpleNamespace(
            attn_tp_size=4,
            attn_tp_rank=1,
            local_ip="127.0.0.1",
            rank_port=1234,
            _timeout_s=60,
        )
        pipe._send = Mock()
        pipe.register_decode(
            9, [1], [], 0, [dict(pp_rank=0, rank_ip="127.0.0.2", rank_port=4321)]
        )
        consumer = json.loads(pipe._send.call_args[0][1][3])
        self.assertEqual(consumer["dcp_size"], 2)


if __name__ == "__main__":
    unittest.main()
