"""Unit tests for the TPU prefill context-parallel layout, strategy, attention
metadata and forward glue (CPU tensors, no JAX)."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from sglang.srt.layers.cp.base import get_cp_strategy, init_cp_strategy
from sglang.srt.layers.cp.utils import is_cp_active
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention_pallas_rpa_base import (
    PallasForwardMetadata,
    PcpRingMetadata,
    _dcp_stripe_group,
)
from sglang_tpu.attention_pallas_rpa_batched import (
    TpuPallasRpaBatchedAttnBackend,
)
from sglang_tpu.compilation.bucketer import Bucketer
from sglang_tpu.cp_forward import (
    TpuCpBatchLayout,
    all_gather_kv,
    cp_last_token_rows,
    get_tpu_cp_layout,
    install_cp_logits_forward,
    pinned_extend_seq_lens_cpu,
    prepare_tpu_cp_forward_batch,
    ring_token_bucket,
    ring_token_buckets,
    wrap_cp_embedding_forward,
)
from sglang_tpu.cp_layout import (
    build_tpu_cp_layout,
    build_tpu_cp_ring_layout,
    local_row_count,
    ring_block_lens,
    ring_block_size,
    ring_local_rows,
    ring_warmup_request_lengths,
    warmup_request_lengths,
)
from sglang_tpu.kv_row_write import kv_write_segments, write_rows_torch
from sglang_tpu.tpu_platform import TpuSRTPlatform

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_CP_FORWARD = "sglang_tpu.cp_forward"

# Two requests shorter than the 8 zigzag blocks of CP 4: the 6-token one ends
# in the late block of rank 2, the 3-token one in the early block of rank 2.
CP = 4
EXTEND = [40, 6, 3, 24]
PREFIX = [0, 7, 100, 0]
PAD_B = 5
CU_EXTEND = np.cumsum([0] + EXTEND)


def _layouts(extend=EXTEND, prefix=PREFIX, pad_b=PAD_B, bucket=32):
    return [
        build_tpu_cp_layout(
            extend_lens=extend,
            prefix_lens=prefix,
            cp_size=CP,
            cp_rank=rank,
            pad_b=pad_b,
            local_rows_padded=bucket,
        )
        for rank in range(CP)
    ]


def _expand_runs(segments, num_segments):
    """``{gathered row: KV slot}`` covered by the write runs."""
    rows = {}
    for src, dst, n in segments[:, : int(num_segments)].T.tolist():
        rows.update({src + i: dst + i for i in range(n)})
    return rows


class TestZigzagLayout(CustomTestCase):
    def test_attention_rows_cover_every_token_once(self):
        """Short requests leave zigzag blocks empty; every token must still sit in
        exactly one rank's attention row, at the causal position its kv_len ends."""
        owners = np.zeros(sum(EXTEND), dtype=np.int64)
        for host in _layouts():
            rows = host.local_rows[: host.local_len]
            self.assertTrue((host.local_rows[host.local_len :] == -1).all())
            n = host.num_rows
            q, kv = host.pseudo_q_lens, host.pseudo_kv_lens
            self.assertTrue((q[:n] > 0).all())
            self.assertEqual(int(q[n:].sum() + kv[n:].sum()), 0)
            self.assertEqual(int(q.sum()), host.local_len)
            starts = np.cumsum(q[:n]) - q[:n]
            for k in range(n):
                req = host.row_reqs[k]
                end = kv[k] - PREFIX[req]
                np.testing.assert_array_equal(
                    rows[starts[k] : starts[k] + q[k]],
                    CU_EXTEND[req] + np.arange(end - q[k], end),
                )
            np.add.at(owners, rows, 1)
        self.assertTrue((owners == 1).all())

    def test_last_token_has_exactly_one_owner(self):
        """The logits step sums each request's last-token row over the ranks, so
        exactly one rank must own it, a short request's included."""
        owners = np.zeros(PAD_B, dtype=np.int64)
        for host in _layouts():
            owners += host.last_owner_mask
            for i in np.flatnonzero(host.last_owner_mask):
                self.assertEqual(
                    host.local_rows[host.last_rows[i]], CU_EXTEND[i + 1] - 1
                )
        self.assertEqual(owners.tolist(), [1, 1, 1, 1, 0])

    def test_gathered_rows_follow_the_rank_major_all_gather(self):
        """The K/V all-gather concatenates every rank's padded local rows; each
        gathered row must name the request and offset of the token held there."""
        layouts = _layouts()
        tokens = [(i, pos) for i, length in enumerate(EXTEND) for pos in range(length)]
        expected = [
            tokens[t] if t >= 0 else (-1, -1)
            for host in layouts
            for t in host.local_rows.tolist()
        ]
        for host in layouts:
            got = list(zip(host.gathered_req.tolist(), host.gathered_pos.tolist()))
            self.assertEqual(got, expected)

    def test_rejects_batches_whose_rows_would_be_lost(self):
        with self.assertRaisesRegex(ValueError, "no token to prefill"):
            _layouts(extend=[0, 16], prefix=[0, 0])
        with self.assertRaisesRegex(ValueError, "exceed the local token bucket"):
            _layouts(extend=[64], prefix=[0], bucket=8)

    def test_warmup_lengths_fill_the_local_bucket_on_every_rank(self):
        """Warmup compiles the exact local shape a served batch pads to."""
        for local_rows, cp_size, max_reqs in ((64, 4, 8), (2048, 8, 64), (10, 4, 3)):
            lengths = warmup_request_lengths(
                local_rows=local_rows, cp_size=cp_size, max_reqs=max_reqs
            )
            self.assertLessEqual(len(lengths), max_reqs)
            for rank in range(cp_size):
                self.assertEqual(local_row_count(lengths, cp_size, rank), local_rows)

    def test_every_rank_pads_to_the_same_local_bucket(self):
        """The per-layer all-gather needs one row count on every rank, while a
        short request loads them unevenly (11, 11, 11 and 10 rows here)."""
        for rank in range(CP):
            batch = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                input_ids=torch.arange(43),
                seq_lens=torch.tensor([10, 40]),
                extend_seq_lens_cpu=[3, 40],
                extend_prefix_lens_cpu=[7, 0],
                mm_inputs=None,
                attn_cp_metadata=SimpleNamespace(),
            )
            real = prepare_tpu_cp_forward_batch(
                batch,
                t_bucketer=Bucketer(name="Token", buckets=[10, 16, 32]),
                b_bucketer=Bucketer(name="Bs", buckets=[1, 2, 4]),
                extend_bs=None,
                cp_size=CP,
                cp_rank=rank,
                device=torch.device("cpu"),
            )
            self.assertEqual(real, (43, 2))
            self.assertEqual(get_tpu_cp_layout(batch).host.local_rows_padded, 16)
            self.assertEqual(batch.attn_cp_metadata.max_rank_len, [16] * CP)
            # Request fields pad to the request bucket; token fields stay global.
            self.assertEqual(batch.seq_lens.shape[0], 4)
            self.assertEqual(batch.input_ids.shape[0], 43)

    def test_write_runs_break_at_slot_gaps_and_padding(self):
        """A run copying through a padding row would write its K/V into a real
        slot, even when the slots on either side are consecutive."""
        slots = np.array([5, 6, 7, 8, 10, 11, 12, 0, 13])
        valid = np.array([1, 1, 1, 1, 1, 1, 1, 0, 1], dtype=bool)
        segments, count = kv_write_segments(slots, valid)
        self.assertEqual(
            segments[:, :count].T.tolist(), [[0, 5, 4], [4, 10, 3], [8, 13, 1]]
        )


class TestTpuZigzagStrategy(CustomTestCase):
    def setUp(self):
        super().setUp()
        platform = patch("sglang.srt.platforms.current_platform", TpuSRTPlatform())
        platform.start()
        self.addCleanup(platform.stop)
        init_cp_strategy(enable_prefill_cp=True, cp_size=CP, cp_strategy="zigzag")
        self.addCleanup(
            init_cp_strategy, enable_prefill_cp=False, cp_size=1, cp_strategy="zigzag"
        )

    @staticmethod
    def _batch(extend_lens, num_tokens=None, mode=ForwardMode.EXTEND):
        return SimpleNamespace(
            forward_mode=mode,
            input_ids=torch.zeros(num_tokens or sum(extend_lens)),
            extend_seq_lens_cpu=extend_lens,
        )

    def test_short_request_keeps_the_batch_context_parallel(self):
        """A request shorter than 2 * cp_size tokens, such as a new turn after a
        prefix-cache hit, must not push its whole batch off the CP program."""
        self.assertTrue(is_cp_active(self._batch([40, 3])))

    def test_batch_of_only_short_requests_keeps_the_plain_program(self):
        """With every request shorter than cp_size tokens some rank would own no
        attention row; such a batch must stay on the plain program."""
        self.assertFalse(is_cp_active(self._batch([CP - 1] * 4)))
        self.assertTrue(is_cp_active(self._batch([CP] + [CP - 1] * 3)))

    def test_mixed_batch_keeps_the_plain_program(self):
        """A mixed batch's running requests are one-token extends; the batch must
        stay on the plain program, since warmup compiles CP for extends only."""
        self.assertFalse(is_cp_active(self._batch([40, 1, 1], mode=ForwardMode.MIXED)))

    def test_plain_program_stand_in_stays_off(self):
        """Inside the graph the pinned lengths alone pick the program."""
        cp = pinned_extend_seq_lens_cpu(2, cp_size=CP, cp_active=True)
        plain = pinned_extend_seq_lens_cpu(2, cp_size=CP, cp_active=False)
        self.assertTrue(is_cp_active(self._batch(cp, num_tokens=64)))
        self.assertFalse(is_cp_active(self._batch(plain, num_tokens=64)))
        decode = self._batch(None, num_tokens=64, mode=ForwardMode.DECODE)
        self.assertFalse(get_cp_strategy().can_apply(64, decode))


class TestCpMetadata(CustomTestCase):
    # Rank 1 of 4 owns zigzag blocks 1 and 6: one row for the 3-token request
    # (its block 6 is empty), two for the 40-token request behind its prefix.
    EXTEND_LENS, PREFIX_LENS, POOL_ROWS = [3, 40], [5, 100], [2, 1]
    PAGE = 16

    def _metadata(self, req_to_token, full_to_swa=None):
        backend = object.__new__(TpuPallasRpaBatchedAttnBackend)
        backend.page_size = self.PAGE
        backend.max_context_len = 256
        backend.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
        backend.token_to_kv_pool = SimpleNamespace(
            full_to_swa_index_mapping=full_to_swa
        )
        backend._is_hybrid_swa = full_to_swa is not None
        backend._host_page_cols = None
        backend._distribution_cache = {}
        host = build_tpu_cp_layout(
            extend_lens=self.EXTEND_LENS,
            prefix_lens=self.PREFIX_LENS,
            cp_size=CP,
            cp_rank=1,
            pad_b=4,
            local_rows_padded=16,
        )
        batch = SimpleNamespace(
            seq_lens=torch.zeros(4, dtype=torch.int32),
            req_pool_indices_cpu=torch.tensor(self.POOL_ROWS + [0, 0]),
        )
        layout = TpuCpBatchLayout(host=host, local_rows=torch.zeros(16).long())
        return host, backend._build_cp_metadata(batch, layout)

    @staticmethod
    def _req_to_token():
        # Pool row r maps column c to slot 512 * r + c; row 0 is the padding row.
        pool_rows = 512 * torch.arange(3, dtype=torch.int32)[:, None]
        table = pool_rows + torch.arange(256, dtype=torch.int32)
        table[0] = 0
        return table

    def test_rows_read_their_own_request(self):
        """A short request leaves fewer than two rows on a rank, so every later row
        must still read its own request's page row, KV end and write slots."""
        req_to_token = self._req_to_token()
        host, metadata = self._metadata(req_to_token)
        self.assertEqual(metadata.kv_lens.tolist(), [7, 110, 135, 0, 0, 0, 0, 0])
        self.assertEqual(metadata.cu_q_lens.tolist(), [0, 1, 6] + [11] * 6)
        self.assertEqual(metadata.distribution.tolist(), [3, 3, 3])
        pages = metadata.page_indices.reshape(8, 256 // self.PAGE)
        for row, req in enumerate([0, 1, 1]):
            page_row = req_to_token[self.POOL_ROWS[req], :: self.PAGE] // self.PAGE
            self.assertEqual(pages[row].tolist(), page_row.tolist())
        gathered = zip(host.gathered_req.tolist(), host.gathered_pos.tolist())
        expected = {
            row: 512 * self.POOL_ROWS[req] + self.PREFIX_LENS[req] + pos
            for row, (req, pos) in enumerate(gathered)
            if req >= 0
        }
        self.assertEqual(
            _expand_runs(metadata.cp_write_segments, metadata.cp_write_num_segments),
            expected,
        )

    def test_hybrid_swa_rows_and_slots_go_through_the_swa_mapping(self):
        """Sliding-window layers read and write the SWA sub-pool; a chunk with
        tokens the SWA allocator did not map must fail, warmup rows must not."""
        req_to_token = self._req_to_token()
        full_to_swa = 4096 - torch.arange(2048)
        _, metadata = self._metadata(req_to_token, full_to_swa)
        full_starts = metadata.page_indices.long() * self.PAGE
        self.assertEqual(
            metadata.swa_page_indices.tolist(),
            (full_to_swa[full_starts] // self.PAGE).tolist(),
        )
        full_rows = _expand_runs(
            metadata.cp_write_segments, metadata.cp_write_num_segments
        )
        self.assertEqual(
            _expand_runs(
                metadata.cp_swa_write_segments, metadata.cp_swa_write_num_segments
            ),
            {row: int(full_to_swa[slot]) for row, slot in full_rows.items()},
        )
        unmapped = torch.zeros(2048, dtype=torch.int64)
        with self.assertRaisesRegex(RuntimeError, "without an SWA slot"):
            self._metadata(req_to_token, unmapped)
        # Warmup rows all point at the padding slot, which is never mapped.
        _, warmup = self._metadata(torch.zeros_like(req_to_token), unmapped)
        self.assertEqual(int(warmup.cp_swa_write_num_segments), 0)


class TestCpEmbeddingAndLogits(CustomTestCase):
    TABLE = torch.arange(20, dtype=torch.float32).reshape(10, 2)

    def test_embedding_wrapper_embeds_global_ids_and_runs_local_rows(self):
        """The vocab-parallel embedding all-reduces over the group, so every rank
        embeds the same global ids; the model then runs on the rank's rows."""
        calls = []
        forward = wrap_cp_embedding_forward(
            lambda *args, **kwargs: calls.append((args, kwargs)),
            lambda ids: self.TABLE[ids],
        )
        global_ids = torch.tensor([3, 1, 4, 1, 5, 9, 0, 0])
        forward(global_ids, None, None, cp_local_rows=torch.tensor([0, 5, 2]), x=1)
        (ids, _, _), kwargs = calls[0]
        self.assertEqual(ids.tolist(), [3, 9, 4])
        self.assertTrue(torch.equal(kwargs["input_embeds"], self.TABLE[[3, 9, 4]]))
        self.assertEqual(kwargs["x"], 1)
        forward(global_ids, None, None)
        self.assertIs(calls[1][0][0], global_ids)

    def test_embedding_wrapper_fills_a_positional_input_embeds_slot(self):
        """A caller passing input_embeds positionally must not get it again as a
        keyword, and the arity stays: dynamo keys its warmups on len(args)."""
        calls = []

        def fn(input_ids, positions, forward_batch, input_embeds=None, proxy=None):
            calls.append((input_embeds, proxy))

        forward = wrap_cp_embedding_forward(fn, lambda ids: self.TABLE[ids])
        ids, proxy = torch.tensor([3, 1, 4]), object()
        forward(ids, None, None, None, proxy, cp_local_rows=torch.tensor([2]))
        self.assertTrue(torch.equal(calls[0][0], self.TABLE[[4]]))
        self.assertIs(calls[0][1], proxy)

    def test_last_token_rows_sum_only_owned_rows(self):
        """A rank zeroes the rows it does not own before the sum; an overflow in
        one of them must not turn into NaN on every rank."""
        hidden = torch.tensor([[0.0], [1.0], [0.0], [float("inf")]])
        rows = cp_last_token_rows(
            hidden,
            last_rows=torch.tensor([1, 3]),
            last_owner_mask=torch.tensor([1, 0]),
            group=SimpleNamespace(all_reduce=lambda t: t + 5.0),
        )
        self.assertEqual(rows[:, 0].tolist(), [6.0, 5.0])

    def test_logits_run_on_one_reduced_row_per_request(self):
        """The processor's last-index gather must see one query row per request;
        batches without context-parallel rows pass straight through."""
        original = MagicMock(return_value="out")
        processor = SimpleNamespace(forward=original, compute_input_logprobs=False)
        wrapper = install_cp_logits_forward(processor)
        self.assertIs(install_cp_logits_forward(processor), wrapper)
        attn = SimpleNamespace(
            cp_last_rows=torch.tensor([2, 0]), cp_last_owner_mask=torch.tensor([1, 1])
        )
        context = SimpleNamespace(attn_backend=SimpleNamespace(forward_metadata=attn))
        group = SimpleNamespace(all_reduce=lambda t: t)
        parallel = SimpleNamespace(attn_cp_group=group)
        metadata = SimpleNamespace(
            from_forward_batch=lambda batch, **kwargs: SimpleNamespace()
        )
        hidden = torch.arange(3, dtype=torch.float32)[:, None]
        batch = SimpleNamespace()
        with (
            patch(f"{_CP_FORWARD}.get_forward_context", return_value=context),
            patch(f"{_CP_FORWARD}.get_parallel", return_value=parallel),
            patch(f"{_CP_FORWARD}.ForwardBatch", SimpleNamespace),
            patch(f"{_CP_FORWARD}.LogitsMetadata", metadata),
        ):
            wrapper("ids", hidden, "head", batch)
            args = original.call_args.args
            self.assertEqual(args[1][:, 0].tolist(), [2.0, 0.0])
            self.assertEqual(args[3].extend_seq_lens.tolist(), [1, 1])
            attn.cp_last_rows = None
            wrapper("ids", hidden, "head", batch)
            original.assert_called_with("ids", hidden, "head", batch, None, None)


# ---------------------------------------------------------------------------
# Ring mode (SGLANG_TPU_PCP_MODE=ring)
# ---------------------------------------------------------------------------

# Page 16, CP 4 (8 zigzag blocks, a logical page of 64 tokens). The 200-token
# request has 32-token blocks, the last two short and empty; the 40-token one
# fits blocks 0..2 of 16 tokens, so rank 3 holds none of it; the 3-token one
# lives on rank 0 alone. Prefixes are a whole logical page, none, and a length
# that is not a multiple of anything.
RING_PAGE = 16
RING_EXTEND = [200, 40, 3]
RING_PREFIX = [128, 0, 37]
RING_PAD_B = 4
RING_BUCKET = 160
RING_CU_EXTEND = np.cumsum([0] + RING_EXTEND)


def _ring_layouts(
    extend=RING_EXTEND,
    prefix=RING_PREFIX,
    pad_b=RING_PAD_B,
    bucket=RING_BUCKET,
    cache_rows_per_req=1,
):
    return [
        build_tpu_cp_ring_layout(
            extend_lens=extend,
            prefix_lens=prefix,
            cp_size=CP,
            cp_rank=rank,
            pad_b=pad_b,
            local_rows_padded=bucket,
            page_size=RING_PAGE,
            cache_rows_per_req=cache_rows_per_req,
        )
        for rank in range(CP)
    ]


class TestRingLayout(CustomTestCase):
    def test_blocks_are_page_aligned_and_filled_in_order(self):
        """The fused kernel reads the gathered chunk by page, so a block is a
        whole number of pages; the tokens fill the blocks front to back."""
        self.assertEqual(ring_block_size(200, CP, RING_PAGE), 32)
        self.assertEqual(ring_block_lens(200, CP, RING_PAGE), [32] * 6 + [8, 0])
        self.assertEqual(ring_block_lens(40, CP, RING_PAGE), [16, 16, 8, 0, 0, 0, 0, 0])
        self.assertEqual(ring_block_lens(3, CP, RING_PAGE), [3] + [0] * 7)
        # 2 * C local rows per request on every rank.
        self.assertEqual(ring_local_rows(RING_EXTEND, CP, RING_PAGE), 64 + 32 + 32)

    def test_every_token_has_one_local_row_at_its_block_offset(self):
        layouts = _ring_layouts()
        gathered = np.concatenate([layout.local_rows for layout in layouts])
        self.assertEqual(
            sorted(gathered[gathered >= 0].tolist()), list(range(sum(RING_EXTEND)))
        )
        for rank, layout in enumerate(layouts):
            ring = layout.ring
            for i, length in enumerate(RING_EXTEND):
                block = int(ring.block_sizes[i])
                for slot, index in enumerate((rank, 2 * CP - 1 - rank)):
                    start = int(ring.req_starts[i]) + slot * block
                    rows = layout.local_rows[start : start + block]
                    real = min(max(length - index * block, 0), block)
                    first = RING_CU_EXTEND[i] + index * block
                    # Real tokens lead the block's slot; padding follows.
                    self.assertEqual(
                        rows.tolist(),
                        list(range(first, first + real)) + [-1] * (block - real),
                    )

    def test_rows_are_identical_on_every_rank_but_for_the_positions(self):
        """The ring rotates in lockstep only if every rank runs the same cache
        rows; a rank holding no token of a request still presents its rows."""
        layouts = _ring_layouts()
        first = layouts[0].ring
        self.assertEqual(first.block_sizes.tolist(), [32, 16, 16, 0])
        self.assertEqual(first.req_starts.tolist(), [0, 64, 96, 0])
        self.assertEqual(first.new_q_lens.tolist(), [32, 32, 16, 16, 16, 16, 0, 0])
        self.assertEqual(first.cache_q_lens.tolist(), [64, 32, 32, 0])
        self.assertEqual(first.cache_kv_lens.tolist(), [128 + 64, 32, 37 + 32, 0])
        self.assertEqual(first.new_kv_lens.tolist(), [328, 328, 40, 40, 40, 40, 0, 0])
        self.assertEqual(first.kv_cache_lens.tolist(), [128, 128, 0, 0, 37, 37, 0, 0])
        self.assertEqual((first.new_num_rows, first.cache_num_rows), (6, 3))
        self.assertTrue(first.has_cache)
        for rank, layout in enumerate(layouts):
            ring = layout.ring
            for name in (
                "block_sizes",
                "new_q_lens",
                "new_kv_lens",
                "kv_cache_lens",
                "new_kv_page_indices",
                "cache_q_lens",
                "cache_kv_lens",
                "cache_row_reqs",
            ):
                self.assertTrue(
                    np.array_equal(getattr(ring, name), getattr(first, name)), name
                )
            self.assertTrue(
                np.array_equal(layout.gathered_req, layouts[0].gathered_req)
            )
            late = 2 * CP - 1 - rank
            self.assertEqual(
                ring.q_positions.tolist(),
                [
                    128 + rank * 32,
                    128 + late * 32,
                    rank * 16,
                    late * 16,
                    37 + rank * 16,
                    37 + late * 16,
                    0,
                    0,
                ],
            )
            # Rows start on pages (the kernel needs 16-token-aligned starts).
            starts = np.concatenate([[0], np.cumsum(ring.new_q_lens)])[:-1]
            self.assertFalse((starts % RING_PAGE).any())

    def test_gathered_page_table_reads_the_chunk_in_sequence_order(self):
        """Page j of a row's table is where chunk tokens [j * page, (j + 1) * page)
        sit in the rank-major all-gathered buffer."""
        layouts = _ring_layouts()
        gathered = np.concatenate([layout.local_rows for layout in layouts])
        ring = layouts[1].ring
        self.assertEqual(ring.gathered_pages, CP * RING_BUCKET // RING_PAGE)
        for i, length in enumerate(RING_EXTEND):
            for row in (2 * i, 2 * i + 1):
                table = ring.new_kv_page_indices[row]
                sources = [
                    table[pos // RING_PAGE] * RING_PAGE + pos % RING_PAGE
                    for pos in range(length)
                ]
                self.assertEqual(
                    gathered[sources].tolist(),
                    list(range(RING_CU_EXTEND[i], RING_CU_EXTEND[i] + length)),
                )
                # Unused entries stay valid page indices.
                self.assertTrue(((table >= 0) & (table < ring.gathered_pages)).all())

    def test_padding_rows_are_never_keys_and_never_written(self):
        """A row's new keys are its request's first ``length`` chunk tokens; every
        padding row sits at a chunk position at or past that, and the gathered
        request / offset tables (the KV write) list real tokens only."""
        layouts = _ring_layouts()
        gathered = np.concatenate([layout.local_rows for layout in layouts])
        host = layouts[2]
        valid = host.gathered_req >= 0
        self.assertTrue(np.array_equal(valid, gathered >= 0))
        self.assertTrue(
            np.array_equal(
                RING_CU_EXTEND[host.gathered_req[valid]] + host.gathered_pos[valid],
                gathered[valid],
            )
        )
        ring = host.ring
        self.assertEqual(
            (ring.new_kv_lens - ring.kv_cache_lens)[:6].tolist(),
            [200, 200, 40, 40, 3, 3],
        )
        for i, length in enumerate(RING_EXTEND):
            block = int(ring.block_sizes[i])
            table = ring.new_kv_page_indices[2 * i]
            for pos in range(length, 2 * CP * block):
                source = table[pos // RING_PAGE] * RING_PAGE + pos % RING_PAGE
                self.assertEqual(gathered[source], -1)

    def test_last_token_has_exactly_one_owner(self):
        layouts = _ring_layouts()
        for i in range(len(RING_EXTEND)):
            owners = [rank for rank in range(CP) if layouts[rank].last_owner_mask[i]]
            self.assertEqual(len(owners), 1)
            layout = layouts[owners[0]]
            self.assertEqual(
                layout.local_rows[layout.last_rows[i]], RING_CU_EXTEND[i + 1] - 1
            )
        # 200 tokens end in block 6 (rank 1's late block), 40 in block 2, 3 in 0.
        self.assertEqual(
            [int(np.argmax([l.last_owner_mask[i] for l in layouts])) for i in range(3)],
            [1, 2, 0],
        )

    def test_cache_phase_can_reuse_the_new_token_rows(self):
        """Two cache rows per request: the same row cuts in both phases."""
        ring = _ring_layouts(cache_rows_per_req=2)[3].ring
        self.assertEqual(ring.cache_q_lens.tolist(), ring.new_q_lens.tolist())
        self.assertEqual(ring.cache_kv_lens.tolist(), [160, 160, 16, 16, 53, 53, 0, 0])
        self.assertEqual(ring.cache_row_reqs.tolist(), [0, 0, 1, 1, 2, 2, 3, 3])
        self.assertEqual(ring.cache_num_rows, 6)

    def test_first_chunk_batch_has_no_cache(self):
        ring = _ring_layouts(prefix=[0, 0, 0])[0].ring
        self.assertFalse(ring.has_cache)
        # Zero cached length: the kernel schedules no ring step for the row.
        self.assertEqual(ring.cache_kv_lens.tolist(), ring.cache_q_lens.tolist())

    def test_rejects_layouts_the_kernel_cannot_read(self):
        with self.assertRaisesRegex(ValueError, "not a multiple of the page size"):
            _ring_layouts(bucket=RING_BUCKET + 8)
        with self.assertRaisesRegex(ValueError, "above the local token bucket"):
            _ring_layouts(bucket=112)
        with self.assertRaisesRegex(ValueError, "multiple of 16"):
            build_tpu_cp_ring_layout(
                extend_lens=[40],
                prefix_lens=[0],
                cp_size=CP,
                cp_rank=0,
                pad_b=1,
                local_rows_padded=48,
                page_size=24,
            )
        with self.assertRaisesRegex(ValueError, "exceed the request bucket"):
            _ring_layouts(pad_b=2)

    def test_warmup_lengths_fill_the_bucket_with_whole_blocks(self):
        for bucket, max_reqs in ((32, 4), (64, 1), (160, 3), (160, 16)):
            lengths = ring_warmup_request_lengths(
                local_rows=bucket, cp_size=CP, page_size=RING_PAGE, max_reqs=max_reqs
            )
            self.assertLessEqual(len(lengths), max_reqs)
            self.assertEqual(ring_local_rows(lengths, CP, RING_PAGE), bucket)
            for length in lengths:
                # Every block is full, so no rank has a padding row.
                self.assertEqual(
                    ring_block_lens(length, CP, RING_PAGE),
                    [ring_block_size(length, CP, RING_PAGE)] * (2 * CP),
                )
        with self.assertRaisesRegex(ValueError, "multiple of 2 x page size"):
            ring_warmup_request_lengths(
                local_rows=48, cp_size=CP, page_size=RING_PAGE, max_reqs=2
            )

    def test_gather_mode_layout_carries_no_ring_rows(self):
        self.assertIsNone(_layouts()[0].ring)


class TestRingBuckets(CustomTestCase):
    def test_only_buckets_of_whole_block_pairs_are_usable(self):
        bucketer = Bucketer(name="Token", buckets=[16, 32, 48, 96, 160])
        self.assertEqual(ring_token_buckets(bucketer, RING_PAGE), [32, 96, 160])
        self.assertEqual(ring_token_bucket(bucketer, 33, RING_PAGE), 96)
        with self.assertRaisesRegex(ValueError, "above every usable token bucket"):
            ring_token_bucket(bucketer, 161, RING_PAGE)

    def test_prepare_buckets_the_reserved_rows(self):
        """The bucketed quantity is 2 * C per request, the same on every rank."""
        for rank in range(CP):
            batch = SimpleNamespace(
                input_ids=torch.zeros(243, dtype=torch.long),
                seq_lens=torch.tensor([328, 40, 40]),
                extend_seq_lens_cpu=list(RING_EXTEND),
                extend_prefix_lens_cpu=list(RING_PREFIX),
                attn_cp_metadata=SimpleNamespace(),
            )
            with patch(
                "sglang_tpu.compilation.compiled_forward_shim._pad_batch_fields"
            ):
                real = prepare_tpu_cp_forward_batch(
                    batch,
                    t_bucketer=Bucketer(name="Token", buckets=[64, 144, 160, 320]),
                    b_bucketer=Bucketer(name="Bs", buckets=[1, 2, 4]),
                    extend_bs=None,
                    cp_size=CP,
                    cp_rank=rank,
                    device=torch.device("cpu"),
                    ring_page_size=RING_PAGE,
                )
            self.assertEqual(real, (243, 3))
            host = get_tpu_cp_layout(batch).host
            # 128 reserved rows: 144 is not a multiple of two pages, 160 is.
            self.assertEqual(host.local_rows_padded, 160)
            self.assertIsNotNone(host.ring)
            self.assertEqual(batch.attn_cp_metadata.max_rank_len, [160] * CP)


class TestRingStrategy(TestTpuZigzagStrategy):
    """Ring mode adds a capacity rule: two page-aligned blocks per request."""

    def setUp(self):
        super().setUp()
        get_cp_strategy().configure_ring(page_size=RING_PAGE, max_local_rows=160)

    def test_batch_that_fits_a_bucket_is_context_parallel(self):
        self.assertTrue(is_cp_active(self._batch(RING_EXTEND)))

    def test_too_many_short_requests_keep_the_striped_program(self):
        """Six short requests reserve 6 * 32 rows, above the largest bucket; the
        batch runs the plain (DCP) extend program instead of failing."""
        self.assertFalse(is_cp_active(self._batch([40] * 6)))
        self.assertTrue(is_cp_active(self._batch([40] * 5)))

    def test_attached_layout_settles_the_decision(self):
        """Inside the graph the batch already carries its layout and pinned
        lengths; the capacity rule must not be re-evaluated on them."""
        batch = self._batch(
            pinned_extend_seq_lens_cpu(6, cp_size=CP, cp_active=True), num_tokens=64
        )
        self.assertFalse(is_cp_active(batch))  # 6 * 32 rows would not fit
        batch._tpu_cp_layout = object()
        self.assertTrue(is_cp_active(batch))


class TestRingMetadata(CustomTestCase):
    """Attention metadata and the owner-restricted KV write of a ring extend."""

    PAGE = RING_PAGE
    STRIDE = RING_PAGE * CP
    MAX_CONTEXT = 512
    # req_to_token rows of the three requests (row 0 is the padding row).
    POOL_ROWS = [3, 1, 2]

    def _req_to_token(self):
        """Widened (virtual) locations, as the DCP allocator hands them out:
        a logical page of ``STRIDE`` tokens per page id, position-aligned."""
        table = torch.zeros(4, self.MAX_CONTEXT, dtype=torch.int32)
        pos = torch.arange(self.MAX_CONTEXT)
        for row in (1, 2, 3):
            # Scattered, non-consecutive logical pages; page 0 is reserved.
            logical_page = 1 + row + 3 * (pos // self.STRIDE)
            table[row] = (logical_page * self.STRIDE + pos % self.STRIDE).to(
                torch.int32
            )
        return table

    def _backend(self, rank, req_to_token):
        backend = object.__new__(TpuPallasRpaBatchedAttnBackend)
        backend.page_size = self.PAGE
        backend.max_context_len = self.MAX_CONTEXT
        backend.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
        backend._is_hybrid_swa = False
        backend._host_page_cols = None
        backend._distribution_cache = {}
        backend._dcp = CP
        backend._dcp_rank = rank
        backend._page_stride = self.STRIDE
        backend._pcp_ring = True
        backend._cp_rank_t = torch.tensor([rank], dtype=torch.int32)
        return backend

    def _metadata(self, rank, req_to_token=None, pool_rows=None, **layout_kwargs):
        req_to_token = self._req_to_token() if req_to_token is None else req_to_token
        host = _ring_layouts(**layout_kwargs)[rank]
        batch = SimpleNamespace(
            seq_lens=torch.zeros(RING_PAD_B, dtype=torch.int32),
            req_pool_indices_cpu=torch.tensor((pool_rows or self.POOL_ROWS) + [0]),
        )
        layout = TpuCpBatchLayout(host=host, local_rows=torch.zeros(1).long())
        backend = self._backend(rank, req_to_token)
        return host, backend._build_cp_metadata(batch, layout)

    def test_page_tables_use_the_logical_stride_and_match_on_every_rank(self):
        req_to_token = self._req_to_token()
        pages_per_seq = self.MAX_CONTEXT // self.STRIDE
        tables = []
        for rank in range(CP):
            _, metadata = self._metadata(rank)
            ring = metadata.cp_ring
            new = ring.new_page_indices.reshape(2 * RING_PAD_B, pages_per_seq)
            cache = ring.cache_page_indices.reshape(RING_PAD_B, pages_per_seq)
            for i, pool_row in enumerate(self.POOL_ROWS):
                expected = (
                    req_to_token[pool_row, :: self.STRIDE] // self.STRIDE
                ).tolist()
                self.assertEqual(new[2 * i].tolist(), expected)
                self.assertEqual(new[2 * i + 1].tolist(), expected)
                self.assertEqual(cache[i].tolist(), expected)
            tables.append((new, cache))
            self.assertEqual(ring.cp_rank.tolist(), [rank])
        for new, cache in tables[1:]:
            self.assertTrue(torch.equal(new, tables[0][0]))
            self.assertTrue(torch.equal(cache, tables[0][1]))

    def test_lengths_are_global_and_rows_are_uniform(self):
        host, metadata = self._metadata(2)
        ring = metadata.cp_ring
        self.assertEqual(ring.new_kv_lens.tolist(), [328, 328, 40, 40, 40, 40, 0, 0])
        self.assertEqual(ring.kv_cache_lens.tolist(), [128, 128, 0, 0, 37, 37, 0, 0])
        self.assertEqual(
            ring.new_cu_q_lens.tolist(), [0, 32, 64, 80, 96, 112, 128, 128, 128]
        )
        self.assertEqual(ring.cache_cu_q_lens.tolist(), [0, 64, 96, 128, 128])
        self.assertEqual(ring.cache_kv_lens.tolist(), [192, 32, 69, 0])
        # SGLang's batched order: all live rows are MIXED rows.
        self.assertEqual(ring.new_distribution.tolist(), [6, 6, 6])
        self.assertEqual(ring.cache_distribution.tolist(), [3, 3, 3])
        self.assertEqual(
            ring.q_positions.tolist(),
            [
                128 + 2 * 32,
                128 + 5 * 32,
                2 * 16,
                5 * 16,
                37 + 2 * 16,
                37 + 5 * 16,
                0,
                0,
            ],
        )
        self.assertEqual(
            ring.new_kv_page_indices.reshape(2 * RING_PAD_B, -1).tolist(),
            host.ring.new_kv_page_indices.tolist(),
        )
        for tensor in (
            ring.cache_kv_lens,
            ring.cache_page_indices,
            ring.new_kv_page_indices,
            ring.q_positions,
        ):
            self.assertEqual(tensor.dtype, torch.int32)
        # The shared fields describe the new-token rows; logits read these.
        self.assertIs(metadata.kv_lens, ring.new_kv_lens)
        self.assertEqual(metadata.cp_last_owner_mask.tolist(), [0, 1, 0, 0])
        self.assertEqual(metadata.cp_last_rows.tolist(), [0, 64 + 8 - 1, 0, 0])

    def test_each_rank_writes_exactly_the_pages_it_owns(self):
        """Position ``pos`` belongs to rank ``(pos // page) % cp`` and lands in
        that rank's pool at ``page_table[pos // stride] * page + pos % page``
        (the same rule DCP decode reads the stripes with). Run the write on
        every rank: each token is stored once, on its owner, and nothing else
        (padding rows, other ranks' pages) is written anywhere."""
        req_to_token = self._req_to_token()
        layouts = _ring_layouts()
        # The all-gathered rows, identical on every rank: global token id + 1.
        gathered = np.concatenate([layout.local_rows for layout in layouts]) + 1
        rows = torch.from_numpy(gathered).to(torch.float32)[:, None]
        pools, bounds = [], set()
        for rank in range(CP):
            _, metadata = self._metadata(rank)
            pool = torch.zeros(40, self.PAGE, 1)
            write_rows_torch(
                metadata.cp_write_num_segments, metadata.cp_write_segments, rows, pool
            )
            pools.append(pool[..., 0])
            bounds.add(tuple(metadata.cp_write_segments.shape))
        # Same table shape on every rank: the rank is data, not structure.
        self.assertEqual(len(bounds), 1)
        expected = [torch.zeros(40, self.PAGE) for _ in range(CP)]
        for i, (length, prefix) in enumerate(zip(RING_EXTEND, RING_PREFIX)):
            pool_row = self.POOL_ROWS[i]
            for offset in range(length):
                pos = prefix + offset
                owner = (pos // self.PAGE) % CP
                page = int(req_to_token[pool_row, pos]) // self.STRIDE
                self.assertEqual(expected[owner][page, pos % self.PAGE], 0)
                expected[owner][page, pos % self.PAGE] = RING_CU_EXTEND[i] + offset + 1
        for rank in range(CP):
            self.assertTrue(torch.equal(pools[rank], expected[rank]), rank)
        self.assertEqual(
            sum(int((pool != 0).sum()) for pool in pools), sum(RING_EXTEND)
        )

    def test_partial_pages_are_written_at_their_in_page_offsets(self):
        """Request 2 continues 37 cached tokens: its first new token goes to row
        37 % 16 of the page rank 2 owns, behind the cached rows of that page."""
        _, metadata = self._metadata(2)
        runs = _expand_runs(metadata.cp_write_segments, metadata.cp_write_num_segments)
        page = int(self._req_to_token()[self.POOL_ROWS[2], 37]) // self.STRIDE
        self.assertIn(page * self.PAGE + 37 % self.PAGE, runs.values())
        self.assertNotIn(page * self.PAGE + 36 % self.PAGE, runs.values())

    def test_warmup_batch_writes_nothing(self):
        """Dummy batches point every request at the reserved page 0."""
        for rank in range(CP):
            _, metadata = self._metadata(rank, pool_rows=[0, 0, 0])
            self.assertEqual(metadata.cp_write_num_segments.tolist(), [0])
            self.assertFalse(metadata.cp_ring.cache_page_indices.any())

    def test_location_out_of_step_with_its_position_is_refused(self):
        """The kernel reads a cached key's rank and row off its position; an
        allocator that put it elsewhere in the logical page would corrupt it."""
        req_to_token = self._req_to_token()
        req_to_token[3, 128:] += self.PAGE
        with self.assertRaisesRegex(
            RuntimeError, "not aligned with its token position"
        ):
            self._metadata(0, req_to_token=req_to_token)

    def test_gather_layout_is_refused_by_a_ring_only_check(self):
        """A ring layout must not reach a backend built for gather mode."""
        host = _ring_layouts()[0]
        backend = self._backend(0, self._req_to_token())
        backend._pcp_ring = False
        batch = SimpleNamespace(
            seq_lens=torch.zeros(RING_PAD_B, dtype=torch.int32),
            req_pool_indices_cpu=torch.tensor(self.POOL_ROWS + [0]),
        )
        with self.assertRaisesRegex(RuntimeError, "not built for"):
            backend._build_cp_metadata(
                batch, TpuCpBatchLayout(host=host, local_rows=torch.zeros(1).long())
            )


class TestRingStripeGroup(CustomTestCase):
    """Which group owns the page stripes and merges decode."""

    @staticmethod
    def _parallel(*, tp_rank, attn_tp, cp, dcp):
        return SimpleNamespace(
            attn_dcp_size=dcp,
            attn_dcp_rank=tp_rank % dcp if dcp > 1 else 0,
            dcp_group="dcp-group",
            attn_cp_rank=(tp_rank // attn_tp) % cp,
            attn_cp_group="attn-cp-group",
            attn_tp_size=attn_tp,
        )

    def test_ring_pcp4_x_tp2_stripes_over_the_attention_cp_group(self):
        """tp_rank = cp_rank * 2 + tp_rank_in_attn_tp: rank 5 is CP rank 2 of
        the group {1, 3, 5, 7}, not DCP rank 1 of SGLang's group {4..7}."""
        parallel = self._parallel(tp_rank=5, attn_tp=2, cp=4, dcp=4)
        self.assertEqual(
            _dcp_stripe_group(parallel, pcp_ring=True), (2, "attn-cp-group", True)
        )

    def test_ring_pcp8_is_unchanged(self):
        parallel = self._parallel(tp_rank=5, attn_tp=1, cp=8, dcp=8)
        self.assertEqual(
            _dcp_stripe_group(parallel, pcp_ring=True), (5, "attn-cp-group", True)
        )

    def test_plain_dcp_keeps_sglang_dcp_group(self):
        parallel = self._parallel(tp_rank=5, attn_tp=8, cp=1, dcp=4)
        self.assertEqual(
            _dcp_stripe_group(parallel, pcp_ring=False), (1, "dcp-group", False)
        )
        parallel = self._parallel(tp_rank=0, attn_tp=8, cp=1, dcp=1)
        self.assertEqual(_dcp_stripe_group(parallel, pcp_ring=False), (0, None, False))


class TestRingForward(CustomTestCase):
    """One layer of a ring extend: gather, read-only ring op, owned-page write."""

    def test_gathers_quantizes_writes_then_calls_the_op(self):
        events = []
        tokens, heads, dim = 4, 2, 8
        k = torch.arange(tokens * heads * dim, dtype=torch.float32).reshape(
            tokens, heads, dim
        )
        v = -k

        def all_gather_into_tensor(out, local):
            events.append("gather")
            out.copy_(torch.cat([local + 1000 * rank for rank in range(CP)]))

        group = SimpleNamespace(
            world_size=CP, all_gather_into_tensor=all_gather_into_tensor
        )

        def quantize(k_in, v_in):
            events.append("quantize")
            return k_in * 0.5, v_in * 0.5

        kv_cache = torch.zeros(3, 4, 2 * heads, 1, dim)
        snapshots = {}

        def ring_op(*args):
            events.append("op")
            snapshots["args"] = args
            snapshots["pool_at_call"] = args[3].clone()
            return args[0] + 1

        ring = PcpRingMetadata(
            **{
                name: torch.tensor([index], dtype=torch.int32)
                for index, name in enumerate(
                    (
                        "cache_kv_lens",
                        "cache_page_indices",
                        "cache_cu_q_lens",
                        "cache_distribution",
                        "new_kv_lens",
                        "new_page_indices",
                        "new_cu_q_lens",
                        "new_distribution",
                        "new_kv_page_indices",
                        "q_positions",
                        "kv_cache_lens",
                        "cp_rank",
                    )
                )
            }
        )
        # Gathered rows 5..6 (rank 1's rows 1..2) go to pool rows 9..10.
        segments = torch.tensor([[5], [9], [2]], dtype=torch.int32)
        backend = object.__new__(TpuPallasRpaBatchedAttnBackend)
        backend._pcp_ring_op = ring_op
        backend.forward_metadata = PallasForwardMetadata(
            kv_lens=ring.new_kv_lens,
            cu_q_lens=ring.new_cu_q_lens,
            cu_kv_lens=ring.new_cu_q_lens,
            page_indices=ring.new_page_indices,
            distribution=ring.new_distribution,
            active_num_tokens=torch.tensor(0),
            cp_write_segments=segments,
            cp_write_num_segments=torch.tensor([1], dtype=torch.int32),
            cp_ring=ring,
        )
        q = torch.ones(tokens, 4, dim)
        parallel = SimpleNamespace(attn_cp_group=group)
        with patch(
            "sglang_tpu.attention_pallas_rpa_base.get_parallel", return_value=parallel
        ):
            out = backend._forward_pcp_ring(q, k, v, kv_cache, quantize, 0.25, 0.5, 2.0)

        self.assertEqual(events, ["gather", "gather", "quantize", "op"])
        self.assertTrue(torch.equal(out, q + 1))
        args = snapshots["args"]
        # The op gets the rank-major gathered K/V, quantized after the gather.
        k_all = torch.cat([k + 1000 * rank for rank in range(CP)]) * 0.5
        v_all = torch.cat([v + 1000 * rank for rank in range(CP)]) * 0.5
        self.assertTrue(torch.equal(args[1], k_all))
        self.assertTrue(torch.equal(args[2], v_all))
        # Twelve metadata tensors in the op's order, then the static scales.
        self.assertEqual([int(t) for t in args[4:16]], list(range(12)))
        self.assertEqual(args[16:], (0.25, 0.5, 2.0))
        # The owned rows were written before the op (so XLA updates the pool in
        # place instead of copying it after the op's read), K and V
        # interleaved, and the op did not change the pool.
        self.assertTrue(torch.equal(snapshots["pool_at_call"], kv_cache))
        flat = kv_cache.flatten(0, 1)
        for row, source in ((9, 5), (10, 6)):
            self.assertTrue(torch.equal(flat[row, 0::2, 0], k_all[source]))
            self.assertTrue(torch.equal(flat[row, 1::2, 0], v_all[source]))
        untouched = torch.ones(12, dtype=torch.bool)
        untouched[9:11] = False
        self.assertFalse(flat[untouched].any())

    def test_all_gather_kv_is_the_gather_half_of_the_gather_mode_write(self):
        """Without a quantizer the gathered rows come back as they are."""
        local = torch.arange(6, dtype=torch.float32).reshape(3, 1, 2)
        group = SimpleNamespace(
            world_size=2,
            all_gather_into_tensor=lambda out, x: out.copy_(torch.cat([x, x + 10])),
        )
        k_all, v_all = all_gather_kv(k=local, v=local * 2, group=group, quantize=None)
        self.assertTrue(torch.equal(k_all, torch.cat([local, local + 10])))
        self.assertTrue(torch.equal(v_all, torch.cat([local * 2, local * 2 + 10])))


if __name__ == "__main__":
    unittest.main()
