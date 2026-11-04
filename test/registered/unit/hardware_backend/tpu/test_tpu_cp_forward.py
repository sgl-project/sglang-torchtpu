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
from sglang_tpu.attention_pallas_rpa_batched import (
    TpuPallasRpaBatchedAttnBackend,
)
from sglang_tpu.compilation.bucketer import Bucketer
from sglang_tpu.cp_forward import (
    TpuCpBatchLayout,
    cp_last_token_rows,
    get_tpu_cp_layout,
    install_cp_logits_forward,
    pinned_extend_seq_lens_cpu,
    prepare_tpu_cp_forward_batch,
    wrap_cp_embedding_forward,
)
from sglang_tpu.cp_layout import (
    build_tpu_cp_layout,
    local_row_count,
    warmup_request_lengths,
)
from sglang_tpu.kv_row_write import kv_write_segments
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


if __name__ == "__main__":
    unittest.main()
