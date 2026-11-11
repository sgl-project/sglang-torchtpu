# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the DeepSeek-V4.1 TPU attention backend control plane.

Guards the per-step metadata the CSA2 and indexer kernels read (request rows,
full / SWA / indexer page tables, c1 / c2 write records, selection lengths, the
valid mask) on ragged chunked prefill, bucket-padded decode and padding rows; the
ratio-2 pair pooling and write addressing across a prefill chunk boundary against
a whole-sequence reference; and the backend's wiring of the B / C ops, with both
replaced by torch reference stubs.
"""

from sglang.test.ci.ci_register import register_cpu_ci

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from sglang.kernels.ops.attention.deepseek_v4_rope import precompute_freqs_cis
from sglang.srt.layers.attention.dsv4.dsv41_sparse import (
    DeepseekV41Compressor,
    RMSNorm,
    rope_tail,
)
from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.models.deepseek_v41.attention_backend import (
    SELECTION_WIDTH,
    DSV41TPUAttnBackend,
    apply_rope,
)

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

# Request slot 1 owns full pages 3 then 5; slot 2 owns full page 2.
# full -> SWA token mapping: full page 3 -> SWA page 1, 5 -> 2, 2 -> 3.
FULL_PAGES = {1: [3, 5], 2: [2]}
SWA_OF_FULL_PAGE = {3: 1, 5: 2, 2: 3}
MAX_CONTEXT = 1024  # 4 full-page columns
NUM_REQ_SLOTS = 4


def _req_to_token() -> torch.Tensor:
    table = torch.zeros((NUM_REQ_SLOTS, MAX_CONTEXT), dtype=torch.int32)
    for slot, pages in FULL_PAGES.items():
        for column, page in enumerate(pages):
            start = column * 256
            table[slot, start : start + 256] = page * 256 + torch.arange(256)
    return table


def _swa_mapping() -> torch.Tensor:
    mapping = torch.zeros(8 * 256, dtype=torch.int64)
    for full_page, swa_page in SWA_OF_FULL_PAGE.items():
        mapping[full_page * 256 : full_page * 256 + 256] = (
            swa_page * 256 + torch.arange(256)
        )
    return mapping


def _full_loc(slot: int, position: int) -> int:
    return FULL_PAGES[slot][position // 256] * 256 + position % 256


def _make_backend(pool, low_ratios=(1, 2)) -> DSV41TPUAttnBackend:
    backend = object.__new__(DSV41TPUAttnBackend)
    backend.device = torch.device("cpu")
    backend.req_to_token = _req_to_token()
    backend.token_to_kv_pool = pool
    backend.max_context_len = MAX_CONTEXT
    backend.page_size = 256
    backend.index_topk = 512
    backend.low_ratios = low_ratios
    backend.forward_metadata = None
    backend.rope_tables = {}
    backend.csa2_ops = None
    backend.indexer_ops = None
    return backend


def _prefill_batch():
    """Slot 1 resumes at position 250 (chunked prefix) for 10 tokens across its
    page boundary; slot 2 prefills 5 tokens; one padding token, one bucket row."""
    positions = list(range(250, 260)) + list(range(5)) + [0]
    out_cache_loc = [_full_loc(1, p) for p in range(250, 260)]
    out_cache_loc += [_full_loc(2, p) for p in range(5)] + [0]
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=3,
        _tpu_real_b=2,
        positions=torch.tensor(positions, dtype=torch.int64),
        out_cache_loc=torch.tensor(out_cache_loc, dtype=torch.int64),
        req_pool_indices=torch.tensor([1, 2, 0], dtype=torch.int64),
        extend_seq_lens=torch.tensor([10, 5, 0], dtype=torch.int32),
        extend_seq_lens_cpu=[10, 5, 0],
    )


def _decode_batch():
    """Slot 1 decodes position 511, slot 2 position 4; bucket padded to 4 rows."""
    return SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        batch_size=4,
        _tpu_real_b=2,
        positions=torch.tensor([511, 4, 0, 0], dtype=torch.int64),
        out_cache_loc=torch.tensor(
            [_full_loc(1, 511), _full_loc(2, 4), 0, 0], dtype=torch.int64
        ),
        req_pool_indices=torch.tensor([1, 2, 0, 0], dtype=torch.int64),
        extend_seq_lens=None,
        extend_seq_lens_cpu=None,
    )


class TestDSV41TPUBackendMetadata(CustomTestCase):
    def _metadata(self, forward_batch, mapping=None):
        pool = SimpleNamespace(
            full_to_swa_index_mapping=_swa_mapping() if mapping is None else mapping
        )
        backend = _make_backend(pool)
        backend.init_forward_metadata(forward_batch)
        return backend.forward_metadata

    def test_prefill_ragged_chunked_prefix(self):
        meta = self._metadata(_prefill_batch())
        self.assertEqual(meta.query_start_loc.tolist(), [0, 10, 15, 15])
        self.assertEqual(meta.token_rows.tolist(), [0] * 10 + [1] * 5 + [-1])
        self.assertEqual(meta.token_req_pool.tolist(), [1] * 10 + [2] * 5 + [0])
        self.assertEqual(meta.valid.tolist(), [True] * 15 + [False])
        self.assertEqual(
            meta.full_pages.tolist(), [[3, 5, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0]]
        )
        # 64-token SWA pages of each logical page; unmapped pages are null.
        self.assertEqual(
            meta.swa_pages.tolist(),
            [
                [4, 5, 6, 7, 8, 9, 10, 11] + [0] * 8,
                [12, 13, 14, 15] + [0] * 12,
                [0] * 16,
            ],
        )
        self.assertEqual(meta.swa_minimum.tolist(), [0, 0, 0])

        # Ratio 1: every real token writes its own record.
        c1 = meta.low_ratio[1]
        loc = meta.out_cache_loc.tolist()
        self.assertEqual(c1.out_loc.tolist(), loc[:15] + [-1])
        self.assertEqual(
            c1.lengths.tolist(), list(range(251, 261)) + list(range(1, 6)) + [0]
        )
        self.assertEqual(
            c1.index_pages.tolist(),
            [
                [6, 7, 10, 11, 0, 1, 0, 1],
                [4, 5, 0, 1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0, 1, 0, 1],
            ],
        )

        # Ratio 2: only odd positions complete a pair; record = full_loc // 2.
        c2 = meta.low_ratio[2]
        expected = []
        for slot, positions in ((1, range(250, 260)), (2, range(5))):
            expected += [_full_loc(slot, p) // 2 if p % 2 else -1 for p in positions]
        self.assertEqual(c2.out_loc.tolist(), expected + [-1])
        self.assertEqual(
            c2.out_loc.tolist()[:10], [-1, 509, -1, 510, -1, 511, -1, 640, -1, 641]
        )
        self.assertEqual(
            c2.lengths.tolist(),
            [125, 126, 126, 127, 127, 128, 128, 129, 129, 130, 0, 1, 1, 2, 2, 0],
        )
        self.assertEqual(c2.index_pages.tolist(), meta.full_pages.tolist())

    def test_records_match_page_tables(self):
        """Indexer and main-KV addressing of logical compressed id pos // r agree
        with the write record full_loc // r on every completed group."""
        for forward_batch in (_prefill_batch(), _decode_batch()):
            meta = self._metadata(forward_batch)
            for ratio, low in meta.low_ratio.items():
                main_records = 256 // ratio
                for t in range(meta.positions.shape[0]):
                    record = int(low.out_loc[t])
                    if record < 0:
                        continue
                    row = int(meta.token_rows[t])
                    logical = int(meta.positions[t]) // ratio
                    index_page = int(low.index_pages[row, logical // 128])
                    self.assertEqual(index_page * 128 + logical % 128, record)
                    main_page = int(meta.full_pages[row, logical // main_records])
                    self.assertEqual(
                        main_page * main_records + logical % main_records, record
                    )
                    self.assertEqual(record, int(meta.out_cache_loc[t]) // ratio)

    def test_decode_bucket_padding(self):
        meta = self._metadata(_decode_batch())
        self.assertEqual(meta.query_start_loc.tolist(), [0, 1, 2, 2, 2])
        self.assertEqual(meta.token_rows.tolist(), [0, 1, -1, -1])
        self.assertEqual(meta.valid.tolist(), [True, True, False, False])
        self.assertEqual(meta.low_ratio[1].out_loc.tolist(), [1535, 516, -1, -1])
        self.assertEqual(meta.low_ratio[1].lengths.tolist(), [512, 5, 0, 0])
        self.assertEqual(meta.low_ratio[2].out_loc.tolist(), [767, -1, -1, -1])
        self.assertEqual(meta.low_ratio[2].lengths.tolist(), [256, 2, 0, 0])
        self.assertEqual(tuple(meta.full_pages.shape), (4, 4))
        self.assertEqual(tuple(meta.swa_pages.shape), (4, 16))

    def test_evicted_swa_page_is_null(self):
        mapping = _swa_mapping()
        mapping[3 * 256 : 4 * 256] = 0  # full page 3's SWA peer freed
        meta = self._metadata(_decode_batch(), mapping=mapping)
        self.assertEqual(meta.swa_pages[0].tolist(), [0] * 4 + [8, 9, 10, 11] + [0] * 8)

    def test_real_token_without_slot_is_padding(self):
        batch = _decode_batch()
        batch.out_cache_loc[1] = 0
        meta = self._metadata(batch)
        self.assertEqual(meta.valid.tolist(), [True, False, False, False])
        self.assertEqual(meta.token_rows.tolist(), [0, -1, -1, -1])
        self.assertEqual(meta.low_ratio[1].out_loc.tolist(), [1535, -1, -1, -1])

    def test_rope_matches_core_rope_tail(self):
        freqs = precompute_freqs_cis(64, 1024, 65536, 160000, 16.0, 32, 1)
        x = torch.randn(5, 3, 512).to(torch.bfloat16)
        positions = torch.tensor([0, 7, 255, 256, 1023])
        expected = rope_tail(x, freqs[positions], 64)
        torch.testing.assert_close(apply_rope(x, positions, freqs), expected)


# =============================================================================
# Ratio-2 compressor across a prefill chunk boundary
# =============================================================================

HIDDEN = 16
HEAD_DIM = 512
INDEX_DIM = 128
RECORDS = 4096


class _RecordingOps:
    """Torch reference stubs for B's publish_main and C's write_index_k."""

    def __init__(self):
        self.main_writes = []
        self.index_writes = []

    def publish_main(self, words, scales, records, latent):
        live = records >= 0
        self.main_writes += records[live].tolist()
        words = words.clone()
        words[records[live].long()] = latent[live].float()
        return words, scales.clone()

    def write_index_k(self, index_k, records, keys, layout):
        live = records >= 0
        self.index_writes += records[live].tolist()
        index_k = index_k.clone()
        index_k[records[live].long()] = keys[live].float()
        return index_k


class _PairPool:
    def __init__(self):
        self.state = CompressStatePool(
            size=NUM_REQ_SLOTS * 2,
            ring_size=2,
            overlap=False,
            head_dim=HEAD_DIM,
            dtype=torch.float32,
            device="cpu",
            enable_memory_saver=False,
            ratio=2,
            request_scoped=True,
        )
        self.words = torch.zeros(RECORDS, HEAD_DIM)
        self.scales = torch.zeros(RECORDS)
        self.index_k = torch.zeros(RECORDS, INDEX_DIM)
        self.full_to_swa_index_mapping = _swa_mapping()

    def get_attention_compress_states(self, layer_id):
        return self.state

    def source_layer_of(self, layer_id):
        return 2

    def get_main_kv(self, layer_id):
        return self.words, self.scales

    def get_index_k(self, layer_id):
        return self.index_k

    def index_k_layout(self, layer_id):
        return (RECORDS // 128, 33, 0, 128)


def _ratio2_layer():
    torch.manual_seed(0)
    compressor = DeepseekV41Compressor(
        hidden_size=HIDDEN,
        head_dim=HEAD_DIM,
        compress_ratio=2,
        eps=1e-6,
        fused_compress=False,
    )
    indexer = SimpleNamespace(
        owns_k=True,
        wk=nn.Linear(HEAD_DIM, INDEX_DIM, bias=False, dtype=torch.bfloat16),
        k_norm=RMSNorm(INDEX_DIM, 1e-6),
    )
    freqs = precompute_freqs_cis(64, 1024, 65536, 160000, 16.0, 32, 1)
    return SimpleNamespace(
        layer_id=2,
        compress_ratio=2,
        compressor=compressor,
        indexer=indexer,
        freqs_cis=freqs,
    )


def _chunk(slots_positions, x_of):
    """A prefill chunk of contiguous per-request positions plus one padding row."""
    positions, locs, rows, counts, slots = [], [], [], [], []
    for slot, chunk_positions in slots_positions:
        positions += chunk_positions
        locs += [_full_loc(slot, p) for p in chunk_positions]
        rows.append(x_of[slot][chunk_positions])
        counts.append(len(chunk_positions))
        slots.append(slot)
    x = torch.cat(rows + [torch.zeros(1, HIDDEN, dtype=torch.bfloat16)])
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(slots),
        positions=torch.tensor(positions + [0]),
        out_cache_loc=torch.tensor(locs + [0]),
        req_pool_indices=torch.tensor(slots),
        extend_seq_lens=torch.tensor(counts),
        extend_seq_lens_cpu=counts,
    )
    return batch, x


class TestDSV41Ratio2AcrossChunks(CustomTestCase):
    def test_pairs_and_records_across_chunk_boundary(self):
        layer = _ratio2_layer()
        pool = _PairPool()
        ops = _RecordingOps()
        backend = _make_backend(pool, low_ratios=(2,))
        backend.csa2_ops = ops
        backend.indexer_ops = ops
        lengths = {1: 9, 2: 5}
        x_of = {
            slot: torch.randn(n, HIDDEN).to(torch.bfloat16)
            for slot, n in lengths.items()
        }
        # Slot 1 splits after its even position 4, slot 2 after its even 2: the
        # first odd token of the next chunk pairs with the carried ring row.
        chunks = (
            [(1, list(range(0, 5))), (2, list(range(0, 3)))],
            [(1, list(range(5, 9))), (2, list(range(3, 5)))],
        )
        for slots_positions in chunks:
            batch, x = _chunk(slots_positions, x_of)
            backend.forward_metadata = None
            backend.forward_low_ratio_sources(
                layer=layer,
                x=x,
                q_lora=None,
                positions=batch.positions,
                forward_batch=batch,
                run_indexer=False,
            )

        expected_records = []
        for slot, n in lengths.items():
            kv, score = layer.compressor.project(x_of[slot])
            for odd in range(1, n, 2):
                pair = slice(odd - 1, odd + 1)
                pooled = layer.compressor.pool_pairs(kv[pair][None], score[pair][None])
                latent = layer.compressor.finish(pooled)
                freqs = layer.freqs_cis[torch.tensor([odd - 1])]
                record = _full_loc(slot, odd) // 2
                expected_records.append(record)
                torch.testing.assert_close(
                    pool.words[record],
                    rope_tail(latent, freqs, 64)[0].float(),
                    msg=f"main latent of slot {slot} pair ending {odd}",
                )
                key = layer.indexer.k_norm(layer.indexer.wk(latent))
                torch.testing.assert_close(
                    pool.index_k[record],
                    rope_tail(key, freqs, 64)[0].float(),
                    msg=f"index key of slot {slot} pair ending {odd}",
                )
        # Exactly one write per completed pair; padding and even tokens write nothing.
        self.assertEqual(sorted(ops.main_writes), sorted(expected_records))
        self.assertEqual(sorted(ops.index_writes), sorted(expected_records))

    def test_ratio1_writes_every_real_token(self):
        torch.manual_seed(0)
        compressor = DeepseekV41Compressor(
            hidden_size=HIDDEN,
            head_dim=HEAD_DIM,
            compress_ratio=1,
            eps=1e-6,
            fused_compress=False,
        )
        layer = SimpleNamespace(
            layer_id=20,
            compress_ratio=1,
            compressor=compressor,
            indexer=None,
            freqs_cis=precompute_freqs_cis(64, 1024, 65536, 160000, 16.0, 32, 1),
        )
        pool = _PairPool()
        ops = _RecordingOps()
        backend = _make_backend(pool, low_ratios=(1,))
        backend.csa2_ops = ops
        batch = _prefill_batch()
        backend.init_forward_metadata(batch)
        backend.forward_metadata.gathered_main = SimpleNamespace(source_layer_id=2)
        x = torch.randn(16, HIDDEN).to(torch.bfloat16)
        backend.forward_low_ratio_sources(
            layer=layer,
            x=x,
            q_lora=None,
            positions=batch.positions,
            forward_batch=batch,
        )
        self.assertIsNone(backend.forward_metadata.gathered_main)
        self.assertEqual(ops.main_writes, batch.out_cache_loc[:15].tolist())
        latent = compressor.finish(compressor.project(x[3:4])[0])
        expected = rope_tail(latent, layer.freqs_cis[batch.positions[3:4]], 64)
        torch.testing.assert_close(
            pool.words[int(batch.out_cache_loc[3])], expected[0].float()
        )


# =============================================================================
# Indexer and attention wiring
# =============================================================================


class _IndexerStub:
    def __init__(self, width=4):
        self.calls = []
        self.width = width

    def select(
        self,
        q,
        weights,
        index_k,
        index_pages,
        requests,
        lengths,
        candidates,
        layout,
        *,
        topk,
        block_size,
        publish,
        consume,
    ):
        self.calls.append(
            dict(
                q=q,
                weights=weights,
                index_pages=index_pages,
                requests=requests,
                lengths=lengths,
                candidates=candidates,
                publish=publish,
                consume=consume,
                topk=topk,
                block_size=block_size,
            )
        )
        ids = torch.arange(self.width, dtype=torch.int32)[None]
        selected = torch.where(ids < lengths[:, None], ids, -1)
        return selected, candidates + 1


def _index_layer(layer_id, *, source, consumer):
    def wq_b(q_lora):
        return q_lora.repeat(1, 32 * 128 // q_lora.shape[1]), None

    indexer = SimpleNamespace(
        wq_b=wq_b,
        n_heads=32,
        index_head_dim=128,
        head_weights_raw=lambda x: x[:, :32].to(torch.bfloat16),
        head_weight_scale=0.5,
        candidate_topk_blocks=3,
        candidate_block_size=8,
        is_candidate_source=source,
        uses_candidates=consumer,
        owns_k=False,
    )
    return SimpleNamespace(
        layer_id=layer_id,
        compress_ratio=1,
        compressor=None,
        indexer=indexer,
        freqs_cis=precompute_freqs_cis(64, 1024, 65536, 160000, 16.0, 32, 1),
    )


class _CSA2Stub:
    def __init__(self):
        self.swa_records = None
        self.main = "unset"

    def swa_metadata(self, swa_table, query_start_loc, positions, minimum):
        meta = torch.zeros(swa_table.shape[0], 4, dtype=torch.int32)
        bounds = torch.zeros(swa_table.shape[0], 2, dtype=torch.int32)
        # Every row nonnegative, so the backend's padding mask is what drops rows.
        return meta, bounds, positions + 100

    def publish_swa(self, pages, records, kv):
        self.swa_records = records.clone()
        return pages.clone()

    def gather_attend(self, *args, main):
        out = self.attention(*args, main=main)
        q = args[0]
        return (
            out,
            q.new_zeros((q.shape[0], 512, 128)),
            q.new_zeros((q.shape[0], 512)),
            torch.zeros(q.shape[0], dtype=torch.int32),
        )

    def attend_prepared(self, *args, gathered):
        return self.attention(*args)

    def attention(
        self,
        q,
        swa_pages,
        swa_meta,
        swa_bounds,
        query_start_loc,
        positions,
        sink,
        cos_sin,
        compress_ratio,
        scale,
        main=None,
    ):
        self.main = main
        self.sink = sink
        return torch.ones_like(q)


class _PayloadCSA2Stub(_CSA2Stub):
    """Boundary reference: gathered rows are immutable snapshots, not pool aliases."""

    def __init__(self):
        super().__init__()
        self.gathers = 0
        self.reuses = 0

    def gather_attend(self, *args, main):
        self.gathers += 1
        words, scales, selected, _, _ = main
        live = selected >= 0
        payload = words[selected.clamp_min(0).long(), :1].clone()
        payload = payload.masked_fill(~live[..., None], 0)
        counts = live.sum(-1).to(torch.int32)
        gathered = (payload, scales[selected.clamp_min(0).long()].clone(), counts)
        return self._result(args[0], gathered), *gathered

    def _result(self, q, gathered):
        words, _, counts = gathered
        return q + words.sum((1, 2))[:, None, None] + counts[:, None, None]

    def attend_prepared(self, *args, gathered):
        self.reuses += 1
        return self._result(args[0], gathered)


class TestDSV41BackendWiring(CustomTestCase):
    def test_prepared_ratio1_publishes_encoded_and_indexes_prerope(self):
        """Main KV must not rotate/encode twice; index keys need pre-RoPE latent."""
        backend = self._backend()
        pool = backend.token_to_kv_pool
        pool.words = torch.zeros(RECORDS, 128, dtype=torch.int32)
        pool.scales = torch.zeros(RECORDS)
        backend.init_forward_metadata(_decode_batch())
        metadata = backend.forward_metadata
        metadata.gathered_main = SimpleNamespace(source_layer_id=2)
        table = torch.zeros(1024, 64)
        table[:, :32] = 0.5
        table[:, 32:] = 0.75
        backend.rope_tables[2] = table
        latent = torch.arange(4 * 512).reshape(4, 512).to(torch.bfloat16) / 512
        encoded = torch.arange(4 * 128, dtype=torch.int32).reshape(4, 128)
        encoded_scales = torch.arange(1, 5, dtype=torch.float32)[:, None]
        prepared = SimpleNamespace(
            weight=torch.empty(5120, 512, dtype=torch.bfloat16),
            gamma=torch.ones(512),
            eps=1e-6,
        )
        layer = SimpleNamespace(
            layer_id=2,
            compress_ratio=1,
            freqs_cis=table,
            _dsv41_compressor=prepared,
            compressor=SimpleNamespace(
                project=mock.Mock(side_effect=AssertionError("generic projection")),
                finish=mock.Mock(side_effect=AssertionError("generic finish")),
            ),
            indexer=SimpleNamespace(
                owns_k=True, wk=nn.Identity(), k_norm=nn.Identity()
            ),
        )
        pool.index_k = torch.zeros(RECORDS, 512)
        backend.indexer_ops = _RecordingOps()

        def publish(words, scales, records, payload, payload_scales):
            live = records >= 0
            out_words, out_scales = words.clone(), scales.clone()
            out_words[records[live].long()] = payload[live]
            out_scales[records[live].long()] = payload_scales[live, 0]
            return out_words, out_scales

        backend.csa2_ops = SimpleNamespace(
            publish_main_encoded=publish,
            publish_main=mock.Mock(side_effect=AssertionError("double encoding")),
        )
        x = torch.zeros(4, 5120, dtype=torch.bfloat16)
        with mock.patch(
            "sglang_tpu.models.deepseek_v41.ops.compressor.compress_rope",
            return_value=(latent, encoded, encoded_scales),
        ) as fused:
            backend._compress_and_publish(layer=layer, x=x, metadata=metadata)
        records = metadata.low_ratio[1].out_loc
        live = records >= 0
        torch.testing.assert_close(pool.words[records[live].long()], encoded[live])
        torch.testing.assert_close(
            pool.scales[records[live].long()], encoded_scales[live, 0]
        )
        expected_keys = apply_rope(latent, metadata.positions.long(), table)
        torch.testing.assert_close(
            pool.index_k[records[live].long()], expected_keys[live].float()
        )
        self.assertIsNone(metadata.gathered_main)
        args = fused.call_args.args
        self.assertIs(args[0], x)
        self.assertIs(args[3], table)
        torch.testing.assert_close(args[4], metadata.positions)
        torch.testing.assert_close(args[5], live[:, None].to(torch.int32))

    def _backend(self):
        pool = _PairPool()
        pool.swa = torch.zeros(4 * 128, 128, dtype=torch.int32)
        pool.get_swa_pages = lambda layer_id: pool.swa
        pool.main_page_records = lambda layer_id: 256
        backend = _make_backend(pool)
        backend.indexer_ops = _IndexerStub()
        backend.csa2_ops = _CSA2Stub()
        return backend

    def test_gathered_payload_reuse_and_invalidation(self):
        """Shared-source layers consume snapshots; source/selection/layout/step
        changes must not silently reuse stale payloads. Q remains layer-local.
        """
        backend = self._backend()
        ops = _PayloadCSA2Stub()
        backend.csa2_ops = ops
        pool = backend.token_to_kv_pool
        batch = _decode_batch()
        backend.init_forward_metadata(batch)
        meta = backend.forward_metadata
        selected = torch.full((4, 512), -1, dtype=torch.int32)
        selected[:, :2] = torch.tensor([1, 2])
        meta.selected[1] = selected
        pool.words[1:3] = 3
        layer = SimpleNamespace(layer_id=3, scaling=0.04, v_head_dim=HEAD_DIM)
        backend.rope_tables[3] = torch.zeros(1024, 64)
        backend.rope_tables[4] = torch.zeros(1024, 64)
        q = torch.zeros(4, 8, HEAD_DIM)
        kv = torch.zeros(4, HEAD_DIM)

        def attend(value=0, ratio=1):
            return backend.forward(
                q + value, kv, kv, layer, batch, compress_ratio=ratio
            )

        def check(out, expected):
            torch.testing.assert_close(out[:2], torch.full_like(out[:2], expected))
            self.assertTrue(torch.all(out[2:] == 0))

        check(attend(), 8)  # two records of value 3, plus exact count 2
        owner = meta.gathered_main
        pool.words[1:3] = 9  # snapshot must not alias its source pool
        layer.layer_id = 4  # same pool source, distinct attention/Q
        check(attend(1), 9)
        self.assertIs(meta.gathered_main, owner)
        self.assertEqual((ops.gathers, ops.reuses), (1, 1))
        check(attend(ratio=0), 1)
        self.assertIs(meta.gathered_main, owner)

        # Equal values in a new selection still require a fresh gather.
        meta.selected[1] = selected.clone()
        check(attend(), 20)
        self.assertEqual(ops.gathers, 2)
        pool.source_layer_of = lambda layer_id: 4
        pool.words[1:3] = 11
        check(attend(), 24)
        self.assertEqual(ops.gathers, 3)
        pool.main_page_records = lambda layer_id: 128
        pool.words[1:3] = 12
        check(attend(), 26)
        self.assertEqual(ops.gathers, 4)

        backend.init_forward_metadata(batch)
        self.assertIsNone(backend.forward_metadata.gathered_main)
        backend.forward_metadata.selected[1] = selected
        pool.words[1:3] = 13
        check(attend(), 28)
        self.assertEqual(ops.gathers, 5)

    def test_candidate_publish_and_consume(self):
        backend = self._backend()
        batch = _prefill_batch()
        x = torch.randn(16, 64)
        q_lora = torch.randn(16, 128)
        consumer = _index_layer(24, source=False, consumer=True)
        with self.assertRaises(RuntimeError):
            backend.forward_low_ratio_sources(
                layer=consumer,
                x=x,
                q_lora=q_lora,
                positions=batch.positions,
                forward_batch=batch,
            )
        source = _index_layer(20, source=True, consumer=False)
        for layer in (source, consumer):
            backend.forward_low_ratio_sources(
                layer=layer,
                x=x,
                q_lora=q_lora,
                positions=batch.positions,
                forward_batch=batch,
            )
        meta = backend.forward_metadata
        published, consumed = backend.indexer_ops.calls[-2:]
        self.assertTrue(published["publish"] and not published["consume"])
        self.assertTrue(consumed["consume"] and not consumed["publish"])
        self.assertEqual(published["candidates"].tolist(), [[-1] * 3] * 16)
        self.assertTrue(
            torch.equal(consumed["candidates"], published["candidates"] + 1)
        )
        self.assertTrue(torch.equal(published["requests"], meta.token_rows))
        self.assertTrue(torch.equal(published["lengths"], meta.low_ratio[1].lengths))
        self.assertTrue(
            torch.equal(published["index_pages"], meta.low_ratio[1].index_pages)
        )
        self.assertEqual(tuple(published["q"].shape), (16, 32, 128))
        self.assertEqual(published["weights"].dtype, torch.float32)
        self.assertEqual((published["topk"], published["block_size"]), (512, 8))
        selected = meta.selected[1]
        self.assertEqual(tuple(selected.shape), (16, SELECTION_WIDTH))
        self.assertEqual(selected[11].tolist()[:5], [0, 1, -1, -1, -1])
        self.assertTrue(torch.all(selected[:, 4:] == -1))

    def test_attention_masks_padding(self):
        backend = self._backend()
        batch = _prefill_batch()
        layer = SimpleNamespace(layer_id=3, scaling=0.04, v_head_dim=HEAD_DIM)
        backend.rope_tables[3] = torch.zeros(1024, 64)
        q = torch.randn(16, 8, HEAD_DIM).to(torch.bfloat16)
        kv = torch.randn(16, HEAD_DIM).to(torch.bfloat16)
        backend.init_forward_metadata(batch)
        selected = torch.full((16, SELECTION_WIDTH), -1, dtype=torch.int32)
        backend.forward_metadata.selected[2] = selected
        out = backend.forward(
            q,
            kv,
            kv,
            layer,
            batch,
            compress_ratio=2,
            attn_sink=torch.arange(8.0),
            save_kv_cache=False,
        )
        stub = backend.csa2_ops
        self.assertEqual(stub.swa_records.tolist()[-1], -1)
        self.assertEqual(
            stub.swa_records.tolist()[:15], (batch.positions[:15] + 100).tolist()
        )
        self.assertTrue(torch.all(out[15] == 0))
        self.assertTrue(torch.all(out[:15] == 1))
        words, scales, sel, pages, records = stub.main
        self.assertIs(sel, selected)
        self.assertIs(words, backend.token_to_kv_pool.words)
        self.assertTrue(torch.equal(pages, backend.forward_metadata.full_pages))
        self.assertEqual(records, 256)
        self.assertEqual(stub.sink.tolist(), list(range(8)))

        backend.forward(q, kv, kv, layer, batch, compress_ratio=0, attn_sink=None)
        self.assertIsNone(stub.main)


if __name__ == "__main__":
    unittest.main()
