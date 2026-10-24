"""CPU unit tests for the decode-context-parallel (DCP) metadata of the TPU
batched RPA attention backend, and for the KV-slot parity the owner-masked write
relies on.

Covered here:

* ``_build_metadata`` under ``dcp=2`` for decode and extend: ``kv_lens`` holds
  the per-rank LOCAL lengths, ``page_indices`` uses the widened page stride,
  and the new ``dcp_kv_lens_global`` / ``dcp_prefix_kv_lens`` /
  ``dcp_cp_rank`` fields carry the global lengths, the local cached-prefix
  lengths and the rank constant.
* The per-request page-table capacity check measures the LOCAL length, so a
  sequence longer than one rank's table but shorter than the group's is served.
* The warmup dummy batch (every seq_len 1) shards to one token on rank 0 and
  an empty shard everywhere else, and an all-chunk extend has an empty prefix.
* ``dcp=1`` leaves the metadata byte-for-byte as the non-DCP path builds it.
* A non-causal layer (cross-attention, ``ENCODER_ONLY``) is refused under DCP.
* A layer whose K/V heads differ from the pool's, or do not divide its query
  heads, is refused under DCP.
* Slot parity over ``TpuPagedTokenToKVPoolAllocator`` with a widened page:
  every token's location satisfies ``loc % dcp == position % dcp`` across
  ``alloc_extend`` (fresh, chunked and page-continuing) and ``alloc_decode``
  sequences, so a logical (widened) page is position-aligned and
  ``build_owner_write_spans`` can claim each token on exactly one rank -- the
  one owning its ``page_size`` run -- at physical slot
  ``(loc // (page_size * dcp)) * page_size + position % page_size``.

Everything here is pure torch-CPU: the backend subclass skips the
JAX-requiring kernel-op registration, as in ``test_pallas_rpa_batched_backend.py``.
"""

import types
import unittest

import torch

from sglang.srt import runtime_context as rc
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.allocator_tpu import (
    TpuPagedTokenToKVPoolAllocator,
)
from sglang_tpu.attention_pallas_rpa_batched import (
    PallasForwardMetadata,
    TpuPallasRpaBatchedAttnBackend,
)
from sglang_tpu.attention_pallas_rpa_dcp import (
    build_owner_write_spans,
    max_write_spans,
)
from sglang_tpu.host_metadata_cache import HostMetadataCache

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit", chips=1)

try:
    # The layers.dcp package init pulls in the Triton DCP kernels and Triton is
    # not a TPU dependency, so a Triton-less image skips the cases that need
    # them instead of failing to collect the file.
    from sglang.srt.layers.dcp.layout import get_dcp_lens

    _DCP_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on the image
    get_dcp_lens = None
    _DCP_IMPORT_ERROR = exc

_requires_dcp_helpers = unittest.skipIf(
    _DCP_IMPORT_ERROR is not None,
    f"sglang.srt.layers.dcp is unavailable: {_DCP_IMPORT_ERROR}",
)


class _DcpBackendUnderTest(TpuPallasRpaBatchedAttnBackend):
    """Backend instance that skips the JAX-requiring kernel-op registration.

    ``__init__`` in the production class builds the Pallas custom ops, which
    imports JAX + torch_tpu. The DCP state the metadata builder reads is set
    here directly; the tests additionally publish the same values on the
    parallel bag so a builder that reads ``get_parallel()`` live sees them too.
    """

    def __init__(
        self,
        page_size: int,
        req_to_token: torch.Tensor,
        dcp: int,
        dcp_rank: int,
        max_context_len: int = None,
    ) -> None:
        AttentionBackend.__init__(self)
        self.device = req_to_token.device
        self.page_size = page_size
        self.speculative_step_id = 0
        self.max_context_len = (
            max_context_len if max_context_len is not None else req_to_token.shape[1]
        )
        self.req_to_token_pool = types.SimpleNamespace(req_to_token=req_to_token)
        self.forward_metadata = None
        self.decode_query_size = 1
        self._op = None
        self._dcp_op = None
        self._distribution_cache = {}
        self._host_metadata_cache = HostMetadataCache()
        self._host_page_cols = None
        # Non-SWA: skips the hybrid-SWA page remap in _build_metadata(_host).
        self._is_hybrid_swa = False
        self._dcp = dcp
        self._dcp_rank = dcp_rank
        self._dcp_group = None
        # One page spans page_size * dcp logical tokens over page_size slots.
        self._page_stride = page_size * dcp
        self._cp_rank_t = torch.tensor([dcp_rank], dtype=torch.int32)
        self._get_dcp_lens = get_dcp_lens if dcp > 1 else None


def _make_forward_batch(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    forward_mode: ForwardMode,
    extend_seq_lens: torch.Tensor = None,
):
    """Minimal fake ForwardBatch carrying only the fields the builder reads."""
    return types.SimpleNamespace(
        seq_lens=seq_lens,
        req_pool_indices=req_pool_indices,
        req_pool_indices_cpu=req_pool_indices.clone(),
        forward_mode=forward_mode,
        extend_seq_lens=extend_seq_lens,
        spec_info=None,
        seq_lens_cpu=seq_lens.clone(),
        extend_seq_lens_cpu=None,
    )


def _build_req_to_token(num_reqs: int, width: int) -> torch.Tensor:
    """req_to_token[r, s] = r * width + s, so expectations are computable."""
    return (
        torch.arange(num_reqs).unsqueeze(1) * width + torch.arange(width).unsqueeze(0)
    ).to(torch.int32)


def _build_meta(backend, fb, dcp: int, dcp_rank: int) -> PallasForwardMetadata:
    """Mirror ``init_forward_metadata`` with the DCP topology published."""
    with rc.get_parallel().override(
        dcp_enabled=dcp > 1,
        dcp_size=dcp,
        dcp_rank=dcp_rank,
        attn_dcp_size=dcp,
        attn_dcp_rank=dcp_rank,
    ):
        return backend._build_metadata(fb, backend._non_empty_batch_size(fb))


def _local_lens(lens, dcp: int, rank: int, page_size: int = 16) -> torch.Tensor:
    # Page-striped: rank r owns page p iff p % dcp == r.
    return get_dcp_lens(
        torch.tensor(lens, dtype=torch.int32), dcp, rank, interleave_size=page_size
    )


@_requires_dcp_helpers
class TestPallasRpaDcpMetadataDecode(CustomTestCase):
    """Decode under dcp=2: local kv_lens, widened page stride, rank constant."""

    PAGE_SIZE = 16
    DCP = 2
    PAGES_PER_SEQ = 4
    MAX_CONTEXT_LEN = PAGE_SIZE * DCP * PAGES_PER_SEQ  # 128
    REQ_TO_TOKEN_WIDTH = MAX_CONTEXT_LEN + 32  # SGLang adds slack columns.
    SEQ_LENS = [100, 33, 8]

    def setUp(self):
        self.req_to_token = _build_req_to_token(8, self.REQ_TO_TOKEN_WIDTH)
        self.req_pool_indices = torch.tensor([5, 2, 7], dtype=torch.int64)
        self.seq_lens = torch.tensor(self.SEQ_LENS, dtype=torch.int32)
        self.fb = _make_forward_batch(
            seq_lens=self.seq_lens,
            req_pool_indices=self.req_pool_indices,
            forward_mode=ForwardMode.DECODE,
        )

    def _meta(self, rank: int) -> PallasForwardMetadata:
        backend = _DcpBackendUnderTest(
            self.PAGE_SIZE,
            self.req_to_token,
            self.DCP,
            rank,
            max_context_len=self.MAX_CONTEXT_LEN,
        )
        return _build_meta(backend, self.fb, self.DCP, rank)

    def test_kv_lens_are_this_ranks_local_lengths(self):
        for rank in range(self.DCP):
            with self.subTest(rank=rank):
                meta = self._meta(rank)
                expected = _local_lens(self.SEQ_LENS, self.DCP, rank)
                self.assertTrue(
                    torch.equal(meta.kv_lens.to(torch.int32), expected.to(torch.int32)),
                    f"{meta.kv_lens.tolist()} != {expected.tolist()}",
                )

    def test_local_lengths_partition_the_global_lengths(self):
        totals = torch.zeros(len(self.SEQ_LENS), dtype=torch.int64)
        for rank in range(self.DCP):
            totals += self._meta(rank).kv_lens.to(torch.int64)
        self.assertTrue(torch.equal(totals, self.seq_lens.to(torch.int64)))

    def test_global_kv_lens_field_is_the_unsharded_length(self):
        meta = self._meta(0)
        self.assertIsNotNone(meta.dcp_kv_lens_global)
        self.assertEqual(meta.dcp_kv_lens_global.dtype, torch.int32)
        self.assertTrue(
            torch.equal(
                meta.dcp_kv_lens_global.to(torch.int32),
                self.seq_lens.to(torch.int32),
            )
        )

    def test_prefix_kv_lens_equal_kv_lens_in_decode(self):
        # Decode runs one pass: the cached part is the whole local shard.
        for rank in range(self.DCP):
            with self.subTest(rank=rank):
                meta = self._meta(rank)
                self.assertTrue(
                    torch.equal(
                        meta.dcp_prefix_kv_lens.to(torch.int32),
                        meta.kv_lens.to(torch.int32),
                    )
                )

    def test_cp_rank_is_a_one_element_int32_tensor(self):
        for rank in range(self.DCP):
            with self.subTest(rank=rank):
                meta = self._meta(rank)
                self.assertEqual(tuple(meta.dcp_cp_rank.shape), (1,))
                self.assertEqual(meta.dcp_cp_rank.dtype, torch.int32)
                self.assertEqual(int(meta.dcp_cp_rank[0]), rank)

    def test_page_indices_use_the_widened_stride(self):
        widened = self.PAGE_SIZE * self.DCP
        for rank in range(self.DCP):
            meta = self._meta(rank)
            self.assertEqual(
                tuple(meta.page_indices.shape),
                (len(self.SEQ_LENS) * self.PAGES_PER_SEQ,),
            )
            self.assertEqual(meta.page_indices.dtype, torch.int32)
            for i, pool_idx in enumerate(self.req_pool_indices.tolist()):
                for j in range(self.PAGES_PER_SEQ):
                    expected = int(self.req_to_token[pool_idx, j * widened]) // widened
                    got = int(meta.page_indices[i * self.PAGES_PER_SEQ + j])
                    self.assertEqual(got, expected, f"rank {rank} seq {i} page {j}")

    def test_page_table_is_identical_on_every_rank(self):
        # Pages are position-aligned, so the same page id backs the same range
        # of absolute positions on every rank of the group.
        first = self._meta(0).page_indices
        for rank in range(1, self.DCP):
            self.assertTrue(torch.equal(first, self._meta(rank).page_indices))

    def test_cu_q_lens_are_unsharded(self):
        # Queries are not sharded by DCP: one query row per decode sequence.
        meta = self._meta(0)
        expected = torch.arange(len(self.SEQ_LENS) + 1, dtype=torch.int32)
        self.assertTrue(torch.equal(meta.cu_q_lens.to(torch.int32), expected))


@_requires_dcp_helpers
class TestPallasRpaDcpMetadataExtend(CustomTestCase):
    """Extend under dcp=2: prefix lengths exclude the chunk being written."""

    PAGE_SIZE = 16
    DCP = 2
    PAGES_PER_SEQ = 8
    MAX_CONTEXT_LEN = PAGE_SIZE * DCP * PAGES_PER_SEQ  # 256
    SEQ_LENS = [40, 31, 7, 9]
    EXTEND_SEQ_LENS = [7, 5, 7, 1]

    def setUp(self):
        self.req_to_token = _build_req_to_token(8, self.MAX_CONTEXT_LEN + 32)
        self.req_pool_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        self.fb = _make_forward_batch(
            seq_lens=torch.tensor(self.SEQ_LENS, dtype=torch.int32),
            req_pool_indices=self.req_pool_indices,
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens=torch.tensor(self.EXTEND_SEQ_LENS, dtype=torch.int32),
        )

    def _meta(self, rank: int) -> PallasForwardMetadata:
        backend = _DcpBackendUnderTest(
            self.PAGE_SIZE,
            self.req_to_token,
            self.DCP,
            rank,
            max_context_len=self.MAX_CONTEXT_LEN,
        )
        return _build_meta(backend, self.fb, self.DCP, rank)

    def test_kv_lens_cover_prefix_plus_chunk(self):
        for rank in range(self.DCP):
            with self.subTest(rank=rank):
                meta = self._meta(rank)
                expected = _local_lens(self.SEQ_LENS, self.DCP, rank)
                self.assertTrue(
                    torch.equal(meta.kv_lens.to(torch.int32), expected.to(torch.int32))
                )

    def test_prefix_kv_lens_exclude_this_steps_tokens(self):
        prefix = [s - e for s, e in zip(self.SEQ_LENS, self.EXTEND_SEQ_LENS)]
        for rank in range(self.DCP):
            with self.subTest(rank=rank):
                meta = self._meta(rank)
                expected = _local_lens(prefix, self.DCP, rank)
                self.assertTrue(
                    torch.equal(
                        meta.dcp_prefix_kv_lens.to(torch.int32),
                        expected.to(torch.int32),
                    ),
                    f"{meta.dcp_prefix_kv_lens.tolist()} != {expected.tolist()}",
                )

    def test_prefix_and_chunk_lengths_reconstruct_the_local_length(self):
        # Pass A reads dcp_prefix_kv_lens, pass B the chunk; together they must
        # be the rank's whole local shard.
        for rank in range(self.DCP):
            meta = self._meta(rank)
            chunk_local = [
                sum(
                    1
                    for p in range(s - e, s)
                    if (p // self.PAGE_SIZE) % self.DCP == rank
                )
                for s, e in zip(self.SEQ_LENS, self.EXTEND_SEQ_LENS)
            ]
            total = meta.dcp_prefix_kv_lens.to(torch.int64) + torch.tensor(
                chunk_local, dtype=torch.int64
            )
            self.assertTrue(torch.equal(total, meta.kv_lens.to(torch.int64)))

    def test_cu_q_lens_is_the_unsharded_cumsum(self):
        meta = self._meta(1)
        expected = torch.tensor([0, 7, 12, 19, 20], dtype=torch.int32)
        self.assertTrue(torch.equal(meta.cu_q_lens.to(torch.int32), expected))

    def test_global_kv_lens_field_is_the_unsharded_length(self):
        meta = self._meta(1)
        self.assertTrue(
            torch.equal(
                meta.dcp_kv_lens_global.to(torch.int32),
                torch.tensor(self.SEQ_LENS, dtype=torch.int32),
            )
        )


@_requires_dcp_helpers
class TestPallasRpaDcpCapacity(CustomTestCase):
    """The page-table capacity check measures one rank's local length."""

    PAGE_SIZE = 16
    DCP = 2
    PAGES_PER_SEQ = 4
    MAX_CONTEXT_LEN = PAGE_SIZE * DCP * PAGES_PER_SEQ  # 128
    LOCAL_CAPACITY = PAGES_PER_SEQ * PAGE_SIZE  # 64

    def _build(self, seq_len: int, rank: int = 0):
        req_to_token = _build_req_to_token(4, self.MAX_CONTEXT_LEN + 32)
        backend = _DcpBackendUnderTest(
            self.PAGE_SIZE,
            req_to_token,
            self.DCP,
            rank,
            max_context_len=self.MAX_CONTEXT_LEN,
        )
        fb = _make_forward_batch(
            seq_lens=torch.tensor([seq_len], dtype=torch.int32),
            req_pool_indices=torch.tensor([1], dtype=torch.int64),
            forward_mode=ForwardMode.DECODE,
        )
        return _build_meta(backend, fb, self.DCP, rank)

    def test_sequence_longer_than_one_ranks_table_is_accepted(self):
        # 100 global tokens exceed the 64-entry local table but stripe to 52
        # on rank 0 (three full 16-page runs plus 4 of the fourth).
        seq_len = 100
        self.assertGreater(seq_len, self.LOCAL_CAPACITY)
        meta = self._build(seq_len)
        self.assertEqual(int(meta.kv_lens[0]), 52)

    def test_local_length_over_capacity_is_refused(self):
        # 130 global tokens stripe to 66 on rank 0 -- past the local table.
        with self.assertRaisesRegex(ValueError, "page-table capacity"):
            self._build(130, rank=0)

    def test_capacity_is_checked_per_rank(self):
        # Rank 1 of a 129-token sequence holds 64 entries and still fits.
        self.assertEqual(int(self._build(129, rank=1).kv_lens[0]), 64)
        with self.assertRaisesRegex(ValueError, "page-table capacity"):
            self._build(129, rank=0)


class TestPallasRpaDcpDisabled(CustomTestCase):
    """dcp=1 must leave the metadata exactly as the non-DCP path builds it."""

    PAGE_SIZE = 16
    PAGES_PER_SEQ = 4
    MAX_CONTEXT_LEN = PAGE_SIZE * PAGES_PER_SEQ  # 64

    def test_no_dcp_fields_and_unwidened_page_stride(self):
        req_to_token = _build_req_to_token(4, self.MAX_CONTEXT_LEN)
        backend = _DcpBackendUnderTest(self.PAGE_SIZE, req_to_token, 1, 0)
        seq_lens = torch.tensor([10, 33], dtype=torch.int32)
        fb = _make_forward_batch(
            seq_lens=seq_lens,
            req_pool_indices=torch.tensor([1, 3], dtype=torch.int64),
            forward_mode=ForwardMode.DECODE,
        )

        meta = _build_meta(backend, fb, 1, 0)

        self.assertIsNone(meta.dcp_kv_lens_global)
        self.assertIsNone(meta.dcp_prefix_kv_lens)
        self.assertIsNone(meta.dcp_cp_rank)
        self.assertTrue(torch.equal(meta.kv_lens.to(torch.int32), seq_lens))
        expected_pages = [
            int(req_to_token[pool_idx, j * self.PAGE_SIZE]) // self.PAGE_SIZE
            for pool_idx in (1, 3)
            for j in range(self.PAGES_PER_SEQ)
        ]
        self.assertEqual(meta.page_indices.tolist(), expected_pages)


@_requires_dcp_helpers
class TestPallasRpaDcpCausalOnly(CustomTestCase):
    """The DCP passes are causal, so a non-causal layer is refused."""

    def test_non_causal_layers_are_refused(self):
        backend = _DcpBackendUnderTest(16, _build_req_to_token(1, 64), 2, 0)
        backend.kv_cache_quant_dtype = None
        backend.token_to_kv_pool = types.SimpleNamespace(
            get_fused_kv_buffer=lambda layer_id: None
        )
        fb = _make_forward_batch(
            seq_lens=torch.tensor([5], dtype=torch.int32),
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            forward_mode=ForwardMode.DECODE,
        )
        backend.forward_metadata = _build_meta(backend, fb, 2, 0)
        q = torch.zeros(1, 2 * 64)
        kv = torch.zeros(1, 64)
        for attrs in (
            {"attn_type": AttentionType.ENCODER_ONLY},
            {"is_cross_attention": True},
        ):
            layer = types.SimpleNamespace(
                layer_id=0,
                tp_q_head_num=2,
                tp_k_head_num=1,
                tp_v_head_num=1,
                qk_head_dim=64,
                v_head_dim=64,
                sliding_window_size=None,
                logit_cap=0,
                scaling=1.0,
                is_cross_attention=False,
                attn_type=AttentionType.DECODER,
            )
            vars(layer).update(attrs)
            with self.subTest(attrs=attrs):
                with self.assertRaisesRegex(NotImplementedError, "causal"):
                    backend._forward(q, kv, kv, layer, fb)


@_requires_dcp_helpers
class TestPallasRpaDcpKvHeads(CustomTestCase):
    """A layer whose K/V heads do not fit the DCP KV pool is refused, not run."""

    def _forward(self, *, q_heads: int, kv_heads: int, pool_kv_heads: int):
        backend = _DcpBackendUnderTest(16, _build_req_to_token(1, 64), 4, 0)
        backend.kv_cache_quant_dtype = None
        backend.token_to_kv_pool = types.SimpleNamespace(
            get_fused_kv_buffer=lambda layer_id: None, head_num=pool_kv_heads
        )
        fb = _make_forward_batch(
            seq_lens=torch.tensor([5], dtype=torch.int32),
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            forward_mode=ForwardMode.DECODE,
        )
        backend.forward_metadata = _build_meta(backend, fb, 4, 0)
        layer = types.SimpleNamespace(
            layer_id=0,
            tp_q_head_num=q_heads,
            tp_k_head_num=kv_heads,
            tp_v_head_num=kv_heads,
            qk_head_dim=64,
            v_head_dim=64,
            sliding_window_size=None,
            logit_cap=0,
            scaling=1.0,
            is_cross_attention=False,
            attn_type=AttentionType.DECODER,
        )
        kv = torch.zeros(1, kv_heads * 64)
        backend._forward(torch.zeros(1, q_heads * 64), kv, kv, layer, fb)

    def test_kv_sharded_over_the_whole_tp_group_is_refused(self):
        # 64 query / 4 KV heads at TP8 DCP4: the model keeps one KV head per
        # rank, while the pool holds the two heads of a tp // dcp shard.
        with self.assertRaisesRegex(ValueError, "KV pool holds 2"):
            self._forward(q_heads=8, kv_heads=1, pool_kv_heads=2)

    def test_query_heads_straddling_kv_heads_are_refused(self):
        # 96 query / 8 KV heads at TP8 DCP8: 12 query heads over 8 KV heads.
        with self.assertRaisesRegex(ValueError, "whole number of query heads"):
            self._forward(q_heads=12, kv_heads=8, pool_kv_heads=8)


@_requires_dcp_helpers
class TestPallasRpaDcpWarmupBatch(CustomTestCase):
    """The compile-bucket warmup's dummy batch (every seq_len 1) under DCP.

    Rank 0 owns position 0 of every sequence and every other rank's shard is
    empty -- the all-zero-shard shape the kernel's empty-shard guard exists
    for, and the shape every warmup bucket produces.
    """

    PAGE_SIZE = 16
    DCP = 4
    PAGES_PER_SEQ = 2
    MAX_CONTEXT_LEN = PAGE_SIZE * DCP * PAGES_PER_SEQ  # 128
    BS = 3

    def _meta(self, rank: int, forward_mode, extend_seq_lens=None):
        backend = _DcpBackendUnderTest(
            self.PAGE_SIZE,
            _build_req_to_token(self.BS, self.MAX_CONTEXT_LEN + 32),
            self.DCP,
            rank,
            max_context_len=self.MAX_CONTEXT_LEN,
        )
        fb = _make_forward_batch(
            seq_lens=torch.ones(self.BS, dtype=torch.int32),
            req_pool_indices=torch.arange(self.BS, dtype=torch.int64),
            forward_mode=forward_mode,
            extend_seq_lens=extend_seq_lens,
        )
        return _build_meta(backend, fb, self.DCP, rank)

    def test_decode_dummy_batch_lands_entirely_on_rank_zero(self):
        for rank in range(self.DCP):
            with self.subTest(rank=rank):
                meta = self._meta(rank, ForwardMode.DECODE)
                owned = 1 if rank == 0 else 0
                self.assertEqual(meta.kv_lens.tolist(), [owned] * self.BS)
                self.assertEqual(meta.dcp_prefix_kv_lens.tolist(), [owned] * self.BS)
                self.assertEqual(meta.dcp_kv_lens_global.tolist(), [1] * self.BS)

    def test_extend_dummy_batch_has_an_empty_cached_prefix(self):
        ones = torch.ones(self.BS, dtype=torch.int32)
        for rank in range(self.DCP):
            with self.subTest(rank=rank):
                meta = self._meta(rank, ForwardMode.EXTEND, extend_seq_lens=ones)
                owned = 1 if rank == 0 else 0
                self.assertEqual(meta.kv_lens.tolist(), [owned] * self.BS)
                self.assertEqual(meta.dcp_prefix_kv_lens.tolist(), [0] * self.BS)

    def test_page_table_keeps_the_widened_stride(self):
        meta = self._meta(1, ForwardMode.DECODE)
        self.assertEqual(
            tuple(meta.page_indices.shape), (self.BS * self.PAGES_PER_SEQ,)
        )


class TestDcpSlotParity(CustomTestCase):
    """``loc % dcp == position % dcp`` over real allocator traffic."""

    PHYSICAL_PAGE = 8
    DCP = 2
    NUM_PHYSICAL_PAGES = 64

    def _allocator(self, dcp: int = None):
        dcp = self.DCP if dcp is None else dcp
        return TpuPagedTokenToKVPoolAllocator(
            size=self.NUM_PHYSICAL_PAGES * self.PHYSICAL_PAGE * dcp,
            page_size=self.PHYSICAL_PAGE * dcp,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )

    @staticmethod
    def _alloc_extend(allocator, prefix_lens, seq_lens, last_loc):
        prefix = torch.tensor(prefix_lens, dtype=torch.int64)
        seq = torch.tensor(seq_lens, dtype=torch.int64)
        last = torch.tensor(last_loc, dtype=torch.int64)
        num_tokens = int((seq - prefix).sum())
        out = allocator.alloc_extend(prefix, prefix, seq, seq, last, num_tokens)
        assert out is not None, "allocator ran out of pages"
        return out.to(torch.int64)

    @staticmethod
    def _alloc_decode(allocator, seq_lens, last_loc):
        seq = torch.tensor(seq_lens, dtype=torch.int64)
        last = torch.tensor(last_loc, dtype=torch.int64)
        out = allocator.alloc_decode(seq, seq, last)
        assert out is not None, "allocator ran out of pages"
        return out.to(torch.int64)

    def _run_traffic(self, chunks, decode_steps, dcp=None):
        """Allocate ``chunks`` (a list of per-request extend sizes) then decode.

        Returns the per-request list of token locations, indexed by position.
        """
        dcp = self.DCP if dcp is None else dcp
        allocator = self._allocator(dcp)
        num_reqs = len(chunks[0])
        locs = [[] for _ in range(num_reqs)]
        lens = [0] * num_reqs
        for chunk in chunks:
            prefix = list(lens)
            seq = [p + c for p, c in zip(prefix, chunk)]
            last_loc = [locs[i][-1] if locs[i] else -1 for i in range(num_reqs)]
            out = self._alloc_extend(allocator, prefix, seq, last_loc)
            offset = 0
            for i, c in enumerate(chunk):
                locs[i].extend(out[offset : offset + c].tolist())
                offset += c
            lens = seq
        for _ in range(decode_steps):
            lens = [length + 1 for length in lens]
            last_loc = [locs[i][-1] for i in range(num_reqs)]
            out = self._alloc_decode(allocator, lens, last_loc)
            for i in range(num_reqs):
                locs[i].append(int(out[i]))
        return locs

    def _assert_parity(self, locs, dcp=None):
        dcp = self.DCP if dcp is None else dcp
        for req, token_locs in enumerate(locs):
            for position, loc in enumerate(token_locs):
                self.assertEqual(
                    loc % dcp,
                    position % dcp,
                    f"request {req} position {position}: loc {loc}",
                )

    def test_fresh_prefill_then_decode(self):
        locs = self._run_traffic([[13, 1, 20]], decode_steps=12)
        self._assert_parity(locs)

    def test_chunked_prefill_continuing_a_partial_page(self):
        # Chunk boundaries that fall inside a widened page are where a
        # non-position-aligned allocator would break parity.
        locs = self._run_traffic([[5, 17, 3], [11, 4, 19], [7, 7, 7]], decode_steps=9)
        self._assert_parity(locs)

    def test_parity_holds_for_dcp_four_and_eight(self):
        for dcp in (4, 8):
            with self.subTest(dcp=dcp):
                locs = self._run_traffic(
                    [[6, 25, 1], [9, 3, 14]], decode_steps=7, dcp=dcp
                )
                self._assert_parity(locs, dcp=dcp)

    def test_locations_are_unique_across_requests(self):
        locs = self._run_traffic([[13, 1, 20]], decode_steps=12)
        flat = [loc for token_locs in locs for loc in token_locs]
        self.assertEqual(len(flat), len(set(flat)))

    def test_no_real_token_lands_on_the_dummy_slot(self):
        # Padding rows are written to physical slot 0, so real tokens must
        # never map there. Page ids start at 1 in the TPU allocator.
        locs = self._run_traffic([[13, 1, 20]], decode_steps=12)
        for token_locs in locs:
            for loc in token_locs:
                self.assertGreater(loc // self.DCP, 0)


class TestOwnerWriteSpans(CustomTestCase):
    """``build_owner_write_spans`` writes each token's physical slot on exactly
    one rank, through spans that never straddle a slot jump."""

    DCP = 4
    PAGE = 8

    def _locations(self, positions):
        # Position-aligned locations in the widened space, page 1 onwards.
        widened = self.PAGE * self.DCP
        return torch.tensor([widened + p for p in positions], dtype=torch.int64)

    def _owns(self, position: int, rank: int) -> bool:
        return (position // self.PAGE) % self.DCP == rank

    def _slot(self, location: int, position: int) -> int:
        widened = self.PAGE * self.DCP
        return (location // widened) * self.PAGE + position % self.PAGE

    @staticmethod
    def _written(spans):
        """(compacted row -> physical slot) pairs the spans write, in order."""
        pairs = []
        for dst, src, n in zip(
            spans.dst.tolist(), spans.src.tolist(), spans.lens.tolist()
        ):
            pairs.extend((src + i, dst + i) for i in range(n))
        return pairs

    def test_each_rank_writes_exactly_its_owned_slots(self):
        positions = torch.arange(17, dtype=torch.int64)
        locations = self._locations(positions.tolist())
        claimed = {}
        for rank in range(self.DCP):
            spans = build_owner_write_spans(
                positions, locations, self.DCP, rank, self.PAGE, 32, 1
            )
            owned = [
                i for i, p in enumerate(positions.tolist()) if self._owns(p, rank)
            ]
            self.assertEqual(spans.index[: len(owned)].tolist(), owned)
            self.assertEqual(int(spans.lens.sum()), len(owned))
            for row, slot in self._written(spans):
                token = owned[row]
                self.assertEqual(slot, self._slot(int(locations[token]), token))
                self.assertNotIn(token, claimed)
                claimed[token] = rank
        self.assertEqual(sorted(claimed), positions.tolist())

    def test_consecutive_owned_slots_share_one_span(self):
        # Rank 1 owns the second 8-token page: positions 8..15, one run.
        positions = torch.arange(16, dtype=torch.int64)
        locations = self._locations(positions.tolist())
        spans = build_owner_write_spans(
            positions, locations, self.DCP, 1, self.PAGE, 16, 1
        )
        self.assertEqual(spans.lens.tolist()[:2], [8, 0])
        self.assertEqual(spans.src.tolist()[0], 0)
        self.assertEqual(spans.dst.tolist()[0], self.PAGE)

    def test_a_slot_jump_starts_a_new_span(self):
        # Two requests whose pages are not adjacent: rank 0 owns page 0 of
        # each (positions 0..7), so it writes two runs of eight.
        positions = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7] * 2, dtype=torch.int64)
        widened = self.PAGE * self.DCP
        locations = torch.tensor(
            [widened + p for p in range(8)] + [3 * widened + p for p in range(8)],
            dtype=torch.int64,
        )
        spans = build_owner_write_spans(
            positions, locations, self.DCP, 0, self.PAGE, 16, 2
        )
        self.assertEqual(spans.lens.tolist()[:3], [8, 8, 0])
        self.assertEqual(spans.src.tolist()[:2], [0, 8])
        self.assertEqual(spans.dst.tolist()[:2], [self.PAGE, 3 * self.PAGE])
        for rank in range(1, self.DCP):
            spans = build_owner_write_spans(
                positions, locations, self.DCP, rank, self.PAGE, 16, 2
            )
            self.assertEqual(int(spans.lens.sum()), 0)

    def test_padding_rows_go_to_the_dummy_slot(self):
        # Sync padding carries out_cache_loc 0 and position 0; only the rank
        # that owns position 0 writes it, into physical slot 0.
        positions = torch.zeros(3, dtype=torch.int64)
        locations = torch.zeros(3, dtype=torch.int64)
        for rank in range(self.DCP):
            spans = build_owner_write_spans(
                positions, locations, self.DCP, rank, self.PAGE, 4, 1
            )
            with self.subTest(rank=rank):
                if rank == 0:
                    self.assertEqual(spans.dst.tolist()[0], 0)
                    self.assertEqual(int(spans.lens.sum()), 3)
                else:
                    self.assertEqual(int(spans.lens.sum()), 0)

    def test_span_table_has_the_bucket_length_and_rejects_overflow(self):
        positions = torch.arange(6, dtype=torch.int64)
        # Every owned token on its own page: more spans than the table holds.
        locations = torch.tensor(
            [(p + 1) * self.PAGE * self.DCP * 2 for p in range(6)], dtype=torch.int64
        )
        spans = build_owner_write_spans(positions, locations, 1, 0, self.PAGE, 6, 5)
        self.assertEqual(spans.dst.numel(), max_write_spans(6, 5, self.PAGE))
        with self.assertRaises(ValueError):
            build_owner_write_spans(positions, locations, 1, 0, self.PAGE, 6, 1)


if __name__ == "__main__":
    unittest.main()
