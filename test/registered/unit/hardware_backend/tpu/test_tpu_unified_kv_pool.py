"""Unified full/sliding-window KV pool on TPU: shared buffers, one page list.

Runs on CPU with no TPU: the pool buffers are CPU tensors and the allocator's
page bookkeeping is host-resident on TPU as well. Every platform check the code
under test makes is patched, so the cases behave the same with or without
torch_tpu installed.
"""

import contextlib
import subprocess
import sys
import threading
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.environ import envs
from sglang.srt.mem_cache import hicache_auto_size
from sglang.srt.mem_cache.allocator import swa as swa_allocator_module
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import kv_to_page_indices
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.pool_host import base as host_base
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.pool_configurator import (
    HybridSWAPoolConfigurator,
    create_memory_pool_configurator,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.allocator_tpu import (
    TpuPagedTokenToKVPoolAllocator,
)
from sglang_tpu.disaggregation.raiden.common import get_raiden_kv_caches
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager
from sglang_tpu.disaggregation.raiden.serde import Blocks, MessageTag, RoomFailed
from sglang_tpu.disaggregation.raiden.state import DecodeState, PrefillState
from sglang_tpu.disaggregation.tpu.wire import WireMessage
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool, TpuSeqAlongLaneKVPool
from sglang_tpu.kv_pool_host import TpuRaidenHostKVCache
from sglang_tpu.tpu_platform import TpuSRTPlatform
from sglang_tpu.unified_kv_pool import (
    TpuUnifiedSWAKVPool,
    TpuUnifiedSWAPoolConfigurator,
    TpuUnifiedSWATokenToKVPoolAllocator,
    _SharedBufferView,
    unified_kv_pool_enabled,
)
from sglang_tpu.unified_kv_pool_host import TpuRaidenUnifiedHostKVCache

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAGE_SIZE = 4
NUM_PAGES = 8
POOL_TOKENS = NUM_PAGES * PAGE_SIZE


class _RankFourPool(TpuFusedMHATokenToKVPool):
    """A pool class a platform may register in place of the fused one: its own
    rank-4 [num_blocks, 2 * heads, head_dim, page_size] buffers, which it creates
    and swaps itself."""

    def _create_buffers(self):
        shape = (self.num_blocks, 2 * self.head_num, self.head_dim, self.page_size)
        self.kv_buffer = [
            torch.zeros(shape, dtype=self.store_dtype) for _ in range(self.layer_num)
        ]
        self._refresh_descriptors()

    def _refresh_descriptors(self):
        self.data_ptrs = torch.tensor(
            [buf.data_ptr() for buf in self.kv_buffer], dtype=torch.uint64
        )
        self.data_strides = torch.tensor([buf[0].nbytes for buf in self.kv_buffer])

    def get_state(self):
        return list(self.kv_buffer)

    def set_state(self, state):
        self.kv_buffer = list(state)
        self._refresh_descriptors()


def _make_pool(
    *,
    full_layer_ids=(0,),
    swa_layer_ids=(1,),
    head_dim=128,
    swa_head_num=1,
    swa_head_dim=128,
    pool_cls=TpuFusedMHATokenToKVPool,
    size=POOL_TOKENS,
):
    """A bf16 unified pool whose full-attention layers have one KV head."""
    return TpuUnifiedSWAKVPool(
        size=size,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        head_num=1,
        head_dim=head_dim,
        swa_head_num=swa_head_num,
        swa_head_dim=swa_head_dim,
        swa_v_head_dim=swa_head_dim,
        swa_attention_layer_ids=list(swa_layer_ids),
        full_attention_layer_ids=list(full_layer_ids),
        device="cpu",
        start_layer=0,
        token_to_kv_pool_class=pool_cls,
    )


def _hybrid_kvc(
    *,
    full_layer_ids=(0, 2),
    swa_layer_ids=(1, 3),
    sliding_window_size=None,
    page_size=PAGE_SIZE,
    head_dim=128,
    swa_kv_heads=1,
    swa_head_dim=128,
):
    """KV-cache config of a bf16 hybrid model whose full-attention layers have
    one KV head; by default head_dim 128, 512 bytes per token and layer."""
    model_config = SimpleNamespace(
        attention_arch=AttentionArch.MHA,
        head_dim=head_dim,
        v_head_dim=head_dim,
        swa_head_dim=swa_head_dim,
        swa_v_head_dim=swa_head_dim,
        get_num_kv_heads=lambda tp_size, dcp_size=1: 1,
        get_swa_num_kv_heads=lambda tp_size, dcp_size=1: swa_kv_heads,
        hf_config=SimpleNamespace(architectures=["GptOssForCausalLM"]),
    )
    return SimpleNamespace(
        is_hybrid_swa=True,
        use_mla_backend=False,
        mambaish_config=None,
        kv_cache_dtype_str="auto",
        kv_cache_dtype=torch.bfloat16,
        model_config=model_config,
        is_draft_worker=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        layer_info=SimpleNamespace(
            start_layer=0,
            full_attention_layer_ids=list(full_layer_ids),
            swa_attention_layer_ids=list(swa_layer_ids),
        ),
        sliding_window_size=sliding_window_size,
        page_size=page_size,
        attn_dp_size=1,
    )


def _fused_rpa_platform(pool_cls=TpuFusedMHATokenToKVPool):
    """The platform as the unified pool sees it under an RPA attention backend
    whose MHA pool class is ``pool_cls``."""
    return patch(
        "sglang_tpu.unified_kv_pool.current_platform",
        SimpleNamespace(get_mha_kv_pool_cls=lambda: pool_cls),
    )


def _tpu_swa_allocator_branches(testcase):
    """Take the TPU branches of the SWA allocator: host-resident mapping and the
    TPU paged sub-allocators."""
    for patcher in (
        patch.object(swa_allocator_module, "_is_tpu", True),
        patch.object(
            swa_allocator_module,
            "TpuPagedTokenToKVPoolAllocator",
            TpuPagedTokenToKVPoolAllocator,
            create=True,
        ),
    ):
        patcher.start()
        testcase.addCleanup(patcher.stop)


def _grouped_allocator(testcase):
    """One full-attention and three sliding-window layers (ids 0 and 1-3):
    three SWA groups over one buffer."""
    _tpu_swa_allocator_branches(testcase)
    pool = _make_pool(full_layer_ids=(0,), swa_layer_ids=(1, 2, 3))
    alloc = TpuUnifiedSWATokenToKVPoolAllocator(
        POOL_TOKENS,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=pool,
        need_sort=False,
    )
    return alloc, pool


def _extend(alloc, *seq_lens):
    """Allocate fresh requests of ``seq_lens`` tokens; returns their slots."""
    num_reqs = len(seq_lens)
    prefix_lens = torch.zeros(num_reqs, dtype=torch.int64)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int64)
    last_loc = torch.full((num_reqs,), -1, dtype=torch.int64)
    return alloc.alloc_extend(
        prefix_lens, prefix_lens, seq_lens_t, seq_lens_t, last_loc, sum(seq_lens)
    )


def _extend_swa_tail(alloc, *, seq_len, swa_tail_len):
    """Allocate one fresh request whose SWA side covers only its last tokens."""
    prefix_lens = torch.zeros(1, dtype=torch.int64)
    seq_lens = torch.tensor([seq_len], dtype=torch.int64)
    return alloc.alloc_extend_swa_tail(
        prefix_lens,
        prefix_lens,
        seq_lens,
        seq_lens,
        torch.tensor([-1], dtype=torch.int64),
        seq_len,
        swa_tail_len,
    )


def _decode(alloc, last_loc, seq_len):
    """Allocate the next token of one request whose new length is ``seq_len``."""
    seq_lens = torch.tensor([seq_len], dtype=torch.int64)
    return alloc.alloc_decode(seq_lens, seq_lens, last_loc)


def _held_and_free_pages(alloc):
    """(full pages held, SWA pages held, free pages), as the leak check sees them."""
    (full_cap, full_free), (swa_cap, swa_free) = alloc.swa_capacity_and_available(
        full_capacity=None, swa_capacity=None
    )
    assert full_free == swa_free
    free = full_free // PAGE_SIZE
    return full_cap // PAGE_SIZE - free, swa_cap // PAGE_SIZE - free, free


class _StubTreeCache:
    """Prefix-cache stand-in: records eviction requests (``evict`` walks
    apart), then runs ``on_evict``."""

    def __init__(self, on_evict=None, *, swa_evictable=0):
        self.requests = []
        self.walks = []
        self._on_evict = on_evict
        self._swa_evictable = swa_evictable

    def supports_prefix_sharing(self):
        return True

    def full_evictable_size(self):
        return 0

    def swa_evictable_size(self):
        return self._swa_evictable

    def evict_for_alloc(self, params):
        self.requests.append((params.num_tokens, params.swa_num_tokens))
        if self._on_evict is not None:
            self._on_evict()

    def evict(self, params):
        self.walks.append((params.num_tokens, params.swa_num_tokens))
        if self._on_evict is not None:
            self._on_evict()


class TestUnifiedPoolStorage(CustomTestCase):
    def test_full_and_swa_layers_share_buffers(self):
        # Three full-attention and two sliding-window layers need three buffers.
        pool = _make_pool(full_layer_ids=(0, 2, 4), swa_layer_ids=(1, 3))
        state = pool.get_state()
        self.assertEqual(len(state), 3)
        self.assertEqual(len({id(buf) for buf in state}), 3)
        # Full layer i and sliding-window layer i use buffer i.
        self.assertIs(pool.get_fused_kv_buffer(0), state[0])
        self.assertIs(pool.get_fused_kv_buffer(1), state[0])
        self.assertIs(pool.get_fused_kv_buffer(3), state[1])
        self.assertIs(pool.get_fused_kv_buffer(4), state[2])
        k_size, v_size = pool.get_kv_size_bytes()
        self.assertEqual(k_size + v_size, sum(buf.nbytes for buf in state))

        # Compiled graphs swap the state in; both kinds of layer must follow.
        new_state = [torch.zeros_like(buf) for buf in state]
        pool.set_state(new_state)
        self.assertIs(pool.get_fused_kv_buffer(1), new_state[0])
        self.assertIs(pool.get_fused_kv_buffer(2), new_state[1])
        self.assertIs(pool.get_fused_kv_buffer(4), new_state[2])

    def test_buffers_and_views_take_the_platform_pool_layout(self):
        """A platform pool class with its own layout keeps it in the unified
        pool, and its attention backends see views of that class."""
        pool = _make_pool(
            full_layer_ids=(0,), swa_layer_ids=(1, 2, 3), pool_cls=_RankFourPool
        )
        (buf,) = pool.get_state()
        self.assertEqual(tuple(buf.shape), (NUM_PAGES + 1, 2, 128, PAGE_SIZE))
        self.assertIsInstance(pool.full_kv_pool.backing, _RankFourPool)
        for view in (pool.full_kv_pool, pool.swa_kv_pool):
            self.assertIsInstance(view, _RankFourPool)
            self.assertIsInstance(view, _SharedBufferView)
        for layer_id in range(4):
            self.assertIs(pool.get_fused_kv_buffer(layer_id), buf)
        self.assertEqual(pool.swa_kv_pool.data_ptrs.tolist(), [buf.data_ptr()] * 3)
        self.assertEqual(sum(pool.get_kv_size_bytes()), buf.nbytes)

        # The pool class's own state swap reaches both kinds of layer.
        new_buf = torch.zeros_like(buf)
        pool.set_state([new_buf])
        self.assertIs(pool.get_fused_kv_buffer(0), new_buf)
        self.assertIs(pool.get_fused_kv_buffer(3), new_buf)
        with get_context().override_server_args(hicache_io_backend="raiden"):
            self.assertIs(
                TpuSRTPlatform().get_mha_kv_pool_host_cls(pool.swa_kv_pool),
                TpuRaidenUnifiedHostKVCache,
            )

    def test_sliding_window_layers_beyond_the_full_count_form_groups(self):
        # Five sliding-window layers over two full ones form ceil(5 / 2) = 3
        # groups by stride, so the pool keeps two buffers instead of five.
        full_ids, swa_ids = (0, 3), (1, 2, 4, 5, 6)
        pool = _make_pool(full_layer_ids=full_ids, swa_layer_ids=swa_ids)
        state = pool.get_state()
        self.assertEqual(len(state), 2)
        self.assertEqual(pool.num_swa_groups, 3)
        self.assertIs(pool.get_fused_kv_buffer(0), state[0])
        self.assertIs(pool.get_fused_kv_buffer(3), state[1])
        # The k-th layer of each group uses buffer k.
        buffer_index = {id(buf): k for k, buf in enumerate(state)}
        self.assertEqual(
            [buffer_index[id(pool.get_fused_kv_buffer(i))] for i in swa_ids],
            [0, 0, 0, 1, 1],
        )
        self.assertEqual([pool.swa_page_group(i) for i in swa_ids], [0, 1, 2, 0, 1])
        # One extra page id per further group and page.
        self.assertEqual(pool.swa_group_pages.shape, (2, NUM_PAGES + 1))

    def test_sliding_window_heads_merge_into_the_full_layer_shape(self):
        # One full head of 256 holds two sliding-window heads of 128 side by
        # side: the buffers keep the full layers' shape.
        pool = _make_pool(head_dim=256, swa_head_num=2, swa_head_dim=128)
        (buf,) = pool.get_state()
        self.assertEqual(tuple(buf.shape), (NUM_PAGES + 1, PAGE_SIZE, 1, 2, 256))
        self.assertIs(pool.get_fused_kv_buffer(1), buf)
        self.assertTrue(pool.swa_kv_heads_merged)
        self.assertFalse(_make_pool().swa_kv_heads_merged)

    def test_merged_heads_attend_as_the_separate_heads(self):
        """Attention over the merged KV head equals each query head's attention
        over its own KV head."""
        from sglang_tpu.unified_kv_heads import merge_swa_kv_heads, split_swa_output

        torch.manual_seed(0)
        tokens, q_heads, kv_heads, head_dim = 5, 4, 2, 8
        q = torch.randn(tokens, q_heads, head_dim, dtype=torch.float64)
        k = torch.randn(tokens, kv_heads, head_dim, dtype=torch.float64)
        v = torch.randn(tokens, kv_heads, head_dim, dtype=torch.float64)

        def attend(q, k, v):
            # Query head i reads KV head i // (q_heads / kv_heads), as RPA does.
            k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
            v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
            scores = torch.einsum("qhd,khd->hqk", q, k) / head_dim**0.5
            return torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), v)

        merged = attend(*merge_swa_kv_heads(q, k, v))
        self.assertEqual(merged.shape, (tokens, q_heads, kv_heads * head_dim))
        torch.testing.assert_close(split_swa_output(merged, kv_heads), attend(q, k, v))

    def test_host_budget_counts_each_shared_buffer_once(self):
        # Each view references the buffers its layers use, so summing the views'
        # buffer bytes would size the host tier for 3 + 2 buffers instead of 3.
        pool = _make_pool(full_layer_ids=(0, 2, 4), swa_layer_ids=(1, 3))
        self.assertEqual(
            hicache_auto_size._pool_bytes(pool),
            sum(buf.nbytes for buf in pool.get_state()),
        )


class TestUnifiedAllocator(CustomTestCase):
    def _make_allocator(self):
        _tpu_swa_allocator_branches(self)
        return TpuUnifiedSWATokenToKVPoolAllocator(
            POOL_TOKENS,
            page_size=PAGE_SIZE,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=_make_pool(),
            need_sort=False,
        )

    def test_a_sliding_window_page_takes_one_page_id_per_group(self):
        alloc, pool = _grouped_allocator(self)
        full_loc = _extend(alloc, PAGE_SIZE + 1)
        # Two full pages and two SWA pages in each of three groups fill all
        # eight pages, and no two of them share an id.
        self.assertEqual(alloc.full_available_size(), 0)
        full_pages = (full_loc // PAGE_SIZE).unique()
        swa_pages = (
            alloc.translate_loc_from_full_to_swa(full_loc) // PAGE_SIZE
        ).unique()
        group_pages = pool.swa_group_pages[:, swa_pages.long()].reshape(-1)
        self.assertEqual(
            sorted(torch.cat([full_pages, swa_pages, group_pages]).tolist()),
            list(range(1, NUM_PAGES + 1)),
        )
        (full_cap, _), (swa_cap, _) = alloc.swa_capacity_and_available(
            full_capacity=None, swa_capacity=None
        )
        self.assertEqual((full_cap, swa_cap), (2 * PAGE_SIZE, 2 * PAGE_SIZE))
        # Both sides track the one page store's pages in one bitmap, the
        # further groups' pages included; a release of a page it does not
        # hold handed out raises.
        bitmap = alloc.full_attn_allocator.page_is_free
        self.assertIs(alloc.swa_attn_allocator.page_is_free, bitmap)
        self.assertFalse(bool(bitmap.any()))

        # Releasing an SWA page returns its page in every group.
        alloc.free_swa(full_loc[:PAGE_SIZE])
        self.assertEqual(alloc.full_available_size(), 3 * PAGE_SIZE)
        self.assertEqual(alloc.swa_available_size(), PAGE_SIZE)
        self.assertEqual(int(bitmap.sum()), 3)
        alloc.free(full_loc)
        self.assertEqual(alloc.full_available_size(), POOL_TOKENS)

    def test_a_release_of_the_padding_page_is_dropped(self):
        """Freeing the SWA side of slots that have none (mapping 0, page 0)
        must not put the padding page on the shared free list, once per group."""
        alloc, _ = _grouped_allocator(self)
        loc = _extend(alloc, PAGE_SIZE)
        free = alloc.full_available_size()
        unmapped = torch.arange(PAGE_SIZE, dtype=torch.int64) + 5 * PAGE_SIZE
        self.assertEqual(int(alloc.full_to_swa_index_mapping[unmapped].max()), 0)
        alloc.free_swa_segment(unmapped, start_pos=0)
        self.assertEqual(alloc.full_available_size(), free)
        alloc.free(loc)
        self.assertEqual(alloc.full_available_size(), POOL_TOKENS)

    def test_retraction_copy_round_trips_every_group(self):
        """A retracted request's host copy restores every sliding-window
        layer's rows, each at its own group's pages of the new allocation."""
        alloc, pool = _grouped_allocator(self)
        loc = _extend(alloc, PAGE_SIZE)
        (buf,) = pool.get_state()
        torch.manual_seed(0)
        buf.copy_(torch.randn(buf.shape).to(buf.dtype))
        rows = buf.flatten(0, 1)

        def layer_slots(full_loc):
            """Slots of the full layer and of SWA layers 1-3 (groups 0-2)."""
            swa = alloc.translate_loc_from_full_to_swa(full_loc).long()
            pages, offsets = swa // PAGE_SIZE, swa % PAGE_SIZE
            companions = pool.swa_group_pages[:, pages].long() * PAGE_SIZE + offsets
            return [full_loc.long(), swa, companions[0], companions[1]]

        expected = [rows[slots].clone() for slots in layer_slots(loc)]
        saved = pool.get_cpu_copy(loc)
        alloc.free(loc)
        _extend(alloc, PAGE_SIZE)  # takes other pages first
        new_loc = _extend(alloc, PAGE_SIZE)
        self.assertNotEqual(
            set(layer_slots(new_loc)[3].tolist()), set(layer_slots(loc)[3].tolist())
        )
        buf.zero_()
        pool.load_cpu_copy(saved, new_loc)
        for layer, slots in enumerate(layer_slots(new_loc)):
            torch.testing.assert_close(
                rows[slots], expected[layer], msg=f"layer {layer}"
            )

    def test_grouped_capacity_and_eviction_charge_every_group(self):
        alloc, _ = _grouped_allocator(self)
        # Two full and two SWA pages take 2 + 3 * 2 = 8 page ids.
        self.assertTrue(alloc.can_reserve(2 * PAGE_SIZE, 2 * PAGE_SIZE))
        self.assertFalse(alloc.can_reserve(3 * PAGE_SIZE, 2 * PAGE_SIZE))
        self.assertFalse(alloc.new_pages_available(0, 3))
        # An evictable SWA page frees one page per group.
        self.assertTrue(
            alloc.can_reserve(
                5 * PAGE_SIZE, 2 * PAGE_SIZE, swa_evictable_tokens=PAGE_SIZE
            )
        )

        _extend(alloc, PAGE_SIZE)
        # Two pages on each side need 2 + 6 = 8 ids and four are free: the tree
        # is asked for four full pages, or two SWA pages of three ids each.
        tree = _StubTreeCache()
        self.assertFalse(alloc.evict_to_free_tokens(tree, 2 * PAGE_SIZE))
        self.assertEqual(tree.requests, [(4 * PAGE_SIZE, 2 * PAGE_SIZE)])

    def test_full_and_swa_pages_are_disjoint_ids_of_one_pool(self):
        alloc = self._make_allocator()
        full_loc = _extend(alloc, PAGE_SIZE + 1, PAGE_SIZE - 1)
        swa_loc = alloc.translate_loc_from_full_to_swa(full_loc)
        full_pages = set((full_loc // PAGE_SIZE).tolist())
        swa_pages = set((swa_loc // PAGE_SIZE).tolist())
        # Two pages for the first request and one for the second, on each side.
        self.assertEqual(len(full_pages), 3)
        self.assertEqual(len(swa_pages), 3)
        # A layer pair shares a buffer, so its pages must never overlap.
        self.assertFalse(full_pages & swa_pages)
        # Page 0 takes the padding writes of both sides.
        self.assertNotIn(0, full_pages | swa_pages)
        self.assertEqual(_held_and_free_pages(alloc), (3, 3, NUM_PAGES - 6))

    def test_pages_freed_by_one_side_serve_the_other(self):
        alloc = self._make_allocator()
        # Six full pages and one SWA page; a static 0.8-ratio split of the same
        # eight pages caps the full side at four.
        long_req = _extend_swa_tail(
            alloc, seq_len=6 * PAGE_SIZE, swa_tail_len=PAGE_SIZE
        )
        self.assertIsNotNone(long_req)
        swa_page = int(alloc.translate_loc_from_full_to_swa(long_req[-1:])[0])
        swa_page //= PAGE_SIZE
        alloc.free_swa(long_req[-PAGE_SIZE:])

        # Two full pages fit only by taking the page the SWA side gave back.
        full_only = _extend_swa_tail(alloc, seq_len=2 * PAGE_SIZE, swa_tail_len=0)
        self.assertIsNotNone(full_only)
        self.assertIn(swa_page, set((full_only // PAGE_SIZE).tolist()))
        self.assertEqual(_held_and_free_pages(alloc), (NUM_PAGES, 0, 0))

    def test_decode_short_of_one_side_takes_no_page(self):
        alloc = self._make_allocator()
        req = _extend(alloc, PAGE_SIZE)
        filler = _extend_swa_tail(alloc, seq_len=5 * PAGE_SIZE, swa_tail_len=0)
        self.assertIsNotNone(filler)
        # One free page left, and the next token opens a page on both sides.
        self.assertEqual(_held_and_free_pages(alloc), (6, 1, 1))
        self.assertIsNone(_decode(alloc, req[-1:], PAGE_SIZE + 1))
        self.assertEqual(_held_and_free_pages(alloc), (6, 1, 1))

        alloc.free_full(filler)
        self.assertIsNotNone(_decode(alloc, req[-1:], PAGE_SIZE + 1))
        self.assertEqual(_held_and_free_pages(alloc), (2, 2, 4))

    def test_every_release_path_returns_pages_to_the_shared_list(self):
        alloc = self._make_allocator()
        first = _extend(alloc, 2 * PAGE_SIZE)
        second = _extend(alloc, PAGE_SIZE)
        second = torch.cat([second, _decode(alloc, second[-1:], PAGE_SIZE + 1)])
        self.assertEqual(_held_and_free_pages(alloc), (4, 4, 0))

        # Sliding-window eviction of the first request's first page.
        alloc.free_swa(first[:PAGE_SIZE])
        self.assertEqual(_held_and_free_pages(alloc), (4, 3, 1))

        alloc.free_group_begin()
        alloc.free(second)
        alloc.free_full(first[:PAGE_SIZE])
        # Grouped frees land only when the group ends.
        self.assertEqual(_held_and_free_pages(alloc), (4, 3, 1))
        alloc.free_group_end()
        self.assertEqual(_held_and_free_pages(alloc), (1, 1, 6))

        alloc.free_segment(first[PAGE_SIZE:], start_pos=PAGE_SIZE)
        self.assertEqual(_held_and_free_pages(alloc), (0, 0, NUM_PAGES))

    def test_can_reserve_counts_both_sides_against_one_budget(self):
        alloc = self._make_allocator()
        half = POOL_TOKENS // 2
        # Four full pages plus four SWA pages fill the eight-page pool exactly.
        self.assertTrue(alloc.can_reserve(half, half))
        self.assertFalse(alloc.can_reserve(half, half + 1))
        # Fractional demand (scaled decode reservations) still takes its page.
        self.assertFalse(alloc.can_reserve(half + 0.5, half))
        self.assertFalse(alloc.can_reserve(-1, 0))
        # Evictable whole pages of either side count; a partial page does not.
        self.assertTrue(
            alloc.can_reserve(half + PAGE_SIZE, half, full_evictable_tokens=PAGE_SIZE)
        )
        self.assertTrue(
            alloc.can_reserve(half, half + PAGE_SIZE, swa_evictable_tokens=PAGE_SIZE)
        )
        self.assertFalse(
            alloc.can_reserve(
                half + PAGE_SIZE, half, full_evictable_tokens=PAGE_SIZE - 1
            )
        )

        _extend(alloc, PAGE_SIZE)
        self.assertFalse(alloc.can_reserve(half, half))
        self.assertTrue(alloc.can_reserve(half, half, empty_pool=True))
        # Admission keeps one token of slack below each side's capacity.
        self.assertTrue(
            alloc.can_reserve(
                POOL_TOKENS - 1, 0, empty_pool=True, require_token_slack=True
            )
        )
        self.assertFalse(
            alloc.can_reserve(POOL_TOKENS, 0, empty_pool=True, require_token_slack=True)
        )

    def test_eviction_requests_the_joint_page_shortfall(self):
        alloc = self._make_allocator()
        cached = _extend(alloc, 3 * PAGE_SIZE)
        # Three tokens-worth of pages on each side need six pages; two are free,
        # so the tree is asked for four pages on the shared list.
        shortfall = 4 * PAGE_SIZE
        nothing_evictable = _StubTreeCache()
        self.assertFalse(alloc.evict_to_free_tokens(nothing_evictable, 3 * PAGE_SIZE))
        self.assertEqual(nothing_evictable.requests, [(shortfall, shortfall)])

        evicts_cached = _StubTreeCache(on_evict=lambda: alloc.free(cached))
        self.assertTrue(alloc.evict_to_free_tokens(evicts_cached, 3 * PAGE_SIZE))
        self.assertEqual(evicts_cached.requests, [(shortfall, shortfall)])

        # Enough free pages: nothing is requested.
        self.assertTrue(alloc.evict_to_free_tokens(nothing_evictable, 3 * PAGE_SIZE))
        self.assertEqual(len(nothing_evictable.requests), 1)

    def test_prealloc_reclaim_counts_both_sides_against_one_list(self):
        """A preallocation reclaims pages for both sides from the one free list;
        checking the sliding-window side alone over-commits the shared pool."""
        alloc = self._make_allocator()
        cached = _extend(alloc, 3 * PAGE_SIZE)
        # Two pages are free: one page per side fits without eviction.
        tree = _StubTreeCache()
        self.assertIsNone(alloc.reclaim_for_prealloc(tree, PAGE_SIZE, PAGE_SIZE))
        self.assertEqual(tree.requests, [])

        # Two pages per side need four, and two are free;
        # checking the SWA side alone would miss the shortfall.
        shortfall = 2 * PAGE_SIZE
        self.assertIsNotNone(
            alloc.reclaim_for_prealloc(tree, 2 * PAGE_SIZE, 2 * PAGE_SIZE)
        )
        self.assertEqual(tree.requests, [(shortfall, shortfall)])

        evicts_cached = _StubTreeCache(on_evict=lambda: alloc.free(cached))
        self.assertIsNone(
            alloc.reclaim_for_prealloc(evicts_cached, 2 * PAGE_SIZE, 2 * PAGE_SIZE)
        )
        self.assertEqual(evicts_cached.requests, [(shortfall, shortfall)])

    def test_decode_fills_the_caller_host_buffer(self):
        """Host-staged decode passes a host buffer for the new full-attention
        locations through the allocator; both unified layers must accept it,
        fill it, and map the locations to the SWA side."""
        alloc = self._make_allocator()
        req = _extend(alloc, PAGE_SIZE)
        seq_lens = torch.tensor([PAGE_SIZE + 1], dtype=torch.int64)
        out_indices_cpu = torch.empty(1, dtype=torch.int64)
        loc = alloc.alloc_decode(
            seq_lens, seq_lens, req[-1:], out_indices_cpu=out_indices_cpu
        )
        self.assertIsNotNone(loc)
        self.assertEqual(out_indices_cpu.tolist(), loc.tolist())
        self.assertGreater(int(alloc.full_to_swa_index_mapping[loc.long()]), 0)
        self.assertEqual(_held_and_free_pages(alloc), (2, 2, 4))

    def test_prealloc_fits_counts_both_sides_against_one_list(self):
        """A PD decode admits a preallocation only if its full and SWA pages fit
        the one free list together; each side's budget counts every free page,
        so checking the sides apart admits requests the allocation then fails."""
        alloc = self._make_allocator()
        _extend(alloc, 2 * PAGE_SIZE)

        def fits(full_pages, swa_pages, tree):
            # The budgets a decode passes with no headroom held back.
            return alloc.prealloc_fits(
                tree,
                full_pages * PAGE_SIZE,
                swa_pages * PAGE_SIZE,
                full_budget_tokens=alloc.full_available_size()
                + tree.full_evictable_size(),
                swa_budget_tokens=alloc.swa_available_size()
                + tree.swa_evictable_size(),
            )

        # Four pages are free: three full and two SWA pages each fit alone.
        self.assertFalse(fits(3, 2, _StubTreeCache()))
        self.assertTrue(fits(2, 2, _StubTreeCache()))
        # A page the tree can evict counts, and the decode reclaims it jointly.
        self.assertTrue(fits(3, 2, _StubTreeCache(swa_evictable=PAGE_SIZE)))
        self.assertTrue(alloc.prealloc_fits_assumes_reclaim())

    def test_prealloc_ceiling_counts_both_sides_against_the_whole_pool(self):
        alloc = self._make_allocator()
        _extend(alloc, 2 * PAGE_SIZE)
        # Pages in use do not lower the ceiling; both sides share all eight.
        self.assertTrue(alloc.prealloc_ceiling_fits(5 * PAGE_SIZE, 3 * PAGE_SIZE))
        self.assertFalse(alloc.prealloc_ceiling_fits(5 * PAGE_SIZE, 3 * PAGE_SIZE + 1))

    def test_grouped_prealloc_charges_a_page_id_per_group(self):
        """A PD decode on a pool with three SWA groups prices each SWA page of a
        preallocation at three page ids in admission, ceiling and reclaim."""
        alloc, _ = _grouped_allocator(self)
        # One full page and one SWA page of three ids are held; four are free.
        cached = _extend(alloc, PAGE_SIZE)

        def fits(full_pages, swa_pages, tree):
            return alloc.prealloc_fits(
                tree,
                full_pages * PAGE_SIZE,
                swa_pages * PAGE_SIZE,
                full_budget_tokens=alloc.full_available_size()
                + tree.full_evictable_size(),
                swa_budget_tokens=alloc.swa_available_size()
                + tree.swa_evictable_size(),
            )

        self.assertTrue(fits(1, 1, _StubTreeCache()))
        # One page more on either side needs five or seven ids.
        self.assertFalse(fits(2, 1, _StubTreeCache()))
        self.assertFalse(fits(1, 2, _StubTreeCache()))
        self.assertTrue(fits(1, 2, _StubTreeCache(swa_evictable=PAGE_SIZE)))
        # The whole pool holds two full and two SWA pages, not a token more.
        self.assertTrue(alloc.prealloc_ceiling_fits(2 * PAGE_SIZE, 2 * PAGE_SIZE))
        self.assertFalse(alloc.prealloc_ceiling_fits(2 * PAGE_SIZE, 2 * PAGE_SIZE + 1))
        # Reclaim asks the tree for the three missing ids, or one SWA page.
        tree = _StubTreeCache(on_evict=lambda: alloc.free(cached))
        self.assertIsNone(alloc.reclaim_for_prealloc(tree, PAGE_SIZE, 2 * PAGE_SIZE))
        self.assertEqual(tree.requests, [(3 * PAGE_SIZE, PAGE_SIZE)])


def _radix_tree(alloc, window):
    """A FULL + SWA radix tree (Python core) over ``alloc``."""
    return UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=ReqToTokenPool(
                size=4, max_context_len=256, device="cpu", enable_memory_saver=False
            ),
            token_to_kv_pool_allocator=alloc,
            page_size=PAGE_SIZE,
            eviction_policy="lru",
            sliding_window_size=window,
            tree_components=(ComponentType.FULL, ComponentType.SWA),
            tree_core_backend="python",
        )
    )


def _unified_arm(
    *, total_pages, static_full_pages, window, policy, retain=10240, free_oow=False
):
    """Unified pool (one full and three sliding-window layers, three groups)
    whose static split is ``static_full_pages`` full pages and the rest SWA;
    ``policy`` and ``retain`` None keep the defaults."""
    pool = TpuUnifiedSWAKVPool(
        size=total_pages * PAGE_SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        head_num=1,
        head_dim=128,
        swa_head_num=1,
        swa_head_dim=128,
        swa_v_head_dim=128,
        swa_attention_layer_ids=[1, 2, 3],
        full_attention_layer_ids=[0],
        device="cpu",
        start_layer=0,
    )
    with (
        envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.override(free_oow),
        (
            contextlib.nullcontext()
            if policy is None
            else tpu_envs.SGLANG_TPU_UNIFIED_KV_EVICTION.override(policy)
        ),
        (
            contextlib.nullcontext()
            if retain is None
            else tpu_envs.SGLANG_TPU_UNIFIED_KV_COMPACT_RETAIN_TOKENS.override(retain)
        ),
    ):
        alloc = TpuUnifiedSWATokenToKVPoolAllocator(
            total_pages * PAGE_SIZE,
            page_size=PAGE_SIZE,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=pool,
            need_sort=False,
            static_full_tokens=static_full_pages * PAGE_SIZE,
        )
    return alloc, _radix_tree(alloc, window)


def _static_arm(*, full_pages, swa_pages, window):
    """The static pools the unified arm of the same bytes stands for."""
    pool = SWAKVPool(
        size=full_pages * PAGE_SIZE,
        size_swa=swa_pages * PAGE_SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        head_num=1,
        head_dim=128,
        swa_head_num=1,
        swa_head_dim=128,
        swa_v_head_dim=128,
        swa_attention_layer_ids=[1, 2, 3],
        full_attention_layer_ids=[0],
        device="cpu",
        token_to_kv_pool_class=TpuFusedMHATokenToKVPool,
    )
    alloc = swa_allocator_module.SWATokenToKVPoolAllocator(
        full_pages * PAGE_SIZE,
        size_swa=swa_pages * PAGE_SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=pool,
        need_sort=False,
    )
    return alloc, _radix_tree(alloc, window)


def _serve(alloc, cache, tokens):
    """One prompt as the scheduler prefills it: match and lock the hit, evict
    for the new tokens, allocate them after the hit, cache the prompt; returns
    the hit length."""
    match = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))
    prefix = match.device_indices.to(torch.int64)
    lock = cache.inc_lock_ref(match.last_device_node)
    new = len(tokens) - len(prefix)
    alloc.evict_to_free_tokens(cache, new)
    pre = torch.tensor([len(prefix)], dtype=torch.int64)
    seq = torch.tensor([len(tokens)], dtype=torch.int64)
    last = torch.tensor([int(prefix[-1]) if len(prefix) else -1], dtype=torch.int64)
    loc = alloc.alloc_extend(pre, pre, seq, seq, last, new)
    assert loc is not None, "the pool cannot hold one prompt"
    cache.insert(
        InsertParams(
            key=RadixKey(array("q", tokens)),
            value=torch.cat([prefix, loc.to(torch.int64)]),
            prev_prefix_len=len(prefix),
        )
    )
    cache.dec_lock_ref(match.last_device_node, lock.to_dec_params())
    return len(prefix)


def _hit(cache, tokens):
    return len(
        cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)))
        ).device_indices
    )


def _live_pages(cache):
    """{prompt prefix ending at a cached page: (full KV live, SWA KV live)}."""
    out, stack = {}, [(cache.tree_core.root_node, ())]
    while stack:
        node, path = stack.pop()
        for child in node.children.values():
            tokens = path + tuple(child.key)
            full = child.component_data[ComponentType.FULL].value is not None
            swa = child.component_data[ComponentType.SWA].value is not None
            for end in range(len(path) + PAGE_SIZE, len(tokens) + 1, PAGE_SIZE):
                out[tokens[:end]] = (full, swa)
            stack.append((child, tokens))
    return out


def _sessions(num_sessions, num_turns, turn_tokens):
    """Prompts of interleaved multi-turn sessions: each turn resends the
    session's history and appends ``turn_tokens`` tokens."""
    prompts = []
    for turn in range(1, num_turns + 1):
        for s in range(num_sessions):
            base = 10_000 * (s + 1)
            prompts.append(list(range(base, base + turn * turn_tokens)))
    return prompts


class TestUnifiedPoolEvictionOrder(CustomTestCase):
    """Shared-pool pressure policies on one full and three sliding-window
    layers (three groups), page 4."""

    SESSION = list(range(1000, 1024))  # three 8-token turns: 2 interior nodes + leaf

    def setUp(self):
        _tpu_swa_allocator_branches(self)
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)

    def _chain(
        self,
        *,
        policy,
        total_pages=36,
        static_full_pages=24,
        window=4,
        sessions=None,
        retain=10240,
        free_oow=False,
    ):
        alloc, cache = _unified_arm(
            total_pages=total_pages,
            static_full_pages=static_full_pages,
            window=window,
            policy=policy,
            retain=retain,
            free_oow=free_oow,
        )
        # Each session's turns are inserted in order, as each turn's request
        # caches its prompt: the earlier turns become interior nodes.
        for tokens, turn_ends in sessions or [(self.SESSION, (8, 16, 24))]:
            loc = _extend(alloc, len(tokens)).to(torch.int64)
            prev = 0
            for end in turn_ends:
                cache.insert(
                    InsertParams(
                        key=RadixKey(array("q", tokens[:end])),
                        value=loc[:end],
                        prev_prefix_len=prev,
                    )
                )
                prev = end
        return alloc, cache

    def _assert_closed(self, alloc, cache):
        # Every page id is free, a cached full page, or one of a cached
        # sliding-window page's group ids.
        self.assertEqual(
            alloc._free_page_count()
            + cache.full_evictable_size() // PAGE_SIZE
            + 3 * cache.swa_evictable_size() // PAGE_SIZE,
            alloc.size_full // PAGE_SIZE,
        )

    def test_compact_is_the_default_policy(self):
        alloc, _ = _unified_arm(
            total_pages=36, static_full_pages=24, window=4, policy=None, retain=None
        )
        self.assertEqual(alloc._eviction_policy, "compact")
        self.assertEqual(alloc._compact_retain_tokens, 4096)

    def test_full_first_evicts_lru_full_kv(self):
        alloc, cache = self._chain(policy="full_first")
        self.assertTrue(alloc.evict_to_free_tokens(cache, 5 * PAGE_SIZE))
        self.assertLess(cache.full_evictable_size(), len(self.SESSION))
        self._assert_closed(alloc, cache)

    def test_static_shadow_reclaims_the_kind_over_its_share(self):
        """SWA holds 18 ids against a static share of 12, full 6 pages of 24:
        the static SWA pool would be the full one, so the SWA walk runs
        (oldest interior turns tombstoned) and every full page stays."""
        alloc, cache = self._chain(policy="static_shadow")
        self.assertEqual(alloc._free_page_count(), 12)
        self._assert_closed(alloc, cache)
        # Five new pages on each side need 20 ids: 8 short.
        self.assertTrue(alloc.evict_to_free_tokens(cache, 5 * PAGE_SIZE))
        self.assertEqual(cache.full_evictable_size(), len(self.SESSION))
        self.assertLess(cache.swa_evictable_size(), len(self.SESSION))
        self.assertGreaterEqual(cache.swa_evictable_size(), 8)
        self._assert_closed(alloc, cache)
        self.assertEqual(_hit(cache, self.SESSION + [7] * 8), len(self.SESSION))

    def test_static_shadow_evicts_full_kv_when_full_is_over_its_share(self):
        alloc, cache = self._chain(policy="static_shadow", static_full_pages=3)
        self.assertTrue(alloc.evict_to_free_tokens(cache, 5 * PAGE_SIZE))
        self.assertLess(cache.full_evictable_size(), len(self.SESSION))
        self._assert_closed(alloc, cache)

    def test_static_shadow_takes_the_kind_the_request_pushes_over_its_share(self):
        """Full KV holds 20 of its 24 pages, SWA 6 of its 12 ids: three new pages
        on each side fit the static full pool and overflow the static SWA pool
        by one page, so the SWA walk is asked for one page, though full KV holds
        the larger fraction of its share."""
        alloc, _ = _unified_arm(
            total_pages=36, static_full_pages=24, window=4, policy="static_shadow"
        )
        _extend_swa_tail(alloc, seq_len=20 * PAGE_SIZE, swa_tail_len=2 * PAGE_SIZE)
        tree = _StubTreeCache()
        alloc.evict_to_free_tokens(tree, 3 * PAGE_SIZE)
        self.assertEqual(tree.requests[0], (0, PAGE_SIZE))
        self.assertEqual(tree.walks, [])

    def test_static_shadow_gives_both_overshoots_in_one_walk(self):
        """Full KV holds 22 of its 24 pages, SWA 12 of its 12 ids: three new
        pages on each side overflow both static pools, by one full page and
        three SWA pages, which one walk is asked for as the static pools ask."""
        alloc, _ = _unified_arm(
            total_pages=36, static_full_pages=24, window=4, policy="static_shadow"
        )
        _extend_swa_tail(alloc, seq_len=22 * PAGE_SIZE, swa_tail_len=4 * PAGE_SIZE)
        tree = _StubTreeCache()
        alloc.evict_to_free_tokens(tree, 3 * PAGE_SIZE)
        self.assertEqual(tree.walks, [(PAGE_SIZE, 3 * PAGE_SIZE)])

    def test_swa_first_asks_for_the_swa_walk_then_full_first(self):
        """Whatever the shares: the SWA walk for the whole shortfall, then the
        joint full-first request for what is left."""
        for static_full_pages in (24, 3):
            with self.subTest(static_full_pages=static_full_pages):
                alloc, _ = _unified_arm(
                    total_pages=36,
                    static_full_pages=static_full_pages,
                    window=4,
                    policy="swa_first",
                )
                # 20 full pages and 2 SWA pages of 3 ids: 10 ids free. Three new
                # pages on each side need 12: 2 short, one SWA page.
                _extend_swa_tail(
                    alloc, seq_len=20 * PAGE_SIZE, swa_tail_len=2 * PAGE_SIZE
                )
                tree = _StubTreeCache()
                self.assertFalse(alloc.evict_to_free_tokens(tree, 3 * PAGE_SIZE))
                self.assertEqual(
                    tree.requests, [(0, PAGE_SIZE), (2 * PAGE_SIZE, PAGE_SIZE)]
                )
                self.assertEqual(tree.walks, [])

    def test_swa_first_tombstones_interior_sliding_window_kv_before_full_kv(self):
        """With full KV over its share (static_shadow would evict full KV here):
        the oldest interior turns lose only SWA KV, every full page stays, the
        chain still hits."""
        alloc, cache = self._chain(policy="swa_first", static_full_pages=3)
        self.assertEqual(alloc._free_page_count(), 12)
        # Five new pages on each side need 20 ids: 8 short.
        self.assertTrue(alloc.evict_to_free_tokens(cache, 5 * PAGE_SIZE))
        self.assertEqual(cache.full_evictable_size(), len(self.SESSION))
        self.assertLess(cache.swa_evictable_size(), len(self.SESSION))
        self.assertGreaterEqual(cache.swa_evictable_size(), 8)
        self.assertGreaterEqual(alloc._free_page_count(), 20)
        self._assert_closed(alloc, cache)
        self.assertEqual(_hit(cache, self.SESSION + [7] * 8), len(self.SESSION))

    def test_compact_first_drops_full_kv_no_match_can_use(self):
        """A leaf whose window was tombstoned (a 4-token leaf under an interior
        node whose SWA KV is gone, window 8) cannot end a match; it and then its
        parent go before a live session loses SWA or full KV (compact and
        swa_first)."""
        for policy in ("compact", "swa_first"):
            with self.subTest(policy=policy):
                self._drops_full_kv_no_match_can_use(policy)

    def _drops_full_kv_no_match_can_use(self, policy):
        idle = list(range(2000, 2020))
        alloc, cache = self._chain(
            policy=policy,
            total_pages=48,
            window=8,
            sessions=[(idle, (16, 20)), (self.SESSION, (8, 16, 24))],
        )
        # The idle session's interior turn loses its SWA KV (LRU), as an
        # earlier SWA walk would leave it.
        cache.evict(EvictParams(swa_num_tokens=16))
        self.assertEqual(_hit(cache, idle + [7] * 8), 0)
        self.assertEqual(alloc._free_page_count(), 16)
        # Six new pages on each side need 24 ids: 8 short, exactly what the
        # dead idle chain holds (4 + 1 full pages, 1 SWA page of 3 ids).
        self.assertTrue(alloc.evict_to_free_tokens(cache, 6 * PAGE_SIZE))
        self.assertEqual(cache.full_evictable_size(), len(self.SESSION))
        self.assertEqual(cache.swa_evictable_size(), len(self.SESSION))
        self.assertEqual(alloc._free_page_count(), 24)
        self._assert_closed(alloc, cache)
        self.assertEqual(_hit(cache, self.SESSION + [7] * 8), len(self.SESSION))

    def test_compact_tombstones_cold_heads_first_and_keeps_tips_matchable(self):
        """Two 32-token prompts, the hot one matched last; window 4, R 16.
        Compaction splits the cold prompt at 16 and tombstones its head: no full
        KV goes, the hot prompt is untouched, and a match ending within R of the
        cold tip (at 28) still hits while one inside the head (at 12) cannot."""
        cold, hot = list(range(3000, 3032)), list(range(4000, 4032))
        alloc, cache = self._chain(
            policy="compact",
            total_pages=80,
            static_full_pages=40,
            retain=16,
            sessions=[(cold, (32,)), (hot, (32,))],
        )
        _hit(cache, hot)  # last use
        # 16 full pages and 16 SWA pages of 3 ids hold 64 of 80 ids. Seven new
        # pages on each side need 28: 12 short, the cold head's 4 SWA pages.
        self.assertEqual(alloc._free_page_count(), 16)
        self.assertTrue(alloc.evict_to_free_tokens(cache, 7 * PAGE_SIZE))
        self.assertEqual(alloc._free_page_count(), 28)
        self.assertEqual(cache.full_evictable_size(), 64)
        self.assertEqual(cache.swa_evictable_size(), 64 - 16)
        self.assertEqual(_hit(cache, hot + [7] * 4), 32)
        self.assertEqual(_hit(cache, cold[:28] + [7] * 4), 28)
        self.assertEqual(_hit(cache, cold[:12] + [7] * 4), 0)
        self._assert_closed(alloc, cache)

    def test_compaction_keeps_one_window_and_page_at_each_chunk_end(self):
        """A 96-token prompt cached in three 32-token chunks, window 7 (just
        under two pages): the insert path splits each chunk's leaf one
        page-aligned window (8 tokens) before its end to cap its SWA lock. Compaction keeps
        window + page (12 tokens) at each chunk end, held by the 8-token tail
        and 4 tokens of its head, and R 12 at the tip; nothing at the lock
        split, where no prompt ended."""
        prompt = list(range(3000, 3096))
        alloc, cache = self._chain(
            policy="compact",
            total_pages=200,
            static_full_pages=100,
            window=7,
            retain=12,
            sessions=[(prompt, (32, 64, 96))],
        )
        self.assertEqual(cache.compact_swa(12, lambda: False), 3 * 20)
        self.assertEqual(cache.swa_evictable_size(), 3 * 12)
        self.assertEqual(cache.full_evictable_size(), 96)
        self._assert_closed(alloc, cache)
        # Chunk ends and a page before them end matches; the lock splits (24,
        # 56, 88) and the tombstoned heads fall back to the previous chunk end.
        for end, hit in (
            (96, 96),
            (92, 92),
            (88, 64),
            (64, 64),
            (60, 60),
            (56, 32),
            (32, 32),
            (28, 28),
            (24, 0),
        ):
            self.assertEqual(_hit(cache, prompt[:end] + [7] * 4), hit, end)

    def test_exhausted_compaction_falls_back_to_the_walk_and_reclaims_dead_full_kv(
        self,
    ):
        """Two 32-token prompts compacted to their window + page tails (window 7,
        R 12): further SWA pressure walks the SWA LRU as the static pools do,
        tombstoning the cold prompt's kept head piece and then evicting its tail
        leaf whole. That leaves full KV no match can use; the next pressured
        allocation's unmatchable pass takes it before any live KV."""
        cold, hot = list(range(4000, 4032)), list(range(5000, 5032))
        alloc, cache = self._chain(
            policy="compact",
            total_pages=64,
            static_full_pages=40,
            window=7,
            retain=12,
            sessions=[(cold, (32,)), (hot, (32,))],
        )
        _hit(cache, hot)  # last use
        self.assertEqual(alloc._free_page_count(), 0)

        held = []

        def pressure(pages):
            self.assertTrue(alloc.evict_to_free_tokens(cache, pages * PAGE_SIZE))
            # Closure, with the page ids the earlier requests took (a full page
            # and three SWA ids per page).
            self.assertEqual(
                alloc._free_page_count()
                + cache.full_evictable_size() // PAGE_SIZE
                + 3 * cache.swa_evictable_size() // PAGE_SIZE
                + 4 * sum(held),
                64,
            )
            _extend(alloc, pages * PAGE_SIZE)  # the request takes its pages
            held.append(pages)

        def kv(prompt, end):
            # (full, SWA) live for the page ending at ``end``; read without a
            # match, which would refresh the LRU order under test.
            return _live_pages(cache).get(tuple(prompt[:end]))

        pressure(5)  # compaction alone: both heads tombstoned
        self.assertEqual(cache.swa_evictable_size(), 2 * 12)
        self.assertEqual(cache.full_evictable_size(), 64)
        self.assertEqual(
            [kv(cold, e) for e in (20, 24, 32)], [(True, False)] + [(True, True)] * 2
        )
        pressure(3)  # walk: the cold head piece (interior) tombstoned; the
        # 8-token tail still holds a window, so the cold tip still ends a match
        self.assertEqual(
            [kv(cold, e) for e in (24, 28, 32)], [(True, False)] + [(True, True)] * 2
        )
        pressure(3)  # walk: the cold tail leaf goes whole, then the hot head piece
        self.assertEqual(
            [kv(cold, e) for e in (20, 24, 32)], [(True, False)] * 2 + [None]
        )
        self.assertEqual([kv(hot, e) for e in (24, 32)], [(True, False), (True, True)])
        # The cold prompt's remaining 24 tokens of full KV serve no match.
        self.assertEqual(cache.full_evictable_size(), 24 + 32)
        pressure(3)  # unmatchable pass: the dead cold chain; walk: the hot tail
        self.assertEqual([kv(cold, e) for e in (20, 24)], [None, None])
        self.assertEqual(cache.full_evictable_size(), 24)
        self.assertEqual(cache.swa_evictable_size(), 0)
        self.assertEqual(_hit(cache, hot + [7] * 4), 0)

    def test_with_out_of_window_frees_every_policy_is_full_first(self):
        alloc, cache = self._chain(policy="compact", free_oow=True)
        self.assertTrue(alloc.evict_to_free_tokens(cache, 5 * PAGE_SIZE))
        self.assertLess(cache.full_evictable_size(), len(self.SESSION))
        self._assert_closed(alloc, cache)


class TestUnifiedPoolAgainstStaticPools(CustomTestCase):
    """Static pools whose SWA pool overflows against unified pools of the same
    bytes (the static split for the shares), one full and three sliding-window
    layers, window 4, page 4."""

    WARM = _sessions(num_sessions=3, num_turns=2, turn_tokens=8)
    SEQUENCE = _sessions(num_sessions=4, num_turns=3, turn_tokens=24)

    def setUp(self):
        _tpu_swa_allocator_branches(self)
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)

    @staticmethod
    def _arms(full_pages, swa_pages, *policies, retain=10240):
        """The static arm, then a unified arm per policy."""
        arms = [_static_arm(full_pages=full_pages, swa_pages=swa_pages, window=4)]
        for policy in policies:
            arms.append(
                _unified_arm(
                    total_pages=full_pages + 3 * swa_pages,
                    static_full_pages=full_pages,
                    window=4,
                    policy=policy,
                    retain=retain,
                )
            )
        return arms

    def test_static_shadow_evicts_a_prefix_of_what_the_static_pools_evict(self):
        """From one state (three sessions of two 8-token turns fill the 12-page
        SWA pool), each new prompt makes static_shadow tombstone and evict a
        part of what the static pools do, in the same order: never anything
        they keep, all of their overshoot once both kinds overflow."""
        # (static full pages, new prompt tokens): one full page short of the
        # prompt (both kinds overflow), then 2 and 4 spare full pages.
        for full_pages, prompt_tokens, same in (
            (14, 16, True),
            (18, 8, False),
            (22, 24, False),
        ):
            with self.subTest(full_pages=full_pages, prompt_tokens=prompt_tokens):
                static, shadow = self._arms(full_pages, 12, "static_shadow")
                for prompt in self.WARM:
                    self.assertEqual(_serve(*static, prompt), _serve(*shadow, prompt))
                before = _live_pages(static[1])
                self.assertEqual(_live_pages(shadow[1]), before)
                prompt = list(range(90_000, 90_000 + prompt_tokens))
                _serve(*static, prompt)
                _serve(*shadow, prompt)
                static_live = _live_pages(static[1])
                shadow_live = _live_pages(shadow[1])
                lost_static = {
                    p for p, kv in before.items() if static_live.get(p) != kv
                }
                lost_shadow = {
                    p for p, kv in before.items() if shadow_live.get(p) != kv
                }
                self.assertTrue(lost_shadow)
                self.assertLessEqual(lost_shadow, lost_static)
                for p in lost_shadow:
                    self.assertEqual(shadow_live.get(p), static_live.get(p))
                self.assertEqual(lost_shadow == lost_static, same)

    def test_static_shadow_hits_at_least_what_the_static_pools_hit(self):
        """Four sessions of three 24-token turns on a 24-page static SWA pool:
        every turn hits at least as many tokens on static_shadow. The final
        states are not nested: a static miss re-prefills the whole prompt and
        refills the SWA KV its walk tombstoned, which a unified hit leaves
        tombstoned."""
        static, shadow = self._arms(96, 24, "static_shadow")
        static_hits = [_serve(*static, prompt) for prompt in self.SEQUENCE]
        shadow_hits = [_serve(*shadow, prompt) for prompt in self.SEQUENCE]
        # The static SWA pool overflowed: its walk tombstoned SWA KV.
        self.assertTrue(any(not swa for _, swa in _live_pages(static[1]).values()))
        for static_hit, shadow_hit in zip(static_hits, shadow_hits):
            self.assertGreaterEqual(shadow_hit, static_hit)
        self.assertGreater(sum(shadow_hits), sum(static_hits))

    def test_compact_keeps_the_node_ends_and_tips_static_shadow_keeps(self):
        """On the same sequence, R 12: every turn hits at least as many tokens on
        compact, and every node end and point within R of a tip that static_shadow
        can end a match at, compact can too. Interior points farther from both
        are not nested: compaction trims the heads of many nodes where the
        static walk tombstones fewer whole ones."""
        _, shadow, compact = self._arms(96, 24, "static_shadow", "compact", retain=12)
        shadow_hits = [_serve(*shadow, prompt) for prompt in self.SEQUENCE]
        compact_hits = [_serve(*compact, prompt) for prompt in self.SEQUENCE]
        for shadow_hit, compact_hit in zip(shadow_hits, compact_hits):
            self.assertGreaterEqual(compact_hit, shadow_hit)
        last = self.SEQUENCE[-4:]
        probes = [
            p[:end] + [7] * 4 for p in last for end in range(24, len(p) + 1, 24)
        ] + [p[: len(p) - 8] + [7] * 4 for p in last]
        for probe in probes:
            self.assertGreaterEqual(_hit(compact[1], probe), _hit(shadow[1], probe))


class TestUnifiedPoolAttentionMetadata(CustomTestCase):
    def test_every_sliding_window_group_reads_its_own_page_table(self):
        """The attention metadata gives each sliding-window layer group the page
        table of its own pages; without it, the layers of groups 1 and 2 would
        read group 0's pages."""
        from sglang_tpu.attention_pallas_rpa_v3 import TpuPallasRpaV3AttnBackend

        alloc, pool = _grouped_allocator(self)
        loc = _extend(alloc, PAGE_SIZE + 1)
        req_to_token = torch.zeros(1, 16 * PAGE_SIZE, dtype=torch.int32)
        req_to_token[0, : loc.numel()] = loc
        override = get_context().override_server_args(page_size=PAGE_SIZE)
        override.install()
        self.addCleanup(override.restore)
        model_runner = SimpleNamespace(
            device=torch.device("cpu"),
            page_size=PAGE_SIZE,
            model_config=SimpleNamespace(context_len=8 * PAGE_SIZE),
            server_args=SimpleNamespace(page_size=PAGE_SIZE),
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            token_to_kv_pool=pool,
            is_hybrid_swa=True,
            kv_cache_dtype=torch.bfloat16,
            is_draft_worker=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        with patch(
            "sglang_tpu.attention_pallas_rpa_base._build_pallas_kernel_op",
            return_value=None,
        ):
            backend = TpuPallasRpaV3AttnBackend(model_runner)
        self.assertEqual(backend._swa_layer_groups, {1: 0, 2: 1, 3: 2})

        seq_lens = torch.tensor([loc.numel()], dtype=torch.int32)
        indices = torch.tensor([0], dtype=torch.int64)
        metadata = backend._build_metadata(
            SimpleNamespace(
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens.clone(),
                req_pool_indices=indices,
                req_pool_indices_cpu=indices.clone(),
                forward_mode=ForwardMode.DECODE,
                extend_seq_lens=None,
                extend_seq_lens_cpu=None,
                spec_info=None,
            ),
            non_empty_bs=1,
        )
        swa_pages = metadata.swa_page_indices[:2].long()
        self.assertEqual(
            swa_pages.tolist(),
            (
                alloc.translate_loc_from_full_to_swa(loc[::PAGE_SIZE]) // PAGE_SIZE
            ).tolist(),
        )
        self.assertEqual(
            metadata.swa_group_page_indices[:, :2].tolist(),
            pool.swa_group_pages[:, swa_pages].tolist(),
        )
        # Each layer reads its group's table, also on paths that bypass
        # _forward (the fused prefill ops of an attention backend).
        backend.forward_metadata = metadata
        for layer_id, is_swa_layer, table, expected in (
            (0, False, None, metadata.page_indices),
            (1, True, 0, metadata.swa_page_indices),
            (2, True, 1, metadata.swa_group_page_indices[0]),
            (3, True, 2, metadata.swa_group_page_indices[1]),
        ):
            pages, page_group = backend._layer_page_indices(
                layer_id, is_swa_layer=is_swa_layer
            )
            self.assertEqual(page_group, table)
            self.assertEqual(pages.tolist(), expected.tolist())


class TestUnifiedPoolImportOrder(CustomTestCase):
    def test_attention_backend_does_not_load_the_kv_pools(self):
        """Out-of-tree plugins may swap the KV pool base classes after the
        attention backends load; loading the pool modules with them would bind
        the TPU pools to the stock classes."""
        code = (
            "import sys, sglang_tpu.attention_pallas_rpa_base; "
            "print([m for m in ('sglang_tpu.kv_pool', 'sglang_tpu.unified_kv_pool') "
            "if m in sys.modules])"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        self.assertEqual(out.stdout.strip().splitlines()[-1], "[]")


class TestUnifiedPoolEnabled(CustomTestCase):
    def test_field_gates_and_rejects_unsupported_configs(self):
        mla_model = SimpleNamespace(
            is_hybrid_swa=True, use_mla_backend=True, mambaish_config=None
        )
        with get_context().override_server_args(tpu_unified_kv_pool=False):
            self.assertFalse(unified_kv_pool_enabled(mla_model))
        # Set on a configuration it cannot serve, the field fails loudly.
        with get_context().override_server_args(tpu_unified_kv_pool=True):
            with self.assertRaisesRegex(ValueError, "MLA"):
                unified_kv_pool_enabled(mla_model)
            # A model without sliding-window layers is rejected as such, not for
            # whatever else it would fail on.
            with self.assertRaisesRegex(ValueError, "without sliding-window layers"):
                unified_kv_pool_enabled(
                    SimpleNamespace(
                        is_hybrid_swa=False, use_mla_backend=True, mambaish_config=None
                    )
                )

    def test_hierarchical_cache_needs_the_raiden_io_backend(self):
        # Only raiden's host pool is shared by both views; the torch_tpu host
        # pool would mirror one view's buffers.
        def enabled_with(io_backend):
            with (
                _fused_rpa_platform(),
                get_context().override_server_args(
                    tpu_unified_kv_pool=True,
                    enable_hierarchical_cache=True,
                    hicache_io_backend=io_backend,
                ),
                get_parallel().override(attn_tp_size=1),
            ):
                return unified_kv_pool_enabled(_hybrid_kvc())

        self.assertTrue(enabled_with("raiden"))
        with self.assertRaisesRegex(ValueError, "hierarchical cache"):
            enabled_with("torch_tpu")

    def test_pool_classes_must_declare_unified_support(self):
        """Another pool class serves the unified pool only if it declares that
        its backends read per-group page tables, and only with the same
        full-attention and sliding-window KV heads."""
        supported = type(
            "_SupportedSeqAlongLanePool",
            (TpuSeqAlongLaneKVPool,),
            {"supports_unified_kv_pool": True},
        )
        with (
            get_context().override_server_args(tpu_unified_kv_pool=True),
            get_parallel().override(attn_tp_size=1),
        ):
            with _fused_rpa_platform(supported):
                self.assertTrue(unified_kv_pool_enabled(_hybrid_kvc()))
                # The merge the fused layout allows is refused for this one.
                with self.assertRaisesRegex(
                    ValueError, "other than the full-attention"
                ):
                    unified_kv_pool_enabled(
                        _hybrid_kvc(head_dim=256, swa_kv_heads=2, swa_head_dim=128)
                    )
            with _fused_rpa_platform(TpuSeqAlongLaneKVPool):
                with self.assertRaisesRegex(ValueError, "TpuSeqAlongLaneKVPool"):
                    unified_kv_pool_enabled(_hybrid_kvc())

    def test_pd_disaggregation_is_refused_with_pp_or_merged_heads(self):
        # Raiden's pipeline transfer would register each shared buffer twice;
        # merged sliding-window heads have not been through a transfer.
        def enabled_with(mode, pp_size, kvc=None):
            with (
                _fused_rpa_platform(),
                get_context().override_server_args(
                    tpu_unified_kv_pool=True, disaggregation_mode=mode
                ),
                get_parallel().override(attn_tp_size=1, pp_size=pp_size),
            ):
                return unified_kv_pool_enabled(kvc or _hybrid_kvc())

        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode):
                self.assertTrue(enabled_with(mode, 1))
                with self.assertRaisesRegex(
                    ValueError, "PD disaggregation with pipeline parallelism"
                ):
                    enabled_with(mode, 2)
                self.assertTrue(
                    enabled_with(
                        mode, 1, _hybrid_kvc(full_layer_ids=(0,), swa_layer_ids=(1, 2))
                    )
                )
                with self.assertRaisesRegex(ValueError, "merged sliding-window heads"):
                    enabled_with(
                        mode,
                        1,
                        _hybrid_kvc(head_dim=256, swa_kv_heads=2, swa_head_dim=128),
                    )

    def test_a_decode_on_the_pipeline_transfer_is_refused(self):
        # A decode fed by a pipeline prefill takes raiden's pipeline transfer,
        # whose per-layer pools ignore the sliding-window groups.
        def enabled_with(mode):
            with (
                _fused_rpa_platform(),
                get_context().override_server_args(
                    tpu_unified_kv_pool=True, disaggregation_mode=mode
                ),
                get_parallel().override(attn_tp_size=1),
            ):
                return unified_kv_pool_enabled(_hybrid_kvc())

        # The override restores the variable only on a normal exit.
        with envs.SGLANG_RAIDEN_PIPELINE_TRANSFER.override(True):
            self.assertTrue(enabled_with("prefill"))
            with self.assertRaisesRegex(ValueError, "SGLANG_RAIDEN_PIPELINE_TRANSFER"):
                enabled_with("decode")

    def test_sliding_window_pages_must_match_or_merge_into_full_ones(self):
        # Sliding-window layers beyond the full count form groups, and two
        # sliding-window heads of 128 merge into one full head of 256; one full
        # head of 128 cannot hold them. In fp8 the full head's page pads its K
        # and V to four rows of 256, twice the sliding-window page's bytes.
        with (
            _fused_rpa_platform(),
            get_context().override_server_args(tpu_unified_kv_pool=True),
            get_parallel().override(attn_tp_size=1),
        ):
            self.assertTrue(
                unified_kv_pool_enabled(
                    _hybrid_kvc(full_layer_ids=(0,), swa_layer_ids=(1, 2))
                )
            )
            self.assertTrue(
                unified_kv_pool_enabled(
                    _hybrid_kvc(head_dim=256, swa_kv_heads=2, swa_head_dim=128)
                )
            )
            with self.assertRaisesRegex(ValueError, "nor fit as one full-attention"):
                unified_kv_pool_enabled(_hybrid_kvc(swa_kv_heads=2, swa_head_dim=128))
            fp8 = _hybrid_kvc(head_dim=256, swa_kv_heads=2, swa_head_dim=128)
            fp8.kv_cache_dtype_str = "fp8_e4m3"
            fp8.kv_cache_dtype = torch.float8_e4m3fn
            with self.assertRaisesRegex(ValueError, "nor fit as one full-attention"):
                unified_kv_pool_enabled(fp8)


class TestUnifiedHostPool(CustomTestCase):
    HICACHE_RATIO = 2.0

    def _host_pools(
        self, *, hicache_size=0, full_layer_ids=(0, 2, 4), swa_layer_ids=(1, 3)
    ):
        """Both kinds' raiden HiCache host pools over a unified pool of three
        full-attention and two sliding-window layers, as the hybrid assembler
        builds them: the platform picks each kind's host-pool class, and each
        kind gets its share of ``hicache_size`` by the views' host weights."""
        override = get_context().override_server_args(
            hicache_io_backend="raiden", hicache_size=hicache_size
        )
        override.install()
        self.addCleanup(override.restore)
        # The raiden engine needs a TPU; the mock records how it is built and
        # driven.
        engine_patch = patch("sglang_tpu.raiden_transfer.RaidenKVTransfer")
        engine_cls = engine_patch.start()
        self.addCleanup(engine_patch.stop)
        # The host pools' RAM admission reads this machine's free memory; give
        # every rank 1 GiB so the result does not depend on the runner.
        for patcher in (
            patch.object(host_base, "ranks_per_host", return_value=1),
            patch.object(
                host_base,
                "available_host_memory_bytes",
                return_value=host_base.HICACHE_HOST_MEMORY_RESERVE_BYTES + (1 << 30),
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        pool = _make_pool(full_layer_ids=full_layer_ids, swa_layer_ids=swa_layer_ids)
        platform = TpuSRTPlatform()
        views = (pool.full_kv_pool, pool.swa_kv_pool)
        host_sizes = hybrid_pool_assembler._split_hicache_size(hicache_size, views)
        full_host, swa_host = (
            platform.get_mha_kv_pool_host_cls(view)(
                view,
                self.HICACHE_RATIO,
                host_size,
                PAGE_SIZE,
                "page_first_fused",
                pool_label=label,
            )
            for view, host_size, label in zip(views, host_sizes, ("full", "swa"))
        )
        return pool, full_host, swa_host, engine_cls

    def test_both_kinds_share_one_engine_and_page_list(self):
        pool, full_host, swa_host, engine_cls = self._host_pools()
        backing = pool.full_kv_pool.backing
        # One engine over the three shared buffers, holding ratio x the whole
        # device pool (16 pages) plus the page the host sizing rounds up by.
        self.assertEqual(backing.layer_num, 3)
        engine_cls.assert_called_once_with(backing, 2 * NUM_PAGES + 1)
        # A host page holds a row of all three buffers, not of the two the
        # sliding-window layers use.
        self.assertEqual(swa_host.size_per_token, 3 * 512)

        taken = full_host.alloc(full_host.available_size())
        self.assertEqual(swa_host.available_size(), 0)
        self.assertIsNone(swa_host.alloc(PAGE_SIZE))
        # A page released through either kind's pool serves the other.
        swa_host.free(taken[:PAGE_SIZE])
        self.assertEqual(full_host.available_size(), PAGE_SIZE)
        full_host.free(taken[PAGE_SIZE:])
        self.assertEqual(swa_host.available_size(), swa_host.size)

        # A fused pool that is not a view keeps a raiden host pool of its own.
        plain_pool = TpuFusedMHATokenToKVPool(
            size=POOL_TOKENS,
            page_size=PAGE_SIZE,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=128,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
        )
        self.assertIs(
            TpuSRTPlatform().get_mha_kv_pool_host_cls(plain_pool),
            TpuRaidenHostKVCache,
        )

    def test_hicache_size_budget_sizes_the_shared_pool_once(self):
        # The assembler splits a 1 GB --hicache-size between the kinds by their
        # host weights; the one shared pool takes the whole budget once.
        pool, _, swa_host, engine_cls = self._host_pools(hicache_size=1)
        # 1e9 bytes at 1536 bytes per token are 651041 tokens: 162760 whole
        # pages plus the one the host sizing rounds up by.
        engine_cls.assert_called_once_with(pool.full_kv_pool.backing, 162761)
        self.assertEqual(swa_host.size, 162761 * PAGE_SIZE)

    def test_transfers_reach_the_shared_engine_with_unchanged_page_ids(self):
        pool, full_host, swa_host, engine_cls = self._host_pools()
        engine = engine_cls.return_value
        full_host.alloc(PAGE_SIZE)
        # Host pages 1 and 2 and device pages 5 and 3, all in the shared spaces.
        host_indices = swa_host.alloc(2 * PAGE_SIZE)
        device_indices = torch.cat(
            [
                torch.arange(5 * PAGE_SIZE, 6 * PAGE_SIZE),
                torch.arange(3 * PAGE_SIZE, 4 * PAGE_SIZE),
            ]
        )
        swa_host.backup_from_device_all_layer(
            pool.swa_kv_pool, host_indices, device_indices, "raiden"
        )
        engine.backup.assert_called_once_with([5, 3], [1, 2])
        swa_host.load_all_layer(pool.swa_kv_pool, host_indices, device_indices)
        engine.load.assert_called_once_with([1, 2], [5, 3])

    def test_a_grouped_sliding_window_page_moves_a_host_page_per_group(self):
        # One full and three SWA layers: an SWA page is a page id per group on
        # the device, and on the host as well.
        pool, full_host, swa_host, engine_cls = self._host_pools(
            full_layer_ids=(0,), swa_layer_ids=(1, 2, 3)
        )
        engine = engine_cls.return_value
        # The device allocator gave SWA page 5 the group pages 6 and 7.
        pool.swa_group_pages[:, 5] = torch.tensor([6, 7], dtype=torch.int32)
        host_indices = swa_host.alloc(PAGE_SIZE)
        self.assertEqual(full_host.available_size(), swa_host.size - 3 * PAGE_SIZE)
        device_indices = torch.arange(5 * PAGE_SIZE, 6 * PAGE_SIZE)

        swa_host.backup_from_device_all_layer(
            pool.swa_kv_pool, host_indices, device_indices, "raiden"
        )
        device_pages, host_pages = engine.backup.call_args.args
        self.assertEqual(device_pages, [5, 6, 7])
        self.assertEqual(host_pages[0], int(host_indices[0]) // PAGE_SIZE)
        self.assertEqual(len(set(host_pages)), 3)
        swa_host.load_all_layer(pool.swa_kv_pool, host_indices, device_indices)
        engine.load.assert_called_once_with(host_pages, [5, 6, 7])

        swa_host.free(host_indices)
        self.assertEqual(full_host.available_size(), swa_host.size)


# Two full-attention and five sliding-window layers: SWA groups of two, two and
# one layers over two shared buffers, as Muse Spark's 16 + 46 layers form
# groups of 16, 15 and 15.
PD_FULL_LAYERS = (0, 3)
PD_SWA_LAYERS = (1, 2, 4, 5, 6)
PD_GROUPS = 3
PD_POOL_PAGES = 64
PD_WINDOW = 2 * PAGE_SIZE
# Five pages and a token: the window starts mid-request.
PD_SEQ_LEN = 5 * PAGE_SIZE + 1
PD_ROOM = 7
PD_DECODE_PORT = 40000
PD_PREFILL_RANK = {"rank_ip": "127.0.0.1", "rank_port": 40001}


def _pd_side(
    testcase,
    *,
    pool_cls,
    full_layer_ids=PD_FULL_LAYERS,
    swa_layer_ids=PD_SWA_LAYERS,
):
    _tpu_swa_allocator_branches(testcase)
    size = PD_POOL_PAGES * PAGE_SIZE
    pool = _make_pool(
        full_layer_ids=full_layer_ids,
        swa_layer_ids=swa_layer_ids,
        pool_cls=pool_cls,
        size=size,
    )
    alloc = TpuUnifiedSWATokenToKVPoolAllocator(
        size,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=pool,
        need_sort=False,
    )
    return pool, alloc


def _window_swa_pages(alloc, full_loc, window_start):
    """The SWA payload prefill.py and decode.py build: the window's SWA pages."""
    swa_loc = alloc.translate_swa_indices_for_transfer(full_loc[window_start:])
    return [int(p) for p in kv_to_page_indices(swa_loc, PAGE_SIZE)]


def _pages(full_loc):
    return [int(p) for p in kv_to_page_indices(full_loc, PAGE_SIZE)]


def _raiden_manager(pool, *, prefill):
    """A raiden manager over ``pool`` whose two engines are mocks recording how
    they were built; no sockets."""
    mgr = object.__new__(RaidenKVManager)
    mgr.kv_args = SimpleNamespace(
        page_size=PAGE_SIZE,
        sliding_window_size=PD_WINDOW,
        state_types=[StateType.SWA],
        engine_rank=0,
    )
    mgr.server_args = SimpleNamespace(context_length=PD_POOL_PAGES * PAGE_SIZE)
    mgr.attn_tp_rank = 0
    mgr.attn_tp_size = 1
    mgr.local_ip = "127.0.0.1"
    mgr.rank_port = PD_DECODE_PORT
    mgr._local_rank = 0
    mgr._control_port = 31000
    mgr._timeout_s = 60.0
    mgr._dynamic_host_staging = True
    mgr._transfer_parallelism = 4
    mgr._engine = None
    mgr._swa_engine = None
    mgr._num_slots = lambda kv_caches, max_blocks, engines=1: 4
    mgr._state_lock = threading.Lock()
    mgr._status_lock = threading.Lock()
    mgr._announce_lock = threading.Lock()
    mgr.request_status = {}
    mgr.failure_records = {}
    mgr.failure_lock = threading.Lock()
    mgr.record_failure = Mock()
    mgr._aborted = {}
    mgr._completed_send = set()
    mgr._completed_recv = set()
    mgr._failed = set()
    mgr._native_failed = set()
    mgr._pulling_cleared = set()
    mgr._prefill = PrefillState(announce_pages=1) if prefill else None
    mgr._decode = None if prefill else DecodeState()
    engines = []

    def build_engine(**kwargs):
        engines.append(Mock(kwargs=kwargs, transfer_address="127.0.0.1:31001"))
        return engines[-1]

    native = SimpleNamespace(KVCacheManager=build_engine)
    with patch.dict(sys.modules, {"tpu_sync.api.torch.kv_cache_manager": native}):
        mgr.set_kv_caches(get_raiden_kv_caches(pool))
    return mgr, engines


def _static_swa_pool(pool_cls):
    """The same model's sliding-window layers in static pools: one tensor each."""
    return SWAKVPool(
        size=PD_POOL_PAGES * PAGE_SIZE,
        size_swa=PD_POOL_PAGES * PAGE_SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        head_num=1,
        head_dim=128,
        swa_attention_layer_ids=list(PD_SWA_LAYERS),
        full_attention_layer_ids=list(PD_FULL_LAYERS),
        device="cpu",
        token_to_kv_pool_class=pool_cls,
    )


def _send_metadata(d_mgr, *, pages, swa_pages):
    """The decode's send_metadata: records the room's pages and returns the
    pull request it sends to the prefill."""
    d_mgr._send_multipart_locked = Mock()
    d_mgr.register_decode_request(
        PD_ROOM,
        pages,
        d_mgr._swa_blocks_from_state_indices([swa_pages]),
        None,
        [PD_PREFILL_RANK],
    )
    return d_mgr._send_multipart_locked.call_args.args[1]


class _PullReqBeforeSwaLayout(WireMessage):
    """PULL_REQ as a prefill without the SWA layout field decodes it."""

    TAG = MessageTag.PULL_REQ
    room: int
    decode_ip: str
    decode_port: int
    decode_pages: int
    decode_swa_pages: int = 0


class TestUnifiedPoolPDTransfer(CustomTestCase):
    """Raiden PD with a pool whose sliding-window layers form several groups: a
    sliding-window page is a page id per group, and every group's page has to
    land in the decode's page of the same group."""

    POOL_CLS = TpuFusedMHATokenToKVPool

    def setUp(self):
        self.p_pool, self.p_alloc = _pd_side(self, pool_cls=self.POOL_CLS)
        self.d_pool, self.d_alloc = _pd_side(self, pool_cls=self.POOL_CLS)
        # The decode holds another request, so its page ids differ.
        _extend(self.d_alloc, 3 * PAGE_SIZE)
        self.p_full = _extend(self.p_alloc, PD_SEQ_LEN)
        self.window_start = (PD_SEQ_LEN - PD_WINDOW) // PAGE_SIZE * PAGE_SIZE
        # A PD decode preallocates SWA KV for the window only.
        self.d_full = _extend_swa_tail(
            self.d_alloc,
            seq_len=PD_SEQ_LEN,
            swa_tail_len=PD_SEQ_LEN - self.window_start,
        )

    def test_engines_register_each_shared_buffer_once(self):
        # Raiden copies a block's row of every tensor an engine wraps: a buffer
        # listed once per sliding-window layer would copy group 0's row thrice.
        mgr, (full_engine, swa_engine) = _raiden_manager(self.p_pool, prefill=True)
        buffers = self.p_pool.full_kv_pool.backing.kv_buffer
        for engine in (full_engine, swa_engine):
            registered = engine.kwargs["kv_caches"]
            self.assertEqual(len(registered), len(buffers))
            for tensor, buffer in zip(registered, buffers):
                self.assertIs(tensor, buffer)
        # A two-page window spans three pages from a mid-page start, and each
        # page moves as one block per group.
        self.assertEqual(swa_engine.kwargs["max_blocks"], 3 * PD_GROUPS)
        self.assertEqual(mgr._pulling.swa_capacity, 4 * 3 * PD_GROUPS)

    def test_one_group_moves_the_per_layer_buffers(self):
        # One group keeps the per-layer tensors and page ids, so the SWA engine
        # still pairs with a static pool's.
        pool, _ = _pd_side(
            self, pool_cls=self.POOL_CLS, full_layer_ids=(0, 2), swa_layer_ids=(1, 3)
        )
        groups = get_raiden_kv_caches(pool)
        self.assertEqual(groups.swa_groups, 1)
        self.assertEqual(len(groups.swa_buffers), len(groups.swa))
        for tensor, layer_tensor in zip(groups.swa_buffers, groups.swa):
            self.assertIs(tensor, layer_tensor)
        self.assertEqual(groups.swa_block_ids([5, 2]), [5, 2])

    def _closing_swa_read(self, *, decode_window_start):
        """One request's closing transfer; returns the SWA read's (prefill,
        decode) block pairs and the SWA blocks on the wire."""
        p_mgr, (_, p_swa_engine) = _raiden_manager(self.p_pool, prefill=True)
        d_mgr, (d_full_engine, d_swa_engine) = _raiden_manager(
            self.d_pool, prefill=False
        )
        p_mgr._register_sender_read = Mock()
        p_mgr._send_multipart_locked = Mock()

        # Prefill: the last chunk's send.
        p_swa_blocks = p_mgr._swa_blocks_from_state_indices(
            [_window_swa_pages(self.p_alloc, self.p_full, self.window_start)]
        )
        p_mgr.append_send_blocks(PD_ROOM, _pages(self.p_full), p_swa_blocks)
        p_mgr._prefill.send_aux[PD_ROOM] = b""
        p_mgr._prefill.reshard_finalized.add(PD_ROOM)
        # The decode's pull request; the prefill replies with BLOCKS.
        p_mgr._on_pull_req(
            _send_metadata(
                d_mgr,
                pages=_pages(self.d_full),
                swa_pages=_window_swa_pages(
                    self.d_alloc, self.d_full, decode_window_start
                ),
            )
        )
        (registered_swa,) = [
            c.args[4]
            for c in p_mgr._register_sender_read.call_args_list
            if c.args[1] is p_swa_engine
        ]
        frames = p_mgr._send_multipart_locked.call_args.args[1]

        d_mgr._on_blocks(frames)
        self.assertTrue(d_mgr.maybe_start_pull(PD_ROOM))
        d_full_engine.start_read.assert_called_once()
        req_id, uuid, _, remote, local = d_swa_engine.start_read.call_args.args
        self.assertEqual((req_id, uuid), (f"swa:{PD_ROOM}", PD_ROOM))
        # The prefill registers exactly the blocks the decode reads.
        self.assertEqual(remote, registered_swa)
        return list(zip(remote, local)), Blocks.from_frames(frames).swa_blocks

    def _group_pairs(self, first_token):
        """(prefill, decode) page of every group of each window page from
        ``first_token`` on."""

        def group_pages(pool, alloc, full_loc):
            return [
                [page] + pool.swa_group_pages[:, page].tolist()
                for page in _window_swa_pages(alloc, full_loc, first_token)
            ]

        pairs = []
        for p_groups, d_groups in zip(
            group_pages(self.p_pool, self.p_alloc, self.p_full),
            group_pages(self.d_pool, self.d_alloc, self.d_full),
        ):
            pairs += zip(p_groups, d_groups)
        return pairs

    def test_every_group_page_lands_in_the_decode_page_of_its_group(self):
        pairs, wire_blocks = self._closing_swa_read(
            decode_window_start=self.window_start
        )
        self.assertEqual(len(pairs), 3 * PD_GROUPS)
        self.assertEqual(pairs, self._group_pairs(self.window_start))
        self.assertEqual(wire_blocks, [p for p, _ in pairs])

    def test_a_decode_holding_the_window_head_pulls_whole_tail_pages(self):
        # A decode whose prefix covers the window's first page allocates the
        # last two pages only; the tail it pairs must be whole pages.
        start = self.window_start + PAGE_SIZE
        pairs, _ = self._closing_swa_read(decode_window_start=start)
        self.assertEqual(len(pairs), 2 * PD_GROUPS)
        self.assertEqual(pairs, self._group_pairs(start))

    def _assert_refused(self, p_mgr):
        """The prefill failed the room and told the decode why."""
        endpoint, frames = p_mgr._send_multipart_locked.call_args.args
        note = RoomFailed.from_frames(frames)
        self.assertEqual(
            (endpoint, note.room), (f"tcp://127.0.0.1:{PD_DECODE_PORT}", PD_ROOM)
        )
        self.assertIn("sliding-window KV layouts differ", note.reason)
        self.assertIn(str(PD_ROOM), p_mgr._failed)
        self.assertNotIn(PD_ROOM, p_mgr._prefill.decode_addr)
        self.assertNotIn(PD_ROOM, p_mgr._prefill.reshard_req)

    def test_a_static_prefill_refuses_a_unified_decode(self):
        # It would read 5 of the decode's 15 blocks: groups 1-2 of most window
        # pages would keep stale rows.
        p_mgr, _ = _raiden_manager(_static_swa_pool(self.POOL_CLS), prefill=True)
        d_mgr, _ = _raiden_manager(self.d_pool, prefill=False)
        p_mgr._send_multipart_locked = Mock()
        p_mgr._on_pull_req(
            _send_metadata(
                d_mgr,
                pages=_pages(self.d_full),
                swa_pages=_window_swa_pages(
                    self.d_alloc, self.d_full, self.window_start
                ),
            )
        )
        self._assert_refused(p_mgr)

    def test_a_unified_prefill_refuses_a_static_decode(self):
        p_mgr, _ = _raiden_manager(self.p_pool, prefill=True)
        d_mgr, _ = _raiden_manager(_static_swa_pool(self.POOL_CLS), prefill=False)
        p_mgr._send_multipart_locked = Mock()
        p_mgr._on_pull_req(_send_metadata(d_mgr, pages=[1, 2], swa_pages=[3]))
        self._assert_refused(p_mgr)

    def test_a_static_prefill_refuses_a_resharding_unified_decode(self):
        p_mgr, _ = _raiden_manager(_static_swa_pool(self.POOL_CLS), prefill=True)
        d_mgr, _ = _raiden_manager(self.d_pool, prefill=False)
        d_mgr._send_multipart_locked = Mock()
        d_mgr._register_reshard_request(
            PD_ROOM,
            _pages(self.d_full),
            d_mgr._swa_blocks_from_state_indices(
                [_window_swa_pages(self.d_alloc, self.d_full, self.window_start)]
            ),
            None,
            [PD_PREFILL_RANK, PD_PREFILL_RANK],
        )
        p_mgr._send_multipart_locked = Mock()
        p_mgr._on_reshard_req(d_mgr._send_multipart_locked.call_args.args[1])
        self._assert_refused(p_mgr)

    def test_only_a_unified_decode_extends_the_pull_request(self):
        # Static peers keep the old message byte for byte; a prefill that
        # predates the field rejects a unified decode's request.
        d_static, _ = _raiden_manager(_static_swa_pool(self.POOL_CLS), prefill=False)
        static_pull = _send_metadata(d_static, pages=[1, 2], swa_pages=[3])
        old = _PullReqBeforeSwaLayout.from_frames(static_pull)
        self.assertEqual(old.to_frames(), static_pull)
        d_mgr, _ = _raiden_manager(self.d_pool, prefill=False)
        unified_pull = _send_metadata(d_mgr, pages=[1, 2], swa_pages=[3])
        with self.assertRaisesRegex(ValueError, "swa_layout"):
            _PullReqBeforeSwaLayout.from_frames(unified_pull)

    def test_the_pipeline_transfer_refuses_the_unified_pool(self):
        # Its per-layer pools would receive every group at group 0's pages.
        mgr = object.__new__(RaidenKVManager)
        mgr._pipeline_enabled = True
        with self.assertRaisesRegex(RuntimeError, "pipeline transfer"):
            mgr.set_kv_caches(get_raiden_kv_caches(self.d_pool))


class TestUnifiedPoolPDTransferRankFour(TestUnifiedPoolPDTransfer):
    """The same transfers over a platform pool class with its own rank-4 layout,
    as an out-of-tree plugin may register: raiden copies each tensor's dim-0 rows."""

    POOL_CLS = _RankFourPool


class TestUnifiedPoolSizing(CustomTestCase):
    """The unified pool gets the bytes of the static full and sliding-window
    pools for the same flags; at 512 bytes per token and layer a full token
    costs (#full + ratio * #SWA) * 512 bytes there, a shared token one row per
    buffer, max(#full, ceil(#SWA / groups)) * 512, and the shared size is whole
    8-page units."""

    def _configurator(self, kvc, **server_args):
        """The unified configurator over ``kvc``'s static policy at ratio 0.5."""
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(
            get_context().override_server_args(
                **{
                    "page_size": kvc.page_size,
                    "swa_full_tokens_ratio": 0.5,
                    "max_total_tokens": None,
                    "chunked_prefill_size": None,
                    **server_args,
                }
            )
        )
        stack.enter_context(get_parallel().override(attn_tp_size=1))
        stack.enter_context(_fused_rpa_platform())
        return TpuUnifiedSWAPoolConfigurator(kvc)

    @staticmethod
    def _sizes(config):
        return (
            config.max_total_num_tokens,
            config.full_max_total_num_tokens,
            config.swa_max_total_num_tokens,
        )

    def test_ratio_pool_shares_the_bytes_of_both_static_pools(self):
        # 1536 bytes per full token size 1000 full + 500 SWA tokens; at 1024
        # bytes per shared token that is 1500, floored to 1472.
        configurator = self._configurator(_hybrid_kvc())
        config = configurator.calculate_pool_sizes(1536 * 1000, PAGE_SIZE)
        self.assertEqual(self._sizes(config), (1000, 1472, 1472))

    def test_unequal_layer_counts_share_by_the_larger_count(self):
        # Three full and one SWA layer: 1792 bytes per full token size 1000 full
        # + 500 SWA tokens; three buffers per shared token hold
        # (3 * 1000 + 500) // 3 = 1166, floored to 1152.
        configurator = self._configurator(
            _hybrid_kvc(full_layer_ids=(0, 1, 2), swa_layer_ids=(3,))
        )
        config = configurator.calculate_pool_sizes(1792 * 1000, PAGE_SIZE)
        self.assertEqual(self._sizes(config), (1000, 1152, 1152))

    def test_grouped_sliding_window_layers_share_by_the_full_count(self):
        # One full and three SWA layers: 1280 bytes per full token size 1000
        # full + 500 SWA tokens. The SWA layers form three groups over the one
        # buffer, so a shared token costs 512 bytes: 2500, floored to 2496.
        configurator = self._configurator(
            _hybrid_kvc(full_layer_ids=(0,), swa_layer_ids=(1, 2, 3))
        )
        config = configurator.calculate_pool_sizes(1280 * 1000, PAGE_SIZE)
        self.assertEqual(self._sizes(config), (1000, 2496, 2496))

    def test_token_cap_keeps_counting_full_attention_tokens(self):
        # gpt-oss at page 64 and ratio 0.8: --max-total-tokens 16400 caps the
        # static pools at 16384 full + 13056 SWA tokens, 29440 shared, floored
        # to 29184. Re-deriving from the reported count must not move it, or
        # the cap path of config_from_budget would shrink the pool again.
        kvc = _hybrid_kvc(page_size=64)
        configurator = self._configurator(kvc, swa_full_tokens_ratio=0.8)
        capped = configurator.calculate_pool_sizes_from_max_tokens(16400, 64)
        self.assertEqual(self._sizes(capped), (16384, 29184, 29184))
        again = configurator.calculate_pool_sizes_from_max_tokens(
            capped.max_total_num_tokens, 64
        )
        self.assertEqual(again, capped)
        # Startup rounds the TPU SWA pool down to whole 8-page units; the pool
        # build asserts that the full and SWA sizes still match after that.
        kvc.device = "tpu"
        kvc.loc_space_scale = 1
        kvc.pool_page_size = 64
        sizes = KVCacheConfigurator._derive_pool_sizes(kvc, config=capped)
        self.assertEqual(sizes.full_max_total_num_tokens, 29184)
        self.assertEqual(sizes.swa_max_total_num_tokens, 29184)

    def test_chunk_cap_pool_shares_the_capped_sliding_window_pool(self):
        # Radix cache off with a request cap sizes the SWA pool from the cap:
        # 2 requests x (window 16 + eviction interval 4 + 2 pages) plus one
        # 32-token chunk and a page = 92 tokens, and the rest of the budget
        # (1000 tokens) goes to the full pool. 1000 + 92 floors to 1088.
        kvc = _hybrid_kvc(sliding_window_size=16)
        with envs.SGLANG_SWA_EVICTION_INTERVAL.override(4):
            configurator = self._configurator(
                kvc,
                max_running_requests=2,
                disable_radix_cache=True,
                chunked_prefill_size=32,
                disable_overlap_schedule=True,
            )
        config = configurator.calculate_pool_sizes(92 * 1024 + 1024 * 1000, PAGE_SIZE)
        self.assertEqual(self._sizes(config), (1000, 1088, 1088))

    def test_only_the_shared_pool_floor_limits_a_small_pool(self):
        # A request holds a window plus a page on each side, so the shared pool
        # needs more than 2 * (64 + 4) = 136 tokens. At ratio 0.25 a 128-token
        # cap shares 128 + 32 = 160 tokens, though the static pools' 32-token
        # SWA pool is below its own 64 + 4 floor; 120 shares 120 + 28 = 148,
        # floored to 128.
        kvc = _hybrid_kvc(sliding_window_size=64)
        configurator = self._configurator(kvc, swa_full_tokens_ratio=0.25)
        with self.assertRaisesRegex(ValueError, "SWA pool"):
            HybridSWAPoolConfigurator(kvc).calculate_pool_sizes_from_max_tokens(
                128, PAGE_SIZE
            )
        config = configurator.calculate_pool_sizes_from_max_tokens(128, PAGE_SIZE)
        self.assertEqual(self._sizes(config), (128, 160, 160))
        with self.assertRaisesRegex(ValueError, "Unified KV pool"):
            configurator.calculate_pool_sizes_from_max_tokens(120, PAGE_SIZE)


class TestUnifiedPoolDispatch(CustomTestCase):
    STATIC_SERVER_ARGS = dict(
        page_size=PAGE_SIZE,
        swa_full_tokens_ratio=0.5,
        max_total_tokens=None,
        chunked_prefill_size=None,
    )

    def test_configurator_follows_the_tpu_field(self):
        hybrid_model = _hybrid_kvc()
        with (
            _fused_rpa_platform(),
            patch(
                "sglang.srt.model_executor.pool_configurator.current_platform",
                TpuSRTPlatform(),
            ),
        ):
            with (
                get_context().override_server_args(**self.STATIC_SERVER_ARGS),
                get_parallel().override(attn_tp_size=1),
            ):
                static = create_memory_pool_configurator(hybrid_model)
            with (
                get_context().override_server_args(
                    tpu_unified_kv_pool=True, **self.STATIC_SERVER_ARGS
                ),
                get_parallel().override(attn_tp_size=1),
            ):
                unified = create_memory_pool_configurator(hybrid_model)
                # Validated before any model-specific branch, so the flag is
                # never silently ignored.
                with self.assertRaisesRegex(
                    ValueError, "without sliding-window layers"
                ):
                    create_memory_pool_configurator(
                        SimpleNamespace(is_hybrid_swa=False)
                    )
        self.assertIs(type(static), HybridSWAPoolConfigurator)
        self.assertIsInstance(unified, TpuUnifiedSWAPoolConfigurator)

    def test_pool_and_allocator_arms_build_the_shared_pair(self):
        # Shared buffers behind two independent free lists would hand the same
        # rows to a full-attention and a sliding-window layer.
        _tpu_swa_allocator_branches(self)
        kvc = _hybrid_kvc()
        kvc.device = "cpu"
        kvc.pool_page_size = PAGE_SIZE
        kvc._tpu_unified_kv_pool_enabled = lambda: (
            KVCacheConfigurator._tpu_unified_kv_pool_enabled(kvc)
        )
        kvc._tpu_mha_pool_class = lambda: _RankFourPool
        # A unified config reports fewer full-attention tokens than it shares;
        # the allocator must take the shared size.
        sizes = SimpleNamespace(
            max_total_num_tokens=POOL_TOKENS // 2,
            full_max_total_num_tokens=POOL_TOKENS,
            swa_max_total_num_tokens=POOL_TOKENS,
        )
        with (
            _fused_rpa_platform(_RankFourPool),
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.current_platform",
                TpuSRTPlatform(),
            ),
            get_context().override_server_args(
                tpu_unified_kv_pool=True, page_size=PAGE_SIZE
            ),
            get_parallel().override(attn_tp_size=1, attn_dcp_size=1),
        ):
            pool = KVCacheConfigurator._build_tpu_swa_kv_pool(
                kvc,
                full_max_total_num_tokens=POOL_TOKENS,
                swa_max_total_num_tokens=POOL_TOKENS,
            )
            allocator = KVCacheConfigurator._build_token_to_kv_pool_allocator(
                kvc,
                sizes=sizes,
                token_to_kv_pool=pool,
                is_dsv4_model=False,
                req_to_token_pool=None,
                token_to_kv_pool_allocator=None,
            )
        self.assertIsInstance(pool, TpuUnifiedSWAKVPool)
        # The shared buffers take the layout of the platform's pool class.
        self.assertIsInstance(pool.full_kv_pool.backing, _RankFourPool)
        self.assertIsInstance(allocator, TpuUnifiedSWATokenToKVPoolAllocator)
        # Either side alone can take every page of the shared pool.
        self.assertEqual(allocator.full_available_size(), pool.size)


if __name__ == "__main__":
    unittest.main()
