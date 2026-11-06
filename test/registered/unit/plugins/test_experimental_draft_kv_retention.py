"""A windowed block draft writes only the extend rows whose SWA slot outlives the extend.

Skipping a row whose slot the radix tree (or the request's decode) keeps silently
drops acceptance on every later hit, so the selection must cover the live set.
"""

from __future__ import annotations

import contextlib
import random
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import checkpoint_kv_cache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.experimental.speculative import block_spec_worker
from sglang_tpu.experimental.speculative.block_spec_worker import TpuBlockSpecWorker
from sglang_tpu.experimental.speculative.draft_kv_retention import (
    CachePublishing,
    DraftKvRetention,
    ExtendRequest,
    extend_request,
    retained_extend_rows,
    retained_position_ranges,
    retention_for_cache,
)

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

# Muse DFlash2: 2048-token window on 256-token pages.
SERVED = DraftKvRetention(
    publishing=CachePublishing.UNIFIED_RADIX, window=2048, page_size=256
)


def _ranges(retention: DraftKvRetention, **req) -> list[tuple[int, int]]:
    return retained_position_ranges(ExtendRequest(**req), retention)


def _with(**changes) -> DraftKvRetention:
    fields = {
        "publishing": SERVED.publishing,
        "window": SERVED.window,
        "page_size": SERVED.page_size,
        "chunk_insert_stride": 16384,
    }
    fields.update(changes)
    return DraftKvRetention(**fields)


class TestRetainedRanges(unittest.TestCase):
    def test_chunk_shorter_than_window_keeps_every_row(self):
        self.assertEqual(
            _ranges(_with(), prefix_len=0, extend_len=1000, prefill_end=1000),
            [(0, 1000)],
        )

    def test_long_prompt_writes_only_stride_and_prompt_end_windows(self):
        retention, end = _with(), 65536
        kept = {
            start: _ranges(
                retention, prefix_len=start, extend_len=4096, prefill_end=end
            )
            for start in range(0, end, 4096)
        }
        # 16384 - 2048 - 256: free_swa_out_of_window_slots keeps max(window, page)
        # below the page-floored pre_len.
        self.assertEqual(kept[12288], [(14080, 16384)])
        self.assertEqual(kept[61440], [(63232, 65536)])
        self.assertEqual(kept[16384], [])
        self.assertEqual(kept[8192], [])
        written = sum(hi - lo for ranges in kept.values() for lo, hi in ranges)
        self.assertEqual(written, 4 * 2304)

    def test_prompt_end_window_spans_two_chunks(self):
        retention = _with(publishing=CachePublishing.PREFILL_END)
        end = 8192 + 1000
        self.assertEqual(
            _ranges(retention, prefix_len=4096, extend_len=4096, prefill_end=end),
            [(6912, 8192)],
        )
        self.assertEqual(
            _ranges(retention, prefix_len=8192, extend_len=1000, prefill_end=end),
            [(8192, 9192)],
        )

    def test_stride_multiple_inside_a_chunk_publishes_the_chunk_end(self):
        self.assertEqual(
            _ranges(_with(), prefix_len=14080, extend_len=4096, prefill_end=65536),
            [(15872, 18176)],
        )

    def test_short_chunks_before_a_multiple_keep_the_window_reaching_back(self):
        # 1024-token chunks: the chunk publishing 16384 needs rows of earlier chunks.
        retention = _with()
        self.assertEqual(
            _ranges(retention, prefix_len=13312, extend_len=1024, prefill_end=65536),
            [(14080, 14336)],
        )
        self.assertEqual(
            _ranges(retention, prefix_len=12288, extend_len=1024, prefill_end=65536),
            [],
        )

    def test_swa_branch_window_is_kept(self):
        retention = _with()
        # A branch inside the chunk publishes it and then the chunk end.
        self.assertEqual(
            _ranges(
                retention,
                prefix_len=8192,
                extend_len=4096,
                prefill_end=65536,
                swa_branch=9216,
            ),
            [(8192, 9216), (9984, 12288)],
        )
        # A branch in a later chunk still needs this chunk's tail of its window.
        self.assertEqual(
            _ranges(
                retention,
                prefix_len=8192,
                extend_len=4096,
                prefill_end=65536,
                swa_branch=13000,
            ),
            [(10752, 12288)],
        )

    def test_pd_prefill_publishes_every_chunk(self):
        retention = _with(publishes_every_chunk=True)
        self.assertEqual(
            _ranges(retention, prefix_len=16384, extend_len=4096, prefill_end=65536),
            [(18176, 20480)],
        )

    def test_radix_disabled_keeps_only_the_prefill_end(self):
        retention = _with(publishing=CachePublishing.PREFILL_END)
        self.assertEqual(
            _ranges(retention, prefix_len=12288, extend_len=4096, prefill_end=65536),
            [],
        )
        self.assertEqual(
            _ranges(
                _with(),
                prefix_len=12288,
                extend_len=4096,
                prefill_end=65536,
                publishes=False,
            ),
            [],
        )

    def test_session_and_unknown_caches_keep_every_row(self):
        self.assertEqual(
            _ranges(
                _with(),
                prefix_len=0,
                extend_len=4096,
                prefill_end=65536,
                keeps_every_row=True,
            ),
            [(0, 4096)],
        )
        self.assertEqual(
            _ranges(
                _with(publishing=CachePublishing.EVERY_ROW),
                prefix_len=0,
                extend_len=4096,
                prefill_end=65536,
            ),
            [(0, 4096)],
        )

    def test_rows_of_requests_with_different_prefix_lens(self):
        reqs = [
            # Mid-prompt chunk, no window kept.
            ExtendRequest(prefix_len=4096, extend_len=4096, prefill_end=65536),
            # A radix hit at 512 finishing its prompt at 1536: every row.
            ExtendRequest(prefix_len=512, extend_len=1024, prefill_end=1536),
            # A mixed-chunk decode row.
            ExtendRequest(prefix_len=900, extend_len=1, prefill_end=901),
        ]
        rows = retained_extend_rows(reqs, _with())
        expected = list(range(4096, 4096 + 1024)) + [4096 + 1024]
        self.assertEqual(rows.tolist(), expected)


# Scaled 1/32 like the mem_cache SWA tests: 16-token pages, a 63-token cache
# window (2048 - 1), a 64-token draft window, 128-token chunks and a 512 stride.
PAGE, CACHE_WINDOW, DRAFT_WINDOW = 16, 63, 64


class _World:
    """A real hybrid-SWA UnifiedRadixCache driven like the scheduler drives
    chunked prefills; tracks which SWA slots hold valid draft KV."""

    def __init__(self) -> None:
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=PAGE)
        )
        size = 16384
        self.req_to_token_pool = ReqToTokenPool(
            size=16, max_context_len=4096, device="cpu", enable_memory_saver=False
        )
        kv_pool = SWAKVPool(
            size=size,
            size_swa=size,
            page_size=PAGE,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=8,
            swa_attention_layer_ids=[0],
            full_attention_layer_ids=[1],
            device="cpu",
        )
        self.allocator = SWATokenToKVPoolAllocator(
            size=size,
            size_swa=size,
            page_size=PAGE,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=kv_pool,
            need_sort=False,
        )
        with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python"):
            self.cache = UnifiedRadixCache(
                params=CacheInitParams(
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.allocator,
                    page_size=PAGE,
                    disable=False,
                    sliding_window_size=CACHE_WINDOW,
                    tree_components=(ComponentType.FULL, ComponentType.SWA),
                    eviction_policy="lru",
                )
            )
        self.assigned: set[int] = set()
        self.valid: set[int] = set()
        self.rows_written = 0
        self.rows_total = 0
        self.rid = 0

    def _alloc(self, n: int) -> torch.Tensor:
        aligned = (n + PAGE - 1) // PAGE * PAGE
        full = self.allocator.full_attn_allocator.alloc(aligned)
        swa = self.allocator.swa_attn_allocator.alloc(aligned)
        assert full is not None and swa is not None, "pool exhausted"
        self.allocator.set_full_to_swa_mapping(full, swa)
        # A page tail past the prompt holds no token.
        self.assigned.difference_update(swa.tolist())
        return full[:n]

    def prefill(
        self,
        tokens: list[int],
        *,
        chunk: int,
        retention: DraftKvRetention,
        mid_prefill: bool,
    ) -> Req:
        req = Req(
            rid=f"r{self.rid}",
            origin_input_text="",
            origin_input_ids=array("q", tokens),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1 << 20),
        )
        self.rid += 1
        self.req_to_token_pool.alloc([req])
        req.init_next_round_input(tree_cache=self.cache)
        req.lock_receipt = self.cache.inc_lock_ref(req.last_node).to_dec_params()
        row, pos = req.kv.req_pool_idx, len(req.prefix_indices)
        if pos:
            self.req_to_token_pool.write((row, slice(0, pos)), req.prefix_indices)
        req.kv.kv_committed_len = req.kv.kv_allocated_len = pos
        while pos < len(tokens):
            end = min(pos + chunk, len(tokens))
            req.set_extend_range(pos, end)
            full = self._alloc(end - pos)
            self.req_to_token_pool.write((row, slice(pos, end)), full)
            self._extend(req, full, prefix_len=pos, retention=retention)
            req.kv.kv_committed_len = req.kv.kv_allocated_len = end
            pos = end
            if pos < len(tokens):
                checkpoint_kv_cache(req, self.cache, mid_prefill=mid_prefill)
                req.init_next_round_input()
        req.output_ids.append(1)
        checkpoint_kv_cache(req, self.cache)
        return req

    def _extend(
        self,
        req: Req,
        full: torch.Tensor,
        *,
        prefix_len: int,
        retention: DraftKvRetention,
    ) -> None:
        swa = self.allocator.translate_loc_from_full_to_swa(full).tolist()
        self.assigned.update(swa)
        # A reused slot holds the previous owner's draft KV.
        self.valid.difference_update(swa)
        er = extend_request(req, prefix_len=prefix_len, extend_len=len(swa))
        rows = retained_extend_rows([er], retention).tolist()
        self.valid.update(swa[i] for i in rows)
        self.rows_written += len(rows)
        self.rows_total += len(swa)

    def live_unwritten(self) -> set[int]:
        a = self.allocator.swa_attn_allocator
        free_pages = [a.free_pages, *a.staged_pages]
        if a.release_pages is not None:
            free_pages.append(a.release_pages)
        pages = torch.cat(free_pages)
        free = set(
            (pages.view(-1, 1) * PAGE + torch.arange(PAGE).view(1, -1))
            .view(-1)
            .tolist()
        )
        return self.assigned - free - self.valid


def _tokens(seed: int, n: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(1, 50000) for _ in range(n)]


class TestRetainedSlotsInRadixWorld(CustomTestCase):
    """Every SWA slot the cache or a request still holds after the prefills
    carries draft KV, while most prompt rows are never projected."""

    def _run(
        self,
        *,
        stride: int,
        chunk: int,
        mid_prefill: bool,
        disagg_mode: str = "null",
        publishing: CachePublishing | None = None,
    ) -> _World:
        world = _World()
        a = _tokens(0, 1500)
        prompts = [
            a,
            a[:1100]
            + _tokens(1, 300),  # hits A's stride checkpoint, SWA branch past it
            a + _tokens(2, 200),  # hits A's prompt end
            _tokens(3, 40),
        ]
        with envs.SGLANG_OPT_UNIFIED_CACHE_CHUNK_INSERT_STRIDE.override(stride):
            retention = retention_for_cache(
                world.cache, draft_window=DRAFT_WINDOW, disagg_mode=disagg_mode
            )
            if publishing is not None:
                retention = DraftKvRetention(
                    publishing=publishing,
                    window=retention.window,
                    page_size=PAGE,
                )
            for tokens in prompts:
                world.prefill(
                    tokens, chunk=chunk, retention=retention, mid_prefill=mid_prefill
                )
        return world

    def test_mixed_with_stride(self):
        world = self._run(stride=512, chunk=128, mid_prefill=True)
        self.assertEqual(world.live_unwritten(), set())
        self.assertLess(world.rows_written, world.rows_total // 2)

    def test_mixed_with_stride_and_chunks_shorter_than_the_window(self):
        world = self._run(stride=512, chunk=48, mid_prefill=True)
        self.assertEqual(world.live_unwritten(), set())
        self.assertLess(world.rows_written, world.rows_total // 2)

    def test_mixed_publishing_every_chunk(self):
        world = self._run(stride=0, chunk=128, mid_prefill=True)
        self.assertEqual(world.live_unwritten(), set())
        self.assertLess(world.rows_written, world.rows_total)

    def test_pd_prefill_publishes_every_chunk(self):
        world = self._run(
            stride=512, chunk=128, mid_prefill=False, disagg_mode="prefill"
        )
        self.assertEqual(world.live_unwritten(), set())

    def test_prefill_end_only_misses_published_windows(self):
        # Guards the check itself: ignoring radix checkpoints leaves live slots unwritten.
        world = self._run(
            stride=512,
            chunk=128,
            mid_prefill=True,
            publishing=CachePublishing.PREFILL_END,
        )
        self.assertNotEqual(world.live_unwritten(), set())


class TestRetentionForCache(unittest.TestCase):
    def test_cache_kinds(self):
        world = _World()
        retention = retention_for_cache(
            world.cache, draft_window=DRAFT_WINDOW, disagg_mode="null"
        )
        self.assertIs(retention.publishing, CachePublishing.UNIFIED_RADIX)
        self.assertEqual(retention.window, DRAFT_WINDOW)
        world.cache.disable = True
        retention = retention_for_cache(
            world.cache, draft_window=DRAFT_WINDOW, disagg_mode="null"
        )
        self.assertIs(retention.publishing, CachePublishing.PREFILL_END)
        world.cache.disable = False
        with envs.SGLANG_OPT_SWA_KEEP_PREFILL_CHUNK_WINDOWS.override(True):
            retention = retention_for_cache(
                world.cache, draft_window=DRAFT_WINDOW, disagg_mode="null"
            )
        self.assertIs(retention.publishing, CachePublishing.EVERY_ROW)


class TestExtendDraftKvWrite(CustomTestCase):
    """The worker gathers the retained rows to a smaller writer bucket, and skips
    the writer when none are retained."""

    BUCKETS = [64, 128, 256]

    def _worker(self) -> tuple[TpuBlockSpecWorker, list]:
        writes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        def write_draft_kv(*, target_hidden, positions, cache_loc):
            self.assertIn(target_hidden.shape[0], self.BUCKETS)
            writes.append((target_hidden, positions, cache_loc))
            return torch.ones(1)

        worker = object.__new__(TpuBlockSpecWorker)
        worker.device = "cpu"
        worker._block_draft_worker = SimpleNamespace(
            # Full slot i maps to SWA slot i + 1000.
            full_to_swa=torch.arange(4096, dtype=torch.int64) + 1000,
            draft_window=64,
            extend_row_bucket=lambda n: min(b for b in self.BUCKETS if b >= n),
            gather_extend_rows=lambda hidden, index: torch.index_select(
                hidden, 0, index
            ),
            draft_context=contextlib.nullcontext,
            write_draft_kv=write_draft_kv,
        )
        return worker, writes

    def _batch(self, *, prefix_len: int, extend_len: int, prefill_end: int):
        world = _World()
        req = SimpleNamespace(
            full_untruncated_fill_ids=array("q", range(prefill_end)),
            swa_branching_seqlen=None,
            skip_radix_cache_insert=False,
            session_id=None,
        )
        return SimpleNamespace(
            reqs=[req],
            prefix_lens=[prefix_len],
            extend_lens=[extend_len],
            out_cache_loc=torch.arange(1, extend_len + 1, dtype=torch.int64),
            tree_cache=world.cache,
        )

    def _write(self, batch, *, rows: int, stride: int = 0):
        worker, writes = self._worker()
        hidden = torch.arange(rows, dtype=torch.float32).view(-1, 1).expand(-1, 4)
        tokens = torch.tensor([7])
        with (
            patch.object(
                block_spec_worker,
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_mode="null"),
            ),
            envs.SGLANG_OPT_UNIFIED_CACHE_CHUNK_INSERT_STRIDE.override(stride),
        ):
            tail = worker._write_extend_draft_kv(
                batch, target_hidden=hidden, target_tokens=tokens
            )
        return tail, tokens, writes

    def test_no_retained_rows_skips_the_writer(self):
        # A mid-prompt chunk the stride defers, far from the prompt end.
        batch = self._batch(prefix_len=0, extend_len=256, prefill_end=2048)
        tail, tokens, writes = self._write(batch, rows=256, stride=1024)
        self.assertEqual(writes, [])
        self.assertIs(tail, tokens)

    def test_retained_rows_are_gathered_to_a_smaller_bucket(self):
        # Chunk end 256 is published: rows [256 - 1 - 64 floored to 16, 256) = [176, 256).
        batch = self._batch(prefix_len=0, extend_len=256, prefill_end=2048)
        tail, _, writes = self._write(batch, rows=256)
        self.assertEqual(len(writes), 1)
        hidden, positions, cache_loc = writes[0]
        self.assertEqual(hidden.shape[0], 128)
        kept = list(range(176, 256))
        self.assertEqual(positions[:80].tolist(), kept)
        self.assertEqual(hidden[:80, 0].tolist(), [float(r) for r in kept])
        # out_cache_loc of row r is r + 1, mapped to SWA slot r + 1001.
        self.assertEqual(cache_loc[:80].tolist(), [r + 1001 for r in kept])
        self.assertEqual(cache_loc[80:].tolist(), [0] * 48)
        self.assertEqual(tail.tolist(), [1.0])

    def test_prompt_end_chunk_writes_every_row_in_its_bucket(self):
        batch = self._batch(prefix_len=0, extend_len=100, prefill_end=100)
        _, _, writes = self._write(batch, rows=128)
        hidden, positions, cache_loc = writes[0]
        self.assertEqual(hidden.shape[0], 128)
        self.assertEqual(positions[:100].tolist(), list(range(100)))


if __name__ == "__main__":
    unittest.main()
