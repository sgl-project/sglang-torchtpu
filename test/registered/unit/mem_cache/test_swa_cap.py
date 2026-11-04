from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import contextlib
import os
import random
import unittest
from array import array
from types import SimpleNamespace
from unittest import mock

import torch
from swa_cache_world import (
    PAGE,
    POISON,
    SWA,
    WINDOW,
    KVWorld,
    cpu_ci_tree_core_selection,
    kv_value,
)

from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeRequest
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.allocator import paged
from sglang.srt.mem_cache.common import checkpoint_kv_cache, release_kv_cache
from sglang.srt.mem_cache.unified_cache import tree_core_registry
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams


class DecodeWorld(KVWorld):
    """A real UnifiedRadixCache driven like the scheduler drives a running
    request: admission match + lock, chunked prefill with checkpoint_kv_cache,
    then decode steps (one KV slot per token, no tree insert) with CAP's
    cap_release, and release_kv_cache at the finish."""

    # -- request lifecycle
    def admit(self, tokens: list[int], chunk: int, media=None) -> Req:
        req = Req(
            rid=f"r{self.rid}",
            origin_input_text="",
            origin_input_ids=array("q", tokens),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1 << 20),
        )
        if media is not None:
            req.multimodal_inputs = SimpleNamespace(
                mm_items=[SimpleNamespace(offsets=[(media[0], media[1] - 1)])]
            )
        self.rid += 1
        self.req_to_token_pool.alloc([req])
        req.init_next_round_input(tree_cache=self.cache)
        req.lock_receipt = self.cache.inc_lock_ref(req.last_node).to_dec_params()
        cached = len(req.prefix_indices)
        row = req.kv.req_pool_idx
        if cached:
            self.req_to_token_pool.write((row, slice(0, cached)), req.prefix_indices)
        req.kv.kv_committed_len = req.kv.kv_allocated_len = cached
        pos = cached
        while pos < len(tokens):
            end = min(pos + chunk, len(tokens))
            req.set_extend_range(pos, end)
            self.req_to_token_pool.write(
                (row, slice(pos, end)), self._alloc(tokens[pos:end], pos)
            )
            req.kv.kv_committed_len = req.kv.kv_allocated_len = end
            pos = end
            if pos < len(tokens):
                checkpoint_kv_cache(req, self.cache)
                req.init_next_round_input()
        req.output_ids.append(self._next_token(req))
        checkpoint_kv_cache(req, self.cache)
        req.cached = cached
        return req

    def decode(self, req: Req, steps: int, cap_every: int = 0) -> None:
        """``steps`` decode steps; with ``cap_every`` the scheduler's SWA work
        runs at that interval: the out-of-window eviction, then cap_release."""
        for _ in range(steps):
            self.decode_step(req)
            if cap_every and req.decode_batch_idx % cap_every == 0:
                self.cache.evict_sliding_windows(
                    req, req.seqlen - 1, eviction_interval=1
                )
                self.cache.cap_release(req)

    def finish(self, req: Req) -> None:
        req.finished_reason = FINISH_LENGTH(length=0)
        release_kv_cache(req, self.cache)

    # -- inspection
    def locked_range(self, req: Req) -> tuple[int, int] | None:
        """[start, end) token positions of the request's SWA segment lock."""
        if req.swa_prefix_lock_released or req.last_node is None:
            return None
        swa = self.cache.components[SWA]
        uuid = req.lock_receipt.get_lock_uuid(SWA)
        path = swa._path_with_positions(self.cache.tree_core.node_by_id(req.last_node))
        top = path[-1][1]
        for n, start, _ in path:
            if uuid is not None and n.component_data[SWA].metadata.get("uuid") == uuid:
                top = start
                break
        return top, path[0][2]

    def node_states(self, req_or_node) -> list[tuple[int, int, int, bool]]:
        """(start, end, swa lock_ref, swa live) along a root path."""
        swa = self.cache.components[SWA]
        node = req_or_node
        if isinstance(req_or_node, Req):
            node = self.cache.tree_core.node_by_id(req_or_node.last_node)
        return [
            (
                s,
                e,
                n.component_data[SWA].lock_ref,
                n.component_data[SWA].value is not None,
            )
            for n, s, e in swa._path_with_positions(node)
        ]


def _tokens(seed: int, n: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(1, 50000) for _ in range(n)]


class TestSWACap(unittest.TestCase):
    PROMPT = 12 * PAGE  # 192 tokens
    # 2048-token chunks against a 2047-token window, as served: every chunk
    # insert leaves the whole chunk with live SWA in the tree.
    CHUNK = 64

    def _world(self, *, cap=True, anchors=0, swa_tokens=4096, full_tokens=8192):
        with (
            envs.SGLANG_OPT_SWA_CAP.override(cap),
            envs.SGLANG_OPT_SWA_CAP_MEDIA_ANCHORS.override(anchors),
        ):
            return DecodeWorld(full_tokens=full_tokens, swa_tokens=swa_tokens)

    def test_lock_shrinks_page_by_page_oldest_first(self):
        w = self._world()
        req = w.admit(_tokens(1, self.PROMPT), self.CHUNK)
        top0, end = w.locked_range(req)
        self.assertEqual(end, self.PROMPT)
        self.assertLessEqual(top0, self.PROMPT - (WINDOW + 1))
        history = [(top0, end)]
        protected0 = w.cache.tree_core.component_protected_size_[SWA]
        for _ in range(8):
            w.decode(req, 8)
            w.cache.cap_release(req)
            cut = (req.seqlen - 1 - WINDOW) // PAGE * PAGE
            rng = w.locked_range(req)
            if cut >= self.PROMPT:
                self.assertIsNone(
                    rng, "the window left the prompt: whole SWA lock released"
                )
                history.append(None)
                continue
            top, e = rng
            self.assertEqual(e, self.PROMPT)
            self.assertEqual(
                top, max(top0, cut), "lock starts at the page-floored window start"
            )
            # Every node below the cut is released, every node from it on is held.
            for s, en, ref, _ in w.node_states(req):
                if en <= top:
                    break
                if s >= top:
                    self.assertGreaterEqual(ref, 1)
            history.append((top, e))
        tops = [h[0] for h in history if h is not None]
        self.assertEqual(
            tops, sorted(tops), "pages are released oldest first, never re-locked"
        )
        self.assertIsNone(history[-1])
        self.assertLess(w.cache.tree_core.component_protected_size_[SWA], protected0)
        w.finish(req)
        w.cache.sanity_check()

    def test_finish_after_a_partial_shrink_releases_the_rest(self):
        """The receipt carries the moved boundary: the finish unpin releases
        exactly the shrunk segment, and no SWA lock is left behind."""
        w = self._world()
        req = w.admit(_tokens(13, self.PROMPT), self.CHUNK)
        top0 = w.locked_range(req)[0]
        w.decode(req, 40, cap_every=32)
        rng = w.locked_range(req)
        self.assertIsNotNone(rng)
        self.assertGreater(rng[0], top0, "the lock shrank")
        w.finish(req)
        w.cache.sanity_check()
        self.assertEqual(w.cache.tree_core.component_protected_size_[SWA], 0)

    def test_pages_another_request_holds_stay_locked(self):
        w = self._world()
        base = _tokens(2, self.PROMPT)
        a = w.admit(base, self.CHUNK)
        b = w.admit(base[: self.PROMPT - 2 * PAGE] + _tokens(3, 2 * PAGE), self.CHUNK)
        b_states = {(s, e): ref for s, e, ref, _ in w.node_states(b)}
        w.decode(a, 3 * WINDOW, cap_every=32)  # A's window leaves its prompt
        self.assertTrue(a.swa_prefix_lock_released)
        b_rng = w.locked_range(b)
        for s, e, ref, live in w.node_states(b):
            if e <= b_rng[0]:
                break
            self.assertGreaterEqual(ref, 1, f"B's window node [{s},{e}) lost its lock")
            self.assertTrue(live)
        w.evict(swa_tokens=10**6)
        for s, e, ref, live in w.node_states(b):
            if e <= b_rng[0]:
                break
            self.assertTrue(live, f"B's locked window node [{s},{e}) was evicted")
        del b_states
        w.finish(a)
        w.finish(b)
        w.cache.sanity_check()

    def test_released_pages_are_evictable_and_never_read(self):
        w = self._world(swa_tokens=1024)
        req = w.admit(_tokens(4, self.PROMPT), self.CHUNK)
        others = [w.admit(_tokens(10 + i, self.PROMPT), self.CHUNK) for i in range(2)]
        for o in others:
            w.finish(o)
        w.decode(req, 40, cap_every=32)
        w.evict(swa_tokens=10**6)
        w.poison_free_swa()
        # The running request's window still reads its own values, no poison.
        row = req.kv.req_pool_idx
        lo = req.seqlen - 1 - WINDOW
        full = w.req_to_token_pool.req_to_token[row, lo : req.seqlen - 1].to(
            torch.int64
        )
        got = w.swa_values(full)
        toks = (list(req.origin_input_ids) + list(req.output_ids))[lo : req.seqlen - 1]
        self.assertEqual(
            got,
            [
                float(torch.tensor(kv_value(t, lo + i)).to(torch.bfloat16))
                for i, t in enumerate(toks)
            ],
        )
        self.assertNotIn(POISON, got)
        w.finish(req)
        w.cache.sanity_check()

    def test_warm_match_equals_cold_values(self):
        w = self._world(swa_tokens=1536, anchors=2)
        sys_prompt = _tokens(5, 6 * PAGE)
        media = _tokens(6, 4 * PAGE)
        reqs = []
        for i in range(6):
            toks = sys_prompt + media + _tokens(20 + i, 2 * PAGE + i)
            r = w.admit(
                toks, self.CHUNK, media=(len(sys_prompt), len(sys_prompt) + len(media))
            )
            w.decode(r, 70, cap_every=32)
            w.finish(r)
            reqs.append(toks)
            if i % 2:
                w.evict(swa_tokens=256)
        w.poison_free_swa()
        for toks in reqs:
            m = w.match(toks)
            n = len(m.device_indices)
            if n == 0:
                continue
            full = m.device_indices.to(torch.int64)
            want = [
                float(torch.tensor(kv_value(t, p)).to(torch.bfloat16))
                for p, t in enumerate(toks[:n])
            ]
            self.assertEqual(
                w.full_values(full), want, "a warm FULL hit equals the cold prefill"
            )
            lo = max(0, n - (WINDOW + 1))
            self.assertEqual(
                w.swa_values(full[lo:]),
                want[lo:],
                "a warm SWA window equals the cold prefill",
            )
        w.cache.sanity_check()

    def test_media_anchor_stays_resident(self):
        sys_prompt = _tokens(7, 6 * PAGE)
        media = _tokens(8, 4 * PAGE)  # the media ends at 160 > window
        boundary = len(sys_prompt) + len(media)
        results = {}
        for anchors in (0, 1):
            w = self._world(anchors=anchors, swa_tokens=1024)
            r = w.admit(
                sys_prompt + media + _tokens(9, 3 * PAGE),
                self.CHUNK,
                media=(len(sys_prompt), boundary),
            )
            w.decode(r, 70, cap_every=32)
            w.finish(r)
            for i in range(3):  # pressure from unrelated requests
                o = w.admit(_tokens(100 + i, self.PROMPT), self.CHUNK)
                w.finish(o)
            w.evict(swa_tokens=10**6)
            m = w.match(sys_prompt + media + _tokens(99, 2 * PAGE))
            results[anchors] = len(m.device_indices)
            if anchors:
                self.assertEqual(len(w.cache._cap_anchors), 1)
        self.assertGreaterEqual(
            results[1], boundary, "the anchored media window keeps its match"
        )
        self.assertLess(
            results[0], boundary, "control: without the anchor the window is evicted"
        )

    def test_anchor_budget_releases_least_recently_used(self):
        w = self._world(anchors=1, swa_tokens=2048)
        sys_prompt = _tokens(11, 6 * PAGE)
        firsts = []
        for i in range(2):
            media = _tokens(30 + i, 4 * PAGE)
            r = w.admit(
                sys_prompt + media + _tokens(40 + i, 2 * PAGE),
                self.CHUNK,
                media=(96, 160),
            )
            firsts.append(
                w.cache.tree_core.node_by_id(next(reversed(w.cache._cap_anchors)))
            )
            w.finish(r)
        self.assertEqual(len(w.cache._cap_anchors), 1)
        self.assertEqual(
            firsts[0].component_data[SWA].lock_ref, 0, "the older anchor was released"
        )
        self.assertGreaterEqual(firsts[1].component_data[SWA].lock_ref, 1)

    def test_cap_off_changes_nothing(self):
        w = self._world(cap=False)
        req = w.admit(_tokens(12, self.PROMPT), self.CHUNK)
        before = w.locked_range(req)
        w.decode(req, 40)
        w.cache.cap_release(req)
        self.assertEqual(w.locked_range(req), before)


class _Admitted(Exception):
    """Stops a prefill pass at batch creation, after its admission loop."""


class _NaiveAllocExtend:
    """CPU stand-in for the Triton alloc_extend kernel of the paged
    allocator: alloc_extend_naive, its reference implementation."""

    def __getitem__(self, grid):
        def run(prefix_lens, seq_lens, last_loc, free_pages, out, _bs, page_size):
            paged.alloc_extend_naive(
                prefix_lens, seq_lens, last_loc, free_pages, out, page_size, out.device
            )

        return run


class TestSWACapAnchorReclaim(unittest.TestCase):
    """CAP media anchors are cache-owned locks (a FULL root path plus an SWA
    window), so memory pressure reclaims them. The case from review, scaled: a
    finished media request's anchor holds 59% of the FULL pool, nothing runs,
    and an unrelated 512-token prompt (50% of the pool) waits."""

    CHUNK = 64
    BOUNDARY = 38 * PAGE  # the media end: 608 of 1024 FULL tokens

    def _pressured_world(self) -> DecodeWorld:
        with (
            envs.SGLANG_OPT_SWA_CAP.override(True),
            envs.SGLANG_OPT_SWA_CAP_MEDIA_ANCHORS.override(1),
        ):
            w = DecodeWorld(full_tokens=1024, swa_tokens=1024)
        r = w.admit(
            _tokens(1, self.BOUNDARY + 2 * PAGE),
            self.CHUNK,
            media=(64, self.BOUNDARY),
        )
        w.decode(r, 70, cap_every=32)
        w.finish(r)
        w.evict(swa_tokens=10**6, full_tokens=10**6)  # all that is evictable
        self.assertEqual(len(w.cache._cap_anchors), 1)
        self.assertEqual(w.cache.full_protected_size(), self.BOUNDARY)
        return w

    def _swa_pressured_world(self) -> DecodeWorld:
        """4 anchors hold 384 of 528 SWA tokens (96 each), nothing runs, and
        everything else is evicted."""
        with (
            envs.SGLANG_OPT_SWA_CAP.override(True),
            envs.SGLANG_OPT_SWA_CAP_MEDIA_ANCHORS.override(4),
        ):
            w = DecodeWorld(full_tokens=8192, swa_tokens=528)
        for i in range(4):
            r = w.admit(_tokens(50 + i, 12 * PAGE), self.CHUNK, media=(96, 160))
            w.finish(r)
        w.evict(swa_tokens=10**6, full_tokens=10**6)  # all that is evictable
        self.assertEqual(len(w.cache._cap_anchors), 4)
        self.assertEqual(w.cache.swa_protected_size(), 384)
        return w

    def _parked_chunk_world(self) -> tuple[DecodeWorld, Req]:
        """The parked-chunk case from review, scaled: 4 anchors hold 384 of
        528 SWA tokens. A 320-token prompt is admitted with them present (its
        first chunk needs 143 of the 144 SWA tokens left) and runs that chunk.
        Then 80 remain, less than the window plus two pages a continuation
        keeps free, so add_chunked_req parks its next chunk."""
        w = self._swa_pressured_world()
        req = self._req(_tokens(4, 20 * PAGE))
        req.init_next_round_input(tree_cache=w.cache)
        adder = self._adder(w, rem_chunk_tokens=self.CHUNK)
        adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)
        self.assertIs(adder.new_chunked_req, req, "admitted, in chunks")
        # Its first chunk, as DecodeWorld.admit runs one.
        w.req_to_token_pool.alloc([req])
        req.lock_receipt = w.cache.inc_lock_ref(req.last_node).to_dec_params()
        w.req_to_token_pool.write(
            (req.kv.req_pool_idx, slice(0, self.CHUNK)),
            w._alloc(list(req.origin_input_ids[: self.CHUNK]), 0),
        )
        req.kv.kv_committed_len = req.kv.kv_allocated_len = self.CHUNK
        checkpoint_kv_cache(req, w.cache)
        self.assertEqual(
            w.allocator.swa_available_size() + w.cache.swa_evictable_size(), 80
        )
        return w, req

    @staticmethod
    def _anchors_flag():
        """SGLANG_OPT_SWA_CAP_MEDIA_ANCHORS while the scheduler code runs;
        patch.dict restores the env even when the body raises."""
        return mock.patch.dict(os.environ, {"SGLANG_OPT_SWA_CAP_MEDIA_ANCHORS": "1"})

    def _req(self, tokens: list[int]) -> Req:
        return Req(
            rid=f"q{tokens[0]}",
            origin_input_text="",
            origin_input_ids=array("q", tokens),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=16),
        )

    def _big(self) -> Req:
        return self._req(_tokens(2, 32 * PAGE))

    def _adder(self, w: DecodeWorld, rem_chunk_tokens=None) -> PrefillAdder:
        """A real PrefillAdder, nothing running."""
        return PrefillAdder(
            page_size=PAGE,
            tree_cache=w.cache,
            token_to_kv_pool_allocator=w.allocator,
            running_batch=None,
            new_token_ratio=1.0,
            rem_input_tokens=8192,
            rem_chunk_tokens=rem_chunk_tokens,
        )

    def _admission(self, w: DecodeWorld, req: Req) -> AddReqResult:
        """One admission check by a real PrefillAdder, nothing running."""
        req.init_next_round_input(tree_cache=w.cache)
        return self._adder(w).add_one_req(
            req, has_chunked_req=False, truncation_align_size=None
        )

    def _prefill_pass(self, w: DecodeWorld, waiting: list[Req], chunked_req=None):
        """One real Scheduler prefill pass (its admission loop with a real
        PrefillAdder) over ``waiting``, nothing running. With ``chunked_req``
        the pass first continues that request in CHUNK-token chunks."""
        s = Scheduler.__new__(Scheduler)
        s.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        s.enable_priority_preemption = False
        s.is_hybrid_swa = True
        s.waiting_queue = list(waiting)
        s.chunked_req = chunked_req
        s.min_free_slots_delayer = None
        s.get_num_allocatable_reqs = lambda *args, **kwargs: 8
        s.policy = SimpleNamespace(
            calc_priority=lambda *args, **kwargs: None,
            shortest_prefill_chunk_limit=lambda *args, **kwargs: None,
        )
        s.processed_tokens_counter = None
        s.chunked_prefill_size = None if chunked_req is None else self.CHUNK
        s.dynamic_chunk_sizer = None
        s.tp_worker = SimpleNamespace(model_runner=SimpleNamespace(attn_backend=None))
        s._tpu_pp_chunk_cost = None
        s.page_size = PAGE
        s.tree_cache = w.cache
        s.token_to_kv_pool_allocator = w.allocator
        s.req_to_token_pool = w.req_to_token_pool
        s.new_token_ratio_tracker = SimpleNamespace(current=1.0)
        s.max_prefill_tokens = 8192
        s.is_mixed_chunk = False
        s.priority_scheduling_preemption_threshold = 0
        s.max_prefill_bs = 0
        s.max_running_requests = 8
        s.dllm_config = None
        s.enable_lora = False
        s.disaggregation_mode = DisaggregationMode.NULL
        s.enable_hicache_storage = False
        s.enable_lmcache = False
        s.enable_hierarchical_cache = False
        s.enable_unified_cache_external_linker = False
        s.truncation_align_size = None
        s._dynamic_first_chunk_cap = lambda req: None
        s.model_config = s.spec_algorithm = None
        s.enable_overlap = False
        return Scheduler._get_new_batch_prefill_raw(
            s,
            prefill_delayer_single_pass=None,
            running_batch=ScheduleBatch(reqs=[], batch_is_full=False),
        )

    def test_release_cap_anchor_least_recently_used_first(self):
        with (
            envs.SGLANG_OPT_SWA_CAP.override(True),
            envs.SGLANG_OPT_SWA_CAP_MEDIA_ANCHORS.override(2),
        ):
            w = DecodeWorld(full_tokens=8192, swa_tokens=2048)
        sys_prompt = _tokens(11, 6 * PAGE)
        anchors = []
        for i in range(2):
            r = w.admit(
                sys_prompt + _tokens(30 + i, 4 * PAGE) + _tokens(40 + i, 2 * PAGE),
                self.CHUNK,
                media=(96, 160),
            )
            anchors.append(
                w.cache.tree_core.node_by_id(next(reversed(w.cache._cap_anchors)))
            )
            w.finish(r)
        self.assertEqual(len(w.cache._cap_anchors), 2)
        self.assertTrue(w.cache.release_cap_anchor())
        self.assertEqual(anchors[0].component_data[SWA].lock_ref, 0)
        self.assertGreaterEqual(anchors[1].component_data[SWA].lock_ref, 1)
        self.assertTrue(w.cache.release_cap_anchor())
        self.assertFalse(w.cache.release_cap_anchor(), "no anchor left")
        self.assertEqual(w.cache.full_protected_size(), 0)
        self.assertEqual(w.cache.swa_protected_size(), 0)
        w.cache.sanity_check()

    def test_the_anchor_alone_blocks_admission(self):
        """Control: the anchored path is protected, so admission never counts
        it and eviction never frees it; the prompt is refused on every check."""
        w = self._pressured_world()
        big = self._big()
        for _ in range(3):
            self.assertEqual(self._admission(w, big), AddReqResult.NO_TOKEN)
        self.assertEqual(len(w.cache._cap_anchors), 1)

    def test_blocked_prefill_pass_releases_the_anchor(self):
        """A pass that admits nothing for lack of memory releases the anchor,
        and the next admission check takes the prompt."""
        w = self._pressured_world()
        big = self._big()
        with self._anchors_flag():
            batch, _ = self._prefill_pass(w, [big])
        self.assertIsNone(batch)
        self.assertEqual(w.cache._cap_anchors, {})
        self.assertEqual(w.cache.full_protected_size(), 0)
        self.assertEqual(self._admission(w, big), AddReqResult.CONTINUE)

    def test_flag_unset_blocked_pass_is_as_on_main(self):
        """With SGLANG_OPT_SWA_CAP_MEDIA_ANCHORS unset a blocked pass never
        calls the release."""
        w = self._pressured_world()
        with mock.patch.object(w.cache, "release_cap_anchor") as release:
            batch, _ = self._prefill_pass(w, [self._big()])
        self.assertIsNone(batch)
        release.assert_not_called()
        self.assertEqual(len(w.cache._cap_anchors), 1)

    def test_pass_that_admits_keeps_the_anchor(self):
        """A pass that admits a request makes progress: the NO_TOKEN of the
        prompt behind it leaves the anchor alone."""
        w = self._pressured_world()
        small = self._req(_tokens(3, 8 * PAGE))
        with (
            self._anchors_flag(),
            mock.patch.object(ScheduleBatch, "init_new", side_effect=_Admitted),
            self.assertRaises(_Admitted),
        ):
            self._prefill_pass(w, [small, self._big()])
        self.assertEqual(len(w.cache._cap_anchors), 1)

    def test_parked_chunk_pass_releases_an_anchor(self):
        """A pass with a parked chunk and an empty waiting queue admits
        nothing for lack of memory too: it releases one anchor, and the pass
        after it takes the chunk. With the flag unset the pass is as on main."""
        w, req = self._parked_chunk_world()
        batch, _ = self._prefill_pass(w, [], chunked_req=req)
        self.assertIsNone(batch, "the next chunk is parked")
        self.assertEqual(len(w.cache._cap_anchors), 4, "flag unset: as on main")
        with self._anchors_flag():
            batch, _ = self._prefill_pass(w, [], chunked_req=req)
        self.assertIsNone(batch)
        self.assertEqual(len(w.cache._cap_anchors), 3, "one anchor released")
        with (
            self._anchors_flag(),
            mock.patch.object(
                ScheduleBatch, "init_new", side_effect=_Admitted
            ) as init_new,
            self.assertRaises(_Admitted),
        ):
            self._prefill_pass(w, [], chunked_req=req)
        self.assertEqual(init_new.call_args.args[0], [req], "the chunk is taken")
        self.assertEqual(len(w.cache._cap_anchors), 3)

    def test_decode_step_that_does_not_fit_releases_the_anchor(self):
        """ScheduleBatch.check_decode_mem releases the anchor before the retract
        loop would take a running request; a step that fits keeps it."""
        w = self._pressured_world()

        def check(num_tokens: int) -> bool:
            batch = SimpleNamespace(
                new_tokens_required_next_decode=lambda selected: num_tokens,
                reqs=[],
                token_to_kv_pool_allocator=w.allocator,
                tree_cache=w.cache,
                spec_algorithm=None,
            )
            return ScheduleBatch.check_decode_mem(batch)

        free = w.allocator.full_available_size()
        with self._anchors_flag():
            self.assertTrue(check(free))
        self.assertEqual(len(w.cache._cap_anchors), 1, "a step that fits keeps it")
        self.assertFalse(check(free + 6 * PAGE), "flag unset: as on main")
        self.assertEqual(len(w.cache._cap_anchors), 1)
        with self._anchors_flag():
            self.assertTrue(check(free + 6 * PAGE))
        self.assertEqual(w.cache._cap_anchors, {})
        w.cache.sanity_check()

    def _pd_decode_queue(self, w: DecodeWorld) -> DecodePreallocQueue:
        """A real DecodePreallocQueue (PD decode with the decode radix cache)
        on the world's cache and pools, nothing running. Only the scheduler
        fields and the transfer plumbing are faked; sending the KV destination
        to prefill is recorded instead."""
        q = DecodePreallocQueue.__new__(DecodePreallocQueue)
        q.pp_size = 1
        q.queue, q.pending_reqs, q.retracted_queue = [], [], []
        q.num_reserved_decode_tokens = PAGE
        q.req_to_token_pool = w.req_to_token_pool
        q.token_to_kv_pool = w.kv_pool
        q.token_to_kv_pool_allocator = w.allocator
        q.tree_cache = w.cache
        q.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
            available_size=lambda: 8
        )
        q.transfer_queue = SimpleNamespace(queue=[])
        q.kv_manager = SimpleNamespace(kv_args=SimpleNamespace(state_types=[]))
        q.scheduler = SimpleNamespace(
            running_batch=SimpleNamespace(reqs=[]),
            waiting_queue=[],
            last_batch=None,
            enable_priority_scheduling=False,
            enable_hisparse=False,
            enable_lora=False,
            enable_decode_hicache=False,
            sliding_window_size=WINDOW,
            tp_worker=SimpleNamespace(is_hybrid_swa=True),
        )
        q._resolve_pending_reqs = lambda: None
        q._update_handshake_waiters = lambda *args: None
        q._send_kv_metadata = mock.Mock()
        return q

    @staticmethod
    @contextlib.contextmanager
    def _pd_decode_config():
        """The decode radix cache on, and the paged allocator's Triton
        alloc_extend kernel replaced by its CPU reference."""
        with (
            get_context().override_server_args(
                page_size=PAGE, disaggregation_decode_enable_radix_cache=True
            ),
            mock.patch.object(paged, "alloc_extend_kernel", _NaiveAllocExtend()),
        ):
            yield

    def _pd_decode_pass(self, q: DecodePreallocQueue, reqs: list[Req]) -> list[str]:
        """One real pop_preallocated pass over ``reqs`` (all waiting for their
        KV destination); the rids it preallocated."""
        q.queue = [
            DecodeRequest(req=r, kv_receiver=None, waiting_for_input=True) for r in reqs
        ]
        with self._pd_decode_config():
            preallocated, failed = q.pop_preallocated()
        self.assertEqual(failed, [])
        return [d.req.rid for d in preallocated]

    def test_idle_pd_decode_prealloc_releases_the_anchor(self):
        """The case from review on a PD decode server: with nothing running,
        the anchor alone keeps DecodePreallocQueue.pop_preallocated from
        reserving the prompt (with the flag unset, as on main, on every pass).
        With the flag set, the first pass releases the anchor, refreshes its
        budget and preallocates the prompt."""
        w = self._pressured_world()
        q = self._pd_decode_queue(w)
        big = self._big()
        for _ in range(3):
            self.assertEqual(self._pd_decode_pass(q, [big]), [], "flag unset")
        self.assertEqual(len(w.cache._cap_anchors), 1)
        q._send_kv_metadata.assert_not_called()
        with self._anchors_flag():
            self.assertEqual(self._pd_decode_pass(q, [big]), [big.rid])
        self.assertEqual(w.cache._cap_anchors, {})
        q._send_kv_metadata.assert_called_once()
        self.assertEqual(len(q._send_kv_metadata.call_args.args[1]), 32 * PAGE)

    def test_pd_decode_pass_that_preallocates_keeps_the_anchor(self):
        """As in prefill admission, a pass that preallocates a request makes
        progress and leaves the anchor alone; the next pass, with nothing
        preallocated yet, releases it for the waiting prompt."""
        w = self._pressured_world()
        q = self._pd_decode_queue(w)
        small, big = self._req(_tokens(3, 8 * PAGE)), self._big()
        with self._anchors_flag():
            self.assertEqual(self._pd_decode_pass(q, [small, big]), [small.rid])
        self.assertEqual(len(w.cache._cap_anchors), 1)
        q.transfer_queue.queue = [SimpleNamespace(req=small, host_staged=False)]
        with self._anchors_flag():
            self.assertEqual(self._pd_decode_pass(q, [big]), [big.rid])
        self.assertEqual(w.cache._cap_anchors, {})

    def test_pd_decode_prealloc_releases_only_the_anchors_it_needs(self):
        """The SWA side: a 512-token prompt with 128 new tokens reserves its
        64-token SWA tail plus 128, 192 of the 144 SWA tokens left. Releasing
        the least recently used anchor (96 tokens) is enough, so the pass
        releases that one only and preallocates the prompt."""
        w = self._swa_pressured_world()
        q = self._pd_decode_queue(w)
        anchors = list(w.cache._cap_anchors)
        req = self._big()
        req.sampling_params.max_new_tokens = 128
        self.assertEqual(self._pd_decode_pass(q, [req]), [], "flag unset")
        self.assertEqual(list(w.cache._cap_anchors), anchors)
        with self._anchors_flag():
            self.assertEqual(self._pd_decode_pass(q, [req]), [req.rid])
        self.assertEqual(list(w.cache._cap_anchors), anchors[1:])

    def test_idle_pd_decode_resume_releases_the_anchor(self):
        """DecodePreallocQueue.resume_retracted_reqs has the same admission
        check: a retracted request is resumed once the anchor is released."""
        w = self._pressured_world()
        q = self._pd_decode_queue(w)
        req = self._big()
        req.output_ids.append(7)
        req.is_retracted = True
        q.retracted_queue = [req]
        with (
            self._pd_decode_config(),
            mock.patch("sglang.srt.disaggregation.decode.restore_kv_cache") as restore,
        ):
            self.assertEqual(q.resume_retracted_reqs(), [], "flag unset")
            self.assertEqual(len(w.cache._cap_anchors), 1)
            with self._anchors_flag():
                resumed = q.resume_retracted_reqs()
        self.assertEqual([r.rid for r in resumed], [req.rid])
        self.assertEqual(w.cache._cap_anchors, {})
        self.assertEqual(q.retracted_queue, [])
        self.assertIs(restore.call_args.args[0], req)
        self.assertFalse(req.is_retracted)


class TestSWACapTreeCoreFallback(unittest.TestCase):
    """SGLANG_OPT_SWA_CAP is a Rust-fallback reason (tree_core_registry): with
    it set, an SWA cache gets the Python TreeCore even where CPU CI picks Rust,
    and CAP works there; with it unset, the cache gets Rust, as on main."""

    PROMPT = TestSWACap.PROMPT
    CHUNK = TestSWACap.CHUNK

    def test_cap_set_selects_the_python_tree_core(self):
        rust = mock.MagicMock(side_effect=AssertionError("Rust TreeCore selected"))
        with (
            cpu_ci_tree_core_selection(rust),
            envs.SGLANG_OPT_SWA_CAP.override(True),
            self.assertLogs(tree_core_registry.logger, "INFO") as logs,
        ):
            w = DecodeWorld(full_tokens=8192, swa_tokens=4096, pin_python=False)
        rust.assert_not_called()
        self.assertEqual(
            [r.getMessage() for r in logs.records],
            [
                "Using the Python TreeCore: "
                "SGLANG_OPT_SWA_CAP requires the Python TreeCore"
            ],
        )
        self.assertEqual(w.cache._tree_core_backend, "python")
        self.assertIsInstance(w.cache.tree_core, UnifiedTreeCore)
        req = w.admit(_tokens(13, self.PROMPT), self.CHUNK)
        top0 = w.locked_range(req)[0]
        w.decode(req, 40, cap_every=32)
        self.assertGreater(w.locked_range(req)[0], top0, "the lock shrank")
        w.finish(req)
        w.cache.sanity_check()
        self.assertEqual(w.cache.tree_core.component_protected_size_[SWA], 0)

    def test_cap_unset_keeps_the_rust_tree_core(self):
        rust = mock.MagicMock()
        with cpu_ci_tree_core_selection(rust):
            w = DecodeWorld(full_tokens=8192, swa_tokens=4096, pin_python=False)
        rust.assert_called_once_with(mock.ANY, w.cache.components)
        self.assertEqual(w.cache._tree_core_backend, "rust")
        self.assertIs(w.cache.tree_core, rust.return_value)
        self.assertFalse(w.cache.components[SWA].cap_enabled)


class TestSWACapText(unittest.TestCase):
    """SGLANG_OPT_SWA_CAP_TEXT (CAP v2): what CAP does to a request without
    media. Its reuse is an exact repeat of the whole prompt, which needs the
    window ending at the page-floored prompt end minus one token."""

    PROMPT = 12 * PAGE + 5  # 197 tokens: the repeat's match ends at 192
    CHUNK = 64

    def _world(self, text, *, cap=True, anchors=0, swa_tokens=4096):
        with (
            envs.SGLANG_OPT_SWA_CAP.override(cap),
            envs.SGLANG_OPT_SWA_CAP_MEDIA_ANCHORS.override(anchors),
            envs.SGLANG_OPT_SWA_CAP_TEXT.override(text),
        ):
            return DecodeWorld(full_tokens=8192, swa_tokens=swa_tokens)

    def _trace(self, w, req, steps=4, every=8):
        """The request's lock range and root-path node states after each burst
        of decode steps with CAP's scheduler work."""
        out = []
        for _ in range(steps):
            w.decode(req, every, cap_every=every)
            out.append((w.locked_range(req), w.node_states(req)))
        return out

    def test_skip_leaves_a_text_request_as_without_cap(self):
        """ "skip": every lock range and node state of a text request decoding
        across its window equals the CAP-off run; "release" shrinks it."""
        traces = {}
        for name, text, cap in (
            ("off", "release", False),
            ("skip", "skip", True),
            ("release", "release", True),
        ):
            w = self._world(text, cap=cap)
            req = w.admit(_tokens(50, self.PROMPT), self.CHUNK)
            traces[name] = self._trace(w, req, steps=12)
            w.finish(req)
            w.cache.sanity_check()
        self.assertEqual(traces["skip"], traces["off"])
        self.assertNotEqual(traces["release"], traces["off"])

    def test_requests_with_media_keep_the_cap_behaviour(self):
        """Both new modes leave a request that carries media exactly as CAP
        does today: the same lock shrink and the same node states."""
        sys_prompt = _tokens(51, 6 * PAGE)
        media = _tokens(52, 4 * PAGE)
        toks = sys_prompt + media + _tokens(53, 2 * PAGE + 3)
        traces = {}
        for text in ("release", "skip", "keep_prompt_end"):
            w = self._world(text, anchors=1)
            req = w.admit(
                toks, self.CHUNK, media=(len(sys_prompt), len(sys_prompt) + len(media))
            )
            traces[text] = self._trace(w, req, steps=12)
            w.finish(req)
            w.cache.sanity_check()
        self.assertEqual(traces["skip"], traces["release"])
        self.assertEqual(traces["keep_prompt_end"], traces["release"])
        self.assertGreater(traces["release"][2][0][0], traces["release"][0][0][0])

    def test_keep_prompt_end_releases_the_lock_but_spares_the_window(self):
        """ "keep_prompt_end" shrinks the lock as CAP does, and spares exactly
        the window ending at the page-floored prompt end minus one token."""
        w = self._world("keep_prompt_end")
        req = w.admit(_tokens(54, self.PROMPT), self.CHUNK)
        skip, exempt = w.cache.components[SWA].cap_text_exempt(req)
        boundary = (self.PROMPT - 1) // PAGE * PAGE
        self.assertFalse(skip)
        self.assertEqual(exempt, (boundary - WINDOW, boundary))
        top0 = w.locked_range(req)[0]
        w.decode(req, 40, cap_every=32)
        self.assertGreater(w.locked_range(req)[0], top0, "the lock shrank")
        w.finish(req)
        w.cache.sanity_check()

    def _repeat_match(self, text):
        """Older finished traffic, then request A decodes with CAP past part of
        its window, then just enough eviction for one window: the length an
        exact repeat of A's prompt matches afterwards."""
        w = self._world(text, swa_tokens=1024)
        for i in range(3):
            o = w.admit(_tokens(60 + i, self.PROMPT), self.CHUNK)
            w.finish(o)
        prompt = _tokens(70, self.PROMPT)
        a = w.admit(prompt, self.CHUNK)
        w.decode(a, 40, cap_every=32)
        w.evict(swa_tokens=WINDOW + 1)
        w.poison_free_swa()
        m = w.match(prompt)
        n = len(m.device_indices)
        if n:
            full = m.device_indices.to(torch.int64)
            want = [
                float(torch.tensor(kv_value(t, p)).to(torch.bfloat16))
                for p, t in enumerate(prompt[:n])
            ]
            self.assertEqual(w.full_values(full), want)
            lo = max(0, n - (WINDOW + 1))
            self.assertEqual(w.swa_values(full[lo:]), want[lo:])
            self.assertNotIn(POISON, w.swa_values(full[lo:]))
        w.finish(a)
        w.cache.sanity_check()
        return n

    def test_an_exact_repeat_keeps_its_match(self):
        """CAP "release" demotes A's prompt-end window to the eviction end, so
        one window of eviction takes it and the repeat matches less; "skip"
        and "keep_prompt_end" keep the whole page-floored prompt match, with
        warm values equal to the cold prefill and no poison read."""
        boundary = (self.PROMPT - 1) // PAGE * PAGE
        got = {t: self._repeat_match(t) for t in ("release", "skip", "keep_prompt_end")}
        self.assertEqual(got["skip"], boundary)
        self.assertEqual(got["keep_prompt_end"], boundary)
        self.assertLess(got["release"], boundary)

    def test_released_text_pages_are_never_read(self):
        """Large-finite poison in every free SWA slot after full eviction: a
        text request decoding under either new mode reads only its own window
        values, and warm matches equal the cold prefill."""
        for text in ("skip", "keep_prompt_end"):
            w = self._world(text, swa_tokens=1024)
            req = w.admit(_tokens(80, self.PROMPT), self.CHUNK)
            for i in range(2):
                o = w.admit(_tokens(81 + i, self.PROMPT), self.CHUNK)
                w.finish(o)
            w.decode(req, 3 * WINDOW, cap_every=32)
            w.evict(swa_tokens=10**6)
            w.poison_free_swa()
            row = req.kv.req_pool_idx
            lo = req.seqlen - 1 - WINDOW
            full = w.req_to_token_pool.req_to_token[row, lo : req.seqlen - 1].to(
                torch.int64
            )
            toks = (list(req.origin_input_ids) + list(req.output_ids))[
                lo : req.seqlen - 1
            ]
            self.assertEqual(
                w.swa_values(full),
                [
                    float(torch.tensor(kv_value(t, lo + i)).to(torch.bfloat16))
                    for i, t in enumerate(toks)
                ],
            )
            w.finish(req)
            w.cache.sanity_check()

    def test_unknown_mode_is_rejected(self):
        from sglang.srt.mem_cache.unified_cache.components import swa as swa_module

        for mode in ("release", "skip", "keep_prompt_end"):
            with envs.SGLANG_OPT_SWA_CAP_TEXT.override(mode):
                self.assertEqual(swa_module._cap_text_mode(), mode)
        with envs.SGLANG_OPT_SWA_CAP_TEXT.override("demote"):
            with self.assertRaises(ValueError):
                swa_module._cap_text_mode()

    def test_default_is_the_cap_behaviour(self):
        self.assertEqual(envs.SGLANG_OPT_SWA_CAP_TEXT.get(), "release")
        w = self._world("release")
        self.assertEqual(
            w.cache.components[SWA].cap_text_exempt(
                w.admit(_tokens(90, self.PROMPT), self.CHUNK)
            ),
            (False, None),
        )


if __name__ == "__main__":
    unittest.main()
