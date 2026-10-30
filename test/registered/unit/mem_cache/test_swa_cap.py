from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import random
import unittest
from array import array
from types import SimpleNamespace

import torch
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import checkpoint_kv_cache, release_kv_cache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

# The served 2047-token window on 512-token pages, scaled 1/32: a window spans
# four 16-token pages and reads [seqlen - 64, seqlen).
PAGE, WINDOW = 16, 63
SWA = ComponentType.SWA
POISON = 30000.0  # large finite poison (a NaN would hide in comparisons)


def _value(token: int, pos: int) -> float:
    """The KV a cold prefill writes for ``token`` at ``pos`` (exact in BF16)."""
    return float((token * 31 + pos * 7) % 251 + 1)


class DecodeWorld:
    """A real UnifiedRadixCache driven like the scheduler drives a running
    request: admission match + lock, chunked prefill with checkpoint_kv_cache,
    then decode steps (one KV slot per token, no tree insert) with CAP's
    cap_release, and release_kv_cache at the finish. Every slot gets the cold
    value of its token, so a reused slot can be checked against a recompute."""

    def __init__(self, *, full_tokens: int, swa_tokens: int):
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=PAGE)
        )
        self.req_to_token_pool = ReqToTokenPool(
            size=32, max_context_len=8192, device="cpu", enable_memory_saver=False
        )
        self.kv_pool = SWAKVPool(
            size=full_tokens,
            size_swa=swa_tokens,
            page_size=PAGE,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=8,
            swa_attention_layer_ids=[0, 2],
            full_attention_layer_ids=[1],
            device="cpu",
        )
        self.allocator = SWATokenToKVPoolAllocator(
            size=full_tokens,
            size_swa=swa_tokens,
            page_size=PAGE,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=self.kv_pool,
            need_sort=False,
        )
        self.cache = UnifiedRadixCache(
            params=CacheInitParams(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.allocator,
                page_size=PAGE,
                disable=False,
                sliding_window_size=WINDOW,
                tree_components=(ComponentType.FULL, ComponentType.SWA),
                eviction_policy="lru",
            )
        )
        self.rid = 0

    # -- KV values
    def _write(
        self, full: torch.Tensor, tokens: list[int], positions: list[int]
    ) -> None:
        vals = torch.tensor(
            [_value(t, p) for t, p in zip(tokens, positions)], dtype=torch.float32
        ).to(torch.bfloat16)
        swa = self.allocator.translate_loc_from_full_to_swa(full)
        for buf in self.kv_pool.full_kv_pool.k_buffer:
            buf[full] = vals.view(-1, 1, 1).expand(-1, buf.shape[1], buf.shape[2])
        for buf in self.kv_pool.swa_kv_pool.k_buffer:
            buf[swa] = vals.view(-1, 1, 1).expand(-1, buf.shape[1], buf.shape[2])

    def poison_free_swa(self) -> None:
        """Poison every free SWA slot (a released page must never be read)."""
        a = self.allocator.swa_attn_allocator
        pages = (
            a.free_pages
            if a.release_pages is None or len(a.release_pages) == 0
            else torch.cat((a.free_pages, a.release_pages))
        )
        slots = (pages.view(-1, 1) * PAGE + torch.arange(PAGE).view(1, -1)).view(-1)
        for buf in self.kv_pool.swa_kv_pool.k_buffer:
            buf[slots] = POISON

    def swa_values(self, full: torch.Tensor) -> list[float]:
        swa = self.allocator.translate_loc_from_full_to_swa(full)
        return self.kv_pool.swa_kv_pool.k_buffer[0][swa, 0, 0].float().tolist()

    def full_values(self, full: torch.Tensor) -> list[float]:
        return self.kv_pool.full_kv_pool.k_buffer[0][full, 0, 0].float().tolist()

    def _alloc(self, tokens: list[int], start: int) -> torch.Tensor:
        n = len(tokens)
        aligned = (n + PAGE - 1) // PAGE * PAGE
        self.allocator.evict_to_free_tokens(self.cache, aligned)
        a = self.allocator
        full = a.full_attn_allocator.alloc(aligned)
        swa = a.swa_attn_allocator.alloc(aligned)
        assert full is not None and swa is not None, "pool exhausted"
        a.set_full_to_swa_mapping(full, swa)
        self._write(full[:n], tokens, list(range(start, start + n)))
        return full[:n]

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

    def _next_token(self, req: Req) -> int:
        return 1 + (len(req.origin_input_ids) * 13 + len(req.output_ids) * 17) % 40000

    def decode(self, req: Req, steps: int, cap_every: int = 0) -> None:
        """``steps`` decode steps (a new page at each page boundary, else the
        next slot of the last page); with ``cap_every`` the scheduler's SWA work
        runs at that interval: the out-of-window eviction, then cap_release."""
        row = req.kv.req_pool_idx
        for _ in range(steps):
            pos = req.seqlen - 1  # the KV of the last token
            tok = req.output_ids[-1]
            if pos % PAGE == 0:
                full = self._alloc([tok], pos)
            else:
                prev = self.req_to_token_pool.req_to_token[row, pos - 1].to(torch.int64)
                full = (prev + 1).view(1)
                self._write(full, [tok], [pos])
            self.req_to_token_pool.write((row, slice(pos, pos + 1)), full)
            req.kv.kv_committed_len = req.kv.kv_allocated_len = pos + 1
            req.output_ids.append(self._next_token(req))
            req.decode_batch_idx = getattr(req, "decode_batch_idx", 0) + 1
            if cap_every and req.decode_batch_idx % cap_every == 0:
                self.cache.evict_sliding_windows(req, req.seqlen - 1, eviction_interval=1)
                self.cache.cap_release(req)

    def finish(self, req: Req) -> None:
        req.finished_reason = FINISH_LENGTH(length=0)
        release_kv_cache(req, self.cache)

    def evict(self, swa_tokens: int, full_tokens: int = 0) -> None:
        self.cache.evict(EvictParams(num_tokens=full_tokens, swa_num_tokens=swa_tokens))

    def match(self, tokens: list[int]):
        return self.cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", tokens), None, limit=len(tokens) - 1)
            )
        )

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
        with envs.SGLANG_OPT_SWA_CAP.override(
            cap
        ), envs.SGLANG_OPT_SWA_CAP_MEDIA_ANCHORS.override(anchors):
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
                float(torch.tensor(_value(t, lo + i)).to(torch.bfloat16))
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
                float(torch.tensor(_value(t, p)).to(torch.bfloat16))
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



class TestSWACapText(unittest.TestCase):
    """SGLANG_OPT_SWA_CAP_TEXT (CAP v2): what CAP does to a request without
    media. Its reuse is an exact repeat of the whole prompt, which needs the
    window ending at the page-floored prompt end minus one token."""

    PROMPT = 12 * PAGE + 5  # 197 tokens: the repeat's match ends at 192
    CHUNK = 64

    def _world(self, text, *, cap=True, anchors=0, swa_tokens=4096):
        with envs.SGLANG_OPT_SWA_CAP.override(
            cap
        ), envs.SGLANG_OPT_SWA_CAP_MEDIA_ANCHORS.override(
            anchors
        ), envs.SGLANG_OPT_SWA_CAP_TEXT.override(
            text
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
        """"skip": every lock range and node state of a text request decoding
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
        """"keep_prompt_end" shrinks the lock as CAP does, and spares exactly
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
                float(torch.tensor(_value(t, p)).to(torch.bfloat16))
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
                    float(torch.tensor(_value(t, lo + i)).to(torch.bfloat16))
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
