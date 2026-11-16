from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import random
import unittest
from array import array
from types import SimpleNamespace
from unittest import mock

import torch
from swa_cache_world import cpu_ci_tree_core_selection

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import checkpoint_kv_cache, release_kv_cache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache import tree_core_registry
from sglang.srt.mem_cache.unified_cache.components.base import (
    ComponentType,
    EvictLayer,
)
from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

# A 2047-token window on 512-token pages, scaled 1/32: a window spans 4 pages.
PAGE, WINDOW = 16, 63


class PrefillWorld:
    """A real UnifiedRadixCache driven through the PD-prefill tree-cache calls:
    admission match + lock, per-chunk alloc and checkpoint_kv_cache, the final
    checkpoint_kv_cache, then release_kv_cache once the transfer is done."""

    def __init__(self, *, full_tokens: int, swa_tokens: int):
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=PAGE)
        )
        self.req_to_token_pool = ReqToTokenPool(
            size=16, max_context_len=8192, device="cpu", enable_memory_saver=False
        )
        kv_pool = SWAKVPool(
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
            kvcache=kv_pool,
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

    def _alloc(self, n: int) -> torch.Tensor:
        aligned = (n + PAGE - 1) // PAGE * PAGE
        self.allocator.evict_to_free_tokens(self.cache, aligned)
        a = self.allocator
        full = a.full_attn_allocator.alloc(aligned)
        swa = a.swa_attn_allocator.alloc(aligned)
        assert full is not None and swa is not None, "pool exhausted"
        a.set_full_to_swa_mapping(full, swa)
        return full[:n]

    def prefill(
        self,
        tokens: list[int],
        chunk: int,
        media: tuple[int, int] | None = None,
        *,
        decode: int = 0,
        output: int = 7,
    ) -> int:
        """Returns the cached prefix length. ``decode`` output tokens (the first is
        ``output``) get KV before the request finishes, so they enter the tree.
        ``media`` is the [start, end) token span of one multimodal item."""
        req = Req(
            rid=f"r{self.rid}",
            origin_input_text="",
            origin_input_ids=array("q", tokens),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        if media is not None:
            # Only the item token offsets (inclusive) matter to the cache.
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
            self.req_to_token_pool.write((row, slice(pos, end)), self._alloc(end - pos))
            req.kv.kv_committed_len = req.kv.kv_allocated_len = end
            pos = end
            if pos < len(tokens):
                checkpoint_kv_cache(req, self.cache)
                req.init_next_round_input()
        req.output_ids.append(output)
        checkpoint_kv_cache(req, self.cache)
        if decode:
            n = len(tokens)
            # The first slots continue the prompt's last page.
            rem = min(-n % PAGE, decode)
            parts = []
            if rem:
                last = self.req_to_token_pool.req_to_token[row, n - 1 : n]
                parts.append(last + torch.arange(1, rem + 1, dtype=last.dtype))
            if decode > rem:
                parts.append(self._alloc(decode - rem).to(torch.int32))
            self.req_to_token_pool.write((row, slice(n, n + decode)), torch.cat(parts))
            req.output_ids.extend([output + 1] * (decode - 1))
            req.kv.kv_committed_len = req.kv.kv_allocated_len = n + decode
        req.finished_reason = FINISH_LENGTH(length=0)
        release_kv_cache(req, self.cache)
        return cached

    def match(self, tokens: list[int]):
        return self.cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", tokens), None, limit=len(tokens) - 1)
            )
        )


def _tokens(seed: int, n: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(1, 50000) for _ in range(n)]


class TestSWAEvictStaleWindowsFirst(unittest.TestCase):
    # 96-token chunks, 1.5 windows each: 3072-token chunks against a 2047-token window.
    CHUNK = 96
    PREFIX = 1024 + 1

    def _world(self, *, enabled: bool, swa_tokens: int = 8192) -> PrefillWorld:
        # The switch applies to the Python TreeCore, which TPU always uses.
        with (
            envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python"),
            envs.SGLANG_OPT_SWA_EVICT_STALE_WINDOWS_FIRST.override(enabled),
        ):
            return PrefillWorld(full_tokens=32768, swa_tokens=swa_tokens)

    def _full_hits(self, *, enabled: bool, sessions: int) -> int:
        """Sessions whose revisit hits the full prefix."""
        w = self._world(enabled=enabled, swa_tokens=4096)
        prefixes = [_tokens(100 + s, self.PREFIX) for s in range(sessions)]
        for p in prefixes:
            w.prefill(p, self.CHUNK)
        hits = [w.prefill(p, self.CHUNK) for p in prefixes]
        w.cache.sanity_check()
        return sum(1 for h in hits if h == 1024)

    def test_round_robin_sessions_survive_swa_pressure(self):
        """Multi-turn shape: S prefixes revisited in order; the SWA pool fits every tail but
        not every prefix's stale windows."""
        sessions = 12
        self.assertLess(self._full_hits(enabled=False, sessions=sessions), sessions)
        self.assertEqual(self._full_hits(enabled=True, sessions=sessions), sessions)

    def _conversation_hits(
        self, *, enabled: bool, conversations: int, turns: int
    ) -> int:
        """Revisits that hit the previous turn's whole prompt. Each turn's prompt is the
        previous prompt plus three pages, and its output token completes a page that the
        next turn does not send back, so every turn leaves an abandoned output leaf."""
        w = self._world(enabled=enabled, swa_tokens=1536)
        prompts = [_tokens(200 + c, 8 * PAGE - 1) for c in range(conversations)]
        hits = 0
        for turn in range(turns):
            for c in range(conversations):
                # The previous turn's prompt ends one token before a page boundary.
                previous = len(prompts[c]) - 3 * PAGE
                cached = w.prefill(prompts[c], self.CHUNK, decode=1)
                hits += turn > 0 and cached == previous // PAGE * PAGE
                prompts[c] = prompts[c] + _tokens(1000 * c + turn, 3 * PAGE)
        w.cache.sanity_check()
        return hits

    def test_round_robin_conversations_survive_swa_pressure(self):
        """Multi-turn conversations: the SWA pool fits every conversation's tail window but
        not the windows of its earlier turns, which only an abandoned output leaf separates
        from the current path."""
        conversations, turns = 12, 8
        revisits = conversations * (turns - 1)
        self.assertEqual(
            self._conversation_hits(
                enabled=True, conversations=conversations, turns=turns
            ),
            revisits,
        )
        self.assertLess(
            self._conversation_hits(
                enabled=False, conversations=conversations, turns=turns
            ),
            revisits,
        )
        # The output leaves are what separate the earlier turns from the path.
        with mock.patch.object(
            SWAComponent, "_abandoned_output_chain", return_value=[]
        ):
            self.assertLess(
                self._conversation_hits(
                    enabled=True, conversations=conversations, turns=turns
                ),
                revisits,
            )

    def _evict_swa(self, w: PrefillWorld, num_tokens: int) -> None:
        w.cache.evict(EvictParams(num_tokens=0, swa_num_tokens=num_tokens))

    def test_other_conversation_tail_keeps_its_place(self):
        """A leaf beside the path holding another conversation's prompt is not an
        abandoned output: neither it nor the shared prefix above it is demoted."""
        w = self._world(enabled=True)
        shared = _tokens(1, 4 * PAGE)
        for f in range(6):
            w.prefill(_tokens(900 + f, 6 * PAGE - 1), 10000)
        a = shared + _tokens(2, 4 * PAGE - 1)
        w.prefill(a, 10000)
        b = shared + _tokens(3, 4 * PAGE - 1)
        w.prefill(b, 10000)
        w.prefill(b + _tokens(4, 3 * PAGE), 10000)
        # One older leaf's worth of SWA: the six unrelated leaves go first.
        self._evict_swa(w, 4 * PAGE)
        self.assertEqual(w.prefill(a + _tokens(5, 3 * PAGE), 10000), 7 * PAGE)
        w.cache.sanity_check()

    def test_shared_prefix_beside_a_cold_leaf_keeps_its_place(self):
        """A new conversation still hits a shared prefix whose only other child is an
        old one-turn conversation that never came back."""
        w = self._world(enabled=True)
        shared = _tokens(1, 4 * PAGE)
        for f in range(6):
            w.prefill(_tokens(900 + f, 6 * PAGE - 1), 10000)
        w.prefill(shared + _tokens(7, 4 * PAGE - 1), 10000)
        b = shared + _tokens(2, 4 * PAGE - 1)
        w.prefill(b, 10000)
        w.prefill(b + _tokens(3, 3 * PAGE), 10000)
        self._evict_swa(w, PAGE)
        self.assertEqual(w.prefill(shared + _tokens(4, 4 * PAGE - 1), 10000), 4 * PAGE)
        w.cache.sanity_check()

    def test_short_first_turn_beside_the_path_keeps_its_place(self):
        """Another conversation whose first message after the shared prefix is under a
        page ends in output much like a dropped reply; its prompt differs from the
        path's, so it and the shared prefix keep their place."""
        for evict in (2 * PAGE, 4 * PAGE):
            w = self._world(enabled=True)
            shared = _tokens(1, 4 * PAGE)
            for f in range(6):
                w.prefill(_tokens(900 + f, 6 * PAGE - 1), 10000, decode=1)
            a = shared + _tokens(2, PAGE - 1)
            w.prefill(a, 10000, decode=1)
            b = shared + _tokens(3, 2 * PAGE - 1)
            w.prefill(b, 10000, decode=1)
            w.prefill(b + [7] + _tokens(4, 3 * PAGE), 10000, decode=1)
            self._evict_swa(w, evict)
            # A's next turn sends its reply back.
            self.assertEqual(
                w.prefill(a + [7] + _tokens(5, 3 * PAGE), 10000, decode=1), 5 * PAGE
            )
            self.assertEqual(
                w.prefill(shared + _tokens(6, 4 * PAGE - 1), 10000, decode=1),
                4 * PAGE,
            )
            w.cache.sanity_check()

    def test_long_dropped_reply_is_crossed(self):
        """A dropped reply longer than a window is split into a chain of output nodes;
        the next turn still demotes it and the windows behind it."""
        w = self._world(enabled=True)
        prompt = _tokens(11, 8 * PAGE - 1)
        # 15 prompt tokens and 65 output tokens past the last full page: 5 pages,
        # more than one window, so the SWA leaf cap splits them.
        w.prefill(prompt, self.CHUNK, decode=65)
        reply = w.cache.tree_core.node_by_id(w.match(prompt + [7, 1]).last_device_node)
        self.assertEqual(len(reply.children), 1)
        demoted = []
        lru = w.cache.tree_core.lru_lists[ComponentType.SWA]
        with mock.patch.object(
            lru, "reset_node_lru", side_effect=lambda node, **_: demoted.append(node)
        ):
            w.prefill(prompt + _tokens(12, 6 * PAGE), self.CHUNK, decode=1)
        self.assertIn(reply, demoted)
        self.assertIn(next(iter(reply.children.values())), demoted)
        # The previous turn's window above the branch.
        self.assertIn(reply.parent, demoted)

    def test_write_back_copies_no_stale_window_to_host(self):
        """Under write_back, evicting a node's SWA first copies its window to host. A
        window the switch demoted is never read, and a copy taken before its device
        eviction can't be reclaimed, so none is taken until a match makes it live again."""
        w = self._world(enabled=True)
        core = w.cache.tree_core
        core.is_write_back = True
        core.has_swa_host_pool = True
        swa = w.cache.components[ComponentType.SWA]
        lru = core.lru_lists[ComponentType.SWA]
        prompt = _tokens(11, 16 * PAGE - 1)
        w.prefill(prompt, self.CHUNK, decode=1)
        # Less than a window more: the window above the dropped reply stays live.
        nxt = prompt + _tokens(12, 2 * PAGE)
        with mock.patch.object(
            lru, "reset_node_lru", wraps=lru.reset_node_lru
        ) as demote:
            w.prefill(nxt, self.CHUNK, decode=1)
        stale = [c.args[0] for c in demote.call_args_list if lru.in_list(c.args[0])]
        self.assertGreater(len(stale), 1)
        # Evicting one copies neither it nor the live windows above it.
        for n in stale:
            self.assertEqual(swa._collect_unbacked_swa_nodes(n), [])
        # A later demotion that does not mark, such as the final-window one, keeps it.
        window = next(n for n in stale if n.output_offset is None)
        lru.reset_node_lru(window)
        self.assertEqual(swa._collect_unbacked_swa_nodes(window), [])
        # write_through copies at insert; the switch leaves its backups alone.
        core.is_write_back = False
        for n in stale:
            self.assertIn(n, swa._collect_unbacked_swa_nodes(n))
        core.is_write_back = True
        # A split leaves both halves stale.
        big = next(n for n in stale if n is not window and len(n.key) >= 2 * PAGE)
        prefix, _ = core._split_node(big.key, big, PAGE)
        for half in (prefix, big):
            self.assertEqual(swa._collect_unbacked_swa_nodes(half), [])
        # A match ending in a stale window makes it live, so it is copied again.
        depth = 0
        node = window
        while node is not core.root_node:
            depth += len(node.key)
            node = node.parent
        w.match(nxt[: depth + 1])
        self.assertIn(window, swa._collect_unbacked_swa_nodes(window))
        # A live window still copies a stale node inside it, which it reads.
        tail = core.node_by_id(w.match(nxt + [1]).last_device_node)
        lru.reset_node_lru(tail.parent, skip_host_backup=True)
        self.assertIn(tail.parent, swa._collect_unbacked_swa_nodes(tail))

    def test_stale_window_host_copy_is_the_next_host_victim(self):
        """write_through copies every window to host at insert (simulated here by giving
        each node a host value). When the device evicts a window the switch demoted, its
        host copy is the next host victim, ahead of the host copies of windows evicted
        earlier, which requests still read."""
        w = self._world(enabled=True)
        core = w.cache.tree_core
        ct = ComponentType.SWA
        swa = w.cache.components[ct]
        lru, host_lru = core.lru_lists[ct], core.host_lru_lists[ct]
        prompt = _tokens(11, 16 * PAGE - 1)
        w.prefill(prompt, self.CHUNK, decode=1)
        nxt = prompt + _tokens(12, 2 * PAGE)
        with mock.patch.object(
            lru, "reset_node_lru", wraps=lru.reset_node_lru
        ) as demote:
            w.prefill(nxt, self.CHUNK, decode=1)
        # The demoted prompt window; a dropped reply's node has an output offset.
        stale = next(
            c.args[0]
            for c in demote.call_args_list
            if lru.skips_host_backup(c.args[0]) and c.args[0].output_offset is None
        )
        live = core.node_by_id(w.match(nxt + [1]).last_device_node).parent
        self.assertFalse(lru.skips_host_backup(live))

        def evict_device_swa(node):
            cd = node.component_data[ct]
            cd.host_value = torch.arange(len(cd.value))
            core._evict_component_and_detach_lru(
                node,
                swa,
                device_frees={ct: []},
                host_frees={ct: []},
                target=EvictLayer.DEVICE,
            )

        evict_device_swa(live)
        evict_device_swa(stale)
        self.assertIs(host_lru.get_lru_no_host_lock(), stale)
        self.assertIs(host_lru.get_prev_no_host_lock(stale), live)

    def test_other_reply_to_the_same_prompt_keeps_its_place(self):
        """Two conversations open with the same message and get different replies; the
        second reply is not a turn that dropped the first, so the first keeps its place."""
        for evict in (PAGE, 4 * PAGE):
            w = self._world(enabled=True)
            shared = _tokens(1, 4 * PAGE)
            for f in range(6):
                w.prefill(_tokens(900 + f, 6 * PAGE - 1), 10000, decode=1)
            first = shared + _tokens(2, PAGE - 1)
            w.prefill(first, 10000, decode=1, output=7)
            w.prefill(first, 10000, decode=1, output=8)
            w.prefill(first + [8] + _tokens(4, 3 * PAGE), 10000, decode=1)
            self._evict_swa(w, evict)
            self.assertEqual(
                w.prefill(first + [7] + _tokens(5, 3 * PAGE), 10000, decode=1),
                5 * PAGE,
            )
            w.cache.sanity_check()

    def test_output_sent_back_as_prompt_is_not_abandoned(self):
        """A reply the next turn sends back verbatim is prompt from then on."""
        w = self._world(enabled=True)
        prompt = _tokens(11, 8 * PAGE - 1)
        w.prefill(prompt, self.CHUNK, decode=1)
        # The output page holds the prompt's last 15 tokens and the output token (7).
        output_page = w.match(prompt + [7, 1]).last_device_node
        node = w.cache.tree_core.node_by_id(output_page)
        self.assertEqual(node.output_offset, PAGE - 1)
        w.prefill(prompt + [7] + _tokens(12, 2 * PAGE), self.CHUNK)
        self.assertIsNone(node.output_offset)
        w.cache.sanity_check()

    def test_branch_window_is_not_stale(self):
        """A window another cached sequence branches from must survive stale-window
        eviction, or every later session recomputes the shared prefix."""
        w = self._world(enabled=True)
        shared = _tokens(7, 160)
        a = shared + _tokens(8, self.PREFIX - 160)
        b = shared + _tokens(9, self.PREFIX - 160)
        w.prefill(a, self.CHUNK)
        # b matches 96 tokens and restores the shared window [96, 160).
        w.prefill(b, self.CHUNK)
        keep = 6 * (WINDOW + 1)  # two windows per tail plus the shared window
        w.cache.evict(
            EvictParams(
                num_tokens=0, swa_num_tokens=w.cache.swa_evictable_size() - keep
            )
        )
        self.assertEqual(len(w.match(shared + _tokens(10, 200)).device_indices), 160)
        w.cache.sanity_check()


class TestSWADemoteFinalWindows(unittest.TestCase):
    """DTF stage 2 (SGLANG_OPT_SWA_DEMOTE_FINAL_WINDOWS): a finished request's
    private tail windows go to the SWA eviction end after it unpins; on a first
    visit, the window ending at the page-floored media end stays put."""

    # 2048-token chunks against a 2047-token window, as served: every chunk insert
    # leaves the whole chunk with live SWA, so the media-end window is in the tree.
    CHUNK = 64
    MEDIA = 336  # system + media tokens: 21 pages, not on a chunk boundary
    TAIL = 400

    def _world(
        self, *, stage1: bool, stage2: bool, swa_tokens: int = 8192
    ) -> PrefillWorld:
        with (
            envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python"),
            envs.SGLANG_OPT_SWA_EVICT_STALE_WINDOWS_FIRST.override(stage1),
            envs.SGLANG_OPT_SWA_DEMOTE_FINAL_WINDOWS.override(stage2),
        ):
            return PrefillWorld(full_tokens=32768, swa_tokens=swa_tokens)

    def _media_hit_after_dead_tails(self, *, stage1: bool, stage2: bool) -> int:
        return self._media_hit(self._world(stage1=stage1, stage2=stage2))

    def _media_hit(self, w: PrefillWorld) -> int:
        """First visit of a media prefix, then newer unrelated requests; keep only
        a few windows of SWA; a sibling (same media, new text) matches."""
        media = _tokens(1, self.MEDIA)
        w.prefill(media + _tokens(2, self.TAIL), self.CHUNK, media=(64, self.MEDIA))
        for k in range(6):
            w.prefill(_tokens(50 + k, self.MEDIA + self.TAIL), self.CHUNK)
        keep = 4 * (WINDOW + 1)
        w.cache.evict(
            EvictParams(
                num_tokens=0, swa_num_tokens=w.cache.swa_evictable_size() - keep
            )
        )
        hit = len(w.match(media + _tokens(3, 200)).device_indices)
        w.cache.sanity_check()
        return hit

    def test_first_visit_media_window_survives_dead_tails(self):
        self.assertLess(
            self._media_hit_after_dead_tails(stage1=False, stage2=False), self.MEDIA
        )
        self.assertEqual(
            self._media_hit_after_dead_tails(stage1=False, stage2=True), self.MEDIA
        )
        self.assertEqual(
            self._media_hit_after_dead_tails(stage1=True, stage2=True), self.MEDIA
        )

    def test_stage1_alone_demotes_the_media_window(self):
        """Stage 1 treats the first-visit media window as stale; stage 2 exempts it."""
        self.assertLess(
            self._media_hit_after_dead_tails(stage1=True, stage2=False), self.MEDIA
        )

    def test_revisit_is_not_first_visit(self):
        """A request whose floor already covers the media has no media window to keep."""
        w = self._world(stage1=False, stage2=True)
        media = _tokens(1, self.MEDIA)
        w.prefill(media + _tokens(2, self.TAIL), self.CHUNK, media=(64, self.MEDIA))
        cached = w.prefill(
            media + _tokens(4, self.TAIL), self.CHUNK, media=(64, self.MEDIA)
        )
        self.assertEqual(cached, self.MEDIA)
        w.cache.sanity_check()

    def test_flag_set_selects_the_python_tree_core(self):
        """The flag is a Rust-fallback reason (tree_core_registry): with it set,
        an SWA cache gets the Python TreeCore even where CPU CI picks Rust, and
        the first-visit media window survives there."""
        rust = mock.MagicMock(side_effect=AssertionError("Rust TreeCore selected"))
        with (
            cpu_ci_tree_core_selection(rust),
            envs.SGLANG_OPT_SWA_DEMOTE_FINAL_WINDOWS.override(True),
            self.assertLogs(tree_core_registry.logger, "INFO") as logs,
        ):
            w = PrefillWorld(full_tokens=32768, swa_tokens=8192)
        rust.assert_not_called()
        self.assertEqual(
            [r.getMessage() for r in logs.records],
            [
                "Using the Python TreeCore: "
                "SGLANG_OPT_SWA_DEMOTE_FINAL_WINDOWS requires the Python TreeCore"
            ],
        )
        self.assertEqual(w.cache._tree_core_backend, "python")
        self.assertIsInstance(w.cache.tree_core, UnifiedTreeCore)
        self.assertEqual(self._media_hit(w), self.MEDIA)

    def test_flag_unset_keeps_the_rust_tree_core(self):
        """With the flag unset, an SWA cache gets the Rust TreeCore, as on main."""
        rust = mock.MagicMock()
        with cpu_ci_tree_core_selection(rust):
            w = PrefillWorld(full_tokens=32768, swa_tokens=8192)
        rust.assert_called_once_with(mock.ANY, w.cache.components)
        self.assertEqual(w.cache._tree_core_backend, "rust")
        self.assertIs(w.cache.tree_core, rust.return_value)
        self.assertFalse(w.cache.components[ComponentType.SWA].demotes_final_windows)

    def test_branch_window_is_not_demoted_at_finish(self):
        """The finish walk stops at a branch point: a window another cached
        sequence branches from keeps its LRU place."""
        w = self._world(stage1=False, stage2=True)
        shared = _tokens(7, 160)
        a = shared + _tokens(8, self.MEDIA + self.TAIL - 160)
        b = shared + _tokens(9, self.MEDIA + self.TAIL - 160)
        w.prefill(a, self.CHUNK)
        w.prefill(b, self.CHUNK)
        keep = 6 * (WINDOW + 1)
        w.cache.evict(
            EvictParams(
                num_tokens=0, swa_num_tokens=w.cache.swa_evictable_size() - keep
            )
        )
        self.assertEqual(len(w.match(shared + _tokens(10, 200)).device_indices), 160)
        w.cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
