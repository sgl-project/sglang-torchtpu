from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import random
import unittest
from array import array

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

    def prefill(self, tokens: list[int], chunk: int) -> int:
        req = Req(
            rid=f"r{self.rid}",
            origin_input_text="",
            origin_input_ids=array("q", tokens),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        self.rid += 1
        self.req_to_token_pool.alloc([req])
        req.init_next_round_input(tree_cache=self.cache)
        req.lock = self.cache.lock(req.last_node)
        cached = req.prefix_len
        row = req.kv.req_pool_idx
        if cached:
            self.req_to_token_pool.write(
                (row, slice(0, cached)), self.cache.prefix_device_indices(req)
            )
        req.kv.kv_committed_len = req.kv.kv_allocated_len = cached
        pos = cached
        while pos < len(tokens):
            end = min(pos + chunk, len(tokens))
            req.extend_end = end
            self.req_to_token_pool.write((row, slice(pos, end)), self._alloc(end - pos))
            req.kv.kv_committed_len = req.kv.kv_allocated_len = end
            pos = end
            if pos < len(tokens):
                checkpoint_kv_cache(req, self.cache)
                req.init_next_round_input()
        req.output_ids.append(7)
        checkpoint_kv_cache(req, self.cache)
        req.finished_reason = FINISH_LENGTH(length=0)
        release_kv_cache(req, self.cache, checkpoint=True)
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
        self.assertEqual(w.match(shared + _tokens(10, 200)).device_prefix_len, 160)
        w.cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
