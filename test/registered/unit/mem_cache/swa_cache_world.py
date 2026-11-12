"""Shared helpers for the hybrid-SWA UnifiedRadixCache unit tests: real FULL +
SWA pools with value-checked KV, and the CPU CI TreeCore selection."""

import importlib.util
import os
from array import array
from contextlib import contextmanager, nullcontext
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache import tree_core_registry
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=0, suite="base-a-test-cpu", disabled="SWA cache test helper")

# The served 2047-token window on 512-token pages, scaled 1/32: a window spans
# four 16-token pages and reads [seqlen - 64, seqlen).
PAGE, WINDOW = 16, 63
SWA = ComponentType.SWA
POISON = 30000.0  # large finite poison (a NaN would hide in comparisons)


def kv_value(token: int, pos: int) -> float:
    """The KV a cold prefill writes for ``token`` at ``pos`` (exact in BF16)."""
    return float((token * 31 + pos * 7) % 251 + 1)


@contextmanager
def cpu_ci_tree_core_selection(rust_factory):
    """Select the TreeCore as CPU CI does: the default backend ("rust"), the
    prebuilt Rust extension, SGLANG_RUST_BUILD_MODE=never, Linux and a
    supported torch, so only a fallback reason keeps a cache on the Python
    TreeCore. No SWA option is set on entry. ``rust_factory`` stands in for the
    extension, which this test never loads."""
    module = tree_core_registry._RUST_TREE_CORE_MODULE
    find_spec = importlib.util.find_spec

    def bundled_find_spec(name, *args, **kwargs):
        return object() if name == module else find_spec(name, *args, **kwargs)

    with (
        mock.patch.dict(os.environ),
        mock.patch.object(tree_core_registry.sys, "platform", "linux"),
        mock.patch.object(tree_core_registry.torch, "__version__", "2.14.1"),
        mock.patch.object(
            tree_core_registry.importlib.util, "find_spec", bundled_find_spec
        ),
        mock.patch.dict(tree_core_registry._TREE_CORE_REGISTRY, {"rust": rust_factory}),
    ):
        for flag in (
            envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND,
            envs.SGLANG_OPT_SWA_EVICT_STALE_WINDOWS_FIRST,
            envs.SGLANG_OPT_SWA_DEMOTE_FINAL_WINDOWS,
            envs.SGLANG_OPT_SWA_CAP,
        ):
            flag.clear()
        envs.SGLANG_RUST_BUILD_MODE.set("never")
        yield


class KVWorld:
    """A real UnifiedRadixCache on real FULL + SWA pools. Every slot gets the
    cold value of its token, so a reused slot can be checked against a
    recompute. Subclasses drive the request lifecycle."""

    def __init__(self, *, full_tokens: int, swa_tokens: int, pin_python: bool = True):
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
        # The helpers read tree nodes (tree_core.node_by_id), which only the
        # Python TreeCore exposes; TPU always uses it. pin_python=False leaves
        # the backend to the deployment's selection.
        with (
            envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python")
            if pin_python
            else nullcontext()
        ):
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
    def _write(self, full: torch.Tensor, tokens: list[int], start: int) -> None:
        vals = torch.tensor(
            [kv_value(t, start + i) for i, t in enumerate(tokens)],
            dtype=torch.float32,
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
        self._write(full[:n], tokens, start)
        return full[:n]

    # -- request lifecycle
    def _next_token(self, req: Req) -> int:
        return 1 + (len(req.origin_input_ids) * 13 + len(req.output_ids) * 17) % 40000

    def decode_step(self, req: Req) -> None:
        """One decode step (one KV slot, no tree insert): a new page at a page
        boundary, else the next slot of the last page."""
        row = req.kv.req_pool_idx
        pos = req.seqlen - 1  # the KV of the last token
        tok = req.output_ids[-1]
        if pos % PAGE == 0:
            full = self._alloc([tok], pos)
        else:
            prev = self.req_to_token_pool.req_to_token[row, pos - 1].to(torch.int64)
            full = (prev + 1).view(1)
            self._write(full, [tok], pos)
        self.req_to_token_pool.write((row, slice(pos, pos + 1)), full)
        req.kv.kv_committed_len = req.kv.kv_allocated_len = pos + 1
        req.output_ids.append(self._next_token(req))
        req.decode_batch_idx = getattr(req, "decode_batch_idx", 0) + 1

    def evict(self, swa_tokens: int, full_tokens: int = 0) -> None:
        self.cache.evict(EvictParams(num_tokens=full_tokens, swa_num_tokens=swa_tokens))

    def match(self, tokens: list[int]):
        return self.cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", tokens), None, limit=len(tokens) - 1)
            )
        )
