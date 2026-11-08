# SPDX-License-Identifier: Apache-2.0
"""Prefill side of the P/D prefix pull: a PrefixFetcher over the reverse Raiden
pipeline transfer.

After bootstrap every PP stage makes the same decision for a request: pull
[P_m, D) when the decode holds D (and locked it), the stage's own device match P_m
falls short by at least SGLANG_TPU_PD_PREFIX_PULL_MIN_TOKENS, and pages fit. It
reserves full pages for [P_m, D) and SWA pages for the window [D - W, D) as the
request's KV, and pulls them (pp_reverse.ReversePull). The request's sender reports
Bootstrapping until the stage's pull settles, so the bootstrap consensus releases
the request on every stage at once; finish() then inserts [0, D) into the stage's
prefix cache and frees the request's row, keeping a lock on the prefix until the
request is admitted, so its normal match starts at D.

A pull that fails reports Failed on its stage. With SGLANG_TPU_PD_PREFIX_PULL_STRICT
the request fails with that reason; otherwise every stage recovers it to compute
the prefix, with a warning and a fallback count each time.
"""

from __future__ import annotations

import logging
import time
from array import array
from dataclasses import dataclass
from typing import Any, Optional

import torch

from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
from sglang.srt.managers.schedule_batch import DecLockRefParams, Req, ReqKvInfo
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang_tpu.disaggregation.prefix_pull import window_start
from sglang_tpu.disaggregation.raiden import pp_reverse
from sglang_tpu.disaggregation.raiden.sender import RaidenKVSender
from sglang_tpu.envs import envs

logger = logging.getLogger(__name__)

SKIP_NO_MEMORY = "no_memory"
SKIP_NO_REQ_SLOT = "no_req_slot"
PEER_STAGE = "peer_stage"


@dataclass
class Fetch:
    room: int
    start: int
    end: int
    # Full-attention KV indices of [start, end), owned by the request until finish.
    kv_loc: torch.Tensor


def _page_ids(indices: torch.Tensor, page_size: int) -> list[int]:
    return [int(i) // page_size for i in indices[::page_size].cpu().tolist()]


def pull_plan(
    *, decode_end: int, prefill_match: int, window: int, page_size: int, min_tokens: int
) -> Optional[int]:
    """The window start of a worthwhile pull of [prefill_match, decode_end), or
    None. The window must lie inside the pulled range, which needs at least a
    window of tokens."""
    if decode_end - prefill_match < max(min_tokens, window, page_size):
        return None
    first = window_start(decode_end, window, page_size)
    return first if first >= prefill_match else None


class PrefixPullFetcher:
    """One PP stage's PrefixFetcher. All calls run on the scheduler thread."""

    def __init__(self, queue: PrefillBootstrapQueue, metrics: Any):
        self.queue = queue
        self.scheduler = queue.scheduler
        self.metrics = metrics
        self.fetches: dict[str, Fetch] = {}
        # Prefix locks from finish() until the request is admitted or dropped.
        self.prefix_locks: dict[str, tuple[Any, DecLockRefParams]] = {}

    @property
    def _pipeline(self):
        return self.queue.kv_manager._pipeline

    # ----- PrefixFetcher ----------------------------------------------------

    def is_fetching(self, req: Req) -> bool:
        return req.rid in self.fetches

    def start(self, req: Req) -> bool:
        pipeline = self._pipeline
        tree = self.scheduler.tree_cache
        if (
            pipeline is None
            or pipeline.reverse is None
            or not isinstance(tree, UnifiedRadixCache)
            or not tree.supports_swa()
        ):
            return False
        # Only Raiden transfers carry a decode prefix report; the PD warmup request
        # runs on a FakeKVSender and has nothing to pull.
        if not isinstance(req.disagg_kv_sender, RaidenKVSender):
            return False
        report = req.disagg_kv_sender.decode_prefix_report()
        if report is None or not report.pull_held or report.pullable_len <= 0:
            return False
        end = report.pullable_len
        match = tree.match_prefix(
            MatchPrefixParams(
                key=RadixKey(
                    array("q", req.origin_input_ids[:end]),
                    req.extra_key,
                    cache_salt=req.cache_salt,
                )
            )
        )
        start = len(match.device_indices)
        page_size = tree.page_size
        window = tree.sliding_window_size
        first = pull_plan(
            decode_end=end,
            prefill_match=start,
            window=window,
            page_size=page_size,
            min_tokens=envs.SGLANG_TPU_PD_PREFIX_PULL_MIN_TOKENS.get(),
        )
        if first is None:
            return False
        receipt = tree.inc_lock_ref(match.last_device_node).to_dec_params()
        kv_loc = self._allocate(
            req, start=start, end=end, first=first, prefix=match.device_indices
        )
        if kv_loc is None:
            tree.dec_lock_ref(match.last_device_node, receipt)
            return False
        pool = self.scheduler.req_to_token_pool
        row = req.kv.req_pool_idx
        if start > 0:
            pool.write((row, slice(0, start)), match.device_indices)
        pool.write((row, slice(start, end)), kv_loc)
        req.prefix_indices = match.device_indices
        req.last_node = match.last_device_node
        req.lock_receipt = receipt
        req.kv.cache_protected_len = start
        req.kv.kv_committed_len = end
        req.kv.kv_allocated_len = end
        req.kv.set_evicted_seqlen(ComponentType.SWA, first)
        allocator = self.scheduler.token_to_kv_pool_allocator
        swa = allocator.translate_swa_indices_for_transfer(kv_loc[first - start :])
        with pipeline.lock:
            pipeline.reverse.start(
                req.bootstrap_room,
                start=start,
                end=end,
                full=_page_ids(kv_loc, page_size),
                swa=_page_ids(swa, page_size),
            )
        self.fetches[req.rid] = Fetch(
            room=req.bootstrap_room, start=start, end=end, kv_loc=kv_loc
        )
        logger.info(
            "Prefix pull room=%d stage=%d: [%d, %d) from the decode",
            req.bootstrap_room,
            pipeline.manager.pp_rank,
            start,
            end,
        )
        return True

    def finish(self, req: Req) -> None:
        fetch = self.fetches.pop(req.rid)
        pipeline = self._pipeline
        with pipeline.lock:
            state, reason = pipeline.reverse.status(fetch.room)
            pipeline.reverse.forget(fetch.room)
        # The sender reports ready only for a pull that landed, or a failed one
        # under the recovery path, which abandon() handles instead.
        if state != pp_reverse.DONE:
            raise RuntimeError(
                f"Prefix pull room={fetch.room} released in state {state}: {reason}"
            )
        tree = self.scheduler.tree_cache
        req.full_untruncated_fill_ids = req.origin_input_ids
        tree.checkpoint(req, up_to=fetch.end)
        self.prefix_locks[req.rid] = (
            req.last_node,
            tree.inc_lock_ref(req.last_node).to_dec_params(),
        )
        release_kv_cache(req, tree, is_insert=False)
        _reset(req)
        self.metrics.pulled(fetch.end - fetch.start)

    def abandon(self, req: Req) -> bool:
        fetch = self.fetches.pop(req.rid)
        pipeline = self._pipeline
        self._wait_settled(fetch)
        with pipeline.lock:
            state, reason = pipeline.reverse.status(fetch.room)
            kind = (
                pipeline.reverse.failure_kind(fetch.room)
                if state == pp_reverse.FAILED
                else PEER_STAGE
            )
            pipeline.reverse.forget(fetch.room)
        release_kv_cache(req, self.scheduler.tree_cache, is_insert=False)
        _reset(req)
        if envs.SGLANG_TPU_PD_PREFIX_PULL_STRICT.get():
            logger.error(
                "Prefix pull room=%d failed (%s): %s; the request fails "
                "(SGLANG_TPU_PD_PREFIX_PULL_STRICT)",
                fetch.room,
                kind,
                reason or "another stage failed",
            )
            self.metrics.failed(kind)
            return False
        logger.warning(
            "Prefix pull room=%d failed (%s): %s; recomputing %d tokens "
            "(SGLANG_TPU_PD_PREFIX_PULL_STRICT=0)",
            fetch.room,
            kind,
            reason or "another stage failed",
            fetch.end - fetch.start,
        )
        self.metrics.fallback(kind)
        return True

    # ----- prefix locks -----------------------------------------------------

    def release_prefix_lock(self, req: Req) -> None:
        held = self.prefix_locks.pop(req.rid, None)
        if held is not None:
            self.scheduler.tree_cache.dec_lock_ref(*held)

    # ----- internals --------------------------------------------------------

    def _allocate(
        self, req: Req, *, start: int, end: int, first: int, prefix: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """The request's row and KV for [start, end), SWA only for [first, end)."""
        allocator = self.scheduler.token_to_kv_pool_allocator
        tree = self.scheduler.tree_cache
        pool = self.scheduler.req_to_token_pool
        if pool.alloc([req]) is None:
            self.metrics.skipped(SKIP_NO_REQ_SLOT)
            return None
        device = allocator.device
        # The allocator continues the prefix's last page.
        last_loc = (
            prefix[-1:].to(dtype=torch.int64, device=device)
            if start > 0
            else torch.tensor([-1], dtype=torch.int64, device=device)
        )

        def alloc():
            return allocator.alloc_extend_swa_tail(
                prefix_lens=torch.tensor([start], dtype=torch.int64, device=device),
                prefix_lens_cpu=torch.tensor([start], dtype=torch.int64),
                seq_lens=torch.tensor([end], dtype=torch.int64, device=device),
                seq_lens_cpu=torch.tensor([end], dtype=torch.int64),
                last_loc=last_loc,
                extend_num_tokens=end - start,
                swa_tail_len=end - first,
            )

        kv_loc = alloc()
        if kv_loc is None:
            tree.evict(EvictParams(num_tokens=end - start, swa_num_tokens=end - first))
            kv_loc = alloc()
        if kv_loc is None:
            pool.free(req)
            self.metrics.skipped(SKIP_NO_MEMORY)
            return None
        return kv_loc

    def _wait_settled(self, fetch: Fetch) -> None:
        """Every stage abandons in the same consensus step, so the pages are
        freed in step too; a transfer still writing them is waited out."""
        pipeline = self._pipeline
        waited = False
        while True:
            with pipeline.lock:
                if not pipeline.reverse.in_flight(fetch.room):
                    break
            waited = True
            # The manager's drain, so forward completions are recorded too.
            self.queue.kv_manager._drain_engine()
            time.sleep(0.001)
        if waited:
            logger.error(
                "Prefix pull room=%d: waited for an in-flight transfer before "
                "freeing its pages",
                fetch.room,
            )


def _reset(req: Req) -> None:
    """Back to the state of a request that never held KV."""
    assert not req.kv.holds_kv
    req.kv = ReqKvInfo()
    req.prefix_indices = torch.empty((0,), dtype=torch.int64)
    req.last_node = None
    req.lock_receipt = DecLockRefParams()
    req.swa_prefix_lock_released = False
    req.swa_stale_window_floor = None
    req.extend_range = None
