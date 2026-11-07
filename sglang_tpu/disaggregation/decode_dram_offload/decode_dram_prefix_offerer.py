"""Decode side of the recall: offer the pushed prefix this decode holds in its
DRAM (HiCache host pool) back to the prefill that serves the session's next
turn, and keep it pinned until the prefill's read settled.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch.distributed as dist

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang_tpu.disaggregation.decode_dram_offload.radix_host_ops import (
    deepest_host_backed_path,
    lock_host_path,
    offer_limit,
    prefix_radix_key,
    unlock_host_path,
)
from sglang_tpu.disaggregation.decode_dram_offload.rank_lockstep import (
    DramOffloadWorkerContext,
)


class DecodeDramPrefixOfferer:
    """Decode side: offer the pushed prefix a request's decode holds in RAM,
    and keep it pinned until queue ownership and the prefill's read end."""

    def __init__(self, ctx: DramOffloadWorkerContext, queue: Any) -> None:
        # ``queue`` is the transfer queue that owns a request once its
        # PULL_REQ is out.
        self._ctx = ctx
        self._queue = queue
        # The host locks of each room's offered path.
        self.offers: Dict[int, List[Any]] = {}
        # Rooms whose request left the queue; their pins wait for the read.
        self._released: set = set()

    def offer_host_prefix(self, decode_req: Any) -> None:
        """Pin and offer the host-held prefix of a request whose PULL_REQ is
        about to be sent."""
        ctx, req = self._ctx, decode_req.req
        if req.bootstrap_room in self.offers:
            return
        key = prefix_radix_key(
            req=req, num_tokens=offer_limit(req=req, page_size=ctx.page_size)
        )
        # Matching can split radix nodes but never starts a load-back.
        match = ctx.cache.match_prefix(MatchPrefixParams(key=key))
        # The deepest matched prefix whose full KV is whole in host RAM.
        node, pages = deepest_host_backed_path(
            ctx.cache, ctx.host_pool, match.best_match_node
        )
        if not ctx.lockstep.agree([len(pages or ())]) or not pages:
            return
        # The prefill reads pages of every node of the path.
        self.offers[req.bootstrap_room] = lock_host_path(ctx.cache, node)
        ctx.channel.offer_resident_prefix(req.bootstrap_room, pages)

    def close_departed_offers(self) -> None:
        """Close the offers of requests that left the transfer queue.

        Queue membership and the deferred releases change in steps every rank
        agrees on, so every rank closes the same rooms in the same step. The
        pages stay pinned until the prefill's native read settles (``step``).
        """
        if len(self.offers) == len(self._released):
            return
        owned = {dr.req.bootstrap_room for dr in self._queue.queue}
        owned.update(
            entry[0].req.bootstrap_room for entry in self._queue._deferred_releases
        )
        for room in self.offers:
            if room not in owned and room not in self._released:
                # Close before the tree pin goes: a late PULL on the listener
                # must not register pages the queue relinquished.
                self._ctx.channel.release_prefix_offer(room)
                self._released.add(room)

    def unpin_settled_host_offers(self) -> None:
        """Unpin the closed offers whose read settled on every rank."""
        ctx = self._ctx
        ctx.channel.drain()
        rooms = sorted(self._released)
        settled = ctx.lockstep.reduce(
            [[int(not ctx.channel.host_prefix_pending(room))] for room in rooms],
            dist.ReduceOp.MIN,
        )
        for room, (done,) in zip(rooms, settled):
            if done:
                unlock_host_path(ctx.cache, self.offers.pop(room))
                self._released.remove(room)

    def before_cache_reset(self) -> None:
        self.close_departed_offers()
        self.unpin_settled_host_offers()
        if self.offers:
            raise RuntimeError(
                "Cannot reset KV memory while host prefix offers are pinned"
            )
