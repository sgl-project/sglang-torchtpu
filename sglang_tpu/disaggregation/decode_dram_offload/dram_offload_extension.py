"""Decode DRAM as an extension of the prefill DRAM offload (HiCache L2).

The L2 controller (``TpuDMAHybridCacheController``) offloads KV from device
memory to this engine's host pool. With
SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM=1 it also holds the engine's
end of the extension to decode DRAM, which moves pages between that host pool
and the host pool of the peer role:

- ``PrefillDramOffloadExtension`` pushes cold branches to the decode that served
  them (prefill_dram_victim_pusher.py) and recalls them when they return
  (prefill_dram_prefix_puller.py).
- ``DecodeDramOffloadExtension`` lands pushed branches
  (decode_dram_victim_receiver.py) and offers them back
  (decode_dram_prefix_offerer.py).

L2 knows neither requests nor peers, so the P/D layer supplies two things: the
raiden channel to the peer rank of a request (``raiden/dram_offload_channel.py``)
and the bootstrap step in which a recall must finish. The bytes move RAM to RAM
through the host pool's own raiden manager: no HBM, no load back, no staging
slot on either side.
"""

from __future__ import annotations

from typing import Any, Sequence

from sglang_tpu.disaggregation.decode_dram_offload.decode_dram_prefix_offerer import (
    DecodeDramPrefixOfferer,
)
from sglang_tpu.disaggregation.decode_dram_offload.decode_dram_victim_receiver import (
    DecodeDramVictimReceiver,
)
from sglang_tpu.disaggregation.decode_dram_offload.prefill_dram_prefix_puller import (
    PrefillDramPrefixPuller,
)
from sglang_tpu.disaggregation.decode_dram_offload.prefill_dram_victim_pusher import (
    PrefillDramVictimPusher,
)
from sglang_tpu.disaggregation.decode_dram_offload.radix_host_ops import (
    anchor_host_pool,
    swa_host_pool,
    window_tokens,
)
from sglang_tpu.disaggregation.decode_dram_offload.rank_lockstep import (
    DramOffloadWorkerContext,
    RankLockstep,
)


def attach_dram_offload_extension(
    *, cache: Any, channel: Any, scheduler: Any, decode_queue: Any = None
) -> Any:
    """Extend ``cache``'s L2 controller to the peer role's DRAM, reached over
    the raiden ``channel``. A decode passes its transfer queue."""
    host_pool, swa_pool = anchor_host_pool(cache), swa_host_pool(cache)
    channel.bind_host_pools(host_pool, swa_pool)
    page_size = int(scheduler.token_to_kv_pool_allocator.page_size)
    ctx = DramOffloadWorkerContext(
        cache=cache,
        host_pool=host_pool,
        page_size=page_size,
        channel=channel,
        lockstep=RankLockstep.of(scheduler),
        swa_host_pool=swa_pool,
        window_tokens=window_tokens(cache, page_size),
    )
    extension = (
        PrefillDramOffloadExtension(ctx)
        if decode_queue is None
        else DecodeDramOffloadExtension(ctx, decode_queue)
    )
    cache.cache_controller.decode_dram_extension = extension
    return extension


class PrefillDramOffloadExtension:
    """The prefill's end: recall and push, driven around each step of the
    bootstrap queue (core_hooks.py)."""

    def __init__(self, ctx: DramOffloadWorkerContext) -> None:
        self.recall = PrefillDramPrefixPuller(ctx)
        self.push = PrefillDramVictimPusher(ctx)

    def before_bootstrap_step(
        self, reqs: Sequence[Any], *, waiting_reqs: Sequence[Any] = (), serving: bool
    ) -> None:
        """Before the bootstrap queue's step, over its queued requests.
        ``serving``: the engine has a request queued or running."""
        self.recall.advance(reqs, waiting_reqs=waiting_reqs)
        self.push.advance(serving=serving)

    def commit_recall(self, req: Any) -> None:
        """A request left bootstrap for admission: insert what was recalled
        for it, or drop a recall still in flight."""
        self.push.note_decode_of(
            req, recalled=self.recall.insert_recalled_host_prefix(req)
        )

    def after_bootstrap_step(
        self, *, bootstrapped: Sequence[Any], queued: Sequence[Any]
    ) -> None:
        """After the bootstrap queue's step. Every rank bootstraps and drops
        the same requests in a step, so the tree inserts are in step."""
        for req in bootstrapped:
            self.commit_recall(req)
        self.recall.drop_departed(queued)

    def holds_host_pages_outside_tree(self) -> bool:
        return (
            self.recall.holds_host_landing_pages() or self.push.has_pushes_in_flight()
        )

    def settle_host_transfers(self) -> None:
        self.recall.free_settled_host_landings()
        # Paused and empty control ticks must retire existing exports too.
        self.push.retire_settled_pushes()

    def release_all(self) -> None:
        self.push.release_all()
        self.recall.release_all()

    def before_cache_reset(self) -> None:
        """Let go of every page held outside the tree, or raise while the
        decode may still read one; the reset is retried."""
        self.release_all()
        self.push.clear_push_routes()


class DecodeDramOffloadExtension:
    """The decode's end: land pushed branches and offer them back, driven by
    the decode queues."""

    def __init__(self, ctx: DramOffloadWorkerContext, queue: Any) -> None:
        self.landing = DecodeDramVictimReceiver(ctx)
        self.source = DecodeDramPrefixOfferer(ctx, queue)

    def advance(self) -> None:
        self.source.close_departed_offers()
        self.source.unpin_settled_host_offers()
        self.landing.advance()

    def settle_host_transfers(self) -> None:
        """A tick of a paused engine, which does not step the decode queues:
        let go of what settled, so that the engine can go idle and be flushed."""
        self.source.close_departed_offers()
        self.source.unpin_settled_host_offers()
        self.landing.finish_settled_host_landings()

    def holds_host_pages_outside_tree(self) -> bool:
        """Whether offered or landing pages sit outside the tree."""
        return bool(self.source.offers) or self.landing.has_host_landings()

    def before_cache_reset(self) -> None:
        """Let go of every page held outside the tree, or raise while a
        prefill may still read or write one; the reset is retried."""
        self.landing.drain_host_landings()
        self.source.before_cache_reset()
