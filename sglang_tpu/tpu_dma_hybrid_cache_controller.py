"""Hybrid HiCache controller for TPU (full + SWA pools) over a TPU DMA backend.

Hybrid sliding-window models run under the unified cache with a
``HostPoolGroup`` of two host pools: the full-attention pool (anchor,
``PoolName.KV``) and the SWA pool (``PoolName.SWA``), each mirroring its own
device sub-pool with its own page-id space. Each ``TpuDMAHostKVPool`` owns a
``TpuDMAKVTransfer`` engine over just its sub-pool's layers, so a transfer
issues one all-layer engine copy per pool.

torch_tpu has no real device streams or events, so the CUDA
``HybridCacheController`` data path (side streams, per-layer load loop,
recorded events) is replaced the same way ``TpuDMAHiCacheController``
replaces the single-pool path: the engine issues every copy off the scheduler
thread and returns poll-only future-events, which serve as the acks and feed a
``BatchDoneCounter`` gating the whole forward instead of a per-layer
``LayerDoneCounter``; index tensors stay host-resident throughout
(``SGLANG_TPU_REQ_TO_TOKEN_ON_HOST=1``).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

import torch

from sglang.srt.managers.cache_controller import BatchDoneCounter, HiCacheAck
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    CacheOperation,
    HybridCacheController,
)
from sglang_tpu.kv_free_fence import (
    KVFreeFence,
    assert_not_forward_write_target,
    deferred_dma_read,
)
from sglang_tpu.tpu_dma_hicache import NOOP_EVENT, CombinedFutureEvent, TransferEvent

logger = logging.getLogger(__name__)

# Pools indexed by KV allocator pages, which the KV free fence keeps from loads
# while a forward may still write them; any other pool's slots are unfenced.
_KV_PAGE_POOLS = frozenset(
    (PoolName.KV, PoolName.SWA, PoolName.DRAFT, PoolName.DRAFT_SWA)
)


def _check_kv_page_pool(name: PoolName) -> None:
    if name not in _KV_PAGE_POOLS:
        raise NotImplementedError(
            f"The TPU hybrid HiCache path does not offload the {name.value} pool."
        )


class TpuDMAHybridCacheController(HybridCacheController):
    """Hybrid (multi-pool) HiCache controller whose transfers go through a
    TPU DMA backend, one all-layer call per pool entry."""

    # Set by the scheduler for the tree's controller; see fence_hicache_backups.
    kv_free_fence: Optional[KVFreeFence] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for entry in self.mem_pool_host.entries:
            _check_kv_page_pool(entry.name)
        # The engine gates the whole forward at once (no per-layer device streams).
        # Swap the per-layer LayerDoneCounter for a BatchDoneCounter and
        # re-register it on the device pool wrapper (SWAKVPool holds the
        # counter itself; its sub-pools carry None), so CompiledForwardShim's
        # whole-batch ``wait()`` resolves against the reload event.
        self.layer_done_counter = BatchDoneCounter()
        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)

    # ----- draft registration guards -----------------------------------------

    def set_draft_kv_pool(self, draft_device_pool, draft_host_pool):
        # Hybrid models register speculative draft KV as a sidecar pool entry
        # (register_host_pool_entry below); this legacy hook is only reachable
        # from the non-unified cache, which hybrid models never use.
        raise NotImplementedError(
            "Hybrid draft KV registers as a sidecar pool entry, not through "
            "the legacy draft hook."
        )

    def set_mtp_draft_pools(self, mtp_draft_device_pools):
        raise NotImplementedError(
            "MTP draft KV offload is not supported on the TPU hybrid HiCache path."
        )

    # ----- sidecar registration ----------------------------------------------

    def register_host_pool_entry(self, entry) -> None:
        _check_kv_page_pool(entry.name)
        super().register_host_pool_entry(entry)
        # Draft sidecar KV is consumed by the draft model's forward, whose
        # CompiledForwardShim gates on the draft pool's own transfer counter.
        # Point it at the same whole-batch counter as the target pools: the
        # reload ack is a conjunction over every pool's DMA, so the draft KV
        # has landed by the time the counter releases the batch.
        if entry.name in (PoolName.DRAFT, PoolName.DRAFT_SWA):
            entry.device_pool.register_layer_transfer_counter(self.layer_done_counter)

    # ----- lifecycle ---------------------------------------------------------

    def reset(self):
        # Drain outstanding DMAs before the base clears the ack queues and the
        # cache frees pages: a page the engine is still copying must not return to
        # a free list mid-DMA.
        for ack in list(self.ack_write_queue) + list(self.ack_load_queue):
            ack.finish_event.synchronize()
        self.layer_done_counter.reset()
        super().reset()

    # ----- index handling -----------------------------------------------------

    def move_indices(self, host_indices, device_indices):
        # Engines address whole pages; the token->page conversion happens in
        # each host pool. Keep index tensors as-is: no stream moves, no
        # host/device relocation.
        return host_indices, device_indices

    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> Optional[torch.Tensor]:
        assert_not_forward_write_target(self.mem_pool_device_allocator)
        device_indices = super().load(
            host_indices=host_indices,
            priority=priority,
            node_id=node_id,
            extra_pools=extra_pools,
        )
        if device_indices is None:
            return None
        # Keep every index tensor host-resident (node.value / prefix_indices
        # stay on CPU under the host-resident req_to_token). The tree-owned
        # PoolTransfer objects were populated in place by
        # _resolve_pool_transfers_allocation, so their device indices are
        # rewritten in place too — the SWA component's commit slices these
        # same tensors.
        device_indices = device_indices.cpu()
        queued = self.load_queue[-1]
        queued.device_indices = device_indices
        for transfer in queued.pool_transfers or []:
            if transfer.device_indices is not None:
                transfer.device_indices = transfer.device_indices.cpu()
        return device_indices

    # ----- transfers ----------------------------------------------------------

    def _issue_pool_dmas(self, op: CacheOperation, load: bool) -> List[TransferEvent]:
        """One all-layer engine copy per pool entry of the merged operation.

        The ``HostPoolGroup`` transfer methods discard the returned
        future-events, so the child host pools are driven directly.
        """
        anchor = self.mem_pool_host.anchor_entry
        events = []
        if load:
            events.append(
                anchor.host_pool.load_all_layer(
                    anchor.device_pool, op.host_indices, op.device_indices
                )
            )
        else:
            events.append(
                anchor.host_pool.backup_from_device_all_layer(
                    anchor.device_pool,
                    op.host_indices,
                    op.device_indices,
                    self.io_backend,
                )
            )
        for transfer in op.pool_transfers or []:
            entry = self.mem_pool_host.entry_map.get(transfer.name)
            if entry is None or entry is anchor:
                continue
            if load:
                events.append(
                    entry.host_pool.load_all_layer(
                        entry.device_pool,
                        transfer.host_indices,
                        transfer.device_indices,
                    )
                )
            else:
                events.append(
                    entry.host_pool.backup_from_device_all_layer(
                        entry.device_pool,
                        transfer.host_indices,
                        transfer.device_indices,
                        self.io_backend,
                    )
                )
        return events

    def start_writing(self) -> None:
        if not self.write_queue:
            return
        op = CacheOperation.merge_ops(self.write_queue)
        self.write_queue.clear()

        finish_event = deferred_dma_read(
            self.kv_free_fence,
            lambda: CombinedFutureEvent(self._issue_pool_dmas(op, load=False)),
        )
        self.ack_write_queue.append(
            HiCacheAck(
                start_event=NOOP_EVENT,
                finish_event=finish_event,
                node_ids=op.node_ids,
                num_tokens=len(op.device_indices),
                num_tokens_by_pool=self._num_tokens_by_pool(op),
                num_bytes=self._transfer_num_bytes(op),
            )
        )

    def start_loading(self) -> int:
        if not self.load_queue:
            return -1
        producer_id = self.layer_done_counter.update_producer()
        op = CacheOperation.merge_ops(self.load_queue)
        self.load_queue.clear()

        # All pools' H2d; each future-event comes back before its DMA lands.
        # The consuming forward's CompiledForwardShim wait() polls the combined
        # event through the whole-batch gate (poll-only; never Await), so the
        # DMAs overlap whatever the scheduler runs before that forward.
        finish_event = CombinedFutureEvent(self._issue_pool_dmas(op, load=True))
        self.layer_done_counter.set_event(producer_id, finish_event)
        self.ack_load_queue.append(
            HiCacheAck(
                start_event=NOOP_EVENT,
                finish_event=finish_event,
                node_ids=op.node_ids,
                num_tokens=len(op.device_indices),
                num_tokens_by_pool=self._num_tokens_by_pool(op),
                num_bytes=self._transfer_num_bytes(op),
            )
        )
        return producer_id

    def is_load_back_event_done(
        self, consumer_index: int, all_reduce: Callable[[torch.Tensor, Any], None]
    ) -> bool:
        """Return True once the load-back in ``consumer_index`` landed on all ranks.

        The controller-level counterpart of the cache's
        ``is_load_back_event_done``: it only answers the question and has no
        side effects. The cache runs ``loading_check`` after a True answer.

        Each ``BatchDoneCounter`` slot holds the load's combined future-event
        itself, or ``None`` before the slot's first load. An engine copy can
        finish at different times on different ranks, while the cache's
        ``loading_check`` issues collectives. ``all_reduce`` combines the local
        results over the cache's rank groups, so every rank makes the same
        decision and enters those collectives together.
        """
        if consumer_index < 0:
            return True
        event = self.layer_done_counter.events[consumer_index]
        done = torch.tensor(
            int(event is None or event.query()), dtype=torch.int, device="cpu"
        )
        all_reduce(done, torch.distributed.ReduceOp.MIN)
        return bool(done.item())
