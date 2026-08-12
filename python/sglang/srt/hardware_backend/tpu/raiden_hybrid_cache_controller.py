"""Raiden-backed hybrid HiCache controller for TPU (full + SWA pools).

Hybrid sliding-window models run under the unified cache with a
``HostPoolGroup`` of two host pools: the full-attention pool (anchor,
``PoolName.KV``) and the SWA pool (``PoolName.SWA``), each mirroring its own
device sub-pool with its own page-id space. Each ``TpuRaidenHostKVCache``
owns a raiden engine over just its sub-pool's layers, so a transfer issues
one all-layer raiden DMA per pool.

torch_tpu has no real device streams or events, so the CUDA
``HybridCacheController`` data path (side streams, per-layer load loop,
recorded events) is replaced the same way ``TpuRaidenHiCacheController``
replaces the single-pool path: poll-only future-events for acks, a
``BatchDoneCounter`` gating the whole forward instead of a per-layer
``LayerDoneCounter``, and index tensors kept host-resident throughout
(``SGLANG_TPU_REQ_TO_TOKEN_ON_HOST=1``).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu.raiden_transfer import (
    NOOP_EVENT,
    CombinedFutureEvent,
    RaidenFutureEvent,
)
from sglang.srt.managers.cache_controller import BatchDoneCounter, HiCacheAck
from sglang.srt.mem_cache.hicache_storage import PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    CacheOperation,
    HybridCacheController,
)

logger = logging.getLogger(__name__)


class TpuRaidenHybridCacheController(HybridCacheController):
    """Hybrid (multi-pool) HiCache controller whose transfers go through
    raiden DMA, one all-layer call per pool entry."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # raiden gates the whole forward at once (no per-layer device streams).
        # Swap the per-layer LayerDoneCounter for a BatchDoneCounter and
        # re-register it on the device pool wrapper (SWAKVPool holds the
        # counter itself; its sub-pools carry None), so CompiledForwardShim's
        # whole-batch ``wait()`` resolves against the reload event.
        self.layer_done_counter = BatchDoneCounter()
        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)

        # Same worker-thread H2D overlap as the single-pool raiden controller:
        # raiden's H2d releases the GIL while it blocks, and the
        # BatchDoneCounter gates the consuming forward on the worker event.
        # SGLANG_TPU_RAIDEN_ASYNC_H2D=0 selects the inline poll-to-complete
        # path.
        self._async_h2d = envs.SGLANG_TPU_RAIDEN_ASYNC_H2D.get()
        self._load_worker = None
        if self._async_h2d:
            from sglang.srt.hardware_backend.tpu.kv_copy_worker import TpuKvCopyWorker

            self._load_worker = TpuKvCopyWorker("raiden-hybrid-h2d")

    # ----- unsupported feature guards ---------------------------------------

    def set_draft_kv_pool(self, draft_device_pool, draft_host_pool):
        raise NotImplementedError(
            "Draft KV offload is not supported on the TPU raiden hybrid path."
        )

    def set_mtp_draft_pools(self, mtp_draft_device_pools):
        raise NotImplementedError(
            "MTP draft KV offload is not supported on the TPU raiden hybrid " "path."
        )

    # ----- lifecycle ---------------------------------------------------------

    def reset(self):
        # Drain outstanding D2H write-backs before the base clears the ack
        # queues and the cache frees host pages: a page raiden is still writing
        # must not return to the free list mid-DMA.
        for ack in list(self.ack_write_queue):
            ack.finish_event.synchronize()
        if self._load_worker is not None:
            self._load_worker.submit(lambda: None).synchronize()
        self.layer_done_counter.reset()
        super().reset()

    # ----- index handling -----------------------------------------------------

    def move_indices(self, host_indices, device_indices):
        # raiden addresses whole pages; the token->page conversion happens in
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

    def _issue_pool_dmas(
        self, op: CacheOperation, load: bool
    ) -> List[RaidenFutureEvent]:
        """One all-layer raiden DMA per pool entry of the merged operation.

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

        finish_event = CombinedFutureEvent(self._issue_pool_dmas(op, load=False))
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

        if self._async_h2d:
            # Issue every pool's H2d on the worker thread and return; the
            # forward's CompiledForwardShim wait() blocks on the worker event,
            # so the (GIL-releasing) DMAs overlap whatever the scheduler runs
            # before the consuming forward.
            def _load():
                for ev in self._issue_pool_dmas(op, load=True):
                    ev.synchronize()

            worker_done = self._load_worker.submit(_load)
            # BatchDoneCounter has 2 slots, so round N reuses round N-2's slot
            # and update_producer() synchronizes on it. Safe only because the
            # single FIFO worker cannot dequeue round N's _load until round
            # N-2's has returned. Parallelizing the worker or growing
            # num_counters would break this invariant.
            self.layer_done_counter.set_event(producer_id, worker_done)
            self.ack_load_queue.append(
                HiCacheAck(
                    start_event=worker_done,
                    finish_event=worker_done,
                    node_ids=op.node_ids,
                    num_tokens=len(op.device_indices),
                    num_tokens_by_pool=self._num_tokens_by_pool(op),
                    num_bytes=self._transfer_num_bytes(op),
                )
            )
            return producer_id

        # Inline path: all pools' H2d, then block until every DMA has landed
        # (poll-only; never Await). TPU's wait_event is a no-op and the
        # consuming forward runs this same step, so the load must complete
        # here.
        finish_event = CombinedFutureEvent(self._issue_pool_dmas(op, load=True))
        finish_event.synchronize()
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
