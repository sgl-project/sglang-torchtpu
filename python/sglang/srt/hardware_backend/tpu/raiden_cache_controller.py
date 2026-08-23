"""HiCache controller for TPU, transferring KV via raiden instead of CUDA.

``HiCacheController`` drives KV transfers on device streams and CUDA events,
neither of which exist on torch_tpu. This subclass overrides only the
transfer-issuing methods (``start_writing`` / ``start_loading`` /
``move_indices``) to route through raiden's DMA; queues, ack draining, and write
policy are inherited. Both directions are asynchronous: raiden issues the copies
on its own background thread and returns a pollable future at once. The D2H
write-back future is polled by ``HiRadixCache.writing_check``; the H2D load-back
future goes into the whole-batch ``BatchDoneCounter``, which gates the consuming
forward on it, so the load overlaps whatever the scheduler runs in between.
"""

from __future__ import annotations

from sglang.srt.hardware_backend.tpu.raiden_transfer import (
    NOOP_EVENT,
    CombinedFutureEvent,
)
from sglang.srt.managers.cache_controller import (
    BatchDoneCounter,
    CacheOperation,
    HiCacheAck,
    HiCacheController,
)


class TpuRaidenHiCacheController(HiCacheController):
    """HiCache controller whose transfers go through raiden DMA."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # raiden gates the whole forward at once (no per-layer device streams),
        # like the torch_tpu path. Swap the base's per-layer LayerDoneCounter for
        # a BatchDoneCounter and re-register it, so CompiledForwardShim's
        # whole-batch ``wait()`` resolves against the reload event.
        self.layer_done_counter = BatchDoneCounter()
        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)

    def reset(self):
        # Drain outstanding DMAs before the base clears the ack queues and
        # HiRadixCache.reset() frees pages: a page raiden is still copying must
        # not return to a free list mid-DMA (silent corruption).
        for ack in list(self.ack_write_queue) + list(self.ack_load_queue):
            ack.finish_event.synchronize()
        self.layer_done_counter.reset()
        super().reset()

    def move_indices(self, host_indices, device_indices):
        # raiden addresses whole pages; the token->page conversion happens in
        # the host pool. Keep both index tensors as-is: no stream moves, no
        # ``.is_cuda`` branching, no host/device relocation.
        return host_indices, device_indices

    def load(self, host_indices, priority=None, node_id=-1):
        # Return the reloaded prefix's device-slot indices on CPU. The allocator
        # hands them back on device, but the slot integers are identical and
        # raiden's DMA addresses pages by int, so moving them to CPU keeps every
        # index tensor host-resident: node.value / prefix_indices stay on CPU and
        # the schedule_policy prefix cat stays single-device under the
        # host-resident req_to_token (SGLANG_TPU_REQ_TO_TOKEN_ON_HOST=1). A device tensor here
        # would torch.cat onto the CPU prefix indices as a cross-device cat.
        device_indices = self.mem_pool_device_allocator.alloc(len(host_indices))
        if device_indices is None:
            return None
        device_indices = device_indices.cpu()
        self.load_queue.append(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        return device_indices

    def set_draft_kv_pool(self, draft_device_pool, draft_host_pool) -> None:
        # Spec decoding piggybacks draft KV on every write-back / load-back
        # with the SAME host and device indices as the target pool (the draft
        # pool shares the target's token allocator, and the draft host pool is
        # sized for 1-to-1 host slots). raiden addresses pages, so that only
        # holds if the draft host pool pages exactly like the primary one.
        primary = self.mem_pool_host
        if type(draft_host_pool) is not type(primary):
            raise TypeError(
                "Draft host pool must be the raiden host pool "
                f"({type(primary).__name__}), got {type(draft_host_pool).__name__}."
            )
        if draft_host_pool.page_size != primary.page_size:
            raise ValueError(
                "Draft host pool page size must match the primary host pool "
                f"({draft_host_pool.page_size} != {primary.page_size})."
            )
        if draft_host_pool.page_num < primary.page_num:
            raise ValueError(
                "Draft host pool must cover every primary host page id "
                f"({draft_host_pool.page_num} < {primary.page_num})."
            )
        super().set_draft_kv_pool(draft_device_pool, draft_host_pool)
        # The draft forward gates on the same whole-batch reload event through
        # its own pool's counter (mirrors the native TpuHiCacheController).
        draft_device_pool.register_layer_transfer_counter(self.layer_done_counter)

    def _backup_all_pools(self, host_indices, device_indices):
        """D2H the target pool — and the draft pool at the same indices — in
        one raiden call each; returns a single event over the DMAs."""
        target_event = self.mem_pool_host.backup_from_device_all_layer(
            self.mem_pool_device, host_indices, device_indices, self.io_backend
        )
        if not self.has_draft:
            return target_event
        draft_event = self.mem_pool_host_draft.backup_from_device_all_layer(
            self.mem_pool_device_draft, host_indices, device_indices, self.io_backend
        )
        return CombinedFutureEvent([target_event, draft_event])

    def _load_all_pools(self, host_indices, device_indices):
        """H2D the target pool — and the draft pool at the same indices — in
        one raiden call each; returns a single event over the DMAs."""
        target_event = self.mem_pool_host.load_all_layer(
            self.mem_pool_device, host_indices, device_indices
        )
        if not self.has_draft:
            return target_event
        draft_event = self.mem_pool_host_draft.load_all_layer(
            self.mem_pool_device_draft, host_indices, device_indices
        )
        return CombinedFutureEvent([target_event, draft_event])

    def start_writing(self) -> None:
        if len(self.write_queue) == 0:
            return

        op = CacheOperation.merge_ops(self.write_queue)
        host_indices, device_indices = self.move_indices(
            op.host_indices, op.device_indices
        )
        self.write_queue.clear()

        # D2H for all layers in one raiden call per pool; the returned
        # future-event is polled by HiRadixCache.writing_check before the
        # device slot is freed.
        finish_event = self._backup_all_pools(host_indices, device_indices)
        self.ack_write_queue.append(
            HiCacheAck(
                start_event=NOOP_EVENT,
                finish_event=finish_event,
                node_ids=op.node_ids,
            )
        )

    def start_loading(self) -> int:
        if len(self.load_queue) == 0:
            return -1

        producer_id = self.layer_done_counter.update_producer()
        op = CacheOperation.merge_ops(self.load_queue)
        host_indices, device_indices = self.move_indices(
            op.host_indices, op.device_indices
        )
        self.load_queue.clear()

        # H2D for all layers in one raiden call per pool; the future-event comes
        # back before the DMA lands. The consuming forward's CompiledForwardShim
        # wait() polls it through the whole-batch gate (poll-only; never Await),
        # so the DMA overlaps whatever the scheduler runs before that forward.
        finish_event = self._load_all_pools(host_indices, device_indices)
        self.layer_done_counter.set_event(producer_id, finish_event)
        self.ack_load_queue.append(
            HiCacheAck(
                start_event=NOOP_EVENT,
                finish_event=finish_event,
                node_ids=op.node_ids,
            )
        )
        return producer_id
