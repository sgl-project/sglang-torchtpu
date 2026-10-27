"""HiCache controller for TPU, transferring KV via a TPU DMA backend instead of CUDA.

``HiCacheController`` drives KV transfers on device streams and CUDA events,
neither of which exist on torch_tpu. This subclass overrides only the
transfer-issuing methods (``start_writing`` / ``start_loading`` /
``move_indices``) to route through the host pool's ``TpuDMAKVTransfer`` engine;
queues, ack draining, and write policy are inherited. Both directions are
asynchronous: the engine issues the copies off the scheduler thread and returns
a pollable future at once. The D2H
write-back future is polled by ``HiRadixCache.writing_check``; the H2D load-back
future goes into the whole-batch ``BatchDoneCounter``, which gates the consuming
forward on it, so the load overlaps whatever the scheduler runs in between.
"""

from __future__ import annotations

from sglang.srt.managers.cache_controller import (
    BatchDoneCounter,
    CacheOperation,
    HiCacheAck,
    HiCacheController,
)
from sglang_tpu.kv_free_fence import (
    assert_not_forward_write_target,
)
from sglang_tpu.tpu_dma_hicache import NOOP_EVENT


class TpuDMAHiCacheController(HiCacheController):
    """HiCache controller whose transfers go through a TPU DMA backend."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The engine gates the whole forward at once (no per-layer device streams),
        # like the torch_tpu path. Swap the base's per-layer LayerDoneCounter for
        # a BatchDoneCounter and re-register it, so CompiledForwardShim's
        # whole-batch ``wait()`` resolves against the reload event.
        self.layer_done_counter = BatchDoneCounter()
        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)

    def reset(self):
        # Drain outstanding DMAs before the base clears the ack queues and
        # HiRadixCache.reset() frees pages: a page the engine is still copying must
        # not return to a free list mid-DMA (silent corruption).
        for ack in list(self.ack_write_queue) + list(self.ack_load_queue):
            ack.finish_event.synchronize()
        self.layer_done_counter.reset()
        super().reset()

    def move_indices(self, host_indices, device_indices):
        # Engines address whole pages; the token->page conversion happens in
        # the host pool. Keep both index tensors as-is: no stream moves, no
        # ``.is_cuda`` branching, no host/device relocation.
        return host_indices, device_indices

    def load(self, host_indices, priority=None, node_id=-1):
        assert_not_forward_write_target(self.mem_pool_device_allocator)
        # Return the reloaded prefix's device-slot indices on CPU. The allocator
        # hands them back on device, but the slot integers are identical and
        # the engine addresses pages by int, so moving them to CPU keeps every
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

    def _backup_all_pools(self, host_indices, device_indices):
        """D2H the target pool in one engine call; returns an event over the copy."""
        return self.mem_pool_host.backup_from_device_all_layer(
            self.mem_pool_device, host_indices, device_indices, self.io_backend
        )

    def _load_all_pools(self, host_indices, device_indices):
        """H2D the target pool in one engine call; returns an event over the copy."""
        return self.mem_pool_host.load_all_layer(
            self.mem_pool_device, host_indices, device_indices
        )

    def start_writing(self) -> None:
        if len(self.write_queue) == 0:
            return

        op = CacheOperation.merge_ops(self.write_queue)
        host_indices, device_indices = self.move_indices(
            op.host_indices, op.device_indices
        )
        self.write_queue.clear()

        # D2H for all layers in one engine call per pool; the returned
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

        # H2D for all layers in one engine call per pool; the future-event comes
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
