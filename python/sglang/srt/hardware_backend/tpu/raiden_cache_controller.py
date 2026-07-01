"""HiCache controller for TPU, transferring KV via raiden instead of CUDA.

``HiCacheController`` drives KV transfers on device streams and CUDA events,
neither of which exist on torch_tpu. This subclass overrides only the
transfer-issuing methods (``start_writing`` / ``start_loading`` /
``move_indices``) to route through raiden's DMA; queues, ack draining, and write
policy are inherited. D2H write-back is async (polled by
``HiRadixCache.writing_check``). H2D load-back runs on a worker thread and the
whole-batch ``BatchDoneCounter`` gates the consuming forward on its completion,
so the load overlaps the prior batch's compute (raiden's H2D releases the GIL
while it blocks). ``RAIDEN_ASYNC_H2D=0`` selects the inline poll-to-complete
fallback, for a raiden build that does not release the GIL on H2d.
"""

from __future__ import annotations

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu.raiden_transfer import NOOP_EVENT
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

        # Run the blocking H2d load-back on a worker thread so it overlaps the
        # forward: raiden's H2d releases the GIL while it blocks, and the
        # BatchDoneCounter gates the consuming forward on the worker's completion.
        # SGLANG_TPU_RAIDEN_ASYNC_H2D=0 selects the inline poll-to-complete path
        # (for a raiden build that does not release the GIL on H2d).
        self._async_h2d = envs.SGLANG_TPU_RAIDEN_ASYNC_H2D.get()
        self._load_worker = None
        if self._async_h2d:
            from sglang.srt.hardware_backend.tpu.kv_copy_worker import TpuKvCopyWorker

            self._load_worker = TpuKvCopyWorker("raiden-h2d")

    def reset(self):
        # Drain outstanding D2H write-backs before the base clears the ack queue
        # and HiRadixCache.reset() frees host pages: a page raiden is still
        # writing must not return to the free list mid-DMA (silent corruption).
        for ack in list(self.ack_write_queue):
            ack.finish_event.synchronize()
        if self._load_worker is not None:
            self._load_worker.submit(lambda: None).synchronize()
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
        # host-resident req_to_token (SGLANG_TPU_HOST_R2T=1). A device tensor here
        # would torch.cat onto the CPU prefix indices as a cross-device cat.
        device_indices = self.mem_pool_device_allocator.alloc(len(host_indices))
        if device_indices is None:
            return None
        device_indices = device_indices.cpu()
        self.load_queue.append(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        return device_indices

    def start_writing(self) -> None:
        if len(self.write_queue) == 0:
            return
        if self.has_draft:
            raise NotImplementedError(
                "Draft KV offload is not supported on the TPU raiden path."
            )

        op = CacheOperation.merge_ops(self.write_queue)
        host_indices, device_indices = self.move_indices(
            op.host_indices, op.device_indices
        )
        self.write_queue.clear()

        # D2H for all layers in one raiden call; the returned future-event is
        # polled by HiRadixCache.writing_check before the device slot is freed.
        finish_event = self.mem_pool_host.backup_from_device_all_layer(
            self.mem_pool_device, host_indices, device_indices, self.io_backend
        )
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
        if self.has_draft:
            raise NotImplementedError(
                "Draft KV offload is not supported on the TPU raiden path."
            )

        producer_id = self.layer_done_counter.update_producer()
        op = CacheOperation.merge_ops(self.load_queue)
        host_indices, device_indices = self.move_indices(
            op.host_indices, op.device_indices
        )
        self.load_queue.clear()

        if self._async_h2d:
            # Issue the H2d on the worker thread and return immediately; the
            # forward's CompiledForwardShim wait() blocks on the worker event, so
            # the (GIL-releasing) H2d overlaps whatever the scheduler runs before
            # the consuming forward.
            mem_pool_host = self.mem_pool_host
            mem_pool_device = self.mem_pool_device

            def _load():
                mem_pool_host.load_all_layer(
                    mem_pool_device, host_indices, device_indices
                ).synchronize()

            worker_done = self._load_worker.submit(_load)
            # BatchDoneCounter has 2 slots, so round N reuses round N-2's slot and
            # update_producer() synchronizes on it. Safe only because the single
            # FIFO worker cannot dequeue round N's _load until round N-2's has
            # returned, so round N-2's event is already complete. Parallelizing the
            # worker or growing num_counters would break this invariant.
            self.layer_done_counter.set_event(producer_id, worker_done)
            self.ack_load_queue.append(
                HiCacheAck(
                    start_event=worker_done,
                    finish_event=worker_done,
                    node_ids=op.node_ids,
                )
            )
            return producer_id

        # Inline path: H2D for all layers in one raiden call, then block until the
        # DMA has landed (poll-only; never Await). TPU's wait_event is a no-op and
        # the consuming forward runs this same step, so the load must complete
        # here.
        finish_event = self.mem_pool_host.load_all_layer(
            self.mem_pool_device, host_indices, device_indices
        )
        finish_event.synchronize()

        # Hand the already-landed event to the whole-batch gate; the consuming
        # forward's CompiledForwardShim wait() then resolves immediately.
        self.layer_done_counter.set_event(producer_id, finish_event)
        self.ack_load_queue.append(
            HiCacheAck(
                start_event=NOOP_EVENT,
                finish_event=finish_event,
                node_ids=op.node_ids,
            )
        )
        return producer_id
