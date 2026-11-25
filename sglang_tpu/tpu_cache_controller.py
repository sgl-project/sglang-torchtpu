# SPDX-License-Identifier: Apache-2.0
"""HiCache controller for the torch_tpu io backend (``--hicache-io-backend torch_tpu``).

Selected by ``TpuSRTPlatform.get_hicache_controller_cls``.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.managers.cache_controller import (
    BatchDoneCounter,
    CacheOperation,
    HiCacheAck,
    HiCacheController,
)
from sglang_tpu.kv_copy_worker import TpuKvCopyWorker


class TpuHiCacheController(HiCacheController):
    """torch_tpu data plane for HiCache.

    Swaps the device-stream machinery for a batch-granular BatchDoneCounter and
    per-direction TpuKvCopyWorker threads (torch_tpu has no per-layer device
    streams, and gates the whole forward at once). Only __init__ and the
    load/backup dispatch differ; everything else is inherited.

    Draft KV follows the same logistics: draft and target
    share device slot indices and the draft host pool is sized to the target's
    slot count, so each backup/reload reuses the target's index pair for a
    second gather/copy on the draft pools, covered by the same done-event.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Whole-batch gate instead of the per-layer device-stream counter; the
        # base already registered a LayerDoneCounter, so swap + re-register.
        self.layer_done_counter = BatchDoneCounter()
        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)
        # One copy worker per direction; they must not share a thread (a slow
        # D2H must not head-of-line block an H2D, and vice versa).
        self._load_worker = TpuKvCopyWorker("kv-offload-h2d")
        # Depth-1 async write-back (see start_writing).
        self._backup_worker = TpuKvCopyWorker("kv-offload-d2h")
        self._backup_inflight = None  # _ThreadDoneEvent | None

    def reset(self):
        if self._backup_inflight is not None:
            self._backup_inflight.synchronize()
            self._backup_inflight = None
        # FIFO workers: a submitted sentinel completes only after every prior
        # transfer task, so synchronizing on it drains the queue.
        self._load_worker.submit(lambda: None).synchronize()
        self._backup_worker.submit(lambda: None).synchronize()
        self.layer_done_counter.reset()
        super().reset()

    def evict_device(self, device_indices: torch.Tensor) -> int:
        """Eviction lifecycle: land all pending reload commits, then free.

        Ordering invariant (commit-before-free): a staged-but-uncommitted
        reload targeting these pages must scatter into the kv_buffers while
        the pages are still owned by their node; freeing first would let a
        later commit write stale bytes over whoever reallocates them.
        """
        self.layer_done_counter.drain_commits()
        # Upstream dropped HiCacheController.evict_device with HiRadixCache
        # (#40787); the TPU HiRadixCache still frees pages through it.
        self.mem_pool_device_allocator.free(device_indices)
        return len(device_indices)

    def start_writing(self) -> None:
        if len(self.write_queue) == 0:
            return

        op = CacheOperation.merge_ops(self.write_queue)
        host_indices, device_indices = self.move_indices(
            op.host_indices, op.device_indices
        )
        self.write_queue.clear()

        # Async write-back (depth-1): gather inline (reads KV before the next
        # forward donates it), drain the slow D2H on the worker thread. Block on
        # the previous backup first to bound in-flight gathered HBM.
        if self._backup_inflight is not None:
            self._backup_inflight.synchronize()

        # Write-back lifecycle: land pending reload commits, then gather.
        # Ordering invariant (commit-before-read): backing up a just-reloaded
        # node must gather post-reload bytes, so its scatter lands first.
        self.layer_done_counter.drain_commits()

        staged = self.mem_pool_host.backup_gather(self.mem_pool_device, device_indices)

        mem_pool_host = self.mem_pool_host

        def _drain():
            if staged is not None:
                mem_pool_host.backup_copy_d2h(staged, host_indices)

        done = self._backup_worker.submit(_drain)
        self._backup_inflight = done
        # Real event: writing_check must see the D2H land before unlocking.
        self.ack_write_queue.append(HiCacheAck(done, done, op.node_ids))

    def start_loading(self) -> int:
        if len(self.load_queue) == 0:
            return -1

        # A dedicated producer thread does the H2D copy + scatter for all layers,
        # then signals the round done; CompiledForwardShim gates the forward on
        # it (no per-layer device events on torch_tpu).
        op = CacheOperation.merge_ops(self.load_queue)
        host_indices, device_indices = self.move_indices(
            op.host_indices, op.device_indices
        )
        self.load_queue.clear()

        producer_id = self.layer_done_counter.update_producer()

        mem_pool_host = self.mem_pool_host
        mem_pool_device = self.mem_pool_device

        # Reload lifecycle: the worker thread STAGES (host gather + H2D into
        # fresh tensors); the scatter into live kv_buffers is deferred as a
        # commit closure that BatchDoneCounter runs at its gate points on the
        # scheduler thread. Donation safety: worker threads never touch the
        # (donatable) kv_buffer handles — only the dispatch thread commits.
        # Values are load_stage's (src_all staging tensor, padded dst page ids).
        staged: dict[str, Optional[tuple[torch.Tensor, torch.Tensor]]] = {}

        def _producer():
            staged["target"] = mem_pool_host.load_stage(
                mem_pool_device, host_indices, device_indices
            )

        def _commit():
            mem_pool_host.load_commit(mem_pool_device, staged.get("target"))

        # Worker event = staging enqueued (gather done, async H2D issued); the
        # commit scatter's operand dependency covers the DMA landing.
        worker_done = self._load_worker.submit(_producer)
        self.layer_done_counter.set_event(producer_id, worker_done)
        self.layer_done_counter.add_commit(worker_done, _commit)
        self.ack_load_queue.append(
            HiCacheAck(
                start_event=worker_done,
                finish_event=worker_done,
                node_ids=op.node_ids,
            )
        )
        return producer_id
