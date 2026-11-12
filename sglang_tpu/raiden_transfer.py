"""Raiden-backed TPU<->host KV transfer substrate for HiCache.

torch_tpu has no real streams and no pinned host memory, so the CUDA HiCache
transfer path can't be ported. This module drives tpu-sync's
``KVCacheManager``: ``D2h`` / ``H2d`` do direct PJRT raw-buffer DMA between the
fused device KV buffer and raiden's DMA-pinned host pool, returning a pollable
``RaidenFuture``. A raiden "block" is one page of the fused buffer, and a single
``D2h`` / ``H2d`` call covers all registered layers; raiden issues the copies
on its own background thread, so the call returns before anything is submitted.
Completion is poll-only (``is_ready()``; never ``Await()``, which can deadlock
the running model), and a DMA that completes with an error raises at the poll.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any, List, Literal, Optional

from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_rank
from sglang.srt.utils.network import get_free_port
from sglang_tpu.disaggregation.raiden.common import PORTS_PER_RANK
from sglang_tpu.envs import envs
from sglang_tpu.tpu_dma_hicache import NOOP_EVENT, page_major_tensors

if TYPE_CHECKING:
    from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool

logger = logging.getLogger(__name__)

HostTransferStatus = Literal["sent", "received", "failed"]


def _transfer_parallelism() -> int:
    """Streams per native transfer: the connector's setting, so a host pool
    opens the same connections per rank pair as the device-engine transfers."""
    return envs.SGLANG_TPU_RAIDEN_TRANSFER_PARALLELISM.get()


class RaidenFutureEvent:
    """Adapts a raiden ``RaidenFuture`` to the CUDA-event surface HiCache's
    ack queues use (``query`` / ``synchronize`` / ``record`` / ``wait``).

    Only ``query`` and ``synchronize`` carry meaning here; ``record`` / ``wait``
    are no-ops so the object can stand in for a ``device_module.Event`` in
    ``HiCacheAck`` without touching the hardware-agnostic drain logic in
    ``HiRadixCache.writing_check`` / ``loading_check``.
    """

    # Poll interval for the blocking flush path.
    _POLL_INTERVAL_S = 0.0005

    def __init__(self, future):
        self._future = future
        # A poll past this raises like a blocking wait: the copy's pages can be
        # neither reused nor released while it may still land.
        self._poll_deadline = (
            time.monotonic() + envs.SGLANG_TPU_RAIDEN_DMA_TIMEOUT_S.get()
        )

    def _raise_if_failed(self) -> None:
        """Surface a ready DMA's error; ``ok()`` is a non-blocking probe."""
        if not self._future.ok():
            raise RuntimeError(f"raiden DMA failed: {self._future.error_message()}")

    def query(self) -> bool:
        """Non-blocking completion poll. ``True`` once the DMA has landed."""
        if self._future is None:
            return True
        if not self._future.is_ready():
            if time.monotonic() > self._poll_deadline:
                raise TimeoutError(
                    "raiden DMA still pending "
                    f"{envs.SGLANG_TPU_RAIDEN_DMA_TIMEOUT_S.get()}s after issue"
                )
            return False
        self._raise_if_failed()
        return True

    def synchronize(self) -> None:
        """Block until complete by POLLING ``is_ready`` — never ``Await``.

        ``RaidenFuture.Await`` (``PJRT_Event_Await`` / ``xla::Future``
        ``BlockUntilReady``) can deadlock against the live model, so the flush
        primitive polls ``is_ready`` instead of awaiting. A DMA that never lands
        raises ``TimeoutError`` past the deadline rather than spinning forever.
        """
        if self._future is None:
            return
        timeout_s = envs.SGLANG_TPU_RAIDEN_DMA_TIMEOUT_S.get()
        deadline = time.monotonic() + timeout_s
        while not self._future.is_ready():
            if time.monotonic() > deadline:
                raise TimeoutError(f"raiden DMA did not complete within {timeout_s}s")
            time.sleep(self._POLL_INTERVAL_S)
        self._raise_if_failed()

    def record(self, *args, **kwargs) -> None:  # noqa: D401 - event-API no-op
        return None

    def wait(self, *args, **kwargs) -> None:  # noqa: D401 - event-API no-op
        return None


# Above every rank's Raiden KV port block on the node (PORTS_PER_RANK ports
# per worker, well under 64 workers).
HOST_POOL_CONTROL_PORT_OFFSET = PORTS_PER_RANK * 64
# A rank's full KV pool, and the sliding-window pool of a hybrid model.
HOST_POOLS_PER_RANK = 2
_host_pools_built = 0


def host_control_port_candidates():
    """Ports for a host pool's control server: a deterministic one per TP rank
    and host pool above the Raiden KV ports first, then free ephemeral ports.

    A port found free by ``get_free_port`` is bound only later, by the native
    constructor, and the kernel can hand it to an outgoing connection in
    between: two of fifteen engine starts in one day failed that way. The
    native server does not refuse a port another manager of this process
    already serves, so two pools offered the same port both bind it and a
    peer's reads reach either manager (seen live on a hybrid model: every
    push failed, "pull already started" or "block not registered").
    """
    global _host_pools_built
    index, _host_pools_built = _host_pools_built, _host_pools_built + 1
    if index < HOST_POOLS_PER_RANK:
        rank = get_tensor_model_parallel_rank()
        yield (
            envs.SGLANG_RAIDEN_KV_PORT.get()
            + HOST_POOL_CONTROL_PORT_OFFSET
            + rank * HOST_POOLS_PER_RANK
            + index
        )
    for _ in range(2):
        yield get_free_port()


class RaidenKVTransfer:
    """Owns a raiden ``KVCacheManager`` over a TPU pool's page-major tensors and
    issues page-granular D2H / H2D transfers.

    The device buffers are registered with raiden by raw PJRT pointer; the
    tensor list MUST stay alive for raiden's lifetime or ``D2h``/``H2d`` will
    dereference freed buffers (segfault). We retain it on ``self``.
    """

    def __init__(
        self,
        device_pool: TpuFusedMHATokenToKVPool,
        num_host_pages: int,
        draft_pools: tuple = (),
    ) -> None:
        from tpu_sync.api.torch.kv_cache_manager import KVCacheManager

        self.device_pool = device_pool
        self.num_host_pages = num_host_pages
        # Packed MTP draft pools ride the target's page ids, after its tensors.
        tensors = [
            t for pool in (device_pool, *draft_pools) for t in page_major_tensors(pool)
        ]
        if len({t.shape[0] for t in tensors}) != 1:
            raise ValueError(
                "Raiden HiCache needs every target and draft page tensor to hold "
                f"the same number of pages; got {sorted({t.shape[0] for t in tensors})}"
            )
        self.layer_num = len(tensors)

        # outer list = raiden "layer" (one per tensor), inner list = shard (one
        # per TPU core here). Keep-alive: the buffers must outlive raiden.
        self._device_tensors = [[t] for t in tensors]
        # raiden issues ``d2h`` / ``h2d`` copies on a background thread only
        # when this variable is set; otherwise the calls block the scheduler
        # thread until every per-layer copy has been submitted. The manager
        # reads it in its constructor.
        os.environ["RAIDEN_ENABLE_ASYNC_DISPATCH"] = "1"
        self._host_transfers: dict[str, HostTransferStatus | None] = {}

        def build(control_port: int):
            return KVCacheManager(
                self._device_tensors,
                local_control_port=control_port,
                host_blocks_to_allocate=num_host_pages,
                parallelism=_transfer_parallelism(),
                timeout_s=float(envs.SGLANG_RAIDEN_CHUNK_TRANSFER_TIMEOUT_S.get()),
                unsafe_skip_buffer_lock=True,
            )

        # The Python host-pool constructor treats zero as cache-only. Victim
        # transfers need a positive control port on the existing HiCache pool;
        # allocating another manager would serve different host memory.
        self._host_control_port = 0
        if not envs.SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM.get():
            self._mgr = build(0)
        else:
            tried: List[int] = []
            for port in host_control_port_candidates():
                try:
                    self._mgr = build(port)
                except RuntimeError as error:
                    # The native constructor binds the port; another socket may
                    # have taken it since it was picked.
                    if "Failed to bind" not in str(error):
                        raise
                    tried.append(port)
                    logger.warning("host pool control port %d is taken", port)
                    continue
                self._host_control_port = port
                break
            else:
                raise RuntimeError(
                    "no free port for the host pool's control server "
                    f"(tried {tried}); check what listens on them"
                )
        logger.info(
            "Raiden KV transfer ready: %d tensors, %d host pages, %.2f MiB per page",
            self.layer_num,
            num_host_pages,
            sum(t.nbytes // t.shape[0] for t in tensors) / 2**20,
        )

    def backup(
        self,
        device_page_ids: List[int],
        host_page_ids: List[int],
        num_pages: Optional[List[int]] = None,
    ) -> RaidenFutureEvent:
        """Device->host (D2H): stage device pages into raiden's host pool.

        Segment ``i`` copies ``num_pages[i]`` consecutive pages (default one)
        from ``device_page_ids[i]`` to ``host_page_ids[i]``.

        No device sync precedes the DMA: raiden's D2H reads HBM on a separate
        hardware path that is not ordered against TC compute. Under overlap
        the caller must not issue it while a forward that may write the source
        pages is in flight; the hybrid controller defers write-through behind
        the KV free fence for that.
        """
        n = len(device_page_ids)
        assert n == len(host_page_ids)
        # raiden reads an empty page list as the whole layer buffer.
        if n == 0:
            return NOOP_EVENT
        fut = self._mgr.d2h(device_page_ids, host_page_ids, num_pages or [1] * n)
        return RaidenFutureEvent(fut)

    def load(
        self,
        host_page_ids: List[int],
        device_page_ids: List[int],
        num_pages: Optional[List[int]] = None,
    ) -> RaidenFutureEvent:
        """Host->device (H2D): load host pages back into the fused KV buffer,
        segmented as in ``backup``."""
        n = len(host_page_ids)
        assert n == len(device_page_ids)
        if n == 0:
            return NOOP_EVENT
        fut = self._mgr.h2d(host_page_ids, device_page_ids, num_pages or [1] * n)
        return RaidenFutureEvent(fut)

    def host_control_address(self, local_ip: str) -> str:
        """Reachable pull-serve endpoint, distinct from the block data port."""
        host = f"[{local_ip}]" if ":" in local_ip else local_ip
        return f"{host}:{self._host_control_port}"

    def register_read(self, req_id: str, uuid: int, block_ids: list[int]) -> None:
        """Keep the host source pages pinned until ``poll_transfer`` reports
        terminal."""
        if req_id in self._host_transfers:
            raise ValueError(f"Raiden transfer request already tracked: {req_id}")
        self._mgr.register_read(req_id, uuid, list(block_ids), source="host")
        # Another request's poll may have cached this registration's report
        # while the native call released the GIL on the control listener.
        self._host_transfers.setdefault(req_id, None)

    def start_read(
        self,
        req_id: str,
        uuid: int,
        remote_endpoint: str | list[dict[str, Any]],
        remote_block_ids: list[int],
        local_block_ids: list[int],
    ) -> None:
        """Pull a peer's registered host pages into the host pages
        ``local_block_ids``; no H2D copy follows."""
        if req_id in self._host_transfers:
            raise ValueError(f"Raiden transfer request already tracked: {req_id}")
        self._mgr.start_read(
            req_id,
            uuid,
            remote_endpoint,
            list(remote_block_ids),
            list(local_block_ids),
            parallelism=_transfer_parallelism(),
            destination="host",
        )
        self._host_transfers.setdefault(req_id, None)

    def poll_transfer(self, req_id: str) -> HostTransferStatus | None:
        """One owner drains native reports and retains every request's result.

        Native terminal reports arrive only after session work has drained.
        A caller's own deadline never authorizes releasing its source pin or
        destination allocation while this method still returns None.
        """
        if self._host_transfers[req_id] is None:
            reports = self._mgr.poll_stats_with_details()
            statuses: tuple[HostTransferStatus, ...] = ("sent", "received", "failed")
            for status, transfers in zip(
                statuses, (reports.sent, reports.received, reports.failed)
            ):
                for transfer in transfers:
                    self._host_transfers[transfer.req_id] = status
                    if status == "failed":
                        logger.warning(
                            "Raiden host transfer %s failed (%s, %s): %s",
                            transfer.req_id,
                            transfer.code,
                            transfer.phase,
                            transfer.reason,
                        )
        return self._host_transfers[req_id]

    def forget_transfer(self, req_id: str) -> None:
        """Drop bookkeeping only after the native session has drained."""
        if self._host_transfers[req_id] is None:
            raise RuntimeError(f"Cannot forget active Raiden transfer: {req_id}")
        del self._host_transfers[req_id]
