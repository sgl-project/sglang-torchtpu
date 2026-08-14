"""``tpu`` KV transfer backend for P/D disaggregation (host ZMQ-push).

This is the **default** TPU backend. It reuses the device-agnostic control
plane (bootstrap discovery, ZMQ coordination, the ``KVPoll`` state machine)
captured in ``base.py`` and plugs in a host *push* data plane:

    decode  --TRSFR--> prefill        (small control msg: "room R ready,
                                        send my pages, here is my sink addr")
    prefill --KVDATA-> decode         (raw KV page bytes + aux row)

The prefill side gathers its KV pages off-device (D2H), pushes the bytes to
the decode manager's ``server_socket``, and the decode side scatters them
H2D into pages it allocated itself. No device pointers ever cross the
process boundary — see ``transfer.py`` for why that matters on TPU.

For the backend-agnostic control-plane scaffolding and the seam a second
backend (e.g. ``tpu_raiden``) plugs into, see ``base.py``.

Threading model (critical on TPU)
---------------------------------
torch_tpu serialises all device work on a single implicit PjRt queue, and
the Pallas RPA v3 attention kernel *donates* the fused ``kv_buffer`` to each
forward. Touching that buffer from another thread while a forward is in
flight raises ``Buffer has been deleted or donated`` and kills the worker.
So every *device* KV op runs on the scheduler's **main thread**, serialised
with the forward loop:

* Prefill D2H gather runs inline in ``TpuKVSender.send`` (called from the
  scheduler's ``send_kv_chunk`` after the prefill forward). Only the
  resulting host bytes are handed to a background thread for the ZMQ push.
* Decode H2D scatter runs inline in ``TpuKVReceiver.poll`` (called from the
  scheduler's disagg-decode polling phase, between forwards). The background
  ZMQ thread only *receives* bytes into host memory and stashes them.

Supports MHA resharding across heterogeneous attention TP:

1. fan-in
2. fan-out
3. GQA replication
4. FP8 odd-head (1 head) resharding via host PendingChunk reassembly

Otherwise scoped to the common single-/multi-host MHA case: no MLA, PP, CP,
DP-attention, or mamba/SWA state.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import logging
import os
import threading
import time
from collections import defaultdict, deque
from typing import Any

import numpy as np
import numpy.typing as npt
import zmq

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.tpu import transfer as tpu_transfer

# Re-exported from base so existing imports (and a future raiden backend)
# keep resolving ``from ...tpu.conn import KVTransferError, _tcp_endpoint``.
from sglang.srt.disaggregation.tpu.base import (  # noqa: F401
    KVTransferError,
    MessageTag,
    TpuKVBootstrapServerBase,
    TpuKVManagerBase,
    TpuKVReceiverBase,
    TpuKVSenderBase,
    _tcp_endpoint,
)
from sglang.srt.disaggregation.tpu.host_staging import (
    PinnedHostPool,
    StageBuf,
    default_pool_max_bytes,
)
from sglang.srt.disaggregation.tpu.transfer import (
    DecodeTransferState,
    PendingChunk,
    PrefillTransferInfo,
)

logger = logging.getLogger(__name__)

# prefill -> decode, one frame per parallel channel (#3, opt-in). Each carries a
# contiguous slice of the layers; the decode reassembles across channels.
# MessageTag.KVDATA_CH is re-exported from base.py.

# Background threads per prefill rank for the (host-only) ZMQ push — i.e. how
# many distinct transfers (rooms/chunks) a rank pushes concurrently. Orthogonal
# to SGLANG_TPU_KV_CHANNELS (parallel streams *within* one transfer); the
# effective stream count is this x channels. See _push / _push_channels.
_PUSH_POOL_WORKERS = 4


def _kv_stats_every() -> int:
    """How many transfers between logged latency summaries (0 disables).
    Set via ``SGLANG_DEBUG_TPU_KV_STATS_EVERY`` (debug/observability only)."""
    try:
        return int(os.environ.get("SGLANG_DEBUG_TPU_KV_STATS_EVERY", "100"))
    except (TypeError, ValueError):
        return 100


def _kv_async() -> bool:
    """issue the prefill D2H asynchronously. Set ``SGLANG_TPU_KV_ASYNC=0`` to fall back to the
    synchronous batched D2H. See gather_pages_to_stage_async for the
    donation-safety reasoning."""
    return os.environ.get("SGLANG_TPU_KV_ASYNC", "1").lower() not in ("0", "", "false")


def _kv_channels() -> int:
    """number of parallel PUSH streams the prefill shards layer frames over
    (default **2**). Each TP rank already pushes its own shard on its own
    socket, so the effective stream count is ``tp_size * channels`` — hence a
    small default rather than torchtpu-vllm's per-rank ``tp_size``. Set
    ``SGLANG_TPU_KV_CHANNELS=1`` for the single-stream path."""
    try:
        return max(1, int(os.environ.get("SGLANG_TPU_KV_CHANNELS", "2")))
    except (TypeError, ValueError):
        return 2


@dataclasses.dataclass
class _KvStats:
    """Cumulative per-stage transfer instrumentation.

    Attributes:
      stage: Transfer stage name (e.g., 'gather' D2H or 'scatter' H2D).
      every: Logging interval in transfer completions.
      n: Total completions recorded.
      secs: Cumulative wall time in seconds.
      nbytes: Cumulative bytes transferred.
    """

    stage: str
    every: int
    n: int = 0
    secs: float = 0.0
    nbytes: int = 0

    def record(self, secs: float, nbytes: int) -> None:
        self.n += 1
        self.secs += secs
        self.nbytes += nbytes
        if self.every and self.n % self.every == 0:
            gbps = (self.nbytes * 8.0 / 1e9) / self.secs if self.secs else 0.0
            logger.info(
                "tpu-kv %s: n=%d avg=%.2fms %.2fGbps (%.1fMB cumulative)",
                self.stage,
                self.n,
                1e3 * self.secs / self.n,
                gbps,
                self.nbytes / 1e6,
            )


class TpuKVManager(TpuKVManagerBase):
    BACKEND_NAME = "tpu"

    def _stage_pool_budget(self) -> int:
        """Pinned-staging budget for this rank's pool. The host runs one pool
        per co-located TP rank, so split the host-RAM fraction across them so
        the aggregate stays within budget (not tp_size x it)."""
        sa = getattr(self, "server_args", None)
        tp = getattr(sa, "tp_size", 1) or 1
        nnodes = getattr(sa, "nnodes", 1) or 1
        local_ranks = max(1, tp // max(1, nnodes))  # TP ranks on this host
        return default_pool_max_bytes(local_shares=local_ranks)

    # ---- role init (called from TpuKVManagerBase.__init__) --------------

    def _init_prefill(self) -> None:
        # The pool here is only used for the (host-only) ZMQ push.
        self._push_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=_PUSH_POOL_WORKERS, thread_name_prefix="tpu-kv-push"
        )
        self._gather_stats = _KvStats("gather", _kv_stats_every())
        # Pinned/aligned reusable host staging for zero-copy D2H + send (#1/#4).
        self._stage_pool = PinnedHostPool(max_bytes=self._stage_pool_budget())
        self._async_d2h = _kv_async()  # #7 (opt-in)
        self._n_channels = _kv_channels()  # #3 (opt-in)
        # Per-(endpoint, channel) PUSH sockets for the parallel path (#3).
        self._chan_socks: dict[str, list] = {}
        self._chan_socks_lock = threading.Lock()
        # Per-room push progress: {room: {"drained": int, "total": int}}.
        # Chunk pushes run on a pool (_PUSH_POOL_WORKERS) and can drain out of
        # order, so conclude Success by drained page count reaching the room
        # total (not by an is_last push), mirroring the decode. See _note_push_*.
        self._push_progress: dict[int, dict] = {}
        self._push_progress_lock = threading.Lock()
        self._serve(MessageTag.TRSFR, self._handle_transfer)

    def _init_decode(self) -> None:
        self.decode_states: dict[int, DecodeTransferState] = {}
        self.decode_states_lock = threading.Lock()
        # room -> queue of received-but-not-yet-scattered chunks. The bg ZMQ
        # thread appends; the main thread (poll) drains + scatters.
        self.decode_pending: dict[int, deque[PendingChunk]] = defaultdict(deque)
        self.reasm_pending: dict[int, dict[tuple, list[PendingChunk]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.decode_pending_lock = threading.Lock()
        self._scatter_stats = _KvStats("scatter", _kv_stats_every())
        self._assemble_stats = _KvStats("assemble", _kv_stats_every())
        self._push_to_recv_stats = _KvStats("push_to_recv", _kv_stats_every())
        # Pinned/aligned reusable host staging for pooled H2D (#1).
        self._stage_pool = PinnedHostPool(max_bytes=self._stage_pool_budget())
        # Multi-channel reassembly buffer (#3), keyed by (room, index_start).
        self._chan_reasm: dict = {}
        self._chan_reasm_lock = threading.Lock()
        # Accept both the single-message (TAG_KVDATA) and per-channel
        # (TAG_KVDATA_CH) formats on the one PULL socket. copy=False so the recv
        # thread can copy each layer frame *once*, straight into pinned staging
        # (see _recv_layers), rather than wire -> bytes -> staging.
        self._serve_router(
            {
                MessageTag.KVDATA_CH: self._handle_kvdata_ch,
            },
            copy=False,
        )

    def _on_pool_attached(self) -> None:
        # Fused RPA v3 layer buffer: [num_blocks, page_size, kv_heads_packed,
        # kv_packing, head_dim]. Cache the packed-lane count (dim 2, == this
        # rank's full KV-head lanes) and kv_packing (dim 3) for MHA resharding
        # slice math. Fall back gracefully for non-5D CPU test pools.
        buf0 = self.layer_buffers[0] if self.layer_buffers else None
        if buf0 is not None and buf0.dim() >= 4:
            self._num_lanes_full = int(buf0.shape[2])
            self._kv_packing = int(buf0.shape[3])
        else:
            self._num_lanes_full = 1
            self._kv_packing = 1
        logger.info(
            "TpuKVManager attached KV pool %s: %d layer buffers, page tail %s, "
            "num_lanes_full=%d kv_packing=%d",
            type(self.kv_pool).__name__,
            len(self.layer_buffers),
            tuple(self.layer_buffers[0].shape[1:]) if self.layer_buffers else None,
            self._num_lanes_full,
            self._kv_packing,
        )

    def _note_push_submitted(self, room: int, total_units: int) -> None:
        """Main thread: register the room's push-progress entry with the total lanes."""
        with self._push_progress_lock:
            if room not in self._push_progress:
                self._push_progress[room] = {"drained": 0, "total": total_units}

    def _note_push_done(self, room: int, push_lane_units: int) -> None:
        """Worker thread: a chunk push of ``n_units`` lane-page units drained
        (``chunk_pages * chunk_lanes``). Conclude Success once the drained units
        reach the room total — the same known-total accounting as the decode's
        scatter side, so completion needs no ``is_last`` marker and is immune to
        pushes draining out of order."""
        with self._push_progress_lock:
            ent = self._push_progress.get(room)
            if ent is None:
                return  # already failed/cleared
            ent["drained"] += push_lane_units
            done = ent["drained"] >= ent["total"]
            if done:
                self._push_progress.pop(room, None)
        if done:
            logger.info("[pd_kv_resharding] prefill room=%s all pushes drained", room)
            self.update_status(room, KVPoll.Success)

    def _note_push_failed(self, room: int) -> None:
        """A chunk push failed: drop the progress entry so a later-draining
        sibling push can't then conclude Success over the recorded failure."""
        with self._push_progress_lock:
            self._push_progress.pop(room, None)

    def _gather_lanes(
        self, page_indices: np.ndarray, lane_slice: tuple[int, int] | None
    ) -> tuple[list[memoryview | bytes], list[StageBuf] | None, Any | None, int]:
        """Gathers lanes by lane slice and page indices, D2H."""
        pool = getattr(self, "_stage_pool", None)
        use_stage = (
            pool is not None
            and self.layer_buffers
            and self.layer_buffers[0].device.type != "cpu"
        )
        d2h_future = None
        if use_stage:
            if getattr(self, "_async_d2h", False):
                d2h_future, stages = tpu_transfer.gather_pages_to_stage_async(
                    self.layer_buffers, page_indices, pool, lane_slice=lane_slice
                )
            else:
                stages = tpu_transfer.gather_pages_to_stage(
                    self.layer_buffers, page_indices, pool, lane_slice=lane_slice
                )
            layer_frames = [st.mv() for st in stages]
            nbytes = sum(st.nbytes for st in stages)
        else:
            stages = None
            layer_frames = tpu_transfer.gather_pages_to_host(
                self.layer_buffers, page_indices, lane_slice=lane_slice
            )
            nbytes = sum(len(b) for b in layer_frames)
        return layer_frames, stages, d2h_future, nbytes

    def gather_and_push(
        self,
        room: int,
        page_indices: np.ndarray,
        index_start: int,
        is_last_chunk: bool,
        aux_index: int | None,
    ) -> None:
        """Gathers pages and push."""
        logger.info("[pd_kv_resharding] prefill room=%s start to gather and push", room)
        if self.request_status.get(room) == KVPoll.Failed:
            return
        self._assert_layer_page_buffers()
        try:
            bucket: dict[str, PrefillTransferInfo] = self.transfer_infos.get(room)
            if not bucket:
                raise KVTransferError(room, "no decode endpoint registered")

            page_indices = np.asarray(page_indices, dtype=np.int64)
            n_pages = len(page_indices)
            plans: list[PrefillTransferInfo] = []
            total_lanes = 0
            for info in bucket.values():
                if not info.if_skip_sending:
                    plans.append(info)
                    total_lanes += len(page_indices) * info.n_lane_to_send
            logger.info("[pd_kv_resharding] prefill room=%s plans: %s", room, plans)
            # If there is no plans to push for the prefill TP rank. Exit early.
            if not plans:
                self._note_push_done(room, 0)
            self._note_push_submitted(room, total_lanes)
            aux_blobs_all: list[bytes] = []
            if is_last_chunk and aux_index is not None:
                aux_blobs_all = tpu_transfer.read_aux_row(self.kv_args, aux_index)
            n_ch = self._n_channels
            stats = self._gather_stats
            # FIXME: Gather full lanes once to host and slice on host to reduce D2H calls.
            for info in plans:
                n_lane_total = info.n_lane_to_send * n_pages
                lane_slice = (
                    (info.src_lane_idx, info.n_lane_to_send)
                    if info.require_slicing
                    else None
                )
                _t0 = time.perf_counter()
                logger.info(
                    "[pd_kv_resharding] starting to gather lanes: %s", str(lane_slice)
                )
                layer_frames, stages, d2h_future, nbytes = self._gather_lanes(
                    page_indices, lane_slice
                )
                logger.info(
                    "[pd_kv_resharding] finish gather lanes: %s", str(lane_slice)
                )
                gather_secs = time.perf_counter() - _t0
                info.gather_time_ms = gather_secs * 1000.0
                if stats is not None:
                    stats.record(gather_secs, nbytes)
                aux_blobs = aux_blobs_all if info.if_send_aux_data else []
                self._push_pool.submit(
                    self._push_channels,
                    info,
                    index_start,
                    n_pages,
                    n_lane_total,
                    aux_blobs,
                    layer_frames,
                    stages,
                    d2h_future,
                    n_ch,
                )

        except Exception as e:
            self._note_push_failed(room)
            if isinstance(e, tpu_transfer.ReshardUnsupportedError):
                # Log error for the odd head for fp8 resharding.
                logger.error(
                    "[pd_kv_resharding] room %s unsupported reshard: %s", room, e
                )
            else:
                logger.exception("KV gather failed for room %s", room)
            self.record_failure(room, f"TPU KV gather failed: {e}")
            self.update_status(room, KVPoll.Failed)

    def _push(
        self,
        room: int,
        endpoint: str,
        index_start: int,
        payload: list[Any],
        n_pages: int,
        units: int,
        stages: list[StageBuf] | None = None,
        d2h_future: Any | None = None,
    ) -> None:
        """Background thread: host-only ZMQ send of the gathered KV bytes.

        With pooled staging (``stages`` set) the layer frames are the pinned
        buffers' memoryviews, sent **zero-copy** (``copy=False``); the buffers
        are returned to the pool only after the send has drained (tracked), so
        a reused slot can never clobber an in-flight send. When the D2H was
        issued async (#7), ``d2h_future`` is waited here — off the scheduler
        thread — before the bytes are read for sending."""
        try:
            if d2h_future is not None:
                d2h_future.wait()
            sock = self._connect(endpoint)
            with self._endpoint_locks[endpoint]:
                if stages is not None:
                    tracker = sock.send_multipart(payload, copy=False, track=True)
                else:
                    tracker = None
                    sock.send_multipart(payload)
            if tracker is not None:
                tracker.wait()
            if stages is not None:
                for st in stages:
                    st.release()
            # Count this chunk's pages toward the room total; Success fires once
            # every page has drained, regardless of push completion order.
            self._note_push_done(room, units)
        except Exception as e:
            # On failure don't recycle the buffers: zmq (copy=False) may still
            # hold them until its send drains — let GC reclaim them instead.
            logger.exception("KV push failed for room %s", room)
            self._note_push_failed(room)
            self.record_failure(room, f"TPU KV push failed: {e}")
            self.update_status(room, KVPoll.Failed)

    def _channel_sockets(self, endpoint: str, n: int) -> list[zmq.Socket]:
        """N cached PUSH sockets to ``endpoint`` (#3) — N parallel TCP streams
        into the decode's single PULL socket. Grows the cache in place so
        already-connected channels are reused, never torn down and reconnected
        (the old code rebuilt all N and leaked the previous sockets)."""
        with self._chan_socks_lock:
            socks = self._chan_socks.setdefault(endpoint, [])
            while len(socks) < n:
                socks.append(self._connect(endpoint))
            return socks

    def _push_channels(
        self,
        info: PrefillTransferInfo,
        index_start: int,
        num_pages: int,
        units: int,
        aux_blobs: list[bytes],
        layer_frames: list[memoryview | bytes],
        stages: list[StageBuf] | None,
        d2h_future: Any | None,
        n_ch: int,
    ) -> None:
        """Background thread (#3): send one endpoint's chunk as ``n_ch`` messages
        over ``n_ch`` parallel sockets, each carrying a contiguous layer slice.
        Aux rides channel 0. Frame: ``[TAG_KVDATA_CH, room, index_start,
        num_pages, dst_lane_start, num_lanes, ch, n_ch, layer_lo, num_layers,
        num_aux] + aux + layer_slice``."""

        n_layers = len(layer_frames)
        try:
            if d2h_future is not None:
                _wait_t0 = time.perf_counter()
                d2h_future.wait()
                info.gather_time_ms += (time.perf_counter() - _wait_t0) * 1000.0

            push_t0 = time.perf_counter()
            info.push_start_time = time.time()
            logger.info(
                "[pd_kv_resharding] start push channels to endpoint=%s (room=%s, num_pages=%d, n_lanes=%d)",
                info.decode_endpoint,
                info.room,
                num_pages,
                info.n_lane_to_send,
            )
            n_layers = len(layer_frames)
            bounds = [(c * n_layers) // n_ch for c in range(n_ch + 1)]
            copy_false = stages is not None
            trackers = []
            socks = self._channel_sockets(info.decode_endpoint, n_ch)
            for ch in range(n_ch):
                lo, hi = bounds[ch], bounds[ch + 1]
                aux = aux_blobs if ch == 0 else []
                msg = (
                    [
                        MessageTag.KVDATA_CH,
                        # Add almost 0 latency for better
                        # readability/extensibility.
                        info.to_json().encode(),
                        str(index_start).encode(),
                        str(num_pages).encode(),
                        str(ch).encode(),
                        str(n_ch).encode(),
                        str(lo).encode(),
                        str(n_layers).encode(),
                        str(len(aux)).encode(),
                    ]
                    + aux
                    + layer_frames[lo:hi]
                )
                sock = socks[ch]
                with self._endpoint_locks[(info.decode_endpoint, ch)]:
                    if copy_false:
                        trackers.append(
                            sock.send_multipart(msg, copy=False, track=True)
                        )
                    else:
                        sock.send_multipart(msg)
            for t in trackers:
                t.wait()
            info.push_time_ms = (time.perf_counter() - push_t0) * 1000.0
            logger.info(
                "[pd_kv_resharding] finish push channels to endpoint=%s (room=%s, push_time_ms=%.2f)",
                info.decode_endpoint,
                info.room,
                info.push_time_ms,
            )
            if stages is not None:
                for st in stages:
                    st.release()
            # See _push: count this chunk's lane-page units toward the room total.
            self._note_push_done(info.room, units)
        except Exception as e:
            logger.exception("KV channel push failed for room %s", info.room)
            self._note_push_failed(info.room)
            self.record_failure(info.room, f"TPU KV channel push failed: {e}")
            self.update_status(info.room, KVPoll.Failed)

    def _recv_layers(self, frames: list[zmq.Frame | bytes]) -> list[StageBuf | bytes]:
        """Background recv thread: turn the wire's per-layer frames into the
        chunk's ``layer_blobs``, mirroring torchtpu-vllm's "unpack directly into
        the staging slot". On the TPU path each frame's bytes are copied **once**
        — straight from ZMQ's buffer into a pooled pinned+aligned ``StageBuf``
        ready for H2D — instead of the old wire -> ``bytes`` -> ``StageBuf`` two
        copies. The StageBufs are held until the main-thread scatter releases
        them (the occupancy cost of dropping the extra copy). On the CPU
        fallback path (no pool / cpu buffers / no batch_transfer) we materialise
        plain ``bytes`` for ``scatter_pages_from_host``.

        ``frames`` may be ``zmq.Frame``s (copy=False recv) or ``bytes`` (tests /
        copy=True) — both are handled."""
        pool = getattr(self, "_stage_pool", None)
        if (
            pool is not None
            and self.layer_buffers
            and self.layer_buffers[0].device.type != "cpu"
        ):
            stages = []
            for f in frames:
                mv = f.buffer if isinstance(f, zmq.Frame) else memoryview(f)
                st = pool.acquire(mv.nbytes)
                st.copy_bytes(mv)  # the one and only host copy: wire -> staging
                stages.append(st)
            return stages
        # CPU fallback: keep raw bytes (scatter_pages_from_host re-aligns).
        return [bytes(f) for f in frames]

    @staticmethod
    def _release_staged(chunk: PendingChunk) -> None:
        """Return any pooled StageBufs held by a chunk that won't be scattered
        (failure / teardown paths). No-op for the bytes (CPU) path."""
        for b in chunk.layer_blobs:
            if isinstance(b, StageBuf):
                b.release()

    def _handle_kvdata_ch(self, msg: list[zmq.Frame]) -> None:
        """Background thread (#3): reassemble one channel's layer slice into the
        per-(room, index_start) buffer; once all channels have arrived, enqueue
        the complete chunk for the main-thread scatter (same as _handle_kvdata).
        Each channel's layer frames are copied once, straight into pinned staging
        (``_recv_layers``)."""
        recv_time = time.time()
        # [TAG, room, index_start, num_pages, dst_lane_start, num_lanes,
        #  ch, n_ch, lo, n_layers, num_aux]
        info = PrefillTransferInfo.from_json(bytes(msg[1]).decode())
        room = info.room
        index_start = int(bytes(msg[2]).decode())
        num_pages = int(bytes(msg[3]).decode())
        num_lanes = info.n_lane_to_send
        ch = int(bytes(msg[4]).decode())
        n_ch = int(bytes(msg[5]).decode())
        lo = int(bytes(msg[6]).decode())
        n_layers = int(bytes(msg[7]).decode())
        num_aux = int(bytes(msg[8]).decode())
        aux_blobs = [bytes(f) for f in msg[9 : 9 + num_aux]]
        # One copy: wire -> pinned staging (or bytes on the CPU fallback path).
        staged = self._recv_layers(msg[9 + num_aux :])
        _nbytes = sum(b.nbytes if isinstance(b, StageBuf) else len(b) for b in staged)
        if info.push_start_time > 0:
            push_to_recv_secs = max(0.0, recv_time - info.push_start_time)
            self._push_to_recv_stats.record(push_to_recv_secs, _nbytes)
        key = (room, index_start, info.src_tp_rank)
        complete = None
        with self._chan_reasm_lock:
            ent = self._chan_reasm.get(key)
            if ent is None:
                ent = {
                    "layers": [None] * n_layers,
                    "got": set(),
                    "aux": [],
                    "num_pages": num_pages,
                    "num_lanes": num_lanes,
                }
                self._chan_reasm[key] = ent
            for i, st in enumerate(staged):
                ent["layers"][lo + i] = st
            if ch == 0 and aux_blobs:
                ent["aux"] = aux_blobs
            ent["got"].add(ch)
            if len(ent["got"]) >= n_ch:
                self._chan_reasm.pop(key, None)
                complete = ent
        logger.info(
            "[pd_kv_resharding] _handle_kvdata_ch frame: room=%s index_start=%d "
            "src_tp_rank=%d dst_tp_rank=%d ch=%d/%d layers=[%d:%d]/%d aux_num=%d "
            "input_bytes=%d metadata=(src_attn_tp_size=%d, dst_attn_tp_size=%d, "
            "n_lane_to_send=%d, resharding_direction=%s)",
            room,
            index_start,
            info.src_tp_rank,
            info.dst_tp_rank,
            ch,
            n_ch,
            lo,
            lo + len(staged),
            n_layers,
            num_aux,
            _nbytes,
            info.src_attn_tp_size,
            info.dst_attn_tp_size,
            info.n_lane_to_send,
            info.resharding_direction,
        )
        if complete is not None:
            chunk = PendingChunk(
                index_start=index_start,
                num_pages=complete["num_pages"],
                layer_blobs=complete["layers"],
                aux_blobs=complete["aux"],
                prefill_transfer_info=info,
            )
            logger.info(
                "[pd_kv_resharding] _handle_kvdata_ch chunk complete: "
                "room=%s index_start=%d src_tp_rank=%d pending_chunk_bytes=%d "
                "num_pages=%d require_reassemble=%s assemble_min_chunks=%d "
                "assemble_key=%s",
                room,
                index_start,
                info.src_tp_rank,
                chunk.nbytes,
                chunk.num_pages,
                chunk.require_reassemble,
                chunk.assemble_min_chunks,
                chunk.assemble_key,
            )
            with self.decode_pending_lock:
                PendingChunk.process_incoming_chunk(
                    self.decode_pending[room],
                    self.reasm_pending[room],
                    chunk,
                    stats=self._assemble_stats,
                )

    def _expected_chunk_bytes(self, num_pages: int, num_lanes: int) -> int:
        """Total bytes a chunk of ``num_pages`` pages x ``num_lanes`` packed
        lanes should carry across all layers, from the fused buffer geometry
        (``page_size * kv_packing * head_dim * dtype_size`` per lane per page).
        Used to hard-validate incoming slices instead of silently reshaping."""
        total = 0
        for buf in self.layer_buffers:
            # buf: [num_blocks, page_size, kv_heads_packed, kv_packing, head_dim]
            page_size = buf.shape[1]
            per_lane = int(np.prod(buf.shape[3:])) * buf.element_size()
            total += num_pages * page_size * num_lanes * per_lane
        return total


class TpuKVSender(TpuKVSenderBase):

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: list[Any] | None = None,
        num_kv_tokens: int | None = None,
    ):
        # Runs on the scheduler main thread; gather is inline (device-safe),
        # the network push is offloaded to a background thread.
        index_start = self.curr_idx
        self.curr_idx += len(kv_indices)
        is_last_chunk = self.curr_idx == self.num_kv_indices
        self.kv_mgr.gather_and_push(
            self.bootstrap_room,
            kv_indices,
            index_start,
            is_last_chunk,
            self.aux_index if is_last_chunk else None,
        )
        self._record_transfer_indices(kv_indices, state_indices)


class TpuKVReceiver(TpuKVReceiverBase):

    def _clear_room_state(self) -> None:
        if getattr(self.kv_mgr, "decode_states", None) is not None:
            with self.kv_mgr.decode_states_lock:
                self.kv_mgr.decode_states.pop(self.bootstrap_room, None)
            with self.kv_mgr.decode_pending_lock:
                dropped = self.kv_mgr.decode_pending.pop(self.bootstrap_room, None)
                reasm_map = self.kv_mgr.reasm_pending.pop(self.bootstrap_room, None)
            for chunk in dropped or ():
                self.kv_mgr._release_staged(chunk)
            if reasm_map:
                for chunk_list in reasm_map.values():
                    for chunk in chunk_list:
                        self.kv_mgr._release_staged(chunk)
        # Drop any partially-reassembled multi-channel chunks for this room (#3),
        # returning the staging their arrived channels already acquired.
        reasm = getattr(self.kv_mgr, "_chan_reasm", None)
        if reasm is not None:
            with self.kv_mgr._chan_reasm_lock:
                for k in [k for k in reasm if k[0] == self.bootstrap_room]:
                    ent = reasm.pop(k, None)
                    for st in (ent or {}).get("layers", ()):
                        if isinstance(st, StageBuf):
                            st.release()


class TpuKVBootstrapServer(TpuKVBootstrapServerBase):
    pass
