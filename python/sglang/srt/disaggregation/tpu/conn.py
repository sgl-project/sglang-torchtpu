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

Fp8 with odd head is not supported yet..

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
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

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

logger = logging.getLogger(__name__)

# Background threads per prefill rank for the (host-only) ZMQ push — i.e. how
# many distinct transfers (rooms/chunks) a rank pushes concurrently. Orthogonal
# to SGLANG_TPU_KV_CHANNELS (parallel streams *within* one transfer); the
# effective stream count is this x channels. See _push_channels.
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


# Emit one ``trace-event`` line to stdout per KV transfer phase (gather / push /
# receive / scatter).
from sglang.srt.disaggregation.tpu.profile_util import (  # noqa: E402
    fmt_bytes,
    perfetto_trace,
)


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


@dataclasses.dataclass
class _DecodeTransferState:
    """Decode-side per-request transfer bookkeeping.

    Attributes:
      dst_page_indices: Block IDs on this process.
      aux_index: Aux state slot index.
      dst_lane_per_page_block: Number of lanes per block.
      num_pages: Total transfer page count.
      dst_lanes_total: Total expected lanes to scatter.
      scattered: Number of lanes scattered so far.
    """

    dst_page_indices: np.ndarray
    aux_index: Optional[int]
    dst_lane_per_page_block: int
    num_pages: int = dataclasses.field(init=False)
    dst_lanes_total: int = dataclasses.field(init=False)
    scattered: int = 0

    def __post_init__(self) -> None:
        self.num_pages = len(self.dst_page_indices)
        self.dst_lanes_total = self.num_pages * self.dst_lane_per_page_block


@dataclasses.dataclass
class _PendingChunk:
    """Received KVDATA chunk awaiting main-thread scatter.

    Attributes:
      index_start: Starting page index for the chunk.
      num_pages: Number of pages in the chunk.
      layer_blobs: Per-layer KV data buffers or stage slots.
      aux_blobs: Aux state byte payloads.
      dst_lane_start: Target lane offset for MHA resharding.
      num_lanes: Number of packed lanes in the chunk.
    """

    index_start: int
    num_pages: int
    layer_blobs: List[StageBuf]
    aux_blobs: List[bytes]
    dst_lane_start: int
    num_lanes: int


@dataclasses.dataclass
class _PrefillTransferInfo:
    """Prefill-side record for target decode endpoint and resharding state.

    Attributes:
      room: Disagg transfer room ID.
      decode_endpoint: Destination socket endpoint address.
      num_pages: Total page count to transfer.
      dst_tp_rank: Target tensor parallel rank.
      decode_attn_tp_size: Attention TP size of the decode engine.
    """

    room: int
    decode_endpoint: str
    num_pages: int
    dst_tp_rank: int = 0
    decode_attn_tp_size: int = 1


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
        self._chan_socks: Dict[str, List] = {}
        self._chan_socks_lock = threading.Lock()
        # Per-room push progress: {room: {"drained": int, "total": int}}.
        # Chunk pushes run on a pool (_PUSH_POOL_WORKERS) and can drain out of
        # order, so conclude Success by drained page count reaching the room
        # total (not by an is_last push), mirroring the decode. See _note_push_*.
        self._push_progress: Dict[int, dict] = {}
        self._push_progress_lock = threading.Lock()
        self._serve(MessageTag.TAG_TRSFR, self._handle_transfer)

    def _init_decode(self) -> None:
        self.decode_states: Dict[int, _DecodeTransferState] = {}
        self.decode_states_lock = threading.Lock()
        # room -> queue of received-but-not-yet-scattered chunks. The bg ZMQ
        # thread appends; the main thread (poll) drains + scatters.
        self.decode_pending: Dict[int, Deque[_PendingChunk]] = defaultdict(deque)
        self.decode_pending_lock = threading.Lock()
        self._scatter_stats = _KvStats("scatter", _kv_stats_every())
        # Pinned/aligned reusable host staging for pooled H2D (#1).
        self._stage_pool = PinnedHostPool(max_bytes=self._stage_pool_budget())
        # Multi-channel reassembly buffer (#3), keyed by (room, index_start).
        self._chan_reasm: Dict = {}
        self._chan_reasm_lock = threading.Lock()
        # copy=False so the recv thread can copy each layer frame *once*, straight
        # into pinned staging (see _recv_layers), rather than wire -> bytes -> staging.
        self._serve(MessageTag.TAG_KVDATA_CH, self._handle_kvdata_ch, copy=False)

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
            self.layer_specs[0].page_block_shape if self.layer_specs else None,
            self._num_lanes_full,
            self._kv_packing,
        )

    def record_decode_endpoint(
        self,
        room: int,
        endpoint: str,
        num_pages: int,
        dst_tp_rank: int = 0,
        decode_attn_tp_size: int = 1,
    ) -> _PrefillTransferInfo:
        # Stash a full record so gather_and_push can read ``.decode_endpoint``.
        return _PrefillTransferInfo(
            room,
            endpoint,
            num_pages,
            dst_tp_rank,
            decode_attn_tp_size,
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

    def _get_page_shape_str(self, n_pages: int, num_lanes: Optional[int] = None) -> str:
        """Perfetto token shape string for layer page buffers."""
        if not self.layer_buffers:
            return f"{n_pages}x{num_lanes if num_lanes is not None else 1}"
        buf0 = self.layer_buffers[0]
        if num_lanes is None:
            num_lanes = int(buf0.shape[2]) if buf0.dim() >= 4 else 1
        if buf0.dim() < 5:
            return f"{n_pages}x{num_lanes}"
        s = buf0.shape
        return f"{n_pages}x{int(s[1])}x{num_lanes}x{int(s[3])}x{int(s[4])}"

    def _compute_kv_transfer_lane_spec(
        self, info: _PrefillTransferInfo
    ) -> tpu_transfer.KvTransferLaneSpec:
        """Computes lane slice metadata for both prefill and decode."""
        prefill_tp = self.attn_tp_size
        if not info.decode_attn_tp_size:
            raise ValueError("Decode attention tp size cannot be None or 0.")
        decode_tp = info.decode_attn_tp_size

        if decode_tp == prefill_tp:
            # Fast path: no resharding, whole page, byte-identical to before.
            return tpu_transfer.KvTransferLaneSpec(
                src_lane_start=0,
                lane_count=self._num_lanes_full,
                dst_lane_start=0,
                num_heads=0,
                sends_aux=True,
            )
        return tpu_transfer.compute_kv_transfer_lane_spec(
            prefill_tp,
            decode_tp,
            self.kv_args.engine_rank,
            info.dst_tp_rank,
            self.kv_args.total_kv_head_num,
            self._kv_packing,
        )

    def _gather_lanes(
        self, page_indices: np.ndarray, lane_slice: Optional[Tuple[int, int]]
    ) -> Tuple[List[memoryview], List[StageBuf], Optional[Any], int]:
        """Gathers lanes by lane slice and page indices, D2H."""
        d2h_future = None
        if getattr(self, "_async_d2h", False):
            d2h_future, stages = tpu_transfer.gather_pages_to_stage_async(
                self.layer_buffers,
                page_indices,
                self._stage_pool,
                lane_slice=lane_slice,
            )
        else:
            stages = tpu_transfer.gather_pages_to_stage(
                self.layer_buffers,
                page_indices,
                self._stage_pool,
                lane_slice=lane_slice,
            )
        layer_frames = [st.mv() for st in stages]
        nbytes = sum(st.nbytes for st in stages)
        return layer_frames, stages, d2h_future, nbytes

    def gather_and_push(
        self,
        room: int,
        page_indices: np.ndarray,
        index_start: int,
        is_last_chunk: bool,
        aux_index: Optional[int],
    ) -> None:
        """Gathers pages and push."""
        if self.request_status.get(room) == KVPoll.Failed:
            return
        self._assert_layer_page_buffers()
        try:
            with self._transfer_infos_lock:
                infos: List[_PrefillTransferInfo] = list(
                    self.transfer_infos.get(room, {}).values()
                )
            if not infos:
                raise KVTransferError(room, "no decode endpoint registered")

            page_indices = np.asarray(page_indices, dtype=np.int64)
            n_pages = len(page_indices)
            plans = []  # (endpoint, lane_slice, dst_lane_start, num_lanes, sends_aux)
            total_lanes = 0
            for info in infos:
                spec = self._compute_kv_transfer_lane_spec(info)
                lane_slice = (
                    None
                    if (
                        spec.src_lane_start == 0
                        and spec.lane_count == self._num_lanes_full
                    )
                    else (spec.src_lane_start, spec.lane_count)
                )
                plans.append(
                    (
                        info.decode_endpoint,
                        lane_slice,
                        spec.dst_lane_start,
                        spec.lane_count,
                        spec.sends_aux,
                    )
                )
                total_lanes += info.num_pages * spec.lane_count
            self._note_push_submitted(room, total_lanes)

            aux_blobs_all: List[bytes] = []
            if is_last_chunk and aux_index is not None:
                aux_blobs_all = tpu_transfer.read_aux_row(self.kv_args, aux_index)

            n_ch = self._n_channels
            stats = self._gather_stats
            n_layers = len(self.layer_buffers)
            # FIXME: Gather full lanes once to host and slice on host to reduce D2H calls.
            for endpoint, lane_slice, dst_lane_start, num_lanes, sends_aux in plans:
                if num_lanes == 0 and not sends_aux:
                    continue
                with perfetto_trace(
                    self.trace_pid, f"gather/r{room}/i{index_start}"
                ) as _sp:
                    _t0 = time.perf_counter()
                    layer_frames, stages, d2h_future, nbytes = self._gather_lanes(
                        page_indices, lane_slice
                    )
                    if stats is not None:
                        stats.record(time.perf_counter() - _t0, nbytes)
                    _sp.args.update(
                        {
                            "shape": lambda: self._get_page_shape_str(
                                n_pages, num_lanes
                            ),
                            "layers": f"{n_layers}L",
                            "size": lambda: fmt_bytes(nbytes),
                        }
                    )
                aux_blobs = aux_blobs_all if sends_aux else []
                units = n_pages * num_lanes
                logger.info(
                    "[pd_kv_resharding] gather_and_push room=%s endpoint=%s "
                    "index_start=%d n_pages=%d lane_slice=%s dst_lane_start=%d "
                    "num_lanes=%d aux=%s bytes=%d",
                    room,
                    endpoint,
                    index_start,
                    n_pages,
                    lane_slice,
                    dst_lane_start,
                    num_lanes,
                    bool(aux_blobs),
                    nbytes,
                )
                self._push_pool.submit(
                    self._push_channels,
                    room,
                    endpoint,
                    index_start,
                    n_pages,
                    dst_lane_start,
                    num_lanes,
                    units,
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

    def _channel_sockets(self, endpoint: str, n: int) -> List[zmq.Socket]:
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
        room: int,
        endpoint: str,
        index_start: int,
        num_pages: int,
        dst_lane_start: int,
        num_lanes: int,
        units: int,
        aux_blobs: List[bytes],
        layer_frames: List[Union[memoryview, bytes]],
        stages: Optional[List[StageBuf]],
        d2h_future: Optional[Any],
        n_ch: int,
    ) -> None:
        """Background thread (#3): send one endpoint's chunk as ``n_ch`` messages
        over ``n_ch`` parallel sockets, each carrying a contiguous layer slice.
        Aux rides channel 0. Frame: ``[TAG_KVDATA_CH, room, index_start,
        num_pages, dst_lane_start, num_lanes, ch, n_ch, layer_lo, num_layers,
        num_aux] + aux + layer_slice``."""

        n_layers = len(layer_frames)
        push_word = f"pushch{n_ch}" if n_ch > 1 else "push"
        push_name = f"{push_word}/r{room}/i{index_start}"
        wait_name = f"gatherwait/r{room}/i{index_start}"
        try:
            if d2h_future is not None:
                with perfetto_trace(self.trace_pid, wait_name):
                    d2h_future.wait()
            with perfetto_trace(
                self.trace_pid,
                push_name,
                shape=lambda: self._get_page_shape_str(num_pages, num_lanes),
                layers=f"{n_layers}L",
            ):
                n_layers = len(layer_frames)
                bounds = [(c * n_layers) // n_ch for c in range(n_ch + 1)]
                copy_false = stages is not None
                trackers = []
                socks = self._channel_sockets(endpoint, n_ch)
                for ch in range(n_ch):
                    lo, hi = bounds[ch], bounds[ch + 1]
                    aux = aux_blobs if ch == 0 else []
                    msg = (
                        [
                            MessageTag.TAG_KVDATA_CH,
                            str(room).encode(),
                            str(index_start).encode(),
                            str(num_pages).encode(),
                            str(dst_lane_start).encode(),
                            str(num_lanes).encode(),
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
                    with self._endpoint_locks[(endpoint, ch)]:
                        if copy_false:
                            trackers.append(
                                sock.send_multipart(msg, copy=False, track=True)
                            )
                        else:
                            sock.send_multipart(msg)
                for t in trackers:
                    t.wait()
            if stages is not None:
                for st in stages:
                    st.release()
            # See _push: count this chunk's lane-page units toward the room total.
            self._note_push_done(room, units)
        except Exception as e:
            logger.exception("KV channel push failed for room %s", room)
            self._note_push_failed(room)
            self.record_failure(room, f"TPU KV channel push failed: {e}")
            self.update_status(room, KVPoll.Failed)

    def _create_staging_buffer_for_chunks(
        self, frames: List[zmq.Frame]
    ) -> List[StageBuf]:
        """Creates staging buffer from received chunks using shared executor."""
        stage_pool = self._stage_pool

        def _copy_frame_to_stage(frame: zmq.Frame) -> StageBuf:
            mv = frame.buffer
            st = stage_pool.acquire(mv.nbytes)
            st.copy_bytes(mv)  # wire -> pinned staging (releases GIL)
            return st

        return list(self._executor.map(_copy_frame_to_stage, frames))

    @staticmethod
    def _release_staged(chunk: _PendingChunk) -> None:
        """Return any pooled StageBufs held by a chunk that won't be scattered
        (failure / teardown paths). No-op for the bytes (CPU) path."""
        for b in chunk.layer_blobs:
            if isinstance(b, StageBuf):
                b.release()

    def register_decode_state(
        self, room: int, dst_indices: np.ndarray, aux_index: Optional[int]
    ) -> None:
        state = _DecodeTransferState(
            np.asarray(dst_indices, dtype=np.int64), aux_index, self._num_lanes_full
        )
        with self.decode_states_lock:
            self.decode_states[room] = state
        logger.info(
            "[pd_kv_resharding] register_decode_state room=%s num_pages=%d "
            "dst_lane_total=%d target=%d",
            room,
            state.num_pages,
            self._num_lanes_full,
            state.dst_lanes_total,
        )

    def _handle_kvdata_ch(self, msg: List[zmq.Frame]) -> None:
        """Background thread (#3): reassemble one channel's layer slice into the
        per-(room, index_start) buffer; once all channels have arrived, enqueue
        the complete chunk for the main-thread scatter.
        Each channel's layer frames are copied once, straight into pinned staging
        (``_recv_layers``)."""
        # [TAG, room, index_start, num_pages, dst_lane_start, num_lanes,
        #  ch, n_ch, lo, n_layers, num_aux]
        room = int(bytes(msg[1]).decode())
        index_start = int(bytes(msg[2]).decode())
        num_pages = int(bytes(msg[3]).decode())
        dst_lane_start = int(bytes(msg[4]).decode())
        num_lanes = int(bytes(msg[5]).decode())
        ch = int(bytes(msg[6]).decode())
        n_ch = int(bytes(msg[7]).decode())
        lo = int(bytes(msg[8]).decode())
        n_layers = int(bytes(msg[9]).decode())
        num_aux = int(bytes(msg[10]).decode())
        aux_blobs = [bytes(f) for f in msg[11 : 11 + num_aux]]
        # One copy: wire -> pinned staging (or bytes on the CPU fallback path).
        staged = self._create_staging_buffer_for_chunks(msg[11 + num_aux :])
        _nbytes = sum(b.nbytes if isinstance(b, StageBuf) else len(b) for b in staged)
        with perfetto_trace(
            self.trace_pid,
            f"recv_{room}_{ch}",
            layers=f"{len(staged)}L",
            size=lambda: fmt_bytes(_nbytes),
        ):
            # Key on dst_lane_start too: under fan-in several prefill sources push
            # to the SAME (room, index_start) but different lane offsets, and their
            # channels must not reassemble into one another.
            key = (room, index_start, dst_lane_start)
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

            if complete is not None:
                chunk = _PendingChunk(
                    index_start,
                    complete["num_pages"],
                    complete["layers"],
                    complete["aux"],
                    dst_lane_start,
                    complete["num_lanes"],
                )
                with self.decode_pending_lock:
                    self.decode_pending[room].append(chunk)

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

    def process_decode_pending(self, room: int) -> None:
        """Runs on the scheduler MAIN thread (from TpuKVReceiver.poll): drain
        any received chunks for ``room`` and scatter them H2D into the decode KV
        pool — each into its KV-head lane sub-range under MHA resharding. Marks
        the room Success once every lane-page unit has landed (a decode rank may
        be filled by several prefill sources at different lane offsets). The
        device touch (index_copy_) is serialised with the forward loop."""
        with self.decode_pending_lock:
            pending = self.decode_pending.get(room)
            if not pending:
                return
            chunks = list(pending)
            pending.clear()

        with self.decode_states_lock:
            state = self.decode_states.get(room)
        if state is None:
            logger.warning("decode KVDATA for unknown room %s, dropping", room)
            return

        self._assert_layer_page_buffers()
        for i, chunk in enumerate(chunks):
            dst_pages = state.dst_page_indices[
                chunk.index_start : chunk.index_start + chunk.num_pages
            ]
            fail_reason = None
            # Validate decode page size.
            if len(dst_pages) != chunk.num_pages:
                fail_reason = (
                    f"page count mismatch: prefill sent {chunk.num_pages} at "
                    f"offset {chunk.index_start} but decode only has "
                    f"{len(dst_pages)} dst pages"
                )
            # Validate decode lane numbers.
            elif chunk.dst_lane_start + chunk.num_lanes > state.dst_lane_per_page_block:
                fail_reason = (
                    f"lane range [{chunk.dst_lane_start}:"
                    f"{chunk.dst_lane_start + chunk.num_lanes}] exceeds this "
                    f"decode rank's {state.dst_lane_per_page_block} lanes"
                )
            # Validate stage buffer size match the chunk.
            else:
                got = sum(b.nbytes for b in chunk.layer_blobs)
                want = self._expected_chunk_bytes(chunk.num_pages, chunk.num_lanes)
                if got != want:
                    fail_reason = (
                        f"[pd_kv_resharding] chunk byte mismatch: got {got} want "
                        f"{want} (num_pages={chunk.num_pages} num_lanes="
                        f"{chunk.num_lanes} dst_lane_start={chunk.dst_lane_start})"
                    )
            if fail_reason is not None:
                # Hand back staging held by this and every still-undrained chunk
                # (scatter would otherwise have released them in its finally).
                for c in chunks[i:]:
                    self._release_staged(c)
                self.record_failure(room, fail_reason)
                self.update_status(room, KVPoll.Failed)
                return

            # Whole-page (equal attn_tp) fast path uses None to skip the narrow.
            full = (
                chunk.dst_lane_start == 0
                and chunk.num_lanes == state.dst_lane_per_page_block
            )
            dst_lane_slice = None if full else (chunk.dst_lane_start, chunk.num_lanes)

            _nbytes = sum(b.nbytes for b in chunk.layer_blobs)
            with perfetto_trace(
                self.trace_pid,
                f"scatter_{room}",
                shape=lambda: self._get_page_shape_str(
                    chunk.num_pages, chunk.num_lanes
                ),
                layers=f"{len(chunk.layer_blobs)}L",
                size=lambda: fmt_bytes(_nbytes),
            ):
                _t0 = time.perf_counter()
                tpu_transfer.scatter_stage_from_host(
                    self.layer_buffers,
                    dst_pages,
                    chunk.layer_blobs,
                    self.layer_specs,
                    dst_lane_slice,
                )
                if self._scatter_stats is not None:
                    self._scatter_stats.record(time.perf_counter() - _t0, _nbytes)

            state.scattered += chunk.num_pages * chunk.num_lanes
            logger.info(
                "[pd_kv_resharding] scatter room=%s index_start=%d num_pages=%d "
                "dst_lane_slice=%s scattered=%d/%d",
                room,
                chunk.index_start,
                chunk.num_pages,
                dst_lane_slice,
                state.scattered,
                state.dst_lanes_total,
            )
            # Aux rides the prefill's last chunk only, so its presence — not an
            # is_last flag — is the signal to write it.
            if chunk.aux_blobs and state.aux_index is not None:
                tpu_transfer.write_aux_row(
                    self.kv_args, state.aux_index, chunk.aux_blobs
                )
            # Conclude only once every lane-page unit has landed. The full target
            # is known up front (the decode allocated exactly these pages and
            # holds a fixed lane count), so completion doesn't depend on chunk
            # ordering, source count, or an is_last marker.
            if state.scattered >= state.dst_lanes_total:
                logger.info("[pd_kv_resharding] decode room=%s complete", room)
                self.update_status(room, KVPoll.Success)


class TpuKVSender(TpuKVSenderBase):
    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List[Any]] = None,
    ) -> None:
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
            # Return staging held by any undrained chunks (e.g. teardown after a
            # failure that wasn't a page-count mismatch).
            for chunk in dropped or ():
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
