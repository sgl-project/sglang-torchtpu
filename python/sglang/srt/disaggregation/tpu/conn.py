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

Deliberately scoped to the common single-/multi-host MHA case: no MLA, PP,
CP, DP-attention, mamba/SWA state, or heterogeneous-TP staging.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import os
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional

import numpy as np
import numpy.typing as npt
import zmq

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.tpu import transfer as tpu_transfer

# Re-exported from base so existing imports (and a future raiden backend)
# keep resolving ``from ...tpu.conn import KVTransferError, _tcp_endpoint``.
from sglang.srt.disaggregation.tpu.base import (  # noqa: F401
    TAG_TRSFR,
    KVTransferError,
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

# prefill -> decode: KV page bytes (+ aux on last chunk). The decode->prefill
# ready notification (TAG_TRSFR) is shared and lives in base.
TAG_KVDATA = b"KVDATA"
# prefill -> decode, one frame per parallel channel (#3, opt-in). Each carries a
# contiguous slice of the layers; the decode reassembles across channels.
TAG_KVDATA_CH = b"KVDATAC"

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


# Emit one ``trace-event`` line to stdout per KV transfer phase (gather / push /
# receive / scatter).
from sglang.srt.disaggregation.tpu.profile_util import (
    _PERFETTO,
    _fmt_bytes,
    perfetto_pid,
    perfetto_token_shape_str,
    trace,
)


class _KvStats:
    """Cumulative per-stage transfer instrumentation (#8). Accumulates wall time
    + bytes for one stage (e.g. ``gather`` D2H, ``scatter`` H2D) and logs a
    rolling summary every ``every`` completions. Cheap: one perf_counter pair
    and a few adds per transfer."""

    __slots__ = ("stage", "every", "n", "secs", "nbytes")

    def __init__(self, stage: str, every: int):
        self.stage = stage
        self.every = every
        self.n = 0
        self.secs = 0.0
        self.nbytes = 0

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


class _DecodeTransferState:
    """Decode-side per-request bookkeeping (kept locally; never sent to
    prefill). ``dst_page_indices`` are *this* process's block ids; their count
    (``num_pages``) is the full transfer size — the decode allocated exactly the
    pages the prefill will send (the delta beyond any cached prefix) and handed
    that count to the prefill in the bootstrap handshake, so both sides agree on
    it up front.

    ``scattered_pages`` accumulates across chunks so we mark the room Success
    only once *every* page has landed. Chunks are pushed by a pool of prefill
    worker threads (``_PUSH_POOL_WORKERS``) over parallel channels and so can
    arrive out of order; completion is a page count reaching ``num_pages``, not
    an ``is_last`` marker (which could conclude early on a short last chunk).
    See ``_note_push_*`` for the prefill-side completion counterpart."""

    __slots__ = ("dst_page_indices", "aux_index", "num_pages", "scattered_pages")

    def __init__(self, dst_page_indices: np.ndarray, aux_index: Optional[int]):
        self.dst_page_indices = dst_page_indices
        self.aux_index = aux_index
        self.num_pages = len(dst_page_indices)
        self.scattered_pages = 0


class _PendingChunk:
    """A KVDATA chunk received off the wire and waiting to be scattered H2D on
    the main thread. ``layer_blobs`` holds one entry per layer: a pooled
    :class:`StageBuf` on the TPU path (the wire bytes were copied straight into
    pinned staging by the recv thread — see ``_recv_layers``) or raw ``bytes``
    on the CPU fallback path. ``aux_blobs`` are always small ``bytes`` rows."""

    __slots__ = ("index_start", "num_pages", "layer_blobs", "aux_blobs")

    def __init__(self, index_start, num_pages, layer_blobs, aux_blobs):
        self.index_start = index_start
        self.num_pages = num_pages
        self.layer_blobs = layer_blobs
        self.aux_blobs = aux_blobs


class _PrefillTransferInfo:
    """Prefill-side record of where to push a room's KV bytes."""

    __slots__ = ("room", "decode_endpoint", "num_pages")

    def __init__(self, room: int, decode_endpoint: str, num_pages: int):
        self.room = room
        self.decode_endpoint = decode_endpoint
        self.num_pages = num_pages


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
        self._trace_pid = self._perfetto_pid()
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
        self._start_prefill_thread()

    def _init_decode(self) -> None:
        self._trace_pid = self._perfetto_pid()
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
        self._start_decode_thread()

    def _on_pool_attached(self) -> None:
        # Fused RPA v3 layer buffer: [num_blocks, page_size, kv_heads_packed,
        # kv_packing, head_dim]. Cache the packed-lane count (dim 2, == this
        # rank's full KV-head lanes) and kv_packing (dim 3).
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
            self.layer_specs[0][1] if self.layer_specs else None,
            self._num_lanes_full,
            self._kv_packing,
        )

    # ---- perfetto tracing (opt-in; see _perfetto_enabled) ---------------

    def _perfetto_pid(self) -> str:
        """Perfetto process-track id based on attention dp rank, tp rank and mode."""
        return perfetto_pid(self)

    def _perfetto_token_shape_str(self, n_pages: int, num_lanes: int) -> str:
        """Perfetto token shape string."""
        return perfetto_token_shape_str(self, n_pages, num_lanes)

    @contextlib.contextmanager
    def _trace(self, name: str):
        """Perfetto trace context manager."""
        with trace(self, name) as span:
            yield span

    # ---- prefill side ---------------------------------------------------

    def _start_prefill_thread(self):
        self._serve(TAG_TRSFR, self._handle_transfer)

    def _record_decode_endpoint(self, room: int, endpoint: str, num_pages: int):
        # Stash a full record so gather_and_push can read ``.decode_endpoint``.
        return _PrefillTransferInfo(room, endpoint, num_pages)

    def _note_push_submitted(self, room: int, total_pages: int) -> None:
        """Main thread: register the room's push-progress entry (with the total
        page count agreed in the bootstrap handshake) before any push is
        submitted, so a failing sibling can invalidate it and a worker can't
        recreate a fresh one behind a recorded failure."""
        with self._push_progress_lock:
            if room not in self._push_progress:
                self._push_progress[room] = {"drained": 0, "total": total_pages}

    def _note_push_done(self, room: int, n_pages: int) -> None:
        """Worker thread: a chunk push of ``n_pages`` pages drained. Conclude
        Success once the drained page count reaches the room total — the same
        known-total accounting as the decode's scatter side, so completion needs
        no ``is_last`` marker and is immune to pushes draining out of order."""
        with self._push_progress_lock:
            ent = self._push_progress.get(room)
            if ent is None:
                return  # already failed/cleared
            ent["drained"] += n_pages
            done = ent["drained"] >= ent["total"]
            if done:
                self._push_progress.pop(room, None)
        if done:
            self.update_status(room, KVPoll.Success)

    def _note_push_failed(self, room: int) -> None:
        """A chunk push failed: drop the progress entry so a later-draining
        sibling push can't then conclude Success over the recorded failure."""
        with self._push_progress_lock:
            self._push_progress.pop(room, None)

    def gather_and_push(
        self,
        room: int,
        page_indices: np.ndarray,
        index_start: int,
        is_last_chunk: bool,
        aux_index: Optional[int],
    ):
        """Runs on the scheduler MAIN thread (from TpuKVSender.send): gather
        the requested KV pages off-device (D2H) and read the aux row, then
        hand the resulting host bytes to a background thread for the ZMQ push.
        The device touch (index_select / .cpu()) is therefore serialised with
        the forward loop; only host work happens on the background thread."""
        self._require_pool()
        try:
            bucket = self.transfer_infos.get(room)
            if not bucket:
                raise KVTransferError(room, "no decode endpoint registered")

            page_indices = np.asarray(page_indices, dtype=np.int64)
            pool = getattr(self, "_stage_pool", None)
            use_stage = (
                pool is not None
                and self.layer_buffers
                and self.layer_buffers[0].device.type != "cpu"
            )
            n_layers = len(self.layer_buffers)
            n_pages = len(page_indices)
            with self._trace(f"gather/r{room}/i{index_start}") as _sp:
                _t0 = time.perf_counter()
                d2h_future = None
                if use_stage:
                    # Zero-copy: D2H straight into pinned pooled buffers; the wire
                    # frames are their memoryviews (sent copy=False in _push).
                    if getattr(self, "_async_d2h", False):
                        d2h_future, stages = tpu_transfer.gather_pages_to_stage_async(
                            self.layer_buffers, page_indices, pool
                        )
                    else:
                        stages = tpu_transfer.gather_pages_to_stage(
                            self.layer_buffers, page_indices, pool
                        )
                    layer_frames = [st.mv() for st in stages]
                    nbytes = sum(st.nbytes for st in stages)
                else:
                    stages = None
                    layer_frames = tpu_transfer.gather_pages_to_host(
                        self.layer_buffers, page_indices
                    )
                    nbytes = sum(len(b) for b in layer_frames)
                stats = getattr(self, "_gather_stats", None)
                if stats is not None:
                    stats.record(time.perf_counter() - _t0, nbytes)
                if _PERFETTO:
                    _sp.name = (
                        f"gather/r{room}/i{index_start}/"
                        f"{self._perfetto_token_shape_str(n_pages, self._num_lanes_full)}/{n_layers}L/"
                        f"{_fmt_bytes(nbytes)}"
                    )

            aux_blobs: List[bytes] = []
            if is_last_chunk and aux_index is not None:
                aux_blobs = tpu_transfer.read_aux_row(self.kv_args, aux_index)

            endpoints = [info.decode_endpoint for info in bucket.values()]
            # Register the room's total (from the handshake) before submitting,
            # so a worker draining fast can't race the entry's creation.
            total_pages = next(iter(bucket.values())).num_pages
            self._note_push_submitted(room, total_pages)
            n_ch = getattr(self, "_n_channels", 1)
            _shape = (
                self._perfetto_token_shape_str(n_pages, self._num_lanes_full)
                if _PERFETTO
                else ""
            )
            _size = _fmt_bytes(nbytes) if _PERFETTO else ""
            _push_word = f"pushch{n_ch}" if n_ch > 1 else "push"
            push_name = (
                f"{_push_word}/r{room}/i{index_start}/{_shape}/{n_layers}L/{_size}"
            )
            wait_name = f"gatherwait/r{room}/i{index_start}/{_size}"

            if n_ch > 1:
                # Parallel path (#3): shard the layers across N streams.
                self._push_pool.submit(
                    self._push_channels,
                    room,
                    endpoints,
                    index_start,
                    len(page_indices),
                    aux_blobs,
                    layer_frames,
                    stages,
                    d2h_future,
                    n_ch,
                    push_name,
                    wait_name,
                )
            else:
                payload = (
                    [
                        TAG_KVDATA,
                        str(room).encode(),
                        str(index_start).encode(),
                        str(len(page_indices)).encode(),
                        str(len(aux_blobs)).encode(),
                    ]
                    + aux_blobs
                    + layer_frames
                )
                self._push_pool.submit(
                    self._push,
                    room,
                    endpoints,
                    payload,
                    len(page_indices),
                    stages,
                    d2h_future,
                    push_name,
                    wait_name,
                )
        except Exception as e:
            logger.exception("KV gather failed for room %s", room)
            self._note_push_failed(room)
            self.record_failure(room, f"TPU KV gather failed: {e}")
            self.update_status(room, KVPoll.Failed)

    def _push(
        self,
        room: int,
        endpoints: List[str],
        payload: List,
        n_pages: int,
        stages=None,
        d2h_future=None,
        push_name: str = "push",
        wait_name: str = "gatherwait",
    ):
        """Background thread: host-only ZMQ send of the gathered KV bytes.

        With pooled staging (``stages`` set) the layer frames are the pinned
        buffers' memoryviews, sent **zero-copy** (``copy=False``); the buffers
        are returned to the pool only after the send has drained (tracked), so
        a reused slot can never clobber an in-flight send. When the D2H was
        issued async (#7), ``d2h_future`` is waited here — off the scheduler
        thread — before the bytes are read for sending."""
        try:
            if d2h_future is not None:
                with self._trace(wait_name):
                    d2h_future.wait()
            with self._trace(push_name):
                trackers = []
                for endpoint in endpoints:
                    sock = self._connect(endpoint)
                    with self._endpoint_locks[endpoint]:
                        if stages is not None:
                            trackers.append(
                                sock.send_multipart(payload, copy=False, track=True)
                            )
                        else:
                            sock.send_multipart(payload)
                for t in trackers:
                    t.wait()
            if stages is not None:
                for st in stages:
                    st.release()
            # Count this chunk's pages toward the room total; Success fires once
            # every page has drained, regardless of push completion order.
            self._note_push_done(room, n_pages)
        except Exception as e:
            # On failure don't recycle the buffers: zmq (copy=False) may still
            # hold them until its send drains — let GC reclaim them instead.
            logger.exception("KV push failed for room %s", room)
            self._note_push_failed(room)
            self.record_failure(room, f"TPU KV push failed: {e}")
            self.update_status(room, KVPoll.Failed)

    def _channel_sockets(self, endpoint: str, n: int) -> List:
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
        endpoints: List[str],
        index_start: int,
        num_pages: int,
        aux_blobs: List[bytes],
        layer_frames: List,
        stages,
        d2h_future,
        n_ch: int,
        push_name: str = "pushch",
        wait_name: str = "gatherwait",
    ):
        """Background thread (#3): send the chunk as ``n_ch`` messages over
        ``n_ch`` parallel sockets, each carrying a contiguous layer slice. Aux
        rides channel 0. Frame: ``[TAG_KVDATA_CH, room, index_start, num_pages,
        ch, n_ch, layer_lo, num_layers, num_aux] + aux + layer_slice``."""
        try:
            if d2h_future is not None:
                with self._trace(wait_name):
                    d2h_future.wait()
            with self._trace(push_name):
                n_layers = len(layer_frames)
                bounds = [(c * n_layers) // n_ch for c in range(n_ch + 1)]
                copy_false = stages is not None
                trackers = []
                for endpoint in endpoints:
                    socks = self._channel_sockets(endpoint, n_ch)
                    for ch in range(n_ch):
                        lo, hi = bounds[ch], bounds[ch + 1]
                        aux = aux_blobs if ch == 0 else []
                        msg = (
                            [
                                TAG_KVDATA_CH,
                                str(room).encode(),
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
            # See _push: count this chunk's pages toward the room total.
            self._note_push_done(room, num_pages)
        except Exception as e:
            logger.exception("KV channel push failed for room %s", room)
            self._note_push_failed(room)
            self.record_failure(room, f"TPU KV channel push failed: {e}")
            self.update_status(room, KVPoll.Failed)

    # ---- decode side ----------------------------------------------------

    def _start_decode_thread(self):
        # Accept both the single-message (TAG_KVDATA) and per-channel
        # (TAG_KVDATA_CH) formats on the one PULL socket. copy=False so the recv
        # thread can copy each layer frame *once*, straight into pinned staging
        # (see _recv_layers), rather than wire -> bytes -> staging.
        self._serve_router(
            {
                TAG_KVDATA: self._handle_kvdata,
                TAG_KVDATA_CH: self._handle_kvdata_ch,
            },
            copy=False,
        )

    def _recv_layers(self, frames):
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
    def _release_staged(chunk: _PendingChunk) -> None:
        """Return any pooled StageBufs held by a chunk that won't be scattered
        (failure / teardown paths). No-op for the bytes (CPU) path."""
        for b in chunk.layer_blobs:
            if isinstance(b, StageBuf):
                b.release()

    def register_decode_state(
        self, room: int, dst_indices: np.ndarray, aux_index: Optional[int]
    ):
        with self.decode_states_lock:
            self.decode_states[room] = _DecodeTransferState(
                np.asarray(dst_indices, dtype=np.int64), aux_index
            )

    def _handle_kvdata(self, msg: List):
        """Background thread: parse a KVDATA frame and stash the chunk. The layer
        bytes are copied once, straight into pinned staging (``_recv_layers``);
        NO device ops here — the H2D scatter happens on the main thread in
        process_decode_pending(). ``msg`` items are ``zmq.Frame``s (copy=False)."""
        # [TAG, room, index_start, num_pages, num_aux] + aux + layers
        room = int(bytes(msg[1]).decode())
        with self._trace(f"recv_{room}"):
            index_start = int(bytes(msg[2]).decode())
            num_pages = int(bytes(msg[3]).decode())
            num_aux = int(bytes(msg[4]).decode())
            aux_blobs = [bytes(f) for f in msg[5 : 5 + num_aux]]
            layer_blobs = self._recv_layers(msg[5 + num_aux :])
            chunk = _PendingChunk(index_start, num_pages, layer_blobs, aux_blobs)
            with self.decode_pending_lock:
                self.decode_pending[room].append(chunk)

    def _handle_kvdata_ch(self, msg: List):
        """Background thread (#3): reassemble one channel's layer slice into the
        per-(room, index_start) buffer; once all channels have arrived, enqueue
        the complete chunk for the main-thread scatter (same as _handle_kvdata).
        Each channel's layer frames are copied once, straight into pinned staging
        (``_recv_layers``)."""
        # [TAG, room, index_start, num_pages, ch, n_ch, lo, n_layers, num_aux]
        room = int(bytes(msg[1]).decode())
        with self._trace(f"recv_{room}"):
            index_start = int(bytes(msg[2]).decode())
            num_pages = int(bytes(msg[3]).decode())
            ch = int(bytes(msg[4]).decode())
            n_ch = int(bytes(msg[5]).decode())
            lo = int(bytes(msg[6]).decode())
            n_layers = int(bytes(msg[7]).decode())
            num_aux = int(bytes(msg[8]).decode())
            aux_blobs = [bytes(f) for f in msg[9 : 9 + num_aux]]
            # One copy: wire -> pinned staging (or bytes on the CPU fallback path).
            staged = self._recv_layers(msg[9 + num_aux :])

            key = (room, index_start)
            complete = None
            with self._chan_reasm_lock:
                ent = self._chan_reasm.get(key)
                if ent is None:
                    ent = {
                        "layers": [None] * n_layers,
                        "got": set(),
                        "aux": [],
                        "num_pages": num_pages,
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
                )
                with self.decode_pending_lock:
                    self.decode_pending[room].append(chunk)

    def process_decode_pending(self, room: int) -> None:
        """Runs on the scheduler MAIN thread (from TpuKVReceiver.poll): drain
        any received chunks for ``room`` and scatter them H2D into the decode
        KV pool. Marks the room Success once every page has landed. The device
        touch (index_copy_) is serialised with the forward loop."""
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

        self._require_pool()
        for i, chunk in enumerate(chunks):
            dst_pages = state.dst_page_indices[
                chunk.index_start : chunk.index_start + chunk.num_pages
            ]
            if len(dst_pages) != chunk.num_pages:
                # Hand back staging held by this and every still-undrained chunk
                # (scatter would otherwise have released them in its finally).
                for c in chunks[i:]:
                    self._release_staged(c)
                self.record_failure(
                    room,
                    f"page count mismatch: prefill sent {chunk.num_pages} at "
                    f"offset {chunk.index_start} but decode only has "
                    f"{len(dst_pages)} dst pages",
                )
                self.update_status(room, KVPoll.Failed)
                return
            with self._trace(f"scatter_{room}"):
                _t0 = time.perf_counter()
                pool = getattr(self, "_stage_pool", None)
                if (
                    pool is not None
                    and self.layer_buffers
                    and self.layer_buffers[0].device.type != "cpu"
                ):
                    tpu_transfer.scatter_stage_from_host(
                        self.layer_buffers,
                        dst_pages,
                        chunk.layer_blobs,
                        self.layer_specs,
                        pool,
                    )
                else:
                    tpu_transfer.scatter_pages_from_host(
                        self.layer_buffers, dst_pages, chunk.layer_blobs, self.layer_specs
                    )
                stats = getattr(self, "_scatter_stats", None)
                if stats is not None:
                    stats.record(
                        time.perf_counter() - _t0,
                        sum(
                            b.nbytes if isinstance(b, StageBuf) else len(b)
                            for b in chunk.layer_blobs
                        ),
                    )
                state.scattered_pages += chunk.num_pages
                # Aux rides the prefill's last chunk only, so its presence — not an
                # is_last flag — is the signal to write it.
                if chunk.aux_blobs and state.aux_index is not None:
                    tpu_transfer.write_aux_row(
                        self.kv_args, state.aux_index, chunk.aux_blobs
                    )
            # Conclude only once every page has landed. The full count is known
            # up front (the decode allocated exactly these pages and handshaked
            # the count to the prefill), so completion doesn't depend on chunk
            # ordering or an is_last marker.
            if state.scattered_pages >= state.num_pages:
                self.update_status(room, KVPoll.Success)


class TpuKVSender(TpuKVSenderBase):
    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List] = None,
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
