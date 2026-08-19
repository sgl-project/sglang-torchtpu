"""TPU KV transfer backend for P/D disaggregation (host ZMQ-push).

The default TPU backend. It reuses the device-agnostic control plane from
disaggregation/common (bootstrap handshake, ZMQ coordination, the
KVPoll state machine) and adds a host push data plane: the prefill
gathers KV pages off-device (D2H) and pushes the bytes to the decode, which
scatters them H2D into pages it allocated. No device pointers cross the
process boundary (see transfer.py).

Threading: torch_tpu serialises device work on one PjRt queue and the RPA v3
kernel donates the fused kv_buffer each forward, so touching it off the
main thread crashes the worker. Every device KV op therefore runs on the
scheduler main thread -- the prefill D2H gather inline in TpuKVSender.send,
the decode H2D scatter inline in TpuKVReceiver.poll -- while background
threads only move host bytes over ZMQ.

Supports MHA resharding across heterogeneous attention TP (fan-in, fan-out,
GQA replication, FP8 odd-head). No MLA, PP, CP, DP-attention, or mamba/SWA.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import enum
import errno
import logging
import os
import threading
import time
from collections import defaultdict, deque
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
import zmq

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
from sglang.srt.disaggregation.tpu import transfer as tpu_transfer
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
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class MessageTag(bytes, enum.Enum):
    """ZMQ message tags for the TPU P/D control and data planes."""

    TRSFR = b"TRSFR"
    KVDATA = b"KVDATA"


class KVTransferError(Exception):
    def __init__(self, bootstrap_room: int, failure_reason: str):
        super().__init__(failure_reason)
        self.bootstrap_room = bootstrap_room
        self.failure_reason = failure_reason

    def __str__(self):
        return f"KVTransferError(room={self.bootstrap_room}): {self.failure_reason}"


def _tcp_endpoint(ip: str, port: int) -> str:
    if ":" in ip and not ip.startswith("["):  # bare IPv6
        ip = f"[{ip}]"
    return f"tcp://{ip}:{port}"


# Background threads per prefill rank for the (host-only) ZMQ push — i.e. how
# many distinct transfers (rooms/chunks) a rank pushes concurrently. Orthogonal
# to SGLANG_TPU_KV_CHANNELS (parallel streams *within* one transfer); the
# effective stream count is this x channels. See _push / _push_channels.
_PUSH_POOL_WORKERS = 4


def _kv_stats_every() -> int:
    """Transfers between logged latency summaries (0 disables).

    Set via ``SGLANG_DEBUG_TPU_KV_STATS_EVERY``.
    """
    try:
        return int(os.environ.get("SGLANG_DEBUG_TPU_KV_STATS_EVERY", "100"))
    except (TypeError, ValueError):
        return 100


def _kv_async() -> bool:
    """Whether to issue the prefill D2H asynchronously (``SGLANG_TPU_KV_ASYNC``).

    See ``gather_pages_to_stage_async`` for the donation-safety reasoning.
    """
    return os.environ.get("SGLANG_TPU_KV_ASYNC", "1").lower() not in ("0", "", "false")


@dataclasses.dataclass
class _KvStats:
    """Cumulative per-stage transfer instrumentation.

    Attributes:
        stage: Transfer stage name (e.g. 'gather' D2H or 'scatter' H2D).
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


class TpuKVManager(CommonKVManager):
    """``tpu`` P/D KV manager: control-plane scaffolding plus the host push/scatter data plane."""

    BACKEND_NAME = "tpu"

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: bool | None = False,
    ):
        # Class name, used in logs, error messages, thread names.
        self._name = type(self).__name__
        if is_mla_backend:
            raise NotImplementedError(f"{self._name} does not support MLA models yet.")
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)

        # Filled in later by the scheduler via attach_kv_pool(); the bootstrap
        # queue owns the pool object and only it knows the concrete tensors.
        self.kv_pool: KVCache | None = None
        self.layer_buffers: list[torch.Tensor] | None = None

        # Per-endpoint send locks (a single zmq PUSH socket must not be used
        # concurrently from multiple sender threads).
        self._endpoint_locks = defaultdict(threading.Lock)

        # Process/track ID for tracing instrumentation
        self._initialize_trace_id()

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # CommonKVManager.__init__ already created self.transfer_infos and
            # self.req_to_decode_prefix_len for the prefill role.
            self._init_prefill()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self._init_decode()
        else:
            raise ValueError(
                f"{self._name} expects disaggregation mode "
                f"{DisaggregationMode.PREFILL} or {DisaggregationMode.DECODE}, "
                f"got {self.disaggregation_mode}"
            )

    def _initialize_trace_id(self):
        mode_str = self.disaggregation_mode.value
        if self.attn_dp_size > 1:
            self.trace_pid = (
                f"tpu-kv-{mode_str}-dp{self.attn_dp_rank}-tp{self.attn_tp_rank}"
            )
        else:
            self.trace_pid = f"tpu-kv-{mode_str}{self.attn_tp_rank}"

    # ---- pool attachment ------------------------------------------------

    def attach_kv_pool(self, kv_pool: KVCache) -> None:
        """Attach the concrete per-layer KV tensors and run backend setup.

        Extracts the block-indexed per-layer buffers, then calls
        ``_on_pool_attached``. Called once at scheduler init.
        """
        if self.kv_pool is not None:
            raise RuntimeError(
                f"{self._name}: a KV pool is already attached. "
                "A second attach is unexpected."
            )
        self.kv_pool = kv_pool
        self.layer_buffers = tpu_transfer.get_layer_page_buffers(kv_pool)
        self._on_pool_attached()

    def _assert_layer_page_buffers(self):
        if self.layer_buffers is None:
            raise RuntimeError(
                f"{self._name}.attach_kv_pool() was never called; the "
                "KV pool tensors are required for TPU P/D transfer."
            )

    # ---- recv-loop scaffolding ------------------------------------------

    def _serve(self, expected_tag: bytes, handler, copy: bool = True) -> None:
        """Start a daemon thread that dispatches tagged messages to ``handler``.

        Pulls multipart messages off ``server_socket`` and dispatches those
        whose leading tag matches ``expected_tag``. ``copy=False`` receives
        zero-copy ``zmq.Frame``s so the handler copies each frame straight to
        its destination.
        """
        name = self._name

        def loop():
            poller = zmq.Poller()
            poller.register(self.server_socket, zmq.POLLIN)
            while not getattr(self, "_closed", False):
                try:
                    socks = dict(poller.poll(timeout=100))
                    if self.server_socket not in socks:
                        continue
                    msg = self.server_socket.recv_multipart(
                        copy=copy, flags=zmq.NOBLOCK
                    )
                except zmq.ContextTerminated:
                    logger.warning("%s recv loop ending: ZMQ context terminated", name)
                    return
                except zmq.Again:
                    continue
                except zmq.ZMQError as e:
                    if e.errno in (zmq.ETERM, zmq.ENOTSOCK, getattr(errno, "EBADF", 9)):
                        logger.warning("%s recv loop ending: %s", name, e)
                        return
                    logger.exception("%s unexpected ZMQ exception", name)
                    continue
                except Exception as e:  # pragma: no cover - socket teardown
                    logger.exception("%s unexpected recv exception: %s", name, e)
                    continue
                if getattr(self, "_closed", False):
                    break
                if not msg or bytes(msg[0]) != expected_tag:
                    logger.warning("%s recv: unexpected msg tag %r", name, msg[:1])
                    continue
                try:
                    handler(msg)
                except Exception:
                    logger.exception(
                        "%s: error handling %r message", name, expected_tag
                    )

        t = threading.Thread(
            target=loop,
            name=f"{name}-{self.disaggregation_mode.value}",
            daemon=True,
        )
        self._serve_thread = t
        t.start()

    def _serve_router(self, handlers: dict, copy: bool = True) -> None:
        """Like ``_serve`` but dispatches by leading tag to one of ``handlers``.

        Used when a role accepts several message types on one socket.
        ``copy=False`` receives zero-copy ``zmq.Frame``s (see ``_serve``).
        """
        name = self._name

        def loop():
            poller = zmq.Poller()
            poller.register(self.server_socket, zmq.POLLIN)
            while not getattr(self, "_closed", False):
                try:
                    socks = dict(poller.poll(timeout=100))
                    if self.server_socket not in socks:
                        continue
                    msg = self.server_socket.recv_multipart(
                        copy=copy, flags=zmq.NOBLOCK
                    )
                except zmq.ContextTerminated:
                    logger.warning("%s recv loop ending: ZMQ context terminated", name)
                    return
                except zmq.Again:
                    continue
                except zmq.ZMQError as e:
                    if e.errno in (zmq.ETERM, zmq.ENOTSOCK, getattr(errno, "EBADF", 9)):
                        logger.warning("%s recv loop ending: %s", name, e)
                        return
                    logger.exception("%s unexpected ZMQ exception", name)
                    continue
                except Exception as e:  # pragma: no cover - socket teardown
                    logger.exception("%s unexpected recv exception: %s", name, e)
                    continue
                if getattr(self, "_closed", False):
                    break
                handler = handlers.get(bytes(msg[0])) if msg else None
                if handler is None:
                    logger.warning("%s recv: unexpected msg tag %r", name, msg[:1])
                    continue
                try:
                    handler(msg)
                except Exception:
                    logger.exception("%s: error handling %r", name, msg[0])

        t = threading.Thread(
            target=loop,
            name=f"{name}-{self.disaggregation_mode.value}",
            daemon=True,
        )
        self._serve_thread = t
        t.start()

    def _handle_transfer(self, msg: list[bytes]) -> None:
        """Prefill side: record a ready decode's endpoint for room R.

        Advances to WaitingForInput once every expected decode rank has reported.
        """
        # [TAG, room, decode_ip, decode_port, aux_present, num_pages,
        #  required_dst_info_num, decode_prefix_len,
        #  dst_tp_rank, decode_attn_tp_size]
        if len(msg) != 10:
            raise ValueError(
                f"Malformed {MessageTag.TRSFR.decode()} message. "
                f"Expected 10 frames, got {len(msg)}."
            )
        room = int(msg[1].decode())
        decode_ip = msg[2].decode()
        decode_port = int(msg[3].decode())
        num_pages = int(msg[5].decode())
        required = int(msg[6].decode())
        prefix_len = int(msg[7].decode())
        dst_tp_rank = int(msg[8].decode())
        decode_attn_tp_size = int(msg[9].decode())

        logger.info(
            "[pd_kv_resharding] _handle_transfer room=%s endpoint=%s:%s num_pages=%d "
            "required=%d dst_tp_rank=%d decode_attn_tp=%d",
            room,
            decode_ip,
            decode_port,
            num_pages,
            required,
            dst_tp_rank,
            decode_attn_tp_size,
        )
        endpoint = _tcp_endpoint(decode_ip, decode_port)
        bucket = self.transfer_infos.setdefault(room, {})
        bucket[endpoint] = self.get_prefill_transfer_info(
            room, endpoint, num_pages, dst_tp_rank, decode_attn_tp_size
        )
        self.req_to_decode_prefix_len[room] = prefix_len
        if len(bucket) >= required:
            logger.info("[pd_kv_resharding] now room: %s, bucket: %s", bucket, room)
            self.update_status(room, KVPoll.WaitingForInput)
            logger.info("[pd_kv_resharding] request status: %s", self.request_status)

    def get_prefill_transfer_info(
        self,
        room: int,
        endpoint: str,
        num_pages: int,
        dst_tp_rank: int = 0,
        decode_attn_tp_size: int = 1,
    ) -> PrefillTransferInfo:
        # Stash a full record so gather_and_push can read ``.decode_endpoint``.
        return PrefillTransferInfo(
            room=room,
            decode_endpoint=endpoint,
            num_pages=num_pages,
            kv_packing=self._kv_packing,
            src_tp_rank=self.kv_args.engine_rank,
            dst_tp_rank=dst_tp_rank,
            src_attn_tp_size=self.attn_tp_size,
            dst_attn_tp_size=decode_attn_tp_size,
            total_num_head=self.kv_args.total_kv_head_num,
            n_layers=len(self.layer_buffers),
            page_size=int(self.layer_buffers[0].shape[1]),
        )

    def register_decode_state(
        self, room: int, dst_indices: np.ndarray, aux_index: int | None
    ) -> None:
        state = DecodeTransferState(
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

    def process_decode_pending(self, room: int) -> None:
        """Drain and scatter received chunks for ``room`` H2D (scheduler main thread).

        Runs from ``TpuKVReceiver.poll`` so the device touch stays serialised
        with the forward loop. Each chunk lands in its KV-head lane sub-range
        (MHA resharding); the room is marked Success once every lane-page unit
        has landed.
        """
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
            info = chunk.prefill_transfer_info
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
            elif (
                info.dst_lane_idx + info.n_lane_to_send > state.dst_lane_per_page_block
            ):
                fail_reason = (
                    f"lane range [{info.dst_lane_idx}:"
                    f"{info.dst_lane_idx + info.n_lane_to_send}] exceeds this "
                    f"decode rank's {state.dst_lane_per_page_block} lanes"
                )
            # Validate stage buffer size match the chunk.
            else:
                got = chunk.nbytes
                want = self._expected_chunk_bytes(chunk.num_pages, info.n_lane_to_send)
                logger.info(
                    "[pd_kv_resharding] process_decode_pending chunk: room=%s index_start=%d "
                    "got_bytes=%d want_bytes=%d (num_pages=%d num_lanes=%d dst_lane_start=%d)",
                    room,
                    chunk.index_start,
                    got,
                    want,
                    chunk.num_pages,
                    info.n_lane_to_send,
                    info.dst_lane_idx,
                )
                if got != want:
                    fail_reason = (
                        f"[pd_kv_resharding] chunk byte mismatch: got {got} want "
                        f"{want} (num_pages={chunk.num_pages} num_lanes="
                        f"{info.n_lane_to_send} dst_lane_start={info.dst_lane_idx})"
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
                info.dst_lane_idx == 0
                and info.n_lane_to_send == state.dst_lane_per_page_block
            )
            dst_lane_slice = None if full else (info.dst_lane_idx, info.n_lane_to_send)

            _nbytes = chunk.nbytes

            _t0 = time.perf_counter()
            pool = self._stage_pool
            tpu_transfer.scatter_stage_from_host(
                self.layer_buffers,
                dst_pages,
                chunk.layer_blobs,
                pool,
                dst_lane_slice,
            )
            self._scatter_stats.record(time.perf_counter() - _t0, _nbytes)
            tpu_transfer.dump_disagg_chunk(
                room=room,
                role="decode",
                source_tp=info.src_attn_tp_size,
                dest_tp=info.dst_attn_tp_size,
                layer_blobs=chunk.layer_blobs,
                aux_blobs=chunk.aux_blobs,
                info=info,
                index_start=chunk.index_start,
                num_pages=chunk.num_pages,
                from_assemble=chunk.from_assemble,
                extra={
                    "src_tp_rank": info.src_tp_rank,
                    "dst_tp_rank": info.dst_tp_rank,
                    "stage": "scatter",
                },
                filename=f"data_scatter_dst{info.dst_tp_rank}_chunk{chunk.index_start}.pkl",
            )
            state.scattered += chunk.num_pages * info.n_lane_to_send
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
                self.update_status(room, KVPoll.Success)

    # ---- tpu push/scatter data plane ------------------------------------

    def _stage_pool_budget(self) -> int:
        """Pinned-staging budget for this rank's pool.

        Splits the host-RAM fraction across the co-located TP ranks so the
        aggregate stays within budget.
        """
        sa = getattr(self, "server_args", None)
        tp = getattr(sa, "tp_size", 1) or 1
        nnodes = getattr(sa, "nnodes", 1) or 1
        local_ranks = max(1, tp // max(1, nnodes))  # TP ranks on this host
        return default_pool_max_bytes(local_shares=local_ranks)

    # ---- role init (called from __init__) -------------------------------

    def _init_prefill(self) -> None:
        # The pool here is only used for the (host-only) ZMQ push.
        self._push_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=_PUSH_POOL_WORKERS, thread_name_prefix="tpu-kv-push"
        )
        self._gather_stats = _KvStats("gather", _kv_stats_every())
        # Pinned/aligned reusable host staging for zero-copy D2H + send (#1/#4).
        self._stage_pool = PinnedHostPool(max_bytes=self._stage_pool_budget())
        self._async_d2h = _kv_async()  # #7 (opt-in)
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
        self._serve_router(
            {
                MessageTag.KVDATA: self._handle_kvdata,
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
        """Main thread: register the room's push-progress entry with its total lanes."""
        with self._push_progress_lock:
            if room not in self._push_progress:
                self._push_progress[room] = {"drained": 0, "total": total_units}

    def _note_push_done(self, room: int, push_lane_units: int) -> None:
        """Worker thread: credit a drained chunk; conclude Success at the room total.

        Uses the same known-total accounting as the decode scatter side, so
        completion needs no ``is_last`` marker and tolerates pushes draining out
        of order.
        """
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
        """Drop the room's progress entry so a later sibling push can't mask the failure."""
        with self._push_progress_lock:
            self._push_progress.pop(room, None)

    def _gather_lanes(
        self, page_indices: np.ndarray, lane_slice: tuple[int, int] | None
    ) -> tuple[list[memoryview | bytes], list[StageBuf] | None, Any | None, int]:
        """Gather pages for the given lane slice off-device (D2H)."""
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
        """Gather this chunk's pages off-device and push them to the decode."""
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
                    self._push,
                    info,
                    index_start,
                    n_pages,
                    n_lane_total,
                    aux_blobs,
                    layer_frames,
                    stages,
                    d2h_future,
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
        info: PrefillTransferInfo,
        index_start: int,
        num_pages: int,
        units: int,
        aux_blobs: list[bytes],
        layer_frames: list[memoryview | bytes],
        stages: list[StageBuf] | None,
        d2h_future: Any | None,
    ) -> None:
        """Background thread: send one chunk over ZMQ to decode endpoint."""
        try:
            if d2h_future is not None:
                _wait_t0 = time.perf_counter()
                d2h_future.wait()
                info.gather_time_ms += (time.perf_counter() - _wait_t0) * 1000.0

            push_t0 = time.perf_counter()
            info.push_start_time = time.time()
            logger.info(
                "[pd_kv_resharding] start push to endpoint=%s (room=%s, num_pages=%d, n_lanes=%d)",
                info.decode_endpoint,
                info.room,
                num_pages,
                info.n_lane_to_send,
            )
            msg = (
                [
                    MessageTag.KVDATA,
                    info.to_json().encode(),
                    str(index_start).encode(),
                    str(num_pages).encode(),
                    str(len(aux_blobs)).encode(),
                ]
                + aux_blobs
                + layer_frames
            )
            sock = self._connect(info.decode_endpoint)
            with self._endpoint_locks[info.decode_endpoint]:
                if stages is not None:
                    tracker = sock.send_multipart(msg, copy=False, track=True)
                else:
                    tracker = None
                    sock.send_multipart(msg)
            if tracker is not None:
                tracker.wait()
            info.push_time_ms = (time.perf_counter() - push_t0) * 1000.0
            logger.info(
                "[pd_kv_resharding] finish push to endpoint=%s (room=%s, push_time_ms=%.2f)",
                info.decode_endpoint,
                info.room,
                info.push_time_ms,
            )
            tpu_transfer.dump_disagg_chunk(
                room=info.room,
                role="prefill",
                source_tp=info.src_attn_tp_size,
                dest_tp=info.dst_attn_tp_size,
                layer_blobs=layer_frames,
                aux_blobs=aux_blobs,
                info=info,
                index_start=index_start,
                num_pages=num_pages,
                extra={
                    "src_tp_rank": info.src_tp_rank,
                    "dst_tp_rank": info.dst_tp_rank,
                    "stage": "push",
                },
                filename=f"data_src{info.src_tp_rank}_dst{info.dst_tp_rank}_chunk{index_start}.pkl",
            )
            if stages is not None:
                for st in stages:
                    st.release()
            self._note_push_done(info.room, units)
        except Exception as e:
            logger.exception("KV push failed for room %s", info.room)
            self._note_push_failed(info.room)
            self.record_failure(info.room, f"TPU KV push failed: {e}")
            self.update_status(info.room, KVPoll.Failed)

    def _recv_layers(self, frames: list[zmq.Frame | bytes]) -> list[StageBuf | bytes]:
        """Background recv thread: turn the wire's layer frames into ``layer_blobs``.

        On the TPU path each frame is copied once, straight from ZMQ's buffer
        into a pooled pinned ``StageBuf`` held until the main-thread scatter
        releases it. ``frames`` may be ``zmq.Frame``s (copy=False) or ``bytes`` (copy=True).
        """
        pool = getattr(self, "_stage_pool", None)
        if pool is not None:
            stages = []
            for f in frames:
                mv = f.buffer if isinstance(f, zmq.Frame) else memoryview(f)
                st = pool.acquire(mv.nbytes)
                st.copy_bytes(mv)  # the one and only host copy: wire -> staging
                stages.append(st)
            return stages
        return [bytes(f) for f in frames]

    @staticmethod
    def _release_staged(chunk: PendingChunk) -> None:
        """Release any pooled StageBufs held by an un-scattered chunk (no-op for bytes)."""
        for b in chunk.layer_blobs:
            if isinstance(b, StageBuf):
                b.release()

    def _handle_kvdata(self, msg: list[zmq.Frame]) -> None:
        """Background thread: process received KV chunk and enqueue for scatter/reassembly."""
        recv_time = time.time()
        # [TAG, info_json, index_start, num_pages, num_aux] + aux + layers
        info = PrefillTransferInfo.from_json(bytes(msg[1]).decode())
        room = info.room
        index_start = int(bytes(msg[2]).decode())
        num_pages = int(bytes(msg[3]).decode())
        num_aux = int(bytes(msg[4]).decode())
        aux_blobs = [bytes(f) for f in msg[5 : 5 + num_aux]]
        staged = self._recv_layers(msg[5 + num_aux :])
        _nbytes = sum(b.nbytes if isinstance(b, StageBuf) else len(b) for b in staged)
        if info.push_start_time > 0:
            push_to_recv_secs = max(0.0, recv_time - info.push_start_time)
            self._push_to_recv_stats.record(push_to_recv_secs, _nbytes)

        chunk = PendingChunk(
            index_start=index_start,
            num_pages=num_pages,
            layer_blobs=staged,
            aux_blobs=aux_blobs,
            prefill_transfer_info=info,
        )
        logger.info(
            "[pd_kv_resharding] _handle_kvdata chunk received: "
            "room=%s index_start=%d src_tp_rank=%d dst_tp_rank=%d "
            "pending_chunk_bytes=%d num_pages=%d require_reassemble=%s "
            "assemble_min_chunks=%d assemble_key=%s",
            room,
            index_start,
            info.src_tp_rank,
            info.dst_tp_rank,
            chunk.nbytes,
            chunk.num_pages,
            chunk.require_reassemble,
            chunk.assemble_min_chunks,
            chunk.assemble_key,
        )
        tpu_transfer.dump_disagg_chunk(
            room=room,
            role="decode",
            source_tp=info.src_attn_tp_size,
            dest_tp=info.dst_attn_tp_size,
            layer_blobs=chunk.layer_blobs,
            aux_blobs=chunk.aux_blobs,
            info=info,
            index_start=chunk.index_start,
            num_pages=chunk.num_pages,
            from_assemble=chunk.from_assemble,
            extra={
                "src_tp_rank": info.src_tp_rank,
                "dst_tp_rank": info.dst_tp_rank,
                "stage": "received_raw",
            },
            filename=f"data_recv_src{info.src_tp_rank}_dst{info.dst_tp_rank}_chunk{chunk.index_start}.pkl",
        )
        with self.decode_pending_lock:
            PendingChunk.process_incoming_chunk(
                self.decode_pending[room],
                self.reasm_pending[room],
                chunk,
                stats=self._assemble_stats,
            )

    def _expected_chunk_bytes(self, num_pages: int, num_lanes: int) -> int:
        """Bytes a chunk of ``num_pages`` x ``num_lanes`` packed lanes carries across layers.

        Derived from the fused buffer geometry to hard-validate incoming slices.
        """
        total = 0
        for buf in self.layer_buffers:
            # buf: [num_blocks, page_size, kv_heads_packed, kv_packing, head_dim]
            page_size = buf.shape[1]
            per_lane = int(np.prod(buf.shape[3:])) * buf.element_size()
            total += num_pages * page_size * num_lanes * per_lane
        return total


class TpuKVSender(CommonKVSender):
    """Prefill-side handle: chunk gating, the inline gather + background push, poll, failure."""

    def __init__(
        self,
        mgr: TpuKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: list[int],
        pp_rank: int,
        req_has_disagg_prefill_dp_rank: bool = False,
    ):
        super().__init__(
            mgr,
            bootstrap_addr,
            bootstrap_room,
            dest_tp_ranks,
            pp_rank,
            req_has_disagg_prefill_dp_rank=req_has_disagg_prefill_dp_rank,
        )
        # Downcast to TpuKVManager to support typing.
        self.kv_mgr = cast(TpuKVManager, self.kv_mgr)
        self.conclude_state = None
        self.init_time = time.time()

    def pop_decode_prefix_len(self) -> int:
        return self.kv_mgr.req_to_decode_prefix_len.pop(self.bootstrap_room, 0)

    def should_send_kv_chunk(self, num_pages: int, last_chunk: bool) -> bool:
        return num_pages > 0 or last_chunk

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

    def _pre_poll(self) -> None:
        """Hook to advance the data plane before reading status; no-op for the host push backend."""

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state
        self._pre_poll()
        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
        elif status == KVPoll.Bootstrapping and self.init_time is not None:
            elapsed = time.time() - self.init_time
            if elapsed >= self.kv_mgr.bootstrap_timeout:
                self.kv_mgr.record_failure(
                    self.bootstrap_room,
                    f"Request {self.bootstrap_room} timed out after "
                    f"{elapsed:.1f}s in KVPoll.Bootstrapping (decode never sent "
                    "its pre-alloc notification).",
                )
                self.conclude_state = KVPoll.Failed
                return KVPoll.Failed
        return status

    def failure_exception(self):
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed
        self.clear()
        with self.kv_mgr.failure_lock:
            reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "Failed due to an unknown reason"
            )
        raise KVTransferError(self.bootstrap_room, reason)


class TpuKVReceiver(CommonKVReceiver):
    """Decode-side handle: the ready notification, poll, failure, and room cleanup.

    The manager's ``process_decode_pending`` does the pull/scatter.
    """

    def __init__(
        self,
        mgr: TpuKVManager,
        bootstrap_addr: str,
        bootstrap_room: int | None = None,
    ):
        self.init_time = None
        super().__init__(mgr, bootstrap_addr, bootstrap_room)

    def send_metadata(
        self,
        kv_indices,
        aux_index: int | None = None,
        state_indices: list | None = None,
        decode_prefix_len: int | None = None,
    ):
        if self.bootstrap_infos is None:
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                f"Could not fetch prefill parallel info from {self.bootstrap_addr}",
            )
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return

        # Record destination pages locally; the decode does its own pull/scatter.
        self.kv_mgr.register_decode_state(self.bootstrap_room, kv_indices, aux_index)
        dst_attn_tp_size = self.kv_mgr.attn_tp_size
        dst_tp_rank = self.kv_mgr.kv_args.engine_rank % max(1, dst_attn_tp_size)
        num_pages = len(kv_indices)
        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            with lock:
                sock.send_multipart(
                    [
                        MessageTag.TRSFR,
                        str(self.bootstrap_room).encode(),
                        self.kv_mgr.local_ip.encode(),
                        str(self.kv_mgr.rank_port).encode(),
                        b"1" if aux_index is not None else b"0",
                        str(num_pages).encode(),
                        str(self.required_dst_info_num).encode(),
                        str(decode_prefix_len or 0).encode(),
                        str(dst_tp_rank).encode(),
                        str(dst_attn_tp_size).encode(),
                    ]
                )
        self.init_time = time.time()

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state
        # Advance the transfer for this room on the main thread (device-safe).
        self.kv_mgr.process_decode_pending(self.bootstrap_room)
        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
        elif status == KVPoll.WaitingForInput and self.init_time is not None:
            elapsed = time.time() - self.init_time
            if elapsed >= self.kv_mgr.waiting_timeout:
                self.kv_mgr.record_failure(
                    self.bootstrap_room,
                    f"Request {self.bootstrap_room} timed out after "
                    f"{elapsed:.1f}s in KVPoll.WaitingForInput (KV never arrived "
                    "from prefill).",
                )
                self.conclude_state = KVPoll.Failed
                return KVPoll.Failed
        return status

    def clear(self) -> None:
        super().clear()
        self._clear_room_state()

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

    def failure_exception(self):
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed
        self.clear()
        with self.kv_mgr.failure_lock:
            reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "Failed due to an unknown reason"
            )
        raise KVTransferError(self.bootstrap_room, reason)


class TpuKVBootstrapServer(CommonKVBootstrapServer):
    """Bootstrap handshake/discovery server; device-agnostic, inherited from common."""
