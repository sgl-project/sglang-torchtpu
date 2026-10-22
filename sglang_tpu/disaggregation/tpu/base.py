"""Backend-agnostic control-plane base classes for TPU P/D disaggregation.

The TPU disaggregation backends all share one control plane — the
device-agnostic bootstrap handshake / ZMQ coordination / ``KVPoll`` state
machine inherited from ``disaggregation/common`` — and differ only in how the
*bulk KV cache* moves between prefill and decode (the "data plane").

This module factors out that shared control-plane scaffolding so each concrete
backend is a thin subclass that implements only the data-plane seam. Two
backends drive the design:

* the in-tree ``tpu`` backend (``conn.py``) pushes KV page bytes over ZMQ
  (host D2H gather -> network -> H2D scatter); it is the default.
* an opt-in ``tpu_sync`` backend (a future ``raiden_conn.py``) hands the
  bulk transfer to the standalone Raiden DMA engine and ships only small
  metadata over ZMQ.

What a new backend implements (the seam)
----------------------------------------
``Manager`` (subclass :class:`TpuKVManagerBase`):
  * ``_init_prefill()`` / ``_init_decode()`` — set up the role's per-request
    state and start its recv loop via :meth:`TpuKVManagerBase._serve`.
  * ``_on_pool_attached()`` — build per-backend transfer state from
    ``self.layer_buffers`` (the shared ``attach_kv_pool`` already extracted the
    per-layer buffers + specs before calling this hook).
  * the data-plane methods the Sender/Receiver call:
      - prefill: a "publish/transfer this room" method invoked from
        ``Sender.send`` (e.g. ``gather_and_push`` / ``register_and_publish``);
      - decode: ``register_decode_state(room, dst_indices, aux_index)`` and
        ``process_decode_pending(room)`` — the latter is invoked from
        ``Receiver.poll`` on the scheduler **main thread** so device touches
        stay serialised with the forward loop.

``Sender`` (subclass :class:`TpuKVSenderBase`): implement ``send``. The poll /
  timeout / failure-exception plumbing is shared; ``_pre_poll`` is an optional
  hook to advance the data plane before reading status.

``Receiver`` (subclass :class:`TpuKVReceiverBase`): implement
  ``_clear_room_state``. ``send_metadata`` (the decode->prefill ready
  notification) and ``poll`` are shared.

``BootstrapServer``: subclass :class:`TpuKVBootstrapServerBase` (device-agnostic;
  usually an empty ``pass``).
"""

from __future__ import annotations

import abc
import logging
import threading
import time
from collections import defaultdict
from typing import cast

import numpy as np
import torch
import zmq

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.server_args import ServerArgs
from sglang_tpu.disaggregation.tpu import transfer as tpu_transfer
from sglang_tpu.disaggregation.tpu.host_staging import StageBuf
from sglang_tpu.disaggregation.tpu.serde import Transfer
from sglang_tpu.disaggregation.tpu.transfer import (
    DecodeTransferState,
    PendingChunk,
    PrefillTransferInfo,
)

logger = logging.getLogger(__name__)


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


class TpuKVManagerBase(CommonKVManager):
    """Shared manager scaffolding: pool attachment, the ZMQ recv-loop starter,
    and role dispatch. Subclasses provide the data-plane methods + role init."""

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: bool | None = False,
    ):
        # Concrete subclass name, used in logs, error messages, thread names.
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

    # ---- pool attachment (shared) ---------------------------------------

    def attach_kv_pool(self, kv_pool: KVCache) -> None:
        """Give the manager the concrete per-layer KV tensors it will transfer.
        Extracts the block-indexed per-layer buffers (shared), then defers
        backend-specific setup to ``_on_pool_attached``. Called once at scheduler
        init."""
        if self.kv_pool is not None:
            raise RuntimeError(
                f"{self._name}: a KV pool is already attached. "
                "A second attach is unexpected."
            )
        self.kv_pool = kv_pool
        self.layer_buffers = tpu_transfer.get_layer_page_buffers(kv_pool)
        self._on_pool_attached()

    def _on_pool_attached(self) -> None:
        """Hook: build any backend transfer state from ``self.layer_buffers``.
        Default just logs the attachment."""
        logger.info(
            "%s attached KV pool %s: %d layer buffers",
            self._name,
            type(self.kv_pool).__name__,
            len(self.layer_buffers) if self.layer_buffers else 0,
        )

    def _assert_layer_page_buffers(self):
        if self.layer_buffers is None:
            raise RuntimeError(
                f"{self._name}.attach_kv_pool() was never called; the "
                "KV pool tensors are required for TPU P/D transfer."
            )

    # ---- recv-loop scaffolding (shared) ---------------------------------

    def _serve(self, expected_tag: bytes, handler, copy: bool = True) -> None:
        """Start a daemon thread that pulls multipart messages off
        ``server_socket``, checks the leading tag, and dispatches to
        ``handler``. Backend recv loops differ only in (tag, handler).

        ``copy=False`` receives zero-copy ``zmq.Frame``s (memoryviews into ZMQ's
        own buffers) instead of materialising ``bytes`` — the handler then copies
        each frame straight into its destination, saving one host copy. The tag
        is read via ``bytes(msg[0])`` so the dispatch works for either form."""
        name = self._name

        def loop():
            while True:
                try:
                    msg = self.server_socket.recv_multipart(copy=copy)
                except zmq.ContextTerminated:
                    logger.warning("%s recv loop ending: ZMQ context terminated", name)
                    return
                except zmq.Again:
                    continue
                except zmq.ZMQError as e:
                    if e.errno in (zmq.ETERM, zmq.ENOTSOCK, zmq.EBADF):
                        logger.warning("%s recv loop ending: %s", name, e)
                        return
                    logger.exception("%s unexpected ZMQ exception", name)
                    continue
                except Exception as e:  # pragma: no cover - socket teardown
                    logger.exception("%s unexpected recv exception: %s", name, e)
                    continue
                if not msg or bytes(msg[0]) != expected_tag:
                    logger.warning("%s recv: unexpected msg tag %r", name, msg[:1])
                    continue
                try:
                    handler(msg)
                except Exception:
                    logger.exception(
                        "%s: error handling %r message", name, expected_tag
                    )

        threading.Thread(
            target=loop,
            name=f"{name}-{self.disaggregation_mode.value}",
            daemon=True,
        ).start()

    def _serve_router(self, handlers: dict, copy: bool = True) -> None:
        """Like :meth:`_serve` but dispatches by leading tag to one of several
        handlers (``{tag: fn}``). Used when a role accepts more than one message
        type on its socket (e.g. single- and multi-channel KV frames).

        ``copy=False`` receives zero-copy ``zmq.Frame``s (see :meth:`_serve`);
        the tag is read via ``bytes(msg[0])`` so dispatch works either way."""
        name = self._name

        def loop():
            while True:
                try:
                    msg = self.server_socket.recv_multipart(copy=copy)
                except zmq.ContextTerminated:
                    logger.warning("%s recv loop ending: ZMQ context terminated", name)
                    return
                except zmq.Again:
                    continue
                except zmq.ZMQError as e:
                    if e.errno in (zmq.ETERM, zmq.ENOTSOCK, zmq.EBADF):
                        logger.warning("%s recv loop ending: %s", name, e)
                        return
                    logger.exception("%s unexpected ZMQ exception", name)
                    continue
                except Exception as e:  # pragma: no cover - socket teardown
                    logger.exception("%s unexpected recv exception: %s", name, e)
                    continue
                handler = handlers.get(bytes(msg[0])) if msg else None
                if handler is None:
                    logger.warning("%s recv: unexpected msg tag %r", name, msg[:1])
                    continue
                try:
                    handler(msg)
                except Exception:
                    logger.exception("%s: error handling %r", name, msg[0])

        threading.Thread(
            target=loop,
            name=f"{name}-{self.disaggregation_mode.value}",
            daemon=True,
        ).start()

    def _handle_transfer(self, msg: list[bytes]) -> None:
        """Prefill side: a decode signalled it is ready for room R. Record the
        decode's control endpoint and advance to WaitingForInput once every
        expected decode rank has reported. Shared across backends; subclasses
        choose what value to stash per endpoint via ``_record_decode_endpoint``."""
        req = Transfer.from_frames(msg)
        room = req.room
        decode_ip = req.decode_ip
        decode_port = req.decode_port
        num_pages = req.num_pages
        required = req.required_dst_info_num
        prefix_len = req.decode_prefix_len
        dst_tp_rank = req.dst_tp_rank
        decode_attn_tp_size = req.decode_attn_tp_size

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

    # ---- abstract role + data-plane seam --------------------------------

    def _init_prefill(self) -> None:
        raise NotImplementedError

    def _init_decode(self) -> None:
        raise NotImplementedError

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
            logger.warning("decode KV data for unknown room %s, dropping", room)
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
            if self.layer_buffers and self.layer_buffers[0].device.type != "cpu":
                tpu_transfer.scatter_stage_from_host(
                    self.layer_buffers,
                    dst_pages,
                    chunk.layer_blobs,
                    pool,
                    dst_lane_slice,
                )
            else:
                tpu_transfer.scatter_pages_from_host(
                    self.layer_buffers,
                    dst_pages,
                    chunk.layer_blobs,
                    dst_lane_slice,
                )
            self._scatter_stats.record(time.perf_counter() - _t0, _nbytes)
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

    @staticmethod
    def _release_staged(chunk: PendingChunk) -> None:
        """Return any pooled StageBufs held by a chunk."""
        for b in chunk.layer_blobs:
            if isinstance(b, StageBuf):
                b.release()

    def _expected_chunk_bytes(self, num_pages: int, num_lanes: int) -> int:
        """Total expected bytes for a chunk."""
        total = 0
        if self.layer_buffers:
            for buf in self.layer_buffers:
                page_size = buf.shape[1]
                per_lane = int(np.prod(buf.shape[3:])) * buf.element_size()
                total += num_pages * page_size * num_lanes * per_lane
        return total

    @abc.abstractmethod
    def gather_and_push(
        self,
        room: int,
        page_indices: np.ndarray,
        index_start: int,
        is_last_chunk: bool,
        aux_index: int | None,
    ) -> None:
        """Gather pages from room and push."""


class TpuKVSenderBase(CommonKVSender):
    """Prefill-side handle. Shared: chunk gating, poll/timeout, failure. The
    subclass implements ``send`` (and optionally ``_pre_poll``)."""

    def __init__(
        self,
        mgr: TpuKVManagerBase,
        bootstrap_addr: str,
        bootstrap_room: int,
        req_has_disagg_prefill_dp_rank: bool = False,
    ):
        super().__init__(
            mgr,
            bootstrap_addr,
            bootstrap_room,
            req_has_disagg_prefill_dp_rank=req_has_disagg_prefill_dp_rank,
        )
        # Downcast to TpuKvManagerBase to support typing.
        self.kv_mgr = cast(TpuKVManagerBase, self.kv_mgr)
        self.conclude_state = None
        self.init_time = time.time()
        self._transfer_start_time: float | None = None

    def _record_transfer_start(self) -> None:
        if self._transfer_start_time is None:
            self._transfer_start_time = time.perf_counter()

    def pop_decode_prefix_len(self) -> int:
        return self.kv_mgr.req_to_decode_prefix_len.pop(self.bootstrap_room, 0)

    def should_send_kv_chunk(self, num_pages: int, last_chunk: bool) -> bool:
        return num_pages > 0 or last_chunk

    def send(
        self,
        kv_indices,
        state_indices: list | None = None,
        num_kv_tokens: int | None = None,
    ):
        raise NotImplementedError

    def _pre_poll(self) -> None:
        """Hook: advance the data plane before reading status (e.g. drain a
        pull engine and mark Success/Failed). Default no-op — the host backend
        marks Success from its background push thread."""

    def poll(self) -> KVPoll:
        return self._poll_with_status(self.kv_mgr.check_status)

    def poll_pp_consensus(self) -> KVPoll:
        # PP stage 0 alone polls a bootstrapping room on a pipeline-parallel
        # prefill, and every stage starts it from _handle_transfer. Reading the
        # least advanced stage keeps the bootstrap deadline on the whole room.
        return self._poll_with_status(self.kv_mgr.check_status_pp_consensus)

    def _poll_with_status(self, check_status) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state
        self._pre_poll()
        status = check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            if (
                status == KVPoll.Success
                and self._transfer_start_time is not None
                and self._transfer_metric.transfer_latency_s is None
            ):
                self._transfer_metric.transfer_latency_s = (
                    time.perf_counter() - self._transfer_start_time
                )
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


class TpuKVReceiverBase(CommonKVReceiver):
    """Decode-side handle. Shared: the decode->prefill ready notification
    (``send_metadata``), poll/timeout, failure. The subclass implements
    ``_clear_room_state`` and the manager's ``process_decode_pending`` does the
    backend-specific pull/scatter."""

    def __init__(
        self,
        mgr: TpuKVManagerBase,
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
                    Transfer(
                        room=self.bootstrap_room,
                        decode_ip=self.kv_mgr.local_ip,
                        decode_port=self.kv_mgr.rank_port,
                        num_pages=num_pages,
                        required_dst_info_num=self.required_dst_info_num,
                        decode_prefix_len=decode_prefix_len or 0,
                        dst_tp_rank=dst_tp_rank,
                        decode_attn_tp_size=dst_attn_tp_size,
                    ).to_frames()
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
        """Hook: drop this room's decode-side bookkeeping. Default no-op."""

    def failure_exception(self):
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed
        self.clear()
        with self.kv_mgr.failure_lock:
            reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "Failed due to an unknown reason"
            )
        raise KVTransferError(self.bootstrap_room, reason)


class TpuKVBootstrapServerBase(CommonKVBootstrapServer):
    """Bootstrap (handshake/discovery) HTTP server — fully device-agnostic,
    inherited unchanged from the common implementation by every backend."""
