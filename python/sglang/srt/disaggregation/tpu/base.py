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
* an opt-in ``tpu_raiden`` backend (a future ``raiden_conn.py``) hands the
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

import logging
import threading
import time
from collections import defaultdict
from typing import List, Optional

import numpy as np

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
from sglang.srt.disaggregation.tpu import transfer as tpu_transfer
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# The decode -> prefill "room R ready; pull from my zmq endpoint" notification.
# Identical across backends (it is pure control), so it lives in the base; each
# backend defines its own *data* tag (e.g. KVDATA / KVMETA) in its own module.
TAG_TRSFR = b"TRSFR"


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


def _resolve_total_kv_heads(kv_args: KVArgs, attn_tp_size: int) -> int:
    """Global KV-head count. Prefer the explicit ``total_kv_head_num`` (set by
    prefill.py / decode.py); fall back to ``kv_head_num * attn_tp_size`` (per-rank
    x TP). Mirrors ``common/staging_buffer.resolve_total_kv_heads`` but avoids the
    dependency for a one-liner."""
    total = getattr(kv_args, "total_kv_head_num", 0) or 0
    if total > 0:
        return total
    return int(getattr(kv_args, "kv_head_num", 0) or 0) * max(1, attn_tp_size)


class TpuKVManagerBase(CommonKVManager):
    """Shared manager scaffolding: pool attachment, the ZMQ recv-loop starter,
    and role dispatch. Subclasses provide the data-plane methods + role init."""

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        # Concrete subclass name, used in logs, error messages, thread names.
        self._name = type(self).__name__
        if is_mla_backend:
            raise NotImplementedError(f"{self._name} does not support MLA models yet.")
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)

        # Filled in later by the scheduler via attach_kv_pool(); the bootstrap
        # queue owns the pool object and only it knows the concrete tensors.
        self.kv_pool = None
        self.layer_buffers: Optional[List] = None
        self.layer_specs = None

        # Per-endpoint send locks (a single zmq PUSH socket must not be used
        # concurrently from multiple sender threads).
        self._endpoint_locks = defaultdict(threading.Lock)

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

    # ---- pool attachment (shared) ---------------------------------------

    def attach_kv_pool(self, kv_pool) -> None:
        """Give the manager the concrete per-layer KV tensors it will transfer.
        Extracts the block-indexed per-layer buffers + their dtype/shape specs
        (shared), then defers backend-specific setup to ``_on_pool_attached``.
        Called once at scheduler init."""
        if self.kv_pool is not None:
            raise RuntimeError(
                f"{self._name}: a KV pool is already attached. "
                "A second attach is unexpected."
            )
        self.kv_pool = kv_pool
        self.layer_buffers = tpu_transfer.get_layer_page_buffers(kv_pool)
        self.layer_specs = tpu_transfer.layer_page_specs(self.layer_buffers)
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

    def _require_pool(self):
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
                except Exception as e:  # pragma: no cover - socket teardown
                    logger.warning("%s recv loop ending: %s", name, e)
                    return
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
                except Exception as e:  # pragma: no cover - socket teardown
                    logger.warning("%s recv loop ending: %s", name, e)
                    return
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

    def _handle_transfer(self, msg: List[bytes]) -> None:
        """Prefill side: a decode signalled it is ready for room R. Record the
        decode's control endpoint and advance to WaitingForInput once every
        expected decode rank has reported. Shared across backends; subclasses
        choose what value to stash per endpoint via ``_record_decode_endpoint``."""
        # [TAG, room, decode_ip, decode_port, aux_present, num_pages,
        #  required_dst_info_num, decode_prefix_len,
        #  dst_tp_rank, decode_attn_tp_size, total_kv_heads]
        if len(msg) != 11:
            raise ValueError(
                f"Malformed {TAG_TRSFR.decode()} message. "
                f"Expected 11 frames, got {len(msg)}."
            )
        room = int(msg[1].decode())
        decode_ip = msg[2].decode()
        decode_port = int(msg[3].decode())
        num_pages = int(msg[5].decode())
        required = int(msg[6].decode())
        prefix_len = int(msg[7].decode())
        dst_tp_rank = int(msg[8].decode())
        decode_attn_tp_size = int(msg[9].decode())
        total_kv_heads = int(msg[10].decode())

        logger.info(
            "[pd_kv_resharding] _handle_transfer room=%s endpoint=%s:%s num_pages=%d "
            "required=%d dst_tp_rank=%d decode_attn_tp=%d total_kv_heads=%d",
            room,
            decode_ip,
            decode_port,
            num_pages,
            required,
            dst_tp_rank,
            decode_attn_tp_size,
            total_kv_heads,
        )
        endpoint = _tcp_endpoint(decode_ip, decode_port)
        bucket = self.transfer_infos.setdefault(room, {})
        bucket[endpoint] = self._record_decode_endpoint(
            room, endpoint, num_pages, dst_tp_rank, decode_attn_tp_size, total_kv_heads
        )
        self.req_to_decode_prefix_len[room] = prefix_len
        if len(bucket) >= required:
            self.update_status(room, KVPoll.WaitingForInput)

    def _record_decode_endpoint(
        self,
        room: int,
        endpoint: str,
        num_pages: int,
        dst_tp_rank: int = 0,
        decode_attn_tp_size: int = 1,
        total_kv_heads: int = 0,
    ):
        """Hook: value to store in ``transfer_infos[room][endpoint]``. Default is
        the endpoint string itself; a backend may store a richer record (the
        ``tpu`` backend stashes the decode's TP identity for resharding)."""
        return endpoint

    # ---- abstract role + data-plane seam --------------------------------

    def _init_prefill(self) -> None:
        raise NotImplementedError

    def _init_decode(self) -> None:
        raise NotImplementedError

    def register_decode_state(
        self, room: int, dst_indices: np.ndarray, aux_index: Optional[int]
    ) -> None:
        raise NotImplementedError

    def process_decode_pending(self, room: int) -> None:
        raise NotImplementedError


class TpuKVSenderBase(CommonKVSender):
    """Prefill-side handle. Shared: chunk gating, poll/timeout, failure. The
    subclass implements ``send`` (and optionally ``_pre_poll``)."""

    def __init__(
        self,
        mgr: TpuKVManagerBase,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank)
        self.conclude_state = None
        self.init_time = time.time()

    def pop_decode_prefix_len(self) -> int:
        return self.kv_mgr.req_to_decode_prefix_len.pop(self.bootstrap_room, 0)

    def should_send_kv_chunk(self, num_pages: int, last_chunk: bool) -> bool:
        return num_pages > 0 or last_chunk

    def send(self, kv_indices, state_indices: Optional[List] = None):
        raise NotImplementedError

    def _pre_poll(self) -> None:
        """Hook: advance the data plane before reading status (e.g. drain a
        pull engine and mark Success/Failed). Default no-op — the host backend
        marks Success from its background push thread."""

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


class TpuKVReceiverBase(CommonKVReceiver):
    """Decode-side handle. Shared: the decode->prefill ready notification
    (``send_metadata``), poll/timeout, failure. The subclass implements
    ``_clear_room_state`` and the manager's ``process_decode_pending`` does the
    backend-specific pull/scatter."""

    def __init__(
        self,
        mgr: TpuKVManagerBase,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        self.init_time = None
        super().__init__(mgr, bootstrap_addr, bootstrap_room)

    def send_metadata(
        self,
        kv_indices,
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        decode_prefix_len: Optional[int] = None,
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

        # Resharding metadata (MHA heterogeneous attn_tp): the prefill needs the
        # decode's TP identity to compute which KV-head slice to send this rank
        # (see transfer.compute_lane_slice / conn.gather_and_push). total_kv_heads
        # is echoed for a prefill-side consistency check.
        dst_attn_tp_size = self.kv_mgr.attn_tp_size
        dst_tp_rank = self.kv_mgr.kv_args.engine_rank % max(1, dst_attn_tp_size)
        total_kv_heads = _resolve_total_kv_heads(self.kv_mgr.kv_args, dst_attn_tp_size)

        num_pages = len(kv_indices)
        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            with lock:
                sock.send_multipart(
                    [
                        TAG_TRSFR,
                        str(self.bootstrap_room).encode(),
                        self.kv_mgr.local_ip.encode(),
                        str(self.kv_mgr.rank_port).encode(),
                        b"1" if aux_index is not None else b"0",
                        str(num_pages).encode(),
                        str(self.required_dst_info_num).encode(),
                        str(decode_prefix_len or 0).encode(),
                        str(dst_tp_rank).encode(),
                        str(dst_attn_tp_size).encode(),
                        str(total_kv_heads).encode(),
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

    pass
