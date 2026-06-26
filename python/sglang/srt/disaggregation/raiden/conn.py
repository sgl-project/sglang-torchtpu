"""TPU prefill/decode disaggregation backend backed by tpu-raiden.

Raiden moves KV pages directly between TPU hosts (D2H -> network -> H2D) and
tracks transfer completion itself. SGLang only provides rendezvous: the decode
worker tells the target prefill worker "I want room R, reply to <my endpoint>",
the prefill worker replies with its page block-ids + raiden endpoint (and the
small aux/first-token blob), and the decode worker then issues a raiden pull.

This is the reverse direction from the push-based RDMA backends (mooncake/nixl):
raiden is consumer-pull, so the per-request block-ids flow prefill -> decode
(raiden's ``start_read`` needs the remote block-ids on the consumer side). The
existing ZMQ rank sockets and HTTP bootstrap server from ``CommonKVManager`` are
reused unchanged for the rendezvous.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
from sglang.srt.disaggregation.raiden.serde import (
    BLOCKS,
    PULL_REQ,
    pack_blocks,
    read_aux_blob,
    unpack_blocks,
    write_aux_blob,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def get_raiden_kv_caches(pool) -> list:
    """Per-layer device KV tensors raiden indexes by page (major dim).

    The TPU fused KV pool stores one [num_blocks, page_size, ...] tensor per
    layer in ``kv_buffer`` and exposes them via ``get_fused_kv_buffer``.
    """
    if hasattr(pool, "get_fused_kv_buffer"):
        start = pool.start_layer
        return [pool.get_fused_kv_buffer(start + i) for i in range(pool.layer_num)]
    if hasattr(pool, "kv_buffer"):
        return list(pool.kv_buffer)
    raise RuntimeError(
        f"Raiden backend requires a fused KV pool exposing kv_buffer; got "
        f"{type(pool).__name__}"
    )


class RaidenKVManager(CommonKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)

        self._engine: Optional[Any] = None  # raiden KVCacheManager
        self._kv_caches: Optional[list] = None
        self.raiden_endpoint: Optional[str] = None
        # Deterministic control port per attn-tp rank (data port is +1). Matches
        # the torchtpu-vllm scheme; reply carries it so decode needs no extra
        # bootstrap field.
        base_port = int(os.environ.get("SGLANG_RAIDEN_KV_PORT", "30000"))
        self._control_port = base_port + 2 * self.attn_tp_rank
        self._timeout_s = float(os.environ.get("SGLANG_RAIDEN_TIMEOUT_S", "120"))

        # Completion sets, drained from raiden and keyed by str(room). Persist
        # until clear() so repeated poll() calls before the framework advances
        # keep reporting the same terminal state.
        self._completed_send: Set[str] = set()
        self._completed_recv: Set[str] = set()
        self._failed: Set[str] = set()
        self._engine_lock = threading.Lock()
        self._push_lock = threading.Lock()
        self._state_lock = threading.Lock()

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # room -> accumulated page block-ids, aux blob, decode reply addr
            self._send_blocks: Dict[int, List[int]] = {}
            self._send_aux: Dict[int, bytes] = {}
            self._decode_addr: Dict[int, Tuple[str, int]] = {}
            self._replied: Set[int] = set()
        else:
            # room -> local page block-ids, remote page block-ids, endpoint
            self._local_blocks: Dict[int, List[int]] = {}
            self._decode_aux_index: Dict[int, int] = {}
            self._remote_blocks: Dict[int, List[int]] = {}
            self._remote_endpoint: Dict[int, str] = {}

        self._listener = threading.Thread(target=self._listen, daemon=True)
        self._listener.start()

    # ----- engine lifecycle -------------------------------------------------

    def set_kv_caches(self, kv_caches: list) -> None:
        """Hand raiden the per-layer device KV tensors and build the engine.

        Called by prefill.py/decode.py right after construction (the KV pool is
        only available there). Raiden indexes each tensor's major dim as a
        block/page, which matches the TPU fused KV pool layout
        ([num_blocks, page_size, ...]).
        """
        self._kv_caches = list(kv_caches)
        self._build_engine()

    def _build_engine(self) -> None:
        if self._engine is not None or self._kv_caches is None:
            return
        # TPU-only import, deferred so the disaggregation module imports cleanly
        # on hosts without tpu_raiden. Raiden owns the transport, host staging,
        # D2H/H2D DMA, and completion tracking; it indexes each KV tensor's major
        # dim as a page, matching the TPU fused KV pool ([num_blocks, page, ...]).
        from tpu_raiden.api.torch.kv_cache_manager import KVCacheManager

        max_blocks = self._max_request_blocks()
        num_slots = self._num_slots()
        self.raiden_endpoint = f"{self.local_ip}:{self._control_port}"
        self._engine = KVCacheManager(
            kv_caches=self._kv_caches,
            node_id=self.attn_tp_rank,
            local_control_port=self._control_port,
            max_blocks=max_blocks,
            num_slots=num_slots,
            timeout_s=self._timeout_s,
        )
        logger.info(
            "RaidenKVManager rank%d engine up | endpoint=%s max_blocks=%d "
            "num_slots=%d",
            self.attn_tp_rank,
            self.raiden_endpoint,
            max_blocks,
            num_slots,
        )

    def _max_request_blocks(self) -> int:
        override = int(os.environ.get("SGLANG_RAIDEN_MAX_BLOCKS", "0"))
        if override > 0:
            return override
        max_len = self.server_args.context_length or int(
            os.environ.get("SGLANG_RAIDEN_MAX_MODEL_LEN", "8192")
        )
        return max(1, math.ceil(max_len / self.kv_args.page_size))

    def _num_slots(self) -> int:
        return max(1, int(os.environ.get("SGLANG_RAIDEN_NUM_SLOTS", "16")))

    def _ensure_engine(self) -> Any:
        if self._engine is None:
            with self._engine_lock:
                self._build_engine()
        if self._engine is None:
            raise RuntimeError(
                "Raiden engine not initialized: set_kv_caches() must be called "
                "with the KV pool tensors before any transfer."
            )
        return self._engine

    def _drain_engine(self) -> None:
        if self._engine is None:
            return
        done_send, done_recv, failed = self._engine.poll_stats()
        if done_send or done_recv or failed:
            with self._state_lock:
                self._completed_send.update(done_send)
                self._completed_recv.update(done_recv)
                self._failed.update(failed)
            if failed:
                logger.error(
                    "RaidenKVManager rank%d failed=%s", self.attn_tp_rank, failed
                )

    def register_to_bootstrap(self):
        # Topology/handshake registration is backend-agnostic; raiden adds no
        # new bootstrap fields (its endpoint rides in the per-request reply).
        super().register_to_bootstrap()

    # ----- rendezvous (ZMQ over the existing per-rank PULL socket) -----------

    def _listen(self) -> None:
        while True:
            try:
                msg = self.server_socket.recv_multipart()
            except Exception as e:
                logger.error("RaidenKVManager listener error: %s", e)
                continue
            try:
                if msg[0] == PULL_REQ:
                    self._on_pull_req(msg)
                elif msg[0] == BLOCKS:
                    self._on_blocks(msg)
                else:
                    logger.warning("RaidenKVManager unknown header: %s", msg[0])
            except Exception as e:
                logger.error("RaidenKVManager message handling error: %s", e)

    def _on_pull_req(self, msg: List[bytes]) -> None:
        room = int(msg[1].decode("ascii"))
        decode_ip = msg[2].decode("ascii")
        decode_port = int(msg[3].decode("ascii"))
        with self._state_lock:
            self._decode_addr[room] = (decode_ip, decode_port)
        self._maybe_reply(room)

    def _on_blocks(self, msg: List[bytes]) -> None:
        room = int(msg[1].decode("ascii"))
        remote_blocks = unpack_blocks(msg[2])
        endpoint = msg[3].decode("ascii")
        aux_blob = msg[4]
        aux_index = None
        with self._state_lock:
            self._remote_blocks[room] = remote_blocks
            self._remote_endpoint[room] = endpoint
            aux_index = self._decode_aux_index.get(room)
        if aux_index is not None and aux_blob:
            write_aux_blob(
                self.kv_args.aux_data_ptrs,
                self.kv_args.aux_item_lens,
                aux_index,
                aux_blob,
            )

    # ----- prefill side ------------------------------------------------------

    def append_send_blocks(self, room: int, block_ids: List[int]) -> None:
        with self._state_lock:
            self._send_blocks.setdefault(room, []).extend(block_ids)

    def finalize_send(self, room: int, aux_index: Optional[int]) -> None:
        engine = self._ensure_engine()
        with self._state_lock:
            blocks = list(self._send_blocks.get(room, []))
            if aux_index is not None:
                self._send_aux[room] = read_aux_blob(
                    self.kv_args.aux_data_ptrs,
                    self.kv_args.aux_item_lens,
                    aux_index,
                )
        engine.register_read(str(room), room, blocks)
        self._maybe_reply(room)

    def _maybe_reply(self, room: int) -> None:
        with self._state_lock:
            if room in self._replied:
                return
            if room not in self._send_blocks or room not in self._decode_addr:
                return
            blocks = list(self._send_blocks[room])
            addr = self._decode_addr[room]
            aux_blob = self._send_aux.get(room, b"")
            self._replied.add(room)
        endpoint = f"tcp://{addr[0]}:{addr[1]}"
        payload = [
            BLOCKS,
            str(room).encode("ascii"),
            pack_blocks(blocks),
            (self.raiden_endpoint or "").encode("ascii"),
            aux_blob,
        ]
        with self._push_lock:
            self._connect(endpoint).send_multipart(payload)

    # ----- decode side -------------------------------------------------------

    def register_decode_request(
        self,
        room: int,
        local_blocks: List[int],
        aux_index: Optional[int],
        bootstrap_infos: List[dict],
    ) -> None:
        if len(bootstrap_infos) != 1:
            # v1 supports symmetric TP only (decode rank N pulls from the single
            # prefill rank N). Heterogeneous TP needs one raiden pull per source
            # rank; fail loudly rather than silently transfer a subset.
            raise NotImplementedError(
                "Raiden disaggregation v1 requires equal prefill/decode attn TP "
                f"size (got {len(bootstrap_infos)} prefill sources for one decode "
                "rank)."
            )
        with self._state_lock:
            self._local_blocks[room] = local_blocks
            if aux_index is not None:
                self._decode_aux_index[room] = aux_index
        payload = [
            PULL_REQ,
            str(room).encode("ascii"),
            self.local_ip.encode("ascii"),
            str(self.rank_port).encode("ascii"),
        ]
        for info in bootstrap_infos:
            endpoint = f"tcp://{info['rank_ip']}:{info['rank_port']}"
            with self._push_lock:
                self._connect(endpoint).send_multipart(payload)

    def maybe_start_pull(self, room: int) -> bool:
        """Issue the raiden pull once the prefill reply has arrived.

        Returns True if the pull was (or had already been) issued.
        """
        with self._state_lock:
            if room not in self._remote_blocks or room not in self._remote_endpoint:
                return False
            local = list(self._local_blocks.get(room, []))
            remote = list(self._remote_blocks[room])
            endpoint = self._remote_endpoint[room]
        # Pull the prefill KV pages that map onto decode's freshly allocated
        # local pages; raiden requires remote/local to be equal length.
        #   - empty local => full prefix-cache hit; pull nothing (avoid
        #     remote[-0:], which would select the whole list).
        #   - len(remote) > len(local) => decode already holds the head prefix;
        #     pull only the tail len(local) prefill pages.
        #   - len(remote) < len(local) => decode allocated a trailing page for
        #     the first generated token that prefill produced no KV for; fill the
        #     head len(remote) local pages and leave that page out of the pull.
        if not local:
            remote = []
        elif len(remote) > len(local):
            remote = remote[-len(local) :]
        elif len(remote) < len(local):
            local = local[: len(remote)]
        engine = self._ensure_engine()
        engine.start_read(str(room), room, endpoint, remote, local)
        return True

    # ----- completion queries (drained by poll()) ---------------------------

    def is_send_done(self, room: int) -> bool:
        return str(room) in self._completed_send

    def is_recv_done(self, room: int) -> bool:
        return str(room) in self._completed_recv

    def is_failed(self, room: int) -> bool:
        return str(room) in self._failed

    def clear_room(self, room: int) -> None:
        rid = str(room)
        with self._state_lock:
            self._completed_send.discard(rid)
            self._completed_recv.discard(rid)
            self._failed.discard(rid)
            for d in (
                getattr(self, "_send_blocks", None),
                getattr(self, "_send_aux", None),
                getattr(self, "_decode_addr", None),
                getattr(self, "_local_blocks", None),
                getattr(self, "_decode_aux_index", None),
                getattr(self, "_remote_blocks", None),
                getattr(self, "_remote_endpoint", None),
            ):
                if d is not None:
                    d.pop(room, None)
            if hasattr(self, "_replied"):
                self._replied.discard(room)


class RaidenKVSender(CommonKVSender):
    def __init__(
        self,
        mgr: RaidenKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank)
        self._last_chunk = False
        self._started = False
        # Raiden is consumer-pull: the prefill need not wait for any decode
        # handshake before computing (decode pulls once prefill registers its
        # blocks). Advance straight to WaitingForInput so the bootstrap queue
        # releases this request into prefill. Don't override a Failed/dummy state
        # already set by CommonKVSender.__init__.
        if (
            not self.kv_mgr.is_dummy_cp_rank
            and self.kv_mgr.check_status(self.bootstrap_room) == KVPoll.Bootstrapping
        ):
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        super().init(num_kv_indices, aux_index)

    def should_send_kv_chunk(self, num_pages: int, last_chunk: bool) -> bool:
        self._last_chunk = last_chunk
        return num_pages > 0 or last_chunk

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List] = None,
    ):
        if self.kv_mgr.is_dummy_cp_rank:
            return
        self._record_transfer_indices(kv_indices, state_indices)
        self.kv_mgr.append_send_blocks(
            self.bootstrap_room, [int(x) for x in kv_indices]
        )
        if self._last_chunk and not self._started:
            self._started = True
            self.kv_mgr.finalize_send(self.bootstrap_room, self.aux_index)
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Transferring)

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state
        if self.kv_mgr.is_dummy_cp_rank:
            self.conclude_state = KVPoll.Success
            return KVPoll.Success
        self.kv_mgr._drain_engine()
        if self.kv_mgr.is_failed(self.bootstrap_room):
            self.conclude_state = KVPoll.Failed
            return KVPoll.Failed
        if self.kv_mgr.is_send_done(self.bootstrap_room):
            self.conclude_state = KVPoll.Success
            return KVPoll.Success
        return self.kv_mgr.check_status(self.bootstrap_room)

    def failure_exception(self):
        reason = self.kv_mgr.failure_records.get(
            self.bootstrap_room, "Raiden send failed"
        )
        raise Exception(reason)

    def clear(self) -> None:
        super().clear()
        self.kv_mgr.clear_room(self.bootstrap_room)


class RaidenKVReceiver(CommonKVReceiver):
    def __init__(
        self,
        mgr: RaidenKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room)
        self._started = False

    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
        decode_prefix_len: Optional[int] = None,
    ):
        if self.conclude_state == KVPoll.Failed:
            return
        self.kv_mgr.register_decode_request(
            self.bootstrap_room,
            [int(x) for x in kv_indices],
            aux_index,
            self.bootstrap_infos,
        )
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state
        self.kv_mgr._drain_engine()
        if self.kv_mgr.is_failed(self.bootstrap_room):
            self.conclude_state = KVPoll.Failed
            return KVPoll.Failed
        if self.kv_mgr.is_recv_done(self.bootstrap_room):
            self.conclude_state = KVPoll.Success
            return KVPoll.Success
        if not self._started and self.kv_mgr.maybe_start_pull(self.bootstrap_room):
            self._started = True
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Transferring)
        return self.kv_mgr.check_status(self.bootstrap_room)

    def failure_exception(self):
        reason = self.kv_mgr.failure_records.get(
            self.bootstrap_room, "Raiden receive failed"
        )
        raise Exception(reason)

    def clear(self) -> None:
        super().clear()
        self.kv_mgr.clear_room(self.bootstrap_room)


class RaidenKVBootstrapServer(CommonKVBootstrapServer):
    pass
