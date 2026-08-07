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

When decode TP < prefill TP (e.g. 8TP->4TP2DP), a decode rank's kv-head slice
spans several prefill ranks, so a whole-block pull cannot place them.
That case switches to raiden's strided push-resharding. Each prefill source
stages its blocks device->host, then pushes its head-slice strided
into the decode block's sub-region; the decode receiver auto-H2Ds the assembled
host staging into HBM. Schedules are built in ``reshard_schedule.py``.
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import os
import threading
import time
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
    MessageTag,
    pack_blocks,
    pack_ints,
    read_aux_blob,
    unpack_blocks,
    unpack_ints,
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
    """KV manager for the raiden prefill/decode transport.

    FIXME: the ZMQ rendezvous scaffolding here (per-rank listener socket,
    handshake message routing, poll draining) overlaps the tpu backend's
    ``base.py`` control plane. Consolidate onto a shared base so raiden reuses it
    instead of reimplementing the coordination.
    """

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
        # Deterministic control port per (dp, tp) rank (data port is
        # +1).
        base_port = int(os.environ.get("SGLANG_RAIDEN_KV_PORT", "30000"))
        self._local_rank = self.attn_dp_rank * self.attn_tp_size + self.attn_tp_rank
        self._control_port = base_port + 2 * self._local_rank
        self._timeout_s = float(os.environ.get("SGLANG_RAIDEN_TIMEOUT_S", "120"))
        # Strided push-resharding for asymmetric TP (decode TP < prefill TP),
        # on by default and auto-engaged per request only when a
        # decode rank is fed by several prefill ranks; symmetric traffic never
        # enters this path. Set SGLANG_RAIDEN_RESHARD=0 to disable it (an
        # asymmetric topology then raises) and keep the multi-slot host staging
        # pool instead of the device-shadow sizing the reshard receiver needs.
        self._reshard_enabled = bool(int(os.environ.get("SGLANG_RAIDEN_RESHARD", "1")))

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

        # Asymmetric-TP resharding state, populated lazily on the push path.
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # room -> decode meta from RESHARD_REQ
            self._reshard_req: Dict[int, dict] = {}
            self._reshard_finalized: Set[int] = set()  # send accumulation complete
            self._reshard_replied: Set[int] = set()  # RESHARD_BLOCKS sent
            self._reshard_ready: Set[int] = set()  # rooms cleared to push
            self._reshard_pushed: Set[int] = set()
            # Push (d2h wait + per-layer push_registered_plan) runs off the
            # _listen dispatch thread. raiden runs SyncPush unlocked, so
            # _reshard_push_lock serializes register+push across workers.
            self._push_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max(
                    1, int(os.environ.get("SGLANG_RAIDEN_PUSH_WORKERS", "1"))
                ),
                thread_name_prefix="raiden-reshard-push",
            )
            self._reshard_push_lock = threading.Lock()
        else:
            # room -> {src_prefill_rank: prefill_block_ids} and per-room meta.
            self._reshard_sources: Dict[int, Dict[int, List[int]]] = {}
            self._reshard_meta: Dict[int, dict] = {}
            self._reshard_registered: Set[int] = set()
        # Round-robin host staging slots for the push path (device->host stage on
        # prefill; host staging on decode is written by raiden's receiver).
        self._reshard_slot_rooms: List[Optional[int]] = [None] * max(
            1, int(os.environ.get("SGLANG_RAIDEN_NUM_SLOTS", "16"))
        )
        self._reshard_slot_of: Dict[int, int] = {}

        self._listener = threading.Thread(target=self._listen, daemon=True)
        self._listener.start()

    # ----- resharding geometry / host-slot helpers --------------------------

    def _make_geometry(self, src_attn_tp: int, dst_attn_tp: int):
        """ReshardGeometry for a (src_attn_tp -> dst_attn_tp) transfer, sized from
        the local fused KV tensor (the K+V head-pair granule matches on both
        sides) and the global kv-head count.
        """
        from sglang.srt.disaggregation.raiden.reshard_schedule import (
            ReshardGeometry,
            head_pair_bytes,
            per_rank_head_count,
        )

        tensor = self._kv_caches[0]
        hp = head_pair_bytes(tensor)
        # Distinct kv-head count. kv_args.total_kv_head_num is a bare annotation
        # that is not set on every KVArgs instance (the decode/receiver side
        # leaves it unset), so fall back to deriving it from the local per-rank
        # count. On the side where the fallback fires the per-rank count times the
        # local attn TP equals the true count (attn TP == kv-head count there).
        total_kv_heads = int(getattr(self.kv_args, "total_kv_head_num", 0) or 0)
        if total_kv_heads <= 0:
            total_kv_heads = per_rank_head_count(tensor) * self.attn_tp_size
        return ReshardGeometry(src_attn_tp, dst_attn_tp, hp, int(total_kv_heads))

    def _host_slot_blocks(self, room: int, n: int) -> List[int]:
        """Host staging block-ids for ``room`` (n blocks) in a round-robin slot.

        Slot ``s`` owns host blocks ``[s*max_blocks, s*max_blocks + n)``; reused
        once the room is cleared. Raises if all slots are in flight.
        """
        slot = self._reshard_slot_of.get(room)
        if slot is None:
            try:
                slot = self._reshard_slot_rooms.index(None)
            except ValueError:
                raise RuntimeError("raiden reshard: no free host staging slot")
            self._reshard_slot_rooms[slot] = room
            self._reshard_slot_of[room] = slot
        base = slot * self._max_request_blocks()
        return [base + i for i in range(n)]

    def _release_host_slot(self, room: int) -> None:
        slot = self._reshard_slot_of.pop(room, None)
        if slot is not None:
            self._reshard_slot_rooms[slot] = None

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
        # Reshard staging is indexed 1-to-1 by device block id. Size it to
        # the fused tensor's block count with a single slot.
        if self._reshard_enabled:
            num_blocks = int(self._kv_caches[0].shape[0])
            max_blocks = num_blocks
            num_slots = 1
        self.raiden_endpoint = f"{self.local_ip}:{self._control_port}"
        self._engine = KVCacheManager(
            kv_caches=self._kv_caches,
            node_id=self._local_rank,
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

    @staticmethod
    def _norm_req_id(rid: str) -> str:
        # Resharded (push) transfers surface as "resharded_transfer_<room>" from
        # poll_stats; strip the prefix so completion maps back to str(room).
        prefix = "resharded_transfer_"
        return rid[len(prefix) :] if rid.startswith(prefix) else rid

    def _drain_engine(self) -> None:
        if self._engine is None:
            return
        done_send, done_recv, failed = self._engine.poll_stats()
        if done_send or done_recv or failed:
            with self._state_lock:
                self._completed_send.update(self._norm_req_id(r) for r in done_send)
                self._completed_recv.update(self._norm_req_id(r) for r in done_recv)
                self._failed.update(self._norm_req_id(r) for r in failed)
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
                if msg[0] == MessageTag.PULL_REQ:
                    self._on_pull_req(msg)
                elif msg[0] == MessageTag.BLOCKS:
                    self._on_blocks(msg)
                elif msg[0] == MessageTag.RESHARD_REQ:
                    self._on_reshard_req(msg)
                elif msg[0] == MessageTag.RESHARD_BLOCKS:
                    self._on_reshard_blocks(msg)
                elif msg[0] == MessageTag.RESHARD_READY:
                    self._on_reshard_ready(msg)
                else:
                    logger.warning("RaidenKVManager unknown header: %s", msg[0])
            except Exception as e:
                logger.error("RaidenKVManager message handling error: %s", e)

    @staticmethod
    def _require_frames(msg: List[bytes], tag: MessageTag, n: int) -> None:
        """Reject a malformed header before it is indexed past its length."""
        if len(msg) != n:
            raise ValueError(
                f"malformed {tag.name} message: expected {n} frames, got {len(msg)}"
            )

    def _on_pull_req(self, msg: List[bytes]) -> None:
        self._require_frames(msg, MessageTag.PULL_REQ, 4)
        room = int(msg[1].decode("ascii"))
        decode_ip = msg[2].decode("ascii")
        decode_port = int(msg[3].decode("ascii"))
        with self._state_lock:
            self._decode_addr[room] = (decode_ip, decode_port)
        self._maybe_reply(room)

    def _on_blocks(self, msg: List[bytes]) -> None:
        self._require_frames(msg, MessageTag.BLOCKS, 5)
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
        # Resharded rooms reply over RESHARD_BLOCKS, never the pull path. If the
        # decode's RESHARD_REQ hasn't arrived yet this is a no-op; _on_reshard_req
        # drives the reply once both sides are present.
        if self._reshard_enabled:
            with self._state_lock:
                self._reshard_finalized.add(room)
            self._maybe_reshard_reply(room)
            return
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
            MessageTag.BLOCKS,
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
            # Asymmetric attention TP (decode attn TP < prefill attn TP): one
            # decode rank is fed by several prefill ranks, each pushing its
            # head-slice strided into the wider decode block.
            if self._reshard_enabled:
                if not local_blocks:
                    # Full prefix-cache hit: decode already holds every page,
                    # nothing to receive.
                    with self._state_lock:
                        self._completed_recv.add(str(room))
                    return
                self._register_reshard_request(
                    room, local_blocks, aux_index, bootstrap_infos
                )
                self.update_status(room, KVPoll.WaitingForInput)
                return
            raise NotImplementedError(
                "Raiden asymmetric attention TP (got "
                f"{len(bootstrap_infos)} prefill sources for one decode rank) "
                "needs strided push-resharding, but it is disabled "
                "(SGLANG_RAIDEN_RESHARD=0); unset it to enable."
            )
        with self._state_lock:
            self._local_blocks[room] = local_blocks
            if aux_index is not None:
                self._decode_aux_index[room] = aux_index
        payload = [
            MessageTag.PULL_REQ,
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
        # start_read needs equal-length, position-paired remote/local. Reconcile
        # the two independent ways they can differ:
        #   - decode prefix-cache hit: decode already holds the head pages and
        #     allocated fewer local pages, so pull only the matching tail of
        #     remote (remote[-len(local):]).
        #   - reserved decode token: decode allocated one extra trailing page that
        #     prefill produced no KV for, so drop it from local.
        # Empty local => full prefix hit; pull nothing (and avoid remote[-0:],
        # which would select the whole list).
        if not local:
            remote = []
        elif len(remote) > len(local):
            remote = remote[-len(local) :]
        elif len(remote) < len(local):
            local = local[: len(remote)]
        engine = self._ensure_engine()
        engine.start_read(str(room), room, endpoint, remote, local)
        return True

    # ----- asymmetric-TP resharding (producer-push) -------------------------
    #
    # Data flows prefill -> decode. Handshake (decode-initiated), per decode rank
    # r fed by prefill ranks {r*ratio .. (r+1)*ratio-1}:
    #   1. decode -> each source: RESHARD_REQ (decode push endpoint + blocks + tp)
    #   2. source -> decode: RESHARD_BLOCKS (its blocks + rank + aux)
    #   3. decode registers its receiver plan, then -> sources: RESHARD_READY
    #   4. source: d2h stage -> register sender plan -> push_registered_plan
    # The strided head-interleave is realized on the receive side; the decode
    # receiver auto-H2Ds host staging into HBM (dst_mem_type=HBM).

    def _register_reshard_request(
        self,
        room: int,
        local_blocks: List[int],
        aux_index: Optional[int],
        bootstrap_infos: List[dict],
    ) -> None:
        # bootstrap_infos is ordered by the decode->prefill target_tp_ranks
        # (range(r*ratio, (r+1)*ratio)); ratio == number of prefill sources.
        ratio = len(bootstrap_infos)
        src_attn_tp = self.attn_tp_size * ratio
        r = self.kv_args.engine_rank % self.attn_tp_size
        source_ranks = list(range(r * ratio, (r + 1) * ratio))
        push_endpoint = self._ensure_engine().transfer_address
        ready_endpoints = [
            f"tcp://{info['rank_ip']}:{info['rank_port']}" for info in bootstrap_infos
        ]
        with self._state_lock:
            self._reshard_meta[room] = dict(
                local_blocks=list(local_blocks),
                source_ranks=list(source_ranks),
                src_attn_tp=int(src_attn_tp),
                aux_index=aux_index,
                ready_endpoints=ready_endpoints,
            )
            self._reshard_sources[room] = {}
            if aux_index is not None:
                self._decode_aux_index[room] = aux_index
        payload = [
            MessageTag.RESHARD_REQ,
            str(room).encode("ascii"),
            self.local_ip.encode("ascii"),
            str(self.rank_port).encode("ascii"),
            push_endpoint.encode("ascii"),
            pack_ints(list(local_blocks)),
            str(self.attn_tp_size).encode("ascii"),
        ]
        for info in bootstrap_infos:
            endpoint = f"tcp://{info['rank_ip']}:{info['rank_port']}"
            with self._push_lock:
                self._connect(endpoint).send_multipart(payload)

    def _on_reshard_req(self, msg: List[bytes]) -> None:
        self._require_frames(msg, MessageTag.RESHARD_REQ, 7)
        room = int(msg[1].decode("ascii"))
        decode_ip = msg[2].decode("ascii")
        decode_port = int(msg[3].decode("ascii"))
        push_endpoint = msg[4].decode("ascii")
        decode_blocks = unpack_ints(msg[5])
        dst_attn_tp = int(msg[6].decode("ascii"))
        with self._state_lock:
            self._reshard_req[room] = dict(
                reply_addr=(decode_ip, decode_port),
                push_endpoint=push_endpoint,
                decode_blocks=decode_blocks,
                dst_attn_tp=dst_attn_tp,
            )
        self._maybe_reshard_reply(room)

    def _maybe_reshard_reply(self, room: int) -> None:
        with self._state_lock:
            # Only reply once, and only after send() finished accumulating this
            # room's blocks (a RESHARD_REQ may arrive mid-accumulation).
            if room in self._reshard_replied:
                return
            if room not in self._reshard_req or room not in self._reshard_finalized:
                return
            self._reshard_replied.add(room)
            addr = self._reshard_req[room]["reply_addr"]
            blocks = list(self._send_blocks.get(room, []))
            aux_blob = self._send_aux.get(room, b"")
        endpoint = f"tcp://{addr[0]}:{addr[1]}"
        payload = [
            MessageTag.RESHARD_BLOCKS,
            str(room).encode("ascii"),
            pack_ints(blocks),
            str(self.attn_tp_rank).encode("ascii"),
            aux_blob,
        ]
        with self._push_lock:
            self._connect(endpoint).send_multipart(payload)

    def _on_reshard_blocks(self, msg: List[bytes]) -> None:
        self._require_frames(msg, MessageTag.RESHARD_BLOCKS, 5)
        room = int(msg[1].decode("ascii"))
        prefill_blocks = unpack_ints(msg[2])
        src_rank = int(msg[3].decode("ascii"))
        aux_blob = msg[4]
        ready = False
        aux_index = None
        with self._state_lock:
            if room not in self._reshard_meta:
                return
            self._reshard_sources[room][src_rank] = prefill_blocks
            aux_index = self._reshard_meta[room].get("aux_index")
            have = set(self._reshard_sources[room])
            need = set(self._reshard_meta[room]["source_ranks"])
            ready = have >= need
        if aux_index is not None and aux_blob:
            write_aux_blob(
                self.kv_args.aux_data_ptrs,
                self.kv_args.aux_item_lens,
                aux_index,
                aux_blob,
            )
        if ready:
            self._register_receiver_and_signal(room)

    def _register_receiver_and_signal(self, room: int) -> None:
        from tpu_raiden.rpc import raiden_service_pb2 as pb2

        from sglang.srt.disaggregation.raiden.reshard_schedule import (
            build_receiver_request,
        )

        with self._state_lock:
            if room in self._reshard_registered:
                return
            self._reshard_registered.add(room)
            meta = self._reshard_meta[room]
            local_blocks = list(meta["local_blocks"])
            src_attn_tp = meta["src_attn_tp"]
            ready_endpoints = list(meta["ready_endpoints"])
            sources_map = dict(self._reshard_sources[room])

        geom = self._make_geometry(src_attn_tp, self.attn_tp_size)
        n = len(local_blocks)
        # A decode prefix hit leaves decode holding fewer (tail) pages than the
        # prefill produced; keep the last n of each source's blocks so both sides
        # pair the same pages.
        sources = []
        for p_rank in sorted(sources_map):
            pb = sources_map[p_rank]
            sources.append((p_rank, pb[-n:] if len(pb) > n else pb))
        live = [self.kv_args.page_size] * n

        engine = self._ensure_engine()
        req = build_receiver_request(
            pb2,
            uuid=room,
            req_id=str(room),
            peer=engine.transfer_address,
            decode_rank=self.attn_tp_rank,
            geom=geom,
            sources=sources,
            dst_block_ids=local_blocks,
            live_tokens=live,
        )
        # register_active_plan (dst_mem_type=HBM) arms the completion barrier from
        # the schedule (one block-push per (block, layer) across all writers) and
        # supplies the strided receive + auto-H2D-to-HBM geometry.
        for widx, (p_rank, pb) in enumerate(sources):
            logger.debug(
                "reshard recv room=%s decode_rank=%s writer=%s src_rank=%s "
                "n_src=%d src[:4]=%s dst n=%d dst[:4]=%s",
                room,
                self.attn_tp_rank,
                widx,
                p_rank,
                len(pb),
                pb[:4],
                len(local_blocks),
                local_blocks[:4],
            )
        engine.register_active_plan(room, req, is_sender=False)

        for endpoint in ready_endpoints:
            payload = [MessageTag.RESHARD_READY, str(room).encode("ascii")]
            with self._push_lock:
                self._connect(endpoint).send_multipart(payload)

    def _on_reshard_ready(self, msg: List[bytes]) -> None:
        self._require_frames(msg, MessageTag.RESHARD_READY, 2)
        room = int(msg[1].decode("ascii"))
        with self._state_lock:
            self._reshard_ready.add(room)
        self._maybe_reshard_push(room)

    def _maybe_reshard_push(self, room: int) -> None:
        with self._state_lock:
            if (
                room not in self._reshard_ready
                or room not in self._reshard_req
                or room not in self._send_blocks
                or room in self._reshard_pushed
            ):
                return
            self._reshard_pushed.add(room)
        self._push_executor.submit(self._do_reshard_push, room)

    def _do_reshard_push(self, room: int) -> None:
        # A push failure marks the room failed so the sender poll surfaces it
        # instead of the request hanging until the decode-side timeout.
        try:
            self._run_reshard_push(room)
        except Exception as e:
            logger.error("raiden reshard push failed for room %s: %s", room, e)
            with self._state_lock:
                self._failed.add(str(room))

    def _run_reshard_push(self, room: int) -> None:
        from tpu_raiden.rpc import raiden_service_pb2 as pb2

        from sglang.srt.disaggregation.raiden.reshard_schedule import (
            build_sender_request,
        )

        with self._state_lock:
            req_meta = dict(self._reshard_req[room])
            prefill_blocks = list(self._send_blocks[room])

        decode_blocks = req_meta["decode_blocks"]
        dst_attn_tp = req_meta["dst_attn_tp"]
        push_endpoint = req_meta["push_endpoint"]
        n = len(decode_blocks)
        if len(prefill_blocks) > n:
            prefill_blocks = prefill_blocks[-n:]
        geom = self._make_geometry(self.attn_tp_size, dst_attn_tp)
        live = [self.kv_args.page_size] * n

        engine = self._ensure_engine()
        # push_registered_plan streams from the host staging pool; the pool is
        # sized as a full device shadow (see _build_engine) and indexed 1-to-1 by
        # device block id, so stage each device block into the host block at the
        # SAME id. The receiver matches on this same src_block_id (the prefill
        # device blocks it learned via RESHARD_BLOCKS).
        fut = engine.d2h(prefill_blocks, prefill_blocks, [1] * n)
        deadline = self._monotonic() + self._timeout_s
        while fut is not None and not fut.is_ready():
            if self._monotonic() > deadline:
                raise TimeoutError(
                    f"raiden reshard D2H stage for room {room} timed out"
                )
            time.sleep(0.0005)
        # is_ready() is true even for a ready-with-error future, so check ok()
        # before pushing -- otherwise a failed d2h would push stale staging.
        if fut is not None and not fut.ok():
            raise RuntimeError(
                f"raiden reshard D2H stage for room {room} failed: "
                f"{fut.error_message()}"
            )

        req = build_sender_request(
            pb2,
            uuid=room,
            req_id=str(room),
            peer=push_endpoint,
            prefill_rank=self.attn_tp_rank,
            decode_rank=self.attn_tp_rank // geom.ratio,
            geom=geom,
            src_block_ids=prefill_blocks,
            dst_block_ids=decode_blocks,
            live_tokens=live,
        )
        logger.debug(
            "reshard push room=%s prefill_rank=%s decode_rank=%s peer=%s "
            "n=%d src[:4]=%s dst[:4]=%s",
            room,
            self.attn_tp_rank,
            self.attn_tp_rank // geom.ratio,
            push_endpoint,
            len(prefill_blocks),
            prefill_blocks[:4],
            decode_blocks[:4],
        )
        # raiden runs SyncPush unlocked, so serialize register+push across workers
        # (concurrent pushes to a decode peer corrupt its shared transport); d2h
        # above is internally locked and can overlap.
        with self._reshard_push_lock:
            engine.register_active_plan(room, req, is_sender=True)
            # The receiver's completion barrier is total_blocks * num_layers, and
            # each push counts its blocks once regardless of how many layers it
            # carries, so push one layer per request to deliver every layer.
            for layer_idx in range(len(self._kv_caches)):
                engine.push_registered_plan(
                    room, push_endpoint, prefill_blocks, decode_blocks, layer_idx
                )
        with self._state_lock:
            self._completed_send.add(str(room))

    @staticmethod
    def _monotonic() -> float:
        return time.monotonic()

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
                getattr(self, "_reshard_req", None),
                getattr(self, "_reshard_meta", None),
                getattr(self, "_reshard_sources", None),
            ):
                if d is not None:
                    d.pop(room, None)
            if hasattr(self, "_replied"):
                self._replied.discard(room)
            for s in (
                getattr(self, "_reshard_finalized", None),
                getattr(self, "_reshard_replied", None),
                getattr(self, "_reshard_ready", None),
                getattr(self, "_reshard_pushed", None),
                getattr(self, "_reshard_registered", None),
            ):
                if s is not None:
                    s.discard(room)
        self._release_host_slot(room)


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
