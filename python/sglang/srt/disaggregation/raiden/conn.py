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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll, StateType
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
from sglang.srt.disaggregation.raiden.chunk_plan import chunk_pull_window
from sglang.srt.disaggregation.raiden.host_pool import SlotAdmission
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

# Pages are announced to the decode in pieces of at least this many tokens. A
# raiden read costs about the same whether it moves 16 pages or 200, so
# announcing every prefill step would spend more on rendezvous than it hides;
# pieces of this size keep the overlap while costing a few reads per request.
CHUNK_ANNOUNCE_TOKENS = 8192


@dataclass(frozen=True)
class RaidenKVCacheGroups:
    """Page-compatible cache groups registered with independent Raiden engines."""

    full: List[Any]
    swa: List[Any]


def get_raiden_kv_caches(pool) -> RaidenKVCacheGroups:
    """Group per-layer device KV tensors by their page-index namespace.

    The TPU fused KV pool stores one [num_blocks, page_size, ...] tensor per
    layer in ``kv_buffer`` and exposes them via ``get_fused_kv_buffer``. Hybrid
    SWA pools allocate full-attention and sliding-window pages independently, so
    they must not share one Raiden block-id list.
    """
    if hasattr(pool, "get_fused_kv_buffer"):
        start = pool.start_layer
        buffers = [pool.get_fused_kv_buffer(start + i) for i in range(pool.layer_num)]
    elif hasattr(pool, "kv_buffer"):
        buffers = list(pool.kv_buffer)
    else:
        raise RuntimeError(
            f"Raiden backend requires a fused KV pool exposing kv_buffer; got "
            f"{type(pool).__name__}"
        )
    layers_mapping = getattr(pool, "layers_mapping", None)
    if layers_mapping:
        full = []
        swa = []
        for offset, buf in enumerate(buffers):
            layer_id = pool.start_layer + offset
            (swa if layers_mapping[layer_id][1] else full).append(buf)
    else:
        full, swa = buffers, []

    for name, group in (("full", full), ("SWA", swa)):
        block_counts = {int(buf.shape[0]) for buf in group}
        if len(block_counts) > 1:
            raise RuntimeError(
                f"Raiden requires uniform {name} KV block counts; got "
                f"{sorted(block_counts)}"
            )
    return RaidenKVCacheGroups(full=full, swa=swa)


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
        self._swa_engine: Optional[Any] = None
        self._kv_caches: Optional[list] = None
        self._swa_kv_caches: Optional[list] = None
        self.raiden_endpoint: Optional[str] = None
        self.swa_raiden_endpoint: Optional[str] = None
        # Each rank reserves control+data ports for full KV and SWA KV.
        base_port = int(os.environ.get("SGLANG_RAIDEN_KV_PORT", "30000"))
        self._local_rank = self.attn_dp_rank * self.attn_tp_size + self.attn_tp_rank
        self._control_port = base_port + 4 * self._local_rank
        self._timeout_s = float(os.environ.get("SGLANG_RAIDEN_TIMEOUT_S", "120"))
        # Strided push-resharding for asymmetric TP (decode TP < prefill TP),
        # on by default and auto-engaged per request only when a
        # decode rank is fed by several prefill ranks; symmetric traffic never
        # enters this path. Set SGLANG_RAIDEN_RESHARD=0 to disable it (an
        # asymmetric topology then raises) and keep the multi-slot host staging
        # pool instead of the device-shadow sizing the reshard receiver needs.
        self._reshard_enabled = bool(int(os.environ.get("SGLANG_RAIDEN_RESHARD", "1")))
        # Whether KV moves chunk by chunk as the prefill commits it, or in one
        # transfer after the whole prefill; decided from the KV layout when the
        # pool tensors are handed over.
        self.chunked_transfer = False

        # Completion sets, drained from raiden and keyed by str(room). Persist
        # until clear() so repeated poll() calls before the framework advances
        # keep reporting the same terminal state.
        self._completed_send: Set[str] = set()
        self._completed_recv: Set[str] = set()
        self._failed: Set[str] = set()
        # Rooms whose pull is issued and not yet settled. Raiden borrows one
        # host staging slot per in-flight transfer and rejects a read it cannot
        # seat, so the decode side admits at most one pull per slot and leaves
        # the rest in the transfer queue.
        self._pulling: Optional[SlotAdmission] = None
        # Reads of rooms the framework dropped while they were still in
        # flight. Raiden cannot cancel a read, so the read keeps its admission
        # until the engine reports it settled.
        self._pulling_cleared: Set[str] = set()
        # Receiver plans of rooms the framework dropped while their push was
        # still in flight, by request id. Raiden cannot cancel a push and
        # keeps delivering into the plan, so the plan stays registered until
        # the engine reports the push settled.
        self._cleared_plans: Dict[str, int] = {}
        self._engine_max_blocks = 0
        self._swa_engine_max_blocks = 0
        self._engine_lock = threading.Lock()
        self._push_lock = threading.Lock()
        self._state_lock = threading.Lock()

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # room -> accumulated page block-ids, aux blob, decode reply addr
            self._send_blocks: Dict[int, List[int]] = {}
            self._send_swa_blocks: Dict[int, List[int]] = {}
            self._send_aux: Dict[int, bytes] = {}
            self._decode_addr: Dict[int, Tuple[str, int]] = {}
            self._replied: Set[int] = set()
            # Chunks announced before the decode asked for the room, held until
            # a PULL_REQ names it (a resharded room never gets one).
            self._held_chunks: Dict[int, List[Tuple[int, List[int]]]] = {}
            self._send_total: Dict[int, int] = {}
            # Pages the decode allocated for the room, so a chunk it already
            # holds is never registered or announced.
            self._decode_pages: Dict[int, int] = {}
            # Chunks announced to a resharding decode, by chunk index, until it
            # asks for their push; how far its announcements reach; and the
            # pushes still running, so the room counts as sent only after the
            # last one.
            self._reshard_chunks: Dict[int, Dict[int, Tuple[int, List[int]]]] = {}
            self._reshard_announced: Dict[int, int] = {}
            self._reshard_pushes_pending: Dict[int, int] = {}
            self._reshard_final_pushed: Set[int] = set()
            # Committed pages not yet announced, gathered until they make a
            # piece worth a read.
            self._pending_offer: Dict[int, Tuple[int, List[int]]] = {}
            self._announce_pages = max(
                1, -(-CHUNK_ANNOUNCE_TOKENS // int(self.kv_args.page_size))
            )
            self._chunk_seq: Dict[int, int] = {}
            self._uuid_seq = 0
        else:
            # room -> local page block-ids, remote page block-ids, endpoint
            self._local_blocks: Dict[int, List[int]] = {}
            self._local_swa_blocks: Dict[int, List[int]] = {}
            self._decode_aux_index: Dict[int, int] = {}
            self._remote_blocks: Dict[int, List[int]] = {}
            self._remote_swa_blocks: Dict[int, List[int]] = {}
            self._remote_endpoint: Dict[int, str] = {}
            self._remote_swa_endpoint: Dict[int, str] = {}
            # Chunk announcements waiting for a staging slot, and the reads
            # already issued for a room.
            self._chunk_queue: Dict[int, List[tuple]] = {}
            self._remote_total: Dict[int, int] = {}
            self._announced_pages: Dict[int, int] = {}
            self._chunk_reads: Dict[int, List[str]] = {}
            self._final_seen: Set[int] = set()
            # Chunk announcements from each resharding source, by chunk index,
            # until every source of the room has announced it; the receive
            # plans registered for the room, by their request id.
            self._reshard_chunk_sources: Dict[
                int, Dict[int, Dict[int, Tuple[int, List[int]]]]
            ] = {}
            self._reshard_plans: Dict[int, Dict[str, int]] = {}
            self._uuid_seq = 0

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
        )

        tensor = self._kv_caches[0]
        hp = head_pair_bytes(tensor)
        # Distinct model kv-head count (model_config.get_total_num_kv_heads()),
        # set on both the prefill and decode KVArgs.
        total_kv_heads = int(self.kv_args.total_kv_head_num)
        return ReshardGeometry(src_attn_tp, dst_attn_tp, hp, total_kv_heads)

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

    def set_kv_caches(self, kv_caches: RaidenKVCacheGroups) -> None:
        """Hand raiden the per-layer device KV tensors and build the engine.

        Called by prefill.py/decode.py right after construction (the KV pool is
        only available there). Raiden indexes each tensor's major dim as a
        block/page, which matches the TPU fused KV pool layout
        ([num_blocks, page_size, ...]).
        """
        self._kv_caches = list(kv_caches.full)
        self._swa_kv_caches = list(kv_caches.swa)
        # Full-attention KV moves piece by piece during the prefill. A hybrid
        # model's sliding-window pool holds only the last window, which is
        # settled when the prefill ends, so it rides on the closing read.
        self.chunked_transfer = True
        if not self._kv_caches:
            raise RuntimeError(
                "Raiden requires at least one full-attention KV cache tensor"
            )
        self._build_engine()

    def _build_engine(self) -> None:
        if self._engine is not None or self._kv_caches is None:
            return
        # TPU-only import, deferred so the disaggregation module imports cleanly
        # on hosts without tpu_sync. Raiden owns the transport, host staging,
        # D2H/H2D DMA, and completion tracking; it indexes each KV tensor's major
        # dim as a page, matching the TPU fused KV pool ([num_blocks, page, ...]).
        from tpu_sync.api.torch.kv_cache_manager import KVCacheManager

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
        self._pulling = SlotAdmission(num_slots)
        self._engine_max_blocks = max_blocks
        logger.info(
            "RaidenKVManager rank%d engine up | endpoint=%s max_blocks=%d "
            "num_slots=%d chunked=%d",
            self.attn_tp_rank,
            self.raiden_endpoint,
            max_blocks,
            num_slots,
            int(self.chunked_transfer),
        )
        if self._swa_kv_caches:
            self.swa_raiden_endpoint = f"{self.local_ip}:{self._control_port + 2}"
            swa_max_blocks = self._swa_max_request_blocks()
            swa_num_slots = self._num_slots()
            self._swa_engine = KVCacheManager(
                kv_caches=self._swa_kv_caches,
                node_id=self._local_rank,
                local_control_port=self._control_port + 2,
                max_blocks=swa_max_blocks,
                num_slots=swa_num_slots,
                timeout_s=self._timeout_s,
            )
            # A room occupies a slot in both engines; the tighter pool binds.
            self._pulling.capacity = min(self._pulling.capacity, swa_num_slots)
            self._swa_engine_max_blocks = swa_max_blocks
            logger.info(
                "RaidenKVManager rank%d SWA engine up | endpoint=%s "
                "max_blocks=%d num_slots=%d",
                self.attn_tp_rank,
                self.swa_raiden_endpoint,
                swa_max_blocks,
                swa_num_slots,
            )

    def _max_request_blocks(self) -> int:
        override = int(os.environ.get("SGLANG_RAIDEN_MAX_BLOCKS", "0"))
        if override > 0:
            return override
        max_len = self.server_args.context_length or int(
            os.environ.get("SGLANG_RAIDEN_MAX_MODEL_LEN", "8192")
        )
        return max(1, math.ceil(max_len / self.kv_args.page_size))

    def _swa_max_request_blocks(self) -> int:
        override = int(os.environ.get("SGLANG_RAIDEN_SWA_MAX_BLOCKS", "0"))
        if override > 0:
            return override
        window_size = self.kv_args.sliding_window_size
        if window_size is None or window_size <= 0:
            raise RuntimeError(
                "Raiden SWA KV caches require a positive sliding-window size"
            )
        # SWA transfer payloads round their window start down to a page boundary.
        max_tokens = window_size + self.kv_args.page_size - 1
        return max(1, math.ceil(max_tokens / self.kv_args.page_size))

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

    @staticmethod
    def _swa_req_id(room: int) -> str:
        return f"swa:{room}"

    @staticmethod
    def _tail_after(blocks: List[int], announced: int) -> Tuple[int, List[int]]:
        """Pages the closing message of a chunked transfer still has to move.

        The last chunk's pages carry no announcement of their own; they travel
        with the closing message under the room's own id. That transfer is also
        what tells the prefill the request is done, so when the last step added
        no pages the final page is moved again rather than leaving the sender
        waiting on an id nobody quotes.
        """
        start, tail = announced, blocks[announced:]
        if not tail and blocks:
            start, tail = len(blocks) - 1, blocks[-1:]
        return start, tail

    def _drain_engine(self) -> None:
        if self._engine is None and self._swa_engine is None:
            return
        for engine in (self._engine, self._swa_engine):
            if engine is None:
                continue
            done_send, done_recv, failed = engine.poll_stats()
            if done_send or done_recv or failed:
                with self._state_lock:
                    self._completed_send.update(self._norm_req_id(r) for r in done_send)
                    self._completed_recv.update(self._norm_req_id(r) for r in done_recv)
                    self._failed.update(self._norm_req_id(r) for r in failed)
                    self._release_pull_slots_locked()
                    settled_plans = self._settled_cleared_plans_locked()
                for uuid in settled_plans:
                    self._unregister_plan(uuid)
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
                elif msg[0] == MessageTag.CHUNK:
                    self._on_chunk(msg)
                elif msg[0] == MessageTag.RESHARD_REQ:
                    self._on_reshard_req(msg)
                elif msg[0] == MessageTag.RESHARD_BLOCKS:
                    self._on_reshard_blocks(msg)
                elif msg[0] == MessageTag.RESHARD_READY:
                    self._on_reshard_ready(msg)
                elif msg[0] == MessageTag.RESHARD_CHUNK:
                    self._on_reshard_chunk(msg)
                elif msg[0] == MessageTag.RESHARD_CHUNK_READY:
                    self._on_reshard_chunk_ready(msg)
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
        self._require_frames(msg, MessageTag.PULL_REQ, 5)
        room = int(msg[1].decode("ascii"))
        decode_ip = msg[2].decode("ascii")
        decode_port = int(msg[3].decode("ascii"))
        decode_pages = int(msg[4].decode("ascii"))
        with self._state_lock:
            self._decode_addr[room] = (decode_ip, decode_port)
            self._decode_pages[room] = decode_pages
            held = self._held_chunks.pop(room, [])
            total_pages = self._send_total.get(room, 0)
        for start, block_ids in held:
            self._announce_chunk(room, start, block_ids, total_pages)
        self._maybe_reply(room)

    def _on_chunk(self, msg: List[bytes]) -> None:
        self._require_frames(msg, MessageTag.CHUNK, 8)
        room = int(msg[1].decode("ascii"))
        req_id = msg[2].decode("ascii")
        uuid = int(msg[3].decode("ascii"))
        start = int(msg[4].decode("ascii"))
        total_pages = int(msg[5].decode("ascii"))
        blocks = unpack_blocks(msg[6])
        endpoint = msg[7].decode("ascii")
        with self._state_lock:
            self._remote_total[room] = total_pages
            # Chunks are announced from two threads and can cross on the wire,
            # so the announced reach is the furthest chunk end, not the last.
            self._announced_pages[room] = max(
                self._announced_pages.get(room, 0), start + len(blocks)
            )
            self._chunk_queue.setdefault(room, []).append(
                (req_id, uuid, start, blocks, endpoint)
            )

    def _on_blocks(self, msg: List[bytes]) -> None:
        self._require_frames(msg, MessageTag.BLOCKS, 7)
        room = int(msg[1].decode("ascii"))
        remote_blocks = unpack_blocks(msg[2])
        endpoint = msg[3].decode("ascii")
        remote_swa_blocks = unpack_blocks(msg[4])
        swa_endpoint = msg[5].decode("ascii")
        aux_blob = msg[6]
        aux_index = None
        with self._state_lock:
            self._remote_blocks[room] = remote_blocks
            self._remote_endpoint[room] = endpoint
            self._remote_swa_blocks[room] = remote_swa_blocks
            self._remote_swa_endpoint[room] = swa_endpoint
            aux_index = self._decode_aux_index.get(room)
            if self.chunked_transfer:
                announced = self._announced_pages.get(room, 0)
                total = len(remote_blocks)
                if self._remote_total.get(room, total) != total:
                    logger.warning(
                        "RaidenKVManager rank%d room %s announced %d pages but "
                        "sent %d",
                        self.attn_tp_rank,
                        room,
                        self._remote_total[room],
                        total,
                    )
                self._remote_total[room] = total
                tail_start, tail = self._tail_after(remote_blocks, announced)
                if tail:
                    self._chunk_queue.setdefault(room, []).append(
                        (str(room), room, tail_start, tail, endpoint)
                    )
                self._final_seen.add(room)
        if aux_index is not None and aux_blob:
            write_aux_blob(
                self.kv_args.aux_data_ptrs,
                self.kv_args.aux_item_lens,
                aux_index,
                aux_blob,
            )

    # ----- prefill side ------------------------------------------------------

    def _swa_blocks_from_state_indices(
        self, state_indices: Optional[List]
    ) -> List[int]:
        if not self._swa_kv_caches:
            return []
        try:
            index = self.kv_args.state_types.index(StateType.SWA)
        except ValueError as exc:
            raise RuntimeError(
                "Raiden has SWA KV tensors but StateType.SWA is not registered"
            ) from exc
        if (
            state_indices is None
            or index >= len(state_indices)
            or state_indices[index] is None
        ):
            raise RuntimeError("Raiden SWA transfer is missing its page indices")
        return [int(x) for x in state_indices[index]]

    def append_send_blocks(
        self,
        room: int,
        block_ids: List[int],
        swa_block_ids: Optional[List[int]] = None,
    ) -> int:
        """Records a chunk's pages and returns its start index in the request."""
        with self._state_lock:
            accumulated = self._send_blocks.setdefault(room, [])
            start = len(accumulated)
            accumulated.extend(block_ids)
            if swa_block_ids is not None:
                self._send_swa_blocks[room] = list(swa_block_ids)
        return start

    def offer_chunk(
        self, room: int, start: int, block_ids: List[int], total_pages: int
    ) -> None:
        """Offers one chunked-prefill step's pages to the decode.

        Pages gather until they make a piece of CHUNK_ANNOUNCE_TOKENS; the
        closing message carries whatever is left. A piece is held until the
        decode has named the room, with a pull request or a resharding
        request, since that is what decides which transport it travels by.
        """
        if not block_ids:
            return
        with self._state_lock:
            self._send_total[room] = total_pages
            p_start, p_blocks = self._pending_offer.get(room, (start, []))
            p_blocks = p_blocks + list(block_ids)
            if len(p_blocks) < self._announce_pages:
                self._pending_offer[room] = (p_start, p_blocks)
                return
            self._pending_offer.pop(room, None)
            start, block_ids = p_start, p_blocks
            if room in self._decode_addr:
                resharded = False
            elif room in self._reshard_req:
                resharded = True
            else:
                self._held_chunks.setdefault(room, []).append((start, list(block_ids)))
                return
        if resharded:
            self._announce_reshard_chunk(room, start, list(block_ids), total_pages)
        else:
            self._announce_chunk(room, start, list(block_ids), total_pages)

    def _announce_chunk(
        self, room: int, start: int, block_ids: List[int], total_pages: int
    ) -> None:
        """Registers a chunk for reading and tells the decode where it is.

        Only the pages the decode will pull are registered: a page it already
        holds from its own prefix cache is cut off the front, so no
        registration is left behind for a read that never comes. The
        consumer's read blocks on a matching producer registration, so the
        req_id and uuid the decode will quote are minted here and travel with
        the announcement.
        """
        with self._state_lock:
            window = chunk_pull_window(
                start, len(block_ids), total_pages, self._decode_pages[room]
            )
            if not window:
                return
            start += window.chunk_offset
            block_ids = block_ids[
                window.chunk_offset : window.chunk_offset + window.length
            ]
            index = self._chunk_seq.get(room, 0)
            self._chunk_seq[room] = index + 1
            self._uuid_seq += 1
            uuid = (self._uuid_seq << 20) | (self._local_rank & 0xFFFFF)
            addr = self._decode_addr[room]
        req_id = f"{room}#{index}"
        self._ensure_engine().register_read(req_id, uuid, block_ids)
        payload = [
            MessageTag.CHUNK,
            str(room).encode("ascii"),
            req_id.encode("ascii"),
            str(uuid).encode("ascii"),
            str(start).encode("ascii"),
            str(total_pages).encode("ascii"),
            pack_blocks(block_ids),
            (self.raiden_endpoint or "").encode("ascii"),
        ]
        endpoint = f"tcp://{addr[0]}:{addr[1]}"
        with self._push_lock:
            self._connect(endpoint).send_multipart(payload)

    def finalize_send(self, room: int, aux_index: Optional[int]) -> None:
        self._ensure_engine()
        with self._state_lock:
            if aux_index is not None:
                self._send_aux[room] = read_aux_blob(
                    self.kv_args.aux_data_ptrs,
                    self.kv_args.aux_item_lens,
                    aux_index,
                )
            self._reshard_finalized.add(room)
            is_reshard = room in self._reshard_req
        # The request type is selected by the decode handshake, not by whether
        # reshard support is globally enabled. Symmetric TP uses PULL_REQ even
        # when resharding is available; asymmetric TP uses RESHARD_REQ.
        if is_reshard:
            self._maybe_reshard_reply(room)
        else:
            self._maybe_reply(room)

    def _maybe_reply(self, room: int) -> None:
        with self._state_lock:
            if room in self._replied:
                return
            if (
                room not in self._reshard_finalized
                or room not in self._send_blocks
                or (self._swa_engine is not None and room not in self._send_swa_blocks)
                or room not in self._decode_addr
            ):
                return
            swa_blocks = list(self._send_swa_blocks.get(room, []))
            blocks = list(self._send_blocks[room])
            addr = self._decode_addr[room]
            aux_blob = self._send_aux.get(room, b"")
            self._replied.add(room)
        if self._swa_engine is not None:
            self._swa_engine.register_read(self._swa_req_id(room), room, swa_blocks)
        self._ensure_engine().register_read(str(room), room, blocks)
        endpoint = f"tcp://{addr[0]}:{addr[1]}"
        payload = [
            MessageTag.BLOCKS,
            str(room).encode("ascii"),
            pack_blocks(blocks),
            (self.raiden_endpoint or "").encode("ascii"),
            pack_blocks(swa_blocks),
            (self.swa_raiden_endpoint or "").encode("ascii"),
            aux_blob,
        ]
        with self._push_lock:
            self._connect(endpoint).send_multipart(payload)

    # ----- decode side -------------------------------------------------------

    def register_decode_request(
        self,
        room: int,
        local_blocks: List[int],
        local_swa_blocks: List[int],
        aux_index: Optional[int],
        bootstrap_infos: List[dict],
    ) -> None:
        if len(bootstrap_infos) != 1:
            if self._swa_engine is not None:
                raise NotImplementedError(
                    "Raiden cannot reshard hybrid SWA KV caches: decode rank "
                    f"received {len(bootstrap_infos)} prefill sources. Use "
                    "symmetric attention TP for prefill and decode."
                )
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
            self._local_swa_blocks[room] = local_swa_blocks
            if aux_index is not None:
                self._decode_aux_index[room] = aux_index
        payload = [
            MessageTag.PULL_REQ,
            str(room).encode("ascii"),
            self.local_ip.encode("ascii"),
            str(self.rank_port).encode("ascii"),
            str(len(local_blocks)).encode("ascii"),
        ]
        for info in bootstrap_infos:
            endpoint = f"tcp://{info['rank_ip']}:{info['rank_port']}"
            with self._push_lock:
                self._connect(endpoint).send_multipart(payload)

    @staticmethod
    def _pair_blocks(
        remote: List[int], local: List[int]
    ) -> Tuple[List[int], List[int]]:
        local = list(local)
        remote = list(remote)
        if not local:
            remote = []
        elif len(remote) > len(local):
            remote = remote[-len(local) :]
        elif len(remote) < len(local):
            local = local[: len(remote)]
        return remote, local

    def _paired_swa_read(self, key: str) -> Optional[str]:
        """The sliding-window read issued alongside the read ``key``, if any.

        A whole-room read and the closing read of a chunked transfer carry the
        sliding-window pages on the second engine; earlier chunks do not.
        """
        if self._swa_engine is None or "#" in key:
            return None
        return self._swa_req_id(int(key))

    def _pull_settled_locked(self, key: str) -> bool:
        """Whether every engine has reported the read ``key`` done or failed.

        Raiden holds a read's host staging until it completes or times out,
        whichever engine it is on, so the read's admission stays claimed until
        both it and its paired sliding-window read have settled.
        """
        settled = key in self._completed_recv or key in self._failed
        swa_rid = self._paired_swa_read(key)
        if swa_rid is not None:
            settled = settled and (
                swa_rid in self._completed_recv or swa_rid in self._failed
            )
        return settled

    def _settled_cleared_plans_locked(self) -> List[int]:
        """Plan uuids of cleared rooms whose push the engine has now settled.

        Their completion bookkeeping is dropped here, as nothing else will
        read it; the caller unregisters the plans outside the lock.
        """
        uuids = []
        for req_id, uuid in list(self._cleared_plans.items()):
            if req_id in self._completed_recv or req_id in self._failed:
                del self._cleared_plans[req_id]
                self._completed_recv.discard(req_id)
                self._failed.discard(req_id)
                uuids.append(uuid)
        return uuids

    def _unregister_plan(self, uuid: int) -> None:
        """Drops a receiver plan whose push has settled.

        Raiden drops a timed-out plan itself, so one it no longer knows has
        already been dropped.
        """
        try:
            self._engine.unregister_active_plan(uuid)
        except RuntimeError as e:
            if "not registered" not in str(e):
                raise

    def _release_pull_slots_locked(self) -> None:
        """Hand back the slots of reads that have settled."""
        if self._pulling is None:
            return
        for key in self._pulling.rooms():
            if not self._pull_settled_locked(key):
                continue
            self._pulling.release(key)
            if key in self._pulling_cleared:
                # The framework already dropped this read's room; nothing will
                # read its completion, so drop it too.
                self._pulling_cleared.discard(key)
                for rid in (key, self._paired_swa_read(key) or key):
                    self._completed_recv.discard(rid)
                    self._failed.discard(rid)

    def _reject_oversized_pull(
        self, room: int, pages: int, limit: int, env: str
    ) -> None:
        """Fail a request whose pages cannot fit one host staging slot."""
        reason = (
            f"Raiden receive failed: request needs {pages} pages but a host "
            f"staging slot holds {limit}. Lower the context length or raise "
            f"{env}."
        )
        self.record_failure(room, reason)
        with self._state_lock:
            self._failed.add(str(room))
        logger.error("RaidenKVManager rank%d %s", self.attn_tp_rank, reason)

    def maybe_start_pull(self, room: int) -> bool:
        """Issue whatever of this room's pull is announced and affordable.

        Returns True once at least one read is in flight for the room. A False
        return is a retry: the caller polls again on the next scheduler pass.
        """
        if self.chunked_transfer:
            return self._start_chunked_pull(room)
        return self._start_whole_pull(room)

    def _start_chunked_pull(self, room: int) -> bool:
        """Issue the next announced chunk once a staging slot can be found.

        A request keeps at most one chunk read in flight, so it occupies one
        staging slot at a time exactly as a single-shot pull does; the next
        chunk is issued when the previous read has settled. Chunks are taken
        in prefill order.
        """
        engine = self._ensure_engine()
        while True:
            with self._state_lock:
                queue = self._chunk_queue.get(room)
                reads = self._chunk_reads.get(room, [])
                started = bool(reads)
                if not queue or room not in self._local_blocks:
                    return started
                if any(rid in self._pulling for rid in reads):
                    return started
                local = self._local_blocks[room]
                total = self._remote_total.get(room, 0)
                req_id, uuid, start, blocks, endpoint = queue[0]
                window = chunk_pull_window(start, len(blocks), total, len(local))
                if not window:
                    # Pages the decode already holds, or the page it reserved
                    # for the first decode token.
                    queue.pop(0)
                    continue
                if window.length > self._engine_max_blocks:
                    oversized = (
                        window.length,
                        self._engine_max_blocks,
                        "SGLANG_RAIDEN_MAX_BLOCKS",
                    )
                    break
                # The closing read also carries the sliding-window pages, on
                # the engine that stages them against its own limit.
                closing = req_id == str(room) and self._swa_engine is not None
                if closing:
                    remote_swa, local_swa = self._pair_blocks(
                        list(self._remote_swa_blocks.get(room, [])),
                        list(self._local_swa_blocks.get(room, [])),
                    )
                    swa_endpoint = self._remote_swa_endpoint.get(room, "")
                    swa_pages = len(set(local_swa))
                    if swa_pages > self._swa_engine_max_blocks:
                        oversized = (
                            swa_pages,
                            self._swa_engine_max_blocks,
                            "SGLANG_RAIDEN_SWA_MAX_BLOCKS",
                        )
                        break
                if not self._pulling.admit(req_id):
                    return started
                queue.pop(0)
                remote_slice = blocks[
                    window.chunk_offset : window.chunk_offset + window.length
                ]
                local_slice = local[
                    window.local_offset : window.local_offset + window.length
                ]
                self._chunk_reads.setdefault(room, []).append(req_id)
            engine.start_read(req_id, uuid, endpoint, remote_slice, local_slice)
            if closing:
                self._swa_engine.start_read(
                    self._swa_req_id(room), room, swa_endpoint, remote_swa, local_swa
                )
        self._reject_oversized_pull(room, *oversized)
        return False

    def _start_whole_pull(self, room: int) -> bool:
        """Issue one read covering the request, once the prefill has replied."""
        with self._state_lock:
            if (
                room not in self._remote_blocks
                or room not in self._remote_endpoint
                or (
                    self._swa_engine is not None and room not in self._remote_swa_blocks
                )
                or (
                    self._swa_engine is not None
                    and room not in self._remote_swa_endpoint
                )
            ):
                return False
            local = list(self._local_blocks.get(room, []))
            remote = list(self._remote_blocks[room])
            endpoint = self._remote_endpoint[room]
            local_swa = list(self._local_swa_blocks.get(room, []))
            remote_swa = list(self._remote_swa_blocks.get(room, []))
            swa_endpoint = self._remote_swa_endpoint.get(room, "")
        # start_read needs equal-length, position-paired remote/local. Reconcile
        # the two independent ways they can differ:
        #   - decode prefix-cache hit: decode already holds the head pages and
        #     allocated fewer local pages, so pull only the matching tail of
        #     remote (remote[-len(local):]).
        #   - reserved decode token: decode allocated one extra trailing page that
        #     prefill produced no KV for, so drop it from local.
        # Empty local => full prefix hit; pull nothing (and avoid remote[-0:],
        # which would select the whole list).
        remote, local = self._pair_blocks(remote, local)
        engine = self._ensure_engine()
        pages = len(set(local))
        if pages > self._engine_max_blocks:
            self._reject_oversized_pull(
                room, pages, self._engine_max_blocks, "SGLANG_RAIDEN_MAX_BLOCKS"
            )
            return False
        # Each engine stages into its own pool, so the sliding-window pages
        # are checked against the sliding-window limit, not the full one.
        if self._swa_engine is not None:
            remote_swa, local_swa = self._pair_blocks(remote_swa, local_swa)
            swa_pages = len(set(local_swa))
            if swa_pages > self._swa_engine_max_blocks:
                self._reject_oversized_pull(
                    room,
                    swa_pages,
                    self._swa_engine_max_blocks,
                    "SGLANG_RAIDEN_SWA_MAX_BLOCKS",
                )
                return False
        with self._state_lock:
            # The receiver polls every scheduler pass; a room whose read is
            # already in flight must not issue again -- a second start_read
            # under the same uuid replaces the engine's receive entry and
            # leaks the staging its predecessor held.
            # Admission is keyed by the read's request id, the same string the
            # engine reports completion under, so a settled read releases its
            # own key; a chunked read's key is its chunk id.
            if str(room) in self._pulling:
                return True
            if not self._pulling.admit(str(room)):
                return False
        engine.start_read(str(room), room, endpoint, remote, local)
        if self._swa_engine is not None:
            self._swa_engine.start_read(
                self._swa_req_id(room), room, swa_endpoint, remote_swa, local_swa
            )
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
            held = self._held_chunks.pop(room, [])
            total_pages = self._send_total.get(room, 0)
        for start, block_ids in held:
            self._announce_reshard_chunk(room, start, block_ids, total_pages)
        self._maybe_reshard_reply(room)

    def _announce_reshard_chunk(
        self, room: int, start: int, block_ids: List[int], total_pages: int
    ) -> None:
        """Tells a resharding decode that one chunk's pages are ready.

        Only the pages the decode will receive are announced. The push itself
        waits until the decode has registered the chunk and asks for it.
        """
        with self._state_lock:
            req = self._reshard_req[room]
            window = chunk_pull_window(
                start, len(block_ids), total_pages, len(req["decode_blocks"])
            )
            if not window:
                return
            start += window.chunk_offset
            block_ids = block_ids[
                window.chunk_offset : window.chunk_offset + window.length
            ]
            index = self._chunk_seq.get(room, 0)
            self._chunk_seq[room] = index + 1
            self._reshard_chunks.setdefault(room, {})[index] = (start, block_ids)
            self._reshard_announced[room] = max(
                self._reshard_announced.get(room, 0), start + len(block_ids)
            )
            addr = req["reply_addr"]
        payload = [
            MessageTag.RESHARD_CHUNK,
            str(room).encode("ascii"),
            str(index).encode("ascii"),
            str(start).encode("ascii"),
            str(total_pages).encode("ascii"),
            pack_ints(block_ids),
            str(self._local_rank).encode("ascii"),
        ]
        endpoint = f"tcp://{addr[0]}:{addr[1]}"
        with self._push_lock:
            self._connect(endpoint).send_multipart(payload)

    def _on_reshard_chunk_ready(self, msg: List[bytes]) -> None:
        self._require_frames(msg, MessageTag.RESHARD_CHUNK_READY, 4)
        room = int(msg[1].decode("ascii"))
        index = int(msg[2].decode("ascii"))
        uuid = int(msg[3].decode("ascii"))
        with self._state_lock:
            if index not in self._reshard_chunks.get(room, {}):
                logger.error(
                    "RaidenKVManager rank%d room %s asked to push unknown chunk %d",
                    self.attn_tp_rank,
                    room,
                    index,
                )
                return
            self._reshard_pushes_pending[room] = (
                self._reshard_pushes_pending.get(room, 0) + 1
            )
        self._push_executor.submit(self._do_reshard_push, room, index, uuid)

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
            str(self._local_rank).encode("ascii"),
            aux_blob,
        ]
        with self._push_lock:
            self._connect(endpoint).send_multipart(payload)

    def _on_reshard_chunk(self, msg: List[bytes]) -> None:
        self._require_frames(msg, MessageTag.RESHARD_CHUNK, 7)
        room = int(msg[1].decode("ascii"))
        index = int(msg[2].decode("ascii"))
        start = int(msg[3].decode("ascii"))
        total_pages = int(msg[4].decode("ascii"))
        blocks = unpack_ints(msg[5])
        src_rank = int(msg[6].decode("ascii"))
        with self._state_lock:
            if room not in self._reshard_meta:
                return
            per_source = self._reshard_chunk_sources.setdefault(room, {}).setdefault(
                index, {}
            )
            per_source[src_rank] = (start, blocks)
            self._remote_total[room] = total_pages
            self._announced_pages[room] = max(
                self._announced_pages.get(room, 0), start + len(blocks)
            )
            # Sources are keyed by the sender's raiden engine id; its
            # attention-TP rank is the id within its replica.
            src_attn_tp = self._reshard_meta[room]["src_attn_tp"]
            have = {nid % src_attn_tp for nid in per_source}
            if have < set(self._reshard_meta[room]["source_ranks"]):
                return
            sources_map = self._reshard_chunk_sources[room].pop(index)
        self._register_reshard_chunk(room, index, total_pages, sources_map)

    def _register_reshard_chunk(
        self,
        room: int,
        index: int,
        total_pages: int,
        sources_map: Dict[int, Tuple[int, List[int]]],
    ) -> None:
        """Registers the receive plan for one chunk and asks its sources to push.

        Every source of a decode rank holds the same pages of the request, so
        they announce the same span; the plan pairs each source's slice of it
        with the decode pages at the same position.
        """
        from tpu_sync.rpc import raiden_service_pb2 as pb2

        from sglang.srt.disaggregation.raiden.reshard_schedule import (
            build_receiver_request,
        )

        spans = {(start, len(blocks)) for start, blocks in sources_map.values()}
        if len(spans) != 1:
            logger.error(
                "RaidenKVManager rank%d room %s chunk %d announced with "
                "differing spans %s",
                self.attn_tp_rank,
                room,
                index,
                sorted(spans),
            )
            with self._state_lock:
                self._failed.add(str(room))
            return
        start, length = spans.pop()
        with self._state_lock:
            meta = self._reshard_meta[room]
            local_blocks = list(meta["local_blocks"])
            src_attn_tp = meta["src_attn_tp"]
            ready_endpoints = list(meta["ready_endpoints"])
        window = chunk_pull_window(start, length, total_pages, len(local_blocks))
        if not window:
            return
        lo, hi = window.chunk_offset, window.chunk_offset + window.length
        sources = [
            (p_rank, sources_map[p_rank][1][lo:hi]) for p_rank in sorted(sources_map)
        ]
        dst = local_blocks[window.local_offset : window.local_offset + window.length]
        with self._state_lock:
            self._uuid_seq += 1
            uuid = (self._uuid_seq << 20) | (self._local_rank & 0xFFFFF)
        req_id = f"{room}#{index}"
        req = build_receiver_request(
            pb2,
            uuid=uuid,
            req_id=req_id,
            peer=self._ensure_engine().transfer_address,
            decode_rank=self.attn_tp_rank,
            geom=self._make_geometry(src_attn_tp, self.attn_tp_size),
            sources=sources,
            dst_block_ids=dst,
            live_tokens=[self.kv_args.page_size] * window.length,
        )
        payload = [
            MessageTag.RESHARD_CHUNK_READY,
            str(room).encode("ascii"),
            str(index).encode("ascii"),
            str(uuid).encode("ascii"),
        ]
        self._publish_reshard_plan(room, req_id, uuid, req, payload, ready_endpoints)

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
            src_attn_tp = self._reshard_meta[room]["src_attn_tp"]
            have = {nid % src_attn_tp for nid in self._reshard_sources[room]}
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

    def _publish_reshard_plan(
        self,
        room: int,
        req_id: str,
        uuid: int,
        req,
        ready_payload: List[bytes],
        ready_endpoints: List[str],
        record_read: bool = True,
    ) -> bool:
        """Registers a receive plan and, if the room is still live, publishes
        it and asks its sources to push, as one step.

        Registration runs outside the state lock and can fail when raiden
        cannot stage the plan; the room can also be cleared meanwhile. The
        plan is recorded and the push requested only under the lock that
        clearing takes, so a sender is never asked to push into pages the
        room no longer owns: a failed registration fails the room, and a
        plan registered for a cleared room is dropped again.
        """
        engine = self._ensure_engine()
        try:
            engine.register_active_plan(uuid, req, is_sender=False)
        except Exception as e:
            reason = f"Raiden receive failed: cannot stage plan {req_id}: {e}"
            self.record_failure(room, reason)
            with self._state_lock:
                self._failed.add(str(room))
            logger.error("RaidenKVManager rank%d %s", self.attn_tp_rank, reason)
            return False
        with self._state_lock:
            if room in self._reshard_meta:
                if record_read:
                    self._chunk_reads.setdefault(room, []).append(req_id)
                self._reshard_plans.setdefault(room, {})[req_id] = uuid
                # Sent under the lock: after a clear wins it, no push can be
                # requested for the pages the room gave back.
                for endpoint in ready_endpoints:
                    with self._push_lock:
                        self._connect(endpoint).send_multipart(ready_payload)
                return True
        self._unregister_plan(uuid)
        return False

    def _register_receiver_and_signal(self, room: int) -> None:
        from tpu_sync.rpc import raiden_service_pb2 as pb2

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
        if self.chunked_transfer:
            # Only the pages no chunk announcement covered are left to move.
            with self._state_lock:
                announced = self._announced_pages.get(room, 0)
            any_blocks = next(iter(sources_map.values()))
            tail_start, tail = self._tail_after(any_blocks, announced)
            window = chunk_pull_window(tail_start, len(tail), len(any_blocks), n)
            with self._state_lock:
                # The closing read is owed from the moment the room can look
                # complete: it is recorded before _final_seen so completion
                # can never be inferred while its registration is pending.
                if window:
                    self._chunk_reads.setdefault(room, []).append(str(room))
                self._final_seen.add(room)
            if not window:
                for endpoint in ready_endpoints:
                    payload = [MessageTag.RESHARD_READY, str(room).encode("ascii")]
                    with self._push_lock:
                        self._connect(endpoint).send_multipart(payload)
                return
            lo = tail_start + window.chunk_offset
            sources = [
                (p_rank, sources_map[p_rank][lo : lo + window.length])
                for p_rank in sorted(sources_map)
            ]
            local_blocks = local_blocks[
                window.local_offset : window.local_offset + window.length
            ]
            n = window.length
        else:
            # A decode prefix hit leaves decode holding fewer (tail) pages than
            # the prefill produced; keep the last n of each source's blocks so
            # both sides pair the same pages.
            sources = []
            for p_rank in sorted(sources_map):
                pb = sources_map[p_rank]
                sources.append((p_rank, pb[-n:] if len(pb) > n else pb))
        live = [self.kv_args.page_size] * n

        req = build_receiver_request(
            pb2,
            uuid=room,
            req_id=str(room),
            peer=self._ensure_engine().transfer_address,
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
        self._publish_reshard_plan(
            room,
            str(room),
            room,
            req,
            [MessageTag.RESHARD_READY, str(room).encode("ascii")],
            ready_endpoints,
            record_read=not self.chunked_transfer,
        )

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
            self._reshard_pushes_pending[room] = (
                self._reshard_pushes_pending.get(room, 0) + 1
            )
        self._push_executor.submit(self._do_reshard_push, room)

    def _do_reshard_push(
        self, room: int, chunk: Optional[int] = None, uuid: Optional[int] = None
    ) -> None:
        # A push failure marks the room failed so the sender poll surfaces it
        # instead of the request hanging until the decode-side timeout.
        try:
            self._run_reshard_push(room, chunk, uuid)
        except Exception as e:
            logger.error("raiden reshard push failed for room %s: %s", room, e)
            with self._state_lock:
                self._failed.add(str(room))
        finally:
            if self.chunked_transfer:
                # The room is sent once its closing push and every chunk push
                # asked for before it have finished.
                with self._state_lock:
                    # A room cleared while its push ran has nothing to finish.
                    if room in self._reshard_pushes_pending:
                        left = self._reshard_pushes_pending[room] - 1
                        self._reshard_pushes_pending[room] = left
                        if chunk is None:
                            self._reshard_final_pushed.add(room)
                        if left == 0 and room in self._reshard_final_pushed:
                            self._completed_send.add(str(room))

    def _run_reshard_push(
        self, room: int, chunk: Optional[int] = None, uuid: Optional[int] = None
    ) -> None:
        from tpu_sync.rpc import raiden_service_pb2 as pb2

        from sglang.srt.disaggregation.raiden.reshard_schedule import (
            build_sender_request,
        )

        with self._state_lock:
            if room not in self._reshard_req:
                return  # the room was cleared before the push ran
            req_meta = dict(self._reshard_req[room])
            if chunk is None:
                prefill_blocks = list(self._send_blocks[room])
                start = self._reshard_announced.get(room, 0)
            else:
                start, prefill_blocks = self._reshard_chunks[room].pop(chunk)
            total_pages = self._send_total.get(room, len(prefill_blocks))

        decode_blocks = req_meta["decode_blocks"]
        dst_attn_tp = req_meta["dst_attn_tp"]
        push_endpoint = req_meta["push_endpoint"]
        n = len(decode_blocks)
        req_id = str(room)
        if self.chunked_transfer:
            if chunk is None:
                uuid = room
                start, prefill_blocks = self._tail_after(prefill_blocks, start)
            else:
                req_id = f"{room}#{chunk}"
            window = chunk_pull_window(start, len(prefill_blocks), total_pages, n)
            if not window:
                return
            prefill_blocks = prefill_blocks[
                window.chunk_offset : window.chunk_offset + window.length
            ]
            decode_blocks = decode_blocks[
                window.local_offset : window.local_offset + window.length
            ]
            n = window.length
        else:
            uuid = room
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
            uuid=uuid,
            req_id=req_id,
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
            engine.register_active_plan(uuid, req, is_sender=True)
            # The receiver's completion barrier is total_blocks * num_layers, and
            # each push counts its blocks once regardless of how many layers it
            # carries, so push one layer per request to deliver every layer.
            for layer_idx in range(len(self._kv_caches)):
                engine.push_registered_plan(
                    uuid, push_endpoint, prefill_blocks, decode_blocks, layer_idx
                )
            # The push is synchronous; the plan has done its work.
            engine.unregister_active_plan(uuid)
        if not self.chunked_transfer:
            with self._state_lock:
                self._completed_send.add(str(room))

    @staticmethod
    def _monotonic() -> float:
        return time.monotonic()

    # ----- completion queries (drained by poll()) ---------------------------

    def is_send_done(self, room: int) -> bool:
        rid = str(room)
        return rid in self._completed_send and (
            self._swa_engine is None or self._swa_req_id(room) in self._completed_send
        )

    def is_recv_done(self, room: int) -> bool:
        if self.chunked_transfer:
            with self._state_lock:
                if room not in self._final_seen or self._chunk_queue.get(room):
                    return False
                reads = list(self._chunk_reads.get(room, []))
            return all(rid in self._completed_recv for rid in reads) and (
                self._swa_engine is None
                or self._swa_req_id(room) in self._completed_recv
            )
        rid = str(room)
        return rid in self._completed_recv and (
            self._swa_engine is None or self._swa_req_id(room) in self._completed_recv
        )

    def is_failed(self, room: int) -> bool:
        """Whether any transfer of the room -- whole, sliding-window, or one of
        its chunks -- has failed."""
        rid, swa_rid, chunk_prefix = str(room), self._swa_req_id(room), f"{room}#"
        with self._state_lock:
            return any(
                key == rid or key == swa_rid or key.startswith(chunk_prefix)
                for key in self._failed
            )

    def clear_room(self, room: int) -> None:
        rid = str(room)
        swa_rid = self._swa_req_id(room)
        chunk_prefix = f"{room}#"

        def belongs(key: str) -> bool:
            return key == rid or key == swa_rid or key.startswith(chunk_prefix)

        with self._state_lock:
            # A read raiden still holds keeps its admission, and a push it is
            # still delivering keeps its receiver plan, each with the
            # completion that will settle it, until the engine reports it.
            held: Set[str] = set()
            if self._pulling is not None:
                for key in self._pulling.rooms():
                    if not belongs(key):
                        continue
                    if self._pull_settled_locked(key):
                        self._pulling.release(key)
                        continue
                    self._pulling_cleared.add(key)
                    held.update((key, self._paired_swa_read(key) or key))
            settled_plans = []
            plans = (getattr(self, "_reshard_plans", None) or {}).pop(room, {})
            for req_id, uuid in plans.items():
                if req_id in self._completed_recv or req_id in self._failed:
                    settled_plans.append(uuid)
                    continue
                self._cleared_plans[req_id] = uuid
                held.add(req_id)
            self._completed_send.difference_update(
                [k for k in self._completed_send if belongs(k)]
            )
            for ids in (self._completed_recv, self._failed):
                ids.difference_update([k for k in ids if belongs(k) and k not in held])
            for d in (
                getattr(self, "_chunk_queue", None),
                getattr(self, "_remote_total", None),
                getattr(self, "_announced_pages", None),
                getattr(self, "_chunk_reads", None),
                getattr(self, "_held_chunks", None),
                getattr(self, "_send_total", None),
                getattr(self, "_decode_pages", None),
                getattr(self, "_chunk_seq", None),
                getattr(self, "_reshard_chunks", None),
                getattr(self, "_pending_offer", None),
                getattr(self, "_reshard_announced", None),
                getattr(self, "_reshard_pushes_pending", None),
                getattr(self, "_reshard_chunk_sources", None),
                getattr(self, "_send_blocks", None),
                getattr(self, "_send_swa_blocks", None),
                getattr(self, "_send_aux", None),
                getattr(self, "_decode_addr", None),
                getattr(self, "_local_blocks", None),
                getattr(self, "_local_swa_blocks", None),
                getattr(self, "_decode_aux_index", None),
                getattr(self, "_remote_blocks", None),
                getattr(self, "_remote_swa_blocks", None),
                getattr(self, "_remote_endpoint", None),
                getattr(self, "_remote_swa_endpoint", None),
                getattr(self, "_reshard_req", None),
                getattr(self, "_reshard_meta", None),
                getattr(self, "_reshard_sources", None),
            ):
                if d is not None:
                    d.pop(room, None)
            if hasattr(self, "_replied"):
                self._replied.discard(room)
            for s in (
                getattr(self, "_final_seen", None),
                getattr(self, "_reshard_final_pushed", None),
                getattr(self, "_reshard_finalized", None),
                getattr(self, "_reshard_replied", None),
                getattr(self, "_reshard_ready", None),
                getattr(self, "_reshard_pushed", None),
                getattr(self, "_reshard_registered", None),
            ):
                if s is not None:
                    s.discard(room)
        for uuid in settled_plans:
            self._unregister_plan(uuid)
        self._release_host_slot(room)


class RaidenKVSender(CommonKVSender):
    def __init__(
        self,
        mgr: RaidenKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
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
        num_kv_tokens: Optional[int] = None,
    ):
        if self.kv_mgr.is_dummy_cp_rank:
            return
        self._record_transfer_indices(kv_indices, state_indices)
        swa_blocks = (
            self.kv_mgr._swa_blocks_from_state_indices(state_indices)
            if state_indices is not None
            else None
        )
        blocks = [int(x) for x in kv_indices]
        start = self.kv_mgr.append_send_blocks(self.bootstrap_room, blocks, swa_blocks)
        if self.kv_mgr.chunked_transfer and not self._last_chunk:
            self.kv_mgr.offer_chunk(
                self.bootstrap_room, start, blocks, self.num_kv_indices
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
        state_indices: Optional[List] = None,
        decode_prefix_len: Optional[int] = None,
    ):
        if self.conclude_state == KVPoll.Failed:
            return
        self.kv_mgr.register_decode_request(
            self.bootstrap_room,
            [int(x) for x in kv_indices],
            self.kv_mgr._swa_blocks_from_state_indices(state_indices),
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
        if self.kv_mgr.maybe_start_pull(self.bootstrap_room) and not self._started:
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
