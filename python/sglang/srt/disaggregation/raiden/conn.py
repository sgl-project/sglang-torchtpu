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
import contextlib
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
from sglang.srt.disaggregation.raiden.host_pool import (
    SlotAdmission,
    node_worker_rank,
    ranks_per_node,
    slot_host_bytes,
    slots_within_budget,
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
from sglang.srt.utils.network import NetworkAddress

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
        # Each rank reserves control+data ports for full KV and SWA KV. The
        # rank counts every worker on the node -- attention-TP ranks of every
        # context-parallel, attention-DP, and system-DP replica -- so
        # replicas never share a port, an engine id, or a uuid namespace.
        base_port = int(os.environ.get("SGLANG_RAIDEN_KV_PORT", "30000"))
        self._local_rank = node_worker_rank(
            self.attn_tp_rank,
            self.attn_tp_size,
            self.attn_dp_rank,
            self.attn_dp_size,
            self.system_dp_rank,
            self.attn_cp_rank,
            self.attn_cp_size,
        )
        self._control_port = base_port + 4 * self._local_rank
        self._timeout_s = float(os.environ.get("SGLANG_RAIDEN_TIMEOUT_S", "120"))
        # Raiden's own switch, read when the engine is built in this process:
        # host staging is allocated per transfer (and per reshard plan) from
        # the pool instead of in fixed slots. On unless the operator sets it
        # to 0; asymmetric attention TP (push-resharding) requires it.
        os.environ.setdefault("TPU_RAIDEN_DYNAMIC_HOST_STAGING", "1")
        self._dynamic_host_staging = (
            os.environ["TPU_RAIDEN_DYNAMIC_HOST_STAGING"] == "1"
        )
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
        # Transfer ids the engine reported failed. Only these mean raiden has
        # released the transfer's staging; a room failed for a logical reason
        # (abort, timeout, peer death) shares its id with its closing read, so
        # room failure must never read as native settlement.
        self._native_failed: Set[str] = set()
        # Rooms whose pull is issued and not yet settled. Raiden borrows one
        # host staging slot per in-flight transfer and rejects a read it cannot
        # seat, so the decode side admits at most one pull per slot and leaves
        # the rest in the transfer queue.
        self._pulling: Optional[SlotAdmission] = None
        # Reads of rooms the framework dropped while they were still in
        # flight. Raiden cannot cancel a read, so the read keeps its admission
        # until the engine reports it settled.
        self._pulling_cleared: Set[str] = set()
        # Rooms this side gave up on, by when. The peer (or this side's own
        # compute) may still act on them; frames and send calls for a
        # tombstoned room are dropped, so it cannot be resurrected. Entries
        # age out rather than leaving with the room, because frames can
        # arrive after the room is cleared.
        self._aborted: Dict[int, float] = {}
        # Receiver plans of rooms the framework dropped while their push was
        # still in flight, by request id. Raiden cannot cancel a push and
        # keeps delivering into the plan, so the plan stays registered until
        # the engine reports the push settled.
        self._cleared_plans: Dict[str, int] = {}
        # Sender plans whose D2H stage timed out while the copy was still
        # running, as (uuid, room, future). Unregistering would free the
        # staging under the copy -- and the room's pages are what the copy
        # reads -- so both are kept until the future settles.
        self._stale_plans: List[Tuple[int, int, Any]] = []
        self._full_engine_max_blocks = 0
        self._swa_engine_max_blocks = 0
        self._engine_lock = threading.Lock()
        # Serializes every request_status read/update/clear for this manager,
        # so a terminal Failed can never interleave with a progress update.
        self._status_lock = threading.Lock()
        # Orders chunk announcements against the closing reply. Both travel
        # the same socket, so whatever is sent first also arrives first; this
        # lock decides that order, and an announcement that loses the race is
        # skipped -- its pages then ride the closing transfer.
        self._announce_lock = threading.Lock()
        self._state_lock = threading.Lock()

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # room -> accumulated page block-ids, aux blob, decode reply addr
            self._send_blocks: Dict[int, List[int]] = {}
            self._send_swa_blocks: Dict[int, List[int]] = {}
            self._send_aux: Dict[int, bytes] = {}
            self._decode_addr: Dict[int, Tuple[str, int]] = {}
            self._replied: Set[int] = set()
            # Reads registered for the decode to pull, by request id and
            # publish time. A lease is the room's claim on its pages: it
            # starts before the (possibly blocking) registration, settles
            # when the engine reports the read served or failed, and ages
            # out after the transfer deadline for a read the decode never
            # comes for.
            self._send_reads: Dict[int, Set[str]] = {}
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
            # Page spans whose chunk plan this decode registered; the
            # closing plan covers exactly the pages outside their union, so a
            # chunk that lost its race anywhere in the range is not skipped.
            self._reshard_covered: Dict[int, List[Tuple[int, int]]] = {}
            # Pages each decode declared for its closing push, from its
            # RESHARD_READY. The closing push moves exactly these.
            self._reshard_ready_pages: Dict[int, List[int]] = {}
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
            # Detects a prefill that stopped answering and fails its rooms.
            self._start_heartbeat_checker_thread()
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
            # Page spans covered by chunk announcements, as (start, end).
            # Chunks are announced from two threads and can cross on the wire,
            # so a later span can arrive first and the covered set can have
            # holes; the closing read pulls exactly the uncovered pages.
            self._announced_spans: Dict[int, List[Tuple[int, int]]] = {}
            self._chunk_reads: Dict[int, List[str]] = {}
            self._final_seen: Set[int] = set()
            # Chunk announcements from each resharding source, by chunk index,
            # until every source of the room has announced it; the receive
            # plans registered for the room, by their request id.
            self._reshard_chunk_sources: Dict[
                int, Dict[int, Dict[int, Tuple[int, List[int]]]]
            ] = {}
            # Chunk indexes whose receive-plan registration is in flight; the
            # closing plan waits for them so completion never overtakes one.
            self._reshard_chunk_registering: Dict[int, Set[int]] = {}
            self._reshard_plans: Dict[int, Dict[str, int]] = {}
            self._uuid_seq = 0

        # Asymmetric-TP resharding state, populated lazily on the push path.
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # room -> decode meta from RESHARD_REQ
            self._reshard_req: Dict[int, dict] = {}
            self._reshard_finalized: Set[int] = set()  # send accumulation complete
            # Chunks the decode acknowledged with a push request. Only these
            # leave the closing transfer: an unacknowledged chunk's pages are
            # still owed, so the closing span keeps them.
            self._reshard_chunk_acked: Dict[int, Set[int]] = {}
            self._reshard_replied: Set[int] = set()  # RESHARD_BLOCKS sent
            self._reshard_ready: Set[int] = set()  # rooms cleared to push
            self._reshard_replied_at: Dict[int, float] = {}  # RESHARD_BLOCKS sent at
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

        engines = 2 if self._swa_kv_caches else 1
        max_blocks = self._max_request_blocks()
        num_slots = self._num_slots(self._kv_caches, max_blocks, engines)
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
        self._full_engine_max_blocks = max_blocks
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
            swa_num_slots = self._num_slots(self._swa_kv_caches, swa_max_blocks, 2)
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

    def _slot_ceiling(self) -> int:
        """Slots that can ever be claimed at once.

        A transfer holds one slot from the moment it is issued until its H2D
        lands. The decode admits a request into the transfer queue before it
        joins the running batch, so the reach is the batch plus the extra
        in-transfer slots reserved alongside it; past that, slots are
        unclaimable.
        """
        batch = int(self.server_args.max_running_requests or 64)
        in_transfer = int(
            getattr(self.server_args, "disaggregation_decode_extra_slots", 0) or 0
        )
        return max(1, batch + in_transfer)

    def _num_slots(
        self, kv_caches: List[Any], max_blocks: int, engines: int = 1
    ) -> int:
        """Slots this engine can afford within the host staging budget.

        The per-node budget is split across the ranks resident on this host
        and across the engines this rank builds, then divided by what one slot
        costs. Deriving the count keeps a large context length from turning
        into a host allocation that fails at startup -- it lowers the transfer
        concurrency instead, and refuses to start when not even one slot fits
        the budget.
        """
        budget = float(os.environ.get("SGLANG_RAIDEN_HOST_POOL_GB", "128")) * 1024**3
        local_ranks = ranks_per_node(
            self.attn_tp_size,
            self.attn_dp_size,
            self.system_dp_size,
            self.server_args.nnodes,
            self.attn_cp_size,
        )
        budget /= local_ranks * max(1, engines)
        slot_bytes = slot_host_bytes(kv_caches, max_blocks)
        slots = slots_within_budget(budget, slot_bytes, self._slot_ceiling())
        logger.info(
            "RaidenKVManager rank%d slot budget | max_blocks=%d slot=%.1fMB "
            "rank_budget=%.1fGB slots=%d",
            self.attn_tp_rank,
            max_blocks,
            slot_bytes / (1024**2),
            budget / (1024**3),
            slots,
        )
        return slots

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
                    norm_failed = {self._norm_req_id(r) for r in failed}
                    self._native_failed.update(norm_failed)
                    self._failed.update(norm_failed)
                    self._release_pull_slots_locked()
                    self._settle_sender_reads_locked()
                    settled_plans = self._settled_cleared_plans_locked()
                for uuid in settled_plans:
                    self._unregister_plan(uuid)
                if failed:
                    logger.error(
                        "RaidenKVManager rank%d failed=%s", self.attn_tp_rank, failed
                    )
        with self._state_lock:
            settled_stale = [e for e in self._stale_plans if e[2].is_ready()]
            for entry in settled_stale:
                self._stale_plans.remove(entry)
        for uuid, _room, _fut in settled_stale:
            self._unregister_plan(uuid)

    def check_status(self, bootstrap_room: int) -> KVPoll:
        with self._status_lock:
            return super().check_status(bootstrap_room)

    def update_status(self, bootstrap_room: int, status: KVPoll) -> None:
        """Make CommonKVManager's sticky terminal transition atomic for Raiden."""
        with self._status_lock:
            return super().update_status(bootstrap_room, status)

    def clear_status(self, bootstrap_room: int) -> None:
        """Serialize room cleanup with every Raiden status transition."""
        with self._status_lock:
            return super().clear_status(bootstrap_room)

    def _fail_room_locked(
        self, room: int, reason: str, *, preserve_reason: bool = False
    ) -> None:
        """Publish a room failure while holding ``_state_lock``.

        ``clear_room`` takes the same lock while removing both the transfer
        marker and its diagnostic. Keeping the failure record write inside
        this transaction means a poll can never observe ``_failed`` before
        its reason exists, and a late writer cannot recreate a record after
        clear.
        """
        if preserve_reason:
            with self.failure_lock:
                self.failure_records.setdefault(room, reason)
        else:
            self.record_failure(room, reason)
        self._failed.add(str(room))

    def _handle_node_failure(self, failed_bootstrap_addr: str) -> None:
        """Fail the rooms of a prefill the heartbeat declared down and close
        this rank's sockets to it, so later sends open fresh ones instead of
        queueing on a dead peer."""
        with self.connection_lock:
            stale = set()
            for key, infos in self.connection_pool.items():
                if not key.startswith(failed_bootstrap_addr):
                    continue
                for info in infos:
                    ip, port = info.get("rank_ip"), info.get("rank_port")
                    if ip and port:
                        stale.add(NetworkAddress(ip, int(port)).to_tcp())
        # Native admission and progress publication use _state_lock. Publish
        # the heartbeat failure under that same lock so update_status cannot
        # observe the old state, lose a race to Failed, and then revive the
        # room with a later Transferring write.
        with self._state_lock:
            super()._handle_node_failure(failed_bootstrap_addr)
        for endpoint in stale:
            self._disconnect_endpoint(endpoint)

    def _on_heartbeat_success(self, bootstrap_addr: str) -> None:
        """Drop settled rooms from the room tracker, which otherwise keeps
        one entry per request served by ``bootstrap_addr`` for the process
        lifetime and is only consumed when that prefill dies."""
        tracked = self.addr_to_rooms_tracker[bootstrap_addr].copy()
        for room in tracked:
            if room not in self.request_status:
                self.addr_to_rooms_tracker[bootstrap_addr].discard(room)

    def _send_multipart_locked(
        self, endpoint: str, parts: List[bytes], is_ipv6: bool = False
    ):
        """Sends on the endpoint's cached socket, re-fetching a socket that
        was disconnected between the lookup and the send.

        A peer declared dead has its socket closed and dropped from the
        cache; a sender that looked its socket up just before that would
        otherwise send on the closed object. Under the send lock the socket
        is only used while it is still the cached one.
        """
        while True:
            sock = self._connect(endpoint, is_ipv6=is_ipv6)
            with self._socket_send_locks[endpoint]:
                with self._socket_lock:
                    if self._socket_cache.get(endpoint) is not sock:
                        continue
                sock.send_multipart(parts)
                return

    def _disconnect_endpoint(self, endpoint: str) -> None:
        """Close and forget the cached PUSH socket for ``endpoint``; the next
        send to it connects anew. Takes the endpoint's send lock so a send in
        progress finishes first."""
        with self._socket_lock:
            sock = self._socket_cache.pop(endpoint, None)
            monitor = self._monitor_cache.pop(endpoint, None)
            send_lock = self._socket_send_locks.get(endpoint)
        if sock is None and monitor is None:
            return
        with send_lock if send_lock is not None else contextlib.nullcontext():
            if monitor is not None:
                monitor.close()
            if sock is not None:
                sock.close(linger=0)

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
                self._dispatch(msg)
            except Exception as e:
                logger.error("RaidenKVManager message handling error: %s", e)

    def _dispatch(self, msg: List[bytes]) -> None:
        """Route one multipart message from the per-rank PULL socket by its tag."""
        if msg and msg[0] == MessageTag.ABORT:
            self._on_abort(msg)
            return
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
        elif msg[0] == MessageTag.CHUNK:
            self._on_chunk(msg)
        elif msg[0] == MessageTag.RESHARD_CHUNK:
            self._on_reshard_chunk(msg)
        elif msg[0] == MessageTag.RESHARD_CHUNK_READY:
            self._on_reshard_chunk_ready(msg)
        else:
            logger.warning("RaidenKVManager unknown header: %s", msg[0])

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
            if room in self._aborted:
                return
            self._decode_addr[room] = (decode_ip, decode_port)
            self._decode_pages[room] = decode_pages
            held = self._held_chunks.pop(room, [])
            total_pages = self._send_total.get(room, 0)
        for start, block_ids in held:
            self._announce_chunk(room, start, block_ids, total_pages)
        self._maybe_reply(room)

    @staticmethod
    def _uncovered_pages(total: int, spans: List[Tuple[int, int]]) -> List[int]:
        """Pages of ``[0, total)`` no span covers, in order.

        A chunk announcement can be skipped when it loses the race with the
        closing reply, so the covered set is not a prefix: whatever it misses
        anywhere in the range rides the closing transfer.
        """
        covered = [False] * total
        for lo, hi in spans:
            for pg in range(max(lo, 0), min(hi, total)):
                covered[pg] = True
        return [pg for pg in range(total) if not covered[pg]]

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
            if room in self._aborted:
                return
            self._remote_total[room] = total_pages
            self._announced_spans.setdefault(room, []).append(
                (start, start + len(blocks))
            )
            self._chunk_queue.setdefault(room, []).append(
                (
                    req_id,
                    uuid,
                    list(range(start, start + len(blocks))),
                    blocks,
                    endpoint,
                )
            )

    def _on_abort(self, msg: List[bytes]) -> None:
        """Fail a room the decode gave up on: ``[ABORT, room, decode ip, port]``.

        The room's send state is dropped and later send calls for it are
        ignored, so a prefill still computing the request cannot register a
        read nobody will pull. A room that already succeeded is left
        untouched; a room this rank does not know yet is tombstoned, so an
        abort that outruns the pull request (they travel on different
        sockets) or the sender's creation still wins.
        """
        try:
            room = int(msg[1].decode("ascii"))
        except (IndexError, ValueError) as e:
            logger.debug("RaidenKVManager ignoring malformed abort: %s", e)
            return
        with self._state_lock:
            # The status check, the tombstone, the failure record, and the
            # Failed status land as one step. A concurrent finalize either
            # sees the tombstone or is overwritten here; a concurrent clear
            # pops the status before its locked sweep, so an abort either
            # sees no status and only tombstones, or writes its records
            # before the sweep that removes them.
            try:
                status = self.check_status(room)
            except KeyError:
                status = None
            if status == KVPoll.Success:
                return
            self._mark_aborted_locked(room)
            if status is None:
                return
            self._fail_room_locked(room, "Aborted by decode-side abort notification.")
            self.update_status(room, KVPoll.Failed)
        self._drop_room_state(room)

    def _mark_aborted_locked(self, room: int) -> None:
        now = time.time()
        self._aborted[room] = now
        horizon = 4 * max(getattr(self, "waiting_timeout", 300.0) or 300.0, 150.0)
        for r, at in list(self._aborted.items()):
            if now - at > horizon:
                del self._aborted[r]

    def is_aborted(self, room: int) -> bool:
        with self._state_lock:
            return room in self._aborted

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
            if room in self._aborted:
                return
            self._remote_blocks[room] = remote_blocks
            self._remote_endpoint[room] = endpoint
            self._remote_swa_blocks[room] = remote_swa_blocks
            self._remote_swa_endpoint[room] = swa_endpoint
            aux_index = self._decode_aux_index.get(room)
            if self.chunked_transfer:
                spans = self._announced_spans.get(room, [])
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
                missing = self._uncovered_pages(total, spans)
                if missing:
                    self._chunk_queue.setdefault(room, []).append(
                        (
                            str(room),
                            room,
                            missing,
                            [remote_blocks[p] for p in missing],
                            endpoint,
                        )
                    )
                self._final_seen.add(room)
            # Written under the lock: a room failing concurrently frees its
            # metadata slot, and the write must never land in a reused one.
            if aux_index is not None and aux_blob and room not in self._aborted:
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
    ) -> Optional[int]:
        """Records a chunk's pages and returns its start index in the request.

        None means the decode aborted the room; the pages are not recorded and
        the caller must not offer them.
        """
        with self._state_lock:
            if room in self._aborted:
                return None
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
            if room in self._aborted:
                return
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
        with self._announce_lock:
            with self._state_lock:
                if (
                    room in self._replied
                    or room in self._aborted
                    or room not in self._decode_pages
                ):
                    # The closing reply is out (or the room is gone): these
                    # pages travel with the closing transfer instead. A late
                    # announcement must never follow the reply -- the sender
                    # counts the room done when the closing read completes.
                    return
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
            self._register_sender_read(
                room, self._ensure_engine(), req_id, uuid, block_ids
            )
            with self._state_lock:
                # Registration may block while an abort seals the room. Keep
                # the native lease, but never publish a fresh pull after the
                # tombstone is visible.
                if room in self._aborted:
                    return
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
            self._send_multipart_locked(endpoint, payload)

    def finalize_send(self, room: int, aux_index: Optional[int]) -> bool:
        """Announce the room's blocks to the decode. False if the room was aborted.

        The room turns Transferring under the same lock that an abort takes
        to fail it, so the two cannot interleave: an abort either sees the
        finalized room and fails it terminally, or wins and this returns
        False without touching the status.
        """
        self._ensure_engine()
        with self._state_lock:
            if room in self._aborted:
                return False
            if aux_index is not None:
                self._send_aux[room] = read_aux_blob(
                    self.kv_args.aux_data_ptrs,
                    self.kv_args.aux_item_lens,
                    aux_index,
                )
            self._reshard_finalized.add(room)
            is_reshard = room in self._reshard_req
            self.update_status(room, KVPoll.Transferring)
        # The request type is selected by the decode handshake, not by whether
        # reshard support is globally enabled. Symmetric TP uses PULL_REQ even
        # when resharding is available; asymmetric TP uses RESHARD_REQ.
        if is_reshard:
            self._maybe_reshard_reply(room)
        else:
            self._maybe_reply(room)
        return True

    def awaiting_decode_since(self, room: int, since: float) -> Optional[float]:
        """When the prefill started waiting on the decode for ``room``.

        None once the transfer no longer depends on a decode message: the
        decode has asked to pull, or has cleared a resharded room to push.
        Also None while a resharding decode has asked for the room but the
        prefill has not announced its blocks -- the prefill may sit in its
        own queue for any while, and until it announces, the decode owes
        nothing. Once blocks (or a chunk's push request) put the ball in the
        decode's court, the window is timed from that latest progress.
        """
        with self._state_lock:
            if room in self._decode_addr or room in self._reshard_ready:
                return None
            if room in self._reshard_req:
                return self._reshard_replied_at.get(room)
            return since

    def fail_decode_silence(self, room: int, since: float, timeout: float) -> bool:
        """Fails ``room`` when the decode has owed it a message for ``timeout``.

        The deadline check and the failure are one step under the lock, so a
        pull or reshard message that lands concurrently either arrives first
        (the room stays live) or finds the tombstone and is dropped -- the
        decode can never be told to pull KV whose room just failed.
        """
        with self._state_lock:
            if room in self._decode_addr or room in self._reshard_ready:
                return False
            if room in self._reshard_req:
                started = self._reshard_replied_at.get(room)
            else:
                started = since
            if started is None or time.time() - started < timeout:
                return False
            self._mark_aborted_locked(room)
            self._fail_room_locked(
                room,
                f"Request {room} timed out after {time.time() - since:.1f}s "
                "waiting for the decode to ask for its KV",
            )
            self.update_status(room, KVPoll.Failed)
        return True

    def _register_sender_read(
        self, room: int, engine: Any, req_id: str, uuid: int, block_ids: List[int]
    ) -> None:
        """Register one producer read while retaining its source-page lease.

        ``register_read`` publishes physical device block ids to raiden. The
        consumer may not pull them until later, so the lease starts before
        the potentially blocking registration and ends only when poll_stats
        reports the request id settled: raiden expires a read nobody pulls at
        its own deadline and reports it, so no second timer runs here. A
        false return means raiden accepted no work, so no lease is left
        behind; a registration failure fails the room.
        """
        with self._state_lock:
            if room in self._aborted:
                return
            self._send_reads.setdefault(room, set()).add(req_id)
        try:
            accepted = bool(engine.register_read(req_id, uuid, block_ids))
        except Exception as exc:
            with self._state_lock:
                self._discard_sender_read_locked(room, req_id)
                if room not in self._aborted:
                    self._mark_aborted_locked(room)
                    self._fail_room_locked(
                        room, f"Raiden register_read failed for {req_id}: {exc}"
                    )
                    self.update_status(room, KVPoll.Failed)
            raise
        if not accepted:
            with self._state_lock:
                self._discard_sender_read_locked(room, req_id)

    def _discard_sender_read_locked(self, room: int, req_id: str) -> None:
        reads = self._send_reads.get(room)
        if reads is None:
            return
        reads.discard(req_id)
        if not reads:
            self._send_reads.pop(room, None)

    def _settle_sender_reads_locked(self) -> None:
        settled = self._completed_send | self._native_failed
        for room, reads in list(self._send_reads.items()):
            reads -= settled
            if not reads:
                self._send_reads.pop(room, None)

    def sender_has_active_io(self, room: int) -> bool:
        """Whether the room's pages may still be under a transfer this side
        serves: a registered read the decode can pull any moment, a running
        push, or a parked staging copy.

        While any holds, the pages must not be handed back -- the transfer
        would read them under their next owner. A read the decode never
        comes for is expired by raiden at its own deadline and settles
        through poll_stats like any other.
        """
        with self._state_lock:
            if self._send_reads.get(room):
                return True
            if getattr(self, "_reshard_pushes_pending", {}).get(room, 0) > 0:
                return True
            for entry in getattr(self, "_stale_plans", []):
                # A parked staging copy still reads the room's pages.
                if len(entry) >= 3 and entry[1] == room:
                    return True
        return False

    def _maybe_reply(self, room: int) -> None:
        with self._state_lock:
            if room in self._replied or room in self._aborted:
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
        with self._announce_lock:
            if self._swa_engine is not None:
                self._register_sender_read(
                    room, self._swa_engine, self._swa_req_id(room), room, swa_blocks
                )
            self._register_sender_read(
                room, self._ensure_engine(), str(room), room, blocks
            )
            with self._state_lock:
                # The room can fail while its reads register; the decode is
                # gone and its own handlers drop frames for the room.
                if room in self._aborted:
                    return
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
            self._send_multipart_locked(endpoint, payload)

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
            # head-slice strided into the wider decode block. The plan stages
            # in host blocks raiden allocates for it, which only per-transfer
            # staging provides.
            if not self._dynamic_host_staging:
                raise NotImplementedError(
                    "Raiden asymmetric attention TP (got "
                    f"{len(bootstrap_infos)} prefill sources for one decode "
                    "rank) needs per-transfer host staging; do not set "
                    "TPU_RAIDEN_DYNAMIC_HOST_STAGING=0."
                )
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
            self._send_multipart_locked(endpoint, payload)

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
        settled = key in self._completed_recv or key in self._native_failed
        swa_rid = self._paired_swa_read(key)
        if swa_rid is not None:
            settled = settled and (
                swa_rid in self._completed_recv or swa_rid in self._native_failed
            )
        return settled

    def _settled_cleared_plans_locked(self) -> List[int]:
        """Plan uuids of cleared rooms whose push the engine has now settled.

        Their completion bookkeeping is dropped here, as nothing else will
        read it; the caller unregisters the plans outside the lock.
        """
        uuids = []
        for req_id, uuid in list(self._cleared_plans.items()):
            if req_id in self._completed_recv or req_id in self._native_failed:
                del self._cleared_plans[req_id]
                self._completed_recv.discard(req_id)
                self._failed.discard(req_id)
                self._native_failed.discard(req_id)
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

    def _reap_stale_plan(self, uuid: int, room: int, fut) -> None:
        """Waits out a parked plan's copy, then drops the plan.

        The request that parked it is already failed and may be the last one
        served, so reclaiming cannot ride on another request's poll.
        """
        while True:
            with self._state_lock:
                if (uuid, room, fut) not in self._stale_plans:
                    return  # a completion sweep already reclaimed it
                if fut.is_ready():
                    self._stale_plans.remove((uuid, room, fut))
                    break
            time.sleep(0.05)
        self._unregister_plan(uuid)

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
                    self._native_failed.discard(rid)

    def _reject_oversized_pull(
        self,
        room: int,
        pages: int,
        max_blocks: int,
        cache_kind: str,
        override_env: str,
    ) -> None:
        """Fail a request whose pages cannot fit one host staging slot."""
        reason = (
            f"Raiden {cache_kind} receive failed: request needs {pages} pages "
            f"but a host staging slot holds {max_blocks}. Raise {override_env} "
            "if this request size is expected."
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
                req_id, uuid, pages, blocks, endpoint = queue[0]
                # The decode holds the last len(local) pages; pages below that
                # base were already here, or reserved for the first decode
                # token, and are not pulled.
                base = total - len(local)
                pairs = [
                    (blk, local[pg - base])
                    for pg, blk in zip(pages, blocks)
                    if pg >= base
                ]
                if not pairs:
                    queue.pop(0)
                    continue
                if len(pairs) > self._full_engine_max_blocks:
                    oversized = (
                        len(pairs),
                        self._full_engine_max_blocks,
                        "full-attention",
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
                            "SWA",
                            "SGLANG_RAIDEN_SWA_MAX_BLOCKS",
                        )
                        break
                if room in self._aborted:
                    # The room failed; its pages are no longer this read's.
                    return started
                if not self._pulling.admit(req_id):
                    return started
                # Status moves with the admission, under the same lock an
                # abort takes: a Transferring write can never follow the
                # abort's Failed.
                self.update_status(room, KVPoll.Transferring)
                queue.pop(0)
                remote_slice = [blk for blk, _ in pairs]
                local_slice = [loc for _, loc in pairs]
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
        full_pages = len(set(local))
        if full_pages > self._full_engine_max_blocks:
            self._reject_oversized_pull(
                room,
                full_pages,
                self._full_engine_max_blocks,
                "full-attention",
                "SGLANG_RAIDEN_MAX_BLOCKS",
            )
            return False
        if self._swa_engine is not None:
            remote_swa, local_swa = self._pair_blocks(remote_swa, local_swa)
            swa_pages = len(set(local_swa))
            if swa_pages > self._swa_engine_max_blocks:
                self._reject_oversized_pull(
                    room,
                    swa_pages,
                    self._swa_engine_max_blocks,
                    "SWA",
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
            if room in self._aborted or not self._pulling.admit(str(room)):
                return False
            # Status moves with the admission, under the same lock an abort
            # takes: a Transferring write can never follow the abort's Failed.
            self.update_status(room, KVPoll.Transferring)
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
            self._send_multipart_locked(endpoint, payload)

    def _on_reshard_req(self, msg: List[bytes]) -> None:
        self._require_frames(msg, MessageTag.RESHARD_REQ, 7)
        room = int(msg[1].decode("ascii"))
        decode_ip = msg[2].decode("ascii")
        decode_port = int(msg[3].decode("ascii"))
        push_endpoint = msg[4].decode("ascii")
        decode_blocks = unpack_ints(msg[5])
        dst_attn_tp = int(msg[6].decode("ascii"))
        with self._state_lock:
            if room in self._aborted:
                return
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
        with self._announce_lock:
            with self._state_lock:
                if (
                    room in self._reshard_replied
                    or room in self._aborted
                    or room not in self._reshard_req
                ):
                    # The closing message is out (or the room is gone): these
                    # pages travel with the closing transfer, whose span the
                    # decode decides from the chunks that did arrive.
                    return
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
            self._send_multipart_locked(endpoint, payload)

    def _on_reshard_chunk_ready(self, msg: List[bytes]) -> None:
        self._require_frames(msg, MessageTag.RESHARD_CHUNK_READY, 4)
        room = int(msg[1].decode("ascii"))
        index = int(msg[2].decode("ascii"))
        uuid = int(msg[3].decode("ascii"))
        with self._state_lock:
            if room in self._aborted:
                return
            chunk = self._reshard_chunks.get(room, {}).get(index)
            if chunk is None:
                logger.error(
                    "RaidenKVManager rank%d room %s asked to push unknown chunk %d",
                    self.attn_tp_rank,
                    room,
                    index,
                )
                return
            acked = self._reshard_chunk_acked.setdefault(room, set())
            if index in acked:
                return
            acked.add(index)
            self._reshard_pushes_pending[room] = (
                self._reshard_pushes_pending.get(room, 0) + 1
            )
            # The decode is making progress; time the wait for its next
            # message from here.
            self._reshard_replied_at[room] = time.time()
        self._push_executor.submit(self._do_reshard_push, room, index, uuid)

    def _maybe_reshard_reply(self, room: int) -> None:
        with self._state_lock:
            # Only reply once, and only after send() finished accumulating this
            # room's blocks (a RESHARD_REQ may arrive mid-accumulation).
            if room in self._reshard_replied:
                return
            if room not in self._reshard_req or room not in self._reshard_finalized:
                return
            addr = self._reshard_req[room]["reply_addr"]
            blocks = list(self._send_blocks.get(room, []))
            aux_blob = self._send_aux.get(room, b"")
        with self._announce_lock:
            with self._state_lock:
                if room not in self._reshard_req:
                    return
                # Committed under the order lock: no announcement can follow.
                self._reshard_replied.add(room)
                self._reshard_replied_at[room] = time.time()
            endpoint = f"tcp://{addr[0]}:{addr[1]}"
            payload = [
                MessageTag.RESHARD_BLOCKS,
                str(room).encode("ascii"),
                pack_ints(blocks),
                str(self._local_rank).encode("ascii"),
                aux_blob,
            ]
            self._send_multipart_locked(endpoint, payload)

    def _on_reshard_chunk(self, msg: List[bytes]) -> None:
        self._require_frames(msg, MessageTag.RESHARD_CHUNK, 7)
        room = int(msg[1].decode("ascii"))
        index = int(msg[2].decode("ascii"))
        start = int(msg[3].decode("ascii"))
        total_pages = int(msg[4].decode("ascii"))
        blocks = unpack_ints(msg[5])
        src_rank = int(msg[6].decode("ascii"))
        with self._state_lock:
            if room not in self._reshard_meta or room in self._aborted:
                return
            per_source = self._reshard_chunk_sources.setdefault(room, {}).setdefault(
                index, {}
            )
            per_source[src_rank] = (start, blocks)
            self._remote_total[room] = total_pages
            # Sources are keyed by the sender's raiden engine id; its
            # attention-TP rank is the id within its replica.
            src_attn_tp = self._reshard_meta[room]["src_attn_tp"]
            have = {nid % src_attn_tp for nid in per_source}
            if have < set(self._reshard_meta[room]["source_ranks"]):
                return
            sources_map = self._reshard_chunk_sources[room].pop(index)
            self._reshard_chunk_registering.setdefault(room, set()).add(index)
        accepted_end = None
        try:
            accepted_end = self._register_reshard_chunk(
                room, index, total_pages, sources_map
            )
        finally:
            with self._state_lock:
                registering = self._reshard_chunk_registering.get(room)
                if registering is not None:
                    registering.discard(index)
                    if not registering:
                        self._reshard_chunk_registering.pop(room, None)
                # A span is covered only once its chunk plan is published,
                # so the closing transfer keeps every span the sources still
                # owe -- including one that lost its race mid-range.
                if accepted_end is not None and room in self._reshard_meta:
                    self._reshard_covered.setdefault(room, []).append(accepted_end)
        self._maybe_register_reshard_final(room)

    def _register_reshard_chunk(
        self,
        room: int,
        index: int,
        total_pages: int,
        sources_map: Dict[int, Tuple[int, List[int]]],
    ) -> Optional[int]:
        """Registers the receive plan for one chunk and asks its sources to push.

        Returns the absolute end of the accepted span, or None when nothing
        was published.

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
            reason = (
                f"Raiden receive failed: room {room} chunk {index} announced "
                f"with differing spans {sorted(spans)}"
            )
            logger.error("RaidenKVManager rank%d %s", self.attn_tp_rank, reason)
            with self._state_lock:
                if room not in self._reshard_meta:
                    return None
                self._failed.add(str(room))
                self.record_failure(room, reason)
            return None
        start, length = spans.pop()
        with self._state_lock:
            meta = self._reshard_meta[room]
            local_blocks = list(meta["local_blocks"])
            src_attn_tp = meta["src_attn_tp"]
            ready_endpoints = list(meta["ready_endpoints"])
        window = chunk_pull_window(start, length, total_pages, len(local_blocks))
        if not window:
            return None
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
        if not self._publish_reshard_plan(
            room, req_id, uuid, req, payload, ready_endpoints
        ):
            return None
        return (start, start + length)

    def _on_reshard_blocks(self, msg: List[bytes]) -> None:
        self._require_frames(msg, MessageTag.RESHARD_BLOCKS, 5)
        room = int(msg[1].decode("ascii"))
        prefill_blocks = unpack_ints(msg[2])
        src_rank = int(msg[3].decode("ascii"))
        aux_blob = msg[4]
        ready = False
        aux_index = None
        with self._state_lock:
            if room not in self._reshard_meta or room in self._aborted:
                return
            self._reshard_sources[room][src_rank] = prefill_blocks
            aux_index = self._reshard_meta[room].get("aux_index")
            src_attn_tp = self._reshard_meta[room]["src_attn_tp"]
            have = {nid % src_attn_tp for nid in self._reshard_sources[room]}
            need = set(self._reshard_meta[room]["source_ranks"])
            ready = have >= need
            # Written under the lock, for the same reason as the pull path.
            if aux_index is not None and aux_blob:
                write_aux_blob(
                    self.kv_args.aux_data_ptrs,
                    self.kv_args.aux_item_lens,
                    aux_index,
                    aux_blob,
                )
        if ready:
            self._maybe_register_reshard_final(room)

    def _maybe_register_reshard_final(self, room: int) -> None:
        """Publish the closing plan only once no chunk can still change it.

        Every source's closing message has arrived (each source's chunks
        precede it on its ordered endpoint), so a chunk group still partial
        here can never complete: it was never acknowledged, every sender
        keeps its pages in the closing transfer, and the partial metadata is
        discarded. A chunk whose registration is still in flight is waited
        out, so the room cannot look complete while it is pending.
        """
        with self._state_lock:
            if (
                room not in self._reshard_meta
                or str(room) in self._failed
                or room in self._reshard_registered
                or self._reshard_chunk_registering.get(room)
            ):
                return
            meta = self._reshard_meta[room]
            src_attn_tp = meta["src_attn_tp"]
            have = {nid % src_attn_tp for nid in self._reshard_sources[room]}
            if have < set(meta["source_ranks"]):
                return
            partial = self._reshard_chunk_sources.pop(room, None)
            if partial:
                logger.debug(
                    "RaidenKVManager rank%d room %s folds partial reshard "
                    "chunks %s into the closing transfer",
                    self.attn_tp_rank,
                    room,
                    sorted(partial),
                )
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
            logger.error("RaidenKVManager rank%d %s", self.attn_tp_rank, reason)
            with self._state_lock:
                if room not in self._reshard_meta:
                    # Cleared while registering: nothing reads a failure for
                    # this room any more, so record none.
                    return False
                # State and reason land as one step, so a poll can never see
                # the failure without its reason and a concurrent clear can
                # never leave either behind.
                self._failed.add(str(room))
                self.record_failure(room, reason)
            return False
        with self._state_lock:
            if room in self._reshard_meta and room not in self._aborted:
                if record_read:
                    self._chunk_reads.setdefault(room, []).append(req_id)
                self._reshard_plans.setdefault(room, {})[req_id] = uuid
                # Sent under the lock: after a clear wins it, no push can be
                # requested for the pages the room gave back.
                for endpoint in ready_endpoints:
                    self._send_multipart_locked(endpoint, ready_payload)
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
            # Only the pages no registered chunk plan covers are left to
            # move. The covered set is not a prefix -- a chunk can lose its
            # race anywhere in the range -- so the closing plan is its exact
            # complement, and the READY tells the sources to push just that.
            any_blocks = next(iter(sources_map.values()))
            total = len(any_blocks)
            base = total - n
            with self._state_lock:
                covered = list(self._reshard_covered.get(room, []))
            missing = [pg for pg in self._uncovered_pages(total, covered) if pg >= base]
            with self._state_lock:
                if room not in self._reshard_meta:
                    return  # cleared while the complement was computed
                # The closing read is owed from the moment the room can look
                # complete: it is recorded before _final_seen so completion
                # can never be inferred while its registration is pending.
                if missing:
                    self._chunk_reads.setdefault(room, []).append(str(room))
                self._final_seen.add(room)
            payload_pages = pack_ints(missing)
            if not missing:
                for endpoint in ready_endpoints:
                    payload = [
                        MessageTag.RESHARD_READY,
                        str(room).encode("ascii"),
                        payload_pages,
                    ]
                    self._send_multipart_locked(endpoint, payload)
                return
            sources = [
                (p_rank, [sources_map[p_rank][pg] for pg in missing])
                for p_rank in sorted(sources_map)
            ]
            local_blocks = [local_blocks[pg - base] for pg in missing]
            n = len(missing)
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
        ready_payload = [MessageTag.RESHARD_READY, str(room).encode("ascii")]
        if self.chunked_transfer:
            ready_payload.append(pack_ints(missing))
        self._publish_reshard_plan(
            room,
            str(room),
            room,
            req,
            ready_payload,
            ready_endpoints,
            record_read=not self.chunked_transfer,
        )

    def _on_reshard_ready(self, msg: List[bytes]) -> None:
        # Two frames from a plain reshard; a chunked one appends the pages
        # its closing plan covers.
        if len(msg) not in (2, 3):
            raise ValueError(
                f"malformed RESHARD_READY message: expected 2 or 3 frames, "
                f"got {len(msg)}"
            )
        room = int(msg[1].decode("ascii"))
        pages = unpack_ints(msg[2]) if len(msg) > 2 else []
        with self._state_lock:
            if room in self._aborted:
                return
            self._reshard_ready.add(room)
            self._reshard_ready_pages[room] = pages
        self._maybe_reshard_push(room)

    def _maybe_reshard_push(self, room: int) -> None:
        with self._state_lock:
            if (
                room in self._aborted
                or room not in self._reshard_ready
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
                self._fail_room_locked(room, f"Raiden reshard push failed: {e}")
        finally:
            # The pending count is this push's lease on the room's pages;
            # it is handed back here in every mode and on every outcome.
            with self._state_lock:
                left = self._reshard_pushes_pending.get(room, 1) - 1
                if left <= 0:
                    self._reshard_pushes_pending.pop(room, None)
                else:
                    self._reshard_pushes_pending[room] = left
                if self.chunked_transfer:
                    # The room is sent once its closing push and every chunk
                    # push asked for before it have finished.
                    if chunk is None:
                        self._reshard_final_pushed.add(room)
                    if (
                        left <= 0
                        and room in self._reshard_final_pushed
                        and room in self._reshard_req
                    ):
                        self._completed_send.add(str(room))

    def _run_reshard_push(
        self, room: int, chunk: Optional[int] = None, uuid: Optional[int] = None
    ) -> None:
        from tpu_sync.rpc import raiden_service_pb2 as pb2

        from sglang.srt.disaggregation.raiden.reshard_schedule import (
            build_sender_request,
        )

        with self._state_lock:
            if room in self._aborted or room not in self._reshard_req:
                return  # the room failed or was cleared before the push ran
            req_meta = dict(self._reshard_req[room])
            if chunk is None:
                prefill_blocks = list(self._send_blocks[room])
                ready_pages = list(self._reshard_ready_pages.get(room, []))
                start = 0
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
                # The decode declared exactly which pages its closing plan
                # covers; pushing this side's frontier instead could skip a
                # chunk that lost its race mid-range on the decode.
                uuid = room
                if not ready_pages:
                    return
                base = total_pages - n
                prefill_blocks = [prefill_blocks[pg] for pg in ready_pages]
                decode_blocks = [decode_blocks[pg - base] for pg in ready_pages]
                n = len(ready_pages)
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
        # Registering the plan first lets raiden choose the host blocks this
        # transfer stages into: the blocks' own ids in the device-shadow
        # layout, or blocks allocated for the plan under demand staging. The
        # receiver matches on src_block_id (the prefill device blocks it
        # learned via RESHARD_BLOCKS), which is unaffected by where the bytes
        # are staged.
        engine.register_active_plan(uuid, req, is_sender=True)
        fut = None
        try:
            host_blocks = engine.plan_host_blocks(uuid, prefill_blocks)
            fut = engine.d2h(prefill_blocks, host_blocks, [1] * n)
            deadline = self._monotonic() + self._timeout_s
            while fut is not None and not fut.is_ready():
                if self._monotonic() > deadline:
                    raise TimeoutError(
                        f"raiden reshard D2H stage for room {room} timed out"
                    )
                time.sleep(0.0005)
            # is_ready() is true even for a ready-with-error future, so check
            # ok() before pushing -- otherwise a failed d2h would push stale
            # staging.
            if fut is not None and not fut.ok():
                raise RuntimeError(
                    f"raiden reshard D2H stage for room {room} failed: "
                    f"{fut.error_message()}"
                )
            # raiden runs SyncPush unlocked, so serialize pushes across workers
            # (concurrent pushes to a decode peer corrupt its shared transport).
            with self._reshard_push_lock:
                # The receiver's completion barrier is total_blocks * num_layers,
                # and each push counts its blocks once regardless of how many
                # layers it carries, so push one layer per request to deliver
                # every layer.
                for layer_idx in range(len(self._kv_caches)):
                    engine.push_registered_plan(
                        uuid, push_endpoint, prefill_blocks, decode_blocks, layer_idx
                    )
        finally:
            # Frees the plan's staging on the sender; the receiver's is freed
            # when its upload completes. A copy that timed out but is still
            # running keeps the plan until it settles -- unregistering would
            # free the staging while the copy writes it.
            if fut is not None and not fut.is_ready():
                with self._state_lock:
                    self._stale_plans.append((uuid, room, fut))
                threading.Thread(
                    target=self._reap_stale_plan, args=(uuid, room, fut), daemon=True
                ).start()
            else:
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

    def room_has_active_io(self, room: int) -> bool:
        """Whether raiden still holds a transfer of the room: an unsettled
        read, or a receive plan a push can still land in.

        While it does, the room's pages must not be handed back -- the
        transfer would write them under their next owner. Raiden's own
        transfer deadline bounds how long this can stay true.
        """
        rid, swa_rid, chunk_prefix = str(room), self._swa_req_id(room), f"{room}#"

        def belongs(key: str) -> bool:
            return key == rid or key == swa_rid or key.startswith(chunk_prefix)

        with self._state_lock:
            if self._pulling is not None:
                for key in self._pulling.rooms():
                    if belongs(key) and not self._pull_settled_locked(key):
                        return True
            for req_id in (getattr(self, "_reshard_plans", None) or {}).get(room, {}):
                if (
                    req_id not in self._completed_recv
                    and req_id not in self._native_failed
                ):
                    return True
        return False

    def clear_room(self, room: int) -> None:
        """Forget the room's transfer state.

        An abort tombstone survives the room: frames for it can still arrive
        and must stay rejected. Tombstones age out on their own.
        """
        self._drop_room_state(room, clear_failure_record=True)

    def _drop_room_state(
        self, room: int, *, clear_failure_record: bool = False
    ) -> None:
        rid = str(room)
        swa_rid = self._swa_req_id(room)
        chunk_prefix = f"{room}#"

        def belongs(key: str) -> bool:
            return key == rid or key == swa_rid or key.startswith(chunk_prefix)

        with self._state_lock:
            if hasattr(self, "_reshard_replied_at"):
                self._reshard_replied_at.pop(room, None)
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
                if req_id in self._completed_recv or req_id in self._native_failed:
                    settled_plans.append(uuid)
                    continue
                self._cleared_plans[req_id] = uuid
                held.add(req_id)
            self._completed_send.difference_update(
                [k for k in self._completed_send if belongs(k)]
            )
            for ids in (self._completed_recv, self._failed, self._native_failed):
                ids.difference_update([k for k in ids if belongs(k) and k not in held])
            for d in (
                getattr(self, "_chunk_queue", None),
                getattr(self, "_remote_total", None),
                getattr(self, "_announced_spans", None),
                getattr(self, "_chunk_reads", None),
                getattr(self, "_held_chunks", None),
                getattr(self, "_send_total", None),
                getattr(self, "_decode_pages", None),
                getattr(self, "_chunk_seq", None),
                getattr(self, "_reshard_chunks", None),
                getattr(self, "_pending_offer", None),
                getattr(self, "_reshard_covered", None),
                getattr(self, "_reshard_ready_pages", None),
                getattr(self, "_reshard_chunk_acked", None),
                getattr(self, "_reshard_chunk_registering", None),
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
            if clear_failure_record:
                # Failure publication takes _state_lock before failure_lock as
                # well, so clear and publication have one total order.
                with self.failure_lock:
                    self.failure_records.pop(room, None)
        for uuid in settled_plans:
            self._unregister_plan(uuid)


class RaidenKVSender(CommonKVSender):
    # A group-wide Failed is delivered only after every rank drained: raiden
    # cannot cancel a transfer, so a peer's pages must not be freed under it.
    group_failure_requires_drain = True

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
        self.init_time = time.time()
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
        if self.kv_mgr.is_dummy_cp_rank or self.kv_mgr.is_aborted(self.bootstrap_room):
            return
        self._record_transfer_indices(kv_indices, state_indices)
        swa_blocks = (
            self.kv_mgr._swa_blocks_from_state_indices(state_indices)
            if state_indices is not None
            else None
        )
        blocks = [int(x) for x in kv_indices]
        start = self.kv_mgr.append_send_blocks(self.bootstrap_room, blocks, swa_blocks)
        if start is None:
            return  # aborted while this chunk was being computed
        if self.kv_mgr.chunked_transfer and not self._last_chunk:
            self.kv_mgr.offer_chunk(
                self.bootstrap_room, start, blocks, self.num_kv_indices
            )
        if self._last_chunk and not self._started:
            self._started = True
            # finalize_send turns the room Transferring itself, under the
            # lock an abort takes to fail it; a status update here could
            # overwrite a Failed that landed in between.
            self.kv_mgr.finalize_send(self.bootstrap_room, self.aux_index)

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state
        if self.kv_mgr.is_dummy_cp_rank:
            self.conclude_state = KVPoll.Success
            return KVPoll.Success
        self.kv_mgr._drain_engine()
        # A recorded failure outranks a completion that lands after it, and
        # Failed is withheld while a stage or push still reads the room's
        # pages -- the scheduler frees them on it.
        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status == KVPoll.Failed or self.kv_mgr.is_failed(self.bootstrap_room):
            if self.kv_mgr.sender_has_active_io(self.bootstrap_room):
                return KVPoll.Transferring
            self.conclude_state = KVPoll.Failed
            return KVPoll.Failed
        if self.kv_mgr.is_send_done(self.bootstrap_room):
            self.conclude_state = KVPoll.Success
            return KVPoll.Success
        if status == KVPoll.Success:
            self.conclude_state = status
            return status
        # Raiden is consumer-pull: the prefill computes without waiting for
        # the decode, so the handshake window is bounded here rather than by
        # the Bootstrapping status. It closes when the decode asks to pull, or
        # clears a resharded room to push; a decode that dies before then
        # would otherwise leave the room (and its KV) waiting forever.
        if self.kv_mgr.fail_decode_silence(
            self.bootstrap_room, self.init_time, self.kv_mgr.bootstrap_timeout
        ):
            self.conclude_state = KVPoll.Failed
            return KVPoll.Failed
        return status

    def drained(self) -> bool:
        """No transfer of this room still runs on this rank."""
        return not self.kv_mgr.sender_has_active_io(self.bootstrap_room)

    def quiesce_group_failure(self) -> None:
        """Seal this rank before the group snapshots its native drain bit."""
        with self.kv_mgr._state_lock:
            self.kv_mgr._mark_aborted_locked(self.bootstrap_room)
            self.kv_mgr._fail_room_locked(
                self.bootstrap_room,
                "Raiden transfer failed on another attention TP/CP rank.",
                preserve_reason=True,
            )
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
        # A local Success may have been cached before a peer contributed the
        # group failure. Re-enter the failure poll path after this collective.
        self.conclude_state = None
        # State that could admit new work is discarded, while the leases added
        # for already accepted reads/stages/pushes intentionally survive.
        self.kv_mgr._drop_room_state(self.bootstrap_room)

    def abort(self) -> None:
        """Seal a local abort, but do not conclude it over retained source KV.

        The scheduler may recycle the request's blocks as soon as this sender
        is terminal. Keep ``conclude_state`` unset until every native read /
        stage / push lease has settled; the normal poll path then publishes
        Failed.
        """
        with self.kv_mgr._state_lock:
            self.kv_mgr._mark_aborted_locked(self.bootstrap_room)
            self.kv_mgr._fail_room_locked(self.bootstrap_room, "Aborted by AbortReq.")
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
        self.kv_mgr._drop_room_state(self.bootstrap_room)
        self.conclude_state = (
            None
            if self.kv_mgr.sender_has_active_io(self.bootstrap_room)
            else KVPoll.Failed
        )

    def failure_exception(self):
        """Drop the room's state, then raise the recorded failure reason."""
        with self.kv_mgr.failure_lock:
            reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "Raiden send failed"
            )
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed
        self.clear()
        raise Exception(reason)

    def clear(self) -> None:
        super().clear()
        self.kv_mgr.clear_room(self.bootstrap_room)


class RaidenKVReceiver(CommonKVReceiver):
    # See RaidenKVSender: a group Failed waits for every rank's transfers.
    group_failure_requires_drain = True

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
        # The waiting timeout counts from here: the request is now known to
        # prefill and nothing else bounds how long its reply can take.
        self.init_time = time.time()

    def poll(self) -> KVPoll:
        """Every terminal condition is checked before any native read is
        issued, so a room that just failed (peer heartbeat, native failure,
        or its own deadline) can never start a transfer in the same poll and
        then hand its pages back while that transfer runs."""
        if self.conclude_state is not None:
            return self.conclude_state
        self.kv_mgr._drain_engine()
        if self.kv_mgr.is_failed(self.bootstrap_room):
            return self._conclude_failed()
        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status == KVPoll.Failed:
            # The heartbeat declared the prefill down; nobody is left to
            # notify.
            return self._conclude_failed(notify=False)
        if self.kv_mgr.is_recv_done(self.bootstrap_room):
            self.conclude_state = KVPoll.Success
            return KVPoll.Success
        if self._check_waiting_timeout() is not None:
            return self._conclude_failed()
        if self.kv_mgr.maybe_start_pull(self.bootstrap_room) and not self._started:
            # The manager turned the room Transferring inside the admission.
            self._started = True
            return KVPoll.Transferring
        return self.kv_mgr.check_status(self.bootstrap_room)

    def abort(self):
        """Explicit local abort: same terminal path as any failure, so active
        transfers keep the room's pages until they settle."""
        self._conclude_failed(reason="Aborted by AbortReq.")

    def drained(self) -> bool:
        """No transfer of this room still runs on this rank."""
        return not self.kv_mgr.room_has_active_io(self.bootstrap_room)

    def quiesce_group_failure(self) -> None:
        """Tombstone this rank before the group snapshots its drain bit."""
        # A local Success can race a peer failure. Clear it so the sticky
        # Failed state below owns every subsequent local poll, and notify the
        # corresponding prefill rank to stop publishing work for this room.
        self.conclude_state = None
        self._conclude_failed(
            reason="Raiden transfer failed on another attention TP/CP rank.",
            preserve_reason=True,
        )

    def _conclude_failed(
        self,
        notify: bool = True,
        reason: Optional[str] = None,
        preserve_reason: bool = False,
    ) -> KVPoll:
        """Terminalize the room: tombstone it so late frames cannot rebuild
        it, and tell the prefill to release it.

        While raiden still holds one of the room's transfers, Failed is
        withheld -- the scheduler frees the room's pages on it, and the
        transfer would write them under their next owner. The tombstone
        already stops anything new; the settled room fails on a later poll.
        """
        with self.kv_mgr._state_lock:
            # Tombstone and terminal status are one transaction with native
            # admission. A read admitted first becomes a drain obligation;
            # otherwise admission observes the tombstone and does nothing.
            self.kv_mgr._mark_aborted_locked(self.bootstrap_room)
            if reason is not None:
                self.kv_mgr._fail_room_locked(
                    self.bootstrap_room, reason, preserve_reason=preserve_reason
                )
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
        if notify:
            self._notify_prefill_abort()
        if self.kv_mgr.room_has_active_io(self.bootstrap_room):
            return KVPoll.Transferring
        self.conclude_state = KVPoll.Failed
        return KVPoll.Failed

    def _notify_prefill_abort(self) -> None:
        """Tell every prefill rank serving this room to release it (once)."""
        if self.abort_notified or getattr(self, "bootstrap_infos", None) is None:
            return
        self._send_abort_notification()
        self.abort_notified = True

    def failure_exception(self):
        """Consume the recorded failure reason and raise it."""
        with self.kv_mgr.failure_lock:
            reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "Raiden receive failed"
            )
        raise Exception(reason)

    def clear(self) -> None:
        super().clear()
        self.kv_mgr.clear_room(self.bootstrap_room)


class RaidenKVBootstrapServer(CommonKVBootstrapServer):
    pass
