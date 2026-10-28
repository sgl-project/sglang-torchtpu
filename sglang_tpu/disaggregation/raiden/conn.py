"""TPU prefill/decode disaggregation backend backed by tpu-sync.

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
import math
import os
import threading
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll, StateType
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_disagg, get_schedule
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.network import NetworkAddress
from sglang_tpu.disaggregation.raiden.common import (
    CHUNK_ANNOUNCE_TOKENS,
    DEFAULT_KV_PORT,
    PORTS_PER_RANK,
    STATE_PORT_OFFSET,
    SWA_PORT_OFFSET,
    RaidenKVCacheGroups,
    _tcp_endpoint,
    get_raiden_kv_caches,
    logger,
)
from sglang_tpu.disaggregation.raiden.host_pool import (
    StagingAdmission,
    node_worker_rank,
    ranks_per_node,
    slot_host_bytes,
    slots_within_budget,
)
from sglang_tpu.disaggregation.raiden.receiver import (
    RaidenKVReceiver,
    RaidenReceiverMixin,
)
from sglang_tpu.disaggregation.raiden.sender import RaidenKVSender, RaidenSenderMixin
from sglang_tpu.disaggregation.raiden.serde import (
    Blocks,
    Chunk,
    MessageTag,
    PullReq,
    ReshardBlocks,
    ReshardChunk,
    ReshardChunkReady,
    ReshardReady,
    ReshardReq,
    RoomFailed,
)
from sglang_tpu.disaggregation.raiden.state import DecodeState, PrefillState
from sglang_tpu.envs import envs as tpu_envs

if TYPE_CHECKING:
    from tpu_sync.api.torch.kv_cache_manager import SettledTransfer

# Re-exported so that "from …raiden.conn import X" keeps resolving for the
# callers and tests that predate the split into common/sender/receiver.
__all__ = [
    "CHUNK_ANNOUNCE_TOKENS",
    "DEFAULT_KV_PORT",
    "RaidenKVBootstrapServer",
    "RaidenKVCacheGroups",
    "RaidenKVManager",
    "RaidenKVReceiver",
    "RaidenKVSender",
    "_tcp_endpoint",
    "get_raiden_kv_caches",
    "logger",
]


def _describe_failure(
    *, side: Literal["send", "receive"], transfer: SettledTransfer
) -> str:
    """Why raiden failed ``transfer``, in raiden's field names, leaving out the
    fields it did not report."""
    parts = [f"Raiden {side} failed: req_id={transfer.req_id}"]
    if transfer.code is not None:
        parts.append(f"code={transfer.code}")
    if transfer.phase is not None:
        parts.append(f"phase={transfer.phase}")
    if transfer.peer is not None:
        parts.append(f"peer={transfer.peer}")
    if transfer.elapsed_ns is not None:
        parts.append(f"elapsed_ms={transfer.elapsed_ns / 1e6:.1f}")
    if transfer.reason is not None:
        parts.append(f"reason={transfer.reason}")
    return " ".join(parts)


class RaidenKVManager(
    RaidenSenderMixin,
    RaidenReceiverMixin,
    CommonKVManager,
):
    """KV manager for the raiden prefill/decode transport.

    FIXME: the ZMQ rendezvous scaffolding here (per-rank listener socket,
    handshake message routing, poll draining) overlaps the tpu backend's
    ``base.py`` control plane. Consolidate onto a shared base so raiden reuses it
    instead of reimplementing the coordination.
    """

    # The legacy transport unless set_kv_caches builds a pipeline transfer.
    _pipeline = None
    _pipeline_enabled = False
    _listener_start_read = False
    # State engine: one row per request for GDN, or multi-page request rings for DSV4.
    _state_engine: Optional[Any] = None
    _state_kv_caches: Tuple[Any, ...] = ()
    _state_type: StateType = StateType.MAMBA
    _state_max_blocks: int = 1
    _state_engine_max_blocks = 0
    _committed_state_row: Optional[Callable[[int], int]] = None
    _read_state_meta: Optional[Callable[[Sequence[int]], bytes]] = None
    _write_state_meta: Optional[Callable[[Sequence[int]], bytes]] = None
    state_raiden_endpoint: Optional[str] = None

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)

        self._message_handlers = {
            PullReq.TAG: self._on_pull_req,
            Blocks.TAG: self._on_blocks,
            ReshardReq.TAG: self._on_reshard_req,
            ReshardBlocks.TAG: self._on_reshard_blocks,
            ReshardReady.TAG: self._on_reshard_ready,
            Chunk.TAG: self._on_chunk,
            ReshardChunk.TAG: self._on_reshard_chunk,
            ReshardChunkReady.TAG: self._on_reshard_chunk_ready,
            RoomFailed.TAG: self._on_room_failed,
        }

        self._engine: Optional[Any] = None  # raiden KVCacheManager
        self._swa_engine: Optional[Any] = None
        self._kv_caches: Optional[list] = None
        self._swa_kv_caches: Optional[list] = None
        self.raiden_endpoint: Optional[str] = None
        self.swa_raiden_endpoint: Optional[str] = None
        self._pipeline = None
        # Pipeline stages always transfer by named layer pools; a DP or TP
        # decode opts in to consume from a pipeline prefill.
        self._pipeline_enabled = self.pp_size > 1 or (
            disaggregation_mode == DisaggregationMode.DECODE
            and envs.SGLANG_RAIDEN_PIPELINE_TRANSFER.get()
        )
        self._pipeline_init_lock = threading.Lock()
        self._pipeline_messages = []
        # Each rank reserves control+data ports for full KV, SWA KV and GDN
        # state. The
        # rank counts every worker on the node -- attention-TP ranks of every
        # context-parallel, attention-DP, and system-DP replica -- so
        # replicas never share a port, an engine id, or a uuid namespace.
        base_port = int(os.environ.get("SGLANG_RAIDEN_KV_PORT", str(DEFAULT_KV_PORT)))
        self._local_rank = node_worker_rank(
            self.attn_tp_rank,
            self.attn_tp_size,
            self.attn_dp_rank,
            self.attn_dp_size,
            self.system_dp_rank,
            self.attn_cp_rank,
            self.attn_cp_size,
            self.pp_rank,
            self.pp_size,
        )
        self._control_port = base_port + PORTS_PER_RANK * self._local_rank
        self._timeout_s = envs.SGLANG_RAIDEN_CHUNK_TRANSFER_TIMEOUT_S.get()
        self._transfer_parallelism = (
            tpu_envs.SGLANG_TPU_RAIDEN_TRANSFER_PARALLELISM.get()
        )
        if self._transfer_parallelism < 1:
            raise ValueError("SGLANG_TPU_RAIDEN_TRANSFER_PARALLELISM must be >= 1")
        # Raiden reads its own TPU_RAIDEN_DYNAMIC_HOST_STAGING when the engine is
        # built in this process. Store the resolved value back so later reads
        # do not mistake the variable set here for the deprecated alias.
        dynamic_host_staging = envs.SGLANG_RAIDEN_DYNAMIC_HOST_STAGING.get()
        envs.SGLANG_RAIDEN_DYNAMIC_HOST_STAGING.set(dynamic_host_staging)
        os.environ["TPU_RAIDEN_DYNAMIC_HOST_STAGING"] = (
            "1" if dynamic_host_staging else "0"
        )
        self._dynamic_host_staging = dynamic_host_staging
        # Whether KV moves chunk by chunk as the prefill commits it, or in one
        # transfer after the whole prefill; decided from the KV layout when the
        # pool tensors are handed over.
        self.chunked_transfer = False

        # Completion sets, drained from raiden and keyed by str(room). Persist
        # until clear() so repeated poll() calls before the framework advances
        # keep reporting the same terminal state.
        self._completed_send: Set[str] = set()
        # Raiden's finish time (perf_counter seconds) per sent id, if reported.
        self._send_done_at: Dict[str, float] = {}
        self._completed_recv: Set[str] = set()
        self._failed: Set[str] = set()
        # Transfer ids the engine reported failed. Only these mean raiden has
        # released the transfer's staging; a room failed for a logical reason
        # (abort, timeout, peer death) shares its id with its closing read, so
        # room failure must never read as native settlement.
        self._native_failed: Set[str] = set()
        # Reads registered for the decode to pull, by request id and
        # publish time. This is populated only by prefill managers, but the
        # shared engine-drain path settles it for both manager roles. A lease
        # is the room's claim on its pages: it starts before the (possibly
        # blocking) registration and settles when the engine reports the read
        # served or failed (raiden expires a read nobody pulls at its
        # deadline). Exception: once the room's closing read has settled, its
        # unread chunk leases are dropped early -- see
        # _settle_sender_reads_locked.
        self._send_reads: Dict[int, Set[str]] = {}
        # Chunk read ids dropped early by that exception. Raiden cannot cancel
        # a read, so each still expires natively at its deadline; when
        # poll_stats reports that expiry, the id is removed from here instead
        # of failing the room.
        self._superseded_sender_reads: Set[str] = set()
        # Reads issued and not yet settled, with the host staging pages each
        # holds. Raiden seats a read in pages of the pool it allocated at
        # startup and rejects a read it cannot seat, so the decode side admits
        # a read only while the pool has room for it and leaves the rest in
        # the transfer queue.
        self._pulling: Optional[StagingAdmission] = None
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
        # Engine name ("full" / "swa") -> monotonic time its poll_stats began
        # raising on every poll; an engine is absent while its polls succeed.
        self._engine_poll_failing_since: Dict[str, float] = {}
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

        self._prefill: Optional[PrefillState] = None
        self._decode: Optional[DecodeState] = None
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._prefill = PrefillState(
                announce_pages=max(
                    1, -(-CHUNK_ANNOUNCE_TOKENS // int(self.kv_args.page_size))
                )
            )
        else:
            # Detects a prefill that stopped answering and fails its rooms.
            self._start_heartbeat_checker_thread()
            self._decode = DecodeState()
            self._listener_start_read = envs.SGLANG_RAIDEN_LISTENER_START_READ.get()

        # Native I/O tracking stays on the manager: it survives room cleanup.
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._reshard_pushes_pending: Dict[int, int] = {}
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
            self._reshard_plans: Dict[int, Dict[str, int]] = {}
        self._listener = threading.Thread(target=self._listen, daemon=True)
        self._listener.start()

    # ----- resharding geometry / host-slot helpers --------------------------

    def _make_geometry(self, src_attn_tp: int, dst_attn_tp: int, *, swa: bool = False):
        """ReshardGeometry for a (src_attn_tp -> dst_attn_tp) transfer, sized from
        the local fused KV tensor (the K+V head-pair granule matches on both
        sides) and the global kv-head count.

        ``swa`` sizes it from the sliding-window sub-pool instead. Its layers
        carry the same kv-head count as the full pool -- windowing bounds how
        many tokens are kept, not how the heads are sharded -- so only the
        head-pair granule is read from the other tensor, and the head-slice math
        is identical. Taking it from the SWA tensor rather than assuming it
        matches keeps the two pools free to differ in dtype or head_dim.
        """
        from sglang_tpu.disaggregation.raiden.reshard_schedule import (
            ReshardGeometry,
            head_pair_bytes,
        )

        tensors = self._swa_kv_caches if swa else self._kv_caches
        if not tensors:
            raise RuntimeError(
                "Raiden has no "
                + ("sliding-window" if swa else "full-attention")
                + " KV tensors to build a reshard geometry from"
            )
        tensor = tensors[0]
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
        self._state_kv_caches = tuple(kv_caches.state)
        self._state_type = kv_caches.state_type
        self._state_max_blocks = kv_caches.state_max_blocks
        self._committed_state_row = kv_caches.committed_state_row
        self._read_state_meta = kv_caches.read_state_meta
        self._write_state_meta = kv_caches.write_state_meta
        if (
            self._state_kv_caches
            and self._state_type == StateType.MAMBA
            and self._committed_state_row is None
        ):
            raise RuntimeError("Raiden GDN state tensors need a committed-row mapping")
        if self._pipeline_enabled:
            if self._state_kv_caches:
                raise NotImplementedError(
                    "Raiden pipeline transfer does not move GDN (linear-attention) "
                    "state yet; serve hybrid linear-attention models without "
                    "pipeline parallelism or SGLANG_RAIDEN_PIPELINE_TRANSFER."
                )
            from sglang_tpu.disaggregation.raiden.pipeline import PipelineTransfer

            pipeline = PipelineTransfer(self, kv_caches)
            self._engine = pipeline.engine
            self.raiden_endpoint = pipeline.controller
            self.chunked_transfer = False
            with self._pipeline_init_lock:
                self._pipeline = pipeline
                for message in self._pipeline_messages:
                    pipeline.on_message(message)
                self._pipeline_messages.clear()
            return
        # Full-attention KV moves piece by piece during the prefill. A hybrid
        # model's sliding-window pool holds only the last window, which is
        # settled when the prefill ends, so it rides on the closing read, as
        # does GDN state.
        self.chunked_transfer = True
        if not self._kv_caches:
            raise RuntimeError(
                "Raiden requires at least one full-attention KV cache tensor"
            )
        self._build_engine()

    def _build_engine(self) -> None:
        if self._engine is not None or self._kv_caches is None:
            return
        engines = 1 + bool(self._swa_kv_caches) + bool(self._state_kv_caches)
        max_blocks = self._max_request_blocks()
        num_slots = self._num_slots(self._kv_caches, max_blocks, engines)
        self.raiden_endpoint, self._engine = self._start_engine(
            kv_caches=self._kv_caches,
            port=self._control_port,
            max_blocks=max_blocks,
            num_slots=num_slots,
        )
        self._full_engine_max_blocks = max_blocks
        pool_pages = num_slots * max_blocks
        swa_pool_pages = None
        state_pool_pages = None
        logger.info(
            "RaidenKVManager rank%d engine up | endpoint=%s max_blocks=%d "
            "num_slots=%d pool_pages=%d chunked=%d parallelism=%d",
            self.attn_tp_rank,
            self.raiden_endpoint,
            max_blocks,
            num_slots,
            pool_pages,
            int(self.chunked_transfer),
            self._transfer_parallelism,
        )
        if self._swa_kv_caches:
            swa_max_blocks = self._swa_max_request_blocks()
            swa_num_slots = self._num_slots(
                self._swa_kv_caches, swa_max_blocks, engines
            )
            self.swa_raiden_endpoint, self._swa_engine = self._start_engine(
                kv_caches=self._swa_kv_caches,
                port=self._control_port + SWA_PORT_OFFSET,
                max_blocks=swa_max_blocks,
                num_slots=swa_num_slots,
            )
            self._swa_engine_max_blocks = swa_max_blocks
            swa_pool_pages = swa_num_slots * swa_max_blocks
            logger.info(
                "RaidenKVManager rank%d SWA engine up | endpoint=%s "
                "max_blocks=%d num_slots=%d pool_pages=%d",
                self.attn_tp_rank,
                self.swa_raiden_endpoint,
                swa_max_blocks,
                swa_num_slots,
                swa_pool_pages,
            )
        if self._state_kv_caches:
            state_max_blocks = self._state_max_blocks
            state_num_slots = self._num_slots(
                list(self._state_kv_caches), state_max_blocks, engines
            )
            self.state_raiden_endpoint, self._state_engine = self._start_engine(
                kv_caches=list(self._state_kv_caches),
                port=self._control_port + STATE_PORT_OFFSET,
                max_blocks=state_max_blocks,
                num_slots=state_num_slots,
            )
            self._state_engine_max_blocks = state_max_blocks
            state_pool_pages = state_num_slots * state_max_blocks
            logger.info(
                "RaidenKVManager rank%d state engine up | endpoint=%s "
                "layers=%d max_blocks=%d num_slots=%d pool_pages=%d",
                self.attn_tp_rank,
                self.state_raiden_endpoint,
                len(self._state_kv_caches),
                state_max_blocks,
                state_num_slots,
                state_pool_pages,
            )
        # Each engine's pool is handed out read by read, so the pages the
        # decode may hold in flight are the pages raiden reserved.
        self._pulling = StagingAdmission(pool_pages, swa_pool_pages, state_pool_pages)

    def _start_engine(
        self, *, kv_caches: List[Any], port: int, max_blocks: int, num_slots: int
    ) -> Tuple[str, Any]:
        """Endpoint and raiden engine serving kv_caches on control port port."""
        # TPU-only import, deferred so the disaggregation module imports cleanly
        # on hosts without tpu_sync. Raiden owns the transport, host staging,
        # D2H/H2D DMA, and completion tracking; it indexes each KV tensor's major
        # dim as a page, matching the TPU fused KV pool ([num_blocks, page, ...]).
        from tpu_sync.api.torch.kv_cache_manager import KVCacheManager

        endpoint = NetworkAddress(host=self.local_ip, port=port).to_host_port_str()
        engine = KVCacheManager(
            kv_caches=kv_caches,
            node_id=self._local_rank,
            local_control_port=port,
            max_blocks=max_blocks,
            num_slots=num_slots,
            timeout_s=self._timeout_s,
            parallelism=self._transfer_parallelism,
        )
        return endpoint, engine

    def _staging_cost(self, pages: int, max_blocks: int) -> int:
        """Host pages a read of ``pages`` pages takes from its engine's pool.

        Per-transfer staging seats exactly the read's pages; fixed slots seat
        every read in a whole slot of ``max_blocks`` pages.
        """
        return pages if self._dynamic_host_staging else max_blocks

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
        """Slots the pool is sized for, at most.

        A read holds its pages from the moment it is issued until its H2D
        lands, and a request keeps one read in flight. The decode admits a
        request into the transfer queue before it joins the running batch, so
        the reach is the batch plus the extra in-transfer slots reserved
        alongside it; a larger pool could never be claimed.
        """
        batch = int(get_schedule().max_running_requests or 64)
        in_transfer = int(get_disagg().disaggregation_decode_extra_slots or 0)
        return max(1, batch + in_transfer)

    def _num_slots(
        self,
        kv_caches: List[Any],
        max_blocks: int,
        engines: int = 1,
        *,
        reserved_bytes: int = 0,
        paired_slot_bytes: int = 0,
    ) -> int:
        """Slots this engine can afford within the host staging budget.

        The per-node budget is split across the ranks resident on this host
        and across the engines this rank builds, less ``reserved_bytes``
        another engine of this rank already took, then divided by what one
        slot costs, plus ``paired_slot_bytes`` when each slot comes with one
        of another engine. Deriving the count keeps a large context length
        from turning into a host allocation that fails at startup -- it
        shrinks the pool instead, and refuses to start when not even one slot
        fits the budget.
        """
        budget = float(os.environ.get("SGLANG_RAIDEN_HOST_POOL_GB", "128")) * 1024**3
        local_ranks = ranks_per_node(
            self.attn_tp_size,
            self.attn_dp_size,
            self.system_dp_size,
            self.server_args.nnodes,
            self.attn_cp_size,
            self.pp_size,
        )
        budget = budget / (local_ranks * max(1, engines)) - reserved_bytes
        slot_bytes = slot_host_bytes(kv_caches, max_blocks) + paired_slot_bytes
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
    def _state_req_id(room: int) -> str:
        return f"state:{room}"

    @classmethod
    def _room_of(cls, rid: str) -> Optional[int]:
        """The room of a whole ("<room>"), sliding-window ("swa:<room>"), GDN
        state ("state:<room>") or chunk ("<room>#<k>") transfer id, resharded
        or not; None for any other id."""
        room = cls._norm_req_id(rid).removeprefix("swa:").removeprefix("state:")
        room = room.partition("#")[0]
        return int(room) if room.isdecimal() else None

    def _is_room_transfer(self, key: str, room: int) -> bool:
        """Whether key is a whole, sliding-window, state or chunk id of room."""
        return key in (
            str(room),
            self._swa_req_id(room),
            self._state_req_id(room),
        ) or key.startswith(f"{room}#")

    def _side_reads_done(self, room: int, done: Set[str]) -> bool:
        """Whether the room's sliding-window and state reads, where their
        engines exist, are in done."""
        return (self._swa_engine is None or self._swa_req_id(room) in done) and (
            self._state_engine is None or self._state_req_id(room) in done
        )

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
        if self._pipeline is not None:
            sent, received, failed = self._pipeline.poll()
            with self._state_lock:
                self._completed_send.update(sent)
                self._completed_recv.update(received)
                self._native_failed.update(failed)
                self._failed.update(failed)
            self._log_settled_transfers(sent=sent, received=received)
            return
        engines = (
            ("full", self._engine),
            ("swa", self._swa_engine),
            ("state", self._state_engine),
        )
        if all(engine is None for _, engine in engines):
            return
        for name, engine in engines:
            if engine is None:
                continue
            status = self._poll_engine_status(name=name, engine=engine)
            if status is None:
                continue
            done_send, done_recv, failed = status
            if done_send or done_recv or failed:
                with self._state_lock:
                    for t in done_send:
                        rid = self._norm_req_id(t.req_id)
                        self._completed_send.add(rid)
                        if t.completed_ns is not None:
                            self._send_done_at[rid] = t.completed_ns / 1e9
                    self._completed_recv.update(
                        self._norm_req_id(t.req_id) for t in done_recv
                    )
                    norm_failed = {self._norm_req_id(t.req_id) for t in failed}
                    superseded = self._superseded_sender_reads
                    real_failed = norm_failed - superseded
                    superseded -= norm_failed
                    self._native_failed.update(real_failed)
                    self._failed.update(real_failed)
                    failures, later_failures = self._record_native_failures_locked(
                        t for t in failed if self._norm_req_id(t.req_id) in real_failed
                    )
                    self._release_pull_slots_locked()
                    self._settle_sender_reads_locked()
                    settled_plans = self._settled_cleared_plans_locked()
                for uuid in settled_plans:
                    self._unregister_plan(uuid)
                self._log_settled_transfers(
                    sent=[self._norm_req_id(t.req_id) for t in done_send],
                    received=[self._norm_req_id(t.req_id) for t in done_recv],
                )
                for failure in failures:
                    logger.error(
                        "RaidenKVManager rank%d %s", self.attn_tp_rank, failure
                    )
                for failure in later_failures:
                    logger.debug(
                        "RaidenKVManager rank%d %s", self.attn_tp_rank, failure
                    )
        with self._state_lock:
            settled_stale = [e for e in self._stale_plans if e[2].is_ready()]
            for entry in settled_stale:
                self._stale_plans.remove(entry)
        for uuid, _room, _fut in settled_stale:
            self._unregister_plan(uuid)

    def _record_native_failures_locked(
        self, failed: Iterable[SettledTransfer]
    ) -> Tuple[List[str], List[str]]:
        """Describe each transfer raiden failed, and record each room's
        earliest description as its failure reason, unless the room was
        cleared: a later request reusing the room id would adopt the record.
        Returns the rooms' earliest descriptions, then their later ones."""
        side = "send" if self._prefill is not None else "receive"
        earliest: Dict[int | str, str] = {}
        later = []
        # A room's later failures may only be symptoms of its first one.
        ordered = sorted(
            failed, key=lambda t: (t.completed_ns is None, t.completed_ns or 0)
        )
        for transfer in ordered:
            failure = _describe_failure(side=side, transfer=transfer)
            room = self._room_of(transfer.req_id)
            key = transfer.req_id if room is None else room
            if key in earliest:
                later.append(failure)
                continue
            earliest[key] = failure
            with self._status_lock:
                live = room is not None and room in self.request_status
            if live:
                with self.failure_lock:
                    self.failure_records.setdefault(room, failure)
        return list(earliest.values()), later

    def _log_settled_transfers(
        self, *, sent: Iterable[str], received: Iterable[str]
    ) -> None:
        # Per-transfer, per-rank lines are DEBUG: at INFO they dominate server
        # logs (several lines per request on every rank). Failures stay ERROR.
        for rid in sent:
            logger.debug(
                "RaidenKVManager rank%d prefill read served | req_id=%s",
                self.attn_tp_rank,
                rid,
            )
        for rid in received:
            logger.debug(
                "RaidenKVManager rank%d decode pull done | req_id=%s",
                self.attn_tp_rank,
                rid,
            )

    def _poll_engine_status(
        self, *, name: str, engine: Any
    ) -> Tuple[List[Any], List[Any], List[Any]] | None:
        """One engine's sent, received and failed transfers, or None if the
        poll raised.

        Runs inside the scheduler's poll, so an escaping exception would take
        down every request on this worker. Raiden raises from
        ``poll_stats_with_details`` only before it takes any completion, so a
        failed poll loses nothing and the next one reports it. Raiden's
        transfer deadlines are reported through it too: once polls have failed
        for longer than one deadline no transfer can settle, and that is raised
        rather than left to hang every request with its pages held.
        """
        try:
            stats = engine.poll_stats_with_details()
        except Exception as e:
            now = self._monotonic()
            since = self._engine_poll_failing_since.setdefault(name, now)
            if now - since > self._timeout_s:
                raise RuntimeError(
                    f"Raiden {name} engine poll_stats has raised on every poll "
                    f"for {now - since:.0f}s, past the {self._timeout_s:.0f}s "
                    "transfer deadline; no transfer can settle"
                ) from e
            if since == now:
                logger.error(
                    "RaidenKVManager rank%d %s engine poll_stats raised, "
                    "retrying on later polls: %s",
                    self.attn_tp_rank,
                    name,
                    e,
                )
            return None
        if self._engine_poll_failing_since.pop(name, None) is not None:
            logger.info(
                "RaidenKVManager rank%d %s engine poll_stats recovered",
                self.attn_tp_rank,
                name,
            )
        return stats.sent, stats.received, stats.failed

    def send_done_time(self, room: int) -> Optional[float]:
        """Raiden's finish time for the room's send, or None if not reported."""
        rids = [str(room)]
        if self._swa_engine is not None:
            rids.append(self._swa_req_id(room))
        if self._state_engine is not None:
            rids.append(self._state_req_id(room))
        times = [self._send_done_at.get(rid) for rid in rids]
        return None if None in times else max(times)

    def _unregister_plan(self, uuid: int) -> None:
        """Drops a receiver plan whose push has settled.

        Raiden drops a timed-out plan itself, so one it no longer knows has
        already been dropped -- which is also why this can sweep both engines
        unconditionally: a hybrid room registers the same uuid on each, and on
        an engine that never held it the drop is the same no-op as for a plan
        raiden already reaped.
        """
        for engine in (self._engine, self._swa_engine):
            if engine is None:
                continue
            try:
                engine.unregister_active_plan(uuid)
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

    def _fail_room_unless_failed_locked(self, room: int, reason: str) -> None:
        """Fail ``room`` for a local abort or a peer rank's failure, keeping
        the reason of any failure that came first. A failed pipeline transfer
        has no recorded reason, and the prefill drops the room's ``_failed``
        ids right after this call, so such a room gets the reason
        ``failure_exception`` would fall back to."""
        if self._is_failed_locked(room):
            reason = (
                "Raiden send failed"
                if self._prefill is not None
                else "Raiden receive failed"
            )
        self._fail_room_locked(room, reason, preserve_reason=True)

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

        A bracketed endpoint (see ``_tcp_endpoint``) is an IPv6 peer, and the
        socket has to opt into IPv6: libzmq accepts a connect to an IPv6
        literal on an IPv4-only socket, then fails the resolve in its I/O
        thread and, with reconnects disabled, never retries -- the PUSH
        socket queues every message and delivers none, so the rendezvous
        stalls silently until the disaggregation timeouts fire.
        """
        is_ipv6 = is_ipv6 or endpoint.startswith("tcp://[")
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
            if sock is not None:
                # libzmq's I/O thread reports a peer drop with a blocking send
                # to the monitor. Once the monitor is closed that send never
                # returns, and no socket of this context delivers again, so
                # the socket stops reporting before either end is closed.
                sock.disable_monitor()
                sock.close(linger=0)
            if monitor is not None:
                monitor.close()

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
        if not msg:
            logger.error("RaidenKVManager received an empty message")
            return
        tag_frame = msg[0]
        if not isinstance(tag_frame, (bytes, bytearray, memoryview)):
            logger.error(
                "RaidenKVManager received a non-bytes message tag: %r", tag_frame
            )
            return
        tag = bytes(tag_frame)
        if not tag:
            logger.error("RaidenKVManager received an empty message tag")
            return
        # CommonKVReceiver still emits ABORT as a positional message.
        if tag == MessageTag.ABORT:
            self._on_abort(msg)
            return
        if tag in (
            MessageTag.PIPELINE_REQ,
            MessageTag.PIPELINE_READY,
            MessageTag.PIPELINE_RELEASE,
        ):
            self._on_pipeline_message(msg)
            return
        handler = self._message_handlers.get(tag)
        if handler is None:
            logger.error("RaidenKVManager unknown message tag: %s", tag)
            return
        handler(msg)

    def _on_pipeline_message(self, msg: List[bytes]) -> None:
        """Hand a frame to the pipeline transfer, or hold it until the pools
        register; a frame the pipeline rejects fails its room."""
        room = int(msg[1])
        try:
            if not self._pipeline_enabled:
                raise ValueError(
                    "Raiden pipeline transfer requires a pipeline-parallel prefill"
                )
            with self._pipeline_init_lock:
                if self._pipeline is None:
                    self._pipeline_messages.append(msg)
                    return
                pipeline = self._pipeline
            pipeline.on_message(msg)
        except Exception as error:
            with self._state_lock:
                self._mark_aborted_locked(room)
                self._fail_room_locked(room, f"Raiden pipeline transfer: {error}")
                self.update_status(room, KVPoll.Failed)
            raise

    @staticmethod
    def _require_frames(msg: List[bytes], tag: MessageTag, n: int) -> None:
        """Validate the frame count of a positional pipeline message."""
        if len(msg) != n:
            raise ValueError(
                f"malformed {tag.name} message: expected {n} frames, got {len(msg)}"
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

    def _state_rows_from_state_indices(
        self, state_indices: Optional[List]
    ) -> List[int]:
        """The committed state row/pages of the request's state slot."""
        if self._state_engine is None:
            return []
        try:
            index = self.kv_args.state_types.index(self._state_type)
        except ValueError as exc:
            raise RuntimeError(
                f"Raiden has state tensors but {self._state_type} is not registered"
            ) from exc
        if (
            state_indices is None
            or index >= len(state_indices)
            or state_indices[index] is None
        ):
            raise RuntimeError("Raiden state transfer is missing its slot")
        slots = [int(x) for x in np.asarray(state_indices[index]).reshape(-1)]
        if self._state_type == StateType.MAMBA:
            if len(slots) != 1:
                raise RuntimeError(
                    f"Raiden GDN state transfer expects one mamba slot, got {slots}"
                )
            return [self._committed_state_row(slots[0])]
        expected = self._state_engine_max_blocks
        if len(slots) != expected:
            raise RuntimeError(
                f"Raiden state transfer expects {expected} pages, got {len(slots)}"
            )
        return slots

    @staticmethod
    def _monotonic() -> float:
        return time.monotonic()

    # ----- completion queries (drained by poll()) ---------------------------

    def is_send_done(self, room: int) -> bool:
        return str(room) in self._completed_send and self._side_reads_done(
            room, self._completed_send
        )

    def is_recv_done(self, room: int) -> bool:
        if self.chunked_transfer:
            with self._state_lock:
                if room not in self._decode.final_seen or self._decode.chunk_queue.get(
                    room
                ):
                    return False
                reads = list(self._decode.chunk_reads.get(room, []))
            return all(
                rid in self._completed_recv for rid in reads
            ) and self._side_reads_done(room, self._completed_recv)
        return str(room) in self._completed_recv and self._side_reads_done(
            room, self._completed_recv
        )

    def is_failed(self, room: int) -> bool:
        """Whether any transfer of the room -- whole, sliding-window, state, or
        one of its chunks -- has failed."""
        with self._state_lock:
            return self._is_failed_locked(room)

    def _is_failed_locked(self, room: int) -> bool:
        return any(self._is_room_transfer(key, room) for key in self._failed)

    def room_has_active_io(self, room: int) -> bool:
        """Whether raiden still holds a transfer of the room: an unsettled
        read, or a receive plan a push can still land in.

        While it does, the room's pages must not be handed back -- the
        transfer would write them under their next owner. Raiden's own
        transfer deadline bounds how long this can stay true.
        """
        if self._pipeline is not None:
            return self._pipeline.active(room)
        with self._state_lock:
            if self._pulling is not None:
                for key in self._pulling.rooms():
                    if self._is_room_transfer(
                        key, room
                    ) and not self._pull_settled_locked(key):
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
        if self._pipeline is not None:
            self._pipeline.clear(room)
        self._drop_room_state(room, clear_failure_record=True)

    def _drop_room_state(
        self, room: int, *, clear_failure_record: bool = False
    ) -> None:
        def belongs(key: str) -> bool:
            return self._is_room_transfer(key, room)

        with self._state_lock:
            if self._prefill is not None:
                self._prefill.reshard_replied_at.pop(room, None)
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
                    held.add(key)
                    held.update(self._paired_reads(key))
                # A read of the room still waiting for staging will not ask
                # again; it must not hold the line for the other rooms.
                waiting = self._pulling.waiting
                if waiting is not None and belongs(waiting):
                    self._pulling.forget(waiting)
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
            for key in [k for k in self._send_done_at if belongs(k)]:
                del self._send_done_at[key]
            for ids in (self._completed_recv, self._failed, self._native_failed):
                ids.difference_update([k for k in ids if belongs(k) and k not in held])
            announced = self._decode.announced_at if self._decode is not None else None
            if announced is not None:
                for key in [k for k in announced if belongs(k)]:
                    del announced[key]
            if self._prefill is not None:
                for table in (
                    self._prefill.held_chunks,
                    self._prefill.send_total,
                    self._prefill.decode_pages,
                    self._prefill.decode_swa_pages,
                    self._prefill.chunk_seq,
                    self._prefill.reshard_chunks,
                    self._prefill.pending_offer,
                    self._prefill.reshard_ready_pages,
                    self._prefill.reshard_chunk_acked,
                    self._prefill.send_blocks,
                    self._prefill.send_block_set,
                    self._prefill.send_swa_blocks,
                    self._prefill.send_state_rows,
                    self._prefill.send_state_meta,
                    self._prefill.send_aux,
                    self._prefill.decode_addr,
                    self._prefill.reshard_req,
                ):
                    table.pop(room, None)
                for table in (
                    self._prefill.replied,
                    self._prefill.reshard_final_pushed,
                    self._prefill.reshard_finalized,
                    self._prefill.reshard_replied,
                    self._prefill.reshard_ready,
                    self._prefill.reshard_pushed,
                ):
                    table.discard(room)
            if self._decode is not None:
                for table in (
                    self._decode.chunk_queue,
                    self._decode.remote_total,
                    self._decode.announced_spans,
                    self._decode.last_announced_at,
                    self._decode.chunk_reads,
                    self._decode.reshard_covered,
                    self._decode.reshard_chunk_registering,
                    self._decode.reshard_chunk_sources,
                    self._decode.local_blocks,
                    self._decode.local_swa_blocks,
                    self._decode.local_state_rows,
                    self._decode.decode_aux_index,
                    self._decode.remote_blocks,
                    self._decode.remote_swa_blocks,
                    self._decode.remote_state_rows,
                    self._decode.remote_endpoint,
                    self._decode.remote_swa_endpoint,
                    self._decode.remote_state_endpoint,
                    self._decode.reshard_meta,
                    self._decode.reshard_sources,
                    self._decode.reshard_swa_sources,
                ):
                    table.pop(room, None)
                for table in (
                    self._decode.final_seen,
                    self._decode.reshard_registered,
                ):
                    table.discard(room)
            if clear_failure_record:
                # Failure publication takes _state_lock before failure_lock as
                # well, so clear and publication have one total order.
                with self.failure_lock:
                    self.failure_records.pop(room, None)
        for uuid in settled_plans:
            self._unregister_plan(uuid)


class RaidenKVBootstrapServer(CommonKVBootstrapServer):
    pass
