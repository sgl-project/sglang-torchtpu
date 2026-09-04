"""Shared control plane and engine setup for TPU Raiden P/D.

``RaidenKVManager`` owns common request lifecycle, ZMQ rendezvous,
topology, host-pool sizing, and the full-attention and optional sliding-window attention
engines. Role-specific protocols live in ``prefill.py`` and ``decode.py``;
``engine.py`` isolates direct calls to the native transfer engine.

Matching-TP transfers use consumer pull. Asymmetric-TP transfers use
prefill push with receiver-side resharding.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll, StateType
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
)
from sglang.srt.disaggregation.raiden.engine import RaidenEngine, RaidenEngineConfig
from sglang.srt.disaggregation.raiden.host_pool import (
    node_worker_rank,
    ranks_per_node,
    slot_host_bytes,
    slots_within_budget,
)
from sglang.srt.disaggregation.raiden.serde import MessageTag
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.network import NetworkAddress

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)

# Base of the per-rank control ports (`base + 4*local_rank`). Not 30000: that
# puts rank 0 on the default HTTP serving port, and the engine wins the bind --
# uvicorn then dies with EADDRINUSE and takes the server down before any
# transfer is attempted.
DEFAULT_KV_PORT = 31000


def _tcp_endpoint(ip: str, port: int) -> str:
    """``tcp://`` endpoint of a peer rank's rendezvous socket.

    Peer addresses travel as bare literals (``rank_ip`` from the bootstrap
    server, the ip frame of PULL_REQ / RESHARD_REQ); an IPv6 literal must be
    bracketed before a port is appended or the port merges into the address.
    The bracket is also what ``_send_multipart_locked`` keys the IPv6 socket
    option on.
    """
    return NetworkAddress(host=ip, port=int(port)).to_tcp()


@dataclass(frozen=True)
class RaidenKVCacheGroups:
    """Page-compatible cache groups registered with independent Raiden engines."""

    full_attn: list[Any]
    sliding_window_attn: list[Any]


def get_raiden_kv_caches(
    pool: KVCache, draft_pool: KVCache | None = None
) -> RaidenKVCacheGroups:
    """Group per-layer device KV tensors by their page-index namespace.

    The TPU fused KV pool stores one [num_blocks, page_size, ...] tensor per
    layer in ``kv_buffer`` and exposes them via ``get_fused_kv_buffer``. Hybrid
    sliding-window-attention pools allocate full-attention and sliding-window pages independently, so
    they must not share one Raiden block-id list. Speculative decoding uses the
    same page indices for target and draft KV, so append the draft tensors to
    the matching namespace and move them in the same Raiden transfer.
    """

    def group_one(current_pool: KVCache) -> RaidenKVCacheGroups:
        if hasattr(current_pool, "get_fused_kv_buffer"):
            start = current_pool.start_layer
            buffers = [
                current_pool.get_fused_kv_buffer(start + i)
                for i in range(current_pool.layer_num)
            ]
        elif hasattr(current_pool, "kv_buffer"):
            buffers = list(current_pool.kv_buffer)
        else:
            raise RuntimeError(
                "Raiden backend requires a fused KV pool exposing kv_buffer; got "
                f"{type(current_pool).__name__}"
            )
        layers_mapping = getattr(current_pool, "layers_mapping", None)
        if layers_mapping:
            full_attn = []
            sliding_window_attn = []
            for offset, buf in enumerate(buffers):
                layer_id = current_pool.start_layer + offset
                (
                    sliding_window_attn if layers_mapping[layer_id][1] else full_attn
                ).append(buf)
            return RaidenKVCacheGroups(
                full_attn=full_attn, sliding_window_attn=sliding_window_attn
            )
        return RaidenKVCacheGroups(full_attn=buffers, sliding_window_attn=[])

    groups = group_one(pool)
    full_attn, sliding_window_attn = list(groups.full_attn), list(
        groups.sliding_window_attn
    )
    if draft_pool is not None and draft_pool is not pool:
        draft_groups = group_one(draft_pool)
        full_attn.extend(draft_groups.full_attn)
        sliding_window_attn.extend(draft_groups.sliding_window_attn)

    for name, group in (("full-attention", full_attn), ("SWA", sliding_window_attn)):
        block_counts = {int(buf.shape[0]) for buf in group}
        if len(block_counts) > 1:
            raise RuntimeError(
                f"Raiden requires uniform {name} KV block counts; got "
                f"{sorted(block_counts)}"
            )
    return RaidenKVCacheGroups(
        full_attn=full_attn, sliding_window_attn=sliding_window_attn
    )


def _request_max_blocks(context_length: int | None, page_size: int) -> int:
    override = int(os.environ.get("SGLANG_RAIDEN_MAX_BLOCKS", "0"))
    if override > 0:
        return override
    max_len = context_length or int(
        os.environ.get("SGLANG_RAIDEN_MAX_MODEL_LEN", "8192")
    )
    return max(1, math.ceil(max_len / page_size))


def _sliding_window_attn_request_max_blocks(
    sliding_window_size: int | None, page_size: int
) -> int:
    override = int(os.environ.get("SGLANG_RAIDEN_SWA_MAX_BLOCKS", "0"))
    if override > 0:
        return override
    if sliding_window_size is None or sliding_window_size <= 0:
        raise RuntimeError(
            "Raiden sliding-window-attention KV caches require a positive sliding-window size"
        )
    max_tokens = sliding_window_size + page_size - 1
    return max(1, math.ceil(max_tokens / page_size))


def _slots_for_engine(
    kv_caches: list[Any],
    max_blocks: int,
    *,
    budget_share: float,
    ceiling: int,
) -> int:
    return slots_within_budget(
        budget_share,
        slot_host_bytes(kv_caches, max_blocks),
        ceiling,
    )


def build_engine_configs(
    kv_groups: RaidenKVCacheGroups,
    *,
    local_ip: str,
    local_rank: int,
    control_port: int,
    timeout_s: float,
    context_length: int | None,
    page_size: int,
    sliding_window_size: int | None,
    slot_ceiling: int,
    local_ranks: int,
    rank: int,
) -> tuple[RaidenEngineConfig, RaidenEngineConfig | None]:
    """Compute the coupled full-attention/sliding-window-attention engine configurations in one budget pass."""
    if not kv_groups.full_attn:
        raise RuntimeError(
            "Raiden requires at least one full-attention KV cache tensor"
        )

    engine_count = 2 if kv_groups.sliding_window_attn else 1
    host_budget = float(os.environ.get("SGLANG_RAIDEN_HOST_POOL_GB", "128")) * 1024**3
    budget_share = host_budget / (max(1, local_ranks) * engine_count)
    full_attn_max_blocks = _request_max_blocks(context_length, page_size)
    full_attn_slots = _slots_for_engine(
        kv_groups.full_attn,
        full_attn_max_blocks,
        budget_share=budget_share,
        ceiling=slot_ceiling,
    )
    full_attn_config = RaidenEngineConfig(
        kv_caches=list(kv_groups.full_attn),
        node_id=local_rank,
        local_control_port=control_port,
        endpoint=NetworkAddress(host=local_ip, port=control_port).to_host_port_str(),
        max_blocks=full_attn_max_blocks,
        num_slots=full_attn_slots,
        timeout_s=timeout_s,
        label="full-attention",
        rank=rank,
    )
    if not kv_groups.sliding_window_attn:
        return full_attn_config, None

    sliding_window_attn_max_blocks = _sliding_window_attn_request_max_blocks(
        sliding_window_size, page_size
    )
    sliding_window_attn_slots = _slots_for_engine(
        kv_groups.sliding_window_attn,
        sliding_window_attn_max_blocks,
        budget_share=budget_share,
        ceiling=slot_ceiling,
    )
    return full_attn_config, RaidenEngineConfig(
        kv_caches=list(kv_groups.sliding_window_attn),
        node_id=local_rank,
        local_control_port=control_port + 2,
        endpoint=NetworkAddress(
            host=local_ip, port=control_port + 2
        ).to_host_port_str(),
        max_blocks=sliding_window_attn_max_blocks,
        num_slots=sliding_window_attn_slots,
        timeout_s=timeout_s,
        label="sliding-window",
        rank=rank,
    )


class RaidenKVManager(CommonKVManager):
    """Coordinates shared request state and the active Raiden role.

    Each ``RaidenEngine`` owns one native page-index namespace. This manager owns ZMQ
    rendezvous I/O, interprets transport events, tracks room lifecycle, and
    dispatches messages to the concrete prefill or decode workflow.

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
        is_mla_backend: bool | None = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)

        self._full_attn_engine: RaidenEngine | None = None
        self._sliding_window_attn_engine: RaidenEngine | None = None
        base_port = int(os.environ.get("SGLANG_RAIDEN_KV_PORT", str(DEFAULT_KV_PORT)))
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
        self._completed_send: set[str] = set()
        self._completed_recv: set[str] = set()
        self._failed: set[str] = set()
        # Transfer ids the engine reported failed. Only these mean raiden has
        # released the transfer's staging; a room failed for a logical reason
        # (abort, timeout, peer death) shares its id with its closing read, so
        # room failure must never read as native settlement.
        self._native_failed: set[str] = set()
        # Shared state: native completions, room failure, and abort lifecycle.
        self._aborted: dict[int, float] = {}
        self._status_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stale_plans: list[tuple[int, int, Any]] = []
        self._uuid_seq = 0

    @property
    def full_attn_engine(self) -> RaidenEngine:
        if self._full_attn_engine is None:
            raise RuntimeError("Raiden KV caches have not been configured")
        return self._full_attn_engine

    @property
    def sliding_window_attn_engine(self) -> RaidenEngine | None:
        return self._sliding_window_attn_engine

    @property
    def has_sliding_window_attn(self) -> bool:
        return self._sliding_window_attn_engine is not None

    def _engines(self):
        if self._full_attn_engine is not None:
            yield self._full_attn_engine
        if self._sliding_window_attn_engine is not None:
            yield self._sliding_window_attn_engine

    # ----- resharding geometry / host-slot helpers --------------------------

    def _make_geometry(
        self, src_attn_tp: int, dst_attn_tp: int, *, sliding_window_attn: bool = False
    ):
        """ReshardGeometry for a (src_attn_tp -> dst_attn_tp) transfer, sized from
        the local fused KV tensor (the K+V head-pair granule matches on both
        sides) and the global kv-head count.

        ``sliding_window_attn`` sizes it from the sliding-window sub-pool instead. Its layers
        carry the same kv-head count as the full-attention pool -- windowing bounds how
        many tokens are kept, not how the heads are sharded -- so only the
        head-pair granule is read from the other tensor, and the head-slice math
        is identical. Taking it from the sliding-window-attention tensor rather than assuming it
        matches keeps the two pools free to differ in dtype or head_dim.
        """
        from sglang.srt.disaggregation.raiden.reshard_schedule import (
            ReshardGeometry,
            head_pair_bytes,
        )

        engine = (
            self.sliding_window_attn_engine
            if sliding_window_attn
            else self.full_attn_engine
        )
        if engine is None:
            raise RuntimeError("Raiden sliding-window engine is not configured")
        tensor = engine.kv_tensor(0)
        hp = head_pair_bytes(tensor)
        # Distinct model kv-head count (model_config.get_total_num_kv_heads()),
        # set on both the prefill and decode KVArgs.
        total_kv_heads = int(self.kv_args.total_kv_head_num)
        return ReshardGeometry(src_attn_tp, dst_attn_tp, hp, total_kv_heads)

    # ----- engine configuration and completion events -----------------------

    def set_kv_caches(
        self, kv_caches: RaidenKVCacheGroups, *, slot_ceiling: int | None = None
    ) -> int:
        """Hand raiden the per-layer device KV tensors and build the engines.

        Called by prefill.py/decode.py right after construction (the KV pool is
        only available there). Raiden indexes each tensor's major dim as a
        block/page, which matches the TPU fused KV pool layout
        ([num_blocks, page_size, ...]).
        """
        # Full-attention KV moves piece by piece during the prefill. A hybrid
        # model's sliding-window pool settles when the prefill ends.
        self.chunked_transfer = True
        if slot_ceiling is None:
            slot_ceiling = max(1, int(self.server_args.max_running_requests or 64))
        local_rank_count = ranks_per_node(
            self.attn_tp_size,
            self.attn_dp_size,
            self.system_dp_size,
            self.server_args.nnodes,
            self.attn_cp_size,
        )
        full_attn_config, sliding_window_attn_config = build_engine_configs(
            kv_caches,
            local_ip=self.local_ip,
            local_rank=self._local_rank,
            control_port=self._control_port,
            timeout_s=self._timeout_s,
            context_length=self.server_args.context_length,
            page_size=self.kv_args.page_size,
            sliding_window_size=self.kv_args.sliding_window_size,
            slot_ceiling=slot_ceiling,
            local_ranks=local_rank_count,
            rank=self.attn_tp_rank,
        )
        self._full_attn_engine = RaidenEngine.create(full_attn_config)
        self._sliding_window_attn_engine = (
            RaidenEngine.create(sliding_window_attn_config)
            if sliding_window_attn_config
            else None
        )
        capacity = self.full_attn_engine.num_slots
        if self.sliding_window_attn_engine is not None:
            capacity = min(capacity, self.sliding_window_attn_engine.num_slots)
        return capacity

    def _next_uuid_locked(self) -> int:
        self._uuid_seq += 1
        return (self._uuid_seq << 20) | (self._local_rank & 0xFFFFF)

    @staticmethod
    def _norm_req_id(rid: str) -> str:
        # Resharded (push) transfers surface as "resharded_transfer_<room>" from
        # poll_stats; strip the prefix so completion maps back to str(room).
        prefix = "resharded_transfer_"
        return rid.removeprefix(prefix)

    @staticmethod
    def _sliding_window_attn_req_id(room: int) -> str:
        return f"swa:{room}"

    def _start_listener(self) -> None:
        self._listener = threading.Thread(target=self._listen, daemon=True)
        self._listener.start()

    def _message_handlers(self):
        raise NotImplementedError

    def _settle_locked(self) -> list[int]:
        raise NotImplementedError

    def _drop_role_state_locked(self, room: int, belongs) -> tuple[set[str], list[int]]:
        raise NotImplementedError

    def _unregister_plan(self, uuid: int) -> None:
        for engine in self._engines():
            engine.unregister_plan(uuid)

    def drain_engine(self) -> None:
        results = [engine.poll() for engine in self._engines()]
        if not results:
            return
        for done_send, done_recv, failed in results:
            if done_send or done_recv or failed:
                with self._state_lock:
                    self._completed_send.update(self._norm_req_id(r) for r in done_send)
                    self._completed_recv.update(self._norm_req_id(r) for r in done_recv)
                    norm_failed = {self._norm_req_id(r) for r in failed}
                    self._native_failed.update(norm_failed)
                    self._failed.update(norm_failed)
                    settled_plans = self._settle_locked()
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

    def fail_room(
        self,
        room: int,
        reason: str | None = None,
        *,
        preserve_reason: bool = False,
        preserve_succeeded: bool = False,
    ) -> bool:
        """Atomically tombstone and fail a known room.

        Unknown rooms are only tombstoned so a late control message cannot
        recreate them. Remote aborts may preserve an already successful room.
        """
        with self._state_lock:
            try:
                status = self.check_status(room)
            except KeyError:
                status = None
            if preserve_succeeded and status == KVPoll.Success:
                return False
            self._mark_aborted_locked(room)
            if status is None:
                return False
            if reason is not None:
                self._fail_room_locked(room, reason, preserve_reason=preserve_reason)
            self.update_status(room, KVPoll.Failed)
            return True

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
        self, endpoint: str, parts: list[bytes], is_ipv6: bool = False
    ) -> None:
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

    # ----- rendezvous (ZMQ over the existing per-rank PULL socket) -----------

    def _listen(self) -> None:
        while True:
            try:
                msg = self.server_socket.recv_multipart()
            except Exception as exc:
                logger.error("%s listener error: %s", type(self).__name__, exc)
                continue
            try:
                self._dispatch(msg)
            except Exception as exc:
                logger.error("%s message handling error: %s", type(self).__name__, exc)

    def _dispatch(self, msg: list[bytes]) -> None:
        """Route one multipart control message to the concrete role manager."""
        if not msg:
            return
        handler = self._message_handlers().get(msg[0])
        if handler is None:
            logger.warning(
                "%s received unknown Raiden message tag %r",
                type(self).__name__,
                msg[0],
            )
            return
        handler(msg)

    @staticmethod
    def _require_frames(msg: list[bytes], tag: MessageTag, n: int) -> None:
        """Reject a malformed header before it is indexed past its length."""
        if len(msg) != n:
            raise ValueError(
                f"malformed {tag.name} message: expected {n} frames, got {len(msg)}"
            )

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

    # ----- KV state helpers -----------------------------------------------

    def sliding_window_attn_blocks_from_state_indices(
        self, state_indices: list | None
    ) -> list[int]:
        if not self.has_sliding_window_attn:
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
            raise RuntimeError(
                "Raiden sliding-window-attention transfer is missing its page indices"
            )
        return [int(x) for x in state_indices[index]]

    # ----- completion queries (drained by poll()) ---------------------------

    def is_failed(self, room: int) -> bool:
        """Whether any transfer of the room -- whole, sliding-window, or one of
        its chunks -- has failed."""
        rid, sliding_window_attn_rid, chunk_prefix = (
            str(room),
            self._sliding_window_attn_req_id(room),
            f"{room}#",
        )
        with self._state_lock:
            return any(
                key == rid
                or key == sliding_window_attn_rid
                or key.startswith(chunk_prefix)
                for key in self._failed
            )

    def clear_room(self, room: int) -> None:
        """Forget the room's transfer state.

        An abort tombstone survives the room: frames for it can still arrive
        and must stay rejected. Tombstones age out on their own.
        """
        self.drop_room_state(room, clear_failure_record=True)

    def drop_room_state(self, room: int, *, clear_failure_record: bool = False) -> None:
        rid = str(room)
        sliding_window_attn_rid = self._sliding_window_attn_req_id(room)
        chunk_prefix = f"{room}#"

        def belongs(key: str) -> bool:
            return (
                key == rid
                or key == sliding_window_attn_rid
                or key.startswith(chunk_prefix)
            )

        with self._state_lock:
            held, settled_plans = self._drop_role_state_locked(room, belongs)
            self._completed_send.difference_update(
                [k for k in self._completed_send if belongs(k)]
            )
            for ids in (self._completed_recv, self._failed, self._native_failed):
                ids.difference_update([k for k in ids if belongs(k) and k not in held])
            if clear_failure_record:
                # Failure publication takes _state_lock before failure_lock as
                # well, so clear and publication have one total order.
                with self.failure_lock:
                    self.failure_records.pop(room, None)
        for uuid in settled_plans:
            self._unregister_plan(uuid)


class RaidenKVBootstrapServer(CommonKVBootstrapServer):
    pass
