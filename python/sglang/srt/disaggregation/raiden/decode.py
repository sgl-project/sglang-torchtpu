"""Decode-side Raiden rendezvous and transfer protocol."""

from __future__ import annotations

import logging
import time

import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVReceiver
from sglang.srt.disaggregation.raiden.chunk_plan import chunk_pull_window
from sglang.srt.disaggregation.raiden.conn import RaidenKVManager, _tcp_endpoint
from sglang.srt.disaggregation.raiden.host_pool import SlotAdmission
from sglang.srt.disaggregation.raiden.serde import (
    MessageTag,
    pack_ints,
    unpack_blocks,
    unpack_ints,
    write_aux_blob,
)

logger = logging.getLogger(__name__)


class RaidenDecodeKVManager(RaidenKVManager):
    def _message_handlers(self):
        return {
            MessageTag.BLOCKS: self._on_blocks,
            MessageTag.CHUNK: self._on_chunk,
            MessageTag.RESHARD_BLOCKS: self._on_reshard_blocks,
            MessageTag.RESHARD_CHUNK: self._on_reshard_chunk,
        }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pulling: SlotAdmission | None = None
        self._pulling_cleared: set[str] = set()
        self._cleared_plans: dict[str, int] = {}
        self._local_blocks: dict[int, list[int]] = {}
        self._local_sliding_window_attn_blocks: dict[int, list[int]] = {}
        self._decode_aux_index: dict[int, int] = {}
        self._remote_blocks: dict[int, list[int]] = {}
        self._remote_sliding_window_attn_blocks: dict[int, list[int]] = {}
        self._remote_endpoint: dict[int, str] = {}
        self._remote_sliding_window_attn_endpoint: dict[int, str] = {}
        self._chunk_queue: dict[int, list[tuple]] = {}
        self._remote_total: dict[int, int] = {}
        self._announced_spans: dict[int, list[tuple[int, int]]] = {}
        self._chunk_reads: dict[int, list[str]] = {}
        self._final_seen: set[int] = set()
        self._reshard_covered: dict[int, list[tuple[int, int]]] = {}
        self._reshard_ready_pages: dict[int, list[int]] = {}
        self._reshard_chunk_sources: dict[
            int, dict[int, dict[int, tuple[int, list[int]]]]
        ] = {}
        self._reshard_chunk_registering: dict[int, set[int]] = {}
        self._reshard_plans: dict[int, dict[str, int]] = {}
        self._reshard_sources: dict[int, dict[int, list[int]]] = {}
        self._reshard_sliding_window_attn_sources: dict[int, dict[int, list[int]]] = {}
        self._reshard_meta: dict[int, dict] = {}
        self._reshard_registered: set[int] = set()
        self._start_heartbeat_checker_thread()
        self._start_listener()

    def set_kv_caches(self, kv_caches) -> None:
        slot_ceiling = max(
            1,
            int(self.server_args.max_running_requests or 64)
            + int(self.server_args.disaggregation_decode_extra_slots or 0),
        )
        capacity = super().set_kv_caches(kv_caches, slot_ceiling=slot_ceiling)
        self._pulling = SlotAdmission(capacity)

    def _settle_locked(self) -> list[int]:
        self._release_pull_slots_locked()
        return self._settled_cleared_plans_locked()

    def _drop_role_state_locked(self, room: int, belongs) -> tuple[set[str], list[int]]:
        held: set[str] = set()
        if self._pulling is not None:
            for key in self._pulling.rooms():
                if not belongs(key):
                    continue
                if self._pull_settled_locked(key):
                    self._pulling.release(key)
                    continue
                self._pulling_cleared.add(key)
                held.update((key, self._paired_sliding_window_attn_read(key) or key))
        settled_plans = []
        for req_id, uuid in self._reshard_plans.pop(room, {}).items():
            if req_id in self._completed_recv or req_id in self._native_failed:
                settled_plans.append(uuid)
            else:
                self._cleared_plans[req_id] = uuid
                held.add(req_id)
        for state in (
            self._chunk_queue,
            self._remote_total,
            self._announced_spans,
            self._chunk_reads,
            self._reshard_covered,
            self._reshard_ready_pages,
            self._reshard_chunk_registering,
            self._reshard_chunk_sources,
            self._local_blocks,
            self._local_sliding_window_attn_blocks,
            self._decode_aux_index,
            self._remote_blocks,
            self._remote_sliding_window_attn_blocks,
            self._remote_endpoint,
            self._remote_sliding_window_attn_endpoint,
            self._reshard_meta,
            self._reshard_sources,
            self._reshard_sliding_window_attn_sources,
        ):
            state.pop(room, None)
        self._final_seen.discard(room)
        self._reshard_registered.discard(room)
        return held, settled_plans

    @staticmethod
    def _uncovered_pages(total: int, spans: list[tuple[int, int]]) -> list[int]:
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

    def _on_chunk(self, msg: list[bytes]) -> None:
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

    def _on_blocks(self, msg: list[bytes]) -> None:
        self._require_frames(msg, MessageTag.BLOCKS, 7)
        room = int(msg[1].decode("ascii"))
        remote_blocks = unpack_blocks(msg[2])
        endpoint = msg[3].decode("ascii")
        remote_sliding_window_attn_blocks = unpack_blocks(msg[4])
        sliding_window_attn_endpoint = msg[5].decode("ascii")
        aux_blob = msg[6]
        aux_index = None
        with self._state_lock:
            if room in self._aborted:
                return
            self._remote_blocks[room] = remote_blocks
            self._remote_endpoint[room] = endpoint
            self._remote_sliding_window_attn_blocks[room] = (
                remote_sliding_window_attn_blocks
            )
            self._remote_sliding_window_attn_endpoint[room] = (
                sliding_window_attn_endpoint
            )
            aux_index = self._decode_aux_index.get(room)
            if self.chunked_transfer:
                spans = self._announced_spans.get(room, [])
                total = len(remote_blocks)
                if self._remote_total.get(room, total) != total:
                    logger.warning(
                        "RaidenKVManager rank%d room %s announced %d pages but sent %d",
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

    def register_decode_request(
        self,
        room: int,
        local_blocks: list[int],
        local_sliding_window_attn_blocks: list[int],
        aux_index: int | None,
        bootstrap_infos: list[dict],
    ) -> None:
        if len(bootstrap_infos) != 1:
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
                # nothing to receive on either engine.
                with self._state_lock:
                    self._completed_recv.add(str(room))
                    if self.has_sliding_window_attn:
                        self._completed_recv.add(self._sliding_window_attn_req_id(room))
                return
            self._register_reshard_request(
                room,
                local_blocks,
                local_sliding_window_attn_blocks,
                aux_index,
                bootstrap_infos,
            )
            self.update_status(room, KVPoll.WaitingForInput)
            return
        with self._state_lock:
            self._local_blocks[room] = local_blocks
            self._local_sliding_window_attn_blocks[room] = (
                local_sliding_window_attn_blocks
            )
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
            endpoint = _tcp_endpoint(ip=info["rank_ip"], port=info["rank_port"])
            self._send_multipart_locked(endpoint, payload)

    @staticmethod
    def _pair_blocks(
        remote: list[int], local: list[int]
    ) -> tuple[list[int], list[int]]:
        local = list(local)
        remote = list(remote)
        if not local:
            remote = []
        elif len(remote) > len(local):
            remote = remote[-len(local) :]
        elif len(remote) < len(local):
            local = local[: len(remote)]
        return remote, local

    def _paired_sliding_window_attn_read(self, key: str) -> str | None:
        """The sliding-window read issued alongside the read ``key``, if any.

        A whole-room read and the closing read of a chunked transfer carry the
        sliding-window pages on the second engine; earlier chunks do not.
        """
        if not self.has_sliding_window_attn or "#" in key:
            return None
        return self._sliding_window_attn_req_id(int(key))

    def _pull_settled_locked(self, key: str) -> bool:
        """Whether every engine has reported the read ``key`` done or failed.

        Raiden holds a read's host staging until it completes or times out,
        whichever engine it is on, so the read's admission stays claimed until
        both it and its paired sliding-window read have settled.
        """
        settled = key in self._completed_recv or key in self._native_failed
        sliding_window_attn_rid = self._paired_sliding_window_attn_read(key)
        if sliding_window_attn_rid is not None:
            settled = settled and (
                sliding_window_attn_rid in self._completed_recv
                or sliding_window_attn_rid in self._native_failed
            )
        return settled

    def _settled_cleared_plans_locked(self) -> list[int]:
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
                for rid in (key, self._paired_sliding_window_attn_read(key) or key):
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
                if len(pairs) > self.full_attn_engine.max_blocks:
                    oversized = (
                        len(pairs),
                        self.full_attn_engine.max_blocks,
                        "full-attention",
                        "SGLANG_RAIDEN_MAX_BLOCKS",
                    )
                    break
                # The closing read also carries the sliding-window pages, on
                # the engine that stages them against its own limit.
                closing = req_id == str(room) and self.has_sliding_window_attn
                if closing:
                    remote_swa, local_swa = self._pair_blocks(
                        list(self._remote_sliding_window_attn_blocks.get(room, [])),
                        list(self._local_sliding_window_attn_blocks.get(room, [])),
                    )
                    sliding_window_attn_endpoint = (
                        self._remote_sliding_window_attn_endpoint.get(room, "")
                    )
                    sliding_window_attn_pages = len(set(local_swa))
                    if (
                        sliding_window_attn_pages
                        > self.sliding_window_attn_engine.max_blocks
                    ):
                        oversized = (
                            sliding_window_attn_pages,
                            self.sliding_window_attn_engine.max_blocks,
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
            self.full_attn_engine.start_read(
                req_id, uuid, endpoint, remote_slice, local_slice
            )
            if closing:
                self.sliding_window_attn_engine.start_read(
                    self._sliding_window_attn_req_id(room),
                    room,
                    sliding_window_attn_endpoint,
                    remote_swa,
                    local_swa,
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
                    self.has_sliding_window_attn
                    and room not in self._remote_sliding_window_attn_blocks
                )
                or (
                    self.has_sliding_window_attn
                    and room not in self._remote_sliding_window_attn_endpoint
                )
            ):
                return False
            local = list(self._local_blocks.get(room, []))
            remote = list(self._remote_blocks[room])
            endpoint = self._remote_endpoint[room]
            local_swa = list(self._local_sliding_window_attn_blocks.get(room, []))
            remote_swa = list(self._remote_sliding_window_attn_blocks.get(room, []))
            sliding_window_attn_endpoint = (
                self._remote_sliding_window_attn_endpoint.get(room, "")
            )
        # start_read needs equal-length, position-paired remote/local. Reconcile
        # the two independent ways they can differ:
        #   - decode prefix-cache hit: decode already holds the head pages and
        #     allocated fewer local pages, so pull only the matching tail of
        #     remote (remote[-len(local):]).
        #   - reserved decode token: decode allocated one extra trailing page that
        #     prefill produced no KV for, so drop it from local.
        # Empty local => full_attn prefix hit; pull nothing (and avoid remote[-0:],
        # which would select the whole list).
        remote, local = self._pair_blocks(remote, local)
        full_attn_pages = len(set(local))
        if full_attn_pages > self.full_attn_engine.max_blocks:
            self._reject_oversized_pull(
                room,
                full_attn_pages,
                self.full_attn_engine.max_blocks,
                "full-attention",
                "SGLANG_RAIDEN_MAX_BLOCKS",
            )
            return False
        if self.has_sliding_window_attn:
            remote_swa, local_swa = self._pair_blocks(remote_swa, local_swa)
            sliding_window_attn_pages = len(set(local_swa))
            if sliding_window_attn_pages > self.sliding_window_attn_engine.max_blocks:
                self._reject_oversized_pull(
                    room,
                    sliding_window_attn_pages,
                    self.sliding_window_attn_engine.max_blocks,
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
        self.full_attn_engine.start_read(str(room), room, endpoint, remote, local)
        if self.has_sliding_window_attn:
            self.sliding_window_attn_engine.start_read(
                self._sliding_window_attn_req_id(room),
                room,
                sliding_window_attn_endpoint,
                remote_swa,
                local_swa,
            )
        return True

    def _register_reshard_request(
        self,
        room: int,
        local_blocks: list[int],
        local_sliding_window_attn_blocks: list[int],
        aux_index: int | None,
        bootstrap_infos: list[dict],
    ) -> None:
        # bootstrap_infos is ordered by the decode->prefill target_tp_ranks
        # (range(r*ratio, (r+1)*ratio)); ratio == number of prefill sources.
        ratio = len(bootstrap_infos)
        src_attn_tp = self.attn_tp_size * ratio
        r = self.kv_args.engine_rank % self.attn_tp_size
        source_ranks = list(range(r * ratio, (r + 1) * ratio))
        push_endpoint = self.full_attn_engine.transfer_address
        sliding_window_attn_push_endpoint = (
            self.sliding_window_attn_engine.transfer_address
            if self.sliding_window_attn_engine is not None
            else ""
        )
        ready_endpoints = [
            _tcp_endpoint(ip=info["rank_ip"], port=info["rank_port"])
            for info in bootstrap_infos
        ]
        with self._state_lock:
            self._reshard_meta[room] = dict(
                local_blocks=list(local_blocks),
                local_sliding_window_attn_blocks=list(
                    local_sliding_window_attn_blocks or []
                ),
                source_ranks=list(source_ranks),
                src_attn_tp=int(src_attn_tp),
                aux_index=aux_index,
                ready_endpoints=ready_endpoints,
            )
            self._reshard_sources[room] = {}
            self._reshard_sliding_window_attn_sources[room] = {}
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
            pack_ints(list(local_sliding_window_attn_blocks or [])),
            sliding_window_attn_push_endpoint.encode("ascii"),
        ]
        for info in bootstrap_infos:
            endpoint = _tcp_endpoint(ip=info["rank_ip"], port=info["rank_port"])
            self._send_multipart_locked(endpoint, payload)

    def _on_reshard_chunk(self, msg: list[bytes]) -> None:
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
        sources_map: dict[int, tuple[int, list[int]]],
    ) -> int | None:
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
            uuid = self._next_uuid_locked()
        req_id = f"{room}#{index}"
        req = build_receiver_request(
            pb2,
            uuid=uuid,
            req_id=req_id,
            peer=self.full_attn_engine.transfer_address,
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

    def _on_reshard_blocks(self, msg: list[bytes]) -> None:
        self._require_frames(msg, MessageTag.RESHARD_BLOCKS, 6)
        room = int(msg[1].decode("ascii"))
        prefill_blocks = unpack_ints(msg[2])
        src_rank = int(msg[3].decode("ascii"))
        aux_blob = msg[4]
        prefill_sliding_window_attn_blocks = unpack_ints(msg[5])
        ready = False
        aux_index = None
        with self._state_lock:
            if room not in self._reshard_meta or room in self._aborted:
                return
            self._reshard_sources[room][src_rank] = prefill_blocks
            self._reshard_sliding_window_attn_sources.setdefault(room, {})[
                src_rank
            ] = prefill_sliding_window_attn_blocks
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
        ready_payload: list[bytes],
        ready_endpoints: list[str],
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
        try:
            self.full_attn_engine.register_plan(uuid, req, is_sender=False)
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

    def _register_reshard_sliding_window_attn_receiver(
        self,
        room: int,
        src_attn_tp: int,
        local_sliding_window_attn_blocks: list[int],
        sliding_window_attn_sources_map: dict[int, list[int]],
    ) -> bool:
        """Arm the unchunked sliding-window receive before READY is sent."""
        engine = self.sliding_window_attn_engine
        if engine is None:
            return True
        if not local_sliding_window_attn_blocks:
            logger.warning(
                "RaidenKVManager rank%d raiden reshard sliding-window attention receive SKIPPED "
                "room=%s -- no local window pages",
                self.attn_tp_rank,
                room,
            )
            with self._state_lock:
                self._completed_recv.add(self._sliding_window_attn_req_id(room))
            return True

        from tpu_sync.rpc import raiden_service_pb2 as pb2

        from sglang.srt.disaggregation.raiden.reshard_schedule import (
            build_receiver_request,
        )

        n = len(local_sliding_window_attn_blocks)
        sources: list[tuple[int, list[int]]] = []
        for p_rank in sorted(sliding_window_attn_sources_map):
            pages = sliding_window_attn_sources_map[p_rank]
            if len(pages) < n:
                reason = (
                    f"Raiden reshard sliding-window attention receive: source rank {p_rank} sent "
                    f"{len(pages)} sliding-window pages, decode needs {n}"
                )
                logger.error("RaidenKVManager rank%d %s", self.attn_tp_rank, reason)
                with self._state_lock:
                    self._failed.add(str(room))
                    self.record_failure(room, reason)
                return False
            sources.append((p_rank, pages[-n:]))

        req = build_receiver_request(
            pb2,
            uuid=room,
            req_id=self._sliding_window_attn_req_id(room),
            peer=engine.transfer_address,
            decode_rank=self.attn_tp_rank,
            geom=self._make_geometry(
                src_attn_tp, self.attn_tp_size, sliding_window_attn=True
            ),
            sources=sources,
            dst_block_ids=local_sliding_window_attn_blocks,
            live_tokens=[self.kv_args.page_size] * n,
        )
        try:
            engine.register_plan(room, req, is_sender=False)
        except Exception as exc:
            reason = f"Raiden reshard sliding-window attention receive: cannot stage plan {room}: {exc}"
            logger.error("RaidenKVManager rank%d %s", self.attn_tp_rank, reason)
            with self._state_lock:
                if room not in self._reshard_meta:
                    return False
                self._failed.add(str(room))
                self.record_failure(room, reason)
            return False
        with self._state_lock:
            if room not in self._reshard_meta or room in self._aborted:
                engine.unregister_plan(room)
                return False
            self._reshard_plans.setdefault(room, {})[
                self._sliding_window_attn_req_id(room)
            ] = room
        logger.info(
            "raiden reshard sliding-window attention receive room=%s decode_rank=%d sources=%d "
            "pages=%d layers=%d",
            room,
            self.attn_tp_rank,
            len(sources),
            n,
            engine.kv_layer_count,
        )
        return True

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
            local_sliding_window_attn_blocks = list(
                meta.get("local_sliding_window_attn_blocks") or []
            )
            src_attn_tp = meta["src_attn_tp"]
            ready_endpoints = list(meta["ready_endpoints"])
            sources_map = dict(self._reshard_sources[room])
            sliding_window_attn_sources_map = dict(
                self._reshard_sliding_window_attn_sources.get(room, {})
            )

        if not self._register_reshard_sliding_window_attn_receiver(
            room,
            src_attn_tp,
            local_sliding_window_attn_blocks,
            sliding_window_attn_sources_map,
        ):
            return

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
            peer=self.full_attn_engine.transfer_address,
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

    def is_recv_done(self, room: int) -> bool:
        if self.chunked_transfer:
            with self._state_lock:
                if room not in self._final_seen or self._chunk_queue.get(room):
                    return False
                reads = list(self._chunk_reads.get(room, []))
            return all(rid in self._completed_recv for rid in reads) and (
                not self.has_sliding_window_attn
                or self._sliding_window_attn_req_id(room) in self._completed_recv
            )
        rid = str(room)
        return rid in self._completed_recv and (
            not self.has_sliding_window_attn
            or self._sliding_window_attn_req_id(room) in self._completed_recv
        )

    def room_has_active_io(self, room: int) -> bool:
        """Whether raiden still holds a transfer of the room: an unsettled
        read, or a receive plan a push can still land in.

        While it does, the room's pages must not be handed back -- the
        transfer would write them under their next owner. Raiden's own
        transfer deadline bounds how long this can stay true.
        """
        rid, sliding_window_attn_rid, chunk_prefix = (
            str(room),
            self._sliding_window_attn_req_id(room),
            f"{room}#",
        )

        def belongs(key: str) -> bool:
            return (
                key == rid
                or key == sliding_window_attn_rid
                or key.startswith(chunk_prefix)
            )

        with self._state_lock:
            if self._pulling is not None:
                for key in self._pulling.rooms():
                    if belongs(key) and not self._pull_settled_locked(key):
                        return True
            for req_id in self._reshard_plans.get(room, {}):
                if (
                    req_id not in self._completed_recv
                    and req_id not in self._native_failed
                ):
                    return True
        return False


class RaidenKVReceiver(CommonKVReceiver):
    # See RaidenKVSender: a group Failed waits for every rank's transfers.
    group_failure_requires_drain = True

    def __init__(
        self,
        mgr: RaidenDecodeKVManager,
        bootstrap_addr: str,
        bootstrap_room: int | None = None,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room)
        self._started = False

    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: int | None = None,
        state_indices: list | None = None,
        decode_prefix_len: int | None = None,
    ):
        if self.conclude_state == KVPoll.Failed:
            return
        self.kv_mgr.register_decode_request(
            self.bootstrap_room,
            [int(x) for x in kv_indices],
            self.kv_mgr.sliding_window_attn_blocks_from_state_indices(state_indices),
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
        self.kv_mgr.drain_engine()
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
        reason: str | None = None,
        preserve_reason: bool = False,
    ) -> KVPoll:
        """Terminalize the room: tombstone it so late frames cannot rebuild
        it, and tell the prefill to release it.

        While raiden still holds one of the room's transfers, Failed is
        withheld -- the scheduler frees the room's pages on it, and the
        transfer would write them under their next owner. The tombstone
        already stops anything new; the settled room fails on a later poll.
        """
        self.kv_mgr.fail_room(
            self.bootstrap_room, reason, preserve_reason=preserve_reason
        )
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
