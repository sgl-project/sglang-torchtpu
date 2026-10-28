"""Decode-side half of the raiden disaggregation backend.

Everything the decode worker does: asking a prefill worker for a room, taking
the announced chunks and block-ids, issuing the raiden pulls, and registering
the receive side of the strided push path used on asymmetric TP.

``RaidenReceiverMixin`` is mixed into ``RaidenKVManager`` in ``conn.py``; it is
not instantiated on its own and deliberately defines no ``__init__``.
"""

from __future__ import annotations

import secrets
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVReceiver, KVTransferError
from sglang.srt.environ import envs
from sglang_tpu.disaggregation.raiden.chunk_plan import chunk_pull_window
from sglang_tpu.disaggregation.raiden.common import _tcp_endpoint, logger
from sglang_tpu.disaggregation.raiden.serde import (
    Blocks,
    Chunk,
    PullReq,
    ReshardBlocks,
    ReshardChunk,
    ReshardChunkReady,
    ReshardReady,
    ReshardReq,
    RoomFailed,
    write_aux_blob,
)

if TYPE_CHECKING:
    from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager


def _present_ids(*req_ids: Optional[str]) -> Tuple[str, ...]:
    return tuple(rid for rid in req_ids if rid is not None)


class RaidenReceiverMixin:
    """Decode-side methods of ``RaidenKVManager``."""

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

    def _record_announcement_locked(self, room: int, req_id: str) -> None:
        now = self._monotonic()
        self._decode.announced_at[req_id] = now
        if room in self._decode.local_blocks:
            self._decode.last_announced_at[room] = now

    def _on_chunk(self, msg: List[bytes]) -> None:
        req = Chunk.from_frames(msg)
        room = req.room
        req_id = req.req_id
        uuid = req.uuid
        start = req.start
        total_pages = req.total_pages
        blocks = req.blocks
        endpoint = req.endpoint
        with self._state_lock:
            if (
                room in self._aborted
                or room in self._decode.final_seen
                or req_id in self._decode.announced_at
            ):
                return
            self._record_announcement_locked(room, req_id)
            self._decode.remote_total[room] = total_pages
            self._decode.announced_spans.setdefault(room, []).append(
                (start, start + len(blocks))
            )
            self._decode.chunk_queue.setdefault(room, []).append(
                (
                    req_id,
                    uuid,
                    list(range(start, start + len(blocks))),
                    blocks,
                    endpoint,
                )
            )
        # The listener handles control messages in order, so start_read must
        # stay cheap.
        if self._listener_start_read and self.chunked_transfer:
            self._start_chunked_pull(room)

    def _on_blocks(self, msg: List[bytes]) -> None:
        req = Blocks.from_frames(msg)
        room = req.room
        remote_blocks = req.blocks
        endpoint = req.endpoint
        remote_swa_blocks = req.swa_blocks
        swa_endpoint = req.swa_endpoint
        aux_blob = req.aux_blob

        aux_index = None
        with self._state_lock:
            if room in self._aborted or room in self._decode.final_seen:
                return
            self._decode.remote_blocks[room] = remote_blocks
            self._decode.remote_endpoint[room] = endpoint
            self._decode.remote_swa_blocks[room] = remote_swa_blocks
            self._decode.remote_swa_endpoint[room] = swa_endpoint
            self._decode.remote_state_rows[room] = list(req.state_blocks)
            self._decode.remote_state_endpoint[room] = req.state_endpoint
            self._record_announcement_locked(room, str(room))
            aux_index = self._decode.decode_aux_index.get(room)
            # Written under the lock: a room failing concurrently frees its
            # metadata slot, and the write must never land in a reused one.
            if aux_index is not None and aux_blob and room not in self._aborted:
                write_aux_blob(
                    self.kv_args.aux_data_ptrs,
                    self.kv_args.aux_item_lens,
                    aux_index,
                    aux_blob,
                )
            if (
                req.state_meta
                and self._write_state_meta is not None
                and room not in self._aborted
            ):
                self._write_state_meta(
                    self._decode.local_state_rows.get(room, ()),
                    req.state_meta,
                )
            if self.chunked_transfer:
                holds_nothing = (
                    room in self._decode.local_blocks
                    and not self._decode.local_blocks[room]
                )
                spans = self._decode.announced_spans.get(room, [])
                total = len(remote_blocks)
                if self._decode.remote_total.get(room, total) != total:
                    logger.warning(
                        "RaidenKVManager rank%d room %s announced %d pages but sent %d",
                        self.attn_tp_rank,
                        room,
                        self._decode.remote_total[room],
                        total,
                    )
                self._decode.remote_total[room] = total
                missing = self._uncovered_pages(total, spans)
                if holds_nothing:
                    # Full prefix hit: BLOCKS and its aux blob complete the KV
                    # engines without a closing read; GDN state is still pulled.
                    self._completed_recv.add(str(room))
                    if self._swa_engine is not None:
                        self._completed_recv.add(self._swa_req_id(room))
                elif missing:
                    self._decode.chunk_queue.setdefault(room, []).append(
                        (
                            str(room),  # Request ID; the integer below is the UUID.
                            room,
                            missing,
                            [remote_blocks[p] for p in missing],
                            endpoint,
                        )
                    )
                self._decode.final_seen.add(room)
        if self._listener_start_read and self.chunked_transfer:
            self._start_chunked_pull(room)

    def _on_room_failed(self, msg: List[bytes]) -> None:
        """Fail a room its prefill has already failed, so its KV pages free on
        the next poll instead of after SGLANG_DISAGGREGATION_WAITING_TIMEOUT.
        A transfer the room already started keeps them until raiden settles it,
        as for any failure."""
        note = RoomFailed.from_frames(msg)
        room = note.room
        with self._state_lock:
            if room in self._aborted:
                return
            try:
                status = self.check_status(room)
            except KeyError:
                return  # already cleared
            if status == KVPoll.Success:
                return
            self._mark_aborted_locked(room)
            self._fail_room_locked(room, note.reason, preserve_reason=True)
            self.update_status(room, KVPoll.Failed)

    # ----- decode side -------------------------------------------------------

    def register_decode_request(
        self,
        room: int,
        local_blocks: List[int],
        local_swa_blocks: List[int],
        aux_index: Optional[int],
        bootstrap_infos: List[dict],
        *,
        local_state_rows: Sequence[int] = (),
    ) -> None:
        if self._pipeline is not None:
            self._pipeline.register_decode(
                room, local_blocks, local_swa_blocks, aux_index, bootstrap_infos
            )
            return
        if len(bootstrap_infos) != 1:
            if self._state_engine is not None:
                raise NotImplementedError(
                    "Raiden push resharding does not move GDN (linear-attention) "
                    f"state yet (got {len(bootstrap_infos)} prefill sources for "
                    "one decode rank); use equal attention TP on prefill and decode."
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
                    "SGLANG_RAIDEN_DYNAMIC_HOST_STAGING=0."
                )
            if not local_blocks:
                # Full prefix-cache hit: decode already holds every page,
                # nothing to receive. A hybrid model's sliding-window pages sit
                # inside that prefix, so its engine has nothing to receive
                # either -- mark both so is_recv_done can settle.
                with self._state_lock:
                    self._completed_recv.add(str(room))
                    if self._swa_engine is not None:
                        self._completed_recv.add(self._swa_req_id(room))
                return
            self._register_reshard_request(
                room, local_blocks, local_swa_blocks, aux_index, bootstrap_infos
            )
            self.update_status(room, KVPoll.WaitingForInput)
            return
        with self._state_lock:
            self._decode.local_blocks[room] = local_blocks
            self._decode.local_swa_blocks[room] = local_swa_blocks
            self._decode.local_state_rows[room] = list(local_state_rows)
            if aux_index is not None:
                self._decode.decode_aux_index[room] = aux_index
        payload = PullReq(
            room=room,
            decode_ip=self.local_ip,
            decode_port=self.rank_port,
            decode_pages=len(local_blocks),
            decode_swa_pages=len(local_swa_blocks),
        ).to_frames()
        for info in bootstrap_infos:
            endpoint = _tcp_endpoint(ip=info["rank_ip"], port=info["rank_port"])
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

    def _paired_state_read(self, key: str) -> Optional[str]:
        """The GDN state read issued alongside the read key, if any; it rides
        on the same reads as the sliding-window one."""
        if self._state_engine is None or "#" in key:
            return None
        return self._state_req_id(int(key))

    def _paired_reads(self, key: str) -> List[str]:
        return [
            rid
            for rid in (self._paired_swa_read(key), self._paired_state_read(key))
            if rid is not None
        ]

    def _pull_settled_locked(self, key: str) -> bool:
        """Whether every engine has reported the read ``key`` done or failed.

        Raiden holds a read's host staging until it completes or times out,
        whichever engine it is on, so the read's admission stays claimed until
        it and its paired sliding-window and state reads have settled.
        """
        return all(
            rid in self._completed_recv or rid in self._native_failed
            for rid in (key, *self._paired_reads(key))
        )

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

    def _release_pull_slots_locked(self) -> None:
        """Hand back the staging of reads that have settled."""
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
                for rid in (key, *self._paired_reads(key)):
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
        with self._state_lock:
            if room in self._decode.local_blocks:
                self._fail_room_locked(room, reason)
        logger.error("RaidenKVManager rank%d %s", self.attn_tp_rank, reason)

    def _start_read_or_fail(
        self,
        *,
        room: int,
        engine: Any,
        req_id: str,
        uuid: int,
        endpoint: str,
        remote: List[int],
        local: List[int],
        unissued: Tuple[str, ...] = (),
    ) -> bool:
        """Issue one admitted read; if the engine raises, fail the room.

        Runs inside the scheduler's poll, where an escaping exception would
        take down every request on this decode worker, or on the listener
        thread, where the admitted read would never settle. Raiden raises from
        ``start_read`` only before it registers the read, so it will never
        report ``req_id`` -- nor the reads in ``unissued``, which this failure
        keeps from being issued. They are settled here so the admission is
        released, and the room concludes once its other reads settle.
        """
        try:
            engine.start_read(req_id, uuid, endpoint, remote, local)
        except Exception as e:
            reason = f"Raiden receive failed: start_read {req_id} raised: {e}"
            logger.error("RaidenKVManager rank%d %s", self.attn_tp_rank, reason)
            with self._state_lock:
                self._native_failed.update((req_id, *unissued))
                self._fail_room_locked(room, reason)
                self._release_pull_slots_locked()
            return False
        logger.debug(
            "RaidenKVManager rank%d decode pull started | room=%s req_id=%s "
            "uuid=%d pages=%d from=%s",
            self.attn_tp_rank,
            room,
            req_id,
            uuid,
            len(remote),
            endpoint,
        )
        return True

    def _announcement_expired_locked(self, room: int, req_id: str) -> bool:
        """Fails ``room`` when its read ``req_id`` waited for a staging slot
        longer than the transfer deadline.

        The prefill registers a read for that long and then releases its
        pages, so a pull issued afterwards can only find nothing there.
        Called under ``_state_lock``.
        """
        announced = self._decode.announced_at.get(req_id)
        if announced is None:
            return False
        waited = self._monotonic() - announced
        if waited < self._timeout_s:
            return False
        self._fail_room_locked(
            room,
            f"Request {room} waited {waited:.0f}s for a host staging slot, "
            f"past the {self._timeout_s:.0f}s transfer deadline after which "
            "the prefill releases its KV",
        )
        # The read will not ask again, so it must not hold the head of the
        # line for the other rooms.
        if self._pulling is not None:
            self._pulling.forget(req_id)
        return True

    def last_announcement_age(self, room: int) -> Optional[float]:
        """Time (seconds) since the room's last chunk or closing announcement,
        or None if it has had none yet."""
        with self._state_lock:
            announced = self._decode.last_announced_at.get(room)
        if announced is None:
            return None
        return self._monotonic() - announced

    def maybe_start_pull(self, room: int) -> bool:
        """Issue whatever of this room's pull is announced and affordable.

        Returns True once at least one read is in flight for the room. A False
        return is a retry: the caller polls again on the next scheduler pass.
        """
        if self._pipeline is not None:
            return self._pipeline.maybe_start_pull(room)
        if self.chunked_transfer:
            return self._start_chunked_pull(room)
        return self._start_whole_pull(room)

    def _start_chunked_pull(self, room: int) -> bool:
        """Issue the next announced chunk once the pool can seat it.

        A request keeps at most one chunk read in flight, so it holds the
        staging of one chunk at a time; the next chunk is issued when the
        previous read has settled. Chunks are taken in prefill order.
        """
        engine = self._ensure_engine()
        if self._state_engine is not None:
            with self._state_lock:
                holds_nothing = (
                    room in self._decode.final_seen
                    and room in self._decode.local_blocks
                    and not self._decode.local_blocks[room]
                )
            if holds_nothing:
                return self._start_state_only_pull(room)
        while True:
            with self._state_lock:
                queue = self._decode.chunk_queue.get(room)
                reads = self._decode.chunk_reads.get(room, [])
                started = bool(reads)
                if not queue or room not in self._decode.local_blocks:
                    return started
                if any(rid in self._pulling for rid in reads):
                    return started
                if room in self._aborted or self._is_failed_locked(room):
                    # The room failed; its pages are no longer this read's.
                    return started
                local = self._decode.local_blocks[room]
                total = self._decode.remote_total.get(room, 0)
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
                    self._pulling.forget(req_id)
                    continue
                full_pages = len({local_block for _, local_block in pairs})
                if full_pages > self._full_engine_max_blocks:
                    oversized = (
                        full_pages,
                        self._full_engine_max_blocks,
                        "full-attention",
                        "SGLANG_RAIDEN_MAX_BLOCKS",
                    )
                    break
                cost = self._staging_cost(full_pages, self._full_engine_max_blocks)
                swa_cost = 0
                state_cost = 0
                # The closing read also carries the sliding-window pages and
                # the GDN state row, each on the engine that stages them.
                closing = req_id == str(room)
                if closing and self._swa_engine is not None:
                    remote_swa, local_swa = self._pair_blocks(
                        list(self._decode.remote_swa_blocks.get(room, [])),
                        list(self._decode.local_swa_blocks.get(room, [])),
                    )
                    swa_endpoint = self._decode.remote_swa_endpoint.get(room, "")
                    swa_pages = len(set(local_swa))
                    if swa_pages > self._swa_engine_max_blocks:
                        oversized = (
                            swa_pages,
                            self._swa_engine_max_blocks,
                            "SWA",
                            "SGLANG_RAIDEN_SWA_MAX_BLOCKS",
                        )
                        break
                    swa_cost = self._staging_cost(
                        swa_pages, self._swa_engine_max_blocks
                    )
                if closing and self._state_engine is not None:
                    remote_state, local_state, state_endpoint = self._state_read_locked(
                        room
                    )
                    state_cost = self._state_engine_max_blocks
                if self._announcement_expired_locked(room, req_id):
                    return started
                if not self._pulling.admit(
                    req_id, cost, swa_cost, state_pages=state_cost
                ):
                    return started
                # Status moves with the admission, under the same lock an
                # abort takes: a Transferring write can never follow the
                # abort's Failed.
                self.update_status(room, KVPoll.Transferring)
                queue.pop(0)
                remote_slice = [blk for blk, _ in pairs]
                local_slice = [loc for _, loc in pairs]
                self._decode.chunk_reads.setdefault(room, []).append(req_id)
            swa_rid = (
                self._swa_req_id(room)
                if closing and self._swa_engine is not None
                else None
            )
            state_rid = (
                self._state_req_id(room)
                if closing and self._state_engine is not None
                else None
            )
            if not self._start_read_or_fail(
                room=room,
                engine=engine,
                req_id=req_id,
                uuid=uuid,
                endpoint=endpoint,
                remote=remote_slice,
                local=local_slice,
                unissued=_present_ids(swa_rid, state_rid),
            ):
                return False
            if swa_rid is not None and not self._start_read_or_fail(
                room=room,
                engine=self._swa_engine,
                req_id=swa_rid,
                uuid=room,
                endpoint=swa_endpoint,
                remote=remote_swa,
                local=local_swa,
                unissued=_present_ids(state_rid),
            ):
                return False
            if state_rid is not None and not self._start_read_or_fail(
                room=room,
                engine=self._state_engine,
                req_id=state_rid,
                uuid=room,
                endpoint=state_endpoint,
                remote=remote_state,
                local=local_state,
            ):
                return False
        self._reject_oversized_pull(room, *oversized)
        return False

    def _start_whole_pull(self, room: int) -> bool:
        """Issue one read covering the request, once the prefill has replied."""
        with self._state_lock:
            if (
                room not in self._decode.remote_blocks
                or room not in self._decode.remote_endpoint
                or (
                    self._swa_engine is not None
                    and room not in self._decode.remote_swa_blocks
                )
                or (
                    self._swa_engine is not None
                    and room not in self._decode.remote_swa_endpoint
                )
                or (
                    self._state_engine is not None
                    and room not in self._decode.remote_state_endpoint
                )
            ):
                return False
            local = list(self._decode.local_blocks.get(room, []))
            remote = list(self._decode.remote_blocks[room])
            endpoint = self._decode.remote_endpoint[room]
            local_swa = list(self._decode.local_swa_blocks.get(room, []))
            remote_swa = list(self._decode.remote_swa_blocks.get(room, []))
            swa_endpoint = self._decode.remote_swa_endpoint.get(room, "")
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
        if not local:
            # Full prefix hit: all pages reside in local radix cache.
            # Mark reception complete for both engines upon receiving BLOCKS metadata.
            with self._state_lock:
                if room in self._aborted:
                    return False
                self._completed_recv.add(str(room))
                if self._swa_engine is not None:
                    self._completed_recv.add(self._swa_req_id(room))
            if self._state_engine is not None:
                # GDN state cannot be rebuilt from cached KV.
                return self._start_state_only_pull(room)
            return True
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
        cost = self._staging_cost(full_pages, self._full_engine_max_blocks)
        swa_cost = 0
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
            swa_cost = self._staging_cost(swa_pages, self._swa_engine_max_blocks)
        state_cost = 0
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
            if room in self._aborted or self._announcement_expired_locked(
                room, str(room)
            ):
                return False
            if self._state_engine is not None:
                remote_state, local_state, state_endpoint = self._state_read_locked(
                    room
                )
                state_cost = self._state_engine_max_blocks
            if not self._pulling.admit(
                str(room), cost, swa_cost, state_pages=state_cost
            ):
                return False
            # Status moves with the admission, under the same lock an abort
            # takes: a Transferring write can never follow the abort's Failed.
            self.update_status(room, KVPoll.Transferring)
        swa_rid = self._swa_req_id(room) if self._swa_engine is not None else None
        state_rid = self._state_req_id(room) if self._state_engine is not None else None
        if not self._start_read_or_fail(
            room=room,
            engine=engine,
            req_id=str(room),
            uuid=room,
            endpoint=endpoint,
            remote=remote,
            local=local,
            unissued=_present_ids(swa_rid, state_rid),
        ):
            return False
        if swa_rid is not None and not self._start_read_or_fail(
            room=room,
            engine=self._swa_engine,
            req_id=swa_rid,
            uuid=room,
            endpoint=swa_endpoint,
            remote=remote_swa,
            local=local_swa,
            unissued=_present_ids(state_rid),
        ):
            return False
        if state_rid is not None and not self._start_read_or_fail(
            room=room,
            engine=self._state_engine,
            req_id=state_rid,
            uuid=room,
            endpoint=state_endpoint,
            remote=remote_state,
            local=local_state,
        ):
            return False
        return True

    def _state_read_locked(self, room: int) -> Tuple[List[int], List[int], str]:
        """Remote rows, local rows and endpoint of the room's state read."""
        remote = list(self._decode.remote_state_rows.get(room, []))
        local = list(self._decode.local_state_rows.get(room, []))
        expected = self._state_engine_max_blocks
        if len(remote) != expected or len(local) != expected:
            raise RuntimeError(
                f"Raiden state read of room {room} needs {expected} row(s) per side, "
                f"got remote={remote} local={local}"
            )
        return remote, local, self._decode.remote_state_endpoint.get(room, "")

    def _start_state_only_pull(self, room: int) -> bool:
        """Pull only the GDN state row of a room whose KV the decode already
        holds; the read is admitted under the room's id like a whole read."""
        state_rid = self._state_req_id(room)
        with self._state_lock:
            if (
                str(room) in self._pulling
                or state_rid in self._completed_recv
                or state_rid in self._native_failed
            ):
                return True
            if (
                room in self._aborted
                or self._is_failed_locked(room)
                or room not in self._decode.remote_state_endpoint
                or self._announcement_expired_locked(room, str(room))
            ):
                return False
            remote, local, endpoint = self._state_read_locked(room)
            if not self._pulling.admit(
                str(room), 0, state_pages=self._state_engine_max_blocks
            ):
                return False
            self.update_status(room, KVPoll.Transferring)
        return self._start_read_or_fail(
            room=room,
            engine=self._state_engine,
            req_id=state_rid,
            uuid=room,
            endpoint=endpoint,
            remote=remote,
            local=local,
        )

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
        local_swa_blocks: List[int],
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
        # A hybrid model's sliding-window pool is a second block-id space on a
        # second engine, so it needs its own destination pages and its own push
        # target; both ride here rather than in a second handshake.
        swa_push_endpoint = (
            self._swa_engine.transfer_address if self._swa_engine is not None else ""
        )
        ready_endpoints = [
            _tcp_endpoint(ip=info["rank_ip"], port=info["rank_port"])
            for info in bootstrap_infos
        ]
        with self._state_lock:
            self._decode.reshard_meta[room] = dict(
                local_blocks=list(local_blocks),
                local_swa_blocks=list(local_swa_blocks or []),
                source_ranks=list(source_ranks),
                src_attn_tp=int(src_attn_tp),
                aux_index=aux_index,
                ready_endpoints=ready_endpoints,
            )
            self._decode.reshard_sources[room] = {}
            self._decode.reshard_swa_sources[room] = {}
            if aux_index is not None:
                self._decode.decode_aux_index[room] = aux_index
        payload = ReshardReq(
            room=room,
            decode_ip=self.local_ip,
            decode_port=self.rank_port,
            push_endpoint=push_endpoint,
            decode_blocks=list(local_blocks),
            dst_attn_tp=self.attn_tp_size,
            decode_swa_blocks=list(local_swa_blocks or []),
            swa_push_endpoint=swa_push_endpoint,
        ).to_frames()
        for info in bootstrap_infos:
            endpoint = _tcp_endpoint(ip=info["rank_ip"], port=info["rank_port"])
            self._send_multipart_locked(endpoint, payload)

    def _on_reshard_chunk(self, msg: List[bytes]) -> None:
        req = ReshardChunk.from_frames(msg)
        room = req.room
        index = req.index
        start = req.start
        total_pages = req.total_pages
        blocks = req.blocks
        src_rank = req.src_rank
        with self._state_lock:
            if (
                room not in self._decode.reshard_meta
                or room in self._aborted
                or room in self._decode.final_seen
                or room in self._decode.reshard_registered
            ):
                return
            per_source = self._decode.reshard_chunk_sources.setdefault(
                room, {}
            ).setdefault(index, {})
            per_source[src_rank] = (start, blocks)
            self._decode.remote_total[room] = total_pages
            # Sources are keyed by the sender's raiden engine id; its
            # attention-TP rank is the id within its replica.
            src_attn_tp = self._decode.reshard_meta[room]["src_attn_tp"]
            have = {nid % src_attn_tp for nid in per_source}
            if have < set(self._decode.reshard_meta[room]["source_ranks"]):
                return
            sources_map = self._decode.reshard_chunk_sources[room].pop(index)
            self._decode.reshard_chunk_registering.setdefault(room, set()).add(index)
        accepted_end = None
        try:
            accepted_end = self._register_reshard_chunk(
                room, index, total_pages, sources_map
            )
        finally:
            with self._state_lock:
                registering = self._decode.reshard_chunk_registering.get(room)
                if registering is not None:
                    registering.discard(index)
                    if not registering:
                        self._decode.reshard_chunk_registering.pop(room, None)
                # A span is covered only once its chunk plan is published,
                # so the closing transfer keeps every span the sources still
                # owe -- including one that lost its race mid-range.
                if accepted_end is not None and room in self._decode.reshard_meta:
                    self._decode.reshard_covered.setdefault(room, []).append(
                        accepted_end
                    )
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

        from sglang_tpu.disaggregation.raiden.reshard_schedule import (
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
                if room not in self._decode.reshard_meta:
                    return None
                self._failed.add(str(room))
                self.record_failure(room, reason)
            return None
        start, length = spans.pop()
        with self._state_lock:
            meta = self._decode.reshard_meta[room]
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
        # Independent decoders share a prefill. Use positive IDs that
        # fit the reshard RPC's signed int64 field.
        uuid = secrets.randbelow((1 << 63) - 1) + 1
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
        payload = ReshardChunkReady(room=room, index=index, uuid=uuid).to_frames()
        if not self._publish_reshard_plan(
            room, req_id, uuid, req, payload, ready_endpoints
        ):
            return None
        return (start, start + length)

    def _on_reshard_blocks(self, msg: List[bytes]) -> None:
        req = ReshardBlocks.from_frames(msg)
        room = req.room
        prefill_blocks = req.prefill_blocks
        src_rank = req.src_rank
        aux_blob = req.aux_blob
        prefill_swa_blocks = req.prefill_swa_blocks
        ready = False
        aux_index = None
        with self._state_lock:
            if (
                room not in self._decode.reshard_meta
                or room in self._aborted
                or room in self._decode.final_seen
                or room in self._decode.reshard_registered
            ):
                return
            self._decode.reshard_sources[room][src_rank] = prefill_blocks
            self._decode.reshard_swa_sources.setdefault(room, {})[src_rank] = (
                prefill_swa_blocks
            )
            aux_index = self._decode.reshard_meta[room].get("aux_index")
            src_attn_tp = self._decode.reshard_meta[room]["src_attn_tp"]
            have = {nid % src_attn_tp for nid in self._decode.reshard_sources[room]}
            need = set(self._decode.reshard_meta[room]["source_ranks"])
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
                room not in self._decode.reshard_meta
                or str(room) in self._failed
                or room in self._decode.reshard_registered
                or self._decode.reshard_chunk_registering.get(room)
            ):
                return
            meta = self._decode.reshard_meta[room]
            src_attn_tp = meta["src_attn_tp"]
            have = {nid % src_attn_tp for nid in self._decode.reshard_sources[room]}
            if have < set(meta["source_ranks"]):
                return
            partial = self._decode.reshard_chunk_sources.pop(room, None)
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
                if room not in self._decode.reshard_meta:
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
            if room in self._decode.reshard_meta and room not in self._aborted:
                if record_read:
                    self._decode.chunk_reads.setdefault(room, []).append(req_id)
                self._reshard_plans.setdefault(room, {})[req_id] = uuid
                # Sent under the lock: after a clear wins it, no push can be
                # requested for the pages the room gave back.
                for endpoint in ready_endpoints:
                    self._send_multipart_locked(endpoint, ready_payload)
                return True
        self._unregister_plan(uuid)
        return False

    def _register_reshard_swa_receiver(
        self,
        room: int,
        src_attn_tp: int,
        local_swa_blocks: List[int],
        swa_sources_map: Dict[int, List[int]],
    ) -> bool:
        """Arm the sliding-window receive of a resharded room. True to continue.

        Non-hybrid models have no second engine and skip this. The window is
        whole -- it is settled only when the prefill ends, so it never chunks
        and always travels with the closing push -- which is why this plan is
        built once from the full SWA page list rather than a complement.

        Pages pair by token order: both sides derive them from ``StateType.SWA``
        over the same window, so the lists must be the same length. They can be
        shorter on the decode after a prefix hit, in which case the tail pairs,
        matching what the non-chunked full path does.
        """
        if self._swa_engine is None:
            return True
        if not local_swa_blocks:
            # Let is_recv_done settle -- but this side has a window engine, so a
            # room with no window pages to place means the two sides disagree
            # about the model being hybrid. See the matching note on the push.
            logger.warning(
                "RaidenKVManager rank%d raiden reshard SWA receive SKIPPED "
                "room=%s -- no local window pages, the window will not arrive",
                self.attn_tp_rank,
                room,
            )
            with self._state_lock:
                self._completed_recv.add(self._swa_req_id(room))
            return True

        from tpu_sync.rpc import raiden_service_pb2 as pb2

        from sglang_tpu.disaggregation.raiden.reshard_schedule import (
            build_receiver_request,
        )

        n = len(local_swa_blocks)
        sources: List[Tuple[int, List[int]]] = []
        for p_rank in sorted(swa_sources_map):
            pages = swa_sources_map[p_rank]
            if len(pages) < n:
                reason = (
                    f"Raiden reshard SWA receive: source rank {p_rank} sent "
                    f"{len(pages)} sliding-window pages, decode needs {n}"
                )
                logger.error("RaidenKVManager rank%d %s", self.attn_tp_rank, reason)
                with self._state_lock:
                    self._failed.add(str(room))
                    self.record_failure(room, reason)
                return False
            sources.append((p_rank, pages[-n:]))

        geom = self._make_geometry(src_attn_tp, self.attn_tp_size, swa=True)
        req = build_receiver_request(
            pb2,
            uuid=room,
            req_id=self._swa_req_id(room),
            peer=self._swa_engine.transfer_address,
            decode_rank=self.attn_tp_rank,
            geom=geom,
            sources=sources,
            dst_block_ids=local_swa_blocks,
            live_tokens=[self.kv_args.page_size] * n,
        )
        try:
            self._swa_engine.register_active_plan(room, req, is_sender=False)
        except Exception as e:
            reason = f"Raiden reshard SWA receive: cannot stage plan {room}: {e}"
            logger.error("RaidenKVManager rank%d %s", self.attn_tp_rank, reason)
            with self._state_lock:
                if room not in self._decode.reshard_meta:
                    return False
                self._failed.add(str(room))
                self.record_failure(room, reason)
            return False
        with self._state_lock:
            if room not in self._decode.reshard_meta or room in self._aborted:
                # Cleared while registering; drop the plan again so the pages
                # cannot be written under their next owner.
                self._unregister_plan(room)
                return False
            self._reshard_plans.setdefault(room, {})[self._swa_req_id(room)] = room
        # INFO for the same reason as the push: without it a skipped window and
        # a transferred one produce identical logs.
        logger.info(
            "raiden reshard SWA receive room=%s decode_rank=%d sources=%d "
            "pages=%d layers=%d",
            room,
            self.attn_tp_rank,
            len(sources),
            n,
            len(self._swa_kv_caches),
        )
        return True

    def _register_receiver_and_signal(self, room: int) -> None:
        from tpu_sync.rpc import raiden_service_pb2 as pb2

        from sglang_tpu.disaggregation.raiden.reshard_schedule import (
            build_receiver_request,
        )

        with self._state_lock:
            if room in self._decode.reshard_registered:
                return
            self._decode.reshard_registered.add(room)
            meta = self._decode.reshard_meta[room]
            local_blocks = list(meta["local_blocks"])
            local_swa_blocks = list(meta.get("local_swa_blocks") or [])
            src_attn_tp = meta["src_attn_tp"]
            ready_endpoints = list(meta["ready_endpoints"])
            sources_map = dict(self._decode.reshard_sources[room])
            swa_sources_map = dict(self._decode.reshard_swa_sources.get(room, {}))

        # Arm the sliding-window receive before the READY goes out: the READY is
        # what tells every source to push, and its closing push carries the SWA
        # pages, so the second engine's plan has to be registered by then. A
        # failure here fails the room, so no READY is sent either way.
        if not self._register_reshard_swa_receiver(
            room, src_attn_tp, local_swa_blocks, swa_sources_map
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
                covered = list(self._decode.reshard_covered.get(room, []))
            missing = [pg for pg in self._uncovered_pages(total, covered) if pg >= base]
            with self._state_lock:
                if room not in self._decode.reshard_meta:
                    return  # cleared while the complement was computed
                # The closing read is owed from the moment the room can look
                # complete: it is recorded before _final_seen so completion
                # can never be inferred while its registration is pending.
                if missing:
                    self._decode.chunk_reads.setdefault(room, []).append(str(room))
                self._decode.final_seen.add(room)
            if not missing:
                for endpoint in ready_endpoints:
                    payload = ReshardReady(room=room).to_frames()
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
        ready_payload = ReshardReady(
            room=room, pages=missing if self.chunked_transfer else []
        ).to_frames()
        self._publish_reshard_plan(
            room,
            str(room),
            room,
            req,
            ready_payload,
            ready_endpoints,
            record_read=not self.chunked_transfer,
        )


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
        # Bounds the wait before send_metadata, see _check_handshake_timeout.
        self._handshake_start = time.time()
        self._handshake_timeout = envs.SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT.get()

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
            local_state_rows=self.kv_mgr._state_rows_from_state_indices(state_indices),
        )
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)
        # The waiting timeout counts from here: the request is now known to
        # prefill and nothing else bounds how long its reply can take.
        self.init_time = time.time()

    def _check_waiting_timeout(self) -> Optional[KVPoll]:
        """Time out a room only after its prefill has gone quiet.

        Each chunk or closing announcement restarts the clock. Until the first
        announcement arrives, the clock runs from send_metadata, so a prefill
        that never responds is still caught. Resharded and pipeline rooms
        record no announcements, so their clock always runs from send_metadata.
        """
        if self.init_time is None:
            return None
        age = self.kv_mgr.last_announcement_age(self.bootstrap_room)
        if age is None:
            return super()._check_waiting_timeout()
        if age < self.kv_mgr.waiting_timeout:
            return None
        # send_metadata predates every announcement, so the base check fails
        # the room here.
        result = super()._check_waiting_timeout()
        if result is not None:
            # Replace the base reason, which counts from send_metadata.
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                f"Request {self.bootstrap_room} timed out after {age:.1f}s without "
                f"a new KV announcement ({time.time() - self.init_time:.1f}s since "
                "send_metadata, SGLANG_DISAGGREGATION_WAITING_TIMEOUT="
                f"{self.kv_mgr.waiting_timeout:.0f}s)",
            )
        return result

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
        reason = self._check_handshake_timeout(status)
        if reason is not None:
            return self._conclude_failed(reason=reason)
        if self.kv_mgr.maybe_start_pull(self.bootstrap_room) and not self._started:
            # The manager turned the room Transferring inside the admission.
            self._started = True
            return KVPoll.Transferring
        return self.kv_mgr.check_status(self.bootstrap_room)

    def _check_handshake_timeout(self, status: KVPoll) -> Optional[str]:
        """Failure reason once the room has waited too long to ask the prefill
        for its KV: in Bootstrapping for the handshake, or in WaitingForInput
        for KV pages on this decode.

        The prefill fails a room whose decode has not asked within the
        bootstrap timeout, so past it the room can no longer be served, and
        preallocating it would only hold KV pages for a pull the prefill
        refuses.
        """
        if self.init_time is not None or status not in (
            KVPoll.Bootstrapping,
            KVPoll.WaitingForInput,
        ):
            return None
        elapsed = time.time() - self._handshake_start
        if elapsed < self._handshake_timeout:
            return None
        if status == KVPoll.Bootstrapping:
            stage = (
                "KVPoll.Bootstrapping before its KV metadata was sent to the prefill"
            )
        else:
            stage = (
                "KVPoll.WaitingForInput for KV pages on the decode. The prefill "
                "gives up on a request the decode has not asked for by then"
            )
        return (
            f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in "
            f"{stage}. Raise SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT on both "
            "the prefill and the decode to wait longer."
        )

    def abort(self):
        """Explicit local abort: same terminal path as any failure, so active
        transfers keep the room's pages until they settle. A room that already
        failed keeps its reason."""
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
        )

    def _conclude_failed(
        self,
        notify: bool = True,
        reason: Optional[str] = None,
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
                self.kv_mgr._fail_room_unless_failed_locked(self.bootstrap_room, reason)
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
        raise KVTransferError(self.bootstrap_room, reason)

    def clear(self) -> None:
        super().clear()
        self.kv_mgr.clear_room(self.bootstrap_room)
