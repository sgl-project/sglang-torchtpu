"""Prefill-side half of the raiden disaggregation backend.

Everything the prefill worker does: answering a decode worker's pull request,
announcing page chunks as they are produced, replying with block-ids, and the
strided push path used when decode TP is smaller than prefill TP.

``RaidenSenderMixin`` is mixed into ``RaidenKVManager`` in ``conn.py``; it is
not instantiated on its own and deliberately defines no ``__init__``.
"""

from __future__ import annotations

import errno
import secrets
import threading
import time
from typing import TYPE_CHECKING, Any, List, Optional, Set

import numpy as np
import numpy.typing as npt
import zmq

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVSender, KVTransferError
from sglang_tpu.disaggregation.raiden.chunk_plan import (
    chunk_pull_window,
    whole_pull_registration,
)
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
    decode_text,
    read_aux_blob,
)

if TYPE_CHECKING:
    from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager

# Send errors whose cause belongs to one peer, so they cost that peer's room
# rather than the engine. An errno not listed here is assumed engine-wide
# (ETERM, ENOMEM, ENOTSOCK, ...) and is re-raised so the scheduler still dies.
ROOM_SCOPED_SEND_ERRNOS = frozenset(
    {
        zmq.EAGAIN,  # the peer stopped draining, or SNDTIMEO expired
        errno.EHOSTUNREACH,  # the peer is unreachable
        errno.EINVAL,  # a malformed endpoint from the decode's own PULL_REQ
        errno.EINTR,  # a signal interrupted the send
    }
)


def _first_repeated_block(
    accumulated: List[int], listed: Set[int], block_ids: List[int]
) -> Optional[tuple[int, int, int]]:
    """The first page of ``block_ids`` already in ``accumulated`` (whose pages
    are ``listed``) or repeated within ``block_ids``, with both positions in
    the room's send order."""
    first_in_send = {}
    for pos, block in enumerate(block_ids, len(accumulated)):
        if block in listed:
            return block, accumulated.index(block), pos
        first = first_in_send.setdefault(block, pos)
        if first != pos:
            return block, first, pos
    return None


class RaidenSenderMixin:
    """Prefill-side methods of ``RaidenKVManager``."""

    def _on_pull_req(self, msg: List[bytes]) -> None:
        req = PullReq.from_frames(msg)
        room = req.room
        decode_ip = req.decode_ip
        decode_port = req.decode_port
        decode_pages = req.decode_pages
        decode_swa_pages = req.decode_swa_pages
        with self._state_lock:
            failed_reason = self._tombstone_reason_locked(room)
            if failed_reason is None:
                self._prefill.decode_addr[room] = (decode_ip, decode_port)
                self._prefill.decode_pages[room] = decode_pages
                self._prefill.decode_swa_pages[room] = decode_swa_pages
                held = self._prefill.held_chunks.pop(room, [])
                total_pages = self._prefill.send_total.get(room, 0)
        if failed_reason is not None:
            self._reply_room_failed(
                room=room,
                endpoint=_tcp_endpoint(ip=decode_ip, port=decode_port),
                reason=failed_reason,
            )
            return
        for start, block_ids in held:
            self._announce_chunk(room, start, block_ids, total_pages)
        self._maybe_reply(room)

    def _on_abort(self, msg: List[bytes]) -> None:
        """Fail a room that decode abandoned through the common ABORT message.

        The room's send state is dropped and later send calls for it are
        ignored, so a prefill still computing the request cannot register a
        read nobody will pull. A room that already succeeded is left
        untouched; a room this rank does not know yet is tombstoned, so an
        abort that outruns the pull request (they travel on different
        sockets) or the sender's creation still wins.
        """
        try:
            _, room_frame, *_ = msg
            room = int(decode_text(room_frame, "room"))
        except (IndexError, ValueError) as error:
            logger.error("RaidenKVManager received malformed ABORT: %s", error)
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
            self._fail_room_unless_failed_locked(
                room, "Aborted by decode-side abort notification."
            )
            self.update_status(room, KVPoll.Failed)
        self._drop_room_state(room)

    def _tombstone_reason_locked(self, room: int) -> Optional[str]:
        """Why a decode asking for ``room`` cannot be served, or None if the
        room is not tombstoned. The specific cause is in this rank's log."""
        failed_at = self._aborted.get(room)
        if failed_at is None:
            return None
        return (
            f"Prefill rank {self.attn_tp_rank} had already failed room {room} "
            f"{time.time() - failed_at:.1f}s before the decode asked for its KV. "
            "The prefill log has the cause"
        )

    def _reply_room_failed(self, *, room: int, endpoint: str, reason: str) -> None:
        """Tell the decode at ``endpoint`` that this rank failed ``room``.

        Without it the decode holds the room's KV pages until its own waiting
        timeout. An undeliverable notice leaves the decode to that timeout.
        """
        try:
            self._send_multipart_locked(
                endpoint, RoomFailed(room=room, reason=reason).to_frames()
            )
        except zmq.ZMQError as exc:
            logger.warning(
                "RaidenKVManager rank%d room %s: failure notice to %s not "
                "delivered (errno %s): %s",
                self.attn_tp_rank,
                room,
                endpoint,
                exc.errno,
                exc,
            )

    def notify_decode_room_failed(self, room: int) -> None:
        """Tell every decode that has asked this rank for ``room`` that the
        room failed here.

        A decode that already sent its last request for the room has nothing
        left to ask, so without this it learns of the failure only from its
        own waiting timeout. Endpoints are read before the send, which runs
        outside ``_state_lock``.
        """
        endpoints = []
        if self._pipeline is not None:
            with self._pipeline.lock:
                endpoints.extend(self._pipeline.replies.get(room, ()))
        with self._state_lock:
            addr = self._prefill.decode_addr.get(room)
            req = self._prefill.reshard_req.get(room)
        if addr is not None:
            endpoints.append(_tcp_endpoint(ip=addr[0], port=addr[1]))
        if req is not None:
            ip, port = req["reply_addr"]
            endpoints.append(_tcp_endpoint(ip=ip, port=port))
        if not endpoints:
            return
        with self.failure_lock:
            reason = self.failure_records.get(room)
        if reason is None:
            reason = (
                f"Prefill rank {self.attn_tp_rank} failed room {room}. "
                "The prefill log has the cause"
            )
        for endpoint in dict.fromkeys(endpoints):
            self._reply_room_failed(room=room, endpoint=endpoint, reason=reason)

    def append_send_blocks(
        self,
        room: int,
        block_ids: List[int],
        swa_block_ids: Optional[List[int]] = None,
        state_block_ids: Optional[List[int]] = None,
    ) -> Optional[int]:
        """Records a chunk's pages and returns its start index in the request.

        None means the room failed; the pages are not recorded and the caller
        must not offer them. A page recorded twice was freed and handed out
        again under a pending transfer: the room fails, naming both positions.
        """
        with self._state_lock:
            if room in self._aborted:
                return None
            accumulated = self._prefill.send_blocks.setdefault(room, [])
            listed = self._prefill.send_block_set.setdefault(room, set())
            start = len(accumulated)
            repeated = _first_repeated_block(accumulated, listed, block_ids)
            if repeated is None:
                accumulated.extend(block_ids)
                listed.update(block_ids)
                if swa_block_ids is not None:
                    self._prefill.send_swa_blocks[room] = list(swa_block_ids)
                if state_block_ids is not None:
                    self._prefill.send_state_rows[room] = list(state_block_ids)
                return start
            block, first, second = repeated
            reason = (
                f"KV page {block} is listed twice for one request, at send "
                f"positions {first} and {second} (this send starts at {start}, "
                f"{len(block_ids)} pages)"
            )
            self._mark_aborted_locked(room)
            self._fail_room_locked(room, reason)
            self.update_status(room, KVPoll.Failed)
            decode_addr = self._prefill.decode_addr.get(room) or (
                self._prefill.reshard_req.get(room, {}).get("reply_addr")
            )
        logger.error(
            "RaidenKVManager rank%d room %s: %s, failing the room",
            self.attn_tp_rank,
            room,
            reason,
        )
        if decode_addr is not None:
            self._reply_room_failed(
                room=room,
                endpoint=_tcp_endpoint(ip=decode_addr[0], port=decode_addr[1]),
                reason=reason,
            )
        self._drop_room_state(room)
        return None

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
            self._prefill.send_total[room] = total_pages
            p_start, p_blocks = self._prefill.pending_offer.get(room, (start, []))
            p_blocks = p_blocks + list(block_ids)
            if len(p_blocks) < self._prefill.announce_pages:
                self._prefill.pending_offer[room] = (p_start, p_blocks)
                return
            self._prefill.pending_offer.pop(room, None)
            start, block_ids = p_start, p_blocks
            if room in self._prefill.decode_addr:
                resharded = False
            elif room in self._prefill.reshard_req:
                resharded = True
            else:
                self._prefill.held_chunks.setdefault(room, []).append(
                    (start, list(block_ids))
                )
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
                    room in self._prefill.replied
                    or room in self._aborted
                    or room not in self._prefill.decode_pages
                ):
                    # The closing reply is out (or the room is gone): these
                    # pages travel with the closing transfer instead. A late
                    # announcement must never follow the reply -- the sender
                    # counts the room done when the closing read completes.
                    return
                window = chunk_pull_window(
                    start, len(block_ids), total_pages, self._prefill.decode_pages[room]
                )
                if not window:
                    return
                start += window.chunk_offset
                block_ids = block_ids[
                    window.chunk_offset : window.chunk_offset + window.length
                ]
                index = self._prefill.chunk_seq.get(room, 0)
                self._prefill.chunk_seq[room] = index + 1
                # Independent prefills share a decoder's native UUID map.
                # Zero means no transfer; reshard RPCs require signed int64.
                uuid = secrets.randbelow((1 << 63) - 1) + 1
                addr = self._prefill.decode_addr[room]
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
            payload = Chunk(
                room=room,
                req_id=req_id,
                uuid=uuid,
                start=start,
                total_pages=total_pages,
                blocks=block_ids,
                endpoint=self.raiden_endpoint or "",
            ).to_frames()
            endpoint = _tcp_endpoint(ip=addr[0], port=addr[1])
            try:
                self._send_multipart_locked(endpoint, payload)
            except zmq.ZMQError as exc:
                # As for the BLOCKS reply in _maybe_reply: an announce this
                # rank cannot deliver fails its own room. Unguarded, the error
                # reaches the scheduler's catch-all and kills every rank.
                if exc.errno not in ROOM_SCOPED_SEND_ERRNOS:
                    raise
                logger.error(
                    "RaidenKVManager rank%d room %s: CHUNK announce to %s "
                    "failed (errno %s), failing the room: %s",
                    self.attn_tp_rank,
                    room,
                    endpoint,
                    exc.errno,
                    exc,
                )
                with self._state_lock:
                    if room not in self._aborted:
                        self._mark_aborted_locked(room)
                        self._fail_room_locked(
                            room, f"CHUNK announce to {endpoint} failed: {exc}"
                        )
                        self.update_status(room, KVPoll.Failed)

    def finalize_send(self, room: int, aux_index: Optional[int]) -> bool:
        """Announce the room's blocks to the decode. False if the room was aborted.

        The room turns Transferring under the same lock that an abort takes
        to fail it, so the two cannot interleave: an abort either sees the
        finalized room and fails it terminally, or wins and this returns
        False without touching the status.
        """
        if self._pipeline is not None:
            return self._pipeline.finalize_send(room, aux_index)
        self._ensure_engine()
        with self._state_lock:
            if room in self._aborted:
                return False
            if aux_index is not None:
                self._prefill.send_aux[room] = read_aux_blob(
                    self.kv_args.aux_data_ptrs,
                    self.kv_args.aux_item_lens,
                    aux_index,
                )
            self._prefill.reshard_finalized.add(room)
            is_reshard = room in self._prefill.reshard_req
            self.update_status(room, KVPoll.Transferring)
        # The request type is selected by the decode handshake, not by whether
        # reshard support is globally enabled. Symmetric TP uses PULL_REQ even
        # when resharding is available; asymmetric TP uses RESHARD_REQ.
        if is_reshard:
            self._maybe_reshard_reply(room)
        else:
            self._maybe_reply(room)
        return True

    def on_forward_launched(self, scheduler, mb_id: int, batch, result) -> None:
        if self._pipeline is not None:
            self._pipeline.final_chunks.launched(scheduler, mb_id, batch, result)

    def on_outputs_copied(self, mb_id: int, next_token_ids) -> None:
        if self._pipeline is not None:
            self._pipeline.final_chunks.tokens_copied(mb_id, next_token_ids)

    def _decode_asked_locked(self, room: int) -> bool:
        """Whether the decode asked to pull the room, cleared it to push, or
        requested its pipeline transfer."""
        return (
            room in self._prefill.decode_addr
            or room in self._prefill.reshard_ready
            or (self._pipeline is not None and room in self._pipeline.replies)
        )

    def _decode_requested_locked(self, room: int) -> bool:
        """Whether the decode has sent its first request for the room: a pull
        request, a reshard request, or a pipeline request. Unlike
        ``_decode_asked_locked`` this counts a reshard request, which arrives
        before the prefill has anything to announce."""
        return (
            room in self._prefill.decode_addr
            or room in self._prefill.reshard_req
            or (self._pipeline is not None and room in self._pipeline.replies)
        )

    def finish_bootstrap_if_decode_requested(self, room: int) -> bool:
        """Moves ``room`` from Bootstrapping to WaitingForInput once the decode
        has requested its KV, and reports whether it has.

        The decode sends that request only after reserving the request's KV
        pages, so a prefill admitted on it never parks KV the decode cannot
        take. The check and the update are one step under the lock, so a
        concurrent abort is never overwritten.
        """
        with self._state_lock:
            if room in self._aborted or not self._decode_requested_locked(room):
                return False
            if self.request_status.get(room) == KVPoll.Bootstrapping:
                self.update_status(room, KVPoll.WaitingForInput)
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
            if self._decode_asked_locked(room):
                return None
            if room in self._prefill.reshard_req:
                return self._prefill.reshard_replied_at.get(room)
            return since

    def fail_decode_silence(self, room: int, since: float, timeout: float) -> bool:
        """Fails ``room`` when the decode has owed it a message for ``timeout``.

        The deadline check and the failure are one step under the lock, so a
        pull or reshard message that lands concurrently either arrives first
        (the room stays live) or finds the tombstone and is answered with
        ROOM_FAILED -- the decode can never be told to pull KV whose room just
        failed.
        """
        with self._state_lock:
            if self._decode_asked_locked(room):
                return False
            if room in self._prefill.reshard_req:
                started = self._prefill.reshard_replied_at.get(room)
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
        the potentially blocking registration and ends when poll_stats
        reports the request id settled: raiden expires a read nobody pulls at
        its own deadline and reports it, so no second timer runs here. The
        one exception is an unread chunk lease of a room whose closing read
        has settled, which ends early (see _settle_sender_reads_locked). A
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
            return
        logger.debug(
            "RaidenKVManager rank%d prefill registered read | room=%s "
            "req_id=%s uuid=%d pages=%d",
            self.attn_tp_rank,
            room,
            req_id,
            uuid,
            len(block_ids),
        )

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
            closing_done = str(room) in self._completed_send and self._side_reads_done(
                room, self._completed_send
            )
            if closing_done:
                # The closing read covers every page the decode asked for
                # (whole_pull_registration), and the decode issues it only
                # after its earlier chunk reads settled and drops any CHUNK
                # that arrives after BLOCKS (final_seen). So an unread chunk
                # lease here will never be pulled, and the room's pages can be
                # released now rather than held to the native deadline.
                # Raiden cannot cancel the read; record it so _drain_engine
                # ignores its eventual expiry instead of failing the room.
                self._superseded_sender_reads.update(reads - settled)
                self._send_reads.pop(room, None)
                continue
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
        through poll_stats like any other, except an unread chunk read of a
        room whose closing read has settled: the decode will never pull it,
        so it stops counting then (see _settle_sender_reads_locked).
        """
        if self._pipeline is not None:
            return self._pipeline.active(room)
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
            if room in self._prefill.replied or room in self._aborted:
                return
            if (
                room not in self._prefill.reshard_finalized
                or room not in self._prefill.send_blocks
                or (
                    self._swa_engine is not None
                    and room not in self._prefill.send_swa_blocks
                )
                or (
                    self._state_engine is not None
                    and room not in self._prefill.send_state_rows
                )
                or room not in self._prefill.decode_addr
            ):
                return
            swa_blocks = list(self._prefill.send_swa_blocks.get(room, []))
            state_rows = list(self._prefill.send_state_rows.get(room, []))
            blocks = list(self._prefill.send_blocks[room])
            addr = self._prefill.decode_addr[room]
            aux_blob = self._prefill.send_aux.get(room, b"")
            decode_pages = self._prefill.decode_pages[room]
            decode_swa_pages = self._prefill.decode_swa_pages[room]
            self._prefill.replied.add(room)
        # Register only the suffix requested by the decode node to prevent
        # unread blocks from leaking leases until deadline timeout.
        read_blocks = whole_pull_registration(blocks, decode_pages)
        swa_read_blocks = whole_pull_registration(swa_blocks, decode_swa_pages)
        # Decode reads both engines concurrently or neither; suppress window
        # registrations on full hits to prevent orphaned leases.
        if not read_blocks:
            swa_read_blocks = []
        with self._announce_lock:
            # GDN state cannot be rebuilt from cached KV, so even a full prefix
            # hit moves it.
            if self._state_engine is not None:
                self._register_sender_read(
                    room,
                    self._state_engine,
                    self._state_req_id(room),
                    room,
                    state_rows,
                )
            if self._swa_engine is not None and swa_read_blocks:
                self._register_sender_read(
                    room,
                    self._swa_engine,
                    self._swa_req_id(room),
                    room,
                    swa_read_blocks,
                )
            if read_blocks:
                self._register_sender_read(
                    room, self._ensure_engine(), str(room), room, read_blocks
                )
            with self._state_lock:
                # The room can fail while its reads register; the decode is
                # gone and its own handlers drop frames for the room.
                if room in self._aborted:
                    return
            endpoint = _tcp_endpoint(ip=addr[0], port=addr[1])
            payload = Blocks(
                room=room,
                blocks=blocks,
                endpoint=self.raiden_endpoint or "",
                swa_blocks=swa_blocks,
                swa_endpoint=self.swa_raiden_endpoint or "",
                aux_blob=aux_blob,
                state_blocks=state_rows,
                state_endpoint=self.state_raiden_endpoint or "",
            ).to_frames()
            try:
                self._send_multipart_locked(endpoint, payload)
            except zmq.ZMQError as exc:
                # The decode is waiting on BLOCKS; fail the room now rather
                # than letting it sit until the endpoint deadline.
                with self._state_lock:
                    if room not in self._aborted:
                        self._mark_aborted_locked(room)
                        self._fail_room_locked(
                            room, f"BLOCKS reply to {endpoint} failed: {exc}"
                        )
                        self.update_status(room, KVPoll.Failed)
                return
        with self._state_lock:
            if room in self._aborted:
                return
            if not read_blocks:
                # Full prefix hit: no read follows, so BLOCKS on the wire
                # completes the send.
                self._completed_send.add(str(room))
            if self._swa_engine is not None and not swa_read_blocks:
                self._completed_send.add(self._swa_req_id(room))

    def _on_reshard_req(self, msg: List[bytes]) -> None:
        if self._state_engine is not None:
            raise NotImplementedError(
                "Raiden push resharding does not move GDN (linear-attention) "
                "state yet; use equal attention TP on prefill and decode."
            )
        req = ReshardReq.from_frames(msg)
        room = req.room
        decode_ip = req.decode_ip
        decode_port = req.decode_port
        push_endpoint = req.push_endpoint
        decode_blocks = req.decode_blocks
        dst_attn_tp = req.dst_attn_tp
        decode_swa_blocks = req.decode_swa_blocks
        swa_push_endpoint = req.swa_push_endpoint
        with self._state_lock:
            failed_reason = self._tombstone_reason_locked(room)
            if failed_reason is None:
                self._prefill.reshard_req[room] = dict(
                    reply_addr=(decode_ip, decode_port),
                    push_endpoint=push_endpoint,
                    decode_blocks=decode_blocks,
                    dst_attn_tp=dst_attn_tp,
                    decode_swa_blocks=decode_swa_blocks,
                    swa_push_endpoint=swa_push_endpoint,
                )
                held = self._prefill.held_chunks.pop(room, [])
                total_pages = self._prefill.send_total.get(room, 0)
        if failed_reason is not None:
            self._reply_room_failed(
                room=room,
                endpoint=_tcp_endpoint(ip=decode_ip, port=decode_port),
                reason=failed_reason,
            )
            return
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
                    room in self._prefill.reshard_replied
                    or room in self._aborted
                    or room not in self._prefill.reshard_req
                ):
                    # The closing message is out (or the room is gone): these
                    # pages travel with the closing transfer, whose span the
                    # decode decides from the chunks that did arrive.
                    return
                req = self._prefill.reshard_req[room]
                window = chunk_pull_window(
                    start, len(block_ids), total_pages, len(req["decode_blocks"])
                )
                if not window:
                    return
                start += window.chunk_offset
                block_ids = block_ids[
                    window.chunk_offset : window.chunk_offset + window.length
                ]
                index = self._prefill.chunk_seq.get(room, 0)
                self._prefill.chunk_seq[room] = index + 1
                self._prefill.reshard_chunks.setdefault(room, {})[index] = (
                    start,
                    block_ids,
                )
                addr = req["reply_addr"]
            payload = ReshardChunk(
                room=room,
                index=index,
                start=start,
                total_pages=total_pages,
                blocks=block_ids,
                src_rank=self._local_rank,
            ).to_frames()
            endpoint = _tcp_endpoint(ip=addr[0], port=addr[1])
            self._send_multipart_locked(endpoint, payload)

    def _on_reshard_chunk_ready(self, msg: List[bytes]) -> None:
        req = ReshardChunkReady.from_frames(msg)
        room = req.room
        index = req.index
        uuid = req.uuid
        with self._state_lock:
            if room in self._aborted:
                return
            chunk = self._prefill.reshard_chunks.get(room, {}).get(index)
            if chunk is None:
                logger.error(
                    "RaidenKVManager rank%d room %s asked to push unknown chunk %d",
                    self.attn_tp_rank,
                    room,
                    index,
                )
                return
            acked = self._prefill.reshard_chunk_acked.setdefault(room, set())
            if index in acked:
                return
            acked.add(index)
            self._reshard_pushes_pending[room] = (
                self._reshard_pushes_pending.get(room, 0) + 1
            )
            # The decode is making progress; time the wait for its next
            # message from here.
            self._prefill.reshard_replied_at[room] = time.time()
        self._push_executor.submit(self._do_reshard_push, room, index, uuid)

    def _maybe_reshard_reply(self, room: int) -> None:
        with self._state_lock:
            # Only reply once, and only after send() finished accumulating this
            # room's blocks (a RESHARD_REQ may arrive mid-accumulation).
            if room in self._prefill.reshard_replied:
                return
            if (
                room not in self._prefill.reshard_req
                or room not in self._prefill.reshard_finalized
            ):
                return
            addr = self._prefill.reshard_req[room]["reply_addr"]
            blocks = list(self._prefill.send_blocks.get(room, []))
            # The sliding-window pages are settled by the time this reply is
            # owed (it is sent once the send accumulation is finalized), so the
            # closing push can carry them; earlier chunks never do.
            swa_blocks = list(self._prefill.send_swa_blocks.get(room, []))
            aux_blob = self._prefill.send_aux.get(room, b"")
        with self._announce_lock:
            with self._state_lock:
                if room not in self._prefill.reshard_req:
                    return
                # Committed under the order lock: no announcement can follow.
                self._prefill.reshard_replied.add(room)
                self._prefill.reshard_replied_at[room] = time.time()
            endpoint = _tcp_endpoint(ip=addr[0], port=addr[1])
            payload = ReshardBlocks(
                room=room,
                prefill_blocks=blocks,
                src_rank=self._local_rank,
                aux_blob=aux_blob,
                prefill_swa_blocks=swa_blocks,
            ).to_frames()
            self._send_multipart_locked(endpoint, payload)

    def _on_reshard_ready(self, msg: List[bytes]) -> None:
        req = ReshardReady.from_frames(msg)
        room = req.room
        pages = req.pages
        with self._state_lock:
            if room in self._aborted:
                return
            self._prefill.reshard_ready.add(room)
            self._prefill.reshard_ready_pages[room] = pages
        self._maybe_reshard_push(room)

    def _maybe_reshard_push(self, room: int) -> None:
        with self._state_lock:
            if (
                room in self._aborted
                or room not in self._prefill.reshard_ready
                or room not in self._prefill.reshard_req
                or room not in self._prefill.send_blocks
                or room in self._prefill.reshard_pushed
            ):
                return
            self._prefill.reshard_pushed.add(room)
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
                        self._prefill.reshard_final_pushed.add(room)
                    if (
                        left <= 0
                        and room in self._prefill.reshard_final_pushed
                        and room in self._prefill.reshard_req
                    ):
                        self._completed_send.add(str(room))

    def _run_reshard_swa_push(
        self, room: int, req_meta: dict, swa_blocks: List[int]
    ) -> None:
        """Push a resharded room's sliding-window pages on the second engine.

        Mirrors the full-attention closing push: stage D2H, then one push per
        SWA layer so the receiver's ``blocks * layers`` barrier completes. The
        window is whole and unchunked, so this runs exactly once per room.

        Marks the window sent on any outcome the room cannot recover from --
        a raiden failure is reported through ``_failed``, and leaving
        ``swa:<room>`` absent instead would hang ``is_send_done`` forever.
        """
        if self._swa_engine is None:
            return
        swa_req_id = self._swa_req_id(room)
        decode_swa_blocks = list(req_meta.get("decode_swa_blocks") or [])
        swa_push_endpoint = req_meta.get("swa_push_endpoint") or ""
        n = len(decode_swa_blocks)
        if not n or not swa_blocks or not swa_push_endpoint:
            # None of these is reachable in a healthy run, so fail the room
            # rather than skip the window. This side has a window engine, so:
            # an empty peer endpoint means the decode has none, i.e. the two
            # sides disagree about the model being hybrid; empty pages on
            # either side mean a hybrid room carrying no window. The one
            # legitimate empty case -- a full prefix hit -- returns in
            # register_decode_request without ever sending a RESHARD_REQ, so it
            # cannot arrive here. Skipping would move only the full-attention KV
            # and still report success, which accuracy alone may not catch.
            reason = (
                f"Raiden reshard SWA push: hybrid room {room} has nothing to "
                f"send (decode_pages={n} local_pages={len(swa_blocks)} "
                f"peer={swa_push_endpoint!r})"
            )
            logger.error("RaidenKVManager rank%d %s", self.attn_tp_rank, reason)
            with self._state_lock:
                self._failed.add(str(room))
                # Settled even on failure: the room is reported through
                # _failed, and leaving swa:<room> absent would hang
                # is_send_done instead of surfacing the reason.
                self._completed_send.add(swa_req_id)
                self.record_failure(room, reason)
            return
        if len(swa_blocks) > n:
            # Decode kept fewer window pages (prefix hit); pair the tail, as the
            # non-chunked full path does.
            swa_blocks = swa_blocks[-n:]
        elif len(swa_blocks) < n:
            reason = (
                f"Raiden reshard SWA push: prefill has {len(swa_blocks)} "
                f"sliding-window pages, decode expects {n}"
            )
            logger.error("RaidenKVManager rank%d %s", self.attn_tp_rank, reason)
            with self._state_lock:
                self._failed.add(str(room))
                self._completed_send.add(swa_req_id)
                self.record_failure(room, reason)
            return

        from tpu_sync.rpc import raiden_service_pb2 as pb2

        from sglang_tpu.disaggregation.raiden.reshard_schedule import (
            build_sender_request,
        )

        geom = self._make_geometry(self.attn_tp_size, req_meta["dst_attn_tp"], swa=True)
        req = build_sender_request(
            pb2,
            uuid=room,
            req_id=swa_req_id,
            peer=swa_push_endpoint,
            prefill_rank=self.attn_tp_rank,
            decode_rank=self.attn_tp_rank // geom.ratio,
            geom=geom,
            src_block_ids=swa_blocks,
            dst_block_ids=decode_swa_blocks,
            live_tokens=[self.kv_args.page_size] * n,
        )
        engine = self._swa_engine
        fut = None
        try:
            engine.register_active_plan(room, req, is_sender=True)
            host_blocks = engine.plan_host_blocks(room, swa_blocks)
            fut = engine.d2h(swa_blocks, host_blocks, [1] * len(swa_blocks))
            deadline = self._monotonic() + self._timeout_s
            while fut is not None and not fut.is_ready():
                if self._monotonic() > deadline:
                    raise TimeoutError(
                        f"raiden reshard SWA D2H stage for room {room} timed out"
                    )
                time.sleep(0.0005)
            if fut is not None and not fut.ok():
                raise RuntimeError(
                    f"raiden reshard SWA D2H stage for room {room} failed: "
                    f"{fut.error_message()}"
                )
            with self._reshard_push_lock:
                for layer_idx in range(len(self._swa_kv_caches)):
                    engine.push_registered_plan(
                        room,
                        swa_push_endpoint,
                        swa_blocks,
                        decode_swa_blocks,
                        layer_idx,
                        parallelism=self._transfer_parallelism,
                    )
            # INFO, not debug: this is the only evidence the window actually
            # crossed. Everything else about a hybrid reshard looks identical
            # whether or not it did, and the integration test greps for it.
            logger.info(
                "raiden reshard SWA pushed room=%s prefill_rank=%d decode_rank=%d "
                "pages=%d layers=%d peer=%s",
                room,
                self.attn_tp_rank,
                self.attn_tp_rank // geom.ratio,
                n,
                len(self._swa_kv_caches),
                swa_push_endpoint,
            )
        except Exception as e:
            reason = f"Raiden reshard SWA push for room {room} failed: {e}"
            logger.error("RaidenKVManager rank%d %s", self.attn_tp_rank, reason)
            with self._state_lock:
                self._failed.add(str(room))
                self.record_failure(room, reason)
        finally:
            if fut is not None and not fut.is_ready():
                with self._state_lock:
                    self._stale_plans.append((room, room, fut))
                threading.Thread(
                    target=self._reap_stale_plan, args=(room, room, fut), daemon=True
                ).start()
            else:
                try:
                    engine.unregister_active_plan(room)
                except RuntimeError as e:
                    if "not registered" not in str(e):
                        raise
            with self._state_lock:
                self._completed_send.add(swa_req_id)

    def _run_reshard_push(
        self, room: int, chunk: Optional[int] = None, uuid: Optional[int] = None
    ) -> None:
        from tpu_sync.rpc import raiden_service_pb2 as pb2

        from sglang_tpu.disaggregation.raiden.reshard_schedule import (
            build_sender_request,
        )

        with self._state_lock:
            if room in self._aborted or room not in self._prefill.reshard_req:
                return  # the room failed or was cleared before the push ran
            req_meta = dict(self._prefill.reshard_req[room])
            if chunk is None:
                prefill_blocks = list(self._prefill.send_blocks[room])
                ready_pages = list(self._prefill.reshard_ready_pages.get(room, []))
                swa_blocks = list(self._prefill.send_swa_blocks.get(room, []))
                start = 0
            else:
                start, prefill_blocks = self._prefill.reshard_chunks[room].pop(chunk)
            total_pages = self._prefill.send_total.get(room, len(prefill_blocks))

        # The window rides on the closing push only, and has to go out even when
        # the closing full-attention push has nothing left to move: chunks can
        # already cover every full page, and the window is never chunked, so
        # returning early below would strand it and is_send_done would never
        # settle.
        if chunk is None:
            self._run_reshard_swa_push(room, req_meta, swa_blocks)

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
                        uuid,
                        push_endpoint,
                        prefill_blocks,
                        decode_blocks,
                        layer_idx,
                        parallelism=self._transfer_parallelism,
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


class RaidenKVSender(CommonKVSender):
    # A group-wide Failed is delivered only after every rank drained: raiden
    # cannot cancel a transfer, so a peer's pages must not be freed under it.
    group_failure_requires_drain = True
    # Whether the decode needs no ROOM_FAILED from this sender: one was sent,
    # or the decode released the room itself.
    _decode_told = False

    @property
    def pages_exposed_before_result(self) -> bool:
        # A transfer may read the pages before the result point, which then
        # keeps them out of the radix cache: stage groups and early close
        # register a pipeline stage's pages at the final chunk's launch.
        pipeline = self.kv_mgr._pipeline
        return pipeline is not None and (pipeline.stage_parts or pipeline.early_close)

    def __init__(
        self,
        mgr: RaidenKVManager,
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
        self._last_chunk = False
        self._started = False
        self._transfer_start_time: Optional[float] = None
        # Raiden's finish time for this room (perf_counter seconds), set on Success.
        self.transfer_finish_time: Optional[float] = None
        # The decode's first request for the room (see poll) ends Bootstrapping.
        # It is sent only after the decode reserved the request's KV pages, so
        # the prefill never computes KV that no decode can take yet.
        self.init_time = time.time()
        # An ABORT that outran this sender only tombstoned the room (_on_abort
        # saw no status). The status now exists, so _on_abort and this check
        # are ordered by _state_lock: one of them fails the room.
        if not self.kv_mgr.is_dummy_cp_rank:
            with self.kv_mgr._state_lock:
                if self.bootstrap_room in self.kv_mgr._aborted:
                    self._decode_told = True
                    self.kv_mgr._fail_room_locked(
                        self.bootstrap_room,
                        "Aborted by decode-side abort notification received "
                        "before the sender was created.",
                        preserve_reason=True,
                    )
                    self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)

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
        if self._transfer_start_time is None and (
            len(kv_indices) > 0 or state_indices is not None
        ):
            self._transfer_start_time = time.perf_counter()
        self._record_transfer_indices(kv_indices, state_indices)
        swa_blocks = (
            self.kv_mgr._swa_blocks_from_state_indices(state_indices)
            if state_indices is not None
            else None
        )
        state_rows = (
            self.kv_mgr._state_rows_from_state_indices(state_indices)
            if state_indices is not None
            else None
        )
        blocks = [int(x) for x in kv_indices]
        start = self.kv_mgr.append_send_blocks(
            self.bootstrap_room,
            blocks,
            swa_block_ids=swa_blocks,
            state_block_ids=state_rows,
        )
        if start is None:
            return  # aborted while this chunk was being computed
        if self.kv_mgr.chunked_transfer and not self._last_chunk:
            self.kv_mgr.offer_chunk(
                self.bootstrap_room, start, blocks, self.num_kv_indices
            )
        elif not self._last_chunk and self.kv_mgr._pipeline is not None:
            # A pipeline stage moves pages already final (a cached prefix,
            # finished chunks) as early parts.
            self.kv_mgr._pipeline.offer_part(self.bootstrap_room, start, blocks)
        if self._last_chunk and not self._started:
            self._started = True
            # finalize_send turns the room Transferring itself, under the
            # lock an abort takes to fail it; a status update here could
            # overwrite a Failed that landed in between.
            self.kv_mgr.finalize_send(self.bootstrap_room, self.aux_index)

    def _finalize_transfer_latency(self) -> None:
        if self._transfer_metric.transfer_latency_s is not None:
            return
        # Save it now: clear() runs before the scheduler stamps the stage.
        self.transfer_finish_time = self.kv_mgr.send_done_time(self.bootstrap_room)
        if self._transfer_start_time is not None:
            finish = self.transfer_finish_time
            if finish is None:
                finish = time.perf_counter()
            self._transfer_metric.transfer_latency_s = (
                finish - self._transfer_start_time
            )

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
        failed = self.kv_mgr.is_failed(self.bootstrap_room)
        if failed:
            # Told before the active-IO wait below: the decode need not wait
            # out its own timeout. A decode ABORT (or a group quiesce) drops
            # the room's failure marker with its state, so it is not answered.
            self._tell_decode_room_failed()
        if status == KVPoll.Failed or failed:
            if self.kv_mgr.sender_has_active_io(self.bootstrap_room):
                return KVPoll.Transferring
            self.conclude_state = KVPoll.Failed
            return KVPoll.Failed
        if self.kv_mgr.is_send_done(self.bootstrap_room):
            self._finalize_transfer_latency()
            self.conclude_state = KVPoll.Success
            return KVPoll.Success
        if status == KVPoll.Success:
            self._finalize_transfer_latency()
            self.conclude_state = status
            return status
        if (
            status == KVPoll.Bootstrapping
            and self.kv_mgr.finish_bootstrap_if_decode_requested(self.bootstrap_room)
        ):
            status = KVPoll.WaitingForInput
        # The decode owes the room a message until it asks to pull, or clears
        # a resharded room to push. A decode that dies before then would
        # otherwise leave the room (and its KV) waiting forever.
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
            self.kv_mgr._fail_room_unless_failed_locked(
                self.bootstrap_room,
                "Raiden transfer failed on another attention TP/CP rank.",
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
        Failed. A room that already failed keeps its reason.
        """
        with self.kv_mgr._state_lock:
            self.kv_mgr._mark_aborted_locked(self.bootstrap_room)
            self.kv_mgr._fail_room_unless_failed_locked(
                self.bootstrap_room, "Aborted by AbortReq."
            )
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
        # Before the drop, which forgets where the decode is.
        self._tell_decode_room_failed()
        self.kv_mgr._drop_room_state(self.bootstrap_room)
        self.conclude_state = (
            None
            if self.kv_mgr.sender_has_active_io(self.bootstrap_room)
            else KVPoll.Failed
        )

    def _tell_decode_room_failed(self) -> None:
        if self._decode_told:
            return
        self._decode_told = True
        self.kv_mgr.notify_decode_room_failed(self.bootstrap_room)

    def failure_exception(self):
        """Drop the room's state, then raise the recorded failure reason."""
        with self.kv_mgr.failure_lock:
            reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "Raiden send failed"
            )
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed
        self.clear()
        raise KVTransferError(self.bootstrap_room, reason)

    def clear(self) -> None:
        super().clear()
        self.kv_mgr.clear_room(self.bootstrap_room)
