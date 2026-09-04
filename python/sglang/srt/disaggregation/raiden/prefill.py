"""Prefill-side Raiden rendezvous and transfer protocol."""

from __future__ import annotations

import concurrent.futures
import logging
import os
import threading
import time

import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVSender
from sglang.srt.disaggregation.raiden.chunk_plan import chunk_pull_window
from sglang.srt.disaggregation.raiden.conn import RaidenKVManager, _tcp_endpoint
from sglang.srt.disaggregation.raiden.serde import (
    MessageTag,
    pack_blocks,
    pack_ints,
    read_aux_blob,
    unpack_ints,
)

logger = logging.getLogger(__name__)

CHUNK_ANNOUNCE_TOKENS = 8192


class RaidenPrefillKVManager(RaidenKVManager):
    def _message_handlers(self):
        return {
            MessageTag.PULL_REQ: self._on_pull_req,
            MessageTag.ABORT: self._on_abort,
            MessageTag.RESHARD_REQ: self._on_reshard_req,
            MessageTag.RESHARD_READY: self._on_reshard_ready,
            MessageTag.RESHARD_CHUNK_READY: self._on_reshard_chunk_ready,
        }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._send_reads: dict[int, set[str]] = {}
        self._announce_lock = threading.Lock()
        self._send_blocks: dict[int, list[int]] = {}
        self._send_sliding_window_attn_blocks: dict[int, list[int]] = {}
        self._send_aux: dict[int, bytes] = {}
        self._decode_addr: dict[int, tuple[str, int]] = {}
        self._replied: set[int] = set()
        self._held_chunks: dict[int, list[tuple[int, list[int]]]] = {}
        self._send_total: dict[int, int] = {}
        self._decode_pages: dict[int, int] = {}
        self._reshard_chunks: dict[int, dict[int, tuple[int, list[int]]]] = {}
        self._reshard_covered: dict[int, list[tuple[int, int]]] = {}
        self._reshard_ready_pages: dict[int, list[int]] = {}
        self._reshard_pushes_pending: dict[int, int] = {}
        self._reshard_final_pushed: set[int] = set()
        self._pending_offer: dict[int, tuple[int, list[int]]] = {}
        self._announce_pages = max(
            1, -(-CHUNK_ANNOUNCE_TOKENS // int(self.kv_args.page_size))
        )
        self._chunk_seq: dict[int, int] = {}
        self._reshard_req: dict[int, dict] = {}
        self._reshard_finalized: set[int] = set()
        self._reshard_chunk_acked: dict[int, set[int]] = {}
        self._reshard_replied: set[int] = set()
        self._reshard_ready: set[int] = set()
        self._reshard_replied_at: dict[int, float] = {}
        self._reshard_pushed: set[int] = set()
        self._push_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, int(os.environ.get("SGLANG_RAIDEN_PUSH_WORKERS", "1"))),
            thread_name_prefix="raiden-reshard-push",
        )
        self._reshard_push_lock = threading.Lock()
        self._start_listener()

    def _settle_locked(self) -> list[int]:
        self._settle_sender_reads_locked()
        return []

    def _drop_role_state_locked(self, room: int, belongs) -> tuple[set[str], list[int]]:
        for state in (
            self._held_chunks,
            self._send_total,
            self._decode_pages,
            self._chunk_seq,
            self._reshard_chunks,
            self._pending_offer,
            self._reshard_covered,
            self._reshard_ready_pages,
            self._reshard_chunk_acked,
            self._send_blocks,
            self._send_sliding_window_attn_blocks,
            self._send_aux,
            self._decode_addr,
            self._reshard_req,
            self._reshard_replied_at,
        ):
            state.pop(room, None)
        for state in (
            self._replied,
            self._reshard_final_pushed,
            self._reshard_finalized,
            self._reshard_replied,
            self._reshard_ready,
            self._reshard_pushed,
        ):
            state.discard(room)
        return set(), []

    @staticmethod
    def _tail_after(blocks: list[int], announced: int) -> tuple[int, list[int]]:
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

    def _on_abort(self, msg: list[bytes]) -> None:
        """Fail a prefill room abandoned by its decode peer."""
        try:
            room = int(msg[1].decode("ascii"))
        except (IndexError, ValueError) as exc:
            logger.debug("Raiden prefill ignoring malformed abort: %s", exc)
            return
        failed = self.fail_room(
            room,
            "Aborted by decode-side abort notification.",
            preserve_succeeded=True,
        )
        if failed:
            self.drop_room_state(room)

    def _on_pull_req(self, msg: list[bytes]) -> None:
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

    def append_send_blocks(
        self,
        room: int,
        block_ids: list[int],
        sliding_window_attn_block_ids: list[int] | None = None,
    ) -> int | None:
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
            if sliding_window_attn_block_ids is not None:
                self._send_sliding_window_attn_blocks[room] = list(
                    sliding_window_attn_block_ids
                )
        return start

    def offer_chunk(
        self, room: int, start: int, block_ids: list[int], total_pages: int
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
        self, room: int, start: int, block_ids: list[int], total_pages: int
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
                uuid = self._next_uuid_locked()
                addr = self._decode_addr[room]
            req_id = f"{room}#{index}"
            self._register_sender_read(room, req_id, uuid, block_ids)
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
                (self.full_attn_engine.endpoint or "").encode("ascii"),
            ]
            endpoint = _tcp_endpoint(ip=addr[0], port=addr[1])
            self._send_multipart_locked(endpoint, payload)

    def finalize_send(self, room: int, aux_index: int | None) -> bool:
        """Announce the room's blocks to the decode. False if the room was aborted.

        The room turns Transferring under the same lock that an abort takes
        to fail it, so the two cannot interleave: an abort either sees the
        finalized room and fails it terminally, or wins and this returns
        False without touching the status.
        """
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

    def _awaiting_decode_since(self, room: int, since: float) -> float | None:
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
        self,
        room: int,
        req_id: str,
        uuid: int,
        block_ids: list[int],
        *,
        sliding_window_attn: bool = False,
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
            engine = (
                self.sliding_window_attn_engine
                if sliding_window_attn
                else self.full_attn_engine
            )
            if engine is None:
                raise RuntimeError("Raiden sliding-window engine is not configured")
            accepted = engine.register_read(req_id, uuid, block_ids)
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
            if self._reshard_pushes_pending.get(room, 0) > 0:
                return True
            for entry in self._stale_plans:
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
                or (
                    self.has_sliding_window_attn
                    and room not in self._send_sliding_window_attn_blocks
                )
                or room not in self._decode_addr
            ):
                return
            sliding_window_attn_blocks = list(
                self._send_sliding_window_attn_blocks.get(room, [])
            )
            blocks = list(self._send_blocks[room])
            addr = self._decode_addr[room]
            aux_blob = self._send_aux.get(room, b"")
            self._replied.add(room)
        with self._announce_lock:
            if self.has_sliding_window_attn:
                self._register_sender_read(
                    room,
                    self._sliding_window_attn_req_id(room),
                    room,
                    sliding_window_attn_blocks,
                    sliding_window_attn=True,
                )
            self._register_sender_read(room, str(room), room, blocks)
            with self._state_lock:
                # The room can fail while its reads register; the decode is
                # gone and its own handlers drop frames for the room.
                if room in self._aborted:
                    return
            endpoint = _tcp_endpoint(ip=addr[0], port=addr[1])
            payload = [
                MessageTag.BLOCKS,
                str(room).encode("ascii"),
                pack_blocks(blocks),
                (self.full_attn_engine.endpoint or "").encode("ascii"),
                pack_blocks(sliding_window_attn_blocks),
                (
                    self.sliding_window_attn_engine.endpoint
                    if self.sliding_window_attn_engine
                    else ""
                ).encode("ascii"),
                aux_blob,
            ]
            self._send_multipart_locked(endpoint, payload)

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

    def _on_reshard_req(self, msg: list[bytes]) -> None:
        self._require_frames(msg, MessageTag.RESHARD_REQ, 9)
        room = int(msg[1].decode("ascii"))
        decode_ip = msg[2].decode("ascii")
        decode_port = int(msg[3].decode("ascii"))
        push_endpoint = msg[4].decode("ascii")
        decode_blocks = unpack_ints(msg[5])
        dst_attn_tp = int(msg[6].decode("ascii"))
        decode_sliding_window_attn_blocks = unpack_ints(msg[7])
        sliding_window_attn_push_endpoint = msg[8].decode("ascii")
        with self._state_lock:
            if room in self._aborted:
                return
            self._reshard_req[room] = dict(
                reply_addr=(decode_ip, decode_port),
                push_endpoint=push_endpoint,
                decode_blocks=decode_blocks,
                dst_attn_tp=dst_attn_tp,
                decode_sliding_window_attn_blocks=decode_sliding_window_attn_blocks,
                sliding_window_attn_push_endpoint=sliding_window_attn_push_endpoint,
            )
            held = self._held_chunks.pop(room, [])
            total_pages = self._send_total.get(room, 0)
        for start, block_ids in held:
            self._announce_reshard_chunk(room, start, block_ids, total_pages)
        self._maybe_reshard_reply(room)

    def _announce_reshard_chunk(
        self, room: int, start: int, block_ids: list[int], total_pages: int
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
            endpoint = _tcp_endpoint(ip=addr[0], port=addr[1])
            self._send_multipart_locked(endpoint, payload)

    def _on_reshard_chunk_ready(self, msg: list[bytes]) -> None:
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
            sliding_window_attn_blocks = list(
                self._send_sliding_window_attn_blocks.get(room, [])
            )
            aux_blob = self._send_aux.get(room, b"")
        with self._announce_lock:
            with self._state_lock:
                if room not in self._reshard_req:
                    return
                # Committed under the order lock: no announcement can follow.
                self._reshard_replied.add(room)
                self._reshard_replied_at[room] = time.time()
            endpoint = _tcp_endpoint(ip=addr[0], port=addr[1])
            payload = [
                MessageTag.RESHARD_BLOCKS,
                str(room).encode("ascii"),
                pack_ints(blocks),
                str(self._local_rank).encode("ascii"),
                aux_blob,
                pack_ints(sliding_window_attn_blocks),
            ]
            self._send_multipart_locked(endpoint, payload)

    def _on_reshard_ready(self, msg: list[bytes]) -> None:
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
        self, room: int, chunk: int | None = None, uuid: int | None = None
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

    def _run_reshard_sliding_window_attn_push(
        self, room: int, req_meta: dict, sliding_window_attn_blocks: list[int]
    ) -> None:
        """Push the whole sliding-window namespace with the closing transfer."""
        engine = self.sliding_window_attn_engine
        if engine is None:
            return
        sliding_window_attn_req_id = self._sliding_window_attn_req_id(room)
        decode_sliding_window_attn_blocks = list(
            req_meta.get("decode_sliding_window_attn_blocks") or []
        )
        sliding_window_attn_push_endpoint = (
            req_meta.get("sliding_window_attn_push_endpoint") or ""
        )
        n = len(decode_sliding_window_attn_blocks)
        if (
            not n
            or not sliding_window_attn_blocks
            or not sliding_window_attn_push_endpoint
        ):
            reason = (
                f"Raiden reshard sliding-window attention push: hybrid room {room} has nothing to "
                f"send (decode_pages={n} local_pages={len(sliding_window_attn_blocks)} "
                f"peer={sliding_window_attn_push_endpoint!r})"
            )
            logger.error("RaidenKVManager rank%d %s", self.attn_tp_rank, reason)
            with self._state_lock:
                self._failed.add(str(room))
                self._completed_send.add(sliding_window_attn_req_id)
                self.record_failure(room, reason)
            return
        if len(sliding_window_attn_blocks) > n:
            sliding_window_attn_blocks = sliding_window_attn_blocks[-n:]
        elif len(sliding_window_attn_blocks) < n:
            reason = (
                f"Raiden reshard sliding-window attention push: prefill has {len(sliding_window_attn_blocks)} "
                f"sliding-window pages, decode expects {n}"
            )
            logger.error("RaidenKVManager rank%d %s", self.attn_tp_rank, reason)
            with self._state_lock:
                self._failed.add(str(room))
                self._completed_send.add(sliding_window_attn_req_id)
                self.record_failure(room, reason)
            return

        from tpu_sync.rpc import raiden_service_pb2 as pb2

        from sglang.srt.disaggregation.raiden.reshard_schedule import (
            build_sender_request,
        )

        geom = self._make_geometry(
            self.attn_tp_size,
            req_meta["dst_attn_tp"],
            sliding_window_attn=True,
        )
        req = build_sender_request(
            pb2,
            uuid=room,
            req_id=sliding_window_attn_req_id,
            peer=sliding_window_attn_push_endpoint,
            prefill_rank=self.attn_tp_rank,
            decode_rank=self.attn_tp_rank // geom.ratio,
            geom=geom,
            src_block_ids=sliding_window_attn_blocks,
            dst_block_ids=decode_sliding_window_attn_blocks,
            live_tokens=[self.kv_args.page_size] * n,
        )
        fut = None
        try:
            engine.register_plan(room, req, is_sender=True)
            host_blocks = engine.plan_host_blocks(room, sliding_window_attn_blocks)
            fut = engine.stage_to_host(
                sliding_window_attn_blocks,
                host_blocks,
                [1] * len(sliding_window_attn_blocks),
            )
            deadline = self._monotonic() + self._timeout_s
            while fut is not None and not fut.is_ready():
                if self._monotonic() > deadline:
                    raise TimeoutError(
                        f"raiden reshard sliding-window attention D2H stage for room {room} timed out"
                    )
                time.sleep(0.0005)
            if fut is not None and not fut.ok():
                raise RuntimeError(
                    f"raiden reshard sliding-window attention D2H stage for room {room} failed: "
                    f"{fut.error_message()}"
                )
            with self._reshard_push_lock:
                for layer_idx in range(engine.kv_layer_count):
                    engine.push_plan(
                        room,
                        sliding_window_attn_push_endpoint,
                        sliding_window_attn_blocks,
                        decode_sliding_window_attn_blocks,
                        layer_idx,
                    )
            logger.info(
                "raiden reshard sliding-window attention pushed room=%s prefill_rank=%d "
                "decode_rank=%d pages=%d layers=%d peer=%s",
                room,
                self.attn_tp_rank,
                self.attn_tp_rank // geom.ratio,
                n,
                engine.kv_layer_count,
                sliding_window_attn_push_endpoint,
            )
        except Exception as exc:
            reason = f"Raiden reshard sliding-window attention push for room {room} failed: {exc}"
            logger.error("RaidenKVManager rank%d %s", self.attn_tp_rank, reason)
            with self._state_lock:
                self._failed.add(str(room))
                self.record_failure(room, reason)
        finally:
            if fut is not None and not fut.is_ready():
                with self._state_lock:
                    self._stale_plans.append((room, room, fut))
                threading.Thread(
                    target=self._reap_stale_plan,
                    args=(room, room, fut),
                    daemon=True,
                ).start()
            else:
                engine.unregister_plan(room)
            with self._state_lock:
                self._completed_send.add(sliding_window_attn_req_id)

    def _run_reshard_push(
        self, room: int, chunk: int | None = None, uuid: int | None = None
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
                sliding_window_attn_blocks = list(
                    self._send_sliding_window_attn_blocks.get(room, [])
                )
                start = 0
            else:
                start, prefill_blocks = self._reshard_chunks[room].pop(chunk)
            total_pages = self._send_total.get(room, len(prefill_blocks))

        # sliding-window attention is unchunked and rides the closing push even if chunks already
        # cover every full-attention page.
        if chunk is None:
            self._run_reshard_sliding_window_attn_push(
                room, req_meta, sliding_window_attn_blocks
            )

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
        self.full_attn_engine.register_plan(uuid, req, is_sender=True)
        fut = None
        try:
            host_blocks = self.full_attn_engine.plan_host_blocks(uuid, prefill_blocks)
            fut = self.full_attn_engine.stage_to_host(
                prefill_blocks, host_blocks, [1] * n
            )
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
                for layer_idx in range(self.full_attn_engine.kv_layer_count):
                    self.full_attn_engine.push_plan(
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
                self._unregister_plan(uuid)
        if not self.chunked_transfer:
            with self._state_lock:
                self._completed_send.add(str(room))

    @staticmethod
    def _monotonic() -> float:
        return time.monotonic()

    def is_send_done(self, room: int) -> bool:
        rid = str(room)
        return rid in self._completed_send and (
            not self.has_sliding_window_attn
            or self._sliding_window_attn_req_id(room) in self._completed_send
        )


class RaidenKVSender(CommonKVSender):
    # A group-wide Failed is delivered only after every rank drained: raiden
    # cannot cancel a transfer, so a peer's pages must not be freed under it.
    group_failure_requires_drain = True

    def __init__(
        self,
        mgr: RaidenPrefillKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: list[int],
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
        self._transfer_start_time: float | None = None
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

    def should_send_kv_chunk(self, num_pages: int, last_chunk: bool) -> bool:
        self._last_chunk = last_chunk
        return num_pages > 0 or last_chunk

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: list | None = None,
        num_kv_tokens: int | None = None,
    ):
        if self.kv_mgr.is_dummy_cp_rank or self.kv_mgr.is_aborted(self.bootstrap_room):
            return
        if self._transfer_start_time is None and (
            len(kv_indices) > 0 or state_indices is not None
        ):
            self._transfer_start_time = time.perf_counter()
        self._record_transfer_indices(kv_indices, state_indices)
        sliding_window_attn_blocks = (
            self.kv_mgr.sliding_window_attn_blocks_from_state_indices(state_indices)
            if state_indices is not None
            else None
        )
        blocks = [int(x) for x in kv_indices]
        start = self.kv_mgr.append_send_blocks(
            self.bootstrap_room, blocks, sliding_window_attn_blocks
        )
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

    def _finalize_transfer_latency(self) -> None:
        if (
            self._transfer_start_time is not None
            and self._transfer_metric.transfer_latency_s is None
        ):
            self._transfer_metric.transfer_latency_s = (
                time.perf_counter() - self._transfer_start_time
            )

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state
        if self.kv_mgr.is_dummy_cp_rank:
            self.conclude_state = KVPoll.Success
            return KVPoll.Success
        self.kv_mgr.drain_engine()
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
            self._finalize_transfer_latency()
            self.conclude_state = KVPoll.Success
            return KVPoll.Success
        if status == KVPoll.Success:
            self._finalize_transfer_latency()
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
        self.kv_mgr.fail_room(
            self.bootstrap_room,
            "Raiden transfer failed on another attention TP/CP rank.",
            preserve_reason=True,
        )
        # A local Success may have been cached before a peer contributed the
        # group failure. Re-enter the failure poll path after this collective.
        self.conclude_state = None
        # State that could admit new work is discarded, while the leases added
        # for already accepted reads/stages/pushes intentionally survive.
        self.kv_mgr.drop_room_state(self.bootstrap_room)

    def abort(self) -> None:
        """Seal a local abort, but do not conclude it over retained source KV.

        The scheduler may recycle the request's blocks as soon as this sender
        is terminal. Keep ``conclude_state`` unset until every native read /
        stage / push lease has settled; the normal poll path then publishes
        Failed.
        """
        self.kv_mgr.fail_room(self.bootstrap_room, "Aborted by AbortReq.")
        self.kv_mgr.drop_room_state(self.bootstrap_room)
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
