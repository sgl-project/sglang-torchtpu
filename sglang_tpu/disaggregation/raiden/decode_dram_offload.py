"""Raiden manager support for the KV offload to decode DRAM.

``RaidenKVManager`` holds one ``DecodeDramOffload`` when the switch is on and calls
it at a few hooks; pd_victim_push.py and pd_host_prefix_recall.py drive it. It
carries two exchanges between a prefill rank and the decode rank it serves,
both moving pages of the HiCache host pool with native host-to-host reads:

- Push: VICTIM_PUSH / VICTIM_ACK / VICTIM_DONE.
- Recall: PULL_REQ names the host pages the decode offers, PREFIX_PULL asks for
  a range of them, PREFIX_READY says they are registered for the prefill's read.
"""

from __future__ import annotations

import functools
import secrets
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

import zmq

from sglang_tpu.disaggregation.raiden.common import _tcp_endpoint, logger
from sglang_tpu.disaggregation.raiden.sender import ROOM_SCOPED_SEND_ERRNOS
from sglang_tpu.disaggregation.raiden.serde import (
    PrefixPull,
    PrefixReady,
    VictimAck,
    VictimDone,
    VictimPush,
)

if TYPE_CHECKING:
    from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager


class DecodeDramOffload:
    # Shares the manager's listener socket, state lock and room bookkeeping,
    # so it takes the manager itself.
    def __init__(self, manager: RaidenKVManager) -> None:
        self._mgr = manager
        self._lock = manager._state_lock
        # Shared by this prefill's ranks; set by the pusher.
        self.victim_namespace = ""

        # The HiCache host pool whose pages move, and this engine's native host
        # reads in flight by request id, as source or as destination. They
        # outlive room cleanup until the HiCache transfer reports them settled.
        self._host_pool: Any = None
        self.host_reads: Dict[str, Any] = {}

        # Prefill, recall: when the decode's PULL_REQ offered pages; the page
        # ids until the puller takes them; rooms whose PREFIX_PULL is out;
        # READY replies as (uuid, endpoint); rooms whose recall is settled
        # either way; outcome ("done" or "failed") of settled reads of live rooms.
        self.offer_since: Dict[int, float] = {}
        self._offers: Dict[int, List[int]] = {}
        self._pull_asked: Set[int] = set()
        self._ready: Dict[int, Tuple[int, str]] = {}
        self._resolved: Set[int] = set()
        self._read_results: Dict[str, str] = {}

        # Decode, recall: the host page ids offered in PULL_REQ until the queue
        # releases them, and the prefill rank to answer.
        self.blocks: Dict[int, List[int]] = {}
        self.prefill_addr: Dict[int, str] = {}

        # Push: sequence numbers of this prefill's open offers, and messages
        # received and not yet taken: VictimAck by seq, the others by
        # (prefill, seq).
        self._victim_open: Set[int] = set()
        self._inbox: Dict[type, Dict[Any, Any]] = {
            VictimPush: {},
            VictimAck: {},
            VictimDone: {},
        }

    @property
    def local_ip(self) -> str:
        return self._mgr.local_ip

    def drain(self) -> None:
        self._mgr._drain_engine()

    def message_handlers(self) -> Dict[bytes, Callable]:
        return {
            PrefixPull.TAG: self._on_prefix_pull,
            PrefixReady.TAG: self._on_prefix_ready,
            **{
                cls.TAG: functools.partial(self._on_victim_message, cls)
                for cls in (VictimPush, VictimAck, VictimDone)
            },
        }

    def forget_room_locked(self, room: int) -> None:
        for table in (
            self.offer_since,
            self._offers,
            self._ready,
            self.blocks,
            self.prefill_addr,
        ):
            table.pop(room, None)
        self._pull_asked.discard(room)
        self._resolved.discard(room)
        self._read_results.pop(self.prefix_req_id(room), None)

    # ----- host reads of either role -------------------------------------------

    @staticmethod
    def prefix_req_id(room: int) -> str:
        # Shaped like a chunk id so the manager's room bookkeeping covers it.
        return f"{room}#px"

    def bind_prefix_host_pool(self, pool: Any) -> None:
        with self._lock:
            self._host_pool = pool

    def host_read_pending_locked(self, room: int) -> bool:
        return any(self._mgr._is_room_transfer(rid, room) for rid in self.host_reads)

    def host_prefix_pending(self, room: int) -> bool:
        with self._lock:
            return self.host_read_pending_locked(room)

    def drain_host_reads(self) -> None:
        # HiCache's wrapper is the sole native-report owner; it caches reports
        # for victim exports and recall alike. Never poll its manager directly.
        with self._lock:
            if not self.host_reads:
                return
            with self._mgr._status_lock:
                live_rooms = set(self._mgr.request_status)
            for req_id, transfer in list(self.host_reads.items()):
                try:
                    status = transfer.poll_transfer(req_id)
                except Exception as exc:
                    logger.warning("Cannot poll host prefix read %s: %s", req_id, exc)
                    continue
                if status is None:
                    continue
                transfer.forget_transfer(req_id)
                del self.host_reads[req_id]
                # Only the reader of a room that is still live asks for the
                # outcome; a failed read fails neither role's room.
                if status != "sent" and self._mgr._room_of(req_id) in live_rooms:
                    self._read_results[req_id] = (
                        "done" if status == "received" else "failed"
                    )

    # ----- recall, prefill side (pd_host_prefix_recall.HostPrefixPuller) -------

    def _active(self) -> bool:
        # The engine holds its KV pools; pd_spill_validate refused every other
        # unsupported setting at startup.
        return self._mgr._kv_caches is not None

    def accept_prefix_offer_locked(self, room: int, blocks: List[int]) -> None:
        if blocks and self._active():
            self._offers[room] = list(blocks)
            self.offer_since[room] = time.time()

    def prefix_pull_open_locked(self, room: int) -> bool:
        """Whether the room must stay in Bootstrapping: the decode offered pages
        and the puller has not settled the recall. A recall that makes no
        progress for the bootstrap timeout is given up on."""
        since = self.offer_since.get(room)
        if since is None or room in self._resolved:
            return False
        if time.time() - since > self._mgr.bootstrap_timeout:
            logger.warning(
                "RaidenKVManager rank%d room %s: decode prefix pull did not "
                "settle in %.0fs, computing the prompt instead",
                self._mgr.attn_tp_rank,
                room,
                self._mgr.bootstrap_timeout,
            )
            self._resolved.add(room)
            return False
        return True

    def prefix_offer_available(self, room: int) -> bool:
        with self._lock:
            return room in self._offers and room not in self._resolved

    def take_prefix_offer(self, room: int) -> Optional[List[int]]:
        """The host page ids the decode offered, handed out once."""
        with self._lock:
            if room in self._resolved:
                return None
            return self._offers.pop(room, None)

    def resolve_prefix_pull(self, room: int) -> None:
        """The recall is settled, pulled or not; the room may leave Bootstrapping."""
        with self._lock:
            self._resolved.add(room)

    def prefix_pull_resolved(self, room: int) -> bool:
        with self._lock:
            return room in self._resolved

    def ask_prefix_pull(self, room: int, first_page: int, num_pages: int) -> bool:
        """Ask the decode to register offered pages [first_page, +num_pages)."""
        with self._lock:
            addr = self._mgr._prefill.decode_addr.get(room)
            if room in self._mgr._aborted or addr is None:
                return False
            self._pull_asked.add(room)
        endpoint = _tcp_endpoint(ip=addr[0], port=addr[1])
        payload = PrefixPull(
            room=room, first_page=first_page, num_pages=num_pages
        ).to_frames()
        try:
            self._mgr._send_multipart_locked(endpoint, payload)
        except zmq.ZMQError as exc:
            if exc.errno not in ROOM_SCOPED_SEND_ERRNOS:
                raise
            logger.error(
                "RaidenKVManager rank%d room %s: PREFIX_PULL to %s failed "
                "(errno %s): %s",
                self._mgr.attn_tp_rank,
                room,
                endpoint,
                exc.errno,
                exc,
            )
            return False
        return True

    def _on_prefix_ready(self, msg: List[bytes]) -> None:
        ready = PrefixReady.from_frames(msg)
        with self._lock:
            if ready.room in self._mgr._aborted or ready.room not in self._pull_asked:
                return
            self._ready[ready.room] = (ready.uuid, ready.endpoint)

    def prefix_pull_ready(self, room: int) -> Optional[Tuple[str, int, str]]:
        """The decode's READY reply as (req_id, uuid, endpoint), if in; uuid 0
        refuses."""
        with self._lock:
            ready = self._ready.get(room)
        return None if ready is None else (self.prefix_req_id(room), *ready)

    def start_host_prefix_read(
        self,
        req_id: str,
        uuid: int,
        endpoint: str,
        remote: List[int],
        local: List[int],
    ) -> None:
        """Read the decode's registered host pages into this engine's host
        pages; no device page and no staging slot is involved."""
        with self._lock:
            if self._mgr._room_of(req_id) in self._mgr._aborted:
                raise RuntimeError("Cannot start a host prefix read after abort")
            if self._host_pool is None:
                raise RuntimeError("Host prefix recall requires a HiCache pool")
            transfer = self._host_pool.transfer
            transfer.start_read(
                req_id,
                uuid,
                endpoint,
                remote,
                None,
                local_host_block_ids=local,
                destination="host",
            )
            self.host_reads[req_id] = transfer

    def prefix_read_state(self, req_id: str) -> str:
        """ "pending" (raiden may still write the pages), "done", "failed", or
        "unknown" once the room's record is gone."""
        with self._lock:
            if req_id in self.host_reads:
                return "pending"
            return self._read_results.get(req_id, "unknown")

    # ----- recall, decode side (pd_host_prefix_recall.HostPrefixSource) --------

    def offer_resident_prefix(self, room: int, page_ids: List[int]) -> None:
        """Offer the host pages of the room's held prefix for recall. Called
        before the room's PULL_REQ is sent, which names them to the prefill."""
        with self._lock:
            if page_ids and room not in self._mgr._aborted:
                self.blocks[room] = [int(x) for x in page_ids]

    def release_prefix_offer(self, room: int) -> None:
        # A late PREFIX_PULL must not register pages the queue relinquished.
        with self._lock:
            self.blocks.pop(room, None)

    def offered_blocks_locked(
        self, room: int, bootstrap_infos: List[dict]
    ) -> List[int]:
        """The host pages the room's PULL_REQ offers; records the prefill rank
        that may ask for them."""
        if len(bootstrap_infos) != 1:
            return []
        blocks = self.blocks.get(room, [])
        if blocks:
            self.prefill_addr[room] = _tcp_endpoint(
                ip=bootstrap_infos[0]["rank_ip"], port=bootstrap_infos[0]["rank_port"]
            )
        return blocks

    def _send_to_prefill(self, *, room: int, addr: str, frames: List[bytes]) -> None:
        # A dead prefill only costs this room its recall; the heartbeat fails
        # the room itself.
        try:
            self._mgr._send_multipart_locked(addr, frames)
        except zmq.ZMQError as exc:
            if exc.errno not in ROOM_SCOPED_SEND_ERRNOS:
                raise
            logger.warning(
                "RaidenKVManager rank%d room %s: prefix message to %s failed "
                "(errno %s): %s",
                self._mgr.attn_tp_rank,
                room,
                addr,
                exc.errno,
                exc,
            )

    def _on_prefix_pull(self, msg: List[bytes]) -> None:
        # Any refusal answers with uuid 0, so the prefill computes the prompt
        # at once; a prefix read never fails the decode's own room.
        req = PrefixPull.from_frames(msg)
        room = req.room
        with self._lock:
            blocks = self.blocks.get(room)
            addr = self.prefill_addr.get(room)
            aborted = room in self._mgr._aborted
        if addr is None:
            return
        wanted = [] if aborted or blocks is None else blocks[req.first_page :]
        wanted = wanted[: req.num_pages]
        uuid, endpoint = 0, ""
        if len(wanted) != req.num_pages:
            logger.warning(
                "RaidenKVManager rank%d room %s: PREFIX_PULL for pages [%d, %d) "
                "cannot be served (offer=%s aborted=%s), refusing",
                self._mgr.attn_tp_rank,
                room,
                req.first_page,
                req.first_page + req.num_pages,
                None if blocks is None else len(blocks),
                aborted,
            )
        else:
            # Independent prefills share a decoder's native UUID map; zero
            # means no transfer, as in _announce_chunk.
            uuid = secrets.randbelow((1 << 63) - 1) + 1
            if self._register_prefix_read(room=room, uuid=uuid, blocks=wanted):
                # The prefill reads from the host pool's own control endpoint.
                endpoint = self._host_pool.transfer.host_control_address(
                    self._mgr.local_ip
                )
            else:
                uuid = 0
        self._send_to_prefill(
            room=room,
            addr=addr,
            frames=PrefixReady(room=room, uuid=uuid, endpoint=endpoint).to_frames(),
        )

    def _register_prefix_read(self, *, room: int, uuid: int, blocks: List[int]) -> bool:
        # An accepted registration keeps the pages owned by the native read
        # until its terminal report arrives.
        req_id = self.prefix_req_id(room)
        try:
            with self._lock:
                if room in self._mgr._aborted or room not in self.blocks:
                    raise RuntimeError("Host prefix offer was released")
                transfer = self._host_pool.transfer
                transfer.register_read(req_id, uuid, blocks, source="host")
                self.host_reads[req_id] = transfer
            return True
        except Exception as exc:
            logger.error(
                "RaidenKVManager rank%d room %s: prefix register_read %s raised: %s",
                self._mgr.attn_tp_rank,
                room,
                req_id,
                exc,
            )
            return False

    # ----- push (pd_victim_push.py) --------------------------------------------

    def decode_peer(self, room: int) -> Optional[Tuple[str, int]]:
        """(ip, port) of the decode rank whose PULL_REQ named the room."""
        with self._lock:
            return self._mgr._prefill.decode_addr.get(room)

    def _send_to_decode(self, *, decode: Tuple[str, int], message: Any) -> bool:
        try:
            self._mgr._send_multipart_locked(
                _tcp_endpoint(ip=decode[0], port=decode[1]), message.to_frames()
            )
        except Exception as exc:  # noqa: BLE001 - a lost message is a declined push
            logger.warning(
                "Raiden %s %d failed to send: %s",
                type(message).__name__,
                message.seq,
                exc,
            )
            return False
        return True

    def send_victim_push(
        self,
        *,
        decode: Tuple[str, int],
        seq: int,
        tokens: List[int],
        blocks: List[int],
        uuid: int,
        address: str,
        extra_key=None,
        cache_salt=None,
    ) -> bool:
        """Offer registered host pages for decode to pull into its host pool."""
        if not blocks:
            return False
        with self._lock:
            self._victim_open.add(seq)
        return self._send_to_decode(
            decode=decode,
            message=VictimPush(
                prefill=self.victim_namespace,
                seq=seq,
                ack_endpoint=_tcp_endpoint(
                    ip=self._mgr.local_ip, port=self._mgr.rank_port
                ),
                tokens=tokens,
                blocks=blocks,
                uuid=uuid,
                address=address,
                extra_key=extra_key,
                cache_salt=cache_salt,
            ),
        )

    def send_victim_done(self, *, decode: Tuple[str, int], seq: int, ok: bool) -> None:
        """Report the all-rank outcome after the producer's native reads
        settle. A lost notice lets the landing expire on its own."""
        self._send_to_decode(
            decode=decode,
            message=VictimDone(prefill=self.victim_namespace, seq=seq, ok=ok),
        )

    def send_victim_ack(self, *, endpoint: str, ack: VictimAck) -> None:
        self._send_to_prefill(room=-1, addr=endpoint, frames=ack.to_frames())

    def _on_victim_message(self, cls: type, msg: List[bytes]) -> None:
        message = cls.from_frames(msg)
        with self._lock:
            if cls is not VictimAck:
                self._inbox[cls][(message.prefill, message.seq)] = message
            elif message.seq in self._victim_open:
                self._inbox[cls][message.seq] = message

    def victim_messages(self, cls: type) -> Dict[Any, Any]:
        """Messages of ``cls`` received and not yet taken, by key."""
        with self._lock:
            return dict(self._inbox[cls])

    def take_victim_message(self, cls: type, key: Any) -> Optional[Any]:
        with self._lock:
            return self._inbox[cls].pop(key, None)

    def forget_victim(self, seq: int) -> None:
        with self._lock:
            self._victim_open.discard(seq)
            self._inbox[VictimAck].pop(seq, None)
