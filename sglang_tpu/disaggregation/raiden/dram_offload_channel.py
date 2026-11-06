"""Raiden channel of the KV offload to decode DRAM.

The offload moves pages between the HiCache host pools of a prefill rank and
the decode rank it serves, with native host-to-host reads. This is its
transport: the control messages on the manager's listener socket, the offer in
PULL_REQ, and the host reads in flight. It knows no cache; the L2 extension in
``sglang_tpu/disaggregation/decode_dram_offload`` drives it and binds the host
pool whose pages move.

- Push: VICTIM_PUSH offers registered host pages; VICTIM_ACK says whether
  the decode stored them.
- Recall: PULL_REQ names the host pages the decode offers, PREFIX_PULL asks for
  a range of them, PREFIX_READY says they are registered for the prefill's read.

A hybrid sliding-window model moves two page sets, each through its own host
pool: the full-attention KV and the last sliding window, which ends where the
full pages end. A recall takes the whole offered window or nothing.

A pair that cannot offload fails its requests with the reason instead of
running without the offload (``peer_switch_mismatch``, ``resharding_refusal``).
"""

from __future__ import annotations

import secrets
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import msgspec
import zmq

from sglang_tpu.disaggregation.raiden import sender as raiden_sender
from sglang_tpu.disaggregation.raiden.common import _tcp_endpoint, logger
from sglang_tpu.disaggregation.raiden.serde import (
    PrefixPull,
    PrefixReady,
    VictimAck,
    VictimPush,
)
from sglang_tpu.envs import envs as tpu_envs

if TYPE_CHECKING:
    from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager


_SWITCH = tpu_envs.SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM


def peer_switch_mismatch(*, decode_enabled: bool) -> Optional[str]:
    """Why this prefill cannot serve a decode: the switch is on for one role
    only."""
    mine = _SWITCH.get()
    if mine == decode_enabled:
        return None
    on, off = ("this engine", "its decode") if mine else ("its decode", "this engine")
    return (
        f"{_SWITCH.name} is on for {on} and off for {off}; "
        "set it to the same value on both roles"
    )


def resharding_refusal(*, decode_enabled: bool) -> Optional[str]:
    """Why a prefill refuses a decode that asks for a resharded (different
    attention TP) transfer: the switch is on here, or on the decode only (it
    would then run without the offload and never say so); None otherwise."""
    if not _SWITCH.get():
        return peer_switch_mismatch(decode_enabled=decode_enabled)
    return (
        f"{_SWITCH.name}=1 does not support resharding: the decode "
        "runs at a different attention TP size than this prefill; use equal "
        f"attention TP on both roles or unset {_SWITCH.name}"
    )


class _RecallState(msgspec.Struct):
    """Prefill: one room's recall of the prefix its decode offered."""

    offered_at: float
    # The offered host page ids, until the puller takes them, and the offered
    # window pages of a hybrid model.
    blocks: Optional[List[int]]
    swa_blocks: List[int]
    # The decode's READY reply as (uuid, endpoint, window uuid, window
    # endpoint); uuid 0 refuses, window uuid 0 registered no window.
    ready: Optional[Tuple[int, str, int, str]] = None
    # Settled, pulled or not: the room may leave Bootstrapping.
    resolved: bool = False


class _OfferState(msgspec.Struct):
    """Decode: the host pages a room offers, full and window (emptied once the
    queue lets go of them), and the prefill rank that may ask for them."""

    pages: List[int]
    swa_pages: List[int]
    prefill_addr: str = ""


class DramOffloadChannel:
    # Shares the manager's listener socket, state lock and room bookkeeping,
    # so it takes the manager itself. Built early in the manager's __init__:
    # nothing of the manager is read here.
    def __init__(self, manager: RaidenKVManager) -> None:
        self._mgr = manager
        # Host read id -> monotonic time its poll began raising on every poll;
        # past the transfer deadline the read is failed.
        self._poll_failing_since: Dict[str, float] = {}
        self._poll_deadline_s = float(
            tpu_envs.SGLANG_RAIDEN_CHUNK_TRANSFER_TIMEOUT_S.get()
        )
        # The HiCache host pools whose pages move, (full, sliding window or
        # None), bound by the L2 extension.
        self._host_pools: Tuple[Any, Any] = (None, None)
        # This engine's native host reads by request id, as source or as
        # destination: None while raiden may still read or write the pages,
        # then whether the read succeeded. A room's reads outlive its cleanup
        # until they settle. ``_read_transfers`` has the pending ones' pools.
        self._reads: Dict[str, Optional[bool]] = {}
        self._read_transfers: Dict[str, Any] = {}
        self._recalls: Dict[int, _RecallState] = {}
        self._offers: Dict[int, _OfferState] = {}
        # Push, prefill: the decode's answer by seq of this prefill's open
        # pushes, None until it arrives.
        self._replies: Dict[int, Optional[bool]] = {}
        # Push, decode: pushes received and not yet taken, by (prefill, seq).
        self._pushes: Dict[Tuple[str, int], VictimPush] = {}

    @property
    def _lock(self) -> Any:
        return self._mgr._state_lock

    def message_handlers(self) -> Dict[bytes, Callable]:
        return {
            PrefixPull.TAG: self._on_prefix_pull,
            PrefixReady.TAG: self._on_prefix_ready,
            VictimPush.TAG: self._on_victim_push,
            VictimAck.TAG: self._on_victim_ack,
        }

    def forget_room_locked(self, room: int) -> None:
        self._recalls.pop(room, None)
        self._offers.pop(room, None)
        for req_id in (self.prefix_req_id(room), self.prefix_req_id(room, swa=True)):
            if self._reads.get(req_id) is not None:
                del self._reads[req_id]

    def _send(self, *, endpoint: str, message: Any) -> bool:
        # A peer that cannot be reached costs only this recall or push; the
        # heartbeat fails the peer's rooms.
        try:
            self._mgr._send_multipart_locked(endpoint, message.to_frames())
        except zmq.ZMQError as exc:
            if exc.errno not in raiden_sender.ROOM_SCOPED_SEND_ERRNOS:
                raise
            logger.warning(
                "RaidenKVManager rank%d: %s to %s failed (errno %s): %s",
                self._mgr.attn_tp_rank,
                type(message).__name__,
                endpoint,
                exc.errno,
                exc,
            )
            return False
        return True

    # ----- native host reads, shared by recall and push -------------------------

    def bind_host_pools(self, full: Any, swa: Any = None) -> None:
        with self._lock:
            self._host_pools = (full, swa)

    def host_control_address(self, *, swa: bool = False) -> str:
        """Where a peer reads this engine's registered host pages from."""
        transfer = self._host_pools[swa].transfer
        return transfer.host_control_address(self._mgr.local_ip)

    def register_host_read(
        self, req_id: str, uuid: int, pages: List[int], *, swa: bool = False
    ) -> bool:
        """Register this engine's host ``pages`` (of its SWA pool with ``swa``)
        as the source of a peer's read. An accepted registration keeps the
        pages owned by the native read until it settles."""
        with self._lock:
            return self._register_host_read_locked(req_id, uuid, pages, swa=swa)

    def _register_host_read_locked(
        self, req_id: str, uuid: int, pages: List[int], *, swa: bool = False
    ) -> bool:
        transfer = self._host_pools[swa].transfer
        try:
            transfer.register_read(req_id, uuid, pages)
        except Exception as exc:  # noqa: BLE001 - the caller declines the transfer
            logger.warning("Cannot register host read %s: %s", req_id, exc)
            return False
        self._reads[req_id] = None
        self._read_transfers[req_id] = transfer
        return True

    def start_host_read(
        self,
        req_id: str,
        uuid: int,
        endpoint: str,
        remote: List[int],
        local: List[int],
        *,
        swa: bool = False,
    ) -> bool:
        """Read a peer's registered host pages into this engine's host pages
        (of its SWA pool with ``swa``); no device page and no staging slot is
        involved."""
        with self._lock:
            if self._mgr._room_of(req_id) in self._mgr._aborted:
                return False
            transfer = self._host_pools[swa].transfer
            try:
                transfer.start_read(req_id, uuid, endpoint, remote, local)
            except Exception as exc:  # noqa: BLE001 - the caller gives the read up
                logger.warning("Cannot start host read %s: %s", req_id, exc)
                return False
            self._reads[req_id] = None
            self._read_transfers[req_id] = transfer
            return True

    def poll_host_reads(self, req_ids: Sequence[str]) -> Optional[bool]:
        """None while raiden may still read or write any of the reads' pages,
        then whether all succeeded. A read whose record is gone did not."""
        with self._lock:
            results = [self._reads.get(req_id, False) for req_id in req_ids]
        return None if None in results else all(results)

    def forget_host_reads(self, req_ids: Sequence[str]) -> None:
        with self._lock:
            for req_id in req_ids:
                if self._reads.get(req_id) is not None:
                    del self._reads[req_id]

    def host_read_pending_locked(self, room: int) -> bool:
        return any(
            done is None and self._mgr._is_room_transfer(req_id, room)
            for req_id, done in self._reads.items()
        )

    def pending_host_reads_locked(self) -> List[str]:
        return [req_id for req_id, done in self._reads.items() if done is None]

    def drain_host_reads(self) -> None:
        # HiCache's wrapper is the sole native-report owner; it caches reports
        # for push and recall alike. Never poll its manager directly.
        with self._lock:
            pending = self.pending_host_reads_locked()
            if not pending:
                return
            with self._mgr._status_lock:
                live_rooms = set(self._mgr.request_status)
            for req_id in pending:
                transfer = self._read_transfers[req_id]
                try:
                    status = transfer.poll_transfer(req_id)
                except Exception as exc:
                    status = self._poll_kept_failing(req_id, exc)
                    if status is None:
                        continue
                else:
                    self._poll_failing_since.pop(req_id, None)
                if status is None:
                    continue
                transfer.forget_transfer(req_id)
                del self._read_transfers[req_id]
                room = self._mgr._room_of(req_id)
                if room is not None and (status == "sent" or room not in live_rooms):
                    # Nobody asks for the outcome of a recall's source or of a
                    # cleared room; a failed read fails neither role's room.
                    del self._reads[req_id]
                else:
                    self._reads[req_id] = status != "failed"

    def _poll_kept_failing(self, req_id: str, exc: Exception) -> Optional[str]:
        """A poll that has raised for longer than the transfer deadline fails
        the read, so its pages are released like any failed read; until then
        the first failure is logged and the read stays pending."""
        now = self._mgr._monotonic()
        since = self._poll_failing_since.setdefault(req_id, now)
        if now - since > self._poll_deadline_s:
            del self._poll_failing_since[req_id]
            logger.error(
                "Host read %s: poll has raised for %.0fs, past the %.0fs transfer "
                "deadline; the read is failed (%s)",
                req_id,
                now - since,
                self._poll_deadline_s,
                exc,
            )
            return "failed"
        if since == now:
            logger.warning("Cannot poll host read %s: %s", req_id, exc)
        return None

    # ----- recall, prefill side (prefill_dram_prefix_puller.PrefillDramPrefixPuller) --------------

    @staticmethod
    def prefix_req_id(room: int, *, swa: bool = False) -> str:
        # Shaped like a chunk id so the manager's room bookkeeping covers it.
        return f"{room}#pxs" if swa else f"{room}#px"

    def accept_prefix_offer_locked(
        self, room: int, blocks: List[int], swa_blocks: List[int]
    ) -> None:
        if blocks and room not in self._recalls:
            self._recalls[room] = _RecallState(
                offered_at=time.time(), blocks=list(blocks), swa_blocks=list(swa_blocks)
            )

    def prefix_pull_open_locked(self, room: int) -> bool:
        """Whether the room must stay in Bootstrapping: the decode offered pages
        and the puller has not settled the recall. A recall that makes no
        progress for the bootstrap timeout is given up on."""
        recall = self._recalls.get(room)
        if recall is None or recall.resolved:
            return False
        if time.time() - recall.offered_at > self._mgr.bootstrap_timeout:
            logger.warning(
                "RaidenKVManager rank%d room %s: decode prefix pull did not "
                "settle in %.0fs, computing the prompt instead",
                self._mgr.attn_tp_rank,
                room,
                self._mgr.bootstrap_timeout,
            )
            recall.resolved = True
            return False
        return True

    def prefix_ready(self, room: int) -> Optional[Tuple[int, str, int, str]]:
        """The decode's READY reply to the room's pull, if in: the uuid and
        endpoint of the full pages, then of the window."""
        with self._lock:
            recall = self._recalls.get(room)
            if recall is None or recall.resolved:
                return None
            return recall.ready

    def take_prefix_offer(self, room: int) -> Optional[Tuple[List[int], List[int]]]:
        """The host page ids the decode offered, full and window, handed out
        once."""
        with self._lock:
            recall = self._recalls.get(room)
            if recall is None or recall.resolved or recall.blocks is None:
                return None
            blocks, recall.blocks = recall.blocks, None
            return blocks, recall.swa_blocks

    def resolve_prefix_pull(self, room: int) -> None:
        """The recall is settled, pulled or not; the room may leave Bootstrapping."""
        with self._lock:
            recall = self._recalls.get(room)
            if recall is not None:
                recall.resolved = True

    def ask_prefix_pull(self, room: int, first_page: int, num_pages: int) -> bool:
        """Ask the decode to register offered pages [first_page, +num_pages)."""
        with self._lock:
            addr = self._mgr._prefill.decode_addr.get(room)
            if room in self._mgr._aborted or addr is None:
                return False
        return self._send(
            endpoint=_tcp_endpoint(ip=addr[0], port=addr[1]),
            message=PrefixPull(room=room, first_page=first_page, num_pages=num_pages),
        )

    def _on_prefix_ready(self, msg: List[bytes]) -> None:
        ready = PrefixReady.from_frames(msg)
        with self._lock:
            # A late reply to a cleared room is dropped.
            recall = self._recalls.get(ready.room)
            if recall is not None and ready.room not in self._mgr._aborted:
                recall.ready = (
                    ready.uuid,
                    ready.endpoint,
                    ready.swa_uuid,
                    ready.swa_endpoint,
                )

    # ----- recall, decode side (decode_dram_prefix_offerer.DecodeDramPrefixOfferer) ---------------

    def offer_resident_prefix(
        self, room: int, page_ids: List[int], swa_page_ids: Sequence[int] = ()
    ) -> None:
        """Offer the host pages of the room's held prefix for recall, with the
        window pages that end where they end on a hybrid model. Called before
        the room's PULL_REQ is sent, which names them to the prefill."""
        with self._lock:
            if page_ids and room not in self._mgr._aborted:
                self._offers[room] = _OfferState(
                    pages=[int(x) for x in page_ids],
                    swa_pages=[int(x) for x in swa_page_ids],
                )

    def release_prefix_offer(self, room: int) -> None:
        # A late PREFIX_PULL must not register pages the queue relinquished;
        # it is still answered, with a refusal.
        with self._lock:
            offer = self._offers.get(room)
            if offer is not None:
                offer.pages = offer.swa_pages = []

    def host_prefix_pending(self, room: int) -> bool:
        with self._lock:
            return self.host_read_pending_locked(room)

    def offered_blocks_locked(
        self, room: int, bootstrap_infos: List[dict]
    ) -> Tuple[List[int], List[int]]:
        """The host pages the room's PULL_REQ offers, full and window; records
        the prefill rank that may ask for them."""
        offer = self._offers.get(room)
        if offer is None:
            return [], []
        offer.prefill_addr = _tcp_endpoint(
            ip=bootstrap_infos[0]["rank_ip"], port=bootstrap_infos[0]["rank_port"]
        )
        return offer.pages, offer.swa_pages

    def _on_prefix_pull(self, msg: List[bytes]) -> None:
        # Any refusal answers with uuid 0, so the prefill computes the prompt
        # at once; a prefix read never fails the decode's own room.
        req = PrefixPull.from_frames(msg)
        room = req.room
        uuid = swa_uuid = 0
        endpoint = swa_endpoint = ""
        with self._lock:
            offer = self._offers.get(room)
            if offer is None:
                return
            wanted = offer.pages[req.first_page : req.first_page + req.num_pages]
            if room in self._mgr._aborted or len(wanted) != req.num_pages:
                logger.warning(
                    "RaidenKVManager rank%d room %s: PREFIX_PULL for pages [%d, %d) "
                    "of %d offered cannot be served, refusing",
                    self._mgr.attn_tp_rank,
                    room,
                    req.first_page,
                    req.first_page + req.num_pages,
                    len(offer.pages),
                )
            else:
                # Independent prefills share a decoder's native UUID map; zero
                # means no transfer, as in _announce_chunk.
                uuid = secrets.randbelow((1 << 63) - 1) + 1
                swa_uuid = (
                    secrets.randbelow((1 << 63) - 1) + 1 if offer.swa_pages else 0
                )
                # An accepted registration stays owned by its native read also
                # when the other one of the pull is refused: raiden cannot
                # withdraw it.
                if self._register_host_read_locked(
                    self.prefix_req_id(room), uuid, wanted
                ) and (
                    not swa_uuid
                    or self._register_host_read_locked(
                        self.prefix_req_id(room, swa=True),
                        swa_uuid,
                        offer.swa_pages,
                        swa=True,
                    )
                ):
                    # The prefill reads from the host pools' own control endpoints.
                    endpoint = self.host_control_address()
                    if swa_uuid:
                        swa_endpoint = self.host_control_address(swa=True)
                else:
                    uuid = swa_uuid = 0
        self._send(
            endpoint=offer.prefill_addr,
            message=PrefixReady(
                room=room,
                uuid=uuid,
                endpoint=endpoint,
                swa_uuid=swa_uuid,
                swa_endpoint=swa_endpoint,
            ),
        )

    # ----- push (prefill_dram_victim_pusher.py, decode_dram_victim_receiver.py) ------------------------

    def decode_peer(self, room: int) -> Optional[Tuple[str, int]]:
        """(ip, port) of the decode rank whose PULL_REQ named the room."""
        with self._lock:
            return self._mgr._prefill.decode_addr.get(room)

    def send_victim_push(
        self,
        *,
        decode: Tuple[str, int],
        prefill: str,
        seq: int,
        tokens: List[int],
        blocks: List[int],
        uuid: int,
        extra_key: Optional[str],
        cache_salt: Optional[str],
        swa_blocks: Sequence[int] = (),
    ) -> bool:
        """Offer registered host pages for decode to pull into its host pools.
        A message that cannot be sent is a declined push."""
        with self._lock:
            self._replies[seq] = None
        return self._send(
            endpoint=_tcp_endpoint(ip=decode[0], port=decode[1]),
            message=VictimPush(
                prefill=prefill,
                seq=seq,
                ack_endpoint=_tcp_endpoint(
                    ip=self._mgr.local_ip, port=self._mgr.rank_port
                ),
                tokens=tokens,
                blocks=blocks,
                uuid=uuid,
                address=self.host_control_address(),
                swa_blocks=list(swa_blocks),
                swa_address=self.host_control_address(swa=True) if swa_blocks else "",
                extra_key=extra_key,
                cache_salt=cache_salt,
            ),
        )

    def victim_reply(self, seq: int) -> Optional[bool]:
        """Whether the decode stored push ``seq``; None until it answers."""
        with self._lock:
            return self._replies.get(seq)

    def forget_victim(self, seq: int) -> None:
        with self._lock:
            self._replies.pop(seq, None)

    def _on_victim_ack(self, msg: List[bytes]) -> None:
        ack = VictimAck.from_frames(msg)
        with self._lock:
            # A late answer to a push that already settled is dropped.
            if ack.seq in self._replies:
                self._replies[ack.seq] = ack.stored

    def _on_victim_push(self, msg: List[bytes]) -> None:
        push = VictimPush.from_frames(msg)
        with self._lock:
            self._pushes[(push.prefill, push.seq)] = push

    def received_pushes(self) -> Dict[Tuple[str, int], VictimPush]:
        """Pushes received and not yet taken."""
        with self._lock:
            return dict(self._pushes)

    def take_push(self, key: Tuple[str, int]) -> Optional[VictimPush]:
        with self._lock:
            return self._pushes.pop(key, None)

    def send_victim_ack(self, *, endpoint: str, seq: int, stored: bool) -> None:
        """Tell the prefill rank whether its push was stored. A lost answer
        lets the push time out there as declined."""
        self._send(endpoint=endpoint, message=VictimAck(seq=seq, stored=stored))
