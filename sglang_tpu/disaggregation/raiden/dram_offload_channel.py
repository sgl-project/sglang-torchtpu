"""Raiden channel of the KV offload to decode DRAM.

The offload moves pages from the HiCache host pool of a prefill rank to that
of the decode rank it serves, with native host-to-host reads. This is its
transport: the control messages on the manager's listener socket and the host
reads in flight. It knows no cache; the L2 extension in
``sglang_tpu/disaggregation/decode_dram_offload`` drives it and binds the host
pool whose pages move.

VICTIM_PUSH offers registered host pages; VICTIM_ACK says whether the decode
stored them.

A pair that cannot offload fails its requests with the reason instead of
running without the offload (``peer_switch_mismatch``, ``resharding_refusal``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import zmq

from sglang_tpu.disaggregation.raiden import sender as raiden_sender
from sglang_tpu.disaggregation.raiden.common import _tcp_endpoint, logger
from sglang_tpu.disaggregation.raiden.serde import (
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


def resharding_refusal() -> Optional[str]:
    """Why a prefill with the switch on refuses a decode that asks for a
    resharded (different attention TP) transfer, or None with the switch off."""
    if not _SWITCH.get():
        return None
    return (
        f"{_SWITCH.name}=1 does not support resharding: the decode "
        "runs at a different attention TP size than this prefill; use equal "
        f"attention TP on both roles or unset {_SWITCH.name}"
    )


class DramOffloadChannel:
    # Shares the manager's listener socket, state lock and room bookkeeping,
    # so it takes the manager itself. Built early in the manager's __init__:
    # nothing of the manager is read here.
    def __init__(self, manager: RaidenKVManager) -> None:
        self._mgr = manager
        # Shared by this prefill's ranks; set by the pusher.
        self.victim_namespace = ""
        # The HiCache host pool whose pages move, bound by the L2 extension.
        self._host_pool: Any = None
        # This engine's native host reads by request id, as source or as
        # destination: None while raiden may still read or write the pages,
        # then whether the read succeeded.
        self._reads: Dict[str, Optional[bool]] = {}
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
            VictimPush.TAG: self._on_victim_push,
            VictimAck.TAG: self._on_victim_ack,
        }

    def _send(self, *, endpoint: str, message: Any) -> bool:
        # A peer that cannot be reached costs only this push; the
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

    # ----- native host reads ---------------------------------------------------

    def bind_host_pool(self, pool: Any) -> None:
        with self._lock:
            self._host_pool = pool

    def host_control_address(self) -> str:
        """Where a peer reads this engine's registered host pages from."""
        return self._host_pool.transfer.host_control_address(self._mgr.local_ip)

    def register_host_read(self, req_id: str, uuid: int, pages: List[int]) -> bool:
        """Register this engine's host ``pages`` as the source of a peer's read.
        An accepted registration keeps the pages owned by the native read
        until it settles."""
        with self._lock:
            return self._register_host_read_locked(req_id, uuid, pages)

    def _register_host_read_locked(
        self, req_id: str, uuid: int, pages: List[int]
    ) -> bool:
        try:
            self._host_pool.transfer.register_read(req_id, uuid, pages, source="host")
        except Exception as exc:  # noqa: BLE001 - the caller declines the transfer
            logger.warning("Cannot register host read %s: %s", req_id, exc)
            return False
        self._reads[req_id] = None
        return True

    def start_host_read(
        self,
        req_id: str,
        uuid: int,
        endpoint: str,
        remote: List[int],
        local: List[int],
    ) -> bool:
        """Read a peer's registered host pages into this engine's host pages;
        no device page and no staging slot is involved."""
        with self._lock:
            try:
                self._host_pool.transfer.start_read(
                    req_id, uuid, endpoint, remote, local, destination="host"
                )
            except Exception as exc:  # noqa: BLE001 - the caller gives the read up
                logger.warning("Cannot start host read %s: %s", req_id, exc)
                return False
            self._reads[req_id] = None
            return True

    def poll_host_read(self, req_id: str) -> Optional[bool]:
        """None while raiden may still read or write the pages, then whether
        the read succeeded. A read whose record is gone did not."""
        with self._lock:
            return self._reads.get(req_id, False)

    def forget_host_read(self, req_id: str) -> None:
        with self._lock:
            if self._reads.get(req_id) is not None:
                del self._reads[req_id]

    def pending_host_reads_locked(self) -> List[str]:
        return [req_id for req_id, done in self._reads.items() if done is None]

    def drain_host_reads(self) -> None:
        # HiCache's wrapper is the sole native-report owner and caches its
        # reports. Never poll its manager directly.
        with self._lock:
            pending = self.pending_host_reads_locked()
            if not pending:
                return
            transfer = self._host_pool.transfer
            for req_id in pending:
                try:
                    status = transfer.poll_transfer(req_id)
                except Exception as exc:
                    logger.warning("Cannot poll host read %s: %s", req_id, exc)
                    continue
                if status is None:
                    continue
                transfer.forget_transfer(req_id)
                self._reads[req_id] = status != "failed"

    # ----- push (prefill_dram_victim_pusher.py, decode_dram_victim_receiver.py) ------------------------

    def decode_peer(self, room: int) -> Optional[Tuple[str, int]]:
        """(ip, port) of the decode rank whose PULL_REQ named the room."""
        with self._lock:
            return self._mgr._prefill.decode_addr.get(room)

    def send_victim_push(
        self,
        *,
        decode: Tuple[str, int],
        seq: int,
        tokens: List[int],
        blocks: List[int],
        uuid: int,
        extra_key: Optional[str],
        cache_salt: Optional[str],
    ) -> bool:
        """Offer registered host pages for decode to pull into its host pool.
        A message that cannot be sent is a declined push."""
        with self._lock:
            self._replies[seq] = None
        return self._send(
            endpoint=_tcp_endpoint(ip=decode[0], port=decode[1]),
            message=VictimPush(
                prefill=self.victim_namespace,
                seq=seq,
                ack_endpoint=_tcp_endpoint(
                    ip=self._mgr.local_ip, port=self._mgr.rank_port
                ),
                tokens=tokens,
                blocks=blocks,
                uuid=uuid,
                address=self.host_control_address(),
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
