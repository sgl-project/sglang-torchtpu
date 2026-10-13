"""Liveness heartbeats for the engine's request-path processes.

Every scheduler rank and the detokenizer stamp a monotonic clock reading into
their own slot of a small /dev/shm file on each loop iteration, idle ones
included. The HTTP process reads the slots when a probe arrives and compares
each to the clock now. A stale slot means that process has not completed a
loop iteration for that long; nothing is sent into the process being judged,
so a hung scheduler is detected by a stale slot, not by a request it never
answers. Ranks on other hosts never appear here, but a hung remote rank blocks
the local ranks in their next collective, so their slots go stale instead.

Layout: a header, then ``SLOT_CAPACITY`` fixed slots. Slot 0 is the
detokenizer; scheduler ranks use ``1 + dp*(pp*tp) + pp_rank*tp + tp_rank``.
A slot whose ``beat_ts`` is zero was never written and is ignored.
"""

from __future__ import annotations

import hashlib
import logging
import mmap
import os
import struct
import time
from typing import Callable, List, Optional

import msgspec

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

MAGIC = b"SLHB"
VERSION = 1
HEADER_STRUCT = struct.Struct("<4sHHI")  # magic, version, slot_capacity, slot_size
# role (u8), paused (u8), rank (u16), beat_ts (f64), started_ts (f64)
SLOT_STRUCT = struct.Struct("<BBHdd")
SLOT_SIZE = 32
SLOT_CAPACITY = 256
FILE_SIZE = HEADER_STRUCT.size + SLOT_CAPACITY * SLOT_SIZE

ROLE_DETOKENIZER = 1
ROLE_SCHEDULER = 2
ROLE_NAMES = {ROLE_DETOKENIZER: "detokenizer", ROLE_SCHEDULER: "scheduler"}

DETOKENIZER_SLOT = 0


def heartbeat_shm_path(instance_id: str) -> str:
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in instance_id)
    digest = hashlib.blake2s(instance_id.encode(), digest_size=4).hexdigest()
    return f"/dev/shm/sglang_heartbeat_{safe}_{digest}.shm"


def scheduler_slot(
    *, dp_rank: int, pp_rank: int, tp_rank: int, pp_size: int, tp_size: int
) -> int:
    return 1 + dp_rank * (pp_size * tp_size) + pp_rank * tp_size + tp_rank


def _slot_offset(slot: int) -> int:
    return HEADER_STRUCT.size + slot * SLOT_SIZE


def _open_board(path: str) -> mmap.mmap:
    """Create the file on first open; concurrent creators write an identical header."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if os.fstat(fd).st_size < FILE_SIZE:
            os.ftruncate(fd, FILE_SIZE)
        board = mmap.mmap(fd, FILE_SIZE)
    finally:
        os.close(fd)
    HEADER_STRUCT.pack_into(board, 0, MAGIC, VERSION, SLOT_CAPACITY, SLOT_SIZE)
    return board


class HeartbeatSlot(msgspec.Struct, frozen=True, kw_only=True):
    role: str
    rank: int
    age_s: float
    paused: bool


class HeartbeatWriter:
    def __init__(
        self,
        path: str,
        *,
        role: int,
        slot: int,
        rank: int,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not 0 <= slot < SLOT_CAPACITY:
            raise ValueError(f"heartbeat slot {slot} outside capacity {SLOT_CAPACITY}")
        self._board = _open_board(path)
        self._offset = _slot_offset(slot)
        self._role = role
        self._rank = rank
        self._clock = clock
        self._paused = 0
        self._started = clock()
        self.beat()

    def beat(self) -> None:
        SLOT_STRUCT.pack_into(
            self._board,
            self._offset,
            self._role,
            self._paused,
            self._rank,
            self._clock(),
            self._started,
        )

    def set_paused(self, paused: bool) -> None:
        """A paused slot is never stale: for known-long steps such as a compile."""
        self._paused = 1 if paused else 0
        self.beat()

    def close(self) -> None:
        self._board.close()


class HeartbeatReader:
    def __init__(self, path: str, clock: Callable[[], float] = time.monotonic):
        self._path = path
        self._clock = clock
        self._board: Optional[mmap.mmap] = None

    def _attach(self) -> bool:
        if self._board is not None:
            return True
        if not os.path.exists(self._path):
            return False
        fd = os.open(self._path, os.O_RDONLY)
        try:
            if os.fstat(fd).st_size < FILE_SIZE:
                return False
            self._board = mmap.mmap(fd, FILE_SIZE, access=mmap.ACCESS_READ)
        finally:
            os.close(fd)
        magic, version, capacity, slot_size = HEADER_STRUCT.unpack_from(self._board, 0)
        if (magic, version, capacity, slot_size) != (
            MAGIC,
            VERSION,
            SLOT_CAPACITY,
            SLOT_SIZE,
        ):
            logger.warning("heartbeat board %s has an unexpected header", self._path)
            self._board.close()
            self._board = None
            return False
        return True

    def read(self) -> List[HeartbeatSlot]:
        """Every slot that has ever been written, with its age now."""
        if not self._attach():
            return []
        now = self._clock()
        slots = []
        for i in range(SLOT_CAPACITY):
            role, paused, rank, beat_ts, _started = SLOT_STRUCT.unpack_from(
                self._board, _slot_offset(i)
            )
            if beat_ts == 0.0:
                continue
            slots.append(
                HeartbeatSlot(
                    role=ROLE_NAMES.get(role, str(role)),
                    rank=rank,
                    age_s=round(max(0.0, now - beat_ts), 3),
                    paused=bool(paused),
                )
            )
        return slots

    def stale(self, max_age_s: float) -> List[HeartbeatSlot]:
        return [s for s in self.read() if not s.paused and s.age_s > max_age_s]

    def close(self) -> None:
        if self._board is not None:
            self._board.close()
            self._board = None


def create_heartbeat_writer(
    instance_id: str, *, role: int, slot: int, rank: int
) -> Optional[HeartbeatWriter]:
    if not envs.SGLANG_ENABLE_HEALTH_HEARTBEAT.get() or not instance_id:
        return None
    try:
        return HeartbeatWriter(
            heartbeat_shm_path(instance_id), role=role, slot=slot, rank=rank
        )
    except (OSError, ValueError) as e:
        logger.warning("heartbeat writer disabled: %s", e)
        return None


def create_heartbeat_reader(instance_id: str) -> Optional[HeartbeatReader]:
    if not envs.SGLANG_ENABLE_HEALTH_HEARTBEAT.get() or not instance_id:
        return None
    return HeartbeatReader(heartbeat_shm_path(instance_id))
