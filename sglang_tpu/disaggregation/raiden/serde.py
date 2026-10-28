"""Raiden wire messages and auxiliary-buffer serialization.

Each typed Raiden message uses the shared ``[TAG, JSON payload]`` framing.
Pipeline messages and ``ABORT`` retain their positional layouts.
"""

from __future__ import annotations

import ctypes
import enum
import struct
from typing import ClassVar, List

import msgspec

from sglang_tpu.disaggregation.tpu.wire import WireMessage, require_non_negative


class MessageTag(bytes, enum.Enum):
    """Identify the message type in the first ZMQ frame.

    The bytes-valued members can be sent directly and compared with the first
    frame received from a per-rank PULL socket.
    """

    # Symmetric-TP pull path (decode TP == prefill TP).
    PULL_REQ = b"RAIDEN_PULL_REQ"  # decode -> prefill: "send me room R's blocks"
    BLOCKS = b"RAIDEN_BLOCKS"  # prefill -> decode: block-ids + endpoint + aux
    CHUNK = b"RAIDEN_CHUNK"  # prefill -> decode: one chunk's pages, mid-prefill

    PIPELINE_REQ = b"RAIDEN_PIPELINE_REQ"
    PIPELINE_READY = b"RAIDEN_PIPELINE_READY"
    PIPELINE_RELEASE = b"RAIDEN_PIPELINE_RELEASE"

    # Asymmetric-TP push-resharding path (decode TP < prefill TP).
    RESHARD_REQ = b"RAIDEN_RESHARD_REQ"  # decode -> prefill: endpoint + blocks + tp
    RESHARD_BLOCKS = b"RAIDEN_RESHARD_BLOCKS"  # prefill -> decode: blocks + rank + aux
    RESHARD_READY = b"RAIDEN_RESHARD_READY"  # decode -> prefill: registered, push now
    RESHARD_CHUNK = (
        b"RAIDEN_RESHARD_CHUNK"  # prefill -> decode: one chunk's pages, mid-prefill
    )
    RESHARD_CHUNK_READY = (
        b"RAIDEN_RESHARD_CHUNK_READY"  # decode -> prefill: chunk registered, push it
    )

    ROOM_FAILED = b"RAIDEN_ROOM_FAILED"  # prefill -> decode: room already failed

    # CommonKVReceiver owns this backend-independent positional message.
    ABORT = b"ABORT"  # decode -> prefill: room, decode ip, decode port


def _require_port(port: int) -> None:
    if not 1 <= port <= 65535:
        raise ValueError(f"invalid decode_port: expected 1..65535, got {port}")


class PullReq(WireMessage):
    """Ask a prefill rank to publish one room's KV blocks."""

    TAG: ClassVar[bytes] = MessageTag.PULL_REQ
    room: int
    decode_ip: str
    decode_port: int
    decode_pages: int
    # Window pages the decode allocated for the room; the sender registers
    # only the window blocks it will read.
    decode_swa_pages: int = 0

    def __post_init__(self) -> None:
        require_non_negative(self.room, "room")
        require_non_negative(self.decode_pages, "decode_pages")
        require_non_negative(self.decode_swa_pages, "decode_swa_pages")
        _require_port(self.decode_port)


class Blocks(WireMessage):
    """Return symmetric-TP KV block locations and auxiliary data."""

    TAG: ClassVar[bytes] = MessageTag.BLOCKS
    room: int
    blocks: list[int]
    endpoint: str
    swa_blocks: list[int]
    swa_endpoint: str
    aux_blob: bytes
    # The prefill's committed state blocks and its state engine endpoint.
    state_blocks: list[int] = msgspec.field(default_factory=list)
    state_endpoint: str = ""
    # Optional opaque host state metadata (e.g., DSV4 ring shifts).
    state_meta: bytes = b""

    def __post_init__(self) -> None:
        require_non_negative(self.room, "room")


class Chunk(WireMessage):
    """Publish one chunk of a symmetric-TP prefill transfer."""

    TAG: ClassVar[bytes] = MessageTag.CHUNK
    room: int
    req_id: str
    uuid: int
    start: int
    total_pages: int
    blocks: list[int]
    endpoint: str

    def __post_init__(self) -> None:
        require_non_negative(self.room, "room")
        require_non_negative(self.uuid, "uuid")
        require_non_negative(self.start, "start")
        require_non_negative(self.total_pages, "total_pages")


class ReshardReq(WireMessage):
    """Request a push transfer into an asymmetric-TP decode layout."""

    TAG: ClassVar[bytes] = MessageTag.RESHARD_REQ
    room: int
    decode_ip: str
    decode_port: int
    push_endpoint: str
    decode_blocks: list[int]
    dst_attn_tp: int
    decode_swa_blocks: list[int]
    swa_push_endpoint: str

    def __post_init__(self) -> None:
        require_non_negative(self.room, "room")
        _require_port(self.decode_port)
        if self.dst_attn_tp < 1:
            raise ValueError(
                f"invalid dst_attn_tp: expected >= 1, got {self.dst_attn_tp}"
            )


class ReshardBlocks(WireMessage):
    """Publish source blocks for a complete resharded transfer."""

    TAG: ClassVar[bytes] = MessageTag.RESHARD_BLOCKS
    room: int
    prefill_blocks: list[int]
    src_rank: int
    aux_blob: bytes
    prefill_swa_blocks: list[int]

    def __post_init__(self) -> None:
        require_non_negative(self.room, "room")
        require_non_negative(self.src_rank, "src_rank")


class ReshardReady(WireMessage):
    """Confirm that decode registered the destination reshard plan."""

    TAG: ClassVar[bytes] = MessageTag.RESHARD_READY
    room: int
    pages: list[int] = msgspec.field(default_factory=list)

    def __post_init__(self) -> None:
        require_non_negative(self.room, "room")


class ReshardChunk(WireMessage):
    """Publish source blocks for one chunk of a resharded transfer."""

    TAG: ClassVar[bytes] = MessageTag.RESHARD_CHUNK
    room: int
    index: int
    start: int
    total_pages: int
    blocks: list[int]
    src_rank: int

    def __post_init__(self) -> None:
        require_non_negative(self.room, "room")
        require_non_negative(self.index, "index")
        require_non_negative(self.start, "start")
        require_non_negative(self.total_pages, "total_pages")
        require_non_negative(self.src_rank, "src_rank")


class ReshardChunkReady(WireMessage):
    """Confirm that decode registered one chunk's reshard plan."""

    TAG: ClassVar[bytes] = MessageTag.RESHARD_CHUNK_READY
    room: int
    index: int
    uuid: int

    def __post_init__(self) -> None:
        require_non_negative(self.room, "room")
        require_non_negative(self.index, "index")
        require_non_negative(self.uuid, "uuid")


class RoomFailed(WireMessage):
    """Tell a decode rank that the prefill rank has already failed a room.

    The prefill sends it when a request for the room's KV (PULL_REQ,
    RESHARD_REQ or PIPELINE_REQ) finds the room tombstoned. It is the mirror
    of ABORT.
    """

    TAG: ClassVar[bytes] = MessageTag.ROOM_FAILED
    room: int
    reason: str

    def __post_init__(self) -> None:
        require_non_negative(self.room, "room")


def decode_text(frame: bytes, field: str) -> str:
    """Decode a named ASCII frame from the legacy ABORT message."""
    try:
        return frame.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid {field}: expected ASCII text") from exc


def read_aux_blob(
    aux_data_ptrs: List[int], aux_item_lens: List[int], aux_index: int
) -> bytes:
    """Serialize every aux buffer's slot at ``aux_index`` into one blob.

    Aux buffers (first sampled token, logprobs, bootstrap_room, ...) live on
    host (CPU) memory, so the slots are read directly via their data pointers.
    """
    parts: List[bytes] = []
    for ptr, item_len in zip(aux_data_ptrs, aux_item_lens):
        addr = ptr + item_len * aux_index
        parts.append(bytes((ctypes.c_byte * item_len).from_address(addr)))
    out = struct.pack("<I", len(parts))
    for p in parts:
        out += struct.pack("<I", len(p)) + p
    return out


def write_aux_blob(
    aux_data_ptrs: List[int], aux_item_lens: List[int], aux_index: int, blob: bytes
) -> None:
    """Inverse of ``read_aux_blob``: scatter the blob into local aux slots."""
    off = 0
    (count,) = struct.unpack_from("<I", blob, off)
    off += 4
    for i in range(count):
        (ln,) = struct.unpack_from("<I", blob, off)
        off += 4
        data = blob[off : off + ln]
        off += ln
        addr = aux_data_ptrs[i] + aux_item_lens[i] * aux_index
        dst = (ctypes.c_byte * ln).from_address(addr)
        dst[:] = data
