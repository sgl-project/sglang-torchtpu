"""Wire-format helpers for the raiden disaggregation rendezvous.

Kept free of heavy deps (no zmq/torch/sglang) so the byte protocol can be unit
tested in isolation.
"""

from __future__ import annotations

import ctypes
import struct
from typing import List

import numpy as np

# ZMQ multipart headers exchanged over the existing per-rank PULL sockets.
PULL_REQ = b"RAIDEN_PULL_REQ"  # decode -> prefill: "send me room R's blocks"
BLOCKS = b"RAIDEN_BLOCKS"  # prefill -> decode: block-ids + endpoint + aux


def pack_blocks(block_ids: List[int]) -> bytes:
    return np.asarray(block_ids, dtype=np.int64).tobytes()


def unpack_blocks(buf: bytes) -> List[int]:
    return np.frombuffer(buf, dtype=np.int64).tolist()


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
