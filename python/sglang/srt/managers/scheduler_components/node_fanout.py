"""Two-level request broadcast for multi-node TP groups.

The tokenizer puts multimodal tensors in shared memory on the source node.
Node leaders exchange the request metadata and a packed buffer of tensor bytes.
Each remote leader receives into a shared-memory segment and repoints the
handles before broadcasting the small requests to its local ranks.
Tensors that could not be put in shared memory keep using the existing tensor
buffer broadcast, including within each node.

Every rank maps the segments copy-on-write in ``_finalize_shm_features``, which
also unlinks them after the TP-wide barrier. Ranks must be ordered contiguously
by node, with the same number of TP ranks on each node.
"""

from __future__ import annotations

import logging
import math
import mmap
import os
import sys
from multiprocessing import shared_memory
from typing import Any, List, Optional

import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.mm_utils import ShmPointerMMData
from sglang.srt.runtime_context import get_parallel, get_serving
from sglang.srt.utils import broadcast_pyobj, print_warning_once
from sglang.srt.utils.stale_shm_cleanup import make_shm_name

logger = logging.getLogger(__name__)

_BROADCASTERS: dict[tuple[int, ...], NodeFanoutBroadcaster] = {}


def node_fanout_enabled() -> bool:
    """Use the same transport decision in the tokenizer and every scheduler.

    Group creation requires one TP group spanning the default process group.
    DP attention and pipeline parallelism retain their existing transport.
    Pre-tokenized requests also retain it because they skip SHM materialization.
    """
    if not envs.SGLANG_ENABLE_NODE_LEADER_REQUEST_BROADCAST.get():
        return False

    parallel = get_parallel()
    if parallel.nnodes <= 1:
        reason = "requires nnodes > 1"
    elif parallel.tp_size <= 1:
        reason = "requires tp_size > 1"
    elif parallel.pp_size != 1:
        reason = "requires pp_size == 1"
    elif parallel.tp_size % parallel.nnodes != 0:
        reason = "requires tp_size to be divisible by nnodes"
    elif parallel.attn_dp_enabled:
        reason = "does not support DP attention"
    elif get_serving().skip_tokenizer_init:
        reason = "requires tokenizer initialization"
    else:
        return True

    print_warning_once(
        "SGLANG_ENABLE_NODE_LEADER_REQUEST_BROADCAST is set, but node fan-out "
        f"request broadcast is disabled: {reason}. Using the existing transport."
    )
    return False


def _tokenized(reqs: List[Any]) -> List[Any]:
    out = []
    for req in reqs:
        if isinstance(req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)):
            out.append(req)
        elif isinstance(
            req, (BatchTokenizedGenerateReqInput, BatchTokenizedEmbeddingReqInput)
        ):
            out.extend(req.batch)
    return out


class NodeFanoutBroadcaster:
    def __init__(self, ranks: List[int], rank_in_group: int, nnodes: int):
        assert len(ranks) % nnodes == 0, (len(ranks), nnodes)
        per_node = len(ranks) // nnodes
        # new_group is collective over the default group: every rank creates
        # every group, in the same order.
        node_groups = [
            dist.new_group(ranks[i * per_node : (i + 1) * per_node], backend="gloo")
            for i in range(nnodes)
        ]
        self._leader_group = dist.new_group(ranks[::per_node], backend="gloo")
        node = rank_in_group // per_node
        self._node_group = node_groups[node]
        self._rank = ranks[rank_in_group]
        self._src = ranks[0]
        self._node_src = ranks[node * per_node]
        self._is_leader = rank_in_group % per_node == 0

    def broadcast(self, reqs: Optional[List[Any]]) -> List[Any]:
        if not self._is_leader:
            return broadcast_pyobj(
                None, self._rank, self._node_group, src=self._node_src
            )
        reqs = broadcast_pyobj(reqs, self._rank, self._leader_group, src=self._src)
        tokenized = _tokenized(reqs)
        self._forward_shm_tensors(tokenized)
        # Keep inline fallbacks inline on every node. Retrying SHM allocation
        # independently could leave SHM on only some nodes, causing ranks to
        # disagree on whether to enter the finalization barrier.
        # Source pointers opened during IPC decoding; remote pointers attach in
        # in_segment. Both leaders and local receivers can now finalize them.
        return broadcast_pyobj(reqs, self._rank, self._node_group, src=self._node_src)

    def _forward_shm_tensors(self, tokenized: List[Any]) -> None:
        """Copy rank 0's /dev/shm tensors into each other node's /dev/shm."""
        slots = _shm_slots(tokenized)
        if not slots:
            return
        sizes = [_nbytes(slot.get()) for slot in slots]
        offsets, total = [], 0
        for size in sizes:
            offsets.append(total)
            total += _page_align(size)
        total = max(total, 1)
        if self._rank == self._src:
            # Pack the tokenizer's mappings at the receivers' offsets for one
            # Gloo call per request batch. A segment that failed to open sends
            # zeros so the leaders stay in step. _finalize_shm_features then
            # rejects the request on every rank.
            packed = torch.empty(total, dtype=torch.uint8)
            for slot, offset, size in zip(slots, offsets, sizes):
                tensor = slot.get().tensor
                dst = packed[offset : offset + size]
                if tensor is None:
                    dst.zero_()
                else:
                    dst.copy_(tensor.view(torch.uint8).reshape(-1))
                del tensor, dst
            dist.broadcast(packed, src=self._src, group=self._leader_group)
            return
        shm = _allocate_segment(total)
        if shm is None:
            # Stay in step with the other leaders, then leave pointers that fail
            # to open, so _finalize_shm_features rejects the request on all ranks.
            dist.broadcast(
                torch.empty(total, dtype=torch.uint8),
                src=self._src,
                group=self._leader_group,
            )
            for slot in slots:
                pointer = slot.get()
                slot.set(
                    ShmPointerMMData.in_segment(
                        make_shm_name("mm-unallocated"), 0, pointer.shape, pointer.dtype
                    )
                )
            return
        try:
            # Receive straight into this node's segment: no extra copy.
            segment = torch.frombuffer(shm.buf, dtype=torch.uint8)
            dist.broadcast(segment[:total], src=self._src, group=self._leader_group)
            del segment
            for slot, offset in zip(slots, offsets):
                pointer = slot.get()
                slot.set(
                    ShmPointerMMData.in_segment(
                        shm.name,
                        offset,
                        pointer.shape,
                        pointer.dtype,
                        pointer.precomputed_hash,
                    )
                )
        finally:
            shm.close()


def _allocate_segment(size: int) -> Optional[shared_memory.SharedMemory]:
    try:
        shm = shared_memory.SharedMemory(
            create=True, size=size, name=make_shm_name("mm")
        )
    except OSError:
        logger.warning("Node fan-out could not create a %d-byte /dev/shm segment", size)
        return None
    if sys.platform == "linux":
        try:
            # Reserve tmpfs pages now: a full /dev/shm is an OSError here rather
            # than a SIGBUS mid-receive.
            os.posix_fallocate(shm._fd, 0, size)  # type: ignore[attr-defined]
        except OSError:
            logger.warning("Node fan-out could not reserve %d bytes of /dev/shm", size)
            shm.close()
            shm.unlink()
            return None
    return shm


class _Slot:
    """One SHM pointer inside an mm item: ``item.attr`` or ``item.attr[index]``."""

    def __init__(self, item, attr: str, index: Optional[int] = None):
        self.item, self.attr, self.index = item, attr, index

    def get(self) -> ShmPointerMMData:
        value = getattr(self.item, self.attr)
        return value if self.index is None else value[self.index]

    def set(self, pointer: ShmPointerMMData) -> None:
        if self.index is None:
            setattr(self.item, self.attr, pointer)
            return
        value = getattr(self.item, self.attr)
        items = list(value)
        items[self.index] = pointer
        setattr(self.item, self.attr, type(value)(items))


def _shm_slots(tokenized: List[Any]) -> List[_Slot]:
    slots = []
    for req in tokenized:
        if not req.mm_inputs:
            continue
        for item in req.mm_inputs.mm_items:
            for attr in ("feature", "precomputed_embeddings"):
                value = getattr(item, attr, None)
                if isinstance(value, ShmPointerMMData):
                    slots.append(_Slot(item, attr))
                elif isinstance(value, (list, tuple)):
                    slots.extend(
                        _Slot(item, attr, i)
                        for i, v in enumerate(value)
                        if isinstance(v, ShmPointerMMData)
                    )
    return slots


def _nbytes(pointer: ShmPointerMMData) -> int:
    return math.prod(pointer.shape) * pointer.dtype.itemsize


def _page_align(size: int) -> int:
    page = mmap.ALLOCATIONGRANULARITY
    return (size + page - 1) // page * page


def get_node_fanout(tp_group, nnodes: int) -> NodeFanoutBroadcaster:
    key = tuple(tp_group.ranks)
    broadcaster = _BROADCASTERS.get(key)
    if broadcaster is None:
        broadcaster = NodeFanoutBroadcaster(
            list(tp_group.ranks), tp_group.rank_in_group, nnodes
        )
        _BROADCASTERS[key] = broadcaster
        logger.info("Node fan-out request broadcast enabled over %d nodes", nnodes)
    return broadcaster
