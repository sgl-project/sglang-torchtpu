# SPDX-License-Identifier: Apache-2.0
"""Share one TPU slice with an RL trainer, so weights move between them over ICI.

Enabled by SGLANG_TPU_SHARED_SLICE="<slice rank offset>,<slice size>,<store host>:<store port>".
The trainer and this engine are two jobs of one slice: libtpu builds the slice from the slice-wide
RANK / WORLD_SIZE, while each job's torch.distributed world keeps its own ranks. Needs a torch_tpu
build with job-local worlds (TORCH_TPU_JOB_DEVICE_IDS), so neither job waits on the other's
programs, and a libtpu without the 0.0.49 SparseCore regression (0.0.48 works).
"""

import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import PrefixStore

from sglang.srt.platforms.interface import WeightUpdateReceiver

logger = logging.getLogger(__name__)

ENV_VAR = "SGLANG_TPU_SHARED_SLICE"
BACKEND_NAME = "tpu_shared_slice"
TRANSFER_ALIGN_BYTES = 4096
_TIMEOUT = timedelta(minutes=30)


@dataclass(frozen=True)
class SharedSlice:
    rank_offset: int
    size: int
    store_host: str
    store_port: int


@dataclass(frozen=True)
class JobRank:
    rank: int
    size: int


def spec() -> Optional[SharedSlice]:
    value = os.environ.get(ENV_VAR)
    if not value:
        return None
    offset, size, address = value.split(",")
    host, port = address.rsplit(":", 1)
    return SharedSlice(int(offset), int(size), host, int(port))


def enter_slice_env(shared: SharedSlice) -> JobRank:
    """Swap the worker's job RANK / WORLD_SIZE for the slice-wide ones libtpu reads at startup."""
    job = JobRank(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]))
    os.environ["RANK"] = os.environ["LOCAL_RANK"] = str(shared.rank_offset + job.rank)
    os.environ["WORLD_SIZE"] = os.environ["LOCAL_WORLD_SIZE"] = str(shared.size)
    return job


class SharedSliceReceiver(WeightUpdateReceiver):
    """Receives each update bucket from the trainer rank paired with this TP rank."""

    def __init__(
        self,
        backend: dist.ProcessGroup,
        store: dist.Store,
        src_slice_rank: int,
        slice_rank: int,
    ):
        self._backend = backend
        self._store = store
        self._src_slice_rank = src_slice_rank
        self._slice_rank = slice_rank
        self._next_tag = 0

    def receive(
        self,
        names: list[str],
        dtypes: list[str | torch.dtype],
        shapes: list[list[int]],
        device: torch.device | str,
    ) -> list[tuple[str, torch.Tensor]]:
        """Ignores names (trainer rank 0's): the paired rank posts its own, as its experts differ."""
        dtypes = [_dtype(dtype) for dtype in dtypes]
        numels = [math.prod(shape) for shape in shapes]
        regions = packed_regions(numels, dtypes)
        offset, size = regions[-1]
        # the trainer packs a bucket into one int16 buffer: many small sends crash libtpu's
        # CopyToRemoteDevice, single large ones are fine
        packed = torch.empty((offset + size) // 2, dtype=torch.int16, device=device)
        # torch_tpu keys a transfer by (src, dst, tag) and never forgets a key: tags must not repeat
        self._next_tag += 1
        self._backend.recv([packed], self._src_slice_rank, self._next_tag).wait()
        # wait() returns once the buffer is announced, not once the data lands; a read waits for it,
        # and freeing the buffer before that crashes the sender
        packed[:1].cpu()
        key = bucket_names_key(self._src_slice_rank, self._slice_rank, self._next_tag)
        own_names = json.loads(self._store.get(key))
        self._store.delete_key(key)
        assert len(own_names) == len(shapes), (len(own_names), len(shapes))
        return [
            (
                name,
                packed[offset // 2 : (offset + numel * dtype.itemsize) // 2]
                .view(dtype)
                .view(shape),
            )
            for name, dtype, shape, numel, (offset, _) in zip(
                own_names, dtypes, shapes, numels, regions
            )
        ]


def packed_regions(
    numels: list[int], dtypes: list[torch.dtype]
) -> list[tuple[int, int]]:
    """(byte offset, padded byte size) of each tensor in a packed bucket: whole 4 KiB regions.

    The trainer lays its buffer out the same way.
    """
    regions, offset = [], 0
    for numel, dtype in zip(numels, dtypes):
        blocks = max(1, math.ceil(numel * dtype.itemsize / TRANSFER_ALIGN_BYTES))
        size = blocks * TRANSFER_ALIGN_BYTES
        regions.append((offset, size))
        offset += size
    return regions


def _dtype(dtype: str | torch.dtype) -> torch.dtype:
    return dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)


@dataclass
class _State:
    backend: Optional[dist.ProcessGroup] = None
    store: Optional[dist.Store] = None
    slice_rank: int = -1


# a worker process joins the slice once, at backend init, before any weight-update group exists
_state = _State()


def join(shared: SharedSlice, job: JobRank) -> None:
    """Join the slice and open the slice-wide send/recv backend; restores the job's RANK / WORLD_SIZE.

    Runs before the first TPU program: the job-local world is fixed at the first compile.
    """
    from torch_tpu._internal.distributed import tpu_distributed

    slice_rank = shared.rank_offset + job.rank
    # starts the runtime, which joins the slice
    device_id = tpu_distributed.global_device_id()
    store = dist.TCPStore(
        shared.store_host, shared.store_port, shared.size, slice_rank == 0, _TIMEOUT
    )
    store.set(f"device:{slice_rank}", str(device_id))
    os.environ["TORCH_TPU_JOB_DEVICE_IDS"] = ",".join(
        store.get(f"device:{shared.rank_offset + rank}").decode()
        for rank in range(job.size)
    )
    _state.backend = tpu_distributed.create_process_group(
        PrefixStore(BACKEND_NAME, store), slice_rank, shared.size, _TIMEOUT
    )
    _state.store, _state.slice_rank = store, slice_rank
    os.environ["RANK"] = os.environ["LOCAL_RANK"] = str(job.rank)
    os.environ["WORLD_SIZE"] = os.environ["LOCAL_WORLD_SIZE"] = str(job.size)
    logger.info(
        "Joined the shared TPU slice as rank %d of %d (job devices %s)",
        slice_rank,
        shared.size,
        os.environ["TORCH_TPU_JOB_DEVICE_IDS"],
    )


def joined() -> bool:
    return _state.backend is not None


def receiver(tp_rank: int) -> SharedSliceReceiver:
    """The trainer's rank i (slice rank i) sends to TP rank i."""
    assert _state.backend is not None, f"{ENV_VAR} is unset or the slice was not joined"
    return SharedSliceReceiver(
        _state.backend,
        _state.store,
        src_slice_rank=tp_rank,
        slice_rank=_state.slice_rank,
    )


def bucket_names_key(src_slice_rank: int, dst_slice_rank: int, tag: int) -> str:
    """Store key under which the trainer posts a bucket's names."""
    return f"bucket_names:{src_slice_rank}:{dst_slice_rank}:{tag}"
