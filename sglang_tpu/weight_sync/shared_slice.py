# SPDX-License-Identifier: Apache-2.0
"""Share one TPU slice with an RL trainer, so weights move between them over ICI.

SGLANG_TPU_SHARED_SLICE="<job slice-rank offset>,<slice size>,<store host>:<store port>"
is set on every engine process (its offset follows the trainer and the engines
before it; the trainer gets miles' MILES_TPU_SHARED_SLICE), with the slice-wide
TORCH_TPU_SLICEBUILDER_ADDRESSES and TORCH_TPU_TOPOLOGY. libtpu builds the slice,
trainer included, from the slice-wide RANK / WORLD_SIZE, while each job keeps its
own torch.distributed world. Needs a torch_tpu with job-local process groups
(TORCH_TPU_JOB_DEVICE_IDS), so no job compiles programs over another's devices.
The trainer's side is miles/platforms/tpu_shared_slice.py.
"""

from __future__ import annotations

import math
import os
from datetime import timedelta
from typing import Any, Optional

import msgspec
import torch.distributed as dist

from sglang.srt.utils.common import temp_set_env
from sglang_tpu._logging import get_logger
from sglang_tpu.envs import envs as tpu_envs

logger = get_logger(__name__)

# Arbitrary; long enough for the other job to start its TPU runtime.
_JOIN_TIMEOUT = timedelta(minutes=30)
# Must match the trainer's: both jobs create the slice's backend on this namespace.
_STORE_PREFIX = "tpu_shared_slice"


class SharedSliceSpec(msgspec.Struct, frozen=True):
    rank_offset: int
    size: int
    store_host: str
    store_port: int

    @classmethod
    def from_env(cls) -> Optional[SharedSliceSpec]:
        value = tpu_envs.SGLANG_TPU_SHARED_SLICE.get()
        if not value:
            return None
        offset, size, address = value.split(",")
        host, port = address.rsplit(":", 1)
        return cls(
            rank_offset=int(offset),
            size=int(size),
            store_host=host,
            store_port=int(port),
        )


class SharedSlice(msgspec.Struct, frozen=True):
    """This process's place in the shared slice and its send/recv backend."""

    spec: SharedSliceSpec
    slice_rank: int
    # torch_tpu's c10d backend spanning the whole slice
    backend: Any


def join(spec: SharedSliceSpec, *, job_rank: int, job_size: int) -> SharedSlice:
    """Join the slice and create its send/recv backend, before the first TPU op.

    libtpu reads the slice-wide RANK / WORLD_SIZE when the runtime starts, and
    torch_tpu fixes the job's devices (TORCH_TPU_JOB_DEVICE_IDS) at its first
    compile; the job's own RANK / WORLD_SIZE are restored afterwards.
    """
    # torch_tpu must load after _prepare_tpu_env has set libtpu's flags
    from torch_tpu._internal.distributed import tpu_distributed

    addresses = os.environ.get("TORCH_TPU_SLICEBUILDER_ADDRESSES", "").split(",")
    if len(addresses) != spec.size:
        raise ValueError(
            f"SGLANG_TPU_SHARED_SLICE names a slice of {spec.size} devices, but "
            f"TORCH_TPU_SLICEBUILDER_ADDRESSES lists {addresses}: both jobs need "
            "the same slice-wide addresses"
        )
    topology = os.environ.get("TORCH_TPU_TOPOLOGY", "")
    if math.prod(int(dim) for dim in topology.split(",") if dim) != spec.size:
        raise ValueError(
            f"SGLANG_TPU_SHARED_SLICE names a slice of {spec.size} devices, but "
            f"TORCH_TPU_TOPOLOGY is {topology!r}: set the slice's topology on every job"
        )
    slice_rank = spec.rank_offset + job_rank
    with temp_set_env(
        RANK=slice_rank,
        LOCAL_RANK=slice_rank,
        WORLD_SIZE=spec.size,
        LOCAL_WORLD_SIZE=spec.size,
    ):
        # starts the runtime, which joins the slice
        device_id = tpu_distributed.global_device_id()
        store = dist.TCPStore(
            spec.store_host,
            spec.store_port,
            spec.size,
            is_master=slice_rank == 0,
            timeout=_JOIN_TIMEOUT,
        )
        store.set(f"device:{slice_rank}", str(device_id))
        os.environ["TORCH_TPU_JOB_DEVICE_IDS"] = ",".join(
            store.get(f"device:{spec.rank_offset + rank}").decode()
            for rank in range(job_size)
        )
        backend = tpu_distributed.create_process_group(
            dist.PrefixStore(_STORE_PREFIX, store),
            slice_rank,
            spec.size,
            _JOIN_TIMEOUT,
        )
    logger.info(
        "Joined the shared TPU slice as rank %d of %d (job devices %s)",
        slice_rank,
        spec.size,
        os.environ["TORCH_TPU_JOB_DEVICE_IDS"],
    )
    return SharedSlice(spec=spec, slice_rank=slice_rank, backend=backend)
