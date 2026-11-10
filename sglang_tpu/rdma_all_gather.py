# SPDX-License-Identifier: Apache-2.0
"""Direct Pallas remote-DMA all-gather ordered by PyTorch distributed rank.

Replaces XLA's reg_all_gather_into_tensor and the zero-padded all-reduce workaround
(sglang_tpu.all_gather_via_all_reduce) in Dynamo-traced TPU graphs.

Under Data Parallelism with DP attention, each rank produces its local shard of
tokens and gathers the full sequence across all TP/DP ranks. In all_gather_via_all_reduce,
each rank wrote into a zero [world, *block] buffer and summed, consuming ~108 ms
per prefill step on 8 chips.

This kernel writes each rank's shard directly from VMEM into every peer's HBM
output buffer in a single hop using Mosaic async remote DMAs (make_async_remote_copy).
The output slot is explicitly mapped from PJRT mesh coordinates to PyTorch rank
order using the physical device assignment, ensuring bit-exact rank ordering
regardless of Hamiltonian torus ring permutation.
"""

import functools
import logging
from collections.abc import Callable, Sequence
from threading import Lock

import torch

logger = logging.getLogger(__name__)

_COLLECTIVE_ID = 11
RDMA_AG_MAX_SINGLE_BYTES = 16 * 1024 * 1024
RDMA_AG_MAX_TOTAL_BYTES = 128 * 1024 * 1024

_OPS: dict[int, Callable] = {}
_OPS_LOCK = Lock()
_RDMA_AG_WORLD: int = 0


def _rdma_all_gather_kernel(
    x_ref,
    out_ref,
    local_sem,
    send_semaphores,
    receive_semaphores,
    *,
    world_size: int,
    slot_for_mesh_rank: tuple[int, ...],
) -> None:
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    mesh_rank = lax.axis_index("model").astype(jnp.int32)
    # Entry barrier: no peer writes into this device's output before it has entered
    # the kernel, so XLA buffer reuse around the call cannot race remote DMAs.
    barrier = pltpu.get_barrier_semaphore()
    for offset in range(1, world_size):
        pl.semaphore_signal(
            barrier,
            1,
            device_id=(jnp.mod(mesh_rank + offset, world_size),),
            device_id_type=pl.DeviceIdType.MESH,
        )
    pl.semaphore_wait(barrier, world_size - 1)

    # Compute target output slot corresponding to this mesh rank.
    # Use scalar lax.select chain (Mosaic TPU does not support dynamic indexing into 1D registers).
    slot = jnp.int32(slot_for_mesh_rank[0])
    for idx in range(1, world_size):
        slot = lax.select(
            mesh_rank == jnp.int32(idx),
            jnp.int32(slot_for_mesh_rank[idx]),
            slot,
        )

    local_copy = pltpu.make_async_copy(x_ref, out_ref.at[slot], local_sem)
    local_copy.start()

    copies = []
    for offset in range(1, world_size):
        peer = jnp.mod(mesh_rank + offset, world_size)
        copy = pltpu.make_async_remote_copy(
            x_ref,
            out_ref.at[slot],
            send_semaphores.at[offset - 1],
            receive_semaphores.at[offset - 1],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copies.append(copy)

    local_copy.wait()
    for copy in copies:
        copy.wait()


def rdma_all_gather_pallas(
    x,
    *,
    world_size: int,
    slot_for_mesh_rank: tuple[int, ...],
    collective_id: int = _COLLECTIVE_ID,
):
    """Gather ``x`` [rows, cols] from every mesh rank into [world_size, rows, cols] in HBM."""
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows, cols = x.shape
    kernel = functools.partial(
        _rdma_all_gather_kernel,
        world_size=world_size,
        slot_for_mesh_rank=slot_for_mesh_rank,
    )
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((world_size, rows, cols), x.dtype),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        scratch_shapes=(
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((world_size - 1,)),
            pltpu.SemaphoreType.DMA((world_size - 1,)),
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id,
            vmem_limit_bytes=32 * 1024 * 1024,
        ),
        name=f"rdma_all_gather_tp{world_size}",
    )(x)


@functools.cache
def _get_mesh(world_size: int):
    from torch_tpu._internal.pallas import get_pallas_mesh

    return get_pallas_mesh(("model",), (world_size,))


def _build_op(world_size: int, slot_for_mesh_rank: tuple[int, ...]) -> Callable:
    import jax
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    mesh = _get_mesh(world_size)
    sharded = jax.shard_map(
        functools.partial(
            rdma_all_gather_pallas,
            world_size=world_size,
            slot_for_mesh_rank=slot_for_mesh_rank,
        ),
        mesh=mesh,
        in_specs=(P("model"),),
        out_specs=P(),
        check_vma=False,
    )

    def _wrapped(x: jax.Array) -> jax.Array:
        return sharded(x)

    perm_tag = "_".join(str(s) for s in slot_for_mesh_rank)
    return cpu_export_jax_op(
        f"sglang_tpu_pallas::rdma_all_gather_tp{world_size}_{perm_tag}",
        _wrapped,
        mesh=mesh,
        input_partition_specs=(P("model"),),
        output_partition_specs=(P(),),
    )


def register_rdma_all_gather(
    world_size: int,
    rank_to_device: Sequence[int] | None = None,
) -> Callable:
    """Pre-build and register the RDMA all-gather custom op before torch.compile tracing."""
    global _RDMA_AG_WORLD
    op = _OPS.get(world_size)
    if op is not None:
        return op
    with _OPS_LOCK:
        op = _OPS.get(world_size)
        if op is None:
            mesh = _get_mesh(world_size)
            if rank_to_device is None:
                from torch_tpu._internal.distributed.device_mesh import (
                    get_device_assignment,
                )

                rank_to_device = [int(v) for v in get_device_assignment()]
            slot_for_mesh_rank = tuple(
                rank_to_device.index(int(d.id)) for d in mesh.devices.flat
            )
            op = _build_op(world_size, slot_for_mesh_rank)
            _OPS[world_size] = op
            _RDMA_AG_WORLD = world_size
            logger.info(
                "Registered Pallas RDMA all-gather: world=%d, slot_for_mesh_rank=%s, rank_to_device=%s",
                world_size,
                slot_for_mesh_rank,
                rank_to_device,
            )
    return op


def _own_storage(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor owns standalone contiguous storage for PJRT/Pallas."""
    if torch.compiler.is_compiling():
        return tensor.contiguous()
    if (
        not tensor.is_contiguous()
        or tensor.storage_offset() != 0
        or tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size()
    ):
        return tensor.clone()
    return tensor


def can_use_rdma_all_gather(group, block: torch.Tensor) -> bool:
    """Return True if block can be gathered via direct Pallas RDMA all-gather."""
    if _RDMA_AG_WORLD <= 1 or group.world_size != _RDMA_AG_WORLD:
        return False
    if block.device.type != "tpu":
        return False
    from sglang.srt.distributed.parallel_state import get_tp_group

    if group is not get_tp_group():
        return False
    if (
        block.dtype not in (torch.bfloat16, torch.float32)
        or block.dim() < 1
        or block.shape[0] <= 0
    ):
        return False

    rows = block.shape[0]
    cols = block.numel() // rows
    nbytes = block.numel() * block.element_size()
    num_chunks = (nbytes + RDMA_AG_MAX_SINGLE_BYTES - 1) // RDMA_AG_MAX_SINGLE_BYTES
    return (
        cols > 0
        and cols % 128 == 0
        and 0 < nbytes <= RDMA_AG_MAX_TOTAL_BYTES
        and (num_chunks <= 1 or rows % num_chunks == 0)
    )


def rdma_all_gather_blocks(block: torch.Tensor, world_size: int) -> torch.Tensor:
    """Execute direct Pallas RDMA all-gather on block, returning full gathered tensor."""
    op = _OPS.get(world_size)
    if op is None:
        raise RuntimeError(
            f"RDMA all-gather op for world_size {world_size} not registered"
        )
    rows = block.shape[0]
    cols = block.numel() // rows
    x2d = block.reshape(rows, cols)
    nbytes = block.numel() * block.element_size()
    num_chunks = (nbytes + RDMA_AG_MAX_SINGLE_BYTES - 1) // RDMA_AG_MAX_SINGLE_BYTES

    if num_chunks <= 1:
        gathered = op(_own_storage(x2d))
    else:
        chunk_rows = rows // num_chunks
        chunks = [
            op(_own_storage(x2d[i * chunk_rows : (i + 1) * chunk_rows].clone()))
            for i in range(num_chunks)
        ]
        gathered = torch.cat(chunks, dim=1)

    return gathered.reshape(world_size * rows, *block.shape[1:])


def rdma_all_gather_into_tensor(
    output: torch.Tensor, block: torch.Tensor, world_size: int
) -> None:
    """Execute direct Pallas RDMA all-gather on block directly into output."""
    op = _OPS.get(world_size)
    if op is None:
        raise RuntimeError(
            f"RDMA all-gather op for world_size {world_size} not registered"
        )
    rows = block.shape[0]
    cols = block.numel() // rows
    x2d = block.reshape(rows, cols)
    nbytes = block.numel() * block.element_size()
    out_3d = output.view(world_size, rows, cols)
    num_chunks = (nbytes + RDMA_AG_MAX_SINGLE_BYTES - 1) // RDMA_AG_MAX_SINGLE_BYTES

    if num_chunks <= 1:
        out_3d.copy_(op(_own_storage(x2d)))
    else:
        chunk_rows = rows // num_chunks
        for i in range(num_chunks):
            chunk_in = _own_storage(x2d[i * chunk_rows : (i + 1) * chunk_rows].clone())
            out_3d[:, i * chunk_rows : (i + 1) * chunk_rows, :].copy_(op(chunk_in))
