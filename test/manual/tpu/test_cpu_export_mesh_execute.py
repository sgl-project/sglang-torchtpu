# SPDX-License-Identifier: Apache-2.0
"""Multi-host gate: Pallas MESH DMA and torch collectives share device order.

Run under torchrun on a configured TPU slice. Both synchronous and native
asynchronous direct compilation must return exact integer rank payloads.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.distributed as dist
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from sglang.srt.hardware_backend.tpu.compilation.direct_compile import direct_compile
from sglang.srt.hardware_backend.tpu.pallas_export import get_torchtpu_export_devices
from sglang.srt.hardware_backend.tpu.pallas_jax_op import cpu_export_jax_op


def main():
    dist.init_process_group("tpu_dist")
    rank, size = dist.get_rank(), dist.get_world_size()
    mesh = Mesh(np.asarray(get_torchtpu_export_devices(size)), ("rank",))
    placement = tuple(int(device.id) for device in mesh.devices.flat)
    if placement == tuple(sorted(placement)):
        raise RuntimeError(
            "This gate needs nonidentity device order to detect the regression"
        )

    def body(x_ref, y_ref, scratch, send_sem, recv_sem):
        destination = (jax.lax.axis_index("rank") + 1) % size
        scratch[0] = x_ref[...]
        transfer = pltpu.make_async_remote_copy(
            scratch.at[0],
            scratch.at[1],
            send_sem,
            recv_sem,
            device_id={"rank": destination},
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        transfer.start()
        transfer.wait()
        y_ref[...] = scratch[1]

    kernel = pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.int32),
        scratch_shapes=(
            pltpu.VMEM((2, 8, 128), jnp.int32),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ),
    )

    def ring(x: jax.Array) -> jax.Array:
        return jax.shard_map(
            kernel, mesh=mesh, in_specs=P("rank"), out_specs=P("rank"), check_vma=False
        )(x)

    op = cpu_export_jax_op(
        "mesh_gate::ring",
        ring,
        mesh=mesh,
        input_partition_specs=(P("rank"),),
        output_partition_specs=(P("rank"),),
    )

    def region(x):
        received = op(x)
        gathered = torch.empty((size,), dtype=torch.int32, device=x.device)
        dist.all_gather_into_tensor(gathered, received[0, :1].contiguous())
        return received, gathered

    value = torch.full((8, 128), rank, dtype=torch.int32, device="tpu")
    expected = torch.full((8, 128), (rank - 1) % size, dtype=torch.int32)
    expected_gather = (torch.arange(size, dtype=torch.int32) - 1) % size
    with torch.no_grad():
        for asynchronous in (False, True):
            compiled = direct_compile(
                region, specialization_key=lambda x: tuple(x.shape)
            )
            received, gathered = (
                compiled.warmup_async(value).wait() if asynchronous else compiled(value)
            )
            assert torch.equal(received.cpu(), expected)
            assert torch.equal(gathered.cpu(), expected_gather)
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "async": asynchronous,
                        "device_assignment": placement,
                        "bitwise_pass": True,
                    }
                ),
                flush=True,
            )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
