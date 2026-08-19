# SPDX-License-Identifier: Apache-2.0
"""TorchTPU JAX custom-op adapter for CPU-only StableHLO export."""

from collections.abc import Callable, Sequence

from jax.sharding import Mesh, PartitionSpec
from sglang.srt.hardware_backend.tpu.pallas_export import tpu_info_for_cpu_export
from torch_tpu._internal.pallas import pallas as pallas_impl


def cpu_export_jax_op(
    name: str,
    function: Callable[..., object],
    *,
    mesh: Mesh | None = None,
    input_partition_specs: Sequence[PartitionSpec] | None = None,
    output_partition_specs: Sequence[PartitionSpec] | None = None,
    donate_argnums: Sequence[int] | None = None,
):
    donate_argnums_tuple = tuple(donate_argnums or ())
    input_partition_specs = tuple(input_partition_specs or ())
    output_partition_specs = tuple(output_partition_specs or ())
    return pallas_impl.jax_op(
        name,
        function,
        donate_argnums=donate_argnums_tuple,
        mesh=mesh,
        input_partition_specs=input_partition_specs or None,
        output_partition_specs=output_partition_specs or None,
        lowering_context=tpu_info_for_cpu_export,
    )
