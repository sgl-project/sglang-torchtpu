# SPDX-License-Identifier: Apache-2.0
"""TorchTPU JAX custom-op adapter with explicit output sharding.

Adapted from the multi-device PCP adapter in google-pytorch/torchtpu-vllm.
TorchTPU's stock ``jax_op`` can globalize sharded inputs, but exported shard-map
outputs may appear replicated in ``out_avals``. This adapter preserves the
requested output shardings and allocates rank-local torch outputs accordingly.
"""

import inspect
from collections.abc import Callable, Sequence

import jax
import torch
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch_tpu._internal.pallas import pallas as pallas_impl

from sglang.srt.hardware_backend.tpu.pallas_export import tpu_info_for_cpu_export


def _torch_placeholder_with_sharding(
    aval: jax.core.ShapedArray | None,
    sharding: NamedSharding | None,
    mesh: Mesh,
) -> torch.Tensor | None:
    if not isinstance(aval, jax.core.ShapedArray):
        return aval
    torch_dtype = pallas_impl.JAX_TO_TORCH_DTYPE_MAP.get(aval.dtype)
    if torch_dtype is None:
        raise NotImplementedError(f"Unsupported Pallas dtype: {aval.dtype}")
    spec = getattr(sharding, "spec", None)
    if spec is None:
        spec = getattr(getattr(aval, "sharding", None), "spec", None)
    return torch.empty(
        pallas_impl.get_local_shape(aval.shape, mesh, spec),
        dtype=torch_dtype,
        device="tpu",
    )


class _ShardedJaxCallable(pallas_impl.JaxCallable):
    def __call__(self, *args, **kwargs):
        self._validate_args(*args)
        kernel_key = pallas_impl._get_kernel_invocation_key(
            self.trace_key, args, kwargs, self.static_argnums
        )
        output_shapes, out_tree = self.output_shapes.get(kernel_key, (None, None))
        kernel_exists = pallas_impl.tpu_torch_pallas.lookup_custom_kernel(
            self.name, kernel_key
        )
        if not output_shapes or not kernel_exists:
            jax_args = pallas_impl.jax_placeholders(
                args,
                mesh=self.mesh,
                partition_specs=self.input_partition_specs,
            )
            with (
                tpu_info_for_cpu_export(),
                jax._src.config.export_ignore_forward_compatibility(True),
            ):
                lowered = self.exported(*jax_args, **kwargs)
            pallas_impl.tpu_torch_pallas.register_custom_kernel(
                self.name,
                kernel_key,
                serialized_mlir_module=lowered.mlir_module_serialized,
            )
            output_shapes = [
                _torch_placeholder_with_sharding(aval, sharding, self.mesh)
                for aval, sharding in zip(
                    lowered.out_avals, self.sglang_output_shardings
                )
            ]
            out_tree = lowered.out_tree
            self.output_shapes[kernel_key] = (output_shapes, out_tree)

        tensor_args = [
            arg
            for index, arg in enumerate(args)
            if arg is not None and index not in self.static_argnums
        ]
        results = pallas_impl.tpu_torch_pallas.call_custom_kernel(
            self.name,
            kernel_key,
            inputs=tensor_args,
            output_shapes=output_shapes,
            donate_argnums=self.donate_argnums,
        )
        for input_index, output_index in self.input_output_aliases.items():
            tensor_args[input_index].copy_(results[output_index])
        return out_tree.unflatten(results)


def sharded_jax_op(
    name: str,
    function: Callable[..., object],
    *,
    mesh: Mesh,
    input_partition_specs: Sequence[PartitionSpec],
    output_partition_specs: Sequence[PartitionSpec],
    donate_argnums: Sequence[int] | None = None,
):
    signature = inspect.signature(function, follow_wrapped=False)
    pallas_impl._verify_signature(signature)
    static_argnums = pallas_impl._infer_static_argnums(signature)
    donate_argnums_tuple = tuple(donate_argnums or ())
    output_shardings = tuple(
        NamedSharding(mesh, spec) for spec in output_partition_specs
    )
    jit_output_shardings = (
        output_shardings[0] if len(output_shardings) == 1 else output_shardings
    )
    jit_function = jax.jit(
        function,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums_tuple,
        out_shardings=jit_output_shardings,
    )
    trace_key = pallas_impl._get_kernel_invocation_key(
        f"{name}_{id(function)}",
        [],
        {
            "static_argnums": static_argnums,
            "donate_argnums": donate_argnums_tuple,
            "output_partition_specs": tuple(map(str, output_partition_specs)),
        },
    )
    wrapped = _ShardedJaxCallable(
        name=name,
        jit_fn=jit_function,
        trace_key=trace_key,
        mesh=mesh,
        input_partition_specs=tuple(input_partition_specs),
        static_argnums=static_argnums,
        donate_argnums=donate_argnums_tuple,
    )
    wrapped.sglang_output_shardings = output_shardings
    custom_op = torch.library.custom_op(name, wrapped, mutates_args=())

    def fake_function(*args, **kwargs):
        jax_args = pallas_impl.jax_placeholders(
            args,
            mesh=mesh,
            partition_specs=tuple(input_partition_specs),
        )
        with (
            tpu_info_for_cpu_export(),
            jax._src.config.export_ignore_forward_compatibility(True),
        ):
            lowered = wrapped.exported(*jax_args, **kwargs)
        return lowered.out_tree.unflatten(
            _torch_placeholder_with_sharding(aval, sharding, mesh)
            for aval, sharding in zip(lowered.out_avals, output_shardings)
        )

    custom_op.register_fake(fake_function)
    return custom_op
