# SPDX-License-Identifier: Apache-2.0
"""TorchTPU JAX custom-op adapter for CPU-only StableHLO export."""

import hashlib
import inspect
from collections.abc import Callable, Sequence

import jax
import torch
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch_tpu._internal.pallas import pallas as pallas_impl

from sglang.srt.hardware_backend.tpu.pallas_export import tpu_info_for_cpu_export


def _partition_axis_size(mesh: Mesh, axes: str | Sequence[str] | None) -> int:
    if axes is None:
        return 1
    if isinstance(axes, str):
        axes = (axes,)

    size = 1
    for axis in axes:
        size *= mesh.shape[axis]
    return size


def _convert_partitioned_shape(
    shape: Sequence[int],
    mesh: Mesh | None,
    partition_spec: PartitionSpec | None,
    operation: Callable[[int, int], int],
) -> tuple[int, ...]:
    if mesh is None or partition_spec is None:
        return tuple(shape)

    converted = tuple(
        operation(size, _partition_axis_size(mesh, axes))
        for size, axes in zip(shape, partition_spec)
    )
    return converted + tuple(shape[len(partition_spec) :])


def _jax_placeholder_with_sharding(
    tensor: torch.Tensor,
    mesh: Mesh | None,
    partition_spec: PartitionSpec | None,
) -> jax.ShapeDtypeStruct:
    if not isinstance(tensor, torch.Tensor):
        return tensor

    jax_dtype = pallas_impl.TORCH_TO_JAX_DTYPE_MAP.get(tensor.dtype)
    if jax_dtype is None:
        raise NotImplementedError(
            f"Unsupported dtype for Pallas kernels: {tensor.dtype}"
        )

    jax_shape = tuple(tensor.shape)
    if tensor.dtype == torch.float4_e2m1fn_x2 and jax_shape:
        if tensor.stride(-1) != 1:
            raise ValueError(
                "expected packed dimension of float4_e2m1fn_x2 tensor to be "
                f"contiguous (stride 1), got stride {tensor.stride(-1)} "
                f"(all strides: {tensor.stride()})"
            )
        jax_shape = jax_shape[:-1] + (jax_shape[-1] * 2,)

    if mesh is None:
        return jax.ShapeDtypeStruct(jax_shape, jax_dtype)
    return jax.ShapeDtypeStruct(
        _convert_partitioned_shape(
            jax_shape,
            mesh,
            partition_spec,
            lambda size, axis_size: size * axis_size,
        ),
        jax_dtype,
        sharding=NamedSharding(mesh, partition_spec),
    )


def _jax_placeholders(
    tensors: Sequence[torch.Tensor] | torch.Tensor,
    mesh: Mesh | None,
    partition_specs: Sequence[PartitionSpec] | PartitionSpec | None,
):
    if isinstance(tensors, torch.Tensor):
        return _jax_placeholder_with_sharding(tensors, mesh, partition_specs)
    if mesh is None:
        return [
            _jax_placeholder_with_sharding(tensor, None, None) for tensor in tensors
        ]
    return [
        _jax_placeholder_with_sharding(tensor, mesh, partition_spec)
        for tensor, partition_spec in zip(tensors, partition_specs)
    ]


def _torch_placeholder_with_sharding(
    aval: jax.core.ShapedArray | None,
    sharding: NamedSharding | None,
    mesh: Mesh | None,
) -> torch.Tensor | None:
    if not isinstance(aval, jax.core.ShapedArray):
        return aval
    torch_dtype = pallas_impl.JAX_TO_TORCH_DTYPE_MAP.get(aval.dtype)
    if torch_dtype is None:
        raise NotImplementedError(f"Unsupported Pallas dtype: {aval.dtype}")
    spec = getattr(sharding, "spec", None)
    if spec is None:
        spec = getattr(getattr(aval, "sharding", None), "spec", None)
    torch_shape = _convert_partitioned_shape(
        aval.shape,
        mesh,
        spec,
        lambda size, axis_size: size // axis_size,
    )
    if torch_dtype == torch.float4_e2m1fn_x2 and torch_shape:
        if torch_shape[-1] % 2 != 0:
            raise ValueError(
                "expected last dimension of JAX local shape to be even to convert "
                f"to packed PyTorch dtype {torch_dtype}, got {torch_shape[-1]} "
                f"(local shape: {torch_shape}, global: {aval.shape})"
            )
        torch_shape = torch_shape[:-1] + (torch_shape[-1] // 2,)
    return torch.empty(torch_shape, dtype=torch_dtype, device="tpu")


def _torch_placeholder(aval, sharding, mesh):
    return _torch_placeholder_with_sharding(aval, sharding, mesh)


class _CpuExportJaxCallable(pallas_impl.JaxCallable):
    def __call__(self, *args, **kwargs):
        self._validate_args(*args)
        kernel_key = pallas_impl._get_kernel_invocation_key(
            self.trace_key, args, kwargs, self.static_argnums
        )
        output_shapes, out_tree = self.output_shapes.get(kernel_key, (None, None))
        mlir_fingerprint = self.kernel_key_to_mlir_fingerprint.get(kernel_key)
        native_kernel_key = (
            f"{kernel_key};mlir={mlir_fingerprint}"
            if mlir_fingerprint is not None
            else None
        )
        kernel_exists = (
            native_kernel_key is not None
            and pallas_impl.tpu_torch_pallas.lookup_custom_kernel(
                self.name, native_kernel_key
            )
        )
        if not output_shapes or not kernel_exists:
            jax_args = _jax_placeholders(
                args,
                mesh=self.mesh,
                partition_specs=self.input_partition_specs,
            )
            with (
                tpu_info_for_cpu_export(),
                jax._src.config.export_ignore_forward_compatibility(True),
            ):
                lowered = self.exported(*jax_args, **kwargs)
            # Match TorchTPU's content identity: ignore outer caller locations,
            # but retain the embedded Mosaic kernel body and its debug info.
            with jax.interpreters.mlir.make_ir_context():
                module = jax.interpreters.mlir.ir.Module.parse(lowered.mlir_module())
                normalized_mlir = module.operation.get_asm(enable_debug_info=False)
            mlir_fingerprint = hashlib.blake2b(
                normalized_mlir.encode("utf-8"), digest_size=16
            ).hexdigest()
            self.kernel_key_to_mlir_fingerprint[kernel_key] = mlir_fingerprint
            native_kernel_key = f"{kernel_key};mlir={mlir_fingerprint}"
            pallas_impl.tpu_torch_pallas.register_custom_kernel(
                self.name,
                native_kernel_key,
                serialized_mlir_module=lowered.mlir_module_serialized,
            )
            shardings = self.sglang_output_shardings or (None,) * len(lowered.out_avals)
            output_shapes = [
                _torch_placeholder(aval, sharding, self.mesh)
                for aval, sharding in zip(lowered.out_avals, shardings)
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
            native_kernel_key,
            inputs=tensor_args,
            output_shapes=output_shapes,
            donate_argnums=self.donate_argnums,
            device_assignment=(
                [int(device.id) for device in self.mesh.devices.flat]
                if self.mesh is not None
                else []
            ),
        )
        for input_index, output_index in self.input_output_aliases.items():
            tensor_args[input_index].copy_(results[output_index])
        return out_tree.unflatten(results)


def cpu_export_jax_op(
    name: str,
    function: Callable[..., object],
    *,
    mesh: Mesh | None = None,
    input_partition_specs: Sequence[PartitionSpec] | None = None,
    output_partition_specs: Sequence[PartitionSpec] | None = None,
    donate_argnums: Sequence[int] | None = None,
):
    signature = inspect.signature(function, follow_wrapped=False)
    pallas_impl._verify_signature(signature)
    static_argnums = pallas_impl._infer_static_argnums(signature)
    donate_argnums_tuple = tuple(donate_argnums or ())
    input_partition_specs = tuple(input_partition_specs or ())
    output_partition_specs = tuple(output_partition_specs or ())
    output_shardings = tuple(
        NamedSharding(mesh, spec) for spec in output_partition_specs
    )
    jit_kwargs = {}
    if output_shardings:
        jit_kwargs["out_shardings"] = (
            output_shardings[0] if len(output_shardings) == 1 else output_shardings
        )
    jit_function = jax.jit(
        function,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums_tuple,
        keep_unused=True,
        **jit_kwargs,
    )
    # The qualified custom-op name is unique within the process and stable
    # across distributed ranks and server restarts.
    trace_key = pallas_impl._get_kernel_invocation_key(
        name,
        [],
        {
            "static_argnums": static_argnums,
            "donate_argnums": donate_argnums_tuple,
            "output_partition_specs": tuple(map(str, output_partition_specs)),
        },
    )
    wrapped = _CpuExportJaxCallable(
        name=name,
        jit_fn=jit_function,
        trace_key=trace_key,
        mesh=mesh,
        input_partition_specs=input_partition_specs or None,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums_tuple,
    )
    wrapped.sglang_output_shardings = output_shardings
    custom_op = torch.library.custom_op(name, wrapped, mutates_args=())

    def fake_function(*args, **kwargs):
        jax_args = _jax_placeholders(
            args,
            mesh=mesh,
            partition_specs=tuple(input_partition_specs),
        )
        with (
            tpu_info_for_cpu_export(),
            jax._src.config.export_ignore_forward_compatibility(True),
        ):
            lowered = wrapped.exported(*jax_args, **kwargs)
        shardings = output_shardings or (None,) * len(lowered.out_avals)
        return lowered.out_tree.unflatten(
            _torch_placeholder(aval, sharding, mesh)
            for aval, sharding in zip(lowered.out_avals, shardings)
        )

    custom_op.register_fake(fake_function)
    return custom_op
