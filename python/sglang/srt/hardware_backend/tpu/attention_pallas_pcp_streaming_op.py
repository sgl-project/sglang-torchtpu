# SPDX-License-Identifier: Apache-2.0
"""Torch custom-op adapter for the PCP streaming RPA kernel.

Bridges the JAX-Mesh-scoped kernel at
``sglang.srt.layers.attention.pallas.pcp_streaming_rpa`` into the torch
custom-op registry so ``torch.compile`` can dispatch to it.
``_PcpStreamingJaxCallable`` depends only on public
``torch_tpu._internal.pallas`` symbols.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here —
# torch_tpu's ``pallas._verify_signature`` inspects ``_pcp_streaming_rpa_kernel``
# below and rejects PEP-563 string annotations (only ``jax.Array`` /
# POD types are accepted, and it does not resolve forward refs). Same
# rule as the base RPA v3 CP backend (see attention_pallas_rpa_v3_cp.py).

import inspect
import os
from collections.abc import Callable, Sequence
from typing import Optional, Tuple

import jax
import torch
from jax.sharding import PartitionSpec
from torch_tpu._internal.pallas import pallas as pallas_impl

from sglang.srt.distributed.pcp_mesh import get_or_create_pcp_mesh
from sglang.srt.environ import envs
from sglang.srt.layers.attention.pallas.pcp_streaming_rpa.kernel import (
    PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE,
)
from sglang.srt.layers.attention.pallas.pcp_streaming_rpa.wrapper import (
    PCP_AXIS_NAME,
    sharded_pcp_ragged_paged_attention,
)

# Read once at import so the check is Dynamo-safe (runtime os.environ.get would
# graph-break under torch.compile).
_PCP_DUMP_MLIR_DIR: Optional[str] = envs.SGLANG_TPU_PCP_DUMP_MLIR.get() or None


def _maybe_dump_mlir(name: str, args: tuple, lowered) -> None:
    if not _PCP_DUMP_MLIR_DIR:
        return
    os.makedirs(_PCP_DUMP_MLIR_DIR, exist_ok=True)
    shape_tag = "_".join(
        "x".join(str(d) for d in a.shape) for a in args if hasattr(a, "shape")
    )
    dump_path = os.path.join(_PCP_DUMP_MLIR_DIR, f"{name}_{shape_tag[:180]}.mlir")
    with open(dump_path, "wb") as f:
        f.write(lowered.mlir_module_serialized)
    try:
        with open(dump_path + ".txt", "w") as f:
            f.write(str(lowered.mlir_module))
    except Exception:
        pass
    print(f"[PCP DUMP] wrote MLIR: {dump_path}", flush=True)


PCP_STREAMING_RPA_INPUT_PARTITION_SPECS = (
    PartitionSpec(PCP_AXIS_NAME),  # kv_cache: local cache blocks
    PartitionSpec(PCP_AXIS_NAME),  # query: local prefill tokens
    PartitionSpec(PCP_AXIS_NAME),  # key: local prefill tokens
    PartitionSpec(PCP_AXIS_NAME),  # value: local prefill tokens
    PartitionSpec(),  # seq_lens
    PartitionSpec(),  # block_tables
    PartitionSpec(),  # query_start_loc
    PartitionSpec(),  # request_distribution
)
PCP_STREAMING_RPA_OUTPUT_PARTITION_SPECS = (
    PartitionSpec(PCP_AXIS_NAME),  # new_kv_cache
    PartitionSpec(PCP_AXIS_NAME),  # output
)


def _torch_placeholder_with_sharding(aval, sharding, mesh):
    if not isinstance(aval, jax.core.ShapedArray):
        return aval
    torch_dtype = pallas_impl.JAX_TO_TORCH_DTYPE_MAP.get(aval.dtype)
    if torch_dtype is None:
        raise NotImplementedError(f"Unsupported dtype for pallas kernels: {aval.dtype}")
    spec = getattr(sharding, "spec", None)
    if spec is None:
        spec = getattr(getattr(aval, "sharding", None), "spec", None)
    return torch.empty(
        pallas_impl.get_local_shape(aval.shape, mesh, spec),
        dtype=torch_dtype,
        device="tpu",
    )


class _PcpStreamingJaxCallable(pallas_impl.JaxCallable):
    """JaxCallable variant that allocates shard_map outputs using
    ``_out_named_shardings`` — torch_tpu's stock JaxCallable derives output
    placeholder shapes from ``lowered.out_avals``, but ``jax.export``
    records shard_map outputs there as replicated, which over-allocates
    torch-side outputs. This subclass consults ``_out_named_shardings`` so
    each rank allocates only its local output slice.

    Aliasing pattern: NO ``input_output_aliases``
    is passed into ``JaxCallable.__init__`` (the base leaves the alias map
    empty), and the outer custom_op is registered with ``mutates_args=()``
    (see :func:`pcp_streaming_jax_op`). The KV writeback is instead done at
    the caller (:func:`invoke_pcp_streaming_op`) via an explicit
    ``kv_cache.copy_(new_kv_cache)`` on the tuple output — Dynamo tracks
    the ``copy_`` as an in-place mutation of the pool tensor, XLA lifts
    the copy_ input and aliases the op output onto the donated pool, and
    the compiled forward gets an in-place pool update. Marking
    ``mutates_args=("kv_cache",)`` here (base RPA v3's pattern) would tell
    Dynamo the op mutates its input, forcing functionalization to clone
    the KV buffer before the call and copy the mutated clone back at graph
    exit — the very anti-pattern torch_tpu's own ``jax_op`` decorator
    warns about (pallas.py:758-764) and the source of ~50% wrong
    first-token output at ISL=4096 on the previous sglang PCP path.
    """

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
                args, mesh=self.mesh, partition_specs=self.input_partition_specs
            )
            with jax._src.config.export_ignore_forward_compatibility(True):
                lowered = self.exported(*jax_args, **kwargs)
            _maybe_dump_mlir(self.name, args, lowered)
            pallas_impl.tpu_torch_pallas.register_custom_kernel(
                self.name,
                kernel_key,
                serialized_mlir_module=lowered.mlir_module_serialized,
            )
            out_shardings = getattr(lowered, "_out_named_shardings", None)
            if out_shardings is None or len(out_shardings) != len(lowered.out_avals):
                out_shardings = [None] * len(lowered.out_avals)
            output_shapes = [
                _torch_placeholder_with_sharding(aval, sharding, self.mesh)
                for aval, sharding in zip(lowered.out_avals, out_shardings)
            ]
            out_tree = lowered.out_tree
            self.output_shapes[kernel_key] = (output_shapes, out_tree)

        tensor_args = [
            arg
            for i, arg in enumerate(args)
            if arg is not None and i not in self.static_argnums
        ]
        results = pallas_impl.tpu_torch_pallas.call_custom_kernel(
            self.name,
            kernel_key,
            inputs=tensor_args,
            output_shapes=output_shapes,
            donate_argnums=self.donate_argnums,
        )

        return out_tree.unflatten(results)


def _named_shardings(mesh, partition_specs: Sequence[PartitionSpec]):
    return tuple(jax.sharding.NamedSharding(mesh, spec) for spec in partition_specs)


def build_pcp_streaming_callable(
    name: str,
    fn: Callable[..., object],
    *,
    donate_argnums: Optional[Sequence[int]],
    mesh,
    input_partition_specs: Sequence[PartitionSpec],
    output_partition_specs: Optional[Sequence[PartitionSpec]] = None,
) -> "_PcpStreamingJaxCallable":
    signature = inspect.signature(fn, follow_wrapped=False)
    pallas_impl._verify_signature(signature)
    static_argnums = pallas_impl._infer_static_argnums(signature)
    donate_argnums_tuple = tuple(donate_argnums or ())
    output_partition_specs_tuple = tuple(
        output_partition_specs or PCP_STREAMING_RPA_OUTPUT_PARTITION_SPECS
    )
    output_shardings = _named_shardings(mesh, output_partition_specs_tuple)
    # ``keep_unused=True`` mirrors torch_tpu's own ``custom_jax_kernel``
    # (pallas.py:661): without it, JAX may drop arguments the trace
    # never reads, which shifts positional indices and can invalidate
    # ``donate_argnums=(0,)``. The PCP kernel does read arg 0, so this
    # is defensive rather than load-bearing; matched the base RPA v3
    # path and kept warm-probe latency stable in testing.
    jit_fn = jax.jit(
        fn,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums_tuple,
        out_shardings=output_shardings,
        keep_unused=True,
    )
    trace_key = pallas_impl._get_kernel_invocation_key(
        f"{name}_{id(fn)}",
        [],
        {
            "static_argnums": static_argnums,
            "donate_argnums": donate_argnums_tuple,
            "output_partition_specs": tuple(map(str, output_partition_specs_tuple)),
        },
    )
    # DO NOT pass ``input_output_aliases`` — this path leaves it empty
    # and does the ``kv_cache.copy_(new_kv_cache)`` writeback OUTSIDE the
    # op boundary (see
    # :func:`invoke_pcp_streaming_op`). Passing ``input_output_aliases``
    # here would activate JaxCallable's internal ``tensor_args[in_idx]
    # .copy_(results[out_idx])`` loop and — combined with
    # ``mutates_args=("kv_cache",)`` on the outer op — force Dynamo to
    # clone-before-call + copy-back-at-graph-exit, defeating the
    # aliasing (per torch_tpu pallas.py:758-764).
    return _PcpStreamingJaxCallable(
        name=name,
        jit_fn=jit_fn,
        trace_key=trace_key,
        mesh=mesh,
        input_partition_specs=tuple(input_partition_specs),
        static_argnums=static_argnums,
        donate_argnums=donate_argnums_tuple,
    )


def pcp_streaming_jax_op(
    name: str,
    fn: Callable[..., object],
    *,
    donate_argnums: Optional[Sequence[int]],
    mesh,
    input_partition_specs: Sequence[PartitionSpec],
    output_partition_specs: Optional[Sequence[PartitionSpec]] = None,
):
    """Register a PCP streaming custom op with explicit output sharding.

    torch_tpu's stock ``pallas.jax_op`` derives output placeholder shapes
    from ``lowered.out_avals``, but ``jax.export`` records shard_map
    outputs there as replicated. Explicit ``out_shardings`` are preserved
    in ``_out_named_shardings``; ``_PcpStreamingJaxCallable`` uses that
    to allocate local torch outputs.
    """
    wrapped_fn = build_pcp_streaming_callable(
        name,
        fn,
        donate_argnums=donate_argnums,
        mesh=mesh,
        input_partition_specs=input_partition_specs,
        output_partition_specs=output_partition_specs,
    )

    # PCP op pattern: ``mutates_args=()`` + return the tuple
    # ``(new_kv_cache, output)`` + caller does explicit
    # ``kv_cache.copy_(new_kv_cache)``. Base RPA v3 uses
    # ``mutates_args=("kv_cache_fused",)`` because it hides the write
    # entirely inside the op; that pattern was flaky under
    # ``torch.compile`` on the PCP shard_map + jax.export path (~50%
    # wrong first-token at ISL=4096).
    #
    # Intermediate ``_pcp_streaming_custom_op`` wrapper is retained (vs
    # vllm's direct ``op_target = wrapped_fn``) because torch.library's
    # schema inference needs a Python-annotated signature and dropping
    # it triggered extra Dynamo recompiles on the sglang stack (warm
    # probe latency spiked from ~1s to ~3s in testing).
    def _pcp_streaming_custom_op(
        kv_cache: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        seq_lens: torch.Tensor,
        block_tables: torch.Tensor,
        query_start_loc: torch.Tensor,
        request_distribution: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return wrapped_fn(
            kv_cache,
            query,
            key,
            value,
            seq_lens,
            block_tables,
            query_start_loc,
            request_distribution,
        )

    # DO NOT CHANGE mutates_args=(). See the file docstring for
    # _PcpStreamingJaxCallable + the invoke_pcp_streaming_op docstring:
    # KV writeback happens at the caller via kv_cache.copy_(new_kv_cache);
    # switching to mutates_args=("kv_cache",) forces functionalization
    # (clone + copy-back-at-graph-exit), which caused ~50% wrong first-
    # token output at ISL=4096 on the previous PCP path (M-series
    # investigation).
    result = torch.library.custom_op(
        name,
        _pcp_streaming_custom_op,
        mutates_args=(),
        device_types=["tpu"],
    )

    # Fake registration: a simple tuple of ``empty_like`` matching the
    # op's (new_kv_cache, output) return.
    def _fake_rpa_op(
        kv_cache: torch.Tensor,
        query: torch.Tensor,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.empty_like(kv_cache), torch.empty_like(query)

    result.register_fake(_fake_rpa_op)
    return result


def get_pcp_streaming_mesh():
    return get_or_create_pcp_mesh(axis_name=PCP_AXIS_NAME)


def invoke_pcp_streaming_op(
    op,
    *,
    kv_cache: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    query_start_loc: torch.Tensor,
    request_distribution: torch.Tensor,
) -> torch.Tensor:
    """Invoke the PCP streaming custom op with the caller-side KV writeback.

    Consolidates the op call, tuple unpack, and ``kv_cache.copy_()`` into
    this single helper. Empirically on the sglang stack this frame layout
    keeps Dynamo happy — moving the ``copy_`` up into the caller frame
    caused warm-probe recompiles and ~3x latency regression
    in testing. Under torch.compile Dynamo tracks the copy_ as an in-place
    mutation of the pool tensor and XLA lifts the copy_ input, aliasing
    the op output onto the donated pool. Base RPA v3's
    ``mutates_args=(kv_cache_fused,)`` pattern (which hides the write
    inside the op) is flaky under torch.compile on the PCP shard_map +
    jax.export path — hence the explicit copy_ here.
    """
    new_kv_cache, output = op(
        kv_cache,
        query,
        key,
        value,
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
    )
    # DO NOT CHANGE the copy_ + tuple return pattern. This is the load-
    # bearing half of the mutates_args=() aliasing contract; see docstring
    # above + pcp_streaming_jax_op's mutates_args comment. Moving the
    # copy_ up into the caller frame caused warm-probe recompiles
    # (~3x latency regression); removing it silently drops KV writes.
    kv_cache.copy_(new_kv_cache)
    return output


def make_pcp_streaming_rpa_kernel(
    *,
    q_scale: Optional[float],
    k_scale: Optional[float],
    v_scale: Optional[float],
    mesh,
    sliding_window: Optional[int],
    sm_scale: Optional[float] = None,
    soft_cap: Optional[float] = None,
    skip_kv_update: bool,
    cp_kv_cache_interleave_size: int,
    max_model_len: Optional[int] = None,
    q_block_size: int = PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE,
    q_compute_size: Optional[int] = None,
):
    """Build a PCP streaming RPA entry with only tensor args in its signature."""
    if soft_cap is not None:
        raise NotImplementedError("PCP streaming RPA does not support logits soft cap.")
    if skip_kv_update:
        raise NotImplementedError("PCP streaming RPA does not support skip_kv_update.")

    def _pcp_streaming_rpa_kernel(
        kv_cache: jax.Array,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
        request_distribution: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        output, new_kv_cache = sharded_pcp_ragged_paged_attention(
            mesh=mesh,
            q=query,
            k=key,
            v=value,
            kv_cache=kv_cache,
            kv_lens=seq_lens,
            page_indices=block_tables,
            cu_q_lens=query_start_loc,
            distribution=request_distribution,
            attention_sink=None,
            sm_scale=sm_scale,
            attention_chunk_size=sliding_window,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            max_context_tokens=max_model_len,
            update_kv_cache=True,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            q_block_size=q_block_size,
            q_compute_size=q_compute_size,
        )
        return new_kv_cache, output

    return _pcp_streaming_rpa_kernel
