# SPDX-License-Identifier: Apache-2.0
"""Ring-mode prefill-context-parallel (PCP) op of the TPU batched RPA backend.

Under ``SGLANG_TPU_PCP_MODE=ring`` the KV cache of a sequence is split across
the ranks of the context-parallel group by page, exactly as under decode
context parallelism: rank ``r`` stores global page ``l * P + r`` and every rank
reads the same page table. A zigzag prefill then runs one fused kernel per
layer on every rank:

* cache phase: each rank attends its own queries against the whole cached
  prefix by rotating the KV blocks of the stripes around the ring (remote DMA
  from rank to rank), so no rank ever holds more than its stripe;
* new-token phase: the same queries continue the online softmax over the
  chunk's own K/V, which the caller all-gathered across the group
  (``cp_forward.all_gather_kv``) and the kernel reads by page through
  ``new_kv_page_indices``.

Because the ring uses remote DMA, the op is a mesh kernel: it is exported as a
``jax.shard_map`` over the whole world with partition specs, every rank passes
its own shard, and all ranks must launch it in the same order with the same
static shapes. Everything rank-dependent (``cp_rank``, ``q_positions``) is a
traced tensor, so all ranks trace one program.

The op reads the KV pool and never writes it: XLA copies every operand that a
kernel containing a remote DMA aliases (see ``muse_megakernel/kernel.py`` and
``glm_flash_megakernel/kernel/decode.py``), which for a KV pool would be a
whole-pool copy per layer. The caller writes the new pages this rank owns with
the local in-place row writer instead (``cp_forward.write_gathered_kv``).

``attention_pallas_rpa_base`` drives the op; gather-mode PCP never calls in
here. The host-side row layout is ``cp_layout.RingLayout``.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here (see
# attention_pallas_rpa_base for why torch_tpu's jax_op needs real annotations).

import torch

from sglang_tpu.pallas_export import tpu_info_for_cpu_export

__all__ = [
    "PCP_RING_AXIS",
    "PCP_RING_OP_NAME",
    "build_pcp_ring_op",
    "pcp_ring_partition_specs",
]

# torch custom-op name. Distinct from the plain batched op and the DCP op so
# all three can be registered in one process and are keyed separately by the
# FX-graph compile cache; bump the suffix when the kernel source changes.
PCP_RING_OP_NAME = "sglang_tpu_pallas::ragged_paged_attention_batched_pcp_ring_v1"

# Mesh axis the ring runs along. The mesh is one-dimensional over the whole
# world, in torch-rank order (``torch_tpu.get_pallas_mesh``).
PCP_RING_AXIS = "model"

# Tensor operands of the op, in order. Every one is passed per rank and
# partitioned along its first dimension, so nothing relies on two ranks holding
# equal values.
_TENSOR_ARGS = (
    "queries",
    "keys",
    "values",
    "kv_cache_fused",
    "cache_kv_lens",
    "cache_page_indices",
    "cache_cu_q_lens",
    "cache_distribution",
    "new_kv_lens",
    "new_page_indices",
    "new_cu_q_lens",
    "new_distribution",
    "new_kv_page_indices",
    "q_positions",
    "kv_cache_lens",
    "cp_rank",
)
# Static (non-traced) arguments after the tensors; they select the generated
# kernel, so they key the export.
_STATIC_ARGS = ("sm_scale", "k_scale", "v_scale")

# torch custom ops are process-global; cache the registration per CP size so
# repeated backend construction reuses it.
_OP_CACHE: dict = {}


def _import_pcp_kernel():
    """The fused ring kernel entry point, imported lazily so this module (and
    the backend) import on hosts without JAX and so CPU tests can stub it."""
    from sglang_tpu.attention.pallas.batched_rpa.wrapper import (
        ragged_paged_attention_pcp_fused,
    )

    return ragged_paged_attention_pcp_fused


def pcp_ring_partition_specs():
    """``(input specs, output specs)`` of the exported mesh function.

    Inputs: one spec per positional argument of the wrapped function (the
    static ones included; they carry no array). Output: the attention output.
    """
    from jax.sharding import PartitionSpec as P

    in_specs = (P(PCP_RING_AXIS),) * len(_TENSOR_ARGS) + (P(),) * len(_STATIC_ARGS)
    out_specs = (P(PCP_RING_AXIS),)
    return in_specs, out_specs


def _invoke_pcp_kernel(
    kernel,
    *,
    cp_size: int,
    queries,
    keys,
    values,
    kv_cache_fused,
    cache_kv_lens,
    cache_page_indices,
    cache_cu_q_lens,
    cache_distribution,
    new_kv_lens,
    new_page_indices,
    new_cu_q_lens,
    new_distribution,
    new_kv_page_indices,
    q_positions,
    kv_cache_lens,
    cp_rank,
    sm_scale,
    k_scale,
    v_scale,
):
    """Call the fused kernel for one rank (inside ``shard_map``).

    Returns this rank's attention output. The pool is read-only
    (``update_kv_cache=False``): the kernel neither aliases nor returns it.
    One program covers batches with and without a cached prefix (a request
    without one gets no ring steps).
    """
    from sglang_tpu.attention_pallas_rpa_batched import kv_layout

    return kernel(
        queries,
        keys,
        values,
        kv_cache_fused,
        cache_kv_lens=cache_kv_lens,
        cache_page_indices=cache_page_indices,
        cache_cu_q_lens=cache_cu_q_lens,
        cache_distribution=cache_distribution,
        new_kv_lens=new_kv_lens,
        new_page_indices=new_page_indices,
        new_cu_q_lens=new_cu_q_lens,
        new_distribution=new_distribution,
        new_kv_page_indices=new_kv_page_indices,
        q_positions=q_positions,
        kv_cache_lens=kv_cache_lens,
        cp_rank=cp_rank,
        cp_group_size=cp_size,
        ring_axis_name=PCP_RING_AXIS,
        ring_mesh_axis_names=(PCP_RING_AXIS,),
        sm_scale=sm_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        kv_layout=kv_layout(),
        update_kv_cache=False,
    )


def build_pcp_ring_op(spec, cp_size: int, *, mesh=None, kernel=None):
    """Register the fused ring kernel as a torch custom op for ``cp_size`` ranks.

    Must be called from the backend constructor, never inside a traced region
    (the registration inspects signatures, which dynamo cannot trace), and
    after ``torch.distributed`` is initialized (the mesh is built from the
    process group).

    The op takes this rank's operands and returns this rank's attention
    output; it mutates nothing (the KV pool is a read-only operand).

    ``mesh`` and ``kernel`` exist for tests: a prebuilt JAX mesh and a stand-in
    for ``ragged_paged_attention_pcp_fused``.
    """
    from sglang_tpu.attention_pallas_rpa_batched import RPA_BATCHED_SPEC

    if spec is not RPA_BATCHED_SPEC:
        raise NotImplementedError(
            f"{spec.op_name} is not the batched RPA kernel; ring-mode prefill "
            "context parallelism on TPU requires --attention-backend "
            "pallas_tpu_rpa_batched."
        )
    cached = _OP_CACHE.get(cp_size)
    if cached is not None:
        return cached

    import jax
    from jax.sharding import NamedSharding
    from torch_tpu._internal.pallas import pallas as pallas_impl

    from sglang_tpu.pallas_jax_op import _CpuExportJaxCallable

    if mesh is None:
        from torch_tpu._internal.pallas import get_pallas_mesh

        mesh = get_pallas_mesh((PCP_RING_AXIS,), (cp_size,))
    if kernel is None:
        kernel = _import_pcp_kernel()
    in_specs, out_specs = pcp_ring_partition_specs()

    def _wrapped(
        queries: jax.Array,
        keys: jax.Array,
        values: jax.Array,
        kv_cache_fused: jax.Array,
        cache_kv_lens: jax.Array,
        cache_page_indices: jax.Array,
        cache_cu_q_lens: jax.Array,
        cache_distribution: jax.Array,
        new_kv_lens: jax.Array,
        new_page_indices: jax.Array,
        new_cu_q_lens: jax.Array,
        new_distribution: jax.Array,
        new_kv_page_indices: jax.Array,
        q_positions: jax.Array,
        kv_cache_lens: jax.Array,
        cp_rank: jax.Array,
        sm_scale: float = 1.0,
        k_scale: float | None = None,
        v_scale: float | None = None,
    ) -> jax.Array:
        def _per_rank(*operands):
            return _invoke_pcp_kernel(
                kernel,
                cp_size=cp_size,
                sm_scale=sm_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                **dict(zip(_TENSOR_ARGS, operands)),
            )

        return jax.shard_map(
            _per_rank,
            mesh=mesh,
            in_specs=in_specs[: len(_TENSOR_ARGS)],
            out_specs=out_specs[0],
            check_vma=False,
        )(
            queries,
            keys,
            values,
            kv_cache_fused,
            cache_kv_lens,
            cache_page_indices,
            cache_cu_q_lens,
            cache_distribution,
            new_kv_lens,
            new_page_indices,
            new_cu_q_lens,
            new_distribution,
            new_kv_page_indices,
            q_positions,
            kv_cache_lens,
            cp_rank,
        )

    static_argnums = tuple(
        range(len(_TENSOR_ARGS), len(_TENSOR_ARGS) + len(_STATIC_ARGS))
    )
    out_sharding = NamedSharding(mesh, out_specs[0])
    # No donation and no aliasing: see the module docstring.
    jit_fn = jax.jit(
        _wrapped,
        static_argnums=static_argnums,
        keep_unused=True,
        out_shardings=out_sharding,
    )
    # The qualified name is unique within the process and stable across ranks
    # and restarts; the callable additionally fingerprints the exported MLIR.
    trace_key = pallas_impl._get_kernel_invocation_key(
        f"jax_{PCP_RING_OP_NAME}",
        [],
        {"cp_size": cp_size, "static_argnums": static_argnums},
    )
    jax_kernel = _CpuExportJaxCallable(
        name=f"jax_{PCP_RING_OP_NAME}_cp{cp_size}",
        jit_fn=jit_fn,
        trace_key=trace_key,
        mesh=mesh,
        input_partition_specs=in_specs,
        static_argnums=static_argnums,
    )
    jax_kernel.sglang_output_shardings = (out_sharding,)

    custom_op = _register_torch_op(jax_kernel)
    _OP_CACHE[cp_size] = custom_op
    return custom_op


def _register_torch_op(jax_kernel):
    """The torch custom op around ``jax_kernel`` (split out so CPU tests can
    exercise the argument plumbing with a stand-in callable)."""

    @torch.library.custom_op(PCP_RING_OP_NAME, mutates_args=(), device_types=["tpu"])
    def custom_op(
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        kv_cache_fused: torch.Tensor,
        cache_kv_lens: torch.Tensor,
        cache_page_indices: torch.Tensor,
        cache_cu_q_lens: torch.Tensor,
        cache_distribution: torch.Tensor,
        new_kv_lens: torch.Tensor,
        new_page_indices: torch.Tensor,
        new_cu_q_lens: torch.Tensor,
        new_distribution: torch.Tensor,
        new_kv_page_indices: torch.Tensor,
        q_positions: torch.Tensor,
        kv_cache_lens: torch.Tensor,
        cp_rank: torch.Tensor,
        sm_scale: float = 1.0,
        k_scale: float | None = None,
        v_scale: float | None = None,
    ) -> torch.Tensor:
        return invoke_jax_kernel(
            jax_kernel,
            queries,
            keys,
            values,
            kv_cache_fused,
            cache_kv_lens,
            cache_page_indices,
            cache_cu_q_lens,
            cache_distribution,
            new_kv_lens,
            new_page_indices,
            new_cu_q_lens,
            new_distribution,
            new_kv_page_indices,
            q_positions,
            kv_cache_lens,
            cp_rank,
            sm_scale,
            k_scale,
            v_scale,
        )

    custom_op.register_fake(_fake_impl)
    return custom_op


def invoke_jax_kernel(jax_kernel, *args) -> torch.Tensor:
    """Run the exported kernel and return this rank's attention output."""
    with tpu_info_for_cpu_export():
        return jax_kernel(*args)


def _fake_impl(queries: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    # Shapes only, without tracing JAX: this is what lets the dynamic token
    # dimension flow through torch.compile.
    return torch.empty_like(queries)
