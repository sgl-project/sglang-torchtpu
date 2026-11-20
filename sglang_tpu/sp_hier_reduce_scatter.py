# SPDX-License-Identifier: Apache-2.0
"""Hierarchical Pallas reduce-scatter for sequence-parallel stages on tpu7x.

On an input-scattered (sequence-parallel) batch every decoder stage ends with
a reduce-scatter of its [T, H] bf16 output over the TP group
(layer_boundary.prepare.sp_reduce_scatter). XLA runs it as one flat collective
over the 8 TensorCores. On a tpu7x 2x2x1 host those are 4 chips of 2 cores, and
kernels/hier_all_reduce.py's row-streaming kernel sums each chip's two cores
first and only then exchanges across chips, streaming 128-row blocks through
VMEM so both ICI links carry traffic on every step.

Opt-in via SGLANG_TPU_SP_HIER_REDUCE_SCATTER. It applies when TP spans one
8-device host (TP = world = 8, one node) of tpu7x, to bf16 [T, H] inputs with
T a multiple of 1024 and a VMEM-fitting H; everything else keeps XLA's
reduce-scatter. At registration every rank checks the kernel against XLA's
reduce-scatter on the same input and falls back if they disagree (e.g. a device
order whose chips are not paired as [2k, 2k + 1]).
"""

import logging

import torch

from sglang_tpu.envs import envs as tpu_envs

logger = logging.getLogger(__name__)

NUM_DEVICES = 8
_ROW_BLOCK = 128  # kernels/hier_all_reduce.py
_SLOTS = 3
# kernels/hier_all_reduce.py's VMEM limit, less headroom for the compiler.
_VMEM_BUDGET_BYTES = (56 << 20) - (2 << 20)

# Built op; None while off. Plain bool for Dynamo-traced eligibility checks.
_OP = None
_READY = False


def scratch_bytes(hidden: int, itemsize: int) -> int:
    """VMEM scratch of the row-streaming kernel for one call (its scratch
    shapes: two [3, 4, 128, H], one [3, 2, 2, 64, H], one [3, 2, 64, H])."""
    half = _ROW_BLOCK // 2
    elems = (
        2 * _SLOTS * 4 * _ROW_BLOCK + _SLOTS * 2 * 2 * half + _SLOTS * 2 * half
    ) * hidden
    return elems * itemsize


def shape_supported(rows: int, hidden: int, dtype: torch.dtype) -> bool:
    chunk, rem = divmod(rows, NUM_DEVICES)
    return (
        dtype == torch.bfloat16
        and rem == 0
        and chunk >= _ROW_BLOCK
        and chunk % _ROW_BLOCK == 0
        and hidden % 128 == 0
        and scratch_bytes(hidden, 2) <= _VMEM_BUDGET_BYTES
    )


def eligible(t: torch.Tensor) -> bool:
    return _READY and t.dim() == 2 and shape_supported(t.shape[0], t.shape[1], t.dtype)


def hier_reduce_scatter(t: torch.Tensor) -> torch.Tensor:
    """[T, H] TP partials -> [T / 8, H]: this rank's row block of the sum."""
    return _OP(t)


def _build_op():
    import jax
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal.pallas import get_pallas_mesh

    from sglang_tpu.kernels.hier_all_reduce import reduce_scatter_stream_local
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    mesh = get_pallas_mesh(("model",), (NUM_DEVICES,))
    spec = P("model", None)

    # torch_tpu reads the operand types from these annotations, so this module
    # must not use `from __future__ import annotations` (it would make them strings).
    def _wrapped(x: jax.Array) -> jax.Array:
        return jax.shard_map(
            lambda local_x: reduce_scatter_stream_local(local_x, axis_name="model"),
            mesh=mesh,
            in_specs=spec,
            out_specs=spec,
            check_vma=False,
        )(x)

    return cpu_export_jax_op(
        f"sglang_tpu_pallas::sp_hier_reduce_scatter_w{NUM_DEVICES}",
        _wrapped,
        mesh=mesh,
        input_partition_specs=(spec,),
        output_partition_specs=(spec,),
    )


def _is_tpu7x() -> bool:
    try:
        from jax.experimental.pallas import tpu as pltpu

        return pltpu.get_tpu_info().chip_version == pltpu.ChipVersion.TPU_7X
    except Exception:  # pragma: no cover - older JAX or no TPU
        import os

        return os.environ.get("TPU_ACCELERATOR_TYPE", "").startswith("tpu7x")


def _self_check(op, tp_group) -> str | None:
    """Every rank runs the kernel and XLA's reduce-scatter on the same
    rank-dependent input; returns why they disagree, or None."""
    rank = tp_group.rank_in_group
    rows, hidden = NUM_DEVICES * _ROW_BLOCK, 256
    gen = torch.Generator().manual_seed(1234 + rank)
    x = (torch.randn(rows, hidden, generator=gen) * (rank + 1)).to(torch.bfloat16)
    x = x.to(tp_group.device)
    want = torch.empty(rows // NUM_DEVICES, hidden, dtype=x.dtype, device=x.device)
    tp_group.reduce_scatter_tensor(want, x.clone())
    got = op(x)
    err = (got.float() - want.float()).abs().max().item()
    scale = want.float().abs().max().item()
    # The two sum the 8 partials in different orders, so bf16 rounding differs.
    if not err <= 0.05 * max(scale, 1.0):
        return f"max |kernel - XLA| = {err:.4g} at max |XLA| = {scale:.4g}"
    return None


def _ranks_agree(flag: bool) -> bool:
    import torch.distributed as dist

    from sglang.srt.distributed.parallel_state import get_world_group

    world_group = get_world_group()
    if world_group.world_size <= 1:
        return flag
    seen = [None] * world_group.world_size
    dist.all_gather_object(seen, bool(flag), group=world_group.cpu_group)
    return all(seen)


def register_from_parallel_state() -> None:
    """Called from TpuPlatform.prepare_model_parallel_groups on every rank,
    after the groups exist and before torch.compile traces the model."""
    global _OP, _READY
    enabled = tpu_envs.SGLANG_TPU_SP_HIER_REDUCE_SCATTER.get()
    if not _ranks_agree(enabled):
        if enabled:
            logger.warning(
                "SP hierarchical reduce-scatter off: SGLANG_TPU_SP_HIER_REDUCE_SCATTER "
                "is not set on every rank."
            )
        return
    import torch.distributed as dist

    from sglang.srt.distributed.parallel_state import get_tp_group
    from sglang.srt.runtime_context import get_server_args

    tp_group = get_tp_group()
    reason = None
    if tp_group.world_size != NUM_DEVICES or dist.get_world_size() != NUM_DEVICES:
        reason = (
            f"TP {tp_group.world_size} over {dist.get_world_size()} ranks; it needs "
            f"TP = world = {NUM_DEVICES}"
        )
    elif get_server_args().nnodes != 1:
        reason = f"{get_server_args().nnodes} hosts; it needs one tpu7x 2x2x1 host"
    elif not _is_tpu7x():
        reason = "not a tpu7x host"
    op = None
    if reason is None:
        op = _build_op()
        reason = _self_check(op, tp_group)
    # Every rank must take the same path, or the Pallas barrier hangs.
    if not _ranks_agree(reason is None):
        reason = reason or "another rank fell back"
    if reason is not None:
        logger.warning("SP hierarchical reduce-scatter off: %s.", reason)
        return
    _OP, _READY = op, True
    logger.info(
        "SP hierarchical reduce-scatter on: bf16 sequence-parallel stage "
        "reduce-scatters of a multiple of %d rows take the tpu7x Pallas kernel.",
        NUM_DEVICES * _ROW_BLOCK,
    )
