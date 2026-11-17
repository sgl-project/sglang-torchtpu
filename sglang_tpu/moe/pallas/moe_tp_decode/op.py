# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# NOTE: do *not* add `from __future__ import annotations` here --
# `sglang_tpu.pallas_jax_op.cpu_export_jax_op` inspects
# `param.annotation is jax.Array` at registration time.
"""TorchTPU custom op and layer preparation for the MoE-TP `moe_tp_decode` kernel.

`fused_moe_decode_op` dispatches here when `--moe-runner-backend
pallas_tpu_fused_decode` is set and the MoE runs tensor-parallel experts
(`moe_ep_size == 1`, attention TP over the whole world). The op takes SGLang's
top-k output, builds the kernel's dense routing table, pads the batch to a
kernel row bucket and returns the all-reduced layer output.
"""

import functools
import logging
from threading import Lock
from typing import Callable, Optional

import torch

from sglang.srt.runtime_context import get_parallel
from sglang_tpu.rank_local import traced_rank

logger = logging.getLogger(__name__)

_MESH_AXIS = "model"
_OPS: dict[tuple, Callable] = {}
_OPS_LOCK = Lock()


def moe_tp_layout_reason() -> Optional[str]:
    """Why the current parallel layout cannot run `moe_tp_decode`, or None."""
    parallel = get_parallel()
    if parallel.moe_ep_size != 1:
        return f"moe_ep_size={parallel.moe_ep_size} (needs MoE-TP, ep_size=1)"
    if parallel.tp_size < 2:
        return "tp_size < 2"
    if parallel.attn_tp_size != parallel.tp_size or parallel.attn_dp_size != 1:
        return "attention DP (the in-kernel all-reduce spans the whole TP group)"
    if parallel.pp_size != 1 or parallel.attn_cp_size != 1:
        return "pipeline or context parallelism"
    return None


@functools.cache
def _get_mesh(world_size: int):
    from torch_tpu._internal.pallas import get_pallas_mesh

    return get_pallas_mesh((_MESH_AXIS,), (world_size,))


def _build_op(
    world_size: int,
    num_experts: int,
    routed_scaling_factor: float,
    swiglu_limit: Optional[float],
) -> Callable:
    import jax
    from jax.sharding import PartitionSpec

    from sglang_tpu.moe.pallas.moe_tp_decode.kernel import moe_tp_decode
    from sglang_tpu.moe.pallas.moe_tp_decode.routing import kernel_inputs
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    mesh = _get_mesh(world_size)
    spec = PartitionSpec(_MESH_AXIS)
    scale_tag = f"{routed_scaling_factor:.6g}".replace(".", "_").replace("-", "m")
    limit_tag = (
        "none"
        if swiglu_limit is None
        else f"{swiglu_limit:.6g}".replace(".", "_").replace("-", "m")
    )
    op_name = (
        f"sglang_tpu_pallas::moe_tp_decode_tp{world_size}_e{num_experts}"
        f"_s{scale_tag}_l{limit_tag}"
    )

    def _body(
        x: jax.Array,
        topk_ids: jax.Array,
        topk_weights: jax.Array,
        w1: jax.Array,
        w1_scale: jax.Array,
        w2: jax.Array,
        w2_scale: jax.Array,
        s_w1: jax.Array,
        s_w1_scale: jax.Array,
        s_w2: jax.Array,
        s_w2_scale: jax.Array,
        rank: jax.Array,
    ) -> jax.Array:
        m = x.shape[0]
        # Every rank sees the same batch, so n_valid agrees across ranks.
        x, wtab, n_valid = kernel_inputs(x, topk_ids, topk_weights, num_experts)
        out = moe_tp_decode(
            x,
            wtab,
            n_valid,
            rank[0],
            w1,
            w2,
            w1_scale,
            w2_scale,
            s_w1,
            s_w1_scale.reshape(-1),
            s_w2,
            s_w2_scale.reshape(-1),
            routed_scale=routed_scaling_factor,
            swiglu_limit=swiglu_limit,
            n_ranks=world_size,
        )
        return out[:m]

    sharded = jax.shard_map(
        _body, mesh=mesh, in_specs=(spec,) * 12, out_specs=spec, check_vma=False
    )

    def _jax_fn(
        x: jax.Array,
        topk_ids: jax.Array,
        topk_weights: jax.Array,
        w1: jax.Array,
        w1_scale: jax.Array,
        w2: jax.Array,
        w2_scale: jax.Array,
        s_w1: jax.Array,
        s_w1_scale: jax.Array,
        s_w2: jax.Array,
        s_w2_scale: jax.Array,
        rank: jax.Array,
    ) -> jax.Array:
        return sharded(
            x,
            topk_ids,
            topk_weights,
            w1,
            w1_scale,
            w2,
            w2_scale,
            s_w1,
            s_w1_scale,
            s_w2,
            s_w2_scale,
            rank,
        )

    return cpu_export_jax_op(
        op_name,
        _jax_fn,
        mesh=mesh,
        input_partition_specs=(spec,) * 12,
        output_partition_specs=(spec,),
    )


def _get_op(
    world_size: int,
    num_experts: int,
    routed_scaling_factor: float,
    swiglu_limit: Optional[float],
) -> Callable:
    key = (
        int(world_size),
        int(num_experts),
        float(routed_scaling_factor),
        None if swiglu_limit is None else float(swiglu_limit),
    )
    op = _OPS.get(key)
    if op is not None:
        return op
    with _OPS_LOCK:
        op = _OPS.get(key)
        if op is None:
            op = _build_op(*key)
            _OPS[key] = op
    return op


def admitted_max_tokens(experts: torch.nn.Module, world_size: int) -> int:
    """Largest kernel row bucket the kernel admits for `experts`, 0 if none.

    Evaluated on shapes only (no device work): hardware, alignment, the
    all-reduce geometry and the VMEM plan.
    """
    import jax
    import jax.numpy as jnp

    from sglang_tpu.moe.pallas.moe_tp_decode.kernel import (
        ROW_BUCKETS,
        can_use_moe_tp_decode,
    )

    def sds(t: torch.Tensor, dtype=None, shape=None):
        jdtype = {
            torch.float8_e4m3fn: jnp.float8_e4m3fn,
            torch.float32: jnp.float32,
            torch.bfloat16: jnp.bfloat16,
        }[t.dtype]
        return jax.ShapeDtypeStruct(
            tuple(t.shape) if shape is None else shape, dtype or jdtype
        )

    w13, w2 = experts.w13_weight, experts.w2_weight
    num_experts, h = int(w13.shape[0]), int(w13.shape[1])
    args = (
        sds(w13),
        sds(w2),
        sds(experts._fmd_w1_scale),
        sds(experts._fmd_w2_scale),
        sds(experts._fmd_sw1),
        sds(experts._fmd_s1s, shape=(experts._fmd_s1s.numel(),)),
        sds(experts._fmd_sw2),
        sds(experts._fmd_s2s, shape=(experts._fmd_s2s.numel(),)),
    )
    best = 0
    for m in ROW_BUCKETS:
        x = jax.ShapeDtypeStruct((m, h), jnp.bfloat16)
        wtab = jax.ShapeDtypeStruct((m, num_experts), jnp.float32)
        if can_use_moe_tp_decode(x, wtab, *args, n_ranks=world_size):
            best = m
    return best


def prepare_moe_tp_decode(
    moe_layer: torch.nn.Module,
    experts: torch.nn.Module,
    routed_scaling_factor: float,
    swiglu_limit: Optional[float],
) -> int:
    """Register the TP rank and build the op; returns the max admitted tokens (0: not admitted).

    Expects `fused_moe_decode_op` to have registered the routed scale buffers
    (`_fmd_w1_scale`, `_fmd_w2_scale`) and the 128-padded shared expert
    (`_fmd_sw1`, `_fmd_s1s`, `_fmd_sw2`, `_fmd_s2s`) on `experts`.
    """
    parallel = get_parallel()
    world_size = int(parallel.tp_size)
    max_tokens = admitted_max_tokens(experts, world_size)
    if max_tokens == 0:
        return 0
    # The kernel addresses its all-reduce peers by mesh index, which is the TP
    # rank here (attention TP spans the world; see moe_tp_layout_reason).
    rank = torch.tensor(
        [int(parallel.tp_rank)], dtype=torch.int32, device=experts.w13_weight.device
    )
    experts.register_buffer("_fmt_rank", rank, persistent=False)
    _get_op(
        world_size,
        int(experts.w13_weight.shape[0]),
        routed_scaling_factor,
        swiglu_limit,
    )
    return max_tokens


def apply_moe_tp_decode(
    moe_layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    experts = moe_layer.experts
    world_size = int(moe_layer._tpu_fused_moe_decode_world_size)
    rank = traced_rank(int(get_parallel().tp_rank), group_size=world_size)
    rank = experts._fmt_rank if rank is None else rank.to(dtype=torch.int32).reshape(1)
    op = _get_op(
        world_size,
        int(experts.w13_weight.shape[0]),
        float(moe_layer._tpu_fused_moe_decode_routed_scale),
        moe_layer._tpu_fused_moe_decode_swiglu_limit,
    )
    return op(
        hidden_states,
        topk_ids,
        topk_weights,
        experts.w13_weight,
        experts._fmd_w1_scale,
        experts.w2_weight,
        experts._fmd_w2_scale,
        experts._fmd_sw1,
        experts._fmd_s1s,
        experts._fmd_sw2,
        experts._fmd_s2s,
        rank,
    )
