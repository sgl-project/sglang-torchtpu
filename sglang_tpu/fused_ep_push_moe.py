# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4/V4.1 FP4 experts on the push-dispatch fused EP MoE kernel.

Selected with ``--moe-runner-backend pallas_tpu_fused_ep_push``, as
``pallas_tpu_fused_rs`` and ``pallas_tpu_fused_decode`` are. The layer runs
``fused_ep_moe_v2(dispatch="push")`` instead of gmm_v2.

Routing: the kernel is handed core's selected expert ids as ``hash_topk_ids``,
so it selects exactly the experts the gmm_v2 path does (hash and biased top-k
layers alike); it recomputes their weights from the router logits with
sqrtsoftplus and renormalization, as core's DeepSeek-V4 routers do. Core
applies ``routed_scaling_factor`` after the experts on TPU, so the kernel is
called with 1.0.

Data flow under attention DP: the MoE input is the DP-gathered batch, the same
rows on every EP rank. The kernel is told so (one token replica group over the
whole mesh): each rank computes a disjoint slice of the rows and the kernel
returns the combined output on every rank. Core still all-reduces the MoE
output, because the TP-sharded shared expert's partial sum rides on it, so
only EP rank 0 keeps the routed output; the all-reduce then counts it once.

Weights: the gmm_v2 packed-FP4 layout (two K codes per byte, ``[E, K/2, N]``,
scales ``[E, K/512, 1, N]``) is converted once at load to the kernel's
``float4_e2m1fn_x2`` ``[E, K, N/2]`` (two N codes per byte) with row-padded
flat f32 scales ``[E * rows, N]`` (``host.fp4_scale_rows``).
"""

from __future__ import annotations

import functools

import msgspec
import torch

PUSH_MOE_ATTR = "_tpu_fused_ep_push_moe"
PUSH_RANK_ATTR = "_tpu_fused_ep_push_rank"

_RHS_QB = 512
_CAPACITY = 128
_SCALE_ROW_TILE = 8  # host.U32_SUBLANE_TILE
# The kernel's clamped activation hardcodes this limit (activations.py).
_KERNEL_SWIGLU_LIMIT = 10.0
_EXPERTS_PER_CHUNK = 8


class PushMoeConfig(msgspec.Struct, frozen=True):
    topk: int
    ep: int
    # None when the checkpoint intermediate is already a multiple of 512.
    logical_inter: int | None


def scale_rows(k: int) -> int:
    """``host.fp4_scale_rows(k, 512, padded=True)`` without importing JAX."""
    blocks = k // _RHS_QB
    return -(-blocks // _SCALE_ROW_TILE) * _SCALE_ROW_TILE


def repack_k_to_n(packed_kn: torch.Tensor) -> torch.Tensor:
    """uint8 ``[E, K/2, N]`` (low nibble = even K) -> uint8 ``[E, K, N/2]``
    (low nibble = even N). Same codes, same byte count."""
    experts, half_k, n = packed_kn.shape
    if n % 2:
        raise ValueError(f"push MoE FP4 weights need an even N, got {n}")
    codes = torch.stack((packed_kn & 0x0F, packed_kn >> 4), dim=2).reshape(
        experts, 2 * half_k, n
    )
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def flat_padded_scales(scale: torch.Tensor, k: int) -> torch.Tensor:
    """f32 ``[E, K/512, 1, N]`` -> row-padded flat ``[E * rows, N]``."""
    experts, blocks, one, n = scale.shape
    if one != 1 or blocks != k // _RHS_QB:
        raise ValueError(
            f"expected gmm_v2 FP4 scales [E, {k // _RHS_QB}, 1, N], "
            f"got {tuple(scale.shape)}"
        )
    rows = scale_rows(k)
    out = scale.new_zeros((experts, rows, n), dtype=torch.float32)
    out[:, :blocks] = scale[:, :, 0].to(torch.float32)
    return out.reshape(experts * rows, n)


def _own_flat_scales(scale: torch.Tensor, k: int) -> torch.Tensor:
    from sglang_tpu.fp8_moe import _own_storage

    # A reshape view would keep the 3-D buffer and its layout under the 2-D shape.
    return _own_storage(flat_padded_scales(scale, k))


def _to_native_fp4(packed_kn: torch.Tensor) -> torch.Tensor:
    fp4 = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4 is None:
        raise RuntimeError(
            "pallas_tpu_fused_ep_push needs TorchTPU's torch.float4_e2m1fn_x2"
        )
    experts, half_k, n = packed_kn.shape
    out = torch.empty((experts, 2 * half_k, n // 2), dtype=fp4, device=packed_kn.device)
    for c0 in range(0, experts, _EXPERTS_PER_CHUNK):
        c1 = min(c0 + _EXPERTS_PER_CHUNK, experts)
        out[c0:c1].copy_(repack_k_to_n(packed_kn[c0:c1]).view(fp4))
    return out


def unsupported_reason(layer: torch.nn.Module) -> str | None:
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    world = torch.distributed.get_world_size()
    if parallel.moe_ep_size != world or parallel.moe_tp_size != 1:
        return (
            f"the kernel's EP mesh is the whole world: needs ep_size == world "
            f"size ({world}) and MoE TP 1, got ep={parallel.moe_ep_size} "
            f"moe_tp={parallel.moe_tp_size}"
        )
    if layer.num_fused_shared_experts:
        return "fused shared experts are not supported"
    if layer.num_local_experts * parallel.moe_ep_size != layer.num_experts:
        return (
            f"{layer.num_experts} experts do not split into equal EP blocks of "
            f"{layer.num_local_experts}"
        )
    limit = layer.moe_runner_config.swiglu_limit
    if limit is not None and float(limit) != _KERNEL_SWIGLU_LIMIT:
        return (
            f"the kernel clamps SwiGLU at {_KERNEL_SWIGLU_LIMIT}, the model at {limit}"
        )
    if layer.w13_weight.dtype != torch.uint8:
        return "only gmm_v2 packed-FP4 (DeepSeek-V4 MXFP4) experts are converted"
    return None


def prepare_fused_ep_push_moe(layer: torch.nn.Module) -> None:
    """Convert a layer's gmm_v2 FP4 experts to the push kernel's operands."""
    from sglang.srt.layers.utils import copy_or_rebind_param
    from sglang.srt.runtime_context import get_parallel
    from sglang_tpu.fp8_moe import _free_replaced_storage

    reason = unsupported_reason(layer)
    if reason is not None:
        raise ValueError(
            f"--moe-runner-backend pallas_tpu_fused_ep_push: {reason}. For the "
            "DSpark draft, set --speculative-moe-runner-backend pallas_tpu."
        )
    hidden = layer.w13_weight.shape[1] * 2
    inter = layer.w2_weight.shape[1] * 2
    w13_scale = _own_flat_scales(layer.w13_weight_scale.data, hidden)
    w2_scale = _own_flat_scales(layer.w2_weight_scale.data, inter)
    w13 = _to_native_fp4(layer.w13_weight.data)
    w2 = _to_native_fp4(layer.w2_weight.data)
    for name, value in (
        ("w13_weight", w13),
        ("w2_weight", w2),
        ("w13_weight_scale", w13_scale),
        ("w2_weight_scale", w2_scale),
    ):
        old = getattr(layer, name).data
        copy_or_rebind_param(layer, name, value)
        _free_replaced_storage(old, getattr(layer, name).data)
    # A graph input, not a trace constant: the SPMD-compiled program serves
    # every rank, which selects its rows and the routed-output owner by it.
    layer.register_buffer(
        PUSH_RANK_ATTR,
        torch.tensor(
            [[get_parallel().moe_ep_rank]], dtype=torch.int32, device=w13.device
        ),
        persistent=False,
    )
    logical = layer.intermediate_size_per_partition
    layer.__dict__[PUSH_MOE_ATTR] = PushMoeConfig(
        topk=int(layer.top_k),
        ep=int(get_parallel().moe_ep_size),
        logical_inter=None if logical == inter else int(logical),
    )


def uses_fused_ep_push_moe(layer: torch.nn.Module) -> bool:
    return PUSH_MOE_ATTR in layer.__dict__


@functools.cache
def _push_op(*, topk: int, ep: int, act_fn: str, logical_inter: int | None):
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal.pallas import get_pallas_mesh

    from sglang_tpu.moe.pallas.fused_moe_v2 import WeightFormat, fused_ep_moe_v2
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    mesh = get_pallas_mesh(("model",), (ep,))
    replicas = (tuple(range(ep)),)

    def moe(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        gating: jax.Array,
        hash_ids: jax.Array,
        rank: jax.Array,
    ) -> jax.Array:
        return fused_ep_moe_v2(
            x,
            w1,
            w2,
            w1_scale,
            w2_scale,
            gating.astype(jnp.float32),
            topk=topk,
            renormalize=True,
            mesh=mesh,
            capacity=_CAPACITY,
            weight_format=WeightFormat.FP4,
            rhs_qb=_RHS_QB,
            act_fn=act_fn,
            rank=rank,
            sharded_plan=True,
            scoring_fn="sqrtsoftplus",
            hash_topk_ids=hash_ids,
            routed_scaling_factor=1.0,
            token_replica_groups=replicas,
            logical_intermediate_size=logical_inter,
            dispatch="push",
            push_options={},
        )

    suffix = f"k{topk}_ep{ep}_{act_fn}_i{logical_inter or 0}"
    return cpu_export_jax_op(
        f"sglang_tpu_pallas::fused_ep_push_moe_{suffix}",
        moe,
        mesh=mesh,
        input_partition_specs=(P("model"),) * 8,
        output_partition_specs=(P("model"),),
    )


def apply_fused_ep_push_moe(
    *, layer: torch.nn.Module, dispatch_output, swiglu_limit: float | None
):
    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

    cfg = layer.__dict__[PUSH_MOE_ATTR]
    x = dispatch_output.hidden_states
    _, topk_ids, router_logits = dispatch_output.topk_output
    if router_logits is None or router_logits.shape[-1] != layer.num_experts:
        raise RuntimeError(
            "pallas_tpu_fused_ep_push needs the router logits over all "
            f"{layer.num_experts} experts in the top-k output"
        )
    topk_ids = topk_ids.to(torch.int32)
    # Padded rows carry id -1; the kernel skips rows whose logits are all -inf.
    live = (topk_ids >= 0).all(dim=-1, keepdim=True)
    gating = torch.where(
        live, router_logits.float(), torch.full_like(router_logits.float(), -torch.inf)
    )
    hash_ids = torch.where(live, topk_ids, torch.zeros_like(topk_ids))
    act_fn = "silu" if swiglu_limit is None else "silu_and_mul_with_clamp"
    op = _push_op(
        topk=cfg.topk, ep=cfg.ep, act_fn=act_fn, logical_inter=cfg.logical_inter
    )
    rank = getattr(layer, PUSH_RANK_ATTR)
    out = op(
        x,
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale,
        layer.w2_weight_scale,
        gating,
        hash_ids,
        rank,
    )
    # Every rank holds the combined routed output; core's all-reduce follows.
    owner = (rank.view(()) == 0).to(out.dtype)
    return StandardCombineInput(hidden_states=out * owner)
