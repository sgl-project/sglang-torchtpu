# SPDX-License-Identifier: Apache-2.0
"""Static-scale fp8 for Kimi-K3's RMSNorm-fed decode GEMVs on TPU (opt-in).

Decode GEMVs are bound by the weight read. Each GEMV handled here reads an
RMSNorm output y = x / rms(x) * g, so |y_i| <= sqrt(K) * |g_i| bounds the fp8
input scale: no per-step amax, and the read halves. SGLANG_TPU_K3_STATIC_FP8
selects which ones:

- moe_front: the replicated latent down-proj and the shared experts' gate_up,
  merged into one GEMV (they read post_attention_layernorm's output);
- latent_up: the replicated latent up-proj (reads routed_expert_norm's output);
- kda_qkvg: the KDA fused [q, k, v, g] projection (reads input_layernorm's);
- mla_qkv_a: the MLA fused q_a / kv_a projection (reads input_layernorm's);
  not validated yet, so "all" leaves it out.

The model only holds the bound StaticFp8Weight attributes and calls them; the
bf16 weights they replace are released. Kimi-K3 on TPU already rejects online
weight updates, so nothing reads the released weights again.
"""

from __future__ import annotations

import logging
from typing import NamedTuple, Optional

import torch
from torch import nn

from sglang_tpu.fp8_linear import (
    StaticFp8Weight,
    pad_static_fp8_n,
    quantize_rmsnorm_fed_weight,
    register_static_fp8_gemv_op,
    static_fp8_gemv_enabled,
)
from sglang_tpu.kernels.static_fp8_gemv import MAX_TOKENS as STATIC_FP8_GEMV_MAX_TOKENS
from sglang_tpu.kimi_k3 import STATIC_FP8_PARTS as PARTS
from sglang_tpu.kimi_k3 import static_fp8_parts as selected_parts
from sglang_tpu.kimi_k3.host_split_op import host_split_op

logger = logging.getLogger(__name__)


def _pack(weight_kn: torch.Tensor, norm_weight: torch.Tensor) -> StaticFp8Weight:
    """Quantize, then pad N for the Pallas GEMV when it is registered."""
    return pad_static_fp8_n(quantize_rmsnorm_fed_weight(weight_kn, norm_weight))


def _dense_weight_kn(linear: nn.Module, k: int) -> Optional[torch.Tensor]:
    """An unquantized linear's weight as [K, N], or None when it is quantized,
    already released, or does not read a K-wide input.

    Linear weights are stored [out, in] (TPU only re-lays out MoE expert
    weights), so the input width must match K; the layout is not guessed
    from the shape.
    """
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    if not isinstance(getattr(linear, "quant_method", None), UnquantizedLinearMethod):
        return None
    if getattr(linear, "input_size", k) != k:
        return None
    w = linear.weight.detach()
    if w.dim() != 2 or w.shape[1] != k:  # dim != 2: released by an earlier bind
        return None
    return w.t()


def _g_spread(norm_weight: torch.Tensor) -> float:
    """max|g| / median|g|: how much the max|g| input scale over-provisions the
    typical channel (1 means no headroom is wasted)."""
    # Load-time log on a [K] vector: compute on host, TPU has no aten::median.
    g = norm_weight.detach().float().abs().cpu()
    return float(g.max() / g.median().clamp_min(1e-12))


def _plain_rmsnorm(norm: nn.Module) -> bool:
    """The |y_i| <= sqrt(K) * |g_i| bound holds for a plain RMSNorm over the
    full width; a partial-width variance breaks it."""
    return getattr(norm, "variance_size_override", None) is None


def _release(*linears: nn.Module) -> int:
    """Drop the bf16 weights a static fp8 GEMV replaced; returns bytes freed.

    The Parameter object stays (its loader attributes too); only its storage
    goes.
    """
    freed = 0
    for linear in linears:
        w = linear.weight
        freed += w.numel() * w.element_size()
        w.data = torch.empty(0, dtype=w.dtype, device=w.device)
    return freed


class HostSplitFront(NamedTuple):
    """Single-token MoE front split across the ranks of a host: one kernel
    (sglang_tpu/kernels/host_split_gemv.py) runs this rank's latent down-proj columns
    and shared gate_up in one fp8 GEMV, its router-gate experts in bf16, and
    the host exchange. Returns (latent, router logits, shared gate_up)."""

    weight: StaticFp8Weight  # [H, latent slice | shared gate_up]
    router_weight: torch.Tensor  # [H, expert slice], the gate's dtype
    op: object

    def __call__(self, x: torch.Tensor):
        w = self.weight
        return self.op(x, w.weight, w.out_scale, w.inv_x, self.router_weight)


class HostSplitUp(NamedTuple):
    """Single-token latent up-proj split by output columns across a host."""

    weight: StaticFp8Weight
    op: object

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        return self.op(x, w.weight, w.out_scale, w.inv_x)


# (hosts, ranks per host) the host split was measured on: tpu7x 2x2x4, TP32 =
# 4 hosts x 8 ranks. Other shapes keep the replicated GEMVs until measured.
VALIDATED_TOPOLOGIES = frozenset({(4, 8)})


def _check_ranks_agree(enabled: bool) -> None:
    """Every rank must take the same path: if one host missed the switch, its
    ranks would keep the replicated GEMVs while the others wait in the host
    barrier, and the first single-token decode would hang. Fail at startup
    instead."""
    import torch.distributed as dist

    if not dist.is_initialized():
        return
    from sglang.srt.distributed.parallel_state import get_world_group

    world_group = get_world_group()
    if world_group.world_size <= 1:
        return
    seen = [None] * world_group.world_size
    dist.all_gather_object(seen, bool(enabled), group=world_group.cpu_group)
    if len(set(seen)) > 1:
        on = [rank for rank, flag in enumerate(seen) if flag]
        raise RuntimeError(
            "SGLANG_TPU_ENABLE_K3_HOST_SPLIT_GEMV must be set the same on every "
            f"rank; it is on for ranks {on} only."
        )


def _host_split_layout() -> Optional[tuple[int, int, int]]:
    """(world, ranks per host, this rank's host-local index) when the host
    split applies, else None. Called once per process from bind_static_fp8,
    on every rank that selected static fp8 parts (all of them when the switch
    is on: static_fp8_parts requires moe_front and latent_up)."""
    from sglang_tpu.envs import envs as tpu_envs

    enabled = tpu_envs.SGLANG_TPU_ENABLE_K3_HOST_SPLIT_GEMV.get()
    _check_ranks_agree(enabled)
    if not enabled:
        return None
    import torch.distributed as dist

    from sglang.srt.distributed.parallel_state import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    from sglang.srt.runtime_context import get_server_args

    server_args = get_server_args()
    world = get_tensor_model_parallel_world_size()
    nnodes = server_args.nnodes
    # TP spanning the world is host-major, with world / nnodes ranks per host.
    if (
        world != dist.get_world_size()
        or world % nnodes
        or (nnodes, world // nnodes) not in VALIDATED_TOPOLOGIES
    ):
        logger.warning(
            "SGLANG_TPU_ENABLE_K3_HOST_SPLIT_GEMV is set but TP %d over %d hosts "
            "is not a measured host-major topology spanning the world (hosts, "
            "ranks per host in %s); the GEMVs stay replicated.",
            world,
            nnodes,
            sorted(VALIDATED_TOPOLOGIES),
        )
        return None
    compile_bs = server_args.tpu_compile_bs
    if compile_bs and 1 not in compile_bs:
        logger.warning(
            "SGLANG_TPU_ENABLE_K3_HOST_SPLIT_GEMV is set but the TPU compile "
            "batch sizes %s have no batch size 1, so no compiled decode step "
            "takes the host-split GEMVs.",
            compile_bs,
        )
    local = world // nnodes
    return world, local, get_tensor_model_parallel_rank() % local


def _split_ranks(n: int, local: int) -> int:
    """Most host ranks (<= local) that split n columns into equal 128-multiples."""
    for used in range(local, 0, -1):
        if n % (used * 128) == 0:
            return used
    return 0


def _slice_bounds(n: int, used: int, index: int) -> tuple[int, int]:
    """Slice ``index`` of ``used`` equal slices of n. Ranks past ``used`` take
    slice 0, which the exchange drops: every rank must hold the same kind of
    tensor so that all ranks compile the same program."""
    size = n // used
    start = (index if index < used else 0) * size
    return start, start + size


def _column_slice(w_kn: torch.Tensor, used: int, index: int) -> torch.Tensor:
    # A fresh base tensor: a view would reach the graph as a per-rank
    # as_strided offset, and ranks compiling different programs can order
    # the collectives differently and deadlock.
    start, end = _slice_bounds(w_kn.shape[1], used, index)
    return w_kn[:, start:end].clone(memory_format=torch.contiguous_format)


def _bind_host_split(moe, in_norm_weight, down, gate_up, up, layout) -> None:
    """Bind the single-token host-split variants next to the replicated
    StaticFp8Weights, which stay for multi-token batches."""
    world, local, index = layout
    if down is not None and gate_up is not None:
        router_kn = moe.gate.weight.detach().t()  # [H, E]
        used_l = _split_ranks(down.shape[1], local)
        used_e = _split_ranks(router_kn.shape[1], local)
        if used_l and used_e:
            latent = _column_slice(down, used_l, index)
            moe._host_split_front = HostSplitFront(
                weight=_pack(torch.cat([latent, gate_up], dim=1), in_norm_weight),
                router_weight=_column_slice(router_kn, used_e, index),
                op=host_split_op(world, local, latent.shape[1], used_l, used_e),
            )
    if up is not None:
        used = _split_ranks(up.shape[1], local)
        if used:
            weight = _pack(
                _column_slice(up, used, index), moe.routed_expert_norm.weight
            )
            moe._host_split_latent_up = HostSplitUp(
                weight=weight,
                op=host_split_op(world, local, weight.weight.shape[1], used),
            )


def _bind_moe(moe, in_norm: nn.Module, parts, layout) -> tuple[list[str], int]:
    """Returns (parts bound on this layer, bytes freed)."""
    bound: list[str] = []
    freed = 0
    if not moe.use_latent_moe:
        return bound, freed
    in_norm_weight = in_norm.weight
    hidden = in_norm_weight.shape[0]
    if (
        "moe_front" in parts
        and _plain_rmsnorm(in_norm)
        and moe.shared_experts is not None
        and not moe._shared_experts_tp_comm
        and not moe._sbo_shared_overlap
        and not moe._eligible_for_fused_front
    ):
        down = _dense_weight_kn(moe.routed_expert_down_proj, hidden)
        gate_up = _dense_weight_kn(moe.shared_experts.gate_up_proj, hidden)
        if down is not None and gate_up is not None:
            # One GEMV for both: they read the same normed hidden state, so the
            # activation is quantized once.
            moe._static_fp8_front = _pack(
                torch.cat([down, gate_up], dim=1), in_norm_weight
            )
            moe._static_fp8_front_split = [down.shape[1], gate_up.shape[1]]
            if layout is not None:
                _bind_host_split(moe, in_norm_weight, down, gate_up, None, layout)
            freed += _release(
                moe.routed_expert_down_proj, moe.shared_experts.gate_up_proj
            )
            bound.append("moe_front")
    if (
        "latent_up" in parts
        and moe.routed_expert_norm is not None
        and _plain_rmsnorm(moe.routed_expert_norm)
    ):
        up = _dense_weight_kn(moe.routed_expert_up_proj, moe.moe_hidden_size)
        if up is not None:
            moe._static_fp8_latent_up = _pack(up, moe.routed_expert_norm.weight)
            if layout is not None:
                _bind_host_split(moe, None, None, None, up, layout)
            freed += _release(moe.routed_expert_up_proj)
            bound.append("latent_up")
    return bound, freed


def _bind_linear(owner, attr: str, linear: Optional[nn.Module], norm: nn.Module):
    if linear is None or not _plain_rmsnorm(norm):
        return 0, 0
    w = _dense_weight_kn(linear, norm.weight.shape[0])
    if w is None or w.shape[1] % 16:
        return 0, 0
    setattr(owner, attr, _pack(w, norm.weight))
    return 1, _release(linear)


def bind_static_fp8(model: nn.Module) -> None:
    """Bind the selected static fp8 GEMVs on every Kimi-K3 decoder layer.

    Runs post-load, before compilation. Idempotent: post_load_weights can run
    twice (from load_weights and from the multimodal wrapper), and the second
    call must not touch the released weights.
    """
    parts = selected_parts()
    if not parts or getattr(model, "_static_fp8_bound", False):
        return
    model._static_fp8_bound = True
    register_static_fp8_gemv_op()
    from sglang.srt.layers.utils import PPMissingLayer
    from sglang.srt.models.kimi_k3 import (
        KimiK3DeltaAttention,
        KimiK3MLAAttention,
        KimiK3MoE,
    )

    counts = dict.fromkeys(sorted(parts), 0)
    spreads = {part: [] for part in counts}
    freed = 0
    layout = _host_split_layout()
    split_counts = {"front": 0, "latent_up": 0}
    for layer in model.model.layers:
        if isinstance(layer, PPMissingLayer):
            continue
        if isinstance(layer.mlp, KimiK3MoE):
            bound, f = _bind_moe(
                layer.mlp, layer.post_attention_layernorm, parts, layout
            )
            if layout is not None:
                split_counts["front"] += layer.mlp._host_split_front is not None
                split_counts["latent_up"] += layer.mlp._host_split_latent_up is not None
            for part in bound:
                counts[part] += 1
                norm_w = (
                    layer.post_attention_layernorm.weight
                    if part == "moe_front"
                    else layer.mlp.routed_expert_norm.weight
                )
                spreads[part].append(_g_spread(norm_w))
            freed += f
        attn = layer.self_attn
        norm = layer.input_layernorm
        if (
            "kda_qkvg" in parts
            and isinstance(attn, KimiK3DeltaAttention)
            and attn.use_full_rank_gate
        ):
            n, f = _bind_linear(
                attn, "_static_fp8_qkvg", getattr(attn, "fused_qkvg_proj", None), norm
            )
            counts["kda_qkvg"] += n
            if n:
                spreads["kda_qkvg"].append(_g_spread(norm.weight))
            freed += f
        if "mla_qkv_a" in parts and isinstance(attn, KimiK3MLAAttention):
            n, f = _bind_linear(
                attn,
                "_static_fp8_qkv_a",
                getattr(attn, "fused_qkv_a_proj_with_mqa", None),
                norm,
            )
            counts["mla_qkv_a"] += n
            if n:
                spreads["mla_qkv_a"].append(_g_spread(norm.weight))
            freed += f
    if freed and not getattr(model, "_derived_weight_cache_error", None):
        model._derived_weight_cache_error = (
            "Online weight updates are not supported with SGLANG_TPU_K3_STATIC_FP8: "
            "the bf16 weights its fp8 GEMVs replace were released at load time."
        )
    logger.info(
        "Kimi-K3 static fp8 GEMVs bound per part (layers): %s; freed %.1f MiB of "
        "bf16 weights",
        counts,
        freed / 2**20,
    )
    spread_summary = {
        part: f"median {sorted(v)[len(v) // 2]:.1f}, max {max(v):.1f}"
        for part, v in spreads.items()
        if v
    }
    if spread_summary:
        logger.info(
            "Static fp8 input-scale headroom, max|g| / median|g| per part over "
            "the bound layers: %s",
            spread_summary,
        )
    if layout is not None:
        logger.info(
            "Kimi-K3 host-split single-token GEMVs (host-local index %d of %d) "
            "bound per part (layers): %s",
            layout[2],
            layout[1],
            split_counts,
        )
    unbound = sorted(part for part, n in counts.items() if n == 0)
    if unbound:
        logger.warning(
            "SGLANG_TPU_K3_STATIC_FP8 selected %s, but no layer qualified "
            "(quantized, wrong layout or a partial-width norm); they stay bf16.",
            unbound,
        )
    if freed:
        if static_fp8_gemv_enabled():
            logger.info(
                "Static fp8 GEMVs run the Pallas kernel for up to %d tokens; "
                "larger calls (prefill, bigger decode or spec-verify batches) "
                "use the XLA fp8 dot.",
                STATIC_FP8_GEMV_MAX_TOKENS,
            )
        else:
            logger.info(
                "Static fp8 GEMVs use the XLA fp8 dot for every batch "
                "(SGLANG_TPU_ENABLE_K3_STATIC_FP8_PALLAS_GEMV=0)."
            )


__all__ = ["PARTS", "StaticFp8Weight", "bind_static_fp8", "selected_parts"]
