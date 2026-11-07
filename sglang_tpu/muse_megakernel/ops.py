# SPDX-License-Identifier: Apache-2.0
"""Muse Spark megakernels: weight preparation, the torch ops, and the forward hook.

``MuseMegakernel.from_model`` runs once after weight loading, before the KV pool is sized. It
builds the stacked K-major copies the kernels stream (attention, pooler and unpooler in fp8,
the router in bf16) and collects the per-layer expert references sglang already owns. ``forward``
replaces the decoder for decode and DFlash target-verify batches and, with
SGLANG_TPU_MUSE_MK_PREFILL, for extend batches (the prefill program, prefill.py); other batches
keep the native path. The LM head, logits and sampling stay in sglang.

A decode / verify step runs as one jitted program per device: the megakernel (collective, reads
the KV pools) and then ``kv_write_call`` (local, scatters this step's K/V rows into the pools in
place). Steps of more than BASE_STEP_MAX_ROWS rows run the megakernel's row-tiled program.
"""

import logging
import math
import os

import jax
import msgspec
import torch
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.fp8_weight_linear import TpuSharedKMajorLinearMethod
from sglang_tpu.kv_pool import (
    TpuFusedMHATokenToKVPool,
    TpuSeqAlongLaneKVPool,
    kv_pack_heads,
)
from sglang_tpu.muse_megakernel import kernel as K

logger = logging.getLogger(__name__)

# Profiling only: in-kernel region names for xprof (kernel.py's _TRACE / _TRACE_INNER).
K._TRACE = tpu_envs.SGLANG_TPU_MK_TRACE_REGIONS.get() >= 1
K._TRACE_INNER = tpu_envs.SGLANG_TPU_MK_TRACE_REGIONS.get() >= 2

_OP_NS = "sglang_tpu_pallas"
# KV pages per attention chunk: 4 beats 2 on long-context verify steps and is neutral
# at short context; 8+ costs VMEM the expert ring needs.
_CHUNK_PAGES = 4
# Software-pipelined attention, chunks per loop iteration (measured at 500k context): the ring
# holds two iterations in use plus refills in flight, which need >= one iteration of slots
# (32 fp8 SAL chunks; 16 bf16 SAL or fused ones).
_ATTN_STREAMS = {"fused": 4, "sal": 4, "sal_fp8": 8}
# fused: a 16-chunk ring (4 MiB); a deeper one costs dense-ring slots
_FUSED_KV_GROUP = 8
# Larger steps run the row-tiled program: the base program keeps the residual stream, the
# collective pads and the expert outputs in VMEM, so its scratch grows with rows (at 128 rows
# no ring fits the budget).
BASE_STEP_MAX_ROWS = 32
# Row-tiled steps: rows per pass, and the padding unit of their rows.
_ROW_TILE = 16


def _dev_int(name: str, default: int) -> int:
    """A tuning knob for kernel sweeps (anvil --env); unset means the tuned default."""
    return int(os.environ.get(name, default))


def _row_tile() -> int:
    return _dev_int("SGLANG_TPU_MUSE_MK_DEV_ROW_TILE", _ROW_TILE)


def step_rows(n_tokens: int, tp: int) -> int:
    """Padded rows of a step: a multiple of the TP size; row-tiled steps also of the row tile
    (each device then owns rows / tp rows of the expert reduce-scatter and u all-gather)."""
    rows = _round_up(n_tokens, tp)
    if rows > BASE_STEP_MAX_ROWS:
        rows = _round_up(n_tokens, math.lcm(_row_tile(), tp))
    return rows


def _k_major(weight: torch.Tensor) -> torch.Tensor:
    return weight.detach().t().contiguous()


def _fp8_channelwise(w_kn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """K-major ``[K, N]`` -> fp8 e4m3 ``[K, N]`` and f32 scales ``[N]`` (column amax / 448)."""
    w = w_kn.float()
    scale = (w.abs().amax(dim=0) / 448.0).clamp_min(1e-12)
    return (w / scale).to(torch.float8_e4m3fn), scale


def _stack(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(tensors, dim=0)


class _ModelGeometry:
    """Static model numbers the kernel needs, read from the loaded model."""

    def __init__(self, model) -> None:
        cfg = model.text_config
        decoder = model.model
        layer0 = decoder.layers[0]
        attn = layer0.self_attn
        moe = layer0.mlp
        self.n_layers = len(decoder.layers)
        self.hidden = cfg.hidden_size
        self.head_dim = attn.head_dim
        self.q_heads = attn.num_heads
        self.kv_heads = attn.num_kv_heads
        self.n_experts = moe.num_experts
        self.top_k = moe.top_k
        self.expert_hidden = moe.expert_hidden_size
        # w2 is [E, I_local, H_e] in the fused-RS expert-TP layout (fp4 packs the last dim)
        self.inter_local = moe.experts.w2_weight.shape[1]
        self.rms_eps = cfg.rms_norm_eps
        self.post_eps = cfg.post_norm_eps
        self.scale_query_by = attn.query_scale
        self.attn_scale = attn.head_dim**-0.5
        self.router_mult = moe.router_multiplier
        self.route_scale = moe.route_scale
        self.router_eps = moe.route_eps
        self.global_layers = tuple(
            i for i, t in enumerate(cfg.layer_types) if t == "full_attention"
        )
        self.window = cfg.sliding_window
        self.rope_layers = tuple(
            i for i, theta in enumerate(cfg.layer_rope_theta) if theta > 0
        )
        if set(self.rope_layers) != set(range(self.n_layers)) - set(self.global_layers):
            raise NotImplementedError("RoPE must be exactly the sliding-window layers")


def _expert_kind(experts) -> str:
    dt = experts.w13_weight.dtype
    if dt == torch.float4_e2m1fn_x2:
        return "fp4"
    if dt == torch.float8_e4m3fn:
        return "fp8"
    raise NotImplementedError(f"unsupported expert weight dtype {dt}")


def _expert_refs(experts, kind: str) -> list[torch.Tensor]:
    """``(w13, w13_scale, w2, w2_scale)`` in the fused-RS expert-TP layout."""
    s13, s2 = experts.w13_weight_scale, experts.w2_weight_scale
    if kind == "fp8" and s13.dim() == 4:
        # channelwise (E, 1, 1, N): give the kernel the (E, 2, N) form the fused-RS path uses
        s13 = s13.reshape(s13.shape[0], 1, s13.shape[-1]).expand(-1, 2, -1).contiguous()
        s2 = s2.reshape(s2.shape[0], 1, s2.shape[-1]).expand(-1, 2, -1).contiguous()
    return [experts.w13_weight, s13, experts.w2_weight, s2]


def expert_groups_of(kinds: tuple, size: int) -> tuple:
    """[lo, hi) runs of at most ``size`` consecutive fp4 layers (() when size is 0)."""
    if size <= 0:
        return ()
    fp4 = [l for l, k in enumerate(kinds) if k == "fp4"]
    return tuple(
        (fp4[i], fp4[min(i + size, len(fp4)) - 1] + 1) for i in range(0, len(fp4), size)
    )


_EXPERT_PARAMS = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")


def _perm_256_cpu(
    t: torch.Tensor, dim: int, *, packed_fp4: bool = False
) -> torch.Tensor:
    """Permute each 256-slice along ``dim`` from ``(16, 16)`` to ``(16, 16)^T`` on CPU."""
    dim = dim % t.ndim
    if packed_fp4 and dim == t.ndim - 1:
        lo = t & 0x0F
        hi = (t >> 4) & 0x0F
        u = torch.stack((lo, hi), dim=-1).reshape(*t.shape[:-1], t.shape[-1] * 2)
        d = u.shape[-1]
        if d % 256:
            raise ValueError(f"_perm_256_cpu requires multiple of 256, got {d}")
        u = (
            u.reshape(*u.shape[:-1], d // 256, 16, 16)
            .transpose(-2, -1)
            .reshape(u.shape)
        )
        return (u[..., 0::2] | (u[..., 1::2] << 4)).contiguous()
    sh = t.shape
    d = sh[dim]
    if d % 256:
        raise ValueError(f"_perm_256_cpu requires multiple of 256, got {d}")
    return (
        t.reshape(*sh[:dim], d // 256, 16, 16, *sh[dim + 1 :])
        .transpose(dim + 1, dim + 2)
        .reshape(sh)
        .contiguous()
    )


def _perm_expert_cpu(j: int, cpu_t: torch.Tensor, *, raw: bool) -> torch.Tensor:
    if j == 0:
        return _perm_256_cpu(
            _perm_256_cpu(cpu_t, -2, packed_fp4=raw), -1, packed_fp4=raw
        )
    if j == 1:
        return _perm_256_cpu(cpu_t, -1)
    if j == 2:
        return _perm_256_cpu(cpu_t, -2, packed_fp4=raw)
    return cpu_t


def _stack_expert_groups(
    decoder, experts: list, groups: tuple, *, kinds: tuple = (), kperm: bool = False
) -> list:
    """Per-layer expert refs (4 per layer) -> refs in kernel unit order.

    Each group's slabs are staged on CPU while freeing the replaced per-layer TPU
    parameters immediately, viewed as ``float4_e2m1fn_x2`` on CPU before upload, and
    synchronized as pure leaf PJRT buffers. Avoids TorchTPU 0.2 lazy ``torch.stack``
    concatenation cascades and ``as_strided`` re-materialization of bound fp4 views;
    both ``muse_prefill`` and native ``FusedMoE`` receive whole stacks and index their
    layer inside Pallas via ``group_offset[1]``.
    """
    if not groups and not kperm:
        return experts
    from sglang_tpu.fp8_moe import _clear_replaced_param, copy_or_rebind_param

    per_layer = [experts[4 * l : 4 * l + 4] for l in range(len(decoder.layers))]
    experts.clear()
    grouped_layers = {li for lo, hi in groups for li in range(lo, hi)}
    if kperm:
        for li, kind in enumerate(kinds):
            if kind != "fp4" or li in grouped_layers:
                continue
            exp_mod = decoder.layers[li].mlp.experts
            for j, name in enumerate(_EXPERT_PARAMS[:3]):
                src = per_layer[li][j]
                device, dtype = src.device, src.dtype
                raw = dtype == torch.float4_e2m1fn_x2
                cpu_t = (src.view(torch.uint8) if raw else src).to("cpu")
                cpu_t = _perm_expert_cpu(j, cpu_t, raw=raw)
                new_t = (cpu_t.view(dtype) if raw else cpu_t).to(device=device)
                del cpu_t
                copy_or_rebind_param(exp_mod, name, new_t)
                per_layer[li][j] = getattr(exp_mod, name)
            torch.accelerator.synchronize()
    if not groups:
        return [t for layer_refs in per_layer for t in layer_refs]
    stacked = {}
    for lo, hi in groups:
        slabs = []
        do_kperm = kperm and bool(kinds) and kinds[lo] == "fp4"
        for j, name in enumerate(_EXPERT_PARAMS):
            first = per_layer[lo][j]
            device = first.device
            dtype = first.dtype
            raw = dtype == torch.float4_e2m1fn_x2
            elem_shape = first.view(torch.uint8).shape if raw else first.shape
            cpu_slab = torch.empty(
                (hi - lo, *elem_shape),
                dtype=torch.uint8 if raw else dtype,
                device="cpu",
            )
            for k, li in enumerate(range(lo, hi)):
                src = per_layer[li][j]
                cpu_slab[k].copy_((src.view(torch.uint8) if raw else src).to("cpu"))
                per_layer[li][j] = None
                _clear_replaced_param(decoder.layers[li].mlp.experts, name)
            if do_kperm and j < 3:
                cpu_slab = _perm_expert_cpu(j, cpu_slab, raw=raw)
            st = (cpu_slab.view(dtype) if raw else cpu_slab).to(device=device)
            del cpu_slab
            torch.accelerator.synchronize()
            for k, li in enumerate(range(lo, hi)):
                exp_mod = decoder.layers[li].mlp.experts
                exp_mod.register_buffer(f"_stacked_{name}", st, persistent=False)
                if j == 0:
                    old_go = getattr(exp_mod, "_tpu_fused_rs_group_offset", None)
                    if old_go is not None:
                        go0 = int(old_go.reshape(-1)[0].item())
                        exp_mod.register_buffer(
                            "_tpu_fused_rs_group_offset",
                            torch.tensor([go0, k], dtype=torch.int32, device=device),
                            persistent=False,
                        )
            slabs.append(st)
        stacked[lo] = slabs
    out, l = [], 0
    starts = dict(groups)
    while l < len(decoder.layers):
        if l in starts:
            out.extend(stacked[l])
            l = starts[l]
        else:
            out.extend(per_layer[l])
            l += 1
    return out


class MuseMegakernel(torch.nn.Module):
    """Holds the kernels' weights and one compiled op per step shape (decode / verify:
    rows, rows per request, capture; prefill: rows, capture)."""

    def __init__(
        self,
        *,
        geo: _ModelGeometry,
        weights: dict,
        experts: list,
        kinds: tuple,
        tp_size: int,
        tp_rank: int,
        max_rows: int,
        max_position: int,
        scale_fp8: bool,
        expert_groups: tuple = (),
        kperm: bool = False,
    ) -> None:
        super().__init__()
        self.geo = geo
        self.kinds = kinds
        self.tp_size = tp_size
        self.max_rows = max_rows
        self.max_position = max_position
        self.scale_fp8 = scale_fp8
        self.expert_groups = expert_groups
        self.kperm = kperm
        self.chunk_pages = _CHUNK_PAGES
        # the pools' layout: SGLANG_TPU_KV_PACK_HEADS stores the rank's heads in one row
        self.kv_packed = (
            kv_pack_heads(
                geo.kv_heads, geo.head_dim, TpuFusedMHATokenToKVPool.HEAD_DIM_ALIGNMENT
            )
            > 1
        )
        self.prefill = tpu_envs.SGLANG_TPU_MUSE_MK_PREFILL.get()
        if self.prefill and scale_fp8:
            raise NotImplementedError(
                "the Muse prefill program reads bf16 NVFP4 block scales "
                "(unset SGLANG_TPU_NVFP4_E4M3_SCALES)"
            )
        # Buffers so --tpu-compile-tracer direct lifts them as graph inputs; the expert
        # entries alias FusedMoE parameters.
        for name in K.WEIGHT_NAMES:
            self.register_buffer(f"w_{name}", weights[name], persistent=False)
        for i, t in enumerate(experts):
            self.register_buffer(f"expert_{i}", t, persistent=False)
        self.n_expert_refs = len(experts)
        self.register_buffer(
            "rank_t",
            torch.tensor([tp_rank], dtype=torch.int32, device=weights["smalls"].device),
            persistent=False,
        )
        self._ops: dict = {}

    # ------------------------------------------------------------------ preparation
    @classmethod
    def from_model(cls, model) -> "MuseMegakernel":
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        if not (
            parallel.moe_ep_size == 1
            and parallel.moe_tp_size == parallel.tp_size == parallel.attn_tp_size
        ):
            raise NotImplementedError(
                "the Muse megakernel needs expert TP across the whole TP group"
            )
        # Remote DMA over one host's devices only; rows are padded to the TP size.
        if parallel.tp_size not in (1, 8):
            raise NotImplementedError(
                f"the Muse megakernel runs at TP 1 or 8, got {parallel.tp_size}"
            )
        if parallel.attn_dcp_size != 1:
            raise NotImplementedError("the Muse megakernel reads unstrided page tables")
        geo = _ModelGeometry(model)
        decoder = model.model
        L, H = geo.n_layers, geo.hidden
        Ep = -(-geo.n_experts // K.LANE) * K.LANE
        device = decoder.norm.weight.device
        qkvg, wo, wpool, wunpool, wrouter, smalls = [], [], [], [], [], []
        experts, kinds = [], []
        # e4m3 NVFP4 block scales come with per-expert globals (SGLANG_TPU_NVFP4_E4M3_SCALES)
        scale_fp8 = any(
            layer.mlp.experts.w13_weight_scale.dtype == torch.float8_e4m3fn
            for layer in decoder.layers
        )
        kperm = tpu_envs.SGLANG_TPU_MUSE_MK_KPERM.get()
        if kperm and scale_fp8:
            raise NotImplementedError(
                "SGLANG_TPU_MUSE_MK_KPERM requires bf16/f32 NVFP4 block scales "
                "(unset SGLANG_TPU_NVFP4_E4M3_SCALES)"
            )
        if kperm:
            from sglang_tpu.fp8_moe import copy_or_rebind_param

            os.environ.setdefault("MUSE_PF_MOE_KPERM", "1")
            os.environ.setdefault("SGLANG_TPU_FP4_DEQUANT_VIA_FP8", "0")
            for layer in decoder.layers:
                moe = layer.mlp
                if _expert_kind(moe.experts) != "fp4":
                    continue
                pw = _perm_256_cpu(moe.pre_expert_proj.weight.detach().to("cpu"), 0)
                copy_or_rebind_param(
                    moe.pre_expert_proj, "weight", pw.to(device=device)
                )
                nw = _perm_256_cpu(moe.pre_expert_norm.weight.detach().to("cpu"), 0)
                copy_or_rebind_param(
                    moe.pre_expert_norm, "weight", nw.to(device=device)
                )
            torch.accelerator.synchronize()
        for layer in decoder.layers:
            attn, moe = layer.self_attn, layer.mlp
            if layer.post_attention_layernorm.weight is not None:
                raise NotImplementedError("affine post-attention norm")
            qkvg.append(
                _k_major(
                    torch.cat([attn.qkv_proj.weight, attn.gate_proj.weight], dim=0)
                )
            )
            wo.append(_k_major(attn.o_proj.weight))
            wpool.append(_k_major(moe.pre_expert_proj.weight))
            wunpool.append(_k_major(moe.post_expert_proj.weight))
            rw = moe.gate.weight.detach()
            rw = rw if rw.shape == (H, geo.n_experts) else rw.t()
            rpad = torch.zeros(H, Ep, dtype=torch.bfloat16, device=device)
            rpad[:, : geo.n_experts] = rw.to(torch.bfloat16)
            wrouter.append(rpad)
            sm = torch.zeros(K.SMALL_ROWS, H, dtype=torch.float32, device=device)
            S = K.SMALL
            sm[S["pre_attn_w"]] = layer.input_layernorm.gain()
            sm[S["attn_alpha"]] = (
                layer.post_attention_residual_gate.alpha.float().reshape(-1)
            )
            sm[S["attn_beta"]] = (
                layer.post_attention_residual_gate.beta.float().reshape(-1)
            )
            sm[S["pre_ffn_w"]] = layer.pre_feedforward_layernorm.gain()
            sm[S["post_ffn_w"]] = layer.post_feedforward_layernorm.gain()
            sm[S["ffn_alpha"]] = (
                layer.post_feedforward_residual_gate.alpha.float().reshape(-1)
            )
            sm[S["ffn_beta"]] = (
                layer.post_feedforward_residual_gate.beta.float().reshape(-1)
            )
            sm[S["pre_expert_w"], : geo.expert_hidden] = moe.pre_expert_norm.gain()
            sm[S["expert_bias"], :Ep] = K.NEG
            sm[S["expert_bias"], : geo.n_experts] = moe.e_score_correction_bias.float()
            sm[S["post_expert_w"], : geo.expert_hidden] = moe.post_expert_norm.gain()
            if scale_fp8 and moe.experts.w13_weight_scale.dtype == torch.float8_e4m3fn:
                E = geo.n_experts
                gs13 = moe.experts._tpu_w13_global_scale
                sm[S["expert_gs"], :E] = gs13[:, 0]
                sm[S["expert_gs"], Ep : Ep + E] = gs13[:, 1]
                sm[S["expert_gs_down"], :E] = moe.experts._tpu_w2_global_scale
            # fp8 e4m3 dense copies with per-output-channel scales (the router stays bf16).
            for name, mats in (
                ("wqkvg", qkvg),
                ("wo", wo),
                ("wpool", wpool),
                ("wunpool", wunpool),
            ):
                mats[-1], scale = _fp8_channelwise(mats[-1])
                sm[S[name + "_s"], : scale.numel()] = scale
            smalls.append(sm)
            kind = _expert_kind(moe.experts)
            kinds.append(kind)
            experts.extend(_expert_refs(moe.experts, kind))
        expert_groups = expert_groups_of(
            tuple(kinds), tpu_envs.SGLANG_TPU_MUSE_MK_EXPERT_GROUP.get()
        )
        experts = _stack_expert_groups(
            decoder, experts, expert_groups, kinds=tuple(kinds), kperm=kperm
        )
        rope = decoder.rotary_emb
        rope_tab = torch.cat(
            [
                rope.cos.repeat_interleave(2, dim=1),
                rope.sin.repeat_interleave(2, dim=1),
            ],
            dim=1,
        ).float()
        # Concatenated, not filled in place: an in-place-built tensor bound into the
        # compiled graph is re-materialized every step (torch-tpu 0.2).
        final = torch.cat(
            [
                decoder.norm.gain()[None].float(),
                torch.zeros(7, H, dtype=torch.float32, device=device),
            ]
        )
        weights = {
            "rope": rope_tab,
            "final_norm": final,
            "smalls": _stack(smalls),
            "wqkvg": _stack(qkvg),
            "wo": _stack(wo),
            "wpool": _stack(wpool),
            "wrouter": _stack(wrouter),
            "wunpool": _stack(wunpool),
        }
        extra = sum(t.numel() * t.element_size() for t in weights.values())
        logger.info(
            "Muse megakernel prepared: %d layers, experts %s, %.2f GiB of K-major copies per device",
            L,
            "/".join(sorted(set(kinds))),
            extra / 2**30,
        )
        mk = cls(
            geo=geo,
            weights=weights,
            experts=experts,
            kinds=tuple(kinds),
            tp_size=parallel.tp_size,
            tp_rank=parallel.tp_rank,
            max_rows=tpu_envs.SGLANG_TPU_MUSE_MK_MAX_ROWS.get(),
            max_position=rope_tab.shape[0],
            scale_fp8=scale_fp8,
            expert_groups=expert_groups,
            kperm=kperm,
        )
        torch.accelerator.synchronize()
        _INSTANCE[:] = [mk]
        _register_op()
        if mk.prefill:
            _register_prefill_op()
        return mk

    def share_dense_weights(self, decoder) -> int:
        """Run the native path's dense linears and routers on the kernel's K-major copies and
        drop their own weights; returns the bytes dropped per device."""
        S = K.SMALL
        dropped = 0
        for i, layer in enumerate(decoder.layers):
            attn, moe = layer.self_attn, layer.mlp
            qkv_cols = attn.qkv_proj.weight.shape[0]
            gate_cols = attn.gate_proj.weight.shape[0]
            plan = (
                (attn.qkv_proj, self.w_wqkvg, 0, qkv_cols, "wqkvg_s"),
                (
                    attn.gate_proj,
                    self.w_wqkvg,
                    qkv_cols,
                    qkv_cols + gate_cols,
                    "wqkvg_s",
                ),
                (attn.o_proj, self.w_wo, 0, attn.o_proj.weight.shape[0], "wo_s"),
                (
                    moe.pre_expert_proj,
                    self.w_wpool,
                    0,
                    moe.pre_expert_proj.weight.shape[0],
                    "wpool_s",
                ),
                (
                    moe.post_expert_proj,
                    self.w_wunpool,
                    0,
                    moe.post_expert_proj.weight.shape[0],
                    "wunpool_s",
                ),
                (moe.gate, self.w_wrouter, 0, self.geo.n_experts, None),
            )
            for linear, stack, c0, c1, scale_name in plan:
                dropped += linear.weight.numel() * linear.weight.element_size()
                linear.quant_method = TpuSharedKMajorLinearMethod(
                    linear,
                    stack=stack,
                    index=i,
                    col_start=c0,
                    col_stop=c1,
                    scales=None if scale_name is None else self.w_smalls,
                    scale_row=0 if scale_name is None else S[scale_name],
                )
                linear.weight = None
        return dropped

    # ------------------------------------------------------------------ op building
    def config(
        self,
        *,
        rows: int,
        q: int,
        n_req: int,
        n_page_entries: int,
        page_size: int,
        capture: tuple,
        kv_sal: bool = False,
        kv_sal_fp8: bool = False,
    ) -> K.MKConfig:
        g = self.geo
        opt = tpu_envs.SGLANG_TPU_MUSE_MK_OPT.get()
        fold_env = tpu_envs.SGLANG_TPU_MUSE_MK_FOLD.get()
        tiled = rows > BASE_STEP_MAX_ROWS
        # few-row steps (decode) gather their rows in one round and have nothing to fold
        fold_base = fold_env and n_req * q > 2 and not tiled
        unshifted = (
            1.02 * g.scale_query_by * g.head_dim * g.attn_scale <= K.UNSHIFTED_MAX
        )
        chunk_pages = 2 * g.inter_local // page_size if kv_sal else self.chunk_pages
        env_swa_tok = tpu_envs.SGLANG_TPU_MUSE_MK_SAL_SWA_CHUNK_TOKENS.get()
        swa_chunk_pages = (
            (env_swa_tok // page_size if env_swa_tok > 0 else (2 if opt else 0))
            if kv_sal
            else 0
        )
        if swa_chunk_pages >= chunk_pages:
            swa_chunk_pages = 0
        attn_opt = 0
        if opt:
            attn_opt = (
                5
                if (
                    tiled
                    and rows > 64
                    and self.max_position >= _ATTN_HOIST_GLOBAL_MIN_POS
                )
                else 3
            )
        cfg = K.MKConfig(
            hidden=g.hidden,
            head_dim=g.head_dim,
            q_heads=g.q_heads,
            kv_heads=g.kv_heads,
            n_layers=g.n_layers,
            n_experts=g.n_experts,
            top_k=g.top_k,
            expert_hidden=g.expert_hidden,
            inter_local=g.inter_local,
            tp=self.tp_size,
            rms_eps=g.rms_eps,
            post_eps=g.post_eps,
            scale_query_by=g.scale_query_by,
            attn_scale=g.attn_scale,
            router_mult=g.router_mult,
            route_scale=g.route_scale,
            router_eps=g.router_eps,
            window=g.window,
            global_layers=g.global_layers,
            expert_kinds=self.kinds,
            page_size=page_size,
            pages_per_seq=n_page_entries // n_req,
            n_pages_entries=n_page_entries,
            max_position=self.max_position,
            rows=rows,
            rows_per_req=q,
            n_req=n_req,
            capture_layers=capture,
            fold_f=fold_base,
            fold_o=fold_base,
            compact_rows=4 if (opt or rows <= _COMPACT4_MAX_ROWS) else 8,
            topk_iter=rows > 8,
            # SAL chunks are one expert w13 slot row of tokens (the ring is a view of those slots)
            chunk_pages=chunk_pages,
            swa_chunk_pages=swa_chunk_pages,
            swa_window_table=_swa_window_table(kv_sal, n_page_entries),
            kv_packed=self.kv_packed and not kv_sal,
            kv_sal=kv_sal,
            kv_sal_fp8=kv_sal_fp8,
            # fp8 cached K (1) and V (2) straight into the MXU in mixed bf16 x fp8 dots
            sal_fp8_dot=tpu_envs.SGLANG_TPU_MK_SAL_FP8_DOT.get()
            if kv_sal and kv_sal_fp8
            else 0,
            scale_fp8=self.scale_fp8,
            expert_groups=self.expert_groups,
            # q and k are RMS-normalized per head: the score bound allows exp without a max
            attn_unshifted=unshifted,
            attn_exp2=unshifted,
            attn_pipe=unshifted and (self.kv_packed or kv_sal),
            attn_streams=_ATTN_STREAMS[
                ("sal_fp8" if kv_sal_fp8 else "sal") if kv_sal else "fused"
            ],
            attn_skip_idle_streams=opt
            or tpu_envs.SGLANG_TPU_MUSE_MK_ATTN_SKIP_IDLE.get(),
            attn_col_mask=bool(swa_chunk_pages)
            and (opt or tpu_envs.SGLANG_TPU_MUSE_MK_ATTN_COL_MASK.get()),
            fold_attn_scale=(opt or tpu_envs.SGLANG_TPU_MK_FOLD_ATTN_SCALE.get())
            and math.frexp(g.attn_scale)[0] == 0.5,
            kv_group=_FUSED_KV_GROUP,
            row_tile=_row_tile() if tiled else 0,
            ex_opt=opt,
            ex_kperm=1 if getattr(self, "kperm", False) else 0,
            ex_via_fp8=opt,
            bd_n_tile=1024 if opt else 0,
            vec_dequant_kperm=opt,
            attn_carry=2 if opt else 0,
            alias_tiled_temps=opt and tiled,
            combine_loop=opt and tiled,
            untiled_max_rows=48 if opt else 128,
            y_quant=1 if (opt and tiled) else 0,
            alias_oy=tiled,
            alias_p=tiled,
            xfull_packed=False,
            attn_opt=attn_opt,
            tiled_vmem_y=2 if (opt and tiled) else 0,
            tiled_o_rot=opt and tiled,
            tiled_fold=opt and tiled and fold_env,
            pair_ag=1 if (opt and tiled) else 0,
            tiled_pair_y=_dev_int(
                "SGLANG_TPU_MUSE_MK_DEV_TILED_PAIR_Y", _default_tiled_pair_y(rows)
            )
            if (opt and tiled)
            else 0,
            pair_o_rs=opt and tiled,
            alias_oag_f=opt and tiled,
            sal_kv_split=opt and kv_sal and kv_sal_fp8,
            attn_head_fp8=1 if (opt and kv_sal and kv_sal_fp8) else 0,
            topk_xpose=opt and rows >= 64,
            pair_y_stream=opt and tiled and rows > 64,
            ex_gs_smem=False,
            ex_w2_stage=opt
            and (tiled or (rows >= 32 and bool(getattr(self, "kperm", False)))),
            # fp8 x fp8 expert LHS loses accuracy at R >= 256 (8K cc=64 verify).
            ex_lhs_fp8=1 if (opt and (tiled or rows >= 32) and rows <= 128) else 0,
            expert_chunk=_dev_int(
                "SGLANG_TPU_MUSE_MK_DEV_EXPERT_CHUNK",
                64 if (opt and tiled and rows > 128) else 32,
            ),
        )
        n_units = len(K.expert_units(cfg))
        ex_split = (
            4
            if (opt and rows >= _EXPERT_SPLIT_MIN_ROWS and n_units <= _ARMS5_MAX_UNITS)
            else 0
        )
        use_c8 = tiled or not cfg.ex_w2_stage
        cfg = msgspec.structs.replace(
            cfg,
            expert_split=ex_split,
            compact_arms=_compact_arms(
                rows, n_units, opt=opt, expert_split=ex_split, use_c8=use_c8
            ),
            compact_rows=(
                8
                if (opt and ex_split > 0 and use_c8)
                else (
                    4
                    if (opt and n_units <= _ARMS5_MAX_UNITS)
                    or (not opt and rows <= _COMPACT4_MAX_ROWS)
                    else 8
                )
            ),
            ex_issue_pairs=n_units <= _ARMS5_MAX_UNITS
            and (rows <= _COMPACT4_MAX_ROWS or (opt and not tiled)),
            ex_bd_all=opt and not tiled and cfg.ex_kperm == 0,
        )
        if cfg.tiled_fold and tiled:
            do_fold = n_req * q > 2 and not (
                cfg.single_row_coll and n_req * q <= cfg.few_rows_max
            )
            cfg = msgspec.structs.replace(
                cfg,
                tiled_vmem_pads=True,
                fold_o=do_fold,
                fold_f=do_fold,
            )
        if tiled:
            # Row-tiled plans at 128-192 rows of 8 need 60.1-60.9 MiB with LLO temporaries;
            # scoped VMEM may go to the 62 MiB v7x limit, as the aliased base plan does.
            cfg = msgspec.structs.replace(cfg, vmem_limit_bytes=_OPT_ALIASED_VMEM_LIMIT)
            vys = (
                (cfg.tiled_vmem_y, 1) if cfg.tiled_vmem_y >= 2 else (cfg.tiled_vmem_y,)
            )
            plans = []
            for vy in vys:
                c_vy = (
                    msgspec.structs.replace(cfg, tiled_vmem_y=vy)
                    if vy != cfg.tiled_vmem_y
                    else cfg
                )
                b_vy = (
                    _TILED_VY2_VMEM_BUDGET
                    if vy >= 2
                    else (
                        (
                            _TILED_VPADS_SMALL_VMEM_BUDGET
                            if rows <= 128
                            else _TILED_VPADS_VMEM_BUDGET
                        )
                        if cfg.tiled_vmem_pads
                        else _SCRATCH_VMEM_BUDGET
                    )
                )
                plans.append((c_vy, b_vy, _TILED_DEPTH_RING_PREFERENCE, vy != vys[-1]))
            # The y_block_send VMEM-Y alternative fails the numerics test at 192 rows of 8
            # (layer 4, global attention + fp8 experts: rows off by 25.6% against the CPU
            # reference; the VMEM-Y and streamed-Y plans pass): off unless asked for.
            if (
                cfg.tiled_vmem_y >= 1
                and cfg.alias_oag_f
                and _dev_int("SGLANG_TPU_MUSE_MK_DEV_YSEND_PLAN", 0)
            ):
                plans.append(
                    (
                        msgspec.structs.replace(
                            cfg,
                            tiled_vmem_y=1,
                            tiled_vmem_pads=False,
                            tiled_o_rot=False,
                            fold_o=False,
                            fold_f=False,
                            pair_ag=0,
                            pair_o_rs=False,
                            tiled_pair_y=0,
                            y_block_send=True,
                            attn_streams=min(cfg.attn_streams, 4),
                        ),
                        _TILED_VY2_VMEM_BUDGET,
                        _TILED_DEPTH_RING_PREFERENCE,
                        True,
                    )
                )
            plans.append(
                (
                    msgspec.structs.replace(
                        cfg,
                        tiled_vmem_y=0,
                        tiled_vmem_pads=False,
                        tiled_o_rot=False,
                        fold_o=False,
                        fold_f=False,
                        pair_ag=0,
                        pair_o_rs=False,
                        tiled_pair_y=0,
                        y_block_send=False,
                        attn_streams=min(cfg.attn_streams, 4),
                    ),
                    _OPT_BASE_VMEM_BUDGET,
                    _TILED_DEPTH_RING_PREFERENCE,
                    False,
                )
            )
        elif opt:
            # Prefer unaliased base scratch when the full 10-slot folded ring already
            # fits in 48 MiB (<=16 rows, or 24 rows at q <= 4); enable VMEM pad/ysend
            # aliasing + packed xfull when needed to keep the 10-slot ring (24 rows at
            # q >= 5 and 32 rows).
            plans = [
                (cfg, _SCRATCH_VMEM_BUDGET, ((3, 11),), False),
                (
                    msgspec.structs.replace(
                        cfg,
                        alias_oy=True,
                        alias_p=True,
                        xfull_packed=True,
                        vmem_limit_bytes=_OPT_ALIASED_VMEM_LIMIT,
                    ),
                    _OPT_BASE_VMEM_BUDGET,
                    _DEPTH_RING_PREFERENCE,
                    False,
                ),
            ]
        else:
            plans = [(cfg, _SCRATCH_VMEM_BUDGET, _DEPTH_RING_PREFERENCE, False)]
        for base_cfg, budget, prefs, step_down_vy in plans:
            for depth, cap in prefs:
                if step_down_vy and (depth < 3 or cap < 5):
                    break
                cand = msgspec.structs.replace(base_cfg, expert_depth=depth)
                slots = K.fit_ring_slots(cand, cap=cap)
                if (
                    tiled
                    and base_cfg.tiled_vmem_y >= 1
                    and slots
                    <= _dev_int(
                        "SGLANG_TPU_MUSE_MK_DEV_TILED_VY_MIN_SLOTS", _TILED_VY_MIN_SLOTS
                    )
                ):
                    continue
                cand = msgspec.structs.replace(
                    cand, ring_slots=slots, stream_depth=slots - 1
                )
                if K.scratch_vmem_bytes(cand) <= budget:
                    if (cand.fold_f or cand.fold_o) and not K.fold_fits(cand):
                        cand = msgspec.structs.replace(cand, fold_f=False, fold_o=False)
                    return cand
        raise ValueError(
            f"no Muse megakernel ring fits the VMEM budget at {rows} rows of {q}"
        )

    def check_kv_pools(self, pool) -> None:
        """Raise unless the KV pools have a layout the kernel reads: bf16 or fp8 SEQ_ALONG_LANE
        pools, or bf16 fused pools with heads packed exactly when the kernel was built for it."""
        pools = (
            (pool.full_kv_pool, pool.swa_kv_pool)
            if isinstance(pool, SWAKVPool)
            else (pool,)
        )
        for p in pools:
            if type(p) is TpuSeqAlongLaneKVPool:
                if p.store_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
                    raise NotImplementedError(
                        f"the Muse megakernel reads bf16 or fp8 SEQ_ALONG_LANE pools, got "
                        f"{p.store_dtype}"
                    )
                continue
            if (
                type(p) is not TpuFusedMHATokenToKVPool
                or p.store_dtype != torch.bfloat16
            ):
                raise NotImplementedError(
                    f"the Muse megakernel reads bf16 fused KV pools, got "
                    f"{type(p).__name__} ({p.store_dtype})"
                )
            if (p.kv_pack_heads > 1) != self.kv_packed:
                raise NotImplementedError(
                    f"KV pool packs {p.kv_pack_heads} heads per row, but the Muse "
                    f"megakernel was built with kv_packed={self.kv_packed}"
                )

    def callable_for(self, cfg: K.MKConfig):
        if cfg not in self._ops:
            self._ops[cfg] = _build_callable(cfg, self.tp_size)
        return self._ops[cfg]

    def prefill_config(self, *, rows: int, capture: tuple):
        from sglang_tpu.muse_megakernel import prefill as PF

        g = self.geo
        return PF.PFConfig(
            hidden=g.hidden,
            head_dim=g.head_dim,
            q_heads=g.q_heads,
            kv_heads=g.kv_heads,
            n_layers=g.n_layers,
            n_experts=g.n_experts,
            top_k=g.top_k,
            expert_hidden=g.expert_hidden,
            tp=self.tp_size,
            rms_eps=g.rms_eps,
            post_eps=g.post_eps,
            scale_query_by=g.scale_query_by,
            attn_scale=g.attn_scale,
            router_mult=g.router_mult,
            route_scale=g.route_scale,
            router_eps=g.router_eps,
            window=g.window,
            global_layers=g.global_layers,
            capture_layers=capture,
            rows=rows,
            # Tuning only: the fused kernels' row tiles (larger ones overflow VMEM).
            attn_out_tile=int(os.environ.get("MUSE_PF_ATTN_OUT_TILE", "64")),
            post_moe_tile=int(os.environ.get("MUSE_PF_POST_MOE_TILE", "128")),
            fp8_gather=tpu_envs.SGLANG_TPU_MUSE_MK_PREFILL_FP8_GATHER.get(),
            expert_groups=self.expert_groups,
        )

    def prefill_callable_for(self, cfg):
        if cfg not in self._ops:
            self._ops[cfg] = _build_prefill_callable(cfg, self.tp_size)
        return self._ops[cfg]

    # ------------------------------------------------------------------ forward
    def handles(self, forward_batch, num_tokens: int) -> bool:
        mode = forward_batch.forward_mode
        if mode.is_decode() or mode.is_target_verify():
            return step_rows(num_tokens, self.tp_size) <= self.max_rows
        if not (self.prefill and mode.is_extend()):
            return False
        # token buckets (tpu_compile_tokens) must tile the prefill kernels
        if num_tokens % _PREFILL_ROW_ALIGN:
            raise ValueError(
                f"{num_tokens}-token extend step: the Muse prefill program needs "
                f"multiples of {_PREFILL_ROW_ALIGN} (tpu_compile_tokens)"
            )
        return True

    def forward(
        self,
        *,
        embeds: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        capture: tuple,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``embeds [T, H]`` bf16 (all-reduced) -> final-normed hidden ``[T, H]`` (+ aux).

        Runs inside the compiled forward: only tensors and Python ints reach the op.
        """
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
        md = backend.forward_metadata
        mode = forward_batch.forward_mode
        if not (mode.is_decode() or mode.is_target_verify()):
            return self._forward_prefill(embeds, positions, backend, capture)
        page_indices = md.page_indices
        swa_page_indices = (
            md.swa_page_indices if md.swa_page_indices is not None else page_indices
        )
        n_req = forward_batch.seq_lens.shape[0]
        T = embeds.shape[0]
        if T % n_req:
            raise ValueError(f"{T} rows do not split evenly over {n_req} requests")
        rows = step_rows(T, self.tp_size)
        pad = rows - T
        emb = embeds.to(torch.bfloat16)
        pos = positions.to(torch.int32)
        if pad:
            emb = torch.cat([emb, emb.new_zeros(pad, emb.shape[1])], dim=0)
            pos = torch.cat([pos, pos.new_zeros(pad)], dim=0)
        n_live = md.active_num_tokens.reshape(1).to(torch.int32)
        meta = torch.cat([n_live, self.rank_t], dim=0)
        pool = backend.token_to_kv_pool
        self.check_kv_pools(pool)
        kv = [pool.get_fused_kv_buffer(i) for i in range(self.geo.n_layers)]
        weights = [getattr(self, f"w_{n}") for n in K.WEIGHT_NAMES]
        experts = [getattr(self, f"expert_{i}") for i in range(self.n_expert_refs)]
        hidden, aux = torch.ops.sglang_tpu_pallas.muse_megakernel(
            meta,
            pos,
            page_indices,
            swa_page_indices,
            emb,
            weights,
            experts,
            kv,
            T // n_req,
            n_req,
            list(capture),
        )
        return hidden[:T], (aux[:T] if capture else None)

    def _forward_prefill(
        self, embeds: torch.Tensor, positions: torch.Tensor, backend, capture: tuple
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        md = backend.forward_metadata
        pool = backend.token_to_kv_pool
        pools = (
            (pool.full_kv_pool, pool.swa_kv_pool)
            if isinstance(pool, SWAKVPool)
            else (pool,)
        )
        for p in pools:
            if (
                type(p) is not TpuSeqAlongLaneKVPool
                or p.store_dtype != torch.float8_e4m3fn
            ):
                raise NotImplementedError(
                    f"the Muse prefill program reads fp8 SEQ_ALONG_LANE pools, got "
                    f"{type(p).__name__} ({p.store_dtype})"
                )
        page_indices = md.page_indices
        swa_page_indices = (
            md.swa_page_indices if md.swa_page_indices is not None else page_indices
        )
        meta = torch.cat([md.active_num_tokens.reshape(1).to(torch.int32), self.rank_t])
        kv = [pool.get_fused_kv_buffer(i) for i in range(self.geo.n_layers)]
        weights = [getattr(self, f"w_{n}") for n in K.WEIGHT_NAMES]
        experts = [getattr(self, f"expert_{i}") for i in range(self.n_expert_refs)]
        hidden, aux = torch.ops.sglang_tpu_pallas.muse_prefill(
            meta,
            positions.to(torch.int32),
            md.kv_lens.to(torch.int32),
            md.cu_q_lens.to(torch.int32),
            md.distribution.to(torch.int32),
            page_indices,
            swa_page_indices,
            embeds.to(torch.bfloat16),
            weights,
            experts,
            kv,
            list(capture),
        )
        return hidden, (aux if capture else None)


# The op implementation runs outside the traced region and needs the prepared weights'
# geometry; one model per process.
_INSTANCE: list = []
_REGISTERED: list = []
_REGISTERED_PREFILL: list = []


def _register_op() -> None:
    """The one generic op: shapes and the Python ints select (and cache) the compiled kernel."""
    if _REGISTERED:
        return

    @torch.library.custom_op(
        f"{_OP_NS}::muse_megakernel", mutates_args=("kv",), device_types=["tpu"]
    )
    def op(
        meta: torch.Tensor,
        pos: torch.Tensor,
        pf: torch.Tensor,
        psw: torch.Tensor,
        emb: torch.Tensor,
        weights: list[torch.Tensor],
        experts: list[torch.Tensor],
        kv: list[torch.Tensor],
        rows_per_req: int,
        n_req: int,
        capture: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        mk = _INSTANCE[0]
        kv_sal = is_sal_pool_shape(tuple(kv[0].shape), mk.geo)
        cfg = mk.config(
            rows=emb.shape[0],
            q=rows_per_req,
            n_req=n_req,
            n_page_entries=pf.shape[0],
            page_size=kv[0].shape[-1] if kv_sal else kv[0].shape[1],
            capture=tuple(capture),
            kv_sal=kv_sal,
            kv_sal_fp8=kv_sal and kv[0].dtype == torch.float8_e4m3fn,
        )
        callable_ = mk.callable_for(cfg)
        with tpu_info_for_cpu_export():
            outs = callable_(meta, pos, pf, psw, emb, *weights, *experts, *kv)
        return outs[0], outs[1]

    @op.register_fake
    def _(meta, pos, pf, psw, emb, weights, experts, kv, rows_per_req, n_req, capture):
        aux_w = len(capture) * emb.shape[1] if capture else K.LANE
        return emb.new_empty((emb.shape[0], emb.shape[1])), emb.new_empty(
            (emb.shape[0], aux_w)
        )

    _REGISTERED.append(True)


def _register_prefill_op() -> None:
    """The extend-step op: the row count and capture select (and cache) the program."""
    if _REGISTERED_PREFILL:
        return

    @torch.library.custom_op(
        f"{_OP_NS}::muse_prefill", mutates_args=("kv",), device_types=["tpu"]
    )
    def op(
        meta: torch.Tensor,
        pos: torch.Tensor,
        kv_lens: torch.Tensor,
        cu_q: torch.Tensor,
        dist: torch.Tensor,
        pf: torch.Tensor,
        psw: torch.Tensor,
        emb: torch.Tensor,
        weights: list[torch.Tensor],
        experts: list[torch.Tensor],
        kv: list[torch.Tensor],
        capture: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        mk = _INSTANCE[0]
        cfg = mk.prefill_config(rows=emb.shape[0], capture=tuple(capture))
        callable_ = mk.prefill_callable_for(cfg)
        with tpu_info_for_cpu_export():
            outs = callable_(
                meta, pos, kv_lens, cu_q, dist, pf, psw, emb, *weights, *experts, *kv
            )
        return outs[0], outs[1]

    @op.register_fake
    def _(meta, pos, kv_lens, cu_q, dist, pf, psw, emb, weights, experts, kv, capture):
        aux_w = len(capture) * emb.shape[1] if capture else K.LANE
        return emb.new_empty((emb.shape[0], emb.shape[1])), emb.new_empty(
            (emb.shape[0], aux_w)
        )

    _REGISTERED_PREFILL.append(True)


# prefill.post_moe_call tiles 128 rows as 16 rows of each of the 8 ranks' blocks
_PREFILL_ROW_ALIGN = 128
# Mosaic's temporaries come on top of the declared scratch and grow with rows per request
# (verify attention): 24 rows of 6 compile at 46.8 MiB declared, 32 rows of 8 overflowed the
# 60 MiB limit at 48.7 MiB (~11.5 MiB of temporaries).
_SCRATCH_VMEM_BUDGET = 48 * 2**20
_OPT_BASE_VMEM_BUDGET = 50 * 2**20
# 32 rows of 8 at that budget needed 60.25 MiB with temporaries; scoped VMEM may go to the
# 62 MiB v7x limit, so the aliased plan takes it rather than a smaller ring.
_OPT_ALIASED_VMEM_LIMIT = 62 * 2**20
# On libtpu 0.0.50 (2026-10-07), row-tiled LLO temporaries add ~10.5-11.0 MiB on top of
# declared scratch; capping row-tiled budgets at 50 MiB keeps total scoped VMEM <= 60.5 MiB
# under the 62 MiB v7x limit across rows=48..128 without LLO retry thrashing.
_TILED_VPADS_VMEM_BUDGET = 50 * 2**20
# 128 rows of 8 (P/D decode cc16): the depth-3, 5-slot VMEM-Y plan declares 48.28 MiB, but its
# LLO temporaries take scoped VMEM to 62.08 MiB, over the 62 MiB limit. Under 48 MiB the
# planner takes the depth-3, 3-slot plan (43.53 MiB): 42.3/67.1 ms at 16k/512k (anvil).
_TILED_VPADS_SMALL_VMEM_BUDGET = 48 * 2**20
_TILED_VY2_VMEM_BUDGET = 50 * 2**20
_ATTN_HOIST_GLOBAL_MIN_POS = 262144
# Past this, pages_swa stays in HBM and each step reads only the requests' windows of it.
# Whole tables that still fit SMEM are slow: at 1M context q8 cc20/cc24 (640/768 KiB) ran
# 10.8-16.9x slower than with the window table (bit-identical outputs); cc4-cc12 (<=384
# KiB) cost 0-1.1% with it (anvil A/B, device time).
_SMEM_PAGE_TABLES_BUDGET = 512 * 2**10
# A VMEM-Y row-tiled plan squeezed to a 2-slot ring has no prefetch room, and its LLO spill
# slots overflow VMEM (cc28 x q8: 49.5 MiB scratch + 16.2 MiB spills > 63.9 MiB); the
# streamed-Y plan fits there with a 6-slot ring.
_TILED_VY_MIN_SLOTS = 2


def _swa_window_table(kv_sal: bool, n_page_entries: int) -> bool:
    mode = tpu_envs.SGLANG_TPU_MUSE_MK_SWA_WINDOW_TABLE.get()
    if mode not in ("auto", "0", "1"):
        raise ValueError(f"SGLANG_TPU_MUSE_MK_SWA_WINDOW_TABLE={mode!r}: auto, 0 or 1")
    if mode == "auto":
        return kv_sal and 2 * 4 * n_page_entries > _SMEM_PAGE_TABLES_BUDGET
    return mode == "1"


# (expert depth, ring cap): the ring is the largest divisor of the 30 dense tiles per layer up
# to the cap (10, 6, 5, then 3). The folds need a stream depth of at least 3 (K.fold_fits), so
# 5 slots or more.
_DEPTH_RING_PREFERENCE = ((3, 11), (3, 6), (3, 5), (3, 4), (1, 4))
# Row-tiled steps: deepest expert prefetch, then the largest dense ring that fits.
_TILED_DEPTH_RING_PREFERENCE = tuple((d, s) for d in (3, 1) for s in range(11, 1, -1))
# Measured device step (real geometry, random routing): the 8-row block-diag variant is
# slower than the 4-row one at 8-24 rows and faster once expert_split >= 32 rows.
_COMPACT4_MAX_ROWS = 24
_EXPERT_SPLIT_MIN_ROWS = 32


# Expert operand units (one leaf each in the expert-DMA layer walks) up to which the expert
# loop still fits the instruction memory with a fifth expert variant or paired DMA issue:
# stacked groups of 15 (6 units) fit, per-layer operands (62 units) swap overlays.
_ARMS5_MAX_UNITS = 8


def _compact_arms(
    rows: int,
    n_units: int,
    *,
    opt: bool = True,
    expert_split: int = 0,
    use_c8: bool = True,
) -> tuple:
    """Compact expert row buckets, one code variant each (the full path is one more); a sixth
    variant overflows the instruction memory into overlay swaps."""
    if n_units > _ARMS5_MAX_UNITS:
        return ()
    if opt:
        if rows <= 8:
            return (1, 2, 3, 4)
        return (1, 2, 4, 8) if (expert_split > 0 and use_c8) else (1, 2, 4)
    if rows > _COMPACT4_MAX_ROWS:
        return ()
    return (1, 2, 3, 4) if rows <= 8 else (1, 2, 3, 5)


def _default_tiled_pair_y(rows: int) -> int:
    if rows < 64:
        return 0
    return 1 if rows < 80 else 2


def is_sal_pool_shape(shape: tuple, geo) -> bool:
    """SEQ_ALONG_LANE blocks are [pages, 2 * kv_heads, head_dim // pk, pk, page_size]; fused
    pools are [pages, page_size, heads, 2, lanes]."""
    return (
        len(shape) == 5
        and shape[1] == 2 * geo.kv_heads
        and shape[2] * shape[3] == geo.head_dim
        and shape[4] % K.LANE == 0
    )


def _round_up(n: int, m: int) -> int:
    return -(-n // m) * m


def _export_program(body, *, tp: int, n_args: int, kv0: int, n_out: int, tag: str):
    """``body`` (per-device, KV pools from argument ``kv0`` on, returned after ``n_out - L``
    outputs) as one jitted ``shard_map`` program over the TP mesh, exported for torch with
    the pools donated and aliased."""
    from torch_tpu._internal.pallas import pallas as pallas_impl

    from sglang_tpu.oneshot_collectives import _get_mesh
    from sglang_tpu.pallas_jax_op import _CpuExportJaxCallable

    mesh = _get_mesh(tp)
    n_kv = n_args - kv0
    sharded = jax.shard_map(
        body,
        mesh=mesh,
        in_specs=(P("model"),) * n_args,
        out_specs=(P("model"),) * n_out,
        check_vma=False,
    )

    def fn(*xs: jax.Array) -> tuple:
        return tuple(sharded(*xs))

    donate = tuple(range(kv0, n_args))
    out_shardings = tuple(NamedSharding(mesh, P("model")) for _ in range(n_out))
    jit_fn = jax.jit(
        fn, donate_argnums=donate, keep_unused=True, out_shardings=out_shardings
    )
    trace_key = pallas_impl._get_kernel_invocation_key(
        f"{tag}_{id(fn)}", [], {"donate": donate}
    )
    callable_ = _CpuExportJaxCallable(
        name=f"jax_{tag}",
        jit_fn=jit_fn,
        trace_key=trace_key,
        mesh=mesh,
        input_partition_specs=(P("model"),) * n_args,
        static_argnums=(),
        donate_argnums=donate,
        input_output_aliases={kv0 + i: n_out - n_kv + i for i in range(n_kv)},
    )
    callable_.sglang_output_shardings = out_shardings
    return callable_


def _build_callable(cfg: K.MKConfig, tp: int):
    """The megakernel + KV scatter as one jitted ``shard_map`` program over the TP mesh."""
    L = cfg.n_layers
    nw = len(K.WEIGHT_NAMES)
    ne = K.n_expert_refs(cfg)
    n_args = 5 + nw + ne + L
    kv0 = 5 + nw + ne
    mk = K.megakernel_call(cfg)
    kvw = K.kv_write_call(cfg)

    def body(*xs):
        hid, aux, newkv = mk(*xs)
        pools = kvw(*xs[:4], newkv, *xs[kv0:])
        return (hid, aux, *pools)

    tag = f"r{cfg.rows}_q{cfg.rows_per_req}_n{cfg.n_req}_c{len(cfg.capture_layers)}_p{cfg.pages_per_seq}"
    callable_ = _export_program(
        body, tp=tp, n_args=n_args, kv0=kv0, n_out=2 + L, tag=f"muse_megakernel_{tag}"
    )
    logger.info(
        "Muse megakernel built: rows=%d q=%d requests=%d capture=%s pages/seq=%d "
        "row_tile=%d ring=%d folds=%s",
        cfg.rows,
        cfg.rows_per_req,
        cfg.n_req,
        cfg.capture_layers,
        cfg.pages_per_seq,
        cfg.row_tile,
        cfg.ring_slots,
        cfg.fold_f or cfg.fold_o,
    )
    return callable_


def _build_prefill_callable(cfg, tp: int):
    """The prefill program (prefill.build_forward) over the TP mesh; it updates the pools."""
    from sglang_tpu.muse_megakernel import prefill as PF

    L = cfg.n_layers
    n_args = 8 + len(K.WEIGHT_NAMES) + K.n_expert_refs(cfg) + L
    callable_ = _export_program(
        PF.build_forward(cfg),
        tp=tp,
        n_args=n_args,
        kv0=n_args - L,
        n_out=2 + L,
        tag=f"muse_prefill_r{cfg.rows}_c{len(cfg.capture_layers)}",
    )
    logger.info(
        "Muse prefill program built: rows=%d capture=%s fp8_gather=%s",
        cfg.rows,
        cfg.capture_layers,
        cfg.fp8_gather,
    )
    return callable_
