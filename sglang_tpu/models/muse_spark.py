# SPDX-License-Identifier: Apache-2.0
"""Muse Spark text decoder on TPU.

Each layer is ``pre-norm -> gated attention -> post-norm -> gated residual``, then
the same around a compressed MoE: ``pre_expert_proj`` maps the hidden state to
``moe_hidden_size``, experts run in that space, each expert output is
RMS-normalized before the router-weighted sum, and ``post_expert_proj`` maps the
result back. The residual stream is kept in fp32.

NVFP4 experts run through the Pallas fused-RS kernel, which applies the
post-expert norm, the weighted combine and the cross-rank reduction. The BF16
experts of the first and last layer are quantized to FP8 on host at load and take
the FP8 fused-RS kernel. The opt-in
``SGLANG_TPU_ENABLE_MUSE_BF16_FIRST_LAST_LAYER_EXPERTS`` keeps them BF16 on
fused-RS when supported, otherwise the BF16 grouped GEMM and
``_combine_slots`` do those steps. This module is the text decoder; ``muse_spark_omni`` adds
the vision tower on top of it and is the registered architecture. Audio tower
weights are skipped.
"""

from __future__ import annotations

import functools
import json
import logging
import os
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.muse_spark import MuseSparkConfig, MuseSparkTextConfig
from sglang.srt.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.utils import get_group_rank_size
from sglang.srt.layers.aux_hidden_states import AUX_HIDDEN_STATES_PROXY_KEY
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod, fp8_dtype
from sglang.srt.layers.quantization.modelopt_quant import ModelOptQuantConfig
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.utils import PPMissingLayer, copy_or_rebind_param
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.layers.utils.layer_list import make_pp_layers
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_disagg, get_parallel, process_model_config
from sglang_tpu.decode_qkv_epilogue_tpu import (
    can_use_muse_decode_qkv_epilogue,
    can_use_muse_decode_rope_step,
    get_decode_rope_step_op,
    muse_decode_fused_qkv_attn,
    warm_decode_qkv_epilogue_ops,
)
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.expert_loading import _wait_for
from sglang_tpu.fp8_moe import _free_replaced_storage
from sglang_tpu.fp8_moe_fused_rs import apply_fp8_fused_rs, prepare_fp8_fused_rs

logger = logging.getLogger(__name__)

# Everything else in the checkpoint belongs to the vision and audio towers.
_TEXT_PREFIX = "model.language_model."
# Static activation scales; the NVFP4 experts run weight-only (W4A16) on TPU.
_UNUSED_SUFFIXES = (".gate_up_proj_input_scale", ".down_proj_input_scale")
# Checkpoint suffix -> parameter suffix, first match wins.
_RENAMES = (
    (".experts.gate_up_proj_weight_scale_2", ".experts.w13_weight_scale_2"),
    (".experts.gate_up_proj_weight_scale", ".experts.w13_weight_scale"),
    (".experts.gate_up_proj", ".experts.w13_weight"),
    (".experts.down_proj_weight_scale_2", ".experts.w2_weight_scale_2"),
    (".experts.down_proj_weight_scale", ".experts.w2_weight_scale"),
    (".experts.down_proj", ".experts.w2_weight"),
    (".experts.post_expert_norm.weight", ".post_expert_norm.weight"),
    (".gate.e_score_correction_bias", ".e_score_correction_bias"),
)
_QKV_SHARDS = ((".q_proj.", "q"), (".k_proj.", "k"), (".v_proj.", "v"))
# The fused-RS kernels apply ``rms_norm((w + 1) * x, 1e-8)`` per expert slot.
_FUSED_RS_POST_EXPERT_NORM_EPS = 1e-8
# Sigmoid router scores are renormalized over the top-k with this epsilon.
_ROUTE_EPS = 1e-15


def _env_get(name: str, default):
    """Read an optional env descriptor from ``tpu_envs`` (tolerant of minimal test mocks)."""
    field = getattr(tpu_envs, name, None)
    return default if field is None else field.get()


def _decode_max_tokens() -> int:
    """Return max decode/verify token rows eligible for fused decode paths (256 or 384)."""
    if bool(_env_get("SGLANG_TPU_MUSE_DECODE_B384", False)) or bool(
        _env_get("SGLANG_TPU_MUSE_VERIFY_BRIDGE_LIFT", False)
    ):
        return 384
    return 256


def _decode_bridge_max_tokens() -> int:
    """Return max token rows eligible for Wave-7/8 decode collective bridges."""
    explicit = int(_env_get("SGLANG_TPU_MUSE_DECODE_BRIDGE_MAX_TOKENS", 0) or 0)
    if explicit > 0:
        return min(explicit, 384)
    if bool(_env_get("SGLANG_TPU_MUSE_VERIFY_BRIDGE_LIFT", False)):
        return 384
    return min(_decode_max_tokens(), 320)


def _can_use_bridge_qkv_epi(
    n_tok: int,
    attn_mod: object,
    cos_sin_step: Optional[torch.Tensor],
    forward_batch: object,
    *,
    max_tok: int = 384,
) -> bool:
    """True when Bridge 2b / L0 can fuse QKV RMSNorm + RoPE/NoPE + FP8 KV quant in-VMEM."""
    if forward_batch is None or attn_mod is None:
        return False
    verify_ok = forward_batch.forward_mode.is_target_verify() and bool(
        _env_get("SGLANG_TPU_MUSE_KERNEL_VERIFY_RPA_FAST", False)
    )
    allow_small_qkv_epi = os.environ.get(
        "SGLANG_TPU_MUSE_DECODE_BRIDGE_QKV_EPI_SMALL", "1"
    ) not in ("0", "false", "False")
    valid_qkv_tok = (
        (n_tok == 8 or (16 <= n_tok <= max_tok and n_tok % 16 == 0))
        if allow_small_qkv_epi
        else (32 <= n_tok <= max_tok and n_tok % 32 == 0)
    )
    qkv_dim = attn_mod.q_size + 2 * attn_mod.kv_size
    valid_qkv_geom = (qkv_dim == 1280 and attn_mod.q_size == 1024) or (
        qkv_dim == 640
        and attn_mod.q_size == 512
        and bool(tpu_envs.SGLANG_TPU_MUSE_DECODE_TP16_QKV_EPI.get())
    )
    if (
        not getattr(attn_mod, "decode_qkv_epilogue", False)
        or not (forward_batch.forward_mode.is_decode() or verify_ok)
        or not valid_qkv_geom
        or not valid_qkv_tok
    ):
        return False
    if getattr(attn_mod, "rotary_emb", None) is not None and cos_sin_step is None:
        return False
    from sglang_tpu.decode_qkv_epilogue_tpu import _resolve_decode_attn_backend

    backend = _resolve_decode_attn_backend(forward_batch)
    fn = getattr(backend, "can_forward_decode_prepared", None)
    return bool(fn is not None and fn(attn_mod.attn, forward_batch))


def _as_1d_fp8_scale(scale: torch.Tensor) -> torch.Tensor:
    """Materialize FP8 weight scale as a fresh 1D ``(N,) float32`` storage-owned tensor.

    TorchTPU allocates 2D ``(1, N)`` tensors in HBM with ``{1,0:T(8,128)}`` (8 padded
    sublanes), whereas Pallas ``pltpu.VMEM`` expects ``{1,0:T(1,128)}``, forcing XLA to
    emit a synchronous ``copy`` op per scale (~290 ops / 0.41 ms per decode step).
    A storage-owned 1D ``(N,) float32`` tensor uses ``{0:T(1024)}`` and lowers
    ``s.reshape(1, N)`` inside ``shard_map`` with 0 synchronous ``copy`` ops.
    """
    if os.environ.get("SGLANG_TPU_MUSE_DECODE_1D_FP8_SCALES", "1").strip() in (
        "0",
        "false",
        "False",
    ):
        return scale
    from sglang_tpu.fp8_moe import _own_storage

    return _own_storage(scale.detach().reshape(-1).to(torch.float32).contiguous())


def _default_tpu7x_chipmate_xor() -> int:
    """Return the verified TPU7x intra-chip chipmate XOR bit (defaults to 1)."""
    try:
        from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_ep_tp2 import (
            TPU7X_CHIPMATE_XOR,
        )

        return int(TPU7X_CHIPMATE_XOR)
    except ImportError:
        return 1


def _resolve_ep_tp2_coords(
    tp_rank: int, ep_size: int, *, chipmate_xor: Optional[int] = None
) -> Tuple[int, int]:
    """Map global ``tp_rank`` in ``[0, 2 * ep_size)`` to ``(chip_id, tp_half)``."""
    xor = _default_tpu7x_chipmate_xor() if chipmate_xor is None else int(chipmate_xor)
    if xor <= 0 or (xor & (xor - 1)) != 0:
        raise ValueError(f"chipmate_xor must be a positive power of two, got {xor}")
    bit = xor.bit_length() - 1
    tp_half = (tp_rank >> bit) & 1
    chip_id = ((tp_rank >> (bit + 1)) << bit) | (tp_rank & (xor - 1))
    if not (0 <= chip_id < ep_size):
        raise ValueError(
            f"tp_rank={tp_rank} with chipmate_xor={xor} maps outside ep_size={ep_size}"
        )
    return chip_id, tp_half


def _is_decode_ep_tp2_layout(parallel) -> bool:
    """Whether opt-in hybrid symmetric EP(P/2)xTP2 decode is enabled for this parallel layout."""
    return (
        bool(_env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_EP_TP2", False))
        and parallel.moe_tp_size == 2
        and parallel.moe_ep_size in (4, 8)
        and parallel.moe_ep_size * parallel.moe_tp_size == parallel.tp_size
        and getattr(parallel, "moe_dp_size", 1) == 1
    )


def _convert_gmm_v2_nvfp4_to_fused_rs_layout(layer: nn.Module) -> None:
    """Convert stock ``gmm_v2`` NVFP4 weights (``uint8`` + 4D scales) in place to
    fused-RS ``float4_e2m1fn_x2`` + 3D ``bfloat16`` scales for hybrid EPxTP2 decode."""
    from sglang_tpu.fp4_moe import (
        _materialize_packed_fp4,
        _transpose_packed_fp4,
    )
    from sglang_tpu.fp8_moe import _own_storage
    from sglang_tpu.fused_rs_sequence_parallel import (
        has_tpu_fused_rs_post_expert_norm,
        prepare_tpu_fused_rs_post_expert_norm,
    )

    if layer.w13_weight.dtype != torch.uint8:
        return
    if has_tpu_fused_rs_post_expert_norm(layer):
        prepare_tpu_fused_rs_post_expert_norm(layer)

    chunk_size = 4
    for name in ("w13_weight", "w2_weight"):
        old_w = getattr(layer, name).data
        old_s = getattr(layer, f"{name}_scale").data
        if old_w.dtype != torch.uint8 or old_w.ndim != 3:
            raise RuntimeError(
                f"Expected rank-3 uint8 gmm_v2 weight for {name}, "
                f"got {old_w.dtype} {tuple(old_w.shape)}"
            )
        w_chunks = []
        for e0 in range(0, old_w.shape[0], chunk_size):
            sub = old_w[e0 : e0 + chunk_size].permute(0, 2, 1).contiguous()
            w_chunks.append(_transpose_packed_fp4(sub))
        new_w_u8 = torch.cat(w_chunks, dim=0) if len(w_chunks) > 1 else w_chunks[0]
        del w_chunks
        new_w = _materialize_packed_fp4(new_w_u8)
        del new_w_u8
        s_3d = old_s.squeeze(2) if old_s.ndim == 4 and old_s.shape[2] == 1 else old_s
        if s_3d.ndim != 3:
            raise RuntimeError(
                f"Expected rank-3/4 scale for {name}_scale, got {tuple(old_s.shape)}"
            )
        new_s = _own_storage(s_3d.to(torch.bfloat16).contiguous())
        _wait_for(new_w, new_s)
        copy_or_rebind_param(layer, name, new_w)
        copy_or_rebind_param(layer, f"{name}_scale", new_s)
        _free_replaced_storage(old_w, getattr(layer, name).data)
        _free_replaced_storage(old_s, getattr(layer, f"{name}_scale").data)
        del old_w, old_s, new_w, new_s
    layer.__dict__["_tpu_ep_tp2_packed_fp4"] = True


def _format_nvfp4_replica_bank_from_raw(
    raw: dict[str, torch.Tensor],
    *,
    device: torch.device,
    compact_k256: bool = False,
    compact_block: Optional[int] = None,
    two_level_pow2: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert raw checkpoint-format replica slices to fused-RS FP4 + bf16 scales."""
    from sglang_tpu.fp4_moe import (
        _fold_w13_global_scale,
        _materialize_packed_fp4,
        _transpose_packed_fp4,
    )
    from sglang_tpu.fp8_moe import _own_storage
    from sglang_tpu.moe.nvfp4_requant import (
        compact_nvfp4_blocks,
        compact_nvfp4_two_level_pow2,
    )

    eff_block = (
        int(compact_block)
        if compact_block is not None
        else (256 if compact_k256 else 0)
    )
    compact_fn = (
        compact_nvfp4_two_level_pow2
        if (two_level_pow2 and eff_block == 256)
        else compact_nvfp4_blocks
    )

    w13_w = raw["w13_weight"].to(device=device, copy=True)
    w13_s = raw["w13_weight_scale"].to(device=device, dtype=torch.float32, copy=True)
    if "w13_weight_scale_2" in raw:
        w13_s = _fold_w13_global_scale(
            w13_s, raw["w13_weight_scale_2"].to(device=device, copy=True)
        )
    w13_u8 = _transpose_packed_fp4(w13_w)
    w13_scale = _own_storage(w13_s.to(torch.bfloat16).permute(0, 2, 1).contiguous())
    if eff_block > 0:
        w13_u8, w13_scale, _, _ = compact_fn(w13_u8, w13_scale, eff_block)
    rep_w1 = _materialize_packed_fp4(w13_u8)

    w2_w = raw["w2_weight"].to(device=device, copy=True)
    w2_s = raw["w2_weight_scale"].to(device=device, dtype=torch.float32, copy=True)
    if "w2_weight_scale_2" in raw:
        w2_s = (
            w2_s
            * raw["w2_weight_scale_2"].to(
                device=device, dtype=torch.float32, copy=True
            )[:, None, None]
        )
    w2_u8 = _transpose_packed_fp4(w2_w)
    w2_scale = _own_storage(w2_s.to(torch.bfloat16).permute(0, 2, 1).contiguous())
    if eff_block > 0:
        w2_u8, w2_scale, _, _ = compact_fn(w2_u8, w2_scale, eff_block)
    rep_w2 = _materialize_packed_fp4(w2_u8)
    _wait_for(rep_w1, w13_scale, rep_w2, w2_scale)
    return rep_w1, w13_scale, rep_w2, w2_scale


def _format_fp8_replica_bank_from_raw(
    raw: dict[str, torch.Tensor],
    *,
    device: torch.device,
    scale_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize raw BF16 checkpoint replica slices ``[R, N, K]`` to per-expert FP8
    ``[R, K, N]`` + ``[R, 1, 1, N]`` scales matching ``_load_expert_stack`` + ``fp8_moe``."""
    from sglang_tpu.fp8_moe import _to_tpu_fp8_moe_scale_layout

    fp8_max = torch.finfo(fp8_dtype).max
    out_w: list[torch.Tensor] = []
    out_s: list[torch.Tensor] = []
    for key in ("w13_weight", "w2_weight"):
        src = raw[key].detach().cpu().to(torch.float32)
        scales = src.abs().amax(dim=(1, 2)).clamp(min=1e-12) / fp8_max
        q = (
            (src / scales[:, None, None])
            .clamp(min=-fp8_max, max=fp8_max)
            .to(fp8_dtype)
            .transpose(1, 2)
            .contiguous()
            .to(device=device, copy=True)
        )
        s = _to_tpu_fp8_moe_scale_layout(
            scales.to(device=device, dtype=scale_dtype, copy=True),
            int(q.shape[-1]),
        )
        out_w.append(q)
        out_s.append(s)
    _wait_for(out_w[0], out_s[0], out_w[1], out_s[1])
    return out_w[0], out_s[0], out_w[1], out_s[1]


def _format_fp8_raw_to_nvfp4_replica_bank(
    raw: dict[str, torch.Tensor],
    *,
    device: torch.device,
    block: int,
    two_level_pow2: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize raw BF16 boundary replica slices ``[R, N, K]`` to NVFP4 ``block``."""
    from sglang_tpu.fp4_moe import _materialize_packed_fp4
    from sglang_tpu.moe.nvfp4_requant import (
        _quantize_wb_two_level_pow2,
        quantize_fp8_to_nvfp4_blocks,
    )

    fp8_w1, fp8_s1, fp8_w2, fp8_s2 = _format_fp8_replica_bank_from_raw(
        raw, device=device, scale_dtype=torch.float32
    )
    if two_level_pow2 and block == 256:
        e1, k1, n1 = fp8_w1.shape
        wf1 = fp8_w1.float() * fp8_s1.reshape(e1, 1, n1).float()
        c1, rep_s1, _ = _quantize_wb_two_level_pow2(wf1.reshape(e1, k1 // block, block, n1))
        e2, k2, n2 = fp8_w2.shape
        wf2 = fp8_w2.float() * fp8_s2.reshape(e2, 1, n2).float()
        c2, rep_s2, _ = _quantize_wb_two_level_pow2(wf2.reshape(e2, k2 // block, block, n2))
    else:
        c1, rep_s1, _, _ = quantize_fp8_to_nvfp4_blocks(fp8_w1, fp8_s1, block)
        c2, rep_s2, _, _ = quantize_fp8_to_nvfp4_blocks(fp8_w2, fp8_s2, block)
    rep_w1 = _materialize_packed_fp4(c1)
    rep_w2 = _materialize_packed_fp4(c2)
    _wait_for(rep_w1, rep_s1, rep_w2, rep_s2)
    return rep_w1, rep_s1, rep_w2, rep_s2


def _param_name(checkpoint_name: str) -> Optional[str]:
    """Model parameter for a checkpoint tensor, or None when the text decoder does not use it."""
    if checkpoint_name == "lm_head.weight":
        return checkpoint_name
    if not checkpoint_name.startswith(_TEXT_PREFIX):
        return None
    name = "model." + checkpoint_name.removeprefix(_TEXT_PREFIX)
    if name.endswith(_UNUSED_SUFFIXES):
        return None
    for old, new in _RENAMES:
        if name.endswith(old):
            return name.removesuffix(old) + new
    return name


def _layer_index(name: str) -> int:
    """Decoder layer index of a ``model.layers.<i>.`` parameter name, else -1."""
    if not name.startswith("model.layers."):
        return -1
    return int(name.split(".", 3)[2])


@functools.lru_cache(maxsize=None)
def _expert_placement(path: str) -> dict[int, torch.Tensor]:
    """Layer -> checkpoint expert id of each loaded expert (SGLANG_TPU_MUSE_KERNEL_EXPERT_PLACEMENT)."""
    if not path:
        return {}
    with open(path) as f:
        layers = json.load(f)["layers"]
    placement = {}
    for layer, ids in layers.items():
        order = torch.tensor(ids, dtype=torch.long)
        if not torch.equal(order.sort().values, torch.arange(order.numel())):
            raise ValueError(f"{path}: layer {layer} placement is not a permutation")
        placement[int(layer)] = order
    return placement


def _placed_expert_order(name: str, num_experts: int) -> Optional[torch.Tensor]:
    """Checkpoint expert id per loaded expert for this parameter's layer; None keeps the order."""
    if (
        tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_SKIP_EXPERT_PLACEMENT.get()
        and get_disagg().disaggregation_mode != "prefill"
    ):
        return None
    path = tpu_envs.SGLANG_TPU_MUSE_KERNEL_EXPERT_PLACEMENT.get()
    order = _expert_placement(path).get(_layer_index(name))
    if order is not None and order.numel() != num_experts:
        raise ValueError(
            f"{path}: {order.numel()} placed experts for {name} with {num_experts}"
        )
    return order


def _place_router_rows(name: str, loaded_weight: torch.Tensor) -> torch.Tensor:
    """Gate rows / score bias in the placed expert order, so routing picks the same experts."""
    if not name.endswith((".mlp.gate.weight", ".mlp.e_score_correction_bias")):
        return loaded_weight
    order = _placed_expert_order(name, loaded_weight.shape[0])
    if order is None:
        return loaded_weight
    return loaded_weight.index_select(0, order.to(loaded_weight.device))


def _to_interleaved_rope(weight: torch.Tensor, num_heads: int) -> torch.Tensor:
    """q/k projection rows from the checkpoint's half-split RoPE layout to interleaved pairs."""
    rows = weight.shape[0]
    return (
        weight.reshape(num_heads, 2, rows // num_heads // 2, *weight.shape[1:])
        .transpose(1, 2)
        .reshape(weight.shape)
        .contiguous()
    )


def _rms_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Weightless RMS normalization over the last dim, computed in fp32."""
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class MuseSparkRMSNorm(nn.Module):
    """RMSNorm with the checkpoint's gain conventions.

    ``gain_center`` is added to the stored weight (zero-centered gammas).
    ``scale_input`` applies the gain before normalizing, as the post-FFN and
    post-expert norms do; otherwise the gain scales the normalized output.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        has_weight: bool = True,
        gain_center: float = 0.0,
        scale_input: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.gain_center = gain_center
        self.scale_input = scale_input
        self.weight = (
            nn.Parameter(torch.empty(hidden_size), requires_grad=False)
            if has_weight
            else None
        )
        # fp32 gain cached by precompute_gain() (SGLANG_TPU_MUSE_KERNEL_ENABLED).
        self.register_buffer("cached_gain", None, persistent=False)

    @torch.no_grad()
    def precompute_gain(self) -> None:
        """Caches gain() so decode stops recomputing `weight + center` every step."""
        if self.weight is not None:
            self.cached_gain = self.gain()

    def gain(self) -> Optional[torch.Tensor]:
        if self.cached_gain is not None:
            return self.cached_gain
        if self.weight is None:
            return None
        return self.weight.float() + self.gain_center

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gain = self.gain()
        if gain is not None and self.scale_input:
            return _rms_normalize(x.float() * gain, self.eps).to(x.dtype)
        out = _rms_normalize(x, self.eps)
        if gain is not None:
            out = out * gain
        return out.to(x.dtype)


class MuseSparkGatedResidual(nn.Module):
    """Norm-preserving gated residual: ``alpha * stream + beta * branch``.

    With ``s = gate / temperature``: ``beta = sigmoid(s)`` and
    ``alpha = sqrt(max(sigmoid(-s) * (1 + beta), eps))``, both per channel.
    """

    def __init__(self, dim: int, temperature: float, eps: float = 1e-3):
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.gate = nn.Parameter(torch.empty(dim), requires_grad=False)
        self.register_buffer("alpha", torch.ones(dim), persistent=False)
        self.register_buffer("beta", torch.ones(dim), persistent=False)

    @torch.no_grad()
    def precompute_coefficients(self) -> None:
        scaled = self.gate.float() / self.temperature
        beta = torch.sigmoid(scaled)
        alpha = torch.sqrt(torch.clamp(torch.sigmoid(-scaled) * (1.0 + beta), self.eps))
        self.alpha = alpha.to(self.gate.device)
        self.beta = beta.to(self.gate.device)

    def forward(self, stream: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        return self.alpha * stream + self.beta * branch.to(stream.dtype)


class MuseSparkInterleavedRope(nn.Module):
    """GPT-J style (interleaved pairs) rotary embedding evaluated in fp32."""

    def __init__(self, head_dim: int, max_position: int, base: float):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
        )
        angles = torch.arange(max_position, dtype=torch.float64)[:, None] * inv_freq
        # [max_position, head_dim // 2] each, fp32.
        self.register_buffer("cos", angles.cos().float(), persistent=False)
        self.register_buffer("sin", angles.sin().float(), persistent=False)
        self.head_dim = head_dim
        # SGLANG_TPU_MUSE_KERNEL_ENABLED: [ceil(N / 2), 2 * head_dim] rows of
        # [cos|sin of position 2i, cos|sin of 2i + 1]. The device keeps a [N, 32] f32
        # table position-minor, so a row gather relayouted both whole tables every step;
        # 128-lane rows (head_dim 64) are gathered in place.
        self.register_buffer("cos_sin_pairs", None, persistent=False)
        if tpu_envs.SGLANG_TPU_MUSE_KERNEL_ENABLED.get():
            # Built from fresh angles, not a view of cos/sin: a view buffer (even
            # cloned) is re-materialized on device every step (95 us).
            even = torch.arange(0, max_position, 2, dtype=torch.float64)
            odd = (even + 1).clamp(max=max_position - 1)
            even, odd = even[:, None] * inv_freq, odd[:, None] * inv_freq
            self.cos_sin_pairs = torch.cat(
                (even.cos(), even.sin(), odd.cos(), odd.sin()), dim=-1
            ).float()

    def _cos_sin_from_pairs(
        self, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Clamp keeps the old past-the-end behavior: such positions use the last row.
        positions = positions.clamp(max=self.cos.shape[0] - 1)
        rows = self.cos_sin_pairs.index_select(0, positions // 2)
        half = self.head_dim
        cos_sin = torch.where(
            (positions % 2 == 0)[:, None], rows[:, :half], rows[:, half:]
        ).unsqueeze(1)
        return cos_sin[..., : half // 2], cos_sin[..., half // 2 :]

    def _rotate(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        num_tokens = x.shape[0]
        pairs = x.float().view(num_tokens, -1, self.head_dim // 2, 2)
        even, odd = pairs[..., 0], pairs[..., 1]
        rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1)
        return rotated.view(x.shape).to(x.dtype)

    def forward(
        self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # [num_tokens, 1, head_dim // 2] broadcasts over heads.
        if self.cos_sin_pairs is not None:
            cos, sin = self._cos_sin_from_pairs(positions)
        else:
            cos = self.cos.index_select(0, positions).unsqueeze(1)
            sin = self.sin.index_select(0, positions).unsqueeze(1)
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)


def _layer_tp_size(layer: nn.Module) -> int:
    """Return ``tp_size`` for a parallel linear layer or mock."""
    tp_size = getattr(layer, "tp_size", None)
    if tp_size is not None:
        return int(tp_size)
    tp_group = getattr(layer, "tp_group", None)
    if tp_group is not None:
        return int(get_group_rank_size(tp_group)[1])
    return int(get_parallel().attn_tp_size)


def _layer_tp_rank(layer: nn.Module) -> int:
    """Return ``tp_rank`` for a parallel linear layer or mock."""
    tp_rank = getattr(layer, "tp_rank", None)
    if tp_rank is not None:
        return int(tp_rank)
    tp_group = getattr(layer, "tp_group", None)
    if tp_group is not None:
        return int(get_group_rank_size(tp_group)[0])
    return int(get_parallel().attn_tp_rank)


class MuseSparkAttention(nn.Module):
    """QK-normed attention with a weightless output norm and a sigmoid output gate."""

    def __init__(
        self,
        config: MuseSparkTextConfig,
        layer_idx: int,
        rotary_emb: MuseSparkInterleavedRope,
        prefix: str,
    ):
        super().__init__()
        parallel = get_parallel()
        tp_size = parallel.attn_tp_size
        tp_rank = parallel.attn_tp_rank
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = max(1, config.num_key_value_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.norm_eps = config.rms_norm_eps
        self.decode_qkv_epilogue = tpu_envs.SGLANG_TPU_MUSE_DECODE_QKV_EPILOGUE.get()
        self.decode_attn_gate_fuse = bool(
            _env_get("SGLANG_TPU_MUSE_DECODE_ATTN_GATE_FUSE", False)
        )
        if self.decode_qkv_epilogue:
            warm_decode_qkv_epilogue_ops()
        self.query_scale = config.qk_scale_factor
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=config.num_attention_heads,
            total_num_kv_heads=config.num_key_value_heads,
            bias=False,
            quant_config=None,
            prefix=add_prefix("qkv_proj", prefix),
            parallel_group="attn_tp",
        )
        self.gate_proj = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.num_attention_heads * self.head_dim,
            bias=False,
            gather_output=False,
            quant_config=None,
            prefix=add_prefix("gate_proj", prefix),
            parallel_group="attn_tp",
        )
        self.o_proj = RowParallelLinear(
            input_size=config.num_attention_heads * self.head_dim,
            output_size=config.hidden_size,
            bias=False,
            reduce_results=True,
            quant_config=None,
            prefix=add_prefix("o_proj", prefix),
            parallel_group="attn_tp",
        )
        # Shared across layers; NoPE layers (theta 0) skip it.
        self.rotary_emb = rotary_emb if config.layer_rope_theta[layer_idx] > 0 else None
        sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_idx,
            sliding_window_size=config.sliding_window - 1 if sliding else -1,
            prefix=add_prefix("attn", prefix),
        )
        self.register_buffer("_qkvg_weight", None, persistent=False)
        self.register_buffer("_qkvg_w8k", None, persistent=False)
        self.register_buffer("_qkvg_scale", None, persistent=False)
        self.register_buffer("_o_proj_wk", None, persistent=False)
        self.register_buffer("_o_proj_w8k", None, persistent=False)
        self.register_buffer("_o_proj_scale", None, persistent=False)
        self._fp8_attn_proj = bool(
            _env_get("SGLANG_TPU_MUSE_DECODE_FP8_ATTN_PROJ", False)
        )

    @torch.no_grad()
    def prepare_fused_qkvg(self, *, fp8_wo: bool = False) -> bool:
        """Concatenate matching column-TP ``[qkv_proj | gate_proj]`` weights (0 extra HBM)."""
        qp, gp = self.qkv_proj, self.gate_proj
        if (
            qp.bias is not None
            or gp.bias is not None
            or qp.weight.ndim != 2
            or gp.weight.ndim != 2
            or qp.weight.dtype != torch.bfloat16
            or gp.weight.dtype != torch.bfloat16
            or qp.weight.shape[1] != gp.weight.shape[1]
            or getattr(gp, "gather_output", False)
            or _layer_tp_size(qp) != _layer_tp_size(gp)
            or _layer_tp_rank(qp) != _layer_tp_rank(gp)
        ):
            return False
        qkv_dim = self.q_size + 2 * self.kv_size
        if qp.weight.shape[0] != qkv_dim or gp.weight.shape[0] != self.q_size:
            return False
        old_qkv = qp.weight.data
        old_gate = gp.weight.data
        # Fresh contiguous tensor (not a view) so TorchTPU never re-materializes per step.
        qkvg = torch.cat((old_qkv.detach(), old_gate.detach()), dim=0).contiguous()
        _wait_for(qkvg)
        self._qkvg_weight = qkvg
        if fp8_wo and qkvg.shape[0] % 128 == 0:
            from sglang_tpu.moe.pallas.fused_rs_muse.attn_ar_res_norm import (
                get_muse_decode_fp8_linear_bf16_op,
                quantize_fp8_k_major,
            )

            w8k, scale = quantize_fp8_k_major(qkvg)
            scale = _as_1d_fp8_scale(scale)
            _wait_for(w8k, scale)
            self._qkvg_w8k = w8k
            self._qkvg_scale = scale
            get_muse_decode_fp8_linear_bf16_op()
        if old_qkv.device.type != "meta":
            empty_qkv = torch.empty(
                (0, old_qkv.shape[1]), dtype=old_qkv.dtype, device=old_qkv.device
            )
            empty_gate = torch.empty(
                (0, old_gate.shape[1]), dtype=old_gate.dtype, device=old_gate.device
            )
            qp.weight = nn.Parameter(empty_qkv, requires_grad=False)
            gp.weight = nn.Parameter(empty_gate, requires_grad=False)
            _free_replaced_storage(old_qkv, qp.weight.data)
            _free_replaced_storage(old_gate, gp.weight.data)
        return True

    @torch.no_grad()
    def prepare_fused_o_proj_tail(self, *, fp8_wo: bool = False) -> bool:
        """Pre-transpose ``o_proj.weight`` to contiguous K-major ``[k_local, H]`` for mode=3."""
        op = self.o_proj
        if (
            op.bias is not None
            or op.weight.ndim != 2
            or op.weight.dtype != torch.bfloat16
            or op.weight.shape[0] % 128 != 0
            or op.weight.shape[1] % 128 != 0
        ):
            return False
        if fp8_wo:
            from sglang_tpu.moe.pallas.fused_rs_muse.attn_ar_res_norm import (
                get_muse_decode_fp8_linear_bf16_op,
                quantize_fp8_k_major,
            )

            w8k, scale = quantize_fp8_k_major(op.weight)
            scale = _as_1d_fp8_scale(scale)
            _wait_for(w8k, scale)
            self._o_proj_w8k = w8k
            self._o_proj_scale = scale
            get_muse_decode_fp8_linear_bf16_op()
        else:
            wk = op.weight.detach().t().contiguous()
            _wait_for(wk)
            self._o_proj_wk = wk
        return True

    @torch.no_grad()
    def prepare_w7_qkv_gate_fp8(self, *, free_bf16: bool = False) -> bool:
        """Prepare separate contiguous K-major FP8 ``qkv_proj`` and ``gate_proj`` weights."""
        import inspect

        from sglang_tpu.kernels.decode_collective_bridges import (
            quantize_weight_fp8_k_major,
        )

        qp, gp = self.qkv_proj, self.gate_proj
        supports_128_tail = (
            "allow_128_tail"
            in inspect.signature(quantize_weight_fp8_k_major).parameters
        )
        qkv_col_align = 128 if supports_128_tail else 256
        if (
            qp.bias is not None
            or gp.bias is not None
            or qp.weight.ndim != 2
            or gp.weight.ndim != 2
            or qp.weight.dtype != torch.bfloat16
            or gp.weight.dtype != torch.bfloat16
            or qp.weight.shape[0] == 0
            or gp.weight.shape[0] == 0
            or qp.weight.shape[1] != gp.weight.shape[1]
            or qp.weight.shape[0] % qkv_col_align != 0
            or gp.weight.shape[0] % 256 != 0
            or qp.weight.shape[1] % 128 != 0
        ):
            return False
        wq, sq = (
            quantize_weight_fp8_k_major(qp.weight, allow_128_tail=True)
            if qp.weight.shape[0] % 256 != 0
            else quantize_weight_fp8_k_major(qp.weight)
        )
        wg, sg = quantize_weight_fp8_k_major(gp.weight)
        sq = _as_1d_fp8_scale(sq)
        sg = _as_1d_fp8_scale(sg)
        _wait_for(wq, sq, wg, sg)
        self.register_buffer("_w7_wqkv_w8k", wq, persistent=False)
        self.register_buffer("_w7_sqkv", sq, persistent=False)
        self.register_buffer("_w7_wgate_w8k", wg, persistent=False)
        self.register_buffer("_w7_sgate", sg, persistent=False)
        if free_bf16 and qp.weight.device.type != "meta":
            old_q, old_g = qp.weight.data, gp.weight.data
            qp.weight = nn.Parameter(
                torch.empty(
                    (0, old_q.shape[1]), dtype=old_q.dtype, device=old_q.device
                ),
                requires_grad=False,
            )
            gp.weight = nn.Parameter(
                torch.empty(
                    (0, old_g.shape[1]), dtype=old_g.dtype, device=old_g.device
                ),
                requires_grad=False,
            )
            _free_replaced_storage(old_q, qp.weight.data)
            _free_replaced_storage(old_g, gp.weight.data)
        return True

    @torch.no_grad()
    def prepare_w7_o_proj_fp8(self, *, free_bf16: bool = False) -> bool:
        """Prepare contiguous K-major FP8 ``o_proj`` weight (sharing ``_o_proj_w8k`` if set)."""
        from sglang_tpu.kernels.decode_collective_bridges import (
            quantize_weight_fp8_k_major,
        )

        op = self.o_proj
        if self._o_proj_w8k is None:
            if (
                op.bias is not None
                or op.weight.ndim != 2
                or op.weight.dtype != torch.bfloat16
                or op.weight.shape[0] == 0
                or op.weight.shape[0] % 256 != 0
                or op.weight.shape[1] % 128 != 0
            ):
                return False
            wo, so = quantize_weight_fp8_k_major(op.weight)
            so = _as_1d_fp8_scale(so)
            _wait_for(wo, so)
            self._o_proj_w8k = wo
            self._o_proj_scale = so
        elif self._o_proj_scale is not None:
            so = _as_1d_fp8_scale(self._o_proj_scale)
            if so is not self._o_proj_scale:
                _wait_for(so)
                self._o_proj_scale = so
        if free_bf16 and op.weight.shape[0] > 0 and op.weight.device.type != "meta":
            old_o = op.weight.data
            op.weight = nn.Parameter(
                torch.empty(
                    (0, old_o.shape[1]), dtype=old_o.dtype, device=old_o.device
                ),
                requires_grad=False,
            )
            _free_replaced_storage(old_o, op.weight.data)
        return True

    def _per_head_normalize(self, x: torch.Tensor) -> torch.Tensor:
        num_tokens = x.shape[0]
        heads = x.view(num_tokens, -1, self.head_dim)
        return _rms_normalize(heads, self.norm_eps).view(x.shape)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        *,
        cos_sin_step: Optional[torch.Tensor] = None,
        skip_o_proj_all_reduce: bool = False,
        return_attn_and_gate: bool = False,
        precomputed_qkv_gate: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        precomputed_prepared_qkvg: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None,
        return_gated_attn: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        qkv_dim = self.q_size + 2 * self.kv_size
        max_dec_tok = _decode_max_tokens()
        gate: Optional[torch.Tensor] = None
        prepared_qkv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        w7_wq = self._buffers.get("_w7_wqkv_w8k") if hasattr(self, "_buffers") else None
        if w7_wq is not None and self.qkv_proj.weight.shape[0] > 0:
            from sglang.srt.runtime_context import get_forward

            if getattr(get_forward(), "sp_active", False):
                w7_wq = None
        if precomputed_prepared_qkvg is not None:
            q_5d_in, kv_m_in, gate = precomputed_prepared_qkvg
            prepared_qkv = (q_5d_in, kv_m_in)
            qkv = gate
        elif precomputed_qkv_gate is not None:
            qkv, gate = precomputed_qkv_gate
        elif w7_wq is not None and (
            (
                forward_batch.forward_mode != ForwardMode.EXTEND
                and hidden_states is not None
                and 8 <= hidden_states.shape[0] <= max_dec_tok
                and hidden_states.shape[0] % 8 == 0
            )
            or (
                self.qkv_proj.weight.shape[0] == 0
                and hidden_states is not None
                and 8 <= hidden_states.shape[0] <= max_dec_tok
                and hidden_states.shape[0] % 8 == 0
            )
        ):
            use_l0_qkv_epi = bool(
                self.__dict__.get("_w8_bridge_qkv_epi", False)
            ) and _can_use_bridge_qkv_epi(
                hidden_states.shape[0],
                self,
                cos_sin_step,
                forward_batch,
                max_tok=384,
            )
            use_direct_l0_epi = (
                not use_l0_qkv_epi
                and bool(
                    _env_get("SGLANG_TPU_MUSE_DECODE_BRIDGE2_DIRECT_QKV_EPI", False)
                )
                and self.decode_qkv_epilogue
                and w7_wq.shape[1] == 1280
                and self._buffers["_w7_wgate_w8k"].shape[1] == 1024
                and can_use_muse_decode_qkv_epilogue(
                    torch.empty(
                        (hidden_states.shape[0], 1280),
                        dtype=torch.bfloat16,
                        device=hidden_states.device,
                    ),
                    self.rotary_emb,
                    self.attn,
                    forward_batch,
                )
            )
            if use_l0_qkv_epi:
                from sglang_tpu.kernels.decode_collective_bridges_w8 import (
                    get_fp8_dual_qkv_epi_gate_op,
                )

                k_sc = float(getattr(self.attn, "k_scale_float", None) or 1.0)
                v_sc = float(getattr(self.attn, "v_scale_float", None) or 1.0)
                if self.rotary_emb is not None:
                    q_5d_l0, kv_m_l0, gate = get_fp8_dual_qkv_epi_gate_op(True)(
                        hidden_states,
                        w7_wq,
                        self._buffers["_w7_sqkv"],
                        self._buffers["_w7_wgate_w8k"],
                        self._buffers["_w7_sgate"],
                        cos_sin_step,
                        float(self.norm_eps),
                        float(self.query_scale),
                        k_sc,
                        v_sc,
                    )
                else:
                    q_5d_l0, kv_m_l0, gate = get_fp8_dual_qkv_epi_gate_op(False)(
                        hidden_states,
                        w7_wq,
                        self._buffers["_w7_sqkv"],
                        self._buffers["_w7_wgate_w8k"],
                        self._buffers["_w7_sgate"],
                        float(self.norm_eps),
                        float(self.query_scale),
                        k_sc,
                        v_sc,
                    )
                prepared_qkv = (q_5d_l0, kv_m_l0)
                qkv = gate
            elif use_direct_l0_epi:
                from sglang_tpu.kernels.decode_collective_bridges_carb import (
                    get_fp8_dual_direct_qkv_epi_gate_op,
                )

                k_sc = float(getattr(self.attn, "k_scale_float", None) or 1.0)
                v_sc = float(getattr(self.attn, "v_scale_float", None) or 1.0)
                if self.rotary_emb is not None:
                    cs_l0 = (
                        cos_sin_step
                        if cos_sin_step is not None
                        else get_decode_rope_step_op()(
                            positions.to(torch.int32), self.rotary_emb.cos_sin_pairs
                        )
                    )
                    q_5d_l0, kv_m_l0, gate = get_fp8_dual_direct_qkv_epi_gate_op(
                        has_rope=True
                    )(
                        hidden_states,
                        w7_wq,
                        self._buffers["_w7_sqkv"],
                        self._buffers["_w7_wgate_w8k"],
                        self._buffers["_w7_sgate"],
                        cs_l0,
                        float(self.norm_eps),
                        float(self.query_scale),
                        k_sc,
                        v_sc,
                    )
                else:
                    q_5d_l0, kv_m_l0, gate = get_fp8_dual_direct_qkv_epi_gate_op(
                        has_rope=False
                    )(
                        hidden_states,
                        w7_wq,
                        self._buffers["_w7_sqkv"],
                        self._buffers["_w7_wgate_w8k"],
                        self._buffers["_w7_sgate"],
                        float(self.norm_eps),
                        float(self.query_scale),
                        k_sc,
                        v_sc,
                    )
                prepared_qkv = (q_5d_l0, kv_m_l0)
                qkv = gate
            else:
                from sglang_tpu.kernels.decode_collective_bridges import (
                    get_fp8_dual_qkv_gate_op,
                )

                qkv, gate = get_fp8_dual_qkv_gate_op()(
                    hidden_states,
                    w7_wq,
                    self._buffers["_w7_sqkv"],
                    self._buffers["_w7_wgate_w8k"],
                    self._buffers["_w7_sgate"],
                )
        elif w7_wq is not None and self.qkv_proj.weight.shape[0] == 0:
            assert hidden_states is not None
            xf = hidden_states
            qkv = (
                torch.matmul(xf, w7_wq.to(xf.dtype)).float() * self._buffers["_w7_sqkv"]
            ).to(xf.dtype)
            gate = (
                torch.matmul(xf, self._buffers["_w7_wgate_w8k"].to(xf.dtype)).float()
                * self._buffers["_w7_sgate"]
            ).to(xf.dtype)
        elif (
            self._qkvg_w8k is not None
            and hidden_states is not None
            and forward_batch.forward_mode != ForwardMode.EXTEND
            and 8 <= hidden_states.shape[0] <= max_dec_tok
        ):
            from sglang_tpu.moe.pallas.fused_rs_muse.attn_ar_res_norm import (
                get_muse_decode_fp8_linear_bf16_op,
            )

            qkvg = get_muse_decode_fp8_linear_bf16_op()(
                hidden_states, self._qkvg_w8k, self._qkvg_scale
            )
            qkv, gate = qkvg.split([qkv_dim, self.q_size], dim=-1)
        elif self._qkvg_weight is not None:
            qkvg = F.linear(hidden_states, self._qkvg_weight)
            qkv, gate = qkvg.split([qkv_dim, self.q_size], dim=-1)
        else:
            qkv, _ = self.qkv_proj(hidden_states)
        post_fused = False
        verify_gate_ok = forward_batch.forward_mode.is_target_verify() and bool(
            _env_get("SGLANG_TPU_MUSE_KERNEL_VERIFY_RPA_FAST", False)
        )
        if prepared_qkv is not None:
            from sglang_tpu.decode_qkv_epilogue_tpu import _resolve_decode_attn_backend

            q_5d_p, kv_m_p = prepared_qkv
            backend = _resolve_decode_attn_backend(forward_batch)
            assert backend is not None
            fuse_gate_p = (
                getattr(self, "decode_attn_gate_fuse", False)
                and (forward_batch.forward_mode.is_decode() or verify_gate_ok)
                and not return_attn_and_gate
                and gate is not None
            )
            attn_output = None
            if fuse_gate_p:
                fn_gated = getattr(backend, "forward_decode_prepared_gated", None)
                if fn_gated is not None:
                    attn_output = fn_gated(
                        q_5d_p, kv_m_p, gate, self.attn, forward_batch
                    )
                    if attn_output is not None:
                        post_fused = True
            if attn_output is None:
                attn_output = backend.forward_decode_prepared(
                    q_5d_p, kv_m_p, self.attn, forward_batch
                )
        elif self.decode_qkv_epilogue and can_use_muse_decode_qkv_epilogue(
            qkv, self.rotary_emb, self.attn, forward_batch
        ):
            fuse_gate = (
                getattr(self, "decode_attn_gate_fuse", False)
                and (forward_batch.forward_mode.is_decode() or verify_gate_ok)
                and not return_attn_and_gate
            )
            if fuse_gate and gate is None:
                gate, _ = self.gate_proj(hidden_states)
            if fuse_gate:
                attn_output, post_fused = muse_decode_fused_qkv_attn(
                    positions,
                    qkv.contiguous(),
                    rotary_emb=self.rotary_emb,
                    attn_layer=self.attn,
                    forward_batch=forward_batch,
                    cos_sin_step=cos_sin_step,
                    norm_eps=self.norm_eps,
                    query_scale=self.query_scale,
                    gate=gate.contiguous(),
                    return_post_fused=True,
                )
            else:
                attn_output = muse_decode_fused_qkv_attn(
                    positions,
                    qkv,
                    rotary_emb=self.rotary_emb,
                    attn_layer=self.attn,
                    forward_batch=forward_batch,
                    cos_sin_step=cos_sin_step,
                    norm_eps=self.norm_eps,
                    query_scale=self.query_scale,
                )
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q = (self._per_head_normalize(q) * self.query_scale).to(qkv.dtype)
            k = self._per_head_normalize(k).to(qkv.dtype)
            if self.rotary_emb is not None:
                q, k = self.rotary_emb(positions, q, k)
            attn_output = self.attn(q, k, v, forward_batch)
        if post_fused:
            gated = attn_output
        else:
            if gate is None:
                gate, _ = self.gate_proj(hidden_states)
            if return_attn_and_gate:
                return attn_output, gate
            attn_output = self._per_head_normalize(attn_output).to(qkv.dtype)
            gated = torch.sigmoid(gate) * attn_output
        if return_gated_attn:
            return gated
        return self._forward_o_proj(
            gated,
            forward_batch,
            skip_o_proj_all_reduce=skip_o_proj_all_reduce,
        )

    def _forward_o_proj(
        self,
        gated: torch.Tensor,
        forward_batch: ForwardBatch,
        *,
        skip_o_proj_all_reduce: bool = False,
    ) -> torch.Tensor:
        max_dec_tok = _decode_max_tokens()
        n_rows = gated.shape[0]
        if self._o_proj_w8k is not None and (
            (
                forward_batch.forward_mode != ForwardMode.EXTEND
                and 8 <= n_rows <= max_dec_tok
            )
            or (
                self.o_proj.weight.shape[0] == 0
                and 8 <= n_rows <= max_dec_tok
                and n_rows % 8 == 0
            )
        ):
            from sglang_tpu.moe.pallas.fused_rs_muse.attn_ar_res_norm import (
                get_muse_decode_fp8_linear_bf16_op,
            )

            o_part = get_muse_decode_fp8_linear_bf16_op()(
                gated, self._o_proj_w8k, self._o_proj_scale
            )
            if (
                skip_o_proj_all_reduce
                or not self.o_proj.reduce_results
                or _layer_tp_size(self.o_proj) <= 1
            ):
                return o_part
            return tensor_model_parallel_all_reduce(o_part)
        if self._o_proj_w8k is not None and self.o_proj.weight.shape[0] == 0:
            o_part = (
                torch.matmul(gated, self._o_proj_w8k.to(gated.dtype)).float()
                * self._o_proj_scale
            ).to(gated.dtype)
            if (
                skip_o_proj_all_reduce
                or not self.o_proj.reduce_results
                or _layer_tp_size(self.o_proj) <= 1
            ):
                return o_part
            return tensor_model_parallel_all_reduce(o_part)
        output, _ = self.o_proj(
            gated,
            skip_all_reduce=skip_o_proj_all_reduce,
        )
        return output


def _uses_quantized_experts(
    quant_config: Optional[QuantizationConfig], experts_prefix: str
) -> bool:
    """Whether a layer's experts are serialized NVFP4.

    The checkpoint excludes whole layers with dotted substrings such as
    ``"layers.0."``, which ModelOpt's component-wise matcher does not accept.
    """
    if quant_config is None:
        return False
    if not isinstance(quant_config, ModelOptQuantConfig):
        raise ValueError(
            f"Muse Spark supports BF16 or ModelOpt NVFP4, got {quant_config.get_name()}"
        )
    norm_prefixes = [experts_prefix + "."]
    if experts_prefix.startswith("model.") and not experts_prefix.startswith(
        "model.language_model."
    ):
        norm_prefixes.append(
            "model.language_model." + experts_prefix.removeprefix("model.") + "."
        )
    elif experts_prefix.startswith("model.language_model."):
        norm_prefixes.append(
            "model." + experts_prefix.removeprefix("model.language_model.") + "."
        )
    return not any(
        any(pattern in pfx or pfx[:-1] == pattern for pfx in norm_prefixes)
        for pattern in quant_config.exclude_modules
    )


class _HostQuantizedFp8MoEMethod(Fp8MoEMethod):
    """Per-expert dynamic FP8 MoE whose BF16 checkpoint weights are quantized on
    host during streaming load so single-device PP stages never hold both the
    BF16 and FP8 expert stacks in HBM."""

    def create_weights(
        self,
        layer: nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,
    ):
        super().create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=fp8_dtype,
            with_bias=with_bias,
            **extra_weight_attrs,
        )
        old_w13_scale = layer.w13_weight_scale.data
        copy_or_rebind_param(
            layer,
            "w13_weight_scale",
            torch.ones(
                num_experts,
                dtype=torch.float32,
                device=layer.w13_weight.device,
            ),
        )
        _free_replaced_storage(old_w13_scale, layer.w13_weight_scale.data)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        from sglang.srt.layers.moe import get_moe_runner_backend
        from sglang_tpu.fp8_moe import prepare_tpu_fp8_moe_weights

        prepare_tpu_fp8_moe_weights(
            layer,
            block_quant=False,
            weight_block_size=None,
            use_mxfp8=False,
            is_fp4_expert=False,
        )
        moe_runner_backend = get_moe_runner_backend()
        if moe_runner_backend.is_auto() or moe_runner_backend.is_pallas_tpu_fused_rs():
            from sglang_tpu.fp8_moe_fused_rs import prepare_fp8_fused_rs

            self.handles_moe_reduction = prepare_fp8_fused_rs(layer)
            if self.handles_moe_reduction and get_parallel().tp_rank == 0:
                logger.info_once("Using TPU Pallas FP8 fused-RS MoE backend")
            if (
                moe_runner_backend.is_pallas_tpu_fused_rs()
                and not self.handles_moe_reduction
            ):
                reason = getattr(
                    layer,
                    "_tpu_fused_rs_unsupported_reason",
                    "unknown incompatibility",
                )
                raise ValueError(
                    "--moe-runner-backend pallas_tpu_fused_rs is not supported by "
                    f"this model or parallel layout: {reason}"
                )
        if hasattr(layer, "dispatcher"):
            layer.dispatcher.set_quant_config(
                {
                    "weight_dtype": layer.w13_weight.dtype,
                    "use_mxfp8": False,
                }
            )

    def forward_tpu(self, layer, dispatch_output):
        if layer.w13_weight.dtype == torch.float4_e2m1fn_x2:
            from sglang_tpu.fp4_moe import apply_tpu_nvfp4_moe

            return apply_tpu_nvfp4_moe(
                layer=layer,
                dispatch_output=dispatch_output,
                activation=self.moe_runner_config.activation,
                apply_router_weight_on_input=self.moe_runner_config.apply_router_weight_on_input,
                no_combine=self.moe_runner_config.no_combine,
                use_fused_rs=bool(self.handles_moe_reduction),
                swiglu_limit=self.moe_runner_config.gemm1_clamp_limit,
            )
        return super().forward_tpu(layer, dispatch_output)


class _HostQuantizedFp8Config(Fp8Config):
    """FP8 config that binds ``_HostQuantizedFp8MoEMethod`` onto ``FusedMoE``."""

    def __init__(self):
        super().__init__(is_checkpoint_fp8_serialized=False)

    def get_quant_method(self, layer: nn.Module, prefix: str):
        if isinstance(layer, FusedMoE):
            return _HostQuantizedFp8MoEMethod(self)
        return super().get_quant_method(layer, prefix)


class MuseTpuBf16MoEMethod(UnquantizedFusedMoEMethod):
    """Runs BF16 experts on fused-RS, which applies Muse's post-expert norm.

    Only expert TP is supported, because the BF16 kernel does not compile under
    expert parallelism. Other backends, and the auto backend on layouts fused-RS
    does not support, keep the unquantized path.
    """

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        from sglang.srt.layers.moe import get_moe_runner_backend

        super().process_weights_after_loading(layer)
        self.handles_moe_reduction = False
        backend = get_moe_runner_backend()
        if backend.is_auto() or backend.is_pallas_tpu_fused_rs():
            if get_parallel().moe_ep_size > 1:
                layer.__dict__["_tpu_fused_rs_unsupported_reason"] = (
                    "BF16 fused-RS requires an expert parallel size of 1"
                )
            else:
                self.handles_moe_reduction = prepare_fp8_fused_rs(layer)
            if backend.is_pallas_tpu_fused_rs() and not self.handles_moe_reduction:
                raise ValueError(
                    "TPU BF16 fused-RS is not supported: "
                    f"{layer._tpu_fused_rs_unsupported_reason}"
                )

    def forward_tpu(self, layer, dispatch_output):
        if not self.handles_moe_reduction:
            return super().forward_tpu(layer, dispatch_output)
        return apply_fp8_fused_rs(
            layer=layer,
            dispatch_output=dispatch_output,
            activation=self.moe_runner_config.activation,
            swiglu_limit=self.moe_runner_config.gemm1_clamp_limit,
        )


def _experts_quant_config(
    quant_config: Optional[QuantizationConfig], experts_prefix: str
) -> Optional[QuantizationConfig]:
    """Quantization of a layer's experts, or None to keep them BF16.

    Muse NVFP4 excludes only the first and last layers.
    """
    if _uses_quantized_experts(quant_config, experts_prefix):
        return quant_config
    # A BF16 checkpoint has no NVFP4 layers, so the flag would reach every layer.
    if (
        quant_config is not None
        and tpu_envs.SGLANG_TPU_ENABLE_MUSE_BF16_FIRST_LAST_LAYER_EXPERTS.get()
    ):
        if tpu_envs.SGLANG_TPU_MUSE_MEGAKERNEL.get():
            raise ValueError(
                "SGLANG_TPU_ENABLE_MUSE_BF16_FIRST_LAST_LAYER_EXPERTS cannot be "
                "combined with SGLANG_TPU_MUSE_MEGAKERNEL, which reads only fp4 "
                "or fp8 experts"
            )
        return None
    # Quantized on host during load, so single-device PP stages never hold the
    # 12 GiB BF16 stack in HBM.
    return _HostQuantizedFp8Config()


def _oneshot_gather_columns(
    shard: torch.Tensor, *, rank_t: torch.Tensor, world_size: int
) -> torch.Tensor:
    """[T, N/tp] column shard -> [T, N] through the one-shot Pallas all-gather."""
    from sglang_tpu.oneshot_collectives import oneshot_all_gather

    tokens, cols = shard.shape
    full = oneshot_all_gather(
        shard.contiguous(), rank_input=rank_t, world_size=world_size
    )
    return (
        full.view(world_size, tokens, cols)
        .transpose(0, 1)
        .reshape(tokens, world_size * cols)
    )


class _OneshotLogitsGatherer(nn.Module):
    """TP vocab-logits gather for SGLANG_TPU_MUSE_KERNEL_ENABLED.

    The default TP gather is the #1700 workaround (sglang_tpu/all_gather_via_all_reduce.py):
    an all-reduce of a zero-padded f32 [tp, T, V/tp] buffer. The one-shot Pallas gather
    sidesteps both that and XLA's all-gather; shards over the cap keep ``fallback``.
    """

    # The kernel stages 1 local shard in VMEM (out_specs writes directly to HBM);
    # 26 MiB covers B <= 256 (12.33 MiB at B=128, 24.66 MiB at B=256 < 32 MiB VMEM).
    _MAX_SHARD_BYTES_DEFAULT = 8 << 20
    _MAX_SHARD_BYTES_26M = 26 << 20

    def __init__(self, fallback, *, tp_rank: int, tp_size: int):
        super().__init__()
        from sglang_tpu.oneshot_collectives import get_oneshot_all_gather_op

        get_oneshot_all_gather_op(tp_size)
        self._fallback = fallback
        self._tp_size = tp_size
        self._max_shard_bytes = (
            self._MAX_SHARD_BYTES_26M
            if tpu_envs.SGLANG_TPU_MUSE_KERNEL_LOGITS_ONESHOT_26M.get()
            else self._MAX_SHARD_BYTES_DEFAULT
        )
        self.register_buffer(
            "_tp_rank_t", torch.tensor([tp_rank], dtype=torch.int32), persistent=False
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.numel() * logits.element_size() > self._max_shard_bytes:
            return self._fallback(logits)
        return _oneshot_gather_columns(
            logits, rank_t=self._tp_rank_t, world_size=self._tp_size
        )


def _gather_token_sharded_chunks(
    chunks: list[torch.Tensor],
    chunk_rows: list[tuple[int, int]],
    tp_size: int,
) -> torch.Tensor:
    """All-gather per-chunk token-sharded MoE outputs back into token order.

    Chunk ``i`` contributes ``chunk_rows[i][0]`` rows per rank (rank ``r`` owns
    chunk rows ``[r * rpr, (r + 1) * rpr)``) and ``chunk_rows[i][1]`` real rows.
    One all-gather covers every chunk.
    """
    local = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
    if tp_size == 1:
        gathered = local
    else:
        gathered = tensor_model_parallel_all_gather(local.contiguous(), dim=0)
    hidden = gathered.shape[-1]
    per_rank = gathered.view(tp_size, local.shape[0], hidden)
    parts = []
    off = 0
    for rpr, rem in chunk_rows:
        parts.append(per_rank[:, off : off + rpr].reshape(tp_size * rpr, hidden)[:rem])
        off += rpr
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)


class MuseSparkMoE(nn.Module):
    def __init__(
        self,
        config: MuseSparkTextConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()
        parallel = get_parallel()
        self._is_hybrid_ep_tp2 = _is_decode_ep_tp2_layout(parallel)
        # Either expert parallelism (whole experts per rank), expert TP (every
        # rank holds an intermediate slice of every expert), or opt-in symmetric
        # hybrid EP(P/2)xTP2 decode (SGLANG_TPU_MUSE_KERNEL_DECODE_EP_TP2).
        if (
            parallel.moe_tp_size != 1
            and parallel.moe_ep_size != 1
            and not self._is_hybrid_ep_tp2
        ):
            raise ValueError("Muse Spark MoE supports EP or expert TP, not both")
        if config.post_norm_eps != _FUSED_RS_POST_EXPERT_NORM_EPS:
            raise ValueError("The fused-RS post-expert norm assumes eps=1e-8")
        self.tp_size = parallel.tp_size
        self.layer_idx = int(layer_idx)
        self.num_layers = int(config.num_hidden_layers)
        self._fused_decode_op = None
        self._fused_decode_sharded_op = None
        self._fused_decode_batches: Tuple[int, ...] = (16, 32)
        self._fused_decode_v2_op = None
        self._fused_decode_v2_sharded_op = None
        self._fused_decode_v2_batches: Tuple[int, ...] = ()
        self._fused_decode_num_replicas = 0
        self._fused_decode_has_spec_slot = False
        self._fused_decode_has_free_params = False
        self._fused_decode_mode = ""
        self._fused_decode_block = 256
        self._fused_decode_replica_cost_mode = "tiles"
        self._decode_allow_prefill_eplb = bool(
            _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_ALLOW_PREFILL_EPLB", False)
        )
        self._decode_sparse_v2 = bool(
            _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_SPARSE_V2", False)
        )
        self._decode_sparse_fp8 = bool(
            _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_SPARSE_FP8", False)
        )
        self.moe_ep_size = parallel.moe_ep_size
        self.moe_tp_size = parallel.moe_tp_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.routed_scaling_factor
        self.route_eps = _ROUTE_EPS
        self.router_multiplier = config.output_multiplier
        self.expert_hidden_size = config.moe_hidden_size

        # Replicated: the fp32 weight is 8 MiB per layer, and at small batch the
        # all-gather a sharded router would need costs more than the weight read.
        self.gate = ReplicatedLinear(
            config.hidden_size,
            self.num_experts,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(self.num_experts, dtype=torch.float32), requires_grad=False
        )
        from sglang_tpu.moe.expert_tp import is_fused_moe_rs_expert_tp

        # With expert TP, the small decode-sized projection outputs are gathered
        # by the one-shot Pallas all-gather, as the fused-RS output is.
        self.oneshot_gather = is_fused_moe_rs_expert_tp(
            parallel.moe_ep_size, parallel.moe_tp_size, parallel.tp_size
        ) or (
            parallel.tp_size > 1
            and get_disagg().disaggregation_mode != "prefill"
            and tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_ONESHOT_MOE_AG.get()
        )
        self._manual_post_gather = self.oneshot_gather or (
            parallel.tp_size > 1
            and get_disagg().disaggregation_mode != "prefill"
            and tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_NORM_LAYOUT.get() >= 4
        )
        self._decode_fused_router = (
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_FUSED_ROUTER.get() >= 1
            and self.num_experts == 256
        )
        if self._decode_fused_router:
            from sglang_tpu.moe.pallas.fused_rs_muse.decode_norm_router import (
                get_muse_decode_router_topk_op,
            )

            get_muse_decode_router_topk_op()
            eplb_r = int(_env_get("SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_REPLICAS", 0))
            if eplb_r > 0 and parallel.moe_ep_size > 1:
                get_muse_decode_router_topk_op(eplb_r, parallel.moe_ep_size)
        if self.oneshot_gather:
            from sglang_tpu.oneshot_collectives import get_oneshot_all_gather_op

            get_oneshot_all_gather_op(self.tp_size)
            self.register_buffer(
                "_tp_rank_t",
                torch.tensor([parallel.tp_rank], dtype=torch.int32),
                persistent=False,
            )
        # Sharded by output channel and gathered: every token needs the full
        # projected vector (experts live on all ranks), and a replicated copy
        # would re-read both 8192x4096 matrices on every rank each step.
        self.pre_expert_proj = ColumnParallelLinear(
            config.hidden_size,
            self.expert_hidden_size,
            bias=False,
            gather_output=not self.oneshot_gather,
            quant_config=None,
            prefix=add_prefix("pre_expert_proj", prefix),
        )
        self.post_expert_proj = ColumnParallelLinear(
            self.expert_hidden_size,
            config.hidden_size,
            bias=False,
            gather_output=not self._manual_post_gather,
            quant_config=None,
            prefix=add_prefix("post_expert_proj", prefix),
        )
        self.pre_expert_norm = MuseSparkRMSNorm(
            self.expert_hidden_size, eps=config.rms_norm_eps, gain_center=1.0
        )
        self.post_expert_norm = MuseSparkRMSNorm(
            self.expert_hidden_size,
            eps=config.post_norm_eps,
            gain_center=1.0,
            scale_input=True,
        )
        experts_prefix = add_prefix("experts", prefix)
        self.quantized_experts = _uses_quantized_experts(quant_config, experts_prefix)
        experts_quant_config = _experts_quant_config(quant_config, experts_prefix)
        experts_quant_method = None
        if experts_quant_config is None:
            experts_quant_method = MuseTpuBf16MoEMethod()
        self.experts = FusedMoE(
            num_experts=self.num_experts,
            hidden_size=self.expert_hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=layer_idx,
            top_k=self.top_k,
            reduce_results=False,
            quant_config=experts_quant_config,
            quant_method=experts_quant_method,
            prefix=experts_prefix,
            activation="silu",
            # Per-slot outputs: the post-expert norm runs before the combine.
            no_combine=True,
        )
        self.experts.supports_tpu_fused_rs_post_expert_norm = True
        self.experts.tpu_fused_rs_post_expert_norm_weight = lambda: (
            self.post_expert_norm.weight.data
        )
        if self._is_hybrid_ep_tp2:
            ep_rank, _ = _resolve_ep_tp2_coords(parallel.tp_rank, parallel.moe_ep_size)
            self.experts.moe_ep_rank = ep_rank

    def attach_preloaded_replicas(
        self,
        *,
        rep_w1: torch.Tensor,
        rep_s1: torch.Tensor,
        rep_w2: torch.Tensor,
        rep_s2: torch.Tensor,
        replica_meta: torch.Tensor,
    ) -> None:
        """Validate and attach §4.5b preloaded hot-expert replica buffers to this MoE layer."""
        ex = self.experts
        num_replicas = int(rep_w1.shape[0])
        if num_replicas not in (1, 2, 4, 6, 8, 12, 16):
            raise ValueError(
                "Preloaded replica slots must be in (1, 2, 4, 6, 8, 12, 16), "
                f"got {num_replicas}"
            )
        for name, rep, base in (
            ("rep_w1", rep_w1, ex.w13_weight),
            ("rep_s1", rep_s1, ex.w13_weight_scale),
            ("rep_w2", rep_w2, ex.w2_weight),
            ("rep_s2", rep_s2, ex.w2_weight_scale),
        ):
            if rep.dtype != base.dtype or tuple(rep.shape) != (
                num_replicas,
                *base.shape[1:],
            ):
                raise ValueError(
                    f"{name} shape/dtype mismatch: got {tuple(rep.shape)} {rep.dtype}, "
                    f"expected {(num_replicas, *base.shape[1:])} {base.dtype}"
                )
        num_owners = int(self.__dict__.get("moe_ep_size", self.tp_size))
        expected_meta_shape = (4, num_owners * num_replicas)
        if (
            replica_meta.dtype != torch.int32
            or tuple(replica_meta.shape) != expected_meta_shape
        ):
            raise ValueError(
                f"replica_meta mismatch: got {tuple(replica_meta.shape)} {replica_meta.dtype}, "
                f"expected {expected_meta_shape} torch.int32"
            )
        self.register_buffer("_fused_decode_rep_w1", rep_w1, persistent=False)
        self.register_buffer("_fused_decode_rep_s1", rep_s1, persistent=False)
        self.register_buffer("_fused_decode_rep_w2", rep_w2, persistent=False)
        self.register_buffer("_fused_decode_rep_s2", rep_s2, persistent=False)
        self.register_buffer(
            "_fused_decode_replica_meta", replica_meta, persistent=False
        )

    def prepare_fused_decode(
        self,
        *,
        target_ep: int = 16,
        mode: str = "legacy",
        num_replicas: int = 0,
        replica_cost_mode: str = "tiles",
        chipmate_xor: Optional[int] = None,
    ) -> None:
        """Bind a validated ordinary-decoder specialization after weight loading."""
        from sglang.srt.runtime_context import get_exec
        from sglang_tpu.fp4_moe import qblock_layout
        from sglang_tpu.fused_rs_sequence_parallel import (
            get_tpu_fused_rs_post_expert_norm_weight,
        )

        ep = self.tp_size
        parallel = get_parallel()
        ex = self.experts
        graph = get_exec().graph
        if mode != "legacy":
            self._prepare_fuse_all_decode(
                mode=mode,
                parallel=parallel,
                graph=graph,
                ex=ex,
                num_replicas=num_replicas,
                replica_cost_mode=replica_cost_mode,
                chipmate_xor=chipmate_xor,
            )
            if bool(
                _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_RETURN_SHARDED", False)
            ) or int(_env_get("SGLANG_TPU_MUSE_DECODE_COLLECTIVE_BRIDGES", 0)) in (
                2,
                3,
                4,
            ):
                self.enable_sharded_moe_out()
            return
        if (
            ep != target_ep
            or ep not in (8, 16)
            or (ep == 8 and not self._decode_fused_router)
            # The fan-out addresses peers using the local expert range. Shared
            # rank-lifted traces require an explicitly lifted owner operand.
            or graph.tpu_compile_tracer != "dynamo"
            or graph.use_spmd_compile
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() != parallel.tp_rank
            or parallel.moe_ep_size != ep
            or parallel.moe_ep_rank != parallel.tp_rank
            or parallel.moe_tp_size != 1
            or parallel.pp_size != 1
            or parallel.attn_dp_size != 1
            or parallel.attn_cp_size != 1
            or parallel.attn_dcp_size != 1
            or int(os.environ.get("WORLD_SIZE", "1")) != ep
            or self.__dict__.get("_eplb_num_replicas", 0)
            or not self.quantized_experts
            or not ex.handles_moe_reduction
            or not hasattr(ex, "_tpu_fused_rs_group_offset")
            or "_stacked_w13_weight" in ex._buffers
            or qblock_layout(ex)
            or (self.num_experts, self.top_k, self.expert_hidden_size) != (256, 8, 4096)
            or not self._decode_fused_router
            or self.route_scale != 1.0
            or self.route_eps != 1e-15
            or tuple(self.gate.weight.shape) != (256, 8192)
            or self.gate.weight.dtype != torch.float32
            or self.e_score_correction_bias.dtype != torch.float32
            or ex.w13_weight.dtype != torch.float4_e2m1fn_x2
            or ex.w2_weight.dtype != torch.float4_e2m1fn_x2
        ):
            return
        local = 256 // ep
        block = 16 if ep == 8 else 256
        norm = get_tpu_fused_rs_post_expert_norm_weight(ex)
        if (
            tuple(ex.w13_weight.shape) != (local, 4096, 4096)
            or tuple(ex.w2_weight.shape) != (local, 4096, 2048)
            or tuple(ex.w13_weight_scale.shape) != (local, 4096 // block, 8192)
            or tuple(ex.w2_weight_scale.shape) != (local, 4096 // block, 4096)
            or ex.w13_weight_scale.dtype != torch.bfloat16
            or ex.w2_weight_scale.dtype != torch.bfloat16
            or norm is None
            or tuple(norm.shape) != (4096,)
            or norm.dtype not in (torch.bfloat16, torch.float32)
            or (ep == 8 and norm.dtype != torch.bfloat16)
        ):
            return
        from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_op import (
            get_fused_decode_op,
        )

        self._register_fused_decode_vecs(norm)
        self.register_buffer(
            "_fused_decode_rank",
            ex._tpu_fused_rs_group_offset // local,
            persistent=False,
        )
        self._fused_decode_batches = (16, 32)
        self._fused_decode_num_replicas = 0
        self._fused_decode_mode = "legacy"
        self._fused_decode_op = (
            get_fused_decode_op(8) if ep == 8 else get_fused_decode_op()
        )

    def _prepare_fuse_all_decode(
        self,
        *,
        mode: str,
        parallel,
        graph,
        ex,
        num_replicas: int,
        replica_cost_mode: str,
        chipmate_xor: Optional[int],
    ) -> None:
        """Bind a Track 1/2/3 fuse-all decode op after strict guardrail validation."""
        from sglang_tpu.fp4_moe import qblock_layout
        from sglang_tpu.fused_rs_sequence_parallel import (
            get_tpu_fused_rs_post_expert_norm_weight,
        )

        world = self.tp_size
        is_full_moe = mode in ("full_moe", "full_stage_a", "full_stage_b")
        allow_eplb = bool(
            self.__dict__.get("_decode_allow_prefill_eplb", False)
            or _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_ALLOW_PREFILL_EPLB", False)
        )
        allow_fp8 = bool(
            mode != "ep_tp2"
            and not self.quantized_experts
            and _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_FP8_FUSE_ALL", False)
        )
        if (
            mode
            not in (
                "sparse",
                "ep_tp2",
                "chipmate",
                "full_moe",
                "full_stage_a",
                "full_stage_b",
            )
            or world not in (8, 16)
            or num_replicas not in (0, 1, 2, 4, 6, 8, 12, 16)
            or replica_cost_mode not in ("active", "tiles", "weighted")
            or graph.tpu_compile_tracer != "dynamo"
            or graph.use_spmd_compile
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() != parallel.tp_rank
            or parallel.pp_size != 1
            or parallel.attn_dp_size != 1
            or parallel.attn_cp_size != 1
            or parallel.attn_dcp_size != 1
            or getattr(parallel, "moe_dp_size", 1) != 1
            or int(os.environ.get("WORLD_SIZE", "1")) != world
            or (
                self.__dict__.get("_eplb_num_replicas", 0)
                and (mode == "ep_tp2" or not allow_eplb)
            )
            or "_stacked_w13_weight" in ex._buffers
            or qblock_layout(ex)
            or (self.num_experts, self.top_k, self.expert_hidden_size) != (256, 8, 4096)
            or self.route_scale != 1.0
            or self.route_eps != 1e-15
            or tuple(self.gate.weight.shape) != (256, 8192)
            or self.gate.weight.dtype != torch.float32
            or self.e_score_correction_bias.dtype != torch.float32
        ):
            return

        norm = get_tpu_fused_rs_post_expert_norm_weight(ex)
        if (
            norm is None
            or tuple(norm.shape) != (4096,)
            or norm.dtype not in (torch.bfloat16, torch.float32)
        ):
            return

        has_shadow_fp4 = "_decode_fp4_w1" in self._buffers
        if not self.quantized_experts and not has_shadow_fp4:
            if allow_fp8:
                e_owner = 256 // world
                w_groups = int(ex.w13_weight.shape[0])
                valid_groups = (
                    (e_owner <= w_groups <= e_owner + 8)
                    if allow_eplb
                    else (w_groups == e_owner)
                )
                w1_sc = getattr(ex, "w13_weight_scale", None)
                w2_sc = getattr(ex, "w2_weight_scale", None)
                if (
                    parallel.moe_ep_size != world
                    or parallel.moe_ep_rank != parallel.tp_rank
                    or parallel.moe_tp_size != 1
                    or not ex.handles_moe_reduction
                    or not hasattr(ex, "_tpu_fused_rs_group_offset")
                    or getattr(ex, "weight_block_size", None) is not None
                    or ex.w13_weight.dtype != fp8_dtype
                    or ex.w2_weight.dtype != fp8_dtype
                    or not valid_groups
                    or tuple(ex.w13_weight.shape) != (w_groups, 4096, 8192)
                    or tuple(ex.w2_weight.shape) != (w_groups, 4096, 4096)
                    or w1_sc is None
                    or w2_sc is None
                    or w1_sc.dtype not in (torch.bfloat16, torch.float32)
                    or w2_sc.dtype not in (torch.bfloat16, torch.float32)
                    or w1_sc.numel() != w_groups * 8192
                    or w2_sc.numel() != w_groups * 4096
                ):
                    return
                from sglang_tpu.fp8_moe import _own_storage
                from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_op import (
                    SUPPORTED_FUSE_ALL_BATCHES,
                    get_muse_fuse_all_decode_op,
                )

                s1_2d = _own_storage(
                    w1_sc.data.reshape(w_groups, 8192)[:e_owner].contiguous()
                )
                s2_2d = _own_storage(
                    w2_sc.data.reshape(w_groups, 4096)[:e_owner].contiguous()
                )
                if num_replicas > 0:
                    bufs = self._buffers
                    rep_w1 = bufs.get("_fused_decode_rep_w1")
                    rep_s1 = bufs.get("_fused_decode_rep_s1")
                    rep_w2 = bufs.get("_fused_decode_rep_w2")
                    rep_s2 = bufs.get("_fused_decode_rep_s2")
                    replica_meta = bufs.get("_fused_decode_replica_meta")
                    if any(
                        t is None
                        for t in (rep_w1, rep_s1, rep_w2, rep_s2, replica_meta)
                    ):
                        return
                    if (
                        rep_w1.dtype != fp8_dtype
                        or tuple(rep_w1.shape) != (num_replicas, 4096, 8192)
                        or rep_w2.dtype != fp8_dtype
                        or tuple(rep_w2.shape) != (num_replicas, 4096, 4096)
                        or rep_s1.numel() != num_replicas * 8192
                        or rep_s2.numel() != num_replicas * 4096
                        or replica_meta.dtype != torch.int32
                        or tuple(replica_meta.shape) != (4, world * num_replicas)
                    ):
                        return
                self.register_buffer("_fused_decode_fp8_s1", s1_2d, persistent=False)
                self.register_buffer("_fused_decode_fp8_s2", s2_2d, persistent=False)
                self._register_fused_decode_vecs(norm)
                self.register_buffer(
                    "_fused_decode_rank",
                    ex._tpu_fused_rs_group_offset // w_groups,
                    persistent=False,
                )
                # Gate FP8 sparse decode to B <= 128 by default (wins -35..-76 us/L at B=16..128),
                # plus B = 384 under DECODE_B384=1 (ring3_8m wins -240..-252 us/L vs stock at B=384).
                fp8_batches = tuple(b for b in SUPPORTED_FUSE_ALL_BATCHES if b <= 128)
                if _decode_max_tokens() >= 384:
                    fp8_batches = (*fp8_batches, 384)
                self._fused_decode_batches = fp8_batches
                self._fused_decode_sharded_op = None
                self._fused_decode_v2_op = None
                self._fused_decode_v2_sharded_op = None
                self._fused_decode_v2_batches = ()
                self._fused_decode_num_replicas = num_replicas
                self._fused_decode_mode = "fp8_sparse"
                self._fused_decode_block = 256
                self._fused_decode_replica_cost_mode = replica_cost_mode
                self._fused_decode_op = get_muse_fuse_all_decode_op(
                    mode="fp8_sparse",
                    ep=world,
                    tp=1,
                    block=256,
                    num_replicas=num_replicas,
                    replica_cost_mode=replica_cost_mode,
                )
                return
            self._prepare_fuse_all_fp8_decode(
                mode=mode,
                world=world,
                parallel=parallel,
                ex=ex,
                norm=norm,
                num_replicas=num_replicas,
                replica_cost_mode=replica_cost_mode,
            )
            return

        w1_chk = self._buffers["_decode_fp4_w1"] if has_shadow_fp4 else ex.w13_weight
        s1_chk = (
            self._buffers["_decode_fp4_s1"] if has_shadow_fp4 else ex.w13_weight_scale
        )
        w2_chk = self._buffers["_decode_fp4_w2"] if has_shadow_fp4 else ex.w2_weight
        s2_chk = (
            self._buffers["_decode_fp4_s2"] if has_shadow_fp4 else ex.w2_weight_scale
        )
        if (
            w1_chk.dtype != torch.float4_e2m1fn_x2
            or w2_chk.dtype != torch.float4_e2m1fn_x2
            or s1_chk.dtype != torch.bfloat16
            or s2_chk.dtype != torch.bfloat16
            or s1_chk.ndim != 3
            or s1_chk.shape[1] <= 0
            or 4096 % s1_chk.shape[1] != 0
        ):
            return

        block = 4096 // int(s1_chk.shape[1])
        if block not in (16, 256) and not (
            block == 512 and mode != "chipmate"
        ):
            return

        if mode == "ep_tp2":
            ep_owners = world // 2
            tp_experts = 2
            e_owner = 256 // ep_owners
            if (
                parallel.moe_ep_size != ep_owners
                or parallel.moe_tp_size != 2
                or tuple(w1_chk.shape) != (e_owner, 4096, 2048)
                or tuple(w2_chk.shape) != (e_owner, 2048, 2048)
                or tuple(s1_chk.shape) != (e_owner, 4096 // block, 4096)
                or tuple(s2_chk.shape) != (e_owner, 2048 // block, 4096)
            ):
                return
            rank_t = torch.tensor(
                [parallel.tp_rank],
                dtype=torch.int32,
                device=w1_chk.device,
            )
        else:
            ep_owners = world
            tp_experts = 1
            e_owner = 256 // ep_owners
            w_groups = int(w2_chk.shape[0])
            valid_groups = (
                (e_owner <= w_groups <= e_owner + 8)
                if allow_eplb
                else (w_groups == e_owner)
            )
            is_pm = tuple(w1_chk.shape) == (2 * w_groups, 4096, 2048) and tuple(
                s1_chk.shape
            ) == (2 * w_groups, 4096 // block, 4096)
            is_std_w1 = tuple(w1_chk.shape) == (w_groups, 4096, 4096) and tuple(
                s1_chk.shape
            ) == (w_groups, 4096 // block, 8192)
            if (
                parallel.moe_ep_size != world
                or parallel.moe_ep_rank != parallel.tp_rank
                or parallel.moe_tp_size != 1
                or not ex.handles_moe_reduction
                or not hasattr(ex, "_tpu_fused_rs_group_offset")
                or not valid_groups
                or (not is_std_w1 and not is_pm)
                or tuple(w2_chk.shape) != (w_groups, 4096, 2048)
                or tuple(s2_chk.shape) != (w_groups, 4096 // block, 4096)
            ):
                return
            rank_t = ex._tpu_fused_rs_group_offset // w_groups

        pre_gain = None
        if mode in ("full_moe", "full_stage_b"):
            pre_w = getattr(self.pre_expert_proj, "weight", None)
            gain_fn = getattr(self.pre_expert_norm, "gain", None)
            pre_gain = gain_fn() if callable(gain_fn) else None
            if (
                pre_w is None
                or tuple(pre_w.shape) != (4096 // world, 8192)
                or pre_w.dtype != torch.bfloat16
                or pre_gain is None
                or tuple(pre_gain.shape) != (4096,)
                or pre_gain.dtype != torch.float32
            ):
                return
        if mode in ("full_moe", "full_stage_a"):
            post_w = getattr(self.post_expert_proj, "weight", None)
            if (
                post_w is None
                or tuple(post_w.shape) != (8192 // world, 4096)
                or post_w.dtype != torch.bfloat16
            ):
                return

        if num_replicas > 0:
            bufs = self._buffers
            rep_w1 = bufs.get("_fused_decode_rep_w1")
            rep_s1 = bufs.get("_fused_decode_rep_s1")
            rep_w2 = bufs.get("_fused_decode_rep_w2")
            rep_s2 = bufs.get("_fused_decode_rep_s2")
            replica_meta = bufs.get("_fused_decode_replica_meta")
            if any(t is None for t in (rep_w1, rep_s1, rep_w2, rep_s2, replica_meta)):
                return
            exp_rep_w1_rows = (
                (2 * num_replicas)
                if (mode != "ep_tp2" and w1_chk.shape[0] == 2 * w2_chk.shape[0])
                else num_replicas
            )
            if (
                rep_w1.dtype != w1_chk.dtype
                or tuple(rep_w1.shape) != (exp_rep_w1_rows, *w1_chk.shape[1:])
                or rep_s1.dtype != s1_chk.dtype
                or tuple(rep_s1.shape) != (exp_rep_w1_rows, *s1_chk.shape[1:])
                or rep_w2.dtype != w2_chk.dtype
                or tuple(rep_w2.shape) != (num_replicas, *w2_chk.shape[1:])
                or rep_s2.dtype != s2_chk.dtype
                or tuple(rep_s2.shape) != (num_replicas, *s2_chk.shape[1:])
                or replica_meta.dtype != torch.int32
                or tuple(replica_meta.shape) != (4, ep_owners * num_replicas)
            ):
                return

        from sglang_tpu.fp8_moe import _own_storage
        import sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_op as _fdo_mod

        SUPPORTED_FUSE_ALL_BATCHES = _fdo_mod.SUPPORTED_FUSE_ALL_BATCHES
        get_muse_ep_tp2_fallback_op = _fdo_mod.get_muse_ep_tp2_fallback_op
        get_muse_fuse_all_decode_op = _fdo_mod.get_muse_fuse_all_decode_op
        _batches_fn = getattr(_fdo_mod, "supported_fuse_all_batches", None)

        self._register_fused_decode_vecs(norm)
        self.register_buffer("_fused_decode_rank", rank_t, persistent=False)
        if pre_gain is not None:
            self.register_buffer(
                "_fused_decode_pre_gain",
                _own_storage(pre_gain.detach().reshape(1, -1).contiguous()),
                persistent=False,
            )
        base_batches = (
            _batches_fn(ep=ep_owners)
            if (mode == "sparse" and _batches_fn is not None)
            else SUPPORTED_FUSE_ALL_BATCHES
        )
        ext_batches = (
            (*base_batches, 384)
            if (_decode_max_tokens() >= 384 and 384 not in base_batches)
            else base_batches
        )
        self._fused_decode_batches = (
            ext_batches
            if (is_full_moe or mode in ("sparse", "sparse_v2"))
            else SUPPORTED_FUSE_ALL_BATCHES
        )
        self._fused_decode_num_replicas = num_replicas
        self._fused_decode_mode = mode
        self._fused_decode_block = block
        self._fused_decode_replica_cost_mode = replica_cost_mode
        self._fused_decode_sharded_op = None
        self._fused_decode_v2_op = None
        self._fused_decode_v2_sharded_op = None
        self._fused_decode_v2_batches = ()
        if is_full_moe:
            from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_op import (
                get_muse_full_moe_decode_op,
            )

            use_kmajor = bool(
                _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_FULL_MOE_KMAJOR", True)
            )
            self._full_moe_use_kmajor = use_kmajor
            self._full_moe_max_bs = int(
                _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_FULL_MOE_MAX_BS", 128)
            )
            if use_kmajor:
                if mode in ("full_moe", "full_stage_b"):
                    self.register_buffer(
                        "_fused_gate_w_km",
                        self.gate.weight.detach().t().contiguous(),
                        persistent=False,
                    )
                    self.register_buffer(
                        "_fused_w_pre_km",
                        self.pre_expert_proj.weight.detach().t().contiguous(),
                        persistent=False,
                    )
                if mode in ("full_moe", "full_stage_a"):
                    self.register_buffer(
                        "_fused_w_post_km",
                        self.post_expert_proj.weight.detach().t().contiguous(),
                        persistent=False,
                    )

            full_submode = "full" if mode == "full_moe" else mode.removeprefix("full_")
            pre_eps = float(getattr(self.pre_expert_norm, "eps", 1e-5))
            self._fused_decode_op = get_muse_full_moe_decode_op(
                full_mode=full_submode,
                ep=ep_owners,
                block=block,
                num_replicas=num_replicas,
                replica_cost_mode=replica_cost_mode,
                router_multiplier=float(self.router_multiplier),
                pre_norm_eps=pre_eps,
            )
            hi_mode = (
                "sparse_v2"
                if (block == 16 and self.__dict__.get("_decode_sparse_v2", False))
                else "sparse"
            )
            self._fused_sparse_hi_op = get_muse_fuse_all_decode_op(
                mode=hi_mode,
                ep=ep_owners,
                tp=1,
                block=block,
                num_replicas=num_replicas,
                replica_cost_mode=replica_cost_mode,
                chipmate_xor=None,
            )
        else:
            has_w3_mask = "_fused_decode_w3_mask" in self._buffers
            force_v2_only = mode == "sparse" and (
                has_w3_mask or (w1_chk.shape[0] == 2 * w2_chk.shape[0])
            )
            self._fused_decode_v2_has_w3 = has_w3_mask
            if force_v2_only:
                self._fused_decode_mode = "sparse_v2"
                v2_only_op = get_muse_fuse_all_decode_op(
                    mode="sparse_v2",
                    ep=ep_owners,
                    tp=1,
                    block=block,
                    num_replicas=num_replicas,
                    replica_cost_mode=replica_cost_mode,
                    has_w3_mask=has_w3_mask,
                )
                self._fused_decode_op = v2_only_op
                self._fused_decode_v2_op = v2_only_op
                self._fused_decode_v2_batches = ext_batches
                if self.__dict__.get("_tpu_v2_extend_fallback", False):
                    # EXTEND through the token-sharded op (the decode-proven path);
                    # the replicated B=256 op hit e0200 semaphore halts on hardware.
                    self._fused_decode_v2_ext_sh_op = get_muse_fuse_all_decode_op(
                        mode="sparse_v2",
                        ep=ep_owners,
                        tp=1,
                        block=block,
                        num_replicas=num_replicas,
                        replica_cost_mode=replica_cost_mode,
                        return_sharded=True,
                        has_w3_mask=has_w3_mask,
                    )
            else:
                has_spec_slot = (
                    mode == "sparse"
                    and "_fused_decode_spec_slot" in self._buffers
                )
                self._fused_decode_has_spec_slot = has_spec_slot
                has_free_params = False
                if mode == "sparse" and tp_experts == 1:
                    has_free_params = self._prepare_nonuniform_free_buffers(
                        ep_owners=ep_owners,
                        batches=ext_batches,
                        device=w1_chk.device,
                    )
                self._fused_decode_has_free_params = has_free_params
                self._fused_decode_op = get_muse_fuse_all_decode_op(
                    mode=mode,
                    ep=ep_owners,
                    tp=tp_experts,
                    block=block,
                    num_replicas=num_replicas,
                    replica_cost_mode=replica_cost_mode,
                    chipmate_xor=chipmate_xor,
                    has_spec_slot=has_spec_slot,
                    has_free_params=has_free_params,
                )
                if mode == "sparse" and self.__dict__.get("_decode_sparse_v2", False):
                    v2_b_env = str(
                        _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_V2_BATCHES", "")
                    ).strip()
                    if v2_b_env.lower() == "all":
                        v2_b = ext_batches
                    elif v2_b_env:
                        v2_b = tuple(
                            int(x.strip()) for x in v2_b_env.split(",") if x.strip()
                        )
                    else:
                        k16_v2_b = (
                            (32, 64, 128, 256, 384)
                            if _decode_max_tokens() >= 384
                            else (32, 64, 128, 256)
                        )
                        v2_b = k16_v2_b if block == 16 else (64,)
                    self._fused_decode_v2_batches = v2_b
                    self._fused_decode_v2_op = get_muse_fuse_all_decode_op(
                        mode="sparse_v2",
                        ep=ep_owners,
                        tp=1,
                        block=block,
                        num_replicas=num_replicas,
                        replica_cost_mode=replica_cost_mode,
                        has_w3_mask=has_w3_mask,
                    )
        if mode == "ep_tp2":
            self._ep_tp2_fallback_op = get_muse_ep_tp2_fallback_op(
                ep=ep_owners,
                tp=tp_experts,
                block=block,
                chipmate_xor=chipmate_xor,
            )

    def _prepare_fuse_all_fp8_decode(
        self,
        *,
        mode: str,
        world: int,
        parallel,
        ex,
        norm: torch.Tensor,
        num_replicas: int = 0,
        replica_cost_mode: str,
    ) -> None:
        """Bind opt-in L0/L61 channelwise FP8 fuse-all decode op (gated to B<=128)."""
        if (
            mode == "ep_tp2"
            or not self.__dict__.get("_decode_sparse_fp8", False)
            or parallel.moe_ep_size != world
            or parallel.moe_ep_rank != parallel.tp_rank
            or parallel.moe_tp_size != 1
            or not ex.handles_moe_reduction
            or not hasattr(ex, "_tpu_fused_rs_group_offset")
            or ex.w13_weight.dtype != torch.float8_e4m3fn
            or ex.w2_weight.dtype != torch.float8_e4m3fn
        ):
            return
        e_owner = 256 // world
        w_groups = int(ex.w13_weight.shape[0])
        allow_eplb = bool(
            self.__dict__.get("_decode_allow_prefill_eplb", False)
            or _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_ALLOW_PREFILL_EPLB", False)
        )
        valid_groups = (
            (e_owner <= w_groups <= e_owner + 8)
            if allow_eplb
            else (w_groups == e_owner)
        )
        if (
            not valid_groups
            or tuple(ex.w13_weight.shape) != (w_groups, 4096, 8192)
            or tuple(ex.w2_weight.shape) != (w_groups, 4096, 4096)
            or tuple(ex.w13_weight_scale.shape) != (w_groups, 1, 1, 8192)
            or tuple(ex.w2_weight_scale.shape) != (w_groups, 1, 1, 4096)
        ):
            return
        eff_replicas = num_replicas
        if eff_replicas > 0:
            bufs = self._buffers
            rep_w1 = bufs.get("_fused_decode_rep_w1")
            rep_s1 = bufs.get("_fused_decode_rep_s1")
            rep_w2 = bufs.get("_fused_decode_rep_w2")
            rep_s2 = bufs.get("_fused_decode_rep_s2")
            replica_meta = bufs.get("_fused_decode_replica_meta")
            if (
                any(t is None for t in (rep_w1, rep_s1, rep_w2, rep_s2, replica_meta))
                or rep_w1.dtype != torch.float8_e4m3fn
                or tuple(rep_w1.shape) != (eff_replicas, 4096, 8192)
                or tuple(rep_w2.shape) != (eff_replicas, 4096, 4096)
                or tuple(replica_meta.shape) != (4, world * eff_replicas)
            ):
                eff_replicas = 0

        from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_op import (
            SUPPORTED_FUSE_ALL_FP8_BATCHES,
            get_muse_fuse_all_decode_op,
        )

        self._register_fused_decode_vecs(norm)
        self.register_buffer(
            "_fused_decode_rank",
            ex._tpu_fused_rs_group_offset // w_groups,
            persistent=False,
        )
        self._fused_decode_batches = SUPPORTED_FUSE_ALL_FP8_BATCHES
        self._fused_decode_sharded_op = None
        self._fused_decode_v2_op = None
        self._fused_decode_v2_sharded_op = None
        self._fused_decode_v2_batches = ()
        self._fused_decode_num_replicas = eff_replicas
        self._fused_decode_mode = "sparse_fp8"
        self._fused_decode_block = 256
        self._fused_decode_replica_cost_mode = replica_cost_mode
        self._fused_decode_op = get_muse_fuse_all_decode_op(
            mode="sparse_fp8",
            ep=world,
            tp=1,
            num_replicas=eff_replicas,
            replica_cost_mode=replica_cost_mode,
        )

    def enable_sharded_moe_out(self) -> bool:
        """Eagerly prewarm token-sharded ``[B // tp_size, 4096]`` fused MoE decode ops."""
        fmode = self.__dict__.get("_fused_decode_mode", "")
        if self._fused_decode_op is None or fmode not in (
            "sparse",
            "sparse_v2",
            "fp8_sparse",
            "sparse_fp8",
            "ep_tp2",
        ):
            return False
        if self.__dict__.get("_fused_decode_sharded_op") is not None:
            return True
        from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_op import (
            get_muse_fuse_all_decode_op,
        )

        num_rep = int(self.__dict__.get("_fused_decode_num_replicas", 0))
        cost_m = str(self.__dict__.get("_fused_decode_replica_cost_mode", "tiles"))
        blk = int(self.__dict__.get("_fused_decode_block", 256))
        has_w3 = bool(self.__dict__.get("_fused_decode_v2_has_w3", False))
        has_spec_slot = (
            fmode == "sparse"
            and bool(self.__dict__.get("_fused_decode_has_spec_slot", False))
        )
        has_free_params = (
            fmode == "sparse"
            and bool(self.__dict__.get("_fused_decode_has_free_params", False))
        )
        self._fused_decode_sharded_op = get_muse_fuse_all_decode_op(
            mode=fmode,
            ep=(self.tp_size // 2 if fmode == "ep_tp2" else self.tp_size),
            tp=(2 if fmode == "ep_tp2" else 1),
            block=blk,
            num_replicas=num_rep,
            replica_cost_mode=cost_m,
            return_sharded=True,
            has_w3_mask=(has_w3 and fmode == "sparse_v2"),
            has_spec_slot=has_spec_slot,
            has_free_params=has_free_params,
        )
        if self.__dict__.get("_fused_decode_v2_op") is not None:
            self._fused_decode_v2_sharded_op = get_muse_fuse_all_decode_op(
                mode="sparse_v2",
                ep=self.tp_size,
                tp=1,
                block=blk,
                num_replicas=num_rep,
                replica_cost_mode=cost_m,
                return_sharded=True,
                has_w3_mask=has_w3,
            )
        return True

    def can_return_sharded_moe_out(self, n_tok: int, *, decode: bool) -> bool:
        """Return True when ``forward(..., return_sharded_moe_out=True)`` emits ``[B // tp_size, 4096]``."""
        if (
            self.__dict__.get("_fused_decode_sharded_op") is None
            or not decode
            or n_tok < self.tp_size
            or n_tok % self.tp_size != 0
            or n_tok not in self.__dict__.get("_fused_decode_batches", ())
        ):
            return False
        allow_eplb = bool(
            self.__dict__.get("_decode_allow_prefill_eplb", False)
            or _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_ALLOW_PREFILL_EPLB", False)
        )
        if self.__dict__.get("_eplb_num_replicas", 0) and not allow_eplb:
            return False
        from sglang_tpu.fused_rs_sequence_parallel import (
            get_fused_moe_sequence_parallel_plan,
        )

        return not any(get_fused_moe_sequence_parallel_plan())

    def _route(self, router_input: torch.Tensor) -> StandardTopKOutput:
        router_logits, _ = self.gate(router_input)
        out = self._route_scaled_logits(router_logits.float() * self.router_multiplier)
        return StandardTopKOutput(
            topk_weights=out.topk_weights,
            topk_ids=out.topk_ids,
            router_logits=router_logits,
        )

    def _combine_slots(
        self, slot_outputs: torch.Tensor, topk_weights: torch.Tensor
    ) -> torch.Tensor:
        # slot_outputs: [num_tokens, top_k, expert_hidden]; non-local slots are 0,
        # and a zero row stays zero through the norm.
        dtype = slot_outputs.dtype
        if self.moe_tp_size > 1:
            # Expert TP: each rank holds a partial output, so sum before the norm,
            # in fp32 like the fused-RS kernel. That is 2 * top_k times the bytes of
            # summing the combined output, acceptable off the fused-RS path.
            slot_outputs = tensor_model_parallel_all_reduce(slot_outputs.float())
        normed = self.post_expert_norm(slot_outputs)
        combined = (normed * topk_weights.unsqueeze(-1)).to(dtype).sum(dim=1)
        if self.tp_size > 1 and self.moe_tp_size == 1:
            combined = tensor_model_parallel_all_reduce(combined)
        return combined

    def _gather_columns(
        self, shard: torch.Tensor, *, transpose_2d: bool = True
    ) -> torch.Tensor:
        """[T, N/tp] column shard -> [T, N] (or raw [tp * T, N/tp] when transpose_2d=False)."""
        max_bytes = 256 * 1024 if self.moe_tp_size > 1 else 128 * 1024
        use_os = self.oneshot_gather and (
            shard.numel() * shard.element_size() <= max_bytes
        )
        if not transpose_2d:
            if use_os:
                from sglang_tpu.oneshot_collectives import oneshot_all_gather

                return oneshot_all_gather(
                    shard.contiguous(),
                    rank_input=self._tp_rank_t,
                    world_size=self.tp_size,
                )
            return tensor_model_parallel_all_gather(shard.contiguous(), dim=0)
        if not use_os:
            # Prefill-sized or >=256 KiB EP8 shards: XLA's bandwidth-oriented all-gather.
            return tensor_model_parallel_all_gather(shard.contiguous())
        return _oneshot_gather_columns(
            shard, rank_t=self._tp_rank_t, world_size=self.tp_size
        )

    @torch.no_grad()
    def prepare_w7_moe_bridges(
        self,
        *,
        do_pre: bool = True,
        do_post: bool = True,
        router_bf16: bool = True,
        free_bf16: bool = False,
        tok_sharded: bool = False,
        chip_sharded: bool = False,
        tp2_wpost: bool = False,
    ) -> Tuple[bool, bool]:
        """Prepare K-major FP8 weights for Wave-7/Wave-8/Wave-9 decode collective bridges 1 & 2."""
        from sglang_tpu.kernels.decode_collective_bridges import (
            quantize_weight_fp8_k_major,
        )

        can_pair_ag = (
            ((tok_sharded and self.tp_size == 8) or (chip_sharded and self.tp_size in (8, 16)))
            and torch.distributed.is_initialized()
        )
        side = get_parallel().tp_rank & 1 if can_pair_ag else 0

        pre_ok = False
        if do_pre:
            wp_mod = self.pre_expert_proj
            gt_mod = self.gate
            if (
                wp_mod.bias is None
                and gt_mod.bias is None
                and wp_mod.weight.ndim == 2
                and gt_mod.weight.ndim == 2
                and wp_mod.weight.dtype == torch.bfloat16
                and wp_mod.weight.shape[0] > 0
                and wp_mod.weight.shape[0] % 256 == 0
                and wp_mod.weight.shape[1] % 128 == 0
                and gt_mod.weight.shape[0] % 256 == 0
                and gt_mod.weight.shape[1] == wp_mod.weight.shape[1]
            ):
                wpre_q, wpre_s_2d = quantize_weight_fp8_k_major(wp_mod.weight)
                wpre_s = _as_1d_fp8_scale(wpre_s_2d)
                r_dt = torch.bfloat16 if router_bf16 else torch.float32
                wrouter_km = gt_mod.weight.detach().to(r_dt).t().contiguous()
                _wait_for(wpre_q, wpre_s_2d, wpre_s, wrouter_km)
                self.register_buffer("_w7_wpre_w8k", wpre_q, persistent=False)
                self.register_buffer("_w7_spre", wpre_s, persistent=False)
                self.register_buffer("_w7_wrouter_km", wrouter_km, persistent=False)
                if can_pair_ag:
                    full_wpre = tensor_model_parallel_all_gather(wpre_q.contiguous())
                    full_spre = tensor_model_parallel_all_gather(wpre_s_2d.contiguous())
                    half_pre = full_wpre.shape[1] // 2
                    pair_wpre = full_wpre[
                        :, side * half_pre : (side + 1) * half_pre
                    ].contiguous()
                    pair_spre = _as_1d_fp8_scale(
                        full_spre[:, side * half_pre : (side + 1) * half_pre]
                    )
                    _wait_for(pair_wpre, pair_spre)
                    del full_wpre, full_spre
                    self.register_buffer(
                        "_w8_wpre_pair_w8k", pair_wpre, persistent=False
                    )
                    self.register_buffer("_w8_spre_pair", pair_spre, persistent=False)
                if free_bf16 and wp_mod.weight.device.type != "meta":
                    old_p = wp_mod.weight.data
                    wp_mod.weight = nn.Parameter(
                        torch.empty(
                            (0, old_p.shape[1]), dtype=old_p.dtype, device=old_p.device
                        ),
                        requires_grad=False,
                    )
                    _free_replaced_storage(old_p, wp_mod.weight.data)
                pre_ok = True

        post_ok = False
        if do_post:
            wpo_mod = self.post_expert_proj
            if (
                wpo_mod.bias is None
                and wpo_mod.weight.ndim == 2
                and wpo_mod.weight.dtype == torch.bfloat16
                and wpo_mod.weight.shape[0] > 0
                and wpo_mod.weight.shape[0] % 256 == 0
                and wpo_mod.weight.shape[1] % 128 == 0
            ):
                wpost_q, wpost_s_2d = quantize_weight_fp8_k_major(wpo_mod.weight)
                wpost_s = _as_1d_fp8_scale(wpost_s_2d)
                _wait_for(wpost_q, wpost_s_2d, wpost_s)
                self.register_buffer("_w7_wpost_w8k", wpost_q, persistent=False)
                self.register_buffer("_w7_spost", wpost_s, persistent=False)
                if tp2_wpost and can_pair_ag and self.tp_size in (8, 16):
                    full_wpost = tensor_model_parallel_all_gather(wpost_q.contiguous())
                    full_spost = tensor_model_parallel_all_gather(
                        wpost_s_2d.contiguous()
                    )
                    half_post = full_wpost.shape[1] // 2
                    pair_wpost = full_wpost[
                        :, side * half_post : (side + 1) * half_post
                    ].contiguous()
                    pair_spost = _as_1d_fp8_scale(
                        full_spost[:, side * half_post : (side + 1) * half_post]
                    )
                    _wait_for(pair_wpost, pair_spost)
                    del full_wpost, full_spost
                    self.register_buffer(
                        "_w9_wpost_pair_w8k", pair_wpost, persistent=False
                    )
                    self.register_buffer("_w9_spost_pair", pair_spost, persistent=False)
                if free_bf16 and wpo_mod.weight.device.type != "meta":
                    old_po = wpo_mod.weight.data
                    wpo_mod.weight = nn.Parameter(
                        torch.empty(
                            (0, old_po.shape[1]),
                            dtype=old_po.dtype,
                            device=old_po.device,
                        ),
                        requires_grad=False,
                    )
                    _free_replaced_storage(old_po, wpo_mod.weight.data)
                post_ok = True
        return pre_ok, post_ok

    def _prepare_nonuniform_free_buffers(
        self,
        *,
        ep_owners: int,
        batches: Tuple[int, ...],
        device: torch.device,
    ) -> bool:
        """Precompute zero-compile per-bucket ``(4,) float32`` FREE params when opted in."""
        from sglang_tpu.fp8_moe import _own_storage
        from sglang_tpu.moe.pallas.fused_rs_muse.decode_hbm_breakers import (
            parse_nonuniform_free_spec,
            read_d1_large_batch_free,
            resolve_nonuniform_free_for_layer,
        )
        from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_routing import (
            parse_free_expert_rows_spec,
        )
        from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse import (
            _read_sparse_env_overrides,
        )

        raw_spec = str(
            _env_get("SGLANG_TPU_MUSE_DECODE_NONUNIFORM_FREE", "")
        ).strip()
        if parse_nonuniform_free_spec(raw_spec) is None:
            return False

        d1_large_batch = read_d1_large_batch_free()
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            base_min_p,
            base_free_rows,
            base_free_max_w,
            base_free_min_slot,
            _,
            _,
        ) = _read_sparse_env_overrides()
        for b_tok in batches:
            mp, f_rows, f_max_w, f_min_slot = resolve_nonuniform_free_for_layer(
                self.layer_idx,
                self.num_layers,
                int(b_tok),
                min_p_topk=base_min_p,
                free_rows=base_free_rows,
                free_max_w=base_free_max_w,
                free_min_slot=base_free_min_slot,
                parse_free_spec_fn=parse_free_expert_rows_spec,
                ep=ep_owners,
                spec=raw_spec,
                d1_large_batch=d1_large_batch,
            )
            vec = torch.tensor(
                [float(mp), float(f_rows), float(f_max_w), float(f_min_slot)],
                dtype=torch.float32,
            )
            buf = vec if device.type == "cpu" else _own_storage(vec.to(device=device, copy=True))
            self.register_buffer(
                f"_fused_decode_free_params_b{int(b_tok)}",
                buf,
                persistent=False,
            )
        return True

    def _free_params_for_batch(self, n_tok: int) -> Optional[torch.Tensor]:
        if not bool(self.__dict__.get("_fused_decode_has_free_params", False)):
            return None
        return self._buffers.get(f"_fused_decode_free_params_b{int(n_tok)}")

    def _register_fused_decode_vecs(self, norm: torch.Tensor) -> None:
        """Register storage-owned 1D and 2D router-bias / post-expert-norm buffers.

        Eliminates 124 per-step ``.unsqueeze(0)`` ``execute_as_strided`` ops across 62 layers
        and allows ``sparse`` / ``sparse_v2`` / ``fp8_sparse`` / ``sparse_fp8`` decode ops to
        consume 1D ``{0:T(1024)}`` vectors directly without ``{1,0:T(8,128)}`` HLO copies.
        """
        from sglang_tpu.fp8_moe import _own_storage

        bias_d = self.e_score_correction_bias.data
        if (
            bias_d.device.type != "cpu"
            or bias_d._base is not None
            or bias_d.storage_offset() != 0
            or bias_d.untyped_storage().nbytes()
            != bias_d.numel() * bias_d.element_size()
        ):
            fresh_b = bias_d.detach().reshape(-1).contiguous()
            self.e_score_correction_bias.data = (
                fresh_b.clone()
                if fresh_b.device.type == "cpu"
                else _own_storage(fresh_b)
            )
        fresh_n1 = norm.detach().reshape(-1).contiguous()
        norm_1d = (
            fresh_n1.clone()
            if fresh_n1.device.type == "cpu"
            else _own_storage(fresh_n1)
        )
        fresh_b2 = self.e_score_correction_bias.detach().reshape(1, -1).contiguous()
        bias_2d = (
            fresh_b2.clone()
            if fresh_b2.device.type == "cpu"
            else _own_storage(fresh_b2)
        )
        fresh_n2 = norm.detach().reshape(1, -1).contiguous()
        norm_2d = (
            fresh_n2.clone()
            if fresh_n2.device.type == "cpu"
            else _own_storage(fresh_n2)
        )
        self.register_buffer("_fused_decode_norm", norm_1d, persistent=False)
        self.register_buffer("_fused_decode_bias_2d", bias_2d, persistent=False)
        self.register_buffer("_fused_decode_norm_2d", norm_2d, persistent=False)

    def _bias_2d(self) -> torch.Tensor:
        buf = self._buffers.get("_fused_decode_bias_2d")
        if buf is not None:
            return buf
        return self.e_score_correction_bias.unsqueeze(0)

    def _norm_2d(self) -> torch.Tensor:
        buf = self._buffers.get("_fused_decode_norm_2d")
        if buf is not None:
            return buf
        return self._fused_decode_norm.unsqueeze(0)

    def _use_1d_decode_vecs(self, fmode: str) -> bool:
        if fmode not in ("sparse", "sparse_v2", "fp8_sparse", "sparse_fp8"):
            return False
        return os.environ.get("SGLANG_TPU_MUSE_DECODE_1D_MOE_VECS", "1").strip() not in (
            "0",
            "false",
            "False",
        )

    def fuses_pre_gather(self, n_tok: int, *, decode: bool) -> bool:
        """Return True when ``forward`` fuses ``gate + pre_expert_proj + AG`` in-kernel."""
        max_bs = int(self.__dict__.get("_full_moe_max_bs", 128))
        if (
            self._fused_decode_op is None
            or not decode
            or self.__dict__.get("_fused_decode_mode", "")
            not in ("full_moe", "full_stage_b")
            or n_tok not in self.__dict__.get("_fused_decode_batches", (16, 32))
            or n_tok > max_bs
        ):
            return False
        allow_eplb = bool(
            self.__dict__.get("_decode_allow_prefill_eplb", False)
            or _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_ALLOW_PREFILL_EPLB", False)
        )
        if self.__dict__.get("_eplb_num_replicas", 0) and not allow_eplb:
            return False
        from sglang_tpu.fused_rs_sequence_parallel import (
            get_fused_moe_sequence_parallel_plan,
        )

        return not any(get_fused_moe_sequence_parallel_plan())

    def _route_scaled_logits(
        self, scaled_logits: torch.Tensor, *, prerouted: bool = False
    ) -> StandardTopKOutput:
        """Compute Top-K routing directly from pre-scaled ``router_logits * router_multiplier``."""
        eplb_replicas = int(self.__dict__.get("_eplb_num_replicas", 0))

        def _map_eplb_fallback(ids_i32: torch.Tensor) -> torch.Tensor:
            if eplb_replicas <= 0:
                return ids_i32
            from sglang_tpu.muse_prefill_kernels.eplb import (
                map_torch_fallback_topk_ids,
            )

            ep_size = int(self.__dict__.get("_eplb_ep_size", 8))
            return map_torch_fallback_topk_ids(
                ids_i32,
                num_replicas=eplb_replicas,
                ep_size=ep_size,
                num_base_local=self.num_experts // ep_size,
            )

        if prerouted:
            topk_weights = scaled_logits[:, : self.top_k].contiguous()
            topk_ids_i32 = _map_eplb_fallback(
                scaled_logits[:, 8 : 8 + self.top_k].to(torch.int32)
            )
            return StandardTopKOutput(
                topk_weights=topk_weights,
                topk_ids=topk_ids_i32,
                router_logits=scaled_logits,
            )
        if (
            self._decode_fused_router
            and 0 <= eplb_replicas <= 8
            and 0 < scaled_logits.shape[0] <= _decode_max_tokens()
            and (scaled_logits.shape[0] == 1 or scaled_logits.shape[0] % 8 == 0)
        ):
            from sglang_tpu.moe.pallas.fused_rs_muse.decode_norm_router import (
                get_muse_decode_router_topk_op,
            )

            router_op = (
                get_muse_decode_router_topk_op()
                if eplb_replicas == 0
                else get_muse_decode_router_topk_op(
                    eplb_replicas, int(self.__dict__.get("_eplb_ep_size", 8))
                )
            )
            topk_weights, topk_ids_i32 = router_op(
                scaled_logits.float(),
                self.e_score_correction_bias,
                self.top_k,
                float(self.route_eps),
                float(self.route_scale),
            )
            return StandardTopKOutput(
                topk_weights=topk_weights,
                topk_ids=topk_ids_i32,
                router_logits=scaled_logits,
            )
        scores = torch.sigmoid(scaled_logits.float())
        _, topk_ids = torch.topk(
            scores + self.e_score_correction_bias, self.top_k, dim=-1
        )
        expert_ids = torch.arange(scores.shape[-1], device=scores.device)
        topk_scores = torch.where(
            topk_ids.unsqueeze(-1) == expert_ids, scores.unsqueeze(1), 0.0
        ).sum(dim=-1)
        topk_weights = (
            topk_scores
            / (topk_scores.sum(dim=-1, keepdim=True) + self.route_eps)
            * self.route_scale
        )
        topk_ids_i32 = _map_eplb_fallback(topk_ids.to(torch.int32))
        return StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids_i32,
            router_logits=scaled_logits,
        )

    def fuses_post_gather(self, n_tok: int, *, decode: bool) -> bool:
        """Return True when ``forward`` returns a gathered ``[B, 8192]`` tensor in-kernel."""
        max_bs = int(self.__dict__.get("_full_moe_max_bs", 128))
        if (
            self._fused_decode_op is None
            or not decode
            or self.__dict__.get("_fused_decode_mode", "")
            not in ("full_moe", "full_stage_a")
            or n_tok not in self.__dict__.get("_fused_decode_batches", (16, 32))
            or n_tok > max_bs
        ):
            return False
        allow_eplb = bool(
            self.__dict__.get("_decode_allow_prefill_eplb", False)
            or _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_ALLOW_PREFILL_EPLB", False)
        )
        if self.__dict__.get("_eplb_num_replicas", 0) and not allow_eplb:
            return False
        from sglang_tpu.fused_rs_sequence_parallel import (
            get_fused_moe_sequence_parallel_plan,
        )

        return not any(get_fused_moe_sequence_parallel_plan())

    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        router_input: Optional[torch.Tensor],
        return_untransposed_gather: bool = False,
        decode: bool = False,
        precomputed_moe_inputs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_unprojected_moe_out: bool = False,
        return_sharded_moe_out: bool = False,
        prerouted_router_topk: bool = False,
    ) -> torch.Tensor:
        use_fused = False
        batches = self.__dict__.get("_fused_decode_batches", (16, 32))
        allow_eplb = bool(
            self.__dict__.get("_decode_allow_prefill_eplb", False)
            or _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_ALLOW_PREFILL_EPLB", False)
        )
        if precomputed_moe_inputs is not None:
            scaled_logits_in, pre_expert_in = precomputed_moe_inputs
            n_tok_in = pre_expert_in.shape[0]
            can_dtype = (
                scaled_logits_in.dtype == torch.float32
                and pre_expert_in.dtype == torch.bfloat16
            )
        else:
            assert router_input is not None and hidden_states is not None
            n_tok_in = router_input.shape[0]
            can_dtype = (
                router_input.dtype == torch.float32
                and hidden_states.dtype == torch.bfloat16
            )
        if (
            self._fused_decode_op is not None
            and decode
            and n_tok_in in batches
            and can_dtype
            and (
                not self.__dict__.get("_eplb_num_replicas", 0)
                or (allow_eplb and self.__dict__.get("_fused_decode_mode") != "ep_tp2")
            )
        ):
            from sglang_tpu.fused_rs_sequence_parallel import (
                get_fused_moe_sequence_parallel_plan,
            )

            use_fused = not any(get_fused_moe_sequence_parallel_plan())

        fmode = self.__dict__.get("_fused_decode_mode", "") if use_fused else ""
        active_op = self._fused_decode_op
        if (
            use_fused
            and fmode in ("full_moe", "full_stage_a", "full_stage_b")
            and n_tok_in > int(self.__dict__.get("_full_moe_max_bs", 128))
        ):
            hi_op = self.__dict__.get("_fused_sparse_hi_op")
            if hi_op is not None:
                fmode = "sparse"
                active_op = hi_op
            else:
                use_fused = False
                fmode = ""

        use_km = bool(self.__dict__.get("_full_moe_use_kmajor", False))
        if (
            precomputed_moe_inputs is None
            and not return_unprojected_moe_out
            and not return_sharded_moe_out
            and use_fused
            and fmode in ("full_moe", "full_stage_b")
        ):
            ex = self.experts
            w1_arg = self._buffers.get("_decode_fp4_w1", ex.w13_weight)
            s1_arg = self._buffers.get("_decode_fp4_s1", ex.w13_weight_scale)
            w2_arg = self._buffers.get("_decode_fp4_w2", ex.w2_weight)
            s2_arg = self._buffers.get("_decode_fp4_s2", ex.w2_weight_scale)
            gate_w_arg = self._fused_gate_w_km if use_km else self.gate.weight
            w_pre_arg = self._fused_w_pre_km if use_km else self.pre_expert_proj.weight
            rep_args = (
                (
                    self._fused_decode_rep_w1,
                    self._fused_decode_rep_s1,
                    self._fused_decode_rep_w2,
                    self._fused_decode_rep_s2,
                    self._fused_decode_replica_meta,
                )
                if self.__dict__.get("_fused_decode_num_replicas", 0) > 0
                else ()
            )
            if fmode == "full_moe":
                w_post_arg = (
                    self._fused_w_post_km if use_km else self.post_expert_proj.weight
                )
                out_full = active_op(
                    self._fused_decode_rank,
                    hidden_states,
                    router_input,
                    gate_w_arg,
                    self._bias_2d(),
                    w_pre_arg,
                    self._fused_decode_pre_gain,
                    w1_arg,
                    s1_arg,
                    w2_arg,
                    s2_arg,
                    self._norm_2d(),
                    w_post_arg,
                    *rep_args,
                )
                if return_untransposed_gather and self.tp_size > 1:
                    b_tok = out_full.shape[0]
                    cols = out_full.shape[1] // self.tp_size
                    return (
                        out_full.view(b_tok, self.tp_size, cols)
                        .transpose(0, 1)
                        .reshape(self.tp_size * b_tok, cols)
                    )
                return out_full

            output = active_op(
                self._fused_decode_rank,
                hidden_states,
                router_input,
                gate_w_arg,
                self._bias_2d(),
                w_pre_arg,
                self._fused_decode_pre_gain,
                w1_arg,
                s1_arg,
                w2_arg,
                s2_arg,
                self._norm_2d(),
                *rep_args,
            )
            output, _ = self.post_expert_proj(output)
            if self._manual_post_gather:
                output = self._gather_columns(
                    output, transpose_2d=not return_untransposed_gather
                )
            return output

        if precomputed_moe_inputs is not None:
            scaled_router_logits, expert_input = precomputed_moe_inputs
            if not use_fused:
                topk_output = self._route_scaled_logits(
                    scaled_router_logits, prerouted=prerouted_router_topk
                )
        else:
            assert router_input is not None and hidden_states is not None
            if use_fused:
                source_logits, _ = self.gate(router_input)
                scaled_router_logits = source_logits.float() * self.router_multiplier
            else:
                topk_output = self._route(router_input)
            w7_wpre = (
                self._buffers.get("_w7_wpre_w8k") if hasattr(self, "_buffers") else None
            )
            if w7_wpre is not None and self.pre_expert_proj.weight.shape[0] == 0:
                expert_input = (
                    torch.matmul(hidden_states, w7_wpre.to(hidden_states.dtype)).float()
                    * self._buffers["_w7_spre"]
                ).to(hidden_states.dtype)
                if (
                    getattr(self.pre_expert_proj, "gather_output", False)
                    and self.tp_size > 1
                ):
                    expert_input = tensor_model_parallel_all_gather(
                        expert_input.contiguous()
                    )
            else:
                expert_input, _ = self.pre_expert_proj(hidden_states)
            if self.oneshot_gather:
                expert_input = self._gather_columns(expert_input)
            expert_input = self.pre_expert_norm(expert_input)
        if (
            not return_unprojected_moe_out
            and not return_sharded_moe_out
            and use_fused
            and fmode == "full_stage_a"
        ):
            ex = self.experts
            w1_arg = self._buffers.get("_decode_fp4_w1", ex.w13_weight)
            s1_arg = self._buffers.get("_decode_fp4_s1", ex.w13_weight_scale)
            w2_arg = self._buffers.get("_decode_fp4_w2", ex.w2_weight)
            s2_arg = self._buffers.get("_decode_fp4_s2", ex.w2_weight_scale)
            w_post_arg = (
                self._fused_w_post_km if use_km else self.post_expert_proj.weight
            )
            rep_args = (
                (
                    self._fused_decode_rep_w1,
                    self._fused_decode_rep_s1,
                    self._fused_decode_rep_w2,
                    self._fused_decode_rep_s2,
                    self._fused_decode_replica_meta,
                )
                if self.__dict__.get("_fused_decode_num_replicas", 0) > 0
                else ()
            )
            out_sa = active_op(
                self._fused_decode_rank,
                scaled_router_logits,
                self._bias_2d(),
                expert_input,
                w1_arg,
                s1_arg,
                w2_arg,
                s2_arg,
                self._norm_2d(),
                w_post_arg,
                *rep_args,
            )
            if return_untransposed_gather and self.tp_size > 1:
                b_tok = out_sa.shape[0]
                cols = out_sa.shape[1] // self.tp_size
                return (
                    out_sa.view(b_tok, self.tp_size, cols)
                    .transpose(0, 1)
                    .reshape(self.tp_size * b_tok, cols)
                )
            return out_sa
        if return_sharded_moe_out:
            sh_op = self.__dict__.get("_fused_decode_sharded_op")
            if (
                not use_fused
                or sh_op is None
                or n_tok_in < self.tp_size
                or n_tok_in % self.tp_size != 0
            ):
                raise RuntimeError(
                    "MuseSparkMoE return_sharded_moe_out=True requires a prewarmed "
                    f"fused sparse decode op and batch divisible by tp_size={self.tp_size}, "
                    f"got use_fused={use_fused}, n_tok={n_tok_in}"
                )
        if use_fused:
            ex = self.experts
            w1_arg = self._buffers.get("_decode_fp4_w1", ex.w13_weight)
            s1_arg = self._buffers.get(
                "_decode_fp4_s1",
                self._buffers.get("_fused_decode_fp8_s1", ex.w13_weight_scale),
            )
            w2_arg = self._buffers.get("_decode_fp4_w2", ex.w2_weight)
            s2_arg = self._buffers.get(
                "_decode_fp4_s2",
                self._buffers.get("_fused_decode_fp8_s2", ex.w2_weight_scale),
            )
            if (
                not prerouted_router_topk
                and bool(_env_get("SGLANG_TPU_MUSE_DECODE_BRIDGE1_FUSE_TOPK", False))
                and int(_env_get("SGLANG_TPU_MUSE_DECODE_COLLECTIVE_BRIDGES", 0))
                in (1, 3, 4)
            ):
                tk_out = self._route_scaled_logits(
                    scaled_router_logits, prerouted=False
                )
                packed_lg = torch.zeros_like(scaled_router_logits, dtype=torch.float32)
                packed_lg[:, : self.top_k] = tk_out.topk_weights
                packed_lg[:, 8 : 8 + self.top_k] = tk_out.topk_ids.to(torch.float32)
                valid_hit = tk_out.topk_weights > 0.0
                e_ids = torch.arange(
                    256, device=scaled_router_logits.device, dtype=torch.int32
                )
                cnts = (
                    ((tk_out.topk_ids.unsqueeze(-1) == e_ids) & valid_hit.unsqueeze(-1))
                    .sum(dim=(0, 1))
                    .to(torch.float32)
                )
                packed_lg[0, 128:256] = cnts[:128]
                packed_lg[1, 128:256] = cnts[128:256]
                scaled_router_logits = packed_lg
            use_1d_vecs = self._use_1d_decode_vecs(fmode)
            base_args = (
                self._fused_decode_rank,
                scaled_router_logits,
                self.e_score_correction_bias if use_1d_vecs else self._bias_2d(),
                expert_input,
                w1_arg,
                s1_arg,
                w2_arg,
                s2_arg,
                self._fused_decode_norm if use_1d_vecs else self._norm_2d(),
            )
            use_v2_b = n_tok_in in self.__dict__.get("_fused_decode_v2_batches", ())
            v2_sh_op = self.__dict__.get("_fused_decode_v2_sharded_op")
            sharded_op = self.__dict__.get("_fused_decode_sharded_op")
            if return_sharded_moe_out:
                fused_op = (
                    v2_sh_op if (use_v2_b and v2_sh_op is not None) else sharded_op
                )
            else:
                fused_op = (
                    self._fused_decode_v2_op
                    if (
                        self.__dict__.get("_fused_decode_v2_op") is not None
                        and use_v2_b
                    )
                    else active_op
                )
            is_v2_call = fused_op in (
                self.__dict__.get("_fused_decode_v2_op"),
                v2_sh_op,
            )
            w3_tail = (
                (self._fused_decode_w3_mask,)
                if (is_v2_call and self.__dict__.get("_fused_decode_v2_has_w3", False))
                else ()
            )
            spec_tail = (
                (self._fused_decode_spec_slot,)
                if (
                    not is_v2_call
                    and self.__dict__.get("_fused_decode_has_spec_slot", False)
                )
                else ()
            )
            fp_buf = (
                self._free_params_for_batch(n_tok_in)
                if (not is_v2_call and fmode == "sparse")
                else None
            )
            fp_tail = (fp_buf,) if fp_buf is not None else ()
            if self.__dict__.get("_fused_decode_num_replicas", 0) > 0:
                output = fused_op(
                    *base_args,
                    self._fused_decode_rep_w1,
                    self._fused_decode_rep_s1,
                    self._fused_decode_rep_w2,
                    self._fused_decode_rep_s2,
                    self._fused_decode_replica_meta,
                    *w3_tail,
                    *spec_tail,
                    *fp_tail,
                )
            else:
                output = fused_op(*base_args, *w3_tail, *spec_tail, *fp_tail)
            if return_sharded_moe_out:
                return output
        elif self.__dict__.get("_tpu_v2_extend_fallback", False):
            v2_ext_op = self.__dict__.get("_fused_decode_v2_op")
            if v2_ext_op is None:
                raise RuntimeError(
                    "MuseSparkMoE _tpu_v2_extend_fallback requires _fused_decode_v2_op"
                )
            ex = self.experts
            w1_arg = self._buffers.get("_decode_fp4_w1", ex.w13_weight)
            s1_arg = self._buffers.get("_decode_fp4_s1", ex.w13_weight_scale)
            w2_arg = self._buffers.get("_decode_fp4_w2", ex.w2_weight)
            s2_arg = self._buffers.get("_decode_fp4_s2", ex.w2_weight_scale)
            scaled_ext_logits = (
                scaled_router_logits
                if precomputed_moe_inputs is not None
                else (topk_output.router_logits.float() * self.router_multiplier)
            )
            use_1d_ext = self._use_1d_decode_vecs("sparse_v2")
            bias_2d_ext = (
                self.e_score_correction_bias if use_1d_ext else self._bias_2d()
            )
            norm_2d_ext = (
                self._fused_decode_norm if use_1d_ext else self._norm_2d()
            )
            rep_tail = (
                (
                    self._fused_decode_rep_w1,
                    self._fused_decode_rep_s1,
                    self._fused_decode_rep_w2,
                    self._fused_decode_rep_s2,
                    self._fused_decode_replica_meta,
                )
                if self.__dict__.get("_fused_decode_num_replicas", 0) > 0
                else ()
            )
            w3_tail = (
                (self._fused_decode_w3_mask,)
                if self.__dict__.get("_fused_decode_v2_has_w3", False)
                else ()
            )
            ext_sh_op = self.__dict__.get("_fused_decode_v2_ext_sh_op")
            n_tok = expert_input.shape[0]
            step_b = 256
            chunks = []
            chunk_rows = []
            for s in range(0, n_tok, step_b):
                lg_c = scaled_ext_logits[s : s + step_b]
                x_c = expert_input[s : s + step_b]
                rem = x_c.shape[0]
                b_pad = 64 if rem <= 64 else (128 if rem <= 128 else 256)
                if rem < b_pad:
                    pad_r = b_pad - rem
                    lg_c = torch.cat(
                        [
                            lg_c,
                            torch.full(
                                (pad_r, lg_c.shape[1]),
                                float("nan"),
                                dtype=lg_c.dtype,
                                device=lg_c.device,
                            ),
                        ],
                        dim=0,
                    )
                    x_c = torch.cat(
                        [
                            x_c,
                            torch.zeros(
                                (pad_r, x_c.shape[1]),
                                dtype=x_c.dtype,
                                device=x_c.device,
                            ),
                        ],
                        dim=0,
                    )
                out_c = (ext_sh_op or v2_ext_op)(
                    self._fused_decode_rank,
                    lg_c,
                    bias_2d_ext,
                    x_c,
                    w1_arg,
                    s1_arg,
                    w2_arg,
                    s2_arg,
                    norm_2d_ext,
                    *rep_tail,
                    *w3_tail,
                )
                if ext_sh_op is None:
                    chunks.append(out_c[:rem])
                else:
                    chunks.append(out_c)
                    chunk_rows.append((b_pad // self.tp_size, rem))
            if ext_sh_op is None:
                output = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
            else:
                output = _gather_token_sharded_chunks(chunks, chunk_rows, self.tp_size)
        else:
            ep_tp2_fallback = self.__dict__.get("_ep_tp2_fallback_op")
            if self.experts.__dict__.get("_tpu_ep_tp2_packed_fp4", False):
                if ep_tp2_fallback is None:
                    raise RuntimeError(
                        "MuseSparkMoE hybrid EPxTP2 packed-FP4 weights require "
                        "_ep_tp2_fallback_op for non-fused batches"
                    )
                ex = self.experts
                max_chunk = 4096
                n_tok = expert_input.shape[0]
                if n_tok <= max_chunk:
                    output = ep_tp2_fallback(
                        self._fused_decode_rank,
                        expert_input,
                        topk_output.topk_weights,
                        topk_output.topk_ids,
                        ex.w13_weight,
                        ex.w13_weight_scale,
                        ex.w2_weight,
                        ex.w2_weight_scale,
                        self._fused_decode_norm,
                    )
                else:
                    chunks = [
                        ep_tp2_fallback(
                            self._fused_decode_rank,
                            expert_input[s : s + max_chunk],
                            topk_output.topk_weights[s : s + max_chunk],
                            topk_output.topk_ids[s : s + max_chunk],
                            ex.w13_weight,
                            ex.w13_weight_scale,
                            ex.w2_weight,
                            ex.w2_weight_scale,
                            self._fused_decode_norm,
                        )
                        for s in range(0, n_tok, max_chunk)
                    ]
                    output = torch.cat(chunks, dim=0)
            else:
                output = self.experts(expert_input, topk_output)
                if not self.experts.handles_moe_reduction:
                    output = self._combine_slots(output, topk_output.topk_weights)
        if return_unprojected_moe_out:
            return output
        w7_wpost = (
            self._buffers.get("_w7_wpost_w8k") if hasattr(self, "_buffers") else None
        )
        if w7_wpost is not None and self.post_expert_proj.weight.shape[0] == 0:
            output = (
                torch.matmul(output, w7_wpost.to(output.dtype)).float()
                * self._buffers["_w7_spost"]
            ).to(output.dtype)
            if (
                getattr(self.post_expert_proj, "gather_output", False)
                and self.tp_size > 1
            ):
                output = tensor_model_parallel_all_gather(output.contiguous())
        else:
            output, _ = self.post_expert_proj(output)
        if self._manual_post_gather:
            output = self._gather_columns(
                output, transpose_2d=not return_untransposed_gather
            )
        return output


class MuseSparkDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MuseSparkTextConfig,
        layer_idx: int,
        rotary_emb: MuseSparkInterleavedRope,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()
        hidden = config.hidden_size
        self.input_layernorm = MuseSparkRMSNorm(
            hidden, eps=config.rms_norm_eps, gain_center=1.0
        )
        self.self_attn = MuseSparkAttention(
            config, layer_idx, rotary_emb, add_prefix("self_attn", prefix)
        )
        self.post_attention_layernorm = MuseSparkRMSNorm(
            hidden, eps=config.post_norm_eps, has_weight=False
        )
        self.post_attention_residual_gate = MuseSparkGatedResidual(
            hidden, temperature=config.residual_gate_temperature
        )
        self.pre_feedforward_layernorm = MuseSparkRMSNorm(
            hidden, eps=config.rms_norm_eps, gain_center=1.0
        )
        self.mlp = MuseSparkMoE(
            config, layer_idx, quant_config, add_prefix("mlp", prefix)
        )
        self.post_feedforward_layernorm = MuseSparkRMSNorm(
            hidden, eps=config.post_norm_eps, gain_center=1.0, scale_input=True
        )
        self.post_feedforward_residual_gate = MuseSparkGatedResidual(
            hidden, temperature=config.residual_gate_temperature
        )
        self._decode_norm_layout = (
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_NORM_LAYOUT.get()
        )
        self._decode_fuse_target_verify = bool(
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_FUSE_TARGET_VERIFY.get()
            or _env_get("SGLANG_TPU_MUSE_VERIFY_BRIDGE_LIFT", False)
        )
        self._decode_col_bridges = int(
            _env_get("SGLANG_TPU_MUSE_DECODE_COLLECTIVE_BRIDGES", 0)
        )
        if self._decode_col_bridges not in (0, 1, 2, 3, 4, 5, 6):
            raise ValueError(
                "SGLANG_TPU_MUSE_DECODE_COLLECTIVE_BRIDGES must be in 0..6, "
                f"got {self._decode_col_bridges}"
            )
        self._decode_fused_o_proj_ar = int(
            _env_get("SGLANG_TPU_MUSE_DECODE_FUSED_O_PROJ_AR", 0)
        )
        self._decode_attn_qkv_tpar_auto = False
        if self._decode_fused_o_proj_ar == 0 and bool(
            _env_get("SGLANG_TPU_MUSE_DECODE_ATTN_QKV_TPAR", False)
        ):
            # Proven fastest lossless composition: standalone XOR=1 bridge_all_reduce (mode=2)
            # feeding nl>=2 muse_decode_res_norm (0/0 LLO spills, wins across B=16..256 on TP8
            # and B=16..128 on TP16), or fused tp_ar_res_norm (mode=1) when decode_norm_layout < 2.
            self._decode_fused_o_proj_ar = 2 if self._decode_norm_layout >= 2 else 1
            self._decode_attn_qkv_tpar_auto = True
        if self._decode_fused_o_proj_ar not in (0, 1, 2, 3):
            raise ValueError(
                "SGLANG_TPU_MUSE_DECODE_FUSED_O_PROJ_AR must be one of (0, 1, 2, 3), "
                f"got {self._decode_fused_o_proj_ar}"
            )
        if self._decode_norm_layout > 0:
            from sglang_tpu.moe.pallas.fused_rs_muse.decode_norm_router import (
                get_muse_decode_res_norm_op,
                get_muse_decode_res_only_op,
                get_muse_decode_rmsnorm_op,
                get_muse_decode_tp_ag_res_norm_op,
                get_muse_decode_tp_ag_res_only_op,
            )

            self.register_buffer(
                "_dummy_gain",
                torch.ones(hidden, dtype=torch.float32),
                persistent=False,
            )
            get_muse_decode_rmsnorm_op()
            if self._decode_norm_layout >= 2:
                get_muse_decode_res_norm_op()
                get_muse_decode_res_only_op()
            if self._decode_norm_layout >= 4:
                get_muse_decode_tp_ag_res_only_op()
                if self._decode_norm_layout >= 5:
                    get_muse_decode_tp_ag_res_norm_op()
        if self._decode_fused_o_proj_ar > 0:
            op_tp = int(getattr(self.self_attn.o_proj, "tp_size", 1))
            if op_tp in (8, 16) and hidden % (128 * op_tp) == 0:
                from sglang_tpu.moe.pallas.fused_rs_muse.attn_ar_res_norm import (
                    get_muse_decode_attn_tail_ar_res_norm_op,
                    get_muse_decode_bridge_ar_op,
                    get_muse_decode_tp_ar_res_norm_op,
                )

                if self._decode_fused_o_proj_ar == 1:
                    get_muse_decode_tp_ar_res_norm_op(op_tp)
                elif self._decode_fused_o_proj_ar == 2:
                    get_muse_decode_bridge_ar_op(op_tp)
                elif self._decode_fused_o_proj_ar == 3:
                    get_muse_decode_attn_tail_ar_res_norm_op(
                        op_tp,
                        fp8_wo=bool(getattr(self.self_attn, "_fp8_attn_proj", False)),
                    )

    def _active_fused_o_proj_ar_mode(
        self, n_tok: int, forward_batch: ForwardBatch
    ) -> int:
        mode = int(getattr(self, "_decode_fused_o_proj_ar", 0))
        if (
            mode == 0
            or forward_batch.forward_mode == ForwardMode.EXTEND
            or n_tok < 16
            or n_tok > _decode_max_tokens()
            or n_tok % 16 != 0
        ):
            return 0
        op = getattr(self.self_attn, "o_proj", None)
        if op is None or not isinstance(op, RowParallelLinear):
            return 0
        tp = _layer_tp_size(op)
        if (
            tp not in (8, 16)
            or not op.reduce_results
            or op.bias is not None
            or getattr(op, "use_dp_attention_reduce", False)
            or getattr(op, "use_decode_attn_tp", False)
            or op.output_size % (128 * tp) != 0
            or int(os.environ.get("WORLD_SIZE", str(tp))) != tp
        ):
            return 0
        from sglang.srt.runtime_context import get_forward

        if getattr(get_forward(), "sp_active", False):
            return 0
        # On 2-host TP=16 at B=256 (4 MiB), native 3D-torus XLA AR beats bridge AR; keep TP8
        # active through B=256 and TP16 active through B=128 under the auto combo flag.
        if (
            getattr(self, "_decode_attn_qkv_tpar_auto", False)
            and tp >= 16
            and n_tok >= 256
        ):
            return 0
        # Mode 1 (tp_ar_res_norm) and Mode 3 (8-op tail) only beat XLA for B <= 128 (#655).
        if mode in (1, 3) and n_tok > 128:
            return 0
        if mode == 3:
            has_fp8 = getattr(self.self_attn, "_o_proj_w8k", None) is not None
            has_bf16 = getattr(self.self_attn, "_o_proj_wk", None) is not None
            if not (has_fp8 or has_bf16) or self.self_attn.head_dim != 128:
                return 0
        return mode

    def _run_fused_o_proj_ar_post_attn(
        self,
        mode: int,
        positions: torch.Tensor,
        attn_input: torch.Tensor,
        stream: torch.Tensor,
        forward_batch: ForwardBatch,
        *,
        cos_sin_step: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run ``self_attn`` + fused ``tp_ar_res_norm`` (mode 1) or 8-op tail (mode 3)."""
        from sglang_tpu.moe.pallas.fused_rs_muse.attn_ar_res_norm import (
            get_muse_decode_attn_tail_ar_res_norm_op,
            get_muse_decode_tp_ar_res_norm_op,
        )

        tp = _layer_tp_size(self.self_attn.o_proj)
        pre_ff_gain = self.pre_feedforward_layernorm.gain()
        assert pre_ff_gain is not None
        cs_kw = {"cos_sin_step": cos_sin_step} if cos_sin_step is not None else {}
        if mode == 3:
            attn_raw, gate = self.self_attn(
                positions,
                attn_input,
                forward_batch,
                return_attn_and_gate=True,
                **cs_kw,
            )
            wo_w8k = getattr(self.self_attn, "_o_proj_w8k", None)
            if wo_w8k is not None:
                return get_muse_decode_attn_tail_ar_res_norm_op(tp, fp8_wo=True)(
                    stream,
                    attn_raw,
                    gate,
                    wo_w8k,
                    self.self_attn._o_proj_scale,
                    self.post_attention_residual_gate.alpha,
                    self.post_attention_residual_gate.beta,
                    pre_ff_gain,
                    float(self.self_attn.norm_eps),
                    float(self.post_attention_layernorm.eps),
                    float(self.pre_feedforward_layernorm.eps),
                )
            return get_muse_decode_attn_tail_ar_res_norm_op(tp, fp8_wo=False)(
                stream,
                attn_raw,
                gate,
                self.self_attn._o_proj_wk,
                self.post_attention_residual_gate.alpha,
                self.post_attention_residual_gate.beta,
                pre_ff_gain,
                float(self.self_attn.norm_eps),
                float(self.post_attention_layernorm.eps),
                float(self.pre_feedforward_layernorm.eps),
            )
        o_partial = self.self_attn(
            positions, attn_input, forward_batch, skip_o_proj_all_reduce=True, **cs_kw
        )
        return get_muse_decode_tp_ar_res_norm_op(tp)(
            stream,
            o_partial,
            self.post_attention_residual_gate.alpha,
            self.post_attention_residual_gate.beta,
            pre_ff_gain,
            float(self.post_attention_layernorm.eps),
            float(self.pre_feedforward_layernorm.eps),
        )

    def _active_w7_bridges(
        self,
        n_tok: int,
        forward_batch: ForwardBatch,
        *,
        is_dec_step: bool,
        next_input_layernorm: Optional[MuseSparkRMSNorm],
        next_layer_attn: Optional[MuseSparkAttention],
        sharded_stream: bool = False,
    ) -> Tuple[int, int]:
        """Return ``(b1_kind, b2_kind)`` for Wave-7 decode collective bridges.

        ``b1_kind``: ``0`` = off, ``1`` = Sub-Bridge 1c (FP8 o_proj + XLA AR + 7-op
        ``post_attn_to_moe_bridge``), ``2`` = Full 10-op ``attn_to_moe_bridge``.
        ``b2_kind``: ``0`` = off, ``1`` = L61 ``moe_post_ag_res_only``, ``2`` = Bridge 2a
        ``moe_post_ag_res_norm``, ``3`` = Full 7-op inter-layer ``moe_to_attn_bridge``.
        """
        mode = self.__dict__.get("_decode_col_bridges", 0)
        br_cap = _decode_bridge_max_tokens()
        if (
            not isinstance(mode, int)
            or mode <= 0
            or forward_batch.forward_mode == ForwardMode.EXTEND
            or n_tok < 8
            or n_tok > br_cap
            or n_tok % 8 != 0
        ):
            return 0, 0
        op = getattr(self.self_attn, "o_proj", None)
        if op is None or not isinstance(op, RowParallelLinear):
            return 0, 0
        tp = _layer_tp_size(op)
        if (
            tp not in (8, 16)
            or not op.reduce_results
            or op.bias is not None
            or getattr(op, "use_dp_attention_reduce", False)
            or getattr(op, "use_decode_attn_tp", False)
            or int(self.mlp.tp_size) != tp
            or int(os.environ.get("WORLD_SIZE", str(tp))) != tp
        ):
            return 0, 0
        from sglang.srt.runtime_context import get_forward

        if getattr(get_forward(), "sp_active", False):
            return 0, 0

        b1_kind = 0
        if (
            mode in (1, 3, 4, 5, 6)
            and self.self_attn._o_proj_w8k is not None
            and "_w7_wpre_w8k" in getattr(self.mlp, "_buffers", {})
            and "_w7_wrouter_km" in getattr(self.mlp, "_buffers", {})
            and not self.mlp.fuses_pre_gather(n_tok, decode=is_dec_step)
        ):
            # Hardware-verified: full 10-op Bridge 1b (Dual-Tree XY Butterfly AR on TP=8,
            # XOR=1 on TP=16) wins across B=8..128 on TP=8 and B=8..64 on TP=16.
            if mode in (5, 6):
                b1_kind = 1
            elif sharded_stream or tp == 8 or n_tok <= 64 or br_cap > 128:
                b1_kind = 2

        b2_kind = 0
        if (
            mode in (2, 3, 4, 5, 6)
            and "_w7_wpost_w8k" in getattr(self.mlp, "_buffers", {})
            and not self.mlp.fuses_post_gather(n_tok, decode=is_dec_step)
        ):
            if next_input_layernorm is None:
                b2_kind = 1
            elif (
                mode in (2, 3, 5)
                and next_layer_attn is not None
                and "_w7_wqkv_w8k" in getattr(next_layer_attn, "_buffers", {})
                and "_w7_wgate_w8k" in getattr(next_layer_attn, "_buffers", {})
            ):
                b2_kind = 3
            else:
                b2_kind = 2
        return b1_kind, b2_kind

    def _can_run_b1_moe_superkernel(
        self,
        n_tok: int,
        *,
        tp: int,
        is_dec_step: bool,
        replicated: bool = False,
    ) -> bool:
        """True when ``self`` and its NVFP4 sparse MoE match Wave-9 superkernels."""
        if not bool(self.__dict__.get("_w9_b1_moe_b2_superkernel", False)) or not is_dec_step:
            return False
        mlp = self.mlp
        valid_b = (
            (8, 16)
            if (replicated and tp == 8)
            else ((16,) if replicated else ((32, 64, 128) if tp == 8 else (64, 128, 256)))
        )
        pre_buf_key = "_w7_wpre_w8k" if replicated else "_w8_wpre_pair_w8k"
        if (
            n_tok not in valid_b
            or mlp.__dict__.get("_fused_decode_mode", "") != "sparse"
            or mlp.__dict__.get("_fused_decode_op") is None
            or n_tok not in mlp.__dict__.get("_fused_decode_batches", ())
            or pre_buf_key not in mlp._buffers
            or "_w7_wrouter_km" not in mlp._buffers
        ):
            return False
        allow_eplb = bool(
            mlp.__dict__.get("_decode_allow_prefill_eplb", False)
            or _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_ALLOW_PREFILL_EPLB", False)
        )
        if mlp.__dict__.get("_eplb_num_replicas", 0) and not allow_eplb:
            return False
        ex = mlp.experts
        has_shadow = "_decode_fp4_w1" in mlp._buffers
        w1_arg = (
            mlp._buffers["_decode_fp4_w1"]
            if has_shadow
            else getattr(ex, "w13_weight", None)
        )
        s1_arg = (
            mlp._buffers["_decode_fp4_s1"]
            if has_shadow
            else getattr(ex, "w13_weight_scale", None)
        )
        if w1_arg is None or s1_arg is None:
            return False
        e_local = 256 // tp
        max_groups = (e_local + 8) if allow_eplb else e_local
        return (
            w1_arg.dtype == torch.float4_e2m1fn_x2
            and w1_arg.ndim == 3
            and (e_local <= int(w1_arg.shape[0]) <= max_groups)
            and tuple(w1_arg.shape[1:]) == (4096, 4096)
            and s1_arg.dtype == torch.bfloat16
            and s1_arg.ndim == 3
            and int(s1_arg.shape[1]) > 0
            and 4096 % int(s1_arg.shape[1]) == 0
            and (4096 // int(s1_arg.shape[1])) in (16, 256, 512)
            and mlp._bias_2d().dtype == torch.float32
            and mlp._norm_2d().dtype in (torch.bfloat16, torch.float32)
        )

    def _try_b1_moe_replicated_b16_superkernel(
        self,
        *,
        positions: torch.Tensor,
        stream: torch.Tensor,
        attn_input: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        n_tok: int,
        tp: int,
        b2_kind: int,
        is_dec_step: bool,
        pa_alpha_arg: torch.Tensor,
        pa_beta_arg: torch.Tensor,
        pre_ff_gain: torch.Tensor,
        pre_exp_gain: torch.Tensor,
        cos_sin_step: Optional[torch.Tensor],
        cs_kw: dict,
        precomputed_qkv_gate: Optional[Tuple[torch.Tensor, torch.Tensor]],
        precomputed_prepared_qkvg: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        precomputed_gated: Optional[torch.Tensor],
        next_input_layernorm: Optional[MuseSparkRMSNorm],
        next_layer_attn: Optional[MuseSparkAttention],
    ):
        """Run replicated ``B=16`` Step (b) superkernel when enabled."""
        if b2_kind != 3 or not self._can_run_b1_moe_superkernel(
            n_tok, tp=tp, is_dec_step=is_dec_step, replicated=True
        ):
            return None
        mlp = self.mlp
        if (
            "_w7_wpost_w8k" not in mlp._buffers
            or next_input_layernorm is None
            or next_layer_attn is None
            or "_w7_wqkv_w8k" not in next_layer_attn._buffers
            or "_w7_wgate_w8k" not in next_layer_attn._buffers
        ):
            return None
        use_qkv_epi = (
            (
                tp == 8
                or (
                    tp == 16
                    and bool(tpu_envs.SGLANG_TPU_MUSE_DECODE_TP16_QKV_EPI.get())
                )
            )
            and bool(self.__dict__.get("_w8_bridge_qkv_epi", False))
            and _can_use_bridge_qkv_epi(
                n_tok,
                next_layer_attn,
                cos_sin_step,
                forward_batch,
                max_tok=384,
            )
        )
        next_has_rope = next_layer_attn.rotary_emb is not None
        if not use_qkv_epi or (next_has_rope and cos_sin_step is None):
            return None
        from sglang_tpu.muse_megakernel.decode.b1_moe_ops import (
            get_b1_moe_b2_replicated_b16_op,
        )

        post_ff_gain = self.post_feedforward_layernorm.gain()
        next_in_gain = next_input_layernorm.gain()
        assert post_ff_gain is not None and next_in_gain is not None
        gated = (
            precomputed_gated
            if precomputed_gated is not None
            else self.self_attn(
                positions,
                attn_input,
                forward_batch,
                precomputed_qkv_gate=precomputed_qkv_gate,
                precomputed_prepared_qkvg=precomputed_prepared_qkvg,
                return_gated_attn=True,
                **cs_kw,
            )
        )
        ex = mlp.experts
        has_shadow = "_decode_fp4_w1" in mlp._buffers
        w1_arg = mlp._buffers["_decode_fp4_w1"] if has_shadow else ex.w13_weight
        s1_arg = mlp._buffers["_decode_fp4_s1"] if has_shadow else ex.w13_weight_scale
        w2_arg = mlp._buffers["_decode_fp4_w2"] if has_shadow else ex.w2_weight
        s2_arg = mlp._buffers["_decode_fp4_s2"] if has_shadow else ex.w2_weight_scale
        w_router_km = mlp._buffers["_w7_wrouter_km"]
        num_rep = int(mlp.__dict__.get("_fused_decode_num_replicas", 0))
        has_spec = bool(mlp.__dict__.get("_fused_decode_has_spec_slot", False))
        cost_mode = str(mlp.__dict__.get("_fused_decode_replica_cost_mode", "tiles"))
        rep_tail = (
            (
                mlp._fused_decode_rep_w1,
                mlp._fused_decode_rep_s1,
                mlp._fused_decode_rep_w2,
                mlp._fused_decode_rep_s2,
                mlp._fused_decode_replica_meta,
            )
            if num_rep > 0
            else ()
        )
        spec_tail = (mlp._fused_decode_spec_slot,) if has_spec else ()
        fp_buf = mlp._free_params_for_batch(n_tok)
        fp_tail = (fp_buf,) if fp_buf is not None else ()
        op_b16 = get_b1_moe_b2_replicated_b16_op(
            tp,
            bf16_router=(w_router_km.dtype == torch.bfloat16),
            num_replicas=num_rep,
            has_spec_slot=has_spec,
            replica_cost_mode=cost_mode,
            has_rope=next_has_rope,
            has_free_params=(fp_buf is not None),
        )
        cs_tail = (cos_sin_step,) if next_has_rope else ()
        return op_b16(
            mlp._fused_decode_rank,
            stream,
            gated,
            self.self_attn._o_proj_w8k,
            self.self_attn._o_proj_scale,
            pa_alpha_arg,
            pa_beta_arg,
            pre_ff_gain,
            mlp._buffers["_w7_wpre_w8k"],
            mlp._buffers["_w7_spre"],
            w_router_km,
            pre_exp_gain,
            mlp._bias_2d(),
            w1_arg,
            s1_arg,
            w2_arg,
            s2_arg,
            mlp._norm_2d(),
            mlp._buffers["_w7_wpost_w8k"],
            mlp._buffers["_w7_spost"],
            post_ff_gain,
            self.post_feedforward_residual_gate.alpha,
            self.post_feedforward_residual_gate.beta,
            next_in_gain,
            next_layer_attn._buffers["_w7_wqkv_w8k"],
            next_layer_attn._buffers["_w7_sqkv"],
            next_layer_attn._buffers["_w7_wgate_w8k"],
            next_layer_attn._buffers["_w7_sgate"],
            *cs_tail,
            *rep_tail,
            *spec_tail,
            *fp_tail,
            float(self.post_attention_layernorm.eps),
            float(self.pre_feedforward_layernorm.eps),
            bool(self.pre_feedforward_layernorm.scale_input),
            float(mlp.router_multiplier),
            float(mlp.pre_expert_norm.eps),
            bool(mlp.pre_expert_norm.scale_input),
            float(self.post_feedforward_layernorm.eps),
            bool(self.post_feedforward_layernorm.scale_input),
            float(next_input_layernorm.eps),
            bool(next_input_layernorm.scale_input),
            float(next_layer_attn.norm_eps),
            float(next_layer_attn.query_scale),
            float(getattr(next_layer_attn.attn, "k_scale_float", None) or 1.0),
            float(getattr(next_layer_attn.attn, "v_scale_float", None) or 1.0),
        )

    def _try_b1_moe_superkernel(
        self,
        *,
        positions: torch.Tensor,
        stream: torch.Tensor,
        attn_input: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        n_tok: int,
        tp: int,
        b2_kind: int,
        is_dec_step: bool,
        pa_alpha_arg: torch.Tensor,
        pa_beta_arg: torch.Tensor,
        pre_ff_gain: torch.Tensor,
        pre_exp_gain: torch.Tensor,
        cos_sin_step: Optional[torch.Tensor],
        cs_kw: dict,
        precomputed_qkv_gate: Optional[Tuple[torch.Tensor, torch.Tensor]],
        precomputed_prepared_qkvg: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        precomputed_gated: Optional[torch.Tensor],
        next_input_layernorm: Optional[MuseSparkRMSNorm],
        next_layer_attn: Optional[MuseSparkAttention],
        final_layernorm: Optional[MuseSparkRMSNorm],
        pending_step_c_kv_writes: Optional[list] = None,
    ):
        """Run fused Step (c) [+RPAd], Step (b) [b1+MoE+b2], or Step (a) [b1+MoE] (``L61``)."""
        if not self._can_run_b1_moe_superkernel(n_tok, tp=tp, is_dec_step=is_dec_step):
            return None
        from sglang_tpu.muse_megakernel.decode.b1_moe_ops import (
            get_b1_moe_b2_chip_sharded_op,
            get_b1_moe_b2_rpad_chip_sharded_op,
            get_b1_moe_chip_sharded_op,
            get_step_c_kv_write_op,
        )

        mlp = self.mlp
        ex = mlp.experts
        w_router_km = mlp._buffers["_w7_wrouter_km"]
        has_shadow = "_decode_fp4_w1" in mlp._buffers
        w1_arg = mlp._buffers["_decode_fp4_w1"] if has_shadow else ex.w13_weight
        s1_arg = mlp._buffers["_decode_fp4_s1"] if has_shadow else ex.w13_weight_scale
        w2_arg = mlp._buffers["_decode_fp4_w2"] if has_shadow else ex.w2_weight
        s2_arg = mlp._buffers["_decode_fp4_s2"] if has_shadow else ex.w2_weight_scale
        num_rep = int(mlp.__dict__.get("_fused_decode_num_replicas", 0))
        has_spec = bool(mlp.__dict__.get("_fused_decode_has_spec_slot", False))
        cost_mode = str(mlp.__dict__.get("_fused_decode_replica_cost_mode", "tiles"))
        rep_tail = (
            (
                mlp._fused_decode_rep_w1,
                mlp._fused_decode_rep_s1,
                mlp._fused_decode_rep_w2,
                mlp._fused_decode_rep_s2,
                mlp._fused_decode_replica_meta,
            )
            if num_rep > 0
            else ()
        )
        spec_tail = (mlp._fused_decode_spec_slot,) if has_spec else ()
        fp_buf = mlp._free_params_for_batch(n_tok)
        fp_tail = (fp_buf,) if fp_buf is not None else ()
        bf16_router = w_router_km.dtype == torch.bfloat16

        if (
            b2_kind == 3
            and "_w9_wpost_pair_w8k" in mlp._buffers
            and next_input_layernorm is not None
            and next_layer_attn is not None
            and "_w7_wqkv_w8k" in next_layer_attn._buffers
            and "_w7_wgate_w8k" in next_layer_attn._buffers
        ):
            post_ff_gain = self.post_feedforward_layernorm.gain()
            next_in_gain = next_input_layernorm.gain()
            assert post_ff_gain is not None and next_in_gain is not None
            use_qkv_epi = (
                (
                    tp == 8
                    or (
                        tp == 16
                        and bool(_env_get("SGLANG_TPU_MUSE_DECODE_TP16_QKV_EPI", False))
                    )
                )
                and bool(self.__dict__.get("_w8_bridge_qkv_epi", False))
                and _can_use_bridge_qkv_epi(
                    n_tok,
                    next_layer_attn,
                    cos_sin_step,
                    forward_batch,
                    max_tok=384,
                )
            )
            next_has_rope = next_layer_attn.rotary_emb is not None
            if use_qkv_epi and next_has_rope and cos_sin_step is None:
                return None
            gated = (
                precomputed_gated
                if precomputed_gated is not None
                else self.self_attn(
                    positions,
                    attn_input,
                    forward_batch,
                    precomputed_qkv_gate=precomputed_qkv_gate,
                    precomputed_prepared_qkvg=precomputed_prepared_qkvg,
                    return_gated_attn=True,
                    **cs_kw,
                )
            )
            cs_tail = (cos_sin_step,) if (use_qkv_epi and next_has_rope) else ()
            epi_scalars = (
                (
                    float(next_layer_attn.norm_eps),
                    float(next_layer_attn.query_scale),
                    float(getattr(next_layer_attn.attn, "k_scale_float", None) or 1.0),
                    float(getattr(next_layer_attn.attn, "v_scale_float", None) or 1.0),
                )
                if use_qkv_epi
                else ()
            )
            step_c_flag = getattr(tpu_envs, "SGLANG_TPU_MUSE_DECODE_STEP_C_SUPERKERNEL", None)
            capture_mode = getattr(forward_batch, "capture_hidden_mode", CaptureHiddenMode.NULL)
            if (
                step_c_flag is not None
                and bool(step_c_flag.get())
                and use_qkv_epi
                and 32 <= n_tok <= 64
                and getattr(next_layer_attn, "decode_attn_gate_fuse", False)
                and (capture_mode is None or not capture_mode.is_full())
            ):
                from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
                    resolve_step_c_rpa_plan,
                )
                from sglang_tpu.attention_pallas_rpa_base import _fused_kv_operands
                from sglang_tpu.decode_qkv_epilogue_tpu import (
                    _resolve_decode_attn_backend,
                )

                backend = _resolve_decode_attn_backend(forward_batch)
                md = getattr(backend, "forward_metadata", None) if backend is not None else None
                rpa_mode_str = str(tpu_envs.SGLANG_TPU_RPA_V3_DECODE_FAST.get()).strip()
                sw = getattr(next_layer_attn.attn, "sliding_window_size", None)
                if sw is not None and sw <= 0:
                    sw = None
                elif sw is not None:
                    sw = int(sw) + 1
                lc_swa_step_c = bool(tpu_envs.SGLANG_TPU_MUSE_MK_STEP_C_LC_SWA.get())
                if (
                    rpa_mode_str
                    and backend is not None
                    and md is not None
                    and getattr(backend, "_decode_prepared_gated_op", None) is not None
                    and getattr(md, "decode_query_size", 0) == 1
                    and (
                        (lc_swa_step_c and sw is not None)
                        or getattr(md, "decode_shared_prefix_pages", None) is None
                    )
                    and backend.can_forward_decode_prepared(
                        next_layer_attn.attn, forward_batch
                    )
                ):
                    kv_cache, kv_layer_idx = _fused_kv_operands(
                        backend.token_to_kv_pool, next_layer_attn.attn.layer_id
                    )
                    kv_lens = backend._kv_lens(forward_batch)
                    page_indices = backend._layer_page_indices(
                        next_layer_attn.attn.layer_id, is_swa_layer=(sw is not None)
                    )
                    cu_q_lens = md.cu_q_lens
                    cu_kv_lens = md.cu_kv_lens
                    distribution = backend._layer_distribution(
                        is_swa_layer=(sw is not None)
                    )
                    sub_pool = getattr(
                        backend.token_to_kv_pool,
                        "swa_kv_pool" if sw is not None else "full_kv_pool",
                        None,
                    )
                    dense2 = bool(
                        getattr(
                            sub_pool,
                            "kv_pack_dense2",
                            getattr(backend, "_kv_pack_dense2", False),
                        )
                    )
                    bd4 = bool(
                        getattr(
                            sub_pool,
                            "kv_pack_bd4",
                            getattr(backend, "_kv_pack_bd4", False),
                        )
                    )
                    phys_rows = int(kv_cache.shape[1 if kv_layer_idx is None else 2])
                    pages_per_seq = int(page_indices.shape[0]) // max(
                        1, int(kv_lens.shape[0])
                    )
                    has_shared_prefix = int(distribution.shape[0]) > 3
                    cfg_rpa, _, _, _, step_c_rpa_ok = resolve_step_c_rpa_plan(
                        rpa_mode_str,
                        sliding_window=sw,
                        pages_per_seq=pages_per_seq,
                        phys_page_rows=phys_rows,
                        kv_pack_dense2=dense2,
                        kv_pack_bd4=bd4,
                        b_tokens=n_tok,
                        max_num_seqs=int(kv_lens.shape[0]),
                        has_shared_prefix=has_shared_prefix,
                    )
                    if step_c_rpa_ok and (tp != 16 or cfg_rpa.num_slots <= 4):
                        op_c = get_b1_moe_b2_rpad_chip_sharded_op(
                            tp,
                            bf16_router=bf16_router,
                            num_replicas=num_rep,
                            has_spec_slot=has_spec,
                            replica_cost_mode=cost_mode,
                            has_rope=next_has_rope,
                            has_free_params=(fp_buf is not None),
                            has_layer_idx=(kv_layer_idx is not None),
                            kv_pack_dense2=dense2,
                            kv_pack_bd4=bd4,
                            rpa_mode_str=rpa_mode_str,
                        )
                        lidx_tail = (kv_layer_idx,) if kv_layer_idx is not None else ()
                        next_stream_c, next_gated_c, kv_merged_c = op_c(
                            mlp._fused_decode_rank,
                            stream,
                            gated,
                            self.self_attn._o_proj_w8k,
                            self.self_attn._o_proj_scale,
                            pa_alpha_arg,
                            pa_beta_arg,
                            pre_ff_gain,
                            mlp._buffers["_w8_wpre_pair_w8k"],
                            mlp._buffers["_w8_spre_pair"],
                            w_router_km,
                            pre_exp_gain,
                            mlp._bias_2d(),
                            w1_arg,
                            s1_arg,
                            w2_arg,
                            s2_arg,
                            mlp._norm_2d(),
                            mlp._buffers["_w9_wpost_pair_w8k"],
                            mlp._buffers["_w9_spost_pair"],
                            post_ff_gain,
                            self.post_feedforward_residual_gate.alpha,
                            self.post_feedforward_residual_gate.beta,
                            next_in_gain,
                            next_layer_attn._buffers["_w7_wqkv_w8k"],
                            next_layer_attn._buffers["_w7_sqkv"],
                            next_layer_attn._buffers["_w7_wgate_w8k"],
                            next_layer_attn._buffers["_w7_sgate"],
                            kv_cache,
                            kv_lens,
                            page_indices,
                            cu_q_lens,
                            cu_kv_lens,
                            distribution,
                            *cs_tail,
                            *rep_tail,
                            *spec_tail,
                            *fp_tail,
                            *lidx_tail,
                            float(self.post_attention_layernorm.eps),
                            float(self.pre_feedforward_layernorm.eps),
                            bool(self.pre_feedforward_layernorm.scale_input),
                            float(mlp.router_multiplier),
                            float(mlp.pre_expert_norm.eps),
                            bool(mlp.pre_expert_norm.scale_input),
                            float(self.post_feedforward_layernorm.eps),
                            bool(self.post_feedforward_layernorm.scale_input),
                            float(next_input_layernorm.eps),
                            bool(next_input_layernorm.scale_input),
                            *epi_scalars,
                            float(next_layer_attn.attn.scaling),
                            int(sw) if sw is not None else -1,
                            float(getattr(next_layer_attn.attn, "k_scale_float", None) or 1.0),
                            float(getattr(next_layer_attn.attn, "v_scale_float", None) or 1.0),
                            float(next_layer_attn.norm_eps),
                        )
                        if pending_step_c_kv_writes is not None:
                            swa_groups = getattr(backend, "_swa_layer_groups", None)
                            if (
                                sw is not None
                                and getattr(md, "swa_page_indices", None) is not None
                            ):
                                g_id = (
                                    int(
                                        swa_groups.get(
                                            next_layer_attn.attn.layer_id, 0
                                        )
                                    )
                                    if swa_groups is not None
                                    else 0
                                )
                                swa_grp = (
                                    0
                                    if (
                                        g_id == 0
                                        or getattr(md, "swa_group_page_indices", None)
                                        is None
                                    )
                                    else g_id
                                )
                            else:
                                swa_grp = -1
                            grp_key = (
                                kv_layer_idx is not None,
                                sw is not None,
                                swa_grp,
                                bool(dense2),
                                bool(bd4),
                            )
                            pending_step_c_kv_writes.append(
                                (
                                    kv_cache,
                                    kv_lens,
                                    page_indices,
                                    cu_q_lens,
                                    cu_kv_lens,
                                    distribution,
                                    kv_layer_idx,
                                    kv_merged_c,
                                    dense2,
                                    bd4,
                                    grp_key,
                                    int(next_layer_attn.attn.layer_id),
                                )
                            )
                        else:
                            get_step_c_kv_write_op(kv_pack_dense2=dense2, kv_pack_bd4=bd4)(
                                kv_merged_c,
                                kv_cache,
                                kv_lens,
                                page_indices,
                                cu_q_lens,
                                cu_kv_lens,
                                distribution,
                                kv_layer_idx,
                            )
                        return next_stream_c, next_gated_c, None, None, None

            op_b2 = get_b1_moe_b2_chip_sharded_op(
                tp,
                bf16_router=bf16_router,
                num_replicas=num_rep,
                has_spec_slot=has_spec,
                replica_cost_mode=cost_mode,
                fuse_qkv_epi=use_qkv_epi,
                has_rope=next_has_rope,
                has_free_params=(fp_buf is not None),
            )
            return op_b2(
                mlp._fused_decode_rank,
                stream,
                gated,
                self.self_attn._o_proj_w8k,
                self.self_attn._o_proj_scale,
                pa_alpha_arg,
                pa_beta_arg,
                pre_ff_gain,
                mlp._buffers["_w8_wpre_pair_w8k"],
                mlp._buffers["_w8_spre_pair"],
                w_router_km,
                pre_exp_gain,
                mlp._bias_2d(),
                w1_arg,
                s1_arg,
                w2_arg,
                s2_arg,
                mlp._norm_2d(),
                mlp._buffers["_w9_wpost_pair_w8k"],
                mlp._buffers["_w9_spost_pair"],
                post_ff_gain,
                self.post_feedforward_residual_gate.alpha,
                self.post_feedforward_residual_gate.beta,
                next_in_gain,
                next_layer_attn._buffers["_w7_wqkv_w8k"],
                next_layer_attn._buffers["_w7_sqkv"],
                next_layer_attn._buffers["_w7_wgate_w8k"],
                next_layer_attn._buffers["_w7_sgate"],
                *cs_tail,
                *rep_tail,
                *spec_tail,
                *fp_tail,
                float(self.post_attention_layernorm.eps),
                float(self.pre_feedforward_layernorm.eps),
                bool(self.pre_feedforward_layernorm.scale_input),
                float(mlp.router_multiplier),
                float(mlp.pre_expert_norm.eps),
                bool(mlp.pre_expert_norm.scale_input),
                float(self.post_feedforward_layernorm.eps),
                bool(self.post_feedforward_layernorm.scale_input),
                float(next_input_layernorm.eps),
                bool(next_input_layernorm.scale_input),
                *epi_scalars,
            )

        ret_sh_step_a = (n_tok // tp) >= 8 and (n_tok // tp) % 8 == 0
        if (
            b2_kind in (1, 2)
            and "_w7_wpost_w8k" in mlp._buffers
            and (ret_sh_step_a or (tp == 16 and n_tok == 64))
        ):
            from sglang_tpu.kernels.decode_collective_bridges_w8 import (
                get_moe_post_chip_sharded_final_norm_op,
            )

            gated = (
                precomputed_gated
                if precomputed_gated is not None
                else self.self_attn(
                    positions,
                    attn_input,
                    forward_batch,
                    precomputed_qkv_gate=precomputed_qkv_gate,
                    precomputed_prepared_qkvg=precomputed_prepared_qkvg,
                    return_gated_attn=True,
                    **cs_kw,
                )
            )
            stream_mid, moe_out_a = get_b1_moe_chip_sharded_op(
                tp,
                bf16_router=bf16_router,
                num_replicas=num_rep,
                has_spec_slot=has_spec,
                replica_cost_mode=cost_mode,
                return_sharded=ret_sh_step_a,
                has_free_params=(fp_buf is not None),
            )(
                mlp._fused_decode_rank,
                stream,
                gated,
                self.self_attn._o_proj_w8k,
                self.self_attn._o_proj_scale,
                pa_alpha_arg,
                pa_beta_arg,
                pre_ff_gain,
                mlp._buffers["_w8_wpre_pair_w8k"],
                mlp._buffers["_w8_spre_pair"],
                w_router_km,
                pre_exp_gain,
                mlp._bias_2d(),
                w1_arg,
                s1_arg,
                w2_arg,
                s2_arg,
                mlp._norm_2d(),
                *rep_tail,
                *spec_tail,
                *fp_tail,
                float(self.post_attention_layernorm.eps),
                float(self.pre_feedforward_layernorm.eps),
                bool(self.pre_feedforward_layernorm.scale_input),
                float(mlp.router_multiplier),
                float(mlp.pre_expert_norm.eps),
                bool(mlp.pre_expert_norm.scale_input),
            )
            w_post_p = mlp._buffers.get("_w9_wpost_pair_w8k", mlp._buffers["_w7_wpost_w8k"])
            s_post_p = mlp._buffers.get("_w9_spost_pair", mlp._buffers["_w7_spost"])
            post_ff_gain = self.post_feedforward_layernorm.gain()
            pf_alpha_arg = self.post_feedforward_residual_gate.alpha
            pf_beta_arg = self.post_feedforward_residual_gate.beta
            assert post_ff_gain is not None
            if b2_kind == 1 and final_layernorm is not None:
                fin_gain = final_layernorm.gain()
                assert fin_gain is not None
                hid_f = get_moe_post_chip_sharded_final_norm_op(
                    tp, sharded_moe_in=ret_sh_step_a, return_stream_too=False
                )(
                    stream_mid,
                    moe_out_a,
                    w_post_p,
                    s_post_p,
                    post_ff_gain,
                    pf_alpha_arg,
                    pf_beta_arg,
                    fin_gain,
                    float(self.post_feedforward_layernorm.eps),
                    bool(self.post_feedforward_layernorm.scale_input),
                    float(final_layernorm.eps),
                    bool(final_layernorm.scale_input),
                )
                return (hid_f,)
            if b2_kind == 2 and next_input_layernorm is not None:
                next_in_gain = next_input_layernorm.gain()
                assert next_in_gain is not None
                return get_moe_post_chip_sharded_final_norm_op(
                    tp, sharded_moe_in=ret_sh_step_a, return_stream_too=True
                )(
                    stream_mid,
                    moe_out_a,
                    w_post_p,
                    s_post_p,
                    post_ff_gain,
                    pf_alpha_arg,
                    pf_beta_arg,
                    next_in_gain,
                    float(self.post_feedforward_layernorm.eps),
                    bool(self.post_feedforward_layernorm.scale_input),
                    float(next_input_layernorm.eps),
                    bool(next_input_layernorm.scale_input),
                )
        return None

    def forward(
        self,
        positions: torch.Tensor,
        stream: torch.Tensor,
        forward_batch: ForwardBatch,
        precomputed_attn_input: Optional[torch.Tensor] = None,
        next_input_layernorm: Optional[MuseSparkRMSNorm] = None,
        cos_sin_step: Optional[torch.Tensor] = None,
        precomputed_qkv_gate: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        precomputed_prepared_qkvg: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None,
        precomputed_gated: Optional[torch.Tensor] = None,
        next_layer_attn: Optional[MuseSparkAttention] = None,
        final_layernorm: Optional[MuseSparkRMSNorm] = None,
        tok_sharded_stream: bool = False,
        chip_sharded_stream: bool = False,
        pending_step_c_kv_writes: Optional[list] = None,
    ) -> (
        torch.Tensor
        | Tuple[torch.Tensor]
        | Tuple[torch.Tensor, torch.Tensor]
        | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | Tuple[torch.Tensor, torch.Tensor, None, None, None]
    ):
        n_tok = (
            positions.shape[0]
            if (tok_sharded_stream or chip_sharded_stream)
            else stream.shape[0]
        )
        cs_kw = {"cos_sin_step": cos_sin_step} if cos_sin_step is not None else {}
        is_dec_step = forward_batch.forward_mode.is_decode() or (
            getattr(self, "_decode_fuse_target_verify", False)
            and forward_batch.forward_mode.is_target_verify()
        )
        b1_kind, b2_kind = self._active_w7_bridges(
            n_tok,
            forward_batch,
            is_dec_step=is_dec_step,
            next_input_layernorm=next_input_layernorm,
            next_layer_attn=next_layer_attn,
            sharded_stream=bool(tok_sharded_stream or chip_sharded_stream),
        )
        if b1_kind > 0 or b2_kind > 0:
            from sglang_tpu.kernels.decode_collective_bridges import (
                get_attn_to_moe_bridge_op,
                get_moe_post_ag_res_norm_op,
                get_moe_post_ag_res_only_op,
                get_moe_to_attn_bridge_op,
                get_post_attn_to_moe_bridge_op,
            )

            tp = _layer_tp_size(self.self_attn.o_proj)
            if chip_sharded_stream:
                assert stream.shape[0] * (tp // 2) == n_tok, (
                    f"chip_sharded_stream shape mismatch: stream={tuple(stream.shape)}, "
                    f"n_tok={n_tok}, tp={tp}"
                )
            attn_input: Optional[torch.Tensor] = None
            if (
                precomputed_gated is None
                and precomputed_qkv_gate is None
                and precomputed_prepared_qkvg is None
            ):
                if precomputed_attn_input is not None:
                    attn_input = precomputed_attn_input
                elif self._decode_norm_layout > 0:
                    from sglang_tpu.moe.pallas.fused_rs_muse.decode_norm_router import (
                        get_muse_decode_rmsnorm_op,
                    )

                    in_gain = self.input_layernorm.gain()
                    assert in_gain is not None
                    attn_input = get_muse_decode_rmsnorm_op()(
                        stream,
                        in_gain,
                        float(self.input_layernorm.eps),
                        True,
                        bool(self.input_layernorm.scale_input),
                        True,
                    )
                else:
                    attn_input = self.input_layernorm(stream).to(torch.bfloat16)

            if b1_kind > 0:
                pa_alpha_arg = self.post_attention_residual_gate.alpha
                pa_beta_arg = self.post_attention_residual_gate.beta
                pre_ff_gain = self.pre_feedforward_layernorm.gain()
                pre_exp_gain = self.mlp.pre_expert_norm.gain()
                assert pre_ff_gain is not None and pre_exp_gain is not None
                prerouted_topk = False
                if b1_kind == 2:
                    if chip_sharded_stream:
                        sk_out = self._try_b1_moe_superkernel(
                            positions=positions,
                            stream=stream,
                            attn_input=attn_input,
                            forward_batch=forward_batch,
                            n_tok=n_tok,
                            tp=tp,
                            b2_kind=b2_kind,
                            is_dec_step=is_dec_step,
                            pa_alpha_arg=pa_alpha_arg,
                            pa_beta_arg=pa_beta_arg,
                            pre_ff_gain=pre_ff_gain,
                            pre_exp_gain=pre_exp_gain,
                            cos_sin_step=cos_sin_step,
                            cs_kw=cs_kw,
                            precomputed_qkv_gate=precomputed_qkv_gate,
                            precomputed_prepared_qkvg=precomputed_prepared_qkvg,
                            precomputed_gated=precomputed_gated,
                            next_input_layernorm=next_input_layernorm,
                            next_layer_attn=next_layer_attn,
                            final_layernorm=final_layernorm,
                            pending_step_c_kv_writes=pending_step_c_kv_writes,
                        )
                        if sk_out is not None:
                            return sk_out
                    elif (
                        not tok_sharded_stream
                        and (n_tok == 16 or (tp == 8 and n_tok == 8))
                        and b2_kind == 3
                    ):
                        sk16_out = self._try_b1_moe_replicated_b16_superkernel(
                            positions=positions,
                            stream=stream,
                            attn_input=attn_input,
                            forward_batch=forward_batch,
                            n_tok=n_tok,
                            tp=tp,
                            b2_kind=b2_kind,
                            is_dec_step=is_dec_step,
                            pa_alpha_arg=pa_alpha_arg,
                            pa_beta_arg=pa_beta_arg,
                            pre_ff_gain=pre_ff_gain,
                            pre_exp_gain=pre_exp_gain,
                            cos_sin_step=cos_sin_step,
                            cs_kw=cs_kw,
                            precomputed_qkv_gate=precomputed_qkv_gate,
                            precomputed_prepared_qkvg=precomputed_prepared_qkvg,
                            precomputed_gated=precomputed_gated,
                            next_input_layernorm=next_input_layernorm,
                            next_layer_attn=next_layer_attn,
                        )
                        if sk16_out is not None:
                            return sk16_out
                    gated = (
                        precomputed_gated
                        if precomputed_gated is not None
                        else self.self_attn(
                            positions,
                            attn_input,
                            forward_batch,
                            precomputed_qkv_gate=precomputed_qkv_gate,
                            precomputed_prepared_qkvg=precomputed_prepared_qkvg,
                            return_gated_attn=True,
                            **cs_kw,
                        )
                    )
                    w_router_km = self.mlp._buffers["_w7_wrouter_km"]
                    if chip_sharded_stream and "_w8_wpre_pair_w8k" in self.mlp._buffers:
                        from sglang_tpu.kernels.decode_collective_bridges_w8 import (
                            get_attn_to_moe_chip_sharded_bridge_op,
                        )

                        stream, scaled_logits, expert_in = (
                            get_attn_to_moe_chip_sharded_bridge_op(
                                tp, bf16_router=(w_router_km.dtype == torch.bfloat16)
                            )(
                                stream,
                                gated,
                                self.self_attn._o_proj_w8k,
                                self.self_attn._o_proj_scale,
                                pa_alpha_arg,
                                pa_beta_arg,
                                pre_ff_gain,
                                self.mlp._buffers["_w8_wpre_pair_w8k"],
                                self.mlp._buffers["_w8_spre_pair"],
                                w_router_km,
                                pre_exp_gain,
                                float(self.post_attention_layernorm.eps),
                                float(self.pre_feedforward_layernorm.eps),
                                bool(self.pre_feedforward_layernorm.scale_input),
                                float(self.mlp.router_multiplier),
                                float(self.mlp.pre_expert_norm.eps),
                                bool(self.mlp.pre_expert_norm.scale_input),
                            )
                        )
                    elif tok_sharded_stream and "_w8_wpre_pair_w8k" in self.mlp._buffers:
                        from sglang_tpu.kernels.decode_collective_bridges_w8 import (
                            get_attn_to_moe_tok_sharded_bridge_op,
                        )

                        stream, scaled_logits, expert_in = (
                            get_attn_to_moe_tok_sharded_bridge_op(
                                tp, bf16_router=(w_router_km.dtype == torch.bfloat16)
                            )(
                                stream,
                                gated,
                                self.self_attn._o_proj_w8k,
                                self.self_attn._o_proj_scale,
                                pa_alpha_arg,
                                pa_beta_arg,
                                pre_ff_gain,
                                self.mlp._buffers["_w8_wpre_pair_w8k"],
                                self.mlp._buffers["_w8_spre_pair"],
                                w_router_km,
                                pre_exp_gain,
                                float(self.post_attention_layernorm.eps),
                                float(self.pre_feedforward_layernorm.eps),
                                bool(self.pre_feedforward_layernorm.scale_input),
                                float(self.mlp.router_multiplier),
                                float(self.mlp.pre_expert_norm.eps),
                                bool(self.mlp.pre_expert_norm.scale_input),
                            )
                        )
                    elif (
                        bool(
                            _env_get("SGLANG_TPU_MUSE_DECODE_BRIDGE1_FUSE_TOPK", False)
                        )
                        and self.mlp.top_k == 8
                        and w_router_km.shape[1] == 256
                    ):
                        from sglang_tpu.kernels.decode_collective_bridges_carb import (
                            get_attn_to_moe_bridge_router_topk_op,
                        )

                        stream, scaled_logits, expert_in = (
                            get_attn_to_moe_bridge_router_topk_op(
                                tp, bf16_router=(w_router_km.dtype == torch.bfloat16)
                            )(
                                stream,
                                gated,
                                self.self_attn._o_proj_w8k,
                                self.self_attn._o_proj_scale,
                                pa_alpha_arg,
                                pa_beta_arg,
                                pre_ff_gain,
                                self.mlp._buffers["_w7_wpre_w8k"],
                                self.mlp._buffers["_w7_spre"],
                                w_router_km,
                                self.mlp._bias_2d(),
                                pre_exp_gain,
                                float(self.post_attention_layernorm.eps),
                                float(self.pre_feedforward_layernorm.eps),
                                bool(self.pre_feedforward_layernorm.scale_input),
                                float(self.mlp.router_multiplier),
                                float(self.mlp.route_eps),
                                float(self.mlp.route_scale),
                                float(self.mlp.pre_expert_norm.eps),
                                bool(self.mlp.pre_expert_norm.scale_input),
                            )
                        )
                        prerouted_topk = True
                    else:
                        stream, scaled_logits, expert_in = get_attn_to_moe_bridge_op(
                            tp, bf16_router=(w_router_km.dtype == torch.bfloat16)
                        )(
                            stream,
                            gated,
                            self.self_attn._o_proj_w8k,
                            self.self_attn._o_proj_scale,
                            pa_alpha_arg,
                            pa_beta_arg,
                            pre_ff_gain,
                            self.mlp._buffers["_w7_wpre_w8k"],
                            self.mlp._buffers["_w7_spre"],
                            w_router_km,
                            pre_exp_gain,
                            float(self.post_attention_layernorm.eps),
                            float(self.pre_feedforward_layernorm.eps),
                            bool(self.pre_feedforward_layernorm.scale_input),
                            float(self.mlp.router_multiplier),
                            float(self.mlp.pre_expert_norm.eps),
                            bool(self.mlp.pre_expert_norm.scale_input),
                        )
                else:
                    attn_out = self.self_attn(
                        positions,
                        attn_input,
                        forward_batch,
                        precomputed_qkv_gate=precomputed_qkv_gate,
                        precomputed_prepared_qkvg=precomputed_prepared_qkvg,
                        **cs_kw,
                    )
                    stream, scaled_logits, expert_in = get_post_attn_to_moe_bridge_op(
                        tp
                    )(
                        stream,
                        attn_out,
                        pa_alpha_arg,
                        pa_beta_arg,
                        pre_ff_gain,
                        self.mlp._buffers["_w7_wpre_w8k"],
                        self.mlp._buffers["_w7_spre"],
                        self.mlp._buffers["_w7_wrouter_km"],
                        pre_exp_gain,
                        float(self.post_attention_layernorm.eps),
                        float(self.pre_feedforward_layernorm.eps),
                        bool(self.pre_feedforward_layernorm.scale_input),
                        float(self.mlp.router_multiplier),
                        float(self.mlp.pre_expert_norm.eps),
                        bool(self.mlp.pre_expert_norm.scale_input),
                    )
                moe_kw = {
                    "precomputed_moe_inputs": (scaled_logits, expert_in),
                    "prerouted_router_topk": prerouted_topk,
                }
                ffn_hidden_arg, router_in_arg = None, None
            else:
                attn_out = self.self_attn(
                    positions,
                    attn_input,
                    forward_batch,
                    precomputed_qkv_gate=precomputed_qkv_gate,
                    precomputed_prepared_qkvg=precomputed_prepared_qkvg,
                    **cs_kw,
                )
                if self._decode_norm_layout >= 2:
                    from sglang_tpu.moe.pallas.fused_rs_muse.decode_norm_router import (
                        get_muse_decode_res_norm_op,
                    )

                    pre_ff_gain = self.pre_feedforward_layernorm.gain()
                    assert pre_ff_gain is not None
                    stream, router_in_arg = get_muse_decode_res_norm_op()(
                        stream,
                        attn_out,
                        self._dummy_gain,
                        self.post_attention_residual_gate.alpha,
                        self.post_attention_residual_gate.beta,
                        pre_ff_gain,
                        float(self.post_attention_layernorm.eps),
                        False,
                        False,
                        float(self.pre_feedforward_layernorm.eps),
                        False,
                    )
                else:
                    stream = self.post_attention_residual_gate(
                        stream, self.post_attention_layernorm(attn_out)
                    )
                    router_in_arg = self.pre_feedforward_layernorm(stream)
                ffn_hidden_arg = router_in_arg.to(torch.bfloat16)
                moe_kw = {}

            if b2_kind > 0:
                can_sh_fn = getattr(self.mlp, "can_return_sharded_moe_out", None)
                has_sh_op = (
                    bool(can_sh_fn(n_tok, decode=is_dec_step))
                    if callable(can_sh_fn)
                    else (
                        self.mlp.__dict__.get("_fused_decode_sharded_op") is not None
                        or self.mlp.__dict__.get("_fused_decode_v2_sharded_op")
                        is not None
                    )
                )
                want_sharded = (
                    is_dec_step
                    and n_tok >= 8 * tp
                    and (n_tok // tp) % 8 == 0
                    and has_sh_op
                )
                moe_out = self.mlp(
                    ffn_hidden_arg,
                    router_in_arg,
                    decode=is_dec_step,
                    return_unprojected_moe_out=True,
                    return_sharded_moe_out=want_sharded,
                    **moe_kw,
                )
                sharded_in = (
                    moe_out.shape[0] * tp == n_tok
                    and moe_out.shape[0] >= 8
                    and moe_out.shape[0] % 8 == 0
                )
                post_ff_gain = self.post_feedforward_layernorm.gain()
                pf_alpha_arg = self.post_feedforward_residual_gate.alpha
                pf_beta_arg = self.post_feedforward_residual_gate.beta
                assert post_ff_gain is not None
                use_qkv_epi = (
                    b2_kind == 3
                    and (
                        tp == 8
                        or (
                            tp == 16
                            and bool(tpu_envs.SGLANG_TPU_MUSE_DECODE_TP16_QKV_EPI.get())
                        )
                    )
                    and bool(self.__dict__.get("_w8_bridge_qkv_epi", False))
                    and _can_use_bridge_qkv_epi(
                        n_tok,
                        next_layer_attn,
                        cos_sin_step,
                        forward_batch,
                        max_tok=384,
                    )
                )
                if chip_sharded_stream and "_w7_wpost_w8k" in self.mlp._buffers:
                    from sglang_tpu.kernels.decode_collective_bridges_w8 import (
                        get_moe_post_chip_sharded_final_norm_op,
                        get_moe_to_attn_chip_sharded_bridge_op,
                    )

                    w_post_p = self.mlp._buffers.get(
                        "_w9_wpost_pair_w8k", self.mlp._buffers["_w7_wpost_w8k"]
                    )
                    s_post_p = self.mlp._buffers.get(
                        "_w9_spost_pair", self.mlp._buffers["_w7_spost"]
                    )
                    if b2_kind == 3:
                        assert (
                            next_input_layernorm is not None
                            and next_layer_attn is not None
                        )
                        next_in_gain = next_input_layernorm.gain()
                        assert next_in_gain is not None
                        next_has_rope = next_layer_attn.rotary_emb is not None
                        if use_qkv_epi:
                            k_sc = float(
                                getattr(next_layer_attn.attn, "k_scale_float", None)
                                or 1.0
                            )
                            v_sc = float(
                                getattr(next_layer_attn.attn, "v_scale_float", None)
                                or 1.0
                            )
                            epi_op = get_moe_to_attn_chip_sharded_bridge_op(
                                tp,
                                sharded_moe_in=sharded_in,
                                fuse_qkv_epi=True,
                                has_rope=next_has_rope,
                            )
                            if next_has_rope:
                                return epi_op(
                                    stream,
                                    moe_out,
                                    w_post_p,
                                    s_post_p,
                                    post_ff_gain,
                                    pf_alpha_arg,
                                    pf_beta_arg,
                                    next_in_gain,
                                    next_layer_attn._buffers["_w7_wqkv_w8k"],
                                    next_layer_attn._buffers["_w7_sqkv"],
                                    next_layer_attn._buffers["_w7_wgate_w8k"],
                                    next_layer_attn._buffers["_w7_sgate"],
                                    cos_sin_step,
                                    float(self.post_feedforward_layernorm.eps),
                                    bool(self.post_feedforward_layernorm.scale_input),
                                    float(next_input_layernorm.eps),
                                    bool(next_input_layernorm.scale_input),
                                    float(next_layer_attn.norm_eps),
                                    float(next_layer_attn.query_scale),
                                    k_sc,
                                    v_sc,
                                )
                            return epi_op(
                                stream,
                                moe_out,
                                w_post_p,
                                s_post_p,
                                post_ff_gain,
                                pf_alpha_arg,
                                pf_beta_arg,
                                next_in_gain,
                                next_layer_attn._buffers["_w7_wqkv_w8k"],
                                next_layer_attn._buffers["_w7_sqkv"],
                                next_layer_attn._buffers["_w7_wgate_w8k"],
                                next_layer_attn._buffers["_w7_sgate"],
                                float(self.post_feedforward_layernorm.eps),
                                bool(self.post_feedforward_layernorm.scale_input),
                                float(next_input_layernorm.eps),
                                bool(next_input_layernorm.scale_input),
                                float(next_layer_attn.norm_eps),
                                float(next_layer_attn.query_scale),
                                k_sc,
                                v_sc,
                            )
                        return get_moe_to_attn_chip_sharded_bridge_op(
                            tp, sharded_moe_in=sharded_in, fuse_qkv_epi=False
                        )(
                            stream,
                            moe_out,
                            w_post_p,
                            s_post_p,
                            post_ff_gain,
                            pf_alpha_arg,
                            pf_beta_arg,
                            next_in_gain,
                            next_layer_attn._buffers["_w7_wqkv_w8k"],
                            next_layer_attn._buffers["_w7_sqkv"],
                            next_layer_attn._buffers["_w7_wgate_w8k"],
                            next_layer_attn._buffers["_w7_sgate"],
                            float(self.post_feedforward_layernorm.eps),
                            bool(self.post_feedforward_layernorm.scale_input),
                            float(next_input_layernorm.eps),
                            bool(next_input_layernorm.scale_input),
                        )
                    if final_layernorm is not None:
                        fin_gain = final_layernorm.gain()
                        assert fin_gain is not None
                        hid_f = get_moe_post_chip_sharded_final_norm_op(
                            tp, sharded_moe_in=sharded_in, return_stream_too=False
                        )(
                            stream,
                            moe_out,
                            w_post_p,
                            s_post_p,
                            post_ff_gain,
                            pf_alpha_arg,
                            pf_beta_arg,
                            fin_gain,
                            float(self.post_feedforward_layernorm.eps),
                            bool(self.post_feedforward_layernorm.scale_input),
                            float(final_layernorm.eps),
                            bool(final_layernorm.scale_input),
                        )
                        return (hid_f,)
                    assert next_input_layernorm is not None
                    next_in_gain = next_input_layernorm.gain()
                    assert next_in_gain is not None
                    return get_moe_post_chip_sharded_final_norm_op(
                        tp, sharded_moe_in=sharded_in, return_stream_too=True
                    )(
                        stream,
                        moe_out,
                        w_post_p,
                        s_post_p,
                        post_ff_gain,
                        pf_alpha_arg,
                        pf_beta_arg,
                        next_in_gain,
                        float(self.post_feedforward_layernorm.eps),
                        bool(self.post_feedforward_layernorm.scale_input),
                        float(next_input_layernorm.eps),
                        bool(next_input_layernorm.scale_input),
                    )
                if tok_sharded_stream and "_w7_wpost_w8k" in self.mlp._buffers:
                    from sglang_tpu.kernels.decode_collective_bridges_w8 import (
                        get_moe_post_tok_sharded_final_norm_op,
                        get_moe_to_attn_tok_sharded_bridge_op,
                    )

                    assert sharded_in
                    w_post_p = self.mlp._buffers["_w7_wpost_w8k"]
                    s_post_p = self.mlp._buffers["_w7_spost"]
                    if b2_kind == 3:
                        assert (
                            next_input_layernorm is not None
                            and next_layer_attn is not None
                        )
                        next_in_gain = next_input_layernorm.gain()
                        assert next_in_gain is not None
                        next_has_rope = next_layer_attn.rotary_emb is not None
                        if use_qkv_epi:
                            k_sc = float(
                                getattr(next_layer_attn.attn, "k_scale_float", None)
                                or 1.0
                            )
                            v_sc = float(
                                getattr(next_layer_attn.attn, "v_scale_float", None)
                                or 1.0
                            )
                            epi_op = get_moe_to_attn_tok_sharded_bridge_op(
                                tp, fuse_qkv_epi=True, has_rope=next_has_rope
                            )
                            if next_has_rope:
                                return epi_op(
                                    stream,
                                    moe_out,
                                    w_post_p,
                                    s_post_p,
                                    post_ff_gain,
                                    pf_alpha_arg,
                                    pf_beta_arg,
                                    next_in_gain,
                                    next_layer_attn._buffers["_w7_wqkv_w8k"],
                                    next_layer_attn._buffers["_w7_sqkv"],
                                    next_layer_attn._buffers["_w7_wgate_w8k"],
                                    next_layer_attn._buffers["_w7_sgate"],
                                    cos_sin_step,
                                    float(self.post_feedforward_layernorm.eps),
                                    bool(self.post_feedforward_layernorm.scale_input),
                                    float(next_input_layernorm.eps),
                                    bool(next_input_layernorm.scale_input),
                                    float(next_layer_attn.norm_eps),
                                    float(next_layer_attn.query_scale),
                                    k_sc,
                                    v_sc,
                                )
                            return epi_op(
                                stream,
                                moe_out,
                                w_post_p,
                                s_post_p,
                                post_ff_gain,
                                pf_alpha_arg,
                                pf_beta_arg,
                                next_in_gain,
                                next_layer_attn._buffers["_w7_wqkv_w8k"],
                                next_layer_attn._buffers["_w7_sqkv"],
                                next_layer_attn._buffers["_w7_wgate_w8k"],
                                next_layer_attn._buffers["_w7_sgate"],
                                float(self.post_feedforward_layernorm.eps),
                                bool(self.post_feedforward_layernorm.scale_input),
                                float(next_input_layernorm.eps),
                                bool(next_input_layernorm.scale_input),
                                float(next_layer_attn.norm_eps),
                                float(next_layer_attn.query_scale),
                                k_sc,
                                v_sc,
                            )
                        return get_moe_to_attn_tok_sharded_bridge_op(
                            tp, fuse_qkv_epi=False
                        )(
                            stream,
                            moe_out,
                            w_post_p,
                            s_post_p,
                            post_ff_gain,
                            pf_alpha_arg,
                            pf_beta_arg,
                            next_in_gain,
                            next_layer_attn._buffers["_w7_wqkv_w8k"],
                            next_layer_attn._buffers["_w7_sqkv"],
                            next_layer_attn._buffers["_w7_wgate_w8k"],
                            next_layer_attn._buffers["_w7_sgate"],
                            float(self.post_feedforward_layernorm.eps),
                            bool(self.post_feedforward_layernorm.scale_input),
                            float(next_input_layernorm.eps),
                            bool(next_input_layernorm.scale_input),
                        )
                    if final_layernorm is not None:
                        fin_gain = final_layernorm.gain()
                        assert fin_gain is not None
                        hid_f = get_moe_post_tok_sharded_final_norm_op(
                            tp, return_stream_too=False
                        )(
                            stream,
                            moe_out,
                            w_post_p,
                            s_post_p,
                            post_ff_gain,
                            pf_alpha_arg,
                            pf_beta_arg,
                            fin_gain,
                            float(self.post_feedforward_layernorm.eps),
                            bool(self.post_feedforward_layernorm.scale_input),
                            float(final_layernorm.eps),
                            bool(final_layernorm.scale_input),
                        )
                        return (hid_f,)
                    assert next_input_layernorm is not None
                    next_in_gain = next_input_layernorm.gain()
                    assert next_in_gain is not None
                    return get_moe_post_tok_sharded_final_norm_op(
                        tp, return_stream_too=True
                    )(
                        stream,
                        moe_out,
                        w_post_p,
                        s_post_p,
                        post_ff_gain,
                        pf_alpha_arg,
                        pf_beta_arg,
                        next_in_gain,
                        float(self.post_feedforward_layernorm.eps),
                        bool(self.post_feedforward_layernorm.scale_input),
                        float(next_input_layernorm.eps),
                        bool(next_input_layernorm.scale_input),
                    )

                w_post = self.mlp._buffers["_w7_wpost_w8k"]
                s_post = self.mlp._buffers["_w7_spost"]
                if b2_kind == 3:
                    assert (
                        next_input_layernorm is not None and next_layer_attn is not None
                    )
                    next_in_gain = next_input_layernorm.gain()
                    assert next_in_gain is not None
                    wq_next = next_layer_attn._buffers["_w7_wqkv_w8k"]
                    wg_next = next_layer_attn._buffers["_w7_wgate_w8k"]
                    if use_qkv_epi:
                        from sglang_tpu.kernels.decode_collective_bridges_w8 import (
                            get_moe_to_attn_qkv_epi_bridge_op,
                        )

                        next_has_rope = next_layer_attn.rotary_emb is not None
                        k_sc = float(
                            getattr(next_layer_attn.attn, "k_scale_float", None) or 1.0
                        )
                        v_sc = float(
                            getattr(next_layer_attn.attn, "v_scale_float", None) or 1.0
                        )
                        epi_w7_op = get_moe_to_attn_qkv_epi_bridge_op(
                            tp, sharded_moe_in=sharded_in, has_rope=next_has_rope
                        )
                        if next_has_rope:
                            return epi_w7_op(
                                stream,
                                moe_out,
                                w_post,
                                s_post,
                                post_ff_gain,
                                pf_alpha_arg,
                                pf_beta_arg,
                                next_in_gain,
                                wq_next,
                                next_layer_attn._buffers["_w7_sqkv"],
                                wg_next,
                                next_layer_attn._buffers["_w7_sgate"],
                                cos_sin_step,
                                float(self.post_feedforward_layernorm.eps),
                                bool(self.post_feedforward_layernorm.scale_input),
                                float(next_input_layernorm.eps),
                                bool(next_input_layernorm.scale_input),
                                float(next_layer_attn.norm_eps),
                                float(next_layer_attn.query_scale),
                                k_sc,
                                v_sc,
                            )
                        return epi_w7_op(
                            stream,
                            moe_out,
                            w_post,
                            s_post,
                            post_ff_gain,
                            pf_alpha_arg,
                            pf_beta_arg,
                            next_in_gain,
                            wq_next,
                            next_layer_attn._buffers["_w7_sqkv"],
                            wg_next,
                            next_layer_attn._buffers["_w7_sgate"],
                            float(self.post_feedforward_layernorm.eps),
                            bool(self.post_feedforward_layernorm.scale_input),
                            float(next_input_layernorm.eps),
                            bool(next_input_layernorm.scale_input),
                            float(next_layer_attn.norm_eps),
                            float(next_layer_attn.query_scale),
                            k_sc,
                            v_sc,
                        )
                    overlap_epi = bool(
                        _env_get("SGLANG_TPU_MUSE_DECODE_BRIDGE2_OVERLAP_EPI", False)
                    )
                    use_direct_b2_epi = (
                        bool(
                            _env_get(
                                "SGLANG_TPU_MUSE_DECODE_BRIDGE2_DIRECT_QKV_EPI", False
                            )
                        )
                        and tp == 8
                        and getattr(next_layer_attn, "decode_qkv_epilogue", False)
                        and wq_next.shape[1] == 1280
                        and wg_next.shape[1] == 1024
                        and can_use_muse_decode_qkv_epilogue(
                            torch.empty(
                                (n_tok, 1280),
                                dtype=torch.bfloat16,
                                device=stream.device,
                            ),
                            next_layer_attn.rotary_emb,
                            next_layer_attn.attn,
                            forward_batch,
                        )
                    )
                    if use_direct_b2_epi:
                        from sglang_tpu.kernels.decode_collective_bridges_carb import (
                            get_moe_to_attn_direct_qkv_epi_bridge_op,
                        )

                        next_has_rope = next_layer_attn.rotary_emb is not None
                        k_sc_n = float(
                            getattr(next_layer_attn.attn, "k_scale_float", None) or 1.0
                        )
                        v_sc_n = float(
                            getattr(next_layer_attn.attn, "v_scale_float", None) or 1.0
                        )
                        if next_has_rope:
                            cs_n = (
                                cos_sin_step
                                if cos_sin_step is not None
                                else get_decode_rope_step_op()(
                                    positions.to(torch.int32),
                                    next_layer_attn.rotary_emb.cos_sin_pairs,
                                )
                            )
                            return get_moe_to_attn_direct_qkv_epi_bridge_op(
                                tp,
                                has_rope=True,
                                sharded_moe_in=sharded_in,
                                overlap_mate_epi=overlap_epi,
                            )(
                                stream,
                                moe_out,
                                w_post,
                                s_post,
                                post_ff_gain,
                                pf_alpha_arg,
                                pf_beta_arg,
                                next_in_gain,
                                wq_next,
                                next_layer_attn._buffers["_w7_sqkv"],
                                wg_next,
                                next_layer_attn._buffers["_w7_sgate"],
                                cs_n,
                                float(self.post_feedforward_layernorm.eps),
                                bool(self.post_feedforward_layernorm.scale_input),
                                float(next_input_layernorm.eps),
                                bool(next_input_layernorm.scale_input),
                                float(next_layer_attn.norm_eps),
                                float(next_layer_attn.query_scale),
                                k_sc_n,
                                v_sc_n,
                            )
                        return get_moe_to_attn_direct_qkv_epi_bridge_op(
                            tp,
                            has_rope=False,
                            sharded_moe_in=sharded_in,
                            overlap_mate_epi=overlap_epi,
                        )(
                            stream,
                            moe_out,
                            w_post,
                            s_post,
                            post_ff_gain,
                            pf_alpha_arg,
                            pf_beta_arg,
                            next_in_gain,
                            wq_next,
                            next_layer_attn._buffers["_w7_sqkv"],
                            wg_next,
                            next_layer_attn._buffers["_w7_sgate"],
                            float(self.post_feedforward_layernorm.eps),
                            bool(self.post_feedforward_layernorm.scale_input),
                            float(next_input_layernorm.eps),
                            bool(next_input_layernorm.scale_input),
                            float(next_layer_attn.norm_eps),
                            float(next_layer_attn.query_scale),
                            k_sc_n,
                            v_sc_n,
                        )
                    if (
                        overlap_epi
                        and wq_next.shape[1] % 256 == 0
                        and wg_next.shape[1] % 256 == 0
                    ):
                        from sglang_tpu.kernels.decode_collective_bridges_carb import (
                            get_moe_to_attn_overlap_bridge_op,
                        )

                        b2_op = get_moe_to_attn_overlap_bridge_op(
                            tp, sharded_moe_in=sharded_in
                        )
                    else:
                        b2_op = get_moe_to_attn_bridge_op(tp, sharded_moe_in=sharded_in)
                    return b2_op(
                        stream,
                        moe_out,
                        w_post,
                        s_post,
                        post_ff_gain,
                        pf_alpha_arg,
                        pf_beta_arg,
                        next_in_gain,
                        wq_next,
                        next_layer_attn._buffers["_w7_sqkv"],
                        wg_next,
                        next_layer_attn._buffers["_w7_sgate"],
                        float(self.post_feedforward_layernorm.eps),
                        bool(self.post_feedforward_layernorm.scale_input),
                        float(next_input_layernorm.eps),
                        bool(next_input_layernorm.scale_input),
                    )
                if b2_kind == 2:
                    assert next_input_layernorm is not None
                    next_in_gain = next_input_layernorm.gain()
                    assert next_in_gain is not None
                    return get_moe_post_ag_res_norm_op(tp, sharded_moe_in=sharded_in)(
                        stream,
                        moe_out,
                        w_post,
                        s_post,
                        post_ff_gain,
                        pf_alpha_arg,
                        pf_beta_arg,
                        next_in_gain,
                        float(self.post_feedforward_layernorm.eps),
                        bool(self.post_feedforward_layernorm.scale_input),
                        float(next_input_layernorm.eps),
                        bool(next_input_layernorm.scale_input),
                    )
                if final_layernorm is not None:
                    from sglang_tpu.kernels.decode_collective_bridges_w8 import (
                        get_moe_post_ag_final_norm_op,
                    )

                    fin_gain = final_layernorm.gain()
                    assert fin_gain is not None
                    hid_w7 = get_moe_post_ag_final_norm_op(
                        tp, sharded_moe_in=sharded_in
                    )(
                        stream,
                        moe_out,
                        w_post,
                        s_post,
                        post_ff_gain,
                        pf_alpha_arg,
                        pf_beta_arg,
                        fin_gain,
                        float(self.post_feedforward_layernorm.eps),
                        bool(self.post_feedforward_layernorm.scale_input),
                        float(final_layernorm.eps),
                        bool(final_layernorm.scale_input),
                    )
                    return (hid_w7,)
                return get_moe_post_ag_res_only_op(tp, sharded_moe_in=sharded_in)(
                    stream,
                    moe_out,
                    w_post,
                    s_post,
                    post_ff_gain,
                    pf_alpha_arg,
                    pf_beta_arg,
                    float(self.post_feedforward_layernorm.eps),
                    bool(self.post_feedforward_layernorm.scale_input),
                )

            # b1_kind > 0 with b2_kind == 0: reuse standard post-FFN epilogue.
            use_tp_ag_fuse = (
                self._decode_norm_layout >= 4
                and self.mlp.tp_size > 1
                and 16 <= n_tok <= _decode_max_tokens()
                and n_tok % 16 == 0
                and not self.mlp.fuses_post_gather(n_tok, decode=is_dec_step)
            )
            ffn_output = self.mlp(
                None,
                None,
                return_untransposed_gather=use_tp_ag_fuse,
                decode=is_dec_step,
                **moe_kw,
            )
            if self._decode_norm_layout > 0:
                from sglang_tpu.moe.pallas.fused_rs_muse.decode_norm_router import (
                    get_muse_decode_res_norm_op,
                    get_muse_decode_res_only_op,
                    get_muse_decode_rmsnorm_op,
                    get_muse_decode_tp_ag_res_norm_op,
                    get_muse_decode_tp_ag_res_only_op,
                )

                post_ff_gain = self.post_feedforward_layernorm.gain()
                assert post_ff_gain is not None
                if use_tp_ag_fuse:
                    if (
                        self._decode_norm_layout >= 5
                        and next_input_layernorm is not None
                    ):
                        next_in_gain = next_input_layernorm.gain()
                        assert next_in_gain is not None
                        return get_muse_decode_tp_ag_res_norm_op()(
                            stream,
                            ffn_output,
                            post_ff_gain,
                            self.post_feedforward_residual_gate.alpha,
                            self.post_feedforward_residual_gate.beta,
                            next_in_gain,
                            int(self.mlp.tp_size),
                            float(self.post_feedforward_layernorm.eps),
                            True,
                            bool(self.post_feedforward_layernorm.scale_input),
                            float(next_input_layernorm.eps),
                            True,
                        )
                    return get_muse_decode_tp_ag_res_only_op()(
                        stream,
                        ffn_output,
                        post_ff_gain,
                        self.post_feedforward_residual_gate.alpha,
                        self.post_feedforward_residual_gate.beta,
                        int(self.mlp.tp_size),
                        float(self.post_feedforward_layernorm.eps),
                        True,
                        bool(self.post_feedforward_layernorm.scale_input),
                    )
                if (
                    self._decode_norm_layout in (3, 5)
                    and next_input_layernorm is not None
                ):
                    next_in_gain = next_input_layernorm.gain()
                    assert next_in_gain is not None
                    return get_muse_decode_res_norm_op()(
                        stream,
                        ffn_output,
                        post_ff_gain,
                        self.post_feedforward_residual_gate.alpha,
                        self.post_feedforward_residual_gate.beta,
                        next_in_gain,
                        float(self.post_feedforward_layernorm.eps),
                        True,
                        bool(self.post_feedforward_layernorm.scale_input),
                        float(next_input_layernorm.eps),
                        True,
                    )
                if self._decode_norm_layout >= 2:
                    return get_muse_decode_res_only_op()(
                        stream,
                        ffn_output,
                        post_ff_gain,
                        self.post_feedforward_residual_gate.alpha,
                        self.post_feedforward_residual_gate.beta,
                        float(self.post_feedforward_layernorm.eps),
                        True,
                        bool(self.post_feedforward_layernorm.scale_input),
                    )
                pf_norm = get_muse_decode_rmsnorm_op()(
                    ffn_output,
                    post_ff_gain,
                    float(self.post_feedforward_layernorm.eps),
                    True,
                    bool(self.post_feedforward_layernorm.scale_input),
                    False,
                )
                return self.post_feedforward_residual_gate(stream, pf_norm)
            return self.post_feedforward_residual_gate(
                stream, self.post_feedforward_layernorm(ffn_output)
            )
        if (
            self._decode_norm_layout > 0
            and forward_batch.forward_mode != ForwardMode.EXTEND
            and 8 <= n_tok <= _decode_max_tokens()
            and n_tok % 8 == 0
        ):
            from sglang_tpu.moe.pallas.fused_rs_muse.decode_norm_router import (
                get_muse_decode_res_norm_op,
                get_muse_decode_res_only_op,
                get_muse_decode_rmsnorm_op,
                get_muse_decode_tp_ag_res_norm_op,
                get_muse_decode_tp_ag_res_only_op,
            )

            if precomputed_attn_input is not None:
                attn_input = precomputed_attn_input
            else:
                in_gain = self.input_layernorm.gain()
                assert in_gain is not None
                attn_input = get_muse_decode_rmsnorm_op()(
                    stream,
                    in_gain,
                    float(self.input_layernorm.eps),
                    True,
                    bool(self.input_layernorm.scale_input),
                    True,
                )
            ar_mode = self._active_fused_o_proj_ar_mode(n_tok, forward_batch)
            if ar_mode in (1, 3):
                stream, router_input, ffn_hidden = self._run_fused_o_proj_ar_post_attn(
                    ar_mode, positions, attn_input, stream, forward_batch, **cs_kw
                )
            else:
                if ar_mode == 2:
                    from sglang_tpu.moe.pallas.fused_rs_muse.attn_ar_res_norm import (
                        get_muse_decode_bridge_ar_op,
                    )

                    o_part = self.self_attn(
                        positions,
                        attn_input,
                        forward_batch,
                        skip_o_proj_all_reduce=True,
                        **cs_kw,
                    )
                    attn_output = get_muse_decode_bridge_ar_op(
                        _layer_tp_size(self.self_attn.o_proj)
                    )(o_part)
                else:
                    attn_output = self.self_attn(
                        positions, attn_input, forward_batch, **cs_kw
                    )
                if self._decode_norm_layout >= 2:
                    pre_ff_gain = self.pre_feedforward_layernorm.gain()
                    assert pre_ff_gain is not None
                    stream, router_input = get_muse_decode_res_norm_op()(
                        stream,
                        attn_output,
                        self._dummy_gain,
                        self.post_attention_residual_gate.alpha,
                        self.post_attention_residual_gate.beta,
                        pre_ff_gain,
                        float(self.post_attention_layernorm.eps),
                        False,
                        False,
                        float(self.pre_feedforward_layernorm.eps),
                        False,
                    )
                else:
                    pa_norm = get_muse_decode_rmsnorm_op()(
                        attn_output,
                        self._dummy_gain,
                        float(self.post_attention_layernorm.eps),
                        False,
                        False,
                        False,
                    )
                    stream = self.post_attention_residual_gate(stream, pa_norm)
                    pre_ff_gain = self.pre_feedforward_layernorm.gain()
                    assert pre_ff_gain is not None
                    router_input = get_muse_decode_rmsnorm_op()(
                        stream,
                        pre_ff_gain,
                        float(self.pre_feedforward_layernorm.eps),
                        True,
                        bool(self.pre_feedforward_layernorm.scale_input),
                        False,
                    )
                ffn_hidden = router_input.to(torch.bfloat16)
            is_dec_step = forward_batch.forward_mode.is_decode() or (
                getattr(self, "_decode_fuse_target_verify", False)
                and forward_batch.forward_mode.is_target_verify()
            )
            use_tp_ag_fuse = (
                self._decode_norm_layout >= 4
                and self.mlp.tp_size > 1
                and 16 <= n_tok <= _decode_max_tokens()
                and n_tok % 16 == 0
                and not self.mlp.fuses_post_gather(n_tok, decode=is_dec_step)
            )
            ffn_output = self.mlp(
                ffn_hidden,
                router_input,
                return_untransposed_gather=use_tp_ag_fuse,
                decode=is_dec_step,
            )
            post_ff_gain = self.post_feedforward_layernorm.gain()
            assert post_ff_gain is not None
            if use_tp_ag_fuse:
                if self._decode_norm_layout >= 5 and next_input_layernorm is not None:
                    next_in_gain = next_input_layernorm.gain()
                    assert next_in_gain is not None
                    return get_muse_decode_tp_ag_res_norm_op()(
                        stream,
                        ffn_output,
                        post_ff_gain,
                        self.post_feedforward_residual_gate.alpha,
                        self.post_feedforward_residual_gate.beta,
                        next_in_gain,
                        int(self.mlp.tp_size),
                        float(self.post_feedforward_layernorm.eps),
                        True,
                        bool(self.post_feedforward_layernorm.scale_input),
                        float(next_input_layernorm.eps),
                        True,
                    )
                return get_muse_decode_tp_ag_res_only_op()(
                    stream,
                    ffn_output,
                    post_ff_gain,
                    self.post_feedforward_residual_gate.alpha,
                    self.post_feedforward_residual_gate.beta,
                    int(self.mlp.tp_size),
                    float(self.post_feedforward_layernorm.eps),
                    True,
                    bool(self.post_feedforward_layernorm.scale_input),
                )
            if self._decode_norm_layout in (3, 5) and next_input_layernorm is not None:
                next_in_gain = next_input_layernorm.gain()
                assert next_in_gain is not None
                return get_muse_decode_res_norm_op()(
                    stream,
                    ffn_output,
                    post_ff_gain,
                    self.post_feedforward_residual_gate.alpha,
                    self.post_feedforward_residual_gate.beta,
                    next_in_gain,
                    float(self.post_feedforward_layernorm.eps),
                    True,
                    bool(self.post_feedforward_layernorm.scale_input),
                    float(next_input_layernorm.eps),
                    True,
                )
            if self._decode_norm_layout >= 2:
                return get_muse_decode_res_only_op()(
                    stream,
                    ffn_output,
                    post_ff_gain,
                    self.post_feedforward_residual_gate.alpha,
                    self.post_feedforward_residual_gate.beta,
                    float(self.post_feedforward_layernorm.eps),
                    True,
                    bool(self.post_feedforward_layernorm.scale_input),
                )
            pf_norm = get_muse_decode_rmsnorm_op()(
                ffn_output,
                post_ff_gain,
                float(self.post_feedforward_layernorm.eps),
                True,
                bool(self.post_feedforward_layernorm.scale_input),
                False,
            )
            return self.post_feedforward_residual_gate(stream, pf_norm)

        attn_input = self.input_layernorm(stream).to(torch.bfloat16)
        ar_mode = self._active_fused_o_proj_ar_mode(n_tok, forward_batch)
        if ar_mode in (1, 3):
            stream, router_input, ffn_hidden = self._run_fused_o_proj_ar_post_attn(
                ar_mode, positions, attn_input, stream, forward_batch, **cs_kw
            )
        else:
            if ar_mode == 2:
                from sglang_tpu.moe.pallas.fused_rs_muse.attn_ar_res_norm import (
                    get_muse_decode_bridge_ar_op,
                )

                o_part = self.self_attn(
                    positions,
                    attn_input,
                    forward_batch,
                    skip_o_proj_all_reduce=True,
                    **cs_kw,
                )
                attn_output = get_muse_decode_bridge_ar_op(
                    _layer_tp_size(self.self_attn.o_proj)
                )(o_part)
            else:
                attn_output = self.self_attn(
                    positions, attn_input, forward_batch, **cs_kw
                )
            stream = self.post_attention_residual_gate(
                stream, self.post_attention_layernorm(attn_output)
            )
            router_input = self.pre_feedforward_layernorm(stream)
            ffn_hidden = router_input.to(torch.bfloat16)
        ffn_output = self.mlp(
            ffn_hidden,
            router_input,
            decode=(
                forward_batch.forward_mode.is_decode()
                or (
                    getattr(self, "_decode_fuse_target_verify", False)
                    and forward_batch.forward_mode.is_target_verify()
                )
            ),
        )
        return self.post_feedforward_residual_gate(
            stream, self.post_feedforward_layernorm(ffn_output)
        )


class MuseSparkTextModel(nn.Module):
    def __init__(
        self,
        config: MuseSparkTextConfig,
        max_position: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()
        self.config = config
        self.pp_group = get_parallel().pp_group
        # Layers whose output stream feeds a DFlash draft.
        self.layers_to_capture: List[int] = []
        # SGLANG_TPU_MUSE_KERNEL_ENABLED: set after weight loading when the layout fits.
        self.prefill_kernels = None
        self._capture_slots_by_layer: dict[int, List[int]] = {}
        self.embed_tokens = (
            VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=None,
                prefix=add_prefix("embed_tokens", prefix),
            )
            if self.pp_group.is_first_rank
            else PPMissingLayer()
        )
        rope_thetas = {theta for theta in config.layer_rope_theta if theta > 0}
        if len(rope_thetas) != 1:
            raise NotImplementedError("Muse Spark RoPE layers must share one theta")
        self.rotary_emb = MuseSparkInterleavedRope(
            config.head_dim, max_position=max_position, base=rope_thetas.pop()
        )
        self.layers, self.start_layer, self.end_layer = make_pp_layers(
            config.num_hidden_layers,
            lambda idx, prefix: MuseSparkDecoderLayer(
                config,
                idx,
                self.rotary_emb,
                quant_config,
                prefix,
            ),
            prefix=add_prefix("layers", prefix),
        )
        # The output norm's gain is centered on zero, unlike the per-layer norms.
        self.norm = (
            MuseSparkRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if self.pp_group.is_last_rank
            else PPMissingLayer()
        )
        self._decode_norm_layout = (
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_NORM_LAYOUT.get()
        )
        self._decode_col_bridges = int(
            _env_get("SGLANG_TPU_MUSE_DECODE_COLLECTIVE_BRIDGES", 0)
        )

    def _set_layers_to_capture(self, layer_ids: List[int]) -> None:
        self.layers_to_capture = list(layer_ids)
        slots_by_layer: dict[int, List[int]] = {}
        for slot, layer_id in enumerate(self.layers_to_capture):
            if self.start_layer <= layer_id < self.end_layer:
                slots_by_layer.setdefault(layer_id, []).append(slot)
        self._capture_slots_by_layer = slots_by_layer

    def _relay_aux_hidden_states(
        self,
        incoming: Optional[torch.Tensor],
        captured_by_slot: dict[int, torch.Tensor],
        num_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        hidden = self.config.hidden_size
        if incoming is None:
            unfilled = torch.zeros(
                (num_tokens, hidden), dtype=torch.bfloat16, device=device
            )
            slots = [unfilled] * len(self.layers_to_capture)
        else:
            slots = list(torch.split(incoming, hidden, dim=-1))
        for slot, captured in captured_by_slot.items():
            slots[slot] = captured
        return torch.cat(slots, dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor] | PPProxyTensors:
        n_tok_in = input_ids.shape[0] if input_embeds is None else input_embeds.shape[0]
        prefill_handles_in = (
            self.prefill_kernels is not None
            and self.prefill_kernels.handles(
                forward_batch, n_tok_in, capturing=bool(self.layers_to_capture)
            )
        )
        col_bridges_mode = self.__dict__.get("_decode_col_bridges", 0)
        pallas_emb_tp = int(self.__dict__.get("_w8_pallas_embed_tp", 0))
        is_dec_or_verify = forward_batch is not None and (
            forward_batch.forward_mode.is_decode()
            or (
                forward_batch.forward_mode.is_target_verify()
                and bool(_env_get("SGLANG_TPU_MUSE_VERIFY_BRIDGE_LIFT", False))
            )
        )
        can_pallas_embed = (
            pallas_emb_tp in (8, 16)
            and bool(self.__dict__.get("_w8_pallas_embed_enabled", False))
            and self.pp_group.is_first_rank
            and self.pp_group.is_last_rank
            and input_embeds is None
            and not prefill_handles_in
            and is_dec_or_verify
            and (n_tok_in == 8 or (16 <= n_tok_in <= 384 and n_tok_in % 16 == 0))
        )
        br_max_tok = _decode_bridge_max_tokens()
        chip_sh_tp = int(self.__dict__.get("_w9_chip_sharded_tp", 8))
        chip_sh_batches = self.__dict__.get("_w9_chip_sharded_batches", ())
        use_chip_sharded_relay = (
            bool(self.__dict__.get("_w9_chip_sharded_bridges", False))
            and bool(self.__dict__.get("_w9_all_layers_chip_sharded_ready", False))
            and chip_sh_tp in (8, 16)
            and self.pp_group.is_first_rank
            and self.pp_group.is_last_rank
            and not prefill_handles_in
            and is_dec_or_verify
            and isinstance(col_bridges_mode, int)
            and col_bridges_mode == 3
            and n_tok_in <= br_max_tok
            and n_tok_in in chip_sh_batches
            and (self.end_layer - 1) not in self._capture_slots_by_layer
            and (
                not bool(_env_get("SGLANG_TPU_MUSE_MK_M2M", False))
                or bool(self.__dict__.get("_w9_m2m_supports_chip_sh", False))
            )
            and int(os.environ.get("WORLD_SIZE", str(chip_sh_tp))) == chip_sh_tp
        )
        tok_sh_batches = self.__dict__.get("_w8_tok_sharded_batches", (64, 128))
        use_tok_sharded_relay = (
            not use_chip_sharded_relay
            and bool(self.__dict__.get("_w8_tok_sharded_bridges", False))
            and bool(self.__dict__.get("_w8_all_layers_tok_sharded_ready", False))
            and self.pp_group.is_first_rank
            and self.pp_group.is_last_rank
            and not prefill_handles_in
            and is_dec_or_verify
            and isinstance(col_bridges_mode, int)
            and col_bridges_mode == 3
            and n_tok_in <= br_max_tok
            and n_tok_in in tok_sh_batches
            and (self.end_layer - 1) not in self._capture_slots_by_layer
        )
        if can_pallas_embed or use_tok_sharded_relay or use_chip_sharded_relay:
            from sglang.srt.runtime_context import get_forward

            if getattr(get_forward(), "sp_active", False):
                can_pallas_embed = False
                use_tok_sharded_relay = False
                use_chip_sharded_relay = False

        carried_attn_input: Optional[torch.Tensor] = None
        if self.pp_group.is_first_rank:
            if (
                input_embeds is None
                and self.prefill_kernels is not None
                and getattr(self.prefill_kernels, "sp_entry", False)
                and prefill_handles_in
            ):
                stream = self.prefill_kernels.embed_stream(self, input_ids)
            elif (
                use_chip_sharded_relay
                and can_pallas_embed
                and pallas_emb_tp == chip_sh_tp
                and chip_sh_tp in (8, 16)
                and n_tok_in >= (32 if chip_sh_tp == 8 else 64)
            ):
                from sglang_tpu.kernels.decode_collective_bridges_w8 import (
                    get_decode_vocab_embed_chip_sharded_op,
                )

                l0_ln = self.layers[self.start_layer].input_layernorm
                l0_gain = l0_ln.gain()
                assert l0_gain is not None
                stream, carried_attn_input = get_decode_vocab_embed_chip_sharded_op(
                    chip_sh_tp
                )(
                    input_ids.to(torch.int32),
                    self.embed_tokens.weight,
                    l0_gain,
                    float(self.config.rms_norm_eps),
                    float(l0_ln.eps),
                    bool(l0_ln.scale_input),
                )
            elif use_chip_sharded_relay:
                from sglang_tpu.rank_local import rank_slice

                if input_embeds is None:
                    embeds = self.embed_tokens(input_ids)
                    input_embeds = _rms_normalize(embeds, self.config.rms_norm_eps).to(
                        embeds.dtype
                    )
                full_stream = input_embeds.float()
                carried_attn_input = (
                    self.layers[self.start_layer]
                    .input_layernorm(full_stream)
                    .to(torch.bfloat16)
                )
                n_chips_tp = chip_sh_tp // 2
                stream = rank_slice(
                    full_stream,
                    0,
                    n_chips_tp,
                    get_parallel().tp_rank // 2,
                    rank_stride=2,
                ).contiguous()
            elif use_tok_sharded_relay and can_pallas_embed and pallas_emb_tp == 8:
                from sglang_tpu.kernels.decode_collective_bridges_w8 import (
                    get_decode_vocab_embed_tok_sharded_op,
                )

                l0_ln = self.layers[self.start_layer].input_layernorm
                l0_gain = l0_ln.gain()
                assert l0_gain is not None
                stream, carried_attn_input = get_decode_vocab_embed_tok_sharded_op(8)(
                    input_ids.to(torch.int32),
                    self.embed_tokens.weight,
                    l0_gain,
                    float(self.config.rms_norm_eps),
                    float(l0_ln.eps),
                    bool(l0_ln.scale_input),
                )
            elif use_tok_sharded_relay:
                from sglang_tpu.rank_local import rank_slice

                if input_embeds is None:
                    embeds = self.embed_tokens(input_ids)
                    input_embeds = _rms_normalize(embeds, self.config.rms_norm_eps).to(
                        embeds.dtype
                    )
                full_stream = input_embeds.float()
                carried_attn_input = (
                    self.layers[self.start_layer]
                    .input_layernorm(full_stream)
                    .to(torch.bfloat16)
                )
                stream = rank_slice(
                    full_stream, 0, 8, get_parallel().tp_rank
                ).contiguous()
            elif can_pallas_embed:
                from sglang_tpu.kernels.decode_collective_bridges_w8 import (
                    get_decode_vocab_embed_norm_op,
                )

                stream = get_decode_vocab_embed_norm_op(pallas_emb_tp)(
                    input_ids.to(torch.int32),
                    self.embed_tokens.weight,
                    float(self.config.rms_norm_eps),
                )
            else:
                if input_embeds is None:
                    # The decoder reads weightless-RMS-normalized token embeddings.
                    embeds = self.embed_tokens(input_ids)
                    input_embeds = _rms_normalize(embeds, self.config.rms_norm_eps).to(
                        embeds.dtype
                    )
                stream = input_embeds.float()
            aux_buffer = None
        else:
            assert pp_proxy_tensors is not None
            stream = (
                pp_proxy_tensors.tensors["hidden_states"].float()
                + pp_proxy_tensors.tensors["residual"].float()
            )
            aux_buffer = pp_proxy_tensors.tensors.get(AUX_HIDDEN_STATES_PROXY_KEY)

        captured_by_slot: dict[int, torch.Tensor] = {}
        kernels = self.prefill_kernels
        if kernels is not None and not kernels.handles(
            forward_batch, stream.shape[0], capturing=bool(self.layers_to_capture)
        ):
            kernels = None
        l61_prune = bool(_env_get("SGLANG_TPU_MUSE_L61_PRUNE", False))
        can_prune_exit = (
            self.pp_group.is_last_rank
            and not self.layers_to_capture
            and forward_batch.forward_mode == ForwardMode.EXTEND
            and not getattr(forward_batch, "return_logprob", False)
            and getattr(forward_batch, "capture_hidden_mode", CaptureHiddenMode.NULL)
            == CaptureHiddenMode.NULL
            and getattr(forward_batch, "token_indices_to_pool", None) is None
            and getattr(forward_batch, "multi_item_delimiter_indices", None) is None
            and getattr(forward_batch, "extend_seq_lens", None) is not None
            and get_parallel().attn_cp_size == 1
        )
        exit_last_index = None
        if can_prune_exit and (
            l61_prune
            or (
                kernels is not None
                and kernels.exit_owner_gather
                and kernels.end == self.end_layer
            )
        ):
            exit_last_index = (
                torch.cumsum(
                    forward_batch.extend_seq_lens,
                    dim=0,
                    dtype=torch.int32,
                )
                - 1
            )
        n_tok_eff = (
            positions.shape[0]
            if (use_tok_sharded_relay or use_chip_sharded_relay)
            else stream.shape[0]
        )
        use_cross_layer_norm = (
            self._decode_norm_layout in (3, 5)
            and forward_batch.forward_mode != ForwardMode.EXTEND
            and kernels is None
            and 8 <= n_tok_eff <= _decode_max_tokens()
            and n_tok_eff % 8 == 0
        )
        use_w7_relay = (
            isinstance(col_bridges_mode, int)
            and col_bridges_mode in (2, 3, 4, 5, 6)
            and forward_batch.forward_mode != ForwardMode.EXTEND
            and kernels is None
            and 8 <= n_tok_eff <= _decode_bridge_max_tokens()
            and n_tok_eff % 8 == 0
        )
        carried_qkv_gate: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        carried_prepared_qkvg: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None
        carried_gated: Optional[torch.Tensor] = None
        final_hidden: Optional[torch.Tensor] = None
        cos_sin_step: Optional[torch.Tensor] = None
        if (
            kernels is None
            and forward_batch is not None
            and forward_batch.forward_mode != ForwardMode.EXTEND
            and stream.ndim == 2
            and getattr(self.rotary_emb, "cos_sin_pairs", None) is not None
        ):
            first_rope_attn = None
            for i in range(self.start_layer, self.end_layer):
                sa = getattr(self.layers[i], "self_attn", None)
                if (
                    sa is not None
                    and getattr(sa, "decode_qkv_epilogue", False)
                    and getattr(sa, "rotary_emb", None) is not None
                ):
                    first_rope_attn = sa
                    break
            if first_rope_attn is not None and can_use_muse_decode_rope_step(
                n_tok_eff,
                self.rotary_emb,
                first_rope_attn.attn,
                forward_batch,
            ):
                cos_sin_step = get_decode_rope_step_op()(
                    positions.to(torch.int32), self.rotary_emb.cos_sin_pairs
                )
        cs_layer_kw = {"cos_sin_step": cos_sin_step} if cos_sin_step is not None else {}
        fuse_final_ln = bool(self.__dict__.get("_w8_fuse_final_norm", False))
        if (
            kernels is None
            and is_dec_or_verify
            and bool(_env_get("SGLANG_TPU_MUSE_MK_M2M", False))
        ):
            from sglang_tpu.muse_megakernel.decode.m2m.ops import (
                maybe_forward_m2m_decode,
            )

            m2m_extra_kw = (
                {"chip_sharded_stream": use_chip_sharded_relay}
                if bool(self.__dict__.get("_w9_m2m_supports_chip_sh", False))
                else {}
            )
            m2m_hidden = maybe_forward_m2m_decode(
                self,
                positions=positions,
                stream=stream,
                carried_attn_input=carried_attn_input,
                cos_sin_step=cos_sin_step,
                forward_batch=forward_batch,
                tok_sharded_stream=use_tok_sharded_relay,
                **m2m_extra_kw,
            )
            if m2m_hidden is not None:
                return m2m_hidden
        layer_idx = self.start_layer
        pending_step_c_kv_writes: list = []
        while layer_idx < self.end_layer:
            if kernels is not None and layer_idx == kernels.start:
                stream = kernels(
                    self,
                    positions,
                    stream,
                    forward_batch,
                    exit_last_index=exit_last_index,
                    captured_by_slot=captured_by_slot
                    if self.layers_to_capture
                    else None,
                )
                carried_attn_input = None
                carried_qkv_gate = None
                carried_prepared_qkvg = None
                carried_gated = None
                layer_idx = kernels.end
                continue
            if (
                l61_prune
                and can_prune_exit
                and layer_idx == self.end_layer - 1
                and exit_last_index is not None
                and exit_last_index.shape[0] <= 128
                and stream.ndim == 2
                and stream.shape[0] > 128
            ):
                pruned_stream = self._maybe_forward_l61_pruned(
                    self.layers[layer_idx],
                    positions,
                    stream,
                    forward_batch,
                    exit_last_index,
                )
                if pruned_stream is not None:
                    stream = pruned_stream
                    layer_idx += 1
                    continue
            if use_cross_layer_norm or use_w7_relay:
                has_next = layer_idx + 1 < self.end_layer
                next_ln = (
                    self.layers[layer_idx + 1].input_layernorm if has_next else None
                )
                relay_kw: dict = {}
                if use_w7_relay:
                    if carried_gated is not None:
                        relay_kw["precomputed_gated"] = carried_gated
                    elif carried_prepared_qkvg is not None:
                        relay_kw["precomputed_prepared_qkvg"] = carried_prepared_qkvg
                    elif carried_qkv_gate is not None:
                        relay_kw["precomputed_qkv_gate"] = carried_qkv_gate
                    if has_next:
                        relay_kw["next_layer_attn"] = self.layers[
                            layer_idx + 1
                        ].self_attn
                    elif (
                        (
                            fuse_final_ln
                            or use_tok_sharded_relay
                            or use_chip_sharded_relay
                        )
                        and self.pp_group.is_last_rank
                        and layer_idx not in self._capture_slots_by_layer
                    ):
                        relay_kw["final_layernorm"] = self.norm
                    if use_chip_sharded_relay:
                        relay_kw["chip_sharded_stream"] = True
                        relay_kw["pending_step_c_kv_writes"] = pending_step_c_kv_writes
                    elif use_tok_sharded_relay:
                        relay_kw["tok_sharded_stream"] = True
                layer_out = self.layers[layer_idx](
                    positions,
                    stream,
                    forward_batch,
                    precomputed_attn_input=carried_attn_input,
                    next_input_layernorm=next_ln,
                    **relay_kw,
                    **cs_layer_kw,
                )
                if isinstance(layer_out, tuple):
                    if len(layer_out) == 5:
                        stream, carried_gated, _, _, _ = layer_out
                        carried_prepared_qkvg = None
                        carried_qkv_gate = None
                        carried_attn_input = None
                    elif len(layer_out) == 4:
                        stream, nq5, nkv, ng = layer_out
                        carried_prepared_qkvg = (nq5, nkv, ng)
                        carried_qkv_gate = None
                        carried_attn_input = None
                        carried_gated = None
                    elif len(layer_out) == 3:
                        stream, nq, ng = layer_out
                        carried_qkv_gate = (nq, ng)
                        carried_prepared_qkvg = None
                        carried_attn_input = None
                        carried_gated = None
                    elif len(layer_out) == 1:
                        (final_hidden,) = layer_out
                        carried_attn_input = None
                        carried_qkv_gate = None
                        carried_prepared_qkvg = None
                        carried_gated = None
                    else:
                        stream, carried_attn_input = layer_out
                        carried_qkv_gate = None
                        carried_prepared_qkvg = None
                        carried_gated = None
                else:
                    stream = layer_out
                    carried_attn_input = None
                    carried_qkv_gate = None
                    carried_prepared_qkvg = None
                    carried_gated = None
            else:
                stream = self.layers[layer_idx](
                    positions, stream, forward_batch, **cs_layer_kw
                )
            slots = self._capture_slots_by_layer.get(layer_idx)
            if slots is not None:
                # Drafts read the operator dtype, not the fp32 residual.
                snap = stream.to(torch.bfloat16)
                if not use_chip_sharded_relay and use_tok_sharded_relay:
                    snap = tensor_model_parallel_all_gather(snap.contiguous(), dim=0)
                for slot in slots:
                    captured_by_slot[slot] = snap
            layer_idx += 1

        if pending_step_c_kv_writes:
            from sglang_tpu.muse_megakernel.decode.b1_moe_ops import (
                flush_pending_step_c_kv_writes,
            )

            flush_pending_step_c_kv_writes(pending_step_c_kv_writes)

        if not self.pp_group.is_last_rank:
            proxy = {
                "hidden_states": stream,
                "residual": torch.zeros_like(stream),
            }
            if self.layers_to_capture:
                proxy[AUX_HIDDEN_STATES_PROXY_KEY] = self._relay_aux_hidden_states(
                    aux_buffer, captured_by_slot, stream.shape[0], stream.device
                )
            return PPProxyTensors(proxy)

        hidden = (
            final_hidden
            if final_hidden is not None
            else self.norm(stream.to(torch.bfloat16))
        )
        if not self.layers_to_capture:
            return hidden
        # One tensor, not a list: packing a list happens outside the compiled
        # graph and costs a separate eager program on every decode step.
        if use_chip_sharded_relay and aux_buffer is None:
            r_pair_cap = stream.shape[0]
            h_dim = self.config.hidden_size
            unfilled = None
            cap_slots: List[torch.Tensor] = []
            for s in range(len(self.layers_to_capture)):
                t = captured_by_slot.get(s)
                if t is None:
                    if unfilled is None:
                        unfilled = torch.zeros(
                            (r_pair_cap, h_dim),
                            dtype=torch.bfloat16,
                            device=hidden.device,
                        )
                    t = unfilled
                cap_slots.append(t)
            local_cat = (
                cap_slots[0]
                if len(cap_slots) == 1
                else torch.cat(cap_slots, dim=-1)
            ).contiguous()
            tot_cols = local_cat.shape[1]
            half_cols = tot_cols // 2
            ag_w = min(half_cols, 1024 if r_pair_cap <= 16 else 512) if tot_cols >= 256 else 0
            vmem_need = (
                r_pair_cap + n_tok_eff + (n_tok_eff // 2 if r_pair_cap == 8 else 0)
            ) * tot_cols * 2
            if (
                r_pair_cap >= 8
                and r_pair_cap % 8 == 0
                and tot_cols % 256 == 0
                and half_cols % 128 == 0
                and ag_w > 0
                and half_cols % ag_w == 0
                and vmem_need <= 48 * 1024 * 1024
            ):
                from sglang_tpu.kernels.decode_collective_bridges_w8 import (
                    get_ag_chip_sharded_capture_op,
                )

                return hidden, get_ag_chip_sharded_capture_op(chip_sh_tp)(local_cat)
            ag_cat = tensor_model_parallel_all_gather(local_cat, dim=0)
            aux = (
                ag_cat.view(chip_sh_tp // 2, 2, r_pair_cap, tot_cols)[:, 0]
                .reshape(n_tok_eff, tot_cols)
                .contiguous()
            )
            return hidden, aux
        aux = self._relay_aux_hidden_states(
            aux_buffer, captured_by_slot, hidden.shape[0], hidden.device
        )
        return hidden, aux

    @staticmethod
    def _maybe_forward_l61_pruned(
        layer: nn.Module,
        positions: torch.Tensor,
        stream: torch.Tensor,
        forward_batch: ForwardBatch,
        exit_last_index: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Fallback L61 pruning when ``MusePrefillKernels`` is inactive on last layer."""
        if not isinstance(layer, MuseSparkDecoderLayer) or not isinstance(
            layer.self_attn, MuseSparkAttention
        ):
            return None
        attn = layer.self_attn
        qkv_w = getattr(attn.qkv_proj, "weight", None)
        gate_w = getattr(attn.gate_proj, "weight", None)
        if (
            attn.num_heads != 16
            or attn.num_kv_heads != 2
            or attn.head_dim != 64
            or attn.attn.k_scale_float not in (None, 1.0)
            or attn.attn.v_scale_float not in (None, 1.0)
            or (
                attn._qkvg_weight is None
                and (
                    (isinstance(qkv_w, torch.Tensor) and qkv_w.shape[0] == 0)
                    or (isinstance(gate_w, torch.Tensor) and gate_w.shape[0] == 0)
                )
            )
        ):
            return None
        from sglang.srt.model_executor.forward_context import get_attn_backend

        be = get_attn_backend()
        if (
            be is None
            or not hasattr(be, "write_kv_only")
            or getattr(be, "_prepared_op", None) is None
            or getattr(be, "kv_cache_quant_dtype", None) != torch.float8_e4m3fn
            or getattr(be, "_dcp", 1) != 1
            or getattr(getattr(be, "forward_metadata", None), "cp_write_segments", None)
            is not None
        ):
            return None
        from sglang_tpu.attention_pallas_rpa_base import (
            _fused_kv_operands,
            _pack_qkv_heads,
            _quantize_kv_to_fp8,
        )
        from sglang_tpu.cp_forward import fused_kv_rows
        from sglang_tpu.muse_prefill_kernels.causal_chunk_stream import (
            ChunkRpaMetaTorch,
        )

        t_total = stream.shape[0]
        bs = exit_last_index.shape[0]
        mlp_batches = getattr(layer.mlp, "_fused_decode_batches", ())
        bs_pad_mlp = next(
            (b for b in (16, 32, 64, 128) if b >= bs and b in mlp_batches), None
        )
        if getattr(layer.mlp, "_fused_decode_op", None) is None or bs_pad_mlp is None:
            return None
        attn_in = layer.input_layernorm(stream).to(torch.bfloat16)
        if attn._qkvg_weight is not None:
            qkvg = F.linear(attn_in, attn._qkvg_weight)
            qkv, gate = qkvg.split(
                [attn.q_size + 2 * attn.kv_size, attn.q_size], dim=-1
            )
        else:
            qkv, _ = attn.qkv_proj(attn_in)
            gate, _ = attn.gate_proj(attn_in)
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        q = (attn._per_head_normalize(q) * attn.query_scale).to(qkv.dtype)
        k = attn._per_head_normalize(k).to(qkv.dtype)
        if attn.rotary_emb is not None:
            q, k = attn.rotary_emb(positions, q, k)
        kv_cache, _ = _fused_kv_operands(be.token_to_kv_pool, attn.attn.layer_id)
        k_fp8 = _quantize_kv_to_fp8(
            k.view(t_total, attn.num_kv_heads, attn.head_dim), 1.0, torch.float8_e4m3fn
        )
        v_fp8 = _quantize_kv_to_fp8(
            v.view(t_total, attn.num_kv_heads, attn.head_dim), 1.0, torch.float8_e4m3fn
        )
        kv_packed = fused_kv_rows(k_fp8, v_fp8, kv_cache).view(torch.float8_e4m3fn)
        be.write_kv_only(kv_packed, attn.attn, forward_batch)

        ext_lens = forward_batch.extend_seq_lens
        active = ext_lens > 0
        eff_idx = torch.where(active, exit_last_index, 0).clamp(0, t_total - 1)
        idx_128 = torch.nn.functional.pad(eff_idx.long(), (0, 128 - bs))
        q_128 = q.index_select(0, idx_128).view(128, attn.num_heads, attn.head_dim)
        k_128 = k_fp8.index_select(0, idx_128)
        v_128 = v_fp8.index_select(0, idx_128)
        q_packed, _, _ = _pack_qkv_heads(q_128, k_128, v_128, attn.num_kv_heads)
        q_packed = q_packed.reshape(
            1, 128, attn.num_heads // 2, 2, attn.num_kv_heads * attn.head_dim
        )
        kv_128 = kv_packed.index_select(0, idx_128)
        gate_128 = gate.index_select(0, idx_128)
        residual_128 = stream.index_select(0, idx_128)

        md = be.forward_metadata
        zero_i32 = md.kv_lens.new_zeros((1,), dtype=torch.int32)
        q_lens_pruned = active.to(torch.int32)
        cu_q_lens_pruned = torch.cat(
            [zero_i32, torch.cumsum(q_lens_pruned, dim=0, dtype=torch.int32)], dim=0
        )
        n_act = q_lens_pruned.sum()
        dist_pruned = torch.stack([zero_i32[0], zero_i32[0], n_act]).to(torch.int32)
        pruned_meta = ChunkRpaMetaTorch(
            kv_lens=md.kv_lens,
            cu_q_lens=cu_q_lens_pruned,
            cu_kv_lens=md.cu_kv_lens,
            page_indices=md.page_indices,
            swa_page_indices=md.swa_page_indices,
            swa_group_page_indices=md.swa_group_page_indices,
            distribution=dist_pruned,
        )
        out_128 = be.forward_prepared(
            q_packed, kv_128, attn.attn, forward_batch, chunk_meta=pruned_meta
        )
        attn_out = attn._per_head_normalize(out_128).to(torch.bfloat16)
        gated = torch.sigmoid(gate_128) * attn_out
        attn_proj = attn._forward_o_proj(gated, forward_batch)
        valid_mask_128 = (
            (torch.arange(128, device=attn_proj.device) < bs)
            & torch.nn.functional.pad(active, (0, 128 - bs))
        ).unsqueeze(1)
        attn_proj = torch.where(valid_mask_128, attn_proj, 0)
        residual_128 = layer.post_attention_residual_gate(
            residual_128, layer.post_attention_layernorm(attn_proj)
        )
        router_in = layer.pre_feedforward_layernorm(residual_128[:bs_pad_mlp])
        ffn_out = layer.mlp(router_in.to(torch.bfloat16), router_in, decode=True)
        residual_out = layer.post_feedforward_residual_gate(
            residual_128[:bs_pad_mlp], layer.post_feedforward_layernorm(ffn_out)
        )
        return residual_out[:bs]


class MuseSparkForConditionalGeneration(nn.Module):
    """Muse Spark served text-only."""

    # Parameters a subclass loads itself (perception towers); the checkpoint
    # completeness check here skips them.
    tower_param_prefixes: Tuple[str, ...] = ()
    # Norm-preserving gated residuals accumulate in fp32 across stage boundaries.
    pp_transfer_dtype = torch.float32
    relays_pp_aux_hidden_states = True

    def __init__(
        self,
        config: MuseSparkConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        text_config = config.text_config
        if set(text_config.mlp_layer_types) != {"sparse"}:
            raise NotImplementedError("Muse Spark on TPU supports MoE layers only")
        self.config = config
        self.text_config = text_config
        self.quant_config = quant_config
        self.pp_group = get_parallel().pp_group
        # The fp32 RoPE table must cover the served context: positions past its
        # end rotate with the last row. 2^17 rows = 32 MiB floor.
        max_position = min(
            text_config.max_position_embeddings,
            max(1 << 17, process_model_config().context_len),
        )
        self.model = MuseSparkTextModel(
            text_config, max_position, quant_config, add_prefix("model", prefix)
        )
        if text_config.tie_word_embeddings and self.pp_group.world_size > 1:
            raise NotImplementedError(
                "Muse Spark pipeline parallelism does not support tied word embeddings"
            )
        self.lm_head = (
            ParallelLMHead(
                text_config.vocab_size,
                text_config.hidden_size,
                quant_config=None,
                prefix=add_prefix("lm_head", prefix),
            )
            if self.pp_group.is_last_rank
            else PPMissingLayer()
        )
        if text_config.tie_word_embeddings:
            self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(
            text_config, logit_scale=text_config.output_multiplier
        )
        # The checkpoint's output GEMM accumulates in fp32.
        self.logits_processor.use_fp32_lm_head = True
        parallel = get_parallel()
        if tpu_envs.SGLANG_TPU_MUSE_KERNEL_ENABLED.get() and parallel.tp_size > 1:
            self.logits_processor._logits_gatherer = _OneshotLogitsGatherer(
                self.logits_processor._logits_gatherer,
                tp_rank=parallel.tp_rank,
                tp_size=parallel.tp_size,
            )
        self.capture_aux_hidden_states = False
        self._megakernel_enabled = (
            tpu_envs.SGLANG_TPU_MUSE_MEGAKERNEL.get() and self.pp_group.world_size == 1
        )
        self._megakernel = None

    @property
    def start_layer(self) -> int:
        return self.model.start_layer

    @property
    def end_layer(self) -> int:
        return self.model.end_layer

    def precompile_kernels_after_loading(self) -> None:
        """Build the megakernel's and prefill kernels' weights once quant post-processing
        has laid out the experts they read, and before the KV pool is sized, so the budget
        counts them."""
        decode_fuse_all = bool(
            _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_FUSE_ALL", False)
        )
        decode_full_moe = bool(
            _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_FULL_MOE", False)
        )
        full_moe_mode = str(
            _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_FULL_MOE_MODE", "full")
        )
        decode_ep_tp2 = bool(_env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_EP_TP2", False))
        decode_chipmate = bool(
            _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_CHIPMATE_REBALANCE", False)
        )
        decode_replicas = int(
            _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_REPLICA_SLOTS", 0)
        )
        replica_cost_mode = str(
            _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_REPLICA_COST_MODE", "tiles")
        )
        decode_allow_prefill_eplb = bool(
            _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_ALLOW_PREFILL_EPLB", False)
        )
        if decode_ep_tp2 and decode_chipmate:
            raise ValueError(
                "SGLANG_TPU_MUSE_KERNEL_DECODE_CHIPMATE_REBALANCE is pure-EP only "
                "and cannot be combined with SGLANG_TPU_MUSE_KERNEL_DECODE_EP_TP2"
            )
        if decode_full_moe:
            if full_moe_mode not in ("full", "stage_a", "stage_b"):
                raise ValueError(
                    "SGLANG_TPU_MUSE_KERNEL_DECODE_FULL_MOE_MODE must be one of "
                    f"('full', 'stage_a', 'stage_b'), got {full_moe_mode!r}"
                )
            if decode_ep_tp2 or decode_chipmate:
                raise ValueError(
                    "SGLANG_TPU_MUSE_KERNEL_DECODE_FULL_MOE requires pure EP8/EP16 "
                    "and cannot be combined with DECODE_EP_TP2 or DECODE_CHIPMATE_REBALANCE"
                )
        if decode_replicas not in (0, 1, 2, 4, 6, 8, 12, 16):
            raise ValueError(
                "SGLANG_TPU_MUSE_KERNEL_DECODE_REPLICA_SLOTS must be one of "
                f"(0, 1, 2, 4, 6, 8, 12, 16), got {decode_replicas}"
            )
        prefill_eplb_mode = str(
            _env_get("SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_MODE", "")
        ).strip()
        if decode_allow_prefill_eplb and prefill_eplb_mode:
            if decode_ep_tp2:
                raise ValueError(
                    "SGLANG_TPU_MUSE_KERNEL_DECODE_ALLOW_PREFILL_EPLB requires pure EP "
                    "and cannot be combined with SGLANG_TPU_MUSE_KERNEL_DECODE_EP_TP2"
                )
            if (
                int(_env_get("SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_CALIB_FORWARDS", 0))
                > 0
            ):
                raise ValueError(
                    "SGLANG_TPU_MUSE_KERNEL_DECODE_ALLOW_PREFILL_EPLB requires static "
                    "prefill EPLB (PREFILL_EPLB_CALIB_FORWARDS must be 0)"
                )
            sync_decode_eplb = bool(
                _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_SYNC_PREFILL_EPLB", False)
            )
            if (
                decode_replicas > 0
                and not sync_decode_eplb
                and bool(_env_get("SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_STATS_PATH", ""))
            ):
                raise ValueError(
                    "SGLANG_TPU_MUSE_KERNEL_DECODE_ALLOW_PREFILL_EPLB cannot combine "
                    "DECODE_REPLICA_SLOTS > 0 with PREFILL_EPLB_STATS_PATH unless "
                    "SGLANG_TPU_MUSE_KERNEL_DECODE_SYNC_PREFILL_EPLB=1 is set"
                )
        is_non_prefill = get_disagg().disaggregation_mode != "prefill"
        compact_k256 = bool(tpu_envs.SGLANG_TPU_MUSE_KERNEL_COMPACT_K256_SCALES.get())
        if (
            decode_ep_tp2
            and tpu_envs.SGLANG_TPU_MUSE_KERNEL_ENABLED.get()
            and not self._megakernel_enabled
            and is_non_prefill
        ):
            for layer_idx in range(self.start_layer, self.end_layer):
                mlp = self.model.layers[layer_idx].mlp
                if getattr(mlp, "quantized_experts", False) and hasattr(mlp, "experts"):
                    _convert_gmm_v2_nvfp4_to_fused_rs_layout(mlp.experts)
        if compact_k256:
            self._compact_block256_expert_scales()
        elif tpu_envs.SGLANG_TPU_MUSE_PROBE_BLOCK256_SCALES.get():
            self._probe_block256_expert_scales()
        q_bound = int(_env_get("SGLANG_TPU_MUSE_QUANT_BOUNDARY_MOE_FP4", 0))
        if q_bound != 0 and not self._megakernel_enabled:
            if abs(q_bound) not in (16, 256):
                raise ValueError(
                    "SGLANG_TPU_MUSE_QUANT_BOUNDARY_MOE_FP4 must be 0 or +-16 / +-256, "
                    f"got {q_bound}"
                )
            if q_bound < 0 or is_non_prefill:
                self._convert_boundary_fp8_experts(q_bound)
        if (
            (decode_full_moe or decode_fuse_all or decode_ep_tp2)
            and tpu_envs.SGLANG_TPU_MUSE_KERNEL_ENABLED.get()
            and not self._megakernel_enabled
            and is_non_prefill
        ):
            if decode_full_moe:
                mode = (
                    "full_moe" if full_moe_mode == "full" else f"full_{full_moe_mode}"
                )
            elif decode_ep_tp2:
                mode = "ep_tp2"
            else:
                mode = "chipmate" if decode_chipmate else "sparse"
            self._finalize_decode_replica_banks(compact_k256=compact_k256)
            w8_inplace_decode = self._prepare_w8_sparse_v2_weights()
            for layer_idx in range(self.start_layer, self.end_layer):
                mlp = self.model.layers[layer_idx].mlp
                layer_replicas = (
                    decode_replicas
                    if decode_replicas > 0
                    and "_fused_decode_replica_meta" in getattr(mlp, "_buffers", {})
                    else 0
                )
                mlp.prepare_fused_decode(
                    mode=mode,
                    num_replicas=layer_replicas,
                    replica_cost_mode=replica_cost_mode,
                )
        else:
            w8_inplace_decode = False
            compact_ep16 = (
                tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_FUSED_EP16.get()
                and tpu_envs.SGLANG_TPU_MUSE_KERNEL_COMPACT_K256_SCALES.get()
            )
            native_ep8 = (
                tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_FUSED_NATIVE_EP8.get()
                and not tpu_envs.SGLANG_TPU_MUSE_KERNEL_COMPACT_K256_SCALES.get()
                and not tpu_envs.SGLANG_TPU_MUSE_PROBE_BLOCK256_SCALES.get()
                and not tpu_envs.SGLANG_TPU_MUSE_KERNEL_BLOCK256_SCALES.get()
            )
            if (
                (compact_ep16 or native_ep8)
                and tpu_envs.SGLANG_TPU_MUSE_KERNEL_ENABLED.get()
                and not self._megakernel_enabled
                and is_non_prefill
            ):
                for layer_idx in range(self.start_layer, self.end_layer):
                    self.model.layers[layer_idx].mlp.prepare_fused_decode(
                        target_ep=16 if compact_ep16 else 8
                    )
        # The prefill kernels run EXTEND batches only; on a P/D decode engine their
        # weight copies would only shrink the KV pool. The decode MoE path keeps the env.
        if (
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_ENABLED.get()
            and get_disagg().disaggregation_mode != "decode"
            and not w8_inplace_decode
        ):
            from sglang_tpu.muse_prefill_kernels.integration import MusePrefillKernels

            self.model.prefill_kernels = MusePrefillKernels.from_model(self)
        if not self._megakernel_enabled and is_non_prefill:
            col_bridges = int(_env_get("SGLANG_TPU_MUSE_DECODE_COLLECTIVE_BRIDGES", 0))
            combo_qkv_tpar = bool(
                _env_get("SGLANG_TPU_MUSE_DECODE_ATTN_QKV_TPAR", False)
            )
            fuse_qkvg = (
                bool(_env_get("SGLANG_TPU_MUSE_FUSE_QKVG_PROJ", False))
                or combo_qkv_tpar
            )
            fp8_attn = bool(_env_get("SGLANG_TPU_MUSE_DECODE_FP8_ATTN_PROJ", False))
            fused_o_ar = int(_env_get("SGLANG_TPU_MUSE_DECODE_FUSED_O_PROJ_AR", 0))
            if fuse_qkvg or fp8_attn or fused_o_ar == 3:
                for layer_idx in range(self.start_layer, self.end_layer):
                    attn = self.model.layers[layer_idx].self_attn
                    # When Wave-7 Bridge 2b owns QKV+Gate, skip duplicate _qkvg_weight/_qkvg_w8k!
                    if (fuse_qkvg or fp8_attn) and col_bridges not in (2, 3, 5):
                        attn.prepare_fused_qkvg(fp8_wo=fp8_attn)
                    if fused_o_ar == 3 or fp8_attn:
                        attn.prepare_fused_o_proj_tail(fp8_wo=fp8_attn)
            if col_bridges > 0 or bool(
                _env_get("SGLANG_TPU_MUSE_DECODE_PALLAS_EMBED", False)
            ):
                self._prepare_decode_collective_bridges(col_bridges)
        if (
            _env_get("SGLANG_TPU_MUSE_FP8_LM_HEAD", False)
            or _env_get("SGLANG_TPU_MUSE_FUSED_FP8_LM_HEAD_TOPK", False)
        ) and hasattr(self.lm_head, "weight"):
            from sglang_tpu.fp8_lm_head import TpuFp8LmHeadMethod

            if not isinstance(
                getattr(self.lm_head, "quant_method", None), TpuFp8LmHeadMethod
            ):
                self.lm_head.quant_method = TpuFp8LmHeadMethod(
                    self.lm_head, use_zsp_kernel=True, tn=1024
                )
        if not self._megakernel_enabled:
            return
        from sglang_tpu.fp8_lm_head import TpuFp8LmHeadMethod
        from sglang_tpu.muse_megakernel.ops import MuseMegakernel

        self._megakernel = MuseMegakernel.from_model(self)
        # The megakernel path also takes the fp8 weight-only LM head.
        if not isinstance(
            getattr(self.lm_head, "quant_method", None), TpuFp8LmHeadMethod
        ):
            self.lm_head.quant_method = TpuFp8LmHeadMethod(self.lm_head)

    def _convert_boundary_fp8_experts(self, q_bound: int) -> None:
        """Convert boundary FP8 MoE layers (``L0`` / ``L61``) to NVFP4 either as a
        decode-only shadow cache (``q_bound > 0``, preserving exact FP8 prefill) or
        in-place (``q_bound < 0``, freeing 3.0 GiB/rank FP8 buffers across prefill+decode)."""
        from sglang_tpu.fused_rs_sequence_parallel import (
            has_tpu_fused_rs_post_expert_norm,
            prepare_tpu_fused_rs_post_expert_norm,
        )
        from sglang_tpu.moe.nvfp4_requant import convert_tpu_fp8_moe_to_nvfp4
        from sglang_tpu.moe.pallas.fused_moe_rs_op import prepare_fused_moe_rs

        block = abs(q_bound)
        in_place = q_bound < 0
        errs = []
        for layer_idx in range(self.start_layer, self.end_layer):
            mlp = self.model.layers[layer_idx].mlp
            ex = mlp.experts
            if (
                getattr(mlp, "quantized_experts", False)
                or ex.w13_weight.dtype != fp8_dtype
            ):
                continue
            shadow, layer_errs = convert_tpu_fp8_moe_to_nvfp4(
                ex, block, in_place=in_place
            )
            errs.extend(layer_errs)
            if in_place:
                if _is_decode_ep_tp2_layout(get_parallel()):
                    ex.__dict__["_tpu_ep_tp2_packed_fp4"] = True
                if ex.handles_moe_reduction:
                    if has_tpu_fused_rs_post_expert_norm(ex):
                        prepare_tpu_fused_rs_post_expert_norm(ex)
                    prepare_fused_moe_rs(ex)
                mlp.quantized_experts = True
            else:
                mlp.register_buffer(
                    "_decode_fp4_w1", shadow["w13_weight"], persistent=False
                )
                mlp.register_buffer(
                    "_decode_fp4_s1", shadow["w13_weight_scale"], persistent=False
                )
                mlp.register_buffer(
                    "_decode_fp4_w2", shadow["w2_weight"], persistent=False
                )
                mlp.register_buffer(
                    "_decode_fp4_s2", shadow["w2_weight_scale"], persistent=False
                )
        if errs and torch.accelerator.is_available():
            torch.accelerator.synchronize()
        if errs:
            logger.info(
                "Muse boundary FP8->NVFP4 (mode=%s, block=%d): %d tensors, mean rel err %.4g",
                "in_place" if in_place else "decode_shadow",
                block,
                len(errs),
                sum(errs) / len(errs),
            )

    def _prepare_decode_collective_bridges(self, mode: int) -> None:
        """Prepare K-major FP8 bridge buffers and eagerly register custom ops outside compile."""
        from sglang_tpu.kernels.decode_collective_bridges import (
            SUPPORTED_WORLD_SIZES,
            prewarm_decode_collective_bridge_ops,
        )
        from sglang_tpu.kernels.decode_collective_bridges_w8 import (
            prewarm_w8_decode_bridge_ops,
        )

        router_bf16 = bool(_env_get("SGLANG_TPU_MUSE_DECODE_BRIDGE_ROUTER_BF16", False))
        free_bf16 = bool(_env_get("SGLANG_TPU_MUSE_DECODE_BRIDGE_FREE_BF16", False))
        chip_sharded = bool(
            _env_get("SGLANG_TPU_MUSE_DECODE_CHIP_SHARDED_BRIDGES", False)
        )
        m2m_enabled = bool(_env_get("SGLANG_TPU_MUSE_MK_M2M", False))
        m2m_supports_chip_sh = False
        if chip_sharded and m2m_enabled:
            try:
                import inspect as _insp
                from sglang_tpu.muse_megakernel.decode.m2m.ops import (
                    maybe_forward_m2m_decode as _m2m_fn,
                )

                m2m_supports_chip_sh = (
                    "chip_sharded_stream" in _insp.signature(_m2m_fn).parameters
                )
            except Exception:
                m2m_supports_chip_sh = False
            if not m2m_supports_chip_sh:
                # Until M2M exposes chip_sharded_stream, keep tok_sharded active for M2M (#2698).
                chip_sharded = False
        tok_sharded = (
            bool(_env_get("SGLANG_TPU_MUSE_DECODE_TOKEN_SHARDED_BRIDGES", False))
            and not chip_sharded
        )
        b1_moe_b2_super = bool(
            _env_get("SGLANG_TPU_MUSE_DECODE_B1_MOE_B2_SUPERKERNEL", False)
        )
        fuse_b1_tk = bool(
            _env_get("SGLANG_TPU_MUSE_DECODE_BRIDGE1_FUSE_TOPK", False)
        )
        eff_superkernel_base = (
            mode == 3
            and b1_moe_b2_super
            and not m2m_enabled
            and not fuse_b1_tk
        )
        eff_b1_moe_b2_super = eff_superkernel_base and chip_sharded
        chip_b2_tp2_wpost = bool(
            chip_sharded
            and (
                tpu_envs.SGLANG_TPU_MUSE_DECODE_CHIP_B2_TP2_WPOST.get()
                or eff_b1_moe_b2_super
            )
        )
        bridge_qkv_epi = bool(_env_get("SGLANG_TPU_MUSE_DECODE_BRIDGE_QKV_EPI", False))
        tp16_qkv_epi = bool(tpu_envs.SGLANG_TPU_MUSE_DECODE_TP16_QKV_EPI.get())
        fuse_final_norm = bool(
            _env_get("SGLANG_TPU_MUSE_DECODE_FUSE_FINAL_NORM", False)
        )
        pallas_embed = bool(_env_get("SGLANG_TPU_MUSE_DECODE_PALLAS_EMBED", False))

        from sglang_tpu.fused_rs_sequence_parallel import (
            get_fused_moe_sequence_parallel_plan,
        )

        sp_moe_active = any(get_fused_moe_sequence_parallel_plan())
        do_b1 = mode in (1, 3, 4, 5, 6)
        do_b2 = mode in (2, 3, 4, 5, 6)
        do_qkvg = mode in (2, 3, 5)
        pk = self.model.prefill_kernels
        tp_seen = 0
        all_tok_sh_ready = mode == 3 and tok_sharded and not sp_moe_active
        all_chip_sh_ready = mode == 3 and chip_sharded and not sp_moe_active
        all_rep_ready = mode == 3 and not sp_moe_active
        for layer_idx in range(self.start_layer, self.end_layer):
            layer = self.model.layers[layer_idx]
            can_free = free_bf16 and (pk is None or (pk.start <= layer_idx < pk.end))
            if do_b1:
                layer.self_attn.prepare_w7_o_proj_fp8(free_bf16=can_free)
            if do_qkvg:
                layer.self_attn.prepare_w7_qkv_gate_fp8(free_bf16=can_free)
            if do_b1 or do_b2:
                layer.mlp.prepare_w7_moe_bridges(
                    do_pre=do_b1,
                    do_post=do_b2,
                    router_bf16=router_bf16,
                    free_bf16=can_free,
                    tok_sharded=tok_sharded,
                    chip_sharded=chip_sharded,
                    tp2_wpost=chip_b2_tp2_wpost,
                )
            want_ret_sh = bool(
                tok_sharded
                or chip_sharded
                or _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_RETURN_SHARDED", False)
                or os.environ.get("SGLANG_TPU_MUSE_KERNEL_DECODE_RETURN_SHARDED", "")
                in ("1", "true", "True")
            )
            if (
                do_b2
                and want_ret_sh
                and callable(getattr(layer.mlp, "enable_sharded_moe_out", None))
            ):
                layer.mlp.enable_sharded_moe_out()
            if all_tok_sh_ready:
                has_pair_w = (
                    "_w8_wpre_pair_w8k" in layer.mlp._buffers
                    and "_w7_wpost_w8k" in layer.mlp._buffers
                )
                has_sh = layer.mlp.__dict__.get("_fused_decode_sharded_op") is not None
                if not (has_pair_w and has_sh):
                    all_tok_sh_ready = False
            if all_chip_sh_ready:
                has_b1_b2_w = (
                    layer.self_attn._o_proj_w8k is not None
                    and "_w8_wpre_pair_w8k" in layer.mlp._buffers
                    and "_w7_wrouter_km" in layer.mlp._buffers
                    and "_w7_wpost_w8k" in layer.mlp._buffers
                )
                has_qkvg_w = layer_idx == self.start_layer or (
                    "_w7_wqkv_w8k" in layer.self_attn._buffers
                    and "_w7_wgate_w8k" in layer.self_attn._buffers
                )
                if not (has_b1_b2_w and has_qkvg_w):
                    all_chip_sh_ready = False
            if all_rep_ready:
                has_rep_w = (
                    layer.self_attn._o_proj_w8k is not None
                    and "_w7_wpre_w8k" in layer.mlp._buffers
                    and "_w7_wrouter_km" in layer.mlp._buffers
                    and "_w7_wpost_w8k" in layer.mlp._buffers
                    and (
                        layer_idx == self.start_layer
                        or (
                            "_w7_wqkv_w8k" in layer.self_attn._buffers
                            and "_w7_wgate_w8k" in layer.self_attn._buffers
                        )
                    )
                )
                if not has_rep_w:
                    all_rep_ready = False
            op = getattr(layer.self_attn, "o_proj", None)
            if op is not None and tp_seen == 0:
                tp_seen = int(getattr(op, "tp_size", 0))

        tok_sh_common = {64, 128, 256, 384}
        if all_tok_sh_ready:
            for layer_idx in range(self.start_layer, self.end_layer):
                l_b = set(
                    self.model.layers[layer_idx].mlp.__dict__.get(
                        "_fused_decode_batches", ()
                    )
                )
                tok_sh_common &= l_b
            if not tok_sh_common:
                all_tok_sh_ready = False

        # Validate VocabParallelEmbedding shard contiguity for Pallas entry lookup.
        emb = self.model.embed_tokens
        parallel = get_parallel()
        emb_tp = int(getattr(parallel, "tp_size", tp_seen))
        emb_si = getattr(emb, "shard_indices", None)
        emb_w = getattr(emb, "weight", None)
        dist_ok = torch.distributed.is_initialized()
        can_pallas_emb = (
            pallas_embed
            and dist_ok
            and emb_tp in SUPPORTED_WORLD_SIZES
            and torch.distributed.get_world_size() == emb_tp
            and emb_si is not None
            and emb_w is not None
            and emb_w.ndim == 2
            and emb_w.shape[0] >= 16
            and emb_w.shape[0] % 8 == 0
            and emb_w.shape[1] % 512 == 0
            and emb_w.dtype == torch.bfloat16
            and emb_si.num_added_elements_padded == 0
            and emb_si.num_org_vocab_padding == 0
            and emb_si.org_vocab_start_index == parallel.tp_rank * emb_w.shape[0]
        )

        # Eagerly register all custom ops BEFORE torch.compile traces forward() (#1025/#1098/#1454).
        warm_tp = tp_seen if tp_seen in SUPPORTED_WORLD_SIZES else emb_tp
        if all_chip_sh_ready and (
            not dist_ok
            or warm_tp not in SUPPORTED_WORLD_SIZES
            or torch.distributed.get_world_size() != warm_tp
        ):
            all_chip_sh_ready = False
        if all_rep_ready and (
            not dist_ok
            or warm_tp not in SUPPORTED_WORLD_SIZES
            or torch.distributed.get_world_size() != warm_tp
        ):
            all_rep_ready = False
        # Require r_pair >= 16 on standard chip-sharded bridges (B=64 TP=16 uses W7+W8 unless
        # Step (b) [b1+MoE+b2] superkernel is enabled, which fuses B=64 TP=16 end-to-end!).
        chip_sh_batches = (
            (64, 128, 256, 384)
            if (warm_tp == 8 or (warm_tp == 16 and eff_b1_moe_b2_super))
            else ((128, 256, 384) if warm_tp == 16 else ())
        )
        eff_bridge_qkv_epi = bridge_qkv_epi and (warm_tp == 8 or tp16_qkv_epi)
        prewarm_ok = warm_tp in SUPPORTED_WORLD_SIZES and dist_ok
        b1_moe_b2_layer_flags = [False] * (self.end_layer - self.start_layer)
        if prewarm_ok:
            if (tok_sharded and all_tok_sh_ready) or (
                chip_sharded and all_chip_sh_ready
            ):
                from sglang_tpu.rank_local import _slice_op

                _slice_op(None)
            if mode > 0:
                prewarm_decode_collective_bridge_ops(
                    warm_tp, mode, bf16_router=router_bf16
                )
            prewarm_w8_decode_bridge_ops(
                warm_tp,
                mode,
                bf16_router=router_bf16,
                tok_sharded=tok_sharded and all_tok_sh_ready,
                chip_sharded=chip_sharded and all_chip_sh_ready,
                bridge_qkv_epi=eff_bridge_qkv_epi,
                fuse_final_norm=fuse_final_norm,
                pallas_embed=can_pallas_emb,
            )
            if m2m_enabled:
                import inspect as _insp_pw
                from sglang_tpu.muse_megakernel.decode.m2m.ops import (
                    prewarm_m2m_decode_ops,
                )

                m2m_pw_kw = (
                    {"chip_sharded": chip_sharded and all_chip_sh_ready}
                    if "chip_sharded"
                    in _insp_pw.signature(prewarm_m2m_decode_ops).parameters
                    else {}
                )
                prewarm_m2m_decode_ops(
                    warm_tp,
                    tok_sharded=tok_sharded and all_tok_sh_ready,
                    **m2m_pw_kw,
                )
            if eff_superkernel_base and (
                (eff_b1_moe_b2_super and all_chip_sh_ready) or all_rep_ready
            ):
                from sglang_tpu.muse_megakernel.decode.b1_moe_ops import (
                    prewarm_b1_moe_superkernel_ops,
                )

                super_cfgs = set()
                rope_vars = set()
                has_free_params = False
                for idx_off, layer_idx in enumerate(range(self.start_layer, self.end_layer)):
                    lyr = self.model.layers[layer_idx]
                    mlp = lyr.mlp
                    has_super_w = (
                        "_w8_wpre_pair_w8k" in mlp._buffers
                        if (eff_b1_moe_b2_super and all_chip_sh_ready)
                        else "_w7_wpre_w8k" in mlp._buffers
                    )
                    if (
                        mlp.__dict__.get("_fused_decode_mode", "") == "sparse"
                        and mlp.__dict__.get("_fused_decode_op") is not None
                        and has_super_w
                        and "_w7_wrouter_km" in mlp._buffers
                    ):
                        b1_moe_b2_layer_flags[idx_off] = True
                        if bool(mlp.__dict__.get("_fused_decode_has_free_params", False)):
                            has_free_params = True
                        super_cfgs.add(
                            (
                                int(mlp.__dict__.get("_fused_decode_num_replicas", 0)),
                                bool(mlp.__dict__.get("_fused_decode_has_spec_slot", False)),
                                str(mlp.__dict__.get("_fused_decode_replica_cost_mode", "tiles")),
                                mlp._buffers["_w7_wrouter_km"].dtype == torch.bfloat16,
                            )
                        )
                        rope_vars.add(lyr.self_attn.rotary_emb is not None)
                if super_cfgs:
                    step_c_flag = getattr(
                        tpu_envs, "SGLANG_TPU_MUSE_DECODE_STEP_C_SUPERKERNEL", None
                    )
                    rpa_m = str(tpu_envs.SGLANG_TPU_RPA_V3_DECODE_FAST.get()).strip()
                    bd4_flag = getattr(tpu_envs, "SGLANG_TPU_KV_PACK_BD4", None)
                    step_c_bd4 = bool(warm_tp == 16 and bd4_flag is not None and bd4_flag.get())
                    step_c_cfgs = (
                        (
                            (False, False, rpa_m),
                            (False, True, rpa_m),
                            (True, False, rpa_m),
                            (True, True, rpa_m),
                            *(
                                (
                                    (False, False, rpa_m, True),
                                    (True, False, rpa_m, True),
                                )
                                if step_c_bd4
                                else ()
                            ),
                        )
                        if (
                            eff_b1_moe_b2_super
                            and all_chip_sh_ready
                            and eff_bridge_qkv_epi
                            and step_c_flag is not None
                            and bool(step_c_flag.get())
                            and rpa_m
                        )
                        else ()
                    )
                    prewarm_b1_moe_superkernel_ops(
                        warm_tp,
                        super_cfgs,
                        bridge_qkv_epi=eff_bridge_qkv_epi,
                        has_rope_variants=tuple(sorted(rope_vars)) or (True,),
                        has_free_params=has_free_params,
                        include_rep_b16=(all_rep_ready and eff_bridge_qkv_epi),
                        step_c_configs=step_c_cfgs,
                    )
            fuse_b2_dqkv = bool(
                _env_get("SGLANG_TPU_MUSE_DECODE_BRIDGE2_DIRECT_QKV_EPI", False)
            )
            ovl_b2 = bool(_env_get("SGLANG_TPU_MUSE_DECODE_BRIDGE2_OVERLAP_EPI", False))
            if fuse_b1_tk or fuse_b2_dqkv or ovl_b2:
                from sglang_tpu.kernels.decode_collective_bridges_carb import (
                    prewarm_carb_bridge_ops,
                )

                prewarm_carb_bridge_ops(
                    warm_tp,
                    bf16_router=router_bf16,
                    fuse_b1_topk=(do_b1 and fuse_b1_tk),
                    fuse_b2_direct_qkv=(do_qkvg and fuse_b2_dqkv),
                    overlap_b2_epi=(do_qkvg and ovl_b2),
                )

        # Gate all Wave-8 / Wave-9 runtime dispatch flags on prewarm_ok so forward() can NEVER
        # trigger lazy cpu_export_jax_op inside torch.compile (#1454).
        qkv_epi_ready = prewarm_ok and eff_bridge_qkv_epi
        for idx_off, layer_idx in enumerate(range(self.start_layer, self.end_layer)):
            lyr = self.model.layers[layer_idx]
            lyr._w8_bridge_qkv_epi = qkv_epi_ready
            lyr.self_attn._w8_bridge_qkv_epi = qkv_epi_ready
            lyr._w9_b1_moe_b2_superkernel = bool(b1_moe_b2_layer_flags[idx_off])
        self.model._w9_m2m_supports_chip_sh = m2m_supports_chip_sh
        self.model._w8_fuse_final_norm = (
            prewarm_ok and fuse_final_norm and (mode in (2, 3, 4, 5, 6))
        )
        self.model._w8_tok_sharded_bridges = prewarm_ok and tok_sharded
        self.model._w8_all_layers_tok_sharded_ready = prewarm_ok and all_tok_sh_ready
        self.model._w8_tok_sharded_batches = (
            tuple(sorted(tok_sh_common)) if (prewarm_ok and all_tok_sh_ready) else ()
        )
        self.model._w9_chip_sharded_bridges = prewarm_ok and chip_sharded
        self.model._w9_all_layers_chip_sharded_ready = prewarm_ok and all_chip_sh_ready
        self.model._w9_chip_sharded_batches = (
            chip_sh_batches if (prewarm_ok and all_chip_sh_ready) else ()
        )
        self.model._w9_chip_sharded_tp = (
            warm_tp if (prewarm_ok and all_chip_sh_ready) else 0
        )
        self.model._w8_pallas_embed_tp = (
            emb_tp if (prewarm_ok and can_pallas_emb) else 0
        )
        self.model._w8_pallas_embed_enabled = bool(prewarm_ok and can_pallas_emb)

    def _finalize_decode_replica_banks(self, *, compact_k256: bool) -> None:
        """Convert pending checkpoint replica slices into device FP4 or FP8 replica banks."""
        spec_slots_map = self.__dict__.pop("_decode_spec_home_slots", None) or {}
        if spec_slots_map:
            parallel = get_parallel()
            spec_owner = int(parallel.moe_ep_rank)
            manual_slot = int(
                _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_SPEC_HOME_SLOT", -1)
            )
            e_owner = self.text_config.num_local_experts // max(1, parallel.moe_ep_size)
            for layer_idx, hot_slots in spec_slots_map.items():
                mlp = self.model.layers[layer_idx].mlp
                if getattr(mlp, "quantized_experts", False) and (
                    "_decode_fp4_w1" in mlp._buffers
                    or mlp.experts.w13_weight.dtype == torch.float4_e2m1fn_x2
                ):
                    slot_val = (
                        manual_slot
                        if 0 <= manual_slot < e_owner
                        else (
                            int(hot_slots[spec_owner])
                            if 0 <= spec_owner < len(hot_slots)
                            else -1
                        )
                    )
                    if 0 <= slot_val < e_owner:
                        dev = mlp.experts.w13_weight.device
                        mlp.register_buffer(
                            "_fused_decode_spec_slot",
                            torch.tensor([slot_val], dtype=torch.int32).to(
                                device=dev, copy=True
                            ),
                            persistent=False,
                        )
        plans = self.__dict__.pop("_decode_replica_plans", None) or {}
        for layer_idx in range(self.start_layer, self.end_layer):
            mlp = self.model.layers[layer_idx].mlp
            raw = mlp.__dict__.pop("_pending_replica_raw", None)
            plan = plans.get(layer_idx)
            if not raw or plan is None:
                continue
            device = mlp.experts.w13_weight.device
            replica_meta = torch.as_tensor(plan.replica_meta, dtype=torch.int32).to(
                device=device, copy=True
            )
            required_fp4 = (
                "w13_weight",
                "w13_weight_scale",
                "w2_weight",
                "w2_weight_scale",
            )
            two_level_pow2 = (
                int(_env_get("SGLANG_TPU_MUSE_KERNEL_FP4_TWO_LEVEL_POW2", 0) or 0) == 1
            )
            if all(key in raw for key in required_fp4):
                w1_k = int(mlp.experts.w13_weight.shape[1])
                s1_blocks = int(mlp.experts.w13_weight_scale.shape[1])
                inferred_block = (
                    (w1_k // s1_blocks)
                    if (s1_blocks > 0 and w1_k % s1_blocks == 0)
                    else 16
                )
                compact_block = (
                    inferred_block
                    if inferred_block > 16
                    else (256 if compact_k256 else 0)
                )
                rep_w1, rep_s1, rep_w2, rep_s2 = _format_nvfp4_replica_bank_from_raw(
                    raw,
                    device=device,
                    compact_k256=(compact_block > 0),
                    compact_block=compact_block,
                    two_level_pow2=two_level_pow2,
                )
                mlp.attach_preloaded_replicas(
                    rep_w1=rep_w1,
                    rep_s1=rep_s1,
                    rep_w2=rep_w2,
                    rep_s2=rep_s2,
                    replica_meta=replica_meta,
                )
                continue
            if "w13_weight" not in raw or "w2_weight" not in raw:
                continue
            has_shadow_fp4 = "_decode_fp4_w1" in mlp._buffers
            if has_shadow_fp4 or mlp.experts.w13_weight.dtype == torch.float4_e2m1fn_x2:
                s1_ref = (
                    mlp._buffers["_decode_fp4_s1"]
                    if has_shadow_fp4
                    else mlp.experts.w13_weight_scale
                )
                block = 4096 // int(s1_ref.shape[1])
                rep_w1, rep_s1, rep_w2, rep_s2 = _format_fp8_raw_to_nvfp4_replica_bank(
                    raw,
                    device=device,
                    block=block,
                    two_level_pow2=(two_level_pow2 and not has_shadow_fp4),
                )
                if has_shadow_fp4:
                    mlp.register_buffer(
                        "_fused_decode_rep_w1", rep_w1, persistent=False
                    )
                    mlp.register_buffer(
                        "_fused_decode_rep_s1", rep_s1, persistent=False
                    )
                    mlp.register_buffer(
                        "_fused_decode_rep_w2", rep_w2, persistent=False
                    )
                    mlp.register_buffer(
                        "_fused_decode_rep_s2", rep_s2, persistent=False
                    )
                    mlp.register_buffer(
                        "_fused_decode_replica_meta", replica_meta, persistent=False
                    )
                else:
                    mlp.attach_preloaded_replicas(
                        rep_w1=rep_w1,
                        rep_s1=rep_s1,
                        rep_w2=rep_w2,
                        rep_s2=rep_s2,
                        replica_meta=replica_meta,
                    )
            elif mlp.experts.w13_weight.dtype == fp8_dtype:
                rep_w1, rep_s1, rep_w2, rep_s2 = _format_fp8_replica_bank_from_raw(
                    raw,
                    device=device,
                    scale_dtype=mlp.experts.w13_weight_scale.dtype,
                )
                mlp.attach_preloaded_replicas(
                    rep_w1=rep_w1,
                    rep_s1=rep_s1,
                    rep_w2=rep_w2,
                    rep_s2=rep_s2,
                    replica_meta=replica_meta,
                )

    def _prepare_w8_sparse_v2_weights(self) -> bool:
        """Prepare optional Cold-Expert W3 (3-bit) and/or 8-MiB contiguous part-major layouts."""
        from sglang.srt.layers.utils import copy_or_rebind_param
        from sglang_tpu.fp8_moe import _free_replaced_storage
        from sglang_tpu.moe.pallas.fused_rs_muse.decode_hbm_breakers import (
            quantize_cold_slots_inplace_torch,
            repack_w1_s1_part_major_torch,
            select_cold_local_slots_np,
        )

        cold_w3 = int(_env_get("SGLANG_TPU_MUSE_DECODE_COLD_W3_EXPERTS", 0))
        cold_mode = (
            str(_env_get("SGLANG_TPU_MUSE_DECODE_COLD_W3_MODE", "inplace"))
            .strip()
            .lower()
        )
        cold_alpha = float(_env_get("SGLANG_TPU_MUSE_DECODE_COLD_W3_ALPHA", 0.75))
        v2_cp = int(_env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_V2_CONTIG_PARTS", 0))
        raw_cp = os.environ.get("SGLANG_TPU_MUSE_KERNEL_DECODE_CONTIG_PARTS", "")
        cp_mode = v2_cp
        if cp_mode == 0 and raw_cp.strip() in ("2", "3"):
            cp_mode = int(raw_cp.strip())
        if cold_w3 <= 0 and cp_mode not in (2, 3):
            return False

        parallel = get_parallel()
        world = parallel.tp_size
        owner = parallel.tp_rank
        e_owner = 256 // world

        counts_by_layer = {}
        stats_path = str(
            _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_REPLICA_STATS_PATH", "")
        ).strip()
        if stats_path and os.path.exists(stats_path):
            try:
                from sglang_tpu.moe.pallas.fused_rs_muse.moe_imbalance_planner import (
                    load_calibration_counts,
                )

                counts_by_layer = load_calibration_counts(
                    stats_path,
                    num_layers=self.config.text_config.num_hidden_layers,
                    num_experts=self.text_config.num_local_experts,
                )
            except Exception as exc:
                logger.warning("Cold-W3 stats load fallback (%s): %s", stats_path, exc)

        mutated_base = False
        w3_errs = []
        for layer_idx in range(self.start_layer, self.end_layer):
            mlp = self.model.layers[layer_idx].mlp
            ex = mlp.experts
            has_shadow = "_decode_fp4_w1" in mlp._buffers
            if not has_shadow and ex.w13_weight.dtype != torch.float4_e2m1fn_x2:
                continue

            if cold_mode == "shadow" and not has_shadow:
                mlp.register_buffer(
                    "_decode_fp4_w1", ex.w13_weight.detach().clone(), persistent=False
                )
                mlp.register_buffer(
                    "_decode_fp4_s1",
                    ex.w13_weight_scale.detach().clone(),
                    persistent=False,
                )
                mlp.register_buffer(
                    "_decode_fp4_w2", ex.w2_weight.detach().clone(), persistent=False
                )
                mlp.register_buffer(
                    "_decode_fp4_s2",
                    ex.w2_weight_scale.detach().clone(),
                    persistent=False,
                )
                has_shadow = True

            w1_t = mlp._buffers["_decode_fp4_w1"] if has_shadow else ex.w13_weight
            s1_t = mlp._buffers["_decode_fp4_s1"] if has_shadow else ex.w13_weight_scale
            w2_t = mlp._buffers["_decode_fp4_w2"] if has_shadow else ex.w2_weight
            s2_t = mlp._buffers["_decode_fp4_s2"] if has_shadow else ex.w2_weight_scale
            if not has_shadow:
                mutated_base = True
                mlp._tpu_v2_extend_fallback = True

            if cold_w3 > 0:
                cold_per_owner = min(e_owner, max(1, cold_w3 // world))
                raw_c = counts_by_layer.get(layer_idx)
                if raw_c is not None:
                    order = _placed_expert_order(
                        f"model.layers.{layer_idx}.mlp.experts.w13_weight",
                        self.text_config.num_local_experts,
                    )
                    placed_c = (
                        raw_c[order.cpu().numpy()] if order is not None else raw_c
                    )
                else:
                    placed_c = (
                        mlp.e_score_correction_bias.detach().float().cpu().numpy()
                    )
                rep_meta = mlp._buffers.get("_fused_decode_replica_meta")
                rep_eids = (
                    tuple(int(x) for x in rep_meta[0].detach().cpu().tolist())
                    if rep_meta is not None
                    else ()
                )
                mask_i32, cold_slots = select_cold_local_slots_np(
                    placed_c,
                    owner=owner,
                    e_local=e_owner,
                    cold_per_owner=cold_per_owner,
                    replicated_eids=rep_eids,
                )
                e1 = quantize_cold_slots_inplace_torch(
                    w1_t, s1_t, cold_slots, alpha=cold_alpha
                )
                e2 = quantize_cold_slots_inplace_torch(
                    w2_t, s2_t, cold_slots, alpha=cold_alpha
                )
                w3_errs.extend((e1, e2))
                mlp.register_buffer(
                    "_fused_decode_w3_mask",
                    torch.tensor([mask_i32], dtype=torch.int32, device=w1_t.device),
                    persistent=False,
                )

            if cp_mode in (2, 3):
                use_dual = cp_mode == 3
                new_w1, new_s1 = repack_w1_s1_part_major_torch(
                    w1_t, s1_t, dual_halves=use_dual
                )
                _wait_for(new_w1, new_s1)
                if has_shadow:
                    mlp.register_buffer("_decode_fp4_w1", new_w1, persistent=False)
                    mlp.register_buffer("_decode_fp4_s1", new_s1, persistent=False)
                else:
                    old_w1 = ex.w13_weight.data
                    old_s1 = ex.w13_weight_scale.data
                    copy_or_rebind_param(ex, "w13_weight", new_w1)
                    copy_or_rebind_param(ex, "w13_weight_scale", new_s1)
                    _free_replaced_storage(old_w1, ex.w13_weight.data)
                    _free_replaced_storage(old_s1, ex.w13_weight_scale.data)
                rep_w1 = mlp._buffers.get("_fused_decode_rep_w1")
                rep_s1 = mlp._buffers.get("_fused_decode_rep_s1")
                if rep_w1 is not None and rep_s1 is not None:
                    n_rw1, n_rs1 = repack_w1_s1_part_major_torch(
                        rep_w1, rep_s1, dual_halves=use_dual
                    )
                    _wait_for(n_rw1, n_rs1)
                    mlp.register_buffer("_fused_decode_rep_w1", n_rw1, persistent=False)
                    mlp.register_buffer("_fused_decode_rep_s1", n_rs1, persistent=False)

            mlp._decode_sparse_v2 = True

        if torch.accelerator.is_available():
            torch.accelerator.synchronize()
        logger.info(
            "Muse Wave-8 sparse_v2 prep: cold_w3=%d (%s, mean rel err %.4g), contig_parts=%d, mutated_base=%s",
            cold_w3,
            cold_mode,
            sum(w3_errs) / max(1, len(w3_errs)),
            cp_mode,
            mutated_base,
        )
        return mutated_base

    def _probe_block256_expert_scales(self) -> None:
        from sglang_tpu.muse_megakernel.common.expert_requant import (
            probe_tpu_nvfp4_block_scales,
        )

        errs = []
        for layer_idx in range(self.start_layer, self.end_layer):
            experts = self.model.layers[layer_idx].mlp.experts
            # Layers 0 and 61 hold fp8 (or bf16) experts.
            if (
                experts.w13_weight.dtype == torch.float4_e2m1fn_x2
                and experts.w13_weight_scale.shape[1] * 16
                == experts.w13_weight.shape[1]
            ):
                errs += probe_tpu_nvfp4_block_scales(experts, 256)
        torch.accelerator.synchronize()
        logger.info(
            "Muse probe: block-256 expert scales, %d tensors, mean rel err %.4g",
            len(errs),
            sum(errs) / max(len(errs), 1),
        )

    def _compact_block256_expert_scales(self) -> None:
        from sglang_tpu.moe.nvfp4_requant import compact_tpu_nvfp4_block_scales

        errs = []
        for layer_idx in range(self.start_layer, self.end_layer):
            experts = self.model.layers[layer_idx].mlp.experts
            if (
                experts.w13_weight.dtype == torch.float4_e2m1fn_x2
                and experts.w13_weight_scale.shape[1] * 16
                == experts.w13_weight.shape[1]
            ):
                errs += compact_tpu_nvfp4_block_scales(experts, 256)
        torch.accelerator.synchronize()
        logger.info(
            "Muse compact: block-256 expert scales, %d tensors, mean rel err %.4g",
            len(errs),
            sum(errs) / max(len(errs), 1),
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def get_attention_sliding_window_size(self) -> Optional[int]:
        return self.text_config.sliding_window - 1

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        **kwargs,
    ):
        megakernel = self._megakernel
        if (
            megakernel is not None
            and input_embeds is None
            and pp_proxy_tensors is None
            and megakernel.handles(forward_batch, input_ids.shape[0])
        ):
            hidden_states, aux_hidden_states = megakernel.forward(
                embeds=self.model.embed_tokens(input_ids),
                positions=positions,
                forward_batch=forward_batch,
                capture=(
                    tuple(sorted(self.model.layers_to_capture))
                    if self.capture_aux_hidden_states
                    else ()
                ),
            )
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )
        if megakernel is not None and megakernel.moe_mode == "ep":
            # the experts hold the EP megakernel layout, which no native MoE reads
            raise RuntimeError(
                f"Muse EP serving: a {forward_batch.forward_mode.name} step of "
                f"{input_ids.shape[0]} tokens fell outside the EP programs (decode/verify up "
                "to SGLANG_TPU_MUSE_MK_MAX_ROWS rows, extend through prefill/spep)"
            )
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if not self.pp_group.is_last_rank:
            return hidden_states
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        elif (
            forward_batch.forward_mode == ForwardMode.EXTEND
            and not getattr(forward_batch, "return_logprob", False)
            and getattr(forward_batch, "capture_hidden_mode", CaptureHiddenMode.NULL)
            == CaptureHiddenMode.NULL
            and hidden_states.shape[0] != input_ids.shape[0]
        ):
            return self._forward_pruned_logits(hidden_states, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
        )

    def _forward_pruned_logits(
        self, pruned_states: torch.Tensor, forward_batch: ForwardBatch
    ):
        from sglang.srt.layers.logits_processor import (
            LogitsMetadata,
            LogitsProcessorOutput,
        )
        from sglang.srt.model_executor.forward_batch_info import LogitsOutputMode

        lp = self.logits_processor
        distributed_top_k = None
        if forward_batch.logits_output_mode in (
            LogitsOutputMode.DISTRIBUTED_TOP_K,
            LogitsOutputMode.DISTRIBUTED_GREEDY,
        ):
            from sglang_tpu.sampler import distributed_top_k_candidate_width

            candidate_width = distributed_top_k_candidate_width(lp, self.lm_head)
            if candidate_width is None:
                raise RuntimeError("DISTRIBUTED_TOP_K is unsupported by this model")
            distributed_top_k = (
                1
                if forward_batch.logits_output_mode
                is LogitsOutputMode.DISTRIBUTED_GREEDY
                else candidate_width
            )
        logits_metadata = LogitsMetadata.from_forward_batch(
            forward_batch, compute_input_logprobs=lp.compute_input_logprobs
        )
        if forward_batch.logits_output_mode is LogitsOutputMode.DISTRIBUTED_GUMBEL_MAX:
            g_temps = getattr(forward_batch, "gumbel_temperatures", None)
            if g_temps is not None:
                g_out = lp._get_distributed_gumbel_max(
                    pruned_states, self.lm_head, g_temps
                )
                if g_out is not None:
                    g_logits, g_ids = g_out
                    return LogitsProcessorOutput(
                        next_token_logits=None,
                        next_token_global_top_k_logits=g_logits,
                        next_token_global_top_k_token_ids=g_ids.to(torch.int64),
                        mm_input_embeds=logits_metadata.mm_input_embeds,
                    )
        if distributed_top_k is not None:
            top_logits, top_ids = lp._get_distributed_top_k(
                pruned_states, self.lm_head, distributed_top_k
            )
            return LogitsProcessorOutput(
                next_token_logits=None,
                next_token_global_top_k_logits=top_logits,
                next_token_global_top_k_token_ids=top_ids.to(torch.int64),
                mm_input_embeds=logits_metadata.mm_input_embeds,
            )
        logits = lp._get_logits(pruned_states, self.lm_head, logits_metadata)
        return LogitsProcessorOutput(
            next_token_logits=logits,
            mm_input_embeds=logits_metadata.mm_input_embeds,
        )

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_dflash_layers_to_capture(self, layer_ids: List[int]) -> None:
        if not layer_ids:
            raise ValueError("DFLASH requires explicit layer_ids for aux capture.")
        num_layers = self.text_config.num_hidden_layers
        bad = [i for i in layer_ids if not 0 <= i < num_layers]
        if bad:
            raise ValueError(f"Capture layer ids {bad} outside [0, {num_layers}).")
        self.capture_aux_hidden_states = self.pp_group.is_last_rank
        self.model._set_layers_to_capture(layer_ids)

    def _load_expert_stack(
        self,
        params: dict,
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        name: str,
    ) -> Optional[str]:
        """Copy this rank's slice of a ``[num_experts, ...]`` expert tensor.

        EP slices whole experts. Expert TP slices the intermediate dim: rows of
        each gate/up half of ``w13*`` (``[E, 2I, ...]``) and the last dim of
        ``w2*`` (``[E, H, I/pack]``). Unquantized BF16 expert stacks are
        quantized to per-expert FP8 on the host before upload so single-device
        PP stages never materialize a 12 GiB BF16 stack or lazy slice-scatter
        chain in HBM.
        """
        parallel = get_parallel()
        num_experts = loaded_weight.shape[0]
        order = _placed_expert_order(name, num_experts)
        if _is_decode_ep_tp2_layout(parallel):
            ep_rank, tp_rank = _resolve_ep_tp2_coords(
                parallel.tp_rank, parallel.moe_ep_size
            )
        else:
            ep_rank, tp_rank = parallel.moe_ep_rank, parallel.moe_tp_rank
        tp_size = parallel.moe_tp_size
        if num_experts == param.shape[0]:
            local_ids = order
        else:
            local_count = num_experts // parallel.moe_ep_size
            start = ep_rank * local_count
            local_ids = None if order is None else order[start : start + local_count]
        if local_ids is not None:
            local = loaded_weight.index_select(0, local_ids.to(loaded_weight.device))
        elif num_experts == param.shape[0]:
            local = loaded_weight
        else:
            local = loaded_weight.narrow(0, start, local_count)
        if tp_size > 1 and local.dim() == 3:
            if ".w13_" in name and local.shape[1] == param.shape[1] * tp_size:
                half, shard = local.shape[1] // 2, param.shape[1] // 2
                local = torch.cat(
                    (
                        local.narrow(1, tp_rank * shard, shard),
                        local.narrow(1, half + tp_rank * shard, shard),
                    ),
                    dim=1,
                )
            elif ".w2_" in name and local.shape[2] == param.shape[2] * tp_size:
                shard = param.shape[2]
                local = local.narrow(2, tp_rank * shard, shard)
        if self.__dict__.get("_decode_replica_plans"):
            self._capture_pending_replica_slice(
                name=name,
                loaded_weight=loaded_weight,
                order=order,
                owner_id=ep_rank,
                tp_size=tp_size,
                tp_rank=tp_rank,
                param_shape=tuple(param.shape),
            )
        if tuple(local.shape) != tuple(param.shape):
            raise RuntimeError(
                f"Expert tensor {name}: parameter {tuple(param.shape)} vs "
                f"checkpoint slice {tuple(local.shape)}"
            )
        device, dtype = param.device, param.dtype
        old = param.data
        param.data = torch.empty(0, dtype=dtype, device=device)
        _free_replaced_storage(old, param.data)

        per_expert_bytes = max(
            1, local[0].numel() * max(param.element_size(), local.element_size())
        )
        step = max(1, (256 << 20) // per_expert_bytes)
        if dtype == fp8_dtype and local.dtype != fp8_dtype:
            fp8_max = torch.finfo(fp8_dtype).max
            chunks: list[torch.Tensor] = []
            scale_chunks: list[torch.Tensor] = []
            prev_threads = torch.get_num_threads()
            host_cpus = os.cpu_count() or 8
            active_ranks = min(host_cpus, max(1, parallel.tp_size, parallel.pp_size))
            torch.set_num_threads(max(1, host_cpus // active_ranks))
            try:
                for start in range(0, local.shape[0], step):
                    src = local[start : start + step].detach().cpu()
                    scales = (
                        src.abs().amax(dim=(1, 2)).to(torch.float32).clamp(min=1e-12)
                        / fp8_max
                    )
                    qchunk = (
                        (src / scales[:, None, None])
                        .clamp(min=-fp8_max, max=fp8_max)
                        .to(fp8_dtype)
                    )
                    chunks.append(qchunk.to(device=device, copy=True))
                    scale_chunks.append(scales)
            finally:
                torch.set_num_threads(prev_threads)
            param.data = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
            del chunks
            scale_name = f"{name}_scale"
            scale_param = params[scale_name]
            old_scale = scale_param.data
            scale_param.data = torch.empty(0, dtype=scale_param.dtype, device=device)
            _free_replaced_storage(old_scale, scale_param.data)
            scale_param.data = torch.cat(scale_chunks, dim=0).to(
                device=device, dtype=scale_param.dtype, copy=True
            )
            del scale_chunks
            _wait_for(param.data, scale_param.data)
            return scale_name

        chunks = [
            local[start : start + step].to(device=device, dtype=dtype, copy=True)
            for start in range(0, local.shape[0], step)
        ]
        param.data = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
        del chunks
        _wait_for(param.data)
        return None

    def _is_online_quant_scale(self, name: str) -> bool:
        """Scales that online FP8 quantization creates for the BF16 experts."""
        index = _layer_index(name)
        if (
            not (self.start_layer <= index < self.end_layer)
            or ".experts." not in name
            or not name.endswith("_scale")
        ):
            return False
        return not self.model.layers[index].mlp.quantized_experts

    def _is_off_stage_param(self, name: str) -> bool:
        """Return True when ``name`` belongs to another pipeline stage."""
        layer_idx = _layer_index(name)
        if layer_idx >= 0:
            return not (self.start_layer <= layer_idx < self.end_layer)
        if name.startswith("model.embed_tokens."):
            return not self.pp_group.is_first_rank
        if name == "lm_head.weight" or name.startswith("model.norm."):
            return not self.pp_group.is_last_rank
        return False

    def _load_qkv_shard(
        self, params: dict, name: str, loaded_weight: torch.Tensor
    ) -> Optional[str]:
        """Load a q/k/v projection into ``qkv_proj``; returns its parameter name."""
        for weight_part, shard_id in _QKV_SHARDS:
            if weight_part not in name:
                continue
            if shard_id != "v":
                text = self.text_config
                heads = (
                    text.num_attention_heads
                    if shard_id == "q"
                    else text.num_key_value_heads
                )
                loaded_weight = _to_interleaved_rope(loaded_weight, heads)
            param_name = name.replace(weight_part, ".qkv_proj.")
            param = params[param_name]
            param.weight_loader(param, loaded_weight, shard_id)
            return param_name
        return None

    def _init_decode_replica_plans(self) -> None:
        """Build per-layer §4.5b preloaded replica plans from calibration stats if enabled."""
        num_replicas = int(_env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_REPLICA_SLOTS", 0))
        stats_path = str(
            _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_REPLICA_STATS_PATH", "")
        )
        enabled = bool(
            _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_FUSE_ALL", False)
            or _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_EP_TP2", False)
            or _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_FULL_MOE", False)
        )
        spec_fast_enabled = bool(
            _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_SPEC_PRIME_FAST", False)
        ) and int(_env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_SPEC_PRIME", 0)) > 0
        if (
            (num_replicas <= 0 and not spec_fast_enabled)
            or not stats_path
            or not enabled
            or get_disagg().disaggregation_mode == "prefill"
        ):
            self._decode_replica_plans = {}
            self._decode_spec_home_slots = {}
            return
        from sglang_tpu.moe.pallas.fused_rs_muse.moe_imbalance_planner import (
            build_preloaded_replica_plan,
            load_calibration_counts,
            select_hot_non_replicated_home_slots,
            validate_replica_plan,
        )

        parallel = get_parallel()
        num_owners = parallel.moe_ep_size
        counts_by_layer = load_calibration_counts(
            stats_path,
            num_layers=self.text_config.num_hidden_layers,
            num_experts=self.text_config.num_local_experts,
        )
        owner_to_chip = (
            [r >> 1 for r in range(num_owners)]
            if parallel.moe_tp_size == 1 and num_owners % 2 == 0
            else None
        )
        sel_mode = str(
            _env_get(
                "SGLANG_TPU_MUSE_KERNEL_DECODE_REPLICA_SELECTION_MODE", "global_top"
            )
        )
        rep_obj = str(
            _env_get(
                "SGLANG_TPU_MUSE_KERNEL_DECODE_REPLICA_OBJECTIVE", "decode_active"
            )
        )
        allow_boundary_rep = bool(
            _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_FP8_FUSE_ALL", False)
            or _env_get("SGLANG_TPU_MUSE_KERNEL_DECODE_SPARSE_FP8", False)
            or int(_env_get("SGLANG_TPU_MUSE_QUANT_BOUNDARY_MOE_FP4", 0)) != 0
        )
        plans = {}
        spec_hot_slots = {}
        for layer_idx in range(self.start_layer, self.end_layer):
            mlp = self.model.layers[layer_idx].mlp
            if not getattr(mlp, "quantized_experts", False) and not allow_boundary_rep:
                continue
            counts = counts_by_layer.get(layer_idx)
            if counts is None:
                continue
            order = _placed_expert_order(
                f"model.layers.{layer_idx}.mlp.experts.w13_weight",
                self.text_config.num_local_experts,
            )
            placement = None if order is None else order.cpu().numpy()
            plan = None
            if num_replicas > 0:
                plan = build_preloaded_replica_plan(
                    counts,
                    num_owners=num_owners,
                    num_replicas_per_owner=num_replicas,
                    placement=placement,
                    owner_to_chip=owner_to_chip,
                    objective=rep_obj,
                    selection_mode=sel_mode,
                )
                validate_replica_plan(plan)
                plans[layer_idx] = plan
            if spec_fast_enabled:
                spec_hot_slots[layer_idx] = select_hot_non_replicated_home_slots(
                    counts,
                    num_owners=num_owners,
                    replica_meta=(plan.replica_meta if plan is not None else None),
                    placement=placement,
                )
        self._decode_replica_plans = plans
        self._decode_spec_home_slots = spec_hot_slots

    def _capture_pending_replica_slice(
        self,
        *,
        name: str,
        loaded_weight: torch.Tensor,
        order: Optional[torch.Tensor],
        owner_id: int,
        tp_size: int,
        tp_rank: int,
        param_shape: Tuple[int, ...],
    ) -> None:
        """Capture this owner's preloaded replica slice during streaming checkpoint load."""
        plans = self.__dict__.get("_decode_replica_plans")
        if not plans:
            return
        layer_idx = _layer_index(name)
        plan = plans.get(layer_idx)
        if plan is None:
            return
        suffix = name.rsplit(".experts.", 1)[-1]
        if suffix not in (
            "w13_weight",
            "w13_weight_scale",
            "w13_weight_scale_2",
            "w2_weight",
            "w2_weight_scale",
            "w2_weight_scale_2",
        ):
            return
        ckpt_ids_arr = getattr(plan, "holder_checkpoint_ids", None)
        if ckpt_ids_arr is not None:
            raw_ids = [int(x) for x in ckpt_ids_arr[owner_id]]
            safe_ids = torch.tensor(
                [x if x >= 0 else 0 for x in raw_ids],
                dtype=torch.long,
                device=loaded_weight.device,
            )
        else:
            raw_ids = [int(x) for x in plan.holder_expert_ids[owner_id]]
            placed_ids = torch.tensor(
                [x if x >= 0 else 0 for x in raw_ids],
                dtype=torch.long,
                device=loaded_weight.device,
            )
            safe_ids = (
                placed_ids
                if order is None
                else order.to(loaded_weight.device).index_select(0, placed_ids)
            )
        rep = loaded_weight.index_select(0, safe_ids)
        if tp_size > 1 and rep.dim() == 3:
            if suffix.startswith("w13_") and rep.shape[1] == param_shape[1] * tp_size:
                half, shard = rep.shape[1] // 2, param_shape[1] // 2
                rep = torch.cat(
                    (
                        rep.narrow(1, tp_rank * shard, shard),
                        rep.narrow(1, half + tp_rank * shard, shard),
                    ),
                    dim=1,
                )
            elif suffix.startswith("w2_") and rep.shape[2] == param_shape[2] * tp_size:
                shard = param_shape[2]
                rep = rep.narrow(2, tp_rank * shard, shard)
        mlp = self.model.layers[layer_idx].mlp
        pending = mlp.__dict__.setdefault("_pending_replica_raw", {})
        pending[suffix] = rep.detach().cpu().contiguous()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        self._init_decode_replica_plans()
        params = dict(self.named_parameters())
        num_layers = self.text_config.num_hidden_layers
        loaded = set()
        skipped = 0
        for checkpoint_name, loaded_weight in weights:
            name = _param_name(checkpoint_name)
            if (
                name is None
                or _layer_index(name) >= num_layers
                or self._is_off_stage_param(name)
            ):
                # Tower tensors, unused scales, off-stage PP parameters, or cut layers.
                skipped += 1
                continue
            if self.text_config.tie_word_embeddings and name == "lm_head.weight":
                continue
            qkv_name = self._load_qkv_shard(params, name, loaded_weight)
            if qkv_name is not None:
                loaded.add(qkv_name)
                continue
            param = params.get(name)
            if param is None:
                raise ValueError(
                    f"Checkpoint tensor {checkpoint_name} has no model parameter"
                )
            loaded_weight = _place_router_rows(name, loaded_weight)
            if ".experts.w" in name:
                scale_name = self._load_expert_stack(params, param, loaded_weight, name)
                if scale_name is not None:
                    loaded.add(scale_name)
            else:
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, loaded_weight)
            loaded.add(name)
        missing = sorted(
            name
            for name in params
            if name not in loaded
            and not name.startswith(self.tower_param_prefixes)
            and not name.endswith(("w13_input_scale", "w2_input_scale"))
            and not self._is_online_quant_scale(name)
        )
        if missing:
            raise RuntimeError(
                f"Muse Spark checkpoint is missing {len(missing)} tensors, "
                f"e.g. {missing[:5]}"
            )
        if skipped:
            logger.info("Skipped %d non-text or unused checkpoint tensors", skipped)
        for layer_idx in range(self.start_layer, self.end_layer):
            layer = self.model.layers[layer_idx]
            layer.post_attention_residual_gate.precompute_coefficients()
            layer.post_feedforward_residual_gate.precompute_coefficients()
        if (
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_ENABLED.get()
            or tpu_envs.SGLANG_TPU_MUSE_KERNEL_DECODE_NORM_LAYOUT.get() > 0
            or int(_env_get("SGLANG_TPU_MUSE_DECODE_FUSED_O_PROJ_AR", 0)) > 0
            or bool(_env_get("SGLANG_TPU_MUSE_DECODE_ATTN_QKV_TPAR", False))
            or int(_env_get("SGLANG_TPU_MUSE_DECODE_COLLECTIVE_BRIDGES", 0)) > 0
        ):
            for module in self.modules():
                if isinstance(module, MuseSparkRMSNorm):
                    module.precompute_gain()
