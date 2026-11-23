# SPDX-License-Identifier: Apache-2.0
"""TorchTPU bridge and model binding for the V4.1 shifted mHC boundary."""

import jax
import torch

from sglang_tpu.envs import envs
from sglang_tpu.models.deepseek_v41.kernels.mhc import shifted_post_pre_rms as kernel

_PACKED = "_tpu_shifted_mhc_mix_weight"
_shifted_op = None


def _shifted_jax(
    y: jax.Array,
    residual: jax.Array,
    post: jax.Array,
    comb: jax.Array,
    packed: jax.Array,
    scale: jax.Array,
    base: jax.Array,
    pre: jax.Array,
    norm_weight: jax.Array,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_eps: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    return kernel.shifted_post_pre_rms(
        y,
        residual,
        post,
        comb,
        packed,
        scale,
        base,
        pre,
        norm_weight=norm_weight,
        rms_eps=rms_eps,
        hc_pre_eps=hc_eps,
        hc_sinkhorn_eps=hc_eps,
        sinkhorn_repeat=sinkhorn_iters,
        norm_eps=norm_eps,
        gate_input_mode=kernel.GateInputMode.COMPILER,
        unpadded_gates=True,
        early_prep=True,
    )


def _fake(y, residual, post, comb, packed, scale, base, pre, norm_weight, *args):
    tokens, mult = pre.shape
    return (
        torch.empty_like(residual),
        post.new_empty((tokens, mult, 1)),
        torch.empty_like(comb),
        torch.empty_like(y),
        torch.empty_like(pre),
    )


def register_shifted_mhc_op():
    global _shifted_op
    if _shifted_op is not None:
        return
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    _shifted_op = cpu_export_jax_op(
        "sglang_tpu_pallas::dsv41_shifted_mhc", _shifted_jax
    )
    _shifted_op.register_fake(_fake)


def pack_mix_weight(fn: torch.Tensor) -> torch.Tensor:
    """Pack [24, 20480] FP32 into [20480, 128] BF16 on the host after loading."""
    weight = fn.detach().cpu().contiguous()

    def truncate(values):
        return (values.view(torch.int32) & -65536).view(torch.float32)

    high = truncate(weight)
    middle = truncate(weight - high)
    low = weight - high - middle
    parts = torch.cat([high, middle, low]).to(torch.bfloat16)
    packed = torch.nn.functional.pad(parts.T, (0, 128 - parts.shape[0]))
    return packed.contiguous().to(fn.device)


def _refresh_packed_buffer(hc):
    packed = pack_mix_weight(hc.fn)
    existing = getattr(hc.norm, _PACKED, None)
    if existing is None:
        # An owned module buffer is a named direct-tracer input. A cached
        # HcSubLayer tuple would keep the real tensor during tracing.
        hc.norm.register_buffer(_PACKED, packed, persistent=False)
    else:
        # Online model loads can run while compiled artifacts retain this buffer.
        existing.copy_(packed)


def prepare_shifted_mhc(layers, *, draft=False):
    """Refresh traceable buffers for destinations of serving boundaries."""
    from sglang.srt.models.deepseek_v4 import DeepseekV4PlatformDecoderLayer
    from sglang_tpu.pallas_export import get_pallas_tpu_device_kind

    if not envs.SGLANG_TPU_ENABLE_DSV41_SHIFTED_MHC.get():
        return

    for index, layer in enumerate(layers):
        if not isinstance(layer, DeepseekV4PlatformDecoderLayer):
            continue
        cfg = layer.hc_cfg
        if (
            not cfg.pre_from_prev
            or cfg.cp_prefill
            or (cfg.mult, cfg.hidden) != (4, 5120)
        ):
            continue
        if layer.hc_attn_fn.device.type != "tpu":
            continue
        if get_pallas_tpu_device_kind() != "TPU7x":
            return
        register_shifted_mhc_op()
        # The draft begins with embeddings and has no incoming attention boundary.
        sublayers = (
            (layer.ffn_hc,)
            if draft and index == 0
            else (
                layer.attn_hc,
                layer.ffn_hc,
            )
        )
        for hc in sublayers:
            if hc.fn.dtype == torch.float32 and hc.fn.shape == (24, 20480):
                _refresh_packed_buffer(hc)


def refresh_shifted_mhc_after_direct_update(model, named_tensors):
    """The raw update API bypasses model.post_load_weights."""
    if _shifted_op is None:
        return
    sublayers = {"hc_attn_fn": "attn_hc", "hc_ffn_fn": "ffn_hc"}
    for name, _ in named_tensors:
        layer_path, _, field = name.rpartition(".")
        sublayer = sublayers.get(field)
        if sublayer is None:
            continue
        hc = getattr(model.get_submodule(layer_path), sublayer, None)
        if hc is not None and getattr(hc.norm, _PACKED, None) is not None:
            _refresh_packed_buffer(hc)


def fused_boundary(hc, y, residual, coefficients, next_boundary, next_hc):
    """Return the new residual and the next sublayer's input and triplet."""
    if (
        y.device.type != "tpu"
        or y.dtype != torch.bfloat16
        or not envs.SGLANG_TPU_ENABLE_DSV41_SHIFTED_MHC.get()
    ):
        return None
    if not kernel.is_supported_shape(y.shape[0], hc.cfg.mult, hc.cfg.hidden):
        return None

    if (
        next_boundary is None
        or not next_boundary.norm_fusable
        or residual.dtype != torch.bfloat16
        or any(gate.dtype != torch.float32 for gate in coefficients)
        or next_hc.scale.dtype != torch.float32
        or next_hc.base.dtype != torch.float32
    ):
        return None
    packed = getattr(next_hc.norm, _PACKED, None)
    if packed is None:
        return None

    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
    from sglang.srt.models.deepseek_v4_mhc import HcNormed, HcState

    if is_batch_invariant_mode_enabled():
        return None

    pre, post, comb = coefficients
    updated, next_post, next_comb, normalized, next_pre = _shifted_op(
        y,
        residual.flatten(1),
        post,
        comb,
        packed,
        next_hc.scale,
        next_hc.base,
        pre,
        next_boundary.norm.weight,
        next_hc.cfg.rms_eps,
        next_hc.cfg.eps,
        next_hc.cfg.sinkhorn_iters,
        next_boundary.norm.variance_epsilon,
    )
    return HcState(
        updated.view_as(residual),
        pre,
        HcNormed(normalized),
        (next_pre, next_post.squeeze(-1), next_comb),
    )
