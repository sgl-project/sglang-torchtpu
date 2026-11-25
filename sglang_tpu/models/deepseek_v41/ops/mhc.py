# SPDX-License-Identifier: Apache-2.0
"""TorchTPU bridge and model binding for the V4.1 shifted mHC boundary."""

from weakref import ref

import torch

_PACKED = "_tpu_shifted_mhc_mix_weight"
_shifted_op = None
_OWNER = "_tpu_shifted_mhc_owner"
_MIX_SUBLAYERS = {"hc_attn_fn": "attn_hc", "hc_ffn_fn": "ffn_hc"}
# Seams of at least this many rows serve unpadded, as the kernel requires.
_DEFAULT_MIN_ROWS = 96


def _min_rows() -> int:
    from sglang_tpu.envs import envs as tpu_envs

    return tpu_envs.SGLANG_TPU_DSV41_SHIFTED_MHC_MIN_ROWS.get()


def seam_rows(rows: int) -> int:
    """Rows the kernel runs for a seam. Below the default floor, an opt-in lower
    SGLANG_TPU_DSV41_SHIFTED_MHC_MIN_ROWS pads to the 16-row granularity
    (_shifted_jax); otherwise the seam must already be a multiple of 16."""
    if rows < _DEFAULT_MIN_ROWS and _min_rows() < _DEFAULT_MIN_ROWS:
        return -(-rows // 16) * 16
    return rows


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
    from sglang_tpu.models.deepseek_v41.kernels.mhc.shifted_post_pre_rms import (
        _shifted_jax,
    )
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


def prepare_shifted_mhc(layers, *, draft=False, weight_names=None):
    """Refresh traceable buffers for destinations of serving boundaries."""
    from sglang.srt.models.deepseek_v4 import DeepseekV4PlatformDecoderLayer
    from sglang_tpu.pallas_export import get_pallas_tpu_device_kind

    touched = None if weight_names is None else set(weight_names)
    prefix = "stages" if draft else "model.layers"
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
        for field, name in _MIX_SUBLAYERS.items():
            if draft and index == 0 and name == "attn_hc":
                continue
            hc = getattr(layer, name)
            if (
                touched is not None
                and f"{prefix}.{index}.{field}" not in touched
                and getattr(hc.norm, _PACKED, None) is not None
            ):
                continue
            if hc.fn.dtype == torch.float32 and hc.fn.shape == (24, 20480):
                _refresh_packed_buffer(hc)
                setattr(hc.norm, _OWNER, (ref(layer), name))


def packed_mhc_sublayers(model):
    """Resolve the sublayers that own packed buffers on this model."""
    for norm in model.modules():
        owner = getattr(norm, _OWNER, None)
        if owner is not None and (layer := owner[0]()) is not None:
            yield getattr(layer, owner[1])


def refresh_shifted_mhc_after_direct_update(model, named_tensors):
    """The raw update API bypasses model.post_load_weights."""
    if _shifted_op is None:
        return
    for name, _ in named_tensors:
        layer_path, _, field = name.rpartition(".")
        sublayer = _MIX_SUBLAYERS.get(field)
        if sublayer is None:
            continue
        hc = getattr(model.get_submodule(layer_path), sublayer, None)
        if hc is not None and getattr(hc.norm, _PACKED, None) is not None:
            _refresh_packed_buffer(hc)


def fused_boundary(hc, y, residual, coefficients, next_boundary, next_hc):
    """Return the new residual and the next sublayer's input and triplet."""
    if y.device.type != "tpu" or y.dtype != torch.bfloat16:
        return None
    from sglang_tpu.models.deepseek_v41.kernels.mhc.shifted_post_pre_rms import (
        is_supported_shape,
    )

    if not is_supported_shape(seam_rows(y.shape[0]), hc.cfg.mult, hc.cfg.hidden):
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


def _use_shifted_mhc_for_rows(tokens: int) -> bool:
    # Small buckets have mixed speedups with the TorchTPU bridge. Serve the
    # measured range where both attention and FFN boundaries consistently win.
    from sglang_tpu.envs import envs as tpu_envs

    return _min_rows() <= tokens <= 8192 and tpu_envs.SGLANG_TPU_DSV41_SHIFTED_MHC.get()


def fused_target_boundary(hc, y, residual, coefficients, boundary):
    if boundary is None or not _use_shifted_mhc_for_rows(y.shape[0]):
        return None
    # The cached HcNextBoundary holds an HcSubLayer from its first forward.
    # Resolve the owner each time so direct compile sees the live parameters.
    owner = getattr(boundary.norm, _OWNER, None)
    if owner is None:
        return None
    layer = owner[0]()
    if layer is None:
        return None
    return fused_boundary(
        hc, y, residual, coefficients, boundary, getattr(layer, owner[1])
    )


def can_run_draft(model, rows, *, has_input_embeds):
    if not model.hc_pre_from_prev_sublayer or not _use_shifted_mhc_for_rows(
        rows.shape[0]
    ):
        return False
    ffn = model.stages[0].ffn_hc
    if ffn is None or getattr(ffn.norm, _PACKED, None) is None:
        return False

    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
    from sglang_tpu.models.deepseek_v41.kernels.mhc.shifted_post_pre_rms import (
        is_supported_shape,
    )

    hidden_size = rows.shape[-1] if has_input_embeds else ffn.cfg.hidden
    return (
        is_supported_shape(seam_rows(rows.shape[0]), model.hc_mult, hidden_size)
        and not is_batch_invariant_mode_enabled()
    )


def run_draft_stages(stages, positions, x, forward_batch):
    from sglang.srt.models.deepseek_v4_mhc import HcState

    state = HcState(x)
    for index, stage in enumerate(stages):
        next_hc = stages[index + 1].attn_hc if index + 1 < len(stages) else None
        state = _run_draft_stage(stage, positions, state, forward_batch, next_hc)
    return state.residual, state.pre


def _run_draft_stage(stage, positions, state, forward_batch, next_hc):
    from sglang.srt.models import deepseek_v4_mhc as mhc

    def finish(hc, rows, state, destination):
        coefficients = state.stats
        if coefficients is None:
            coefficients = mhc.mix_stats(hc, state.residual)
        if destination is not None:
            boundary = mhc.make_boundary(
                destination.norm, accepts_mxfp8=False, hc=destination
            )
            fused = fused_boundary(
                hc, rows, state.residual, coefficients, boundary, destination
            )
            if fused is not None:
                return fused
        pre, post, comb = coefficients
        updated = stage.hc_post(rows, state.residual, post, comb)
        return mhc.HcState(updated, pre)

    x = mhc.combine(stage.attn_hc, state)
    with stage.self_attn.maybe_use_decode_attn_tp(forward_batch):
        x = stage.self_attn(positions, x, forward_batch)
    state = finish(stage.attn_hc, x, state, stage.ffn_hc)
    x = mhc.combine(stage.ffn_hc, state)
    x = stage._run_ffn(x, forward_batch)
    return finish(stage.ffn_hc, x, state, next_hc)
