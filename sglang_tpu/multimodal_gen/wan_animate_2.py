# SPDX-License-Identifier: Apache-2.0
"""Wan-Animate-2 on TPU workers: the in-context attention on the Pallas flash kernel.

Runtime patch, tracked debt:
- Why: the in-context self-attention of Wan-Animate-2 runs compiled
  ``flex_attention`` over a padded ``[generation | reference video]`` layout
  with a block mask (``create_block_mask(_compile=True)``), both through the
  inductor backend, which does not target TPU. Around it, the regionally
  compiled blocks take their block index and the per-DiT K/V dicts, so
  torch.compile builds one graph per block (40 compiles per mode, past
  dynamo's recompile limit); the interleaved RoPE reads the head dim with
  stride 2.
- How: :func:`install` rebinds
  (1) ``WanAnimate2TransformerBlock._in_context_attention``: on TPU tensors,
      the attention that the flex block mask and score_mod define, as flash
      attention calls merged by log-sum-exp (as the CUDA split path does):
      every query over the generation keys (with ``log_scale`` on the keys of
      generation frame 1 as its own call), and generation frame ``j >= 1``
      over reference frame ``j - 1`` (one call with the frames as heads). The
      zero keys that the full-clip layout adds (a short clip's missing frames,
      slot padding) score 0 and only add their count, times ``exp(log_scale)``
      inside frame 1, to the softmax denominator. Partial outputs stay float32
      and round once;
  (2) ``WanAnimate2Transformer3DModel._get_block_mask_from_layout``: no block
      mask on TPU (the attention above needs none);
  (3) ``_apply_rope_interleaved`` of the block module: bfloat16 TPU tensors
      rotate through the pair swap of :mod:`sglang_tpu.multimodal_gen.rope`
      (the same float32 rotation rounded once, without stride-2 lane access);
  (4) ``WanAnimate2TransformerBlock.__call__``: a block called with its
      index and the per-DiT K/V dicts gets index 0 and single-entry dicts, so
      every block compiles to the same graph; the reference pass's K/V are
      stored back under the real index;
  (5) ``Wan_Animate_2_14B_Config`` (also in the server process): the TPU
      attention backend (``--attention-backend tpu_splash_attn``) is accepted,
      since the in-context attention no longer runs on flex, and CFG parallel
      is refused: two CFG halves running SP subgroups at once are not
      reproducible on TPU, and the unconditional branch skips block 9, so the
      halves would not issue the same collectives.
- Upstream PR or issue: none.
- Remove when: Wan-Animate-2's in-context attention takes a platform
  attention implementation and its blocks take their own reference K/V.
"""

from __future__ import annotations

import dataclasses
import math

import torch

from sglang_tpu.multimodal_gen import tpu_splash_attn as splash
from sglang_tpu.multimodal_gen.rope import apply_rotary_embedding_pair_swap

# Devices that take these paths.
_DEVICE_TYPES = ("tpu",)

_STOCK: dict[str, object] = {}


def _attend(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[B, H, S, D]`` attention: float32 output and natural-log LSE ``[B, H, S]``."""
    if (
        query.device.type in _DEVICE_TYPES
        and splash._ring_kernel_fallback_reason(query, key, value) is None
    ):
        splash._register_ops()
        if splash._FLASH_RING_OP is not None:
            # A [row offset 0, all keys] shard choice: the kernel then returns
            # its float32 output, unrounded for the merges below.
            kv_info = torch.tensor(
                [0, key.shape[2]], dtype=torch.int32, device=query.device
            )
            out, lse2 = splash._FLASH_RING_OP(
                query.contiguous(),
                key.contiguous(),
                value.contiguous(),
                kv_info,
                None,
                None,
                scale,
            )
            return out, lse2[:, :, 0] * math.log(2.0)
    return splash._attention_with_lse_reference(query, key, value, scale)


def _merge(
    out_a: torch.Tensor, lse_a: torch.Tensor, out_b: torch.Tensor, lse_b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention over two key sets from each set's float32 output and LSE."""
    peak = torch.maximum(lse_a, lse_b)
    w_a, w_b = torch.exp(lse_a - peak), torch.exp(lse_b - peak)
    total = w_a + w_b
    out = (out_a * w_a[..., None] + out_b * w_b[..., None]) / total[..., None]
    return out, peak + torch.log(total)


def zero_key_weight(layout, log_scale: float) -> float:
    """Softmax weight of the zero keys every generation query attends.

    The flex mask spans a full-length clip in frame slots of
    ``max_tokens_per_frame`` keys: the slots of a short clip's missing frames
    and the slot ends past this clip's tokens per frame are zero keys, score
    0. Those inside generation frame 1's slot carry ``log_scale``.
    """
    slot = layout.max_tokens_per_frame
    real = layout.generation_video_num_tokens
    zeros = layout.max_generation_video_num_tokens - real
    frame_1_zeros = (
        slot - layout.generation_video_tokens_per_frame
        if layout.generation_video_f >= 2
        else slot
    )
    return (zeros - frame_1_zeros) + frame_1_zeros * math.exp(log_scale)


def in_context_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    reference_video_k: torch.Tensor,
    reference_video_v: torch.Tensor,
    layout,
    log_scale: float,
) -> torch.Tensor:
    """The flex in-context attention of ``WanAnimate2TransformerBlock``.

    ``q``/``k``/``v`` are ``[B, S, H, D]`` with this clip's generation tokens
    first (rows past them are SP tail padding, returned as they are, as the
    flex path does); ``reference_video_*`` are the slot layout of
    ``_pad_reference_video_kv``, ``[B, reference_f * slot, H, D]``, whose slot
    padding the per-frame call attends as zero keys like flex does.
    """
    batch, _, heads, head_dim = q.shape
    scale = head_dim**-0.5
    frames = layout.generation_video_f
    per_frame = layout.generation_video_tokens_per_frame
    slot = layout.max_tokens_per_frame
    real = layout.generation_video_num_tokens
    reference_frames = layout.reference_video_f
    if reference_frames != frames - 1:
        raise ValueError(
            f"Wan-Animate-2 in-context attention expects one reference frame per "
            f"generation frame after the first: {reference_frames} for {frames}"
        )
    qh, kh, vh = (t[:, :real].transpose(1, 2) for t in (q, k, v))

    # Every query over the generation keys; log_scale biases generation frame 1.
    if log_scale == 0.0:
        out, lse = _attend(qh, kh, vh, scale)
    else:
        frame_1 = slice(per_frame, 2 * per_frame)
        out, lse = _attend(qh, kh[:, :, frame_1], vh[:, :, frame_1], scale)
        rest = (
            torch.cat([t[:, :, : frame_1.start], t[:, :, frame_1.stop :]], dim=2)
            for t in (kh, vh)
        )
        out, lse = _merge(*_attend(qh, *rest, scale), out, lse + log_scale)

    # Generation frame j >= 1 also attends reference frame j - 1: frames as heads.
    if frames > 1:
        rows = batch * heads
        q_frames = qh[:, :, per_frame:].reshape(rows, frames - 1, per_frame, head_dim)
        k_frames, v_frames = (
            t[:, : reference_frames * slot]
            .view(batch, reference_frames, slot, heads, head_dim)
            .permute(0, 3, 1, 2, 4)
            .reshape(rows, reference_frames, slot, head_dim)
            for t in (reference_video_k, reference_video_v)
        )
        out_r, lse_r = _attend(q_frames, k_frames, v_frames, scale)
        out_r = out_r.view(batch, heads, real - per_frame, head_dim)
        lse_r = lse_r.view(batch, heads, real - per_frame)
        out_1, lse_1 = _merge(
            out[:, :, per_frame:], lse[:, :, per_frame:], out_r, lse_r
        )
        out = torch.cat([out[:, :, :per_frame], out_1], dim=2)
        lse = torch.cat([lse[:, :, :per_frame], lse_1], dim=2)

    zero_weight = zero_key_weight(layout, log_scale)
    if zero_weight:
        # sum / (sum + zeros) = sigmoid(lse - log(zeros)).
        out = out * torch.sigmoid(lse - math.log(zero_weight))[..., None]
    out = out.to(q.dtype).transpose(1, 2)
    if q.shape[1] == real:
        return out
    return torch.cat([out, q[:, real:]], dim=1)


def _in_context_attention(
    self,
    q,
    k,
    v,
    reference_video_k,
    reference_video_v,
    block_mask,
    layout,
    reference_video_k_scale=None,
    reference_video_v_scale=None,
):
    if q.device.type not in _DEVICE_TYPES:
        return _STOCK["_in_context_attention"](
            self,
            q,
            k,
            v,
            reference_video_k,
            reference_video_v,
            block_mask,
            layout,
            reference_video_k_scale,
            reference_video_v_scale,
        )
    if reference_video_k_scale is not None or reference_video_v_scale is not None:
        raise NotImplementedError("FP8 reference K/V is CUDA-only")
    return in_context_attention(
        q,
        k,
        v,
        reference_video_k,
        reference_video_v,
        layout,
        float(self.log_scale),
    )


def _get_block_mask_from_layout(self, layout):
    if self._device.type in _DEVICE_TYPES:
        return None
    return _STOCK["_get_block_mask_from_layout"](self, layout)


def _apply_rope_interleaved(x, cos, sin):
    if x.device.type in _DEVICE_TYPES and x.dtype == torch.bfloat16:
        return apply_rotary_embedding_pair_swap(x, cos, sin)
    return _STOCK["_apply_rope_interleaved"](x, cos, sin)


def _block_call(self, *args, **kwargs):
    hidden_states = kwargs.get("hidden_states")
    if (
        "index" not in kwargs
        or hidden_states is None
        or hidden_states.device.type not in _DEVICE_TYPES
    ):
        return _STOCK["__call__"](self, *args, **kwargs)
    index = kwargs.pop("index")
    k_cache, v_cache = kwargs.pop("k_cache"), kwargs.pop("v_cache")
    if kwargs["kv_cache_mode"] == "ref":
        k_one, v_one = {}, {}
        out = _STOCK["__call__"](
            self, *args, index=0, k_cache=k_one, v_cache=v_one, **kwargs
        )
        k_cache[index], v_cache[index] = k_one[0], v_one[0]
    else:
        for name in ("k_scale_cache", "v_scale_cache"):
            if kwargs.get(name) is not None:
                raise NotImplementedError("FP8 reference K/V is CUDA-only")
        out = _STOCK["__call__"](
            self,
            *args,
            index=0,
            k_cache={0: k_cache[index]},
            v_cache={0: v_cache[index]},
            **kwargs,
        )
    return out


def _validate_server_args(self, server_args):
    from sglang.multimodal_gen.runtime.platforms import (
        AttentionBackendEnum,
        current_platform,
    )

    oot = AttentionBackendEnum.OOT.name.lower()
    if (
        not current_platform.is_tpu()
        or server_args.attention_backend != oot
        or server_args.requested_component_attention_backend("transformer") is not None
    ):
        return _STOCK["validate_server_args"](self, server_args)
    # The stock check rejects any explicit backend because the in-context
    # attention runs on flex; on TPU it runs on the kernels above, and the
    # reference pass and cross-attention take the platform backend.
    server_args.attention_backend = None
    try:
        return _STOCK["validate_server_args"](self, server_args)
    finally:
        server_args.attention_backend = oot


def _get_model_deployment_config(self):
    return dataclasses.replace(
        _STOCK["get_model_deployment_config"](self),
        auto_enable_cfg_parallel=False,
        supports_cfg_parallel=False,
    )


def install_config() -> None:
    """The pipeline-config patches; the launcher validates with them too."""
    from sglang.multimodal_gen.configs.pipeline_configs import wan

    config = wan.Wan_Animate_2_14B_Config
    for name, new in (
        ("validate_server_args", _validate_server_args),
        ("get_model_deployment_config", _get_model_deployment_config),
    ):
        if getattr(config, name) is not new:
            _STOCK[name] = getattr(config, name)
            setattr(config, name, new)


def install() -> None:
    from sglang.multimodal_gen.runtime.models.dits import wan_animate_2 as dit
    from sglang.multimodal_gen.runtime.models.dits import wan_animate_2_block as block

    blk, model = block.WanAnimate2TransformerBlock, dit.WanAnimate2Transformer3DModel
    for owner, name, new in (
        (blk, "_in_context_attention", _in_context_attention),
        (blk, "__call__", _block_call),
        (model, "_get_block_mask_from_layout", _get_block_mask_from_layout),
        (block, "_apply_rope_interleaved", _apply_rope_interleaved),
    ):
        if getattr(owner, name) is not new:
            _STOCK[name] = getattr(owner, name)
            setattr(owner, name, new)
    install_config()
