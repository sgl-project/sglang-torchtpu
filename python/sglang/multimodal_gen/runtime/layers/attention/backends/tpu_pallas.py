# SPDX-License-Identifier: Apache-2.0
"""Pallas flash-attention backend for TPU (torch_tpu).

Wraps the srt TPU pallas flash-attention kernel as a torch custom op via
``torch_tpu._internal.pallas.jax_op`` so it is usable from the diffusion
attention layers both eagerly and under torch.compile.

Constraints of the underlying kernel:
- non-causal only (video DiT self/cross attention);
- query/key/value must share the head count (no GQA);
- sequence lengths are padded to the kernel block sizes, with segment ids
  masking the padding out of the softmax.
"""

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum

# Registered torch custom-op overloads, keyed by (sm_scale, block_q, block_kv).
# One op per configuration: the pallas kernel bakes the scale and block sizes
# into the traced program.
_TPU_FLASH_OPS: dict[tuple[float, int, int], object] = {}

# Block sizes from the Wan2.2 720p sweep: (2048, 1024) is fastest for long
# sequences; the smaller pair serves short (<=2048) KV lengths without
# excess padding. Larger block_q spills registers on current TPUs.
_LONG_SEQ_BLOCKS = (2048, 1024)
_SHORT_SEQ_BLOCKS = (1024, 512)

_VMEM_LIMIT_BYTES = 128 * 1024 * 1024


def _tpu_flash_blocks(kv_len: int) -> tuple[int, int]:
    if kv_len <= 2048:
        return _SHORT_SEQ_BLOCKS
    return _LONG_SEQ_BLOCKS


def _get_tpu_flash_op(sm_scale: float, block_q: int, block_kv: int):
    """Create (once) and return the custom-op overload for a configuration."""
    key = (sm_scale, block_q, block_kv)
    op = _TPU_FLASH_OPS.get(key)
    if op is not None:
        return op

    try:
        import jax
        from torch_tpu._internal import pallas

        from sglang.srt.hardware_backend.tpu.kernels.flash_attention import (
            BlockSizes,
            SegmentIds,
            flash_attention,
        )
    except ImportError as e:
        raise ImportError(
            "The TPU pallas attention backend requires torch_tpu and jax. "
            "Install the torch_tpu wheel (with its jax dependency) or select "
            "the TORCH_SDPA attention backend."
        ) from e

    def _jax_flash(q, k, v, q_seg, kv_seg):
        return flash_attention(
            q,
            k,
            v,
            segment_ids=SegmentIds(q=q_seg, kv=kv_seg),
            causal=False,
            sm_scale=sm_scale,
            block_sizes=BlockSizes(
                block_q=block_q,
                block_k_major=block_kv,
                block_k=block_kv,
                block_b=1,
            ),
            vmem_limit_bytes=_VMEM_LIMIT_BYTES,
        )

    _jax_flash.__annotations__ = {
        "q": jax.Array,
        "k": jax.Array,
        "v": jax.Array,
        "q_seg": jax.Array,
        "kv_seg": jax.Array,
        "return": jax.Array,
    }

    op_name = (
        f"sglang_diffusion::flash_s{sm_scale!r}_q{block_q}_kv{block_kv}".replace(
            ".", "_"
        ).replace("-", "m")
    )
    registered = pallas.jax_op(op_name, _jax_flash)

    def _fake(q, k, v, q_seg, kv_seg):
        return torch.empty_like(q)

    registered.register_fake(_fake)
    # Return the OpOverload, not the CustomOpDef: dynamo inlines the latter's
    # python __call__ into the traced graph and breaks on the jax internals.
    overload = getattr(torch.ops.sglang_diffusion, op_name.split("::", 1)[1]).default
    _TPU_FLASH_OPS[key] = overload
    return overload


def _tpu_flash_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Non-causal SDPA on [batch, heads, seq, head_dim] via the pallas kernel."""
    batch, _, len_q, _ = query.shape
    len_kv = key.shape[-2]
    block_q, block_kv = _tpu_flash_blocks(len_kv)
    pad_q = (-len_q) % block_q
    pad_kv = (-len_kv) % block_kv
    query = torch.nn.functional.pad(query, (0, 0, 0, pad_q)).contiguous()
    key = torch.nn.functional.pad(key, (0, 0, 0, pad_kv)).contiguous()
    value = torch.nn.functional.pad(value, (0, 0, 0, pad_kv)).contiguous()
    device = query.device
    # Segment ids: 0 = real tokens, 1 = padding; mismatched segments are
    # masked out of the softmax, so padding never attends or is attended to.
    q_seg = torch.cat(
        [
            torch.zeros(batch, len_q, dtype=torch.int32, device=device),
            torch.ones(batch, pad_q, dtype=torch.int32, device=device),
        ],
        dim=1,
    )
    kv_seg = torch.cat(
        [
            torch.zeros(batch, len_kv, dtype=torch.int32, device=device),
            torch.ones(batch, pad_kv, dtype=torch.int32, device=device),
        ],
        dim=1,
    )
    op = _get_tpu_flash_op(float(scale), block_q, block_kv)
    out = op(query, key, value, q_seg, kv_seg)
    return out[..., :len_q, :] if pad_q else out


class TpuPallasAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.TPU_PALLAS_FLASH

    @staticmethod
    def get_impl_cls() -> type["TpuPallasAttentionImpl"]:
        return TpuPallasAttentionImpl


class TpuPallasAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        if causal:
            raise NotImplementedError(
                "The TPU pallas attention backend supports non-causal "
                "attention only; select TORCH_SDPA for causal attention."
            )
        if num_kv_heads is not None and num_kv_heads != num_heads:
            raise NotImplementedError(
                "The TPU pallas attention backend does not support GQA "
                f"(num_heads={num_heads}, num_kv_heads={num_kv_heads})."
            )
        self.softmax_scale = (
            float(softmax_scale) if softmax_scale is not None else head_size**-0.5
        )
        # Register both block variants now: creating the op involves a jax
        # trace, which must not happen for the first time inside a
        # dynamo-traced forward.
        for blocks in (_LONG_SEQ_BLOCKS, _SHORT_SEQ_BLOCKS):
            _get_tpu_flash_op(self.softmax_scale, *blocks)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        output = _tpu_flash_sdpa(query, key, value, scale=self.softmax_scale)
        return output.transpose(1, 2)
