# SPDX-License-Identifier: Apache-2.0
"""All-layer span-addressed K/V write into SEQ_ALONG_LANE draft KV buffers.

The SEQ_ALONG_LANE counterpart of kv_fused_rope_write for the DFlash draft
cache: K arrives rotated, both K and V are cast to the cache dtype, packed with
the token on the lane axis, and every layer's spans are written by the
vendored sal_kv_span_write kernel. Each layer buffer is donated and aliased to
its own output.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import torch

from sglang_tpu.kernels.sal_kv_span_write import (
    pack_kv_seq_along_lane,
    write_spans_seq_along_lane,
)

_OP_NAME = "sglang_tpu::kv_sal_span_write"


def _op_name(num_layers: int) -> str:
    return f"{_OP_NAME}_{num_layers}"


def sal_span_write_kv_jax(
    dst_starts: jax.Array,
    src_starts: jax.Array,
    span_lens: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *kv_bufs: jax.Array,
) -> list[jax.Array]:
    """Write every layer's K/V spans, aliasing kv_bufs.

    Args:
      dst_starts: Destination flat slot per span [num_spans], int32.
      src_starts: Source token index per span [num_spans], int32.
      span_lens: Token count per span [num_spans], int32; 0 skips the span.
      k: Rotated keys [num_layers, num_tokens, num_kv_heads, head_dim].
      v: Values, shaped like k.
      *kv_bufs: One SEQ_ALONG_LANE buffer per layer, each [num_blocks,
        2 * num_kv_heads, packed_head_dim, packing, page_size].

    Returns:
      One updated buffer per layer.
    """
    page_size = int(kv_bufs[0].shape[-1])
    descriptors = jnp.stack([src_starts, dst_starts, span_lens], axis=0)
    num_segments = jnp.asarray([dst_starts.shape[0]], dtype=jnp.int32)
    out = []
    for layer, buf in enumerate(kv_bufs):
        captured = pack_kv_seq_along_lane(
            k[layer].astype(buf.dtype), v[layer].astype(buf.dtype), page_size=page_size
        )
        out.append(write_spans_seq_along_lane(buf, captured, descriptors, num_segments))
    return out


@functools.lru_cache(maxsize=None)
def get_sal_span_write_op(num_layers: int) -> Callable[..., None]:
    """Build or return the torch custom op writing num_layers SAL buffers in place."""
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    # Unannotated for the same reason as kv_fused_rope_write's _fused_rope_write.
    def _sal_span_write(dst_starts, src_starts, span_lens, k, v, *kv_bufs):
        return sal_span_write_kv_jax(dst_starts, src_starts, span_lens, k, v, *kv_bufs)

    aliases = {5 + i: i for i in range(num_layers)}
    jax_kernel = _torch_pallas.custom_jax_kernel(
        _sal_span_write,
        name=f"jax_kv_sal_span_write_{num_layers}",
        donate_argnums=tuple(aliases),
        input_output_aliases=aliases,
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        _op_name(num_layers), mutates_args=("kv",), device_types=["tpu"]
    )
    def op(
        dst_starts: torch.Tensor,
        src_starts: torch.Tensor,
        span_lens: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv: list[torch.Tensor],
    ) -> None:
        with tpu_info_for_cpu_export():
            jax_kernel(dst_starts, src_starts, span_lens, k, v, *kv)

    @op.register_fake
    def _(
        dst_starts: torch.Tensor,
        src_starts: torch.Tensor,
        span_lens: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv: list[torch.Tensor],
    ) -> None:
        return None

    return op


@torch._dynamo.assume_constant_result
def _sal_span_write_op_name(num_layers: int) -> str:
    """Build the op eagerly, even under dynamo (see kv_fused_rope_write)."""
    get_sal_span_write_op(num_layers)
    return _op_name(num_layers)


def sal_span_write_kv(
    *,
    dst_starts: torch.Tensor,
    src_starts: torch.Tensor,
    span_lens: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_bufs: list[torch.Tensor],
) -> None:
    """Write all layers' rotated K and V spans into SAL kv_bufs in place.

    Args:
      dst_starts: Destination flat slot per span [num_spans]; spans stay in a page.
      src_starts: Source token index per span [num_spans].
      span_lens: Token count per span [num_spans]; entries may be 0.
      k: Rotated keys [num_layers, num_tokens, num_kv_heads, head_dim].
      v: Values, shaped like k.
      kv_bufs: One SEQ_ALONG_LANE buffer per layer; mutated in place.
    """
    namespace, name = _sal_span_write_op_name(len(kv_bufs)).split("::")
    op = getattr(getattr(torch.ops, namespace), name)
    op(
        dst_starts.to(torch.int32),
        src_starts.to(torch.int32),
        span_lens.to(torch.int32),
        k,
        v,
        kv_bufs,
    )
