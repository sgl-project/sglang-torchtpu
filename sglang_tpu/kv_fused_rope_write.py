# SPDX-License-Identifier: Apache-2.0
"""Fused neox RoPE + all-layer in-place KV write for the DFlash draft cache.

One Pallas launch rotates K and scatters every draft layer's K/V pair into that
layer's fused TPU KV buffer, replacing the per-layer apply_k_rope plus separate
scatter chain.

The grid walks spans, not tokens: a span is a stretch of source tokens whose
destination slots are contiguous, so the scatter lives in a DMA address rather
than in the grid -- a 4096-token extend costs ~32 DMAs per layer instead of
4096. A span never crosses a page, since that is as far as destination slots
stay consecutive; extend and verify differ only in how they are cut, so one
loop serves both. BlockSpec pipelining cannot cover the source side
because a span's source offset is arbitrary, so the kernel issues its own async
copies out of HBM, mirroring ragged_paged_attention_v3's _update_kv_cache.

Each layer buffer is donated and aliased to its own output -- the in-place
contract kv_page_write, the RPA v3 attention kernel, and
TpuFusedMHATokenToKVPool.set_kv_buffer all share. K arrives normalized (post
apply_k_norm) and unrotated; the K/V projections stay outside as XLA GEMMs.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import torch
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_OP_NAME = "sglang_tpu::kv_fused_rope_write"


def _op_name(num_layers: int, max_span_len: int) -> str:
    """``namespace::name`` of the op built for one layer count and span bound."""
    return f"{_OP_NAME}_{num_layers}_{max_span_len}"


def _kernel(dst_ref, src_ref, len_ref, k_hbm, v_hbm, cos_hbm, sin_hbm, *rest):
    """Rotate one span's K and write its K/V pair into every layer.

    Pallas kernel refs are tracers with no public static type, hence the bare
    parameters. dst_ref, src_ref and len_ref are the scalar-prefetched span
    table. k_hbm, v_hbm [num_layers, num_tokens, num_kv_heads, head_dim] and
    cos_hbm, sin_hbm [num_tokens, 1, head_dim // 2] stay in HBM. rest is the
    donated layer buffers, their aliased outputs, then the VMEM staging
    scratches and the two DMA semaphore arrays.
    """
    num_layers = (len(rest) - 7) // 2
    outs = rest[num_layers : 2 * num_layers]
    k_vmem, v_vmem, cos_vmem, sin_vmem, out_vmem, in_sem, out_sem = rest[
        2 * num_layers :
    ]
    span = pl.program_id(0)
    n, src, dst = len_ref[span], src_ref[span], dst_ref[span]

    # Span tables are padded to a static worst case, so most grids end in empty
    # spans; the rope and packing below cost a full max_span_len regardless of
    # n, hence the guard rather than relying on zero-length DMAs.
    @pl.when(n > 0)
    def _write_span():
        reads = [
            pltpu.make_async_copy(
                cos_hbm.at[pl.ds(src, n)], cos_vmem.at[pl.ds(0, n)], in_sem.at[0]
            ),
            pltpu.make_async_copy(
                sin_hbm.at[pl.ds(src, n)], sin_vmem.at[pl.ds(0, n)], in_sem.at[1]
            ),
        ]
        for layer in range(num_layers):
            reads.append(
                pltpu.make_async_copy(
                    k_hbm.at[layer, pl.ds(src, n)],
                    k_vmem.at[layer, pl.ds(0, n)],
                    in_sem.at[2 + 2 * layer],
                )
            )
            reads.append(
                pltpu.make_async_copy(
                    v_hbm.at[layer, pl.ds(src, n)],
                    v_vmem.at[layer, pl.ds(0, n)],
                    in_sem.at[3 + 2 * layer],
                )
            )
        for copy in reads:
            copy.start()
        for copy in reads:
            copy.wait()

        cos, sin = cos_vmem[...], sin_vmem[...]
        half = cos.shape[-1]
        writes = []
        for layer in range(num_layers):
            k, v = k_vmem[layer], v_vmem[layer]
            x1, x2 = k[..., :half], k[..., half:]
            rotated = jnp.concatenate(
                [x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1
            )
            # Interleave per head into the fused pool's K/V-adjacent layout.
            max_span_len, heads, head_dim = k.shape
            kv = jnp.stack([rotated, v], axis=2).reshape(
                max_span_len, 2 * heads, head_dim
            )
            shape = out_vmem.shape
            heads_pad = shape[2] * shape[3] - kv.shape[1]
            dim_pad = shape[4] - kv.shape[2]
            if heads_pad or dim_pad:
                kv = jnp.pad(kv, ((0, 0), (0, heads_pad), (0, dim_pad)))
            out_vmem[layer] = kv.reshape(max_span_len, *shape[2:]).astype(
                out_vmem.dtype
            )
            buf = outs[layer]
            flat = buf.reshape(buf.shape[0] * buf.shape[1], *buf.shape[2:])
            writes.append(
                pltpu.make_async_copy(
                    out_vmem.at[layer, pl.ds(0, n)],
                    flat.at[pl.ds(dst, n)],
                    out_sem.at[layer],
                )
            )
        for copy in writes:
            copy.start()
        for copy in writes:
            copy.wait()


def fused_rope_write_kv_jax(
    dst_starts: jax.Array,
    src_starts: jax.Array,
    span_lens: jax.Array,
    k: jax.Array,
    v: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    *kv_bufs: jax.Array,
    max_span_len: int,
) -> list[jax.Array]:
    """Rope K and write every layer's K/V into its buffer, aliasing kv_bufs.

    Args:
      dst_starts: Destination flat slot per span [num_spans], int32,
        scalar-prefetched.
      src_starts: Source token index per span [num_spans], int32,
        scalar-prefetched.
      span_lens: Token count per span [num_spans], int32, scalar-prefetched.
      k: Normalized, unrotated keys [num_layers, num_tokens, num_kv_heads,
        head_dim].
      v: Values, shaped like k.
      cos: Cosine rows gathered by position [num_tokens, 1, head_dim // 2]. The
        singleton axis keeps the block's last two dims equal to the array's,
        which TPU tiling requires.
      sin: Sine rows, shaped like cos.
      *kv_bufs: One fused buffer per layer, each [num_blocks, page_size,
        kv_heads_packed, kv_packing, padded_head_dim].
      max_span_len: Upper bound on span_lens; sizes the VMEM staging scratches.

    Returns:
      One updated buffer per layer, each aliased to the corresponding kv_bufs
      entry.
    """
    num_layers, _, heads, head_dim = k.shape
    tail = tuple(kv_bufs[0].shape[2:])
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=3,
        grid=(dst_starts.shape[0],),
        in_specs=[hbm] * (4 + len(kv_bufs)),
        out_specs=[hbm] * len(kv_bufs),
        scratch_shapes=[
            pltpu.VMEM((num_layers, max_span_len, heads, head_dim), k.dtype),
            pltpu.VMEM((num_layers, max_span_len, heads, head_dim), v.dtype),
            pltpu.VMEM((max_span_len, 1, cos.shape[-1]), cos.dtype),
            pltpu.VMEM((max_span_len, 1, sin.shape[-1]), sin.dtype),
            pltpu.VMEM((num_layers, max_span_len, *tail), kv_bufs[0].dtype),
            pltpu.SemaphoreType.DMA((2 * num_layers + 2,)),
            pltpu.SemaphoreType.DMA((num_layers,)),
        ],
    )
    # Operands are (dst_starts, src_starts, span_lens, k, v, cos, sin,
    # *kv_bufs); alias each buffer onto its own output so the layer is updated
    # in place.
    return pl.pallas_call(
        _kernel,
        grid_spec=grid_spec,
        out_shape=[jax.ShapeDtypeStruct(buf.shape, buf.dtype) for buf in kv_bufs],
        input_output_aliases={7 + i: i for i in range(len(kv_bufs))},
    )(dst_starts, src_starts, span_lens, k, v, cos, sin, *kv_bufs)


@functools.lru_cache(maxsize=None)
def get_fused_rope_write_op(num_layers: int, max_span_len: int) -> Callable[..., None]:
    """Build or return the torch custom op for a layer count and span bound.

    Args:
      num_layers: Draft layers written per launch; the op signature and its
        donated argument positions depend on it.
      max_span_len: Upper bound on span length; sizes the kernel's VMEM
        scratches.

    Returns:
      A custom op taking (dst_starts, src_starts, span_lens, k, v, cos, sin,
      kv) and mutating every tensor in kv.
    """
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    # Tensors only: custom_jax_kernel traces a partial's `max_span_len` since
    # torch_tpu 76284a1fa; register a functools.partial once it reaches jax.jit static.
    # Unannotated: when dynamo traces this builder it rebuilds the closure with misaligned
    # __annotations__, and custom_jax_kernel's inspect.signature(eval_str=True) evals them.
    def _fused_rope_write(dst_starts, src_starts, span_lens, k, v, cos, sin, *kv_bufs):
        return fused_rope_write_kv_jax(
            dst_starts,
            src_starts,
            span_lens,
            k,
            v,
            cos,
            sin,
            *kv_bufs,
            max_span_len=max_span_len,
        )

    aliases = {7 + i: i for i in range(num_layers)}
    jax_kernel = _torch_pallas.custom_jax_kernel(
        _fused_rope_write,
        name=f"jax_kv_fused_rope_write_{num_layers}_{max_span_len}",
        donate_argnums=tuple(aliases),
        input_output_aliases=aliases,
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        _op_name(num_layers, max_span_len),
        mutates_args=("kv",),
        device_types=["tpu"],
    )
    def op(
        dst_starts: torch.Tensor,
        src_starts: torch.Tensor,
        span_lens: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv: list[torch.Tensor],
    ) -> None:
        with tpu_info_for_cpu_export():
            jax_kernel(dst_starts, src_starts, span_lens, k, v, cos, sin, *kv)

    @op.register_fake
    def _(
        dst_starts: torch.Tensor,
        src_starts: torch.Tensor,
        span_lens: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv: list[torch.Tensor],
    ) -> None:
        return None

    return op


@torch._dynamo.assume_constant_result
def _fused_rope_write_op_name(num_layers: int, max_span_len: int) -> str:
    """Build the op eagerly, even under dynamo, and return its torch.ops name.

    Dynamo ignores lru_cache and cannot trace the builder (custom_op's
    infer_schema, the jax kernel's id()), so calling it in a compiled region
    breaks the graph.
    """
    get_fused_rope_write_op(num_layers, max_span_len)
    return _op_name(num_layers, max_span_len)


def fused_rope_write_kv(
    *,
    dst_starts: torch.Tensor,
    src_starts: torch.Tensor,
    span_lens: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    kv_bufs: list[torch.Tensor],
    max_span_len: int,
) -> None:
    """Rope K and write all layers into kv_bufs in place.

    Args:
      dst_starts: Destination flat slot per span [num_spans].
      src_starts: Source token index per span [num_spans].
      span_lens: Token count per span [num_spans]; entries may be 0.
      k: Normalized, unrotated keys [num_layers, num_tokens, num_kv_heads,
        head_dim].
      v: Values, shaped like k.
      cos: Cosine rows gathered by position [num_tokens, head_dim // 2].
      sin: Sine rows, shaped like cos.
      kv_bufs: One fused buffer per layer, each [num_blocks, page_size,
        kv_heads_packed, kv_packing, padded_head_dim]; mutated in place.
      max_span_len: Upper bound on span_lens.
    """
    # Resolved through torch.ops by name so dynamo keeps the caller in one graph.
    namespace, name = _fused_rope_write_op_name(len(kv_bufs), max_span_len).split("::")
    op = getattr(getattr(torch.ops, namespace), name)
    op(
        dst_starts.to(torch.int32),
        src_starts.to(torch.int32),
        span_lens.to(torch.int32),
        k,
        v,
        cos.unsqueeze(1),
        sin.unsqueeze(1),
        kv_bufs,
    )


def fused_rope_write_kv_ref(
    *,
    loc: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    kv_bufs: list[torch.Tensor],
) -> None:
    """Eager per-token equivalent of the fused write, for tests.

    Spells out what the kernel fuses -- neox rotation, K/V head interleave, head
    and dim padding, store-dtype cast, scatter -- as separate torch ops. Takes
    flat per-token slots rather than a span table, so it also stands in for the
    per-layer path the kernel replaces.

    Args:
      loc: Flat slot index per token [num_tokens].
      k: See fused_rope_write_kv.
      v: See fused_rope_write_kv.
      cos: See fused_rope_write_kv.
      sin: See fused_rope_write_kv.
      kv_bufs: See fused_rope_write_kv.
    """
    half = cos.shape[-1]
    cos_b, sin_b = cos[:, None, :], sin[:, None, :]
    num_tokens, heads, head_dim = k.shape[1:]
    for layer, buf in enumerate(kv_bufs):
        keys = k[layer]
        x1, x2 = keys[..., :half], keys[..., half:]
        rotated = torch.cat([x1 * cos_b - x2 * sin_b, x2 * cos_b + x1 * sin_b], dim=-1)
        kv = torch.stack([rotated, v[layer]], dim=2).reshape(
            num_tokens, 2 * heads, head_dim
        )
        packed, packing, padded_head_dim = buf.shape[2:]
        kv = torch.nn.functional.pad(
            kv, (0, padded_head_dim - head_dim, 0, packed * packing - 2 * heads)
        )
        buf.flatten(0, 1).index_copy_(
            0,
            loc,
            kv.reshape(num_tokens, packed, packing, padded_head_dim).to(buf.dtype),
        )
