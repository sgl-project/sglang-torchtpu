# SPDX-License-Identifier: Apache-2.0
"""Span-based in-place row write into one fused TPU KV buffer.

The decode-context-parallel path writes only the tokens a rank owns, so the
attention kernel's own KV write is skipped and the rows land here: the host
compacts the owned rows into source order and cuts them into spans whose
physical slots are consecutive, and one Pallas launch issues a DMA per span
from the row buffer straight into the pool. The buffer is donated and aliased
to the output, the in-place contract the RPA attention kernels and
kv_fused_rope_write share.

jax is imported eagerly: torch_tpu resolves the JAX-level signature with
inspect.signature(eval_str=True), so annotations have to be real names here.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import jax
import torch
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_OP_NAME = "sglang_tpu::kv_span_write"


def _kernel(dst_ref, src_ref, len_ref, rows_hbm, buf_hbm, out_hbm, sem):
    """Copy one span of rows into the buffer's flat slot range.

    dst_ref, src_ref and len_ref are the scalar-prefetched span table;
    rows_hbm [num_rows, *tail] and the buffer [num_blocks, page_size, *tail]
    stay in HBM, out_hbm is the buffer's aliased output.
    """
    del buf_hbm
    span = pl.program_id(0)
    n, src, dst = len_ref[span], src_ref[span], dst_ref[span]
    flat = out_hbm.reshape(out_hbm.shape[0] * out_hbm.shape[1], *out_hbm.shape[2:])

    # Span tables are padded to a static bound, so most grids end in empty spans.
    @pl.when(n > 0)
    def _copy():
        copy = pltpu.make_async_copy(
            rows_hbm.at[pl.ds(src, n)], flat.at[pl.ds(dst, n)], sem.at[0]
        )
        copy.start()
        copy.wait()


def kv_span_write_jax(
    dst_starts: jax.Array,
    src_starts: jax.Array,
    span_lens: jax.Array,
    rows: jax.Array,
    kv_buf: jax.Array,
) -> jax.Array:
    """Write ``rows`` spans into ``kv_buf`` in place and return the aliased buffer.

    Args:
      dst_starts: Destination flat slot per span [num_spans], int32.
      src_starts: First row of each span in ``rows`` [num_spans], int32.
      span_lens: Row count per span [num_spans], int32; 0 skips the span.
      rows: Rows in the buffer's tail layout [num_rows, kv_heads_packed,
        kv_packing, padded_head_dim], the buffer's dtype.
      kv_buf: Fused buffer [num_blocks, page_size, kv_heads_packed,
        kv_packing, padded_head_dim], donated.
    """
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=3,
        grid=(dst_starts.shape[0],),
        in_specs=[hbm, hbm],
        out_specs=hbm,
        scratch_shapes=[pltpu.SemaphoreType.DMA((1,))],
    )
    # Operands are (dst_starts, src_starts, span_lens, rows, kv_buf); the
    # buffer is aliased onto the single output.
    return pl.pallas_call(
        _kernel,
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct(kv_buf.shape, kv_buf.dtype),
        input_output_aliases={4: 0},
    )(dst_starts, src_starts, span_lens, rows, kv_buf)


@functools.lru_cache(maxsize=None)
def get_kv_span_write_op() -> Callable[..., None]:
    """Build or return the torch custom op wrapping ``kv_span_write_jax``."""
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.utils.pallas_export import tpu_info_for_cpu_export

    jax_kernel = _torch_pallas.custom_jax_kernel(
        kv_span_write_jax,
        name="jax_kv_span_write",
        donate_argnums=(4,),
        input_output_aliases={4: 0},
        warn_deprecated=False,
    )

    @torch.library.custom_op(_OP_NAME, mutates_args=("kv",), device_types=["tpu"])
    def op(
        dst_starts: torch.Tensor,
        src_starts: torch.Tensor,
        span_lens: torch.Tensor,
        rows: torch.Tensor,
        kv: torch.Tensor,
    ) -> None:
        with tpu_info_for_cpu_export():
            jax_kernel(dst_starts, src_starts, span_lens, rows, kv)

    @op.register_fake
    def _(
        dst_starts: torch.Tensor,
        src_starts: torch.Tensor,
        span_lens: torch.Tensor,
        rows: torch.Tensor,
        kv: torch.Tensor,
    ) -> None:
        return None

    return op


def kv_span_write_ref(
    rows: torch.Tensor,
    kv_buf: torch.Tensor,
    index: torch.Tensor,
    dst: torch.Tensor,
    src: torch.Tensor,
    lens: torch.Tensor,
) -> torch.Tensor:
    """Pure-torch reference of the compaction gather plus span write."""
    out = kv_buf.clone()
    flat = out.view(out.shape[0] * out.shape[1], *out.shape[2:])
    compact = rows[index]
    for d, s, n in zip(dst.tolist(), src.tolist(), lens.tolist()):
        if n:
            flat[d : d + n] = compact[s : s + n]
    return out
