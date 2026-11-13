# SPDX-License-Identifier: Apache-2.0
"""In-place row-run writes into a fused TPU KV layer buffer.

``write_rows`` copies ``rows[src : src + n]`` into token slots ``dst .. dst + n``
of the layer buffer for every run ``(src, dst, n)`` of a segment table. Each
run is staged through VMEM with double-buffered DMAs (a direct HBM-to-HBM DMA
runs an order of magnitude slower on TPU), and the output aliases the
buffer, so the layer is updated in place and never re-laid out. JAX is
imported only when a kernel is built, so the CPU fallback in
`write_rows_torch` works without it.
"""

from __future__ import annotations

import numpy as np
import torch

_OP_NAME = "sglang_tpu::kv_write_rows"
_OP = None
# Rows per staged DMA; two buffers of this many rows live in VMEM (8 MiB for
# 4 KiB rows).
PIECE_ROWS = 1024


def kv_write_segments(
    dest_slots: np.ndarray, valid: np.ndarray, max_segments: int | None = None
) -> tuple[torch.Tensor, int]:
    """Runs ``[3, max_segments]`` i32 of ``(source row, destination slot, length)``
    over consecutive valid rows landing in consecutive slots, and their count.

    The table defaults to one run per row; the attention backend passes
    ``write_run_bound`` and leaves padding rows out, so a warmup batch has no
    runs and a served batch about one per page.
    """
    slots = np.asarray(dest_slots, dtype=np.int64)
    valid = np.asarray(valid, dtype=bool)
    if max_segments is None:
        max_segments = max(int(slots.shape[0]), 1)
    continues = np.zeros(slots.shape[0], dtype=bool)
    if slots.shape[0] > 1:
        continues[1:] = valid[1:] & valid[:-1] & (slots[1:] == slots[:-1] + 1)
    starts = np.flatnonzero(valid & ~continues)
    ends = np.flatnonzero(valid & ~np.append(continues[1:], False))
    lengths = ends - starts + 1
    # Runs longer than the VMEM staging buffer become consecutive pieces.
    pieces = np.maximum(-(-lengths // PIECE_ROWS), 1) if lengths.size else lengths
    run_of_piece = np.repeat(np.arange(starts.shape[0]), pieces)
    offset = (
        np.arange(run_of_piece.shape[0]) - np.repeat(np.cumsum(pieces) - pieces, pieces)
    ) * PIECE_ROWS
    count = int(run_of_piece.shape[0])
    if count > max_segments:
        raise ValueError(
            f"{count} K/V write runs exceed the {max_segments} the write kernel "
            "takes; raise --page-size or lower the extend chunk."
        )
    segments = np.zeros((3, max_segments), dtype=np.int32)
    segments[0, :count] = starts[run_of_piece] + offset
    segments[1, :count] = slots[starts][run_of_piece] + offset
    segments[2, :count] = np.minimum(lengths[run_of_piece] - offset, PIECE_ROWS)
    return torch.from_numpy(segments), count


def write_run_bound(rows: int, *, page_size: int, blocks: int) -> int:
    """Runs a served batch of ``rows`` gathered rows can need: one per page and
    per zigzag block edge, plus the pieces long runs split into. The table is
    prefetched into SMEM, so it is sized for this rather than one run per row."""
    return -(-rows // page_size) + 2 * blocks + -(-rows // PIECE_ROWS) + 1


def _kernel(
    num_segments_ref,
    segments_ref,
    rows_ref,
    kv_in_ref,
    kv_out_ref,
    buf,
    in_sems,
    out_sems,
):
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    del kv_in_ref
    num_segments = jnp.clip(num_segments_ref[0], 0, segments_ref.shape[1])

    def _in_copy(i, slot):
        src = segments_ref[0, i]
        n = segments_ref[2, i]
        return pltpu.make_async_copy(
            src_ref=rows_ref.at[pl.ds(src, n)],
            dst_ref=buf.at[slot, pl.ds(0, n)],
            sem=in_sems.at[slot],
        )

    def _out_copy(i, slot):
        dst = segments_ref[1, i]
        n = segments_ref[2, i]
        return pltpu.make_async_copy(
            src_ref=buf.at[slot, pl.ds(0, n)],
            dst_ref=kv_out_ref.at[pl.ds(dst, n)],
            sem=out_sems.at[slot],
        )

    # Piece i reads into buffer i % 2 once piece i - 2 has left it; its write
    # overlaps the next piece's read.
    @pl.loop(0, num_segments)
    def _piece(i):
        slot = i % 2

        @pl.when(i >= 2)
        def _free_slot():
            _out_copy(i - 2, slot).wait()

        _in_copy(i, slot).start()
        _in_copy(i, slot).wait()
        _out_copy(i, slot).start()

    @pl.loop(jnp.maximum(num_segments - 2, 0), num_segments)
    def _drain(i):
        _out_copy(i, i % 2).wait()


def write_rows(num_segments, segments, rows, kv, *, interpret: bool = False):
    """Return ``kv`` with the runs of ``segments`` written; the output aliases ``kv``."""
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    flat = kv.reshape((-1, *kv.shape[2:]))
    out = pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(flat.shape, flat.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=(
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            ),
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=(
                pltpu.VMEM((2, PIECE_ROWS, *rows.shape[1:]), rows.dtype),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
            ),
            grid=(1,),
        ),
        input_output_aliases={3: 0},
        interpret=interpret,
        name="kv_write_rows",
    )(num_segments, segments, rows, flat)
    return out.reshape(kv.shape)


def get_write_rows_op():
    """Torch custom op ``kv_write_rows(num_segments, segments, rows, kv)`` mutating ``kv``."""
    global _OP
    if _OP is not None:
        return _OP

    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    # Tensors only: custom_jax_kernel traces keyword defaults like `interpret` since
    # torch_tpu 76284a1fa; register write_rows itself once they reach jax.jit static.
    def _write_rows(num_segments, segments, rows, kv):
        return write_rows(num_segments, segments, rows, kv)

    jax_kernel = _torch_pallas.custom_jax_kernel(
        _write_rows,
        name="jax_kv_write_rows",
        donate_argnums=(3,),
        input_output_aliases={3: 0},
        warn_deprecated=False,
    )

    @torch.library.custom_op(_OP_NAME, mutates_args=("kv",), device_types=["tpu"])
    def kv_write_rows(
        num_segments: torch.Tensor,
        segments: torch.Tensor,
        rows: torch.Tensor,
        kv: torch.Tensor,
    ) -> None:
        with tpu_info_for_cpu_export():
            jax_kernel(num_segments, segments, rows, kv)

    @kv_write_rows.register_fake
    def _(
        num_segments: torch.Tensor,
        segments: torch.Tensor,
        rows: torch.Tensor,
        kv: torch.Tensor,
    ) -> None:
        return None

    _OP = kv_write_rows
    return _OP


def write_rows_torch(
    num_segments: torch.Tensor,
    segments: torch.Tensor,
    rows: torch.Tensor,
    kv: torch.Tensor,
) -> None:
    """Write the runs of ``segments`` from ``rows`` into ``kv`` in place."""
    if kv.device.type == "tpu":
        get_write_rows_op()(num_segments, segments, rows, kv)
        return
    flat = kv.flatten(0, 1)
    for src, dst, n in segments[:, : int(num_segments)].T.tolist():
        flat[dst : dst + n] = rows[src : src + n]


_OPS: dict = {}


def _build_scatter_ops():
    import jax
    import jax.numpy as jnp
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.kv_page_write import read_pages, write_pages
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    def write_rows_scatter(block_ids, row_dst, rows, kv):
        n = block_ids.shape[0]
        page = kv.shape[1]
        pages = read_pages(block_ids, kv)  # [n, page, ...] pool dtype
        flat = pages.reshape(n * page, -1)
        flat = flat.at[row_dst].set(rows.astype(flat.dtype))
        return write_pages(block_ids, flat.reshape(n, *kv.shape[1:]), kv)

    rows_kernel = _torch_pallas.custom_jax_kernel(
        write_rows_scatter,
        name="jax_kv_write_rows_scatter",
        donate_argnums=(3,),
        input_output_aliases={3: 0},
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        "sglang_tpu::kv_write_rows_scatter", mutates_args=("kv",), device_types=["tpu"]
    )
    def kv_write_rows_scatter(
        block_ids: torch.Tensor,
        row_dst: torch.Tensor,
        rows: torch.Tensor,
        kv: torch.Tensor,
    ) -> None:
        with tpu_info_for_cpu_export():
            rows_kernel(block_ids, row_dst, rows, kv)

    @kv_write_rows_scatter.register_fake
    def _(
        block_ids: torch.Tensor,
        row_dst: torch.Tensor,
        rows: torch.Tensor,
        kv: torch.Tensor,
    ) -> None:
        return None

    _OPS["write_rows"] = kv_write_rows_scatter

    # ---- page-granular variant (layout aligned to the pool pages) --------
    def write_pages_from_rows(
        block_ids, page_src, boundary_pos, boundary_lead, rows, kv
    ):
        """kv[block_ids[i]] = rows.pages[page_src[i]], except that page
        block_ids[boundary_pos[j]] keeps its first boundary_lead[j] rows."""
        page = kv.shape[1]
        n = block_ids.shape[0]
        pages = rows.astype(kv.dtype).reshape(-1, page, rows.shape[-1])[
            page_src
        ]  # [n, page, E]
        keep = read_pages(block_ids[boundary_pos], kv).reshape(
            boundary_pos.shape[0], page, -1
        )
        row_ids = jax.lax.broadcasted_iota(
            jnp.int32, (boundary_pos.shape[0], page, 1), 1
        )
        blended = jnp.where(
            row_ids < boundary_lead[:, None, None], keep, pages[boundary_pos]
        )
        pages = pages.at[boundary_pos].set(blended)
        return write_pages(block_ids, pages.reshape(n, *kv.shape[1:]), kv)

    pages_kernel = _torch_pallas.custom_jax_kernel(
        write_pages_from_rows,
        name="jax_kv_write_pages_from_rows",
        donate_argnums=(5,),
        input_output_aliases={5: 0},
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        "sglang_tpu::kv_write_pages_from_rows",
        mutates_args=("kv",),
        device_types=["tpu"],
    )
    def kv_write_pages_from_rows(
        block_ids: torch.Tensor,
        page_src: torch.Tensor,
        boundary_pos: torch.Tensor,
        boundary_lead: torch.Tensor,
        rows: torch.Tensor,
        kv: torch.Tensor,
    ) -> None:
        with tpu_info_for_cpu_export():
            pages_kernel(block_ids, page_src, boundary_pos, boundary_lead, rows, kv)

    @kv_write_pages_from_rows.register_fake
    def _(block_ids, page_src, boundary_pos, boundary_lead, rows, kv) -> None:
        return None

    _OPS["write_pages_from_rows"] = kv_write_pages_from_rows


def ensure_kv_row_ops():
    """Build the ops once (outside torch.compile); returns the registry."""
    if not _OPS:
        _build_scatter_ops()
        from sglang_tpu.kv_page_write import (
            get_read_pages_op,
            get_write_pages_op,
        )

        get_read_pages_op()
        get_write_pages_op()
    return _OPS
