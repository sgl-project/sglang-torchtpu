# SPDX-License-Identifier: Apache-2.0
"""Attention chunk geometry and page-table padding."""

from typing import Any, NamedTuple

import jax.numpy as jnp

from sglang_tpu.muse_megakernel.mk.consts import LANE


class _AttnGeom(NamedTuple):
    """Chunk geometry of one attention body: ``cp`` pages per chunk, ``kr`` ring slots fetched in

    groups of ``kg``. ``is_global`` is a Python bool when the body is specialized
    per layer
    type, else the layer's traced flag.
    """

    cp: int
    kg: int
    kr: int
    is_global: Any


# A dynamic DMA offset into a 1-D int32 HBM ref (one 128-lane tile per row) must be a whole
# number of lane tiles: Mosaic rejects word-aligned (8-entry) offsets there. The table is
# padded to whole tiles so an aligned window can reach its last entries.
TABLE_ALIGN = LANE


def table_entries_padded(c) -> int:
    return -(-c.n_pages_entries // TABLE_ALIGN) * TABLE_ALIGN


def pad_table(c, table):
    """pages_swa for swa_window_table kernels: whole TABLE_ALIGN tiles (pad never read)."""
    tp_pad = table_entries_padded(c)
    pad = tp_pad - c.n_pages_entries
    if pad == 0 or table.shape[0] == c.n_swa_groups * tp_pad:
        return table
    if c.n_swa_groups == 1:
        return jnp.pad(table, (0, pad))
    return jnp.pad(
        table.reshape(c.n_swa_groups, c.n_pages_entries), ((0, 0), (0, pad))
    ).reshape(-1)


def swa_window_entries(c) -> int:
    """pages_swa entries one request's window is read from: the window's chunks (validate()
    bounds them by the SWA ring), widened so the DMA starts on a TABLE_ALIGN boundary."""
    cp = c.swa_chunk_pages or c.chunk_pages
    cpt = cp * c.page_size
    pages = ((c.window + cpt - 2) // cpt + 1) * cp
    return -(-(pages + TABLE_ALIGN - 1) // TABLE_ALIGN) * TABLE_ALIGN
