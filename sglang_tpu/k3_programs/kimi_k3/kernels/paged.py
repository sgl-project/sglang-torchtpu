# Adapted from vllm-project/vllm-torchtpu (Apache-2.0), commit c78529c; modified.
# Source: src/vllm_torchtpu/megakernel/kimi_k3/kernels/paged.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The serving step of the Kimi K3 decode megakernel over vLLM's unified pools.

vLLM keeps each MLA layer's latent pages in its unified pools (``[blocks, 128,
2, 640]`` bf16, token ``t`` of a block at ``[t // 2, t % 2]``; FP8 packs 512
tokens), each DCP rank holding its share of every sequence's pages, and the
KDA state in per-request slots. Two pieces serve it:

* :func:`decode_paged` -- one decode step for ``batch_size`` rows: gathers the
  rows' KDA slots, runs the fused decoder with the pools as read-only
  operands (``decode_local``), scatters the KDA rows back and commits the new
  latent rows with :func:`commit_latent_rows`. The fused decoder
  communicates across devices, and XLA copies every operand such a kernel
  updates in place (a full copy of the pools per step), so the only in-place
  pool update is the separate, non-communicating commit.
* :func:`commit_latent_rows` -- writes each row's new latent on the rank that
  owns its position.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from sglang_tpu.k3_programs.kimi_k3.config import active as _config
from sglang_tpu.k3_programs.kimi_k3.kernels import mla_dcp
from sglang_tpu.k3_programs.kimi_k3.kernels.decode_megakernel import (
    MLA_LAYERS,
    decode_local,
)
from sglang_tpu.k3_programs.kimi_k3.layout import (
    INDEX_SUBPAGE_ROWS,
    INDEX_SUBPAGE_TOKENS,
)


def latent_rows_from_handoff(new_keys):
    """``[batch, M, 640]`` latent rows from a DCP decoder's handoff (packed in
    new-key rows 0..2, see ``decode_megakernel.mla``)."""
    return jnp.concatenate(
        (new_keys[:, :, 0], new_keys[:, :, 1], new_keys[:, :, 2, :128]), axis=-1
    )


def _commit_latent_rows_fused(pools, rows, positions, blocks, owned, pool_map):
    """Alias all unified pools in one non-communicating cache commit.

    Overlap the layers' native-tile reads. Retire all writes before
    advancing to the next token so rows sharing a tile preserve each other.
    """
    batch, layers, _ = rows.shape
    packing, dtype = pools[0].shape[-2], pools[0].dtype
    page_tokens = pools[0].shape[-3] * packing
    tile_tokens = 8 * packing
    metadata = jnp.stack((positions[:batch], blocks, owned.astype(jnp.int32)))
    count = len(pools)

    def kernel(meta, values, *refs):
        outputs = refs[count : 2 * count]
        tiles, semaphores = refs[-2:]

        @pl.loop(0, batch)
        def commit_row(row):
            @pl.when(meta[2, row] != 0)
            def owned_row():
                position, block = meta[0, row], meta[1, row]
                first = pl.multiple_of((position % page_tokens // tile_tokens) * 8, 8)

                def normalized_block(index):
                    size = outputs[index].shape[0]
                    target = jnp.where(block < 0, block + size, block)
                    return target, (target >= 0) & (target < size)

                for layer, index in enumerate(pool_map):
                    target, valid = normalized_block(index)

                    @pl.when(valid)
                    def read(layer=layer, index=index, target=target):
                        tpu.make_async_copy(
                            outputs[index].at[target, pl.ds(first, 8)],
                            tiles.at[layer],
                            semaphores.at[layer],
                        ).start()

                for layer, index in enumerate(pool_map):
                    target, valid = normalized_block(index)

                    @pl.when(valid)
                    def update(layer=layer, index=index, target=target):
                        copy = tpu.make_async_copy(
                            outputs[index].at[target, pl.ds(first, 8)],
                            tiles.at[layer],
                            semaphores.at[layer],
                        )
                        copy.wait()
                        token = packing * jax.lax.broadcasted_iota(
                            jnp.int32, (8, packing, 640), 0
                        ) + jax.lax.broadcasted_iota(jnp.int32, (8, packing, 640), 1)
                        tiles[layer] = jnp.where(
                            token == position % tile_tokens,
                            values[row, layer][None, None, :],
                            tiles[layer],
                        )
                        tpu.make_async_copy(
                            tiles.at[layer],
                            outputs[index].at[target, pl.ds(first, 8)],
                            semaphores.at[layer],
                        ).start()

                for layer, index in enumerate(pool_map):
                    target, valid = normalized_block(index)

                    @pl.when(valid)
                    def retire(layer=layer, index=index, target=target):
                        tpu.make_async_copy(
                            tiles.at[layer],
                            outputs[index].at[target, pl.ds(first, 8)],
                            semaphores.at[layer],
                        ).wait()

    return tuple(
        pl.pallas_call(
            kernel,
            out_shape=tuple(jax.ShapeDtypeStruct(p.shape, p.dtype) for p in pools),
            grid_spec=tpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=(pl.BlockSpec(memory_space=tpu.VMEM),)
                + (pl.BlockSpec(memory_space=tpu.HBM),) * count,
                out_specs=(pl.BlockSpec(memory_space=tpu.HBM),) * count,
                scratch_shapes=(
                    tpu.VMEM((layers, 8, packing, 640), dtype),
                    tpu.SemaphoreType.DMA((layers,)),
                ),
            ),
            input_output_aliases={2 + i: i for i in range(count)},
            name="kimi_commit_unified_pools",
        )(metadata, rows.astype(dtype), *pools)
    )


def commit_latent_rows(
    pools,
    rows,
    block_tables,
    positions,
    pool_map,
    *,
    group=mla_dcp.GROUP,
    owner_table=None,
    global_pages=None,
):
    """Write each row's new latent into vLLM's unified pools on the rank that
    owns its position (XLA level, inside ``shard_map`` over ``tp``): MLA layer
    ``m``'s row at position ``p`` lands in ``pools[pool_map[m]]`` at token
    ``p % page_tokens`` of block ``block_tables[b, e]``, ``e = p // page_tokens``
    (global tables) or ``p // (page_tokens group)`` (local tables).
    ``owner_table``: as the decoder's ``dcp_owner_table`` (default the mesh
    position). ``global_pages``: default ``mla_dcp.GLOBAL_PAGES``."""
    if global_pages is None:
        global_pages = mla_dcp.GLOBAL_PAGES
    member = mla_dcp.owner_index(owner_table) % group
    pools = list(pools)
    packing = pools[0].shape[-2]
    page_tokens = pools[0].shape[-3] * packing
    pages = positions[: rows.shape[0]] // page_tokens
    owned = pages % group == member
    # Local tables hold one entry per ``group`` global pages.
    blocks = block_tables[
        jnp.arange(rows.shape[0]), pages if global_pages else pages // group
    ]
    # One fused commit for all MLA layers; the generic scatter below handles
    # other layouts and repeated pool maps (ordered by layer, not token).
    if (
        _config().fused_cache_commit
        and rows.shape[0] in (1, 2, 4, 8, 16, 32, 48, 64)
        and rows.shape[1:] == (len(pool_map), 640)
        and 0 < len(pool_map) <= MLA_LAYERS
        and len(set(pool_map)) == len(pool_map)
        and all(0 <= index < len(pools) for index in pool_map)
        and all(
            pool.shape[0] > 0
            and pool.shape[1:] == (128, packing, 640)
            and pool.dtype == pools[0].dtype
            for pool in pools
        )
    ):
        return _commit_latent_rows_fused(
            pools, rows, positions, blocks, owned, pool_map
        )
    pairs = positions[: rows.shape[0]] % page_tokens // packing
    halves = positions[: rows.shape[0]] % packing
    # One scatter per pool for all rows; rows this rank does not own point
    # past the pool and are dropped.
    for layer, index in enumerate(pool_map):
        pool = pools[index]
        targets = jnp.where(owned, blocks, pool.shape[0])
        pools[index] = pool.at[targets, pairs, halves].set(
            rows[:, layer].astype(pool.dtype), mode="drop"
        )
    return tuple(pools)


def commit_draft_rows(
    pools,
    rows,
    ids,
    rows_per,
    pages,
    pool_map,
    *,
    group=mla_dcp.GROUP,
    owner_table=None,
    global_pages=None,
    epilogue=None,
):
    """:func:`commit_latent_rows` for one request's DSpark verify rows, read
    from its packed ids (``MlaDsparkDraft._build_single``): ``rows``
    ``[layers, rows_per, 640]`` (cast to the pool dtype), row ``r`` at
    position ``ids[rows_per] + r`` on the block table ``ids[3 rows_per + 1
    :][:pages]``.

    One non-communicating kernel aliasing the pools computes each row's
    block and owner as the XLA commit does, reads each latent tile (8 pool
    rows, ``8 packing`` tokens) its owned rows touch once, merges those rows
    in VMEM in row order and writes the tile back once -- the bytes
    ``_commit_latent_rows_fused`` writes with a read, merge and write per
    row (a verify's consecutive rows share one or two tiles).

    ``epilogue``: ``(scalars, out_shapes, fn)`` -- more scalar operands
    (SMEM, after ``ids``), SMEM outputs (after the pools) and ``fn(ids,
    *scalars, *outputs)`` run after the commit; returns ``(*pools,
    *outputs)``. (XLA copies a communicating kernel's outputs once more on
    their way out of the program; this kernel's leave as they are.)"""
    if global_pages is None:
        global_pages = mla_dcp.GLOBAL_PAGES
    pools = tuple(pools)
    layers = len(pool_map)
    count = len(pools)
    packing, dtype = pools[0].shape[-2], pools[0].dtype
    page_tokens = pools[0].shape[-3] * packing
    tile_tokens = 8 * packing
    size = pools[0].shape[0]
    table_start = 3 * rows_per + 1
    if (
        rows.shape != (layers, rows_per, 640)
        or len(set(pool_map)) != layers
        or not all(0 <= index < count for index in pool_map)
        or any(pool.shape != pools[0].shape or pool.dtype != dtype for pool in pools)
        or pools[0].shape[1:] != (128, packing, 640)
        or ids.shape[0] < table_start + pages
    ):
        raise ValueError("commit_draft_rows: unexpected operand shapes")

    scalars, out_shapes, epilogue_fn = epilogue or ((), (), None)
    extra = len(scalars)

    def kernel(ids_ref, *refs):
        scalar_refs, (values, *refs) = refs[:extra], refs[extra:]
        outputs = refs[count : 2 * count]
        epilogue_out = refs[2 * count : 2 * count + len(out_shapes)]
        tiles, held, semaphores = refs[2 * count + len(out_shapes) :]
        member = mla_dcp.owner_index(owner_table) % group

        def tile_copies(target, first, write):
            copies = []
            for layer, index in enumerate(pool_map):
                window = outputs[index].at[target, pl.ds(first, 8)]
                source, destination = (
                    (tiles.at[layer], window) if write else (window, tiles.at[layer])
                )
                copies.append(
                    tpu.make_async_copy(source, destination, semaphores.at[layer])
                )
            return copies

        def transfer(target, first, write):
            copies = tile_copies(target, first, write)
            for item in copies:
                item.start()
            for item in copies:
                item.wait()

        def flush():  # write the held tile back
            @pl.when(held[0] != 0)
            def write_back():
                transfer(held[1], held[2], True)
                held[0] = 0

        held[0] = 0
        base = ids_ref[rows_per]
        token = packing * jax.lax.broadcasted_iota(
            jnp.int32, (8, packing, 640), 0
        ) + jax.lax.broadcasted_iota(jnp.int32, (8, packing, 640), 1)
        for r in range(rows_per):
            position = base + r
            page = position // page_tokens
            entry = page if global_pages else page // group
            # As XLA's gather, an entry past the table reads its last.
            entry = jnp.minimum(jnp.maximum(entry, 0), pages - 1)
            block = ids_ref[table_start + entry]
            target = jnp.where(block < 0, block + size, block)
            owned = (page % group == member) & (target >= 0) & (target < size)
            first = pl.multiple_of(position % page_tokens // tile_tokens * 8, 8)

            @pl.when(owned)
            def merge_row(r=r, position=position, target=target, first=first):
                same = (held[0] != 0) & (held[1] == target) & (held[2] == first)

                @pl.when(jnp.logical_not(same))
                def load():
                    flush()
                    transfer(target, first, False)
                    held[0] = 1
                    held[1] = target
                    held[2] = first

                for layer in range(layers):
                    tiles[layer] = jnp.where(
                        token == position % tile_tokens,
                        values[layer, r][None, None, :],
                        tiles[layer],
                    )

        flush()
        if epilogue_fn is not None:
            epilogue_fn(ids_ref, *scalar_refs, *epilogue_out)

    return tuple(
        pl.pallas_call(
            kernel,
            out_shape=tuple(jax.ShapeDtypeStruct(p.shape, p.dtype) for p in pools)
            + tuple(out_shapes),
            grid_spec=tpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1 + extra,
                in_specs=(pl.BlockSpec(memory_space=tpu.VMEM),)
                + (pl.BlockSpec(memory_space=pl.ANY),) * count,
                out_specs=(pl.BlockSpec(memory_space=pl.ANY),) * count
                + (pl.BlockSpec(memory_space=tpu.SMEM),) * len(out_shapes),
                scratch_shapes=(
                    tpu.VMEM((layers, 8, packing, 640), dtype),
                    tpu.SMEM((3,), jnp.int32),
                    tpu.SemaphoreType.DMA((layers,)),
                ),
            ),
            input_output_aliases={2 + extra + i: i for i in range(count)},
            name="kimi_commit_draft_rows",
        )(ids, *scalars, rows.astype(dtype), *pools)
    )


# The fused DSA commit's runs (a verify's consecutive tokens) and the runs of
# distinct blocks it keeps in flight together.
_RUN_ROWS = 8
_RUN_SLOTS = 8


def _dsa_commit_fusable(pools, index_pools, rows, records, block_tables, pool_map):
    """Whether :func:`_commit_dsa_rows_fused` takes these operands."""
    packing = pools[0].shape[-2]
    page_tokens = pools[0].shape[-3] * packing
    index_shape = (page_tokens // INDEX_SUBPAGE_TOKENS * INDEX_SUBPAGE_ROWS, 4, 128)
    return (
        _config().dsa_fused_commit
        and mla_dcp.GROUP == 32
        and rows.shape[1:] == (len(pool_map), 640)
        and records.shape == (rows.shape[0], len(pool_map), 2, 128)
        and records.dtype == jnp.int32
        and block_tables.ndim == 2
        and block_tables.shape[0] >= rows.shape[0]
        and len(index_pools) == len(pool_map)
        and len(set(pool_map)) == len(pool_map)
        and all(0 <= index < len(pools) for index in pool_map)
        and page_tokens % INDEX_SUBPAGE_TOKENS == 0
        and all(
            pool.shape == pools[0].shape and pool.dtype == pools[0].dtype
            for pool in pools
        )
        and pools[0].shape[0] > 0
        and pools[0].shape[1:] == (128, packing, 640)
        and all(
            pool.shape == (index_pools[0].shape[0], *index_shape)
            and pool.dtype == jnp.uint8
            for pool in index_pools
        )
        and index_pools[0].shape[0] > 0
    )


def _commit_dsa_rows_fused(
    pools,
    index_pools,
    rows,
    records,
    block_tables,
    positions,
    pool_map,
    *,
    owner_table=None,
):
    """Commit a Kimi-K3.1 step's latent rows (``commit_latent_rows``) and
    index-key records (``dsa_dcp.commit_index_rows``, raw ``[batch, 24, 2,
    128]`` int32) in one non-communicating kernel aliasing every pool.

    Owned rows form runs: consecutive rows at consecutive positions in one
    block and one 128-token index sub-page, at most ``_RUN_ROWS``. Per MLA
    layer a run reads one window of latent rows, one of key rows (never the
    scale row) and the scale row, merges its rows in VMEM and writes the
    windows back. Up to ``_RUN_SLOTS`` runs with distinct blocks are in
    flight together; a run whose block is in flight starts the next wave
    after the writes land, so rows sharing a tile (repeated null-block
    targets) land in row order."""
    batch = rows.shape[0]
    commit, scratch = _dsa_commit_parts(
        pools, index_pools, batch, block_tables, pool_map, owner_table=owner_table
    )
    count = len(pools) + len(index_pools)
    lat_count = len(pools)

    def kernel(position_ref, table_ref, rows_hbm, records_hbm, *refs):
        commit(
            position_ref,
            table_ref,
            rows_hbm,
            records_hbm,
            refs[count : count + lat_count],
            refs[count + lat_count : 2 * count],
            refs[2 * count :],
        )

    return pl.pallas_call(
        kernel,
        out_shape=tuple(
            jax.ShapeDtypeStruct(p.shape, p.dtype) for p in (*pools, *index_pools)
        ),
        grid_spec=tpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=(pl.BlockSpec(memory_space=pl.ANY),) * (2 + count),
            out_specs=(pl.BlockSpec(memory_space=pl.ANY),) * count,
            scratch_shapes=scratch,
        ),
        input_output_aliases={4 + i: i for i in range(count)},
        compiler_params=tpu.CompilerParams(
            disable_bounds_checks=not _config().bounds_checks
        ),
        name="kimi_commit_dsa_pools",
    )(
        positions[:batch],
        block_tables.reshape(-1),
        rows.astype(pools[0].dtype),
        records,
        *pools,
        *index_pools,
    )


def _dsa_commit_parts(
    pools, index_pools, batch, block_tables, pool_map, *, owner_table=None, handoff=None
):
    """:func:`_commit_dsa_rows_fused`'s kernel body and its VMEM/SMEM scratch:
    ``(commit, scratch)``, ``commit(position_ref, table_ref, rows_hbm,
    records_hbm, lat_out, idx_out, scratch_refs)``. ``handoff`` (the
    decoder's new-key handoff shape ``[batch, 24, H, 256]``, BF16; see
    ``latent_rows_from_handoff``): ``rows_hbm`` is the handoff and the rows
    are built in VMEM, ``[batch, 24, 640]`` in the pool dtype (FP8: the
    convert XLA's ``astype`` lowers to), on the first wave (ranks without
    owned rows read neither)."""
    layers = len(pool_map)
    packing, dtype = pools[0].shape[-2], pools[0].dtype
    block_rows = pools[0].shape[-3]
    page_tokens = block_rows * packing
    entries = block_tables.shape[-1]
    run = min(batch, _RUN_ROWS)
    slots = min(batch, _RUN_SLOTS)
    lat_window = (run + packing - 2) // packing + 1
    key_window = (run + 2) // 4 + 1
    lat_blocks, idx_blocks = pools[0].shape[0], index_pools[0].shape[0]
    sub_tokens, sub_rows = INDEX_SUBPAGE_TOKENS, INDEX_SUBPAGE_ROWS
    group = mla_dcp.GROUP
    global_pages = mla_dcp.GLOBAL_PAGES

    def div(x, d):  # floor division and modulo (a shift / mask for 2^k)
        return x >> (d.bit_length() - 1) if d & (d - 1) == 0 else x // d

    def mod(x, d):
        return x & (d - 1) if d & (d - 1) == 0 else x % d

    def commit(
        position_ref, table_ref, rows_hbm, records_hbm, lat_out, idx_out, scratch_refs
    ):
        (lat_buf, key_buf, scale_buf, row_buf, record_buf) = scratch_refs[:5]
        # slot: the wave's runs (first row, rows, targets); wave: (runs, loaded)
        slot, wave, sems = scratch_refs[5:8]
        member = mla_dcp.owner_index(owner_table) % group
        if handoff is None:
            row_copy = tpu.make_async_copy(rows_hbm, row_buf, sems.at[3])
        else:
            handoff_buf, staging = scratch_refs[8:10]
            row_copy = tpu.make_async_copy(rows_hbm, handoff_buf, sems.at[3])
        record_copy = tpu.make_async_copy(records_hbm, record_buf, sems.at[4])

        def build_rows():  # the handoff's latent rows, a row at a time
            def build_row(b, carry):
                # Each layer's three handoff rows as FP32 128-lane tiles
                # (staging tile 2 m + j: lanes 128 j.. of layer m's rows),
                # then each part of the 640 lanes for every layer at once.
                for layer in range(layers):
                    value = handoff_buf[b, layer, 0:3].astype(jnp.float32)
                    for j in range(2):
                        staging[pl.ds(8 * (2 * layer + j), 3), :] = value[
                            :, 128 * j : 128 * (j + 1)
                        ]
                parts = [
                    staging[pl.ds(8 * j + k, layers, stride=16), :]
                    for k, j in ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0))
                ]
                row_buf[b] = (
                    jnp.concatenate(parts, axis=1).astype(jnp.bfloat16).astype(dtype)
                )
                return carry

            jax.lax.fori_loop(0, batch, build_row, 0)

        def place(b):  # (position, block, owned) of row b, as the XLA commits
            position = position_ref[b]
            page = div(position, page_tokens)
            entry = page if global_pages else div(page, group)
            entry = jnp.where(entry < 0, entry + entries, entry)
            entry = jnp.minimum(jnp.maximum(entry, 0), entries - 1)
            return position, table_ref[b * entries + entry], mod(page, group) == member

        def targets(block):  # the normalized latent / index block, else -1
            def target(size):
                normal = jnp.where(block < 0, block + size, block)
                return jnp.where((normal >= 0) & (normal < size), normal, -1)

            return target(lat_blocks), target(idx_blocks)

        def windows(w):
            first, lat, idx = slot[0, w], slot[2, w], slot[3, w]
            offset = mod(position_ref[first], page_tokens)
            sub = div(offset, sub_tokens) * sub_rows
            lat_row = jnp.minimum(div(offset, packing), block_rows - lat_window)
            key_row = sub + jnp.minimum(
                div(mod(offset, sub_tokens), 4), sub_rows - 1 - key_window
            )
            return first, offset, sub, lat_row, key_row, lat, idx

        def transfer(w, write):
            _, _, sub, lat_row, key_row, lat, idx = windows(w)

            def copy(hbm, vmem, sem):
                return tpu.make_async_copy(
                    *((vmem, hbm) if write else (hbm, vmem)), sems.at[sem]
                )

            @pl.when(lat >= 0)
            def _():
                for layer, index in enumerate(pool_map):
                    copy(
                        lat_out[index].at[lat, pl.ds(lat_row, lat_window)],
                        lat_buf.at[w, layer],
                        0,
                    ).start()

            @pl.when(idx >= 0)
            def _():
                for layer in range(layers):
                    copy(
                        idx_out[layer].at[idx, pl.ds(key_row, key_window)],
                        key_buf.at[w, layer],
                        1,
                    ).start()
                    copy(
                        idx_out[layer].at[idx, sub + sub_rows - 1],
                        scale_buf.at[w, layer],
                        2,
                    ).start()

        def settle(w):  # a slot's copies, one wait per kind
            _, _, _, _, _, lat, idx = windows(w)

            @pl.when(lat >= 0)
            def _():
                tpu.make_async_copy(lat_buf.at[w], lat_buf.at[w], sems.at[0]).wait()

            @pl.when(idx >= 0)
            def _():
                tpu.make_async_copy(key_buf.at[w], key_buf.at[w], sems.at[1]).wait()
                tpu.make_async_copy(scale_buf.at[w], scale_buf.at[w], sems.at[2]).wait()

        def merge(w):
            first, offset, sub, lat_row, key_row, _, _ = windows(w)

            @pl.loop(0, slot[1, w])
            def _(i):
                row, token = first + i, offset + i
                pair = div(token, packing) - lat_row
                lane = mod(token, sub_tokens)
                key = sub + div(lane, 4) - key_row
                half = jax.lax.broadcasted_iota(jnp.int32, (packing, 640), 0)
                sublane = jax.lax.broadcasted_iota(jnp.int32, (4, 128), 0)
                lanes = jax.lax.broadcasted_iota(jnp.int32, (4, 128), 1)
                for layer in range(layers):
                    lat_buf[w, layer, pair] = jnp.where(
                        half == mod(token, packing),
                        row_buf[row, layer][None, :],
                        lat_buf[w, layer, pair],
                    )
                    key_buf[w, layer, key] = jnp.where(
                        sublane == mod(token, 4),
                        record_buf[row, layer, 0:1] & 0xFF,
                        key_buf[w, layer, key].astype(jnp.int32),
                    ).astype(jnp.uint8)
                    scale_buf[w, layer] = jnp.where(
                        (sublane == 0) & (lanes == lane),
                        tpu.roll(record_buf[row, layer, 1:2] & 0xFF, lane, 1),
                        scale_buf[w, layer].astype(jnp.int32),
                    ).astype(jnp.uint8)

        def commit_wave(waiting, loaded):
            @pl.when(loaded == 0)
            def _():
                row_copy.start()
                record_copy.start()

            pl.loop(0, waiting)(lambda w: transfer(w, False))

            @pl.when(loaded == 0)
            def _():
                row_copy.wait()
                record_copy.wait()
                if handoff is not None:
                    build_rows()

            pl.loop(0, waiting)(settle)
            pl.loop(0, waiting)(merge)
            pl.loop(0, waiting)(lambda w: transfer(w, True))
            pl.loop(0, waiting)(settle)

        def plan(b, carry):
            # Rows extend the open run or close it; a closed run joins the
            # wave unless its block is in flight or the wave is full (then
            # the wave commits first). Rows batch, batch + 1 flush.
            first, size, block, last = carry
            position, row_block, owned = place(jnp.minimum(b, batch - 1))
            owned = owned & (b < batch)
            joins = (
                (size > 0)
                & owned
                & (row_block == block)
                & (position == last + 1)
                & (mod(position, sub_tokens) != 0)
                & (size < run)
            )
            closes = (size > 0) & ~joins
            final = b == batch + 1

            @pl.when(closes | final)
            def _():
                lat, idx = targets(block)
                busy = final | (wave[0] == slots)
                for w in range(slots):
                    busy = busy | (
                        (w < wave[0]) & ((slot[2, w] == lat) | (slot[3, w] == idx))
                    )

                @pl.when(busy & (wave[0] > 0))
                def _():
                    commit_wave(wave[0], wave[1])
                    wave[0] = 0
                    wave[1] = 1

                @pl.when(closes)
                def _():
                    slot[0, wave[0]] = first
                    slot[1, wave[0]] = size
                    slot[2, wave[0]] = lat
                    slot[3, wave[0]] = idx
                    wave[0] = wave[0] + 1

            first = jnp.where(joins, first, b)
            size = jnp.where(joins, size + 1, owned.astype(jnp.int32))
            block = jnp.where(joins, block, row_block)
            return first, size, block, position

        wave[0] = 0
        wave[1] = 0
        zero = jnp.int32(0)
        jax.lax.fori_loop(0, batch + 2, plan, (zero,) * 4)

    scratch = (
        tpu.VMEM((slots, layers, lat_window, packing, 640), dtype),
        tpu.VMEM((slots, layers, key_window, 4, 128), jnp.uint8),
        tpu.VMEM((slots, layers, 4, 128), jnp.uint8),
        tpu.VMEM((batch, layers, 640), dtype),
        tpu.VMEM((batch, layers, 2, 128), jnp.int32),
        tpu.SMEM((4, slots), jnp.int32),
        tpu.SMEM((2,), jnp.int32),
        tpu.SemaphoreType.DMA((5,)),
    )
    if handoff is not None:
        scratch += (
            tpu.VMEM(tuple(handoff), jnp.bfloat16),
            tpu.VMEM((layers * 16, 128), jnp.float32),
        )
    return commit, scratch


def _commit_kda_states(
    conv_slots, recurrent_slots, conv, recurrent, slots, source_rows=None
):
    """Copy complete KDA rows back without XLA's per-row scatter loop.

    Row writes finish in input order, including repeated null-slot indices.
    The banks alias their outputs so untouched slots retain their bytes.
    With ``source_rows``, DMA only the selected verifier post-states directly
    from their HBM buffer; do not gather a second intermediate with XLA.
    """
    batch = slots.shape[0]
    selected = source_rows is not None
    prefetch = 2 if selected else 1

    def body(indices, *args):
        sources, args = (args[0], args[1:]) if selected else (None, args)
        (
            old_conv,
            old_state,
            new_conv,
            new_state,
            out_conv,
            out_state,
            conv_buffer,
            state_buffer,
            semaphores,
        ) = args

        @pl.loop(0, batch)
        def row(i):
            source = sources[i] if sources is not None else i
            reads = (
                tpu.make_async_copy(
                    new_conv.at[pl.ds(source, 1)], conv_buffer, semaphores.at[0]
                ),
                tpu.make_async_copy(
                    new_state.at[pl.ds(source, 1)], state_buffer, semaphores.at[1]
                ),
            )
            for item in reads:
                item.start()
            for item in reads:
                item.wait()
            target = indices[i]
            writes = (
                tpu.make_async_copy(
                    conv_buffer, out_conv.at[pl.ds(target, 1)], semaphores.at[0]
                ),
                tpu.make_async_copy(
                    state_buffer, out_state.at[pl.ds(target, 1)], semaphores.at[1]
                ),
            )
            for item in writes:
                item.start()
            for item in writes:
                item.wait()

    return pl.pallas_call(
        body,
        name="kda_selected_state_commit" if selected else "kda_state_commit",
        out_shape=(
            jax.ShapeDtypeStruct(conv_slots.shape, conv_slots.dtype),
            jax.ShapeDtypeStruct(recurrent_slots.shape, recurrent_slots.dtype),
        ),
        input_output_aliases={prefetch: 0, prefetch + 1: 1},
        grid_spec=tpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=prefetch,
            in_specs=[pl.BlockSpec(memory_space=tpu.HBM)] * 4,
            out_specs=[pl.BlockSpec(memory_space=tpu.HBM)] * 2,
            scratch_shapes=(
                tpu.VMEM((1, *conv.shape[1:]), conv.dtype),
                tpu.VMEM((1, *recurrent.shape[1:]), recurrent.dtype),
                tpu.SemaphoreType.DMA((2,)),
            ),
        ),
    )(
        slots,
        *((source_rows,) if selected else ()),
        conv_slots,
        recurrent_slots,
        conv,
        recurrent,
    )


def commit_verify_tail(
    conv_slots,
    recurrent_slots,
    conv,
    recurrent,
    slots,
    source_rows,
    pools,
    index_pools,
    handoff,
    records,
    block_tables,
    positions,
    pool_map,
    *,
    owner_table=None,
):
    """A verify's commits in one non-communicating kernel
    (KernelConfig.verify_fused_tail): the accepted KDA post-states
    (:func:`_commit_kda_states` with ``source_rows``) and the step's latent
    rows and DSA index keys (:func:`_commit_dsa_rows_fused`), the latent rows
    built in VMEM from the decoder's BF16 new-key ``handoff`` (``[batch, 24,
    H, 256]``, ``latent_rows_from_handoff``; FP8 pools: the same converts).
    The first KDA row's post-state reads start before the DSA commit, which
    runs under them; then the rows' writes, in row order. Returns
    ``(conv_slots, recurrent_slots, *pools, *index_pools)`` with every
    operand aliased (the same bytes as the two kernels)."""
    batch, requests = handoff.shape[0], slots.shape[0]
    pools, index_pools = tuple(pools), tuple(index_pools)
    commit, dsa_scratch = _dsa_commit_parts(
        pools,
        index_pools,
        batch,
        block_tables,
        pool_map,
        owner_table=owner_table,
        handoff=handoff.shape,
    )
    count = len(pools) + len(index_pools)
    lat_count = len(pools)
    dsa_count = len(dsa_scratch)

    def kernel(
        position_ref,
        table_ref,
        target_ref,
        source_ref,
        handoff_hbm,
        records_hbm,
        old_conv,
        old_state,
        new_conv,
        new_state,
        *refs,
    ):
        del old_conv, old_state  # aliased by out_conv / out_state
        out_conv, out_state = refs[count : count + 2]
        lat_out = refs[count + 2 : count + 2 + lat_count]
        idx_out = refs[count + 2 + lat_count : 2 * count + 2]
        scratch = refs[2 * count + 2 :]
        conv_buffer, state_buffer, semaphores = scratch[dsa_count:]

        def reads(i):
            source = source_ref[i]
            return (
                tpu.make_async_copy(
                    new_conv.at[pl.ds(source, 1)], conv_buffer, semaphores.at[0]
                ),
                tpu.make_async_copy(
                    new_state.at[pl.ds(source, 1)], state_buffer, semaphores.at[1]
                ),
            )

        def writes(i):
            target = target_ref[i]
            return (
                tpu.make_async_copy(
                    conv_buffer, out_conv.at[pl.ds(target, 1)], semaphores.at[0]
                ),
                tpu.make_async_copy(
                    state_buffer, out_state.at[pl.ds(target, 1)], semaphores.at[1]
                ),
            )

        for item in reads(0):
            item.start()
        commit(
            position_ref,
            table_ref,
            handoff_hbm,
            records_hbm,
            lat_out,
            idx_out,
            scratch[:dsa_count],
        )

        @pl.loop(0, requests)
        def row(i):
            @pl.when(i > 0)
            def _():
                for item in reads(i):
                    item.start()

            for item in reads(i):
                item.wait()
            for item in writes(i):
                item.start()
            for item in writes(i):
                item.wait()

    operands = (conv_slots, recurrent_slots, *pools, *index_pools)
    # The decoder's outputs stay in HBM (with pl.ANY XLA stages the ones that
    # fit into VMEM with copies first); the aliased banks and pools as the
    # two kernels take them.
    hbm, anywhere = (
        pl.BlockSpec(memory_space=tpu.HBM),
        pl.BlockSpec(memory_space=pl.ANY),
    )
    return pl.pallas_call(
        kernel,
        out_shape=tuple(jax.ShapeDtypeStruct(p.shape, p.dtype) for p in operands),
        grid_spec=tpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            in_specs=(hbm, hbm, anywhere, anywhere, hbm, hbm) + (anywhere,) * count,
            out_specs=(pl.BlockSpec(memory_space=pl.ANY),) * (2 + count),
            scratch_shapes=(
                *dsa_scratch,
                tpu.VMEM((1, *conv.shape[1:]), conv.dtype),
                tpu.VMEM((1, *recurrent.shape[1:]), recurrent.dtype),
                tpu.SemaphoreType.DMA((2,)),
            ),
        ),
        input_output_aliases={6: 0, 7: 1, **{10 + i: 2 + i for i in range(count)}},
        compiler_params=tpu.CompilerParams(
            disable_bounds_checks=not _config().bounds_checks,
            vmem_limit_bytes=_config().vmem_limit_mb * 1024**2,
        ),
        name="kimi_commit_verify_tail",
    )(
        positions[:batch],
        block_tables.reshape(-1),
        slots,
        source_rows,
        handoff,
        records,
        conv_slots,
        recurrent_slots,
        conv,
        recurrent,
        *pools,
        *index_pools,
    )


def decode_paged(
    weights,
    conv_slots,
    recurrent_slots,
    pool,
    tokens,
    positions,
    block_tables,
    slots,
    *,
    batch_size,
    dcp_pool_map,
    dcp_owner_table=None,
    state_row=None,
    defer_kda_commit=False,
    defer_latent_commit=False,
    dsa_pools=0,
    dsa_centres=None,
    **decode_options,
):
    """One decode step on this rank (inside a ``tp`` shard_map).

    ``conv_slots [S, 69, 3, 3, 3, 128]`` / ``recurrent_slots [S, 69, 3, 128,
    128]`` hold every request's KDA state; row ``b`` uses slot ``slots[b]``
    (padding rows the null slot, pages the null page). ``pool`` is the tuple
    of vLLM's unified pools (MLA layer ``m`` in ``pool[dcp_pool_map[m]]``),
    read in place; ``block_tables`` are the DCP tables. Returns ``(logits
    [batch, V/32], conv_slots, recurrent_slots, pool)`` with the slots and new
    latent rows committed (callers donate them); ``aux_hidden_layers`` adds
    ``aux [L, batch, hidden]`` after the logits.

    Speculative verify: the rows are one sequence's anchor and drafts;
    ``state_row`` (traced) picks the row whose input KDA state the sequence
    continues from. Every row's post-state and latent are committed; rejected
    drafts lie past the next step's pool limit. ``defer_kda_commit``
    (verifiers): return per-row post-states (under attention DP this group's
    rows, shard-major); the caller commits accepted rows.

    Kimi-K3.1 (``dsa_pools`` = 24): the last ``dsa_pools`` entries of ``pool``
    are the MLA layers' DSA index-key pools; the kernel selects each row's
    tokens from them and every row's new index keys are committed beside its
    latents (returned in ``pool``'s order).

    ``dsa_centres`` (``KernelConfig.dsa_seeded_window``): int32 ``[S, 24,
    R]``, the selection's window centres per KDA slot, MLA layer and row of
    the request's step (``b % sequence_rows``; plain steps row 0), returned
    updated after the other outputs (every row of a live slot; the null slot
    0 is never written, so every rank keeps the same array). Any values give
    the same selection: they only decide whether the selection takes two
    count exchanges or three. Without ``dsa_centres`` a seeded selection
    starts every row from the constant centre 0.

    ``defer_latent_commit`` (verifiers with ``defer_kda_commit``,
    KernelConfig.verify_fused_tail): where the fused DSA commit applies, the
    pools come back uncommitted and the outputs end with ``(handoff,
    records, block_tables, positions)`` for :func:`commit_verify_tail` (else
    None: committed here).
    """
    if (
        tokens.shape != (batch_size,)
        or positions.shape != (batch_size,)
        or slots.shape != (batch_size,)
    ):
        raise ValueError("tokens, positions and slots must have one entry per row")
    dp = decode_options.get("attention_dp", 1)
    group_rows = None
    state_index = slots
    if dp > 1:
        # This rank's DP group attends its rows of each local row owner's rows
        # (see decode_megakernel.stack); KDA state row k * G + p holds shard
        # k of group row p, at bank entry ``s + k * n`` for the row's slot s
        # of the group's n slots (G = batch / dp; pools.group_slots).
        owners = decode_options.get("sequence_parallel_size", 4)
        per_pair = batch_size // dp // owners
        data_group = jax.lax.axis_index("tp") // (32 // dp)
        row = jnp.arange(batch_size // dp)
        group_rows = (
            row // per_pair * (batch_size // owners)
            + data_group * per_pair
            + row % per_pair
        )
        group_slots = conv_slots.shape[0] // dp
        local = slots[group_rows]
        local = jnp.where(local == 0, 0, local - data_group * (group_slots - 1))
        state_index = jnp.concatenate([local + k * group_slots for k in range(dp)])
    # The stack reads the input KDA states from their slots itself where it
    # can (DP1 above four rows or a selected verify row; DP2 at B2..B128).
    # DP2 indices already include each held head shard's bank offset.
    in_place = (
        (state_row is not None or batch_size > 4)
        and dp == 1
        or dp == 2
        and batch_size in (2, 4, 8, 16, 32, 64, 96, 128)
    )
    if in_place:
        conv, recurrent = conv_slots, recurrent_slots
    else:
        conv, recurrent = conv_slots[state_index], recurrent_slots[state_index]
    pools = tuple(pool)
    index_pools = None
    if dsa_pools:
        pools, index_pools = pools[:-dsa_pools], pools[-dsa_pools:]
    from sglang_tpu.k3_programs.kimi_k3.kernels import dsa_dcp

    seeded = bool(dsa_pools) and dsa_dcp.seeded_window(batch_size)
    centres_in = None
    sequence_rows = decode_options.get("sequence_rows", 1)
    if seeded:
        centres_in = (
            jnp.zeros((dsa_pools * batch_size,), jnp.int32)
            if dsa_centres is None
            else dsa_dcp.gather_centres(dsa_centres, slots, sequence_rows)
        )
    result = decode_local(
        weights,
        (conv, recurrent, *pools),
        tokens,
        positions,
        batch_size=batch_size,
        block_tables=block_tables,
        dcp_pool_map=dcp_pool_map,
        dcp_owner_table=dcp_owner_table,
        state_row=state_row,
        state_slots=state_index if in_place else None,
        live_rows=slots != 0,
        index_pools=index_pools,
        dsa_centres=centres_in,
        **decode_options,
    )
    aux = None
    if decode_options.get("aux_hidden_layers"):
        logits, aux, (conv, recurrent, new_keys, new_values, *records) = result
    else:
        logits, (conv, recurrent, new_keys, new_values, *records) = result
    if seeded:
        *records, centres_out = records
        if dsa_centres is not None:
            dsa_centres = dsa_dcp.scatter_centres(
                dsa_centres, slots, sequence_rows, centres_out
            )
    # DP2 returns shard-major rows; commit each to its shard-bank index,
    # preserving untouched slots and ordered null-slot writes.
    if defer_kda_commit:
        # Keep the input banks alive until acceptance. Return the native
        # post-states without touching persistent slots or gathering rows.
        conv_slots, recurrent_slots = conv, recurrent
    elif batch_size > 16 and dp in (1, 2):
        conv_slots, recurrent_slots = _commit_kda_states(
            conv_slots, recurrent_slots, conv, recurrent, state_index
        )
    else:
        conv_slots = conv_slots.at[state_index].set(conv)
        recurrent_slots = recurrent_slots.at[state_index].set(recurrent)
    latents = latent_rows_from_handoff(new_keys)
    if group_rows is not None:
        # The handoff holds the group's rows first; only their owners commit.
        latents, block_tables, positions = (
            latents[: group_rows.shape[0]],
            block_tables[group_rows],
            positions[group_rows],
        )
    pending = None
    if (
        index_pools is not None
        and group_rows is None
        and _dsa_commit_fusable(
            pools, index_pools, latents, records[0], block_tables, dcp_pool_map
        )
    ):
        if defer_latent_commit and defer_kda_commit:
            # The caller commits them with the accepted KDA states.
            pending = (new_keys, records[0], block_tables, positions)
            pool = (*pools, *index_pools)
        else:
            pool = _commit_dsa_rows_fused(
                pools,
                index_pools,
                latents,
                records[0],
                block_tables,
                positions,
                dcp_pool_map,
                owner_table=dcp_owner_table,
            )
        index_pools = None  # committed with the latents
    else:
        pool = commit_latent_rows(
            pools,
            latents,
            block_tables,
            positions,
            dcp_pool_map,
            owner_table=dcp_owner_table,
        )
    if index_pools is not None:
        (records,) = records  # [batch, 24, 2, 128] int32: key bytes | UE8M0 byte
        records = jnp.concatenate(
            (records[:, :, 0, :], records[:, :, 1, :1]), axis=-1
        ).astype(jnp.uint8)
        index_pools = dsa_dcp.commit_index_rows(
            index_pools,
            records,
            block_tables,
            positions,
            group=32,
            owner_table=dcp_owner_table,
            global_pages=mla_dcp.GLOBAL_PAGES,
        )
        pool = (*pool, *index_pools)
    extra = () if dsa_centres is None else (dsa_centres,)
    if defer_latent_commit:
        extra += (pending,)
    if aux is not None:
        return logits[0], aux, conv_slots, recurrent_slots, pool, *extra
    return logits[0], conv_slots, recurrent_slots, pool, *extra
