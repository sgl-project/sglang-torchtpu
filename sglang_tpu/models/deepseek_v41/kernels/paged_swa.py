# SPDX-License-Identifier: Apache-2.0
"""Native BF16 page production for direct-DMA sliding-window attention."""

import functools
from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc


@functools.partial(jax.jit, donate_argnums=(0,))
def write_swa_pages(
    pages: jax.Array | tuple[jax.Array, jax.Array],
    records: jax.Array,
    values: jax.Array,
    *,
    prepared: tuple | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Write physical token records, preserving adjacent BF16 rows and null page.

    Records address concatenated banks. Negative and out-of-range records are
    dropped; page zero is reserved. Valid records must be unique within a call.
    Each physical word row has one writer, preserving both halves of packed
    words. The aliased output preserves untouched rows without copying the pool.
    """
    banks = pages if isinstance(pages, tuple) else (pages,)
    if (isinstance(pages, tuple) and len(banks) != 2) or any(
        bank.ndim != 3
        or bank.shape[0] < 1
        or bank.shape[-1] != 512
        or bank.dtype != jnp.bfloat16
        for bank in banks
    ):
        raise ValueError("SWA owners must be BF16 [pages,page_size,512]")
    page_size = banks[0].shape[1]
    if (
        page_size not in (8, 16, 32, 64) and (page_size < 128 or page_size % 128)
    ) or any(bank.shape[1] != page_size for bank in banks):
        raise ValueError("SWA banks need matching page sizes: 8,16,32,64 or 128*n")
    if values.ndim != 2 or values.shape[1] != 512 or values.dtype != jnp.bfloat16:
        raise ValueError("SWA values must be BF16 [N,512]")
    if records.shape != values.shape[:1] or records.dtype != jnp.int32:
        raise ValueError("SWA records must be INT32 [N]")
    if not records.size:
        return pages
    info = pltpu.get_tpu_info().sparse_core
    assert info is not None, "SWA publication requires SparseCore"
    lanes, cores = info.num_lanes, info.num_cores * info.num_subcores
    mesh = plsc.VectorSubcoreMesh(
        num_cores=info.num_cores,
        num_subcores=info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    if prepared is None:
        from sglang_tpu.models.deepseek_v41.kernels import decode_preparation as prep

        if prep.admit(
            "swa_pack",
            None if page_size == 64 else f"SWA page size {page_size}, kernel 64",
            tokens=records.shape[0],
            buckets=prep.SWA_PACK_TOKENS,
        ):
            prepared = prep.swa_pack(banks, records, values)
    result, base = [], 0
    for bank in banks:
        capacity = bank.shape[0] * page_size
        if prepared is not None:
            destinations, masks, words = prepared[len(result)]
        else:
            valid = (records >= max(base, page_size)) & (records < base + capacity)
            local = jnp.where(valid, records - base, capacity)
            order = jnp.argsort(local, stable=True)
            ids = local[order]
            next_ids = jnp.concatenate((ids[1:], jnp.array([capacity], jnp.int32)))
            # TC BF16 tiles pack adjacent token rows into one physical word row.
            pairs = ids // 2
            boundary = (ids < capacity) & jnp.concatenate(
                (jnp.ones(1, jnp.bool_), pairs[1:] != pairs[:-1])
            )
            even = ids % 2 == 0
            has_odd = (~even) | (next_ids // 2 == pairs)
            bits = jax.lax.bitcast_convert_type(values[order], jnp.uint16).astype(
                jnp.uint32
            )
            next_bits = jnp.concatenate((bits[1:], jnp.zeros((1, 512), jnp.uint32)))
            low = jnp.where(even[:, None], bits, jnp.uint32(0))
            high = jnp.where(even[:, None], next_bits, bits)
            high = jnp.where(has_odd[:, None], high, jnp.uint32(0))
            words = jax.lax.bitcast_convert_type(low | (high << 16), jnp.int32)
            masks = jnp.where(even, jnp.int32(65535), jnp.int32(0)) | jnp.where(
                has_odd, jnp.int32(-65536), jnp.int32(0)
            )
            padding = -records.size % (lanes * cores)
            destinations = jnp.pad(
                jnp.where(boundary, pairs, -1), (0, padding), constant_values=-1
            )
            # SC cannot reshape a lane vector for a column broadcast in the kernel.
            masks = jnp.broadcast_to(
                jnp.pad(masks, (0, padding))[:, None], (records.size + padding, 512)
            )
            words = jnp.pad(words, ((0, padding), (0, 0)))
        # Keep the page-shaped owner: reshaping the whole pool can break aliasing.
        owner = jax.new_ref(bank)

        def kernel(dst_ref, mask_ref, words_ref, owner_ref, old_ref, sem):
            core = jax.lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))
            cache = owner_ref.bitcast(jnp.int32)
            old_ref[...] = jnp.zeros(old_ref.shape, jnp.int32)

            def body(destinations_ref, masks_ref, updates_ref):
                destinations = destinations_ref[...]
                masks = masks_ref[...]
                reads, writes = [], []
                for i in range(lanes):
                    pair = jnp.maximum(destinations[i], 0)
                    cache_row = cache.at[
                        pair // cache.shape[1], pl.ds(pair % cache.shape[1], 1), :
                    ]
                    partial = (destinations[i] >= 0) & (masks[i, 0] != -1)
                    read = pltpu.make_async_copy(
                        cache_row, old_ref.at[pl.ds(i, 1), :], sem.at[i]
                    )
                    reads.append((partial, read))
                    writes.append(
                        (
                            destinations[i] >= 0,
                            pltpu.make_async_copy(
                                old_ref.at[pl.ds(i, 1), :], cache_row, sem.at[i]
                            ),
                        )
                    )

                    @pl.when(partial)
                    def start_read(read=read):
                        read.start()

                for partial, read in reads:

                    @pl.when(partial)
                    def wait_read(read=read):
                        read.wait()

                old_ref[...] = (old_ref[...] & ~masks) | updates_ref[...]
                for valid, write in writes:

                    @pl.when(valid)
                    def start_write(write=write):
                        write.start()

                for valid, write in writes:

                    @pl.when(valid)
                    def wait_write(write=write):
                        write.wait()

            pltpu.emit_pipeline(
                body,
                grid=(destinations.shape[0] // (lanes * cores),),
                tiling=pltpu.Tiling.SPARSE_CORE,
                in_specs=(
                    pl.BlockSpec((lanes,), lambda b: (b * cores + core,)),
                    pl.BlockSpec((lanes, 512), lambda b: (b * cores + core, 0)),
                    pl.BlockSpec((lanes, 512), lambda b: (b * cores + core, 0)),
                ),
            )(dst_ref, mask_ref, words_ref)

        cast(
            Callable[..., tuple[()]],
            pl.kernel(
                kernel,
                mesh=mesh,
                out_type=(),
                scratch_types=(
                    pltpu.VMEM((lanes, 512), jnp.int32),
                    pltpu.SemaphoreType.DMA((lanes,)),
                ),
                compiler_params=pltpu.CompilerParams(
                    use_tc_tiling_on_sc=True,
                    needs_layout_passes=True,
                    disable_bounds_checks=True,
                ),
                name="write_swa_owned_pages",
            ),
        )(destinations, masks, words, owner)
        result.append(jax.freeze(owner))
        base += capacity
    return (result[0], result[1]) if isinstance(pages, tuple) else result[0]


_WINDOW_BLOCK = 128


@functools.partial(jax.jit, donate_argnums=(0,))
def write_swa_pages_with_window(
    pages: jax.Array,
    records: jax.Array,
    values: jax.Array,
    history_src: jax.Array,
    history_dst: jax.Array,
    commit_token: jax.Array,
    commit_dst: jax.Array,
) -> jax.Array:
    """Copy request-window history & commit rows on SparseCore, then write chunk kv.

    Replaces the TensorCore ``swa.reshape(-1, 512)[history_src]`` gather +
    ``torch.cat((kv, history, kv[commit_token]))`` + 17k-row sort/pack on every
    layer during extend. Each 128-row request block is owned by one SparseCore
    subcore so read-modify-write on packed BF16 pairs is race-free, and inactive
    padding requests (all -1 destinations) exit immediately with zero DMA.
    """
    m = history_dst.shape[0]
    page_size = pages.shape[1]
    if m == 0 or m % _WINDOW_BLOCK != 0 or page_size != 64:
        # Fallback for unusual test shapes.
        flat = pages.reshape(-1, 512)
        all_kv = jnp.concatenate((values, flat[history_src], values[commit_token]), axis=0)
        all_rec = jnp.concatenate((records, history_dst, commit_dst), axis=0)
        return write_swa_pages(pages, all_rec, all_kv)

    info = pltpu.get_tpu_info().sparse_core
    assert info is not None, "SWA window copy requires SparseCore"
    lanes, cores = info.num_lanes, info.num_cores * info.num_subcores
    num_chunks = _WINDOW_BLOCK // lanes
    mesh = plsc.VectorSubcoreMesh(
        num_cores=info.num_cores,
        num_subcores=info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    num_reqs = m // _WINDOW_BLOCK
    pad_reqs = (-num_reqs) % cores
    pad_elems = pad_reqs * _WINDOW_BLOCK
    if pad_elems:
        h_src = jnp.pad(history_src, (0, pad_elems), constant_values=0)
        h_dst = jnp.pad(history_dst, (0, pad_elems), constant_values=-1)
        c_tok = jnp.pad(commit_token, (0, pad_elems), constant_values=0)
        c_dst = jnp.pad(commit_dst, (0, pad_elems), constant_values=-1)
    else:
        h_src, h_dst, c_tok, c_dst = history_src, history_dst, commit_token, commit_dst

    # Promote chunk kv BF16 bits to int32 [T, 512] so any single row can be DMA'd.
    kv_words = jax.lax.bitcast_convert_type(values, jnp.uint16).astype(jnp.int32)
    owner = jax.new_ref(pages)
    capacity = pages.shape[0] * page_size

    def kernel(
        h_src_ref, h_dst_ref, c_tok_ref, c_dst_ref, kv_ref, owner_ref, buf_src, buf_dst, sem
    ):
        core = jax.lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))
        cache = owner_ref.bitcast(jnp.int32)  # [pages, 32, 512]
        half_page = cache.shape[1]

        def copy_one_req(hs_ref, hd_ref, ct_ref, cd_ref):
            def h_chunk(c, _):
                hd = hd_ref[pl.ds(c * lanes, lanes)]
                hs = hs_ref[pl.ds(c * lanes, lanes)]
                for j in range(lanes):
                    dst = hd[j]
                    valid = (dst >= page_size) & (dst < capacity)

                    @pl.when(valid)
                    def _move(dst=dst, src=hs[j]):
                        s_pair = src // 2
                        d_pair = dst // 2
                        s_odd = (src & 1) != 0
                        d_odd = (dst & 1) != 0
                        s_slice = cache.at[
                            s_pair // half_page, pl.ds(s_pair % half_page, 1), :
                        ]
                        d_slice = cache.at[
                            d_pair // half_page, pl.ds(d_pair % half_page, 1), :
                        ]
                        r_src = pltpu.make_async_copy(s_slice, buf_src, sem.at[0])
                        r_dst = pltpu.make_async_copy(d_slice, buf_dst, sem.at[1])
                        r_src.start()
                        r_dst.start()
                        r_src.wait()
                        r_dst.wait()
                        s_word = buf_src[0]
                        d_word = buf_dst[0]
                        val16 = jnp.where(
                            s_odd,
                            jax.lax.shift_right_logical(s_word, jnp.int32(16)),
                            s_word & jnp.int32(0xFFFF),
                        )
                        merged = jnp.where(
                            d_odd,
                            (d_word & jnp.int32(0xFFFF)) | (val16 << jnp.int32(16)),
                            (d_word & jnp.int32(-65536)) | val16,
                        )
                        buf_dst[0] = merged
                        w_dst = pltpu.make_async_copy(buf_dst, d_slice, sem.at[0])
                        w_dst.start()
                        w_dst.wait()

                return None

            jax.lax.fori_loop(0, num_chunks, h_chunk, None)

            def c_chunk(c, _):
                cd = cd_ref[pl.ds(c * lanes, lanes)]
                ct = ct_ref[pl.ds(c * lanes, lanes)]
                for j in range(lanes):
                    dst = cd[j]
                    valid = (dst >= page_size) & (dst < capacity)

                    @pl.when(valid)
                    def _move(dst=dst, tok=ct[j]):
                        d_pair = dst // 2
                        d_odd = (dst & 1) != 0
                        k_slice = kv_ref.at[pl.ds(tok, 1), :]
                        d_slice = cache.at[
                            d_pair // half_page, pl.ds(d_pair % half_page, 1), :
                        ]
                        r_src = pltpu.make_async_copy(k_slice, buf_src, sem.at[0])
                        r_dst = pltpu.make_async_copy(d_slice, buf_dst, sem.at[1])
                        r_src.start()
                        r_dst.start()
                        r_src.wait()
                        r_dst.wait()
                        val16 = buf_src[0] & jnp.int32(0xFFFF)
                        d_word = buf_dst[0]
                        merged = jnp.where(
                            d_odd,
                            (d_word & jnp.int32(0xFFFF)) | (val16 << jnp.int32(16)),
                            (d_word & jnp.int32(-65536)) | val16,
                        )
                        buf_dst[0] = merged
                        w_dst = pltpu.make_async_copy(buf_dst, d_slice, sem.at[0])
                        w_dst.start()
                        w_dst.wait()

                return None

            jax.lax.fori_loop(0, num_chunks, c_chunk, None)

        pltpu.emit_pipeline(
            copy_one_req,
            grid=((num_reqs + pad_reqs) // cores,),
            tiling=pltpu.Tiling.SPARSE_CORE,
            in_specs=(
                pl.BlockSpec((_WINDOW_BLOCK,), lambda b: (b * cores + core,)),
                pl.BlockSpec((_WINDOW_BLOCK,), lambda b: (b * cores + core,)),
                pl.BlockSpec((_WINDOW_BLOCK,), lambda b: (b * cores + core,)),
                pl.BlockSpec((_WINDOW_BLOCK,), lambda b: (b * cores + core,)),
            ),
        )(h_src_ref, h_dst_ref, c_tok_ref, c_dst_ref)

    cast(
        Callable[..., tuple[()]],
        pl.kernel(
            kernel,
            mesh=mesh,
            out_type=(),
            scratch_types=(
                pltpu.VMEM((1, 512), jnp.int32),
                pltpu.VMEM((1, 512), jnp.int32),
                pltpu.SemaphoreType.DMA((2,)),
            ),
            compiler_params=pltpu.CompilerParams(
                use_tc_tiling_on_sc=True,
                needs_layout_passes=True,
                disable_bounds_checks=True,
            ),
            name="copy_swa_window_rows",
        ),
    )(h_src, h_dst, c_tok, c_dst, kv_words, owner)
    updated = jax.freeze(owner)
    return write_swa_pages(updated, records, values)
