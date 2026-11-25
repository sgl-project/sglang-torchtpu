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


def _pair_window_ops(
    src: jax.Array,
    dst: jax.Array,
    *,
    page_size: int,
    capacity: int,
    pad_to: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Pair adjacent even/odd rows within each 128-row request block.

    In WindowPlan, consecutive token positions map to consecutive even/odd SWA
    rows within each request, so pairing (k, k+1) makes every valid destination
    word-pair unique across the whole batch (race-free across all SC subcores)
    and eliminates read-modify-write for ~99% of history/commit rows.

    Returns (d_pair, s_lo, s_hi, flags) padded to a multiple of ``pad_to``:
      flags bit 0 (1): write low BF16 half from s_lo
      flags bit 1 (2): write high BF16 half from s_hi
      d_pair < 0: inactive slot (zero DMA).
    """
    num_reqs = dst.shape[0] // _WINDOW_BLOCK
    s2 = src.reshape(num_reqs, _WINDOW_BLOCK)
    d2 = dst.reshape(num_reqs, _WINDOW_BLOCK)
    valid = (d2 >= page_size) & (d2 < capacity)
    is_even = (d2 & 1) == 0

    # Check if next lane in the same request is the matching odd partner (d + 1).
    next_d = jnp.concatenate(
        (d2[:, 1:], jnp.full((num_reqs, 1), -1, dtype=d2.dtype)), axis=1
    )
    next_s = jnp.concatenate(
        (s2[:, 1:], jnp.zeros((num_reqs, 1), dtype=s2.dtype)), axis=1
    )
    next_valid = jnp.concatenate(
        (valid[:, 1:], jnp.zeros((num_reqs, 1), dtype=jnp.bool_)), axis=1
    )
    has_next_partner = valid & is_even & next_valid & (next_d == d2 + 1)

    # Check if previous lane in the same request already claimed this odd row.
    claimed_by_prev = jnp.concatenate(
        (jnp.zeros((num_reqs, 1), dtype=jnp.bool_), has_next_partner[:, :-1]), axis=1
    )

    active = valid & ~claimed_by_prev
    has_lo = active & is_even
    has_hi = (active & ~is_even) | has_next_partner
    d_pair = jnp.where(active, d2 // 2, jnp.int32(-1)).reshape(-1)
    s_lo = jnp.where(has_lo, s2, jnp.int32(0)).reshape(-1)
    s_hi = jnp.where(
        has_next_partner, next_s, jnp.where(active & ~is_even, s2, jnp.int32(0))
    ).reshape(-1)
    flags = (
        jnp.where(has_lo, jnp.int32(1), jnp.int32(0))
        | jnp.where(has_hi, jnp.int32(2), jnp.int32(0))
    ).reshape(-1)

    padding = (-d_pair.shape[0]) % pad_to
    if padding:
        d_pair = jnp.pad(d_pair, (0, padding), constant_values=-1)
        s_lo = jnp.pad(s_lo, (0, padding), constant_values=0)
        s_hi = jnp.pad(s_hi, (0, padding), constant_values=0)
        flags = jnp.pad(flags, (0, padding), constant_values=0)
    return d_pair, s_lo, s_hi, flags


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
    layer. Pairs adjacent even/odd rows per request so all 32 SparseCore
    subcores copy 16 lanes in parallel with direct full-word DMAs (no RMW for
    full pairs) and zero DMA for inactive/padding rows.
    """
    m = history_dst.shape[0]
    page_size = pages.shape[1]
    # For tiny batches (m < 1024, e.g. B < 8), the single gather+write_swa_pages
    # launch beats launching 4 separate SparseCore kernels.
    if m < 1024 or m % _WINDOW_BLOCK != 0 or page_size != 64:
        flat = pages.reshape(-1, 512)
        all_kv = jnp.concatenate(
            (values, flat[history_src], values[commit_token]), axis=0
        )
        all_rec = jnp.concatenate((records, history_dst, commit_dst), axis=0)
        return write_swa_pages(pages, all_rec, all_kv)

    info = pltpu.get_tpu_info().sparse_core
    assert info is not None, "SWA window copy requires SparseCore"
    lanes, cores = info.num_lanes, info.num_cores * info.num_subcores
    pad_to = lanes * cores
    mesh = plsc.VectorSubcoreMesh(
        num_cores=info.num_cores,
        num_subcores=info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    capacity = pages.shape[0] * page_size

    h_dpair, h_slo, h_shi, h_flags = _pair_window_ops(
        history_src, history_dst, page_size=page_size, capacity=capacity, pad_to=pad_to
    )

    # Split history ops into:
    # (1) fast aligned full pairs (lo+hi both valid, s_lo even, s_hi == s_lo + 1):
    #     direct 32-bit word copy, zero RMW, ~99% of all history rows.
    # (2) remaining partial/unaligned history entries (at most 2 boundary rows/req):
    #     at most 1-4 rows per SC subcore, tiny 4 KB VMEM footprint.
    h_fast_full = (
        (h_dpair >= 0)
        & (h_flags == 3)
        & ((h_slo & 1) == 0)
        & (h_shi == h_slo + 1)
    )
    hf_dpair = jnp.where(h_fast_full, h_dpair, jnp.int32(-1))
    hf_spair = jnp.where(h_fast_full, h_slo // 2, jnp.int32(0))

    h_rem = (h_dpair >= 0) & ~h_fast_full
    hr_dpair = jnp.where(h_rem, h_dpair, jnp.int32(-1))
    hr_slo = jnp.where(h_rem & ((h_flags & 1) != 0), h_slo, jnp.int32(-1))
    hr_shi = jnp.where(h_rem & ((h_flags & 2) != 0), h_shi, jnp.int32(-1))

    # Each valid commit_dst maps a unique token t in [0, T) to its ring slot.
    # Scatter into [T] so commit writes reuse `values` directly (zero gather,
    # and hits `swa_pack` decode fusion at T=96/192/288).
    valid_commit = (commit_dst >= page_size) & (commit_dst < capacity)
    token_commit_dst = jnp.full((values.shape[0],), -1, dtype=jnp.int32).at[
        commit_token
    ].max(jnp.where(valid_commit, commit_dst, jnp.int32(-1)))

    owner = jax.new_ref(pages)

    # Pass 1: fast full-pair history copy (1 VMEM buffer, lanes=16 semaphores, zero RMW).
    def history_fast_kernel(dp_ref, sp_ref, owner_ref, buf, sem):
        core = jax.lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))
        cache = owner_ref.bitcast(jnp.int32)  # [pages, 32, 512]
        half_page = cache.shape[1]

        def body(dp_r, sp_r):
            dp = dp_r[...]
            sp = sp_r[...]
            reads, writes = [], []
            for i in range(lanes):
                active = dp[i] >= 0
                s_pair = jnp.maximum(sp[i], 0)
                d_pair = jnp.maximum(dp[i], 0)
                s_slice = cache.at[s_pair // half_page, pl.ds(s_pair % half_page, 1), :]
                d_slice = cache.at[d_pair // half_page, pl.ds(d_pair % half_page, 1), :]
                r = pltpu.make_async_copy(s_slice, buf.at[pl.ds(i, 1), :], sem.at[i])
                w = pltpu.make_async_copy(buf.at[pl.ds(i, 1), :], d_slice, sem.at[i])
                reads.append((active, r))
                writes.append((active, w))

                @pl.when(active)
                def _start_r(r=r):
                    r.start()

            for active, r in reads:
                @pl.when(active)
                def _wait_r(r=r):
                    r.wait()

            for active, w in writes:
                @pl.when(active)
                def _start_w(w=w):
                    w.start()

            for active, w in writes:
                @pl.when(active)
                def _wait_w(w=w):
                    w.wait()

        pltpu.emit_pipeline(
            body,
            grid=(hf_dpair.shape[0] // pad_to,),
            tiling=pltpu.Tiling.SPARSE_CORE,
            in_specs=(
                pl.BlockSpec((lanes,), lambda b: (b * cores + core,)),
                pl.BlockSpec((lanes,), lambda b: (b * cores + core,)),
            ),
        )(dp_ref, sp_ref)

    cast(
        Callable[..., tuple[()]],
        pl.kernel(
            history_fast_kernel,
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
            name="copy_swa_window_history_fast",
        ),
    )(hf_dpair, hf_spair, owner)

    # Pass 2: boundary / partial history rows (<= 2 per request, 4 KB VMEM).
    def history_rem_kernel(dp_ref, slo_ref, shi_ref, owner_ref, b_src, b_dst, sem):
        core = jax.lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))
        cache = owner_ref.bitcast(jnp.int32)
        half_page = cache.shape[1]

        def body(dp_r, slo_r, shi_r):
            dp = dp_r[...]
            slo = slo_r[...]
            shi = shi_r[...]
            for i in range(lanes):
                d_pair = dp[i]

                @pl.when(d_pair >= 0)
                def _move(d_pair=d_pair, s_lo=slo[i], s_hi=shi[i]):
                    d_slice = cache.at[
                        d_pair // half_page, pl.ds(d_pair % half_page, 1), :
                    ]
                    rd = pltpu.make_async_copy(d_slice, b_dst, sem.at[0])
                    rd.start()
                    rd.wait()

                    @pl.when(s_lo >= 0)
                    def _do_lo():
                        sp = s_lo // 2
                        s_slice = cache.at[sp // half_page, pl.ds(sp % half_page, 1), :]
                        rs = pltpu.make_async_copy(s_slice, b_src, sem.at[0])
                        rs.start()
                        rs.wait()
                        sw = b_src[0]
                        val16 = jnp.where(
                            (s_lo & 1) != 0,
                            jax.lax.shift_right_logical(sw, jnp.int32(16)),
                            sw & jnp.int32(0xFFFF),
                        )
                        b_dst[0] = (b_dst[0] & jnp.int32(-65536)) | val16

                    @pl.when(s_hi >= 0)
                    def _do_hi():
                        sp = s_hi // 2
                        s_slice = cache.at[sp // half_page, pl.ds(sp % half_page, 1), :]
                        rs = pltpu.make_async_copy(s_slice, b_src, sem.at[0])
                        rs.start()
                        rs.wait()
                        sw = b_src[0]
                        val16 = jnp.where(
                            (s_hi & 1) != 0,
                            jax.lax.shift_right_logical(sw, jnp.int32(16)),
                            sw & jnp.int32(0xFFFF),
                        )
                        b_dst[0] = (b_dst[0] & jnp.int32(0xFFFF)) | (
                            val16 << jnp.int32(16)
                        )

                    wd = pltpu.make_async_copy(b_dst, d_slice, sem.at[0])
                    wd.start()
                    wd.wait()

        pltpu.emit_pipeline(
            body,
            grid=(hr_dpair.shape[0] // pad_to,),
            tiling=pltpu.Tiling.SPARSE_CORE,
            in_specs=(
                pl.BlockSpec((lanes,), lambda b: (b * cores + core,)),
                pl.BlockSpec((lanes,), lambda b: (b * cores + core,)),
                pl.BlockSpec((lanes,), lambda b: (b * cores + core,)),
            ),
        )(dp_ref, slo_ref, shi_ref)

    cast(
        Callable[..., tuple[()]],
        pl.kernel(
            history_rem_kernel,
            mesh=mesh,
            out_type=(),
            scratch_types=(
                pltpu.VMEM((1, 512), jnp.int32),
                pltpu.VMEM((1, 512), jnp.int32),
                pltpu.SemaphoreType.DMA((1,)),
            ),
            compiler_params=pltpu.CompilerParams(
                use_tc_tiling_on_sc=True,
                needs_layout_passes=True,
                disable_bounds_checks=True,
            ),
            name="copy_swa_window_history_rem",
        ),
    )(hr_dpair, hr_slo, hr_shi, owner)

    updated = jax.freeze(owner)
    updated = write_swa_pages(updated, token_commit_dst, values)
    return write_swa_pages(updated, records, values)
