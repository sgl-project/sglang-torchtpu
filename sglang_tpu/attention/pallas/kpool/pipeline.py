# SPDX-License-Identifier: Apache-2.0
"""Compile-time schedules. Numerical stages and DMA views live separately.

The short schedule drains each group before the next group's aliased reads.
The long schedule carries raw prefetches across tiles, with one ordered store
stream. Both preserve a partial destination quad through read/modify/write.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from .compute import pack_pools, pool_rows
from .io import LongQueryIO, ShortQueryIO


@dataclass(frozen=True)
class ShortQueryPipeline:
    """Batch independent tiles; serialize a group with shared destinations."""

    group_size: int

    def emit(self, io: ShortQueryIO, ape_ref, mask_ref):
        group_size = self.group_size
        desc_ref, raw_buf, out_buf = io.desc, io.raw_buffer, io.output_buffer

        @pl.loop(0, desc_ref.shape[1] // group_size)
        def _group(group):
            begin = group * group_size

            @pl.when(desc_ref[7, begin] != 0)
            def _shared_quad():
                @pl.loop(0, group_size)
                def _ordered(slot):
                    i = begin + slot

                    @pl.when(desc_ref[6, i] != 0)
                    def _live():
                        io.read(i, slot).start()
                        io.read_old(i, slot).start()
                        io.read(i, slot).wait()
                        packed = pack_pools(pool_rows(raw_buf[slot], ape_ref[...]))
                        mask = mask_ref[pl.ds(i, 1), :]
                        io.read_old(i, slot).wait()
                        out_buf[slot] = (out_buf[slot] & ~mask) | (packed & mask)
                        io.write(i, slot).start()
                        io.write(i, slot).wait()

            @pl.when(desc_ref[7, begin] == 0)
            def _independent():
                @pl.loop(0, group_size)
                def _start(slot):
                    i = begin + slot

                    @pl.when(desc_ref[6, i] != 0)
                    def _live():
                        io.read(i, slot).start()
                        io.read_old(i, slot).start()

                @pl.loop(0, group_size)
                def _wait_raw(slot):
                    i = begin + slot

                    @pl.when(desc_ref[6, i] != 0)
                    def _live():
                        io.read(i, slot).wait()

                # All four pools of each aligned tile stay adjacent through packing;
                # no Q=1 extraction or packing across unrelated requests is needed.
                raw = raw_buf[...].reshape((group_size * 16, 256))
                packed = pack_pools(pool_rows(raw, ape_ref[...]))
                mask = mask_ref[pl.ds(begin, group_size), :]

                @pl.loop(0, group_size)
                def _wait_old(slot):
                    i = begin + slot

                    @pl.when(desc_ref[6, i] != 0)
                    def _live():
                        io.read_old(i, slot).wait()

                old = out_buf[...].reshape((group_size, 256))
                out_buf[...] = ((old & ~mask) | (packed & mask)).reshape(
                    (group_size, 1, 256)
                )

                @pl.loop(0, group_size)
                def _start_write(slot):
                    i = begin + slot

                    @pl.when(desc_ref[6, i] != 0)
                    def _live():
                        io.write(i, slot).start()

                @pl.loop(0, group_size)
                def _wait_write(slot):
                    i = begin + slot

                    @pl.when(desc_ref[6, i] != 0)
                    def _live():
                        io.write(i, slot).wait()


@dataclass(frozen=True)
class LongQueryPipeline:
    """Two-slot raw prefetch, bounded compute, ordered cache writes, drain."""

    tile_rows: int

    def emit(self, io: LongQueryIO, ape_ref, pooled_buf, count):
        tile_rows = self.tile_rows
        word_rows = tile_rows // 16
        desc_ref, raw_buf, out_buf = io.desc, io.raw_buffer, io.output_buffer

        for initial in range(2):

            @pl.when(count > initial)
            def _prime(initial=initial):
                io.read(initial).start()

        def _page(i, slot):
            io.read(i).wait()

            if tile_rows <= 32:
                pooled = pool_rows(raw_buf[slot], ape_ref[...])
            else:
                # Retain the larger DMA while bounding pooling's register live
                # set. The VMEM staging also bounds values carried across loops.
                @pl.loop(0, tile_rows // 32)
                def _pool_chunk(chunk):
                    value = pool_rows(
                        raw_buf[slot, pl.ds(chunk * 32, 32), :], ape_ref[...]
                    )
                    # BF16 staging tiles require 16-row alignment in this runtime;
                    # FP32 permits an 8-row store. Preserve the BF16 rounding above.
                    pooled_buf[pl.ds(chunk * 8, 8), :] = value.astype(jnp.float32)

                pooled = pooled_buf[...]

            # Every raw Ref load precedes this refill. Subsequent stages only use
            # SSA values; no consumer rereads the overwritten VMEM generation.
            @pl.when(i + 2 < count)
            def _refill():
                io.read(i + 2).start()

            packed = pack_pools(pooled)
            mask = jnp.zeros((word_rows, 256), jnp.uint32)
            for lane in range(4):
                ends = (
                    jax.lax.broadcasted_iota(jnp.int32, (word_rows, 256), 0) * 16
                    + lane * 4
                    + 3
                )
                live = (ends >= desc_ref[3, i]) & (ends < desc_ref[4, i])
                mask = mask | (
                    jnp.where(live, jnp.uint32(255), jnp.uint32(0)) << (lane * 8)
                )

            # Serialize the *read* too. A later tile may update different bytes
            # in exactly the same physical destination block as an earlier tile.
            @pl.when(i > 0)
            def _prior_store():
                io.write(i - 1).wait()

            @pl.when((desc_ref[3, i] > 3) | (desc_ref[4, i] < tile_rows))
            def _preserve():
                old = io.read_old(i, slot)
                old.start()
                old.wait()
                out_buf[slot, :, :] = (out_buf[slot, :, :] & ~mask) | (packed & mask)

            @pl.when((desc_ref[3, i] <= 3) & (desc_ref[4, i] >= tile_rows))
            def _overwrite():
                out_buf[slot, :, :] = packed

            io.write(i).start()

        # Keep the two-slot period explicit; the number of pages stays dynamic.
        @pl.loop(0, pl.cdiv(count, 2))
        def _pair(pair):
            for slot in range(2):

                @pl.when(pair * 2 + slot < count)
                def _active(slot=slot):
                    _page(pair * 2 + slot, slot)

        @pl.when(count > 0)
        def _drain():
            io.write(count - 1).wait()
