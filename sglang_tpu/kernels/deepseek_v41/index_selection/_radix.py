# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: src/vllm_torchtpu/kernels/deepseek_v41/index_selection/publication.py
# Revision: dc1cd126e71c2248f3fbf2bef0d8d24e1e3768d7 (epic/dsv41)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Google LLC
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Private VMEM radix primitive shared by block-maximum selection stages."""

import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu_sc as plsc

from sglang_tpu.kernels.deepseek_v41.index_selection.streaming_topk import (
    FP32_BITS,
    ORDERED_NEG_INF,
    RADIX_BITS,
    RADIX_BUCKETS,
    RADIX_MASK,
)


class Radix:
    """Radix selection over ordered FP32 key bits staged in VMEM.

    Byte histograms refine the quota-th largest key, then one pass emits the
    winners in candidate order, so IDs come out ascending without a sort.
    """

    def __init__(self, histogram, lanes):
        self.histogram, self.lanes = histogram, lanes
        self.lane_ids = jnp.arange(lanes, dtype=jnp.int32)

    def positions(self, i):
        return i * self.lanes + self.lane_ids

    def clear(self):
        @plsc.parallel_loop(0, RADIX_BUCKETS, self.lanes, unroll=4)
        def clear(i):
            self.histogram[pl.ds(i, self.lanes)] = jnp.zeros(self.lanes, jnp.int32)

    def add(self, bits):
        """Count the top byte of the keys above -inf."""
        digits = (bits >> (FP32_BITS - RADIX_BITS)).astype(jnp.int32)
        counts, last = plsc.scan_count(digits, mask=bits > ORDERED_NEG_INF)
        plsc.addupdate_scatter(self.histogram, [digits], counts, mask=last)

    def _bucket(self, bucket):
        index = jnp.full(self.lanes, bucket, jnp.int32)
        return plsc.load_gather(self.histogram, [index])[0]

    def pivot(self, quota):
        """The bucket holding the quota-th largest counted key."""
        histogram, lanes = self.histogram, self.lanes
        zero = jnp.asarray(0, jnp.int32)

        @plsc.parallel_loop(0, RADIX_BUCKETS, lanes, carry=zero, unroll=2)
        def total(i, running):
            cumulative = plsc.cumsum(histogram[pl.ds(i, lanes)]) + running
            histogram[pl.ds(i, lanes)] = cumulative
            return cumulative[-1]

        @plsc.parallel_loop(0, RADIX_BUCKETS, lanes, carry=zero, unroll=2)
        def find(i, bucket):
            below = histogram[pl.ds(i, lanes)] <= total - quota
            return bucket + plsc.all_reduce_population_count(below)[0]

        bucket = jnp.minimum(find, RADIX_MASK)
        cumulative = self._bucket(bucket)
        previous = self._bucket(jnp.maximum(bucket - 1, 0))
        count = cumulative - jnp.where(bucket > 0, previous, 0)
        return bucket, quota - (total - cumulative), total, count

    def threshold(self, values, chunks, quota):
        """The quota-th largest key; the top byte's histogram is filled."""
        lanes = self.lanes
        prefix, remaining, total, count = self.pivot(jnp.int32(quota))
        keep_all = total <= quota
        done = keep_all | (remaining == count)
        remaining_bits = jnp.int32(FP32_BITS - RADIX_BITS)
        for shift in reversed(range(0, FP32_BITS - RADIX_BITS, RADIX_BITS)):
            self.clear()

            @plsc.parallel_loop(0, jnp.where(done, 0, chunks), unroll=8)
            def refine(i, shift=shift, prefix=prefix):
                bits = values[pl.ds(i * lanes, lanes)].view(jnp.uint32)
                match = (bits >> (shift + RADIX_BITS)) == prefix.astype(jnp.uint32)
                digits = ((bits >> shift) & RADIX_MASK).astype(jnp.int32)
                counts, last = plsc.scan_count(digits, mask=match)
                plsc.addupdate_scatter(self.histogram, [digits], counts, mask=last)

            bucket, next_remaining, _, count = self.pivot(remaining)
            prefix = jnp.where(done, prefix, (prefix << RADIX_BITS) | bucket)
            remaining = jnp.where(done, remaining, next_remaining)
            remaining_bits = jnp.where(done, remaining_bits, shift)
            done |= remaining == count
        lower = prefix.astype(jnp.uint32) << remaining_bits.astype(jnp.uint32)
        gate = jnp.where(keep_all, jnp.uint32(ORDERED_NEG_INF + 1), lower)
        return total, lower, gate, remaining, done

    def emit(self, values, chunks, quota, selection, destination, ids=None):
        """Write `ids(i)` (default: positions) of the selected keys, -1 after."""
        lanes, ids = self.lanes, ids or self.positions
        total, lower, gate, remaining, done = selection

        @plsc.parallel_loop(
            jnp.minimum(total // lanes * lanes, quota), quota, lanes, unroll=4
        )
        def initialize(i):
            destination[pl.ds(i, lanes)] = jnp.full(lanes, -1, jnp.int32)

        def store(i, count, selected):
            ranks = plsc.cumsum(selected.astype(jnp.int32))
            plsc.store_scatter(destination, [count + ranks - 1], ids(i), mask=selected)
            return count + plsc.all_reduce_population_count(selected)

        @plsc.parallel_loop(
            0, jnp.where(done, chunks, 0), carry=jnp.zeros(lanes, jnp.int32), unroll=8
        )
        def emit_whole(i, count):
            bits = values[pl.ds(i * lanes, lanes)].view(jnp.uint32)
            return store(
                i, count, (bits >= gate) & (bits > jnp.uint32(ORDERED_NEG_INF))
            )

        @plsc.parallel_loop(
            0,
            jnp.where(done, 0, chunks),
            carry=(jnp.zeros(lanes, jnp.int32), jnp.zeros(lanes, jnp.int32)),
            unroll=4,
        )
        def emit_ranked(i, state):
            count, ties = state
            bits = values[pl.ds(i * lanes, lanes)].view(jnp.uint32)
            equal = bits == lower
            ranks = plsc.cumsum(equal.astype(jnp.int32)) + ties
            selected = (bits > lower) | (equal & (ranks <= remaining))
            return store(i, count, selected), ties + plsc.all_reduce_population_count(
                equal
            )
