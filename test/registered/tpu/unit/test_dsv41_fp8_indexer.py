# SPDX-License-Identifier: Apache-2.0
"""FP8-only index-cache/scorer regressions; no model checkpoint is required."""

import os
import sys

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=600,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "deepseek-v4.1-flash", "accuracy", "compile"],
)

os.environ.setdefault("JAX_PLATFORMS", "tpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sglang_tpu.kernels.index_selection.fp8_paged_mqa_logits import fp8_paged_mqa_logits
from sglang_tpu.kernels.index_selection.packed_fp8_cache import (
    PAGE_RECORDS,
    PAGE_ROWS,
    PackedFP8Layout,
    quantize_index_rows,
    write_fp8_records,
)


def read_fp8_records(
    banks: tuple[jax.Array, ...], records: jax.Array, layout: PackedFP8Layout
) -> jax.Array:
    """Test reference decoder for packed FP8 records; invalid slots are zero."""
    layout.validate(tuple(bank.shape[0] for bank in banks))
    shape = records.shape
    flat = records.reshape(-1)
    valid = (flat >= 0) & (flat < layout.pages * layout.page_size)
    safe = jnp.clip(flat, 0, layout.pages * layout.page_size - 1)
    base = layout.page_offset + (safe // layout.page_size) * layout.page_stride
    columns = safe % layout.page_size
    addresses = (
        base[:, None] * PAGE_RECORDS
        + jnp.arange(PAGE_ROWS)[None, :] * layout.page_size
        + columns[:, None]
    )
    words = jnp.zeros(addresses.shape, jnp.int32)
    offset = 0
    for bank in banks:
        local = addresses // PAGE_RECORDS - offset
        active = valid[:, None] & (local >= 0) & (local < bank.shape[0])
        gathered = bank[jnp.clip(local, 0, bank.shape[0] - 1), addresses % PAGE_RECORDS]
        words = jnp.where(active, gathered, words)
        offset += bank.shape[0]
    bits = (words[:, :32, None].astype(jnp.uint32) >> (jnp.arange(4) * 8)) & 255
    values = jax.lax.bitcast_convert_type(
        bits.astype(jnp.uint8).reshape(-1, PAGE_RECORDS), jnp.float8_e4m3fn
    )
    scales = jax.lax.bitcast_convert_type(words[:, 32], jnp.float32)
    result = (values.astype(jnp.float32) * scales[:, None]).astype(jnp.bfloat16)
    return jnp.where(valid[:, None], result, 0).reshape(*shape, PAGE_RECORDS)


def _fixture(tokens, groups, width, page_size):
    rng = np.random.default_rng(20261026)
    page_count = (width + page_size - 1) // page_size
    layout = PackedFP8Layout(page_count * groups, page_size=page_size)
    keys = jnp.asarray(rng.normal(size=(layout.pages * page_size, 128)), jnp.bfloat16)
    q = jnp.asarray(rng.normal(size=(tokens, 32, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.normal(size=(tokens, 32)), jnp.float32)
    table = jnp.asarray(
        rng.permutation(layout.pages).reshape(groups, page_count), jnp.int32
    )
    # Repeated offsets exercise empty requests; nonuniform counts exercise tails.
    offsets = np.sort(rng.integers(0, tokens + 1, size=groups - 1))
    offsets = np.concatenate(([0], offsets, [tokens])).astype(np.int32)
    lengths = rng.integers(0, width + 1, size=tokens, dtype=np.int32)
    if tokens:
        lengths[0] = 0
        lengths[-1] = width
    return layout, keys, q, weights, table, jnp.asarray(offsets), jnp.asarray(lengths)


@pytest.mark.parametrize("page_size", [64, 128, 256])
def test_fp8_cache_write_preserves_untouched_records(page_size):
    layout = PackedFP8Layout(4, page_size=page_size)
    banks = (jnp.zeros((layout.pages * layout.page_rows, 128), jnp.int32),)
    rng = np.random.default_rng(17)
    values = jnp.asarray(rng.normal(size=(page_size * 2 + 3, 128)), jnp.bfloat16)
    records = jnp.arange(values.shape[0], dtype=jnp.int32)
    write = jax.jit(lambda b, r, v: write_fp8_records(b, r, v, layout))
    banks = write(banks, records, values)
    quant, scales, _ = quantize_index_rows(values)
    expected = (quant.astype(jnp.float32) * scales[:, None]).astype(jnp.bfloat16)
    np.testing.assert_array_equal(read_fp8_records(banks, records, layout), expected)
    # Invalid writes must not corrupt the first/last physical page; shared pages
    # span the writer's 128-input batch boundary.
    updated = write(
        banks, jnp.array([-1, layout.pages * page_size], jnp.int32), values[:2]
    )
    np.testing.assert_array_equal(updated[0], banks[0])
    replacements = jnp.array([1, page_size - 1, page_size + 1], jnp.int32)
    banks = write(banks, replacements, values[:3])
    expected = expected.at[replacements].set(expected[:3])
    np.testing.assert_array_equal(read_fp8_records(banks, records, layout), expected)


# Actual token buckets through the configured 8192 prefill chunk, plus irregular
# tails, mixed/empty requests, decode concurrency, and a long-context key tail.
@pytest.mark.parametrize(
    "tokens,groups,width,page_size",
    [
        (1, 1, 4096, 128),
        (7, 4, 4096, 64),
        (48, 48, 4096, 256),
        (128, 8, 4096, 128),
        (256, 16, 4096, 128),
        (512, 32, 4096, 128),
        (1024, 8, 4096, 256),
        (2048, 8, 4096, 128),
        (4096, 8, 4096, 128),
        (8192, 8, 4096, 128),
        (3, 2, 65537, 128),
    ],
)
def test_fp8_scorer_shapes(tokens, groups, width, page_size, record_property):
    layout, keys, q, weights, table, offsets, lengths = _fixture(
        tokens, groups, width, page_size
    )
    banks = (jnp.zeros((layout.pages * layout.page_rows, 128), jnp.int32),)
    # Populate history in actual prefill-sized writes. A serving step writes
    # at most its token bucket, never every request's entire historical cache.
    write = jax.jit(lambda b, r, k: write_fp8_records(b, r, k, layout))
    for start in range(0, keys.shape[0], 8192):
        stop = min(start + 8192, keys.shape[0])
        banks = write(banks, jnp.arange(start, stop, dtype=jnp.int32), keys[start:stop])
    q8, qs, _ = quantize_index_rows(q)
    actual = fp8_paged_mqa_logits(
        q8,
        banks[0],
        weights,
        lengths,
        table,
        query_start_loc=offsets,
        max_model_len=width,
        q_scale=qs,
        query_size=6,
        cache_layout=layout,
    )
    actual.block_until_ready()
    assert actual.shape == (tokens, width)
    # Check every visibility mask, but compute numerical references for a bounded
    # sample of rows so the 8192-row test does not build a second huge score graph.
    actual_host = np.asarray(actual)
    mask = np.arange(width)[None, :] < np.asarray(lengths)[:, None]
    np.testing.assert_array_equal(np.isneginf(actual_host), ~mask)
    sampled = np.unique(np.linspace(0, tokens - 1, min(tokens, 12), dtype=int))
    errors = []
    for row in sampled:
        visible = int(lengths[row])
        if not visible:
            continue
        request = int(np.searchsorted(np.asarray(offsets)[1:], row, side="right"))
        slots = (
            table[request, jnp.arange(visible) // page_size] * page_size
            + jnp.arange(visible) % page_size
        )
        decoded = read_fp8_records(banks, slots, layout).astype(jnp.float32)
        query = q8[row].astype(jnp.float32) * qs[row, :, None]
        reference = (jnp.maximum(query @ decoded.T, 0) * weights[row, :, None]).sum(0)
        ref = np.asarray(reference).astype(np.float64)
        got = actual_host[row, :visible].astype(np.float64)
        diff = np.sum((got - ref) ** 2) / max(np.sum(got**2 + ref**2), 1e-20)
        errors.append(diff)
        assert diff < 1e-3, (row, diff)
    record_property("max_calc_diff", max(errors, default=0))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
