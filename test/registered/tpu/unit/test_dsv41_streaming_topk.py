# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: tests/kernels/test_dsv41_streaming_topk.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-row prefill streaming top-k: stable ties and finite padding contract."""

import math
import os
import sys

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=30,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "deepseek-v4.1-flash", "accuracy", "compile"],
)

os.environ.setdefault("JAX_PLATFORMS", "tpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sglang_tpu.kernels.deepseek_v41.index_selection.streaming_topk import (
    _block_visit_order,
    streaming_topk,
)


@pytest.mark.parametrize("num_blocks", [1, 2, 3, 4, 8, 128, 129, 1024, 65536])
def test_block_visit_order_is_bijection(num_blocks):
    tail, stride, offset = _block_visit_order(num_blocks)
    if num_blocks <= 2:
        visited = [0, 1][:num_blocks]
    else:
        assert math.gcd(stride, tail) == 1
        assert (tail - 1) * stride + offset <= np.iinfo(np.int32).max
        visited = [0] + [1 + (step * stride + offset) % tail for step in range(tail)]
    assert visited[0] == 0
    assert sorted(visited) == list(range(num_blocks))


@pytest.mark.parametrize("block,threshold", [(8192, 8192), (16384, 32768)])
@pytest.mark.parametrize("width", [65536, 65408])
@pytest.mark.parametrize(
    "distribution",
    [
        "random",
        "ascending",
        "descending",
        "equal",
        "cross_block_ties",
        "signed_zero",
        "masked",
    ],
)
def test_streaming_topk_contract(block, threshold, width, distribution):
    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore selection requires TPU")
    rng = np.random.default_rng(17)
    scores = rng.normal(size=(32, width)).astype(np.float32)
    if distribution == "ascending":
        scores[:] = np.arange(width, dtype=np.float32)
    elif distribution == "descending":
        scores[:] = -np.arange(width, dtype=np.float32)
    elif distribution == "equal":
        scores[:] = 1
    elif distribution == "cross_block_ties":
        # Equal Kth values survive repeated candidate compactions, with later
        # blocks also supplying larger winners that displace earlier ties.
        scores[:] = np.arange(width) // 4096 % 5
    elif distribution == "signed_zero":
        scores[:] = 0
        scores[:, ::2] = np.float32(-0.0)
    elif distribution == "masked":
        scores[:, ::3] = -np.inf
        scores[:4] = -np.inf
    lengths = np.resize(
        np.array([0, 1, 511, 512, 513, 8191, 8193, 16383, 50176, width], np.int32), 32
    )
    actual = np.asarray(
        streaming_topk(
            jnp.asarray(scores),
            valid_lengths=jnp.asarray(lengths),
            k=512,
            block_size=block,
            reconstruct_threshold=threshold,
        )
    )
    for row, length in enumerate(lengths):
        ids = actual[row]
        ids = ids[(ids >= 0) & (scores[row, np.maximum(ids, 0)] > -np.inf)]
        eligible = np.flatnonzero((np.arange(width) < length) & (scores[row] > -np.inf))
        bits = scores[row, eligible].view(np.uint32)
        keys = bits ^ np.where(bits >> 31, np.uint32(0xFFFFFFFF), np.uint32(0x80000000))
        order = np.lexsort((eligible, ~keys))[:512]
        np.testing.assert_array_equal(np.sort(ids), np.sort(eligible[order]))
        assert np.unique(ids).size == ids.size


@pytest.mark.parametrize("mode", ["with_lengths", "full_width", "mapped_full_width"])
def test_streaming_topk_inside_fori_loop(mode):
    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore selection requires TPU")
    tiles, tile_rows, width, k, cand_block = 2, 16, 65536, 512, 64
    rows = tiles * tile_rows
    rng = np.random.default_rng(29)
    scores_np = rng.normal(size=(rows, width)).astype(np.float32)
    lengths_np = rng.integers(k + 1, width + 1, size=(rows,), dtype=np.int32)
    mapping_np = np.broadcast_to(
        np.arange(width // cand_block, dtype=np.int32) * 2 + 3,
        (rows, width // cand_block),
    ).copy()
    scores = jnp.asarray(scores_np)
    lengths = jnp.asarray(lengths_np)
    mapping = jnp.asarray(mapping_np)

    def run_tile(tile_scores, tile_lengths, tile_mapping):
        if mode == "with_lengths":
            return streaming_topk(tile_scores, valid_lengths=tile_lengths, k=k)
        if mode == "full_width":
            return streaming_topk(tile_scores, valid_lengths=None, k=k)
        return streaming_topk(
            tile_scores,
            valid_lengths=None,
            logical_blocks=tile_mapping,
            k=k,
            unordered=True,
        )

    @jax.jit
    def run_looped(scores, lengths, mapping):
        init = jnp.full((rows, k), -1, jnp.int32)

        def body(tile, out):
            start = tile * tile_rows
            selected = run_tile(
                jax.lax.dynamic_slice_in_dim(scores, start, tile_rows),
                jax.lax.dynamic_slice_in_dim(lengths, start, tile_rows),
                jax.lax.dynamic_slice_in_dim(mapping, start, tile_rows),
            )
            return jax.lax.dynamic_update_slice_in_dim(out, selected, start, axis=0)

        return jax.lax.fori_loop(0, tiles, body, init)

    looped = np.asarray(run_looped(scores, lengths, mapping))
    direct = np.asarray(run_tile(scores, lengths, mapping))
    if mode == "mapped_full_width":
        np.testing.assert_array_equal(np.sort(looped, axis=1), np.sort(direct, axis=1))
    else:
        np.testing.assert_array_equal(looped, direct)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
