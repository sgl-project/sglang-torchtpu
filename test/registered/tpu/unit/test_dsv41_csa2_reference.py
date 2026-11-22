# SPDX-License-Identifier: Apache-2.0
"""CSA2 kernels against the vLLM dense reference with BF16 SWA pages."""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang_tpu.models.deepseek_v41.kernels.csa2.cache import (
    encode_csa2_kv,
    prepare_swa_metadata,
)
from sglang_tpu.models.deepseek_v41.kernels.csa2.csa2 import csa2
from sglang_tpu.models.deepseek_v41.kernels.csa2.pipeline import csa2_pipeline
from sglang_tpu.models.deepseek_v41.kernels.csa2.reference import prefill_reference

register_tpu_ci(
    est_time=120,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "deepseek-v4.1-flash", "accuracy", "compile"],
)


@pytest.mark.parametrize(
    "phase, lengths, tokens",
    [
        ("decode", [6] * 8, 48),
        # Same shape and final boundary as decode. Only the interior
        # boundaries show that this batch needs the ragged path.
        ("prefill", [17, 0, 8, 7, 6, 4, 2, 4], 48),
        # A request crosses the 96-row boundary and the last chunk has padding.
        ("pipeline", [40, 0, 70, 11], 144),
    ],
    ids=["uniform-decode", "ragged-prefill", "chunked-pipeline"],
)
def test_matches_dense_reference(phase, lengths, tokens):
    if jax.default_backend() != "tpu":
        pytest.skip("CSA2 async matmul and DMA require TPU")
    rng = np.random.default_rng(2026)
    page_size, columns, heads, records_count = 64, 8, 64, 1024
    batch = len(lengths)

    def random_bf16(shape):
        return jnp.asarray(rng.standard_normal(shape, dtype=np.float32), jnp.bfloat16)

    starts = jnp.asarray(np.r_[0, np.cumsum(lengths)], jnp.int32)
    live = sum(lengths)
    positions = np.zeros(tokens, np.int32)
    positions[:live] = np.concatenate(
        [
            np.arange(length) + 139 + 17 * request
            for request, length in enumerate(lengths)
        ]
    )
    positions = jnp.asarray(positions)
    table = jnp.asarray(
        rng.permutation(np.arange(1, batch * columns + 1, dtype=np.int32)).reshape(
            batch, columns
        )
    )
    metadata, bounds, _ = prepare_swa_metadata(
        table,
        starts,
        positions,
        jnp.full((batch,), 75, jnp.int32),
        page_size=page_size,
        page_stride=page_size,
        row_offset=0,
        non_causal=phase == "decode",
    )
    bounds = bounds if phase == "decode" else None
    q = random_bf16((tokens, heads, 512))
    pool, scales = encode_csa2_kv(random_bf16((records_count, 512)))
    counts = rng.choice(np.array([0, 1, 127, 511, 512], np.int32), size=tokens)
    counts[live:] = 0
    counts = jnp.asarray(counts)
    records = jnp.asarray(
        rng.integers(0, records_count, size=(tokens, 512), dtype=np.int32)
    )
    keep = jnp.arange(512) < counts[:, None]
    words = jnp.where(keep[..., None], pool[records], 0)
    gathered_scales = jnp.where(keep, scales[records], 0)
    # Both negative and positive out-of-bounds gather indices must zero the lane.
    invalid = jnp.where(jnp.arange(512) % 2 == 0, -1, records_count + 7)
    records = jnp.where(keep, records, invalid)
    swa = random_bf16((batch * columns + 1, page_size, 512)).at[0].set(0)
    mid = swa.shape[0] // 2
    swa_banks = (swa[:mid], swa[mid:]) if phase != "prefill" else swa
    sink = jnp.linspace(-0.5, 0.5, heads, dtype=jnp.float32)
    angles = jnp.repeat(jnp.asarray(rng.normal(size=(tokens, 64)), jnp.float32), 2, -1)
    rope = jnp.stack((jnp.cos(angles), jnp.sin(angles) * (-1) ** jnp.arange(128)), 1)
    expected = prefill_reference(
        q,
        words,
        gathered_scales,
        swa_banks,
        starts,
        positions,
        sink,
        rope,
        swa_metadata=metadata,
        swa_bounds=bounds,
        main_lengths=counts,
    )
    if phase == "pipeline":
        actual, actual_words, actual_scales = csa2_pipeline(
            q,
            (pool[:512], pool[512:]),
            (scales[:512], scales[512:]),
            records,
            swa_banks,
            starts,
            positions,
            sink,
            rope,
            metadata,
            bounds,
            counts,
            chunk_size=96,
            bq=6,
            scale=512**-0.5,
            compress_ratio=2,
            deep_gather=True,
        )
        np.testing.assert_array_equal(actual_words, words)
        np.testing.assert_array_equal(actual_scales, gathered_scales)
    else:
        actual = csa2(
            q,
            words,
            gathered_scales,
            swa_banks,
            starts,
            positions,
            sink,
            rope,
            swa_metadata=metadata,
            swa_bounds=bounds,
            main_lengths=counts,
            bq=6,
        )
    np.testing.assert_allclose(
        np.asarray(actual, np.float32),
        np.asarray(expected, np.float32),
        atol=3e-3,
        rtol=3e-2,
    )
    np.testing.assert_array_equal(actual[live:], 0)


def test_casts_normalized_pv_weights_to_bf16():
    if jax.default_backend() != "tpu":
        pytest.skip("CSA2 async matmul and DMA require TPU")

    # The first query has one nonzero main key, one zero SWA key, and a
    # sink. All logits are zero, so each attention weight is 1/3. Folding
    # the main scale into 1/3 before the BF16 cast returns 0.333984375.
    # Rounding the unnormalized weight and dividing after PV gives 0.33203125.
    tokens, heads = 8, 64
    words, scales = encode_csa2_kv(jnp.ones((1, 512), jnp.bfloat16))
    q = jnp.zeros((tokens, heads, 512), jnp.bfloat16)
    main_words = jnp.zeros((tokens, 512, 128), jnp.int32).at[0, 0].set(words[0])
    main_scales = jnp.zeros((tokens, 512), jnp.float32).at[0, 0].set(scales[0])
    swa = jnp.zeros((1, 64, 512), jnp.bfloat16)
    starts = jnp.array([0] + [1] * tokens, jnp.int32)
    positions = jnp.zeros((tokens,), jnp.int32)
    sink = jnp.zeros((heads,), jnp.float32)
    rope = jnp.zeros((tokens, 2, 128), jnp.float32).at[:, 0, :].set(1.0)
    metadata = jnp.zeros((tokens, 3), jnp.int32)
    counts = jnp.zeros((tokens,), jnp.int32).at[0].set(1)
    actual = csa2(
        q,
        main_words,
        main_scales,
        swa,
        starts,
        positions,
        sink,
        rope,
        swa_metadata=metadata,
        main_lengths=counts,
        bq=6,
    )
    np.testing.assert_array_equal(np.asarray(actual[0], np.float32), 0.333984375)
    np.testing.assert_array_equal(actual[1:], 0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
