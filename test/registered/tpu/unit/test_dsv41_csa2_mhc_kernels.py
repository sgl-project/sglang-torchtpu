# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: tests/kernels/deepseek_v4/o_projection_test.py,
#         tests/kernels/deepseek_v41/test_csa2_prefill.py,
#         tests/kernels/deepseek_v41/test_csa2_pipeline.py,
#         tests/kernels/deepseek_v41/test_shifted_post_pre_rms.py
# SPDX-License-Identifier: Apache-2.0
"""Registered TPU unit tests for DSV4.1 CSA2, cache publish/gather, mHC, and wo_a_projection kernels."""

import os
import sys

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=300,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "deepseek-v4.1-flash", "accuracy", "compile"],
)

os.environ.setdefault("JAX_PLATFORMS", "tpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.kernels.deepseek_v4.o_projection import (
    LANE,
    _rotate_gptj,
    wo_a_projection,
)
from sglang_tpu.kernels.deepseek_v41.csa2.cache import (
    encode_csa2_kv,
    gather_csa2_kv_sc,
    prepare_swa_metadata,
    publish_csa2_kv,
    swa_word_pages,
    write_swa_word_pool,
)
from sglang_tpu.kernels.deepseek_v41.csa2.csa2 import (
    csa2,
    csa2_gather_chunk_size,
    csa2_pipeline,
)
from sglang_tpu.kernels.deepseek_v41.mhc.delayed import (
    _delayed_pre_flat_small,
    _post_and_collapse_jax,
    delayed_post_pre_flat_jax,
    delayed_pre_flat_jax,
    delayed_pre_jax,
)


def _pack_swa_word_pages(swa_bf16: jax.Array) -> jax.Array:
    u16 = jax.lax.bitcast_convert_type(
        swa_bf16.reshape(swa_bf16.shape[0], swa_bf16.shape[1], 2, 2, 128).transpose(
            0, 2, 1, 3, 4
        ),
        jnp.uint16,
    ).astype(jnp.uint32)
    words_u32 = u16[:, :, :, 0, :] | (u16[:, :, :, 1, :] << 16)
    return jax.lax.bitcast_convert_type(words_u32, jnp.int32)


def _unpack_swa_word_pages(pages_i32: jax.Array) -> jax.Array:
    u32 = jax.lax.bitcast_convert_type(pages_i32, jnp.uint32)
    lo = (u32 & 0xFFFF).astype(jnp.uint16)
    hi = ((u32 >> 16) & 0xFFFF).astype(jnp.uint16)
    bf16_u16 = jnp.stack((lo, hi), axis=3).transpose(0, 2, 1, 3, 4)
    return jax.lax.bitcast_convert_type(bf16_u16, jnp.bfloat16).reshape(
        pages_i32.shape[0], pages_i32.shape[2], 512
    )


def prefill_reference(
    q: jax.Array,
    words: jax.Array,
    scales: jax.Array,
    swa_kv: jax.Array | tuple[jax.Array, jax.Array],
    starts: jax.Array,
    positions: jax.Array,
    sink: jax.Array,
    rope: jax.Array,
    *,
    swa_metadata: jax.Array | None = None,
    main_lengths: jax.Array | None = None,
    swa_bounds: jax.Array | None = None,
    scale: float = 512**-0.5,
    compress_ratio: int = 2,
) -> jax.Array:
    """Independent JAX attention reference for validating CSA2 modes."""
    tokens = q.shape[0]
    if swa_metadata is None:
        assert not isinstance(swa_kv, tuple)
        swa = swa_kv
        valid = jnp.arange(128)[None, :] < jnp.minimum(positions + 1, 128)[:, None]
        swa = jnp.where(valid[..., None], swa, 0)
    else:
        banks = swa_kv if isinstance(swa_kv, tuple) else (swa_kv,)
        if banks[0].dtype == jnp.int32:
            banks = tuple(_unpack_swa_word_pages(b) for b in banks)
        pages, bounds = swa_metadata, swa_bounds
        request = jnp.minimum(
            jnp.searchsorted(starts[1:], jnp.arange(tokens), side="right"),
            pages.shape[0] - 1,
        )
        page_size = banks[0].shape[1]
        left = jnp.maximum(0, positions - 127) if bounds is None else bounds[request, 0]
        right = positions + 1 if bounds is None else bounds[request, 1]
        keys = left[:, None] + jnp.arange(128 if bounds is None else 192)
        columns = (keys - pages[request, 0, None]) // page_size
        valid = (keys < right[:, None]) & (keys >= pages[request, 1, None])
        valid &= (columns >= 0) & (columns < pages.shape[1] - 2)
        columns = jnp.clip(columns, 0, pages.shape[1] - 3)
        page_ids = pages[request[:, None], columns + 2]
        swa = banks[0][jnp.clip(page_ids, 0, banks[0].shape[0] - 1), keys % page_size]
        if len(banks) == 2:
            high = banks[1][
                jnp.clip(page_ids - banks[0].shape[0], 0, banks[1].shape[0] - 1),
                keys % page_size,
            ]
            swa = jnp.where((page_ids < banks[0].shape[0])[..., None], swa, high)
        swa = jnp.where(valid[..., None], swa, 0)
    if words.dtype == jnp.int32:
        bits = (
            words[:, :, None, :] >> (8 * jnp.arange(4))[None, None, :, None]
        ).astype(jnp.uint8)
        main = lax.bitcast_convert_type(bits, jnp.float8_e4m3fn).astype(jnp.bfloat16)
    else:
        main = words.astype(jnp.bfloat16)
    main = main.reshape(tokens, 512, 512)
    lengths = (
        jnp.minimum((positions + 1) // compress_ratio, 512)
        if main_lengths is None
        else main_lengths
    )
    main_logits = jnp.einsum(
        "thd,tkd->thk", q, main, preferred_element_type=jnp.float32
    )
    main_logits *= scales[:, None, :] * scale
    main_logits = jnp.where(
        jnp.arange(512) < lengths[:, None, None], main_logits, -jnp.inf
    )
    swa_logits = (
        jnp.einsum("thd,tkd->thk", q, swa, preferred_element_type=jnp.float32) * scale
    )
    swa_logits = jnp.where(valid[:, None, :], swa_logits, -jnp.inf)
    logits = jnp.concatenate((main_logits, swa_logits), axis=-1)
    maximum = jnp.maximum(logits.max(axis=-1, keepdims=True), sink[None, :, None])
    weights = jnp.exp(logits - maximum)
    weights /= weights.sum(axis=-1, keepdims=True) + jnp.exp(
        sink[None, :, None] - maximum
    )
    main_p = (weights[:, :, :512] * scales[:, None, :]).astype(jnp.bfloat16)
    swa_p = weights[:, :, 512:].astype(jnp.bfloat16)
    output = jnp.einsum(
        "thk,tkd->thd", main_p, main, preferred_element_type=jnp.float32
    )
    output += jnp.einsum("thk,tkd->thd", swa_p, swa, preferred_element_type=jnp.float32)
    tail = output[:, :, 384:]
    tail = (
        tail * rope[:, None, 0, :]
        + tail[:, :, jnp.arange(128) ^ 1] * rope[:, None, 1, :]
    )
    output = jnp.concatenate((output[:, :, :384], tail), axis=-1).astype(jnp.bfloat16)
    return jnp.where((jnp.arange(tokens) < starts[-1])[:, None, None], output, 0)


def _prefill_inputs(
    banked=False,
    non_causal=False,
    heads=8,
    page_size=64,
    word_swa=False,
    fp8_main=False,
):
    rng = np.random.default_rng(103)
    tokens = 27
    starts = np.array([0, 0, 17, 17, 25, 25], np.int32)
    positions = np.array([*range(120, 137), *range(250, 258), 0, 0], np.int32)

    def random(shape):
        return jnp.asarray(rng.normal(size=shape) * 0.2, jnp.bfloat16)

    main = random((tokens, 512, 4, 128)).astype(jnp.float8_e4m3fn)
    bits = jax.lax.bitcast_convert_type(main, jnp.uint8).astype(jnp.uint32)
    words = jnp.bitwise_or.reduce(
        bits << (8 * jnp.arange(4, dtype=jnp.uint32))[None, None, :, None], axis=2
    ).view(jnp.int32)
    base = np.array([0, 0, 0, 64, 0], np.int32)
    visible = np.array([0, 20, 0, 130, 0], np.int32)
    columns = 384 // page_size
    table = np.arange(1, 5 * columns + 1, dtype=np.int32).reshape(5, columns)
    metadata = jnp.asarray(np.column_stack((base, visible, table)))
    swa = random((table.size + 1, page_size, 512)).at[0].set(0)
    bounds = np.array([[0, 0], [0, 137], [0, 0], [123, 258], [0, 0]], np.int32)
    for request in (1, 3):
        absolute = base[request] + np.arange(384).reshape(columns, page_size)
        keep = (absolute >= visible[request]) & (absolute < bounds[request, 1])
        swa = swa.at[table[request]].set(
            jnp.where(keep[..., None], swa[table[request]], jnp.nan)
        )
    if word_swa:
        swa = _pack_swa_word_pages(swa)
    if banked:
        split = 3 * columns + 1
        swa = (swa[:split], swa[split:])
    angles = jnp.repeat(jnp.asarray(rng.normal(size=(tokens, 64)), jnp.float32), 2, -1)
    rope = jnp.stack(
        (jnp.cos(angles), jnp.sin(angles) * jnp.tile(jnp.array([1, -1]), 64)), 1
    )
    lengths = jnp.resize(jnp.array([0, 1, 127, 511, 512], jnp.int32), tokens)
    args = (
        random((tokens, heads, 512)),
        main if fp8_main else words,
        jnp.asarray(rng.uniform(0.1, 0.3, (tokens, 512)), jnp.float32),
        swa,
        jnp.asarray(starts),
        jnp.asarray(positions),
        jnp.linspace(-1, 1, heads, dtype=jnp.float32),
        rope,
    )
    kwargs = dict(
        swa_metadata=metadata,
        main_lengths=lengths,
        swa_bounds=jnp.asarray(bounds) if non_causal else None,
    )
    return args, kwargs


@pytest.mark.parametrize(
    "bq,banked,non_causal,heads,page_size,word_swa,fp8_main",
    [
        (6, False, False, 8, 64, False, False),
        (16, True, True, 8, 64, False, False),
        (16, True, False, 128, 64, False, False),
        (6, False, False, 64, 32, False, False),
        (16, False, True, 64, 32, False, False),
        (6, False, False, 8, 64, True, True),
        (16, True, True, 8, 32, True, False),
    ],
)
def test_csa2_prefill_visibility_and_tail_rows(
    bq, banked, non_causal, heads, page_size, word_swa, fp8_main
):
    args, kwargs = _prefill_inputs(
        banked, non_causal, heads, page_size, word_swa=word_swa, fp8_main=fp8_main
    )
    expected = prefill_reference(*args, **kwargs)
    actual = csa2(*args, **kwargs, bq=bq, interpret=(jax.default_backend() != "tpu"))
    np.testing.assert_allclose(
        np.asarray(actual, np.float32),
        np.asarray(expected, np.float32),
        atol=3e-3,
        rtol=3e-2,
    )
    np.testing.assert_array_equal(actual[25:], 0)


@pytest.mark.parametrize("fp8_main", [False, True])
def test_csa2_unpaged_swa_and_direct_fp8(fp8_main):
    args, _ = _prefill_inputs(fp8_main=fp8_main)
    q, main_kv, scales, _, starts, positions, sink, rope = args
    rng = np.random.default_rng(77)
    seq_kv = jnp.asarray(rng.normal(size=(5, 512, 512)) * 0.2, jnp.bfloat16)
    tokens = q.shape[0]
    request = jnp.minimum(
        jnp.searchsorted(starts[1:], jnp.arange(tokens), side="right"),
        4,
    )
    pos_window = jnp.maximum(0, positions[:, None] - 127) + jnp.arange(128)[None, :]
    swa = seq_kv[request[:, None], pos_window]
    args_unpaged = (q, main_kv, scales, swa, starts, positions, sink, rope)
    expected = prefill_reference(*args_unpaged, compress_ratio=2)
    actual = csa2(
        *args_unpaged,
        bq=6,
        compress_ratio=2,
        interpret=(jax.default_backend() != "tpu"),
    )
    np.testing.assert_allclose(
        np.asarray(actual, np.float32),
        np.asarray(expected, np.float32),
        atol=3e-3,
        rtol=3e-2,
    )
    np.testing.assert_array_equal(actual[25:], 0)


@pytest.mark.parametrize("non_causal", [False, True])
@pytest.mark.parametrize("valid_tokens", [0, 17])
def test_csa2_single_request_direct_output_preserves_padding(non_causal, valid_tokens):
    args, kwargs = _prefill_inputs(non_causal=non_causal)
    q, words, scales, swa, _, positions, sink, rope = args

    def padded(x):
        return jnp.pad(x[:17], ((0, 15), *[(0, 0)] * (x.ndim - 1)))

    args = (
        padded(q),
        padded(words),
        padded(scales),
        swa,
        jnp.array([0, valid_tokens], jnp.int32),
        padded(positions),
        sink,
        padded(rope),
    )
    kwargs = dict(
        swa_metadata=kwargs["swa_metadata"][1:2],
        main_lengths=padded(kwargs["main_lengths"]),
        swa_bounds=kwargs["swa_bounds"][1:2] if non_causal else None,
    )
    expected = prefill_reference(*args, **kwargs)
    actual = csa2(*args, **kwargs, bq=16, interpret=(jax.default_backend() != "tpu"))
    np.testing.assert_allclose(
        np.asarray(actual, np.float32),
        np.asarray(expected, np.float32),
        atol=3e-3,
        rtol=3e-2,
    )
    np.testing.assert_array_equal(actual[valid_tokens:], 0)


@pytest.mark.parametrize("tokens", [8, 16, 24, 64, 72])
def test_csa2_short_batches(tokens):
    if jax.default_backend() != "tpu":
        pytest.skip("Async MXU operations require TPU")
    args, kwargs = _prefill_inputs()
    q, words, scales, swa, _, _, sink, rope = args
    args = (
        jnp.resize(q, (tokens, 8, 512)),
        jnp.resize(words, (tokens, 512, 128)),
        jnp.resize(scales, (tokens, 512)),
        jnp.nan_to_num(swa),
        jnp.array([0, tokens], jnp.int32),
        jnp.arange(120, 120 + tokens, dtype=jnp.int32),
        sink,
        jnp.resize(rope, (tokens, 2, 128)),
    )
    kwargs = dict(
        swa_metadata=kwargs["swa_metadata"][1:2],
        main_lengths=jnp.resize(kwargs["main_lengths"], (tokens,)),
    )
    actual = csa2(*args, **kwargs, bq=8)
    expected = prefill_reference(*args, **kwargs)
    np.testing.assert_allclose(actual, expected, atol=3e-3, rtol=3e-2)


@pytest.mark.parametrize("page_size", [16, 32, 64])
@pytest.mark.parametrize("banked", [False, True])
def test_swa_word_pool_publish_round_trip(page_size, banked):
    if jax.default_backend() != "tpu" or pltpu.get_tpu_info().sparse_core is None:
        pytest.skip("SparseCore SWA publication requires TPU with SparseCore")
    num_pages = 4
    rows = num_pages * 2 * page_size
    raw = jnp.zeros((rows, 128), jnp.int32)
    owner = (raw[: rows // 2], raw[rows // 2 :]) if banked else raw
    records = np.array(
        [
            -1,
            0,
            page_size - 1,
            page_size,
            page_size + 1,
            2 * page_size - 1,
            2 * page_size,
            3 * page_size + 1,
            4 * page_size,
        ],
        dtype=np.int32,
    )
    values = (
        np.arange(len(records) * 512, dtype=np.float32).reshape(-1, 512) % 97
    ).astype(np.float32)
    expected = np.zeros((num_pages * page_size, 512), dtype=np.float32)
    for rec, val in zip(records, values):
        if page_size <= rec < num_pages * page_size:
            expected[rec] = val
    updated = write_swa_word_pool(
        owner,
        jnp.asarray(records),
        jnp.asarray(values, jnp.bfloat16),
        page_size=page_size,
    )
    pages = swa_word_pages(updated, page_size)
    joined_pages = jnp.concatenate(pages, axis=0) if banked else pages
    unpacked = _unpack_swa_word_pages(joined_pages).reshape(-1, 512)
    np.testing.assert_array_equal(
        np.asarray(unpacked, np.float32),
        np.asarray(jnp.asarray(expected, jnp.bfloat16), np.float32),
    )


@pytest.mark.parametrize("banked", [False, True])
def test_csa2_kv_publish_and_gather_round_trip(banked):
    if jax.default_backend() != "tpu" or pltpu.get_tpu_info().sparse_core is None:
        pytest.skip("SparseCore CSA2 publish/gather requires TPU with SparseCore")
    rng = np.random.default_rng(7)
    total_records = 256
    words = jnp.zeros((total_records, 128), jnp.int32)
    scales = jnp.zeros((total_records,), jnp.float32)
    word_owner = (words[:113], words[113:]) if banked else words
    scale_owner = (scales[:113], scales[113:]) if banked else scales

    pub_records = np.array([5, 19, 112, 113, 114, 240, -1, 300], dtype=np.int32)
    latent = jnp.asarray(
        rng.standard_normal((len(pub_records), 512)) * 0.2, jnp.bfloat16
    )
    expected_words, expected_scales = encode_csa2_kv(latent)

    word_owner, scale_owner = publish_csa2_kv(
        word_owner,
        scale_owner,
        jnp.asarray(pub_records),
        latent,
    )

    gather_ids = np.full((2, 512), -1, dtype=np.int32)
    gather_ids[0, :6] = [5, 19, 112, 113, 114, 240]
    gather_ids[1, :4] = [113, -1, 300, 5]
    got_words, got_scales = gather_csa2_kv_sc(
        word_owner,
        scale_owner,
        jnp.asarray(gather_ids),
    )
    np.testing.assert_array_equal(got_words[0, :6], expected_words[:6])
    np.testing.assert_array_equal(got_scales[0, :6], expected_scales[:6])
    np.testing.assert_array_equal(got_words[0, 6:], 0)
    np.testing.assert_array_equal(got_scales[0, 6:], 0)
    np.testing.assert_array_equal(got_words[1, 0], expected_words[3])
    np.testing.assert_array_equal(got_words[1, 1], 0)
    np.testing.assert_array_equal(got_scales[1, 1], 0)
    np.testing.assert_array_equal(got_words[1, 2], 0)
    np.testing.assert_array_equal(got_scales[1, 2], 0)
    np.testing.assert_array_equal(got_words[1, 3], expected_words[0])
    np.testing.assert_array_equal(got_scales[1, 3], expected_scales[0])


@pytest.mark.parametrize("banked", [False, True])
@pytest.mark.parametrize("non_causal", [False, True])
def test_csa2_pipeline_chunked_attention(banked, non_causal):
    if jax.default_backend() != "tpu" or pltpu.get_tpu_info().sparse_core is None:
        pytest.skip("CSA2 pipeline requires TPU with SparseCore")
    tokens, heads, chunk_size, page_size = 193, 8, 64, 32
    rng = np.random.default_rng(31)

    def random(shape):
        return jnp.asarray(rng.standard_normal(shape) * 0.2, jnp.bfloat16)

    q = random((tokens, heads, 512))
    words, scales = encode_csa2_kv(random((257, 512)))
    starts = jnp.array([0, 0, 43, 43, 103, 155, 155], jnp.int32)
    positions = jnp.array(
        [*range(160, 203), *range(240, 300), *range(330, 382)] + [0] * 38,
        jnp.int32,
    )
    counts = jnp.array(rng.integers(0, 513, tokens), jnp.int32)
    counts = jnp.where(jnp.arange(tokens) < 155, counts, 0)
    records = jnp.array(rng.integers(0, 257, (tokens, 512)), jnp.int32)
    records = jnp.where(jnp.arange(512) < counts[:, None], records, -1)
    selected = jnp.maximum(records, 0)
    expected_words = jnp.where(records[:, :, None] >= 0, words[selected], 0)
    expected_scales = jnp.where(records >= 0, scales[selected], 0)
    table = jnp.arange(1, 97, dtype=jnp.int32).reshape(6, 16)
    metadata, bounds, _ = prepare_swa_metadata(
        table,
        starts,
        positions,
        jnp.array([0, 150, 0, 200, 300, 0], jnp.int32),
        page_size=page_size,
        page_stride=page_size,
        row_offset=0,
        non_causal=non_causal,
    )
    swa = random((97, page_size, 512)).at[0].set(0)
    word_banks = (words[:113], words[113:]) if banked else words
    scale_banks = (scales[:113], scales[113:]) if banked else scales
    swa_banks = (swa[:48], swa[48:]) if banked else swa
    angles = jnp.asarray(rng.standard_normal((tokens, 64)) * 0.1)
    rope = jnp.stack(
        (
            jnp.concatenate((jnp.ones_like(angles), jnp.cos(angles)), axis=1),
            jnp.concatenate((jnp.zeros_like(angles), jnp.sin(angles)), axis=1),
        ),
        axis=1,
    )
    sink = jnp.linspace(-1, 1, heads, dtype=jnp.float32)

    expected = csa2(
        q,
        expected_words,
        expected_scales,
        swa_banks,
        starts,
        positions,
        sink,
        rope,
        swa_metadata=metadata,
        swa_bounds=bounds,
        main_lengths=counts,
        scale=512**-0.5,
        compress_ratio=2,
        bq=6,
    )
    actual, gathered_words, gathered_scales = csa2_pipeline(
        q,
        word_banks,
        scale_banks,
        records,
        swa_banks,
        starts,
        positions,
        sink,
        rope,
        metadata,
        bounds,
        counts,
        chunk_size=chunk_size,
        bq=6,
        scale=512**-0.5,
        compress_ratio=2,
    )
    np.testing.assert_array_equal(gathered_words, expected_words)
    np.testing.assert_array_equal(gathered_scales, expected_scales)
    np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-2)
    np.testing.assert_array_equal(actual[155:], 0)
    assert csa2_gather_chunk_size(512) == 512
    assert csa2_gather_chunk_size(1024) == 128


@pytest.mark.parametrize("tokens", [64, 96, 128, 256, 384, 512, 1024])
def test_delayed_mhc_pallas_matches_xla_reference(tokens):
    if jax.default_backend() != "tpu":
        pytest.skip("Pallas mHC kernel requires TPU")
    mult, dim = 4, 5120
    mixes = mult * (mult + 2)
    ks = jax.random.split(jax.random.PRNGKey(tokens), 7)
    x = (jax.random.normal(ks[0], (tokens, dim)) * 0.5).astype(jnp.bfloat16)
    flat = (jax.random.normal(ks[1], (tokens, mult * dim)) * 0.5).astype(jnp.bfloat16)
    post_mix = jax.nn.sigmoid(jax.random.normal(ks[2], (tokens, mult, 1))) * 2.0
    comb_mix = jax.nn.softmax(jax.random.normal(ks[3], (tokens, mult, mult)), axis=-1)
    prev_pre = jax.nn.sigmoid(jax.random.normal(ks[4], (tokens, mult))) + 1e-6
    fn = (jax.random.normal(ks[5], (mixes, mult * dim)) * 0.01).astype(jnp.float32)
    scale = jnp.array([0.1, 0.2, 0.3], jnp.float32)
    base = (jax.random.normal(ks[6], (mixes,)) * 0.1).astype(jnp.float32)
    consts = (1e-20, 1e-6, 1e-6, 2.0, 20)

    got_pre = jax.jit(lambda *a: delayed_pre_flat_jax(*a, *consts))(
        flat, fn, scale, base, prev_pre
    )
    ref_pre = jax.jit(lambda *a: _delayed_pre_flat_small(*a, *consts))(
        flat, fn, scale, base, prev_pre
    )
    for r, g, rtol in zip(ref_pre, got_pre, (1e-4, 1e-4, 2.0**-6, 1e-4)):
        r_f, g_f = np.asarray(r, np.float32), np.asarray(g, np.float32)
        bound = np.maximum(np.abs(r_f), np.abs(g_f)) * rtol + 1e-5
        assert np.all(np.abs(r_f - g_f) <= bound)

    got_post_pre = jax.jit(lambda *a: delayed_post_pre_flat_jax(*a, *consts))(
        x, flat, post_mix, comb_mix, fn, scale, base, prev_pre
    )

    def _ref_post_pre(x, flat, post_mix, comb_mix, fn, scale, base, prev_pre):
        flat_new, value = _post_and_collapse_jax(x, flat, post_mix, comb_mix, prev_pre)
        post, comb, _, new_pre = delayed_pre_jax(
            flat_new.reshape(tokens, mult, dim),
            flat_new,
            fn,
            scale,
            base,
            prev_pre,
            *consts,
        )
        return flat_new, post, comb, value, new_pre

    ref_post_pre = jax.jit(_ref_post_pre)(
        x, flat, post_mix, comb_mix, fn, scale, base, prev_pre
    )
    for r, g, rtol in zip(
        ref_post_pre, got_post_pre, (2.0**-7, 1e-4, 1e-4, 2.0**-6, 1e-4)
    ):
        r_f, g_f = np.asarray(r, np.float32), np.asarray(g, np.float32)
        bound = np.maximum(np.abs(r_f), np.abs(g_f)) * rtol + 1e-5
        assert np.all(np.abs(r_f - g_f) <= bound)


def _o_projection_ref(activations, wo_a, wo_a_scale, cos_sin, *, num_groups, head_dim):
    x = activations
    if cos_sin is not None:
        cos = cos_sin[:, :LANE]
        sin = cos_sin[:, LANE:]
        lo = head_dim - LANE
        tail = x[:, :, lo:].astype(jnp.float32)
        roped = tail * cos[:, None, :] + _rotate_gptj(tail) * sin[:, None, :]
        x = jnp.concatenate([x[:, :, :lo], roped.astype(x.dtype)], axis=-1)
    num_tokens = x.shape[0]
    o_f = x.reshape(num_tokens, num_groups, -1)
    w = wo_a.reshape(o_f.shape[-1], num_groups, -1)
    s = wo_a_scale.reshape(num_groups, -1)
    z = (
        jnp.einsum(
            "tgd,dgr->tgr",
            o_f,
            w.astype(jnp.bfloat16),
            preferred_element_type=jnp.float32,
        )
        * s.astype(jnp.bfloat16)[None, ...]
    )
    return z.astype(jnp.bfloat16).reshape(num_tokens, -1)


@pytest.mark.parametrize("num_tokens", [13, 193, 256, 512, 4100])
@pytest.mark.parametrize("use_rope", [False, True])
def test_wo_a_projection_matches_reference(num_tokens, use_rope):
    if jax.default_backend() != "tpu":
        pytest.skip("wo_a_projection Pallas kernel requires TPU")
    rng = np.random.default_rng(num_tokens)
    num_groups, heads_per_group, head_dim, lora_rank = 4, 8, 512, 512
    reduction = heads_per_group * head_dim
    num_heads = num_groups * heads_per_group
    activations = jnp.asarray(
        rng.standard_normal((num_tokens, num_heads, head_dim)) * 0.2,
        dtype=jnp.bfloat16,
    )
    wo_a = jnp.asarray(
        rng.standard_normal((reduction, num_groups * lora_rank)) * 0.2,
        dtype=jnp.float8_e4m3fn,
    )
    wo_a_scale = jnp.asarray(
        rng.uniform(0.5, 1.5, size=num_groups * lora_rank),
        dtype=jnp.float32,
    )
    cos_sin = None
    if use_rope:
        angles = jnp.repeat(
            jnp.asarray(rng.standard_normal((num_tokens, 32)) * 0.1, jnp.float32),
            2,
            axis=-1,
        )
        cos = jnp.concatenate([jnp.ones((num_tokens, 64)), jnp.cos(angles)], axis=-1)
        sin = jnp.concatenate([jnp.zeros((num_tokens, 64)), jnp.sin(angles)], axis=-1)
        cos_sin = jnp.concatenate([cos, sin], axis=-1)

    got = wo_a_projection(
        activations,
        wo_a,
        wo_a_scale,
        cos_sin,
        quantize_activations=True,
    )
    ref = _o_projection_ref(
        activations,
        wo_a,
        wo_a_scale,
        cos_sin,
        num_groups=num_groups,
        head_dim=head_dim,
    )
    assert got.shape == (num_tokens, num_groups * lora_rank)
    np.testing.assert_allclose(
        np.asarray(got, np.float32),
        np.asarray(ref, np.float32),
        atol=1.5,
        rtol=5e-2,
    )


@pytest.mark.parametrize("num_tokens", [5, 13, 193, 256, 512, 1032, 4100])
def test_wo_a_projection_tiling_and_padding_contract(monkeypatch, num_tokens):
    """Verify 8-aligned tile_t/sub_t selection and output slicing for arbitrary token counts."""
    import sglang_tpu.kernels.deepseek_v4.o_projection as o_proj

    captured = {}

    def _fake_pallas_call(fn, *, out_shape, grid, in_specs, out_specs, **kwargs):
        tile_t, tile_r = out_specs.block_shape
        num_sub_t = fn.keywords["num_sub_t"]
        sub_t = tile_t // num_sub_t
        captured["padded_tokens"] = out_shape.shape[0]
        captured["tile_t"] = tile_t
        captured["sub_t"] = sub_t
        captured["grid"] = grid

        def _call(x, wo_a, wo_a_scale, cos_sin):
            return jnp.arange(
                out_shape.shape[0] * out_shape.shape[1], dtype=out_shape.dtype
            ).reshape(out_shape.shape)

        return _call

    monkeypatch.setattr(o_proj.pl, "pallas_call", _fake_pallas_call)
    num_groups, heads_per_group, head_dim, lora_rank = 2, 8, 512, 128
    x = jnp.zeros((num_tokens, num_groups * heads_per_group, head_dim), jnp.bfloat16)
    wo_a = jnp.zeros(
        (heads_per_group * head_dim, num_groups * lora_rank), jnp.float8_e4m3fn
    )
    wo_a_scale = jnp.ones((num_groups * lora_rank,), jnp.float32)
    cos_sin = jnp.zeros((num_tokens, 2 * LANE), jnp.float32)

    out = o_proj.wo_a_projection.__wrapped__(x, wo_a, wo_a_scale, cos_sin)
    assert out.shape == (num_tokens, num_groups * lora_rank)
    assert captured["padded_tokens"] >= num_tokens
    assert captured["padded_tokens"] % captured["tile_t"] == 0
    assert captured["tile_t"] % 8 == 0
    assert captured["sub_t"] % 8 == 0
    assert captured["tile_t"] % captured["sub_t"] == 0
    if num_tokens > 64:
        assert captured["tile_t"] >= 64


@pytest.mark.parametrize(
    "tokens,expect_pallas,expected_tile",
    [
        (32, False, None),
        (64, True, 64),
        (96, True, 96),
        (100, False, None),
        (128, True, 128),
        (256, True, 128),
        (384, True, 64),
        (512, True, 64),
        (768, True, 64),
        (1024, True, 64),
    ],
)
def test_delayed_mhc_pallas_gate_selection(
    monkeypatch, tokens, expect_pallas, expected_tile
):
    """Verify T=384, 512, 768 route through the Pallas mHC path with aligned tiles."""
    import sglang_tpu.kernels.deepseek_v41.mhc.delayed as delayed_mod

    pre_tiles = []
    post_pre_tiles = []

    def _fake_pre_sums(flat, weight, previous_pre, mixes, tile):
        pre_tiles.append(tile)
        t = flat.shape[0]
        dim = flat.shape[1] // previous_pre.shape[1]
        return (
            jnp.zeros((t, mixes), jnp.float32),
            jnp.ones((t, 1), jnp.float32),
            jnp.zeros((t, dim), flat.dtype),
        )

    def _fake_post_pre_sums(
        sub_x, flat, post_layer_mix, comb_res_mix, weight, previous_pre, mixes, tile
    ):
        post_pre_tiles.append(tile)
        t, width = flat.shape
        dim = width // previous_pre.shape[1]
        return (
            jnp.zeros((t, width), flat.dtype),
            jnp.zeros((t, mixes), jnp.float32),
            jnp.ones((t, 1), jnp.float32),
            jnp.zeros((t, dim), flat.dtype),
        )

    monkeypatch.setattr(delayed_mod, "_flat_pre_sums", _fake_pre_sums)
    monkeypatch.setattr(delayed_mod, "_flat_post_pre_sums", _fake_post_pre_sums)

    mult, dim = 4, 128
    mixes = mult * (mult + 2)
    x = jnp.zeros((tokens, dim), jnp.bfloat16)
    flat = jnp.zeros((tokens, mult * dim), jnp.bfloat16)
    post_mix = jnp.ones((tokens, mult, 1), jnp.float32)
    comb_mix = jnp.ones((tokens, mult, mult), jnp.float32) * 0.25
    prev_pre = jnp.ones((tokens, mult), jnp.float32) * 0.25
    fn = jnp.zeros((mixes, mult * dim), jnp.float32)
    scale = jnp.array([0.1, 0.2, 0.3], jnp.float32)
    base = jnp.zeros((mixes,), jnp.float32)
    consts = (1e-20, 1e-6, 1e-6, 2.0, 20)

    delayed_pre_flat_jax(flat, fn, scale, base, prev_pre, *consts)
    delayed_post_pre_flat_jax(
        x, flat, post_mix, comb_mix, fn, scale, base, prev_pre, *consts
    )
    if expect_pallas:
        assert pre_tiles == [expected_tile]
        assert post_pre_tiles == [expected_tile]
    else:
        assert pre_tiles == []
        assert post_pre_tiles == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
