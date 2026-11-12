# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the DeepSeek-V4.1 indexer torch ops: fakes match the traced
JAX shapes, and the key writer lands each record on the contract's page.

The fakes are what torch.compile traces; a fake whose shape drifts from the
exported JAX function only fails at TPU run time, so both are compared here.
"""

import importlib.util
import os
import sys
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci, register_tpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")
register_tpu_ci(
    est_time=30,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "deepseek-v4.1-flash", "accuracy", "compile"],
)

_HAS_JAX = importlib.util.find_spec("jax") is not None
pytestmark = pytest.mark.skipif(not _HAS_JAX, reason="requires JAX")

if _HAS_JAX:
    import jax
    import jax.numpy as jnp

    from sglang_tpu import pallas_export
    from sglang_tpu.kernels.deepseek_v41.index_selection.fp8.cache import (
        PackedFP8Layout,
        read_fp8_records,
    )
    from sglang_tpu.models.deepseek_v41.ops import indexer

FULL_PAGE = 256


def _layout(pages):
    return (pages, 33, 0, 128)


@pytest.mark.parametrize(
    "tokens,requests,index_pages,count,publish,consume",
    [
        (1, 1, 8, 0, False, False),
        (7, 3, 16, 0, False, False),
        (16, 2, 32, 2048, True, False),
        (16, 2, 32, 2048, False, True),
        (300, 2, 32, 128, False, True),
        (5, 2, 32, 32, True, False),
        (0, 1, 4, 0, False, False),
    ],
)
def test_select_fake_matches_traced_shapes(
    tokens, requests, index_pages, count, publish, consume
):
    pages = requests * index_pages
    static_args = (*_layout(pages), 512, 8, publish, consume)
    arrays = (
        jax.ShapeDtypeStruct((tokens, 32, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((tokens, 32), jnp.float32),
        jax.ShapeDtypeStruct((pages * 33, 128), jnp.int32),
        jax.ShapeDtypeStruct((requests, index_pages), jnp.int32),
        jax.ShapeDtypeStruct((tokens,), jnp.int32),
        jax.ShapeDtypeStruct((tokens,), jnp.int32),
        jax.ShapeDtypeStruct((tokens, count), jnp.int32),
    )
    # Export traces for TPU7x on the CPU, as cpu_export_jax_op does.
    with (
        patch.object(pallas_export, "get_pallas_tpu_device_kind", return_value="TPU7x"),
        pallas_export.tpu_info_for_cpu_export(),
    ):
        traced = jax.eval_shape(lambda *a: indexer._select(*a, *static_args), *arrays)
    fake = indexer._fake_select(
        *(torch.empty(a.shape, dtype=torch.int32, device="meta") for a in arrays),
        *static_args,
    )
    for t, f in zip(traced, fake):
        assert tuple(f.shape) == t.shape and f.dtype == torch.int32
    assert traced[0].shape == (tokens, min(512, index_pages * 128))


@pytest.mark.parametrize("ratio", [1, 2])
def test_write_lands_records_on_contract_pages(ratio):
    """Record full_loc // r of compressed position j reads back through the
    indexer page table: ratio 1 expands full page p to 2p, 2p + 1."""
    full_pages, per_request = 9, 4
    pages = full_pages * FULL_PAGE // ratio // 128
    layout = _layout(pages)
    rng = np.random.default_rng(7)
    table = rng.permutation(np.arange(1, full_pages))[:per_request].astype(np.int32)
    width = per_request * FULL_PAGE // ratio
    j = np.arange(width)
    full_loc = table[j * ratio // FULL_PAGE] * FULL_PAGE + j * ratio % FULL_PAGE
    records = (full_loc // ratio).astype(np.int32)
    keys = rng.normal(size=(width, 128)).astype(np.float32)
    index_k = torch.zeros((pages * 33, 128), dtype=torch.int32)
    fake = indexer._fake_write_index_k(index_k, records, keys, *layout)
    assert fake.shape == index_k.shape and fake.dtype == torch.int32
    written = indexer._write_index_k(
        jnp.asarray(index_k.numpy()),
        jnp.asarray(np.concatenate([records, [-1]]).astype(np.int32)),
        jnp.asarray(np.concatenate([keys, np.ones((1, 128))]), jnp.bfloat16),
        *layout,
    )
    index_pages = (
        np.stack([2 * table, 2 * table + 1], -1).reshape(-1) if ratio == 1 else table
    )
    logical = index_pages[j // 128] * 128 + j % 128
    read = read_fp8_records((written,), jnp.asarray(logical), PackedFP8Layout(*layout))
    expected = np.asarray(jnp.asarray(keys, jnp.bfloat16).astype(jnp.float32))
    np.testing.assert_allclose(
        np.asarray(read, np.float32), expected, rtol=0.07, atol=1e-3
    )
    # Nothing but the given records changed: page 0 (full page 0) stays null.
    assert not np.asarray(written[: (2 if ratio == 1 else 1) * 33]).any()


def test_mask_pages_past_rows_invalidates_trailing_and_idle_columns():
    index_pages = jnp.array(
        [
            [6, 7, 10, 11, 0, 1, 0, 1],
            [4, 5, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0, 1],
        ],
        dtype=jnp.int32,
    )
    requests = jnp.array([0, 0, 1, -1], dtype=jnp.int32)
    # Row 0 max visible length = 300 -> ceil(300/128) = 3 pages (cols 0..2 kept)
    # Row 1 max visible length = 128 -> ceil(128/128) = 1 page  (col 0 kept)
    # Row 2 has no live queries -> all -1
    lengths = jnp.array([250, 300, 128, 999], dtype=jnp.int32)
    masked = np.asarray(
        indexer._mask_pages_past_rows(index_pages, requests, lengths, 128)
    )
    np.testing.assert_array_equal(
        masked,
        np.array(
            [
                [6, 7, 10, -1, -1, -1, -1, -1],
                [4, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ],
            dtype=np.int32,
        ),
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
