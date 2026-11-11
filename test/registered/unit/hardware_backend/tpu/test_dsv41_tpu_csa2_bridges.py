# SPDX-License-Identifier: Apache-2.0
"""CPU tracing/fake contracts and addressing for the live DSV4.1 CSA2 bridges.

No TorchTPU or TPU hardware is needed. Numerical and compiled device execution
remain in the registered TPU CSA2 test file.
"""

import inspect
import os
import sys
from unittest import mock

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.models.deepseek_v41.kernels.csa2.cache import prepare_main_records
from sglang_tpu.models.deepseek_v41.ops import csa2 as ops

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

MAX_POSITION = 2048
SCALE = 512**-0.5


def _theta(max_position=MAX_POSITION):
    # Interleaved-pair RoPE angles [max_position, 32], as freqs_cis = e^{i theta}.
    freqs = 10000.0 ** (-np.arange(0, ops.ROPE_DIM, 2) / ops.ROPE_DIM)
    return np.arange(max_position)[:, None] * freqs[None, :]


def _cos_sin(theta):
    """``get_or_build_cos_sin_cache(freqs_cis)``: ``[cos | sin]`` FP32."""
    freqs_cis = torch.polar(
        torch.ones(theta.shape, dtype=torch.float64), torch.from_numpy(theta)
    ).to(torch.complex64)
    real_imag = torch.view_as_real(freqs_cis)
    return torch.cat([real_imag[..., 0], real_imag[..., 1]], dim=-1).float()


def _inverse_rope_reference(output, theta_rows):
    """``output`` [..., H, 512]; rotate pairs of the last 64 by conj(e^{i theta})."""
    result = output.astype(np.float64)
    tail = result[..., 512 - ops.ROPE_DIM :].reshape(*result.shape[:-1], -1, 2)
    z = (tail[..., 0] + 1j * tail[..., 1]) * np.exp(-1j * theta_rows)[..., None, :]
    tail = np.stack((z.real, z.imag), axis=-1).reshape(*result.shape[:-1], -1)
    result[..., 512 - ops.ROPE_DIM :] = tail
    return result


def _tpu_info_patch():
    from sglang_tpu import pallas_export

    return mock.patch.object(
        pallas_export, "get_pallas_tpu_device_kind", return_value="TPU7x"
    )


# =============================================================================
# CPU: bridges, fakes, addressing
# =============================================================================

_BATCH, _TOKENS, _COLUMNS = 3, 40, 8
_RATIO = 2
_RECORDS = 256 // _RATIO


def _bridge_examples():
    """name -> (array inputs, static inputs) for each JAX bridge."""
    i32 = np.int32
    table = ((_BATCH, _COLUMNS), i32)
    starts = ((_BATCH + 1,), i32)
    positions = ((_TOKENS,), i32)
    pages = ((20 * 128, 128), i32)
    q = ((_TOKENS, 16, 512), ml_dtypes.bfloat16)
    common = (starts, positions, ((16,), np.float32), ((MAX_POSITION, 64), np.float32))
    meta = ((_BATCH, 2 + ops.swa_metadata_columns(_TOKENS)), i32)
    examples = {
        "swa_metadata": ((table, starts, positions, ((_BATCH,), i32)), ()),
        "publish_swa": ((pages, positions, ((_TOKENS, 512), ml_dtypes.bfloat16)), ()),
        "publish_main": (
            (
                ((4 * _RECORDS, 128), i32),
                ((4 * _RECORDS,), np.float32),
                ((24,), i32),
                ((24, 512), ml_dtypes.bfloat16),
            ),
            (),
        ),
        "publish_encoded": (
            (
                ((4 * _RECORDS, 128), i32),
                ((4 * _RECORDS,), np.float32),
                ((24,), i32),
                ((24, 128), i32),
                ((24, 1), np.float32),
            ),
            (),
        ),
        "attend_swa": ((q, pages, *common, meta), (SCALE,)),
        "attend_main": (
            (
                q,
                ((4 * _RECORDS, 128), i32),
                ((4 * _RECORDS,), np.float32),
                ((_TOKENS, 512), i32),
                ((_BATCH, 4), i32),
                pages,
                *common,
                meta,
            ),
            (_RECORDS, _RATIO, SCALE),
        ),
    }
    examples["gather_attend"] = examples["attend_main"]
    examples["attend_prepared"] = (
        (
            q,
            ((_TOKENS, 512, 128), i32),
            ((_TOKENS, 512), np.float32),
            ((_TOKENS,), i32),
            pages,
            *common,
            meta,
        ),
        (_RATIO, SCALE),
    )
    return examples


def _torch_dtype(dtype):
    return {
        np.dtype(np.int32): torch.int32,
        np.dtype(np.float32): torch.float32,
        np.dtype(ml_dtypes.bfloat16): torch.bfloat16,
        np.dtype(ml_dtypes.float8_e4m3fn): torch.float8_e4m3fn,
    }[np.dtype(dtype)]


@pytest.mark.parametrize("name", sorted(ops._DEFINITIONS))
def test_bridge_traces_match_fakes(name):
    function, fake, donated = ops._DEFINITIONS[name]
    arrays, static_args = _bridge_examples()[name]
    # cpu_export_jax_op makes every non-jax.Array parameter static.
    parameters = list(inspect.signature(function).parameters.values())
    tensors = [p for p in parameters if p.annotation is jax.Array]
    assert [p.annotation for p in parameters[len(tensors) :]] == [
        type(s) for s in static_args
    ]
    assert len(tensors) == len(arrays)
    assert all(i < len(tensors) for i in donated)

    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    specs = [jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in arrays]
    with _tpu_info_patch(), tpu_info_for_cpu_export():
        traced = jax.eval_shape(lambda *xs: function(*xs, *static_args), *specs)
    fakes = fake(
        *(torch.empty(s, dtype=_torch_dtype(d), device="meta") for s, d in arrays),
        *static_args,
    )
    traced, fakes = jax.tree.leaves(traced), jax.tree.leaves(fakes)
    assert [(tuple(t.shape), _torch_dtype(t.dtype)) for t in traced] == [
        (tuple(f.shape), f.dtype) for f in fakes
    ]
    for i in donated:
        # A donated pool returns in its own shape so XLA can alias it.
        assert tuple(traced[donated.index(i)].shape) == arrays[i][0]


def test_cos_sin_table_matches_backend_builder():
    theta = _theta(64)
    freqs_cis = torch.polar(torch.ones(theta.shape), torch.from_numpy(theta).float())
    try:
        from sglang_tpu.dsv4.dsv4_memory_pool import get_or_build_cos_sin_cache
    except ImportError as error:
        pytest.skip(f"dsv4 memory pool unavailable: {error}")
    torch.testing.assert_close(
        # FP32 polar against an FP64 reference; the check is the [cos | sin] order.
        get_or_build_cos_sin_cache(freqs_cis),
        _cos_sin(theta),
        atol=1e-5,
        rtol=0,
    )


def test_inverse_rope_table_undoes_rope():
    rng = np.random.default_rng(0)
    theta = _theta()
    positions = rng.integers(0, MAX_POSITION, 17).astype(np.int32)
    table = np.asarray(
        ops._inverse_rope(jnp.asarray(_cos_sin(theta).numpy()), jnp.asarray(positions))
    )
    assert table.shape == (17, 2, 128) and table.dtype == np.float32
    output = rng.normal(size=(17, 4, 512)).astype(np.float32)
    tail = output[..., 384:]
    # The kernel's finish(): rotated = tail * rope[0] + tail[lane ^ 1] * rope[1].
    rotated = (
        tail * table[:, None, 0] + tail[..., np.arange(128) ^ 1] * table[:, None, 1]
    )
    expected = _inverse_rope_reference(output, theta[positions])[..., 384:]
    np.testing.assert_allclose(rotated, expected, atol=1e-5)


def test_swa_metadata_records_and_padding():
    rng = np.random.default_rng(1)
    lengths = [5, 0, 70]
    positions = np.zeros(_TOKENS, np.int32)
    positions[:5] = np.arange(60, 65)
    positions[5 : 75 - 40 + 5] = 0
    lengths = [5, 0, 30]
    positions[5:35] = np.arange(100, 130)
    starts = np.array([0, 5, 5, 35], np.int32)
    table = rng.permutation(np.arange(1, 1 + _BATCH * _COLUMNS)).astype(np.int32)
    table = table.reshape(_BATCH, _COLUMNS)
    meta, bounds, records = map(
        np.asarray,
        ops._swa_metadata(
            jnp.asarray(table),
            jnp.asarray(starts),
            jnp.asarray(positions),
            jnp.zeros(_BATCH, jnp.int32),
        ),
    )
    assert meta.shape == (_BATCH, 2 + ops.swa_metadata_columns(_TOKENS))
    requests = np.repeat(np.arange(_BATCH), lengths)
    real = positions[: starts[-1]]
    expected = table[requests, real // 64] * 64 + real % 64
    np.testing.assert_array_equal(records[: starts[-1]], expected)
    np.testing.assert_array_equal(records[starts[-1] :], -1)


@pytest.mark.parametrize("ratio", [1, 2])
def test_main_records_are_full_loc_over_ratio(ratio):
    """Read addressing (full page id as main page id) equals the publish record
    ``full_loc // r`` of the record's last token."""
    rng = np.random.default_rng(2 + ratio)
    page_records = 256 // ratio
    lengths = np.array([700, 1, 300])
    columns = 4
    full_pages = rng.permutation(np.arange(1, 1 + len(lengths) * columns))
    full_pages = full_pages.reshape(len(lengths), columns).astype(np.int32)
    tokens = len(lengths)
    positions = (lengths - 1).astype(np.int32)
    starts = np.arange(tokens + 1, dtype=np.int32)
    selected = np.full((tokens, 512), -1, np.int32)
    for t, p in enumerate(positions):
        count = (p + 1) // ratio
        ids = rng.permutation(count)[:512]
        selected[t, : len(ids)] = ids
    records, counts = map(
        np.asarray,
        prepare_main_records(
            jnp.asarray(selected),
            jnp.asarray(full_pages),
            jnp.asarray(starts),
            page_size=page_records,
        ),
    )
    for t in range(tokens):
        ids = selected[t, : counts[t]]
        last = ids * ratio + ratio - 1
        full_loc = full_pages[t, last // 256] * 256 + last % 256
        np.testing.assert_array_equal(records[t, : counts[t]], full_loc // ratio)
        np.testing.assert_array_equal(records[t, counts[t] :], -1)
        assert counts[t] == min(512, (positions[t] + 1) // ratio)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
