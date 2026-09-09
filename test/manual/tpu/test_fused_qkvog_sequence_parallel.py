# SPDX-License-Identifier: Apache-2.0

"""TP16 correctness tests for the fused SP attention-entry kernel."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sglang.srt.hardware_backend.tpu.kernels.sequence_parallel.fused_rmsnorm_qkvog_mm_a2a import (
    fused_rmsnorm_qkvog_mm_a2a,
    jax_rmsnorm_qkvog_mm_a2a,
)

_TP_SIZE = 16
_HIDDEN_SIZE = 8192
_HEAD_DIM = 64
_TOKENS = (1024, 2048, 4096, 8192, 16384)
_EPS = 1e-5
_EPS_QK = 1e-5
_QUERY_SCALE = 43.7840518911 / (_HEAD_DIM**0.5)
_BF16_TOLERANCE = (2e-2, 1e-1)
_FP8_TOLERANCE = (0.15, 0.5)
_NONIDENTITY_PHYSICAL_TO_LOGICAL = (
    8,
    9,
    14,
    15,
    10,
    11,
    12,
    13,
    0,
    1,
    6,
    7,
    2,
    3,
    4,
    5,
)


def make_hypercube_tp_mesh(axis_name):
    devices = sorted(
        jax.devices(),
        key=lambda device: (
            tuple(device.coords),
            int(device.core_on_chip),
            int(device.id),
        ),
    )
    return jax.sharding.Mesh(
        np.asarray(devices),
        (axis_name,),
        axis_types=(jax.sharding.AxisType.Explicit,),
    )


@pytest.fixture(scope="module")
def tp16_mesh():
    devices = jax.devices()
    if len(devices) != _TP_SIZE or not all(
        device.platform == "tpu" for device in devices
    ):
        pytest.skip("requires one TP16 TPU slice")
    return make_hypercube_tp_mesh("tp")


def _make_weight_and_gain(mesh):
    weight_sharding = NamedSharding(mesh, P(None, None))
    gain_sharding = NamedSharding(mesh, P(None))

    @functools.partial(
        jax.jit,
        out_shardings=(weight_sharding, gain_sharding),
    )
    def make_operands():
        weight = jax.random.normal(
            jax.random.key(0),
            (_HIDDEN_SIZE, _TP_SIZE * 1152),
            dtype=jnp.bfloat16,
        ) / jnp.sqrt(jnp.float32(_HIDDEN_SIZE))
        gain = jnp.asarray(0.75, jnp.bfloat16) + (
            jnp.arange(_HIDDEN_SIZE, dtype=jnp.bfloat16) % 19
        ) / jnp.asarray(64, jnp.bfloat16)
        return weight.astype(jnp.bfloat16), gain

    operands = make_operands()
    jax.block_until_ready(operands)
    return operands


@pytest.fixture(scope="module")
def weight_and_gain(tp16_mesh):
    return _make_weight_and_gain(tp16_mesh)


def _make_inputs(mesh, tokens, *, replicated=False, x_dtype=jnp.bfloat16):
    token_sharding = NamedSharding(mesh, P(None, None) if replicated else P("tp", None))

    @functools.partial(
        jax.jit,
        out_shardings=(token_sharding, token_sharding, token_sharding),
    )
    def make_inputs():
        x = jax.random.normal(
            jax.random.fold_in(jax.random.key(1), tokens),
            (tokens, _HIDDEN_SIZE),
            dtype=jnp.bfloat16,
        ).astype(x_dtype)
        angles = jax.random.uniform(
            jax.random.fold_in(jax.random.key(2), tokens),
            (tokens, _HEAD_DIM // 2),
            minval=-4.0,
            maxval=4.0,
        )
        return (
            x,
            jnp.repeat(jnp.cos(angles), 2, axis=-1),
            jnp.repeat(jnp.sin(angles), 2, axis=-1),
        )

    inputs = make_inputs()
    jax.block_until_ready(inputs)
    return inputs


def test_narrow_sp_replicated_input_matches_token_sharded_input(
    tp16_mesh,
):
    """Narrow SP can enter through rank-local slices of its replicated input."""
    # The runtime bridge uses Auto mesh axes so GSPMD can satisfy the local-slice
    # constraint. The distributed-kernel tests use Explicit axes to pin their
    # input layouts, so rebuild only this integration case with that
    # mesh type while preserving the same topology-aware device order.
    auto_mesh = jax.sharding.Mesh(
        np.asarray(tp16_mesh.devices, dtype=object),
        ("tp",),
        axis_types=(jax.sharding.AxisType.Auto,),
    )
    tokens = 1024
    weight, gain = _make_weight_and_gain(auto_mesh)
    replicated = _make_inputs(auto_mesh, tokens, replicated=True)
    token_sharded = _make_inputs(auto_mesh, tokens)
    token_sharding = NamedSharding(auto_mesh, P("tp", None))
    kwargs = {
        "mesh": auto_mesh,
        "axis_name": "tp",
        "eps": _EPS,
        "eps_qk": _EPS_QK,
        "scale_query_by": _QUERY_SCALE,
        "use_qk_norm": True,
        "use_rope": True,
        "kv_dtype": jnp.float8_e4m3fn,
    }

    @jax.jit
    def run_from_replicated(x, cos, sin):
        x = jax.lax.with_sharding_constraint(x, token_sharding)
        cos = jax.lax.with_sharding_constraint(cos, token_sharding)
        sin = jax.lax.with_sharding_constraint(sin, token_sharding)
        return fused_rmsnorm_qkvog_mm_a2a(x, weight, gain, cos, sin, **kwargs)

    @jax.jit
    def run_from_token_shards(x, cos, sin):
        return fused_rmsnorm_qkvog_mm_a2a(x, weight, gain, cos, sin, **kwargs)

    actual = run_from_replicated(*replicated)
    expected = run_from_token_shards(*token_sharded)
    jax.block_until_ready((actual, expected))

    for actual_part, expected_part in zip(actual, expected, strict=True):
        _assert_addressable_shards_equal(actual_part, expected_part)


def _index_key(shard):
    return tuple(
        (part.start or 0, part.stop or 0, part.step or 1) for part in shard.index
    )


def _paired_addressable_shards(actual, expected):
    actual_shards = sorted(actual.addressable_shards, key=_index_key)
    expected_shards = sorted(expected.addressable_shards, key=_index_key)
    assert [shard.index for shard in actual_shards] == [
        shard.index for shard in expected_shards
    ]
    return zip(actual_shards, expected_shards, strict=True)


def _assert_addressable_shards_equal(actual, expected):
    for actual_shard, expected_shard in _paired_addressable_shards(actual, expected):
        np.testing.assert_array_equal(
            np.asarray(actual_shard.data),
            np.asarray(expected_shard.data),
        )


def _assert_addressable_shards_close(actual, expected):
    rtol, atol = (
        _FP8_TOLERANCE if actual.dtype == jnp.float8_e4m3fn else _BF16_TOLERANCE
    )
    for actual_shard, expected_shard in _paired_addressable_shards(actual, expected):
        np.testing.assert_allclose(
            np.asarray(actual_shard.data, dtype=np.float32),
            np.asarray(expected_shard.data, dtype=np.float32),
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize("kv_dtype", (jnp.bfloat16, jnp.float8_e4m3fn))
@pytest.mark.parametrize("tokens", _TOKENS)
def test_matches_jax_reference_and_is_deterministic(
    tp16_mesh,
    weight_and_gain,
    tokens,
    kv_dtype,
):
    weight, gain = weight_and_gain
    x, cos, sin = _make_inputs(tp16_mesh, tokens)
    kwargs = {
        "mesh": tp16_mesh,
        "axis_name": "tp",
        "eps": _EPS,
        "eps_qk": _EPS_QK,
        "scale_query_by": _QUERY_SCALE,
        "use_qk_norm": True,
        "use_rope": True,
        "kv_dtype": kv_dtype,
    }
    candidate_fn = jax.jit(functools.partial(fused_rmsnorm_qkvog_mm_a2a, **kwargs))
    reference_fn = jax.jit(functools.partial(jax_rmsnorm_qkvog_mm_a2a, **kwargs))

    actual = candidate_fn(x, weight, gain, cos, sin)
    repeated = candidate_fn(x, weight, gain, cos, sin)
    expected = reference_fn(x, weight, gain, cos, sin)
    jax.block_until_ready((actual, repeated, expected))

    expected_shapes = (
        (tokens, 128, _HEAD_DIM),
        (tokens, 16, _HEAD_DIM),
        (tokens, 16, _HEAD_DIM),
        (tokens, 128, _HEAD_DIM),
    )
    assert tuple(part.shape for part in actual) == expected_shapes
    assert tuple(part.dtype for part in actual) == (
        jnp.bfloat16,
        kv_dtype,
        kv_dtype,
        jnp.bfloat16,
    )
    for actual_part, repeated_part, expected_part in zip(
        actual, repeated, expected, strict=True
    ):
        _assert_addressable_shards_equal(actual_part, repeated_part)
        _assert_addressable_shards_close(actual_part, expected_part)


@pytest.mark.parametrize("tokens", (1024, 4096))
def test_schedule_owned_collective_channel_is_reuse_safe(
    tp16_mesh,
    weight_and_gain,
    tokens,
):
    """Reuse one hybrid/direct executable enough times to expose rank runahead."""
    weight, gain = weight_and_gain
    x, cos, sin = _make_inputs(tp16_mesh, tokens)
    kwargs = {
        "mesh": tp16_mesh,
        "axis_name": "tp",
        "eps": _EPS,
        "eps_qk": _EPS_QK,
        "scale_query_by": _QUERY_SCALE,
        "use_qk_norm": True,
        "use_rope": True,
        "kv_dtype": jnp.bfloat16,
    }
    candidate_fn = jax.jit(functools.partial(fused_rmsnorm_qkvog_mm_a2a, **kwargs))
    reference_fn = jax.jit(functools.partial(jax_rmsnorm_qkvog_mm_a2a, **kwargs))
    expected = reference_fn(x, weight, gain, cos, sin)
    jax.block_until_ready(expected)

    # Eight sequential launches match the established distributed-kernel reuse
    # regression pattern. The first launch compiles; every later launch reuses
    # both the executable and its schedule-owned global barrier channel.
    for _ in range(8):
        actual = candidate_fn(x, weight, gain, cos, sin)
        jax.block_until_ready(actual)
        for actual_part, expected_part in zip(actual, expected, strict=True):
            _assert_addressable_shards_close(actual_part, expected_part)


def test_fp32_input_norm_matches_jax_reference(tp16_mesh, weight_and_gain):
    """Preserve FP32 input math until the BF16 projection boundary."""
    tokens = 1024
    weight, gain = weight_and_gain
    x, cos, sin = _make_inputs(tp16_mesh, tokens, x_dtype=jnp.float32)
    kwargs = {
        "mesh": tp16_mesh,
        "axis_name": "tp",
        "eps": _EPS,
        "eps_qk": _EPS_QK,
        "scale_query_by": _QUERY_SCALE,
        "use_qk_norm": True,
        "use_rope": True,
        "kv_dtype": jnp.bfloat16,
    }
    candidate = jax.jit(functools.partial(fused_rmsnorm_qkvog_mm_a2a, **kwargs))(
        x, weight, gain, cos, sin
    )
    expected = jax.jit(functools.partial(jax_rmsnorm_qkvog_mm_a2a, **kwargs))(
        x, weight, gain, cos, sin
    )
    jax.block_until_ready((candidate, expected))
    for actual_part, expected_part in zip(candidate, expected, strict=True):
        _assert_addressable_shards_close(actual_part, expected_part)


def test_nonidentity_rank_order_preserves_logical_tokens_and_heads(
    tp16_mesh, weight_and_gain
):
    """Exercise the non-identity Torch rank order used by the serving slice."""
    tokens = 1024
    local_rows = tokens // _TP_SIZE
    p2l = jnp.asarray(_NONIDENTITY_PHYSICAL_TO_LOGICAL, dtype=jnp.int32)
    weight, gain = weight_and_gain
    x, cos, sin = _make_inputs(tp16_mesh, tokens, x_dtype=jnp.float32)
    token_sharding = NamedSharding(tp16_mesh, P("tp", None))
    weight_sharding = NamedSharding(tp16_mesh, P(None, None))
    metadata_sharding = NamedSharding(tp16_mesh, P("tp"))

    @functools.partial(
        jax.jit,
        out_shardings=(token_sharding, token_sharding, token_sharding),
    )
    def physical_order_inputs(x, cos, sin):
        return tuple(
            value.reshape(_TP_SIZE, local_rows, -1)[p2l].reshape(value.shape)
            for value in (x, cos, sin)
        )

    @functools.partial(jax.jit, out_shardings=weight_sharding)
    def physical_order_weight(weight):
        return weight.reshape(_HIDDEN_SIZE, _TP_SIZE, 1152)[:, p2l, :].reshape(
            weight.shape
        )

    metadata_rows = np.asarray(
        [
            [physical, logical, *_NONIDENTITY_PHYSICAL_TO_LOGICAL]
            for physical, logical in enumerate(_NONIDENTITY_PHYSICAL_TO_LOGICAL)
        ],
        dtype=np.int32,
    )
    metadata = jax.device_put(metadata_rows.reshape(-1), metadata_sharding)
    physical_x, physical_cos, physical_sin = physical_order_inputs(x, cos, sin)
    physical_weight = physical_order_weight(weight)
    kwargs = {
        "mesh": tp16_mesh,
        "axis_name": "tp",
        "eps": _EPS,
        "eps_qk": _EPS_QK,
        "scale_query_by": _QUERY_SCALE,
        "use_qk_norm": True,
        "use_rope": True,
        "kv_dtype": jnp.bfloat16,
    }
    candidate = jax.jit(functools.partial(fused_rmsnorm_qkvog_mm_a2a, **kwargs))(
        physical_x,
        physical_weight,
        gain,
        physical_cos,
        physical_sin,
        rank_metadata=metadata,
    )
    reference = jax.jit(functools.partial(jax_rmsnorm_qkvog_mm_a2a, **kwargs))(
        x, weight, gain, cos, sin
    )

    @functools.partial(
        jax.jit,
        out_shardings=NamedSharding(tp16_mesh, P(None, "tp", None)),
    )
    def physical_order_heads(value):
        heads_per_rank = value.shape[1] // _TP_SIZE
        return value.reshape(tokens, _TP_SIZE, heads_per_rank, _HEAD_DIM)[
            :, p2l, :, :
        ].reshape(value.shape)

    expected = [physical_order_heads(value) for value in reference]

    jax.block_until_ready((candidate, expected))
    for actual_part, expected_part in zip(candidate, expected, strict=True):
        _assert_addressable_shards_close(actual_part, expected_part)
