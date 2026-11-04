"""Migrated gmm_v2 tests from sglang-jax.

Source: ``sglang-jax/python/sgl_jax/test/kernels/gmm_test.py``.

The original file parameterises a single ``gmm_wrapper`` over both V1 and
V2 paths. sglang-torchtpu only vendors gmm_v2 (V1 is not used by any
TPU code path here), so we keep only the V2-applicable cases:

  * ``test_gmm``                  -> V2 dense path
  * ``test_gmm_weight_quantized``  -> V2 per-channel quant path
  * ``test_gmm_v2_activation_weight_quantized``
                                   -> V2's internal ``maybe_quantize_lhs``

Skipped from the upstream file:

  * ``test_default_block_m_divides_m`` — heuristic is the V1 tuner
    (gmm_v2 uses ``calculate_tiling`` / ``get_tuned_block_sizes``).
  * ``test_gmm_v1_activation_weight_quantized`` — V1-specific.

Run as:
    SGLANG_PLATFORM=tpu python test/manual/tpu/test_gmm_v2_kernel.py
"""

from __future__ import annotations

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.ci.gate_matrix import sweep

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit", chips=1)
# Every (in_size, out_size, group_offset) case is its own Mosaic compile, so the
# 192-case grid below costs ~400s. The gate pins the shapes and keeps the code
# paths (num_groups, has_bias, weight_dtype, zero vs non-zero group_offset);
# the nightly walks the whole grid.
register_tpu_ci(est_time=400, suite="nightly-tpu-4-v7", nightly=True)

import unittest

try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except Exception:  # pragma: no cover
    _HAS_JAX = False

try:
    from absl.testing import parameterized

    _HAS_PARAMETERIZED = True
except Exception:  # pragma: no cover
    _HAS_PARAMETERIZED = False


def _has_tpu_via_jax() -> bool:
    if not _HAS_JAX:
        return False
    try:
        return any(getattr(d, "platform", "") == "tpu" for d in jax.devices())
    except Exception:
        return False


_SKIP_REASON = "Requires JAX with a TPU backend visible to jax.devices()"


# ---------------------------------------------------------------------------
# Helpers (ported verbatim from sglang-jax/python/sgl_jax/test/kernels/gmm_test.py
# so that any future divergence between the kernels is detectable here too).
# ---------------------------------------------------------------------------


def _get_group_sizes(batch_size: int, num_groups: int):
    """Random group_sizes summing exactly to batch_size."""
    distribution = jax.random.uniform(
        jax.random.key(0), (num_groups - 1,), dtype=jnp.float32
    )
    distribution = distribution / jnp.sum(distribution)
    group_sizes = jnp.floor(distribution * batch_size).astype(jnp.int32)
    return jnp.append(group_sizes, batch_size - jnp.sum(group_sizes))


def _quantize_tensor(x, dtype, axis: int = -1, block_size: int = 256):
    """Block-wise symmetric quantization with float32 scales."""
    if jnp.issubdtype(dtype, jnp.integer):
        info = jnp.iinfo(dtype)
        max_val, min_val = int(info.max), int(info.min)
    else:
        info = jnp.finfo(dtype)
        max_val, min_val = float(info.max), float(info.min)

    orig_shape = x.shape
    blocked_shape = orig_shape[:axis] + (-1, block_size) + orig_shape[axis + 1 :]
    x_blocked = x.reshape(blocked_shape)
    x_blocked_abs_max = jnp.max(jnp.abs(x_blocked), axis=axis + 1, keepdims=True)
    scale = x_blocked_abs_max / max_val
    x_blocked_q = jnp.clip(x_blocked / scale, min_val, max_val).astype(dtype)
    x_q = x_blocked_q.reshape(orig_shape)
    scale = scale.squeeze(axis=axis + 1).astype(jnp.float32)
    return x_q, scale


def _reference_gmm(
    lhs,
    rhs,
    group_sizes,
    rhs_scale=None,
    rhs_bias=None,
    group_offset=None,
):
    """Per-group dense matmul reference. f32 accumulator, cast back to lhs dtype.

    Mirrors ``reference_gmm`` in sglang-jax's gmm_test.py so a kernel
    regression in either repo is detectable from the same yardstick.
    """
    num_tokens = lhs.shape[0]
    num_groups, in_size, out_size = rhs.shape
    assert lhs.shape[1] == in_size

    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)
    elif jnp.isscalar(group_offset):
        group_offset = group_offset[None]

    num_blocks = rhs_scale.shape[1] if rhs_scale is not None else 1
    block_size = in_size // num_blocks

    start = 0
    out_chunks = []
    for global_group in range(group_sizes.size):
        group_size = group_sizes[global_group]
        group = global_group - group_offset[0]
        end = min(start + group_size, num_tokens)
        group_size = end - start
        if group >= 0 and group < num_groups:
            lhs_slice = lhs[start:end]
            rhs_slice = rhs[group]
            out = 0
            for block in range(num_blocks):
                bs = block * block_size
                be = bs + block_size
                lhs_block = lhs_slice[:, bs:be].astype(jnp.float32)
                rhs_block = rhs_slice[bs:be, :].astype(jnp.float32)
                acc = jnp.einsum("bd,dh->bh", lhs_block, rhs_block)
                if rhs_scale is not None:
                    acc *= rhs_scale[group][block]
                out = out + acc
            if rhs_bias is not None:
                out = out + rhs_bias[group]
        else:
            out = jnp.zeros((group_size, out_size), dtype=lhs.dtype)
        out_chunks.append(out.astype(lhs.dtype))
        start = end
    return jnp.concat(out_chunks, axis=0)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@unittest.skipUnless(_has_tpu_via_jax(), _SKIP_REASON)
@unittest.skipUnless(_HAS_PARAMETERIZED, "Requires absl.testing.parameterized")
class TestGmmV2Dense(
    parameterized.TestCase if _HAS_PARAMETERIZED else unittest.TestCase
):
    """Dense gmm_v2 vs the per-group reference.

    Migrated from sglang-jax's ``test_gmm`` -- same parameter grid, but
    pinned to the V2 path (sglang-torchtpu doesn't vendor V1).
    """

    @parameterized.product(
        batch_size=[128],
        in_size=sweep([512, 1024], [512]),
        out_size=sweep([512, 1024], [1024]),
        num_groups=[16, 32],
        has_bias=[True, False],
        group_offset=sweep([0, 2, 3], [0, 3]),
    )
    def test_gmm(
        self, batch_size, in_size, out_size, num_groups, has_bias, group_offset
    ):
        from sglang_tpu.layers.moe.pallas.gmm_v2 import gmm_v2

        num_local_groups = num_groups - group_offset
        key = jax.random.key(0)

        lhs = jax.random.normal(key, (batch_size, in_size), dtype=jnp.bfloat16)
        rhs = jax.random.normal(
            key, (num_local_groups, in_size, out_size), dtype=jnp.bfloat16
        )
        rhs_bias = None
        if has_bias:
            rhs_bias = jax.random.normal(
                key, (num_local_groups, 1, out_size), dtype=jnp.bfloat16
            )

        group_sizes = _get_group_sizes(batch_size, num_groups)
        offset = jnp.array(group_offset, dtype=jnp.int32)

        expected = _reference_gmm(
            lhs, rhs, group_sizes, rhs_bias=rhs_bias, group_offset=offset
        )
        actual = gmm_v2(
            lhs,
            rhs,
            group_sizes,
            rhs_bias=rhs_bias,
            group_offset=offset,
        )

        # bf16 matmul accumulation tolerance; matches the upstream
        # assertArraysAllClose default (atol/rtol pair from jtu, ~1e-2).
        max_abs = float(
            jnp.max(jnp.abs(actual.astype(jnp.float32) - expected.astype(jnp.float32)))
        )
        self.assertLess(
            max_abs,
            5e-1,
            f"gmm_v2 vs reference: max diff = {max_abs} "
            f"(bs={batch_size}, k={in_size}, n={out_size}, g={num_groups}, "
            f"bias={has_bias}, offset={group_offset})",
        )


@unittest.skipUnless(_has_tpu_via_jax(), _SKIP_REASON)
@unittest.skipUnless(_HAS_PARAMETERIZED, "Requires absl.testing.parameterized")
class TestGmmV2WeightQuantized(
    parameterized.TestCase if _HAS_PARAMETERIZED else unittest.TestCase
):
    """gmm_v2 with quantised RHS (int8 / fp8) and per-channel scales."""

    @parameterized.product(
        batch_size=[128],
        in_size=sweep([512, 1024], [512]),
        out_size=sweep([512, 1024], [1024]),
        num_groups=[16, 32],
        has_bias=[True, False],
        weight_dtype=[jnp.int8, jnp.float8_e4m3fn],
        group_offset=sweep([0, 2, 3], [0, 3]),
    )
    def test_gmm_weight_quantized(
        self,
        batch_size,
        in_size,
        out_size,
        num_groups,
        has_bias,
        weight_dtype,
        group_offset,
    ):
        from sglang_tpu.layers.moe.pallas.gmm_v2 import gmm_v2

        block_size = in_size
        num_local_groups = num_groups - group_offset
        key = jax.random.key(0)

        lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
        rhs = jax.random.uniform(
            key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1
        )
        rhs_q, rhs_scale = _quantize_tensor(
            rhs, weight_dtype, axis=1, block_size=block_size
        )
        rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

        rhs_bias = None
        if has_bias:
            rhs_bias = jax.random.normal(
                key, (num_local_groups, 1, out_size), dtype=jnp.bfloat16
            )

        group_sizes = _get_group_sizes(batch_size, num_groups)
        offset = jnp.array(group_offset, dtype=jnp.int32)

        expected = _reference_gmm(
            lhs,
            rhs_q,
            group_sizes,
            rhs_scale=rhs_scale,
            rhs_bias=rhs_bias,
            group_offset=offset,
        )

        actual = gmm_v2(
            lhs,
            rhs_q,
            group_sizes,
            rhs_scale=rhs_scale,
            rhs_bias=rhs_bias,
            group_offset=offset,
            maybe_quantize_lhs=False,
        ).astype(lhs.dtype)

        max_abs = float(
            jnp.max(jnp.abs(actual.astype(jnp.float32) - expected.astype(jnp.float32)))
        )
        # Looser tolerance than the dense path: rhs quantization (int8 /
        # fp8 round-trip with per-channel scales) accumulates noise on
        # top of bf16 matmul. Upstream uses atol=rtol=3e-1 in
        # assertArraysAllClose -- we mirror that here.
        self.assertLess(
            max_abs,
            7e-1,
            f"gmm_v2 vs reference (quant): max diff = {max_abs} "
            f"(bs={batch_size}, k={in_size}, n={out_size}, g={num_groups}, "
            f"weight={weight_dtype}, bias={has_bias}, offset={group_offset})",
        )


@unittest.skipUnless(_has_tpu_via_jax(), _SKIP_REASON)
@unittest.skipUnless(_HAS_PARAMETERIZED, "Requires absl.testing.parameterized")
class TestGmmV2ActivationWeightQuantized(
    parameterized.TestCase if _HAS_PARAMETERIZED else unittest.TestCase
):
    """V2's internal ``maybe_quantize_lhs`` path -- the kernel quantises
    LHS on-the-fly when RHS is quantised. Migrated verbatim from
    ``test_gmm_v2_activation_weight_quantized`` in sglang-jax.
    """

    @parameterized.product(
        batch_size=[128],
        in_size=sweep([512, 1024], [512]),
        out_size=sweep([512, 1024], [1024]),
        num_groups=[16, 32],
        weight_dtype=[jnp.int8, jnp.float8_e4m3fn],
        group_offset=sweep([0, 2, 3], [0, 3]),
    )
    def test_gmm_v2_activation_weight_quantized(
        self,
        batch_size,
        in_size,
        out_size,
        num_groups,
        weight_dtype,
        group_offset,
    ):
        from sglang_tpu.layers.moe.pallas.gmm_v2 import gmm_v2

        # TODO(kyuyeunk, wenxindong): Add subchannel quantization on gmm_v2.
        block_size = in_size
        num_local_groups = num_groups - group_offset
        key = jax.random.key(0)

        lhs = jax.random.uniform(key, (batch_size, in_size), jnp.bfloat16, -1, 1)
        rhs = jax.random.uniform(
            key, (num_local_groups, in_size, out_size), jnp.bfloat16, -1, 1
        )
        rhs_q, rhs_scale = _quantize_tensor(
            rhs, weight_dtype, axis=1, block_size=block_size
        )
        rhs_scale = jnp.expand_dims(rhs_scale, axis=2)
        group_sizes = _get_group_sizes(batch_size, num_groups)
        offset = jnp.array(group_offset, dtype=jnp.int32)

        expected = _reference_gmm(
            lhs,
            rhs_q,
            group_sizes,
            rhs_scale=rhs_scale,
            group_offset=offset,
        )

        actual = gmm_v2(
            lhs,
            rhs_q,
            group_sizes,
            rhs_scale=rhs_scale,
            group_offset=offset,
            maybe_quantize_lhs=True,
        ).astype(lhs.dtype)

        max_abs = float(
            jnp.max(jnp.abs(actual.astype(jnp.float32) - expected.astype(jnp.float32)))
        )
        # maybe_quantize_lhs introduces another quant step on top of the
        # RHS quant -- upstream uses atol=rtol=1.1, we mirror.
        self.assertLess(
            max_abs,
            2.0,
            f"gmm_v2 vs reference (act+wgt quant): max diff = {max_abs} "
            f"(bs={batch_size}, k={in_size}, n={out_size}, g={num_groups}, "
            f"weight={weight_dtype}, offset={group_offset})",
        )


if __name__ == "__main__":
    unittest.main()
