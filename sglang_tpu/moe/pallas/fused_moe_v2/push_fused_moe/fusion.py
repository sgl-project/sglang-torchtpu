# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: src/vllm_torchtpu/kernels/experimental/decode_preparation.py
# SPDX-License-Identifier: Apache-2.0
"""Admission and Pallas helpers for optional push routing fusion."""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu._logging import get_logger
from sglang_tpu.pallas_export import get_tpu_generation

logger = get_logger(__name__)

# Per-shard token counts whose fused routing metadata was lowered and checked
# against the native tables on v7.
ROUTING_TOKENS = frozenset({96, 192, 256, 288, 384})
MEASURED_TPU_VERSION = 7

_reported: set[tuple[str, str]] = set()


def admit(region: str, problem: str | None, *, tokens: int, buckets: frozenset) -> bool:
    """Use fusion for a supported TPU target, layout and measured token bucket."""
    if get_tpu_generation() < 0:
        return False
    if problem is None and tokens not in buckets:
        if tokens > max(buckets):
            return False
        problem = f"{tokens} is not a measured bucket {sorted(buckets)}"
    if problem is None:
        return True
    if (region, problem) not in _reported:
        _reported.add((region, problem))
        logger.warning("push fused EP MoE '%s' falls back: %s", region, problem)
    return False


def first_problem(*checks: tuple[bool, str]) -> str | None:
    """Description of the first failed ``(ok, description)`` layout check."""
    return next((description for ok, description in checks if not ok), None)


def tpu_version_problem(version: int = MEASURED_TPU_VERSION) -> str | None:
    """Buckets were measured on one TPU generation; others keep native paths."""
    actual = get_tpu_generation()
    if actual < 0:
        return None  # `admit` rejects an untargeted CPU/GPU silently.
    return None if actual == version else f"TPU v{actual} (measured on v{version})"


def prefix_sum(a, axis=None, dtype=None, out=None):
    """Integer-only inclusive scan with explicit shift/add dependencies."""
    assert out is None
    if axis is None:
        a, axis = a.reshape(-1), 0
    if dtype is not None:
        a = a.astype(dtype)
    assert jnp.issubdtype(a.dtype, jnp.integer)
    distance = 1
    while distance < a.shape[axis]:
        tail = lax.slice_in_dim(a, 0, a.shape[axis] - distance, axis=axis)
        pad_shape = list(a.shape)
        pad_shape[axis] = distance
        a = a + jnp.concatenate((jnp.zeros(pad_shape, a.dtype), tail), axis=axis)
        distance *= 2
    return a


def fuse_function(fn, args, *, name, interpret=False):
    """Real Pallas boundary, with the same externally visible array outputs."""
    out_shape = jax.eval_shape(fn, *args)
    leaves, tree = jax.tree.flatten(out_shape)

    def kernel(*refs):
        outputs = fn(*(r[...] for r in refs[: len(args)]))
        for ref, value in zip(refs[len(args) :], jax.tree.leaves(outputs)):
            ref[...] = value

    call = pl.pallas_call(
        kernel,
        out_shape=tuple(leaves),
        name=name,
        interpret=interpret,
        compiler_params=None if interpret else _half_vmem(),
    )
    return lambda *xs: jax.tree.unflatten(tree, call(*xs))


def _half_vmem() -> pltpu.CompilerParams:
    return pltpu.CompilerParams(
        vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes // 2
    )
