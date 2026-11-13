# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: src/vllm_torchtpu/kernels/experimental/decode_preparation.py
# Revision: cbf592e2be66ebbd5c33be8ff84e2a6a2314ea15 (epic/dsv41)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission policy for fused decode-preparation regions.

Only the subset the index-selection kernels use is carried: the source's AUTO
policy (fuse measured buckets, otherwise fall back and log once) and the
indexer write bucket. The source's OFF/REQUIRE modes and the other regions'
buckets belong to model-layer fusions not ported here.
"""

import logging

import jax

logger = logging.getLogger(__name__)

# Token counts whose fused FP8 index-key write was measured on TPU v7.
INDEXER_TOKENS = frozenset({128, 256, 288, 384, 512})

_reported: set[tuple[str, str]] = set()


def admit(
    region: str,
    problem: str | None = None,
    *,
    tokens: int | None = None,
    buckets: frozenset | None = None,
) -> bool:
    """Whether ``region`` takes its fused path.

    ``problem`` names a violated layout contract. ``tokens`` must be one of
    ``buckets``; counts above the largest bucket are prefill-sized and fall
    back silently. Non-TPU backends always take the native path.
    """
    if jax.default_backend() != "tpu":
        return False
    if problem is None and buckets is not None and tokens not in buckets:
        if tokens is not None and tokens > max(buckets):
            return False
        problem = f"{tokens} is not a measured bucket {sorted(buckets)}"
    if problem is None:
        return True
    if (region, problem) not in _reported:
        _reported.add((region, problem))
        logger.warning("DSv4.1 decode fusion '%s' falls back: %s", region, problem)
    return False


def first_problem(*checks: tuple[bool, str]) -> str | None:
    """Description of the first failed ``(ok, description)`` layout check."""
    return next((description for ok, description in checks if not ok), None)
