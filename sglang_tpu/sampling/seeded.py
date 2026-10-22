# SPDX-License-Identifier: Apache-2.0
"""Stateless TPU sampling keyed by request seed, absolute position and column."""

import torch

from sglang.srt.compilation.compiled_op import compiled_op
from sglang.srt.speculative.spec_common import seeded_verify_coins

_MASK32 = 0xFFFFFFFF

# Verify coins hash columns [0, draft_token_num] at the committed length and the
# target sampler column 0 at its position. A draft drawn in one of those columns
# at a later position would reuse the coin a following verify step reads there,
# correlating acceptance with the rejected draft and biasing the output.
DRAFT_COLUMN = _MASK32


def _mul32(value, constant):
    # Keep every intermediate below signed-int64 overflow.
    low = (value & 0xFFFF) * constant
    high = (((value >> 16) * constant) & 0xFFFF) << 16
    return (low + high) & _MASK32


def _rotate32(value, bits):
    return ((value << bits) | (value >> (32 - bits))) & _MASK32


def _mix32(state, block):
    block = _mul32(block & _MASK32, 0xCC9E2D51)
    block = _mul32(_rotate32(block, 15), 0x1B873593)
    state = _rotate32(state ^ block, 13)
    return (state * 5 + 0xE6546B64) & _MASK32


def murmur_hash32(seed, positions, columns):
    """The shared four-block Murmur3 hash, represented in positive int64 lanes."""
    assert seed.ndim == positions.ndim == columns.ndim == 1
    assert seed.shape == positions.shape
    seed = seed.to(torch.int64)[:, None]
    positions = positions.to(torch.int64)[:, None]
    state = torch.zeros_like(seed)
    state = _mix32(state, seed)
    state = _mix32(state, seed >> 32)
    state = _mix32(state, positions)
    state = _mix32(state, columns.to(torch.int64)[None, :])
    state = state ^ 16
    state = _mul32(state ^ (state >> 16), 0x85EBCA6B)
    state = _mul32(state ^ (state >> 13), 0xC2B2AE35)
    return state ^ (state >> 16)


def hash_to_uniform(hashed):
    """Match float64(hash / UINT32_MAX) -> float32 without TPU float64.

    Integer quotient/remainder retain the 24-bit significand. Values within
    half a float64 ULP of a float32 midpoint first round to that midpoint,
    then to the even significand; simply rounding the rational differs there.
    """
    shifts = torch.arange(32, dtype=torch.int64, device=hashed.device)
    exponent = ((hashed[..., None] >> shifts) != 0).sum(-1).sub(1).clamp_min(0)
    numerator = hashed << (55 - exponent)
    quotient = torch.div(numerator, _MASK32, rounding_mode="floor")
    remainder = numerator - quotient * _MASK32
    delta = 2 * remainder - _MASK32
    significand = quotient + torch.where(delta.abs() <= 7, quotient & 1, delta > 0)
    uniform = significand.float() * torch.pow(2.0, (exponent - 55).float())
    return uniform.clamp_max(1.0 - 2**-24)


def _uniforms(seed, positions, columns):
    return hash_to_uniform(murmur_hash32(seed, positions, columns))


@compiled_op()
def verify_coins(seed, positions, width):
    return seeded_verify_coins(
        sampling_seed=seed,
        seq_lens=positions,
        draft_token_num=width,
        device=positions.device,
        uniform_fn=_uniforms,
    )


def sample_categorical(weights, coins):
    """Inverse CDF at explicit half-open coins, skipping zero-probability bins."""
    cumulative = weights.float().cumsum(-1)
    threshold = coins * cumulative[:, -1]
    return (cumulative <= threshold[:, None]).sum(-1).clamp_max(weights.shape[-1] - 1)


def _sample_at_column(weights, seed, positions, column):
    columns = torch.full((1,), column, dtype=torch.int64, device=positions.device)
    return sample_categorical(weights, _uniforms(seed, positions, columns)[:, 0])


@compiled_op()
def sample_with_seed(weights, seed, positions):
    return _sample_at_column(weights, seed, positions, 0)


@compiled_op()
def sample_draft_with_seed(weights, seed, positions):
    """Seeded draft draws, keyed apart from every verify and target coin."""
    return _sample_at_column(weights, seed, positions, DRAFT_COLUMN)


@compiled_op()
def draft_uniforms(seed, seq_lens, num_drafts):
    """[bs, num_drafts] seeded chain-draft uniforms, keyed as sample_draft_with_seed.

    Draft j (0-based) sits at absolute position seq_lens + j + 1.
    """
    columns = torch.full((1,), DRAFT_COLUMN, dtype=torch.int64, device=seq_lens.device)
    return torch.stack(
        [
            _uniforms(seed, seq_lens + step + 1, columns)[:, 0]
            for step in range(num_drafts)
        ],
        dim=1,
    )
