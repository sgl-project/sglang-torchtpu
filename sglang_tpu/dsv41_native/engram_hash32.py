# SPDX-License-Identifier: Apache-2.0
"""The engram n-gram hash in int32 arithmetic, bit-exact with the int64 reference.

The reference hashes a position as `(xor_k token_k * multiplier_k) % prime`, with 63-bit
multipliers. XLA on TPU emulates 64-bit integers, and its emulation of this arithmetic aborted
the compiler (r6: "Unsupported conversion from vmreg/vreg to U64"). Here every 63-bit value is
held as 9 limbs of 7 bits in int32:

- product: token (< 2**17) times a 7-bit limb is < 2**24; carries are propagated limb by limb.
  The reference bounds multipliers so token * multiplier < 2**63, so 9 limbs hold the product.
- xor: limb-wise, since xor works bit by bit.
- modulo: value % p = (sum_i limb_i * (2**(7 i) % p)) % p. With p < 2**24 each term is
  < 127 * 2**24 < 2**31, and the terms are reduced before they are summed.
"""

from __future__ import annotations

import torch

LIMB_BITS = 7
NUM_LIMBS = 9  # 63 bits
_LIMB_MASK = (1 << LIMB_BITS) - 1
MAX_PRIME = 1 << 24
MAX_TOKEN = 1 << 17


def multiplier_limbs(multipliers: torch.Tensor) -> torch.Tensor:
    """[..., n] int64 multipliers (< 2**63) -> [..., n, NUM_LIMBS] int32, least significant first."""
    # Host-side table setup: pin every tensor to CPU, the model is built under the TPU device.
    m = multipliers.to(torch.int64).cpu()
    assert (m >= 0).all()
    shifts = torch.arange(NUM_LIMBS, dtype=torch.int64, device="cpu") * LIMB_BITS
    return ((m.unsqueeze(-1) >> shifts) & _LIMB_MASK).to(torch.int32)


def limb_weights(primes: torch.Tensor) -> torch.Tensor:
    """[...] primes -> [..., NUM_LIMBS] int32 weights 2**(7 i) % p."""
    p = primes.to(torch.int64).cpu()
    assert (p > 1).all() and (p < MAX_PRIME).all(), "limb modulo needs primes < 2**24"
    w = [torch.full_like(p, 1)]
    for _ in range(NUM_LIMBS - 1):
        w.append((w[-1] << LIMB_BITS) % p)
    return torch.stack(w, dim=-1).to(torch.int32)


def _times_token(tokens: torch.Tensor, m_limbs: torch.Tensor) -> torch.Tensor:
    """tokens [...] int32 (< 2**17) times m_limbs [..., NUM_LIMBS] -> product limbs."""
    c = tokens.unsqueeze(-1) * m_limbs  # each < 2**24
    limbs, carry = [], torch.zeros_like(c[..., 0])
    for i in range(NUM_LIMBS):
        v = c[..., i] + carry
        limbs.append(v & _LIMB_MASK)
        carry = v >> LIMB_BITS
    return torch.stack(limbs, dim=-1)


def _mod(
    limbs: torch.Tensor, weights: torch.Tensor, primes: torch.Tensor
) -> torch.Tensor:
    """limbs [..., L] (a value < 2**63) mod primes [...]; weights [..., L] as limb_weights."""
    terms = (limbs * weights) % primes.unsqueeze(-1)
    return terms.sum(-1) % primes


def engram_hash32(
    tokens: torch.Tensor,
    m_limbs: torch.Tensor,
    primes: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Hash ids per n-gram size and head, before the per-head table offsets.

    tokens: [..., n] int32 compressed ids, newest first (shift 0 .. n - 1).
    m_limbs: [layers, n, NUM_LIMBS] int32, from multiplier_limbs.
    primes: [layers, n - 1, heads] int32. weights: [layers, n - 1, heads, NUM_LIMBS] int32.
    Returns [..., layers, (n - 1) * heads] int32, the reference `rolling % prime` values.
    """
    n = tokens.shape[-1]
    products = _times_token(tokens.unsqueeze(-2), m_limbs)  # [..., layers, n, L]
    rolling, hashes = products[..., 0, :], []
    for i in range(1, n):
        rolling = torch.bitwise_xor(rolling, products[..., i, :])
        hashes.append(
            _mod(rolling.unsqueeze(-2), weights[:, i - 1], primes[:, i - 1])
        )  # [..., layers, heads]
    return torch.cat(hashes, dim=-1)
