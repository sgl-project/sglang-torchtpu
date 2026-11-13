# SPDX-License-Identifier: Apache-2.0
"""Torch op for the DeepSeek-V4.1 Engram n-gram hash without 64-bit integers.

``sglang.srt.layers.engram.compute_engram_hash_ids`` multiplies mapped token
ids by 63-bit multipliers in int64, then XORs and reduces them modulo primes.
TPU lowering rejects 64-bit integer vector values, so this computes the same
integers on uint32 pairs (adapted from vllm-torchtpu epic/dsv41
``kernels/deepseek_v41/engram_hash.py``): products from 15-bit multiplier
limbs, the rolling XOR on both halves, and the remainder one byte at a time.
Exact while mapped ids are below ``MAX_SOURCE``, multipliers below 2**60 and
primes below ``MAX_PRIME``.

The int64 buffers cross the Torch boundary as int32 word pairs
(``view(torch.int32)``, a bitcast), so no 64-bit arithmetic runs on TPU.
"""

# PEP 563 must stay off so cpu_export_jax_op sees real jax.Array annotations.

from threading import Lock

import jax
import jax.numpy as jnp
import torch

LIMB_BITS = 15
LIMBS = 4
MAX_SOURCE = (2**32 - 1) // (2**LIMB_BITS - 1)
MAX_PRIME = 2**24

_OP_NAME = "sglang_tpu_pallas::dsv41_engram_hash"
_OP = None
_OP_LOCK = Lock()


def _limbs(words: jax.Array) -> jax.Array:
    """uint32 [..., 2] (low, high) words of a value below 2**60 -> [..., LIMBS]."""
    low, high = words[..., 0], words[..., 1]
    mask = (1 << LIMB_BITS) - 1
    return jnp.stack(
        (
            low & mask,
            (low >> LIMB_BITS) & mask,
            ((low >> (2 * LIMB_BITS)) | (high << (32 - 2 * LIMB_BITS))) & mask,
            (high >> (3 * LIMB_BITS - 32)) & mask,
        ),
        axis=-1,
    )


def _product(value: jax.Array, limbs: jax.Array):
    """64-bit ``value * multiplier`` as (high, low) uint32."""
    high = jnp.zeros_like(value)
    low = jnp.zeros_like(value)
    for i in range(LIMBS):
        part = value * limbs[..., i]
        shift = LIMB_BITS * i
        if shift == 0:
            part_high, part_low = jnp.zeros_like(part), part
        elif shift < 32:
            part_high, part_low = part >> (32 - shift), part << shift
        else:
            part_high, part_low = part << (shift - 32), jnp.zeros_like(part)
        total = low + part_low
        high = high + part_high + (total < low).astype(jnp.uint32)
        low = total
    return high, low


def _remainder(high: jax.Array, low: jax.Array, prime: jax.Array) -> jax.Array:
    """(high * 2**32 + low) mod prime, prime < 2**24."""
    rest = jnp.zeros(jnp.broadcast_shapes(high.shape, prime.shape), jnp.uint32)
    for word in (high, low):
        for shift in (24, 16, 8, 0):
            rest = (rest * 256 + ((word >> shift) & 0xFF)) % prime
    return rest


def _engram_hash(
    tokens: jax.Array,
    blocked: jax.Array,
    token_map: jax.Array,
    multipliers: jax.Array,
    primes: jax.Array,
    offsets: jax.Array,
    pad_id: int,
) -> jax.Array:
    """int32 [T, L, (n - 1) * H] equal to ``compute_engram_hash_ids``.

    tokens int32 [T, n]; blocked bool [T, n]; token_map int32 [V];
    multipliers int32 [L, n, 2] (low, high) words; primes int32 [L, n - 1, H];
    offsets int32 [L, (n - 1) * H].
    """
    count, ngram = tokens.shape
    layers = multipliers.shape[0]
    mapped = jnp.take(token_map, tokens, mode="clip")
    source = jnp.where(blocked, pad_id, mapped).astype(jnp.uint32)
    limbs = _limbs(multipliers.astype(jnp.uint32))  # [L, n, LIMBS]
    # Every (layer, shift) at once, tokens on lanes: [L, n, T].
    high, low = _product(source.T[None], limbs[:, :, None, :])
    rolling_high, rolling_low = high[:, 0], low[:, 0]
    highs, lows = [], []
    for shift in range(1, ngram):
        rolling_high = rolling_high ^ high[:, shift]
        rolling_low = rolling_low ^ low[:, shift]
        highs.append(rolling_high)
        lows.append(rolling_low)
    # [L, n-1, 1, T] against primes [L, n-1, H, 1] -> [L, n-1, H, T].
    rest = _remainder(
        jnp.stack(highs, axis=1)[:, :, None],
        jnp.stack(lows, axis=1)[:, :, None],
        primes.astype(jnp.uint32)[..., None],
    )
    hashed = rest.reshape(-1, count).T.reshape(count, layers, -1)
    return hashed.astype(jnp.int32) + offsets[None]


def _fake_engram_hash(tokens, blocked, token_map, multipliers, primes, offsets, pad_id):
    return tokens.new_empty(
        (tokens.shape[0], multipliers.shape[0], primes.shape[1] * primes.shape[2])
    )


def _op():
    global _OP
    if _OP is not None:
        return _OP
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    with _OP_LOCK:
        if _OP is None:
            op = cpu_export_jax_op(_OP_NAME, _engram_hash)
            op.register_fake(_fake_engram_hash)
            _OP = op
    return _OP


def register_engram_ops() -> None:
    """Register the op; call before torch.compile traces the model."""
    _op()


# int32 tables of the process's one EngramHasher, built on the host at load
# so no 64-bit value is converted on the TPU.
_TABLES = None


def prepare_hash_tables(hasher) -> None:
    """Check the exact range and stage int32 copies of the hash tables."""
    token_map, multipliers = hasher.token_map.cpu(), hasher.multipliers.cpu()
    primes, offsets = hasher.primes.cpu(), hasher.offsets.cpu()
    if (
        int(token_map.max()) >= MAX_SOURCE
        or not 0 <= int(hasher.pad_id) < MAX_SOURCE
        or int(multipliers.min()) < 0
        or int(multipliers.max()) >= 2 ** (LIMB_BITS * LIMBS)
        or int(primes.max()) >= MAX_PRIME
        or int(offsets.max()) + int(primes.max()) >= 2**31
    ):
        raise ValueError("Engram tables exceed the 32-bit TPU hash's exact range")
    layers, ngram = multipliers.shape
    device = hasher.token_map.device
    global _TABLES
    _TABLES = tuple(
        t.to(device)
        for t in (
            token_map.to(torch.int32),
            multipliers.contiguous().view(torch.int32).view(layers, ngram, 2),
            primes.to(torch.int32),
            offsets.to(torch.int32),
        )
    )


def engram_hash_ids(tokens: torch.Tensor, blocked: torch.Tensor, pad_id: int):
    """``compute_engram_hash_ids`` in 32-bit arithmetic; int32 row ids."""
    assert _TABLES is not None, "prepare_hash_tables was not called"
    return _op()(tokens.to(torch.int32), blocked, *_TABLES, int(pad_id))
