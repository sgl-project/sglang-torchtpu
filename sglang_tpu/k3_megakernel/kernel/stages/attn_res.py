# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 attention residual ("block residual") as in-kernel value math.

Matches sglang's TPU path (layers/attn_residual.py aggregate_stream_torch +
RMSNorm.forward_native): bf16 normed rows and bf16 scores, f32 softmax over
the valid rows, each ``p_j * row_j`` rounded to bf16 before an f32 sum.
The bank always has ``bank.shape[0]`` rows; rows ``>= nvb`` are masked so the
same code serves every layer of an in-kernel layer loop. All tensors are f32
values holding bf16-representable numbers.
"""

import jax
import jax.numpy as jnp
import msgspec


def bf16(x: jax.Array) -> jax.Array:
    return x.astype(jnp.bfloat16).astype(jnp.float32)


def lane_sum(x: jax.Array, tree: bool = False) -> jax.Array:
    """``[R, H]`` -> ``[R, 1]`` f32 row sums. ``tree``: add the 128-lane column blocks pairwise
    (log depth) before the in-vreg reduction, instead of the compiler's sequential vreg chain;
    a different f32 summation order (not bit-identical to ``jnp.sum``)."""
    if not tree or x.shape[-1] % 128:
        return jnp.sum(x, axis=-1, keepdims=True)
    parts = [x[:, i : i + 128] for i in range(0, x.shape[-1], 128)]
    while len(parts) > 1:
        parts = [parts[i] + parts[i + 1] for i in range(0, len(parts) - 1, 2)] + parts[
            len(parts) & ~1 :
        ]
    return jnp.sum(parts[0], axis=-1, keepdims=True)


def inv_rms(x: jax.Array, eps: float, tree: bool = False) -> jax.Array:
    """``[R, H]`` -> ``[R, 1]`` f32 ``1 / sqrt(mean(x^2) + eps)``."""
    if tree:
        var = lane_sum(x * x, tree=True) / x.shape[-1]
    else:
        var = jnp.mean(x * x, axis=-1, keepdims=True)
    return 1.0 / jnp.sqrt(var + eps)


def rmsnorm(
    x: jax.Array,
    w: jax.Array,
    eps: float,
    inv: jax.Array | None = None,
    tree: bool = False,
) -> jax.Array:
    """``[R, H]`` -> bf16-valued f32: f32 normalize, ``* w`` in f32, one round.
    ``inv``: the rows' ``inv_rms`` when it is already known."""
    if inv is None:
        inv = inv_rms(x, eps, tree)
    return bf16(x * inv * w.astype(jnp.float32))


def aggregate(
    prefix: jax.Array,
    bank: jax.Array,
    nvb,
    proj_w: jax.Array,
    norm_w: jax.Array,
    eps: float,
) -> jax.Array:
    """Mix ``[bank[0..nvb-1], prefix]``; ``prefix`` ``[R, H]``, ``bank`` ``[NB, R, H]``.

    ``nvb`` may be a traced scalar. With ``nvb == 0`` the result is ``prefix``.
    """
    nb = bank.shape[0]
    rows = [bank[j] for j in range(nb)] + [prefix]
    scores = [
        bf16(
            jnp.sum(
                rmsnorm(row, norm_w, eps) * proj_w.astype(jnp.float32),
                axis=-1,
                keepdims=True,
            )
        )
        for row in rows
    ]  # each [R, 1]
    valid = [j < nvb for j in range(nb)] + [True]
    m = scores[-1]
    for j in range(nb):
        m = jnp.where(valid[j], jnp.maximum(m, scores[j]), m)
    exps = [jnp.where(valid[j], jnp.exp(scores[j] - m), 0.0) for j in range(nb + 1)]
    total = exps[0]
    for e in exps[1:]:
        total = total + e
    mixed = None
    for j in range(nb + 1):
        term = jnp.where(valid[j], bf16((exps[j] / total) * rows[j]), 0.0)
        mixed = term if mixed is None else mixed + term
    return bf16(mixed)


class MixOpts(msgspec.Struct, frozen=True):
    """Numeric options of ``aggregate_packed``. ``tree``: log-depth f32 sums (``lane_sum``; one
    sublane reduction across rows) for a shorter dependency chain; not bit-identical.
    ``fused_score``: score a row as ``bf16(inv_rms * sum(row * norm_w * proj_w))``, skipping the
    bf16 round of the normed row, so a row's two reductions run side by side (a relaxation)."""

    tree: bool = False
    fused_score: bool = False


def row_scores(
    x: jax.Array,
    inv: jax.Array,
    proj_w: jax.Array,
    norm_w: jax.Array,
    eps: float,
    opts: MixOpts = MixOpts(),
) -> jax.Array:
    """``[R, H]`` rows with their ``inv_rms`` ``[R, 1]`` -> bf16-valued scores ``[R, 1]``."""
    w = proj_w.astype(jnp.float32)
    if opts.fused_score:
        return bf16(inv * lane_sum(x * (norm_w.astype(jnp.float32) * w), opts.tree))
    return bf16(lane_sum(rmsnorm(x, norm_w, eps, inv) * w, opts.tree))


def aggregate_packed(
    prefix: jax.Array,
    bank: jax.Array,
    bank_inv: jax.Array,
    nvb,
    proj_w: jax.Array,
    norm_w: jax.Array,
    eps: float,
    opts: MixOpts = MixOpts(),
    bank_sc: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """``aggregate`` for one row with the bank as a dense ``[NB, H]`` tile, straight-line
    (no branches, so the scheduler can overlap it with the neighbouring work).

    ``bank_inv`` ``[NB, 1]`` holds each bank row's ``inv_rms`` (cached when the row is
    written; see ``bank_write``). All bank rows are normalized and scored in one pass, or
    ``bank_sc`` gives their ``row_scores`` computed earlier. With the default ``opts`` the f32
    sums across rows keep ``aggregate``'s sequential order with zeros for the rows ``>= nvb``,
    so the result is bit-identical. Returns ``(mixed [1, H], inv_rms(prefix))``.
    """
    nb, tree = bank.shape[0], opts.tree
    prefix_inv = inv_rms(prefix, eps, tree)
    if bank_sc is None:
        bank_sc = row_scores(bank, bank_inv, proj_w, norm_w, eps, opts)  # [NB, 1]
    last = row_scores(prefix, prefix_inv, proj_w, norm_w, eps, opts)  # [1, 1]
    valid = jax.lax.broadcasted_iota(jnp.int32, (nb, 1), 0) < nvb
    m = jnp.maximum(
        last, jnp.max(jnp.where(valid, bank_sc, -jnp.inf), axis=0, keepdims=True)
    )  # exact
    e = jnp.where(valid, jnp.exp(bank_sc - m), 0.0)
    last_e = jnp.exp(last - m)
    if tree:
        total = jnp.sum(e, axis=0, keepdims=True) + last_e
    else:
        total = e[0:1]
        for k in range(1, nb):
            total = total + e[k : k + 1]
        total = total + last_e
    terms = jnp.where(valid, bf16((e / total) * bank), 0.0)
    if tree:
        mixed = jnp.sum(terms, axis=0, keepdims=True)
    else:
        mixed = terms[0:1]
        for k in range(1, nb):
            mixed = mixed + terms[k : k + 1]
    return bf16(mixed + bf16((last_e / total) * prefix)), prefix_inv


def bank_write(bank_ref, inv_ref, row: jax.Array, row_inv: jax.Array, slot) -> None:
    """Stores ``row`` ``[1, H]`` and its ``inv_rms`` into bank slot ``slot`` (traced)."""
    nb = bank_ref.shape[0]
    hit = jax.lax.broadcasted_iota(jnp.int32, (nb, 1), 0) == slot
    bank_ref[...] = jnp.where(hit, row, bank_ref[...])
    inv_ref[...] = jnp.where(hit, row_inv, inv_ref[...])


def aggregate_rows(
    prefix: jax.Array,
    bank: jax.Array,
    bank_inv: jax.Array,
    nvb,
    proj_w: jax.Array,
    norm_w: jax.Array,
    eps: float,
    opts: MixOpts = MixOpts(),
) -> tuple[jax.Array, jax.Array]:
    """``aggregate_packed`` for R rows with one bank per row, slot-major: ``bank`` ``[NB, R, H]``,
    ``bank_inv`` ``[NB, R, 1]``, ``prefix`` ``[R, H]``. Every op runs on full ``[R, H]`` tiles; the
    sums over slots run in slot order (``aggregate``'s order; with ``opts.tree`` a reordering of
    ``aggregate_packed``'s cross-slot tree sum). Returns ``(mixed [R, H], inv_rms(prefix) [R, 1])``."""
    nb, tree = bank.shape[0], opts.tree
    prefix_inv = inv_rms(prefix, eps, tree)
    last = row_scores(prefix, prefix_inv, proj_w, norm_w, eps, opts)  # [R, 1]
    sc = [
        row_scores(bank[j], bank_inv[j], proj_w, norm_w, eps, opts) for j in range(nb)
    ]
    valid = [j < nvb for j in range(nb)]
    m = last
    for j in range(nb):
        m = jnp.where(valid[j], jnp.maximum(m, sc[j]), m)
    e = [jnp.where(valid[j], jnp.exp(sc[j] - m), 0.0) for j in range(nb)]
    last_e = jnp.exp(last - m)
    total = e[0]
    for j in range(1, nb):
        total = total + e[j]
    total = total + last_e
    mixed = None
    for j in range(nb):
        term = jnp.where(valid[j], bf16((e[j] / total) * bank[j]), 0.0)
        mixed = term if mixed is None else mixed + term
    return bf16(mixed + bf16((last_e / total) * prefix)), prefix_inv
