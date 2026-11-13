# SPDX-License-Identifier: Apache-2.0
"""Pallas prefill kernel for Inkling2 block attention residuals plus the RMSNorm that follows.

One call reads every residual source once per token tile and returns the normed bf16 inputs
of the next sub-layer. The XLA form reads each source two or three times (inverse RMS, logits,
weighted sum) and materializes the fp32 mixture.
"""

# PEP 563 must remain disabled so `cpu_export_jax_op` sees `jax.Array` type
# objects rather than stringified annotations at registration time.

import functools
from threading import Lock
from typing import Optional

import jax
import jax.numpy as jnp
import torch
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Residual blocks + partial + the attention output of a block-boundary layer.
MAX_SOURCES = 10
_TILE_TOKENS = 64
# Below this many tokens (decode, MTP verify) the XLA path stays.
MIN_TOKENS = 512
_VMEM_LIMIT = 56 * 1024 * 1024


def _kernel(
    *refs,
    n: int,
    nq: int,
    has_addend: bool,
    normalize: bool,
    eps_res: float,
    eps_norm: float,
):
    src_refs = refs[:n]
    pos = n
    add_ref = refs[pos] if has_addend else None
    pos += has_addend
    q_ref, w_ref, out_ref = refs[pos : pos + 3]

    def load(j):
        v = src_refs[j][...].astype(jnp.float32)
        if add_ref is not None and j == n - 1:
            v = v + add_ref[...].astype(jnp.float32)
        return v

    queries = q_ref[...]
    inv, logits = [], [[] for _ in range(nq)]
    for j in range(n):
        v = load(j)
        r = jax.lax.rsqrt(jnp.mean(v * v, axis=-1, keepdims=True) + eps_res)
        inv.append(r)
        for q in range(nq):
            logits[q].append(
                jnp.sum(v * queries[q : q + 1, :], axis=-1, keepdims=True) * r
            )
    probs = []
    for q in range(nq):
        m = functools.reduce(jnp.maximum, logits[q])
        exps = [jnp.exp(lg - m) for lg in logits[q]]
        total = functools.reduce(jnp.add, exps)
        probs.append(
            [e / total * r if normalize else e / total for e, r in zip(exps, inv)]
        )
    accs = [None] * nq
    for j in range(n):
        v = load(j)
        for q in range(nq):
            term = probs[q][j] * v
            accs[q] = term if accs[q] is None else accs[q] + term
    weights = w_ref[...]
    for q in range(nq):
        x = accs[q]
        y = x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps_norm)
        out_ref[q] = (y * weights[q : q + 1, :]).astype(out_ref.dtype)


def attn_res_norm_jax(
    sources: list,
    addend: Optional[jax.Array],
    queries: jax.Array,
    norm_weights: jax.Array,
    *,
    normalize: bool,
    eps_res: float,
    eps_norm: float,
) -> jax.Array:
    """[Q, T, H] bf16: RMSNorm_q(sum_i softmax_i(<s_i, q> * r_i) * (r_i *) s_i) per query row q.

    sources are bf16 [T, H]; addend (bf16 [T, H]) is added to the last source in fp32;
    queries are fp32 [Q, H] already scaled by H ** -0.5; norm_weights fp32 [Q, H].
    """
    tokens, hidden = sources[0].shape
    nq = queries.shape[0]
    n = len(sources)
    has_addend = addend is not None
    tile = _TILE_TOKENS
    row_spec = pl.BlockSpec((tile, hidden), lambda i: (i, 0))
    vec_spec = pl.BlockSpec((nq, hidden), lambda i: (0, 0))
    return pl.pallas_call(
        functools.partial(
            _kernel,
            n=n,
            nq=nq,
            has_addend=has_addend,
            normalize=normalize,
            eps_res=eps_res,
            eps_norm=eps_norm,
        ),
        out_shape=jax.ShapeDtypeStruct((nq, tokens, hidden), jnp.bfloat16),
        grid=(tokens // tile,),
        in_specs=[row_spec] * (n + has_addend) + [vec_spec, vec_spec],
        out_specs=pl.BlockSpec((nq, tile, hidden), lambda i: (0, i, 0)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=_VMEM_LIMIT
        ),
        name=f"inkling2_attn_res_n{n}_q{nq}{'_add' if has_addend else ''}",
    )(*sources, *([addend] if has_addend else []), queries, norm_weights)


_OP = None
_OP_LOCK = Lock()


def register_inkling2_attn_res_op():
    """Register the op before the model is traced; returns it."""
    global _OP
    with _OP_LOCK:
        if _OP is not None:
            return _OP
        # Lazy: the kernel itself must import without torch_tpu (raw-JAX unit test).
        from sglang_tpu.pallas_jax_op import cpu_export_jax_op

        def _op(
            s0: jax.Array,
            s1: Optional[jax.Array],
            s2: Optional[jax.Array],
            s3: Optional[jax.Array],
            s4: Optional[jax.Array],
            s5: Optional[jax.Array],
            s6: Optional[jax.Array],
            s7: Optional[jax.Array],
            s8: Optional[jax.Array],
            s9: Optional[jax.Array],
            addend: Optional[jax.Array],
            queries: jax.Array,
            norm_weights: jax.Array,
            normalize: bool,
            eps_res: float,
            eps_norm: float,
        ) -> jax.Array:
            sources = [
                s for s in (s0, s1, s2, s3, s4, s5, s6, s7, s8, s9) if s is not None
            ]
            return attn_res_norm_jax(
                sources,
                addend,
                queries,
                norm_weights,
                normalize=normalize,
                eps_res=eps_res,
                eps_norm=eps_norm,
            )

        _OP = cpu_export_jax_op("sglang_tpu_pallas::inkling2_attn_res_norm", _op)
        return _OP


def supports_attn_res_norm(tokens: int, num_sources: int) -> bool:
    return (
        tokens >= MIN_TOKENS
        and tokens % _TILE_TOKENS == 0
        and num_sources <= MAX_SOURCES
    )


def attn_res_norm_tpu(
    sources: list[torch.Tensor],
    addend: Optional[torch.Tensor],
    queries: torch.Tensor,
    norm_weights: torch.Tensor,
    *,
    normalize: bool,
    eps_res: float,
    eps_norm: float,
) -> torch.Tensor:
    """Torch entry; see attn_res_norm_jax. Returns [Q, T, H] bf16."""
    op = register_inkling2_attn_res_op()
    slots = list(sources) + [None] * (MAX_SOURCES - len(sources))
    return op(*slots, addend, queries, norm_weights, normalize, eps_res, eps_norm)
