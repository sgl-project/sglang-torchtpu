# SPDX-License-Identifier: Apache-2.0
"""The jitted device loop: ``n`` B1 greedy steps inside one shard_map program.

Per step (every rank): embedding of the current token (XLA) -> stack kernel
(Pallas, states aliased in place) -> vocab-sharded lm_head dot -> one
all-gather of each rank's packed (max, global id) -> greedy token. The KV
length grows by one per step; the MLA layers are NoPE, so no positions.
"""

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sglang_tpu.k3_megakernel.kernel.stack import TOKEN_META, state_inputs
from sglang_tpu.k3_megakernel.mesh import AXIS
from sglang_tpu.k3_megakernel.model_io import embed as _embed
from sglang_tpu.k3_megakernel.model_io import greedy as _greedy


def build_program(
    *,
    mesh,
    kernel,
    spec,
    shapes: dict,
    steps: int,
    forced: bool,
    embed_mode: str,
    vocab: int,
    probe: bool = False,
):
    """Returns ``f(stack_w, vocab_w, consts, states, token, kv_len, forced_tokens)
    -> (states, token, kv_len, tokens [world, steps])`` with the states donated.

    ``token`` / ``kv_len`` are [world] int32 (identical rows); ``kv_len`` counts the
    token being fed. With ``forced``, step i feeds ``forced_tokens[:, i]`` instead of
    the previous greedy token (prefill by stepping). ``probe`` (debug) also returns the last
    step's per-layer outputs [world, L, 1, H] and expert ids [world, L, top_k]."""
    names = kernel.input_names()
    state_names = state_inputs(spec)
    # fused_head: the kernel embeds the fed token (meta[TOKEN_META]) and returns the greedy token (output 3).
    fused = kernel.fused_head
    nf = kernel.n_fixed_outputs
    if fused and embed_mode != "full":
        raise ValueError("fused_head needs --embed full")
    call, _ = kernel.pallas_call(shapes, alias_states=True)
    world = mesh.devices.size
    vocab_shard = vocab // world

    def local(stack_w, vocab_w, consts, states, token, kv_len, forced_tokens):
        sw = {k: v[0] for k, v in stack_w.items()}
        table, head = vocab_w["embed"][0], vocab_w["lm_head"][0]
        rank = jax.lax.axis_index(AXIS)

        def step(i, carry):
            st, tok, kv, out, dbg = carry
            if forced:
                tok = forced_tokens[0, i]
            ins = dict(sw, tables=consts["tables"][0], pages=consts["pages"][0], **st)
            if fused:
                meta = [kv] + [jnp.int32(0)] * (TOKEN_META - 1) + [tok]
                ins.update(embed=table, lm_head=head, meta=jnp.stack(meta))
            else:
                ins.update(
                    x0=_embed(
                        table, tok, mode=embed_mode, vocab_shard=vocab_shard, rank=rank
                    ),
                    meta=jnp.stack([kv, jnp.int32(0)]),
                )
            outs = call(*[ins[n] for n in names])
            st = dict(zip(state_names, outs[nf:]))
            nxt = (
                outs[3][0]
                if fused
                else _greedy(outs[0], head, vocab_shard=vocab_shard, rank=rank)
            )
            return (
                st,
                nxt,
                kv + 1,
                out.at[i].set(nxt),
                (outs[1], outs[2]) if probe else dbg,
            )

        st = {k: v[0] for k, v in states.items()}
        rows = spec.num_layers if getattr(kernel, "layer_out", True) else 1
        dbg = (
            jnp.zeros((rows, 1, 7168), jnp.bfloat16),
            jnp.zeros((spec.num_layers, 16), jnp.int32),
        )
        carry = (st, token[0], kv_len[0], jnp.zeros((steps,), jnp.int32), dbg)
        st, tok, kv, out, dbg = jax.lax.fori_loop(0, steps, step, carry)
        res = ({k: v[None] for k, v in st.items()}, tok[None], kv[None], out[None])
        return res + ((dbg[0][None], dbg[1][None]) if probe else ())

    ps = P(AXIS)
    sharded = jax.shard_map(
        local, mesh=mesh, in_specs=ps, out_specs=ps, check_vma=False
    )
    sh = NamedSharding(mesh, ps)
    return jax.jit(sharded, donate_argnums=(3,), out_shardings=sh)
