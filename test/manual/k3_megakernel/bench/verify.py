# SPDX-License-Identifier: Apache-2.0
"""Speculative-verify device loops for the anchor benchmark (``StackKernel(verify_rows=R)``).

``build_verify_program``: ``steps`` verify steps of R rows inside one shard_map program, with
the anchor protocol's fixed-acceptance mode: every step keeps ``acc`` rows (``acc`` tokens: the inputs of
rows 1..acc-1 plus the bonus token predicted by row acc-1), advances kv_len by ``acc`` and reads
the KDA state of the next step from checkpoint row ``acc - 1``. The drafts of the next step are
the greedy predictions of rows acc..R-1 (a deterministic self-draft; no draft model).

``build_oracle_program``: the same R forced rows as R B1 steps (the reference for the verify
kernel): final rows, captured layer rows and the KDA state after every row.
"""

import jax
import jax.numpy as jnp
from bench.program import _embed
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sglang_tpu.k3_megakernel.kernel.stack import input_names, state_inputs
from sglang_tpu.k3_megakernel.mesh import AXIS


def capture_layers(num_layers: int) -> tuple[int, ...]:
    """DSpark's target layers [7, 23, 51, 67, 83]; on short stacks the 13-layer remap [0, 3, 6, 9, 12]."""
    full = (7, 23, 51, 67, 83)
    if num_layers > full[-1]:
        return full
    return tuple(l for l in (0, 3, 6, 9, 12) if l < num_layers)


def _verify_meta(kv_len, read_row, rows: int):
    """The shared kernel entry layout [kv_len, KDA read row, KDA write base, n_rows, write stride,
    reset]: one request slot (checkpoint rows 0..R-1), stride 1, no reset."""
    return jnp.stack(
        [kv_len, read_row, jnp.int32(0), jnp.int32(rows), jnp.int32(1), jnp.int32(0)]
    )


def _embed_rows(table, tokens, *, mode, vocab_shard, rank, flat=False):
    """[R, 1, H] (``flat``: [R, H]) bf16 rows of ``tokens`` [R]."""
    rows = [
        _embed(table, tokens[i], mode=mode, vocab_shard=vocab_shard, rank=rank)
        for i in range(tokens.shape[0])
    ]
    return jnp.concatenate(rows) if flat else jnp.stack(rows)


def _greedy_rows(final, head, *, vocab_shard, rank):
    """Greedy ids [R] of ``final`` [R, 1, H] or [R, H] (vocab-sharded head; ties to the lowest id)."""
    final = final.reshape(final.shape[0], final.shape[-1])
    logits = jax.lax.dot_general(
        final, head, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
    )  # [R, V/world]
    local_id = jnp.argmax(logits, axis=1).astype(jnp.int32)
    best = jnp.take_along_axis(logits, local_id[:, None], axis=1)[:, 0]
    packed = jnp.stack(
        [best, (local_id + rank * vocab_shard).astype(jnp.float32)], axis=1
    )  # [R, 2]
    every = jax.lax.all_gather(packed, AXIS)  # [world, R, 2]
    win = jnp.argmax(every[:, :, 0], axis=0)  # [R]
    return jnp.take_along_axis(every[:, :, 1], win[None], axis=0)[0].astype(jnp.int32)


def build_verify_program(
    *,
    mesh,
    kernel,
    spec,
    shapes: dict,
    steps: int,
    acc: int,
    embed_mode: str,
    vocab: int,
):
    """``f(stack_w, vocab_w, consts, states, token, kv_len, read_row, drafts) -> (states, token, kv_len,
    read_row, drafts, tokens [world, steps * acc])``; ``drafts`` is [world, R - 1]."""
    names = input_names(spec)
    state_names = state_inputs(spec)
    call, _ = kernel.pallas_call(shapes, alias_states=True)
    rows = kernel.verify_rows
    assert 1 <= acc <= rows, acc
    vocab_shard = vocab // mesh.devices.size

    def local(stack_w, vocab_w, consts, states, token, kv_len, read_row, drafts):
        sw = {k: v[0] for k, v in stack_w.items()}
        table, head = vocab_w["embed"][0], vocab_w["lm_head"][0]
        rank = jax.lax.axis_index(AXIS)

        def step(i, carry):
            st, tok, kv, rr, dr, out = carry
            toks = jnp.concatenate([tok[None], dr])  # [R]
            x0 = _embed_rows(
                table,
                toks,
                mode=embed_mode,
                vocab_shard=vocab_shard,
                rank=rank,
                flat=kernel.rows_mode,
            )
            ins = dict(
                sw,
                tables=consts["tables"][0],
                pages=consts["pages"][0],
                x0=x0,
                meta=_verify_meta(kv, rr, kernel.verify_rows),
                **st,
            )
            outs = call(*[ins[n] for n in names])
            st = dict(zip(state_names, outs[3:]))
            g = _greedy_rows(outs[0], head, vocab_shard=vocab_shard, rank=rank)  # [R]
            kept = jnp.concatenate([toks[1:acc], g[acc - 1 : acc]])  # acc tokens
            out = jax.lax.dynamic_update_slice(out, kept, (i * acc,))
            nxt = (
                jnp.concatenate([g[acc:], jnp.broadcast_to(g[rows - 1 :], (acc - 1,))])
                if acc > 1
                else g[1:]
            )
            return st, g[acc - 1], kv + acc, jnp.int32(acc - 1), nxt[: rows - 1], out

        st = {k: v[0] for k, v in states.items()}
        carry = (
            st,
            token[0],
            kv_len[0],
            read_row[0],
            drafts[0],
            jnp.zeros((steps * acc,), jnp.int32),
        )
        st, tok, kv, rr, dr, out = jax.lax.fori_loop(0, steps, step, carry)
        return (
            {k: v[None] for k, v in st.items()},
            tok[None],
            kv[None],
            rr[None],
            dr[None],
            out[None],
        )

    ps = P(AXIS)
    sharded = jax.shard_map(
        local, mesh=mesh, in_specs=ps, out_specs=ps, check_vma=False
    )
    return jax.jit(sharded, donate_argnums=(3,), out_shardings=NamedSharding(mesh, ps))


def build_oracle_program(
    *,
    mesh,
    kernel,
    spec,
    shapes: dict,
    rows: int,
    capture: tuple,
    embed_mode: str,
    vocab: int,
):
    """``f(stack_w, vocab_w, consts, states, kv_len, tokens [world, R]) -> (states, final [world, R, H],
    capture [world, C, R, H], kda [world, R, n_kda states...])``: R forced B1 steps (``kernel`` must
    have ``layer_out=True``); ``kda`` holds every KDA pool row 0 after each step."""
    names = input_names(spec)
    state_names = state_inputs(spec)
    kda_names = [n for n in state_names if n.startswith("kda_")]
    call, _ = kernel.pallas_call(shapes, alias_states=True)
    vocab_shard = vocab // mesh.devices.size

    def local(stack_w, vocab_w, consts, states, kv_len, tokens):
        sw = {k: v[0] for k, v in stack_w.items()}
        table = vocab_w["embed"][0]
        rank = jax.lax.axis_index(AXIS)
        st = {k: v[0] for k, v in states.items()}
        finals, caps, kdas, ids = [], [], [], []
        for t in range(rows):
            x0 = _embed(
                table, tokens[0, t], mode=embed_mode, vocab_shard=vocab_shard, rank=rank
            )
            ins = dict(
                sw,
                tables=consts["tables"][0],
                pages=consts["pages"][0],
                x0=x0,
                meta=jnp.stack([kv_len[0] + t, jnp.int32(0)]),
                **st,
            )
            outs = call(*[ins[n] for n in names])
            st = dict(zip(state_names, outs[3:]))
            finals.append(outs[0][0])
            ids.append(outs[2])
            caps.append(jnp.stack([outs[1][l, 0] for l in capture]))
            kdas.append([st[n][0] for n in kda_names])
        kda = [jnp.stack([k[i] for k in kdas]) for i in range(len(kda_names))]
        return (
            {k: v[None] for k, v in st.items()},
            jnp.stack(finals)[None],
            jnp.stack(caps, axis=1)[None],
            tuple(x[None] for x in kda),
            jnp.concatenate(ids, axis=1)[None],
        )

    ps = P(AXIS)
    sharded = jax.shard_map(
        local, mesh=mesh, in_specs=ps, out_specs=ps, check_vma=False
    )
    return jax.jit(sharded, donate_argnums=(3,), out_shardings=NamedSharding(mesh, ps))


def build_verify_once(*, mesh, kernel, spec, shapes: dict, embed_mode: str, vocab: int):
    """``f(stack_w, vocab_w, consts, states, kv_len, read_row, tokens [world, R]) -> (states, final [world, R, H],
    capture [world, C, R, H])``: one verify call (KDA checkpoints written to rows 0..R-1)."""
    names = input_names(spec)
    state_names = state_inputs(spec)
    call, _ = kernel.pallas_call(shapes, alias_states=True)
    vocab_shard = vocab // mesh.devices.size

    def local(stack_w, vocab_w, consts, states, kv_len, read_row, tokens):
        sw = {k: v[0] for k, v in stack_w.items()}
        table = vocab_w["embed"][0]
        rank = jax.lax.axis_index(AXIS)
        x0 = _embed_rows(
            table,
            tokens[0],
            mode=embed_mode,
            vocab_shard=vocab_shard,
            rank=rank,
            flat=kernel.rows_mode,
        )
        ins = dict(
            sw,
            tables=consts["tables"][0],
            pages=consts["pages"][0],
            x0=x0,
            meta=_verify_meta(kv_len[0], read_row[0], kernel.verify_rows),
            **{k: v[0] for k, v in states.items()},
        )
        outs = call(*[ins[n] for n in names])
        st = dict(zip(state_names, outs[3:]))
        final, cap = outs[0], outs[1]
        final = final.reshape(final.shape[0], final.shape[-1])
        cap = cap.reshape(cap.shape[0], cap.shape[1], cap.shape[-1])
        return (
            {k: v[None] for k, v in st.items()},
            final[None],
            cap[None],
            outs[2][None],
        )

    ps = P(AXIS)
    sharded = jax.shard_map(
        local, mesh=mesh, in_specs=ps, out_specs=ps, check_vma=False
    )
    return jax.jit(sharded, donate_argnums=(3,), out_shardings=NamedSharding(mesh, ps))


def _first(a):
    import numpy as np

    return np.asarray(a.addressable_shards[0].data)[0].astype(np.float32)


def _rel(a, b) -> float:
    import numpy as np

    d = float(np.linalg.norm(a - b))
    return d / max(float(np.linalg.norm(b)), 1e-30)


def run_check(bench) -> dict:
    """One verify call vs R forced B1 steps from the same post-prompt state (rank 0's view):
    relative / max-abs differences of the final rows, captured rows, KDA checkpoints and MLA cache."""
    import numpy as np

    rows = bench.vkernel.verify_rows
    states = bench.fresh_states()
    forced = bench._rows(bench.prompt, np.int32)
    tok, kv = bench._rows(0, np.int32), bench._rows(1, np.int32)
    states, tok, kv, _ = bench.prefill(
        bench.stack_w, bench.vocab_w, bench.consts, states, tok, kv, forced
    )
    copy = lambda st: jax.tree.map(lambda a: a.copy(), st)
    _, _, _, cont = bench.decode7(
        bench.stack_w,
        bench.vocab_w,
        bench.consts,
        copy(states),
        tok,
        kv,
        bench._rows([0], np.int32),
    )
    t0 = int(np.asarray(tok.addressable_shards[0].data)[0])
    toks = [t0] + [
        int(x) for x in np.asarray(cont.addressable_shards[0].data)[0][: rows - 1]
    ]
    toks_d = bench._rows(toks, np.int32)
    o_states, o_final, o_cap, o_kda, o_ids = bench.oracle(
        bench.stack_w, bench.vocab_w, bench.consts, copy(states), kv, toks_d
    )
    v_states, v_final, v_cap, v_ids = bench.vonce(
        bench.stack_w,
        bench.vocab_w,
        bench.consts,
        states,
        kv,
        bench._rows(0, np.int32),
        toks_d,
    )
    res = {"tokens": toks}
    oi, vi = (
        np.asarray(a.addressable_shards[0].data)[0] for a in (o_ids, v_ids)
    )  # [L, R * top_k]
    k = oi.shape[1] // rows
    res["routing_differs"] = [
        [l, t]
        for l in range(oi.shape[0])
        for t in range(rows)
        if set(oi[l, t * k : (t + 1) * k]) != set(vi[l, t * k : (t + 1) * k])
    ]
    of, vf = _first(o_final), _first(v_final)
    res["final_rel"] = [_rel(vf[t], of[t]) for t in range(rows)]
    res["final_bitexact"] = bool(np.array_equal(vf, of))
    oc, vc = _first(o_cap), _first(v_cap)
    res["capture_rel"] = [
        [_rel(vc[c, t], oc[c, t]) for t in range(rows)] for c in range(oc.shape[0])
    ]
    res["capture_bitexact"] = bool(np.array_equal(vc, oc))
    kda_names = [n for n in state_inputs(bench.spec) if n.startswith("kda_")]
    worst = 0.0
    exact = True
    for i, n in enumerate(kda_names):
        vs, os_ = _first(v_states[n]), _first(o_kda[i])  # [R, ...] each
        worst = max(worst, max(_rel(vs[t], os_[t]) for t in range(rows)))
        exact &= bool(np.array_equal(vs, os_))
    res["kda_checkpoint_rel_max"], res["kda_bitexact"] = worst, exact
    mla = [n for n in state_inputs(bench.spec) if n.startswith("mla_")]
    res["mla_bitexact"] = all(
        bool(np.array_equal(_first(v_states[n]), _first(o_states[n]))) for n in mla
    )
    res["mla_rel_max"] = max(
        [_rel(_first(v_states[n]), _first(o_states[n])) for n in mla] or [0.0]
    )
    return res
