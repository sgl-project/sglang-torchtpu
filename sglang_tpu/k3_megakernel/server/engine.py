# SPDX-License-Identifier: Apache-2.0
"""Standalone K3 megakernel generation engine (pure JAX, no sglang scheduler).

Weights are loaded exactly as the anchor harness does (``server.loader.load_weights``, same
flags). One request at a time: the prompt is prefilled in 8-token rows-mode blocks
(``StackKernel(verify_rows=8, verify_mode="rows", prefill_blocks=1)``) inside one jitted
device loop over the chunks, then greedy decode runs ``K`` B1 steps per device call (the
fused LM head picks each token on device). Tokens of a call are returned to the caller
after the call; the over-generated tail past a stop token / max_tokens is dropped.

State layout: one request slot. KDA pools have 2 rows: row 0 is the null slot the rows
kernel sends its padding / intermediate checkpoint writes to, row 1 holds the request's
state (decode reads / writes row 1 too). MLA pages: an identity page table over
``capacity`` tokens. A new request resets the KDA state (meta reset flag of the first
chunk) and overwrites the MLA cache positions it uses (attention masks by kv length).

Speculative decoding (``Engine(draft=...)``, ``generate_spec``): each device call runs ``rounds``
DSpark rounds on device: the draft proposes ``NDRAFT`` tokens after the anchor, one 8-row verify
pass (``StackKernel(verify_rows=8)``, which also emits the draft's captured layers) checks them,
the longest prefix equal to the target argmax plus the bonus token is emitted and the committed
rows' captures go into the draft context. KDA pools then hold 1 + 2 * 8 rows: two banks of 8
checkpoint rows (a verify reads its state from one bank and writes its row checkpoints to the
other; the next round reads the checkpoint of its last accepted row).

The draft protocol (``server.draft.DraftModel``; the device functions run inside the engine's
shard_map on per-rank views, collectives over ``mesh.AXIS``, static shapes):
- ``init_state(mesh, capacity)`` (host): the draft state, a pytree of ``[world, ...]`` arrays
  sharded ``P(AXIS)``; its weights, if any, under key ``"w"``;
- ``reset(ds) -> ds`` (optional): run at the start of every request's prefill;
- ``append_context(ds, captures [8, C, H] bf16, positions [8] int32, n_valid) -> ds``: the
  captured target rows of the tokens at ``positions`` (only the first ``n_valid`` rows are real);
- ``propose(ds, anchor, pos, embed, lm_head) -> [NDRAFT] int32``: drafts for positions
  pos + 1 .. pos + NDRAFT (``pos``: the anchor's 0-based position);
- optional ``attach_lm_head(lm_head)`` (host) and ``capacity_multiple``.

Multi-host: every process runs the same engine calls in the same order (SPMD); process 0
decides what to run and the others follow (``server.app`` broadcasts the requests).
Every rank holds the same greedy tokens, so all processes see the same stop decisions.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sglang_tpu.k3_megakernel import model_io
from sglang_tpu.k3_megakernel.kernel.stack import (
    TOKEN_META,
    StackKernel,
    state_inputs,
)
from sglang_tpu.k3_megakernel.mesh import AXIS
from sglang_tpu.k3_megakernel.model_io import greedy as _greedy
from sglang_tpu.k3_megakernel.server import loader

BLOCK = 8  # rows per prefill block
VROWS = 8  # verify rows: anchor + NDRAFT drafts
NDRAFT = VROWS - 1  # sglang DSpark: speculative_num_draft_tokens 8
# Spec KDA pool: row 0 null, rows 1..8 and 9..16 two checkpoint banks (a verify reads its state from
# one bank and writes its 8 row checkpoints to the other). The prefill leaves the state in row ROW = 1.
KDA_ROWS_SPEC = 1 + 2 * VROWS
ROW = 1  # KDA pool row of the request (row 0 = null slot)
HIDDEN = 7168


def capture_layers(num_layers: int) -> tuple[int, ...]:
    """DSpark's target layers [7, 23, 51, 67, 83]; on short stacks the 13-layer remap [0, 3, 6, 9, 12]."""
    full = (7, 23, 51, 67, 83)
    if num_layers > full[-1]:
        return full
    return tuple(l for l in (0, 3, 6, 9, 12) if l < num_layers)


def greedy_rows(final, head, *, vocab_shard, rank):
    """Greedy ids [R] of ``final`` [R, H] (vocab-sharded head; ties to the lowest id)."""
    logits = jax.lax.dot_general(
        final, head, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
    )  # [R, V/world]
    local_id = jnp.argmax(logits, axis=1).astype(jnp.int32)
    best = jnp.take_along_axis(logits, local_id[:, None], axis=1)[:, 0]
    packed = jnp.stack(
        [best, (local_id + rank * vocab_shard).astype(jnp.float32)], axis=1
    )
    every = jax.lax.all_gather(packed, AXIS)  # [world, R, 2]
    win = jnp.argmax(every[:, :, 0], axis=0)
    return jnp.take_along_axis(every[:, :, 1], win[None], axis=0)[0].astype(jnp.int32)


def _log(msg):
    print(
        f"[k3srv p{jax.process_index()} {time.strftime('%H:%M:%S')}] {msg}", flush=True
    )


def _without_w(ds):
    """The draft state minus its weights ("w"): the weights are loop-invariant, so they stay out of the round
    loop's carry and the conds (a cond's outputs are fresh buffers: carrying them costs copies)."""
    return {k: v for k, v in ds.items() if k != "w"}


def _with_w(ds, w):
    return ds if w is None else dict(ds, w=w)


class Engine:
    def __init__(
        self,
        args,
        mesh,
        *,
        decode_steps: int = 32,
        capacity: int = 16384,
        lookahead: int = 1,
        weights=None,
        draft=None,
        spec_rounds: int = 8,
        verify_mode: str = "rows",
    ):
        """``draft``: DSpark draft (see the module docstring) for ``generate_spec``; ``spec_rounds``: its
        default rounds per device call. ``verify_mode="loop"``: the verify kernel runs one stack pass per
        row (bit-identical to B1 decode; parity checks only)."""
        self.args, self.mesh = args, mesh
        self.world = mesh.devices.size
        if draft is not None:
            mult = getattr(draft, "capacity_multiple", 1)
            if capacity % mult:
                capacity += mult - capacity % mult
                _log(
                    f"capacity rounded up to {capacity} (multiple of the draft's {mult})"
                )
        self.K, self.capacity, self.lookahead = decode_steps, capacity, lookahead
        self.spec = loader.stack_spec(args, mesh)
        kw = loader.kernel_kwargs(args)
        from sglang_tpu.k3_megakernel.ops import SERVING_KERNEL_OPTIONS

        refused = sorted(set(kw) - SERVING_KERNEL_OPTIONS - {"fused_head"})
        if refused:
            raise ValueError(
                f"--kernel-opt {refused}: benchmark-only, runtime-set or unsupported in serving"
            )
        self.kernel = StackKernel(self.spec, **kw)
        pkw = {
            k: v
            for k, v in kw.items()
            if k not in ("debug_repeat", "debug_skip", "layer_out")
        }
        # DSpark: the prefill and verify kernels also emit the draft's captured layers.
        self.draft = draft
        self.caps = capture_layers(self.spec.num_layers) if draft is not None else ()
        cap = dict(capture_layers=self.caps, capture_mixture=True) if self.caps else {}
        self.pkernel = StackKernel(
            self.spec,
            **pkw,
            **cap,
            verify_rows=BLOCK,
            verify_mode="rows",
            prefill_blocks=1,
        )
        if draft is not None:
            self.vkernel = StackKernel(
                self.spec, **pkw, **cap, verify_rows=VROWS, verify_mode=verify_mode
            )
        assert self.pkernel.rows_mode and self.pkernel.prefill_blocks == 1
        _log(
            f"spec {self.spec}; kernel options {kw}; fused_head {self.kernel.fused_head}; K {self.K}; "
            f"capacity {capacity}"
        )
        self.stack_w, self.vocab_w, self.load_s = weights or loader.load_weights(
            args, mesh
        )
        self.vocab = self.vocab_w["lm_head"].shape[1] * self.world
        self.consts = {
            "tables": model_io.put_replicated(
                mesh, np.asarray(self.spec.tables(), np.int32)
            ),
            "pages": model_io.put_replicated(mesh, model_io.page_table(capacity)),
        }
        st_shapes = model_io.state_shapes(
            self.kernel, self.spec, capacity, kda_rows=KDA_ROWS_SPEC if draft else 2
        )
        self._fresh = model_io.make_fresh_states(mesh, st_shapes)
        self.states = self._fresh()
        shapes = {n: (tuple(a.shape[1:]), a.dtype) for n, a in self.stack_w.items()}
        shapes.update(st_shapes)
        shapes.update(
            {n: (tuple(a.shape[1:]), a.dtype) for n, a in self.vocab_w.items()}
        )
        shapes.update(
            tables=((4, args.layers), np.int32),
            pages=(tuple(self.consts["pages"].shape[1:]), np.int32),
        )
        self.shapes = shapes
        self.max_prompt = capacity - 2 * self.K
        self._decode = {self.K: self._build_decode()}
        self._prefill = self._build_prefill()
        self._prefill_loop = (
            None  # built on first use (debug / parity with the harness)
        )
        if draft is not None:
            # DraftModel keeps its own transposed lm_head copy.
            if hasattr(draft, "attach_lm_head"):
                draft.attach_lm_head(self.vocab_w["lm_head"])
            self.dstate = draft.init_state(mesh, capacity)
            self._spec_fns = {}  # rounds per call -> jitted spec loop
        self.spec_rounds = spec_rounds
        self.kv = None  # kv length of the next token to feed (host int)
        self.tok = None  # device [world] next token
        self.kv_dev = None  # device [world] kv length

    # ----- device programs -----
    def _build_decode(self, forced: bool = False, steps: int = 0):
        """``forced=False``: ``f(..., states, token, kv_len)`` -> (states, token, kv_len, tokens [world, K]), K greedy
        steps. ``forced=True`` (token-loop prefill, B1 steps as the harness): ``f(..., states, tokens [world, cap],
        n [world])`` -> (states, next token); KDA row 1 is zeroed first (a fresh request)."""
        kernel, spec, K = self.kernel, self.spec, steps or self.K
        names = kernel.input_names()
        state_names = state_inputs(spec)
        fused, nf = kernel.fused_head, kernel.n_fixed_outputs
        call, _ = kernel.pallas_call(
            dict(
                self.shapes,
                meta=((TOKEN_META + 1,), np.int32),
                x0=((1, HIDDEN), jnp.bfloat16),
            ),
            alias_states=True,
        )
        vshard = self.vocab // self.world
        row = jnp.int32(ROW)

        def one(sw, consts, table, head, rank, st, tok, kv):
            ins = dict(sw, tables=consts["tables"][0], pages=consts["pages"][0], **st)
            if fused:
                ins.update(
                    embed=table,
                    lm_head=head,
                    meta=jnp.stack(
                        [kv, row, row] + [jnp.int32(0)] * (TOKEN_META - 3) + [tok]
                    ),
                )
            else:
                ins.update(
                    x0=jax.lax.dynamic_slice_in_dim(table, tok, 1, axis=0),
                    meta=jnp.stack([kv, row]),
                )
            outs = call(*[ins[n] for n in names])
            nxt = (
                outs[3][0]
                if fused
                else _greedy(outs[0], head, vocab_shard=vshard, rank=rank)
            )
            return dict(zip(state_names, outs[nf:])), nxt

        def local(stack_w, vocab_w, consts, states, token, kv_len):
            sw = {k: v[0] for k, v in stack_w.items()}
            table, head = vocab_w["embed"][0], vocab_w["lm_head"][0]
            rank = jax.lax.axis_index(AXIS)
            st = {k: v[0] for k, v in states.items()}
            if forced:  # token = the prompt [cap], kv_len = its length n
                toks, n = token[0], kv_len[0]
                st = {
                    k: (v.at[ROW].set(0) if k.startswith("kda_") else v)
                    for k, v in st.items()
                }

                def fstep(i, carry):
                    st, _ = carry
                    return one(sw, consts, table, head, rank, st, toks[i], i + 1)

                st, tok = jax.lax.fori_loop(0, n, fstep, (st, jnp.int32(0)))
                return {k: v[None] for k, v in st.items()}, tok[None]

            def step(i, carry):
                st, tok, kv, out = carry
                st, nxt = one(sw, consts, table, head, rank, st, tok, kv)
                return st, nxt, kv + 1, out.at[i].set(nxt)

            st, tok, kv, out = jax.lax.fori_loop(
                0, K, step, (st, token[0], kv_len[0], jnp.zeros((K,), jnp.int32))
            )
            return {k: v[None] for k, v in st.items()}, tok[None], kv[None], out[None]

        ps = P(AXIS)
        f = jax.shard_map(
            local, mesh=self.mesh, in_specs=ps, out_specs=ps, check_vma=False
        )
        return jax.jit(
            f, donate_argnums=(3,), out_shardings=NamedSharding(self.mesh, ps)
        )

    def _build_prefill(self):
        """``f(stack_w, vocab_w, consts, states, tokens [world, cap], n [world], kv0 [world][, dstate])``: feeds
        tokens[:n] at kv lengths kv0, kv0 + 1, ... in 8-token blocks; returns (states, next token[, dstate]).
        With a draft, every block's captured rows go to ``draft.append_context`` (positions kv - 1)."""
        kernel, spec = self.pkernel, self.spec
        names = kernel.input_names()
        state_names = state_inputs(spec)
        nf = kernel.n_fixed_outputs
        call, _ = kernel.pallas_call(
            dict(
                self.shapes,
                meta=((6,), np.int32),
                x0=((1, BLOCK, HIDDEN), jnp.bfloat16),
            ),
            alias_states=True,
        )
        vshard = self.vocab // self.world
        row = jnp.int32(ROW)
        draft = self.draft

        def local(stack_w, vocab_w, consts, states, tokens, n_tok, kv0, *dstate):
            sw = {k: v[0] for k, v in stack_w.items()}
            table, head = vocab_w["embed"][0], vocab_w["lm_head"][0]
            rank = jax.lax.axis_index(AXIS)
            toks, n, k0 = tokens[0], n_tok[0], kv0[0]

            def chunk(c, carry):
                st, _, ds = carry
                ids = jax.lax.dynamic_slice_in_dim(toks, c * BLOCK, BLOCK)
                x0 = jnp.take(table, ids, axis=0)[None]  # [1, 8, H]
                m = jnp.minimum(n - c * BLOCK, BLOCK)
                kv = k0 + c * BLOCK
                reset = ((c == 0) & (k0 == 1)).astype(jnp.int32)
                meta = jnp.stack([kv, row, row, m, jnp.int32(0), reset])
                ins = dict(
                    sw,
                    tables=consts["tables"][0],
                    pages=consts["pages"][0],
                    x0=x0,
                    meta=meta,
                    **st,
                )
                outs = call(*[ins[nm] for nm in names])
                if draft is not None:  # captures [C, 1, 8, H] -> [8, C, H]
                    caps = jnp.transpose(outs[1][:, 0], (1, 0, 2))
                    pos = kv - 1 + jnp.arange(BLOCK, dtype=jnp.int32)
                    ds = _without_w(draft.append_context(_with_w(ds, dw), caps, pos, m))
                return dict(zip(state_names, outs[nf:])), outs[0][0], ds

            st = {k: v[0] for k, v in states.items()}
            # A fresh request also zeroes its KDA row here, on top of the kernel's reset flag (meta[5]).
            fresh = k0 == 1
            st = {
                k: (
                    v.at[ROW].set(jnp.where(fresh, jnp.zeros_like(v[ROW]), v[ROW]))
                    if k.startswith("kda_")
                    else v
                )
                for k, v in st.items()
            }
            ds = dw = None
            if draft is not None:
                ds = jax.tree.map(lambda v: v[0], dstate[0])
                if hasattr(draft, "reset"):
                    ds = draft.reset(ds)
                ds, dw = _without_w(ds), ds.get("w")
            nchunks = (n + BLOCK - 1) // BLOCK
            st, final, ds = jax.lax.fori_loop(
                0, nchunks, chunk, (st, jnp.zeros((BLOCK, HIDDEN), jnp.bfloat16), ds)
            )
            last = jax.lax.dynamic_slice_in_dim(final, (n - 1) % BLOCK, 1, axis=0)
            tok = _greedy(last, head, vocab_shard=vshard, rank=rank)
            out = ({k: v[None] for k, v in st.items()}, tok[None])
            if draft is not None:
                out += (jax.tree.map(lambda v: v[None], _with_w(ds, dw)),)
            return out

        ps = P(AXIS)
        f = jax.shard_map(
            local, mesh=self.mesh, in_specs=ps, out_specs=ps, check_vma=False
        )
        donate = (3, 7) if draft is not None else (3,)
        return jax.jit(
            f, donate_argnums=donate, out_shardings=NamedSharding(self.mesh, ps)
        )

    def _build_spec(self, rounds: int):
        """``f(stack_w, vocab_w, consts, states, dstate, token, kv_len, read_row, hooks)`` -> (states, dstate, token,
        kv_len, read_row, out [world, rounds, VROWS + 1], greedy [world, rounds, VROWS]): ``rounds`` speculative
        rounds on device.
        A round: drafts = propose(anchor), verify [anchor, drafts, pad] (8 rows, KDA checkpoints to the other bank),
        accept a = longest prefix of drafts equal to the target argmax, emit drafts[:a] + argmax[a]; the captures of
        the a + 1 committed rows go to the draft, the next round reads checkpoint row a. ``out[r]`` = [8 emitted ids
        (first n valid), n] ([VROWS + 1]). ``hooks`` [world, 2] (benchmarks; zeros in serving): [forced tokens per
        round (0 = off), 1 = zero drafts without running the draft]."""
        kernel, spec, draft = self.vkernel, self.spec, self.draft
        names = kernel.input_names()
        state_names = state_inputs(spec)
        nf = kernel.n_fixed_outputs
        x0_shape = (VROWS, HIDDEN) if kernel.rows_mode else (VROWS, 1, HIDDEN)
        call, _ = kernel.pallas_call(
            dict(self.shapes, meta=((6,), np.int32), x0=(x0_shape, jnp.bfloat16)),
            alias_states=True,
        )
        vshard = self.vocab // self.world
        idx = jnp.arange(VROWS, dtype=jnp.int32)

        def local(
            stack_w, vocab_w, consts, states, dstate, token, kv_len, read_row, hooks
        ):
            force, zero_drafts = hooks[0, 0], hooks[0, 1] != 0
            sw = {k: v[0] for k, v in stack_w.items()}
            table, head = vocab_w["embed"][0], vocab_w["lm_head"][0]
            rank = jax.lax.axis_index(AXIS)

            def rnd(r, carry):
                st, ds, tok, kv, rr, out, gout = carry
                drafts = jax.lax.cond(
                    zero_drafts,  # verify + loop cost without the draft
                    lambda: jnp.zeros((NDRAFT,), jnp.int32),
                    lambda: draft.propose(
                        _with_w(ds, dw), tok, kv - 1, embed=table, lm_head=head
                    ).astype(jnp.int32),
                )  # [NDRAFT]
                toks = jnp.concatenate([tok[None], drafts])
                base = jnp.where(rr <= VROWS, jnp.int32(1 + VROWS), jnp.int32(1))
                meta = jnp.stack(
                    [kv, rr, base, jnp.int32(VROWS), jnp.int32(1), jnp.int32(0)]
                )
                ins = dict(
                    sw,
                    tables=consts["tables"][0],
                    pages=consts["pages"][0],
                    x0=jnp.take(table, toks, axis=0).reshape(x0_shape),
                    meta=meta,
                    **st,
                )
                outs = call(*[ins[nm] for nm in names])
                st = dict(zip(state_names, outs[nf:]))
                hid = outs[0].reshape(VROWS, HIDDEN)
                g = greedy_rows(hid, head, vocab_shard=vshard, rank=rank)
                a = jnp.sum(jnp.cumprod((drafts == g[:NDRAFT]).astype(jnp.int32)))
                # PR #8 timing protocol (force > 0): keep force - 1 drafts whatever the target says
                a = jnp.where(force > 0, jnp.clip(force - 1, 0, NDRAFT), a)
                bonus = g[a]
                emit = jnp.where(idx < a, jnp.roll(toks, -1), bonus)
                emit = jnp.concatenate([emit, (a + 1)[None]])  # [VROWS + 1]
                caps = jnp.transpose(
                    outs[1].reshape(len(self.caps), VROWS, HIDDEN), (1, 0, 2)
                )
                ds = _without_w(
                    draft.append_context(_with_w(ds, dw), caps, kv - 1 + idx, a + 1)
                )
                return (
                    st,
                    ds,
                    bonus,
                    kv + a + 1,
                    base + a,
                    out.at[r].set(emit),
                    gout.at[r].set(g),
                )

            st = {k: v[0] for k, v in states.items()}
            ds = jax.tree.map(lambda v: v[0], dstate)
            ds, dw = _without_w(ds), ds.get("w")
            z = jnp.zeros((rounds, VROWS), jnp.int32)
            z1 = jnp.zeros((rounds, VROWS + 1), jnp.int32)
            carry = (st, ds, token[0], kv_len[0], read_row[0], z1, z)
            st, ds, tok, kv, rr, out, gout = jax.lax.fori_loop(0, rounds, rnd, carry)
            return (
                {k: v[None] for k, v in st.items()},
                jax.tree.map(lambda v: v[None], _with_w(ds, dw)),
                tok[None],
                kv[None],
                rr[None],
                out[None],
                gout[None],
            )

        ps = P(AXIS)
        f = jax.shard_map(
            local, mesh=self.mesh, in_specs=ps, out_specs=ps, check_vma=False
        )
        return jax.jit(
            f, donate_argnums=(3, 4), out_shardings=NamedSharding(self.mesh, ps)
        )

    # ----- host side -----
    def reset_states(self):
        """Zero KDA / MLA pools (debug: rules out leftovers of earlier requests)."""
        self.states = self._fresh()

    def _rows(self, values, dtype=np.int32):
        return model_io.put_replicated(self.mesh, np.asarray(values, dtype))

    @staticmethod
    def _host(a):
        return np.asarray(a.addressable_shards[0].data)[0]

    def prefill(self, prompt: list[int], mode: str = "rows") -> int:
        """Feeds the prompt from position 0; returns the first generated token (device [world]). ``mode="loop"``: one B1 step
        per prompt token (bit-identical to the harness's forced prefill)."""
        n = len(prompt)
        if not 0 < n <= self.max_prompt:
            raise ValueError(f"prompt length {n} not in [1, {self.max_prompt}]")
        buf = np.zeros(self.capacity, np.int32)
        buf[:n] = prompt
        if mode == "loop":
            if self._prefill_loop is None:
                self._prefill_loop = self._build_decode(forced=True)
            self.states, self.tok = self._prefill_loop(
                self.stack_w,
                self.vocab_w,
                self.consts,
                self.states,
                self._rows(buf),
                self._rows(n),
            )
        else:
            args = (
                self.stack_w,
                self.vocab_w,
                self.consts,
                self.states,
                self._rows(buf),
                self._rows(n),
                self._rows(1),
            )
            if self.draft is not None:
                self.states, self.tok, self.dstate = self._prefill(*args, self.dstate)
            else:
                self.states, self.tok = self._prefill(*args)
        self.kv, self.kv_dev = n + 1, self._rows(n + 1)
        self.rr_dev = self._rows(ROW)
        return self.tok

    def _dispatch(self, steps: int):
        """Enqueues ``steps`` greedy steps after the last fed token; returns their tokens (device [world, steps])
        without waiting. kv length and the token stay on device, so back-to-back calls need no host round trip."""
        if self.kv + steps > self.capacity:
            raise ValueError("capacity exhausted")
        if steps not in self._decode:
            self._decode[steps] = self._build_decode(steps=steps)
        self.states, self.tok, self.kv_dev, out = self._decode[steps](
            self.stack_w, self.vocab_w, self.consts, self.states, self.tok, self.kv_dev
        )
        self.kv += steps
        return out

    def _dispatch_spec(self, rounds: int, hooks: tuple[int, int]):
        """Enqueues ``rounds`` speculative rounds; returns (out [world, rounds, 9], greedy [world, rounds, 8]) on
        device without waiting."""
        if self.kv + rounds * VROWS + VROWS > self.capacity:
            raise ValueError("capacity exhausted")
        if rounds not in self._spec_fns:
            self._spec_fns[rounds] = self._build_spec(rounds)
        (
            self.states,
            self.dstate,
            self.tok,
            self.kv_dev,
            self.rr_dev,
            out,
            g,
        ) = self._spec_fns[rounds](
            self.stack_w,
            self.vocab_w,
            self.consts,
            self.states,
            self.dstate,
            self.tok,
            self.kv_dev,
            self.rr_dev,
            self._rows(list(hooks)),
        )
        self.kv += rounds * VROWS  # upper bound (host does not know the acceptance yet)
        return out, g

    def generate_spec(
        self,
        prompt: list[int],
        max_tokens: int,
        stop_ids=(),
        on_tokens=None,
        stop_check=None,
        rounds: int | None = None,
        lookahead: int | None = None,
        check_accept: bool = True,
        force: int = 0,
        zero_drafts: bool = False,
    ):
        """Speculative greedy generation (DSpark draft + 8-row verify). Each device call runs ``rounds`` rounds
        (default ``spec_rounds``; 1 to 8 tokens each); ``lookahead`` calls stay in flight. The host reads each
        call's tokens, stops at a stop id / max_tokens / ``stop_check`` and drops the overshoot (the next
        request's prefill resets all state). ``check_accept``: re-derives every round's acceptance on the host
        from the returned target argmax. Benchmark hooks (output meaningless): ``force`` > 0 emits ``force``
        tokens per round (``force`` - 1 drafts + bonus) whatever the target says (PR #8 timing protocol);
        ``zero_drafts`` verifies token 0 drafts without running the draft.
        Returns (ids, finish reason, timings dict with the accept length)."""
        assert self.draft is not None, "engine built without a draft"
        rounds = self.spec_rounds if rounds is None else rounds
        hooks = (int(force), int(bool(zero_drafts)))
        stop_ids = set(stop_ids)
        la = self.lookahead if lookahead is None else lookahead
        max_tokens = max(1, min(max_tokens, self.capacity - len(prompt) - 2))
        t0 = time.perf_counter()
        first_dev = self.prefill(prompt)
        self.kv = len(prompt) + 1

        def cap_left():
            return self.kv + rounds * VROWS + VROWS <= self.capacity

        inflight = [self._dispatch_spec(rounds, hooks) for _ in range(la) if cap_left()]
        pending = [int(self._host(first_dev))]
        t1 = time.perf_counter()
        ids, reason, calls = [], None, 0
        correct = []  # drafts accepted per round
        while True:
            new = []
            for t in pending:
                if t in stop_ids:
                    reason = "stop"
                    break
                new.append(t)
                if len(ids) + len(new) >= max_tokens:
                    reason = "length"
                    break
            ids += new
            if reason is None and stop_check is not None and stop_check(ids, len(new)):
                reason = "stop"
            if reason is None and not inflight:
                reason = "length"  # capacity
            if on_tokens is not None:
                on_tokens(new, reason is not None, reason)
            if reason is not None:
                break
            if cap_left():
                inflight.append(self._dispatch_spec(rounds, hooks))
            out, g = inflight.pop(0)
            out, g = self._host(out), self._host(g)
            pending = []
            for r in range(rounds):
                n = int(out[r, VROWS])
                if check_accept and not force:
                    drafts = out[r, : n - 1]
                    assert (drafts == g[r, : n - 1]).all() and int(
                        out[r, n - 1]
                    ) == int(g[r, n - 1]), (r, out[r], g[r])
                correct.append(n - 1)
                pending += [int(x) for x in out[r, :n]]
            calls += 1
        t2 = time.perf_counter()
        # Rounds whose tokens the request consumed (not the overshoot).
        used, got = [], 1
        for a in correct:
            if got >= len(ids):
                break
            used.append(a)
            got += a + 1
        timing = dict(
            prompt_tokens=len(prompt),
            completion_tokens=len(ids),
            prefill_s=t1 - t0,
            decode_s=t2 - t1,
            decode_calls=calls,
            rounds=rounds,
            rounds_run=len(correct),
            forced=force,
            zero_drafts=bool(zero_drafts),
            # tokens per consumed round, bonus included
            accept_length=1.0 + float(np.mean(used)) if used else 0.0,
            ms_per_round=1e3 * (t2 - t1) / max(1, len(correct)),
            tpot_ms=1e3 * (t2 - t1) / max(1, len(ids) - 1),
        )
        return ids, reason, timing

    def compiled_steps(self) -> list[int]:
        """Steps-per-call values a request may ask for (``k3_steps``): the variants warmup compiles."""
        return self._sizes(self.K)

    @staticmethod
    def _sizes(k: int) -> list[int]:
        """Compiled steps-per-call variants: K, K/4, K/16, ..., 1 (the tail of a request takes smaller ones)."""
        out = [k]
        while out[-1] > 1:
            out.append(max(1, out[-1] // 4))
        return out

    def generate(
        self,
        prompt: list[int],
        max_tokens: int,
        stop_ids=(),
        on_tokens=None,
        stop_check=None,
        prefill_mode: str = "rows",
        steps: int = 0,
    ):
        """Greedy generation; ``on_tokens(new_ids, finished, reason)`` after the prefill and every call.
        ``stop_check(ids, n_new) -> bool`` (deterministic: every process runs it; ``n_new`` = tokens added by
        the last call) ends the request after a call.
        Calls run K steps; near max_tokens smaller compiled variants (K/4, K/16, .., 1) fill the remainder, so
        a length-limited request runs no extra steps. ``lookahead`` calls stay in flight ahead of the host's
        read-back (the device never waits for the host); a stop token wastes at most the in-flight calls.
        Returns (ids, finish reason, timings dict). Stop token excluded from the ids."""
        sizes = self._sizes(steps or self.K)
        stop_ids = set(stop_ids)
        max_tokens = max(1, min(max_tokens, self.capacity - len(prompt) - 2))
        planned = [1]  # tokens produced or in flight (the prefill gives the first)

        def dispatch():
            left = max_tokens - planned[0]
            if left <= 0:
                return None
            n = next(x for x in sizes if x <= left)
            planned[0] += n
            return self._dispatch(n)

        t0 = time.perf_counter()
        first_dev = self.prefill(prompt, prefill_mode)
        inflight = [
            o for o in (dispatch() for _ in range(self.lookahead)) if o is not None
        ]
        pending = [int(self._host(first_dev))]
        t1 = time.perf_counter()
        ids, reason, calls = [], None, 0
        while True:
            new = []
            for t in pending:
                if t in stop_ids:
                    reason = "stop"
                    break
                new.append(t)
                if len(ids) + len(new) >= max_tokens:
                    reason = "length"
                    break
            ids += new
            if reason is None and stop_check is not None and stop_check(ids, len(new)):
                reason = "stop"
            if on_tokens is not None:
                on_tokens(new, reason is not None, reason)
            if reason is not None:
                break
            nxt = dispatch()
            if nxt is not None:
                inflight.append(nxt)
            pending = [int(t) for t in self._host(inflight.pop(0))]
            calls += 1
        t2 = time.perf_counter()
        timing = dict(
            prompt_tokens=len(prompt),
            completion_tokens=len(ids),
            prefill_s=t1 - t0,
            decode_s=t2 - t1,
            decode_calls=calls,
            steps=sizes[0],
            tpot_ms=1e3 * (t2 - t1) / max(1, len(ids) - 1),
        )
        return ids, reason, timing

    def warmup(self):
        t0 = time.perf_counter()
        self.generate(
            [1008, 18985, 387] * 3, 2 + sum(self._sizes(self.K)), ()
        )  # compiles every size
        if self.draft is not None:
            self.generate_spec([1008, 18985, 387] * 3, 8, ())
        _log(f"warmup (compile) {time.perf_counter() - t0:.1f}s")
