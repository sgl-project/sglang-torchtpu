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
ROW = 1  # KDA pool row of the request (row 0 = null slot)
HIDDEN = 7168


def _log(msg):
    print(
        f"[k3srv p{jax.process_index()} {time.strftime('%H:%M:%S')}] {msg}", flush=True
    )


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
    ):
        self.args, self.mesh = args, mesh
        self.world = mesh.devices.size
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
        self.pkernel = StackKernel(
            self.spec, **pkw, verify_rows=BLOCK, verify_mode="rows", prefill_blocks=1
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
        st_shapes = model_io.state_shapes(self.kernel, self.spec, capacity, kda_rows=2)
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
        """``f(stack_w, vocab_w, consts, states, tokens [world, cap], n [world], kv0 [world])``: feeds
        tokens[:n] at kv lengths kv0, kv0 + 1, ... in 8-token blocks; returns (states, next token)."""
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

        def local(stack_w, vocab_w, consts, states, tokens, n_tok, kv0):
            sw = {k: v[0] for k, v in stack_w.items()}
            table, head = vocab_w["embed"][0], vocab_w["lm_head"][0]
            rank = jax.lax.axis_index(AXIS)
            toks, n, k0 = tokens[0], n_tok[0], kv0[0]

            def chunk(c, carry):
                st, _ = carry
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
                return dict(zip(state_names, outs[nf:])), outs[0][0]

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
            nchunks = (n + BLOCK - 1) // BLOCK
            st, final = jax.lax.fori_loop(
                0, nchunks, chunk, (st, jnp.zeros((BLOCK, HIDDEN), jnp.bfloat16))
            )
            last = jax.lax.dynamic_slice_in_dim(final, (n - 1) % BLOCK, 1, axis=0)
            tok = _greedy(last, head, vocab_shard=vshard, rank=rank)
            return {k: v[None] for k, v in st.items()}, tok[None]

        ps = P(AXIS)
        f = jax.shard_map(
            local, mesh=self.mesh, in_specs=ps, out_specs=ps, check_vma=False
        )
        return jax.jit(
            f, donate_argnums=(3,), out_shardings=NamedSharding(self.mesh, ps)
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
            self.states, self.tok = self._prefill(
                self.stack_w,
                self.vocab_w,
                self.consts,
                self.states,
                self._rows(buf),
                self._rows(n),
                self._rows(1),
            )
        self.kv, self.kv_dev = n + 1, self._rows(n + 1)
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
        _log(f"warmup (compile) {time.perf_counter() - t0:.1f}s")
