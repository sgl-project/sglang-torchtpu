# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 KDA (Kimi Delta Attention) decode step as an in-kernel stage.

Follows sglang's TPU path (attention_kda_pallas.py -> gdn/wrapper.py,
compute_conv1d.py, compute_kda.py recurrent_kda_per_seq; spec in the
megakernel LOG): bf16 projections, f32 causal conv + silu over the raw
pre-conv window, l2-normalized q/k, bounded gate, and the delta-rule update
on an f32 state whose two contractions use bf16-rounded operands (Mosaic's
default matmul precision there). Heads are split over all ranks; the o_proj
partials are all-reduced in f32 and rounded once.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from sglang_tpu.k3_megakernel.config import (
    K3MK_COLLECTIVE_ID,
    Relax,
    Topology,
)
from sglang_tpu.k3_megakernel.kernel.collectives import entry_barrier
from sglang_tpu.k3_megakernel.kernel.stages.common import (
    DN,
    OutProjSpec,
    bf16,
    bind_stream,
    build_sharded,
    phase,
    stream_scratch,
)

HEAD_DIM = 128
GATE_LOWER_BOUND = -5.0


def _bf16_operand(x):
    return x.astype(jnp.bfloat16).astype(jnp.float32)


def _l2norm(x):
    return x * jax.lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + 1e-6)


def _head_matvec(x, s):
    """``x`` [h, 128] f32, ``s`` [h, 128, 128] f32 -> [h, 128]: per head x @ s on the MXU with
    bf16 operands and f32 accumulation (the same operands as the VPU form; another sum order).
    The row is broadcast to 8 sublanes to keep the MXU operand tile-shaped."""
    h = x.shape[0]
    xb = jnp.broadcast_to(x.astype(jnp.bfloat16)[:, None, :], (h, 8, HEAD_DIM))
    r = jax.lax.dot_general(
        xb,
        s.astype(jnp.bfloat16),
        (((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )
    return r[:, 0, :]


def kda_core(
    qkv,
    a,
    b_row,
    gate,
    conv_state,
    rec_state,
    conv_w,
    dt,
    a_log,
    norm_w,
    *,
    heads,
    eps,
    mxu=False,
    fast=False,
):
    """One token. ``qkv`` [1, 3hd] and ``a``/``gate`` [1, hd] bf16-valued f32,
    ``b_row`` [1, 128] (first ``heads`` lanes real), ``conv_state`` [3, 3hd],
    ``rec_state`` [h, 128, 128]. Returns (o_normed bf16-valued [1, hd],
    new conv state, new recurrent state)."""
    hd = heads * HEAD_DIM
    window = [conv_state[0:1], conv_state[1:2], conv_state[2:3], qkv]
    acc = window[0] * conv_w[0:1]
    for t in range(1, 4):
        acc = acc + window[t] * conv_w[t : t + 1]
    act = acc * jax.nn.sigmoid(acc)
    new_conv = jnp.concatenate(window[1:], axis=0)

    v = act[:, 2 * hd :].reshape(heads, HEAD_DIM)
    if fast:  # q and k normalized in one reduction
        qk = _l2norm(act[:, : 2 * hd].reshape(2 * heads, HEAD_DIM))
        q, k = qk[:heads] * (HEAD_DIM**-0.5), qk[heads:]
    else:
        q = _l2norm(act[:, :hd].reshape(heads, HEAD_DIM)) * (HEAD_DIM**-0.5)
        k = _l2norm(act[:, hd : 2 * hd].reshape(heads, HEAD_DIM))
    # beta per head as a column: pick lane i of the b row for head row i.
    lanes = jax.lax.broadcasted_iota(jnp.int32, (heads, 128), 1)
    rows = jax.lax.broadcasted_iota(jnp.int32, (heads, 128), 0)
    beta = jax.nn.sigmoid(
        jnp.sum(jnp.where(lanes == rows, b_row, 0.0), axis=1, keepdims=True)
    )

    a_h = a.reshape(heads, HEAD_DIM)
    log_decay = GATE_LOWER_BOUND * jax.nn.sigmoid(
        jnp.exp(a_log.reshape(heads, HEAD_DIM)) * (a_h + dt.reshape(heads, HEAD_DIM))
    )
    s = rec_state * jnp.exp(log_decay)[:, :, None]
    if fast:
        # o = q . (s + k v_new^T) = q . s + (q . k) v_new: both contractions read the decayed state,
        # so the output no longer waits for the state update (f32 / bf16-operand reordering).
        mv = (
            _head_matvec
            if mxu
            else (
                lambda x, st: jnp.sum(
                    _bf16_operand(x)[:, :, None] * _bf16_operand(st), axis=1
                )
            )
        )
        v_upd, q_s = mv(k, s), mv(q, s)
        v_new = beta * (v - v_upd)
        qk_dot = jnp.sum(_bf16_operand(q) * _bf16_operand(k), axis=1, keepdims=True)
        o = bf16(q_s + qk_dot * v_new)
        s = s + k[:, :, None] * v_new[:, None, :]
    elif mxu:
        v_upd = _head_matvec(k, s)
    else:
        v_upd = jnp.sum(_bf16_operand(k)[:, :, None] * _bf16_operand(s), axis=1)
    if not fast:
        v_new = beta * (v - v_upd)
        s = s + k[:, :, None] * v_new[:, None, :]
        if mxu:
            o = bf16(_head_matvec(q, s))
        else:
            o = bf16(jnp.sum(_bf16_operand(q)[:, :, None] * _bf16_operand(s), axis=1))

    g = gate.reshape(heads, HEAD_DIM)
    rstd = jax.lax.rsqrt(jnp.mean(o * o, axis=-1, keepdims=True) + eps)
    y = bf16(o * rstd * norm_w * jax.nn.sigmoid(g))
    return y.reshape(1, hd), new_conv, s


def _row_prep(qkv, a, b_rows, conv_state, conv_w, dt, a_log, *, heads):
    """The state-independent part of ``kda_core`` for R rows at once: causal conv over
    [3 old rows; R new rows], silu, l2-normalized q / k, beta and the decay factor.
    Returns per-row lists of q, k, v ([h, 128]), beta ([h, 1]), decay ([h, 128])."""
    rows, hd = qkv.shape[0], heads * HEAD_DIM
    ext = jnp.concatenate([conv_state, qkv], axis=0)  # [3 + R, 3hd]
    acc = ext[0:rows] * conv_w[0:1]
    for t in range(1, 4):
        acc = acc + ext[t : t + rows] * conv_w[t : t + 1]
    act = acc * jax.nn.sigmoid(acc)  # [R, 3hd]
    q = _l2norm(act[:, :hd].reshape(rows * heads, HEAD_DIM)) * (HEAD_DIM**-0.5)
    k = _l2norm(act[:, hd : 2 * hd].reshape(rows * heads, HEAD_DIM))
    v = act[:, 2 * hd :].reshape(rows * heads, HEAD_DIM)
    lanes = jax.lax.broadcasted_iota(jnp.int32, (rows * heads, 128), 1)
    head = jax.lax.broadcasted_iota(jnp.int32, (rows * heads, 128), 0) % heads
    b_rep = jnp.concatenate(
        [jnp.broadcast_to(b_rows[r : r + 1], (heads, 128)) for r in range(rows)], axis=0
    )
    beta = jax.nn.sigmoid(
        jnp.sum(jnp.where(lanes == head, b_rep, 0.0), axis=1, keepdims=True)
    )
    dt_r = jnp.tile(dt.reshape(heads, HEAD_DIM), (rows, 1))
    ea = jnp.tile(jnp.exp(a_log.reshape(heads, HEAD_DIM)), (rows, 1))
    decay = jnp.exp(
        GATE_LOWER_BOUND
        * jax.nn.sigmoid(ea * (a.reshape(rows * heads, HEAD_DIM) + dt_r))
    )
    cut = lambda x: [x[r * heads : (r + 1) * heads] for r in range(rows)]
    return cut(q), cut(k), cut(v), cut(beta), cut(decay), ext


def kda_core_rows(
    qkv,
    a,
    b_rows,
    gate,
    conv_state,
    rec_state,
    conv_w,
    dt,
    a_log,
    norm_w,
    *,
    heads,
    eps,
    on_row,
):
    """``kda_core`` for R rows: everything that does not touch the recurrent state is computed
    for all rows at once; only the state update runs row by row. ``on_row(t, conv_t, rec_t)``
    receives the state after row t. Returns o_normed bf16-valued [R, hd]."""
    rows, hd = qkv.shape[0], heads * HEAD_DIM
    q, k, v, beta, decay, ext = _row_prep(
        qkv, a, b_rows, conv_state, conv_w, dt, a_log, heads=heads
    )
    s = rec_state
    outs = []
    for t in range(rows):
        s = s * decay[t][:, :, None]
        v_upd = jnp.sum(_bf16_operand(k[t])[:, :, None] * _bf16_operand(s), axis=1)
        v_new = beta[t] * (v[t] - v_upd)
        s = s + k[t][:, :, None] * v_new[:, None, :]
        outs.append(jnp.sum(_bf16_operand(q[t])[:, :, None] * _bf16_operand(s), axis=1))
        on_row(t, ext[t + 1 : t + 4], s)
    o = bf16(jnp.concatenate(outs, axis=0))  # [R * h, 128]
    g = gate.reshape(rows * heads, HEAD_DIM)
    rstd = jax.lax.rsqrt(jnp.mean(o * o, axis=-1, keepdims=True) + eps)
    y = bf16(
        o * rstd * jnp.tile(norm_w, (rows * heads, 1))[:, :HEAD_DIM] * jax.nn.sigmoid(g)
    )
    return y.reshape(rows, hd)


class KdaStage:
    """KDA layer for one decode row on ``topo`` (heads = 96 / world per rank)."""

    def __init__(
        self,
        topo: Topology,
        *,
        hidden: int = 7168,
        heads: int = 96,
        eps: float = 1e-5,
        relax: Relax = Relax(),
        collective_id: int = K3MK_COLLECTIVE_ID,
        repeat: int = 1,
        rows: int = 1,
    ):
        self.topo, self.hidden, self.eps, self.collective_id, self.repeat = (
            topo,
            hidden,
            eps,
            collective_id,
            repeat,
        )
        self.rows = rows
        self.relax = relax
        self.heads = heads // topo.world
        self.hd = self.heads * HEAD_DIM
        self.out = OutProjSpec(topo, self.hd, hidden, relax, rows=rows)
        self.skip_core = (
            False  # benchmark-only: no recurrence, zero output, state untouched
        )
        self.mxu = False  # the recurrence's two contractions on the MXU (``kda_core(mxu=True)``)
        self.fast_core = False  # ``kda_core(fast=True)``: shorter dependent chain
        self.streams = ((hidden, 4 * self.hd + 256), self.out.weight_shape)

    def scratch(self):
        shapes = []
        for k, n in self.streams:
            shapes += stream_scratch(k, n, self.relax.dense_dtype)
        return shapes + self.out.scratch_shapes()

    def body(
        self,
        x,
        proj,
        fb_ref,
        oproj,
        conv_w,
        dt,
        a_log,
        norm_w,
        conv_ref,
        rec_ref,
        out_proj,
    ):
        """``x`` bf16 [1, H]; updates ``conv_ref``/``rec_ref`` in place; ``out_proj`` is the bound
        ``self.out``. Returns bf16-valued f32 [1, H]."""
        hd = self.hd
        y = proj.matmul(x, prefetched=False)
        qkvg = bf16(y[:, : 4 * hd])
        fa = y[:, 4 * hd : 4 * hd + 128].astype(jnp.bfloat16)
        b_row = bf16(y[:, 4 * hd + 128 : 4 * hd + 256])
        a = bf16(
            jax.lax.dot_general(fa, fb_ref[...], DN, preferred_element_type=jnp.float32)
        )
        if self.skip_core:
            o = jnp.zeros((1, hd), jnp.float32) + jnp.sum(a) * 0.0
        else:
            with phase("kda_core"):
                o, new_conv, new_rec = kda_core(
                    qkvg[:, : 3 * hd],
                    a,
                    b_row,
                    qkvg[:, 3 * hd :],
                    conv_ref[...],
                    rec_ref[...],
                    conv_w,
                    dt,
                    a_log,
                    norm_w,
                    heads=self.heads,
                    eps=self.eps,
                    mxu=self.mxu,
                    fast=self.fast_core,
                )
                conv_ref[...] = new_conv
                rec_ref[...] = new_rec
        with phase("attn_ar"):  # o_proj stream + its reduction
            return out_proj(o.astype(jnp.bfloat16), oproj)

    def body_rows(
        self,
        x,
        proj,
        fb_ref,
        oproj,
        conv_w,
        dt,
        a_log,
        norm_w,
        conv_ref,
        rec_ref,
        out_proj,
        checkpoint,
    ):
        """``body`` for ``x`` bf16 [R, H] (speculative verify): the projections and o_proj run on all
        rows at once, the recurrence row by row; ``checkpoint(t, conv_t, rec_t)`` receives the state
        after row t as values. ``conv_ref`` / ``rec_ref`` hold the final state on return."""
        hd = self.hd
        y = proj.matmul(x, prefetched=False)
        qkvg = bf16(y[:, : 4 * hd])
        fa = y[:, 4 * hd : 4 * hd + 128].astype(jnp.bfloat16)
        b_rows = bf16(y[:, 4 * hd + 128 : 4 * hd + 256])
        a = bf16(
            jax.lax.dot_general(fa, fb_ref[...], DN, preferred_element_type=jnp.float32)
        )

        def on_row(t, conv_t, rec_t):
            if (
                t == x.shape[0] - 1
            ):  # before the checkpoint: the caller may DMA out of these refs
                conv_ref[...] = conv_t
                rec_ref[...] = rec_t
            checkpoint(t, conv_t, rec_t)

        if (
            self.skip_core
        ):  # benchmark-only: checkpoints of the unchanged state keep the DMAs balanced
            conv0, rec0 = conv_ref[...], rec_ref[...]
            for t in range(x.shape[0]):
                checkpoint(t, conv0, rec0)
            return out_proj(
                jnp.zeros((x.shape[0], hd), jnp.bfloat16)
                + (jnp.sum(a) * 0).astype(jnp.bfloat16),
                oproj,
            )
        with phase("kda_core"):
            o = kda_core_rows(
                qkvg[:, : 3 * hd],
                a,
                b_rows,
                qkvg[:, 3 * hd :],
                conv_ref[...],
                rec_ref[...],
                conv_w,
                dt,
                a_log,
                norm_w,
                heads=self.heads,
                eps=self.eps,
                on_row=on_row,
            )
        with phase("attn_ar"):
            return out_proj(o.astype(jnp.bfloat16), oproj)

    def kernel(self, *refs):
        """Inputs: x, f_b, conv_w, dt, a_log, o_norm, conv, rec, proj, o_proj[, proj_s, o_proj_s];
        outputs: out, conv, rec; then scratch."""
        n_in = 12 if self.relax.dense_fp8 else 10
        (
            x_ref,
            fb_ref,
            conv_w_ref,
            dt_ref,
            alog_ref,
            norm_ref,
            conv_in,
            rec_in,
            proj_hbm,
            oproj_hbm,
        ) = refs[:10]
        scales = refs[10:n_in] if self.relax.dense_fp8 else (None, None)
        out_ref, conv_ref, rec_ref, conv_ck, rec_ck = refs[n_in : n_in + 5]
        scratch = refs[n_in + 5 :]
        me = jax.lax.axis_index("x")
        ns = len(scratch) - len(self.out.scratch_shapes())
        proj = bind_stream(proj_hbm, scratch[: ns // 2], scales[0])
        oproj = bind_stream(oproj_hbm, scratch[ns // 2 : ns], scales[1])
        ar = self.out.bind(me, scratch[ns:])
        ar.init()
        entry_barrier(me, self.topo)
        conv_ref[...] = conv_in[...]
        rec_ref[...] = rec_in[...]

        def checkpoint(t, conv_t=None, rec_t=None):
            conv_ck[t] = conv_ref[...] if conv_t is None else conv_t
            rec_ck[t] = rec_ref[...] if rec_t is None else rec_t

        def run():
            args = (
                proj,
                fb_ref,
                oproj,
                conv_w_ref[...],
                dt_ref[...],
                alog_ref[...],
                norm_ref[...],
                conv_ref,
                rec_ref,
                ar,
            )
            if self.rows > 1:
                out_ref[...] = self.body_rows(x_ref[...], *args, checkpoint).astype(
                    jnp.bfloat16
                )
                return
            # Rows one by one through the B1 body: the oracle of ``body_rows``.
            for t in range(x_ref.shape[0]):
                out_ref[t : t + 1] = self.body(x_ref[t : t + 1], *args).astype(
                    jnp.bfloat16
                )
                checkpoint(t)

        if self.repeat == 1:
            run()
        else:

            @pl.loop(0, self.repeat)
            def _(i):
                run()

    def build(self, mesh, n_rows: int = 1):
        """``n_rows`` > 1 with ``rows == 1``: the rows run one by one through ``body`` (verify oracle).
        Extra outputs: the state after every row, ``[n_rows, 3, 3hd]`` / ``[n_rows, h, 128, 128]``."""
        hd = self.hd
        n_rows = max(n_rows, self.rows)
        n_hbm = 4 if self.relax.dense_fp8 else 2
        return build_sharded(
            self.kernel,
            mesh=mesh,
            in_spaces=("vmem",) * 8 + ("hbm",) * n_hbm,
            out_shapes=(
                jax.ShapeDtypeStruct((n_rows, self.hidden), jnp.bfloat16),
                jax.ShapeDtypeStruct((3, 3 * hd), jnp.float32),
                jax.ShapeDtypeStruct((self.heads, HEAD_DIM, HEAD_DIM), jnp.float32),
                jax.ShapeDtypeStruct((n_rows, 3, 3 * hd), jnp.float32),
                jax.ShapeDtypeStruct(
                    (n_rows, self.heads, HEAD_DIM, HEAD_DIM), jnp.float32
                ),
            ),
            out_spaces=("vmem",) * 5,
            scratch=self.scratch(),
            collective_id=self.collective_id,
        )
